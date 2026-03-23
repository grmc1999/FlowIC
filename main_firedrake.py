import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import firedrake as fd

# Depending on your installation, these imports may vary slightly.
from firedrake.adjoint import Control, compute_gradient

torch.set_default_dtype(torch.float64)


# =========================================================
# 1. Utility: boundary enforcement
# =========================================================
def enforce_zero_dirichlet(u: torch.Tensor) -> torch.Tensor:
    """
    Enforce u[..., 0] = u[..., -1] = 0.
    Works for shape (N,) or (B, N).
    """
    out = u.clone()
    out[..., 0] = 0.0
    out[..., -1] = 0.0
    return out


# =========================================================
# 2. PyTorch generator: vector field + RK4 flow
# =========================================================
class SimpleVectorField(nn.Module):
    """
    Same idea as the JAX model:
    receives state x and scalar time t, returns dx/dt.
    """
    def __init__(self, n_points: int, hidden_dim: int = 256):
        super().__init__()
        self.n_points = n_points
        self.net = nn.Sequential(
            nn.Linear(n_points + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_points),
        )

    def forward(self, x: torch.Tensor, t) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)

        t_vec = torch.full(
            (x.shape[0], 1),
            fill_value=t.item(),
            dtype=x.dtype,
            device=x.device,
        )
        inp = torch.cat([x, t_vec], dim=-1)
        return self.net(inp)


def rk4_integrate_vector_field(model: nn.Module,
                               z0: torch.Tensor,
                               n_steps: int = 20) -> torch.Tensor:
    """
    Fixed-step RK4 integration from t=0 to t=1.
    z0 can be shape (N,) or (B, N).
    """
    h = 1.0 / n_steps
    y = z0
    t = 0.0

    for _ in range(n_steps):
        k1 = model(y, t)
        k2 = model(y + 0.5 * h * k1, t + 0.5 * h)
        k3 = model(y + 0.5 * h * k2, t + 0.5 * h)
        k4 = model(y + h * k3, t + h)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h

    return y


def generate_ic(model: nn.Module,
                batch_size: int,
                n_points: int,
                noise_scale: float = 0.5,
                rk_steps: int = 20,
                device: str = "cpu") -> torch.Tensor:
    """
    Noise -> continuous flow -> candidate initial condition.
    """
    z0 = noise_scale * torch.randn(batch_size, n_points, device=device)
    ic = rk4_integrate_vector_field(model, z0, n_steps=rk_steps)
    ic = enforce_zero_dirichlet(ic)
    return ic


# =========================================================
# 3. Firedrake heat equation (forward solver)
# =========================================================
class HeatEquation1D:
    """
    Variational implicit Euler solver for:
        u_t - alpha * u_xx = 0
    on [0, L] with homogeneous Dirichlet boundary conditions.
    """
    def __init__(self,
                 n_points: int = 64,
                 length: float = 1.0,
                 alpha: float = 0.05,
                 dt: float = 1e-3,
                 num_steps: int = 200,
                 degree: int = 1):
        self.n_points = n_points
        self.length = length
        self.num_steps = num_steps

        self.mesh = fd.IntervalMesh(n_points - 1, length)
        self.V = fd.FunctionSpace(self.mesh, "CG", degree)

        self.alpha = fd.Constant(alpha)
        self.dt = fd.Constant(dt)

        self.u_trial = fd.TrialFunction(self.V)
        self.v_test = fd.TestFunction(self.V)

        self.bc = fd.DirichletBC(self.V, fd.Constant(0.0), "on_boundary")

        # Implicit Euler:
        # (u^{n+1}, v) + dt * alpha * (grad u^{n+1}, grad v) = (u^n, v)
        self.a_form = (
            self.u_trial * self.v_test
            + self.dt * self.alpha * fd.dot(fd.grad(self.u_trial), fd.grad(self.v_test))
        ) * fd.dx

    def tensor_to_function(self, x: torch.Tensor, name: str = "state") -> fd.Function:
        """
        Serial helper:
        assumes CG1 nodal ordering matches the line-grid ordering.
        Good for this first 1D prototype.
        """
        x_cpu = x.detach().cpu().double().contiguous()
        f = fd.Function(self.V, name=name)
        f.dat.data[:] = x_cpu.numpy()
        f.dat.data[0] = 0.0
        f.dat.data[-1] = 0.0
        return f

    def function_to_tensor(self,
                           f: fd.Function,
                           device=None,
                           dtype=torch.float64) -> torch.Tensor:
        arr = f.dat.data_ro.copy()
        out = torch.from_numpy(arr).to(dtype=dtype)
        if device is not None:
            out = out.to(device)
        return out

    def solve_from_function(self, u0: fd.Function) -> fd.Function:
        u_n = fd.Function(self.V, name="u_n")
        u_n.assign(u0)

        u_np1 = fd.Function(self.V, name="u_np1")

        for _ in range(self.num_steps):
            L_form = (u_n * self.v_test) * fd.dx
            fd.solve(
                self.a_form == L_form,
                u_np1,
                bcs=self.bc,
                solver_parameters={
                    "ksp_type": "cg",
                    "pc_type": "sor",
                },
            )
            u_n.assign(u_np1)

        return u_np1.copy(deepcopy=True)


# =========================================================
# 4. PyTorch <-> Firedrake differentiable bridge
# =========================================================
class FiredrakeHeatSolveFunction(torch.autograd.Function):
    """
    Custom autograd bridge:
    forward: Torch tensor IC -> Firedrake solve -> Torch tensor final state
    backward: uses firedrake-adjoint to get d(output)/d(IC)^T * grad_output
    """
    @staticmethod
    def forward(ctx, ic_tensor: torch.Tensor, pde: HeatEquation1D):
        device = ic_tensor.device
        dtype = ic_tensor.dtype

        u0 = pde.tensor_to_function(ic_tensor, name="control_ic")
        uT = pde.solve_from_function(u0)

        ctx.pde = pde
        ctx.u0 = u0
        ctx.uT = uT
        ctx.device = device
        ctx.dtype = dtype

        y = pde.function_to_tensor(uT, device=device, dtype=dtype)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Computes vector-Jacobian product:
            grad_output^T * (duT/du0)
        by defining the scalar functional:
            J = <uT, grad_output>
        and differentiating J wrt the control u0.
        """
        pde = ctx.pde
        weight = pde.tensor_to_function(
            grad_output.detach(),
            name="adjoint_weight"
        )

        J = fd.assemble(fd.inner(ctx.uT, weight) * fd.dx)
        grad_u0 = compute_gradient(J, Control(ctx.u0))

        grad_tensor = pde.function_to_tensor(
            grad_u0,
            device=ctx.device,
            dtype=ctx.dtype,
        )
        return grad_tensor, None


class DifferentiableHeatSolver(nn.Module):
    def __init__(self, pde: HeatEquation1D):
        super().__init__()
        self.pde = pde

    def forward(self, ic: torch.Tensor) -> torch.Tensor:
        return FiredrakeHeatSolveFunction.apply(ic, self.pde)


if __name__ == "__main__":
device = "cpu"

N = 64
L = 1.0
alpha = 0.05
dt_physics = 0.001
steps_physics = 200

pde = HeatEquation1D(
    n_points=N,
    length=L,
    alpha=alpha,
    dt=dt_physics,
    num_steps=steps_physics,
)

solver = DifferentiableHeatSolver(pde).to(device)

x_grid = torch.linspace(0.0, L, N, device=device)

# Ground truth IC (same as your JAX code)
gt_ic = torch.exp(-100.0 * (x_grid - 0.3) ** 2) + 0.5 * torch.exp(-100.0 * (x_grid - 0.7) ** 2)
gt_ic = enforce_zero_dirichlet(gt_ic)

with torch.no_grad():
    gt_final = solver(gt_ic)

print("Objetivo: encontrar la condición inicial que generó el estado final observado.")


# =========================================================
# 6. Model + optimizer
# =========================================================
model = SimpleVectorField(n_points=N, hidden_dim=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 2000
batch_size = 8     # Firedrake is not batched internally; start small
rk_steps = 20

loss_history = []
best_ic = None
best_final = None

print("\nComenzando entrenamiento...")

for epoch in range(epochs):
    optimizer.zero_grad()

    # 1. Generate a batch of candidate ICs
    pred_ic = generate_ic(
        model=model,
        batch_size=batch_size,
        n_points=N,
        noise_scale=0.5,
        rk_steps=rk_steps,
        device=device,
    )

    # 2. Solve PDE for each candidate
    pred_final = torch.stack([solver(pred_ic[k]) for k in range(batch_size)], dim=0)

    # 3. Compare with observed final state
    loss = torch.mean(torch.abs(pred_final - gt_final.unsqueeze(0)))
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    # keep best sample in current batch for visualization
    with torch.no_grad():
        sample_errors = torch.mean(
            torch.abs(pred_final - gt_final.unsqueeze(0)),
            dim=1
        )
        idx = torch.argmin(sample_errors)
        best_ic = pred_ic[idx].detach().cpu()
        best_final = pred_final[idx].detach().cpu()

    if epoch % 100 == 0:
        print(f"Iteración {epoch}: Loss = {loss.item():.6e}")


# =========================================================
# 7. Visualization
# =========================================================
gt_ic_cpu = gt_ic.detach().cpu()
gt_final_cpu = gt_final.detach().cpu()
x_grid_cpu = x_grid.detach().cpu()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x_grid_cpu, gt_ic_cpu, "k--", linewidth=2, label="Real IC (Secreta)")
plt.plot(x_grid_cpu, best_ic, "r-", linewidth=2, label="Flow Generada")
plt.title("Condición Inicial (t=0)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(x_grid_cpu, gt_final_cpu, "k--", linewidth=2, label="Observación Real")
plt.plot(x_grid_cpu, best_final, "b-", linewidth=2, label="Simulación desde Flow")
plt.title(f"Estado Final (t={dt_physics * steps_physics:.2f})")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(loss_history)
plt.yscale("log")
plt.title("Convergencia del Error")
plt.xlabel("Iteraciones")
plt.ylabel("MAE Loss")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.tight_layout()
plt.savefig(f'exp_epochs_{i}_samples_{args.n_samples}_lr_{args.lr}_generative_noise_{args.gen_noise}.png')
plt.show()