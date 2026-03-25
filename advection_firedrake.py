from abc import ABC, abstractmethod
from SolverBase import BaseFiredrakeOperator
from models import SimpleVectorField,generate_ic,enforce_zero_dirichlet,rk4_integrate_vector_field
import argparse
import os
import torch
import torch.nn as nn
import firedrake as fd
import matplotlib.pyplot as plt

from tqdm import tqdm
from firedrake.ml.pytorch.fem_operator import fem_operator
from firedrake.adjoint import Control, ReducedFunctional

torch.set_default_dtype(torch.float64)


#def enforce_zero_dirichlet(u: torch.Tensor) -> torch.Tensor:
#    """
#    Enforce u[..., 0] = u[..., -1] = 0.
#    Works for shape (N,) or (B, N).
#    """
#    out = u.clone()
#    out[..., 0] = 0.0
#    out[..., -1] = 0.0
#    return out
#
#
#class SimpleVectorField(nn.Module):
#    """
#    Receives state x and scalar time t, returns dx/dt.
#    """
#    def __init__(self, n_points: int, hidden_dim: int = 256):
#        super().__init__()
#        self.n_points = n_points
#        self.net = nn.Sequential(
#            nn.Linear(n_points + 1, hidden_dim),
#            nn.GELU(),
#            nn.Linear(hidden_dim, hidden_dim),
#            nn.GELU(),
#            nn.Linear(hidden_dim, n_points),
#        )
#
#    def forward(self, x: torch.Tensor, t) -> torch.Tensor:
#        if x.ndim == 1:
#            x = x.unsqueeze(0)
#
#        if not torch.is_tensor(t):
#            t = torch.tensor(t, dtype=x.dtype, device=x.device)
#
#        t_vec = torch.full(
#            (x.shape[0], 1),
#            fill_value=t.item(),
#            dtype=x.dtype,
#            device=x.device,
#        )
#        inp = torch.cat([x, t_vec], dim=-1)
#        return self.net(inp)


#def rk4_integrate_vector_field(
#    model: nn.Module,
#    z0: torch.Tensor,
#    n_steps: int = 20
#) -> torch.Tensor:
#    """
#    Fixed-step RK4 integration from t=0 to t=1.
#    z0 can be shape (N,) or (B, N).
#    """
#    h = 1.0 / n_steps
#    y = z0
#    t = 0.0
#
#    for _ in range(n_steps):
#        k1 = model(y, t)
#        k2 = model(y + 0.5 * h * k1, t + 0.5 * h)
#        k3 = model(y + 0.5 * h * k2, t + 0.5 * h)
#        k4 = model(y + h * k3, t + h)
#        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
#        t += h
#
#    return y
#
#
#def generate_ic(
#    model: nn.Module,
#    batch_size: int,
#    n_points: int,
#    noise_scale: float = 0.5,
#    rk_steps: int = 20,
#    device: str = "cpu",
#) -> torch.Tensor:
#    """
#    Noise -> continuous flow -> candidate initial condition.
#    """
#    z0 = noise_scale * torch.randn(batch_size, n_points, device=device)
#    ic = rk4_integrate_vector_field(model, z0, n_steps=rk_steps)
#    ic = enforce_zero_dirichlet(ic)
#    return ic
#
"""
class BaseFiredrakeOperator(nn.Module, ABC):
    def __init__(self):
        super().__init__()

        self.mesh = self.build_mesh()
        self.V = self.build_function_space(self.mesh)

        self.setup_problem()

        self.control = self.build_control()
        self.rf = self.build_reduced_functional()
        self.F_torch = fem_operator(self.rf)

    @abstractmethod
    def build_mesh(self):
        pass

    def build_function_space(self, mesh):
        return fd.FunctionSpace(mesh, "CG", 1)

    @abstractmethod
    def setup_problem(self):
        pass

    def build_control(self):
        return fd.Function(self.V, name="control")

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @abstractmethod
    def solve_annotated(self, control: fd.Function) -> fd.Function:
        pass

    def get_dof_coordinates(self):
        #coords = self.V.tabulate_dof_coordinates()
        coords = fd.Function(fd.VectorFunctionSpace(solver.V.mesh(),"DG",0)).interpolate(fd.SpatialCoordinate(solver.V.mesh())).dat.data # [n_points]
        gdim = self.mesh.geometric_dimension()
        return coords.reshape((-1, gdim))[:, 0]
    
    def build_reduced_functional(self):
        fd.adjoint.continue_annotation()
        try:
            output = self.solve_annotated(self.control)
            rf = ReducedFunctional(output, Control(self.control))
        finally:
            fd.adjoint.stop_annotating()
        return rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_output = False

        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_output = True

        x = self.preprocess_input(x)
        y = self.F_torch(x)

        if squeeze_output and y.ndim == 2 and y.shape[0] == 1:
            y = y.squeeze(0)

        return y
"""

class LinearAdvection1DOperator(BaseFiredrakeOperator):
    """
    1D periodic linear advection:
        u_t + c u_x = 0

    Discretization:
    - Periodic interval mesh
    - DG(0) space
    - implicit Euler in time
    - upwind numerical flux on interior facets
    """
    def __init__(
        self,
        n_cells: int = 64,
        length: float = 1.0,
        velocity: float = 1.0,
        dt: float = 1e-3,
        num_steps: int = 200,
    ):
        self.n_cells = n_cells
        self.length = length
        self.velocity_value = velocity
        self.dt_value = dt
        self.num_steps = num_steps
        super().__init__()

    def build_mesh(self):
        return fd.PeriodicIntervalMesh(self.n_cells, self.length)

    def build_function_space(self, mesh):
        # DG0 keeps one DOF per cell, which makes plotting and generator sizing simple
        return fd.FunctionSpace(mesh, "DG", 0)

    def setup_problem(self):
        self.c = fd.Constant(self.velocity_value)
        self.dt = fd.Constant(self.dt_value)

        self.u_trial = fd.TrialFunction(self.V)
        self.v_test = fd.TestFunction(self.V)

        self.n = fd.FacetNormal(self.mesh)

        # Normal flux on the '+' side of each interior facet
        self.cn = self.c * self.n[0]("+") if callable(getattr(self.n[0], "__call__", None)) else self.c * self.n[0]
        self.flux_n = self.c * self.n[0]("+")  # scalar in 1D

        # Upwind state selected from the sign of c·n
        self.u_up = fd.conditional(
            fd.gt(self.flux_n, 0.0),
            self.u_trial("+"),
            self.u_trial("-"),
        )

        # DG implicit Euler form:
        # (u^{n+1}, v) + dt * < (c n)_+ * u_up, jump(v) > = (u^n, v)
        self.a_form = (
            self.u_trial * self.v_test * fd.dx
            + self.dt * self.flux_n * self.u_up * fd.jump(self.v_test) * fd.dS
        )

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        # periodic problem: no Dirichlet enforcement
        return x

    def solve_annotated(self, u0: fd.Function) -> fd.Function:
        u_n = fd.Function(self.V, name="u_n")
        u_n.assign(u0)

        u_np1 = fd.Function(self.V, name="u_np1")

        for _ in range(self.num_steps):
            L_form = u_n * self.v_test * fd.dx

            fd.solve(
                self.a_form == L_form,
                u_np1,
                solver_parameters={
                    "ksp_type": "gmres",
                    "pc_type": "bjacobi",
                    "sub_pc_type": "ilu",
                },
            )
            u_n.assign(u_np1)

        return u_n.copy(deepcopy=True)


def plot_1D(
    gt_ic_cpu,
    gt_final_cpu,
    x_grid_cpu,
    pred_ic,
    pred_final,
    loss_history,
    loss_ic,
    lr=1e-4,
    epoch=0,
    n_samples=8,
    dt_physics=1e-4,
    steps_physics=0,
    exp_dir = ""
):
    pred_ic_mean = torch.mean(pred_ic, dim=0).detach().cpu().numpy()
    pred_ic_std = torch.std(pred_ic, dim=0).detach().cpu().numpy()

    pred_final_mean = torch.mean(pred_final, dim=0).detach().cpu().numpy()
    pred_final_std = torch.std(pred_final, dim=0).detach().cpu().numpy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(x_grid_cpu, gt_ic_cpu, "k--", linewidth=2, label="Real IC")
    plt.plot(x_grid_cpu, pred_ic_mean, "r-", linewidth=2, label="Generated Flow")
    plt.fill_between(
        x_grid_cpu,
        pred_ic_mean + pred_ic_std,
        pred_ic_mean - pred_ic_std,
        color="r",
        alpha=0.3,
    )
    plt.title("Initial condition (t=0)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(x_grid_cpu, gt_final_cpu, "k--", linewidth=2, label="Ground Truth")
    plt.plot(x_grid_cpu, pred_final_mean, "r-", linewidth=2, label="Predicted Final")
    plt.fill_between(
        x_grid_cpu,
        pred_final_mean + pred_final_std,
        pred_final_mean - pred_final_std,
        color="r",
        alpha=0.3,
    )
    plt.title(f"Final state (t={dt_physics * steps_physics:.2f})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(loss_history, color="b", label = "final state error")
    plt.plot(loss_ic, color="r", label = "initial state error")
    plt.yscale("log")
    plt.title("Error convergence")
    plt.xlabel("Iterations")
    plt.ylabel("MAE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{exp_dir}/exp_epochs_{epoch}_samples_{n_samples}_lr_{lr}_generative_noise.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear advection inverse design")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dt_physics", type=float, default=0.001)
    parser.add_argument("--steps_physics", type=int, default=200)
    parser.add_argument("--N", type=int, default=64)   # interpreted as number of cells now
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--gen_noise", type=float, default=0.5)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--exp_dir", type=str, default="convection")

    args = parser.parse_args()
    device = args.device

    os.makedirs(args.exp_dir, exist_ok=True)

    solver = LinearAdvection1DOperator(
        n_cells=args.N,
        length=args.L,
        velocity=args.velocity,
        dt=args.dt_physics,
        num_steps=args.steps_physics,
    ).to(device)

    # State dimension now comes from the Firedrake space
    state_dim = solver.V.dim()

    # DG0 coordinates = cell centers
    x_grid_np = solver.get_dof_coordinates()
    x_grid = torch.tensor(x_grid_np, dtype=torch.float64, device=device)

    # Ground-truth initial condition in the DG space
    gt_ic = (
        torch.exp(-120.0 * (x_grid - 0.25) ** 2)
        + 0.7 * torch.exp(-180.0 * (x_grid - 0.70) ** 2)
    )

    with torch.no_grad():
        gt_final = solver(gt_ic)

    model = SimpleVectorField(n_points=state_dim, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    batch_size = args.n_samples
    rk_steps = 20
    loss_history = []
    loss_ic_history = []

    for epoch in tqdm(range(args.epochs)):
        optimizer.zero_grad()

        pred_ic = generate_ic(
            model=model,
            batch_size=batch_size,
            n_points=state_dim,
            noise_scale=args.gen_noise,
            rk_steps=rk_steps,
            device=device,
        )

        pred_final = torch.stack(
            [solver(pred_ic[k]) for k in range(batch_size)],
            dim=0,
        )

        loss = torch.mean(torch.abs(pred_final - gt_final.unsqueeze(0)))
        loss.backward()
        optimizer.step()

        loss_ic = torch.mean(torch.abs(pred_ic - gt_ic.unsqueeze(0))).detach().cpu().numpy()

        loss_ic_history.append(loss_ic.item())
        loss_history.append(loss.item())

        if epoch % 4 == 0:
            print(f"Iteración {epoch}: Loss = {loss.item():.6e}")

            gt_ic_cpu = gt_ic.detach().cpu().numpy()
            gt_final_cpu = gt_final.detach().cpu().numpy()
            x_grid_cpu = x_grid.detach().cpu().numpy()

            plot_1D(
                gt_ic_cpu=gt_ic_cpu,
                gt_final_cpu=gt_final_cpu,
                x_grid_cpu=x_grid_cpu,
                pred_ic=pred_ic,
                pred_final=pred_final,
                loss_history=loss_history,
                loss_ic = loss_ic_history,
                lr=args.lr,
                epoch=epoch,
                n_samples=args.n_samples,
                dt_physics=args.dt_physics,
                steps_physics=args.steps_physics,
                exp_dir = args.exp_dir
            )