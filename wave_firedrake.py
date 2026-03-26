from abc import ABC, abstractmethod
from SolverBase import BaseFiredrakeOperator
from models import SimpleVectorField,generate_ic,enforce_zero_dirichlet,rk4_integrate_vector_field
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import firedrake as fd
import matplotlib.pyplot as plt

from tqdm import tqdm
from firedrake.ml.pytorch.fem_operator import fem_operator
from firedrake.adjoint import Control, ReducedFunctional
from models import enforce_zero_dirichlet,SimpleVectorField,generate_ic

torch.set_default_dtype(torch.float64)




class WaveEquation1DOperator(BaseFiredrakeOperator):
    """
    1D wave equation:
        u_tt - c^2 u_xx = 0
    on [0, L] with homogeneous Dirichlet BCs.

    Design choice for compatibility with BaseFiredrakeOperator:
    - control: initial displacement u(x,0)
    - fixed:   initial velocity u_t(x,0) = 0
    - output:  final displacement u(x,T)

    Time discretization:
    - central difference in time
    - CG in space
    - each step solves a mass-matrix system
    """
    def __init__(
        self,
        n_points: int = 64,
        length: float = 1.0,
        wave_speed: float = 1.0,
        dt: float = 1e-3,
        num_steps: int = 200,
        degree: int = 1,
    ):
        self.n_points = n_points
        self.length = length
        self.wave_speed_value = wave_speed
        self.dt_value = dt
        self.num_steps = num_steps
        self.degree = degree
        super().__init__()

    def build_mesh(self):
        return fd.IntervalMesh(self.n_points - 1, self.length)

    def build_function_space(self, mesh):
        return fd.FunctionSpace(mesh, "CG", self.degree)

    def setup_problem(self):
        self.c = fd.Constant(self.wave_speed_value)
        self.dt = fd.Constant(self.dt_value)

        self.u_trial = fd.TrialFunction(self.V)
        self.v_test = fd.TestFunction(self.V)

        self.bc = fd.DirichletBC(self.V, fd.Constant(0.0), "on_boundary")

        # Mass matrix on the left-hand side
        self.mass_form = (self.u_trial * self.v_test) * fd.dx

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        return enforce_zero_dirichlet(x)

    def solve_annotated(self, u0: fd.Function) -> fd.Function:
        """
        Uses:
          u^1 from Taylor expansion with zero initial velocity
          u^{n+1} = 2u^n - u^{n-1} + dt^2 c^2 u_xx^n
        written in weak form.

        Weak forms:
          (u^1, v) = (u^0, v) - 0.5 dt^2 c^2 (grad u^0, grad v)
          (u^{n+1}, v) = (2u^n - u^{n-1}, v) - dt^2 c^2 (grad u^n, grad v)
        """
        u_prev = fd.Function(self.V, name="u_prev")
        u_prev.assign(u0)

        if self.num_steps == 0:
            return u_prev.copy(deepcopy=True)

        # First step: zero initial velocity
        u_curr = fd.Function(self.V, name="u_curr")
        L_init = (
            u_prev * self.v_test
            - 0.5 * (self.dt ** 2) * (self.c ** 2) * fd.dot(fd.grad(u_prev), fd.grad(self.v_test))
        ) * fd.dx

        fd.solve(
            self.mass_form == L_init,
            u_curr,
            bcs=self.bc,
            solver_parameters={
                "ksp_type": "cg",
                "pc_type": "sor",
            },
        )

        if self.num_steps == 1:
            return u_curr.copy(deepcopy=True)

        u_next = fd.Function(self.V, name="u_next")

        for _ in range(1, self.num_steps):
            L_step = (
                (2.0 * u_curr - u_prev) * self.v_test
                - (self.dt ** 2) * (self.c ** 2) * fd.dot(fd.grad(u_curr), fd.grad(self.v_test))
            ) * fd.dx

            fd.solve(
                self.mass_form == L_step,
                u_next,
                bcs=self.bc,
                solver_parameters={
                    "ksp_type": "cg",
                    "pc_type": "sor",
                },
            )

            u_prev.assign(u_curr)
            u_curr.assign(u_next)

        return u_curr.copy(deepcopy=True)


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
    exp_dir = "wave"
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
    parser = argparse.ArgumentParser(description="1D wave inverse design")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dt_physics", type=float, default=5e-4)
    parser.add_argument("--steps_physics", type=int, default=400)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--gen_noise", type=float, default=0.5)
    parser.add_argument("--wave_speed", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--exp_dir", type=str, default="wave")
    parser.add_argument("--generative", action = "store_true")
    parser.add_argument("--noisy_obs", action = "store_false")
    args = parser.parse_args()
    
    os.makedirs(args.exp_dir, exist_ok=True)

    device = args.device

    solver = WaveEquation1DOperator(
        n_points=args.N,
        length=args.L,
        wave_speed=args.wave_speed,
        dt=args.dt_physics,
        num_steps=args.steps_physics,
        degree=1,
    ).to(device)

    state_dim = solver.V.dim() # num of grid points
    x_grid = torch.linspace(0.0, args.L, state_dim, device=device)

    # Ground-truth initial displacement
    gt_ic = (
        1.0 * torch.sin(torch.pi * x_grid / args.L)
        + 0.35 * torch.sin(3.0 * torch.pi * x_grid / args.L)
    )
    gt_ic = enforce_zero_dirichlet(gt_ic)

    with torch.no_grad():
        gt_final_o = solver(gt_ic)
        if args.noisy_obs:
            gt_final = gt_final_o + torch.randn(gt_final_o.shape)*0.05

    if args.generative:
        print("train generative")
        model = SimpleVectorField(n_points=state_dim, hidden_dim=256).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        print("train simplests")
        model = torch.rand((args.n_samples,state_dim),requires_grad = True).to(device)
        optimizer = torch.optim.Adam([model], lr=args.lr)

    batch_size = args.n_samples
    rk_steps = 20
    loss_history = []
    loss_ic_history = []

    for epoch in tqdm(range(args.epochs)):
        optimizer.zero_grad()

        if args.generative:
            pred_ic = generate_ic(
            model=model,
            batch_size=batch_size,
            n_points=state_dim,
            noise_scale=args.gen_noise,
            rk_steps=rk_steps,
            device=device,
        )
        else:
            pred_ic = model

        pred_final = torch.stack([solver(pred_ic[k]) for k in range(batch_size)], dim=0)

        loss = torch.mean(torch.abs(pred_final - gt_final_o.unsqueeze(0)))
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

            np.save(np.stack(loss_ic_history, axis = 0),"loss_ic_history")
            np.save(np.stack(loss_history, axis = 0),"loss_history")
            np.save(gt_ic_cpu,f"gt_ic_cpu_{epoch}.npy")
            np.save(gt_final_cpu,f"gt_final_cpu_{epoch}.npy")

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