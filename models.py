import numpy as np
import torch
from torch import nn


def enforce_zero_dirichlet(u: torch.Tensor) -> torch.Tensor:
    """
    Enforce u[..., 0] = u[..., -1] = 0.
    Works for shape (N,) or (B, N).
    """
    out = u.clone()
    out[..., 0] = 0.0
    out[..., -1] = 0.0
    return out


class SimpleVectorField(nn.Module):
    """
    Receives state x and scalar time t, returns dx/dt.
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


class SimpleParams(nn.Module):
    """
    Receives state x and scalar time t, returns dx/dt.
    """
    def __init__(self, n_points: int, hidden_dim: int = 256):
        super().__init__()
        self.n_points = n_points
        self.IC = torch.from_numpy(np.random.uniform(0,1,(n_points))).requires_grad_(True)

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



def rk4_integrate_vector_field(
    model: nn.Module,
    z0: torch.Tensor,
    n_steps: int = 20
) -> torch.Tensor:
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


def generate_ic(
    model: nn.Module,
    batch_size: int,
    n_points: int,
    noise_scale: float = 0.5,
    rk_steps: int = 20,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Noise -> continuous flow -> candidate initial condition.
    """
    z0 = noise_scale * torch.randn(batch_size, n_points, device=device)
    ic = rk4_integrate_vector_field(model, z0, n_steps=rk_steps)
    ic = enforce_zero_dirichlet(ic)
    return ic