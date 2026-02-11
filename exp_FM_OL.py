import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import flax.linen as nn
import optax
import diffrax
import matplotlib.pyplot as plt
from typing import Union
from dataclasses import dataclass
from jax.tree_util import register_dataclass
from functools import partial


#@jax.tree_util.register_dataclass
@partial(register_dataclass, data_fields = [], meta_fields=['N', 'L','dt_physics','steps_physics'])
@dataclass
class domain1D:
    N: int = 64
    L: float = 1.0
    dt_physics: float = 0.001
    steps_physics: int = 200
    def __post_init__(self):
        self.dx = self.L/(self.N-1)

@jit
def sample_valid_ic(key, domain, scale=0.5, smooth=True):
    """
    Sample a random *valid* initial condition u0:
    - Dirichlet BC: u0[0]=u0[-1]=0
    - Optional smoothness via a simple low-pass filter
    - Bounded via tanh
    """
    z = random.normal(key, (domain.N,)) * scale

    if smooth:
        # Simple low-pass in Fourier domain (keeps only low frequencies)
        Z = jnp.fft.rfft(z)
        k = Z.shape[0]
        cutoff = jnp.maximum(2, k // 8)  # keep ~12.5% lowest frequencies
        mask = (jnp.arange(k) < cutoff).astype(Z.dtype)
        z = jnp.fft.irfft(Z * mask, n=domain.N)

    u0 = jnp.tanh(z)  # keep bounded
    u0 = u0.at[0].set(0.0)
    u0 = u0.at[-1].set(0.0)
    return u0


@partial(jit, static_argnums=(2,))
def solve_heat_equation_random(key, domain, alpha):
    """
    Every time you call this, it samples a new random valid IC and solves physics.
    Returns (ic, final).
    """
    ic = sample_valid_ic(key, domain)
    final = solve_heat_equation(ic, domain, alpha)
    return ic, final


# ==========================================
# 2. Física Diferenciable (Estable)
# ==========================================
@partial(jit, static_argnums=(2,))
def solve_heat_equation(ic, domain, alpha):
    """
    Resuelve la ecuación del calor en 1D.
    Entrada: ic (Condición Inicial) vector de tamaño (N,)
    Salida: estado final tras 'steps_physics'
    """
    # Función de un paso de tiempo (Euler Explícito Vectorizado)

    #N = 64            # Puntos en la línea
    N = domain.N
    #L = 1.0           # Longitud
    L = domain.L
    #dx = L / (N - 1)
    dx = domain.dx
    #alpha = 0.05      # Coeficiente de difusión
    dt_physics = domain.dt_physics
    steps_physics = domain.steps_physics # Pasos de simulación física

    diag = -2.0 * jnp.ones(N)
    off_diag = jnp.ones(N - 1)
    laplacian_matrix = (jnp.diag(diag) +
                        jnp.diag(off_diag, k=1) +
                        jnp.diag(off_diag, k=-1)) / (dx**2)
    def step_fn(u, _):
        # u_new = u + dt * alpha * (Laplacian @ u)
        dudt = alpha * jnp.dot(laplacian_matrix, u)
        u_new = u + dt_physics * dudt

        # Boundary Conditions (Dirichlet): Extremos fijos a 0
        u_new = u_new.at[0].set(0.0)
        u_new = u_new.at[-1].set(0.0)
        return u_new, None

    final_u, _ = jax.lax.scan(step_fn, ic, None, length=steps_physics)
    return final_u

# ==========================================
# 3. Flow Matching Network (MLP)
# ==========================================
from typing import Callable

import flax.linen as nn
from jax import numpy as jnp
from jax import random


def normal(stddev=1e-2, dtype = jnp.float32) -> Callable:
  def init(key, shape, dtype=dtype):
    keys = random.split(key)
    return random.normal(keys[0], shape) * stddev
  return init

class SpectralConv2d(nn.Module):
  out_channels: int = 32
  modes1: int = 12
  modes2: int = 12

  @nn.compact
  def __call__(self, x):
    # x.shape: [batch, height, width, in_channels]

    # Initialize parameters
    in_channels = x.shape[-1]
    scale = 1/(in_channels * self.out_channels)
    in_channels = x.shape[-1]
    height = x.shape[1]
    width = x.shape[2]

    # Checking that the modes are not more than the input size
    assert self.modes1 <= height//2 + 1
    assert self.modes2 <= width//2 + 1
    assert height % 2 == 0 # Only tested for even-sized inputs
    assert width % 2 == 0 # Only tested for even-sized inputs

    # The model assumes real inputs and therefore uses a real
    # fft. For a 2D signal, the conjugate symmetry of the
    # transform is exploited to reduce the number of operations.
    # Given an input signal of dimesions (N, C, H, W), the
    # output signal will have dimensions (N, C, H, W//2+1).
    # Therefore the kernel weigths will have different dimensions
    # for the two axis.
    kernel_1_r = self.param(
      'kernel_1_r',
      normal(scale, jnp.float32),
      (in_channels, self.out_channels, self.modes1, self.modes2),
      jnp.float32
    )
    kernel_1_i = self.param(
      'kernel_1_i',
      normal(scale, jnp.float32),
      (in_channels, self.out_channels, self.modes1, self.modes2),
      jnp.float32
    )
    kernel_2_r = self.param(
      'kernel_2_r',
      normal(scale, jnp.float32),
      (in_channels, self.out_channels, self.modes1, self.modes2),
      jnp.float32
    )
    kernel_2_i = self.param(
      'kernel_2_i',
      normal(scale, jnp.float32),
      (in_channels, self.out_channels, self.modes1, self.modes2),
      jnp.float32
    )

    # Perform fft of the input
    x_ft = jnp.fft.rfftn(x, axes=(1, 2))

    # Multiply the center of the spectrum by the kernel
    out_ft = jnp.zeros_like(x_ft)
    s1 = jnp.einsum(
      'bijc,coij->bijo',
      x_ft[:, :self.modes1, :self.modes2, :],
      kernel_1_r + 1j*kernel_1_i)
    s2 = jnp.einsum(
      'bijc,coij->bijo',
      x_ft[:, -self.modes1:, :self.modes2, :],
      kernel_2_r + 1j*kernel_2_i)
    out_ft = out_ft.at[:, :self.modes1, :self.modes2, :].set(s1)
    out_ft = out_ft.at[:, -self.modes1:, :self.modes2, :].set(s2)

    # Go back to the spatial domain
    y = jnp.fft.irfftn(out_ft, axes=(1, 2))

    return y

class FourierStage(nn.Module):
  out_channels: int = 32
  modes1: int = 12
  modes2: int = 12
  activation: Callable = nn.gelu

  @nn.compact
  def __call__(self, x):
    x_fourier = SpectralConv2d(
      out_channels=self.out_channels,
      modes1=self.modes1,
      modes2=self.modes2
    )(x)
    x_local = nn.Conv(
      self.out_channels,
      (1,1),
    )(x)
    return self.activation(x_fourier + x_local)


class FNO2D(nn.Module):
  r'''
  Fourier Neural Operator for 2D signals.

  Implemented from
  https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py

  Attributes:
    modes1: Number of modes in the first dimension.
    modes2: Number of modes in the second dimension.
    width: Number of channels to which the input is lifted.
    depth: Number of Fourier stages
    channels_last_proj: Number of channels in the hidden layer of the last
      2-layers Fully Connected (channel-wise) network
    activation: Activation function to use
    out_channels: Number of output channels, >1 for non-scalar fields.
  '''
  modes1: int = 12
  modes2: int = 12
  width: int = 32
  depth: int = 4
  channels_last_proj: int = 128
  activation: Callable = nn.gelu
  out_channels: int = 1
  padding: int = 0 # Padding for non-periodic inputs

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    # Generate coordinate grid, and append to input channels
    grid = self.get_grid(x)
    x = jnp.concatenate([x, grid], axis=-1)

    # Lift the input to a higher dimension
    x = nn.Dense(self.width)(x)

    # Pad input
    if self.padding > 0:
      x = jnp.pad(
        x,
        ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
        mode='constant'
      )

    # Apply Fourier stages, last one has no activation
    # (can't find this in the paper, but is in the original code)
    for depthnum in range(self.depth):
      activation = self.activation if depthnum < self.depth - 1 else lambda x: x
      x = FourierStage(
        out_channels=self.width,
        modes1=self.modes1,
        modes2=self.modes2,
        activation=activation
      )(x)

    # Unpad
    if self.padding > 0:
      x = x[:, :-self.padding, :-self.padding, :]

    # Project to the output channels
    x = nn.Dense(self.channels_last_proj)(x)
    x = self.activation(x)
    x = nn.Dense(self.out_channels)(x)

    return x


  @staticmethod
  def get_grid(x):
    x1 = jnp.linspace(0, 1, x.shape[1])
    x2 = jnp.linspace(0, 1, x.shape[2])
    x1, x2 = jnp.meshgrid(x1, x2, indexing = 'ij')
    grid = jnp.expand_dims(jnp.stack([x1, x2], axis=-1), 0)
    batched_grid = jnp.repeat(grid, x.shape[0], axis=0)
    return batched_grid
# ==========================================
# 4. Generador (Integrador ODE)
# ==========================================

@partial(jit, static_argnums=(4,1,5))
def generate_ic(params, model, rng_key, domain, gen_noise, stochastic):
    """Transfoma Ruido -> Condición Inicial Candidata"""
    #if isinstance(stochastic,str):
    if stochastic=="normal":
        #z0 = random.normal(rng_key, (domain.N,)) * gen_noise # Ruido inicial
        z0 = getattr(random,stochastic)(rng_key, (domain.N,)) * gen_noise # Ruido inicial
    elif stochastic=="uniform":
        z0 = random.ball(rng_key, 1, p=2, shape=(domain.N,))
    else:
        z0 = jnp.zeros((domain.N,))

    def ode_func(t, y, args):
        return model.apply(params, y, t)

    # Solver ODE para el Flow
    term = diffrax.ODETerm(ode_func)
    solver = diffrax.Tsit5() # Solver Runge-Kutta rápido
    sol = diffrax.diffeqsolve(term, solver, t0=0.0, t1=1.0, dt0=0.01, y0=z0)

    generated_ic = sol.ys[-1]

    # Forzamos condiciones de borde 0 para que sea físicamente válido
    generated_ic = generated_ic.at[0].set(0.0)
    generated_ic = generated_ic.at[-1].set(0.0)

    return generated_ic


@partial(jit, static_argnums=(3,4,6,7))
def loss_fn(params, key,domain,alpha,n_samples, gt_ic, gen_noise, stochastic):
    # 1. Flow: Generar IC candidata
    key, key_, *keys = random.split(key,num=n_samples + 2 )
    

    pred_ic = jax.vmap(lambda k: generate_ic(params, model, k, domain, gen_noise, stochastic))(jnp.array(keys))
    gt_final = jax.vmap(lambda ic: solve_heat_equation(gt_ic , domain, alpha))(jnp.array(keys))
    pred_final = jax.vmap(lambda ic: solve_heat_equation(ic, domain, alpha))(pred_ic)
    

    loss = jnp.mean((pred_final - gt_final)**2)
    ic_loss = jnp.mean((gt_ic - pred_ic)**2)
    return loss, (pred_ic, pred_final,ic_loss,gt_ic,gt_final)

    ## 3. Error: Comparar con el estado final real
    #loss = jnp.mean(jax.numpy.absolute(pred_final - gt_final))
    #return loss, (pred_ic, pred_final)

@partial(jit, static_argnums=(4,5,7,8))
def train_step(params, opt_state, key, domain, alpha, n_samples, gt_ic, gen_noise, stochastic):
    (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(params, key,  domain, alpha, n_samples, gt_ic,gen_noise, stochastic)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Exps')
    parser.add_argument('--n_samples', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--dt_physics', type=float, required=True)
    parser.add_argument('--steps_physics', type=int, default=500)
    parser.add_argument('--N', type=int, default='outputs')
    parser.add_argument('--epochs', type=int, default='outputs')
    parser.add_argument('--L', type=float, default=0)
    parser.add_argument('--gen_noise', type=float, default=0.5)
    parser.add_argument('--stochastic', type=str, default="constant")
    
    args = parser.parse_args()

    domain = domain1D(
        N = args.N,
        L = args.L,
        dt_physics = args.dt_physics ,
        steps_physics = args.steps_physics,
    )
    alpha = 0.05

    x_grid = jnp.linspace(0, domain.L, domain.N)
    gt_ic = jnp.exp(-100 * (x_grid - 0.3)**2) + 0.5 * jnp.exp(-100 * (x_grid - 0.7)**2)

    # Simulamos para obtener lo que "vemos" en la realidad (Target)
    gt_final = solve_heat_equation(gt_ic,domain,alpha)

    print("Objetivo: Encontrar la curva inicial que generó el estado final observado.")

    #model = SimpleVectorField()
    model = FNO2D(modes1=12, modes2=12, width=32, depth=4)
    key = random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, 64, 64, 1)))
    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(params)


    # Bucle de entrenamiento
    loss_history = []
    ic_loss_history = []
    print("\nComenzando entrenamiento...")

    
    #epochs = 5000
    for i in range(args.epochs):
        key, subkey = random.split(key)
        params, opt_state, loss, (curr_ic, curr_final, ic_loss,gt_ic,gt_final) = train_step(params, opt_state, subkey, domain, alpha, args.n_samples, gt_ic,args.gen_noise, args.stochastic)
        loss_history.append(loss)
        ic_loss_history.append(ic_loss)
    
        if i % 100 == 0:
            print(f"Iteración {i}: Loss = {loss:.6f}")
    
    # ==========================================
    # 7. Visualización
    # ==========================================
            plt.figure(figsize=(20, 5))

            # Gráfica 1: Condiciones Iniciales (Lo que el modelo imagina vs Realidad)
            curr_ic_m = jnp.mean(curr_ic,axis=0)
            curr_ic_v = jnp.std(curr_ic,axis=0)

            curr_gt_m = jnp.mean(curr_final,axis=0)
            curr_gt_v = jnp.std(curr_final,axis=0)
            plt.subplot(1, 4, 1)
            plt.plot(x_grid, gt_ic, 'k--', label='Real IC', linewidth=2)
            plt.plot(x_grid, curr_ic_m, 'r-', label='Generated Flow', linewidth=2)
            plt.fill_between(x_grid, curr_ic_m + curr_ic_v,curr_ic_m - curr_ic_v,color = "r",alpha=0.5, linewidth=2)
            plt.title("Initial Condition (t=0)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 4, 2)
            plt.plot(x_grid, gt_ic, 'k--', label='Real IC', linewidth=2)
            plt.plot(x_grid, jnp.mean(curr_ic,0), 'r-', label='Generated Flow', linewidth=2)
            plt.title("Initial Condition (t=0)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Gráfica 2: Estado Final (Lo que observamos)
            plt.subplot(1, 4, 3)
            plt.plot(x_grid, gt_final[0], 'k--', label='Observation', linewidth=2)
            plt.plot(x_grid, curr_gt_m, 'b-', label='Solution from flow', linewidth=2)
            plt.fill_between(x_grid, curr_gt_m + curr_gt_v,curr_gt_m - curr_gt_v,color = "b",alpha=0.5, linewidth=2)
            plt.title(f"Final state (t={args.dt_physics*args.steps_physics:.2f})")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Gráfica 3: Curva de Aprendizaje
            plt.subplot(1, 4, 4)
            plt.plot(loss_history)
            plt.plot(ic_loss_history)
            plt.yscale('log')
            plt.title(f"Convergence {args.n_samples}")
            plt.xlabel("Iterations")
            plt.ylabel("MSE Loss")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'exp_epochs_{i}_samples_{args.n_samples}_lr_{args.lr}_generative_noise_{args.gen_noise}.png')
    #plt.show()