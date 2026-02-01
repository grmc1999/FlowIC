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
#class SimpleVectorField(n.Module):
#    """
#    Red simple que recibe el estado actual (vector N) y el tiempo t,
#    y predice la velocidad de cambio para la EDO generativa.
#    """
#    @nn.compact
#    def __call__(self, x, t):
#        # x shape: (N,)
#        # t shape: escalar
#
#        # Concatenamos el tiempo al vector de entrada
#        t_vec = jnp.ones((1,)) * t
#        inp = jnp.concatenate([x, t_vec], axis=0)
#
#        # MLP simple: Dense -> Gelu -> Dense
#        h = nn.Dense(256)(inp)
#        #h = nn.BatchNorm(128)(h)
#        h = nn.gelu(h)
#        h = nn.Dense(256)(h)
#        #h = nn.BatchNorm(128)(h)
#        h = nn.gelu(h)
#        # Salida del mismo tamaño que la entrada física (N)
#        out = nn.Dense(64)(h)
#        return out
    
class SimpleVectorField(nn.Module):
    """
    Red Convolucional (CNN 1D) para capturar correlaciones espaciales.
    Recibe estado (N,) y tiempo t, predice velocidad (N,).
    """
    @nn.compact
    def __call__(self, x, t):
        # x shape: (N,)
        # t shape: escalar
        
        N_points = x.shape[0]
        
        # 1. Preparar input para Conv: (Sequence, Channels) -> (N, 1)
        x_in = x.reshape((N_points, 1))
        
        # 2. Canal de Tiempo: (N, 1) lleno con el valor de t
        t_in = jnp.full((N_points, 1), t)
        
        # Concatenamos: Input es (N, 2) -> valor y tiempo en cada punto espacial
        inp = jnp.concatenate([x_in, t_in], axis=-1)

        # 3. Backbone Convolucional (Respetando localidad espacial)
        # Padding 'SAME' mantiene la dimensión N
        h = nn.Conv(features=64, kernel_size=5, padding='SAME')(inp)
        h = nn.gelu(h)
        
        # 4. Proyección de salida
        # Queremos volver a 1 solo canal (velocidad)
        h = nn.Conv(features=1, kernel_size=3, padding='SAME')(h)
        
        # Flatten para volver a (N,)
        out = h.reshape((N_points,))
        return out

# ==========================================
# 4. Generador (Integrador ODE)
# ==========================================
def generate_ic(params, model, rng_key, domain):
    """Transfoma Ruido -> Condición Inicial Candidata"""
    z0 = random.normal(rng_key, (domain.N,)) * 0.5 # Ruido inicial

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


@partial(jit, static_argnums=(3,4))
def loss_fn(params, key,domain,alpha,n_samples, gt_ic):
    # 1. Flow: Generar IC candidata
    key, key_, *keys = random.split(key,num=n_samples + 2 )
    #pred_ic = generate_ic(params, model, key)
#
    ## 2. Física: Simular futuro
    #pred_final = solve_heat_equation(pred_ic)
    #pred_ic,pred_final = jax.vmap(lambda key: (
    #    generate_ic(params,model,key,domain),
    #    solve_heat_equation_random(generate_ic(params,model,key,domain),domain,alpha)
    #    ),0)(jnp.array(keys))

    #gt_ic,gt_final = solve_heat_equation_random(jnp.array(key_), domain, alpha)
    

    pred_ic = jax.vmap(lambda k: generate_ic(params, model, k, domain))(jnp.array(keys))
    gt_final = jax.vmap(lambda ic: solve_heat_equation(gt_ic , domain, alpha))(jnp.array(keys))
    pred_final = jax.vmap(lambda ic: solve_heat_equation(ic, domain, alpha))(pred_ic)
    

    loss = jnp.mean((pred_final - gt_final)**2)
    ic_loss = jnp.mean((gt_ic - pred_ic)**2)
    return loss, (pred_ic, pred_final,ic_loss,gt_ic,gt_final)

    ## 3. Error: Comparar con el estado final real
    #loss = jnp.mean(jax.numpy.absolute(pred_final - gt_final))
    #return loss, (pred_ic, pred_final)

@partial(jit, static_argnums=(4,5))
def train_step(params, opt_state, key, domain, alpha, n_samples, gt_ic):
    (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(params, key,  domain, alpha, n_samples, gt_ic)
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

    model = SimpleVectorField()
    key = random.PRNGKey(0)
    params = model.init(key, jnp.zeros(domain.N), 0.0)
    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(params)


    # Bucle de entrenamiento
    loss_history = []
    ic_loss_history = []
    print("\nComenzando entrenamiento...")

    
    #epochs = 5000
    for i in range(args.epochs):
        key, subkey = random.split(key)
        params, opt_state, loss, (curr_ic, curr_final, ic_loss,gt_ic,gt_final) = train_step(params, opt_state, subkey, domain, alpha, args.n_samples, gt_ic)
        loss_history.append(loss)
        ic_loss_history.append(ic_loss)
    
        if i % 100 == 0:
            print(f"Iteración {i}: Loss = {loss:.6f}")
    
    # ==========================================
    # 7. Visualización
    # ==========================================
            plt.figure(figsize=(20, 5))

            # Gráfica 1: Condiciones Iniciales (Lo que el modelo imagina vs Realidad)
            plt.subplot(1, 4, 1)
            plt.plot(x_grid, gt_ic, 'k--', label='Real IC (Secreta)', linewidth=2)
            plt.plot(x_grid, curr_ic[0], 'r-', label='Flow Generada', linewidth=2)
            plt.plot(x_grid, curr_ic[2], 'r-', label='Flow Generada', linewidth=2)
            plt.plot(x_grid, curr_ic[4], 'r-', label='Flow Generada', linewidth=2)
            plt.plot(x_grid, curr_ic[6], 'r-', label='Flow Generada', linewidth=2)
            plt.plot(x_grid, curr_ic[7], 'r-', label='Flow Generada', linewidth=2)
            plt.title("Condición Inicial (t=0)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 4, 2)
            plt.plot(x_grid, gt_ic, 'k--', label='Real IC (Secreta)', linewidth=2)
            plt.plot(x_grid, jnp.mean(curr_ic,0), 'r-', label='Flow Generada', linewidth=2)
            plt.title("Condición Inicial (t=0)")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Gráfica 2: Estado Final (Lo que observamos)
            plt.subplot(1, 4, 3)
            plt.plot(x_grid, gt_final[0], 'k--', label='Observación Real', linewidth=2)
            plt.plot(x_grid, curr_final[0], 'b-', label='Simulación desde Flow', linewidth=2)
            plt.title(f"Estado Final (t={args.dt_physics*args.steps_physics:.2f})")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Gráfica 3: Curva de Aprendizaje
            plt.subplot(1, 4, 4)
            plt.plot(loss_history)
            plt.plot(ic_loss_history)
            plt.yscale('log')
            plt.title(f"Convergencia del Error samples {args.n_samples}")
            plt.xlabel("Iteraciones")
            plt.ylabel("MSE Loss")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'exp_epochs_{i}_samples_{args.n_samples}_lr_{args.lr}.png')
    #plt.show()
