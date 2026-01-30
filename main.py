import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import flax.linen as nn
import optax
import diffrax
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuración del Dominio 1D
# ==========================================
N = 64            # Puntos en la línea
L = 1.0           # Longitud
dx = L / (N - 1)
alpha = 0.05      # Coeficiente de difusión
dt_physics = 0.001
steps_physics = 200 # Pasos de simulación física

# Matriz del Laplaciano 1D (Diferencias Finitas) pre-calculada
# Esto hace la física extremadamente rápida y estable
diag = -2.0 * jnp.ones(N)
off_diag = jnp.ones(N - 1)
laplacian_matrix = (jnp.diag(diag) +
                    jnp.diag(off_diag, k=1) +
                    jnp.diag(off_diag, k=-1)) / (dx**2)

# ==========================================
# 2. Física Diferenciable (Estable)
# ==========================================
@jit
def solve_heat_equation(ic):
    """
    Resuelve la ecuación del calor en 1D.
    Entrada: ic (Condición Inicial) vector de tamaño (N,)
    Salida: estado final tras 'steps_physics'
    """
    # Función de un paso de tiempo (Euler Explícito Vectorizado)
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
class SimpleVectorField(nn.Module):
    """
    Red simple que recibe el estado actual (vector N) y el tiempo t,
    y predice la velocidad de cambio para la EDO generativa.
    """
    @nn.compact
    def __call__(self, x, t):
        # x shape: (N,)
        # t shape: escalar

        # Concatenamos el tiempo al vector de entrada
        t_vec = jnp.ones((1,)) * t
        inp = jnp.concatenate([x, t_vec], axis=0)

        # MLP simple: Dense -> Gelu -> Dense
        h = nn.Dense(256)(inp)
        #h = nn.BatchNorm(128)(h)
        h = nn.gelu(h)
        h = nn.Dense(256)(h)
        #h = nn.BatchNorm(128)(h)
        h = nn.gelu(h)
        # Salida del mismo tamaño que la entrada física (N)
        out = nn.Dense(N)(h)
        return out

# ==========================================
# 4. Generador (Integrador ODE)
# ==========================================
def generate_ic(params, model, rng_key):
    """Transfoma Ruido -> Condición Inicial Candidata"""
    z0 = random.normal(rng_key, (N,)) * 0.5 # Ruido inicial

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

# ==========================================
# 5. Preparar el "Ground Truth" (El Objetivo)
# ==========================================
# Creamos una condición inicial real (Doble pico gaussiano)
x_grid = jnp.linspace(0, L, N)
gt_ic = jnp.exp(-100 * (x_grid - 0.3)**2) + 0.5 * jnp.exp(-100 * (x_grid - 0.7)**2)

# Simulamos para obtener lo que "vemos" en la realidad (Target)
gt_final = solve_heat_equation(gt_ic)

print("Objetivo: Encontrar la curva inicial que generó el estado final observado.")

# ==========================================
# 6. Entrenamiento (Inverse Physics Design)
# ==========================================
model = SimpleVectorField()
key = random.PRNGKey(0)
params = model.init(key, jnp.zeros(N), 0.0)
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

@jit
def loss_fn(params, key):
    # 1. Flow: Generar IC candidata
    key, *keys = random.split(key,num=100)
    #pred_ic = generate_ic(params, model, key)
#
    ## 2. Física: Simular futuro
    #pred_final = solve_heat_equation(pred_ic)
    pred_ic,pred_final = jax.vmap(lambda key: (generate_ic(params,model,key),solve_heat_equation(generate_ic(params,model,key))),0)(jnp.array(keys))

    # 3. Error: Comparar con el estado final real
    loss = jnp.mean(jax.numpy.absolute(pred_final - gt_final))
    return loss, (pred_ic, pred_final)

@jit
def train_step(params, opt_state, key):
    (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(params, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

# Bucle de entrenamiento
loss_history = []
print("\nComenzando entrenamiento...")

epochs = 5000
for i in range(epochs):
    key, subkey = random.split(key)
    params, opt_state, loss, (curr_ic, curr_final) = train_step(params, opt_state, subkey)
    loss_history.append(loss)

    if i % 100 == 0:
        print(f"Iteración {i}: Loss = {loss:.6f}")

# ==========================================
# 7. Visualización
# ==========================================
plt.figure(figsize=(15, 5))

# Gráfica 1: Condiciones Iniciales (Lo que el modelo imagina vs Realidad)
plt.subplot(1, 3, 1)
plt.plot(x_grid, gt_ic, 'k--', label='Real IC (Secreta)', linewidth=2)
plt.plot(x_grid, curr_ic, 'r-', label='Flow Generada', linewidth=2)
plt.title("Condición Inicial (t=0)")
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfica 2: Estado Final (Lo que observamos)
plt.subplot(1, 3, 2)
plt.plot(x_grid, gt_final, 'k--', label='Observación Real', linewidth=2)
plt.plot(x_grid, curr_final, 'b-', label='Simulación desde Flow', linewidth=2)
plt.title(f"Estado Final (t={dt_physics*steps_physics:.2f})")
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfica 3: Curva de Aprendizaje
plt.subplot(1, 3, 3)
plt.plot(loss_history)
plt.yscale('log')
plt.title("Convergencia del Error")
plt.xlabel("Iteraciones")
plt.ylabel("MSE Loss")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()