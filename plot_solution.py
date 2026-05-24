import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ── параметры ─────────────────────────────────────────────────────────────
L0     = 10.8
omega0 = 0.952
gamma  = 0.0023
beta   = 0

Gamma = gamma / omega0
B     = beta * L0

eps   = 0.0033
delta = 0.0
a     = 2.0 + delta

A0_m   = 1.0          # начальная амплитуда, м
y0     = [A0_m / L0, 0.0]

# ── интегрирование ────────────────────────────────────────────────────────
N   = 300             # число периодов подкачки
T   = 2.0 * np.pi / a
tau_end = N * T
pts_per_period = 200
tau_eval = np.linspace(0, tau_end, N * pts_per_period)

def rhs(tau, y):
    phi, dphi = y
    ell  = 1.0 + eps * np.sin(a * tau)
    dell = a * eps * np.cos(a * tau)
    ddphi = (
        - 2.0 * (dell / ell) * dphi
        - np.sin(phi) / ell
        - 2.0 * Gamma * dphi
        - B * ell * dphi * abs(dphi)
    )
    return [dphi, ddphi]

sol = solve_ivp(rhs, [0.0, tau_end], y0,
                method='RK45', t_eval=tau_eval,
                rtol=1e-9, atol=1e-11)

# ── перевод в физические единицы ──────────────────────────────────────────
t_phys = sol.t / omega0          # с
x_m    = sol.y[0] * L0           # м

# ── графики ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(t_phys, x_m, 'b', lw=0.6)
ax.set_xlabel('Время $t$, с')
ax.set_ylabel('Отклонение $x = L_0\\varphi$, м')
ax.set_title(f'Решение нелинейного уравнения маятника  '
             f'($\\varepsilon={eps}$)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('out/solution_eps005_delta0.png', dpi=150, bbox_inches='tight')
print(f'Установившаяся амплитуда: {np.abs(x_m[-10*pts_per_period:]).max():.3f} м')
plt.show()
