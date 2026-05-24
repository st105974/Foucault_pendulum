import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ── параметры маятника ─────────────────────────────────────────────────────
L      = 10.8         # м
g_acc  = 9.81         # м/с²
omega0 = np.sqrt(g_acc / L)   # ≈ 0.953
m      = 1.5          # кг
gamma  = 0.0023       # с⁻¹
beta   = 0.0075       # м⁻¹  (из F = m·β·v²)

phi0   = 0.1          # рад  →  A0 = L·sin(φ0) ≈ 1.08 м
T_END  = 300.0         # с

# ── правая часть (физическое время) ────────────────────────────────────────
# φ̈ + 2γ φ̇ + (β·L) φ̇|φ̇| + (g/L) sin φ = 0
def rhs(t, y, gamma_, beta_):
    phi, dphi = y
    return [
        dphi,
        -2.0 * gamma_ * dphi
        - beta_ * L * dphi * abs(dphi)
        - (g_acc / L) * np.sin(phi),
    ]

# ── полная энергия (точная, нелинейная) ────────────────────────────────────
def energy(phi, dphi):
    return 0.5 * m * L**2 * dphi**2 + m * g_acc * L * (1.0 - np.cos(phi))

# ── два сценария ───────────────────────────────────────────────────────────
scenarios = {
    "Только линейное (β=0)":      dict(gamma_=gamma, beta_=0.0),
    "Только квадратичное (γ=0)": dict(gamma_=0.0,   beta_=beta),
}

t_eval = np.linspace(0, T_END, 2001)
results = {}

for name, params in scenarios.items():
    sol = solve_ivp(
        lambda t, y: rhs(t, y, **params),
        [0.0, T_END],
        [phi0, 0.0],
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )
    phi  = sol.y[0]
    dphi = sol.y[1]
    E    = energy(phi, dphi)
    A    = L * np.abs(np.sin(phi))  # мгновенное горизонт. смещение
    results[name] = dict(t=sol.t, phi=phi, dphi=dphi, E=E)

# ── численная мощность потерь в момент t=0 (при A0≈1.08 м) ─────────────────
print("=" * 78)
print(f"Параметры:  m={m} кг,  L={L} м,  ω0={omega0:.4f} рад/с")
print(f"            γ={gamma},  β={beta} м⁻¹,  φ0={phi0} рад,  A0={L*np.sin(phi0):.3f} м")
print("=" * 78)

# аналитические оценки на A0
A0 = L * np.sin(phi0)
P_v_analytic = gamma * m * omega0**2 * A0**2
P_q_analytic = m * beta * (A0 * omega0)**3 * 4.0 / (3.0 * np.pi)

print(f"\n[АНАЛИТИКА на A={A0:.3f} м]")
print(f"  Линейное:      ⟨P_loss⟩ = γ·m·ω0²·A² ≈ {P_v_analytic*1e3:7.3f} мВт")
print(f"  Квадратичн.: ⟨P_loss⟩ = m·β·(Aω0)³·4/(3π) ≈ {P_q_analytic*1e3:7.3f} мВт")
print(f"  Отношение P_q/P_v ≈ {P_q_analytic/P_v_analytic:.2f}")

# численные средние за первый период (T ≈ 6.6 с)
T_period = 2.0 * np.pi / omega0
print(f"\n[ЧИСЛЕННО: средние за первый период T≈{T_period:.2f} с]")
for name, r in results.items():
    mask = r["t"] <= T_period
    E_start = r["E"][0]
    E_end   = r["E"][mask][-1]
    dt      = r["t"][mask][-1] - r["t"][0]
    P_avg   = (E_start - E_end) / dt
    print(f"  {name:25s}: ⟨P⟩ ≈ {P_avg*1e3:7.3f} мВт")

# ── суммарная диссипация и относительное падение энергии за 300 с ─────────
print(f"\n[ЧИСЛЕННО за {T_END:.0f} с]")
for name, r in results.items():
    E0   = r["E"][0]
    E_end = r["E"][-1]
    print(f"  {name:25s}:  E(0)={E0*1e3:.2f} мДж  →  E({T_END:.0f})={E_end*1e3:.2f} мДж  "
          f"(потеряно {(E0-E_end)/E0*100:.1f}%)")

# ── график ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

colors = {"Только линейное (β=0)": "tab:blue", "Только квадратичное (γ=0)": "tab:red"}

for name, r in results.items():
    ax.plot(r["t"], r["E"] * 1e3, color=colors[name], lw=1.4, label=name)
ax.set_xlabel("время t, с")
ax.set_ylabel("E(t), мДж")
ax.set_title("Полная энергия маятника (без подкачки)")
ax.grid(True); ax.legend()

plt.tight_layout()
plt.savefig("out/compare_energy_decay.png", dpi=140, bbox_inches="tight")
print(f"\nГрафик сохранён в out/compare_energy_decay.png")