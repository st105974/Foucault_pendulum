"""
Поиск устойчивого предельного цикла для параметрически возбуждаемого
маятника с переменной длиной.

Критерий принятия точки delta = (1 ∧ 2 ∧ 3):

  1) Периодичность отображения Пуанкаре на периоде T_c = 2*T_L
     (отклик основного параметрического резонанса a/2):
         max_k ||y(k*T_c) - y((k-1)*T_c)|| / ||y((k-1)*T_c)|| < eta_rel

  2) Стабилизация амплитуды на последних m_amp периодах:
         max_k |A_k - <A>| / <A> < eta_amp,
     где A_k = (max(phi) - min(phi)) / 2 на k-м периоде T_c.

  3) Близость к целевой амплитуде:
         |<A>·L0 - A_target_m| < tol_A.

Амплитуда никогда не вычисляется как ||[phi, v]||: это работает только
для гармонического осциллятора в 1:1, а здесь у параметрического цикла
свой период (T_c = 2*T_L) и нелинейность.

Сравнение для отображения Пуанкаре идёт по полному фазовому вектору
[phi, v], а не только по phi.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ── параметры маятника ────────────────────────────────────────────────────
L0     = 10.8       # длина подвеса, м
omega0 = 0.952      # собственная частота, рад/с
gamma  = 0.0023     # вязкое трение, 1/с
beta   = 0     # коэф. квадратичного сопротивления, 1/м

Gamma = gamma / omega0   # безразмерный вязкий коэффициент
B     = beta * L0        # безразмерный квадратичный коэффициент
eps   = 0.0033             # амплитуда модуляции длины
A_target_m = 1.0    # целевая амплитуда, м


# ── правая часть системы ──────────────────────────────────────────────────
def rhs(tau, y, eps, delta):
    """
    Система первого порядка для phi(tau) и v(tau)=phi'(tau).

    Уравнение в безразмерном времени:
        phi'' + 2 (L'/L) phi' + sin(phi)/L + 2 Gamma phi' + B L phi'|phi'| = 0,
        L(tau) = 1 + eps sin(a tau),  a = 2 + delta.
    """
    phi, v = y
    a    = 2.0 + delta
    L    = 1.0 + eps * np.sin(a * tau)
    Ldot = eps * a * np.cos(a * tau)
    dphi = v
    dv   = (
        -2.0 * (Ldot / L) * v
        - np.sin(phi) / L
        - 2.0 * Gamma * v
        - B * L * v * abs(v)
    )
    return [dphi, dv]


# ── анализ одного delta ──────────────────────────────────────────────────
def analyze_delta(delta, eps=eps,
                  N_cycles=1000,
                  m_check=10,
                  m_amp=10,
                  pts_per_period=1000,
                  eta_rel=1e-4,
                  eta_amp=1e-3,
                  tol_A=0.05,
                  y0=(0.1, 0.0)):
    """
    Возвращает словарь с критериями устойчивого предельного цикла
    при данной расстройке delta.

    Параметры:
      N_cycles        — длительность интегрирования в периодах T_c
      m_check         — число последних переходов для проверки Пуанкаре
      m_amp           — число последних периодов для оценки амплитуды
      pts_per_period  — число точек dense_output на одном периоде T_c
      eta_rel         — порог относительной невязки Пуанкаре
      eta_amp         — порог относительного разброса амплитуд
      tol_A           — допуск близости к A_target_m, м
      y0              — начальные условия (phi(0), v(0))
    """
    a    = 2.0 + delta
    T_L  = 2.0 * np.pi / a    # период подкачки длины
    T_c  = 2.0 * T_L          # период предельного цикла маятника
    tau_end = N_cycles * T_c

    # длительная интеграция с плотным выводом для произвольной выборки
    sol = solve_ivp(
        lambda tau, y: rhs(tau, y, eps, delta),
        [0.0, tau_end],
        list(y0),
        method='DOP853',
        rtol=1e-9, atol=1e-11,
        dense_output=True,
    )

    # ── 1. отображение Пуанкаре на периоде T_c ────────────────────────────
    # снимки полного фазового вектора в моменты tau_k = k*T_c
    k_vals = np.arange(N_cycles - m_check, N_cycles + 1)
    tau_k  = k_vals * T_c
    y_k    = sol.sol(tau_k)                       # форма (2, m_check+1)

    diffs = y_k[:, 1:] - y_k[:, :-1]              # переходы
    refs  = np.linalg.norm(y_k[:, :-1], axis=0) + 1e-12
    e_k   = np.linalg.norm(diffs, axis=0) / refs
    poincare_error_max = float(e_k.max())
    periodic_ok = poincare_error_max < eta_rel

    # ── 2. амплитуда как полуразмах phi на последних m_amp периодах ──────
    A_k_m = np.empty(m_amp)
    for j in range(m_amp):
        tau_a = (N_cycles - m_amp + j)     * T_c
        tau_b = (N_cycles - m_amp + j + 1) * T_c
        tau_grid = np.linspace(tau_a, tau_b, pts_per_period)
        phi_grid = sol.sol(tau_grid)[0]
        A_rad   = 0.5 * (phi_grid.max() - phi_grid.min())
        A_k_m[j] = L0 * A_rad

    A_mean_m = float(A_k_m.mean())
    A_std_m  = float(A_k_m.std())

    if A_mean_m > 1e-12:
        amp_rel_variation = float(np.max(np.abs(A_k_m - A_mean_m)) / A_mean_m)
    else:
        amp_rel_variation = float('inf')
    amplitude_stable_ok = amp_rel_variation < eta_amp

    # ── 3. близость к целевой амплитуде ──────────────────────────────────
    near_target_ok = abs(A_mean_m - A_target_m) < tol_A

    accepted = bool(periodic_ok and amplitude_stable_ok and near_target_ok)

    return {
        'delta':                delta,
        'a':                    a,
        'T_L':                  T_L,
        'T_c':                  T_c,
        'A_mean_m':             A_mean_m,
        'A_std_m':              A_std_m,
        'poincare_error_max':   poincare_error_max,
        'amp_rel_variation':    amp_rel_variation,
        'periodic_ok':          periodic_ok,
        'amplitude_stable_ok':  amplitude_stable_ok,
        'near_target_ok':       near_target_ok,
        'accepted':             accepted,
    }


# ── сканирование сетки delta ─────────────────────────────────────────────
def scan_deltas(delta_grid, eps=eps, verbose=True, **kwargs):
    """Прогон analyze_delta по всей сетке. Возвращает список результатов."""
    results = []
    n = len(delta_grid)
    for i, d in enumerate(delta_grid):
        res = analyze_delta(d, eps=eps, **kwargs)
        results.append(res)
        if verbose and (i + 1) % max(1, n // 20) == 0:
            print(f'  [{i+1:>4}/{n}]  delta={d:+.5f}  '
                  f'A={res["A_mean_m"]:.4f}  acc={int(res["accepted"])}')
    return results


# ── группировка accepted=True в смежные интервалы ────────────────────────
def group_intervals(deltas_accepted):
    """
    Из отсортированного списка принятых delta возвращает список
    (delta_left, delta_right) — смежных интервалов.
    Разрыв в сетке более 1.5 шагов разделяет интервалы.
    """
    if len(deltas_accepted) == 0:
        return []
    deltas = np.sort(np.asarray(deltas_accepted))
    if len(deltas) == 1:
        return [(float(deltas[0]), float(deltas[0]))]
    diffs  = np.diff(deltas)
    step   = np.median(diffs)
    breaks = np.where(diffs > 1.5 * step)[0]

    intervals = []
    start = 0
    for b in breaks:
        intervals.append((float(deltas[start]), float(deltas[b])))
        start = b + 1
    intervals.append((float(deltas[start]), float(deltas[-1])))
    return intervals


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('═' * 60)
    print(f'Параметры:  Gamma = {Gamma:.6f}    B = {B:.6f}    eps = {eps}')
    print('═' * 60)

    delta_grid = np.linspace(-0.02, 0.02, 400)
    print(f'Сетка: delta ∈ [{delta_grid[0]:+.3f}, {delta_grid[-1]:+.3f}], '
          f'N = {len(delta_grid)}, шаг {delta_grid[1]-delta_grid[0]:.5f}')
    print()

    results = scan_deltas(delta_grid)

    # ── подробная таблица ────────────────────────────────────────────────
    print()
    header = (f"{'delta':>9} {'A_mean,м':>10} {'A_std,м':>10} "
              f"{'poiErr':>10} {'ampVar':>10} {'P':>2} {'S':>2} {'N':>2} {'ACC':>4}")
    print(header)
    print('-' * len(header))
    for r in results:
        print(f"{r['delta']:+9.5f} "
              f"{r['A_mean_m']:10.4f} "
              f"{r['A_std_m']:10.2e} "
              f"{r['poincare_error_max']:10.2e} "
              f"{r['amp_rel_variation']:10.2e} "
              f"{int(r['periodic_ok']):>2} "
              f"{int(r['amplitude_stable_ok']):>2} "
              f"{int(r['near_target_ok']):>2} "
              f"{int(r['accepted']):>4}")

    # ── интервалы accepted=True ──────────────────────────────────────────
    accepted_deltas = [r['delta'] for r in results if r['accepted']]
    intervals = group_intervals(accepted_deltas)

    print()
    print('═' * 60)
    print(f'Найдено интервалов с предельным циклом ~ {A_target_m} м: {len(intervals)}')
    for left, right in intervals:
        print(f'  delta ∈ [{left:+.5f},  {right:+.5f}]   ширина {right - left:.5f}')

    A_max = max(r['A_mean_m'] for r in results)
    print()
    print(f'Максимальная средняя амплитуда по сетке: {A_max:.4f} м')
    print('═' * 60)

    # ── графики ──────────────────────────────────────────────────────────
    deltas = np.array([r['delta']               for r in results])
    Ameans = np.array([r['A_mean_m']            for r in results])
    pErrs  = np.array([r['poincare_error_max']  for r in results])
    accept = np.array([r['accepted']            for r in results])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax = axes[0]
    ax.plot(deltas, Ameans, 'b-', lw=1)
    ax.plot(deltas[accept], Ameans[accept], 'go', ms=4, label='Соответствует критерию')
    ax.axhline(A_target_m, color='r', ls='--', label=f'цель {A_target_m} м')
    ax.axhspan(A_target_m - 0.05, A_target_m + 0.05, alpha=0.15, color='r',
               label='допуск ±0.05 м')
    ax.set_ylabel('Средняя амплитуда, м')
    ax.set_title(f'Поиск устойчивого предельного цикла  ($\\varepsilon = {eps}$)')
    ax.legend(loc='lower center', ncol=3)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(deltas, pErrs, 'b-')
    ax.axhline(1e-4, color='r', ls='--', label='порог $\\eta_{rel}=10^{-4}$')
    ax.set_xlabel('$\\delta$')
    ax.set_ylabel('Ошибка периодичности за 10 последних периодов')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('out/limit_cycle_search.png', dpi=150, bbox_inches='tight')
    plt.show()
