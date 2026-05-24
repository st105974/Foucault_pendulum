"""
Численное интегрирование сферического маятника переменной длины L(t)
с учётом параметрической подкачки, линейного трения и
кориолисовых эффектов вращения Земли --- по системе уравнений,
выведенных в vkr.tex (раздел «Оценка влияния нелинейных эффектов...»).

Координаты (rho, phi, theta) -- сферические, локальный базис правой
тройки (x, y, z) = (вверх, на север, на запад):
    rho   = L(t)        -- длина нити;
    phi               -- полярный угол отклонения нити от вертикали -e_x;
    theta             -- азимут проекции груза в плоскости Oyz, от +e_y.

Положение груза:
    x = -L cos(phi)
    y =  L sin(phi) cos(theta)
    z =  L sin(phi) sin(theta)

Уравнения движения (после деления на m L^2):

    phi_ddot = -2 (L_dot/L + gamma) phi_dot
               + sin(phi) cos(phi) theta_dot^2
               - (g/L) sin(phi)
               - 2 Omega_E (L_dot/L) cos(lambda) sin(theta)
               + 2 Omega_E sin(phi) (sin(lambda) cos(phi)
                                     - cos(lambda) sin(phi) cos(theta)) theta_dot

    theta_ddot = -2 (L_dot/L + gamma) theta_dot
                 - 2 cot(phi) phi_dot theta_dot
                 - 2 Omega_E (sin(lambda) cot(phi)
                              - cos(lambda) cos(theta)) phi_dot
                 - 2 Omega_E (L_dot/L) (sin(lambda)
                              + cos(lambda) cot(phi) cos(theta))

Замечание: cot(phi) сингулярен при phi -> 0; это координатная особенность
сферического базиса в точке вертикали. Для эллиптического орбитального
движения phi не достигает нуля, поэтому интегрирование устойчиво.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ============================================================
# Параметры маятника (значения те же, что в double_dim_pen.py)
# ============================================================

omega0  = 0.952              # рад/с
epsilon = 0.00325            # амплитуда модуляции длины

g  = 9.81                   # м/с^2
l0 = g / omega0**2          # м -- средняя длина подвеса

m       = 1.5             # кг (15 г)
beta    = 0.0046           # 1/с -- коэффициент линейного сопротивления (2 gamma = beta)
gamma   = beta / 2
damping = beta              # для удобства совпадает с 2 gamma


# ============================================================
# Вращение Земли, широта 60° N
# ============================================================

Omega_E      = 7.2921159e-5  # рад/с
latitude_deg = 60.0
latitude     = np.deg2rad(latitude_deg)

sin_lambda = np.sin(latitude)
cos_lambda = np.cos(latitude)

print(f"l0 = {l0:.6f} м")
print(f"beta = {beta:.6f} 1/с,  gamma = {gamma:.6f} 1/с")
print(f"Omega_E sin(lambda) = {Omega_E * sin_lambda:.9e} рад/с  (вертикальная составляющая)")
print(f"Omega_E cos(lambda) = {Omega_E * cos_lambda:.9e} рад/с  (горизонтальная составляющая)")


# ============================================================
# Закон изменения длины и её логарифмическая производная
#   L(t)        = l0 (1 + epsilon sin(2 omega0 t))
#   lambda_l(t) = L_dot / L
# ============================================================

def length(t):
    return l0 * (1.0 + epsilon * np.sin(2.0 * omega0 * t))


def lambda_l(t):
    return (
        2.0 * epsilon * omega0 * np.cos(2.0 * omega0 * t)
        / (1.0 + epsilon * np.sin(2.0 * omega0 * t))
    )


# ============================================================
# Правая часть ОДУ
# Y = [phi, theta, phi_dot, theta_dot]
# ============================================================

def rhs(t, Y):
    phi, theta, phi_dot, theta_dot = Y

    ell = length(t)
    lam = lambda_l(t)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_th  = np.sin(theta)
    cos_th  = np.cos(theta)

    # Защита от особенности в окрестности phi = 0.
    if abs(sin_phi) < 1e-10:
        sin_phi = 1e-10 if sin_phi >= 0 else -1e-10
    cot_phi = cos_phi / sin_phi

    # ------------------------------------------------------------
    # Уравнение для phi
    # ------------------------------------------------------------
    phi_ddot = (
        - 2.0 * (lam + gamma) * phi_dot
        + sin_phi * cos_phi * theta_dot**2
        - (g / ell) * sin_phi
        - 2.0 * Omega_E * lam * cos_lambda * sin_th
        + 2.0 * Omega_E * sin_phi
            * (sin_lambda * cos_phi - cos_lambda * sin_phi * cos_th)
            * theta_dot
    )

    # ------------------------------------------------------------
    # Уравнение для theta
    # ------------------------------------------------------------
    theta_ddot = (
        - 2.0 * (lam + gamma) * theta_dot
        - 2.0 * cot_phi * phi_dot * theta_dot
        - 2.0 * Omega_E * (sin_lambda * cot_phi - cos_lambda * cos_th) * phi_dot
        - 2.0 * Omega_E * lam * (sin_lambda + cos_lambda * cot_phi * cos_th)
    )

    return np.array([phi_dot, theta_dot, phi_ddot, theta_ddot])


# ============================================================
# Начальные условия (по аналогии с double_dim_pen.py)
#
# Малое отклонение от равновесия phi = 0 (вертикали).
# Перпендикулярная начальная скорость v_perp придаёт эллиптичность
# и удерживает phi > 0, обходя сферическую особенность в нуле.
# ============================================================

delta_phi0 = 0.1           # рад -- отклонение нити от вертикали
phi0       = delta_phi0
theta0     = 0.0            # начальная азимутальная плоскость -- xy

phi_dot0   = 0.0            # рад/с

v_perp0      = 0.2         # м/с -- касательная (азимутальная) скорость
ell_initial  = length(0.0)
theta_dot0   = v_perp0 / (ell_initial * np.sin(phi0))

Y0 = np.array([phi0, theta0, phi_dot0, theta_dot0])

print(f"phi0       = {phi0:.6f} рад")
print(f"theta0     = {theta0:.6f} рад")
print(f"phi_dot0   = {phi_dot0:.6f} рад/с")
print(f"v_perp0    = {v_perp0:.6f} м/с")
print(f"theta_dot0 = {theta_dot0:.6f} рад/с")


# ============================================================
# Интегрирование
# ============================================================

t_start = 0.0
t_end   = 2 * 3600        # 2 часа

n_points = 200_000
t_eval   = np.linspace(t_start, t_end, n_points)

sol = solve_ivp(
    rhs,
    (t_start, t_end),
    Y0,
    t_eval=t_eval,
    method="DOP853",
    rtol=1e-9,
    atol=1e-11,
    max_step=0.02,
)

if not sol.success:
    print("Интегрирование завершилось с ошибкой:")
    print(sol.message)


# ============================================================
# Извлечение решения
# ============================================================

t         = sol.t
phi       = sol.y[0]
theta     = sol.y[1]
phi_dot   = sol.y[2]
theta_dot = sol.y[3]

ell_values = length(t)

# Декартовы координаты груза:
# x -- вертикаль (отрицательная, потому что груз ниже подвеса),
# y -- на север, z -- на запад.
x = -ell_values * np.cos(phi)
y =  ell_values * np.sin(phi) * np.cos(theta)
z =  ell_values * np.sin(phi) * np.sin(theta)


# ============================================================
# Графики: проекция траектории на плоскость yz на трёх
# временных интервалах. Сохраняем PNG-файлы для вставки в работу.
# ============================================================

theta_unwrapped = np.unwrap(theta)

time_intervals = [
    (0,     300),
    (300,   1000),
    (1000,  7200),
]

# Файлы складываем рядом со скриптом (в той же папке out/),
# имена совпадают с теми, что подключаются в vkr.tex.
yz_files = [
    "yz_trajectory_0_300.png",
    "yz_trajectory_300_1000.png",
    "yz_trajectory_1000_7200.png",
]

# Общие пределы по осям, чтобы все три PNG получались одинаковых пропорций
yz_extent = 1.1 * max(np.max(np.abs(y)), np.max(np.abs(z)))

for (t1, t2), out_name in zip(time_intervals, yz_files):
    mask = (t >= t1) & (t <= t2)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(y[mask], z[mask], linewidth=1.0, color="tab:blue")
    ax.scatter(y[mask][0],  z[mask][0],  s=35, color="tab:green",
               label=f"начало ({t1:.0f} с)", zorder=3)
    ax.scatter(y[mask][-1], z[mask][-1], s=35, color="tab:red",
               label=f"конец ({t2:.0f} с)", zorder=3)
    ax.axhline(0.0, color="lightgrey", linewidth=0.7)
    ax.axvline(0.0, color="lightgrey", linewidth=0.7)
    ax.set_xlabel("y (север), м")
    ax.set_ylabel("z (запад), м")
    ax.set_title(rf"Проекция траектории на $Oyz$, {t1:.0f}--{t2:.0f} с")
    ax.set_xlim(-yz_extent, yz_extent)
    ax.set_ylim(-yz_extent, yz_extent)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_name, dpi=150)
    print(f"Сохранён график: {out_name}")
    plt.show()


# ============================================================
# Анализ скорости поворота плоскости качаний:
#   1) численная оценка из траектории (положение большой полуоси
#      эллипса в каждом «качании»);
#   2) аналитический вклад эффекта Фуко: Omega_F = -Omega_E sin(lambda)
#      (минус, т.к. в северном полушарии прецессия CW при взгляде с +x,
#      а наш азимут theta = atan2(z, y) растёт CCW);
#   3) аналитический вклад прецессии Эйри (эллиптичность траектории):
#      Omega_A = (3/8) omega_0 * a * b / L^2,
#      где a, b -- большая и малая полуоси эллипса.
# ============================================================

# 1) Численно: находим локальные максимумы |r|^2 в плоскости yz
#    (моменты, когда груз пересекает большую ось эллипса) и следим
#    за её ориентацией во времени.
r2 = y**2 + z**2
peaks_mask = (r2[1:-1] > r2[:-2]) & (r2[1:-1] > r2[2:])
peaks_idx  = np.where(peaks_mask)[0] + 1

t_peaks = t[peaks_idx]
y_peaks = y[peaks_idx]
z_peaks = z[peaks_idx]
r_peaks = np.sqrt(r2[peaks_idx])

# Ориентация большой полуоси (период pi, поэтому удваиваем угол)
angle_peaks    = np.arctan2(z_peaks, y_peaks)
angle_peaks_un = np.unwrap(2.0 * angle_peaks) / 2.0

# Аппроксимация на разогретом участке (исключаем первые ~10 c)
fit_lo, fit_hi = 50.0, min(2000.0, t_end)
fit_mask = (t_peaks >= fit_lo) & (t_peaks <= fit_hi)

if np.sum(fit_mask) >= 3:
    slope_sim, _ = np.polyfit(t_peaks[fit_mask], angle_peaks_un[fit_mask], 1)
else:
    slope_sim = np.nan

# 2) Аналитический Фуко (знак отрицательный в нашей CCW-конвенции)
Omega_F = -Omega_E * sin_lambda

# 3) Аналитический Эйри (положительный для CCW-орбиты, что соответствует
#    нашим начальным условиям theta_dot0 > 0)
a_init  = ell_initial * np.sin(phi0)            # большая полуось (м)
b_init  = v_perp0 / omega0                       # малая полуось (м)
Omega_A = (3.0 / 8.0) * omega0 * (a_init * b_init) / ell_initial**2

# Также оценим Эйри с использованием амплитуд, усреднённых по интервалу
# фитирования (учитывает затухание / подкачку):
if np.sum(fit_mask) >= 3:
    a_avg = np.mean(r_peaks[fit_mask])
    # минимум |r| в окрестности каждого пика (грубая оценка малой полуоси)
    r_all   = np.sqrt(r2)
    troughs_mask = (r_all[1:-1] < r_all[:-2]) & (r_all[1:-1] < r_all[2:])
    troughs_idx  = np.where(troughs_mask)[0] + 1
    t_tr = t[troughs_idx]
    r_tr = r_all[troughs_idx]
    tr_mask = (t_tr >= fit_lo) & (t_tr <= fit_hi)
    b_avg = np.mean(r_tr[tr_mask]) if np.sum(tr_mask) >= 3 else b_init
    Omega_A_avg = (3.0 / 8.0) * omega0 * (a_avg * b_avg) / ell_initial**2
else:
    a_avg = b_avg = np.nan
    Omega_A_avg = np.nan

print()
print("=" * 78)
print("Скорость поворота плоскости качаний (положительное -- CCW при взгляде с +x):")
print("=" * 78)
print(f"  Из симуляции (фит на {fit_lo:.0f}-{fit_hi:.0f} c):")
print(f"     Omega_sim       = {slope_sim:+.4e} рад/с  "
      f"= {np.rad2deg(slope_sim)*3600:+.3f} °/час")
print()
print("  Аналитика (начальные амплитуды a = L sin(phi0), b = v_perp/omega0):")
print(f"     a = {a_init:.4f} м,  b = {b_init:.4f} м")
print(f"     Omega_F (Фуко)  = {Omega_F:+.4e} рад/с  "
      f"= {np.rad2deg(Omega_F)*3600:+.3f} °/час")
print(f"     Omega_A (Эйри)  = {Omega_A:+.4e} рад/с  "
      f"= {np.rad2deg(Omega_A)*3600:+.3f} °/час")
print(f"     Сумма F + A     = {Omega_F + Omega_A:+.4e} рад/с  "
      f"= {np.rad2deg(Omega_F + Omega_A)*3600:+.3f} °/час")
print()
if not np.isnan(Omega_A_avg):
    print("  Аналитика (амплитуды a, b усреднены по интервалу фита):")
    print(f"     a_avg = {a_avg:.4f} м,  b_avg = {b_avg:.4f} м")
    print(f"     Omega_A_avg     = {Omega_A_avg:+.4e} рад/с  "
          f"= {np.rad2deg(Omega_A_avg)*3600:+.3f} °/час")
    print(f"     Сумма F + A_avg = {Omega_F + Omega_A_avg:+.4e} рад/с  "
          f"= {np.rad2deg(Omega_F + Omega_A_avg)*3600:+.3f} °/час")
print("=" * 78)


# ------------------------------------------------------------
# Мгновенная скорость прецессии плоскости качаний.
# Для каждого пика |r| оцениваем локальный наклон ориентации
# большой полуоси скользящей линейной регрессией по соседним пикам,
# а затем строим этот наклон как функцию времени и сравниваем
# с горизонтальной прямой Omega_F (чистый эффект Фуко).
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

if len(t_peaks) >= 5:
    # Скользящая линейная регрессия в окне из N_window соседних пиков.
    N_window = 41
    half_w   = N_window // 2

    precession_rate = np.full(len(t_peaks), np.nan)
    for i in range(len(t_peaks)):
        i_lo = max(0, i - half_w)
        i_hi = min(len(t_peaks), i + half_w + 1)
        if i_hi - i_lo >= 3:
            slope_i, _ = np.polyfit(t_peaks[i_lo:i_hi], angle_peaks_un[i_lo:i_hi], 1)
            precession_rate[i] = slope_i

    # Переводим в °/час для наглядности
    rate_deg_per_hour = np.rad2deg(precession_rate) * 3600.0

    ax.plot(t_peaks, rate_deg_per_hour,
            color="tab:blue", linewidth=1.2,
            label=r"оценка $\dot\alpha(t)$ по скользящей регрессии")

    # ----------------------------------------------------------------
    # Мгновенная аналитическая оценка скорости Эйри-прецессии:
    # Omega_A(t) = (3/8) omega_0 * a(t) * b(t) / L^2,
    # где a(t) -- огибающая по пикам |r|, b(t) -- огибающая по минимумам |r|.
    # ----------------------------------------------------------------
    r_all        = np.sqrt(r2)
    troughs_mask = (r_all[1:-1] < r_all[:-2]) & (r_all[1:-1] < r_all[2:])
    troughs_idx  = np.where(troughs_mask)[0] + 1
    t_tr = t[troughs_idx]
    r_tr = r_all[troughs_idx]

    # Интерполируем b(t) в моменты пиков, чтобы a и b были на общем t_peaks
    b_at_peaks = np.interp(t_peaks, t_tr, r_tr) if len(t_tr) >= 2 else \
                 np.full_like(t_peaks, b_init)

    Omega_A_t  = (3.0 / 8.0) * omega0 * r_peaks * b_at_peaks / ell_initial**2
    Omega_A_t_deg_per_hour = np.rad2deg(Omega_A_t) * 3600.0

    ax.plot(t_peaks, Omega_A_t_deg_per_hour,
            color="tab:green", linewidth=1.2, linestyle="-",
            label=r"эллиптическая прецессия $\Omega_A(t)=\frac{3}{8}\omega_0\,a(t)\,b(t)/L^2$")

# Горизонтальная прямая -- чистая фуковская скорость
Omega_F_deg_per_hour = np.rad2deg(Omega_F) * 3600.0
ax.axhline(Omega_F_deg_per_hour, color="tab:red", linewidth=1.5, linestyle="--",
           label=rf"скорость Фуко-прецессии $\Omega_F = {Omega_F_deg_per_hour:+.2f}$ °/час")

ax.axhline(0.0, color="lightgrey", linewidth=0.7)
ax.set_xlabel("t, с")
ax.set_ylabel(r"скорость прецессии, °/час")
ax.set_title("Сравнение угловых скоростей прецессии плоскости")
ax.grid(True)
ax.legend(loc="best")
plt.tight_layout()
fig.savefig("precession_contributions.png", dpi=150, bbox_inches="tight")
print("Сохранён график: precession_contributions.png")
plt.show()


# ============================================================
# Ускоренная анимация эволюции траектории на плоскости Oyz:
#   7200 с физического времени  ->  30 с видео.
# Сохраняется одновременно в GIF (для встройки в vkr_pres.tex
# с запуском по клику) и в MP4 (для воспроизведения в плеере).
# ============================================================

import matplotlib.animation as animation

anim_duration_sec = 30.0                              # длительность ролика, с
anim_fps          = 25                                # кадров/с в выходном файле
n_anim_frames     = int(anim_duration_sec * anim_fps) # = 750 кадров
speedup_factor    = t_end / anim_duration_sec         # = 7200/30 = 240

# Физические моменты, соответствующие каждому кадру
frame_phys_times = np.linspace(t_start, t_end, n_anim_frames)
# Индексы ближайших точек интегрирования
frame_indices = np.searchsorted(t, frame_phys_times, side="right") - 1
frame_indices = np.clip(frame_indices, 0, len(t) - 1)

trail_window_sec = 50.0  # длина видимого «хвоста» траектории, с

fig_anim, ax_anim = plt.subplots(figsize=(6.5, 6.5))
extent = 1.05 * max(np.max(np.abs(y)), np.max(np.abs(z)))
ax_anim.set_xlim(-extent, extent)
ax_anim.set_ylim(-extent, extent)
ax_anim.set_aspect("equal")
ax_anim.set_xlabel("y (север), м")
ax_anim.set_ylabel("z (запад), м")
ax_anim.axhline(0.0, color="lightgrey", linewidth=0.7)
ax_anim.axvline(0.0, color="lightgrey", linewidth=0.7)
ax_anim.grid(True)
ax_anim.set_title(
    f"Эволюция траектории маятника Фуко (ускорено ×{int(round(speedup_factor))})"
)

(line_trail,) = ax_anim.plot([], [], linewidth=1.0, color="tab:blue", alpha=0.7)
(point_now,)  = ax_anim.plot([], [], "o", color="tab:red", markersize=7)
time_text = ax_anim.text(
    0.02, 0.96, "", transform=ax_anim.transAxes, fontsize=10,
    verticalalignment="top"
)


def anim_init():
    line_trail.set_data([], [])
    point_now.set_data([], [])
    time_text.set_text("")
    return line_trail, point_now, time_text


def anim_update(frame):
    end_t   = frame_phys_times[frame]
    start_t = max(t_start, end_t - trail_window_sec)
    mask = (t >= start_t) & (t <= end_t)
    line_trail.set_data(y[mask], z[mask])
    idx = frame_indices[frame]
    point_now.set_data([y[idx]], [z[idx]])
    time_text.set_text(f"t = {end_t:7.1f} с / {t_end:.0f} с")
    return line_trail, point_now, time_text


anim = animation.FuncAnimation(
    fig_anim, anim_update, init_func=anim_init,
    frames=n_anim_frames, interval=1000.0 / anim_fps, blit=True
)

# Сохраняем GIF (Pillow входит в зависимости matplotlib, внешний софт не нужен)
gif_path = "trajectory_evolution.gif"
anim.save(gif_path, writer=animation.PillowWriter(fps=anim_fps), dpi=100)
print(
    f"Сохранён GIF: {gif_path} "
    f"(длительность {anim_duration_sec:.0f} с, ускорение ×{int(round(speedup_factor))})"
)

# Сохраняем MP4 (нужен ffmpeg в PATH)
mp4_path = "trajectory_evolution.mp4"
try:
    mp4_writer = animation.FFMpegWriter(
        fps=anim_fps, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"]   # совместимо с Windows Media Player
    )
    anim.save(mp4_path, writer=mp4_writer, dpi=100)
    print(f"Сохранён MP4: {mp4_path}")
except Exception as exc:
    print(f"Не удалось сохранить MP4: {exc}")
    print(
        "Для MP4 нужен внешний ffmpeg в PATH "
        "(macOS: brew install ffmpeg; Ubuntu: apt install ffmpeg; "
        "Windows: choco install ffmpeg)."
    )

plt.close(fig_anim)
