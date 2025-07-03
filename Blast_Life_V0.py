"""
scalvy_hybrid_pipeline.py
-------------------------
• Fit BLAST-Lite NMC-811 model to Ramadass 45 °C & 55 °C data
• Run a 60-s-step hybrid thermal + ageing simulation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from blast import models
import time, psutil, os
print("PID =", os.getpid())            # shows which process to watch
start = time.time()

# --------------------------------------------------------------------
# helper functions
# --------------------------------------------------------------------
def ramadass_profile(q_ah: float, n_cycles: int,
                     temp_c: float, dt_s: int = 60) -> pd.DataFrame:
    """
    3-h square-wave cycle:
      1 h discharge (+q_ah A)  → 10 min rest (0 A)
      1 h charge    (-q_ah A)  → 10 min rest (0 A)
    Returns DataFrame(Time_s, SOC, Temperature_C)
    """
    segs = [(3600,  q_ah),
            (600,   0.0),
            (3600, -q_ah),
            (600,   0.0)]

    rows, t_clock, soc = [], 0.0, 1.0
    for _ in range(n_cycles):
        for secs, cur in segs:
            for k in range(0, secs, dt_s):
                # Coulomb count (Ah → SOC)
                soc -= cur * dt_s / (q_ah * 3600)
                soc = np.clip(soc, 0, 1)
                rows.append((t_clock, soc, temp_c))
                t_clock += dt_s

    return pd.DataFrame(rows,
                        columns=["Time_s", "SOC", "Temperature_C"])


def fit_scalar(temp_c: int, df_exp: pd.DataFrame,
               profile: pd.DataFrame, cell_cls) -> float:
    """
    Tune degradation_scalar so BLAST-Lite capacity vs. time matches
    experimental curve (minimise 1 – R²).
    """
    y_true = (df_exp.capacity_ah / df_exp.capacity_ah.iloc[0]).values
    t_exp_h = df_exp.time_s.values / 3600

    def loss(scale_nd):
        scale = scale_nd[0]
        cell = cell_cls(degradation_scalar=scale)
        cell.simulate_battery_life(profile)
        y_pred = cell.outputs['q']
        t_pred_h = cell.stressors['t_days'] * 24
        n = min(len(y_true), len(y_pred))
        return 1 - r2_score(y_true[:n], y_pred[:n])

    res = minimize(loss, x0=[1.0], bounds=[(0.2, 5)], tol=1e-3)
    print(f"{temp_c:>3} °C  scalar={res.x[0]:.3f}   R²={1-res.fun:.3f}")
    return float(res.x[0])


# --------------------------------------------------------------------
# 1.  load digitised Ramadass curves  (EDIT these two paths)
# --------------------------------------------------------------------
df55 = pd.read_csv(Path("ramadass_55C.csv"), names=["cycle", "capacity_ah"])
df45 = pd.read_csv(Path("ramadass_45C.csv"), names=["cycle", "capacity_ah"])
# convert text → numeric, drop any row that failed the conversion
for df in (df55, df45):
    df["cycle"]       = pd.to_numeric(df["cycle"],       errors="coerce")
    df["capacity_ah"] = pd.to_numeric(df["capacity_ah"], errors="coerce")
    df.dropna(inplace=True)

cycle_h = 3.0
df55["time_s"] = df55.cycle * cycle_h * 3600
df45["time_s"] = df45.cycle * cycle_h * 3600

# --------------------------------------------------------------------
# 2.  build matching SOC-swing profiles
# --------------------------------------------------------------------
Q_NOM = 2.0                       # Ah of LG MJ1 cell
current_1C = Q_NOM
n55 = int(df55.time_s.iloc[-1] / (3 * 3600))
n45 = int(df45.time_s.iloc[-1] / (3 * 3600))

profile55 = ramadass_profile(current_1C, n55, 55)
profile45 = ramadass_profile(current_1C, n45, 45)

# --------------------------------------------------------------------
# 3.  optimise degradation_scalar
# --------------------------------------------------------------------
Cell = models.Nmc811_GrSi_LGMJ1_4Ah_Battery
scalar55 = fit_scalar(55, df55, profile55, Cell)
scalar45 = fit_scalar(45, df45, profile45, Cell)
# A scalar of 1.0 is the baseline, and we expect 45C to age slower than 55C.
print("\n!!! WARNING: Manually overriding scalars for diagnostic purposes. !!!")
scalar45 = 0.8
scalar55 = 1.0
# --- End of Diagnostic Step ---
def scalar_vs_T(T_c: float) -> float:
    # Define the known temperature and scalar points
    known_temps = [45, 55]
    known_scalars = [scalar45, scalar55]

    # Use numpy.interp, but prevent extrapolation by clamping the output.
    # If T_c is below 45, it will be clamped to scalar45.
    # If T_c is above 55, it will be clamped to scalar55.
    return np.interp(T_c, known_temps, known_scalars, left=scalar45, right=scalar55)

# --------------------------------------------------------------------
# 4.  hybrid thermal + ageing run
# --------------------------------------------------------------------
P = dict(m=0.045, Cp=1040, R0=0.030, Rf=1e-4, hA=3.5)

ambient_c = 30
# Use the correct 4.0 Ah capacity from the model
cell = Cell(degradation_scalar=scalar_vs_T(ambient_c))
Q_NOM = cell.cap  # Get nominal capacity directly from the model (4.0 Ah)

T_k = ambient_c + 273.15  # Cell temperature in Kelvin
R = P["R0"]
soc = 1.0
dt_s = 60
SIM_H = 600
steps = int(SIM_H * 3600 / dt_s)
clock_s = 0.0

log_t, log_T, log_Q, log_R = [], [], [], []

for _ in range(steps):
    # Store state at the beginning of the step
    soc_prev = soc
    T_k_prev = T_k
    clock_s_prev = clock_s

    clock_s += dt_s

    # Determine the correct current (I) for this specific step in the cycle
    Q_now = cell.cap * cell.outputs['q'][-1]
    I_dis = 1.0 * Q_now
    I_cha = -I_dis

    τ = clock_s_prev % 10800  # 3-h square wave (10800 s)
    if τ < 3600:  # 0s to 3600s (discharge)
        I = I_dis
    elif τ < 4200:  # 3600s to 4200s (rest)
        I = 0.0
    elif τ < 7800:  # 4200s to 7800s (charge)
        I = I_cha
    else:  # 7800s to 8400s (rest)
        I = 0.0

    # 1. Thermal Step (using the correct current I for this interval)
    dTdt = (-P["hA"] * (T_k - (ambient_c + 273.15)) + I ** 2 * (R + P["Rf"])) \
           / (P["m"] * P["Cp"])
    T_k += dTdt * dt_s

    # 2. Coulomb Counting Step (to find the new SOC)
    # Use the correct nominal capacity, Q_NOM
    soc -= I * dt_s / (Q_NOM * 3600)
    soc = np.clip(soc, 0, 1)

    # 3. Ageing Step
    # Pass the states from the start (prev) and end of the interval
    T_c_prev = T_k_prev - 273.15
    T_c_now = T_k - 273.15
    cell.update_battery_state(np.array([clock_s_prev, clock_s]),
                              np.array([soc_prev, soc]),
                              np.array([T_c_prev, T_c_now]))

    # Update resistance for the next step based on the new aged state
    Q_frac = cell.outputs['q'][-1]
    if 'r' in cell.outputs:
        R = P["R0"] * cell.outputs['r'][-1]

    # Log results
    log_t.append(clock_s / 3600)
    log_T.append(T_k - 273.15)
    log_Q.append(Q_frac * 100)
    log_R.append(R * 1000)

print("\nHybrid simulation done.")

# --------------------------------------------------------------------
# 5.  plots
# --------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].plot(log_t, log_T); ax[0].set(xlabel="Time [h]", ylabel="Cell T [°C]")
ax[1].plot(log_t, log_Q); ax[1].set(xlabel="Time [h]", ylabel="Capacity [%]")
ax[2].plot(log_t, log_R); ax[2].set(xlabel="Time [h]", ylabel="DCIR [mΩ]")
fig.suptitle(f"Hybrid model – ambient {ambient_c} °C, "
             f"scalar {scalar_vs_T(ambient_c):.2f}")
plt.tight_layout(); plt.show()
