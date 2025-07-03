import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from blast import models

# ---- 1.  BLAST-Lite cell object, already fitted ----
cell = models.Nmc811_GrSi_LGMJ1_4Ah_Battery()
# (Assume you ran cell.fit(df25, temperature=25) and 45 °C earlier)

# ---- 2.  Physical + thermal parameters ----
p = dict(
    m_cell      = 0.045,        # kg  <-- weigh & update
    Cp_cell     = 1040,         # J kg-1 K-1
    R_cell_init = 0.030,        # Ω   <-- pulse test later
    R_contact   = 1e-4,         # Ω   small, leave as-is
    hA          = 3.5,          # W K-1  adjust after 1 temp trace
    T_inf       = 298.0         # K   ambient
)

# ---- 3.  Current profile: simple 1 C CC-rest-1 C CC loop ----
Q_nom  = 4.0        # Ah (MJ1 cell)
I_dis  = 4.0        # A (1 C discharge)
I_cha  = -4.0       # A (-1 C charge)
rest_s = 600        # 10-min rest

def I_profile(t):
    """Three-hour cycle: 1 h discharge, 10 min rest, 1 h charge, 10 min rest"""
    τ = t % 10800           # 3 h = 10800 s
    if   τ < 3600:   return I_dis
    elif τ < 4200:   return 0.0
    elif τ < 7800:   return I_cha
    elif τ < 8400:   return 0.0
    else:            return I_dis   # wrap, but shouldn't reach here

# ---- 4.  Hybrid ODE --------------------------------------------------
R = 8.314

def rhs(t, y):
    T, Q, Rcell = y
    I   = I_profile(t)
    # thermal ----------------------------------------------------------
    q_ohm = I**2 * (Rcell + p["R_contact"])
    dTdt  = (-p["hA"]*(T - p["T_inf"]) + q_ohm) / (p["m_cell"]*p["Cp_cell"])
    # capacity fade ----------------------------------------------------
    cap_loss_rate = cell.capacity_model.capacity_fade_rate(T, I, d_soc=1.0, dt=1/3600)
    dQdt = -cap_loss_rate
    # resistance rise (very simple linear w.r.t capacity loss) ---------
    k_R  = p["R_cell_init"] * 1.2      # tunable factor
    dRdt = k_R * (cap_loss_rate / Q_nom)
    return [dTdt, dQdt, dRdt]

# ---- 5.  Integrate  200 cycles (≈ 600 h) -----------------------------
t_end = 600 * 3600
y0    = [p["T_inf"], Q_nom, p["R_cell_init"]]
sol   = solve_ivp(rhs, [0, t_end], y0, max_step=60,
                  t_eval=np.linspace(0, t_end, 2000))

time_h = sol.t/3600
T_C    = sol.y[0]-273.15
Q_pct  = sol.y[1]/Q_nom*100
R_mΩ   = sol.y[2]*1000

# ---- 6.  Plots -------------------------------------------------------
fig, ax = plt.subplots(1,3, figsize=(14,4))
ax[0].plot(time_h, T_C);   ax[0].set(xlabel='Time [h]', ylabel='Cell T [°C]')
ax[1].plot(time_h, Q_pct); ax[1].set(xlabel='Time [h]', ylabel='Capacity [%]')
ax[2].plot(time_h, R_mΩ);  ax[2].set(xlabel='Time [h]', ylabel='DCIR [mΩ]')
fig.suptitle('Hybrid BLAST-Lite + Thermal Model'); plt.tight_layout(); plt.show()
