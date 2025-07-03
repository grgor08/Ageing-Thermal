import pybamm as pb
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 0)  Common experiment  (single 1 C discharge–charge cycle)
# ---------------------------------------------------------------
N_cycles = 5
exp = pb.Experiment([
    "Discharge at 1C until 2.5 V",
    "Charge at 1C until 4.2 V",
] * N_cycles)
# ---------------------------------------------------------------
# 1)  Shared parameter set  (OKane2022 → works for SPMe & DFN)
# ---------------------------------------------------------------
params_base = pb.ParameterValues("OKane2022").copy()
Q_nom = 3.2                 # Ah – choose one number for *both* models
params_base.update({"Nominal cell capacity [A.h]": Q_nom})

# ---------------------------------------------------------------
# 2)  Thermal pass  (SPMe + lumped thermal)
# ---------------------------------------------------------------
model_T   = pb.lithium_ion.SPMe({"thermal": "lumped"})
sim_T     = pb.Simulation(model_T,
                          parameter_values=params_base,
                          experiment=exp)
sol_T     = sim_T.solve()

T_interp = pb.Interpolant(sol_T.t,
                          sol_T["Volume-averaged cell temperature [K]"](sol_T.t).flatten(),
                          pb.t,
                          extrapolate=True)      # allow tiny overshoot

# ---------------------------------------------------------------
# 3)  Ageing pass  (DFN + isothermal + ageing models)
# ---------------------------------------------------------------
deg_opts = {
    "thermal": "isothermal",
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "partially reversible",
    "particle mechanics": ("swelling and cracking", "swelling only"),
    "loss of active material": "stress-driven",
}
model_deg  = pb.lithium_ion.DFN(deg_opts)

params_deg = params_base.copy()                 # ← same chemistry!
params_deg.update({"Ambient temperature [K]": T_interp})

sim_deg = pb.Simulation(model_deg,
                        parameter_values=params_deg,
                        experiment=exp,
                        solver=pb.IDAKLUSolver())  # CasadiSolver() also works
sol_deg = sim_deg.solve()

# ---------------------------------------------------------------
# 4)  Overlay & RMSE
# ---------------------------------------------------------------
plt.figure(figsize=(6,4))
plt.plot(sol_T.t,   sol_T ["Voltage [V]"](sol_T.t),   lw=1.8, label="Thermal SPMe")
plt.plot(sol_deg.t, sol_deg["Voltage [V]"](sol_deg.t), lw=1.8, ls="--",
         label="DFN + Ageing")
plt.xlabel("Time [s]"); plt.ylabel("Voltage [V]"); plt.legend(); plt.tight_layout(); plt.show()

# Numerical metric
common_t = np.intersect1d(sol_T.t, sol_deg.t)
rmse = np.sqrt(np.mean((sol_T["Voltage [V]"](common_t)
                       - sol_deg["Voltage [V]"](common_t))**2))
print(f"RMSE = {rmse*1e3:.2f} mV")

# -------- capacity vs cycle -----------
cycle_caps = [
    np.max(cyc_sol["Discharge capacity [A.h]"].data)   # single number / cycle
    for cyc_sol in sol_deg.cycles
]

plt.figure()
plt.plot(range(1, len(cycle_caps)+1), cycle_caps, marker='o')
plt.xlabel("Cycle"); plt.ylabel("Discharge capacity [Ah]")
plt.title(f"Fade over {len(cycle_caps)} discharge cycles")
plt.tight_layout(); plt.show()
T_vec = sol_T["Volume-averaged cell temperature [K]"](sol_T.t)
print(f"Max ΔT over cycle = {T_vec.max()-T_vec.min():.2f} K")

