import gc
import math
import pickle

import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
#from Cycling_Ageing_V0 import experiment

# ------------------------------------------------------------------ #
# 1.  modelling the four coupled degradation mechanisms
# ------------------------------------------------------------------ #

deg_param=  pybamm.ParameterValues("OKane2022")
var_pts = {
    "x_n": 5,  # negative electrode
    "x_s": 5,  # separator
    "x_p": 5,  # positive electrode
    "r_n": 30,  # negative particle
    "r_p": 30,  # positive particle
}
#print(deg_param)
Q_new  = 4.0
Q_ref  = deg_param["Nominal cell capacity [A.h]"]
deg_param.update({"Nominal cell capacity [A.h]": Q_new})
scale = Q_new / Q_ref
deg_param.update({
    "Electrode width [m]":  deg_param["Electrode width [m]"]  * scale**0.9,
    "Electrode height [m]": deg_param["Electrode height [m]"] * scale**0.9,
})

r_cyl = 21.30e-3 / 2          # 10.65 mm
h_cyl = 70.30e-3              # 70.3 mm

V_cyl   = math.pi * r_cyl**2 * h_cyl                   # ≈ 2.5e-5 m³
A_cyl   = 2 * math.pi * r_cyl * h_cyl + 2 * math.pi * r_cyl**2   # side + ends

deg_param.update({
    "Cell volume [m3]":            V_cyl,
    #"Cell cooling surface area [m2]": A_cyl,
    # voltage limits:
    "Upper voltage cut-off [V]":   4.2,
    "Lower voltage cut-off [V]":   2.5,
    "SEI kinetic rate constant [m.s-1]": 1e-14
})

# deg_param.update({
#     "Nominal cell capacity [A.h]":   4.0,            # spec sheet
#     "Cell radius [m]":               10.65e-3,       # 21.30 mm Ø
#     "Cell height [m]":               70.3e-3,        # 70.30 mm length
#     "Upper voltage cut-off [V]":     4.2,
#     "Lower voltage cut-off [V]":     2.5,
#     "SEI kinetic rate constant [m.s-1]": 1e-14})
#
# # OKane2022 was 5.15 Ah → scale area by 4.0 / 5.15
# scale = 4.0 / deg_param["Cell capacity [A.h]"]  # original is 5.15
# deg_param.update({
#     "Electrode width [m]":        deg_param["Electrode width [m]"]   * scale**0.5,
#     "Electrode height [m]":       deg_param["Electrode height [m]"]  * scale**0.5,
#     "Electrode area [m2]":        deg_param["Electrode area [m2]"]   * scale,     # redundant but harmless
# })

deg_model= pybamm.lithium_ion.DFN({
    "SEI": "solvent-diffusion limited",
    "thermal": "lumped",
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "irreversible",
    "lithium plating porosity change": "true",  # alias for "SEI porosity change"
    "particle mechanics": ("swelling and cracking", "swelling only"),
    "SEI on cracks": "true",
    "loss of active material": "stress-driven",
    }
)

# deg_model= pybamm.lithium_ion.SPM(
#     {
#         "cell geometry": "arbitrary",
#         "thermal": "lumped",


stioc_initi=deg_param.set_initial_stoichiometries(1)

# ------------------------------------------------------------------ #
# 2.  Defining a cycling protocol
# ------------------------------------------------------------------ #
n_cycles= 250

### replicate the spec-sheet life test
# one_cycle = [
#     pybamm.step.string("Charge at 6 A until 4.2 V"),
#     pybamm.step.string("Hold at 4.2 V until 0.1 A"),
#     pybamm.step.string("Rest for 10 minutes"),
#     pybamm.step.string("Discharge at 35 A until 2.5 V"),
#     pybamm.step.string("Rest for 30 minutes"),
# ]
# exp = pybamm.Experiment(one_cycle * 250, period="2 minutes")
exp = pybamm.Experiment(
    [
        (
            "Discharge at 1C until 3V",
            "Rest for 1 hour",
            "Charge at 1C until 4.2V",
            "Hold at 4.2V until C/50",
        )
    ]
    * n_cycles
)
# ------------------------------------------------------------------ #
# 3.  Solving the model
# ------------------------------------------------------------------ #
#solver = pybamm.CasadiSolver()
solver = pybamm.IDAKLUSolver()
# solver.extra_options = {
#     "max_num_steps":        100000,   # prevent mxstep exit
#     "max_num_newton_iters": 10
# }
# ------------------------------------------------------------------
# 2.  Temperatures to sweep (°C)
# ------------------------------------------------------------------
#temps_C = [25, 45, 55]
temps_C = [25]  # for debugging
temps_K = [t + 273.15 for t in temps_C]

results = {}
for T, label in zip(temps_K, temps_C):
    # clone the parameter object so each run is independent
    pars = deg_param.copy()
    pars.update({
        "Ambient temperature [K]" : T,
        "Initial temperature [K]" : T,
    })
    sim = pybamm.Simulation(deg_model,
                            parameter_values=pars,
                            experiment=exp,
                            var_pts=var_pts,
                            solver=solver,
                            )
    print(f"Solving {label} °C …")
    #sol = sim.solve()
    sol = sim.solve(save_at_cycles=10)
    # with open(f"{T}C_cap.pkl", "wb") as f:
    #     pickle.dump(sol.summary_variables, f)
    # del sol;
    # gc.collect()
    results[label] = sol

# single_cycle_exp = pybamm.Experiment([
#     "Discharge at 1C until 3V",
#     "Rest for 1 hour",
#     "Charge at 1C until 4.2V",
#     "Hold at 4.2V until C/50",
# ])


# ------------------------------------------------------------------
# 3.  Compare capacity-fade vs. cycle for the three temperatures
# ------------------------------------------------------------------
# print("experiment has", len(experiment.operations), "steps")
# cap = sol.summary_variables["Capacity [A.h]"]
# print("summary array length =", len(cap))


print(sol.summary_variables.all_variables)
plt.figure(figsize=(6,4))
colours = cycle(["tab:blue", "tab:orange", "tab:red"])
for T, c in zip(temps_C, colours):
    sol   = results[T]
    cycles = sol.summary_variables.cycle_number
    Q     = sol.summary_variables["Capacity [A.h]"]
    plt.plot(cycles, Q, label=f"{T} °C", color=c)
plt.xlabel("Cycle number")
plt.ylabel("Capacity [A h]")
plt.title("Capacity fade of N21700CGP vs. temperature")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 4.  Inspect full summary panels for a single temperature (e.g. 55 °C)
# ------------------------------------------------------------------
pybamm.plot_summary_variables(results[75])
#
#
# sim= pybamm.Simulation(deg_model, parameter_values=deg_param, experiment= exp,
#                        solver=solver)
# sol= sim.solve()

