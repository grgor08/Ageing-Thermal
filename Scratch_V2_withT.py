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
#print(deg_param)
deg_param.update({"SEI kinetic rate constant [m.s-1]": 1e-14})

deg_model= pybamm.lithium_ion.DFN({
    "SEI": "solvent-diffusion limited",
    "thermal": "lumped",
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "partially reversible",
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
n_cycles= 50
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
# ------------------------------------------------------------------
# 2.  Temperatures to sweep (°C)
# ------------------------------------------------------------------
temps_C = [25, 45, 55]
#temps_C = [55]  # for debugging
temps_K = [t + 273.15 for t in temps_C]

results = {}
for T, label in zip(temps_K, temps_C):
    # clone the parameter object so each run is independent
    pars = deg_param.copy()
    pars.update({
        "Ambient temperature [K]" : T,
        "Initial temperature [K]" : T,
    })
    pars.set_initial_stoichiometries(1)
    sim = pybamm.Simulation(deg_model,
                            parameter_values=pars,
                            experiment=exp,
                            solver=solver)
    print(f"Solving {label} °C …")
    sol = sim.solve()
    results[label] = sol

single_cycle_exp = pybamm.Experiment([
    "Discharge at 1C until 3V",
    "Rest for 1 hour",
    "Charge at 1C until 4.2V",
    "Hold at 4.2V until C/50",
])

import pandas as pd

# Prepare data for export
export_data = []
for T in temps_C:
    sol = results[T]
    cycles = sol.summary_variables.cycle_number
    Q = sol.summary_variables["Capacity [A.h]"]
    Q_mAh = Q * 1000  # Convert to mAh
    cap_loss_mAh = Q_mAh[0] - Q_mAh[-1]
    cap_loss_pct = 100 * (Q_mAh[0] - Q_mAh[-1]) / Q_mAh[0]
    for cycle, cap in zip(cycles, Q_mAh):
        export_data.append({
            "Temperature (°C)": T,
            "Cycle": cycle,
            "Capacity (mAh)": cap,
        })
    # Add summary row for this temperature


    # ... inside your loop:
    for cycle_num, cap in zip(cycles, Q_mAh):
        export_data.append({
            "Temperature (°C)": T,
            "Cycle": cycle_num,
            "Capacity (mAh)": cap,
        })
print(f" The capacity loss at {T} °C is {cap_loss_mAh:.2f} mAh, "
      f"which is {cap_loss_pct:.2f}% of the initial capacity.")
# Convert to DataFrame and export
df = pd.DataFrame(export_data)
df.to_excel("capacity_results.xlsx", index=False)
# ------------------------------------------------------------------
# 3.  Compare capacity-fade vs. cycle for the three temperatures
# ------------------------------------------------------------------
# print("experiment has", len(experiment.operations), "steps")
# cap = sol.summary_variables["Capacity [A.h]"]
# print("summary array length =", len(cap))


#print(sol.summary_variables.all_variables)
plt.figure(figsize=(6,4))
colours = cycle(["tab:blue", "tab:orange", "tab:red"])
for T, c in zip(temps_C, colours):
    sol   = results[T]
    cycles = sol.summary_variables.cycle_number
    Q     = sol.summary_variables["Capacity [A.h]"]
    plt.plot(cycles, Q, label=f"{T} °C", color=c)
plt.xlabel("Cycle number")
plt.ylabel("Capacity [A h]")
plt.title("Capacity retention vs. temperature, 1 C cycling")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 4.  Inspect full summary panels for a single temperature (e.g. 55 °C)
# ------------------------------------------------------------------
pybamm.plot_summary_variables(results[55])


sim= pybamm.Simulation(deg_model, parameter_values=deg_param, experiment= exp,
                       solver=solver)
sol= sim.solve()

