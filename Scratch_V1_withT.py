import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Cycling_Ageing_V0 import experiment

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
n_cycles= 300
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
sim= pybamm.Simulation(deg_model, parameter_values=deg_param, experiment= exp,
                       solver=solver)
sol= sim.solve()
# sot_sol= sorted(sol.summary_variables.all_variables)
# print(sot_sol)

# ------------------------------------------------------------------ #
# 4. plotting
# ------------------------------------------------------------------ #

pybamm.plot_summary_variables(sol)
plt.show()
