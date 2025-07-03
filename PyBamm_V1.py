import pybamm as pb
import numpy as np
import math
from matplotlib import pyplot as plt


ageing_opts = {
    "thermal":           "lumped",            # optional but recommended
    "SEI":               "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating":   "partially reversible",
    "lithium plating porosity change": "true",
    "particle mechanics": ("swelling and cracking", "swelling only"),
    "SEI on cracks":     "true",
    "loss of active material": "stress-driven",
}
model = pb.lithium_ion.DFN(ageing_opts)
param = pb.ParameterValues("OKane2022")

# ---------- cell geometry ----------
diam, height = 0.021, 0.070
param.update({
    "Nominal cell capacity [A.h]":    4.0,
    "Cell volume [m3]": 2.4e-5 ,            # 2.4e-5
    "Negative electrode thickness [m]": 70e-6,
    "Positive electrode thickness [m]": 65e-6,
    "Total heat transfer coefficient [W.m-2.K-1]": 5,
    "Contact resistance [Ohm]": 0.012,
})

var_pts = {
    "x_n": 5, "x_s": 5, "x_p": 5,   # electrode/separator
    "r_n": 30, "r_p": 30,           # particle radial points
}

########   Define the cycling protocol ###########
cycle_N = 10
exp = pb.Experiment(
    ["Hold at 4.2 V until C/100",
     "Rest for 4 hours",
     "Discharge at 0.1C until 2.5 V"] +
    ["Charge at 0.3C until 4.2 V",
     "Hold at 4.2 V until C/100",
     "Discharge at 1C until 2.5 V"] * cycle_N +
    ["Discharge at 0.1C until 2.5 V"]
)

###### SOlver #################
solver = pb.IDAKLUSolver()
sim    = pb.Simulation(model,
                       parameter_values=param,
                       experiment=exp,
                       solver=solver,
                       var_pts=var_pts)
sol = sim.solve()

# --- extract discharge-capacity trace ---------------------------------
t  = sol["Time [h]"].entries                          # global time axis
Qd = sol["Discharge capacity [A.h]"].entries          # variable defined by DFN

plt.figure()
plt.plot(t, Qd)
plt.xlabel("Time [h]")
plt.ylabel("Discharge capacity [A·h]")
plt.title("Instantaneous discharge capacity during each cycle")
plt.tight_layout()
plt.show()

########Plot key degradation outputs #########

Qt       = sol["Throughput capacity [A.h]"].entries
Q_SEI    = sol["Loss of capacity to negative SEI [A.h]"].entries
Q_SEIcr  = sol["Loss of capacity to negative SEI on cracks [A.h]"].entries
Q_plate  = sol["Loss of capacity to negative lithium plating [A.h]"].entries
Q_side   = sol["Total capacity lost to side reactions [A.h]"].entries
LLI_pct  = sol["Loss of lithium inventory [%]"].entries
Qt = sol["Throughput capacity [A.h]"].entries
LLI = sol["Loss of lithium inventory [%]"].entries
LAM_neg = sol["Loss of active material in negative electrode [%]"].entries
LAM_pos = sol["Loss of active material in positive electrode [%]"].entries
plt.figure()
plt.plot(Qt, LLI, label="LLI")
plt.plot(Qt, LAM_neg, label="LAM (negative)")
plt.plot(Qt, LAM_pos, label="LAM (positive)")
plt.xlabel("Throughput capacity [A.h]")
plt.ylabel("Degradation modes [%]")
plt.legend()
plt.show()
sim.plot()