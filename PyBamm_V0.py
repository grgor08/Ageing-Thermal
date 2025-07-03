import pybamm as pb
import numpy as np
import math

# 1. Geometry helpers
diam, height = 0.021, 0.070         # m
cell_volume  = math.pi * (diam/2)**2 * height   # 2.42e-5 m³
cell_mass    = 0.070                # kg
print(cell_volume)

param = pb.ParameterValues("OKane2022")
param.update({
      "Nominal cell capacity [A.h]": 4,
    "Negative electrode thickness [m]": 70e-6,
    "Positive electrode thickness [m]": 65e-6,
    "Cell volume [m3]": 1.4245241304079233e-05,
    "Separator specific heat capacity [J.kg-1.K-1]": 1040, #
    "Positive electrode specific heat capacity [J.kg-1.K-1]": 1040,
    "Negative electrode specific heat capacity [J.kg-1.K-1]": 1040,
    "Total heat transfer coefficient [W.m-2.K-1]": 5,
    "Contact resistance [Ohm]": 0.012,
})

options = {"thermal": "lumped", "SEI": "reaction limited"}
model   = pb.lithium_ion.DFN(options)
sim     = pb.Simulation(model, parameter_values=param)

t_eval  = np.linspace(0, 3600, 1000)           # 1 h discharge
sim.solve(t_eval)
sim.plot()
