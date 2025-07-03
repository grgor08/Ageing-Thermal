import pybamm
import numpy as np
import math
import matplotlib.pyplot as plt

### Model usage ####

full_thermal_model = pybamm.lithium_ion.SPMe(
    {"thermal": "x-full"}, name="full thermal model"
)
lumped_thermal_model = pybamm.lithium_ion.SPMe(
    {"thermal": "lumped"}, name="lumped thermal model"
)
models = [full_thermal_model, lumped_thermal_model]

### pick  parameter set ####

parameter_values = pybamm.ParameterValues("Marquis2019")

full_params = parameter_values.copy()
full_params.update(
    {
        "Negative current collector"
        + " surface heat transfer coefficient [W.m-2.K-1]": 5,
        "Positive current collector"
        + " surface heat transfer coefficient [W.m-2.K-1]": 5,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 0,
        "Edge heat transfer coefficient [W.m-2.K-1]": 0,
    }
)

A = parameter_values["Electrode width [m]"] * parameter_values["Electrode height [m]"]
lumped_params = parameter_values.copy()
lumped_params.update(
    {
        "Total heat transfer coefficient [W.m-2.K-1]": 5,
        "Cell cooling surface area [m2]": 2 * A,
    }
)
params = [full_params, lumped_params]
# loop over the models and solve
sols = []
for model, param in zip(models, params):
    sim = pybamm.Simulation(model, parameter_values=param)
    sim.solve([0, 3600])
    sols.append(sim.solution)


# plot
output_variables = [
    "Voltage [V]",
    "X-averaged cell temperature [K]",
    "Cell temperature [K]",
]
pybamm.dynamic_plot(sols, output_variables)

# plot the results
pybamm.dynamic_plot(
    sols,
    [
        "Volume-averaged cell temperature [K]",
        "Volume-averaged total heating [W.m-3]",
        "Current [A]",
        "Voltage [V]",
    ],
)

model_no_contact_resistance = pybamm.lithium_ion.SPMe(
    {"cell geometry": "arbitrary", "thermal": "lumped", "contact resistance": "false"},
    name="lumped thermal model",
)
model_contact_resistance = pybamm.lithium_ion.SPMe(
    {"cell geometry": "arbitrary", "thermal": "lumped", "contact resistance": "true"},
    name="lumped thermal model with contact resistance",
)
models = [model_no_contact_resistance, model_contact_resistance]

parameter_values = pybamm.ParameterValues("Marquis2019")
lumped_params = parameter_values.copy()
lumped_params_contact_resistance = parameter_values.copy()

lumped_params_contact_resistance.update(
    {
        "Contact resistance [Ohm]": 0.05,
    }
)
params = [lumped_params, lumped_params_contact_resistance]
sols = []
for model, param in zip(models, params):
    sim = pybamm.Simulation(model, parameter_values=param)
    sim.solve([0, 3600])
    sols.append(sim.solution)


output_variables = [
    "Voltage [V]",
    "X-averaged cell temperature [K]",
    "Cell temperature [K]",
]
pybamm.dynamic_plot(sols, output_variables)
