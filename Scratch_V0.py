import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------------------ #
# 1.  modelling the four coupled degradation mechanisms
# ------------------------------------------------------------------ #

deg_param=  pybamm.ParameterValues("OKane2022")
#print(deg_param)

deg_model= pybamm.lithium_ion.SPM(
    {
        "SEI": "solvent-diffusion limited",
        "SEI porosity change": "true",
        "lithium plating": "partially reversible",
        "lithium plating porosity change": "true",  # alias for "SEI porosity change"
        "particle mechanics": ("swelling and cracking", "swelling only"),
        "SEI on cracks": "true",
        "loss of active material": "stress-driven",
    }
)

# ------------------------------------------------------------------ #
# 2.  Defining a cycling protocol
# ------------------------------------------------------------------ #

cycle_number= 10
