"""
Cycle-life “debug harness” for PyBaMM
------------------------------------
•  Creates a DFN-based model with up-to-date degradation options
•  Runs an n-cycle CC/CV experiment under several option bundles
•  Produces:
    – capacity-fade curve
    – key degradation summaries (LLI & LAM-neg)
    – electrode-potential plot for the last *completed* charge step
Tested with PyBaMM ≥ 23.1

Author:  Yasir Ibrahim
"""


#!/usr/bin/env python3
import pybamm as pb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import least_squares
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1.  Load experimental capacity-fade curve
# --------------------------------------------------------------------------- #

CSV_PATH = Path("ramadass_45C.csv")        # << change if needed
exp_df   = pd.read_csv(CSV_PATH)
exp_cycles = exp_df.Cycle.values.astype(float)
exp_cap    = exp_df.Capacity_Ah.values.astype(float)


N_CYCLES = int(exp_cycles[-1])
# ------------------------------------------------------------------ #
# 2.  EXPERIMENT & SIMULATION SET-UP
# ------------------------------------------------------------------ #



def create_experiment(n_cycles: int) -> pb.Experiment:
    cycle = (
        "Discharge at 0.5C for 2 hours or until 2.5 V",
        "Charge at 0.5C for 3 hours or until 4.1 V",
        "Hold at 4.1 V for 2 hour or until 0.05C",
    )
    return pb.Experiment([cycle] * n_cycles)


def setup_simulation():
    options = {
        "thermal": "lumped",
        "SEI": "solvent-diffusion limited",
        "lithium plating": "none",
        "particle mechanics": "none",
    }
    model = pb.lithium_ion.DFN(options=options)

    params = pb.ParameterValues("OKane2022").copy()
    params.update({                       # Sony US-18650 conditions
        "Nominal cell capacity [A.h]": 1.30,
        "Ambient temperature [K]":     298.15,
        "Initial temperature [K]":     298.15,
        # ----- Ramadass baseline SEI numbers -----------------
        "SEI kinetic rate constant [m.s-1]":      1e-14,
        "SEI resistivity [Ohm.m]":              200,
        "Initial SEI thickness [m]":            30e-9,
        "SEI solvent diffusivity [m2.s-1]":               1e-22,
    })

    sim = pb.Simulation(
        model,
        experiment=create_experiment(N_CYCLES),
        parameter_values=params,
        solver=pb.IDAKLUSolver(rtol=1e-6, atol=1e-8),
        output_variables=[
            "Discharge capacity [A.h]",
            "Cycle number",
        ],
    )
    return sim


sim = setup_simulation()

## --------------------------------------------------------------------------- #
# 3.  Objective function for least-squares fit
# --------------------------------------------------------------------------- #
fit_keys = [
    "SEI kinetic rate constant [m.s-1]",
    "SEI resistivity [Ohm.m]",
]
x0 = np.array([sim.parameter_values[k] for k in fit_keys])  # initial guess
bounds = ([1e-16,  50],        # lower bounds  (k_SEI, rho_SEI)
          [1e-12, 500])        # upper bounds


def residuals(x):
    sim.parameter_values.update(dict(zip(fit_keys, x)))
    try:
        sol = sim.solve()
    except pb.SolverError:
        # solver blew up → huge penalty
        return 1e3 * np.ones_like(exp_cap)

    sv = sol.summary_variables.get_summary_variables()
    # ---- robust capacity extraction ----
    if "Discharge capacity [A.h]" in sv:
        model_cycles = sv["Cycle number"]
        model_Q      = sv["Discharge capacity [A.h]"]
    else:  # fall back to step-level data
        model_cycles = np.arange(len(sol.cycles), dtype=float)
        model_Q      = [
            cyc.steps[0]["Discharge capacity [A.h]"].data[-1]
            for cyc in sol.cycles
        ]
        if len(model_Q) < len(exp_cap):        # run truncated early
            return 1e3 * np.ones_like(exp_cap)

    interp_Q = np.interp(exp_cycles, model_cycles, model_Q)
    return interp_Q - exp_cap


result = least_squares(residuals, x0, bounds=bounds, verbose=2)
best_params = dict(zip(fit_keys, result.x))

print("\nBest-fit parameters:")
for k, v in best_params.items():
    print(f"  {k:40s} = {v:.3e}")

print(f"\nRMSE  : {np.sqrt(np.mean(result.fun**2)):.4f} Ah")
print(f"R^2   : {1 - np.var(result.fun) / np.var(exp_cap):.4f}")

# rerun simulation once with best parameters for clean plots
sim.parameter_values.update(best_params)
best_sol = sim.solve()
sv_best  = best_sol.summary_variables.get_summary_variables()

# --------------------------------------------------------------------------- #
# 4.  Plot experimental vs. model capacity fade
# --------------------------------------------------------------------------- #
plt.figure(figsize=(8, 5))
plt.plot(exp_cycles, exp_cap*1e3, "ko", label="Ramadass data")
plt.plot(sv_best["Cycle number"],
         sv_best["Discharge capacity [A.h]"]*1e3,
         "b-", label="PyBaMM model")
plt.xlabel("Cycle number")
plt.ylabel("Capacity [mAh]")
plt.title("Capacity-fade validation, 0.5C/0.5C, 25 °C")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()