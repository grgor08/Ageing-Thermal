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


# ------------------------------------------------------------------ #
# 1.  EXPERIMENT & SIMULATION SET-UP
# ------------------------------------------------------------------ #



def create_experiment(n_cycles: int) -> pb.Experiment:
    cycle = (
        "Discharge at 1C until 2.5 V",   # step 1
        "Charge at 0.5C until 4.2 V",      # step 2
    )
    return pb.Experiment([cycle] * n_cycles)


def setup_simulation(model_options, experiment, temp_K=298.15):
    params = pb.ParameterValues("OKane2022").copy()
    params.update({
        "Nominal cell capacity [A.h]": 3.2,
        "Total heat transfer coefficient [W.m-2.K-1]": 2,
        "Ambient temperature [K]": temp_K,
        "Initial temperature [K]": temp_K,
    })

    model = pb.lithium_ion.DFN(options=model_options)
    solver = pb.IDAKLUSolver(rtol=1e-6, atol=1e-8)

    return pb.Simulation(model, experiment=experiment,
                         parameter_values=params, solver=solver)


# ------------------------------------------------------------------ #
# 2.  POST-PROCESS HELPERS
# ------------------------------------------------------------------ #

def summary(sol):
    """Return the cycle-averaged summary-variables dict (PyBaMM ≥ 23)."""
    return sol.summary_variables.get_summary_variables()

def get_discharge_capacity(sol) -> np.ndarray:
    sv = sol.summary_variables.get_summary_variables()
    if "Discharge capacity [A.h]" in sv:          # ← new guard
        return np.asarray(sv["Discharge capacity [A.h]"])[1:]

    caps = [cyc.steps[0]["Discharge capacity [A.h]"].data[-1]
            for cyc in sol.cycles[1:]]
    return np.array(caps)


def plot_capacity_fade(sol, title):
    caps = get_discharge_capacity(sol)
    loss_mAh = (caps[0] - caps) * 1e3
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(caps)+1), loss_mAh, "o-")
    plt.xlabel("Cycle number")
    plt.ylabel("Capacity loss [mAh]")
    plt.title(f"Capacity fade – {title}")
    plt.grid(True); plt.tight_layout(); plt.show()


def plot_degradation_variables(sol, title):
    sv = summary(sol)
    cycle = sv["Cycle number"]
    lli   = sv["Loss of lithium inventory [%]"]
    plt.figure(figsize=(10, 6))
    plt.plot(cycle, lli, "o-", label="Lithium-inventory loss [%]")
    lam_key = "Loss of active material in negative electrode [%]"
    if lam_key in sv:
        plt.plot(cycle, sv[lam_key], "s-", label="Negative-electrode LAM [%]")
    plt.xlabel("Cycle number"); plt.ylabel("Loss [%]")
    plt.title(f"Degradation mechanisms – {title}")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


def plot_electrode_potentials(sol, cycle_idx, title):
    # back-track to last *completed* charge step
    for idx in range(cycle_idx, -1, -1):
        if len(sol.cycles[idx].steps) > 1:
            charge_data = sol.cycles[idx].steps[1]
            break
    else:
        print("No completed charge step found.");  return

    t = charge_data["Time [h]"].data
    v = charge_data["Terminal voltage [V]"].data
    n_p = charge_data["X-averaged negative electrode potential [V]"].data
    p_p = charge_data["X-averaged positive electrode potential [V]"].data

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Time [h]"); ax1.set_ylabel("Potential [V]")
    ax1.plot(t, n_p, "b-", label="Negative electrode")
    ax1.plot(t, p_p, "r-", label="Positive electrode")
    ax1.axhline(0, color="b", ls="--", label="Plating threshold")
    ax1.legend(loc="center left")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Terminal voltage [V]", color="g")
    ax2.plot(t, v, "g-", label="Terminal V")
    ax2.tick_params(axis="y", labelcolor="g")
    ax2.legend(loc="center right")
    plt.title(f"Electrode potentials – cycle {idx} ({title})")
    plt.grid(True); plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ #
# 3.  MAIN
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    N_CYCLES = 100
    experiment = create_experiment(N_CYCLES)

    model_configurations = {
        "SEI_Only": {
            "thermal": "lumped",
            "SEI": "solvent-diffusion limited",
        },
        "SEI_and_Plating": {
            "thermal": "lumped",
            "SEI": "solvent-diffusion limited",
            "lithium plating": "partially reversible",
        },
        "Full_Model": {
            "thermal": "lumped",
            "SEI": "solvent-diffusion limited",
            "lithium plating": "partially reversible",
            "particle mechanics": "swelling and cracking",
            "loss of active material": "stress-driven",
        },
    }

    for name, options in model_configurations.items():
        print(f"\n{'='*22}  RUNNING: {name}  {'='*22}")
        sim = setup_simulation(options, experiment)
        solution = sim.solve()

        # ----- print correct cycle count
        full_cycles = int(summary(solution)["Cycle number"][-1])
        print(f"Termination reason: {solution.termination}")
        print(f"Completed full cycles: {full_cycles} / {N_CYCLES}")

        # ----- plots
        plot_capacity_fade(solution, name)
        plot_degradation_variables(solution, name)
        plot_electrode_potentials(solution,
                                  cycle_idx=len(solution.cycles)-1,
                                  title=name)
