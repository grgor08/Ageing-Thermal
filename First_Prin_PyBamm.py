"""
sony_window_capacity_fade_okane.py
DFN + SEI model using OKane-2022 parameters
Voltage window, current and nominal capacity are set to match
Sony US-18650 cycling in Ramadass 2002.
"""

import pybamm as pb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ---------------- USER SETTINGS ----------------
import pybamm as pb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time

# ---------------- USER SETTINGS ----------------
Start = time.process_time()
n_cycles = 5
temps_C = [25, 45, 55]
out_dir = Path("results")
save_csv = True
# -----------------------------------------------

# 1. model + base parameters
ageing_opts = {
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "thermal": "lumped",
}
model = pb.lithium_ion.DFN(ageing_opts)

base_pv = pb.ParameterValues("OKane2022")
base_pv.update({
    "Upper voltage cut-off [V]": 4.10,
    "Lower voltage cut-off [V]": 2.75,
    "Nominal cell capacity [A.h]": 1.4,  # so ‘1 C’ = 1.4 A
})

# 2. cycling protocol
cycle = [
    "Charge at 1C for 2 hours or until 4.1 V",
    "Hold at 4.1 V for 1 hour or until -0.035 A",
    "Rest for 10 minutes",
    "Discharge at 1C for 4 hours or until 2.75 V",
    "Rest for 10 minutes",
]
experiment = pb.Experiment(cycle * n_cycles, period="10 seconds")

# 3. temperature sweep
solver = pb.CasadiSolver(rtol=1e-6, atol=1e-8)
results = {}

for T in temps_C:
    print(f"\n=== Simulating {n_cycles} cycles at {T} °C ===")
    pv = base_pv.copy()
    pv.update({
        "Ambient temperature [K]": 273.15 + T,
        "Initial temperature [K]": 273.15 + T,
    })

    sim = pb.Simulation(
        model,
        parameter_values=pv,
        experiment=experiment,
        solver=solver,
    )
    sol = sim.solve(initial_soc=0)

    cap_per_cycle = []
    prev_total = 0.0

    for i, cyc in enumerate(sol.cycles):
        print(f"Cycle {i + 1} type:", type(cyc))  # Check what type the object is
        print(f"Cycle {i + 1} _variables:", list(cyc._variables.keys()))  # Access private _variables attribute
        print("Available variables in full solution:")
        print(list(sol._variables.keys()))

        # Safely attempt to get discharge capacity
        if "Discharge capacity [A.h]" in cyc._variables.keys():
            total = cyc["Discharge capacity [A.h]"].data[-1]
            increment = total - prev_total
            cap_per_cycle.append(increment)
            prev_total = total
        else:
            print(f"Warning: 'Discharge capacity [A.h]' not found for cycle {i + 1}")
            cap_per_cycle.append(np.nan)

    # Remove NaNs if any
    cap_per_cycle = [v for v in cap_per_cycle if not np.isnan(v)]

    if len(cap_per_cycle) > 0:
        df = pd.DataFrame({
            "cycle": np.arange(1, len(cap_per_cycle) + 1),
            "capacity_Ah": cap_per_cycle,
        })
        df["relative_capacity"] = df["capacity_Ah"] / df["capacity_Ah"][0]
        results[T] = df
        if save_csv:
            out_dir.mkdir(exist_ok=True)
            df.to_csv(out_dir / f"capacity_vs_cycle_{T}C.csv", index=False)
    else:
        print(f"No valid capacity data for T={T}C")
        results[T] = pd.DataFrame()  # Empty df

# Plotting section
plt.figure(figsize=(10, 6))
for T, df in results.items():
    if not df.empty:
        plt.plot(df["cycle"], df["relative_capacity"], 'o-', label=f"{T} °C", markersize=4)
    else:
        print(f"No data to plot for {T} °C")

plt.xlabel("Cycle number")
plt.ylabel("Relative Capacity")
plt.title("Capacity Fade Simulation")
plt.grid(True)
plt.legend()
plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()
print("Elapsed time:", time.process_time() - Start)