import pybamm as pb
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 0)  Common experiment  – 5× (1 C discharge → 2.5 V, 1 C charge → 4.2 V)
# ---------------------------------------------------------------
N_cycles = 20
exp = pb.Experiment([
    "Discharge at 1C until 2.5 V",
    "Charge at 1C until 4.2 V",
] * N_cycles)

# ---------------------------------------------------------------
# 1)  Shared parameter set
# ---------------------------------------------------------------
params = pb.ParameterValues("OKane2022").copy()
params.update({"Nominal cell capacity [A.h]": 3.2})

# ---------------------------------------------------------------
# 2)  Fully-coupled DFN  (electro-thermal-ageing all in one)
# ---------------------------------------------------------------
opts_full = {
    "thermal": "lumped",                 # <-- fully coupled
    "SEI": "solvent-diffusion limited",
    "SEI porosity change": "true",
    "lithium plating": "partially reversible",
    "particle mechanics": ("swelling and cracking", "swelling only"),
    "loss of active material": "stress-driven",
}
model = pb.lithium_ion.DFN(opts_full)
sim   = pb.Simulation(model, parameter_values=params,
                      experiment=exp, solver=pb.IDAKLUSolver())
sol   = sim.solve()

# ---------------------------------------------------------------
# 3)  Voltage profile
# ---------------------------------------------------------------
plt.figure(figsize=(6,4))
plt.plot(sol.t, sol["Voltage [V]"](sol.t), lw=1.8)
plt.xlabel("Time [s]"); plt.ylabel("Voltage [V]")
plt.title("Fully-coupled DFN ({} cycles)".format(N_cycles))
plt.tight_layout(); plt.show()

# ---------------------------------------------------------------
# 4)  Capacity vs cycle  (discharge only)
# ---------------------------------------------------------------

from scipy.interpolate import make_interp_spline
caps = [np.max(c["Discharge capacity [A.h]"].data) for c in sol.cycles]
x     = np.arange(1, len(caps)+1)                  # 1,2,3…
x_new = np.linspace(1, len(caps), 200)             # 200 dense points
y_new = make_interp_spline(x, caps, k=3)(x_new)    # cubic spline

plt.figure(figsize=(6,4))
plt.plot(x_new, y_new, lw=2)                       # smooth line
plt.xlabel("Cycle"); plt.ylabel("Discharge capacity [Ah]")
plt.title(f"Capacity fade over {len(caps)} discharge cycles")
plt.tight_layout(); plt.show()

# ---------------------------------------------------------------
# 5)  Temperature rise
# ---------------------------------------------------------------
T  = sol["Volume-averaged cell temperature [K]"](sol.t)
print(f"Max ΔT over {N_cycles} cycles = {T.max()-T.min():.2f} K")

from scipy.interpolate import make_interp_spline