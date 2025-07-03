import pybamm as pb, numpy as np, matplotlib.pyplot as plt

N_cycles = 100
exp = pb.Experiment([
    "Discharge at 1C until 2.5 V",
    "Charge at 1C until 4.2 V",
] * N_cycles)

def run_model(temp_K, coupled=True):
    params = pb.ParameterValues("OKane2022").copy()
    params.update({"Nominal cell capacity [A.h]": 3.2,
                   "Ambient temperature [K]": temp_K})
    if coupled:
        params.update({"Initial temperature [K]": temp_K})
        opts = {"thermal": "lumped",          # fully-coupled
                "SEI": "solvent-diffusion limited",
                "SEI porosity change": "true",
                "lithium plating": "partially reversible",
                "particle mechanics": ("swelling and cracking", "swelling only"),
                "loss of active material": "stress-driven"}
    else:
        opts = {"thermal": "isothermal",
                "SEI": "solvent-diffusion limited",
                "SEI porosity change": "true",
                "lithium plating": "partially reversible",
                "particle mechanics": ("swelling and cracking", "swelling only"),
                "loss of active material": "stress-driven"}

    model = pb.lithium_ion.DFN(opts)
    sim   = pb.Simulation(model, parameter_values=params,
                          experiment=exp, solver=pb.IDAKLUSolver())
    return sim.solve()

# --- run at 25 °C and 45 °C (isothermal) ---
sol_25 = run_model(298.15, coupled=True)   # reference
sol_45 = run_model(318.15, coupled=True)   # hot

T = sol_25["Volume-averaged cell temperature [K]"](sol_25.t)
print(f"ΔT, 25 °C run: {T.max()-T.min():.1f} K")


# ------- capacity vs cycle ----------
def discharge_caps(sol):
    caps = []
    for cyc in sol.cycles:
        I = cyc["Current [A]"].data
        if np.mean(I) > 0:                 # PyBaMM convention: +ve = discharge
            caps.append(np.max(cyc["Discharge capacity [A.h]"].data))
    return caps

caps_25 = discharge_caps(sol_25)
caps_45 = discharge_caps(sol_45)

x = np.arange(1, len(caps_25)+1)
caps25_rel = caps_25 - caps_25[0]
caps45_rel = caps_45 - caps_45[0]

plt.figure(figsize=(6,4))
plt.plot(x, caps25_rel*1e3, lw=2, label="25 °C")   # mAh loss
plt.plot(x, caps45_rel*1e3, lw=2, label="45 °C")
plt.xlabel("Cycle"); plt.ylabel("Capacity loss [mAh]")
plt.title("Relative capacity fade (voltage-limited, isothermal DFN)")
plt.legend(); plt.tight_layout(); plt.show()



