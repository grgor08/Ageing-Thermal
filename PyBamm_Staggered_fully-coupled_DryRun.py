import pybamm as pb, numpy as np, matplotlib.pyplot as plt

N_cycles = 800
exp = pb.Experiment([
    "Discharge at 3C until 2.5 V",
    "Charge at 1C until 4.2 V",
] * N_cycles)

def run_model(temp_K, coupled=True):

    params = pb.ParameterValues("OKane2022").copy()
    params.update({"Nominal cell capacity [A.h]": 3.2,
                   "Total heat transfer coefficient [W.m-2.K-1]": 2,
                   "Ambient temperature [K]": temp_K})
    if coupled:
        params.update({"Initial temperature [K]": temp_K})
        opts = {"thermal": "lumped",          # fully-coupled q
                "SEI": "solvent-diffusion limited",
                "SEI porosity change": "false",
                "lithium plating": "irreversible",
                "particle mechanics": ("swelling and cracking", "swelling only"),
                "loss of active material": "stress-driven"}
    else:
        opts = {"thermal": "isothermal",
                "SEI": "solvent-diffusion limited",
                "SEI porosity change": "false",
                "lithium plating": "irreversible",
                "particle mechanics": ("swelling and cracking", "swelling only"),
                "loss of active material": "stress-driven"}

    model = pb.lithium_ion.DFN(opts)
    solver = pb.IDAKLUSolver()
    solver.rtol = 1e-6
    solver.atol = 1e-8
    sim = pb.Simulation(model, parameter_values=params,
                        experiment=exp, solver=solver)
    return sim.solve()

# --- run at 25 °C and 45 °C (isothermal) ---
sol_iso = run_model(298.15, coupled=False)   # reference
sol_cpl = run_model(298.15, coupled=True)   # hot




# ------- capacity vs cycle ----------
def discharge_caps(sol):
    caps = []
    for cyc in sol.cycles:
        I = cyc["Current [A]"].data
        if np.mean(I) < 0:                 # PyBaMM convention: +ve = discharge
            caps.append(np.max(cyc["Discharge capacity [A.h]"].data))
    return caps

caps_iso = discharge_caps(sol_iso)
caps_cpl = discharge_caps(sol_cpl)

caps_iso_rel = caps_iso[1:] - caps_iso[1]
caps_cpl_rel = caps_cpl[1:] - caps_cpl[1]

x = np.arange(2, len(caps_iso)+1)
plt.figure(figsize=(6,4))
plt.plot(x, caps_iso_rel*1e3, label="Isothermal")
plt.plot(x, caps_cpl_rel*1e3, label="Coupled", lw=2)
plt.xlabel("Cycle"); plt.ylabel("Capacity loss [mAh]")
plt.title("3 C / 1 C – effect of self-heating")
plt.legend(); plt.tight_layout(); plt.show()

# Temperature rise
T = sol_cpl["Volume-averaged cell temperature [K]"](sol_cpl.t)
print(f"Coupled run max ΔT ≈ {T.max()-T.min():.1f} K")
Q_nom = 3.2
loss_iso = caps_iso[-1] - caps_iso[0]     # Ah
loss_cpl = caps_cpl[-1] - caps_cpl[0]

print(f"Isothermal loss : {loss_iso*1e3:.1f} mAh  ({loss_iso/Q_nom*100:.2f} %)")
print(f"Coupled loss    : {loss_cpl*1e3:.1f} mAh  ({loss_cpl/Q_nom*100:.2f} %)")
print(f"Acceleration factor ≈ {loss_cpl/loss_iso:.1f}×")




