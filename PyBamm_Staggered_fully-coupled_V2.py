import pybamm as pb, numpy as np, matplotlib.pyplot as plt, time

# ---------------------------------------------------------------
N_cycles = 800                             # discharge–charge pairs
Q_nom    = 3.2                             # Ah (sets 1 C current)
T_list   = [298.15, 318.15]                # 25 °C, 45 °C
I_1C     = Q_nom                           # 1C in A (because 1 Ah = 1 A·h)
t_seg    = 3700                            # 1C discharge or charge ≈ 1 h ≈ 3600 s
dis_cmd = "Discharge at 1C until 2.5 V"
chg_cmd = "Charge at 1C until 4.2 V"
# ---------------------------------------------------------------

def make_sim(params, current_A):
    """Return a Simulation object set to constant current `current_A`."""
    model = pb.lithium_ion.DFN({
        "thermal": "isothermal",
        "SEI": "solvent-diffusion limited",
        "SEI porosity change": "true",
        "lithium plating": "partially reversible",
        "particle mechanics": ("swelling and cracking", "swelling only"),
        "loss of active material": "stress-driven",
    })
    params = params.copy()
    params.update({"Current function [A]": current_A})
    return pb.Simulation(model, parameter_values=params,
                         solver=pb.CasadiSolver(mode="fast"))

def run_many_cycles(temp_K):
    params = pb.ParameterValues("OKane2022").copy()
    params.update({"Nominal cell capacity [A.h]": Q_nom,
                   "Ambient temperature [K]":   temp_K})

    # make discharge and charge sims once (they share parameter set)
    sim_dis = make_sim(params,  +I_1C)
    sim_chg = make_sim(params,  -I_1C)

    caps = []
    sol_prev = None
    t0 = time.perf_counter()

    for k in range(N_cycles):
        # -------- discharge segment --------
        sol_d = sim_dis.solve(experiment=pb.Experiment([dis_cmd]),
                              starting_solution=sol_prev)
        caps.append(sol_d["Discharge capacity [A.h]"].data[-1])

        # -------- charge segment ----------
        sol_prev = sim_chg.solve(experiment=pb.Experiment([chg_cmd]),
                                 starting_solution=sol_d)

        if (k+1) % 100 == 0:
            print(f"{k+1}/{N_cycles} cycles completed…")

    elapsed = time.perf_counter() - t0
    return np.array(caps), elapsed

# ---------------- run for each temperature ----------------------
results = {}
for Tk in T_list:
    caps, secs = run_many_cycles(Tk)
    results[Tk] = (caps, secs)
    print(f"{Tk-273.15:.0f} °C finished in {secs/60:.1f} min")

# -------------------- plotting ----------------------------------
x = np.arange(1, N_cycles+1)

plt.figure(figsize=(7,4))
for Tk,(caps,_) in results.items():
    plt.plot(x, caps, lw=1.8, label=f"{Tk-273.15:.0f} °C")
plt.xlabel("Cycle"); plt.ylabel("Discharge capacity [Ah]")
plt.title(f"Capacity fade – isothermal DFN, {N_cycles} cycles")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(7,4))
for Tk,(caps,_) in results.items():
    plt.plot(x, (caps - caps[0])*1e3, lw=1.8, label=f"{Tk-273.15:.0f} °C")
plt.xlabel("Cycle"); plt.ylabel("Capacity loss [mAh]")
plt.title("Relative capacity fade")
plt.legend(); plt.tight_layout(); plt.show()

