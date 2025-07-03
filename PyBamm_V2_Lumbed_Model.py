import pybamm as pb
import numpy as np
import math
import matplotlib.pyplot as plt   # <-- keep the canonical alias



# ---------------------------------------------------------------------
# 1. OPTIONS  – switched thermal o
# ---------------------------------------------------------------------
ageing_opts = {
    "thermal"                 : "lumped",            # uniform T in the jelly-roll
    "SEI"                     : "solvent-diffusion limited",
    "SEI porosity change"     : "true",
    "lithium plating"         : "partially reversible",
    "lithium plating porosity change": "true",
    "particle mechanics"      : ("swelling and cracking", "swelling only"),
    "SEI on cracks"           : "true",
    "loss of active material" : "stress-driven",
}
model = pb.lithium_ion.DFN(ageing_opts)

# ---------------------------------------------------------------------
# 2. PARAMETER SET  – thermal parameters are already in OKane2022
#
# ---------------------------------------------------------------------
param = pb.ParameterValues("OKane2022")
diam, height = 0.021, 0.070                       # m
cell_vol      = math.pi*(diam/2)**2 * height      # ≈ 2.43×10⁻⁵ m³

param.update(
    {
        "Nominal cell capacity [A.h]"      : 4.0,
        "Cell volume [m3]"                 : cell_vol,
        # external cooling: h·A  [W m⁻² K⁻¹]
        "Total heat transfer coefficient [W.m-2.K-1]" : 8,  # adjust as needed
        "Contact resistance [Ohm]"         : 0.012,

    }
)

# ---------------------------------------------------------------------
# 3. MESH DENSITY & CYCLING PROTOCOL (unchanged)
# ---------------------------------------------------------------------
var_pts = {"x_n":5,"x_s":5,"x_p":5,"r_n":30,"r_p":30}

cycle_N = 10
exp = pb.Experiment(
    ["Hold at 4.2 V until C/100",
     "Rest for 4 hours",
     "Discharge at 0.1C until 2.5 V"] +
    ["Charge at 0.3C until 4.2 V",
     "Hold at 4.2 V until C/100",
     "Discharge at 1C until 2.5 V"]*cycle_N +
    ["Discharge at 0.1C until 2.5 V"]
)

solver = pb.IDAKLUSolver()
sim    = pb.Simulation(model, parameter_values=param,
                       experiment=exp, solver=solver, var_pts=var_pts)

sol = sim.solve()

# -------- cycle-wise scalars ---------------------------
Qd_loop = np.array([cyc["Discharge capacity [A.h]"].entries[-1]
                    for cyc in sol.cycles])
loop_id = np.arange(1, len(Qd_loop)+1)

plt.figure()
plt.scatter(loop_id, Qd_loop, s=25)
plt.xlabel("Ageing loop #"); plt.ylabel("Discharge capacity [A·h]")

# ------------------------------------------------------------------
# 4.  THERMAL VARIABLES
# ------------------------------------------------------------------
t_h   = sol["Time [h]"].entries
T_C   = sol["X-averaged cell temperature [K]"].entries - 273.15  # °C

q_tot = sol["Volume-averaged total heating [W.m-3]"].entries * cell_vol
q_ohm = sol["Volume-averaged Ohmic heating [W.m-3]"].entries * cell_vol
q_rxn = sol["Volume-averaged irreversible electrochemical heating [W.m-3]"].entries * cell_vol
q_rev = sol["Volume-averaged reversible heating [W.m-3]"].entries * cell_vol


# ------------------------------------------------------------------
# 5.  AGEING VARIABLES
# ------------------------------------------------------------------
Qt       = sol["Throughput capacity [A.h]"].entries
LLI      = sol["Loss of lithium inventory [%]"].entries
LAM_neg  = sol["Loss of active material in negative electrode [%]"].entries
LAM_pos  = sol["Loss of active material in positive electrode [%]"].entries
# --- NEW: build cycle-wise time axis and IDs ------------------------
# (each 'cycle' in sol.cycles is itself a Solution with its own .t array)
t_cycle  = np.concatenate([cy.t / 3600 for cy in sol.cycles])          # h
cycle_id = np.concatenate([i * np.ones_like(cy.t, int)                 # 0,1,2…
                           for i, cy in enumerate(sol.cycles)])

# --- one scalar per cycle straight from the summary -----------------
Qd_loop = np.array([cyc["Discharge capacity [A.h]"].entries[-1]
                    for cyc in sol.cycles])
loop_id = np.arange(1, len(Qd_loop)+1)

plt.figure()
plt.scatter(loop_id, Qd_loop, s=25)
plt.xlabel("Ageing loop #"); plt.ylabel("Discharge capacity [A·h]")

# -------- down-sample for plotting ---------------------
step       = 200             #
t_h_plot   = t_h[::step]
T_C_plot   = T_C[::step]
q_tot_plot = q_tot[::step]
q_ohm_plot = q_ohm[::step]
q_rxn_plot = q_rxn[::step]
q_rev_plot = q_rev[::step]
# ------------------------------------------------------------------
# 6.  PLOT 1 – THERMAL BEHAVIOUR
# ------------------------------------------------------------------
Qd_check = sol["Discharge capacity [A.h]"].entries
plt.figure(); plt.scatter(cycle_id, Qd_check, s=8)
plt.xlabel("Cycle"); plt.ylabel("Discharge capacity [A·h]")
# 6. THERMAL PLOT ----------------------------------------------------
figT, axT = plt.subplots(figsize=(7,4))
axT.plot(t_h_plot, T_C_plot, c="tab:red", label="Cell T")
axT.set_xlabel("Time [h]"); axT.set_ylabel("T [°C]", color="tab:red")
axT.tick_params(axis="y", labelcolor="tab:red")

axQ = axT.twinx()
axQ.plot(t_h_plot, q_ohm_plot, label="Ohmic")
axQ.plot(t_h_plot, q_rxn_plot, label="Irreversible")
axQ.plot(t_h_plot, q_rev_plot, label="Reversible")
axQ.plot(t_h_plot, q_tot_plot, c="k", lw=1.2, label="Total")
axQ.set_ylabel("Heat generation [W]")
axQ.legend(loc="upper right")
figT.tight_layout()

# 7. AGEING PLOT -----------------------------------------------------
figA, axA = plt.subplots(figsize=(7,4))
axA.plot(Qt, LLI,      label="LLI",     c="tab:blue")
axA.plot(Qt, LAM_neg,  label="LAM neg", c="tab:orange", ls="--")
axA.plot(Qt, LAM_pos,  label="LAM pos", c="tab:green",  ls=":")
axA.set_xlabel("Throughput capacity [A·h]")
axA.set_ylabel("Degradation [%]")
axA.legend()
figA.tight_layout()

plt.show()
