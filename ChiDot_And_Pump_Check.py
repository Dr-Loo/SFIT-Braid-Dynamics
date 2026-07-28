import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 16: chi_dot_RMS Decomposition and Direct A_pump(C) Check")
print("=" * 70)

g_fixed = 0.42
T_bath_values = [0.001, 0.003, 0.006, 0.008, 0.01, 0.02, 0.04]
T_total = 5000.0
N_TRIALS = 5
C_MAX = 0.2

results = []

for T_bath in T_bath_values:
    print(f"\nRunning T_bath = {T_bath}  ({N_TRIALS} trials x T = {T_total})...")

    chi_dot_rms_trials = []
    chi_rms_trials = []
    C_mean_trials = []
    A_pump_mean_trials = []
    A_pump_max_trials = []

    for trial in range(N_TRIALS):
        seed = 9000 + int(T_bath * 100000) + trial
        sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)

        # NOTE: this assumes sim.run() can also return chi and chi_dot arrays,
        # and that the solver exposes its own A_pump(C) function or a stored
        # trace of pump activation. Adjust the unpacking / attribute names
        # below to match your actual SFIT_Solver interface -- these are the
        # two things this script needs that weren't used in earlier scripts.
        t_arr, C_arr, chi_arr, chi_dot_arr = sim.run(
            T_total=T_total, sample_every=10, return_fields=True
        )

        transient_cutoff = t_arr > (5 * 3.14)
        C_steady = C_arr[transient_cutoff]
        chi_steady = chi_arr[transient_cutoff]
        chi_dot_steady = chi_dot_arr[transient_cutoff]

        chi_rms_trials.append(np.sqrt(np.mean(chi_steady**2)))
        chi_dot_rms_trials.append(np.sqrt(np.mean(chi_dot_steady**2)))
        C_mean_trials.append(np.mean(C_steady))

        # Direct A_pump(C) evaluation -- requires the solver to expose the
        # actual pump functional, not just an activity proxy. If SFIT_Solver
        # has e.g. sim.A_pump(C) or a stored self.A_pump_trace, use that
        # directly instead of recomputing from scratch below.
        if hasattr(sim, "A_pump"):
            A_pump_vals = np.array([sim.A_pump(c) for c in C_steady])
        else:
            raise AttributeError(
                "SFIT_Solver has no A_pump(C) method exposed -- "
                "add one (or a stored trace) before this check can run. "
                "This is a hard requirement, not an approximation to skip."
            )

        A_pump_mean_trials.append(np.mean(A_pump_vals))
        A_pump_max_trials.append(np.max(A_pump_vals))

    chi_rms = np.mean(chi_rms_trials)
    chi_dot_rms = np.mean(chi_dot_rms_trials)
    C_mean = np.mean(C_mean_trials)
    A_pump_mean = np.mean(A_pump_mean_trials)
    A_pump_max = np.mean(A_pump_max_trials)

    print(f"  chi_RMS      = {chi_rms:.4f}")
    print(f"  chi_dot_RMS  = {chi_dot_rms:.4f}")
    print(f"  C_loc_mean   = {C_mean:.4f}  (C_max={C_MAX})")
    print(f"  A_pump mean  = {A_pump_mean:.6f}")
    print(f"  A_pump max   = {A_pump_max:.6f}")
    if C_mean > C_MAX and A_pump_mean > 0.01:
        print("  *** WARNING: C_loc exceeds C_max but A_pump is still "
              "non-negligible -- pump-shutoff assumption may be wrong ***")

    results.append({
        "T_bath": T_bath, "chi_rms": chi_rms, "chi_dot_rms": chi_dot_rms,
        "C_mean": C_mean, "A_pump_mean": A_pump_mean, "A_pump_max": A_pump_max,
    })

# --- Power-law fits for chi_RMS and chi_dot_RMS vs T_bath ---
def power_law(T_bath, A, p):
    return A * T_bath**p

T_bath_arr = np.array([r["T_bath"] for r in results])
chi_rms_arr = np.array([r["chi_rms"] for r in results])
chi_dot_rms_arr = np.array([r["chi_dot_rms"] for r in results])

print("\n" + "=" * 70)
print("EQUIPARTITION CHECK: position-like vs momentum-like scaling")
print("=" * 70)

for name, arr in [("chi_RMS (position-like)", chi_rms_arr),
                   ("chi_dot_RMS (momentum-like)", chi_dot_rms_arr)]:
    popt, _ = curve_fit(power_law, T_bath_arr, arr, p0=[1.0, 0.5], maxfev=10000)
    print(f"  {name}: exponent p = {popt[1]:.4f} (equipartition predicts 0.5)")

print("\nIf both exponents are close to 0.5, both quadratic terms "
      "(position and momentum) scale identically with T_bath -- the "
      "complete equipartition signature. If they diverge, only one "
      "degree of freedom is thermalizing as expected.")

# --- A_pump(C) directly vs T_bath ---
print("\n" + "=" * 70)
print("PUMP-SHUTOFF CHECK")
print("=" * 70)
for r in results:
    status = "OFF (confirms pump-shutoff mechanism)" if r["A_pump_mean"] < 0.01 else "ACTIVE"
    print(f"  T_bath={r['T_bath']:<8} A_pump_mean={r['A_pump_mean']:.6f}  [{status}]")

# --- Plots ---
fig, (ax_rms, ax_pump) = plt.subplots(1, 2, figsize=(13, 5.5))

ax_rms.loglog(T_bath_arr, chi_rms_arr, "o-", label="chi_RMS (position-like)")
ax_rms.loglog(T_bath_arr, chi_dot_rms_arr, "s-", label="chi_dot_RMS (momentum-like)")
ax_rms.set_xlabel(r"$T_{bath}$")
ax_rms.set_ylabel("RMS amplitude")
ax_rms.set_title("Equipartition Check: Position vs Momentum Scaling")
ax_rms.legend()
ax_rms.grid(True, which="both", alpha=0.3)

Tb = [r["T_bath"] for r in results]
ax_pump.plot(Tb, [r["A_pump_mean"] for r in results], "o-", label="A_pump mean")
ax_pump.plot(Tb, [r["A_pump_max"] for r in results], "s--", label="A_pump max", alpha=0.6)
ax_pump.axhline(0.01, color="red", linestyle=":", alpha=0.5, label="shutoff threshold (0.01)")
ax_pump.set_xscale("log")
ax_pump.set_xlabel(r"$T_{bath}$")
ax_pump.set_ylabel("A_pump(C)")
ax_pump.set_title("Direct Pump Activation vs Noise")
ax_pump.legend()
ax_pump.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sfit_chidot_and_pump_check.png", dpi=300)
print("\nSaved: sfit_chidot_and_pump_check.png")
plt.show()
