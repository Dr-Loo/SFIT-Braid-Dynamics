import numpy as np
import matplotlib.pyplot as plt
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 12: Fine-Grained T_bath Sweep -- Locating Confinement Breakdown")
print("=" * 70)

g_fixed = 0.42
# Coarse-sweep values already known, plus the new fine-grained points between
# 0.003 (still near baseline) and 0.01 (already 3-4x baseline).
T_bath_values = [0.001, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.02, 0.04]
T_total = 5000.0
N_TRIALS = 5

C_MAX = 0.2
BASELINE_AMPLITUDE = 0.1892  # deterministic skeleton at g=0.42, T_bath=0
DEPARTURE_THRESHOLD = 0.20   # flag any condition >20% above baseline as "departed"

results = []

for T_bath in T_bath_values:
    print(f"\nRunning g = {g_fixed:.2f}, T_bath = {T_bath}  "
          f"({N_TRIALS} trials x T = {T_total})...")

    trial_means = []
    trial_maxes = []

    for trial in range(N_TRIALS):
        seed = 6000 + int(T_bath * 100000) + trial
        sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)
        t_arr, C_arr = sim.run(T_total=T_total, sample_every=10)

        # Discard an initial transient (first ~5 periods) before computing
        # steady-state statistics, consistent with the earlier convergence
        # checks.
        transient_cutoff = t_arr > (5 * 3.14)
        C_steady = C_arr[transient_cutoff]

        trial_means.append(np.mean(C_steady))
        trial_maxes.append(np.max(C_steady))

    mean_of_means = np.mean(trial_means)
    std_of_means = np.std(trial_means)
    mean_of_maxes = np.mean(trial_maxes)

    frac_above_baseline = (mean_of_means - BASELINE_AMPLITUDE) / BASELINE_AMPLITUDE
    departed = frac_above_baseline > DEPARTURE_THRESHOLD

    print(f"  mean C_loc:  {mean_of_means:.4f} +/- {std_of_means:.4f}")
    print(f"  max C_loc (avg across trials): {mean_of_maxes:.4f}")
    print(f"  fractional departure from baseline: {frac_above_baseline:+.3f}")
    print(f"  vs C_max ceiling (0.2): "
          f"{'WITHIN' if mean_of_means <= C_MAX * 1.1 else 'EXCEEDS'}")
    if departed:
        print(f"  *** DEPARTED from confined regime "
              f"(>{DEPARTURE_THRESHOLD*100:.0f}% above baseline) ***")

    results.append({
        "T_bath": T_bath,
        "mean": mean_of_means,
        "std": std_of_means,
        "max": mean_of_maxes,
        "departed": departed,
    })

# --- Locate the transition: first T_bath where departure flag trips ---
print("\n" + "=" * 70)
print("TRANSITION LOCATION")
print("=" * 70)
departed_values = [r["T_bath"] for r in results if r["departed"]]
confined_values = [r["T_bath"] for r in results if not r["departed"]]
if departed_values and confined_values:
    transition_lower = max(confined_values)
    transition_upper = min(departed_values)
    print(f"  Confinement holds up through T_bath = {transition_lower}")
    print(f"  Departure confirmed by T_bath = {transition_upper}")
    print(f"  -> Transition lies in ({transition_lower}, {transition_upper})")
elif not departed_values:
    print("  No departure detected across the tested range.")
else:
    print("  All tested points show departure -- extend sweep to lower T_bath.")

# --- Plot: mean amplitude vs T_bath, with baseline and C_max reference lines ---
fig, ax = plt.subplots(figsize=(9, 6))
Tb = [r["T_bath"] for r in results]
means = [r["mean"] for r in results]
stds = [r["std"] for r in results]
colors = ["tab:red" if r["departed"] else "tab:blue" for r in results]

ax.errorbar(Tb, means, yerr=stds, fmt="none", ecolor="gray", alpha=0.5, zorder=1)
ax.scatter(Tb, means, c=colors, s=80, zorder=2)
ax.axhline(BASELINE_AMPLITUDE, color="green", linestyle="--", alpha=0.6,
           label=f"Deterministic baseline ({BASELINE_AMPLITUDE})")
ax.axhline(C_MAX, color="black", linestyle=":", alpha=0.6,
           label=f"C_max ceiling ({C_MAX})")
ax.axhline(BASELINE_AMPLITUDE * (1 + DEPARTURE_THRESHOLD), color="orange",
           linestyle="--", alpha=0.5,
           label=f"Departure threshold (+{DEPARTURE_THRESHOLD*100:.0f}%)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$T_{bath}$")
ax.set_ylabel(r"Mean steady-state $C_{loc}$")
ax.set_title(f"Locating the Confinement-Breakdown Transition (g={g_fixed})")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("sfit_confinement_transition.png", dpi=300)
print("\nSaved: sfit_confinement_transition.png")

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'T_bath':<10}{'Mean C_loc':<14}{'Max C_loc':<14}{'Status':<12}")
for r in results:
    status = "DEPARTED" if r["departed"] else "confined"
    print(f"{r['T_bath']:<10}{r['mean']:<14.4f}{r['max']:<14.4f}{status:<12}")

plt.show()
