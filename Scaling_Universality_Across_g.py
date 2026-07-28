import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 14: Scaling Law Universality Across g")
print("=" * 70)

g_values = [0.42, 0.48, 0.58, 0.65]
T_bath_values = [0.001, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.02, 0.04]
T_total = 5000.0
N_TRIALS = 5


def model_linear(T_bath, C0, k):
    return C0 + k * T_bath


def model_quadrature(T_bath, C0, k):
    return np.sqrt(C0**2 + (k * T_bath)**2)


def model_powerlaw(T_bath, A, p):
    return A * T_bath**p


models = {
    "linear": (model_linear, [0.1, 50.0]),
    "quadrature": (model_quadrature, [0.1, 50.0]),
    "power_law": (model_powerlaw, [50.0, 1.0]),
}

per_g_results = {}

for g in g_values:
    print(f"\n{'='*70}")
    print(f"g = {g:.2f}")
    print(f"{'='*70}")

    T_bath_arr = np.array(T_bath_values)
    mean_C = np.zeros(len(T_bath_values))
    std_C = np.zeros(len(T_bath_values))

    for i, T_bath in enumerate(T_bath_values):
        trial_means = []
        for trial in range(N_TRIALS):
            seed = 8000 + int(g * 1000) * 1000 + int(T_bath * 100000) + trial
            sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g, T_bath=T_bath, seed=seed)
            t_arr, C_arr = sim.run(T_total=T_total, sample_every=10)
            transient_cutoff = t_arr > (5 * 3.14)
            trial_means.append(np.mean(C_arr[transient_cutoff]))
        mean_C[i] = np.mean(trial_means)
        std_C[i] = np.std(trial_means)
        print(f"  T_bath={T_bath:<8} mean C_loc={mean_C[i]:.4f} +/- {std_C[i]:.4f}")

    fits = {}
    for name, (func, p0) in models.items():
        try:
            popt, pcov = curve_fit(func, T_bath_arr, mean_C, p0=p0, sigma=std_C,
                                    absolute_sigma=True, maxfev=10000)
            residuals = mean_C - func(T_bath_arr, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((mean_C - np.mean(mean_C))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            perr = np.sqrt(np.diag(pcov))
            fits[name] = {"popt": popt, "perr": perr, "r_squared": r_squared}
            print(f"  {name}: params={popt}, errs={perr}, R^2={r_squared:.5f}")
        except RuntimeError:
            print(f"  {name}: fit failed")

    per_g_results[g] = {
        "T_bath": T_bath_arr, "mean_C": mean_C, "std_C": std_C, "fits": fits,
    }

# --- Cross-g comparison: is k universal? ---
print("\n" + "=" * 70)
print("CROSS-g COMPARISON: LINEAR-MODEL SLOPE k(g)")
print("=" * 70)
print(f"{'g':<8}{'C0':<16}{'k':<16}{'R^2':<10}")
k_values, k_errs, g_list = [], [], []
for g in g_values:
    if "linear" in per_g_results[g]["fits"]:
        popt = per_g_results[g]["fits"]["linear"]["popt"]
        perr = per_g_results[g]["fits"]["linear"]["perr"]
        r2 = per_g_results[g]["fits"]["linear"]["r_squared"]
        print(f"{g:<8.2f}{popt[0]:<8.4f}+/-{perr[0]:<6.4f}"
              f"{popt[1]:<8.3f}+/-{perr[1]:<6.3f}{r2:<10.5f}")
        k_values.append(popt[1])
        k_errs.append(perr[1])
        g_list.append(g)

k_values = np.array(k_values)
k_errs = np.array(k_errs)
k_mean = np.average(k_values, weights=1.0 / k_errs**2)
k_spread = np.max(k_values) - np.min(k_values)
k_spread_frac = k_spread / k_mean if k_mean else float("nan")

print(f"\nWeighted mean k across g: {k_mean:.3f}")
print(f"Spread (max-min): {k_spread:.3f}  ({k_spread_frac*100:.1f}% of mean)")
if k_spread_frac < 0.10:
    print("-> k appears UNIVERSAL across g (spread < 10% of mean)")
else:
    print("-> k VARIES with g (spread >= 10% of mean) -- check for a trend, "
          "e.g. against g itself or against gamma_eff at each g's operating point")

# --- Plot: k(g) with error bars ---
fig, (ax_data, ax_k) = plt.subplots(1, 2, figsize=(14, 5.5))

for g in g_values:
    r = per_g_results[g]
    ax_data.errorbar(r["T_bath"], r["mean_C"], yerr=r["std_C"], fmt="o-",
                      label=f"g={g:.2f}", markersize=5, capsize=2)
ax_data.set_xlabel(r"$T_{bath}$")
ax_data.set_ylabel(r"Mean steady-state $C_{loc}$")
ax_data.set_title("Mean C_loc vs T_bath, by g")
ax_data.legend(fontsize=9)
ax_data.grid(True, alpha=0.3)

ax_k.errorbar(g_list, k_values, yerr=k_errs, fmt="o", markersize=9, capsize=4)
ax_k.axhline(k_mean, color="gray", linestyle="--", alpha=0.6,
             label=f"weighted mean k={k_mean:.2f}")
ax_k.set_xlabel("g")
ax_k.set_ylabel("Fitted slope k (linear model)")
ax_k.set_title("Thermal-Response Slope k vs Coupling g")
ax_k.legend()
ax_k.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sfit_scaling_universality_across_g.png", dpi=300)
print("\nSaved: sfit_scaling_universality_across_g.png")
plt.show()
