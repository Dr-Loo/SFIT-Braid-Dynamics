import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 13: Curve Fitting -- Mean C_loc vs T_bath Scaling Law")
print("=" * 70)

g_fixed = 0.42
T_bath_values = [0.001, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.02, 0.04]
T_total = 5000.0
N_TRIALS = 5

BASELINE_AMPLITUDE = 0.1892  # deterministic skeleton, g=0.42, T_bath=0

# --- Reuse the same measurement procedure as the transition sweep ---
T_bath_arr = np.array(T_bath_values)
mean_C = np.zeros(len(T_bath_values))
std_C = np.zeros(len(T_bath_values))

for i, T_bath in enumerate(T_bath_values):
    print(f"\nRunning g = {g_fixed:.2f}, T_bath = {T_bath}  "
          f"({N_TRIALS} trials x T = {T_total})...")
    trial_means = []
    for trial in range(N_TRIALS):
        seed = 7000 + int(T_bath * 100000) + trial
        sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)
        t_arr, C_arr = sim.run(T_total=T_total, sample_every=10)
        transient_cutoff = t_arr > (5 * 3.14)
        trial_means.append(np.mean(C_arr[transient_cutoff]))
    mean_C[i] = np.mean(trial_means)
    std_C[i] = np.std(trial_means)
    print(f"  mean C_loc: {mean_C[i]:.4f} +/- {std_C[i]:.4f}")

# --- Candidate model 1: simple offset + linear ---
def model_linear(T_bath, C0, k):
    return C0 + k * T_bath

# --- Candidate model 2: quadrature sum of deterministic floor + linear thermal term ---
def model_quadrature(T_bath, C0, k):
    return np.sqrt(C0**2 + (k * T_bath)**2)

# --- Candidate model 3: power law, for comparison (no assumed floor) ---
def model_powerlaw(T_bath, A, p):
    return A * T_bath**p

models = {
    "linear (C0 + k*T)": (model_linear, [BASELINE_AMPLITUDE, 50.0]),
    "quadrature (sqrt(C0^2 + (k*T)^2))": (model_quadrature, [BASELINE_AMPLITUDE, 50.0]),
    "power law (A * T^p)": (model_powerlaw, [1.0, 0.5]),
}

print("\n" + "=" * 70)
print("CURVE FITS")
print("=" * 70)

fit_results = {}
for name, (func, p0) in models.items():
    try:
        popt, pcov = curve_fit(func, T_bath_arr, mean_C, p0=p0, sigma=std_C,
                                absolute_sigma=True, maxfev=10000)
        residuals = mean_C - func(T_bath_arr, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((mean_C - np.mean(mean_C))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        perr = np.sqrt(np.diag(pcov))

        print(f"\n{name}:")
        print(f"  parameters: {[f'{p:.4f} +/- {e:.4f}' for p, e in zip(popt, perr)]}")
        print(f"  R^2 = {r_squared:.5f}")
        print(f"  residual std = {np.std(residuals):.5f}")

        fit_results[name] = {
            "func": func, "popt": popt, "r_squared": r_squared, "residuals": residuals,
        }
    except RuntimeError as e:
        print(f"\n{name}: fit failed ({e})")

# --- Identify best fit by R^2 ---
if fit_results:
    best_name = max(fit_results, key=lambda k: fit_results[k]["r_squared"])
    print(f"\nBest fit by R^2: {best_name} "
          f"(R^2 = {fit_results[best_name]['r_squared']:.5f})")

# --- Plots: data + all fits, and residuals ---
fig, (ax_fit, ax_resid) = plt.subplots(1, 2, figsize=(14, 5.5))

ax_fit.errorbar(T_bath_arr, mean_C, yerr=std_C, fmt="o", color="black",
                 label="Data", markersize=7, zorder=5)
T_smooth = np.linspace(T_bath_arr.min(), T_bath_arr.max(), 300)
for name, res in fit_results.items():
    ax_fit.plot(T_smooth, res["func"](T_smooth, *res["popt"]),
                label=f"{name} (R²={res['r_squared']:.4f})", linewidth=1.8)
ax_fit.axhline(BASELINE_AMPLITUDE, color="green", linestyle=":", alpha=0.5,
               label="Deterministic baseline")
ax_fit.set_xlabel(r"$T_{bath}$")
ax_fit.set_ylabel(r"Mean steady-state $C_{loc}$")
ax_fit.set_title(f"Scaling Law Fits (g={g_fixed})")
ax_fit.legend(fontsize=8)
ax_fit.grid(True, alpha=0.3)

for name, res in fit_results.items():
    ax_resid.plot(T_bath_arr, res["residuals"], "o-", label=name, markersize=6)
ax_resid.axhline(0, color="gray", linestyle="--", alpha=0.6)
ax_resid.set_xlabel(r"$T_{bath}$")
ax_resid.set_ylabel("Residual (data - fit)")
ax_resid.set_title("Fit Residuals")
ax_resid.legend(fontsize=8)
ax_resid.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sfit_scaling_law_fits.png", dpi=300)
print("\nSaved: sfit_scaling_law_fits.png")

# --- Log-log plot, useful for distinguishing power-law vs linear-with-offset ---
fig2, ax_log = plt.subplots(figsize=(7, 5.5))
ax_log.loglog(T_bath_arr, mean_C, "o", color="black", markersize=7, label="Data")
for name, res in fit_results.items():
    ax_log.loglog(T_smooth, res["func"](T_smooth, *res["popt"]),
                  label=name, linewidth=1.5)
ax_log.set_xlabel(r"$T_{bath}$ (log)")
ax_log.set_ylabel(r"Mean $C_{loc}$ (log)")
ax_log.set_title("Log-Log View")
ax_log.legend(fontsize=8)
ax_log.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("sfit_scaling_law_loglog.png", dpi=300)
print("Saved: sfit_scaling_law_loglog.png")

plt.show()
