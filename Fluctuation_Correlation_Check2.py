import numpy as np
import matplotlib.pyplot as plt
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 20: Direct std(C_loc)/mean(C_loc) and Field Correlation Length")
print("=" * 70)

g_fixed = 0.42
T_bath_values = [0.006, 0.008, 0.010, 0.020, 0.040]  # spans weak -> strong damping
T_total = 5000.0
N_TRIALS = 5
C_MAX = 0.2
GAMMA_0, BETA_DAMP, P_DAMP = 0.02, 2.0, 2.0
OMEGA = 1.0
M_CHI = 0.85

def gamma_eff(C):
    return GAMMA_0 * (1 + BETA_DAMP * (C / C_MAX) ** P_DAMP)

def spatial_autocorr(chi_snapshot, dx):
    """Normalized spatial autocorrelation of a single field snapshot."""
    c = chi_snapshot - np.mean(chi_snapshot)
    n = len(c)
    full = np.correlate(c, c, mode="full")
    ac = full[n - 1:] / full[n - 1]  # normalize to ac[0] = 1
    lags = np.arange(len(ac)) * dx
    return lags, ac

results = []

for T_bath in T_bath_values:
    print(f"\nRunning g={g_fixed:.2f}, T_bath={T_bath} ({N_TRIALS} trials)...")
    C_series_all = []
    corr_lengths = []
    
    for trial in range(N_TRIALS):
        seed = 10000 + int(T_bath * 100000) + trial
        sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)
        
        # FIX: use return_fields=True and unpack all 5 returned arrays
        t_arr, C_arr, chi_snapshots, chi_dot_snapshots, pump_data = sim.run(
            T_total=T_total, sample_every=10, return_fields=True
        )
        
        dx = 50.0 / 256
        transient_cutoff = t_arr > (5 * 3.14)
        C_steady = C_arr[transient_cutoff]
        C_series_all.append(C_steady)
        
        # Correlation length from a handful of steady-state snapshots,
        # fit to the first 1/e crossing of the spatial autocorrelation.
        snap_idxs = np.where(transient_cutoff)[0][::max(1, len(chi_snapshots) // 20)]
        for idx in snap_idxs[:10]:
            lags, ac = spatial_autocorr(chi_snapshots[idx], dx)
            below = np.where(ac < 1 / np.e)[0]
            if len(below) > 0:
                corr_lengths.append(lags[below[0]])
                
    C_all = np.concatenate(C_series_all)
    mean_C = np.mean(C_all)
    std_C = np.std(C_all)
    rel_fluct = std_C / mean_C
    mean_corr_length = np.mean(corr_lengths) if corr_lengths else float("nan")
    
    dx = 50.0 / 256
    corr_length_gridpoints = mean_corr_length / dx if not np.isnan(mean_corr_length) else float("nan")
    N_win = 76
    N_eff = N_win / corr_length_gridpoints if (not np.isnan(corr_length_gridpoints) and corr_length_gridpoints > 0) else float("nan")
    g_eff = gamma_eff(mean_C)
    g_eff_over_omega = g_eff / OMEGA
    
    print(f"  mean C_loc = {mean_C:.4f}, std C_loc = {std_C:.4f}")
    print(f"  MEASURED std/mean = {rel_fluct:.4f}  (vs idealized 1/sqrt(76) = {1/np.sqrt(76):.4f})")
    print(f"  measured correlation length = {mean_corr_length:.3f} ({corr_length_gridpoints:.2f} grid points)")
    
    if not np.isnan(N_eff):
        print(f"  effective N (76 / corr_length_gridpoints) = {N_eff:.2f} -> 1/sqrt(N_eff) = {1/np.sqrt(N_eff):.4f}")
    else:
        print("  N_eff: n/a")
        
    print(f"  gamma_eff = {g_eff:.3f}, gamma_eff/omega = {g_eff_over_omega:.3f}")
    
    results.append({
        "T_bath": T_bath, "mean_C": mean_C, "std_C": std_C, "rel_fluct": rel_fluct,
        "corr_length_gridpoints": corr_length_gridpoints, "N_eff": N_eff,
        "gamma_eff_over_omega": g_eff_over_omega,
    })

print("\n" + "=" * 70)
print("SUMMARY: measured relative fluctuation vs damping regime")
print("=" * 70)
print(f"{'T_bath':<10}{'std/mean':<12}{'1/sqrt(76)':<14}{'1/sqrt(N_eff)':<16}{'gamma/omega':<12}")
for r in results:
    n_eff_term = f"{1/np.sqrt(r['N_eff']):.4f}" if not np.isnan(r["N_eff"]) else "n/a"
    print(f"{r['T_bath']:<10.3f}{r['rel_fluct']:<12.4f}{1/np.sqrt(76):<14.4f}"
          f"{n_eff_term:<16}{r['gamma_eff_over_omega']:<12.3f}")

fig, ax = plt.subplots(figsize=(8, 6))
Tb = [r["T_bath"] for r in results]
ax.plot(Tb, [r["rel_fluct"] for r in results], "o-", label="Measured std(C_loc)/mean(C_loc)")
ax.axhline(1/np.sqrt(76), color="gray", linestyle="--", label="Idealized 1/sqrt(76)=0.11")
ax2 = ax.twinx()
ax2.plot(Tb, [r["gamma_eff_over_omega"] for r in results], "s--", color="tab:red",
         label="gamma_eff/omega")
ax2.axhline(1.0, color="tab:red", linestyle=":", alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel(r"$T_{bath}$")
ax.set_ylabel("Relative fluctuation of C_loc")
ax2.set_ylabel(r"$\gamma_{eff}/\omega$", color="tab:red")
ax.set_title("Measured Fluctuation Size vs Damping Crossover")
ax.legend(loc="upper left", fontsize=9)
ax2.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sfit_fluctuation_vs_damping.png", dpi=300)
print("\nSaved: sfit_fluctuation_vs_damping.png")
plt.show()