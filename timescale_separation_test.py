import numpy as np
import matplotlib.pyplot as plt
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 19: Damping Crossover Analysis (CORRECTED)")
print("Splitting at weak (T_bath < 0.015) vs strong (T_bath > 0.015) damping")
print("=" * 70)

g_fixed = 0.42
T_bath_weak = [0.006, 0.008, 0.01]
T_bath_strong = [0.02, 0.04]
T_total = 2000.0
N_TRIALS = 3

def compute_stats(T_bath_values, label):
    print(f"\n{label} regime:")
    print("-" * 70)
    
    C_means = []
    chi2_means = []
    chi4_means = []
    
    for T_bath in T_bath_values:
        C_list = []
        chi2_list = []
        chi4_list = []
        
        for trial in range(N_TRIALS):
            seed = 9000 + int(T_bath * 10000) + trial
            sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)
            
            n_steps = int(T_total / sim.dt)
            sample_every = 10
            
            for step in range(n_steps):
                t = step * sim.dt
                sim.step(t)
                if step % sample_every == 0 and t > 200.0:
                    # FIX: Pass full arrays to compute_C_loc, which handles windowing internally
                    C_loc = sim.compute_C_loc(sim.chi, sim.chi_dot)
                    
                    # Then extract windowed field for moment calculations
                    chi_win = sim.chi[sim.idx_min:sim.idx_max]
                    chi2 = np.mean(chi_win**2)
                    chi4 = np.mean(chi_win**4)
                    
                    C_list.append(C_loc)
                    chi2_list.append(chi2)
                    chi4_list.append(chi4)
        
        C_mean = np.mean(C_list)
        chi2_mean = np.mean(chi2_list)
        chi4_mean = np.mean(chi4_list)
        
        C_means.append(C_mean)
        chi2_means.append(chi2_mean)
        chi4_means.append(chi4_mean)
        
        # Compute γ_eff
        gamma_eff = 0.02 * (1 + 2 * (C_mean / 0.2)**2)
        ratio = chi4_mean / (3 * chi2_mean**2)
        
        print(f"  T_bath={T_bath:.3f}: ⟨C_loc⟩={C_mean:.4f}, γ_eff={gamma_eff:.3f}, "
              f"γ_eff/ω={gamma_eff:.3f}, ⟨χ⁴⟩/(3⟨χ²⟩²)={ratio:.4f}")
    
    # Fit linear scaling C = C0 + k*T
    T_arr = np.array(T_bath_values)
    C_arr = np.array(C_means)
    
    if len(T_arr) >= 2:
        k, C0 = np.polyfit(T_arr, C_arr, 1)
        
        # Compute R²
        C_pred = k * T_arr + C0
        SS_res = np.sum((C_arr - C_pred)**2)
        SS_tot = np.sum((C_arr - np.mean(C_arr))**2)
        R2 = 1 - SS_res / SS_tot if SS_tot > 0 else 0.0
        
        print(f"\n  Linear fit: k = {k:.3f}, C0 = {C0:.4f}, R² = {R2:.6f}")
        
        # Compute Gaussian moment ratio
        ratios = np.array([chi4_means[i] / (3 * chi2_means[i]**2) for i in range(len(T_bath_values))])
        ratio_mean = np.mean(ratios)
        ratio_std = np.std(ratios)
        print(f"  Gaussian moment ratio: {ratio_mean:.4f} ± {ratio_std:.4f}")
        
        return k, C0, R2, ratio_mean, ratio_std, T_arr, C_arr, ratios
    else:
        print(f"\n  Not enough points for linear fit")
        return None, None, None, None, None, None, None, None

# Analyze weakly-damped regime
res_weak = compute_stats(T_bath_weak, "WEAKLY-DAMPED")
k_weak, C0_weak, R2_weak, ratio_weak_mean, ratio_weak_std, T_weak, C_weak, ratios_weak = res_weak

# Analyze strongly-damped regime
res_strong = compute_stats(T_bath_strong, "STRONGLY-DAMPED")
k_strong, C0_strong, R2_strong, ratio_strong_mean, ratio_strong_std, T_strong, C_strong, ratios_strong = res_strong

print("\n" + "=" * 70)
print("COMPARISON ACROSS DAMPING CROSSOVER")
print("=" * 70)

if k_weak is not None and k_strong is not None:
    k_diff = abs(k_weak - k_strong)
    k_avg = (k_weak + k_strong) / 2
    k_pct_diff = 100 * k_diff / k_avg if k_avg != 0 else 0
    
    ratio_diff = abs(ratio_weak_mean - ratio_strong_mean)
    ratio_avg = (ratio_weak_mean + ratio_strong_mean) / 2
    ratio_pct_diff = 100 * ratio_diff / ratio_avg if ratio_avg != 0 else 0
    
    print(f"Slope k:")
    print(f"  Weakly-damped:  {k_weak:.3f}")
    print(f"  Strongly-damped: {k_strong:.3f}")
    print(f"  Difference: {k_diff:.3f} ({k_pct_diff:.2f}%)")
    
    print(f"\nGaussian moment ratio ⟨χ⁴⟩/(3⟨χ²⟩²):")
    print(f"  Weakly-damped:  {ratio_weak_mean:.4f} ± {ratio_weak_std:.4f}")
    print(f"  Strongly-damped: {ratio_strong_mean:.4f} ± {ratio_strong_std:.4f}")
    print(f"  Difference: {ratio_diff:.4f} ({ratio_pct_diff:.2f}%)")
    
    if k_pct_diff < 5 and ratio_pct_diff < 1:
        print("\n✓ UNIVERSALITY CONFIRMED: k and Gaussian statistics are statistically")
        print("  indistinguishable across the damping crossover.")
        print("  This is a STRONGER result than the timescale-separation argument explains.")
    else:
        print("\n✗ REGIME DEPENDENCE DETECTED: k or Gaussian statistics differ across")
        print("  the damping crossover. The mechanism may be regime-dependent.")

# Plot the split
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: C_loc vs T_bath with split fits
ax1.plot(T_weak, C_weak, 'o', color='blue', markersize=8, label='Weakly-damped')
ax1.plot(T_strong, C_strong, 's', color='red', markersize=8, label='Strongly-damped')

if k_weak is not None:
    T_fit_weak = np.linspace(0.006, 0.01, 100)
    ax1.plot(T_fit_weak, k_weak * T_fit_weak + C0_weak, 'b--', linewidth=2, 
             label=f'Weak fit: k={k_weak:.2f}')

if k_strong is not None:
    T_fit_strong = np.linspace(0.02, 0.04, 100)
    ax1.plot(T_fit_strong, k_strong * T_fit_strong + C0_strong, 'r--', linewidth=2,
             label=f'Strong fit: k={k_strong:.2f}')

ax1.axvline(0.015, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Damping crossover')
ax1.set_xlabel(r'$T_{\text{bath}}$', fontsize=12)
ax1.set_ylabel(r'$\langle C_{\text{loc}} \rangle$', fontsize=12)
ax1.set_title('Linear Scaling Across Damping Crossover', fontsize=13)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Gaussian moment ratio (FIXED plotting logic)
ax2.errorbar(T_weak, ratios_weak, yerr=ratio_weak_std, fmt='o', color='blue', markersize=8, 
             label=f'Weak: {ratio_weak_mean:.4f}±{ratio_weak_std:.4f}')
ax2.errorbar(T_strong, ratios_strong, yerr=ratio_strong_std, fmt='s', color='red', markersize=8,
             label=f'Strong: {ratio_strong_mean:.4f}±{ratio_strong_std:.4f}')
ax2.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Gaussian prediction')
ax2.axvline(0.015, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax2.set_xlabel(r'$T_{\text{bath}}$', fontsize=12)
ax2.set_ylabel(r'$\langle\chi^4\rangle / (3\langle\chi^2\rangle^2)$', fontsize=12)
ax2.set_title('Gaussian Statistics Across Damping Crossover', fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.95, 1.05)

plt.tight_layout()
plt.savefig("sfit_damping_crossover_analysis.png", dpi=300)
print("\n✓ Saved: sfit_damping_crossover_analysis.png")
plt.show()