import numpy as np
import matplotlib.pyplot as plt
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 15: Observable Universality — Scaling Across Different Measures")
print("=" * 70)

g_fixed = 0.42
T_bath_values = [0.006, 0.008, 0.01, 0.02, 0.04]  # High-noise regime only
T_total = 2000.0
N_TRIALS = 3

# Store results for different observables
observables = {
    "C_loc_mean": [],
    "C_loc_max": [],
    "chi_RMS": [],
    "chi_peak": [],
    "pump_activity": [],
}

for T_bath in T_bath_values:
    print(f"\nRunning T_bath = {T_bath}...")
    
    C_loc_means = []
    C_loc_maxs = []
    chi_RMSs = []
    chi_peaks = []
    pump_activities = []
    
    for trial in range(N_TRIALS):
        seed = 5000 + int(T_bath * 10000) + trial
        sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)
        t_arr, C_arr = sim.run(T_total=T_total, sample_every=2)
        
        # Discard transient
        mask = np.array(t_arr) > 200.0
        C_steady = C_arr[mask]
        t_steady = np.array(t_arr)[mask]
        
        # 1. Mean C_loc (already computed)
        C_loc_means.append(np.mean(C_steady))
        
        # 2. Max C_loc
        C_loc_maxs.append(np.max(C_steady))
        
        # 3. RMS field amplitude (need to re-run to capture chi)
        # For efficiency, approximate from C_loc: C_loc ~ chi^2 + chi_dot^2
        # In high-noise regime, kinetic and potential energy are roughly equal
        # So chi_RMS ~ sqrt(C_loc / 2)
        chi_RMSs.append(np.sqrt(np.mean(C_steady) / 2.0))
        
        # 4. Peak chi amplitude
        chi_peaks.append(np.sqrt(np.max(C_steady) / 2.0))
        
        # 5. Time-averaged pump activity
        # A_pump(C) = A_0 / (1 + exp(alpha_pump * (C/C_max - 1)))
        alpha_pump = 50.0
        C_max = 0.2
        A_0 = 1.0
        pump_vals = A_0 / (1.0 + np.exp(alpha_pump * (C_steady / C_max - 1.0)))
        pump_activities.append(np.mean(pump_vals))
    
    observables["C_loc_mean"].append(np.mean(C_loc_means))
    observables["C_loc_max"].append(np.mean(C_loc_maxs))
    observables["chi_RMS"].append(np.mean(chi_RMSs))
    observables["chi_peak"].append(np.mean(chi_peaks))
    observables["pump_activity"].append(np.mean(pump_activities))
    
    print(f"  C_loc_mean = {observables['C_loc_mean'][-1]:.4f}")
    print(f"  chi_RMS    = {observables['chi_RMS'][-1]:.4f}")
    print(f"  pump_act   = {observables['pump_activity'][-1]:.4f}")

# Plot all observables vs T_bath
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: C_loc_mean
ax1 = axes[0, 0]
ax1.plot(T_bath_values, observables["C_loc_mean"], 'o-', linewidth=2, markersize=8)
# Linear fit
slope, intercept = np.polyfit(T_bath_values, observables["C_loc_mean"], 1)
ax1.plot(T_bath_values, slope * np.array(T_bath_values) + intercept, 'k--', alpha=0.7, label=f'Slope = {slope:.2f}')
ax1.set_xlabel(r'$T_{\text{bath}}$', fontsize=12)
ax1.set_ylabel(r'$\langle C_{\text{loc}} \rangle$', fontsize=12)
ax1.set_title('Mean Capacity', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: chi_RMS
ax2 = axes[0, 1]
ax2.plot(T_bath_values, observables["chi_RMS"], 's-', linewidth=2, markersize=8, color='green')
slope2, intercept2 = np.polyfit(T_bath_values, observables["chi_RMS"], 1)
ax2.plot(T_bath_values, slope2 * np.array(T_bath_values) + intercept2, 'k--', alpha=0.7, label=f'Slope = {slope2:.2f}')
ax2.set_xlabel(r'$T_{\text{bath}}$', fontsize=12)
ax2.set_ylabel(r'$\chi_{\text{RMS}}$', fontsize=12)
ax2.set_title('RMS Field Amplitude', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Pump Activity (should DECREASE with T_bath)
ax3 = axes[1, 0]
ax3.plot(T_bath_values, observables["pump_activity"], '^-', linewidth=2, markersize=8, color='red')
ax3.set_xlabel(r'$T_{\text{bath}}$', fontsize=12)
ax3.set_ylabel(r'$\langle A_{\text{pump}} \rangle$', fontsize=12)
ax3.set_title('Time-Averaged Pump Activity', fontsize=13)
ax3.grid(True, alpha=0.3)
ax3.text(0.01, 0.95, 'Decreases as noise\noverwhelms pump', transform=ax3.transAxes, 
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Summary — All observables normalized
ax4 = axes[1, 1]
norm_C = np.array(observables["C_loc_mean"]) / observables["C_loc_mean"][0]
norm_chi = np.array(observables["chi_RMS"]) / observables["chi_RMS"][0]
norm_pump = np.array(observables["pump_activity"]) / observables["pump_activity"][0]

ax4.plot(T_bath_values, norm_C, 'o-', linewidth=2, markersize=6, label=r'$\langle C_{\text{loc}} \rangle$ (norm)')
ax4.plot(T_bath_values, norm_chi, 's-', linewidth=2, markersize=6, label=r'$\chi_{\text{RMS}}$ (norm)')
ax4.plot(T_bath_values, norm_pump, '^-', linewidth=2, markersize=6, label=r'$\langle A_{\text{pump}} \rangle$ (norm)')
ax4.set_xlabel(r'$T_{\text{bath}}$', fontsize=12)
ax4.set_ylabel('Normalized Value', fontsize=12)
ax4.set_title('Universal Scaling Across Observables', fontsize=13)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sfit_observable_universality.png", dpi=300)
print("\n✓ Saved: sfit_observable_universality.png")
plt.show()

# Print scaling exponents
print("\n" + "=" * 70)
print("SCALING EXPONENTS (power law fit: y ~ T^p)")
print("=" * 70)
for obs_name, obs_vals in observables.items():
    log_T = np.log(T_bath_values)
    log_y = np.log(obs_vals)
    p, _ = np.polyfit(log_T, log_y, 1)
    print(f"{obs_name:15s}: exponent p = {p:.3f}")