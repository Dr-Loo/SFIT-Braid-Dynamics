import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TARGETED CONVERGENCE CHECK: Peak Amplitude Statistics at High Noise")
print("Condition: g=0.42, T_bath=0.04")
print("Comparing dt=0.005 vs dt=0.0025")
print("=" * 70)

g_fixed = 0.42
T_bath = 0.04
T_total = 2000.0  # Long enough for robust steady-state statistics
N_TRIALS = 5
seed_base = 4200

# EXACT parameters from the original amplitude script (Test 11)
PEAK_DISTANCE_TIME = 2.5
PEAK_PROMINENCE = 0.03

dt_values = [0.005, 0.0025]
results = {}

for dt in dt_values:
    print(f"\nRunning with dt = {dt}...")
    all_peaks_amp = []
    trial_peak_counts = []
    
    for trial in range(N_TRIALS):
        seed = seed_base + trial
        sim = SFIT_Solver(N=256, L=50.0, dt=dt, g=g_fixed, T_bath=T_bath, seed=seed)
        # sample_every=2 ensures dt_sample = 0.01 for dt=0.005, and 0.005 for dt=0.0025
        t_arr, C_arr = sim.run(T_total=T_total, sample_every=2)
        dt_sample = t_arr[1] - t_arr[0]
        
        # Discard transient (first 200 time units)
        mask = np.array(t_arr) > 200.0
        C_steady = C_arr[mask]
        
        # EXACT find_peaks parameters from Test 11
        distance_samples = max(1, int(PEAK_DISTANCE_TIME / dt_sample))
        peaks, _ = find_peaks(C_steady, prominence=PEAK_PROMINENCE, distance=distance_samples)
        
        if len(peaks) > 5:
            all_peaks_amp.extend(C_steady[peaks].tolist())
            trial_peak_counts.append(len(peaks))
        else:
            print(f"  Warning: Trial {trial} had too few peaks ({len(peaks)}).")

    results[dt] = {
        "global_mean_amp": np.mean(all_peaks_amp),
        "global_std_amp": np.std(all_peaks_amp),
        "mean_peak_count": np.mean(trial_peak_counts)
    }

    print(f"  Global Mean Peak Amplitude: {results[dt]['global_mean_amp']:.4f}")
    print(f"  Global Std Peak Amplitude:  {results[dt]['global_std_amp']:.4f}")
    print(f"  Mean Peaks per Trial:       {results[dt]['mean_peak_count']:.1f}")

# Comparison
dt1, dt2 = 0.005, 0.0025
mean_ratio = results[dt1]['global_mean_amp'] / results[dt2]['global_mean_amp']
std_ratio = results[dt1]['global_std_amp'] / results[dt2]['global_std_amp']

print("\n" + "=" * 70)
print("CONVERGENCE ANALYSIS:")
print("=" * 70)
print(f"Mean Peak Amplitude Ratio (dt=0.005 / dt=0.0025): {mean_ratio:.3f}")
print(f"Std Peak Amplitude Ratio   (dt=0.005 / dt=0.0025): {std_ratio:.3f}")

if 0.90 <= mean_ratio <= 1.10 and 0.85 <= std_ratio <= 1.15:
    print("\n✓ PASS: Peak amplitude statistics are converged within ~10%.")
    print("The dt=0.005 results are trustworthy for Paper 5.")
else:
    print("\n✗ FAIL: Significant divergence in peak amplitude statistics.")
    print("The high-noise regime requires dt=0.0025 (or smaller) for accurate results.")

# Visual check: overlay one trial from each dt (first 500 time units for clarity)
print("\nGenerating visual overlay...")
sim1 = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed_base)
t1, C1 = sim1.run(T_total=500.0, sample_every=2) 

sim2 = SFIT_Solver(N=256, L=50.0, dt=0.0025, g=g_fixed, T_bath=T_bath, seed=seed_base)
t2, C2 = sim2.run(T_total=500.0, sample_every=4) # sample_every=4 matches dt_sample=0.01

plt.figure(figsize=(10, 5))
plt.plot(t1, C1, linewidth=1, alpha=0.7, label=f'dt = 0.005 (Mean Peak = {results[0.005]["global_mean_amp"]:.3f})')
plt.plot(t2, C2, linewidth=1, alpha=0.7, label=f'dt = 0.0025 (Mean Peak = {results[0.0025]["global_mean_amp"]:.3f})')

plt.xlabel('Time $t$', fontsize=12)
plt.ylabel(r'$C_{\rm loc}(t)$', fontsize=12)
plt.title(f'Targeted Convergence: High Noise ($T_{{\text{{bath}}}} = {T_bath}$)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sfit_targeted_amplitude_convergence.png", dpi=300)
print("✓ Saved: sfit_targeted_amplitude_convergence.png")
plt.show()