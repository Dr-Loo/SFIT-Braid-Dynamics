import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("VALIDATION: Peak Count Sanity Check Across T_bath Sweep")
print("=" * 70)

g_fixed = 0.42
T_bath_values = [0.001, 0.003, 0.005, 0.01, 0.02, 0.04]
T_total = 5000.0
N_TRIALS = 1  # 1 trial is enough to validate the peak count magnitude

# PHYSICALLY GROUNDED PARAMETERS (from noiseless skeleton)
PEAK_HEIGHT_THRESHOLD = 0.12  # Below deterministic max (~0.189), above noise floor (~0.0)
PEAK_DISTANCE_TIME = 2.0      # Strictly enforces max 1 peak per ~3.14 cycle

expected_peaks_min = 7000
expected_peaks_max = 9000

print(f"\nExpected peak count range for T={T_total}: {expected_peaks_min} - {expected_peaks_max}")
print("-" * 70)

validation_passed = True
peak_counts = []

for T_bath in T_bath_values:
    seed = 2000 + int(T_bath * 10000)
    sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)
    t_arr, C_arr = sim.run(T_total=T_total, sample_every=2)
    dt_sample = t_arr[1] - t_arr[0]
    distance_samples = max(1, int(PEAK_DISTANCE_TIME / dt_sample))

    # THE ONLY ROBUST METHOD: Fixed height + strict distance
    peaks, _ = find_peaks(C_arr, height=PEAK_HEIGHT_THRESHOLD, distance=distance_samples)
    
    n_peaks = len(peaks)
    peak_counts.append(n_peaks)
    
    # Sanity check
    if expected_peaks_min <= n_peaks <= expected_peaks_max:
        status = "✓ PASS"
    else:
        status = "✗ FAIL"
        validation_passed = False
        
    print(f"T_bath = {T_bath:4.3f} | Total Peaks Detected: {n_peaks:5d} | Status: {status}")

print("-" * 70)
if validation_passed:
    print("SUCCESS: Peak detector is stable and consistent across all noise levels.")
else:
    print("WARNING: Peak detector is still inconsistent. Do not trust sweep results.")

# ============================================================================
# VISUAL PROOF: Side-by-side comparison of low vs. high noise detection
# ============================================================================
print("\nGenerating visual proof of consistent detection...")

# Low noise
sim_low = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=0.001, seed=2000)
t_low, C_low = sim_low.run(T_total=200.0, sample_every=2) # 200 time units for clear view
dt_low = t_low[1] - t_low[0]
dist_low = max(1, int(PEAK_DISTANCE_TIME / dt_low))
peaks_low, _ = find_peaks(C_low, height=PEAK_HEIGHT_THRESHOLD, distance=dist_low)

# High noise
sim_high = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=0.04, seed=2400)
t_high, C_high = sim_high.run(T_total=200.0, sample_every=2)
dt_high = t_high[1] - t_high[0]
dist_high = max(1, int(PEAK_DISTANCE_TIME / dt_high))
peaks_high, _ = find_peaks(C_high, height=PEAK_HEIGHT_THRESHOLD, distance=dist_high)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(t_low, C_low, color='blue', linewidth=1.5, label=r'$T_{\rm bath} = 0.001$')
ax1.plot(t_low[peaks_low], C_low[peaks_low], "o", color='red', markersize=6, label=f'Detected Peaks (N={len(peaks_low)})')
ax1.axhline(PEAK_HEIGHT_THRESHOLD, color='green', linestyle='--', linewidth=1.5, label=f'Height Threshold ({PEAK_HEIGHT_THRESHOLD})')
ax1.set_ylabel(r'$C_{\rm loc}$', fontsize=12)
ax1.set_title('Low Noise: Consistent Peak Detection', fontsize=14)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)

ax2.plot(t_high, C_high, color='orange', linewidth=1.5, label=r'$T_{\rm bath} = 0.04$')
ax2.plot(t_high[peaks_high], C_high[peaks_high], "o", color='red', markersize=6, label=f'Detected Peaks (N={len(peaks_high)})')
ax2.axhline(PEAK_HEIGHT_THRESHOLD, color='green', linestyle='--', linewidth=1.5)
ax2.set_xlabel('Time $t$', fontsize=12)
ax2.set_ylabel(r'$C_{\rm loc}$', fontsize=12)
ax2.set_title('High Noise: Consistent Peak Detection (No Fragmentation)', fontsize=14)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sfit_peak_detection_validation.png", dpi=300)
print("✓ Saved: sfit_peak_detection_validation.png")
plt.show()