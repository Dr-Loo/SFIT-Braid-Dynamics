import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("DIAGNOSTIC: Verify Peak Detection Logic (FIXED)")
print("=" * 70)

g_fixed = 0.42
T_bath = 0.001
T_total = 5000.0

# Parameters from the validator
PEAK_HEIGHT_THRESHOLD = 0.12
PEAK_DISTANCE_TIME = 2.0

sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=2000)
t_arr, C_arr = sim.run(T_total=T_total, sample_every=2)
dt_sample = t_arr[1] - t_arr[0]
distance_samples = max(1, int(PEAK_DISTANCE_TIME / dt_sample))

print(f"\nSimulation parameters:")
print(f"  T_total = {T_total}")
print(f"  dt_sample = {dt_sample}")
print(f"  Total time steps = {len(t_arr)}")

print(f"\nPeak detection parameters:")
print(f"  height = {PEAK_HEIGHT_THRESHOLD}")
print(f"  distance (time) = {PEAK_DISTANCE_TIME}")
print(f"  distance (samples) = {distance_samples}")

# Detect peaks
peaks, properties = find_peaks(C_arr, height=PEAK_HEIGHT_THRESHOLD, distance=distance_samples)

print(f"\nResults:")
print(f"  Total peaks detected: {len(peaks)}")
print(f"  Expected peaks (T/3.14): {T_total / 3.14:.0f}")
print(f"  Ratio: {len(peaks) / (T_total / 3.14):.3f}")

# Check peak intervals
if len(peaks) > 1:
    peak_times = t_arr[peaks]
    intervals = np.diff(peak_times)
    
    print(f"\nPeak interval statistics:")
    print(f"  Mean interval: {np.mean(intervals):.3f}")
    print(f"  Median interval: {np.median(intervals):.3f}")
    print(f"  Std interval: {np.std(intervals):.3f}")
    print(f"  Min interval: {np.min(intervals):.3f}")
    print(f"  Max interval: {np.max(intervals):.3f}")
    
    # Check how many intervals are close to 3.14
    normal_mask = np.abs(intervals - 3.14) <= 0.15 * 3.14
    frac_normal = np.mean(normal_mask)
    print(f"  Fraction within 15% of 3.14: {frac_normal:.3f}")
    
    # Check for escape events (intervals > 2 * 3.14)
    escape_mask = intervals > (2.0 * 3.14)
    n_escapes = np.sum(escape_mask)
    print(f"  Escape events (interval > 6.28): {n_escapes}")
    
    if n_escapes > 0:
        escape_intervals = intervals[escape_mask]
        print(f"  Escape interval range: [{np.min(escape_intervals):.2f}, {np.max(escape_intervals):.2f}]")

# Visual check: plot first 100 time units
# FIXED: Correctly filter the peaks array based on time
valid_peaks = peaks[t_arr[peaks] < 100]

fig, ax = plt.subplots(figsize=(12, 5))
time_mask = t_arr < 100
ax.plot(t_arr[time_mask], C_arr[time_mask], linewidth=1.5, color='blue', label=r'$C_{\rm loc}(t)$')
ax.plot(t_arr[valid_peaks], C_arr[valid_peaks], "o", color='red', 
        markersize=6, label=f'Detected Peaks (N={len(valid_peaks)} in first 100 units)')
ax.axhline(PEAK_HEIGHT_THRESHOLD, color='green', linestyle='--', linewidth=1.5, 
           label=f'Height Threshold ({PEAK_HEIGHT_THRESHOLD})')
ax.set_xlabel('Time $t$')
ax.set_ylabel(r'$C_{\rm loc}$')
ax.set_title('Visual Check: First 100 Time Units')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sfit_peak_diagnostic_fixed.png", dpi=300)
print("\n✓ Saved: sfit_peak_diagnostic_fixed.png")
plt.show()

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
if 0.9 <= len(peaks) / (T_total / 3.14) <= 1.1:
    print("✓ Peak detector is working correctly!")
    print(f"  Detected {len(peaks)} peaks vs expected ~{T_total / 3.14:.0f}")
else:
    print("✗ Peak detector may have issues")
    print(f"  Detected {len(peaks)} peaks vs expected ~{T_total / 3.14:.0f}")