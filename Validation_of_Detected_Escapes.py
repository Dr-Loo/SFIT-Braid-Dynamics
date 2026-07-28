import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("STEP 1: VISUAL VALIDATION OF DETECTED ESCAPES")
print("=" * 70)

g_fixed = 0.42
T_bath = 0.001
T_total = 5000.0

# VALIDATED PARAMETERS
PEAK_HEIGHT_THRESHOLD = 0.12
PEAK_DISTANCE_TIME = 2.0
T_PERIOD_REFERENCE = 3.14
ESCAPE_MULTIPLE = 2.0

sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=2000)
t_arr, C_arr = sim.run(T_total=T_total, sample_every=2)
dt_sample = t_arr[1] - t_arr[0]
distance_samples = max(1, int(PEAK_DISTANCE_TIME / dt_sample))

peaks, _ = find_peaks(C_arr, height=PEAK_HEIGHT_THRESHOLD, distance=distance_samples)
peak_times = t_arr[peaks]
intervals = np.diff(peak_times)

# Find the indices of the longest intervals
escape_mask = intervals > (ESCAPE_MULTIPLE * T_PERIOD_REFERENCE)
escape_indices = np.where(escape_mask)[0]

print(f"Total peaks: {len(peaks)}")
print(f"Total escapes detected (interval > {ESCAPE_MULTIPLE * T_PERIOD_REFERENCE:.2f}): {len(escape_indices)}")

if len(escape_indices) > 0:
    # Plot up to the first 4 longest escapes for visual validation
    n_to_plot = min(4, len(escape_indices))
    
    # Sort escape indices by interval length (descending)
    sorted_escapes = escape_indices[np.argsort(intervals[escape_indices])[::-1]]
    
    fig, axes = plt.subplots(n_to_plot, 1, figsize=(12, 3 * n_to_plot), sharex=True)
    if n_to_plot == 1:
        axes = [axes]
        
    for i, idx in enumerate(sorted_escapes[:n_to_plot]):
        ax = axes[i]
        t_start = peak_times[idx]
        t_end = peak_times[idx + 1]
        duration = t_end - t_start
        
        # Mask for this specific interval
        mask = (t_arr >= t_start - 5.0) & (t_arr <= t_end + 5.0) # Add 5 units of padding
        
        ax.plot(t_arr[mask], C_arr[mask], color='blue', linewidth=1.5, label=r'$C_{\rm loc}(t)$')
        
        # Mark the bounding peaks
        ax.plot([t_start, t_end], [C_arr[np.abs(t_arr - t_start).argmin()], C_arr[np.abs(t_arr - t_end).argmin()]], 
                'ro', markersize=8, label='Detected Peaks')
        
        ax.axhline(PEAK_HEIGHT_THRESHOLD, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Height Threshold')
        ax.set_ylabel(r'$C_{\rm loc}$', fontsize=12)
        ax.set_title(f'Escape Event {i+1}: Duration = {duration:.2f} time units', fontsize=13)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel('Time $t$', fontsize=12)
    plt.tight_layout()
    plt.savefig("sfit_visual_escape_validation.png", dpi=300)
    print("\n✓ Saved: sfit_visual_escape_validation.png")
    plt.show()
else:
    print("\nNo escapes detected to validate. This strongly suggests a stable limit cycle with no genuine dormancy.")