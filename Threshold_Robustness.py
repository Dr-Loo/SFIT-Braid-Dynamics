import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("STEP 2: THRESHOLD ROBUSTNESS STUDY")
print("=" * 70)

g_fixed = 0.42
T_bath_values = [0.001, 0.005, 0.02] # Test a few key noise levels
T_total = 5000.0
N_TRIALS = 3

T_PERIOD_REFERENCE = 3.14
MULTIPLIERS = [1.5, 2.0, 2.5, 3.0]

PEAK_HEIGHT_THRESHOLD = 0.12
PEAK_DISTANCE_TIME = 2.0

fig, ax = plt.subplots(figsize=(10, 6))

for T_bath in T_bath_values:
    escape_counts_per_mult = []
    
    for mult in MULTIPLIERS:
        total_escapes = 0
        
        for trial in range(N_TRIALS):
            seed = 2000 + int(T_bath * 10000) + trial
            sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)
            t_arr, C_arr = sim.run(T_total=T_total, sample_every=2)
            dt_sample = t_arr[1] - t_arr[0]
            distance_samples = max(1, int(PEAK_DISTANCE_TIME / dt_sample))
            
            peaks, _ = find_peaks(C_arr, height=PEAK_HEIGHT_THRESHOLD, distance=distance_samples)
            
            if len(peaks) > 1:
                peak_times = t_arr[peaks]
                intervals = np.diff(peak_times)
                escape_mask = intervals > (mult * T_PERIOD_REFERENCE)
                total_escapes += np.sum(escape_mask)
                
        escape_counts_per_mult.append(total_escapes)
        print(f"T_bath={T_bath:4.3f} | Mult={mult:3.1f}x ({mult*T_PERIOD_REFERENCE:4.2f} units) | Total Escapes: {total_escapes}")
        
    ax.plot(MULTIPLIERS, escape_counts_per_mult, 'o-', linewidth=2, markersize=8, label=f'$T_{{bath}} = {T_bath}$')

ax.set_xlabel(r'Escape Threshold Multiplier ($\times T_{\rm period}$)', fontsize=13)
ax.set_ylabel('Total Escape Events Detected', fontsize=13)
ax.set_title('Robustness of Escape Detection to Threshold Definition', fontsize=14)
ax.set_xticks(MULTIPLIERS)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("sfit_threshold_robustness.png", dpi=300)
print("\n✓ Saved: sfit_threshold_robustness.png")
plt.show()