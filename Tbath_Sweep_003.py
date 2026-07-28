import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sfit_branching_capacity_engine import SFIT_Solver

print("VISUAL CHECK: Peak Fragmentation at T_bath=0.003")

sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=0.42, T_bath=0.003, seed=2030)
t_arr, C_arr = sim.run(T_total=200.0, sample_every=2)
dt_sample = t_arr[1] - t_arr[0]
distance_samples = int(2.0 / dt_sample)

# Old method (fixed height - causes fragmentation)
peaks_old, _ = find_peaks(C_arr, height=0.1419, distance=distance_samples)

# New method (prominence - should fix fragmentation)
peaks_new, _ = find_peaks(C_arr, prominence=0.05, distance=distance_samples)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(t_arr, C_arr, linewidth=1, color='blue')
ax1.plot(t_arr[peaks_old], C_arr[peaks_old], "x", color='red', markersize=8, label=f'Fixed Height (N={len(peaks_old)} peaks)')
ax1.axhline(0.1419, color='green', linestyle=':', linewidth=1.5, label='Height threshold')
ax1.set_ylabel(r'$C_{\rm loc}$')
ax1.set_title('OLD METHOD: Fixed Height Threshold (Fragmentation)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(t_arr, C_arr, linewidth=1, color='blue')
ax2.plot(t_arr[peaks_new], C_arr[peaks_new], "o", color='orange', markersize=8, label=f'Prominence (N={len(peaks_new)} peaks)')
ax2.set_xlabel('Time $t$')
ax2.set_ylabel(r'$C_{\rm loc}$')
ax2.set_title('NEW METHOD: Prominence-Based (Should Fix Fragmentation)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sfit_peak_fragmentation_visual_check.png", dpi=300)
print("Saved: sfit_peak_fragmentation_visual_check.png")
plt.show()