import numpy as np
import matplotlib.pyplot as plt
from sfit_branching_capacity_engine import SFIT_Solver

# Run noiseless simulations at several g values
g_values = [0.42, 0.48, 0.58, 0.65]
results = {}

for g in g_values:
    print(f"\nRunning g = {g:.2f} with T_bath = 0...")
    sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g, T_bath=0.0, seed=42)
    t_arr, C_arr = sim.run(T_total=2000.0, sample_every=10)
    
    # Classify the dynamics
    C_max_val = np.max(C_arr)
    C_min_val = np.min(C_arr)
    
    if C_max_val < 0.15:
        regime = "dormant (fixed point)"
    elif C_max_val > 5.0:
        regime = "runaway"
    else:
        # Check if periodic
        # Simple test: find peaks and check if period is constant
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(C_arr, height=0.18)
        if len(peaks) > 2:
            periods = np.diff(peaks) * 0.05  # sample_every * dt
            if np.std(periods) / np.mean(periods) < 0.01:
                regime = f"periodic limit cycle (T = {np.mean(periods):.2f})"
            else:
                regime = "quasi-periodic or chaotic"
        else:
            regime = "transient or irregular"
    
    results[g] = {'t': t_arr, 'C': C_arr, 'regime': regime}
    print(f"  C_loc range: [{C_min_val:.4f}, {C_max_val:.4f}]")
    print(f"  Regime: {regime}")

# Plot results
fig, axes = plt.subplots(len(g_values), 1, figsize=(12, 10), sharex=True)

for i, g in enumerate(g_values):
    ax = axes[i]
    ax.plot(results[g]['t'], results[g]['C'], linewidth=1.5)
    ax.axhline(0.2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel(f'g = {g:.2f}')
    ax.set_title(results[g]['regime'], fontsize=10)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.savefig("sfit_noiseless_skeleton.png", dpi=300)
print("\nSaved: sfit_noiseless_skeleton.png")
plt.show()