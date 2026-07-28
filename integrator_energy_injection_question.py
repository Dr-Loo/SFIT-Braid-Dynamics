import numpy as np
import matplotlib.pyplot as plt
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("CRITICAL NUMERICS CHECK: Timestep Convergence at High Noise")
print("Testing g=0.42, T_bath=0.04 with dt=0.005 vs dt=0.0025")
print("=" * 70)

g_fixed = 0.42
T_bath_high = 0.04
T_total = 1000.0  # Long enough to reach steady state, short enough to run fast
seed = 4200

results = {}
dt_values = [0.005, 0.0025]

for dt in dt_values:
    print(f"\nRunning with dt = {dt}...")
    sim = SFIT_Solver(N=256, L=50.0, dt=dt, g=g_fixed, T_bath=T_bath_high, seed=seed)
    
    # We need the full field to check actual chi amplitude, not just C_loc
    t_arr = []
    C_arr = []
    chi_max_arr = []
    
    n_steps = int(T_total / dt)
    sample_every = max(1, int(0.05 / dt)) # Sample every 0.05 time units
    
    for step in range(n_steps):
        t = step * dt
        sim.step(t)
        if step % sample_every == 0:
            t_arr.append(t)
            C_arr.append(sim.compute_C_loc(sim.chi, sim.chi_dot))
            chi_max_arr.append(np.max(np.abs(sim.chi)))
            
    C_arr = np.array(C_arr)
    chi_max_arr = np.array(chi_max_arr)
    
    # Discard first 200 time units as transient
    transient_mask = np.array(t_arr) > 200.0
    C_steady = C_arr[transient_mask]
    chi_steady = chi_max_arr[transient_mask]
    
    results[dt] = {
        "C_mean": np.mean(C_steady),
        "C_max": np.max(C_steady),
        "chi_max_mean": np.mean(chi_steady),
        "chi_max_max": np.max(chi_steady),
        "t": np.array(t_arr)[transient_mask],
        "C": C_steady
    }
    
    print(f"  Steady-state C_loc:  Mean = {results[dt]['C_mean']:.4f}, Max = {results[dt]['C_max']:.4f}")
    print(f"  Steady-state |chi|:  Mean Max = {results[dt]['chi_max_mean']:.4f}, Absolute Max = {results[dt]['chi_max_max']:.4f}")

# Compare the two
dt1, dt2 = 0.005, 0.0025
c_mean_ratio = results[dt1]['C_mean'] / results[dt2]['C_mean']
chi_max_ratio = results[dt1]['chi_max_mean'] / results[dt2]['chi_max_mean']

print("\n" + "=" * 70)
print("CONVERGENCE ANALYSIS:")
print("=" * 70)
print(f"C_loc Mean Ratio (dt=0.005 / dt=0.0025): {c_mean_ratio:.3f}")
print(f"|chi| Max Mean Ratio (dt=0.005 / dt=0.0025): {chi_max_ratio:.3f}")

if 0.90 <= c_mean_ratio <= 1.10 and 0.90 <= chi_max_ratio <= 1.10:
    print("\n✓ PASS: Results are converged within 10%. dt=0.005 is trustworthy.")
else:
    print("\n✗ FAIL: Significant numerical drift detected.")
    print("The high-amplitude results at T_bath=0.04 are likely integration artifacts.")
    print("The entire T_bath sweep must be re-run with dt=0.0025 (or smaller).")

# Visual comparison of the steady-state C_loc
plt.figure(figsize=(10, 5))
plt.plot(results[0.005]['t'], results[0.005]['C'], linewidth=1, alpha=0.7, label=f'dt = {0.005} (Mean C = {results[0.005]["C_mean"]:.3f})')
plt.plot(results[0.0025]['t'], results[0.0025]['C'], linewidth=1, alpha=0.7, label=f'dt = {0.0025} (Mean C = {results[0.0025]["C_mean"]:.3f})')
plt.xlabel('Time $t$', fontsize=12)
plt.ylabel(r'$C_{\rm loc}(t)$', fontsize=12)
plt.title('Timestep Convergence Check: High Noise ($T_{\text{bath}} = 0.04$)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sfit_timestep_convergence_check.png", dpi=300)
print("\n✓ Saved: sfit_timestep_convergence_check.png")
plt.show()