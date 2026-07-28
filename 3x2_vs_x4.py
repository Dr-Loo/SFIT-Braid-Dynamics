import numpy as np
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 18: Gaussian Moment Relation Check")
print("Verify: <chi^4> ≈ 3<chi^2>^2 for Gaussian field")
print("=" * 70)

g_fixed = 0.42
T_bath_values = [0.006, 0.01, 0.02, 0.04]
T_total = 2000.0
N_TRIALS = 3

print("\nRunning simulations and computing moments...")
for T_bath in T_bath_values:
    chi2_list = []
    chi4_list = []
    
    for trial in range(N_TRIALS):
        seed = 8000 + int(T_bath * 10000) + trial
        sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)
        
        n_steps = int(T_total / sim.dt)
        sample_every = 10
        
        for step in range(n_steps):
            t = step * sim.dt
            sim.step(t)
            if step % sample_every == 0 and t > 200.0:
                chi_win = sim.chi[sim.idx_min:sim.idx_max]
                chi2_list.append(np.mean(chi_win**2))
                chi4_list.append(np.mean(chi_win**4))
    
    chi2_mean = np.mean(chi2_list)
    chi4_mean = np.mean(chi4_list)
    chi4_gaussian = 3 * chi2_mean**2
    ratio = chi4_mean / chi4_gaussian
    
    print(f"\nT_bath = {T_bath}:")
    print(f"  <chi^2> = {chi2_mean:.6f}")
    print(f"  <chi^4> (measured) = {chi4_mean:.8f}")
    print(f"  3<chi^2>^2 (Gaussian prediction) = {chi4_gaussian:.8f}")
    print(f"  Ratio (measured / Gaussian) = {ratio:.3f}")
    if 0.9 < ratio < 1.1:
        print(f"  ✓ Consistent with Gaussian statistics")
    else:
        print(f"  ✗ Deviates from Gaussian statistics")

print("\n" + "=" * 70)
print("Interpretation:")
print("If ratio ≈ 1, the field is approximately Gaussian, confirming that")
print("the chi^4 term is a passive observable, not an active energy participant.")
print("=" * 70)