import numpy as np
import matplotlib.pyplot as plt
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 17: Per-Term Energy Scaling Exponents (T_bath >= 0.006)")
print("=" * 70)

g_fixed = 0.42
T_bath_values = [0.006, 0.008, 0.01, 0.02, 0.04]
T_total = 2000.0
N_TRIALS = 3

kinetic_means = []
gradient_means = []
mass_pot_means = []
nonlinear_means = []

for T_bath in T_bath_values:
    print(f"\nRunning T_bath = {T_bath}...")
    kin_list, grad_list, mass_list, nl_list = [], [], [], []
    
    for trial in range(N_TRIALS):
        seed = 7000 + int(T_bath * 10000) + trial
        sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)
        
        n_steps = int(T_total / sim.dt)
        sample_every = 10
        
        for step in range(n_steps):
            t = step * sim.dt
            sim.step(t)
            # Discard transient (first 200 time units)
            if step % sample_every == 0 and t > 200.0:
                chi_win = sim.chi[sim.idx_min:sim.idx_max]
                chi_dot_win = sim.chi_dot[sim.idx_min:sim.idx_max]
                
                grad = np.gradient(chi_win, sim.dx)
                kin = 0.5 * np.sum(chi_dot_win**2) * sim.dx
                grad_e = 0.5 * np.sum(grad**2) * sim.dx
                mass = 0.5 * sim.m_chi**2 * np.sum(chi_win**2) * sim.dx
                nl = 0.25 * sim.lambda_c * np.sum(chi_win**4) * sim.dx
                
                kin_list.append(kin)
                grad_list.append(grad_e)
                mass_list.append(mass)
                nl_list.append(nl)
                
    kinetic_means.append(np.mean(kin_list))
    gradient_means.append(np.mean(grad_list))
    mass_pot_means.append(np.mean(mass_list))
    nonlinear_means.append(np.mean(nl_list))
    
    print(f"  Kinetic:    {kinetic_means[-1]:.6f}")
    print(f"  Gradient:   {gradient_means[-1]:.6f}")
    print(f"  Mass Pot:   {mass_pot_means[-1]:.6f}")
    print(f"  Nonlinear:  {nonlinear_means[-1]:.8f}") # 8 decimals to show it's not exactly zero

print("\n" + "=" * 70)
print("SCALING EXPONENTS (Fit: E ~ T^p)")
print("=" * 70)

def fit_exponent(T_vals, E_vals, name):
    log_T = np.log(T_vals)
    log_E = np.log(E_vals)
    p, intercept = np.polyfit(log_T, log_E, 1)
    r2 = 1 - np.sum((log_E - (p * log_T + intercept))**2) / np.sum((log_E - np.mean(log_E))**2)
    print(f"{name:<12}: exponent p = {p:.4f}  (R^2 = {r2:.5f})")
    return p

p_kin = fit_exponent(T_bath_values, kinetic_means, "Kinetic")
p_grad = fit_exponent(T_bath_values, gradient_means, "Gradient")
p_mass = fit_exponent(T_bath_values, mass_pot_means, "Mass Potential")
p_nl = fit_exponent(T_bath_values, nonlinear_means, "Nonlinear (χ⁴)")

print("\nExpected: Quadratic terms (Kinetic, Gradient, Mass Pot) should have p ≈ 1.0")
print("if equipartition holds. Nonlinear term may differ but is negligible.")