import numpy as np
import matplotlib.pyplot as plt
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 16: Energy Budget Decomposition & Pump Shutdown Check")
print("=" * 70)

g_fixed = 0.42
T_bath_low = 0.001
T_bath_high = 0.02  # Deep in the departed regime
T_total = 1000.0
sample_every = 10

def compute_energy_components(chi_2d, chi_dot_2d, dx, m_chi, lambda_c, idx_min, idx_max):
    # Slice to the exact same observation window used by compute_C_loc
    chi_win = chi_2d[:, idx_min:idx_max]
    chi_dot_win = chi_dot_2d[:, idx_min:idx_max]
    
    # Compute gradient ONLY along the spatial axis (axis=1)
    grad = np.gradient(chi_win, dx, axis=1)
    
    # Sum along the spatial axis (axis=1) to get energy per time step
    kinetic = 0.5 * np.sum(chi_dot_win**2, axis=1) * dx
    gradient = 0.5 * np.sum(grad**2, axis=1) * dx
    mass_pot = 0.5 * m_chi**2 * np.sum(chi_win**2, axis=1) * dx
    nonlinear_pot = 0.25 * lambda_c * np.sum(chi_win**4, axis=1) * dx
    
    return kinetic, gradient, mass_pot, nonlinear_pot

print(f"\n[1/2] Running Low Noise (T_bath = {T_bath_low})...")
sim_low = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath_low, seed=42)
t_low, C_low, chi_low, chi_dot_low, pump_low = sim_low.run(T_total=T_total, sample_every=sample_every, return_fields=True)

print(f"[2/2] Running High Noise (T_bath = {T_bath_high})...")
sim_high = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath_high, seed=42)
t_high, C_high, chi_high, chi_dot_high, pump_high = sim_high.run(T_total=T_total, sample_every=sample_every, return_fields=True)

# Compute energy components for steady state (discard first 200 time units)
mask_low = t_low > 200.0
mask_high = t_high > 200.0

kin_low, grad_low, mass_low, nl_low = compute_energy_components(
    chi_low[mask_low], chi_dot_low[mask_low], sim_low.dx, sim_low.m_chi, sim_low.lambda_c, sim_low.idx_min, sim_low.idx_max)

kin_high, grad_high, mass_high, nl_high = compute_energy_components(
    chi_high[mask_high], chi_dot_high[mask_high], sim_high.dx, sim_high.m_chi, sim_high.lambda_c, sim_high.idx_min, sim_high.idx_max)

print("\n" + "=" * 70)
print("STEADY-STATE ENERGY BUDGET (Time-Averaged, Window Only)")
print("=" * 70)
print(f"{'Component':<15} | {'Low Noise (0.001)':<20} | {'High Noise (0.02)':<20}")
print("-" * 70)
print(f"{'Kinetic':<15} | {np.mean(kin_low):<20.4f} | {np.mean(kin_high):<20.4f}")
print(f"{'Gradient':<15} | {np.mean(grad_low):<20.4f} | {np.mean(grad_high):<20.4f}")
print(f"{'Mass Potential':<15} | {np.mean(mass_low):<20.4f} | {np.mean(mass_high):<20.4f}")
print(f"{'Nonlinear (χ⁴)':<15} | {np.mean(nl_low):<20.4f} | {np.mean(nl_high):<20.4f}")
print(f"{'TOTAL (C_loc)':<15} | {np.mean(C_low[mask_low]):<20.4f} | {np.mean(C_high[mask_high]):<20.4f}")
print("-" * 70)
print(f"{'Mean Pump Amp':<15} | {np.mean(pump_low[mask_low]):<20.4f} | {np.mean(pump_high[mask_high]):<20.4f}")
print("=" * 70)

# Plotting the Pump Shutdown
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(t_low, pump_low, color='blue', linewidth=1, label=f'T_bath = {T_bath_low}')
ax1.plot(t_high, pump_high, color='red', linewidth=1, label=f'T_bath = {T_bath_high}')
ax1.set_xlabel('Time $t$', fontsize=12)
ax1.set_ylabel('Pump Amplitude $A_{\text{pump}}(C_{\text{loc}})$', fontsize=12)
ax1.set_title('Capacity-Closed Pump Shutdown', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plotting the Energy Components for High Noise
ax2.plot(t_high[mask_high], kin_high, label='Kinetic', linewidth=1.5)
ax2.plot(t_high[mask_high], grad_high, label='Gradient', linewidth=1.5)
ax2.plot(t_high[mask_high], mass_high, label='Mass Potential', linewidth=1.5)
ax2.plot(t_high[mask_high], nl_high, label='Nonlinear (χ⁴)', linewidth=1.5)
ax2.set_xlabel('Time $t$', fontsize=12)
ax2.set_ylabel('Energy Component', fontsize=12)
ax2.set_title(f'Energy Budget Decomposition (High Noise, T_bath={T_bath_high})', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sfit_energy_budget_decomposition.png", dpi=300)
print("\n✓ Saved: sfit_energy_budget_decomposition.png")
plt.show()