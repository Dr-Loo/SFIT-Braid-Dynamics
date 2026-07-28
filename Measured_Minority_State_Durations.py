import numpy as np
import matplotlib.pyplot as plt
from sfit_branching_capacity_engine import SFIT_Solver

print("="*70)
print("TEST 4: Clean State Duration Measurement (No Debounce Artifacts)")
print("="*70)

g_values = [0.42, 0.48, 0.58]
T_total = 5000.0

# Thresholds to define the two basins clearly, avoiding the noisy boundary
C_active_threshold = 0.15  # Clearly inside the active limit cycle
C_dormant_threshold = 0.05 # Clearly inside the dormant basin

fig, ax = plt.subplots(figsize=(10, 6))

for g in g_values:
    print(f"\nRunning g = {g:.2f} for T = {T_total}...")
    sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g, T_bath=0.001, seed=42 + int(g*10))
    t_arr, C_arr = sim.run(T_total=T_total, sample_every=2) # High res
    
    dt_sample = t_arr[1] - t_arr[0]
    
    # Identify which state is the "rare" one for this coupling
    fraction_active = np.mean(C_arr > C_active_threshold)
    
    if fraction_active < 0.5:
        # Mostly dormant: measure duration of ACTIVE bursts (Kramers activation)
        print("  -> System is mostly dormant. Measuring ACTIVE burst durations.")
        in_state = C_arr > C_active_threshold
        state_name = "Active Burst"
    else:
        # Mostly active: measure duration of DORMANT periods (Kramers deactivation)
        print("  -> System is mostly active. Measuring DORMANT period durations.")
        in_state = C_arr < C_dormant_threshold
        state_name = "Dormant Period"
        
    # Find durations of contiguous periods in the rare state
    # (Simple run-length encoding)
    durations = []
    current_duration = 0
    
    for is_in in in_state:
        if is_in:
            current_duration += dt_sample
        else:
            if current_duration > 0:
                durations.append(current_duration)
            current_duration = 0
            
    if current_duration > 0:
        durations.append(current_duration)
        
    durations = np.array(durations)
    
    if len(durations) > 10:
        # Compute survival function for the rare state durations
        t_vals = np.linspace(0.1, np.max(durations), 200)
        S_t = np.array([np.sum(durations > t) / len(durations) for t in t_vals])
        
        ax.plot(t_vals, S_t, linewidth=2.5, label=f'g = {g:.2f} ({state_name}, N={len(durations)})')
        
        # Fit the tail (exclude very short durations which might be noise flickering)
        min_fit_duration = 5.0 # Must be > deterministic period (3.14) to be a true escape
        mask = (t_vals >= min_fit_duration) & (S_t > 0.01) & (S_t < 0.5)
        
        if np.sum(mask) > 5:
            t_fit = t_vals[mask]
            S_fit = S_t[mask]
            slope, intercept = np.polyfit(t_fit, np.log(S_fit), 1)
            tau_kramers = -1.0 / slope
            
            t_line = np.array([t_fit[0], t_fit[-1]])
            S_line = np.exp(slope * t_line + intercept)
            ax.plot(t_line, S_line, 'k--', linewidth=1.5, alpha=0.8)
            
            print(f"  -> Clean Kramers residence time tau = {tau_kramers:.2f} time units")
            print(f"  -> Escape rate Gamma = {1.0/tau_kramers:.5f}")
    else:
        print(f"  -> Too few rare events detected to fit.")

ax.set_yscale('log')
ax.set_xlabel(r'Duration of Rare State Excursion $\Delta t$', fontsize=14)
ax.set_ylabel(r'Survival Probability $S(t)$', fontsize=14)
ax.set_title('Clean Kramers Residence Times (State Duration Method)', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig("sfit_clean_kramers_residence_times.png", dpi=300)
print("\n✓ Saved: sfit_clean_kramers_residence_times.png")
plt.show()