import numpy as np
import matplotlib.pyplot as plt
from sfit_branching_capacity_engine import SFIT_Solver

# Custom event detection WITHOUT the long refractory period, to catch ALL crossings
def detect_all_upward_crossings(C_t, t_t, threshold=0.2):
    crossings = []
    for i in range(1, len(C_t)):
        if C_t[i-1] < threshold and C_t[i] >= threshold:
            crossings.append(t_t[i])
    return np.array(crossings)

print("="*70)
print("TEST 3: Residence Time Survival Function S(t)")
print("Testing the Arrhenius / Kramers Escape Hypothesis")
print("="*70)

g_values = [0.42, 0.48, 0.58]
T_total = 5000.0  # Long run to get good statistics for the rare long escapes

fig, ax = plt.subplots(figsize=(10, 6))

for g in g_values:
    print(f"\nRunning g = {g:.2f} for T = {T_total}...")
    # Seed varies slightly to ensure independent realizations
    sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g, T_bath=0.001, seed=42 + int(g*10))
    
    # High temporal resolution (sample_every=2) for accurate crossing times
    t_arr, C_arr = sim.run(T_total=T_total, sample_every=2) 
    
    crossings = detect_all_upward_crossings(C_arr, t_arr, threshold=0.2)
    
    if len(crossings) > 10:
        # Inter-Event Intervals (IEIs)
        ieis = np.diff(crossings)
        
        # Compute empirical survival function S(t) = P(IEI > t)
        # We focus on t > 10 to filter out the deterministic limit cycle period (~3.14)
        # and isolate the stochastic "escape" residence times.
        t_vals = np.linspace(10.0, np.max(ieis), 200)
        S_t = np.array([np.sum(ieis > t) / len(ieis) for t in t_vals])
        
        # Plot on semi-log scale (linear x, log y)
        # A straight line here proves S(t) ~ exp(-t / tau) -> Exponential / Arrhenius
        ax.plot(t_vals, S_t, linewidth=2.5, label=f'g = {g:.2f} (N={len(ieis)} crossings)')
        
        # Fit the tail to extract the Kramers residence time tau
        # Use the middle of the tail (S_t between 0.005 and 0.2) to avoid finite-size noise
        mask = (S_t > 0.005) & (S_t < 0.2)
        if np.sum(mask) > 5:
            t_fit = t_vals[mask]
            S_fit = S_t[mask]
            
            # log(S) = -t/tau + C  =>  slope = -1/tau
            slope, intercept = np.polyfit(t_fit, np.log(S_fit), 1)
            tau_kramers = -1.0 / slope
            
            # Plot the linear fit line
            t_line = np.array([t_fit[0], t_fit[-1]])
            S_line = np.exp(slope * t_line + intercept)
            ax.plot(t_line, S_line, 'k--', linewidth=1.5, alpha=0.8)
            
            print(f"  -> Kramers residence time tau = {tau_kramers:.2f} time units")
            print(f"  -> Escape rate Gamma = {1.0/tau_kramers:.5f}")
    else:
        print(f"  Too few crossings detected.")

ax.set_yscale('log')
ax.set_xlabel(r'Inter-Event Interval $\Delta t$ (Residence Time)', fontsize=14)
ax.set_ylabel(r'Survival Probability $S(t) = P(\Delta t > t)$', fontsize=14)
ax.set_title('Residence Time Distribution (Tests for Exponential/Arrhenius Tail)', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig("sfit_residence_survival_function.png", dpi=300)
print("\n✓ Saved: sfit_residence_survival_function.png")
plt.show()