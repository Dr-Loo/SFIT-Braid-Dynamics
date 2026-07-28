import numpy as np
import matplotlib.pyplot as plt
from sfit_branching_capacity_engine import SFIT_Solver

print("="*70)
print("TEST 1: Paired-Trajectory Divergence (Transverse Stability)")
print("="*70)

# Parameters for the divergence test
g_test = 0.48
dt = 0.005
T_total_div = 50.0  # 50 time units is plenty to see the decay
n_steps = int(T_total_div / dt)

# Initialize two identical solvers
np.random.seed(42)
sim1 = SFIT_Solver(N=256, L=50.0, dt=dt, g=g_test, T_bath=0.0, seed=42)
sim2 = SFIT_Solver(N=256, L=50.0, dt=dt, g=g_test, T_bath=0.0, seed=42)

# Introduce a tiny perturbation to the initial state (chi and chi_dot)
delta_0 = 1e-8
sim2.chi = sim1.chi + delta_0
sim2.chi_dot = sim1.chi_dot + delta_0

t_arr = []
log_delta_arr = []
delta_norm_arr = []

for step in range(n_steps):
    t = step * dt
    sim1.step(t)
    sim2.step(t)
    
    if step % 10 == 0:  # Sample every 10 steps
        t_arr.append(t)
        # Compute L2 norm of the difference in the full state vector
        delta_chi = sim1.chi - sim2.chi
        delta_chi_dot = sim1.chi_dot - sim2.chi_dot
        delta_norm = np.sqrt(np.sum(delta_chi**2) + np.sum(delta_chi_dot**2))
        
        delta_norm_arr.append(delta_norm)
        log_delta_arr.append(np.log(max(delta_norm, 1e-16))) # Avoid log(0)

# Plot the divergence
plt.figure(figsize=(10, 5))
plt.plot(t_arr, log_delta_arr, linewidth=2, color='blue')
plt.axhline(np.log(1e-15), color='red', linestyle='--', label='Machine Precision Floor (~1e-15)')
plt.xlabel('Time $t$', fontsize=12)
plt.ylabel(r'$\log \|\delta x(t)\|$', fontsize=12)
plt.title(f'Paired Trajectory Separation (g={g_test}, T_bath=0)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sfit_paired_trajectory_divergence.png", dpi=300)
print("✓ Saved: sfit_paired_trajectory_divergence.png")
plt.show()

print("\n" + "="*70)
print("TEST 2: Noise-Induced Escape Time Distribution f(T_esc)")
print("="*70)

# Parameters for the escape test
g_values_escape = [0.42, 0.48, 0.58]
T_total_escape = 2000.0  # Long run to capture rare escapes
T_window = 50.0          # Escape criterion: no crossing for 50 time units

escape_times = {g: [] for g in g_values_escape}
escaped_counts = {g: 0 for g in g_values_escape}

for g in g_values_escape:
    print(f"\nRunning g = {g:.2f} with T_bath = 0.001...")
    sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g, T_bath=0.001, seed=42)
    
    # We need to track crossings manually during the run
    n_steps = int(T_total_escape / sim.dt)
    last_crossing_time = -T_window  # Initialize to allow immediate crossing
    
    escaped = False
    t_escape = T_total_escape
    
    for step in range(n_steps):
        t = step * sim.dt
        sim.step(t)
        
        # Check for upward crossing of C_max
        if step > 0:
            C_prev = sim.compute_C_loc(sim.chi - sim.chi_dot * sim.dt, sim.chi_dot) # Approx prev state
            C_curr = sim.compute_C_loc(sim.chi, sim.chi_dot)
            
            if C_prev < sim.C_max and C_curr >= sim.C_max:
                # Upward crossing detected
                if t - last_crossing_time >= 10.0:  # Respect refractory time
                    last_crossing_time = t
                    
                    # If we had previously flagged an escape, this means it re-activated!
                    # (For this test, we just reset the escape timer)
                    if escaped:
                        escaped = False 
                        print(f"  [g={g:.2f}] Re-activated at t={t:.2f} (False alarm or intermittent)")
        
        # Check escape criterion
        if not escaped and (t - last_crossing_time) > T_window:
            escaped = True
            t_escape = t
            escaped_counts[g] += 1
            escape_times[g].append(t_escape)
            print(f"  [g={g:.2f}] ESCAPE DETECTED at t={t_escape:.2f}")

    if not escaped:
        print(f"  [g={g:.2f}] No escape detected within T={T_total_escape}")

# Plotting the escape time distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, g in enumerate(g_values_escape):
    ax = axes[i]
    data = escape_times[g]
    
    if len(data) > 0:
        counts, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.7, color='green', edgecolor='black')
        ax.set_yscale('log')
        ax.set_xlabel(r'Escape Time $T_{\rm esc}$', fontsize=12)
        ax.set_ylabel('Probability Density (log)', fontsize=12)
        ax.set_title(f'g = {g:.2f} (N={len(data)} escapes)', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add exponential guide line
        if len(data) > 5:
            mean_t = np.mean(data)
            x_fit = np.linspace(min(data), max(data), 100)
            
            # Correctly scale the exponential to match the peak of the histogram
            y_fit_unscaled = (1.0 / mean_t) * np.exp(-x_fit / mean_t)
            scale_factor = np.max(counts) / y_fit_unscaled[0]
            y_fit = y_fit_unscaled * scale_factor
            
            ax.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'Exp. Guide (mean={mean_t:.1f})')
            ax.legend()
    else:
        ax.text(0.5, 0.5, 'No escapes detected', ha='center', va='center', fontsize=14)
        ax.set_title(f'g = {g:.2f}', fontsize=14)

plt.tight_layout()
plt.savefig("sfit_escape_time_distribution.png", dpi=300)
print("\n✓ Saved: sfit_escape_time_distribution.png")
plt.show()