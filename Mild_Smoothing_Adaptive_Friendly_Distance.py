import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 10 (FINAL): Robust Detection with Mild Smoothing & Flexible Distance")
print("=" * 70)

g_fixed = 0.42
T_bath_values = [0.001, 0.003, 0.005, 0.01, 0.02, 0.04]
T_total = 5000.0
N_TRIALS = 5

# We will dynamically calculate the expected peaks based on a baseline of ~3.14, 
# but we allow the actual median to be lower if the dynamics speed up.
EXPECTED_PEAKS_BASELINE = int(N_TRIALS * T_total / 3.14) # ~7962

# ROBUST PARAMETERS
# 1. Mild smoothing: window=11 (0.11 time units) kills high-freq noise but preserves peaks
SG_WINDOW = 11  
SG_POLYORDER = 3

# 2. Fixed height: 0.10 is safely above the noise floor (~0.0) but below deterministic peaks (~0.19)
PEAK_HEIGHT_THRESHOLD = 0.10

# 3. Flexible distance: 1.5 time units allows legitimate faster oscillations (period ~2.0-2.5) 
# while still preventing extreme fragmentation (noise bumps < 1.5 apart).
PEAK_DISTANCE_TIME = 1.5  

fig_intervals, ax_intervals = plt.subplots(figsize=(10, 6))
summary = []

for T_bath in T_bath_values:
    print(f"\nRunning g = {g_fixed:.2f}, T_bath = {T_bath}  ({N_TRIALS} trials x T = {T_total})...")

    all_outlier_durations = []
    all_intervals = []
    total_peaks = 0

    for trial in range(N_TRIALS):
        seed = 2000 + int(T_bath * 10000) + trial
        sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)
        t_arr, C_arr = sim.run(T_total=T_total, sample_every=2)
        dt_sample = t_arr[1] - t_arr[0]
        distance_samples = max(1, int(PEAK_DISTANCE_TIME / dt_sample))

        # 1. Apply VERY MILD smoothing to remove high-frequency thermal jaggedness
        C_smoothed = savgol_filter(C_arr, window_length=SG_WINDOW, polyorder=SG_POLYORDER)

        # 2. Detect peaks on smoothed signal with flexible distance and fixed height
        peaks, _ = find_peaks(C_smoothed, height=PEAK_HEIGHT_THRESHOLD, distance=distance_samples)
        
        total_peaks += len(peaks)

        if len(peaks) < 3:
            continue

        peak_times = t_arr[peaks]
        intervals = np.diff(peak_times)
        all_intervals.extend(intervals.tolist())

    all_intervals = np.array(all_intervals)
    
    # Calculate robust statistics
    median_interval = np.median(all_intervals)
    mad = np.median(np.abs(all_intervals - median_interval))
    mad_scaled = 1.4826 * mad 
    
    # Modified Z-score for outlier detection
    modified_z = 0.6745 * (all_intervals - median_interval) / mad_scaled if mad_scaled > 0 else 0
    outlier_mask = modified_z > 3.5
    outlier_durations = all_intervals[outlier_mask]
    all_outlier_durations.extend(outlier_durations.tolist())

    # Diagnostics
    ratio = total_peaks / EXPECTED_PEAKS_BASELINE
    drift_warning = ""
    # We now check if the ratio is *too high* (fragmentation) or *too low* (merging)
    if ratio > 1.15:
        drift_warning = "\n  *** WARNING: Ratio > 1.15. Possible peak fragmentation. ***"
    elif ratio < 0.85:
        drift_warning = "\n  *** WARNING: Ratio < 0.85. Possible peak merging/skipping. ***"

    print(f"  peak count per trial: ~{total_peaks//N_TRIALS} (baseline expected ~{EXPECTED_PEAKS_BASELINE//N_TRIALS}, ratio {ratio:.3f}){drift_warning}")
    print(f"  normal interval: median={median_interval:.3f}, MAD(scaled)={mad_scaled:.3f}")
    print(f"  outliers (modified z > 3.5): {len(outlier_durations)}")
    
    if len(outlier_durations) > 0:
        print(f"  outlier duration range: [{np.min(outlier_durations):.2f}, {np.max(outlier_durations):.2f}], median={np.median(outlier_durations):.2f}")
        print(f"  escape rate: {len(outlier_durations) / (N_TRIALS * T_total):.5f} events/time-unit")
    else:
        print(f"  outlier duration range: N/A")
        print(f"  escape rate: 0.00000 events/time-unit")

    summary.append({
        "T_bath": T_bath, 
        "ratio": ratio, 
        "median_interval": median_interval,
        "n_outliers": len(outlier_durations),
        "rate": len(outlier_durations) / (N_TRIALS * T_total)
    })

    # Plotting intervals distribution for visual check
    ax_intervals.hist(all_intervals, bins=100, density=True, alpha=0.5, label=f'T_bath={T_bath}')

# Plot: Interval distributions
ax_intervals.set_xlabel(r'Inter-Peak Interval $\Delta t$', fontsize=13)
ax_intervals.set_ylabel('Probability Density', fontsize=13)
ax_intervals.set_title('Inter-Peak Interval Distributions (Should show stable, distinct peaks)', fontsize=14)
ax_intervals.axvline(3.14, color='red', linestyle='--', linewidth=2, label='Low-Noise Period (3.14)')
ax_intervals.legend(fontsize=10)
ax_intervals.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("sfit_interval_distributions_final.png", dpi=300)
print("\n✓ Saved: sfit_interval_distributions_final.png")

# Arrhenius check
valid_points = [s for s in summary if s["n_outliers"] >= 3]
if len(valid_points) >= 2:
    Tb = [s["T_bath"] for s in valid_points]
    rates = [s["rate"] for s in valid_points]
    
    rates_safe = [max(r, 1e-6) for r in rates]
    inv_Tb = 1.0 / np.array(Tb)
    ln_rates = np.log(rates_safe)
    
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(inv_Tb, ln_rates, "o", markersize=8, color='darkblue')
    
    if len(valid_points) >= 3:
        slope, intercept = np.polyfit(inv_Tb, ln_rates, 1)
        Ea = -slope
        ax2.plot(inv_Tb, slope * inv_Tb + intercept, "k--",
                 label=f"Linear fit (Slope = {slope:.4f}, $E_a \\approx$ {Ea:.4f})")
        print(f"\nArrhenius Fit: Slope = {slope:.4f} -> Activation Energy Ea = {Ea:.4f}")
    
    ax2.set_xlabel(r"$1/T_{\text{bath}}$", fontsize=13)
    ax2.set_ylabel(r"$\ln(\Gamma)$ (Escape Rate)", fontsize=13)
    ax2.set_title("Arrhenius Check: Escape Rate vs Inverse Noise", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("sfit_arrhenius_rate_check_final.png", dpi=300)
    print("✓ Saved: sfit_arrhenius_rate_check_final.png")
else:
    print("\nNote: Not enough valid outlier points (>=3) to attempt Arrhenius fit.")

plt.show()