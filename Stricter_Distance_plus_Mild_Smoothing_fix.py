import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 9 (FINAL): Robust Outlier Detection with Smoothing")
print("=" * 70)

g_fixed = 0.42
T_bath_values = [0.001, 0.003, 0.005, 0.01, 0.02, 0.04]
T_total = 5000.0
N_TRIALS = 5

T_PERIOD_REFERENCE = 3.14
EXPECTED_PEAKS = int(N_TRIALS * T_total / T_PERIOD_REFERENCE) # ~7962

# ROBUST PARAMETERS
# Distance set to 2.8 (90% of period) strictly enforces 1 peak per cycle
PEAK_DISTANCE_TIME = 2.8  
# Smoothing window must be odd and < period/dt_sample. 
# For dt_sample=0.01, period=3.14 -> ~314 samples. Window of 51 is safe and effective.
SG_WINDOW = 51  
SG_POLYORDER = 3

fig_rates, ax_rates = plt.subplots(figsize=(10, 6))
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

        # 1. Apply mild Savitzky-Golay filter to remove high-frequency noise jaggedness
        # This preserves the macroscopic peak structure while flattening thermal bumps
        C_smoothed = savgol_filter(C_arr, window_length=SG_WINDOW, polyorder=SG_POLYORDER)

        # 2. Detect peaks on the SMOOTHED signal with strict distance
        # We use a low height threshold because smoothing handles the noise
        peaks, _ = find_peaks(C_smoothed, height=0.05, distance=distance_samples)
        
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
    mad_scaled = 1.4826 * mad # Scale to approximate standard deviation for normal distribution
    
    # Modified Z-score for outlier detection
    modified_z = 0.6745 * (all_intervals - median_interval) / mad_scaled if mad_scaled > 0 else 0
    outlier_mask = modified_z > 3.5
    outlier_durations = all_intervals[outlier_mask]
    all_outlier_durations.extend(outlier_durations.tolist())

    # Diagnostics
    ratio = total_peaks / EXPECTED_PEAKS
    drift_warning = ""
    if abs(ratio - 1.0) > 0.10:
        drift_warning = "\n  *** WARNING: recovery ratio drifted >10% from ideal -- inspect this T_bath ***"

    print(f"  peak count per trial: ~{total_peaks//N_TRIALS} (expected ~{EXPECTED_PEAKS//N_TRIALS}, ratio {ratio:.3f}){drift_warning}")
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

# Plot 1: Interval distributions
ax_intervals.set_xlabel(r'Inter-Peak Interval $\Delta t$', fontsize=13)
ax_intervals.set_ylabel('Probability Density', fontsize=13)
ax_intervals.set_title('Inter-Peak Interval Distributions (Should peak near 3.14)', fontsize=14)
ax_intervals.axvline(3.14, color='red', linestyle='--', linewidth=2, label='True Period (3.14)')
ax_intervals.legend(fontsize=10)
ax_intervals.grid(True, alpha=0.4)
plt.figure(fig_intervals.number)
plt.tight_layout()
plt.savefig("sfit_interval_distributions_smoothed.png", dpi=300)
print("\n✓ Saved: sfit_interval_distributions_smoothed.png")

# Plot 2: Escape Rate vs T_bath (Arrhenius check)
valid_points = [s for s in summary if s["n_outliers"] >= 3]
if len(valid_points) >= 2:
    Tb = [s["T_bath"] for s in valid_points]
    rates = [s["rate"] for s in valid_points]
    
    # Avoid log(0)
    rates_safe = [max(r, 1e-6) for r in rates]
    inv_Tb = 1.0 / np.array(Tb)
    ln_rates = np.log(rates_safe)
    
    ax_rates.plot(inv_Tb, ln_rates, "o", markersize=8, color='darkblue')
    
    if len(valid_points) >= 3:
        slope, intercept = np.polyfit(inv_Tb, ln_rates, 1)
        Ea = -slope
        ax_rates.plot(inv_Tb, slope * inv_Tb + intercept, "k--",
                      label=f"Linear fit (Slope = {slope:.4f}, $E_a \\approx$ {Ea:.4f})")
        print(f"\nArrhenius Fit: Slope = {slope:.4f} -> Activation Energy Ea = {Ea:.4f}")
    else:
        print("\nNote: Only 2 valid points. Arrhenius slope not plotted, but trend is visible.")
    
    ax_rates.set_xlabel(r"$1/T_{\text{bath}}$", fontsize=13)
    ax_rates.set_ylabel(r"$\ln(\Gamma)$ (Escape Rate)", fontsize=13)
    ax_rates.set_title("Arrhenius Check: Escape Rate vs Inverse Noise", fontsize=14)
    ax_rates.legend(fontsize=11)
    ax_rates.grid(True, alpha=0.4)
    plt.figure(fig_rates.number)
    plt.tight_layout()
    plt.savefig("sfit_arrhenius_rate_check_smoothed.png", dpi=300)
    print("✓ Saved: sfit_arrhenius_rate_check_smoothed.png")
else:
    print("\nNote: Not enough valid outlier points (>=3) to attempt Arrhenius fit.")

plt.show()