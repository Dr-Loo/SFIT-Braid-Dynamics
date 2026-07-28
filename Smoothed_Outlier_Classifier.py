import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 8 (FINAL): Smoothed Outlier-Based Escape Detection")
print("=" * 70)

g_fixed = 0.42
T_bath_values = [0.001, 0.003, 0.005, 0.01, 0.02, 0.04]
T_total = 5000.0
N_TRIALS = 5

PEAK_HEIGHT = 0.12
PEAK_DISTANCE_TIME = 2.0
MODIFIED_Z_THRESHOLD = 3.5

# Savitzky-Golay pre-filter: window is set relative to the deterministic
# period (~3.14), NOT a fixed sample count, so it stays correct if dt_sample
# changes. ~0.5 * period suppresses sub-period noise while preserving the
# genuine oscillation shape (unlike a plain moving average, Sav-Gol keeps
# peak height/timing much more faithfully).
SMOOTH_WINDOW_FRACTION_OF_PERIOD = 0.5
SMOOTH_POLYORDER = 3


def smoothed_peaks(t_arr, C_arr, dt_sample):
    window_time = SMOOTH_WINDOW_FRACTION_OF_PERIOD * 3.14
    window_samples = max(SMOOTH_POLYORDER + 2, int(window_time / dt_sample))
    if window_samples % 2 == 0:
        window_samples += 1  # savgol requires an odd window
    C_smooth = savgol_filter(C_arr, window_length=window_samples,
                              polyorder=SMOOTH_POLYORDER)
    distance_samples = max(1, int(PEAK_DISTANCE_TIME / dt_sample))
    peaks, _ = find_peaks(C_smooth, height=PEAK_HEIGHT, distance=distance_samples)
    return peaks, C_smooth


fig, (ax_dist, ax_rate) = plt.subplots(1, 2, figsize=(14, 5.5))
summary = []

for T_bath in T_bath_values:
    print(f"\nRunning g = {g_fixed:.2f}, T_bath = {T_bath}  "
          f"({N_TRIALS} trials x T = {T_total})...")

    all_intervals = []
    all_peak_counts = []
    total_observed_time = 0.0

    for trial in range(N_TRIALS):
        seed = 4000 + int(T_bath * 10000) + trial
        sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)
        t_arr, C_arr = sim.run(T_total=T_total, sample_every=2)
        dt_sample = t_arr[1] - t_arr[0]

        peaks, _ = smoothed_peaks(t_arr, C_arr, dt_sample)
        all_peak_counts.append(len(peaks))

        if len(peaks) < 3:
            continue

        peak_times = t_arr[peaks]
        intervals = np.diff(peak_times)
        all_intervals.extend(intervals.tolist())
        total_observed_time += t_arr[-1] - t_arr[0]

    all_intervals = np.array(all_intervals)
    n_peaks_total = sum(all_peak_counts)
    expected_total = (T_total / 3.14) * N_TRIALS
    recovery_ratio = n_peaks_total / expected_total if expected_total else float("nan")

    print(f"  peak count per trial: {all_peak_counts}")
    print(f"  total peaks: {n_peaks_total}  (expected ~{expected_total:.0f}, "
          f"ratio {recovery_ratio:.3f})")

    # Tighter check than before: flag any meaningful deviation from the
    # clean T_bath=0.001 baseline (0.98), not just a wide sanity band.
    if abs(recovery_ratio - 1.0) > 0.10:
        print("  *** WARNING: recovery ratio drifted >10% from ideal -- "
              "inspect this T_bath before trusting its escape statistics ***")

    if len(all_intervals) < 20:
        print("  -> too few intervals to build a distribution")
        summary.append({"T_bath": T_bath, "rate": 0.0, "durations": np.array([]),
                         "recovery_ratio": recovery_ratio, "median_interval": float("nan")})
        continue

    median_interval = np.median(all_intervals)
    mad = np.median(np.abs(all_intervals - median_interval))
    mad_scaled = mad * 1.4826 if mad > 0 else np.std(all_intervals)

    print(f"  normal interval: median={median_interval:.3f}, "
          f"MAD(scaled)={mad_scaled:.3f}")

    if mad_scaled == 0:
        print("  -> zero MAD; no outliers possible")
        summary.append({"T_bath": T_bath, "rate": 0.0, "durations": np.array([]),
                         "recovery_ratio": recovery_ratio, "median_interval": median_interval})
        continue

    modified_z = 0.6745 * (all_intervals - median_interval) / mad_scaled
    outlier_mask = modified_z > MODIFIED_Z_THRESHOLD
    outlier_durations = all_intervals[outlier_mask]
    n_outliers = len(outlier_durations)
    rate = n_outliers / total_observed_time if total_observed_time > 0 else 0.0

    print(f"  outliers (modified z > {MODIFIED_Z_THRESHOLD}): {n_outliers}")
    if n_outliers > 0:
        print(f"  outlier duration range: [{outlier_durations.min():.2f}, "
              f"{outlier_durations.max():.2f}], median={np.median(outlier_durations):.2f}")
    print(f"  escape rate: {rate:.5f} events/time-unit")

    summary.append({
        "T_bath": T_bath, "rate": rate, "durations": outlier_durations,
        "recovery_ratio": recovery_ratio, "median_interval": median_interval,
    })

# --- Cross-check: median interval should stay near 3.14 at every T_bath now ---
print("\n" + "=" * 70)
print("CROSS-CHECK: median normal interval by T_bath (should stay ~3.14 "
      "if fragmentation is actually fixed)")
print("=" * 70)
for s in summary:
    flag = "" if abs(s["median_interval"] - 3.14) < 0.15 else "  <-- still drifting"
    print(f"  T_bath={s['T_bath']:<6} median_interval={s['median_interval']:.3f}{flag}")

# --- Plots ---
plot_data = [s["durations"] for s in summary if len(s["durations"]) > 0]
plot_labels = [f"{s['T_bath']}" for s in summary if len(s["durations"]) > 0]
if plot_data:
    ax_dist.boxplot(plot_data, labels=plot_labels, showmeans=True)
    ax_dist.set_xlabel(r"$T_{bath}$")
    ax_dist.set_ylabel("Escape (outlier interval) duration")
    ax_dist.set_title("Escape Duration Distribution vs Noise (smoothed)")
    ax_dist.grid(True, alpha=0.4)
else:
    ax_dist.text(0.5, 0.5, "No outliers detected at any T_bath",
                 ha="center", va="center", transform=ax_dist.transAxes)

Tb = [s["T_bath"] for s in summary]
rates = [s["rate"] for s in summary]
ax_rate.plot(Tb, rates, "o-", markersize=8)
ax_rate.set_xlabel(r"$T_{bath}$")
ax_rate.set_ylabel("Escape rate (events / time unit)")
ax_rate.set_title("Escape Rate vs Noise Strength (smoothed)")
ax_rate.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("sfit_smoothed_outlier_escape_analysis.png", dpi=300)
print("\nSaved: sfit_smoothed_outlier_escape_analysis.png")

# --- Arrhenius check ---
valid = [(s["T_bath"], s["rate"]) for s in summary if s["rate"] > 0]
if len(valid) >= 3:
    Tb_v, rate_v = zip(*valid)
    inv_Tb = 1.0 / np.array(Tb_v)
    ln_rate = np.log(np.array(rate_v))
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(inv_Tb, ln_rate, "o", markersize=8)
    slope, intercept = np.polyfit(inv_Tb, ln_rate, 1)
    ax2.plot(inv_Tb, slope * inv_Tb + intercept, "k--",
              label=f"fit slope = {slope:.4f}")
    ax2.set_xlabel(r"$1/T_{bath}$")
    ax2.set_ylabel(r"$\ln(\mathrm{rate})$")
    ax2.set_title("Arrhenius Check: ln(escape rate) vs 1/T_bath")
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("sfit_arrhenius_rate_check_smoothed.png", dpi=300)
    print("Saved: sfit_arrhenius_rate_check_smoothed.png")
    print("\nFor genuine Kramers escape, slope should be negative "
          "(rate falls as 1/T_bath grows, i.e. as noise weakens).")
else:
    print("\nNot enough T_bath points with detected outliers for an "
          "Arrhenius check yet.")

plt.show()
