import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 7: Outlier-Based Escape Detection (No Fixed Duration Cutoff)")
print("=" * 70)

g_fixed = 0.42
T_bath_values = [0.001, 0.003, 0.005, 0.01, 0.02, 0.04]
T_total = 5000.0
N_TRIALS = 5

# Validated peak-detection parameters (from Parameter_accumulation_verification.py,
# 0.978 peak-recovery ratio at T_bath=0.001). Re-validate at each T_bath before
# trusting the sweep -- see the peak-count check printed below.
PEAK_HEIGHT = 0.12
PEAK_DISTANCE_TIME = 2.0

# Robust-outlier threshold on the modified z-score (Iglewicz & Hoaglin convention;
# 3.5 is the standard "probable outlier" cutoff). No fixed multiple of the period
# is used anywhere -- each T_bath's own interval distribution sets its own bar.
MODIFIED_Z_THRESHOLD = 3.5

fig, (ax_dist, ax_rate) = plt.subplots(1, 2, figsize=(14, 5.5))
summary = []

for T_bath in T_bath_values:
    print(f"\nRunning g = {g_fixed:.2f}, T_bath = {T_bath}  "
          f"({N_TRIALS} trials x T = {T_total})...")

    all_intervals = []          # pooled across trials -- accumulator declared
    all_peak_counts = []        # OUTSIDE the trial loop; extend(), never overwrite
    total_observed_time = 0.0

    for trial in range(N_TRIALS):
        seed = 3000 + int(T_bath * 10000) + trial
        sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)
        t_arr, C_arr = sim.run(T_total=T_total, sample_every=2)
        dt_sample = t_arr[1] - t_arr[0]
        distance_samples = max(1, int(PEAK_DISTANCE_TIME / dt_sample))

        peaks, _ = find_peaks(C_arr, height=PEAK_HEIGHT, distance=distance_samples)
        all_peak_counts.append(len(peaks))

        if len(peaks) < 3:
            continue

        peak_times = t_arr[peaks]
        intervals = np.diff(peak_times)
        all_intervals.extend(intervals.tolist())
        total_observed_time += t_arr[-1] - t_arr[0]

    all_intervals = np.array(all_intervals)
    n_peaks_total = sum(all_peak_counts)
    expected_peaks_per_trial = T_total / 3.14  # deterministic-skeleton period
    expected_total = expected_peaks_per_trial * N_TRIALS
    recovery_ratio = n_peaks_total / expected_total if expected_total else float("nan")

    print(f"  peak count per trial: {all_peak_counts}")
    print(f"  total peaks: {n_peaks_total}  (expected ~{expected_total:.0f}, "
          f"ratio {recovery_ratio:.3f})")

    if recovery_ratio < 0.5 or recovery_ratio > 1.3:
        print("  *** WARNING: peak recovery ratio out of sane range -- "
              "do not trust this T_bath's escape statistics until the "
              "detector is re-validated at this noise level ***")

    if len(all_intervals) < 20:
        print("  -> too few intervals to build a distribution")
        summary.append({"T_bath": T_bath, "n_outliers": 0, "rate": 0.0,
                         "durations": np.array([]), "recovery_ratio": recovery_ratio})
        continue

    # Robust statistics: median + MAD instead of mean/std, since a handful of
    # real long escapes would otherwise inflate the mean and hide themselves.
    median_interval = np.median(all_intervals)
    mad = np.median(np.abs(all_intervals - median_interval))
    mad_scaled = mad * 1.4826 if mad > 0 else np.std(all_intervals)  # normal-consistent scale

    if mad_scaled == 0:
        print("  -> zero MAD (perfectly regular intervals); no outliers possible")
        summary.append({"T_bath": T_bath, "n_outliers": 0, "rate": 0.0,
                         "durations": np.array([]), "recovery_ratio": recovery_ratio})
        continue

    modified_z = 0.6745 * (all_intervals - median_interval) / mad_scaled
    outlier_mask = modified_z > MODIFIED_Z_THRESHOLD  # one-sided: only long intervals count as escapes
    outlier_durations = all_intervals[outlier_mask]
    n_outliers = len(outlier_durations)

    # Rate = events per unit observed time -- this is the physically meaningful
    # Kramers-comparable quantity, decoupled from how long each event happens
    # to last. A fixed-duration cutoff conflates frequency and duration; this
    # doesn't.
    rate = n_outliers / total_observed_time if total_observed_time > 0 else 0.0

    print(f"  normal interval: median={median_interval:.3f}, "
          f"MAD(scaled)={mad_scaled:.3f}")
    print(f"  outliers (modified z > {MODIFIED_Z_THRESHOLD}): {n_outliers}")
    if n_outliers > 0:
        print(f"  outlier duration range: [{outlier_durations.min():.2f}, "
              f"{outlier_durations.max():.2f}], median={np.median(outlier_durations):.2f}")
    print(f"  escape rate: {rate:.5f} events/time-unit")

    summary.append({
        "T_bath": T_bath,
        "n_outliers": n_outliers,
        "rate": rate,
        "durations": outlier_durations,
        "recovery_ratio": recovery_ratio,
    })

# --- Plot 1: distribution of outlier (escape) durations across T_bath ---
plot_data = [s["durations"] for s in summary if len(s["durations"]) > 0]
plot_labels = [f"{s['T_bath']}" for s in summary if len(s["durations"]) > 0]
if plot_data:
    ax_dist.boxplot(plot_data, labels=plot_labels, showmeans=True)
    ax_dist.set_xlabel(r"$T_{bath}$")
    ax_dist.set_ylabel("Escape (outlier interval) duration")
    ax_dist.set_title("Escape Duration Distribution vs Noise")
    ax_dist.grid(True, alpha=0.4)
else:
    ax_dist.text(0.5, 0.5, "No outliers detected at any T_bath",
                 ha="center", va="center", transform=ax_dist.transAxes)

# --- Plot 2: escape rate vs T_bath -- the actual Kramers-comparable observable ---
Tb = [s["T_bath"] for s in summary]
rates = [s["rate"] for s in summary]
ax_rate.plot(Tb, rates, "o-", markersize=8)
ax_rate.set_xlabel(r"$T_{bath}$")
ax_rate.set_ylabel("Escape rate (events / time unit)")
ax_rate.set_title("Escape Rate vs Noise Strength")
ax_rate.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("sfit_outlier_escape_analysis.png", dpi=300)
print("\nSaved: sfit_outlier_escape_analysis.png")

# --- Arrhenius check on rate directly (not on a duration-filtered count) ---
valid = [(s["T_bath"], s["rate"]) for s in summary if s["rate"] > 0]
if len(valid) >= 3:
    Tb_v, rate_v = zip(*valid)
    inv_Tb = 1.0 / np.array(Tb_v)
    ln_rate = np.log(np.array(rate_v))
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(inv_Tb, ln_rate, "o", markersize=8)
    slope, intercept = np.polyfit(inv_Tb, ln_rate, 1)
    ax2.plot(inv_Tb, slope * inv_Tb + intercept, "k--",
              label=f"fit slope (−barrier estimate) = {slope:.4f}")
    ax2.set_xlabel(r"$1/T_{bath}$")
    ax2.set_ylabel(r"$\ln(\mathrm{rate})$")
    ax2.set_title("Arrhenius Check: ln(escape rate) vs 1/T_bath")
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("sfit_arrhenius_rate_check.png", dpi=300)
    print("Saved: sfit_arrhenius_rate_check.png")
    print("\nNote: for genuine Kramers escape, ln(rate) should DECREASE "
          "linearly with 1/T_bath (i.e. slope should be negative). Check "
          "sign and linearity before interpreting the fit.")
else:
    print("\nNot enough T_bath points with detected outliers for an "
          "Arrhenius check yet.")

plt.show()
