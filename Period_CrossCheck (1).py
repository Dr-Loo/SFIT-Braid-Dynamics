import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch, correlate
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 10: FFT / Autocorrelation / Peak-Detector Period Cross-Check")
print("=" * 70)

g_fixed = 0.42
T_bath_values = [0.001, 0.003, 0.005, 0.01, 0.02, 0.04]
T_total = 5000.0
N_TRIALS = 5

PEAK_HEIGHT = 0.12
PEAK_DISTANCE_TIME = 2.0
T_PERIOD_SKELETON = 3.14  # deterministic reference


def period_from_fft(t_arr, C_arr, dt_sample):
    # Remove DC offset before spectral estimation.
    C_detrended = C_arr - np.mean(C_arr)
    freqs, psd = welch(C_detrended, fs=1.0 / dt_sample, nperseg=min(4096, len(C_arr)))
    # Ignore the zero-frequency bin; find the dominant nonzero peak.
    nonzero = freqs > 1e-6
    if not np.any(nonzero):
        return float("nan")
    f_dominant = freqs[nonzero][np.argmax(psd[nonzero])]
    return 1.0 / f_dominant if f_dominant > 0 else float("nan")


def period_from_autocorr(C_arr, dt_sample, max_lag_time=20.0):
    C_detrended = C_arr - np.mean(C_arr)
    max_lag_samples = int(max_lag_time / dt_sample)
    full_corr = correlate(C_detrended, C_detrended, mode="full")
    mid = len(full_corr) // 2
    ac = full_corr[mid: mid + max_lag_samples]
    ac = ac / ac[0]
    # First local maximum after the initial descent from lag 0 (skip the
    # trivial peak at lag 0 itself).
    peaks, _ = find_peaks(ac)
    if len(peaks) == 0:
        return float("nan"), ac
    first_peak_lag = peaks[0] * dt_sample
    return first_peak_lag, ac


def period_from_peaks(t_arr, C_arr, dt_sample):
    distance_samples = max(1, int(PEAK_DISTANCE_TIME / dt_sample))
    peaks, _ = find_peaks(C_arr, height=PEAK_HEIGHT, distance=distance_samples)
    if len(peaks) < 3:
        return float("nan")
    intervals = np.diff(t_arr[peaks])
    return np.median(intervals)


results = []

for T_bath in T_bath_values:
    print(f"\nT_bath = {T_bath}  ({N_TRIALS} trials x T = {T_total})...")

    fft_periods, ac_periods, peak_periods = [], [], []

    for trial in range(N_TRIALS):
        seed = 5000 + int(T_bath * 10000) + trial
        sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)
        t_arr, C_arr = sim.run(T_total=T_total, sample_every=2)
        dt_sample = t_arr[1] - t_arr[0]

        fft_periods.append(period_from_fft(t_arr, C_arr, dt_sample))
        ac_period, _ = period_from_autocorr(C_arr, dt_sample)
        ac_periods.append(ac_period)
        peak_periods.append(period_from_peaks(t_arr, C_arr, dt_sample))

    fft_mean = np.nanmean(fft_periods)
    ac_mean = np.nanmean(ac_periods)
    peak_mean = np.nanmean(peak_periods)

    print(f"  FFT period:            {fft_mean:.3f}")
    print(f"  Autocorrelation period: {ac_mean:.3f}")
    print(f"  Peak-detector period:   {peak_mean:.3f}")

    spread = np.nanmax([fft_mean, ac_mean, peak_mean]) - np.nanmin([fft_mean, ac_mean, peak_mean])
    if spread > 0.15 * T_PERIOD_SKELETON:
        # Identify which method(s) disagree with the other two.
        vals = {"FFT": fft_mean, "Autocorr": ac_mean, "PeakDetector": peak_mean}
        median_val = np.nanmedian(list(vals.values()))
        outliers = [k for k, v in vals.items() if abs(v - median_val) > 0.1 * T_PERIOD_SKELETON]
        print(f"  *** DISAGREEMENT (spread={spread:.3f}): "
              f"{', '.join(outliers) if outliers else 'unclear'} diverges from the others ***")
    else:
        print(f"  -> all three methods agree within tolerance (spread={spread:.3f})")

    results.append({
        "T_bath": T_bath, "fft": fft_mean, "autocorr": ac_mean, "peak": peak_mean,
    })

# --- Summary plot: all three period estimates vs T_bath ---
fig, ax = plt.subplots(figsize=(9, 6))
Tb = [r["T_bath"] for r in results]
ax.plot(Tb, [r["fft"] for r in results], "o-", label="FFT", markersize=8)
ax.plot(Tb, [r["autocorr"] for r in results], "s-", label="Autocorrelation", markersize=8)
ax.plot(Tb, [r["peak"] for r in results], "^-", label="Peak Detector", markersize=8)
ax.axhline(T_PERIOD_SKELETON, color="gray", linestyle="--", alpha=0.6,
           label=f"Deterministic skeleton ({T_PERIOD_SKELETON})")
ax.set_xlabel(r"$T_{bath}$")
ax.set_ylabel("Estimated period")
ax.set_title(f"Period Estimate Cross-Check vs Noise (g={g_fixed})")
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("sfit_period_crosscheck.png", dpi=300)
print("\nSaved: sfit_period_crosscheck.png")

print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'T_bath':<10}{'FFT':<10}{'Autocorr':<12}{'PeakDet':<10}{'Spread':<10}")
for r in results:
    spread = max(r["fft"], r["autocorr"], r["peak"]) - min(r["fft"], r["autocorr"], r["peak"])
    print(f"{r['T_bath']:<10}{r['fft']:<10.3f}{r['autocorr']:<12.3f}{r['peak']:<10.3f}{spread:<10.3f}")

plt.show()
