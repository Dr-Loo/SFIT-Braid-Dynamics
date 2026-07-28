import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 5: Peak-Interval Escape Detection (Threshold-Free)")
print("=" * 70)

g_values = [0.42, 0.48, 0.58, 0.65]
T_total = 5000.0
N_TRIALS = 5          # independent noise realizations per g, for adequate event counts
MIN_EVENTS_FOR_FIT = 10

# Deterministic reference period, from the noiseless skeleton (T_bath=0) runs.
# Recompute per-g if you suspect the period is not exactly g-independent —
# see the flagged concern about the identical T=3.14 across all four couplings.
T_PERIOD_REFERENCE = 3.14
ESCAPE_MULTIPLE = 2.0     # an inter-peak gap > ESCAPE_MULTIPLE * T_period counts as an escape
NOISE_TOLERANCE = 0.15    # allowed fractional jitter around T_period before calling it "normal"

fig, ax = plt.subplots(figsize=(10, 6))
results = {}

for g in g_values:
    print(f"\nRunning g = {g:.2f}  ({N_TRIALS} trials x T = {T_total})...")

    all_intervals = []
    all_escape_durations = []

    for trial in range(N_TRIALS):
        seed = 1000 + int(g * 100) * 100 + trial
        sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g, T_bath=0.001, seed=seed)
        t_arr, C_arr = sim.run(T_total=T_total, sample_every=2)
        dt_sample = t_arr[1] - t_arr[0]

        # Per-run adaptive peak threshold instead of a hardcoded absolute value
        # (fixes the g=0.42 false-negative risk from the original 0.18 floor).
        run_peak_floor = 0.5 * np.max(C_arr)
        peaks, _ = find_peaks(C_arr, height=run_peak_floor)

        if len(peaks) < 3:
            print(f"  trial {trial}: too few peaks detected (N={len(peaks)}), skipping")
            continue

        peak_times = t_arr[peaks]
        intervals = np.diff(peak_times)
        all_intervals.extend(intervals.tolist())

        # An interval much longer than the deterministic period is a genuine
        # escape: the field failed to complete another oscillation on schedule.
        escape_mask = intervals > (ESCAPE_MULTIPLE * T_PERIOD_REFERENCE)
        escape_durs = intervals[escape_mask]
        all_escape_durations.extend(escape_durs.tolist())

    all_intervals = np.array(all_intervals)
    all_escape_durations = np.array(all_escape_durations)

    normal_mask = np.abs(all_intervals - T_PERIOD_REFERENCE) <= NOISE_TOLERANCE * T_PERIOD_REFERENCE
    frac_normal = np.mean(normal_mask) if len(all_intervals) else float("nan")

    print(f"  total inter-peak intervals: {len(all_intervals)}")
    print(f"  fraction matching normal period (within {NOISE_TOLERANCE*100:.0f}%): {frac_normal:.3f}")
    print(f"  escape events (interval > {ESCAPE_MULTIPLE}x period): {len(all_escape_durations)}")

    results[g] = {
        "intervals": all_intervals,
        "escapes": all_escape_durations,
        "n_events": len(all_escape_durations),
    }

    if len(all_escape_durations) >= MIN_EVENTS_FOR_FIT:
        durations = np.sort(all_escape_durations)
        t_vals = np.linspace(durations.min(), durations.max(), 200)
        S_t = np.array([np.mean(durations > t) for t in t_vals])

        ax.plot(t_vals, S_t, linewidth=2.5,
                label=f"g={g:.2f} (N={len(durations)})")

        # Fit only the well-populated part of the tail; guard against a
        # near-empty/near-zero-slope fit blowing up like the earlier bug.
        mask = (S_t > 0.02) & (S_t < 0.9)
        if np.sum(mask) >= 5:
            t_fit, S_fit = t_vals[mask], S_t[mask]
            slope, intercept = np.polyfit(t_fit, np.log(S_fit), 1)
            if slope < -1e-6:  # must be a genuine decay, not noise-flat
                tau = -1.0 / slope
                print(f"  -> Kramers residence time tau = {tau:.2f}")
                print(f"  -> Escape rate Gamma = {1.0/tau:.5f}")
                t_line = np.array([t_fit[0], t_fit[-1]])
                ax.plot(t_line, np.exp(slope * t_line + intercept),
                        "k--", linewidth=1.2, alpha=0.7)
            else:
                print("  -> Fitted slope ~0 or positive; reject (insufficient decay signal)")
        else:
            print("  -> Not enough populated tail points to fit reliably")
    else:
        print(f"  -> Too few escape events (N={len(all_escape_durations)} < {MIN_EVENTS_FOR_FIT}) to fit")

ax.set_yscale("log")
ax.set_xlabel(r"Escape-Interval Duration $\Delta t$", fontsize=14)
ax.set_ylabel(r"Survival Probability $S(t)$", fontsize=14)
ax.set_title("Kramers Residence Times via Peak-Interval Detection", fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("sfit_peak_interval_escape.png", dpi=300)
print("\nSaved: sfit_peak_interval_escape.png")
plt.show()