import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 6: T_bath Sweep at Fixed g (Peak-Interval Detection)")
print("=" * 70)

g_fixed = 0.42
T_bath_values = [0.001, 0.003, 0.005, 0.01, 0.02, 0.04]
T_total = 5000.0
N_TRIALS = 5
MIN_EVENTS_FOR_FIT = 10

T_PERIOD_REFERENCE = 3.14
ESCAPE_MULTIPLE = 2.0
NOISE_TOLERANCE = 0.15
PEAK_DISTANCE_TIME = 2.0    # minimum time between accepted peaks; adjust to match your fixed distance param, in time units

# Fixed peak-height threshold, calibrated from the noiseless (T_bath=0) skeleton
# at this specific g, NOT adaptive to each noisy run's own max (that reintroduces
# the anti-correlation bug: noise spikes push the adaptive floor up and cause
# missed peaks at higher T_bath, faking a "more noise -> fewer peaks" artifact).
# Replace this with the actual C_max from your own noiseless run at g_fixed.
C_MAX_NOISELESS_SKELETON = 0.1892  # from earlier printed output at g=0.42
PEAK_HEIGHT_THRESHOLD = 0.75 * C_MAX_NOISELESS_SKELETON

fig, ax = plt.subplots(figsize=(10, 6))
summary = []

for T_bath in T_bath_values:
    print(f"\nRunning g = {g_fixed:.2f}, T_bath = {T_bath}  ({N_TRIALS} trials x T = {T_total})...")

    all_escape_durations = []
    all_intervals = []

    for trial in range(N_TRIALS):
        seed = 2000 + int(T_bath * 10000) + trial
        sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)
        t_arr, C_arr = sim.run(T_total=T_total, sample_every=2)
        dt_sample = t_arr[1] - t_arr[0]
        distance_samples = max(1, int(PEAK_DISTANCE_TIME / dt_sample))

        peaks, _ = find_peaks(C_arr, height=PEAK_HEIGHT_THRESHOLD, distance=distance_samples)

        if len(peaks) < 3:
            continue

        peak_times = t_arr[peaks]
        intervals = np.diff(peak_times)
        all_intervals.extend(intervals.tolist())

        escape_mask = intervals > (ESCAPE_MULTIPLE * T_PERIOD_REFERENCE)
        all_escape_durations.extend(intervals[escape_mask].tolist())

    all_intervals = np.array(all_intervals)
    all_escape_durations = np.array(all_escape_durations)
    n_events = len(all_escape_durations)

    frac_normal = (np.mean(np.abs(all_intervals - T_PERIOD_REFERENCE) <= NOISE_TOLERANCE * T_PERIOD_REFERENCE)
                   if len(all_intervals) else float("nan"))

    print(f"  total inter-peak intervals: {len(all_intervals)}")
    print(f"  fraction matching normal period: {frac_normal:.3f}")
    print(f"  escape events: {n_events}")

    tau = None
    if n_events >= MIN_EVENTS_FOR_FIT:
        durations = np.sort(all_escape_durations)
        t_vals = np.linspace(durations.min(), durations.max(), 200)
        S_t = np.array([np.mean(durations > t) for t in t_vals])

        ax.plot(t_vals, S_t, linewidth=2, label=f"T_bath={T_bath} (N={n_events})")

        mask = (S_t > 0.02) & (S_t < 0.9)
        if np.sum(mask) >= 5:
            t_fit, S_fit = t_vals[mask], S_t[mask]
            slope, intercept = np.polyfit(t_fit, np.log(S_fit), 1)
            if slope < -1e-6:
                tau = -1.0 / slope
                print(f"  -> tau = {tau:.2f}, Gamma = {1.0/tau:.5f}")
            else:
                print("  -> slope ~0; reject fit")
        else:
            print("  -> not enough populated tail to fit")
    else:
        print(f"  -> too few events (N={n_events} < {MIN_EVENTS_FOR_FIT}) to fit")

    summary.append({"T_bath": T_bath, "n_events": n_events, "tau": tau})

ax.set_yscale("log")
ax.set_xlabel(r"Escape-Interval Duration $\Delta t$", fontsize=13)
ax.set_ylabel(r"Survival Probability $S(t)$", fontsize=13)
ax.set_title(f"Escape Statistics vs Noise Strength (g={g_fixed})", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("sfit_tbath_sweep.png", dpi=300)
print("\nSaved: sfit_tbath_sweep.png")

# Arrhenius check: ln(tau) vs 1/T_bath should be roughly linear if Kramers-like
valid = [(s["T_bath"], s["tau"]) for s in summary if s["tau"] is not None and s["tau"] > 0]
if len(valid) >= 3:
    Tb, taus = zip(*valid)
    inv_Tb = 1.0 / np.array(Tb)
    ln_tau = np.log(np.array(taus))
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(inv_Tb, ln_tau, "o", markersize=8)
    slope2, intercept2 = np.polyfit(inv_Tb, ln_tau, 1)
    ax2.plot(inv_Tb, slope2 * inv_Tb + intercept2, "k--",
              label=f"fit slope (barrier estimate) = {slope2:.4f}")
    ax2.set_xlabel(r"$1/T_{bath}$")
    ax2.set_ylabel(r"$\ln \tau$")
    ax2.set_title("Arrhenius Check: ln(tau) vs 1/T_bath")
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("sfit_arrhenius_check.png", dpi=300)
    print("Saved: sfit_arrhenius_check.png")
else:
    print("\nNot enough valid tau estimates yet for an Arrhenius check "
          "(need >=3 T_bath values with a successful fit).")

plt.show()