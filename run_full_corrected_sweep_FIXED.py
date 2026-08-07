import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import json
import os
import time
from tqdm import tqdm

# ==============================================================================
# CORRECTED full sweep: uses generate_xy_mc_corrected's actual measurement
# phase (previously missing entirely) and does proper two-level averaging
# (within-seed over measurement snapshots, then across-seed).
#
# Expect roughly 20-30x the original pipeline's runtime (~45 min -> ~1 day)
# given measure_interval=100 (30 snapshots/seed instead of 1). Checkpointing
# is included so a crash or interruption doesn't lose completed temperatures.
# ==============================================================================


def compute_vortex_density(theta):
    L = theta.shape[0]
    n_vortices = 0
    for i in range(L):
        for j in range(L):
            t00 = theta[i, j]
            t10 = theta[(i + 1) % L, j]
            t11 = theta[(i + 1) % L, (j + 1) % L]
            t01 = theta[i, (j + 1) % L]
            d1 = np.angle(np.exp(1j * (t10 - t00)))
            d2 = np.angle(np.exp(1j * (t11 - t10)))
            d3 = np.angle(np.exp(1j * (t01 - t11)))
            d4 = np.angle(np.exp(1j * (t00 - t01)))
            sum_d = d1 + d2 + d3 + d4
            q = np.round(sum_d / (2 * np.pi))
            if np.abs(q) > 0.5:
                n_vortices += 1
    return n_vortices / (L * L)


def compute_significant_1_cycles(theta, tau_fraction=0.15):
    L = theta.shape[0]
    cc = gd.CubicalComplex(dimensions=theta.shape, top_dimensional_cells=theta.flatten())
    cc.compute_persistence()
    diag_1 = cc.persistence_intervals_in_dimension(1)
    finite_diag_1 = [interval for interval in diag_1 if interval[1] != np.inf]
    if not finite_diag_1:
        return 0.0
    lifespans = np.array([interval[1] - interval[0] for interval in finite_diag_1])
    tau = tau_fraction * np.max(lifespans)
    return np.sum(lifespans > tau) / (L * L)


def generate_xy_mc_corrected(L, T, n_discard=1000, n_meas=3000,
                              measure_interval=100, seed=None):
    """CORRECTED: actually runs the n_meas measurement phase described in
    Section 2.2, sampling every `measure_interval` sweeps. See the
    standalone version of this function for the full explanation of why
    measure_interval is not 1."""
    if seed is not None:
        np.random.seed(seed)
    theta = np.random.uniform(-np.pi, np.pi, (L, L))

    def one_sweep(theta):
        for i in range(L):
            for j in range(L):
                e_old = -(np.cos(theta[i, j] - theta[(i - 1) % L, j])
                          + np.cos(theta[i, j] - theta[(i + 1) % L, j])
                          + np.cos(theta[i, j] - theta[i, (j - 1) % L])
                          + np.cos(theta[i, j] - theta[i, (j + 1) % L]))
                theta_new = theta[i, j] + np.random.uniform(-0.8, 0.8)
                e_new = -(np.cos(theta_new - theta[(i - 1) % L, j])
                          + np.cos(theta_new - theta[(i + 1) % L, j])
                          + np.cos(theta_new - theta[i, (j - 1) % L])
                          + np.cos(theta_new - theta[i, (j + 1) % L]))
                if np.random.rand() < np.exp(-(e_new - e_old) / T):
                    theta[i, j] = theta_new
        return theta

    for _ in range(n_discard):
        theta = one_sweep(theta)

    snapshots = []
    for sweep_idx in range(1, n_meas + 1):
        theta = one_sweep(theta)
        if sweep_idx % measure_interval == 0:
            snapshots.append(np.angle(np.exp(1j * theta)))

    return snapshots


CHECKPOINT_FILE = "corrected_sweep_checkpoint.json"


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        # JSON keys are strings; convert back to float temperatures
        return {float(k): v for k, v in data.items()}
    return {}


def save_checkpoint(results_by_T):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({str(k): v for k, v in results_by_T.items()}, f)


def run_full_corrected_sweep(measure_interval=100, n_meas=3000):
    temperatures = np.linspace(0.60, 1.50, 19)
    L = 32
    n_seeds = 15
    n_expected_snapshots = n_meas // measure_interval

    print(f"Running CORRECTED full n={n_seeds} sweep.")
    print(f"measure_interval={measure_interval} -> {n_expected_snapshots} "
          f"snapshots/seed (vs. original's 1).")
    print(f"Expect roughly {n_expected_snapshots}x the original ~45 min "
          f"runtime -- this WILL take many hours. Checkpointing to "
          f"'{CHECKPOINT_FILE}' after each completed temperature.")

    # Resume from checkpoint if one exists.
    results_by_T = load_checkpoint()
    if results_by_T:
        print(f"Resuming: {len(results_by_T)} temperature(s) already "
              f"completed in a previous run.")

    start_time = time.time()

    for T_idx, T in enumerate(temperatures):
        if T in results_by_T:
            print(f"\nT={T:.2f}: already completed, skipping.")
            continue

        print(f"\nT={T:.2f} ({T_idx+1}/{len(temperatures)})...")
        seed_vortex_means = []
        seed_tda_means = []

        for seed_idx in tqdm(range(n_seeds), desc=f"  Seeds (T={T:.2f})"):
            unique_seed = seed_idx * 5000 + int(T * 100)
            snapshots = generate_xy_mc_corrected(
                L, T, n_discard=1000, n_meas=n_meas,
                measure_interval=measure_interval, seed=unique_seed
            )

            vortex_vals = [compute_vortex_density(s) for s in snapshots]
            tda_vals = [compute_significant_1_cycles(s, tau_fraction=0.15)
                        for s in snapshots]

            # Within-seed average over the measurement-phase snapshots.
            seed_vortex_means.append(np.mean(vortex_vals))
            seed_tda_means.append(np.mean(tda_vals))

        # Across-seed average and SEM.
        vortex_mean = float(np.mean(seed_vortex_means))
        vortex_sem = float(np.std(seed_vortex_means, ddof=1) / np.sqrt(n_seeds))
        tda_mean = float(np.mean(seed_tda_means))
        tda_sem = float(np.std(seed_tda_means, ddof=1) / np.sqrt(n_seeds))

        results_by_T[T] = {
            "vortex_mean": vortex_mean, "vortex_sem": vortex_sem,
            "tda_mean": tda_mean, "tda_sem": tda_sem,
        }
        save_checkpoint(results_by_T)

        elapsed = time.time() - start_time
        print(f"  vortex = {vortex_mean:.5f} +/- {vortex_sem:.5f}   "
              f"tda = {tda_mean:.5f} +/- {tda_sem:.5f}")
        print(f"  [checkpoint saved; elapsed {elapsed/3600:.2f} h]")

    # --- Final table ---
    T_array = np.array(sorted(results_by_T.keys()))
    vortex_mean = np.array([results_by_T[T]["vortex_mean"] for T in T_array])
    vortex_sem = np.array([results_by_T[T]["vortex_sem"] for T in T_array])
    tda_mean = np.array([results_by_T[T]["tda_mean"] for T in T_array])
    tda_sem = np.array([results_by_T[T]["tda_sem"] for T in T_array])

    print("\n" + "=" * 80)
    print("CORRECTED FULL n=15 SWEEP RESULTS (proper measurement-phase averaging)")
    print("=" * 80)
    print(f"{'T':<6} | {'TDA Mean':<10} | {'TDA SEM':<10} | {'Vortex Mean':<12} | {'Vortex SEM':<12}")
    print("-" * 80)
    for i in range(len(T_array)):
        print(f"{T_array[i]:<6.2f} | {tda_mean[i]:<10.4f} | {tda_sem[i]:<10.4f} | "
              f"{vortex_mean[i]:<12.4f} | {vortex_sem[i]:<12.4f}")
    print("=" * 80)

    # Sanity check against the paper's original (buggy) T=0.60 value.
    if 0.60 in results_by_T:
        print(f"\nSanity check: original (buggy, single-snapshot) T=0.60 gave "
              f"vortex=0.0027+/-0.0005.")
        print(f"Corrected (proper measurement-averaged) T=0.60 gives "
              f"vortex={results_by_T[0.60]['vortex_mean']:.4f}"
              f"+/-{results_by_T[0.60]['vortex_sem']:.4f}.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(T_array, vortex_mean, yerr=vortex_sem, fmt='s-', color='purple',
                alpha=0.7, label='Vortex Density (corrected averaging)', capsize=4)
    ax.errorbar(T_array, tda_mean, yerr=tda_sem, fmt='o-', color='teal',
                label='TDA Density (corrected averaging)', capsize=4)
    ax.axvline(0.893, color='red', linestyle='--', label=r'$T_{BKT} \approx 0.893$')
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel('Density')
    ax.set_title('Corrected Curves: Proper Measurement-Phase Averaging')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("corrected_sweep_proper_averaging.png", dpi=300)
    print("\nSaved: corrected_sweep_proper_averaging.png")
    plt.show()


if __name__ == "__main__":
    run_full_corrected_sweep()
