import numpy as np

def generate_xy_mc_corrected(L, T, n_discard=1000, n_meas=3000,
                              measure_interval=100, seed=None):
    """
    CORRECTED version of the function that produced the finished paper's
    Table 2. The original ran n_discard equilibration sweeps and returned
    a SINGLE snapshot immediately -- no measurement phase at all, despite
    Section 2.2 describing n_meas=3000 measurement sweeps with per-sweep
    sampling.

    This version actually runs the measurement phase and returns a LIST
    of wrapped snapshots, sampled at `measure_interval` sweeps apart
    (not every single sweep -- see note on `measure_interval` below).

    Parameters
    ----------
    n_discard : int
        Equilibration sweeps before measurement begins (unchanged: 1000).
    n_meas : int
        Total length of the measurement phase, in sweeps (unchanged: 3000).
    measure_interval : int
        Take one snapshot every `measure_interval` sweeps within the
        measurement phase, rather than every single sweep.

        Two reasons for this, not just performance:
        1. Successive Metropolis sweeps are strongly autocorrelated --
           measuring every single sweep oversamples nearly-identical
           configurations and doesn't actually buy much additional
           independent information per unit of compute.
        2. Performance is not optional here: TDA density uses GUDHI
           persistent homology, which is expensive per call. Measuring
           all 3000 sweeps per seed would mean 15 seeds x 19 T x 3000 =
           855,000 GUDHI calls -- at the original pipeline's demonstrated
           rate (~45 min for 15 seeds x 19 T x 1 snapshot = 285 calls),
           that's on the order of MONTHS of runtime, not a rerun you can
           practically do.

        With measure_interval=100, you get n_meas/measure_interval = 30
        snapshots per seed -- 15 x 19 x 30 = 8,550 total measurement
        calls, ~30x the original pipeline's cost. At the original's
        demonstrated rate, that's roughly 22-24 hours. Still a real
        commitment, but tractable overnight rather than impossible.
        Adjust measure_interval up (fewer, cheaper measurements) or down
        (more, costlier, better-averaged) depending on your compute
        budget -- but do NOT set it to 1 without recalculating the
        expected runtime first.

    Returns
    -------
    list of np.ndarray
        n_meas // measure_interval wrapped (L, L) configurations, one
        per sampled measurement sweep. The caller should compute
        vortex/TDA density on EACH snapshot and average over the list
        for that seed, then average again across seeds -- this is the
        two-level averaging (within-seed, then across-seed) that
        Section 2.2 actually describes and that the original code never
        implemented.
    """
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

    # --- Equilibration phase (identical to the original) ---
    for _ in range(n_discard):
        theta = one_sweep(theta)

    # --- Measurement phase: the part that was MISSING entirely before ---
    snapshots = []
    for sweep_idx in range(1, n_meas + 1):
        theta = one_sweep(theta)
        if sweep_idx % measure_interval == 0:
            # Wrap at sampling time -- the dynamics itself is shift-invariant
            # (cos() doesn't care about accumulated 2*pi drift), so wrapping
            # only when we sample, rather than every step, doesn't change
            # the physics, only when we pay the wrapping cost.
            snapshots.append(np.angle(np.exp(1j * theta)))

    return snapshots
