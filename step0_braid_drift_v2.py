"""
step0_braid_drift_v2.py -- Step 0: does the Part I braid action make channel populations
a martingale?  (Rewrite of step0_braid_drift.py; supersedes the "v2" draft.)

QUESTION
  Under the Part I crossing process, is E[x(t+1) - x(t) | F_t] = 0, where x_k = psi_k^2?
  A martingale is required by the fair-absorbing-exchange route to Born weights.

DYNAMICS (Part I v6 planar substrate, sfit_substrate_v6_2D_braid_structure.py, unchanged)
  N = 3 defects, speed c = 2, Poisson direction scattering gamma = 2 (uniform kick in
  [-pi, pi]), harmonic trap k = 0.1, Euler step dt. Crossings = changes of x-ordering.
  All seeds are integrated together (vectorised); each seed is an independent realisation.

REPRESENTATIONS
  R8  v8 helicity action (Part I -> Part II integration script): a crossing of particles p, q
      swaps their helicities -> 8x8 permutation matrix on the 3-bit helicity basis.
      Sign-independent, so a non-adjacent order change (several swaps inside one step) is
      applied EXACTLY as the net permutation; nothing is dropped for R8.
  R2  v7 hand-specified B3 rep, rho(s1) = [[1,1],[0,1]], rho(s2) = [[1,0],[-1,1]].
      Not orthogonal; shares are renormalised after each crossing (projective dynamics,
      reported as a secondary diagnostic only). Sign-dependent, so non-adjacent order
      changes cannot be decoded and are dropped AND counted.

MEASUREMENTS (all fixed before running)
  M1  Signed one-crossing drift E[dx | x] = (Pbar_s - I) x for R8, per seed s, at 200 fixed
      test points x (Dirichlet(1), seed 7). 95% CI across seeds (t-interval, 9 dof).
  M2  Finer filtration: the same drift conditioned on the x-ordering of the defects just
      before the crossing (the physical state that determines which pair can cross).
  M3  POSITIVE CONTROL (must give zero drift): populations of the S3 isotypic sectors,
      ||P_sector psi||^2, and of the helicity-weight classes. Both commute with every R8,
      so their drift must vanish to machine precision; otherwise the run is INVALID.
  M4  Exact invariance along the ORDERED trajectory: sort(x) unchanged after every crossing.
  M5  R2 secondary: signed drift of x_1 binned in x_1, CI across seeds.
  M6  Extractor completeness vs dt on seed 1 (unresolved = order changes that are not a
      single adjacent swap); reported, not asserted.

DECISION (computed, printed from the numbers)
  Martingale REJECTED for a representation if, at some test point, the 95% CI of some
  drift component excludes 0. The positive control must pass for the run to be valid.

PRE-REGISTRATION
  This script writes nothing to "pre-register" itself. Deposit this file and its SHA-256
  (printed with --hash) on Zenodo/OSF BEFORE the production run and cite that DOI.

Usage: python step0_braid_drift_v2.py [--quick] [--hash]
"""
import sys, hashlib, pathlib, itertools, json
import numpy as np

QUICK = "--quick" in sys.argv
DT = 0.005
T_RUN = 2000.0 if QUICK else 20000.0
SEEDS = list(range(1, 11))
N_TEST = 200
C, GAMMA, K_TRAP, N = 2.0, 2.0, 0.1, 3
T95_9DOF = 2.262          # two-sided 95% t quantile, 9 degrees of freedom (10 seeds)


# --------------------------------------------------------------------------- dynamics
def simulate(seeds, T, dt):
    """Integrate all seeds together. Returns, per seed, a list of crossing events
    (k, sign, p, q, order_before) for clean adjacent swaps, the list of net permutations
    for every order change (for R8), and counts."""
    S = len(seeds)
    rngs = [np.random.default_rng(s) for s in seeds]
    X = np.stack([r.uniform(-3, 3, (N, 2)) for r in rngs])          # (S, N, 2)
    th = np.stack([r.uniform(0, 2 * np.pi, N) for r in rngs])        # (S, N)
    noise = np.random.default_rng(12345 + sum(seeds))                # shared stream, independent per seed slot
    prev = np.argsort(X[:, :, 0], axis=1)
    clean = [[] for _ in range(S)]
    netperm = [[] for _ in range(S)]
    n_change = np.zeros(S, int); n_unres = np.zeros(S, int)
    for _ in range(int(T / dt)):
        v = C * np.stack([np.cos(th), np.sin(th)], axis=2)
        X += (v - K_TRAP * X) * dt
        hit = noise.random((S, N)) < GAMMA * dt
        if hit.any():
            th[hit] += noise.uniform(-np.pi, np.pi, hit.sum())
        cur = np.argsort(X[:, :, 0], axis=1)
        changed = np.flatnonzero((cur != prev).any(1))
        for s in changed:
            o, c_ = prev[s], cur[s]
            n_change[s] += 1
            netperm[s].append((o.copy(), c_.copy()))
            d = np.flatnonzero(o != c_)
            if d.size == 2 and d[1] - d[0] == 1:
                k = d[0]; p, q = c_[k], c_[k + 1]
                clean[s].append((int(k), 1 if X[s, p, 1] > X[s, q, 1] else -1, int(p), int(q), tuple(int(z) for z in o)))
            else:
                n_unres[s] += 1
        prev = cur
    return clean, netperm, n_change, n_unres


# --------------------------------------------------------------------------- representations
def idx(bits):
    return bits[0] * 4 + bits[1] * 2 + bits[2]


def R8_from_orders(before, after):
    """Helicity stays with the RANK position while particles move (each crossing swaps the
    two particles' helicities). Net effect on helicity-by-particle: particle after[r] takes
    the helicity that particle before[r] had. Exact for any order change."""
    M = np.zeros((8, 8))
    for i in range(8):
        h = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
        h2 = [0, 0, 0]
        for r in range(N):
            h2[after[r]] = h[before[r]]
        M[idx(h2), i] = 1
    return M


def R8_pair(p, q):
    M = np.zeros((8, 8))
    for i in range(8):
        h = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
        h[p], h[q] = h[q], h[p]
        M[idx(h), i] = 1
    return M


A1 = np.array([[1., 1.], [0., 1.]]); A2 = np.array([[1., 0.], [-1., 1.]])
R2 = {(0, 1): A1, (0, -1): np.linalg.inv(A1), (1, 1): A2, (1, -1): np.linalg.inv(A2)}


def s3_group_8d():
    """All six particle permutations acting on the helicity basis."""
    mats = []
    for perm in itertools.permutations(range(N)):
        M = np.zeros((8, 8))
        for i in range(8):
            h = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
            h2 = [h[perm.index(j)] for j in range(N)]
            M[idx(h2), i] = 1
        mats.append((perm, M))
    return mats


def isotypic_projectors():
    """Character projectors of S3 on the 8D helicity space: trivial, sign, standard (V)."""
    def parity(p):
        return np.prod([np.sign(p[j] - p[i]) for i in range(N) for j in range(i + 1, N)])
    def cycle_type(p):
        fixed = sum(p[i] == i for i in range(N))
        return {3: "e", 1: "t", 0: "c"}[fixed]
    chi = {"triv": {"e": 1, "t": 1, "c": 1}, "sign": {"e": 1, "t": -1, "c": 1},
           "V": {"e": 2, "t": 0, "c": -1}}
    dim = {"triv": 1, "sign": 1, "V": 2}
    G = s3_group_8d()
    return {name: dim[name] / 6 * sum(chi[name][cycle_type(p)] * M for p, M in G)
            for name in chi}


# --------------------------------------------------------------------------- statistics
def mean_ci(a, axis=0):
    a = np.asarray(a, float)
    m = a.mean(axis); se = a.std(axis, ddof=1) / np.sqrt(a.shape[axis])
    return m, m - T95_9DOF * se, m + T95_9DOF * se


def main():
    here = pathlib.Path(__file__).resolve()
    digest = hashlib.sha256(here.read_bytes()).hexdigest()
    if "--hash" in sys.argv:
        print(digest); return
    print("=" * 92)
    print(f"Step 0 v2  |  dt={DT}  T={T_RUN:g}  seeds={SEEDS[0]}..{SEEDS[-1]}  "
          f"{'(QUICK)' if QUICK else ''}\nscript sha256 {digest}")
    print("=" * 92)

    clean, netperm, n_change, n_unres = simulate(SEEDS, T_RUN, DT)
    print("\nCrossing extraction per seed:")
    for i, s in enumerate(SEEDS):
        print(f"  seed {s:2d}: order changes={n_change[i]:6d}  clean adjacent={len(clean[i]):6d}  "
              f"unresolved={n_unres[i]:3d} ({n_unres[i]/max(n_change[i],1):.2%})  "
              f"[R8 uses all {n_change[i]}; R2 drops the unresolved]")

    # generator / pair frequencies with CI across seeds
    gens = [(0, 1), (0, -1), (1, 1), (1, -1)]
    fg = np.array([[np.mean([(e[0], e[1]) == g for e in clean[i]]) for g in gens] for i in range(len(SEEDS))])
    pairs = [(0, 1), (0, 2), (1, 2)]
    fp = np.array([[np.mean([tuple(sorted((e[2], e[3]))) == pq for e in clean[i]]) for pq in pairs]
                   for i in range(len(SEEDS))])
    print("\nGenerator frequencies (mean, 95% CI across seeds):")
    for j, g in enumerate(gens):
        m, lo, hi = mean_ci(fg[:, j]); print(f"  sigma_{g[0]+1}^{g[1]:+d}: {m:.4f} [{lo:.4f}, {hi:.4f}]")
    print("Particle-pair frequencies:")
    for j, pq in enumerate(pairs):
        m, lo, hi = mean_ci(fp[:, j]); print(f"  pair {pq}: {m:.4f} [{lo:.4f}, {hi:.4f}]")

    # ----------------------------------------------------------------- M1 / M3: R8 drift
    Xt = np.random.default_rng(7).dirichlet(np.ones(8), N_TEST)          # test points (x)
    Psi = np.sqrt(Xt)                                                   # nonnegative amplitudes
    Pbar = [sum(R8_from_orders(b, a) for b, a in netperm[i]) / len(netperm[i]) for i in range(len(SEEDS))]
    D = np.array([Xt @ (P - np.eye(8)).T for P in Pbar])               # (seeds, tests, 8)
    m, lo, hi = mean_ci(D, axis=0)
    excl = (lo > 0) | (hi < 0)
    frac_pts = np.mean(excl.any(1))
    print("\nM1  R8 signed one-crossing drift E[dx|x] at 200 test points:")
    print(f"  max |mean drift| = {np.abs(m).max():.4f}; test points with a component whose CI excludes 0: "
          f"{frac_pts:.1%}")
    worst = np.argmax(np.abs(m).max(1)); kk = np.argmax(np.abs(m[worst]))
    print(f"  example x = {np.round(Xt[worst], 3)}\n    component {kk}: drift {m[worst, kk]:+.4f} "
          f"CI [{lo[worst, kk]:+.4f}, {hi[worst, kk]:+.4f}]")

    Pi = isotypic_projectors()
    W = np.array([[sum(Xt[t, i] for i in range(8) if bin(i).count("1") == w) for w in range(4)]
                  for t in range(N_TEST)])
    ctrl_sector, ctrl_weight = 0.0, 0.0
    for i in range(len(SEEDS)):
        for b, a in netperm[i][:2000]:
            M = R8_from_orders(b, a)
            for name, P in Pi.items():
                before = np.einsum("ti,ij,tj->t", Psi, P, Psi)
                after = np.einsum("ti,ij,tj->t", Psi @ M.T, P, Psi @ M.T)
                ctrl_sector = max(ctrl_sector, np.abs(after - before).max())
            W2 = np.array([[sum((Xt[t] @ M.T)[j] for j in range(8) if bin(j).count("1") == w)
                            for w in range(4)] for t in range(0, N_TEST, 20)])
            ctrl_weight = max(ctrl_weight, np.abs(W2 - W[::20]).max())
    ranks = {n: int(round(np.trace(P))) for n, P in Pi.items()}
    ctrl_ok = ctrl_sector < 1e-12 and ctrl_weight < 1e-12
    print(f"\nM3  POSITIVE CONTROL (must be 0): sector populations ||P psi||^2 {ranks} -> "
          f"max change {ctrl_sector:.1e}; helicity-weight classes -> max change {ctrl_weight:.1e}  "
          f"[{'PASS' if ctrl_ok else 'FAIL: RUN INVALID'}]")

    # ----------------------------------------------------------------- M2: finer filtration
    print("\nM2  R8 drift conditioned on the ordering just before the crossing (clean events):")
    orders = sorted({e[4] for i in range(len(SEEDS)) for e in clean[i]})
    rows = []
    for o in orders:
        per_seed = []
        for i in range(len(SEEDS)):
            ev = [e for e in clean[i] if e[4] == o]
            if len(ev) < 20:
                continue
            P = sum(R8_pair(e[2], e[3]) for e in ev) / len(ev)
            per_seed.append(np.abs(Xt @ (P - np.eye(8)).T).max(1).mean())
        if per_seed:
            rows.append((o, np.mean(per_seed)))
    uncond = np.mean([np.abs(Xt @ (P - np.eye(8)).T).max(1).mean() for P in Pbar])
    for o, v in rows:
        print(f"  ordering {o}: mean max|drift| = {v:.4f}")
    print(f"  unconditional         : mean max|drift| = {uncond:.4f}  "
          f"(finer conditioning {'increases' if min(v for _, v in rows) > uncond else 'does not increase'} the drift)")

    # ----------------------------------------------------------------- M4: ordered invariance
    x0 = np.array([0.4, 0.3, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02]); dev = 0.0
    for i in range(len(SEEDS)):
        x = x0.copy()
        for b, a in netperm[i]:
            x = R8_from_orders(b, a) @ x
            dev = max(dev, np.abs(np.sort(x) - np.sort(x0)).max())
    print(f"\nM4  sorted(x) along every ordered trajectory: max deviation {dev:.1e} "
          f"(permutation: populations rearranged, never concentrated)")

    # ----------------------------------------------------------------- M5: R2 secondary
    bins = np.linspace(0, 1, 11)
    th0 = np.random.default_rng(8).uniform(0, np.pi, 4000)
    P0 = np.stack([np.cos(th0), np.sin(th0)], 1); x1 = P0[:, 0] ** 2
    which = np.digitize(x1, bins) - 1
    per = []
    for i in range(len(SEEDS)):
        w = {g: np.mean([(e[0], e[1]) == g for e in clean[i]]) for g in gens}
        dx = sum(w[g] * ((P0 @ R2[g].T)[:, 0] ** 2 / ((P0 @ R2[g].T) ** 2).sum(1) - x1) for g in gens)
        per.append([dx[which == b].mean() for b in range(10)])
    m2, lo2, hi2 = mean_ci(np.array(per), axis=0)
    print("\nM5  R2 (projective, secondary) signed drift of x_1 by x_1 bin:")
    for b in range(10):
        print(f"  x_1 in [{bins[b]:.1f},{bins[b+1]:.1f}): {m2[b]:+.4f} [{lo2[b]:+.4f}, {hi2[b]:+.4f}]")
    r2_reject = bool(((lo2 > 0) | (hi2 < 0)).any())

    # ----------------------------------------------------------------- M6: dt convergence
    print("\nM6  extractor completeness vs dt (seed 1, T = 2000):")
    conv = {}
    for dt in (0.01, 0.005, 0.0025):
        _, _, nc, nu = simulate([1], 2000.0, dt)
        conv[dt] = (int(nc[0]), int(nu[0]))
        print(f"  dt={dt:<7} order changes={nc[0]:5d}  unresolved={nu[0]:3d} ({nu[0]/max(nc[0],1):.2%})")

    # ----------------------------------------------------------------- decision
    r8_reject = frac_pts > 0
    print("\n" + "=" * 92)
    print("DECISION (computed):")
    if not ctrl_ok:
        print("  Positive control FAILED -> run invalid, no conclusion.")
    else:
        print(f"  R8: martingale {'REJECTED' if r8_reject else 'NOT rejected'} "
              f"({frac_pts:.0%} of test points have a drift component with CI excluding 0).")
        print(f"  R2: martingale {'REJECTED' if r2_reject else 'NOT rejected'} (secondary, projective dynamics).")
        print("  Sector populations are exactly invariant (positive control): braiding does not move")
        print("  weight between S3 sectors, so it supplies neither drift nor absorption in that basis.")
    print("=" * 92)
    out = dict(sha256=digest, dt=DT, T=T_RUN, seeds=SEEDS, n_change=n_change.tolist(),
               n_unresolved=n_unres.tolist(), r8_frac_points_rejected=float(frac_pts),
               r8_max_abs_drift=float(np.abs(m).max()), control_sector=float(ctrl_sector),
               control_weight=float(ctrl_weight), sorted_invariance_dev=float(dev),
               r2_drift_by_bin=m2.tolist(), r2_reject=r2_reject, dt_convergence={str(k): v for k, v in conv.items()})
    json.dump(out, open(here.with_name("step0_v2_results.json"), "w"), indent=1)
    print(f"Results: {here.with_name('step0_v2_results.json')}")


if __name__ == "__main__":
    main()