"""
orientation_stats.py -- Stages 1-2: substrate-only orientation statistics of Part I ring crossings.
No populations are simulated. Output is frozen to orientation_stats.json (with this script's hash)
BEFORE any prediction is computed from it.

DYNAMICS (identical to cross_test.py): ring L = 5, N = 3 defects, speed 1, Poisson velocity
reversals at rate GAMMA, Euler step dt = 0.02, crossings detected per pair (0,1), (0,2), (1,2) in
that order within a step. Pair index p in {0, 1, 2}.
ORIENTATION CONVENTION: s_n = +1 if the LOWER-index defect of the pair overtook the higher, else -1.
With d_p(x) = x_a (e_a - x) - x_b (e_b - x) for p = (a, b), a < b, the cross_test.py single jump is
exactly x -> x + EPS s_n d_{p_n}(x).

STAGE 1  pi_p = P(p_n = p) and E[s | p] for each pair, with 99% intervals (first-order drift).
STAGE 2  K_pq(l) = E[ 1{p_n = p, p_{n+l} = q} s_n s_{n+l} ] for l = 1..LMAX, estimated per run and
         averaged over runs (99% intervals across runs). Also the crossing rate per unit time.
Repeated for GAMMA in {0.5, 1, 2}. GAMMA = 1 is the value used in cross_test.py.

Usage: python orientation_stats.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np

QUICK = "--quick" in sys.argv
R = 400 if QUICK else 2000
T = 500.0 if QUICK else 2000.0
L, C, DT = 5.0, 1.0, 0.02
GAMMAS = [0.5, 1.0, 2.0]
LMAX = 200
SEED = 20261104
PAIRS = [(0, 1), (0, 2), (1, 2)]
Z99 = 2.5758


def crossing_sequences(rng, gamma):
    X = rng.uniform(0, L, (R, 3)); V = C * rng.choice([-1, 1], (R, 3))
    fl = np.stack([np.floor((X[:, i] - X[:, j]) / L) for i, j in PAIRS], 1)
    seq_p = [[] for _ in range(R)]; seq_s = [[] for _ in range(R)]
    for step in range(int(T / DT)):
        X += V * DT
        V[rng.random((R, 3)) < gamma * DT] *= -1
        nfl = np.stack([np.floor((X[:, i] - X[:, j]) / L) for i, j in PAIRS], 1)
        ch = nfl != fl
        if ch.any():
            for p in range(3):
                rows = np.flatnonzero(ch[:, p])
                for r in rows:
                    # X_i - X_j increased through a multiple of L  <=>  lower index i overtook j
                    seq_p[r].append(p); seq_s[r].append(1 if nfl[r, p] > fl[r, p] else -1)
        fl = nfl
    return [np.array(a, np.int8) for a in seq_p], [np.array(a, np.int8) for a in seq_s]


def stats(seq_p, seq_s):
    n_runs = len(seq_p)
    pi = np.zeros((n_runs, 3)); ms = np.full((n_runs, 3), np.nan)
    K = np.zeros((n_runs, LMAX + 1, 3, 3))
    for r in range(n_runs):
        p, s = seq_p[r].astype(int), seq_s[r].astype(int); n = p.size
        for q in range(3):
            sel = p == q; pi[r, q] = sel.mean()
            if sel.any():
                ms[r, q] = s[sel].mean()
        for l in range(1, LMAX + 1):
            if n <= l:
                break
            prod = s[:-l] * s[l:]
            idx = p[:-l] * 3 + p[l:]
            K[r, l] = (np.bincount(idx, weights=prod, minlength=9) / (n - l)).reshape(3, 3)
    def ci(a, axis=0):
        a = np.asarray(a, float); m = np.nanmean(a, axis); se = np.nanstd(a, axis, ddof=1) / np.sqrt(a.shape[axis])
        return m, m - Z99 * se, m + Z99 * se
    return ci(pi), ci(ms), ci(K)


def main():
    here = pathlib.Path(__file__).resolve()
    digest = hashlib.sha256(here.read_bytes()).hexdigest()
    if "--hash" in sys.argv:
        print(digest); return
    t0 = time.time()
    print("=" * 96)
    print(f"Orientation statistics (substrate only) | runs={R} T={T:g} LMAX={LMAX} {'(QUICK)' if QUICK else ''}\n"
          f"script sha256 {digest}")
    print("=" * 96)
    out = dict(sha256=digest, R=R, T=T, L=L, C=C, DT=DT, LMAX=LMAX, pairs=PAIRS,
               convention="s=+1 if lower-index defect overtook", gammas={})
    ss = np.random.SeedSequence(SEED).spawn(len(GAMMAS))
    for gi, g in enumerate(GAMMAS):
        sp, sg = crossing_sequences(np.random.default_rng(ss[gi]), g)
        n_ev = np.array([a.size for a in sp])
        (pi, pilo, pihi), (ms, mslo, mshi), (K, Klo, Khi) = stats(sp, sg)
        rate = n_ev.mean() / T
        print(f"\nGAMMA = {g}: crossings per run {n_ev.mean():.0f} (rate {rate:.3f} per unit time)")
        print("  STAGE 1  pair   pi_p                      E[s | p]")
        for q in range(3):
            print(f"           {PAIRS[q]}  {pi[q]:.4f} [{pilo[q]:.4f},{pihi[q]:.4f}]   "
                  f"{ms[q]:+.4f} [{mslo[q]:+.4f},{mshi[q]:+.4f}]")
        same = np.array([np.trace(K[l]) for l in range(LMAX + 1)])
        diff = np.array([K[l].sum() - np.trace(K[l]) for l in range(LMAX + 1)])
        print("  STAGE 2  K summed over p = q (same pair) and p != q (shared-defect pairs), by lag:")
        for l in (1, 2, 3, 5, 10, 20, 50, 100, 200):
            if l <= LMAX:
                print(f"           l={l:3d}: same {same[l]:+.5f}   different {diff[l]:+.5f}")
        for cut in (25, 50, 100, 200):
            print(f"           cumulative sum to l={cut:3d}: same {same[1:cut+1].sum():+.4f}   different {diff[1:cut+1].sum():+.4f}")
        out["gammas"][str(g)] = dict(rate=rate, crossings_per_run=float(n_ev.mean()),
                                     pi=pi.tolist(), pi_ci=[pilo.tolist(), pihi.tolist()],
                                     mean_s=ms.tolist(), mean_s_ci=[mslo.tolist(), mshi.tolist()],
                                     K=K.tolist(), K_lo=Klo.tolist(), K_hi=Khi.tolist())
    p = here.with_name("orientation_stats.json")
    json.dump(out, open(p, "w"))
    print(f"\nFrozen output: {p}\nsha256 of output {hashlib.sha256(p.read_bytes()).hexdigest()}")
    print(f"Wall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
