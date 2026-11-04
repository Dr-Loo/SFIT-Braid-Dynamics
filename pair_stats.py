"""
pair_stats.py -- Step 1 of the reduced-model test: substrate-only statistics of WHICH pair crosses.
No populations are simulated. Output frozen to pair_stats.json (with this script's hash) BEFORE
reduced_model.py computes any prediction from it.

DYNAMICS: identical to cross_test.py / orientation_stats.py (ring L = 5, 3 defects, speed 1,
reversal rate GAMMA, dt = 0.02, pairs (0,1), (0,2), (1,2) checked in that order within a step).

MEASURED, for GAMMA in {0.5, 1, 2}:
  * the topological orientation rule s_n g(p_n) = (-1)^n const, g = (+1, -1, +1):
    fraction of consecutive crossings violating it (expected: only same-step ordering ambiguity);
  * order-k transition probabilities P(p_{n+1} | p_{n-k+1..n}) for k = 1..4, and the
    stationary distribution of k-histories.
MARKOV ORDER RULE (fixed in advance): the order used by the reduced model is the smallest k in 1..4
  such that, for every (k+1)-history seen at least 2,000 times, |P(next | k+1-history) -
  P(next | its last k)| < 0.005. If no k <= 4 passes, k = 4 is used and the result is flagged.
  Counts for every k = 1..4 are frozen as well, so the reduced model can report its prediction
  at each order (convergence with k is itself a result).

Usage: python pair_stats.py [--quick] [--hash]
"""
import sys, json, hashlib, pathlib, time
import numpy as np

QUICK = "--quick" in sys.argv
_argv = sys.argv; sys.argv = ["x"]
import orientation_stats as O                     # reuse the identical crossing extractor
sys.argv = _argv
O.R = 400 if QUICK else 2000
O.T = 500.0 if QUICK else 2000.0
O.DT = 0.02
GAMMAS = [0.5, 1.0, 2.0]
KMAX = 4
MIN_COUNT = 2000
TOL = 0.005
SEED = 20261105
G = np.array([1, -1, 1])


def hist_codes(p, k):
    """Integer code of the k-history ending at each position (positions k-1 .. n-1)."""
    c = np.zeros(p.size - k + 1, np.int64)
    for j in range(k):
        c = c * 3 + p[j:p.size - k + 1 + j]
    return c


def transitions(seqs, k):
    counts = np.zeros((3 ** k, 3), np.int64)
    for p in seqs:
        p = p.astype(np.int64)
        if p.size <= k:
            continue
        h = hist_codes(p[:-1], k); nxt = p[k:]
        np.add.at(counts, (h, nxt), 1)
    return counts


def main():
    here = pathlib.Path(__file__).resolve()
    digest = hashlib.sha256(here.read_bytes()).hexdigest()
    if "--hash" in sys.argv:
        print(digest); return
    t0 = time.time()
    print("=" * 96)
    print(f"Pair statistics (substrate only) | runs={O.R} T={O.T:g} dt={O.DT} {'(QUICK)' if QUICK else ''}\n"
          f"script sha256 {digest}")
    print("=" * 96)
    out = dict(sha256=digest, R=O.R, T=O.T, DT=O.DT, g=G.tolist(), KMAX=KMAX, MIN_COUNT=MIN_COUNT, TOL=TOL, gammas={})
    ss = np.random.SeedSequence(SEED).spawn(len(GAMMAS))
    for gi, gam in enumerate(GAMMAS):
        sp, sg = O.crossing_sequences(np.random.default_rng(ss[gi]), gam)
        bad = tot = 0
        for p, s in zip(sp, sg):
            t = s.astype(int) * G[p.astype(int)]
            bad += int(np.sum(t[1:] == t[:-1])); tot += t.size - 1
        T = {k: transitions(sp, k) for k in range(1, KMAX + 2)}
        chosen, diffs = None, {}
        for k in range(1, KMAX + 1):
            hi, lo = T[k + 1], T[k]
            worst = 0.0
            for code in range(3 ** (k + 1)):
                n = hi[code].sum()
                if n < MIN_COUNT:
                    continue
                parent = code % (3 ** k)                   # drop the oldest pair
                pl = lo[parent] / lo[parent].sum(); ph = hi[code] / n
                worst = max(worst, float(np.abs(ph - pl).max()))
            diffs[k] = worst
            if chosen is None and worst < TOL:
                chosen = k
        flagged = chosen is None
        if flagged:
            chosen = KMAX
        P1 = T[1] / T[1].sum(1, keepdims=True)
        print(f"\nGAMMA = {gam}: {tot + len(sp)} crossings; topological rule violated in {bad} of {tot} "
              f"consecutive pairs ({bad / tot:.2e})")
        print("  order-1 transition matrix P(next | last):")
        for a in range(3):
            print(f"    from {O.PAIRS[a]}: " + "  ".join(f"{P1[a, b]:.4f}" for b in range(3)))
        print("  Markov check, max |P(next | k+1 history) - P(next | k history)|: " +
              ", ".join(f"k={k}: {v:.4f}" for k, v in diffs.items()))
        print(f"  order used: k = {chosen}{'  (FLAGGED: no order <= 4 passed)' if flagged else ''}")
        Tk = T[chosen]
        hist_counts = transitions(sp, chosen).sum(1)
        out["gammas"][str(gam)] = dict(rule_violation=bad / tot, order=chosen, flagged=flagged,
                                       markov_diffs=diffs, P1=P1.tolist(),
                                       trans_counts=Tk.tolist(), hist_counts=hist_counts.tolist(),
                                       trans_counts_all={str(k): T[k].tolist() for k in range(1, KMAX + 1)},
                                       crossings=int(tot + len(sp)))
    p = here.with_name("pair_stats.json")
    json.dump(out, open(p, "w"))
    print(f"\nFrozen output: {p}\nsha256 of output {hashlib.sha256(p.read_bytes()).hexdigest()}")
    print(f"Wall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
