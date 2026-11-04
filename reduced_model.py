"""
reduced_model.py -- Step 2: predict the crossing-driven record frequencies with NO free parameters.

INPUT (frozen before this script was written to run): pair_stats.json, whose sha256 is printed and
checked against EXPECTED_INPUT_SHA below.

REDUCED MODEL
  * pair sequence p_n: order-k Markov chain with the frozen transition counts (k = 1..4);
    the first k pairs are drawn from the frozen stationary distribution of k-histories;
    a history never seen in the substrate data falls back to the next lower order.
  * orientation: the topological rule, exactly: s_n = sigma0 (-1)^n g(p_n), g = (+1, -1, +1),
    sigma0 = +-1 with probability 1/2. (The real extraction violates this in ~0.4% of consecutive
    crossings because of same-step ordering; the reduced model ignores that.)
  * populations: the cross_test.py single carrier-mediated jump x -> x + EPS s d_p(x),
    d_p = x_a (e_a - x) - x_b (e_b - x), p = (a, b), a < b; record = first x_k >= 0.999;
    x(0) = (0.5, 0.3, 0.2).
  * assumption stated: the real runs start from random positions, so their first few crossings are
    not stationary; the reduced model starts stationary.
NO quantity in this script is adjusted to the measured record frequencies.

PREDICTIONS PRODUCED
  GAMMA = 1:   k = 1, 2, 3, 4 at EPS = 0.30, 0.15, 0.10, 0.05   (40,000 runs each)
  GAMMA = 0.5, 2:   k = 4 at EPS = 0.30, 0.10                    (40,000 runs each)
  Primary prediction = order chosen by pair_stats.py (k = 4, flagged: no order <= 4 passed its
  Markov check). Predictions are written to reduced_model_predictions.json BEFORE the comparison.

COMPARISON (criterion fixed in advance)
  For GAMMA = 1 the measured values from cross_test.py (40,000 runs, independent substrate seeds)
  are read only after the predictions are written. Agreement: the measured P(channel 0) lies within
  the 99% interval of the prediction combined with the measurement (difference / combined SE < 2.576).
  GAMMA = 0.5 and 2 are compared later, by validate_gamma.py.

Usage: python reduced_model.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np

QUICK = "--quick" in sys.argv
RUNS = 4000 if QUICK else 40000
X0 = np.array([0.5, 0.3, 0.2]); H = 0.999
G = np.array([1, -1, 1]); PAIRS = [(0, 1), (0, 2), (1, 2)]
MAX_EVENTS = 400000
EXPECTED_INPUT_SHA = "b0ea955b21fee83dc6354ea44f4f989a2b8fb251d05f6e18522eb1cf1fa202c8"
SEED = 20261106
Z99 = 2.5758


def d_pair(x, p):
    r = np.arange(len(p)); a = np.array([PAIRS[q][0] for q in range(3)])[p]; b = np.array([PAIRS[q][1] for q in range(3)])[p]
    xa, xb = x[r, a], x[r, b]
    d = -(xa - xb)[:, None] * x
    d[r, a] += xa; d[r, b] -= xb
    return d


class Chain:
    def __init__(self, stats, k):
        self.k = k
        self.T = {j: np.array(stats["trans_counts_all"][str(j)], float) for j in range(1, k + 1)}
        self.P = {}
        for j, T in self.T.items():
            n = T.sum(1, keepdims=True)
            self.P[j] = np.divide(T, n, out=np.full_like(T, np.nan), where=n > 0)
        hc = self.T[k].sum(1); self.h0 = hc / hc.sum()

    def start(self, rng, R):
        codes = rng.choice(3 ** self.k, R, p=self.h0)
        hist = np.zeros((R, self.k), int)
        c = codes.copy()
        for j in range(self.k - 1, -1, -1):
            hist[:, j] = c % 3; c //= 3
        return hist                                    # oldest .. newest

    def step(self, rng, hist):
        R = hist.shape[0]; out = np.empty(R, int); u = rng.random(R)
        todo = np.ones(R, bool)
        for j in range(self.k, 0, -1):                 # fall back to lower order if unseen
            code = np.zeros(R, int)
            for col in range(self.k - j, self.k):
                code = code * 3 + hist[:, col]
            P = self.P[j][code]
            ok = todo & np.isfinite(P[:, 0])
            if ok.any():
                cp = np.cumsum(P[ok], 1)
                out[ok] = (u[ok][:, None] > cp).sum(1).clip(0, 2)
                todo &= ~ok
            if not todo.any():
                break
        hist = np.roll(hist, -1, 1); hist[:, -1] = out
        return out, hist


def simulate(stats, k, eps, rng):
    ch = Chain(stats, k)
    hist = ch.start(rng, RUNS)
    sigma = rng.choice([-1, 1], RUNS)
    x = np.tile(X0, (RUNS, 1)); done = np.zeros(RUNS, bool); rec = np.full(RUNS, -1)
    n = 0
    while n < MAX_EVENTS and not done.all():
        p, hist = ch.step(rng, hist)
        s = sigma * (1 if n % 2 == 0 else -1) * G[p]
        live = ~done
        x[live] = x[live] + (s[live] * eps)[:, None] * d_pair(x[live], p[live])
        x[live] = np.clip(x[live], 0, None); x[live] /= x[live].sum(1, keepdims=True)
        hit = live & (x.max(1) >= H)
        rec[hit] = x[hit].argmax(1); done |= hit
        n += 1
    ok = rec >= 0; P = np.bincount(rec[ok], minlength=3) / max(ok.sum(), 1)
    se0 = math.sqrt(P[0] * (1 - P[0]) / max(ok.sum(), 1))
    return dict(P=P.tolist(), se0=se0, absorbed=float(done.mean()), events=n)


def main():
    here = pathlib.Path(__file__).resolve()
    digest = hashlib.sha256(here.read_bytes()).hexdigest()
    if "--hash" in sys.argv:
        print(digest); return
    t0 = time.time()
    inp = here.with_name("pair_stats.json")
    in_sha = hashlib.sha256(inp.read_bytes()).hexdigest()
    print("=" * 96)
    print(f"Reduced model | runs={RUNS} {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}\n"
          f"input pair_stats.json sha256 {in_sha}  ({'matches frozen' if in_sha == EXPECTED_INPUT_SHA else 'DOES NOT MATCH frozen'})")
    print("=" * 96)
    if in_sha != EXPECTED_INPUT_SHA:
        print("Input changed since it was frozen: stopping."); return
    stats = json.load(open(inp))["gammas"]
    jobs = [("1.0", k, e) for k in (1, 2, 3, 4) for e in (0.30, 0.15, 0.10, 0.05)]
    jobs += [(g, 4, e) for g in ("0.5", "2.0") for e in (0.30, 0.10)]
    ss = np.random.SeedSequence(SEED).spawn(len(jobs))
    preds = {}
    print("PREDICTIONS (written before any comparison)")
    for ji, (g, k, e) in enumerate(jobs):
        r = simulate(stats[g], k, e, np.random.default_rng(ss[ji]))
        key = f"gamma{g}_k{k}_eps{e:.2f}"; preds[key] = r
        print(f"  gamma={g} k={k} eps={e:.2f}: P={np.round(r['P'], 4).tolist()}  "
              f"P(ch0)-0.5 = {r['P'][0]-0.5:+.4f} +- {r['se0']:.4f}  absorbed={r['absorbed']:.4f}", flush=True)
    pp = here.with_name("reduced_model_predictions.json")
    json.dump(dict(script_sha256=digest, input_sha256=in_sha, runs=RUNS, predictions=preds), open(pp, "w"), indent=1)
    print(f"\nPredictions frozen: {pp}\nsha256 {hashlib.sha256(pp.read_bytes()).hexdigest()}")

    print("\nCOMPARISON with cross_test.py (gamma = 1, measured, 40,000 runs):")
    meas = json.load(open(here.with_name("cross_test_results.json")))["arms"]
    for e, name in ((0.30, "eps0.30"), (0.15, "eps0.15"), (0.10, "eps0.10")):
        m = meas[name]["P"][0]; mse = math.sqrt(m * (1 - m) / 40000)
        for k in (1, 2, 3, 4):
            r = preds[f"gamma1.0_k{k}_eps{e:.2f}"]; z = (r["P"][0] - m) / math.hypot(r["se0"], mse)
            print(f"  eps={e:.2f} k={k}: predicted {r['P'][0]:.4f}  measured {m:.4f}  z={z:+.2f}  "
                  f"{'AGREE' if abs(z) < Z99 else 'DISAGREE'}")
    print(f"\nWall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
