"""
validate_gamma.py -- Step 3: run the REAL crossing-driven carrier process at GAMMA = 0.5 and 2 and
compare with the reduced-model predictions frozen in reduced_model_predictions.json.

Process: exactly cross_test.py's single-jump arms (ring L = 5, 3 defects, speed 1, dt = 0.02,
single carrier-mediated jump per crossing, first passage x_k >= 0.999, x(0) = (0.5, 0.3, 0.2)),
with the reversal rate set to GAMMA, at EPS = 0.30 and 0.10, 40,000 runs each, new seeds.
Criterion (fixed in advance): agreement if |predicted - measured| / combined SE < 2.576 for P(ch0),
for each (GAMMA, EPS), using the primary prediction (order k = 4).

Usage: python validate_gamma.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np

QUICK = "--quick" in sys.argv
_argv = sys.argv; sys.argv = ["x"]
import cross_test as C
sys.argv = _argv
RUNS = 4000 if QUICK else 40000
SEED = 20261107
Z99 = 2.5758


def main():
    here = pathlib.Path(__file__).resolve()
    digest = hashlib.sha256(here.read_bytes()).hexdigest()
    if "--hash" in sys.argv:
        print(digest); return
    t0 = time.time()
    pf = here.with_name("reduced_model_predictions.json")
    pred = json.load(open(pf))["predictions"]
    print("=" * 96)
    print(f"Validation at GAMMA = 0.5, 2 | runs={RUNS} {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}\n"
          f"predictions file sha256 {hashlib.sha256(pf.read_bytes()).hexdigest()}")
    print("=" * 96)
    C.ARMS = [("eps0.30", 0.30, False, False), ("eps0.10", 0.10, False, False)]
    out = {}
    ss = np.random.SeedSequence(SEED).spawn(2 * (RUNS // C.BATCH))
    si = 0
    for g in (0.5, 2.0):
        C.GAMMA = g
        recs = [[], []]
        for b in range(RUNS // C.BATCH):
            rec, done, _, _, _ = C.batch(np.random.default_rng(ss[si]), C.BATCH); si += 1
            recs[0].append(rec[0]); recs[1].append(rec[1])
            print(f"  gamma={g} batch {b+1}/{RUNS // C.BATCH} ({time.time()-t0:.0f} s)", flush=True)
        for ai, e in enumerate((0.30, 0.10)):
            rec = np.concatenate(recs[ai]); ok = rec >= 0
            P = np.bincount(rec[ok], minlength=3) / ok.sum(); m = P[0]
            mse = math.sqrt(m * (1 - m) / ok.sum())
            pr = pred[f"gamma{g}_k4_eps{e:.2f}"]; z = (pr["P"][0] - m) / math.hypot(pr["se0"], mse)
            key = f"gamma{g}_eps{e:.2f}"
            out[key] = dict(measured=P.tolist(), predicted=pr["P"], z=z, agree=bool(abs(z) < Z99),
                            unresolved=float(1 - ok.mean()))
            print(f"  gamma={g} eps={e:.2f}: measured {np.round(P, 4).tolist()}  predicted "
                  f"{np.round(pr['P'], 4).tolist()}  z={z:+.2f}  {'AGREE' if abs(z) < Z99 else 'DISAGREE'}")
    json.dump(dict(sha256=digest, runs=RUNS, results=out), open(here.with_name("validate_gamma_results.json"), "w"), indent=1)
    print(f"Wall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
