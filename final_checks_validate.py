"""
final_checks_validate.py -- run the REAL carrier process for D1, D2 (dt = 0.01) and E1 (gamma = 4,
dt = 0.01) and compare with final_checks_predictions.json.

Process: cross_test.py's single-jump arm, EPS = 0.3, x(0) = (0.5, 0.3, 0.2), 40,000 runs, new seeds, with
GAMMA and DT set per configuration. Criterion: |z| < 2.576 on P(0), with z computed from the TOTAL
prediction SE (statistical + truncation) and the measurement SE; all three components are also reported.
For D1, D2 the dt-shift of the measurement is reported against the dt = 0.02 measurements
(gamma = 1: 0.5548; gamma = 2: 0.6372 pooled from 0.6334 and 0.6410, 40,000 runs each).

Usage: python final_checks_validate.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np

QUICK = "--quick" in sys.argv
_argv = sys.argv; sys.argv = ["x"]
import cross_test as C
sys.argv = _argv
RUNS = 4000 if QUICK else 40000
SEED = 20261113
Z99 = 2.5758
OLD = {"D1": (0.5548, 40000), "D2": (0.6372, 80000)}


def main():
    here = pathlib.Path(__file__).resolve()
    digest = hashlib.sha256(here.read_bytes()).hexdigest()
    if "--hash" in sys.argv:
        print(digest); return
    t0 = time.time()
    pf = here.with_name("final_checks_predictions.json")
    pred = json.load(open(pf))
    print("=" * 100)
    print(f"Final 1D checks: validation | runs={RUNS} {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}\n"
          f"predictions file sha256 {hashlib.sha256(pf.read_bytes()).hexdigest()}")
    print("=" * 100)
    C.ARMS = [("eps0.30", 0.30, False, False)]; C.X0 = np.array(pred["x0"])
    out = {}; ss = np.random.SeedSequence(SEED).spawn(len(pred["configs"]))
    for ci, (name, pr) in enumerate(pred["configs"].items()):
        C.GAMMA = pr["gamma"]; C.DT = pr["dt"]
        seeds = ss[ci].spawn(RUNS // C.BATCH); recs = []
        for b in range(RUNS // C.BATCH):
            rec, _, _, _, _ = C.batch(np.random.default_rng(seeds[b]), C.BATCH); recs.append(rec[0])
            print(f"  {name} batch {b + 1}/{RUNS // C.BATCH} ({time.time() - t0:.0f} s)", flush=True)
        rec = np.concatenate(recs); ok = rec >= 0
        P = np.bincount(rec[ok], minlength=3) / ok.sum(); mse = np.sqrt(P * (1 - P) / ok.sum())
        pp = np.array(pr["primary"]["P"]); pse = np.array(pr["primary"]["se"])
        z0 = (pp[0] - P[0]) / math.hypot(pr["primary"]["se_total0"], mse[0])
        zc = (pp - P) / np.hypot(pse, mse)
        r = dict(gamma=pr["gamma"], dt=pr["dt"], k=pr["kstar"], converged=pr["converged"], measured=P.tolist(),
                 predicted=pp.tolist(), z0_total=z0, z_components_stat=zc.tolist(), agree=bool(abs(z0) < Z99),
                 unresolved=float(1 - ok.mean()))
        print(f"\n{name}: gamma={pr['gamma']}, dt={pr['dt']}, k*={pr['kstar']}{'' if pr['converged'] else ' (NOT converged, flagged)'}")
        print(f"  measured  {np.round(P, 4).tolist()} (+- {np.round(mse, 4).tolist()})")
        print(f"  predicted {np.round(pp, 4).tolist()} (+- stat {pr['primary']['se'][0]:.4f}, total {pr['primary']['se_total0']:.4f} on P0)")
        print(f"  z(P0, total SE) = {z0:+.2f} -> {'AGREE' if abs(z0) < Z99 else 'DISAGREE'};  z per component (stat SE) {np.round(zc, 2).tolist()}")
        if name in OLD:
            m, n = OLD[name]; d = P[0] - m; sd = math.hypot(mse[0], math.sqrt(m * (1 - m) / n))
            r["dt_shift"] = (d, sd)
            print(f"  dt-shift of the measurement (0.01 - 0.02): {d:+.4f} +- {sd:.4f}")
        out[name] = r
    json.dump(dict(sha256=digest, runs=RUNS, results=out), open(here.with_name("final_checks_results.json"), "w"), indent=1)
    print(f"\nWall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()