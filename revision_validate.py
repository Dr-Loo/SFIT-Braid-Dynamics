"""
revision_validate.py -- (1) the REAL carrier process at gamma = 0.5 (dt = 0.02, EPS = 0.3, 40,000 runs, new
seeds) compared with the frozen R1 prediction in revision_predictions.json; (2) the k = 0 ablation compared
with every gamma measured so far; (3) the distribution of k* over bootstrap replicates, for gamma = 0.5
(from revision_predictions.json) and gamma = 1, 2 (from history_order_predictions.json, if present).

k* per replicate: the smallest k such that for every j > k, |P_j^(b) - P_k^(b)| < 2.576 * SE(P_j - P_k),
the SE taken from the spread over all replicates (the same rule as for the point prediction).
Criterion for (1), fixed in advance: |z| < 2.576 on P(0) with the total prediction SE.

Usage: python revision_validate.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np

QUICK = "--quick" in sys.argv
_argv = sys.argv; sys.argv = ["x"]
import cross_test as C
sys.argv = _argv
RUNS = 4000 if QUICK else 40000
SEED = 20261115
Z99 = 2.5758
MEASURED = {"0.5 (earlier run, 0.4852)": 0.4852, "1.0": 0.5548, "1.5": 0.5993, "2.0": 0.6372, "4.0 (dt 0.01)": 0.7214}


def kstar_dist(boot):
    boot = np.asarray(boot)[:, :, 0] if np.asarray(boot).ndim == 3 else np.asarray(boot)
    B, K1 = boot.shape; kmax = K1 - 1
    se = {(k, j): float((boot[:, j] - boot[:, k]).std(ddof=1)) for k in range(1, kmax + 1) for j in range(k + 1, kmax + 1)}
    ks = []
    for b in range(B):
        kb = kmax
        for k in range(1, kmax + 1):
            if all(abs(boot[b, j] - boot[b, k]) < Z99 * se[(k, j)] for j in range(k + 1, kmax + 1)):
                kb = k; break
        ks.append(kb)
    return np.bincount(ks, minlength=kmax + 1)[1:]


def main():
    here = pathlib.Path(__file__).resolve()
    digest = hashlib.sha256(here.read_bytes()).hexdigest()
    if "--hash" in sys.argv:
        print(digest); return
    t0 = time.time()
    pf = here.with_name("revision_predictions.json"); pred = json.load(open(pf))
    print("=" * 100)
    print(f"Revision validation | runs={RUNS} {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}\n"
          f"predictions file sha256 {hashlib.sha256(pf.read_bytes()).hexdigest()}")
    print("=" * 100)
    C.ARMS = [("eps0.30", 0.30, False, False)]; C.X0 = np.array(pred["x0"]); C.GAMMA = 0.5; C.DT = 0.02
    seeds = np.random.SeedSequence(SEED).spawn(RUNS // C.BATCH); recs = []
    for b in range(RUNS // C.BATCH):
        rec, _, _, _, _ = C.batch(np.random.default_rng(seeds[b]), C.BATCH); recs.append(rec[0])
        print(f"  gamma=0.5 batch {b + 1}/{RUNS // C.BATCH} ({time.time() - t0:.0f} s)", flush=True)
    rec = np.concatenate(recs); ok = rec >= 0
    P = np.bincount(rec[ok], minlength=3) / ok.sum(); mse = np.sqrt(P * (1 - P) / ok.sum())
    R1 = pred["R1"]; pp = np.array(R1["primary"]["P"])
    z0 = (pp[0] - P[0]) / math.hypot(R1["primary"]["se_total0"], mse[0])
    zc = (pp - P) / np.hypot(np.array(R1["primary"]["se"]), mse)
    print(f"\n(1) gamma=0.5, current pipeline, k*={R1['kstar']}{'' if R1['converged'] else ' (NOT converged)'}")
    print(f"    measured  {np.round(P, 4).tolist()} (+- {np.round(mse, 4).tolist()})")
    print(f"    predicted {np.round(pp, 4).tolist()} (total SE on P0 {R1['primary']['se_total0']:.4f})")
    print(f"    z(P0) = {z0:+.2f} -> {'AGREE' if abs(z0) < Z99 else 'DISAGREE'};  z per component {np.round(zc, 2).tolist()}")
    print(f"    bias P0 - 0.5: measured {P[0] - 0.5:+.4f}, predicted {pp[0] - 0.5:+.4f}  "
          f"(sign {'reversed' if P[0] < 0.5 else 'NOT reversed'} in the measurement)")
    meas = dict(MEASURED); meas["0.5 (this run)"] = float(P[0])

    print("\n(2) k = 0 ablation (no pair memory) against measured P(0):")
    k0 = pred["R2"]
    for key, m in meas.items():
        g = key.split()[0]
        if g in k0:
            print(f"    gamma={key}: measured {m:.4f}   k = 0 prediction {k0[g]['P'][0]:.4f}   difference {k0[g]['P'][0] - m:+.4f}")
    print(f"    uniform-pi k = 0 (any gamma): {k0['uniform']['P'][0]:.4f} +- {k0['uniform']['se0']:.4f}")
    spread = max(meas.values()) - min(meas.values())
    print(f"    measured P(0) spans {spread:.3f} across gamma; the k = 0 model's gamma dependence comes only through pi.")

    print("\n(3) distribution of k* over bootstrap replicates (counts for k = 1..8):")
    print(f"    gamma=0.5: {kstar_dist(R1['boot']).tolist()}   (point k* = {R1['kstar']})")
    hf = here.with_name("history_order_predictions.json")
    if hf.exists():
        hp = json.load(open(hf))["gammas"]
        for g, r in hp.items():
            print(f"    gamma={g}: {kstar_dist(r['boot']).tolist()}   (point k* = {r['kstar']})")
    else:
        print("    history_order_predictions.json not found: gamma = 1, 2 skipped")
    json.dump(dict(sha256=digest, measured_gamma05=P.tolist(), z0=z0, z_components=zc.tolist(), agree=bool(abs(z0) < Z99)),
              open(here.with_name("revision_results.json"), "w"), indent=1)
    print(f"\nWall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()