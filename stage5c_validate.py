"""
stage5c_validate.py -- SFIT Part V Stage 5C, VALIDATION step: carrier runs on the exchange-rule substrate
(stage5c_core.carrier, new seeds) for every arm frozen in stage5c_predictions.json, scored with the criteria
stated in stage5c_predict.py.
Usage: python stage5c_validate.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np
from scipy.stats import chi2
import stage5c_core as C5

QUICK = "--quick" in sys.argv
SEED = 20261915
Z99 = 2.5758


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    here = pathlib.Path(__file__).resolve(); digest = sha(here)
    if "--hash" in sys.argv:
        print(f"stage5c_validate.py {digest}"); return
    pf = here.with_name("stage5c_predictions_quick.json" if QUICK else "stage5c_predictions.json")
    pred = json.load(open(pf)); psha = sha(pf); t0 = time.time(); bar = "=" * 116
    print(bar); print(f"Part V Stage 5C VALIDATION {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}\npredictions {pf.name} sha256 {psha}\n"
                      f"  made by stage5c_predict.py {pred['script_sha256']}"); print(bar)
    for f, h in pred["file_sha256"].items():
        if sha(here.with_name(f)) != h:
            print(f"{f} differs from the file used for the predictions: stopping."); return
    if pred["quick"] != QUICK:
        print("Quick setting differs: stopping."); return
    ss = np.random.SeedSequence(SEED).spawn(len(pred["arms"]) + 1); res = []
    for ai, a in enumerate(pred["arms"]):
        coin = np.random.default_rng(ss[-1].spawn(1)[0]) if a.get("coin") else None
        rec, ncr = C5.carrier(np.random.default_rng(ss[ai]), a["n"], a["gamma_rt"], a["eps"], a["x0"], a["rule"], a["p"], K=pred["K"],
                              coin_rng=coin, H=a["H"])
        ok = rec >= 0; P = np.bincount(rec[ok], minlength=3) / max(ok.sum(), 1); se = math.sqrt(P[0] * (1 - P[0]) / max(ok.sum(), 1))
        p = a["prediction"]; z = (P[0] - p["P0"]) / math.hypot(se, p["se0"]); zb = (P[0] - a["born_finiteH"]) / math.hypot(se, a.get("born_se", 0.0))
        zx = (P[0] - a["x0"][0]) / se
        res.append(dict(config=a["config"], x0=a["x0"], H=a["H"], eps=a["eps"], coin=bool(a.get("coin")), n=a["n"], P=P.tolist(), P0=float(P[0]),
                        se0=se, predicted=p["P0"], pred_se=p["se0"], z=z, born_finiteH=a["born_finiteH"], z_born=zb, z_vs_x0=zx,
                        absorbed=float(ok.mean()), median_crossings=float(np.median(ncr))))
        print(f"  {a['config']} x0={str(a['x0']):16s} H={a['H']:<6} eps={a['eps']:<4} n={a['n']:6d}{' coin' if a.get('coin') else ''}:"
              f" P0 {P[0]:.4f}+-{se:.4f}  predicted {p['P0']:.4f}+-{p['se0']:.4f} z={z:+.2f}  | Born finite-H {a['born_finiteH']:.4f} z={zb:+.2f}"
              f"  | vs x0 z={zx:+.2f}  med.cr {np.median(ncr):.0f}  [{time.time() - t0:.0f} s]", flush=True)
    sub = pred["substrate"]
    G_ok = all(sub[c]["consistent"] == sub[c]["runs"] for c in ("Z1", "Z2", "A1"))
    ctrl = next(r for r in res if r["coin"]); C_ok = abs(ctrl["z"]) < Z99
    model = [r for r in res if not r["coin"]]
    X2 = sum(r["z"] ** 2 for r in model); pX2 = float(chi2.sf(X2, len(model)))
    Q1 = all(abs(r["z"]) < Z99 for r in model) and pX2 > 0.01
    zero = [r for r in model if r["config"] in ("Z1", "Z2")]
    Q2 = all(abs(r["z_born"]) < Z99 for r in zero if r["config"] == "Z1")
    amp = [r for r in model if r["config"] == "A1" and r["eps"] == 0.3 and r["x0"] in ([0.5, 0.3, 0.2], [0.8, 0.2, 0.0])]
    Q3 = all(r["z_vs_x0"] > Z99 for r in amp)
    print("\n" + bar + "\nDECISION (computed, criteria as frozen in stage5c_predict.py):")
    print(f"  [{'PASS' if G_ok else 'FAIL'}] G  geometry: " + ", ".join(f"{c} {sub[c]['consistent']}/{sub[c]['runs']}" for c in ("Z1", "Z2", "A1")))
    print(f"  [{'PASS' if C_ok else 'FAIL'}] C  fair-coin control: {ctrl['P0']:.4f} vs {ctrl['predicted']:.4f} (z = {ctrl['z']:+.2f})")
    print(f"  [{'CONFIRMED' if Q1 else 'NOT CONFIRMED'}] Q1 measured-theta law: chi2 = {X2:.1f} on {len(model)}, p = {pX2:.3f}; max |z| = {max(abs(r['z']) for r in model):.2f}")
    print(f"  [{'CONFIRMED' if Q2 else 'NOT CONFIRMED'}] Q2 Z1 arms agree with finite-H Born (Z2 reported, not scored): "
          + ", ".join(f"{r['config']} {r['x0']}: z={r['z_born']:+.2f}" for r in zero))
    print(f"  [{'CONFIRMED' if Q3 else 'NOT CONFIRMED'}] Q3 amplifying regime, P0 > x0: " + ", ".join(f"{r['x0']}: {r['P0']:.4f} (z={r['z_vs_x0']:+.2f})" for r in amp))
    v1 = "ZERO-PERSISTENCE RECORD LAW CONFIRMED" if (G_ok and C_ok and Q1 and Q2) else "zero-persistence record law NOT confirmed as pre-stated"
    v2 = "AMPLIFYING REGIME CONFIRMED" if (G_ok and C_ok and Q3) else "amplifying regime NOT confirmed as pre-stated"
    print(f"  -> {v1}\n  -> {v2}")
    print(bar)
    of = here.with_name("stage5c_results_quick.json" if QUICK else "stage5c_results.json")
    json.dump(dict(script_sha256=digest, predictions_sha256=psha, arms=res, G=G_ok, C=C_ok, Q1=Q1, chi2=X2, p_chi2=pX2, Q2=Q2, Q3=Q3,
                   verdicts=[v1, v2]), open(of, "w"), indent=1, default=float)
    print(f"Results: {of}  sha256 {sha(of)}\nWall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()