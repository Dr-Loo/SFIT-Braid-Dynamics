"""
stage4b_validate.py -- SFIT Part V Stage 4b, VALIDATION step: carrier runs (new seeds) for every arm frozen in
stage4b_predictions.json, evaluated with the criteria stated in stage4b_predict.py. The carrier is
stage3_core.carrier, unchanged; the threshold is set per arm through stage3_core.H.
Usage: python stage4b_validate.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np
import stage3_core as C3

QUICK = "--quick" in sys.argv
SEED = 20261615
Z99 = 2.5758


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    here = pathlib.Path(__file__).resolve(); digest = sha(here); core_sha = sha(here.with_name("stage3_core.py"))
    if "--hash" in sys.argv:
        print(f"stage4b_validate.py {digest}\nstage3_core.py      {core_sha}"); return
    pf = here.with_name("stage4b_predictions_quick.json" if QUICK else "stage4b_predictions.json")
    pred = json.load(open(pf)); psha = sha(pf); t0 = time.time(); bar = "=" * 116
    print(bar); print(f"Part V Stage 4b VALIDATION {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}\ncore   sha256 {core_sha}\n"
                      f"predictions {pf.name} sha256 {psha}\n  made by stage4b_predict.py {pred['script_sha256']}"); print(bar)
    if pred["core_sha256"] != core_sha or pred["quick"] != QUICK:
        print("Core file or quick setting differs from the prediction run: stopping."); return
    arms = pred["arms"]; ss = np.random.SeedSequence(SEED).spawn(len(arms) + 1); res = []
    for ai, a in enumerate(arms):
        C3.H = a["H"]
        coin = np.random.default_rng(ss[-1].spawn(1)[0]) if a.get("coin") else None
        rec, ncr = C3.carrier(np.random.default_rng(ss[ai]), a["n"], a["gamma_rt"], a["eps"], a["x0"], kappa=a["kappa"], K=pred["K"], coin_rng=coin)
        ok = rec >= 0; P = np.bincount(rec[ok], minlength=3) / max(ok.sum(), 1); se = math.sqrt(P[0] * (1 - P[0]) / max(ok.sum(), 1))
        p = a["prediction"]; off = P[0] - p["P0"]; ose = math.hypot(se, p["se0"]); z = off / ose
        res.append(dict(diag=a["diag"], config=a["config"], x0=a["x0"], H=a["H"], eps=a["eps"], n=a["n"], P=P.tolist(), P0=float(P[0]),
                        se0=se, predicted=p["P0"], pred_se=p["se0"], offset=off, offset_se=ose, z=z, absorbed=float(ok.mean()),
                        median_crossings=float(np.median(ncr))))
        print(f"  {a['diag']:2s} {a['config']} x0={str(a['x0']):16s} H={a['H']:<6} eps={a['eps']:<5} n={a['n']:6d}  P0 {P[0]:.4f}+-{se:.4f}"
              f"  predicted {p['P0']:.4f}+-{p['se0']:.4f}  offset {off:+.4f}+-{ose:.4f}  z={z:+.2f}  med.cr {np.median(ncr):.0f}"
              f"  abs {ok.mean():.4f}  [{time.time() - t0:.0f} s]", flush=True)
    C3.H = 0.999
    get = lambda d, c, e: next(r for r in res if r["diag"] == d and r["config"] == c and r["eps"] == e)

    ctrl = next(r for r in res if r["diag"] == "C"); C_ok = abs(ctrl["z"]) < Z99
    s3, s1, s05 = get("D1", "S2", 0.3), get("D1", "S2", 0.1), get("D1", "S2", 0.05)
    b3, b1 = get("D1", "B3", 0.3), get("D1", "B3", 0.1)
    a_ = s3["offset"] < 0 and b3["offset"] < 0
    b_ = abs(s05["z"]) < Z99 and abs(b1["z"]) < Z99
    c_ = abs(s05["offset"]) < abs(s3["offset"])
    D1 = "EXPLAINED" if (a_ and b_ and c_) else ("NOT EXPLAINED" if (abs(s05["z"]) >= Z99 or abs(b1["z"]) >= Z99) else "OPEN")

    d2 = [r for r in res if r["diag"] == "D2" and r["eps"] == 0.3]
    w = np.array([1 / r["offset_se"] ** 2 for r in d2]); o = np.array([r["offset"] for r in d2])
    dbar = float((w * o).sum() / w.sum()); dse = float(1 / math.sqrt(w.sum())); lo, hi = dbar - Z99 * dse, dbar + Z99 * dse
    chi_het = float((w * (o - dbar) ** 2).sum())
    if lo <= 0 <= hi and Z99 * dse <= 0.003:
        D2 = "NO RESIDUAL"
    elif lo > 0 or hi < 0:
        D2 = "RESIDUAL"
    else:
        D2 = "UNRESOLVED"
    epsdep = []
    for c in ("S2", "N2"):
        r3, r1 = get("D2", c, 0.3), get("D2", c, 0.1)
        dd = r1["P0"] - r3["P0"]; dse_ = math.hypot(r1["se0"], r3["se0"]); epsdep.append((c, r3["offset"], r1["offset"], dd, dse_, abs(dd) < Z99 * dse_))
    print("\n" + bar + "\nDECISION (computed, criteria as frozen in stage4b_predict.py):")
    print(f"  [{'PASS' if C_ok else 'FAIL'}] C  fair-coin control: {ctrl['P0']:.4f} vs {ctrl['predicted']:.4f} (z = {ctrl['z']:+.2f})")
    print(f"  D1 two-channel, H = 0.99: S2 offsets eps 0.3/0.1/0.05 = {s3['offset']:+.4f}, {s1['offset']:+.4f}, {s05['offset']:+.4f}"
          f" (z {s3['z']:+.2f}, {s1['z']:+.2f}, {s05['z']:+.2f});  B3 eps 0.3/0.1 = {b3['offset']:+.4f}, {b1['offset']:+.4f} (z {b3['z']:+.2f}, {b1['z']:+.2f})")
    print(f"     (a) both eps 0.3 offsets negative: {a_}   (b) smallest eps agrees with the law: {b_}   (c) S2 offset shrinks: {c_}   -> {D1}")
    print("  D2 three-channel offsets at eps 0.3: " + ", ".join(f"{r['config']} {r['offset']:+.4f}+-{r['offset_se']:.4f}" for r in d2))
    print(f"     pooled delta = {dbar:+.4f} +- {dse:.4f}, 99% [{lo:+.4f}, {hi:+.4f}]; heterogeneity chi2 = {chi_het:.1f} on {len(d2) - 1}  -> {D2}")
    for c, o3, o1, dd, dse_, same in epsdep:
        print(f"     {c}: offset eps 0.3 {o3:+.4f}, eps 0.1 {o1:+.4f}; P(0.1) - P(0.3) = {dd:+.4f}+-{dse_:.4f}  -> {'EPS-independent' if same else 'EPS-dependent'}")
    print(bar)
    of = here.with_name("stage4b_results_quick.json" if QUICK else "stage4b_results.json")
    json.dump(dict(script_sha256=digest, core_sha256=core_sha, predictions_sha256=psha, arms=res, C=C_ok, D1=D1, D2=D2, delta=dbar, delta_se=dse,
                   delta_99=[lo, hi], heterogeneity_chi2=chi_het, eps_dependence=epsdep), open(of, "w"), indent=1, default=float)
    print(f"Results: {of}  sha256 {sha(of)}\nWall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()