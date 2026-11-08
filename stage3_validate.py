"""
stage3_validate.py -- SFIT Part V Stage 3, VALIDATION step: live carrier runs for every arm frozen in
stage3_predictions.json (new substrate seeds, 20,000 runs per arm), evaluated with the criteria stated in
stage3_predict.py. Stops if the prediction file was made with a different stage3_core.py or quick setting.

Usage: python stage3_validate.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np
import stage3_core as C3

QUICK = "--quick" in sys.argv
N_CAR = 2000 if QUICK else 20000
SEED = 20261315
Z99 = 2.5758
FLOOR = 0.03


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    here = pathlib.Path(__file__).resolve(); digest = sha(here); core_sha = sha(here.with_name("stage3_core.py"))
    if "--hash" in sys.argv:
        print(f"stage3_validate.py {digest}\nstage3_core.py     {core_sha}"); return
    pf = here.with_name("stage3_predictions_quick.json" if QUICK else "stage3_predictions.json")
    pred = json.load(open(pf)); psha = sha(pf)
    t0 = time.time(); bar = "=" * 120
    print(bar)
    print(f"Part V Stage 3 VALIDATION | carrier runs per arm = {N_CAR} {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}\n"
          f"core   sha256 {core_sha}\npredictions {pf.name} sha256 {psha}\n  made by stage3_predict.py {pred['script_sha256']}")
    print(bar)
    if pred["core_sha256"] != core_sha or pred["quick"] != QUICK:
        print("Core file or quick setting differs from the prediction run: stopping."); return
    eps = pred["eps"]; keys = list(pred["arms"])
    ss = np.random.SeedSequence(SEED).spawn(len(keys) + 1)
    res = {}
    print(f"\n  {'arm':30s} {'c=1-2q':>7} {'measured P':>26} {'P0 +- SE':>15} {'b':>8} {'reduced':>15} {'z':>6}"
          f" {'law/affine':>15} {'z':>6} {'power':>15} {'z':>6}")
    for ai, key in enumerate(keys):
        a = pred["arms"][key]; st = pred["substrate"][a["config"]]
        coin = np.random.default_rng(ss[-1].spawn(1)[0]) if a.get("coin") else None
        rec, ncr = C3.carrier(np.random.default_rng(ss[ai]), N_CAR, st["gamma"], eps, a["x0"], kappa=st["kappa"], K=st["K"],
                              coin_rng=coin)
        ok = rec >= 0; P = np.bincount(rec[ok], minlength=3) / max(ok.sum(), 1)
        se = [math.sqrt(P[c] * (1 - P[c]) / max(ok.sum(), 1)) for c in range(3)]
        r = dict(P=P.tolist(), se=se, se0=se[0], b=float(P[0] - a["x0"][0]), absorbed=float(ok.mean()),
                 median_crossings=float(np.median(ncr)), c=st["c"])
        cols = ""
        for lab in ("reduced", "law", "affine", "power"):
            if lab in a:
                z = (P[0] - a[lab]["P0"]) / math.hypot(se[0], a[lab]["se0"]); r[f"z_{lab}"] = z
        def fmt(lab):
            return f"{a[lab]['P0']:8.4f}+-{a[lab]['se0']:.4f} {r['z_' + lab]:+6.2f}" if lab in a else " " * 22
        res[key] = r
        print(f"  {key:30s} {st['c']:7.4f} {str(np.round(P, 4).tolist()):>26} {P[0]:7.4f}+-{se[0]:.4f} {r['b']:+8.4f} "
              f"{fmt('reduced')} {fmt('law') if 'law' in a else fmt('affine')} {fmt('power')}"
              f"  med.cr {np.median(ncr):.0f} abs {ok.mean():.4f} [{time.time() - t0:.0f} s]", flush=True)

    part = lambda p: [k for k in keys if pred["arms"][k]["part"] == p and not pred["arms"][k].get("coin")]
    A, Bk, Ck = part("A"), part("B"), part("C")
    ok = lambda k, lab: abs(res[k][f"z_{lab}"]) < Z99
    chi = lambda ks, lab: sum(res[k][f"z_{lab}"] ** 2 for k in ks if f"z_{lab}" in res[k])
    law_ok = all(ok(k, "law") for k in A); red_ok_A = all(ok(k, "reduced") for k in A)
    coin = res["B3_coin"]; coin_ok = abs(coin["b"] / coin["se0"]) < Z99
    above = [abs(res[k]["b"]) - Z99 * res[k]["se0"] > FLOOR for k in Bk]
    below = [abs(res[k]["b"]) + Z99 * res[k]["se0"] < FLOOR for k in Bk]
    floor = "FLOOR APPARENT" if all(above) else ("DRIVABLE TOWARD ZERO" if any(below) else "UNDETERMINED")
    law_ok_B = all(ok(k, "law") for k in Bk); red_ok_B = all(ok(k, "reduced") for k in Bk)
    sym = next(k for k in Ck if abs(pred["arms"][k]["x0"][0] - 1 / 3) < 1e-9)
    sym_ok = all(abs(res[sym]["P"][c] - 1 / 3) < Z99 * res[sym]["se"][c] for c in range(3))
    rep = next(k for k in Ck if pred["arms"][k]["x0"] == [0.5, 0.3, 0.2])
    zrep = (res[rep]["P"][0] - 0.4211) / math.hypot(res[rep]["se0"], 0.0035)
    covered = lambda lab: [k for k in Ck if lab in pred["arms"][k] and k != sym and k != rep]
    aff_ok = all(ok(k, "affine") for k in covered("affine")); pow_ok = all(ok(k, "power") for k in covered("power"))
    red_ok_C = all(ok(k, "reduced") for k in Ck)

    print("\n" + bar + "\nDECISION (computed, criteria as frozen in stage3_predict.py):")
    print(f"  3A collapse  lambda law (lambda = {pred['lam']['value']:.4f}): {'SUPPORTED' if law_ok else 'NOT SUPPORTED'}"
          f"  chi2 = {chi(A, 'law'):.1f} / {len(A)};  reduced model: {'all agree' if red_ok_A else 'NOT all agree'}  chi2 = {chi(A, 'reduced'):.1f} / {len(A)}")
    print(f"  3B floor     {floor}: |b| at gamma/K = " + ", ".join(
        f"{pred['substrate'][pred['arms'][k]['config']]['gamma_over_K']:g}: {abs(res[k]['b']):.4f}+-{res[k]['se0']:.4f}" for k in Bk))
    print(f"               lambda law on B arms: {'all agree' if law_ok_B else 'NOT all agree'} (chi2 {chi(Bk, 'law'):.1f} / {len(Bk)});"
          f" reduced model: {'all agree' if red_ok_B else 'NOT all agree'} (chi2 {chi(Bk, 'reduced'):.1f} / {len(Bk)})")
    print(f"               fair-coin control on B3: b/SE = {coin['b'] / coin['se0']:+.2f}  -> {'PASS' if coin_ok else 'FAIL'}")
    print(f"  3C map       symmetric point 1/3 in every channel: {'PASS' if sym_ok else 'FAIL'} {np.round(res[sym]['P'], 4).tolist()};"
          f"  replication of Stage 2 (0.4211): z = {zrep:+.2f}")
    print(f"               affine law (alpha = {pred['alpha']['value']:.4f}): {'SUPPORTED' if aff_ok else 'NOT SUPPORTED'} (chi2 {chi(covered('affine'), 'affine'):.1f} / {len(covered('affine'))})")
    print(f"               power law  (beta  = {pred['beta']['value']:.4f}): {'SUPPORTED' if pow_ok else 'NOT SUPPORTED'} (chi2 {chi(covered('power'), 'power'):.1f} / {len(covered('power'))})")
    print(f"               reduced model at every x0: {'all agree' if red_ok_C else 'NOT all agree'} (chi2 {chi(Ck, 'reduced'):.1f} / {len(Ck)})")
    if not (coin_ok and sym_ok):
        print("  NOTE: a control failed; the carrier code is suspect and no other result should be used.")
    print(bar)
    of = here.with_name("stage3_results_quick.json" if QUICK else "stage3_results.json")
    json.dump(dict(script_sha256=digest, core_sha256=core_sha, predictions_sha256=psha, N_CAR=N_CAR, arms=res,
                   law_supported=law_ok, reduced_A=red_ok_A, floor=floor, coin_ok=coin_ok, sym_ok=sym_ok, z_replication=zrep,
                   affine_supported=aff_ok, power_supported=pow_ok, reduced_C=red_ok_C), open(of, "w"), indent=1, default=float)
    print(f"Results: {of}  sha256 {sha(of)}\nWall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()