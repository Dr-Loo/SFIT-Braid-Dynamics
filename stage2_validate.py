"""
stage2_validate.py -- SFIT Part V Stage 2, VALIDATION step: the live carrier on the planar substrate.
Reads the frozen stage2_predictions.json (prints its sha256; stops if it was made with a different
stage2_core.py or in quick mode), runs the carrier with NEW substrate seeds, and evaluates Q1-Q5 exactly
as stated in stage2_predict.py.

ARMS: S1-S6 at EPS = 0.3; S2 at EPS = 0.15 and 0.1; fair-coin control on S2 at EPS = 0.3.
      20,000 carrier runs each (SE of P(channel 0) ~ 0.0035). Substrate starts from random positions.
z (Q1) = (measured - predicted) / sqrt(SE_meas^2 + SE_pred_total^2);  bias b = measured P(channel 0) - 0.5.

Usage: python stage2_validate.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np
import stage2_core as C2

QUICK = "--quick" in sys.argv
N_CAR = 2000 if QUICK else 20000
SEED = 20261215
Z99 = 2.5758


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    here = pathlib.Path(__file__).resolve(); digest = sha(here)
    core_sha = sha(here.with_name("stage2_core.py"))
    if "--hash" in sys.argv:
        print(f"stage2_validate.py {digest}\nstage2_core.py     {core_sha}"); return
    pf = here.with_name("stage2_predictions_quick.json" if QUICK else "stage2_predictions.json")
    pred = json.load(open(pf)); pred_sha = sha(pf)
    t0 = time.time(); bar = "=" * 112
    print(bar)
    print(f"Part V Stage 2 VALIDATION | carrier runs per arm = {N_CAR} {'(QUICK)' if QUICK else ''}\n"
          f"script sha256 {digest}\ncore   sha256 {core_sha}\npredictions {pf.name} sha256 {pred_sha}\n"
          f"  made by stage2_predict.py {pred['script_sha256']}")
    print(bar)
    if pred["core_sha256"] != core_sha:
        print("stage2_core.py differs from the one that made the predictions: stopping."); return
    if pred["quick"] != QUICK:
        print("Prediction file and this run disagree on quick mode: stopping."); return
    x0 = pred["x0"]; cfg = pred["configs"]
    keys = list(pred["arms"].keys())
    ss = np.random.SeedSequence(SEED).spawn(len(keys) + 1)
    res = {}
    print(f"\n  {'arm':16s} {'q_bar':>6} {'measured P':>26} {'P0 +- SE':>16} {'predicted P0':>16} {'k*':>4} {'z':>6}"
          f" {'bias b':>8} {'b/SE':>7} {'median cr.':>10}")
    for ai, key in enumerate(keys):
        a = pred["arms"][key]; g, kap = cfg[a["config"]]
        coin = np.random.default_rng(ss[-1].spawn(1)[0]) if a.get("coin") else None
        rec, ncr = C2.carrier(np.random.default_rng(ss[ai]), N_CAR, g, a["eps"], x0, kappa=kap, coin_rng=coin)
        ok = rec >= 0; P = np.bincount(rec[ok], minlength=3) / max(ok.sum(), 1)
        se = math.sqrt(P[0] * (1 - P[0]) / max(ok.sum(), 1))
        pr = a["primary"]; z = (P[0] - pr["P0"]) / math.hypot(se, pr["se_total0"])
        b = P[0] - 0.5
        res[key] = dict(P=P.tolist(), se0=se, absorbed=float(ok.mean()), median_crossings=float(np.median(ncr)),
                        predicted=pr["P0"], pred_se=pr["se_total0"], z=z, bias=b, bias_z=b / se,
                        qbar=pred["substrate"][a["config"]]["qbar"])
        print(f"  {key:16s} {res[key]['qbar']:6.3f} {str(np.round(P, 4).tolist()):>26} {P[0]:8.4f}+-{se:.4f} "
              f"{pr['P0']:8.4f}+-{pr['se_total0']:.4f} {str(a.get('kstar', '-')):>4} {z:+6.2f} {b:+8.4f} {b / se:+7.1f}"
              f" {np.median(ncr):10.0f}   absorbed {ok.mean():.4f}  [{time.time() - t0:.0f} s]", flush=True)

    model = [k for k in keys if not pred["arms"][k].get("coin")]
    q1 = {k: abs(res[k]["z"]) < Z99 for k in model}
    q2 = abs(res["S2_eps0.3_coin"]["bias_z"]) < Z99
    b3, b1 = abs(res["S2_eps0.3"]["bias"]), abs(res["S2_eps0.1"]["bias"])
    q3 = b1 > b3 / 2
    seq = [abs(res[f"{c}_eps0.3"]["bias"]) for c in ("S2", "S4", "S5", "S6")]
    q4 = all(seq[i + 1] < seq[i] for i in range(3))
    q5 = all(abs(res[f"{c}_eps0.3"]["bias_z"]) > Z99 for c in ("S1", "S2", "S3"))
    print("\n" + bar)
    print("DECISION (computed, criteria as frozen in stage2_predict.py):")
    print(f"  [{'CONFIRMED' if all(q1.values()) else 'NOT CONFIRMED'}] Q1 every arm agrees with its prediction: "
          + ", ".join(f"{k}:{'ok' if v else 'NO'}" for k, v in q1.items()))
    print(f"  [{'CONFIRMED' if q2 else 'NOT CONFIRMED'}] Q2 fair-coin control consistent with Born (b/SE = {res['S2_eps0.3_coin']['bias_z']:+.2f})")
    print(f"  [{'CONFIRMED' if q3 else 'NOT CONFIRMED'}] Q3 bias does not vanish with eps: |b(0.1)| = {b1:.4f} vs |b(0.3)|/2 = {b3 / 2:.4f}"
          f"  (|b(0.15)| = {abs(res['S2_eps0.15']['bias']):.4f})")
    print(f"  [{'CONFIRMED' if q4 else 'NOT CONFIRMED'}] Q4 |b| decreases with kappa (0, 0.3, 1, 3): " + ", ".join(f"{v:.4f}" for v in seq))
    print(f"  [{'CONFIRMED' if q5 else 'NOT CONFIRMED'}] Q5 Part I physics (kappa = 0) excludes Born in every arm")
    if not q2:
        print("  NOTE: the fair-coin control failed, so the carrier code is suspect; no other result should be used.")
    print(bar)
    of = here.with_name("stage2_results_quick.json" if QUICK else "stage2_results.json")
    json.dump(dict(script_sha256=digest, core_sha256=core_sha, predictions_sha256=pred_sha, N_CAR=N_CAR, arms=res,
                   Q1=q1, Q2=q2, Q3=q3, Q4=q4, Q5=q5), open(of, "w"), indent=1, default=float)
    print(f"Results: {of}  sha256 {sha(of)}\nWall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()