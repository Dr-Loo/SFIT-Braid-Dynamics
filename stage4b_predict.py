"""
stage4b_predict.py -- SFIT Part V Stage 4b, PREDICTION step: two diagnostics of the Stage 4 residuals.
Substrate-derived theta and the analytic law only; no new carrier outcome is computed or read.
Frozen to stage4b_predictions.json; commit it in its OWN commit before running stage4b_validate.py.

STAGE 4 OUTCOME (frozen criteria): LAW NOT SUPPORTED, chi^2 = 40.6 on 19 arms (p = 0.003).
Post hoc breakdown (not a test): 12.4 of the 40.6 came from the two two-channel arms at H = 0.99
(z = -2.72, -2.23), whose records form after only 23-28 crossings; 9 of the 10 three-channel arms had z > 0
(the carrier shows slightly less bias than predicted), partly because the frozen three-channel predictions
carried Monte Carlo noise of +-0.0025 (e.g. S2: frozen 0.4095, converged 0.4138 from three later runs).

INPUTS (hash-checked): theta and SE(theta) per configuration are taken unchanged from the frozen
stage4_predictions.json (sha256 6beb1107...); the solver is record_sde / edge_law from stage4_predict.py
(sha256 cae823fe...). No new substrate statistics.

D1  Low-threshold two-channel residual. Hypothesis: a finite-jump effect that vanishes as EPS -> 0.
    Arms, x0 = (0.7, 0.3, 0), H = 0.99: S2 at EPS 0.3, 0.1, 0.05 (40,000 runs each);
    B3 at EPS 0.3 (40,000) and 0.1 (20,000). Prediction: the exact law (I_y - I_{1-H}) / (1 - 2 I_{1-H}).
    EXPLAINED if (a) both EPS = 0.3 offsets (carrier - law) are negative, (b) at the smallest EPS of each
    configuration the carrier agrees with the law (|z| < 2.576), and (c) at S2, |offset(0.05)| < |offset(0.3)|.
    NOT EXPLAINED if the smallest-EPS arm of either configuration excludes the law at 99%. Otherwise OPEN.
D2  Three-channel offset. x0 = (0.5, 0.3, 0.2), H = 0.999, configurations S2, B3, N1, N2, N3.
    Predictions: SDE, dt = 0.002 (converged within noise: 0.4137, 0.4138 against 0.4140 at dt = 0.0005),
    160,000 paths each, in four independent blocks of 40,000 (their spread is printed as a check).
    Carrier: EPS 0.3, 80,000 runs, all five (pooled 99% half-width ~0.0027); EPS 0.1, 20,000 runs, at S2 and N2.
    Offset delta_i = carrier - theory at EPS 0.3; pooled delta = inverse-variance mean, 99% interval.
       NO RESIDUAL       the 99% interval contains 0 and its half-width is <= 0.003.
       RESIDUAL          the 99% interval excludes 0; delta is then reported as a measured correction beyond
                         the diffusion limit, and its EPS dependence is read from S2 and N2:
                         "EPS-independent" if |delta(0.1) - delta(0.3)| < 2.576 SE at both.
       UNRESOLVED        otherwise.
C   Control: fair coin, S2, x0 = (0.7, 0.3, 0), H = 0.99, EPS 0.3, 20,000 runs: Born finite-H 0.7041.
Usage: python stage4b_predict.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np
import stage4_predict as P4

QUICK = "--quick" in sys.argv
STAGE4_PRED_SHA = "6beb1107cf74a4e632914688bfaee674b529b33f5dbac80430ed29936a17ca18"
STAGE4_SCRIPT_SHA = "cae823fe745e3df8925a55f29416c8c6a2a763f2861489b3f251b962a3f71235"
STD, E7 = [0.5, 0.3, 0.2], [0.7, 0.3, 0.0]
BLOCKS = 4
R_BLOCK = 2000 if QUICK else 40000
R_SENS = 2000 if QUICK else 20000
DT = 0.002
SEED = 20261601
ARMS = ([dict(diag="D1", config="S2", x0=E7, H=0.99, eps=e, n=40000) for e in (0.3, 0.1, 0.05)]
        + [dict(diag="D1", config="B3", x0=E7, H=0.99, eps=0.3, n=40000), dict(diag="D1", config="B3", x0=E7, H=0.99, eps=0.1, n=20000)]
        + [dict(diag="D2", config=c, x0=STD, H=0.999, eps=0.3, n=80000) for c in ("S2", "B3", "N1", "N2", "N3")]
        + [dict(diag="D2", config=c, x0=STD, H=0.999, eps=0.1, n=20000) for c in ("S2", "N2")]
        + [dict(diag="C", config="S2", x0=E7, H=0.99, eps=0.3, n=20000, coin=True)])


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    here = pathlib.Path(__file__).resolve(); digest = sha(here)
    s4p = here.with_name("stage4_predictions.json"); s4s = here.with_name("stage4_predict.py")
    core_sha = sha(here.with_name("stage3_core.py"))
    if "--hash" in sys.argv:
        print(f"stage4b_predict.py {digest}\nstage4_predict.py  {sha(s4s)}\nstage3_core.py     {core_sha}"); return
    t0 = time.time(); bar = "=" * 108
    print(bar); print(f"Part V Stage 4b PREDICTIONS {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}\ncore   sha256 {core_sha}")
    ok4 = sha(s4p) == STAGE4_PRED_SHA and sha(s4s) == STAGE4_SCRIPT_SHA
    print(f"stage4_predictions.json {sha(s4p)[:16]}..., stage4_predict.py {sha(s4s)[:16]}...: {'match Stage 4' if ok4 else 'DO NOT MATCH Stage 4'}")
    print(bar)
    if not ok4 and not QUICK:
        print("Stopping."); return
    s4 = json.load(open(s4p)); th = {k: (v["theta"], v["theta_se"]) for k, v in s4["substrate"].items()}
    subs = {k: dict(gamma_rt=v["gamma_rt"], kappa=v["kappa"]) for k, v in s4["substrate"].items()}
    print("theta (frozen in Stage 4): " + ", ".join(f"{k} {t:.4f}+-{s:.4f}" for k, (t, s) in th.items()))
    ss = np.random.SeedSequence(SEED).spawn(20)
    d2 = {}
    print("\nD2 three-channel predictions, x0 = (0.5, 0.3, 0.2), H = 0.999 (SDE, dt = 0.002):")
    for ci, c in enumerate(("S2", "B3", "N1", "N2", "N3")):
        t_, t_se = th[c]; blocks = []
        for b in range(BLOCKS):
            blocks.append(P4.record_sde(STD, t_, 0.999, R_BLOCK, ss[ci].spawn(BLOCKS)[b], dt=DT)[0])
        P0 = float(np.mean(blocks)); se_mc = math.sqrt(P0 * (1 - P0) / (BLOCKS * R_BLOCK))
        sseed = ss[10 + ci]; dth = max(2 * t_se, 0.01)
        d = (P4.record_sde(STD, t_ + dth, 0.999, R_SENS, sseed, dt=0.004)[0] - P4.record_sde(STD, t_ - dth, 0.999, R_SENS, sseed, dt=0.004)[0]) / (2 * dth)
        se0 = math.sqrt(se_mc ** 2 + (d * t_se) ** 2 + 0.0008 ** 2)
        d2[c] = dict(P0=P0, se0=se0, se_mc=se_mc, blocks=[float(x) for x in blocks], dPdtheta=float(d), theta=t_)
        print(f"   {c}: theta {t_:.4f}  blocks " + ", ".join(f"{x:.4f}" for x in blocks)
              + f"  (spread sd {np.std(blocks, ddof=1):.4f}, binomial {math.sqrt(P0 * (1 - P0) / R_BLOCK):.4f})"
              f"  -> {P0:.4f} +- {se0:.4f}   [Stage 4 frozen {next(a['prediction']['P0'] for a in s4['arms'] if a['config'] == c and a['x0'] == STD and a['H'] == 0.999 and a['eps'] == 0.3):.4f}]"
              f"  [{time.time() - t0:.0f} s]", flush=True)
    arms = []
    print("\nArms:")
    for a in ARMS:
        c = a["config"]; t_, t_se = th[c]
        if a.get("coin"):
            p = dict(P0=(a["x0"][0] - (1 - a["H"])) / (2 * a["H"] - 1), se0=0.0, method="Born finite-H, fair coin")
        elif a["x0"][2] == 0.0:
            P0 = P4.edge_law(t_, a["x0"][0], a["H"])
            d = (P4.edge_law(t_ + 1e-4, a["x0"][0], a["H"]) - P4.edge_law(t_ - 1e-4, a["x0"][0], a["H"])) / 2e-4
            p = dict(P0=P0, se0=abs(d) * t_se, method="exact two-channel")
        else:
            p = dict(P0=d2[c]["P0"], se0=d2[c]["se0"], method="SDE (D2 block above)")
        arms.append(dict(a, n=(200 if QUICK else a["n"]), gamma_rt=subs[c]["gamma_rt"], kappa=subs[c]["kappa"], theta=t_, prediction=p))
        print(f"   {a['diag']:2s} {c} x0={a['x0']} H={a['H']:<6} eps={a['eps']:<5} n={arms[-1]['n']:6d}  predicted {p['P0']:.4f} +- {p['se0']:.4f}  ({p['method']})")
    out = dict(script_sha256=digest, core_sha256=core_sha, stage4_predictions_sha256=sha(s4p), quick=QUICK, K=s4["K"], d2=d2, arms=arms)
    pf = here.with_name("stage4b_predictions_quick.json" if QUICK else "stage4b_predictions.json")
    json.dump(out, open(pf, "w"), indent=1, default=float)
    print(f"\nFrozen: {pf}\nsha256 {sha(pf)}\nWall time {time.time() - t0:.0f} s\nNext: commit this file on its own, then run stage4b_validate.py")


if __name__ == "__main__":
    main()