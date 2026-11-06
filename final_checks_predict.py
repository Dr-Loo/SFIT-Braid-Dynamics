"""
final_checks_predict.py -- substrate-only predictions for the last 1D checks. No carrier outcome is read.
Output frozen to final_checks_predictions.json before final_checks_validate.py runs.

CONFIGURATIONS (x(0) = (0.5, 0.3, 0.2), EPS = 0.3, first passage 0.999)
  D1  gamma = 1, dt = 0.01   time-step check (earlier: dt = 0.02)
  D2  gamma = 2, dt = 0.01   time-step check (earlier: dt = 0.02)
  E1  gamma = 4, dt = 0.01   extrapolation beyond the tested range; k up to 10
      (gamma * dt = 0.04, the same per-step reversal probability as gamma = 2 at dt = 0.02)

METHOD: as history_order_test.py / test_B_predict.py, with
  * B = 50 group-bootstrap replicates (was 20);
  * KMAX = 8 for D1, D2 and 10 for E1;
  * converged order k* by the cumulative rule; if not converged within KMAX, k = KMAX is used and flagged;
  * truncation systematic reported: |P_KMAX(0) - P_k*(0)|, and a total SE = sqrt(stat^2 + trunc^2).
New seeds throughout.

PRE-STATED READING
  Time step: the dt = 0.01 prediction and the dt = 0.01 measurement must agree (|z| < 2.576 on P(0));
  the dt-shift of the measurement (0.01 vs 0.02) is reported, with no prediction for its size.
  Extrapolation: E1 agrees, or, if not converged within k <= 10, the limit of the order-k model is reported.

Usage: python final_checks_predict.py [--quick] [--boot B] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np

QUICK = "--quick" in sys.argv
B = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else (4 if QUICK else 50)
_argv = sys.argv; sys.argv = ["x"]
import pair_stats as PS
import ordering_fix as OF
import history_order_test as HOT
sys.argv = _argv

CONFIGS = [("D1", 1.0, 0.01, 8), ("D2", 2.0, 0.01, 8), ("E1", 4.0, 0.01, 10)]
X0 = [0.5, 0.3, 0.2]
N_RUNS_SUB = 1000 if QUICK else 10000
T_SUB = 500.0 if QUICK else 2000.0
N_GROUPS = 10 if QUICK else 50
N_POINT = 10000 if QUICK else 80000
N_BOOT_RUNS = 4000 if QUICK else 20000
SEED = 20261112
Z99 = 2.5758


def rec_P(rec):
    ok = rec >= 0
    return np.bincount(rec[ok], minlength=3) / ok.sum()


def main():
    here = pathlib.Path(__file__).resolve()
    digest = hashlib.sha256(here.read_bytes()).hexdigest()
    if "--hash" in sys.argv:
        print(digest); return
    t0 = time.time()
    print("=" * 100)
    print(f"Final 1D checks: predictions | substrate {N_RUNS_SUB} x T={T_SUB:g}, B={B} {'(QUICK)' if QUICK else ''}\n"
          f"script sha256 {digest}")
    print("=" * 100)
    OF.X0 = np.array(X0)
    ss = np.random.SeedSequence(SEED).spawn(3 * len(CONFIGS))
    out = dict(sha256=digest, eps=HOT.EPS, x0=X0, B=B, configs={})
    for ci, (name, gam, dt, kmax) in enumerate(CONFIGS):
        OF.DT = dt
        per = N_RUNS_SUB // N_GROUPS
        counts = {k: np.zeros((N_GROUPS, 3 ** k, 3), np.int64) for k in range(1, kmax + 2)}
        gseeds = ss[3 * ci].spawn(N_GROUPS); viol = 0; ncross = 0
        for g in range(N_GROUPS):
            sp, sg, _, _, _ = OF.sequences_ordered(np.random.default_rng(gseeds[g]), gam, per, T_SUB)
            b_, t_ = OF.violations(sp, sg); viol += b_; ncross += t_ + len(sp)
            for k in range(1, kmax + 2):
                counts[k][g] = PS.transitions(sp, k)
        print(f"\n{name}: gamma={gam}, dt={dt}: {ncross} crossings, rule violations {viol} ({time.time() - t0:.0f} s)")
        stats_from = lambda idx: {"trans_counts_all": {str(k): counts[k][idx].sum(0).tolist() for k in range(1, kmax + 1)}}
        full = stats_from(np.arange(N_GROUPS))
        diag = {k: HOT.g2_stat(counts[k].sum(0).astype(float), counts[k + 1].sum(0).astype(float)) for k in range(1, kmax + 1)}
        sim_seed = ss[3 * ci + 1]
        point = {k: rec_P(HOT.simulate_rec(full, k, N_POINT, sim_seed)) for k in range(1, kmax + 1)}
        rng_b = np.random.default_rng(ss[3 * ci + 2])
        boot = np.zeros((B, kmax + 1, 3))
        for b in range(B):
            idx = rng_b.integers(0, N_GROUPS, N_GROUPS); st = stats_from(idx)
            for k in range(1, kmax + 1):
                boot[b, k] = rec_P(HOT.simulate_rec(st, k, N_BOOT_RUNS, sim_seed))
            if (b + 1) % 10 == 0 or QUICK:
                print(f"  bootstrap {b + 1}/{B} ({time.time() - t0:.0f} s)", flush=True)
        rows = []
        print(f"  {'k':>2} {'P_k':>26} {'SE(P0)':>7}  {'G2/df':>8}")
        for k in range(1, kmax + 1):
            p = point[k]
            se = [math.sqrt(max(boot[:, k, c].var(ddof=1) - p[c] * (1 - p[c]) / N_BOOT_RUNS, 0) + p[c] * (1 - p[c]) / N_POINT)
                  for c in range(3)]
            rows.append(dict(k=k, P=p.tolist(), se=se, G2_per_df=diag[k][0] / max(diag[k][1], 1)))
            print(f"  {k:>2} {str(np.round(p, 4).tolist()):>26} {se[0]:7.4f}  {rows[-1]['G2_per_df']:8.2f}")
        kstar = None
        for k in range(1, kmax + 1):
            if all(abs(point[j][0] - point[k][0]) < Z99 * float((boot[:, j, 0] - boot[:, k, 0]).std(ddof=1))
                   for j in range(k + 1, kmax + 1)):
                kstar = k; break
        conv = kstar is not None and kstar < kmax
        ku = kstar if conv else kmax
        trunc = abs(point[kmax][0] - point[ku][0])
        prim = dict(rows[ku - 1]); prim["trunc"] = trunc
        prim["se_total0"] = math.sqrt(prim["se"][0] ** 2 + trunc ** 2)
        print(f"  converged at k* = {kstar}{'' if conv else f'  (NOT converged within k <= {kmax}: using k = {kmax}, FLAGGED)'}")
        print(f"  primary P = {np.round(point[ku], 4).tolist()};  SE(P0) stat {prim['se'][0]:.4f}, truncation {trunc:.4f}, "
              f"total {prim['se_total0']:.4f}")
        out["configs"][name] = dict(gamma=gam, dt=dt, kmax=kmax, rows=rows, kstar=kstar, converged=conv,
                                    primary=prim, violations=viol, crossings=ncross)
    pf = here.with_name("final_checks_predictions.json")
    json.dump(out, open(pf, "w"), indent=1)
    print(f"\nFrozen: {pf}\nsha256 {hashlib.sha256(pf.read_bytes()).hexdigest()}\nWall time {time.time() - t0:.0f} s")
    print("Next: python final_checks_validate.py")


if __name__ == "__main__":
    main()