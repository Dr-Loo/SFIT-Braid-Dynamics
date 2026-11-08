"""
stage2_predict.py -- SFIT Part V Stage 2, PREDICTION step. Substrate only: no carrier outcome is computed
or read. Output frozen to stage2_predictions.json, to be committed before stage2_validate.py runs.

QUESTION. Stage 1b: the braid sign is balanced overall but not fair given its history. Does the resulting
record bias survive, how large is it, and does faster transverse mixing remove it?

CARRIER LAW (stage2_core.py): the over-crossing defect gains, x -> x + EPS u d_ab(x). This is the
label-independent form of the Part IV jump; x + EPS s d_ab would depend on how the defects are numbered.
Born weights for this carrier = x(0) = (0.5, 0.3, 0.2) (a fair u makes x a martingale; first-passage
bound (1 - H)/H = 0.001).

CONFIGURATIONS (substrate)        gamma   kappa   (kappa: transverse-order scrambling rate; 0 = Part I)
   S1                              0.25    0
   S2                              1       0
   S3                              4       0
   S4                              1       0.3
   S5                              1       1
   S6                              1       3
ARMS predicted: every configuration at EPS = 0.3; S2 also at EPS = 0.15 and 0.1; plus the fair-coin
control on S2 at EPS = 0.3 (predicted Born, no model needed).

REDUCED MODEL (Part IV pipeline, alphabet changed). Token = (pair, over-strand u), 6 values. Order-k chain
(k = 1..5) with the transition counts from independent substrate runs (2000 x T = 2000 per configuration,
in 50 groups), fallback to lower order for unseen histories, stationary start; the same jump and stopping
rule as the carrier. The counts are symmetrised under y -> -y (u -> -u), an exact symmetry of the
substrate, so the model has E[u | p] = 0 exactly. k = 0: tokens i.i.d. with the measured frequencies (no memory; should give Born).
  Point prediction: 40,000 runs per k. Uncertainty: B = 20 group-bootstrap replicates of the counts,
  8,000 runs each, common random numbers across k and replicates.
  SE(P) = sqrt(max(var_boot - P(1-P)/8000, 0) + P(1-P)/40000).
  Order k*: smallest k with |P_j - P_k| < 2.576 SD_boot(P_j - P_k) for all j > k (cumulative rule).
  Primary prediction = P_{k*}, with truncation term |P_5 - P_{k*}| added in quadrature; if no k < 5
  qualifies the arm is FLAGGED and P_5 is used with truncation |P_5 - P_4|.
No quantity here is adjusted to carrier outcomes.

PRE-STATED PREDICTIONS (checked by stage2_validate.py)
  Q1  Each arm: carrier P(channel 0) agrees with the primary prediction, |z| < 2.576.
  Q2  Fair-coin control: carrier P(channel 0) consistent with 0.5 at 99%.
  Q3  The bias b = P(channel 0) - 0.5 does not vanish as EPS decreases: |b(0.1)| > |b(0.3)| / 2 at S2.
      Reason: for consecutive jumps on the same pair, eps(u d(x) + u' d(x')) with x' = x + eps u d(x)
      = eps(u + u') d(x) + eps^2 u u' (grad d . d)(x) + O(eps^3). E[u u'] = 1 - 2q != 0 makes the second
      term a drift of order eps^2 per crossing, the same order as the per-crossing variance, so over the
      ~1/eps^2 crossings to absorption it gives an O(1) shift that does not go to zero with eps.
  Q4  At gamma = 1, |b| decreases monotonically with kappa = 0, 0.3, 1, 3.
  Q5  Within Part I physics (kappa = 0, S1-S3) every arm excludes Born (|b| / SE > 2.576).
DISCLOSURE: before these were written, a 4,000-run carrier look at gamma = 0.5, 2, 8 (EPS = 0.3, kappa = 0)
gave P(channel 0) ~ 0.42, 0.39, 0.39, and the fair-coin control at gamma = 2 gave 0.50. Those gamma
values are not validation arms. Q2 and Q5 were informed by that look; Q1, Q3, Q4 were not (no reduced
model, no EPS < 0.3, no kappa > 0 had been run).

Usage: python stage2_predict.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np
import stage2_core as C2

QUICK = "--quick" in sys.argv
X0 = [0.5, 0.3, 0.2]
CONFIGS = {"S1": (0.25, 0.0), "S2": (1.0, 0.0), "S3": (4.0, 0.0), "S4": (1.0, 0.3), "S5": (1.0, 1.0), "S6": (1.0, 3.0)}
ARMS = [(c, 0.3) for c in CONFIGS] + [("S2", 0.15), ("S2", 0.1)]
R_SUB = 300 if QUICK else 2000
T_SUB = 500.0 if QUICK else 2000.0
N_GROUPS = 10 if QUICK else 50
KMAX = 3 if QUICK else 5
N_POINT = 4000 if QUICK else 40000
N_BOOT = 2000 if QUICK else 8000
B = 4 if QUICK else 20
SEED = 20261201
Z99 = 2.5758


def rec_P(rec):
    ok = rec >= 0
    return np.bincount(rec[ok], minlength=3) / max(ok.sum(), 1)


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    here = pathlib.Path(__file__).resolve(); digest = sha(here)
    core = here.with_name("stage2_core.py"); core_sha = sha(core)
    if "--hash" in sys.argv:
        print(f"stage2_predict.py {digest}\nstage2_core.py    {core_sha}"); return
    t0 = time.time(); bar = "=" * 104
    print(bar)
    print(f"Part V Stage 2 PREDICTIONS {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}\ncore   sha256 {core_sha}")
    print(bar)
    ss = np.random.SeedSequence(SEED).spawn(len(CONFIGS) + 1)
    sim_seed = ss[-1]
    out = dict(script_sha256=digest, core_sha256=core_sha, quick=QUICK, x0=X0, configs=CONFIGS, KMAX=KMAX,
               N_POINT=N_POINT, N_BOOT=N_BOOT, B=B, substrate={}, arms={})
    counts_by = {}
    print("\nSubstrate statistics (independent runs):")
    for ci, (name, (g, kap)) in enumerate(CONFIGS.items()):
        runs = C2.events_fast(np.random.default_rng(ss[ci]), R_SUB, T_SUB, g, kappa=kap)
        grp = [[runs[r] for r in range(R_SUB) if r % N_GROUPS == q] for q in range(N_GROUPS)]
        gc = [C2.transition_counts(z, KMAX) for z in grp]          # raw counts per group
        counts_by[name] = gc
        fl = tot = 0; ncr = 0
        for p, s, u, t in runs:
            ncr += p.size
            for q in range(3):
                i = np.flatnonzero(p == q); uu = u[i[1:]] * u[i[:-1]]; fl += int((uu < 0).sum()); tot += uu.size
        qbar = fl / max(tot, 1)
        out["substrate"][name] = dict(gamma=g, kappa=kap, crossings=ncr, qbar=qbar, Euu_same_pair=1 - 2 * qbar)
        print(f"  {name} gamma={g:<5} kappa={kap:<4} crossings={ncr:9d}  q_bar={qbar:.4f}  E[u u'](same pair)={1 - 2 * qbar:+.4f}"
              f"  [{time.time() - t0:.0f} s]", flush=True)

    print("\nReduced-model predictions (P(channel 0); Born = 0.5):")
    arm_seeds = np.random.SeedSequence(SEED + 1).spawn(len(ARMS))
    for ai, (name, eps) in enumerate(ARMS):
        gc = counts_by[name]; full = C2.symmetrize({k: sum(z[k] for z in gc) for k in range(1, KMAX + 1)})
        point = {k: rec_P(C2.reduced(full, k, eps, X0, N_POINT, np.random.default_rng(sim_seed))) for k in range(0, KMAX + 1)}
        rb = np.random.default_rng(arm_seeds[ai])
        boot = np.zeros((B, KMAX + 1, 3))
        for b in range(B):
            pick = rb.integers(0, N_GROUPS, N_GROUPS)
            st = C2.symmetrize({k: sum(gc[q][k] for q in pick) for k in range(1, KMAX + 1)})
            for k in range(1, KMAX + 1):
                boot[b, k] = rec_P(C2.reduced(st, k, eps, X0, N_BOOT, np.random.default_rng(sim_seed)))
        rows = {}
        for k in range(1, KMAX + 1):
            p = point[k]
            se = [math.sqrt(max(boot[:, k, c].var(ddof=1) - p[c] * (1 - p[c]) / N_BOOT, 0) + p[c] * (1 - p[c]) / N_POINT) for c in range(3)]
            rows[k] = dict(P=p.tolist(), se=se)
        kstar = None
        for k in range(1, KMAX):
            if all(abs(point[j][0] - point[k][0]) < Z99 * float((boot[:, j, 0] - boot[:, k, 0]).std(ddof=1)) for j in range(k + 1, KMAX + 1)):
                kstar = k; break
        conv = kstar is not None
        ku = kstar if conv else KMAX
        trunc = abs(point[KMAX][0] - point[ku][0]) if conv else abs(point[KMAX][0] - point[KMAX - 1][0])
        se_tot = math.sqrt(rows[ku]["se"][0] ** 2 + trunc ** 2)
        key = f"{name}_eps{eps}"
        out["arms"][key] = dict(config=name, eps=eps, rows=rows, k0=point[0].tolist(), kstar=kstar, converged=conv,
                                primary=dict(k=ku, P=point[ku].tolist(), P0=float(point[ku][0]), se_total0=se_tot, trunc=trunc),
                                boot=boot.tolist())
        print(f"  {key:12s} " + "  ".join(f"k={k}:{point[k][0]:.4f}" for k in range(0, KMAX + 1))
              + f"   k*={kstar}{'' if conv else ' FLAGGED'}  primary {point[ku][0]:.4f} +- {se_tot:.4f}  [{time.time() - t0:.0f} s]", flush=True)
    out["arms"]["S2_eps0.3_coin"] = dict(config="S2", eps=0.3, coin=True,
                                         primary=dict(k=None, P=X0, P0=0.5, se_total0=0.0, trunc=0.0))
    pf = here.with_name("stage2_predictions_quick.json" if QUICK else "stage2_predictions.json")
    json.dump(out, open(pf, "w"), indent=1, default=float)
    print(f"\nFrozen: {pf}\nsha256 {sha(pf)}\nWall time {time.time() - t0:.0f} s")
    print("Next: commit this file (and both scripts), then run stage2_validate.py")


if __name__ == "__main__":
    main()