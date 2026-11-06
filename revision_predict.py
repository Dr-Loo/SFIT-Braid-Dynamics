"""
revision_predict.py -- predictions for the Part IV revision. Substrate only; no carrier outcome is read.
Output frozen to revision_predictions.json before revision_validate.py runs.

R1  gamma = 0.5 through the CURRENT pipeline (replaces the earlier k = 4, fixed-order prediction):
    ordered extraction, dt = 0.02, 10,000 substrate runs x T = 2,000 in 50 groups, k = 1..8,
    B = 50 group-bootstrap replicates (saved, for the k* distribution), 80,000 point runs,
    cumulative convergence rule, truncation term, EPS = 0.3, x(0) = (0.5, 0.3, 0.2).
R2  k = 0 ablation (no pair memory): pairs drawn independently with the measured stationary
    frequencies pi_p, orientation from the topological rule, same jump and stopping rule; at
    gamma = 0.5 (from the R1 data), 1 and 2 (2,000 substrate runs x T = 2,000 each, new seeds).
    Also k = 0 with exactly uniform pi (1/3 each): if pi is uniform, the k = 0 model has no gamma
    dependence at all, so any gamma dependence of the bias must come from pair memory.
    PRE-STATED: the k = 0 predictions do not reproduce the measured gamma dependence.

Usage: python revision_predict.py [--quick] [--boot B] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np

QUICK = "--quick" in sys.argv
B = int(sys.argv[sys.argv.index("--boot") + 1]) if "--boot" in sys.argv else (4 if QUICK else 50)
_argv = sys.argv; sys.argv = ["x"]
import pair_stats as PS
import ordering_fix as OF
import reduced_model as RM
import history_order_test as HOT
sys.argv = _argv

X0 = np.array([0.5, 0.3, 0.2]); EPS = 0.30; H = 0.999
G = np.array([1, -1, 1])
N_RUNS_SUB = 1000 if QUICK else 10000
N_RUNS_K0 = 400 if QUICK else 2000
T_SUB = 500.0 if QUICK else 2000.0
N_GROUPS = 10 if QUICK else 50
KMAX = 8
N_POINT = 10000 if QUICK else 80000
N_BOOT_RUNS = 4000 if QUICK else 20000
SEED = 20261114
Z99 = 2.5758


def rec_P(rec):
    ok = rec >= 0
    return np.bincount(rec[ok], minlength=3) / ok.sum()


def sim_k0(pi, runs, rng):
    """No pair memory: i.i.d. pairs with frequencies pi, topological orientation rule."""
    sigma = rng.choice([-1, 1], runs)
    x = np.tile(X0, (runs, 1)); done = np.zeros(runs, bool); rec = np.full(runs, -1); n = 0
    while n < RM.MAX_EVENTS and not done.all():
        p = rng.choice(3, runs, p=pi)
        s = sigma * (1 if n % 2 == 0 else -1) * G[p]; live = ~done
        x[live] = x[live] + (s[live] * EPS)[:, None] * RM.d_pair(x[live], p[live])
        x[live] = np.clip(x[live], 0, None); x[live] /= x[live].sum(1, keepdims=True)
        hit = live & (x.max(1) >= H); rec[hit] = x[hit].argmax(1); done |= hit; n += 1
    return rec


def main():
    here = pathlib.Path(__file__).resolve()
    digest = hashlib.sha256(here.read_bytes()).hexdigest()
    if "--hash" in sys.argv:
        print(digest); return
    t0 = time.time()
    print("=" * 100)
    print(f"Revision predictions | B={B} {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}")
    print("=" * 100)
    OF.X0 = X0; OF.DT = 0.02
    ss = np.random.SeedSequence(SEED).spawn(10)
    out = dict(sha256=digest, eps=EPS, x0=X0.tolist(), B=B)

    # ---------------- R1: gamma = 0.5, current pipeline
    gam = 0.5; per = N_RUNS_SUB // N_GROUPS
    counts = {k: np.zeros((N_GROUPS, 3 ** k, 3), np.int64) for k in range(1, KMAX + 2)}
    gseeds = ss[0].spawn(N_GROUPS); viol = 0; ncross = 0
    for g in range(N_GROUPS):
        sp, sg, _, _, _ = OF.sequences_ordered(np.random.default_rng(gseeds[g]), gam, per, T_SUB)
        b_, t_ = OF.violations(sp, sg); viol += b_; ncross += t_ + len(sp)
        for k in range(1, KMAX + 2):
            counts[k][g] = PS.transitions(sp, k)
    print(f"\nR1 gamma=0.5: {ncross} crossings, rule violations {viol} ({time.time() - t0:.0f} s)")
    stats_from = lambda idx: {"trans_counts_all": {str(k): counts[k][idx].sum(0).tolist() for k in range(1, KMAX + 1)}}
    full = stats_from(np.arange(N_GROUPS))
    diag = {k: HOT.g2_stat(counts[k].sum(0).astype(float), counts[k + 1].sum(0).astype(float)) for k in range(1, KMAX + 1)}
    sim_seed = ss[1]
    point = {k: rec_P(HOT.simulate_rec(full, k, N_POINT, sim_seed)) for k in range(1, KMAX + 1)}
    pi05 = counts[1].sum(0).sum(1).astype(float); pi05 /= pi05.sum()
    point[0] = rec_P(sim_k0(pi05, N_POINT, np.random.default_rng(sim_seed)))
    rng_b = np.random.default_rng(ss[2]); boot = np.zeros((B, KMAX + 1, 3))
    for b in range(B):
        idx = rng_b.integers(0, N_GROUPS, N_GROUPS); st = stats_from(idx)
        for k in range(1, KMAX + 1):
            boot[b, k] = rec_P(HOT.simulate_rec(st, k, N_BOOT_RUNS, sim_seed))
        if (b + 1) % 10 == 0 or QUICK:
            print(f"  bootstrap {b + 1}/{B} ({time.time() - t0:.0f} s)", flush=True)
    rows = []
    print(f"  {'k':>2} {'P_k':>26} {'SE(P0)':>7}  {'G2/df':>8}")
    for k in range(1, KMAX + 1):
        p = point[k]
        se = [math.sqrt(max(boot[:, k, c].var(ddof=1) - p[c] * (1 - p[c]) / N_BOOT_RUNS, 0) + p[c] * (1 - p[c]) / N_POINT)
              for c in range(3)]
        rows.append(dict(k=k, P=p.tolist(), se=se, G2_per_df=diag[k][0] / max(diag[k][1], 1)))
        print(f"  {k:>2} {str(np.round(p, 4).tolist()):>26} {se[0]:7.4f}  {rows[-1]['G2_per_df']:8.2f}")
    kstar = None
    for k in range(1, KMAX + 1):
        if all(abs(point[j][0] - point[k][0]) < Z99 * float((boot[:, j, 0] - boot[:, k, 0]).std(ddof=1))
               for j in range(k + 1, KMAX + 1)):
            kstar = k; break
    conv = kstar is not None and kstar < KMAX; ku = kstar if conv else KMAX
    trunc = abs(point[KMAX][0] - point[ku][0])
    prim = dict(rows[ku - 1]); prim["trunc"] = trunc; prim["se_total0"] = math.sqrt(prim["se"][0] ** 2 + trunc ** 2)
    print(f"  converged at k* = {kstar}{'' if conv else ' (NOT converged, FLAGGED)'}; primary P = {np.round(point[ku], 4).tolist()} "
          f"(SE total {prim['se_total0']:.4f});  k = 0 (no memory): P = {np.round(point[0], 4).tolist()}")
    out["R1"] = dict(gamma=gam, dt=0.02, rows=rows, kstar=kstar, converged=conv, primary=prim,
                     k0=point[0].tolist(), pi=pi05.tolist(), boot=boot.tolist(), violations=viol, crossings=ncross)

    # ---------------- R2: k = 0 ablation at gamma = 1, 2 (plus uniform pi)
    out["R2"] = {"0.5": dict(pi=pi05.tolist(), P=point[0].tolist())}
    print("\nR2 k = 0 ablation (no pair memory), N = %d runs each:" % N_POINT)
    print(f"  gamma=0.5: pi={np.round(pi05, 4).tolist()}  P={np.round(point[0], 4).tolist()}")
    for gi, g in enumerate((1.0, 2.0)):
        sp, _, _, _, _ = OF.sequences_ordered(np.random.default_rng(ss[3 + gi]), g, N_RUNS_K0, T_SUB)
        pi = np.bincount(np.concatenate(sp).astype(int), minlength=3).astype(float); pi /= pi.sum()
        P = rec_P(sim_k0(pi, N_POINT, np.random.default_rng(sim_seed)))
        out["R2"][str(g)] = dict(pi=pi.tolist(), P=P.tolist())
        print(f"  gamma={g}: pi={np.round(pi, 4).tolist()}  P={np.round(P, 4).tolist()}")
    Pu = rec_P(sim_k0(np.ones(3) / 3, N_POINT, np.random.default_rng(sim_seed)))
    out["R2"]["uniform"] = dict(pi=[1 / 3] * 3, P=Pu.tolist(), se0=math.sqrt(Pu[0] * (1 - Pu[0]) / N_POINT))
    print(f"  uniform pi: P={np.round(Pu, 4).tolist()}  (+- {out['R2']['uniform']['se0']:.4f}; independent of gamma)")
    pf = here.with_name("revision_predictions.json")
    json.dump(out, open(pf, "w"), indent=1)
    print(f"\nFrozen: {pf}\nsha256 {hashlib.sha256(pf.read_bytes()).hexdigest()}\nWall time {time.time() - t0:.0f} s")
    print("Next: python revision_validate.py")


if __name__ == "__main__":
    main()