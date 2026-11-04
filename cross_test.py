"""
cross_test.py -- crossing signs as the exchange sign: is the residual outcome bias real,
does it depend on the jump size, and does the mean population stay at x(0)?

CONTEXT
  carrier_jump_test.py: Part I ring crossings (v4 dynamics) used as the sign of the carrier-mediated
  jump x -> x + s EPS x_k (e_k - x) gave P(largest) = 0.516 +- 0.005 (10,000 runs, EPS = 0.3), and
  eps_check_cross.py gave 0.504 / 0.503 +- 0.008 at EPS = 0.3 / 0.1 (4,000 runs). The crossing signs
  are NOT conditionally fair: E[s | previous sign on the channel] = -0.150. So the populations are
  not a martingale under their own filtration. This experiment asks what that does to the outcomes.

REVISION v1 (after a --quick code check of v0, before any production run)
  v0 applied each crossing as TWO sequential jumps (+1 on the overtaker, then -1 on the
  overtaken). The quick check showed that its coin-orientation control was NOT Born
  (P(ch0) = 0.449, mean population drifting down at every checkpoint): the second jump acts on
  the state left by the first, so the pair has a second-order drift even with a fair coin.
  The CROSS result of carrier_jump_test.py therefore mixed two effects. v1 applies each crossing
  as ONE jump that is linear in its sign (below), so a fair orientation gives an exact martingale,
  and keeps the v0 paired update as two extra arms to document the artefact.

DYNAMICS
  Ring L = 5, N = 3 defects (one per channel), speed 1, Poisson velocity reversals at rate 1,
  Euler step dt = 0.02. Record: first channel with x_k >= H = 0.999. x(0) = (0.5, 0.3, 0.2).
  Crossing of channels a, b with orientation s (+1 if a overtakes b), SINGLE carrier-mediated jump:
      x -> x + s EPS [ x_a (e_a - x) - x_b (e_b - x) ]
  Derived: sums to zero; vertices are fixed points; populations stay >= 0 for EPS <= 0.5;
  drift = EPS [...] E[s | F], so x is a martingale iff the crossing orientation is conditionally fair.
  For each run, ONE substrate realisation drives all arms at once (paired comparison):
    eps0.30 / eps0.15 / eps0.10   single jump, real crossing orientation
    CTL0.30                        single jump, orientation = independent fair coin  (control)
    PAIRreal0.30 / PAIRcoin0.30    the v0 sequential +1/-1 update, real / coin orientation

QUESTIONS AND PRE-STATED PREDICTIONS
  Q1 (statistical) Bias b = P(record = channel 0) - 0.5, with 99% Wilson intervals, per arm.
     Two hypotheses are stated, and the data decide between them:
       H0  b = 0 at every EPS (drift averages out along paths; Born holds);
       H1  b > 0, the same at every EPS (anti-correlated signs give a second-order drift
           ~ EPS^2 rho per event; crossings keep a diffusive part ~ EPS^2; the ratio is
           EPS-independent, so the bias is too).
     Decision: H0 retained if every arm's 99% interval for b contains 0 AND the Born equivalence
     test passes (upper 99% bootstrap bound of TV(P, x(0)) <= 0.02). H1 supported if the pooled
     bias interval excludes 0 and no paired difference between EPS arms excludes 0.
     Any other pattern is reported as it is.
  Q2 (dynamical) Mean stopped population E[x_0(min(t, tau))] at event counts
     25, 50, 100, 200, 400, 800, 1600, 3200, with 99% intervals. A process whose mean is preserved
     stays at 0.5; a residual drift shows as a slow departure.
  Q3 (structure, no prediction) Where the sign memory comes from: E[s | previous sign] split by
     whether the previous event on the channel came from the SAME defect pair, and by the time
     since that event. Measured on a separate, unstopped substrate run (2,000 runs x T = 500).
  CONTROL CTL must pass the Born test and show |E[s | previous sign]| < 0.01; otherwise the run is
     invalid.
  PAIR arms (documentation of the v0 artefact): PAIRcoin0.30 is predicted NOT Born (TV_lo > 0.02),
     as seen in the v0 quick check. No prediction for PAIRreal0.30.

POWER
  40,000 runs: SE of P(channel 0) ~ 0.0025, so a bias of 0.01 is ~4 SE.

Usage: python cross_test.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np

QUICK = "--quick" in sys.argv
TRIALS = 4000 if QUICK else 40000
BATCH = 4000
X0 = np.array([0.5, 0.3, 0.2]); NCH = 3
H, EPS_TV, CONF, N_BOOT = 0.999, 0.02, 0.99, 2000
L, C, GAMMA, DT = 5.0, 1.0, 1.0, 0.02
MAX_STEPS = 4_000_000
# (name, eps, coin orientation?, paired v0 update?)
ARMS = [("eps0.30", 0.30, False, False), ("eps0.15", 0.15, False, False), ("eps0.10", 0.10, False, False),
        ("CTL0.30", 0.30, True, False), ("PAIRreal0.30", 0.30, False, True), ("PAIRcoin0.30", 0.30, True, True)]
CHECKS = [25, 50, 100, 200, 400, 800, 1600, 3200]
SEED = 20261103
Z99 = 2.5758
PAIRS = [(0, 1), (0, 2), (1, 2)]


def jump_single(x, a, b, s, eps):
    """One carrier-mediated jump for a crossing of channels a, b with orientation s."""
    r = np.arange(len(a)); xa = x[r, a]; xb = x[r, b]
    d = -(xa - xb)[:, None] * x
    d[r, a] += xa; d[r, b] -= xb
    x = x + (s * eps)[:, None] * d
    x = np.clip(x, 0.0, None)
    return x / x.sum(1, keepdims=True)


def jump(x, k, s, eps):
    r = np.arange(len(k)); xk = x[r, k]
    e = np.zeros_like(x); e[r, k] = 1.0
    x = x + (s * eps * xk)[:, None] * (e - x)
    x = np.clip(x, 0.0, None)
    return x / x.sum(1, keepdims=True)


class Memory:
    def __init__(self, R):
        self.prev = np.zeros((R, NCH), int); self.c = np.zeros((2, 2), np.int64)

    def add(self, rows, k, s):
        p = self.prev[rows, k]
        for a, pa in ((1, 1), (0, -1)):
            for b, sb in ((1, 1), (0, -1)):
                self.c[a, b] += int(np.sum((p == pa) & (s == sb)))
        self.prev[rows, k] = s

    def mean(self, a):
        n = self.c[a].sum(); m = (self.c[a, 1] - self.c[a, 0]) / max(n, 1)
        se = math.sqrt(max(1 - m * m, 1e-12) / max(n, 1))
        return m, m - Z99 * se, m + Z99 * se, int(n)


def batch(rng, R):
    X = rng.uniform(0, L, (R, NCH)); V = C * rng.choice([-1, 1], (R, NCH))
    fl = np.stack([np.floor((X[:, i] - X[:, j]) / L) for i, j in PAIRS], 1)
    A = len(ARMS)
    x = np.tile(X0, (A, R, 1)); done = np.zeros((A, R), bool); rec = np.full((A, R), -1)
    nev = np.zeros((A, R), int)
    snap = np.full((A, len(CHECKS), R, NCH), np.nan)
    mems = [Memory(R) for _ in ARMS]
    for step in range(MAX_STEPS):
        live = ~done.all(0)
        if not live.any():
            break
        X[live] += V[live] * DT
        flip = live[:, None] & (rng.random((R, NCH)) < GAMMA * DT)
        V[flip] *= -1
        nfl = np.stack([np.floor((X[:, i] - X[:, j]) / L) for i, j in PAIRS], 1)
        ch = (nfl != fl) & live[:, None]
        if ch.any():
            for p, (i, j) in enumerate(PAIRS):
                rows_all = np.flatnonzero(ch[:, p])
                if rows_all.size == 0:
                    continue
                up = nfl[rows_all, p] > fl[rows_all, p]
                a_real = np.where(up, i, j); b_real = np.where(up, j, i)
                coin = rng.random(rows_all.size) < 0.5
                a_ctl = np.where(coin, i, j); b_ctl = np.where(coin, j, i)
                for ai, (name, eps, ctl, paired) in enumerate(ARMS):
                    act = ~done[ai, rows_all]
                    if not act.any():
                        continue
                    rows = rows_all[act]
                    a = (a_ctl if ctl else a_real)[act]; b = (b_ctl if ctl else b_real)[act]
                    for kk, ss in ((a, 1), (b, -1)):
                        mems[ai].add(rows, kk, np.full(rows.size, ss))
                    if paired:
                        for kk, ss in ((a, 1), (b, -1)):
                            x[ai, rows] = jump(x[ai, rows], kk, np.full(rows.size, ss), eps)
                    else:
                        x[ai, rows] = jump_single(x[ai, rows], a, b, np.ones(rows.size), eps)
                    before = nev[ai, rows].copy(); nev[ai, rows] += 2
                    for ci, cnum in enumerate(CHECKS):
                        hitc = (before < cnum) & (nev[ai, rows] >= cnum)
                        if hitc.any():
                            snap[ai, ci, rows[hitc]] = x[ai, rows[hitc]]
                    hit = x[ai, rows].max(1) >= H
                    if hit.any():
                        hr = rows[hit]; rec[ai, hr] = x[ai, hr].argmax(1); done[ai, hr] = True
        fl = nfl
    for ai in range(A):                                      # stopped process: x(min(t, tau))
        for ci in range(len(CHECKS)):
            miss = np.isnan(snap[ai, ci, :, 0])
            snap[ai, ci, miss] = x[ai, miss]
    return rec, done, snap, mems, nev


def memory_structure(rng, R=2000, T=500.0):
    """Unstopped substrate: E[s | previous sign] by same/different pair and by time gap."""
    X = rng.uniform(0, L, (R, NCH)); V = C * rng.choice([-1, 1], (R, NCH))
    fl = np.stack([np.floor((X[:, i] - X[:, j]) / L) for i, j in PAIRS], 1)
    prev_s = np.zeros((R, NCH), int); prev_pair = np.full((R, NCH), -1); prev_t = np.zeros((R, NCH))
    gaps = [0, 0.5, 1, 2, 5, 1e9]
    acc = {}                                               # key -> [sum s*prev, count]
    t = 0.0
    for step in range(int(T / DT)):
        X += V * DT; t += DT
        V[rng.random((R, NCH)) < GAMMA * DT] *= -1
        nfl = np.stack([np.floor((X[:, i] - X[:, j]) / L) for i, j in PAIRS], 1)
        ch = nfl != fl
        for p, (i, j) in enumerate(PAIRS):
            rows = np.flatnonzero(ch[:, p])
            if rows.size == 0:
                continue
            up = nfl[rows, p] > fl[rows, p]
            a = np.where(up, i, j); b = np.where(up, j, i)
            for kk, ss in ((a, 1), (b, -1)):
                ps = prev_s[rows, kk]; ok = ps != 0
                same = prev_pair[rows, kk] == p
                gap = t - prev_t[rows, kk]
                for key_same in (True, False):
                    for g in range(len(gaps) - 1):
                        sel = ok & (same == key_same) & (gap >= gaps[g]) & (gap < gaps[g + 1])
                        if sel.any():
                            k_ = (key_same, g); v = acc.setdefault(k_, [0.0, 0])
                            v[0] += float(np.sum(ss * ps[sel])); v[1] += int(sel.sum())
                prev_s[rows, kk] = ss; prev_pair[rows, kk] = p; prev_t[rows, kk] = t
        fl = nfl
    return acc, gaps


def wilson(k, n, z=Z99):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    here = pathlib.Path(__file__).resolve()
    digest = hashlib.sha256(here.read_bytes()).hexdigest()
    if "--hash" in sys.argv:
        print(digest); return
    t0 = time.time()
    print("=" * 104)
    print(f"Crossing-sign test | runs={TRIALS} (batches of {BATCH}) H={H} x0={X0.tolist()} "
          f"{'(QUICK)' if QUICK else ''}\nscript sha256 {digest}")
    print("=" * 104)
    ss = np.random.SeedSequence(SEED).spawn(TRIALS // BATCH + 2)
    A = len(ARMS)
    recs = [[] for _ in ARMS]; dones = [[] for _ in ARMS]; snaps = [[] for _ in ARMS]
    mem_tot = np.zeros((A, 2, 2), np.int64); evs = [[] for _ in ARMS]
    for bi in range(TRIALS // BATCH):
        rec, done, snap, mems, nev = batch(np.random.default_rng(ss[bi]), BATCH)
        for ai in range(A):
            recs[ai].append(rec[ai]); dones[ai].append(done[ai]); snaps[ai].append(snap[ai])
            mem_tot[ai] += mems[ai].c; evs[ai].append(nev[ai])
        print(f"  batch {bi+1}/{TRIALS//BATCH} done ({time.time()-t0:.0f} s)", flush=True)

    out = {}; rng_b = np.random.default_rng(ss[-1])
    print(f"\nQ1  outcomes   (bias b = P(ch0) - 0.5, 99% Wilson; Born test: TV_hi <= {EPS_TV})")
    for ai, (name, eps, ctl, paired) in enumerate(ARMS):
        rec = np.concatenate(recs[ai]); done = np.concatenate(dones[ai])
        ok = rec >= 0; nr = int(ok.sum()); cnt = np.bincount(rec[ok], minlength=NCH); P = cnt / nr
        lo, hi = wilson(cnt[0], nr)
        boot = rng_b.multinomial(nr, P, size=N_BOOT) / nr
        tvh = float(np.percentile(0.5 * np.abs(boot - X0).sum(1), 100 * (1 + CONF) / 2))
        m = Memory(1); m.c = mem_tot[ai]; mp = m.mean(1)
        out[name] = dict(eps=eps, control=ctl, paired=paired, P=P.tolist(), bias=float(P[0] - 0.5),
                         bias_ci=(lo - 0.5, hi - 0.5), tv=float(0.5 * np.abs(P - X0).sum()), tv_hi=tvh,
                         born_pass=bool(tvh <= EPS_TV and done.mean() >= 0.99), absorbed=float(done.mean()),
                         sign_memory=mp[:3], median_events=float(np.median(np.concatenate(evs[ai]))))
        print(f"  {name}: P={np.round(P, 4).tolist()}  b={P[0]-0.5:+.4f} [{lo-0.5:+.4f},{hi-0.5:+.4f}]  "
              f"TV_hi={tvh:.4f} Born {'PASS' if out[name]['born_pass'] else 'FAIL'}  "
              f"E[s|prev+]={mp[0]:+.4f}  absorbed={done.mean():.4f}  median events={out[name]['median_events']:.0f}")

    print("\n  Paired differences in P(ch0) between EPS arms (same crossings), 99% interval:")
    r = {n: (np.concatenate(recs[ai]) == 0).astype(float) for ai, (n, _, c, pr) in enumerate(ARMS) if not c and not pr}
    diffs = {}
    for a_, b_ in (("eps0.30", "eps0.10"), ("eps0.30", "eps0.15"), ("eps0.15", "eps0.10")):
        d = r[a_] - r[b_]; m = d.mean(); se = d.std(ddof=1) / math.sqrt(d.size)
        diffs[f"{a_}-{b_}"] = (m, m - Z99 * se, m + Z99 * se)
        print(f"    {a_} - {b_}: {m:+.4f} [{m - Z99*se:+.4f},{m + Z99*se:+.4f}]")
    pooled = np.concatenate([r[n] for n in r]);
    # arms are paired, so pool per-run means to get an honest interval
    per_run = np.mean([r[n] for n in r], axis=0); pm = per_run.mean() - 0.5
    pse = per_run.std(ddof=1) / math.sqrt(per_run.size)
    print(f"  Pooled bias over the three EPS arms (per-run average): {pm:+.4f} [{pm - Z99*pse:+.4f},{pm + Z99*pse:+.4f}]")

    print("\nQ2  mean stopped population E[x_0(min(t,tau))] - 0.5 by event count (99% interval)")
    traj = {}
    for ai, (name, eps, ctl, paired) in enumerate(ARMS):
        S = np.concatenate(snaps[ai], axis=1)[:, :, 0]          # (checks, runs)
        row = []
        for ci, cnum in enumerate(CHECKS):
            v = S[ci]; m = v.mean() - 0.5; se = v.std(ddof=1) / math.sqrt(v.size)
            row.append((cnum, m, m - Z99 * se, m + Z99 * se))
        traj[name] = row
        print(f"  {name}: " + "  ".join(f"{c_}:{m:+.4f}{'*' if (lo > 0 or hi < 0) else ''}" for c_, m, lo, hi in row))
    print("  (* = 99% interval excludes 0)")

    print("\nQ3  sign-memory structure (unstopped substrate, 2,000 runs x T = 500):")
    acc, gaps = memory_structure(np.random.default_rng(ss[-2]))
    q3 = {}
    for same in (True, False):
        for g in range(len(gaps) - 1):
            if (same, g) in acc:
                s_, n_ = acc[(same, g)]; m = s_ / n_
                lab = f"{'same pair' if same else 'other pair'}, gap [{gaps[g]:g},{gaps[g+1]:g})"
                q3[lab] = (m, n_)
                print(f"  {lab:32}: E[s * s_prev] = {m:+.4f}  (n={n_})")

    ctl = out["CTL0.30"]
    valid = ctl["born_pass"] and abs(ctl["sign_memory"][0]) < 0.01
    pc = out["PAIRcoin0.30"]; pc_lo = None
    real = [out[n] for n in ("eps0.30", "eps0.15", "eps0.10")]
    h0 = all(o["bias_ci"][0] <= 0 <= o["bias_ci"][1] and o["born_pass"] for o in real)
    pooled_excl = (pm - Z99 * pse) > 0 or (pm + Z99 * pse) < 0
    no_eps_dep = all(lo <= 0 <= hi for (_, lo, hi) in diffs.values())
    h1 = pooled_excl and pm > 0 and no_eps_dep
    print("\n" + "=" * 104)
    print("DECISION (computed):")
    if not valid and QUICK:
        print("  QUICK mode: underpowered; code check only.")
    elif not valid:
        print("  RUN INVALID: the control failed.")
    print(f"  Control CTL: Born {'PASS' if ctl['born_pass'] else 'FAIL'}, sign memory {ctl['sign_memory'][0]:+.4f}")
    print(f"  PAIRcoin0.30 (v0 artefact): P(ch0) = {pc['P'][0]:.4f}, bias {pc['bias']:+.4f} "
          f"[{pc['bias_ci'][0]:+.4f},{pc['bias_ci'][1]:+.4f}]  -> predicted NOT Born: "
          f"{'CONFIRMED' if not pc['born_pass'] and (pc['bias_ci'][1] < 0 or pc['bias_ci'][0] > 0) else 'NOT CONFIRMED'}")
    print(f"  H0 (no bias, Born at every EPS): {'RETAINED' if h0 else 'REJECTED'}")
    print(f"  H1 (positive, EPS-independent bias): {'SUPPORTED' if h1 else 'NOT SUPPORTED'}")
    if not h0 and not h1:
        print("  Neither hypothesis fits; see the numbers above.")
    print("=" * 104)
    json.dump(dict(sha256=digest, runs=TRIALS, arms=out, paired_diffs=diffs, pooled_bias=(pm, pm - Z99 * pse, pm + Z99 * pse),
                   trajectory=traj, memory_structure=q3, valid=bool(valid), H0=bool(h0), H1=bool(h1)),
              open(here.with_name("cross_test_results.json"), "w"), indent=1, default=float)
    print(f"Wall time {time.time() - t0:.0f} s. Results: {here.with_name('cross_test_results.json')}")


if __name__ == "__main__":
    main()
