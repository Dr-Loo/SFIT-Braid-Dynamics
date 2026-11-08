"""
stage1_braid_sign.py -- SFIT Part V, Stage 1 (substrate only; no carrier, no populations).
Is the Part I planar braid-generator sign conditionally fair given the event history?

DYNAMICS (Part I v6 planar substrate, sfit_substrate_v6_2D_braid_structure.py, unchanged physics)
  N = 3 defects in the plane, speed c = 2, Poisson direction scattering at rate GAMMA = 2 (heading
  kicked by a uniform angle in [-pi, pi]), harmonic trap K_TRAP = 0.1, Euler step dt:
      X <- X + (c (cos th, sin th) - K_TRAP X) dt,   then scattering.
  Within a step positions move linearly, so a change of x-order between defects a, b happens at the
  exact fraction f = d0 / (d0 - d1) of the step (d = x_a - x_b), and y is interpolated at f.
  Crossings inside one step are processed in order of f (Part IV lesson).
  Generator and sign follow the Part I extractor (Appendix B): after the swap, the two defects occupy
  ranks k, k+1 in the x-order; generator i = k (0 or 1, i.e. sigma_1, sigma_2); sign s = +1 if the
  defect now on the LEFT is higher in y, else -1. A swap of non-adjacent ranks is counted as unresolved.

THE FILTRATION (fixed in advance)
  The carrier sees only the event sequence, so the relevant history is the sequence of (generator,
  sign) tokens. Fairness at depth k:  E[ s_n | tokens n-k..n-1, i_n ] = 0.  Conditioning on substrate
  variables the carrier never sees (positions, headings) is NOT the test: given the full state the sign
  is nearly determined for any substrate.

MEASUREMENTS
  M1  k = 0: P(s = +1 | i) per generator, 99% interval (marginal balance).
  M2  k = 1..8: likelihood-ratio test of s independent of the depth-k history given i:
        G2 = 2 sum n(h,i,s) ln[ p(s | h, i) / p(s | i) ],  df = (number of histories seen) - 1, per i;
      p-value from chi-square; histories seen fewer than 50 times (for a given i) are pooled into one
      cell so that the chi-square approximation holds. [Pooling added after the --quick code check: without
      it the fair-coin control was rejected at k = 7, 8 because of sparse cells.] Also the predictability
        Pi_k = sum_{h,i} n(h,i) |E[s | h, i]| / n   (0 for a fair sign, 1 for a determined one),
      with a 99% group-bootstrap interval, and the largest |E[s | h, i]| among cells with n >= 2000.
  M3  Sign correlation E[s_n s_{n+l}], for the SAME pair of defects and for different pairs, at
      lags l = 1..60; and by time gap for consecutive crossings of the same defect pair.
  M4  Generator-sequence autocorrelation P(i_{n+l} = i_n) - sum pi_i^2.
  M5  Time step: M1-M3 summary at dt = 0.02, 0.01, 0.005 (smaller samples at the smaller steps).

CONTROLS (same generator sequence; failure invalidates the run)
  C1  fair coin signs: M2 must NOT reject at any k (p > 0.01) and Pi_k must be consistent with the
      value expected for a fair sign at this sample size (shown by the control itself).
  C2  anti-correlated synthetic signs: s_n = -s_{n-1} with probability 0.6 for the same defect pair,
      fair otherwise: M2 must reject (p < 0.01) at k = 1.

PREDICTIONS (written before running)
  P1  Quick recrossings of the same defect pair carry opposite signs:
      E[s_n s_{n-1} | same defect pair, gap < 1] < 0 (99% interval below 0).
  P2  Hence the sign is NOT conditionally fair: M2 rejects at k = 1.
  Open (no prediction): whether the same-pair sign correlation decays with lag (unlike Part IV's
  orientation plateau) or persists.

DECISION
  CANDIDATE   if M2 does not reject at any k = 1..8 (all p > 0.01) and C1, C2 behave as required
              -> Stage 2 tests the carrier.
  NOT FAIR    if M2 rejects at some k -> Stage 2 becomes a Part IV-style prediction of the bias.
  INVALID     if a control fails.

Usage: python stage1_braid_sign.py [--quick] [--hash]
Prediction/decision outputs should be committed to the repository before any Stage 2 carrier run.
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np
from scipy.stats import chi2

QUICK = "--quick" in sys.argv
N, C, GAMMA, K_TRAP = 3, 2.0, 2.0, 0.1
DT_MAIN = 0.02
R_MAIN = 600 if QUICK else 5000
T_MAIN = 2000.0 if QUICK else 4000.0
R_DT = 300 if QUICK else 1500
T_DT = 1000.0 if QUICK else 3000.0
N_GROUPS = 10 if QUICK else 50
KMAX = 8
MIN_CELL = 2000
LMAX = 60
B_BOOT = 20 if QUICK else 100
SEED = 20261120
Z99 = 2.5758
PAIRS = [(0, 1), (0, 2), (1, 2)]
PAIR_ID = {(0, 1): 0, (0, 2): 1, (1, 2): 2, (1, 0): 0, (2, 0): 1, (2, 1): 2}


# ------------------------------------------------------------------ substrate
def run_substrate(rng, R, T, dt):
    """Returns per-run arrays: gen (0/1), sign (+-1), pair id (0..2), time; plus counts."""
    X = rng.uniform(-3, 3, (R, N, 2)); th = rng.uniform(0, 2 * np.pi, (R, N))
    order = [list(np.argsort(X[r, :, 0])) for r in range(R)]
    ev = [[[], [], [], []] for _ in range(R)]
    unresolved = 0; multi = 0
    t = 0.0
    for step in range(int(T / dt)):
        X0 = X.copy()
        V = C * np.stack([np.cos(th), np.sin(th)], 2)
        X += (V - K_TRAP * X) * dt
        hit = rng.random((R, N)) < GAMMA * dt
        if hit.any():
            th[hit] += rng.uniform(-np.pi, np.pi, hit.sum())
        d0 = np.stack([X0[:, a, 0] - X0[:, b, 0] for a, b in PAIRS], 1)
        d1 = np.stack([X[:, a, 0] - X[:, b, 0] for a, b in PAIRS], 1)
        ch = np.sign(d0) != np.sign(d1)
        rows = np.flatnonzero(ch.any(1))
        for r in rows:
            cand = []
            for p in np.flatnonzero(ch[r]):
                f = d0[r, p] / (d0[r, p] - d1[r, p])
                cand.append((f, p))
            cand.sort()
            multi += len(cand) > 1
            for f, p in cand:
                a, b = PAIRS[p]
                o = order[r]; ia, ib = o.index(a), o.index(b)
                if abs(ia - ib) != 1:
                    unresolved += 1
                    o[ia], o[ib] = o[ib], o[ia]
                    continue
                k = min(ia, ib)
                o[ia], o[ib] = o[ib], o[ia]          # swap: now o[k] is the new left defect
                left, right = o[k], o[k + 1]
                yl = X0[r, left, 1] + f * (X[r, left, 1] - X0[r, left, 1])
                yr = X0[r, right, 1] + f * (X[r, right, 1] - X0[r, right, 1])
                e = ev[r]
                e[0].append(k); e[1].append(1 if yl > yr else -1); e[2].append(p); e[3].append(t + f * dt)
        t += dt
    out = [(np.array(e[0], np.int8), np.array(e[1], np.int8), np.array(e[2], np.int8), np.array(e[3]))
           for e in ev]
    return out, unresolved, multi


# ------------------------------------------------------------------ statistics
def tokens(gen, sign):
    return gen.astype(np.int64) * 2 + (sign.astype(np.int64) + 1) // 2      # 0..3


def history_counts(runs, k, signs=None):
    """counts[h, i, s] with h the depth-k token history code (4^k), i next generator, s next sign."""
    cnt = np.zeros((4 ** k if k > 0 else 1, 2, 2), np.int64)
    for ri, (g, s, _, _) in enumerate(runs):
        s = s if signs is None else signs[ri]
        n = g.size
        if n <= k:
            continue
        tok = tokens(g, s)
        h = np.zeros(n - k, np.int64)
        for j in range(k):
            h = h * 4 + tok[j:n - k + j]
        gi = g[k:].astype(np.int64); si = (s[k:].astype(np.int64) + 1) // 2
        np.add.at(cnt, (h, gi, si), 1)
    return cnt


def lr_test(cnt, min_lr=50):
    """Histories with fewer than min_lr events (for a given i) are pooled into one cell, so that the
    chi-square approximation holds (expected counts >= ~25 per sign)."""
    G2 = 0.0; df = 0
    for i in range(2):
        c0 = cnt[:, i, :].astype(float); tot0 = c0.sum(1)
        big = tot0 >= min_lr; rare = (tot0 > 0) & ~big
        c = np.vstack([c0[big], c0[rare].sum(0, keepdims=True)]) if rare.any() else c0[big]
        tot = c.sum(1); seen = tot > 0
        if seen.sum() < 2:
            continue
        marg = c[seen].sum(0) / c[seen].sum()
        ph = c[seen] / tot[seen, None]
        m = c[seen] > 0
        G2 += 2 * float(np.sum(c[seen][m] * np.log(ph[m] / np.broadcast_to(marg, ph.shape)[m])))
        df += int(seen.sum()) - 1
    return G2, df, float(chi2.sf(G2, df)) if df > 0 else float("nan")


def predictability(cnt):
    n = cnt.sum(2).astype(float); m = (cnt[:, :, 1] - cnt[:, :, 0]) / np.maximum(n, 1)
    tot = n.sum()
    big = n >= MIN_CELL
    return float((n * np.abs(m)).sum() / tot), float(np.abs(m[big]).max()) if big.any() else float("nan")


def sign_correlations(runs, signs=None):
    same = np.zeros((LMAX + 1, 2)); diff = np.zeros((LMAX + 1, 2))
    gaps = [0, 0.25, 0.5, 1, 2, 5, 1e9]; gap_acc = np.zeros((len(gaps) - 1, 2))
    for ri, (g, s, p, tm) in enumerate(runs):
        s = (s if signs is None else signs[ri]).astype(int); p = p.astype(int); n = s.size
        for l in range(1, LMAX + 1):
            if n <= l:
                break
            prod = s[:-l] * s[l:]; sp = p[:-l] == p[l:]
            same[l] += (prod[sp].sum(), sp.sum()); diff[l] += (prod[~sp].sum(), (~sp).sum())
        # consecutive crossings of the same defect pair, by time gap
        for q in range(3):
            idx = np.flatnonzero(p == q)
            if idx.size < 2:
                continue
            prod = s[idx[1:]] * s[idx[:-1]]; gap = tm[idx[1:]] - tm[idx[:-1]]
            for b in range(len(gaps) - 1):
                sel = (gap >= gaps[b]) & (gap < gaps[b + 1])
                gap_acc[b] += (prod[sel].sum(), sel.sum())
    return same, diff, gaps, gap_acc


def mean_ci(sumv, n):
    m = sumv / max(n, 1); se = math.sqrt(max(1 - m * m, 1e-12) / max(n, 1))
    return m, m - Z99 * se, m + Z99 * se


def analyse(runs, rng, label, signs=None, boot=True):
    res = {}
    allg = np.concatenate([r[0] for r in runs]).astype(int)
    alls = np.concatenate([r[1] if signs is None else signs[i] for i, r in enumerate(runs)]).astype(int)
    for i in range(2):
        sel = allg == i; k_ = int((alls[sel] == 1).sum()); n_ = int(sel.sum())
        p = k_ / n_; se = math.sqrt(p * (1 - p) / n_)
        res[f"P(s=+1|i={i})"] = (p, p - Z99 * se, p + Z99 * se, n_)
    res["k"] = {}
    groups = np.array_split(np.arange(len(runs)), N_GROUPS)
    for k in range(1, KMAX + 1):
        cnt = history_counts(runs, k, signs)
        G2, df, pval = lr_test(cnt); Pi, mx = predictability(cnt)
        row = dict(G2=G2, df=df, p=pval, Pi=Pi, max_abs_cell=mx)
        if boot:
            gc = [history_counts([runs[j] for j in gidx], k, None if signs is None else [signs[j] for j in gidx])
                  for gidx in groups]
            bs = []
            for _ in range(B_BOOT):
                pick = rng.integers(0, N_GROUPS, N_GROUPS)
                bs.append(predictability(sum(gc[q] for q in pick))[0])
            row["Pi_ci"] = (float(np.percentile(bs, 0.5)), float(np.percentile(bs, 99.5)))
        res["k"][k] = row
    return res


def main():
    here = pathlib.Path(__file__).resolve()
    digest = hashlib.sha256(here.read_bytes()).hexdigest()
    if "--hash" in sys.argv:
        print(digest); return
    t0 = time.time()
    print("=" * 104)
    print(f"Part V Stage 1: braid-generator sign | main dt={DT_MAIN}, runs={R_MAIN} x T={T_MAIN:g} "
          f"{'(QUICK)' if QUICK else ''}\nscript sha256 {digest}")
    print("=" * 104)
    ss = np.random.SeedSequence(SEED).spawn(8)
    runs, unres, multi = run_substrate(np.random.default_rng(ss[0]), R_MAIN, T_MAIN, DT_MAIN)
    ncross = sum(r[0].size for r in runs)
    print(f"\nSubstrate: {ncross} crossings, steps with >1 crossing {multi}, non-adjacent swaps (unresolved) "
          f"{unres} ({unres / max(ncross, 1):.1e})  [{time.time() - t0:.0f} s]")
    rng = np.random.default_rng(ss[1])

    # ---------------- real signs
    real = analyse(runs, rng, "real")
    print("\nM1 marginal balance:")
    for i in range(2):
        p, lo, hi, n = real[f"P(s=+1|i={i})"]
        print(f"   P(s=+1 | sigma_{i + 1}) = {p:.4f} [{lo:.4f},{hi:.4f}]  (n={n})")
    print("\nM2 conditional fairness given the (generator, sign) history:")
    print(f"   {'k':>2} {'G2':>12} {'df':>7} {'p-value':>10} {'Pi_k':>8} {'99% interval':>18} {'max|E[s|h,i]|':>14}")
    for k, r in real["k"].items():
        print(f"   {k:>2} {r['G2']:12.1f} {r['df']:7d} {r['p']:10.2e} {r['Pi']:8.4f} "
              f"[{r['Pi_ci'][0]:.4f},{r['Pi_ci'][1]:.4f}] {r['max_abs_cell']:14.4f}")

    # ---------------- controls on the same generator sequence
    rc = np.random.default_rng(ss[2])
    coin = [rc.choice(np.array([-1, 1], np.int8), r[0].size) for r in runs]
    anti = []
    for r in runs:
        p = r[2].astype(int); s = rc.choice([-1, 1], p.size); last = {}
        for n_ in range(p.size):
            q = p[n_]
            if q in last and rc.random() < 0.6:
                s[n_] = -last[q]
            last[q] = s[n_]
        anti.append(s.astype(np.int8))
    c1 = analyse(runs, rng, "coin", coin)
    c2 = analyse(runs, rng, "anti", anti, boot=False)
    # a depth with no testable cells (p = nan, too little data) is reported, not counted
    c1_ok = all(r["p"] > 0.01 for r in c1["k"].values() if np.isfinite(r["p"]))
    c2_ok = c2["k"][1]["p"] < 0.01
    print("\nControls (same generator sequence):")
    print("   C1 fair coin:   p-values " + ", ".join(f"k={k}: {r['p']:.2f}" for k, r in c1["k"].items())
          + f"  -> {'PASS' if c1_ok else 'FAIL'}")
    print("      fair-coin Pi_k (finite-sample floor): " + ", ".join(f"k={k}: {r['Pi']:.4f}" for k, r in c1["k"].items()))
    print(f"   C2 anti-correlated: p(k=1) = {c2['k'][1]['p']:.1e}  -> {'PASS' if c2_ok else 'FAIL'}")

    # ---------------- correlations
    same, diff, gaps, gap_acc = sign_correlations(runs)
    print("\nM3 sign correlation E[s_n s_{n+l}] (99% interval), same defect pair vs different pairs:")
    m3 = {}
    for l in (1, 2, 3, 5, 10, 20, 30, 60):
        ms = mean_ci(*same[l]); md = mean_ci(*diff[l]); m3[l] = (ms, md)
        print(f"   l={l:2d}: same pair {ms[0]:+.4f} [{ms[1]:+.4f},{ms[2]:+.4f}] (n={int(same[l][1])})   "
              f"different {md[0]:+.4f} [{md[1]:+.4f},{md[2]:+.4f}]")
    print("   consecutive crossings of the SAME defect pair, by time gap:")
    gap_rows = []
    for b in range(len(gaps) - 1):
        m = mean_ci(*gap_acc[b]); gap_rows.append((gaps[b], gaps[b + 1], m, int(gap_acc[b][1])))
        print(f"     gap [{gaps[b]:g},{gaps[b + 1]:g}): {m[0]:+.4f} [{m[1]:+.4f},{m[2]:+.4f}] (n={int(gap_acc[b][1])})")

    allg = np.concatenate([r[0] for r in runs]).astype(int)
    pi = np.bincount(allg, minlength=2) / allg.size; base = float((pi ** 2).sum())
    num = np.zeros(LMAX + 1); den = np.zeros(LMAX + 1)
    for r in runs:
        g = r[0].astype(int)
        for l in range(1, LMAX + 1):
            if g.size <= l:
                break
            num[l] += (g[:-l] == g[l:]).sum(); den[l] += g.size - l
    gac = [float(num[l] / den[l] - base) for l in range(1, LMAX + 1)]
    print("\nM4 generator autocorrelation P(i_{n+l} = i_n) - sum pi^2: " +
          ", ".join(f"l={l}: {gac[l - 1]:+.4f}" for l in (1, 2, 3, 5, 10, 30)))

    # ---------------- time-step check
    print("\nM5 time-step check (smaller samples):")
    dtrows = {}
    for di, dt in enumerate((0.02, 0.01, 0.005)):
        rr, u, _ = run_substrate(np.random.default_rng(ss[3 + di]), R_DT, T_DT, dt)
        a = analyse(rr, rng, f"dt{dt}", boot=False)
        sm, _, _, ga = sign_correlations(rr)
        m1 = mean_ci(*sm[1]); g0 = mean_ci(*ga[0])
        dtrows[str(dt)] = dict(P_plus=[a[f"P(s=+1|i={i})"][0] for i in range(2)], p_k1=a["k"][1]["p"],
                               Pi_k1=a["k"][1]["Pi"], same_pair_l1=m1, same_pair_gap_lt_025=g0,
                               unresolved=u, crossings=int(sum(x[0].size for x in rr)))
        print(f"   dt={dt:<6} crossings={dtrows[str(dt)]['crossings']:8d} unresolved={u:5d}  "
              f"P(s=+1)={np.round(dtrows[str(dt)]['P_plus'], 4).tolist()}  Pi_1={a['k'][1]['Pi']:.4f}  "
              f"same-pair l=1: {m1[0]:+.4f}  same-pair gap<0.25: {g0[0]:+.4f}")

    # ---------------- decision
    p1 = all(row[2][2] < 0 for row in gap_rows[:3] if row[3] > 0)
    rejects = [k for k, r in real["k"].items() if r["p"] < 0.01]
    valid = c1_ok and c2_ok
    print("\n" + "=" * 104)
    print("DECISION (computed):")
    print(f"  [{'CONFIRMED' if p1 else 'NOT CONFIRMED'}] P1 quick same-pair recrossings carry opposite signs (gap < 1)")
    print(f"  [{'CONFIRMED' if 1 in rejects else 'NOT CONFIRMED'}] P2 the sign is not conditionally fair at k = 1")
    if not valid:
        print("  RUN INVALID: a control failed." if not QUICK else "  QUICK mode: code check only.")
    elif rejects:
        print(f"  NOT FAIR: history dependence detected at k = {rejects}. Stage 2 = Part IV-style prediction of the bias.")
    else:
        print("  CANDIDATE: no history dependence detected at k <= 8. Stage 2 = carrier test.")
    print("=" * 104)
    out = dict(sha256=digest, dt=DT_MAIN, runs=R_MAIN, T=T_MAIN, crossings=ncross, unresolved=unres,
               real=real, control_coin=c1, control_anti={"k1": c2["k"][1]}, M3=m3, M3_gaps=gap_rows,
               M4=gac, M5=dtrows, valid=bool(valid), P1=bool(p1), rejects=rejects)
    pf = here.with_name("stage1_results.json")
    json.dump(out, open(pf, "w"), indent=1, default=float)
    print(f"Results: {pf}  sha256 {hashlib.sha256(pf.read_bytes()).hexdigest()}\nWall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()