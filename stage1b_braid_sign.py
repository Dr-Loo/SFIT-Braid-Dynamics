"""
stage1b_braid_sign.py -- SFIT Part V, Stage 1b (substrate only; no carrier, no populations).
Is the Part I planar braid-generator sign conditionally fair given the event history?

WHY 1b. Stage 1 (stage1_braid_sign.py, sha256 b895238d... as run) was INVALID: its fair-coin control
was rejected at k = 4, 5, 6, 8. Calibration afterwards (60 independent fair-coin draws on 3.5e6 real
substrate crossings) showed the asymptotic chi-square p-value of the pooled G2 test is anti-conservative
at depth: share of draws with p <= 0.05 was 2-3% at k = 1-4, 10% at k = 5, 22% at k = 7, 15% at k = 8;
the mean p at k = 7 was 0.25. In addition Stage 1 required p > 0.01 at eight correlated depths with no
multiplicity correction. Stage 1b changes ONLY the inference and adds one mechanism check; the substrate,
extractor, seed and sample sizes are unchanged, so the main run must reproduce Stage 1's crossing count
(8718597) exactly.

DYNAMICS (Part I v6 planar substrate, unchanged; identical to Stage 1)
  N = 3 defects, speed c = 2, Poisson heading kicks at rate GAMMA = 2 (uniform in [-pi, pi]), harmonic
  trap K_TRAP = 0.1, Euler step dt. Crossings within a step are ordered by f = d0 / (d0 - d1); y is
  interpolated at f. Generator i = rank k after the swap (sigma_1, sigma_2); sign s = +1 if the defect
  now on the LEFT is higher in y. Non-adjacent swaps are counted as unresolved.

FILTRATION (unchanged): the (generator, sign) token history. Fairness at depth k:
  E[ s_n | tokens n-k..n-1, i_n ] = 0.

INFERENCE (revised)
  M2  For k = 1..8 the pooled G2 statistic of Stage 1 (histories seen < 50 times per i pooled).
      p-values are MONTE CARLO: G2 is recomputed for B = 999 fair-coin sign sequences on the observed
      generator sequence, p_MC = (1 + #{G2_null >= G2_obs}) / (B + 1). Holm correction across the eight
      depths at alpha = 0.01. (With B = 999 the smallest p_MC is 0.001 <= 0.01/8, so every depth can reject.)
      The null conditions on the observed generator sequence; it is exact if generators do not depend on
      past signs, approximate otherwise. The chi-square p-value is still printed, with its false-alarm rate
      on the null replicates, to document the Stage 1 miscalibration.
      Predictability, two forms:
        Pi_k      plug-in sum n(h,i)|E[s|h,i]|/n, printed with the fair-coin floor (mean over null
                  replicates) and the excess Pi_k - floor. No bootstrap interval (Stage 1's percentile
                  interval excluded its own point estimate at k = 7, 8: finite-sample overfitting).
        Pi_hold   out-of-sample: runs split into 50 groups; the sign of E[s|h,i] is estimated on the even
                  groups and scored as mean s * sign_hat on the odd groups, and vice versa. Zero in
                  expectation for a fair sign at any sample size; 99% interval from the spread over the
                  50 groups (t_49).
  M6  Mechanism check (new). For two consecutive crossings of the same defect pair (a < b by label), let
      u = +1 if a is higher in y at the crossing. Between them the pair's x-order flips exactly once, so
      s = L u with L flipping:  s_n s_m = - u_n u_m  for every such pair of crossings. With q the
      probability that the pair's y-order changed during the gap,  E[s_n s_m] = -(1 - 2q).
      Given C3 this is an identity, not a fit: the measured physics is q(gap).
      On the ring (Part IV) there is no y, so q = 0 and the product is -1 at every gap. Reported by gap bin.

MEASUREMENTS kept from Stage 1: M1 marginal balance; M3 sign correlations (same pair vs different pairs,
  and by gap); M4 generator autocorrelation; M5 time step dt = 0.02, 0.01, 0.005.

CONTROLS (failure invalidates the run)
  C1  an independent held-out fair-coin sequence (not among the B null replicates) through the same
      Monte Carlo + Holm pipeline: must not reject at any k.
  C2  anti-correlated synthetic signs (s_n = -s_prev(same pair) w.p. 0.6): must reject at k = 1 (Holm).
  C3  M6 identity: zero violations of s_n s_m = -u_n u_m.
  C4  unresolved (non-adjacent) swaps below 1e-4 of crossings.

PREDICTIONS (written before running)
  P1  E[s s' | same pair, consecutive, gap < 1] < 0 (99% interval below 0)        [as Stage 1]
  P2  The sign is not conditionally fair at k = 1 (Holm, Monte Carlo)              [as Stage 1]
  P3  The y-order flip probability q increases monotonically across the six gap bins.
  P4  Pi_hold at k = 1 has its 99% interval above 0 (the history predicts the sign out of sample).
  Reproducibility: the main run has exactly 8718597 crossings (same seed and substrate as Stage 1).

DECISION
  INVALID     if any of C1-C4 fails.
  NOT FAIR    if valid and Holm rejects at some k (listed) -> Stage 2 = prediction of the record bias
              as a function of the y-mixing rate.
  CANDIDATE   if valid and no depth rejects -> Stage 2 = carrier test.

Usage: python stage1b_braid_sign.py [--quick] [--hash]
--quick is a code check only (small sample; same B = 999). Commit stage1b_results.json before any Stage 2 run.
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np
from scipy.stats import chi2, t as tdist

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
MIN_LR = 50
LMAX = 60
B_NULL = 999
ALPHA = 0.01
SEED = 20261120
Z99 = 2.5758
EXPECTED_CROSSINGS = 8718597
PAIRS = [(0, 1), (0, 2), (1, 2)]
GAPS = [0, 0.25, 0.5, 1, 2, 5, 1e9]


# ------------------------------------------------------------------ substrate (Stage 1 + y-order u)
def run_substrate(rng, R, T, dt):
    """Per-run arrays: gen (0/1), sign (+-1), pair id (0..2), time, u (+1 if lower label is higher)."""
    X = rng.uniform(-3, 3, (R, N, 2)); th = rng.uniform(0, 2 * np.pi, (R, N))
    order = [list(np.argsort(X[r, :, 0])) for r in range(R)]
    ev = [[[], [], [], [], []] for _ in range(R)]
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
                o[ia], o[ib] = o[ib], o[ia]
                left, right = o[k], o[k + 1]
                yl = X0[r, left, 1] + f * (X[r, left, 1] - X0[r, left, 1])
                yr = X0[r, right, 1] + f * (X[r, right, 1] - X0[r, right, 1])
                ya = yl if left == a else yr; yb = yr if left == a else yl
                e = ev[r]
                e[0].append(k); e[1].append(1 if yl > yr else -1); e[2].append(p); e[3].append(t + f * dt)
                e[4].append(1 if ya > yb else -1)
        t += dt
    out = [(np.array(e[0], np.int8), np.array(e[1], np.int8), np.array(e[2], np.int8), np.array(e[3]),
            np.array(e[4], np.int8)) for e in ev]
    return out, unresolved, multi


# ------------------------------------------------------------------ fast history counting
class Seq:
    """All runs concatenated; position-in-run lets depth-k histories stay inside one run."""
    def __init__(self, runs):
        self.G = np.concatenate([r[0] for r in runs]).astype(np.int64)
        self.S = np.concatenate([r[1] for r in runs]).astype(np.int64)
        lens = np.array([r[0].size for r in runs])
        self.pos = np.concatenate([np.arange(n) for n in lens])
        self.run = np.repeat(np.arange(len(runs)), lens)
        self.group = self.run % N_GROUPS

    def cells(self, S=None):
        """Yields (k, valid mask, cell index (h, i) code) for k = 1..KMAX given signs S."""
        S = self.S if S is None else S
        tok = self.G * 2 + (S + 1) // 2
        H = np.zeros_like(tok)
        for k in range(1, KMAX + 1):
            Hn = np.empty_like(H); Hn[0] = 0
            Hn[1:] = H[:-1] * 4 + tok[:-1]
            H = Hn
            valid = self.pos >= k
            yield k, valid, H * 2 + self.G

    def counts(self, S=None):
        S = self.S if S is None else S
        sb = (S + 1) // 2
        out = {}
        for k, valid, cell in self.cells(S):
            idx = cell[valid] * 2 + sb[valid]
            out[k] = np.bincount(idx, minlength=4 ** k * 4).reshape(4 ** k, 2, 2)
        return out


def lr_test(cnt, min_lr=MIN_LR):
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
    big = n >= MIN_CELL
    return float((n * np.abs(m)).sum() / n.sum()), float(np.abs(m[big]).max()) if big.any() else float("nan")


def pi_hold(seq, S=None):
    """Two-fold cross-fitted out-of-sample predictability per k: (mean, lo99, hi99)."""
    S = seq.S if S is None else S
    half = seq.group % 2
    res = {}
    for k, valid, cell in seq.cells(S):
        gsum = np.zeros(N_GROUPS); gn = np.zeros(N_GROUPS)
        for train in (0, 1):
            tr = valid & (half == train); te = valid & (half != train)
            ncell = 4 ** k * 2
            bal = np.bincount(cell[tr], weights=S[tr], minlength=ncell)
            score = S[te] * np.sign(bal)[cell[te]]
            gsum += np.bincount(seq.group[te], weights=score, minlength=N_GROUPS)
            gn += np.bincount(seq.group[te], minlength=N_GROUPS)
        ok = gn > 0; gm = gsum[ok] / gn[ok]
        mean = float(gsum.sum() / gn.sum()); se = float(gm.std(ddof=1) / math.sqrt(ok.sum()))
        q = float(tdist.ppf(0.995, ok.sum() - 1))
        res[k] = (mean, mean - q * se, mean + q * se)
    return res


def holm(pvals, alpha=ALPHA):
    ks = sorted(pvals, key=lambda k: pvals[k]); rej = []
    for j, k in enumerate(ks):
        if pvals[k] <= alpha / (len(ks) - j):
            rej.append(k)
        else:
            break
    return sorted(rej)


# ------------------------------------------------------------------ correlations and M6
def sign_correlations(runs, signs=None):
    same = np.zeros((LMAX + 1, 2)); diff = np.zeros((LMAX + 1, 2))
    gap_acc = np.zeros((len(GAPS) - 1, 2)); flip = np.zeros((len(GAPS) - 1, 2)); viol = 0
    for ri, r in enumerate(runs):
        g, s, p, tm, u = r
        s = (s if signs is None else signs[ri]).astype(int); p = p.astype(int); u = u.astype(int); n = s.size
        for l in range(1, LMAX + 1):
            if n <= l:
                break
            prod = s[:-l] * s[l:]; sp = p[:-l] == p[l:]
            same[l] += (prod[sp].sum(), sp.sum()); diff[l] += (prod[~sp].sum(), (~sp).sum())
        for q in range(3):
            idx = np.flatnonzero(p == q)
            if idx.size < 2:
                continue
            prod = s[idx[1:]] * s[idx[:-1]]; uu = u[idx[1:]] * u[idx[:-1]]
            gap = tm[idx[1:]] - tm[idx[:-1]]
            if signs is None:
                viol += int((prod != -uu).sum())
            for b in range(len(GAPS) - 1):
                sel = (gap >= GAPS[b]) & (gap < GAPS[b + 1])
                gap_acc[b] += (prod[sel].sum(), sel.sum())
                flip[b] += ((uu[sel] < 0).sum(), sel.sum())
    return same, diff, gap_acc, flip, viol


def mean_ci(sumv, n):
    m = sumv / max(n, 1); se = math.sqrt(max(1 - m * m, 1e-12) / max(n, 1))
    return m, m - Z99 * se, m + Z99 * se


def marginal(seq, S=None):
    S = seq.S if S is None else S
    out = []
    for i in range(2):
        sel = seq.G == i; n_ = int(sel.sum()); p = float((S[sel] == 1).sum() / n_)
        se = math.sqrt(p * (1 - p) / n_); out.append((p, p - Z99 * se, p + Z99 * se, n_))
    return out


def stats_of(seq, S=None):
    cnt = seq.counts(S)
    return {k: (*lr_test(c), *predictability(c)) for k, c in cnt.items()}


# ------------------------------------------------------------------ main
def main():
    here = pathlib.Path(__file__).resolve()
    digest = hashlib.sha256(here.read_bytes()).hexdigest()
    if "--hash" in sys.argv:
        print(digest); return
    t0 = time.time()
    bar = "=" * 108
    print(bar)
    print(f"Part V Stage 1b: braid-generator sign | main dt={DT_MAIN}, runs={R_MAIN} x T={T_MAIN:g}, "
          f"B_null={B_NULL} {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}")
    print(bar)
    ss = np.random.SeedSequence(SEED).spawn(8)
    runs, unres, multi = run_substrate(np.random.default_rng(ss[0]), R_MAIN, T_MAIN, DT_MAIN)
    seq = Seq(runs); ncross = seq.G.size
    repro = (ncross == EXPECTED_CROSSINGS) if not QUICK else None
    print(f"\nSubstrate: {ncross} crossings (Stage 1: {EXPECTED_CROSSINGS}; "
          f"{'reproduced' if repro else 'n/a in quick mode' if QUICK else 'NOT REPRODUCED'}), "
          f"steps with >1 crossing {multi}, unresolved {unres}  [{time.time() - t0:.0f} s]")

    # ---------------- real, controls, null replicates
    rc = np.random.default_rng(ss[2])
    coin = rc.choice(np.array([-1, 1]), ncross).astype(np.int64)
    anti_runs = []
    for r in runs:
        p = r[2].astype(int); s = rc.choice([-1, 1], p.size); last = {}
        for n_ in range(p.size):
            q = p[n_]
            if q in last and rc.random() < 0.6:
                s[n_] = -last[q]
            last[q] = s[n_]
        anti_runs.append(s)
    anti = np.concatenate(anti_runs).astype(np.int64)
    obs = {"real": stats_of(seq), "C1 coin": stats_of(seq, coin), "C2 anti": stats_of(seq, anti)}

    rn = np.random.default_rng(ss[6])
    null = np.zeros((B_NULL, KMAX + 1)); null_chi_p = np.zeros((B_NULL, KMAX + 1)); null_pi = np.zeros((B_NULL, KMAX + 1))
    tn = time.time()
    for b in range(B_NULL):
        Sb = rn.choice(np.array([-1, 1]), ncross).astype(np.int64)
        for k, (G2, df, p, Pi, _) in stats_of(seq, Sb).items():
            null[b, k] = G2; null_chi_p[b, k] = p; null_pi[b, k] = Pi
        if b in (9, 99) or (b + 1) % 250 == 0:
            el = time.time() - tn
            print(f"   null replicates {b + 1}/{B_NULL}  ({el:.0f} s, ~{el / (b + 1) * (B_NULL - b - 1):.0f} s left)", flush=True)

    def mc(label):
        return {k: (1 + int((null[:, k] >= obs[label][k][0]).sum())) / (B_NULL + 1) for k in range(1, KMAX + 1)}
    pmc = {lab: mc(lab) for lab in obs}
    rej = {lab: holm(pmc[lab]) for lab in obs}
    floor = {k: float(null_pi[:, k].mean()) for k in range(1, KMAX + 1)}
    chi_far = {k: float((null_chi_p[:, k] <= ALPHA).mean()) for k in range(1, KMAX + 1)}

    print("\nM1 marginal balance:")
    m1 = marginal(seq)
    for i, (p, lo, hi, n) in enumerate(m1):
        print(f"   P(s=+1 | sigma_{i + 1}) = {p:.4f} [{lo:.4f},{hi:.4f}]  (n={n})")

    ph = pi_hold(seq); ph_coin = pi_hold(seq, coin)
    print("\nM2 conditional fairness given the (generator, sign) history  (Monte Carlo p, Holm at alpha = 0.01):")
    print(f"   {'k':>2} {'G2':>12} {'df':>7} {'null G2 mean':>13} {'p_MC':>7} {'p_chi2':>9} {'chi2 FAR':>9}"
          f" {'Pi_k':>7} {'floor':>7} {'excess':>7}   {'Pi_hold [99%]':>26} {'max|E|':>7}")
    for k in range(1, KMAX + 1):
        G2, df, pc, Pi, mx = obs["real"][k]
        print(f"   {k:>2} {G2:12.1f} {df:7d} {null[:, k].mean():13.1f} {pmc['real'][k]:7.3f} {pc:9.2e} {chi_far[k]:9.3f}"
              f" {Pi:7.4f} {floor[k]:7.4f} {Pi - floor[k]:7.4f}   {ph[k][0]:.4f} [{ph[k][1]:.4f},{ph[k][2]:.4f}] {mx:7.4f}")
    print(f"   Holm rejects at k = {rej['real']}")
    print("   (chi2 FAR = share of null replicates the asymptotic chi-square would reject at 0.01; should be 0.01)")

    c1_ok = len(rej["C1 coin"]) == 0
    c2_ok = 1 in rej["C2 anti"]
    print("\nControls:")
    print("   C1 held-out fair coin: p_MC " + ", ".join(f"k={k}: {pmc['C1 coin'][k]:.3f}" for k in range(1, KMAX + 1))
          + f"  Holm rejects {rej['C1 coin']}  -> {'PASS' if c1_ok else 'FAIL'}")
    print("      coin Pi_hold: " + ", ".join(f"k={k}: {ph_coin[k][0]:+.4f}" for k in range(1, KMAX + 1)))
    print(f"   C2 anti-correlated: p_MC(k=1) = {pmc['C2 anti'][1]:.3f}  Holm rejects {rej['C2 anti']}  -> {'PASS' if c2_ok else 'FAIL'}")

    same, diff, gap_acc, flip, viol = sign_correlations(runs)
    c3_ok = viol == 0
    c4_ok = unres / max(ncross, 1) < 1e-4
    print(f"   C3 identity s_n s_m = -u_n u_m: {viol} violations  -> {'PASS' if c3_ok else 'FAIL'}")
    print(f"   C4 unresolved swaps {unres} ({unres / max(ncross, 1):.1e})  -> {'PASS' if c4_ok else 'FAIL'}")

    print("\nM3 sign correlation E[s_n s_{n+l}] (99% interval), same defect pair vs different pairs:")
    m3 = {}
    for l in (1, 2, 3, 5, 10, 20, 30, 60):
        ms = mean_ci(*same[l]); md = mean_ci(*diff[l]); m3[l] = (ms, md)
        print(f"   l={l:2d}: same pair {ms[0]:+.4f} [{ms[1]:+.4f},{ms[2]:+.4f}] (n={int(same[l][1])})   "
              f"different {md[0]:+.4f} [{md[1]:+.4f},{md[2]:+.4f}]")

    print("\nM6 consecutive crossings of the same defect pair: y-order flip probability q and E[s s'] (equals -(1 - 2q) exactly, given C3)")
    m6 = []
    for b in range(len(GAPS) - 1):
        n = int(gap_acc[b][1]); e = mean_ci(*gap_acc[b]); qv = flip[b][0] / max(n, 1)
        qse = math.sqrt(max(qv * (1 - qv), 1e-12) / max(n, 1))
        m6.append(dict(gap=(GAPS[b], GAPS[b + 1]), n=n, E=e, q=(qv, qv - Z99 * qse, qv + Z99 * qse), pred=-(1 - 2 * qv)))
        print(f"   gap [{GAPS[b]:g},{GAPS[b + 1]:g}): n={n:8d}  q={qv:.4f} [{qv - Z99 * qse:.4f},{qv + Z99 * qse:.4f}]"
              f"  E[s s']={e[0]:+.4f}")
    p3 = all(m6[b + 1]["q"][0] > m6[b]["q"][0] for b in range(len(m6) - 1))

    allg = seq.G; pi_ = np.bincount(allg, minlength=2) / allg.size; base = float((pi_ ** 2).sum())
    gac = {}
    for l in (1, 2, 3, 5, 10, 30):
        ok = seq.pos[l:] >= l
        gac[l] = float((allg[l:][ok] == allg[:-l][ok]).mean() - base)
    print("\nM4 generator autocorrelation P(i_{n+l} = i_n) - sum pi^2: " + ", ".join(f"l={l}: {v:+.4f}" for l, v in gac.items()))

    print("\nM5 time-step check (smaller samples; Pi_hold at k = 1):")
    dtrows = {}
    for di, dt in enumerate((0.02, 0.01, 0.005)):
        rr, u, _ = run_substrate(np.random.default_rng(ss[3 + di]), R_DT, T_DT, dt)
        sq = Seq(rr); h1 = pi_hold(sq)[1]
        sm, _, ga, fl, _ = sign_correlations(rr)
        l1 = mean_ci(*sm[1]); g0 = mean_ci(*ga[0]); q0 = fl[0][0] / max(fl[0][1], 1)
        dtrows[str(dt)] = dict(crossings=int(sq.G.size), unresolved=u, P_plus=[x[0] for x in marginal(sq)],
                               Pi_hold_k1=h1, same_pair_l1=l1, gap_lt_025=g0, q_gap_lt_025=q0)
        print(f"   dt={dt:<6} crossings={sq.G.size:8d} unresolved={u:4d}  P(s=+1)={[round(x[0], 4) for x in marginal(sq)]}"
              f"  Pi_hold_1={h1[0]:.4f}  same-pair l=1: {l1[0]:+.4f}  gap<0.25: {g0[0]:+.4f} (q={q0:.4f})")

    # ---------------- decision
    p1 = all(r["E"][2] < 0 for r in m6[:3] if r["n"] > 0)
    p2 = 1 in rej["real"]
    p4 = ph[1][1] > 0
    valid = c1_ok and c2_ok and c3_ok and c4_ok
    print("\n" + bar)
    print("DECISION (computed):")
    if not QUICK:
        print(f"  [{'OK' if repro else 'FAILED'}] reproducibility: {ncross} crossings vs Stage 1 {EXPECTED_CROSSINGS}")
    for tag, ok, txt in (("P1", p1, "quick same-pair recrossings carry opposite signs (gap < 1)"),
                         ("P2", p2, "the sign is not conditionally fair at k = 1 (Monte Carlo, Holm)"),
                         ("P3", p3, "y-order flip probability q increases across the gap bins"),
                         ("P4", p4, "Pi_hold at k = 1 has its 99% interval above 0")):
        print(f"  [{'CONFIRMED' if ok else 'NOT CONFIRMED'}] {tag} {txt}")
    if QUICK:
        print("  QUICK mode: code check only.")
    elif not valid:
        print("  RUN INVALID: a control failed (" + ", ".join(n for n, ok in (("C1", c1_ok), ("C2", c2_ok), ("C3", c3_ok), ("C4", c4_ok)) if not ok) + ").")
    elif rej["real"]:
        print(f"  NOT FAIR: history dependence at k = {rej['real']}. Stage 2 = record bias vs y-mixing rate.")
    else:
        print("  CANDIDATE: no history dependence detected at k <= 8. Stage 2 = carrier test.")
    print(bar)
    out = dict(sha256=digest, quick=QUICK, dt=DT_MAIN, runs=R_MAIN, T=T_MAIN, B_null=B_NULL, crossings=ncross,
               reproduced=repro, unresolved=unres, M1=m1,
               M2={k: dict(G2=obs["real"][k][0], df=obs["real"][k][1], p_chi2=obs["real"][k][2], p_MC=pmc["real"][k],
                           null_G2_mean=float(null[:, k].mean()), null_G2_q99=float(np.quantile(null[:, k], 0.99)),
                           chi2_false_alarm=chi_far[k], Pi=obs["real"][k][3], Pi_floor=floor[k],
                           Pi_hold=ph[k], max_abs_cell=obs["real"][k][4]) for k in range(1, KMAX + 1)},
               holm_rejects=rej, p_MC=pmc, C1_Pi_hold=ph_coin, C3_violations=viol, M3=m3, M6=m6, M4=gac, M5=dtrows,
               valid=bool(valid), P1=bool(p1), P2=bool(p2), P3=bool(p3), P4=bool(p4))
    pf = here.with_name("stage1b_results_quick.json" if QUICK else "stage1b_results.json")
    json.dump(out, open(pf, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(f"Results: {pf}  sha256 {hashlib.sha256(pf.read_bytes()).hexdigest()}\nWall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()