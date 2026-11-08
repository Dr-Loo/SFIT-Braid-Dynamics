"""
stage3_core.py -- shared code for SFIT Part V Stage 3 (imported by stage3_predict.py and stage3_validate.py;
both print this file's sha256). Identical to stage2_core.py except that the trap strength K is a parameter
and the initial box scales with the trap length: positions start uniform in [-B, B]^2 with B = 0.15 C / K
(B = 3 at K = 0.1, as before). With K = 0.1 it reproduces stage2_core.py exactly (stage3_predict.py --selftest).
SCALING: with t' = K t and lengths in units of C / K the dynamics depend only on gamma / K (and K dt), and
crossings, signs and over-strand labels are invariant under the length rescaling. So for N = 3 with uniform
heading kicks, Part I has ONE physical parameter for this question: gamma / K.

SUBSTRATE: the Part I v6 planar substrate of Stage 1/1b, vectorised over runs, with the same random-number
consumption, so that with the carrier off it reproduces stage1b's event sequence exactly (checked by
stage2_predict.py --selftest). N = 3 defects, speed C = 2, heading kicks at rate gamma (uniform angle),
trap K_TRAP = 0.1, Euler step DT = 0.02; crossings in a step are applied in order of f = d0/(d0 - d1),
y interpolated at f.
MIXING KNOB (not in Part I; kappa = 0 is Part I exactly): at rate kappa per run, the three y-coordinates
of a run are permuted uniformly at random at the end of a step (x, headings untouched). This keeps the
set of y values and scrambles only the transverse ORDER. kappa -> infinity makes u a fair coin.

EVENT for a crossing of pair p = (a, b), a < b:
  s  braid sign (Stage 1): +1 if the defect now on the left is higher in y
  u  over-strand label: +1 if a is higher in y (a passes over b), -1 otherwise.   s = L u, L = +1 if a is left.

CARRIER JUMP (covariant form of the Part IV law): the over-crossing defect gains,
      x -> x + EPS * d_{over,under}(x) = x + EPS * u * d_ab(x),
      d_ab(x) = x_a (e_a - x) - x_b (e_b - x);  clip at 0, renormalise; record = first x_k >= H = 0.999.
  Why not x + EPS s d_ab: s does not change sign when the labels a, b are exchanged while d_ab does, so
  that law would depend on how the defects are numbered. u d_ab is label-independent. Equivalently the
  jump is s * d_{left,right}: the braid word (generator, sign) acting on the defects at those ranks.
  The populations are a martingale iff E[u_n | F_{n-1}, p_n] = 0.  For consecutive crossings of the same
  pair, u u' = -s s' (Stage 1b C3), so E[u u'] = 1 - 2q: the over-strand persists unless the pair's
  y-order flips in between.
"""
import numpy as np

N, C, K_TRAP, DT = 3, 2.0, 0.1, 0.02
H = 0.999
PAIRS = [(0, 1), (0, 2), (1, 2)]
PA = np.array([0, 0, 1]); PB = np.array([1, 2, 2])


def d_pair(x, p):
    r = np.arange(len(p)); a = PA[p]; b = PB[p]
    xa, xb = x[r, a], x[r, b]
    d = -(xa - xb)[:, None] * x
    d[r, a] += xa; d[r, b] -= xb
    return d


class Substrate:
    """R planar substrates stepped together. step() returns the crossings of this step as up to three
    ordered slots: list of (rows, pair, s, u, time)."""
    def __init__(self, rng, R, gamma, dt=DT, kappa=0.0, K=K_TRAP):
        self.rng, self.R, self.g, self.dt, self.kappa, self.K = rng, R, gamma, dt, kappa, K
        box = 3.0 * (K_TRAP / K)                     # = 0.15 C / K; exactly 3 at K = 0.1
        self.X = rng.uniform(-box, box, (R, N, 2)); self.th = rng.uniform(0, 2 * np.pi, (R, N)); self.t = 0.0

    def keep(self, m):
        self.X = self.X[m]; self.th = self.th[m]; self.R = int(m.sum())

    def step(self):
        rng, X, th, dt = self.rng, self.X, self.th, self.dt
        X0 = X.copy()
        V = C * np.stack([np.cos(th), np.sin(th)], 2)
        X += (V - self.K * X) * dt
        hit = rng.random((self.R, N)) < self.g * dt
        if hit.any():
            th[hit] += rng.uniform(-np.pi, np.pi, hit.sum())
        d0 = X0[:, PA, 0] - X0[:, PB, 0]; d1 = X[:, PA, 0] - X[:, PB, 0]
        ch = np.sign(d0) != np.sign(d1)
        out = []
        if ch.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where(ch, d0 / (d0 - d1), np.inf)
            order = np.argsort(f, 1, kind="stable"); fs = np.take_along_axis(f, order, 1)
            for j in range(3):
                rows = np.flatnonzero(np.isfinite(fs[:, j]))
                if rows.size == 0:
                    break
                p = order[rows, j]; fr = fs[rows, j]; a = PA[p]; b = PB[p]
                ya = X0[rows, a, 1] + fr * (X[rows, a, 1] - X0[rows, a, 1])
                yb = X0[rows, b, 1] + fr * (X[rows, b, 1] - X0[rows, b, 1])
                u = np.where(ya > yb, 1, -1)
                L = np.where(d1[rows, p] < 0, 1, -1)          # a is left after the swap
                out.append((rows, p, (L * u).astype(np.int8), u.astype(np.int8), self.t + fr * dt))
        if self.kappa > 0:
            sc = np.flatnonzero(rng.random(self.R) < self.kappa * dt)
            if sc.size:
                perm = np.argsort(rng.random((sc.size, N)), 1)
                self.X[sc, :, 1] = np.take_along_axis(self.X[sc, :, 1], perm, 1)
        self.t += dt
        return out


def events_fast(rng, R, T, gamma, dt=DT, kappa=0.0, K=K_TRAP):
    """Substrate only: per-run arrays (pair, s, u, time)."""
    S = Substrate(rng, R, gamma, dt, kappa, K)
    buf = {k: [] for k in ("r", "p", "s", "u", "t")}
    for _ in range(int(T / dt)):
        for rows, p, s, u, tm in S.step():
            buf["r"].append(rows); buf["p"].append(p); buf["s"].append(s); buf["u"].append(u); buf["t"].append(tm)
    cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in buf.items()}
    o = np.argsort(cat["r"], kind="stable")          # stable: keeps time order within a run
    r = cat["r"][o]; cuts = np.searchsorted(r, np.arange(R + 1))
    srt = {k: cat[k][o] for k in ("p", "s", "u", "t")}          # sort once (re-sorting per run exhausted memory)
    return [tuple(srt[k][cuts[i]:cuts[i + 1]].copy() for k in ("p", "s", "u", "t")) for i in range(R)]


def carrier(rng, R, gamma, eps, x0, kappa=0.0, coin_rng=None, max_T=2.0e5, dt=DT, K=K_TRAP):
    """Live carrier on the substrate. coin_rng given -> u replaced by a fair coin (control: Born weights).
    Returns record per run (-1 if not absorbed) and the number of crossings each run used."""
    S = Substrate(rng, R, gamma, dt, kappa, K)
    x = np.tile(np.asarray(x0, float), (R, 1)); rec = np.full(R, -1); ncr = np.zeros(R, np.int64)
    idx = np.arange(R)                       # original run index of each live substrate row
    live = np.ones(R, bool)                  # per live row
    for it in range(int(max_T / dt)):
        for rows, p, s, u, tm in S.step():
            m = live[rows]
            if not m.any():
                continue
            rows, p, u = rows[m], p[m], u[m]
            sg = u if coin_rng is None else coin_rng.choice(np.array([-1, 1]), rows.size)
            xr = x[rows] + (eps * sg)[:, None] * d_pair(x[rows], p)
            xr = np.clip(xr, 0, None); xr /= xr.sum(1, keepdims=True); x[rows] = xr
            ncr[idx[rows]] += 1
            hit = xr.max(1) >= H
            if hit.any():
                rec[idx[rows[hit]]] = xr[hit].argmax(1); live[rows[hit]] = False
        if not live.any():
            break
        if it % 500 == 499 and live.mean() < 0.7:          # drop finished runs (speed only)
            S.keep(live); x = x[live]; idx = idx[live]; live = live[live]
    return rec, ncr


# ------------------------------------------------------------------ token statistics and reduced model
A = 6                                         # token = pair * 2 + (u + 1) // 2


def tokens(p, u):
    return p.astype(np.int64) * 2 + (u.astype(np.int64) + 1) // 2


def transition_counts(runs, kmax):
    """counts[k] has shape (A**k, A): k-token history (oldest most significant) -> next token."""
    out = {k: np.zeros((A ** k, A), np.int64) for k in range(1, kmax + 1)}
    for p, s, u, t in runs:
        tok = tokens(p, u); n = tok.size; h = np.zeros(n, np.int64)
        for k in range(1, kmax + 1):
            if n <= k:
                break
            h = h[:-1] * A + tok[k - 1:n - 1] if k > 1 else tok[:n - 1].copy()
            np.add.at(out[k], (h, tok[k:]), 1)
    return out


def symmetrize(counts):
    """Average the counts with their y-reflected copies (every u -> -u in history and next token).
    y -> -y is an exact symmetry of the substrate, so E[u | p] = 0 exactly; without this, sampling noise in
    the u balance would act as a spurious drift in the reduced model."""
    out = {}
    for k, T in counts.items():
        h = np.arange(A ** k); hf = np.zeros_like(h); c = h.copy(); mult = 1
        for _ in range(k):
            hf += ((c % A) ^ 1) * mult; c //= A; mult *= A
        nf = np.arange(A) ^ 1
        out[k] = (T + T[hf][:, nf]) / 2.0
    return out


class TokenChain:
    """Order-k chain on tokens with fallback to lower order for unseen histories; stationary start."""
    def __init__(self, counts, k):
        self.k = k; self.P = {}
        for j in range(1, k + 1):
            T = counts[j].astype(float); n = T.sum(1, keepdims=True)
            self.P[j] = np.divide(np.cumsum(T, 1), n, out=np.full_like(T, np.nan), where=n > 0)
        hc = counts[k].sum(1).astype(float); self.h0 = hc / hc.sum()

    def start(self, rng, R):
        c = rng.choice(A ** self.k, R, p=self.h0); hist = np.zeros((R, self.k), np.int64)
        for j in range(self.k - 1, -1, -1):
            hist[:, j] = c % A; c //= A
        return hist

    def step(self, rng, hist):
        R = hist.shape[0]; out = np.empty(R, np.int64); u = rng.random(R); todo = np.ones(R, bool)
        for j in range(self.k, 0, -1):
            code = np.zeros(R, np.int64)
            for col in range(self.k - j, self.k):
                code = code * A + hist[:, col]
            cp = self.P[j][code]; ok = todo & np.isfinite(cp[:, 0])
            if ok.any():
                out[ok] = (u[ok][:, None] > cp[ok]).sum(1).clip(0, A - 1); todo &= ~ok
            if not todo.any():
                break
        hist = np.roll(hist, -1, 1); hist[:, -1] = out
        return out, hist


def reduced(counts, k, eps, x0, R, rng, max_events=10 ** 6):
    """Reduced model: tokens from the order-k chain (k = 0: i.i.d. with the measured frequencies),
    the same jump and stopping rule as carrier(). Returns record per run."""
    x = np.tile(np.asarray(x0, float), (R, 1)); rec = np.full(R, -1); live = np.ones(R, bool)
    if k > 0:
        ch = TokenChain(counts, k); hist = ch.start(rng, R)
    else:
        pi = counts[1].sum(0).astype(float); pi /= pi.sum()
    n = 0
    while n < max_events and live.any():
        if k > 0:
            tok, hist = ch.step(rng, hist)
        else:
            tok = rng.choice(A, R, p=pi)
        p = tok // 2; u = 2 * (tok % 2) - 1
        L = np.flatnonzero(live)
        xr = x[L] + (eps * u[L])[:, None] * d_pair(x[L], p[L])
        xr = np.clip(xr, 0, None); xr /= xr.sum(1, keepdims=True); x[L] = xr
        hit = xr.max(1) >= H
        rec[L[hit]] = xr[hit].argmax(1); live[L[hit]] = False
        n += 1
    return rec