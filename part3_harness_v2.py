"""
part3_harness_v2.py  --  SFIT Part III (rerun specification v2): record statistics from a single depleted carrier
                     coupled locally to explicit detectors.

PHYSICAL MODEL (everything that determines the outcome is listed here)
--------------------------------------------------------------------
One carrier mode with complex amplitude a (energy |a|^2, initially E0 = 1) and an internal
state psi (complex, unit norm). Each channel k has a LOCAL damped detector oscillator b_k.
Coupled-mode equations (resonant, rotating frame):

    da/dt   = -i sum_k conj(G_k) b_k
    db_k/dt = -i G_k a - (gamma/2) b_k
    dW_k/dt = gamma |b_k|^2            (energy absorbed by detector k)

with G_k = g * kappa_k * psi_k. The coupling is LINEAR in psi_k. Energy is conserved:
|a|^2 + sum|b_k|^2 + sum W_k = E0 (checked every run). No psi^2 is written anywhere; the
power law of absorption is measured (diagnostic D1).

Detector k fires when W_k crosses its threshold H_k (crossing time interpolated in the step).
LOCAL back-reaction: when detector k fires, ITS OWN coupling becomes kappa*g*psi_k
(kappa >= 1). Nothing else changes. No rule ends the run, blocks other detectors, or picks
a winner. Any exclusion must come from the shared, finite carrier.

Substrate noise: Poisson reversals (rate nu) flip the carrier phase, a -> -a (Part I's
velocity reversals acting on the single carrier).

Threshold laws (the open ingredient, varied, never tuned):
    'fixed'        H_k = theta
    'exponential'  H_k ~ Exp(mean theta)          -- ASSUMED memoryless law, not derived
    'gamma2'       H_k ~ Gamma(shape 2, mean theta)

METRICS (fixed before running)
-----------------------------
M1  P(k) = P(detector k fires first | at least one fires). Born test: upper 99% bootstrap
    bound of TV(P, |psi|^2) <= EPS.
M2  gated alpha: each run is one gate. P_i = P(i fires in run), P_ij = P(i and j fire in run),
    alpha = sum_{i!=j} P_ij / sum_{i!=j} P_i P_j. Independent detectors give 1.
    Exclusion test: upper 99% bootstrap bound of alpha < 1.
Outcomes (v2): A = Born + STRONG exclusion (alpha_hi < 0.1); A~ = Born + PARTIAL (alpha_hi < 1);
B = Born, no exclusion; C / C~ = strong / partial exclusion, not Born; D = neither.
A config is UNRESOLVED (not classified) if > 1% of runs end with carrier energy > DELTA.

CONTROLS (separate code paths, auto pass/fail; any failure invalidates the run)
IC-0  synthetic independent Bernoulli detectors              -> alpha CI contains 1
IC-1  fixed thresholds, no reversals, kappa=1                -> largest |psi_k| first in >99%
IC-2  psi-independent coupling (|psi_k| replaced by 1/sqrt n) -> P(k) uniform (TV_hi <= EPS)
IC-3  classical limit: kappa=1, exponential thresholds        -> alpha CI contains 1
IC-4  imposed draw k ~ |psi_k|^2, only k fires (no physics)   -> Born and alpha_hi < 1
IC-5  psi shuffled between channels, compared with ORIGINAL  -> TV_lo > EPS (mismatch seen)

PREDICTIONS (written before running; compared automatically at the end)
P1  exponential thresholds: Born for every kappa, nu, theta, state (common time change W_k ~ |psi_k|^2 U(t)).
P2  exponential: alpha = 1 (CI contains 1) at kappa = 1; alpha_hi < 1 for kappa >= 2, decreasing
    in kappa, at both theta. NO prediction is made for whether strong exclusion is reached.
P3  fixed thresholds: largest |psi_k| fires first in >99% of runs for EVERY nu and theta.
    (Reversals act on the shared carrier, so they are common-mode and cannot reorder channels.)
P4  gamma2 thresholds: not Born (TV_lo > EPS), biased toward the largest component.
D1  measured absorption exponent d ln W_k / d ln|psi_k| = 2 within 1e-3.

CHANGES FROM v1 (all fixed before the v2 run)
  * Simultaneous first crossings inside one step are re-integrated with 50 RK4 sub-steps
    to decide which detector fired first. Residual ambiguity (still simultaneous at the
    sub-step level) must stay below 0.1% of records, else the configuration is INVALID.
  * Two threshold scales THETA in {0.05, 0.3} for every law. THETA = 0.05 lies below the
    smallest |psi_k|^2 used (0.1), so every channel can fire under fixed thresholds and
    gated alpha is defined there.
  * Exclusion criterion stated in two grades: STRONG alpha_hi < 0.1 (the Part III blueprint
    criterion) and PARTIAL alpha_hi < 1. Outcomes: A (Born + strong), A~ (Born + partial),
    B (Born, no exclusion), C / C~ (exclusion strong / partial, not Born), D (neither).
  * Fixed-threshold alpha is reported; with fixed thresholds and W_k ~ |psi_k|^2 U(t) the
    firing ORDER is deterministic, so P(first) is never Born there (prediction P3).
  * --hash prints the script SHA-256 for the deposit. Nothing is written before running.

Usage: python part3_harness_v2.py [--quick] [--hash]
"""
import sys, json, time
import numpy as np

# ------------------------------------------------------------------ fixed parameters
G0, GAMMA, E0 = 1.0, 4.0, 1.0
THETAS = (0.05, 0.3)
N_SUB = 50
TIE_BUDGET = 1e-3
STRONG = 0.1
DT, T_MAX, DELTA = 0.01, 60.0, 1e-3
EPS, CONF, N_BOOT = 0.02, 0.99, 1000
BASE_SEED = 20261029
QUICK = "--quick" in sys.argv
N_RUNS = 4000 if QUICK else 20000

STATES = {
    "S2a": np.sqrt([0.2, 0.8]),
    "S3a": np.sqrt([0.5, 0.3, 0.2]),
    "S3c": np.sqrt([0.7, 0.2, 0.1]),
    "S3p": np.sqrt([0.5, 0.3, 0.2]) * np.exp(1j * np.array([0.0, 2.1, -0.7])),  # complex phases
    "S4a": np.sqrt([0.4, 0.3, 0.2, 0.1]),
}


# ------------------------------------------------------------------ thresholds
def draw_thresholds(law, rng, shape, THETA):
    if law == "fixed":
        return np.full(shape, THETA)
    if law == "exponential":
        return rng.exponential(THETA, shape)
    if law == "gamma2":
        return rng.gamma(2.0, THETA / 2.0, shape)
    raise ValueError(law)


# ------------------------------------------------------------------ dynamics
def resolve_substeps(a0, b0, W0, G0row, Hrow, ks, rhs):
    """Re-integrate one run over one step with N_SUB RK4 sub-steps; return the crossing
    time (fraction of DT) of each channel in ks, and whether the earliest two still fall
    in the same sub-step (residual ambiguity)."""
    h = DT / N_SUB
    a, b, W = np.array([a0]), b0[None, :].copy(), W0[None, :].copy()
    G = G0row[None, :]
    times = {}
    for j in range(N_SUB):
        k1 = rhs(a, b, G); k2 = rhs(a + .5*h*k1[0], b + .5*h*k1[1], G)
        k3 = rhs(a + .5*h*k2[0], b + .5*h*k2[1], G); k4 = rhs(a + h*k3[0], b + h*k3[1], G)
        Wn = W + h/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        for k in ks:
            if k not in times and Wn[0, k] >= Hrow[k]:
                fr = (Hrow[k] - W[0, k]) / max(Wn[0, k] - W[0, k], 1e-300)
                times[k] = (j + fr) / N_SUB
        a = a + h/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        b = b + h/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]); W = Wn
    for k in ks:                       # numerical safety: fall back to the step end
        times.setdefault(k, 1.0)
    ts = sorted(times.values())
    amb = len(ts) > 1 and int(ts[0] * N_SUB) == int(ts[1] * N_SUB)
    return times, amb


def simulate(psi_coupling, law, kappa, nu, n_runs, rng, theta=0.3):
    """Integrate the coupled-mode equations for n_runs independent runs (vectorised).
    psi_coupling: complex amplitudes used in the coupling G_k (normally psi itself)."""
    psi_c = np.asarray(psi_coupling, complex)
    n = psi_c.size
    a = np.ones(n_runs, complex) * np.sqrt(E0)
    b = np.zeros((n_runs, n), complex)
    W = np.zeros((n_runs, n))
    H = draw_thresholds(law, rng, (n_runs, n), theta)
    fired = np.zeros((n_runs, n), bool)
    t_fire = np.full((n_runs, n), np.inf)
    W_at_first = np.full((n_runs, n), np.nan)     # W vector at first firing (diagnostics)
    E_at_first = np.full(n_runs, np.nan)          # total absorbed energy at first firing
    ties = 0
    residual_ties = 0
    G = np.tile(G0 * psi_c, (n_runs, 1))

    def rhs(a, b, G):
        da = -1j * np.sum(np.conj(G) * b, axis=1)
        db = -1j * G * a[:, None] - 0.5 * GAMMA * b
        dW = GAMMA * np.abs(b) ** 2
        return da, db, dW

    t = 0.0
    live = np.ones(n_runs, bool)
    while t < T_MAX and live.any():
        idx = np.flatnonzero(live)
        a_, b_, W_, G_ = a[idx], b[idx], W[idx], G[idx]
        k1 = rhs(a_, b_, G_)
        k2 = rhs(a_ + 0.5 * DT * k1[0], b_ + 0.5 * DT * k1[1], G_)
        k3 = rhs(a_ + 0.5 * DT * k2[0], b_ + 0.5 * DT * k2[1], G_)
        k4 = rhs(a_ + DT * k3[0], b_ + DT * k3[1], G_)
        a_n = a_ + DT / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        b_n = b_ + DT / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        W_n = W_ + DT / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])

        # threshold crossings (interpolated within the step)
        cross = (~fired[idx]) & (W_n >= H[idx])
        if cross.any():
            frac = np.where(cross, (H[idx] - W_) / np.maximum(W_n - W_, 1e-300), np.inf)
            rows = np.flatnonzero(cross.any(1))
            for r in rows:
                gi = idx[r]
                ks = np.flatnonzero(cross[r])
                first_ever = not fired[gi].any()
                sub_t = None
                if first_ever and ks.size > 1:
                    ties += 1
                    sub_t, amb = resolve_substeps(a_[r], b_[r], W_[r], G_[r], H[gi], ks, rhs)
                    residual_ties += int(amb)
                if first_ever:
                    f = frac[r, ks].min() if sub_t is None else min(sub_t.values())
                    W_at_first[gi] = W_[r] + f * (W_n[r] - W_[r])
                    E_at_first[gi] = W_at_first[gi].sum()
                for k in ks:
                    fired[gi, k] = True
                    t_fire[gi, k] = t + (frac[r, k] if sub_t is None else sub_t[k]) * DT
                    G_[r, k] = kappa * G0 * psi_c[k]      # LOCAL back-reaction
            G[idx] = G_

        # Poisson reversals of the carrier (shared substrate noise)
        if nu > 0:
            flip = rng.random(idx.size) < nu * DT
            a_n = np.where(flip, -a_n, a_n)

        a[idx], b[idx], W[idx] = a_n, b_n, W_n
        remaining = np.abs(a_n) ** 2 + np.sum(np.abs(b_n) ** 2, axis=1)
        live[idx] = remaining > DELTA * E0
        t += DT

    remaining = np.abs(a) ** 2 + np.sum(np.abs(b) ** 2, axis=1)
    energy_err = np.max(np.abs(remaining + W.sum(1) - E0))
    first = np.where(fired.any(1), np.argmin(t_fire, axis=1), -1)
    return dict(fired=fired, first=first, residual_ties=residual_ties, unresolved=float(np.mean(remaining > DELTA * E0)),
                energy_err=float(energy_err), ties=ties, W_at_first=W_at_first,
                E_at_first=E_at_first, E_final=W.sum(1))


# ------------------------------------------------------------------ metrics
def first_freqs(first, n):
    ok = first >= 0
    return np.bincount(first[ok], minlength=n), int(ok.sum())


def tv(p, q):
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum()


def tv_ci(counts, n_rec, target, rng):
    p = counts / max(n_rec, 1)
    boot = rng.multinomial(n_rec, p, size=N_BOOT) / max(n_rec, 1)
    tvs = 0.5 * np.abs(boot - target).sum(1)
    lo, hi = np.percentile(tvs, [100 * (1 - CONF) / 2, 100 * (1 + CONF) / 2])
    return tv(p, target), lo, hi


def alpha_gated(F):
    F = F.astype(float)
    P = F.mean(0)
    Pij = (F.T @ F) / F.shape[0]
    off = ~np.eye(F.shape[1], dtype=bool)
    den = np.outer(P, P)[off].sum()
    return Pij[off].sum() / den if den > 0 else np.nan


def alpha_ci(F, rng):
    """alpha is undefined (nan) when fewer than two channels ever fire: no pair to correlate."""
    a0 = alpha_gated(F)
    if not np.isfinite(a0):
        return np.nan, np.nan, np.nan
    N = F.shape[0]
    vals = np.array([alpha_gated(F[rng.integers(0, N, N)]) for _ in range(N_BOOT)])
    vals = vals[np.isfinite(vals)]
    if vals.size < 0.9 * N_BOOT:
        return a0, np.nan, np.nan
    lo, hi = np.percentile(vals, [100 * (1 - CONF) / 2, 100 * (1 + CONF) / 2])
    return a0, lo, hi


def hazard_linearity(E_first, E_final, theta=0.3):
    """Cumulative hazard of the FIRST firing vs total absorbed energy E.
    Returns slope(first half)/slope(second half) on E in [0, min(0.8 E0, 3 theta)]; 1 = memoryless."""
    grid = np.linspace(0, min(0.8 * E0, 3 * theta), 41)   # v2: scaled to the threshold scale
    S = np.array([np.mean(~(E_first <= e)) for e in grid])   # nan (no fire) counts as survived
    S = np.clip(S, 1e-12, 1)
    Hc = -np.log(S)
    h1 = (Hc[20] - Hc[0]) / (grid[20] - grid[0])
    h2 = (Hc[40] - Hc[20]) / (grid[40] - grid[20])
    return h1 / h2 if h2 > 0 else np.nan


def absorption_exponent(W_at_first, psi):
    """Slope of mean ln W_k vs ln|psi_k| at the moment of first firing (pre back-reaction)."""
    ok = np.isfinite(W_at_first).all(1) & (W_at_first > 0).all(1)
    if ok.sum() < 10 or np.ptp(np.abs(psi)) < 1e-9:
        return np.nan
    y = np.log(W_at_first[ok]).mean(0)
    return np.polyfit(np.log(np.abs(psi)), y, 1)[0]


def evaluate(sim, target, rng):
    n = target.size
    counts, n_rec = first_freqs(sim["first"], n)
    t0, tlo, thi = tv_ci(counts, n_rec, target, rng)
    a0, alo, ahi = alpha_ci(sim["fired"], rng)
    return dict(P=(counts / max(n_rec, 1)).round(4).tolist(), target=np.round(target, 4).tolist(),
                tv=t0, tv_lo=tlo, tv_hi=thi, alpha=a0, alpha_lo=alo, alpha_hi=ahi,
                no_record=1 - n_rec / sim["first"].size, unresolved=sim["unresolved"],
                energy_err=sim["energy_err"], tie_frac=sim["ties"] / max(n_rec, 1),
                residual_tie_frac=sim.get("residual_ties", 0) / max(n_rec, 1),
                mean_fired=float(sim["fired"].sum(1).mean()))


def outcome(r):
    if r["unresolved"] > 0.01:
        return "UNRESOLVED"
    if r.get("residual_tie_frac", 0) > TIE_BUDGET:
        return "INVALID-TIES"
    born = r["tv_hi"] <= EPS
    if not np.isfinite(r["alpha_hi"]):
        return "U-Born" if born else "U-notBorn"     # alpha undefined: fewer than 2 channels can fire
    if r["alpha_hi"] < STRONG:
        return "A" if born else "C"
    if r["alpha_hi"] < 1:
        return "A~" if born else "C~"
    return "B" if born else "D"


def rng_for(*key):
    return np.random.default_rng(np.random.SeedSequence([BASE_SEED, *key]))


# ------------------------------------------------------------------ controls
def run_controls():
    out, ok = {}, True
    psi = STATES["S3a"]; tgt = np.abs(psi) ** 2; n = psi.size

    # IC-0 synthetic independent detectors (pipeline only)
    r = rng_for(900)
    F = r.random((N_RUNS, n)) < np.array([0.6, 0.4, 0.25])
    a0, alo, ahi = alpha_ci(F, r)
    p = alo <= 1 <= ahi
    out["IC-0 independent Bernoulli -> alpha CI contains 1"] = (p, f"alpha={a0:.4f} [{alo:.4f},{ahi:.4f}]")

    # IC-1 fixed thresholds, no reversals -> largest first
    s = simulate(psi, "fixed", 1.0, 0.0, N_RUNS, rng_for(901))
    frac = np.mean(s["first"] == np.argmax(tgt))
    p = frac > 0.99
    out["IC-1 fixed, nu=0 -> largest first >99%"] = (p, f"largest first in {frac:.4f}")

    # IC-2 psi-independent coupling -> uniform
    s = simulate(np.ones(n) / np.sqrt(n), "exponential", 1.0, 1.0, N_RUNS, rng_for(902))
    e = evaluate(s, np.ones(n) / n, rng_for(912))
    p = e["tv_hi"] <= EPS
    out["IC-2 psi-independent coupling -> uniform"] = (p, f"P={e['P']} TV_hi={e['tv_hi']:.4f}")

    # IC-3 classical limit (no back-reaction) -> alpha = 1
    s = simulate(psi, "exponential", 1.0, 1.0, N_RUNS, rng_for(903))
    e = evaluate(s, tgt, rng_for(913))
    p = e["alpha_lo"] <= 1 <= e["alpha_hi"]
    out["IC-3 kappa=1 classical limit -> alpha CI contains 1"] = (
        p, f"alpha={e['alpha']:.4f} [{e['alpha_lo']:.4f},{e['alpha_hi']:.4f}], fired/run={e['mean_fired']:.2f}")

    # IC-4 imposed Born draw, single firing (no physics)
    r = rng_for(904)
    k = r.choice(n, N_RUNS, p=tgt)
    F = np.zeros((N_RUNS, n), bool); F[np.arange(N_RUNS), k] = True
    sim = dict(fired=F, first=k, unresolved=0.0, energy_err=0.0, ties=0)
    e = evaluate(sim, tgt, r)
    p = (e["tv_hi"] <= EPS) and (e["alpha_hi"] < 1)
    out["IC-4 imposed |psi|^2 draw -> Born and alpha<1"] = (p, f"TV_hi={e['tv_hi']:.4f} alpha_hi={e['alpha_hi']:.4f}")

    # IC-5 shuffled psi, compared with ORIGINAL target -> mismatch detected
    s = simulate(psi[[2, 0, 1]], "exponential", 4.0, 1.0, N_RUNS, rng_for(905))
    e = evaluate(s, tgt, rng_for(915))
    p = e["tv_lo"] > EPS
    out["IC-5 shuffled psi vs original -> TV_lo > eps"] = (p, f"TV_lo={e['tv_lo']:.4f}")

    for name, (p, msg) in out.items():
        ok &= bool(p)
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {msg}")
    return ok, {k: dict(passed=bool(v[0]), detail=v[1]) for k, v in out.items()}


# ------------------------------------------------------------------ main grid
def main():
    import hashlib, pathlib
    here = pathlib.Path(__file__).resolve()
    digest = hashlib.sha256(here.read_bytes()).hexdigest()
    if "--hash" in sys.argv:
        print(digest); return
    t0 = time.time()
    print("=" * 110)
    print(f"SFIT Part III harness v2  (runs per config = {N_RUNS}{', QUICK' if QUICK else ''})\nscript sha256 {digest}")
    print("=" * 110)
    print("\nControls (THETA = 0.3):")
    ok, ctrl = run_controls()
    report = dict(sha256=digest,
                  params=dict(G0=G0, GAMMA=GAMMA, E0=E0, THETAS=THETAS, DT=DT, N_SUB=N_SUB, T_MAX=T_MAX,
                              DELTA=DELTA, EPS=EPS, CONF=CONF, N_BOOT=N_BOOT, N_RUNS=N_RUNS,
                              STRONG=STRONG, TIE_BUDGET=TIE_BUDGET, BASE_SEED=BASE_SEED),
                  controls=ctrl, valid=ok, grid=[])
    out = here.with_name("part3_v2_results.json")
    if not ok and not QUICK:
        print("\nCONTROLS FAILED: run invalid, no test-model results reported.")
        json.dump(report, open(out, "w"), indent=1, default=float)
        return
    if not ok:
        print("  QUICK mode: controls are underpowered at 4000 runs; continuing for a code check ONLY.\n"
              "  Nothing from a --quick run is a result.\n")
    else:
        print("  All controls passed.\n")

    configs = []
    for sname in STATES:
        for theta in THETAS:
            for law in ("exponential", "fixed", "gamma2"):
                for kappa in (1.0, 2.0, 4.0, 8.0):
                    configs.append((sname, theta, law, kappa, 1.0))
    for theta in THETAS:
        for law in ("exponential", "fixed"):
            for nu in (0.0, 5.0):
                for kappa in (1.0, 4.0):
                    configs.append(("S3a", theta, law, kappa, nu))

    hdr = (f"{'state':5} {'th':>4} {'law':11} {'kap':>3} {'nu':>2} | {'P(first)':32} | {'TV [lo,hi]':22} | "
           f"{'alpha [lo,hi]':22} | {'fired':>5} {'norec':>5} {'rtie':>7} | out")
    print(hdr); print("-" * len(hdr))
    for ci, (sname, theta, law, kappa, nu) in enumerate(configs):
        psi = STATES[sname]; tgt = np.abs(psi) ** 2
        s = simulate(psi, law, kappa, nu, N_RUNS, rng_for(ci), theta=theta)
        e = evaluate(s, tgt, rng_for(10000 + ci))
        e.update(state=sname, theta=theta, law=law, kappa=kappa, nu=nu,
                 largest_first=float(np.mean(s["first"] == np.argmax(tgt))),
                 hazard_ratio=float(hazard_linearity(s["E_at_first"], s["E_final"], theta)),
                 exponent=float(absorption_exponent(s["W_at_first"], psi)))
        e["outcome"] = outcome(e)
        report["grid"].append(e)
        print(f"{sname:5} {theta:4g} {law:11} {kappa:3g} {nu:2g} | {str(e['P']):32} | "
              f"{e['tv']:.4f} [{e['tv_lo']:.4f},{e['tv_hi']:.4f}] | "
              f"{e['alpha']:.3f} [{e['alpha_lo']:.3f},{e['alpha_hi']:.3f}] | "
              f"{e['mean_fired']:5.2f} {e['no_record']:5.3f} {e['residual_tie_frac']:7.1e} | {e['outcome']}", flush=True)

    G = report["grid"]
    print("\nIntegrity (computed):")
    print(f"  max energy-conservation error           : {max(r['energy_err'] for r in G):.2e}")
    print(f"  max raw same-step first-crossing fraction: {max(r['tie_frac'] for r in G):.2e} (resolved by sub-steps)")
    print(f"  max residual tie fraction (budget {TIE_BUDGET:g})  : {max(r['residual_tie_frac'] for r in G):.2e}")
    print(f"  max unresolved fraction                 : {max(r['unresolved'] for r in G):.4f}")
    ex = [r["exponent"] for r in G if np.isfinite(r["exponent"])]
    print(f"  absorption exponent d lnW/d ln|psi|      : min {min(ex):.5f}, max {max(ex):.5f}")

    def sel(**kw):
        return [r for r in G if all(r[k] == v for k, v in kw.items())]

    preds = {}
    preds["P1 exponential -> Born everywhere"] = all(r["tv_hi"] <= EPS for r in sel(law="exponential"))
    p2 = True
    for sname in STATES:
        for theta in THETAS:
            rs = sorted(sel(law="exponential", state=sname, nu=1.0, theta=theta), key=lambda r: r["kappa"])
            p2 &= rs[0]["alpha_lo"] <= 1 <= rs[0]["alpha_hi"]
            p2 &= all(r["alpha_hi"] < 1 for r in rs[1:])
            p2 &= bool(np.all(np.diff([r["alpha"] for r in rs]) < 0))
    preds["P2 alpha=1 at kappa=1; <1 and decreasing for kappa>=2 (exponential, both thetas)"] = p2
    preds["P3 fixed -> largest first >99% for every nu, theta"] = all(r["largest_first"] > 0.99 for r in sel(law="fixed"))
    preds["P4 gamma2 -> not Born, biased to largest"] = all(
        r["tv_lo"] > EPS and r["P"][int(np.argmax(r["target"]))] > max(r["target"]) for r in sel(law="gamma2"))
    preds["D1 exponent = 2 within 1e-3"] = all(abs(x - 2) < 1e-3 for x in ex)
    print("\nPredictions fixed before the run (computed verdicts):")
    for k, v in preds.items():
        print(f"  [{'CONFIRMED' if v else 'NOT CONFIRMED'}] {k}")
    report["predictions"] = {k: bool(v) for k, v in preds.items()}

    print("\nOutcome counts by law and theta  (A = Born + strong exclusion alpha_hi<0.1; A~ = Born + partial):")
    for law in ("exponential", "fixed", "gamma2"):
        for theta in THETAS:
            oc = [r["outcome"] for r in sel(law=law, theta=theta)]
            print(f"  {law:11} theta={theta:<4g}: " + ", ".join(f"{o}={oc.count(o)}" for o in sorted(set(oc))))
    strong = [r for r in G if r["alpha_hi"] < STRONG]
    best = min(G, key=lambda r: r["alpha"] if np.isfinite(r["alpha"]) else np.inf)
    print(f"\nStrong exclusion (alpha_hi < {STRONG}) reached in {len(strong)} of {len(G)} configurations.")
    print(f"Lowest alpha: {best['alpha']:.3f} [{best['alpha_lo']:.3f},{best['alpha_hi']:.3f}] "
          f"({best['state']}, theta={best['theta']}, {best['law']}, kappa={best['kappa']}, nu={best['nu']})")
    print("\nHazard-linearity ratio (S3a, nu=1; 1 = memoryless), by kappa 1,2,4,8:")
    for theta in THETAS:
        for law in ("exponential", "fixed", "gamma2"):
            hr = [r["hazard_ratio"] for r in sorted(sel(law=law, state="S3a", nu=1.0, theta=theta), key=lambda r: r["kappa"])]
            print(f"  theta={theta:<4g} {law:11}: " + ", ".join(f"{h:.3f}" if np.isfinite(h) else "nan" for h in hr))
    json.dump(report, open(out, "w"), indent=1, default=float)
    print(f"\nWall time {time.time() - t0:.0f} s. Results: {out}")


if __name__ == "__main__":
    main()