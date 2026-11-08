"""
stage3_predict.py -- SFIT Part V Stage 3, PREDICTION step. Substrate only (plus the frozen Stage 2 carrier
results, read and hash-checked); no new carrier outcome is computed. Output frozen to
stage3_predictions.json, to be committed before stage3_validate.py runs.

Carrier, jump law, stopping rule, reduced model: as Stage 2 (stage3_core.py = stage2_core.py + trap K as a
parameter). EPS = 0.3 throughout. Bias b = P(channel 0) - x0[0].

INPUT FROM STAGE 2 (stage2_results.json, sha256 checked against STAGE2_SHA): the six EPS = 0.3 arms
S1..S6 (bias b_i, its SE, q_bar_i) and the S2 record vector P.

3A COLLAPSE TEST. Leading candidate: b = -lambda (1 - 2 q_bar), x0 = (0.5, 0.3, 0.2).
   lambda fitted on the Stage 2 arms as a random-effects mean of rho_i = -b_i / c_i, c_i = 1 - 2 q_bar_i,
   (DerSimonian-Laird; tau = spread of rho beyond statistical error). Prediction for a new configuration
   with measured c: b = -lambda c, SE = c sqrt(SE_lambda^2 + tau^2). Compared prospectively with the
   reduced model (Stage 2 pipeline) on the same new configurations:
      A1 (gamma 0.5, kappa 0.1)  A2 (0.5, 0.6)  A3 (2, 0.1)  A4 (2, 0.6)  A5 (4, 1)        [K = 0.1]
3B PART I FLOOR. With t' = K t and lengths in units of C / K the substrate depends only on gamma / K (and
   K dt); this is exact: rescaled runs reproduce the same crossings bit for bit. So for N = 3 with uniform
   heading kicks, the infimum of |b| over Part I parameters is a scan in gamma / K. Stage 2 covered
   gamma / K = 2.5, 10, 40 (S1-S3). New arms, Part I numerics (K = 0.1, dt = 0.02, kappa = 0):
      B1 gamma / K = 0.1   B2 0.25   B3 0.5   B4 1        (gamma = 0.01, 0.025, 0.05, 0.1)
   plus the fair-coin control on B3. Substrate statistics use longer runs at small gamma (fewer crossings).
3C RECORD-LAW MAP on S2 (gamma 1, kappa 0) with new substrate statistics, at
      x0 = (1/3,1/3,1/3) [symmetry control], (0.5,0.3,0.2) [replication], (0.6,0.2,0.2), (0.4,0.4,0.2),
           (0.7,0.2,0.1), (0.8,0.1,0.1), (0.7,0.3,0), (0.9,0.1,0)  [the last two: two active channels].
   Two candidate laws, each with ONE constant fitted on the Stage 2 S2 measurement P0 = 0.4211:
      affine:  P = (1 - alpha) x0 + alpha / 3        (three-channel points only: it would give an inactive
                                                     channel weight alpha/3, which is impossible)
      power:   P_k = x0_k^beta / sum_j x0_j^beta     (all points)
   Prediction SE: from the Stage 2 SE of P0 propagated through alpha or beta. Reduced-model predictions
   at every point.

PRE-STATED CRITERIA (evaluated by stage3_validate.py; agreement = |z| < 2.576 at 99%)
  3A  LAW SUPPORTED if the lambda law agrees with all five A arms; the reduced model is scored the same way.
  3B  FLOOR APPARENT if all four B arms exclude |b| <= 0.03 at 99%; DRIVABLE TOWARD ZERO if any B arm has
      |b| + 2.576 SE < 0.03; UNDETERMINED otherwise. Fair-coin control on B3 must give Born.
      Expectation (from substrate statistics and lambda, stated before any 3B carrier run): q_bar peaks
      near gamma / K = 0.5, so |b| has an interior minimum of about lambda (1 - 2 q_bar_max).
      Limit stated in advance: gamma / K < 0.1 is not scanned; there crossings become rare (the defects
      settle near the trap's fixed points) and q_bar already falls again between 0.5 and 0.1.
  3C  C0: the symmetric point gives 1/3 in every channel. Replication: (0.5,0.3,0.2) agrees with Stage 2.
      Each candidate law is scored on P0 at every point it covers; a law is SUPPORTED if all agree.

Usage: python stage3_predict.py [--quick] [--hash] [--selftest]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np
from scipy.optimize import brentq
import stage3_core as C3

QUICK = "--quick" in sys.argv
STAGE2_SHA = "ecaec1a28b4bc5abd0bfe188e17ba655d6ad15b0454d76d5c5616f182b4dcfd2"
EPS = 0.3
X0STD = [0.5, 0.3, 0.2]
CONFIGS = {  # name: (gamma, kappa, K, approximate crossing rate per unit time, for sizing the runs)
    "A1": (0.5, 0.1, 0.1, 0.21), "A2": (0.5, 0.6, 0.1, 0.21), "A3": (2.0, 0.1, 0.1, 0.44),
    "A4": (2.0, 0.6, 0.1, 0.44), "A5": (4.0, 1.0, 0.1, 0.62),
    "B1": (0.01, 0.0, 0.1, 0.017), "B2": (0.025, 0.0, 0.1, 0.035), "B3": (0.05, 0.0, 0.1, 0.057),
    "B4": (0.1, 0.0, 0.1, 0.088), "S2n": (1.0, 0.0, 0.1, 0.30)}
X0MAP = [[1 / 3, 1 / 3, 1 / 3], [0.5, 0.3, 0.2], [0.6, 0.2, 0.2], [0.4, 0.4, 0.2], [0.7, 0.2, 0.1],
         [0.8, 0.1, 0.1], [0.7, 0.3, 0.0], [0.9, 0.1, 0.0]]
R_SUB = 300 if QUICK else 2000
TARGET = 6e4 if QUICK else 1.2e6          # substrate crossings per configuration
N_GROUPS = 10 if QUICK else 50
KMAX = 3 if QUICK else 5
N_POINT = 4000 if QUICK else 40000
N_BOOT = 2000 if QUICK else 8000
B = 4 if QUICK else 20
SEED = 20261301
Z99 = 2.5758


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def rec_P(rec):
    ok = rec >= 0
    return np.bincount(rec[ok], minlength=3) / max(ok.sum(), 1)


def substrate_stats(name, seed):
    g, kap, K, rate = CONFIGS[name]
    T = min(TARGET / (R_SUB * rate), 60000.0)
    runs = C3.events_fast(np.random.default_rng(seed), R_SUB, T, g, kappa=kap, K=K)
    gc = [C3.transition_counts([runs[r] for r in range(R_SUB) if r % N_GROUPS == q], KMAX) for q in range(N_GROUPS)]
    fl = tot = ncr = 0
    for p, s, u, t in runs:
        ncr += p.size
        for q in range(3):
            i = np.flatnonzero(p == q); uu = u[i[1:]] * u[i[:-1]]; fl += int((uu < 0).sum()); tot += uu.size
    qbar = fl / max(tot, 1); qse = math.sqrt(qbar * (1 - qbar) / max(tot, 1))
    return gc, dict(gamma=g, kappa=kap, K=K, gamma_over_K=g / K, T=T, crossings=ncr, qbar=qbar, qbar_se=qse, c=1 - 2 * qbar)


def reduced_prediction(gc, x0, seed_arm, sim_seed):
    full = C3.symmetrize({k: sum(z[k] for z in gc) for k in range(1, KMAX + 1)})
    point = {k: rec_P(C3.reduced(full, k, EPS, x0, N_POINT, np.random.default_rng(sim_seed))) for k in range(0, KMAX + 1)}
    rb = np.random.default_rng(seed_arm); boot = np.zeros((B, KMAX + 1, 3))
    for b in range(B):
        pick = rb.integers(0, N_GROUPS, N_GROUPS)
        st = C3.symmetrize({k: sum(gc[q][k] for q in pick) for k in range(1, KMAX + 1)})
        for k in range(1, KMAX + 1):
            boot[b, k] = rec_P(C3.reduced(st, k, EPS, x0, N_BOOT, np.random.default_rng(sim_seed)))
    se = {k: [math.sqrt(max(boot[:, k, c].var(ddof=1) - point[k][c] * (1 - point[k][c]) / N_BOOT, 0)
                        + point[k][c] * (1 - point[k][c]) / N_POINT) for c in range(3)] for k in range(1, KMAX + 1)}
    kstar = None
    for k in range(1, KMAX):
        if all(abs(point[j][0] - point[k][0]) < Z99 * float((boot[:, j, 0] - boot[:, k, 0]).std(ddof=1)) for j in range(k + 1, KMAX + 1)):
            kstar = k; break
    conv = kstar is not None; ku = kstar if conv else KMAX
    trunc = abs(point[KMAX][0] - point[ku][0]) if conv else abs(point[KMAX][0] - point[KMAX - 1][0])
    return dict(points={k: point[k].tolist() for k in point}, se=se, kstar=kstar, converged=conv, k=ku,
                P=point[ku].tolist(), P0=float(point[ku][0]), se0=math.sqrt(se[ku][0] ** 2 + trunc ** 2), trunc=trunc)


def power_P(x0, beta):
    w = np.array([v ** beta if v > 0 else 0.0 for v in x0]); return w / w.sum()


def main():
    here = pathlib.Path(__file__).resolve(); digest = sha(here); core_sha = sha(here.with_name("stage3_core.py"))
    if "--hash" in sys.argv:
        print(f"stage3_predict.py {digest}\nstage3_core.py    {core_sha}"); return
    if "--selftest" in sys.argv:
        import stage2_core as C2
        a = C3.events_fast(np.random.default_rng(3), 40, 300, 1.0); b = C2.events_fast(np.random.default_rng(3), 40, 300, 1.0)
        same = all(all(np.array_equal(x, y) for x, y in zip(r, q)) for r, q in zip(a, b))
        a = C3.events_fast(np.random.default_rng(4), 30, 400, 0.05, K=0.1, dt=0.02)
        b = C3.events_fast(np.random.default_rng(4), 30, 80, 0.25, K=0.5, dt=0.004)
        scal = all(np.array_equal(r[0], q[0]) and np.array_equal(r[2], q[2]) for r, q in zip(a, b))
        print(f"stage3_core reproduces stage2_core at K = 0.1: {same}\nrescaled run (gamma, K, dt) x 5 gives identical crossings: {scal}")
        return
    t0 = time.time(); bar = "=" * 108
    print(bar); print(f"Part V Stage 3 PREDICTIONS {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}\ncore   sha256 {core_sha}"); print(bar)

    # ---------------- Stage 2 input
    s2f = here.with_name("stage2_results.json"); s2sha = sha(s2f)
    print(f"Stage 2 input {s2f.name} sha256 {s2sha} ({'matches' if s2sha == STAGE2_SHA else 'DOES NOT MATCH'} the Stage 2 output)")
    if s2sha != STAGE2_SHA and not QUICK:
        print("Stopping."); return
    s2 = json.load(open(s2f))["arms"]
    rho, srho, cs = [], [], []
    for n in ("S1", "S2", "S3", "S4", "S5", "S6"):
        a = s2[f"{n}_eps0.3"]; c = 1 - 2 * a["qbar"]; rho.append(-a["bias"] / c); srho.append(a["se0"] / c); cs.append(c)
    rho, srho = np.array(rho), np.array(srho); w = 1 / srho ** 2
    rfe = float((w * rho).sum() / w.sum()); Qh = float((w * (rho - rfe) ** 2).sum()); df = len(rho) - 1
    tau2 = max(0.0, (Qh - df) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (srho ** 2 + tau2); lam = float((wr * rho).sum() / wr.sum()); lam_se = float(1 / math.sqrt(wr.sum()))
    print(f"\n3A lambda (random effects over S1-S6): lambda = {lam:.4f} +- {lam_se:.4f}, tau = {math.sqrt(tau2):.4f} "
          f"(heterogeneity Q = {Qh:.1f} on {df} df)")
    P0s2 = s2["S2_eps0.3"]["P"][0]; se_s2 = s2["S2_eps0.3"]["se0"]
    alpha = (0.5 - P0s2) / (0.5 - 1 / 3); alpha_se = se_s2 / (0.5 - 1 / 3)
    beta = brentq(lambda bb: power_P(X0STD, bb)[0] - P0s2, 0.01, 3.0)
    dbdP = 1 / ((power_P(X0STD, beta + 1e-5)[0] - power_P(X0STD, beta - 1e-5)[0]) / 2e-5); beta_se = abs(dbdP) * se_s2
    print(f"3C laws fitted on Stage 2 S2 (P0 = {P0s2:.4f} +- {se_s2:.4f}): alpha = {alpha:.4f} +- {alpha_se:.4f}, beta = {beta:.4f} +- {beta_se:.4f}")
    out = dict(script_sha256=digest, core_sha256=core_sha, stage2_sha256=s2sha, quick=QUICK, eps=EPS, KMAX=KMAX,
               lam=dict(value=lam, se=lam_se, tau=math.sqrt(tau2), Q=Qh, rho=rho.tolist(), rho_se=srho.tolist()),
               alpha=dict(value=alpha, se=alpha_se), beta=dict(value=beta, se=beta_se), substrate={}, arms={})

    # ---------------- substrate statistics
    ss = np.random.SeedSequence(SEED).spawn(len(CONFIGS) + 1); sim_seed = ss[-1]
    gcs = {}
    print("\nSubstrate statistics (independent runs):")
    for ci, name in enumerate(CONFIGS):
        gc, st = substrate_stats(name, ss[ci]); gcs[name] = gc; out["substrate"][name] = st
        print(f"  {name:4s} gamma={st['gamma']:<5} kappa={st['kappa']:<4} gamma/K={st['gamma_over_K']:<6.3g} T={st['T']:<8.0f}"
              f" crossings={st['crossings']:9d}  q_bar={st['qbar']:.4f}+-{st['qbar_se']:.4f}  c=1-2q={st['c']:.4f}  [{time.time() - t0:.0f} s]", flush=True)

    # ---------------- arms
    arms = [(n, X0STD, "A" if n[0] == "A" else "B") for n in CONFIGS if n[0] in "AB"]
    arms += [("S2n", x, "C") for x in X0MAP]
    arm_seeds = np.random.SeedSequence(SEED + 1).spawn(len(arms))
    print(f"\nPredictions (eps = {EPS}); P(channel 0):")
    for ai, (name, x0, part) in enumerate(arms):
        key = f"{name}_x{'-'.join(f'{v:.3g}' for v in x0)}"
        r = reduced_prediction(gcs[name], x0, arm_seeds[ai], sim_seed)
        st = out["substrate"][name]
        arm = dict(part=part, config=name, x0=x0, reduced=r)
        if part in "AB":
            c = st["c"]; arm["law"] = dict(P0=x0[0] - lam * c, se0=c * math.sqrt(lam_se ** 2 + tau2))
        if part == "C":
            if all(v > 0 for v in x0):
                arm["affine"] = dict(P0=(1 - alpha) * x0[0] + alpha / 3, se0=alpha_se * abs(1 / 3 - x0[0]))
            pp = power_P(x0, beta); dp = (power_P(x0, beta + 1e-5)[0] - power_P(x0, beta - 1e-5)[0]) / 2e-5
            arm["power"] = dict(P0=float(pp[0]), P=pp.tolist(), se0=abs(dp) * beta_se)
        out["arms"][key] = arm
        extra = "".join(f"  {lab} {arm[lab]['P0']:.4f}+-{arm[lab]['se0']:.4f}" for lab in ("law", "affine", "power") if lab in arm)
        print(f"  {part} {key:26s} reduced k*={r['kstar']}{'' if r['converged'] else '(FLAGGED)'} {r['P0']:.4f}+-{r['se0']:.4f}"
              f" P={np.round(r['P'], 4).tolist()}{extra}  [{time.time() - t0:.0f} s]", flush=True)
    out["arms"]["B3_coin"] = dict(part="B", config="B3", x0=X0STD, coin=True, reduced=dict(P0=0.5, se0=0.0, P=X0STD))
    pf = here.with_name("stage3_predictions_quick.json" if QUICK else "stage3_predictions.json")
    json.dump(out, open(pf, "w"), indent=1, default=float)
    print(f"\nFrozen: {pf}\nsha256 {sha(pf)}\nWall time {time.time() - t0:.0f} s\nNext: commit it, then run stage3_validate.py")


if __name__ == "__main__":
    main()