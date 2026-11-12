"""
stage5c_predict.py -- SFIT Part V Stage 5C, PREDICTION step: zero-persistence record test.
Substrate statistics + the analytic law only; no carrier outcome computed or read. Frozen to
stage5c_predictions.json; commit it in its OWN commit before stage5c_validate.py.

CONFIGURATIONS (exchange rule of stage5_core.py; K = 0.1, dt = 0.02, kappa = 0). Fixed from the Stage 5B scan
before any carrier run; p values are the scanned grid values, not interpolated:
   Z1  r = 0.5, p = 0.5    candidate zero persistence (5B: Gamma_kk(80) = -0.009 +- 0.007)
   Z2  r = 3,   p = 1.0    second near-zero point at another r (5B: +0.014 +- 0.007)
   A1  r = 0.5, p = 1.0    negative persistence, theta > 1 (5B: Gamma_kk(80) = -0.148 +- 0.004)
STEP 1  Independent re-estimate of Gamma_kk (new seeds; 2000 runs, ~1.2e6 crossings, 50 groups): same-pair /
        cross-pair / total at lags 40, 80 (primary), 160, and the block-variance estimate (block 200).
        theta = 1 / (1 + 3 Gamma_kk(80)); defined for Gamma_kk > -1/3.
STEP 2  Geometry and braid activity (stage5c_core.braid_checks), with the Part I baseline at r = 0.5 for comparison:
        every run's crossing sequence must be realisable by adjacent transpositions with matching left/right
        outcomes; braid activity = crossings per unit time, sigma_1 share, and letters surviving free reduction
        per crossing (sigma_i sigma_i^-1 cancelled).
STEP 3  Record-law predictions at H = 0.999, EPS = 0.3, three starting weights per configuration:
        (0.5,0.3,0.2) and (0.7,0.2,0.1) by the log-weight SDE (dt 0.002, 4 blocks x 40,000 paths),
        (0.8,0.2,0) by the exact two-channel law. The finite-H Born reference uses the same solver at theta = 1
        (160,000 paths for three channels; exact for two). SE: Monte Carlo + propagated SE(theta) + 0.0008 (time step).
        Also printed for each arm: the finite-H Born reference (theta = 1) and the ideal Born weight x0[0].
        Extra arm: A1 (0.5,0.3,0.2) at EPS = 0.1 (finite-step check in the amplifying regime).
        Control: Z1, fair coin, (0.7,0.3,0), H = 0.99: Born finite-H 0.7041.
PRE-STATED CRITERIA (stage5c_validate.py; 99% throughout)
  G   geometry: 100% of runs consistent at all three configurations (STEP 2).
  C   fair-coin control agrees with 0.7041.
  Q1  every model arm |z| < 2.576 against its measured-theta prediction, and chi^2 over the model arms p > 0.01.
  Q2  Z1: every arm agrees with its finite-H Born reference (theta = 1) at 99%. (Z2's agreement with Born is
      reported but not scored: its 5B value, theta ~ 0.96, may differ from 1 measurably at 40,000 runs.)
  Q3  A1: P0 exceeds x0[0] at 99% for (0.5,0.3,0.2) and (0.8,0.2,0) -- records sharper than Born -- and agrees
      with the prediction (part of Q1).
  ZERO-PERSISTENCE RECORD LAW CONFIRMED if G, C, Q1 and Q2 hold; AMPLIFYING REGIME CONFIRMED if G, C and Q3 hold.
Usage: python stage5c_predict.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np
import stage5_core as S5
import stage5c_core as C5
import stage4_predict as P4

QUICK = "--quick" in sys.argv
K = 0.1
CONFIGS = {"Z1": (0.05, "exchange", 0.5, 0.057), "Z2": (0.3, "exchange", 1.0, 0.16), "A1": (0.05, "exchange", 1.0, 0.057),
           "B0": (0.05, None, 0.0, 0.057)}          # B0: Part I baseline, geometry/activity comparison only
STD, X721, E82, E73 = [0.5, 0.3, 0.2], [0.7, 0.2, 0.1], [0.8, 0.2, 0.0], [0.7, 0.3, 0.0]
ARMS = [dict(config=c, x0=x, H=0.999, eps=0.3, n=40000) for c in ("Z1", "Z2", "A1") for x in (STD, X721, E82)]
ARMS += [dict(config="A1", x0=STD, H=0.999, eps=0.1, n=10000), dict(config="Z1", x0=E73, H=0.99, eps=0.3, n=20000, coin=True)]
R_SUB = 300 if QUICK else 2000
TARGET = 6e4 if QUICK else 1.2e6
NG = 10 if QUICK else 50
LMAX = 160
BLOCK = 200
BLOCKS, RB = (2, 2000) if QUICK else (4, 40000)
SEED = 20261901
PA, PB = S5.PA, S5.PB


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def gamma_parts(runs):
    ns = np.zeros((NG, LMAX)); nc = np.zeros((NG, LMAX)); den = np.zeros((NG, LMAX)); blk = []
    for r, (p, s, u, t) in enumerate(runs):
        p = p.astype(int); u = u.astype(float); n = p.size; g = r % NG
        if n < 2:
            continue
        W = np.zeros((n, 3)); i = np.arange(n); W[i, PA[p]] = u; W[i, PB[p]] = -u
        for l in range(1, min(LMAX, n - 1) + 1):
            prod = (W[:-l] * W[l:]).sum(1); sm = p[:-l] == p[l:]
            ns[g, l - 1] += prod[sm].sum(); nc[g, l - 1] += prod[~sm].sum(); den[g, l - 1] += 3 * (n - l)
        if n >= BLOCK:
            m = n // BLOCK; B = W[:m * BLOCK].reshape(m, BLOCK, 3).sum(1); blk.append((B ** 2).mean(1))
    sup = (den > 0).all(0); Ls = int(np.argmin(sup)) if not sup.all() else LMAX
    Cs = ns[:, :Ls] / den[:, :Ls]; Cc = nc[:, :Ls] / den[:, :Ls]; out = {}
    f = lambda v: [float(v.mean()), float(v.std(ddof=1) / math.sqrt(NG))]
    for L in (40, 80, 160):
        if L <= Ls:
            out[str(L)] = dict(same=f(Cs[:, :L].sum(1)), cross=f(Cc[:, :L].sum(1)), total=f(Cs[:, :L].sum(1) + Cc[:, :L].sum(1)))
    if blk:
        v = np.concatenate(blk) / BLOCK; out["block"] = [float((v.mean() - 2 / 3) / 2), float(v.std(ddof=1) / math.sqrt(v.size) / 2)]
    out["lag1"] = [float(Cs[:, 0].mean()), float(Cc[:, 0].mean())]
    return out


def main():
    here = pathlib.Path(__file__).resolve(); digest = sha(here)
    files = ("stage5_core.py", "stage5c_core.py", "stage3_core.py", "stage4_predict.py")
    shas = {f: sha(here.with_name(f)) for f in files}
    if "--hash" in sys.argv:
        print(f"stage5c_predict.py {digest}\n" + "\n".join(f"{f:18s} {h}" for f, h in shas.items())); return
    t0 = time.time(); bar = "=" * 112
    print(bar); print(f"Part V Stage 5C PREDICTIONS {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}\n" + "\n".join(f"{f} {h[:16]}..." for f, h in shas.items())); print(bar)
    ss = np.random.SeedSequence(SEED).spawn(len(CONFIGS) + len(ARMS) + 2)
    th = {}; sub = {}
    print("\nSTEP 1-2: persistence and braid checks")
    for ci, (name, (g, rule, p, rate)) in enumerate(CONFIGS.items()):
        T = TARGET / (R_SUB * rate)
        runs = S5.events(np.random.default_rng(ss[ci]), R_SUB, T, g, rule, p, K=K)
        gp = gamma_parts(runs); ok, tot, ncr, red, g0 = C5.braid_checks(runs)
        G80, se80 = gp["80"]["total"]; t_ = 1 / (1 + 3 * G80); t_se = 3 * se80 * t_ ** 2
        th[name] = (t_, t_se)
        sub[name] = dict(gamma_rt=g, rule=rule, p=p, T=T, crossings=ncr, rate=ncr / (R_SUB * T), gamma=gp, theta=t_, theta_se=t_se,
                         consistent=ok, runs=tot, reduced_per_crossing=red / max(ncr, 1), sigma1_share=g0 / max(ncr, 1))
        print(f"  {name} r={g / K:g} rule={rule} p={p}: crossings {ncr} (rate {ncr / (R_SUB * T):.3f}); geometry consistent {ok}/{tot};"
              f" sigma_1 share {g0 / max(ncr, 1):.3f}; reduced letters per crossing {red / max(ncr, 1):.3f}   [{time.time() - t0:.0f} s]")
        for L in ("40", "80", "160"):
            if L in gp:
                print(f"      lag<={L:>3}: Gamma_same {gp[L]['same'][0]:+.4f}+-{gp[L]['same'][1]:.4f}  Gamma_cross {gp[L]['cross'][0]:+.4f}+-{gp[L]['cross'][1]:.4f}"
                      f"  Gamma_kk {gp[L]['total'][0]:+.4f}+-{gp[L]['total'][1]:.4f}")
        if "block" in gp:
            print(f"      block-variance Gamma_kk {gp['block'][0]:+.4f}+-{gp['block'][1]:.4f};  lag-1 same/cross {gp['lag1'][0]:+.4f}/{gp['lag1'][1]:+.4f}")
        print(f"      theta = {t_:.4f} +- {t_se:.4f}", flush=True)

    print("\nSTEP 3: record-law predictions, P(record = 0)")
    arms = []
    for ai, a in enumerate(ARMS):
        c = a["config"]; t_, t_se = th[c]; x0, H = a["x0"], a["H"]
        if a.get("coin"):
            pr = dict(P0=(x0[0] - (1 - H)) / (2 * H - 1), se0=0.0, method="Born finite-H, fair coin"); born = pr["P0"]; born_se = 0.0
        elif x0[2] == 0.0:
            P0 = P4.edge_law(t_, x0[0], H); d = (P4.edge_law(t_ + 1e-4, x0[0], H) - P4.edge_law(t_ - 1e-4, x0[0], H)) / 2e-4
            pr = dict(P0=P0, se0=abs(d) * t_se, method="exact two-channel"); born = P4.edge_law(1.0, x0[0], H); born_se = 0.0
        else:
            sq = ss[len(CONFIGS) + ai]; bl = [P4.record_sde(x0, t_, H, RB, sq.spawn(BLOCKS)[b], dt=0.002)[0] for b in range(BLOCKS)]
            P0 = float(np.mean(bl)); se_mc = math.sqrt(P0 * (1 - P0) / (BLOCKS * RB)); dth = max(2 * t_se, 0.01); s2 = sq.spawn(BLOCKS + 1)[-1]
            d = (P4.record_sde(x0, t_ + dth, H, RB // 2, s2, dt=0.004)[0] - P4.record_sde(x0, t_ - dth, H, RB // 2, s2, dt=0.004)[0]) / (2 * dth)
            pr = dict(P0=P0, se0=math.sqrt(se_mc ** 2 + (d * t_se) ** 2 + 0.0008 ** 2), method="SDE", blocks=[float(b) for b in bl])
            nb = BLOCKS * RB; born = float(P4.record_sde(x0, 1.0, H, nb, sq.spawn(BLOCKS + 2)[-1], dt=0.002)[0])
            born_se = math.sqrt(born * (1 - born) / nb + 0.0008 ** 2)
        g, rule, p, _ = CONFIGS[c]
        arms.append(dict(a, n=(200 if QUICK else a["n"]), gamma_rt=g, rule=rule, p=p, theta=t_, prediction=pr, born_finiteH=born, born_se=born_se))
        print(f"  {c} x0={x0} H={H} eps={a['eps']:<4} theta={t_:.4f}: predicted {pr['P0']:.4f} +- {pr['se0']:.4f} ({pr['method']})"
              f"   Born finite-H {born:.4f}   ideal Born {x0[0]:.3f}   [{time.time() - t0:.0f} s]", flush=True)
    out = dict(script_sha256=digest, file_sha256=shas, quick=QUICK, K=K, substrate=sub, arms=arms)
    pf = here.with_name("stage5c_predictions_quick.json" if QUICK else "stage5c_predictions.json")
    json.dump(out, open(pf, "w"), indent=1, default=float)
    print(f"\nFrozen: {pf}\nsha256 {sha(pf)}\nWall time {time.time() - t0:.0f} s\nNext: commit this file on its own, then run stage5c_validate.py")


if __name__ == "__main__":
    main()