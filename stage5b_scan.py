"""
stage5b_scan.py -- SFIT Part V Stage 5B: local crossing rules and the integrated over-strand persistence.
Substrate only (no carrier); descriptive scan. Crossing rules from stage5_core.py:
  "separate"  recrossing suppression: after a crossing the pair moves apart in x (bounce-through).
  "exchange"  transverse exchange: after a crossing the over-strand heads down and the under-strand up.
Each rule is applied with probability p per crossing; p = 0 is Part I exactly.

QUESTION. Can a local, exchangeable, braid-preserving crossing rule drive gamma_GK to zero, and if so, is the
zero robust (the same rule parameter works across r = gamma_rt / K) or tuned (it moves with r)?

GRID. r in {0.5, 3, 10} (K = 0.1, dt = 0.02, kappa = 0). Baseline p = 0 once per r.
      separate: p in {0.25, 0.5, 0.75, 1}.  exchange: p in {0.25, 0.375, 0.5, 0.625, 0.75, 1}.
      1000 independent runs per point (~4e5 crossings), 50 groups.
MEASURED per point (definitions as Stage 5a; every row at ONE lag cutoff):
  Gamma_same, Gamma_cross, Gamma_kk at lags 40 / 80 (primary) / 160, block-variance Gamma_kk (block 200),
  gamma_GK = 1.5 Gamma_kk, theta = 1 / (1 + 3 Gamma_kk), lag-1 C_same and C_cross;
  braid structure kept: crossings per unit time, P(next crossing is the same pair), q_bar (same-pair y-order
  flip probability between consecutive crossings), E[s s'] for consecutive same-pair crossings (= -(1-2 q_bar)).
ZERO CROSSING. For each rule and r, p* is the first p at which the piecewise-linear curve Gamma_kk(80)(p)
reaches zero from above; its band is [first zero of the lower 99% curve, first zero of the upper 99% curve]. "Robust" would mean the p* intervals
at the three r overlap; otherwise the zero is a tuning that moves with r. Stated before running.
NOTE. A short exploratory probe (300 runs, before this script) indicated: "separate" changes gamma_GK little;
"exchange" lowers it strongly and at r = 0.5 crosses zero near p = 0.5, with negative values (theta > 1) at
p = 1; at r = 10 it stays positive even at p = 1. This scan measures that properly.
Outputs: stage5b_scan.json, stage5b_scan.pdf/.png. Usage: python stage5b_scan.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np
import stage5_core as S5

QUICK = "--quick" in sys.argv
K = 0.1
RS = {"r0.5": (0.05, 0.057), "r3": (0.3, 0.16), "r10": (1.0, 0.30)}       # gamma_rt, crossings per unit time
RULES = {"separate": [0.25, 0.5, 0.75, 1.0], "exchange": [0.25, 0.375, 0.5, 0.625, 0.75, 1.0]}
R = 150 if QUICK else 1000
TARGET = 2e4 if QUICK else 4e5
NG = 10 if QUICK else 50
LMAX = 160
CUTS = (40, 80, 160)
BLOCK = 200
SEED = 20261801
Z99 = 2.5758
PA = np.array([0, 0, 1]); PB = np.array([1, 2, 2])


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def analyse(runs, T):
    ns = np.zeros((NG, LMAX)); nc = np.zeros((NG, LMAX)); den = np.zeros((NG, LMAX)); blk = []
    ncr = 0; same_next = 0; pairs = 0; flips = 0; tot = 0
    for r, (p, s, u, t) in enumerate(runs):
        p = p.astype(int); u = u.astype(float); n = p.size; g = r % NG; ncr += n
        if n < 2:
            continue
        same_next += int((p[1:] == p[:-1]).sum()); pairs += n - 1
        for q in range(3):
            j = np.flatnonzero(p == q); uu = u[j[1:]] * u[j[:-1]]; flips += int((uu < 0).sum()); tot += uu.size
        W = np.zeros((n, 3)); i = np.arange(n); W[i, PA[p]] = u; W[i, PB[p]] = -u
        for l in range(1, min(LMAX, n - 1) + 1):
            prod = (W[:-l] * W[l:]).sum(1); sm = p[:-l] == p[l:]
            ns[g, l - 1] += prod[sm].sum(); nc[g, l - 1] += prod[~sm].sum(); den[g, l - 1] += 3 * (n - l)
        if n >= BLOCK:
            m = n // BLOCK; B = W[:m * BLOCK].reshape(m, BLOCK, 3).sum(1); blk.append((B ** 2).mean(1))
    sup = (den > 0).all(0); Ls = int(np.argmin(sup)) if not sup.all() else LMAX
    Cs = ns[:, :Ls] / den[:, :Ls]; Cc = nc[:, :Ls] / den[:, :Ls]; G = NG
    out = dict(lags=Ls, crossings=ncr, rate=ncr / (len(runs) * T), p_next_same=same_next / max(pairs, 1),
               qbar=flips / max(tot, 1), Ess_same=-(1 - 2 * flips / max(tot, 1)),
               C1_same=float(Cs[:, 0].mean()), C1_cross=float(Cc[:, 0].mean()), cut={})
    for L in CUTS:
        if L > Ls:
            continue
        a = Cs[:, :L].sum(1); b = Cc[:, :L].sum(1); c = a + b
        f = lambda v: [float(v.mean()), float(v.std(ddof=1) / math.sqrt(G))]
        out["cut"][str(L)] = dict(same=f(a), cross=f(b), total=f(c), theta=1 / (1 + 3 * float(c.mean())))
    if blk:
        v = np.concatenate(blk) / BLOCK; out["block"] = [float((v.mean() - 2 / 3) / 2), float(v.std(ddof=1) / math.sqrt(v.size) / 2)]
    return out


def main():
    here = pathlib.Path(__file__).resolve(); digest = sha(here)
    if "--hash" in sys.argv:
        print(f"stage5b_scan.py {digest}\nstage5_core.py  {sha(here.with_name('stage5_core.py'))}\nstage3_core.py  {sha(here.with_name('stage3_core.py'))}"); return
    t0 = time.time(); bar = "=" * 118
    print(bar); print(f"Part V Stage 5B: crossing rules and over-strand persistence {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}\n"
                      f"stage5_core sha256 {sha(here.with_name('stage5_core.py'))}"); print(bar)
    pts = [(rn, None, 0.0) for rn in RS] + [(rn, rule, p) for rule, ps in RULES.items() for rn in RS for p in ps]
    ss = np.random.SeedSequence(SEED).spawn(len(pts)); res = {}
    print(f"\n{'r':5s} {'rule':9s} {'p':>6} {'rate':>6} {'P(same next)':>12} {'q_bar':>6} {'Gamma_same(80)':>17} {'Gamma_cross(80)':>17}"
          f" {'Gamma_kk(80)':>17} {'theta':>6} {'block':>16} {'lag-1 s/c':>14}")
    for i, (rn, rule, p) in enumerate(pts):
        g, rate = RS[rn]; T = TARGET / (R * rate)
        runs = S5.events(np.random.default_rng(ss[i]), R, T, g, rule, p, K=K)
        a = analyse(runs, T); a.update(r=rn, gamma_rt=g, rule=rule or "none", p=p); key = f"{rn}_{rule or 'none'}_{p}"; res[key] = a
        c = a["cut"]["80"]; blk = a.get("block", [float("nan")] * 2)
        print(f"{rn:5s} {str(rule or 'none'):9s} {p:6.3f} {a['rate']:6.3f} {a['p_next_same']:12.3f} {a['qbar']:6.3f}"
              f" {c['same'][0]:+8.4f}+-{c['same'][1]:.4f} {c['cross'][0]:+8.4f}+-{c['cross'][1]:.4f} {c['total'][0]:+8.4f}+-{c['total'][1]:.4f}"
              f" {c['theta']:6.3f} {blk[0]:+8.4f}+-{blk[1]:.4f} {a['C1_same']:+.3f}/{a['C1_cross']:+.3f}  [{time.time() - t0:.0f} s]", flush=True)
    print("\nCutoff check (Gamma_kk at lags 40 / 80 / 160):")
    for k, a in res.items():
        print(f"   {k:22s} " + "  ".join(f"{L}: {v['total'][0]:+.4f}+-{v['total'][1]:.4f}" for L, v in a["cut"].items()))
    print("\nZero crossings of Gamma_kk(80):")
    zeros = {}
    for rule, ps in RULES.items():
        for rn in RS:
            seq = [(0.0, res[f"{rn}_none_0.0"])] + [(p, res[f"{rn}_{rule}_{p}"]) for p in ps]
            P = [q for q, _ in seq]
            Y = [a_["cut"]["80"]["total"][0] for _, a_ in seq]; E = [Z99 * a_["cut"]["80"]["total"][1] for _, a_ in seq]

            def cross(v):                      # first p at which the piecewise-linear curve v(p) reaches 0 from above
                for k in range(len(v) - 1):
                    if v[k] > 0 >= v[k + 1]:
                        return P[k] + (P[k + 1] - P[k]) * v[k] / (v[k] - v[k + 1])
                return None
            pc = cross(Y)
            found = None
            if pc is not None:
                lo = cross([y - e for y, e in zip(Y, E)]); hi = cross([y + e for y, e in zip(Y, E)])
                found = dict(p_star=pc, interval=[lo if lo is not None else 0.0, hi])   # hi None: beyond the scanned range
            zeros[f"{rule}_{rn}"] = found
            print(f"   {rule:9s} {rn:5s}: " + (f"p* = {found['p_star']:.3f}, 99% band [{found['interval'][0]:.3f}, "
                                                + (f"{found['interval'][1]:.3f}]" if found['interval'][1] is not None else f"> {ps[-1]}]")
                                                if found else f"no sign change for p <= {ps[-1]} (Gamma_kk(80) at p = {ps[-1]}: "
                                                f"{seq[-1][1]['cut']['80']['total'][0]:+.4f})"))
        iv = [zeros[f"{rule}_{rn}"]["interval"] for rn in RS if zeros[f"{rule}_{rn}"]]
        if len(iv) == len(RS):
            robust = max(a for a, b in iv) <= min((b if b is not None else 9.0) for a, b in iv)
            print(f"   {rule}: zero {'ROBUST (intervals overlap across r)' if robust else 'TUNED (p* moves with r)'}")
        else:
            print(f"   {rule}: zero not reached at every r in the scanned range -> not robust within this rule family")
    stem = "stage5b_scan_quick" if QUICK else "stage5b_scan"
    json.dump(dict(script_sha256=digest, quick=QUICK, points=res, zeros=zeros), open(here.with_name(stem + ".json"), "w"), default=float)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        for j, (comp, lab) in enumerate((("total", "Gamma_kk"), ("same", "Gamma_same"), ("cross", "Gamma_cross"))):
            for rule, ls in (("separate", "--"), ("exchange", "-")):
                for rn, col in zip(RS, ("#1f77b4", "#2ca02c", "#d62728")):
                    ps = [0.0] + RULES[rule]; ks = [f"{rn}_none_0.0"] + [f"{rn}_{rule}_{p}" for p in RULES[rule]]
                    y = [res[k]["cut"]["80"][comp][0] for k in ks]; e = [Z99 * res[k]["cut"]["80"][comp][1] for k in ks]
                    ax[j].errorbar(ps, y, yerr=e, ls=ls, marker="o", ms=3, color=col, capsize=2, label=f"{rule}, {rn}" if j == 0 else None)
            ax[j].axhline(0, color="0.5", lw=0.7); ax[j].set_xlabel("rule probability p"); ax[j].set_title(f"{lab} (lag <= 80)", fontsize=10)
        ax[0].legend(fontsize=7, frameon=False); fig.tight_layout()
        fig.savefig(here.with_name(stem + ".pdf")); fig.savefig(here.with_name(stem + ".png"), dpi=130)
        print(f"\nFigure: {stem}.pdf / .png")
    except Exception as e:
        print(f"\n(figure skipped: {e})")
    print(f"Wall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()