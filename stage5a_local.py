"""
stage5a_local.py -- SFIT Part V Stage 5a (local): are there NEGATIVE same-pair or cross-pair correlations
anywhere in lag space, even though both integrated sums are positive?
Substrate only; descriptive. Regenerates exactly the Stage 5a runs (same configurations, sizes and seeds,
imported from stage5a_persistence.py) and keeps the per-group, per-lag arrays that Stage 5a averaged away.

WHY. If negative windows exist, cancellation is a matter of making them larger (engineer cancellation).
If C_same(l) and C_cross(l) are non-negative at every resolvable lag, the only route to gamma_GK -> 0 is to
suppress the positive memory itself (suppress recrossing / rank persistence).

TESTS (per configuration and component, C = C_same or C_cross):
  1. Pointwise: z(l) = mean_l / SE_l over the 50 groups. Count lags with z < -2.576 against the number
     expected by chance (0.005 x lags), and apply Benjamini-Hochberg at q = 0.01 to the one-sided p-values
     for C(l) < 0.
  2. Windows: sums of C over lag windows [1], [2], [3], [4-5], [6-10], [11-20], [21-40], [41-80], [81-160],
     [161-320], computed per group so the SE includes correlations between lags. A window is
     SIGNIFICANTLY NEGATIVE if its upper bound mean + z* SE < 0, with z* = 3.29 (one-sided alpha = 0.01 / 20,
     Bonferroni over 10 windows x 2 components per configuration).
  3. Negative mass: the sum of the significantly negative windows, as a fraction of the positive integrated
     sum to lag 80, i.e. how much cancellation the substrate already contains.
VERDICT per configuration (stated before running): NEGATIVE STRUCTURE if any window of either component is
significantly negative; otherwise NONE RESOLVED (with the smallest window upper bound reported, which caps any
undetected negative window).
Check: the lag-80 cumulative sums must equal those printed by stage5a_persistence.py (same runs).
Outputs: stage5a_local.json, stage5a_local.npz (group arrays), stage5a_local.pdf/.png.
Usage: python stage5a_local.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np
from scipy.stats import norm
import stage5a_persistence as S5        # same CONFIGS, R, TARGET, NG, LMAX, SEED, K, and --quick handling
import stage3_core as C3

WINDOWS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 160), (161, 320)]
ZSTAR = norm.isf(0.01 / 20)            # 3.29
PA = np.array([0, 0, 1]); PB = np.array([1, 2, 2])


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def group_arrays(runs):
    NG, L = S5.NG, S5.LMAX
    ns = np.zeros((NG, L)); nc = np.zeros((NG, L)); den = np.zeros((NG, L))
    for r, (p, s, u, t) in enumerate(runs):
        p = p.astype(int); u = u.astype(float); n = p.size; g = r % NG
        if n < 2:
            continue
        W = np.zeros((n, 3)); i = np.arange(n); W[i, PA[p]] = u; W[i, PB[p]] = -u
        for l in range(1, min(L, n - 1) + 1):
            prod = (W[:-l] * W[l:]).sum(1); same = p[:-l] == p[l:]
            ns[g, l - 1] += prod[same].sum(); nc[g, l - 1] += prod[~same].sum(); den[g, l - 1] += 3 * (n - l)
    sup = (den > 0).all(0); Ls = int(np.argmin(sup)) if not sup.all() else L
    return ns[:, :Ls] / den[:, :Ls], nc[:, :Ls] / den[:, :Ls]


def bh(p, q):
    m = p.size; o = np.argsort(p); thr = q * np.arange(1, m + 1) / m
    ok = p[o] <= thr; k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    rej = np.zeros(m, bool); rej[o[:k]] = True; return rej


def main():
    here = pathlib.Path(__file__).resolve(); digest = sha(here)
    if "--hash" in sys.argv:
        print(f"stage5a_local.py       {digest}\nstage5a_persistence.py {sha(here.with_name('stage5a_persistence.py'))}\n"
              f"stage3_core.py         {sha(here.with_name('stage3_core.py'))}"); return
    t0 = time.time(); bar = "=" * 112
    print(bar); print(f"Part V Stage 5a (local): negative correlations in lag space {'(QUICK)' if S5.QUICK else ''}\n"
                      f"script sha256 {digest}\nruns as in stage5a_persistence.py {sha(here.with_name('stage5a_persistence.py'))[:16]}..."); print(bar)
    ss = np.random.SeedSequence(S5.SEED).spawn(len(S5.CONFIGS)); res = {}; arrays = {}
    for ci, (name, g, kap, rate) in enumerate(S5.CONFIGS):
        T = min(S5.TARGET / (S5.R * rate), 60000.0)
        runs = C3.events_fast(np.random.default_rng(ss[ci]), S5.R, T, g, kappa=kap, K=S5.K)
        Cs, Cc = group_arrays(runs); G = Cs.shape[0]; Ls = Cs.shape[1]
        arrays[name + "_same"] = Cs; arrays[name + "_cross"] = Cc
        out = dict(lags=Ls)
        cum80 = {}
        print(f"\n{name}: r={g / S5.K:g} kappa={kap}  lags {Ls}  [{time.time() - t0:.0f} s]")
        for comp, C in (("same", Cs), ("cross", Cc)):
            m = C.mean(0); se = C.std(0, ddof=1) / math.sqrt(G); z = m / se
            pneg = norm.cdf(z)                                   # one-sided p for C(l) < 0
            nneg = int((z < -2.576).sum()); expct = 0.005 * Ls; rej = bh(pneg, 0.01)
            wins = []
            for a, b in WINDOWS:
                if b > Ls:
                    continue
                w = C[:, a - 1:b].sum(1); wm = float(w.mean()); ws = float(w.std(ddof=1) / math.sqrt(G))
                wins.append(dict(window=[a, b], sum=wm, se=ws, upper=wm + ZSTAR * ws, sig_negative=bool(wm + ZSTAR * ws < 0),
                                 sig_positive=bool(wm - ZSTAR * ws > 0)))
            c80 = float(C[:, :min(80, Ls)].sum(1).mean()); cum80[comp] = c80
            negmass = sum(w_["sum"] for w_ in wins if w_["sig_negative"])
            out[comp] = dict(pointwise_neg_count=nneg, expected_by_chance=expct, bh_neg_lags=(np.nonzero(rej)[0] + 1).tolist(),
                             windows=wins, cum80=c80, negative_mass=negmass,
                             min_upper=min(w_["upper"] for w_ in wins), first_lags=m[:6].tolist())
            print(f"   {comp:5s} lags 1-6: " + " ".join(f"{v:+.4f}" for v in m[:6])
                  + f"   pointwise z<-2.576: {nneg} (chance {expct:.1f});  BH q=0.01 negative lags: {out[comp]['bh_neg_lags'][:10]}")
            print("          windows: " + "  ".join(f"[{w_['window'][0]}-{w_['window'][1]}] {w_['sum']:+.4f}+-{w_['se']:.4f}"
                                                      + ("(-)" if w_["sig_negative"] else "(+)" if w_["sig_positive"] else "")
                                                      for w_ in wins))
        neg = any(w_["sig_negative"] for c in ("same", "cross") for w_ in out[c]["windows"])
        verdict = "NEGATIVE STRUCTURE" if neg else "NONE RESOLVED"
        out["verdict"] = verdict; out["cum80_check"] = cum80
        print(f"   lag-80 sums (must match Stage 5a): same {cum80['same']:+.4f}, cross {cum80['cross']:+.4f}")
        if neg:
            for c in ("same", "cross"):
                if out[c]["negative_mass"] < 0:
                    print(f"   negative mass in {c}: {out[c]['negative_mass']:+.4f} = {out[c]['negative_mass'] / (cum80['same'] + cum80['cross']):+.1%} of Gamma_kk(80)")
        print(f"   VERDICT: {verdict}   (smallest window upper bound: same {out['same']['min_upper']:+.4f}, cross {out['cross']['min_upper']:+.4f})")
        res[name] = out
    stem = "stage5a_local_quick" if S5.QUICK else "stage5a_local"
    json.dump(dict(script_sha256=digest, quick=S5.QUICK, z_star=ZSTAR, configs=res), open(here.with_name(stem + ".json"), "w"), default=float)
    np.savez_compressed(here.with_name(stem + ".npz"), **arrays)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        names = [c[0] for c in S5.CONFIGS]; fig, ax = plt.subplots(2, len(names), figsize=(3.0 * len(names), 5.4))
        for j, nm in enumerate(names):
            for r_, comp, col in ((0, "same", "#1f77b4"), (1, "cross", "#d62728")):
                C = arrays[f"{nm}_{comp}"]; m = C.mean(0); se = C.std(0, ddof=1) / math.sqrt(C.shape[0]); L = np.arange(1, m.size + 1)
                ax[r_, j].plot(L, m, color=col, lw=1.1); ax[r_, j].fill_between(L, m - 2.576 * se, m + 2.576 * se, color=col, alpha=0.2, lw=0)
                ax[r_, j].axhline(0, color="0.5", lw=0.6); ax[r_, j].set_xscale("log")
                lo = min(-0.01, float((m - 2.576 * se)[2:].min())); hi = max(0.02, float(m[1:].max()) * 1.1)
                ax[r_, j].set_ylim(lo, hi)
            ax[0, j].set_title(nm.replace("_", ", "), fontsize=9); ax[1, j].set_xlabel("lag (crossings)")
        ax[0, 0].set_ylabel("C_same(lag)"); ax[1, 0].set_ylabel("C_cross(lag)")
        fig.tight_layout(); fig.savefig(here.with_name(stem + ".pdf")); fig.savefig(here.with_name(stem + ".png"), dpi=130)
        print(f"\nFigure: {stem}.pdf / .png")
    except Exception as e:
        print(f"\n(figure skipped: {e})")
    print(f"Wall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()