"""
stage5a_persistence.py -- SFIT Part V Stage 5a: decomposition of the integrated over-strand persistence.
Substrate only (no carrier). Descriptive analysis; nothing is predicted or tested, so nothing is frozen.

QUESTION. Can an exchangeable, topologically nontrivial stochastic braid substrate have zero integrated
Green-Kubo persistence through cancellation of competing correlations? First step: measure the competing
parts in the Part I substrate.

DEFINITIONS. omega_k(n) = +1 / -1 / 0 if defect k is the over-strand / under-strand / not involved at
crossing n (crossing-indexed, within one run). C(l) = (1/3) sum_k E[omega_k(n) omega_k(n+l)], the
expectation taken over all positions n of every run (pairs never span two runs). Gamma_kk = sum_{l>=1} C(l),
gamma_GK = (3/2) Gamma_kk, theta = 1 / (1 + 2 gamma_GK).
A product omega_k(n) omega_k(n+l) is nonzero only if defect k takes part in both crossings, so each term is
either SAME (the same defect pair crosses again) or CROSS (k crosses a different partner):
    C(l) = C_same(l) + C_cross(l),     Gamma_kk = Gamma_same + Gamma_cross     (exact, checked numerically).
For a SAME term the product is u_n u_{n+l} (the pair's over-strand label), i.e. persistence of that pair's
y-order. For a CROSS term it is +1 if k is over (or under) in both crossings, i.e. persistence of k's
y-rank relative to different partners.

ESTIMATION. Per configuration: 2000 independent runs, sized to ~1.2e6 crossings, split into 50 groups by run
index. Each group gives its own C_same(l), C_cross(l) (sums of products over positions inside its runs,
divided by the number of position pairs at that lag); cumulative sums are formed per group; the estimate is
the mean over groups and its SE the spread over groups / sqrt(50). Cumulative sums reported at lags 40, 80,
160, 320 (where every group has data); each printed row uses ONE cutoff for every column, and
theta = 1 / (1 + 2 gamma_GK) = 1 / (1 + 3 Gamma_kk). Stage 4's primary cutoff was lag 80. Independent check of the total: variance growth of non-overlapping block sums of omega_k
(block length 300; biased low by about (2/300) sum_l l C(l)): Var/n = C(0) + 2 Gamma_kk, C(0) = 2/3.

CONFIGURATIONS: Part I (kappa = 0, K = 0.1) at r = gamma_rt / K = 0.1, 0.5, 3, 10, 30; and one non-Part-I
reference with transverse scrambling (r = 10, kappa = 1).
Outputs: stage5a_persistence.json, stage5a_persistence.pdf (+ .png).
Usage: python stage5a_persistence.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np
import stage3_core as C3

QUICK = "--quick" in sys.argv
K = 0.1
CONFIGS = [("r0.1", 0.01, 0.0, 0.017), ("r0.5", 0.05, 0.0, 0.057), ("r3", 0.3, 0.0, 0.16), ("r10", 1.0, 0.0, 0.30),
           ("r30", 3.0, 0.0, 0.52), ("r10_kappa1", 1.0, 1.0, 0.30)]   # name, gamma_rt, kappa, crossings per unit time
R = 300 if QUICK else 2000
TARGET = 6e4 if QUICK else 1.2e6
NG = 10 if QUICK else 50
LMAX = 320
CUTS = (40, 80, 160, 320)
BLOCK = 300
SEED = 20261701
PA = np.array([0, 0, 1]); PB = np.array([1, 2, 2])


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def analyse(runs):
    ns = np.zeros((NG, LMAX)); nc = np.zeros((NG, LMAX)); nt = np.zeros((NG, LMAX)); den = np.zeros((NG, LMAX))
    blk = []
    for r, (p, s, u, t) in enumerate(runs):
        p = p.astype(int); u = u.astype(float); n = p.size; g = r % NG
        if n < 2:
            continue
        W = np.zeros((n, 3)); i = np.arange(n); W[i, PA[p]] = u; W[i, PB[p]] = -u
        for l in range(1, min(LMAX, n - 1) + 1):
            prod = (W[:-l] * W[l:]).sum(1); same = p[:-l] == p[l:]
            ns[g, l - 1] += prod[same].sum(); nc[g, l - 1] += prod[~same].sum(); nt[g, l - 1] += prod.sum()
            den[g, l - 1] += 3 * (n - l)
        if n >= BLOCK:
            m = n // BLOCK; B = W[:m * BLOCK].reshape(m, BLOCK, 3).sum(1)
            blk.append((B ** 2).mean(1))
    sup = (den > 0).all(0); Lsup = int(np.argmin(sup)) if not sup.all() else LMAX   # lags supported in every group
    ns, nc, nt, den = ns[:, :Lsup], nc[:, :Lsup], nt[:, :Lsup], den[:, :Lsup]
    Cs, Cc, Ct = ns / den, nc / den, nt / den; G = NG
    out = {"L_supported": Lsup}
    for name, C in (("same", Cs), ("cross", Cc), ("total", Ct)):
        cum = np.cumsum(C, 1)
        out[name] = dict(C=C.mean(0).tolist(), C_se=(C.std(0, ddof=1) / math.sqrt(G)).tolist(),
                         cum={str(L): [float(cum[:, L - 1].mean()), float(cum[:, L - 1].std(ddof=1) / math.sqrt(G))] for L in CUTS if L <= Lsup})
    out["recon_maxdiff"] = float(np.abs(Cs + Cc - Ct).max())
    if blk:
        v = np.concatenate(blk) / BLOCK
        out["block_Gamma"] = [float((v.mean() - 2 / 3) / 2), float(v.std(ddof=1) / math.sqrt(v.size) / 2)]
    out["groups_used"] = int(G)
    return out


def main():
    here = pathlib.Path(__file__).resolve(); digest = sha(here); core = sha(here.with_name("stage3_core.py"))
    if "--hash" in sys.argv:
        print(f"stage5a_persistence.py {digest}\nstage3_core.py         {core}"); return
    t0 = time.time(); bar = "=" * 112
    print(bar); print(f"Part V Stage 5a: persistence decomposition {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}\ncore   sha256 {core}"); print(bar)
    ss = np.random.SeedSequence(SEED).spawn(len(CONFIGS)); res = {}
    for ci, (name, g, kap, rate) in enumerate(CONFIGS):
        T = min(TARGET / (R * rate), 60000.0)
        runs = C3.events_fast(np.random.default_rng(ss[ci]), R, T, g, kappa=kap, K=K)
        a = analyse(runs); a.update(gamma_rt=g, kappa=kap, r=g / K, T=T, crossings=int(sum(x[0].size for x in runs)))
        res[name] = a
        print(f"\n{name}: r={g / K:g} kappa={kap} crossings={a['crossings']}  (lags supported: {a['L_supported']})  [{time.time() - t0:.0f} s]")
        print(f"   lag-1: C_same {a['same']['C'][0]:+.4f}  C_cross {a['cross']['C'][0]:+.4f}  (C(0) = 2/3)")
        print(f"   {'cut':>4} {'Gamma_same':>16} {'Gamma_cross':>16} {'Gamma_kk':>16} {'same share':>10} {'gamma_GK':>9} {'theta':>7}")
        rows = {}
        for L in a["total"]["cum"]:              # every column of a row uses the SAME cutoff
            gs, gc, gt = a["same"]["cum"][L], a["cross"]["cum"][L], a["total"]["cum"][L]
            gam = 1.5 * gt[0]; th = 1 / (1 + 2 * gam)      # = 1 / (1 + 3 Gamma_kk)
            rows[L] = dict(share_same=gs[0] / gt[0], gamma_GK=gam, theta=th)
            print(f"   {L:>4} {gs[0]:+8.4f}+-{gs[1]:.4f} {gc[0]:+8.4f}+-{gc[1]:.4f} {gt[0]:+8.4f}+-{gt[1]:.4f} {gs[0] / gt[0]:10.3f} {gam:9.3f} {th:7.4f}")
        a["by_cut"] = rows
        print(f"   reconstruction |C_same + C_cross - C_total| max = {a['recon_maxdiff']:.1e}")
        if "block_Gamma" in a:
            print(f"   block-variance check (no lag cut, block {BLOCK}): Gamma_kk = {a['block_Gamma'][0]:+.4f}+-{a['block_Gamma'][1]:.4f}")
    json.dump(dict(script_sha256=digest, core_sha256=core, quick=QUICK, configs=res), open(here.with_name(
        "stage5a_persistence_quick.json" if QUICK else "stage5a_persistence.json"), "w"), default=float)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        names = [c[0] for c in CONFIGS]; fig, ax = plt.subplots(2, len(names), figsize=(3.0 * len(names), 5.6), sharex=True)
        for j, nm in enumerate(names):
            a = res[nm]; L = np.arange(1, a["L_supported"] + 1)
            for part, col in (("same", "#1f77b4"), ("cross", "#d62728"), ("total", "0.2")):
                C = np.array(a[part]["C"]); se = np.array(a[part]["C_se"])
                ax[0, j].plot(L, C, color=col, lw=1.2 if part != "total" else 0.9, ls="-" if part != "total" else ":", label=part)
                ax[0, j].fill_between(L, C - 2.576 * se, C + 2.576 * se, color=col, alpha=0.15, lw=0)
                ax[1, j].plot(L, np.cumsum(C), color=col, lw=1.4 if part != "total" else 1.0, ls="-" if part != "total" else ":")
            for k in (0, 1):
                ax[k, j].axhline(0, color="0.6", lw=0.6); ax[k, j].set_xscale("log")
            ax[0, j].set_title(nm.replace("_", ", "), fontsize=9); ax[1, j].set_xlabel("lag (crossings)")
        ax[0, 0].set_ylabel("C(lag), per-k average"); ax[1, 0].set_ylabel("cumulative sum"); ax[0, 0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        stem = "stage5a_persistence_quick" if QUICK else "stage5a_persistence"
        fig.savefig(here.with_name(stem + ".pdf")); fig.savefig(here.with_name(stem + ".png"), dpi=130)
        print(f"\nFigure: {stem}.pdf / .png")
    except Exception as e:
        print(f"\n(figure skipped: {e})")
    print(f"Wall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()