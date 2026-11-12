"""
stage5d_audit_scan.py -- SFIT Part V Stage 5D: (A) exchangeability audit of the Stage 5C substrates and
(B) parameter-free crossing rules across r = gamma_rt / K. Substrate only; descriptive; nothing frozen.

(A) EXCHANGEABILITY AUDIT. The one-number reduction theta = 1/(1 + 3 Gamma_kk) assumes the defects are
exchangeable, which fixes the 3x3 matrix Gamma_km = sum_{l=1..80} E[omega_k(n) omega_m(n+l)] to
gamma (I - J/3): equal diagonal, off-diagonal / diagonal = -1/2, symmetric. Checked on Z1 (r 0.5, exchange
0.5), Z2 (r 3, exchange 1), A1 (r 0.5, exchange 1) and the Part I baseline B0 (r 0.5):
   - diagonal equality across k: chi^2 on 2 df of the three Gamma_kk about their mean (group-level deviations);
   - each of the six off-diagonal entries against -diag/2, group SE; max |z| must be < 2.94 (1%, Bonferroni over 6).
     (The AVERAGE off-diagonal / diagonal ratio is identically -1/2 because rows sum to zero, so it is not a test;
     equality of the individual entries is.)
   - antisymmetric part max |Gamma_km - Gamma_mk| / SE (it does not enter the drift, reported only);
   - row sums sum_m Gamma_km (identically 0 because sum_m omega_m = 0 at every crossing: a code check);
   - centering: E[u | pair] for each pair (expected 0) and the pair frequencies (expected 1/3).
   PASS for a configuration if the diagonal chi^2 has p > 0.01, the off-diagonal max |z| < 2.94, and every
   |E[u | pair]| is within 2.576 group SE of 0. (Stated before running.)
(B) PARAMETER-FREE RULES (stage5d_core.py), applied at every crossing:
   randomize_y  each defect's transverse direction redrawn as a fair coin;  tumble  full heading redraw.
   r in {0.1, 0.5, 3, 10, 30}, plus Part I baselines at the same r from new runs.
   Reported per point: Gamma_same, Gamma_cross, Gamma_kk at lag 40 / 80 / 160, block-variance Gamma_kk,
   theta, crossing rate, P(next same pair), q_bar, reduced braid letters per crossing.
   A rule is called ROBUSTLY NEAR ZERO if |Gamma_kk(80)| < 0.02 (|theta - 1| < ~0.06) with its 99% interval inside
   +-0.03 at every r. Otherwise the rule is r-dependent. (Stated before running.)
Outputs: stage5d_audit_scan.json, stage5d_scan.pdf/.png. Usage: python stage5d_audit_scan.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np
from scipy.stats import chi2
import stage5d_core as D5
import stage5c_core as C5

QUICK = "--quick" in sys.argv
K = 0.1
AUDIT = {"B0": (0.05, None, 0.0, 0.057), "Z1": (0.05, "exchange", 0.5, 0.057), "Z2": (0.3, "exchange", 1.0, 0.16),
         "A1": (0.05, "exchange", 1.0, 0.057)}
RS = {"r0.1": (0.01, 0.017), "r0.5": (0.05, 0.057), "r3": (0.3, 0.16), "r10": (1.0, 0.30), "r30": (3.0, 0.52)}
RULES = [None, "randomize_y", "tumble"]
R_A, TGT_A = (200, 4e4) if QUICK else (1000, 6e5)
R_B, TGT_B = (150, 2e4) if QUICK else (1000, 4e5)
NG = 10 if QUICK else 50
LMAX = 160
BLOCK = 200
SEED = 20262001
Z99 = 2.5758
ZOFF = 2.94          # 1% two-sided, Bonferroni over the 6 off-diagonal entries
PA, PB = D5.S5.PA, D5.S5.PB


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def wmat(p, u):
    n = p.size; W = np.zeros((n, 3)); i = np.arange(n); W[i, PA[p]] = u; W[i, PB[p]] = -u; return W


def audit(runs):
    L = 80; M = np.zeros((NG, L, 3, 3)); Cn = np.zeros((NG, L)); upair = np.zeros((NG, 3)); npair = np.zeros((NG, 3))
    for r, (p, s, u, t) in enumerate(runs):
        p = p.astype(int); u = u.astype(float); n = p.size; g = r % NG
        if n < 2:
            continue
        W = wmat(p, u)
        for q in range(3):
            sel = p == q; upair[g, q] += u[sel].sum(); npair[g, q] += sel.sum()
        for l in range(1, min(L, n - 1) + 1):
            M[g, l - 1] += W[:-l].T @ W[l:]; Cn[g, l - 1] += n - l
    G = (M / Cn[:, :, None, None]).sum(1)                       # per-group Gamma_km = sum_l E[omega_k(n) omega_m(n+l)]
    m = G.mean(0); se = G.std(0, ddof=1) / math.sqrt(NG)
    diag = np.array([G[:, k, k] for k in range(3)]); dm = diag.mean(1); dse = diag.std(1, ddof=1) / math.sqrt(NG)
    dev = diag - diag.mean(0, keepdims=True)                    # per-group deviation of each Gamma_kk from the k-mean
    c2 = float(((dev.mean(1) / (dev.std(1, ddof=1) / math.sqrt(NG))) ** 2).sum() * 2 / 3); p_diag = float(chi2.sf(c2, 2))
    dg = np.trace(G, axis1=1, axis2=2) / 3                       # per-group mean diagonal
    offz = []
    for k_ in range(3):
        for m_ in range(3):
            if k_ != m_:
                d_ = G[:, k_, m_] + dg / 2; offz.append(float(d_.mean() / (d_.std(ddof=1) / math.sqrt(NG))))
    off_maxz = float(np.max(np.abs(offz)))
    anti = float(np.max(np.abs(m - m.T) / np.maximum(np.hypot(se, se.T), 1e-12)))
    rows = float(np.abs(G.sum(2)).max())
    eug = upair / np.maximum(npair, 1); eu = upair.sum(0) / npair.sum(0); eu_se = eug.std(0, ddof=1) / math.sqrt(NG)
    freq = npair.sum(0) / npair.sum()
    ok = p_diag > 0.01 and off_maxz < ZOFF and all(abs(eu) < Z99 * eu_se)
    return dict(Gamma=m.tolist(), Gamma_se=se.tolist(), diag=dm.tolist(), diag_se=dse.tolist(), chi2_diag=c2, p_diag=p_diag,
                offdiag_z=offz, offdiag_max_z=off_maxz, antisym_max_z=anti, row_sum_max=rows, E_u_pair=eu.tolist(), E_u_se=eu_se.tolist(),
                pair_freq=freq.tolist(), PASS=bool(ok))


def persistence(runs, T):
    ns = np.zeros((NG, LMAX)); nc = np.zeros((NG, LMAX)); den = np.zeros((NG, LMAX)); blk = []
    ncr = 0; sn = 0; pr = 0; fl = 0; tt = 0
    for r, (p, s, u, t) in enumerate(runs):
        p = p.astype(int); u = u.astype(float); n = p.size; g = r % NG; ncr += n
        if n < 2:
            continue
        sn += int((p[1:] == p[:-1]).sum()); pr += n - 1
        for q in range(3):
            j = np.flatnonzero(p == q); uu = u[j[1:]] * u[j[:-1]]; fl += int((uu < 0).sum()); tt += uu.size
        W = wmat(p, u)
        for l in range(1, min(LMAX, n - 1) + 1):
            prod = (W[:-l] * W[l:]).sum(1); sm = p[:-l] == p[l:]
            ns[g, l - 1] += prod[sm].sum(); nc[g, l - 1] += prod[~sm].sum(); den[g, l - 1] += 3 * (n - l)
        if n >= BLOCK:
            m_ = n // BLOCK; B = W[:m_ * BLOCK].reshape(m_, BLOCK, 3).sum(1); blk.append((B ** 2).mean(1))
    sup = (den > 0).all(0); Ls = int(np.argmin(sup)) if not sup.all() else LMAX
    Cs = ns[:, :Ls] / den[:, :Ls]; Cc = nc[:, :Ls] / den[:, :Ls]
    f = lambda v: [float(v.mean()), float(v.std(ddof=1) / math.sqrt(NG))]
    out = dict(crossings=ncr, rate=ncr / (len(runs) * T), p_next_same=sn / max(pr, 1), qbar=fl / max(tt, 1), cut={})
    for L in (40, 80, 160):
        if L <= Ls:
            a = Cs[:, :L].sum(1); b = Cc[:, :L].sum(1)
            out["cut"][str(L)] = dict(same=f(a), cross=f(b), total=f(a + b))
    if blk:
        v = np.concatenate(blk) / BLOCK; out["block"] = [float((v.mean() - 2 / 3) / 2), float(v.std(ddof=1) / math.sqrt(v.size) / 2)]
    ok, tot, n_, red, g0 = C5.braid_checks(runs)
    out.update(consistent=ok, runs=tot, reduced_per_crossing=red / max(n_, 1))
    c80 = out["cut"].get("80", out["cut"][max(out["cut"], key=int)])["total"][0]; out["theta"] = 1 / (1 + 3 * c80)
    return out


def main():
    here = pathlib.Path(__file__).resolve(); digest = sha(here)
    deps = ("stage5d_core.py", "stage5_core.py", "stage5c_core.py", "stage3_core.py")
    if "--hash" in sys.argv:
        print(f"stage5d_audit_scan.py {digest}\n" + "\n".join(f"{f:21s} {sha(here.with_name(f))}" for f in deps)); return
    t0 = time.time(); bar = "=" * 116
    print(bar); print(f"Part V Stage 5D: exchangeability audit and parameter-free rules {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}"); print(bar)
    ss = np.random.SeedSequence(SEED).spawn(len(AUDIT) + len(RS) * len(RULES)); res = dict(audit={}, scan={})
    print("\n(A) Exchangeability audit, Gamma_km summed to lag 80:")
    for i, (nm, (g, rule, p, rate)) in enumerate(AUDIT.items()):
        T = TGT_A / (R_A * rate); runs = D5.events(np.random.default_rng(ss[i]), R_A, T, g, rule, p, K=K)
        a = audit(runs); res["audit"][nm] = a
        print(f"  {nm} ({rule or 'Part I'} p={p}, r={g / K:g}):  diag " + ", ".join(f"{d:+.4f}+-{s:.4f}" for d, s in zip(a["diag"], a["diag_se"]))
              + f"  chi2={a['chi2_diag']:.2f} (p={a['p_diag']:.2f});  off-diagonal vs -diag/2: max |z| {a['offdiag_max_z']:.2f};"
              f"  antisym max {a['antisym_max_z']:.1f} SE;  row sums {a['row_sum_max']:.1e}")
        print(f"      E[u|pair] = " + ", ".join(f"{e:+.4f}+-{s:.4f}" for e, s in zip(a["E_u_pair"], a["E_u_se"]))
              + "  pair freq " + ", ".join(f"{f_:.3f}" for f_ in a["pair_freq"]) + f"   -> {'PASS' if a['PASS'] else 'FAIL'}   [{time.time() - t0:.0f} s]", flush=True)
    print(f"\n(B) Parameter-free rules:\n  {'r':5s} {'rule':12s} {'rate':>6} {'P(same)':>7} {'q_bar':>6} {'Gamma_same(80)':>17} {'Gamma_cross(80)':>17}"
          f" {'Gamma_kk(80)':>17} {'theta':>6} {'block':>16} {'reduced/cr':>10} {'geom':>9}")
    k = len(AUDIT)
    for rule in RULES:
        for rn, (g, rate) in RS.items():
            T = TGT_B / (R_B * rate); runs = D5.events(np.random.default_rng(ss[k]), R_B, T, g, rule, 1.0 if rule else 0.0, K=K); k += 1
            a = persistence(runs, T); a.update(r=rn, rule=rule or "none"); res["scan"][f"{rn}_{rule or 'none'}"] = a
            c = a["cut"].get("80") or a["cut"][max(a["cut"], key=int)]; b = a.get("block", [float("nan")] * 2)
            print(f"  {rn:5s} {str(rule or 'none'):12s} {a['rate']:6.3f} {a['p_next_same']:7.3f} {a['qbar']:6.3f} {c['same'][0]:+8.4f}+-{c['same'][1]:.4f}"
                  f" {c['cross'][0]:+8.4f}+-{c['cross'][1]:.4f} {c['total'][0]:+8.4f}+-{c['total'][1]:.4f} {a['theta']:6.3f} {b[0]:+8.4f}+-{b[1]:.4f}"
                  f" {a['reduced_per_crossing']:10.3f} {a['consistent']:4d}/{a['runs']:<4d}  [{time.time() - t0:.0f} s]", flush=True)
    print("\nRobustness (criterion: |Gamma_kk(80)| < 0.02 and 99% interval inside +-0.03 at every r):")
    for rule in RULES[1:]:
        rows = [res["scan"][f"{rn}_{rule}"]["cut"].get("80") for rn in RS]
        ok = all(r_ and abs(r_["total"][0]) < 0.02 and abs(r_["total"][0]) + Z99 * r_["total"][1] < 0.03 for r_ in rows)
        print(f"  {rule:12s}: " + ", ".join(f"{rn} {r_['total'][0]:+.3f}" for rn, r_ in zip(RS, rows) if r_) + f"   -> {'ROBUSTLY NEAR ZERO' if ok else 'r-dependent'}")
    stem = "stage5d_audit_scan_quick" if QUICK else "stage5d_audit_scan"
    json.dump(dict(script_sha256=digest, quick=QUICK, **res), open(here.with_name(stem + ".json"), "w"), default=float)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.7)); rv = [0.1, 0.5, 3, 10, 30]
        for j, comp in enumerate(("total", "same", "cross")):
            for rule, col in zip(RULES, ("0.3", "#1f77b4", "#d62728")):
                ks = [f"{rn}_{rule or 'none'}" for rn in RS]
                y = [res["scan"][k_]["cut"].get("80", {}).get(comp, [np.nan, np.nan])[0] for k_ in ks]
                e = [Z99 * res["scan"][k_]["cut"].get("80", {}).get(comp, [np.nan, np.nan])[1] for k_ in ks]
                ax[j].errorbar(rv, y, yerr=e, marker="o", ms=3.5, capsize=2, color=col, label=rule or "Part I")
            ax[j].axhline(0, color="0.6", lw=0.7); ax[j].set_xscale("log"); ax[j].set_xlabel(r"$r=\gamma_{rt}/K$")
            ax[j].set_title({"total": r"$\Gamma_{kk}$", "same": r"$\Gamma_{same}$", "cross": r"$\Gamma_{cross}$"}[comp] + " (lag $\\leq$ 80)")
        ax[0].legend(fontsize=8, frameon=False); fig.tight_layout()
        stemf = "stage5d_scan_quick" if QUICK else "stage5d_scan"
        fig.savefig(here.with_name(stemf + ".pdf")); fig.savefig(here.with_name(stemf + ".png"), dpi=130); print(f"\nFigure: {stemf}.pdf / .png")
    except Exception as e:
        print(f"\n(figure skipped: {e})")
    print(f"Wall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()