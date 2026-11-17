#!/usr/bin/env python3
"""
stage5g_gamma2_integral.py -- SFIT Part V Stage 5G: candidate second-order coefficient Gamma_2.

  Gamma(r) = 13/30 + Gamma_1 r + Gamma_2 r^2 + o(r^2)        (assumes the expansion exists: CANDIDATE, not a theorem)
  Gamma_2  = 0.75 V_2 - 1.15 nu_2 - 1.5 Gamma_1 nu_1          (exact quotient expansion of Gamma = (V/nu - 2/3)/2)

  V_2  = V_2^single  + V_2^pair      nu_2 = nu_2^single + nu_2^pair
  single: 9 int_0^1 F(q) ln(q)/q dq = -9 int_0^inf F(e^-a) a da          (second order of the Beta(3r,1) law)
  pair:   9 sum_{s>=0} int int K(u,v;s)/(uv) du dv = 9 sum_s int int K da db,  u = e^-a, v = e^-b
          K_V(u,v;s) = E[ DD(L^2) + 2 DD(L) (H(U) + H(Z)) ]   (four-arm mixed difference, exact ballistic tails)
          K_nu(u,v;s) = E[ DD(count) ]                         (a mean: no squares, no tails)
  DD f = f(u,v) - f(u,0) - f(0,v) + f(0,0) on the SAME headings and movers (common random numbers).
  Window: gap u in interval 1, s ballistic intervals, gap v in interval s+2, then one completion interval (q=0).
  For s >= 1 the completed blocks are additive, so DD(L) = DD(count) = 0 EXACTLY per sample (checked below) and
  K_V(s) = 2 E[Delta_u L Delta_v L]; it decays like (2/3)^(s-1). nu_2^pair therefore comes from s = 0 only.
  Rewards are averaged over the three defects k (exchangeable), a free variance reduction.

Requires stage5f_insertion_kernels.py (frozen Stage 5F kernel) in the same folder: its single-gap kernel is reused
unchanged, and its H_ballistic supplies the exact resolvent tails.
"""
import argparse, hashlib, json, time
import numpy as np
from scipy.stats import qmc
import stage5f_insertion_kernels as ik

PA = np.array([0, 0, 1]); PB = np.array([1, 2, 2])
COEF = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1]], float)   # COEF[k, pair] = 1{PA=k} - 1{PB=k}
GAMMA1, GAMMA1_ERR, NU1 = -0.332, 0.003, -1.28870               # frozen Stage 5F values


def interval(X, E, q):
    """Exact crossings while every defect relaxes X -> E + (X-E)q', q' from 1 down to q. Returns (marks, count, X_end)."""
    dx0 = X[:, PA, 0] - X[:, PB, 0]; dx1 = E[:, PA, 0] - E[:, PB, 0]
    dy0 = X[:, PA, 1] - X[:, PB, 1]; dy1 = E[:, PA, 1] - E[:, PB, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        root = dx1 / (dx1 - dx0)
    hit = np.isfinite(root) & (root > q[:, None]) & (root < 1.0)
    dy = dy1 + (dy0 - dy1) * np.where(hit, root, 0.0)
    hit &= np.abs(dy) > 1e-13
    marks = np.sign(dy) * hit                                    # sgn(y_A - y_B) at each crossing
    return marks, hit.sum(1), E + (X - E) * q[:, None, None]


def window(U, movers, heads, qs):
    """Run a window from fixed-point state U. Returns rewards L (N,3), crossing count C (N,), end state Z."""
    N = U.shape[0]; rows = np.arange(N); X = U.copy(); E = U.copy()
    L = np.zeros((N, 3)); C = np.zeros(N)
    for t in range(movers.shape[1]):
        E[rows, movers[:, t], 0] = np.cos(heads[:, t]); E[rows, movers[:, t], 1] = np.sin(heads[:, t])
        m, c, X = interval(X, E, qs[t])
        L += m @ COEF.T; C += c
    return L, C, E


def H_all(pos):
    return np.stack([ik.H_ballistic(pos, k) for k in range(3)], 1)


def pair_kernel(rng, N, s, Lcut):
    """One batch: per-sample integrands K_V and K_nu at log-uniform (a,b) in [0,Lcut]^2, plus exactness diagnostics."""
    n = s + 3
    ph = rng.uniform(0, 2 * np.pi, (N, 3)); U = np.stack((np.cos(ph), np.sin(ph)), 2)
    movers = rng.integers(0, 3, (N, n)); heads = rng.uniform(0, 2 * np.pi, (N, n))
    u = np.exp(-Lcut * rng.random(N)); v = np.exp(-Lcut * rng.random(N)); z = np.zeros(N)
    arms = {}
    for key, (qa, qb) in {"uv": (u, v), "u0": (u, z), "0v": (z, v), "00": (z, z)}.items():
        qs = [z] * n; qs[0] = qa; qs[s + 1] = qb
        arms[key] = window(U, movers, heads, qs)
    Z = arms["00"][2]
    dd = lambda f: f(arms["uv"]) - f(arms["u0"]) - f(arms["0v"]) + f(arms["00"])
    ddL = dd(lambda a: a[0]); ddL2 = dd(lambda a: a[0] ** 2); ddC = dd(lambda a: a[1])
    for key in ("uv", "u0", "0v"):                               # all arms must end in the same fixed-point state
        assert np.allclose(arms[key][2], Z)
    KV = (ddL2 + 2 * ddL * (H_all(U) + H_all(Z))).mean(1)       # average over k
    return KV, ddC, np.abs(ddL).max(), np.abs(ddC).max()


def single_gap(seed, log2n, Lcut):
    """Second-order Beta term from the frozen Stage 5F kernel: returns (V2_single, nu2_single) for one Sobol scramble."""
    ang, q = ik.sobol_angles(log2n, seed, True, Lcut)
    _, _, allv = ik.five_case_mean(ang, q)
    a = -np.log(q)
    I_nu = Lcut * np.mean(allv[0] * a); I_V = Lcut * np.mean(allv[1] * a)
    return -9 * I_V, -9 * I_nu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", type=int, default=16)
    p.add_argument("--batch", type=int, default=1 << 18, help="samples per batch")
    p.add_argument("--batches", type=int, default=4, help="batches per group and s")
    p.add_argument("--smax", type=int, default=6)
    p.add_argument("--Lpair", type=float, default=10.0, help="pair cutoff: u,v >= e^-Lpair")
    p.add_argument("--Lsingle", type=float, default=20.0)
    p.add_argument("--log2-single", type=int, default=18)
    p.add_argument("--seed", type=int, default=20261116)
    p.add_argument("--json", default="stage5g_gamma2_candidate.json")
    a = p.parse_args()
    sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    ksha = hashlib.sha256(open(ik.__file__, "rb").read()).hexdigest()
    print(f"Script SHA-256: {sha}\nKernel SHA-256 (stage5f_insertion_kernels.py): {ksha}")
    print(f"seed={a.seed} groups={a.groups} samples per (group,s)={a.batch*a.batches} smax={a.smax} "
          f"Lpair={a.Lpair} Lsingle={a.Lsingle}\nCANDIDATE coefficient; assumes the second-order expansion exists.\n")
    se = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))
    t0 = time.time()

    # --- single-gap second-order Beta term (independent Sobol scrambles) ---
    sg = np.array([single_gap(a.seed + 1000 + g, a.log2_single, a.Lsingle) for g in range(a.groups)])
    V2s, nu2s = sg[:, 0], sg[:, 1]
    print(f"single-gap: V2 = {V2s.mean():+.4f} +- {se(V2s):.4f}   nu2 = {nu2s.mean():+.4f} +- {se(nu2s):.4f}")

    # --- pair terms ---
    rng = np.random.default_rng(a.seed)
    KV = np.zeros((a.groups, a.smax + 1)); Knu0 = np.zeros(a.groups); maxL = maxC = 0.0
    for g in range(a.groups):
        for s in range(a.smax + 1):
            acc = []; accn = []
            for _ in range(a.batches):
                kv, kn, mL, mC = pair_kernel(rng, a.batch, s, a.Lpair)
                acc.append(kv.mean()); accn.append(kn.mean())
                if s >= 1: maxL, maxC = max(maxL, mL), max(maxC, mC)
            KV[g, s] = a.Lpair ** 2 * np.mean(acc)                # int int K da db over [0,Lpair]^2
            if s == 0: Knu0[g] = a.Lpair ** 2 * np.mean(accn)
        print(f"  group {g+1:2d}/{a.groups} done  ({time.time()-t0:.0f} s)", flush=True)
    print(f"\nexactness check, s>=1: max|DD L| = {maxL:.1e}, max|DD count| = {maxC:.1e}   (both must be 0)")
    print("pair kernel by separation (9 x int int K da db):")
    for s in range(a.smax + 1):
        print(f"  s={s}: {9*KV[:, s].mean():+.5f} +- {9*se(KV[:, s]):.5f}")
    tail = 2 * KV[:, a.smax]                                     # s > smax, dominant mode (2/3)^(s-1)
    V2p_g = 9 * (KV.sum(1) + tail); nu2p_g = 9 * Knu0
    print(f"  tail s>{a.smax} (geometric, dominant mode): {9*tail.mean():+.5f} +- {9*se(tail):.5f}")
    print(f"pair: V2 = {V2p_g.mean():+.4f} +- {se(V2p_g):.4f}   nu2 = {nu2p_g.mean():+.4f} +- {se(nu2p_g):.4f}")

    # --- assembly (groups paired: single scramble g with pair group g) ---
    V2 = V2s + V2p_g; nu2 = nu2s + nu2p_g; quot = -1.5 * GAMMA1 * NU1
    G2 = 0.75 * V2 - 1.15 * nu2 + quot
    G2_err = float(np.hypot(se(G2), 1.5 * GAMMA1_ERR * abs(NU1)))
    print(f"\nV2 = {V2.mean():+.4f} +- {se(V2):.4f}   nu2 = {nu2.mean():+.4f} +- {se(nu2):.4f}   "
          f"quotient term -1.5*Gamma1*nu1 = {quot:+.4f}")
    print(f"CANDIDATE Gamma_2 = {G2.mean():+.4f} +- {G2_err:.4f}  (numerical; includes Gamma_1 uncertainty)")
    print("Cross-check only (data already seen): Stage 5F fitted r^2 coefficient C = +0.66 +- 0.10")
    print(f"pair truncation: |K| <= C w(min(u,v)) implies the region u or v < e^-{a.Lpair:g} is O(Lpair^3 e^-Lpair)")
    out = {"Gamma2": float(G2.mean()), "Gamma2_err": G2_err, "V2": [float(V2.mean()), se(V2)], "nu2": [float(nu2.mean()), se(nu2)],
           "V2_single": [float(V2s.mean()), se(V2s)], "nu2_single": [float(nu2s.mean()), se(nu2s)],
           "V2_pair": [float(V2p_g.mean()), se(V2p_g)], "nu2_pair": [float(nu2p_g.mean()), se(nu2p_g)],
           "pair_by_s": [[float(9 * KV[:, s].mean()), 9 * se(KV[:, s])] for s in range(a.smax + 1)],
           "quotient_term": quot, "Gamma1": GAMMA1, "nu1": NU1, "script_sha256": sha, "kernel_sha256": ksha,
           "args": vars(a), "status": "candidate; requires (A), (B), cluster bound for k>=3"}
    s_json = json.dumps(out, indent=1); open(a.json, "w").write(s_json)
    print(f"\nSaved {a.json}  SHA-256: {hashlib.sha256(s_json.encode()).hexdigest()}")


if __name__ == "__main__":
    main()