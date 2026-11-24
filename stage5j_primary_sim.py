"""
stage5j_primary_sim.py -- SFIT Part V Stage 5J (Stage 1 of the post-5I programme): primary estimator, FROZEN.

Exact event-driven Part I substrate (same law as stage5f_event_sim.py): three non-interacting defects,
dX_i/dt = e(phi_i) - X_i (time unit 1/K), global tumbles at rate 3r, one uniformly chosen defect gets a
fresh uniform heading per tumble. Between tumbles X(q) = E + (X0 - E) q with q = exp(-tau), so every pair's
longitudinal gap is a + b q and crosses at most once, at q* = -a/b (closed form).

Reward: fixed-defect reward R_k(n) = sum over partners j of omega_k = sgn(y_k - y_j) at each crossing of pair
(k, j) in interval n (Stage 5I, eq. 2). Estimators (frozen):
  c(m)   = mean over chains, k, and n of R_k(n) R_k(n+m), NO mean subtraction (mean is 0 by reflection)
  nu     = crossings (all three pairs) per tumble
  V_L    = c(0) + 2 sum_{m=1}^{L} c(m)
  Gamma_L = (V_L / nu - 2/3) / 2
  Lag cutoff M = max(60, floor(25 / (1 - rho))), rho = 3r / (1 + 3r).  Primary: Gamma_M.  Truncation
  systematic: |Gamma_M - Gamma_2M| from the same data.  Burn-in B = 4M tumbles from positions at the fixed points.
  T = max(4000, 20 M) tumbles per chain. Groups are independent; SE = std(ddof=1)/sqrt(G) across groups.

Usage: python stage5j_primary_sim.py r G C seed [--T T]      (prints one JSON line)
"""
import sys, json, hashlib, numpy as np

PA = np.array([0, 0, 1]); PB = np.array([1, 2, 2])

def cutoff(r):
    rho = 3*r/(1+3*r); return int(max(60, np.floor(25/(1-rho))))

def run_group(r, C, T, M, rng, chunk=50):
    B = 4*M
    ph = rng.uniform(0, 2*np.pi, (C, 3)); E = np.stack((np.cos(ph), np.sin(ph)), 2); X = E.copy()
    R = np.zeros((C, T, 3), np.int8); ncr = 0; rows = np.arange(C)
    for t in range(-B, T):
        i = rng.integers(0, 3, C); new = rng.uniform(0, 2*np.pi, C)
        E[rows, i, 0] = np.cos(new); E[rows, i, 1] = np.sin(new)
        qe = np.exp(-rng.exponential(1.0/(3*r), C)); D = X - E
        a = E[:, PA, 0] - E[:, PB, 0]; b = D[:, PA, 0] - D[:, PB, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            qs = -a/b
        hit = np.isfinite(qs) & (qs > qe[:, None]) & (qs < 1.0); qs = np.where(hit, qs, 0.0)
        dy = (E[:, PA, 1]-E[:, PB, 1]) + (D[:, PA, 1]-D[:, PB, 1])*qs
        u = (np.sign(dy)*hit).astype(np.int8)
        if t >= 0:
            for p in range(3):
                R[:, t, PA[p]] += u[:, p]; R[:, t, PB[p]] -= u[:, p]
            ncr += int(hit.sum())
        X = E + D*qe[:, None, None]
    L = 2*M; acc = np.zeros(L+1)
    for s in range(0, C, chunk):                       # chunked FFT autocorrelation (memory-safe)
        F = np.fft.rfft(R[s:s+chunk].astype(np.float64), n=2*T, axis=1)
        acc += np.fft.irfft(F*np.conj(F), n=2*T, axis=1)[:, :L+1].sum((0, 2))
    c = acc/(3*C*(T - np.arange(L+1)))
    return c, ncr/(C*T)

def main():
    a = sys.argv[1:]; Tover = None
    if "--T" in a:
        j = a.index("--T"); Tover = int(a[j+1]); a = a[:j] + a[j+2:]
    r = float(a[0]); G, C, seed = int(a[1]), int(a[2]), int(a[3])
    M = cutoff(r); T = Tover if Tover else max(4000, 20*M)
    rng = np.random.default_rng(seed); cs, ns = [], []
    for g in range(G):
        c, n = run_group(r, C, T, M, rng); cs.append(c); ns.append(n)
    cs = np.array(cs); ns = np.array(ns)
    VM = cs[:, 0] + 2*cs[:, 1:M+1].sum(1); V2 = cs[:, 0] + 2*cs[:, 1:].sum(1)
    GM = 0.5*(VM/ns - 2/3); G2 = 0.5*(V2/ns - 2/3)
    se = lambda x: float(x.std(ddof=1)/np.sqrt(len(x)))
    sha = hashlib.sha256(open(__file__, 'rb').read().replace(b'\r\n', b'\n')).hexdigest()  # CRLF-normalized
    print(json.dumps({"script": "stage5j_primary_sim.py", "sha256": sha, "r": r, "seed": seed, "G": G, "C": C,
        "T": T, "M": M, "tumbles": G*C*T,
        "nu": [float(ns.mean()), se(ns)], "V": [float(VM.mean()), se(VM)],
        "Gamma": [float(GM.mean()), se(GM)], "Gamma_2M": [float(G2.mean()), se(G2)],
        "trunc_sys": float(abs(GM.mean() - G2.mean())),
        "nu_sqrt_r": [float(ns.mean()*np.sqrt(r)), se(ns)*float(np.sqrt(r))],
        "B": 4*M, "Gamma_groups": GM.tolist(), "Gamma_2M_groups": G2.tolist(), "nu_groups": ns.tolist(),
        "V_groups": VM.tolist()}))

if __name__ == "__main__":
    main()
