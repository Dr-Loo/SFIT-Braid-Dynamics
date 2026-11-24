"""
stage5j_plainloop_sim.py -- SFIT Part V Stage 5J: INDEPENDENT plain-loop re-implementation, FROZEN.

Written separately from the vectorized simulators to catch definition and code errors, not noise.
Deliberate differences from stage5j_primary_sim.py:
  * physical-time parametrization: interval duration tau ~ Exp(3r); gap(t) = a + b exp(-t)
  * crossing time found by BISECTION on [0, tau] (no closed-form q*)
  * one long chain per group, scalar loops, DIRECT running lag sums (circular buffer, no FFT)
  * NumPy legacy RNG inside the loop (different generator and stream from the primary code)
Same frozen estimator definitions: c(m) without mean subtraction, M = max(60, floor(25/(1-rho))),
B = 4M, Gamma = (V_M/nu - 2/3)/2.

Exploratory (NOT scored): event-index token autocorrelation rho_e(l), l = 1..L_E, for each fixed defect k,
events ordered by crossing time. Under the renewal-to-event time change (Stage 5J step 2),
   Gamma = (2/3) * sum_{l>=1} rho_e(l)        [ballistic check: sum = 13/20]
so this run also gives an empirical check of that identity and of the sign of each rho_e(l).

numba is used if installed (pip install numba); otherwise the same code runs in pure Python (slow: use small T).
Usage: python stage5j_plainloop_sim.py r G T seed [--LE 100]
"""
import sys, json, hashlib, math, numpy as np
try:
    from numba import njit
except ImportError:                      # pure-Python fallback, identical logic
    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]

@njit(cache=True)
def chain(r, T, M, B, LE, seed):
    np.random.seed(seed)
    ex = np.zeros(3); ey = np.zeros(3); x = np.zeros(3); y = np.zeros(3)
    for i in range(3):
        a = np.random.uniform(0.0, 2*math.pi); ex[i] = math.cos(a); ey[i] = math.sin(a); x[i] = ex[i]; y[i] = ey[i]
    hist = np.zeros((3, M+1)); acc = np.zeros(M+1); cnt = np.zeros(M+1)
    ebuf = np.zeros((3, LE)); nev = np.zeros(3, np.int64); eacc = np.zeros(LE+1); ecnt = np.zeros(LE+1)
    ncr = 0
    ev_t = np.zeros(6); ev_k = np.zeros(6, np.int64); ev_s = np.zeros(6)
    for t in range(-B, T):
        i = np.random.randint(0, 3); a = np.random.uniform(0.0, 2*math.pi)
        ex[i] = math.cos(a); ey[i] = math.sin(a)
        tau = np.random.exponential(1.0/(3.0*r))
        Rk = np.zeros(3); ne = 0
        for p in range(3):
            A = 0 if p < 2 else 1; Bd = 1 if p == 0 else 2
            g0 = x[A] - x[Bd]
            ga = ex[A] - ex[Bd]; gb = (x[A]-ex[A]) - (x[Bd]-ex[Bd])
            g1 = ga + gb*math.exp(-tau)
            if g0*g1 < 0.0:
                lo = 0.0; hi = tau
                for _ in range(80):                       # bisection on monotone gap
                    mid = 0.5*(lo+hi)
                    gm = ga + gb*math.exp(-mid)
                    if gm*g0 > 0.0: lo = mid
                    else: hi = mid
                ts = 0.5*(lo+hi); w = math.exp(-ts)
                dy = (ey[A]-ey[Bd]) + ((y[A]-ey[A]) - (y[Bd]-ey[Bd]))*w
                s = 1.0 if dy > 0 else -1.0
                Rk[A] += s; Rk[Bd] -= s
                ev_t[ne] = ts; ev_k[ne] = A; ev_s[ne] = s; ne += 1
                ev_t[ne] = ts; ev_k[ne] = Bd; ev_s[ne] = -s; ne += 1
                if t >= 0: ncr += 1
        w = math.exp(-tau)
        for j in range(3):
            x[j] = ex[j] + (x[j]-ex[j])*w; y[j] = ey[j] + (y[j]-ey[j])*w
        if t < 0:
            continue
        for k in range(3):                                # direct running lag sums
            for m in range(M, 0, -1):
                hist[k, m] = hist[k, m-1]
            hist[k, 0] = Rk[k]
            mmax = min(M, t)
            for m in range(mmax+1):
                acc[m] += Rk[k]*hist[k, m]; cnt[m] += 1.0
        # exploratory event-index correlations, events in time order
        for e1 in range(ne):
            for e2 in range(e1+1, ne):
                if ev_t[e2] < ev_t[e1]:
                    tt = ev_t[e1]; ev_t[e1] = ev_t[e2]; ev_t[e2] = tt
                    kk = ev_k[e1]; ev_k[e1] = ev_k[e2]; ev_k[e2] = kk
                    ss = ev_s[e1]; ev_s[e1] = ev_s[e2]; ev_s[e2] = ss
        for e in range(ne):
            k = ev_k[e]; s = ev_s[e]
            lmax = min(LE, nev[k])
            for l in range(1, lmax+1):
                eacc[l] += s*ebuf[k, (nev[k]-l) % LE]; ecnt[l] += 1.0
            ebuf[k, nev[k] % LE] = s; nev[k] += 1
    return acc/cnt, ncr/T, eacc[1:]/np.maximum(ecnt[1:], 1.0)

def main():
    a = sys.argv[1:]; LE = 100
    if "--LE" in a:
        j = a.index("--LE"); LE = int(a[j+1]); a = a[:j] + a[j+2:]
    r = float(a[0]); G, T, seed = int(a[1]), int(a[2]), int(a[3])
    rho = 3*r/(1+3*r); M = int(max(60, math.floor(25/(1-rho)))); B = 4*M
    cs, ns, es = [], [], []
    for g in range(G):
        c, n, e = chain(r, T, M, B, LE, seed*1000 + g); cs.append(c); ns.append(n); es.append(e)
    cs = np.array(cs); ns = np.array(ns); es = np.array(es)
    V = cs[:, 0] + 2*cs[:, 1:].sum(1); Gm = 0.5*(V/ns - 2/3)
    Ge = (2/3)*es.sum(1)
    se = lambda z: float(z.std(ddof=1)/np.sqrt(len(z)))
    em = es.mean(0); ese = es.std(0, ddof=1)/np.sqrt(G)
    sha = hashlib.sha256(open(__file__, 'rb').read().replace(b'\r\n', b'\n')).hexdigest()  # CRLF-normalized
    print(json.dumps({"script": "stage5j_plainloop_sim.py", "sha256": sha, "r": r, "seed": seed, "G": G, "T": T,
        "M": M, "tumbles": G*T, "nu": [float(ns.mean()), se(ns)], "V": [float(V.mean()), se(V)],
        "Gamma": [float(Gm.mean()), se(Gm)],
        "Gamma_groups": Gm.tolist(), "nu_groups": ns.tolist(),
        "exploratory_event_index": {"L_E": LE, "Gamma_event": [float(Ge.mean()), se(Ge)],
            "rho_e_first10": em[:10].tolist(), "rho_e_first10_se": ese[:10].tolist(),
            "min_z_over_lags": float(np.min(em/np.maximum(ese, 1e-12))),
            "n_lags_negative_below_3se": int(np.sum(em < -3*ese))}}))

if __name__ == "__main__":
    main()
