"""
step0_braid_drift.py -- Step 0: does the Part I braid action make x_k = psi_k^2 a martingale?

Dynamics: Part I v6 planar substrate (sfit_substrate_v6_2D_braid_structure.py), unchanged:
N=3, c=2, gamma=2 (Poisson direction scattering), k_trap=0.1, dt=0.02; crossings extracted
as adjacent x-order swaps, sign from y (Appendix B convention).
Representations applied to the carrier psi at each extracted crossing:
  R8: v8 helicity action (Part I -> Part II integration): a crossing of particles p, q swaps
      their helicities -> 8x8 permutation matrix on the helicity basis (sign-independent).
  R2: v7 hand-specified B3 rep, rho(s1)=[[1,1],[0,1]], rho(s2)=[[1,0],[-1,1]] (+ inverses).
Measured: one-crossing conditional drift E[dx | x] at random x, and the ensemble mean of x
after many crossings compared with x(0). Martingale <=> drift = 0 for every x.
"""
import numpy as np

def crossing_events(T, seed, N=3, c=2.0, gamma=2.0, k_trap=0.1, dt=0.02):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, (N, 2)); th = rng.uniform(0, 2*np.pi, N)
    order_prev = np.argsort(X[:, 0]); ev = []
    for _ in range(int(T / dt)):
        v = c * np.stack([np.cos(th), np.sin(th)], 1)
        X += (v - k_trap * X) * dt
        s = rng.random(N) < gamma * dt
        th[s] += rng.uniform(-np.pi, np.pi, s.sum())
        oc = np.argsort(X[:, 0])
        if not np.array_equal(order_prev, oc):
            d = np.flatnonzero(order_prev != oc)
            if d.size == 2 and abs(d[0] - d[1]) == 1:
                k = d.min(); pl, pr = oc[k], oc[k+1]
                ev.append((k, 1 if X[pl, 1] > X[pr, 1] else -1, pl, pr))
        order_prev = oc
    return ev

def R8(p, q):
    M = np.zeros((8, 8))
    for i in range(8):
        bits = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
        bits[p], bits[q] = bits[q], bits[p]
        M[bits[0]*4 + bits[1]*2 + bits[2], i] = 1
    return M

A1 = np.array([[1., 1.], [0., 1.]]); A2 = np.array([[1., 0.], [-1., 1.]])
def R2(k, s):
    A = A1 if k == 0 else A2
    return A if s == 1 else np.linalg.inv(A)

def shares(psi):
    return psi**2 / np.sum(psi**2)

ev = crossing_events(T=20000.0, seed=1)
print(f"Part I v6 dynamics: {len(ev)} crossings extracted")
gens = np.array([(k, s) for k, s, _, _ in ev])
pairs = [tuple(sorted((p, q))) for _, _, p, q in ev]
for g in [(0, 1), (0, -1), (1, 1), (1, -1)]:
    print(f"  generator sigma_{g[0]+1}^{g[1]:+d}: {np.mean((gens[:,0]==g[0])&(gens[:,1]==g[1])):.3f}")
for pq in sorted(set(pairs)):
    print(f"  particle pair {pq}: {pairs.count(pq)/len(pairs):.3f}")

rng = np.random.default_rng(7)
print("\n[R8] v8 helicity-swap representation (8D permutation matrices)")
mats8 = [R8(p, q) for _, _, p, q in ev]
Pbar = sum(mats8) / len(mats8)                      # empirical one-crossing average map
X0 = rng.dirichlet(np.ones(8), 2000)                # random population vectors on the simplex
drift = X0 @ Pbar.T - X0                            # E[x'|x] - x  (x permutes exactly: x'=P x)
print(f"  one-crossing |E[dx|x]|: median {np.median(np.abs(drift).max(1)):.4f}, "
      f"max {np.abs(drift).max():.4f}  (martingale requires 0)")
x0 = np.array([0.4, 0.3, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02])
# exact ensemble evolution: x is permuted, so E[x(t)] = Pbar^t x0
for t in (1, 5, 20, 200):
    print(f"  E[x] after {t:3d} crossings: {np.round(np.linalg.matrix_power(Pbar, t) @ x0, 3)}")
print(f"  x(0)                       : {np.round(x0, 3)}")
print("  any one-hot (physical helicity) state stays one-hot: x is ALWAYS a vertex, never a superposition")

print("\n[R2] v7 hand-specified B3 representation (2D, not orthogonal)")
g2 = [R2(k, s) for k, s, _, _ in ev]
print(f"  orthogonal? {all(np.allclose(M.T @ M, np.eye(2)) for M in (A1, A2))}; "
      f"||rho(s1) (0,1)|| = {np.linalg.norm(A1 @ np.array([0., 1.])):.3f} (norm not preserved)")
th0 = rng.uniform(0, np.pi, 4000); P0 = np.stack([np.cos(th0), np.sin(th0)], 1)
X1 = np.array([shares(p) for p in P0])
D = []
for M in (A1, np.linalg.inv(A1), A2, np.linalg.inv(A2)):
    D.append(np.array([shares(M @ p) for p in P0]) - X1)
w = [np.mean((gens[:,0]==k)&(gens[:,1]==s)) for k, s in [(0,1),(0,-1),(1,1),(1,-1)]]
drift2 = sum(wi * Di for wi, Di in zip(w, D))
print(f"  one-crossing |E[dx_1|x]|: median {np.median(np.abs(drift2[:,0])):.4f}, max {np.abs(drift2[:,0]).max():.4f}")
psi = np.tile(np.sqrt([0.7, 0.3]), (4000, 1)); L = 300
for r in range(4000):
    j = rng.integers(0, len(g2) - L)
    for M in g2[j:j+L]:
        psi[r] = M @ psi[r]; psi[r] /= np.linalg.norm(psi[r])
xf = psi**2
print(f"  x_1(0) = 0.700 -> E[x_1] after {L} crossings = {xf[:,0].mean():.3f}; "
      f"fraction within 0.01 of a vertex = {np.mean(np.minimum(xf[:,0], 1-xf[:,0]) < 0.01):.3f}")