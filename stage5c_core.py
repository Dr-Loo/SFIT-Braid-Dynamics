"""
stage5c_core.py -- SFIT Part V Stage 5C: carrier and geometry checks on the rule substrate of stage5_core.py.
carrier(): identical to stage3_core.carrier except that the substrate is stage5_core.RuleSubstrate.
braid_checks(): per run, (i) geometric consistency -- the crossing sequence is realisable by adjacent
transpositions from some initial left-to-right order, with each crossing's recorded left/right outcome
(L = s u) matching that order; (ii) braid activity -- generator index, free reduction of the braid word
(sigma_i sigma_i^-1 cancels), and the reduced word length per crossing.
"""
import itertools
import numpy as np
import stage3_core as C3
import stage5_core as S5

PA, PB = C3.PA, C3.PB


def carrier(rng, R, gamma, eps, x0, rule=None, p=0.0, kappa=0.0, K=C3.K_TRAP, coin_rng=None, H=0.999, max_T=2.0e5, dt=C3.DT):
    S = S5.RuleSubstrate(rng, R, gamma, dt, kappa, K, rule, p)
    x = np.tile(np.asarray(x0, float), (R, 1)); rec = np.full(R, -1); ncr = np.zeros(R, np.int64)
    idx = np.arange(R); live = np.ones(R, bool)
    for it in range(int(max_T / dt)):
        for rows, pp, s, u, tm in S.step():
            m = live[rows]
            if not m.any():
                continue
            rows, pp, u = rows[m], pp[m], u[m]
            sg = u if coin_rng is None else coin_rng.choice(np.array([-1, 1]), rows.size)
            xr = x[rows] + (eps * sg)[:, None] * C3.d_pair(x[rows], pp)
            xr = np.clip(xr, 0, None); xr /= xr.sum(1, keepdims=True); x[rows] = xr
            ncr[idx[rows]] += 1
            hit = xr.max(1) >= H
            if hit.any():
                rec[idx[rows[hit]]] = xr[hit].argmax(1); live[rows[hit]] = False
        if not live.any():
            break
        if it % 500 == 499 and live.mean() < 0.7:
            S.keep(live); x = x[live]; idx = idx[live]; live = live[live]
    return rec, ncr


def braid_checks(runs):
    """Returns (runs consistent, runs total, crossings, reduced-word letters, generator-0 share)."""
    ok = 0; tot = 0; ncr = 0; red = 0; g0 = 0
    for p, s, u, t in runs:
        p = p.astype(int); L = (s.astype(int) * u.astype(int)) > 0       # True if a is left after the swap
        n = p.size; tot += 1; ncr += n
        if n == 0:
            ok += 1; continue
        found = None
        for order in itertools.permutations(range(3)):
            pos = {d: i for i, d in enumerate(order)}; gens = np.empty(n, int); good = True
            for k in range(n):
                a, b = PA[p[k]], PB[p[k]]
                if abs(pos[a] - pos[b]) != 1:
                    good = False; break
                gens[k] = min(pos[a], pos[b]); pos[a], pos[b] = pos[b], pos[a]
                if (pos[a] < pos[b]) != L[k]:
                    good = False; break
            if good:
                found = gens; break
        if found is None:
            continue
        ok += 1; g0 += int((found == 0).sum())
        stack = []
        for g_, sg in zip(found, s.astype(int)):
            if stack and stack[-1] == (g_, -sg):
                stack.pop()
            else:
                stack.append((g_, sg))
        red += len(stack)
    return ok, tot, ncr, red, g0