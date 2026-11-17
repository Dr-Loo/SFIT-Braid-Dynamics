#!/usr/bin/env python3
"""Stage 5G: fixed-(u,v) adjacent-gap four-arm response diagnostic.

For each common heading/mover path, compare exactly four complete trajectories:
    (u,v), (u,0), (0,v), (0,0).
Other relaxation gaps are zero.  Compute the *samplewise* mixed differences:
    D_square = S_uv**2 - S_u0**2 - S_0v**2 + S_00**2
    D_count  = N_uv    - N_u0    - N_0v    + N_00
The finite-window mixed difference of G_M is E[D_square]/M.

This is NOT importance sampling and NOT a proof of Eq. 17. No geometric
fast-rejection shortcut is used: the suggested shortcut was invalid for both
crossing deletions and transverse sign flips. Fixed tiny (u,v) values produce
rare events, so '0 observed' is under-resolution, not an exact zero.

Examples:
    python stage5g_adjacent_fixed_uv.py --samples 32768 --groups 8
    python stage5g_adjacent_fixed_uv.py --u 0.1 0.01 --v 0.1 0.01 --samples 131072
    python stage5g_adjacent_fixed_uv.py --u 0.1 --v 0.1 --samples 100000 --before 20 --after 20
    python stage5g_adjacent_fixed_uv.py --samples 200000 --groups 12 --json stage5g_fixed_uv.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PAIRS = ((0, 1), (0, 2), (1, 2))


def interval_reward_count(start: np.ndarray, target: np.ndarray,
                          q: float, k: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Exact harmonic-trap crossing rewards in a single interval.

    start/target: (n,3,2). Each pair separation is a+b*exp(-t).
    q: interval endpoint factor exp(-interval_length), in [0,1].
    """
    n = start.shape[0]
    reward = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.float64)
    for a,b in PAIRS:
        dx_target = target[:,a,0] - target[:,b,0]
        dx_start = start[:,a,0] - start[:,b,0]
        denominator = dx_start - dx_target
        root = np.full(n, 2.0, dtype=np.float64)
        valid = np.abs(denominator) > 1e-14
        np.divide(-dx_target, denominator, out=root, where=valid)
        hit = (q < root) & (root < 1.0)
        count += hit
        if k in (a,b):
            dy_target = target[:,a,1] - target[:,b,1]
            dy_start = start[:,a,1] - start[:,b,1]
            dy_at_root = dy_target + root * (dy_start - dy_target)
            coeff = 1 if k == a else -1
            reward += coeff * hit * np.sign(dy_at_root)
    return reward, count


def four_arm_batch(u: float, v: float, n: int, rng: np.random.Generator,
                   before: int, after: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Paired four-arm sample-wise D_square, D_count for n heading histories."""
    angles = rng.uniform(-math.pi, math.pi, size=(n,3))
    target = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    # All four arms begin with the same positions and share future headings.
    positions = np.broadcast_to(target, (4,n,3,2)).copy()
    total_reward = np.zeros((4,n), dtype=np.float64)
    total_count = np.zeros((4,n), dtype=np.float64)
    row_idx = np.arange(n)

    # Arm order is uv, u0, 0v, 00.
    q_a = (u,u,0.0,0.0)
    q_b = (v,0.0,v,0.0)
    M = before + 2 + after
    for t in range(M):
        mover = rng.integers(0,3,size=n)
        new_angle = rng.uniform(-math.pi, math.pi, size=n)
        # New target is determined by refreshed heading, not current position.
        target[row_idx,mover,0] = np.cos(new_angle)
        target[row_idx,mover,1] = np.sin(new_angle)
        for arm in range(4):
            q = (q_a[arm] if t == before else
                 q_b[arm] if t == before + 1 else 0.0)
            rw, ct = interval_reward_count(positions[arm], target, q, k=k)
            total_reward[arm] += rw
            total_count[arm] += ct
            positions[arm] = target + q*(positions[arm]-target)

    # These are the EXACT sample-by-sample single-gap subtractions.
    # Do not replace them with R_a * Delta_R_b; that is a different object.
    S_uv, S_u0, S_0v, S_00 = total_reward
    N_uv, N_u0, N_0v, N_00 = total_count
    sq = S_uv**2 - S_u0**2 - S_0v**2 + S_00**2
    ct = N_uv - N_u0 - N_0v + N_00
    return sq, ct


def w(x: float) -> float:
    return x*(1+math.log(1/x)) if x > 0 else 0.0


def summarize(u: float, v: float, samples: int, groups: int,
              batch: int, before: int, after: int, k: int,
              seed: int) -> dict:
    # Same seed across grid cells yields common heading paths for
    # equal batch/group partitioning (paired comparisons across u/v).
    seeds = np.random.SeedSequence(seed).spawn(groups)
    counts_per_group = [samples//groups + int(g < samples%groups)
                        for g in range(groups)]
    means_sq, means_ct = [], []
    nonzero_sq, nonzero_ct, groups_nonzero = 0,0,0
    max_abs_sq = 0.0
    for ss, ng in zip(seeds, counts_per_group):
        rng = np.random.default_rng(ss)
        sum_sq, sum_ct, g_nonzero_sq = 0.0,0.0,0
        for offset in range(0, ng, batch):
            n = min(batch, ng-offset)
            sq,ct = four_arm_batch(u,v,n,rng,before,after,k)
            sum_sq += float(sq.sum())
            sum_ct += float(ct.sum())
            g_nonzero_sq += int(np.count_nonzero(sq))
            nonzero_ct += int(np.count_nonzero(ct))
            if sq.size:
                max_abs_sq = max(max_abs_sq, float(np.max(np.abs(sq))))
        nonzero_sq += g_nonzero_sq
        groups_nonzero += int(g_nonzero_sq>0)
        means_sq.append(sum_sq/ng)
        means_ct.append(sum_ct/ng)
    sq_means = np.asarray(means_sq)
    ct_means = np.asarray(means_ct)
    sq_mean = float(sq_means.mean())
    ct_mean = float(ct_means.mean())
    sq_se = float(sq_means.std(ddof=1)/math.sqrt(groups))
    ct_se = float(ct_means.std(ddof=1)/math.sqrt(groups))
    M = before+2+after
    wmin = w(min(u,v))
    wproduct = w(u)*w(v)
    unresolved = (nonzero_sq<30 or groups_nonzero<min(groups,5))
    return {
        'u':u, 'v':v, 'samples':samples, 'groups':groups, 'window':M,
        'mixed_square':sq_mean, 'se_mixed_square':sq_se,
        'mixed_G_M':sq_mean/M, 'se_mixed_G_M':sq_se/M,
        'mixed_count':ct_mean, 'se_mixed_count':ct_se,
        'w_min':wmin, 'w_product':wproduct,
        'ratio_min':sq_mean/wmin if wmin else None,
        'se_ratio_min':sq_se/wmin if wmin else None,
        'ratio_product':sq_mean/wproduct if wproduct else None,
        'se_ratio_product':sq_se/wproduct if wproduct else None,
        'nonzero_square_paths':nonzero_sq,
        'nonzero_count_paths':nonzero_ct,
        'nonzero_groups':groups_nonzero,
        'max_abs_square':max_abs_sq,
        'underresolved':unresolved,
        'z_abs':abs(sq_mean)/sq_se if sq_se>0 else None,
    }


def fmt_ratio(v: float | None, se: float | None, unresolved: bool) -> str:
    if v is None:
        return 'n/a'
    if unresolved and se == 0:
        return '0 obs (SE n/a)'
    return f'{v:+.3g} +/- {se:.2g}'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--u',type=float,nargs='+',default=[.1,.01,.001,.0001,.00001],help='Fixed first-gap relaxation factors')
    p.add_argument('--v',type=float,nargs='+',default=[.1,.01,.001,.0001,.00001],help='Fixed second-gap relaxation factors')
    p.add_argument('--samples',type=int,default=16384,help='Heading histories per (u,v) grid cell')
    p.add_argument('--groups',type=int,default=8)
    p.add_argument('--batch',type=int,default=4096)
    p.add_argument('--before',type=int,default=8)
    p.add_argument('--after',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261115)
    p.add_argument('--k',type=int,choices=(0,1,2),default=0)
    p.add_argument('--json',type=str,default='',help='Optional JSON results filename')
    a = p.parse_args()
    if a.samples < a.groups or a.groups<2 or a.batch<1 or a.before<0 or a.after<0:
        p.error('samples>=groups>=2, batch>=1 and nonnegative before/after required')
    if any(not 0<=q<1 for q in a.u+a.v):
        p.error('All u and v must be in [0,1)')
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print('Stage 5G fixed-(u,v) adjacent four-arm diagnostic')
    print('Script SHA-256:', digest)
    print('NOT importance sampling. Other q=0; finite window; NO theorem verdict.')
    print('K = E[S_uv^2-S_u0^2-S_0v^2+S_00^2],  DeltaDelta G_M = K/M')
    print('Rows marked sparse are NOT resolved, even when 0 is observed.')
    print('u          v          K +/- SE               K/w(min) +/- SE           K/[w(u)w(v)] +/- SE     hits    status')
    print('-'*132)
    results=[]
    for u in a.u:
        for v in a.v:
            d=summarize(u,v,a.samples,a.groups,a.batch,a.before,a.after,a.k,a.seed)
            results.append(d)
            if d['nonzero_square_paths']==0:
                K_text='0 observed (SE n/a)'
            else:
                K_text=f'{d["mixed_square"]:+.5g} +/- {d["se_mixed_square"]:.2g}'
            print(f'{u:<10.5g} {v:<10.5g} {K_text:<23} '
                  f'{fmt_ratio(d["ratio_min"],d["se_ratio_min"],d["underresolved"]):<26} '
                  f'{fmt_ratio(d["ratio_product"],d["se_ratio_product"],d["underresolved"]):<29} '
                  f'{d["nonzero_square_paths"]:>5}  '
                  f'{"SPARSE" if d["underresolved"] else "sampled"}')
    if a.json:
        Path(a.json).write_text(json.dumps({'script_sha256':digest,
             'arguments':vars(a),'results':results},indent=2)+'\n')
        print('Saved:',a.json)
    print('Interpretation: neither a finite grid nor apparent ratio stability proves a uniform bound.')
    print('Both w(min) and w(u)w(v) are integrable with respect to du/u dv/v;')
    print('the latter is a stronger conjecture, not a requirement for an O(r^2) integral.')


if __name__ == '__main__':
    main()
