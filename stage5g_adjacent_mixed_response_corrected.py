#!/usr/bin/env python3
"""Stage 5G diagnostic: adjacent, isolated two-gap mixed differences.

This script estimates Delta_a Delta_b G_M for the full paired four-arm
cumulative-reward *squared* functional, with q=0 at all other tumble gaps.
It is a DIAGNOSTIC, NOT a proof of uniform multi-insertion bounds at finite r.

Examples:
  python stage5g_adjacent_mixed_response_corrected.py
  python stage5g_adjacent_mixed_response_corrected.py --samples 200000 --groups 12
  python stage5g_adjacent_mixed_response_corrected.py --r 0.01 0.005 0.002 0.001 --samples 400000

Time scale K=1, target radius c/K=1; 3 equally likely independently
refreshed headings, global gaps q ~ Beta(3r,1).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

PAIRS = ((0, 1), (0, 2), (1, 2))


def rewards_and_counts(start: np.ndarray, target: np.ndarray, q: np.ndarray,
                       k: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Exact interval crossing root, first start to target with q_end=q.

    start,target: (batch,3,2), q: (batch,). Each longitudinal pair can
    cross at most once per relaxation interval.
    """
    batch = start.shape[0]
    reward = np.zeros(batch, dtype=np.float64)
    count = np.zeros(batch, dtype=np.float64)
    for a,b in PAIRS:
        da = target[:, a, 0] - target[:, b, 0]
        db = (start[:, a, 0] - start[:, b, 0]) - da
        root = np.empty(batch, dtype=np.float64)
        root.fill(2.)
        valid = np.abs(db) > 1e-14
        np.divide(-da, db, out=root, where=valid)
        hit = (root > q) & (root < 1.)
        count += hit
        if k == a or k == b:
            dy = ((target[:, a, 1] - target[:, b, 1])
                  + root * ((start[:, a, 1] - start[:, b, 1])
                            - (target[:, a, 1] - target[:, b, 1])))
            coeff = 1 if k == a else -1
            reward += coeff * hit * np.sign(dy)
    return reward, count


def one_batch(r: float, n: int, rng: np.random.Generator,
              before: int, after: int, k: int) -> tuple[np.ndarray,np.ndarray]:
    """Return paired per-row mixed-squared-reward and mixed crossing count.

    Four arms share all headings/choices: (u,v), (u,0), (0,v), (0,0).
    For other intervals q=0. Includes ballistic pre/post rewards, and
    exactly propagates every inserted residual into later intervals.
    """
    u, v = rng.beta(3*r,1,size=(2,n))
    idx = np.arange(n)
    angles = rng.uniform(-np.pi, np.pi, size=(n,3))
    targets = np.stack((np.cos(angles),np.sin(angles)),axis=-1)
    positions = np.broadcast_to(targets, (4,n,3,2)).copy()
    totals = np.zeros((4,n),dtype=np.float64)
    counts = np.zeros((4,n),dtype=np.float64)
    insert_a=before
    insert_b=before+1
    M = before+2+after
    for t in range(M):
        mover = rng.integers(0,3,size=n)
        angle = rng.uniform(-np.pi,np.pi,size=n)
        targets[idx,mover,0]=np.cos(angle)
        targets[idx,mover,1]=np.sin(angle)
        for arm in range(4):
            if t == insert_a and arm in (0,1):
                q=u
            elif t == insert_b and arm in (0,2):
                q=v
            else:
                q=np.zeros(n,dtype=np.float64)
            rw, ct = rewards_and_counts(positions[arm], targets, q, k=k)
            totals[arm] +=rw
            counts[arm] +=ct
            positions[arm] = targets+q[:,None,None]*(positions[arm]-targets)
    # paired (uv) - (u0) - (0v) + (00), in the specified order.
    mixed_square = (totals[0]**2 - totals[1]**2
                   - totals[2]**2 + totals[3]**2)
    mixed_count = counts[0]-counts[1]-counts[2]+counts[3]
    # mixed_square / M is Delta_a Delta_b G_M for this finite window.
    return mixed_square, mixed_count


def estimate(r: float, total_n: int, groups: int, batch: int,
             before: int, after: int, k:int, seed:int) -> dict:
    seeds = np.random.SeedSequence(seed).spawn(groups)
    ns = [total_n//groups+(g < total_n%groups) for g in range(groups)]
    squares = []
    counts = []
    for g,(ss, ng) in enumerate(zip(seeds,ns)):
        rng=np.random.default_rng(ss)
        xs=[]; ys=[]
        for start in range(0,ng,batch):
            n=min(batch,ng-start)
            x,y=one_batch(r,n,rng,before,after,k)
            xs.append(x); ys.append(y)
        squares.append(np.concatenate(xs).mean())
        counts.append(np.concatenate(ys).mean())
    sq=np.asarray(squares); co=np.asarray(counts)
    se_sq=np.std(sq,ddof=1)/np.sqrt(groups)
    se_co=np.std(co,ddof=1)/np.sqrt(groups)
    scale=r*r*math.log(1/r)**2
    M=before+2+after
    return dict(r=r,samples=total_n,window=M,
                mixed_square=float(sq.mean()),se_mixed_square=float(se_sq),
                mixed_GM=float(sq.mean()/M),se_mixed_GM=float(se_sq/M),
                mixed_count=float(co.mean()),se_mixed_count=float(se_co),
                scale=scale,ratio=float(sq.mean()/scale),
                ratio_se=float(se_sq/scale),
                signal_to_noise=float(abs(sq.mean())/se_sq) if se_sq else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--r',nargs='+',type=float,default=[0.01,0.005,0.002,0.001])
    parser.add_argument('--samples',type=int,default=65536,help='Total independent paths per r')
    parser.add_argument('--groups',type=int,default=8,help='Independent groups for SE')
    parser.add_argument('--batch',type=int,default=8192,help='Batch to cap memory use')
    parser.add_argument('--before',type=int,default=12,help='Ballistic intervals preceding the adjacent gaps')
    parser.add_argument('--after',type=int,default=12,help='Ballistic intervals following the adjacent gaps')
    parser.add_argument('--k',type=int,default=0,choices=[0,1,2])
    parser.add_argument('--seed',type=int,default=20261115)
    parser.add_argument('--json',type=str,default='',help='Optional saved JSON output file')
    args=parser.parse_args()
    assert all(0<r<1 for r in args.r)
    assert args.groups>=2 and args.samples>=args.groups
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print('Script SHA-256:',digest)
    print('Four-arm mixed finite difference of G_M (isolated adjacent gaps)')
    print('IMPORTANT: finite-window, other q=0; NOT the full finite-r stationary cluster estimate.')
    print('r       E[DeltaDelta S^2] +/- SE      /r^2ln^2 +/- SE    |z|  E[DeltaDelta count] +/- SE')
    out=[]
    for r in args.r:
        d=estimate(r,args.samples,args.groups,args.batch,args.before,args.after,args.k,args.seed+int(r*1e8))
        out.append(d)
        print(f'{r:.4f}  {d["mixed_square"]:+.6g} +/- {d["se_mixed_square"]:.2g}    '
              f'{d["ratio"]:+.4g} +/- {d["ratio_se"]:.2g}    '
              f'{d["signal_to_noise"]:.2f}    '
              f'{d["mixed_count"]:+.4g} +/- {d["se_mixed_count"]:.2g}')
    if args.json:
        Path(args.json).write_text(json.dumps(dict(script_sha256=digest,arguments=vars(args),results=out),indent=2)+'\n')
    print('No finite set of r values can prove or disprove a uniform asymptotic bound.')
    print('Compare --before/--after settings; look for robust signal above MC noise.')

if __name__=='__main__':
    main()
