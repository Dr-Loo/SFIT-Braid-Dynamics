#!/usr/bin/env python3
"""Stage 5G: exact ballistic-tail mixed response for two isolated gaps.

K_inf(u,v;s) is the infinite-window expectation of the four-arm squared
cumulative-reward difference for two fixed nonzero relaxation factors,
with ALL other factors zero.  The first gap is at interval 0 and the second
at interval s+1; interval s+2 completes all residual flights (q=0).

Four arms use IDENTICAL headings and mover choices: (u,v),(u,0),(0,v),(0,0).
The finite local reward sums L_ab have an exact common past/future tail:

 K_inf = E[DeltaDelta(L^2) + 2 DeltaDelta(L)*(H(U)+H(W))],

where U is the heading target immediately before the first tumble and W
is the target after the completion tumble. H=(I-P)^(-1)h is the exact
ballistic Poisson resolvent, H=3*h_1+3/2*h_2. Unlike the old script,
DO NOT divide K_inf by the number of intervals.

For s>=1, the two insertion blocks do not overlap, and the local total
reward is additive, DeltaDelta(L)=0 *samplewise*. Thus
 K_inf(u,v;s)=2*E[(L_u0-L_00)*(L_0v-L_00)].

Optionally test finite windows 12,24 (or chosen lengths) with the EXACT
truncated ballistic resolvent H_N = sum_{j=0}^{N-1}P^j h. This computes
the expected effect of N pre-steps and N post-steps without sampling noisy
ballistic tails. It converges at rates (2/3)^N and (1/3)^N.

The program is a diagnostic of isolated two-gap interactions, NOT a proof
of any uniform bound for the stationary finite-r model. Ordinary MC
sampling of small u/v can yield very sparse nonzero responses.

Example:
 python stage5g_mixed_resolvent.py --u 0.1 0.05 --v 0.1 0.05 \
        --separation 0 1 2 --samples 131072 --groups 8 --json stage5g_mixed.json
 python stage5g_mixed_resolvent.py --u 0.1 --v 0.1 \
        --separation 0 1 2 --check-windows 12 24 --samples 131072
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PAIRS = ((0, 1), (0, 2), (1, 2))


def marks(start: np.ndarray, target: np.ndarray, q: float,
          k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact signed reward and total crossing count during a trap interval.

    Start and target arrays have shape (batch,3,2). q = exp(-K*tau).
    Each longitudinal pair has at most one crossing in the interval.
    """
    n = len(start)
    reward = np.zeros(n, dtype=float)
    count = np.zeros(n, dtype=float)
    for a,b in PAIRS:
        a1 = target[:,a,0]-target[:,b,0]
        a0 = start[:,a,0]-start[:,b,0]
        denom = a0-a1
        root = np.full(n,2.0)
        safe = denom != 0.
        np.divide(-a1,denom,out=root,where=safe)
        hits = (root>q)&(root<1.)
        count += hits
        if k in (a,b):
            y1 = target[:,a,1]-target[:,b,1]
            y0 = start[:,a,1]-start[:,b,1]
            # Use a harmless value for root when there is no crossing.
            x = np.where(hits,root,0.)
            sign = np.sign(y1+x*(y0-y1))
            reward += (1 if k==a else -1)*hits*sign
    return reward,count


def H_ballistic(positions: np.ndarray,k:int,steps:int|None=None) -> np.ndarray:
    """Ballistic reward resolvent, either infinite or truncated to N terms.

    h = h1+h2, with P h1=(2/3)h1, P h2=(1/3)h2.
    H_N = [3(1-(2/3)^N)] h1 + [3/2(1-(1/3)^N)] h2.
    N=None corresponds to the full Poisson resolvent.
    """
    x=positions[:,:,0]
    y=positions[:,:,1]
    v=.5+np.arcsin(np.clip(x,-1.,1.))/np.pi
    eps=np.sign(y)
    other=[j for j in range(3) if j!=k]
    p_ik=np.where(v[:,other]<v[:,[k]],1-v[:,[k]],v[:,[k]])
    p_ki=np.where(v[:,[k]]<v[:,other],1-v[:,other],v[:,other])
    h=(eps[:,k]*np.sum(p_ik,axis=1)-np.sum(eps[:,other]*p_ki,axis=1))/3.
    h1=(4./3.)*eps[:,k]*v[:,k]*(1-v[:,k])-(2./3.)*np.sum(
        eps[:,other]*v[:,other]*(1-v[:,other]),axis=1)
    h2=h-h1
    if steps is None:
        return 3.*h1+1.5*h2
    return 3.*(1.-(2./3.)**steps)*h1+1.5*(1.-(1./3.)**steps)*h2


def random_heading_positions(rng:np.random.Generator,n:int) -> np.ndarray:
    angles=rng.uniform(-math.pi,math.pi,size=(n,3))
    return np.stack((np.cos(angles),np.sin(angles)),axis=-1)


def refresh(target:np.ndarray,rng:np.random.Generator) -> np.ndarray:
    """Draw one new mover choice and heading for each row, preserving targets."""
    n=len(target)
    idx=rng.integers(0,3,size=n)
    angle=rng.uniform(-math.pi,math.pi,size=n)
    new=target.copy()
    new[np.arange(n),idx,0]=np.cos(angle)
    new[np.arange(n),idx,1]=np.sin(angle)
    return new




def one_batch(u:float,v:float,s:int,n:int,rng:np.random.Generator,
              k:int,windows:list[int],verify:bool=True) -> dict[str,np.ndarray]:
    """Per-row mixed response, with exact H tails and optional finite windows."""
    U=random_heading_positions(rng,n)
    target=U.copy()
    positions=np.broadcast_to(U,(4,n,3,2)).copy()
    S=np.zeros((4,n))
    counts=np.zeros((4,n))

    # First gap at time 0; second at time s+1; completion at s+2.
    # All other intervals are ballistic. Arm ordering: uv,u0,0v,00.
    for t in range(s+3):
        target=refresh(target,rng)
        for arm in range(4):
            q=(u if (t==0 and arm in (0,1)) else
               v if (t==s+1 and arm in (0,2)) else 0.)
            reward,count=marks(positions[arm],target,q,k)
            S[arm]+=reward
            counts[arm]+=count
            positions[arm]=target+q*(positions[arm]-target)

    W=target
    if verify:
        # Completion interval ensures that all arms coincide at W.
        assert all(np.array_equal(positions[arm],W) for arm in range(4))
    S_uv,S_u0,S_0v,S_00=S
    n_uv,n_u0,n_0v,n_00=counts
    mixed_local=(S_uv*S_uv-S_u0*S_u0-S_0v*S_0v+S_00*S_00)
    mixed_total=S_uv-S_u0-S_0v+S_00
    mixed_count=n_uv-n_u0-n_0v+n_00
    htail=H_ballistic(U,k)+H_ballistic(W,k)
    mixed_resolvent=mixed_local+2.*mixed_total*htail
    if s>=1 and verify:
        # Both insertions have disjoint completed intervals: no mixed
        # first difference survives and the cross product is the answer.
        if not np.all(mixed_total==0):
            raise AssertionError('Nonzero mixed first difference for disjoint completed gaps')
        delta_u=S_u0-S_00
        delta_v=S_0v-S_00
        if not np.array_equal(mixed_local,2.*delta_u*delta_v):
            raise AssertionError('Separated-gap cross-product identity failed')
    if verify and (u==0 or v==0):
        assert np.all(mixed_resolvent==0)
        assert np.all(mixed_count==0)
    result={'K':mixed_resolvent,'K_local':mixed_local,'K_tail':2.*mixed_total*htail,
            'D_S':mixed_total,'D_N':mixed_count}
    for t in windows:
        # Exact expectation of a finite +/-t ballistic tail given U,W.
        # This is Rao-Blackwellized; no additional Monte Carlo noise.
        h_t = H_ballistic(U,k,steps=t)+H_ballistic(W,k,steps=t)
        result[f'window_{t}']=mixed_local+2.*mixed_total*h_t
    return result


def summarize(u:float,v:float,s:int,samples:int,groups:int,batch:int,k:int,
              seed:int,windows:list[int]) -> dict:
    group_seeds=np.random.SeedSequence(seed).spawn(groups)
    ns=[samples//groups+int(i<samples%groups) for i in range(groups)]
    keys=['K','K_local','K_tail','D_S','D_N']+[f'window_{t}' for t in windows]
    group_means={key:[] for key in keys}
    nonzero=0
    nonzero_groups=0
    for gs,ng in zip(group_seeds,ns):
        rng=np.random.default_rng(gs)
        sums={key:0. for key in keys}
        group_hits=0
        for offset in range(0,ng,batch):
            n=min(batch,ng-offset)
            one=one_batch(u,v,s,n,rng,k,windows)
            for key in keys:
                sums[key]+=float(one[key].sum())
            group_hits+=int(np.count_nonzero(one['K']))
        nonzero+=group_hits
        nonzero_groups+=int(group_hits>0)
        for key in keys:
            group_means[key].append(sums[key]/ng)
    out={'u':u,'v':v,'separation':s,'samples':samples,'groups':groups,
         'local_intervals':s+3,'nonzero_response_paths':nonzero,
         'nonzero_groups':nonzero_groups,
         'underresolved':bool(nonzero<30 or nonzero_groups<min(5,groups))}
    def stats(x:np.ndarray) -> tuple[float,float]:
        return float(x.mean()),float(x.std(ddof=1)/np.sqrt(groups))
    for key in keys:
        arr=np.array(group_means[key])
        mean,se=stats(arr)
        out[key]=mean
        out['se_'+key]=se
        if key.startswith('window_'):
            a,b=stats(arr-np.array(group_means['K']))
            out[f'{key}_minus_K']=a
            out[f'se_{key}_minus_K']=b
    wmin=w(min(u,v)); wprod=w(u)*w(v)
    out['w_min']=wmin
    out['w_product']=wprod
    out['K_over_wmin']=out['K']/wmin if wmin else None
    out['se_K_over_wmin']=out['se_K']/wmin if wmin else None
    out['K_over_wproduct']=out['K']/wprod if wprod else None
    out['se_K_over_wproduct']=out['se_K']/wprod if wprod else None
    return out


def w(x:float) -> float:
    return x*(1+math.log(1/x)) if x>0 else 0.


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--u',nargs='+',type=float,default=[0.1,0.05,0.01])
    parser.add_argument('--v',nargs='+',type=float,default=[0.1,0.05,0.01])
    parser.add_argument('--separation',nargs='+',type=int,default=[0,1,2],
                        help='Number of ballistic intervals separating two inserted gaps')
    parser.add_argument('--samples',type=int,default=32768)
    parser.add_argument('--groups',type=int,default=8)
    parser.add_argument('--batch',type=int,default=4096)
    parser.add_argument('--k',type=int,choices=[0,1,2],default=0)
    parser.add_argument('--seed',type=int,default=20261116)
    parser.add_argument('--check-windows',nargs='*',type=int,default=[],metavar='N',
                        help='Optional EXACT truncated H_N tail checks (e.g. 12 24)')
    parser.add_argument('--json',default='',help='Optional JSON results file')
    a=parser.parse_args()
    if a.groups<2 or a.samples<a.groups or a.batch<1 or any(s<0 for s in a.separation):
        parser.error('Require groups>=2, samples>=groups, batch>=1, separation>=0')
    if any(not 0<=q<1 for q in a.u+a.v):
        parser.error('Require u,v in [0,1)')
    if any(n<1 for n in a.check_windows):
        parser.error('Finite windows must be positive integers')
    windows=sorted(set(a.check_windows))
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print('Stage 5G fixed-(u,v;s) isolated two-gap kernel with exact ballistic H tails')
    print('SHA-256:',digest)
    print('K=lim_M E[D_M] (UNNORMALIZED). No K/M. Other q=0; no finite-r theorem.')
    if windows: print('Window checks use exact truncated H_N, not random sampled tails.')
    print('s=0 means adjacent insertion gaps; s>=1 means s intervening ballistic intervals.')
    print('u          v          s   K +/- SE                 K/w(min) +/- SE       K/[w(u)w(v)] +/- SE    hits  status')
    print('-'*140)
    results=[]
    for u in a.u:
        for v in a.v:
            for s in a.separation:
                d=summarize(u,v,s,a.samples,a.groups,a.batch,a.k,a.seed,windows)
                results.append(d)
                K_txt=(f'{d["K"]:+.6g} +/- {d["se_K"]:.2g}' if d['nonzero_response_paths']
                       else '0 observed (SE n/a)')
                r_txt=(f'{d["K_over_wmin"]:+.4g} +/- {d["se_K_over_wmin"]:.2g}' if d['w_min'] and d['nonzero_response_paths']
                       else 'n/a')
                rp_txt=(f'{d["K_over_wproduct"]:+.4g} +/- {d["se_K_over_wproduct"]:.2g}' if d['w_product'] and d['nonzero_response_paths']
                       else 'n/a')
                print(f'{u:<10.4g} {v:<10.4g} {s:>1}   {K_txt:<24} {r_txt:<22} {rp_txt:<23} '
                      f'{d["nonzero_response_paths"]:>5}  '
                      f'{"SPARSE" if d["underresolved"] else "sampled"}')
                for n in windows:
                    print(f'      finite +/-{n:>2}: {d[f"window_{n}"]:+.6g} +/- {d[f"se_window_{n}"]:.2g}; '
                          f'paired minus H: {d[f"window_{n}_minus_K"]:+.3g} +/- {d[f"se_window_{n}_minus_K"]:.2g}')
    if a.json:
        Path(a.json).write_text(json.dumps({'script_sha256':digest,'args':vars(a),'results':results},indent=2)+'\n')
        print('Saved:',a.json)
    print('Interpretation: small-u/v cells may be rare-event under-resolved. Finite grids do not prove uniform bounds.')
    print('Independent groups provide numerical SE; no theorem verdict is automated.')


if __name__=='__main__':
    main()
