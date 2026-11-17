#!/usr/bin/env python3
"""Stage 5G: exact event-driven, fresh-seed prospective crossing-count test.

At each global tumble one of three preferred unit-circle targets is renewed.
All defects relax to their preferred targets exponentially until the next
renewal. For each pair the longitudinal gap is affine in q=exp(-t), so the
interval can contain at most one crossing, found without discretization.

Run predictions freeze separately BEFORE running the primary test.
Example:
  python stage5g_nu_prospective.py --mode smoke
  python stage5g_nu_prospective.py --mode primary
The JSON predictions file is read-only; it is never recalibrated.
"""
import argparse, csv, hashlib, json, math, time
from pathlib import Path
import numpy as np
from numba import njit

@njit

def _group_exact_count(r, n_tumbles, burnin, seed):
    np.random.seed(seed)
    X = np.empty((3,2),dtype=np.float64)
    T = np.empty((3,2),dtype=np.float64)
    for i in range(3):
        ang=2*math.pi*np.random.random()-math.pi
        T[i,0]=math.cos(ang); T[i,1]=math.sin(ang)
        X[i,0]=T[i,0]; X[i,1]=T[i,1]
    crossing=0
    countsq=0
    for n in range(burnin+n_tumbles):
        mover=np.random.randint(0,3)
        angle=2*math.pi*np.random.random()-math.pi
        T[mover,0]=math.cos(angle); T[mover,1]=math.sin(angle)
        u=np.random.random()
        q=math.exp(math.log(u)/(3*r)) if u>0 else 0.0
        this_count=0
        for i in range(2):
            for j in range(i+1,3):
                a=T[i,0]-T[j,0]
                b=(X[i,0]-X[j,0])-a
                if b!=0.0:
                    root=-a/b
                    if q<root<1.0:
                        this_count+=1
        if n>=burnin:
            crossing+=this_count
            countsq+=this_count*this_count
        for i in range(3):
            for d in range(2):
                X[i,d]=T[i,d]+q*(X[i,d]-T[i,d])
    return crossing, countsq


def get_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode',choices=('smoke','primary'),default='smoke')
    ap.add_argument('--predictions',default='stage5g_nu_predictions_frozen.json')
    ap.add_argument('--output',default='stage5g_nu_newseed_results.json')
    ap.add_argument('--group-csv',default='stage5g_nu_newseed_groups.csv')
    ap.add_argument('--groups',type=int,default=None)
    ap.add_argument('--tumbles',type=int,default=None,help='Measured tumbles per independent group')
    ap.add_argument('--burnin',type=int,default=None)
    a=ap.parse_args()
    predfile=Path(a.predictions)
    if not predfile.is_file(): ap.error('Missing frozen predictions file: '+str(predfile))
    pred=json.loads(predfile.read_text())
    rs=list(map(float,pred['r_values']))
    n_groups=a.groups if a.groups is not None else (4 if a.mode=='smoke' else 20)
    tumbles=a.tumbles if a.tumbles is not None else (100000 if a.mode=='smoke' else 2000000)
    burnin=a.burnin if a.burnin is not None else 2000
    if n_groups<2 or tumbles<1 or burnin<0: ap.error('Invalid groups/tumbles/burnin')
    if a.mode=='primary' and (n_groups<20 or tumbles<2000000):
        ap.error('Primary protocol requires >=20 groups and >=2,000,000 tumbles/group')
    seed=int(pred['primary_seed'] if a.mode=='primary' else pred['smoke_seed'])
    seedseq=np.random.SeedSequence(seed)
    child=seedseq.spawn(len(rs)*n_groups)
    seeds=[int(s.generate_state(1,dtype=np.uint32)[0]%(2**31-1)) for s in child]
    print('Stage 5G prospective nu, exact event-driven crossing counts;',a.mode.upper())
    print('Script SHA256:',get_sha(__file__))
    print('Frozen prediction SHA256:',get_sha(predfile))
    print('groups=',n_groups,'tumbles/group=',tumbles,'burnin=',burnin,'seed root=',seed)
    t0=time.time(); raw=[]; summary=[]
    for ir,r in enumerate(rs):
        vals=[]
        for ig in range(n_groups):
            s=seeds[ir*n_groups+ig]
            nc,nc2=_group_exact_count(r,tumbles,burnin,s)
            mu=nc/tumbles
            vals.append(mu)
            raw.append({'r':r,'group':ig+1,'seed':s,'tumbles':tumbles,'crossings':int(nc),'nu':mu,'E_N2':nc2/tumbles})
        vals=np.array(vals)
        mean=float(np.mean(vals)); se=float(np.std(vals,ddof=1)/math.sqrt(n_groups))
        intercept=float(pred['nu0']); nu1=float(pred['nu1']); nu2=float(pred['nu2_candidate'])
        lin=intercept+nu1*r; quad=lin+nu2*r*r
        diff=mean-quad
        nu2eff=(mean-lin)/(r*r)
        z=diff/math.hypot(se,math.hypot(float(pred['nu1_se'])*r,float(pred['nu2_se'])*r*r))
        summary.append({'r':r,'nu_measured':mean,'se_between_groups':se,
            'nu_linear_pred':lin,'nu_quadratic_pred_candidate':quad,
            'residual_to_quad':diff,'z_vs_quad':z,'nu2_effective':nu2eff,
            'n_groups':n_groups,'tumbles_per_group':tumbles})
        print(f'r={r:0.3f} nu={mean:.7f} +/- {se:.7f} | linear={lin:.7f} candidate_quad={quad:.7f} | z={z:+.2f} | nu2_eff={nu2eff:.3f}')
    results={'status':'prospective fresh seed, candidate nu2 (not proved)','mode':a.mode,
      'script_sha256':get_sha(__file__),'predictions_sha256':get_sha(predfile),
      'r_values':rs,'groups':n_groups,'tumbles_per_group':tumbles,'burnin':burnin,
      'root_seed':seed,'elapsed_seconds':time.time()-t0,'results':summary}
    # Smoke results deliberately not written to primary-named files.
    out=Path(a.output if a.mode=='primary' else a.output.replace('.json','_SMOKE.json'))
    rawpath=Path(a.group_csv if a.mode=='primary' else a.group_csv.replace('.csv','_SMOKE.csv'))
    out.write_text(json.dumps(results,indent=2)+'\n')
    with rawpath.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(raw[0].keys())); writer.writeheader();writer.writerows(raw)
    print('Saved',out,rawpath)
if __name__=='__main__': main()
