#!/usr/bin/env python3
"""SFIT Part III: neutral exchange / absorption audit (candidate model, v0.1).

The proposed carrier has M discrete capacity units and channel shares x_k=n_k/M.
At each exchange step n' is sampled from Multinomial(M, x). Thus:
  E[x'_k | F_t] = x_k, sum_k x_k=1, and simplex vertices are absorbing.
The first absorbing vertex defines the record.

IMPORTANT: this multinomial exchange kernel is an additional Part III
hypothesis. It is NOT derived from the Part I braid representation.
Coincidence alpha is deliberately NOT computed: an event-time/detector law
is absent. Defining one global absorbing record would force alpha=0 by
construction and would not be evidence for coincidence suppression.
"""
import argparse, json, math, platform, sys
from pathlib import Path
import numpy as np

STATES = {
    "S2a": [0.2, 0.8], "S2b": [0.5, 0.5],
    "S3a": [0.5, 0.3, 0.2], "S3b": [1/3, 1/3, 1/3],
    "S3c": [0.7, 0.2, 0.1],
    "S4a": [0.4, 0.3, 0.2, 0.1], "S4b": [0.25]*4,
}

def allocate(p, M):
    raw = np.asarray(p)*M
    n = np.floor(raw).astype(int)
    rem = M-int(n.sum())
    if rem:
        n[np.argsort(-(raw-n))[:rem]] += 1
    return n

def wilson(k, n, z=2.576):
    if n == 0: return [None, None]
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-h), min(1,c+h)]

def tv(a,b):
    return float(0.5*np.abs(np.asarray(a)-np.asarray(b)).sum())

def run_state(p, trials, M, max_steps, rng):
    p=np.asarray(p,dtype=float)
    nchan=len(p); n0=allocate(p,M); x0=n0/M
    counts=np.repeat(n0[None,:],trials,axis=0)
    outcome=np.full(trials,-1,dtype=int)
    steps=np.full(trials,-1,dtype=int)
    active=np.ones(trials,dtype=bool)
    for step in range(1,max_steps+1):
        ids=np.flatnonzero(active)
        if not len(ids): break
        # Vectorized sequential-binomial construction of a multinomial draw.
        # This is exact and avoids one Python-level RNG call per trajectory.
        probs=counts[ids]/M
        nxt=np.zeros((len(ids),nchan),dtype=int)
        remaining_n=np.full(len(ids),M,dtype=int)
        remaining_p=np.ones(len(ids),dtype=float)
        for k in range(nchan-1):
            pk=np.divide(probs[:,k],remaining_p,
                         out=np.zeros_like(remaining_p),where=remaining_p>0)
            pk=np.clip(pk,0.0,1.0)
            draw=rng.binomial(remaining_n,pk)
            nxt[:,k]=draw
            remaining_n-=draw
            remaining_p-=probs[:,k]
        nxt[:,-1]=remaining_n
        counts[ids]=nxt
        hit=(nxt>0).sum(axis=1)==1
        if np.any(hit):
            hids=ids[hit]
            outcome[hids]=nxt[hit].argmax(axis=1)
            steps[hids]=step
            active[hids]=False
    ok=outcome>=0; nr=int(ok.sum())
    oc=np.bincount(outcome[ok],minlength=nchan) if nr else np.zeros(nchan,dtype=int)
    freq=oc/nr if nr else np.full(nchan,np.nan)
    ci=[wilson(int(k),nr) for k in oc]
    # Independent one-step conditional-drift audit at x0.
    nd=min(50000,max(5000,trials))
    # Vectorized exact multinomial sample for the drift audit.
    probs=np.repeat(x0[None,:],nd,axis=0)
    draws=np.zeros((nd,nchan),dtype=float)
    rem_n=np.full(nd,M,dtype=int); rem_p=np.ones(nd)
    for k in range(nchan-1):
        pk=np.divide(probs[:,k],rem_p,out=np.zeros_like(rem_p),where=rem_p>0)
        d=rng.binomial(rem_n,np.clip(pk,0,1))
        draws[:,k]=d/M
        rem_n-=d; rem_p-=probs[:,k]
    draws[:,-1]=rem_n/M
    delta=draws.mean(axis=0)-x0
    se=draws.std(axis=0,ddof=1)/math.sqrt(nd)
    st=steps[ok]
    return {
      "target":p.tolist(),"initial_counts":n0.tolist(),"initial_shares":x0.tolist(),
      "capacity":M,"trials":trials,"resolved":nr,"unresolved":trials-nr,
      "resolved_fraction":nr/trials,"outcome_counts":oc.tolist(),
      "outcome_frequencies":freq.tolist(),"wilson_99":ci,
      "tv_vs_quantized_initial":tv(freq,x0) if nr else None,
      "tv_vs_requested_target":tv(freq,p) if nr else None,
      "mean_absorption_steps":float(st.mean()) if len(st) else None,
      "median_absorption_steps":float(np.median(st)) if len(st) else None,
      "conditional_drift_mean_delta":delta.tolist(),
      "conditional_drift_se":se.tolist(),
      "conditional_drift_z":(delta/np.maximum(se,1e-15)).tolist(),
      "energy_conservation_error":0.0,
      "alpha":None,"alpha_status":"NOT MODELED: no detector event-time law"
    }

def alpha_independent_control(nchan=3, pfire=.2, trials=100000, seed=7):
    rng=np.random.default_rng(seed)
    f=rng.random((trials,nchan))<pfire
    pi=f.mean(axis=0); num=den=0.0
    for i in range(nchan):
      for j in range(nchan):
        if i==j: continue
        num+=np.mean(f[:,i]&f[:,j]); den+=pi[i]*pi[j]
    return {"definition":"all-trials joint firing (not time-window coincidence)",
            "P_i":pi.tolist(),"alpha":float(num/den),"expected_alpha":1.0}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials",type=int,default=20000)
    ap.add_argument("--capacity",type=int,default=2000)
    ap.add_argument("--max-steps",type=int,default=20000)
    ap.add_argument("--seed",type=int,default=20261029)
    ap.add_argument("--states",nargs="*",default=list(STATES))
    ap.add_argument("--output",default="sfit_part_iii_exchange_audit.json")
    a=ap.parse_args()
    if a.trials<1 or a.capacity<2 or a.max_steps<1: ap.error("positive trials/max-steps and capacity >=2 required")
    if set(a.states)-set(STATES): ap.error(f"unknown states: {sorted(set(a.states)-set(STATES))}")
    ss=np.random.SeedSequence(a.seed).spawn(len(a.states)+1)
    results={name:run_state(STATES[name],a.trials,a.capacity,a.max_steps,
                            np.random.default_rng(ss[i]))
             for i,name in enumerate(a.states)}
    control=alpha_independent_control(seed=a.seed+1)
    payload={
      "title":"SFIT Part III — neutral exchange / absorption audit",
      "version":"0.1","status":"candidate exchange law; not derived from Part I",
      "configuration":{"trials":a.trials,"capacity":a.capacity,
                       "max_steps":a.max_steps,"seed":a.seed},
      "runtime":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__},
      "model":{
        "state":"integer loads n_k; x_k=n_k/M; sum n_k=M",
        "exchange":"n' ~ Multinomial(M,x)",
        "conditional_mean":"E[x'_k | F_t]=x_k",
        "conserved":"total capacity M",
        "absorbing":"simplex vertices",
        "record":"first channel at global absorbing vertex",
        "caveat":"Multinomial exchange is an extra Part III postulate, not a Part I consequence."
      },
      "results":results,
      "coincidence":{
        "sfit_alpha":None,
        "status":"UNRESOLVED: detector event-time/energy law absent. A single global absorbing record would force alpha=0 by construction.",
        "independent_bernoulli_control":control
      },
      "interpretation":{
        "born_result":"For a bounded martingale absorbed at a vertex, optional stopping gives P(K=k)=x_k(0), subject to finite-capacity quantization and unresolved trajectories.",
        "not_established":["derivation of exchange kernel from braid dynamics",
          "physical identity of the exchanged energy-bearing variable",
          "detector event timing and energy uptake",
          "time-window coincidence suppression alpha<1",
          "full carrier-detector back-reaction"]
      }
    }
    Path(a.output).write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print("SFIT PART III — NEUTRAL EXCHANGE / ABSORPTION AUDIT")
    print(f"trials={a.trials:,}; capacity={a.capacity:,}; max_steps={a.max_steps:,}")
    print("state  target                  observed                TV(target) resolved")
    for name,r in results.items():
      print(f"{name:5s} {str(np.round(r['target'],3)):23s} {str(np.round(r['outcome_frequencies'],3)):23s} "
            f"{r['tv_vs_requested_target']!s:10s} {r['resolved_fraction']:.4f}")
    print(f"independent Bernoulli alpha control={control['alpha']:.4f} (expected 1)")
    print("SFIT alpha: NOT MODELED — detector event law is still missing.")
    print(f"wrote {Path(a.output).resolve()}")

if __name__=="__main__": main()
