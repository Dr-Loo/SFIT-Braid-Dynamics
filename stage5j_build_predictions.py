"""
stage5j_build_predictions.py -- builds the FROZEN Stage 5J prediction file from EXPLORATORY data only.
Inputs are the 5E exact value, the 5F primary outputs (r = 0.005..0.04) and the 2026-11-23 exploratory scan
(gamma_scan.py). Writes stage5j_predictions.json containing predictions, tolerances, run plan, frozen seeds,
and SHA-256 of every frozen script. Run once; then timestamp the JSON (Zenodo / git commit) BEFORE any primary run.
"""
import json, hashlib, numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

H = lambda f: hashlib.sha256(open(f, 'rb').read().replace(b'\r\n', b'\n')).hexdigest()   # CRLF-normalized
rng = np.random.default_rng(20261124)
cut = lambda q: int(max(60, np.floor(25/(1 - 3*q/(1+3*q)))))

# ---- exploratory data: (r, Gamma, SE, nu, nu_SE, V, V_SE)
D = [(0.0, 13/30, 0.0, 2/3, 0.0, 46/45, 0.0),
     (0.005, 0.431820, 0.000294, 0.660324, 3.15e-5, 1.010498, 3.90e-4),
     (0.01, 0.430126, 0.000236, 0.654127, 2.59e-5, 0.998798, 3.13e-4),
     (0.02, 0.426567, 0.000299, 0.642094, 3.12e-5, 0.975856, 3.92e-4),
     (0.04, 0.421123, 0.000325, 0.619818, 3.40e-5, 0.935252, 4.13e-4),
     (0.1, 0.406747, 0.000803, 0.563769, 7.67e-5, 0.834469, 9.44e-4),
     (0.25, 0.384344, 0.000417, 0.468202, 9.02e-5, 0.672036, 3.72e-4),
     (0.35, 0.375918, 0.000805, 0.425782, 8.28e-5, 0.603973, 7.06e-4),
     (0.5, 0.371167, 0.000587, 0.378962, 6.59e-5, 0.533958, 4.75e-4),
     (0.75, 0.374722, 0.000836, 0.326323, 1.13e-4, 0.462110, 5.07e-4),
     (1.0, 0.387002, 0.001545, 0.290896, 7.64e-5, 0.419085, 8.96e-4),
     (2.0, 0.450094, 0.001899, 0.215778, 4.18e-5, 0.338093, 8.36e-4),
     (4.0, 0.584507, 0.002785, 0.156753, 4.17e-5, 0.287748, 8.59e-4),
     (8.0, 0.803980, 0.003977, 0.112534, 4.36e-5, 0.255974, 9.62e-4),
     (16.0, 1.154977, 0.004402, 0.080150, 5.49e-5, 0.238576, 7.24e-4)]
D = np.array(D); r, G, Gse, nu, nuse, V, Vse = D.T
x = np.sqrt(r)

# ---- intermediate predictions: primary = natural cubic spline in sqrt(r); alternatives = PCHIP in sqrt(r),
#      and weighted local quadratic in sqrt(r) on the 5 nearest points. sigma_model = max |alt - primary|.
def interps(Gv, xq):
    p = CubicSpline(x, Gv, bc_type='natural')(xq)
    a1 = PchipInterpolator(x, Gv)(xq)
    a2 = []
    for xx in np.atleast_1d(xq):
        idx = np.argsort(abs(x - xx))[:5]; w = 1/np.maximum(Gse[idx], 1e-4)
        a2.append(np.polyval(np.polyfit(x[idx], Gv[idx], 2, w=w), xx))
    return p, a1, np.array(a2)

r_mid = np.array([0.4, 0.5, 0.6, 1.5, 3.0]); xq = np.sqrt(r_mid)
p, a1, a2 = interps(G, xq)
mc = np.array([interps(G + rng.normal(0, 1, len(G))*Gse, xq)[0] for _ in range(2000)])
mid = [{"r": float(rr), "Gamma_pred": float(pp), "sigma_stat": float(s), "sigma_model": float(max(abs(b1-pp), abs(b2-pp))),
        "alternatives": [float(b1), float(b2)]} for rr, pp, s, b1, b2 in zip(r_mid, p, mc.std(0), a1, a2)]

# ---- large-r predictions (r = 4, 8, 16 used). Exact identity Gamma = V/(2 nu) - 1/3.
L = r >= 4; rl, nul, nusel, Vl, Vsel = r[L], nu[L], nuse[L], V[L], Vse[L]
RICE = (8/np.pi**2)/np.sqrt(2*np.pi)        # E|cos a - cos b| / sqrt(2 pi): leading nu*sqrt(r) from Rice's formula
def large(nu_, V_, rq):
    s = nu_*np.sqrt(rl)
    A = np.c_[np.ones(3), 1/np.sqrt(rl)]; w = 1/(nusel*np.sqrt(rl))
    cfree = np.linalg.lstsq(A*w[:, None], s*w, rcond=None)[0]                 # nu sqrt r = c + a/sqrt r
    afix = np.sum(w**2*(s - RICE)/rl)/np.sum(w**2/rl**2)                       # nu sqrt r = RICE + a/r
    wv = 1/Vsel; cv = np.linalg.lstsq(A*wv[:, None], V_*wv, rcond=None)[0]     # V = Vinf + b/sqrt r
    nus_rice = RICE + afix/rq; nus_free = cfree[0] + cfree[1]/np.sqrt(rq)
    Vq = cv[0] + cv[1]/np.sqrt(rq)
    G_struct = Vq/(2*nus_rice/np.sqrt(rq)) - 1/3
    ratio = np.mean((V_/(2*nu_) - 1/3)[1:]/np.sqrt(rl[1:]))                    # mean Gamma/sqrt r at r = 8, 16
    G_ratio = ratio*np.sqrt(rq)
    return G_struct, G_ratio, nus_rice, nus_free, cv, afix, cfree, ratio

rq = np.array([32.0, 64.0])
Gs, Gr, nr, nf, cv, afix, cfree, ratio = large(nul, Vl, rq)
mcG, mcN = [], []
for _ in range(2000):
    o = large(nul + rng.normal(0, 1, 3)*nusel, Vl + rng.normal(0, 1, 3)*Vsel, rq); mcG.append(o[0]); mcN.append(o[2])
mcG = np.array(mcG); mcN = np.array(mcN)
big = [{"r": float(q), "Gamma_pred": float(g), "sigma_stat": float(s), "sigma_model": float(abs(g - gr)),
        "alternative_constant_ratio": float(gr)} for q, g, s, gr in zip(rq, Gs, mcG.std(0), Gr)]
rice = [{"r": float(q), "nu_sqrt_r_pred": float(a), "sigma_stat": float(s), "sigma_model": float(abs(a - b)),
         "alternative_free_fit": float(b)} for q, a, s, b in zip(rq, nr, mcN.std(0), nf)]

refs = [{"r": 0.02, "Gamma": 0.426567, "SE": 0.000299, "nu": 0.642094, "nu_SE": 3.12e-5, "source": "Stage 5F primary"}] + \
       [{"r": float(d[0]), "Gamma": float(d[1]), "SE": float(d[2]), "nu": float(d[3]), "nu_SE": float(d[4]),
         "source": "exploratory scan 2026-11-23"} for d in D if d[0] in (0.25, 0.5, 1.0, 2.0)]

pred = {
 "stage": "SFIT Part V Stage 5J -- programme step 1: independent pre-registered confirmation of Gamma(r)",
 "status": "FROZEN before any primary run",
 "frozen_scripts": {f: H(f) for f in ["stage5j_primary_sim.py", "stage5j_plainloop_sim.py",
                                      "stage5j_score.py", "stage5j_build_predictions.py"]},
 "estimator": {"reward": "fixed-defect reward R_k(n) (Stage 5I eq. 2), averaged over k",
               "cutoff": "M = max(60, floor(25/(1-rho))), rho = 3r/(1+3r); burn-in 4M; T = max(4000, 20M)",
               "mean_subtraction": False, "Gamma": "(V_M/nu - 2/3)/2", "truncation_systematic": "|Gamma_M - Gamma_2M|"},
 "run_plan": {
   "primary": [dict(r=q, G=32, C=c, seed=sd, M=cut(q), T=max(4000, 20*cut(q)), B=4*cut(q)) for q, c, sd in
               [(0.4, 2000, 9104), (0.5, 2000, 9105), (0.6, 2000, 9106), (1.5, 2000, 9115), (3.0, 2000, 9130),
                (32.0, 500, 9320), (64.0, 500, 9640)]],
   "plainloop": [{"r": q, "G": 32, "T": 5000000, "seed": s, "M": cut(q)} for q, s in
                 [(0.02, 8002), (0.25, 8025), (0.5, 8050), (1.0, 8100), (2.0, 8200)]],
   "rule": "exactly these runs, once each; no extra, duplicate, or re-seeded runs; every run is scored"},
 "predictions": {"P1_intermediate": mid, "P3_large_r_Gamma": big, "P4_rice_nu_sqrt_r": rice,
                 "rice_constant": float(RICE), "fit_details": {"V_inf_and_b": cv.tolist(), "rice_a_over_r": float(afix),
                 "free_nu_fit": cfree.tolist(), "constant_ratio": float(ratio)}},
 "references_for_I1": refs,
 "tests": {
   "z_threshold": 3.5,
   "z_threshold_rationale": "19 z-tests (I1: 10, P1: 5, P3: 2, P4: 2); SEs from G = 32 groups (31 dof). Two-sided "
       "t_31 tail at 3.5 is 0.0014 per test, family-wise false-FAIL ~2.7% if every prediction is correct "
       "(|z| <= 3 with G = 16 would have been ~16%).",
   "sigma_obs": "per point, sigma_obs = sqrt(SE_Gamma^2 + Delta^2), Delta = mean over groups of (Gamma_M - Gamma_2M); "
                "the truncation systematic always enters every Gamma test",
   "V1_truncation": "per primary point, from the per-group differences Delta_g: the truncation is MATERIAL iff "
                    "|mean Delta| > 3 SE(Delta) AND |mean Delta| > SE_Gamma. A material point is INVALID: excluded from "
                    "P1-P3 and makes the outcome at best INCONCLUSIVE. V1 never produces FAIL by itself.",
   "I1_independence": "plain-loop vs references: |z| <= z_threshold for Gamma and nu at each r, and chi2 p >= 0.01 for each quantity",
   "P1_intermediate": "|z| <= z_threshold at each valid r, z = (obs - pred)/sqrt(sigma_obs^2 + sigma_stat^2 + sigma_model^2)",
   "P2_lower_bounds": "LOWER-CONFIDENCE-BOUND test (the claim, not compatibility), at every valid intermediate point: "
                      "(a) positivity  Gamma - 5 sigma_obs > 0;  (b) floor  Gamma - 3 sigma_obs >= 0.36",
   "P3_large_r": "|z| <= z_threshold at r = 32, 64 (same z definition)",
   "P4_rice": "|z| <= z_threshold for nu*sqrt(r) at r = 32, 64 (independent of V1: nu has no lag truncation)",
   "outcome": "INVALID if any provenance check fails (hash, plan, duplicates, extra runs, missing runs, group-level "
              "recomputation). Else FAIL if any of I1, P1, P2, P3, P4 fails on valid points. Else INCONCLUSIVE if any "
              "point is V1-material (listing which tests could not be scored). Else PASS.",
   "scope": "A PASS prospectively confirms the positive persistence curve at the listed r only. It is evidence for, "
            "not a proof of, Gamma(r) > 0 for all r."},
 "interpretation_frozen": {
   "I1_fail": "the two implementations disagree: code/definition error somewhere; no Gamma claim stands until resolved",
   "P2_fail": "positivity or floor claim not confirmed at a tested r; Stage 4 premise must be re-examined",
   "P1_fail_only": "interpolation model wrong near the minimum; positivity claim unaffected if P2 passes",
   "P3_fail_only": "large-r functional form wrong; positivity unaffected",
   "P4_fail_only": "Rice leading constant or its correction wrong; crossing-rate asymptotics to be revisited",
   "exploratory_event_index": "NOT scored; reported only (sign of rho_e(l) at all lags; Gamma_event vs Gamma)"}}
json.dump(pred, open("stage5j_predictions.json", "w"), indent=1)
print(json.dumps(pred["predictions"], indent=1))
print("TIMESTAMP THIS HASH (CRLF-normalized SHA-256 of stage5j_predictions.json):", H("stage5j_predictions.json"))
