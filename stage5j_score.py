"""
stage5j_score.py -- scores Stage 5J against the FROZEN stage5j_predictions.json.
Usage: python stage5j_score.py stage5j_predictions.json primary_output.jsonl plainloop_output.jsonl EXPECTED_JSON_SHA256
EXPECTED_JSON_SHA256 is the hash of the timestamped prediction file (CRLF-normalized SHA-256, as printed by the
builder). Every run line is scored, including failures; the outcome is PASS / FAIL / INCONCLUSIVE / INVALID as
defined in the frozen "outcome" rule.
"""
import sys, json, hashlib, numpy as np
from scipy.stats import chi2

nh = lambda f: hashlib.sha256(open(f, 'rb').read().replace(b'\r\n', b'\n')).hexdigest()   # CRLF-normalized
se = lambda x: float(np.std(x, ddof=1)/np.sqrt(len(x)))

def load(f):
    return [json.loads(l) for l in open(f, encoding="utf-8-sig") if l.strip().startswith("{")]

def provenance(plan, lines, script, fz, prob):
    """Exactly the planned runs, once each, frozen script, frozen parameters, consistent group-level data."""
    runs = {}
    for d in lines:
        k = round(float(d["r"]), 6)
        if k in runs: prob.append(f"{script}: duplicate run at r={k}")
        runs[k] = d
    planned = {round(float(q["r"]), 6): q for q in plan}
    for k in runs:
        if k not in planned: prob.append(f"{script}: unplanned run at r={k}")
    for k, q in planned.items():
        d = runs.get(k)
        if d is None: prob.append(f"{script}: missing run r={k}"); continue
        if d.get("script") != script or d.get("sha256") != fz[script]:
            prob.append(f"{script} r={k}: script name/hash differs from frozen")
        for f in ("seed", "G", "C", "T", "M", "B"):
            if f in q and d.get(f) != q[f]: prob.append(f"{script} r={k}: {f}={d.get(f)} != plan {q[f]}")
        for name, arr in (("Gamma", "Gamma_groups"), ("nu", "nu_groups")):
            g = np.asarray(d.get(arr, []), float)
            if len(g) != q["G"] or not np.isclose(g.mean(), d[name][0], rtol=1e-9, atol=1e-12) \
                    or not np.isclose(se(g), d[name][1], rtol=1e-6, atol=1e-12):
                prob.append(f"{script} r={k}: {name} summary inconsistent with its group data")
    return runs

def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    pf, prim_f, pl_f, expected = sys.argv[1:5]
    P = json.load(open(pf)); Z = float(P["tests"]["z_threshold"]); sha_json = nh(pf); fz = P["frozen_scripts"]; prob = []
    if sha_json != expected.strip().lower(): prob.append("prediction file hash differs from the timestamped hash")
    if nh(__file__) != fz["stage5j_score.py"]: prob.append("scoring script hash differs from frozen")
    prim = provenance(P["run_plan"]["primary"], load(prim_f), "stage5j_primary_sim.py", fz, prob)
    pl = provenance(P["run_plan"]["plainloop"], load(pl_f), "stage5j_plainloop_sim.py", fz, prob)
    res = {}

    # ---- V1: truncation materiality from per-group differences; sigma_obs includes the systematic
    v1, sobs = {}, {}
    for r, d in prim.items():
        D = np.asarray(d["Gamma_groups"]) - np.asarray(d["Gamma_2M_groups"])
        m, sD, sG = float(D.mean()), se(D), d["Gamma"][1]
        material = abs(m) > 3*sD and abs(m) > sG
        v1[r] = {"Delta_mean": m, "Delta_SE": sD, "SE_Gamma": sG, "material": material}
        sobs[r] = float(np.hypot(sG, m))
    invalid = {r for r, v in v1.items() if v["material"]}
    res["V1_truncation"] = {"status": "INCONCLUSIVE" if invalid else "PASS",
                            "invalid_points": sorted(invalid), "points": v1}

    def tri(rows):            # PASS / FAIL on valid points; INCONCLUSIVE if nothing failed but some were invalid
        if any(x.get("ok") is False for x in rows): return "FAIL"
        if any(x.get("status") == "INVALID (V1)" for x in rows) or not rows: return "INCONCLUSIVE"
        return "PASS"

    def zt(d, r, p):
        return (d["Gamma"][0] - p["Gamma_pred"])/np.sqrt(sobs[r]**2 + p["sigma_stat"]**2 + p["sigma_model"]**2)

    # ---- I1 independence
    rows, zg, zn = [], [], []
    for ref in P["references_for_I1"]:
        d = pl.get(round(ref["r"], 6))
        if d is None: rows.append({"r": ref["r"], "ok": False, "status": "MISSING"}); continue
        a = (d["Gamma"][0] - ref["Gamma"])/np.hypot(d["Gamma"][1], ref["SE"])
        b = (d["nu"][0] - ref["nu"])/np.hypot(d["nu"][1], ref["nu_SE"])
        zg.append(a); zn.append(b)
        rows.append({"r": ref["r"], "Gamma_plainloop": d["Gamma"], "Gamma_ref": ref["Gamma"], "z_Gamma": a,
                     "nu_plainloop": d["nu"], "nu_ref": ref["nu"], "z_nu": b, "ok": abs(a) <= Z and abs(b) <= Z})
    pg = float(chi2.sf(np.sum(np.square(zg)), len(zg))) if zg else 0.0
    pn = float(chi2.sf(np.sum(np.square(zn)), len(zn))) if zn else 0.0
    i1 = "PASS" if rows and all(x["ok"] for x in rows) and pg >= 0.01 and pn >= 0.01 else "FAIL"
    res["I1_independence"] = {"status": i1, "chi2_p_Gamma": pg, "chi2_p_nu": pn, "points": rows}

    # ---- P1 intermediate values; P2 lower confidence bounds
    p1, p2 = [], []
    for p in P["predictions"]["P1_intermediate"]:
        r = round(p["r"], 6); d = prim.get(r)
        if d is None:
            p1.append({"r": p["r"], "ok": False, "status": "MISSING"}); p2.append({"r": p["r"], "ok": False, "status": "MISSING"}); continue
        if r in invalid:
            p1.append({"r": p["r"], "status": "INVALID (V1)"}); p2.append({"r": p["r"], "status": "INVALID (V1)"}); continue
        z = zt(d, r, p); g, s = d["Gamma"][0], sobs[r]
        p1.append({"r": p["r"], "obs": g, "sigma_obs": s, "pred": p["Gamma_pred"], "z": z, "ok": abs(z) <= Z})
        p2.append({"r": p["r"], "obs": g, "sigma_obs": s, "LCB5": g - 5*s, "LCB3": g - 3*s,
                   "positivity_ok": g - 5*s > 0, "floor_ok": g - 3*s >= 0.36, "ok": g - 5*s > 0 and g - 3*s >= 0.36})
    res["P1_intermediate"] = {"status": tri(p1), "points": p1}
    res["P2_lower_bounds"] = {"status": tri(p2), "points": p2}

    # ---- P3 large-r Gamma; P4 Rice (no truncation dependence)
    p3, p4 = [], []
    for p in P["predictions"]["P3_large_r_Gamma"]:
        r = round(p["r"], 6); d = prim.get(r)
        if d is None: p3.append({"r": p["r"], "ok": False, "status": "MISSING"}); continue
        if r in invalid: p3.append({"r": p["r"], "status": "INVALID (V1)"}); continue
        z = zt(d, r, p)
        p3.append({"r": p["r"], "obs": d["Gamma"][0], "sigma_obs": sobs[r], "pred": p["Gamma_pred"],
                   "alt": p["alternative_constant_ratio"], "z": z, "ok": abs(z) <= Z})
    for p in P["predictions"]["P4_rice_nu_sqrt_r"]:
        d = prim.get(round(p["r"], 6))
        if d is None: p4.append({"r": p["r"], "ok": False, "status": "MISSING"}); continue
        o, s = d["nu_sqrt_r"]
        z = (o - p["nu_sqrt_r_pred"])/np.sqrt(s**2 + p["sigma_stat"]**2 + p["sigma_model"]**2)
        p4.append({"r": p["r"], "obs": [o, s], "pred": p["nu_sqrt_r_pred"], "z": z, "ok": abs(z) <= Z})
    res["P3_large_r"] = {"status": tri(p3), "points": p3}
    res["P4_rice"] = {"status": tri(p4), "points": p4}
    res["exploratory_event_index_NOT_SCORED"] = {str(r): d.get("exploratory_event_index") for r, d in pl.items()}

    keys = ["I1_independence", "P1_intermediate", "P2_lower_bounds", "P3_large_r", "P4_rice"]
    if prob: overall = "INVALID"
    elif any(res[k]["status"] == "FAIL" for k in keys): overall = "FAIL"
    elif invalid or any(res[k]["status"] == "INCONCLUSIVE" for k in keys): overall = "INCONCLUSIVE"
    else: overall = "PASS"
    out = {"predictions_sha256": sha_json, "provenance_problems": prob, "overall": overall, "results": res,
           "scope": P["tests"]["scope"]}
    json.dump(out, open("stage5j_score_output.json", "w"), indent=1, default=float)
    print(f"predictions sha256 (normalized): {sha_json}")
    for x in prob: print("PROVENANCE:", x)
    for k in ["V1_truncation"] + keys: print(f"{k:18s} {res[k]['status']}")
    print("OVERALL:", overall, " (all results, including failures: stage5j_score_output.json)")

if __name__ == "__main__":
    main()
