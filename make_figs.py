"""Figures for SFIT Part III. Data: part3_v2_results.json, step0_v2_results.json (identical on both
machines); exchange-audit bias arm transcribed from the author's run (sha256 dd06dbd4...)."""
import json, math, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.labelcolor": "#333", "xtick.color": "#555",
                     "ytick.color": "#555", "lines.linewidth": 2, "legend.frameon": False})
B, O, A, Y, GR = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9a9a94"
H = json.load(open("../part3_v2_results.json"))["grid"]
S0 = json.load(open("../step0_v2_results.json"))

# Fig 1: alpha vs kappa
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for th, c, lab in ((0.3, B, r"$\theta=0.3$"), (0.05, O, r"$\theta=0.05$")):
    for s in ["S2a", "S3a", "S3c", "S3p", "S4a"]:
        rs = sorted([r for r in H if r["law"] == "exponential" and r["theta"] == th and r["nu"] == 1.0 and r["state"] == s],
                    key=lambda r: r["kappa"])
        k = [r["kappa"] for r in rs]; a = np.array([r["alpha"] for r in rs])
        lo = a - np.array([r["alpha_lo"] for r in rs]); hi = np.array([r["alpha_hi"] for r in rs]) - a
        ax.errorbar(k, a, yerr=[lo, hi], color=c, marker="o", ms=4, lw=1.2, alpha=0.85, capsize=0,
                    label=lab if s == "S2a" else None)
ax.axhline(1, color=GR, lw=1, ls="--"); ax.axhline(0.1, color=GR, lw=1, ls=":")
ax.text(8.3, 1.04, "independent", va="bottom", fontsize=7, color="#555")
ax.text(8.3, 0.1, "strong-exclusion\ncriterion", va="center", fontsize=7, color="#555")
ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
ax.set_xlim(0.85, 16); ax.set_ylim(0, 1.1)
ax.set_xlabel(r"back-reaction strength $\kappa$"); ax.set_ylabel(r"gated coincidence $\alpha$")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout(); fig.savefig("fig_alpha.pdf"); fig.savefig("fig_alpha.png", dpi=200)

# Fig 2: bias arm vs Kimura
M, p = 2000, 0.2
obs = [(0.0, 0.1993, 0.1921, 0.2066), (2.5e-5, 0.2087, 0.2013, 0.2161), (1e-4, 0.2351, 0.2274, 0.2429),
       (2.5e-4, 0.2872, 0.2790, 0.2955), (1e-3, 0.5653, 0.5563, 0.5743)]
s = np.linspace(1e-6, 4.2, 300); u = (1 - np.exp(-s * p)) / (1 - np.exp(-s))
fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.fill_between([0, 4.2], p - 0.02, p + 0.02, color=GR, alpha=0.18, lw=0)
ax.axhline(p, color=GR, lw=1, ls="--")
ax.plot(s, u, color=B, label="Kimura diffusion (predicted)")
x = [2 * M * b for b, *_ in obs]; y = [o[1] for o in obs]
ax.errorbar(x, y, yerr=[[o[1] - o[2] for o in obs], [o[3] - o[1] for o in obs]], fmt="o", color=O, ms=5,
            capsize=0, label="simulation, 99% Wilson")
ax.text(3.0, 0.235, r"Born $\pm\,0.02$", fontsize=7, color="#555")
ax.set_xlabel(r"bias $2Mb$"); ax.set_ylabel("P(channel 0 recorded)"); ax.set_xlim(-0.1, 4.2)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig("fig_bias.pdf"); fig.savefig("fig_bias.png", dpi=200)

# Fig 3: R2 drift by x1 bin
d = np.array(S0["r2_drift_by_bin"]); xc = np.arange(0.05, 1.0, 0.1)
fig, ax = plt.subplots(figsize=(4.2, 3.0))
ax.axhline(0, color=GR, lw=1, ls="--")
ax.plot(xc, d, color=B, marker="o", ms=5)
ax.text(0.52, 0.02, "martingale: zero drift", fontsize=7, color="#555")
ax.set_xlabel(r"share $x_1$ before crossing"); ax.set_ylabel(r"mean change $E[\Delta x_1\,|\,x_1]$")
fig.tight_layout(); fig.savefig("fig_step0.pdf"); fig.savefig("fig_step0.png", dpi=200)

# Fig 4: threshold law comparison, S3c, theta=0.3, kappa=4, nu=1
pick = {law: next(r for r in H if r["state"] == "S3c" and r["theta"] == 0.3 and r["kappa"] == 4.0
                  and r["nu"] == 1.0 and r["law"] == law)["P"] for law in ("exponential", "gamma2", "fixed")}
series = [("Born $|\\psi_k|^2$", [0.7, 0.2, 0.1], GR), ("exponential", pick["exponential"], B),
          ("gamma, shape 2", pick["gamma2"], O), ("fixed", pick["fixed"], A)]
fig, ax = plt.subplots(figsize=(4.2, 3.0)); w = 0.19
for i, (lab, v, c) in enumerate(series):
    ax.bar(np.arange(3) + (i - 1.5) * (w + 0.01), v, w, color=c, label=lab)
ax.set_xticks(range(3)); ax.set_xticklabels([r"$k=1$", r"$k=2$", r"$k=3$"])
ax.set_ylabel("P(detector $k$ fires first)"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig("fig_laws.pdf"); fig.savefig("fig_laws.png", dpi=200)
print("ok")
