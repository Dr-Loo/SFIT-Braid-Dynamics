"""
stage5_core.py -- SFIT Part V Stage 5: Part I substrate with local crossing rules (stage3_core.Substrate,
extended). With rule = None (or p = 0) it reproduces stage3_core exactly (no extra random numbers drawn).

Crossing rules act on the headings of the two defects of a crossing, right after the crossing, each applied
with probability p per crossing. Both are symmetric under relabelling of defects and under x -> -x and
y -> -y, so exchangeability and the u -> -u symmetry of Part V are kept. Paths stay continuous, so every
crossing is still a well-defined braid generator.
  "separate"  (recrossing suppression): the defect now on the left gets a leftward x-velocity and the one on
              the right a rightward x-velocity (heading mirrored in x where needed; |v_x| and v_y kept).
              The pair moves apart after passing, which suppresses immediate recrossing.
  "exchange"  (transverse exchange): the over-strand (higher) defect gets a downward y-velocity and the
              under-strand an upward one (heading mirrored in y where needed). The pair tends to swap
              heights before it meets again, which makes the next over-strand label of the same pair
              anti-persistent.
"""
import numpy as np
import stage3_core as C3

N, C, DT = C3.N, C3.C, C3.DT
PA, PB = C3.PA, C3.PB


class RuleSubstrate(C3.Substrate):
    def __init__(self, rng, R, gamma, dt=DT, kappa=0.0, K=C3.K_TRAP, rule=None, p=0.0):
        super().__init__(rng, R, gamma, dt, kappa, K)
        self.rule = rule if (rule and p > 0) else None; self.p = p

    def step(self):
        out = super().step()
        if self.rule is None or not out:
            return out
        th = self.th; X = self.X
        for rows, p, s, u, tm in out:
            app = self.rng.random(rows.size) < self.p
            if not app.any():
                continue
            rows, p, u = rows[app], p[app], u[app]
            a = PA[p]; b = PB[p]
            if self.rule == "separate":
                # who is left now: compare current x
                a_left = X[rows, a, 0] < X[rows, b, 0]
                left = np.where(a_left, a, b); right = np.where(a_left, b, a)
                tl = th[rows, left]; tr = th[rows, right]
                tl = np.where(np.cos(tl) > 0, np.pi - tl, tl); tr = np.where(np.cos(tr) < 0, np.pi - tr, tr)
                th[rows, left] = tl; th[rows, right] = tr
            elif self.rule == "exchange":
                over = np.where(u > 0, a, b); under = np.where(u > 0, b, a)
                to = th[rows, over]; tu = th[rows, under]
                to = np.where(np.sin(to) > 0, -to, to); tu = np.where(np.sin(tu) < 0, -tu, tu)
                th[rows, over] = to; th[rows, under] = tu
        return out


def events(rng, R, T, gamma, rule=None, p=0.0, dt=DT, kappa=0.0, K=C3.K_TRAP):
    S = RuleSubstrate(rng, R, gamma, dt, kappa, K, rule, p)
    buf = {k: [] for k in ("r", "p", "s", "u", "t")}
    for _ in range(int(T / dt)):
        for rows, pp, s, u, tm in S.step():
            buf["r"].append(rows); buf["p"].append(pp); buf["s"].append(s); buf["u"].append(u); buf["t"].append(tm)
    cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in buf.items()}
    o = np.argsort(cat["r"], kind="stable"); r = cat["r"][o]; cuts = np.searchsorted(r, np.arange(R + 1))
    srt = {k: cat[k][o] for k in ("p", "s", "u", "t")}
    return [tuple(srt[k][cuts[i]:cuts[i + 1]].copy() for k in ("p", "s", "u", "t")) for i in range(R)]