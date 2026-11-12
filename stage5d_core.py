"""
stage5d_core.py -- SFIT Part V Stage 5D: parameter-free crossing rules on the Part I substrate
(extends stage5_core.RuleSubstrate; its rules "separate" and "exchange" are unchanged).
New rules, applied at EVERY crossing (no tunable parameter), to both defects of the crossing:
  "randomize_y"  the sign of each defect's transverse velocity is redrawn as a fair coin, independently for the
                 two defects (|v_x|, |v_y| and the x-direction kept): the crossing erases transverse direction.
  "tumble"       each defect's heading is redrawn uniformly on the circle: the crossing is a full tumble.
Both are symmetric under relabelling and under x -> -x and y -> -y; paths stay continuous.
"""
import numpy as np
import stage3_core as C3
import stage5_core as S5


class Rule5D(S5.RuleSubstrate):
    def __init__(self, rng, R, gamma, dt=C3.DT, kappa=0.0, K=C3.K_TRAP, rule=None, p=0.0):
        new = rule in ("randomize_y", "tumble")
        super().__init__(rng, R, gamma, dt, kappa, K, None if new else rule, p)
        self.rule5d = rule if new else None

    def step(self):
        out = super().step()
        if self.rule5d is None or not out:
            return out
        th = self.th
        for rows, p, s, u, tm in out:
            for d in (S5.PA[p], S5.PB[p]):
                if self.rule5d == "randomize_y":
                    flip = self.rng.random(rows.size) < 0.5
                    th[rows[flip], d[flip]] = -th[rows[flip], d[flip]]
                else:
                    th[rows, d] = self.rng.uniform(0, 2 * np.pi, rows.size)
        return out


def events(rng, R, T, gamma, rule=None, p=0.0, dt=C3.DT, kappa=0.0, K=C3.K_TRAP):
    S = Rule5D(rng, R, gamma, dt, kappa, K, rule, p)
    buf = {k: [] for k in ("r", "p", "s", "u", "t")}
    for _ in range(int(T / dt)):
        for rows, pp, s, u, tm in S.step():
            buf["r"].append(rows); buf["p"].append(pp); buf["s"].append(s); buf["u"].append(u); buf["t"].append(tm)
    cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in buf.items()}
    o = np.argsort(cat["r"], kind="stable"); r = cat["r"][o]; cuts = np.searchsorted(r, np.arange(R + 1))
    srt = {k: cat[k][o] for k in ("p", "s", "u", "t")}
    return [tuple(srt[k][cuts[i]:cuts[i + 1]].copy() for k in ("p", "s", "u", "t")) for i in range(R)]