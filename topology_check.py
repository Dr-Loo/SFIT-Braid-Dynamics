"""topology_check.py -- is the crossing orientation on the ring fixed by the cyclic order?
Prediction (derived): with three defects on a ring, every crossing reverses the cyclic order, and
given the cyclic order the two defects of a pair are adjacent across a unique arc, so the overtaking
direction is determined. Hence s_n g(p_n) = (-1)^n * const, g = (+1, -1, +1) for pairs (0,1),(0,2),(1,2).
Violations can only come from several crossings falling in one Euler step (ordering ambiguity),
so their rate must shrink with dt."""
import sys, numpy as np
sys.argv = ["x", "--quick"]
import orientation_stats as O
O.R = 300; O.T = 500.0
g = np.array([1, -1, 1])
for dt in (0.02, 0.01, 0.005):
    O.DT = dt
    sp, sg = O.crossing_sequences(np.random.default_rng(5), 1.0)
    bad = tot = 0
    for p, s in zip(sp, sg):
        t = s.astype(int) * g[p.astype(int)]
        bad += int(np.sum(t[1:] == t[:-1])); tot += len(t) - 1
    print(f"dt={dt:<6}: consecutive crossings violating the rule: {bad} of {tot} ({bad/tot:.2e})")
