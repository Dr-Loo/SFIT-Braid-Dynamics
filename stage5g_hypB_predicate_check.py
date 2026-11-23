#!/usr/bin/env python3
"""
stage5g_hypB_predicate_check.py -- exact verification of hypotheses (H1) and (H2) for Hypothesis B.

Three-interval adjacent window (Stage 5G, Sec. 4). Fixed points on the unit circle:
  U (initial), Y = U^{i<-V}, Z = Y^{j<-W}, Q = Z^{l<-X}.
  Interval 1: start U,   target Y, end factor s1      P1 = Y + s1 (U - Y)
  Interval 2: start P1,  target Z, end factor s2      P2 = Z + s2 (P1 - Z)
  Interval 3: start P2,  target Q, complete (end factor 0)
For a pair (a,b) in an interval with relative start S and relative target T, the gap is g(q) = T_x + q (S_x - T_x),
q from 1 down to the end factor e. The crossing decision and the over/under sign are functions of the signs of
  g_start = S_x,   g_end = T_x + e (S_x - T_x),   d = T_y S_x - T_x S_y
(the over/under sign equals sgn(d) sgn(g_start)). So the three-interval reward is a function of finitely many sign
predicates, each a polynomial in the circle coordinates and the relaxation parameters.

(H1) perturb s1 = s with s2 = v a parameter; (H2) perturb s2 = s with s1 = u a parameter.
For every predicate we evaluate the s = 0 family at exact rational points of the circle and test whether the resulting
univariate polynomials in the parameter have a common root in [0,1]. No common root  =>  the family is not identically
zero for ANY parameter value in [0,1] (exact arithmetic, no floating point).
Predicates that ARE identically zero at s = 0 are reported; the script then checks the structural reason
(static pair at s = 0: S = T, so no crossing can exist and only the gap predicates can create one).
"""
import itertools, sympy as sp

par = sp.Symbol("p")                                      # the parameter (v for H1, u for H2)

def circ(t):
    t = sp.Rational(t)
    return sp.Matrix([(1 - t**2) / (1 + t**2), 2 * t / (1 + t**2)])

def windows(pattern, ang, s1, s2):
    i, j, l = pattern
    U = [ang["U0"], ang["U1"], ang["U2"]]
    Y = list(U); Y[i] = ang["V"]
    Z = list(Y); Z[j] = ang["W"]
    Q = list(Z); Q[l] = ang["X"]
    P1 = [Y[k] + s1 * (U[k] - Y[k]) for k in range(3)]
    P2 = [Z[k] + s2 * (P1[k] - Z[k]) for k in range(3)]
    return [(U, Y, s1), (P1, Z, s2), (P2, Q, 0)]

def predicates(pattern, ang, s1, s2):
    out = {}
    for n, (S, T, e) in enumerate(windows(pattern, ang, s1, s2), 1):
        for a, b in itertools.combinations(range(3), 2):
            dS = S[a] - S[b]; dT = T[a] - T[b]
            out[(n, (a, b), "g_start")] = dS[0]
            out[(n, (a, b), "g_end")] = dT[0] + e * (dS[0] - dT[0])
            out[(n, (a, b), "d")] = dT[1] * dS[0] - dT[0] * dS[1]
            out[(n, (a, b), "static")] = (dS - dT)
    return out

PATTERNS = {"i=j=l": (0, 0, 0), "i=j!=l": (0, 0, 1), "i=l!=j": (0, 1, 0), "j=l!=i": (0, 1, 1), "all distinct": (0, 1, 2)}
TS = [["1/3", "2/7", "-5/4", "3/2", "-1/5", "7/3"], ["-2/5", "4/3", "1/7", "-3/2", "5/6", "-7/4"],
      ["3/5", "-1/3", "6/5", "-4/7", "2/9", "9/4"], ["-6/7", "5/3", "-2/9", "1/4", "-8/5", "3/7"],
      ["2/3", "-7/5", "4/9", "-1/6", "11/7", "-3/8"], ["5/2", "1/9", "-4/5", "7/6", "-2/3", "1/8"]]
NAMES = ["U0", "U1", "U2", "V", "W", "X"]

def check(hyp):
    bad = []
    s = sp.Symbol("s")
    for pname, pat in PATTERNS.items():
        evals = []
        for tset in TS:
            ang = {nm: circ(t) for nm, t in zip(NAMES, tset)}
            pr = predicates(pat, ang, *( (s, par) if hyp == "H1" else (par, s) ))
            evals.append({k: sp.expand(v.subs(s, 0)) if k[2] != "static" else v.subs(s, 0) for k, v in pr.items()})
        for key in evals[0]:
            if key[2] == "static":
                continue
            polys = [sp.Poly(e[key], par) for e in evals]
            if all(pl.is_zero for pl in polys):
                static = all(all(sp.simplify(c) == 0 for c in e[(key[0], key[1], "static")]) for e in evals)
                ok = static and key[2] == "d"
                bad.append((pname, key, "identically zero", "static pair at s=0: d irrelevant" if ok else "UNEXPLAINED"))
                continue
            # (H3) structural factor: if every evaluation is divisible by par^m, divide it out (par > 0 on (0,1),
            # so the sign decision is unchanged there) and test the reduced family on the CLOSED interval [0,1].
            m = min(min(mon[0] for mon in pl.monoms()) for pl in polys if not pl.is_zero)
            if m > 0:
                polys = [sp.Poly(sp.expand(pl.as_expr() / par**m), par) for pl in polys]
            g = polys[0]
            for pl in polys[1:]:
                g = sp.gcd(g, pl)
            roots = [r for r in sp.real_roots(g) if 0 <= r <= 1] if g.degree() > 0 else []
            if roots:
                bad.append((pname, key, f"common root(s) in [0,1] after removing par^{m}: {roots}", "UNEXPLAINED"))
            elif m > 0:
                bad.append((pname, key, f"structural factor par^{m} removed; reduced family nonzero on [0,1]", "H3"))
    return bad

if __name__ == "__main__":
    import hashlib
    print("SHA-256:", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
    for hyp in ("H1", "H2"):
        bad = check(hyp)
        unexplained = [b for b in bad if b[3] == "UNEXPLAINED"]
        print(f"\n{hyp}: {len(PATTERNS)} patterns x 3 intervals x 3 pairs x 3 predicates checked")
        for b in bad:
            print("  ", b)
        print(f"{hyp} VERIFIED" if not unexplained else f"{hyp} NOT VERIFIED ({len(unexplained)} unexplained)")