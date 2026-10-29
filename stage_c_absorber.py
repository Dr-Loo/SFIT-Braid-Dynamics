"""
stage_c_absorber.py  --  does the |psi_k|^2 dependence EMERGE from a linear coupling?

Model (classical, nothing quadratic put in by hand)
---------------------------------------------------
Each channel k has its own environment element: a damped oscillator driven at
resonance by a force LINEAR in the carrier amplitude, F_k(t) = g * psi_k * cos(w0 t + phi_k),
plus thermal noise at temperature kT. Optionally anharmonic (Duffing term beta*x^3).

    x'' + gamma x' + w0^2 x + beta x^3 = F_k(t) + sqrt(2 gamma kT) xi(t)

The environment's record is triggered when the work it has absorbed from the drive,
W_k(t) = int F_k x' dt, first exceeds a threshold E_k. The first channel to trigger
gives the record.

What is derived vs assumed
--------------------------
DERIVED (measured, not set): how the absorbed power depends on psi_k (exponent p_emp).
ASSUMED (stated, varied):    the threshold statistics E_k. Three cases are run:
                              exponential (memoryless), fixed, uniform.

Measurements
------------
[1] Absorbed-power exponent: fit log P vs log|psi_k| (linear and anharmonic oscillator).
[2] Record frequencies vs Born for each threshold law (harmonic, weak noise).
[3] Thermal background: record frequencies as kT increases (spurious triggers).
Conclusions are computed from the numbers; nothing is printed unconditionally.

Usage: python stage_c_absorber.py
"""
import numpy as np

W0, GAMMA, G, DT = 1.0, 0.05, 0.1, 0.05


def steady_state(amp, phi):
    """Linear steady state at resonance: x = (F/(gamma w0)) sin(w0 t + phi)."""
    a = amp / (GAMMA * W0)
    return a * np.sin(phi), a * W0 * np.cos(phi)


def simulate(psi, n_runs, threshold, kT=1e-4, beta=0.0, t_max=4000.0, rng=None,
             return_power=False):
    """Vectorised over runs x channels. Returns winner per run (-1 = no trigger)."""
    rng = rng or np.random.default_rng()
    psi = np.asarray(psi, float)
    n = psi.size
    amp = G * np.abs(psi)[None, :] * np.ones((n_runs, 1))
    sgn = np.sign(psi)[None, :]
    phi = rng.uniform(0, 2 * np.pi, (n_runs, n))
    x, v = steady_state(amp, phi)
    if beta == 0.0:
        pass
    else:  # start anharmonic runs at rest and discard a burn-in for power measurement
        x, v = np.zeros_like(x), np.zeros_like(x)
    E = threshold(rng, (n_runs, n))
    W = np.zeros((n_runs, n))
    winner = np.full(n_runs, -1)
    active = np.ones(n_runs, bool)
    t, steps = 0.0, int(t_max / DT)
    noise_amp = np.sqrt(2 * GAMMA * kT / DT)
    burn = int(10 / GAMMA / DT) if beta else 0
    P_acc, P_cnt = np.zeros(n), 0
    for s in range(steps):
        F = sgn * amp * np.cos(W0 * t + phi)
        a = F - GAMMA * v - W0 ** 2 * x - beta * x ** 3 + noise_amp * rng.standard_normal(x.shape)
        v = v + a * DT
        x = x + v * DT
        dW = F * v * DT
        t += DT
        if return_power:
            if s >= burn:
                P_acc += dW.mean(0) / DT; P_cnt += 1
            continue
        W[active] += dW[active]
        hit = active[:, None] & (W >= E)
        if hit.any():
            rows = np.flatnonzero(hit.any(1))
            # earliest within this step is ambiguous at resolution DT: break ties by overshoot
            over = np.where(hit[rows], W[rows] - E[rows], -np.inf)
            winner[rows] = over.argmax(1)
            active[rows] = False
            if not active.any():
                break
    if return_power:
        return P_acc / P_cnt
    return winner


def thr_exponential(mean):
    return lambda rng, shape: rng.exponential(mean, shape)


def thr_fixed(value):
    return lambda rng, shape: np.full(shape, value)


def thr_uniform(lo, hi):
    return lambda rng, shape: rng.uniform(lo, hi, shape)


def thr_gamma(shape, mean=1.0):
    """Gamma thresholds with fixed mean; shape=1 is exponential, large shape -> fixed."""
    return lambda rng, size: rng.gamma(shape, mean / shape, size)


def race_exact(psi_sq, shape, n_grid=20000):
    """Exact win probabilities for constant absorbed power P_k ~ psi_k^2 and gamma
    thresholds: T_k = E_k / P_k, P(win k) = int f_k(t) prod_{j!=k} S_j(t) dt."""
    from scipy.stats import gamma as G_
    P = np.asarray(psi_sq, float)
    t = np.linspace(0, 40.0 / P.min(), n_grid)
    out = []
    for k in range(len(P)):
        fk = G_.pdf(t, shape, scale=1.0 / (shape * P[k]))
        Sj = np.prod([G_.sf(t, shape, scale=1.0 / (shape * P[j])) for j in range(len(P)) if j != k], axis=0)
        out.append(np.trapezoid(fk * Sj, t))
    out = np.array(out)
    return out / out.sum()


def tv(a, b):
    return 0.5 * np.sum(np.abs(np.asarray(a) - np.asarray(b)))


def freqs(winner, n):
    ok = winner >= 0
    return np.bincount(winner[ok], minlength=n) / max(ok.sum(), 1), 1 - ok.mean()


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z / (1 + z * z / n) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return c - h, c + h


def main():
    rng = np.random.default_rng(2026)
    print("=" * 84)
    print("STAGE C (absorber): is the psi^2 dependence derived from a LINEAR coupling?")
    print("=" * 84)

    # ---------------------------------------------------------------- [1]
    print("\n[1] Absorbed power vs drive amplitude (coupling is linear in psi_k)")
    amps = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
    for beta, label in ((0.0, "harmonic"), (0.5, "anharmonic, beta=0.5")):
        P = np.array([simulate([a], 400, thr_fixed(np.inf), kT=0.0, beta=beta, t_max=600,
                               rng=rng, return_power=True)[0] for a in amps])
        slope_all = np.polyfit(np.log(amps), np.log(P), 1)[0]
        slope_weak = np.polyfit(np.log(amps[:3]), np.log(P[:3]), 1)[0]
        slope_strong = np.polyfit(np.log(amps[-2:]), np.log(P[-2:]), 1)[0]
        print(f"  {label:22s} P(psi) = {np.array2string(P, precision=5)}")
        print(f"  {'':22s} exponent: weak drive {slope_weak:.3f}, strong drive {slope_strong:.3f}, all {slope_all:.3f}")

    # ---------------------------------------------------------------- [2]
    print("\n[2] Record frequencies (harmonic, kT = 1e-4) under three threshold laws")
    states = [np.sqrt([0.7, 0.2, 0.1]), np.sqrt([0.5, 0.3, 0.2]), np.sqrt([0.2, 0.8])]
    laws = [("exponential (memoryless)", thr_exponential(1.0)),
            ("fixed", thr_fixed(1.0)),
            ("uniform [0, 2]", thr_uniform(0.0, 2.0))]
    N = 4000
    results = {}
    for name, law in laws:
        print(f"  threshold law: {name}")
        worst = 0.0
        for psi in states:
            w = simulate(psi, N, law, kT=1e-4, rng=rng)
            f, unr = freqs(w, psi.size)
            born = psi ** 2
            k = (f * (w >= 0).sum()).round().astype(int)
            ci = [wilson(ki, (w >= 0).sum()) for ki in k]
            inside = all(lo <= b <= hi for (lo, hi), b in zip(ci, born))
            worst = max(worst, np.max(np.abs(f - born)))
            print(f"    |psi|^2={np.round(born, 3)} -> freq={np.round(f, 3)}  "
                  f"{'all Born values inside 95% CI' if inside else 'Born OUTSIDE 95% CI'}"
                  f"{'' if unr == 0 else f'  (no record in {unr:.1%})'}")
        results[name] = worst

    # ---------------------------------------------------------------- [3]
    print("\n[3] Thermal background (exponential thresholds, psi^2 = (0.7, 0.2, 0.1))")
    psi = np.sqrt([0.7, 0.2, 0.1])
    for kT in (1e-4, 1e-2, 1e-1):
        w = simulate(psi, N, thr_exponential(1.0), kT=kT, rng=rng)
        f, _ = freqs(w, 3)
        print(f"    kT = {kT:g}: freq = {np.round(f, 3)}  (Born {np.round(psi**2, 3)})")

    # ---------------------------------------------------------------- [4]
    print("\n[4] Threshold memory: gamma thresholds with shape k (k = 1 is exponential)")
    print("    TV = total-variation distance to Born. Noise floor = TV of an exact Born sampler")
    print("    with the same number of runs (median of 200 draws).")
    N4 = 4000
    floor = np.median([tv(rng.multinomial(N4, [0.7, 0.2, 0.1]) / N4, [0.7, 0.2, 0.1]) for _ in range(200)])
    print(f"    noise floor (N = {N4}): TV ~ {floor:.4f}")
    print(f"    {'k':>5} | {'TV sim, state A':>15} | {'TV exact race, A':>16} | {'TV sim, state B':>15} | freq A")
    gamma_tv = {}
    for kshape in (1, 1.5, 2, 4, 10, 50):
        row = []
        for psi in (np.sqrt([0.7, 0.2, 0.1]), np.sqrt([0.5, 0.3, 0.2])):
            w = simulate(psi, N4, thr_gamma(kshape), kT=1e-4, rng=rng)
            f, _ = freqs(w, 3)
            row.append((tv(f, psi ** 2), f))
        exact = race_exact([0.7, 0.2, 0.1], kshape)
        gamma_tv[kshape] = row[0][0]
        print(f"    {kshape:>5g} | {row[0][0]:15.4f} | {tv(exact, [0.7, 0.2, 0.1]):16.4f} | "
              f"{row[1][0]:15.4f} | {np.round(row[0][1], 3)}")

    print("\n" + "=" * 84)
    print("READING (computed from the numbers above):")
    for name, worst in results.items():
        print(f"  {name:26s} max |freq - Born| = {worst:.3f}")
    passing = [k for k, d in gamma_tv.items() if d <= 3 * floor]
    print(f"  gamma shapes within 3x the noise floor of Born: {passing if passing else 'none'}")
    print("  The exponent in [1] is measured, not set. Born frequencies in [2] require, in")
    print("  addition, the threshold law listed; check which law(s) pass before concluding.")
    print("=" * 84)


if __name__ == "__main__":
    main()