"""
exchange_audit_v2.py -- SFIT Part III: fair-absorbing-exchange route to Born weights,
with a falsification arm.  (Rerun specification v2 of sfit_part_iii_exchange_audit.py.)

STATUS OF THE MODEL
  The exchange kernel is a Part III HYPOTHESIS, not derived from Part I. It is also GLOBAL
  (each update uses all channel shares), so it does not by itself satisfy the locality
  requirement. This audit tests the mathematics of the route, not its physical origin.

KERNEL (unchanged from v0.1)
  M capacity units, integer loads n_k, shares x_k = n_k / M.
  K0 (fair):   n' ~ Multinomial(M, x)            ->  E[x' | x] = x exactly; vertices absorbing.
  Kb (biased): n' ~ Multinomial(M, x~), x~_0 = x_0 (1+b) / (1 + b x_0), x~_k = x_k / (1 + b x_0)
               ->  drift on channel 0 = b x_0 (1 - x_0) / (1 + b x_0).
  Record = the channel holding all M units when the run is absorbed.

PRE-STATED PREDICTIONS
  E1  K0: Born. Upper 99% bootstrap bound of TV(P, x(0)) <= EPS = 0.02 for every state.
      x(0) is the quantised initial share vector; TV to the requested |psi|^2 is also reported.
  E2  K0 conditional drift: one-step E[dx | x] = 0 at states visited along trajectories
      (steps 1, 10, 100, 1000). Fraction of components with |z| > 3 consistent with chance
      (expected 0.27%; flagged if > 1%).
  E3  Falsification arm (two channels, x_0(0) = 0.2): the absorption probability of channel 0
      follows the Kimura diffusion formula u = (1 - exp(-2 M b p)) / (1 - exp(-2 M b)),
      p = 0.2, for b in {0, 2.5e-5, 1e-4, 2.5e-4, 1e-3}: prediction inside the 99% Wilson
      interval of the observed frequency. Born (u = p) holds only at b = 0.
  E4  Tolerance: to first order u - p = M b p (1 - p), so keeping |u - p| <= EPS requires
      |b| <= EPS / (M p (1 - p)): the allowed bias shrinks as 1/M (computed and printed).

CONTROLS (any failure invalidates the run)
  C1  K0 on S3a passes E1 (positive control).
  C2  Uniform initial shares compared with the S3c target: lower bound of TV > EPS
      (the pipeline detects a mismatch).
  C3  Unresolved runs (not absorbed within MAX_STEPS) < 0.1% of trials for every state.

NOT MODELED: detector event times, hence no coincidence alpha. A single global absorbing
record would give alpha = 0 by construction, which would not be evidence of exclusion.

Usage: python exchange_audit_v2.py [--quick] [--hash]
"""
import sys, json, math, hashlib, pathlib, time
import numpy as np

QUICK = "--quick" in sys.argv
TRIALS = 4000 if QUICK else 20000
M = 2000
MAX_STEPS = 60000
EPS, CONF, N_BOOT = 0.02, 0.99, 2000
SEED = 20261101
BIASES = [0.0, 2.5e-5, 1e-4, 2.5e-4, 1e-3]
STATES = {
    "S2a": [0.2, 0.8], "S2b": [0.5, 0.5],
    "S3a": [0.5, 0.3, 0.2], "S3b": [1/3, 1/3, 1/3], "S3c": [0.7, 0.2, 0.1],
    "S4a": [0.4, 0.3, 0.2, 0.1], "S4b": [0.25] * 4,
}
Z99 = 2.5758


def allocate(p, M):
    raw = np.asarray(p) * M
    n = np.floor(raw).astype(int)
    rem = M - int(n.sum())
    if rem:
        n[np.argsort(-(raw - n))[:rem]] += 1
    return n


def multinomial_rows(rng, M, probs):
    """Exact multinomial draws, one per row of probs, by sequential binomials."""
    rows, k = probs.shape
    out = np.zeros((rows, k), int)
    rem_n = np.full(rows, M); rem_p = np.ones(rows)
    for j in range(k - 1):
        pj = np.clip(np.divide(probs[:, j], rem_p, out=np.zeros(rows), where=rem_p > 1e-15), 0, 1)
        d = rng.binomial(rem_n, pj)
        out[:, j] = d; rem_n -= d; rem_p -= probs[:, j]
    out[:, -1] = rem_n
    return out


def biased(x, b):
    if b == 0.0:
        return x
    y = x.copy(); y[:, 0] *= (1 + b)
    return y / y.sum(1, keepdims=True)


def run(n0, trials, rng, b=0.0, snap_steps=()):
    n = np.repeat(n0[None, :], trials, 0)
    outcome = np.full(trials, -1); active = np.ones(trials, bool); snaps = {}
    for step in range(1, MAX_STEPS + 1):
        ids = np.flatnonzero(active)
        if ids.size == 0:
            break
        n[ids] = multinomial_rows(rng, M, biased(n[ids] / M, b))
        if step in snap_steps:
            alive = ids[(n[ids] > 0).sum(1) > 1]
            snaps[step] = n[alive[:500]].copy()
        done = (n[ids] == M).any(1)
        if done.any():
            d = ids[done]; outcome[d] = n[d].argmax(1); active[d] = False
    return outcome, snaps


def wilson(k, n, z=Z99):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def tv_ci(counts, target, rng):
    nr = counts.sum(); p = counts / nr
    boot = rng.multinomial(nr, p, size=N_BOOT) / nr
    tvs = 0.5 * np.abs(boot - target).sum(1)
    lo, hi = np.percentile(tvs, [100 * (1 - CONF) / 2, 100 * (1 + CONF) / 2])
    return 0.5 * np.abs(p - target).sum(), lo, hi


def drift_audit(snaps, rng, n_children=2000):
    zs = []
    for step, states in snaps.items():
        for s in states:
            x = s / M
            kids = multinomial_rows(rng, M, np.repeat(x[None, :], n_children, 0)) / M
            dx = kids.mean(0) - x
            se = kids.std(0, ddof=1) / math.sqrt(n_children)
            ok = se > 0
            zs.extend((dx[ok] / se[ok]).tolist())
    zs = np.array(zs)
    return zs.size, float(np.mean(np.abs(zs) > 3)) if zs.size else float("nan"), float(np.abs(zs).max()) if zs.size else float("nan")


def kimura(p, b):
    if b == 0.0:
        return p
    s = 2 * M * b
    return (1 - math.exp(-s * p)) / (1 - math.exp(-s))


def main():
    here = pathlib.Path(__file__).resolve()
    digest = hashlib.sha256(here.read_bytes()).hexdigest()
    if "--hash" in sys.argv:
        print(digest); return
    t0 = time.time()
    ss = np.random.SeedSequence(SEED).spawn(len(STATES) + len(BIASES) + 4)
    print("=" * 100)
    print(f"Exchange audit v2 | trials={TRIALS} M={M} max_steps={MAX_STEPS} {'(QUICK)' if QUICK else ''}\nscript sha256 {digest}")
    print("=" * 100)

    res = {}; valid = True
    print(f"\nE1/E2  fair kernel K0: Born test (99% bootstrap TV bound <= {EPS}) and conditional drift")
    print(f"{'state':5} {'x(0)':28} {'P(record)':30} {'TV [lo,hi]':24} {'unres':>7} {'drift |z|>3':>11}")
    for i, (name, p) in enumerate(STATES.items()):
        rng = np.random.default_rng(ss[i])
        n0 = allocate(p, M); x0 = n0 / M
        out, snaps = run(n0, TRIALS, rng, snap_steps=(1, 10, 100, 1000))
        ok = out >= 0; counts = np.bincount(out[ok], minlength=len(p))
        t, lo, hi = tv_ci(counts, x0, rng)
        t_req = 0.5 * np.abs(counts / counts.sum() - np.asarray(p)).sum()
        unres = 1 - ok.mean()
        nz, frac3, zmax = drift_audit(snaps, rng)
        res[name] = dict(target=p, x0=x0.tolist(), counts=counts.tolist(), tv=t, tv_lo=lo, tv_hi=hi,
                         tv_requested=t_req, unresolved=unres, wilson99=[wilson(c, counts.sum()) for c in counts],
                         drift_components=nz, drift_frac_z3=frac3, drift_max_z=zmax,
                         born_pass=bool(hi <= EPS))
        valid &= unres < 1e-3
        print(f"{name:5} {str(np.round(x0, 3)):28} {str(np.round(counts / counts.sum(), 4)):30} "
              f"{t:.4f} [{lo:.4f},{hi:.4f}] {unres:7.1e} {frac3:10.2%}")

    print("\nControls:")
    c1 = res["S3a"]["born_pass"]
    rng = np.random.default_rng(ss[-1])
    out, _ = run(allocate([1/3] * 3, M), TRIALS, rng)
    cnt = np.bincount(out[out >= 0], minlength=3)
    _, lo_c2, _ = tv_ci(cnt, np.array(STATES["S3c"]), rng)
    c2 = lo_c2 > EPS
    c3 = all(r["unresolved"] < 1e-3 for r in res.values())
    for label, v, d in [("C1 K0 on S3a passes Born", c1, f"TV_hi={res['S3a']['tv_hi']:.4f}"),
                        ("C2 uniform start vs S3c target detected", c2, f"TV_lo={lo_c2:.4f}"),
                        ("C3 unresolved < 0.1% for every state", c3, f"max={max(r['unresolved'] for r in res.values()):.1e}")]:
        print(f"  [{'PASS' if v else 'FAIL'}] {label}: {d}")
    valid &= c1 and c2 and c3

    print("\nE3  falsification arm: biased kernel, two channels, p = 0.2  (prediction = Kimura diffusion)")
    print(f"  {'b':>8} {'2Mb':>6} {'P(ch0) observed':>16} {'99% Wilson':>18} {'Kimura':>8} {'inside':>7} {'|P-p|':>7}")
    bias_rows = []; e3 = True
    n0 = allocate([0.2, 0.8], M)
    for j, b in enumerate(BIASES):
        rng = np.random.default_rng(ss[len(STATES) + j])
        out, _ = run(n0, TRIALS, rng, b=b)
        ok = out >= 0; k0 = int((out[ok] == 0).sum()); nr = int(ok.sum())
        lo, hi = wilson(k0, nr); pred = kimura(0.2, b); inside = lo <= pred <= hi
        e3 &= inside
        bias_rows.append(dict(b=b, observed=k0 / nr, wilson99=(lo, hi), kimura=pred, inside=inside))
        print(f"  {b:8.1e} {2*M*b:6.2f} {k0/nr:16.4f} [{lo:.4f},{hi:.4f}] {pred:8.4f} {str(inside):>7} {abs(k0/nr-0.2):7.4f}")
    tol = EPS / (M * 0.2 * 0.8)
    print(f"\nE4  first-order tolerance: |P - p| <= {EPS} requires |b| <= EPS/(M p(1-p)) = {tol:.2e} "
          f"at M = {M} (and shrinks as 1/M).")

    e1 = all(r["born_pass"] for r in res.values())
    e2 = all(r["drift_frac_z3"] <= 0.01 for r in res.values())
    print("\n" + "=" * 100)
    print("DECISION (computed):")
    if not valid:
        print("  RUN INVALID: a control failed or unresolved runs exceeded budget. No conclusion drawn.")
    print(f"  [{'CONFIRMED' if e1 else 'NOT CONFIRMED'}] E1 fair kernel gives Born for every state")
    print(f"  [{'CONFIRMED' if e2 else 'NOT CONFIRMED'}] E2 conditional drift consistent with zero along trajectories")
    print(f"  [{'CONFIRMED' if e3 else 'NOT CONFIRMED'}] E3 biased kernel follows Kimura; Born only at b = 0")
    print(f"  E4 bias tolerance at M = {M}: |b| <= {tol:.1e}")
    print("  Scope: the kernel is a hypothesis and is global; this verifies the route, not its origin.")
    print("=" * 100)
    json.dump(dict(sha256=digest, trials=TRIALS, M=M, max_steps=MAX_STEPS, eps=EPS, states=res,
                   bias_arm=bias_rows, tolerance=tol, valid=bool(valid), E1=e1, E2=e2, E3=e3),
              open(here.with_name("exchange_audit_v2_results.json"), "w"), indent=1, default=float)
    print(f"Wall time {time.time()-t0:.0f} s. Results: {here.with_name('exchange_audit_v2_results.json')}")


if __name__ == "__main__":
    main()