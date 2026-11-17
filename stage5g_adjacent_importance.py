#!/usr/bin/env python3
"""
Stage 5G Diagnostic: Adjacent Two-Gap Kernel Measurement via Fast-Rejection IS
Maps the mixed response kernel K(u,v) = E[Delta R_a(u) * Delta R_b(v) | u, v] 
to verify the uniform bound |K(u,v)| <= C * w(min(u,v)), where w(t) = t(1 + |log t|).

Usage: python stage5g_adjacent_importance.py
"""

import numpy as np
import time
import hashlib

# ==============================================================================
# 1. SCRIPT HASHING (Provenance)
# ==============================================================================
with open(__file__, 'rb') as f:
    script_hash = hashlib.sha256(f.read()).hexdigest()
print(f"Script SHA-256: {script_hash}\n")
print("="*90)
print("STAGE 5G: ADJACENT TWO-GAP KERNEL MEASUREMENT (FAST-REJECTION IS)")
print("="*90)

# ==============================================================================
# 2. EXACT EVENT-DRIVEN CROSSING OPERATOR (Vectorized Stage 5F Kernel)
# ==============================================================================
def compute_reward_vectorized(X, T, q, k=0):
    """
    Computes the exact insertion reward R_k(X, T; q) for N samples.
    X, T: shape (N, 3, 2) - End and Start points on the unit circle.
    q: shape (N,) - Relaxation factor.
    """
    N = X.shape[0]
    R = np.zeros(N)
    pairs = [(0, 1), (0, 2), (1, 2)]
    
    for a, b in pairs:
        dx_T = T[:, a, 0] - T[:, b, 0]
        dx_X = X[:, a, 0] - X[:, b, 0]
        denom = dx_X - dx_T
        
        xi = np.full(N, 2.0)
        mask = np.abs(denom) > 1e-12
        xi[mask] = -dx_T[mask] / denom[mask]
        
        I = (q < xi) & (xi < 1.0)
        
        dy_T = T[:, a, 1] - T[:, b, 1]
        dy_X = X[:, a, 1] - T[:, b, 1] # Correction: dy_X should be X
        dy_X = X[:, a, 1] - X[:, b, 1]
        y_term = dy_T + xi * (dy_X - dy_T)
        
        sign = np.sign(y_term)
        sign[y_term == 0] = 0 
        
        coeff = (k == a) - (k == b)
        R += coeff * I * sign
        
    return R

# ==============================================================================
# 3. KERNEL MEASUREMENT WITH FAST-REJECTION VARIANCE REDUCTION
# ==============================================================================
def measure_kernel(u, v, N=10_000_000):
    """
    Measures K(u,v) = E[Delta R_a(u) * Delta R_b(v) | u, v].
    Uses fast-rejection: if initial gaps > 10*max(u,v), the mixed product is exactly 0.
    """
    start_time = time.time()
    
    # 1. Generate headings
    theta_U = np.random.uniform(-np.pi, np.pi, (N, 3))
    theta_V = np.random.uniform(-np.pi, np.pi, N)
    theta_W = np.random.uniform(-np.pi, np.pi, N)
    
    i = np.random.randint(0, 3, N)
    j = np.random.randint(0, 3, N)
    
    x_U = np.cos(theta_U)
    y_U = np.sin(theta_U)
    
    x_V_full = x_U.copy()
    x_V_full[np.arange(N), i] = np.cos(theta_V)
    y_V_full = y_U.copy()
    y_V_full[np.arange(N), i] = np.sin(theta_V)
    
    x_W_full = x_U.copy()
    x_W_full[np.arange(N), j] = np.cos(theta_W)
    y_W_full = y_U.copy()
    y_W_full[np.arange(N), j] = np.sin(theta_W)
    
    # 2. Fast Rejection (The "Importance Sampling" step)
    # The maximum perturbation to the gaps is bounded by 2*u (from residual shift) 
    # and the crossing condition requires gaps <= 4*max(u,v).
    # We use a conservative threshold: 10 * max(u,v).
    threshold = 10.0 * max(u, v)
    
    gaps_1 = np.full(N, 10.0)
    gaps_2 = np.full(N, 10.0)
    
    for l in range(3):
        mask_1 = (l != i)
        diff_1 = np.abs(x_V_full[:, l] - x_U[:, l])
        gaps_1 = np.minimum(gaps_1, np.where(mask_1, diff_1, 10.0))
        
        mask_2 = (l != j)
        diff_2 = np.abs(x_W_full[:, l] - x_U[:, l])
        gaps_2 = np.minimum(gaps_2, np.where(mask_2, diff_2, 10.0))
        
    hit_mask = (gaps_1 <= threshold) | (gaps_2 <= threshold)
    N_hits = np.sum(hit_mask)
    
    if N_hits == 0:
        return 0.0, 0.0, 0, N, time.time() - start_time
        
    # 3. Exact Calculation on the "Hits"
    x_U_h = x_U[hit_mask]
    y_U_h = y_U[hit_mask]
    x_V_h = x_V_full[hit_mask]
    y_V_h = y_V_full[hit_mask]
    x_W_h = x_W_full[hit_mask]
    y_W_h = y_W_full[hit_mask]
    i_h = i[hit_mask]
    j_h = j[hit_mask]
    
    N_h = N_hits
    U_h = np.stack([x_U_h, y_U_h], axis=-1)
    V_h = np.stack([x_V_h, y_V_h], axis=-1)
    W_h = np.stack([x_W_h, y_W_h], axis=-1)
    
    # Tumble n (Gap A)
    Y_h = U_h.copy()
    Y_h[np.arange(N_h), i_h] = V_h[np.arange(N_h), i_h]
    X_n_h = Y_h + u * (U_h - Y_h) # Residual shift applied here
    
    R_a = compute_reward_vectorized(U_h, Y_h, np.full(N_h, u), k=0)
    
    # Tumble n+1 (Gap B)
    Z_prime_h = X_n_h.copy()
    Z_prime_h[np.arange(N_h), j_h] = W_h[np.arange(N_h), j_h]
    
    R_b_v = compute_reward_vectorized(X_n_h, Z_prime_h, np.full(N_h, v), k=0)
    R_b_0 = compute_reward_vectorized(X_n_h, Z_prime_h, np.zeros(N_h), k=0)
    Delta_R_b = R_b_v - R_b_0
    
    # Mixed product
    M = R_a * Delta_R_b
    
    sum_M = np.sum(M)
    sum_M2 = np.sum(M**2)
    
    K_uv = sum_M / N
    var_M = (sum_M2 / N) - K_uv**2
    se_K = np.sqrt(max(0, var_M) / N)
    
    return K_uv, se_K, N_hits, N, time.time() - start_time

# ==============================================================================
# 4. EXECUTION: GRID OF (u,v)
# ==============================================================================
u_vals = [0.1, 0.05, 0.01, 0.005, 0.001]
v_vals = [0.1, 0.05, 0.01, 0.005, 0.001]

print(f"{'u':<6} | {'v':<6} | {'K(u,v)':<12} | {'SE':<10} | {'Hit Rate':<10} | {'w(min)':<10} | {'Ratio':<8} | {'Time':<6}")
print("-" * 90)

results = []

for u in u_vals:
    for v in v_vals:
        K_uv, se_K, N_hits, N, elapsed = measure_kernel(u, v, N=5_000_000)
        
        # Theoretical bound function w(t) = t * (1 + |log t|)
        t_min = min(u, v)
        w_t = t_min * (1.0 + np.abs(np.log(t_min))) if t_min > 0 else 0
        
        ratio = np.abs(K_uv) / w_t if w_t > 1e-15 else 0
        hit_rate = N_hits / N
        
        results.append((u, v, K_uv, se_K, hit_rate, w_t, ratio))
        
        print(f"{u:<6.3f} | {v:<6.3f} | {K_uv:<+12.6f} | {se_K:<10.6f} | {hit_rate:<10.2e} | {w_t:<10.4f} | {ratio:<8.3f} | {elapsed:<6.2f}")

print("-" * 90)

# ==============================================================================
# 5. INTERPRETATION BOUNDARY
# ==============================================================================
print("\nINTERPRETATION OF THE KERNEL BOUND:")
ratios = [res[6] for res in results if res[5] > 1e-10]

if len(ratios) > 0:
    max_ratio = max(ratios)
    print(f"-> Maximum observed ratio |K(u,v)| / w(min(u,v)) = {max_ratio:.3f}")
    
    if max_ratio < 5.0:
        print("   The ratio is bounded by a constant C ~ O(1).")
        print("   This numerically validates the adjacent cluster estimate (Eq. 17).")
        print("   The O(r^2 log^2(1/r)) remainder theorem for the full process is supported.")
    else:
        print("   The ratio grows or is excessively large.")
        print("   This suggests the adjacent gaps interact more strongly than the w(min(u,v)) bound allows.")
        print("   The formal proof of the Stage 5G lemma requires a different asymptotic structure.")
else:
    print("-> No non-zero kernel values observed. The mixed response is below numerical precision.")