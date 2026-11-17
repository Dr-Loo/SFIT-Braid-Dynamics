#!/usr/bin/env python3
"""
Stage 5G Diagnostic: Adjacent Two-Gap Mixed Response
Tests the uniform multi-insertion cluster estimate (Eq. 17) for adjacent gaps.
Verifies if the covariance of adjacent insertion rewards scales as O(r^2 log^2(1/r)).

Usage: python stage5g_adjacent_mixed_response.py
"""

import numpy as np
import hashlib
import time
import matplotlib.pyplot as plt

# ==============================================================================
# 1. SCRIPT HASHING (Provenance)
# ==============================================================================
with open(__file__, 'rb') as f:
    script_hash = hashlib.sha256(f.read()).hexdigest()
print(f"Script SHA-256: {script_hash}\n")
print("="*80)
print("STAGE 5G: ADJACENT TWO-GAP MIXED RESPONSE DIAGNOSTIC")
print("="*80)

# ==============================================================================
# 2. EXACT EVENT-DRIVEN CROSSING OPERATOR (Vectorized Stage 5F Kernel)
# ==============================================================================
def compute_reward(X, T, q, k=0):
    """
    Computes the exact insertion reward R_k(X, T; q) for N samples.
    X, T: shape (N, 3, 2) - End and Start points on the unit circle.
    q: shape (N,) - Relaxation factor.
    k: int - Defect index (0, 1, or 2).
    """
    N = X.shape[0]
    R = np.zeros(N)
    
    # Pairs (a, b) with a < b
    pairs = [(0, 1), (0, 2), (1, 2)]
    
    for a, b in pairs:
        # Longitudinal (x) coordinates
        dx_T = T[:, a, 0] - T[:, b, 0]
        dx_X = X[:, a, 0] - X[:, b, 0]
        denom = dx_X - dx_T
        
        # Crossing time xi
        xi = np.full(N, 2.0) # Default to 2.0 so that (q < xi < 1) is False
        mask = np.abs(denom) > 1e-12
        xi[mask] = -dx_T[mask] / denom[mask]
        
        # Indicator I_ab = 1 if q < xi < 1
        I = (q < xi) & (xi < 1.0)
        
        # Transverse (y) coordinates
        dy_T = T[:, a, 1] - T[:, b, 1]
        dy_X = X[:, a, 1] - X[:, b, 1]
        
        # y-coordinate at the crossing time
        y_term = dy_T + xi * (dy_X - dy_T)
        
        # Sign of the crossing
        sign = np.sign(y_term)
        sign[y_term == 0] = 0 
        
        # Coefficient for defect k
        coeff = (k == a) - (k == b)
        
        # Accumulate reward
        R += coeff * I * sign
        
    return R

# ==============================================================================
# 3. MONTE CARLO INTEGRATION FOR ADJACENT GAPS
# ==============================================================================
def run_diagnostic(r, N_samples=2_000_000):
    """
    Computes the expected mixed product E[Delta R_a * Delta R_b] for a given r.
    """
    start_time = time.time()
    
    # 1. Generate initial state U and fresh headings V, W
    theta_U = np.random.uniform(-np.pi, np.pi, (N_samples, 3))
    theta_V = np.random.uniform(-np.pi, np.pi, (N_samples, 3))
    theta_W = np.random.uniform(-np.pi, np.pi, (N_samples, 3))
    
    U = np.stack([np.cos(theta_U), np.sin(theta_U)], axis=-1)
    V = np.stack([np.cos(theta_V), np.sin(theta_V)], axis=-1)
    W = np.stack([np.cos(theta_W), np.sin(theta_W)], axis=-1)
    
    # 2. Tumble n (Gap A)
    i = np.random.randint(0, 3, N_samples)
    Y = U.copy()
    Y[np.arange(N_samples), i] = V[np.arange(N_samples)]
    
    u = np.random.beta(3 * r, 1, N_samples)
    
    # Position after tumble n: X_n = Y + u * (U - Y)
    X_n = Y + u[:, None, None] * (U - Y)
    
    # Reward for Gap A: Delta R_a = R(U, Y; u) - R(Y, Y; 0)
    # Note: R(Y, Y; 0) is exactly 0 because start=end implies no crossings.
    R_a = compute_reward(U, Y, u, k=0)
    
    # 3. Tumble n+1 (Gap B)
    j = np.random.randint(0, 3, N_samples)
    Z_prime = X_n.copy()
    Z_prime[np.arange(N_samples), j] = W[np.arange(N_samples)]
    
    v = np.random.beta(3 * r, 1, N_samples)
    
    # Reward for Gap B: Delta R_b = R(X_n, Z'; v) - R(X_n, Z'; 0)
    R_b_v = compute_reward(X_n, Z_prime, v, k=0)
    R_b_0 = compute_reward(X_n, Z_prime, np.zeros(N_samples), k=0)
    Delta_R_b = R_b_v - R_b_0
    
    # 4. Mixed Product
    # Since E[Delta R] = 0 by symmetry, E[Delta R_a * Delta R_b] is the covariance.
    M = R_a * Delta_R_b
    mean_M = np.mean(M)
    se_M = np.std(M, ddof=1) / np.sqrt(N_samples)
    
    elapsed = time.time() - start_time
    return mean_M, se_M, elapsed

# ==============================================================================
# 4. EXECUTION AND SCALING CHECK
# ==============================================================================
r_values = [0.01, 0.005, 0.002, 0.001]
results = []

print(f"{'r':<8} | {'E[M]':<12} | {'SE':<10} | {'r^2 log^2(1/r)':<16} | {'Ratio':<10} | {'Time (s)':<8}")
print("-" * 85)

for r in r_values:
    mean_M, se_M, elapsed = run_diagnostic(r, N_samples=2_000_000)
    
    # Theoretical bound scaling factor
    bound_factor = (r**2) * (np.log(1/r)**2)
    ratio = abs(mean_M) / bound_factor if bound_factor > 0 else 0
    
    results.append((r, mean_M, se_M, bound_factor, ratio))
    
    print(f"{r:<8.4f} | {mean_M:<+12.6f} | {se_M:<10.6f} | {bound_factor:<16.6e} | {ratio:<10.4f} | {elapsed:<8.2f}")

print("-" * 85)

# ==============================================================================
# 5. INTERPRETATION BOUNDARY
# ==============================================================================
print("\nINTERPRETATION OF THE SCALING:")
ratios = [res[4] for res in results]

# Check if the ratio is roughly bounded (e.g., doesn't grow by more than a factor of 3 
# as r decreases by a factor of 10, allowing for some logarithmic drift).
if max(ratios) < 3 * min(ratios) + 1e-3:
    print("-> RESULT: BOUND CONFIRMED.")
    print("   The ratio |E[M]| / (r^2 log^2(1/r)) remains stable as r -> 0.")
    print("   This numerically validates the adjacent cluster estimate (Eq. 17).")
    print("   The O(r^2 log^2(1/r)) remainder theorem for the full process is supported.")
else:
    print("-> RESULT: BOUND VIOLATED OR SINGULAR.")
    print("   The ratio grows significantly as r -> 0.")
    print("   This suggests the adjacent gaps interact more strongly than the w(min(u,v)) bound allows.")
    print("   The O(r^2 log^2(1/r)) remainder theorem may fail, requiring a different asymptotic structure.")

# ==============================================================================
# 6. VISUALIZATION
# ==============================================================================
plt.figure(figsize=(8, 5))
plt.plot(r_values, ratios, 'o-', label='Ratio: |E[M]| / (r² log²(1/r))')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Reversal rate ratio r = γ/K')
plt.ylabel('Ratio to theoretical bound')
plt.title('Stage 5G Diagnostic: Adjacent Two-Gap Mixed Response Scaling')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('stage5g_scaling_diagnostic.png', dpi=300)
print("\nPlot saved to: stage5g_scaling_diagnostic.png")