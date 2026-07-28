import numpy as np
import matplotlib.pyplot as plt
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("CRITICAL NUMERICS CHECK: Timestep Convergence at High Noise")
print("Testing g=0.42, T_bath=0.04 with dt=0.005 vs dt=0.0025")
print("=" * 70)

g_fixed = 0.42
T_bath_high = 0.04
T_total = 1000.0  # Long enough to reach steady state, short enough to run fast
seed = 4200

results = {}
dt_values = [0.005, 0.0025]

for dt in dt_values:
    print(f"\nRunning with dt = {dt}...")
    sim = SFIT_Solver(N=256, L=50.0, dt=dt, g=g_fixed, T_bath=T_bath_high, seed=seed)
    
    # We need the full field to check actual chi amplitude, not just C_loc
    t_arr = []
    C_arr = []
    chi_max_arr = []
    
    n_steps = int(T_total / dt)
    sample_every = max(1, int(0.05 / dt)) # Sample every 0.05 time units
    
    for step in range(n_steps):
        t = step * dt
        sim.step(t)
        if step % sample_every == 0:
            t_arr.append(t)
            C_arr.append(sim.compute_C_loc(sim.chi, sim.chi_dot))
            chi_max_arr.append(np.max(np.abs(sim.chi)))
            
    C_arr = np.array(C_arr)
    chi_max_arr = np.array(chi_max_arr)
    
    # Discard first 200 time units as transient
    transient_mask = np.array(t_arr) > 200.0
    C_steady = C_arr[transient_mask]
    chi_steady = chi_max_arr[transient_mask]
    
    results[dt] = {
        "C_mean": np.mean(C_steady),
        "C_max": np.max(C_steady),
        "chi_max_mean": np.mean(chi_steady),
        "chi_max_max": np.max(chi_steady),
        "t": np.array(t_arr)[transient_mask],
        "C": C_steady
    }
    
    print(f"  Steady-state C_loc:  Mean = {results[dt]['C_mean']:.4f}, Max = {results[dt]['C_max']:.4f}")
    print(f"  Steady-state |chi|:  Mean Max = {results[dt]['chi_max_mean']:.4f}, Absolute Max = {results[dt]['chi_max_max']:.4f}")

# Compare the two
dt1, dt2 = 0.005, 0.0025
c_mean_ratio = results[dt1]['C_mean'] / results[dt2]['C_mean']
chi_max_ratio = results[dt1]['chi_max_mean'] / results[dt2]['chi_max_mean']

print("\n" + "=" * 70)
print("CONVERGENCE ANALYSIS:")
print("=" * 70)
print(f"C_loc Mean Ratio (dt=0.005 / dt=0.0025): {c_mean_ratio:.3f}")
print(f"|chi| Max Mean Ratio (dt=0.005 / dt=0.0025): {chi_max_ratio:.3f}")

if 0.90 <= c_mean_ratio <= 1.10 and 0.90 <= chi_max_ratio <= 1.10:
    print("\n✓ PASS: Results are converged within 10%. dt=0.005 is trustworthy.")
else:
    print("\n✗ FAIL: Significant numerical drift detected.")
    print("The high-amplitude results at T_bath=0.04 are likely integration artifacts.")
    print("The entire T_bath sweep must be re-run with dt=0.0025 (or smaller).")

# Visual comparison of the steady-state C_loc
plt.figure(figsize=(10, 5))import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate, welch
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TRANSITION ANALYSIS: Confinement Breakdown & 5-Method Spectral Check")
print("=" * 70)

g_fixed = 0.42
# Fine sweep to locate the exact transition point
T_bath_fine = [0.003, 0.004, 0.005, 0.006, 0.008, 0.010]
T_total = 2000.0  # Long enough for good spectral stats, fast enough to run
N_TRIALS = 3

# Robust peak detection parameters
PEAK_DISTANCE_TIME = 2.0
PEAK_PROMINENCE = 0.05

print("\n[1/2] Fine-grained Amplitude Sweep to locate confinement breakdown...")
amp_means = []
for T_bath in T_bath_fine:
    trial_amps = []
    for trial in range(N_TRIALS):
        seed = 5000 + int(T_bath * 10000) + trial
        sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)
        t_arr, C_arr = sim.run(T_total=T_total, sample_every=2)
        
        # Discard first 200 time units as transient
        mask = np.array(t_arr) > 200.0
        C_steady = C_arr[mask]
        trial_amps.append(np.mean(C_steady))
        
    mean_amp = np.mean(trial_amps)
    amp_means.append(mean_amp)
    print(f"  T_bath = {T_bath:.3f} -> Mean C_loc = {mean_amp:.4f}")

# Plot Amplitude Transition
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(T_bath_fine, amp_means, 'o-', linewidth=2, markersize=8, color='darkred')
ax1.set_xlabel(r'$T_{\text{bath}}$', fontsize=13)
ax1.set_ylabel(r'Mean Steady-State $C_{\text{loc}}$', fontsize=13)
ax1.set_title('Amplitude Transition: Locating Confinement Breakdown', fontsize=14)
ax1.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("sfit_amplitude_transition_fine.png", dpi=300)
print("\n✓ Saved: sfit_amplitude_transition_fine.png")


print("\n[2/2] 5-Method Spectral Comparison at Key Regimes...")
# Pick 3 representative points: confined, transition, breakdown
# Adjust these based on the plot above, e.g., 0.003, 0.006, 0.010
key_Tbaths = [0.003, 0.006, 0.010] 
dt_sample = 0.01  # sample_every=2 * dt=0.005

spectral_results = {tb: {} for tb in key_Tbaths}

fig2, axes = plt.subplots(3, 2, figsize=(14, 12))

for idx, T_bath in enumerate(key_Tbaths):
    print(f"\nAnalyzing T_bath = {T_bath}...")
    sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=6000)
    t_arr, C_arr = sim.run(T_total=T_total, sample_every=2)
    
    # Discard transient
    mask = np.array(t_arr) > 200.0
    t_steady = np.array(t_arr)[mask]
    C_steady = C_arr[mask]
    
    # ---------------------------------------------------------
    # METHOD 1: Peak Detector Period
    # ---------------------------------------------------------
    distance_samples = max(1, int(PEAK_DISTANCE_TIME / dt_sample))
    peaks, _ = find_peaks(C_steady, prominence=PEAK_PROMINENCE, distance=distance_samples)
    if len(peaks) > 2:
        intervals = np.diff(t_steady[peaks])
        period_peak = np.median(intervals)
    else:
        period_peak = np.nan
    spectral_results[T_bath]['period_peak'] = period_peak
    
    # ---------------------------------------------------------
    # METHOD 2 & 5: FFT and Power Spectrum (Welch)
    # ---------------------------------------------------------
    C_detrended = C_steady - np.mean(C_steady)
    freqs, psd = welch(C_detrended, fs=1.0/dt_sample, nperseg=min(1024, len(C_detrended)))
    
    # Top 5 FFT frequencies
    top_5_idx = np.argsort(psd)[-5:][::-1]
    top_5_freqs = freqs[top_5_idx]
    top_5_powers = psd[top_5_idx]
    spectral_results[T_bath]['top_5_freqs'] = top_5_freqs
    
    # Dominant frequency (excluding DC at index 0)
    dom_idx = np.argmax(psd[1:]) + 1
    dom_freq = freqs[dom_idx]
    period_fft = 1.0 / dom_freq if dom_freq > 0 else np.nan
    spectral_results[T_bath]['period_fft'] = period_fft
    
    # ---------------------------------------------------------
    # METHOD 3 & 4: Autocorrelation (First peak & 1/e decay)
    # ---------------------------------------------------------
    C_norm = (C_detrended) / (np.std(C_detrended) + 1e-8)
    acf = correlate(C_norm, C_norm, mode='full')
    acf = acf[len(acf)//2 : len(acf)//2 + len(C_detrended)//2]
    acf = acf / acf[0]
    lags = np.arange(len(acf)) * dt_sample
    
    # First peak after lag > 1.0
    valid_lags = lags > 1.0
    acf_valid = acf[valid_lags]
    lags_valid = lags[valid_lags]
    peaks_acf, _ = find_peaks(acf_valid, prominence=0.1)
    
    if len(peaks_acf) > 0:
        period_acf = lags_valid[peaks_acf[0]]
    else:
        period_acf = np.nan
    spectral_results[T_bath]['period_acf'] = period_acf
    
    # 1/e decay (coherence time)
    minima_acf, _ = find_peaks(-acf_valid, prominence=0.05)
    if len(minima_acf) > 0:
        below_e = np.where(acf_valid[minima_acf] < 0.368)[0]
        if len(below_e) > 0:
            tau_coh = lags_valid[minima_acf[below_e[0]]]
        else:
            tau_coh = lags_valid[-1]
    else:
        tau_coh = np.nan
    spectral_results[T_bath]['tau_coh'] = tau_coh
    
    # Print summary
    print(f"  1. Peak Detector Period: {period_peak:.3f}")
    print(f"  2. FFT Top 5 Freqs (Hz): {top_5_freqs}")
    print(f"     -> FFT Dominant Period: {period_fft:.3f}")
    print(f"  3. ACF First Peak Period: {period_acf:.3f}")
    print(f"  4. ACF 1/e Coherence Time: {tau_coh:.3f}")
    print(f"  5. Power Spectrum: (Plotted below)")
    
    # Plotting
    ax_period = axes[idx, 0]
    ax_psd = axes[idx, 1]
    
    # Plot ACF
    ax_period.plot(lags[:100], acf[:100], linewidth=1.5, color='blue')
    ax_period.axhline(0.368, color='red', linestyle='--', alpha=0.7, label='1/e threshold')
    ax_period.set_ylabel('ACF $R(\\tau)$', fontsize=11)
    ax_period.set_title(f'T_bath = {T_bath} (Coherence: {tau_coh:.1f})', fontsize=12)
    ax_period.set_xlim(0, 50)
    ax_period.grid(True, alpha=0.3)
    if idx == 2:
        ax_period.set_xlabel('Lag Time $\\tau$', fontsize=11)
        
    # Plot PSD
    ax_psd.plot(freqs, psd, linewidth=1.5, color='darkgreen')
    ax_psd.set_ylabel('Power Spectral Density', fontsize=11)
    ax_psd.set_title(f'T_bath = {T_bath} (Dom Freq: {dom_freq:.3f} Hz)', fontsize=12)
    ax_psd.set_xlim(0, 1.0)  # Focus on low frequencies where the action is
    ax_psd.grid(True, alpha=0.3)
    if idx == 2:
        ax_psd.set_xlabel('Frequency (Hz)', fontsize=11)

plt.tight_layout()
plt.savefig("sfit_5_method_spectral_comparison.png", dpi=300)
print("\n✓ Saved: sfit_5_method_spectral_comparison.png")
plt.show()
plt.plot(results[0.005]['t'], results[0.005]['C'], linewidth=1, alpha=0.7, label=f'dt = {0.005} (Mean C = {results[0.005]["C_mean"]:.3f})')
plt.plot(results[0.0025]['t'], results[0.0025]['C'], linewidth=1, alpha=0.7, label=f'dt = {0.0025} (Mean C = {results[0.0025]["C_mean"]:.3f})')
plt.xlabel('Time $t$', fontsize=12)
plt.ylabel(r'$C_{\rm loc}(t)$', fontsize=12)
plt.title('Timestep Convergence Check: High Noise ($T_{\text{bath}} = 0.04$)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sfit_timestep_convergence_check.png", dpi=300)
print("\n✓ Saved: sfit_timestep_convergence_check.png")
plt.show()