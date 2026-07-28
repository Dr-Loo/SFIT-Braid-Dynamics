import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate
from sfit_branching_capacity_engine import SFIT_Solver

print("=" * 70)
print("TEST 11: Intrinsic Oscillator Properties vs T_bath")
print("Measuring Period, Amplitude, Phase Diffusion, and Autocorrelation")
print("=" * 70)

g_fixed = 0.42
T_bath_values = [0.001, 0.003, 0.005, 0.01, 0.02, 0.04]
T_total = 2000.0  # Long enough for good ACF statistics, short enough to run quickly
N_TRIALS = 3

# Conservative peak finding for amplitude stats (avoids fragmentation)
PEAK_DISTANCE_TIME = 2.5  # Must be < true period (3.14) but > noise ripples
PEAK_PROMINENCE = 0.03    # Ensures we only catch macroscopic peaks

results = {
    "T_bath": [],
    "T_dom": [],       # Dominant period from ACF
    "tau_coh": [],     # Coherence time (phase diffusion) from ACF decay
    "amp_mean": [],    # Mean peak amplitude
    "amp_std": [],     # Amplitude jitter (linewidth proxy)
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for T_bath in T_bath_values:
    print(f"\nAnalyzing g = {g_fixed:.2f}, T_bath = {T_bath}...")
    
    # Aggregate stats over trials
    t_doms = []
    tau_cohs = []
    amp_means = []
    amp_stds = []
    
    for trial in range(N_TRIALS):
        seed = 3000 + int(T_bath * 10000) + trial
        sim = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=T_bath, seed=seed)
        t_arr, C_arr = sim.run(T_total=T_total, sample_every=2)
        dt_sample = t_arr[1] - t_arr[0]
        
        # ---------------------------------------------------------
        # 1. AUTOCORRELATION FUNCTION (ACF) for Period & Coherence
        # ---------------------------------------------------------
        # Normalize C_arr to have zero mean and unit variance for stable ACF
        C_norm = (C_arr - np.mean(C_arr)) / (np.std(C_arr) + 1e-8)
        
        # Compute ACF (only positive lags, up to half the signal length)
        acf = correlate(C_norm, C_norm, mode='full')
        acf = acf[len(acf)//2 : len(acf)//2 + len(C_norm)//2]
        acf = acf / acf[0]  # Normalize so ACF(0) = 1
        
        lags = np.arange(len(acf)) * dt_sample
        
        # Find dominant period: first major peak in ACF after tau > 1.0
        valid_lags = lags > 1.0
        acf_valid = acf[valid_lags]
        lags_valid = lags[valid_lags]
        
        # Use prominence to find the first true oscillation peak in ACF
        peaks_acf, _ = find_peaks(acf_valid, prominence=0.1)
        if len(peaks_acf) > 0:
            first_peak_idx = peaks_acf[0]
            t_dom = lags_valid[first_peak_idx]
            t_doms.append(t_dom)
            
            # Coherence time: find where the ACF envelope decays to 1/e (~0.368)
            # We look at the local minima between peaks to trace the envelope
            minima_acf, _ = find_peaks(-acf_valid, prominence=0.05)
            if len(minima_acf) > 0:
                # Find first minimum that drops below 1/e
                below_ethresh = np.where(acf_valid[minima_acf] < 0.368)[0]
                if len(below_ethresh) > 0:
                    tau_coh = lags_valid[minima_acf[below_ethresh[0]]]
                else:
                    tau_coh = lags_valid[-1] # Coherence lasts longer than our window
                tau_cohs.append(tau_coh)
            else:
                tau_cohs.append(np.nan)
        else:
            t_doms.append(np.nan)
            tau_cohs.append(np.nan)
            
        # ---------------------------------------------------------
        # 2. AMPLITUDE STATISTICS (using conservative peak finding)
        # ---------------------------------------------------------
        distance_samples = max(1, int(PEAK_DISTANCE_TIME / dt_sample))
        peaks, _ = find_peaks(C_arr, prominence=PEAK_PROMINENCE, distance=distance_samples)
        
        if len(peaks) > 5:
            peak_amps = C_arr[peaks]
            amp_means.append(np.mean(peak_amps))
            amp_stds.append(np.std(peak_amps))
        else:
            amp_means.append(np.nan)
            amp_stds.append(np.nan)

    # Aggregate trial results
    results["T_bath"].append(T_bath)
    results["T_dom"].append(np.nanmean(t_doms))
    results["tau_coh"].append(np.nanmean(tau_cohs))
    results["amp_mean"].append(np.nanmean(amp_means))
    results["amp_std"].append(np.nanmean(amp_stds))
    
    print(f"  -> Dominant Period: {results['T_dom'][-1]:.3f}")
    print(f"  -> Coherence Time:  {results['tau_coh'][-1]:.3f}")
    print(f"  -> Amp Mean:        {results['amp_mean'][-1]:.3f} ± {results['amp_std'][-1]:.3f}")

# ============================================================================
# PLOTTING THE INTRINSIC PROPERTIES
# ============================================================================

# Plot 1: Dominant Period vs T_bath
ax1 = axes[0, 0]
ax1.plot(results["T_bath"], results["T_dom"], 'o-', linewidth=2, markersize=8, color='blue')
ax1.axhline(3.14, color='red', linestyle='--', label='Deterministic Period (π)')
ax1.set_xlabel(r'$T_{\text{bath}}$', fontsize=12)
ax1.set_ylabel(r'Dominant Period $T_{\text{dom}}$', fontsize=12)
ax1.set_title('Period Stability vs Noise', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Coherence Time (Phase Diffusion) vs T_bath
ax2 = axes[0, 1]
ax2.plot(results["T_bath"], results["tau_coh"], 's-', linewidth=2, markersize=8, color='green')
ax2.set_xlabel(r'$T_{\text{bath}}$', fontsize=12)
ax2.set_ylabel(r'Coherence Time $\tau_{\text{coh}}$ (1/e decay)', fontsize=12)
ax2.set_title('Phase Diffusion vs Noise', fontsize=13)
ax2.grid(True, alpha=0.3)

# Plot 3: Mean Amplitude vs T_bath
ax3 = axes[1, 0]
ax3.plot(results["T_bath"], results["amp_mean"], '^-', linewidth=2, markersize=8, color='purple')
ax3.set_xlabel(r'$T_{\text{bath}}$', fontsize=12)
ax3.set_ylabel(r'Mean Peak Amplitude $\langle C_{\text{peak}} \rangle$', fontsize=12)
ax3.set_title('Amplitude vs Noise', fontsize=13)
ax3.grid(True, alpha=0.3)

# Plot 4: Amplitude Jitter (Linewidth Proxy) vs T_bath
ax4 = axes[1, 1]
ax4.plot(results["T_bath"], results["amp_std"], 'd-', linewidth=2, markersize=8, color='orange')
ax4.set_xlabel(r'$T_{\text{bath}}$', fontsize=12)
ax4.set_ylabel(r'Amplitude Jitter $\sigma_{\text{amp}}$', fontsize=12)
ax4.set_title('Amplitude Fluctuations vs Noise', fontsize=13)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sfit_intrinsic_properties_vs_tbath.png", dpi=300)
print("\n✓ Saved: sfit_intrinsic_properties_vs_tbath.png")
plt.show()

# ============================================================================
# PLOT ACF FOR VISUAL CONFIRMATION (Low vs High Noise)
# ============================================================================
fig_acf, ax_acf = plt.subplots(figsize=(10, 5))

# Low noise ACF
sim_low = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=0.001, seed=3000)
_, C_low = sim_low.run(T_total=2000.0, sample_every=2)
C_norm_low = (C_low - np.mean(C_low)) / np.std(C_low)
acf_low = correlate(C_norm_low, C_norm_low, mode='full')
acf_low = acf_low[len(acf_low)//2 : len(acf_low)//2 + len(C_low)//2] / acf_low[len(acf_low)//2]
lags_low = np.arange(len(acf_low)) * 0.01
ax_acf.plot(lags_low, acf_low, label=r'$T_{\text{bath}} = 0.001$', linewidth=2)

# High noise ACF
sim_high = SFIT_Solver(N=256, L=50.0, dt=0.005, g=g_fixed, T_bath=0.04, seed=3400)
_, C_high = sim_high.run(T_total=2000.0, sample_every=2)
C_norm_high = (C_high - np.mean(C_high)) / np.std(C_high)
acf_high = correlate(C_norm_high, C_norm_high, mode='full')
acf_high = acf_high[len(acf_high)//2 : len(acf_high)//2 + len(C_high)//2] / acf_high[len(acf_high)//2]
lags_high = np.arange(len(acf_high)) * 0.01
ax_acf.plot(lags_high, acf_high, label=r'$T_{\text{bath}} = 0.04$', linewidth=2)

ax_acf.axhline(0.368, color='red', linestyle='--', label='1/e Decay Threshold')
ax_acf.set_xlabel('Lag Time $\tau$', fontsize=12)
ax_acf.set_ylabel('Autocorrelation $R(\tau)$', fontsize=12)
ax_acf.set_title('Autocorrelation Function: Coherence Decay', fontsize=14)
ax_acf.set_xlim(0, 50)
ax_acf.legend()
ax_acf.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sfit_acf_visual_comparison.png", dpi=300)
print("✓ Saved: sfit_acf_visual_comparison.png")
plt.show()