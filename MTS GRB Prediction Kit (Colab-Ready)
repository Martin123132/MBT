 
📌 
MTS GRB Prediction Kit (Colab-Ready)
# ===========================
#  MTS GRB Prediction Kit
#  Single-System + Comparison
# ===========================

import numpy as np
import matplotlib.pyplot as plt

c = 3.0e8

# ======== CONFIG: GRBs TO TEST ========
grbs = [
    ("GRB 130427A", 138.0, 130.0, False),  # long collapsar
    ("GRB 050904", 225.0, 200.0, False),   # long, high-z
    ("GRB 090423", 10.0, 8.0, False),      # high-z, short-ish
    ("GRB 170817A", 2.0, 1.5, True),       # short merger GRB
]

# ========= SINGLE-GRB SWEEP + PLOT =========
def mts_single_grb(name, T90_obs, highlight_t0, is_short):
    t0_values = np.linspace(0.5 * T90_obs, 1.5 * T90_obs, 25)
    v_frac, gamma_vals = [], []
    for t0 in t0_values:
        ratio = (T90_obs / t0)**2 - 1
        if ratio > 0:
            v = np.sqrt(ratio) * c
            v_over_c = min(v / c, 0.999)
            gamma = 1 / np.sqrt(1 - v_over_c**2)
        else:
            v_over_c, gamma = 0, 1
        v_frac.append(v_over_c)
        gamma_vals.append(gamma)

    v_frac, gamma_vals = np.array(v_frac), np.array(gamma_vals)
    idx = np.argmin(np.abs(t0_values - highlight_t0))
    v_mark, Γ_mark = v_frac[idx], gamma_vals[idx]

    # Plot with dual y-axis
    fig, ax1 = plt.subplots(figsize=(7,5))
    ax1.plot(t0_values, v_frac, marker='o', color='tab:blue', label="v/c")
    ax1.axvline(highlight_t0, color='red', linestyle='--',
                label=f"t₀={highlight_t0:.1f}s → v/c≈{v_mark:.3f}, Γ≈{Γ_mark:.2f}")
    ax1.set_xlabel("Engine Timescale t₀ (s)")
    ax1.set_ylabel("Required v/c", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(t0_values, gamma_vals, linestyle=':', marker='x', color='tab:green', alpha=0.6)
    ax2.set_ylabel("Lorentz Factor Γ", color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.suptitle(f"MTS Prediction for {name}")
    fig.tight_layout()
    plt.show()

    print(f"{name}: T90_obs={T90_obs}s → At t₀={highlight_t0}s: v/c≈{v_mark:.3f}, Γ≈{Γ_mark:.2f}")

# Run single GRB sweeps one by one
for name, T90_obs, t0, is_short in grbs:
    mts_single_grb(name, T90_obs, t0, is_short)

# ========= COMBINED COMPARISON PLOT =========
plt.figure(figsize=(9,6))
for name, T90_obs, highlight_t0, is_short in grbs:
    t0_values = np.linspace(0.5 * T90_obs, 1.5 * T90_obs, 25)
    v_frac = []
    for t0 in t0_values:
        ratio = (T90_obs / t0)**2 - 1
        v_frac.append(np.sqrt(ratio) if ratio > 0 else 0)

    v_frac = np.array(v_frac)
    color = 'red' if is_short else 'blue'
    plt.plot(t0_values, v_frac, marker='o', color=color, alpha=0.7,
             label=f"{name} ({'short' if is_short else 'long'})")
    idx = np.argmin(np.abs(t0_values - highlight_t0))
    v_mark = v_frac[idx]
    plt.axvline(highlight_t0, color=color, linestyle='--', alpha=0.4)
    plt.text(highlight_t0, v_mark+0.1, f"v/c≈{v_mark:.2f}", color=color, fontsize=9, ha='center')

plt.xlabel("Engine Timescale t₀ (s)")
plt.ylabel("Required v/c")
plt.title("MTS-Implied v/c vs Engine Timescale (Long vs Short GRBs)")
plt.legend()
plt.grid(True)
plt.show()
 
✅ What This Gives You
•	Single-system sweeps with v/c + Γ vs t₀ for each GRB (one by one).
•	Combined comparison plot (all GRBs together, with highlights + v/c annotations).
•	Ready to extend: you can just add more GRBs to the grbs list at the top and rerun.
 

