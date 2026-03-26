import numpy as np
import matplotlib.pyplot as plt

N = 100            # photons
L_values = [50, 100, 150]   # nm
x0 = np.linspace(-300, 300, 5000)   # nm

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
colors = ['C0', 'C1', 'C2']

for L, color in zip(L_values, colors):
    den  = 2 * x0**2 + L**2 / 2
    dden = 4 * x0

    p1 = (x0 + L/2)**2 / den
    p2 = (x0 - L/2)**2 / den   # = 1 - p1

    dp1 = (2 * (x0 + L/2) * den - (x0 + L/2)**2 * dden) / den**2
    dp2 = (2 * (x0 - L/2) * den - (x0 - L/2)**2 * dden) / den**2

    F   = N * (dp1**2 / np.maximum(p1, 1e-15) + dp2**2 / np.maximum(p2, 1e-15))
    CRB = np.sqrt(1.0 / (2*np.maximum(F, 1e-15)))

    label = rf'$L = {L}$ nm'
    axes[0].plot(x0, F,   color=color, lw=2, label=label)
    axes[1].plot(x0, CRB, color=color, lw=2)


# ── Left: Fisher information ───────────────────────────────────────────────────
axes[0].set_xlabel(r'$x_0$ (nm)', fontsize=12)
axes[0].set_ylabel(r'Fisher information (nm$^{-2}$)', fontsize=12)
axes[0].set_title(rf'Fisher information ($N={N}$ photons)', fontsize=12)
axes[0].set_xlim(-75, 75)
axes[0].grid(True, alpha=0.3)

# ── Right: CRB ────────────────────────────────────────────────────────────────
axes[1].set_xlabel(r'$x_0$ (nm)', fontsize=12)
axes[1].set_ylabel(r'CRB (nm)', fontsize=12)
axes[1].set_title(rf'Cramér–Rao bound ($N={N}$ photons)', fontsize=12)
axes[1].set_xlim(-75, 75)
axes[1].set_ylim(0, 12)
axes[1].grid(True, alpha=0.3)

# Single shared legend below the figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(L_values) + 1,
           fontsize=11, bbox_to_anchor=(0.5, 0))

plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
plt.savefig('figs/quadratic_1D_CRB.png', dpi=150, bbox_inches='tight')
plt.show()
