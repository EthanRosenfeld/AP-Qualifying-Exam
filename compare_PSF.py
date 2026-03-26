import numpy as np
import matplotlib.pyplot as plt

sigma = 250
L_values = [25, 50, 100]
x0 = np.linspace(-2 * sigma, 2 * sigma, 4000)

fig, ax = plt.subplots(figsize=(8, 5))

# Gaussian: per-photon Fisher information is constant
ax.plot(x0, np.full_like(x0, 1 / sigma**2), 'k--', lw=2,
        label=rf'Gaussian, $\sigma={sigma}$ nm')

colors = ['C0', 'C1', 'C2']
for L, color in zip(L_values, colors):
    den = 2 * x0**2 + L**2 / 2

    p1 = (x0 + L/2)**2 / den
    p2 = (x0 - L/2)**2 / den

    dden = 4 * x0
    dp1 = (2 * (x0 + L/2) * den - (x0 + L/2)**2 * dden) / den**2
    dp2 = (2 * (x0 - L/2) * den - (x0 - L/2)**2 * dden) / den**2

    F = dp1**2 / np.maximum(p1, 1e-15) + dp2**2 / np.maximum(p2, 1e-15)
    ax.plot(x0, F, color=color, lw=2, label=rf'MINFLUX, $L={L}$ nm')

ax.set_xlabel(r'Molecule position $x_0$ (nm)', fontsize=13)
ax.set_ylabel(r'Per-photon Fisher information', fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(-2 * sigma, 2 * sigma)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figs/compare_fisher.png', dpi=150)
plt.show()