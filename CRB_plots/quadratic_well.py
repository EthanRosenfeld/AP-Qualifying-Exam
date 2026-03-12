import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# sigma_CRB = L / (4 * sqrt(N))

# Left: sigma_CRB vs L (fix N)
N_fixed = 100
L_vals = np.linspace(100e-9, 1000e-9, 300)  # meters
sigma_vs_L = L_vals / (4 * np.sqrt(N_fixed)) * 1e9  # nm

axes[0].plot(L_vals * 1e9, sigma_vs_L)
axes[0].set_xlabel("L (nm)")
axes[0].set_ylabel(r"$\sigma_{\mathrm{CRB}}$ (nm)")
axes[0].set_title(rf"$\sigma_{{\mathrm{{CRB}}}}$ vs $L$ ($N={N_fixed}$)")

# Right: sigma_CRB vs N (fix L)
L_fixed = 600e-9
N_vals = np.linspace(1, 1000, 500)
sigma_vs_N = L_fixed / (4 * np.sqrt(N_vals)) * 1e9  # nm

axes[1].plot(N_vals, sigma_vs_N)
axes[1].set_xlabel("N")
axes[1].set_ylabel(r"$\sigma_{\mathrm{CRB}}$ (nm)")
axes[1].set_title(rf"$\sigma_{{\mathrm{{CRB}}}}$ vs $N$ ($L=600$ nm)")

plt.suptitle(r"$\sigma_{\mathrm{CRB}} = \frac{L}{4\sqrt{N}}$", fontsize=14)
plt.tight_layout()

# --- Second figure: sigma_CRB = 1/sqrt(A) vs A ---
A_vals = np.linspace(1e-3, 1e-1, 300)

fig2, ax2 = plt.subplots(figsize=(6, 5))

ax2.plot(A_vals, 1 / np.sqrt(A_vals))
ax2.set_xlabel("A")
ax2.set_ylabel(r"$\sigma_{\mathrm{CRB}}$ (nm)")
ax2.set_title(r"$\sigma_{\mathrm{CRB}} = \frac{1}{\sqrt{A}}$")

plt.tight_layout()

plt.show()
