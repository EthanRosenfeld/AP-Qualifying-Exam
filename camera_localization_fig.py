import numpy as np
import matplotlib.pyplot as plt

sigma = 100e-9   # m
a = 100e-9       # pixel size, m
K_side = 9

# High-resolution Gaussian
lim = K_side / 2 * a
x = np.linspace(-lim, lim, 600)
X, Y = np.meshgrid(x, x)
I = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Pixel grid edges (nm)
edges = (np.arange(K_side + 1) - K_side // 2 - 0.5) * a * 1e9

# Pixel-integrated values via erf
from scipy.special import erf

idx = np.arange(K_side) - K_side // 2
xc = idx * a
yc = idx * a
XC, YC = np.meshgrid(xc, yc)

def gauss_integral(x0, y0, a, sigma):
    ex = erf((x0 + a/2) / (np.sqrt(2)*sigma)) - erf((x0 - a/2) / (np.sqrt(2)*sigma))
    ey = erf((y0 + a/2) / (np.sqrt(2)*sigma)) - erf((y0 - a/2) / (np.sqrt(2)*sigma))
    return 0.25 * ex * ey

I_pix = gauss_integral(XC, YC, a, sigma)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Left: continuous Gaussian with grid overlay
ax.imshow(I, extent=[-lim*1e9, lim*1e9, -lim*1e9, lim*1e9],
          origin='lower', cmap='inferno')
for e in edges:
    ax.axhline(e, color='w', lw=0.8)
    ax.axvline(e, color='w', lw=0.8)
ax.set_xlabel('x (nm)')
ax.set_ylabel('y (nm)')
ax.set_title('PSF with pixel grid')
ax.set_aspect('equal')

# Right: pixel-integrated
ax2.pcolor(edges, edges, I_pix, cmap='inferno')
for e in edges:
    ax2.axhline(e, color='w', lw=0.8)
    ax2.axvline(e, color='w', lw=0.8)
ax2.set_xlabel('x (nm)')
ax2.set_ylabel('y (nm)')
ax2.set_title('Pixel-integrated counts')
ax2.set_aspect('equal')

import os; os.makedirs('figs', exist_ok=True)
plt.tight_layout()
plt.savefig('figs/camera_localization_fig.png', dpi=150, bbox_inches='tight')
plt.show()

# PSF only (no pixel grid)
fig2, ax3 = plt.subplots(figsize=(5, 5))
ax3.imshow(I, extent=[-lim*1e9, lim*1e9, -lim*1e9, lim*1e9],
           origin='lower', cmap='inferno')
ax3.set_xlabel('x (nm)')
ax3.set_ylabel('y (nm)')
ax3.set_title('PSF')
ax3.set_aspect('equal')
plt.tight_layout()
plt.savefig('figs/camera_psf_only.png', dpi=150, bbox_inches='tight')
plt.show()
