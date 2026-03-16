import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

# Normalized units:
#   Pupil: rho = r / R, inside aperture rho <= 1
#   PSF:   u = pi * D * r / lambda; Rayleigh criterion: first zero at u_0 = 1.2197

N = 500
u_rayleigh = 1.2197  # first zero of J1(pi*u) / (pi*u)

# --- Pupil: two delta functions -> two shifted point sources in pupil plane ---
# Represent as two spots at ±x0 in normalized pupil coords
rho = np.linspace(-1.5, 1.5, N)
RX, RY = np.meshgrid(rho, rho)
R_pupil = np.sqrt(RX**2 + RY**2)
P = (R_pupil <= 1).astype(float)

# Mark the two point source positions in the pupil (as delta functions -> just P)
# The pupil function is the same aperture; the two-point pattern comes from incoherent sum

# --- Two-point PSF (incoherent): sum of two Airy patterns offset by Rayleigh separation ---
u_max = 7.0
u = np.linspace(-u_max, u_max, N)
UX, UY = np.meshgrid(u, u)

def airy(UX, UY, x0, y0):
    U = np.sqrt((UX - x0)**2 + (UY - y0)**2)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(U == 0, 1.0, (2*j1(np.pi*U) / (np.pi*U))**2)

# Sources placed symmetrically along u_x, separated by exactly the Rayleigh criterion
sep = u_rayleigh
I_total = airy(UX, UY, -sep/2, 0) + airy(UX, UY, +sep/2, 0)
I_total /= I_total.max()

# --- Figure ---
fig = plt.figure(figsize=(12, 5))

# Left: 3D pupil function
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(RX, RY, P, rstride=1, cstride=1, cmap='Blues',
                 linewidth=0, antialiased=False)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$y$')
ax1.set_title('Pupil')
ax1.set_zlim(0, 1.5)

# Right: 3D two-point PSF
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(UX, UY, np.sqrt(I_total), rstride=1, cstride=1, cmap='Blues',
                 linewidth=0, antialiased=False)
ax2.set_xlabel(r'$u_x$')
ax2.set_ylabel(r'$u_y$')
ax2.set_title(r'Two-point PSF (Rayleigh separation)')

import os; os.makedirs('figs', exist_ok=True)
plt.tight_layout()
plt.savefig('figs/circular_PSF_2point.png', dpi=150, bbox_inches='tight')
plt.show()
