import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

# Normalized units:
#   Pupil: rho = r / R  (R = aperture radius), so rho <= 1 is inside aperture
#   PSF:   u = pi * D * r / lambda = 2*pi*R*r/lambda (first zero at u ~ 1.22*pi)
#          here we parameterize so first zero of J1 is visible

N = 500

# --- Pupil ---
rho = np.linspace(-1.5, 1.5, N)
RX, RY = np.meshgrid(rho, rho)
R_pupil = np.sqrt(RX**2 + RY**2)
P = (R_pupil <= 1).astype(float)

# --- PSF (Airy pattern) ---
u_max = 7.0   # normalized image radius; first zero ~1.22, second ~2.23
u = np.linspace(-u_max, u_max, N)
UX, UY = np.meshgrid(u, u)
U = np.sqrt(UX**2 + UY**2)
with np.errstate(invalid='ignore', divide='ignore'):
    airy = np.where(U == 0, 1.0, (2*j1(np.pi*U) / (np.pi*U))**2)

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

# Right: 3D PSF surface
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(UX, UY, np.sqrt(airy), rstride=1, cstride=1, cmap='Blues',
                 linewidth=0, antialiased=False)
ax2.set_xlabel(r'$u_x$')
ax2.set_ylabel(r'$u_y$')
ax2.set_title('PSF')

plt.tight_layout()
import os; os.makedirs('figs', exist_ok=True)
plt.savefig('figs/circular_PSF.png', dpi=150, bbox_inches='tight')
plt.show()
