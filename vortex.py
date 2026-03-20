import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
lam = 640e-9   # wavelength (m)
NA  = 1.4      # numerical aperture
f   = 1e-3     # focal length (m)

R_pupil = NA * f   # pupil radius (m)

# Pupil-plane grid
# Large padding (pad x R_pupil) -> fine focal-plane pixel size ~ lam/(2*pad*NA)
N   = 2048
pad = 20
L   = pad * R_pupil   # grid half-width (m)

x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y)
R     = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

# Overfilled Gaussian: w0 >> R_pupil so aperture is nearly uniformly illuminated
w0 = 3 * R_pupil   # Gaussian amplitude at edge: exp(-1/9) ~ 0.9 (very flat)

gaussian = np.exp(-R**2 / w0**2)
vortex   = np.exp(1j * Theta)
aperture = (R <= R_pupil).astype(float)

field = gaussian * vortex * aperture

# 2D FFT -> focal-plane field
# Focal-plane coordinate: x_f = freq_x * lambda * f
fft_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
intensity  = np.abs(fft_field)**2

dx      = x[1] - x[0]
freq    = np.fft.fftshift(np.fft.fftfreq(N, d=dx))   # cycles/m
x_f_nm  = freq * lam * f * 1e9                        # focal-plane coords in nm

# Crop to ±600 nm to frame the doughnut ring
# (expected ring radius ~ lam/(2*NA) ~ 229 nm)
plot_range_nm = 600
mask = np.abs(x_f_nm) <= plot_range_nm
idx  = np.where(mask)[0]
i0, i1 = idx[0], idx[-1] + 1

intensity_crop = intensity[i0:i1, i0:i1]
coord_nm       = x_f_nm[i0:i1]
intensity_crop /= intensity_crop.max()

fig, ax = plt.subplots(figsize=(6, 5))
extent = [coord_nm[0], coord_nm[-1], coord_nm[0], coord_nm[-1]]
im = ax.imshow(intensity_crop, origin='lower', extent=extent,
               cmap='inferno', vmin=0, vmax=1, interpolation='bilinear')

ax.set_xlabel('x (nm)', fontsize=13)
ax.set_ylabel('y (nm)', fontsize=13)
ax.set_title(fr'Doughnut beam  ($\lambda$={int(lam*1e9)} nm, NA={NA})', fontsize=14)
fig.colorbar(im, ax=ax, label='Intensity (norm.)')

plt.tight_layout()
plt.savefig('figs/vortex_doughnut.png', dpi=150)
plt.show()
