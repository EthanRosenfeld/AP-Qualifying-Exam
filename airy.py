import numpy as np
import matplotlib.pyplot as plt

# Pupil plane coordinates
N = 10000
x_pupil = np.linspace(-3, 3, N)

# Rectangular pupil: 1 inside |k_x/k_max| <= 1, 0 outside
pupil = np.where(np.abs(x_pupil) <= 1.0, 1.0, 0.0)

# Zero-pad for high-res PSF sampling
pad_factor = 32
N_padded = N * pad_factor
pupil_padded = np.zeros(N_padded)
pupil_padded[N_padded // 2 - N // 2 : N_padded // 2 + N // 2] = pupil

# FFT to get PSF amplitude, shift so DC is at center
psf_amp = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(pupil_padded)))

# Image plane coordinates (frequency axis)
dx = x_pupil[1] - x_pupil[0]
freq = np.fft.fftshift(np.fft.fftfreq(N_padded, d=dx))

# Intensities, normalized to peak = 1
pupil_intensity = pupil**2 / pupil.max()**2
psf_intensity = np.abs(psf_amp)**2

psf_intensity /= psf_intensity.max()

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(x_pupil, pupil_intensity, color="steelblue", lw=2)
axes[0].set_xlabel(r"$k_x/k_{\max}$")
axes[0].set_ylabel("Intensity")
axes[0].set_title("Pupil")
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(-0.05, 1.15)

# Show only the central lobe region of the PSF
plot_range = 3
mask = np.abs(freq) <= plot_range
lambda_ = 1.0
z2 = 1.0
u = freq * lambda_ * z2

axes[1].plot(u[mask], psf_intensity[mask], color="tomato", lw=2)
axes[1].set_xlabel(r"$u$")
axes[1].set_ylabel("Intensity")
axes[1].set_title("PSF")
axes[1].set_xlim(-plot_range, plot_range)
axes[1].set_ylim(-0.05, 1.15)

plt.tight_layout()
plt.savefig("figs/airy.png", dpi=150)
plt.show()
