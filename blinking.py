import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
fov       = 3e-6    # 10 µm field of view
pixel     = 20e-9    # 50 nm pixel size
Npix      = int(fov / pixel)   # 200 x 200
sigma_psf = 150e-9   # diffraction-limited PSF sigma

n_fluors  = 140
n_frames  = 250
p_on      = 0.025    # prob OFF → ON per frame (sparse blinking)
p_off     = 0.20     # prob ON  → OFF per frame

rng = np.random.default_rng(7)

# Random fluorophore positions
margin = 0.05 * fov
pos = rng.uniform(margin, fov - margin, (n_fluors, 2))

# Precompute PSF for each fluorophore (float32 to save memory)
coords = (np.arange(Npix) + 0.5) * pixel
Xg, Yg = np.meshgrid(coords, coords)
psfs = np.zeros((n_fluors, Npix, Npix), dtype=np.float32)
for i, (px, py) in enumerate(pos):
    r2 = (Xg - px)**2 + (Yg - py)**2
    psfs[i] = np.exp(-r2 / (2*sigma_psf**2)).astype(np.float32)

# Initial states: all OFF
states = np.zeros(n_fluors, dtype=bool)

def next_frame():
    global states
    # Stochastic transitions
    off_mask = ~states
    states[off_mask] = rng.random(off_mask.sum()) < p_on
    states[ states] = ~(rng.random(states.sum()) < p_off)

    # Build image from active fluorophores
    img = np.zeros((Npix, Npix), dtype=np.float32)
    on_idx = np.where(states)[0]
    if len(on_idx):
        img = psfs[on_idx].sum(axis=0)

    # Add Poisson shot noise + Gaussian readout noise
    img = rng.poisson(img * 300).astype(np.float32)
    img += rng.normal(0, 8, img.shape).astype(np.float32)
    img += 20   # background offset
    return np.clip(img, 0, None)

# Pre-generate all frames
print('Generating frames...', flush=True)
all_frames = [next_frame() for _ in range(n_frames)]
vmax = np.percentile(np.stack(all_frames), 99.5)

# --- Figure ---
fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
ax.set_facecolor('white')
ax.axis('off')

im = ax.imshow(all_frames[0], cmap='Reds_r', origin='lower',
               extent=[0, fov*1e6, 0, fov*1e6],
               vmin=0, vmax=vmax, interpolation='nearest')

fig.patch.set_facecolor('white')

def update(i):
    im.set_data(all_frames[i])
    return im,

ani = animation.FuncAnimation(fig, update, frames=n_frames,
                               interval=80, blit=True)

plt.tight_layout()
import os; os.makedirs('figs', exist_ok=True)
ani.save('figs/blinking.gif', writer='pillow', fps=15, dpi=150)
plt.show()

