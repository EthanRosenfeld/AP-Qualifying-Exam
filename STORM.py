import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import cdist

# Parameters (match blinking.py)
fov       = 3e-6
pixel     = 50e-9
Npix      = int(fov / pixel)   # 200
sigma_psf = 150e-9
d_min     = 3 * sigma_psf      # isolation criterion: >3σ from any other ON molecule

n_fluors  = 140
n_frames  = 10
p_on      = 0.025
p_off     = 0.20

rng = np.random.default_rng(7)

# Fluorophore positions
margin = 0.05 * fov
pos = rng.uniform(margin, fov - margin, (n_fluors, 2))

# Precompute PSFs
coords = (np.arange(Npix) + 0.5) * pixel
Xg, Yg = np.meshgrid(coords, coords)
psfs = np.zeros((n_fluors, Npix, Npix), dtype=np.float32)
for i, (px, py) in enumerate(pos):
    r2 = (Xg - px)**2 + (Yg - py)**2
    psfs[i] = np.exp(-r2 / (2*sigma_psf**2)).astype(np.float32)

states = np.zeros(n_fluors, dtype=bool)

def next_state():
    global states
    off_mask = ~states
    states[off_mask] = rng.random(off_mask.sum()) < p_on
    states[ states] = ~(rng.random(states.sum()) < p_off)

def build_image(on_idx):
    img = np.zeros((Npix, Npix), dtype=np.float32)
    if len(on_idx):
        img = psfs[on_idx].sum(axis=0)
    img = rng.poisson(img * 300).astype(np.float32)
    img += rng.normal(0, 8, img.shape).astype(np.float32)
    img += 20
    return np.clip(img, 0, None)

def isolated(on_idx):
    """Return indices (into on_idx) of molecules separated from all others by > d_min."""
    if len(on_idx) < 2:
        return np.arange(len(on_idx))
    pts = pos[on_idx]
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    return np.where(D.min(axis=1) > d_min)[0]

# Pre-generate frames
all_imgs   = []
all_locs   = []   # list of (x_nm, y_nm) arrays for isolated molecules

for _ in range(n_frames):
    next_state()
    on_idx = np.where(states)[0]
    all_imgs.append(build_image(on_idx))
    iso = isolated(on_idx)
    if len(iso):
        locs = pos[on_idx[iso]] * 1e6   # µm for plotting
    else:
        locs = np.empty((0, 2))
    all_locs.append(locs)

vmax = np.percentile(np.stack(all_imgs), 99.5)

# --- Figure ---
fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
ax.set_facecolor('white')
ax.axis('off')

im = ax.imshow(all_imgs[0], cmap='Reds_r', origin='lower',
               extent=[0, fov*1e6, 0, fov*1e6],
               vmin=0, vmax=vmax, interpolation='nearest')

locs0 = all_locs[0]
scat, = ax.plot(locs0[:,0] if len(locs0) else [],
                locs0[:,1] if len(locs0) else [],
                'x', color='lime', ms=8, mew=2, zorder=5)

frame_txt = ax.text(0.02, 0.97, 'frame 1', transform=ax.transAxes,
                    color='black', fontsize=9, va='top')

fig.patch.set_facecolor('white')

def update(i):
    im.set_data(all_imgs[i])
    locs = all_locs[i]
    if len(locs):
        scat.set_data(locs[:,0], locs[:,1])
    else:
        scat.set_data([], [])
    frame_txt.set_text(f'frame {i+1}')
    return im, scat, frame_txt

ani = animation.FuncAnimation(fig, update, frames=n_frames,
                               interval=600, blit=False)

plt.tight_layout()
import os; os.makedirs('figs', exist_ok=True)
ani.save('figs/STORM.gif', writer='pillow', fps=2, dpi=150)
plt.show()
