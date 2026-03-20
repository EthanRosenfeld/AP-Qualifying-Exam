import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
sigma_cam = 100e-9   # m, Gaussian PSF sigma for camera
L         = 100e-9   # m, MINFLUX constellation radius
fwhm      = 100e-9   # m, doughnut FWHM
fov       = 350e-9   # m, field of view

# Grid
Ng = 400
x  = np.linspace(-fov/2, fov/2, Ng)
X, Y = np.meshgrid(x, x)

def gaussian_psf():
    return np.exp(-(X**2 + Y**2) / (2*sigma_cam**2))

def doughnut(cx, cy):
    r2 = (X - cx)**2 + (Y - cy)**2
    return 4*np.e*np.log(2) * r2/fwhm**2 * np.exp(-4*np.log(2)*r2/fwhm**2)

def smoothstep(t):
    return t*t*(3 - 2*t)

# MINFLUX constellation
alphas  = [0, 2*np.pi/3, 4*np.pi/3]
centers = np.array([(0.0, 0.0)] +
                   [(L/2*np.cos(a), L/2*np.sin(a)) for a in alphas])

# ── Frame list ──────────────────────────────────────────────────────────────
# Each frame: (image_array, cam_markers, minflux_markers, emitter_visible)

n_cam_dwell   = 60   # show Gaussian before measurement
n_cam_measure = 40   # show Gaussian + blue marker
n_transition  = 20   # blank/pause before MINFLUX starts
n_move        = 25   # frames to travel between MINFLUX positions
n_dwell       = 35   # frames to pause at each MINFLUX position
n_rounds      = 1

frames = []   # (img, cam_mkrs, mf_mkrs, phase)

# Phase 1a: show Gaussian, no marker yet
gauss = gaussian_psf()
for _ in range(n_cam_dwell):
    frames.append((gauss, [], [], 'camera'))

# Phase 1b: drop blue camera marker
for _ in range(n_cam_measure):
    frames.append((gauss, [(0.0, 0.0)], [], 'camera'))

# Phase 2: MINFLUX
sequence = list(range(len(centers))) * n_rounds
mf_dropped = []

for idx, pos_idx in enumerate(sequence):
    p1 = centers[pos_idx]
    p0 = centers[sequence[idx - 1]] if idx > 0 else centers[sequence[-1]]

    for j in range(n_move):
        t  = smoothstep(j / n_move)
        cx = p0[0] + t*(p1[0] - p0[0])
        cy = p0[1] + t*(p1[1] - p0[1])
        frames.append((doughnut(cx, cy), [(0.0, 0.0)], list(mf_dropped), 'minflux'))

    mf_dropped = mf_dropped + [p1]
    for j in range(n_dwell):
        frames.append((doughnut(p1[0], p1[1]), [(0.0, 0.0)], list(mf_dropped), 'minflux'))

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))

img0 = frames[0][0]
im = ax.imshow(img0,
               extent=[-fov/2*1e9, fov/2*1e9, -fov/2*1e9, fov/2*1e9],
               origin='lower', cmap='inferno', vmin=0, vmax=1)


cam_scat = ax.scatter([], [], c='deepskyblue', s=60, marker='+',
                      linewidths=2, zorder=6)
mf_scat  = ax.scatter([], [], c='red', s=40, marker='+',
                      linewidths=1.5, zorder=5)
emitter, = ax.plot(0, 0, 'o', color='lime', ms=3, zorder=10)

ax.set_xlabel('x (nm)')
ax.set_ylabel('y (nm)')
title = ax.set_title('Camera localization')

def update(i):
    img, cam_mkrs, mf_mkrs, phase = frames[i]
    im.set_data(img)
    title.set_text('Camera localization' if phase == 'camera' else 'MINFLUX localization')
    if cam_mkrs:
        cam_scat.set_offsets(np.array(cam_mkrs)*1e9)
    else:
        cam_scat.set_offsets(np.empty((0, 2)))
    if mf_mkrs:
        mf_scat.set_offsets(np.array(mf_mkrs)*1e9)
    return im, cam_scat, mf_scat, emitter, title

ani = animation.FuncAnimation(fig, update, frames=len(frames),
                               interval=40, blit=False)

plt.tight_layout()
import os; os.makedirs('figs', exist_ok=True)
ani.save('figs/camera_then_MINFLUX_animation.gif', writer='pillow', fps=25, dpi=150)
plt.show()