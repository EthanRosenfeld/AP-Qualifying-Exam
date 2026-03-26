import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
L    = 100e-9   # m, MINFLUX constellation radius
fwhm = 100e-9   # m, doughnut FWHM
fov  = 350e-9   # m, field of view

# Grid
Ng = 400
x  = np.linspace(-fov/2, fov/2, Ng)
X, Y = np.meshgrid(x, x)

def doughnut(cx, cy):
    r2 = (X - cx)**2 + (Y - cy)**2
    return 4*np.e*np.log(2) * r2/fwhm**2 * np.exp(-4*np.log(2)*r2/fwhm**2)

def smoothstep(t):
    return t*t*(3 - 2*t)

# MINFLUX constellation
alphas  = [0, 2*np.pi/3, 4*np.pi/3]
centers = np.array([(0.0, 0.0)] +
                   [(L/2*np.cos(a), L/2*np.sin(a)) for a in alphas])

n_move   = 25   # frames to travel between positions
n_dwell  = 35   # frames to pause at each position
n_rounds = 1

frames = []   # (img, mf_markers)

sequence   = list(range(len(centers))) * n_rounds
mf_dropped = []

for idx, pos_idx in enumerate(sequence):
    p1 = centers[pos_idx]
    p0 = centers[sequence[idx - 1]] if idx > 0 else centers[sequence[-1]]

    for j in range(n_move):
        t  = smoothstep(j / n_move)
        cx = p0[0] + t*(p1[0] - p0[0])
        cy = p0[1] + t*(p1[1] - p0[1])
        frames.append((doughnut(cx, cy), list(mf_dropped)))

    mf_dropped = mf_dropped + [p1]
    for j in range(n_dwell):
        frames.append((doughnut(p1[0], p1[1]), list(mf_dropped)))

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))

im = ax.imshow(frames[0][0],
               extent=[-fov/2*1e9, fov/2*1e9, -fov/2*1e9, fov/2*1e9],
               origin='lower', cmap='inferno', vmin=0, vmax=1)

mf_scat = ax.scatter([], [], c='red', s=40, marker='+',
                     linewidths=1.5, zorder=5)
emitter, = ax.plot(0, 0, 'o', color='lime', ms=3, zorder=10)

ax.set_xlabel('x (nm)')
ax.set_ylabel('y (nm)')
ax.set_title('MINFLUX localization')

def update(i):
    img, mf_mkrs = frames[i]
    im.set_data(img)
    if mf_mkrs:
        mf_scat.set_offsets(np.array(mf_mkrs)*1e9)
    else:
        mf_scat.set_offsets(np.empty((0, 2)))
    return im, mf_scat, emitter

ani = animation.FuncAnimation(fig, update, frames=len(frames),
                               interval=40, blit=False)

plt.tight_layout()
import os; os.makedirs('figs', exist_ok=True)
ani.save('figs/minflux_animation.gif', writer='pillow', fps=25, dpi=150)
plt.show()
