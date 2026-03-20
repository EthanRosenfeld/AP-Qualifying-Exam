import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters matching 4doughnut_2d_CRB.py
L    = 100e-9    # m, constellation radius
fwhm = 100e-9    # m, doughnut FWHM
fov  = 350e-9    # m, field of view

# 4 beam positions
alphas  = [0, 2*np.pi/3, 4*np.pi/3]
centers = np.array([(0.0, 0.0)] +
                   [(L/2*np.cos(a), L/2*np.sin(a)) for a in alphas])

# Grid
N = 400
x = np.linspace(-fov/2, fov/2, N)
X, Y = np.meshgrid(x, x)

def doughnut(cx, cy):
    r2 = (X - cx)**2 + (Y - cy)**2
    return 4*np.e*np.log(2) * r2/fwhm**2 * np.exp(-4*np.log(2)*r2/fwhm**2)

# Smooth ease-in/out interpolation
def smoothstep(t):
    return t*t*(3 - 2*t)

# Build frame list: each entry is (cx, cy, dropped_markers)
n_move  = 25   # frames to travel between positions
n_dwell = 35   # frames to pause at each position
n_rounds = 3   # full cycles through all 4 positions

sequence = list(range(len(centers))) * n_rounds

frames = []
dropped = []   # accumulates beam-center markers

for idx, pos_idx in enumerate(sequence):
    p1 = centers[pos_idx]

    # Move from previous position
    if idx == 0:
        p0 = centers[sequence[-1]]   # start from last position for clean loop
    else:
        p0 = centers[sequence[idx - 1]]

    for j in range(n_move):
        t = smoothstep(j / n_move)
        cx = p0[0] + t*(p1[0] - p0[0])
        cy = p0[1] + t*(p1[1] - p0[1])
        frames.append((cx, cy, list(dropped)))

    # Dwell — drop marker at start of dwell
    dropped = dropped + [p1]
    for j in range(n_dwell):
        frames.append((p1[0], p1[1], list(dropped)))

# --- Figure ---
fig, ax = plt.subplots(figsize=(5, 5))

cx0, cy0, _ = frames[0]
im = ax.imshow(doughnut(cx0, cy0),
               extent=[-fov/2*1e9, fov/2*1e9, -fov/2*1e9, fov/2*1e9],
               origin='lower', cmap='inferno', vmin=0, vmax=1)

# Emitter at origin (animated so it renders correctly with blit=True)
emitter, = ax.plot(0, 0, 'o', color='lime', ms=3, zorder=10)

# Constellation positions (faint guide dots)
for c in centers:
    ax.plot(c[0]*1e9, c[1]*1e9, 'o', color='white', ms=3, alpha=0.3, zorder=3)

# Accumulated measurement markers (start empty)
scat = ax.scatter([], [], c='white', s=40, marker='+',
                  linewidths=1.5, zorder=4)

ax.set_xlabel('x (nm)')
ax.set_ylabel('y (nm)')

def update(i):
    cx, cy, mkrs = frames[i]
    im.set_data(doughnut(cx, cy))
    if mkrs:
        arr = np.array(mkrs) * 1e9
        scat.set_offsets(arr)
    return im, scat, emitter

ani = animation.FuncAnimation(fig, update, frames=len(frames),
                               interval=40, blit=True)

plt.tight_layout()
import os; os.makedirs('figs', exist_ok=True)
ani.save('figs/static_MINFLUX_animation.gif', writer='pillow', fps=25, dpi=150)
plt.show()
