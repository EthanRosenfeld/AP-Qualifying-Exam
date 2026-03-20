import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fwhm = 100e-9   # m
fov  = 2e-6     # 5 um field of view
N    = 400
n_frames = 600

# Spatial grid
x = np.linspace(-fov/2, fov/2, N)
X, Y = np.meshgrid(x, x)

def doughnut(cx, cy):
    r2 = (X - cx)**2 + (Y - cy)**2
    return 4*np.e*np.log(2) * r2/fwhm**2 * np.exp(-4*np.log(2)*r2/fwhm**2)

# Smooth closed random curve via low-frequency Fourier modes (periodic by construction)
rng = np.random.default_rng()
bound = fov/2 * 0.7
t = np.linspace(0, 2*np.pi, n_frames, endpoint=False)
traj_x = np.zeros(n_frames)
traj_y = np.zeros(n_frames)
for k in range(1, 5):
    traj_x += rng.standard_normal()*np.cos(k*t) + rng.standard_normal()*np.sin(k*t)
    traj_y += rng.standard_normal()*np.cos(k*t) + rng.standard_normal()*np.sin(k*t)
traj_x = traj_x / np.max(np.abs(traj_x)) * bound
traj_y = traj_y / np.max(np.abs(traj_y)) * bound

# --- Figure ---
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(doughnut(traj_x[0], traj_y[0]),
               extent=[-fov/2*1e6, fov/2*1e6, -fov/2*1e6, fov/2*1e6],
               origin='lower', cmap='inferno', vmin=0, vmax=1)
dot, = ax.plot(traj_x[0]*1e6, traj_y[0]*1e6, 'o',
               color='lime', ms=2, markeredgecolor='none')
ax.set_xlabel('x (μm)')
ax.set_ylabel('y (μm)')

def update(i):
    cx, cy = traj_x[i], traj_y[i]
    im.set_data(doughnut(cx, cy))
    dot.set_data([cx*1e6], [cy*1e6])
    return im, dot

ani = animation.FuncAnimation(fig, update, frames=n_frames,
                               interval=40, blit=True)
plt.tight_layout()
import os; os.makedirs('figs', exist_ok=True)
ani.save('figs/gedankenexperiment.gif', writer='pillow', fps=50, dpi=150)
plt.show()
