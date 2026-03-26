import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── Grid ──────────────────────────────────────────────────────────────────────
n    = 500
lim  = 3.0
x    = np.linspace(-lim, lim, n)
X, Y = np.meshgrid(x, x)

# ── Gaussian beam helper ───────────────────────────────────────────────────────
w0  = 0.18   # beam waist at focus (grid units)
z_R = 1.8    # Rayleigh range

def gaussian_beam(prop_deg):
    """2-D Gaussian beam propagating at prop_deg degrees (CCW from +x)."""
    theta = np.radians(prop_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    t =  X*dx + Y*dy   # longitudinal coordinate
    s = -X*dy + Y*dx   # transverse coordinate
    w = w0 * np.sqrt(1 + (t / z_R)**2)
    return np.exp(-s**2 / (2 * w**2))

# Red excitation beam: propagates lower-right (enters from upper-left, 315°)
red_beam  = gaussian_beam(315.0)

# Blue reactivation beam: propagates lower-left (enters from upper-right, 225°)
# Symmetric with red about the vertical axis
blue_beam = gaussian_beam(225.0)

# ── Precompute radial distance ─────────────────────────────────────────────────
R2 = X**2 + Y**2

def make_frame(alpha):
    """alpha=1 → fully ON (green), alpha=0 → fully OFF (transparent).
    White background with subtractive beams; fluorescence via alpha-blend."""
    img = np.ones((n, n, 3))
    # Green excitation beam: absorb R and B to leave a green tint
    img[:,:,0] -= 0.35 * red_beam
    img[:,:,2] -= 0.40 * red_beam
    # Blue reactivation beam: absorb R and G to leave a blue tint
    img[:,:,0] -= 0.20 * blue_beam
    img[:,:,1] -= 0.28 * blue_beam
    img = np.clip(img, 0, 1)
    # Fluorophore ON: alpha-blend toward bright red
    core = np.exp(-R2 / (2 * 0.09**2))
    glow = np.exp(-R2 / (2 * 0.30**2))
    fluo_mask = alpha * np.tanh(core + 0.30 * glow)
    red = np.array([0.95, 0.05, 0.05])
    img = img * (1 - fluo_mask[:,:,None]) + red * fluo_mask[:,:,None]
    return img

# ── Stochastic blinking sequence ──────────────────────────────────────────────
rng      = np.random.default_rng(42)
n_frames = 300
p_off    = 0.04   # OFF per frame when ON  → mean ON  ~25 frames @ 25 fps ≈ 1 s
p_on     = 0.08   # ON  per frame when OFF → mean OFF ~12 frames @ 25 fps ≈ 0.5 s

state = True
raw   = []
for _ in range(n_frames):
    if state:
        if rng.random() < p_off:
            state = False
    else:
        if rng.random() < p_on:
            state = True
    raw.append(float(state))

# Smooth transitions with an exponential filter (τ = 3 frames)
tau     = 3.0
alpha_f = 1 - np.exp(-1 / tau)
smooth  = np.zeros(n_frames)
smooth[0] = raw[0]
for i in range(1, n_frames):
    smooth[i] = alpha_f * raw[i] + (1 - alpha_f) * smooth[i-1]

frames_data = [make_frame(a) for a in smooth]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
ax.set_facecolor('white')
ax.axis('off')

im = ax.imshow(frames_data[0], origin='lower',
               extent=[-lim, lim, -lim, lim], interpolation='bilinear')

# Static labels pointing to each beam
ax.annotate(
    'Excitation', xy=(-1.4, 1.4), xytext=(-2.4, 2.3),
    color='#CC1100', fontsize=11, fontweight='bold', ha='center', va='bottom',
    arrowprops=dict(arrowstyle='->', color='#CC1100', lw=1.5),
)

ax.annotate(
    'Reactivation', xy=(1.4, 1.4), xytext=(2.4, 2.3),
    color='#3355CC', fontsize=11, fontweight='bold', ha='center', va='bottom',
    arrowprops=dict(arrowstyle='->', color='#3355CC', lw=1.5),
)

def update(i):
    im.set_data(frames_data[i])
    return (im,)

ani = animation.FuncAnimation(fig, update, frames=n_frames,
                               interval=40, blit=True)

plt.tight_layout()
import os; os.makedirs('figs', exist_ok=True)
ani.save('figs/photoactivation.gif', writer='pillow', fps=25, dpi=150)
plt.show()
