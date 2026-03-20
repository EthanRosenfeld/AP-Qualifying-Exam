import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Grid
n = 400
lim = 2.0
x = np.linspace(-lim, lim, n)
X, Y = np.meshgrid(x, x)
R2 = X**2 + Y**2

def gauss(sigma):
    return np.exp(-R2 / (2*sigma**2))

# Beam/emission profiles
beam_647 = gauss(0.38)
beam_405 = gauss(0.32)
emission = gauss(0.60)
fluo_dot = gauss(0.06)

# RGB colors
c_647  = np.array([0.0, 0.85, 0.0])   # green false color  647 nm
c_700  = np.array([1.0, 0.05, 0.0])   # bright red  671 nm emission
c_405  = np.array([0.55, 0.0, 1.0])   # violet      405 nm
c_dot  = np.array([0.75, 0.0, 0.0])   # dark red dot (ON)
c_off  = np.array([0.65, 0.65, 0.65]) # gray dot (OFF state)

def make_frame(exc647, emis, exc405, fluo_on):
    img = np.ones((n, n, 3))  # white background
    # Absorptive blending: tints white toward beam color
    img -= exc647 * beam_647[:,:,None] * (1 - c_647)
    img -= emis   * emission[:,:,None] * (1 - c_700)
    img -= exc405 * beam_405[:,:,None] * (1 - c_405)
    img -= 0.9    * fluo_dot[:,:,None] * (1 - (c_dot if fluo_on else c_off))
    return np.clip(img, 0, 1)

def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t*t*(3 - 2*t)

# Build frames
frames_data = []   # (img, title, title_color)

n_on      = 60
n_fadeout = 25
n_off     = 40
n_react   = 45
n_fadein  = 25
n_on2     = 50

# Phase 1: 647nm excitation + fluorescence ON (pulse emission gently)
for i in range(n_on):
    pulse = 1.0 + 0.12*np.sin(2*np.pi*i/15)
    frames_data.append((make_frame(0.7, 0.9*pulse, 0, True),
                        '647 nm excitation  →  671 nm fluorescence', '#CC2200'))

# Phase 2: fade out → OFF
for i in range(n_fadeout):
    t = smoothstep(i / n_fadeout)
    frames_data.append((make_frame(0.7*(1-t), 0.9*(1-t), 0, True),
                        'chemical change → fluorescence OFF', 'black'))

# Phase 3: dark / OFF state
for i in range(n_off):
    frames_data.append((make_frame(0, 0, 0, False),
                        'fluorescence OFF', 'black'))

# Phase 4: 405nm reactivation beam appears
for i in range(n_react):
    t = smoothstep(i / n_react)
    frames_data.append((make_frame(0, 0, 0.85*t, False),
                        '405 nm reactivation', '#6600CC'))

# Phase 5: fade emission back in, turn fluorophore ON
for i in range(n_fadein):
    t = smoothstep(i / n_fadein)
    fluo_on = t > 0.4
    frames_data.append((make_frame(0.7*t, 0.9*t, 0.85*(1-t), fluo_on),
                        '405 nm → fluorescence ON again', '#6600CC'))

# Phase 6: back to ON with 647nm + emission
for i in range(n_on2):
    pulse = 1.0 + 0.12*np.sin(2*np.pi*i/15)
    frames_data.append((make_frame(0.7, 0.9*pulse, 0, True),
                        '647 nm excitation  →  671 nm fluorescence', '#CC2200'))

# --- Figure ---
fig, ax = plt.subplots(figsize=(5, 5), facecolor='white')
ax.set_facecolor('white')
ax.axis('off')

im = ax.imshow(frames_data[0][0], origin='lower',
               extent=[-lim, lim, -lim, lim], interpolation='bilinear')

title = ax.set_title(frames_data[0][1], color=frames_data[0][2],
                     fontsize=11, pad=8)
fig.patch.set_facecolor('white')

def update(i):
    img, txt, col = frames_data[i]
    im.set_data(img)
    title.set_text(txt)
    title.set_color(col)
    return im, title

ani = animation.FuncAnimation(fig, update, frames=len(frames_data),
                               interval=40, blit=False)

plt.tight_layout()
import os; os.makedirs('figs', exist_ok=True)
ani.save('figs/photoactivation.gif', writer='pillow', fps=25, dpi=150)
plt.show()
