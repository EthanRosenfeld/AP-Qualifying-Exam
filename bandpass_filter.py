import numpy as np
import matplotlib.pyplot as plt

f_cut = 1.0

# One-sided frequency domain
f = np.linspace(0, 3.0, 1000)
H = (f <= f_cut).astype(float)

# Time domain impulse response
t = np.linspace(0, 6, 10000)
h = 2 * f_cut * np.sinc(2 * f_cut * t)

fig, axes = plt.subplots(2, 1, figsize=(4, 7))

axes[0].plot(f, H)
axes[0].set_xlabel('frequency')
axes[0].set_ylabel('H(f)')
axes[0].set_title('Hard bandpass')

axes[1].plot(t, h)
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('time')
axes[1].set_ylabel('h(t)')
axes[1].set_title('Impulse response')

plt.tight_layout()
import os; os.makedirs('figs', exist_ok=True)
plt.savefig('figs/bandpass_filter.png', dpi=150, bbox_inches='tight')
plt.show()
