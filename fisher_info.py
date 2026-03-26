import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 2000)

x0 = 0.0
dx0 = 0.5

def p(x, x0, sigma):
    return np.exp(-(x - x0)**2 / (2 * sigma**2)) / (np.sqrt(2*np.pi) * sigma)

sigma_small = 0.2
sigma_large = 2.0

fig, ax = plt.subplots(1, 2, figsize=(12, 3))

# small sigma
y1 = p(x, x0, sigma_small)
y2 = p(x, x0 + dx0, sigma_small)
ax[0].plot(x, y1, label=f'x0={x0}')
ax[0].plot(x, y2, '--', label=f'x0={x0+dx0}')
ax[0].set_title('small σ')
ax[0].legend()

# large sigma (same colors)
y3 = p(x, x0, sigma_large)
y4 = p(x, x0 + dx0, sigma_large)
ax[1].plot(x, y3, label=f'x0={x0}')
ax[1].plot(x, y4, '--', label=f'x0={x0+dx0}')
ax[1].set_title('large σ')
ax[1].legend()

for a in ax:
    a.set_xlabel('x')
    a.set_ylabel('p(x | x0)')

plt.tight_layout()
plt.savefig('figs/fisher_info.png', dpi=150, bbox_inches='tight')
plt.show()