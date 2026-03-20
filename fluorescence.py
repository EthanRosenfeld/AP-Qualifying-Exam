import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 500)  # x = I/I_sat

R_f = x / (1 + x)

fig, ax = plt.subplots()
ax.plot(x, R_f)
ax.axvline(1, color='gray', linestyle='--', label=r'$I = I_\mathrm{sat}$')
ax.set_xlabel(r'$I/I_\mathrm{sat}$')
ax.set_ylabel(r'$R_f$ (arb. units)')
# ax.set_title(r'Fluorescence Photon Rate $R_f = \frac{I}{1 + I/I_\mathrm{sat}}$')
ax.legend()
plt.tight_layout()
plt.savefig('figs/fluorescence.png', dpi=150)
plt.show()
