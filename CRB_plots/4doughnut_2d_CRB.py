import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Parameters
L = 100       # nm, radius of outer beam circle
f = 300       # nm, FWHM of doughnut
N = 100      # detected photons
coeff = 4 * np.log(2) / f**2  # a = 4ln2/f^2

# Beam centers (nm)
alphas = [0, 2*np.pi/3, 4*np.pi/3]
centers = [(0.0, 0.0)] + [(L/2 * np.cos(a), L/2 * np.sin(a)) for a in alphas]

# 2D grid
n = 400
lim = 75  # nm, half-width → 150 nm total field of view
x = np.linspace(-lim, lim, n)
y = np.linspace(-lim, lim, n)
X, Y = np.meshgrid(x, y)

# Accumulate intensities and their derivatives
Is, dI_dxs, dI_dys = [], [], []
for (bx, by) in centers:
    dx = X - bx
    dy = Y - by
    u = dx**2 + dy**2
    exp_term = np.exp(-coeff * u)
    I = u * exp_term
    factor = 2 * exp_term * (1 - coeff * u)
    Is.append(I)
    dI_dxs.append(dx * factor)
    dI_dys.append(dy * factor)

S = sum(Is)
dS_dx = sum(dI_dxs)
dS_dy = sum(dI_dys)

# Fisher matrix elements
Fxx = np.zeros_like(X)
Fyy = np.zeros_like(X)
Fxy = np.zeros_like(X)

for i in range(4):
    p = Is[i] / S
    dp_dx = (dI_dxs[i] * S - Is[i] * dS_dx) / S**2
    dp_dy = (dI_dys[i] * S - Is[i] * dS_dy) / S**2
    mask = p > 0
    Fxx[mask] += dp_dx[mask]**2 / p[mask]
    Fyy[mask] += dp_dy[mask]**2 / p[mask]
    Fxy[mask] += dp_dx[mask] * dp_dy[mask] / p[mask]

Fxx *= N
Fyy *= N
Fxy *= N

# sigma_CRB = sqrt((Fxx + Fyy) / (2 * det(F)))
det_F = Fxx * Fyy - Fxy**2
sigma_CRB = np.sqrt((Fxx + Fyy) / (2 * det_F))  # nm
sigma_CRB[det_F <= 0] = np.nan

# Color plot
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(
    sigma_CRB,
    origin='lower',
    extent=[-lim, lim, -lim, lim],
    cmap='viridis',
    vmin=0,
    vmax=30,
)
plt.colorbar(im, ax=ax, label=r'$\sigma_{\mathrm{CRB}}$ (nm)')
ax.set_xlabel('x (nm)')
ax.set_ylabel('y (nm)')
ax.set_title(r'CRB Localization Precision — 4-Doughnut Beam ($N=%d$, $L=%d$ nm)' % (N, L))

# Mark beam centers
for (bx, by) in centers:
    ax.plot(bx, by, 'w+', markersize=12, markeredgewidth=2)

plt.tight_layout()

# --- Slice figure: sigma_CRB along x=0 for several L values ---
def compute_sigma_crb(X, Y, L_val, f_val, N_val):
    c = 4 * np.log(2) / f_val**2
    alphas_ = [0, 2*np.pi/3, 4*np.pi/3]
    ctrs = [(0.0, 0.0)] + [(L_val/2 * np.cos(a), L_val/2 * np.sin(a)) for a in alphas_]
    Is_, dIx_, dIy_ = [], [], []
    for (bx, by) in ctrs:
        dx = X - bx;  dy = Y - by
        u = dx**2 + dy**2
        e = np.exp(-c * u)
        Is_.append(u * e)
        fac = 2 * e * (1 - c * u)
        dIx_.append(dx * fac);  dIy_.append(dy * fac)
    S_ = sum(Is_);  dSx = sum(dIx_);  dSy = sum(dIy_)
    Fxx_ = np.zeros_like(X);  Fyy_ = np.zeros_like(X);  Fxy_ = np.zeros_like(X)
    for i in range(4):
        p = Is_[i] / S_
        dpx = (dIx_[i] * S_ - Is_[i] * dSx) / S_**2
        dpy = (dIy_[i] * S_ - Is_[i] * dSy) / S_**2
        m = p > 0
        Fxx_[m] += dpx[m]**2 / p[m];  Fyy_[m] += dpy[m]**2 / p[m];  Fxy_[m] += dpx[m]*dpy[m]/p[m]
    Fxx_ *= N_val;  Fyy_ *= N_val;  Fxy_ *= N_val
    det_ = Fxx_ * Fyy_ - Fxy_**2
    sig = np.sqrt((Fxx_ + Fyy_) / (2 * det_))
    sig[det_ <= 0] = np.nan
    return sig

L_vals_slice = [50, 75, 100, 125, 150]
y_slice = np.linspace(-lim, lim, 800)
X0 = np.zeros_like(y_slice)

fig2, ax2 = plt.subplots(figsize=(7, 5))
for L_val in L_vals_slice:
    sig_slice = compute_sigma_crb(X0, y_slice, L_val, f, N)
    ax2.semilogy(y_slice, sig_slice, label=f'L = {L_val} nm')

ax2.set_xlabel('y (nm)')
ax2.set_ylabel(r'$\sigma_{\mathrm{CRB}}$ (nm)')
ax2.set_title(r'Slice at $x=0$ for varying $L$ ($N=%d$)' % N)
ax2.legend()
ax2.set_ylim(1, 15)
ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
plt.tight_layout()

plt.show()
