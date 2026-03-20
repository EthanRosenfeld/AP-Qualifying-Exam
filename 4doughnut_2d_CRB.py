import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import erf

# Parameters
L = 100       # nm, radius of outer beam circle
f = 360       # nm, FWHM of doughnut
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

# --- Helper: compute sigma_CRB on arbitrary (X, Y) arrays ---
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
    Fxx_ = np.zeros_like(X, dtype=float)
    Fyy_ = np.zeros_like(X, dtype=float)
    Fxy_ = np.zeros_like(X, dtype=float)
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

# CRB at origin vs N for each L
N_vals = np.unique(np.round(np.logspace(0, 3, 60)).astype(int))
crb_origin = {}
for L_val in L_vals_slice:
    sig_N1 = compute_sigma_crb(np.array([0.0]), np.array([0.0]), L_val, f, 1)[0]
    crb_origin[L_val] = sig_N1 / np.sqrt(N_vals)

# --- Single figure with 3 subplots ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Subplot 1: 2D CRB color map
ax = axes[0]
im = ax.imshow(
    sigma_CRB,
    origin='lower',
    extent=[-lim, lim, -lim, lim],
    cmap='viridis',
    vmin=0,
    vmax=30,
)
fig.colorbar(im, ax=ax, label=r'$\sigma_{\mathrm{CRB}}$ (nm)')
ax.set_xlabel('x (nm)')
ax.set_ylabel('y (nm)')
ax.set_title(r'CRB 2D map')
for (bx, by) in centers:
    ax.plot(bx, by, 'w+', markersize=12, markeredgewidth=2)

# Subplot 2: slice at x=0 for varying L
ax2 = axes[1]
y_slice = np.linspace(-lim, lim, 800)
X0 = np.zeros_like(y_slice)
for L_val in L_vals_slice:
    sig_slice = compute_sigma_crb(X0, y_slice, L_val, f, N)
    ax2.semilogy(y_slice, sig_slice, label=f'L = {L_val} nm')
ax2.set_xlabel('y (nm)')
ax2.set_ylabel(r'$\sigma_{\mathrm{CRB}}$ (nm)')
ax2.set_title(r'Slice at $x=0$ for varying $L$ ($N=%d$)' % N)
ax2.legend(fontsize=8)
ax2.set_ylim(1, 15)
ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())

# Camera CRB (infinite SBR) at origin, scales as 1/sqrt(N)
sigma_psf_cam = 120.0   # nm
a_cam         = 100.0   # nm pixel size
K_side_cam    = 9
idx_cam = np.arange(K_side_cam) - K_side_cam//2
xc, yc = idx_cam*a_cam, idx_cam*a_cam
XXc, YYc = np.meshgrid(xc, yc)
pix_cam = np.column_stack((XXc.ravel(), YYc.ravel()))

def psf_pix(xm, ym):
    xi, yi = pix_cam[:,0], pix_cam[:,1]
    ex = erf((xi+a_cam/2-xm)/(np.sqrt(2)*sigma_psf_cam)) - erf((xi-a_cam/2-xm)/(np.sqrt(2)*sigma_psf_cam))
    ey = erf((yi+a_cam/2-ym)/(np.sqrt(2)*sigma_psf_cam)) - erf((yi-a_cam/2-ym)/(np.sqrt(2)*sigma_psf_cam))
    return 0.25*ex*ey

eps = 1e-6
p0  = psf_pix(0.0, 0.0); p0 /= p0.sum()
px  = (psf_pix(eps, 0.0) - psf_pix(-eps, 0.0))/(2*eps)
py  = (psf_pix(0.0, eps) - psf_pix(0.0, -eps))/(2*eps)
Fxx_cam = np.sum(px*px/p0); Fyy_cam = np.sum(py*py/p0); Fxy_cam = np.sum(px*py/p0)
F_cam = np.array([[Fxx_cam, Fxy_cam],[Fxy_cam, Fyy_cam]])
sigma_cam_N1 = np.sqrt(np.trace(np.linalg.inv(F_cam))/2)  # nm, at N=1
cam_crb = sigma_cam_N1 / np.sqrt(N_vals)

# Subplot 3: CRB at origin vs N for each L
ax3 = axes[2]
for L_val in L_vals_slice:
    ax3.plot(N_vals, crb_origin[L_val], label=f'L = {L_val} nm')
ax3.plot(N_vals, cam_crb, 'k-', lw=1.5, label='perfect camera')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('N (photons)')
ax3.set_ylabel(r'$\sigma_{\mathrm{CRB}}$ at origin (nm)')
ax3.set_title(r'CRB at origin vs $N$')
ax3.set_ylim(1, 100)
ax3.set_xlim(1, 1000)
ax3.legend(fontsize=7)
ax3.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax3.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

plt.tight_layout()
import os; os.makedirs('figs', exist_ok=True)
plt.savefig('figs/4doughnut_2d_CRB.png', dpi=150, bbox_inches='tight')
plt.show()
