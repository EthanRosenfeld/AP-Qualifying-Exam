import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Parameters
L    = 100     # nm, radius of outer beam circle
f    = 300     # nm, FWHM of doughnut
N    = 100     # detected photons
SBR0 = 5.0    # signal-to-background ratio at the origin
K    = 4       # number of beams

coeff = 4 * np.log(2) / f**2  # a = 4ln2/f^2

# Beam centers (nm)
alphas  = [0, 2*np.pi/3, 4*np.pi/3]
centers = [(0.0, 0.0)] + [(L/2 * np.cos(a), L/2 * np.sin(a)) for a in alphas]

# 2D grid
n   = 401   # odd so origin falls exactly on a grid point
lim = 75  # nm, half-width → 150 nm total field of view
x   = np.linspace(-lim, lim, n)
y   = np.linspace(-lim, lim, n)
X, Y = np.meshgrid(x, y)

# ── Step 1: intensities and their derivatives ─────────────────────────────────
Is, dI_dxs, dI_dys = [], [], []
for (bx, by) in centers:
    dx = X - bx
    dy = Y - by
    u  = dx**2 + dy**2
    exp_term = np.exp(-coeff * u)
    I        = u * exp_term
    factor   = 2 * exp_term * (1 - coeff * u)
    Is.append(I)
    dI_dxs.append(dx * factor)
    dI_dys.append(dy * factor)

S     = sum(Is)
dS_dx = sum(dI_dxs)
dS_dy = sum(dI_dys)

# ── Step 2: ideal (background-free) probabilities and derivatives ─────────────
p0s, dp0_dxs, dp0_dys = [], [], []
for i in range(K):
    p0     = Is[i] / S
    dp0_dx = (dI_dxs[i] * S - Is[i] * dS_dx) / S**2
    dp0_dy = (dI_dys[i] * S - Is[i] * dS_dy) / S**2
    p0s.append(p0)
    dp0_dxs.append(dp0_dx)
    dp0_dys.append(dp0_dy)

# ── Step 3: position-dependent SBR ───────────────────────────────────────────
# SBR(x,y) = SBR0 * S(x,y) / S(0,0)
i0      = n // 2
S0      = float(S[i0, i0])          # total signal at origin, read directly
SBR_map = SBR0 * S / S0
dSBR_dx = SBR0 * dS_dx / S0
dSBR_dy = SBR0 * dS_dy / S0

# ── Step 4: background-corrected probabilities and derivatives ────────────────
if np.isinf(SBR0):
    alpha     = np.ones_like(S)
    dalpha_dx = np.zeros_like(S)
    dalpha_dy = np.zeros_like(S)
else:
    s         = SBR_map
    alpha     = s / (s + 1)
    dalpha_dx = dSBR_dx / (s + 1)**2
    dalpha_dy = dSBR_dy / (s + 1)**2

Fxx = np.zeros_like(X)
Fyy = np.zeros_like(X)
Fxy = np.zeros_like(X)

for i in range(K):
    p0     = p0s[i]
    dp0_dx = dp0_dxs[i]
    dp0_dy = dp0_dys[i]

    # corrected probability: mixture of signal and uniform background
    p  = alpha * p0 + (1 - alpha) / K

    # corrected derivatives (product rule + chain rule on α)
    dp_dx = alpha * dp0_dx + (p0 - 1/K) * dalpha_dx
    dp_dy = alpha * dp0_dy + (p0 - 1/K) * dalpha_dy

    mask = p > 0
    Fxx[mask] += dp_dx[mask]**2        / p[mask]
    Fyy[mask] += dp_dy[mask]**2        / p[mask]
    Fxy[mask] += dp_dx[mask]*dp_dy[mask] / p[mask]

Fxx *= N
Fyy *= N
Fxy *= N

# ── Step 5: σ_CRB = sqrt((Fxx + Fyy) / (2 detF)) ────────────────────────────
det_F    = Fxx * Fyy - Fxy**2
sigma_CRB = np.sqrt((Fxx + Fyy) / (2 * det_F))  # nm
sigma_CRB[det_F <= 0] = np.nan

# Analytic check at origin (ideal, background-free):
sigma_origin_analytic = (L / (2 * np.sqrt(2 * N))) / (1 - L**2 * np.log(2) / f**2)
print(f"Analytic σ_CRB at origin (ideal): {sigma_origin_analytic:.3f} nm")

# ── Color plot ────────────────────────────────────────────────────────────────
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
ax.set_title(
    r'CRB with Background — 4-Doughnut ($N=%d$, $L=%d$ nm, SBR$_0$=%.1f)' % (N, L, SBR0)
)
for (bx, by) in centers:
    ax.plot(bx, by, 'w+', markersize=12, markeredgewidth=2)
plt.tight_layout()

# ── Slice: x=0 for several SBR0 values ───────────────────────────────────────
def compute_sigma_crb_sbr(y_arr, L_val, f_val, N_val, SBR0_val):
    X_ = np.zeros_like(y_arr)
    Y_ = y_arr
    c  = 4 * np.log(2) / f_val**2
    ctrs = [(0.0, 0.0)] + [(L_val/2 * np.cos(a), L_val/2 * np.sin(a))
                            for a in [0, 2*np.pi/3, 4*np.pi/3]]
    Is_, dIx_, dIy_ = [], [], []
    for (bx, by) in ctrs:
        dx = X_ - bx;  dy = Y_ - by
        u  = dx**2 + dy**2
        e  = np.exp(-c * u)
        Is_.append(u * e)
        fac = 2 * e * (1 - c * u)
        dIx_.append(dx * fac);  dIy_.append(dy * fac)
    S_  = sum(Is_);  dSx = sum(dIx_);  dSy = sum(dIy_)
    i0_ = np.argmin(np.abs(y_arr))
    S0_ = float(S_[i0_])            # total signal at origin
    if np.isinf(SBR0_val):
        alph = np.ones_like(S_)
        dalx = np.zeros_like(S_)
        daly = np.zeros_like(S_)
    else:
        sbr  = SBR0_val * S_ / S0_
        alph = sbr / (sbr + 1)
        dalx = SBR0_val * dSx / S0_ / (sbr + 1)**2
        daly = SBR0_val * dSy / S0_ / (sbr + 1)**2
    Fxx_ = np.zeros_like(y_arr);  Fyy_ = np.zeros_like(y_arr);  Fxy_ = np.zeros_like(y_arr)
    for i in range(4):
        p0  = Is_[i] / S_
        dp0x = (dIx_[i] * S_ - Is_[i] * dSx) / S_**2
        dp0y = (dIy_[i] * S_ - Is_[i] * dSy) / S_**2
        p   = alph * p0 + (1 - alph) / 4
        dpx = alph * dp0x + (p0 - 0.25) * dalx
        dpy = alph * dp0y + (p0 - 0.25) * daly
        m   = p > 0
        Fxx_[m] += dpx[m]**2 / p[m];  Fyy_[m] += dpy[m]**2 / p[m];  Fxy_[m] += dpx[m]*dpy[m]/p[m]
    Fxx_ *= N_val;  Fyy_ *= N_val;  Fxy_ *= N_val
    det_ = Fxx_ * Fyy_ - Fxy_**2
    sig  = np.sqrt((Fxx_ + Fyy_) / (2 * det_))
    sig[det_ <= 0] = np.nan
    return sig

SBR_vals = [5e-1, 1, 2, 5, 10, np.inf]
y_slice  = np.linspace(-lim, lim, 800)

fig2, ax2 = plt.subplots(figsize=(7, 5))
for sbr_val in SBR_vals:
    label = f'SBR$_0$ = {sbr_val:.0f}' if np.isfinite(sbr_val) else 'SBR$_0$ = ∞ (ideal)'
    sig_slice = compute_sigma_crb_sbr(y_slice, L, f, N, sbr_val)
    ax2.semilogy(y_slice, sig_slice, label=label)

ax2.set_xlabel('y (nm)')
ax2.set_ylabel(r'$\sigma_{\mathrm{CRB}}$ (nm)')
ax2.set_title(r'Slice at $x=0$, varying SBR$_0$ ($N=%d$, $L=%d$ nm)' % (N, L))
ax2.legend()
ax2.set_ylim(1, 50)
ax2.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
plt.tight_layout()

plt.show()
