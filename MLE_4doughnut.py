import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# parameters
L = 100e-9
f = 2*np.sqrt(2*np.log(2))*87e-9
N = 1000
# r_true = (np.random.rand(2) - 0.5) * L
r_true = L*np.array([10e-9, 10e-9])

# 4 beam centers
alpha = 2*np.pi*np.arange(1, 4)/3
r_beams = np.vstack((
    np.array([[0.0, 0.0]]),
    0.5*L*np.column_stack((np.cos(alpha), np.sin(alpha)))
))

def intensity(r, r0):
    d = r - r0
    d2 = np.sum(d*d, axis=-1)
    return d2*np.exp(-4*np.log(2)*d2/f**2)

def probs(r):
    I = np.stack([intensity(r, r0) for r0 in r_beams], axis=-1)
    return I/np.sum(I, axis=-1, keepdims=True)

def loglikelihood(r, n):
    p = np.clip(probs(r), 1e-300, None)
    return np.sum(n*np.log(p), axis=-1)

# synthetic measurement
p_true = probs(r_true[None, :])[0]
n = np.random.multinomial(N, p_true)

# successive grid refinement (matches MINFLUX_simulation: 25 pts/axis, steps 5/1/0.1/0.01 nm)
widths = np.array([5e-9, 1e-9, 0.1e-9, 0.01e-9]) * 24   # full width = step * (n_pts-1)
m = 25

path = []
ells = []
steps = []

center = np.array([0.0, 0.0])
for k, width in enumerate(widths):
    x = np.linspace(center[0] - width/2, center[0] + width/2, m)
    y = np.linspace(center[1] - width/2, center[1] + width/2, m)
    X, Y = np.meshgrid(x, y, indexing='xy')
    R = np.stack((X, Y), axis=-1)

    ell = loglikelihood(R, n)
    iy, ix = np.unravel_index(np.argmax(ell), ell.shape)
    center = np.array([X[iy, ix], Y[iy, ix]])

    path.append(center.copy())
    ells.append(ell[iy, ix])
    if k:
        steps.append(np.linalg.norm(path[-1] - path[-2]))

path = np.array(path)
ells = np.array(ells)
steps = np.array(steps)

# final likelihood map for visualization
pad = 20e-9
all_x = np.concatenate((r_beams[:, 0], path[:, 0], [r_true[0]]))
all_y = np.concatenate((r_beams[:, 1], path[:, 1], [r_true[1]]))

x = np.linspace(all_x.min() - pad, all_x.max() + pad, 201)
y = np.linspace(all_y.min() - pad, all_y.max() + pad, 201)
X, Y = np.meshgrid(x, y, indexing='xy')
R = np.stack((X, Y), axis=-1)
ell = loglikelihood(R, n)
iy_ml, ix_ml = np.unravel_index(np.argmax(ell), ell.shape)
r_ml = np.array([X[iy_ml, ix_ml], Y[iy_ml, ix_ml]])

# error vs N sweep for multiple L values
L_vals_crb = np.array([25, 50, 75, 100]) * 1e-9

def run_mle_L(N_sweep, L_val, r_t, p_t):
    alpha_ = 2*np.pi*np.arange(1, 4)/3
    beams_ = np.vstack((np.array([[0.0, 0.0]]),
                        0.5*L_val*np.column_stack((np.cos(alpha_), np.sin(alpha_)))))
    def intensity_(r, r0):
        d = r - r0; d2 = np.sum(d*d, axis=-1)
        return d2*np.exp(-4*np.log(2)*d2/f**2)
    def probs_(r):
        I = np.stack([intensity_(r, r0) for r0 in beams_], axis=-1)
        return I/np.sum(I, axis=-1, keepdims=True)
    def loglik_(r, n_):
        p = np.clip(probs_(r), 1e-300, None)
        return np.sum(n_*np.log(p), axis=-1)
    n_sweep = np.random.multinomial(N_sweep, p_t)
    center_s = np.array([0.0, 0.0])
    for width in widths:
        x_s = np.linspace(center_s[0] - width/2, center_s[0] + width/2, m)
        y_s = np.linspace(center_s[1] - width/2, center_s[1] + width/2, m)
        X_s, Y_s = np.meshgrid(x_s, y_s, indexing='xy')
        R_s = np.stack((X_s, Y_s), axis=-1)
        ell_s = loglik_(R_s, n_sweep)
        iy_s, ix_s = np.unravel_index(np.argmax(ell_s), ell_s.shape)
        center_s = np.array([X_s[iy_s, ix_s], Y_s[iy_s, ix_s]])
    return np.linalg.norm(center_s - r_t)

N_vals = np.unique(np.round(np.logspace(0, 3.1, 40)).astype(int))
n_trials = 20
total = len(L_vals_crb)
errors_mean_L = {}
for li, L_val in enumerate(L_vals_crb):
    print(f'MLE sweep {li+1}/{total} (L={L_val*1e9:.0f} nm)', flush=True)
    alpha_ = 2*np.pi*np.arange(1, 4)/3
    beams_ = np.vstack((np.array([[0.0, 0.0]]),
                        0.5*L_val*np.column_stack((np.cos(alpha_), np.sin(alpha_)))))
    def intensity_L(r, r0):
        d = r - r0; d2 = np.sum(d*d, axis=-1)
        return d2*np.exp(-4*np.log(2)*d2/f**2)
    def probs_L(r):
        I = np.stack([intensity_L(r, r0) for r0 in beams_], axis=-1)
        return I/np.sum(I, axis=-1, keepdims=True)
    p_t = probs_L(r_true[None, :])[0]
    means = np.zeros(len(N_vals))
    for i, Nv in enumerate(N_vals):
        errs = [run_mle_L(Nv, L_val, r_true, p_t) for _ in range(n_trials)]
        means[i] = np.mean(errs)
    errors_mean_L[L_val] = means

# CRB at r_true for several L values (scalar, scales as 1/sqrt(N))
def compute_crb_at_point(r, L_val, f_val):
    coeff = 4*np.log(2)/f_val**2
    alpha_ = 2*np.pi*np.arange(1, 4)/3
    beams_ = np.vstack((np.array([[0.0, 0.0]]),
                        0.5*L_val*np.column_stack((np.cos(alpha_), np.sin(alpha_)))))
    Is_, dIx_, dIy_ = [], [], []
    for r0 in beams_:
        dx, dy = r[0] - r0[0], r[1] - r0[1]
        u = dx**2 + dy**2
        e = np.exp(-coeff*u)
        fac = 2*e*(1 - coeff*u)
        Is_.append(u*e); dIx_.append(dx*fac); dIy_.append(dy*fac)
    S_ = sum(Is_); dSx = sum(dIx_); dSy = sum(dIy_)
    Fxx = Fyy = Fxy = 0.0
    for i in range(4):
        p = Is_[i]/S_
        if p <= 0:
            continue
        dpx = (dIx_[i]*S_ - Is_[i]*dSx)/S_**2
        dpy = (dIy_[i]*S_ - Is_[i]*dSy)/S_**2
        Fxx += dpx**2/p; Fyy += dpy**2/p; Fxy += dpx*dpy/p
    det_ = Fxx*Fyy - Fxy**2
    if det_ <= 0:
        return np.nan
    return np.sqrt((Fxx + Fyy)/(2*det_))  # at N=1

crb_lines = {}
for L_val in L_vals_crb:
    s = compute_crb_at_point(r_true, L_val, f)
    crb_lines[L_val] = s/np.sqrt(N_vals)

# camera CRB (from camera_CRB.py)
sigma_psf_cam = 120e-9
a_cam = 100e-9
K_side_cam = 9
d_cam = 2
idx_cam = np.arange(K_side_cam) - K_side_cam//2
xc = idx_cam * a_cam
yc = idx_cam * a_cam
xxc, yyc = np.meshgrid(xc, yc)
pixels_cam = np.column_stack((xxc.ravel(), yyc.ravel()))
K_cam = pixels_cam.shape[0]

def psf_pixel_cam(xm, ym):
    xi, yi = pixels_cam[:, 0], pixels_cam[:, 1]
    ex = erf((xi+a_cam/2-xm)/(np.sqrt(2)*sigma_psf_cam)) - erf((xi-a_cam/2-xm)/(np.sqrt(2)*sigma_psf_cam))
    ey = erf((yi+a_cam/2-ym)/(np.sqrt(2)*sigma_psf_cam)) - erf((yi-a_cam/2-ym)/(np.sqrt(2)*sigma_psf_cam))
    return 0.25*ex*ey

def p_i_cam(xm, ym, SBR):
    psf = psf_pixel_cam(xm, ym)
    if np.isinf(SBR):
        return psf/np.sum(psf)
    return 1/(K_cam+SBR) + SBR/(K_cam+SBR)*psf

def sigma_crb_cam(N_val, SBR):
    eps = 1e-12
    p  = p_i_cam(0.0, 0.0, SBR)
    px = (p_i_cam(eps, 0.0, SBR) - p_i_cam(-eps, 0.0, SBR))/(2*eps)
    py = (p_i_cam(0.0, eps, SBR) - p_i_cam(0.0, -eps, SBR))/(2*eps)
    Fxx = np.sum(px*px/p); Fyy = np.sum(py*py/p); Fxy = np.sum(px*py/p)
    F = np.array([[Fxx, Fxy], [Fxy, Fyy]])
    C = np.linalg.inv(N_val * F)
    return np.sqrt(np.trace(C)/d_cam)

cam_SBRs = [np.inf]
cam_crb = {sbr: np.array([sigma_crb_cam(Nv, sbr) for Nv in N_vals]) for sbr in cam_SBRs}

# plots
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

im = ax[0].contourf(X*1e9, Y*1e9, ell, levels=50)
ax[0].plot(r_beams[:, 0]*1e9, r_beams[:, 1]*1e9, 'wo', ms=8, mec='k', mew=1, label='beam centers')
ax[0].plot(r_true[0]*1e9, r_true[1]*1e9, 'rx', ms=8, mew=2, label='true position')
ax[0].plot(r_ml[0]*1e9, r_ml[1]*1e9, 'g+', ms=10, mew=2, label='max likelihood')
ax[0].set_xlim(x[0]*1e9, x[-1]*1e9)
ax[0].set_ylim(y[0]*1e9, y[-1]*1e9)
ax[0].set_aspect('equal')
ax[0].legend(fontsize=7)
ax[0].set_xlabel('x (nm)')
ax[0].set_ylabel('y (nm)')
ax[0].set_title('MLE path in space')
fig.colorbar(im, ax=ax[0], label='log likelihood')

prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
for ci, L_val in enumerate(L_vals_crb):
    color = prop_cycle[ci % len(prop_cycle)]
    ax[1].plot(N_vals, errors_mean_L[L_val]*1e9, 'o-', ms=3, lw=1.2, color=color)
    ax[1].plot(N_vals, crb_lines[L_val]*1e9, '--', lw=1.5, color=color)
    # inline label at the last point of the CRB line
    ax[1].text(N_vals[-1]*1.05, crb_lines[L_val][-1]*1e9,
               'L=%.0f nm' % (L_val*1e9), color=color, fontsize=7, va='center')
for sbr in cam_SBRs:
    label = 'perfect camera'
    ax[1].plot(N_vals, cam_crb[sbr]*1e9, '-', lw=1.5, color='0.05')
    ax[1].text(N_vals[-1]*1.05, cam_crb[sbr][-1]*1e9,
               label, color='0.05', fontsize=7, va='center')
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xlabel('N (photons)')
ax[1].xaxis.set_major_formatter(plt.ScalarFormatter())
ax[1].yaxis.set_major_formatter(plt.ScalarFormatter())
ax[1].set_ylabel('error (nm)')
ax[1].set_title('MLE error vs photon number')

plt.tight_layout()
import os; os.makedirs('figs', exist_ok=True)
plt.savefig('figs/MLE_4doughnut.png', dpi=150, bbox_inches='tight')
plt.show()

print('beam centers (nm):')
print(r_beams*1e9)
print('true position (nm):', r_true*1e9)
print('probabilities:', p_true)
print('counts:', n)
print('final estimate (nm):', path[-1]*1e9)
print('error (nm):', np.linalg.norm(path[-1] - r_true)*1e9)