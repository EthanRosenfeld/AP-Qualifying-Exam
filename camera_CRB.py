import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

sigma_psf = 120e-9
a = 100e-9
K_side = 9
d = 2

idx = np.arange(K_side) - K_side//2
x = idx*a
y = idx*a
xx,yy = np.meshgrid(x,y)
pixels = np.column_stack((xx.ravel(),yy.ravel()))
K = pixels.shape[0]

def psf_pixel(xm,ym):
    xi = pixels[:,0]
    yi = pixels[:,1]

    ex = erf((xi+a/2-xm)/(np.sqrt(2)*sigma_psf)) - erf((xi-a/2-xm)/(np.sqrt(2)*sigma_psf))
    ey = erf((yi+a/2-ym)/(np.sqrt(2)*sigma_psf)) - erf((yi-a/2-ym)/(np.sqrt(2)*sigma_psf))

    return 0.25*ex*ey

def p_i(xm,ym,SBR):
    psf = psf_pixel(xm,ym)

    if np.isinf(SBR):
        return psf/np.sum(psf)

    return 1/(K+SBR) + SBR/(K+SBR)*psf

def fisher(xm,ym,N,SBR):
    eps = 1e-12

    p  = p_i(xm,ym,SBR)
    px = (p_i(xm+eps,ym,SBR)-p_i(xm-eps,ym,SBR))/(2*eps)
    py = (p_i(xm,ym+eps,SBR)-p_i(xm,ym-eps,SBR))/(2*eps)

    F = np.zeros((2,2))
    F[0,0] = np.sum(px*px/p)
    F[0,1] = np.sum(px*py/p)
    F[1,0] = F[0,1]
    F[1,1] = np.sum(py*py/p)

    return N*F

def sigma_crb(N,SBR):
    F = fisher(0.0,0.0,N,SBR)
    C = np.linalg.inv(F)
    return np.sqrt(np.trace(C)/d)

N = np.logspace(1,4,200)

import os; os.makedirs('figs', exist_ok=True)

fig0, ax0 = plt.subplots()
ax0.loglog(N, sigma_psf/np.sqrt(N)*1e9, 'r--', label=r'$\sigma/\sqrt{N}$')
ax0.set_xlabel("photons")
ax0.set_ylabel("CRB uncertainty (nm)")
ax0.legend()
fig0.savefig('figs/camera_CRB.png', dpi=150, bbox_inches='tight')

fig2, ax2 = plt.subplots()
sig_inf = np.array([sigma_crb(n, np.inf) for n in N])
ax2.loglog(N, sig_inf*1e9, label=r'CRB perfect pixelated camera')
ax2.loglog(N, sigma_psf/np.sqrt(N)*1e9, 'r--', label=r'$\sigma/\sqrt{N}$')
ax2.set_xlabel("photons")
ax2.set_ylabel("CRB uncertainty (nm)")
ax2.legend()
fig2.savefig('figs/camera_CRB_inf.png', dpi=150, bbox_inches='tight')

# --- new plot: approach theory limit with smaller pixels ---
def sigma_crb_pixsize(N_phot, a_px):
    """CRB (SBR=inf) for a given pixel size a_px, centred emitter."""
    # Keep physical coverage fixed at ±5*sigma_psf regardless of pixel size
    K_side_px = max(int(np.ceil(10 * sigma_psf / a_px)), 9)
    if K_side_px % 2 == 0:
        K_side_px += 1
    idx_px = np.arange(K_side_px) - K_side_px // 2
    xp = idx_px * a_px
    xx_px, yy_px = np.meshgrid(xp, xp)
    pix = np.column_stack((xx_px.ravel(), yy_px.ravel()))
    xi, yi = pix[:, 0], pix[:, 1]

    def psf_p(xm, ym):
        ex = erf((xi + a_px/2 - xm)/(np.sqrt(2)*sigma_psf)) - erf((xi - a_px/2 - xm)/(np.sqrt(2)*sigma_psf))
        ey = erf((yi + a_px/2 - ym)/(np.sqrt(2)*sigma_psf)) - erf((yi - a_px/2 - ym)/(np.sqrt(2)*sigma_psf))
        return 0.25 * ex * ey

    def p_norm(xm, ym):
        q = psf_p(xm, ym)
        return q / q.sum()

    eps = 1e-12
    p  = p_norm(0, 0)
    px = (p_norm(eps, 0) - p_norm(-eps, 0)) / (2*eps)
    py = (p_norm(0, eps) - p_norm(0, -eps)) / (2*eps)

    Fxx = N_phot * np.sum(px**2 / p)
    Fyy = N_phot * np.sum(py**2 / p)
    Fxy = N_phot * np.sum(px * py / p)
    det = Fxx*Fyy - Fxy**2
    return np.sqrt((Fxx + Fyy) / (2 * det))

pixel_sizes = [100e-9, 30e-9, 10e-9, 3e-9]  # metres

fig3, ax3 = plt.subplots()
for a_px in pixel_sizes:
    sig_px = np.array([sigma_crb_pixsize(n, a_px) for n in N])
    ax3.loglog(N, sig_px*1e9, label=f'a = {int(a_px*1e9)} nm')
ax3.loglog(N, sigma_psf/np.sqrt(N)*1e9, 'r--', label=r'$\sigma/\sqrt{2N}$')
ax3.set_xlabel("photons")
ax3.set_ylabel("CRB uncertainty (nm)")
ax3.set_title("Camera CRB vs pixel size (SBR=∞)")
ax3.legend()
fig3.savefig('figs/camera_CRB_pixsize.png', dpi=150, bbox_inches='tight')

plt.show()
