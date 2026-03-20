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

SBR_list = [np.inf,500,50,5]

for SBR in SBR_list:
    sig = np.array([sigma_crb(n,SBR) for n in N])
    plt.loglog(N, sig*1e9, label=f"SBR={SBR}")

plt.xlabel("photons")
plt.ylabel("CRB uncertainty (nm)")
plt.legend()
import os; os.makedirs('figs', exist_ok=True)
plt.savefig('figs/camera_CRB.png', dpi=150, bbox_inches='tight')

fig2, ax2 = plt.subplots()
sig_inf = np.array([sigma_crb(n, np.inf) for n in N])
ax2.loglog(N, sig_inf*1e9)
ax2.set_xlabel("photons")
ax2.set_ylabel("CRB uncertainty (nm)")
fig2.savefig('figs/camera_CRB_inf.png', dpi=150, bbox_inches='tight')

# --- 2D CRB map over emitter positions (SBR=inf, N=100) ---
N_map = 100
eps_map = 1e-12
lim_map = 150e-9   # show ±1.5 pixels
n_map = 200
xm_arr = np.linspace(-lim_map, lim_map, n_map)
ym_arr = np.linspace(-lim_map, lim_map, n_map)
XM, YM = np.meshgrid(xm_arr, ym_arr)   # (n_map, n_map)

# Vectorised psf: shape (n_map, n_map, K)
xi = pixels[:,0]; yi = pixels[:,1]

def psf_grid(XM, YM):
    XM_ = XM[..., np.newaxis]; YM_ = YM[..., np.newaxis]
    ex = erf((xi - XM_ + a/2)/(np.sqrt(2)*sigma_psf)) - erf((xi - XM_ - a/2)/(np.sqrt(2)*sigma_psf))
    ey = erf((yi - YM_ + a/2)/(np.sqrt(2)*sigma_psf)) - erf((yi - YM_ - a/2)/(np.sqrt(2)*sigma_psf))
    return 0.25*ex*ey   # (n_map, n_map, K)

psf0  = psf_grid(XM, YM)
S0    = psf0.sum(axis=-1, keepdims=True)
p0    = psf0 / S0

psf_px = psf_grid(XM + eps_map, YM); psf_mx = psf_grid(XM - eps_map, YM)
psf_py = psf_grid(XM, YM + eps_map); psf_my = psf_grid(XM, YM - eps_map)

dpx = (psf_px/psf_px.sum(axis=-1,keepdims=True) - psf_mx/psf_mx.sum(axis=-1,keepdims=True)) / (2*eps_map)
dpy = (psf_py/psf_py.sum(axis=-1,keepdims=True) - psf_my/psf_my.sum(axis=-1,keepdims=True)) / (2*eps_map)

Fxx_map = N_map * np.sum(dpx**2 / p0, axis=-1)
Fyy_map = N_map * np.sum(dpy**2 / p0, axis=-1)
Fxy_map = N_map * np.sum(dpx*dpy / p0, axis=-1)

det_map = Fxx_map*Fyy_map - Fxy_map**2
sigma_map = np.sqrt((Fxx_map + Fyy_map) / (2*det_map))
sigma_map[det_map <= 0] = np.nan

fig3, ax3 = plt.subplots(figsize=(7, 6))
im3 = ax3.imshow(sigma_map*1e9,
                 origin='lower',
                 extent=[-lim_map*1e9, lim_map*1e9, -lim_map*1e9, lim_map*1e9],
                 cmap='viridis', vmin=0, vmax=30)
fig3.colorbar(im3, ax=ax3, label=r'$\sigma_{\mathrm{CRB}}$ (nm)')
# Mark pixel centers within the shown FOV
for (px_c, py_c) in pixels:
    if abs(px_c) <= lim_map*1e9 and abs(py_c) <= lim_map*1e9:
        ax3.plot(px_c*1e9, py_c*1e9, 'w+', markersize=10, markeredgewidth=1.5)
ax3.set_xlabel('x (nm)')
ax3.set_ylabel('y (nm)')
ax3.set_title(r'Camera CRB 2D map ($N=%d$, SBR=$\infty$)' % N_map)
plt.tight_layout()
fig3.savefig('figs/camera_CRB_2D_map.png', dpi=150, bbox_inches='tight')

plt.show()
