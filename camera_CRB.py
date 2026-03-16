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
plt.show()
