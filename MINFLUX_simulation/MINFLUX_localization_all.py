import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import label
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, 'data')
figs_dir   = os.path.join(script_dir, 'figs')
os.makedirs(figs_dir, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
t_ms         = np.load(os.path.join(data_dir, 't_ms.npy'))
beam_photons = np.load(os.path.join(data_dir, 'beam_photons.npy'))   # (N_BEAMS, N_BINS)
per_mol      = np.load(os.path.join(data_dir, 'per_molecule_photons.npy'))  # (N_MOL, N_BINS)
molecules    = np.load(os.path.join(data_dir, 'molecules.npy'))              # (N_MOL, 2) metres
total_photons = beam_photons.sum(axis=0)

N_MOL   = per_mol.shape[0]
N_BEAMS = beam_photons.shape[0]

# ── Beam geometry (must match MINFLUX_blinking.py) ────────────────────────────
L_beam = 50e-9
FWHM   = 100e-9
alphas = [0, 2*np.pi/3, 4*np.pi/3]
beam_centers = np.array(
    [(0.0, 0.0)] + [(L_beam/2 * np.cos(a), L_beam/2 * np.sin(a)) for a in alphas]
)

# ── Doughnut model ────────────────────────────────────────────────────────────
_coeff = 4 * np.log(2) / FWHM**2

def intensity(r, r0):
    d = r - r0
    d2 = np.sum(d * d, axis=-1)
    return d2 * np.exp(-_coeff * d2)

def probs(r):
    I = np.stack([intensity(r, bc) for bc in beam_centers], axis=-1)
    s = I.sum(axis=-1, keepdims=True)
    s[s == 0] = 1.0
    return I / s

def log_likelihood(r, n):
    p = np.clip(probs(r), 1e-300, None)
    return np.sum(n * np.log(p), axis=-1)

# ── Grid MLE ──────────────────────────────────────────────────────────────────
step_sizes = np.array([5e-9, 1e-9, 0.1e-9, 0.01e-9])
n_pts      = 25

def mle_grid(n_counts):
    center = np.array([0.0, 0.0])
    for step in step_sizes:
        half = step * (n_pts - 1) / 2
        x = np.linspace(center[0] - half, center[0] + half, n_pts)
        y = np.linspace(center[1] - half, center[1] + half, n_pts)
        X, Y = np.meshgrid(x, y, indexing='xy')
        R = np.stack((X, Y), axis=-1)
        ll = log_likelihood(R, n_counts)
        iy, ix = np.unravel_index(np.argmax(ll), ll.shape)
        center = np.array([X[iy, ix], Y[iy, ix]])
    return center

# ── Find all clean single-fluorophore bursts ──────────────────────────────────
burst_mask = total_photons > 0
labeled, n_bursts = label(burst_mask)

burst_results = []   # list of (mol_id, r_est)

for burst_id in range(1, n_bursts + 1):
    burst_bins  = np.where(labeled == burst_id)[0]
    mol_counts  = per_mol[:, burst_bins].sum(axis=1)
    active_mols = np.where(mol_counts > 0)[0]
    if len(active_mols) != 1 or len(burst_bins) < 5:
        continue
    mol_id   = active_mols[0]
    n_counts = beam_photons[:, burst_bins].sum(axis=1)
    r_est    = mle_grid(n_counts)
    burst_results.append((mol_id, r_est))
    print(f"  Burst {burst_id}: fluorophore {mol_id+1}, "
          f"n={int(n_counts.sum())}, "
          f"est=({r_est[0]*1e9:.2f}, {r_est[1]*1e9:.2f}) nm, "
          f"true=({molecules[mol_id,0]*1e9:.2f}, {molecules[mol_id,1]*1e9:.2f}) nm")

print(f"\nLocalised {len(burst_results)} clean bursts.")

# ── Colours ───────────────────────────────────────────────────────────────────
cmap_mol   = plt.get_cmap('tab10')
mol_colors = [cmap_mol(i) for i in range(N_MOL)]

# ── Group estimates by fluorophore ────────────────────────────────────────────
from collections import defaultdict
mol_estimates = defaultdict(list)
for mol_id, r_est in burst_results:
    mol_estimates[mol_id].append(r_est * 1e9)

# ── Figure: scatter (left) + Gaussian heatmap (right) ─────────────────────────
half = 15.0   # nm
grid_n = 300
xg = np.linspace(-half, half, grid_n)
yg = np.linspace(-half, half, grid_n)
XX, YY = np.meshgrid(xg, yg)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor('white')
for a in (ax, ax2):
    a.set_facecolor('white')

# ── Left: scatter of MLE estimates ────────────────────────────────────────────
for mol_id, r_est in burst_results:
    rx, ry = r_est * 1e9
    ax.plot(rx, ry, 'o', color=mol_colors[mol_id], ms=2,
            markeredgewidth=0, zorder=6)

ax.set_xlim(-half, half); ax.set_ylim(-half, half)
ax.set_aspect('equal')
ax.set_xlabel('x (nm)'); ax.set_ylabel('y (nm)')

# ── Right: Gaussian heatmap ────────────────────────────────────────────────────
# White background image, alpha-composite each fluorophore's Gaussian
img = np.ones((grid_n, grid_n, 3))

for mol_id in range(N_MOL):
    pts = np.array(mol_estimates[mol_id])   # (N, 2) nm
    if len(pts) < 2:
        continue
    mu  = pts.mean(axis=0)
    sx  = pts[:, 0].std()
    sy  = pts[:, 1].std()
    if sx < 1e-6 or sy < 1e-6:
        continue
    G = np.exp(-0.5 * ((XX - mu[0])**2 / sx**2 + (YY - mu[1])**2 / sy**2))
    G /= G.max()
    c = np.array(mol_colors[mol_id][:3])
    img = img * (1 - G[:, :, None]) + c * G[:, :, None]

ax2.imshow(img, extent=[-half, half, -half, half],
           origin='lower', aspect='equal', interpolation='bilinear')

ax2.set_xlim(-half, half); ax2.set_ylim(-half, half)
ax2.set_xlabel('x (nm)'); ax2.set_ylabel('y (nm)')

# ── Shared legend to the right ─────────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], marker='o', color=mol_colors[i], markersize=6,
           markeredgewidth=0, linestyle='None', label=f'Fluorophore {i+1}')
    for i in range(N_MOL)
]
ax2.legend(handles=legend_handles, fontsize=8,
           loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

plt.subplots_adjust(left=0.07, right=0.82, top=0.95, bottom=0.10, wspace=0.35)
plt.savefig(os.path.join(figs_dir, 'MINFLUX_localization_all.png'),
            dpi=150, bbox_inches='tight')
plt.show()
