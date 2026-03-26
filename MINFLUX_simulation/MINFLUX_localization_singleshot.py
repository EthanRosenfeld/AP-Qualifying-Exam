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
L_beam = 100e-9
FWHM   = 100e-9
alphas = [0, 2*np.pi/3, 4*np.pi/3]
beam_centers = np.array(
    [(0.0, 0.0)] + [(L_beam/2 * np.cos(a), L_beam/2 * np.sin(a)) for a in alphas]
)

# ── Doughnut model ────────────────────────────────────────────────────────────
_coeff = 4 * np.log(2) / FWHM**2

def intensity(r, r0):
    """r: (..., 2), r0: (2,) → scalar intensity at each r"""
    d = r - r0
    d2 = np.sum(d * d, axis=-1)
    return d2 * np.exp(-_coeff * d2)

def probs(r):
    """r: (..., 2) → (..., N_BEAMS) normalised probabilities"""
    I = np.stack([intensity(r, bc) for bc in beam_centers], axis=-1)
    s = I.sum(axis=-1, keepdims=True)
    s[s == 0] = 1.0
    return I / s

def log_likelihood(r, n):
    """r: (..., 2), n: (N_BEAMS,) photon counts → scalar log-likelihood at each r"""
    p = np.clip(probs(r), 1e-300, None)
    return np.sum(n * np.log(p), axis=-1)

# ── Grid MLE ──────────────────────────────────────────────────────────────────
# step sizes in metres; each stage uses 25 points per axis
step_sizes = np.array([5e-9, 1e-9, 0.1e-9, 0.01e-9])
n_pts      = 25   # points per axis per stage

def mle_grid(n_counts):
    """Run successive grid refinement MLE. Returns estimated position (2,) in metres."""
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

# ── Select which valid burst to analyse (1-indexed) ───────────────────────────
BURST_INDEX = 7   # ← change this to pick a different burst

# ── Collect all clean single-fluorophore bursts ────────────────────────────────
burst_mask = total_photons > 0
labeled, n_bursts = label(burst_mask)

valid_bursts = []   # list of (burst_bins, mol_id)
for burst_id in range(1, n_bursts + 1):
    burst_bins  = np.where(labeled == burst_id)[0]
    mol_counts  = per_mol[:, burst_bins].sum(axis=1)
    active_mols = np.where(mol_counts > 0)[0]
    if len(active_mols) == 1 and len(burst_bins) >= 5:
        valid_bursts.append((burst_bins, active_mols[0]))

print(f"{len(valid_bursts)} valid bursts found. Using burst {BURST_INDEX}.")

if BURST_INDEX < 1 or BURST_INDEX > len(valid_bursts):
    raise ValueError(f"BURST_INDEX={BURST_INDEX} out of range (1–{len(valid_bursts)}).")

selected_bins, selected_mol_id = valid_bursts[BURST_INDEX - 1]

n_counts = beam_photons[:, selected_bins].sum(axis=1)   # (N_BEAMS,)
r_est    = mle_grid(n_counts)

print(f"Fluorophore {selected_mol_id + 1}, "
      f"n={int(n_counts.sum())}, "
      f"est=({r_est[0]*1e9:.2f}, {r_est[1]*1e9:.2f}) nm, "
      f"true=({molecules[selected_mol_id, 0]*1e9:.2f}, {molecules[selected_mol_id, 1]*1e9:.2f}) nm")

burst_results = [(selected_mol_id, n_counts, r_est)]

# ── Colours ───────────────────────────────────────────────────────────────────
cmap_mol = plt.get_cmap('tab10')
mol_colors = [cmap_mol(i) for i in range(N_MOL)]

# ── FOV plot ──────────────────────────────────────────────────────────────────
fov = 120e-9
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# True fluorophore positions (numbered, coloured)
for i, (mx, my) in enumerate(molecules * 1e9):
    ax.text(mx, my, str(i + 1), color=mol_colors[i],
            fontsize=13, fontweight='bold', ha='center', va='center', zorder=5)

# MLE estimates (coloured dot + coordinate label)
for mol_id, _, r_est in burst_results:
    rx, ry = r_est * 1e9
    c = mol_colors[mol_id]
    ax.plot(rx, ry, 'o', color=c, ms=7, zorder=6)
    ax.text(rx + 2, ry + 2,
            f'({rx:.1f}, {ry:.1f}) nm',
            color=c, fontsize=7, va='bottom', zorder=7)

# Axes
half = 15.0   # nm
ax.set_xlim(-half, half)
ax.set_ylim(-half, half)
ax.set_aspect('equal')
ax.set_xlabel('x (nm)')
ax.set_ylabel('y (nm)')
# ax.set_title('MINFLUX MLE localization')

# Legend
legend_handles = [
    Line2D([0], [0], marker='o', color=mol_colors[selected_mol_id], markersize=8,
           linestyle='None', label='MLE estimate'),
] + [
    Line2D([0], [0], marker='$' + str(i+1) + '$', color=mol_colors[i],
           markersize=10, linestyle='None', label=f'Fluorophore {i+1}')
    for i in range(N_MOL)
]
ax.legend(handles=legend_handles, fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(figs_dir, 'MINFLUX_localization.png'),
            dpi=150, bbox_inches='tight')
plt.show()
