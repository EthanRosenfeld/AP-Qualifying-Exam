import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import label
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, 'data')
figs_dir   = os.path.join(script_dir, 'figs')
os.makedirs(figs_dir, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
t_ms          = np.load(os.path.join(data_dir, 't_ms.npy'))
beam_photons  = np.load(os.path.join(data_dir, 'beam_photons.npy'))   # (N_BEAMS, N_BINS)
per_mol       = np.load(os.path.join(data_dir, 'per_molecule_photons.npy'))  # (N_MOL, N_BINS)
total_photons = beam_photons.sum(axis=0)

# Convert photon counts to count rate in kHz
dt_s = (t_ms[1] - t_ms[0]) * 1e-3        # bin width in seconds
beam_rate_khz = beam_photons / dt_s / 1e3  # (N_BEAMS, N_BINS)

N_BEAMS = beam_photons.shape[0]
N_MOL   = per_mol.shape[0]

# ── Colours ───────────────────────────────────────────────────────────────────
cmap_beam = plt.get_cmap('Set1')
beam_colors = [cmap_beam(i) for i in range(N_BEAMS)]          # r0–r3
cmap_mol  = plt.get_cmap('tab10')
mol_colors  = [cmap_mol(i)  for i in range(N_MOL)]            # fluorophores

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ── Background highlights: per-burst fluorophore attribution ──────────────────
# A burst is a contiguous run of bins where total_photons > 0
burst_mask = total_photons > 0
labeled, n_bursts = label(burst_mask)

for burst_id in range(1, n_bursts + 1):
    burst_bins = np.where(labeled == burst_id)[0]
    t_start = t_ms[burst_bins[0]]
    t_end   = t_ms[burst_bins[-1]]

    # Which molecules contributed photons during this burst?
    mol_counts = per_mol[:, burst_bins].sum(axis=1)   # (N_MOL,)
    active_mols = np.where(mol_counts > 0)[0]

    if len(active_mols) > 1:
        color = (0.75, 0.75, 0.75, 0.35)   # gray for overlaps
    elif len(active_mols) == 1:
        c = mol_colors[active_mols[0]]
        color = (*c[:3], 0.25)
    else:
        continue

    ax.axvspan(t_start, t_end, color=color, linewidth=0, zorder=0)

# ── Stacked lines: beam contributions ─────────────────────────────────────────
cumulative = np.zeros(len(t_ms))
for j in range(N_BEAMS):
    cumulative += beam_rate_khz[j]
    ax.plot(t_ms, cumulative, color=beam_colors[j], lw=1.2, zorder=2,
            label=f'$r_{j}$')

# ── Axes ──────────────────────────────────────────────────────────────────────
T_s = t_ms[-1] / 1e3
ax.set_xlim(t_ms[0], t_ms[-1])
ax.set_xticks([0, t_ms[-1]])
ax.set_xticklabels(['0 s', f'{T_s:.0f} s'])
ax.set_xlabel('Time')
ax.set_ylabel('Count rate (kHz)')
# ax.set_title('MINFLUX photon time trace — beam contributions and fluorophore bursts')

# ── Legend ────────────────────────────────────────────────────────────────────
beam_patches = [mpatches.Patch(color=beam_colors[j], label=f'$r_{j}$')
                for j in range(N_BEAMS)]
mol_patches  = [mpatches.Patch(facecolor=(*mol_colors[i][:3], 0.4),
                                edgecolor='none',
                                label=f'Fluorophore {i+1}')
                for i in range(N_MOL)]
overlap_patch = mpatches.Patch(facecolor=(0.75, 0.75, 0.75, 0.5),
                                edgecolor='none', label='>1 fluorophore active')

ax.legend(handles=beam_patches + mol_patches + [overlap_patch],
          fontsize=8, loc='upper right', ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(figs_dir, 'MINFLUX_position_timetagging.png'),
            dpi=150, bbox_inches='tight')
plt.show()
