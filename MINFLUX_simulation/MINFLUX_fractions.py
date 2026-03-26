import numpy as np
import matplotlib.pyplot as plt
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
total_photons = beam_photons.sum(axis=0)

dt_s = (t_ms[1] - t_ms[0]) * 1e-3   # bin width in seconds
N_BEAMS = beam_photons.shape[0]
N_MOL   = per_mol.shape[0]

# ── Colours (match other scripts) ─────────────────────────────────────────────
cmap_beam  = plt.get_cmap('Set1')
beam_colors = [cmap_beam(i) for i in range(N_BEAMS)]
cmap_mol   = plt.get_cmap('tab10')
mol_colors  = [cmap_mol(i) for i in range(N_MOL)]

# ── Select which valid burst to analyse (1-indexed) ───────────────────────────
BURST_INDEX = 7   # ← change this to pick a different burst

# ── Collect all clean single-fluorophore bursts ────────────────────────────────
burst_mask = total_photons > 0
labeled, n_bursts = label(burst_mask)

valid_bursts = []
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

t_burst      = t_ms[selected_bins]
beam_burst   = beam_photons[:, selected_bins]          # (N_BEAMS, burst_len)
rate_burst   = beam_burst / dt_s / 1e3                 # kHz

print(f"Fluorophore {selected_mol_id + 1}, "
      f"t = {t_burst[0]:.1f}–{t_burst[-1]:.1f} ms, "
      f"{len(t_burst)} bins, "
      f"{int(beam_burst.sum())} total photons")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4),
                                         gridspec_kw={'width_ratios': [3, 1]})
fig.patch.set_facecolor('white')
for ax in (ax_left, ax_right):
    ax.set_facecolor('white')

# Left: count rate per beam position
for j in range(N_BEAMS):
    ax_left.plot(t_burst, rate_burst[j], color=beam_colors[j],
                 lw=1.5, label=f'$r_{j}$', zorder=2)

ax_left.set_xlabel('Time (ms)')
ax_left.set_ylabel('Count rate (kHz)')
ax_left.set_title(f'Fluorophore {selected_mol_id + 1}')
ax_left.set_xlim(t_burst[0], t_burst[-1])
ax_left.legend(fontsize=9, loc='upper right')

# Right: total photons per beam position
beam_totals = beam_burst.sum(axis=1)   # (N_BEAMS,)
bar_positions = np.arange(N_BEAMS)
ax_right.bar(bar_positions, beam_totals,
             color=beam_colors, edgecolor='none', zorder=2)
ax_right.set_xticks(bar_positions)
ax_right.set_xticklabels([f'$r_{j}$' for j in range(N_BEAMS)], fontsize=11)
ax_right.set_ylabel('Total photons')
ax_right.set_title('Photons per position')

plt.tight_layout()
plt.savefig(os.path.join(figs_dir, 'MINFLUX_fractions.png'),
            dpi=150, bbox_inches='tight')
plt.show()
