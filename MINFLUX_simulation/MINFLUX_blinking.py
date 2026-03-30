import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

rng = np.random.default_rng(42)

# ── Field of view ─────────────────────────────────────────────────────────────
fov = 120e-9   # m

# ── 4x4 grid of candidate positions, 6 nm spacing ────────────────────────────
grid_spacing = 6e-9   # m
idx = np.arange(4) - 1.5
gx, gy = np.meshgrid(idx * grid_spacing, idx * grid_spacing)
grid_points = np.column_stack([gx.ravel(), gy.ravel()])

chosen    = rng.choice(len(grid_points), size=5, replace=False)
molecules = grid_points[chosen]

# ── 4-doughnut excitation beam centres ───────────────────────────────────────
L_beam = 50e-9
alphas = [0, 2*np.pi/3, 4*np.pi/3]
beam_centers = np.array(
    [(0.0, 0.0)] + [(L_beam/2 * np.cos(a), L_beam/2 * np.sin(a)) for a in alphas]
)

# ── Per-fluorophore colours ───────────────────────────────────────────────────
cmap   = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(len(molecules))]

# ── Simulation parameters ─────────────────────────────────────────────────────
T        = 60.0         # s
T_plot   = 2.0          # s  — time range shown in plots
dt       = 1e-3         # s  (1 ms bins)
t_ms     = np.arange(0, T, dt) * 1e3   # in ms for plotting
N_BINS   = len(t_ms)
RATE_ON  = 30000         # photons / s when ON
N_MOL    = len(molecules)
ACT_THR  = 20_000       # Hz  — activate when count rate drops below this
ACT_WIN  = 50           # bins (50 ms) over which count rate is assessed
FWHM     = 100e-9       # m, doughnut FWHM (matches 4doughnut_2d_CRB.py)
N_BEAMS  = len(beam_centers)

# Precompute per-molecule doughnut probabilities for each beam position
# p_beam[m, j] = prob. photon from molecule m is detected at beam j
_coeff = 4 * np.log(2) / FWHM**2
_beam_intensities = np.zeros((N_MOL, N_BEAMS))
for m in range(N_MOL):
    for j, bc in enumerate(beam_centers):
        r2 = np.sum((molecules[m] - bc)**2)
        _beam_intensities[m, j] = r2 * np.exp(-_coeff * r2)
# Normalise rows; guard against all-zero rows
_row_sums = _beam_intensities.sum(axis=1, keepdims=True)
_row_sums[_row_sums == 0] = 1.0
p_beam = _beam_intensities / _row_sums   # (N_MOL, N_BEAMS)

def simulate(k_on_base, k_off, seed=None):
    """
    Discrete-time simulation with activation-beam feedback.

    Activation beam is ON only when:
      (1) no fluorophore is currently emitting, AND
      (2) total count rate over the last 50 ms < 20 kHz.

    Returns
    -------
    photons   : (N_MOL, N_BINS)  photon counts per bin per molecule
    act_beam  : (N_BINS,)        1 = activation beam ON
    emission  : (N_BINS,)        1 = at least one fluorophore is ON
    """
    seed = 269880394
    sim_rng      = np.random.default_rng(seed)
    photons      = np.zeros((N_MOL,   N_BINS), dtype=float)
    beam_photons = np.zeros((N_BEAMS, N_BINS), dtype=float)
    act_beam     = np.zeros(N_BINS, dtype=float)
    emission     = np.zeros(N_BINS, dtype=float)
    states       = np.zeros(N_MOL,  dtype=bool)

    for b in range(N_BINS):
        # ── Activation-beam logic ─────────────────────────────────────────
        if b >= ACT_WIN:
            recent_rate = photons[:, b - ACT_WIN:b].sum() / (ACT_WIN * dt)
            low_rate    = recent_rate < ACT_THR
        else:
            low_rate = False

        activate    = low_rate
        act_beam[b] = float(activate)
        emission[b] = 0.0
        k_on_eff    = k_on_base

        # ── Emit photons then stochastically update states ────────────────
        for m in range(N_MOL):
            if states[m]:
                n_ph = sim_rng.poisson(RATE_ON * dt)
                photons[m, b] = n_ph
                if n_ph > 0:
                    # Split photons among beam positions by doughnut weights
                    counts = sim_rng.multinomial(n_ph, p_beam[m])
                    beam_photons[:, b] += counts
                if sim_rng.random() < k_off * dt:
                    states[m] = False
            else:
                if k_on_eff > 0 and sim_rng.random() < k_on_eff * dt:
                    states[m] = True

    return photons, beam_photons, act_beam, emission

# ── Figure layout ─────────────────────────────────────────────────────────────
# Right column: N_MOL photon traces + total photons
n_right       = N_MOL + 1
height_ratios = [1] * N_MOL + [1]

fig = plt.figure(figsize=(13, 12))
gs  = gridspec.GridSpec(
    n_right, 2,
    height_ratios=height_ratios,
    left=0.07, right=0.97, top=0.95, bottom=0.06,
    wspace=0.35, hspace=0.35,
)

ax_fov    = fig.add_subplot(gs[:, 0])
ax_traces = [fig.add_subplot(gs[i, 1]) for i in range(N_MOL)]
ax_total  = fig.add_subplot(gs[N_MOL, 1])

# Share x-axis across all right-hand panels
for ax in ax_traces[1:] + [ax_total]:
    ax.sharex(ax_traces[0])

# ── FOV plot ──────────────────────────────────────────────────────────────────
for i, (mx, my) in enumerate(molecules * 1e9):
    ax_fov.text(mx, my, str(i + 1), color=colors[i],
                fontsize=13, fontweight='bold', ha='center', va='center', zorder=5)

ax_fov.scatter(beam_centers[:, 0] * 1e9, beam_centers[:, 1] * 1e9,
               s=300, facecolors='none', edgecolors='red', linewidths=1.5, zorder=4)
for i, (bx, by) in enumerate(beam_centers * 1e9):
    ax_fov.text(bx, by, f'r{i}', color='red',
                fontsize=11, fontweight='bold', ha='center', va='center', zorder=5)
half = fov / 2 * 1e9
ax_fov.set_xlim(-half, half)
ax_fov.set_ylim(-half, half)
ax_fov.set_aspect('equal')
ax_fov.set_xlabel('x (nm)')
ax_fov.set_ylabel('y (nm)')
ax_fov.set_title('Real-space field of view')

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='red', markeredgewidth=1.5, markersize=10,
           label='Beam position'),
] + [
    Line2D([0], [0], marker='$' + str(i+1) + '$', color=colors[i],
           markersize=10, linestyle='None', label=f'Fluorophore {i+1}')
    for i in range(N_MOL)
]
ax_fov.legend(handles=legend_handles, fontsize=8, loc='upper right')

# ── Initial simulation ────────────────────────────────────────────────────────
k_on  = 2.0
k_off = 25.0
seed = int(np.random.default_rng().integers(0, 2**32))
print(f"Random seed: {seed}")
ph0, bp0, _, _ = simulate(k_on, k_off, seed=seed)

# Per-fluorophore photon traces
lines = []
for i, ax in enumerate(ax_traces):
    ln, = ax.plot(t_ms, ph0[i], lw=0.8, color=colors[i])
    lines.append(ln)
    ax.set_xlim(0, T_plot * 1e3)
    ax.set_ylabel(str(i + 1), color=colors[i], fontsize=9, rotation=0, labelpad=10)
    ax.tick_params(axis='y', labelsize=7)
    ax.set_xticklabels([])

ax_traces[0].set_title('Photon counts')

# Total photon trace
ln_total, = ax_total.plot(t_ms, ph0.sum(axis=0), lw=0.8, color='black')
ax_total.set_ylabel('Total photons', fontsize=8, rotation=0, labelpad=28)
ax_total.tick_params(axis='y', labelsize=7)
ax_total.set_xticks([0, T_plot * 1e3])
ax_total.set_xticklabels(['0 s', f'{T_plot:.0f} s'])
ax_total.set_xlabel('')


import os
script_dir = os.path.dirname(os.path.abspath(__file__))
figs_dir   = os.path.join(script_dir, 'figs')
data_dir   = os.path.join(script_dir, 'data')
os.makedirs(figs_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Save figure
plt.savefig(os.path.join(figs_dir, 'MINFLUX_simulation.png'), dpi=150, bbox_inches='tight')

# Save time axis and total photon trace
total_photons = ph0.sum(axis=0)
np.save(os.path.join(data_dir, 't_ms.npy'),          t_ms)
np.save(os.path.join(data_dir, 'total_photons.npy'), total_photons)

# Save per-beam photon counts and fractions (shape: N_BEAMS × N_BINS)
np.save(os.path.join(data_dir, 'beam_photons.npy'), bp0)
total_safe = np.where(total_photons > 0, total_photons, 1.0)
beam_fractions = bp0 / total_safe[np.newaxis, :]   # (N_BEAMS, N_BINS)
np.save(os.path.join(data_dir, 'beam_fractions.npy'), beam_fractions)
np.save(os.path.join(data_dir, 'per_molecule_photons.npy'), ph0)   # (N_MOL, N_BINS)
np.save(os.path.join(data_dir, 'molecules.npy'), molecules)        # (N_MOL, 2) in metres

plt.show()
