"""
synthetic_visualizer.py

Generates an animated 3D GIF of a synthetic LZ-like S2 event:
  Phase 1 – Red interaction point in the liquid xenon bulk
  Phase 2 – Point converts to blue ionisation electrons
  Phase 3 – Electrons undergo random-walk diffusion while drifting upward
  Phase 4 – Photons propagate upward to the PMT array (solid-angle weighted)
  Phase 5 – PMTs illuminate proportional to photon count

Usage:
    python synthetic_visualizer.py --channel_positions data/pmt_xy_42.h5
    python synthetic_visualizer.py --channel_positions data/pmt_xy_42.h5 \\
        --output event_plots/my_event.gif --seed 7 --n_electrons 15 --fps 24
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.animation import FuncAnimation, PillowWriter

from generate_synthetic import load_pmt_positions, H_EL_TO_PMT_CM, ELECTRON_SPREAD_CM


# ──────────────────────────────────────────────────────────────────────────────
# Colours (white background palette)
# ──────────────────────────────────────────────────────────────────────────────
BG          = 'white'
WALL_COL    = '#aabbcc'
SURFACE_COL = '#3377bb'
PMT_IDLE    = '#99aabb'
PMT_LIT     = '#ff8800'
ELEC_COL    = '#1166ee'
PHOT_COL    = '#00aa44'
INTR_COL    = '#cc2200'

# ──────────────────────────────────────────────────────────────────────────────
# Animation phases  (exclusive end)
# ──────────────────────────────────────────────────────────────────────────────
F_IDLE    = (0,   20)    # steady red interaction point
F_SPAWN   = (20,  35)    # red → blue electrons
F_DRIFT   = (35,  95)    # random-walk drift  (60 steps, one per frame)
F_PHOTON  = (95,  140)   # photons fly to PMTs
F_GLOW    = (140, 160)   # PMTs glow
N_FRAMES  = 160
N_DRIFT_STEPS = F_DRIFT[1] - F_DRIFT[0]


# ──────────────────────────────────────────────────────────────────────────────
# Pre-computation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sample_vertex(pmt_positions: np.ndarray):
    R = float(np.max(np.linalg.norm(pmt_positions, axis=1)))
    r = R * 0.5 * np.sqrt(np.random.uniform(0, 1))
    a = np.random.uniform(0, 2 * np.pi)
    return r * np.cos(a), r * np.sin(a)


def _electron_trajectories(
    x_v: float, y_v: float, z_v: float,
    n_el: int, n_steps: int, spread_cm: float,
) -> np.ndarray:
    """Return (n_el, n_steps+1, 3) random-walk trajectories."""
    pos = np.zeros((n_el, n_steps + 1, 3))
    pos[:, 0, 0] = np.random.normal(x_v, 0.3, n_el)
    pos[:, 0, 1] = np.random.normal(y_v, 0.3, n_el)
    pos[:, 0, 2] = z_v

    dz_mean = -z_v / n_steps
    sig_t   = spread_cm * 1.8 / np.sqrt(n_steps)   # exaggerated for visibility
    sig_z   = 0.35 * abs(dz_mean)

    for s in range(n_steps):
        pos[:, s+1, 0] = pos[:, s, 0] + np.random.normal(0, sig_t, n_el)
        pos[:, s+1, 1] = pos[:, s, 1] + np.random.normal(0, sig_t, n_el)
        pos[:, s+1, 2] = np.minimum(
            pos[:, s, 2] + np.random.normal(dz_mean, sig_z, n_el), 0.0
        )
    return pos


def _photon_paths(
    electron_end_xy: np.ndarray,
    pmt_positions: np.ndarray,
    n_ph_each: int,
    z_pmt: float,
):
    """Emit n_ph_each photons per electron, routed by solid angle.

    Returns per-electron lists of (starts, ends) arrays so the caller can
    stagger emission timing independently for each electron.
    """
    h = z_pmt
    per_el_starts, per_el_ends = [], []
    counts = np.zeros(len(pmt_positions), dtype=int)

    for ex, ey in electron_end_xy:
        dx = pmt_positions[:, 0] - ex
        dy = pmt_positions[:, 1] - ey
        d2 = dx**2 + dy**2
        w  = h / (h**2 + d2) ** 1.5
        w /= w.sum()
        targets = np.random.choice(len(pmt_positions), size=n_ph_each, p=w)
        per_el_starts.append(
            np.column_stack([np.full(n_ph_each, ex), np.full(n_ph_each, ey), np.zeros(n_ph_each)])
        )
        per_el_ends.append(
            np.column_stack([pmt_positions[targets, 0], pmt_positions[targets, 1],
                             np.full(n_ph_each, z_pmt)])
        )
        counts[targets] += 1

    return per_el_starts, per_el_ends, counts


def _pmt_radius(pmt_positions: np.ndarray) -> float:
    """Return PMT circle radius in data-units (cm) = half nearest-neighbour distance."""
    from scipy.spatial.distance import cdist
    D = cdist(pmt_positions, pmt_positions)
    np.fill_diagonal(D, np.inf)
    return float(D.min()) / 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Scene helpers
# ──────────────────────────────────────────────────────────────────────────────

def _draw_tpc(ax, R: float, z_bot: float):
    theta = np.linspace(0, 2 * np.pi, 120)

    # Vertical lines
    for a in np.linspace(0, 2 * np.pi, 18, endpoint=False):
        ax.plot([R * np.cos(a)] * 2, [R * np.sin(a)] * 2, [z_bot, 0],
                color=WALL_COL, alpha=0.25, linewidth=0.5, zorder=1)

    # Cathode ring (bottom)
    ax.plot(R * np.cos(theta), R * np.sin(theta), z_bot,
            color=WALL_COL, alpha=0.8, linewidth=1.2, zorder=1)
    # Liquid surface ring
    ax.plot(R * np.cos(theta), R * np.sin(theta), 0,
            color=SURFACE_COL, alpha=0.8, linewidth=1.8, zorder=2)
    for a in np.linspace(0, 2 * np.pi, 10, endpoint=False):
        ax.plot([0, R * np.cos(a)], [0, R * np.sin(a)], [0, 0],
                color=SURFACE_COL, alpha=0.12, linewidth=0.5, zorder=1)


def _draw_pmts(ax, pmt_positions: np.ndarray, z_pmt: float,
               pmt_r: float, lit_frac: np.ndarray | None = None):
    if lit_frac is None:
        lit_frac = np.zeros(len(pmt_positions))
    theta = np.linspace(0, 2 * np.pi, 28)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    for i, (px, py) in enumerate(pmt_positions):
        f     = float(lit_frac[i])
        color = PMT_LIT if f > 0 else PMT_IDLE
        lw    = 0.8 + f * 2.0
        ax.plot(
            px + pmt_r * cos_t,
            py + pmt_r * sin_t,
            np.full_like(theta, z_pmt),
            color=color, linewidth=lw, alpha=min(1.0, 0.7 + f * 0.3),
            zorder=10,
        )


def _configure_ax(ax, R: float, z_bot: float, z_pmt: float):
    ax.cla()
    ax.set_facecolor(BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('none')
    ax.grid(False)
    ax.set_axis_off()
    m = 0.12
    ax.set_xlim(-R * (1 + m), R * (1 + m))
    ax.set_ylim(-R * (1 + m), R * (1 + m))
    ax.set_zlim(z_bot * 1.2, z_pmt * 1.8)
    try:
        ax.set_box_aspect([1, 1, 2.2])
    except Exception:
        pass
    ax.view_init(elev=22, azim=-50)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def generate_event_gif(
    pmt_positions: np.ndarray,
    output_path: str = "event_plots/event_animation.gif",
    n_electrons: int = 14,
    n_photons_per_electron: int = 6,
    z_depth_cm: float = 25.0,
    fps: int = 20,
    seed: int | None = None,
    dpi: int = 90,
) -> None:
    if seed is not None:
        np.random.seed(seed)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    R     = float(np.max(np.linalg.norm(pmt_positions, axis=1)))
    z_v   = -abs(z_depth_cm)
    z_bot = z_v * 1.15
    z_pmt = H_EL_TO_PMT_CM

    FIG_W = 5.5
    fig = plt.figure(figsize=(FIG_W, FIG_W), facecolor=BG)
    ax  = fig.add_subplot(111, projection='3d', computed_zorder=False)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    pmt_r = _pmt_radius(pmt_positions)

    # ── Pre-compute ───────────────────────────────────────────────────────────
    x_v, y_v     = _sample_vertex(pmt_positions)
    trajectories = _electron_trajectories(x_v, y_v, z_v, n_electrons,
                                           N_DRIFT_STEPS, ELECTRON_SPREAD_CM)
    e_final_xy   = trajectories[:, -1, :2]

    # When does each electron first reach the liquid surface (z >= -0.5 cm)?
    at_surface   = trajectories[:, :, 2] >= -0.5          # (n_el, n_steps+1)
    arrive_steps = np.argmax(at_surface, axis=1)           # first True index
    never        = ~at_surface.any(axis=1)
    arrive_steps[never] = N_DRIFT_STEPS

    # Photons emit at the actual frame the electron hits the surface during drift.
    # This lets early-arriving electrons send photons while later electrons still drift.
    PHOTON_TRAVEL = 32                                     # frames per photon flight
    emit_frames   = F_DRIFT[0] + arrive_steps              # absolute frame of surface arrival

    per_el_starts, per_el_ends, pmt_counts = _photon_paths(
        e_final_xy, pmt_positions, n_photons_per_electron, z_pmt
    )
    max_counts = max(1, pmt_counts.max())

    # ── Frame update ──────────────────────────────────────────────────────────
    def update(frame: int):
        _configure_ax(ax, R, z_bot, z_pmt)
        _draw_tpc(ax, R, z_bot)

        # Phase 1: steady red interaction point
        if frame < F_SPAWN[0]:
            _draw_pmts(ax, pmt_positions, z_pmt, pmt_r)
            ax.scatter([x_v], [y_v], [z_v],
                       color=INTR_COL, s=55, alpha=1.0,
                       depthshade=False, zorder=5)

        # Phase 2: red → blue electrons appear
        elif frame < F_DRIFT[0]:
            _draw_pmts(ax, pmt_positions, z_pmt, pmt_r)
            t      = (frame - F_SPAWN[0]) / (F_SPAWN[1] - F_SPAWN[0])
            red_a  = max(0.0, 1.0 - t * 2.5)
            blue_a = min(1.0, t * 2.5)
            if red_a > 0:
                ax.scatter([x_v], [y_v], [z_v],
                           color=INTR_COL, s=55, alpha=red_a,
                           depthshade=False, zorder=5)
            if blue_a > 0:
                ax.scatter(trajectories[:, 0, 0],
                           trajectories[:, 0, 1],
                           trajectories[:, 0, 2],
                           c=ELEC_COL, s=30, alpha=blue_a,
                           depthshade=False, zorder=5)

        # Phases 3 + 4: drift and photon emission overlap.
        # Electrons that have reached the surface are already sending photons
        # while later electrons are still drifting upward.
        elif frame < F_GLOW[0]:
            _draw_pmts(ax, pmt_positions, z_pmt, pmt_r)

            # Still-drifting electrons
            if frame < F_PHOTON[0]:
                step      = frame - F_DRIFT[0]
                still_drifting = arrive_steps > step
                if still_drifting.any():
                    idx = np.where(still_drifting)[0]
                    ax.scatter(trajectories[idx, step, 0],
                               trajectories[idx, step, 1],
                               trajectories[idx, step, 2],
                               color=ELEC_COL, s=30, alpha=0.9,
                               depthshade=False, zorder=6)

            # In-flight photons for every electron that has already arrived
            all_pos = []
            for e_idx in range(n_electrons):
                ef = emit_frames[e_idx]
                if frame < ef:
                    continue
                t = min((frame - ef) / PHOTON_TRAVEL, 1.0)
                if t >= 1.0:
                    continue  # already at PMT, hide until glow phase
                pos = per_el_starts[e_idx] + t * (per_el_ends[e_idx] - per_el_starts[e_idx])
                all_pos.append(pos)
            if all_pos:
                ph_pos = np.concatenate(all_pos, axis=0)
                ax.scatter(ph_pos[:, 0], ph_pos[:, 1], ph_pos[:, 2],
                           color=PHOT_COL, s=9, alpha=0.8,
                           depthshade=False, zorder=6)

        # Phase 5: PMTs glow
        else:
            t    = (frame - F_GLOW[0]) / (F_GLOW[1] - F_GLOW[0])
            glow = np.clip(pmt_counts / max_counts * min(1.0, t * 2.5), 0, 1)
            _draw_pmts(ax, pmt_positions, z_pmt, pmt_r, lit_frac=glow)

    # ── Render ────────────────────────────────────────────────────────────────
    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 // fps)
    writer = PillowWriter(fps=fps)
    print(f"Rendering {N_FRAMES} frames → {output_path}")
    anim.save(output_path, writer=writer, dpi=dpi,
              savefig_kwargs={'facecolor': BG})
    plt.close(fig)
    print(f"Saved {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a 3D animated GIF of a synthetic LZ S2 event"
    )
    parser.add_argument("--channel_positions", type=str, default="data/pmt_xy.h5")
    parser.add_argument("--output",            type=str, default="event_plots/event_animation.gif")
    parser.add_argument("--n_electrons",       type=int,   default=14)
    parser.add_argument("--n_photons",         type=int,   default=6)
    parser.add_argument("--z_depth",           type=float, default=25.0)
    parser.add_argument("--fps",               type=int,   default=20)
    parser.add_argument("--seed",              type=int,   default=None)
    parser.add_argument("--dpi",               type=int,   default=90)
    args = parser.parse_args()

    pmt_pos = load_pmt_positions(args.channel_positions)
    generate_event_gif(
        pmt_positions=pmt_pos,
        output_path=args.output,
        n_electrons=args.n_electrons,
        n_photons_per_electron=args.n_photons,
        z_depth_cm=args.z_depth,
        fps=args.fps,
        seed=args.seed,
        dpi=args.dpi,
    )
