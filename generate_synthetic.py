import argparse
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

SAMPLERATE        = 10      # ns per time bin
G2                = 47.35   # mean EL photons per drifted electron
DIFFWIDTH_NS      = 300.0   # longitudinal diffusion sigma (ns); keeps electron cloud
                             #   dense enough for smooth waveforms
EGASWIDTH_NS      = 1004.0  # per-electron EL emission time sigma (ns); calibrated so
                             #   sqrt(DIFFWIDTH² + EGASWIDTH²) ≈ 1048 ns matches real S2 width
PHDWIDTH_NS       = 20.0    # single-photon pulse width sigma (ns); models PMT response + digitizer anti-aliasing
ELECTRON_SPREAD_CM = 3.0    # transverse electron diffusion sigma from vertex (cm)
H_EL_TO_PMT_CM   = 6.0    # effective photon transport height (cm); physical gap ~6 cm

DIFFWIDTH  = DIFFWIDTH_NS  / SAMPLERATE
EGASWIDTH  = EGASWIDTH_NS  / SAMPLERATE
PHDWIDTH   = PHDWIDTH_NS   / SAMPLERATE   # = 1.0 bin

N_TIME_BINS = 1000
MU_TIME     = N_TIME_BINS / 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pmt_positions(channel_positions_path: str) -> np.ndarray:
    """Return PMT (x, y) positions in cm, shape (N_PMT, 2)."""
    with h5py.File(channel_positions_path, 'r') as f:
        if 'TA_PMTs_xy' in f:
            positions = f['TA_PMTs_xy'][:] / 10.0   # mm → cm
        elif 'positions' in f:
            positions = f['positions'][:]
        elif 'xy' in f:
            positions = f['xy'][:]
        else:
            raise ValueError(
                f"No recognised positions dataset in {channel_positions_path}. "
                f"Keys: {list(f.keys())}"
            )
    return positions.astype(np.float32)


def solid_angle_weights(
    electron_x: np.ndarray,
    electron_y: np.ndarray,
    pmt_positions: np.ndarray,
    h: float,
) -> np.ndarray:
    """
    Compute solid-angle-based routing weights for each (electron, PMT) pair.

    For a photon emitted isotropically from height h above the PMT plane at
    position (ex, ey), the fraction hitting PMT i at (xi, yi) scales as:
        w_i ∝ h / (h² + d_i²)^(3/2)
    where d_i = ||(ex, ey) - (xi, yi)||.

    Parameters
    ----------
    electron_x, electron_y : (E,) positions of electrons in cm
    pmt_positions          : (P, 2) PMT positions in cm
    h                      : height from EL region to PMT plane (cm)

    Returns
    -------
    weights : (E, P) normalised probability matrix
    """
    # d² : (E, P)
    dx = electron_x[:, None] - pmt_positions[None, :, 0]   # (E, P)
    dy = electron_y[:, None] - pmt_positions[None, :, 1]
    d2 = dx**2 + dy**2
    w  = h / (h**2 + d2) ** 1.5                             # ∝ solid angle
    w /= w.sum(axis=1, keepdims=True)                        # normalise rows
    return w


def generate_event(
    pmt_positions: np.ndarray,
    n_time_bins: int = N_TIME_BINS,
    mu_time: float = MU_TIME,
    n_electrons: int | None = None,
) -> tuple[np.ndarray, float, float, float]:
    n_pmt = len(pmt_positions)

    # uniform within detector footprint
    detector_radius = float(np.max(np.linalg.norm(pmt_positions, axis=1)))
    r     = detector_radius * np.sqrt(np.random.uniform(0.0, 1.0))
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    x_vertex = r * np.cos(theta)
    y_vertex = r * np.sin(theta)

    if n_electrons is None:
        n_electrons = int(np.random.randint(20, 120))

    electron_times = np.random.normal(mu_time, DIFFWIDTH, size=n_electrons)
    electron_x = np.random.normal(x_vertex, ELECTRON_SPREAD_CM, size=n_electrons)
    electron_y = np.random.normal(y_vertex, ELECTRON_SPREAD_CM, size=n_electrons)

    # EL photon counts per electron
    n_photons_per_e = np.random.poisson(G2, size=n_electrons).astype(np.int64)

    if n_photons_per_e.sum() == 0:
        return np.zeros((n_pmt, n_time_bins), dtype=np.float32), x_vertex, y_vertex, float(mu_time)

    # Solid-angle weights: (n_electrons, n_pmt)
    weights = solid_angle_weights(electron_x, electron_y, pmt_positions, H_EL_TO_PMT_CM)

    # Accumulate photon counts into (N_PMT, n_time_bins)
    # For each electron: multinomial draw distributes its photons across PMTs,
    # then its arrival time is smeared and binned.
    photon_counts = np.zeros((n_pmt, n_time_bins), dtype=np.float32)
    for e in range(n_electrons):
        n_ph = int(n_photons_per_e[e])
        if n_ph == 0:
            continue
        # Distribute photons across PMTs by solid angle (multinomial)
        pmt_hits = np.random.multinomial(n_ph, weights[e])   # (n_pmt,)

        # Each photon gets an independent EL timing offset
        # We draw n_ph times then assign per-PMT sums via the multinomial counts
        photon_times = np.random.normal(electron_times[e], EGASWIDTH, size=n_ph)
        # Assign photons to PMTs in multinomial order
        pmt_assign = np.repeat(np.arange(n_pmt), pmt_hits)   # (n_ph,)
        t_bins = np.rint(photon_times).astype(np.int64)
        valid = (t_bins >= 0) & (t_bins < n_time_bins)
        np.add.at(photon_counts, (pmt_assign[valid], t_bins[valid]), 1.0)

    # Apply Gaussian pulse shape via 1-D convolution along time axis
    waveform = gaussian_filter1d(photon_counts, sigma=PHDWIDTH, axis=1)
    return waveform.astype(np.float32), x_vertex, y_vertex, MU_TIME


def generate_dataset(
    n_samples: int,
    channel_positions_path: str,
    output_path: str,
    n_time_bins: int = N_TIME_BINS,
    mu_time: float = MU_TIME,
    seed: int | None = None,
    write_chunk: int = 256,
) -> None:
    if seed is not None:
        np.random.seed(seed)

    pmt_positions = load_pmt_positions(channel_positions_path)
    n_pmt         = len(pmt_positions)
    det_radius    = float(np.max(np.linalg.norm(pmt_positions, axis=1)))

    print(f"PMT array : {n_pmt} channels, detector radius {det_radius:.1f} cm")
    print(f"Waveform  : {n_time_bins} bins × {SAMPLERATE} ns = {n_time_bins * SAMPLERATE} ns")
    print(f"Generating: {n_samples:,} events → {output_path}")

    wf_buf  = np.zeros((write_chunk, n_pmt, n_time_bins), dtype=np.float32)
    xc_buf  = np.zeros(write_chunk, dtype=np.float32)
    yc_buf  = np.zeros(write_chunk, dtype=np.float32)
    dt_buf  = np.zeros(write_chunk, dtype=np.float32)

    with h5py.File(output_path, 'w') as f:
        dset_wf = f.create_dataset(
            'waveforms',
            shape=(n_samples, n_pmt, n_time_bins),
            dtype=np.float32,
            chunks=(min(write_chunk, n_samples), n_pmt, n_time_bins),
        )
        dset_xc = f.create_dataset('xc', shape=(n_samples,), dtype=np.float32)
        dset_yc = f.create_dataset('yc', shape=(n_samples,), dtype=np.float32)
        dset_dt = f.create_dataset('dt', shape=(n_samples,), dtype=np.float32)

        written = 0
        buf_i   = 0
        for i in tqdm(range(n_samples), desc="Generating", ncols=100):
            wf, xc, yc, dt = generate_event(
                pmt_positions,
                n_time_bins=n_time_bins,
                mu_time=mu_time,
            )
            wf_buf[buf_i] = wf
            xc_buf[buf_i] = xc
            yc_buf[buf_i] = yc
            dt_buf[buf_i] = dt
            buf_i += 1

            if buf_i == write_chunk or i == n_samples - 1:
                dset_wf[written:written + buf_i] = wf_buf[:buf_i]
                dset_xc[written:written + buf_i] = xc_buf[:buf_i]
                dset_yc[written:written + buf_i] = yc_buf[:buf_i]
                dset_dt[written:written + buf_i] = dt_buf[:buf_i]
                written += buf_i
                buf_i = 0

        # Metadata
        f.attrs['n_samples']          = n_samples
        f.attrs['n_channels']         = n_pmt
        f.attrs['n_time_bins']        = n_time_bins
        f.attrs['samplerate_ns']      = SAMPLERATE
        f.attrs['mu_time_bins']       = mu_time
        f.attrs['G2']                 = G2
        f.attrs['DIFFWIDTH_NS']        = DIFFWIDTH_NS
        f.attrs['EGASWIDTH_NS']       = EGASWIDTH_NS
        f.attrs['PHDWIDTH_NS']        = PHDWIDTH_NS
        f.attrs['ELECTRON_SPREAD_CM'] = ELECTRON_SPREAD_CM
        f.attrs['H_EL_TO_PMT_CM']    = H_EL_TO_PMT_CM

    print(f"Done. {n_samples:,} events saved to {output_path}")


def visualize_events(
    channel_positions_path: str,
    n: int = 4,
    n_time_bins: int = N_TIME_BINS,
    mu_time: float = MU_TIME,
    seed: int | None = None,
    output_dir: str = "event_plots",
) -> None:
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if seed is not None:
        np.random.seed(seed)

    pmt_positions = load_pmt_positions(channel_positions_path)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.5 * n), squeeze=False)

    for i in range(n):
        wf, xc, yc, dt = generate_event(pmt_positions, n_time_bins=n_time_bins, mu_time=mu_time)

        ax_xy, ax_t = axes[i]

        # Left: PMT hit map (total charge per PMT)
        charge = wf.sum(axis=1)
        sc = ax_xy.scatter(
            pmt_positions[:, 0], pmt_positions[:, 1],
            c=charge, cmap="viridis", s=80, edgecolors="k", linewidths=0.3,
        )
        ax_xy.scatter([xc], [yc], marker="x", color="red", s=80,
                      linewidths=1.5, label=f"({xc:.1f}, {yc:.1f}) cm")
        ax_xy.set_title(f"Event {i+1}  PMT hit map", fontsize=9)
        ax_xy.set_xlabel("x (cm)")
        ax_xy.set_ylabel("y (cm)")
        ax_xy.set_aspect("equal")
        ax_xy.legend(fontsize=7)
        plt.colorbar(sc, ax=ax_xy, label="Charge (AU)")

        # Right: summed waveform across all PMTs
        t_axis = np.arange(n_time_bins)
        ax_t.plot(t_axis, wf.sum(axis=0), linewidth=0.8, color="steelblue")
        ax_t.axvline(dt, color="red", linestyle="--", linewidth=0.8,
                     label=f"dt={dt:.0f} bins")
        ax_t.set_title(f"Event {i+1}  summed waveform", fontsize=9)
        ax_t.set_xlabel("Time bin")
        ax_t.set_ylabel("Amplitude (AU)")
        ax_t.legend(fontsize=7)

    seed_str = f"seed{seed}" if seed is not None else "unseeded"
    fig.suptitle(f"Synthetic SS events  ({seed_str})", fontweight="bold")
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"view_synthetic_events_{seed_str}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic LZ-like S2 events")
    parser.add_argument("--channel_positions",  type=str,   default="data/pmt_xy.h5")
    parser.add_argument("--n_time_bins",        type=int,   default=N_TIME_BINS)
    parser.add_argument("--mu_time",            type=float, default=MU_TIME)
    parser.add_argument("--seed",               type=int,   default=None)

    sub = parser.add_subparsers(dest="cmd")

    p_vis = sub.add_parser("visualize", help="Generate and plot a few events without saving")
    p_vis.add_argument("--n", type=int, default=4, help="Number of events to visualize")
    p_vis.add_argument("--output_dir", type=str, default="event_plots")

    p_gen = sub.add_parser("generate", help="Generate and save a dataset")
    p_gen.add_argument("--n_samples",   type=int, default=10_000)
    p_gen.add_argument("--output",      type=str, default="data/synthetic_events.h5")
    p_gen.add_argument("--write_chunk", type=int, default=256)

    args = parser.parse_args()

    if args.cmd == "visualize":
        visualize_events(
            channel_positions_path=args.channel_positions,
            n=args.n,
            n_time_bins=args.n_time_bins,
            mu_time=args.mu_time,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.cmd == "generate":
        generate_dataset(
            n_samples=args.n_samples,
            channel_positions_path=args.channel_positions,
            output_path=args.output,
            n_time_bins=args.n_time_bins,
            mu_time=args.mu_time,
            seed=args.seed,
            write_chunk=args.write_chunk,
        )
    else:
        parser.print_help()
