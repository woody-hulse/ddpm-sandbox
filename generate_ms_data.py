"""
Generate Multi-Scatter (MS) data from Single-Scatter (SS) events.

Each time bin represents 10ns. This script:
1. Loads SS events from an h5 file
2. Pairs events and translates one along the time axis
3. Sums the waveforms to create MS events
4. Records delta_mu (time difference in ns) and other metadata
"""
import argparse
import os
from typing import Tuple, Optional

import h5py
import numpy as np
from tqdm import tqdm


NS_PER_BIN = 10.0  # Each z (time) bin represents 10ns


def load_ss_data(h5_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load single-scatter data from h5 file."""
    with h5py.File(h5_path, 'r') as f:
        waveforms = f['waveforms'][:]  # (N, C, T)
        xc = f['xc'][:].astype(np.float32)
        yc = f['yc'][:].astype(np.float32)
        dt = f['dt'][:].astype(np.float32)
    return waveforms, xc, yc, dt


def shift_waveform(waveform: np.ndarray, shift_bins: int) -> np.ndarray:
    """
    Shift waveform along time axis.
    
    Args:
        waveform: Shape (C, T) - channels x time
        shift_bins: Number of bins to shift (positive = later in time)
    
    Returns:
        Shifted waveform with same shape, zero-padded
    """
    C, T = waveform.shape
    shifted = np.zeros_like(waveform)
    
    if shift_bins == 0:
        return waveform.copy()
    elif shift_bins > 0:
        if shift_bins < T:
            shifted[:, shift_bins:] = waveform[:, :T - shift_bins]
    else:
        shift_bins = -shift_bins
        if shift_bins < T:
            shifted[:, :T - shift_bins] = waveform[:, shift_bins:]
    
    return shifted


def generate_ms_event(
    wf1: np.ndarray,
    wf2: np.ndarray,
    dt1: float,
    dt2: float,
    delta_bins: int,
) -> Tuple[np.ndarray, float]:
    """
    Generate a multi-scatter event from two single-scatter events.
    
    The first event is kept at its original position, and the second
    event is shifted by delta_bins relative to it.
    
    Args:
        wf1: First waveform (C, T)
        wf2: Second waveform (C, T)
        dt1: Time of first event
        dt2: Time of second event
        delta_bins: Time shift in bins for second event
    
    Returns:
        ms_waveform: Combined waveform (C, T)
        delta_mu_ns: Time difference in nanoseconds
    """
    wf2_shifted = shift_waveform(wf2, delta_bins)
    ms_waveform = wf1 + wf2_shifted
    delta_mu_ns = delta_bins * NS_PER_BIN
    return ms_waveform, delta_mu_ns


def generate_ms_dataset(
    ss_waveforms: np.ndarray,
    ss_xc: np.ndarray,
    ss_yc: np.ndarray,
    ss_dt: np.ndarray,
    n_ms_events: int,
    delta_range: Tuple[int, int] = (-20, 20),
    seed: Optional[int] = None,
    single_node: bool = False,
) -> dict:
    """
    Generate a dataset of multi-scatter events.
    
    Args:
        ss_waveforms: SS waveforms (N, C, T)
        ss_xc, ss_yc, ss_dt: SS event positions
        n_ms_events: Number of MS events to generate
        delta_range: Range of time shifts in bins (min, max)
        seed: Random seed
        single_node: If True, sum over channels so output shape is (n_events, 1, T)
    
    Returns:
        Dictionary with MS data arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_ss = len(ss_waveforms)
    n_channels, n_time = ss_waveforms.shape[1], ss_waveforms.shape[2]
    
    out_channels = 1 if single_node else n_channels
    ms_waveforms = np.zeros((n_ms_events, out_channels, n_time), dtype=ss_waveforms.dtype)
    ms_delta_mu = np.zeros(n_ms_events, dtype=np.float32)
    ms_delta_bins = np.zeros(n_ms_events, dtype=np.int32)
    
    ms_idx1 = np.zeros(n_ms_events, dtype=np.int64)
    ms_idx2 = np.zeros(n_ms_events, dtype=np.int64)
    ms_xc1 = np.zeros(n_ms_events, dtype=np.float32)
    ms_yc1 = np.zeros(n_ms_events, dtype=np.float32)
    ms_dt1 = np.zeros(n_ms_events, dtype=np.float32)
    ms_xc2 = np.zeros(n_ms_events, dtype=np.float32)
    ms_yc2 = np.zeros(n_ms_events, dtype=np.float32)
    ms_dt2 = np.zeros(n_ms_events, dtype=np.float32)
    
    for i in tqdm(range(n_ms_events), desc="Generating MS events"):
        idx1 = np.random.randint(0, n_ss)
        idx2 = np.random.randint(0, n_ss)
        while idx2 == idx1:
            idx2 = np.random.randint(0, n_ss)
        
        delta_bins = np.random.randint(delta_range[0], delta_range[1] + 1)
        
        wf1 = ss_waveforms[idx1]
        wf2 = ss_waveforms[idx2]
        
        ms_wf, delta_mu_ns = generate_ms_event(
            wf1, wf2, ss_dt[idx1], ss_dt[idx2], delta_bins
        )
        
        if single_node:
            ms_wf = ms_wf.sum(axis=0, keepdims=True).astype(ms_waveforms.dtype)
        
        ms_waveforms[i] = ms_wf
        ms_delta_mu[i] = delta_mu_ns
        ms_delta_bins[i] = delta_bins
        ms_idx1[i] = idx1
        ms_idx2[i] = idx2
        ms_xc1[i] = ss_xc[idx1]
        ms_yc1[i] = ss_yc[idx1]
        ms_dt1[i] = ss_dt[idx1]
        ms_xc2[i] = ss_xc[idx2]
        ms_yc2[i] = ss_yc[idx2]
        ms_dt2[i] = ss_dt[idx2]
    
    return {
        'waveforms': ms_waveforms,
        'delta_mu': ms_delta_mu,
        'delta_bins': ms_delta_bins,
        'idx1': ms_idx1,
        'idx2': ms_idx2,
        'xc1': ms_xc1,
        'yc1': ms_yc1,
        'dt1': ms_dt1,
        'xc2': ms_xc2,
        'yc2': ms_yc2,
        'dt2': ms_dt2,
    }


def save_ms_dataset(data: dict, output_path: str, single_node: bool = False):
    """Save MS dataset to h5 file."""
    with h5py.File(output_path, 'w') as f:
        for key, value in data.items():
            f.create_dataset(key, data=value, compression='gzip')
        
        f.attrs['ns_per_bin'] = NS_PER_BIN
        f.attrs['description'] = 'Multi-scatter events generated from single-scatter data'
        if single_node:
            f.attrs['single_node'] = True
    
    print(f"Saved MS dataset to {output_path}")
    print(f"  Events: {len(data['waveforms'])}")
    print(f"  Waveform shape: {data['waveforms'].shape}")
    print(f"  Delta mu range: [{data['delta_mu'].min():.1f}, {data['delta_mu'].max():.1f}] ns")


def main():
    parser = argparse.ArgumentParser(description='Generate multi-scatter data from single-scatter events')
    parser.add_argument('--input', '-i', type=str, default='data/tritium_ss_42.h5',
                        help='Input SS h5 file path')
    parser.add_argument('--output', '-o', type=str, default='data/tritium_ms_42.h5',
                        help='Output MS h5 file path')
    parser.add_argument('--n-events', '-n', type=int, default=10000,
                        help='Number of MS events to generate')
    parser.add_argument('--delta-min', type=int, default=-30,
                        help='Minimum time shift in bins')
    parser.add_argument('--delta-max', type=int, default=30,
                        help='Maximum time shift in bins')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--single-node', action='store_true',
                        help='Sum over channels so each event has shape (1, T); one node per time bin')
    args = parser.parse_args()
    
    print(f"Loading SS data from {args.input}...")
    ss_waveforms, ss_xc, ss_yc, ss_dt = load_ss_data(args.input)
    print(f"  Loaded {len(ss_waveforms)} SS events")
    print(f"  Waveform shape: {ss_waveforms.shape}")
    
    print(f"\nGenerating {args.n_events} MS events...")
    if args.single_node:
        print("  Mode: single-node (sum over channels -> shape (1, T) per event)")
    print(f"  Delta range: [{args.delta_min}, {args.delta_max}] bins")
    print(f"  Delta range: [{args.delta_min * NS_PER_BIN:.0f}, {args.delta_max * NS_PER_BIN:.0f}] ns")
    
    ms_data = generate_ms_dataset(
        ss_waveforms, ss_xc, ss_yc, ss_dt,
        n_ms_events=args.n_events,
        delta_range=(args.delta_min, args.delta_max),
        seed=args.seed,
        single_node=args.single_node,
    )
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    save_ms_dataset(ms_data, args.output, single_node=args.single_node)


if __name__ == '__main__':
    main()
