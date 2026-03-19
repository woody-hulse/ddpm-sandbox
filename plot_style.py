"""plot_style.py — Shared matplotlib style for all figures.

Import and call apply_style() once before creating figures.

Colour conventions:
    COLORS["ae"]       — GraphAE / AE reconstructions
    COLORS["diffae"]   — DiffAE reconstructions
    COLORS["truth"]    — Ground truth / raw data
    COLORS["baseline"] — Baseline / reference lines
    COLORS["model3"]   — Third model (e.g. end-to-end encoder)
    COLORS["model4"]   — Fourth model
"""

import matplotlib

# ---------------------------------------------------------------------------
# Colour palette (colorblind-friendly blue/red pair + accents)
# ---------------------------------------------------------------------------

COLORS: dict = {
    "ae":       "#2166AC",   # deep blue      — GraphAE / AE
    "diffae":   "#D6604D",   # muted red      — DiffAE
    "truth":    "#222222",   # near-black     — ground truth / raw
    "baseline": "#888888",   # medium gray    — baseline reference
    "model3":   "#1A9850",   # forest green   — end-to-end encoder
    "model4":   "#762A83",   # purple         — fourth model
    "lop_left":  "#D6604D",  # same as diffae — lopsided left
    "lop_right": "#2166AC",  # same as ae     — lopsided right
    "lop_none":  "#AAAAAA",  # light gray     — no augmentation
}

# Ordered list for sequential model assignment
MODEL_COLORS: list = [
    COLORS["ae"],
    COLORS["diffae"],
    COLORS["model3"],
    COLORS["model4"],
    "#B35806",   # dark amber
    "#4393C3",   # light blue
]


# ---------------------------------------------------------------------------
# Academic rcParams
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Apply consistent academic matplotlib rcParams.

    Call once at the top of each script that generates figures,
    after `import matplotlib.pyplot as plt`.
    """
    matplotlib.rcParams.update({
        # --- typography ---
        "font.size":          10,
        "axes.labelsize":     11,
        "axes.titlesize":     11,
        "figure.titlesize":   13,
        "legend.fontsize":    9,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,

        # --- axes ---
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,

        # --- grid (subtle) ---
        "axes.grid":          True,
        "grid.color":         "#E5E5E5",
        "grid.linestyle":     "-",
        "grid.linewidth":     0.5,
        "axes.axisbelow":     True,

        # --- ticks ---
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "xtick.minor.size":   2.0,
        "ytick.minor.size":   2.0,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,

        # --- lines ---
        "lines.linewidth":    1.5,

        # --- legend ---
        "legend.frameon":     True,
        "legend.framealpha":  0.95,
        "legend.edgecolor":   "0.8",
        "legend.fancybox":    False,
        "legend.borderpad":   0.4,

        # --- saving ---
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
    })
