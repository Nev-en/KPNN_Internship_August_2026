"""House style for every figure in src/comp -- import it, never restyle locally.

The rules
---------
* At most 160 mm wide and 160 mm high (``subplots`` refuses anything larger).
* Every text element 6 pt, Liberation Sans, black.
* SVG with a transparent background. Text stays text (editable); only dense
  scatter points are embedded as a raster, at RASTER_DPI.

Liberation Sans is metric-compatible with Arial, so Arial is the fallback: on a
machine without Liberation Sans the layout is measured with Arial and comes out
identical, and the SVG still names Liberation Sans first.

One encoding per meaning, in every figure
-----------------------------------------
    condition           colour      CONDITION_COLORS (validated categorical, dataviz slots 1-5)
    layer               marker      genes = circle, TFs = square
    class               fill        class 0 = filled, class 1 = open
    reference run       colour      black (Karla / kpnn v1), the compared run keeps the condition colour
    port stage          colour      STAGE_COLORS, an ordinal grey ramp: the darker, the closer to v1
    y = x, zero lines   grey guides, never data colours

Condition colours sit below 3:1 contrast on white for INS, TGFB1 and TNFA, so a
condition is always named in text (panel title or tick label), never by colour
alone.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MM = 1 / 25.4
MAX_WIDTH_MM = 160
MAX_HEIGHT_MM = 160
FONT = "Liberation Sans"
FONT_SIZE = 6
EXT = "svg"
RASTER_DPI = 300

CONDITIONS = ["IFNB", "IFNG", "INS", "TGFB1", "TNFA"]
CONDITION_COLORS = {
    "IFNB": "#2a78d6",
    "IFNG": "#eb6834",
    "INS": "#1baf7a",
    "TGFB1": "#eda100",
    "TNFA": "#e87ba4",
}

LAYER_NAMES = {0: "layer 0 — genes", 1: "layer 1 — TFs"}
LAYER_MARKERS = {0: "o", 1: "s"}
LAYER_POINT_AREA = {0: 1.2, 1: 5.0}     # scatter marker area in pt²

CLASS_NAMES = {0: "class 0", 1: "class 1"}

REFERENCE_COLOR = "black"

# How a run is named in a figure. The comparison tables keep their column names
# ("Karla KPNN", ...); only the text drawn in the figure changes, so every
# figure names the same run the same way.
DISPLAY_NAMES = {
    "Karla KPNN": "KPNN",
    "kpnn2 port (v1 rng)": "KPNN2",
    "kpnn2 port": "KPNN2 (torch rng)",   # the port before rng="v1" existed
    "kpnn2 port alt": "KPNN2 (no reseed)",  # kpnn2_port_alt.py, the earliest port
    "kpnn2 port noinfusion": "KPNN2 (v1 rng, no infusion)",  # kpnn2_port_noinfusion.py
}

STAGE_COLORS = {
    "own_init": "#a8a6a0",      # kpnn2's own initialisation / no reseed
    "infused": "#6b6a65",       # v1 starting weights, own training draws
    "v1_rng": "#1f1f1e",        # v1 starting weights and v1 training draws
}

GUIDE_COLOR = "0.55"            # y = x and similar reference lines
ZERO_COLOR = "0.85"             # zero lines and grids
LINE_WIDTH = 0.6


def apply() -> None:
    """Set the rcParams. Called on import, so importing the module is enough."""
    black = "black"
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "Arial"],
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "legend.title_fontsize": FONT_SIZE,
        "figure.titlesize": FONT_SIZE,
        "figure.labelsize": FONT_SIZE,
        "axes.titleweight": "normal",
        "figure.titleweight": "normal",
        "text.color": black,
        "axes.labelcolor": black,
        "axes.titlecolor": black,
        "axes.edgecolor": black,
        "xtick.color": black,
        "ytick.color": black,
        "xtick.labelcolor": black,
        "ytick.labelcolor": black,
        "legend.labelcolor": black,
        "legend.frameon": False,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "axes.labelpad": 2,
        "axes.titlepad": 3,
        "lines.linewidth": LINE_WIDTH,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "savefig.format": EXT,
        "svg.fonttype": "none",
    })


apply()


def subplots(width_mm: float, height_mm: float, *args, **kwargs):
    """plt.subplots sized in mm, capped at MAX_WIDTH_MM x MAX_HEIGHT_MM."""
    if width_mm > MAX_WIDTH_MM or height_mm > MAX_HEIGHT_MM:
        raise ValueError(f"figure {width_mm}x{height_mm} mm exceeds "
                         f"{MAX_WIDTH_MM}x{MAX_HEIGHT_MM} mm")
    kwargs.setdefault("layout", "constrained")
    return plt.subplots(*args, figsize=(width_mm * MM, height_mm * MM), **kwargs)


def save(fig, path: Path) -> Path:
    """Write fig as a transparent SVG next to path (the suffix is replaced)."""
    width, height = (v / MM for v in fig.get_size_inches())
    if width > MAX_WIDTH_MM + 0.01 or height > MAX_HEIGHT_MM + 0.01:
        raise ValueError(f"{path.name}: {width:.0f}x{height:.0f} mm exceeds the size cap")
    out = Path(path).with_suffix(f".{EXT}")
    fig.savefig(out, dpi=RASTER_DPI, transparent=True)
    plt.close(fig)
    return out


def label_overrides(x_col: str, y_col: str) -> None:
    """Let X_LABEL / Y_LABEL rename the two runs of one comparison.

    For when the default name fits most figures but not this one, e.g. Karla's
    run is "KPNN" next to KPNN2 but "KPNN (Karla)" next to a local v1 rerun.
    """
    for col, env in ((x_col, "X_LABEL"), (y_col, "Y_LABEL")):
        if os.environ.get(env):
            DISPLAY_NAMES[col] = os.environ[env]


def display(name: str) -> str:
    """The name a run carries in figures (DISPLAY_NAMES, else unchanged)."""
    return DISPLAY_NAMES.get(name, name)


def face(color: str, cls: int) -> str:
    """Marker face for a class: filled for class 0, open for class 1."""
    return color if cls == 0 else "none"


def class_legend_handles(marker: str = "s") -> list:
    """Neutral legend entries that explain the fill encoding of classes."""
    return [
        Line2D([], [], ls="none", marker=marker, ms=3, mec="black",
               mfc=face("black", cls), mew=0.6, label=CLASS_NAMES[cls])
        for cls in (0, 1)
    ]


def tidy(ax) -> None:
    """Hide the top and right spines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
