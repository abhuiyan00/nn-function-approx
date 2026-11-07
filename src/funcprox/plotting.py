"""One figure style for the whole project (report + slides).

Palette: validated categorical order (8 slots, fixed order, never cycled), one-hue
sequential blue for magnitudes, blue/red diverging with a grey midpoint for signed
errors. Text never uses series colours; grid is a recessive 1 px solid line.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
TRUTH = "#0b0b0b"          # exact function: ink, dashed
BAND = "#f0efec"           # neutral shading (training box, intervals)

SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#104281", "#3987e5", "#f0efec", "#e66767", "#9e2a2a"])


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0, "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": TEXT,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 6.0, "font.size": 9.5, "text.color": TEXT,
        "figure.titlesize": 12.5, "figure.titleweight": "bold", "figure.dpi": 110,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "figure.constrained_layout.use": True,
    })


def color(i: int) -> str:
    """Categorical slot i (0-based). Fixed order; more than 8 series must be folded."""
    if i >= len(SERIES):
        raise ValueError("more than 8 categorical series - fold into 'other' or facet")
    return SERIES[i]


def marker_kw(i: int) -> dict:
    return dict(color=color(i), marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)


def truth_kw() -> dict:
    return dict(color=TRUTH, lw=1.2, ls=(0, (4, 3)), label="exact f")


def note(ax, text: str, loc: str = "upper left") -> None:
    """Small caption inside an axis, secondary ink."""
    xy = {"upper left": (0.02, 0.97, "left", "top"), "upper right": (0.98, 0.97, "right", "top"),
          "lower left": (0.02, 0.03, "left", "bottom"), "lower right": (0.98, 0.03, "right", "bottom")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3], fontsize=8, color=TEXT_2)


def save(fig, path: str | Path, subtitle: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if subtitle:
        fig.text(0.01, -0.01, subtitle, fontsize=7.5, color=MUTED, ha="left", va="top")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


setup()
