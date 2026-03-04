from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
from cycler import cycler

BLOG_PALETTE = [
    "#8B4513",
    "#A55233",
    "#2E2B2B",
    "#78716C",
    "#666666",
]


def blog_rcparams() -> dict:
    return {
        "figure.facecolor": "#FDFCF9",
        "axes.facecolor": "#FDFCF9",
        "savefig.facecolor": "#FDFCF9",
        "axes.edgecolor": "#78716C",
        "axes.labelcolor": "#292524",
        "xtick.color": "#292524",
        "ytick.color": "#292524",
        "text.color": "#292524",
        "axes.grid": True,
        "grid.color": "#78716C",
        "grid.alpha": 0.18,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "axes.prop_cycle": cycler(color=BLOG_PALETTE),
    }


def apply_blog_plot_style() -> None:
    mpl.rcParams.update(blog_rcparams())


@contextmanager
def blog_plot_context():
    with mpl.rc_context(blog_rcparams()):
        yield
