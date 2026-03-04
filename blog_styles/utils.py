"""Reusable plotting utilities styled for the blog.

All helpers apply the blog palette and rc settings automatically.  They
return ``(fig, ax)`` tuples so callers can fine-tune before ``plt.show()``.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import BLOG_PALETTE


def plot_labeled_heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str = "",
    ax: plt.Axes | None = None,
    cmap: str = "viridis",
    colorbar: bool = True,
    **imshow_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a labelled heatmap (e.g. adjacency or precision matrix).

    Parameters
    ----------
    matrix : 2-D array.
    labels : tick labels for both axes.
    title : axis title.
    ax : optional pre-existing Axes.
    cmap : Matplotlib colourmap.
    colorbar : whether to add a colourbar.
    **imshow_kwargs : forwarded to ``ax.imshow``.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    else:
        fig = ax.figure
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, **imshow_kwargs)
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax, shrink=0.8)
    return fig, ax


def plot_method_comparison(
    df: pd.DataFrame,
    methods: Sequence[str],
    date_col: str,
    value_col: str,
    title: str = "",
    ylabel: str = "",
    ax: plt.Axes | None = None,
    **plot_kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Line chart comparing several methods over time.

    Parameters
    ----------
    df : DataFrame with at least ``date_col``, ``'method'``, ``value_col``.
    methods : method names to plot.
    date_col, value_col : column names.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))
    else:
        fig = ax.figure
    for m in methods:
        s = df[df["method"] == m].sort_values(date_col)
        ax.plot(s[date_col], s[value_col], label=m, lw=1.1, **plot_kwargs)
    if title:
        ax.set_title(title)
    ax.set_xlabel(date_col.replace("_", " ").title())
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend()
    fig.autofmt_xdate()
    return fig, ax


def plot_empirical_cdf(
    data: dict[str, np.ndarray],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Empirical CDF",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Empirical CDF for several named series.

    Parameters
    ----------
    data : mapping ``{label: 1-D array}``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure
    for name, arr in data.items():
        x = np.asarray(arr, dtype=float)
        x = x[np.isfinite(x)]
        x = np.sort(x)
        if len(x) == 0:
            continue
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, lw=1.2, label=name)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.tight_layout()
    return fig, ax
