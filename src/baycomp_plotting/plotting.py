"""
Functions for plotting baycomp's posterior distributions.

``tern``
    A ternary plot for posteriors over (left, rope, right) probabilities.

``dens``
    A density plot for the posterior of a CorrelatedTTest.

Author: Mario Juez-Gil <mariojg@ubu.es>
"""
from __future__ import annotations

import types
from math import cos, pi, sin, sqrt
from typing import Sequence

import matplotlib.colors as clrs
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.path import Path
from scipy import stats
from scipy.interpolate import interpn

__all__ = ["Color", "tern", "dens"]


_BLUES_CMAP_RGBA = np.ones((256, 4))
_BLUES_CMAP_RGBA[:, 0] = np.linspace(199 / 256, 8 / 256, 256)
_BLUES_CMAP_RGBA[:, 1] = np.linspace(224 / 256, 64 / 256, 256)
_BLUES_CMAP_RGBA[:, 2] = np.linspace(252 / 256, 129 / 256, 256)
BLUES_CMAP = ListedColormap(_BLUES_CMAP_RGBA)


class Color:
    """Alternative palette of four colors for matplotlib plots."""

    BLUE = clrs.to_hex((0 / 255, 142 / 255, 206 / 255))
    GRAY = clrs.to_hex((77 / 255, 80 / 255, 94 / 255))
    BORDEAUX = clrs.to_hex((208 / 255, 33 / 255, 85 / 255))
    GREEN = clrs.to_hex((5 / 255, 126 / 255, 121 / 255))


def _safe_round(values: Sequence[float], places: int) -> list[float]:
    """Round ``values`` to ``places`` decimals while preserving their sum.

    Largest-remainder method: distribute the rounding residual to the entries
    with the largest fractional parts so that, for probabilities summing to 1,
    the rounded triplet still sums to 1.0 to ``places`` decimals.
    """
    if not values:
        return []
    multiplier = 10 ** places
    scaled = [v * multiplier for v in values]
    floors = [int(s) for s in scaled]
    diff = int(round(sum(scaled))) - sum(floors)
    if diff > 0:
        order = sorted(range(len(scaled)), key=lambda i: scaled[i] - floors[i], reverse=True)
        for i in order[:diff]:
            floors[i] += 1
    elif diff < 0:
        order = sorted(range(len(scaled)), key=lambda i: scaled[i] - floors[i])
        for i in order[: -diff]:
            floors[i] -= 1
    return [f / multiplier for f in floors]


def _project(pts: np.ndarray) -> np.ndarray:
    """Project barycentric coordinates onto the 2D simplex triangle."""
    sqrt_3 = sqrt(3)
    pi_6 = pi / 6
    p1, p2, p3 = pts.T / sqrt_3
    x = (p2 - p1) * cos(pi_6) + 0.5
    y = p3 - (p1 + p2) * sin(pi_6) + 1 / (2 * sqrt_3)
    return np.vstack((x, y)).T


def _process_names(names: Sequence[str]) -> list[str]:
    """Escape characters in ``names`` so they survive matplotlib's mathtext."""
    return [n.replace("-", "{-}").replace(" ", "\\ ") for n in names]


def _add_posterior(ax, p, label: str, ls="-", color: str = Color.BLUE) -> None:
    """Render a CorrelatedTTest posterior on ``ax`` and update the y-scale.

    Reads and updates two ad-hoc attributes on ``ax``: ``max_y`` (current y-axis
    upper bound) and ``zo`` (z-order counter, decremented for each posterior so
    earlier ones stay on top).
    """
    targs = (p.df, p.mean, np.sqrt(p.var))
    x = np.linspace(
        min(stats.t.ppf(0.005, *targs), -1.05 * p.rope),
        max(stats.t.ppf(0.995, *targs), 1.05 * p.rope),
        100,
    )
    y = stats.t.pdf(x, *targs)
    y = y / y.sum()
    ax.plot(
        x, y, c=color, linestyle=ls, linewidth=2,
        label=r"$\mathrm{%s}$" % label, zorder=ax.zo,
    )
    ax.fill_between(x, y, facecolor=color, alpha=0.1, edgecolor="none", zorder=ax.zo)
    ax.zo -= 1

    curr_max_y = float(y.max()) * 1.02
    if ax.max_y is None or curr_max_y > ax.max_y:
        ax.max_y = curr_max_y
        tick = ax.max_y / 3
        yticks = [0, tick, 2 * tick, 3 * tick]
        ax.set_ylim(0, ax.max_y)
        ax.set_yticks(yticks, labels=[r"$%.3f$" % v for v in yticks])


def tern(p, names: Sequence[str] = ("L", "R")):
    """Ternary plot for a posterior over (left, rope, right) probabilities.

    Parameters
    ----------
    p
        A baycomp posterior with ``probs()`` returning three values that sum to 1
        and ``sample`` of shape ``(n_samples, 3)`` (e.g. ``HierarchicalTest`` or
        ``SignedRankTest``).
    names
        Two-element sequence with labels for the left and right vertices.
    """
    names = _process_names(list(names))
    plt.style.use("classic")

    fig, ax = plt.subplots()
    fig.patch.set_alpha(0)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    cx, cy = _project(np.array([[1 / 3, 1 / 3, 1 / 3]]))[0]
    for x, y in _project(np.array([[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])):
        ax.add_line(Line2D([cx, x], [cy, y], color="k", lw=2))

    vert_coords = [(0.0, 0.0), (0.5, sqrt(3) / 2), (1.0, 0.0), (0.0, 0.0)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    triangle = Path(vert_coords, codes)
    patch = patches.PathPatch(triangle, facecolor="none", lw=3)
    ax.add_patch(patch)

    probs = _safe_round(list(p.probs()), places=4)
    ax.text(-0.04, -0.02, r"$\mathrm{%s}$" % names[0], ha="center", va="top", fontsize=30)
    ax.text(-0.04, -0.12, r"$\mathbf{(%.4f)}$" % probs[0], ha="center", va="top", fontsize=32)
    ax.text(0.5, 1, r"$\mathrm{ROPE}$", ha="center", va="bottom", fontsize=30)
    ax.text(0.5, 0.87, r"$\mathbf{(%.4f)}$" % probs[1], ha="center", va="bottom", fontsize=32)
    ax.text(1.04, -0.02, r"$\mathrm{%s}$" % names[1], ha="center", va="top", fontsize=30)
    ax.text(1.04, -0.12, r"$\mathbf{(%.4f)}$" % probs[2], ha="center", va="top", fontsize=32)

    tripts = _project(p.sample[:, [0, 2, 1]])
    data, xe, ye = np.histogram2d(tripts[:, 0], tripts[:, 1], bins=30)
    z = interpn(
        (0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])),
        data,
        np.vstack([tripts[:, 0], tripts[:, 1]]).T,
        method="splinef2d",
        bounds_error=False,
    )
    idx = z.argsort()
    ax.scatter(
        tripts[:, 0][idx],
        tripts[:, 1][idx],
        c=z[idx],
        clip_path=patch,
        s=50,
        linewidth=0,
        cmap=BLUES_CMAP,
        rasterized=True,
    )

    fig.tight_layout()
    return fig


def dens(p, label: str, ls="-", color: str = Color.BLUE):
    """Density plot for the posterior of a ``baycomp.CorrelatedTTest``.

    Parameters
    ----------
    p
        A ``CorrelatedTTest`` posterior with attributes ``df``, ``mean``, ``var``
        and ``rope``.
    label
        Math-text label shown in the legend (LaTeX-style spacing must be escaped).
    ls
        Matplotlib line style.
    color
        Density colour, e.g. one of the :class:`Color` constants.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with an extra ``add_posterior(p, label, ls, color)`` method that
        appends another posterior on top of the existing axes.
    """
    plt.style.use("classic")
    plt.rc("legend", fontsize=25)

    fig, ax = plt.subplots()
    fig.patch.set_alpha(0)
    ax.axvline(0.01, c="darkorange", linewidth=2, zorder=101)
    ax.axvline(-0.01, c="darkorange", linewidth=2, zorder=101)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("none")
    ax.tick_params(
        axis="both", which="major", labelsize=30, direction="inout", width=1, length=5,
    )
    ax.set_xticks([])

    ax.max_y = None
    ax.zo = 100
    _add_posterior(ax, p, label, ls, color)
    fig.add_posterior = types.MethodType(_add_posterior, ax)

    fig.tight_layout()
    return fig
