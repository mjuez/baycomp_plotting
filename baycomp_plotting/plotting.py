"""
Functions for plotting baycomp's posterior plots.

``tern``
    A ternary plot.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from math import sqrt, sin, cos, pi
from iteround import saferound
from scipy.interpolate import interpn

__all__ = ['tern']

# Custom colormap -> dark blue means high density, and light blue low
BLUES_CMAP = np.ones((256, 4))
BLUES_CMAP[:, 0] = np.linspace(199/256, 8/256, 256)
BLUES_CMAP[:, 1] = np.linspace(224/256, 64/256, 256)
BLUES_CMAP[:, 2] = np.linspace(252/256, 129/256, 256)
BLUES_CMAP = ListedColormap(BLUES_CMAP)

def project(pts):
    SQRT_3 = sqrt(3)
    PI_6 = pi / 6

    p1, p2, p3 = pts.T / SQRT_3
    x = (p2 - p1) * cos(PI_6) + .5
    y = p3 - (p1 + p2) * sin(PI_6) + 1 / (2 * SQRT_3)

    return np.vstack((x, y)).T


def tern(p):
    plt.style.use('classic')

    fig, ax = plt.subplots()
    fig.patch.set_alpha(0)
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    ax.set_xlim(-.1, 1.1)
    ax.set_ylim(-.1, 1.1)

    # inner lines
    cx, cy = project(np.array([[1/3, 1/3, 1/3]]))[0] # central point
    for x, y in project(np.array([[.5, .5, 0], [.5, 0, .5], [0, .5, .5]])):
        ax.add_line(Line2D([cx, x], [cy, y], color='k', lw=2))

    # outer border of the triangle
    vert_coords = [(0., 0.), (.5, sqrt(3)/2), (1., 0.), (0., 0.)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    triangle = Path(vert_coords, codes)
    patch = patches.PathPatch(triangle, facecolor='none', lw=3)
    ax.add_patch(patch)

    # vertices texts
    probs = saferound(list(p.probs()), places=4)
    # L
    ax.text(-.04, -.02, r'$\mathrm{L}$', ha='center', va='top', fontsize=30)
    ax.text(-.04, -.12, r'$\mathbf{(%.4f)}$'%probs[0], ha='center', va='top', fontsize=32)
    # ROPE
    ax.text(.5, 1, r'$\mathrm{ROPE}$', ha='center', va='bottom', fontsize=30)
    ax.text(.5, .87, r'$\mathbf{(%.4f)}$'%probs[1], ha='center', va='bottom', fontsize=32)
    # R
    ax.text(1.04, -.02, r'$\mathrm{R}$', ha='center', va='top', fontsize=30)
    ax.text(1.04, -.12, r'$\mathbf{(%.4f)}$'%probs[2], ha='center', va='top', fontsize=32)

    # draw points
    tripts = project(p.sample[:, [0, 2, 1]])
    data, xe, ye = np.histogram2d(tripts[:, 0], tripts[:, 1], bins=30)
    z = interpn((.5 * (xe[1:] + xe[:-1]), .5 * (ye[1:] + ye[:-1])), data, 
            np.vstack([tripts[:, 0], tripts[:, 1]]).T, method='splinef2d', 
            bounds_error=False)
    idx = z.argsort()
    ax.scatter(tripts[:, 0][idx], tripts[:, 1][idx], c=z[idx], clip_path=patch,
            s=50, edgecolor='', cmap=BLUES_CMAP, rasterized=True)

    fig.tight_layout()
    return fig
