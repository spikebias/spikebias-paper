"""Panels a-f of Fig. 2: single-site voltage traces, drawn at the size of the panel in the figure

author: laquitainesteeve@gmail.com
"""

import numpy as np
from matplotlib import pyplot as plt

FIG2_SCALE = 34.2 / 27.55
TRACE_W, SPINE_H = 40.8, 31.68
X_OFFSET, Y_OFFSET = 2.04, 1.56     # detached axes


def plot_single_site_trace(t_ms, v, color, yticks=(-84, 0, 35), scale=FIG2_SCALE, xlabel="Time (ms)", ylabel=None):
    """Voltage trace of one site: axes of the size of the figure f"""
    s = scale
    w, h = TRACE_W * s, SPINE_H * s
    left, right, top, bottom = 24.0, 8.0, 6.0, 20.0
    W, H = left + w + right, bottom + h + top
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([left / W, bottom / H, w / W, h / H])
    ax.plot(t_ms, v, color=color, linewidth=0.5)
    ax.set_xlim(0, 1000)
    ax.set_ylim(yticks[0], yticks[-1])
    ax.set_yticks(list(yticks))
    ax.set_xticks([0, 500, 1000])
    ax.tick_params(labelsize=5, length=3, width=0.5, pad=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_position(("outward", X_OFFSET * s))
    ax.spines["bottom"].set_position(("outward", Y_OFFSET * s))
    ax.set_xlabel(xlabel, fontsize=5, labelpad=1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=5, labelpad=1)
    return fig
