"""Panels of Fig. 4b-c: box plots of the log firing rates and of their variances in each layer

author: laquitainesteeve@gmail.com
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.patches import Polygon
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

LAYERS = ["L1", "L2/3", "L4", "L5", "L6"]
EXPERIMENTS = ["M", "NS", "E", "S"]
# colours of the boxes
COLORS = {"M": (0.6, 0.6, 0.6), "NS": (0.804, 0.235, 0.243), "E": (0.875, 0.494, 0.125), "S": (0.282, 0.494, 0.659)}
ALPHA = 0.4
LINEWIDTH = 0.5
FONT = 5
# star of the figure
STAR = np.array([[0.861, 1.428], [0.002, 0.291], [-0.865, 1.428], [-1.299, 1.244], [-0.4, 0.182], [-2.068, 0.182], [-2.068, -0.182], [-0.4, -0.182],
                 [-1.299, -1.244], [-0.865, -1.428], [0.002, -0.291], [0.861, -1.428], [1.299, -1.244], [0.4, -0.182], [2.068, -0.182], [2.068, 0.182],
                 [0.4, 0.182], [1.299, 1.244]])
NS_DX, NS_DY = (1.25, 2.48, 2.5), (-0.4, -3.1, -3.1)       # start of the 'ns' text

# geometry measured on paper/figures/fig4.pdf
PANELS = {
    "b": dict(
        value="firing rate", xlabel="Log10 of firing rate", region=(143.0, 66.0, 226.0, 322.0),
        x_zero=179.32, x_per_unit=18.99, box_h=6.85,
        slot0=[69.25, 118.25, 167.3, 216.3, 265.3], slot_dy=8.6,                  # y of the M box of each layer, spacing of the boxes
        spine_x=157.29, spine_y=(72.36, 288.0), ticks_y=[88.14, 134.16, 182.17, 229.18, 278.2], tick_len=7.0,
        axis_y=298.78, axis_x=(160.33, 221.11), xticks=[(-1, 160.33, 159.85), (0, 179.32, 179.5), (1, 198.32, 198.65), (2, 217.3, 217.2)],
        xlabel_xy=(193.22, 316.64), tick_label_y=309.8,
        bracket_x=[208.45, 214.15, 219.85], star_dx=[3.14, 3.33, 3.9],
    ),
    "c": dict(
        value="Var(log(firing rate))", xlabel="Var of log10 of firing rate", region=(245.0, 66.0, 316.0, 322.0),
        x_zero=258.94, x_per_unit=84.4, box_h=6.7,
        slot0=[72.3, 120.2, 168.1, 216.05, 264.0], slot_dy=8.4,
        spine_x=256.41, spine_y=(70.57, 286.21), ticks_y=[86.94, 134.86, 180.79, 228.70, 276.63], tick_len=7.0,
        axis_y=296.99, axis_x=(258.94, 309.58), xticks=[(0, 258.94, 259.5), (0.5, 301.14, 302.55)],
        xlabel_xy=(284.44, 314.16), tick_label_y=308.1,
        bracket_x=[291.91, 298.35, 304.83], star_dx=[2.7, 2.9, 3.8],
        min_units_M=3,              # layers with fewer Marques-Smith units are not tested
    ),
}


def symbol(p):
    return "ns" if p >= 0.05 else "*" if p >= 0.01 else "**" if p >= 0.001 else "***"


def too_few_units_M(df, layer, min_units_M):
    """True when the layer has fewer than min_units_M Marques-Smith units"""
    if min_units_M is None:
        return False
    n = df[(df["layer"] == layer) & (df["experiment"] == "M")]["n_units"]
    return len(n) > 0 and n.iloc[0] < min_units_M


def mann_whitney(df, value, layer, exp, min_units_M=None):
    """P of the two-sided Mann-Whitney U test of an experiment against M in a layer"""
    m = df[(df["layer"] == layer) & (df["experiment"] == "M")][value].dropna().values
    t = df[(df["layer"] == layer) & (df["experiment"] == exp)][value].dropna().values
    if len(m) == 0 or len(t) == 0 or too_few_units_M(df, layer, min_units_M):
        return None
    return mannwhitneyu(m, t, method="auto").pvalue


def plot_source_data(df, panel):
    """Panel b or c from the source data"""
    g = PANELS[panel]
    x0, y0, x1, y1 = g["region"]
    fig = plt.figure(figsize=((x1 - x0) / 72, (y1 - y0) / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    to_x = lambda v: g["x_zero"] + g["x_per_unit"] * np.asarray(v, dtype=float)
    k = dict(color="k", linewidth=LINEWIDTH, solid_capstyle="butt")

    # axes: layer ticks and value axis
    ax.plot([g["spine_x"]] * 2, g["spine_y"], **k)
    for y in g["ticks_y"]:
        ax.plot([g["spine_x"] - g["tick_len"], g["spine_x"]], [y, y], **k)
    ax.plot(g["axis_x"], [g["axis_y"]] * 2, **k)
    for v, x, xl in g["xticks"]:
        ax.plot([x, x], [g["axis_y"], g["axis_y"] + 5.0], **k)
        ax.text(xl, g["tick_label_y"] + 0.0, f"{v:g}", ha="center", va="baseline", fontsize=FONT)
    ax.text(*g["xlabel_xy"], g["xlabel"], ha="center", va="baseline", fontsize=FONT)

    # boxes
    centres = {}
    for li, layer in enumerate(LAYERS):
        for ei, exp in enumerate(EXPERIMENTS):
            y = g["slot0"][li] + ei * g["slot_dy"]
            centres[(layer, exp)] = y
            v = df[(df["layer"] == layer) & (df["experiment"] == exp)][g["value"]].dropna().values.astype(float)
            if len(v) == 0:
                continue
            s = boxplot_stats(v, whis=1.5)[0]
            stats = {k_: (to_x(s[k_]) if k_ != "fliers" else to_x(s["fliers"])) for k_ in ("med", "q1", "q3", "cilo", "cihi", "whislo", "whishi", "fliers")}
            c = COLORS[exp]
            ax.bxp([stats], positions=[y], widths=g["box_h"], vert=False, shownotches=True, patch_artist=True, manage_ticks=False,
                   boxprops=dict(facecolor=c + (ALPHA,), edgecolor=c, linewidth=LINEWIDTH), whiskerprops=dict(color=c, linewidth=LINEWIDTH),
                   capprops=dict(color=c, linewidth=LINEWIDTH), medianprops=dict(color=c, linewidth=LINEWIDTH),
                   flierprops=dict(marker="o", markerfacecolor="w", markeredgecolor=c, markersize=2.7, markeredgewidth=LINEWIDTH))

    # brackets and stars: P values Holm-corrected for the comparisons of the panel
    tests = [(layer, ei, exp, mann_whitney(df, g["value"], layer, exp, g.get("min_units_M"))) for layer in LAYERS for ei, exp in enumerate(EXPERIMENTS[1:])]
    tests = [t_ for t_ in tests if t_[3] is not None]
    p_holm = multipletests([t_[3] for t_ in tests], method="holm")[1]
    for (layer, ei, exp, _), p in zip(tests, p_holm):
        xb = g["bracket_x"][ei]
        ya, yb = centres[(layer, "M")] - 0.35, centres[(layer, exp)] + (1.8, 1.5, 1.3)[ei]
        ax.plot([xb, xb], [ya, yb], **k)
        sym = symbol(p)
        ym = (ya + yb) / 2
        if sym == "ns":
            ax.text(xb + NS_DX[ei], ym + NS_DY[ei], "ns", rotation=270, rotation_mode="anchor", ha="left", va="baseline", fontsize=FONT, fontweight="bold")
        else:
            for j in range(len(sym)):
                c = np.array([xb + g["star_dx"][ei], ym + (j - (len(sym) - 1) / 2) * 3.24 + 0.15])
                ax.add_patch(Polygon(STAR + c, closed=True, facecolor="k", edgecolor="none"))
    return fig
