"""Panel t of Fig. 2 with the statistics and the median +- 95% CI labels, from the source data

author: laquitainesteeve@gmail.com
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kruskal

LAYERS = ["L1", "L2/3", "L4", "L5", "L6"]
COLORS = {"H": "k", "DS": np.array([26, 152, 80]) / 255}
LAYER_W = 27.55
DECADE = 18.6
BOX_W = 4.3 / LAYER_W                # box width, in layers
FIG2_SCALE = 34.2 / LAYER_W
# and 0.805 times as high on the vertical axis
FIG2_YSCALE = 36.0 / 44.8
# centre of the H and DS boxes of each layer relative to the layer's tick, in pt
CENTRES = {"H": [-3.55, -2.85, -4.45, -3.55, -7.65], "DS": [5.35, 4.9, 3.3, 4.0, 2.5]}
Y_LOG = (-2.65, -0.26)               # y limits, log10
LABEL_COLOR = (0.55, 0.55, 0.55)


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return 1.96 * np.std(x) / np.sqrt(len(x))


def one_digit(v):
    """Value with one significant digit, in the format: 0.05, 7e-4"""
    if v == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(v))))
    r = round(v, -exp)
    if abs(r) >= 0.01:
        return f"{r:.{max(0, -exp)}f}".rstrip("0").rstrip(".") if abs(r) < 1 else f"{r:g}"
    return f"{r / 10 ** exp:.0f}e{exp}"


def label(x):
    """'median ± CI'"""
    return f"{one_digit(np.median(x))} ± {one_digit(ci95(x))}"


def stars(p):
    return "ns" if p >= 0.05 else "*" if p >= 0.01 else "**" if p >= 0.001 else "***"


def plot_ap_power_dense(df, xlabel=False, scale=FIG2_SCALE, yscale=FIG2_YSCALE):
    """Box plots of the AP power of Horvath and the dense-probe model in each layer, with statistics and labels"""
    layer_w, decade = LAYER_W * scale, DECADE * yscale
    left, right, top, bottom = 31.0 + 6.0 * (scale - 1), 8.0, 16.0, 8.0 + (8.0 if xlabel else 0.0)
    data_w = (4 + 0.5 + 0.5) * layer_w
    data_h = (Y_LOG[1] - Y_LOG[0]) * decade
    W, H = left + data_w + right, bottom + data_h + top
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([left / W, bottom / H, data_w / W, data_h / H])
    ax.set_yscale("log")
    ax.set_ylim(10 ** Y_LOG[0], 10 ** Y_LOG[1])
    ax.set_xlim(-0.5, 4.5)
    for i, layer in enumerate(LAYERS):
        pair = {}
        for exp in ("H", "DS"):
            shift = CENTRES[exp][i] * scale / layer_w
            x = df[(df["Experiment"] == exp) & (df["Layer"] == layer)]["power"].dropna().values
            pair[exp] = x
            c = COLORS[exp]
            bp = ax.boxplot([x], positions=[i + shift], widths=BOX_W, notch=True, patch_artist=True, showfliers=True,
                            boxprops=dict(facecolor=c, edgecolor=c, linewidth=0.5), whiskerprops=dict(color=c, linewidth=0.5),
                            capprops=dict(color=c, linewidth=0.5), medianprops=dict(color="w", linewidth=0.5),
                            flierprops=dict(marker="o", markersize=2, markerfacecolor=c, markeredgecolor=c, markeredgewidth=0.3))
            # label: H centred on its box, DS ending just below its lower whisker
            q1, q3 = np.percentile(x, [25, 75])
            low = x[x >= q1 - 1.5 * (q3 - q1)].min()
            if exp == "H":
                ax.text(i + shift - 5.6 * scale / layer_w, np.median(x), label(x), rotation=90, ha="center", va="center", fontsize=5, color=LABEL_COLOR)
            else:
                ax.text(i + shift - 4.0 * scale / layer_w, low * 0.8, label(x), rotation=90, ha="center", va="bottom", fontsize=5, color=LABEL_COLOR)
        # significance of the difference between the two boxes
        p = kruskal(pair["H"], pair["DS"]).pvalue
        top_val = max(np.percentile(pair["H"], 75) + 1.5 * (np.percentile(pair["H"], 75) - np.percentile(pair["H"], 25)),
                      np.percentile(pair["DS"], 75) + 1.5 * (np.percentile(pair["DS"], 75) - np.percentile(pair["DS"], 25)),
                      pair["H"].max() * 0.45)
        y = top_val * 10 ** 0.5
        tick = y / 10 ** 0.18
        ax.plot([i + CENTRES["H"][i] / LAYER_W] * 2 + [i + CENTRES["DS"][i] / LAYER_W] * 2, [tick, y, y, tick], color="k", linewidth=0.5, clip_on=False)
        ax.text(i + (CENTRES["H"][i] + CENTRES["DS"][i]) / 2 / LAYER_W, y * 10 ** 0.04, stars(p), ha="center", va="bottom", fontsize=7, fontweight="bold", clip_on=False)
    ax.annotate("AP power (µV²/Hz)", (0, 0.5), xycoords="axes fraction", xytext=(-15.6 * scale, 0), textcoords="offset points", rotation=90, ha="center", va="center", fontsize=6)
    ax.set_yticks([0.1, 0.01], ["", ""])
    for val, exponent in ((0.1, "\u22121"), (0.01, "\u22122")):     # 10 and its exponent as two texts of 6 and 5 pt
        ax.annotate("10", (0, val), xycoords=("axes fraction", "data"), xytext=(-7, 0), textcoords="offset points", ha="right", va="center", fontsize=6)
        ax.annotate(exponent, (0, val), xycoords=("axes fraction", "data"), xytext=(-6.5, 3), textcoords="offset points", ha="left", va="center", fontsize=5)
    ax.tick_params(axis="y", which="major", length=3, width=0.5)
    ax.tick_params(axis="y", which="minor", length=1.8, width=0.5)
    ax.set_xticks(range(5), LAYERS if xlabel else [""] * 5)
    ax.tick_params(axis="x", length=3, width=0.5, labelsize=6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_position(("outward", 0))
    ax.spines["bottom"].set_position(("outward", 5 * yscale))
    ax.spines["bottom"].set_bounds(0, 4)
    if xlabel:
        ax.set_xlabel("Layer", fontsize=6, labelpad=1)
    return fig


# ---- panel u: frequency scaling slope of the dense probe
UNIT = 20.0
Y_LIN = (-0.09, 2.14)                          # y limits
CENTRES_U = {"H": [-0.25, -3.0, -5.75, -1.45, -3.15], "DS": [3.8, 1.0, -1.8, 2.7, 0.95]}


def plot_slope_dense(df, scale=FIG2_SCALE, yscale=FIG2_YSCALE):
    """Panel u of Fig. 2"""
    layer_w, unit = LAYER_W * scale, UNIT * yscale
    left, right, top, bottom = 31.0 + 6.0 * (scale - 1), 8.0, 20.0, 26.0
    data_w = (4 + 0.5 + 0.5) * layer_w
    data_h = (Y_LIN[1] - Y_LIN[0]) * unit
    W, H = left + data_w + right, bottom + data_h + top
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([left / W, bottom / H, data_w / W, data_h / H])
    ax.set_ylim(*Y_LIN)
    ax.set_xlim(-0.5, 4.5)
    for i, layer in enumerate(LAYERS):
        pair = {}
        for exp in ("H", "DS"):
            shift = CENTRES_U[exp][i] / LAYER_W
            x = df[(df["Experiment"] == exp) & (df["Layer"] == layer)]["Alpha"].dropna().values
            pair[exp] = x
            c = COLORS[exp]
            ax.boxplot([x], positions=[i + shift], widths=BOX_W, notch=True, patch_artist=True, showfliers=True,
                       boxprops=dict(facecolor=c, edgecolor=c, linewidth=0.5), whiskerprops=dict(color=c, linewidth=0.5),
                       capprops=dict(color=c, linewidth=0.5), medianprops=dict(color="w", linewidth=0.5),
                       flierprops=dict(marker="o", markersize=2, markerfacecolor=c, markeredgecolor=c, markeredgewidth=0.3))
            q1, q3 = np.percentile(x, [25, 75])
            low = x[x >= q1 - 1.5 * (q3 - q1)].min()
            lab = f"{two_digits(np.median(x))} ± {one_digit(ci95(x))}"
            if exp == "H":
                ax.text(i + shift - 5.6 * scale / layer_w, np.median(x), lab, rotation=90, ha="center", va="center", fontsize=5, color=LABEL_COLOR)
            else:
                ax.text(i + shift - 4.0 * scale / layer_w, max(low, 0) + 0.04, lab, rotation=90, ha="center", va="bottom", fontsize=5, color=LABEL_COLOR)
        p = kruskal(pair["H"], pair["DS"]).pvalue
        q1h, q3h = np.percentile(pair["H"], [25, 75])
        top_val = max(pair["H"][pair["H"] <= q3h + 1.5 * (q3h - q1h)].max(), np.percentile(pair["DS"], 75) * 1.0)
        y = top_val + 0.32
        xs = [i + CENTRES_U["H"][i] / LAYER_W] * 2 + [i + CENTRES_U["DS"][i] / LAYER_W] * 2
        ax.plot(xs, [y - 0.12, y, y, y - 0.12], color="k", linewidth=0.5, clip_on=False)
        ax.text(i + (CENTRES_U["H"][i] + CENTRES_U["DS"][i]) / 2 / LAYER_W, y + 0.02, stars(p), ha="center", va="bottom", fontsize=7, fontweight="bold", clip_on=False)
    ax.annotate("Frequency scaling slope", (0, 0.5), xycoords="axes fraction", xytext=(-21 * scale, 0), textcoords="offset points", rotation=90, ha="center", va="center", fontsize=5)
    ax.annotate("in multi-unit band (µV²/Hz)", (0, 0.5), xycoords="axes fraction", xytext=(-14 * scale, 0), textcoords="offset points", rotation=90, ha="center", va="center", fontsize=5)
    ax.set_yticks([0, 1, 2])
    ax.tick_params(axis="y", length=3, width=0.5, labelsize=6, pad=1.5)
    ax.set_xticks(range(5), LAYERS)
    ax.tick_params(axis="x", length=3, width=0.5, labelsize=6, pad=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_bounds(0 - 0.09, 2.14)
    ax.spines["bottom"].set_position(("outward", 5.4 * yscale))
    ax.spines["bottom"].set_bounds(0, 4)
    ax.set_xlabel("Layer", fontsize=6, labelpad=1)
    return fig


def two_digits(v):
    """Median with 2 significant digits for values of 1 and above, 1 significant digit below"""
    return f"{v:.1f}" if abs(v) >= 1 else one_digit(v)
