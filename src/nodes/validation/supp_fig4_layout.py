"""Layout of the panels of Supplementary Fig. 4

author: laquitainesteeve@gmail.com
"""

from matplotlib import pyplot as plt
from matplotlib.text import Text

FONT, TITLE = 5, 7

# panel a: left of the axes of each sorter
A_LEFT = [34.57, 135.23, 237.88, 335.40, 430.06]
A_TOP = [50.0, 47.8, 47.9, 49.4, 48.8]
A_W, A_H, A_PITCH = 55.78, 30.5, 36.5
A_CANVAS = (0.0, 0.0, 504.0, 300.0)                      # x0, y0, width and height of the figure in the page

# panels b and c
B_LEFT = [37.24, 84.39, 128.02, 172.44, 216.62]
B_TOP = [409.25, 409.45, 409.28, 409.04, 408.18]
B_W, B_H = 33.47, 150.0
B_CANVAS = (0.0, 372.0, 262.0, 242.0)
C_LEFT = [303.89, 347.02, 387.55, 429.26, 471.45]
C_TOP = [405.18, 404.54, 403.96, 403.53, 403.16]
C_W, C_H = 27.89, 150.0
C_CANVAS = (262.0, 372.0, 242.0, 242.0)


def new_figure(canvas):
    """Figure with the size of the canvas"""
    x0, y0, w, h = canvas
    fig = plt.figure(figsize=(w / 72, h / 72))
    fig._origin = canvas
    return fig


def add_axes_pt(fig, left, top, width, height):
    """Axes at of the page, y down, in pt"""
    x0, y0, w, h = fig._origin
    return fig.add_axes([(left - x0) / w, 1 - (top - y0 + height) / h, width / w, height / h])


def text_pt(fig, x, y, s, **kw):
    """Text centred at of the page, y down, in pt"""
    x0, y0, w, h = fig._origin
    kw = {"ha": "center", "va": "center", **kw}
    return fig.text((x - x0) / w, 1 - (y - y0) / h, s, **kw)


def axes_a(fig):
    """[sorter][row] axes of panel a"""
    return [[add_axes_pt(fig, A_LEFT[c], A_TOP[c] + r * A_PITCH, A_W, A_H) for r in range(6)] for c in range(5)]


def axes_bc(fig, left, top, width, height):
    return [add_axes_pt(fig, left[c], top[c], width, height) for c in range(5)]


def set_text_sizes(fig, titles=()):
    """5 pt for all the text, 7 pt for the titles: the texts in `titles` and the bold texts of 7 pt or more"""
    fig.canvas.draw()
    for ax in fig.axes:
        ax.tick_params(which="both", labelsize=FONT)
    fig.canvas.draw()
    for t in fig.findobj(Text):
        if not t.get_text().strip():
            continue
        is_title = t.get_text() in titles or (t.get_fontweight() in ("bold", 700) and t.get_fontsize() >= 6.9)
        t.set_fontsize(TITLE if is_title else FONT)
        if is_title:
            t.set_fontweight("bold")
    fig.canvas.draw()
