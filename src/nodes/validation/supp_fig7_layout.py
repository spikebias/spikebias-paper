"""Layout of the panels of Supplementary Fig. 7, drawn from the source data

author: laquitainesteeve@gmail.com
"""

import seaborn as sns

from src.nodes.validation.supp_fig4_layout import add_axes_pt, new_figure, set_text_sizes, text_pt

A_CANVAS, A_AXES, A_YLIM = (0.0, 0.0, 300.0, 175.0), (60.6, 17.2, 216.0, 145.9), (-2.97, 1.3)
B_CANVAS, B_AXES, B_YLIM = (0.0, 205.0, 300.0, 170.0), (60.6, 217.2, 216.0, 143.6), (-3.11, 1.35)
C_CANVAS, C_AXES, C_YLIM, C_XLIM = (300.0, 0.0, 240.0, 140.0), (340.8, 14.3, 147.8, 104.8), (0.0, 3.0), (-3.0, 0.75)
D_CANVAS, D_AXES, D_YLIM, D_XLIM = (300.0, 150.0, 240.0, 115.0), (336.0, 160.0, 162.1, 82.0), (0.0, 2.5), (-0.2, 4.2)
E_CANVAS, E_AXES, E_YLIM, E_XLIM = (300.0, 265.0, 240.0, 115.0), (336.0, 281.6, 162.1, 79.2), (-1.545, 0.0), (-0.2, 4.2)


def move_violin_legend(ax):
    """Legend of the violins at the bottom left of the axes"""
    if ax.get_legend() is not None:
        sns.move_legend(ax, "lower left", bbox_to_anchor=(0.0, 0.02), frameon=True)
