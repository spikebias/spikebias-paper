"""Rewrite the text of existing svg figures with explicit font attributes, which Adobe Illustrator reads

author: laquitainesteeve@gmail.com

Usage:

    PYTHONPATH=$PWD envs/spikebias/bin/python src/pipes/figures/fix_svg_fonts.py [folder]    # default: figures/source_data
"""

import glob
import os
import sys

from src.nodes.figure_style import fix_svg_fonts

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "figures/source_data"
    total = 0
    for path in sorted(glob.glob(os.path.join(folder, "**", "*.svg"), recursive=True)):
        n = fix_svg_fonts(path)
        if n:
            total += 1
            print(f"{path}: {n} text elements")
    print(f"{total} files fixed")
