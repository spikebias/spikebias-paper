"""Merge the source data of every panel into one Excel file of less than 30 MB, one sheet per figure

author: laquitainesteeve@gmail.com

Usage:

    PYTHONPATH=$PWD envs/spikebias/bin/python src/pipes/source_data/merge_source_data.py
"""

import glob
import math
import os
import re
import time

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font

SOURCE_DIR = "figures/source_data"
OUTPUT = os.path.join("paper", "submission", "Source_Data.xlsx")
MAX_ROWS = 1_048_576  # rows of an Excel sheet


def figure_of(panel: str):
    """('fig', 2, 'a') for fig2a, for supp_fig6c"""
    m = re.fullmatch(r"(supp_fig|fig)(\d+)([a-z]?)", panel)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def sheet_name(kind: str, number: int) -> str:
    return f"Fig. {number}" if kind == "fig" else f"Supplementary Fig. {number}"


def panel_title(kind: str, number: int, letter: str) -> str:
    return sheet_name(kind, number) + letter


def clean(value):
    """Excel cannot store NaN or infinite values: write an empty cell"""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_book(output: str, selection: dict, t0: float):
    """Write one workbook: selection maps (kind, number) to a list of (letter, files)"""
    wb = Workbook(write_only=True)
    bold = Font(bold=True)
    for kind, number in sorted(selection, key=lambda k: (k[0] == "supp_fig", k[1])):
        ws = wb.create_sheet(sheet_name(kind, number))
        n_rows = 0
        for letter, files in sorted(selection[(kind, number)]):
            title = panel_title(kind, number, letter)
            tables = [(f, s, d) for f in files for s, d in pd.read_excel(f, sheet_name=None).items()]
            for f, s, d in tables:
                label = title if len(tables) == 1 else f"{title} ({os.path.basename(f)}, sheet {s})"
                cell = WriteOnlyCell(ws, value=label)
                cell.font = bold
                ws.append([cell])
                ws.append([str(c) for c in d.columns])
                for row in d.itertuples(index=False):
                    ws.append([clean(v) for v in row])
                ws.append([])
                n_rows += len(d) + 3
        assert n_rows <= MAX_ROWS, f"{sheet_name(kind, number)} has {n_rows} rows, more than an Excel sheet can hold"
        print(f"{sheet_name(kind, number)}: {len(selection[(kind, number)])} panels, {n_rows:,} rows ({time.time() - t0:.0f} s)", flush=True)
    wb.save(output)
    print(f"saved {output} ({os.path.getsize(output) / 1e6:.1f} MB) in {time.time() - t0:.0f} s", flush=True)


def main():
    t0 = time.time()
    panels = {}
    for folder in sorted(glob.glob(os.path.join(SOURCE_DIR, "*"))):
        key = figure_of(os.path.basename(folder))
        if key is None or not os.path.isdir(folder):
            continue
        files = sorted(glob.glob(os.path.join(folder, "*.xlsx")))
        if files:
            panels.setdefault(key[:2], []).append((key[2], files))

    write_book(OUTPUT, panels, t0)


if __name__ == "__main__":
    main()
