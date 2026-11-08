"""Holm-corrected P values of Supplementary Fig. 3d-f, written into their source data

author: laquitainesteeve@gmail.com

Usage:

    PYTHONPATH=$PWD envs/spikebias/bin/python src/pipes/source_data/add_statistics_supp_fig3.py
"""

import pandas as pd
from statsmodels.stats.multitest import multipletests

PANELS = ["d", "e", "f"]

if __name__ == "__main__":
    for panel in PANELS:
        path = f"figures/source_data/supp_fig3{panel}/supp_fig3{panel}.xlsx"
        sheets = pd.read_excel(path, sheet_name=None)
        stats = sheets["statistics"].drop(columns="p_holm", errors="ignore")
        stats.insert(stats.columns.get_loc("p_one_sided") + 1, "p_holm", multipletests(stats["p_one_sided"], method="holm")[1])
        sheets["statistics"] = stats
        with pd.ExcelWriter(path) as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
        print(f"supp_fig3{panel}"); print(stats.to_string(index=False))
