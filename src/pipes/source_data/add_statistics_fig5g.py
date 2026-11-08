"""Holm-corrected P values of the weights of Fig. 5g, written into its source data

author: laquitainesteeve@gmail.com

Usage:

    PYTHONPATH=$PWD envs/spikebias/bin/python src/pipes/source_data/add_statistics_fig5g.py
"""

import pandas as pd
from statsmodels.stats.multitest import multipletests

PATH = "figures/source_data/fig5g/fig5g.xlsx"

if __name__ == "__main__":
    df = pd.read_excel(PATH).drop(columns="p_value_holm", errors="ignore")
    df.insert(df.columns.get_loc("p_value") + 1, "p_value_holm", multipletests(df["p_value"], method="holm")[1])
    df.to_excel(PATH, index=False)
    print(df[["feature", "weight", "p_value", "p_value_holm", "significance"]].to_string(index=False))
