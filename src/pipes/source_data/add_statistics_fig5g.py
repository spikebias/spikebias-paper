"""Holm-corrected P values of the weights of Fig. 5g, written into its source data

author: laquitainesteeve@gmail.com

Reads figures/8_source_data/fig5g/fig5g.xlsx (weights of the model of good units, two-sided Wald test of each weight, uncorrected P in "p_value") and
adds the column "p_value_holm": the P values Holm-corrected for the 10 weights of the panel. The three weights marked in the figure (firing range,
firing rate and mad_ratio) remain significant (P = 0.0007, 0.0014 and 0.0044). The published source data come from an earlier waveform extraction
(see fig5g.ipynb), so the column is added to the stored file rather than recomputed. Run merge_source_data.py afterwards.

Usage:

    PYTHONPATH=$PWD envs/spikebias/bin/python src/pipes/source_data/add_statistics_fig5g.py
"""

import pandas as pd
from statsmodels.stats.multitest import multipletests

PATH = "figures/8_source_data/fig5g/fig5g.xlsx"

if __name__ == "__main__":
    df = pd.read_excel(PATH).drop(columns="p_value_holm", errors="ignore")
    df.insert(df.columns.get_loc("p_value") + 1, "p_value_holm", multipletests(df["p_value"], method="holm")[1])
    df.to_excel(PATH, index=False)
    print(df[["feature", "weight", "p_value", "p_value_holm", "significance"]].to_string(index=False))
