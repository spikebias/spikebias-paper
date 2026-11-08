"""Statistics of Fig. 5a, written next to its source data

author: laquitainesteeve@gmail.com

Usage:

    PYTHONPATH=$PWD envs/spikebias/bin/python src/pipes/source_data/add_statistics_fig5.py
"""

import numpy as np
import pandas as pd
from scipy import stats

SOURCE = "figures/source_data/fig5a"
TESTS = {"M, NS, NS-2X": ["M", "NS", "NS-2X"], "M, E": ["M", "E"], "M, S": ["M", "S"], "H, DS": ["H", "DS"], "M, H": ["M", "H"], "NS, DS": ["NS", "DS"]}


def main():
    d = pd.concat([pd.read_excel(f"{SOURCE}/fig5a_npx.xlsx"), pd.read_excel(f"{SOURCE}/fig5a_dense.xlsx")])
    single = d[d["unit_type"] == "single-unit"].set_index("experiment")["n_units"]
    multi = d[d["unit_type"] == "multi-unit"].set_index("experiment")["n_units"]
    rows = []
    for name, ks in TESTS.items():
        pct = np.array([single[k] / (single[k] + multi[k]) * 100 for k in ks])
        g = stats.chisquare(f_obs=pct)
        rows.append(dict(compared_experiments=name, n_units=int(sum(single[k] + multi[k] for k in ks)), df=len(ks) - 1,
                         chi_square_goodness_of_fit_on_percentages=g.statistic, p_value_goodness_of_fit_on_percentages=g.pvalue))
    out = pd.DataFrame(rows)
    out.to_excel(f"{SOURCE}/fig5a_statistics.xlsx", index=False)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
