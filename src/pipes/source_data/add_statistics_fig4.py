"""Statistics of Fig. 4b-c, written next to their source data

author: laquitainesteeve@gmail.com

Usage:

    PYTHONPATH=$PWD envs/spikebias/bin/python src/pipes/source_data/add_statistics_fig4.py
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from src.nodes.validation.firing_rate_plots import LAYERS, PANELS, symbol, too_few_units_M

SOURCE = "figures/source_data"
SEED = 0
N_BOOT = 10_000


def median_difference(m, x, to_rate, rng):
    """Difference of the medians and its 95% bootstrap confidence interval, in the unit given by to_rate"""
    m, x = to_rate(np.asarray(m)), to_rate(np.asarray(x))
    boot = [np.median(rng.choice(x, len(x))) - np.median(rng.choice(m, len(m))) for _ in range(N_BOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return np.median(x) - np.median(m), lo, hi


def per_layer(df, value, to_rate, min_units_M=None):
    rng = np.random.default_rng(SEED)
    rows = []
    for layer in LAYERS:
        m = df[(df["layer"] == layer) & (df["experiment"] == "M")][value].dropna().values
        too_few = too_few_units_M(df, layer, min_units_M)     # not tested: fewer than min_units_M Marques-Smith units
        for exp in ["NS", "E", "S"]:
            x = df[(df["layer"] == layer) & (df["experiment"] == exp)][value].dropna().values
            if len(m) == 0 or len(x) == 0:
                continue
            diff, lo, hi = median_difference(m, x, to_rate, rng)
            row = dict(layer=layer, comparison=f"{exp} vs M", test="Mann-Whitney U (two-sided)", n_M=len(m))
            if min_units_M is not None:
                row["n_units_M"] = int(df[(df["layer"] == layer) & (df["experiment"] == "M")]["n_units"].iloc[0])   # Marques-Smith units resampled
            row.update(n_simulation=len(x), median_M=to_rate(np.median(m)), median_simulation=to_rate(np.median(x)), difference_of_medians=diff,
                       ci95_low=lo, ci95_high=hi)
            if too_few:
                row.update(test=f"not tested (fewer than {min_units_M} Marques-Smith units)", U_M=np.nan, p_value=np.nan)
            else:
                res = mannwhitneyu(m, x, alternative="two-sided", method="auto")
                row.update(U_M=res.statistic, p_value=res.pvalue)
            rows.append(row)
    out = pd.DataFrame(rows)
    tested = out["p_value"].notna()
    out["p_value_holm"] = np.nan
    out.loc[tested, "p_value_holm"] = multipletests(out.loc[tested, "p_value"].values, method="holm")[1]   # Holm correction for the comparisons of the panel
    out["significance"] = [symbol(p) if t else "not tested" for p, t in zip(out["p_value_holm"], tested)]
    return out


def layers_5_6(df, value):
    rng = np.random.default_rng(SEED)
    rate = lambda e: 10 ** df[df["experiment"].isin(e) & df["layer"].isin(["L5", "L6"])][value].dropna().values   # spikes/s
    m = rate(["M"])
    rows = []
    for name, exps in (("NS vs M", ["NS"]), ("E vs M", ["E"]), ("NS and E pooled vs M", ["NS", "E"]), ("S vs M", ["S"])):
        x = rate(exps)
        diff = np.median(x) - np.median(m)
        boot = [np.median(rng.choice(x, len(x))) - np.median(rng.choice(m, len(m))) for _ in range(N_BOOT)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res = mannwhitneyu(m, x, alternative="two-sided", method="auto")
        rows.append(dict(comparison=name, layers="L5 and L6", n_M=len(m), n_simulation=len(x), median_M_spikes_per_s=np.median(m),
                         median_simulation_spikes_per_s=np.median(x), difference_spikes_per_s=diff, ci95_low=lo, ci95_high=hi, test="Mann-Whitney U (two-sided)",
                         U_M=res.statistic, p_value=res.pvalue))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for panel, to_rate in (("b", lambda v: 10 ** v), ("c", lambda v: v)):
        value = PANELS[panel]["value"]
        df = pd.read_excel(f"{SOURCE}/fig4{panel}/fig4{panel}.xlsx")
        pl = per_layer(df, value, to_rate, PANELS[panel].get("min_units_M"))
        with pd.ExcelWriter(f"{SOURCE}/fig4{panel}/fig4{panel}_statistics.xlsx") as w:
            pl.to_excel(w, sheet_name="per_layer", index=False)
            if panel == "b":
                layers_5_6(df, value).to_excel(w, sheet_name="layers_5_6", index=False)
        print(f"fig4{panel}: {pl.p_value.notna().sum()} tests ({pl.p_value.isna().sum()} comparisons not tested), smallest P {pl.p_value.min():.1e}, largest P {pl.p_value.max():.1e}")
    print(pd.read_excel(f"{SOURCE}/fig4b/fig4b_statistics.xlsx", sheet_name="layers_5_6").round(3).to_string())
