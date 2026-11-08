# Source data

Source data of the figures of the paper, as requested by Nature Communications: one Excel file, `paper/submission/Source_Data.xlsx`, with one sheet per figure ("Fig. 2" to "Fig. 7", "Supplementary Fig. 1" to "Supplementary Fig. 9").

## Content

* `Source_Data.xlsx`: in each sheet, the panels follow each other from top to bottom. Each table starts with a title row in bold ("Fig. 2a"; the file and sheet names are added when a panel has several tables), then the column names and the data, followed by an empty row.
* `<panel>/` (e.g. `fig2a/`, `supp_fig6c/`): the source data of one panel (`.xlsx`) and its figure (`.svg`), written by the notebook of the same name in `notebooks/2_results_editorial_requests/` (e.g. `fig2a.ipynb`). Each notebook replots its panel from the saved `.xlsx`, which shows that the source data is sufficient to reproduce the panel. The traces and spectra of Supplementary Fig. 2 (`supp_fig2a` to `supp_fig2g`) are saved and plotted downsampled for display (minimum and maximum of each 1-ms block of the traces; lowest and highest point of each spectrum in each of 300 log-spaced frequency bins; `downsample_for_display` in `src/nodes/validation/trace_plots.py` and `psd_plots.py`), so that the panels are unchanged and the Source Data fit in one file; the power-law fits and the statistics use the full-resolution data.

Fig. 1 has no source data (it shows no data graphs).

## How to generate `Source_Data.xlsx`

1. Run the notebook of each panel in `notebooks/2_results_editorial_requests/` (kernel `spikebias`, environment `envs/spikebias.yml`). Each notebook writes `figures/8_source_data/<panel>/<panel>.xlsx` and `.svg`. The data the notebooks need are downloaded from DANDI (dandiset 001250) with `0_download_recordings.ipynb`, `0_download_ground_truths.ipynb` and `0_download_sortings.ipynb`.
2. Merge the panels into one file (about 40 seconds), from the root of the repository:

```bash
PYTHONPATH=$PWD envs/spikebias/bin/python src/pipes/source_data/merge_source_data.py
```

The script reads every `figures/8_source_data/<panel>/*.xlsx` and writes `paper/submission/Source_Data.xlsx`. Rerun it whenever a panel's source data changes.

## Size

`Source_Data.xlsx` is about 27 MB, below the journal's limit of 30 MB per file. The largest sheets are Fig. 7 (214,491 rows), Fig. 2 (169,589 rows), Fig. 3 (130,675 rows) and Supplementary Fig. 2 (52,428 rows).
