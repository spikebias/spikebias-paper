import pandas as pd
import spikeinterface as si
import os
import spikeinterface as si
import spikeinterface.core.template_tools as ttools
from spikeinterface import comparison
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
import seaborn as sns
from sklearn import metrics
# import cebra.models
import pickle
import copy
from sklearn.preprocessing import StandardScaler
import shutil
from spikeinterface.postprocessing import compute_principal_components
from spikeinterface.qualitymetrics import compute_quality_metrics as qm
from src.nodes.metrics.quality import get_scores
import logging
import logging.config
import yaml

job_kwargs = dict(n_jobs=-1, progress_bar=True)

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


# def get_waveformExtractor_for_single_units

# Args

# Returns

#     # compute

#         # get single units

#         # load WaveformExtractor

#         # create waveformExtractor for single units

#         # setup two properties required to calculate some quality metrics

#         # add pca to waveform extractor


# def get_waveformExtractor_for_single_units

# Args

# Returns

#     # load, else compute



#         # get "good" sorted single-unit ids

#         # load WaveformExtractor

#         # create waveformExtractor for single units

#         # setup two properties required to calculate some quality metrics

#         # add pca to waveform extractor


def mad(data):
    mean_data = np.mean(data)
    return np.mean(np.absolute(data - mean_data))


def get_mad_ratio(spike_amp, noise_amp):
    """calculate an sd_ratio robust to outliers"""
    mad_unit = mad(spike_amp)  # twice smaller than std
    mad_noise = mad(noise_amp)
    return mad_unit / mad_noise


def get_best_site_mad_noise(we, max_chids, unit):

    # get waveforms
    wv, _ = we.get_waveforms(unit_id=unit, with_index=True)

    # get channel ids (sparse)
    c_ids = we.sparsity.unit_id_to_channel_ids[unit]

    # get nearest channel
    max_chid = max_chids[unit]
    max_chid_ix = np.where(c_ids == max_chid)[0][0]
    return wv[:, :, max_chid_ix].flatten()


def get_mad_ratio_all_units(unit_ids, WeNS, spike_amp):
    max_chids = ttools.get_template_extremum_channel(WeNS, peak_sign="both")
    mad_ratio = []
    for unit in unit_ids:
        noise_amp = get_best_site_mad_noise(WeNS, max_chids, unit)
        mad_ratio.append(get_mad_ratio(spike_amp[unit], noise_amp))
    return mad_ratio


def add_spike_amplitude_extension(we, n_sites, load_if_exists: bool, job_kwargs:dict):
    """Add spike amplitudes to WaveformExtractor"""

    # these two properties are required to compute amplitudes
    we.recording.set_property("gain_to_uV", np.ones((n_sites,)))
    we.recording.set_property("offset_to_uV", np.zeros((n_sites,)))

    # compute spike amplitudes
    # or it as an extension
    if not load_if_exists:
        _ = si.postprocessing.compute_spike_amplitudes(we, outputs="by_unit", **job_kwargs)
    else:
        we.load_extension("spike_amplitudes")

    # unit-test
    assert we.has_extension("spike_amplitudes"), "load spike_amplitudes extension"
    return we


def get_quality_metrics(we,
                        load_qm_if_exists=False,
                        skip_pc_metrics=True, 
                        job_kwargs:dict={}):
    """compute cell quality metrics"""
    # add spike amplitudes to waveform extractor
    we = add_spike_amplitude_extension(we, n_sites=we.get_num_channels(), 
                                         load_if_exists=load_qm_if_exists, 
                                         job_kwargs=job_kwargs)

    # compute quality metrics
    qmetrics = qm(
        we, 
        qm_params={
            "amplitude_cutoff": {
                "peak_sign": "neg",
                "num_histogram_bins": 100,
                "histogram_smoothing_value": 3,
                "amplitudes_bins_min_ratio": 0,
            }
        },
        load_if_exists=load_qm_if_exists,
        skip_pc_metrics=skip_pc_metrics,
        **job_kwargs,
    )
    
    # select these metrics
    qmetrics = qmetrics[
        [
            "amplitude_cutoff",
            "firing_range",
            "firing_rate",
            "isi_violations_ratio",
            "presence_ratio",
            "rp_contamination",
            "rp_violations",
            "sd_ratio",
            "snr",
        ]
    ]

    # add quality metric
    assert we.has_extension("principal_components"), """run before compute_principal_components(waveform_extractor=we, n_components=5, mode="by_channel_local")"""

    # - add silhouette metrics
    silhouette = qm(
        we,
        metric_names=["silhouette"],
        skip_pc_metrics=False,
        **job_kwargs,
    )
    qmetrics["silhouette"] = silhouette.values

    # report missing metrics
    print("****************** Analysing data completion ***************")
    print("Data completion:", qmetrics.notna().sum())
    print("quality metrics are:", qmetrics.columns)
    return qmetrics


def get_sorted_single_unit_ids_from_file(quality_path, exp, sorter):

    # load evaluated sorted single units
    unit_quality = pd.read_csv(quality_path)
    df = unit_quality[
        (unit_quality["experiment"] == exp) & (unit_quality["sorter"] == sorter)
    ]
    return df.sorted.values



def engineer_quality_metrics(
    single_unit_ids: np.array, we, load_qm_if_exists=False,
    job_kwargs={}
):
    """Compute metrics of quality for each sorted single-unit"""
    # pre-compute quality metrics
    print("\nComputing spikeinterface quality metrics:")
    qmetrics = get_quality_metrics(we, 
                                   load_qm_if_exists=load_qm_if_exists,
                                   job_kwargs=job_kwargs)

    # pre-compute negative spike amplitudes
    print("\nExtracing spike amplitudes:")
    spike_amp = si.postprocessing.compute_spike_amplitudes(
        we, peak_sign="neg", outputs="by_unit", 
        load_if_exists=load_qm_if_exists, **job_kwargs
    )[0]

    # add mad_ratio
    print("\nCalculating MAD ratios:")
    mad_ratio = get_mad_ratio_all_units(qmetrics.index, we, spike_amp)
    qmetrics["mad_ratio"] = mad_ratio

    # spikeinterface's unit_ids are strings
    qmetrics.index = qmetrics.index.astype(int)

    # sanity check units match
    missing = set(qmetrics.index) - set(single_unit_ids)
    if missing:
        raise ValueError(f"{len(missing)} qmetric ids missing from scores.columns: {sorted(missing)[:10]}")

    # filter qualified single-units (in sorting_quality.py pipeline)
    qmetrics = qmetrics.loc[single_unit_ids, :]

    print("\nQuality metrics used: ")
    print(qmetrics.columns)
    return qmetrics


def get_good_and_bad_units(quality_path: str, exp="NS", sorter="KS4"):
    """filter "good" and "bad" sorted single-units"""
    
    # load table of sorted units categorized based
    # on their biases ("overmerging", "oversplitting"
    # ...)
    unit_quality = pd.read_csv(quality_path)

    # select good units
    df = unit_quality[
        (unit_quality["quality"].str.contains("good"))
        & (unit_quality["experiment"] == exp)
        & (unit_quality["sorter"] == sorter)
    ]
    good_unit_ids = df.sorted.values

    # select bad units
    df = unit_quality[
        (~unit_quality["quality"].str.contains("good"))
        & (unit_quality["experiment"] == exp)
        & (unit_quality["sorter"] == sorter)
    ]
    bad_unit_ids = df.sorted.values
    return good_unit_ids, bad_unit_ids


def format_dataset(qmetric, good_unit_id, bad_unit_id, Sorting, SortingTrue, delta_time=1.3):

    # compute agreement between the spike trains of sorted and ground truth units
    scores = get_scores(SortingTrue, Sorting, delta_time=delta_time)

    # cast as integers to match qmetric.index
    scores.columns = scores.columns.astype(int)

    # sanity check units match
    missing = set(qmetric.index) - set(scores.columns)
    if missing:
        raise ValueError(f"{len(missing)} qmetric ids missing from scores.columns: {sorted(missing)[:10]}")

    scores = scores.loc[:, qmetric.index].max().values

    # build dataset
    dataset = copy.copy(qmetric)

    # curate metrics
    # - presence_ratio is constant
    # - note that firing rate is correlated with firing range so we dropped firing rate
    # - note that sd_ratio and mad_ratio are strongly correlated
    dataset = dataset.drop(columns=["presence_ratio"])

    # selected metrics
    predictive_metrics = dataset.columns

    # add predicted labels
    dataset["quality_label"] = np.nan  # for classification
    dataset.loc[good_unit_id, "quality_label"] = 1
    dataset.loc[bad_unit_id, "quality_label"] = 0

    # unit-test
    assert (
        np.isnan(dataset.quality_label).any() == False
    ), "there should be no np.nan in quality label"

    return dataset, predictive_metrics


def load_dataset(single_unit_id, good_unit_id, bad_unit_id, Sorting,
                 SortingTrue, we, load_qm_if_exists=False, 
                 delta_time=1.3, job_kwargs={}):
    """Engineer the dataset of quality metrics features to train the classifier"""
    # engineer unit quality metrics
    qmetric = engineer_quality_metrics(
        single_unit_id, we, load_qm_if_exists=load_qm_if_exists,
        job_kwargs=job_kwargs
    )
    
    # create a matrix of quality metrics features
    dataset, predictors = format_dataset(qmetric, good_unit_id, 
                                         bad_unit_id, Sorting, SortingTrue, 
                                         delta_time=delta_time)
    
    # filter units without infinite-value feature
    infdata = dataset.index[dataset.sum(axis=1) == np.inf]
    logger.info(f"nb of units before filtering units with inf feature: {len(dataset)}")

    # print number of units after curation of missing data
    dataset = dataset.drop(index=infdata)
    logger.info(f"nb of units after clearning missing data: {len(dataset)}")

    return {"dataset": dataset, 
            "predictors": predictors, 
            "good_unit_id": good_unit_id, 
            "bad_unit_id": bad_unit_id, 
            "qmetric": qmetric, 
            "single_unit_id": single_unit_id
            }