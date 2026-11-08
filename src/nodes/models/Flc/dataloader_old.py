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


def get_waveformExtractor_for_single_units(
    sort_path: str,
    study_path,
    save_path: str,
    n_sites=384,
    load_if_exists: bool = False,
    add_pca: bool = True,
    n_components=5,
):
    """Setup WaveformExtractors to calculate quality metrics for single units"""

    # compute
    if not load_if_exists:

        # get single units
        Sorting = si.load_extractor(sort_path)
        su_ix = np.where(Sorting.get_property("KSLabel") == "good")[0]
        su_unit_ids = Sorting.unit_ids[su_ix]

        # load WaveformExtractor
        We = si.WaveformExtractor.load_from_folder(study_path)

        # create waveformExtractor for single units
        shutil.rmtree(save_path, ignore_errors=True)
        WeSu = We.select_units(unit_ids=su_unit_ids, new_folder=save_path)

        # setup two properties required to calculate some quality metrics
        WeSu.recording.set_property("gain_to_uV", np.ones((n_sites,)))
        WeSu.recording.set_property("offset_to_uV", np.zeros((n_sites,)))

        # add pca to waveform extractor
        if add_pca:
            _ = compute_principal_components(
                waveform_extractor=WeSu,
                n_components=n_components,
                mode="by_channel_local",
                **job_kwargs,
            )
    else:
        # or load existing
        WeSu = si.WaveformExtractor.load_from_folder(save_path)
    return WeSu


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

    # get channel ids
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
    if not load_if_exists:
        _ = si.postprocessing.compute_spike_amplitudes(we, outputs="by_unit", **job_kwargs)
    else:
        we.load_extension("spike_amplitudes")

    # unit-test
    assert we.has_extension("spike_amplitudes"), "load spike_amplitudes extension"
    return we


def get_quality_metrics(sorting_path: str, 
                        study_path_all: str, 
                        study_path_su: str, 
                        n_sites=384, 
                        load_qm_if_exists=False,
                        load_we_if_exists=True,
                        add_pca=False,
                        skip_pc_metrics=True, 
                        job_kwargs:dict={}):
    """_summary_"""
    # setup waveform extractor to compute unit quality metrics
    WeNs = get_waveformExtractor_for_single_units(
        sorting_path, study_path_all, study_path_su, n_sites=n_sites, 
        load_if_exists=load_we_if_exists, add_pca=add_pca
    )

    # add spike amplitudes to waveform extractor
    WeNs = add_spike_amplitude_extension(WeNs, n_sites=n_sites, 
                                         load_if_exists=load_qm_if_exists, 
                                         job_kwargs=job_kwargs)

    # compute quality metrics
    qmetrics = qm(
        WeNs,
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

    # add silhouette metric
    silhouette = qm(
        WeNs,
        metric_names=["silhouette"],
        skip_pc_metrics=False,
        **job_kwargs,
    )
    qmetrics["silhouette"] = silhouette.values

    # handle missing metrics
    print("****************** Analysing data completion ***************")
    print("Data completion:", qmetrics.notna().sum())
    print("quality metrics are:", qmetrics.columns)
    return qmetrics


def get_evaluated_sorted_single_unit_ids(quality_path, exp, sorter):

    # load evaluated sorted single units
    unit_quality = pd.read_csv(quality_path)
    df = unit_quality[
        (unit_quality["experiment"] == exp) & (unit_quality["sorter"] == sorter)
    ]
    return df.sorted.values


def get_evaluated_sorted_single_units_quality_metrics(
    single_unit_ids:np.array, sorting_path:str, 
    study_all_path:str, study_su_path:str, 
    load_we_if_exists=True, add_pca=False, 
    load_qm_if_exists=False,
    job_kwargs={}
):
    """Compute metrics of quality for each sorted single-unit"""
    # get WaveformExtractor
    We = si.WaveformExtractor.load_from_folder(study_su_path)
    print("DEBUGGING We.unit_id", We.unit_ids)

    # pre-compute quality metrics
    print("\nComputing spikeinterface quality metrics:")
    qmetrics = get_quality_metrics(sorting_path, 
                                   study_all_path, 
                                   study_su_path, 
                                   We.get_num_channels(), 
                                   load_we_if_exists=load_we_if_exists, 
                                   load_qm_if_exists=load_qm_if_exists,
                                   add_pca=add_pca, 
                                   job_kwargs=job_kwargs)
    print("DEBUGGING qmetrics", qmetrics)

    # pre-compute negative spike amplitudes
    print("\nExtracing spike amplitudes:")
    spike_amp = si.postprocessing.compute_spike_amplitudes(
        We, peak_sign="neg", outputs="by_unit", 
        load_if_exists=load_qm_if_exists, **job_kwargs
    )[0]

    # add mad_ratio
    print("\nCalculating MAD ratios:")
    mad_ratio = get_mad_ratio_all_units(qmetrics.index, We, spike_amp)
    qmetrics["mad_ratio"] = mad_ratio

    # filter qualified single-units
    qmetrics = qmetrics.loc[single_unit_ids, :]
    return qmetrics


def get_good_and_bad_units(quality_path: str, exp="NS", sorter="KS4"):
    """_summary_"""
    
    # load table of sorted units categorized based
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


def format_dataset(qmetric, good_unit_id, bad_unit_id, sorting_path, GT_ns_10m):

    # predicted scores
    Sorting = si.load_extractor(sorting_path)
    SortingTrue = si.load_extractor(GT_ns_10m)
    scores = get_scores(SortingTrue, Sorting, 1.3)
    scores = scores.loc[:, qmetric.index].max().values

    # build dataset
    dataset = copy.copy(qmetric)

    # curate metrics
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


def load_dataset(quality_path, exp, sorter, sorting_path, STUDY_ns, STUDY_ns_su, GT_ns_10m, 
                load_we_if_exists=False, add_pca=False, load_qm_if_exists=False, job_kwargs={}):

    # get evaluated sorted single-units
    single_unit_id = get_evaluated_sorted_single_unit_ids(
        quality_path, exp, sorter
    )
    print("DEBUGGING: single_unit_id", single_unit_id)
    print("DEBUGGING: len(single_unit_id)", len(single_unit_id))
    
    # get unit quality metrics
    qmetric = get_evaluated_sorted_single_units_quality_metrics(
        single_unit_id, sorting_path, STUDY_ns, STUDY_ns_su, 
        load_we_if_exists=load_we_if_exists,
        add_pca=add_pca,
        load_qm_if_exists=load_qm_if_exists, 
        job_kwargs=job_kwargs
    )
    
    # label good and bad units
    good_unit_id, bad_unit_id = get_good_and_bad_units(quality_path, exp=exp, sorter=sorter)
    
    # make dataset table and predictive metrics
    dataset, predictors = format_dataset(qmetric, good_unit_id, bad_unit_id, sorting_path, GT_ns_10m)
    
    # curate units
    logger.info("CURATION ----------------------------")
    
    # filter units without infinite-value feature
    infdata = dataset.index[dataset.sum(axis=1) == np.inf]
    logger.info(f"nb of units before filtering units with inf feature: {len(dataset)}")
    dataset = dataset.drop(index=infdata)
    logger.info(f"nb of units after: {len(dataset)}")
    return {"dataset": dataset, "predictors": predictors, "good_unit_id": good_unit_id, "bad_unit_id": bad_unit_id, "qmetric": qmetric, "single_unit_id": single_unit_id}