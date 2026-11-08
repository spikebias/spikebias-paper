"""Replicate Buccino's synthetic model with MEArec

author: laquitainesteeve@gmail.com
"""

import argparse
import json
import os
import shutil
import time
import warnings

import MEArec as mr
import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se

warnings.filterwarnings("ignore")

PARAMS_PATH = "assets/mearec/buccino_rep/input_params.json"
TEMPLATES_PATH = "temp/pros_of_details/templates.h5"
RECORDING_PATH = "dataset/00_raw/recording_buccino_rep"
H5_PATH = "temp/synthetic/buccino_rep_recording.h5"
SAVE_PARAMS = {"n_jobs": 20, "verbose": True, "progress_bar": True, "overwrite": True, "chunk_size": 50000}


def standardize_site_location_property(Recording, zero_coord=0):
    """3D locations of the sites"""
    site_pos = np.insert(Recording.get_channel_locations(), zero_coord, 0, axis=1)
    Recording.set_property("location3D", site_pos)
    return Recording


def set_metadata_to_mearec_simulations(recording_generator, Recording, Sorting):
    """Cell features in the ground truth, layer of the sites in the recording"""
    features = ["cell_type", "snr", "mtype", "soma_position", "bursting", "exp_decay", "max_spikes_per_burst", "max_burst_duration"]
    for feature in features:
        Sorting.set_property(feature, np.array([st.annotations[feature] for st in recording_generator.spiketrains]))
    Recording.set_property("layer", np.array(["L5"] * 384))
    return Recording, Sorting


def main(output, duration):
    t0 = time.time()
    rec_prms = json.load(open(PARAMS_PATH))
    if duration is not None:
        rec_prms["spiketrains"]["duration"] = duration
    os.makedirs(os.path.dirname(H5_PATH), exist_ok=True)

    # simulate
    rec_gen = mr.gen_recordings(params=rec_prms, templates=TEMPLATES_PATH)
    mr.save_recording_generator(rec_gen, H5_PATH)
    Recording = se.MEArecRecordingExtractor(H5_PATH)
    Recording.set_property("layers", np.array(["L5"] * 384))
    Sorting = se.MEArecSortingExtractor(H5_PATH)
    os.remove(H5_PATH)

    # metadata and saving
    Recording, Sorting = set_metadata_to_mearec_simulations(rec_gen, Recording, Sorting)
    Recording = standardize_site_location_property(Recording)
    shutil.rmtree(output, ignore_errors=True)
    Recording.save(folder=output, **SAVE_PARAMS)
    print(f"saved {output} ({si.load_extractor(output)}) in {time.time() - t0:.0f} s")
    print("properties:", si.load_extractor(output).get_property_keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=RECORDING_PATH, help="folder of the recording to create")
    parser.add_argument("--duration", type=float, default=None, help="duration in seconds (default: that of the parameter file, 600 s)")
    args = parser.parse_args()
    main(args.output, args.duration)
