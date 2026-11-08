"""Write the datasets as NWB files and upload them to the draft of DANDI dandiset 001250

author: laquitainesteeve@gmail.com

Usage (environment envs/demo: neuroconv, spikeinterface and dandi; the API key is read from the environment):

    cd /home/steeve/steeve/epfl/code/spikebias-paper
    DANDI_API_KEY=<key> PYTHONPATH=$PWD envs/demo/bin/python src/pipes/dandi/upload/revision.py
"""

import html
import json
import os
import threading
import time
import traceback
import uuid
from datetime import datetime

from dateutil.tz import tzlocal

STAGING = "/home/steeve/steeve/epfl/dandi_upload"
DANDISET_DIR = os.path.join(STAGING, "001250")
STATUS_PATH = os.path.join(STAGING, "status.json")
PAGE_PATH = os.path.join(STAGING, "progress.html")
LOG_PATH = os.path.join(STAGING, "upload.log")
SORTING = "dataset/01_intermediate/sorting"
SORTERS = {"KS4": ("ks4", "Kilosort 4"), "KS3": ("ks3", "Kilosort 3"), "KS2_5": ("ks25", "Kilosort 2.5"), "KS2": ("ks2", "Kilosort 2"),
           "KS": ("ks", "Kilosort"), "HS": ("hs", "Herdingspikes")}
RECORDING_BYTES = 384 * 32000 * 600 * 4      # float32 traces of a MEArec model


def sorting_items():
    """(name, kind, sources, subject, session, description, dandi path) of the small files"""
    items = []
    evoked = ("Biophysical simulation of neuropixels in the evoked regime, sampled at 40 kHz. Background noise and gain are fitted to "
              "Marques-Smith dataset. {} sorting of the first 10 minutes.")
    dense = ("Biophysical simulation of dense probe at depth 1 in the spontaneous regime. Background noise and gain are fitted to Horvath "
             "depth 1 dataset. {} sorting of the first 10 minutes.")
    buccino = ("Synthetic MEArec recording of Buccino et al. (2020), dandiset 000034 (sub-MEAREC-250neuron-Neuropixels). {} sorting of the "
               "first 10 minutes.")
    for folder, subject, text in (("npx_evoked_40khz", "npx-evoked-sf-40Hz-fitted", evoked), ("dense_probe1", "003-fitted", dense),
                                  ("buccino", "mearec-synthetics-buccino", buccino)):
        for key, (short, sorter) in SORTERS.items():
            items.append((f"{subject} {sorter}", "sorting", [f"{SORTING}/{folder}/Sorting{key}_10m"], subject, f"{short}_10m",
                          text.format(sorter), f"sub-{subject}/sub-{subject}_ses-{short}-10m.nwb"))
    for n in (2, 3, 4, 5):
        subject = f"mearec-synthetics-s{n}"
        items.append((f"{subject} Kilosort 4", "sorting", [f"{SORTING}/synth{n}/SortingKS4_10m"], subject, "ks4_10m",
                      f"MEArec synthetic model S{n} (Supplementary Fig. 6c-e). Kilosort 4 sorting of the first 10 minutes.",
                      f"sub-{subject}/sub-{subject}_ses-ks4-10m.nwb"))
    spont = "Biophysical simulation of neuropixels in the spontaneous regime. Background noise and gain are fitted to Marques-Smith dataset. "
    items += [
        ("001-fitted Kilosort 4 (V100)", "sorting", [f"{SORTING}/npx_spont_v100/SortingKS4_10m"], "001-fitted", "ks4_v100_10m",
         spont + "Kilosort 4 sorting of the first 10 minutes run on a Tesla V100 GPU (the sorting of the original study).",
         "sub-001-fitted/sub-001-fitted_ses-ks4-v100-10m.nwb"),
        ("001-fitted ground truth (V100)", "sorting", ["dataset/00_raw/ground_truth_npx_spont_v100_10m"], "001-fitted", "ground_truth_v100_10m",
         spont + "Ground truth spike trains of the first 10 minutes (1,421 neurons), the version of the ground truth used with the Tesla V100 "
         "Kilosort 4 sorting.", "sub-001-fitted/sub-001-fitted_ses-ground-truth-v100-10m.nwb"),
        ("vivo-marques-smith Kilosort 4 (RTX 5090)", "sorting", [f"{SORTING}/marques/SortingKS4_10m_rtx5090"], "vivo-marques-smith",
         "ks4_rtx5090_10m", "Marques-Smith neuropixels in the spontaneous regime. First 10 minutes of the Kilosort 4 sorting of the whole "
         "recording, run on an RTX 5090 GPU.", "sub-vivo-marques-smith/sub-vivo-marques-smith_ses-ks4-rtx5090-10m.nwb"),
        ("002-fitted threshold crossing", "sorting", [f"{SORTING}/npx_evoked/SortingTC"], "002-fitted", "tc",
         "Biophysical simulation of neuropixels in the evoked regime. Background noise and gain are fitted to Marques-Smith dataset. "
         "Multi-unit threshold crossing on each recording site (no spike sorting), whole recording.", "sub-002-fitted/sub-002-fitted_ses-tc.nwb"),
    ]
    return items


def recording_items():
    """the large MEArec recordings with their ground truth"""
    text = ("MEArec synthetic model S{n} (Supplementary Fig. 6c-e), one of the models S2-S5 constrained in turn to match the cell-to-electrode "
            "distances, cell distances, firing rates, firing ranges and SNR of the biophysical model: Neuropixels sites in layer 5, 32 kHz, "
            "600 s. Recording and ground truth.")
    return [(f"mearec-synthetics-s{n} recording", "recording", [f"dataset/00_raw/recording_synth{n}", f"dataset/00_raw/ground_truth_synth{n}"],
             f"mearec-synthetics-s{n}", f"s{n}", text.format(n=n), f"sub-mearec-synthetics-s{n}/sub-mearec-synthetics-s{n}_ecephys.nwb")
            for n in (3, 4, 5)]


class Progress:
    """status of each dataset, saved as json and as a self-refreshing html page"""

    STAGES = ["queued", "writing NWB", "validating", "computing checksum", "uploading", "registering asset", "done"]

    def __init__(self, items):
        self.lock = threading.Lock()
        self.started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.rows = [{"name": it[0], "path": it[6], "stage": "queued", "percent": 0.0, "bytes": None, "uploaded": 0, "note": "",
                      "updated": ""} for it in items]

    def set(self, i, **kw):
        with self.lock:
            self.rows[i].update(kw, updated=datetime.now().strftime("%H:%M:%S"))
            self.save()

    def save(self):
        json.dump({"started": self.started, "rows": self.rows}, open(STATUS_PATH, "w"), indent=1)
        open(PAGE_PATH, "w").write(self.page())

    def page(self):
        done = sum(r["stage"] == "done" for r in self.rows)
        failed = sum(r["stage"] == "failed" for r in self.rows)
        total_bytes = sum(r["bytes"] or 0 for r in self.rows)
        sent = sum(r["uploaded"] for r in self.rows)
        overall = 100 * done / len(self.rows)
        rows = []
        for r in self.rows:
            pct = 100.0 if r["stage"] == "done" else r["percent"]
            cls = {"done": "ok", "failed": "bad", "queued": "idle"}.get(r["stage"], "run")
            size = f"{r['bytes'] / 1e9:.2f} GB" if r["bytes"] and r["bytes"] > 1e8 else (f"{r['bytes'] / 1e6:.1f} MB" if r["bytes"] else "")
            rows.append(f"<tr class='{cls}'><td><b>{html.escape(r['name'])}</b><div class='path'>{html.escape(r['path'])}</div></td>"
                        f"<td>{size}</td><td>{html.escape(r['stage'])}</td>"
                        f"<td><div class='bar'><div class='fill' style='width:{pct:.1f}%'></div><span>{pct:.0f}%</span></div></td>"
                        f"<td class='note'>{html.escape(r['note'])}</td><td>{r['updated']}</td></tr>")
        log = ""
        if os.path.exists(LOG_PATH):
            log = html.escape("".join(open(LOG_PATH).readlines()[-12:]))
        state = "finished" if done + failed == len(self.rows) else "running"
        return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="5"><title>DANDI upload progress</title><style>
:root {{ --bg:#fafafa; --fg:#1d1d1f; --muted:#6e6e73; --line:#e5e5ea; --track:#e8e8ed; --run:#0a84ff; --ok:#30a46c; --bad:#e5484d; --card:#fff; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111113; --fg:#ececf0; --muted:#9a9aa2; --line:#2a2a2e; --track:#2a2a2e; --card:#1a1a1d; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; margin:0; padding:24px 16px; }}
main {{ max-width:1100px; margin:0 auto; }} h1 {{ font-size:20px; margin:0 0 4px; }} .sub {{ color:var(--muted); margin-bottom:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:16px; }}
.big .bar {{ height:22px; }} table {{ width:100%; border-collapse:collapse; }} th, td {{ text-align:left; padding:8px 6px; border-bottom:1px solid var(--line); vertical-align:middle; }}
th {{ color:var(--muted); font-weight:600; font-size:12px; }} .path {{ color:var(--muted); font-size:11px; word-break:break-all; }}
.bar {{ position:relative; height:16px; background:var(--track); border-radius:8px; overflow:hidden; min-width:160px; }}
.fill {{ height:100%; background:var(--run); transition:width .5s; }} tr.ok .fill {{ background:var(--ok); }} tr.bad .fill {{ background:var(--bad); width:100% !important; }}
.bar span {{ position:absolute; inset:0; text-align:center; font-size:11px; line-height:16px; color:var(--fg); }}
.big .bar span {{ line-height:22px; font-size:12px; }} .big .fill {{ background:var(--ok); }}
td.note {{ color:var(--muted); font-size:12px; max-width:260px; }} pre {{ white-space:pre-wrap; font-size:11px; color:var(--muted); margin:0; }}
.table-wrap {{ overflow-x:auto; }}
</style></head><body><main>
<h1>Upload to DANDI dandiset 001250 (draft)</h1>
<div class="sub">Started {self.started} · status: <b>{state}</b> · {done} of {len(self.rows)} done{f", {failed} failed" if failed else ""} ·
{sent / 1e9:.2f} of {total_bytes / 1e9:.2f} GB sent · page refreshes every 5 s · last update {datetime.now().strftime("%H:%M:%S")}</div>
<div class="card big"><div class="bar"><div class="fill" style="width:{overall:.1f}%"></div><span>{done} / {len(self.rows)} datasets</span></div></div>
<div class="card table-wrap"><table><thead><tr><th>Dataset</th><th>Size</th><th>Stage</th><th>Progress of the stage</th><th>Note</th><th>Updated</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<div class="card"><pre>{log}</pre></div></main></body></html>"""


def log(msg):
    with open(LOG_PATH, "a") as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')} {msg}\n")


def write_nwb(kind, sources, subject, session, description, out_path):
    import spikeinterface as si
    from neuroconv.tools.spikeinterface import add_sorting
    from pynwb import NWBHDF5IO
    from src.nodes.dandi import write_nwb as wn

    params = dict(subject_id=subject, session_id=session, session_description=description, experiment_description=description,
                  identifier=str(uuid.uuid4()), session_start_time=datetime.now(tzlocal()), related_publications=None)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if kind == "recording":
        wn.run(sources[0], sources[1], out_path, params)
        return
    nwbfile = wn.init_nwbfile(**params)
    add_sorting(sorting=si.load_extractor(sources[0]), nwbfile=nwbfile)
    with NWBHDF5IO(path=out_path, mode="w") as io:
        io.write(nwbfile)


def watch_size(progress, i, path, expected, stop):
    while not stop.is_set():
        if os.path.exists(path):
            progress.set(i, percent=min(99.0, 100 * os.path.getsize(path) / expected))
        stop.wait(5)


def patch_session_duration():
    """dandi computes the session duration from the first and last spike of each unit by indexing the spike times with the units' index"""
    import hdmf
    import numpy as np
    import pynwb
    import dandi.pynwb_utils as pu

    original = pu._get_session_duration

    def duration(nwb):
        try:
            return original(nwb)
        except TypeError:
            starts, ends = [], []
            for obj in nwb.objects.values():
                if isinstance(obj, pynwb.base.TimeSeries):
                    if obj.timestamps is not None and len(obj.timestamps) > 0:
                        starts.append(float(obj.timestamps[0]))
                        ends.append(float(obj.timestamps[-1]))
                    elif obj.starting_time is not None and obj.rate and obj.data is not None:
                        starts.append(float(obj.starting_time))
                        ends.append(float(obj.starting_time + len(obj.data) / obj.rate))
                elif isinstance(obj, hdmf.common.DynamicTable) and "spike_times" in obj.colnames:
                    spikes = np.asarray(obj["spike_times"].target.data[:])
                    if spikes.size:
                        starts.append(float(spikes.min()))
                        ends.append(float(spikes.max()))
            return max(ends) - min(starts) if starts and ends else None

    pu._get_session_duration = duration


def upload(progress, i, dandiset, out_path):
    from dandi.dandiapi import NotFoundError
    from dandi.files import dandi_file
    from dandi.validate_types import Severity

    progress.set(i, stage="validating", percent=0.0, bytes=os.path.getsize(out_path))
    dfile = dandi_file(out_path, DANDISET_DIR)
    errors = [v for v in dfile.get_validation_errors() if v.severity is not None and v.severity >= Severity.ERROR]
    if errors:
        raise RuntimeError("validation: " + "; ".join(str(getattr(e, "message", e)) for e in errors)[:400])
    progress.set(i, stage="computing checksum", percent=0.0)
    digest = dfile.get_digest()
    metadata = dfile.get_metadata(digest=digest, ignore_errors=False).model_dump(mode="json", exclude_none=True)
    try:
        replacing = dandiset.get_asset_by_path(dfile.path)
    except NotFoundError:
        replacing = None
    for r in dfile.iter_upload(dandiset, metadata, jobs=5, replacing=replacing):
        if r["status"] == "uploading":
            progress.set(i, stage="uploading", percent=r["progress"], uploaded=r["current"])
        elif r["status"] == "producing asset":
            progress.set(i, stage="registering asset", percent=100.0, uploaded=os.path.getsize(out_path))
        elif r["status"] == "done":
            progress.set(i, stage="done", percent=100.0, uploaded=os.path.getsize(out_path),
                         note=("replaced the earlier file" if replacing is not None else "new file"))
        else:
            progress.set(i, stage=r["status"], percent=0.0)


def main():
    from dandi.dandiapi import DandiAPIClient

    os.makedirs(STAGING, exist_ok=True)
    patch_session_duration()
    items = sorting_items() + recording_items()
    progress = Progress(items)
    progress.save()
    log(f"{len(items)} datasets to write and upload")
    client = DandiAPIClient.for_dandi_instance("dandi", token=os.environ["DANDI_API_KEY"])
    dandiset = client.get_dandiset("001250", "draft")
    for i, (name, kind, sources, subject, session, description, path) in enumerate(items):
        out_path = os.path.join(DANDISET_DIR, path)
        try:
            log(f"{name}: writing {path}")
            progress.set(i, stage="writing NWB", percent=0.0)
            stop = threading.Event()
            if kind == "recording":
                threading.Thread(target=watch_size, args=(progress, i, out_path, RECORDING_BYTES, stop), daemon=True).start()
            t0 = time.time()
            write_nwb(kind, sources, subject, session, description, out_path)
            stop.set()
            log(f"{name}: written in {time.time() - t0:.0f} s ({os.path.getsize(out_path) / 1e6:.1f} MB); uploading")
            upload(progress, i, dandiset, out_path)
            log(f"{name}: uploaded")
            if kind == "recording":
                os.remove(out_path)
                log(f"{name}: local copy deleted")
        except Exception as e:
            stop.set()
            progress.set(i, stage="failed", note=f"{type(e).__name__}: {e}"[:400])
            log(f"{name}: FAILED {type(e).__name__}: {e}")
            log(traceback.format_exc().splitlines()[-1])
    log("finished")
    progress.save()


if __name__ == "__main__":
    main()
