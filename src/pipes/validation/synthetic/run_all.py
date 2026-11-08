"""Reproduce the synthetic recordings S2-S5, their features and their Kilosort4 sortings

author: laquitainesteeve@gmail.com
"""

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = "src/pipes/validation/synthetic"
MEAREC = "temp/mearec_1_8_0"
SI_ENV = "envs/mearec_si_0_103_0"
N_JOBS = "8"


def run(cmd, **env):
    subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ, **env})


def install():
    if (ROOT / MEAREC / "MEArec").exists():
        return print(f"MEArec 1.8.0 already in {MEAREC}")
    run([f"{SI_ENV}/bin/pip", "install", "--no-deps", "--target", MEAREC, "MEArec==1.8.0"])
    run(["patch", "-p1", "-d", MEAREC, "-i", str(ROOT / HERE / "mearec_1_8_0_spike_traces.patch")])


def simulate():
    for s in ("s3", "s4", "s5", "s2"):
        if (ROOT / f"dataset/00_raw/recording_synth{s[1:]}").exists():
            print(f"{s} already simulated")
            continue
        run([f"{SI_ENV}/bin/python", f"{HERE}/simulate_mearec.py", "--name", s, "--n-jobs", N_JOBS],
            MEAREC_SKIP_SPIKE_TRACES="1", PYTHONPATH=f"{ROOT}:{ROOT / MEAREC}")


def buccino_rep():
    if (ROOT / "dataset/00_raw/recording_buccino_rep").exists():
        return print("buccino_rep already simulated")
    run([f"{SI_ENV}/bin/python", f"{HERE}/simulate_buccino_rep.py"], PYTHONPATH=str(ROOT))


def features():
    for s in ("ns", "buccino", "s2", "s3", "s4", "s5"):
        if (ROOT / f"dataset/01_intermediate/supplementary/synthetic_features/{s}.npz").exists():
            print(f"{s} features exist")
            continue
        run([f"{SI_ENV}/bin/python", f"{HERE}/compute_features.py", "--name", s, "--n-jobs", N_JOBS], PYTHONPATH=str(ROOT))


def sort():
    for s in ("s2", "s3", "s4", "s5"):
        if (ROOT / f"dataset/01_intermediate/sorting/synth{s[1:]}/SortingKS4_10m").exists():
            print(f"{s} sorted")
            continue
        run(["envs/spikebias/bin/python", f"{HERE}/sort_ks4.py", "--name", s], PYTHONPATH=str(ROOT))


STEPS = {"install": install, "simulate": simulate, "buccino_rep": buccino_rep, "features": features, "sort": sort}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=[*STEPS, "all"], nargs="?", default="all")
    args = parser.parse_args()
    for name, step in STEPS.items():
        if args.step in (name, "all"):
            step()
