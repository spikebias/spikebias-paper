"""Generate the cell templates used by the MEArec simulations S2-S5

author: laquitainesteeve@gmail.com

Usage (about 75 min, 13 GB, 32 cores):

    PYTHONPATH=$PWD envs/mearec_si_0_103_0/bin/python src/pipes/validation/synthetic/generate_templates.py         --output dataset/00_raw/mearec_templates/templates.h5
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from importlib.metadata import version

import MEArec as mr

PARAMS_PATH = "assets/mearec/global_templates.json"
MANIFEST_PATH = "assets/mearec/cell_models_sha256.json"
EXPECTED_VERSIONS = {"MEArec": "1.9.0", "NEURON": "8.2.4", "LFPy": "2.3.5", "LFPykit": "0.5.1", "numpy": "1.26.4", "scipy": "1.13.1", "h5py": "3.14.0"}
MODEL_SUFFIXES = {".asc", ".mod", ".hoc", ".json", ".dat", ".tsv", ".txt"}
MODEL_NAMES = {"VERSION", "README", "LICENSE", "CHANGELOG"}


def model_files(cells: Path) -> dict:
    """sha256 of the files that define the cell models"""
    files = sorted(
        p for p in cells.rglob("*")
        if p.is_file() and "x86_64" not in p.parts and not p.name.startswith(("run", "."))
        and (p.suffix in MODEL_SUFFIXES or p.name in MODEL_NAMES)
    )
    return {str(p.relative_to(cells)): sha256(p) for p in files}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="path of the templates .h5 file to create")
    parser.add_argument("--n", type=int, default=None, help="templates per cell model (default: the 1,500 of the paper)")
    parser.add_argument("--n-jobs", type=int, default=None, help="parallel jobs (default: all cores)")
    parser.add_argument("--write-manifest", action="store_true", help="write the sha256 manifest of the current cell models and exit")
    parser.add_argument("--skip-check", action="store_true", help="do not check the cell models against the manifest")
    args = parser.parse_args()

    params = json.load(open(PARAMS_PATH))
    if args.n is not None:
        params["n"] = args.n
    cells = Path(mr.get_default_cell_models_folder())
    if args.write_manifest:
        json.dump(model_files(cells), open(MANIFEST_PATH, "w"), indent=1)
        raise SystemExit(f"wrote {MANIFEST_PATH}")
    versions = {name: version(name) for name in EXPECTED_VERSIONS}
    if not args.skip_check:
        wrong = {k: (versions[k], v) for k, v in EXPECTED_VERSIONS.items() if versions[k] != v}
        if wrong:
            raise SystemExit(f"package versions differ from those of the paper (found, expected): {wrong}")
        expected, found = json.load(open(MANIFEST_PATH)), model_files(cells)
        diff = sorted(k for k in expected.keys() | found.keys() if expected.get(k) != found.get(k))
        if diff:
            raise SystemExit(f"the cell models in {cells} differ from the models of the paper in {len(diff)} files, e.g. {diff[:3]}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # MEArec writes and removes a file "tmp_params_path.yaml" in the working directory
    out = out.resolve()
    cells = cells.resolve()
    work_dir = tempfile.mkdtemp(prefix="gen_templates_")
    os.chdir(work_dir)
    t0 = time.time()
    tpl_gen = mr.gen_templates(cell_models_folder=str(cells), params=params, n_jobs=args.n_jobs, parallel=True)
    mr.save_template_generator(tpl_gen, str(out))

    provenance = {
        "versions": versions,
        "params": params,
        "cell_models_folder": str(cells),
        "cell_models_sha256": model_files(cells),
        "templates_sha256": sha256(out),
        "duration_s": round(time.time() - t0),
    }
    json.dump(provenance, open(str(out) + ".json", "w"), indent=2)
    os.chdir(out.parent)
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"saved {out} in {provenance['duration_s']} s")
