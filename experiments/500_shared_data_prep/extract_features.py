"""
HISTORICAL (500-file experiment) -- not the canonical pipeline. See
scripts/extract_features.py for the canonical (1000-file) version.
Kept for reference/reproducibility of the 500_unweighted and
500_class_weighted experiments only.

Runs BRepNet's feature-extraction pipeline
(external/BRepNet/pipeline/extract_brepnet_data_from_step.py) on the
500-file verified MFCAD subset (mfcad_subset.txt in this same folder),
producing one .npz per file with per-face/per-edge/per-coedge UV-grid
features in outputs/brepnet_features/ (bulk cache dir, unmoved).

The extraction pipeline globs **/*.step under whatever directory it's
given, so we first copy just the subset's STEP files into an isolated
folder (outputs/brepnet_subset_step/) rather than pointing it at the
full 15,488-file dataset directory.

Verified separately: the npz face order here matches occwl's raw
traversal order exactly, which is the same order
src/data/mfcad_loader.py's LabeledSample.labels is aligned to -- so no
extra reordering is needed between the two.

Run with the `brepnet` conda environment's python (needs occwl/OCC):
    <path to conda>/envs/brepnet/python.exe experiments/500_shared_data_prep/extract_features.py
"""

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BREPNET_DIR = PROJECT_ROOT / "external" / "BRepNet"
sys.path.insert(0, str(BREPNET_DIR))

STEP_DIR = PROJECT_ROOT / "data" / "MFCAD" / "dataset" / "step"
SUBSET_LIST = Path(__file__).resolve().parent / "mfcad_subset.txt"
ISOLATED_STEP_DIR = PROJECT_ROOT / "outputs" / "brepnet_subset_step"
FEATURES_OUT_DIR = PROJECT_ROOT / "outputs" / "brepnet_features"
FEATURE_LIST_PATH = BREPNET_DIR / "feature_lists" / "all.json"


def main():
    from pipeline.extract_brepnet_data_from_step import extract_brepnet_data_from_step

    stems = SUBSET_LIST.read_text().split()
    print(f"Subset has {len(stems)} files")

    ISOLATED_STEP_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        src = STEP_DIR / f"{stem}.step"
        dst = ISOLATED_STEP_DIR / f"{stem}.step"
        if not dst.exists():
            shutil.copy(src, dst)

    extract_brepnet_data_from_step(
        ISOLATED_STEP_DIR,
        FEATURES_OUT_DIR,
        feature_list_path=FEATURE_LIST_PATH,
        force_regeneration=False,
    )

    npz_files = sorted(FEATURES_OUT_DIR.glob("*.npz"))
    print(f"\nProduced {len(npz_files)} / {len(stems)} npz files")
    missing = sorted(set(stems) - {f.stem for f in npz_files})
    if missing:
        print(f"WARNING: {len(missing)} files failed extraction: {missing}")


if __name__ == "__main__":
    main()
