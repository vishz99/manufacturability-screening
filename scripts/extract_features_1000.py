"""
Same as scripts/extract_features.py, scaled to the 1000-file subset
(outputs/mfcad_subset_1000.txt), writing to separate output directories
so the existing 500-file extraction stays intact for comparison.

Run with the `brepnet` conda environment's python (needs occwl/OCC):
    <path to conda>/envs/brepnet/python.exe scripts/extract_features_1000.py
"""

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BREPNET_DIR = PROJECT_ROOT / "external" / "BRepNet"
sys.path.insert(0, str(BREPNET_DIR))

STEP_DIR = PROJECT_ROOT / "data" / "MFCAD" / "dataset" / "step"
SUBSET_LIST = PROJECT_ROOT / "outputs" / "mfcad_subset_1000.txt"
ISOLATED_STEP_DIR = PROJECT_ROOT / "outputs" / "brepnet_subset_step_1000"
FEATURES_OUT_DIR = PROJECT_ROOT / "outputs" / "brepnet_features_1000"
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
