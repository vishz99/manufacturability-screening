"""
Phase 2, step 2 deliverable: verifies what face order
external/BRepNet/pipeline/extract_brepnet_data_from_step.py writes its
per-face npz arrays in, and confirms it matches occwl's raw compound
traversal order -- the same order src/data/mfcad_loader.py's
LabeledSample.labels is aligned to (see mfcad_loader.py's module
docstring for why that's the correct order to align to, established in
Phase 1).

Two separate orderings could plausibly have applied here:
  (a) occwl's raw traversal order (Compound.faces())
  (b) the embedded STEP entity name order (what mfcad_loader.py uses
      internally before re-aligning back to occwl's raw order)
This script checks the extraction pipeline's npz output against (a).

Method: normalized face-area fingerprints. The extraction pipeline
rescales each body into a [-1,1]^3 box before computing features, so
absolute face areas differ from the original geometry's areas -- but
area_i / sum(areas) is exactly preserved under uniform scaling, giving
an unambiguous, scale-invariant per-face signature to compare position
by position between the npz output and a fresh occwl load of the
original (unscaled) STEP file.

Checks:
  1. The hand-checked file from Phase 1 (0-0-0-0-0-23.step), freshly
     extracted here since it isn't part of the 500-file training subset.
  2. A random sample of 30 files from the verified subset
     (outputs/mfcad_subset.txt), reusing their already-extracted npz in
     outputs/brepnet_features/ (produced by scripts/extract_features.py).

Run with the `brepnet` conda environment's python (needs occwl/OCC):
    <path to conda>/envs/brepnet/python.exe scripts/verify_extraction_face_order.py
"""

import random
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BREPNET_DIR = PROJECT_ROOT / "external" / "BRepNet"
sys.path.insert(0, str(BREPNET_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

STEP_DIR = PROJECT_ROOT / "data" / "MFCAD" / "dataset" / "step"
SUBSET_LIST = PROJECT_ROOT / "experiments" / "500_shared_data_prep" / "mfcad_subset.txt"  # 500-scale list, matches outputs/brepnet_features/
SUBSET_FEATURES_DIR = PROJECT_ROOT / "outputs" / "brepnet_features"
FEATURE_LIST_PATH = BREPNET_DIR / "feature_lists" / "all.json"

HAND_CHECKED_STEM = "0-0-0-0-0-23"
BATCH_SAMPLE_SIZE = 30
BATCH_RANDOM_SEED = 2

SCRATCH_DIR = PROJECT_ROOT / "outputs" / "_face_order_check_scratch"
AREA_FEATURE_COL = 5  # face_features columns: Plane,Cyl,Cone,Sphere,Torus,Area,RationalNurbs


def extract_one_file(stem, isolated_input_dir, output_dir):
    """Isolate a single STEP file into its own dir and extract it, since
    extract_brepnet_data_from_step globs **/*.step under whatever
    directory it's given."""
    from pipeline.extract_brepnet_data_from_step import extract_brepnet_data_from_step

    isolated_input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(STEP_DIR / f"{stem}.step", isolated_input_dir / f"{stem}.step")
    extract_brepnet_data_from_step(
        isolated_input_dir, output_dir,
        feature_list_path=FEATURE_LIST_PATH,
        force_regeneration=True,
    )
    return output_dir / f"{stem}.npz"


def compare_npz_to_occwl_raw_order(stem, npz_path):
    from data.mfcad_loader import load_labeled_sample

    data = np.load(npz_path)
    npz_areas = data["face_features"][:, AREA_FEATURE_COL]

    sample = load_labeled_sample(STEP_DIR / f"{stem}.step")
    raw_areas = np.array([f.area() for f in sample.faces])  # occwl raw traversal order

    if len(npz_areas) != len(raw_areas):
        return {"stem": stem, "status": "COUNT_MISMATCH", "n_npz": len(npz_areas), "n_occwl": len(raw_areas)}

    npz_norm = npz_areas / npz_areas.sum()
    raw_norm = raw_areas / raw_areas.sum()
    positional_diff = float(np.max(np.abs(npz_norm - raw_norm)))

    return {
        "stem": stem,
        "status": "MATCH" if positional_diff < 1e-6 else "POSITIONAL_MISMATCH",
        "n_faces": len(npz_areas),
        "max_positional_diff": positional_diff,
        "npz_areas_norm": npz_norm,
        "raw_areas_norm": raw_norm,
        "labels": sample.labels,
    }


def main():
    print("=" * 70)
    print(f"CHECK 1: hand-checked file {HAND_CHECKED_STEM}")
    print("=" * 70)
    npz_path = extract_one_file(
        HAND_CHECKED_STEM,
        SCRATCH_DIR / "hand_checked_input",
        SCRATCH_DIR / "hand_checked_npz",
    )
    result = compare_npz_to_occwl_raw_order(HAND_CHECKED_STEM, npz_path)
    print(f"npz face count: {result['n_faces']}")
    for i in range(result["n_faces"]):
        print(
            f"  face {i}: npz_normalized_area={result['npz_areas_norm'][i]:.6f}  "
            f"occwl_raw_normalized_area={result['raw_areas_norm'][i]:.6f}  "
            f"label={result['labels'][i]}"
        )
    print(f"max positional diff: {result['max_positional_diff']:.8f}")
    print(f"STATUS: {result['status']}")

    print()
    print("=" * 70)
    print(f"CHECK 2: random sample of {BATCH_SAMPLE_SIZE} files from the verified subset")
    print("=" * 70)
    subset_stems = SUBSET_LIST.read_text().split()
    rng = random.Random(BATCH_RANDOM_SEED)
    sample_stems = rng.sample(subset_stems, BATCH_SAMPLE_SIZE)

    n_match = 0
    n_mismatch = 0
    n_missing_npz = 0
    for stem in sample_stems:
        npz_path = SUBSET_FEATURES_DIR / f"{stem}.npz"
        if not npz_path.exists():
            print(f"  {stem}: SKIPPED (no npz found in {SUBSET_FEATURES_DIR}, run scripts/extract_features.py first)")
            n_missing_npz += 1
            continue
        result = compare_npz_to_occwl_raw_order(stem, npz_path)
        if result["status"] == "MATCH":
            n_match += 1
        else:
            n_mismatch += 1
            print(f"  {stem}: {result['status']} (details: {result})")

    print(f"\nBatch of {len(sample_stems)}: matched={n_match} mismatched={n_mismatch} missing_npz={n_missing_npz}")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    all_ok = result["status"] != "POSITIONAL_MISMATCH" and n_mismatch == 0
    if all_ok:
        print("npz face order == occwl raw traversal order (positionally), confirmed on "
              f"1 hand-checked file + {n_match} random subset files.")
    else:
        print("MISMATCH DETECTED -- see details above. Do not assume npz order == occwl raw order.")


if __name__ == "__main__":
    main()
