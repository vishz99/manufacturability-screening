"""
Canonical pipeline script (reorganized -- this was build_subset_1000.py
until the 1000-file subset became the accepted scale; the original
500-file version now lives at experiments/500_shared_data_prep/build_subset.py).

Draws a random, verified 1000-file working subset from
outputs/mfcad_manifest.csv (the full 15,488-file validation scan, still
100% valid, produced by scripts/build_manifest.py), re-verifies EVERY
chosen file with a fresh load_labeled_sample() call (count match +
embedded-name permutation check -- the same alignment checks proven
necessary in Phase 1, not skipped just because the process is
known-good), and reports per-class face counts.

Run with the `brepnet` conda environment's python:
    <path to conda>/envs/brepnet/python.exe scripts/build_subset.py
"""

import csv
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "mfcad_manifest.csv"
DATA_DIR = PROJECT_ROOT / "data" / "MFCAD" / "dataset" / "step"
SUBSET_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "mfcad_subset.txt"

SUBSET_SIZE = 1000
RANDOM_SEED = 1  # different from the 500-file subset's seed (0) -- a fresh draw, not required to be a superset

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data.mfcad_loader import FEAT_NAMES, load_labeled_sample  # noqa: E402


def main():
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found at {MANIFEST_PATH}. Run build_manifest.py first.")

    valid_stems = []
    n_total = 0
    n_ok = 0
    with open(MANIFEST_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            if row["ok"] == "True":
                n_ok += 1
                valid_stems.append(row["stem"])

    print(f"Manifest: {n_total} files scanned, {n_ok} valid ({n_ok / n_total:.1%})")

    if SUBSET_SIZE > len(valid_stems):
        raise SystemExit(f"Requested subset size {SUBSET_SIZE} > {len(valid_stems)} valid files")

    rng = random.Random(RANDOM_SEED)
    subset = rng.sample(valid_stems, SUBSET_SIZE)
    subset.sort()

    SUBSET_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUBSET_OUTPUT_PATH, "w") as f:
        f.write("\n".join(subset) + "\n")
    print(f"Wrote {len(subset)} stems to {SUBSET_OUTPUT_PATH}")

    # Same Phase 1 re-verification: fresh load_labeled_sample() per file
    # (count match, embedded-STEP-name permutation check, label range
    # check) -- not skipped, run on all 1000 files.
    class_counts = Counter()
    n_faces_total = 0
    failures = []
    for i, stem in enumerate(subset, start=1):
        step_path = DATA_DIR / f"{stem}.step"
        try:
            sample = load_labeled_sample(step_path)
        except Exception as e:
            failures.append((stem, str(e)))
            continue
        n_faces_total += len(sample.labels)
        class_counts.update(sample.labels)
        if i % 200 == 0:
            print(f"  ...re-verified {i}/{len(subset)}...")

    print(f"\nRe-verified {len(subset) - len(failures)}/{len(subset)} subset files")
    if failures:
        print("UNEXPECTED failures during re-verification (manifest said OK):")
        for stem, reason in failures:
            print(f"  {stem}: {reason}")
    assert not failures, "Re-verification failures on a manifest-OK file -- investigate before proceeding"

    print(f"\nTotal faces in subset: {n_faces_total}")
    print("Per-class face-label counts:")
    for idx, name in enumerate(FEAT_NAMES):
        print(f"  [{idx:2d}] {name:30s} {class_counts.get(idx, 0)}")

    missing_classes = [FEAT_NAMES[i] for i in range(len(FEAT_NAMES)) if class_counts.get(i, 0) == 0]
    if missing_classes:
        print(f"\nWARNING: classes with zero examples in subset: {missing_classes}")


if __name__ == "__main__":
    main()
