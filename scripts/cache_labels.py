"""
Canonical pipeline script (reorganized -- this was cache_labels_1000.py
until the 1000-file subset became the accepted scale; the original
500-file version now lives at experiments/500_shared_data_prep/cache_labels.py).

Precomputes and caches per-face labels for the 1000-file subset
(outputs/mfcad_subset.txt) into outputs/mfcad_labels_cache_1000/
(directory name kept as-is -- not renamed during the reorg, since it's
a large regeneratable cache with ~1000 tracked files, not a curated result).

Run with the `brepnet` conda environment's python (needs occwl/OCC):
    <path to conda>/envs/brepnet/python.exe scripts/cache_labels.py
"""

import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

STEP_DIR = PROJECT_ROOT / "data" / "MFCAD" / "dataset" / "step"
SUBSET_LIST = PROJECT_ROOT / "outputs" / "mfcad_subset.txt"
LABELS_CACHE_DIR = PROJECT_ROOT / "outputs" / "mfcad_labels_cache_1000"

N_FILES_TO_SPOT_CHECK = 20
SPOT_CHECK_SEED = 4


def main():
    from data.mfcad_loader import load_labeled_sample

    stems = SUBSET_LIST.read_text().split()
    print(f"Caching labels for {len(stems)} subset files")

    LABELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_failed = 0
    for i, stem in enumerate(stems, start=1):
        step_path = STEP_DIR / f"{stem}.step"
        try:
            sample = load_labeled_sample(step_path)
        except Exception as e:
            print(f"  FAILED to cache {stem}: {e}")
            n_failed += 1
            continue

        labels = np.array(sample.labels, dtype=np.int64)
        out_path = LABELS_CACHE_DIR / f"{stem}.seg"
        np.savetxt(out_path, labels, fmt="%d")
        n_ok += 1

        if i % 200 == 0:
            print(f"  ...cached {i}/{len(stems)}...")

    print(f"\nCached {n_ok}/{len(stems)} files ({n_failed} failed)")
    assert n_failed == 0, "Some subset files failed to cache -- investigate before training"

    print(f"\nSpot-checking {N_FILES_TO_SPOT_CHECK} cached files against fresh load_labeled_sample() calls...")
    rng = random.Random(SPOT_CHECK_SEED)
    spot_stems = rng.sample(stems, N_FILES_TO_SPOT_CHECK)

    n_match = 0
    for stem in spot_stems:
        cached = np.loadtxt(LABELS_CACHE_DIR / f"{stem}.seg", dtype=np.int64)
        if cached.ndim == 0:
            cached = np.expand_dims(cached, 0)

        fresh_sample = load_labeled_sample(STEP_DIR / f"{stem}.step")
        fresh = np.array(fresh_sample.labels, dtype=np.int64)

        match = cached.shape == fresh.shape and np.array_equal(cached, fresh)
        status = "MATCH" if match else "MISMATCH"
        n_match += int(match)
        print(f"  {stem}: {status}")

    print(f"\nSpot check: {n_match}/{N_FILES_TO_SPOT_CHECK} cached files exactly match a fresh reload")
    assert n_match == N_FILES_TO_SPOT_CHECK, "Cache does not match fresh loader output -- do not trust this cache"
    print("CACHE VERIFIED.")


if __name__ == "__main__":
    main()
