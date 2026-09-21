"""
Scans the full MFCAD dataset and validates every STEP/.face_truth pair
using src/data/mfcad_loader.py's name-based alignment check. Writes a
manifest (outputs/mfcad_manifest.csv) recording, per file: whether it's
usable, its face count, and (if excluded) why.

Run with the `brepnet` conda environment's python (needs occwl):
    <path to conda>/envs/brepnet/python.exe scripts/build_manifest.py

Takes roughly 90 minutes for the full ~15,488-file dataset on this
machine (~0.35s/file, dominated by STEP parsing).
"""

import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "MFCAD" / "dataset" / "step"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "mfcad_manifest.csv"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data.mfcad_loader import validate_pair  # noqa: E402

def main():
    step_files = sorted(DATA_DIR.glob("*.step"))
    print(f"Found {len(step_files)} STEP files under {DATA_DIR}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    n_ok = 0
    n_fail = 0

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "ok", "num_faces", "reason"])

        for i, step_path in enumerate(step_files, start=1):
            result = validate_pair(step_path)
            writer.writerow([result.stem, result.ok, result.num_faces, result.reason])
            if result.ok:
                n_ok += 1
            else:
                n_fail += 1

            if i % 500 == 0 or i == len(step_files):
                f.flush()
                elapsed = time.time() - start
                rate = i / elapsed
                eta = (len(step_files) - i) / rate if rate > 0 else float("inf")
                print(
                    f"[{i}/{len(step_files)}] ok={n_ok} fail={n_fail} "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )

    print(f"\nDone. ok={n_ok} fail={n_fail} total={len(step_files)}")
    print(f"Manifest written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
