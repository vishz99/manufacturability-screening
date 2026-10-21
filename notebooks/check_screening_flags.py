"""
Runs screening end to end on a handful of MFCAD files (outside the
training subset where possible) and prints the flags, so the rule
table's real behavior can be eyeballed against real parts rather than
only unit-tested in isolation (tests/test_rules.py checks the rule
table's structure, not what it produces on real predictions).

Run with the `brepnet` conda environment's python:
    <path to conda>/envs/brepnet/python.exe notebooks/check_screening_flags.py
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ui.analysis import analyse_part  # noqa: E402
from screening.rules import DISCLAIMER  # noqa: E402

STEP_DIR = PROJECT_ROOT / "data" / "MFCAD" / "dataset" / "step"
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "mfcad_manifest.csv"
SUBSET_PATH = PROJECT_ROOT / "outputs" / "mfcad_subset.txt"

N_FILES = 5


def pick_stems_outside_subset(n):
    subset = set(SUBSET_PATH.read_text().split())
    stems = []
    with open(MANIFEST_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["ok"] == "True" and row["stem"] not in subset:
                stems.append(row["stem"])
                if len(stems) >= n:
                    break
    return stems


def main():
    stems = pick_stems_outside_subset(N_FILES)
    print(f"Checking {len(stems)} files outside the training subset\n")

    for stem in stems:
        step_path = STEP_DIR / f"{stem}.step"
        print("=" * 70)
        print(f"{stem}  ({step_path})")
        analysis = analyse_part(step_path)
        print(f"Faces: {analysis.prediction.num_faces}" + (
            f"  accuracy={analysis.accuracy:.4f}" if analysis.accuracy is not None else ""
        ))
        if not analysis.flags:
            print("Flags: none")
        for f in analysis.flags:
            print(f"  [{f.severity:6s}] {f.class_name:28s} {f.message} (reliability={f.reliability})")
        print()

    print(DISCLAIMER)


if __name__ == "__main__":
    main()
