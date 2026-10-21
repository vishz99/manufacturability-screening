"""
Critical check: predict_step_file() (src/inference/predict_part.py),
run through its FULL pipeline (subprocess extraction -> tessellation ->
BRepNetDataset standardization -> model -> un-permute), should
reproduce ~0.6035 overall accuracy on the validation split -- the same
number scripts/report_per_class_stats.py gets by reading
already-extracted features directly from outputs/brepnet_features_1000/.

If these diverge substantially, the new single-file pipeline's feature
standardization or face order differs from the original training
pipeline somewhere (the two most likely suspects per the project's own
history of bugs), and that must be fixed before anything built on
predict_part.py (rules, the app) can be trusted.

Run with the `brepnet` conda environment's python:
    <path to conda>/envs/brepnet/python.exe notebooks/check_predict_part_accuracy.py [n_files]

n_files (optional, default all 200): sample size from the validation
split, for a quicker smoke check before running the full set.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference.predict_part import predict_step_file, checkpoint_metadata  # noqa: E402

STEP_DIR = PROJECT_ROOT / "data" / "MFCAD" / "dataset" / "step"
SPLIT_PATH = PROJECT_ROOT / "outputs" / "mfcad_train_val_split.json"
LABELS_CACHE_DIR = PROJECT_ROOT / "outputs" / "mfcad_labels_cache_1000"

ACCEPTED_VAL_ACC = 0.6035
GAP_TOLERANCE = 0.03  # allow small noise (e.g. a handful of float/argmax tie differences)


def main():
    import numpy as np

    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else None

    split = json.loads(SPLIT_PATH.read_text())
    val_stems = split["val_set"]
    if n_files:
        val_stems = val_stems[:n_files]

    meta = checkpoint_metadata()
    print(f"Checkpoint: epoch {meta['saved_epoch']}, recorded val_acc={meta['saved_epoch_val_acc']:.4f}")
    print(f"Running predict_step_file() over {len(val_stems)} validation files...\n")

    total_correct = 0
    total_faces = 0
    failures = []

    for i, stem in enumerate(val_stems, start=1):
        step_path = STEP_DIR / f"{stem}.step"
        true_labels = np.loadtxt(LABELS_CACHE_DIR / f"{stem}.seg", dtype=np.int64)
        if true_labels.ndim == 0:
            true_labels = np.expand_dims(true_labels, 0)

        try:
            result = predict_step_file(step_path)
        except Exception as e:
            failures.append((stem, str(e)))
            continue

        if result.num_faces != len(true_labels):
            failures.append((stem, f"face count mismatch: predicted {result.num_faces}, true {len(true_labels)}"))
            continue

        preds = [fp.predicted_class for fp in result.predictions]
        correct = sum(1 for p, t in zip(preds, true_labels) if p == t)
        total_correct += correct
        total_faces += len(true_labels)

        if i % 25 == 0:
            running_acc = total_correct / total_faces if total_faces else 0.0
            print(f"  ...{i}/{len(val_stems)} files, running accuracy={running_acc:.4f}")

    print(f"\nProcessed {len(val_stems) - len(failures)}/{len(val_stems)} files ({len(failures)} failures)")
    if failures:
        print("Failures:")
        for stem, reason in failures[:20]:
            print(f"  {stem}: {reason}")

    if total_faces == 0:
        print("\nNo faces scored -- cannot compute accuracy.")
        sys.exit(1)

    accuracy = total_correct / total_faces
    gap = abs(accuracy - ACCEPTED_VAL_ACC)
    print(f"\npredict_step_file() pipeline accuracy: {total_correct}/{total_faces} = {accuracy:.4f}")
    print(f"Accepted checkpoint's reported val_acc: {ACCEPTED_VAL_ACC:.4f}")
    print(f"Gap: {gap:.4f} (tolerance: {GAP_TOLERANCE})")

    if gap > GAP_TOLERANCE:
        print("\nGATE FAILED: gap exceeds tolerance. The single-file prediction "
              "pipeline's standardization or face order likely differs from the "
              "training pipeline's. Fix before continuing.")
        sys.exit(1)

    print("\nGATE PASSED.")


if __name__ == "__main__":
    main()
