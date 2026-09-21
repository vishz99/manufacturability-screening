"""
Diagnostic requested before Phase 4: reports, for the 500-file verified
subset,
  1. per-class face counts (from the cached labels actually used for
     training, outputs/mfcad_labels_cache/), broken down by the full
     subset and by the train/val split used in scripts/train_head.py.
  2. per-class accuracy of the trained head (outputs/mfcad_head_checkpoint.pt)
     on the val set -- which classes it's actually getting right and
     which it's failing on entirely.

Reuses the same verified BRepNetDataset/brepnet_collate_fn pipeline and
train/val split file (outputs/mfcad_train_val_split.json) that training
used, so the accuracy numbers here are directly comparable to
scripts/train_head.py's final epoch.

Run with the `brepnet` conda environment's python:
    <path to conda>/envs/brepnet/python.exe scripts/report_per_class_stats.py [checkpoint_path]

An optional first argument overrides which checkpoint to evaluate
(default outputs/mfcad_head_checkpoint.pt) -- e.g. pass
outputs/mfcad_head_checkpoint_class_weighted.pt to evaluate the
class-weighted run (scripts/train_head_class_weighted.py) instead.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BREPNET_DIR = PROJECT_ROOT / "external" / "BRepNet"
sys.path.insert(0, str(BREPNET_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FEATURES_DIR = PROJECT_ROOT / "outputs" / "brepnet_features"
LABELS_CACHE_DIR = PROJECT_ROOT / "outputs" / "mfcad_labels_cache"
SUBSET_LIST = PROJECT_ROOT / "outputs" / "mfcad_subset.txt"
SPLIT_DATASET_FILE_PATH = PROJECT_ROOT / "outputs" / "mfcad_train_val_split.json"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "mfcad_head_checkpoint.pt"
KERNEL_PATH = BREPNET_DIR / "kernels" / "winged_edge.json"
INPUT_FEATURES_PATH = BREPNET_DIR / "feature_lists" / "all.json"

NUM_MFCAD_CLASSES = 16


def count_labels_for_stems(stems):
    counts = Counter()
    for stem in stems:
        labels = np.loadtxt(LABELS_CACHE_DIR / f"{stem}.seg", dtype=np.int64)
        if labels.ndim == 0:
            labels = np.expand_dims(labels, 0)
        counts.update(labels.tolist())
    return counts


def main():
    from data.mfcad_loader import FEAT_NAMES
    from dataloaders.brepnet_dataset import BRepNetDataset, brepnet_collate_fn
    from model.brepnet_backbone import load_frozen_backbone_with_new_head
    import argparse

    # --- Part 1: per-class face counts ---
    all_stems = SUBSET_LIST.read_text().split()
    split = json.loads(SPLIT_DATASET_FILE_PATH.read_text())
    train_stems = split["train_set"]
    val_stems = split["val_set"]

    full_counts = count_labels_for_stems(all_stems)
    train_counts = count_labels_for_stems(train_stems)
    val_counts = count_labels_for_stems(val_stems)

    print("=" * 90)
    print(f"PER-CLASS FACE COUNTS -- 500-file verified subset (from cached training labels)")
    print("=" * 90)
    print(f"{'idx':>3} {'class name':30} {'subset (500)':>13} {'train (400)':>12} {'val (100)':>10}")
    for idx, name in enumerate(FEAT_NAMES):
        print(f"{idx:>3} {name:30} {full_counts.get(idx, 0):>13} {train_counts.get(idx, 0):>12} {val_counts.get(idx, 0):>10}")
    total_subset = sum(full_counts.values())
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    print(f"{'':>3} {'TOTAL':30} {total_subset:>13} {total_train:>12} {total_val:>10}")

    zero_classes = [FEAT_NAMES[i] for i in range(NUM_MFCAD_CLASSES) if full_counts.get(i, 0) == 0]
    if zero_classes:
        print(f"\nWARNING: classes with zero examples anywhere in the subset: {zero_classes}")

    # --- Part 2: per-class accuracy of the trained head on val set ---
    print("\n" + "=" * 90)
    print("PER-CLASS ACCURACY -- trained head evaluated on the val set (100 files)")
    print("=" * 90)

    opts = argparse.Namespace(
        dataset_dir=str(FEATURES_DIR),
        dataset_file=str(SPLIT_DATASET_FILE_PATH),
        label_dir=str(LABELS_CACHE_DIR),
        kernel=str(KERNEL_PATH),
        input_features=str(INPUT_FEATURES_PATH),
    )
    val_dataset = BRepNetDataset(opts, "val_set")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=brepnet_collate_fn)

    checkpoint_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHECKPOINT_PATH
    print(f"Evaluating checkpoint: {checkpoint_path}")

    model = load_frozen_backbone_with_new_head(NUM_MFCAD_CLASSES)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.classification_layer.load_state_dict(checkpoint["classification_layer_state_dict"])
    model.eval()

    per_class_correct = Counter()
    per_class_total = Counter()
    per_class_predicted = Counter()  # how many times the model output each class, regardless of truth
    confusion_most_common_wrong = {i: Counter() for i in range(NUM_MFCAD_CLASSES)}
    true_to_pred_breakdown = {i: Counter() for i in range(NUM_MFCAD_CLASSES)}  # full prediction breakdown per true class

    with torch.no_grad():
        for batch in val_loader:
            face_embeddings = model.create_face_embeddings(
                batch["face_features"],
                batch["face_point_grids"],
                batch["edge_features"],
                batch["edge_point_grids"],
                batch["coedge_features"],
                batch["coedge_point_grids"],
                batch["face_kernel_tensor"],
                batch["edge_kernel_tensor"],
                batch["coedge_kernel_tensor"],
                batch["coedges_of_edges"],
                batch["coedges_of_small_faces"],
                batch["coedges_of_big_faces"],
            )
            logits = model.classification_layer(face_embeddings)
            preds = logits.argmax(dim=1)
            labels = batch["labels"]

            for true_label, pred_label in zip(labels.tolist(), preds.tolist()):
                per_class_total[true_label] += 1
                per_class_predicted[pred_label] += 1
                true_to_pred_breakdown[true_label][pred_label] += 1
                if pred_label == true_label:
                    per_class_correct[true_label] += 1
                else:
                    confusion_most_common_wrong[true_label][pred_label] += 1

    overall_correct = sum(per_class_correct.values())
    overall_total = sum(per_class_total.values())
    print(f"Overall val accuracy: {overall_correct}/{overall_total} = {overall_correct / overall_total:.4f}\n")

    print(f"{'idx':>3} {'class name':30} {'n_val':>7} {'correct':>7} {'accuracy':>9} {'most common wrong pred':>28}")
    never_predicted_correctly = []
    for idx, name in enumerate(FEAT_NAMES):
        n = per_class_total.get(idx, 0)
        c = per_class_correct.get(idx, 0)
        acc_str = f"{c / n:.4f}" if n > 0 else "n/a"
        if n > 0 and c == 0:
            never_predicted_correctly.append(name)

        wrong = confusion_most_common_wrong[idx]
        if wrong:
            top_wrong_idx, top_wrong_count = wrong.most_common(1)[0]
            wrong_str = f"{FEAT_NAMES[top_wrong_idx]} ({top_wrong_count}x)"
        else:
            wrong_str = "-"
        print(f"{idx:>3} {name:30} {n:>7} {c:>7} {acc_str:>9} {wrong_str:>28}")

    print(f"\nClasses present in val set but NEVER predicted correctly (0% recall): {never_predicted_correctly if never_predicted_correctly else 'none'}")

    # --- Part 3: overall predicted-class distribution ---
    print("\n" + "=" * 90)
    print("PREDICTED-CLASS DISTRIBUTION -- what the model actually outputs across all val faces")
    print("=" * 90)
    print(f"{'idx':>3} {'class name':30} {'n_true':>7} {'n_predicted':>12} {'pred_share':>10}")
    for idx, name in enumerate(FEAT_NAMES):
        n_pred = per_class_predicted.get(idx, 0)
        print(f"{idx:>3} {name:30} {per_class_total.get(idx, 0):>7} {n_pred:>12} {n_pred / overall_total:>10.4f}")
    print(f"{'':>3} {'TOTAL':30} {overall_total:>7} {sum(per_class_predicted.values()):>12}")

    never_predicted_at_all = [FEAT_NAMES[i] for i in range(NUM_MFCAD_CLASSES) if per_class_predicted.get(i, 0) == 0]
    print(f"\nClasses the model NEVER predicts at all (0 predictions across entire val set): "
          f"{never_predicted_at_all if never_predicted_at_all else 'none'}")

    print("\n--- Zero-recall classes: full prediction breakdown for their true faces ---")
    for name in never_predicted_correctly:
        idx = FEAT_NAMES.index(name)
        dist = true_to_pred_breakdown[idx]
        n = per_class_total.get(idx, 0)
        breakdown = ", ".join(f"{FEAT_NAMES[p]}={c}" for p, c in dist.most_common())
        print(f"  [{idx:2d}] {name:28} n={n:4d}  predicted as: {breakdown}")


if __name__ == "__main__":
    main()
