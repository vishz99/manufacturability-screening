"""
Same three views as scripts/report_per_class_stats.py (per-class face
counts, per-class accuracy, predicted-class distribution), scaled to
the 1000-file subset and parameterized so it can evaluate any of the
three trained variants (A/B/C).

Run with the `brepnet` conda environment's python:
    <path to conda>/envs/brepnet/python.exe scripts/report_per_class_stats_1000.py \
        outputs/mfcad_head_checkpoint_1000_A.pt [hidden_size]

hidden_size is optional; pass it (e.g. 128) when evaluating a variant
that used an MLP head (must match --hidden-size given to train_head_1000.py
for that checkpoint), omit it for a plain linear head.
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

FEATURES_DIR = PROJECT_ROOT / "outputs" / "brepnet_features_1000"
LABELS_CACHE_DIR = PROJECT_ROOT / "outputs" / "mfcad_labels_cache_1000"
SUBSET_LIST = PROJECT_ROOT / "outputs" / "mfcad_subset_1000.txt"
SPLIT_DATASET_FILE_PATH = PROJECT_ROOT / "outputs" / "mfcad_train_val_split_1000.json"
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
    assert len(sys.argv) >= 2, "Usage: report_per_class_stats_1000.py <checkpoint_path> [hidden_size]"
    checkpoint_path = Path(sys.argv[1])
    hidden_size = int(sys.argv[2]) if len(sys.argv) > 2 else None

    from data.mfcad_loader import FEAT_NAMES
    from dataloaders.brepnet_dataset import BRepNetDataset, brepnet_collate_fn
    from model.brepnet_backbone import load_frozen_backbone_with_new_head
    import argparse

    all_stems = SUBSET_LIST.read_text().split()
    split = json.loads(SPLIT_DATASET_FILE_PATH.read_text())
    train_stems = split["train_set"]
    val_stems = split["val_set"]

    full_counts = count_labels_for_stems(all_stems)
    train_counts = count_labels_for_stems(train_stems)
    val_counts = count_labels_for_stems(val_stems)

    print("=" * 90)
    print("PER-CLASS FACE COUNTS -- 1000-file verified subset (from cached training labels)")
    print("=" * 90)
    print(f"{'idx':>3} {'class name':30} {'subset (1000)':>14} {'train (800)':>12} {'val (200)':>10}")
    for idx, name in enumerate(FEAT_NAMES):
        print(f"{idx:>3} {name:30} {full_counts.get(idx, 0):>14} {train_counts.get(idx, 0):>12} {val_counts.get(idx, 0):>10}")
    total_subset = sum(full_counts.values())
    print(f"{'':>3} {'TOTAL':30} {total_subset:>14} {sum(train_counts.values()):>12} {sum(val_counts.values()):>10}")

    print("\n" + "=" * 90)
    print(f"PER-CLASS ACCURACY -- checkpoint={checkpoint_path.name} hidden_size={hidden_size} -- val set ({len(val_stems)} files)")
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

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    # nn.Sequential names submodules by position index -- if a Dropout layer
    # was present at training time, the final Linear's state_dict key shifts
    # (e.g. "2.weight" -> "3.weight"), so the architecture must be
    # reconstructed with the SAME dropout setting used at training time, not
    # just the same hidden_size, or load_state_dict will raise a key
    # mismatch. Prefer what's recorded in the checkpoint itself over the CLI
    # arg (falls back to the CLI hidden_size for older checkpoints that
    # predate the "dropout"/"hidden_size" fields).
    if hidden_size is None:
        hidden_size = checkpoint.get("hidden_size")
    dropout = checkpoint.get("dropout")
    print(f"Reconstructing model with hidden_size={hidden_size} dropout={dropout} (from checkpoint metadata)")

    model = load_frozen_backbone_with_new_head(NUM_MFCAD_CLASSES, hidden_size=hidden_size, dropout=dropout)
    model.classification_layer.load_state_dict(checkpoint["classification_layer_state_dict"])
    model.eval()

    per_class_correct = Counter()
    per_class_total = Counter()
    per_class_predicted = Counter()
    confusion_most_common_wrong = {i: Counter() for i in range(NUM_MFCAD_CLASSES)}
    true_to_pred_breakdown = {i: Counter() for i in range(NUM_MFCAD_CLASSES)}

    with torch.no_grad():
        for batch in val_loader:
            face_embeddings = model.create_face_embeddings(
                batch["face_features"], batch["face_point_grids"],
                batch["edge_features"], batch["edge_point_grids"],
                batch["coedge_features"], batch["coedge_point_grids"],
                batch["face_kernel_tensor"], batch["edge_kernel_tensor"], batch["coedge_kernel_tensor"],
                batch["coedges_of_edges"], batch["coedges_of_small_faces"], batch["coedges_of_big_faces"],
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
    zero_recall = []
    for idx, name in enumerate(FEAT_NAMES):
        n = per_class_total.get(idx, 0)
        c = per_class_correct.get(idx, 0)
        acc_str = f"{c / n:.4f}" if n > 0 else "n/a"
        if n > 0 and c == 0:
            zero_recall.append(name)
        wrong = confusion_most_common_wrong[idx]
        if wrong:
            top_wrong_idx, top_wrong_count = wrong.most_common(1)[0]
            wrong_str = f"{FEAT_NAMES[top_wrong_idx]} ({top_wrong_count}x)"
        else:
            wrong_str = "-"
        print(f"{idx:>3} {name:30} {n:>7} {c:>7} {acc_str:>9} {wrong_str:>28}")

    print(f"\nZero-recall classes ({len(zero_recall)}): {zero_recall if zero_recall else 'none'}")

    print("\n" + "=" * 90)
    print("PREDICTED-CLASS DISTRIBUTION")
    print("=" * 90)
    print(f"{'idx':>3} {'class name':30} {'n_true':>7} {'n_predicted':>12} {'pred_share':>10}")
    for idx, name in enumerate(FEAT_NAMES):
        n_pred = per_class_predicted.get(idx, 0)
        print(f"{idx:>3} {name:30} {per_class_total.get(idx, 0):>7} {n_pred:>12} {n_pred / overall_total:>10.4f}")

    never_predicted_at_all = [FEAT_NAMES[i] for i in range(NUM_MFCAD_CLASSES) if per_class_predicted.get(i, 0) == 0]
    print(f"\nClasses NEVER predicted at all: {never_predicted_at_all if never_predicted_at_all else 'none'}")


if __name__ == "__main__":
    main()
