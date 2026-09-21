"""
HISTORICAL (500-file, class-weighted experiment) -- not the canonical
pipeline. See scripts/train_head.py (--suffix C --epochs 40
--hidden-size 128 --save-best) for the accepted result. Kept for
reference/reproducibility only.

Same as experiments/500_unweighted/train_head.py (frozen BRepNet
backbone, kept in .eval() mode throughout, new 16-class head trained
with Adam scoped only to classification_layer, same 400/100 train/val
split, same 15 epochs), except the cross-entropy loss is class-weighted:
weight_c inversely proportional to class c's face count in the
TRAINING split, using the standard "balanced" formula
weight_c = n_train_faces / (num_classes * count_c).

Motivation: the unweighted run collapsed onto ~4-5 of 16 classes
(dominated by "stock" at ~30% of faces), leaving 8 classes at 0%
recall, 7 of them never predicted at all. Class weighting penalizes
misclassifying rare classes more heavily, which should push the head
to actually use its rare-class logits instead of ignoring them.

Saves to a SEPARATE checkpoint (checkpoint.pt, in this same folder) so
the unweighted baseline (experiments/500_unweighted/checkpoint.pt)
stays intact for direct comparison.

Run with the `brepnet` conda environment's python:
    <path to conda>/envs/brepnet/python.exe experiments/500_class_weighted/train_head_class_weighted.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BREPNET_DIR = PROJECT_ROOT / "external" / "BRepNet"
sys.path.insert(0, str(BREPNET_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FEATURES_DIR = PROJECT_ROOT / "outputs" / "brepnet_features"
LABELS_CACHE_DIR = PROJECT_ROOT / "outputs" / "mfcad_labels_cache"
SUBSET_LIST = PROJECT_ROOT / "experiments" / "500_shared_data_prep" / "mfcad_subset.txt"
FEATURE_STANDARDIZATION_PATH = (
    BREPNET_DIR / "example_files" / "feature_standardization" / "s2.0.0_step_all_features.json"
)
KERNEL_PATH = BREPNET_DIR / "kernels" / "winged_edge.json"
INPUT_FEATURES_PATH = BREPNET_DIR / "feature_lists" / "all.json"

# Same split file as experiments/500_unweighted/train_head.py -- reused,
# not regenerated, so the train/val split is identical between the two runs.
SPLIT_DATASET_FILE_PATH = PROJECT_ROOT / "experiments" / "500_shared_data_prep" / "mfcad_train_val_split.json"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoint.pt"

NUM_MFCAD_CLASSES = 16
N_TRAIN = 400
N_VAL = 100
SPLIT_SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3


def get_or_build_split():
    """Reuse the existing split file if train_head.py already made one
    with the same seed/sizes, so this run is on the identical split."""
    if SPLIT_DATASET_FILE_PATH.exists():
        existing = json.loads(SPLIT_DATASET_FILE_PATH.read_text())
        if len(existing.get("train_set", [])) == N_TRAIN and len(existing.get("val_set", [])) == N_VAL:
            print(f"Reusing existing split file: {SPLIT_DATASET_FILE_PATH}")
            return SPLIT_DATASET_FILE_PATH

    stems = SUBSET_LIST.read_text().split()
    assert len(stems) == N_TRAIN + N_VAL
    rng = random.Random(SPLIT_SEED)
    shuffled = stems[:]
    rng.shuffle(shuffled)
    train_stems = sorted(shuffled[:N_TRAIN])
    val_stems = sorted(shuffled[N_TRAIN:])

    standardization = json.loads(FEATURE_STANDARDIZATION_PATH.read_text())
    dataset = {
        "feature_standardization": standardization["feature_standardization"],
        "train_set": train_stems,
        "val_set": val_stems,
    }
    SPLIT_DATASET_FILE_PATH.write_text(json.dumps(dataset, indent=2))
    print(f"Built new split: {len(train_stems)} train / {len(val_stems)} val (seed={SPLIT_SEED})")
    return SPLIT_DATASET_FILE_PATH


def compute_class_weights(train_stems):
    """
    weight_c = n_train_faces / (num_classes * count_c)  ("balanced" scheme)

    Classes with fewer training faces get a larger weight, so
    misclassifying them costs more in the loss. A class with average
    frequency gets weight ~1.0.
    """
    counts = Counter()
    for stem in train_stems:
        labels = np.loadtxt(LABELS_CACHE_DIR / f"{stem}.seg", dtype=np.int64)
        if labels.ndim == 0:
            labels = np.expand_dims(labels, 0)
        counts.update(labels.tolist())

    n_train_faces = sum(counts.values())
    weights = torch.zeros(NUM_MFCAD_CLASSES, dtype=torch.float32)
    for c in range(NUM_MFCAD_CLASSES):
        count_c = counts.get(c, 0)
        assert count_c > 0, f"Class {c} has zero training examples -- balanced weighting would give it infinite weight"
        weights[c] = n_train_faces / (NUM_MFCAD_CLASSES * count_c)

    return weights, counts


def make_opts(dataset_file_path):
    import argparse

    return argparse.Namespace(
        dataset_dir=str(FEATURES_DIR),
        dataset_file=str(dataset_file_path),
        label_dir=str(LABELS_CACHE_DIR),
        kernel=str(KERNEL_PATH),
        input_features=str(INPUT_FEATURES_PATH),
    )


def verify_optimizer_scoping(model, optimizer):
    backbone_param_ids = {
        id(p) for n, p in model.named_parameters() if not n.startswith("classification_layer")
    }
    head_param_ids = {
        id(p) for n, p in model.named_parameters() if n.startswith("classification_layer")
    }
    optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}

    print("\n--- Optimizer scoping verification ---")
    overlap_with_backbone = optimizer_param_ids & backbone_param_ids
    overlap_with_head = optimizer_param_ids & head_param_ids
    print(f"  optimizer params that ARE backbone params (expect 0): {len(overlap_with_backbone)}")
    print(f"  optimizer params that ARE head params (expect {len(head_param_ids)}): {len(overlap_with_head)}")
    assert len(overlap_with_backbone) == 0
    assert overlap_with_head == head_param_ids
    print("OPTIMIZER SCOPING VERIFIED: only the new head's parameters are being optimized.")


def run_epoch(model, dataloader, optimizer, train, class_weights):
    # Kept in .eval() for BOTH phases -- requires_grad=False does not stop
    # BatchNorm buffers inside surface_encoder/curve_encoder from updating
    # in .train() mode (confirmed empirically in the unweighted run). This
    # is safe since classification_layer is a plain nn.Linear.
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_faces = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in dataloader:
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
            labels = batch["labels"]

            loss = F.cross_entropy(logits, labels, weight=class_weights, reduction="mean")

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            num_faces_in_batch = labels.shape[0]
            total_loss += loss.item() * num_faces_in_batch
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_faces += num_faces_in_batch

    mean_loss = total_loss / total_faces
    accuracy = total_correct / total_faces
    return mean_loss, accuracy, total_faces


def main():
    from dataloaders.brepnet_dataset import BRepNetDataset, brepnet_collate_fn
    from model.brepnet_backbone import load_frozen_backbone_with_new_head
    from data.mfcad_loader import FEAT_NAMES

    dataset_file_path = get_or_build_split()
    split = json.loads(dataset_file_path.read_text())
    train_stems = split["train_set"]

    class_weights, train_counts = compute_class_weights(train_stems)
    print("\nClass weights (balanced, inverse training frequency):")
    for idx, name in enumerate(FEAT_NAMES):
        print(f"  [{idx:2d}] {name:28} train_count={train_counts.get(idx, 0):5d}  weight={class_weights[idx]:.4f}")

    opts = make_opts(dataset_file_path)
    train_dataset = BRepNetDataset(opts, "train_set")
    val_dataset = BRepNetDataset(opts, "val_set")
    print(f"\ntrain_dataset: {len(train_dataset)} solids, val_dataset: {len(val_dataset)} solids")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=brepnet_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=brepnet_collate_fn)

    model = load_frozen_backbone_with_new_head(NUM_MFCAD_CLASSES)
    optimizer = torch.optim.Adam(model.classification_layer.parameters(), lr=LEARNING_RATE)

    verify_optimizer_scoping(model, optimizer)

    print(f"\n--- Training for {NUM_EPOCHS} epochs (batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, class-weighted loss) ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, n_train_faces = run_epoch(model, train_loader, optimizer, train=True, class_weights=class_weights)
        val_loss, val_acc, n_val_faces = run_epoch(model, val_loader, optimizer, train=False, class_weights=class_weights)
        print(
            f"epoch {epoch:2d}/{NUM_EPOCHS}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} (n={n_train_faces})  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} (n={n_val_faces})"
        )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classification_layer_state_dict": model.classification_layer.state_dict(),
            "num_classes": NUM_MFCAD_CLASSES,
            "in_features": model.classification_layer.in_features,
            "class_weighted": True,
        },
        CHECKPOINT_PATH,
    )
    print(f"\nSaved trained head checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
