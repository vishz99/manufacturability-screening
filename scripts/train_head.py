"""
Canonical pipeline script (reorganized -- this was train_head_1000.py
until the 1000-file subset became the accepted scale; the original
500-file unweighted/class-weighted versions now live at
experiments/500_unweighted/train_head.py and
experiments/500_class_weighted/train_head_class_weighted.py).

THE ACCEPTED RESULT (outputs/final/) was produced by:
    python train_head.py --suffix C --epochs 40 --hidden-size 128 --save-best

Trains a new MFCAD segmentation head on the 1000-file subset
(outputs/mfcad_subset.txt / outputs/brepnet_features_1000 /
outputs/mfcad_labels_cache_1000 -- the last two directory names were
kept as-is during the reorg since they're large regeneratable caches),
on top of the frozen BRepNet backbone, with class-weighted
cross-entropy loss and the eval-mode-throughout fix (see run_epoch()
below for why).

Parameterized by CLI args so every variant explored (A: linear/15ep,
B: linear/40ep [+60ep via experiments/1000_variant_B_continued_60ep/],
C: MLP-128/40ep [accepted], D: MLP-128/lr=1e-4, E: MLP-128/dropout=0.3)
was run from this one script without duplicating the training loop:

    python train_head.py --suffix <name> --epochs <N> [--hidden-size <H>] [--lr <LR>] [--dropout <P>] [--save-best]

Train/val split is 800/200 (80/20, proportional to the 400/100 split
used for the 500-file subset), built once and reused across every
variant for a fair comparison (outputs/mfcad_train_val_split.json).

Saves, per run:
  - outputs/mfcad_head_checkpoint_1000_<suffix>.pt (trained head weights)
  - outputs/mfcad_training_history_1000_<suffix>.json (full per-epoch
    train/val loss+accuracy curves, not just the final epoch, so
    convergence/overfitting can be inspected afterward)

Run with the `brepnet` conda environment's python:
    <path to conda>/envs/brepnet/python.exe scripts/train_head.py --suffix A --epochs 15
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BREPNET_DIR = PROJECT_ROOT / "external" / "BRepNet"
sys.path.insert(0, str(BREPNET_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FEATURES_DIR = PROJECT_ROOT / "outputs" / "brepnet_features_1000"
LABELS_CACHE_DIR = PROJECT_ROOT / "outputs" / "mfcad_labels_cache_1000"
SUBSET_LIST = PROJECT_ROOT / "outputs" / "mfcad_subset.txt"
FEATURE_STANDARDIZATION_PATH = (
    BREPNET_DIR / "example_files" / "feature_standardization" / "s2.0.0_step_all_features.json"
)
KERNEL_PATH = BREPNET_DIR / "kernels" / "winged_edge.json"
INPUT_FEATURES_PATH = BREPNET_DIR / "feature_lists" / "all.json"

SPLIT_DATASET_FILE_PATH = PROJECT_ROOT / "outputs" / "mfcad_train_val_split.json"

NUM_MFCAD_CLASSES = 16
N_TRAIN = 800
N_VAL = 200
SPLIT_SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 1e-3


def get_or_build_split():
    if SPLIT_DATASET_FILE_PATH.exists():
        existing = json.loads(SPLIT_DATASET_FILE_PATH.read_text())
        if len(existing.get("train_set", [])) == N_TRAIN and len(existing.get("val_set", [])) == N_VAL:
            print(f"Reusing existing split file: {SPLIT_DATASET_FILE_PATH}")
            return SPLIT_DATASET_FILE_PATH

    stems = SUBSET_LIST.read_text().split()
    assert len(stems) == N_TRAIN + N_VAL, f"Expected {N_TRAIN + N_VAL} subset files, found {len(stems)}"
    rng = random.Random(SPLIT_SEED)
    shuffled = stems[:]
    rng.shuffle(shuffled)
    train_stems = sorted(shuffled[:N_TRAIN])
    val_stems = sorted(shuffled[N_TRAIN:])
    assert len(set(train_stems) & set(val_stems)) == 0

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
        assert count_c > 0, f"Class {c} has zero training examples"
        weights[c] = n_train_faces / (NUM_MFCAD_CLASSES * count_c)
    return weights, counts


def make_opts(dataset_file_path):
    return argparse.Namespace(
        dataset_dir=str(FEATURES_DIR),
        dataset_file=str(dataset_file_path),
        label_dir=str(LABELS_CACHE_DIR),
        kernel=str(KERNEL_PATH),
        input_features=str(INPUT_FEATURES_PATH),
    )


def verify_optimizer_scoping(model, optimizer):
    backbone_param_ids = {id(p) for n, p in model.named_parameters() if not n.startswith("classification_layer")}
    head_param_ids = {id(p) for n, p in model.named_parameters() if n.startswith("classification_layer")}
    optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}

    print("\n--- Optimizer scoping verification ---")
    overlap_with_backbone = optimizer_param_ids & backbone_param_ids
    overlap_with_head = optimizer_param_ids & head_param_ids
    print(f"  optimizer params that ARE backbone params (expect 0): {len(overlap_with_backbone)}")
    print(f"  optimizer params that ARE head params (expect {len(head_param_ids)}): {len(overlap_with_head)}")
    assert len(overlap_with_backbone) == 0
    assert overlap_with_head == head_param_ids
    print("OPTIMIZER SCOPING VERIFIED.")


def run_epoch(model, dataloader, optimizer, train, class_weights):
    # Backbone always .eval() -- requires_grad=False does not stop BatchNorm
    # running-stat buffers from drifting in .train() mode (see scripts/train_head.py).
    model.eval()
    # The head follows the actual phase, so nn.Dropout (variant E) is active
    # during training and disabled during validation, as intended. For heads
    # with no Dropout/BatchNorm sublayers (plain Linear, or MLP without
    # dropout) this is a no-op -- identical behavior to always-eval.
    model.classification_layer.train(train)

    total_loss = 0.0
    total_correct = 0
    total_faces = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in dataloader:
            face_embeddings = model.create_face_embeddings(
                batch["face_features"], batch["face_point_grids"],
                batch["edge_features"], batch["edge_point_grids"],
                batch["coedge_features"], batch["coedge_point_grids"],
                batch["face_kernel_tensor"], batch["edge_kernel_tensor"], batch["coedge_kernel_tensor"],
                batch["coedges_of_edges"], batch["coedges_of_small_faces"], batch["coedges_of_big_faces"],
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

    return total_loss / total_faces, total_correct / total_faces, total_faces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", required=True, help="Label for this variant, e.g. A/B/C")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=None, help="If set, use an MLP head with this hidden size instead of a plain linear head")
    parser.add_argument("--save-best", action="store_true", help="Save the checkpoint from the best-val-accuracy epoch instead of the final epoch")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Learning rate (default {LEARNING_RATE})")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout after the hidden layer (only meaningful with --hidden-size); default None = no dropout")
    args = parser.parse_args()

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

    head_desc = "linear" if args.hidden_size is None else f"MLP(hidden={args.hidden_size}, dropout={args.dropout})"
    model = load_frozen_backbone_with_new_head(NUM_MFCAD_CLASSES, hidden_size=args.hidden_size, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.classification_layer.parameters(), lr=args.lr)

    verify_optimizer_scoping(model, optimizer)

    print(f"\n--- Variant {args.suffix}: {head_desc} head, {args.epochs} epochs, batch_size={BATCH_SIZE}, lr={args.lr}, class-weighted loss, save_best={args.save_best} ---")
    import copy

    history = []
    best_val_acc = -1.0
    best_epoch = None
    best_state_dict = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, n_train_faces = run_epoch(model, train_loader, optimizer, train=True, class_weights=class_weights)
        val_loss, val_acc, n_val_faces = run_epoch(model, val_loader, optimizer, train=False, class_weights=class_weights)
        print(
            f"epoch {epoch:2d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} (n={n_train_faces})  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} (n={n_val_faces})"
        )
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.classification_layer.state_dict())

    if args.save_best:
        print(f"\nBest val_acc={best_val_acc:.4f} occurred at epoch {best_epoch}/{args.epochs} -- saving that checkpoint (not the final epoch).")
        saved_state_dict = best_state_dict
        saved_epoch = best_epoch
        saved_val_acc = best_val_acc
    else:
        saved_state_dict = model.classification_layer.state_dict()
        saved_epoch = args.epochs
        saved_val_acc = history[-1]["val_acc"]

    checkpoint_path = PROJECT_ROOT / "outputs" / f"mfcad_head_checkpoint_1000_{args.suffix}.pt"
    torch.save(
        {
            "classification_layer_state_dict": saved_state_dict,
            "num_classes": NUM_MFCAD_CLASSES,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "lr": args.lr,
            "class_weighted": True,
            "saved_epoch": saved_epoch,
            "saved_epoch_val_acc": saved_val_acc,
            "is_best_checkpoint": args.save_best,
        },
        checkpoint_path,
    )
    print(f"\nSaved checkpoint to {checkpoint_path}")

    history_path = PROJECT_ROOT / "outputs" / f"mfcad_training_history_1000_{args.suffix}.json"
    history_path.write_text(json.dumps({
        "suffix": args.suffix, "epochs": args.epochs, "hidden_size": args.hidden_size,
        "history": history,
    }, indent=2))
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
