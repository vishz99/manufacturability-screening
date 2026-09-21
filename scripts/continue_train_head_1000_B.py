"""
Continues variant B's training (outputs/mfcad_head_checkpoint_1000_B.pt,
a plain linear head trained for 40 epochs on the 1000-file subset) for
20 more epochs, to see whether the curve -- which was still improving
with no overfitting signal at epoch 40 -- keeps improving, plateaus, or
starts overfitting by epoch 60.

CAVEAT, stated plainly: we only checkpointed the head's weights after
epoch 40, not the Adam optimizer's internal momentum/variance state, so
this is "continue from epoch 40's weights with a freshly-initialized
optimizer," not a byte-for-byte continuation of the exact same 40-epoch
trajectory. This is the standard practical meaning of "continue
training" when only model weights were checkpointed, but it's not
identical to training straight through for 60 epochs uninterrupted --
worth knowing when interpreting epoch 41's numbers relative to epoch 40's.

Tracks the best val-accuracy epoch across the FULL 60-epoch run
(comparing epoch 40's already-known val_acc against epochs 41-60) and
saves that as a separate best checkpoint, using the same
save-best-and-verify discipline as train_head_1000.py --save-best.

Run with the `brepnet` conda environment's python:
    <path to conda>/envs/brepnet/python.exe scripts/continue_train_head_1000_B.py
"""

import copy
import json
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
SPLIT_DATASET_FILE_PATH = PROJECT_ROOT / "outputs" / "mfcad_train_val_split_1000.json"
KERNEL_PATH = BREPNET_DIR / "kernels" / "winged_edge.json"
INPUT_FEATURES_PATH = BREPNET_DIR / "feature_lists" / "all.json"

EXISTING_CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "mfcad_head_checkpoint_1000_B.pt"
EXISTING_HISTORY_PATH = PROJECT_ROOT / "outputs" / "mfcad_training_history_1000_B.json"

OUT_CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "mfcad_head_checkpoint_1000_B60.pt"
OUT_BEST_CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "mfcad_head_checkpoint_1000_B60_best.pt"
OUT_HISTORY_PATH = PROJECT_ROOT / "outputs" / "mfcad_training_history_1000_B60.json"

NUM_MFCAD_CLASSES = 16
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
ADDITIONAL_EPOCHS = 20


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
        assert count_c > 0
        weights[c] = n_train_faces / (NUM_MFCAD_CLASSES * count_c)
    return weights


def run_epoch(model, dataloader, optimizer, train, class_weights):
    model.eval()  # frozen-backbone BatchNorm-buffer-drift fix, same as before
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
            n = labels.shape[0]
            total_loss += loss.item() * n
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_faces += n
    return total_loss / total_faces, total_correct / total_faces, total_faces


def main():
    from dataloaders.brepnet_dataset import BRepNetDataset, brepnet_collate_fn
    from model.brepnet_backbone import load_frozen_backbone_with_new_head

    existing_history = json.loads(EXISTING_HISTORY_PATH.read_text())["history"]
    assert len(existing_history) == 40
    best_so_far = max(existing_history, key=lambda e: e["val_acc"])
    print(f"Loaded existing 40-epoch history. Best so far: epoch {best_so_far['epoch']} val_acc={best_so_far['val_acc']:.4f}")

    split = json.loads(SPLIT_DATASET_FILE_PATH.read_text())
    train_stems = split["train_set"]
    class_weights = compute_class_weights(train_stems)

    opts = __import__("argparse").Namespace(
        dataset_dir=str(FEATURES_DIR),
        dataset_file=str(SPLIT_DATASET_FILE_PATH),
        label_dir=str(LABELS_CACHE_DIR),
        kernel=str(KERNEL_PATH),
        input_features=str(INPUT_FEATURES_PATH),
    )
    train_dataset = BRepNetDataset(opts, "train_set")
    val_dataset = BRepNetDataset(opts, "val_set")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=brepnet_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=brepnet_collate_fn)

    # Load B's epoch-40 weights into a fresh model (linear head).
    model = load_frozen_backbone_with_new_head(NUM_MFCAD_CLASSES, hidden_size=None)
    checkpoint_40 = torch.load(EXISTING_CHECKPOINT_PATH, weights_only=False)
    model.classification_layer.load_state_dict(checkpoint_40["classification_layer_state_dict"])
    optimizer = torch.optim.Adam(model.classification_layer.parameters(), lr=LEARNING_RATE)

    print(f"\n--- Continuing variant B from epoch 40 for {ADDITIONAL_EPOCHS} more epochs (to 60 total) ---")
    new_history = []
    best_val_acc = best_so_far["val_acc"]
    best_epoch = best_so_far["epoch"]
    best_state_dict = checkpoint_40["classification_layer_state_dict"]  # in case epoch 40 stays best

    for offset in range(1, ADDITIONAL_EPOCHS + 1):
        epoch = 40 + offset
        train_loss, train_acc, n_train = run_epoch(model, train_loader, optimizer, train=True, class_weights=class_weights)
        val_loss, val_acc, n_val = run_epoch(model, val_loader, optimizer, train=False, class_weights=class_weights)
        print(
            f"epoch {epoch:2d}/60  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} (n={n_train})  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} (n={n_val})"
        )
        entry = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
        new_history.append(entry)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.classification_layer.state_dict())

    full_history = existing_history + new_history
    OUT_HISTORY_PATH.write_text(json.dumps({"suffix": "B60", "epochs": 60, "hidden_size": None, "history": full_history}, indent=2))
    print(f"\nSaved combined 60-epoch history to {OUT_HISTORY_PATH}")

    torch.save(
        {
            "classification_layer_state_dict": model.classification_layer.state_dict(),
            "num_classes": NUM_MFCAD_CLASSES,
            "hidden_size": None,
            "class_weighted": True,
            "saved_epoch": 60,
            "saved_epoch_val_acc": full_history[-1]["val_acc"],
            "is_best_checkpoint": False,
        },
        OUT_CHECKPOINT_PATH,
    )
    print(f"Saved final (epoch 60) checkpoint to {OUT_CHECKPOINT_PATH}")

    print(f"\nBest val_acc across all 60 epochs: {best_val_acc:.4f} at epoch {best_epoch}")
    torch.save(
        {
            "classification_layer_state_dict": best_state_dict,
            "num_classes": NUM_MFCAD_CLASSES,
            "hidden_size": None,
            "class_weighted": True,
            "saved_epoch": best_epoch,
            "saved_epoch_val_acc": best_val_acc,
            "is_best_checkpoint": True,
        },
        OUT_BEST_CHECKPOINT_PATH,
    )
    print(f"Saved best-epoch checkpoint to {OUT_BEST_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
