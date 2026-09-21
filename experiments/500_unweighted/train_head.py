"""
HISTORICAL (500-file, unweighted experiment) -- not the canonical
pipeline. See scripts/train_head.py (--suffix C --epochs 40
--hidden-size 128 --save-best) for the accepted result. Kept for
reference/reproducibility only.

Trains ONLY the new 16-class MFCAD segmentation head on top of
the frozen BRepNet backbone (src/model/brepnet_backbone.py).

Reuses external/BRepNet's own BRepNetDataset/brepnet_collate_fn
unmodified for batch construction -- pointing --label_dir at
outputs/mfcad_labels_cache/ (plain-text .seg files, one per subset
file, produced by experiments/500_shared_data_prep/cache_labels.py)
lets BRepNetDataset's already-verified small/big-face permutation
logic (notebooks/verify_dataset_face_permutation.py) handle label
alignment for us, rather than reimplementing it.

Forward passes go through model.create_face_embeddings(...) directly
(never model.forward()/model(...) -- see src/model/brepnet_backbone.py
and notebooks/verify_freeze.py for why forward() is broken/unused code).

Run with the `brepnet` conda environment's python:
    <path to conda>/envs/brepnet/python.exe experiments/500_unweighted/train_head.py
"""

import json
import random
import sys
from pathlib import Path

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

SPLIT_DATASET_FILE_PATH = PROJECT_ROOT / "experiments" / "500_shared_data_prep" / "mfcad_train_val_split.json"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoint.pt"

NUM_MFCAD_CLASSES = 16
N_TRAIN = 400
N_VAL = 100
SPLIT_SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3


def build_train_val_split():
    stems = SUBSET_LIST.read_text().split()
    assert len(stems) == N_TRAIN + N_VAL, f"Expected {N_TRAIN + N_VAL} subset files, found {len(stems)}"

    rng = random.Random(SPLIT_SEED)
    shuffled = stems[:]
    rng.shuffle(shuffled)
    train_stems = sorted(shuffled[:N_TRAIN])
    val_stems = sorted(shuffled[N_TRAIN:])
    assert len(set(train_stems) & set(val_stems)) == 0, "Train/val overlap!"

    standardization = json.loads(FEATURE_STANDARDIZATION_PATH.read_text())
    dataset = {
        "feature_standardization": standardization["feature_standardization"],
        "train_set": train_stems,
        "val_set": val_stems,
    }
    SPLIT_DATASET_FILE_PATH.write_text(json.dumps(dataset, indent=2))
    print(f"Split: {len(train_stems)} train / {len(val_stems)} val (seed={SPLIT_SEED})")
    return SPLIT_DATASET_FILE_PATH


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
    """
    Concretely proves the optimizer only touches the new head's
    parameters -- not by assumption, by checking object identity between
    every tensor the optimizer holds and every backbone tensor in the
    model.
    """
    backbone_param_ids = {
        id(p) for n, p in model.named_parameters() if not n.startswith("classification_layer")
    }
    head_param_ids = {
        id(p) for n, p in model.named_parameters() if n.startswith("classification_layer")
    }

    optimizer_param_ids = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer_param_ids.add(id(p))

    print("\n--- Optimizer scoping verification ---")
    print(f"optimizer.param_groups: {len(optimizer.param_groups)} group(s), "
          f"{len(optimizer_param_ids)} tensor(s) total")
    overlap_with_backbone = optimizer_param_ids & backbone_param_ids
    overlap_with_head = optimizer_param_ids & head_param_ids
    print(f"  optimizer params that ARE backbone params (expect 0): {len(overlap_with_backbone)}")
    print(f"  optimizer params that ARE head params (expect {len(head_param_ids)}): {len(overlap_with_head)}")
    for group in optimizer.param_groups:
        for p in group["params"]:
            print(f"    optimizer tensor shape={tuple(p.shape)} requires_grad={p.requires_grad}")

    assert len(overlap_with_backbone) == 0, "FAIL: optimizer holds backbone parameters"
    assert overlap_with_head == head_param_ids, "FAIL: optimizer is missing some head parameters"
    print("OPTIMIZER SCOPING VERIFIED: only the new head's parameters are being optimized.")


def run_epoch(model, dataloader, optimizer, train):
    # The backbone is frozen (requires_grad=False), but that alone does NOT
    # stop BatchNorm layers inside surface_encoder/curve_encoder from
    # updating their running_mean/running_var buffers in .train() mode --
    # confirmed empirically (16/16 buffers changed after a single train-mode
    # forward pass with no optimizer step). So the model is kept in .eval()
    # mode for BOTH phases. This is safe for what we're actually training:
    # classification_layer is a plain nn.Linear, which has no train/eval-
    # dependent behavior (no dropout, no batchnorm), so this doesn't change
    # what the head learns -- it just stops the "frozen" backbone from
    # silently drifting.
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

            loss = F.cross_entropy(logits, labels, reduction="mean")

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

    dataset_file_path = build_train_val_split()
    opts = make_opts(dataset_file_path)

    train_dataset = BRepNetDataset(opts, "train_set")
    val_dataset = BRepNetDataset(opts, "val_set")
    print(f"train_dataset: {len(train_dataset)} solids, val_dataset: {len(val_dataset)} solids")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=brepnet_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=brepnet_collate_fn)

    model = load_frozen_backbone_with_new_head(NUM_MFCAD_CLASSES)
    optimizer = torch.optim.Adam(model.classification_layer.parameters(), lr=LEARNING_RATE)

    verify_optimizer_scoping(model, optimizer)

    print(f"\n--- Training for {NUM_EPOCHS} epochs (batch_size={BATCH_SIZE}, lr={LEARNING_RATE}) ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, n_train_faces = run_epoch(model, train_loader, optimizer, train=True)
        val_loss, val_acc, n_val_faces = run_epoch(model, val_loader, optimizer, train=False)
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
        },
        CHECKPOINT_PATH,
    )
    print(f"\nSaved trained head checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
