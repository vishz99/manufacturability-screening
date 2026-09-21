"""
Phase 2 deliverable: concrete proof that
  (a) BRepNet's pretrained backbone loads correctly and is frozen --
      requires_grad=False for every backbone parameter, and after a
      backward pass those parameters' .grad stays None, and
  (b) the newly-attached 16-class head is trainable -- requires_grad=True,
      and after the same backward pass its .grad is populated and nonzero.

Builds ONE real batch from an already-extracted MFCAD file (via
BRepNet's own BRepNetDataset + brepnet_collate_fn, run in "test_set"
mode so no .seg label file is required -- MFCAD's .face_truth labels
aren't in BRepNet's .seg format anyway, and this check doesn't need
real labels), then runs it through the model's actual forward path.

IMPORTANT: this calls model.create_face_embeddings(...) directly, NOT
model.forward(...). BRepNet.forward()'s signature (Xf, Xe, Xc, Kf, Ke,
Kc, Ce, Cf, Csf -- 9 args) does not match create_face_embeddings'
actual signature (Xf, Gf, Xe, Ge, Xc, Gc, Kf, Ke, Kc, Ce, Cf, Csf -- 12
args, point grids included) that it calls internally. forward() looks
like dead/broken code in the upstream repo -- BRepNet's own training
and eval code (brepnet_step in models/brepnet.py) never calls
forward(), it calls create_face_embeddings(...) directly with the
correct 12 args. We do the same here.

Run with the `brepnet` conda environment's python:
    <path to conda>/envs/brepnet/python.exe scripts/verify_freeze.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BREPNET_DIR = PROJECT_ROOT / "external" / "BRepNet"
sys.path.insert(0, str(BREPNET_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FEATURES_DIR = PROJECT_ROOT / "outputs" / "brepnet_features"
FEATURE_STANDARDIZATION_PATH = (
    BREPNET_DIR / "example_files" / "feature_standardization" / "s2.0.0_step_all_features.json"
)
KERNEL_PATH = BREPNET_DIR / "kernels" / "winged_edge.json"
INPUT_FEATURES_PATH = BREPNET_DIR / "feature_lists" / "all.json"

NUM_MFCAD_CLASSES = 16


def build_one_file_dataset_file(stem, out_path):
    standardization = json.loads(FEATURE_STANDARDIZATION_PATH.read_text())
    dataset = {
        "feature_standardization": standardization["feature_standardization"],
        "test_set": [stem],
    }
    out_path.write_text(json.dumps(dataset))


def main():
    from model.brepnet_backbone import load_frozen_backbone_with_new_head
    from dataloaders.brepnet_dataset import BRepNetDataset, brepnet_collate_fn

    npz_files = sorted(FEATURES_DIR.glob("*.npz"))
    assert npz_files, f"No extracted features found in {FEATURES_DIR}. Run experiments/500_shared_data_prep/extract_features.py first."
    stem = npz_files[0].stem
    print(f"Using extracted file: {stem}")

    dataset_file_path = FEATURES_DIR / "_verify_freeze_dataset.json"
    build_one_file_dataset_file(stem, dataset_file_path)

    opts = argparse.Namespace(
        dataset_dir=str(FEATURES_DIR),
        dataset_file=str(dataset_file_path),
        label_dir=None,
        kernel=str(KERNEL_PATH),
        input_features=str(INPUT_FEATURES_PATH),
    )
    dataset = BRepNetDataset(opts, "test_set")
    batch = brepnet_collate_fn([dataset[0]])
    print(f"Batch built: {batch['face_features'].shape[0]} faces, "
          f"{batch['edge_features'].shape[0]} edges, "
          f"{batch['coedge_features'].shape[0]} coedges")

    model = load_frozen_backbone_with_new_head(NUM_MFCAD_CLASSES)
    model.train()

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
    print(f"logits shape: {tuple(logits.shape)}  (expect [num_faces_in_batch, {NUM_MFCAD_CLASSES}])")
    assert logits.shape[1] == NUM_MFCAD_CLASSES

    dummy_loss = logits.sum()
    dummy_loss.backward()

    backbone_params = [(n, p) for n, p in model.named_parameters() if not n.startswith("classification_layer")]
    head_params = [(n, p) for n, p in model.named_parameters() if n.startswith("classification_layer")]

    n_backbone_requires_grad = sum(1 for _, p in backbone_params if p.requires_grad)
    n_backbone_with_grad = sum(1 for _, p in backbone_params if p.grad is not None)

    print(f"\nBackbone params: {len(backbone_params)} tensors")
    print(f"  requires_grad=True count (expect 0): {n_backbone_requires_grad}")
    print(f"  .grad is not None count (expect 0):  {n_backbone_with_grad}")

    print(f"\nNew head params: {len(head_params)} tensors")
    for n, p in head_params:
        has_grad = p.grad is not None
        grad_nonzero = has_grad and bool(torch.any(p.grad != 0))
        print(f"  {n}: shape={tuple(p.shape)} requires_grad={p.requires_grad} "
              f"grad_is_not_none={has_grad} grad_has_nonzero_entries={grad_nonzero}")

    assert n_backbone_requires_grad == 0, "FAIL: some backbone params are trainable"
    assert n_backbone_with_grad == 0, "FAIL: some backbone params received a gradient"
    assert all(p.requires_grad for _, p in head_params), "FAIL: new head params are not trainable"
    assert all(
        p.grad is not None and bool(torch.any(p.grad != 0)) for _, p in head_params
    ), "FAIL: new head params did not receive a nonzero gradient"

    print("\nFREEZE VERIFICATION PASSED: backbone received zero gradient and stayed frozen; "
          "new head received a nonzero gradient and is trainable.")


if __name__ == "__main__":
    main()
