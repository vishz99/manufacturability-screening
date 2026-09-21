"""
Loads BRepNet's pretrained backbone (frozen) with a freshly-initialized
segmentation head sized for MFCAD's 16 feature classes.

The pretrained checkpoint (external/BRepNet/example_files/pretrained_models/
pretrained_s2.0.0_extended_step_uv_net_features_0816_183419.ckpt) was
trained on Fusion 360 Gallery's 8-class segmentation task -- confirmed
directly from external/BRepNet/example_files/pretrained_models/
segment_names.json: ["ExtrudeSide", "ExtrudeEnd", "CutSide", "CutEnd",
"Fillet", "Chamfer", "RevolveSide", "RevolveEnd"]. MFCAD uses a different
16-class taxonomy (see src/data/mfcad_loader.py's FEAT_NAMES), so we do
NOT reuse or fine-tune the old head.

Loading strategy:
  1. Load the checkpoint with its ORIGINAL num_classes=8 architecture,
     so BRepNet.load_from_checkpoint's strict state_dict load succeeds
     (a mismatched classification_layer shape would make it fail).
  2. Freeze every parameter (requires_grad=False).
  3. Discard the old 8-class classification_layer and attach a fresh
     nn.Linear(num_filters, num_classes) -- a new module's parameters
     default to requires_grad=True, so this new head is trainable.

Requires the `occwl` / `pythonocc-core` / pytorch_lightning stack (the
`brepnet` conda environment), not this project's own .venv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BREPNET_DIR = PROJECT_ROOT / "external" / "BRepNet"
if str(BREPNET_DIR) not in sys.path:
    sys.path.insert(0, str(BREPNET_DIR))

# Importing this module applies BRepNet's torch.load(weights_only=False)
# monkeypatch as a side effect -- needed because PyTorch >=2.6 defaults
# torch.load to weights_only=True, which rejects the pytorch_lightning
# classes pickled into BRepNet's official checkpoint. The checkpoint is
# a trusted, first-party release.
import eval.evaluate_folder  # noqa: F401,E402
from eval.evaluate_folder import get_argument_parser  # noqa: E402
from models.brepnet import BRepNet  # noqa: E402

PRETRAINED_CHECKPOINT = (
    BREPNET_DIR
    / "example_files"
    / "pretrained_models"
    / "pretrained_s2.0.0_extended_step_uv_net_features_0816_183419.ckpt"
)
SEGMENT_NAMES_FILE = BREPNET_DIR / "example_files" / "pretrained_models" / "segment_names.json"
ORIGINAL_NUM_CLASSES = 8  # the checkpoint's original Fusion 360 Gallery task


def _build_opts(num_classes: int):
    """
    Builds a BRepNet opts namespace using the same argument parser
    evaluate_folder.py uses, so architecture hyperparameters (num_filters,
    num_layers, etc.) default to exactly what the pretrained checkpoint
    was trained with -- this same defaults-only pattern is already
    proven to load this checkpoint correctly (used by
    run_batch_embeddings.py / evaluate_folder.py in the sibling Track A
    project). dataset_dir/dataset_file are required by the parser but
    unused at model-construction time as long as --segment_names points
    to a real file.
    """
    parser = get_argument_parser()
    return parser.parse_args(
        [
            "--dataset_dir", "unused",
            "--dataset_file", "unused",
            "--segment_names", str(SEGMENT_NAMES_FILE),
            "--num_classes", str(num_classes),
            # Explicit absolute paths -- the parser's own defaults
            # ("kernels/winged_edge.json", "feature_lists/all.json") are
            # relative to external/BRepNet and only resolve correctly if
            # the process cwd happens to be that directory.
            "--kernel", str(BREPNET_DIR / "kernels" / "winged_edge.json"),
            "--input_features", str(BREPNET_DIR / "feature_lists" / "all.json"),
        ]
    )


def load_frozen_backbone_with_new_head(num_classes: int, checkpoint_path: Path = PRETRAINED_CHECKPOINT) -> BRepNet:
    """
    Returns a BRepNet model with the pretrained backbone frozen and a
    fresh, trainable classification_layer sized for `num_classes`.
    """
    opts = _build_opts(ORIGINAL_NUM_CLASSES)
    model = BRepNet.load_from_checkpoint(str(checkpoint_path), opts=opts)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classification_layer.in_features
    model.classification_layer = nn.Linear(in_features, num_classes)

    return model
