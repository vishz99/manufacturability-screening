"""
Runs the accepted model on an arbitrary STEP file and returns per-face
predictions.

Pitfalls this reuses solutions for (see docs/project-summary.md and the
modules referenced below for where each was found):
  - model.forward() is broken upstream; always call
    create_face_embeddings(...) with its real 12 positional args
    (src/model/brepnet_backbone.py).
  - the backbone must stay in .eval() at all times, even during any
    future training, because its BatchNorm layers drift in .train()
    mode regardless of requires_grad (scripts/train_head.py).
  - never re-sort faces ourselves -- the npz row order already equals
    occwl's Compound.faces() order (src/data/mfcad_loader.py,
    notebooks/verify_extraction_face_order.py). The only re-ordering
    applied here is BRepNetDataset's own small/big-face permutation,
    undone via old_to_new_face_indices exactly as
    notebooks/inspect_predictions.ipynb already does.
  - a new part must be standardized with the ORIGINAL TRAINING feature
    statistics (s2.0.0_step_all_features.json), never statistics
    computed from the new part itself -- the model was trained on
    features normalized against the training distribution, not any one
    part's own range.
  - extract_brepnet_data_from_step globs a whole folder, so a single
    file is isolated into its own temp directory first
    (src/inference/_extract_worker.py, same pattern as
    scripts/extract_features.py).

A malformed STEP file can segfault pythonocc outright, not just raise a
Python exception, so all occwl-touching work for an arbitrary/untrusted
file runs in a subprocess (_extract_worker.py). This module only reads
back the worker's plain-data output (npz arrays, a JSON edge list) and
never touches occwl's geometry kernel on the caller's file directly.

Requires the `brepnet` conda environment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BREPNET_DIR = PROJECT_ROOT / "external" / "BRepNet"
sys.path.insert(0, str(BREPNET_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

WORKER_SCRIPT = Path(__file__).resolve().parent / "_extract_worker.py"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "final" / "mfcad_segmentation_head.pt"
FEATURE_STANDARDIZATION_PATH = (
    BREPNET_DIR / "example_files" / "feature_standardization" / "s2.0.0_step_all_features.json"
)
KERNEL_PATH = BREPNET_DIR / "kernels" / "winged_edge.json"
INPUT_FEATURES_PATH = BREPNET_DIR / "feature_lists" / "all.json"

NUM_MFCAD_CLASSES = 16
EXTRACTION_TIMEOUT_SECONDS = 180

# Fixed stem used for every single-file prediction, decoupling the
# BRepNetDataset "test_set" entry from the real (possibly messy/user
# uploaded) filename -- the worker always writes "part.npz" regardless
# of the original name.
_SINGLE_FILE_STEM = "part"


class PartAnalysisError(Exception):
    """Raised when a STEP file can't be read or processed. Safe to show
    the message directly to a user (st.error) -- it never contains a
    raw traceback from inside the subprocess, only a short reason."""


@dataclass
class FacePrediction:
    face_index: int
    predicted_class: int
    confidence: float
    probabilities: List[float]  # length NUM_MFCAD_CLASSES


@dataclass
class PartPrediction:
    step_path: Path
    num_faces: int
    predictions: List[FacePrediction]  # predictions[i] is for face i, canonical order
    tess_verts: np.ndarray  # [V, 3]
    tess_tris: np.ndarray  # [T, 3] vertex indices
    tess_face_of_tri: np.ndarray  # [T] face index per triangle
    adjacency_edges: List[tuple]  # (face_i, face_j) pairs sharing an edge


_model_cache = {}


def _load_model():
    """Loads and caches the accepted model. Reconstructs the architecture
    from the checkpoint's own saved metadata (hidden_size/dropout), same
    as scripts/report_per_class_stats.py -- never assumed."""
    if "model" in _model_cache:
        return _model_cache["model"]

    from model.brepnet_backbone import load_frozen_backbone_with_new_head

    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
    model = load_frozen_backbone_with_new_head(
        NUM_MFCAD_CLASSES,
        hidden_size=checkpoint["hidden_size"],
        dropout=checkpoint.get("dropout"),
    )
    model.classification_layer.load_state_dict(checkpoint["classification_layer_state_dict"])
    model.eval()
    _model_cache["model"] = model
    _model_cache["checkpoint_meta"] = {
        "saved_epoch": checkpoint["saved_epoch"],
        "saved_epoch_val_acc": checkpoint["saved_epoch_val_acc"],
    }
    return model


def checkpoint_metadata() -> dict:
    """Returns the accepted checkpoint's saved epoch/val_acc, loading the
    model (and populating the cache) if needed."""
    _load_model()
    return _model_cache["checkpoint_meta"]


def _run_extraction_worker(step_path: Path, output_dir: Path) -> int:
    """Runs _extract_worker.py as a subprocess. Returns the number of
    faces on success. Raises PartAnalysisError with a clean message on
    any failure, including a worker crash/segfault (nonzero exit with
    no parseable JSON)."""
    try:
        result = subprocess.run(
            [sys.executable, str(WORKER_SCRIPT), str(step_path), str(output_dir)],
            capture_output=True,
            text=True,
            timeout=EXTRACTION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise PartAnalysisError(
            f"Extraction timed out after {EXTRACTION_TIMEOUT_SECONDS}s -- the file may be too complex."
        )

    stdout_lines = [l for l in result.stdout.splitlines() if l.strip()]
    status = None
    if stdout_lines:
        try:
            status = json.loads(stdout_lines[-1])
        except json.JSONDecodeError:
            status = None

    if status is None:
        # The subprocess crashed (e.g. a pythonocc segfault) before it
        # could print its JSON status line -- this is exactly the
        # failure mode subprocess isolation exists to catch.
        raise PartAnalysisError(
            f"The STEP file could not be processed (reader crashed, exit code {result.returncode}). "
            "The file is likely malformed or not a valid STEP file."
        )

    if not status.get("ok"):
        raise PartAnalysisError(status.get("error", "Unknown extraction failure."))

    return status["num_faces"]


def predict_step_file(step_path: Path) -> PartPrediction:
    """Runs the full pipeline on one STEP file: isolated-subprocess
    extraction + tessellation + adjacency, then in-process model
    inference. Raises PartAnalysisError on any failure; never raises an
    occwl/pythonocc exception directly (those only ever happen inside
    the subprocess)."""
    step_path = Path(step_path)
    if not step_path.exists():
        raise PartAnalysisError(f"File not found: {step_path}")

    from dataloaders.brepnet_dataset import BRepNetDataset, brepnet_collate_fn

    with tempfile.TemporaryDirectory(prefix="mfg_screen_") as tmp:
        tmp_dir = Path(tmp)
        num_faces = _run_extraction_worker(step_path, tmp_dir)

        npz_path = tmp_dir / "part.npz"
        npz_data = dict(np.load(npz_path))
        tess_verts = npz_data["tess_verts"]
        tess_tris = npz_data["tess_tris"]
        tess_face_of_tri = npz_data["tess_face_of_tri"]
        adjacency = json.loads((tmp_dir / "adjacency.json").read_text())

        # Single-file "test_set" BRepNetDataset entry, same pattern as
        # notebooks/verify_freeze.py -- standardized with the ORIGINAL
        # training statistics, not anything computed from this part.
        standardization = json.loads(FEATURE_STANDARDIZATION_PATH.read_text())
        dataset_file_path = tmp_dir / "dataset_file.json"
        dataset_file_path.write_text(json.dumps({
            "feature_standardization": standardization["feature_standardization"],
            "test_set": [_SINGLE_FILE_STEM],
        }))

        opts = argparse.Namespace(
            dataset_dir=str(tmp_dir),
            dataset_file=str(dataset_file_path),
            label_dir=None,
            kernel=str(KERNEL_PATH),
            input_features=str(INPUT_FEATURES_PATH),
        )
        dataset = BRepNetDataset(opts, "test_set")
        item = dataset[0]
        batch = brepnet_collate_fn([item])

        model = _load_model()
        with torch.no_grad():
            face_embeddings = model.create_face_embeddings(
                batch["face_features"], batch["face_point_grids"],
                batch["edge_features"], batch["edge_point_grids"],
                batch["coedge_features"], batch["coedge_point_grids"],
                batch["face_kernel_tensor"], batch["edge_kernel_tensor"], batch["coedge_kernel_tensor"],
                batch["coedges_of_edges"], batch["coedges_of_small_faces"], batch["coedges_of_big_faces"],
            )
            logits_batch_order = model.classification_layer(face_embeddings)
            probs_batch_order = F.softmax(logits_batch_order, dim=1).numpy()

        # Undo BRepNetDataset's internal small/big-face permutation to get
        # back to canonical (npz/occwl) face order -- same gather as
        # notebooks/inspect_predictions.ipynb, not assumed to be identity.
        old_to_new = item["old_to_new_face_indices"].numpy()
        probs = probs_batch_order[old_to_new]

        if probs.shape[0] != num_faces:
            raise PartAnalysisError(
                f"Internal face-count mismatch after inference ({probs.shape[0]} vs {num_faces}) -- aborting rather than guessing."
            )

        predictions = []
        for face_idx in range(num_faces):
            p = probs[face_idx]
            cls = int(np.argmax(p))
            predictions.append(FacePrediction(
                face_index=face_idx,
                predicted_class=cls,
                confidence=float(p[cls]),
                probabilities=p.tolist(),
            ))

        return PartPrediction(
            step_path=step_path,
            num_faces=num_faces,
            predictions=predictions,
            tess_verts=tess_verts,
            tess_tris=tess_tris,
            tess_face_of_tri=tess_face_of_tri,
            adjacency_edges=[tuple(e) for e in adjacency["edges"]],
        )


if __name__ == "__main__":
    import argparse as _argparse

    parser = _argparse.ArgumentParser(description="Predict per-face machining-feature classes for a STEP file.")
    parser.add_argument("step_path", type=str)
    args = parser.parse_args()

    from data.mfcad_loader import FEAT_NAMES

    result = predict_step_file(Path(args.step_path))
    print(f"{result.num_faces} faces")
    for fp in result.predictions:
        print(f"  face {fp.face_index:3d}: {FEAT_NAMES[fp.predicted_class]:28s} confidence={fp.confidence:.3f}")
