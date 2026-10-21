"""
The same per-class summary as part_summary.py, but built from MFCAD's
true labels instead of model predictions -- for validation-split files
only (uploaded parts have no ground truth). Lets the app show
prediction vs. ground truth side by side.

Loads geometry directly via occwl (no subprocess isolation): MFCAD files
are a trusted, already-validated dataset (every file in it was checked
in Phase 1 -- see src/data/mfcad_loader.py), unlike an arbitrary
uploaded STEP file, so the segfault risk subprocess isolation guards
against in predict_part.py does not apply here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BREPNET_DIR = PROJECT_ROOT / "external" / "BRepNet"
if str(BREPNET_DIR) not in sys.path:
    sys.path.insert(0, str(BREPNET_DIR))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.mfcad_loader import FEAT_NAMES, load_labeled_sample  # noqa: E402
from inference.part_summary import ClassSummary, summarize_classes  # noqa: E402


@dataclass
class GroundTruthPart:
    step_path: Path
    num_faces: int
    true_labels: List[int]  # canonical (occwl raw traversal) order
    adjacency_edges: List[Tuple[int, int]]


def load_ground_truth(step_path: Path) -> GroundTruthPart:
    """Loads a validation-split STEP file's true per-face labels and
    face-adjacency graph. Faces/labels are in the same canonical order
    used everywhere else in this project (occwl Compound.faces()
    traversal order == npz row order, verified in
    notebooks/verify_extraction_face_order.py)."""
    from occwl.entity_mapper import EntityMapper
    from occwl.graph import face_adjacency

    step_path = Path(step_path)
    sample = load_labeled_sample(step_path)

    mapper = EntityMapper(sample.compound)
    occwl_idx_to_canonical = {
        mapper.face_index(f.topods_shape()): i for i, f in enumerate(sample.faces)
    }
    graph = face_adjacency(sample.compound)
    edges = []
    if graph is not None:
        for u, v in graph.edges():
            cu = occwl_idx_to_canonical.get(u)
            cv = occwl_idx_to_canonical.get(v)
            if cu is not None and cv is not None:
                edges.append((cu, cv))

    return GroundTruthPart(
        step_path=step_path,
        num_faces=len(sample.faces),
        true_labels=list(sample.labels),
        adjacency_edges=edges,
    )


def ground_truth_summary(gt: GroundTruthPart) -> List[ClassSummary]:
    """Same ClassSummary shape as part_summary.summarize_classes(), with
    confidence_per_face=None (there's no confidence for a ground-truth
    label) and reliability describing the MODEL's known recall for that
    class, not anything about the ground truth itself."""
    return summarize_classes(
        class_per_face=gt.true_labels,
        feat_names=FEAT_NAMES,
        confidence_per_face=None,
        adjacency_edges=gt.adjacency_edges,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Show ground-truth per-face labels for a validation STEP file.")
    parser.add_argument("step_path", type=str)
    args = parser.parse_args()

    gt = load_ground_truth(Path(args.step_path))
    print(f"{gt.num_faces} faces")
    for i, lbl in enumerate(gt.true_labels):
        print(f"  face {i:3d}: {FEAT_NAMES[lbl]}")

    print("\nClass summary:")
    for s in ground_truth_summary(gt):
        print(f"  [{s.class_index:2d}] {s.class_name:28s} faces={s.face_count} instances={s.instance_count} reliability={s.reliability}")
