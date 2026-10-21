"""
Orchestration layer: runs prediction, ground-truth comparison (when
available) and rule screening for one STEP file, in the shape both
scripts/analyse_part.py (CLI) and app.py (Streamlit) consume.

Not a new analysis technique -- just wires together
src/inference/predict_part.py, src/inference/part_summary.py,
src/inference/ground_truth.py and src/screening/rules.py.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.mfcad_loader import FEAT_NAMES  # noqa: E402
from inference.ground_truth import GroundTruthPart, ground_truth_summary, load_ground_truth  # noqa: E402
from inference.part_summary import ClassSummary, summarize_classes  # noqa: E402
from inference.predict_part import PartPrediction, predict_step_file  # noqa: E402
from screening.rules import DEFAULT_CONFIDENCE_THRESHOLD, Flag, screen_part  # noqa: E402


@dataclass
class PartAnalysis:
    step_path: Path
    prediction: PartPrediction
    predicted_summary: List[ClassSummary]
    flags: List[Flag]
    confidence_threshold: float
    ground_truth: Optional[GroundTruthPart]
    ground_truth_summary_: Optional[List[ClassSummary]]
    accuracy: Optional[float]  # correct faces / total, only if ground truth exists


def has_ground_truth(step_path: Path) -> bool:
    return Path(step_path).with_suffix(".face_truth").exists()


def analyse_part(step_path: Path, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> PartAnalysis:
    """Runs the full pipeline for one STEP file. Ground truth is
    included automatically when a sibling .face_truth file exists
    (true for every MFCAD file, never true for an uploaded part)."""
    step_path = Path(step_path)
    prediction = predict_step_file(step_path)

    predicted_classes = [fp.predicted_class for fp in prediction.predictions]
    confidences = [fp.confidence for fp in prediction.predictions]
    predicted_summary = summarize_classes(
        class_per_face=predicted_classes,
        feat_names=FEAT_NAMES,
        confidence_per_face=confidences,
        adjacency_edges=prediction.adjacency_edges,
    )

    flags = screen_part(prediction.predictions, threshold=confidence_threshold)

    gt = None
    gt_summary = None
    accuracy = None
    if has_ground_truth(step_path):
        gt = load_ground_truth(step_path)
        if gt.num_faces == prediction.num_faces:
            gt_summary = ground_truth_summary(gt)
            correct = sum(1 for p, t in zip(predicted_classes, gt.true_labels) if p == t)
            accuracy = correct / gt.num_faces

    return PartAnalysis(
        step_path=step_path,
        prediction=prediction,
        predicted_summary=predicted_summary,
        flags=flags,
        confidence_threshold=confidence_threshold,
        ground_truth=gt,
        ground_truth_summary_=gt_summary,
        accuracy=accuracy,
    )
