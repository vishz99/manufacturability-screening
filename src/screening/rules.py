"""
Rule-based manufacturability flags, derived from the predicted
machining-feature class per face. These are indicative flags based on
feature type alone, not a dimensional or tolerance check -- see
DISCLAIMER below, which the app must always display alongside any
flags.

Rule table (covers all 16 MFCAD classes; a class can match more than
one rule, since these are independent manufacturing concerns):

  severity=review, depth control:
    blind features -- rectangular_blind_step, triangular_blind_step,
    rectangular_blind_slot, rectangular_pocket, triangular_pocket,
    6sides_pocket

  severity=review, internal corner radius must be at least the tool
  radius:
    pockets and blind slots -- rectangular_blind_slot,
    rectangular_pocket, triangular_pocket, 6sides_pocket

  severity=review, acute corners need small tools:
    triangular and 6-sided profiles -- triangular_through_slot,
    triangular_passage, 6sides_passage, triangular_blind_step,
    triangular_pocket, 6sides_pocket

  severity=review, multi-axis or extra setup:
    slanted and 2-sided through-steps -- slanted_through_step,
    2sides_through_step

  severity=info:
    through features and chamfer -- rectangular_through_slot,
    rectangular_passage, rectangular_through_step, chamfer

  no flag:
    stock
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.mfcad_loader import FEAT_NAMES  # noqa: E402
from inference.part_summary import get_class_reliabilities  # noqa: E402

DISCLAIMER = "Indicative only: feature-type flags from a 60%-accuracy model, no dimensional or tolerance checks."

DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# class index -> name, for readability of the table below
_N = {name: i for i, name in enumerate(FEAT_NAMES)}

RULES = [
    {
        "name": "blind features",
        "severity": "review",
        "message": "depth control",
        "classes": [
            _N["rectangular_blind_step"], _N["triangular_blind_step"],
            _N["rectangular_blind_slot"], _N["rectangular_pocket"],
            _N["triangular_pocket"], _N["6sides_pocket"],
        ],
    },
    {
        "name": "pockets and blind slots",
        "severity": "review",
        "message": "internal corner radius must be at least the tool radius",
        "classes": [
            _N["rectangular_blind_slot"], _N["rectangular_pocket"],
            _N["triangular_pocket"], _N["6sides_pocket"],
        ],
    },
    {
        "name": "triangular and 6-sided profiles",
        "severity": "review",
        "message": "acute corners need small tools",
        "classes": [
            _N["triangular_through_slot"], _N["triangular_passage"],
            _N["6sides_passage"], _N["triangular_blind_step"],
            _N["triangular_pocket"], _N["6sides_pocket"],
        ],
    },
    {
        "name": "slanted and 2-sided through-steps",
        "severity": "review",
        "message": "multi-axis or extra setup",
        "classes": [_N["slanted_through_step"], _N["2sides_through_step"]],
    },
    {
        "name": "through features and chamfer",
        "severity": "info",
        "message": "standard through-feature or edge-break, no special concern",
        "classes": [
            _N["rectangular_through_slot"], _N["rectangular_passage"],
            _N["rectangular_through_step"], _N["chamfer"],
        ],
    },
    {
        "name": "stock",
        "severity": "none",
        "message": "unmachined stock face",
        "classes": [_N["stock"]],
    },
]


@dataclass
class Flag:
    rule_name: str
    severity: str  # "review" or "info" (stock produces no Flag at all)
    class_index: int
    class_name: str
    message: str
    reliability: str
    max_confidence: float


def classes_covered_by_rules() -> set:
    """Every class index that appears in at least one rule -- used by
    tests/test_rules.py to confirm all 16 classes are covered."""
    covered = set()
    for rule in RULES:
        covered.update(rule["classes"])
    return covered


def screen_part(
    face_predictions: Sequence,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> List[Flag]:
    """Builds flags from a list of per-face predictions (objects with
    .predicted_class and .confidence, e.g. inference.predict_part.
    FacePrediction). A flag fires for a rule+class pair when at least
    one face of that class has confidence >= threshold. Low-reliability
    classes get their message worded as "Possible ..." rather than
    stated outright, since the model is known to be unreliable there.

    Stock and any class matching no rule never produce a Flag.
    """
    max_confidence_per_class = {}
    for fp in face_predictions:
        cls = fp.predicted_class
        if fp.confidence > max_confidence_per_class.get(cls, -1.0):
            max_confidence_per_class[cls] = fp.confidence

    reliabilities = get_class_reliabilities()

    flags: List[Flag] = []
    for rule in RULES:
        if rule["severity"] == "none":
            continue
        for cls in rule["classes"]:
            max_conf = max_confidence_per_class.get(cls)
            if max_conf is None or max_conf < threshold:
                continue

            recall = reliabilities.get(cls)
            reliability = "unknown" if recall is None else (
                "low" if recall < 0.40 else "medium" if recall < 0.65 else "high"
            )
            message = rule["message"]
            if reliability == "low":
                message = f"Possible {message}"

            flags.append(Flag(
                rule_name=rule["name"],
                severity=rule["severity"],
                class_index=cls,
                class_name=FEAT_NAMES[cls],
                message=message,
                reliability=reliability,
                max_confidence=max_conf,
            ))

    return flags


if __name__ == "__main__":
    print(f"{len(RULES)} rules, {len(classes_covered_by_rules())} classes covered")
    uncovered = set(range(len(FEAT_NAMES))) - classes_covered_by_rules()
    print(f"Uncovered classes: {uncovered if uncovered else 'none'}")
    print(f"\nDisclaimer: {DISCLAIMER}")
