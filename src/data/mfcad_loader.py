"""
Loader for the MFCAD dataset (github.com/hducg/MFCAD): pairs each STEP
file's occwl faces with its per-face machining-feature label.

IMPORTANT -- label alignment is NOT positional.
MFCAD's own dataset_visualizer.py (data/MFCAD/occ/dataset_visualizer.py)
does not trust the face order produced by traversing the shape (e.g.
occwl's `Compound.faces()` / `load_step(...)[0].faces()`). Instead it
reads an integer identifier embedded in each STEP face entity's `Name`
attribute (via STEPControl_Reader's TransferReader) and uses THAT to
index into the .face_truth label list.

We verified empirically (300-file random sample) that raw traversal
order differs from the embedded-name order in effectively every file,
while indexing by embedded name always gives a face count matching
len(labels) and a clean set of names == {0, ..., N-1} for a single-solid
file. So this loader always aligns by embedded name, never by position.

Requires the `occwl` / `pythonocc-core` stack (the `brepnet` conda
environment), not this project's own .venv.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from occwl.compound import Compound
from occwl.face import Face

# Class order taken verbatim from data/MFCAD/occ/dataset_visualizer.py
# (FEAT_NAMES). face_truth label integers index into this list.
FEAT_NAMES: List[str] = [
    "rectangular_through_slot",
    "triangular_through_slot",
    "rectangular_passage",
    "triangular_passage",
    "6sides_passage",
    "rectangular_through_step",
    "2sides_through_step",
    "slanted_through_step",
    "rectangular_blind_step",
    "triangular_blind_step",
    "rectangular_blind_slot",
    "rectangular_pocket",
    "triangular_pocket",
    "6sides_pocket",
    "chamfer",
    "stock",
]
NUM_CLASSES = len(FEAT_NAMES)


class MFCADAlignmentError(Exception):
    """Raised when a STEP/.face_truth pair cannot be safely aligned."""


@dataclass
class LabeledSample:
    stem: str
    step_path: Path
    compound: Compound
    faces: List[Face]  # occwl Face objects, in compound traversal order
    labels: List[int]  # labels[i] corresponds to faces[i]


def _label_path_for(step_path: Path) -> Path:
    return step_path.with_suffix(".face_truth")


def load_labeled_sample(step_path: Path) -> LabeledSample:
    """Load one MFCAD STEP file and its per-face labels, aligned by the
    face's embedded STEP entity name (not traversal order).

    Raises MFCADAlignmentError with a specific reason if the file's
    labels can't be safely aligned (count mismatch, unparseable/missing
    name, or the names aren't a clean permutation of range(len(labels))).
    """
    step_path = Path(step_path)
    label_path = _label_path_for(step_path)

    if not step_path.exists():
        raise MFCADAlignmentError(f"missing STEP file: {step_path}")
    if not label_path.exists():
        raise MFCADAlignmentError(f"missing .face_truth file: {label_path}")

    with open(label_path, "rb") as f:
        raw_labels = pickle.load(f)

    compound, attrs = Compound.load_step_with_attributes(str(step_path))
    faces = list(compound.faces())

    if len(faces) != len(raw_labels):
        raise MFCADAlignmentError(
            f"face count {len(faces)} != label count {len(raw_labels)}"
        )

    names: List[int] = []
    for face in faces:
        attr = attrs.get(face)
        if attr is None or not attr.get("name"):
            raise MFCADAlignmentError("face missing embedded STEP entity name")
        try:
            names.append(int(attr["name"]))
        except ValueError as e:
            raise MFCADAlignmentError(f"unparseable face name: {attr['name']!r}") from e

    if sorted(names) != list(range(len(raw_labels))):
        raise MFCADAlignmentError(
            f"face names are not a clean permutation of 0..{len(raw_labels) - 1}"
        )

    # Reorder: labels[i] must correspond to faces[i]. raw_labels is indexed
    # by the embedded name, so faces[i]'s label is raw_labels[names[i]].
    aligned_labels = [raw_labels[n] for n in names]

    bad = [lbl for lbl in aligned_labels if not (0 <= lbl < NUM_CLASSES)]
    if bad:
        raise MFCADAlignmentError(f"label value(s) out of range 0..{NUM_CLASSES - 1}: {bad}")

    return LabeledSample(
        stem=step_path.stem,
        step_path=step_path,
        compound=compound,
        faces=faces,
        labels=aligned_labels,
    )


def validate_pair(step_path: Path) -> "ValidationResult":
    """Lightweight pass/fail check (no geometry kept) for full-dataset scans."""
    try:
        sample = load_labeled_sample(step_path)
        return ValidationResult(stem=sample.stem, ok=True, num_faces=len(sample.faces), reason=None)
    except MFCADAlignmentError as e:
        return ValidationResult(stem=Path(step_path).stem, ok=False, num_faces=None, reason=str(e))


@dataclass
class ValidationResult:
    stem: str
    ok: bool
    num_faces: Optional[int]
    reason: Optional[str]
