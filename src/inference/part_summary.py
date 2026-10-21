"""
Per-class summary of a part's faces: face count, approximate instance
count, mean confidence, and a reliability label for that class.

"Approximate instances": faces of the same predicted (or true) class
that are adjacent (share an edge) are grouped into one instance via
connected components over the face-adjacency graph. This is a real
geometric check (via occwl.graph.face_adjacency), not a guess, but it
is still approximate: two physically separate instances of the same
feature that happen to share a face by coincidence of tessellation, or
one instance whose faces got split into disconnected predicted classes
by a misclassification, will not be counted correctly. Treat instance
counts as indicative, not exact.

Reliability is derived from outputs/final/per_class_report.txt's
PER-CLASS ACCURACY table (validation recall per class) -- not
hardcoded, so it stays in sync if that report is ever regenerated:
    recall < 0.40  -> low
    recall < 0.65  -> medium
    otherwise      -> high
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PER_CLASS_REPORT_PATH = PROJECT_ROOT / "outputs" / "final" / "per_class_report.txt"

RELIABILITY_LOW_MAX = 0.40
RELIABILITY_MEDIUM_MAX = 0.65

_ROW_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+\.\d+)\s")


@dataclass
class ClassSummary:
    class_index: int
    class_name: str
    face_count: int
    instance_count: Optional[int]  # None if adjacency couldn't be computed
    mean_confidence: Optional[float]  # None for ground truth (no confidence)
    reliability: str  # "low", "medium", "high", or "unknown"
    validation_recall: Optional[float]


def _reliability_for_recall(recall: Optional[float]) -> str:
    if recall is None:
        return "unknown"
    if recall < RELIABILITY_LOW_MAX:
        return "low"
    if recall < RELIABILITY_MEDIUM_MAX:
        return "medium"
    return "high"


def get_class_reliabilities() -> Dict[int, float]:
    """Parses the PER-CLASS ACCURACY table from
    outputs/final/per_class_report.txt and returns {class_index: recall}.
    Raises if the file is missing or the table can't be found -- a
    silent empty result here would make every class look "unknown"
    without any visible reason, which is worse than failing loudly."""
    if not PER_CLASS_REPORT_PATH.exists():
        raise FileNotFoundError(f"Per-class report not found: {PER_CLASS_REPORT_PATH}")

    text = PER_CLASS_REPORT_PATH.read_text(encoding="utf-8", errors="ignore")
    if "PER-CLASS ACCURACY" not in text:
        raise ValueError(f"'PER-CLASS ACCURACY' section not found in {PER_CLASS_REPORT_PATH}")

    section = text.split("PER-CLASS ACCURACY", 1)[1]
    recalls = {}
    for line in section.splitlines():
        m = _ROW_RE.match(line)
        if m:
            idx, _name, _n, _correct, acc = m.groups()
            recalls[int(idx)] = float(acc)

    if not recalls:
        raise ValueError(f"No per-class accuracy rows parsed from {PER_CLASS_REPORT_PATH}")
    return recalls


def _connected_components(num_faces: int, same_class_edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    """Plain union-find connected components, no external graph library
    required beyond what's already used to build the edge list."""
    parent = list(range(num_faces))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in same_class_edges:
        union(a, b)

    groups: Dict[int, List[int]] = {}
    for face_idx in range(num_faces):
        groups.setdefault(find(face_idx), []).append(face_idx)
    return list(groups.values())


def summarize_classes(
    class_per_face: Sequence[int],
    feat_names: Sequence[str],
    confidence_per_face: Optional[Sequence[float]] = None,
    adjacency_edges: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[ClassSummary]:
    """Builds one ClassSummary per class present on this part (classes
    with zero faces are omitted). confidence_per_face=None for ground
    truth (no model confidence exists). adjacency_edges=None skips
    instance counting (reported as None, not guessed at)."""
    num_faces = len(class_per_face)
    reliabilities = get_class_reliabilities()

    faces_by_class: Dict[int, List[int]] = {}
    for face_idx, cls in enumerate(class_per_face):
        faces_by_class.setdefault(cls, []).append(face_idx)

    summaries = []
    for cls in sorted(faces_by_class):
        face_indices = faces_by_class[cls]
        face_count = len(face_indices)

        instance_count = None
        if adjacency_edges is not None:
            face_set = set(face_indices)
            same_class_edges = [(a, b) for a, b in adjacency_edges if a in face_set and b in face_set]
            local_index = {f: i for i, f in enumerate(face_indices)}
            local_edges = [(local_index[a], local_index[b]) for a, b in same_class_edges]
            instance_count = len(_connected_components(face_count, local_edges))

        mean_confidence = None
        if confidence_per_face is not None:
            mean_confidence = sum(confidence_per_face[i] for i in face_indices) / face_count

        recall = reliabilities.get(cls)
        summaries.append(ClassSummary(
            class_index=cls,
            class_name=feat_names[cls],
            face_count=face_count,
            instance_count=instance_count,
            mean_confidence=mean_confidence,
            reliability=_reliability_for_recall(recall),
            validation_recall=recall,
        ))

    return summaries
