"""
CLI: predicts per-face machining features for a STEP file, prints a
per-class summary and manufacturability flags, and optionally compares
against ground truth (automatically, when a sibling .face_truth file
exists -- true for MFCAD files, never true for an arbitrary upload).

Usage:
    <brepnet python> scripts/analyse_part.py <step_path> [--threshold 0.5] [--json] [--use-ground-truth]

--use-ground-truth: show the ground-truth class summary/accuracy
(already computed automatically if available); without this flag the
ground-truth section is simply omitted from the printed/JSON output,
even if it was available.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.mfcad_loader import FEAT_NAMES  # noqa: E402
from ui.analysis import analyse_part  # noqa: E402


def _flag_to_dict(f):
    return dataclasses.asdict(f)


def _summary_to_dict(s):
    return dataclasses.asdict(s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step_path", type=str)
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for flags (default 0.5)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a formatted report")
    parser.add_argument("--use-ground-truth", action="store_true", help="Include ground-truth comparison in the output, when available")
    args = parser.parse_args()

    analysis = analyse_part(Path(args.step_path), confidence_threshold=args.threshold)

    if args.json:
        output = {
            "step_path": str(analysis.step_path),
            "num_faces": analysis.prediction.num_faces,
            "confidence_threshold": analysis.confidence_threshold,
            "predicted_summary": [_summary_to_dict(s) for s in analysis.predicted_summary],
            "flags": [_flag_to_dict(f) for f in analysis.flags],
        }
        if args.use_ground_truth and analysis.ground_truth_summary_ is not None:
            output["accuracy"] = analysis.accuracy
            output["ground_truth_summary"] = [_summary_to_dict(s) for s in analysis.ground_truth_summary_]
        print(json.dumps(output, indent=2))
        return

    print(f"Part: {analysis.step_path}")
    print(f"Faces: {analysis.prediction.num_faces}")
    if args.use_ground_truth and analysis.accuracy is not None:
        print(f"Accuracy vs. ground truth: {analysis.accuracy:.4f}")

    print("\nPredicted feature summary:")
    print(f"{'class':30s} {'faces':>6s} {'instances':>10s} {'mean conf':>10s} {'reliability':>12s}")
    for s in analysis.predicted_summary:
        inst = s.instance_count if s.instance_count is not None else "n/a"
        conf = f"{s.mean_confidence:.3f}" if s.mean_confidence is not None else "n/a"
        print(f"{s.class_name:30s} {s.face_count:>6d} {str(inst):>10s} {conf:>10s} {s.reliability:>12s}")

    if args.use_ground_truth and analysis.ground_truth_summary_ is not None:
        print("\nGround-truth feature summary:")
        print(f"{'class':30s} {'faces':>6s} {'instances':>10s}")
        for s in analysis.ground_truth_summary_:
            inst = s.instance_count if s.instance_count is not None else "n/a"
            print(f"{s.class_name:30s} {s.face_count:>6d} {str(inst):>10s}")

    print(f"\nManufacturability flags (confidence >= {args.threshold}):")
    if not analysis.flags:
        print("  none")
    for f in analysis.flags:
        print(f"  [{f.severity:6s}] {f.class_name:28s} {f.message} (reliability={f.reliability}, max_conf={f.max_confidence:.2f})")

    from screening.rules import DISCLAIMER
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
