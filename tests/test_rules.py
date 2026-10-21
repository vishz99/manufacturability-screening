"""
Tests for src/screening/rules.py. Uses stdlib unittest (no pytest
install required -- only streamlit/plotly were added to the
environment for this feature).

Run with the `brepnet` conda environment's python:
    <path to conda>/envs/brepnet/python.exe -m unittest tests.test_rules -v
"""

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.mfcad_loader import FEAT_NAMES, NUM_CLASSES  # noqa: E402
from screening.rules import (  # noqa: E402
    DISCLAIMER,
    RULES,
    classes_covered_by_rules,
    screen_part,
)


@dataclass
class _FakeFacePrediction:
    """Minimal stand-in for inference.predict_part.FacePrediction --
    screen_part() only reads .predicted_class and .confidence."""
    predicted_class: int
    confidence: float


class TestRuleTableCoverage(unittest.TestCase):
    def test_all_16_classes_covered(self):
        covered = classes_covered_by_rules()
        self.assertEqual(covered, set(range(NUM_CLASSES)))

    def test_stock_has_no_flag_rule(self):
        stock_idx = FEAT_NAMES.index("stock")
        stock_rules = [r for r in RULES if stock_idx in r["classes"]]
        self.assertEqual(len(stock_rules), 1)
        self.assertEqual(stock_rules[0]["severity"], "none")

    def test_every_rule_has_a_severity_and_message(self):
        for rule in RULES:
            self.assertIn("severity", rule)
            self.assertIn(rule["severity"], {"review", "info", "none"})
            self.assertTrue(rule["message"])

    def test_disclaimer_is_nonempty(self):
        self.assertTrue(DISCLAIMER.strip())


class TestScreenPart(unittest.TestCase):
    def test_no_faces_no_flags(self):
        self.assertEqual(screen_part([]), [])

    def test_stock_only_produces_no_flags(self):
        stock_idx = FEAT_NAMES.index("stock")
        preds = [_FakeFacePrediction(stock_idx, 0.99) for _ in range(5)]
        self.assertEqual(screen_part(preds), [])

    def test_below_threshold_produces_no_flag(self):
        idx = FEAT_NAMES.index("rectangular_pocket")
        preds = [_FakeFacePrediction(idx, 0.4)]
        flags = screen_part(preds, threshold=0.5)
        self.assertEqual(flags, [])

    def test_at_or_above_threshold_fires(self):
        idx = FEAT_NAMES.index("rectangular_pocket")
        preds = [_FakeFacePrediction(idx, 0.5)]
        flags = screen_part(preds, threshold=0.5)
        self.assertTrue(len(flags) > 0)
        self.assertTrue(all(f.class_index == idx for f in flags))

    def test_pocket_triggers_both_blind_and_corner_radius_rules(self):
        idx = FEAT_NAMES.index("rectangular_pocket")
        preds = [_FakeFacePrediction(idx, 0.9)]
        flags = screen_part(preds, threshold=0.5)
        messages = {f.message for f in flags}
        self.assertTrue(any("depth control" in m for m in messages))
        self.assertTrue(any("corner radius" in m for m in messages))

    def test_low_reliability_class_worded_as_possible(self):
        # rectangular_blind_step has validation recall 0.1512 -> "low"
        idx = FEAT_NAMES.index("rectangular_blind_step")
        preds = [_FakeFacePrediction(idx, 0.9)]
        flags = screen_part(preds, threshold=0.5)
        self.assertTrue(flags)
        for f in flags:
            self.assertEqual(f.reliability, "low")
            self.assertTrue(f.message.startswith("Possible "))

    def test_high_reliability_class_not_worded_as_possible(self):
        # rectangular_through_slot has validation recall 0.9677 -> "high"
        idx = FEAT_NAMES.index("rectangular_through_slot")
        preds = [_FakeFacePrediction(idx, 0.9)]
        flags = screen_part(preds, threshold=0.5)
        self.assertTrue(flags)
        for f in flags:
            self.assertEqual(f.reliability, "high")
            self.assertFalse(f.message.startswith("Possible "))


if __name__ == "__main__":
    unittest.main()
