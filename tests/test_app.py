"""
Smoke test for app.py using Streamlit's built-in headless AppTest
(streamlit.testing.v1) -- runs the script without a browser or server
and inspects what it rendered. Not a replacement for actually running
`streamlit run app.py` and clicking around, just a fast regression
check that the script doesn't crash and produces the expected
structure for one validation part.

Run with the `brepnet` conda environment's python:
    <path to conda>/envs/brepnet/python.exe -m unittest tests.test_app -v
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(PROJECT_ROOT / "app.py")
APP_TEST_TIMEOUT = 120  # seconds -- first run loads the model + runs a real inference


class TestApp(unittest.TestCase):
    def test_app_runs_without_exception_on_default_validation_part(self):
        at = AppTest.from_file(APP_PATH, default_timeout=APP_TEST_TIMEOUT)
        at.run()
        self.assertFalse(at.exception, f"app.py raised: {at.exception}")

    def test_app_shows_a_summary_table_for_the_default_part(self):
        at = AppTest.from_file(APP_PATH, default_timeout=APP_TEST_TIMEOUT)
        at.run()
        self.assertFalse(at.exception)
        # "Validation part" is the default radio choice, so a part should
        # already be analysed with no further interaction required.
        self.assertTrue(len(at.dataframe) >= 1, "expected a summary dataframe to be rendered")

    def test_app_shows_the_disclaimer(self):
        at = AppTest.from_file(APP_PATH, default_timeout=APP_TEST_TIMEOUT)
        at.run()
        self.assertFalse(at.exception)
        captions = [c.value for c in at.caption]
        self.assertTrue(any("Indicative only" in c for c in captions), "disclaimer caption not found")


if __name__ == "__main__":
    unittest.main()
