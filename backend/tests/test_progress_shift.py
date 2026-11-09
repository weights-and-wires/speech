"""A progress row stored under a pre-shift slug follows the new slug.

Run from the repository root:

    python3 backend/tests/test_progress_shift.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app import db  # noqa: E402
from backend.app.config import get_settings  # noqa: E402


class ProgressShiftTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self._old_data = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = self.tmp.name
        get_settings.cache_clear()
        db.init()

    def tearDown(self) -> None:
        get_settings.cache_clear()
        if self._old_data is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self._old_data
        self.tmp.cleanup()

    def test_pre_shift_slug_resolves_to_the_new_slug(self) -> None:
        db.set_progress("ada", "002-beta", "solved")
        stored = db.get_progress("ada")["002-beta"]["updated_at"]
        carried = db.carry_progress({"002-beta": ["003-beta"]})
        self.assertEqual(carried, 1)
        progress = db.get_progress("ada")
        self.assertEqual(progress["003-beta"]["status"], "solved")
        self.assertEqual(progress["003-beta"]["updated_at"], stored)
        self.assertNotIn("002-beta", progress)
        self.assertEqual(db.carry_progress({"002-beta": ["003-beta"]}), 0)
        self.assertEqual(db.get_progress("ada")["003-beta"]["status"], "solved")

    def test_older_rename_follows_a_shifted_value(self) -> None:
        db.set_progress("ada", "old-course-lesson", "started")
        db.carry_progress(
            {
                "old-course-lesson": ["001-alpha", "003-beta"],
                "002-beta": ["003-beta"],
            }
        )
        progress = db.get_progress("ada")
        self.assertEqual(progress["001-alpha"]["status"], "started")
        self.assertEqual(progress["003-beta"]["status"], "started")
        self.assertNotIn("old-course-lesson", progress)


if __name__ == "__main__":
    unittest.main()
