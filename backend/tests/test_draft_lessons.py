"""Draft lessons follow the built-frontend check in the curriculum loader.

Run from the repository root:

    python3 backend/tests/test_draft_lessons.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app import curriculum  # noqa: E402
from backend.app.config import get_settings  # noqa: E402


PUBLISHED = """---
title: Published
slug: 001-published
part: "Part 1 — Sound and signals"
draft: false
---

# Published
"""

DRAFT = """---
title: Draft
slug: 002-draft
part: "Part 1 — Sound and signals"
exercise: 002-draft
draft: true
---

# Draft
"""


class DraftLessonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.content = root / "content"
        lessons = self.content / "lessons"
        lessons.mkdir(parents=True)
        (lessons / "001-published.md").write_text(PUBLISHED, encoding="utf-8")
        (lessons / "002-draft.md").write_text(DRAFT, encoding="utf-8")
        exercise = self.content / "exercises" / "002-draft"
        exercise.mkdir(parents=True)
        (exercise / "meta.json").write_text(
            json.dumps({"id": "002-draft", "title": "Draft"}) + "\n",
            encoding="utf-8",
        )
        self.dist = root / "dist"
        self._old_data = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = str(root / "data")
        get_settings.cache_clear()
        settings = get_settings()
        self._content = settings.content_dir
        self._dist = settings.frontend_dist
        settings.content_dir = self.content
        settings.frontend_dist = self.dist
        curriculum.reload_content()

    def tearDown(self) -> None:
        settings = get_settings()
        settings.content_dir = self._content
        settings.frontend_dist = self._dist
        curriculum.reload_content()
        get_settings.cache_clear()
        if self._old_data is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self._old_data
        self.tmp.cleanup()

    def slugs(self) -> list[str]:
        curriculum.reload_content()
        return [lesson["slug"] for lesson in curriculum.load_lessons()]

    def test_drafts_are_visible_without_the_built_frontend(self) -> None:
        self.assertFalse(self.dist.exists())
        self.assertEqual(self.slugs(), ["001-published", "002-draft"])
        self.assertIsNotNone(curriculum.get_lesson("002-draft"))
        self.assertIn("002-draft", curriculum.load_exercises())

    def test_drafts_are_omitted_when_the_bundle_is_served(self) -> None:
        self.dist.mkdir()
        self.assertEqual(self.slugs(), ["001-published"])
        self.assertIsNone(curriculum.get_lesson("002-draft"))
        self.assertNotIn("002-draft", curriculum.load_exercises())
        self.assertEqual(len(curriculum.lesson_list()), 1)


if __name__ == "__main__":
    unittest.main()
