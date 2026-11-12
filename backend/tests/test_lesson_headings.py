"""The lesson list carries each lesson's plain-text headings for search.

Run from the repository root:

    python3 backend/tests/test_lesson_headings.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app import curriculum  # noqa: E402
from backend.app.config import get_settings  # noqa: E402


LESSON = """---
title: Alpha
slug: 001-alpha
part: "Part 1 — Sound and signals"
draft: false
---

# Alpha

A fixture paragraph.

## First *section*

```python
## not a heading
```

### A `code` subsection with $x$

## Second section
"""


class LessonHeadingsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        lessons = root / "content" / "lessons"
        lessons.mkdir(parents=True)
        (lessons / "001-alpha.md").write_text(LESSON, encoding="utf-8")
        self._old_data = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = str(root / "data")
        get_settings.cache_clear()
        settings = get_settings()
        self._content = settings.content_dir
        settings.content_dir = root / "content"
        curriculum.reload_content()

    def tearDown(self) -> None:
        get_settings().content_dir = self._content
        curriculum.reload_content()
        get_settings.cache_clear()
        if self._old_data is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self._old_data
        self.tmp.cleanup()

    def test_lessons_list_plain_text_headings(self) -> None:
        listed = {lesson["slug"]: lesson for lesson in curriculum.lesson_list()}
        self.assertEqual(
            listed["001-alpha"]["headings"],
            [
                {"text": "First section", "level": 2},
                {"text": "A code subsection with x", "level": 3},
                {"text": "Second section", "level": 2},
            ],
        )
        self.assertNotIn("body", listed["001-alpha"])


if __name__ == "__main__":
    unittest.main()
