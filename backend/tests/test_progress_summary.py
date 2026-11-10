"""The hub's progress summary, and its path-scoped CORS allowance.

Run from the repository root:

    python3 backend/tests/test_progress_summary.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import curriculum, db  # noqa: E402
from backend.app.config import get_settings  # noqa: E402
from backend.app.main import app  # noqa: E402

HUB = "https://courses.ifkash.dev"
URL = "/api/progress/summary"
SECRET = "test-secret"
SIGNED_IN = {"x-sso-proxy": SECRET, "remote-user": "ada"}

LESSON = """---
title: {title}
slug: {slug}
part: "Part 1 — Sound and signals"
draft: {draft}
---

# {title}
"""


class ProgressSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        lessons = root / "content" / "lessons"
        lessons.mkdir(parents=True)
        for slug, title, draft in [
            ("001-alpha", "Alpha", "false"),
            ("002-beta", "Beta", "false"),
            ("003-gamma", "Gamma", "false"),
            ("004-hidden", "Hidden", "true"),
        ]:
            (lessons / f"{slug}.md").write_text(
                LESSON.format(slug=slug, title=title, draft=draft), encoding="utf-8"
            )
        # A built frontend hides draft lessons, as in production.
        (root / "dist").mkdir()
        self._old_data = os.environ.get("DATA_DIR")
        os.environ["DATA_DIR"] = str(root / "data")
        get_settings.cache_clear()
        settings = get_settings()
        settings.content_dir = root / "content"
        settings.frontend_dist = root / "dist"
        settings.sso_proxy_secret = SECRET
        curriculum.reload_content()
        db.init()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        curriculum.reload_content()
        get_settings.cache_clear()
        if self._old_data is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = self._old_data
        self.tmp.cleanup()

    def get(self, path: str = URL, **headers: str):
        return self.client.get(path, headers=headers)

    def test_summary_counts_the_served_lessons(self) -> None:
        db.set_progress("ada", "001-alpha", "solved")
        db.set_progress("ada", "002-beta", "started")
        # Drafts and retired slugs are not in the list, so they do not count.
        db.set_progress("ada", "004-hidden", "solved")
        db.set_progress("ada", "999-retired", "solved")
        db.set_progress("bo", "003-gamma", "solved")
        time.sleep(0.01)
        db.set_progress("ada", "002-beta", "started")
        time.sleep(0.01)
        db.set_progress("ada", "999-retired", "started")
        response = self.get(**SIGNED_IN)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"done": 1, "total": 3, "last": {"slug": "002-beta", "title": "Beta"}},
        )

    def test_summary_is_empty_without_progress(self) -> None:
        response = self.get(**SIGNED_IN)
        self.assertEqual(response.json(), {"done": 0, "total": 3, "last": None})

    def test_signed_out_is_401_with_cors_for_the_hub(self) -> None:
        response = self.get(origin=HUB)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers["access-control-allow-origin"], HUB)
        self.assertEqual(response.headers["access-control-allow-credentials"], "true")
        self.assertIn("Origin", response.headers["vary"])

    def test_signed_in_response_carries_cors_for_the_hub(self) -> None:
        response = self.get(origin=HUB, **SIGNED_IN)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], HUB)
        self.assertEqual(response.headers["access-control-allow-credentials"], "true")

    def test_other_origins_get_no_allow_origin_header(self) -> None:
        for origin in ("https://evil.example", HUB + ".evil.example", "http://courses.ifkash.dev"):
            response = self.get(origin=origin, **SIGNED_IN)
            self.assertNotIn("access-control-allow-origin", response.headers)
            self.assertNotIn("access-control-allow-credentials", response.headers)
            self.assertIn("Origin", response.headers["vary"])
        response = self.get(**SIGNED_IN)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_other_paths_get_no_cors_headers(self) -> None:
        response = self.get("/api/health", origin=HUB)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("access-control-allow-origin", response.headers)
        self.assertNotIn("vary", response.headers)
        response = self.get("/api/lessons", origin=HUB)
        self.assertEqual(response.status_code, 401)
        self.assertNotIn("access-control-allow-origin", response.headers)


if __name__ == "__main__":
    unittest.main()
