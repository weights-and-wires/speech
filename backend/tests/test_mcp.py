"""The MCP endpoint, its bearer tokens, and the token management API.

Run from the repository root:

    .venv/bin/python -m unittest discover -s backend/tests -t .
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SECRET = "test-proxy-secret"
SIGNED_IN = {"Remote-User": "Portal", "X-SSO-Proxy": SECRET}
OTHER = {"Remote-User": "other", "X-SSO-Proxy": SECRET}
ACCEPT = "application/json, text/event-stream"

LESSON = """---
title: The FFT and windows
slug: 001-alpha
part: "Part 1 — Sound and signals"
summary: Split the DFT in half, then fade each frame.
minutes: 40
gpu: false
objectives:
  - State one fact.
  - State a second fact.
  - State a third fact.
exercise: 001-alpha
---

# The FFT and windows

A frame of speech needs a spectrum.

## Why the matrix is too slow

The matrix repeats work, so a radix-2 butterfly halves it.

```figure
<svg viewBox="0 0 10 10" role="img" aria-label="Two halves combine into one spectrum"><rect width="10" height="10"/></svg>
Figure 1. The butterfly. Notice that two bins share one product.
```

## Spectral leakage

A tone between two bins smears into every bin.

```demo
{"id": "window-leakage", "caption": "Switch the window and watch the far bins drop."}
```
"""

SECOND = """---
title: Mel scales
slug: 002-beta
part: "Part 2 — Features"
summary: Hearing is not linear.
minutes: 30
exercise: 002-beta
---

# Mel scales

Spectral leakage returns here, with a second look.

```demo
{"id": "window-leakage", "caption": "Again, in a new context.", "props": {"window": "hann"}}
```
"""

TESTS = """from speechcheck import check, equal


@check("add returns the sum")
def _(sub):
    equal(sub.add(2, 3), 5)
"""


class MCPTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        lessons = self.root / "content" / "lessons"
        lessons.mkdir(parents=True)
        (lessons / "001-alpha.md").write_text(LESSON, encoding="utf-8")
        (lessons / "002-beta.md").write_text(SECOND, encoding="utf-8")
        for slug in ("001-alpha", "002-beta"):
            exercise = self.root / "content" / "exercises" / slug
            exercise.mkdir(parents=True)
            (exercise / "meta.json").write_text(
                json.dumps({"id": slug, "title": slug, "brief": "Add.", "timeout": 30}),
                encoding="utf-8",
            )
            (exercise / "starter.py").write_text("def add(a, b):\n    raise NotImplementedError\n")
            (exercise / "solution.py").write_text("def add(a, b):\n    return a + b\n")
            (exercise / "tests.py").write_text(TESTS)
        self._env = {
            "DATA_DIR": str(self.root / "data"),
            "CONTENT_DIR": str(self.root / "content"),
            "FRONTEND_DIST": str(self.root / "missing-dist"),
            "BOOK_DIR": str(self.root / "book"),
            "SSO_PROXY_SECRET": SECRET,
            "SSO_USERS": "portal:reader",
        }
        os.environ.update(self._env)
        from fastapi.testclient import TestClient

        from backend.app import curriculum, db
        from backend.app.config import get_settings
        from backend.app.main import app

        self.get_settings = get_settings
        self.curriculum = curriculum
        self.db = db
        get_settings.cache_clear()
        curriculum.reload_content()
        # Entering the client runs the lifespan, which starts the MCP session manager.
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        for key in self._env:
            os.environ.pop(key, None)
        os.environ.pop("MCP_PUBLIC_URL", None)
        self.get_settings.cache_clear()
        self.curriculum.reload_content()
        self._dir.cleanup()

    # Helpers ------------------------------------------------------------

    def _new_token(self, name: str = "laptop", headers: dict | None = None) -> dict:
        response = self.client.post(
            "/api/mcp/tokens", json={"name": name}, headers=headers or SIGNED_IN
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _rpc(self, token: str | None, method: str, params: dict | None = None,
             path: str = "/mcp", extra: dict | None = None):
        headers = {"Accept": ACCEPT, "Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        headers.update(extra or {})
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        return self.client.post(path, headers=headers, json=body, follow_redirects=False)

    def _call(self, token: str, tool: str, **arguments):
        response = self._rpc(token, "tools/call", {"name": tool, "arguments": arguments})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()["result"]
        text = result["content"][0]["text"]
        return result.get("isError", False), (text if result.get("isError") else json.loads(text))

    def _call_with(self, token: str, headers: dict, tool: str, **arguments):
        response = self._rpc(
            token, "tools/call", {"name": tool, "arguments": arguments}, extra=headers
        )
        result = response.json()["result"]
        return result.get("isError", False), json.loads(result["content"][0]["text"])

    # Token management ---------------------------------------------------

    def test_token_api_needs_sso(self) -> None:
        self.assertEqual(self.client.get("/api/mcp/tokens").status_code, 401)
        self.assertEqual(self.client.post("/api/mcp/tokens", json={"name": "x"}).status_code, 401)
        self.assertEqual(self.client.delete("/api/mcp/tokens/1").status_code, 401)

    def test_create_list_and_revoke(self) -> None:
        listed = self.client.get("/api/mcp/tokens", headers=SIGNED_IN).json()
        self.assertEqual(listed, {"url": "http://testserver/mcp", "tokens": []})

        created = self._new_token()
        self.assertEqual(set(created), {"id", "name", "prefix", "created_at", "token"})
        self.assertTrue(created["token"].startswith("speechmcp_"))
        self.assertEqual(created["prefix"], created["token"][:12])
        self.assertEqual(created["name"], "laptop")

        tokens = self.client.get("/api/mcp/tokens", headers=SIGNED_IN).json()["tokens"]
        self.assertEqual(
            tokens,
            [
                {
                    "id": created["id"],
                    "name": "laptop",
                    "prefix": created["prefix"],
                    "created_at": created["created_at"],
                    "last_used_at": None,
                }
            ],
        )
        # The secret itself is never stored, only its hash.
        with self.db.session() as connection:
            row = dict(connection.execute("SELECT * FROM mcp_tokens").fetchone())
        self.assertNotIn(created["token"], json.dumps(row))
        self.assertEqual(row["username"], "reader")  # after SSO_USERS mapping
        self.assertEqual(len(row["token_hash"]), 64)

        # Another reader neither sees nor revokes it.
        self.assertEqual(self.client.get("/api/mcp/tokens", headers=OTHER).json()["tokens"], [])
        response = self.client.delete(f"/api/mcp/tokens/{created['id']}", headers=OTHER)
        self.assertEqual(response.status_code, 404)

        response = self.client.delete(f"/api/mcp/tokens/{created['id']}", headers=SIGNED_IN)
        self.assertEqual(response.json(), {"revoked": True})
        self.assertEqual(self.client.get("/api/mcp/tokens", headers=SIGNED_IN).json()["tokens"], [])
        response = self.client.delete(f"/api/mcp/tokens/{created['id']}", headers=SIGNED_IN)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self._rpc(created["token"], "ping").status_code, 401)

    def test_public_url_setting(self) -> None:
        os.environ["MCP_PUBLIC_URL"] = "https://speech.ifkash.dev/mcp"
        self.get_settings.cache_clear()
        listed = self.client.get("/api/mcp/tokens", headers=SIGNED_IN).json()
        self.assertEqual(listed["url"], "https://speech.ifkash.dev/mcp")

    def test_name_is_validated(self) -> None:
        for name in ("", "   ", "x" * 65):
            response = self.client.post(
                "/api/mcp/tokens", json={"name": name}, headers=SIGNED_IN
            )
            self.assertEqual(response.status_code, 422, name)

    def test_at_most_ten_active_tokens(self) -> None:
        for i in range(10):
            self._new_token(f"t{i}")
        response = self.client.post("/api/mcp/tokens", json={"name": "t10"}, headers=SIGNED_IN)
        self.assertEqual(response.status_code, 409)

    def test_cli_creates_lists_and_revokes(self) -> None:
        from backend.app import mcp_tokens

        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(mcp_tokens.main(["create", "reader", "--name", "vm"]), 0)
        token = out.getvalue().strip().splitlines()[-1]
        self.assertTrue(token.startswith("speechmcp_"))
        self.assertEqual(self._rpc(token, "ping").status_code, 200)

        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(mcp_tokens.main(["list", "reader"]), 0)
        self.assertIn(token[:12], out.getvalue())
        self.assertNotIn(token, out.getvalue())

        token_id = self.db.list_tokens("reader")[0]["id"]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(mcp_tokens.main(["revoke", str(token_id)]), 0)
        self.assertEqual(self._rpc(token, "ping").status_code, 401)

    # Authentication on /mcp ---------------------------------------------

    def test_mcp_rejects_missing_and_bad_tokens(self) -> None:
        response = self._rpc(None, "ping")
        self.assertEqual(response.status_code, 401)
        self.assertTrue(response.headers["www-authenticate"].startswith("Bearer"))

        response = self._rpc("speechmcp_not-a-real-token", "ping")
        self.assertEqual(response.status_code, 401)
        self.assertIn('error="invalid_token"', response.headers["www-authenticate"])

        # A token of another course's format never reaches the database.
        self.assertEqual(self._rpc("rlmcp_not-a-real-token", "ping").status_code, 401)

        response = self._rpc(None, "ping", extra={"Authorization": "Basic cmVhZGVyOng="})
        self.assertEqual(response.status_code, 401)

    def test_mcp_ignores_sso_headers(self) -> None:
        response = self._rpc(None, "ping", extra=SIGNED_IN)
        self.assertEqual(response.status_code, 401)

        # With a valid token, SSO headers can't change who the caller is.
        token = self._new_token(headers=OTHER)["token"]
        _, progress = self._call_with(token, SIGNED_IN, "get_progress")
        self.assertEqual(progress["reader"], "other")

    def test_hub_secret_acts_as_the_named_reader(self) -> None:
        hub = {"X-MCP-Hub-Secret": "hub-secret", "X-MCP-Hub-User": "Portal"}

        # Off by default: without MCP_HUB_SECRET the headers grant nothing.
        self.assertEqual(self._rpc(None, "ping", extra=hub).status_code, 401)

        os.environ["MCP_HUB_SECRET"] = "hub-secret"
        self.get_settings.cache_clear()
        try:
            wrong = {**hub, "X-MCP-Hub-Secret": "guess"}
            self.assertEqual(self._rpc(None, "ping", extra=wrong).status_code, 401)
            no_user = {"X-MCP-Hub-Secret": "hub-secret"}
            self.assertEqual(self._rpc(None, "ping", extra=no_user).status_code, 401)

            response = self._rpc(
                None, "tools/call", {"name": "get_progress", "arguments": {}}, extra=hub
            )
            self.assertEqual(response.status_code, 200, response.text)
            progress = json.loads(response.json()["result"]["content"][0]["text"])
            # The Authelia name maps through SSO_USERS, as a browser session does.
            self.assertEqual(progress["reader"], "reader")
        finally:
            os.environ.pop("MCP_HUB_SECRET", None)
            self.get_settings.cache_clear()

    def test_well_known_and_mcp_subpaths_are_not_the_spa(self) -> None:
        # Without a built frontend there is no catch-all, so a 404 is the floor;
        # `test_spa_catch_all_returns_json_404` covers the catch-all itself.
        for path in ("/.well-known/oauth-authorization-server", "/mcp/other"):
            self.assertEqual(self.client.get(path).status_code, 404, path)

    # The protocol and the tools -----------------------------------------

    def test_initialize_and_list_tools(self) -> None:
        token = self._new_token()["token"]
        for path in ("/mcp", "/mcp/"):
            response = self._rpc(
                token,
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
                path=path,
            )
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()["result"]
            self.assertEqual(result["protocolVersion"], "2025-06-18")
            self.assertIn("tools", result["capabilities"])
            self.assertEqual(result["serverInfo"]["name"], "speech-course")

        response = self.client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {token}", "Accept": ACCEPT},
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        self.assertEqual(response.status_code, 202)

        tools = {tool["name"] for tool in self._rpc(token, "tools/list").json()["result"]["tools"]}
        self.assertEqual(
            tools,
            {
                "course_overview", "list_lessons", "get_lesson", "search", "list_exercises",
                "get_exercise", "get_solution", "list_demos", "get_demo", "get_document",
                "get_book_info", "get_progress", "list_drafts", "get_draft", "save_draft",
                "delete_draft", "set_progress", "run_exercise",
            },
        )
        templates = self._rpc(token, "resources/templates/list").json()["result"]
        self.assertEqual(
            [t["uriTemplate"] for t in templates["resourceTemplates"]],
            ["speech://lessons/{slug}"],
        )

    def test_last_used_is_recorded(self) -> None:
        token = self._new_token()["token"]
        self.assertEqual(self._rpc(token, "ping").status_code, 200)
        tokens = self.client.get("/api/mcp/tokens", headers=SIGNED_IN).json()["tokens"]
        self.assertIsNotNone(tokens[0]["last_used_at"])

    def test_get_lesson_does_not_mark_progress(self) -> None:
        token = self._new_token()["token"]
        error, lesson = self._call(token, "get_lesson", slug="001-alpha")
        self.assertFalse(error)
        self.assertIn("radix-2 butterfly", lesson["body"])
        self.assertEqual(lesson["next"], "002-beta")
        self.assertEqual(lesson["exercise"], "001-alpha")
        self.assertIsNone(lesson["status"])
        self.assertEqual(self.db.get_progress("reader"), {})

        error, message = self._call(token, "get_lesson", slug="nope")
        self.assertTrue(error)
        self.assertIn("No lesson", message)

        response = self._rpc(token, "resources/read", {"uri": "speech://lessons/002-beta"})
        self.assertIn("Mel scales", response.json()["result"]["contents"][0]["text"])

    def test_lesson_text_replaces_figures_and_demos(self) -> None:
        token = self._new_token()["token"]
        _, lesson = self._call(token, "get_lesson", slug="001-alpha")
        body = lesson["body"]
        self.assertNotIn("<svg", body)
        self.assertNotIn("```figure", body)
        self.assertNotIn("```demo", body)
        self.assertIn("[Diagram: Two halves combine into one spectrum]", body)
        self.assertIn("Figure 1. The butterfly.", body)
        self.assertIn("[Interactive demo `window-leakage`. Switch the window", body)
        self.assertEqual(
            lesson["demos"],
            [{"id": "window-leakage", "caption": "Switch the window and watch the far bins drop."}],
        )

        _, raw = self._call(token, "get_lesson", slug="001-alpha", format="raw")
        self.assertIn("<svg", raw["body"])
        self.assertIn("```demo", raw["body"])

    def test_search(self) -> None:
        token = self._new_token()["token"]
        error, found = self._call(token, "search", query="SPECTRAL leakage", limit=50)
        self.assertFalse(error)
        # The research notes in docs/ also match, so look at the lessons alone.
        lessons = [r for r in found["results"] if r["kind"] == "lesson"]
        self.assertEqual([r["slug"] for r in lessons], ["001-alpha", "002-beta"])
        hit = lessons[0]
        self.assertEqual(hit["section"], "Spectral leakage")
        self.assertIn("spectral leakage", hit["snippet"].lower())

        # SVG markup is not searchable text; a figure's caption is.
        _, found = self._call(token, "search", query="<rect")
        self.assertEqual(found["total"], 0)
        _, found = self._call(token, "search", query="two bins share one product")
        hits = [r for r in found["results"] if r["kind"] == "lesson"]
        self.assertEqual([r["slug"] for r in hits], ["001-alpha"])
        self.assertIn("Figure 1. The butterfly", hits[0]["snippet"])

        _, found = self._call(token, "search", query="spectral", limit=1)
        self.assertGreaterEqual(found["total"], 2)
        self.assertEqual(len(found["results"]), 1)

        # Documents are searchable too, and name their own slug.
        _, found = self._call(token, "search", query="content contract")
        self.assertIn(("document", "CONTENT_CONTRACT.md"), {(r["kind"], r["slug"]) for r in found["results"]})

        error, message = self._call(token, "search", query="   ")
        self.assertTrue(error)

    def test_overview_exercises_and_solutions(self) -> None:
        token = self._new_token()["token"]
        _, overview = self._call(token, "course_overview")
        self.assertEqual(overview["reader"], "reader")
        self.assertEqual(overview["counts"]["lessons"], 2)
        self.assertEqual(overview["counts"]["exercises"], 2)
        self.assertEqual(overview["counts"]["parts"], 2)
        self.assertEqual(overview["parts"][0]["lessons"][0]["slug"], "001-alpha")
        self.assertIsNone(overview["book"])
        self.assertIn("CURRICULUM.md", overview["documents"])

        _, listed = self._call(token, "list_exercises")
        self.assertEqual([e["id"] for e in listed["exercises"]], ["001-alpha", "002-beta"])
        self.assertEqual(listed["exercises"][0]["lessons"], ["001-alpha"])
        self.assertFalse(listed["exercises"][0]["has_draft"])
        self.assertNotIn("starter", listed["exercises"][0])

        _, exercise = self._call(token, "get_exercise", exercise_id="001-alpha")
        self.assertIn("NotImplementedError", exercise["starter"])
        self.assertIn("@check", exercise["tests"])
        self.assertNotIn("solution", exercise)
        self.assertEqual(exercise["lessons"], ["001-alpha"])
        _, solution = self._call(token, "get_solution", exercise_id="001-alpha")
        self.assertIn("return a + b", solution["solution"])

        error, _ = self._call(token, "get_exercise", exercise_id="nope")
        self.assertTrue(error)

    def test_demos(self) -> None:
        token = self._new_token()["token"]
        _, listed = self._call(token, "list_demos")
        by_id = {d["id"]: d for d in listed["demos"]}
        leakage = by_id["window-leakage"]
        self.assertEqual([p["slug"] for p in leakage["lessons"]], ["001-alpha", "002-beta"])
        # Titles come from the registry in the checkout, not from the lessons.
        self.assertTrue(leakage["title"])
        # A registered demo no lesson here places is still listed.
        self.assertEqual(by_id["stft-explorer"]["lessons"], [])

        _, demo = self._call(token, "get_demo", demo_id="window-leakage")
        self.assertEqual(demo["path"], "/demos/window-leakage")
        self.assertEqual(demo["lessons"][1]["props"], {"window": "hann"})
        self.assertNotIn("props", demo["lessons"][0])
        error, message = self._call(token, "get_demo", demo_id="nope")
        self.assertTrue(error)
        self.assertIn("No demo", message)

    def test_documents(self) -> None:
        token = self._new_token()["token"]
        _, document = self._call(token, "get_document", name="curriculum")
        self.assertEqual(document["name"], "CURRICULUM.md")
        self.assertIn("# Curriculum", document["markdown"])
        _, document = self._call(token, "get_document", name="docs/DEMOS")
        self.assertEqual(document["name"], "DEMOS.md")
        _, document = self._call(token, "get_document", name="research/02-asr-and-ssl.md")
        self.assertEqual(document["name"], "research/02-asr-and-ssl.md")

        for name in ("../README", "../README.md", "research/../../README.md", "/etc/passwd", "research"):
            error, _ = self._call(token, "get_document", name=name)
            self.assertTrue(error, name)

    def test_book_info(self) -> None:
        token = self._new_token()["token"]
        _, info = self._call(token, "get_book_info")
        self.assertEqual(info, {"book": None})

        book = self.root / "book"
        book.mkdir()
        (book / "speech-and-audio.pdf").write_bytes(b"%PDF-1.7\n")
        (book / "manifest.json").write_text(
            json.dumps({"file": "speech-and-audio.pdf", "title": "Speech", "pages": 3, "bytes": 9}),
            encoding="utf-8",
        )
        _, info = self._call(token, "get_book_info")
        self.assertEqual(info["book"]["pages"], 3)
        self.assertEqual(info["book"]["url"], "/book/speech-and-audio.pdf")

    def test_drafts_progress_and_runs_act_as_the_token_owner(self) -> None:
        token = self._new_token()["token"]
        _, saved = self._call(token, "save_draft", exercise_id="001-alpha", code="x = 1\n")
        self.assertTrue(saved["saved"])
        self.assertEqual(self.db.get_draft("reader", "001-alpha"), "x = 1\n")
        _, draft = self._call(token, "get_draft", exercise_id="001-alpha")
        self.assertEqual(draft["code"], "x = 1\n")
        _, drafts = self._call(token, "list_drafts")
        self.assertEqual(
            [(d["exercise_id"], d["characters"]) for d in drafts["drafts"]], [("001-alpha", 6)]
        )
        _, listed = self._call(token, "list_exercises")
        self.assertTrue(listed["exercises"][0]["has_draft"])

        _, marked = self._call(token, "set_progress", slug="002-beta", status="started")
        self.assertEqual(marked["status"], "started")

        code = "def add(a, b):\n    print('adding')\n    return a + b\n"
        error, result = self._call(token, "run_exercise", exercise_id="001-alpha", code=code)
        self.assertFalse(error, result)
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["passed"], result["total"]), (1, 1))
        self.assertEqual(result["checks"][0]["status"], "passed")
        self.assertEqual(result["lessons_solved"], ["001-alpha"])
        self.assertEqual(self.db.get_progress("reader")["001-alpha"]["status"], "solved")
        self.assertEqual(self.db.get_draft("reader", "001-alpha"), code)

        _, progress = self._call(token, "get_progress")
        self.assertEqual(progress["summary"]["done"], 1)
        self.assertEqual(progress["runs"][0]["exercise_id"], "001-alpha")
        self.assertEqual(
            [(row["slug"], row["status"]) for row in progress["lessons"]],
            [("001-alpha", "solved"), ("002-beta", "started")],
        )
        _, progress = self._call(token, "get_progress", runs=0)
        self.assertEqual(progress["runs"], [])

        error, result = self._call(
            token, "run_exercise", exercise_id="002-beta", code="print('hi')\n", mode="run"
        )
        self.assertFalse(error, result)
        self.assertIn("hi", result["stdout"])
        self.assertEqual(self.db.get_progress("reader")["002-beta"]["status"], "started")

        _, deleted = self._call(token, "delete_draft", exercise_id="001-alpha")
        self.assertFalse(deleted["saved"])
        self.assertIsNone(self.db.get_draft("reader", "001-alpha"))

    def test_run_describes_clips_and_figures_without_their_data(self) -> None:
        token = self._new_token()["token"]
        code = (
            "import numpy as np\n"
            "from speechlab import play, show_plot, show_spectrogram\n"
            "t = np.arange(8000) / 8000\n"
            "play(np.sin(2 * np.pi * 440 * t), 8000, label='A440')\n"
            "show_plot((t, np.sin(2 * np.pi * 3 * t)), label='slow sine', xlabel='s')\n"
            "show_spectrogram(np.abs(np.random.default_rng(0).normal(size=(33, 20))),"
            " sample_rate=8000, hop_length=80, label='noise')\n"
        )
        error, result = self._call(
            token, "run_exercise", exercise_id="001-alpha", code=code, mode="run"
        )
        self.assertFalse(error, result)
        self.assertTrue(result["ok"], result)
        kinds = [item["kind"] for item in result["media"]]
        self.assertEqual(kinds, ["audio", "line", "heatmap"])
        audio, line, heatmap = result["media"]
        self.assertEqual((audio["label"], audio["sample_rate"], audio["seconds"]), ("A440", 8000, 1.0))
        self.assertEqual(line["series"][0]["points"], 8000)
        self.assertEqual(heatmap["shape"][1], 20)
        self.assertNotIn("base64", json.dumps(result["media"]))

    def test_rest_run_stream_still_records_and_solves(self) -> None:
        # `/api/run` shares `runs.start_run` with `run_exercise`.
        code = "def add(a, b):\n    return a + b\n"
        response = self.client.post(
            "/api/run",
            headers=SIGNED_IN,
            json={"exercise_id": "001-alpha", "mode": "check", "code": code},
        )
        self.assertEqual(response.status_code, 200)
        events = [
            json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
        ]
        self.assertEqual(events[0]["type"], "begin")
        self.assertEqual(events[-1]["type"], "done")
        self.assertTrue(events[-1]["ok"], events)
        self.assertEqual(self.db.get_progress("reader")["001-alpha"]["status"], "solved")
        self.assertEqual(self.db.recent_runs("reader")[0]["passed"], 1)
        missing = self.client.post(
            "/api/run",
            headers=SIGNED_IN,
            json={"exercise_id": "nope", "mode": "check", "code": ""},
        )
        self.assertEqual(missing.status_code, 404)

    def test_spa_catch_all_returns_json_404(self) -> None:
        # A built frontend mounts the catch-all; `.well-known/` and `mcp/...`
        # must not fall through to index.html there.
        from fastapi.testclient import TestClient

        dist = self.root / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html>app</html>", encoding="utf-8")
        os.environ["FRONTEND_DIST"] = str(dist)
        self.get_settings.cache_clear()
        import importlib

        from backend.app import main

        main = importlib.reload(main)
        with TestClient(main.app) as client:
            self.assertEqual(client.get("/lessons/x").text, "<html>app</html>")
            for path in ("/.well-known/oauth-authorization-server", "/mcp/other", "/api/nope"):
                response = client.get(path)
                self.assertEqual(response.status_code, 404, path)
                self.assertEqual(response.json(), {"detail": "Not found"}, path)
        os.environ["FRONTEND_DIST"] = str(self.root / "missing-dist")
        self.get_settings.cache_clear()
        importlib.reload(main)


if __name__ == "__main__":
    unittest.main()
