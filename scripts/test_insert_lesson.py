"""Unit tests for scripts/insert_lesson.py.

Run from the repository root:

    python3 scripts/test_insert_lesson.py
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import insert_lesson  # noqa: E402


def load_check_content():
    spec = importlib.util.spec_from_file_location(
        "check_content", ROOT / "scripts" / "check_content.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


DEMO = (
    '{"id": "stft-explorer", "caption": "Lengthen the window until the two tones separate."}'
)
MARKER = 'id="c002-arrow"'


def snapshot(root: Path) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            found[str(path.relative_to(root))] = path.read_bytes()
    return found


def write_lesson(path: Path, slug: str, part: str, body: str, prereqs: list[str]) -> None:
    prereq_block = "\n".join(f"  - {item}" for item in prereqs)
    prereq_yaml = f"prereqs:\n{prereq_block}\n" if prereqs else ""
    path.write_text(
        f"""---
title: {slug}
slug: {slug}
part: "{part}"
summary: A fixture lesson.
minutes: 30
gpu: false
objectives:
  - First objective for the fixture.
  - Second objective for the fixture.
  - Third objective for the fixture.
{prereq_yaml}exercise: {slug}
---
{body}
""",
        encoding="utf-8",
    )


def write_exercise(directory: Path, slug: str, brief: str) -> None:
    directory.mkdir(parents=True)
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "id": slug,
                "title": slug,
                "brief": brief,
                "needs_gpu": False,
                "timeout": 60,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "starter.py").write_text("# fixture\n", encoding="utf-8")
    (directory / "solution.py").write_text("# fixture\n", encoding="utf-8")
    (directory / "tests.py").write_text("# fixture\n", encoding="utf-8")


def make_course(root: Path) -> None:
    lessons = root / "content" / "lessons"
    exercises = root / "content" / "exercises"
    docs = root / "docs"
    lessons.mkdir(parents=True)
    exercises.mkdir(parents=True)
    docs.mkdir(parents=True)
    part = "Part 1 — Sound and signals"
    write_lesson(
        lessons / "001-alpha.md",
        "001-alpha",
        part,
        "\n# Alpha\n\nSee [Beta](/lessons/002-beta) for the next step.\n",
        [],
    )
    write_lesson(
        lessons / "002-beta.md",
        "002-beta",
        part,
        "\n# Beta\n\n"
        "See [the project](/lessons/003p-project-name) and "
        "[Alpha](/lessons/001-alpha).\n\n"
        "The id 002-beta stays in this sentence.\n\n"
        "```demo\n"
        f"{DEMO}\n"
        "```\n\n"
        "```figure\n"
        '<svg viewBox="0 0 560 80" role="img" aria-label="A bar">\n'
        f"  <defs><marker {MARKER}></marker></defs>\n"
        '  <rect x="10" y="10" width="40" height="20" class="fill-a"/>\n'
        "</svg>\n"
        "Figure 1. A bar.\n"
        "```\n",
        ["001-alpha", "003p-project-name"],
    )
    write_lesson(
        lessons / "003p-project-name.md",
        "003p-project-name",
        part,
        "\n# Project\n\nSee [Beta](/lessons/002-beta).\n",
        ["002-beta"],
    )
    write_exercise(exercises / "001-alpha", "001-alpha", "Alpha stays.")
    write_exercise(exercises / "002-beta", "002-beta", "Keep this brief.")
    write_exercise(
        exercises / "003p-project-name",
        "003p-project-name",
        "Project brief.",
    )
    (root / "content" / "renames.json").write_text(
        json.dumps(
            {"old-course-lesson": ["002-beta", "003p-project-name"]},
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    (docs / "CURRICULUM.md").write_text(
        "\n".join(
            [
                "# Curriculum",
                "",
                "| Slug | Title | The exercise | Demos |",
                "|---|---|---|---|",
                "| `001-alpha` | Alpha | Does alpha. | none |",
                "| `002-beta` | Beta | Does beta. | `stft-explorer` |",
                "| `003p-project-name` | Project | Does the project. | none |",
                "",
            ]
        ),
        encoding="utf-8",
    )


class InsertLessonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_course(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_dry_run_changes_nothing(self) -> None:
        before = snapshot(self.root)
        report = insert_lesson.insert_lessons(
            self.root, "001-alpha", "gamma", dry_run=True
        )
        self.assertEqual(before, snapshot(self.root))
        self.assertIn("002-beta -> 003-beta", report.lines)
        self.assertIn("003p-project-name -> 004p-project-name", report.lines)
        self.assertIn("Add draft 002-gamma.", report.lines)
        self.assertIn("Each draft copies Part 1 — Sound and signals.", report.lines)
        self.assertTrue(report.lines[-1].startswith("Dry run:"))

    def test_shift_rewrites_links_ids_and_keeps_demo_markup(self) -> None:
        insert_lesson.insert_lessons(self.root, "001-alpha", "gamma")

        self.assertFalse((self.root / "content" / "lessons" / "002-beta.md").exists())
        self.assertTrue((self.root / "content" / "lessons" / "003-beta.md").exists())
        self.assertTrue(
            (self.root / "content" / "lessons" / "004p-project-name.md").exists()
        )
        self.assertFalse(
            (self.root / "content" / "exercises" / "003p-project-name").exists()
        )

        beta = (self.root / "content" / "lessons" / "003-beta.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("slug: 003-beta", beta)
        self.assertIn("exercise: 003-beta", beta)
        self.assertIn("  - 004p-project-name", beta)
        self.assertIn("  - 001-alpha", beta)
        self.assertIn("](/lessons/004p-project-name)", beta)
        self.assertIn("](/lessons/001-alpha)", beta)
        self.assertNotIn("/lessons/003p-project-name", beta)
        self.assertNotIn("/lessons/002-beta", beta)
        self.assertIn("The id 002-beta stays in this sentence.", beta)
        self.assertIn(DEMO, beta)
        self.assertIn(MARKER, beta)

        alpha = (self.root / "content" / "lessons" / "001-alpha.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("slug: 001-alpha", alpha)
        self.assertIn("](/lessons/003-beta)", alpha)
        self.assertNotIn("/lessons/002-beta", alpha)

        project = (
            self.root / "content" / "lessons" / "004p-project-name.md"
        ).read_text(encoding="utf-8")
        self.assertIn("slug: 004p-project-name", project)
        self.assertIn("](/lessons/003-beta)", project)
        self.assertIn("  - 003-beta", project)

        meta = json.loads(
            (self.root / "content" / "exercises" / "003-beta" / "meta.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(meta["id"], "003-beta")
        self.assertEqual(meta["brief"], "Keep this brief.")
        project_meta = json.loads(
            (
                self.root / "content" / "exercises" / "004p-project-name" / "meta.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(project_meta["id"], "004p-project-name")

        renames = json.loads(
            (self.root / "content" / "renames.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            renames["old-course-lesson"],
            ["003-beta", "004p-project-name"],
        )
        self.assertEqual(renames["002-beta"], ["003-beta"])
        self.assertEqual(renames["003p-project-name"], ["004p-project-name"])

        curriculum = (self.root / "docs" / "CURRICULUM.md").read_text(encoding="utf-8")
        self.assertIn("| `003-beta` | Beta | Does beta. | `stft-explorer` |", curriculum)
        self.assertIn("| `004p-project-name` | Project | Does the project. | none |", curriculum)
        self.assertNotIn("`002-beta`", curriculum)
        self.assertNotIn("`003p-project-name`", curriculum)
        self.assertIn("`stft-explorer`", curriculum)

        stub = (self.root / "content" / "lessons" / "002-gamma.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("draft: true", stub)
        self.assertIn('part: "Part 1 — Sound and signals"', stub)
        self.assertIn("slug: 002-gamma", stub)
        self.assertIn("  - 001-alpha", stub)
        self.assertIn("# Gamma", stub)
        stub_meta = json.loads(
            (self.root / "content" / "exercises" / "002-gamma" / "meta.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(stub_meta["id"], "002-gamma")
        self.assertIn(
            "| `002-gamma` | Gamma | A draft exercise that returns its input unchanged. | none |",
            curriculum,
        )
        # The draft row sits after the lesson that stayed put.
        self.assertLess(curriculum.index("`001-alpha`"), curriculum.index("`002-gamma`"))
        self.assertLess(curriculum.index("`002-gamma`"), curriculum.index("`003-beta`"))

    def test_count_inserts_distinct_stubs_and_keeps_project_marker(self) -> None:
        report = insert_lesson.insert_lessons(
            self.root, "001-alpha", "gamma", count=2
        )
        self.assertEqual(report.stubs, ["002-gamma", "003-gamma-2"])
        self.assertIn("002-beta -> 004-beta", report.lines)
        self.assertIn("003p-project-name -> 005p-project-name", report.lines)
        first = (self.root / "content" / "lessons" / "002-gamma.md").read_text(
            encoding="utf-8"
        )
        second = (self.root / "content" / "lessons" / "003-gamma-2.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("draft: true", first)
        self.assertIn("draft: true", second)
        self.assertIn('part: "Part 1 — Sound and signals"', first)
        self.assertIn('part: "Part 1 — Sound and signals"', second)
        self.assertIn("  - 001-alpha", first)
        self.assertIn("  - 002-gamma", second)
        self.assertTrue((self.root / "content" / "exercises" / "003-gamma-2").is_dir())
        self.assertTrue(
            (self.root / "content" / "lessons" / "005p-project-name.md").is_file()
        )

    def test_second_shift_keeps_the_older_slug_pointing_at_the_lesson(self) -> None:
        insert_lesson.insert_lessons(self.root, "001-alpha", "extra")
        insert_lesson.insert_lessons(self.root, "001-alpha", "extra-two")
        renames = json.loads(
            (self.root / "content" / "renames.json").read_text(encoding="utf-8")
        )
        self.assertEqual(renames["old-course-lesson"], ["004-beta", "005p-project-name"])
        self.assertEqual(renames["002-beta"], ["004-beta"])
        self.assertEqual(renames["003-beta"], ["004-beta"])
        self.assertTrue(
            (self.root / "content" / "lessons" / "004-beta.md").is_file()
        )
        self.assertTrue(
            (self.root / "content" / "lessons" / "002-extra-two.md").is_file()
        )

    def test_unknown_anchor_leaves_the_tree_unchanged(self) -> None:
        before = snapshot(self.root)
        with self.assertRaises(insert_lesson.InsertError):
            insert_lesson.insert_lessons(self.root, "009-missing", "gamma")
        self.assertEqual(before, snapshot(self.root))

    def test_cli_dry_run_exits_zero(self) -> None:
        with redirect_stdout(io.StringIO()) as captured:
            code = insert_lesson.main(
                ["--after", "001-alpha", "--slug", "gamma", "--dry-run"],
                root=self.root,
            )
        self.assertEqual(code, 0)
        self.assertIn("Dry run: no files changed.", captured.getvalue())
        self.assertTrue((self.root / "content" / "lessons" / "002-beta.md").is_file())

    def test_stub_passes_the_content_checker(self) -> None:
        insert_lesson.insert_lessons(self.root, "003p-project-name", "probe")
        check_content = load_check_content()
        stub = self.root / "content" / "lessons" / "004-probe.md"
        self.assertTrue(stub.is_file())
        problems: list[str] = []
        original = check_content.EXERCISES
        check_content.EXERCISES = self.root / "content" / "exercises"
        try:
            check_content.check_lesson(stub, problems, set())
        finally:
            check_content.EXERCISES = original
        self.assertEqual(problems, [])


class DraftWordCountTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.check_content = load_check_content()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_lesson(self, draft: bool, extra: str) -> Path:
        flag = "draft: true\n" if draft else ""
        path = Path(self.tmp.name) / "001-short.md"
        path.write_text(
            f"""---
title: Short
slug: 001-short
part: "Part 1 — Sound and signals"
summary: A short fixture.
minutes: 15
gpu: false
{flag}objectives:
  - State the idea this chapter teaches.
  - Name the function the exercise asks you to write.
  - Say what the checks verify.
exercise: 001-short
---

# Short

A short opening.

## What you already know

- One fact.

## Build it step by step

1. Write the function.

## The exercise

You write the function.

```figure
<svg viewBox="0 0 560 80" role="img" aria-label="A bar">
  <rect x="10" y="10" width="40" height="20" class="fill-a"/>
</svg>
Figure 1. A bar.
```

{extra}
""",
            encoding="utf-8",
        )
        return path

    def problems_for(self, path: Path) -> list[str]:
        problems: list[str] = []
        with mock.patch.object(self.check_content, "check_exercise"):
            self.check_content.check_lesson(path, problems, set())
        return problems

    def test_draft_skips_the_word_minimum(self) -> None:
        problems = self.problems_for(self.write_lesson(draft=True, extra=""))
        self.assertEqual(problems, [])

    def test_published_lesson_keeps_the_word_minimum(self) -> None:
        problems = self.problems_for(self.write_lesson(draft=False, extra=""))
        self.assertTrue(any("words" in problem for problem in problems))

    def test_draft_keeps_the_upper_bound(self) -> None:
        extra = " ".join(["syllable"] * 2700)
        problems = self.problems_for(self.write_lesson(draft=True, extra=extra))
        self.assertTrue(any("split the chapter" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
