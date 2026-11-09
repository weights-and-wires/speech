#!/usr/bin/env python3
"""Insert one or more draft lessons and shift every later lesson.

    python3 scripts/insert_lesson.py --after 030-attention-encoder-decoders \\
        --slug new-slug-name
    python3 scripts/insert_lesson.py --after 030-attention-encoder-decoders \\
        --slug new-slug-name --count 2
    python3 scripts/insert_lesson.py --after 030-attention-encoder-decoders \\
        --slug new-slug-name --dry-run

The lesson named by `--after` stays where it is. Every later lesson file and
exercise directory moves forward by `--count` (the default is 1). The new
draft lessons occupy the numbers that open up. Each draft is marked
`draft: true` and copies `part` from the lesson that stayed put.

A lesson id is three digits, an optional `p`, and a slug, for example
`030-attention-encoder-decoders` or `030p-project-name`. This course numbers
every lesson as `NNN-slug`. Shifting increments the digits and keeps the `p`
when a filename has one.

Reader progress stays attached. Each shifted slug is recorded in
`content/renames.json` as the old slug mapped to a list of the new slug.
An older entry whose value still names that slug is updated to the slug it
moved to, which is the mapping `backend/app/curriculum.py` already loads.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Three digits, an optional project marker, and a hyphenated name.
SLUG_TOKEN = r"\d{3}p?-[a-z0-9]+(?:-[a-z0-9]+)*"
LESSON_LINK = re.compile(rf"(/lessons/)({SLUG_TOKEN})")
BACKTICK_SLUG = re.compile(rf"`({SLUG_TOKEN})`")
NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class InsertError(Exception):
    """The plan is rejected, and the tree is left unchanged."""


@dataclass(frozen=True, order=True)
class LessonId:
    """One lesson id, split so a shift can increment the number and keep `p`."""

    number: int
    project: str
    name: str

    @classmethod
    def parse(cls, slug: str) -> LessonId | None:
        match = re.fullmatch(r"(\d{3})(p?)-([a-z0-9]+(?:-[a-z0-9]+)*)", slug)
        if match is None:
            return None
        return cls(int(match.group(1)), match.group(2), match.group(3))

    @property
    def slug(self) -> str:
        return f"{self.number:03d}{self.project}-{self.name}"

    def shifted(self, count: int) -> LessonId:
        return LessonId(self.number + count, self.project, self.name)


@dataclass
class Report:
    """What the tool did, or what a dry run would do."""

    lines: list[str]
    mapping: dict[str, str]
    stubs: list[str]


def split_front_matter(text: str) -> tuple[str, str]:
    """Return the YAML block and the body, without the `---` markers."""
    if not text.startswith("---\n"):
        raise InsertError("a lesson does not start with YAML front matter")
    end = text.find("\n---\n", 3)
    if end < 0:
        raise InsertError("a lesson front matter block does not close")
    return text[4:end], text[end + 5 :]


def join_front_matter(front: str, body: str) -> str:
    return f"---\n{front}\n---\n{body}"


def field_value(front: str, name: str) -> str:
    """The raw YAML value of one top-level field, quotes included."""
    for line in front.split("\n"):
        match = re.fullmatch(rf"{name}:\s*(.*)", line)
        if match:
            return match.group(1).strip()
    raise InsertError(f"the lesson has no {name} field")


def unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def title_from_name(name: str) -> str:
    """Sentence case for a hyphenated slug name."""
    words = name.replace("-", " ")
    return words[:1].upper() + words[1:]


def lesson_paths(root: Path) -> dict[str, Path]:
    directory = root / "content" / "lessons"
    if not directory.is_dir():
        raise InsertError("content/lessons is missing")
    found: dict[str, Path] = {}
    for path in sorted(directory.glob("*.md")):
        ident = LessonId.parse(path.stem)
        if ident is None:
            raise InsertError(
                f"{path.name} does not match NNN-slug or NNNp-slug"
            )
        found[ident.slug] = path
    return found


def exercise_paths(root: Path) -> dict[str, Path]:
    directory = root / "content" / "exercises"
    if not directory.is_dir():
        raise InsertError("content/exercises is missing")
    found: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        ident = LessonId.parse(path.name)
        if ident is None:
            raise InsertError(
                f"content/exercises/{path.name} does not match NNN-slug or NNNp-slug"
            )
        found[ident.slug] = path
    return found


def stub_names(slug: str, count: int) -> list[str]:
    """Distinct names derived from `slug`. The first keeps the given name."""
    if count == 1:
        return [slug]
    return [slug if index == 1 else f"{slug}-{index}" for index in range(1, count + 1)]


def rewrite_front_matter(front: str, mapping: dict[str, str]) -> str:
    """Rewrite `slug`, `exercise`, and `prereqs` entries that moved."""
    lines = front.split("\n")
    rewritten: list[str] = []
    in_prereqs = False
    for line in lines:
        if re.fullmatch(r"prereqs:\s*", line):
            in_prereqs = True
            rewritten.append(line)
            continue
        if in_prereqs:
            item = re.fullmatch(r"(\s*-\s+)\"?([A-Za-z0-9-]+)\"?\s*", line)
            if item:
                old = item.group(2)
                if old in mapping:
                    rewritten.append(f"{item.group(1)}{mapping[old]}")
                else:
                    rewritten.append(line)
                continue
            in_prereqs = False
        field = re.fullmatch(r"(slug|exercise):\s+\"?([A-Za-z0-9-]+)\"?\s*", line)
        if field:
            old = field.group(2)
            if old in mapping:
                rewritten.append(f"{field.group(1)}: {mapping[old]}")
            else:
                rewritten.append(line)
            continue
        rewritten.append(line)
    return "\n".join(rewritten)


def rewrite_links(body: str, mapping: dict[str, str]) -> str:
    """Replace `/lessons/<slug>` targets. Demo and figure markup stay as they are."""

    def replace(match: re.Match[str]) -> str:
        old = match.group(2)
        return match.group(1) + mapping.get(old, old)

    return LESSON_LINK.sub(replace, body)


def rewrite_lesson_text(text: str, mapping: dict[str, str]) -> str:
    front, body = split_front_matter(text)
    return join_front_matter(
        rewrite_front_matter(front, mapping),
        rewrite_links(body, mapping),
    )


def rewrite_meta_id(text: str, mapping: dict[str, str]) -> str:
    """Replace the `id` field when it names a shifted exercise. Leave the rest."""
    try:
        meta = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InsertError(f"meta.json is not valid JSON: {exc}") from exc
    old = str(meta.get("id", ""))
    if old not in mapping:
        return text
    pattern = re.compile(r'("id"\s*:\s*")' + re.escape(old) + r'(")')
    updated, count = pattern.subn(r"\g<1>" + mapping[old] + r"\g<2>", text, count=1)
    if count != 1:
        raise InsertError(f"meta.json id {old} was not rewritten")
    return updated


def rewrite_curriculum(text: str, mapping: dict[str, str], anchor: str, rows: list[str]) -> str:
    """Update slug cells, then insert the draft rows after the anchor row."""

    def replace(match: re.Match[str]) -> str:
        old = match.group(1)
        return "`" + mapping.get(old, old) + "`"

    updated = BACKTICK_SLUG.sub(replace, text)
    lines = updated.split("\n")
    needle = f"`{anchor}`"
    for index, line in enumerate(lines):
        if line.startswith("|") and needle in line:
            lines[index + 1 : index + 1] = rows
            return "\n".join(lines)
    raise InsertError(f"docs/CURRICULUM.md has no table row for {anchor}")


def curriculum_has_row(text: str, slug: str) -> bool:
    needle = f"`{slug}`"
    return any(line.startswith("|") and needle in line for line in text.split("\n"))


def update_renames(data: dict, mapping: dict[str, str]) -> dict[str, list[str]]:
    """Point old progress at the slug a lesson moved to.

    Existing values that still name a shifted slug are rewritten first, so a
    reader who finished an earlier course id follows the lesson through more
    than one shift. Each shifted slug is then recorded when it is not already
    a key.
    """
    rewritten: dict[str, list[str]] = {}
    for key, values in data.items():
        if not isinstance(values, list):
            raise InsertError(f"renames entry {key} is not a list")
        rewritten[str(key)] = [mapping.get(str(item), str(item)) for item in values]
    for old in sorted(mapping, key=lambda slug: LessonId.parse(slug) or LessonId(0, "", slug)):
        if old not in rewritten:
            rewritten[old] = [mapping[old]]
    return rewritten


def lesson_markdown(slug: str, title: str, part_value: str, prereq: str) -> str:
    summary = "A draft chapter. Replace this stub with the lesson."
    return f"""---
title: {json.dumps(title, ensure_ascii=False)}
slug: {slug}
part: {part_value}
summary: {json.dumps(summary, ensure_ascii=False)}
minutes: 15
gpu: false
draft: true
objectives:
  - State the idea this chapter teaches.
  - Name the function the exercise asks you to write.
  - Say what the checks verify.
prereqs:
  - {prereq}
exercise: {slug}
---

# {title}

This draft holds a place in the course. It names the problem the chapter solves, so the page has an opening, and it stops there.

You replace this text with the chapter. The exercise checks one function so the files load.

## What you already know

You need three facts before the exercise:

- The previous chapter is the one this draft follows.
- A numpy array stores a sequence of numbers and has a shape.
- A function can return the array it was given.

## A placeholder for the idea

The exercise asks for a function that returns its input unchanged. That proves the files load. You replace the function with the one the chapter teaches.

```figure
<svg viewBox="0 0 560 120" role="img" aria-label="A wide bar labeled placeholder">
  <rect x="40" y="40" width="480" height="40" class="fill-soft-a stroke-a"/>
  <text x="280" y="66" text-anchor="middle">placeholder</text>
</svg>
Figure 1. This draft has a figure so the chapter meets the structure rules. Replace it with a figure that carries the idea.
```

## Build it step by step

1. **`pass_through(x)`** takes a 1-D numpy array of samples and returns those samples unchanged. Keep the shape. The usual mistake is to return a Python list, which drops the shape the checks expect.

## The exercise

You write `pass_through`. The checks confirm that it returns the same samples, the same shape, and the same values you passed in.
"""


def exercise_files(slug: str) -> dict[str, str]:
    """The four exercise files for a draft stub."""
    meta = {
        "id": slug,
        "title": "Return the samples unchanged",
        "brief": "Return the array of samples you were given, with the same shape and values.",
        "needs_gpu": False,
        "timeout": 60,
        "packages": ["numpy"],
        "hints": [
            "pass_through returns its argument as a numpy array. Keep the shape."
        ],
    }
    starter = f'''"""Draft exercise for {slug}.

Fill in `pass_through`, then press Run.
"""

import numpy as np


def pass_through(x: np.ndarray) -> np.ndarray:
    """Return `x` unchanged, as a 1-D array of the same shape and values."""
    raise NotImplementedError("pass_through")


if __name__ == "__main__":
    samples = np.array([0.25, -0.5, 0.0])
    print(pass_through(samples))
'''
    solution = f'''"""Draft exercise for {slug}: the worked function."""

import numpy as np


def pass_through(x: np.ndarray) -> np.ndarray:
    """Return `x` unchanged, as a 1-D array of the same shape and values."""
    return np.asarray(x, dtype=float)


if __name__ == "__main__":
    samples = np.array([0.25, -0.5, 0.0])
    print(pass_through(samples))
'''
    tests = '''"""Checks for the draft pass-through exercise."""

import numpy as np

from speechcheck import check, close, has, shape


@check("pass_through keeps the shape of a three-sample clip")
def test_shape(sub):
    fn = has(sub, "pass_through")
    got = fn(np.array([0.25, -0.5, 0.0]))
    shape(got, (3,), label="three samples")


@check("pass_through returns the samples it was given")
def test_values(sub):
    fn = has(sub, "pass_through")
    want = np.array([0.25, -0.5, 0.0])
    close(fn(want), want, label="three samples")


@check("pass_through keeps a single sample")
def test_one(sub):
    fn = has(sub, "pass_through")
    close(fn(np.array([0.5])), np.array([0.5]), label="one sample")


@check("pass_through keeps an empty clip empty")
def test_empty(sub):
    fn = has(sub, "pass_through")
    shape(fn(np.array([])), (0,), label="empty clip")


@check("pass_through keeps a negative sample negative")
def test_negative(sub):
    fn = has(sub, "pass_through")
    close(fn(np.array([-0.25])), np.array([-0.25]), label="negative sample")


@check("pass_through keeps eight samples in order")
def test_order(sub):
    fn = has(sub, "pass_through")
    want = np.arange(8) / 8
    close(fn(want), want, label="eight samples")
'''
    return {
        "meta.json": json.dumps(meta, indent=2) + "\n",
        "starter.py": starter,
        "solution.py": solution,
        "tests.py": tests,
    }


def insert_lessons(
    root: Path,
    after: str,
    slug: str,
    count: int = 1,
    dry_run: bool = False,
) -> Report:
    """Shift later lessons by `count` and write that many draft stubs.

    When `dry_run` is true, return the plan and leave every file as it is.
    """
    if count < 1:
        raise InsertError("--count must be at least 1")
    if LessonId.parse(slug) is not None or NAME_RE.fullmatch(slug) is None:
        raise InsertError(
            "--slug is the name only, in lowercase words separated by hyphens"
        )
    anchor = LessonId.parse(after)
    if anchor is None:
        raise InsertError(
            "--after must be a lesson id such as 030-attention-encoder-decoders"
        )
    if anchor.number + count > 999:
        raise InsertError("lesson numbers stay three digits")

    lessons = lesson_paths(root)
    exercises = exercise_paths(root)
    if after not in lessons:
        raise InsertError(f"no lesson named {after}")

    mapping: dict[str, str] = {}
    for slug_id in set(lessons) | set(exercises):
        ident = LessonId.parse(slug_id)
        if ident is None or not ident > anchor:
            continue
        if ident.number + count > 999:
            raise InsertError(f"{slug_id} would pass 999")
        mapping[slug_id] = ident.shifted(count).slug

    destinations = list(mapping.values())
    if len(destinations) != len(set(destinations)):
        raise InsertError("two lessons would move to the same id")

    stubs: list[str] = []
    for index, name in enumerate(stub_names(slug, count), start=1):
        stubs.append(LessonId(anchor.number + index, "", name).slug)
    if len(stubs) != len(set(stubs)):
        raise InsertError("the draft slugs are not distinct")

    staying = (set(lessons) | set(exercises)) - set(mapping)
    occupied = staying | set(mapping.values())
    for stub in stubs:
        if stub in occupied:
            raise InsertError(f"the draft {stub} collides with an existing lesson")
        occupied.add(stub)

    curriculum_path = root / "docs" / "CURRICULUM.md"
    if not curriculum_path.is_file():
        raise InsertError("docs/CURRICULUM.md is missing")
    curriculum_text = curriculum_path.read_text(encoding="utf-8")
    if not curriculum_has_row(curriculum_text, after):
        raise InsertError(f"docs/CURRICULUM.md has no table row for {after}")

    anchor_text = lessons[after].read_text(encoding="utf-8")
    part_value = field_value(split_front_matter(anchor_text)[0], "part")
    part_display = unquote(part_value)

    renames_path = root / "content" / "renames.json"
    current_renames: dict = {}
    if mapping and renames_path.is_file():
        current_renames = json.loads(renames_path.read_text(encoding="utf-8"))
        if not isinstance(current_renames, dict):
            raise InsertError("content/renames.json must be an object")
        for key, values in current_renames.items():
            if not isinstance(values, list):
                raise InsertError(f"renames entry {key} is not a list")

    noun = "lesson" if count == 1 else "lessons"
    lines = [f"Insert {count} draft {noun} after {after}."]
    if mapping:
        lines.append(f"Shift {len(mapping)} later lesson ids by {count}:")
        for old in sorted(mapping, key=lambda item: LessonId.parse(item) or LessonId(0, "", item)):
            lines.append(f"{old} -> {mapping[old]}")
    else:
        lines.append("No later lesson moves.")
    lines.append(f"Each draft copies {part_display}.")
    for stub in stubs:
        lines.append(f"Add draft {stub}.")

    if dry_run:
        lines.append("Dry run: no files changed.")
        return Report(lines, mapping, stubs)

    for old in sorted(
        mapping,
        key=lambda item: LessonId.parse(item) or LessonId(0, "", item),
        reverse=True,
    ):
        new = mapping[old]
        lesson = lessons.get(old)
        if lesson is not None:
            lesson.rename(lesson.with_name(f"{new}.md"))
        exercise = exercises.get(old)
        if exercise is not None:
            exercise.rename(exercise.with_name(new))

    lessons_dir = root / "content" / "lessons"
    for path in sorted(lessons_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        updated = rewrite_lesson_text(text, mapping)
        if updated != text:
            path.write_text(updated, encoding="utf-8")

    exercises_dir = root / "content" / "exercises"
    for directory in sorted(exercises_dir.iterdir()):
        meta_path = directory / "meta.json"
        if not meta_path.is_file():
            continue
        text = meta_path.read_text(encoding="utf-8")
        updated = rewrite_meta_id(text, mapping)
        if updated != text:
            meta_path.write_text(updated, encoding="utf-8")

    if mapping:
        updated_renames = update_renames(current_renames, mapping)
        renames_path.write_text(
            json.dumps(updated_renames, indent=1) + "\n",
            encoding="utf-8",
        )

    rows = []
    for stub in stubs:
        ident = LessonId.parse(stub)
        assert ident is not None
        title = title_from_name(ident.name)
        rows.append(
            f"| `{stub}` | {title} | "
            "A draft exercise that returns its input unchanged. | none |"
        )
    curriculum_path.write_text(
        rewrite_curriculum(curriculum_text, mapping, after, rows),
        encoding="utf-8",
    )

    previous = after
    for stub in stubs:
        ident = LessonId.parse(stub)
        assert ident is not None
        title = title_from_name(ident.name)
        lesson_path = lessons_dir / f"{stub}.md"
        if lesson_path.exists():
            raise InsertError(f"{lesson_path.name} already exists")
        lesson_path.write_text(
            lesson_markdown(stub, title, part_value, previous),
            encoding="utf-8",
        )
        exercise_dir = exercises_dir / stub
        if exercise_dir.exists():
            raise InsertError(f"content/exercises/{stub} already exists")
        exercise_dir.mkdir()
        for name, contents in exercise_files(stub).items():
            (exercise_dir / name).write_text(contents, encoding="utf-8")
        previous = stub

    lines.append(f"Wrote {count} draft {noun}.")
    return Report(lines, mapping, stubs)


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Insert draft lessons and shift every later lesson."
    )
    parser.add_argument(
        "--after",
        required=True,
        help="The lesson that stays put, as NNN-slug.",
    )
    parser.add_argument(
        "--slug",
        required=True,
        help="The name of the new lesson, without a number.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="How many draft lessons to insert. The default is 1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and change nothing.",
    )
    args = parser.parse_args(argv)
    try:
        report = insert_lessons(
            root or ROOT,
            args.after,
            args.slug,
            count=args.count,
            dry_run=args.dry_run,
        )
    except InsertError as exc:
        print(exc, file=sys.stderr)
        return 2
    print("\n".join(report.lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
