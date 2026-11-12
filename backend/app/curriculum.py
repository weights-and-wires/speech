"""Loads lessons and exercises from `content/`.

A lesson is one Markdown file with YAML front matter. An exercise is a directory
holding `meta.json`, `starter.py`, `tests.py`, and `solution.py`. Both are cached
after the first read; `reload_content` clears the cache while you are writing.

The format is specified in `docs/CONTENT_CONTRACT.md`.

A lesson with `draft: true` stays in this list while `frontend/dist` is
absent. The deployed app serves that bundle, and `main._mount_frontend`
mounts it only when the directory exists. Drafts follow the same check, so
a stub is hidden when the bundle is served.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import frontmatter

from .config import get_settings


def _text_list(value: Any) -> list[str]:
    """Coerce a front-matter list to strings.

    An unquoted colon in a YAML list item parses as a mapping rather than a
    string, and a mapping reaching the client renders as nothing at all — React
    refuses to render an object and blanks the page. `scripts/check_content.py`
    rejects that at authoring time; this keeps a slip from taking the app down.
    """
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            out.extend(f"{k}: {v}" for k, v in item.items())
        else:
            out.append(str(item))
    return out


_FENCE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.DOTALL | re.MULTILINE)
_HEADING = re.compile(r"^(#{2,3})[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


def _plain(text: str) -> str:
    """Strip inline Markdown and math delimiters from heading text."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[`*$]", "", text).strip()


def _headings(body: str) -> list[dict[str, Any]]:
    """The `##` and `###` headings of a lesson body, in order, as plain text.

    Each entry is `{"text": ..., "level": 2 or 3}`. Headings inside fenced code
    blocks are skipped. The search dialog matches on these, and the reader gives
    every `##` heading an anchor built from the same text.
    """
    out: list[dict[str, Any]] = []
    for match in _HEADING.finditer(_FENCE.sub("", body)):
        text = _plain(match.group(2))
        if text:
            out.append({"text": text, "level": len(match.group(1))})
    return out


def _lessons_dir() -> Path:
    return get_settings().content_dir / "lessons"


def _exercises_dir() -> Path:
    return get_settings().content_dir / "exercises"


def _drafts_visible() -> bool:
    """Whether draft lessons belong in the list this process serves.

    Production serves the built frontend from `frontend/dist`. That is the
    directory `main._mount_frontend` mounts, and it mounts nothing when the
    directory is absent. A local checkout has no bundle there, so drafts stay
    visible while you write them.
    """
    return not get_settings().frontend_dist.is_dir()


def _draft_exercise_ids() -> set[str]:
    """Exercise ids of draft lessons, when this process hides drafts."""
    if _drafts_visible():
        return set()
    directory = _lessons_dir()
    if not directory.is_dir():
        return set()
    hidden: set[str] = set()
    for path in directory.glob("*.md"):
        meta = frontmatter.load(path).metadata
        if meta.get("draft") is not True:
            continue
        hidden.add(str(meta.get("slug") or path.stem))
        exercise = meta.get("exercise")
        if exercise:
            hidden.add(str(exercise))
    return hidden


def renames() -> dict[str, list[str]]:
    """Old lesson slugs and the slugs that replaced them, from `renames.json`.

    When a lesson is split or renamed, its entry here lets readers keep their
    progress. A missing file means nothing was renamed.
    """
    path = get_settings().content_dir / "renames.json"
    if not path.is_file():
        return {}
    return {str(k): [str(s) for s in v] for k, v in json.loads(path.read_text()).items()}


@lru_cache
def load_lessons() -> list[dict[str, Any]]:
    """Every lesson, ordered by the numeric prefix of its filename.

    Draft lessons are included while `frontend/dist` is absent, and omitted
    when that bundle is on disk.
    """
    lessons: list[dict[str, Any]] = []
    directory = _lessons_dir()
    if not directory.is_dir():
        return lessons
    for path in sorted(directory.glob("*.md")):
        post = frontmatter.load(path)
        meta = dict(post.metadata)
        if meta.get("draft") is True and not _drafts_visible():
            continue
        slug = str(meta.get("slug") or path.stem)
        lessons.append(
            {
                "slug": slug,
                "title": meta.get("title", slug),
                "part": meta.get("part", "Uncategorised"),
                "summary": meta.get("summary", ""),
                "minutes": meta.get("minutes"),
                "objectives": _text_list(meta.get("objectives", [])),
                "prereqs": _text_list(meta.get("prereqs", [])),
                "exercise": meta.get("exercise"),
                "gpu": bool(meta.get("gpu", False)),
                "headings": _headings(post.content),
                "body": post.content,
            }
        )
    return lessons


def lesson_list() -> list[dict[str, Any]]:
    """Lesson metadata without the body, for the sidebar and the dashboard."""
    return [{k: v for k, v in lesson.items() if k != "body"} for lesson in load_lessons()]


def parts() -> list[dict[str, Any]]:
    """Lessons grouped into parts, in the order the parts first appear."""
    grouped: list[dict[str, Any]] = []
    index: dict[str, dict[str, Any]] = {}
    for lesson in lesson_list():
        part = lesson["part"]
        if part not in index:
            index[part] = {"part": part, "lessons": []}
            grouped.append(index[part])
        index[part]["lessons"].append(lesson)
    return grouped


def get_lesson(slug: str) -> dict[str, Any] | None:
    """One lesson with its body, and the slugs on either side of it."""
    lessons = load_lessons()
    order = {lesson["slug"]: i for i, lesson in enumerate(lessons)}
    if slug not in order:
        return None
    i = order[slug]
    lesson = dict(lessons[i])
    lesson["prev"] = lessons[i - 1]["slug"] if i > 0 else None
    lesson["next"] = lessons[i + 1]["slug"] if i + 1 < len(lessons) else None
    lesson["index"] = i + 1
    lesson["count"] = len(lessons)
    return lesson


@lru_cache
def load_exercises() -> dict[str, dict[str, Any]]:
    exercises: dict[str, dict[str, Any]] = {}
    root = _exercises_dir()
    if not root.is_dir():
        return exercises
    hidden = _draft_exercise_ids()
    for path in sorted(root.iterdir()):
        meta_file = path / "meta.json"
        if not meta_file.is_file():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except json.JSONDecodeError:
            continue
        exercise_id = str(meta.get("id", path.name))
        if exercise_id in hidden or path.name in hidden:
            continue

        def read(name: str) -> str:
            target = path / name
            return target.read_text() if target.is_file() else ""

        exercises[exercise_id] = {
            "id": exercise_id,
            "title": meta.get("title", exercise_id),
            "brief": meta.get("brief", ""),
            "needs_gpu": bool(meta.get("needs_gpu", False)),
            "timeout": int(meta.get("timeout", 60)),
            "packages": meta.get("packages", []),
            "hints": meta.get("hints", []),
            "starter": read("starter.py"),
            "solution": read("solution.py"),
            "directory": str(path),
        }
    return exercises


def get_exercise(exercise_id: str) -> dict[str, Any] | None:
    return load_exercises().get(exercise_id)


def public_exercise(exercise: dict[str, Any]) -> dict[str, Any]:
    """The exercise payload for the client, with the solution withheld."""
    return {k: v for k, v in exercise.items() if k not in ("solution", "directory")}


def reload_content() -> None:
    load_lessons.cache_clear()
    load_exercises.cache_clear()
