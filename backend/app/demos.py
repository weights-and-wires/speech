"""The interactive demos that lessons embed, as the lessons place them.

A lesson places a demo with a fenced ```demo block that holds one JSON object
with the demo's `id`, a `caption`, and optional `props`. The demo itself is a
React component in `frontend/src/demos/`, so this module reads two sources:
the lesson bodies, for where each demo appears, and the registry's source,
`frontend/src/demos/index.ts`, for each demo's title and its order in the
gallery. A checkout that has no registry file, such as an image built without
it, still lists every demo that a lesson places, with a null title.
"""

from __future__ import annotations

import json
import re
from typing import Any

from . import curriculum
from .config import REPO_ROOT

REGISTRY = REPO_ROOT / "frontend" / "src" / "demos" / "index.ts"

# The same block pattern as `scripts/check_content.py` and the lesson renderer.
DEMO_BLOCK = re.compile(r"^```demo[ \t]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
# `"id": {` on a line of its own, then `title: "..."` on the next line.
REGISTRY_ENTRY = re.compile(
    r'^\s*"([a-z0-9-]+)":\s*\{\s*\n\s*title:\s*"((?:[^"\\]|\\.)*)"', re.MULTILINE
)


def registry_titles() -> dict[str, str]:
    """Demo id to title, in gallery order. Empty when the registry isn't on disk."""
    try:
        source = REGISTRY.read_text(encoding="utf-8")
    except OSError:
        return {}
    return {demo_id: _unquote(title) for demo_id, title in REGISTRY_ENTRY.findall(source)}


def _unquote(literal: str) -> str:
    """A TypeScript double-quoted string's value. Its escapes are JSON's."""
    try:
        return str(json.loads(f'"{literal}"'))
    except ValueError:
        return literal


def blocks(body: str) -> list[dict[str, Any]]:
    """The demo blocks of a lesson body, in order, skipping malformed JSON."""
    found = []
    for raw in DEMO_BLOCK.findall(body):
        try:
            parsed = json.loads(raw)
        except ValueError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("id"), str):
            found.append(parsed)
    return found


def demo_list() -> list[dict[str, Any]]:
    """Every demo: its id, title, and the lessons that place it.

    Demos come in gallery order, which is the registry's. A demo that a lesson
    places but the registry lacks follows them, so the list is never missing
    something a reader can see.
    """
    titles = registry_titles()
    demos: dict[str, dict[str, Any]] = {
        demo_id: {"id": demo_id, "title": title, "lessons": []}
        for demo_id, title in titles.items()
    }
    for lesson in curriculum.load_lessons():
        for block in blocks(lesson["body"]):
            demo = demos.setdefault(
                block["id"], {"id": block["id"], "title": None, "lessons": []}
            )
            placement: dict[str, Any] = {
                "slug": lesson["slug"],
                "caption": block.get("caption", ""),
            }
            if isinstance(block.get("props"), dict) and block["props"]:
                placement["props"] = block["props"]
            demo["lessons"].append(placement)
    return list(demos.values())


def get_demo(demo_id: str) -> dict[str, Any] | None:
    return next((demo for demo in demo_list() if demo["id"] == demo_id), None)
