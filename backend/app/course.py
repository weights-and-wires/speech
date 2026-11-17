"""Course-wide facts shared by the REST API and the MCP server."""

from __future__ import annotations

import json
from typing import Any

from . import curriculum, db
from .config import Settings


def book_info(settings: Settings) -> dict[str, Any] | None:
    """The PDF book on disk, as the book build's manifest describes it.

    The image build prints the PDF into the book folder, so a fresh checkout
    that hasn't run the book build has none. Then this is None and the header
    shows no download button, rather than one that leads to a 404.
    """
    try:
        manifest = json.loads((settings.book_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = manifest.get("file")
    if not isinstance(name, str) or "/" in name or not (settings.book_dir / name).is_file():
        return None
    return {
        "title": manifest.get("title"),
        "pages": manifest.get("pages"),
        "bytes": manifest.get("bytes"),
        "built": manifest.get("built"),
        "commit": manifest.get("commit"),
        "url": f"/book/{name}",
    }


def progress_summary(user: str) -> dict[str, Any]:
    """Counts for the hub: lessons done, lessons served, and the latest one touched.

    Only lessons in the current list count, so rows for removed or draft
    lessons are ignored.
    """
    titles = {lesson["slug"]: lesson["title"] for lesson in curriculum.lesson_list()}
    rows = {slug: row for slug, row in db.get_progress(user).items() if slug in titles}
    done = sum(1 for row in rows.values() if row["status"] == "solved")
    last = None
    if rows:
        slug = max(rows, key=lambda key: rows[key]["updated_at"])
        last = {"slug": slug, "title": titles[slug]}
    return {"done": done, "total": len(titles), "last": last}
