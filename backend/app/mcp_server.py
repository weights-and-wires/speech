"""An MCP server that gives AI agents the whole course.

The server speaks the MCP Streamable HTTP transport at `/mcp`, statelessly and
with plain JSON responses, so each request stands alone and needs no session.
It exposes lessons, exercises, solutions, the interactive demos, the course
documents, and the book's details as tools, lessons as resources, and the
caller's progress, drafts, and runs.

Every request needs `Authorization: Bearer <token>`, where the token comes
from `mcp_tokens`. The token alone decides who the caller is. The SSO headers
that `auth.current_user` trusts are never read here, because Caddy routes
`/mcp` without Authelia and so can't vouch for them.
"""

from __future__ import annotations

import functools
import hmac
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from . import course, curriculum, db, demos
from .config import REPO_ROOT, get_settings
from .mcp_tokens import user_for_token
from .runs import start_run

# The ASGI scope key that carries the authenticated reader to the tools.
USER_SCOPE_KEY = "speech.mcp_user"

COURSE_TITLE = "Speech and audio"
COURSE_DESCRIPTION = (
    "A hands-on course in speech and audio processing that you learn by building it. "
    "Short chapters run from a sampled waveform through spectrograms, speech recognition, "
    "and speech synthesis to speech language models. Each chapter explains one idea, "
    "works a small example by hand, and links an auto-graded Python exercise whose "
    "checks verify the results. Lessons also embed interactive demos, which an agent "
    "can't run but can read about."
)

INSTRUCTIONS = (
    f"{COURSE_DESCRIPTION}\n\n"
    "Start with `course_overview` for the syllabus and the reader's progress. Use "
    "`search` to find where a topic is taught, `get_lesson` to read a chapter, and "
    "`get_exercise` for its brief, starter code, and checks. `run_exercise` runs code "
    "as the reader, saves it as their draft, and on a passing check marks the lesson "
    "solved. Only call `get_solution` when the reader asks for the answer."
)

DOCS_DIR = REPO_ROOT / "docs"

# Bounds that keep a tool result readable.
MAX_CODE_LENGTH = 200_000
MAX_STREAM_CHARS = 16_000
MAX_CHECK_TEXT = 4_000
MAX_SEARCH_RESULTS = 50
MAX_RUNS = 100

mcp = FastMCP(
    name="speech-course",
    instructions=INSTRUCTIONS,
    stateless_http=True,
    json_response=True,
    # The bearer token is the protection here. DNS-rebinding checks guard
    # unauthenticated servers on localhost, and would reject `speech.ifkash.dev`.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


# Plumbing ---------------------------------------------------------------


def _tool(*, writes: bool = False, destructive: bool = False) -> Callable[[Callable[..., Any]], Any]:
    """Register a tool that returns JSON text only.

    Structured output would send every result twice, as text and as an object,
    which doubles a lesson's size for no gain.
    """
    return mcp.tool(
        structured_output=False,
        annotations=ToolAnnotations(
            readOnlyHint=not writes, destructiveHint=destructive, openWorldHint=False
        ),
    )


def _threaded(fn: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
    """Run a blocking tool body (SQLite, file reads) off the event loop."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))

    return wrapper


def _user(ctx: Context) -> str:
    request = ctx.request_context.request
    user = request.scope.get(USER_SCOPE_KEY) if request is not None else None
    if not user:
        # The endpoint authenticates before the server runs, so this means a
        # transport that bypassed it.
        raise ToolError("Not authenticated.")
    return str(user)


def _exercise_or_error(exercise_id: str) -> dict[str, Any]:
    exercise = curriculum.get_exercise(exercise_id)
    if exercise is None:
        raise ToolError(f"No exercise with ID {exercise_id!r}. Lessons name theirs in `exercise`.")
    return exercise


def _lessons_for(exercise_id: str) -> list[str]:
    return [
        lesson["slug"]
        for lesson in curriculum.lesson_list()
        if lesson.get("exercise") == exercise_id
    ]


def _clip(text: str, limit: int) -> str:
    """Keep the start and the end of long output, where the useful parts are."""
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n... [{omitted} characters omitted] ...\n{text[-tail:]}"


def _lesson_summary(lesson: dict[str, Any], progress: dict[str, Any]) -> dict[str, Any]:
    return {
        "slug": lesson["slug"],
        "title": lesson["title"],
        "part": lesson["part"],
        "summary": lesson["summary"],
        "minutes": lesson.get("minutes"),
        "exercise": lesson.get("exercise"),
        "gpu": lesson.get("gpu", False),
        "status": progress.get(lesson["slug"], {}).get("status"),
    }


# Lesson text ------------------------------------------------------------

_FIGURE_BLOCK = re.compile(r"^```figure[ \t]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)
_DEMO_BLOCK = demos.DEMO_BLOCK
_ARIA_LABEL = re.compile(r'aria-label="([^"]*)"')


def _readable(body: str) -> str:
    """A lesson body an agent can read: figures and demos become text.

    A ```figure block is an inline SVG, hundreds of lines of coordinates, and
    it follows with a caption. This keeps the caption and the SVG's own
    accessible description. A ```demo block names an interactive component,
    which an agent can't use, so this keeps its id and caption. Everything
    else, including code blocks and math, is untouched.
    """

    def figure(match: re.Match[str]) -> str:
        source = match.group(1)
        end = source.rfind("</svg>")
        svg = source[:end] if end >= 0 else source
        caption = source[end + len("</svg>") :].strip() if end >= 0 else ""
        label = _ARIA_LABEL.search(svg)
        lines = []
        if label:
            lines.append(f"[Diagram: {label.group(1)}]")
        if caption:
            lines.append(caption)
        return "> " + "\n> ".join(lines) if lines else "> [Diagram]"

    def demo(match: re.Match[str]) -> str:
        blocks = demos.blocks(f"```demo\n{match.group(1)}```")
        if not blocks:
            return "> [Interactive demo]"
        block = blocks[0]
        caption = f" {block['caption']}" if block.get("caption") else ""
        return f"> [Interactive demo `{block['id']}`.{caption}]"

    return _DEMO_BLOCK.sub(demo, _FIGURE_BLOCK.sub(figure, body))


# Course and lessons -----------------------------------------------------


@_tool()
@_threaded
def course_overview(ctx: Context) -> dict[str, Any]:
    """Get the whole syllabus and the reader's progress in one call.

    Returns the course title and description; every part with its lessons
    (slug, title, summary, minutes, linked exercise ID, and the reader's status:
    null, "started", or "solved"); counts of lessons, exercises, and demos; the
    PDF book's details; the reference documents; whether a GPU worker is
    configured; and a progress summary. Call this first to orient yourself,
    then `get_lesson` with a slug to read a chapter.
    """
    user = _user(ctx)
    settings = get_settings()
    progress = db.get_progress(user)
    parts = [
        {
            "part": part["part"],
            "lessons": [_lesson_summary(lesson, progress) for lesson in part["lessons"]],
        }
        for part in curriculum.parts()
    ]
    documents = _documents()
    return {
        "title": COURSE_TITLE,
        "description": COURSE_DESCRIPTION,
        "reader": user,
        "parts": parts,
        "counts": {
            "parts": len(parts),
            "lessons": sum(len(part["lessons"]) for part in parts),
            "exercises": len(curriculum.load_exercises()),
            "demos": len(demos.demo_list()),
            "documents": len(documents),
        },
        "gpu_available": settings.gpu_available,
        "book": course.book_info(settings),
        "progress": course.progress_summary(user),
        "documents": documents,
    }


@_tool()
@_threaded
def list_lessons(ctx: Context, part: str | None = None) -> dict[str, Any]:
    """List lessons in course order, with their headings and objectives.

    Args:
        part: Only lessons whose part name contains this text, ignoring case,
            for example "Part 3" or "recognition". Omit it for every lesson.

    Each lesson has its slug, title, part, summary, minutes, objectives,
    prerequisites, `##`/`###` headings, linked exercise ID, and the reader's
    status. Bodies are omitted; call `get_lesson` for the full text.
    """
    progress = db.get_progress(_user(ctx))
    needle = (part or "").strip().lower()
    lessons = []
    for index, lesson in enumerate(curriculum.lesson_list(), start=1):
        if needle and needle not in str(lesson["part"]).lower():
            continue
        entry = _lesson_summary(lesson, progress)
        entry.update(
            index=index,
            objectives=lesson["objectives"],
            prereqs=lesson["prereqs"],
            headings=lesson["headings"],
        )
        lessons.append(entry)
    return {"count": len(lessons), "lessons": lessons}


@_tool()
@_threaded
def get_lesson(
    ctx: Context, slug: str, format: Literal["text", "raw"] = "text"
) -> dict[str, Any]:
    """Read one lesson: its full Markdown body and its front matter.

    Args:
        slug: The lesson's slug, such as "006-the-fft-and-windows".
        format: "text" (the default) replaces each figure's SVG with its
            accessible description and caption, and each interactive demo with
            its ID and caption. "raw" returns the Markdown exactly as authored,
            SVG included, which is long.

    Returns the title, part, summary, minutes, objectives, prerequisites,
    headings, the Markdown `body` (math is LaTeX between `$` or `$$`), the
    demos the lesson places (`demos`, for `get_demo`), the previous and next
    slugs, the lesson's position, the linked exercise ID and title, and the
    reader's status. Reading a lesson here doesn't mark it started.
    """
    lesson = curriculum.get_lesson(slug)
    if lesson is None:
        raise ToolError(f"No lesson with slug {slug!r}. Call `list_lessons` for the slugs.")
    payload = dict(lesson)
    if format == "text":
        payload["body"] = _readable(lesson["body"])
    payload["demos"] = [
        {"id": block["id"], "caption": block.get("caption", "")}
        for block in demos.blocks(lesson["body"])
    ]
    exercise_id = lesson.get("exercise")
    exercise = curriculum.get_exercise(str(exercise_id)) if exercise_id else None
    payload["exercise_title"] = exercise["title"] if exercise else None
    payload["status"] = db.get_progress(_user(ctx)).get(slug, {}).get("status")
    return payload


_HEADING_LINE = re.compile(r"^#{1,3}[ \t]+(.+?)[ \t#]*$", re.MULTILINE)


def _snippet(text: str, at: int, length: int) -> str:
    start = max(0, at - 80)
    end = min(len(text), at + length + 160)
    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
    return ("..." if start > 0 else "") + snippet + ("..." if end < len(text) else "")


def _section(body: str, at: int) -> str | None:
    """The heading of the section that position `at` falls in."""
    heading = None
    for match in _HEADING_LINE.finditer(body):
        if match.start() > at:
            break
        heading = match.group(1).strip()
    return heading


def _search_documents() -> list[dict[str, Any]]:
    found = []
    for lesson in curriculum.load_lessons():
        found.append(
            {
                "kind": "lesson",
                "slug": lesson["slug"],
                "title": str(lesson["title"]),
                "part": lesson["part"],
                "summary": str(lesson.get("summary", "")),
                "headings": " \n".join(h["text"] for h in lesson["headings"]),
                "body": _readable(lesson["body"]),
            }
        )
    for exercise in curriculum.load_exercises().values():
        found.append(
            {
                "kind": "exercise",
                "slug": exercise["id"],
                "title": str(exercise["title"]),
                "part": None,
                "summary": str(exercise["brief"]),
                "headings": "",
                "body": "\n".join([exercise["brief"], *map(str, exercise["hints"])]),
            }
        )
    for name, description in _documents().items():
        text = _document_path(name).read_text(encoding="utf-8")
        found.append(
            {
                "kind": "document",
                "slug": name,
                "title": description,
                "part": None,
                "summary": "",
                "headings": " \n".join(m.group(1) for m in _HEADING_LINE.finditer(text)),
                "body": text,
            }
        )
    return found


@_tool()
@_threaded
def search(query: str, limit: int = 10) -> dict[str, Any]:
    """Search lessons, exercises, and course documents by text, ignoring case.

    Args:
        query: Words to find, such as "spectral leakage" or "CTC loss". A
            document matches when it contains every word; an exact phrase match
            ranks higher, as do matches in titles and headings.
        limit: The most results to return, 1 to 50. Default 10.

    Each result has its kind ("lesson", "exercise", or "document"), slug, title,
    part, the section heading the hit falls under, and a short snippet around the
    hit. Pass a lesson slug to `get_lesson`, an exercise's slug (its ID) to
    `get_exercise`, and a document's slug (its name) to `get_document`.
    """
    phrase = " ".join(query.lower().split())
    terms = phrase.split()
    if not terms:
        raise ToolError("The query is empty.")
    limit = max(1, min(int(limit), MAX_SEARCH_RESULTS))

    results = []
    for doc in _search_documents():
        title = doc["title"].lower()
        headings = doc["headings"].lower()
        body_lower = doc["body"].lower()
        haystack = " ".join((title, doc["summary"].lower(), headings, body_lower))
        if not all(term in haystack for term in terms):
            continue
        score = 0.0
        if phrase in title:
            score += 20
        score += 6 * sum(term in title for term in terms)
        if phrase in headings:
            score += 8
        score += 2 * sum(term in headings for term in terms)
        score += min(body_lower.count(phrase), 10) * 1.5
        score += sum(min(body_lower.count(term), 10) for term in terms) * 0.2

        at = body_lower.find(phrase)
        length = len(phrase)
        if at < 0:
            at = next((body_lower.find(t) for t in terms if t in body_lower), -1)
            length = len(terms[0])
        if at >= 0:
            snippet = _snippet(doc["body"], at, length)
            section = _section(doc["body"], at)
        else:
            snippet = doc["summary"]
            section = None
        results.append(
            {
                "kind": doc["kind"],
                "slug": doc["slug"],
                "title": doc["title"],
                "part": doc["part"],
                "section": section,
                "snippet": snippet,
                "score": round(score, 2),
            }
        )
    results.sort(key=lambda result: result["score"], reverse=True)
    return {"query": query, "total": len(results), "results": results[:limit]}


# Exercises --------------------------------------------------------------


@_tool()
@_threaded
def list_exercises(ctx: Context) -> dict[str, Any]:
    """List every exercise, in course order, without its code.

    Each has its ID, title, brief, timeout in seconds, whether it needs a GPU,
    the packages it imports, the lessons that link to it, and whether the reader
    has a saved draft. Call `get_exercise` for the starter code and checks.
    """
    drafts = {row["exercise_id"] for row in db.list_drafts(_user(ctx))}
    linked: dict[str, list[str]] = {}
    for lesson in curriculum.lesson_list():
        if lesson.get("exercise"):
            linked.setdefault(str(lesson["exercise"]), []).append(lesson["slug"])
    exercises = [
        {
            "id": exercise["id"],
            "title": exercise["title"],
            "brief": exercise["brief"],
            "timeout": exercise["timeout"],
            "needs_gpu": exercise["needs_gpu"],
            "packages": exercise["packages"],
            "lessons": linked.get(exercise["id"], []),
            "has_draft": exercise["id"] in drafts,
        }
        for exercise in curriculum.load_exercises().values()
    ]
    return {"count": len(exercises), "exercises": exercises}


@_tool()
@_threaded
def get_exercise(exercise_id: str) -> dict[str, Any]:
    """Get an exercise: its brief, hints, starter code, and the checks' source.

    Args:
        exercise_id: The exercise ID, which a lesson names in its `exercise`
            field. It usually equals the lesson slug.

    Returns the title, brief, hints, `starter` (the file the reader edits),
    `tests` (the source of tests.py, the checks a `check` run applies), the
    timeout in seconds, whether it needs a GPU, the packages it imports, and the
    lessons that link to it. The worked solution is withheld; see `get_solution`.
    """
    exercise = _exercise_or_error(exercise_id)
    payload = curriculum.public_exercise(exercise)
    tests = Path(exercise["directory"]) / "tests.py"
    payload["tests"] = tests.read_text(encoding="utf-8") if tests.is_file() else ""
    payload["lessons"] = _lessons_for(exercise_id)
    return payload


@_tool()
@_threaded
def get_solution(exercise_id: str) -> dict[str, Any]:
    """Get an exercise's worked solution.

    Args:
        exercise_id: The exercise ID.

    The solution gives the answer away. Only fetch it when the reader asks to
    see it, or to compare after they have solved the exercise.
    """
    exercise = _exercise_or_error(exercise_id)
    return {"id": exercise_id, "solution": exercise["solution"]}


# Demos, documents, and the book -----------------------------------------


@_tool()
@_threaded
def list_demos() -> dict[str, Any]:
    """List the interactive demos that lessons embed, in gallery order.

    Each has its ID, its title, and the lessons that place it, with the
    caption each lesson gives it. A demo is a browser component with controls
    and sound, so an agent reads its description rather than running it. The
    reader opens one at /demos/ID on the site.
    """
    found = demos.demo_list()
    return {"count": len(found), "demos": found}


@_tool()
@_threaded
def get_demo(demo_id: str) -> dict[str, Any]:
    """Get one interactive demo: its title and where lessons place it.

    Args:
        demo_id: The demo's ID, from `list_demos` or a lesson's `demos`.

    Returns the ID, the title, the path the reader opens it at, and each
    lesson that places it, with that lesson's caption (the sentence about what
    to notice) and any options the lesson sets.
    """
    found = demos.get_demo(demo_id)
    if found is None:
        raise ToolError(f"No demo with ID {demo_id!r}. Call `list_demos` for the IDs.")
    return {**found, "path": f"/demos/{demo_id}"}


def _documents() -> dict[str, str]:
    """The reference documents under `docs/`, by name, with a one-line description.

    A name is the path under `docs/`, such as "CURRICULUM.md" or
    "research/02-asr-and-ssl.md". The description is the document's title.
    Nothing else on disk is reachable through `get_document`.
    """
    paths = sorted(DOCS_DIR.glob("*.md")) + sorted((DOCS_DIR / "research").glob("*.md"))
    found = {}
    for path in paths:
        title = path.stem
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
        except OSError:
            continue
        found[path.relative_to(DOCS_DIR).as_posix()] = title
    return found


def _document_path(name: str) -> Path:
    return DOCS_DIR / name


@_tool()
@_threaded
def get_document(name: str) -> dict[str, Any]:
    """Read one of the course's reference documents, in Markdown.

    Args:
        name: A document's name from `course_overview`: "CURRICULUM.md" for the
            curriculum and the reasoning behind its order, "CONTENT_CONTRACT.md"
            for the format every lesson and exercise follows, "DEMOS.md" for how
            demos are built, or "research/NAME.md" for a part's research notes.
            Case and the ".md" are optional.
    """
    documents = _documents()
    wanted = name.strip().removeprefix("docs/").lower()
    wanted = wanted if wanted.endswith(".md") else wanted + ".md"
    match = next((known for known in documents if known.lower() == wanted), None)
    if match is None:
        raise ToolError(f"No document {name!r}. Choose one of: {', '.join(documents)}.")
    return {
        "name": match,
        "description": documents[match],
        "markdown": _document_path(match).read_text(encoding="utf-8"),
    }


@_tool()
@_threaded
def get_book_info() -> dict[str, Any]:
    """Get the details of the course's PDF book, or null when this server has none.

    Returns the title, page count, size in bytes, build time, and commit. The
    book is the lessons, the exercises, and the solutions in one PDF. Its
    download needs a signed-in browser session, so an agent can't fetch it; read
    the same text with `get_lesson`, `get_exercise`, and `get_solution`.
    """
    return {"book": course.book_info(get_settings())}


# The reader's state -----------------------------------------------------


@_tool()
@_threaded
def get_progress(ctx: Context, runs: int = 10) -> dict[str, Any]:
    """Get the reader's progress: per-lesson status, totals, and recent runs.

    Args:
        runs: How many recent runs to return, 0 to 100. Default 10.

    `lessons` lists every lesson the reader has started or solved, in course
    order, with its status and when it last changed (Unix seconds). `stats` has
    the solved count, run count, and compute seconds. `runs` has the most
    recent runs, newest first.
    """
    user = _user(ctx)
    progress = db.get_progress(user)
    lessons = [
        {
            "slug": lesson["slug"],
            "title": lesson["title"],
            "status": progress[lesson["slug"]]["status"],
            "updated_at": progress[lesson["slug"]]["updated_at"],
        }
        for lesson in curriculum.lesson_list()
        if lesson["slug"] in progress
    ]
    return {
        "reader": user,
        "summary": course.progress_summary(user),
        "lessons": lessons,
        "stats": db.stats(user),
        "runs": db.recent_runs(user, limit=max(0, min(int(runs), MAX_RUNS))),
    }


@_tool()
@_threaded
def list_drafts(ctx: Context) -> dict[str, Any]:
    """List the exercises the reader has saved code for, newest first.

    Each has the exercise ID, the draft's length in characters, and when it was
    saved (Unix seconds). Call `get_draft` for the code.
    """
    drafts = [
        row
        for row in db.list_drafts(_user(ctx))
        if curriculum.get_exercise(row["exercise_id"]) is not None
    ]
    return {"count": len(drafts), "drafts": drafts}


@_tool()
@_threaded
def get_draft(ctx: Context, exercise_id: str) -> dict[str, Any]:
    """Get the reader's saved code for an exercise.

    Args:
        exercise_id: The exercise ID.

    `code` is null when the reader has no draft; the editor then shows the
    starter code from `get_exercise`.
    """
    _exercise_or_error(exercise_id)
    return {"exercise_id": exercise_id, "code": db.get_draft(_user(ctx), exercise_id)}


@_tool(writes=True)
@_threaded
def save_draft(ctx: Context, exercise_id: str, code: str) -> dict[str, Any]:
    """Save code as the reader's draft for an exercise.

    Args:
        exercise_id: The exercise ID.
        code: The whole file, which replaces the reader's current draft. The
            reader sees it in the editor the next time they open the lesson.
    """
    _exercise_or_error(exercise_id)
    if len(code) > MAX_CODE_LENGTH:
        raise ToolError(f"The code is longer than {MAX_CODE_LENGTH} characters.")
    db.save_draft(_user(ctx), exercise_id, code)
    return {"exercise_id": exercise_id, "saved": True}


@_tool(writes=True, destructive=True)
@_threaded
def delete_draft(ctx: Context, exercise_id: str) -> dict[str, Any]:
    """Delete the reader's saved code for an exercise, so the editor shows the starter.

    Args:
        exercise_id: The exercise ID.

    This is the editor's reset button. The code is gone for good, so only call
    it when the reader asks.
    """
    _exercise_or_error(exercise_id)
    db.delete_draft(_user(ctx), exercise_id)
    return {"exercise_id": exercise_id, "saved": False}


@_tool(writes=True)
@_threaded
def set_progress(
    ctx: Context, slug: str, status: Literal["started", "solved"]
) -> dict[str, Any]:
    """Mark a lesson started or solved for the reader.

    Args:
        slug: The lesson's slug.
        status: "started" or "solved". A solved lesson is never downgraded to
            started. A passing `run_exercise` check marks its lesson solved on
            its own, so you rarely need this.
    """
    if curriculum.get_lesson(slug) is None:
        raise ToolError(f"No lesson with slug {slug!r}.")
    user = _user(ctx)
    db.set_progress(user, slug, status)
    return {"slug": slug, "status": db.get_progress(user)[slug]["status"]}


def _bounds(values: list[Any]) -> tuple[float | None, float | None]:
    numbers = [v for v in values if isinstance(v, (int, float))]
    return (min(numbers), max(numbers)) if numbers else (None, None)


def _media_summary(event: dict[str, Any]) -> dict[str, Any]:
    """Describe an `audio` or `figure` event without its payload.

    A clip is base64 WAV and a heatmap is base64 pixels, which an agent can't
    use as text and which would fill its context. This keeps what a reader
    would learn from the caption: what it is, how big, and over which axes.
    """
    summary: dict[str, Any] = {"label": event.get("label"), "note": event.get("note")}
    if event.get("check") is not None:
        summary["check"] = event["check"]
    if event.get("type") == "audio":
        summary.update(
            kind="audio",
            seconds=event.get("seconds"),
            sample_rate=event.get("sample_rate"),
            channels=event.get("channels"),
            truncated=bool(event.get("truncated")),
        )
    elif event.get("kind") == "heatmap":
        summary.update(
            kind="heatmap",
            shape=event.get("shape"),
            x=event.get("x"),
            y=event.get("y"),
            value=event.get("value"),
        )
    elif event.get("kind") == "line":
        series = []
        for item in event.get("series") or []:
            x_min, x_max = _bounds(item.get("x") or [])
            y_min, y_max = _bounds(item.get("y") or [])
            series.append(
                {
                    "label": item.get("label"),
                    "points": item.get("points"),
                    "x_range": [x_min, x_max],
                    "y_range": [y_min, y_max],
                }
            )
        summary.update(
            kind="line", xlabel=event.get("xlabel"), ylabel=event.get("ylabel"), series=series
        )
    else:
        summary["kind"] = event.get("kind") or event.get("type")
    return summary


@_tool(writes=True)
async def run_exercise(
    ctx: Context,
    exercise_id: str,
    code: str,
    mode: Literal["check", "run"] = "check",
) -> dict[str, Any]:
    """Run code for an exercise on the course's runner, as the reader.

    Args:
        exercise_id: The exercise ID.
        code: The whole submission file, starting from the starter code.
        mode: "check" runs the exercise's checks against the code and reports
            each one. "run" executes the file and returns its output.

    This does what the editor's Run and Check buttons do: it saves the code as
    the reader's draft, records the run, and marks the linked lessons solved
    when every check passes. It waits for the run to finish, which can take up
    to the exercise's timeout.

    Returns `ok`, `passed` and `total` checks, `seconds`, the backend used,
    each check's description, status, and failure message, the program's
    stdout and stderr (long output is trimmed in the middle), and any errors.
    Clips and figures that the code plays or plots come back as `media`
    entries that describe them, without the audio or pixels.
    """
    user = _user(ctx)
    if len(code) > MAX_CODE_LENGTH:
        raise ToolError(f"The code is longer than {MAX_CODE_LENGTH} characters.")
    settings = get_settings()
    run = await anyio.to_thread.run_sync(
        lambda: start_run(user, exercise_id, mode, code, settings)
    )
    if run is None:
        raise ToolError(f"No exercise with ID {exercise_id!r}.")

    stdout: list[str] = []
    stderr: list[str] = []
    checks: list[dict[str, Any]] = []
    media: list[dict[str, Any]] = []
    errors: list[str] = []
    status: list[str] = []
    done: dict[str, Any] = {"ok": False}
    async for event in run.events:
        kind = event.get("type")
        if kind == "stdout":
            stdout.append(str(event.get("text", "")))
        elif kind == "stderr":
            stderr.append(str(event.get("text", "")))
        elif kind in ("audio", "figure"):
            media.append(_media_summary(event))
        elif kind == "check":
            check = {
                "index": event.get("index"),
                "description": event.get("description"),
                "status": event.get("status"),
                "seconds": event.get("seconds"),
            }
            for key in ("message", "output"):
                if event.get(key):
                    check[key] = _clip(str(event[key]), MAX_CHECK_TEXT)
            checks.append(check)
        elif kind == "error":
            errors.append(str(event.get("message", "")))
        elif kind == "status":
            status.append(str(event.get("text", "")))
        elif kind == "done":
            done = event

    return {
        "exercise_id": exercise_id,
        "mode": mode,
        "backend": run.backend,
        "ok": bool(done.get("ok")),
        "passed": done.get("passed"),
        "total": done.get("total"),
        "seconds": done.get("seconds"),
        "timeout": bool(done.get("timeout")),
        "checks": checks,
        "stdout": _clip("".join(stdout), MAX_STREAM_CHARS),
        "stderr": _clip("".join(stderr), MAX_STREAM_CHARS),
        "errors": errors,
        "status": status,
        "media": media,
        "lessons_solved": _lessons_for(exercise_id) if mode == "check" and done.get("ok") else [],
    }


# Resources --------------------------------------------------------------


@mcp.resource(
    "speech://lessons/{slug}",
    name="lesson",
    description=(
        "A lesson's Markdown body, with figures and demos as text. "
        "The slug comes from `list_lessons`."
    ),
    mime_type="text/markdown",
)
@_threaded
def lesson_resource(slug: str) -> str:
    lesson = curriculum.get_lesson(slug)
    if lesson is None:
        raise ValueError(f"No lesson with slug {slug!r}.")
    return _readable(lesson["body"])


# The HTTP endpoint ------------------------------------------------------

_manager: StreamableHTTPSessionManager | None = None


@asynccontextmanager
async def lifespan() -> AsyncIterator[None]:
    """Run the MCP session manager for the life of the app.

    A session manager runs only once, so each app start makes a new one. That
    lets tests start the app more than once in one process.
    """
    global _manager
    manager = StreamableHTTPSessionManager(
        app=mcp._mcp_server,
        json_response=True,
        stateless=True,
        security_settings=mcp.settings.transport_security,
    )
    async with manager.run():
        _manager = manager
        try:
            yield
        finally:
            _manager = None


def _unauthorized(error: str | None, description: str) -> JSONResponse:
    challenge = 'Bearer realm="speech-mcp"'
    if error:
        challenge += f', error="{error}"'
    return JSONResponse(
        {"error": error or "unauthorized", "error_description": description},
        status_code=401,
        headers={"WWW-Authenticate": challenge},
    )


class _Endpoint:
    """The ASGI app at `/mcp`: check the bearer token, then hand over to MCP.

    A class rather than a function, because Starlette wraps a plain function
    route as a request handler instead of mounting it as an ASGI app.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await _serve(scope, receive, send)


def _hub_user(headers: Headers) -> str | None:
    """The reader the MCP hub vouches for, or None without a valid hub secret.

    The hub sends its shared secret and the reader's Authelia username. The
    username maps through SSO_USERS, as a signed-in browser session does.
    Caddy strips both headers from public requests to /mcp.
    """
    settings = get_settings()
    secret = settings.mcp_hub_secret
    sent = headers.get("x-mcp-hub-secret", "")
    name = headers.get("x-mcp-hub-user", "").strip()
    if not secret or not sent or not name:
        return None
    if not hmac.compare_digest(sent.encode(), secret.encode()):
        return None
    return settings.sso_names.get(name.lower(), name)


async def _serve(scope: Scope, receive: Receive, send: Send) -> None:
    headers = Headers(scope=scope)
    hub_user = _hub_user(headers)
    if hub_user is not None:
        await _dispatch(hub_user, scope, receive, send)
        return
    scheme, _, token = headers.get("authorization", "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        response = _unauthorized(None, "Send an MCP token as 'Authorization: Bearer <token>'.")
        await response(scope, receive, send)
        return
    user = await anyio.to_thread.run_sync(user_for_token, token)
    if user is None:
        response = _unauthorized("invalid_token", "The token is unknown or revoked.")
        await response(scope, receive, send)
        return
    await _dispatch(user, scope, receive, send)


async def _dispatch(user: str, scope: Scope, receive: Receive, send: Send) -> None:
    if _manager is None:
        response = JSONResponse({"error": "The MCP server isn't running."}, status_code=503)
        await response(scope, receive, send)
        return
    await _manager.handle_request({**scope, USER_SCOPE_KEY: user}, receive, send)


endpoint = _Endpoint()
