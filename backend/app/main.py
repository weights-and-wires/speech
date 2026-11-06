"""The HTTP surface: SSO identity, content, progress, drafts, and run streams.

All course APIs and the PDF book require a Caddy-authenticated identity.
Health and session are public; the SPA is served from `frontend/dist`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import curriculum, db
from .auth import current_user, sso_user
from .config import Settings, get_settings
from .runner import RunRequest, RunnerError, choose_backend, run_local, run_runpod
from .runner import runpod as runpod_backend

app = FastAPI(title="Weights and Wires — Speech and audio", docs_url=None, redoc_url=None)


@app.on_event("startup")
def startup() -> None:
    db.init()
    db.carry_progress(curriculum.renames())


# Models -----------------------------------------------------------------


class DraftBody(BaseModel):
    code: str = Field(max_length=200_000)


class ProgressBody(BaseModel):
    status: str = Field(pattern="^(started|solved)$")


class RunBody(BaseModel):
    exercise_id: str = Field(min_length=1, max_length=128)
    mode: str = Field(pattern="^(run|check)$")
    code: str = Field(max_length=200_000)
    prefer_gpu: bool = False


# Health and session -----------------------------------------------------


@app.get("/api/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    lessons = curriculum.lesson_list()
    return {
        "ok": True,
        "commit": settings.git_sha,
        "lessons": len(lessons),
        "exercises": len(curriculum.load_exercises()),
        "gpu": settings.gpu_available,
        "book_pages": (book_info(settings) or {}).get("pages", 0),
    }


@app.get("/api/session")
def session_state(
    request: Request, settings: Settings = Depends(get_settings)
) -> dict[str, Any]:
    name = sso_user(request, settings)
    return {"signed_in": bool(name), "username": name}

# Content ----------------------------------------------------------------



@app.get("/api/lessons")
def lessons(user: str = Depends(current_user)) -> dict[str, Any]:
    return {"parts": curriculum.parts(), "progress": db.get_progress(user)}


@app.get("/api/lessons/{slug}")
def lesson(slug: str, user: str = Depends(current_user)) -> dict[str, Any]:
    found = curriculum.get_lesson(slug)
    if found is None:
        raise HTTPException(status_code=404, detail="No such lesson")

    payload: dict[str, Any] = dict(found)
    exercise_id = found.get("exercise")
    if exercise_id:
        exercise = curriculum.get_exercise(str(exercise_id))
        if exercise:
            payload["exercise_detail"] = curriculum.public_exercise(exercise)
            payload["draft"] = db.get_draft(user, str(exercise_id))
    db.set_progress(user, slug, "started")
    return payload


@app.get("/api/exercises/{exercise_id}/solution")
def solution(exercise_id: str, user: str = Depends(current_user)) -> dict[str, Any]:
    exercise = curriculum.get_exercise(exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail="No such exercise")
    return {"id": exercise_id, "solution": exercise["solution"]}


def book_info(settings: Settings) -> dict[str, Any] | None:
    """The PDF book on disk, as the book build's manifest describes it.

    The PDF is built on the deploying machine and copied beside the app, so a
    fresh checkout has none. Then this is None and the header shows no download
    button, rather than one that leads to a 404.
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


@app.get("/api/book")
def book(
    user: str = Depends(current_user), settings: Settings = Depends(get_settings)
) -> dict[str, Any]:
    """The whole course as one PDF, or None when this server has no PDF."""
    return {"book": book_info(settings)}


@app.get("/book/{name}")
def book_file(
    name: str, user: str = Depends(current_user), settings: Settings = Depends(get_settings)
) -> FileResponse:
    """The PDF, sent as a download. Only the file the manifest names is served."""
    info = book_info(settings)
    if info is None or info["url"] != f"/book/{name}":
        raise HTTPException(status_code=404, detail="No such book")
    return FileResponse(settings.book_dir / name, media_type="application/pdf", filename=name)


# Drafts and progress ----------------------------------------------------


@app.put("/api/drafts/{exercise_id}")
def put_draft(
    exercise_id: str, body: DraftBody, user: str = Depends(current_user)
) -> dict[str, Any]:
    if curriculum.get_exercise(exercise_id) is None:
        raise HTTPException(status_code=404, detail="No such exercise")
    db.save_draft(user, exercise_id, body.code)
    return {"saved": True}


@app.delete("/api/drafts/{exercise_id}")
def drop_draft(exercise_id: str, user: str = Depends(current_user)) -> dict[str, Any]:
    db.delete_draft(user, exercise_id)
    return {"saved": False}


@app.put("/api/progress/{slug}")
def put_progress(
    slug: str, body: ProgressBody, user: str = Depends(current_user)
) -> dict[str, Any]:
    db.set_progress(user, slug, body.status)
    return {"slug": slug, "status": body.status}


@app.get("/api/me")
def me(user: str = Depends(current_user), settings: Settings = Depends(get_settings)):
    return {
        "username": user,
        "stats": db.stats(user),
        "runs": db.recent_runs(user),
        "gpu": settings.gpu_available,
    }


@app.get("/api/me/export")
def export(user: str = Depends(current_user)) -> PlainTextResponse:
    return PlainTextResponse(
        db.export_all(user),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{user}-speech.json"'},
    )


@app.get("/api/compute")
async def compute(
    user: str = Depends(current_user), settings: Settings = Depends(get_settings)
) -> dict[str, Any]:
    return {
        "local": {
            "workers": settings.local_workers,
            "memory_mb": settings.local_memory_mb,
            "timeout_ceiling": settings.local_timeout_ceiling,
        },
        "runpod": await runpod_backend.probe(settings),
    }


# The run stream ---------------------------------------------------------


@app.post("/api/run")
async def run(
    body: RunBody,
    user: str = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    exercise = curriculum.get_exercise(body.exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail="No such exercise")

    request = RunRequest(
        exercise_id=body.exercise_id,
        mode=body.mode,
        code=body.code,
        timeout=int(exercise["timeout"]),
        needs_gpu=bool(exercise["needs_gpu"]) or body.prefer_gpu,
    )
    backend = choose_backend(request, settings)

    # A draft is saved on every run, so a closed tab never loses work.
    db.save_draft(user, body.exercise_id, body.code)

    async def stream() -> AsyncIterator[bytes]:
        summary: dict[str, Any] = {"ok": False}
        yield _sse({"type": "begin", "backend": backend, "mode": body.mode})
        try:
            if backend == "runpod":
                source = run_runpod(request, settings)
            else:
                source = run_local(request, Path(exercise["directory"]), settings)
            media = _MediaBudget()
            async for event in source:
                if event.get("type") == "done":
                    summary = event
                yield media.frame(event)
        except RunnerError as exc:
            yield _sse({"type": "error", "message": str(exc)})
            yield _sse({"type": "done", "ok": False})
        except asyncio.CancelledError:
            # The reader navigated away. Nothing to report, and the local runner
            # kills its process group on the way out.
            raise
        except Exception as exc:  # noqa: BLE001 - the stream must always close
            yield _sse({"type": "error", "message": f"The runner failed: {exc}"})
            yield _sse({"type": "done", "ok": False})
        else:
            db.record_run(user, body.exercise_id, body.mode, backend, summary)
            if body.mode == "check" and summary.get("ok"):
                for lesson in curriculum.lesson_list():
                    if lesson.get("exercise") == body.exercise_id:
                        db.set_progress(user, lesson["slug"], "solved")

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _sse(event: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(event, default=str)}\n\n".encode()


# `speechlab` keeps one clip under about 2.7 MB of base64 and a run under 8 MB,
# so these ceilings sit above that and only catch an event built by hand, or a
# worker running older code. They keep one run from holding the page's memory.
MAX_MEDIA_EVENT_BYTES = 4_000_000
MAX_MEDIA_RUN_BYTES = 12_000_000


class _MediaBudget:
    """Passes `audio` and `figure` events through untouched, within a budget."""

    def __init__(self) -> None:
        self.spent = 0

    def frame(self, event: dict[str, Any]) -> bytes:
        framed = _sse(event)
        if event.get("type") not in ("audio", "figure"):
            return framed
        size = len(framed)
        if size > MAX_MEDIA_EVENT_BYTES or self.spent + size > MAX_MEDIA_RUN_BYTES:
            what = event.get("kind") or event.get("type")
            return _sse(
                {
                    "type": "stderr",
                    "text": (
                        f"The server dropped a {what} of {size / 1e6:.1f} MB, because it "
                        "is over the size limit for one run's figures and clips."
                    ),
                }
            )
        self.spent += size
        return framed


# Static assets ----------------------------------------------------------


def _mount_frontend() -> None:
    dist = get_settings().frontend_dist.resolve()
    if not dist.is_dir():
        return
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    index = dist / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> Any:
        if path.startswith("api/"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        candidate = dist / path
        if path and candidate.is_file() and dist in candidate.resolve().parents:
            return FileResponse(candidate)
        return FileResponse(index)


_mount_frontend()
