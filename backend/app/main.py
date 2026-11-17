"""The HTTP surface: SSO identity, content, progress, drafts, and run streams.

All course APIs and the PDF book require a Caddy-authenticated identity.
Health and session are public; the SPA is served from `frontend/dist`. The MCP
server at `/mcp` is the exception: it ignores the proxy identity and takes a
bearer token.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import course, curriculum, db, mcp_server, mcp_tokens
from .auth import current_user, sso_user
from .config import Settings, get_settings
from .runner import runpod as runpod_backend
from .runs import start_run


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    db.init()
    db.carry_progress(curriculum.renames())
    async with mcp_server.lifespan():
        yield


app = FastAPI(
    title="Weights and Wires — Speech and audio",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


# The hub at this origin reads the progress summary from the browser.
HUB_ORIGIN = "https://courses.ifkash.dev"
SUMMARY_PATH = "/api/progress/summary"


@app.middleware("http")
async def hub_cors(request: Request, call_next: Any) -> Any:
    """Allow the hub origin to read the progress summary, and nothing else.

    This is a middleware rather than `CORSMiddleware` so that it applies to one
    path only, and so that the 401 from `current_user` carries the headers too.
    """
    response = await call_next(request)
    if request.url.path == SUMMARY_PATH:
        response.headers.append("Vary", "Origin")
        if request.headers.get("origin") == HUB_ORIGIN:
            response.headers["Access-Control-Allow-Origin"] = HUB_ORIGIN
            response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


# Models -----------------------------------------------------------------


class DraftBody(BaseModel):
    code: str = Field(max_length=200_000)


class ProgressBody(BaseModel):
    status: str = Field(pattern="^(started|solved)$")


class TokenBody(BaseModel):
    name: str = Field(min_length=1, max_length=mcp_tokens.MAX_NAME_LENGTH)


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
        "book_pages": (course.book_info(settings) or {}).get("pages", 0),
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


@app.get("/api/book")
def book(
    user: str = Depends(current_user), settings: Settings = Depends(get_settings)
) -> dict[str, Any]:
    """The whole course as one PDF, or None when this server has no PDF."""
    return {"book": course.book_info(settings)}


@app.get("/book/{name}")
def book_file(
    name: str, user: str = Depends(current_user), settings: Settings = Depends(get_settings)
) -> FileResponse:
    """The PDF, sent as a download. Only the file the manifest names is served."""
    info = course.book_info(settings)
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


@app.get("/api/progress/summary")
def progress_summary(user: str = Depends(current_user)) -> dict[str, Any]:
    """Counts for the hub: lessons done, lessons served, and the latest one touched."""
    return course.progress_summary(user)


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
    started = start_run(
        user, body.exercise_id, body.mode, body.code, settings, prefer_gpu=body.prefer_gpu
    )
    if started is None:
        raise HTTPException(status_code=404, detail="No such exercise")

    async def stream() -> AsyncIterator[bytes]:
        media = _MediaBudget()
        async for event in started.events:
            yield media.frame(event)

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


# MCP --------------------------------------------------------------------


def _token_view(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in ("id", "name", "prefix", "created_at", "last_used_at")}


@app.get("/api/mcp/tokens")
def list_mcp_tokens(
    request: Request,
    user: str = Depends(current_user),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """The MCP endpoint's URL and the caller's active tokens, without secrets."""
    url = settings.mcp_public_url or str(request.base_url).rstrip("/") + "/mcp"
    return {"url": url, "tokens": [_token_view(t) for t in db.list_tokens(user)]}


@app.post("/api/mcp/tokens")
def create_mcp_token(body: TokenBody, user: str = Depends(current_user)) -> dict[str, Any]:
    """Create a token for the caller. The response is the only copy of the secret."""
    try:
        record = mcp_tokens.create_token(user, body.name)
    except mcp_tokens.TokenLimitError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {key: record[key] for key in ("id", "name", "prefix", "created_at", "token")}


@app.delete("/api/mcp/tokens/{token_id}")
def revoke_mcp_token(token_id: int, user: str = Depends(current_user)) -> dict[str, Any]:
    if not db.revoke_token(token_id, username=user):
        raise HTTPException(status_code=404, detail="No such token")
    return {"revoked": True}


# Both spellings are routes of their own, so neither redirects: MCP clients
# don't follow a redirect on POST. They are registered before the SPA catch-all.
app.router.add_route("/mcp", mcp_server.endpoint, include_in_schema=False)
app.router.add_route("/mcp/", mcp_server.endpoint, include_in_schema=False)


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
        # .well-known/ too: an MCP client probes it for OAuth metadata, and an
        # HTML page with 200 there reads as a broken OAuth server.
        if path.startswith(("api/", "mcp/", ".well-known/")):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        candidate = dist / path
        if path and candidate.is_file() and dist in candidate.resolve().parents:
            return FileResponse(candidate)
        return FileResponse(index)


_mount_frontend()
