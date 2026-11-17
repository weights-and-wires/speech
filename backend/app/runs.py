"""One run of a reader's submission, shared by `/api/run` and the MCP server.

`start_run` saves the draft, picks the backend, and returns the event stream.
The stream records the run and marks the lesson solved after a passing check,
so both callers have the same side effects.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from . import curriculum, db
from .config import Settings
from .runner import RunRequest, RunnerError, choose_backend, run_local, run_runpod


@dataclass
class Run:
    backend: str
    events: AsyncIterator[dict[str, Any]]


def start_run(
    user: str,
    exercise_id: str,
    mode: str,
    code: str,
    settings: Settings,
    prefer_gpu: bool = False,
) -> Run | None:
    """Prepare a run, or return None when the exercise doesn't exist."""
    exercise = curriculum.get_exercise(exercise_id)
    if exercise is None:
        return None

    request = RunRequest(
        exercise_id=exercise_id,
        mode=mode,
        code=code,
        timeout=int(exercise["timeout"]),
        needs_gpu=bool(exercise["needs_gpu"]) or prefer_gpu,
    )
    backend = choose_backend(request, settings)

    # A draft is saved on every run, so a closed tab never loses work.
    db.save_draft(user, exercise_id, code)

    async def events() -> AsyncIterator[dict[str, Any]]:
        summary: dict[str, Any] = {"ok": False}
        yield {"type": "begin", "backend": backend, "mode": mode}
        try:
            if backend == "runpod":
                source = run_runpod(request, settings)
            else:
                source = run_local(request, Path(exercise["directory"]), settings)
            async for event in source:
                if event.get("type") == "done":
                    summary = event
                yield event
        except RunnerError as exc:
            yield {"type": "error", "message": str(exc)}
            yield {"type": "done", "ok": False}
        except asyncio.CancelledError:
            # The reader navigated away. Nothing to report, and the local runner
            # kills its process group on the way out.
            raise
        except Exception as exc:  # noqa: BLE001 - the stream must always close
            yield {"type": "error", "message": f"The runner failed: {exc}"}
            yield {"type": "done", "ok": False}
        else:
            db.record_run(user, exercise_id, mode, backend, summary)
            if mode == "check" and summary.get("ok"):
                for lesson in curriculum.lesson_list():
                    if lesson.get("exercise") == exercise_id:
                        db.set_progress(user, lesson["slug"], "solved")

    return Run(backend=backend, events=events())
