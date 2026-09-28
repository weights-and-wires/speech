"""Runs a submission in a subprocess inside this container.

The submission is copied into a scratch directory beside the exercise's
`tests.py`, and the harness is started on it as a separate process in its own
process group, under CPU-time, address-space, file-size, and process-count
limits. When the wall clock runs out the whole group is killed.

This is a resource boundary, not a security boundary. The course is for two
people who are signed in, and the container has no secrets in its environment
beyond its own session key. Do not expose this runner to the public.
"""

from __future__ import annotations

import asyncio
import json
import os
import resource
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator

from ..config import Settings
from .base import Event, RunRequest

HARNESS = Path(__file__).resolve().parent / "harness.py"

# The longest line the harness can write. An `audio` event carries a whole WAV
# file as base64 on one line, up to about 2.7 MB, and asyncio's default of
# 64 KiB would end the run with a LimitOverrunError.
LINE_LIMIT = 16 << 20

# One semaphore for the whole process, so several readers pressing Run at once
# cannot start more interpreters than the container has cores for.
_gate: asyncio.Semaphore | None = None


def _semaphore(settings: Settings) -> asyncio.Semaphore:
    global _gate
    if _gate is None:
        _gate = asyncio.Semaphore(settings.local_workers)
    return _gate


def _limits(memory_mb: int, cpu_seconds: int):
    """Return a `preexec_fn` that caps what the child can consume.

    Memory is deliberately not capped here. `RLIMIT_AS` bounds virtual address
    space, which numpy and torch over-reserve by design, so a limit low enough
    to be useful stops torch from importing at all. The container's own
    `mem_limit` in `docker-compose.yml` bounds resident memory instead, which is
    the quantity that matters, and the kernel enforces it for the whole cgroup.

    Setting `memory_mb` above zero turns `RLIMIT_AS` back on, for a host with no
    cgroup to fall back on.
    """

    def apply() -> None:
        os.setsid()
        # Each limit is applied on its own: a platform that rejects one still
        # gets the others, and a failure here would otherwise surface as an
        # opaque "Exception occurred in preexec_fn".
        wanted: list[tuple[int, tuple[int, int]]] = [
            (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5)),
            # 64 MiB of output is far more than any exercise writes.
            (resource.RLIMIT_FSIZE, (64 << 20, 64 << 20)),
            (resource.RLIMIT_CORE, (0, 0)),
        ]
        if memory_mb > 0:
            memory = memory_mb * 1024 * 1024
            wanted.append((resource.RLIMIT_AS, (memory, memory)))
        if hasattr(resource, "RLIMIT_NPROC"):
            wanted.append((resource.RLIMIT_NPROC, (256, 256)))

        for which, values in wanted:
            try:
                resource.setrlimit(which, values)
            except (ValueError, OSError):
                continue

    return apply


async def run_local(
    request: RunRequest, exercise_dir: Path, settings: Settings
) -> AsyncIterator[Event]:
    timeout = min(int(request.timeout), settings.local_timeout_ceiling)

    async with _semaphore(settings):
        yield {"type": "status", "text": "Starting the local runner"}

        with tempfile.TemporaryDirectory(prefix="speech-run-") as scratch:
            work = Path(scratch)
            tests = exercise_dir / "tests.py"
            if request.mode == "check":
                if not tests.is_file():
                    yield {
                        "type": "error",
                        "message": f"{request.exercise_id} has no checks to run",
                    }
                    yield {"type": "done", "ok": False, "seconds": 0.0}
                    return
                shutil.copy(tests, work / "tests.py")

            submission = work / "submission.py"
            submission.write_text(request.code)

            environment = {
                "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                "HOME": scratch,
                "TMPDIR": scratch,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "MPLBACKEND": "Agg",
                # Keep BLAS from claiming every core for one reader's run.
                "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                # Where `speechlab.asset_path` looks. See `Settings.assets_dir`.
                "SPEECH_ASSETS": str(settings.assets_dir),
            }

            started = time.perf_counter()
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                str(HARNESS),
                request.mode,
                str(work),
                str(submission),
                cwd=str(work),
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=_limits(settings.local_memory_mb, timeout),
                limit=LINE_LIMIT,
            )

            saw_done = False
            timed_out = False
            try:
                async for event in _stream(process, timeout):
                    if event.get("type") == "done":
                        saw_done = True
                    yield event
            except asyncio.TimeoutError:
                timed_out = True
            finally:
                await _terminate(process)

            if timed_out:
                yield {
                    "type": "error",
                    "message": (
                        f"The run passed its {timeout} second limit and was stopped. "
                        "An infinite loop, or a problem sized larger than the "
                        "exercise asks for, is the usual cause."
                    ),
                }
                yield {
                    "type": "done",
                    "ok": False,
                    "seconds": time.perf_counter() - started,
                    "timeout": True,
                }
                return

            if not saw_done:
                stderr = (await process.stderr.read()).decode("utf-8", "replace")
                message = "The runner stopped before finishing."
                if process.returncode == -signal.SIGKILL:
                    limit = (
                        f"{settings.local_memory_mb} MiB"
                        if settings.local_memory_mb > 0
                        else "the container's memory limit"
                    )
                    message = (
                        "The run was killed, which almost always means it asked "
                        f"for more memory than {limit} allows."
                    )
                yield {
                    "type": "error",
                    "message": message,
                    "detail": stderr[-4000:] or None,
                }
                yield {
                    "type": "done",
                    "ok": False,
                    "seconds": time.perf_counter() - started,
                }


async def _stream(
    process: asyncio.subprocess.Process, timeout: int
) -> AsyncIterator[Event]:
    """Yield one event per line of the harness's stdout, until it closes."""
    assert process.stdout is not None
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError
        try:
            line = await asyncio.wait_for(process.stdout.readline(), remaining)
        except asyncio.TimeoutError:
            raise
        except (ValueError, asyncio.LimitOverrunError):
            # speechlab keeps its events far below this, so only a harness
            # bypassed by hand can get here.
            yield {
                "type": "error",
                "message": f"The run wrote a line longer than {LINE_LIMIT >> 20} MiB and was stopped.",
            }
            return
        if not line:
            return
        text = line.decode("utf-8", "replace").rstrip("\n")
        if not text:
            continue
        try:
            yield json.loads(text)
        except json.JSONDecodeError:
            # The harness wraps the reader's own prints, so a line that is not
            # JSON came from the interpreter itself and is worth showing.
            yield {"type": "stderr", "text": text}


async def _terminate(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        await asyncio.wait_for(process.wait(), 5)
    except asyncio.TimeoutError:
        pass
