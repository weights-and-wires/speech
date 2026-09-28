"""Runs one exercise inside the sandbox and reports as JSON lines on stdout.

The process is started with three arguments: the mode, the directory holding the
exercise's `tests.py`, and the path to the reader's submitted code. In `run` mode
the submitted file is executed as `__main__` and its output is streamed. In
`check` mode the file is imported as a module, `tests.py` is imported beside it,
and every check registered with `@speechcheck.check` runs in order.

Every line on stdout is one JSON object. The reader's own prints are wrapped as
`stdout` events, so a print that happens to look like JSON cannot be confused
with a harness event.

`speechlab`, which sits next to this file, writes `audio` and `figure` events
to the same channel. `connect` hands it `emit` before any reader code runs, so
a clip or a plot lands between the prints around it, in the order it happened.
"""

from __future__ import annotations

import builtins
import importlib.util
import io
import json
import os
import runpy
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType

CHANNEL = sys.stdout


def emit(event: dict) -> None:
    CHANNEL.write(json.dumps(event, default=str) + "\n")
    CHANNEL.flush()


class LineStream(io.TextIOBase):
    """A writable stream that forwards each completed line as an event."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._buffer = ""

    def write(self, text: str) -> int:  # type: ignore[override]
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            emit({"type": self.kind, "text": line})
        # A long line with no newline still has to reach the reader eventually.
        if len(self._buffer) > 8192:
            emit({"type": self.kind, "text": self._buffer})
            self._buffer = ""
        return len(text)

    def flush(self) -> None:  # type: ignore[override]
        if self._buffer:
            emit({"type": self.kind, "text": self._buffer})
            self._buffer = ""

    def isatty(self) -> bool:  # type: ignore[override]
        return False


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def format_exception(exc: BaseException, submission: Path) -> str:
    """A traceback trimmed to the frames inside the reader's own file."""
    frames = traceback.extract_tb(exc.__traceback__)
    own = [f for f in frames if Path(f.filename).name == submission.name]
    lines = []
    for frame in own or frames[-3:]:
        where = f"line {frame.lineno}"
        if frame.name != "<module>":
            where += f", in {frame.name}"
        lines.append(f"  {where}")
        if frame.line:
            lines.append(f"    {frame.line}")
    head = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    return "\n".join(lines + [head])


def run_mode(submission: Path) -> int:
    out, err = LineStream("stdout"), LineStream("stderr")
    started = time.perf_counter()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            runpy.run_path(str(submission), run_name="__main__")
        out.flush()
        err.flush()
    except NotImplementedError as exc:
        out.flush()
        err.flush()
        emit(
            {
                "type": "error",
                "message": "You have not filled this in yet.",
                "detail": format_exception(exc, submission),
            }
        )
        emit({"type": "done", "ok": False, "seconds": time.perf_counter() - started})
        return 1
    except BaseException as exc:  # noqa: BLE001 - the reader's code can raise anything
        out.flush()
        err.flush()
        emit(
            {
                "type": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "detail": format_exception(exc, submission),
            }
        )
        emit({"type": "done", "ok": False, "seconds": time.perf_counter() - started})
        return 1
    emit({"type": "done", "ok": True, "seconds": time.perf_counter() - started})
    return 0


def check_mode(submission: Path, exercise_dir: Path) -> int:
    import speechcheck
    import speechlab

    started = time.perf_counter()
    out, err = LineStream("stdout"), LineStream("stderr")

    try:
        with redirect_stdout(out), redirect_stderr(err):
            sub = load_module(submission, "submission")
        out.flush()
        err.flush()
    except BaseException as exc:  # noqa: BLE001
        out.flush()
        err.flush()
        emit(
            {
                "type": "error",
                "message": f"Your file did not import: {type(exc).__name__}: {exc}",
                "detail": format_exception(exc, submission),
            }
        )
        emit({"type": "done", "ok": False, "seconds": time.perf_counter() - started})
        return 1

    try:
        load_module(exercise_dir / "tests.py", "exercise_tests")
    except BaseException as exc:  # noqa: BLE001
        emit({"type": "error", "message": f"The checks failed to load: {exc}"})
        emit({"type": "done", "ok": False, "seconds": time.perf_counter() - started})
        return 1

    checks = list(speechcheck.REGISTRY)
    emit({"type": "plan", "total": len(checks)})

    passed = 0
    for index, item in enumerate(checks):
        speechcheck.drain_timings()
        # A figure a check draws says which check drew it.
        speechlab._CONTEXT["check"] = index
        began = time.perf_counter()
        captured = io.StringIO()
        status, message = "passed", None
        try:
            with redirect_stdout(captured), redirect_stderr(captured):
                item.func(sub)
        except speechcheck.CheckFailure as exc:
            status, message = "failed", str(exc)
        except NotImplementedError as exc:
            status = "failed"
            message = "not implemented yet" + (f": {exc}" if str(exc) else "")
        except BaseException as exc:  # noqa: BLE001
            status = "failed"
            message = (
                f"{type(exc).__name__}: {exc}\n"
                f"{format_exception(exc, submission)}"
            )
        else:
            passed += 1
        event = {
            "type": "check",
            "index": index,
            "description": item.description,
            "status": status,
            "seconds": time.perf_counter() - began,
        }
        if message:
            event["message"] = message
        printed = captured.getvalue().strip()
        if printed:
            event["output"] = printed[:4000]
        speechlab._CONTEXT.pop("check", None)
        timings = speechcheck.drain_timings()
        if timings:
            event["timings"] = timings
        emit(event)

    ok = passed == len(checks) and checks
    emit(
        {
            "type": "done",
            "ok": bool(ok),
            "passed": passed,
            "total": len(checks),
            "seconds": time.perf_counter() - started,
        }
    )
    return 0 if ok else 1


def main() -> int:
    if len(sys.argv) != 4:
        emit({"type": "error", "message": "harness: mode, exercise dir, submission"})
        return 2
    mode, exercise_dir, submission = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])

    # The reader's code lives beside the checks, and `speechcheck` and
    # `speechlab` sit next to this file, so all of them resolve without the
    # reader importing anything unusual.
    sys.path.insert(0, str(submission.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.chdir(submission.parent)

    import speechlab

    speechlab.connect(emit)

    # `input()` has no terminal here, and a reader who calls it would otherwise
    # see the run stop at the timeout with no explanation.
    def no_input(prompt: str = "") -> str:
        raise RuntimeError("This runner has no standard input, so input() cannot work.")

    builtins.input = no_input  # type: ignore[assignment]

    if mode == "run":
        return run_mode(submission)
    if mode == "check":
        return check_mode(submission, exercise_dir)
    emit({"type": "error", "message": f"harness: unknown mode {mode!r}"})
    return 2


if __name__ == "__main__":
    sys.exit(main())
