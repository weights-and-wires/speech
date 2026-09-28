"""Backends that execute a reader's submission and stream the result.

Two exist. `local` runs the code in a subprocess inside this container, under
CPU, memory, and process limits, and handles everything the course does on the
CPU. `runpod` sends the same submission to a RunPod serverless worker with a
GPU attached, for the few lessons that need one.

Both expose the same shape: an async generator of event dictionaries. The event
types are `status`, `plan`, `stdout`, `stderr`, `audio`, `figure`, `check`,
`error`, and `done`.
"""

from .base import Event, RunRequest, RunnerError, choose_backend
from .local import run_local
from .runpod import run_runpod

__all__ = [
    "Event",
    "RunRequest",
    "RunnerError",
    "choose_backend",
    "run_local",
    "run_runpod",
]
