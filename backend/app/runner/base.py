"""The shape every execution backend shares."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import Settings

Event = dict[str, Any]


class RunnerError(RuntimeError):
    """A failure in the runner itself, as opposed to in the reader's code."""


@dataclass
class RunRequest:
    exercise_id: str
    mode: str  # "run" or "check"
    code: str
    timeout: int
    needs_gpu: bool


def choose_backend(request: RunRequest, settings: Settings) -> str:
    """Pick the backend for one request.

    A lesson that asks for a GPU gets one when RunPod is configured, and falls
    back to the local CPU runner when it is not. Everything else runs locally,
    because a round trip to a cold serverless worker costs more than the work.
    """
    if request.needs_gpu and settings.gpu_available:
        return "runpod"
    return "local"
