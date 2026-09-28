"""Runs a submission on a RunPod serverless worker with a GPU attached.

The worker image is built from `gpu/`. Its handler is a generator that yields the
same event dictionaries the local harness writes, so the frontend cannot tell the
two backends apart except by the badge.

A serverless worker that has scaled to zero takes a while to come back, so this
backend reports queue position and worker state as `status` events while the job
waits. The lessons that use it are the ones where a real GPU is the point.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

import httpx

from ..config import Settings
from .base import Event, RunRequest, RunnerError

BASE = "https://api.runpod.ai/v2"

# How often to ask for new output while a job is running, and while it waits.
POLL_RUNNING = 1.0
POLL_QUEUED = 2.0


async def run_runpod(
    request: RunRequest, settings: Settings
) -> AsyncIterator[Event]:
    if not settings.gpu_available:
        raise RunnerError("No RunPod endpoint is configured")

    endpoint = f"{BASE}/{settings.runpod_endpoint_id}"
    headers = {
        "Authorization": f"Bearer {settings.runpod_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "exercise_id": request.exercise_id,
            "mode": request.mode,
            "code": request.code,
            "timeout": min(int(request.timeout), settings.runpod_timeout),
        }
    }

    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=60.0) as client:
        yield {"type": "status", "text": f"Queuing on RunPod ({settings.runpod_gpu})"}
        try:
            response = await client.post(
                f"{endpoint}/run", headers=headers, json=payload
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            yield {
                "type": "error",
                "message": f"RunPod would not accept the job: {exc}",
            }
            yield {"type": "done", "ok": False, "seconds": time.perf_counter() - started}
            return

        job_id = response.json().get("id")
        if not job_id:
            yield {"type": "error", "message": "RunPod returned no job id"}
            yield {"type": "done", "ok": False, "seconds": time.perf_counter() - started}
            return

        saw_done = False
        announced = ""
        deadline = time.monotonic() + settings.runpod_timeout + 120

        try:
            while time.monotonic() < deadline:
                stream = await client.get(f"{endpoint}/stream/{job_id}", headers=headers)
                stream.raise_for_status()
                body = stream.json()
                status = body.get("status", "IN_QUEUE")

                if status != announced and status in ("IN_QUEUE", "IN_PROGRESS"):
                    announced = status
                    yield {
                        "type": "status",
                        "text": (
                            "Waiting for a worker to start"
                            if status == "IN_QUEUE"
                            else "Running on the GPU"
                        ),
                    }

                for event in _events(body.get("stream", [])):
                    if event.get("type") == "done":
                        saw_done = True
                    yield event

                if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                    if status != "COMPLETED" and not saw_done:
                        yield {
                            "type": "error",
                            "message": f"The RunPod job ended as {status}.",
                            "detail": str(body.get("error"))[:2000] or None,
                        }
                    break

                await asyncio.sleep(
                    POLL_QUEUED if status == "IN_QUEUE" else POLL_RUNNING
                )
            else:
                await _cancel(client, endpoint, headers, job_id)
                yield {
                    "type": "error",
                    "message": "The GPU job passed its time limit and was cancelled.",
                }
        except httpx.HTTPError as exc:
            yield {"type": "error", "message": f"Lost contact with RunPod: {exc}"}

        if not saw_done:
            yield {
                "type": "done",
                "ok": False,
                "seconds": time.perf_counter() - started,
            }


def _events(chunks: list[Any]) -> list[Event]:
    """Flatten RunPod's stream envelopes into the events the handler yielded."""
    out: list[Event] = []
    for chunk in chunks:
        output = chunk.get("output") if isinstance(chunk, dict) else chunk
        if isinstance(output, dict):
            out.append(output)
        elif isinstance(output, list):
            out.extend(item for item in output if isinstance(item, dict))
    return out


async def _cancel(
    client: httpx.AsyncClient, endpoint: str, headers: dict[str, str], job_id: str
) -> None:
    try:
        await client.post(f"{endpoint}/cancel/{job_id}", headers=headers)
    except httpx.HTTPError:
        pass


async def probe(settings: Settings) -> dict[str, Any]:
    """Ask the endpoint how many workers it has, for the compute badge."""
    if not settings.gpu_available:
        return {"configured": False}
    endpoint = f"{BASE}/{settings.runpod_endpoint_id}/health"
    headers = {"Authorization": f"Bearer {settings.runpod_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(endpoint, headers=headers)
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPError as exc:
        return {"configured": True, "reachable": False, "detail": str(exc)}
    return {
        "configured": True,
        "reachable": True,
        "gpu": settings.runpod_gpu,
        "workers": body.get("workers", {}),
        "jobs": body.get("jobs", {}),
    }
