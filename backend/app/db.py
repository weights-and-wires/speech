"""SQLite storage for progress, drafts, and run history.

Everything here is keyed by username, so two readers working through the course
keep separate state. The database lives under `DATA_DIR`, which is a volume in
the deployed container.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS progress (
    username    TEXT NOT NULL,
    slug        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'started',
    updated_at  REAL NOT NULL,
    PRIMARY KEY (username, slug)
);

CREATE TABLE IF NOT EXISTS drafts (
    username    TEXT NOT NULL,
    exercise_id TEXT NOT NULL,
    code        TEXT NOT NULL,
    updated_at  REAL NOT NULL,
    PRIMARY KEY (username, exercise_id)
);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT NOT NULL,
    exercise_id TEXT NOT NULL,
    mode        TEXT NOT NULL,
    backend     TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    passed      INTEGER,
    total       INTEGER,
    seconds     REAL,
    started_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS runs_by_user
    ON runs (username, started_at DESC);
"""


def _connect() -> sqlite3.Connection:
    path: Path = get_settings().db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


@contextmanager
def session() -> Iterator[sqlite3.Connection]:
    connection = _connect()
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init() -> None:
    with session() as connection:
        connection.executescript(SCHEMA)


# Progress ---------------------------------------------------------------


def set_progress(username: str, slug: str, status: str) -> None:
    """Record a lesson as `started` or `solved`, never downgrading a solve."""
    with session() as connection:
        existing = connection.execute(
            "SELECT status FROM progress WHERE username = ? AND slug = ?",
            (username, slug),
        ).fetchone()
        if existing and existing["status"] == "solved" and status != "solved":
            return
        connection.execute(
            "INSERT INTO progress (username, slug, status, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT (username, slug) DO UPDATE SET "
            "status = excluded.status, updated_at = excluded.updated_at",
            (username, slug, status, time.time()),
        )


def get_progress(username: str) -> dict[str, dict[str, Any]]:
    with session() as connection:
        rows = connection.execute(
            "SELECT slug, status, updated_at FROM progress WHERE username = ?",
            (username,),
        ).fetchall()
    return {
        row["slug"]: {"status": row["status"], "updated_at": row["updated_at"]}
        for row in rows
    }


# Drafts -----------------------------------------------------------------


def save_draft(username: str, exercise_id: str, code: str) -> None:
    with session() as connection:
        connection.execute(
            "INSERT INTO drafts (username, exercise_id, code, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT (username, exercise_id) DO UPDATE SET "
            "code = excluded.code, updated_at = excluded.updated_at",
            (username, exercise_id, code, time.time()),
        )


def get_draft(username: str, exercise_id: str) -> str | None:
    with session() as connection:
        row = connection.execute(
            "SELECT code FROM drafts WHERE username = ? AND exercise_id = ?",
            (username, exercise_id),
        ).fetchone()
    return row["code"] if row else None


def delete_draft(username: str, exercise_id: str) -> None:
    with session() as connection:
        connection.execute(
            "DELETE FROM drafts WHERE username = ? AND exercise_id = ?",
            (username, exercise_id),
        )


# Runs -------------------------------------------------------------------


def record_run(
    username: str,
    exercise_id: str,
    mode: str,
    backend: str,
    summary: dict[str, Any],
) -> None:
    with session() as connection:
        connection.execute(
            "INSERT INTO runs "
            "(username, exercise_id, mode, backend, ok, passed, total, seconds, started_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                username,
                exercise_id,
                mode,
                backend,
                1 if summary.get("ok") else 0,
                summary.get("passed"),
                summary.get("total"),
                summary.get("seconds"),
                time.time(),
            ),
        )


def recent_runs(username: str, limit: int = 25) -> list[dict[str, Any]]:
    with session() as connection:
        rows = connection.execute(
            "SELECT exercise_id, mode, backend, ok, passed, total, seconds, started_at "
            "FROM runs WHERE username = ? ORDER BY started_at DESC LIMIT ?",
            (username, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def stats(username: str) -> dict[str, Any]:
    with session() as connection:
        solved = connection.execute(
            "SELECT COUNT(*) AS n FROM progress WHERE username = ? AND status = 'solved'",
            (username,),
        ).fetchone()["n"]
        total_runs = connection.execute(
            "SELECT COUNT(*) AS n FROM runs WHERE username = ?", (username,)
        ).fetchone()["n"]
        seconds = connection.execute(
            "SELECT COALESCE(SUM(seconds), 0) AS s FROM runs WHERE username = ?",
            (username,),
        ).fetchone()["s"]
    return {"solved": solved, "runs": total_runs, "compute_seconds": seconds}


def export_all(username: str) -> str:
    """Everything stored for one reader, as JSON. Used by the account page."""
    with session() as connection:
        payload = {
            table: [
                dict(row)
                for row in connection.execute(
                    f"SELECT * FROM {table} WHERE username = ?", (username,)
                ).fetchall()
            ]
            for table in ("progress", "drafts", "runs")
        }
    return json.dumps(payload, indent=2)
