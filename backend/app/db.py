"""SQLite storage for progress, drafts, run history, and MCP access tokens.

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

CREATE TABLE IF NOT EXISTS mcp_tokens (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    username     TEXT NOT NULL,
    name         TEXT NOT NULL,
    token_hash   TEXT NOT NULL UNIQUE,
    prefix       TEXT NOT NULL,
    created_at   REAL NOT NULL,
    last_used_at REAL,
    revoked      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS mcp_tokens_by_user
    ON mcp_tokens (username, revoked);
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


def carry_progress(renames: dict[str, list[str]]) -> int:
    """Move progress on renamed lessons to the lessons that replaced them.

    `renames` maps an old slug to its new slugs. Every reader's status on an
    old slug is copied to each new one, without overwriting a status the
    reader already has there, and the old row is deleted, so a second run
    changes nothing. Returns the number of old rows carried over.
    """
    carried = 0
    with session() as connection:
        for old, new in renames.items():
            rows = connection.execute(
                "SELECT username, status, updated_at FROM progress WHERE slug = ?", (old,)
            ).fetchall()
            for row in rows:
                for slug in new:
                    connection.execute(
                        "INSERT OR IGNORE INTO progress (username, slug, status, updated_at) "
                        "VALUES (?, ?, ?, ?)",
                        (row["username"], slug, row["status"], row["updated_at"]),
                    )
                carried += 1
            if old not in new:
                connection.execute("DELETE FROM progress WHERE slug = ?", (old,))
    return carried


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


def list_drafts(username: str) -> list[dict[str, Any]]:
    """The reader's saved drafts, newest first, without the code itself."""
    with session() as connection:
        rows = connection.execute(
            "SELECT exercise_id, LENGTH(code) AS characters, updated_at "
            "FROM drafts WHERE username = ? ORDER BY updated_at DESC",
            (username,),
        ).fetchall()
    return [dict(row) for row in rows]


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


# MCP tokens -------------------------------------------------------------
#
# Only the SHA-256 of a token is stored. The secret itself is shown once, when
# it is created, by `mcp_tokens.create_token`.

# Writing `last_used_at` on every request would turn each MCP call into a write,
# so it is refreshed at most once per this many seconds.
TOKEN_TOUCH_INTERVAL = 60.0


def insert_token(username: str, name: str, token_hash: str, prefix: str) -> dict[str, Any]:
    created = time.time()
    with session() as connection:
        cursor = connection.execute(
            "INSERT INTO mcp_tokens (username, name, token_hash, prefix, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (username, name, token_hash, prefix, created),
        )
        token_id = cursor.lastrowid
    return {"id": token_id, "name": name, "prefix": prefix, "created_at": created}


def list_tokens(username: str | None = None) -> list[dict[str, Any]]:
    """Active tokens, oldest first: one reader's, or everyone's when `username` is None."""
    query = (
        "SELECT id, username, name, prefix, created_at, last_used_at "
        "FROM mcp_tokens WHERE revoked = 0"
    )
    params: tuple[Any, ...] = ()
    if username is not None:
        query += " AND username = ?"
        params = (username,)
    with session() as connection:
        rows = connection.execute(query + " ORDER BY id", params).fetchall()
    return [dict(row) for row in rows]


def count_tokens(username: str) -> int:
    with session() as connection:
        return connection.execute(
            "SELECT COUNT(*) AS n FROM mcp_tokens WHERE username = ? AND revoked = 0",
            (username,),
        ).fetchone()["n"]


def revoke_token(token_id: int, username: str | None = None) -> bool:
    """Revoke one active token. With `username`, only that reader's own.

    Returns False when no matching active token exists.
    """
    query = "UPDATE mcp_tokens SET revoked = 1 WHERE id = ? AND revoked = 0"
    params: tuple[Any, ...] = (token_id,)
    if username is not None:
        query += " AND username = ?"
        params = (token_id, username)
    with session() as connection:
        return connection.execute(query, params).rowcount > 0


def token_user(token_hash: str) -> str | None:
    """The username an active token belongs to, refreshing its last use."""
    now = time.time()
    with session() as connection:
        row = connection.execute(
            "SELECT id, username, last_used_at FROM mcp_tokens "
            "WHERE token_hash = ? AND revoked = 0",
            (token_hash,),
        ).fetchone()
        if row is None:
            return None
        if row["last_used_at"] is None or now - row["last_used_at"] >= TOKEN_TOUCH_INTERVAL:
            connection.execute(
                "UPDATE mcp_tokens SET last_used_at = ? WHERE id = ?", (now, row["id"])
            )
    return row["username"]
