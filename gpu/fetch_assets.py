#!/usr/bin/env python3
"""Downloads the assets in `gpu/assets.json` into one directory.

The GPU worker's Dockerfile runs this at build time, so every model and clip a
lab needs is inside the image and a job never waits on the network:

    python gpu/fetch_assets.py gpu/assets.json /assets

Run it with a local directory to give the CPU runner the same files, which it
finds through `SPEECH_ASSETS` (the repository's Git-ignored `assets/`):

    python gpu/fetch_assets.py gpu/assets.json assets

Each entry lands in `<out>/<name>/`, and `<out>/manifest.json` records what
was fetched. Two kinds of entry exist:

- `"type": "huggingface"` downloads files from a Hugging Face repository at a
  pinned `revision`, a commit hash. `files` lists the file names or glob
  patterns to take, so a model brings its weights and config and not every
  checkpoint in its history. `repo_type` is `model` (the default) or
  `dataset`. This needs `huggingface_hub`.
- `"type": "url"` downloads one file over HTTPS, checks it against `sha256`,
  saves it as `filename`, and unpacks it when `extract` is `zip` or `tar`.

A gated repository needs a token, read from the `HF_TOKEN` environment
variable or from the build secret `/run/secrets/hf_token`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
SHA = re.compile(r"^[0-9a-f]{40}$")
USER_AGENT = "weights-and-wires-speech/1.0"
# The image has to stay small enough to cold-start. Raise it deliberately.
MAX_TOTAL_BYTES = int(float(os.environ.get("SPEECH_ASSETS_MAX_GB", "12")) * (1 << 30))


def fail(message: str) -> None:
    raise SystemExit(f"fetch_assets: {message}")


def token() -> str | None:
    value = os.environ.get("HF_TOKEN", "").strip()
    secret = Path("/run/secrets/hf_token")
    if not value and secret.is_file():
        value = secret.read_text().strip()
    return value or None


def validate(entries: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        where = f"entry {index} ({entry.get('name', '?')})"
        name = entry.get("name")
        if not isinstance(name, str) or not NAME.match(name):
            fail(f"{where}: name must be lowercase letters, digits, '.', '_', or '-'")
        if name in seen or name == "manifest.json":
            fail(f"{where}: the name {name!r} is taken")
        seen.add(name)
        kind = entry.get("type")
        if kind == "huggingface":
            if not isinstance(entry.get("repo"), str) or "/" not in entry["repo"]:
                fail(f"{where}: repo must look like 'owner/name'")
            if not SHA.match(str(entry.get("revision", ""))):
                fail(f"{where}: revision must be a 40-character commit hash, so the build is repeatable")
            if entry.get("repo_type", "model") not in ("model", "dataset"):
                fail(f"{where}: repo_type must be 'model' or 'dataset'")
            files = entry.get("files")
            if not isinstance(files, list) or not files or not all(isinstance(f, str) for f in files):
                fail(f"{where}: files must be a non-empty list of names or glob patterns")
        elif kind == "url":
            if not str(entry.get("url", "")).startswith("https://"):
                fail(f"{where}: url must be https")
            if not re.match(r"^[0-9a-f]{64}$", str(entry.get("sha256", ""))):
                fail(f"{where}: sha256 must be the file's 64-character hex digest")
            if not NAME.match(str(entry.get("filename", ""))):
                fail(f"{where}: filename must be a plain file name")
            if entry.get("extract") not in (None, "zip", "tar"):
                fail(f"{where}: extract must be null, 'zip', or 'tar'")
        else:
            fail(f"{where}: type must be 'huggingface' or 'url'")


def fetch_huggingface(entry: dict[str, Any], target: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        fail("huggingface_hub is not installed; pip install huggingface_hub")
    snapshot_download(
        repo_id=entry["repo"],
        repo_type=entry.get("repo_type", "model"),
        revision=entry["revision"],
        allow_patterns=entry["files"],
        local_dir=str(target),
        token=token(),
    )
    # The download cache under the target is only bookkeeping.
    shutil.rmtree(target / ".cache", ignore_errors=True)
    if not any(p.is_file() for p in target.rglob("*")):
        fail(f"{entry['name']}: no file in {entry['repo']} matched {entry['files']}")


def fetch_url(entry: dict[str, Any], target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    path = target / entry["filename"]
    digest = hashlib.sha256()
    request = urllib.request.Request(entry["url"], headers={"User-Agent": USER_AGENT})
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as out:
                while chunk := response.read(1 << 20):
                    digest.update(chunk)
                    out.write(chunk)
            break
        except OSError as exc:
            if attempt == 3:
                fail(f"{entry['name']}: could not download {entry['url']}: {exc}")
            digest = hashlib.sha256()
            time.sleep(3 * attempt)
    if digest.hexdigest() != entry["sha256"]:
        fail(f"{entry['name']}: sha256 is {digest.hexdigest()}, the list says {entry['sha256']}")
    if entry.get("extract") == "zip":
        with zipfile.ZipFile(path) as archive:
            archive.extractall(target)
        path.unlink()
    elif entry.get("extract") == "tar":
        with tarfile.open(path) as archive:
            archive.extractall(target, filter="data")
        path.unlink()


def size_of(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: fetch_assets.py ASSETS.json OUTPUT_DIR", file=sys.stderr)
        return 2
    listing, out = Path(sys.argv[1]), Path(sys.argv[2])
    entries = json.loads(listing.read_text()).get("assets", [])
    if not isinstance(entries, list):
        fail("'assets' must be a list")
    validate(entries)
    out.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    total = 0
    for entry in entries:
        target = out / entry["name"]
        if target.exists():
            shutil.rmtree(target)
        started = time.perf_counter()
        if entry["type"] == "huggingface":
            fetch_huggingface(entry, target)
            source = f"{entry['repo']}@{entry['revision'][:12]}"
        else:
            fetch_url(entry, target)
            source = entry["url"]
        size = size_of(target)
        total += size
        if total > MAX_TOTAL_BYTES:
            fail(f"the assets pass {MAX_TOTAL_BYTES / (1 << 30):.0f} GiB; set SPEECH_ASSETS_MAX_GB to allow more")
        manifest.append({"name": entry["name"], "source": source, "bytes": size})
        print(f"  {entry['name']}: {size / 1e6:.1f} MB from {source} in {time.perf_counter() - started:.0f} s")

    (out / "manifest.json").write_text(json.dumps({"assets": manifest, "bytes": total}, indent=2) + "\n")
    print(f"Fetched {len(manifest)} asset(s), {total / 1e6:.1f} MB, into {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
