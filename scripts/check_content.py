#!/usr/bin/env python3
"""Validates lessons and exercises against docs/CONTENT_CONTRACT.md.

Run it over everything, or over one lesson while you write it:

    python3 scripts/check_content.py
    python3 scripts/check_content.py 005-logistic-regression

CONTENT_DIR points the checker at another tree, such as a draft of a rewrite.
Each lesson needs at least one ```figure block: one `<svg viewBox=...>` that
colours itself with the fill-* and stroke-* classes, then a caption. Every
exercise file must compile, so a starter with a syntax error cannot pass by
failing its checks.

For each exercise it runs the checks twice: once against `solution.py`, which
must pass every check, and once against `starter.py`, which must fail at least
one. An exercise whose starter already passes is not an exercise. It also
presses Run on both: the solution's `__main__` block must finish, and the
starter's must stop at a `NotImplementedError`.

A chapter body runs 800 to 2600 words, not counting figure and demo blocks.

Every ```demo block must be valid JSON naming a demo in
`frontend/src/demos/index.ts`, with a caption.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# CONTENT_DIR points the checker at another tree, such as a draft of a rewrite.
CONTENT = Path(os.environ.get("CONTENT_DIR") or ROOT / "content").resolve()
LESSONS = CONTENT / "lessons"
EXERCISES = CONTENT / "exercises"
HARNESS = ROOT / "backend" / "app" / "runner" / "harness.py"
REGISTRY = ROOT / "frontend" / "src" / "demos" / "index.ts"
DEMO_BLOCK = re.compile(r"^```demo[ \t]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
MAX_DEMOS = 3

REQUIRED_FRONTMATTER = {
    "title", "slug", "part", "summary", "minutes", "gpu", "objectives", "exercise"
}
REQUIRED_SECTIONS = [
    "## What you already know",
    "## Build it step by step",
    "## The exercise",
]
# Colours in a figure come from the fill-* and stroke-* classes, so it follows
# the theme.
RAW_COLOUR = re.compile(
    r"""(?:fill|stroke|color)\s*[=:]\s*["']?\s*(?:#|rgb|hsl|black|white|red|blue|green)""",
    re.I,
)
FIGURE_BLOCK = re.compile(r"^```figure[ \t]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
CALLOUTS = ("KEY", "TIP", "WARNING", "RECAP")
BANNED = [
    r"\bsimply\b", r"\bjust simply\b", r"\beasily\b", r"\bobviously\b",
    r"\bnote that\b", r"\be\.g\.", r"\bi\.e\.", r"\betc\.", r"\bcurrently\b",
    r"\bstate of the art\b", r"\band/or\b",
]


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        raise ValueError("the file does not start with YAML front matter")
    end = text.index("\n---\n", 3)
    raw, body = text[4:end], text[end + 5 :]
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "This checker needs PyYAML, because the front-matter faults it has to "
            "catch are the ones a simpler parser cannot see. pip install pyyaml"
        ) from None
    return yaml.safe_load(raw) or {}, body

def registry_ids() -> set[str]:
    """The demo ids the app can render, read from the registry's source.

    Each entry in `DEMOS` is written `"id": {` on a line of its own, which is
    specific enough not to catch the type declarations around it.
    """
    if not REGISTRY.is_file():
        return set()
    return set(re.findall(r'^\s*"([a-z0-9-]+)":\s*\{', REGISTRY.read_text(), re.MULTILINE))


def run_harness(mode: str, exercise: Path, submission: Path, timeout: int) -> list[dict]:
    env_path = f"{ROOT}"
    proc = subprocess.run(
        [sys.executable, str(HARNESS), mode, str(exercise), str(submission)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(exercise),
        env={
            **__import__("os").environ,
            "PYTHONPATH": env_path,
            "SPEECH_ASSETS": str(ROOT / "assets"),
        },
    )
    events = []
    for line in proc.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not events and proc.stderr:
        events.append({"type": "error", "message": proc.stderr[-2000:]})
    return events


def check_exercise(exercise_id: str, problems: list[str]) -> None:
    where = EXERCISES / exercise_id
    if not where.is_dir():
        problems.append(f"{exercise_id}: no exercise directory")
        return
    for name in ("meta.json", "starter.py", "tests.py", "solution.py"):
        if not (where / name).is_file():
            problems.append(f"{exercise_id}: missing {name}")
            return

    try:
        meta = json.loads((where / "meta.json").read_text())
    except json.JSONDecodeError as exc:
        problems.append(f"{exercise_id}: meta.json is not valid JSON: {exc}")
        return
    for field in ("id", "title", "brief", "needs_gpu", "timeout"):
        if field not in meta:
            problems.append(f"{exercise_id}: meta.json has no {field!r}")
    if meta.get("id") != exercise_id:
        problems.append(f"{exercise_id}: meta.json id is {meta.get('id')!r}")

    timeout = int(meta.get("timeout", 60))

    # A starter that does not parse still "fails" its checks, which would pass
    # the test below, so every file must compile first.
    for name in ("starter.py", "solution.py", "tests.py"):
        source = (where / name).read_text()
        try:
            compile(source, str(where / name), "exec")
        except SyntaxError as exc:
            problems.append(f"{exercise_id}: {name} does not parse: line {exc.lineno}: {exc.msg}")
            return

    # A temporary copy keeps `solution.py` out of the import path the reader gets.
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copy(where / "tests.py", work / "tests.py")
        shutil.copy(where / "solution.py", work / "submission_solution.py")
        shutil.copy(where / "starter.py", work / "submission_starter.py")

        events = run_harness("check", work, work / "submission_solution.py", timeout + 30)
        done = next((e for e in events if e["type"] == "done"), None)
        checks = [e for e in events if e["type"] == "check"]
        if done is None:
            err = next((e for e in events if e["type"] == "error"), {})
            problems.append(
                f"{exercise_id}: solution.py did not finish. {err.get('message', '')}"
            )
        elif not done.get("ok"):
            failed = [c for c in checks if c["status"] == "failed"]
            detail = "; ".join(
                f"{c['description']} -> {c.get('message', '')[:200]}" for c in failed[:3]
            )
            problems.append(f"{exercise_id}: solution.py fails its own checks: {detail}")
        elif len(checks) < 4:
            problems.append(
                f"{exercise_id}: only {len(checks)} checks, the contract asks for six"
            )
        elif done.get("seconds", 0) > timeout * 0.6:
            problems.append(
                f"{exercise_id}: solution takes {done['seconds']:.0f}s against a "
                f"timeout of {timeout}s; raise the timeout or shrink the checks"
            )

        events = run_harness("check", work, work / "submission_starter.py", timeout + 30)
        done = next((e for e in events if e["type"] == "done"), None)
        if done is not None and done.get("ok"):
            problems.append(f"{exercise_id}: starter.py already passes every check")

        # Run, as the button does. The solution's __main__ block must work,
        # and whatever it plays or draws must be well formed.
        events = run_harness("run", work, work / "submission_solution.py", timeout + 30)
        done = next((e for e in events if e["type"] == "done"), None)
        if done is None or not done.get("ok"):
            err = next((e for e in events if e["type"] == "error"), {})
            problems.append(
                f"{exercise_id}: running solution.py fails: {err.get('message', 'no result')}"
                + (f"\n      {err['detail'][-400:]}" if err.get("detail") else "")
            )
        for event in events:
            problem = media_problem(event)
            if problem:
                problems.append(f"{exercise_id}: solution.py's run {problem}")

        events = run_harness("run", work, work / "submission_starter.py", timeout + 30)
        err = next((e for e in events if e["type"] == "error"), {})
        if "NotImplementedError" not in str(err.get("detail", "")):
            problems.append(
                f"{exercise_id}: running starter.py should stop at a NotImplementedError, "
                f"got {err.get('message') or 'a clean finish'}"
            )


def media_problem(event: dict) -> str | None:
    """What is wrong with an audio or figure event, or None."""
    kind = event.get("type")
    if kind == "audio":
        if not event.get("wav_base64") or not event.get("sample_rate"):
            return "sent an audio event with no WAV data"
        if event.get("truncated"):
            return f"played a clip that was cut short: {event.get('note')}"
    elif kind == "figure":
        if event.get("kind") not in ("heatmap", "line"):
            return f"sent a figure of unknown kind {event.get('kind')!r}"
        if event.get("kind") == "heatmap" and not event.get("data_base64"):
            return "sent a heatmap with no data"
    elif kind == "stderr" and str(event.get("text", "")).startswith("speechlab:"):
        return f"went over the output limits: {event['text']}"
    return None


def check_lesson(path: Path, problems: list[str], seen_exercises: set[str]) -> None:
    text = path.read_text()
    try:
        meta, body = parse_frontmatter(text)
    except ValueError as exc:
        problems.append(f"{path.name}: {exc}")
        return

    missing = REQUIRED_FRONTMATTER - set(meta)
    if missing:
        problems.append(f"{path.name}: front matter has no {sorted(missing)}")
        return
    if meta["slug"] != path.stem:
        problems.append(f"{path.name}: slug {meta['slug']!r} does not match the filename")
    if len(str(meta["summary"])) > 160:
        problems.append(f"{path.name}: summary is longer than 160 characters")
    if not isinstance(meta.get("objectives"), list) or len(meta["objectives"]) < 3:
        problems.append(f"{path.name}: needs at least three objectives")

    for field in ("title", "part", "summary", "slug", "exercise"):
        if not isinstance(meta.get(field), str):
            problems.append(
                f"{path.name}: {field} parsed as "
                f"{type(meta.get(field)).__name__}, not a string. An unquoted "
                "colon does this; wrap the value in double quotes."
            )
    for field in ("objectives", "prereqs"):
        for i, item in enumerate(meta.get(field) or []):
            if not isinstance(item, str):
                problems.append(
                    f"{path.name}: {field}[{i}] parsed as "
                    f"{type(item).__name__}, not a string. An unquoted colon "
                    "makes YAML read the item as a mapping; wrap it in double "
                    "quotes."
                )

    for heading in REQUIRED_SECTIONS:
        if heading not in body:
            problems.append(f"{path.name}: no {heading!r} section")
    if not body.lstrip().startswith("# "):
        problems.append(f"{path.name}: the body must open with a single '# ' heading")

    figures = FIGURE_BLOCK.findall(body)
    if not figures:
        problems.append(f"{path.name}: no ```figure block; every chapter needs one")
    for i, figure in enumerate(figures, 1):
        if "<svg" not in figure or "</svg>" not in figure or "viewBox" not in figure:
            problems.append(f"{path.name}: figure {i} needs one <svg viewBox=...>...</svg>")
        if RAW_COLOUR.search(figure):
            problems.append(
                f"{path.name}: figure {i} sets a raw colour; use the fill-* and "
                "stroke-* classes so it follows the theme"
            )
        caption = figure[figure.rfind("</svg>") + 6 :].strip()
        if not caption:
            problems.append(f"{path.name}: figure {i} has no caption after </svg>")

    # Figure and demo blocks are drawing, not reading, so they do not count.
    counted = DEMO_BLOCK.sub("", FIGURE_BLOCK.sub("", body))
    words = len(counted.split())
    if words < 800:
        problems.append(f"{path.name}: {words} words, the contract asks for 900 or more")
    if words > 2600:
        problems.append(f"{path.name}: {words} words; split the chapter, the contract caps it near 2000")

    prose = re.sub(r"```.*?```", "", body, flags=re.S)
    for pattern in BANNED:
        found = re.search(pattern, prose, flags=re.I)
        if found:
            problems.append(f"{path.name}: avoid {found.group(0)!r}")

    for kind in re.findall(r"^>\s*\[!(\w+)\]", body, flags=re.M):
        if kind not in CALLOUTS:
            problems.append(
                f"{path.name}: unknown callout [!{kind}]; use one of {', '.join(CALLOUTS)}"
            )
    recaps = len(re.findall(r"^>\s*\[!RECAP\]", body, flags=re.M))
    if recaps > 1:
        problems.append(f"{path.name}: {recaps} recap callouts, the contract allows one")

    for line in prose.split("\n"):
        if re.match(r"^\s*(>\s*)?\$\$.+\$\$\s*$", line):
            problems.append(
                f"{path.name}: a one-line $$...$$ renders as inline math; "
                "put $$ on its own lines above and below"
            )
            break

    if body.count("$$") % 2 != 0:
        problems.append(f"{path.name}: an unclosed $$ display-math block")

    demos = DEMO_BLOCK.findall(body)
    if len(demos) > MAX_DEMOS:
        problems.append(f"{path.name}: {len(demos)} demos, the contract allows {MAX_DEMOS}")
    known = registry_ids()
    for block in demos:
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name}: a demo block is not valid JSON: {exc}")
            continue
        if not isinstance(parsed, dict):
            problems.append(f"{path.name}: a demo block must be a JSON object")
            continue
        demo_id = parsed.get("id")
        if demo_id not in known:
            problems.append(f"{path.name}: there is no demo called {demo_id!r} in {REGISTRY.relative_to(ROOT)}")
        if not isinstance(parsed.get("caption"), str) or not parsed["caption"].strip():
            problems.append(f"{path.name}: the demo {demo_id!r} has no caption")
        extra = set(parsed) - {"id", "caption", "props"}
        if extra:
            problems.append(f"{path.name}: the demo {demo_id!r} has unknown keys {sorted(extra)}")
        if "props" in parsed and not isinstance(parsed["props"], dict):
            problems.append(f"{path.name}: the demo {demo_id!r} has props that are not an object")

    exercise_id = meta["exercise"]
    seen_exercises.add(exercise_id)
    check_exercise(exercise_id, problems)


def main() -> int:
    wanted = sys.argv[1:]
    paths = sorted(LESSONS.glob("*.md"))
    if wanted:
        paths = [p for p in paths if p.stem in wanted or any(w in p.stem for w in wanted)]
        if not paths:
            print(f"No lesson matches {wanted}")
            return 2

    problems: list[str] = []
    seen: set[str] = set()
    for path in paths:
        check_lesson(path, problems, seen)

    if not wanted:
        orphans = {d.name for d in EXERCISES.iterdir() if d.is_dir()} - seen
        for orphan in sorted(orphans):
            problems.append(f"{orphan}: an exercise no lesson refers to")

    for problem in problems:
        print(f"FAIL  {problem}")
    print(f"\n{len(paths)} lesson(s) checked, {len(problems)} problem(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
