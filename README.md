# Speech and audio

A hands-on course in speech and audio processing that you learn by building it.
Each lesson explains an idea in plain words, works a small example by hand, and
then walks you through implementing it, in an editor beside the text, against
checks that verify the results rather than the shape of your code.

The reading pane is on the left and the editor is on the right, the way a
programming-problem site arranges them. **Run** executes your file and streams
its output, including any clip it plays and any spectrogram or plot it draws.
**Check** runs the exercise's checks and reports them one at a time as they
finish. Interactive demos sit inside the lessons, so you can hear aliasing or
stretch a spectrogram's window before you write the code that does it.

## Who it is for

You can write Python and have used numpy a little. The lessons define every term
and symbol when it first appears, and show each formula next to the numpy code
that computes it. Each lesson ends with a step-by-step guide to its exercise.

## What is in here

The repository is laid out as follows.

| Path | Contents |
|---|---|
| `content/lessons/` | The lessons, in Markdown with YAML front matter. |
| `content/exercises/` | One directory per exercise: a starter file, checks, and a worked solution. |
| `backend/` | FastAPI: auth, content, progress, drafts, and the run stream. |
| `backend/app/runner/` | The harness, the `speechcheck` assertion library, the `speechlab` output helpers, and the two execution backends. |
| `frontend/` | React and Vite: the reader, the editor, the run console, and the demos. |
| `frontend/src/demos/` | The interactive demos, their plotting kit, and a small signal-processing library. |
| `frontend/src/book/` | The print layout for the PDF book, and `frontend/scripts/build-book.mjs`, which prints it. |
| `gpu/` | The RunPod serverless worker, the assets it bakes in, and the script that fetches them. |
| `docs/` | The content contract, the demo guide, and the curriculum. |
| `scripts/` | Content validation, the book's content dump, the deploy script, and the RunPod setup. |

## The curriculum

<!-- TO BE REWRITTEN BY ORCHESTRATOR -->

The curriculum is still being written. `content/lessons/99-smoke.md` is a
temporary lesson that exercises every part of the app; it goes once the real
lessons arrive.

[The full list, with what each exercise delivers](docs/CURRICULUM.md).

## Run it locally

You need Python 3.12 or later and Node 20 or later.

1. Copy the environment file and set a session secret:

   ```bash
   cp .env.example .env
   python3 -c "import secrets; print(secrets.token_urlsafe(48))"
   ```

   Put that value in `SESSION_SECRET`, and set `DATA_DIR` to a local
   directory such as `data`.

2. Install and start the backend:

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r backend/requirements.txt
   .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
   .venv/bin/pip install soundfile==0.14.0 pyyaml
   .venv/bin/uvicorn backend.app.main:app --reload --port 8000
   ```

3. In another terminal, start the frontend:

   ```bash
   cd frontend && npm install && npm run dev
   ```

Open `http://localhost:5173`. The Vite dev server proxies `/api` to the backend
on port 8000. To use a backend on another port, set `API_TARGET`, for example
`API_TARGET=http://127.0.0.1:8016 npm run dev`.
Sign in with one of the names in `APP_USERS` and the password in `APP_PASSWORD`.

`.claude/launch.json` holds the same two servers on ports 8016 and 5176, for
the preview tools.

## Write a lesson

`docs/CONTENT_CONTRACT.md` is the binding specification: the file layout, the
front matter, the required sections, the prose style, the `speechcheck` API
the checks are written against, the `speechlab` helpers that play and plot,
and the rules for demos and assets. `docs/DEMOS.md` explains how to build a
demo.

Validate what you write:

```bash
.venv/bin/python scripts/check_content.py 99-smoke
```

```bash
node scripts/check_math.mjs content/lessons/99-smoke.md
```

That checks the front matter, the required sections, the word count, and every
demo block against the demo registry; runs every check against `solution.py`,
which must pass all of them, and against `starter.py`, which must fail at least
one; and presses **Run** on both, where the solution must finish and the starter
must stop at a `NotImplementedError`. Omit the slug to validate everything. The
same script runs in CI on every push.

`check_math.mjs` compiles every math segment with the frontend's own KaTeX, so
math that fails only at render time fails here instead. Run `npm install` in
`frontend/` first.

To test the demos' signal processing, run `npm test` in `frontend/`.

## How a run works

Pressing **Run** or **Check** posts your code to `/api/run`, which streams
server-sent events back until the run finishes. The event types are `begin`,
`status`, `plan`, `stdout`, `stderr`, `audio`, `figure`, `check`, `error`, and
`done`.

The submission executes in a subprocess in its own process group, started with
`python -I`, under a CPU-time limit and a wall clock, with the container's
`mem_limit` bounding memory. When the wall clock runs out, the whole process
group is stopped. The harness wraps the reader's own prints as events, so a print
that happens to look like JSON can't be confused with a harness event.

`speechlab`, beside the harness, writes the `audio` and `figure` events to the
same channel. A clip is 16-bit WAV, cut at 30 seconds or about 2 MB; a heatmap
is pooled to at most 256 by 512 cells and sent as 8-bit levels; a line plot
keeps at most 2000 points a series. A run sends at most 32 clips and figures,
8 MB in all. The backend passes these events through unchanged, and drops any
over its own ceiling of 4 MB an event or 12 MB a run.

This is a resource boundary, not a security boundary. Anyone who can sign in can
run code in the container. The app is for two named people on a private host;
don't open it wider without putting the runner in a sandbox that isolates the
filesystem and the network.

## Authentication

Two people share one password, set in `APP_PASSWORD`, and sign in under one of
the names in `APP_USERS`. The username isn't a secret: it scopes progress,
drafts, and run history so that two readers keep separate state. A signed,
HTTP-only cookie carries the session. Set `COOKIE_SECURE=true` once the app is
served over HTTPS.

## The GPU backend

A lesson marked `gpu: true` runs on a RunPod serverless worker when one is
configured. Its checks run on the CPU like every other exercise, and the GPU is
there for the larger run in the starter file's `__main__` block.

`backend/app/runner/runpod.py` sends a submission to the endpoint and streams
the worker's events back. The worker in `gpu/` carries the same harness, the
same `speechcheck` and `speechlab`, and the same exercise directories, so a
lesson behaves identically on either backend. Its image adds `transformers`,
`huggingface_hub`, `safetensors`, and `soundfile` to PyTorch with CUDA.

With `RUNPOD_ENDPOINT_ID` unset, the GPU lessons fall back to the local CPU
runner and everything still works. To turn the GPU on, complete the following
steps:

1. Let the publish workflow build and push the worker. It runs on every push
   that touches `gpu/`, `content/exercises/`, or the harness, and tags
   `ghcr.io/weights-and-wires/speech-gpu-worker:latest`.

2. Give RunPod read access to that image. A package published from Actions
   starts private, and RunPod pulls anonymously. Make the package public under
   the organisation's **Packages** settings. The alternative is a container
   registry credential in the RunPod console holding a GitHub token with
   `read:packages`, which hands that token to RunPod.

3. Create the endpoint and read back its ID:

   ```bash
   RUNPOD_API_KEY=... python3 scripts/runpod_endpoint.py
   ```

   The endpoint prefers an A100 80GB (`AMPERE_80`) and falls back to an L40 or
   L40S (`ADA_48_PRO`) or an RTX 4090 (`ADA_24`). It scales from zero to at
   most two workers, and a worker goes idle after 10 seconds. The script
   refuses to run while the image is private, and says so. Pass
   `--skip-pull-check` when you took the registry-credential route instead.

4. Put `RUNPOD_ENDPOINT_ID` in the deployed `.env` and restart:

   ```bash
   ssh ifkash@vm.ifkash.dev "cd ~/docs/speech && docker compose up -d"
   ```

`/api/compute` reports what the endpoint has: whether it is reachable, how many
workers are warm, and how many jobs are queued.

### Assets

A lab that needs a model or a clip loads it with `speechlab.asset_path`, never
from the network at run time. `gpu/assets.json` lists what to bake into the
worker: Hugging Face files at a pinned revision, or HTTPS downloads with a
SHA-256. `gpu/fetch_assets.py` downloads them into `/assets` when the image is
built, and the worker sets `SPEECH_ASSETS=/assets`.

The CPU runner sets `SPEECH_ASSETS` to the repository's `assets/` directory,
which Git ignores and which is usually empty, so a lab checks
`speechlab.has_asset` and falls back to something small. To give your own
machine the same files, run the following command:

```bash
.venv/bin/pip install huggingface_hub
.venv/bin/python gpu/fetch_assets.py gpu/assets.json assets
```

## Deploy

The VM holds the tree at `~/docs/speech` and runs it behind the shared Caddy on
the external `edge` network. This bundle never publishes 80 or 443 itself.

```bash
./scripts/deploy.sh
```

The repository is private and the organisation disables deploy keys, so the VM
holds no GitHub credential. The script ships the committed tree with
`git archive` over SSH, rebuilds, and waits until `/api/health` reports the
commit it sent. `.env` on the VM is never touched, because it holds the session
secret and the RunPod key. `Caddyfile.example` holds the vhost block for
`speech.ifkash.dev`, including the buffering and timeout settings the run
stream needs.

The VM has little disk and memory to spare, so the image is built to share its
largest layers with the ml course: the `Dockerfile` is identical to that
course's up to and including the CPU `torch` install, and the packages this
course adds, such as `soundfile`, come in a later layer. Keep it that way. The
container is capped at 1.5 CPUs and 1536 MB, and `LOCAL_WORKERS=2` runs two
submissions at once. The script prunes build cache older than three days after
each deploy.

The script also builds the PDF book on your workstation, because printing it
needs Chrome. It copies the PDF to `~/docs/speech/book` on the VM, which the
container mounts read-only. For more information, see
[The PDF book](#the-pdf-book).

To reach the app before its DNS record resolves, tunnel the container's
localhost port:

```bash
ssh -L 8096:127.0.0.1:8096 ifkash@vm.ifkash.dev
```

## The PDF book

The **PDF** button in the header downloads the whole course as one book: every
lesson, every exercise with its starter file and hints, and the solutions in an
appendix. Each demo appears as a still figure, drawn in its starting state, with
a line that names the lesson online where you can use it. The book has a
contents page with page numbers, a bookmark for every lesson and section, and
running heads. Like the lessons, the download needs a signed-in session.

To build the book, run the following command:

```bash
node frontend/scripts/build-book.mjs
```

The build needs Google Chrome or Chromium, found through `CHROME_PATH` or in
the usual install locations. It dumps the lessons with the backend's own loader
(`scripts/dump_book.py`), renders the print layout in `frontend/src/book/` with
the site's Markdown pipeline, and prints it with headless Chrome. It prints
more than once, so that the contents carry real page numbers. The PDF,
`speech-and-audio.pdf`, and its `manifest.json` land in `book/`, which Git
ignores.

The build fails on any formula KaTeX can't render and on anything wider than
the page, because Chrome shrinks every page to fit the widest element. It skips
itself when no lesson or layout file has changed; to rebuild anyway, pass
`--force`. CI builds the book on every push and keeps it as a workflow
artifact for seven days.

The lockfile must come from npm 10, because the image's `npm ci` rejects a
lockfile that npm 11 writes. After you change a dependency, run
`npx npm@10 install` in `frontend/`.

## Licence

MIT. See [LICENSE](LICENSE).
