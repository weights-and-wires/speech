# Stage 1: build the single-page app.
FROM node:22-alpine AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# Stage 2: print the PDF book.
#
# The book is built in the image so that a deploy is only a pull: the image
# carries the PDF that matches its own chapters, and no workstation has to build
# and copy one. Chromium, fonts, node_modules, and the loader's Python packages
# stay in this stage. The final image receives only the PDF and its manifest.
#
# This stage uses Debian rather than Alpine because Chromium and its fonts are
# better supported there, and because the print layout needs a node_modules
# built for glibc, which the Alpine `frontend` stage can't provide.
#
# Layers run from the least to the most frequently changed input, so a change
# to a backend file other than the loader's inputs reuses everything up to the
# last COPY. Don't pass GIT_SHA into this stage: a per-commit value would
# rebuild the book on every push. The manifest's commit is "unknown" for that
# reason.
FROM node:22-bookworm-slim AS book

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        chromium python3 fonts-liberation fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

ENV CHROME_PATH=/usr/bin/chromium \
    PYTHON=/opt/venv/bin/python

WORKDIR /book

COPY frontend/package.json frontend/package-lock.json* ./frontend/
RUN npm --prefix frontend ci --no-audit --no-fund

# The loader needs only the backend's requirements, not torch.
COPY backend/requirements.txt ./backend/requirements.txt
# uv resolves and installs several times faster than pip.
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv
RUN uv venv --python /usr/bin/python3 /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache -r backend/requirements.txt

# Only what the book is made from, as build-book.mjs hashes it.
COPY backend/app/__init__.py backend/app/config.py backend/app/curriculum.py ./backend/app/
COPY scripts/dump_book.py ./scripts/dump_book.py
COPY content/ ./content/
COPY frontend/book.html frontend/vite.book.config.ts frontend/tsconfig.json ./frontend/
# Only the print layout's import closure, not all of src/, so a change to the
# site's UI doesn't reprint the book. Printing takes most of a deploy's time.
# Keep this list in step with BOOK_SOURCES in build-book.mjs. A file missing
# here fails the Vite build with an unresolved import. src/demos/ is copied
# whole: the book prints every demo, and a new demo is imported through its
# index.ts. Type-only imports such as lib/api.ts are erased by Vite.
COPY frontend/src/book/ ./frontend/src/book/
COPY frontend/src/demos/ ./frontend/src/demos/
COPY frontend/src/ui/*.css ./frontend/src/ui/
COPY frontend/src/index.css ./frontend/src/
COPY frontend/src/components/Markdown.tsx frontend/src/components/Mark.tsx ./frontend/src/components/
COPY frontend/src/components/figures/HeatmapCanvas.tsx frontend/src/components/figures/LineCanvas.tsx ./frontend/src/components/figures/
COPY frontend/src/lib/colormap.ts frontend/src/lib/events.ts frontend/src/lib/lessonOutline.ts frontend/src/lib/parts.ts frontend/src/lib/plotting.ts frontend/src/lib/rehypeLesson.ts frontend/src/lib/useMediaQuery.ts ./frontend/src/lib/
COPY frontend/scripts/build-book.mjs ./frontend/scripts/
RUN node frontend/scripts/build-book.mjs --force --out /out \
    && rm -rf /out/.build

# Stage 3: the Python service, which also serves the built assets and runs the
# readers' submissions.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    UV_SYSTEM_PYTHON=1 \
    DATA_DIR=/data

WORKDIR /app

# uv installs into the system Python, as pip did, several times faster.
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv

COPY backend/requirements.txt ./backend/requirements.txt
RUN uv pip install -r backend/requirements.txt

# The exercises import torch on the CPU. The default wheel carries every CUDA
# library, which this container has no use for, so pull the CPU build instead.
# uv takes the dependencies of a package from the one index it is given here,
# as pip did, and the CPU index carries them.
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1

# Everything above matches the ml course's image when both courses use the same
# uv version and requirements.txt, so the two share those layers on the VM and
# this image costs only what follows. Packages the speech course adds go here,
# never into backend/requirements.txt, which would change the layers above.
# The soundfile wheel bundles libsndfile.
RUN uv pip install soundfile==0.14.0

COPY backend/ ./backend/
COPY content/ ./content/
COPY docs/ ./docs/
# The demo registry's source, which the MCP server reads for demo titles.
COPY frontend/src/demos/index.ts ./frontend/src/demos/index.ts
COPY --from=frontend /build/dist ./frontend/dist

# The commit this image was built from, reported by /api/health. Declared after
# the COPY steps so changing it does not invalidate the layers above.
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

RUN useradd --create-home --uid 10001 app
RUN mkdir -p /data
RUN chown -R app:app /data /app

# After the chown above, with its own --chown, so the PDF is stored once in
# the image and not a second time as a chown copy.
COPY --from=book --chown=app:app /out/speech-and-audio.pdf /out/manifest.json ./book/

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
