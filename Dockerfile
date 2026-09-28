# Stage 1: build the single-page app.
FROM node:22-alpine AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build

# Stage 2: the Python service, which also serves the built assets and runs the
# readers' submissions.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install -r backend/requirements.txt

# The exercises import torch on the CPU. The default wheel carries every CUDA
# library, which this container has no use for, so pull the CPU build instead.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1

# Everything above is identical to the ml course's image, so the two share
# those layers on the VM and this image costs only what follows. Packages the
# speech course adds go here, never into backend/requirements.txt, which would
# change the layers above. The soundfile wheel bundles libsndfile.
RUN pip install soundfile==0.14.0

COPY backend/ ./backend/
COPY content/ ./content/
COPY docs/ ./docs/
COPY --from=frontend /build/dist ./frontend/dist

# The commit this image was built from, reported by /api/health. Declared after
# the COPY steps so changing it does not invalidate the layers above.
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

RUN useradd --create-home --uid 10001 app
RUN mkdir -p /data
RUN chown -R app:app /data /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
