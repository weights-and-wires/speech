# syntax=docker/dockerfile:1
# The RunPod serverless worker for the exercises marked `gpu: true`.
#
# It carries the same harness, the same `speechcheck` and `speechlab` modules,
# and the same exercise directories the app runs locally, so a lesson behaves
# identically on either backend. It also carries the models and clips listed
# in `gpu/assets.json`, downloaded once here at build time into /assets.
#
# Build from the repository root, so the context includes `content/` and
# `backend/`:
#
#   docker build -f gpu/Dockerfile -t ghcr.io/weights-and-wires/speech-gpu-worker:latest .
#
# The publish workflow does this on every push that touches the worker.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    TOKENIZERS_PARALLELISM=false \
    SPEECH_REPO=/repo \
    SPEECH_ASSETS=/assets

# numpy and scipy match the app's CPU image, so a check computes the same
# numbers on either backend. The soundfile wheel bundles libsndfile.
RUN pip install --no-cache-dir \
        numpy==2.3.3 \
        scipy==1.16.2 \
        runpod==1.7.7 \
        soundfile==0.14.0 \
        huggingface_hub==1.33.0 \
        safetensors==0.8.0 \
        transformers==5.17.0

WORKDIR /repo

# The assets come before the code, so a change to an exercise does not
# download every model again. The layer is rebuilt only when the list or the
# fetcher changes.
# A gated Hugging Face repository needs a token, passed as the build secret
# `hf_token`; without one, public repositories still download.
COPY gpu/assets.json gpu/fetch_assets.py /repo/gpu/
RUN --mount=type=secret,id=hf_token \
    python /repo/gpu/fetch_assets.py /repo/gpu/assets.json /assets

# Only the modules the harness needs, not the whole web application.
COPY backend/app/runner/harness.py /repo/backend/app/runner/harness.py
COPY backend/app/runner/speechcheck.py /repo/backend/app/runner/speechcheck.py
COPY backend/app/runner/speechlab.py /repo/backend/app/runner/speechlab.py
COPY content/exercises /repo/content/exercises
COPY gpu/handler.py /repo/gpu/handler.py

CMD ["python", "-u", "/repo/gpu/handler.py"]
