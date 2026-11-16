# AIORI backend: the replay server (all study API routes plus /api/replay/*), CPU inference with the frozen models.
# Source code only. Data, checkpoints, results and sensor metadata are mounted read-only at run time (compose.yaml)
# and must never be copied into this image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir numpy==2.5.4 pandas==3.1.0 \
 && pip install --no-cache-dir torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Mirrors the repository's src/ folder: the code resolves its data from this root by default.
WORKDIR /srv/src
COPY src/rebuild_aiori_metadata.py rebuild_aiori_metadata.py
COPY src/aiori/ aiori/
COPY src/aiori2/ aiori2/
COPY src/experiments/accuracy_optimization/aopt/ experiments/accuracy_optimization/aopt/
COPY src/app/backend/ app/backend/
COPY src/app/data/percentage_error.json app/data/percentage_error.json

# The web service serves the frontend; this folder stays empty so the API serves no pages of its own.
RUN mkdir -p /srv/no-frontend
ENV AIORI_HOST=0.0.0.0 \
    AIORI_REPLAY_PORT=8010 \
    AIORI_FRONTEND_DIR=/srv/no-frontend

USER nobody
EXPOSE 8010
CMD ["python", "app/backend/replay_server.py"]
