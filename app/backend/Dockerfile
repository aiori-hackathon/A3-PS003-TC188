# AIORI Traffic Intelligence API (+ the built React front end), CPU only.
# Build from the repository root after `npm run build` in app/web:
#   docker build -f app/backend/Dockerfile -t aiori-traffic .
# The frozen research artifacts are private and are NOT copied into the image. Mount them read-only at run time:
#   docker run -p 8000:8000 \
#     -v /secure/nine_month:/artifacts/data:ro -v /secure/runs:/artifacts/runs:ro -v /secure/final:/artifacts/final:ro \
#     aiori-traffic
# Never push an image or volume that contains them to a public registry.
FROM python:3.12-slim

WORKDIR /srv
RUN pip install --no-cache-dir numpy==2.5.4 pandas==3.1.0 \
 && pip install --no-cache-dir torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu

COPY aiori/ aiori/
COPY app/backend/ app/backend/
COPY app/data/percentage_error.json app/data/percentage_error.json
COPY app/web/dist/ app/web/dist/

ENV AIORI_HOST=0.0.0.0 \
    AIORI_PORT=8000 \
    AIORI_DATA_DIR=/artifacts/data \
    AIORI_RUNS_DIR=/artifacts/runs \
    AIORI_FINAL_DIR=/artifacts/final \
    AIORI_FRONTEND_DIR=/srv/app/web/dist

USER nobody
EXPOSE 8000
CMD ["python", "app/backend/server.py"]
