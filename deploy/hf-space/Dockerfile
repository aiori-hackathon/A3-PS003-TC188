# AIORI Traffic Intelligence API on a Hugging Face Docker Space (free CPU basic).
# The image contains only public code. Model checkpoints and processed data live in a PRIVATE Hugging Face dataset
# and are downloaded at start-up with the Space secret HF_TOKEN; they are never part of this public Space repository.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir numpy==2.5.4 pandas==3.1.0 huggingface_hub \
 && pip install --no-cache-dir torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu

RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user
ARG REPO=https://github.com/aiori-hackathon/A3-PS003-TC188.git
RUN git clone --depth 1 "$REPO" app
COPY --chown=user start.sh /home/user/start.sh

ENV AIORI_HOST=0.0.0.0 AIORI_PORT=7860 AIORI_CANDIDATE=1 AIORI_THREADS=2 \
    AIORI_FRONTEND_DIR=/home/user/app/app/frontend PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["sh", "/home/user/start.sh"]
