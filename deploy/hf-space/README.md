---
title: AIORI Traffic Intelligence API
emoji: 🚦
colorFrom: gray
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

Inference API for AIORI Traffic Intelligence (PS-003): real next-hour traffic-flow forecasts from the study's frozen
original GRU and the optimized MLP. Source: https://github.com/aiori-hackathon/A3-PS003-TC188

Endpoints: `GET /api/health`, `/api/metrics`, `/api/sensors`, `/api/demo`, `/api/models`; `POST /api/predict`.
Model checkpoints and processed data are private and are not part of this Space repository.
