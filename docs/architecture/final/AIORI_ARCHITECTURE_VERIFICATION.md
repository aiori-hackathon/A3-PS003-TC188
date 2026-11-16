# AIORI Architecture Verification

**Status: PASS.** All 15 diagrams and 8 documents were produced from inspected source and live state. Every diagram passed Archify's automated gates. No application file was changed.

## 1. Method

- **Read-only audit** of the working tree at Git HEAD `dad975f` (branch `docs/update-live-link`). Most of the 2022–2024 code is **uncommitted or untracked**: `src/aiori2`, `src/app/backend/replay*.py`, `disruptions.py`, `disruption_detect.py`, the replay frontend files. The architecture therefore describes the working tree, not the commit.
- **Archify source links.** For that reason Archify diagrams carry no `meta.repository` / `sources` links: Archify verifies source links against committed bytes, and the uncommitted files would fail that check. Evidence is listed below as path:line instead.
- **Live state:** checked with `Get-NetTCPConnection`, process command lines, `Get-ScheduledTask` and the cloudflared process list. No server was started, stopped or restarted for this task.
- **Experiments:** no training, evaluation or data processing was run.

## 2. Archify capabilities used

| Capability | Available | Used for |
|---|---|---|
| Native diagram types: architecture, workflow, sequence, dataflow, lifecycle, erd, tree, class, timeline, waterfall, explainer | yes (skill CLI `archify.mjs` v3.2.0; it is a local skill, not an MCP server) | architecture (8), sequence (4), dataflow (2), workflow (2), tree (1) |
| Deployment diagram type | no dedicated type | architecture diagram with workstation boundary (HLD-05) |
| Source-editable output | yes: `candidate.json` beside each HTML | all diagrams |
| Interactive exploration | yes: standalone HTML viewer (zoom, route tracing, lens, light/dark, Live/Still motion) | all diagrams |
| Export | viewer exports PNG, SVG, WebM | available; not pre-generated |
| Automated gates (`finalize`) | validate, deliver, strict check, real-browser check | all 15 passed |
| Module/API documentation | via node and card text | API and module details are in the Markdown inventories |

No paid service was used, and nothing was sent to a third party.

## 3. Diagram validation

| Diagram | Type | finalize | Notes |
|---|---|---|---|
| HLD-01 | architecture | pass | screenshots inspected (light, 2048 px) |
| HLD-02 | architecture | pass | automated only |
| HLD-03 | architecture | pass | automated only |
| HLD-04 | architecture | pass (`review-2`) | FL card corrected after checking the Stage 3 run config (100 local steps, not 10) |
| HLD-05 | architecture | pass | automated only |
| LLD-01 | tree | pass | first draft too wide to read; root levels merged |
| LLD-02 | architecture | pass | automated only |
| LLD-03 | sequence | pass | note spacing repaired |
| LLD-04 | sequence | pass | automated only |
| LLD-05 | dataflow | pass (`review-3`) | one wrong flow (thresholds → quality) corrected to readings + mask → quality. Re-laid out; screenshot inspected; one remaining crossing (day type / readings + mask) |
| LLD-06 | sequence | pass | automated only |
| LLD-07 | sequence | pass (`review-2`) | participant label shortened; unverified server learning rate removed from a card |
| LLD-08 | workflow | pass | automated only |
| LLD-09 | dataflow | pass | automated only |
| LLD-10 | workflow | pass | automated only |

"Automated only" means Archify's browser check passed (containment, readability, no overflow); a person has not looked at it.

## 4. Evidence (source paths and verified facts)

| Fact | Evidence |
|---|---|
| Forecast API endpoints, validation, errors, static serving, CORS | `src/app/backend/server.py:345-432` |
| Engine loads nine-month data, local-only GRU seed 1 (clipped) and centralised GRU seed 0; optional candidate | `server.py:85-146`, `server.py:109-114` |
| GRU input (1, 12, 4): scaled flow, observed flag, time-of-day sin/cos; custom readings forward-filled ≤ 6 then training median | `server.py:168-191` |
| Predict body limits, models, observations 0–5000, response fields | `server.py:193-268`, `server.py:397-419` |
| Defaults 127.0.0.1:8000, CPU, 2 threads; env variables | `server.py:7-8, 42-47, 89-90, 438` |
| Candidate is opt-in, historical only, not federated | `src/app/backend/candidate.py:1-12`, `server.py:111` |
| replay_server subclasses server.Handler; parameter sets; disruption routing; port 8010 | `src/app/backend/replay_server.py:46-47, 51, 61, 91, 109, 133-143` |
| Replay data paths, timeline check, mmap, model loading with manifest/hash/config checks | `src/app/backend/replay.py:43-45, 94-128, 161-204` |
| Replay forecast contract (t0 = T + 1, inputs, 422 cases, comparison) | `replay.py:328-401` |
| Disruptions loading checks, list/detail/series/compare/robustness, limits | `src/app/backend/disruptions.py:54-63, 80-125, 149-430` |
| Detector stages, thresholds, quality bits, grouping, ids, build | `src/app/backend/disruption_detect.py:89-487` |
| Catalog: 16,445 events; thresholds −4.93 / +5.33; chance check 718 vs ~256 | `src/results/aiori2/phase9b_disruptions/disruption-detector-v1/catalog.json` |
| Arms and scalers | `src/aiori2/arms.py:18-22` |
| Five models | `src/aiori2/models/factory.py:19, 36` |
| Eight server rules | `src/aiori2/fl/algorithms.py:89-206` |
| Single-process FL engine, participation, local steps | `src/aiori2/fl/engine.py:1-7, 126-160` |
| Runner, checkpoints, early stopping | `src/aiori2/run.py:1-10, 218-460` |
| Stage plan: 40 / 56 / Stage 3 | `src/aiori2/plan.py:1-10`; 96 run folders in `phase7_stage1`; 24 in `phase8c_stage3` |
| Stage 3 winner run config: 100 local steps, SGD lr 0.01, batch 128, server momentum 0.9, 20 epochs × 2 M windows, patience 3 | `src/results/aiori2/phase8c_stage3/fedavg_geo-A-mlp-medium-fedavgm-s20261027-ba24d6a0a6/config.json` |
| Stage 3 ranking (MLP + FedAvgM A first) | `src/results/aiori2/phase8c_stage3/analysis/stage3_configs.csv` |
| Frozen one-time evaluation | `src/aiori2/final_eval.py:1-12` |
| Test WAPE 10.506% / 10.546% | `src/results/aiori2/phase8c_stage3/analysis/PHASE8C_REPORT.md`; `replay.py:51` |
| Frontend routes and gating | `src/app/web/src/App.tsx:7-35` |
| Page → API usage | grep of `api.`, `replayApi.`, `disruptionsApi.` in `src/app/web/src/pages` |
| Study API client | `src/app/web/src/lib/api.ts:1-38`; `lib/data.tsx:21-24` |
| Frontend stack and versions | `src/app/web/package.json`: React 19.3.1, React Router 8.5.1, Recharts 3.10.1, Leaflet 1.9.4, Tailwind 4.3.3, Vite 8.3.5, TypeScript 7.0.2 |
| Keep-alive behaviour | `.keepalive/keepalive.ps1` |
| Stage 3 watchdog | `src/aiori2_watchdog/README.md`, `config.json` (`auto_resume: false`) |

## 5. Contracts verified

| Contract | Value |
|---|---|
| Sensors | 200 (4 regional clients × 50) |
| Data | XTraffic 2022-01-01 00:00 … 2024-12-31 23:55, 5-minute steps, 315,648 steps |
| Model input / output | 12 steps → 12 steps (60 min) |
| Units | vehicles per 5-minute interval |
| Models compared | MLP, GRU, TCN, PatchTST, N-HiTS |
| Federated algorithms | FedAvg, FedProx, FedAdam, FedYogi, FedAdagrad, SCAFFOLD, FedNova, FedAvgM |
| Selected model | MLP + FedAvgM, history A (trained on 2023); served for geo and IID partitions, seed 20261027 |
| Coordinates | provisional |
| Congestion | not shown |

## 6. Preservation

| Check | Result |
|---|---|
| Git status before vs after | identical except the new untracked `docs/` folder; HEAD unchanged (`dad975f`) |
| SHA-256 snapshot of 25,839 repository files (files > 20 MB by size + mtime) taken before work | 0 removed. 2 changed: `src/results/aiori2/phase8c_stage3_watchdog/watchdog.log` and `watchdog_status.json`, written by the scheduled Stage 3 watchdog's own 10-minute tick, not by this task. 110 added, all under `docs/architecture/final/` |
| Application source, builds, backend, tests | unchanged |
| Datasets, checkpoints, scalers, manifests, results | unchanged |
| Existing architecture documents (`architecture/`, `.archify/`) | unchanged |
| Servers, ports, tunnel, scheduled tasks | not touched |
| Commits, pushes, deployments | none |

## 7. Uncertainty and limitations of this documentation

- The deployment files (Dockerfile, Hugging Face script, Vercel config) exist, but whether they were ever used was not verified; they are documented as configuration only.
- The 2023 GRU study internals (`src/aiori`) are summarised from module names, the server's use of them and reports, not traced line by line.
- `src/experiments/accuracy_optimization` (candidate model training) is summarised, not traced.
- Federated algorithm server rules are summarised from class names and docstrings; their maths was not re-derived.
- Twelve diagrams have automated checks only; HLD-01 and LLD-05 were also inspected as screenshots.
- The Phase 10 redesign is shown only as an isolated preview (port 8020), because it is not approved or active.
- A live process was observed on port 8020; it is a design preview started in an earlier session and is outside the final architecture.
