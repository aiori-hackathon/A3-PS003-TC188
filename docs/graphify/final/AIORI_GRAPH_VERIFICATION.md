# AIORI graph verification

**Status: PASS.** The graph has 55 nodes, 57 links and 4 views.
- Every node carries evidence paths.
- Code relationships were confirmed with Graphify's static analysis.
- All four views were inspected as screenshots at 1600 × 900.
- No existing file was changed.

## 1. Graphify: what is installed and what it does

| Item | Finding |
|---|---|
| Installed | `graphifyy` 0.9.97 (Python package, `graphify` CLI) plus its command-line skill. It is not an MCP server unless `--mcp` is started. |
| What it builds | Knowledge graphs of code (AST: imports, calls, inheritance, file containment) and, with an LLM backend, of documents, papers and images. It then detects communities. |
| Graph types | Code dependency graph; concept/relationship graph; interactive force-directed node-edge graph; collapsible file tree (`graphify tree`); Mermaid call-flow page (`export callflow-html`). There is no fixed-layout, multi-view presentation diagram. |
| Exports | HTML (vis-network, loaded from unpkg), SVG, GraphML, Obsidian vault, wiki Markdown, Neo4j/FalkorDB Cypher, JSON (`graph.json`, node-link). |
| Links from nodes | Nodes carry `source_file` and `source_location`. Its HTML shows them in a side panel but does not open files. |

**How Graphify was used**

1. **Verify code relationships.** Ran `graphify extract --code-only` (local, no API key, nothing sent anywhere) on a scratch copy of `src/app/backend` (6 files), `src/aiori2` and `src/app/web/src`. The result was 811 nodes and 2,587 edges. The cross-file edges were then checked against the links drawn (§3).
2. **Render the curated source.** Ran `graphify export html` on `AIORI_GRAPH_SOURCE.json` to produce `AIORI_GRAPH_GRAPHIFY.html`.

**Why the main page is separate.** Graphify's own HTML is force-directed. It cannot hold fixed layered layouts, bands or separate views, so it could not meet the "understood within one minute" goal. The main page is therefore a small self-contained HTML/SVG renderer (`build/template.html`) that uses no libraries and reads the same Graphify-format source.

The existing `graphify-out/` (dated 2026-10-30) was not touched: extraction ran in the scratch folder.

## 2. Nodes: source of each fact

| View | Nodes | Evidence used (full paths are in each node's `evidence`) |
|---|---|---|
| Complete system | 16 | `server.py:85-146, 345-432`; `replay_server.py:46-143`; `replay.py:43-204`; `disruptions.py:54-430`; `disruption_detect.py:89-487`; `aiori2/pipeline.py`; `aiori2/fl/engine.py`; `aiori2/final_eval.py`; `App.tsx:7-35`; live port and process check from the architecture task |
| Training & experimentation | 15 (XTraffic is shared with view 1, so 55 unique nodes in total) | `aiori2/arms.py:1-40`; `models/factory.py`; `fl/algorithms.py:89-206`; `plan.py:3-6`; `validate.py:118-125`; `experiments/conservative/A/experiment.json`; `phase7_stage1/stage2_results.csv`; `PHASE8C_REPORT.md §3-7` |
| Forecasting & data flow | 13 | `replay.py:9-10, 51, 117-121, 161-204, 328-401`; `arms.py:5-6`; `disruptions.py:149-430`; `components/replay/ForecastView.tsx` |
| Engineering & results | 12 | `PHASE8C_REPORT.md`; `final_manifest.json`; `final_eval.py`; `catalog.json` (`summary`, `audit.chance_coordination`); `PHASE9B_DISRUPTION_REPORT.md:74` |

## 3. Links checked against Graphify's code graph

These code links were confirmed by the AST extraction (counts are Graphify edges between the files):

| Link in the graph | Graphify evidence |
|---|---|
| Extended API reuses the Study API (separate process) | `replay_server.py → server.py` imports |
| Extended API serves Replay and the Explorer | `replay_server.py → replay.py` (9 calls), `→ disruptions.py` (7 calls) |
| Explorer opens and uses Replay | `disruptions.py → replay.py` (13 calls); `Disruptions.tsx → lib/replay.ts` |
| Study API uses the optimised candidate | `server.py → candidate.py` (3 calls) |
| Replay loads federated models with training-only scalers | `replay.py → aiori2/models/factory.py`, `→ aiori2/arms.py` |
| Training runner uses the FL engine, models, arms and preprocessing | `run.py → fl/engine.py`, `models/factory.py`, `arms.py`, `data.py`, `pipeline.py` |
| FL engine uses the algorithms and models | `fl/engine.py → fl/algorithms.py`, `models/factory.py`, `arms.py` |
| Selection and test reuse the runner and the persistence baseline | `final_eval.py → run.py`, `baseline.py` |
| Replay and the Explorer share the forecast chart | `Replay.tsx` and `Disruptions.tsx → components/replay/ForecastView.tsx` |

The remaining links are data or file relationships, not code imports. They were checked as follows:

| Links | Checked against |
|---|---|
| Reads of arrays, catalog, checkpoints and manifests | load paths in `replay.py` and `disruptions.py` |
| Writes by preprocessing and detection | output paths in `pipeline.py` and `disruption_detect.py` |
| Stage flow and frozen selection | `plan.py` and `PHASE8C_REPORT.md` |

## 4. Numbers

| Figure | Value | Source |
|---|---|---|
| Sensors / steps | 200 / 315,648 five-minute steps (2022-01-01 to 2024-12-31) | `aiori2/__init__.py`, architecture verification §5 |
| Splits | train ≤ 2023-12-31 (A: 2023, B: 2022–2023); val 2024-01-01–06-30; test 2024-07-01–12-31 | `experiment.json`, `validate.py:118-120` |
| Stage 1 | 40 configurations | `plan.py:3` |
| Stage 2 | 64 configurations, 8 identical to Stage 1, so 56 new runs | `plan.py:4-5`; `stage2_results.csv` (64 rows, 8 `reused_stage1`) |
| Stage 3 | 24 runs, 18,920 rounds, 7,568,000 local steps, 10.15 GPU-hours, ≤ 20 epochs, 3 seeds, RTX PRO 4000 | `PHASE8C_REPORT.md §4-6` |
| Winner | MLP + FedAvgM, history A; validation score 10.866 | `PHASE8C_REPORT.md §7` |
| Test WAPE | IID 10.506%, geographic 10.546% (3-seed means); persistence 12.495% | `PHASE8C_REPORT.md`; `replay.py:51` |
| Catalog | 16,445 candidate events; 718 with ≥ 3 sensors against ~256 expected by chance | `catalog.json` `summary.n_events`, `audit.chance_coordination` |
| Detector thresholds | z −4.93 / +5.33 and ≥ 25% deviation | `catalog.json` `thresholds` (architecture verification §4) |

**Labelling rules applied in the graph**
- Stage 3 figures are marked "Stage 3 only".
- WAPE is defined on the page as an error measure, not accuracy.
- Anomalies are called "candidate" and "not confirmed incidents".

## 5. Comparison with the Archify HLD and LLD

| Archify diagram | Covered in this graph |
|---|---|
| HLD-01, HLD-02, HLD-05 | View 1 |
| HLD-04, LLD-07 | View 2 |
| LLD-03, LLD-04, LLD-08 | View 3 |
| LLD-05, LLD-06 | Explorer and catalog in views 1 and 3 |

Every implemented subsystem in the HLD has a node:
- runtime: web app, both servers, Forecasting, Replay and the Explorer;
- stored on disk: arrays, models, catalog and results;
- offline: preprocessing, federated training, selection and test, and anomaly detection.

**Left out on purpose**
- Inactive deployment files: tunnel, Docker, Hugging Face, Vercel.
- The keep-alive and Stage 3 watchdogs.
- The legacy static frontend.
- Individual API routes and component files; these are in the Archify LLD.
- The unapproved Phase 10 preview on port 8020.

## 6. Rendered checks

**Screenshots.** Taken with headless Microsoft Edge at 1600 × 900 for all four views and inspected:
- No overlapping boxes. A script measured every pair of boxes in every view and found 0 overlaps.
- No clipped text.
- No arrows crossing in view 1.
- In view 2, the inputs converging on the training runner fan in without crossing boxes.

**Fixes made after the first inspection**
- Text was too small, so the detail panel became an overlay and fonts were enlarged.
- Four edge labels collided with boxes on short arrows; they were removed or moved beside the line.
- Two cards overflowed, so cards now size to their text.
- Two view 3 boxes touched, so that view was respaced.

**Interaction probe** (a scratch copy of the page with a test script):

| Check | Result |
|---|---|
| Clicking a box opens its panel and dims unrelated items | 11 boxes dimmed, 4 links kept |
| Enter key selects a box | yes |
| Esc closes the panel | yes |
| Zoom changes the view | yes |
| Exported SVG parses as valid XML | 14 KB |
| Switching views redraws the graph | yes |

**Dark mode** (added after review). All four views were screenshotted with the system set to dark and inspected. A probe confirmed:
- System → Light → Dark switches the canvas and page colours.
- A selected box stays selected across a switch.
- Exported SVG carries the active theme's colours.

**Not checked**
- The PNG download, which needs a real click to save a file.
- Touch gestures.
- The Graphify page offline; it loads vis-network from unpkg. It rendered with internet access: 55 nodes, 57 edges and 7 named groups.

## 7. Preservation

| Check | Result |
|---|---|
| SHA-256 snapshot of 25,839 files (taken before the architecture task) | 0 removed. 2 changed: `phase8c_stage3_watchdog/watchdog.log` and `watchdog_status.json`, written by the scheduled watchdog's own 10-minute tick. Large files: size and mtime unchanged. |
| New files | only `docs/graphify/final/` (8 files), apart from the earlier `docs/architecture/final/` |
| `docs/architecture/final`, `architecture/`, `.archify/`, `graphify-out/`, `web-phase10/` | newest file times all predate this task |
| Git | status unchanged except the untracked `docs/`; HEAD `dad975f`; nothing committed or pushed |
| Servers, ports, tunnel, scheduled tasks | not touched; no training or evaluation was run |
