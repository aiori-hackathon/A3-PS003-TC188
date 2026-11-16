# AIORI 2.0 Phase 10: UI redesign, progress and report

Last updated 2026-11-15. A fresh session should read `PHASE10_UI_CONTINUE.md` first, then this file.

## A. Status

**PARTIAL: a complete, working, verified preview is ready for review; some page-level refinement is not done.**

- **Done.** The redesign covers every page through a new design system, new navigation and a rebuilt Overview. Every existing feature works in the preview.
- **Not done.** Replay, the Disruption Explorer and Forecasting kept their Phase 9 layouts and received the new look only; no control or layout was reworked (section J).
- **Untouched.** The working apps on ports 8000 and 8010 are unchanged. Nothing was committed, pushed, deployed or activated.

## B. Design direction

The old interface was near-black with a single green accent, the most common look for a generated dashboard. The redesign takes its language from the freeway the sensors sit on.

- **Surface.** A light engineering sheet: page `#eceeed`, white cards, 1 px rules, no card shadows. One soft shadow is kept for things that float (tooltips, the menu, the Overview sign).
- **Signature element.** Navigation is an overhead guide sign: green panel, white inset border, white lettering, the current page as a white plate. Overview repeats the idea once, as a distance sign listing where you can go with one real figure per destination. Everything else stays quiet.
- **Type.** Overpass Variable, an open-source family drawn from Highway Gothic sign lettering, with Overpass Mono for measurements.
- **Colour has fixed jobs, borrowed from sign colours:**

| Colour | Job |
|---|---|
| Green `#006a4e` | navigation, primary action, the model's forecast |
| Amber `#a85a00` | values observed afterwards, anomalous readings |
| Blue `#1f5fa8` | below-typical flow |
| Red `#c2362e` | missing readings, errors |
| Greys | reference series, typical flow |

- **Navigation is grouped by task:** Explore (Overview, Replay, Disruptions), Forecast (Forecasting), Evidence (Performance, Federated), Reference (Methodology, About). The full band shows from 1280 px; below that a menu sheet lists the full page names.
- **Honesty kept visible.**
  - The Overview panel for the selected model says the test WAPE is a whole-period average, not the expected error of one forecast, and says nothing about unusual events.
  - The footer repeats that nothing is live.
  - The old header label "Jan–Sep 2023", wrong for the 2022–2024 features, is replaced by "Historical data · not live".
- **Computed contrast:** secondary text 5.7:1 on white and 4.9:1 on the page; amber 5.1:1; white on sign green 6.6:1.

## C. Page inventory (every feature and where it is now)

| Feature | Where in the redesign | Change |
|---|---|---|
| Overview: example forecast, findings list, stat strip, results table | `/`, under "The original PS-003 study" | kept; section now labelled as the 2023 study |
| Overview entry points (forecast a recorded hour, forecast from your readings) | `/`, rows of the distance sign | were two buttons; same destinations |
| New on Overview: links to Replay and Disruptions, selected-model panel | `/` | added; figures read from the replay service, not hard-coded |
| Forecasting (historical hour, future forecast, examples, chart, step table) | `/forecast` | restyled only |
| Model performance (comparison, MAE/RMSE bars, tables) | `/performance` | restyled only |
| Federated learning, Methodology, About | `/federated`, `/methodology`, `/about` | restyled only |
| Not-found page | any unknown path | restyled only |
| Replay: date, slider, coverage strip, Play/Pause, previous/next, speed, map, sensor pick, Observed/Predicted, lead time, IID/Geographic, forecast, comparison table, provenance, deep links, back-to-event link | `/replay` | restyled; map now uses light tiles and dark marker outlines |
| Disruption Explorer: overview, filters, sorting, paging, list, detail, flow chart with band, missing markers, quality warnings, sensor table, comparison, replay link, forecast, robustness, catalog-wide table, limitations | `/disruptions` | restyled only |
| Service status, GitHub link, "Run a forecast" shortcut, skip link, focus-to-heading on navigation | sign band | kept; moved from sidebar and header |

Nothing was removed and no placeholder control was added.

## D. Frameworks and libraries

- **Kept:** React 19, TypeScript, Vite 8, Tailwind 4, React Router, Recharts, Leaflet, lucide. They already cover every need; replacing any would add risk for no visible gain.
- **Added (workspace only):** `@fontsource-variable/overpass` 5.3.0 and `@fontsource-variable/overpass-mono` 5.3.0. The fonts are self-hosted, so there is no runtime network dependency.
- **Not added:** component kits, motion libraries, table or query libraries. The app has one shared component file; a second design system would compete with it.
- **Design skills used for guidance:** frontend-design and impeccable (already installed).

## E. Higgsfield usage

| Question | Answer |
|---|---|
| MCP connected? | Yes |
| Plan and credits | Free plan, **0 credits** (read with the balance tool) |
| Generation tool available in this session? | **No**: only balance, catalog, workspace and import tools are exposed |
| Assets generated | None |
| Generation calls | 0 |
| Credits consumed | None by this work: no generation was possible or attempted. The one call made was the read-only balance check |
| Skipped because of cost uncertainty | All generation. With 0 credits and no generation tool, nothing could be verified as free |

All visuals are code: CSS, inline SVG and type. The app has no Higgsfield dependency.

## F. Implementation

Everything is inside `src/app/web-phase10/` (a copy of `src/app/web` source, with its own `node_modules`).

**Rewritten**
- `src/components/layout/Shell.tsx`: sign-band navigation, menu sheet, footer.
- `src/index.css`: tokens, fonts, light scheme, selection, caret, range and checkbox colours.
- `src/components/replay/replay.css`: light map.

**Edited**
- `src/components/ui/index.tsx`: buttons, tags, segmented control, tables, headings, form controls.
- `src/pages/Overview.tsx`: distance sign, selected-model panel, section label.
- `src/components/replay/ReplayMap.tsx`: marker outlines.
- 13 files recoloured by one fixed old-to-new colour table: charts, the other pages, `lib/study.ts`, `lib/replay.ts`.
- `package.json`, `package-lock.json`: two font packages and the `build:phase10` script.
- `.gitignore`: `dist-phase10/`.

**New outside the workspace**
- This file and `PHASE10_UI_CONTINUE.md`.
- The backup folder (section I).

No backend file changed.

## G. Preview

Address: **http://127.0.0.1:8020/**

It is running now as a background job of the session that built it, and may stop when that session ends. To start it again from the repository root (PowerShell):
```
$env:AIORI_FRONTEND_DIR = "$PWD\src\app\web-phase10\dist-phase10"; $env:AIORI_REPLAY_PORT = "8020"; $env:AIORI_CANDIDATE = "1"; .venv\Scripts\python.exe src\app\backend\replay_server.py
```
To rebuild after edits, in `src\app\web-phase10`:
```
npm run build:phase10
```
The preview uses the same backend code as port 8010. The two environment variables only choose the frontend folder and the port.

## H. Testing

| Check | Result |
|---|---|
| TypeScript `tsc --noEmit` | clean |
| Vitest (5 files) | 30 passed |
| `npm run build:phase10` | OK |
| Production-mode build of the workspace (scratch folder) | OK; no Replay or Disruptions page chunks and no replay API calls. One sentence of Overview text naming the Disruption Explorer is bundled but never shown in that mode |
| Console errors on the preview | none |

**Browser checks on port 8020 (desktop 1440 px)**
- All 8 routes and the not-found page render.
- Overview: the sign shows 16,445 candidate events and both test WAPE values, read from the service.
- Forecasting page: "Run forecast" draws the chart with no error.
- Disruptions:
  - Opening an event works.
  - The forecast shows MAE 98.3, the same value as before the redesign.
  - The robustness table renders.
  - The LA-42 filter gives 97 events.
- Replay:
  - The deep link opens 2023-08-20 16:00 with LA-42 selected.
  - Play advances to 16:20.
  - The forecast runs.
  - The Predicted layer shows "issued at 16:20 for 17:20".
  - "Back to candidate event" returns to the event.

**Phone (375 px)**
- No horizontal overflow on Overview, Replay, Disruptions, Forecasting or Performance.
- The menu opens with 8 links.
- Escape closes it and returns focus to the menu button.
- The closed menu is inert.

**Accessibility observations**
- Skip link, focus moved to the page heading after navigation, and visible focus rings (white on the sign, green elsewhere) are in place.
- Colour is never the only signal in charts: dashes, markers and labels are kept.
- Not done: a full keyboard walk through every control, and a screen-reader pass.

## I. Preservation

| Check (SHA-256 against `BASELINE.json`) | Result |
|---|---|
| `src/app/web` source, 46 files | unchanged |
| Production build `web/dist`, 26 files | unchanged |
| Extended build `web/dist-replay`, 33 files | unchanged |
| Backend: `server.py`, `replay_server.py`, `replay.py`, `disruptions.py`, `disruption_detect.py` | unchanged |
| Stage 3 artifacts and core data, 116 files | unchanged |
| Ports 8000 and 8010 | both answering; 8010 still serves its own build |
| Git | HEAD `dad975f`; nothing committed or pushed |
| Tunnel, watchdog, scheduled tasks, deployment | not touched |

**Rollback point:** `src/app/backups/phase10_20261115-152653/` (verified copy of the Phase 9A/9B frontend source plus `BASELINE.json`).

**To discard the redesign:** stop the port 8020 process and delete `src/app/web-phase10/`. Nothing else needs undoing.

## J. Progress and continuation

- [x] 1. Audit, backup, isolated workspace, design system
- [x] 2. Shell, shared components, chart and map colours
- [x] 3. Core pages: Overview rebuilt; Forecasting, Performance, Federated, Methodology and About carry the system
- [ ] 4. Replay: restyled and verified; layout and controls not reworked
- [ ] 5. Disruption Explorer: restyled and verified; layout not reworked
- [ ] 6. Consistency pass: partly done (phone layout, menu, focus); the full pass remains
- [x] 7. Verification of what exists

**Open items, most useful first**
1. Replay: the page heading, tags and control card push the map below the first screen at 900 px height. Move the timeline beside or under the map, or shorten the header.
2. Disruption Explorer: the affected-sensors table clips its "Flags" column inside the narrow card (it scrolls sideways). Give the column room or move flags under the sensor name.
3. Disruption Explorer: twelve filters show at once. Group the less-used ones behind "More filters".
4. Forecasting: review the left control column at tablet widths.
5. Chart tooltips on Replay and Disruptions use Recharts' default box with palette colours; the original pages use the custom tooltip. Share one.
6. Full keyboard and screen-reader pass; check every chart colour pair for colour-blind separation.
7. The 13-file recolour was a fixed colour table. Moving chart colours into one theme module would make later changes one edit.

## K. Approval requirement

The redesign is not active anywhere except the preview on port 8020. Making it the working interface would mean replacing `src/app/web` source with the workspace and rebuilding `dist-replay` (and, separately, `dist` for production). That needs explicit approval and is not started.
