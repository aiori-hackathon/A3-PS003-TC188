# AIORI UI refinement: dark mode, outdated text, Replay map overlap

Date: 2026-11-15. Scope: the isolated Phase 10 workspace `src/app/web-phase10` only (preview http://127.0.0.1:8020/). This report sits beside the Phase 10 reports and does not replace them.

## 1. Status: **PASS**

All three fixes are in and verified:
1. the Replay map overlap;
2. dark mode;
3. outdated copy.

One check is partly verified: map zooming (see §8 and §10). Nothing is committed, deployed or activated.

## 2. Root cause of the map overlap

- The navigation band is `position: sticky; top: 0; z-index: 30` (`Shell.tsx`).
- Leaflet positions its panes and controls with `z-index` 400–1000.
- The map's container (`ReplayMap.tsx`) had `overflow: hidden` but did not create a stacking context.
- As a result, Leaflet's z-indexes competed directly with the sticky navigation in the page's root stacking context, and won.
- When the page scrolled the map under the navigation, the map canvas was painted over it.

**How it was reproduced.** `elementFromPoint` at the navigation's position, with the map scrolled under it:
- 446 px wide: `CANVAS` at all three probe points.
- 1440 px wide: `CANVAS` at 3 of 5 points.

## 3. Fix

One class: `isolate` (`isolation: isolate`) on the map container in `ReplayMap.tsx`.

- Leaflet's z-indexes now stay inside the map.
- No z-index value was changed anywhere.
- Leaflet, its controls and its interactions are unchanged.

**After the fix.** The navigation is on top at every probe point:
- at 446 px and 1440 px;
- in Light and Dark.

The mobile menu sheet also opens above the map.

## 4. Dark mode

- **Tokens.** The existing Tailwind 4 tokens in `src/index.css` gain a `:root[data-theme="dark"]` block with the same token roles and night values.
  - Text and lines are lifted for contrast.
  - Green, amber, blue and red keep their jobs.
  - The guide-sign navigation stays sign green.
- **Light mode.** Every Light value is unchanged.
- **New tokens.** Six were added: `grid`, `ref`, `cursor`, `danger-ink`, `accent-tint`, `marker`. Each holds the exact hex value the components used inline.
- **Inline colours.** Hard-coded hex colours in charts, SVGs and tables now read the tokens (`var(--color-…)`), so they follow the theme. In Light they resolve to the same colours as before.
- **Theme switching.**
  - A new `src/lib/theme.ts` (about 30 lines, no dependency) reads, toggles and stores the theme.
  - It saves the choice in `localStorage` under `aiori-theme`.
  - It follows the system theme until the user chooses.
- **No flash.** A small inline script in `index.html` sets `data-theme` before the first paint.
- **Toggle.** A moon/sun icon button in the navigation band, next to the GitHub icon, with the same size and style.
- **Map.**
  - Tiles are inverted and desaturated in dark mode.
  - Controls, tooltip and attribution use the tokens.
  - Canvas markers can't read CSS variables, so their outline colours are read from the tokens and re-applied when the theme changes.
  - The relative-flow marker colours are unchanged in both themes.
- **Not added.** No new package and no new animation.

## 5. Outdated or incorrect text corrected

| Where | Before | After | Why |
|---|---|---|---|
| Replay header | "the frozen AIORI 2.0 federated model" | "the frozen federated model (MLP + FedAvgM, history A)" | The project is AIORI; "2.0" was an internal phase name. The model is named as in the frozen manifest. |
| Disruptions header | "the frozen AIORI 2.0 model" | "the frozen federated model (MLP + FedAvgM)" | Same reason |
| Disruptions forecast card | "the deployed Phase 9A model" | "the model served by Historical Traffic Replay" | An internal phase label was visible to users. |
| Overview section | "AIORI 2.0: the selected federated model" | "The selected federated model" | Internal version label |
| About, run instructions | `app\backend\server.py`, `app/web`, `app\tests`, `app/analysis/…` | `src\…` paths; adds that Replay and the Explorer run on `replay_server.py` (:8010) | The codebase moved into `src/`, so the old paths do not exist. |
| About, data line | "Data: XTraffic 2023" | "XTraffic 2022–2024 (the original GRU study used January–September 2023)" | Replay and the Explorer use 2022–2024. |
| Methodology header | Generic description that reads as if it applies to everything | Says it describes the original 2023 GRU study, and that the 2022–2024 study used the same rule: trained up to 2023, selected on Jan–Jun 2024 validation, frozen, tested once on Jul–Dec 2024 | The page's split and figures are 2023-only. |
| Federated header | "The study simulated four clients…" | Names the original 2023 GRU study, notes the figures are from it, and that the 2022–2024 study compared eight algorithms and selected MLP + FedAvgM | It previously implied FedAvg on the GRU was the final result. |
| `index.html` meta | Description named only the 2023 GRU models; `theme-color` was the pre-redesign black `#0e0e0e` | Describes forecasting, replay and the candidate-anomaly explorer over 2022–2024; `theme-color` is sign green | Outdated metadata |

**Sources checked:**
- `src/aiori2/validate.py`;
- the history A `experiment.json` (split dates);
- `PHASE8C_REPORT.md` (winner, test WAPE 10.506% IID and 10.546% geographic);
- `replay.py:51, 171-174`;
- `aiori2/fl/algorithms.py` (8 algorithms);
- the repository layout.

**Checked and left unchanged because they were already correct:**
- the 2023-study figures on Overview, Forecasting and Performance, which are all labelled as 2023;
- the 16,445 candidate-event wording;
- "not live" labels;
- the WAPE caveats.

No metric value was changed: they are read from the services.

## 6. Files modified

All under `src/app/web-phase10/`:
- **New:** `src/lib/theme.ts`
- **Theme and toggle:** `src/index.css`, `index.html`, `src/components/layout/Shell.tsx`
- **Map fix and map theming:** `src/components/replay/ReplayMap.tsx`, `src/components/replay/replay.css`
- **Inline colours → tokens (same Light values):**
  - `src/components/charts/index.tsx`, `src/components/replay/ForecastView.tsx`, `src/components/study.tsx`, `src/components/ui/index.tsx`
  - `src/lib/study.ts`
  - `src/pages/Forecast.tsx`, `src/pages/Performance.tsx`
- **Colours and copy:** `src/pages/Replay.tsx`, `src/pages/Disruptions.tsx`, `src/pages/About.tsx`, `src/pages/Overview.tsx`
- **Copy only:** `src/pages/Methodology.tsx`, `src/pages/Federated.tsx`
- **Rebuilt:** `dist-phase10/`

Nothing outside the workspace changed, apart from the backup folder (§9).

## 7. Unchanged

- Buttons, fonts, spacing, layouts and navigation structure.
- Recharts, Leaflet, Tailwind and lucide.
- Interactions, APIs and backend.
- Models, data and results.
- Archify and Graphify outputs.
- The production build (`web/dist`) and the Phase 9 build (`web/dist-replay`).

**Light-mode pixel comparison.** I took 1440 × 1000 screenshots of all 8 routes on the old build (backup, served temporarily on :8021, now stopped) and on the new one. The only differences are:
- the new theme icon in the navigation band, which shifts the status text and GitHub icon left by one button's width;
- the corrected text lines, which move the content below them on Methodology, Federated and About;
- the Forecasting page's measured inference time (2.46 ms vs 2.72 ms), which differs on every run.

## 8. Tests and browser checks

| Check | Result |
|---|---|
| `tsc --noEmit` | clean |
| `vitest run` | 30 / 30 passed (5 files) |
| `npm run build:phase10` | OK |
| All 8 routes in Light and Dark (headless Edge screenshots, inspected) | render; dark text, tables, tags, charts and map readable |
| Theme toggle | switches; the label changes; the choice is stored and survives a reload; follows the system theme when no choice is stored |
| Chart colours | a Disruption event's chart lines resolve to the token colours in both themes |
| Map overlap | fixed at 446 px and 1440 px, Light and Dark; the old build reproduced it at both widths |
| Mobile menu over map | the sheet is on top |
| Zoom control | reachable and not covered |
| Console errors | none |
| Forecasting | the forecast computes (page shows "computed in … ms") |
| Replay / Disruptions | both load; an event opens with its charts |
| `web/`, `web/dist`, `web/dist-replay`, backend vs Phase 10 baseline hashes | 0 mismatches |
| Ports 8000, 8010, 8020 | all answer `/api/health` |
| Git | status unchanged, HEAD `dad975f` |

## 9. Backup and rollback

- **Backup:** `src/app/backups/phase10_refine_20261115-221854/`
  - holds the workspace `src`, `index.html`, `package.json`, `vite.config.ts` and the previous `dist-phase10`;
  - `MANIFEST.sha256` lists 73 files, verified against the workspace before editing;
  - `GIT_STATUS_BEFORE.txt` records the git status.
- **Rollback.** Copy the backup's `src`, `index.html` and `dist-phase10` back over `src/app/web-phase10/`, then reload :8020. No server restart is needed.
- The earlier Phase 10 rollback (`backups/phase10_20261115-152653`) is unchanged.

## 10. Limitations

- **Map zooming not confirmed.** A scripted zoom-button click did not advance Leaflet's zoom animation in the embedded browser. This was the same on the old build, so it is a test-environment limit, not a regression. The zoom code is untouched; only a CSS class was added to the container.
- **Scroll-wheel zoom.** Wheel scrolling over the map still zooms the map rather than scrolling the page. That is Leaflet's existing behaviour and was left unchanged, because changing it would alter a map interaction.
- **Dark-mode contrast.** Checked by eye on screenshots, not with an automated contrast audit.
- **Dark-mode screenshots at phone width.** Not taken; the overlap fix was checked at 446 px.
