# AIORI final graph

One interactive page that explains the finished AIORI platform in four views. Every box links to the code or result file that backs it.

## Files

| File | What it is |
|---|---|
| `AIORI_FINAL_GRAPH.html` | **The main graph.** Four views, fixed layouts, details on click, export. Works offline; double-click to open. |
| `AIORI_GRAPH_GRAPHIFY.html` | The same 55 nodes and 57 links drawn by Graphify's own explorer (force layout, search, group filters). Loads `vis-network` from unpkg, so it needs internet. |
| `AIORI_GRAPH_SOURCE.json` | Editable source for both pages, in Graphify's node-link format. Adds a position per view, a short detail text and evidence paths to each node. |
| `AIORI_GRAPH_VERIFICATION.md` | How each node, link and number was checked. |
| `build/build_graph.py` | Checks the source and rebuilds `AIORI_FINAL_GRAPH.html` from `build/template.html`. |
| `build/graphify_labels.json` | Group names for the Graphify export. |

## The four views

| View | Question it answers |
|---|---|
| 1. Complete system | What AIORI is made of, and what runs when. Three bands: runtime (website in use), stored on disk, offline batch jobs. |
| 2. Training & experimentation | What was trained, compared and selected: data, the design space, the three-stage search, Stage 3 figures, and the validation-only choice. |
| 3. Forecasting & data flow | How one 60-minute forecast is made: readings → validity → gap filling → 12-step window → scaling → frozen model → 12-step forecast → comparison with what happened. |
| 4. Engineering & verified results | What was built (counts), what was run (Stage 3, selection, test), and what was measured (test WAPE, baseline). |

**Colours and lines.**
- Colour shows what a box is: person, web interface, running service, stored data, offline job, implementation count, or measured result.
- Solid lines are runtime requests or reads.
- Dashed purple lines are offline jobs producing or feeding something.
- Dotted lines are code reuse, a link between pages, or data used only for comparison.

## How to use it

- **Open:** double-click `AIORI_FINAL_GRAPH.html`. No server is needed.
- **Switch views:**
  - Click a tab, or press ← / → when a tab has focus.
  - Each view has its own address (`#system`, `#training`, `#forecast`, `#results`), so you can link to one directly.
- **Explore:**
  - Click a box, or Tab to it and press Enter. A panel shows what it does, its connections and its evidence, and dims everything that isn't connected.
  - If a box appears in more than one view, the panel links to the other views.
  - Esc or × closes the panel.
- **Move around:** drag to pan, scroll or use + / − to zoom, and press Fit to reset.
- **Theme:** follows the system light/dark setting by default. The *Theme* button cycles System → Light → Dark, and this browser remembers the choice. Exports use the theme on screen.
- **Export:** *Export SVG* and *Export PNG* (2×) save the current view. The SVG carries its own styles.
- **Edit:** change `AIORI_GRAPH_SOURCE.json`, then run:

  ```
  python docs/graphify/final/build/build_graph.py
  graphify export html --graph docs/graphify/final/AIORI_GRAPH_SOURCE.json --labels docs/graphify/final/build/graphify_labels.json
  ```

  Run the second command from a scratch folder, because it writes `graph.html` to the current directory. Copy that file in as `AIORI_GRAPH_GRAPHIFY.html`.

## Reading notes

- Federated learning is simulated offline: four clients run in one process on one workstation GPU. The website never trains.
- Two server processes make up the running app:
  - `:8000` serves the 2023 GRU study models.
  - `:8010` reuses its code and adds Replay and the Disruption Explorer, which use the frozen MLP + FedAvgM.
- The Cloudflare tunnel, Docker, Hugging Face and Vercel files exist but are not running, so they are left out.
- The Phase 10 redesign preview on port 8020 is not approved, so it is left out.
- Stage 3 figures (24 runs, 18,920 rounds, 7,568,000 steps, 10.15 GPU-hours) cover Stage 3 only.
- Test WAPE is an error measure (lower is better), not an accuracy percentage.
- The 16,445 disruptions are statistical candidates, not confirmed incidents.

The full HLD and LLD are in `docs/architecture/final/`.
