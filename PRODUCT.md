# Product

<!-- impeccable:product-schema 1 -->

> Inferred from the project lead's written brief of 2026-10-29 (no interview round: the brief instructed autonomous work without routine questions). Facts below come from that brief and from the repository's verified reports; items marked *inferred* are assumptions.

## Platform

web

## Stack
delegated: Python standard-library HTTP server with real PyTorch inference (src/app/backend), and a React 19 + TypeScript + Vite front end (src/app/web) with Tailwind CSS 4, React Router, Recharts and Lucide icons, using shadcn/ui-style primitives written in place. Node.js LTS was installed with the project lead's approval on 2026-10-29. The earlier dependency-free HTML/CSS/JavaScript front end stays in app/frontend as a fallback.

## Users
Hackathon judges, research reviewers and the project team (*inferred*: technically literate, evaluating a federated-learning traffic-forecasting study in a demo session on a laptop or projector).

## Product Purpose
AIORI Traffic Intelligence shows what the completed AIORI PS-003 study found and lets a reviewer run the study's frozen GRU model on historical traffic to see a genuine next-hour forecast. Success: a reviewer understands the result (centralised vs local-only vs federated, versus simple baselines) and trusts that every number and forecast is real.

## Positioning
A traffic-forecasting demo whose every figure traces to a frozen, hash-verified research artifact: the forecasts come from the study's own checkpoints and normalisation, and the metrics come from a held-out test evaluated once.

## Operating Context
Historical XTraffic 2023 flow data (January–September), 200 sensors in four provisional geographic client groups, 5-minute intervals, 12-step input and 12-step forecast. Nothing is live; all observations are historical.

## Capabilities and Constraints
- Real inference with the validation-selected local-only GRU (negative predictions clipped at 0) and a centralised GRU for comparison.
- No retraining; the held-out test is not re-evaluated; no raw sensor IDs, file paths or checkpoints are exposed.
- Percentage errors (WAPE, sMAPE) are computed read-only from frozen predictions.
- Public hosting is undecided (needs publication approval for data-derived content).

## Evidence on Hand
`docs/phase9_results.md`, `docs/phase9_executive_summary.md`, `results/phase9/final/analysis.json` (aggregate), frozen run artifacts under `results/phase9/` (local only). No testimonials, users or deployment claims exist; none may be invented.

## Product Principles
1. Every number is traceable to a frozen artifact; nothing is mocked.
2. Historical is always labelled historical; nothing implies live traffic.
3. Lower error is better, and every comparison says against what.
4. Limitations are part of the product, not a footnote.

## Accessibility & Inclusion
WCAG AA contrast, keyboard navigation and visible focus, reduced-motion respected (*inferred* standard practice; no product-specific requirement given).
