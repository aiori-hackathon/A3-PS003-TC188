# Design system — AIORI Traffic Intelligence

Direction (pinned by the project lead, 2026-10-29): a calm, premium developer-platform dark interface at the quality level of the Supabase site and dashboard, without copying its layouts, logo or assets. Minimal, precise, readable; one muted emerald accent; 1px borders instead of shadows.

## Tokens
| Token | Value | Use |
|---|---|---|
| `--bg` | `#0e0e0e` | page |
| `--bg-raised` | `#131313` | sidebar, top bar |
| `--surface` | `#181818` | cards |
| `--surface-2` | `#1f1f1f` | hover, table header, inputs |
| `--border` | `#262626` | default 1px border |
| `--border-strong` | `#353535` | inputs, emphasis |
| `--text` | `#ededed` | primary text |
| `--text-2` | `#a3a3a3` | secondary |
| `--text-3` | `#8a8a8a` | tertiary (≥4.5:1 on surface) |
| `--accent` | `#3bb27e` | final model, primary action, active nav |
| `--accent-soft` | `rgba(59,178,126,.12)` | selected backgrounds |
| `--accent-line` | `rgba(59,178,126,.35)` | selected borders |
| `--observed` | `#d6a24a` | real observations shown after the fact (truth), and the "not validated" caution on future forecasts |
| `--danger` | `#e5534b` | errors |

Chart series (Recharts in the React app): final model = accent; other approaches = neutral greys (`#bdbdbd`, `#8a8a8a`, `#5f5f5f`) with dash patterns; persistence = dotted grey; observed history = `--text`; observed afterwards = `--observed` dashed.

## Type
Geist (variable, self-hosted from @fontsource-variable, SIL OFL) for everything; Geist Mono only for code. Replaced the original Segoe UI Variable on 2026-10-29 so hosted builds render the same on every platform. Tabular numerals everywhere data appears.
Scale (px): 12 · 13 · 14 · 16 · 18 · 22 · 28 · 40 (display). Weights 400 / 500 / 600. Headings tracking −0.02em (display −0.03em). Body 14/1.6; prose measure ≤ 70ch.

## Space and shape
4-px base: 4 · 8 · 12 · 16 · 20 · 24 · 32 · 40 · 56 · 72. Radii: 6 controls, 10 cards, 999 pills. Elevation by border and surface step, not shadow (one soft shadow only for floating tooltips and the mobile drawer).

## Components
- Card: `--surface`, 1px `--border`, radius 10; optional header row (title 14/600 + secondary meta) separated by a 1px rule.
- Stat card: label 13 `--text-2`, value 28/600 tabular, footnote 12 `--text-3`.
- Buttons: primary (accent fill, near-black text), secondary (`--surface-2` + border), ghost; 36px high, 6px radius.
- Inputs/selects: 36px, `--surface-2`, 1px `--border-strong`, focus = accent border + 3px accent-soft ring.
- Segmented toggle and pills for model comparison; tables with 1px row rules and tabular numbers.
- Bars: CSS bar rows (label · bar · value), final model in accent, others grey.

## Breakpoints
≥ 1100 two-column grids; 760–1099 single column with sidebar; < 760 sidebar becomes a drawer, cards stack, tables scroll inside their card.

## Implementation (React app, src/app/web)
Tokens live as Tailwind 4 theme variables in `src/app/web/src/index.css` (`bg`, `raised`, `surface`, `surface-2/3`, `line`, `line-strong`, `fg`, `fg-2/3`, `accent`, `observed`, `danger`, `g1–g3`). Primitives are in `src/components/ui`, charts in `src/components/charts`. Stats sit in one bordered strip with 1px dividers rather than separate cards. Motion: one entrance on the overview headline; charts do not animate (Recharts ignores reduced-motion settings, and drawing must not depend on animation frames).
