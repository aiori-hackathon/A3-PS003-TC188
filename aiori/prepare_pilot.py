"""Phases 3-5 for the pilot: chronological split, leak-safe preprocessing, window index.

Run from the project root (after aiori.flow_audit):
    .venv\\Scripts\\python.exe -m aiori.prepare_pilot

Reads the raw array read-only and the pilot manifest; writes only to data/processed/:
  split_config.json, scaler_parameters.json, pilot_preprocessed.npz, window_index.npz,
  preprocessing_report.md
Windows are stored as (sensor_position, t0) indices and materialised on demand with
aiori.windows.gather_windows; no training happens here.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from aiori import config, data_io, preprocessing, splits, windows

log = logging.getLogger(__name__)

SCALER_RATIONALE = (
    "per_sensor: each client can fit its own sensors' parameters locally without pooling data "
    "across clients, so all four arms share an identical, federation-compatible preprocessing; "
    "pilot sensor flow levels also differ by more than an order of magnitude."
)


def load_selection(path, node_order, min_sensors, max_sensors):
    sel = pd.read_csv(path)
    for col in ("array_index", "station_id"):
        if col not in sel.columns:
            raise ValueError(f"Selection manifest lacks column {col!r}")
    if not min_sensors <= len(sel) <= max_sensors:
        raise ValueError(f"Selection has {len(sel)} sensors; expected {min_sensors}-{max_sensors}")
    if not sel["array_index"].is_unique or not sel["station_id"].is_unique:
        raise ValueError("Selection manifest has duplicate array_index or station_id")
    if "eligible" in sel.columns and not sel["eligible"].astype(bool).all():
        raise ValueError("Selection manifest contains ineligible sensors")
    idx = sel["array_index"].to_numpy()
    if idx.min() < 0 or idx.max() >= len(node_order):
        raise ValueError("Selection array_index out of range")
    # Consistency check only: confirms the manifest used node_order, not that the
    # public array row i is truly sensor node_order[i] (supported but unconfirmed).
    if not np.array_equal(node_order[idx], sel["station_id"].to_numpy()):
        raise ValueError("Manifest station_id disagrees with node_order[array_index]")
    return sel


def _weekdays(first, last):
    days = pd.date_range(first[:10], last[:10], freq="D")
    return f"{days[0]:%a %d %b} - {days[-1]:%a %d %b} ({len(days)} days)"


def write_report(path, args, sel, split_cfg, scaler, observed, imputed, idx, n_targets_obs):
    s = split_cfg["boundaries"]
    lines = [
        f"# Pilot preprocessing report (phases 3-5) — {args.run_label}",
        "",
        f"Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `aiori/prepare_pilot.py`",
        f"Sensors: {len(sel)} from `{args.selection}`; channel {config.FLOW} (flow, working convention)",
        f"Source array: `{args.array}` (memory-mapped, read-only; size and mtime verified unchanged)",
        "",
        "## Chronological split",
        "",
        "| Split | Index range [start, end) | Timestamps (notebook convention) | Days |",
        "|---|---|---|---|",
    ]
    for k in splits.SPLIT_NAMES:
        ts = split_cfg["timestamps"][k]
        lines.append(f"| {k} | {s[k][0]}-{s[k][1]} | {ts['first']} to {ts['last']} | {_weekdays(ts['first'], ts['last'])} |")
    lines += [
        "",
        f"Window rule: {args.input_len} input steps -> {args.horizon} target steps; a window belongs to the split "
        "containing **all** of its targets; inputs may use earlier steps (past context only). Windows "
        "with zero observed targets are dropped; partially observed targets are kept and masked.",
        "",
        "## Normalisation",
        "",
        f"- Mode: `{scaler['mode']}` z-score fitted on observed **training-split** values only "
        f"(steps [{scaler['fit_range'][0]}, {scaler['fit_range'][1]})).",
        f"- Rationale: {SCALER_RATIONALE}" if scaler["mode"] == "per_sensor" else "- Mode chosen via --scaler-mode.",
        f"- Mean range {min(scaler['mean']):.1f}-{max(scaler['mean']):.1f}; std range "
        f"{min(scaler['std']):.1f}-{max(scaler['std']):.1f} (vehicles / 5 min, unit unconfirmed).",
        "",
        "## Missing values",
        "",
        "Inputs: causal fill = forward fill up to "
        f"{args.max_ffill} steps within the same split, then training time-of-day median, then training mean; an "
        "observed/imputed flag is kept as an input feature. Targets: never filled; NaN preserved and masked.",
        "",
        "| Split | Cells | Missing (imputed for inputs) | Windows | Observed target cells | Windows with any missing target |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in splits.SPLIT_NAMES:
        a, b = s[k]
        lines.append(
            f"| {k} | {observed[:, a:b].size:,} | {int(imputed[:, a:b].sum()):,} | {len(idx[k]):,} | "
            f"{n_targets_obs[k][0]:,} | {n_targets_obs[k][1]:,} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- **Row-to-ID mapping is supported but unconfirmed:** station IDs come from `node_order.npy[array_index]`; "
        "matching lengths and ID sets do not prove the public array's row order.",
        "- Sensors come from the selection manifest above, after the audit's exact-duplicate exclusion "
        "(counts in `flow_quality_report.md`; context in `docs/data_quality_notes.md`). Near-duplicates are "
        "not detected.",
        "- January 2023 only: three validation days and six test days support a pipeline pilot, not seasonal claims.",
        "- No time-of-day/day-of-week features yet; inputs are [scaled flow, observed flag].",
        "",
        "## Files",
        "",
    ]
    lines += [f"- `{Path(args.out_dir) / f}`" for f in (
        "split_config.json", "scaler_parameters.json", "pilot_preprocessed.npz", "window_index.npz",
        "preprocessing_report.md")]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--array", type=Path, default=config.ARRAY_PATH)
    p.add_argument("--node-order", type=Path, default=config.NODE_ORDER_PATH)
    p.add_argument("--selection", type=Path, default=config.PROCESSED_DIR / "sensor_selection.csv")
    p.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--scaler-mode", choices=("per_sensor", "global"), default="per_sensor")
    p.add_argument("--max-ffill", type=int, default=6, help="max forward-fill steps (6 = 30 min)")
    p.add_argument("--input-len", type=int, default=windows.INPUT_LEN)
    p.add_argument("--horizon", type=int, default=windows.HORIZON)
    p.add_argument("--fractions", type=float, nargs=3, default=splits.DEFAULT_FRACTIONS)
    p.add_argument("--steps-per-day", type=int, default=config.STEPS_PER_DAY)
    p.add_argument("--min-sensors", type=int, default=20)
    p.add_argument("--max-sensors", type=int, default=50)
    p.add_argument("--no-shape-check", action="store_true")
    p.add_argument("--run-label", default="primary pilot", help="label written into the report title")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw_paths = [args.array, args.node_order]
    out_dir = data_io.check_output_dir(args.out_dir, raw_paths)
    before = {p: data_io.file_fingerprint(p) for p in raw_paths if Path(p).is_file()}

    array = data_io.load_traffic_array(args.array, None if args.no_shape_check else config.EXPECTED_P01_SHAPE)
    node_order = data_io.load_node_order(args.node_order, n_sensors=array.shape[0])
    sel = load_selection(args.selection, node_order, args.min_sensors, args.max_sensors)
    rows = sel["array_index"].to_numpy()
    flow = np.asarray(array[rows][:, :, config.FLOW], dtype=np.float64)   # (S, T); NaN = missing
    del array

    n_steps = flow.shape[1]
    split = splits.make_splits(n_steps, tuple(args.fractions), args.steps_per_day)
    split_cfg = splits.split_config(split, n_steps, tuple(args.fractions), args.steps_per_day)

    observed = preprocessing.observed_mask(flow)
    scaler = preprocessing.fit_scaler(flow, split, mode=args.scaler_mode)
    tod = preprocessing.fit_tod_median(flow, split, args.steps_per_day)
    filled, imputed = preprocessing.causal_fill(flow, split, tod, args.max_ffill, args.steps_per_day)
    inputs = preprocessing.transform(filled, scaler)
    targets = preprocessing.transform(flow, scaler)          # NaN kept: targets are never filled

    idx, n_targets_obs = {}, {}
    for name in splits.SPLIT_NAMES:
        idx[name] = windows.window_index(observed, split, name, args.input_len, args.horizon)
        start, end = split[name]
        t0 = idx[name][:, 1]
        if len(t0) and ((t0 < start).any() or (t0 + args.horizon > end).any()):
            raise AssertionError(f"{name} window targets cross a split boundary")
        _, _, M = windows.gather_windows(inputs, imputed, targets, observed, idx[name],
                                         args.input_len, args.horizon)
        n_targets_obs[name] = (int(M.sum()), int((~M).any(axis=1).sum()))
        log.info("%s: %d windows, %d observed target cells", name, len(t0), n_targets_obs[name][0])

    out_dir.mkdir(parents=True, exist_ok=True)
    splits.save_split_config(out_dir / "split_config.json", split_cfg)
    scaler_out = dict(scaler, channel=config.FLOW, array_index=rows.tolist(),
                      station_id=sel["station_id"].tolist(), id_mapping_status=config.ID_MAPPING_STATUS,
                      rationale=SCALER_RATIONALE if args.scaler_mode == "per_sensor" else "")
    (out_dir / "scaler_parameters.json").write_text(json.dumps(scaler_out, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "pilot_preprocessed.npz",
        array_index=rows, station_id=sel["station_id"].to_numpy(),
        flow_raw=flow.astype(np.float32), observed_mask=observed,
        inputs_scaled=inputs.astype(np.float32), input_imputed=imputed,
        targets_scaled=targets.astype(np.float32), tod_median=tod,
    )
    np.savez_compressed(out_dir / "window_index.npz", **idx)
    write_report(out_dir / "preprocessing_report.md", args, sel, split_cfg, scaler, observed, imputed,
                 idx, n_targets_obs)

    after = {p: data_io.file_fingerprint(p) for p in before}
    if after != before:
        raise AssertionError("Raw input files changed during preprocessing")
    log.info("Wrote preprocessing outputs to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
