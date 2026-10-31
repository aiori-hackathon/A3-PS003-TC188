"""Channel 0 (flow) per-sensor quality audit and deterministic pilot-sensor selection.

Run from the project root:
    .venv\\Scripts\\python.exe -m aiori.flow_audit

Reads the raw array read-only (memory-mapped, in sensor batches) and writes only to
data/processed/. Primary eligibility uses the training period only; whole-month
statistics are kept as informational `month_*` columns. A separate missing-data
validation selection (never mixed into the primary pilot) is also written.
Station IDs, Districts and clients are attached via node_order[i], which is
supported but unconfirmed as the public array's row order.
"""

import argparse
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from aiori import config, data_io, splits

log = logging.getLogger(__name__)

DEFAULT_MIN_COVERAGE_PCT = 99.0
DEFAULT_MAX_ZERO_PCT = 50.0
DEFAULT_PER_CLIENT = 10
DEFAULT_SALT = "ps003-pilot-v1"
DEFAULT_VALIDATION_PER_CLIENT = 2
VALIDATION_SALT = "ps003-missing-validation-v1"

# Reference figures for 2023_p01 Channel 0 from the project handoff (section 5).
HANDOFF_P01_FLOW = {
    "nan_cells": 26_459_359, "zero_cells": 3_627_470, "fully_observed": 10_613,
    "partially_observed": 3_793, "fully_missing": 2_566, "constant": 11,
}


def _longest_true_run(mask):
    """Longest run of consecutive True values in each row of a 2-D boolean array."""
    run = np.zeros(mask.shape[0], dtype=np.int64)
    best = np.zeros(mask.shape[0], dtype=np.int64)
    for t in range(mask.shape[1]):
        run = (run + 1) * mask[:, t]
        np.maximum(best, run, out=best)
    return best


def audit_flow(array, channel=config.FLOW, batch_size=512):
    """Per-sensor statistics for one channel of a (sensor, time, channel) array.

    Missing = non-finite (NaN, and inf if any). Zero is a valid observation.
    """
    n_sensors, n_steps, _ = array.shape
    parts = []
    for start in range(0, n_sensors, batch_size):
        stop = min(start + batch_size, n_sensors)
        block = np.asarray(array[start:stop, :, channel], dtype=np.float64)
        finite = np.isfinite(block)
        count = finite.sum(axis=1)
        has = count > 0
        safe = np.where(has, count, 1)
        mean = np.where(finite, block, 0.0).sum(axis=1) / safe
        var = (np.where(finite, block - mean[:, None], 0.0) ** 2).sum(axis=1) / safe
        lo = np.where(finite, block, np.inf).min(axis=1)
        hi = np.where(finite, block, -np.inf).max(axis=1)
        zeros = (finite & (block == 0)).sum(axis=1)
        raw = np.ascontiguousarray(array[start:stop, :, channel])
        hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in raw]
        nan_if_empty = lambda v: np.where(has, v, np.nan)  # noqa: E731
        parts.append(pd.DataFrame({
            "array_index": np.arange(start, stop),
            "n_steps": n_steps,
            "finite_count": count,
            "missing_count": n_steps - count,
            "inf_count": np.isinf(block).sum(axis=1),
            "zero_count": zeros,
            "zero_pct": nan_if_empty(100.0 * zeros / safe),
            "mean": nan_if_empty(mean),
            "std": nan_if_empty(np.sqrt(var)),
            "min": nan_if_empty(lo),
            "max": nan_if_empty(hi),
            "max_gap_steps": _longest_true_run(~finite),
            "fully_missing": ~has,
            "constant": has & (hi == lo),
            "series_sha256": hashes,
        }))
        log.debug("audited sensors %d-%d", start, stop - 1)
    stats = pd.concat(parts, ignore_index=True)
    # Byte-identical observed series across different sensors are almost certainly
    # upstream copies/fills, not independent measurements.
    group_size = stats.groupby("series_sha256")["array_index"].transform("size")
    stats["duplicate_group_size"] = np.where(stats["fully_missing"], 1, group_size)
    stats["duplicate_series"] = stats["duplicate_group_size"] > 1
    stats.insert(4, "missing_pct", 100.0 * stats["missing_count"] / n_steps)
    stats.insert(5, "coverage_pct", 100.0 - stats["missing_pct"])
    return stats


def assign_eligibility(stats, min_coverage_pct, max_zero_pct):
    """Add `eligible` and the first failing `exclusion_reason` ('' when eligible).

    Uses the unprefixed statistic columns, which main() computes on the training
    period only. `duplicate_series` means "excluded as a precaution", not "invalid".
    """
    out = stats.copy()
    reason = np.select(
        [out["fully_missing"], out["constant"], out["duplicate_series"],
         out["coverage_pct"] < min_coverage_pct, out["zero_pct"] > max_zero_pct],
        ["fully_missing", "constant", "duplicate_series", "low_coverage", "high_zero_fraction"],
        default="",
    )
    out["exclusion_reason"] = reason
    out["eligible"] = reason == ""
    return out


def _tie_key(array_index, salt):
    return hashlib.sha256(f"{salt}:{int(array_index)}".encode()).hexdigest()


def select_pilot(audit, per_group, group_col="proposed_client", salt=DEFAULT_SALT, floor_col=None):
    """Pick `per_group` eligible sensors from every group.

    Rank: within each group, sort eligible sensors by missing_count ascending, then by
    sha256(f"{salt}:{array_index}") as a reproducible, version-independent tie-break.
    Floors: if `floor_col` is given (e.g. "District"), every value of it present in the
    group first receives its best-ranked eligible sensor; the remaining slots are filled
    in rank order. Forecasting performance is never used.
    """
    groups = sorted(audit[group_col].unique())
    eligible = audit[audit["eligible"]].copy()
    short = {g: int((eligible[group_col] == g).sum()) for g in groups
             if (eligible[group_col] == g).sum() < per_group}
    if short:
        raise ValueError(f"Too few eligible sensors for {per_group} per group: {short}")
    eligible["_tie"] = eligible["array_index"].map(lambda i: _tie_key(i, salt))
    eligible = eligible.sort_values([group_col, "missing_count", "_tie"])

    parts = []
    for g in groups:
        ranked = eligible[eligible[group_col] == g]
        floor_idx = []
        if floor_col is not None:
            values = sorted(audit.loc[audit[group_col] == g, floor_col].unique())
            empty = [v for v in values if not (ranked[floor_col] == v).any()]
            if empty:
                raise ValueError(f"No eligible sensors for {floor_col} {empty} in group {g}")
            if len(values) > per_group:
                raise ValueError(f"Group {g} has {len(values)} {floor_col} values but only {per_group} slots")
            floor_idx = [ranked.index[ranked[floor_col] == v][0] for v in values]
        rest = [i for i in ranked.index if i not in floor_idx][: per_group - len(floor_idx)]
        chosen = ranked[ranked.index.isin(floor_idx + rest)].copy()
        chosen["selection_basis"] = np.where(chosen.index.isin(floor_idx), f"{floor_col}_floor", "rank")
        parts.append(chosen)
    chosen = pd.concat(parts)
    chosen["selection_rank"] = chosen.groupby(group_col).cumcount() + 1
    return chosen.drop(columns="_tie").reset_index(drop=True)


def select_missing_validation(audit, per_group, coverage_range=(99.0, 100.0), exclude=(),
                              group_col="proposed_client", salt=VALIDATION_SALT):
    """Separate diagnostic set of naturally incomplete sensors (never the primary pilot).

    Candidates pass the primary (training-period) quality rules, have whole-month coverage
    in [lo, hi) and at least one missing step in the validation/test period
    (`eval_missing_count`). Uses whole-month data by design: its only purpose is to
    exercise fill and target-mask code on real gaps. Ordered by a salted sha256 tie-break.
    """
    lo, hi = coverage_range
    cand = audit[
        audit["eligible"]
        & (audit["month_coverage_pct"] >= lo) & (audit["month_coverage_pct"] < hi)
        & (audit["eval_missing_count"] >= 1)
        & ~audit["array_index"].isin(list(exclude))
    ].copy()
    groups = sorted(audit[group_col].unique())
    short = {g: int((cand[group_col] == g).sum()) for g in groups if (cand[group_col] == g).sum() < per_group}
    if short:
        raise ValueError(f"Too few missing-data validation candidates for {per_group} per group: {short}")
    cand["_tie"] = cand["array_index"].map(lambda i: _tie_key(i, salt))
    chosen = cand.sort_values([group_col, "_tie"]).groupby(group_col).head(per_group).copy()
    chosen["selection_rank"] = chosen.groupby(group_col).cumcount() + 1
    chosen["selection_basis"] = "missing_validation"
    return chosen.drop(columns="_tie").reset_index(drop=True)


def attach_metadata(stats, aligned):
    """Join provisional station ID, District, County and geographic client by array row."""
    meta = aligned[["station_id", "District", "County"]].copy()
    meta["District"] = pd.to_numeric(meta["District"], errors="coerce")
    meta["proposed_client"] = meta["District"].map(config.CLIENT_BY_DISTRICT)
    if meta["proposed_client"].isna().any():
        bad = sorted(meta.loc[meta["proposed_client"].isna(), "District"].unique().tolist())
        raise ValueError(f"Districts without a client mapping: {bad}")
    meta.insert(0, "array_index", np.arange(len(meta)))
    meta.insert(2, "id_mapping_status", config.ID_MAPPING_STATUS)
    out = meta.merge(stats, on="array_index", how="inner", validate="one_to_one")
    if len(out) != len(stats) or not out["station_id"].is_unique:
        raise ValueError("Metadata join changed sensor count or produced duplicate IDs")
    return out


def build_audit(array, aligned, train_range, min_coverage, max_zero_pct, batch_size=512):
    """Training-period statistics (unprefixed, used for eligibility) plus whole-month
    statistics (`month_*`, informational only)."""
    a, b = train_range
    train = audit_flow(array[:, a:b, :], channel=config.FLOW, batch_size=batch_size)
    month = audit_flow(array, channel=config.FLOW, batch_size=batch_size)
    month = month.drop(columns="array_index").add_prefix("month_").assign(array_index=month["array_index"])
    audit = attach_metadata(train, aligned).merge(month, on="array_index", how="inner", validate="one_to_one")
    audit.insert(3, "stats_period", f"train steps [{a}, {b})")
    audit["eval_missing_count"] = audit["month_missing_count"] - audit["missing_count"]
    return assign_eligibility(audit, min_coverage, max_zero_pct)


def _summary(audit, selected, validation, array_shape):
    n_sensors, n_steps, _ = array_shape
    m = lambda c: audit[f"month_{c}"]  # noqa: E731
    dups = audit[audit["duplicate_series"]]
    mdups = audit[m("duplicate_series")]
    reasons = audit["exclusion_reason"]
    funnel, remaining = [], n_sensors
    for r in ("fully_missing", "constant", "duplicate_series", "low_coverage", "high_zero_fraction"):
        removed = int((reasons == r).sum())
        remaining -= removed
        funnel.append((r, removed, remaining))
    return {
        "sensors": n_sensors, "steps": n_steps, "cells": n_sensors * n_steps,
        "month_nan_cells": int(m("missing_count").sum() - m("inf_count").sum()),
        "month_inf_cells": int(m("inf_count").sum()),
        "month_zero_cells": int(m("zero_count").sum()),
        "month_finite_cells": int(m("finite_count").sum()),
        "month_fully_observed": int((m("missing_count") == 0).sum()),
        "month_partially_observed": int(((m("missing_count") > 0) & ~m("fully_missing")).sum()),
        "month_fully_missing": int(m("fully_missing").sum()),
        "month_constant": int(m("constant").sum()),
        "month_duplicate_series": len(mdups),
        "month_duplicate_groups": int(mdups["month_series_sha256"].nunique()),
        "duplicate_series": len(dups),
        "duplicate_groups": int(dups["series_sha256"].nunique()),
        "largest_duplicate_group": int(dups["duplicate_group_size"].max()) if len(dups) else 0,
        "duplicate_groups_multi_district": int((dups.groupby("series_sha256")["District"].nunique() > 1).sum()),
        "duplicate_train_only": int((audit["duplicate_series"] & ~m("duplicate_series")).sum()),
        "funnel": funnel,
        "eligible": int(audit["eligible"].sum()),
        "selected": len(selected),
        "validation": len(validation),
    }


def write_report(path, audit, selected, validation, summary, args, array_shape, train_range, outputs):
    s = summary
    a, b = train_range
    pct = lambda x, y: f"{100.0 * x / y:.4f}%" if y else "n/a"  # noqa: E731
    lines = [
        "# Channel 0 (flow) quality audit and pilot sensor selection",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `aiori/flow_audit.py`",
        f"Source array: `{args.array}` (shape {tuple(array_shape)}, read memory-mapped; not modified)",
        "",
        "## Eligibility period",
        "",
        f"All primary selection statistics use the **training period only**: steps [{a}, {b}) of the "
        "chronological split (see `split_config.json`). Validation/test completeness, zero share and "
        "values do not influence the primary pilot. Whole-month statistics are kept as `month_*` columns "
        "for information and for the handoff comparison only.",
        "",
        "## Primary eligibility rules (design choice)",
        "",
        "Applied in order to the training-period Channel 0 series; the first failing rule is `exclusion_reason`:",
        "",
        "1. not fully missing,",
        "2. not constant (min == max over observed values),",
        "3. **exact-duplicate rule (separate):** training-period series not byte-identical (SHA-256 of the "
        "float32 bytes, NaN pattern included) to any other sensor's. All members of a duplicate group are "
        "excluded *as a precaution*; this does not establish that every member is invalid,",
        f"4. training-period coverage >= {args.min_coverage}%,",
        f"5. zero readings <= {args.max_zero_pct}% of observed training values (zeros remain valid observations).",
        "",
        "## Selection rule",
        "",
        f"Per provisional client, {args.per_client} sensors. **District floor:** each District in the client first "
        "gets its best-ranked eligible sensor; remaining slots follow the rank order. Rank = training "
        f"`missing_count` ascending, then `sha256(\"{args.salt}:<array_index>\")`. Deterministic; no forecasting "
        "results used.",
        "",
        "## Whole-month summary (informational; compared with the handoff)",
        "",
        "| Measure | This audit | Handoff reference |",
        "|---|---:|---:|",
    ]
    ref = HANDOFF_P01_FLOW if tuple(array_shape) == config.EXPECTED_P01_SHAPE else {}
    rows = [
        ("Sensors", f"{s['sensors']:,}", ""),
        ("Time steps", f"{s['steps']:,}", ""),
        ("NaN cells", f"{s['month_nan_cells']:,} ({pct(s['month_nan_cells'], s['cells'])})", ref.get("nan_cells")),
        ("Inf cells", f"{s['month_inf_cells']:,}", "0" if ref else ""),
        ("Zero readings (of finite)", f"{s['month_zero_cells']:,} ({pct(s['month_zero_cells'], s['month_finite_cells'])})",
         ref.get("zero_cells")),
        ("Fully observed sensors", f"{s['month_fully_observed']:,}", ref.get("fully_observed")),
        ("Partially observed sensors", f"{s['month_partially_observed']:,}", ref.get("partially_observed")),
        ("Fully missing sensors", f"{s['month_fully_missing']:,}", ref.get("fully_missing")),
        ("Constant sensors", f"{s['month_constant']:,}", ref.get("constant")),
        ("Exact-duplicate sensors (whole month)",
         f"{s['month_duplicate_series']:,} in {s['month_duplicate_groups']:,} groups", "not reported"),
    ]
    for name, mine, theirs in rows:
        theirs = f"{theirs:,}" if isinstance(theirs, int) else (theirs or "")
        lines.append(f"| {name} | {mine} | {theirs} |")

    lines += ["", "## Training-period eligibility funnel", "", "| Rule | Removed | Remaining |", "|---|---:|---:|",
              f"| all sensors | | {s['sensors']:,} |"]
    lines += [f"| {r} | {n:,} | {left:,} |" for r, n, left in s["funnel"]]
    lines += ["",
              f"Exact duplicates on the training period: {s['duplicate_series']:,} sensors in {s['duplicate_groups']:,} "
              f"groups (largest {s['largest_duplicate_group']:,}; groups spanning >1 District: "
              f"{s['duplicate_groups_multi_district']}). {s['duplicate_train_only']:,} of them are identical during "
              "training but diverge later, so they are not whole-month duplicates.",
              ""]

    def client_district_table(df, title):
        t = df.groupby(["proposed_client", "District"]).size()
        out = ["", f"### {title}", "", "| Client | District | Sensors |", "|---|---:|---:|"]
        out += [f"| {c} | {d} | {n} |" for (c, d), n in t.items()]
        return out

    lines += client_district_table(audit[audit["eligible"]], "Eligible sensors by client and District")
    lines += client_district_table(selected, f"Primary pilot ({len(selected)} sensors) by client and District")
    lines += ["",
              f"- Training coverage: min {selected['coverage_pct'].min():.3f}%; whole-month coverage: min "
              f"{selected['month_coverage_pct'].min():.3f}% (informational)",
              f"- Selected via District floor: {int((selected['selection_basis'] == 'District_floor').sum())}; via rank: "
              f"{int((selected['selection_basis'] == 'rank').sum())}",
              f"- Zero readings (training): {selected['zero_pct'].min():.2f}%-{selected['zero_pct'].max():.2f}% of observed",
              ""]
    if len(validation):
        lines += ["## Missing-data validation selection (separate; not part of the primary pilot)", "",
                  f"{len(validation)} sensors, {args.validation_per_client} per client: pass the primary rules, whole-January "
                  f"coverage in [{args.validation_coverage[0]}%, {args.validation_coverage[1]}%), at least one missing step in "
                  "validation/test, not in the primary pilot. Uses whole-month coverage by design, only to exercise "
                  "real-data input fill and target masking.", "",
                  "| array_index | Provisional station_id | Client | District | Train missing | Val/test missing | Month coverage |",
                  "|---:|---:|---|---:|---:|---:|---:|"]
        lines += [f"| {r.array_index} | {r.station_id} | {r.proposed_client} | {r.District} | {r.missing_count} | "
                  f"{r.eval_missing_count} | {r.month_coverage_pct:.3f}% |" for r in validation.itertuples()]
        lines.append("")
    lines += ["## Limitations and caveats", "",
              "- **Row-to-ID mapping is supported but unconfirmed.** `station_id`, `District`, `County` and "
              "`proposed_client` are attached via `node_order.npy[array_index]`. Geographic attribution is provisional.",
              "- Channel 0 = flow is the upstream-notebook working convention; units are not independently confirmed.",
              "- Exact-duplicate exclusion is precautionary. Near-duplicates and non-identical imputed series are not "
              "detected. A genuine sensor inside a duplicate group would be excluded too.",
              "- One month only. Eligibility thresholds are design choices for a pilot, not ground truth.",
              "", "## Files", ""]
    lines += [f"- `{p}`" for p in outputs]
    lines += ["", "## Reproduce", "", "```bash", ".venv/Scripts/python.exe -m aiori.flow_audit", "```", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--array", type=Path, default=config.ARRAY_PATH)
    p.add_argument("--node-order", type=Path, default=config.NODE_ORDER_PATH)
    p.add_argument("--metadata", type=Path, default=config.METADATA_PATH)
    p.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--per-client", type=int, default=DEFAULT_PER_CLIENT)
    p.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE_PCT)
    p.add_argument("--max-zero-pct", type=float, default=DEFAULT_MAX_ZERO_PCT)
    p.add_argument("--salt", default=DEFAULT_SALT)
    p.add_argument("--validation-per-client", type=int, default=DEFAULT_VALIDATION_PER_CLIENT,
                   help="missing-data validation sensors per client (0 = skip)")
    p.add_argument("--validation-coverage", type=float, nargs=2, default=(99.0, 100.0))
    p.add_argument("--steps-per-day", type=int, default=config.STEPS_PER_DAY)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--no-shape-check", action="store_true",
                   help="skip the expected 2023_p01 shape check (tests / other months)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw_paths = [args.array, args.node_order, args.metadata]
    out_dir = data_io.check_output_dir(args.out_dir, raw_paths)
    before = {p: data_io.file_fingerprint(p) for p in raw_paths if Path(p).is_file()}

    expected = None if args.no_shape_check else config.EXPECTED_P01_SHAPE
    array = data_io.load_traffic_array(args.array, expected_shape=expected)
    node_order = data_io.load_node_order(args.node_order, n_sensors=array.shape[0])
    aligned = data_io.align_metadata(data_io.load_metadata(args.metadata), node_order)
    train_range = splits.make_splits(array.shape[1], steps_per_day=args.steps_per_day)["train"]

    log.info("Auditing channel %d for %d sensors (eligibility on train steps %s) ...",
             config.FLOW, array.shape[0], train_range)
    audit = build_audit(array, aligned, train_range, args.min_coverage, args.max_zero_pct, args.batch_size)
    selected = select_pilot(audit, per_group=args.per_client, salt=args.salt, floor_col="District")
    selected["selection_reason"] = (
        "eligible (train period); " + selected["selection_basis"] + "; rank " + selected["selection_rank"].astype(str)
        + " in " + selected["proposed_client"]
    )
    if not selected["station_id"].is_unique or not selected["array_index"].is_unique:
        raise AssertionError("Duplicate sensors in pilot selection")
    validation = audit.iloc[0:0]
    if args.validation_per_client > 0:
        validation = select_missing_validation(audit, args.validation_per_client, tuple(args.validation_coverage),
                                               exclude=selected["array_index"])

    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "flow_sensor_audit.csv"
    selection_path = out_dir / "sensor_selection.csv"
    validation_path = out_dir / "missing_validation_selection.csv"
    report_path = out_dir / "flow_quality_report.md"
    audit.to_csv(audit_path, index=False)
    selected.to_csv(selection_path, index=False)
    outputs = [audit_path, selection_path, report_path]
    if len(validation):
        validation.to_csv(validation_path, index=False)
        outputs.insert(2, validation_path)
    summary = _summary(audit, selected, validation, array.shape)
    write_report(report_path, audit, selected, validation, summary, args, array.shape, train_range, outputs)

    del array  # release the memmap before re-checking raw files
    after = {p: data_io.file_fingerprint(p) for p in before}
    if after != before:
        raise AssertionError(f"Raw input files changed during the audit: {before} -> {after}")

    log.info("Summary: %s", {k: v for k, v in summary.items() if k != "funnel"})
    log.info("Wrote %s", ", ".join(map(str, outputs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
