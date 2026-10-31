"""Read-only verification of the Phase 8 geo200 selection and checkpoint (stage S2).

Run from the project root (only after the S2 stage has produced data/processed/geo200/):
    .venv\\Scripts\\python.exe scripts/verify_geo200.py

Reuses scripts/verify_artifacts.py checks (raw hashes, inputs, preprocessed artifacts) and adds the
selection checks of the Phase 8 plan (section 4.1/4.2), plus the report-only near-duplicate screen
(D8-11). Writes only data/processed/geo200/verification_results.json.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import verify_artifacts as va  # noqa: E402
from aiori import config, select_geo200 as sel, splits  # noqa: E402

GEO = config.PROCESSED_DIR / "geo200"


def check_selection(r, node_order, cfg, out_dir, root):
    s = "selection"
    manifest = pd.read_csv(out_dir / "sensor_selection.csv", keep_default_na=False, na_values=[""])
    summary = json.loads((out_dir / "selection_summary.json").read_text(encoding="utf-8"))
    per = cfg["per_client"]
    counts = manifest["proposed_client"].value_counts()
    r.check(s, f"{4 * per} sensors, {per} per client", len(manifest) == 4 * per and (counts == per).all() and len(counts) == 4,
            counts.sort_index().tolist())
    r.check(s, "array_index unique", manifest["array_index"].is_unique)
    r.check(s, "all eligible, no exact-copy series", manifest["eligible"].astype(bool).all()
            and not manifest["duplicate_series"].astype(bool).any())
    r.check(s, "client == CLIENT_BY_DISTRICT[District]",
            (manifest["District"].map(config.CLIENT_BY_DISTRICT) == manifest["proposed_client"]).all())
    r.check(s, "station_id == node_order[array_index] (consistency only, not verification)",
            np.array_equal(node_order[manifest["array_index"]], manifest["station_id"].to_numpy()))
    paths = cfg["inputs"]
    audit, meta, exclude = sel.load_tables(root / paths["audit"], root / paths["metadata"],
                                           [root / p for p in paths["pilot_manifests"]])
    r.check(s, "no overlap with the pilot sensors", not manifest["array_index"].isin(exclude).any(), len(exclude))
    r.check(s, "input SHA-256 == approved", all(sel.sha256_file(root / n) == h for n, h in cfg["input_sha256"].items()))
    again, _ = sel.stratified_selection(sel.selection_frame(audit, meta), exclude, per)
    r.check(s, "selection reproduces exactly from the frozen inputs",
            again["array_index"].tolist() == manifest["array_index"].tolist()
            and again["selection_basis"].tolist() == manifest["selection_basis"].tolist())
    floors = {(c, d) for c, info in summary["clients"].items() for d, n in info["district_candidates"].items()}
    picked = {(c, str(int(d))) for c, d in manifest[["proposed_client", "District"]].itertuples(index=False)}
    r.check(s, "every District with candidates has >= 1 selected sensor", floors <= picked, sorted(floors - picked))
    r.info(s, "fallbacks", summary["fallbacks"] or "none")
    r.info(s, "type groups selected", manifest["type_group"].value_counts().sort_index().to_dict())
    return manifest


def check_checkpoint(r, out_dir):
    s = "checkpoint"
    primary = json.loads((config.PROCESSED_DIR / "split_config.json").read_text(encoding="utf-8"))
    geo = json.loads((out_dir / "split_config.json").read_text(encoding="utf-8"))
    r.check(s, "split_config == primary run", geo == primary)
    record = json.loads((out_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    r.check(s, "artifact SHA-256 == checkpoint_manifest",
            all(sel.sha256_file(out_dir / n) == h for n, h in record["artifact_sha256"].items()))
    r.check(s, "raw fingerprints unchanged during S2", record["raw_fingerprints_before"] == record["raw_fingerprints_after"])
    r.check(s, "only 2023_p01.npy opened", record["raw_arrays_opened"] == ["2023_p01.npy"], record["raw_arrays_opened"])


def near_duplicates(r, out_dir):
    with np.load(out_dir / "pilot_preprocessed.npz") as d:
        flow = d["flow_raw"]
    a, b = splits.load_split_config(out_dir / "split_config.json")["train"]
    pairs = sel.near_duplicate_pairs(flow[:, a:b])
    r.info("near-duplicates (D8-11, report only)", "pairs with training-period r >= 0.999",
           {"n_pairs": len(pairs), "sensor_positions": [p[:2] for p in pairs]})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path, default=ROOT / "experiments" / "configs" / "phase8_selection.json")
    p.add_argument("--out-dir", type=Path, default=GEO)
    args = p.parse_args(argv)
    cfg = sel.load_config(args.config)

    r = va.Results()
    va.check_raw(r)
    arr, node_order = va.check_inputs(r)
    manifest_path = args.out_dir / "sensor_selection.csv"
    check_selection(r, node_order, cfg, args.out_dir, ROOT)
    va.check_preprocessed(r, arr, "geo200", args.out_dir, manifest_path)
    check_checkpoint(r, args.out_dir)
    near_duplicates(r, args.out_dir)

    n_pass = sum(x["status"] == "PASS" for x in r.rows)
    out = args.out_dir / "verification_results.json"
    out.write_text(json.dumps(dict(run_at_utc=f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}", passed=n_pass,
                                   failed=len(r.failed), results=r.rows), indent=2, default=str), encoding="utf-8")
    print(f"\n{n_pass} passed, {len(r.failed)} failed. Results: {out}")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
