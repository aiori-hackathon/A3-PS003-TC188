"""Phase 8 stage S2: geo200 selection, then the unchanged prepare_pilot into its own folder.

Run from the project root (only after gate G1):
    .venv\\Scripts\\python.exe scripts/prepare_geo200.py

Order: refuse a non-empty output folder -> fingerprint the raw files (size, mtime) -> check the
selection inputs against the approved SHA-256s -> select (training-period columns only) -> write the
manifest -> aiori.prepare_pilot.main(...) unchanged -> check that the only raw array opened was
2023_p01.npy and that no raw file changed -> write checkpoint_manifest.json.
Outputs (git-ignored, local only): sensor_selection.csv, selection_summary.json, selection_report.md,
the prepare_pilot artifacts and checkpoint_manifest.json.
"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aiori import config, data_io, prepare_pilot, select_geo200 as sel, windows  # noqa: E402

RAW_ARRAY_NAME = "2023_p01.npy"


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path, default=ROOT / "experiments" / "configs" / "phase8_selection.json")
    p.add_argument("--root", type=Path, default=ROOT, help="base for the relative paths in the config")
    p.add_argument("--array", type=Path, default=config.ARRAY_PATH)
    p.add_argument("--node-order", type=Path, default=config.NODE_ORDER_PATH)
    p.add_argument("--raw-metadata", type=Path, default=config.METADATA_PATH, help="fingerprinted only, never read")
    p.add_argument("--out-dir", type=Path, default=None, help="default: the config's out_dir")
    p.add_argument("--no-shape-check", action="store_true", help="synthetic tests only")
    return p.parse_args(argv)


def raw_fingerprints(array, node_order, metadata):
    """Size and mtime only (no reads) of every monthly file beside the array, node_order and the raw metadata."""
    files = sorted(Path(array).parent.glob("2023_p*.npy")) + [Path(node_order), Path(metadata)]
    return {p.name: list(data_io.file_fingerprint(p)) for p in files}


def git_state():
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout  # noqa: E731
        return {"commit": run("rev-parse", "HEAD").strip(), "dirty": run("status", "--porcelain").splitlines()}
    except (OSError, subprocess.CalledProcessError) as e:
        return {"error": str(e)}


def checkpoint_counts(out_dir, manifest):
    """Sensors per client, windows per split and client, masked target cells for train and val.

    Test windows are counted (the frozen pipeline stores them) but their targets are not read.
    """
    with np.load(out_dir / "pilot_preprocessed.npz") as d:
        obs = d["observed_mask"]
    client = manifest["proposed_client"].to_numpy(str)
    out = {"sensors_per_client": manifest["proposed_client"].value_counts().sort_index().to_dict(), "splits": {}}
    with np.load(out_dir / "window_index.npz") as w:
        for name in w.files:
            idx = w[name]
            entry = {"windows": int(len(idx)),
                     "windows_per_client": pd.Series(client[idx[:, 0]]).value_counts().sort_index().to_dict()}
            if name != "test":
                m = obs[idx[:, :1], idx[:, 1:2] + np.arange(windows.HORIZON)]
                entry.update(masked_target_cells=int((~m).sum()), windows_with_masked_target=int((~m).any(axis=1).sum()))
            out["splits"][name] = entry
    return out


def main(argv=None):
    args = parse_args(argv)
    cfg = sel.load_config(args.config)
    out_dir = sel.check_output_dir(args.out_dir or args.root / cfg["out_dir"])   # before any data is read
    if args.array.name != RAW_ARRAY_NAME:
        raise SystemExit(f"Only {RAW_ARRAY_NAME} may be used; got {args.array.name}")
    before = raw_fingerprints(args.array, args.node_order, args.raw_metadata)

    inputs = {name: args.root / name for name in cfg["input_sha256"]}
    input_sha = sel.check_input_hashes(inputs, cfg["input_sha256"])
    paths = cfg["inputs"]
    audit, meta, exclude = sel.load_tables(args.root / paths["audit"], args.root / paths["metadata"],
                                           [args.root / p for p in paths["pilot_manifests"]])
    selected, summary = sel.stratified_selection(sel.selection_frame(audit, meta), exclude, cfg["per_client"])
    summary["input_sha256"] = input_sha
    manifest = sel.build_manifest(audit, selected)
    sel.write_outputs(out_dir, manifest, summary)

    opened, real_loader = [], data_io.load_traffic_array

    def recording_loader(path, *a, **k):
        opened.append(Path(path).resolve())
        return real_loader(path, *a, **k)

    pp = cfg["prepare_pilot"]
    pp_argv = ["--array", str(args.array), "--node-order", str(args.node_order),
               "--selection", str(out_dir / "sensor_selection.csv"), "--out-dir", str(out_dir),
               "--min-sensors", str(pp["min_sensors"]), "--max-sensors", str(pp["max_sensors"]),
               "--run-label", pp["run_label"]] + (["--no-shape-check"] if args.no_shape_check else [])
    data_io.load_traffic_array = recording_loader
    try:
        prepare_pilot.main(pp_argv)
    finally:
        data_io.load_traffic_array = real_loader
    if opened != [args.array.resolve()]:
        raise AssertionError(f"Unexpected raw arrays opened: {[p.name for p in opened]}")
    after = raw_fingerprints(args.array, args.node_order, args.raw_metadata)
    if after != before:
        raise AssertionError("Raw files changed during the geo200 stage")

    artifacts = sorted(p for p in out_dir.iterdir() if p.is_file())
    record = {
        "stage": "S2 geo200 selection and checkpoint",
        "created_utc": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}",
        "command": [Path(sys.executable).name, Path(__file__).name, *(argv if argv is not None else sys.argv[1:])],
        "prepare_pilot_args": pp_argv[4:],
        "config_sha256": sel.sha256_file(args.config), "input_sha256": input_sha,
        "artifact_sha256": {p.name: sel.sha256_file(p) for p in artifacts},
        "raw_fingerprints_before": before, "raw_fingerprints_after": after,
        "raw_arrays_opened": [p.name for p in opened], "id_mapping_status": config.ID_MAPPING_STATUS,
        "git": git_state(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "counts": checkpoint_counts(out_dir, manifest),
    }
    (out_dir / "checkpoint_manifest.json").write_text(json.dumps(record, indent=2, default=int), encoding="utf-8")
    print(f"geo200: {len(manifest)} sensors, {len(summary['fallbacks'])} fallbacks; wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
