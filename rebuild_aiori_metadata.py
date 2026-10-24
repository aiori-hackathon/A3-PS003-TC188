"""
Rebuild derived XTraffic metadata files for AIORI PS-003.

Expected inputs in the project root:
  - node_order.npy
  - sensor_meta_feature.csv  (tab-separated despite its .csv extension)
  - year_2023/2023_p01.npy    (used only for a lightweight shape check)

Outputs:
  - data/processed/sensor_metadata_aligned.csv
  - data/processed/proposed_client_assignment.csv

This script does not modify raw inputs. The relationship between public
2023_p01.npy row order and node_order.npy remains supported but unconfirmed.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
NODE_ORDER_PATH = ROOT / "node_order.npy"
METADATA_PATH = ROOT / "sensor_meta_feature.csv"
ARRAY_PATH = ROOT / "year_2023" / "2023_p01.npy"
OUTPUT_DIR = ROOT / "data" / "processed"

CLIENT_BY_DISTRICT = {
    7: "client_0_la_ventura",
    4: "client_1_bay_area",
    5: "client_2_coastal_south_central",
    12: "client_2_coastal_south_central",
    3: "client_3_inland_central",
    6: "client_3_inland_central",
    8: "client_3_inland_central",
    10: "client_3_inland_central",
}


def main() -> int:
    missing = [str(p) for p in (NODE_ORDER_PATH, METADATA_PATH, ARRAY_PATH) if not p.is_file()]
    if missing:
        print("ERROR: Required input file(s) not found:")
        for path in missing:
            print(f"  - {path}")
        print("\nPut these raw files in the project root, or edit the paths in this script.")
        return 1

    node_order = np.load(NODE_ORDER_PATH, allow_pickle=False)
    if node_order.ndim != 1:
        raise ValueError(f"node_order.npy must be 1-D; got shape {node_order.shape}")
    if pd.Index(node_order).has_duplicates:
        raise ValueError("node_order.npy contains duplicate IDs.")

    # Memory-map only: do not load the full 1.8 GB array into RAM.
    traffic = np.load(ARRAY_PATH, mmap_mode="r", allow_pickle=False)
    if traffic.ndim != 3:
        raise ValueError(f"Expected 3-D traffic array; got shape {traffic.shape}")
    if traffic.shape[0] != len(node_order):
        raise ValueError(
            f"Array sensor axis ({traffic.shape[0]}) does not match "
            f"node_order length ({len(node_order)})."
        )
    print(f"Traffic array: shape={traffic.shape}, dtype={traffic.dtype} (memory-mapped)")

    # The source file is tab-separated despite the .csv extension.
    metadata = pd.read_csv(METADATA_PATH, sep="\t")
    required = {"station_id", "District"}
    absent = required - set(metadata.columns)
    if absent:
        raise ValueError(f"Metadata missing required columns: {sorted(absent)}")
    if metadata["station_id"].duplicated().any():
        dupes = metadata.loc[metadata["station_id"].duplicated(), "station_id"].head().tolist()
        raise ValueError(f"Metadata has duplicate station_id values, examples: {dupes}")

    if len(metadata) != len(node_order):
        raise ValueError(
            f"Metadata rows ({len(metadata)}) != node_order length ({len(node_order)})."
        )

    node_ids = pd.Index(node_order)
    metadata_ids = pd.Index(metadata["station_id"])
    missing_metadata = node_ids.difference(metadata_ids)
    extra_metadata = metadata_ids.difference(node_ids)
    if len(missing_metadata) or len(extra_metadata):
        raise ValueError(
            f"Sensor ID sets differ: missing metadata IDs={len(missing_metadata)}, "
            f"extra metadata IDs={len(extra_metadata)}. Do not continue until reconciled."
        )

    # Align metadata row i to node_order[i]. This alignment is internally
    # consistent but does not prove the public tensor's row-order provenance.
    indexed = metadata.set_index("station_id", drop=False)
    aligned = indexed.loc[node_order].reset_index(drop=True)
    if not np.array_equal(aligned["station_id"].to_numpy(), node_order):
        raise AssertionError("Aligned metadata station_id order does not match node_order.")

    district_num = pd.to_numeric(aligned["District"], errors="coerce")
    aligned["District"] = district_num
    aligned["proposed_client"] = district_num.map(CLIENT_BY_DISTRICT)

    unassigned = aligned["proposed_client"].isna()
    if unassigned.any():
        unknown = sorted(aligned.loc[unassigned, "District"].dropna().unique().tolist())
        missing_district_count = int(aligned.loc[unassigned, "District"].isna().sum())
        raise ValueError(
            "Some sensors cannot be assigned. "
            f"Unmapped districts={unknown}; rows with missing District={missing_district_count}. "
            "No output files written."
        )

    assignment_columns = [
        col for col in ["station_id", "District", "County", "City", "Lat", "Lng", "proposed_client"]
        if col in aligned.columns
    ]
    assignment = aligned[assignment_columns].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aligned_path = OUTPUT_DIR / "sensor_metadata_aligned.csv"
    assignment_path = OUTPUT_DIR / "proposed_client_assignment.csv"

    # Write derived outputs only.
    aligned.to_csv(aligned_path, index=False)
    assignment.to_csv(assignment_path, index=False)

    # Read-after-write checks.
    aligned_check = pd.read_csv(aligned_path)
    assignment_check = pd.read_csv(assignment_path)
    if len(aligned_check) != len(node_order):
        raise AssertionError("Aligned output row count changed after writing.")
    if not np.array_equal(aligned_check["station_id"].to_numpy(), node_order):
        raise AssertionError("Written aligned metadata order failed validation.")
    if assignment_check["station_id"].nunique() != len(node_order):
        raise AssertionError("Assignment output has duplicate or missing station IDs.")
    if assignment_check["proposed_client"].isna().any():
        raise AssertionError("Assignment output contains unassigned sensors.")

    print("\nSUCCESS — derived files recreated and validated.")
    print(f"  {aligned_path}")
    print(f"  {assignment_path}")
    print("\nSensors per client:")
    print(assignment_check.groupby("proposed_client")["station_id"].nunique().to_string())
    print("\nDistricts per client:")
    print(
        assignment_check.groupby("proposed_client")["District"]
        .agg(lambda s: sorted(s.dropna().unique().tolist()))
        .to_string()
    )
    print(
        "\nCAVEAT: This validates metadata alignment and assignment consistency only. "
        "It does not independently prove that public array row i corresponds to "
        "node_order.npy[i]."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
