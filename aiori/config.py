"""Project paths and fixed data conventions."""

from pathlib import Path

# Reuse the district -> client map from the existing rebuild script instead of
# keeping a second copy. Requires the project root on sys.path (run from root).
from rebuild_aiori_metadata import CLIENT_BY_DISTRICT  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
ARRAY_PATH = ROOT / "year_2023" / "2023_p01.npy"
NODE_ORDER_PATH = ROOT / "node_order.npy"
METADATA_PATH = ROOT / "sensor_meta_feature.csv"  # tab-separated despite .csv
PROCESSED_DIR = ROOT / "data" / "processed"

EXPECTED_P01_SHAPE = (16972, 8928, 3)

# Raw channel order of the public .npy, following upstream data_ana.ipynb.
# Working convention, not an independently confirmed unit/mapping.
FLOW, OCCUPANCY, SPEED = 0, 1, 2

STEPS_PER_DAY = 288  # 5-minute slots

# Public row i <-> node_order[i] is supported but unconfirmed (handoff section 8).
ID_MAPPING_STATUS = "provisional: node_order[array_index]; row-to-ID mapping unconfirmed"
