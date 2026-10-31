"""Validated, read-only loading of the raw XTraffic files."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return path


def load_traffic_array(path, expected_shape=None):
    """Memory-map a (sensor, time, channel) float32 array without loading it into RAM."""
    arr = np.load(_require_file(path), mmap_mode="r", allow_pickle=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3-D (sensor, time, channel) array; got shape {arr.shape}")
    if arr.dtype != np.float32:
        raise ValueError(f"Expected dtype float32; got {arr.dtype}")
    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"Unexpected array shape {arr.shape}; expected {tuple(expected_shape)}")
    log.info("Traffic array %s: shape=%s dtype=%s (memory-mapped)", path, arr.shape, arr.dtype)
    return arr


def load_node_order(path, n_sensors=None):
    node_order = np.load(_require_file(path), allow_pickle=False)
    if node_order.ndim != 1:
        raise ValueError(f"node_order must be 1-D; got shape {node_order.shape}")
    if pd.Index(node_order).has_duplicates:
        raise ValueError("node_order contains duplicate sensor IDs")
    if n_sensors is not None and len(node_order) != n_sensors:
        raise ValueError(f"node_order length {len(node_order)} != array sensor axis {n_sensors}")
    return node_order


def load_metadata(path):
    # The source file is tab-separated despite its .csv extension.
    return pd.read_csv(_require_file(path), sep="\t")


def align_metadata(metadata, node_order):
    """Return metadata rows ordered so that row i describes node_order[i].

    This is internally consistent, but it does not prove that public array row i
    is sensor node_order[i]; that mapping is supported but unconfirmed.
    """
    if "station_id" not in metadata.columns:
        raise ValueError("Metadata has no station_id column")
    if metadata["station_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate station_id values")
    ids, meta_ids = pd.Index(node_order), pd.Index(metadata["station_id"])
    if len(ids.difference(meta_ids)) or len(meta_ids.difference(ids)):
        raise ValueError(
            f"Sensor ID sets differ: {len(ids.difference(meta_ids))} node IDs without metadata, "
            f"{len(meta_ids.difference(ids))} metadata IDs not in node_order"
        )
    aligned = metadata.set_index("station_id", drop=False).loc[node_order].reset_index(drop=True)
    assert np.array_equal(aligned["station_id"].to_numpy(), np.asarray(node_order))
    return aligned


def file_fingerprint(path):
    st = Path(path).stat()
    return st.st_size, st.st_mtime_ns


def check_output_dir(out_dir, raw_paths):
    """Refuse to write outputs into a raw file or directly beside the raw inputs."""
    out = Path(out_dir).resolve()
    for raw in map(lambda p: Path(p).resolve(), raw_paths):
        if out == raw or out == raw.parent:
            raise ValueError(f"Output dir {out} would write into raw data location {raw}")
    return out
