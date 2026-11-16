"""The four AIORI experimental arms on the three-year data: who trains on which sensors, with which scaler.

An arm is a list of data streams. Centralised = one stream of all 200 sensors (pooled scaler, no aggregation).
Local = four independent geographic streams (no aggregation). FedAvg-geo / FedAvg-IID = four federated clients
(geographic / study IID partition) whose models are aggregated. A stream only ever sees its own sensors' windows,
and its scaler was fitted on those sensors' training observations only (aiori2.pipeline.fit_scalers).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aiori2.data import ThreeYearWindows

ARMS = {
    "centralised": dict(partition=None, scaler="pooled", aggregation=False, label="Centralized"),
    "local": dict(partition="client_geo", scaler="client_geo", aggregation=False, label="Local"),
    "fedavg_iid": dict(partition="client_iid", scaler="client_iid", aggregation=True, label="IID federated"),
    "fedavg_geo": dict(partition="client_geo", scaler="client_geo", aggregation=True, label="Non-IID (geographic) federated"),
}
HISTORIES = ("A", "B")


@dataclass
class Stream:
    name: str                   # "all" or the client name
    sensors: np.ndarray         # sensor positions (0..199) this stream may read
    scaler_scope: str


def streams(root, arm):
    """The data streams of an arm, from the Phase 2 sensor manifest (existing assignments, never regenerated)."""
    spec = ARMS[arm]
    sens = pd.read_csv(Path(root) / "metadata" / "sensors.csv")
    if spec["partition"] is None:
        return [Stream("all", sens["position"].to_numpy(), spec["scaler"])]
    col = sens[spec["partition"]].to_numpy()
    return [Stream(c, np.flatnonzero(col == c), spec["scaler"]) for c in sorted(set(col))]


def loaders(root, arm, history, split, policy="conservative", eval_mask=None):
    """{stream name: ThreeYearWindows restricted to that stream's sensors and scaler}."""
    return {s.name: ThreeYearWindows(root, policy, history, split, s.scaler_scope, sensors=s.sensors, eval_mask=eval_mask)
            for s in streams(root, arm)}


def epoch_sample(n, k, seed, epoch):
    """Deterministic uniform sample of k of n window positions (without replacement) for one epoch."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
    if k >= n:
        return rng.permutation(n)
    return np.sort(rng.choice(n, size=k, replace=False))


def manifest(root, arm, history, policy, quality_sensitivity, seed, config_sha256, data_sha256, out_dir):
    exp = json.loads((Path(root) / "experiments" / policy / history / "experiment.json").read_text(encoding="utf-8"))
    sens = pd.read_csv(Path(root) / "metadata" / "sensors.csv")
    spec = ARMS[arm]
    return dict(
        arm=arm, label=spec["label"], training_history=history, history_description=exp["description"],
        train_dates=exp["splits_dates"]["train"], val_dates=exp["splits_dates"]["val"], test_dates=exp["splits_dates"]["test"],
        date_convention="[start, end) at 00:00 reconstructed Pacific local time",
        n_sensors=int(len(sens)), station_ids=sens["station_id"].astype(int).tolist(),
        partition=spec["partition"] or "none (all 200 sensors pooled)", aggregation=spec["aggregation"],
        clients={s.name: dict(positions=s.sensors.tolist(), station_ids=sens["station_id"].to_numpy()[s.sensors].astype(int).tolist())
                 for s in streams(root, arm)},
        iid_partition_seed=20261027, missing_value_policy=policy, quality_sensitivity=quality_sensitivity,
        scaler_scope=spec["scaler"], input_len=exp["input_len"], horizon=exp["horizon"], input_channels=exp["input_channels"],
        min_valid_targets=exp["min_valid_targets"], random_seed=seed,
        tensor_layout=dict(X="(B, 12, 1) float32, flow scaled with the stream's scaler, finite",
                           Y="(B, 12) float32 raw flow, NaN where the target is not valid", M="(B, 12) bool valid-target mask",
                           sensor="(B,) int64 position 0..199", prediction="(B, 12) in scaled units; invert with loader.inverse(z, sensor)"),
        data_root=str(Path(root).as_posix()), experiment_dir=f"experiments/{policy}/{history}",
        config_sha256=config_sha256, data_manifest_sha256=data_sha256, output_dir=f"results/aiori2/{arm}_{history}_{policy}")
