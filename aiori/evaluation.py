"""Raw-unit evaluation of z-scaled predictions, reference forecasts, and prediction records.

Predictions are inverted with the training-fit per-sensor scaler of each window's sensor
position. Truth is read directly from `flow_raw` (NaN where missing), never imputed.
Negative predictions are not clipped; their share is reported separately.
"""

import json

import numpy as np

from aiori import baselines, metrics, preprocessing
from aiori.windows import HORIZON


def to_raw(z, run, sensor_pos):
    return preprocessing.inverse_transform(z, run.scaler, sensors=np.asarray(sensor_pos))


def target_truth(run, index, horizon=HORIZON):
    cols = index[:, 1][:, None] + np.arange(horizon)
    return np.asarray(run.flow_raw, dtype=np.float64)[index[:, 0][:, None], cols]


def score(y_hat, run, index, y_hat_z=None):
    """Full metric table for raw-unit predictions on windows `index` (N, 2)."""
    y_true = target_truth(run, index, y_hat.shape[1])
    mask = np.isfinite(y_true)
    sensor_pos = index[:, 0]
    per_sensor = metrics.by_sensor(y_hat, y_true, mask, sensor_pos)
    out = {
        "pooled": metrics.pooled(y_hat, y_true, mask),
        "sensor_macro": metrics.sensor_macro(per_sensor),
        "by_horizon": metrics.by_horizon(y_hat, y_true, mask),
        "by_client": metrics.by_client(y_hat, y_true, mask, sensor_pos, run.client),
        "by_sensor_pos": {int(k): v for k, v in per_sensor.items()},
        "negative_prediction_rate": metrics.negative_rate(y_hat, mask),
        "n_windows": int(len(index)),
    }
    if y_hat_z is not None:
        y_z = preprocessing.transform(y_true, run.scaler, sensors=sensor_pos)
        out["normalised"] = metrics.pooled(y_hat_z, y_z, mask)
    return out


def evaluate_predictions(pred, run):
    index = np.stack([pred["sensor_pos"], pred["t0"]], axis=1)
    y_hat = to_raw(pred["y_hat_z"], run, pred["sensor_pos"])
    if not np.array_equal(np.isfinite(target_truth(run, index)), pred["mask"]):
        raise ValueError("Prediction mask disagrees with the raw targets")
    return score(y_hat, run, index, pred["y_hat_z"])


def evaluate_baselines(run, split):
    index = run.windows[split]
    return {name: score(fn(run, index), run, index) for name, fn in baselines.BASELINES.items()}


def write_prediction_record(path, pred, run, meta):
    """predictions_<arm>_<seed>_<split>.npz (spec section 4.4); `meta` is stored as JSON."""
    sensor_pos = np.asarray(pred["sensor_pos"])
    index = np.stack([sensor_pos, pred["t0"]], axis=1)
    codes, client = np.unique(run.client, return_inverse=True)
    np.savez_compressed(
        path,
        sensor_pos=sensor_pos.astype(np.int32), array_index=run.array_index[sensor_pos],
        client=client[sensor_pos].astype(np.int8), client_names=codes,
        t0=np.asarray(pred["t0"]).astype(np.int32),
        y_hat=to_raw(pred["y_hat_z"], run, sensor_pos).astype(np.float32),
        y_true=target_truth(run, index).astype(np.float32),
        mask=np.asarray(pred["mask"], bool), y_hat_z=np.asarray(pred["y_hat_z"], np.float32),
        meta=json.dumps(meta, sort_keys=True),
    )
