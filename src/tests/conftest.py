import numpy as np
import pytest

from aiori import config, preprocessing, windows

P = 24                                   # steps per day in synthetic runs
SPLITS = {"train": (0, 120), "val": (120, 168), "test": (168, 216)}
HAS_DATA = (config.PROCESSED_DIR / "pilot_preprocessed.npz").is_file() and \
    (config.PROCESSED_DIR / "missing_validation" / "pilot_preprocessed.npz").is_file()


def pytest_configure(config):
    config.addinivalue_line("markers", "data: needs the local processed artifacts in data/processed/")


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="data/processed/ artifacts not present")
    for item in items:
        if "data" in item.keywords and not HAS_DATA:
            item.add_marker(skip)


def make_run(n_sensors=3, seed=0, name="synthetic", clients=None):
    """A small Run built with the real preprocessing functions; sensor 2 has missing values."""
    torch = pytest.importorskip("torch")  # noqa: F841 - dataset needs torch
    from aiori.dataset import Run

    rng = np.random.default_rng(seed)
    t = np.arange(SPLITS["test"][1])
    flow = 50 + 30 * np.sin(2 * np.pi * t / P)[None] * rng.uniform(0.5, 2, (n_sensors, 1)) \
        + rng.normal(0, 3, (n_sensors, len(t)))
    flow[2, 30:34] = np.nan               # input gap in train
    flow[2, 140:143] = np.nan             # missing targets in val
    observed = preprocessing.observed_mask(flow)
    scaler = preprocessing.fit_scaler(flow, SPLITS)
    tod = preprocessing.fit_tod_median(flow, SPLITS, P)
    filled, imputed = preprocessing.causal_fill(flow, SPLITS, tod, 6, P)
    idx = {k: windows.window_index(observed, SPLITS, k) for k in SPLITS}
    return Run(
        name=name, inputs_scaled=preprocessing.transform(filled, scaler).astype(np.float32),
        input_imputed=imputed, targets_scaled=preprocessing.transform(flow, scaler).astype(np.float32),
        observed_mask=observed, flow_raw=flow.astype(np.float32), tod_median=tod,
        array_index=np.arange(100, 100 + n_sensors), station_id=np.arange(9000, 9000 + n_sensors),
        scaler=dict(scaler, array_index=list(range(100, 100 + n_sensors)),
                    station_id=list(range(9000, 9000 + n_sensors))),
        windows=idx, client=np.array(clients or ["a", "a", "b"][:n_sensors]), steps_per_day=P,
    )


@pytest.fixture
def synthetic_run():
    return make_run()
