import json

import pytest

from aiori import splits


def test_january_default_split_is_day_aligned():
    s = splits.make_splits(8928)
    assert s == {"train": (0, 6336), "val": (6336, 7200), "test": (7200, 8928)}
    assert all(start % 288 == 0 for start, _ in s.values())


def test_splits_cover_range_without_overlap():
    s = splits.make_splits(10 * 288)
    assert s["train"][0] == 0 and s["test"][1] == 10 * 288
    assert s["train"][1] == s["val"][0] and s["val"][1] == s["test"][0]


def test_validate_rejects_bad_boundaries():
    with pytest.raises(ValueError):
        splits.validate_splits({"train": (0, 10), "val": (9, 20), "test": (20, 30)}, 30)
    with pytest.raises(ValueError):
        splits.validate_splits({"train": (0, 10), "val": (11, 20), "test": (20, 30)}, 30)
    with pytest.raises(ValueError):
        splits.validate_splits({"train": (0, 10), "val": (10, 20), "test": (20, 29)}, 30)


def test_partial_days_rejected():
    with pytest.raises(ValueError, match="whole number of days"):
        splits.make_splits(1000)


def test_split_config_round_trip(tmp_path):
    s = splits.make_splits(8928)
    cfg = splits.split_config(s, n_steps=8928)
    assert cfg["timestamps"]["val"]["first"] == "2023-01-23 00:00"
    assert cfg["timestamps"]["test"]["last"] == "2023-01-31 23:55"
    path = tmp_path / "split.json"
    splits.save_split_config(path, cfg)
    assert splits.load_split_config(path) == s
    assert json.loads(path.read_text())["index_convention"].startswith("half-open")
