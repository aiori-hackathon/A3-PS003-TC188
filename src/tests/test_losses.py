import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aiori.losses import masked_mse  # noqa: E402


def _data(seed=0):
    rng = np.random.default_rng(seed)
    y_hat = rng.normal(size=(6, 12))
    y = rng.normal(size=(6, 12))
    mask = rng.random((6, 12)) > 0.3
    return y_hat, y, mask


def test_matches_numpy_reference():
    y_hat, y, mask = _data()
    loss, n = masked_mse(torch.tensor(y_hat), torch.tensor(y), torch.tensor(mask))
    assert n == mask.sum()
    assert loss.item() == pytest.approx(((y_hat - y)[mask] ** 2).mean())


def test_nan_targets_give_finite_loss_and_zero_gradient_on_masked_cells():
    y_hat, y, mask = _data(1)
    y[~mask] = np.nan                              # raw NaN targets, not pre-zeroed
    p = torch.tensor(y_hat, requires_grad=True)
    loss, _ = masked_mse(p, torch.tensor(y), torch.tensor(mask))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(p.grad).all()
    assert (p.grad[~torch.tensor(mask)] == 0).all()
    expected = 2 * (y_hat - np.nan_to_num(y)) * mask / mask.sum()
    assert np.allclose(p.grad.numpy(), expected)


def test_nan_targets_do_not_reach_model_weights():
    lin = torch.nn.Linear(3, 12)
    y = torch.full((4, 12), float("nan"))
    mask = torch.zeros(4, 12, dtype=torch.bool)
    y[:, 0], mask[:, 0] = 1.0, True
    loss, _ = masked_mse(lin(torch.ones(4, 3)), y, mask)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in lin.parameters())


def test_fully_masked_batch_returns_none():
    y_hat, y, _ = _data()
    loss, n = masked_mse(torch.tensor(y_hat), torch.tensor(y), torch.zeros(6, 12, dtype=torch.bool))
    assert loss is None and n == 0
