"""AIORI 2.0 Phase 4: five forecasters, masked loss, checkpoints and parameter exchange (no training)."""

import json
import re

import numpy as np
import pytest
import torch

from aiori2 import arms, core
from aiori2.losses import batch_tensors, masked_loss, scaled_targets, train_step
from aiori2.models import MODELS, average_states, build_model, count_parameters, load_checkpoint, save_checkpoint
from aiori2.models.reuse import aopt_models
from test_aiori2 import synth  # noqa: F401

NAMES = list(MODELS)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# 1-8: factory, shapes, channels, finiteness, devices

@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("channels", [1, 2])
def test_factory_shapes_and_batch_sizes(name, channels):
    m = build_model(name, in_channels=channels, seed=0)
    for b in (1, 3, 8, 64):
        y = m(torch.randn(b, 12, channels))
        assert y.shape == (b, 12) and y.dtype == torch.float32 and torch.isfinite(y).all()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("device", DEVICES)
def test_forward_backward_step(name, device):
    m = build_model(name, seed=0).to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    x, y = torch.randn(16, 12, 1, device=device), torch.randn(16, 12, device=device) * 50 + 100
    M = torch.rand(16, 12, device=device) > 0.2
    before = {k: v.clone() for k, v in m.state_dict().items()}
    loss = train_step(m, opt, (x, torch.where(M, y, torch.nan), M, torch.full((16,), 100.0, device=device),
                               torch.full((16,), 50.0, device=device), None))
    assert loss is not None and np.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert any(not torch.equal(before[k], v) for k, v in m.state_dict().items())


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build_model("stgcn")


# 9-15: masked loss and scaling

def _loss_case():
    pred = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], requires_grad=True)
    y = torch.tensor([[110.0, float("nan"), 130.0], [100.0, 90.0, float("nan")]])
    M = torch.isfinite(y)
    return pred, y, M, torch.tensor([100.0, 100.0]), torch.tensor([10.0, 10.0])


def test_masked_mae_and_mse_values():
    pred, y, M, mean, std = _loss_case()
    # scaled valid targets: row0 [1, -, 3], row1 [0, -1, -]; errors 0, 0, 0, 1
    assert masked_loss(pred, y, M, mean, std, "mae").item() == pytest.approx(0.25)
    assert masked_loss(pred, y, M, mean, std, "mse").item() == pytest.approx(0.25)
    pred2 = pred.detach() + torch.tensor([[2.0, 0, 0], [0, 0, 0]])
    assert masked_loss(pred2, y, M, mean, std, "mse").item() == pytest.approx((4 + 1) / 4)


def test_nan_targets_never_reach_loss_or_gradients():
    pred, y, M, mean, std = _loss_case()
    loss = masked_loss(pred, y, M, mean, std)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    assert (pred.grad[~M] == 0).all()
    assert torch.isfinite(scaled_targets(y, M, mean, std)).all()


def test_all_invalid_batch_skips_update():
    m = build_model("mlp", seed=0)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    M = torch.zeros(4, 12, dtype=torch.bool)
    assert train_step(m, opt, (torch.randn(4, 12, 1), torch.full((4, 12), torch.nan), M, torch.ones(4), torch.ones(4), None)) is None
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())


@pytest.mark.parametrize("name", NAMES)
def test_invalid_target_values_do_not_change_loss_or_gradients(name):
    m = build_model(name, seed=0).eval()                             # eval: no dropout noise between the two passes
    x = torch.randn(8, 12, 1)
    M = torch.rand(8, 12, generator=torch.Generator().manual_seed(1)) > 0.3
    y = torch.randn(8, 12) * 30 + 100
    mean, std = torch.full((8,), 100.0), torch.full((8,), 30.0)
    out = []
    for fill in (torch.nan, 1e9):
        m.zero_grad()
        loss = masked_loss(m(x), torch.where(M, y, torch.as_tensor(fill)), M, mean, std)
        loss.backward()
        # N-HiTS: the last block's backcast feeds nothing, so its weights get no gradient (as in the original design)
        out.append((loss.item(), [torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in m.parameters()]))
    assert out[0][0] == out[1][0] and all(torch.equal(a, b) for a, b in zip(out[0][1], out[1][1]))


def test_scaling_round_trip_with_stream_scalers(synth):
    ld = arms.loaders(synth.out, "fedavg_geo", "A", "val")["client_1"]
    X, Y, M, mean, std, s = batch_tensors(ld, np.arange(min(len(ld), 200)))
    z = scaled_targets(Y, M, mean, std)
    assert np.allclose(ld.inverse(z.numpy(), s.numpy())[M.numpy()], Y[M].numpy(), atol=1e-3)
    assert np.allclose(z[M].numpy(), ld.scale(Y.numpy(), s.numpy())[M.numpy()], atol=1e-5)


# 16-20: architecture-specific behaviour

def test_mlp_is_the_reused_residual_mlp():
    m = build_model("mlp", seed=0)
    body = m.net.body
    assert m.net.kind == "mlp" and len(body["blocks"]) == 3 and isinstance(body["blocks"][0][0], torch.nn.LayerNorm)
    x = torch.randn(4, 12, 1)
    with torch.no_grad():
        r = body["inp"](x.flatten(1)); r1 = r + body["blocks"][0].eval()(r)
        assert not torch.equal(r, r1)                                  # the skip adds the block output
    m2 = build_model("mlp", seed=0)
    for p in m2.net.body["blocks"].parameters():
        torch.nn.init.zeros_(p)                                        # zero blocks -> pure skip path
    with torch.no_grad():
        assert torch.allclose(m2.eval().net.represent(x), m2.net.body["inp"](x.flatten(1)))


def test_gru_equals_the_study_gru_and_reads_the_sequence():
    from aiori.model import ForecastGRU
    m = build_model("gru", seed=0, layers=1, hidden=32, dropout=0.0).eval()
    ref = ForecastGRU(1, 32)
    sd = {k.replace("net.body.", "gru.").replace("net.head.net.", "head."): v for k, v in m.state_dict().items()}
    ref.load_state_dict(sd, strict=True)
    x = torch.randn(6, 12, 1)
    with torch.no_grad():
        assert torch.allclose(m(x), ref(x), atol=1e-6)
        x2 = x.clone(); x2[:, 0] += 5                                   # the first step matters too
        assert not torch.allclose(m(x), m(x2))


def test_tcn_is_causal_and_matches_the_study_tcn():
    m = build_model("tcn", seed=0, dropout=0.0).eval()
    x = torch.randn(3, 12, 1)
    for t in range(12):
        x2 = x.clone(); x2[:, t] += 10
        with torch.no_grad():
            a, b = m.features(x), m.features(x2)
        assert torch.equal(a[:, :, :t], b[:, :, :t]) and not torch.equal(a[:, :, t:], b[:, :, t:])
    assert m.receptive_field >= 12
    ref = aopt_models.Forecaster("tcn", past_dim=1, future_dim=0, input_len=12, hidden=96, layers=3).eval()
    sd = {}
    for k, v in ref.state_dict().items():
        k2 = re.sub(r"body\.(\d+)\.1\.", r"blocks.\1.conv.", k).replace("head.net.", "head.")
        sd[k2] = v
    m.load_state_dict(sd, strict=True)
    with torch.no_grad():
        assert torch.allclose(m(x), ref(x), atol=1e-6)


def test_patchtst_patches_and_attention():
    m = build_model("patchtst", seed=0, dropout=0.0).eval()
    assert m.n_patches == 6 and m.pad == 2
    x = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
    p = m.patchify(x)
    assert p.shape == (1, 6, 4)
    assert p[0, 0].tolist() == [0, 1, 2, 3] and p[0, 5].tolist() == [10, 11, 11, 11]   # end-padded last patch
    m5 = build_model("patchtst", seed=0, patch_len=5, stride=3)                 # (12 + pad - 5) % 3 == 0 -> pad 5
    assert m5.pad == 5 and m5.n_patches == 5 and m5(torch.randn(2, 12, 1)).shape == (2, 12)
    x2 = torch.arange(24, dtype=torch.float32).reshape(1, 12, 2)
    assert m.__class__(in_channels=2).patchify(x2)[0, 0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    seen = []
    h = m.encoder.layers[0].self_attn.register_forward_hook(lambda mod, inp, out: seen.append(inp[0].shape))
    with torch.no_grad():
        y = m(torch.randn(2, 12, 1))
    h.remove()
    assert seen and seen[0][1] == 6 and isinstance(m.encoder.layers[0].self_attn, torch.nn.MultiheadAttention)
    x3 = torch.randn(1, 12, 1); x4 = x3.clone(); x4[:, 0] += 3                      # the first patch reaches the forecast
    with torch.no_grad():
        assert not torch.allclose(m(x3), m(x4))


def test_nhits_doubly_residual_structure():
    m = build_model("nhits", seed=0, dropout=0.0).eval()
    x = torch.randn(5, 12, 1)
    with torch.no_grad():
        p = m.parts(x)
    assert torch.allclose(p["forecast"], sum(p["forecasts"]), atol=1e-6)
    assert torch.allclose(p["residual"], x[..., 0] - sum(p["backcasts"]), atol=1e-5)
    assert [b.n_forecast for b in m.blocks] == [3, 6, 12]
    assert all(isinstance(b.pool, torch.nn.MaxPool1d) for b in m.blocks[:2])
    c = p["forecasts"][0]                                               # 3 coarse points interpolated to 12
    assert torch.allclose(c[:, 0], m.blocks[0].forecast(m.blocks[0].mlp(m.blocks[0].pool(x.transpose(1, 2)).flatten(1)))[:, 0], atol=1e-6)
    m2 = build_model("nhits", in_channels=2, seed=0)
    assert m2(torch.randn(2, 12, 2)).shape == (2, 12)


# 21-25: checkpoints, determinism, clients, aggregation

@pytest.mark.parametrize("name", NAMES)
def test_checkpoint_round_trip(name, tmp_path):
    m = build_model(name, seed=3).eval()
    p = tmp_path / f"{name}.pt"
    save_checkpoint(m, p, dict(note="test"))
    with pytest.raises(FileExistsError):
        save_checkpoint(m, p)
    r, meta = load_checkpoint(p)
    x = torch.randn(4, 12, 1)
    with torch.no_grad():
        assert torch.equal(m(x), r.eval()(x))
    assert meta["data_contract"].startswith("aiori2_3yr") and meta["seed"] == 3


@pytest.mark.parametrize("name", NAMES)
def test_deterministic_initialisation_and_global_rng_untouched(name):
    torch.manual_seed(123); before = torch.rand(1)
    torch.manual_seed(123); a = build_model(name, seed=7); after = torch.rand(1)
    b = build_model(name, seed=7)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    assert torch.equal(before, after)
    c = build_model(name, seed=8)
    assert any(not torch.equal(a.state_dict()[k], c.state_dict()[k]) for k in a.state_dict())


@pytest.mark.parametrize("name", NAMES)
def test_federated_parameter_compatibility(name):
    a, b = build_model(name, seed=1), build_model(name, seed=2)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(sa[k].shape == sb[k].shape for k in sa)
    assert len(list(a.buffers())) == 0                                   # no running statistics to reconcile
    assert not any(re.search(r"emb|mean|std|scaler", k) for k in sa)     # no client data or scalers inside
    avg = average_states([sa, sb], [3, 1])
    for k in sa:
        assert torch.allclose(avg[k], 0.75 * sa[k] + 0.25 * sb[k], atol=1e-6)
    c = build_model(name, seed=9)
    c.load_state_dict(avg, strict=True)
    with torch.no_grad():
        assert torch.isfinite(c(torch.randn(2, 12, 1))).all()
    with pytest.raises(ValueError):
        average_states([sa, {k: v[..., :1] if v.dim() else v for k, v in sb.items()}], [1, 1])


def test_non_floating_entries_must_agree():
    s1 = {"w": torch.ones(2), "n": torch.tensor(3)}
    assert average_states([s1, {"w": torch.zeros(2), "n": torch.tensor(3)}], [1, 1])["n"] == 3
    with pytest.raises(ValueError):
        average_states([s1, {"w": torch.zeros(2), "n": torch.tensor(4)}], [1, 1])


@pytest.mark.parametrize("name", NAMES)
def test_client_models_train_on_own_streams_only(synth, name):
    """Every client of both federated arms gets the same architecture; each trains from its own stream alone."""
    shapes = None
    for arm in ("fedavg_iid", "fedavg_geo", "local"):
        for cname, ld in arms.loaders(synth.out, arm, "A", "train").items():
            m = build_model(name, seed=0)
            sh = {k: tuple(v.shape) for k, v in m.state_dict().items()}
            shapes = shapes or sh
            assert sh == shapes
            batch = batch_tensors(ld, arms.epoch_sample(len(ld), 16, 0, 0))
            assert np.isin(batch[5].numpy(), ld.sensors).all()
            assert train_step(m, torch.optim.Adam(m.parameters(), 1e-3), batch) is not None


# 26-28: real batches, protected files, portability

REAL = core.load_config()
HAS = (REAL["_out"] / "phase3" / "phase3_validation.json").is_file()


@pytest.mark.skipif(not HAS, reason="Phase 2/3 outputs not generated")
@pytest.mark.parametrize("name", NAMES)
def test_real_batches_from_both_histories(name):
    for h in arms.HISTORIES:
        ld = arms.loaders(REAL["_out"], "centralised", h, "train")["all"]
        batch = batch_tensors(ld, arms.epoch_sample(len(ld), 64, 0, 1))
        m = build_model(name, seed=0)
        assert train_step(m, torch.optim.Adam(m.parameters(), 1e-3), batch) is not None
        with torch.no_grad():
            assert torch.isfinite(m.eval()(batch[0])).all()


@pytest.mark.skipif(not HAS, reason="Phase 2/3 outputs not generated")
def test_protected_models_and_checkpoints_unchanged():
    from aiori2 import pipeline
    res = pipeline.verify(REAL)
    assert res["ok"] and res["protected_files"] == 176


def test_new_code_is_portable():
    files = list((core.SRC / "aiori2").rglob("*.py"))
    assert files
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert "wintypes" not in text and "powershell" not in text.lower() and not re.search(r"[A-Z]:\\\\", text), f
