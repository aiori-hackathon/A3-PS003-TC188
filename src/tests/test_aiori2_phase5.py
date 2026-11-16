"""AIORI 2.0 Phase 5: eight federated algorithms — hand-calculated references, equivalences, distinctions,
engine isolation, reproducibility, resume and accounting. No training runs beyond a few tiny rounds."""

import math

import numpy as np
import pytest
import torch
from torch import nn

from aiori2 import arms, core
from aiori2.fl import ALGORITHMS, FederatedRun, local_train, make_config, validate, weighted_sum
from aiori2.fl import algorithms as A
from aiori2.models import MODELS
from test_aiori2 import synth  # noqa: F401

T = {"w": torch.zeros(1)}                                              # scalar parameter template


def alg(name, clients=(), **kw):
    cfg = make_config(algorithm=name, **kw)
    return A.make(name, cfg, T, clients)


def res(*deltas, steps=None, clients=None):
    steps = steps or [1] * len(deltas)
    clients = clients or [f"c{i}" for i in range(len(deltas))]
    return [dict(client=c, delta={"w": torch.tensor([float(d)])}, steps=s) for c, d, s in zip(clients, deltas, steps)]


def w_of(state):
    return float(state["w"][0])


# ---- A: registry, configuration, aggregation primitives ----

def test_registry_has_exactly_the_eight_algorithms():
    assert set(ALGORITHMS) == {"fedavg", "fedprox", "fedadam", "fedyogi", "fedadagrad", "scaffold", "fednova", "fedavgm"}


@pytest.mark.parametrize("kw", [dict(arm="centralised"), dict(arm="local"), dict(algorithm="scaffold", client_optimizer="adam"),
                                dict(algorithm="scaffold", client_momentum=0.9), dict(algorithm="fednova", mu=0.1),
                                dict(algorithm="fednova", client_optimizer="adam"), dict(algorithm="fedavg", mu=0.1),
                                dict(weighting="sensors"), dict(participation=0), dict(algorithm="nope"), dict(history="C"),
                                dict(beta2=1.0), dict(tau=0)])
def test_invalid_configurations_are_rejected(kw):
    with pytest.raises(ValueError):
        make_config(**kw)


def test_weighted_sum_and_rejections():
    out = weighted_sum([{"w": torch.tensor([1.0])}, {"w": torch.tensor([3.0])}], [1, 3])
    assert out["w"].item() == pytest.approx(2.5)
    with pytest.raises(ValueError):
        weighted_sum([{"w": torch.tensor([1.0])}], [0])
    with pytest.raises(ValueError):
        weighted_sum([{"w": torch.zeros(1)}, {"w": torch.zeros(2)}], [1, 1])
    with pytest.raises(FloatingPointError):
        weighted_sum([{"w": torch.tensor([float("nan")])}], [1])
    with pytest.raises(FloatingPointError):
        alg("fedavg").update(T, res(float("inf")), [1])


# ---- B: FedAvg ----

def test_fedavg_hand_weighted_average_and_one_client_identity():
    a = alg("fedavg")
    assert w_of(a.update(T, res(1, 3), [1, 3])) == pytest.approx(2.5)
    assert w_of(a.update({"w": torch.tensor([2.0])}, res(-0.75), [7])) == pytest.approx(1.25)   # = the client model


# ---- C: FedProx ----

def test_fedprox_exact_proximal_gradient():
    m = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(0.0)
    anchor = {"weight": torch.zeros(1, 1)}
    cfg = make_config(algorithm="fedprox", mu=1.0, client_lr=0.1, grad_clip=0)
    loss = lambda mod, b: 0.5 * (mod.weight.sum() - 3.0) ** 2
    local_train(m, [None, None], cfg, anchor=anchor, loss_fn=loss)
    # step 1: grad = (0-3) + 1*(0-0) = -3 -> 0.3; step 2: grad = (0.3-3) + 1*(0.3-0) = -2.4 -> 0.54
    assert m.weight.item() == pytest.approx(0.54, abs=1e-7)
    assert anchor["weight"].item() == 0                                   # reference never moves


def test_fedprox_mu_zero_equals_fedavg(synth):
    s = []
    for a in ("fedavg", "fedprox"):
        r = FederatedRun(make_config(algorithm=a, mu=0.0, local_steps=2, batch_size=8), synth.out)
        r.run(2)
        s.append(r.model.state_dict())
    assert all(torch.equal(s[0][k], s[1][k]) for k in s[0])


# ---- D-F: FedAdam / FedYogi / FedAdagrad ----

def _opt_ref(kind, deltas, b1=0.9, b2=0.99, tau=0.1, lr=1.0):
    w, m, v, out = 0.0, 0.0, tau ** 2, []
    for d in deltas:
        d = float(np.float32(d))                                          # deltas arrive as float32 tensors
        m = b1 * m + (1 - b1) * d
        if kind == "fedadam":
            v = b2 * v + (1 - b2) * d * d
        elif kind == "fedyogi":
            v = v - (1 - b2) * d * d * math.copysign(1, v - d * d) if v != d * d else v
        else:
            v = v + d * d
        w = w + lr * m / (math.sqrt(v) + tau)
        out.append((w, m, v))
    return out


SEQ = [1.0, -2.0, 0.01, 0.5]


@pytest.mark.parametrize("kind", ["fedadam", "fedyogi", "fedadagrad"])
def test_adaptive_server_optimizers_match_hand_calculation(kind):
    a = alg(kind, server_lr=1.0, beta1=0.9, beta2=0.99, tau=0.1)
    w = T
    for d, (ew, em, ev) in zip(SEQ, _opt_ref(kind, SEQ)):
        w = a.update(w, res(d), [1])
        # parameters are float32 between rounds (server state float64), so agreement is to float32 precision
        assert w_of(w) == pytest.approx(ew, abs=1e-7)
        assert a.state["m"]["w"].item() == pytest.approx(em, rel=1e-6) and a.state["v"]["w"].item() == pytest.approx(ev, rel=1e-6)


def test_adaptive_second_moments_are_distinct():
    v = {k: [x[2] for x in _opt_ref(k, SEQ)] for k in ("fedadam", "fedyogi", "fedadagrad")}
    assert v["fedadam"][0] == pytest.approx(0.0199) and v["fedyogi"][0] == pytest.approx(0.02) and v["fedadagrad"][0] == pytest.approx(1.01)
    # step 3 (tiny delta): Yogi's v shrinks additively by (1-b2)d^2, Adam's decays multiplicatively, Adagrad only grows
    assert v["fedyogi"][2] == pytest.approx(v["fedyogi"][1] - 0.01 * 1e-4)
    assert v["fedadam"][2] == pytest.approx(0.99 * v["fedadam"][1] + 0.01 * 1e-4)
    assert v["fedadagrad"][2] > v["fedadagrad"][1]
    finals = {k: alg(k, server_lr=1.0, tau=0.1) for k in v}
    ws = {}
    for k, a in finals.items():
        w = T
        for d in SEQ:
            w = a.update(w, res(d), [1])
        ws[k] = w_of(w)
    assert len({round(x, 9) for x in ws.values()}) == 3


# ---- G: FedAvgM ----

def test_fedavgm_momentum_and_zero_momentum_equivalence():
    a = alg("fedavgm", server_momentum=0.9, server_lr=1.0)
    w = a.update(T, res(1.0), [1])
    assert w_of(w) == pytest.approx(1.0) and a.state["momentum"]["w"].item() == pytest.approx(-1.0)
    w = a.update(w, res(1.0), [1])
    assert w_of(w) == pytest.approx(2.9)                                  # v = 0.9*(-1) - 1 = -1.9
    z, f = alg("fedavgm", server_momentum=0.0, server_lr=1.0), alg("fedavg")
    wz = wf = T
    for d in SEQ:
        wz, wf = z.update(wz, res(d, 2 * d), [1, 3]), f.update(wf, res(d, 2 * d), [1, 3])
        assert w_of(wz) == pytest.approx(w_of(wf))
    b = alg("fedavgm", server_momentum=0.9)
    b.load_state_dict(a.state_dict())
    assert w_of(a.update(w, res(0.5), [1])) == pytest.approx(w_of(b.update(w, res(0.5), [1])))


# ---- H: SCAFFOLD ----

def _quad_client(a_i):
    return lambda mod, b: 0.5 * (mod.weight.sum() - a_i) ** 2


def _scaffold_round(sc, w, targets, steps, lr, cfg, participants):
    results = []
    for c in participants:
        m = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            m.weight.fill_(w)
        corr = {"weight": sc.correction(c)["w"].reshape(1, 1)}
        st = local_train(m, [None] * steps[c], cfg, correction=corr, loss_fn=_quad_client(targets[c]))
        delta = {"w": torch.tensor([m.weight.item() - w])}
        r = dict(client=c, delta=delta, steps=st["steps"])
        if st["steps"]:
            r["dc"] = sc.client_control(c, delta, st["steps"], lr)
            results.append(r)
    new = sc.update({"w": torch.tensor([w])}, results, [1] * len(results)) if results else {"w": torch.tensor([w])}
    return w_of(new)


def test_scaffold_matches_hand_calculation():
    cfg = make_config(algorithm="scaffold", client_lr=0.1, grad_clip=0)
    sc = A.make("scaffold", cfg, T, ["a", "b"])
    targets, steps = {"a": 1.0, "b": 5.0}, {"a": 2, "b": 2}
    w1 = _scaffold_round(sc, 0.0, targets, steps, 0.1, cfg, ["a", "b"])
    # hand: c=c_i=0; client a: y=0 -> 0.1 -> 0.19 ; client b: 0 -> 0.5 -> 0.95
    assert w1 == pytest.approx((0.19 + 0.95) / 2)
    ca, cb = (0 - 0.19) / (2 * 0.1), (0 - 0.95) / (2 * 0.1)              # c_i+ = (w - y_i)/(K lr)
    assert sc.state["c_a"]["w"].item() == pytest.approx(ca) and sc.state["c_b"]["w"].item() == pytest.approx(cb)
    assert sc.state["c"]["w"].item() == pytest.approx((ca + cb) / 2)
    # round 2, client a: y <- y - lr*((y - 1) + c - c_a)
    c = (ca + cb) / 2
    y = w1
    for _ in range(2):
        y = y - 0.1 * ((y - 1.0) + c - ca)
    m = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(w1)
    local_train(m, [None, None], cfg, correction={"weight": sc.correction("a")["w"].reshape(1, 1)}, loss_fn=_quad_client(1.0))
    assert m.weight.item() == pytest.approx(y, rel=1e-6)


def test_scaffold_corrects_drift_with_unequal_steps():
    """Quadratic clients with optima 1 and 5, 2 vs 8 local steps: FedAvg is biased, SCAFFOLD reaches the optimum 3."""
    lr = 0.1
    cfg_s = make_config(algorithm="scaffold", client_lr=lr, grad_clip=0)
    sc = A.make("scaffold", cfg_s, T, ["a", "b"])
    targets, steps = {"a": 1.0, "b": 5.0}, {"a": 2, "b": 8}
    w = 0.0
    for _ in range(300):
        w = _scaffold_round(sc, w, targets, steps, lr, cfg_s, ["a", "b"])
    wa = 0.0
    for _ in range(300):
        ys = [t + (wa - t) * (1 - lr) ** steps[c] for c, t in targets.items()]
        wa = sum(ys) / 2
    assert w == pytest.approx(3.0, abs=1e-6) and abs(wa - 3.0) > 0.3


def test_scaffold_partial_participation_and_zero_steps():
    cfg = make_config(algorithm="scaffold", client_lr=0.1, grad_clip=0)
    sc = A.make("scaffold", cfg, T, ["a", "b", "c"])
    _scaffold_round(sc, 0.0, {"a": 1.0, "b": 5.0, "c": 9.0}, {"a": 3, "b": 0, "c": 3}, 0.1, cfg, ["a", "b"])
    ca = sc.state["c_a"]["w"].item()
    assert ca != 0 and sc.state["c_b"]["w"].item() == 0 and sc.state["c_c"]["w"].item() == 0   # b: 0 steps; c: absent
    assert sc.state["c"]["w"].item() == pytest.approx(ca / 3)              # c = mean over all N clients
    st = sc.state_dict()
    sc2 = A.make("scaffold", cfg, T, ["a", "b", "c"])
    sc2.load_state_dict(st)
    assert all(torch.equal(sc2.state[k]["w"], sc.state[k]["w"]) for k in sc.state)


# ---- I: FedNova ----

def test_fednova_normalisation_by_hand():
    a = alg("fednova")
    eq = a.update(T, res(-1.0, -2.0, steps=[3, 3]), [1, 1])
    assert w_of(eq) == pytest.approx(w_of(alg("fedavg").update(T, res(-1.0, -2.0), [1, 1])))   # equal steps = FedAvg
    un = a.update(T, res(-1.0, -2.0, steps=[1, 4]), [1, 1])
    # d = [-1/1, -2/4] -> mean -0.75; tau_eff = 2.5 -> -1.875 (FedAvg would give -1.5)
    assert w_of(un) == pytest.approx(-1.875) and a.last_tau_eff == pytest.approx(2.5)
    with pytest.raises(ValueError):
        a.update(T, res(-1.0, steps=[0]), [1])


@pytest.mark.parametrize("tau, rho", [(1, 0.5), (2, 0.5), (5, 0.9), (4, 0.0)])
def test_fednova_momentum_norm_matches_heavy_ball(tau, rho):
    # coefficient of gradient g_k in the accumulated update is sum_{j=0}^{tau-1-k} rho^j
    brute = sum(sum(rho ** j for j in range(tau - k)) for k in range(tau))
    assert A.FedNova.a_norm(tau, rho) == pytest.approx(brute)
    m = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(0.0)
    cfg = make_config(algorithm="fednova", client_lr=0.1, client_momentum=rho, grad_clip=0)
    local_train(m, [None] * tau, cfg, loss_fn=lambda mod, b: mod.weight.sum())   # constant gradient 1
    assert m.weight.item() == pytest.approx(-0.1 * brute, rel=1e-6)            # PyTorch SGD momentum = heavy ball


# ---- J: every algorithm with every model (synthetic data) ----

@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("algorithm", sorted(["fedavg", "fedprox", "fedavgm", "fedadam", "fedyogi", "fedadagrad", "scaffold", "fednova"]))
def test_algorithm_model_matrix(synth, algorithm, model):
    r = FederatedRun(make_config(algorithm=algorithm, model=model, arm="fedavg_geo", local_steps=2, batch_size=8, rounds=2), synth.out)
    before = {k: v.clone() for k, v in r.model.state_dict().items()}
    log = r.run(2)
    st = r.model.state_dict()
    assert all(torch.isfinite(v).all() for v in st.values()) and r.round == 2
    assert any(not torch.equal(before[k], st[k]) for k in st)
    assert all(x["aggregated"] == sorted(r.clients) and all(c["steps"] == 2 for c in x["clients"].values()) for x in log)
    assert all(np.isfinite(c["loss"]) for x in log for c in x["clients"].values())


def test_nhits_unused_parameters_stay_finite_and_unmoved(synth):
    for a in ("fedavg", "fedprox", "scaffold", "fednova", "fedadam"):
        r = FederatedRun(make_config(algorithm=a, model="nhits", local_steps=2, batch_size=8), synth.out)
        k = [n for n, p in r.model.named_parameters() if n.startswith("blocks.2.backcast")]
        before = {n: r.model.state_dict()[n].clone() for n in k}
        r.run(2)
        for n in k:
            v = r.model.state_dict()[n]
            assert torch.isfinite(v).all()
            assert torch.equal(v, before[n])                                    # true gradient is zero: never moves


# ---- K: data integration and isolation ----

@pytest.mark.parametrize("arm", ["fedavg_iid", "fedavg_geo"])
@pytest.mark.parametrize("history", ["A", "B"])
def test_clients_isolated_on_training_data_only(synth, arm, history):
    r = FederatedRun(make_config(algorithm="scaffold", arm=arm, history=history, local_steps=1, batch_size=8), synth.out)
    assert len(r.clients) == 4
    own = {s.name: s.sensors for s in arms.streams(synth.out, arm)}
    meta = None
    for c, ld in r.loaders.items():
        assert len(own[c]) == 50 and np.isin(np.asarray(ld.index[:, 0]), own[c]).all()
        meta = ld.meta
        assert int(np.asarray(ld.index[:, 1]).max()) + 12 <= meta["splits_index"]["train"][1]   # never val/test
        for b in r.client_batches(c, 1):
            assert np.isin(b[5].numpy(), own[c]).all() and b[2].dtype == torch.bool
    assert sorted(r.alg.state) == sorted(["c"] + [f"c_{c}" for c in r.clients])
    log = r.run(1)
    assert set(log[0]["clients"]) == set(r.clients)


def test_partial_participation_is_deterministic(synth):
    r = FederatedRun(make_config(participation=0.5, local_steps=1, batch_size=8), synth.out)
    p = [r.participants(i) for i in range(1, 6)]
    assert all(len(x) == 2 for x in p) and p == [r.participants(i) for i in range(1, 6)] and len({tuple(x) for x in p}) > 1
    log = r.run(1)
    assert log[0]["aggregated"] == p[0]


def test_zero_step_client_is_excluded(synth):
    names = sorted(s.name for s in arms.streams(synth.out, "fedavg_geo"))
    steps = {n: 2 for n in names}
    steps[names[0]] = 0
    for a in ("fednova", "scaffold", "fedavg"):
        r = FederatedRun(make_config(algorithm=a, arm="fedavg_geo", local_steps=steps, batch_size=8), synth.out)
        log = r.run(1)
        assert names[0] not in log[0]["aggregated"] and len(log[0]["aggregated"]) == 3
        if a == "scaffold":
            assert float(r.alg.state[f"c_{names[0]}"][next(iter(r.alg.keys))].abs().sum()) == 0


# ---- L: reproducibility and resume ----

@pytest.mark.parametrize("algorithm", ["fedadam", "scaffold", "fednova", "fedavgm"])
def test_same_seed_repeats_and_resume_equals_uninterrupted(synth, tmp_path, algorithm):
    cfg = make_config(algorithm=algorithm, model="gru", local_steps=2, batch_size=8)
    a = FederatedRun(cfg, synth.out); a.run(3)
    b = FederatedRun(cfg, synth.out); b.run(3)
    assert all(torch.equal(a.model.state_dict()[k], b.model.state_dict()[k]) for k in a.model.state_dict())
    c = FederatedRun(cfg, synth.out); c.run(1)
    p = tmp_path / "ck.pt"
    c.save(p)
    with pytest.raises(FileExistsError):
        c.save(p)
    d = FederatedRun(cfg, synth.out)
    d.resume(p)
    d.run(2)
    assert d.round == 3
    assert all(torch.equal(a.model.state_dict()[k], d.model.state_dict()[k]) for k in a.model.state_dict())
    for k, v in a.alg.state.items():
        assert all(torch.equal(v[n], d.alg.state[k][n]) for n in v)
    other = FederatedRun(make_config(algorithm=algorithm, model="gru", local_steps=3, batch_size=8), synth.out)
    with pytest.raises(ValueError):
        other.resume(p)
    e = FederatedRun(make_config(algorithm=algorithm, model="gru", local_steps=2, batch_size=8, seed=5), synth.out); e.run(1)
    assert any(not torch.equal(e.model.state_dict()[k], c.model.state_dict()[k]) for k in e.model.state_dict())


# ---- M: communication accounting ----

def test_communication_accounting(synth):
    logs = {}
    for a in ("fedavg", "scaffold", "fednova"):
        r = FederatedRun(make_config(algorithm=a, local_steps=1, batch_size=8), synth.out)
        logs[a] = r.run(1)[0]
        n = sum(v.numel() * v.element_size() for v in r.model.state_dict().values() if v.is_floating_point())
        per = next(iter(logs[a]["clients"].values()))
        if a == "fedavg":
            assert per["download_bytes"] == n and per["upload_bytes"] == n + 8
        elif a == "scaffold":
            assert per["download_bytes"] == 2 * n and per["upload_bytes"] == 2 * n + 8
        else:
            assert per["download_bytes"] == n and per["upload_bytes"] == n + 16
        assert logs[a]["total_bytes"] == sum(c["download_bytes"] + c["upload_bytes"] for c in logs[a]["clients"].values())


# ---- real data (skipped until the Phase 2 outputs exist) ----

REAL = core.load_config()
HAS = (REAL["_out"] / "phase3" / "phase3_validation.json").is_file()


@pytest.mark.skipif(not HAS, reason="Phase 2/3 outputs not generated")
@pytest.mark.parametrize("arm", ["fedavg_iid", "fedavg_geo"])
@pytest.mark.parametrize("history", ["A", "B"])
def test_real_one_round(arm, history):
    r = FederatedRun(make_config(algorithm="scaffold", arm=arm, history=history, model="patchtst", local_steps=2, batch_size=16))
    log = r.run(1)
    assert len(log[0]["aggregated"]) == 4 and all(torch.isfinite(v).all() for v in r.model.state_dict().values())


def test_new_fl_code_is_portable():
    for f in (core.SRC / "aiori2" / "fl").rglob("*.py"):
        t = f.read_text(encoding="utf-8")
        assert "wintypes" not in t and "powershell" not in t.lower(), f
