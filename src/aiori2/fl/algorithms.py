"""Server-side rules of the eight federated algorithms (single sign convention, see aiori2/fl/README.md).

Notation: global parameters w (dict name -> tensor), client i returns Delta_i = w_i - w (new minus old) after tau_i
local optimizer steps; p_i are the participants' normalised weights. All server arithmetic is float64 and the result
is cast back to the parameter dtype. Server state persists across rounds and round-trips through state_dict().
"""

import math

import torch

REGISTRY = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def _check(tensors, what):
    for k, v in tensors.items():
        if not torch.isfinite(v).all():
            raise FloatingPointError(f"non-finite {what} for {k}")


def weighted_sum(dicts, weights):
    """sum_i p_i * d_i in float64, with p normalised; rejects zero total weight, shape mismatch and NaN/Inf."""
    w = torch.as_tensor(weights, dtype=torch.float64)
    if len(dicts) != len(w) or len(dicts) == 0:
        raise ValueError("need one weight per client update")
    if (w < 0).any() or float(w.sum()) <= 0:
        raise ValueError("client weights must be non-negative with a positive total")
    w = w / w.sum()
    keys = list(dicts[0])
    out = {}
    for k in keys:
        if any(list(d) != keys for d in dicts):
            raise ValueError("client updates have different parameter names")
        ref = dicts[0][k]
        if any(d[k].shape != ref.shape for d in dicts):
            raise ValueError(f"shape mismatch for {k}")
        out[k] = sum(float(wi) * d[k].double() for wi, d in zip(w, dicts))
    _check(out, "aggregate")
    return out


class Algorithm:
    """Base: FedAvg-style weighted delta, server step w <- w + server_lr * Delta."""
    name = "base"
    client_optimizers = ("sgd", "adam")
    uses_prox = False
    scaffold = False
    nova = False
    version = "1"

    def __init__(self, cfg, template):
        self.cfg = cfg
        self.keys = [k for k, v in template.items() if v.is_floating_point()]
        self.state = {}

    def zeros(self, template):
        return {k: torch.zeros_like(template[k], dtype=torch.float64) for k in self.keys}

    def aggregate(self, results, weights):
        return weighted_sum([r["delta"] for r in results], weights)

    def server_step(self, w, delta):
        return {k: w[k] + self.cfg["server_lr"] * delta[k] for k in self.keys}

    def update(self, global_state, results, weights):
        _check({f"{r['client']}:{k}": v for r in results for k, v in r["delta"].items()}, "client delta")
        delta = self.aggregate(results, weights)
        w = {k: global_state[k].double() for k in self.keys}
        new = self.server_step(w, delta)
        _check(new, "global parameters")
        out = dict(global_state)
        for k in self.keys:
            out[k] = new[k].to(global_state[k].dtype)
        return out

    def state_dict(self):
        return {k: {n: t.clone() for n, t in v.items()} if isinstance(v, dict) else v for k, v in self.state.items()}

    def load_state_dict(self, st):
        self.state = {k: {n: t.clone().double() for n, t in v.items()} if isinstance(v, dict) else v for k, v in st.items()}


@register
class FedAvg(Algorithm):
    """McMahan et al. (2017): w <- w + eta_s * sum_i p_i Delta_i (eta_s = 1: weighted parameter average)."""
    name = "fedavg"


@register
class FedProx(Algorithm):
    """Li et al. (2020): local objective L_i(w) + mu/2 ||w - w_round_start||^2; aggregation as FedAvg."""
    name = "fedprox"
    uses_prox = True


@register
class FedAvgM(Algorithm):
    """Hsu et al. (2019) server momentum on the pseudo-gradient g = -Delta: v <- beta v + g; w <- w - eta_s v."""
    name = "fedavgm"

    def server_step(self, w, delta):
        v = self.state.setdefault("momentum", self.zeros(w))
        b = self.cfg["server_momentum"]
        for k in self.keys:
            v[k] = b * v[k] - delta[k]
        return {k: w[k] - self.cfg["server_lr"] * v[k] for k in self.keys}


class FedOpt(Algorithm):
    """Reddi et al. (2021), Adaptive Federated Optimization, Algorithm 2 (no bias correction):
    m <- b1 m + (1-b1) Delta;  v <- rule(v, Delta^2) with v0 = tau^2;  w <- w + eta_s m / (sqrt(v) + tau)."""

    def second_moment(self, v, d2):
        raise NotImplementedError

    def server_step(self, w, delta):
        b1, tau = self.cfg["beta1"], self.cfg["tau"]
        m = self.state.setdefault("m", self.zeros(w))
        if "v" not in self.state:
            self.state["v"] = {k: torch.full_like(w[k], tau ** 2) for k in self.keys}
        v = self.state["v"]
        out = {}
        for k in self.keys:
            m[k] = b1 * m[k] + (1 - b1) * delta[k]
            v[k] = self.second_moment(v[k], delta[k] * delta[k])
            out[k] = w[k] + self.cfg["server_lr"] * m[k] / (v[k].sqrt() + tau)
        return out


@register
class FedAdam(FedOpt):
    name = "fedadam"

    def second_moment(self, v, d2):
        b2 = self.cfg["beta2"]
        return b2 * v + (1 - b2) * d2


@register
class FedYogi(FedOpt):
    name = "fedyogi"

    def second_moment(self, v, d2):
        return v - (1 - self.cfg["beta2"]) * d2 * torch.sign(v - d2)


@register
class FedAdagrad(FedOpt):
    name = "fedadagrad"

    def second_moment(self, v, d2):
        return v + d2


@register
class Scaffold(Algorithm):
    """Karimireddy et al. (2020), SCAFFOLD with control-variate Option II, local plain SGD:
    local step  y <- y - eta_l (g_i(y) + c - c_i)
    after K_i steps  c_i+ = c_i - c + (w - y_i) / (K_i eta_l);  Delta_i = y_i - w;  dc_i = c_i+ - c_i
    server  w <- w + eta_g sum_{i in S} p_i Delta_i;  c <- c + (1/N) sum_{i in S} dc_i   (N = all clients).
    Clients with K_i = 0 do not change c_i and are not aggregated."""
    name = "scaffold"
    client_optimizers = ("sgd",)
    scaffold = True

    def __init__(self, cfg, template, clients=()):
        super().__init__(cfg, template)
        self.state = {"c": self.zeros(template)}
        for c in clients:
            self.state[f"c_{c}"] = self.zeros(template)
        self.n_clients = len(clients)

    def correction(self, client):
        c, ci = self.state["c"], self.state[f"c_{client}"]
        return {k: (c[k] - ci[k]) for k in self.keys}

    def client_control(self, client, delta, steps, lr):
        """Option II update; returns dc_i and stores c_i+ (only for clients that completed >= 1 step)."""
        c, ci = self.state["c"], self.state[f"c_{client}"]
        new = {k: ci[k] - c[k] - delta[k].double() / (steps * lr) for k in self.keys}
        dc = {k: new[k] - ci[k] for k in self.keys}
        self.state[f"c_{client}"] = new
        return dc

    def update(self, global_state, results, weights):
        out = super().update(global_state, results, weights)
        c = self.state["c"]
        for r in results:
            for k in self.keys:
                c[k] = c[k] + r["dc"][k] / self.n_clients
        _check(c, "global control variate")
        return out


@register
class FedNova(Algorithm):
    """Wang et al. (2020), FedNova with local SGD (optional heavy-ball momentum rho, no proximal term):
    ||a_i||_1 = tau_i (rho = 0) or [tau_i - rho (1 - rho^tau_i) / (1 - rho)] / (1 - rho);
    d_i = Delta_i / ||a_i||_1 (normalised update);  tau_eff = sum_i p_i ||a_i||_1;
    w <- w + eta_s * tau_eff * sum_i p_i d_i.  Equal tau_i and rho = 0 give exactly FedAvg."""
    name = "fednova"
    client_optimizers = ("sgd",)
    nova = True

    @staticmethod
    def a_norm(steps, rho):
        if steps <= 0:
            return 0.0
        if rho == 0:
            return float(steps)
        return (steps - rho * (1 - rho ** steps) / (1 - rho)) / (1 - rho)

    def aggregate(self, results, weights):
        a = [self.a_norm(r["steps"], self.cfg["client_momentum"]) for r in results]
        if min(a) <= 0:
            raise ValueError("FedNova received a client with zero local steps")
        norm = [{k: v / ai for k, v in r["delta"].items()} for r, ai in zip(results, a)]
        w = torch.as_tensor(weights, dtype=torch.float64)
        tau_eff = float((w / w.sum() * torch.as_tensor(a, dtype=torch.float64)).sum())
        self.last_tau_eff = tau_eff
        d = weighted_sum(norm, weights)
        return {k: tau_eff * v for k, v in d.items()}


def make(name, cfg, template, clients=()):
    cls = REGISTRY[name]
    return cls(cfg, template, clients) if cls.scaffold else cls(cfg, template)
