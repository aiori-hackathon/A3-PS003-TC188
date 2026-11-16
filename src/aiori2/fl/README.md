# aiori2.fl — eight federated algorithms (AIORI 2.0, Phase 5)

Single-process simulation of the four clients of the **IID** (`fedavg_iid`) or **Non-IID geographic**
(`fedavg_geo`) arm. Centralized and Local are rejected by the configuration validator. Clients are the Phase 3
streams (4 × 50 sensors, unchanged partitions); each trains only on its own memory-mapped training windows with its
own scaler. The server receives parameter deltas (and, for SCAFFOLD, control-variate deltas) — never data.

```python
from aiori2.fl import FederatedRun, make_config
cfg = make_config(algorithm="scaffold", model="patchtst", arm="fedavg_geo", history="B", local_steps=10, batch_size=512)
run = FederatedRun(cfg); run.run(rounds=3); run.save("results/aiori2/<new-dir>/round3.pt")
```
Defaults: `aiori2/configs/fl/defaults.json` (mechanics only, untuned). Smoke test: `python -m aiori2.fl.smoke --device cpu|cuda`.

## Conventions (one for all algorithms)

| Symbol | Meaning |
|---|---|
| `w` | global parameters at the start of round t (every participant loads exactly these) |
| `w_i` | client i's parameters after its local steps; `Δ_i = w_i − w` (new minus old) |
| `τ_i` | optimizer steps client i actually took (batches with no valid target are skipped and not counted) |
| `n_i`, `p_i` | client weight and its normalisation over the round's aggregated clients, `p_i = n_i / Σ n_j` |
| `η_l`, `η_s` | client learning rate (`client_lr`), server learning rate (`server_lr`) |
| `g = −Δ` | server pseudo-gradient (descent direction) where an algorithm needs one |

Weighting (`weighting`, same for every algorithm in a comparison): **`train_windows`** (default; the client's
number of training windows with ≥ 1 valid target, i.e. its dataset size as in FedAvg), `valid_targets_used`
(valid target elements in this round's batches) or `uniform`. Clients are not assumed equal: the four clients'
window counts differ slightly. Order per round: select participants → local training → server update → next round.
Clients with τ_i = 0 are not aggregated. Client optimizers are created fresh every round and never shared.

| Algorithm | Server rule (Δ = Σ p_i Δ_i unless stated) | State | Restrictions | Reference |
|---|---|---|---|---|
| FedAvg | `w ← w + η_s Δ` (η_s = 1: weighted parameter average) | — | SGD or Adam client | McMahan et al., AISTATS 2017 |
| FedProx | as FedAvg; local loss `L_i(w_i) + μ/2 ‖w_i − w‖²` over trainable parameters, `w` fixed for the round | — | μ only here | Li et al., MLSys 2020 |
| FedAvgM | `v ← β v + g`, `w ← w − η_s v` | `v` | — | Hsu, Qi & Brown, arXiv:1909.06335 (2019) |
| FedAdam | `m ← β1 m + (1−β1)Δ`, `v ← β2 v + (1−β2)Δ²`, `w ← w + η_s m/(√v + τ)` | `m`, `v` (v0 = τ²) | no bias correction | Reddi et al., ICLR 2021, Alg. 2 |
| FedYogi | as FedAdam with `v ← v − (1−β2) Δ² sign(v − Δ²)` | `m`, `v` | | Reddi et al. 2021 |
| FedAdagrad | as FedAdam with `v ← v + Δ²` | `m`, `v` | | Reddi et al. 2021 |
| SCAFFOLD | local `y ← y − η_l (∇f_i(y) + c − c_i)`; `c_i⁺ = c_i − c + (w − y_i)/(τ_i η_l)` (**Option II**); `w ← w + η_s Σ p_i Δ_i`; `c ← c + (1/N) Σ_{i∈S} (c_i⁺ − c_i)` | `c`, `c_i` per client name | plain SGD (no momentum), no μ | Karimireddy et al., ICML 2020 |
| FedNova | `‖a_i‖₁ = τ_i` (ρ = 0) or `[τ_i − ρ(1−ρ^τ_i)/(1−ρ)]/(1−ρ)`; `τ_eff = Σ p_i ‖a_i‖₁`; `w ← w + η_s τ_eff Σ p_i Δ_i / ‖a_i‖₁` | — | SGD with optional heavy-ball momentum ρ (PyTorch SGD, dampening 0); no μ | Wang et al., NeurIPS 2020 |

Citations are from memory of the published papers; page/equation numbers were not re-checked online.

## Notes

- Server arithmetic runs in float64 on the CPU; parameters are cast back to their dtype (float32).
- The legacy `aopt.train.Server` uses the same FedAvgM/FedOpt equations (pseudo-gradient sign, v0 = τ², no bias
  correction); it was not imported because it is tied to the aopt data store and keeps persistent client Adam state.
  Neither the legacy code nor the study's `aiori.federated` FedAvg was changed.
- **Unused parameters** (requires_grad but grad None, e.g. the last N-HiTS block's backcast): their true gradient is
  zero, so SGD leaves them unchanged; FedProx includes them in the penalty (zero gradient while they stay at `w`);
  SCAFFOLD applies only the correction term (with c, c_i starting at 0 these stay 0). Frozen parameters
  (requires_grad False) are never updated or penalised. Optimizer steps are counted per optimizer call, not per
  gradient. No architecture change was needed.
- Gradient clipping (`grad_clip`) applies to the data (+ proximal) gradient; SCAFFOLD adds its correction after
  clipping, and Option II's `c_i⁺` uses the actual movement, so it stays consistent.
- Determinism: client selection, batch sampling and per-client torch seeds derive from (seed, round, client), so a
  resumed run repeats an uninterrupted one on CPU. CUDA kernels (e.g. GRU, attention) may not be bitwise deterministic,
  and results are not claimed identical across GPUs or PyTorch versions.
- Checkpoints (`FederatedRun.save/resume`): global parameters, server state (momentum, m, v, c, every c_i), round,
  config + hash, client list, model config, data-manifest hash and the round log; tensors + JSON only
  (`weights_only=True`), atomic, never overwritten; resume rejects a different config or client set.
- Communication (per round, logical payload, no network measured): download = parameters (+ c for SCAFFOLD); upload =
  Δ_i + one float64 weight (+ Δc_i for SCAFFOLD, + ‖a_i‖₁ for FedNova). Checkpoint writes are not counted.
