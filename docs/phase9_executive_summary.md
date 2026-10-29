# Phase 9 executive summary

Generated 2026-10-28T22:17:23Z. Scope: XTraffic flow, January 1 – September 30, 2023, 200 sensors in four provisional geographic clients; train Jan 1 – Jul 10, validation Jul 11 – Aug 7, held-out test Aug 8 – Sep 30 (evaluated once).

- Completed training runs: 12 of 12 planned (centralised_seed0, centralised_seed1, centralised_seed2, fedavg_geo_seed0, fedavg_geo_seed1, fedavg_geo_seed2, fedavg_iid_seed0, fedavg_iid_seed1, fedavg_iid_seed2, local_seed0, local_seed1, local_seed2).
- Best test MAE: Local-only GRU 16.990 (RMSE 28.301); persistence 24.421.
- C1 Geographic FedAvg − Centralised GRU: +0.228 MAE, 95% CI [0.147, 0.318], supported on the nine-month test split (matched seeds [0, 1, 2]).
- C2 Geographic FedAvg − Local-only GRU: +0.372 MAE, 95% CI [0.208, 0.548], supported on the nine-month test split (matched seeds [0, 1, 2]).
- C3 IID FedAvg − Geographic FedAvg: -0.088 MAE, 95% CI [-0.138, -0.043], detectable but below the practical threshold on the nine-month test split (matched seeds [0, 1, 2]).
- C4 Local-only GRU − Centralised GRU: -0.144 MAE, 95% CI [-0.275, 0.000], not distinguishable on the nine-month test split (matched seeds [0, 1, 2]).

January Phase 8 results (docs/phase8_test_results.md) are separate and unchanged.
