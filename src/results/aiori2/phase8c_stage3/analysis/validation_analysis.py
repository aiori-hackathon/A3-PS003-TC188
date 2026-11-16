"""Phase 8C Stage 3 validation-only analysis (no test data is read here).

Run from src/:  ../.venv-gpu/Scripts/python.exe results/aiori2/phase8c_stage3/analysis/validation_analysis.py
Writes stage3_runs.csv, stage3_cells.csv, stage3_configs.csv, stage3_validation_summary.json, selection.json and
plots into this folder. Ranking and test-selection follow stage3_protocol.json exactly.
"""
import csv
import json
import statistics as st
from pathlib import Path

from svgplot import Plot

HERE = Path(__file__).resolve().parent
S3 = HERE.parent
S1 = S3.parent / "phase7_stage1"
PROTO = json.loads((S3 / "stage3_protocol.json").read_text(encoding="utf-8"))
CFG = json.loads((S3.parents[2] / "aiori2_watchdog" / "config.json").read_text(encoding="utf-8"))
CLIENTS = ["client_0_la_ventura", "client_1_bay_area", "client_2_coastal_south_central", "client_3_inland_central"]


def load(d):
    return json.loads((d / "run.json").read_text(encoding="utf-8"))


rows = []
for rid in CFG["expected_run_ids"]:
    r = load(S3 / rid)
    c, b = r["config"], r["best"]
    assert r["status"] in ("completed", "early_stopped") and r["run_id"] == rid
    pc = b["val_full_by_client_geo"]
    rows.append(dict(run_id=rid, model=c["model"], history=c["history"], algorithm=c["algorithm"],
                     partition="IID" if c["arm"] == "fedavg_iid" else "geo", seed=c["seed"], status=r["status"],
                     epochs_run=len(r["history"]), best_epoch=b["epoch"], val_full_wape=b["val_full"]["wape_pct"],
                     val_full_mae=b["val_full"]["mae"], val_full_rmse=b["val_full"]["rmse"],
                     **{f"c{i}_wape": pc[k] for i, k in enumerate(CLIENTS)}, worst_client_wape=max(pc.values()),
                     train_minutes=round(r["counters"]["train_seconds"] / 60, 2), resumes=len(r.get("resumes") or []),
                     code_sha8=r["code_sha256"][:8], subset_curve=[h["val"]["wape_pct"] for h in r["history"]],
                     loss_curve=[h["train_loss"] for h in r["history"]]))

cfg_key = lambda x: (x["model"], x["algorithm"], x["history"])
cells, configs = {}, {}
for x in rows:
    cells.setdefault(cfg_key(x) + (x["partition"],), []).append(x)
cell_rows = []
for k, xs in cells.items():
    w = [x["val_full_wape"] for x in xs]
    cell_rows.append(dict(model=k[0], algorithm=k[1], history=k[2], partition=k[3], n=len(w), mean_wape=st.mean(w),
                          sd_wape=st.stdev(w), min_wape=min(w), max_wape=max(w),
                          mean_mae=st.mean(x["val_full_mae"] for x in xs), mean_rmse=st.mean(x["val_full_rmse"] for x in xs),
                          mean_worst_client=st.mean(x["worst_client_wape"] for x in xs),
                          **{f"c{i}_mean": st.mean(x[f"c{i}_wape"] for x in xs) for i in range(4)},
                          early_stopped=sum(x["status"] == "early_stopped" for x in xs)))
for c in cell_rows:
    configs.setdefault((c["model"], c["algorithm"], c["history"]), {})[c["partition"]] = c
cfg_rows = []
for k, d in configs.items():
    six = [x["val_full_wape"] for x in rows if cfg_key(x) == k]
    cfg_rows.append(dict(model=k[0], algorithm=k[1], history=k[2], score=(d["IID"]["mean_wape"] + d["geo"]["mean_wape"]) / 2,
                         iid_mean=d["IID"]["mean_wape"], geo_mean=d["geo"]["mean_wape"], geo_minus_iid=d["geo"]["mean_wape"] - d["IID"]["mean_wape"],
                         sd_6runs=st.stdev(six), pooled_within_cell_sd=(st.mean([d["IID"]["sd_wape"] ** 2, d["geo"]["sd_wape"] ** 2])) ** 0.5))
cfg_rows.sort(key=lambda r: r["score"])
for i, r in enumerate(cfg_rows, 1):
    r["rank"] = i

# protocol: runner-up tested if within one pooled seed SD of the winner.
# pooled seed SD = sqrt(mean within-cell variance) over all 8 cells (pre-stated reading); alternatives reported.
pooled_all = st.mean(c["sd_wape"] ** 2 for c in cell_rows) ** 0.5
win, ru = cfg_rows[0], cfg_rows[1]
gap = ru["score"] - win["score"]
alts = {"pooled_all_8_cells": pooled_all,
        "pooled_winner_and_runnerup_cells": st.mean([win["pooled_within_cell_sd"] ** 2, ru["pooled_within_cell_sd"] ** 2]) ** 0.5,
        "winner_cells_only": win["pooled_within_cell_sd"]}
include_ru = gap <= pooled_all

# history-matched model comparison from Stage 2 (10 epochs, seed 20261027, FedAvgM): MLP vs N-HiTS with the same history
s2 = {r["run_id"]: r for r in csv.DictReader(open(S1 / "stage2_results.csv", encoding="utf-8"))}
matched = []
for h in ("A", "B"):
    for part in ("IID", "geo"):
        pick = {m: next((float(r["val_full_wape_pct"]) for r in s2.values() if r["model"] == m and r["history"] == h
                         and r["partition"] == part and r["algorithm"] == "fedavgm"), None) for m in ("mlp", "nhits")}
        matched.append(dict(history=h, partition=part, mlp=pick["mlp"], nhits=pick["nhits"],
                            mlp_minus_nhits=None if None in pick.values() else pick["mlp"] - pick["nhits"]))

# 10-epoch Stage 2 references for the same 8 finalist cells (never pooled with 20-epoch runs)
s2_fin = []
for rid in PROTO["finalist_source_run_ids"].split(","):
    r = s2.get(rid) or {}
    s2_fin.append(dict(run_id=rid, model=r.get("model"), algorithm=r.get("algorithm"), history=r.get("history"),
                       partition=r.get("partition"), val_full_wape_10ep=float(r["val_full_wape_pct"]) if r else None))

# algorithm contrasts on N-HiTS B, matched by partition and seed
alg = {}
for part in ("IID", "geo"):
    for a in ("fedavgm", "fedadam"):
        d = []
        for s in PROTO["seeds"]:
            g = lambda alg_: next(x["val_full_wape"] for x in rows if x["model"] == "nhits" and x["history"] == "B"
                                  and x["algorithm"] == alg_ and x["partition"] == part and x["seed"] == s)
            d.append(g(a) - g("fedavg"))
        alg[f"{a}_minus_fedavg_{part}"] = dict(per_seed=d, mean=st.mean(d), min=min(d), max=max(d))

# Stage 1 references for the test (10 epochs, single seed, unmatched exposure)
s1_ref = {}
for name in sorted(p.name for p in S1.iterdir() if p.is_dir() and not p.name.startswith(("fedavg_", "plan"))):
    r = load(S1 / name)
    if r.get("best") and r["best"].get("val_full"):
        s1_ref[name] = r["best"]["val_full"]["wape_pct"]
best_s1 = min(s1_ref, key=s1_ref.get)
wm, wh = win["model"], win["history"]
s1_pick = [best_s1] + [n for n in s1_ref if n.startswith((f"centralised-{wh}-{wm}-", f"local-{wh}-{wm}-"))]

selection_models = []
test_cfgs = [(win["model"], win["algorithm"], win["history"], "winner")]
if include_ru:
    test_cfgs.append((ru["model"], ru["algorithm"], ru["history"], "runner-up"))
test_cfgs.append(("nhits", "fedavg", "B", "baseline"))
for m, a, h, role in test_cfgs:
    for x in rows:
        if cfg_key(x) == (m, a, h):
            selection_models.append(dict(label=f"{role}:{m}-{a}-{h}-{x['partition']}-s{x['seed']}", run_dir=str(S3 / x["run_id"])))
for n in dict.fromkeys(s1_pick):
    selection_models.append(dict(label=f"stage1-ref-10ep-single-seed:{n}", run_dir=str(S1 / n)))
selection = dict(models=selection_models, persistence=["A", "B"],
                 policy=dict(protocol_sha256=CFG["protocol_sha256"], rule=PROTO["test_selection_rule"],
                             winner=win, runner_up=ru, runner_up_gap=gap, pooled_seed_sd=pooled_all,
                             pooled_sd_alternatives=alts, runner_up_included=include_ru,
                             note="selection decided from validation only; no test data read before freezing"))

# ---- outputs
def wcsv(path, rs, drop=()):
    keys = [k for k in rs[0] if k not in drop]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader(); w.writerows(rs)

wcsv(HERE / "stage3_runs.csv", rows, drop=("subset_curve", "loss_curve"))
wcsv(HERE / "stage3_cells.csv", sorted(cell_rows, key=lambda c: c["mean_wape"]))
wcsv(HERE / "stage3_configs.csv", cfg_rows)
summary = dict(runs=len(rows), configs=cfg_rows, cells=cell_rows, algorithm_contrasts_nhits_B=alg,
               history_matched_model_pairs_stage2_10ep=matched, stage2_10ep_finalists=s2_fin,
               stage1_refs_val_full=s1_ref, best_stage1=best_s1, selection_policy=selection["policy"],
               early_stopped=[x["run_id"] for x in rows if x["status"] == "early_stopped"],
               resumed=[x["run_id"] for x in rows if x["resumes"]],
               curves={x["run_id"]: dict(subset_val_wape=x["subset_curve"], train_loss=x["loss_curve"]) for x in rows})
(HERE / "stage3_validation_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
(HERE / "selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")

# plots (plain SVG; matplotlib is not installed in the project venvs)
colors = {("mlp", "fedavgm"): "#1b7837", ("nhits", "fedavgm"): "#2166ac", ("nhits", "fedadam"): "#b35806", ("nhits", "fedavg"): "#6a6a6a"}
order = sorted(cell_rows, key=lambda c: c["mean_wape"])
lab = [f"{c['model']} {c['algorithm']} {c['history']} {c['partition']}" for c in order]
p = Plot("Stage 3 cells: 3 seeds x 20 epochs (dots = seeds, bar = mean +/- SD)", "Full-validation WAPE (%)", len(order), 10.75, 11.12, lab)
for i, c in enumerate(order):
    col = colors[(c["model"], c["algorithm"])]
    for x in cells[(c["model"], c["algorithm"], c["history"], c["partition"])]:
        p.dot(i, x["val_full_wape"], col, 3, .55)
    p.errbar(i, c["mean_wape"], c["sd_wape"], col)
p.save(HERE / "plot_cells_seeds.svg")

for part in ("IID", "geo"):
    p = Plot(f"Validation-subset WAPE by epoch ({part}); dotted line = 10-epoch Stage 2 budget", "subset WAPE (%)", 20, 10.75, 11.6,
             [str(e) for e in range(1, 21)], legend={f"{k[0]} {k[1]}": v for k, v in colors.items()})
    p.vline(9, "#000")
    for x in rows:
        if x["partition"] == part:
            p.line(list(range(len(x["subset_curve"]))), x["subset_curve"], colors[(x["model"], x["algorithm"])])
    p.save(HERE / f"plot_convergence_{part}.svg")

p = Plot("Per geographic client WAPE, mean of 3 seeds", "WAPE (%)", len(order), 8.5, 13.5, lab,
         legend={c: k for c, k in zip(CLIENTS, ("#4477aa", "#66ccee", "#228833", "#ccbb44"))})
for i, c in enumerate(order):
    for j, col in enumerate(("#4477aa", "#66ccee", "#228833", "#ccbb44")):
        p.bar(i + (j - 1.5) * .2, c[f"c{j}_mean"], .18, col)
p.save(HERE / "plot_per_client.svg")

for r in cfg_rows:
    print(f"{r['rank']} {r['model']:6s} {r['algorithm']:8s} {r['history']} score {r['score']:.4f} IID {r['iid_mean']:.4f} geo {r['geo_mean']:.4f} sd6 {r['sd_6runs']:.4f}")
print(f"gap {gap:.4f} pooled_sd {pooled_all:.4f} alts {({k: round(v, 4) for k, v in alts.items()})} runner-up included: {include_ru}")
print("alg", {k: round(v["mean"], 4) for k, v in alg.items()})
print("matched", matched)
print("best stage1", best_s1, s1_ref[best_s1], "refs", s1_pick)
print("selection models", len(selection_models))
