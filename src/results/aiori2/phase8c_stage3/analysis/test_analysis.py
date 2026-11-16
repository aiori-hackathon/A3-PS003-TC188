"""Summarise the one-time frozen test evaluation (reads final_test_results.json only; evaluates nothing).

Run from src/:  ../.venv-gpu/Scripts/python.exe results/aiori2/phase8c_stage3/analysis/test_analysis.py
"""
import csv
import json
import statistics as st
from pathlib import Path

from svgplot import Plot

HERE = Path(__file__).resolve().parent
res = json.loads((HERE / "final_test_results.json").read_text(encoding="utf-8"))
man = json.loads((HERE / "final_manifest.json").read_text(encoding="utf-8"))
val = {m["run_id"]: m["val_full"]["wape_pct"] for m in man["models"]}
assert res["split"] == "test" and res["same_target_set"]

rows = []
for m in res["models"] + res["persistence"]:
    geo = m["by_client_geo"]
    rows.append(dict(label=m["label"], run_id=m.get("run_id", ""), test_wape=m["wape_pct"], test_mae=m["mae"], test_rmse=m["rmse"],
                     val_wape=val.get(m.get("run_id")), n_valid=m["n_valid"],
                     **{f"c{i}_wape": geo[k]["wape_pct"] for i, k in enumerate(sorted(geo))},
                     worst_client_wape=max(v["wape_pct"] for v in geo.values()),
                     h1_wape=m["by_horizon"][0]["wape_pct"] if "wape_pct" in m["by_horizon"][0] else None,
                     h12_wape=m["by_horizon"][-1]["wape_pct"] if "wape_pct" in m["by_horizon"][-1] else None))
with open(HERE / "final_test_table.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)

def group(prefix, part):
    return [r for r in rows if r["label"].startswith(prefix) and f"-{part}-" in r["label"]]

summary = {}
for role in ("winner", "baseline"):
    for part in ("IID", "geo"):
        g = group(role, part)
        t = [r["test_wape"] for r in g]
        summary[f"{role}_{part}"] = dict(n=len(t), mean=st.mean(t), sd=st.stdev(t), min=min(t), max=max(t),
                                         mae=st.mean(r["test_mae"] for r in g), rmse=st.mean(r["test_rmse"] for r in g),
                                         worst_client=st.mean(r["worst_client_wape"] for r in g),
                                         val_mean=st.mean(r["val_wape"] for r in g))
paired = {}
for part in ("IID", "geo"):
    d = [a["test_wape"] - b["test_wape"] for a, b in zip(group("winner", part), group("baseline", part))]   # same seed order
    paired[part] = dict(per_seed=d, mean=st.mean(d), min=min(d), max=max(d))
pers = next(r["test_wape"] for r in rows if r["label"] == "persistence-A")
out = dict(manifest_sha256=res["manifest_sha256"], same_target_set=res["same_target_set"], n_valid=rows[0]["n_valid"],
           groups=summary, winner_minus_baseline_by_seed=paired, persistence_wape=pers,
           stage1_refs={r["label"].split(":", 1)[1]: r["test_wape"] for r in rows if r["label"].startswith("stage1")},
           val_to_test_shift={k: v["mean"] - v["val_mean"] for k, v in summary.items()},
           note="test evaluated once from the frozen manifest; no selection, retraining or re-evaluation follows")
(HERE / "final_test_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

lab = [r["label"].replace("stage1-ref-10ep-single-seed:", "S1 ").split("-s2026")[0].replace("-medium-none", "") for r in rows]
p = Plot("One-time test (Jul-Dec 2024): WAPE per frozen model", "Test WAPE (%)", len(rows), 10.0, 12.6, lab)
for i, r in enumerate(rows):
    col = "#1b7837" if r["label"].startswith("winner") else "#6a6a6a" if r["label"].startswith("baseline") else "#7b3294" if r["label"].startswith("stage1") else "#b2182b"
    p.bar(i, r["test_wape"], .6, col)
p.save(HERE / "plot_test_wape.svg")

for k, v in summary.items():
    print(f"{k:14s} test {v['mean']:.4f} sd {v['sd']:.4f} [{v['min']:.4f},{v['max']:.4f}] mae {v['mae']:.3f} rmse {v['rmse']:.3f} worst {v['worst_client']:.3f} val {v['val_mean']:.4f}")
print("paired winner-baseline", {k: (round(v["mean"], 4), round(v["min"], 4), round(v["max"], 4)) for k, v in paired.items()})
print("persistence", pers, "stage1", out["stage1_refs"], "n_valid", out["n_valid"])
for r in rows:
    print(r["label"][:55], round(r["test_wape"], 4), [round(r[f"c{i}_wape"], 3) for i in range(4)], r["h1_wape"] and round(r["h1_wape"], 3), r["h12_wape"] and round(r["h12_wape"], 3))
