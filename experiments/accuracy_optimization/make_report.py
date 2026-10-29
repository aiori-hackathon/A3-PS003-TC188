"""Ranked, aggregate-only results table from the experiment registry (no station ids, rows or paths).

    .venv-gpu\\Scripts\\python.exe experiments/accuracy_optimization/make_report.py
Writes docs/accuracy_optimization_results.json (machine-readable) and prints a Markdown table.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "accuracy_optimization"
BASE = 8.2226  # reproduced frozen local_seed1 (clipped) validation WAPE


def features(c):
    f = [f"L{c.get('input_len', 12)}"]
    f += [n for n, k in (("calendar", "future_cal"), ("profile", "profile"), ("lags 1d/1w", "lags")) if c.get(k)]
    if c.get("emb_dim"):
        f.append("sensor emb")
    return ", ".join(f)


def main():
    rows = []
    for line in (OUT / "registry.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("status") != "completed":
            continue
        v, c = e["validation"], e.get("config", {})
        rows.append(dict(exp_id=e["exp_id"], algorithm=e["algorithm"], architecture=e.get("architecture"),
                         input=features(c) if "input_len" in c else "-", loss=c.get("loss", "-"),
                         val_wape_pct=round(v["wape_pct"], 3), val_mae=round(v["mae"], 3), val_rmse=round(v["rmse"], 3),
                         val_smape_pct=round(v["smape_pct"], 3), rel_improvement_pct=round(100 * (BASE - v["wape_pct"]) / BASE, 1),
                         runtime_min=round(e.get("runtime_s", 0) / 60, 1) if e.get("runtime_s") else None,
                         by_client_wape_pct={k: round(x["wape_pct"], 2) for k, x in v["by_client"].items()},
                         evaluation=c.get("evaluation", "validation, single fit")))
    rows.sort(key=lambda r: r["val_wape_pct"])
    (ROOT / "docs" / "accuracy_optimization_results.json").write_text(json.dumps(dict(baseline_validation_wape_pct=BASE, ranked=rows), indent=1))
    print("| # | Experiment | Algorithm | Arch | Inputs | Loss | Val WAPE | Val MAE | Val RMSE | Val sMAPE | vs baseline | Min |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        print(f"| {i} | {r['exp_id']} | {r['algorithm']} | {r['architecture']} | {r['input']} | {r['loss']} | {r['val_wape_pct']:.3f}% | "
              f"{r['val_mae']:.3f} | {r['val_rmse']:.3f} | {r['val_smape_pct']:.3f}% | {r['rel_improvement_pct']:+.1f}% | {r['runtime_min'] or ''} |")


if __name__ == "__main__":
    main()
