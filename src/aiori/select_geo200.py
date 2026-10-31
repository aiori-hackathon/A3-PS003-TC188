"""Phase 8 stratified selection of ~200 sensors (50 per geographic client).

Implements docs/phase8_200_sensor_technical_spec.md section 2 (decisions D8-1 to D8-4):
  - candidates: eligible sensors (training-period rules A1/A2, unchanged) minus the 48 pilot sensors;
  - strata: client x sensor-type group x training-mean-flow tertile within (client, type group);
  - allocation: largest remainder across type groups, k // 3 per tertile, remainder mid -> high -> low;
  - draw: sha256(f"{salt}:{array_index}") order within each stratum. Missing counts are never ranked;
  - fallbacks for short strata and a District floor, each logged.

Only the training-period columns in SELECTION_COLUMNS reach the selection; whole-month (month_*) and
validation/test (eval_*) columns are refused. Type, District and client labels come through the
provisional node_order join (row-to-ID mapping supported but unconfirmed), so strata are provisional.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aiori import config
from aiori.flow_audit import _tie_key

SALT = "aiori-geo200-v1"
PER_CLIENT = 50
TYPE_GROUP = {"Mainline": "mainline", "HOV": "hov", "On Ramp": "ramp", "Off Ramp": "ramp",
              "Fwy-Fwy": "other", "Coll/Dist": "other", "Conventional Highway": "other"}
TERTILES = ("low", "mid", "high")
REMAINDER_ORDER = ("mid", "high", "low")
# Training-period audit columns plus provisional Type. duplicate_series (training-period exact copy)
# is only a guard: eligible already excludes every copy-group member (A2).
SELECTION_COLUMNS = ("array_index", "proposed_client", "District", "eligible", "duplicate_series", "mean", "Type")
FORBIDDEN_PREFIXES = ("month_", "eval_")
MANIFEST_EXTRA = ("type_group", "flow_tertile", "stratum", "selection_rank", "selection_basis", "selection_salt")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def check_input_hashes(paths, expected):
    """Fail closed if any input differs from the SHA-256 recorded at approval time."""
    actual = {name: sha256_file(path) for name, path in paths.items()}
    bad = {n: actual[n] for n in expected if actual.get(n) != expected[n]}
    missing = sorted(set(paths) - set(expected))
    if bad or missing:
        raise ValueError(f"Selection inputs differ from the approved hashes: changed {sorted(bad)}, "
                         f"no recorded hash for {missing}")
    return actual


def check_output_dir(out_dir):
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {out} is not empty")
    return out


def selection_frame(audit, meta):
    """Allow-listed selection inputs: audit training-period columns joined with provisional Type by row."""
    if len(meta) != len(audit) or not np.array_equal(audit["array_index"].to_numpy(), np.arange(len(audit))):
        raise ValueError("Audit must have one row per array index, in order, aligned with the metadata")
    if "station_id" in meta and not np.array_equal(meta["station_id"].to_numpy(), audit["station_id"].to_numpy()):
        raise ValueError("Metadata station_id order disagrees with the audit (alignment check only)")
    frame = audit[[c for c in SELECTION_COLUMNS if c != "Type"]].copy()
    frame["Type"] = meta["Type"].to_numpy()
    return frame


def candidates(frame, exclude=()):
    """Eligible sensors not in `exclude`, with type group; checks the client and duplicate rules."""
    extra = [c for c in frame.columns if c not in SELECTION_COLUMNS]
    leaked = [c for c in frame.columns if c.startswith(FORBIDDEN_PREFIXES)]
    if extra or leaked:
        raise ValueError(f"Selection accepts only {SELECTION_COLUMNS}; got extra columns {extra + leaked}")
    eligible = frame["eligible"].astype(bool)
    if (eligible & frame["duplicate_series"].astype(bool)).any():
        raise ValueError("An exact-copy series is marked eligible (rule A2 violated)")
    clients = pd.to_numeric(frame["District"], errors="coerce").map(config.CLIENT_BY_DISTRICT)
    if not (clients.fillna("") == frame["proposed_client"].fillna("")).all():
        raise ValueError("proposed_client disagrees with CLIENT_BY_DISTRICT[District]")
    cand = frame[eligible & frame["proposed_client"].notna()
                 & ~frame["array_index"].isin(np.asarray(list(exclude), dtype=np.int64))].copy()
    unknown = sorted(set(cand["Type"].astype(str)) - set(TYPE_GROUP))
    if unknown:
        raise ValueError(f"Candidates with unmapped Type {unknown}")
    cand["type_group"] = cand["Type"].map(TYPE_GROUP)
    cand["_tie"] = cand["array_index"].map(lambda i: _tie_key(i, SALT))
    return cand


def largest_remainder(counts, total):
    """Proportional integer allocation; remainder ties: larger group first, then name."""
    names = sorted(counts)
    n = sum(counts.values())
    quota = {g: total * counts[g] / n for g in names}
    alloc = {g: int(np.floor(quota[g])) for g in names}
    order = sorted(names, key=lambda g: (-(quota[g] - alloc[g]), -counts[g], g))
    for g in order[: total - sum(alloc.values())]:
        alloc[g] += 1
    return alloc, order


def tertile_allocation(k):
    alloc = {t: k // 3 for t in TERTILES}
    for t in REMAINDER_ORDER[: k % 3]:
        alloc[t] += 1
    return alloc


def flow_tertiles(mean):
    """Training-mean-flow tertile labels; rank(method="first") then qcut, so ties are deterministic."""
    n = len(mean)
    rank = mean.rank(method="first")
    if n >= 3:
        return pd.qcut(rank, 3, labels=list(TERTILES)).astype(str)
    return pd.Series(["mid"] if n == 1 else ["low", "high"], index=rank.sort_values().index).loc[mean.index]


def _select_client(cand, per_client, log):
    """Draw one client's sensors; returns rows in draw order with stratum labels and basis."""
    counts = cand["type_group"].value_counts().to_dict()
    if sum(counts.values()) < per_client:
        raise ValueError(f"Only {sum(counts.values())} candidates for {per_client} slots")
    alloc, group_order = largest_remainder(counts, per_client)
    cand = cand.copy()
    cand["flow_tertile"] = pd.concat([flow_tertiles(g["mean"]) for _, g in cand.groupby("type_group")])
    cand = cand.sort_values("_tie")
    taken, basis, strata = [], {}, []

    def take(pool, k, label):
        got = [i for i in pool.index if i not in basis][:k]
        for i in got:
            basis[i] = label
        taken.extend(got)
        return len(got)

    for g in sorted(alloc):
        tert = tertile_allocation(alloc[g])
        short = 0
        for t in TERTILES:
            pool = cand[(cand["type_group"] == g) & (cand["flow_tertile"] == t)]
            got = take(pool, tert[t], "stratified_hash")
            strata.append(dict(type_group=g, flow_tertile=t, candidates=len(pool), allocated=tert[t], drawn=got))
            short += tert[t] - got
        for t in REMAINDER_ORDER:
            if short:
                got = take(cand[(cand["type_group"] == g) & (cand["flow_tertile"] == t)], short, "fallback_tertile")
                if got:
                    log.append(dict(rule="fallback_tertile", type_group=g, to_tertile=t, n=got))
                short -= got
        alloc[g] = alloc[g] - short                 # carried to the client-level shortfall below
    short = per_client - len(taken)
    for g in group_order:
        for t in REMAINDER_ORDER:
            if short:
                got = take(cand[(cand["type_group"] == g) & (cand["flow_tertile"] == t)], short, "fallback_type")
                if got:
                    log.append(dict(rule="fallback_type", to_type_group=g, to_tertile=t, n=got))
                short -= got
    if short:
        raise ValueError(f"Could not fill {per_client} slots")

    chosen = cand.loc[taken].copy()
    chosen["selection_basis"] = [basis[i] for i in taken]
    chosen = _district_floor(cand, chosen, log)
    return chosen, dict(type_allocation=alloc, strata=strata)


def _district_floor(cand, chosen, log):
    """A3-style floor: every District with candidates keeps >= 1 selected sensor."""
    for d in sorted(cand["District"].unique()):
        if (chosen["District"] == d).any():
            continue
        sizes = chosen.groupby(["type_group", "flow_tertile"]).size()
        largest = sorted(sizes.index, key=lambda s: (-sizes[s], s))[0]
        in_stratum = chosen[(chosen["type_group"] == largest[0]) & (chosen["flow_tertile"] == largest[1])]
        district_n = chosen["District"].value_counts()
        removable = [i for i in in_stratum.index[::-1] if district_n[chosen.at[i, "District"]] > 1]
        if not removable:
            raise ValueError(f"District floor for {d} would empty another District")
        out = removable[0]
        pool = cand[(cand["District"] == d) & ~cand.index.isin(chosen.index)]
        same = pool[pool["type_group"] == chosen.at[out, "type_group"]]
        new = (same if len(same) else pool).index[0]
        row = cand.loc[[new]].assign(selection_basis="district_floor")
        chosen = pd.concat([chosen.drop(index=out), row])
        log.append(dict(rule="district_floor", district=int(d), replaced_stratum="/".join(largest),
                        same_type_group=bool(len(same))))
    return chosen


def stratified_selection(frame, exclude=(), per_client=PER_CLIENT):
    """Select `per_client` sensors per client from an allow-listed frame. Returns (selected, summary)."""
    cand = candidates(frame, exclude)
    parts, per_client_summary, log = [], {}, []
    for client in sorted(cand["proposed_client"].unique()):
        client_log = []
        chosen, info = _select_client(cand[cand["proposed_client"] == client], per_client, client_log)
        chosen["selection_rank"] = np.arange(1, len(chosen) + 1)
        parts.append(chosen)
        districts = cand.loc[cand["proposed_client"] == client, "District"]
        info.update(candidates=int(len(districts)), selected=int(len(chosen)),
                    district_candidates={int(k): int(v) for k, v in districts.value_counts().sort_index().items()},
                    district_selected={int(k): int(v) for k, v in chosen["District"].value_counts().sort_index().items()},
                    type_selected=chosen["type_group"].value_counts().sort_index().to_dict(),
                    fallbacks=client_log)
        per_client_summary[client] = info
        log.extend(dict(client=client, **x) for x in client_log)
    selected = pd.concat(parts)
    selected["stratum"] = selected["proposed_client"] + "/" + selected["type_group"] + "/" + selected["flow_tertile"]
    selected["selection_salt"] = SALT
    selected = selected.drop(columns="_tie").reset_index(drop=True)
    if not selected["array_index"].is_unique:
        raise AssertionError("A sensor was selected twice")
    summary = dict(salt=SALT, per_client=per_client, n_candidates=int(len(cand)), n_selected=int(len(selected)),
                   n_excluded_pilot=int(frame["array_index"].isin(list(exclude)).sum()),
                   columns_used=list(SELECTION_COLUMNS), clients=per_client_summary, fallbacks=log,
                   id_mapping_status=config.ID_MAPPING_STATUS)
    return selected, summary


def build_manifest(audit, selected):
    """Full audit rows (same columns as the pilot manifest) in selection order, plus the Phase 8 columns."""
    extra = selected.set_index("array_index")[list(MANIFEST_EXTRA)]
    rows = audit.set_index("array_index").loc[selected["array_index"]]
    return rows.join(extra).reset_index()


def load_tables(audit_path, meta_path, pilot_manifests):
    audit = pd.read_csv(audit_path, keep_default_na=False, na_values=[""])
    audit["exclusion_reason"] = audit["exclusion_reason"].astype(object).fillna("")
    meta = pd.read_csv(meta_path, usecols=["station_id", "Type"])
    exclude = sorted(set().union(*(pd.read_csv(p, usecols=["array_index"])["array_index"] for p in pilot_manifests)))
    return audit, meta, exclude


def load_config(path):
    """Selection config (experiments/configs/phase8_selection.json); the salt must be the documented one."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    need = {"salt", "per_client", "inputs", "input_sha256", "out_dir", "prepare_pilot"}
    if need - set(cfg):
        raise ValueError(f"Selection config lacks {sorted(need - set(cfg))}")
    if cfg["salt"] != SALT:
        raise ValueError(f"Salt {cfg['salt']!r} is not the documented seed {SALT!r}")
    if set(cfg["inputs"]) != {"audit", "metadata", "pilot_manifests"}:
        raise ValueError("inputs must name audit, metadata and pilot_manifests")
    names = [cfg["inputs"]["audit"], cfg["inputs"]["metadata"], *cfg["inputs"]["pilot_manifests"]]
    if set(cfg["input_sha256"]) != set(names):
        raise ValueError("input_sha256 must record exactly the selection inputs")
    pp = cfg["prepare_pilot"]
    if not (isinstance(cfg["per_client"], int) and pp["min_sensors"] <= 4 * cfg["per_client"] <= pp["max_sensors"]):
        raise ValueError("4 x per_client must lie within prepare_pilot min/max sensors")
    return cfg


def near_duplicate_pairs(flow_train, threshold=0.999):
    """Report-only (D8-11): sensor-position pairs whose training-period flow correlation >= threshold.

    Pairwise-complete correlation; constant or empty overlaps give NaN and are ignored. Never
    changes the selection.
    """
    corr = pd.DataFrame(np.asarray(flow_train, dtype=np.float64).T).corr(min_periods=2).to_numpy()
    i, j = np.triu_indices(len(corr), k=1)
    hit = corr[i, j] >= threshold
    return [(int(a), int(b), float(c)) for a, b, c in zip(i[hit], j[hit], corr[i, j][hit])]


def write_outputs(out_dir, manifest, summary):
    out = check_output_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / "sensor_selection.csv", index=False)
    (out / "selection_summary.json").write_text(json.dumps(summary, indent=2, default=int), encoding="utf-8")
    lines = ["# geo200 selection (local only: contains sensor-level IDs)", "",
             f"- Salt `{summary['salt']}`; {summary['n_selected']} selected from {summary['n_candidates']} candidates "
             f"({summary['n_excluded_pilot']} pilot sensors excluded).",
             f"- ID mapping: {summary['id_mapping_status']}.", f"- Fallbacks: {len(summary['fallbacks'])}.", "",
             "| client | type | tertile | array_index | station_id | District | basis |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r.proposed_client} | {r.type_group} | {r.flow_tertile} | {r.array_index} | {r.station_id} | "
              f"{r.District} | {r.selection_basis} |" for r in manifest.itertuples()]
    (out / "selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
