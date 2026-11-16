"""Rebuild AIORI_FINAL_GRAPH.html from AIORI_GRAPH_SOURCE.json and check the source.

Run from anywhere:  python docs/graphify/final/build/build_graph.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
src = json.loads((OUT / "AIORI_GRAPH_SOURCE.json").read_text(encoding="utf-8"))

# structural checks: unique ids, every link endpoint placed in the link's view, every view has nodes
ids = [n["id"] for n in src["nodes"]]
assert len(ids) == len(set(ids)), "duplicate node id"
nodes = {n["id"]: n for n in src["nodes"]}
views = {v["id"] for v in src["graph"]["views"]}
for n in src["nodes"]:
    assert set(n["pos"]) <= views, f"{n['id']}: unknown view"
    assert n.get("evidence"), f"{n['id']}: no evidence"
for l in src["links"]:
    assert l["view"] in views, l
    for end in (l["source"], l["target"]):
        assert l["view"] in nodes[end]["pos"], f"{end} not placed in view {l['view']}"
for v in views:
    assert any(v in n["pos"] for n in src["nodes"]), f"view {v} empty"

data = json.dumps(src, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = (HERE / "template.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", data, 1)
(OUT / "AIORI_FINAL_GRAPH.html").write_text(html, encoding="utf-8")
print(f"ok: {len(ids)} nodes, {len(src['links'])} links, {len(views)} views -> AIORI_FINAL_GRAPH.html")
