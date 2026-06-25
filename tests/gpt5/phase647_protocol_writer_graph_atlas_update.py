#!/usr/bin/env python3
"""Build Phase 647 writer-graph atlas update rows from confirm results."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase647_protocol_writer_graph_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(model: str):
    path = OUT_ROOT / f"phase647_{model}_protocol_writer_graph_audit_confirm.json"
    if not path.exists():
        return None, path
    return json.loads(path.read_text(encoding="utf-8")), path


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    nodes = []
    edges = []
    evidence = []
    report = ["# Phase 647 Writer Graph Atlas Update\n"]

    for model in MODELS:
        data, path = load(model)
        if data is None:
            continue
        base = {r["mode"]: r for r in data["summary"]["by_mode"] if r["kind"] == "baseline"}
        original = base.get("original", {})
        inline = base.get("inline", {})
        report.append(f"## {model}\n")
        report.append(
            f"- baselines: original exact={original.get('exact')}/{original.get('n')}, "
            f"newline={original.get('newline_top0')}/{original.get('n')}; "
            f"inline exact={inline.get('exact')}/{inline.get('n')}, newline={inline.get('newline_top0')}/{inline.get('n')}"
        )

        for family_name, rows in [
            ("sufficiency", data["summary"]["best_sufficiency_restore"][:12]),
            ("necessity", data["summary"]["best_necessity_remove"][:12]),
        ]:
            report.append(f"### {family_name}\n")
            for rank, row in enumerate(rows, start=1):
                node_id = (
                    f"writer:{model}:{family_name}:"
                    f"{row.get('scope')}:{row.get('interval') or 'single'}:"
                    f"L{row.get('layer') if row.get('layer') is not None else 'multi'}:"
                    f"{row.get('component')}"
                )
                nodes.append({
                    "node_id": node_id,
                    "node_type": "writer_candidate",
                    "model": model,
                    "family": family_name,
                    "rank": rank,
                    "scope": row.get("scope"),
                    "interval": row.get("interval"),
                    "layer": row.get("layer"),
                    "layers": row.get("layers"),
                    "component": row.get("component"),
                    "exact": row.get("exact"),
                    "n": row.get("n"),
                    "newline_top0": row.get("newline_top0"),
                    "mean_prefix_rank": row.get("mean_prefix_rank"),
                })
                evidence_id = f"phase647:{model}:{row['mode']}"
                evidence.append({
                    "evidence_id": evidence_id,
                    "node_id": node_id,
                    "model": model,
                    "mode": row["mode"],
                    "family": family_name,
                    "direction": row.get("direction"),
                    "scope": row.get("scope"),
                    "interval": row.get("interval"),
                    "layer": row.get("layer"),
                    "layers": row.get("layers"),
                    "component": row.get("component"),
                    "control": row.get("control"),
                    "n": row.get("n"),
                    "exact": row.get("exact"),
                    "tok0_hit": row.get("tok0_hit"),
                    "newline_top0": row.get("newline_top0"),
                    "mean_prefix_rank": row.get("mean_prefix_rank"),
                    "source_path": str(path),
                })
                edges.append({
                    "source": "mechanism:value_short_answer_protocol",
                    "target": node_id,
                    "edge_type": f"has_{family_name}_writer_candidate",
                    "model": model,
                    "evidence_id": evidence_id,
                })
                report.append(
                    f"- {rank}. `{row['mode']}` exact={row.get('exact')}/{row.get('n')}, "
                    f"newline={row.get('newline_top0')}/{row.get('n')}, rank={row.get('mean_prefix_rank'):.2f}"
                )
            report.append("")

    write_jsonl(OUT_ROOT / "phase647_writer_graph_nodes.jsonl", nodes)
    write_jsonl(OUT_ROOT / "phase647_writer_graph_edges.jsonl", edges)
    write_jsonl(OUT_ROOT / "phase647_writer_graph_evidence.jsonl", evidence)
    (OUT_ROOT / "phase647_writer_graph_atlas_update.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "nodes": len(nodes),
        "edges": len(edges),
        "evidence": len(evidence),
        "report": str(OUT_ROOT / "phase647_writer_graph_atlas_update.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
