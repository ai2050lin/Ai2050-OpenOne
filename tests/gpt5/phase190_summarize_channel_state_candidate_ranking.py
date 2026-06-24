#!/usr/bin/env python3
"""Summarize Phase190 cross-model channel-state ranking results."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/gpt5_phase190_channel_state_candidate_ranking")
MODELS = ("qwen3", "glm4", "deepseek7b")
CONTROL_KINDS = {"random_control", "wrong_relation_control", "common_control"}


def fmt(x: float) -> str:
    return f"{x:.4f}"


def compact(row: dict) -> dict:
    return {
        "key": row["key"],
        "switch": row["switch"],
        "n": row["n"],
        "switch_rate": row["switch_rate"],
        "mean_margin_gain": row["mean_margin_gain"],
        "mean_common_delta": row["mean_common_delta"],
        "positive_margin_rate": row["positive_margin_rate"],
        "kind": row["kind"],
        "position": row["position"],
        "layer": row["layer"],
        "k": row["k"],
        "alpha": row["alpha"],
    }


def rank(rows: list[dict], control: bool | None = None) -> list[dict]:
    items = []
    for r in rows:
        is_control = r["kind"] in CONTROL_KINDS
        if control is not None and is_control != control:
            continue
        items.append(r)
    return sorted(
        items,
        key=lambda r: (
            r["switch"],
            r["switch_rate"],
            r["mean_margin_gain"],
            r["positive_margin_rate"],
        ),
        reverse=True,
    )


def load_model(model: str) -> dict:
    path = ROOT / f"phase190_{model}_channel_state_candidate_ranking_confirm.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = list(data["summary"]["by_key"].values())
    non_control = rank(rows, control=False)
    controls = rank(rows, control=True)
    return {
        "model": model,
        "source": str(path),
        "n_cases": data["n_cases"],
        "n_target_cases_seen": data["n_target_cases_seen"],
        "n_rows": data["n_rows"],
        "total_time_min": data["total_time_min"],
        "nodes": data["nodes"],
        "best_non_control": [compact(r) for r in non_control[:12]],
        "best_control": [compact(r) for r in controls[:8]],
        "max_non_control_switch": non_control[0]["switch"] if non_control else 0,
        "max_control_switch": controls[0]["switch"] if controls else 0,
        "max_non_control_margin": max(r["mean_margin_gain"] for r in non_control) if non_control else 0.0,
        "max_control_margin": max(r["mean_margin_gain"] for r in controls) if controls else 0.0,
    }


def node_label(node: dict | list | tuple) -> str:
    if isinstance(node, dict):
        return f"{node.get('position')}/L{node.get('layer')}"
    if isinstance(node, (list, tuple)) and len(node) >= 2:
        return f"{node[0]}/L{node[1]}"
    return str(node)


def md_table(rows: list[dict]) -> list[str]:
    lines = [
        "| key | switch | margin_gain | common_delta | positive_margin_rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['key']}` | {r['switch']}/{r['n']} | "
            f"{fmt(r['mean_margin_gain'])} | {fmt(r['mean_common_delta'])} | "
            f"{fmt(r['positive_margin_rate'])} |"
        )
    return lines


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": 190,
        "models": [load_model(model) for model in MODELS],
    }
    (ROOT / "phase190_cross_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Phase190 Cross-Model Summary",
        "",
        "Objective: test whether MLP z-channel groups at Phase594 candidate nodes can causally improve correct-vs-old-top-wrong candidate ranking.",
        "",
        "All three models used confirm mode with 128 generated cases; rows below are the target subset where the base prompt was wrong and the repair prompt was correct.",
        "",
        "## Model Overview",
        "",
        "| model | target rows | nodes | time min | max non-control switch | max control switch | max non-control margin | max control margin |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in summary["models"]:
        nodes = ", ".join(node_label(node) for node in m["nodes"])
        lines.append(
            f"| {m['model']} | {m['n_target_cases_seen']} | {nodes} | "
            f"{m['total_time_min']:.2f} | {m['max_non_control_switch']} | "
            f"{m['max_control_switch']} | {fmt(m['max_non_control_margin'])} | "
            f"{fmt(m['max_control_margin'])} |"
        )

    for m in summary["models"]:
        lines.extend(["", f"## {m['model']} Best Non-Control", ""])
        lines.extend(md_table(m["best_non_control"][:8]))
        lines.extend(["", f"## {m['model']} Best Controls", ""])
        lines.extend(md_table(m["best_control"][:6]))

    lines.extend([
        "",
        "## Objective Reading",
        "",
        "- Positive but weak causal signal: every model had at least one intervention that switched a target case, but switch rates stayed low.",
        "- Control contamination is significant: random/wrong-relation/common controls also produced switches, especially DS7B query_relation L19. This prevents a strong claim that the selected channel groups are a clean candidate-ranking mechanism.",
        "- The result supports a stricter interpretation of Phase189: Phase594 MLP nodes contain ranking-relevant channel directions, but channel selection by simple readout/top-k is not yet a stable repair rule.",
    ])
    (ROOT / "phase190_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
