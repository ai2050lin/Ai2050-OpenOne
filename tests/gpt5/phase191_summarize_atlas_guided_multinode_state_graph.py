#!/usr/bin/env python3
"""Summarize Phase191 atlas-guided multi-node state graph results."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/gpt5_phase191_atlas_guided_multinode_state_graph")
MODELS = ("qwen3", "glm4", "deepseek7b")
CONTROL_KINDS = {"random_control", "wrong_relation_control", "shuffled_node_control"}


def fmt(x: float) -> str:
    return f"{x:.4f}"


def compact(row: dict) -> dict:
    return {
        "key": row["key"],
        "kind": row["kind"],
        "set_size": row["set_size"],
        "switch": row["switch"],
        "n": row["n"],
        "switch_rate": row["switch_rate"],
        "mean_margin_gain": row["mean_margin_gain"],
        "mean_common_delta": row["mean_common_delta"],
        "positive_margin_rate": row["positive_margin_rate"],
    }


def rank(rows: list[dict], control: bool | None = None, set_size: int | None = None) -> list[dict]:
    out = []
    for row in rows:
        is_control = row["kind"] in CONTROL_KINDS
        if control is not None and is_control != control:
            continue
        if set_size is not None and row["set_size"] != set_size:
            continue
        out.append(row)
    return sorted(
        out,
        key=lambda r: (r["switch"], r["switch_rate"], r["mean_margin_gain"], r["positive_margin_rate"]),
        reverse=True,
    )


def load_model(model: str) -> dict:
    path = ROOT / f"phase191_{model}_atlas_guided_multinode_state_graph_confirm.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = list(data["summary"]["by_key"].values())
    signal = rank(rows, control=False)
    control = rank(rows, control=True)
    by_size = {}
    for size in sorted({r["set_size"] for r in rows}):
        sig = rank(rows, control=False, set_size=size)
        ctrl = rank(rows, control=True, set_size=size)
        by_size[str(size)] = {
            "best_signal": compact(sig[0]) if sig else None,
            "best_control": compact(ctrl[0]) if ctrl else None,
            "signal_max_switch": sig[0]["switch"] if sig else 0,
            "control_max_switch": ctrl[0]["switch"] if ctrl else 0,
            "signal_max_margin": max((r["mean_margin_gain"] for r in sig), default=0.0),
            "control_max_margin": max((r["mean_margin_gain"] for r in ctrl), default=0.0),
        }
    return {
        "model": model,
        "source": str(path),
        "n_cases": data["n_cases"],
        "n_target_cases_seen": data["n_target_cases_seen"],
        "n_rows": data["n_rows"],
        "total_time_min": data["total_time_min"],
        "atlas_target": data["atlas_target"],
        "nodes": data["nodes"],
        "best_signal": [compact(r) for r in signal[:10]],
        "best_control": [compact(r) for r in control[:10]],
        "by_size": by_size,
        "max_signal_switch": signal[0]["switch"] if signal else 0,
        "max_control_switch": control[0]["switch"] if control else 0,
        "max_signal_margin": max((r["mean_margin_gain"] for r in signal), default=0.0),
        "max_control_margin": max((r["mean_margin_gain"] for r in control), default=0.0),
    }


def node_label(node: dict) -> str:
    return f"{node.get('position')}/L{node.get('layer')}"


def md_table(rows: list[dict]) -> list[str]:
    lines = [
        "| key | kind | set | switch | margin_gain | common_delta | positive_margin_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['key']}` | {row['kind']} | {row['set_size']} | "
            f"{row['switch']}/{row['n']} | {fmt(row['mean_margin_gain'])} | "
            f"{fmt(row['mean_common_delta'])} | {fmt(row['positive_margin_rate'])} |"
        )
    return lines


def evidence_update(model: dict) -> str:
    if model["max_signal_switch"] > model["max_control_switch"] and model["max_signal_margin"] > model["max_control_margin"]:
        return "possible_upgrade_candidate"
    if model["max_signal_switch"] <= model["max_control_switch"]:
        return "downgrade_or_hold_due_to_control_pollution"
    return "hold_due_to_weak_margin"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    summary = {"phase": 191, "models": [load_model(model) for model in MODELS]}
    for model in summary["models"]:
        model["evidence_update"] = evidence_update(model)
    (ROOT / "phase191_cross_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Phase191 Cross-Model Summary",
        "",
        "Objective: test whether atlas-selected multi-node MLP z-state graph interventions upgrade candidate-ranking evidence beyond Phase190 single-node/channel results.",
        "",
        "All models used 128 confirm cases. Rows are target cases where base was wrong and repair prompt was correct.",
        "",
        "## Model Overview",
        "",
        "| model | target rows | nodes | time min | max signal switch | max control switch | max signal margin | max control margin | evidence update |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in summary["models"]:
        nodes = ", ".join(node_label(n) for n in model["nodes"])
        lines.append(
            f"| {model['model']} | {model['n_target_cases_seen']} | {nodes} | "
            f"{model['total_time_min']:.2f} | {model['max_signal_switch']} | "
            f"{model['max_control_switch']} | {fmt(model['max_signal_margin'])} | "
            f"{fmt(model['max_control_margin'])} | {model['evidence_update']} |"
        )

    for model in summary["models"]:
        lines.extend(["", f"## {model['model']} Best Signal", ""])
        lines.extend(md_table(model["best_signal"][:8]))
        lines.extend(["", f"## {model['model']} Best Control", ""])
        lines.extend(md_table(model["best_control"][:8]))
        lines.extend(["", f"## {model['model']} By Set Size", ""])
        lines.extend([
            "| set size | best signal switch | best signal margin | best control switch | best control margin |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ])
        for size, item in model["by_size"].items():
            sig = item["best_signal"]
            ctrl = item["best_control"]
            lines.append(
                f"| {size} | {sig['switch'] if sig else 0}/{sig['n'] if sig else 0} | "
                f"{fmt(sig['mean_margin_gain']) if sig else '0.0000'} | "
                f"{ctrl['switch'] if ctrl else 0}/{ctrl['n'] if ctrl else 0} | "
                f"{fmt(ctrl['mean_margin_gain']) if ctrl else '0.0000'} |"
            )

    lines.extend([
        "",
        "## Objective Reading",
        "",
        "- The uploaded interpretation is correct: Phase191 is not a retreat from the atlas, but atlas-guided causal drilling with explicit evidence update.",
        "- Multi-node signal did not cleanly beat controls across models. Qwen3 and GLM4 had signal switches, but controls remained comparable; DS7B's best switch was a wrong-relation control.",
        "- Evidence should not upgrade to Level5. The candidate-ranking edge should hold as weak Level4 candidate at best, with control pollution and static z-patch insufficiency recorded as failure types.",
        "- The next gap is not more static node/channel patching; it is forward dynamics/state-transition testing where downstream winner selection is measured over the full autoregressive trajectory.",
    ])
    (ROOT / "phase191_cross_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    atlas_updates = {
        "phase": 191,
        "edge": "candidate-specific ranking repair",
        "old_level": "Level3 strong transition / weak Level4 channel contribution",
        "new_level": "hold at weak Level4 candidate; do not upgrade to Level5",
        "validated_nodes": {
            model["model"]: [node_label(n) for n in model["nodes"]]
            for model in summary["models"]
        },
        "failure_types": [
            "control_pollution",
            "multi_node_static_z_patch_insufficiency",
            "candidate_ranking_not_portable_by_static_node_set",
            "winner_switch_rate_too_low",
        ],
        "next_gap": "forward_dynamics_state_transition_test",
        "stop_condition_triggered": "static single-node/channel and static multi-node z-patch routes failed to cleanly beat controls",
    }
    (ROOT / "phase191_atlas_update.json").write_text(
        json.dumps(atlas_updates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    atlas_lines = [
        "# Phase191 Atlas Update",
        "",
        f"- edge: {atlas_updates['edge']}",
        f"- old_level: {atlas_updates['old_level']}",
        f"- new_level: {atlas_updates['new_level']}",
        f"- next_gap: {atlas_updates['next_gap']}",
        f"- stop_condition_triggered: {atlas_updates['stop_condition_triggered']}",
        "",
        "## Validated Nodes",
    ]
    for model, nodes in atlas_updates["validated_nodes"].items():
        atlas_lines.append(f"- {model}: {', '.join(nodes)}")
    atlas_lines.extend(["", "## Failure Types"])
    for failure in atlas_updates["failure_types"]:
        atlas_lines.append(f"- {failure}")
    (ROOT / "phase191_atlas_update.md").write_text("\n".join(atlas_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
