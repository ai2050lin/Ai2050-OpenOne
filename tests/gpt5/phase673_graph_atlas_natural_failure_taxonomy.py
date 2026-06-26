#!/usr/bin/env python3
"""
Phase 673: Graph Atlas Natural Failure Taxonomy.

Post-processes Phase 672 natural trajectory rows into failure classes and
selects high-quality internal-test entry points. This phase does not run model
inference.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


IN_ROOT = Path("results/glm5_phase672_graph_atlas_counterfactual_natural_trajectory_audit")
OUT_ROOT = Path("results/glm5_phase673_graph_atlas_natural_failure_taxonomy")
MODELS = ["qwen3", "glm4", "deepseek7b"]

EXPECTED_CLASS = {
    "short": "short_or_other",
    "value_only": "short_or_other",
    "intent_only_existence": "short_or_other",
    "sentence": "sentence_or_explanation",
    "explanation": "sentence_or_explanation",
    "json": "json",
    "protocol_only_json": "json",
    "label": "label",
    "list": "list",
}


def read_rows(model: str) -> list[dict]:
    path = IN_ROOT / f"phase672_{model}_natural_trajectory_rows.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_rank(row: dict) -> int | None:
    return row["first_token_metric"].get("expected_rank")


def margin(row: dict) -> float | None:
    return row["first_token_metric"].get("expected_minus_best_competitor")


def expected_format_class(row: dict) -> str:
    return EXPECTED_CLASS.get(row["format_name"], "unknown")


def classify(row: dict) -> str:
    if row["match"]["normalized_exact"]:
        return "success"
    rank = first_rank(row)
    top_cat = row["first_token_metric"].get("top1_category")
    gen_cls = row["generated_class"]
    exp_cls = expected_format_class(row)
    contains = row["match"]["contains_value"]
    first_ok = rank == 1

    if not first_ok:
        if top_cat in {"space", "newline", "word_or_explanation", "json_or_quote", "list_marker"}:
            return "readout_competitor_failure"
        return "readout_other_failure"

    if row["family"] == "same_prefix_different_continuation" and not row["continuation"]["token1_match"]:
        return "continuation_transition_failure"
    if row["family"] == "same_prefix_different_continuation" and not row["continuation"]["token2_match"]:
        return "late_continuation_failure"

    if not contains:
        if gen_cls != exp_cls:
            return "protocol_route_failure"
        return "value_binding_failure"

    if gen_cls != exp_cls:
        return "protocol_route_failure"

    if contains and not row["match"]["compact_exact"]:
        return "format_surface_failure"

    return "other_generation_failure"


def add_count(d: dict, *keys: str) -> None:
    cur = d
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = cur.get(keys[-1], 0) + 1


def summarize_model(model: str, rows: list[dict]) -> dict:
    class_counts: dict[str, int] = {}
    by_family: dict[str, dict[str, int]] = defaultdict(dict)
    by_format: dict[str, dict[str, int]] = defaultdict(dict)
    by_node: dict[str, dict[str, int]] = defaultdict(dict)
    examples: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        cls = classify(row)
        row["failure_class"] = cls
        class_counts[cls] = class_counts.get(cls, 0) + 1
        by_family[row["family"]][cls] = by_family[row["family"]].get(cls, 0) + 1
        by_format[row["format_name"]][cls] = by_format[row["format_name"]].get(cls, 0) + 1
        for node in row["target_nodes"]:
            by_node[node][cls] = by_node[node].get(cls, 0) + 1
        if cls != "success" and len(examples[cls]) < 8:
            examples[cls].append({
                "case_id": row["case_id"],
                "family": row["family"],
                "format": row["format_name"],
                "expected": row["expected_output"],
                "generated": row["generated_text"][:160],
                "top1_category": row["first_token_metric"].get("top1_category"),
                "expected_rank": first_rank(row),
                "margin": margin(row),
                "generated_class": row["generated_class"],
                "contains_value": row["match"]["contains_value"],
            })

    n = max(1, len(rows))
    failure_counts = {k: v for k, v in class_counts.items() if k != "success"}
    dominant_failure = max(failure_counts.items(), key=lambda kv: kv[1])[0] if failure_counts else "none"
    return {
        "model": model,
        "n": len(rows),
        "success_rate": class_counts.get("success", 0) / n,
        "class_counts": dict(sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_family": {k: dict(sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))) for k, v in sorted(by_family.items())},
        "by_format": {k: dict(sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))) for k, v in sorted(by_format.items())},
        "by_node": {k: dict(sorted(v.items(), key=lambda kv: (-kv[1], kv[0]))) for k, v in sorted(by_node.items())},
        "dominant_failure": dominant_failure,
        "examples": dict(examples),
    }


def pick_entry_points(model_summaries: list[dict]) -> list[dict]:
    entries = []
    by_model = {m["model"]: m for m in model_summaries}

    ds = by_model.get("deepseek7b")
    if ds:
        entries.extend([
            {
                "priority": 1,
                "model": "deepseek7b",
                "target": "same_format_random_value",
                "failure_class": "readout_competitor_failure",
                "reason": "DS7B fails nonce short values despite in-context record; likely protocol/readout prior dominates context value binding.",
                "next_internal_test": "trace short-value first-token readout and compare word/explanation competitors at final residual.",
            },
            {
                "priority": 2,
                "model": "deepseek7b",
                "target": "same_prefix_different_continuation",
                "failure_class": "readout_competitor_failure / continuation_transition_failure",
                "reason": "DS7B almost never enters shared-prefix value route; continuation test is blocked by earlier readout/protocol failure.",
                "next_internal_test": "first fix or localize readout/protocol entry before token1 transition patching.",
            },
            {
                "priority": 3,
                "model": "deepseek7b",
                "target": "list format",
                "failure_class": "protocol_route_failure",
                "reason": "List marker route is often beaten by explanation/word route, exposing grammar/protocol competition.",
                "next_internal_test": "compare list marker '-' against explanation word competitors at protocol field layers.",
            },
        ])

    qw = by_model.get("qwen3")
    if qw:
        entries.append({
            "priority": 4,
            "model": "qwen3",
            "target": "same_value_different_format",
            "failure_class": "protocol_route_failure / format_surface_failure",
            "reason": "qwen3 first-token top1 is nearly closed, but format exactness drops sharply.",
            "next_internal_test": "protocol surface formation after first expected token, especially explanation/list/json formatting.",
        })

    gl = by_model.get("glm4")
    if gl:
        entries.append({
            "priority": 5,
            "model": "glm4",
            "target": "different_value_same_format",
            "failure_class": "readout_competitor_failure",
            "reason": "GLM4 shows space/newline competition in otherwise simple short-value controls.",
            "next_internal_test": "space/newline readout source under synthetic in-context value binding.",
        })

    return sorted(entries, key=lambda x: x["priority"])


def write_markdown(payload: dict) -> str:
    lines = [
        "# Phase 673 Graph Atlas Natural Failure Taxonomy",
        "",
        f"- generated: `{payload['timestamp']}`",
        "",
        "## Model Failure Classes",
        "",
    ]
    for ms in payload["models"]:
        lines += [
            f"### {ms['model']}",
            "",
            f"- success_rate: `{ms['success_rate']:.3f}`",
            f"- dominant_failure: `{ms['dominant_failure']}`",
            "",
            "| class | count |",
            "|---|---:|",
        ]
        for cls, n in ms["class_counts"].items():
            lines.append(f"| {cls} | {n} |")
        lines.append("")

    lines += ["## Internal Entry Points", "", "| priority | model | target | failure_class | next_internal_test |", "|---:|---|---|---|---|"]
    for e in payload["entry_points"]:
        lines.append(
            f"| {e['priority']} | {e['model']} | {e['target']} | {e['failure_class']} | {e['next_internal_test']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Phase 672 becomes useful only after failures are separated by class.",
        "- DS7B should not immediately enter token1 writer patching; its natural failures often happen before the value route is entered.",
        "- qwen3 and GLM4 are better candidates for protocol/format continuation studies because first-token readout is mostly closed.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    model_summaries = []
    all_rows = {}
    for model in MODELS:
        rows = read_rows(model)
        all_rows[model] = rows
        model_summaries.append(summarize_model(model, rows))

    payload = {
        "phase": 673,
        "title": "Graph Atlas Natural Failure Taxonomy and Internal Entry Selection",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(IN_ROOT),
        "models": model_summaries,
        "entry_points": pick_entry_points(model_summaries),
        "hard_limits": [
            "Failure classes are heuristic labels over natural outputs, not causal proof.",
            "Some explanation outputs may be semantically acceptable despite normalized exact failure.",
            "Internal entry points still need activation-level tests.",
        ],
    }
    (OUT_ROOT / "phase673_failure_taxonomy.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_ROOT / "phase673_failure_taxonomy.md").write_text(write_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "phase": payload["phase"],
        "models": [
            {
                "model": m["model"],
                "success_rate": m["success_rate"],
                "dominant_failure": m["dominant_failure"],
                "class_counts": m["class_counts"],
            }
            for m in model_summaries
        ],
        "entry_points": payload["entry_points"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
