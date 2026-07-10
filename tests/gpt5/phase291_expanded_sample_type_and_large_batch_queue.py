#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

PHASE = 291
SCHEMA_VERSION = "2.18.0"
ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase291_expanded_sample_type_and_large_batch_queue"
MODELS = ["qwen3", "glm4", "deepseek7b"]

EXPANSION_VARIANTS = [
    {
        "variant_id": "stop_word_hard",
        "variant_type": "boundary",
        "output_protocol": "short_stop",
        "boundary_type": "explicit_stop_word",
        "continuation_trigger": "stop_instruction",
        "expected_pattern": "short",
        "suffix": "\nAnswer with only the final answer. Stop immediately after it.\nAnswer:",
        "channel_focus": "stop_boundary",
    },
    {
        "variant_id": "newline_stop",
        "variant_type": "boundary",
        "output_protocol": "newline_stop",
        "boundary_type": "newline_boundary",
        "continuation_trigger": "newline_boundary",
        "expected_pattern": "short",
        "suffix": "\nFinal answer on one line only:\n",
        "channel_focus": "stop_boundary",
    },
    {
        "variant_id": "double_newline_stop",
        "variant_type": "boundary",
        "output_protocol": "double_newline_stop",
        "boundary_type": "double_newline_boundary",
        "continuation_trigger": "paragraph_boundary",
        "expected_pattern": "short",
        "suffix": "\nWrite the answer, then leave the response complete.\nAnswer:\n",
        "channel_focus": "natural_language_continue",
    },
    {
        "variant_id": "comma_continue_pressure",
        "variant_type": "continuation",
        "output_protocol": "short",
        "boundary_type": "comma_boundary",
        "continuation_trigger": "comma_continue",
        "expected_pattern": "short",
        "suffix": "\nAnswer with one word, not a phrase, even if a comma would continue the sentence.\nAnswer:",
        "channel_focus": "local_syntax_continue",
    },
    {
        "variant_id": "because_suppression",
        "variant_type": "continuation",
        "output_protocol": "answer_only",
        "boundary_type": "because_boundary",
        "continuation_trigger": "because_suppressed",
        "expected_pattern": "short",
        "suffix": "\nAnswer only. Do not use because. Do not explain.\nAnswer:",
        "channel_focus": "explanation_relation_continue",
    },
    {
        "variant_id": "for_relation_pressure",
        "variant_type": "continuation",
        "output_protocol": "short",
        "boundary_type": "relation_boundary",
        "continuation_trigger": "for_relation",
        "expected_pattern": "short",
        "suffix": "\nGive the answer only; do not add 'for' or a reason.\nAnswer:",
        "channel_focus": "explanation_relation_continue",
    },
    {
        "variant_id": "json_closed_brace",
        "variant_type": "structure",
        "output_protocol": "json_closed",
        "boundary_type": "json_closed_brace",
        "continuation_trigger": "json_structure",
        "expected_pattern": "json",
        "suffix": "\nReturn exactly {\"answer\":\"...\"} and nothing after the closing brace.\n",
        "channel_focus": "protocol_json_continue",
    },
    {
        "variant_id": "json_no_markdown",
        "variant_type": "structure",
        "output_protocol": "json_no_markdown",
        "boundary_type": "json_no_fence",
        "continuation_trigger": "json_structure",
        "expected_pattern": "json",
        "suffix": "\nReturn raw JSON only, no markdown fence, no extra text.\n",
        "channel_focus": "protocol_json_continue",
    },
    {
        "variant_id": "numbered_list_one",
        "variant_type": "structure",
        "output_protocol": "numbered_list_one",
        "boundary_type": "numbered_list_boundary",
        "continuation_trigger": "list_item",
        "expected_pattern": "list",
        "suffix": "\nReturn exactly one numbered item.\n1.",
        "channel_focus": "list_structure_continue",
    },
    {
        "variant_id": "markdown_bullet_one",
        "variant_type": "structure",
        "output_protocol": "markdown_bullet_one",
        "boundary_type": "bullet_boundary",
        "continuation_trigger": "list_item",
        "expected_pattern": "list",
        "suffix": "\nReturn exactly one bullet and do not add a second bullet.\n-",
        "channel_focus": "list_structure_continue",
    },
    {
        "variant_id": "quote_close",
        "variant_type": "boundary",
        "output_protocol": "quoted_answer",
        "boundary_type": "quote_boundary",
        "continuation_trigger": "quote_close",
        "expected_pattern": "short",
        "suffix": "\nPut only the answer in quotes and stop after the closing quote.\n\"",
        "channel_focus": "answer_boundary_continue",
    },
    {
        "variant_id": "bilingual_answer_only",
        "variant_type": "cross_lingual",
        "output_protocol": "answer_only",
        "boundary_type": "language_boundary",
        "continuation_trigger": "language_switch",
        "expected_pattern": "short",
        "suffix": "\n只输出最终答案，不要解释。Answer only:\n",
        "channel_focus": "natural_language_continue",
    },
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def root_key(case_id: str, variant_id: str) -> str:
    suffix = f"_{variant_id}"
    return case_id[: -len(suffix)] if case_id.endswith(suffix) else case_id


def state_labels(variant: dict[str, str]) -> dict[str, int]:
    protocol = variant["output_protocol"]
    trigger = variant["continuation_trigger"]
    boundary = variant["boundary_type"]
    return {
        "S_content": 1,
        "S_target": 1,
        "S_protocol": 1,
        "S_boundary": 1 if boundary != "none" else 0,
        "S_done": 0,
        "S_continue": 1 if trigger not in {"stop_instruction", "because_suppressed"} else 0,
        "S_stop": 1 if "stop" in protocol or trigger in {"stop_instruction", "because_suppressed"} else 0,
        "S_structure": 1 if "json" in protocol or "list" in protocol or "bullet" in protocol else 0,
    }


def select_roots(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    roots: dict[str, dict[str, Any]] = {}
    for row in cases:
        if row.get("variant_id") == "base":
            roots[root_key(str(row["case_id"]), str(row["variant_id"]))] = row
    return list(roots.values())


def build_candidates(root_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    created = now()
    for base in root_cases:
        base_root = root_key(str(base["case_id"]), str(base["variant_id"]))
        question = str(base["prompt"]).split("\nAnswer:")[0].strip()
        for variant in EXPANSION_VARIANTS:
            case_id = f"{base_root}_{variant['variant_id']}"
            states = state_labels(variant)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase291",
                    "created_at": created,
                    "case_id": case_id,
                    "source_case_id": base["case_id"],
                    "source_root_id": base_root,
                    "family_id": base["family_id"],
                    "mode_id": base["mode_id"],
                    "variant_id": variant["variant_id"],
                    "variant_type": variant["variant_type"],
                    "prompt_template_id": f"{base['family_id']}_{base['mode_id']}_{variant['variant_id']}_v4",
                    "prompt": question + variant["suffix"],
                    "target": base["target"],
                    "expected_answer": base.get("expected_answer") or base["target"],
                    "expected_pattern": variant["expected_pattern"],
                    "output_protocol": variant["output_protocol"],
                    "boundary_type": variant["boundary_type"],
                    "done_label": "not_done_prompt",
                    "continuation_trigger": variant["continuation_trigger"],
                    "channel_focus": variant["channel_focus"],
                    "scoring_risk": "medium" if variant["expected_pattern"] in {"json", "list"} else base.get("scoring_risk", "low"),
                    "difficulty": "phase291_expanded_sample_type",
                    "tags": [base["family_id"], base["mode_id"], variant["variant_type"], variant["output_protocol"], variant["channel_focus"]],
                    "target_aliases": base.get("target_aliases") or [base["target"]],
                    "test_levels": [
                        "behavior",
                        "readout_competition",
                        "layer_path",
                        "component_path",
                        "causal_audit",
                        "rollout",
                        "closure",
                    ],
                    "path_schema_id": f"phase291:path_schema:{base['family_id']}:{base['mode_id']}:{variant['variant_id']}",
                    "measurement_status": "planned_not_measured",
                    **states,
                }
            )
    return rows


def pressure_maps() -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    readout = {}
    for row in read_jsonl(V2 / "phase290_family_model_readout_summary.jsonl"):
        readout[(str(row.get("family_id")), str(row.get("model")))] = float(row.get("mean_top_continue_vs_stop_margin") or 0.0)
    gap = {}
    for row in read_jsonl(V2 / "phase288_gap_heatmap_rows.jsonl"):
        gap[(str(row.get("family_id")), str(row.get("model")))] = float(row.get("open_gap_total") or 0.0)
    return readout, gap


def model_plan(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    readout_pressure, gap_pressure = pressure_maps()
    rows = []
    created = now()
    for candidate in candidates:
        for model in MODELS:
            family = str(candidate["family_id"])
            channel = str(candidate["channel_focus"])
            pressure = readout_pressure.get((family, model), 0.0)
            gap = gap_pressure.get((family, model), 0.0)
            protocol_bonus = 2.0 if channel in {"list_structure_continue", "protocol_json_continue", "protocol_format_continue"} else 0.5
            priority = pressure + gap / 40.0 + protocol_bonus
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase291",
                    "created_at": created,
                    "plan_id": f"phase291:model_plan:{model}:{candidate['case_id']}",
                    "model": model,
                    "case_id": candidate["case_id"],
                    "family_id": family,
                    "mode_id": candidate["mode_id"],
                    "variant_id": candidate["variant_id"],
                    "variant_type": candidate["variant_type"],
                    "channel_focus": channel,
                    "prompt": candidate["prompt"],
                    "target": candidate["target"],
                    "expected_pattern": candidate["expected_pattern"],
                    "output_protocol": candidate["output_protocol"],
                    "priority_score": round(priority, 6),
                    "readout_pressure": round(pressure, 6),
                    "gap_pressure": round(gap, 6),
                    "batch_tier": "tier2_full_expansion",
                    "measurement_status": "planned_not_measured",
                }
            )
    return rows


def selected_large_batch(plan: list[dict[str, Any]], per_family_model: int = 36) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in plan:
        buckets[(str(row["family_id"]), str(row["model"]))].append(row)
    selected: list[dict[str, Any]] = []
    for key, bucket in sorted(buckets.items()):
        by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in bucket:
            by_variant[str(row["variant_id"])].append(row)
        for rows in by_variant.values():
            rows.sort(key=lambda r: (-float(r["priority_score"]), str(r["case_id"])))
        picked: list[dict[str, Any]] = []
        idx = 0
        while len(picked) < per_family_model:
            added = False
            for variant in sorted(by_variant):
                rows = by_variant[variant]
                if idx < len(rows):
                    picked.append(rows[idx])
                    added = True
                    if len(picked) >= per_family_model:
                        break
            if not added:
                break
            idx += 1
        selected.extend(picked)
    selected.sort(key=lambda r: (-float(r["priority_score"]), str(r["model"]), str(r["family_id"]), str(r["case_id"])))
    out = []
    for rank, row in enumerate(selected, 1):
        out.append({**row, "batch_tier": "tier1_large_balanced_batch", "phase291_batch_rank": rank})
    return out


def coverage_rows(candidates: list[dict[str, Any]], selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    cand_group = Counter((r["family_id"], r["variant_id"], r["channel_focus"]) for r in candidates)
    sel_group = Counter((r["family_id"], r["variant_id"], r["channel_focus"]) for r in selected)
    for (family, variant, channel), count in sorted(cand_group.items()):
        selected_count = sel_group.get((family, variant, channel), 0)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase291",
                "created_at": now(),
                "coverage_id": f"phase291:sample_type_coverage:{family}:{variant}",
                "family_id": family,
                "variant_id": variant,
                "channel_focus": channel,
                "candidate_cases": count,
                "selected_model_case_rows": selected_count,
                "selection_rate_vs_all_models": round(selected_count / (count * len(MODELS)), 6) if count else 0.0,
            }
        )
    return rows


def update_v2(summary: dict[str, Any]) -> None:
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    for name in [
        "phase291_expanded_case_candidates",
        "phase291_full_model_test_plan_rows",
        "phase291_selected_large_batch_queue",
        "phase291_sample_type_coverage_rows",
    ]:
        files[name] = f"{name}.jsonl"
    files["phase291_summary"] = "phase291_summary.json"
    files["phase291_report"] = "phase291_report.md"
    manifest["latest_sample_expansion_phase"] = "Phase291"
    manifest["phase291_summary"] = summary
    write_json(V2 / "manifest.json", manifest)

    client = read_json(V2 / "client_index.json")
    for item in [
        "phase291_summary.json",
        "phase291_selected_large_batch_queue.jsonl",
        "phase291_sample_type_coverage_rows.jsonl",
    ]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase291_summary_ref"] = "phase291_summary.json"
    client["phase291_large_batch_queue_ref"] = "phase291_selected_large_batch_queue.jsonl"
    write_json(V2 / "client_index.json", client)

    schema = read_json(V2 / "schema.json")
    tables = schema.setdefault("tables", {})
    tables["phase291_expanded_case_candidates"] = "expanded planned sample cases; not model-measured yet"
    tables["phase291_full_model_test_plan_rows"] = "all expanded sample x model planned rows"
    tables["phase291_selected_large_batch_queue"] = "balanced high-priority large batch queue for sequential qwen3, GLM4, DS7B testing"
    tables["phase291_sample_type_coverage_rows"] = "expanded sample type coverage by family and variant"
    write_json(V2 / "schema.json", schema)


def main() -> None:
    existing_cases = read_jsonl(V2 / "cases.jsonl")
    roots = select_roots(existing_cases)
    candidates = build_candidates(roots)
    plan = model_plan(candidates)
    selected = selected_large_batch(plan)
    coverage = coverage_rows(candidates, selected)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase291",
        "created_at": now(),
        "model_test_status": "not_run_sample_expansion_and_queue_only",
        "source_existing_cases": len(existing_cases),
        "source_root_cases": len(roots),
        "new_variant_types": len(EXPANSION_VARIANTS),
        "expanded_case_candidates": len(candidates),
        "full_model_test_plan_rows": len(plan),
        "selected_large_batch_queue_rows": len(selected),
        "sample_type_coverage_rows": len(coverage),
        "candidate_family_counts": dict(Counter(str(r["family_id"]) for r in candidates)),
        "candidate_channel_focus_counts": dict(Counter(str(r["channel_focus"]) for r in candidates)),
        "selected_by_model": dict(Counter(str(r["model"]) for r in selected)),
        "selected_by_family": dict(Counter(str(r["family_id"]) for r in selected)),
        "selected_by_channel_focus": dict(Counter(str(r["channel_focus"]) for r in selected)),
        "progress_estimate": {
            "pattern_family_atlas": 0.70,
            "sample_type_coverage": 0.62,
            "physical_distribution_puzzle": 0.63,
            "feature_mining": 0.42,
            "mechanism_audit": 0.41,
            "closure": 0.20,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "phase291_expanded_case_candidates.jsonl": candidates,
        "phase291_full_model_test_plan_rows.jsonl": plan,
        "phase291_selected_large_batch_queue.jsonl": selected,
        "phase291_sample_type_coverage_rows.jsonl": coverage,
    }
    for name, rows in outputs.items():
        write_jsonl(OUT / name, rows)
        write_jsonl(V2 / name, rows)
    write_json(OUT / "phase291_summary.json", summary)
    write_json(V2 / "phase291_summary.json", summary)
    report = "\n".join(
        [
            "# Phase291 Expanded Sample Type And Large Batch Queue",
            "",
            f"- source_existing_cases: {summary['source_existing_cases']}",
            f"- source_root_cases: {summary['source_root_cases']}",
            f"- new_variant_types: {summary['new_variant_types']}",
            f"- expanded_case_candidates: {summary['expanded_case_candidates']}",
            f"- full_model_test_plan_rows: {summary['full_model_test_plan_rows']}",
            f"- selected_large_batch_queue_rows: {summary['selected_large_batch_queue_rows']}",
            f"- selected_by_model: {json.dumps(summary['selected_by_model'], ensure_ascii=False)}",
            f"- selected_by_channel_focus: {json.dumps(summary['selected_by_channel_focus'], ensure_ascii=False)}",
            "",
            "These rows are planned samples and queues, not model-measured results.",
        ]
    ) + "\n"
    (OUT / "phase291_report.md").write_text(report, encoding="utf-8")
    (V2 / "phase291_report.md").write_text(report, encoding="utf-8")
    update_v2(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
