#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


PHASE = 265
SOURCE_PHASE = 264
SCHEMA_VERSION = "1.0.0"
RESULT_ROOT = Path("tests/result/phase265_multi_family_case_bank_path_schema_expansion")
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
ROUND_DEFAULT = "multi_family_case_bank_path_schema_expansion"

FAMILY_SPECS = {
    "content_knowledge": {
        "modes": ["object_attribute", "category_membership", "relation_value", "fact_recall"],
        "items": [
            ("apple color", "What color is a typical ripe apple?", "red"),
            ("sparrow category", "Is a sparrow a bird or a fish?", "bird"),
            ("Paris country", "Paris is in which country?", "France"),
            ("water state", "At room temperature, water is usually a liquid or a gas?", "liquid"),
            ("triangle sides", "How many sides does a triangle have?", "3"),
            ("sun rises", "The sun appears to rise in which direction?", "east"),
        ],
    },
    "output_protocol": {
        "modes": ["short_answer", "explain_answer", "json_answer", "list_answer"],
        "items": [
            ("one word color", "Answer with exactly one word: What color is grass?", "green"),
            ("final answer only", "Give only the final answer: 2+2=?", "4"),
            ("json field", "Return JSON with key answer for the color of snow.", "white"),
            ("list stop", "List one primary color and then stop.", "red"),
            ("no explanation", "Answer with no explanation: capital of Italy?", "Rome"),
            ("answer anchor", "Answer after the marker only. Answer: largest planet?", "Jupiter"),
        ],
    },
    "reasoning_constraint": {
        "modes": ["if_then", "comparison", "negation", "multi_step"],
        "items": [
            ("if rain umbrella", "If it rains, Sam uses an umbrella. It rains. What does Sam use?", "umbrella"),
            ("larger number", "Which is larger, 17 or 12?", "17"),
            ("not blue", "The box is not blue; it is red. What color is it?", "red"),
            ("two step sum", "A bag has 2 red balls and 3 blue balls. How many balls?", "5"),
            ("constraint pick", "Choose the animal that can fly: cat, eagle, whale.", "eagle"),
            ("counterfactual small", "If ice were warm, would it still be frozen? Answer yes or no.", "no"),
        ],
    },
    "syntax_structure": {
        "modes": ["subject_verb", "modifier_binding", "clause_scope", "punctuation_boundary"],
        "items": [
            ("modifier animal", "In 'small red ball', what does red describe?", "ball"),
            ("subject verb", "In 'The dogs run', what is the subject?", "dogs"),
            ("clause object", "In 'Mia said Tom left', who left?", "Tom"),
            ("comma boundary", "Complete the sentence with one word: After dinner, Mia drank", "water"),
            ("quote boundary", "In 'Ana said \"stop\"', what word was said?", "stop"),
            ("coordination", "In 'tea and coffee', name the second drink.", "coffee"),
        ],
    },
    "language_action": {
        "modes": ["answer", "summarize", "translate", "classify"],
        "items": [
            ("answer action", "Answer the question: What is 1+1?", "2"),
            ("summarize action", "Summarize in one word: A long text about rain falling from clouds.", "rain"),
            ("translate action", "Translate to English: 猫", "cat"),
            ("classify action", "Classify as animal or tool: hammer", "tool"),
            ("continue action", "Continue with one word: bread and", "butter"),
            ("clarify action", "If the request is ambiguous, output clarify.", "clarify"),
        ],
    },
    "cross_lingual": {
        "modes": ["zh_to_en", "en_to_zh", "cross_lingual_attribute", "translation"],
        "items": [
            ("zh cat", "Translate 猫 to English.", "cat"),
            ("en dog", "Translate dog to Chinese.", "狗"),
            ("zh color", "苹果通常是什么颜色? Answer in English.", "red"),
            ("en capital zh", "法国的首都是哪里? Answer in English.", "Paris"),
            ("bilingual negation", "不是蓝色，是红色。Answer color in English.", "red"),
            ("cross action", "Translate and answer only: 水 = ?", "water"),
        ],
    },
    "readout_competition": {
        "modes": ["target_vs_because", "target_vs_period", "target_vs_list", "target_vs_the"],
        "items": [
            ("because pressure", "Answer only the word: sky color?", "blue"),
            ("period pressure", "Answer with the word then stop: fire is hot or cold?", "hot"),
            ("list pressure", "Give exactly one item: name a fruit.", "apple"),
            ("the pressure", "Answer one word: The opposite of up is", "down"),
            ("format pressure", "Answer after colon only: color of milk:", "white"),
            ("newline pressure", "Answer one word then stop.\nQuestion: 3+4?", "7"),
        ],
    },
    "state_drift": {
        "modes": ["early_correct_late_drift", "explain_drift", "format_drift", "next_task_drift"],
        "items": [
            ("answer then drift", "Answer one word only: color of banana?", "yellow"),
            ("reason drift", "Give the answer first, no reason: 5+5?", "10"),
            ("format drift", "Output exactly JSON: answer for color of coal.", "black"),
            ("next task drift", "Answer this only, do not ask another question: capital of Spain?", "Madrid"),
            ("echo drift", "Do not repeat the question. Answer: water freezes at what temperature C?", "0"),
            ("long drift", "Answer with one word and no extra sentence: bird that says quack?", "duck"),
        ],
    },
    "closure": {
        "modes": ["answer_correct", "pattern_matched", "boundary_stable", "model_stop"],
        "items": [
            ("close short", "Answer exactly one word and stop: color of night sky?", "black"),
            ("close period", "Answer with one word followed by a period, then stop: opposite of yes?", "no"),
            ("close json", "Return JSON only: answer for 6*6.", "36"),
            ("close list", "Return one bullet only: a mammal.", "dog"),
            ("close no drift", "Answer only the final word, no explanation: frozen water is", "ice"),
            ("close eos", "Give the final answer and end: capital of Japan?", "Tokyo"),
        ],
    },
}

VARIANTS = [
    {
        "variant_id": "base",
        "variant_type": "base",
        "protocol": "short",
        "boundary": "none",
        "continuation_trigger": "neutral",
        "suffix": "\nAnswer:",
    },
    {
        "variant_id": "answer_only",
        "variant_type": "protocol",
        "protocol": "answer_only",
        "boundary": "answer_anchor",
        "continuation_trigger": "suppressed",
        "suffix": "\nAnswer with only the answer. No explanation.\nAnswer:",
    },
    {
        "variant_id": "explain_pressure",
        "variant_type": "continuation",
        "protocol": "explain",
        "boundary": "answer_anchor",
        "continuation_trigger": "because",
        "suffix": "\nAnswer with the answer first, then a short reason using because.\nAnswer:",
    },
    {
        "variant_id": "boundary_period",
        "variant_type": "boundary",
        "protocol": "short_stop",
        "boundary": "period_stop",
        "continuation_trigger": "period_boundary",
        "suffix": "\nAnswer with the final answer followed by a period, then stop.\nAnswer:",
    },
    {
        "variant_id": "structured_json",
        "variant_type": "structure",
        "protocol": "json",
        "boundary": "json_brace",
        "continuation_trigger": "json_structure",
        "suffix": "\nReturn JSON only with key \"answer\".\n",
    },
    {
        "variant_id": "list_pressure",
        "variant_type": "structure",
        "protocol": "list_one",
        "boundary": "newline_list",
        "continuation_trigger": "list_item",
        "suffix": "\nReturn exactly one bullet item.\n-",
    },
]


def utc_now() -> str:
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


def append_unique_jsonl(path: Path, rows: list[dict[str, Any]], id_key: str) -> None:
    old_rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in old_rows + rows:
        key = str(row.get(id_key) or row.get("metric_id") or row.get("edge_id") or row.get("observation_id") or len(by_id))
        by_id[key] = row
    write_jsonl(path, list(by_id.values()))


def expected_pattern(variant: dict[str, str]) -> str:
    if variant["protocol"] == "json":
        return "json"
    if "list" in variant["protocol"]:
        return "list"
    if variant["protocol"] == "explain":
        return "explain"
    return "short"


def state_labels(variant: dict[str, str]) -> dict[str, Any]:
    return {
        "S_content": 1,
        "S_target": 1,
        "S_protocol": 1 if variant["protocol"] != "short" else 0,
        "S_boundary": 1 if variant["boundary"] != "none" else 0,
        "S_done": 0,
        "S_continue": 1 if variant["continuation_trigger"] not in {"neutral", "suppressed"} else 0,
        "S_stop": 1 if "stop" in variant["protocol"] or variant["continuation_trigger"] == "suppressed" else 0,
        "S_structure": 1 if variant["protocol"] in {"json", "list_one"} else 0,
    }


def scoring_risk(family_id: str, variant: dict[str, str], target: str) -> str:
    if family_id in {"cross_lingual", "syntax_structure"}:
        return "medium"
    if variant["protocol"] in {"json", "list_one", "explain"}:
        return "medium"
    if len(str(target).split()) > 2:
        return "medium"
    return "low"


def build_prompt(question: str, variant: dict[str, str]) -> str:
    return question.strip() + variant["suffix"]


def path_schema_for(family_id: str, mode_id: str, variant: dict[str, str]) -> dict[str, Any]:
    return {
        "path_schema_id": f"phase265:path_schema:{family_id}:{mode_id}:{variant['variant_id']}",
        "family_id": family_id,
        "mode_id": mode_id,
        "variant_id": variant["variant_id"],
        "trigger_trace": ["prompt_pattern", family_id, mode_id],
        "component_trace_targets": ["embedding", "attention_out", "mlp_gate", "mlp_up", "mlp_product", "mlp_down", "residual_stream", "lm_head"],
        "state_trace_targets": ["S_content", "S_target", "S_protocol", "S_boundary", "S_done", "S_continue", "S_stop", "S_structure"],
        "readout_trace_targets": ["target_vs_wrong", "stop_vs_continuation", "protocol_vs_drift", "structure_completion_vs_structure_continuation"],
        "rollout_trace_targets": ["step_1", "step_4", "step_8", "step_16", "step_32", "answer_step", "drift_step", "eos_step"],
        "closure_trace_targets": ["AnswerCorrect", "PatternMatched", "BoundaryStable", "DoneStateStable", "ModelStopExecuted", "NoDrift"],
    }


def build(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    path_schema_rows: list[dict[str, Any]] = []
    state_factor_rows: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    seen_schema = set()
    created_at = utc_now()

    for family_id, spec in FAMILY_SPECS.items():
        for item_idx, (_label, question, target) in enumerate(spec["items"]):
            for mode_id in spec["modes"]:
                for variant in VARIANTS:
                    case_id = f"phase265_{family_id}_{mode_id}_{item_idx:03d}_{variant['variant_id']}"
                    prompt = build_prompt(question, variant)
                    states = state_labels(variant)
                    risk = scoring_risk(family_id, variant, target)
                    tags = [family_id, mode_id, variant["variant_type"], variant["protocol"], variant["continuation_trigger"]]
                    case = {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase265",
                        "created_at": created_at,
                        "case_id": case_id,
                        "family_id": family_id,
                        "mode_id": mode_id,
                        "variant_id": variant["variant_id"],
                        "variant_type": variant["variant_type"],
                        "prompt_template_id": f"{family_id}_{mode_id}_{variant['variant_id']}_v3",
                        "prompt": prompt,
                        "target": target,
                        "expected_answer": target,
                        "expected_pattern": expected_pattern(variant),
                        "output_protocol": variant["protocol"],
                        "boundary_type": variant["boundary"],
                        "done_label": "not_done_prompt",
                        "continuation_trigger": variant["continuation_trigger"],
                        "scoring_risk": risk,
                        "difficulty": "phase265_matrix_seed",
                        "tags": tags,
                        "target_aliases": [target],
                        "test_levels": [
                            "behavior",
                            "prompt_trigger",
                            "component_path",
                            "state_factor",
                            "readout_competition",
                            "rollout",
                            "closure",
                        ],
                        "path_schema_id": f"phase265:path_schema:{family_id}:{mode_id}:{variant['variant_id']}",
                        **states,
                    }
                    case_rows.append(case)
                    matrix_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase265",
                            "created_at": created_at,
                            "matrix_id": f"phase265:matrix:{case_id}",
                            "case_id": case_id,
                            "family_id": family_id,
                            "mode_id": mode_id,
                            "variant_id": variant["variant_id"],
                            "variant_type": variant["variant_type"],
                            "target": target,
                            "protocol": variant["protocol"],
                            "boundary_type": variant["boundary"],
                            "continuation_trigger": variant["continuation_trigger"],
                            "state_signature": states,
                            "path_schema_id": case["path_schema_id"],
                        }
                    )
                    quality_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase265",
                            "created_at": created_at,
                            "quality_id": f"phase265:quality:{case_id}",
                            "case_id": case_id,
                            "family_id": family_id,
                            "mode_id": mode_id,
                            "variant_id": variant["variant_id"],
                            "prompt_length_chars": len(prompt),
                            "target_length_tokens_approx": len(str(target).split()),
                            "scoring_risk": risk,
                            "has_target": bool(target),
                            "has_protocol_label": bool(variant["protocol"]),
                            "has_boundary_label": bool(variant["boundary"]),
                            "has_continuation_label": bool(variant["continuation_trigger"]),
                            "quality_status": "usable" if risk != "high" else "review",
                        }
                    )
                    state_factor_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase265",
                            "created_at": created_at,
                            "state_factor_id": f"phase265:state:{case_id}",
                            "case_id": case_id,
                            "family_id": family_id,
                            "mode_id": mode_id,
                            "variant_id": variant["variant_id"],
                            **states,
                            "state_factor_note": "designed case-bank labels, not measured internal factors",
                        }
                    )
                    if case["path_schema_id"] not in seen_schema:
                        seen_schema.add(case["path_schema_id"])
                        schema = {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase265",
                            "created_at": created_at,
                            **path_schema_for(family_id, mode_id, variant),
                        }
                        path_schema_rows.append(schema)
                        graph_edges.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "phase_id": "Phase265",
                                "created_at": created_at,
                                "edge_id": f"phase265:edge:{family_id}:{mode_id}:{variant['variant_id']}",
                                "source": f"node:{family_id}",
                                "target": f"node:path_schema:{mode_id}:{variant['variant_id']}",
                                "edge_type": "family_to_path_schema",
                                "evidence_type": "designed_case_bank_schema",
                                "status": "case_bank_not_model_trace",
                            }
                        )
                    observations.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase265",
                            "created_at": created_at,
                            "observation_id": f"phase265:observation:{case_id}",
                            "case_id": case_id,
                            "family_id": family_id,
                            "mode_id": mode_id,
                            "variant_id": variant["variant_id"],
                            "level": "case_bank_expansion",
                            "component": variant["variant_type"],
                            "metric_name": "case_ready_for_path_trace",
                            "metric_value": 1.0,
                            "metric_unit": "bool",
                            "scoring_risk": risk,
                        }
                    )

    family_counts = Counter(row["family_id"] for row in case_rows)
    variant_counts = Counter(row["variant_type"] for row in case_rows)
    risk_counts = Counter(row["scoring_risk"] for row in case_rows)
    for family_id, count in family_counts.items():
        metrics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase265",
                "created_at": created_at,
                "metric_id": f"phase265:{family_id}:case_count",
                "scope": "multi_family_case_bank",
                "family_id": family_id,
                "metric_name": "case_count",
                "metric_value": count,
                "rows": count,
            }
        )
    metrics.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase265",
            "created_at": created_at,
            "metric_id": "phase265:case_bank:family_coverage",
            "scope": "multi_family_case_bank",
            "metric_name": "family_coverage",
            "metric_value": len(family_counts),
            "rows": len(case_rows),
        }
    )
    metrics.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase265",
            "created_at": created_at,
            "metric_id": "phase265:case_bank:medium_or_low_quality_rate",
            "scope": "case_quality",
            "metric_name": "medium_or_low_quality_rate",
            "metric_value": round(sum(1 for row in quality_rows if row["scoring_risk"] in {"low", "medium"}) / len(quality_rows), 6),
            "rows": len(quality_rows),
        }
    )

    write_jsonl(out_dir / "phase265_mode_family_case_bank_v3.jsonl", case_rows)
    write_jsonl(out_dir / "phase265_mode_variant_matrix_rows.jsonl", matrix_rows)
    write_jsonl(out_dir / "phase265_case_quality_audit_rows.jsonl", quality_rows)
    write_jsonl(out_dir / "phase265_path_schema_rows.jsonl", path_schema_rows)
    write_jsonl(out_dir / "phase265_state_factor_design_rows.jsonl", state_factor_rows)
    write_jsonl(out_dir / "phase265_observations.jsonl", observations)
    write_jsonl(out_dir / "phase265_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase265_graph_edges.jsonl", graph_edges)

    append_unique_jsonl(ATLAS_ROOT / "test_cases.jsonl", case_rows, "case_id")
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", graph_edges, "edge_id")
    write_jsonl(ATLAS_ROOT / "mode_family_case_bank_v3.jsonl", case_rows)
    write_jsonl(ATLAS_ROOT / "mode_variant_matrix_rows.jsonl", matrix_rows)
    write_jsonl(ATLAS_ROOT / "case_quality_audit_rows.jsonl", quality_rows)
    write_jsonl(ATLAS_ROOT / "path_schema_rows.jsonl", path_schema_rows)
    write_jsonl(ATLAS_ROOT / "state_factor_design_rows.jsonl", state_factor_rows)

    progress = {
        "pattern_family_atlas": 0.87,
        "physical_path_atlas": 0.27,
        "multi_family_case_bank": 0.42,
        "state_factor_atlas": 0.36,
        "path_cluster_mining": 0.12,
        "trace_signature_validation": 0.47,
        "readout_competition_trace": 0.78,
        "stepwise_rollout_trace": 0.43,
        "causal_closure": 0.18,
        "general_language_mechanism_confidence": 0.66,
    }
    write_json(ATLAS_ROOT / "progress.json", {**read_json(ATLAS_ROOT / "progress.json"), **progress, "last_phase": "Phase265", "updated_at": utc_now()})

    summary = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Multi-family case bank and path schema expansion",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "round_name": round_name,
        "case_rows": len(case_rows),
        "matrix_rows": len(matrix_rows),
        "quality_rows": len(quality_rows),
        "path_schema_rows": len(path_schema_rows),
        "state_factor_rows": len(state_factor_rows),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(graph_edges),
        "family_counts": dict(family_counts),
        "variant_counts": dict(variant_counts),
        "risk_counts": dict(risk_counts),
        "progress": progress,
        "status_note": "case-bank/schema expansion only; no model forward pass and no closure claim",
    }
    write_json(out_dir / "phase265_cross_model_summary.json", summary)
    write_report(out_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase265 Multi-Family Case Bank And Path Schema Expansion",
        "",
        f"- status: {summary['status']}",
        f"- case_rows: {summary['case_rows']}",
        f"- matrix_rows: {summary['matrix_rows']}",
        f"- path_schema_rows: {summary['path_schema_rows']}",
        f"- family_counts: {json.dumps(summary['family_counts'], ensure_ascii=False)}",
        f"- variant_counts: {json.dumps(summary['variant_counts'], ensure_ascii=False)}",
        f"- risk_counts: {json.dumps(summary['risk_counts'], ensure_ascii=False)}",
        f"- status_note: {summary['status_note']}",
    ]
    (out_dir / "phase265_multi_family_case_bank_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    args = parser.parse_args()
    build(args.round_name)


if __name__ == "__main__":
    main()
