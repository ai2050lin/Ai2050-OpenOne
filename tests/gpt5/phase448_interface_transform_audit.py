#!/usr/bin/env python3
"""Phase448 static interface and surface-transform audit.

This stage is protocol analysis only. It does not load models, use CUDA, or
collect physical traces.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SAMPLES_PATH = ROOT / "tests" / "gpt5" / "result" / "phase446_antishortcut_static_contract" / "phase446_samples.jsonl"
PHASE447_PATH = ROOT / "tests" / "gpt5" / "result" / "phase447_failure_decomposition" / "phase447_failure_decomposition.json"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase448_interface_transform_audit"
OUT_PATH = OUT_DIR / "phase448_interface_transform_audit.json"

TRANSFORM_DESIGN_INTENT = {
    "lexical_rewrite": "rename frame words while preserving fact-query order",
    "order_rewrite": "place query before evidence",
    "distance_rewrite": "insert neutral text between facts and query",
    "boundary_rewrite": "compress fact sentence boundaries with semicolons",
    "syntax_rewrite": "explicitly name facts and statement with a stronger instruction frame",
    "query_rewrite": "replace the queried-statement marker with a natural question frame",
}

RECOMMENDED_REPLACEMENTS = {
    "order_rewrite": "Keep evidence before the query; vary only local wording, not evidence-query order.",
    "boundary_rewrite": "Preserve one sentence per fact; test boundary separately with harmless whitespace or bullet separators.",
    "query_rewrite": "If the query marker is changed, also rewrite the final instruction to point to 'the claim' or 'the statement above'.",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def text_metrics(text: str) -> dict[str, Any]:
    lower = text.lower()
    queried_count = len(re.findall(r"\bqueried statement\b", lower))
    evidence_count = len(re.findall(r"\bevidence\b", lower))
    facts_frame = int("use only the following facts" in lower or "given these records" in lower)
    query_markers = [
        marker for marker in [
            "queried statement:",
            "check this claim:",
            "the statement to judge is:",
            "decide whether this is true:",
        ]
        if marker in lower
    ]
    query_pos = min((lower.find(marker) for marker in query_markers if lower.find(marker) >= 0), default=-1)
    evidence_pos = lower.find("evidence:")
    instruction_pos = lower.find("reply with a")
    return {
        "chars": len(text),
        "sentences_approx": sum(text.count(mark) for mark in [".", "?", "!"]),
        "semicolon_count": text.count(";"),
        "queried_statement_marker_count": queried_count,
        "query_marker_count": len(query_markers),
        "evidence_marker_count": evidence_count,
        "has_facts_frame": bool(facts_frame),
        "query_before_evidence": query_pos >= 0 and evidence_pos >= 0 and query_pos < evidence_pos,
        "query_after_instruction": query_pos >= 0 and instruction_pos >= 0 and query_pos > instruction_pos,
        "instruction_mentions_queried_statement_without_marker": (
            "queried statement" in lower
            and "queried statement:" not in lower
            and "reply with a if the queried statement is true" in lower
        ),
    }


def audit_surface_variants(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_transform: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for variant in row["surface_variants"]:
            transform = variant["transform"]
            metrics = text_metrics(variant["text"])
            for key, value in metrics.items():
                if isinstance(value, bool):
                    by_transform[transform][key] += int(value)
                elif isinstance(value, int):
                    by_transform[transform][f"{key}_sum"] += value
            by_transform[transform]["n"] += 1
            risk_reasons = risk_reasons_for_transform(transform, metrics)
            if risk_reasons and len(examples[transform]) < 5:
                examples[transform].append({
                    "sample_id": row["sample_id"],
                    "risk_reasons": "; ".join(risk_reasons),
                    "text": variant["text"],
                })

    report = {}
    for transform, counts in sorted(by_transform.items()):
        n = counts["n"]
        report[transform] = {
            "design_intent": TRANSFORM_DESIGN_INTENT.get(transform, ""),
            "n": n,
            "query_before_evidence_rate": counts["query_before_evidence"] / n if n else 0.0,
            "instruction_marker_mismatch_rate": counts["instruction_mentions_queried_statement_without_marker"] / n if n else 0.0,
            "avg_semicolon_count": counts["semicolon_count_sum"] / n if n else 0.0,
            "avg_sentence_mark_count": counts["sentences_approx_sum"] / n if n else 0.0,
            "interface_risk": transform in {"order_rewrite", "boundary_rewrite", "query_rewrite"},
            "examples": examples[transform],
        }
    return report


def risk_reasons_for_transform(transform: str, metrics: dict[str, Any]) -> list[str]:
    reasons = []
    if metrics["query_before_evidence"]:
        reasons.append("query appears before evidence")
    if metrics["instruction_mentions_queried_statement_without_marker"]:
        reasons.append("final instruction names queried statement after the marker was removed")
    if transform == "boundary_rewrite" and metrics["semicolon_count"] >= 3:
        reasons.append("multiple fact boundaries are compressed into one semicolon chain")
    return reasons


def phase447_failure_table(phase447: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in phase447["by_model_task_transform"]:
        task_key = f"{row['model']}/{row['ability']}/{row['task']}"
        out.setdefault(task_key, {})[row["transform"]] = {
            "n": row["n"],
            "semantic": row["semantic"],
            "wrong": row["wrong"],
            "other": row["other"],
            "semantic_rate": row["semantic_rate"],
        }
    return out


def redesign_matrix(surface_report: dict[str, Any], phase447: dict[str, Any]) -> list[dict[str, Any]]:
    failure_table = phase447_failure_table(phase447)
    rows = []
    target_task = "glm4/knowledge_network/relation_truth_judgment"
    target_failures = failure_table.get(target_task, {})
    for transform, surface in sorted(surface_report.items()):
        behavior = target_failures.get(transform, {})
        semantic_rate = behavior.get("semantic_rate")
        risk = surface["interface_risk"]
        priority = "low"
        if risk and semantic_rate is not None and semantic_rate <= 0.55:
            priority = "critical"
        elif semantic_rate is not None and semantic_rate <= 0.75:
            priority = "high"
        elif risk:
            priority = "medium"
        rows.append({
            "transform": transform,
            "glm4_knowledge_semantic_rate": semantic_rate,
            "interface_risk": risk,
            "priority": priority,
            "diagnosis": diagnose_transform(transform, semantic_rate, surface),
            "recommended_replacement": RECOMMENDED_REPLACEMENTS.get(transform, "Keep as a control transform unless later behavior data contradicts it."),
        })
    return rows


def diagnose_transform(transform: str, semantic_rate: float | None, surface: dict[str, Any]) -> str:
    if transform == "order_rewrite":
        return "High risk: query-before-evidence changes reading order and may induce premature answer formation."
    if transform == "boundary_rewrite":
        return "High risk: semicolon compression removes clean fact boundaries and may collapse membership and trait statements."
    if transform == "query_rewrite":
        return "High risk: query marker is removed while the final instruction still refers to the queried statement."
    if transform == "syntax_rewrite" and semantic_rate == 1.0:
        return "Best control: stronger facts/statement frame improves surface stability without changing logic."
    if transform in {"lexical_rewrite", "distance_rewrite"}:
        return "Moderate control: mostly preserves evidence-query order and marker clarity."
    return "No special diagnosis."


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(SAMPLES_PATH)
    phase447 = load_json(PHASE447_PATH)
    surface_report = audit_surface_variants(rows)
    matrix = redesign_matrix(surface_report, phase447)
    out = {
        "schema_version": "phase448_interface_transform_audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "static_protocol_audit_no_model_run_no_cuda_no_physical_trace",
        "source_phase": ["phase446_static_contract", "phase447_failure_decomposition"],
        "surface_transform_audit": surface_report,
        "glm4_knowledge_redesign_matrix": matrix,
        "authorization": {
            "model_rerun_authorized": False,
            "physical_trace_authorized": False,
            "next_step": "freeze_phase448_v2_interface_protocol_before_any_model_rerun",
        },
        "summary": {
            "critical_transforms": [
                item["transform"] for item in matrix if item["priority"] == "critical"
            ],
            "best_control_transform": "syntax_rewrite",
            "main_conclusion": (
                "The three failing GLM4 knowledge transforms overlap with interface-level risk: "
                "query-before-evidence order, semicolon boundary compression, and query marker/instruction mismatch."
            ),
        },
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
