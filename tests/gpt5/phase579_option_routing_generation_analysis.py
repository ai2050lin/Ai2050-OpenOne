#!/usr/bin/env python3
"""Evaluate Phase579 full generation with the frozen semantic gates."""

from __future__ import annotations

import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase578_choice_world"
PROTOCOL_PATH = OUT_DIR / "phase579_option_routing_generation_protocol.json"
DECISION_PATH = OUT_DIR / "phase579_option_routing_generation_decision.json"
VARIANTS = ("target_first", "target_second")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "target_rate": rate([row["semantic_event"] == "target" for row in rows]),
        "foil_rate": rate([row["semantic_event"] == "foil" for row in rows]),
        "unrecoverable_rate": rate(
            [row["semantic_event"] == "unrecoverable" for row in rows]
        ),
        "strict_sequence_rate": rate(
            [bool(row["strict_sequence_correct"]) for row in rows]
        ),
    }


def analyze() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    gates = protocol["generation_gate"]
    model_results: dict[str, Any] = {}
    passed_branches: dict[str, list[str]] = {}
    for model, relations in protocol["confirmed_relations_by_model"].items():
        rows_path = OUT_DIR / f"phase579_{model}_option_routing_generation_rows.jsonl.gz"
        summary_path = OUT_DIR / f"phase579_{model}_option_routing_generation_summary.json"
        summary = read_json(summary_path)
        if summary["rows_sha256"] != sha256_file(rows_path):
            raise RuntimeError(f"Phase579 generation row hash drift: {model}")
        if summary["sealed_split_read"]:
            raise RuntimeError("Phase579 generation crossed the sealed boundary")
        rows = list(read_jsonl_gz(rows_path))
        by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        by_case: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            if row["sealed"] or row["relation"] not in relations:
                raise RuntimeError(f"Phase579 invalid generation row: {model}")
            by_key[
                (
                    row["relation"],
                    row["variant"],
                    row["condition"],
                    row["execution_repeat"],
                )
            ].append(row)
            by_case[(row["case_id"], row["condition"], row["variant"])][
                row["execution_repeat"]
            ] = row

        repeat_exact_flags = []
        for repeat_rows in by_case.values():
            if set(repeat_rows) != set(protocol["repeats"]):
                raise RuntimeError(f"Phase579 incomplete generation repeats: {model}")
            values = [
                repeat_rows[repeat]["normalized_generated"]
                for repeat in protocol["repeats"]
            ]
            repeat_exact_flags.append(len(set(values)) == 1)
        repeat_exact_rate = rate(repeat_exact_flags)

        relation_results = {}
        model_passed = []
        for relation in relations:
            variant_results = {}
            variant_passes = []
            for variant in VARIANTS:
                condition_metrics = {
                    condition: {
                        repeat: summarize(
                            by_key[(relation, variant, condition, repeat)]
                        )
                        for repeat in protocol["repeats"]
                    }
                    for condition in protocol["conditions"]
                }
                natural_target_floor = min(
                    condition_metrics["natural_baseline"][repeat]["target_rate"]
                    for repeat in protocol["repeats"]
                )
                natural_foil_ceiling = max(
                    condition_metrics["natural_baseline"][repeat]["foil_rate"]
                    for repeat in protocol["repeats"]
                )
                swap_target_ceiling = max(
                    condition_metrics["option_score_swap"][repeat]["target_rate"]
                    for repeat in protocol["repeats"]
                )
                swap_foil_floor = min(
                    condition_metrics["option_score_swap"][repeat]["foil_rate"]
                    for repeat in protocol["repeats"]
                )
                control_target_floor = min(
                    condition_metrics["object_relation_score_swap_control"][repeat][
                        "target_rate"
                    ]
                    for repeat in protocol["repeats"]
                )
                target_drop = natural_target_floor - swap_target_ceiling
                foil_gain = swap_foil_floor - natural_foil_ceiling
                control_gap = control_target_floor - swap_target_ceiling

                restore_flags = []
                for row in rows:
                    if row["relation"] != relation or row["variant"] != variant:
                        continue
                    if row["condition"] != "natural_baseline":
                        continue
                    restore = next(
                        candidate
                        for candidate in rows
                        if candidate["case_id"] == row["case_id"]
                        and candidate["variant"] == variant
                        and candidate["condition"] == "option_score_swap_restore"
                        and candidate["execution_repeat"] == row["execution_repeat"]
                    )
                    restore_flags.append(
                        restore["normalized_generated"] == row["normalized_generated"]
                    )
                restore_exact_rate = rate(restore_flags)
                passed = bool(
                    natural_target_floor
                    >= gates["minimum_natural_target_rate_each_order"]
                    and target_drop
                    >= gates["minimum_score_swap_target_rate_drop_each_order"]
                    and foil_gain
                    >= gates["minimum_score_swap_foil_rate_gain_each_order"]
                    and control_gap
                    >= gates[
                        "minimum_score_swap_vs_nonoption_target_drop_gap_each_order"
                    ]
                    and repeat_exact_rate
                    >= gates["minimum_repeat_exact_match_rate"]
                    and restore_exact_rate
                    >= gates["minimum_restore_exact_match_to_natural_rate"]
                )
                variant_passes.append(passed)
                variant_results[variant] = {
                    "condition_metrics": condition_metrics,
                    "natural_target_rate_floor": natural_target_floor,
                    "score_swap_target_rate_ceiling": swap_target_ceiling,
                    "score_swap_foil_rate_floor": swap_foil_floor,
                    "score_swap_target_rate_drop": target_drop,
                    "score_swap_foil_rate_gain": foil_gain,
                    "score_swap_vs_nonoption_target_drop_gap": control_gap,
                    "repeat_exact_match_rate_all_conditions": repeat_exact_rate,
                    "restore_exact_match_to_natural_rate": restore_exact_rate,
                    "pass": passed,
                }
            relation_pass = all(variant_passes)
            if relation_pass:
                model_passed.append(relation)
            relation_results[relation] = {
                "both_option_orders_pass": relation_pass,
                "by_variant": variant_results,
            }
        model_results[model] = {
            "relation_results": relation_results,
            "passed_relations": model_passed,
            "repeat_exact_match_rate_all_conditions": repeat_exact_rate,
            "rows_sha256": sha256_file(rows_path),
            "summary_sha256": sha256_file(summary_path),
        }
        passed_branches[model] = model_passed

    decision = {
        "schema_version": "phase579_option_routing_generation_decision.v1",
        "phase_id": protocol["phase_id"],
        "created_at": now(),
        "status": "complete",
        "model_results": model_results,
        "passed_generation_relations_by_model": passed_branches,
        "any_generation_branch_passed": any(passed_branches.values()),
        "sealed_validation_authorized": any(passed_branches.values()),
        "candidate_logit_causal_effect_does_not_imply_generation_effect": True,
        "sealed_split_read": False,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
    }
    write_json(DECISION_PATH, decision)
    print(
        json.dumps(
            {
                "passed_generation_relations_by_model": passed_branches,
                "any_generation_branch_passed": decision[
                    "any_generation_branch_passed"
                ],
                "sealed_validation_authorized": decision[
                    "sealed_validation_authorized"
                ],
                "sealed_split_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return decision


if __name__ == "__main__":
    analyze()
