#!/usr/bin/env python3
"""Apply frozen Phase579 gates without changing coordinates or thresholds."""

from __future__ import annotations

import argparse
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
PROTOCOL_PATH = OUT_DIR / "phase579_option_routing_causal_protocol.json"
DISCOVERY_DECISION_PATH = (
    OUT_DIR / "phase579_option_routing_causal_discovery_decision.json"
)
CONFIRMATION_DECISION_PATH = (
    OUT_DIR / "phase579_option_routing_causal_confirmation_decision.json"
)
VARIANTS = ("target_first", "target_second")
RELATIONS = ("category", "outer_color")


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


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def condition_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margin_effects = [float(row["candidate_margin_effect"]) for row in rows]
    score_effects = [float(row["option_score_margin_effect"]) for row in rows]
    weight_effects = [float(row["option_weight_margin_effect"]) for row in rows]
    candidate_deltas = [float(row["maximum_candidate_score_delta"]) for row in rows]
    return {
        "case_count": len(rows),
        "world_count": len({row["world_id"] for row in rows}),
        "candidate_margin_effect_mean": mean(margin_effects),
        "candidate_margin_effect_negative_rate": rate(
            [value < 0.0 for value in margin_effects]
        ),
        "option_score_margin_effect_mean": mean(score_effects),
        "option_score_margin_effect_negative_rate": rate(
            [value < 0.0 for value in score_effects]
        ),
        "option_weight_margin_effect_mean": mean(weight_effects),
        "option_weight_margin_effect_negative_rate": rate(
            [value < 0.0 for value in weight_effects]
        ),
        "intervention_foil_win_rate": rate(
            [bool(row["intervention_foil_wins"]) for row in rows]
        ),
        "maximum_candidate_score_delta": max(candidate_deltas)
        if candidate_deltas
        else 0.0,
        "maximum_absolute_candidate_margin_effect": max(
            (abs(value) for value in margin_effects), default=0.0
        ),
        "maximum_absolute_option_weight_margin_effect": max(
            (abs(value) for value in weight_effects), default=0.0
        ),
    }


def analyze(stage: str) -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    split = "causal_discovery" if stage == "discovery" else "causal_confirmation"
    if stage == "confirmation":
        if not DISCOVERY_DECISION_PATH.exists():
            raise RuntimeError("Phase579 confirmation analysis requires discovery")
        discovery = read_json(DISCOVERY_DECISION_PATH)
        authorized_in = discovery["authorized_confirmation_relations_by_model"]
    else:
        authorized_in = {
            model: list(RELATIONS) for model in protocol["authorized_models"]
        }

    gates = protocol["discovery_gate"]
    model_results: dict[str, Any] = {}
    authorized_out: dict[str, list[str]] = {}
    for model in protocol["authorized_models"]:
        allowed = set(authorized_in.get(model, []))
        if not allowed:
            continue
        rows_path = OUT_DIR / f"phase579_{model}_{split}_option_routing_causal_rows.jsonl.gz"
        summary_path = OUT_DIR / f"phase579_{model}_{split}_option_routing_causal_summary.json"
        if not rows_path.exists() or not summary_path.exists():
            raise RuntimeError(f"Phase579 missing {stage} artifacts for {model}")
        summary = read_json(summary_path)
        if summary["rows_sha256"] != sha256_file(rows_path):
            raise RuntimeError(f"Phase579 {model}/{stage} row hash drift")
        if not summary["attention_reconstruction_pass"] or summary["sealed_split_read"]:
            raise RuntimeError(f"Phase579 {model}/{stage} failed quality boundary")

        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl_gz(rows_path):
            if row["model"] != model or row["split"] != split or row["sealed"]:
                raise RuntimeError(f"Phase579 invalid {model}/{stage} row")
            grouped[(row["relation"], row["variant"], row["condition"])].append(row)

        relation_results: dict[str, Any] = {}
        passed_relations = []
        for relation in RELATIONS:
            if relation not in allowed:
                continue
            by_variant: dict[str, Any] = {}
            variant_passes = []
            for variant in VARIANTS:
                metrics = {
                    condition: condition_metrics(grouped[(relation, variant, condition)])
                    for condition in protocol["conditions"]
                }
                score_swap = metrics["option_score_swap"]
                control = metrics["object_relation_score_swap_control"]
                score_restore = metrics["option_score_swap_restore"]
                weight_restore = metrics["option_weight_swap_restore"]
                enough_worlds = (
                    score_swap["world_count"] >= gates["minimum_relation_world_count"]
                )
                route_pass = (
                    score_swap["option_weight_margin_effect_negative_rate"]
                    >= gates["option_route_effect_negative_rate"]
                )
                candidate_pass = (
                    score_swap["candidate_margin_effect_negative_rate"]
                    >= gates["option_score_swap_margin_effect_negative_rate"]
                    and score_swap["candidate_margin_effect_mean"]
                    <= gates["option_score_swap_margin_effect_mean_maximum"]
                )
                specificity_gap = (
                    control["candidate_margin_effect_mean"]
                    - score_swap["candidate_margin_effect_mean"]
                )
                specificity_pass = (
                    specificity_gap
                    >= gates["score_swap_vs_nonoption_control_gap_minimum"]
                )
                restore_pass = (
                    score_restore["maximum_absolute_candidate_margin_effect"]
                    <= gates["restore_candidate_margin_maximum_absolute_delta"]
                    and score_restore["maximum_absolute_option_weight_margin_effect"]
                    <= gates["restore_route_margin_maximum_absolute_delta"]
                    and weight_restore["maximum_absolute_candidate_margin_effect"]
                    <= gates["restore_candidate_margin_maximum_absolute_delta"]
                    and weight_restore["maximum_absolute_option_weight_margin_effect"]
                    <= gates["restore_route_margin_maximum_absolute_delta"]
                )
                passed = bool(
                    enough_worlds
                    and route_pass
                    and candidate_pass
                    and specificity_pass
                    and restore_pass
                )
                variant_passes.append(passed)
                by_variant[variant] = {
                    "condition_metrics": metrics,
                    "score_swap_vs_nonoption_candidate_effect_gap": specificity_gap,
                    "enough_worlds": enough_worlds,
                    "physical_route_gate_pass": route_pass,
                    "candidate_effect_gate_pass": candidate_pass,
                    "specificity_gate_pass": specificity_pass,
                    "restore_gate_pass": restore_pass,
                    "pass": passed,
                }
            relation_pass = all(variant_passes)
            if relation_pass:
                passed_relations.append(relation)
            relation_results[relation] = {
                "both_option_orders_pass": relation_pass,
                "by_variant": by_variant,
            }
        model_results[model] = {
            "relation_results": relation_results,
            "passed_relations": passed_relations,
            "summary_sha256": sha256_file(summary_path),
            "rows_sha256": sha256_file(rows_path),
        }
        authorized_out[model] = passed_relations

    output_path = (
        DISCOVERY_DECISION_PATH
        if stage == "discovery"
        else CONFIRMATION_DECISION_PATH
    )
    decision = {
        "schema_version": f"phase579_option_routing_causal_{stage}_decision.v1",
        "phase_id": protocol["phase_id"],
        "created_at": now(),
        "status": "complete",
        "stage": stage,
        "split": split,
        "model_results": model_results,
        "authorized_confirmation_relations_by_model": authorized_out
        if stage == "discovery"
        else {},
        "confirmed_generation_relations_by_model": authorized_out
        if stage == "confirmation"
        else {},
        "any_branch_passed": any(authorized_out.values()),
        "coordinates_or_thresholds_changed_after_natural_discovery": False,
        "causal_discovery_internal_state_read": True,
        "causal_confirmation_internal_state_read": stage == "confirmation",
        "sealed_split_read": False,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
    }
    write_json(output_path, decision)
    print(
        json.dumps(
            {
                "stage": stage,
                "authorized_relations_by_model": authorized_out,
                "any_branch_passed": decision["any_branch_passed"],
                "sealed_split_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("discovery", "confirmation"))
    args = parser.parse_args()
    analyze(args.stage)


if __name__ == "__main__":
    main()
