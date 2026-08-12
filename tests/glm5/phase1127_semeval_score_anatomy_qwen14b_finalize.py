#!/usr/bin/env python3
"""Finalize Phase1127 score anatomy and Qwen3-14B behavior gates."""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1126_semeval_lexsub_natural_cloze_protocol as source_protocol
import phase1127_semeval_score_anatomy_qwen14b_protocol as protocol


MODEL_PATHS = {
    "qwen3_4b": source_protocol.OUT_ROOT / "behavior" / "qwen3",
    "glm4": source_protocol.OUT_ROOT / "behavior" / "glm4",
    "deepseek7b": source_protocol.OUT_ROOT / "behavior" / "deepseek7b",
    "qwen3_14b": protocol.OUT_ROOT / "behavior" / "qwen3_14b",
}
FIELD_BY_COMPONENT = {
    "candidate": "candidate_logp",
    "suffix": "suffix_mean_logp",
    "total": "total_score",
}


def median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def meets(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def finite_value(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def interaction(table: dict[tuple[str, str], float]) -> float:
    return 0.5 * (
        (table[("a", "a")] - table[("a", "b")])
        + (table[("b", "b")] - table[("b", "a")])
    )


def summarize_component(
    units: list[dict[str, Any]],
    component: str,
    finite_rate: float,
    thresholds: dict[str, float],
    include_lexical: bool,
) -> dict[str, Any]:
    complete = [row for row in units if row[f"{component}_complete"]]
    active = [float(row[f"{component}_active_z"]) for row in complete]
    matched = [float(row[f"{component}_matched_z"]) for row in complete]
    advantages = [a - abs(n) for a, n in zip(active, matched)]
    active_median = median(active)
    advantage_median = median(advantages)
    metrics: dict[str, Any] = {
        "unit_count": len(units),
        "complete_unit_count": len(complete),
        "finite_rate": finite_rate,
        "active_mean": mean(active),
        "active_median": active_median,
        "active_positive_rate": sum(value > 0.0 for value in active) / len(active) if active else 0.0,
        "matched_mean": mean(matched),
        "matched_abs_median": median([abs(value) for value in matched]),
        "matched_advantage_mean": mean(advantages),
        "matched_advantage_median": advantage_median,
        "matched_advantage_positive_rate": (
            sum(value > 0.0 for value in advantages) / len(advantages) if advantages else 0.0
        ),
    }
    gates = {
        "finite": finite_rate >= thresholds["finite_rate_min"],
        "active_positive": metrics["active_positive_rate"] >= thresholds["active_positive_rate_min"],
        "active_median": meets(active_median, thresholds["active_median_min"]),
        "matched_advantage_median": meets(
            advantage_median, thresholds["matched_advantage_median_min"]
        ),
        "matched_advantage_positive": (
            metrics["matched_advantage_positive_rate"]
            >= thresholds["matched_advantage_positive_rate_min"]
        ),
    }
    if include_lexical:
        lexical = [float(row["lexical_active_z"]) for row in complete]
        lexical_abs_median = median([abs(value) for value in lexical])
        lexical_advantage = (
            active_median - lexical_abs_median
            if active_median is not None and lexical_abs_median is not None
            else None
        )
        metrics["lexical_active_abs_median"] = lexical_abs_median
        metrics["lexical_zero_advantage"] = lexical_advantage
        gates["lexical_zero_advantage"] = meets(
            lexical_advantage, thresholds["lexical_zero_advantage_min"]
        )
    return {"metrics": metrics, "gates": gates, "passed": all(gates.values())}


def build_units(model: str, details: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[(row["partition"], int(row["panel_index"]), int(row["replica"]), row["route"])].append(row)

    units: list[dict[str, Any]] = []
    additivity_errors: list[float] = []
    for partition in protocol.PARTITIONS:
        panel_replicas = sorted({
            (key[1], key[2]) for key in grouped if key[0] == partition
        })
        for panel_index, replica in panel_replicas:
            route_values: dict[str, dict[str, float | None]] = {
                component: {} for component in protocol.COMPONENTS
            }
            lexical_values: dict[str, float] = {}
            item = None
            pos = None
            for route in protocol.ROUTES:
                rows = grouped[(partition, panel_index, replica, route)]
                if len(rows) != 4:
                    raise RuntimeError(f"incomplete route: {model}/{partition}/{panel_index}/{replica}/{route}")
                for component, field in FIELD_BY_COMPONENT.items():
                    values = {
                        (row["context_sense"], row["candidate_side"]): finite_value(row, field)
                        for row in rows
                    }
                    route_values[component][route] = (
                        interaction({key: float(value) for key, value in values.items()})
                        if all(value is not None for value in values.values())
                        else None
                    )
                lexical_table = {
                    (row["context_sense"], row["candidate_side"]): float(row["lexical_overlap"])
                    for row in rows
                }
                lexical_values[route] = interaction(lexical_table)
                item = rows[0]["item"]
                pos = rows[0]["pos"]

            unit: dict[str, Any] = {
                "model": model,
                "partition": partition,
                "panel_index": panel_index,
                "replica": replica,
                "item": item,
                "pos": pos,
                "lexical_active_z": lexical_values["active"],
                "lexical_matched_z": lexical_values["matched_deranged"],
            }
            for component in protocol.COMPONENTS:
                active = route_values[component]["active"]
                matched = route_values[component]["matched_deranged"]
                complete = active is not None and matched is not None
                unit[f"{component}_active_z"] = active
                unit[f"{component}_matched_z"] = matched
                unit[f"{component}_matched_advantage"] = active - abs(matched) if complete else None
                unit[f"{component}_complete"] = complete
            if all(unit[f"{component}_complete"] for component in protocol.COMPONENTS):
                for route in ("active", "matched"):
                    error = abs(
                        float(unit[f"total_{route}_z"])
                        - float(unit[f"candidate_{route}_z"])
                        - float(unit[f"suffix_{route}_z"])
                    )
                    additivity_errors.append(error)
            units.append(unit)

    finite_anatomy = {}
    for component, field in FIELD_BY_COMPONENT.items():
        finite_count = sum(finite_value(row, field) is not None for row in details)
        finite_anatomy[component] = {
            "finite_count": finite_count,
            "case_count": len(details),
            "finite_rate": finite_count / len(details),
        }
    finite_anatomy["candidate_only_nonfinite"] = sum(
        finite_value(row, "candidate_logp") is None and finite_value(row, "suffix_mean_logp") is not None
        for row in details
    )
    finite_anatomy["suffix_only_nonfinite"] = sum(
        finite_value(row, "candidate_logp") is not None and finite_value(row, "suffix_mean_logp") is None
        for row in details
    )
    finite_anatomy["both_components_nonfinite"] = sum(
        finite_value(row, "candidate_logp") is None and finite_value(row, "suffix_mean_logp") is None
        for row in details
    )
    diagnostics = {
        "finite_anatomy": finite_anatomy,
        "additivity_count": len(additivity_errors),
        "additivity_max_abs_error": max(additivity_errors) if additivity_errors else None,
        "additivity_median_abs_error": median(additivity_errors),
    }
    return units, diagnostics


def anatomy_summary(units: list[dict[str, Any]], partition: str) -> dict[str, Any]:
    rows = [row for row in units if row["partition"] == partition and all(
        row[f"{component}_complete"] for component in protocol.COMPONENTS
    )]
    shares = []
    same_sign = []
    candidate_total_sign = []
    suffix_total_sign = []
    for row in rows:
        candidate = float(row["candidate_active_z"])
        suffix = float(row["suffix_active_z"])
        total = float(row["total_active_z"])
        denominator = abs(candidate) + abs(suffix)
        if denominator > 0.0:
            shares.append(abs(candidate) / denominator)
        same_sign.append((candidate > 0.0) == (suffix > 0.0))
        candidate_total_sign.append((candidate > 0.0) == (total > 0.0))
        suffix_total_sign.append((suffix > 0.0) == (total > 0.0))
    count = len(rows)
    return {
        "complete_unit_count": count,
        "candidate_absolute_share_median": median(shares),
        "candidate_suffix_same_sign_rate": sum(same_sign) / count if count else 0.0,
        "candidate_total_sign_agreement": sum(candidate_total_sign) / count if count else 0.0,
        "suffix_total_sign_agreement": sum(suffix_total_sign) / count if count else 0.0,
    }


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["passed"] or audit["protocol_digest"] != prereg["protocol_digest"]:
        raise RuntimeError("Phase1127 protocol is not authorized")

    all_units: list[dict[str, Any]] = []
    model_results: dict[str, Any] = {}
    all_additivity_errors: list[float] = []
    for model, path in MODEL_PATHS.items():
        scores_path = path / "scores.jsonl"
        summary_path = path / "summary.json"
        if not scores_path.exists() or not summary_path.exists():
            raise RuntimeError(f"missing behavior output: {model}")
        details = protocol.read_jsonl(scores_path)
        summary = protocol.read_json(summary_path)
        expected_digest = (
            prereg["source"]["source_score_digests"][model.replace("_4b", "")]
            if model != "qwen3_14b"
            else summary["detail_digest"]
        )
        source_digest = source_protocol.digest(details) if model != "qwen3_14b" else protocol.digest(details)
        if source_digest != expected_digest:
            raise RuntimeError(f"detail digest mismatch: {model}")
        if model == "qwen3_14b" and summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError("Qwen3-14B protocol mismatch")

        units, diagnostics = build_units(model, details)
        all_units.extend(units)
        if diagnostics["additivity_max_abs_error"] is not None:
            all_additivity_errors.append(float(diagnostics["additivity_max_abs_error"]))
        partitions: dict[str, Any] = {}
        for partition in protocol.PARTITIONS:
            partition_details = [row for row in details if row["partition"] == partition]
            partition_units = [row for row in units if row["partition"] == partition]
            component_results = {}
            for component, field in FIELD_BY_COMPONENT.items():
                finite_rate = sum(finite_value(row, field) is not None for row in partition_details) / len(partition_details)
                component_results[component] = summarize_component(
                    partition_units,
                    component,
                    finite_rate,
                    prereg["thresholds"] if component == "total" else prereg["component_thresholds"],
                    include_lexical=component == "total",
                )
            partitions[partition] = {
                "components": component_results,
                "anatomy": anatomy_summary(units, partition),
            }
        model_results[model] = {
            "source_summary": summary,
            "diagnostics": diagnostics,
            "partitions": partitions,
            "total_passed_both_partitions": all(
                partitions[partition]["components"]["total"]["passed"]
                for partition in protocol.PARTITIONS
            ),
            "candidate_passed_both_partitions": all(
                partitions[partition]["components"]["candidate"]["passed"]
                for partition in protocol.PARTITIONS
            ),
            "suffix_passed_both_partitions": all(
                partitions[partition]["components"]["suffix"]["passed"]
                for partition in protocol.PARTITIONS
            ),
        }

    max_additivity_error = max(all_additivity_errors) if all_additivity_errors else None
    predictions = {
        "P1_inputs_and_protocol": bool(audit["passed"]),
        "P2_score_additivity": (
            max_additivity_error is not None
            and max_additivity_error <= float(prereg["score_identity"]["additivity_tolerance"])
        ),
        "P3_qwen4_candidate_component": model_results["qwen3_4b"]["candidate_passed_both_partitions"],
        "P4_qwen4_suffix_component": model_results["qwen3_4b"]["suffix_passed_both_partitions"],
        "P5_qwen14_total_behavior": model_results["qwen3_14b"]["total_passed_both_partitions"],
        "P6_same_family_replication": (
            model_results["qwen3_4b"]["total_passed_both_partitions"]
            and model_results["qwen3_14b"]["total_passed_both_partitions"]
        ),
        "P7_phase1126_cross_model_gate_reopened": False,
        "P8_hidden_authorized": False,
    }
    final_core = {
        "schema_version": "phase1127_semeval_score_anatomy_qwen14b_final.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "source_phase1126_final_digest": prereg["source"]["phase1126_final_digest"],
        "model_results": model_results,
        "score_additivity_max_abs_error": max_additivity_error,
        "predictions": predictions,
        "auto_continue": {
            "value": False,
            "reason": (
                "The Phase1126 fixed cross-model gate remains failed. A same-family 14B endpoint and a post hoc "
                "score decomposition cannot authorize hidden_holdout, hidden-state, component, or causal work."
            ),
        },
        "evidence_boundary": {
            "score_anatomy": "post hoc diagnostic over frozen outputs",
            "qwen14": "prospective same-family behavior endpoint",
            "not_authorized": ["cross-architecture E3", "hidden representation", "component", "causal mechanism"],
        },
        "unit_digest": protocol.digest(all_units),
    }
    final = dict(final_core)
    final["final_digest"] = protocol.digest(final_core)
    protocol.write_jsonl(protocol.OUT_ROOT / "analysis" / "interaction_units.jsonl", all_units)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
