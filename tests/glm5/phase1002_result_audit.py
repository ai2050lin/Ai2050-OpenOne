#!/usr/bin/env python3
"""Integrity and claim-boundary audit for all Phase 1002 results."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1002_multitoken_scpg_r2"
)
R1_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1002_multitoken_scpg"
)
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        yield float(value)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from iter_numbers(item)
        return
    if isinstance(value, list):
        for item in value:
            yield from iter_numbers(item)


def numeric_audit() -> dict[str, Any]:
    parsed_files = 0
    parsed_rows = 0
    nonfinite_count = 0
    for path in sorted(RESULT_ROOT.rglob("*")):
        if not path.is_file() or path.name == "final_audit.json":
            continue
        if path.suffix == ".json":
            objects = [read_json(path)]
        elif path.suffix == ".jsonl":
            objects = read_jsonl(path)
            parsed_rows += len(objects)
        else:
            continue
        parsed_files += 1
        for value in objects:
            nonfinite_count += sum(
                not math.isfinite(number)
                for number in iter_numbers(value)
            )
    return {
        "parsed_json_or_jsonl_files": parsed_files,
        "parsed_jsonl_rows": parsed_rows,
        "nonfinite_number_count": nonfinite_count,
        "pass": parsed_files > 0 and nonfinite_count == 0,
    }


def protocol_audit() -> dict[str, Any]:
    prereg = read_json(RESULT_ROOT / "preregistered_protocol.json")
    models = {}
    for model in MODELS:
        root = RESULT_ROOT / "protocol" / model
        cases = read_jsonl(root / "cases.jsonl")
        pairs = read_jsonl(root / "pairs.jsonl")
        discovery = read_jsonl(
            root / "discovery_selected_pairs.jsonl"
        )
        confirmation = read_jsonl(
            root / "confirmation_selected_pairs.jsonl"
        )
        discovery_worlds = {
            row["world_id"] for row in cases
            if row["split"] == "discovery"
        }
        confirmation_worlds = {
            row["world_id"] for row in cases
            if row["split"] == "confirmation"
        }
        discovery_names = {
            entity for row in cases
            if row["split"] == "discovery"
            for entity in row["base_entities"]
        }
        confirmation_names = {
            entity for row in cases
            if row["split"] == "confirmation"
            for entity in row["base_entities"]
        }
        discovery_pair_ids = {row["pair_id"] for row in discovery}
        confirmation_pair_ids = {
            row["pair_id"] for row in confirmation
        }
        checks = {
            "case_count_4096": len(cases) == 4096,
            "pair_count_2048": len(pairs) == 2048,
            "selected_128_per_split": (
                len(discovery) == 128
                and len(confirmation) == 128
            ),
            "selected_pairs_disjoint": not (
                discovery_pair_ids & confirmation_pair_ids
            ),
            "worlds_disjoint": not (
                discovery_worlds & confirmation_worlds
            ),
            "all_pairs_change_two_positions": all(
                len(row["changed_positions"]) == 2 for row in pairs
            ),
        }
        models[model] = {
            "checks": checks,
            "pass": all(checks.values()),
            "discovery_world_count": len(discovery_worlds),
            "confirmation_world_count": len(confirmation_worlds),
            "entity_name_overlap_count": len(
                discovery_names & confirmation_names
            ),
            "entity_vocabulary_is_not_holdout": bool(
                discovery_names & confirmation_names
            ),
        }
    r1_qwen = read_json(
        R1_ROOT / "behavior" / "qwen3" / "summary.json"
    )
    checks = {
        "protocol_revision_2": prereg["protocol_revision"] == 2,
        "calibration_not_reused": not prereg[
            "calibration_data_reused_in_formal_test"
        ],
        "r1_failure_preserved": not r1_qwen["behavior_gate_pass"],
        "all_model_protocols_pass": all(
            value["pass"] for value in models.values()
        ),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "models": models,
        "r1_qwen_discovery_exact_sentence": r1_qwen[
            "split_summary"
        ]["discovery"]["natural_exact_sentence_rate"],
        "r1_qwen_confirmation_exact_sentence": r1_qwen[
            "split_summary"
        ]["confirmation"]["natural_exact_sentence_rate"],
    }


def result_count_audit() -> dict[str, Any]:
    checks = {}
    for model in MODELS:
        checks[f"{model}_behavior_teacher_4096"] = (
            count_jsonl(
                RESULT_ROOT
                / "behavior"
                / model
                / "teacher_forced_rows.jsonl"
            )
            == 4096
        )
        checks[f"{model}_behavior_natural_4096"] = (
            count_jsonl(
                RESULT_ROOT
                / "behavior"
                / model
                / "natural_rows.jsonl"
            )
            == 4096
        )
        checks[f"{model}_source_controls_3072"] = (
            count_jsonl(
                RESULT_ROOT
                / "source_controls"
                / model
                / "rows.jsonl"
            )
            == 3072
        )
        checks[f"{model}_source_coverage_5632"] = (
            count_jsonl(
                RESULT_ROOT
                / "source_context_coverage"
                / model
                / "rows.jsonl"
            )
            == 5632
        )
        checks[f"{model}_temporal_generation_1024"] = (
            count_jsonl(
                RESULT_ROOT
                / "temporal_rollout"
                / model
                / "generation_rows.jsonl"
            )
            == 1024
        )
        checks[f"{model}_cache_audit_128"] = (
            count_jsonl(
                RESULT_ROOT
                / "temporal_rollout"
                / model
                / "cache_audit_rows.jsonl"
            )
            == 128
        )
    checks["qwen_bf16_kv_rows_128"] = (
        count_jsonl(
            RESULT_ROOT
            / "kv_cache_decomposition"
            / "qwen3_bf16"
            / "rows.jsonl"
        )
        == 128
    )
    return {"checks": checks, "pass": all(checks.values())}


def gate_summary() -> dict[str, Any]:
    sections = {
        "behavior": ("all_models_pass", None),
        "frozen_topology": ("cross_model_pass", "pass_count"),
        "temporal_rollout": ("cross_model_pass", "pass_count"),
        "kv_cache_decomposition": (
            "cross_model_pass",
            "pass_count",
        ),
        "kv_value_layer_localization": (
            "cross_model_pass",
            "pass_count",
        ),
        "kv_value_position_localization": (
            "cross_model_pass",
            "pass_count",
        ),
        "kv_value_background_refinement": (
            "cross_model_pass",
            "pass_count",
        ),
        "source_controls": ("cross_model_pass", "pass_count"),
        "source_context_coverage": (
            "cross_model_pass",
            "pass_count",
        ),
    }
    result = {}
    for section, (pass_key, count_key) in sections.items():
        summary = read_json(RESULT_ROOT / section / "summary.json")
        result[section] = {
            "cross_model_pass": bool(summary[pass_key]),
            "pass_count": (
                int(summary[count_key])
                if count_key is not None
                else 3 if summary[pass_key] else 0
            ),
        }
    qwen_bf16 = read_json(
        RESULT_ROOT
        / "kv_cache_decomposition"
        / "qwen3_bf16"
        / "summary.json"
    )
    result["qwen_bf16_kv_audit"] = {
        "cross_model_pass": qwen_bf16["cache_transport_pass"],
        "pass_count": 1 if qwen_bf16["cache_transport_pass"] else 0,
    }
    return result


def frozen_selection_audit() -> dict[str, Any]:
    values = {}
    for model in MODELS:
        summary = read_json(
            RESULT_ROOT
            / "frozen_topology"
            / model
            / "summary.json"
        )
        values[model] = {
            "selection_uses_phase1002": summary[
                "selection_uses_phase1002"
            ],
            "primary_pass": summary["primary_pass"],
        }
    return {
        "models": values,
        "pass": all(
            not value["selection_uses_phase1002"]
            for value in values.values()
        ),
    }


def source_structure_audit() -> dict[str, Any]:
    models = {}
    for model in MODELS:
        summary = read_json(
            RESULT_ROOT
            / "source_context_coverage"
            / model
            / "summary.json"
        )
        split_values = {}
        for split in ("discovery", "confirmation"):
            values = summary["split_summary"][split]
            split_values[split] = {
                "correct_entity_pair_source_rate": values[
                    "correct_entity_pair"
                ]["source_rate"],
                "third_answer_entity_pair_donor_rate": values[
                    "mismatch_entity_pair"
                ]["donor_rate"],
                "third_answer_semantic_anchors_donor_rate": values[
                    "mismatch_semantic_anchors"
                ]["donor_rate"],
                "full_context_donor_rate": values[
                    "mismatch_causal_all"
                ]["donor_rate"],
                "noop_agreement": values[
                    "target_noop_causal_all"
                ]["prediction_agreement"],
            }
        models[model] = {
            "splits": split_values,
            "donor_audit": summary["donor_audit"],
            "pass": summary["context_coverage_pass"],
        }
    repeated_checks = {
        "within_world_entity_pair_controls_output": all(
            values[
                "correct_entity_pair_source_rate"
            ] >= 0.99
            for model in models.values()
            for values in model["splits"].values()
        ),
        "cross_world_entity_pair_not_complete_packet": all(
            values[
                "third_answer_entity_pair_donor_rate"
            ] < 0.50
            for model in models.values()
            for values in model["splits"].values()
        ),
        "five_semantic_anchors_are_sufficient_here": all(
            values[
                "third_answer_semantic_anchors_donor_rate"
            ] >= 0.99
            for model in models.values()
            for values in model["splits"].values()
        ),
        "noop_exact": all(
            values["noop_agreement"] == 1.0
            for model in models.values()
            for values in model["splits"].values()
        ),
    }
    return {
        "models": models,
        "repeated_checks": repeated_checks,
        "pass": all(repeated_checks.values()),
        "claim": (
            "The two entity positions are a strong within-world causal "
            "control cut, not a complete cross-world semantic packet. "
            "The fixed entity, color, and query-name anchor set is "
            "sufficient for this task in all three models."
        ),
    }


def main() -> None:
    numeric = numeric_audit()
    protocol = protocol_audit()
    counts = result_count_audit()
    frozen = frozen_selection_audit()
    gates = gate_summary()
    source_structure = source_structure_audit()
    integrity_checks = {
        "numeric": numeric["pass"],
        "protocol": protocol["pass"],
        "counts": counts["pass"],
        "frozen_selection": frozen["pass"],
        "source_structure_recomputed": source_structure["pass"],
    }
    payload = {
        "schema_version": "phase1002_final_audit.v1",
        "phase": 1002,
        "status": "complete",
        "execution_integrity": {
            "checks": integrity_checks,
            "pass": all(integrity_checks.values()),
        },
        "numeric_audit": numeric,
        "protocol_audit": protocol,
        "result_count_audit": counts,
        "frozen_selection_audit": frozen,
        "scientific_gate_matrix": gates,
        "source_structure_audit": source_structure,
        "overall_classification": (
            "PARTIAL_GENERATION_ALIGNED_LOCAL_CAUSAL_TOPOLOGY"
        ),
        "closed_claims": [
            (
                "The fixed multi-token behavior denominator is valid for "
                "all three models."
            ),
            (
                "A source intervention applied once before generation is "
                "transported mainly through cache values in this task."
            ),
            (
                "Localized value-layer arms and a query-conditioned "
                "position bottleneck repeat functionally across models."
            ),
            (
                "Semantic-step restoration, not protocol-step "
                "restoration, carries natural answer recovery."
            ),
            source_structure["claim"],
        ],
        "claims_not_closed": [
            "complete language coding mechanism",
            "neuron-level mechanism",
            "global minimal circuit",
            "cross-model physical isomorphism",
            "general knowledge, syntax, or reasoning mechanism",
            "formula-level law of intelligence",
        ],
        "hard_failures_preserved": [
            "Qwen primary frozen receiver topology failed.",
            "Qwen temporal receiver-gated rollout remained diagnostic.",
            "Qwen fine background-region compression failed.",
            (
                "The preregistered semantic source control passed only "
                "one of three models."
            ),
            (
                "DeepSeek strict third-answer donors required limited "
                "reuse because a perfect per-template matching did not "
                "exist."
            ),
        ],
        "next_phase_needed": True,
        "next_phase": (
            "Freeze the five-anchor source state and localized value "
            "layers, then decompose cache-value KV heads and channels "
            "under cross-attribute, three-entity, and relation tasks."
        ),
        "automatic_continuation_decision": (
            "The necessary adaptive source-coverage follow-up was "
            "completed inside Phase 1002. Do not extend the current "
            "denominator post hoc; preregister Phase 1003 separately."
        ),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    output = RESULT_ROOT / "final_audit.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["execution_integrity"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
