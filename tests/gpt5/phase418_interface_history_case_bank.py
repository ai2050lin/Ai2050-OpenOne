#!/usr/bin/env python3
"""Freeze the Phase418 interface-by-history paired denominator.

The design deliberately records prompt serialization and token-length changes.
It is a paired factorial audit, not a claim that chat/completion histories can be
made literally orthogonal at the token level.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "tests/gpt5/result/phase416_formal_world_physical_atlas"
SOURCE_CASES = SOURCE_ROOT / "phase416_registered_cases.jsonl"
PHASE417_ROOT = ROOT / "tests/gpt5/result/phase417_native_generation_physical_atlas"
OUT = ROOT / "tests/gpt5/result/phase418_interface_history_atlas"
PHASE_ID = "Phase418-InterfaceHistoryRegisteredDenominator"
SCHEMA_VERSION = "92.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
INTERFACES = ("chat", "completion")
HISTORIES = ("none", "compatible", "irrelevant", "conflict", "override")
FAMILY_QUOTA = 10


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def phase417_prerequisites() -> dict[str, Any]:
    cells = {}
    for model in MODELS:
        path = PHASE417_ROOT / "models" / model / "phase417_native_generation_complete.json"
        payload = read_json(path)
        cells[model] = {
            "case_count": payload["case_count"],
            "pass_count": payload["native_generation_case_pass_count"],
            "qualified": bool(payload["native_generation_qualification_pass"]),
        }
    return cells


def round_robin_family_ids(qwen_rows: list[dict[str, Any]], family_id: str) -> list[str]:
    """Select ten cases while balancing mechanisms and frozen split labels."""
    family_rows = [row for row in qwen_rows if row["family_id"] == family_id]
    groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(
        family_rows,
        key=lambda item: (
            item["mechanism_id"],
            0 if item["split"] == "discovery" else 1,
            int(item["item_index"]),
            item["template_id"],
        ),
    ):
        groups[row["mechanism_id"]].append(row)

    selected: list[str] = []
    mechanism_order = sorted(groups)
    while len(selected) < FAMILY_QUOTA:
        changed = False
        for mechanism in mechanism_order:
            if groups[mechanism] and len(selected) < FAMILY_QUOTA:
                selected.append(groups[mechanism].popleft()["semantic_case_id"])
                changed = True
        if not changed:
            break
    if len(selected) != FAMILY_QUOTA:
        raise RuntimeError(f"Could not select {FAMILY_QUOTA} cases for {family_id}: {len(selected)}")
    return selected


def freeze() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    source_rows = read_jsonl(SOURCE_CASES)
    qwen_rows = [row for row in source_rows if row["model"] == "qwen3"]
    families = sorted({row["family_id"] for row in qwen_rows})
    if len(families) != 4:
        raise RuntimeError(f"Expected four formal families, found {families}")

    selected_by_family = {
        family: round_robin_family_ids(qwen_rows, family) for family in families
    }
    selected_ids = {semantic_id for values in selected_by_family.values() for semantic_id in values}
    source_index = {(row["model"], row["semantic_case_id"]): row for row in source_rows}
    created_at = now()
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for family in families:
            for semantic_id in selected_by_family[family]:
                source = source_index.get((model, semantic_id))
                if source is None:
                    raise RuntimeError(f"Missing aligned Phase416 source: {model}/{semantic_id}")
                for interface in INTERFACES:
                    for history in HISTORIES:
                        condition_id = f"phase418_{model}_{semantic_id}_{interface}_{history}"
                        rows.append(
                            {
                                **source,
                                "schema_version": SCHEMA_VERSION,
                                "phase_id": PHASE_ID,
                                "created_at": created_at,
                                "phase418_condition_id": condition_id,
                                "phase418_source_case_id": source["case_id"],
                                "interface": interface,
                                "history_condition": history,
                                "history_present": history != "none",
                                "pure_history_level": history != "override",
                                "composite_override_condition": history == "override",
                                "registered_pairing_key": f"{model}:{semantic_id}",
                                "prompt_serialization_frozen_but_model_local": True,
                                "token_length_equality_required": False,
                                "terminal_suffix_token_alignment_required": True,
                                "descriptive_physical_collection_authorized": True,
                                "causal_intervention_authorized": False,
                                "single_neuron_scan_authorized": False,
                            }
                        )

    rows.sort(
        key=lambda row: (
            MODELS.index(row["model"]),
            row["family_id"],
            row["semantic_case_id"],
            INTERFACES.index(row["interface"]),
            HISTORIES.index(row["history_condition"]),
        )
    )
    prerequisites = phase417_prerequisites()
    counts = Counter((row["model"], row["family_id"]) for row in rows)
    valid = bool(
        len(rows) == 1200
        and len(selected_ids) == 40
        and all(cell["qualified"] for cell in prerequisites.values())
        and all(counts[(model, family)] == 100 for model in MODELS for family in families)
        and len({row["phase418_condition_id"] for row in rows}) == len(rows)
    )
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": created_at,
        "objective": "map_paired_interface_history_physical_differences_without_claiming_token_level_orthogonality",
        "model_order": list(MODELS),
        "families": families,
        "semantic_cases_per_family": FAMILY_QUOTA,
        "semantic_cases_per_model": 40,
        "conditions_per_semantic_case": 10,
        "condition_count_per_model": 400,
        "condition_count": 1200,
        "interfaces": list(INTERFACES),
        "histories": list(HISTORIES),
        "factor_notes": {
            "none": "current task only",
            "compatible": "prior same-task answer agrees with the registered target",
            "irrelevant": "prior unrelated exchange provides a length/content control",
            "conflict": "prior same-task answer is a registered distractor",
            "override": "conflicting prior plus an explicit current-turn override; composite, not a pure history level",
        },
        "prompt_contract": {
            "current_task_text_unchanged_for_pure_history_levels": True,
            "override_current_task_has_registered_override_prefix": True,
            "shared_terminal_literal": "Final answer:",
            "chat_uses_model_local_chat_template": True,
            "completion_uses_frozen_plain_narrative_serialization": True,
            "completion_avoids_chat_like_pseudo_role_markers": True,
            "record_full_prompt_hash_and_token_count": True,
            "require_exact_prompt_length_across_conditions": False,
            "require_shared_terminal_suffix_token_ids_within_model": True,
        },
        "execution_contract": {
            "native_model_generate": True,
            "passive_hooks_already_qualified_by_phase417": True,
            "max_new_tokens": 4,
            "collect_prompt_prefill_current_prediction_position": True,
            "ignore_cached_vectors_for_factor_map": True,
            "core_components": [
                "layer_input",
                "attention_output",
                "mlp_output",
                "residual_increment",
                "layer_output",
            ],
        },
        "registered_contrasts": {
            "history": "interface/history minus same-interface/none",
            "interface": "completion/history minus chat/history",
            "interaction": "(completion/history-completion/none)-(chat/history-chat/none)",
        },
        "analysis_contract": {
            "absolute_scale_audit": True,
            "relative_write_rate": True,
            "layer_local_median_mad_standardization": True,
            "exact_vector_delta_direction_consistency": True,
            "token_length_delta_strata": [0, 2, 8],
            "discovery_and_calibration_kept_separate": True,
            "cross_model_hidden_spaces_not_directly_aligned": True,
            "no_functional_name_from_high_norm_or_high_z_alone": True,
        },
        "behavior_gate": {
            "minimum_cases_per_model_family_interface_history": 10,
            "minimum_target_event_match_rate_for_functional_cell": 0.70,
            "failure_effect": "retain_physical_rows_but_withhold_functional_cell_interpretation",
        },
        "stop_rules": [
            "terminal_suffix_token_alignment_failure_blocks_model_execution",
            "nonfinite_or_component_ledger_failure_blocks_model_atlas",
            "length_or_position_dominance_blocks_interface_history_mechanism_claim",
            "late_depth_absolute_norm_dominance_is_scale_bias_not_mechanism",
            "descriptive_direction_consistency_does_not_authorize_causal_or_neuron_claims",
            "override_condition_is_never_interpreted_as_a_pure_history_main_effect",
        ],
        "claim_boundary": "paired_formal_world_interface_history_behavior_and_reduced_physical_difference_atlas_only",
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase418-RegisteredDenominatorQualification",
        "created_at": created_at,
        "valid": valid,
        "case_count": len(rows),
        "semantic_case_count": len(selected_ids),
        "family_count": len(families),
        "model_count": len(MODELS),
        "per_model_family_condition_count": {
            f"{model}:{family}": counts[(model, family)] for model in MODELS for family in families
        },
        "phase417_prerequisites": prerequisites,
        "model_execution_authorized": valid,
        "causal_intervention_authorized": False,
        "neuron_scan_authorized": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    return rows, protocol, summary


def main() -> None:
    rows, protocol, summary = freeze()
    write_jsonl(OUT / "phase418_registered_conditions.jsonl", rows)
    write_json(OUT / "phase418_protocol.json", protocol)
    write_json(OUT / "phase418_denominator_qualification.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
