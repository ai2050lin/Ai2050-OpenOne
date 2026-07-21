#!/usr/bin/env python3
"""Freeze Phase579 causal tests from Phase578 repeated natural option routes."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase577_natural_choice_protocol as choice  # noqa: E402
import phase578_choice_world_protocol as source  # noqa: E402
import phase578_natural_trace_protocol as trace_protocol  # noqa: E402


PHASE = "Phase579"
OUT_DIR = source.OUT_DIR
NATURAL_ANALYSIS_PATH = OUT_DIR / "phase578_natural_structure_analysis.json"
NATURAL_DECISION_PATH = OUT_DIR / "phase578_natural_structure_decision.json"
PROTOCOL_PATH = OUT_DIR / "phase579_option_routing_causal_protocol.json"
CONDITIONS = (
    "natural_baseline",
    "option_score_swap",
    "option_score_equalize",
    "object_relation_score_swap_control",
    "option_score_swap_restore",
    "option_weight_swap",
    "option_weight_swap_restore",
    "option_value_swap_positive_control",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def freeze() -> Path:
    natural_analysis = read_json(NATURAL_ANALYSIS_PATH)
    natural_decision = read_json(NATURAL_DECISION_PATH)
    trace = read_json(trace_protocol.TRACE_PROTOCOL_PATH)
    if natural_decision["analysis_sha256"] != sha256_file(NATURAL_ANALYSIS_PATH):
        raise RuntimeError("Phase578 natural analysis/decision hash drift")
    if natural_analysis["causal_holdout_internal_state_read"]:
        raise RuntimeError("Phase578 causal holdout was already read")
    if natural_analysis["sealed_split_read"]:
        raise RuntimeError("Phase578 sealed split was already read")

    selected_coordinates: dict[str, dict[str, Any]] = {}
    holdout_ids: dict[str, dict[str, list[str]]] = {}
    holdout_counts: dict[str, dict[str, dict[str, int]]] = {}
    behavior_artifacts: dict[str, Any] = {}
    for model in natural_decision["causal_protocol_authorized_models"]:
        model_decision = natural_decision["model_decisions"][model]
        candidates = model_decision["causal_candidate_coordinates"]
        by_relation = {item["relation"]: item for item in candidates}
        if set(by_relation) != set(choice.RELATIONS):
            raise RuntimeError(f"Phase579 missing relation coordinate for {model}")
        selected_coordinates[model] = by_relation

        registry_path = OUT_DIR / f"phase578_{model}_behavior_registry.json"
        registry = read_json(registry_path)
        selected = registry["causal_holdout_world_ids_by_split"]
        if canonical_hash(selected) != trace["causal_holdout_world_id_hash_by_model"][model]:
            raise RuntimeError(f"Phase579 causal holdout hash drift: {model}")
        if any(
            len(selected[split]) != source.CAUSAL_HOLDOUT_WORLDS_PER_SPLIT
            for split in source.OPEN_SPLITS
        ):
            raise RuntimeError(f"Phase579 causal holdout count drift: {model}")
        holdout_ids[model] = selected
        behavior_artifacts[model] = {
            "registry_sha256": sha256_file(registry_path),
            "natural_trace_summary_sha256": sha256_file(
                OUT_DIR / f"phase578_{model}_natural_trace_summary.json"
            ),
        }

    selected_sets = {
        model: set().union(*(set(ids) for ids in splits.values()))
        for model, splits in holdout_ids.items()
    }
    case_lookup: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(source.SOURCE_CASES_PATH):
        case_lookup.setdefault(row["world_id"], row)
    for model, splits in holdout_ids.items():
        holdout_counts[model] = {}
        for split, ids in splits.items():
            if not set(ids).issubset(selected_sets[model]):
                raise RuntimeError("Phase579 holdout selection drift")
            relations = Counter(case_lookup[world_id]["relation"] for world_id in ids)
            objects = Counter(
                "fruit" if case_lookup[world_id]["is_fruit"] else "control"
                for world_id in ids
            )
            holdout_counts[model][split] = {
                **{f"relation_{key}": value for key, value in relations.items()},
                **{f"object_{key}": value for key, value in objects.items()},
                "world_count": len(ids),
            }

    payload = {
        "schema_version": "phase579_option_routing_causal_protocol.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "authorized_models": sorted(selected_coordinates),
        "selected_coordinates_by_model_and_relation": selected_coordinates,
        "causal_holdout_world_ids_by_model_and_split": holdout_ids,
        "causal_holdout_counts": holdout_counts,
        "variants": ["target_first", "target_second"],
        "conditions": list(CONDITIONS),
        "execution": {
            "torch_dtype": "torch.bfloat16",
            "device": "cuda",
            "world_batch_size": 8,
            "right_padding_and_explicit_position_ids": True,
            "patch_receiver": "answer_boundary",
            "patch_all_heads": True,
            "patch_only_selected_layer_for_world_relation": True,
            "score_swap_preserves_within_group_deviations": True,
            "weight_swap_preserves_total_probability_mass": True,
            "restore_directly_returns_natural_tensor": True,
        },
        "discovery_gate": {
            "minimum_relation_world_count": 24,
            "option_score_swap_margin_effect_negative_rate": 0.65,
            "option_score_swap_margin_effect_mean_maximum": -0.02,
            "score_swap_vs_nonoption_control_gap_minimum": 0.01,
            "option_route_effect_negative_rate": 0.90,
            "restore_candidate_margin_maximum_absolute_delta": 1e-4,
            "restore_route_margin_maximum_absolute_delta": 1e-6,
            "both_option_orders_must_pass": True,
        },
        "confirmation_rule": {
            "open_confirmation_internal_state_can_be_read_only_after_discovery_pass": True,
            "same_selected_coordinate_operator_and_thresholds": True,
            "model_relation_branches_are_confirmed_independently": True,
            "full_generation_can_run_only_for_confirmed_branches": True,
            "sealed_can_open_only_after_full_generation_pass": True,
        },
        "claim_boundary": {
            "candidate_route_not_knowledge_storage": True,
            "candidate_route_not_upstream_object_attribute_binding": True,
            "no_cross_model_coordinate_identity_claim": True,
            "no_head_channel_parameter_neuron_scan": True,
        },
        "causal_discovery_internal_state_read": False,
        "causal_confirmation_internal_state_read": False,
        "sealed_split_read": False,
        "natural_analysis_sha256": sha256_file(NATURAL_ANALYSIS_PATH),
        "natural_decision_sha256": sha256_file(NATURAL_DECISION_PATH),
        "trace_protocol_sha256": sha256_file(trace_protocol.TRACE_PROTOCOL_PATH),
        "source_cases_sha256": sha256_file(source.SOURCE_CASES_PATH),
        "behavior_artifacts": behavior_artifacts,
    }
    if PROTOCOL_PATH.exists():
        existing = read_json(PROTOCOL_PATH)
        ignored = {"created_at"}
        left = {key: value for key, value in existing.items() if key not in ignored}
        right = {key: value for key, value in payload.items() if key not in ignored}
        if left != right:
            raise RuntimeError("Phase579 frozen causal protocol drift")
    else:
        write_json(PROTOCOL_PATH, payload)
    print(
        json.dumps(
            {
                "authorized_models": payload["authorized_models"],
                "selected_coordinates": selected_coordinates,
                "causal_holdout_counts": holdout_counts,
                "condition_count": len(CONDITIONS),
                "causal_discovery_internal_state_read": False,
                "causal_confirmation_internal_state_read": False,
                "sealed_split_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return PROTOCOL_PATH


if __name__ == "__main__":
    freeze()
