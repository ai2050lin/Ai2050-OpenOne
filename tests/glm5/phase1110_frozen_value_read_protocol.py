#!/usr/bin/env python3
"""Freeze Phase1110 key/body value-read decomposition at Phase1109 heads."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1098_relative_relation_geometry_protocol as tools
import phase1108_exact_key_event_protocol as source
import phase1109_attention_routing_protocol as phase1109


PHASE = 1110
PROTOCOL_REVISION = 1
MODELS = source.MODELS
AUTHORIZED_MODELS = phase1109.AUTHORIZED_MODELS
DENIED_MODELS = phase1109.DENIED_MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
SOURCE_NAMES = ("key0", "body0", "key1", "body1", "outside")
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1110_frozen_value_read"
SOURCE_ROOT = phase1109.OUT_ROOT
SOURCE_PREREG = SOURCE_ROOT / "protocol" / "preregistration.json"
SOURCE_AUDIT = SOURCE_ROOT / "audit" / "result_audit.json"
SOURCE_DECISIONS = SOURCE_ROOT / "analysis" / "model_decisions.json"
SOURCE_FINAL = SOURCE_ROOT / "analysis" / "final_summary.json"


write_json = tools.write_json
write_jsonl = tools.write_jsonl
read_json = tools.read_json
read_jsonl = tools.read_jsonl
digest = tools.digest


THRESHOLDS = {
    "minimum_value_finite_fraction": 0.999,
    "maximum_head_reconstruction_relative_error": 0.005,
    "maximum_key_value_matched_distance": 1e-5,
    "minimum_body_value_matched_distance": 0.02,
    "minimum_body_over_key_distance_advantage": 0.02,
    "minimum_selected_body_av_distance_advantage": 0.03,
    "minimum_exact_over_ordinal_selection_advantage": 0.03,
    "minimum_body_over_key_readout_alignment": 0.01,
    "minimum_positive_relation_pairs": 3,
    "minimum_models": 2,
}


PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "The Phase1109 result audit, frozen selected heads, confirmation cases, "
        "source partitions, and key-before-payload causal-order checks all pass."
    ),
    "P2": (
        "Qwen3 and GLM4 run in FP16 without quantization; DS7B remains denied "
        "because Phase1108 did not authorize hidden access for it."
    ),
    "P3": (
        "For every frozen event, the sum of disjoint A-times-V source "
        "contributions reconstructs the captured pre-o_proj head vector within tolerance."
    ),
    "P4": (
        "Across matched conflict/congruent states, the changed relation's key V "
        "is causally invariant while its post-key body V changes in two models, "
        "both key regimes, and at least three relation pairs."
    ),
    "P5": (
        "A-times-V change from the altered body is larger when that relation is "
        "selected than when it is the distractor, in both models and key regimes."
    ),
    "P6": (
        "The selected-minus-distractor body effect is larger for exact-key than "
        "ordinal routing in both models and key regimes."
    ),
    "P7": (
        "The target-body contribution has greater direct candidate-readout "
        "alignment than the causally blind target-key contribution."
    ),
    "P8": (
        "This phase is descriptive. Phase1109 P6/P7 failures remain binding, so "
        "no head, Q/K/V, neuron, or causal intervention is automatically authorized."
    ),
}


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inclusive(start_end: list[int]) -> list[int]:
    start, end = (int(value) for value in start_end)
    return list(range(start, end + 1))


def source_positions(row: dict[str, Any]) -> dict[str, list[int]]:
    records = {
        index: set(inclusive(row["record_spans"][f"record{index}"]))
        for index in (0, 1)
    }
    keys = {
        index: set(inclusive(row["key_spans"][f"key{index}"]))
        for index in (0, 1)
    }
    bodies = {index: records[index] - keys[index] for index in (0, 1)}
    occupied = records[0] | records[1]
    outside = set(range(len(row["input_ids"]))) - occupied
    return {
        "key0": sorted(keys[0]),
        "body0": sorted(bodies[0]),
        "key1": sorted(keys[1]),
        "body1": sorted(bodies[1]),
        "outside": sorted(outside),
    }


def augment_case(
    row: dict[str, Any], source_row: dict[str, Any], case_index: int,
) -> dict[str, Any]:
    positions = source_positions(row)
    return {
        "schema_version": "phase1110_frozen_value_read_case.v1",
        "phase": PHASE,
        "model": row["model"],
        "case_index": case_index,
        "record_id": row["record_id"].replace("phase1109", "phase1110", 1),
        "source_record_id": row["record_id"],
        "unit_id": row["unit_id"].replace("phase1109", "phase1110", 1),
        "relation_pair": row["relation_pair"],
        "surface": row["surface"],
        "split": row["split"],
        "template": int(row["template"]),
        "item_index": int(row["item_index"]),
        "state": row["state"],
        "label_regime": row["label_regime"],
        "route_type": row["route_type"],
        "congruence": row["congruence"],
        "target_relation": int(row["target_relation"]),
        "relation_order": int(row["relation_order"]),
        "orientation": int(row["orientation"]),
        "expected_class": row["expected_class"],
        "candidate_first_token_ids": row["candidate_first_token_ids"],
        "input_ids": [int(value) for value in row["input_ids"]],
        "query_position": int(row["query_positions"]["selector_end"]),
        "source_positions": positions,
        "entity0": source_row["entity0"],
        "entity1": source_row["entity1"],
        "displayed_relations": source_row["displayed_relations"],
        "source_prompt_digest": row["source_prompt_digest"],
    }


def audit_cases(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_unit.setdefault(row["unit_id"], []).append(row)
    partition_checks = []
    key_before_payload = []
    for row in rows:
        groups = row["source_positions"]
        sets = [set(groups[name]) for name in SOURCE_NAMES]
        partition_checks.append(
            all(sets[i].isdisjoint(sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets)))
            and set().union(*sets) == set(range(len(row["input_ids"])))
            and all(groups[name] for name in SOURCE_NAMES)
        )
        candidate_ids = {
            int(value[0]) for value in row["candidate_first_token_ids"].values()
        }
        for index in (0, 1):
            body = groups[f"body{index}"]
            key = groups[f"key{index}"]
            entity_positions = [
                position for position in body
                if int(row["input_ids"][position]) in candidate_ids
            ]
            key_before_payload.append(bool(entity_positions) and max(key) < min(entity_positions))
    checks = {
        "confirmation_case_count": len(rows) == 3072,
        "confirmation_unit_count": len(by_unit) == 48,
        "units_have_64_states": all(len(values) == 64 for values in by_unit.values()),
        "confirmation_only": all(row["split"] == "confirmation" for row in rows),
        "record_ids_unique": len({row["record_id"] for row in rows}) == len(rows),
        "state_cubes_exact": all({row["state"] for row in values} == set(source.STATES) for values in by_unit.values()),
        "source_partition_disjoint_exhaustive": all(partition_checks),
        "keys_precede_entity_payload": all(key_before_payload),
        "query_after_records": all(
            row["query_position"] > max(row["source_positions"]["body0"] + row["source_positions"]["body1"])
            for row in rows
        ),
    }
    return {
        "model": model,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "case_digest": digest(rows),
    }


def main() -> None:
    source_prereg = read_json(SOURCE_PREREG)
    source_audit = read_json(SOURCE_AUDIT)
    source_decisions = read_json(SOURCE_DECISIONS)
    source_final = read_json(SOURCE_FINAL)
    source_checks = {
        "source_result_audit_passed": bool(source_audit["all_checks_passed"]),
        "source_phase_exact": int(source_final["phase"]) == 1109,
        "source_protocol_digest_matches": source_final["protocol_digest"] == source_prereg["protocol_digest"],
        "source_p4_p5_passed": bool(source_final["prospective_predictions"]["P4"]) and bool(source_final["prospective_predictions"]["P5"]),
        "source_p6_p7_failed": not bool(source_final["prospective_predictions"]["P6"]) and not bool(source_final["prospective_predictions"]["P7"]),
        "source_p8_failed": not bool(source_final["prospective_predictions"]["P8"]),
    }
    if not all(source_checks.values()):
        raise RuntimeError(f"Phase1109 source checks failed: {source_checks}")

    protocol_root = OUT_ROOT / "protocol"
    protocol_root.mkdir(parents=True, exist_ok=True)
    selected_events = {}
    case_digests = {}
    model_audits = {}
    for model in MODELS:
        source_rows = list(read_jsonl(SOURCE_ROOT / "protocol" / f"cases.{model}.jsonl"))
        phase1108_rows = {
            row["record_id"]: row
            for row in read_jsonl(source.OUT_ROOT / "protocol" / f"cases.{model}.jsonl")
        }
        confirmation = [row for row in source_rows if row["split"] == "confirmation"]
        rows = [
            augment_case(row, phase1108_rows[row["source_record_id"]], index)
            for index, row in enumerate(confirmation)
        ]
        audit = audit_cases(rows, model)
        if not audit["all_checks_passed"]:
            raise RuntimeError(f"{model} case audit failed: {audit['checks']}")
        write_jsonl(protocol_root / f"cases.{model}.jsonl", rows)
        case_digests[model] = audit["case_digest"]
        model_audits[model] = audit
        if model in AUTHORIZED_MODELS:
            events = source_decisions[model]["selection"]["selected_events"]
            if len(events) != 4 or any(event["query_role"] != "selector_end" for event in events):
                raise RuntimeError(f"{model} frozen event set drift")
            selected_events[model] = events

    prereg = {
        "schema_version": "phase1110_frozen_value_read_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "authorized_models": list(AUTHORIZED_MODELS),
        "denied_models": list(DENIED_MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "source_names": list(SOURCE_NAMES),
        "relation_pairs": list(source.RELATION_PAIRS),
        "states": list(source.STATES),
        "selected_events": selected_events,
        "thresholds": THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "source": {
            "phase": 1109,
            "protocol_digest": source_prereg["protocol_digest"],
            "final_summary_digest": source_final["final_summary_digest"],
            "source_checks": source_checks,
            "file_hashes": {
                "preregistration": file_sha256(SOURCE_PREREG),
                "result_audit": file_sha256(SOURCE_AUDIT),
                "model_decisions": file_sha256(SOURCE_DECISIONS),
                "final_summary": file_sha256(SOURCE_FINAL),
            },
        },
        "case_digests": case_digests,
        "interpretive_limits": [
            "Selected heads are frozen from Phase1109 qualification; no reselection is permitted.",
            "A-times-V content is a descriptive read-path observable, not causal use.",
            "The exact-key registry is artificial and does not establish semantic addressing.",
            "Direct unembedding alignment at an early layer is a diagnostic, not the model's final logit contribution.",
            "Phase1109 P6/P7 failures remain binding regardless of Phase1110 outcomes.",
        ],
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    audit = {
        "schema_version": "phase1110_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "source_checks": source_checks,
        "model_audits": model_audits,
        "all_checks_passed": all(source_checks.values()) and all(
            value["all_checks_passed"] for value in model_audits.values()
        ),
    }
    audit["audit_digest"] = digest(audit)
    write_json(protocol_root / "audit.json", audit)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "case_count_per_model": 3072,
        "selected_event_counts": {key: len(value) for key, value in selected_events.items()},
        "all_checks_passed": audit["all_checks_passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
