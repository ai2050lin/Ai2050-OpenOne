#!/usr/bin/env python3
"""Freeze the Phase1109 attention-routing observable map."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1018_language_pattern_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1098_relative_relation_geometry_protocol as tools
import phase1108_exact_key_event_protocol as source


PHASE = 1109
PROTOCOL_REVISION = 1
MODELS = source.MODELS
AUTHORIZED_MODELS = ("qwen3", "glm4")
DENIED_MODELS = ("deepseek7b",)
PRECISION = "fp16"
QUANTIZATION = "none"
QUERY_ROLES = (
    "pre_selector",
    "selector_end",
    "query_end",
    "answer_boundary",
)
LABEL_REGIMES = source.LABEL_REGIMES
ROUTE_TYPES = source.ROUTE_TYPES
CONGRUENCES = source.CONGRUENCES
MAX_SELECTED_EVENTS = 4
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1109_attention_routing_map"
SOURCE_ROOT = source.OUT_ROOT
SOURCE_PREREG = SOURCE_ROOT / "protocol" / "preregistration.json"
SOURCE_AUDIT = SOURCE_ROOT / "audit" / "result_audit.json"
SOURCE_AUTHORIZATION = SOURCE_ROOT / "analysis" / "behavior_authorization.json"
SOURCE_FINAL = SOURCE_ROOT / "analysis" / "final_summary.json"


write_json = tools.write_json
write_jsonl = tools.write_jsonl
read_json = tools.read_json
read_jsonl = tools.read_jsonl
digest = tools.digest


THRESHOLDS = {
    "minimum_attention_finite_fraction": 0.999,
    "maximum_deterministic_identity_error": 1e-8,
    "maximum_pre_selector_identity_error": 1e-7,
    "minimum_total_key_attention_mass": 0.005,
    "minimum_exact_key_following": 0.15,
    "minimum_lexical_over_ordinal_advantage": 0.05,
    "minimum_positive_relation_pairs": 3,
    "minimum_execution_modulation": 0.03,
    "minimum_cross_model_curve_cosine": 0.75,
    "maximum_cross_model_curve_mae": 0.20,
    "minimum_models": 2,
}


PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "The inherited Phase1108 source, behavior authorization, token spans, "
        "causal ordering, and protocol digests pass before hidden access."
    ),
    "P2": (
        "Qwen3 and GLM4 remain the only behavior-authorized models; DS7B is "
        "recorded as denied and receives no hidden-state access."
    ),
    "P3": (
        "Both authorized FP16/no-quantization scans return finite eager "
        "attention matrices with deterministic identity and pre-selector-zero audits."
    ),
    "P4": (
        "A qualification-selected head-role ensemble repeats on confirmation "
        "with exact-key following and lexical-over-ordinal advantage in both "
        "relation and neutral key regimes for two models."
    ),
    "P5": (
        "The confirmed ensemble is positive for at least three of four frozen "
        "relation pairs in both key regimes and both models."
    ),
    "P6": (
        "Conflict-minus-congruent execution modulation reaches the frozen "
        "threshold in both regimes and both models; generic token matching alone fails this gate."
    ),
    "P7": (
        "The normalized attention-routing depth profile repeats across Qwen3 "
        "and GLM4 at the frozen cosine and MAE thresholds."
    ),
    "P8": (
        "Only P2-P7 together may authorize a separately preregistered causal "
        "study. A head hotspot or P4/P5 without P6 remains descriptive."
    ),
}


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _marked_key_spans(row: dict[str, Any]) -> dict[str, tuple[int, int, str]]:
    raw = str(row["raw_prompt"])
    relation0, relation1 = source.PAIR_RELATIONS[row["relation_pair"]]
    displayed = list(row["displayed_relations"])
    facts = (str(row["fact1_text"]), str(row["fact2_text"]))
    result: dict[str, tuple[int, int, str]] = {}
    for relation_index, relation in enumerate((relation0, relation1)):
        display_index = displayed.index(relation)
        fact_text = facts[display_index]
        fact_start = raw.find(fact_text)
        if fact_start < 0:
            raise RuntimeError(f"missing fact text for {row['record_id']}")
        label = str(row["relation_labels"][relation])
        local = fact_text.find(label)
        if local < 0:
            raise RuntimeError(f"missing key label for {row['record_id']}")
        start = fact_start + local
        result[f"key{relation_index}"] = (start, start + len(label), label)
    return result


def augment_case(tokenizer, row: dict[str, Any]) -> dict[str, Any]:
    key_spans = offset_token_spans(
        tokenizer,
        str(row["rendered_prompt"]),
        str(row["raw_prompt"]),
        _marked_key_spans(row),
    )
    displayed = list(row["displayed_relations"])
    relation0, relation1 = source.PAIR_RELATIONS[row["relation_pair"]]
    fact_spans = (
        tuple(int(value) for value in row["role_spans"]["fact1_end"]),
        tuple(int(value) for value in row["role_spans"]["facts_end"]),
    )
    record_spans = {
        "record0": fact_spans[displayed.index(relation0)],
        "record1": fact_spans[displayed.index(relation1)],
    }
    selector_start = int(row["role_positions"]["selector_start"])
    query_positions = {
        "pre_selector": selector_start - 1,
        "selector_end": int(row["role_positions"]["selector_end"]),
        "query_end": int(row["role_positions"]["query_end"]),
        "answer_boundary": int(row["role_positions"]["answer_boundary"]),
    }
    return {
        "schema_version": "phase1109_attention_routing_case.v1",
        "phase": PHASE,
        "model": row["model"],
        "record_id": row["record_id"].replace("phase1108", "phase1109", 1),
        "source_record_id": row["record_id"],
        "unit_id": row["unit_id"].replace("phase1108", "phase1109", 1),
        "relation_pair": row["relation_pair"],
        "family": row["family"],
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
        "key_spans": {key: list(value) for key, value in key_spans.items()},
        "record_spans": {key: list(value) for key, value in record_spans.items()},
        "query_positions": query_positions,
        "source_prompt_digest": row["prompt_digest"],
    }


def audit_cases(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    source_rows = list(read_jsonl(SOURCE_ROOT / "protocol" / f"cases.{model}.jsonl"))
    expected_count = len(source_rows)
    checks = {
        "case_count_matches_source": len(rows) == expected_count == 6144,
        "record_ids_unique": len({row["record_id"] for row in rows}) == len(rows),
        "units_have_64_states": all(
            sum(1 for item in rows if item["unit_id"] == unit_id) == 64
            for unit_id in {row["unit_id"] for row in rows}
        ),
        "query_roles_exact": all(
            set(row["query_positions"]) == set(QUERY_ROLES) for row in rows
        ),
        "query_positions_ordered": all(
            0 <= row["query_positions"]["pre_selector"]
            < row["query_positions"]["selector_end"]
            <= row["query_positions"]["query_end"]
            < row["query_positions"]["answer_boundary"]
            < len(row["input_ids"])
            for row in rows
        ),
        "key_spans_in_records": all(
            row["record_spans"][f"record{index}"][0]
            <= row["key_spans"][f"key{index}"][0]
            <= row["key_spans"][f"key{index}"][1]
            <= row["record_spans"][f"record{index}"][1]
            for row in rows for index in (0, 1)
        ),
        "source_prompts_preserved": all(
            row["source_prompt_digest"] == source_row["prompt_digest"]
            and row["input_ids"] == source_row["input_ids"]
            for row, source_row in zip(rows, source_rows)
        ),
        "factor_balance": all(
            sum(1 for row in rows if row[field] == value) * 2 == len(rows)
            for field, values in (
                ("label_regime", LABEL_REGIMES),
                ("route_type", ROUTE_TYPES),
                ("congruence", CONGRUENCES),
                ("target_relation", (0, 1)),
                ("relation_order", (0, 1)),
                ("orientation", (0, 1)),
            )
            for value in values
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
    source_authorization = read_json(SOURCE_AUTHORIZATION)
    source_final = read_json(SOURCE_FINAL)
    source_checks = {
        "source_protocol_audit_passed": bool(source_audit["all_checks_passed"]),
        "source_result_audit_passed": bool(source_audit["all_checks_passed"]),
        "source_phase_exact": int(source_final["phase"]) == 1108,
        "source_authorized_models_exact": tuple(source_authorization["authorized_models"])
        == AUTHORIZED_MODELS,
        "source_denied_models_exact": tuple(
            model for model in MODELS
            if model not in source_authorization["authorized_models"]
        ) == DENIED_MODELS,
        "source_authorization_digest_matches": (
            source_authorization["authorization_digest"]
            == source_final["behavior_authorization_digest"]
        ),
    }
    if not all(source_checks.values()):
        raise RuntimeError(f"Phase1108 source audit failed: {source_checks}")

    protocol_root = OUT_ROOT / "protocol"
    protocol_root.mkdir(parents=True, exist_ok=True)
    model_audits = {}
    case_digests = {}
    for model in MODELS:
        tokenizer = tokenizer_for(model)
        source_rows = list(read_jsonl(SOURCE_ROOT / "protocol" / f"cases.{model}.jsonl"))
        rows = [augment_case(tokenizer, row) for row in source_rows]
        audit = audit_cases(rows, model)
        if not audit["all_checks_passed"]:
            raise RuntimeError(f"{model} Phase1109 case audit failed: {audit['checks']}")
        write_jsonl(protocol_root / f"cases.{model}.jsonl", rows)
        model_audits[model] = audit
        case_digests[model] = audit["case_digest"]

    prereg = {
        "schema_version": "phase1109_attention_routing_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "authorized_models": list(AUTHORIZED_MODELS),
        "denied_models": list(DENIED_MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "query_roles": list(QUERY_ROLES),
        "label_regimes": list(LABEL_REGIMES),
        "route_types": list(ROUTE_TYPES),
        "congruences": list(CONGRUENCES),
        "relation_pairs": list(source.RELATION_PAIRS),
        "max_selected_events": MAX_SELECTED_EVENTS,
        "thresholds": THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "source": {
            "phase": 1108,
            "protocol_digest": source_prereg["protocol_digest"],
            "behavior_authorization_digest": source_authorization["authorization_digest"],
            "final_digest": source_final["final_summary_digest"],
            "source_checks": source_checks,
            "source_file_hashes": {
                "preregistration": file_sha256(SOURCE_PREREG),
                "result_audit": file_sha256(SOURCE_AUDIT),
                "behavior_authorization": file_sha256(SOURCE_AUTHORIZATION),
                "final_summary": file_sha256(SOURCE_FINAL),
            },
        },
        "case_digests": case_digests,
        "interpretive_limits": [
            "Attention weight is a routing observable, not proof of information use.",
            "Exact token matching is not abstract relation semantics.",
            "Qualification events cannot be replaced after confirmation.",
            "No head, QKV, neuron, or causal claim is authorized by a descriptive map.",
        ],
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    audit = {
        "schema_version": "phase1109_protocol_audit.v1",
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
        "case_count_per_model": 6144,
        "authorized_models": list(AUTHORIZED_MODELS),
        "denied_models": list(DENIED_MODELS),
        "all_checks_passed": audit["all_checks_passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
