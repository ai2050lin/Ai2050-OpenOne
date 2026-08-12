#!/usr/bin/env python3
"""Freeze the one-shot Phase1129 DS7B attention numerical refinement."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import transformers
from transformers.models.qwen2 import modeling_qwen2


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1128_fp16_numeric_formation_protocol as source_protocol


PHASE = 1129
MODEL = "deepseek7b"
TARGET_LAYER = 27
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1129_ds7b_attention_numeric_refinement"
SOURCE_ROOT = source_protocol.OUT_ROOT

EXPECTED_SOURCE_PROTOCOL_DIGEST = "2f6bbef914cd5880b7806abae6208b603cb4e60935aef882b5716627fc94865a"
EXPECTED_SOURCE_FINAL_DIGEST = "3dfbf3391f062e52ed70aef0d477de191ed9351b81a63d209d0732bb7f9d6c52"
EXPECTED_SOURCE_AUDIT_DIGEST = "3765156f8b200cef873f998e555e094be3d05a645b5544a448c309d138bc7364"
EXPECTED_SOURCE_SCAN_DIGEST = "7dacb5be1ece169fdd4f90de7f5adf37ad8a1ae4a8cb42904957663ec4e3c8e7"
EXPECTED_TRANSFORMERS_VERSION = "5.14.1"
EXPECTED_ATTENTION_SOURCE_DIGEST = "3d7ae1485294d0c8d57827617a3abed805bcf0c4107ead3d1a47f67eee7dcb1b"
EXPECTED_EAGER_SOURCE_DIGEST = "cb34ccea28710ae8d5d94a241b704ea3d974048f85372f0b1b3c2a8a2ef1ee20"

EVENT_REGISTRY = (
    {"order": 0, "name": "attention_norm", "event_class": "attention_input"},
    {"order": 1, "name": "q_proj_queries", "event_class": "q_projection"},
    {"order": 2, "name": "k_proj_prefix", "event_class": "k_projection"},
    {"order": 3, "name": "v_proj_prefix", "event_class": "v_projection"},
    {"order": 4, "name": "pre_softmax_scores", "event_class": "qk_score"},
    {"order": 5, "name": "softmax_output_fp32", "event_class": "softmax"},
    {"order": 6, "name": "attention_weights_fp16", "event_class": "attention_weight_cast"},
    {"order": 7, "name": "o_proj_input", "event_class": "value_aggregation"},
    {"order": 8, "name": "o_proj_output", "event_class": "output_projection"},
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def source_digest(value: Any) -> str:
    return source_protocol.digest(value)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) for row in rows)
    path.write_text(payload + "\n", encoding="utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    source_prereg = read_json(SOURCE_ROOT / "protocol" / "preregistration.json")
    source_final = read_json(SOURCE_ROOT / "analysis" / "final_summary.json")
    source_audit = read_json(SOURCE_ROOT / "audit" / "result_audit.json")
    source_scan = read_json(SOURCE_ROOT / "scan" / MODEL / "summary.json")
    source_cases = read_jsonl(SOURCE_ROOT / "protocol" / f"cases.{MODEL}.jsonl")
    source_case_results = read_jsonl(SOURCE_ROOT / "scan" / MODEL / "cases.jsonl")

    attention_digest = sha256_text(inspect.getsource(modeling_qwen2.Qwen2Attention.forward))
    eager_digest = sha256_text(inspect.getsource(modeling_qwen2.eager_attention_forward))
    prereg_core = {
        "schema_version": "phase1129_ds7b_attention_numeric_refinement_preregistration.v1",
        "phase": PHASE,
        "objective": (
            "Perform the single numerical subcomponent refinement authorized by Phase1128: determine whether "
            "DS7B layer-27 FP16 non-finites first appear in Q/K/V projection, QK scores, softmax, A-times-V, or O projection."
        ),
        "authorization": {
            "phase1128_protocol_digest": source_prereg["protocol_digest"],
            "phase1128_final_digest": source_final["final_digest"],
            "phase1128_audit_digest": source_audit["audit_digest"],
            "phase1128_scan_summary_digest": source_scan["summary_digest"],
            "authorized_model": MODEL,
            "authorized_scope": source_final["automatic_refinement"]["authorized_scope"],
            "source_exact_event": source_final["model_results"][MODEL]["dominant_first_event"],
            "source_exact_event_fraction": source_final["model_results"][MODEL]["dominant_first_event_fraction"],
        },
        "model": MODEL,
        "target_layer": TARGET_LAYER,
        "precision": "fp16",
        "quantization": "none",
        "case_policy": {
            "all_frozen_phase1128_cases": True,
            "case_count": 320,
            "source_nonfinite_count": 314,
            "source_finite_count": 6,
            "hidden_holdout_scored": False,
            "batch_size": 4,
            "case_digest": source_prereg["source"]["links"][MODEL]["case_digest"],
            "phase1128_case_result_digest": source_scan["case_detail_digest"],
        },
        "implementation": {
            "transformers_version": transformers.__version__,
            "qwen2_attention_forward_sha256": attention_digest,
            "qwen2_eager_attention_forward_sha256": eager_digest,
            "attention_implementation": "eager",
            "softmax_instrument": (
                "temporarily wrap torch.nn.functional.softmax only while target-layer self-attention is active; "
                "delegate computation unchanged to the original function"
            ),
        },
        "event_registry": list(EVENT_REGISTRY),
        "measurement": {
            "query_scope": "candidate and suffix prediction positions frozen in Phase1126",
            "key_value_scope": "causally available prefix through the latest scored prediction position",
            "pre_softmax_overflow_rule": "NaN or positive infinity, or a row with no finite score; negative-infinity mask entries alone are not failures",
            "all_other_event_failure_rule": "any NaN or positive/negative infinity",
            "raw_tensors_saved": False,
            "semantic_similarity_computed": False,
        },
        "root_precedence": [event["event_class"] for event in EVENT_REGISTRY[1:]],
        "predictions": {
            "P1": "all Phase1128 authorization, implementation-source, case, precision, and event checks pass",
            "P2": "the instrumented replay exactly preserves all Phase1128 candidate/suffix/total finite flags",
            "P3": "all 314 source-nonfinite cases receive one earliest numerical root classification",
            "P4": "root counts are descriptive; QK-score overflow is plausible but not assumed",
            "P5": "the result closes this numerical refinement regardless of sign",
        },
        "forbidden": [
            "no model other than the Phase1128-authorized DS7B endpoint",
            "no hidden holdout, case deletion, retry, threshold change, BF16, FP32, or quantization",
            "no raw Q/K/V, score, probability, value, residual, or logit tensor persistence",
            "no semantic, relation, content, hotspot, causal, intervention, or precision-superiority claim",
            "no further automatic numerical localization after this one-shot refinement",
        ],
        "auto_continue": False,
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)

    source_cases_by_index = {int(row["case_index"]): row for row in source_case_results}
    checks = {
        "source_protocol_digest": source_prereg["protocol_digest"] == EXPECTED_SOURCE_PROTOCOL_DIGEST,
        "source_final_digest": source_final["final_digest"] == EXPECTED_SOURCE_FINAL_DIGEST,
        "source_audit_digest": source_audit["audit_digest"] == EXPECTED_SOURCE_AUDIT_DIGEST,
        "source_scan_digest": source_scan["summary_digest"] == EXPECTED_SOURCE_SCAN_DIGEST,
        "source_audit_passed": source_audit["passed"] is True,
        "source_auto_authorized_ds_only": source_final["automatic_refinement"]["value"] is True
        and source_final["automatic_refinement"]["authorized_models"] == [MODEL],
        "source_exact_event": source_final["model_results"][MODEL]["dominant_first_event"]
        == f"layer_{TARGET_LAYER}.attention_output",
        "source_exact_event_fraction": source_final["model_results"][MODEL]["dominant_first_event_fraction"] == 1.0,
        "source_case_count": len(source_cases) == 320 and len(source_case_results) == 320,
        "source_case_digest": source_protocol.source_protocol.digest(source_cases)
        == source_prereg["source"]["links"][MODEL]["case_digest"],
        "source_result_digest": source_protocol.digest(source_case_results) == source_scan["case_detail_digest"],
        "source_finite_split": sum(source_cases_by_index[index]["source_total_finite"] for index in range(320)) == 6,
        "transformers_version": transformers.__version__ == EXPECTED_TRANSFORMERS_VERSION,
        "attention_source_digest": attention_digest == EXPECTED_ATTENTION_SOURCE_DIGEST,
        "eager_source_digest": eager_digest == EXPECTED_EAGER_SOURCE_DIGEST,
        "event_orders_unique": len({event["order"] for event in EVENT_REGISTRY}) == len(EVENT_REGISTRY),
        "auto_continue_false": prereg["auto_continue"] is False,
    }
    audit_core = {
        "schema_version": "phase1129_ds7b_attention_numeric_refinement_protocol_audit.v1",
        "phase": PHASE,
        "checks": checks,
        "passed_count": sum(checks.values()),
        "total_count": len(checks),
        "passed": all(checks.values()),
        "protocol_digest": prereg["protocol_digest"],
    }
    audit = dict(audit_core)
    audit["audit_digest"] = digest(audit_core)
    write_jsonl(OUT_ROOT / "protocol" / "cases.deepseek7b.jsonl", source_cases)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", prereg)
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["passed"]:
        raise RuntimeError("Phase1129 protocol audit failed")


if __name__ == "__main__":
    main()
