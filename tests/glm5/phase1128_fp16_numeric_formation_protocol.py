#!/usr/bin/env python3
"""Freeze the Phase1128 FP16 numerical-formation localization protocol."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import MODEL_CONFIGS
import phase1126_semeval_lexsub_natural_cloze_protocol as source_protocol


PHASE = 1128
MODELS = ("qwen3", "glm4", "deepseek7b")
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1128_fp16_numeric_formation"
SOURCE_ROOT = source_protocol.OUT_ROOT

EXPECTED_SOURCE_PROTOCOL_DIGEST = "f76079467e1c829fbc2a875a52b37e0b26a32fc3d49e0dfb1ee14a042ec9ba4d"
EXPECTED_SOURCE_FINAL_DIGEST = "093a69a78fa77a8cd2674a6de51d049042c2e8e6c7c224105747b582d4049f25"
EXPECTED_SOURCE_AUDIT_DIGEST = "f4fc42968ce0bd731a411361ed59f0bcf3172994dfbd8168bdd8ec4c4a52223b"
EXPECTED_SOURCE_DETAIL_DIGESTS = {
    "qwen3": "fc07a534bdafa6579c294d1a793b35f673121f665e48f1574cb5813dc638e22e",
    "glm4": "e48131e232e4fec789e991bb3e2aca97e785c76d67fcb055501f21680f0f20e4",
    "deepseek7b": "de712cce8b99a2f8524d1e14696db65fed26380eaab60cf24ba6be11b6d53af1",
}
EXPECTED_LAYERS = {"qwen3": 36, "glm4": 40, "deepseek7b": 28}
EXPECTED_HIDDEN = {"qwen3": 2560, "glm4": 4096, "deepseek7b": 3584}
EXPECTED_VOCAB = {"qwen3": 151936, "glm4": 151552, "deepseek7b": 152064}
EXPECTED_FINITE_COUNTS = {"qwen3": 320, "glm4": 315, "deepseek7b": 6}

EVENTS_PER_LAYER = (
    "layer_input",
    "attention_norm",
    "attention_output",
    "mlp_norm",
    "mlp_output",
    "layer_output",
)
REFINABLE_EVENT_CLASSES = (
    "attention_norm",
    "attention_output",
    "mlp_norm",
    "mlp_output",
    "layer_output",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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


def event_registry(layer_count: int) -> list[dict[str, Any]]:
    result = [{"order": 0, "name": "embedding", "event_class": "embedding", "layer": None}]
    order = 1
    suffixes = (
        ("input", "layer_input"),
        ("attention_norm", "attention_norm"),
        ("attention_output", "attention_output"),
        ("mlp_norm", "mlp_norm"),
        ("mlp_output", "mlp_output"),
        ("output", "layer_output"),
    )
    for layer in range(layer_count):
        for suffix, event_class in suffixes:
            result.append({
                "order": order,
                "name": f"layer_{layer}.{suffix}",
                "event_class": event_class,
                "layer": layer,
            })
            order += 1
    result.append({"order": order, "name": "final_norm", "event_class": "final_norm", "layer": None})
    result.append({"order": order + 1, "name": "selected_logits", "event_class": "selected_logits", "layer": None})
    return result


def finite_components(row: dict[str, Any]) -> dict[str, bool]:
    return {
        "candidate": math.isfinite(float(row["candidate_logp"])),
        "suffix": math.isfinite(float(row["suffix_mean_logp"])),
        "total": math.isfinite(float(row["total_score"])),
    }


def main() -> None:
    source_prereg = read_json(SOURCE_ROOT / "protocol" / "preregistration.json")
    source_final = read_json(SOURCE_ROOT / "analysis" / "final_summary.json")
    source_audit = read_json(SOURCE_ROOT / "audit" / "result_audit.json")

    model_specs: dict[str, Any] = {}
    source_links: dict[str, Any] = {}
    copied_cases: dict[str, list[dict[str, Any]]] = {}
    config_checks: dict[str, bool] = {}
    for model_name in MODELS:
        cases = read_jsonl(SOURCE_ROOT / "protocol" / f"cases.{model_name}.jsonl")
        scores = read_jsonl(SOURCE_ROOT / "behavior" / model_name / "scores.jsonl")
        summary = read_json(SOURCE_ROOT / "behavior" / model_name / "summary.json")
        config_path = Path(MODEL_CONFIGS[model_name]["path"]) / "config.json"
        config = read_json(config_path)
        layer_count = int(config["num_hidden_layers"])
        hidden_size = int(config["hidden_size"])
        vocab_size = int(config["vocab_size"])
        model_specs[model_name] = {
            "local_path": str(Path(MODEL_CONFIGS[model_name]["path"]).relative_to(ROOT)).replace("\\", "/"),
            "model_type": config["model_type"],
            "layer_count": layer_count,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "expected_event_count_per_case": len(event_registry(layer_count)),
            "event_registry": event_registry(layer_count),
            "source_parameter_count": sum(int(value) for value in summary["precision"]["parameter_dtypes"].values()),
        }
        source_links[model_name] = {
            "case_digest": source_prereg["case_digests"][model_name],
            "score_detail_digest": summary["detail_digest"],
            "source_finite_count": summary["finite_count"],
            "source_component_finite_counts": {
                component: sum(finite_components(row)[component] for row in scores)
                for component in ("candidate", "suffix", "total")
            },
        }
        copied_cases[model_name] = cases
        config_checks[f"{model_name}_layer_count"] = layer_count == EXPECTED_LAYERS[model_name]
        config_checks[f"{model_name}_hidden_size"] = hidden_size == EXPECTED_HIDDEN[model_name]
        config_checks[f"{model_name}_vocab_size"] = vocab_size == EXPECTED_VOCAB[model_name]

    prereg_core = {
        "schema_version": "phase1128_fp16_numeric_formation_preregistration.v1",
        "phase": PHASE,
        "objective": (
            "Replay all 320 frozen Phase1126 cases and locate the first FP16 non-finite event at scored "
            "prediction positions. This is an instrument-boundary audit, not a semantic hidden-state scan."
        ),
        "source": {
            "phase1126_protocol_digest": source_prereg["protocol_digest"],
            "phase1126_final_digest": source_final["final_digest"],
            "phase1126_audit_digest": source_audit["audit_digest"],
            "material_digest": source_prereg["material_digest"],
            "links": source_links,
        },
        "models": list(MODELS),
        "precision": "fp16",
        "quantization": "none",
        "execution_order": list(MODELS),
        "case_policy": {
            "all_frozen_behavior_cases": True,
            "case_count_per_model": 320,
            "hidden_holdout_scored": False,
            "same_exact_length_batching_as_phase1126": True,
            "batch_sizes": {"qwen3": 10, "glm4": 4, "deepseek7b": 4},
        },
        "measurement": {
            "scope": "only candidate and suffix prediction positions already scored in Phase1126",
            "events_per_layer": list(EVENTS_PER_LAYER),
            "event_scalar_fields": ["finite_count", "nonfinite_count", "max_abs_finite", "dtype", "device"],
            "raw_hidden_vectors_saved": False,
            "semantic_similarity_computed": False,
            "first_event_rule": "minimum frozen event order with nonfinite_count > 0",
            "source_score_replay": "unchanged Phase1126 candidate-sum plus suffix-mean score",
        },
        "model_specs": model_specs,
        "predictions": {
            "P1": "all source, case, model-config, precision, and event-registry identity checks pass",
            "P2": "rerun candidate, suffix, and total finite flags exactly reproduce Phase1126 for every case",
            "P3": "every source-nonfinite case has an observed first non-finite event in the frozen registry",
            "P4": "Qwen3 is a healthy reference with no tracked non-finite event",
            "P5": "first-event concentration is reported descriptively; no layer or component is assumed in advance",
            "P6": "no numerical event is interpreted as a semantic representation or behavioral mechanism",
        },
        "automatic_refinement_gate": {
            "minimum_source_nonfinite_cases": 50,
            "minimum_same_exact_event_fraction": 0.90,
            "allowed_event_classes": list(REFINABLE_EVENT_CLASSES),
            "requires_exact_source_parity": True,
            "meaning": (
                "passing authorizes one separately frozen numerical subcomponent audit only; it does not authorize "
                "semantic, hotspot, causal, BF16, FP32, or behavior-rescue work"
            ),
        },
        "forbidden": [
            "no case deletion, retry-after-result, threshold change, or hidden-holdout scoring",
            "no BF16, FP32, quantization, or precision substitution",
            "no raw hidden vector, Gram, probe, attention identity, neuron, SAE, patching, or causal analysis",
            "no reinterpretation of a numerical boundary as content or semantic mechanism",
            "no use of this audit to backfill or rescue the failed Phase1126 cross-model behavior gate",
            "no automatic refinement unless the frozen gate passes",
        ],
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)

    checks = {
        "source_protocol_digest": source_prereg["protocol_digest"] == EXPECTED_SOURCE_PROTOCOL_DIGEST,
        "source_final_digest": source_final["final_digest"] == EXPECTED_SOURCE_FINAL_DIGEST,
        "source_audit_digest": source_audit["audit_digest"] == EXPECTED_SOURCE_AUDIT_DIGEST,
        "source_audit_passed": source_audit["passed"] is True,
        "source_hidden_gate_closed": source_final["auto_continue"]["value"] is False,
        "source_detail_digests": all(
            source_links[model]["score_detail_digest"] == EXPECTED_SOURCE_DETAIL_DIGESTS[model]
            for model in MODELS
        ),
        "source_finite_counts": all(
            source_links[model]["source_finite_count"] == EXPECTED_FINITE_COUNTS[model]
            for model in MODELS
        ),
        "all_case_counts_320": all(len(copied_cases[model]) == 320 for model in MODELS),
        "only_behavior_partitions": all(
            set(row["partition"] for row in copied_cases[model])
            == set(source_protocol.BEHAVIOR_PARTITIONS)
            for model in MODELS
        ),
        "case_digests_match": all(
            source_protocol.digest(copied_cases[model]) == source_prereg["case_digests"][model]
            for model in MODELS
        ),
        **config_checks,
        "event_registries_unique": all(
            len({event["order"] for event in model_specs[model]["event_registry"]})
            == len(model_specs[model]["event_registry"])
            for model in MODELS
        ),
        "auto_gate_is_narrow": prereg["automatic_refinement_gate"]["minimum_source_nonfinite_cases"] >= 50
        and prereg["automatic_refinement_gate"]["minimum_same_exact_event_fraction"] >= 0.90,
    }
    audit_core = {
        "schema_version": "phase1128_fp16_numeric_formation_protocol_audit.v1",
        "phase": PHASE,
        "checks": checks,
        "passed_count": sum(checks.values()),
        "total_count": len(checks),
        "passed": all(checks.values()),
        "protocol_digest": prereg["protocol_digest"],
    }
    audit = dict(audit_core)
    audit["audit_digest"] = digest(audit_core)

    for model_name, rows in copied_cases.items():
        write_jsonl(OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl", rows)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", prereg)
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["passed"]:
        raise RuntimeError("Phase1128 protocol audit failed")


if __name__ == "__main__":
    main()
