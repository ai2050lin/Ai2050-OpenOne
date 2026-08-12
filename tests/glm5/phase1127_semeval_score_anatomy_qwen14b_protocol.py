#!/usr/bin/env python3
"""Freeze the Phase1127 SemEval score-anatomy and Qwen3-14B replication protocol."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1126_semeval_lexsub_natural_cloze_protocol as source_protocol


PHASE = 1127
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1127_semeval_score_anatomy_qwen14b"
SOURCE_ROOT = source_protocol.OUT_ROOT
MODEL_ROOT = ROOT / "models" / "hf" / "Qwen3-14B"
PHASE1118_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1118_qwen3_14b_fp16_offload_smoke"
PHASE1119_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1119_qwen3_4b_14b_scale"

EXPECTED_SOURCE_PROTOCOL_DIGEST = "f76079467e1c829fbc2a875a52b37e0b26a32fc3d49e0dfb1ee14a042ec9ba4d"
EXPECTED_SOURCE_FINAL_DIGEST = "093a69a78fa77a8cd2674a6de51d049042c2e8e6c7c224105747b582d4049f25"
EXPECTED_SOURCE_AUDIT_DIGEST = "f4fc42968ce0bd731a411361ed59f0bcf3172994dfbd8168bdd8ec4c4a52223b"
EXPECTED_PHASE1118_AUDIT_DIGEST = "856e8a1d5987f4f1f8e455010fb23b9b0cc49d35b8c1fc73d221db0af45d1280"
EXPECTED_QWEN14B_MANIFEST_DIGEST = "92ae3a1b0dbf063ac1ecaca5bab4d95f32a52fe1de19277c418d42866974417c"
EXPECTED_QWEN14B_PARAMETER_COUNT = 14_768_307_200
EXPECTED_QWEN_TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"

SOURCE_MODELS = ("qwen3", "glm4", "deepseek7b")
PARTITIONS = ("discovery", "independent_confirmation")
ROUTES = ("active", "matched_deranged")
COMPONENTS = ("candidate", "suffix", "total")
BATCH_SIZE = 16


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


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


def main() -> None:
    source_prereg = read_json(SOURCE_ROOT / "protocol" / "preregistration.json")
    source_audit = read_json(SOURCE_ROOT / "audit" / "result_audit.json")
    source_final = read_json(SOURCE_ROOT / "analysis" / "final_summary.json")
    phase1118_protocol = read_json(PHASE1118_ROOT / "protocol" / "protocol.json")
    phase1118_audit = read_json(PHASE1118_ROOT / "audit" / "result_audit.json")
    phase1119_prereg = read_json(PHASE1119_ROOT / "protocol" / "preregistration.json")
    phase1119_audit = read_json(PHASE1119_ROOT / "audit" / "result_audit.json")

    source_cases = read_jsonl(SOURCE_ROOT / "protocol" / "cases.qwen3.jsonl")
    source_summaries: dict[str, Any] = {}
    source_score_digests: dict[str, str] = {}
    for model in SOURCE_MODELS:
        summary = read_json(SOURCE_ROOT / "behavior" / model / "summary.json")
        scores = read_jsonl(SOURCE_ROOT / "behavior" / model / "scores.jsonl")
        if source_protocol.digest(scores) != summary["detail_digest"]:
            raise RuntimeError(f"Phase1126 detail digest mismatch: {model}")
        source_summaries[model] = summary
        source_score_digests[model] = summary["detail_digest"]

    qwen4_tokenizer = ROOT / "models" / "hf" / "qwen3-4b" / "tokenizer.json"
    qwen14_tokenizer = MODEL_ROOT / "tokenizer.json"
    tokenizer_hashes = {
        "qwen3_4b": sha256_file(qwen4_tokenizer),
        "qwen3_14b": sha256_file(qwen14_tokenizer),
    }

    thresholds = dict(source_prereg["thresholds"])
    component_thresholds = {
        "finite_rate_min": thresholds["finite_rate_min"],
        "active_positive_rate_min": thresholds["active_positive_rate_min"],
        "active_median_min": thresholds["active_median_min"],
        "matched_advantage_median_min": thresholds["matched_advantage_median_min"],
        "matched_advantage_positive_rate_min": thresholds["matched_advantage_positive_rate_min"],
    }

    prereg_core = {
        "schema_version": "phase1127_semeval_score_anatomy_qwen14b_preregistration.v1",
        "phase": PHASE,
        "objective": (
            "Decompose the already frozen Phase1126 natural-cloze score into candidate-span and suffix-use "
            "interactions, then prospectively test the unchanged carrier on Qwen3-14B in FP16."
        ),
        "evidence_split": {
            "score_anatomy": "post hoc diagnostic over frozen Phase1126 outputs; it is not independent confirmation",
            "qwen3_14b": "prospective model endpoint; no Qwen3-14B SemEval score was read before this freeze",
        },
        "source": {
            "phase1126_protocol_digest": source_prereg["protocol_digest"],
            "phase1126_final_digest": source_final["final_digest"],
            "phase1126_audit_digest": source_audit["audit_digest"],
            "phase1126_material_digest": source_prereg["material_digest"],
            "source_score_digests": source_score_digests,
            "source_models": list(SOURCE_MODELS),
            "source_authorized_models": source_final["authorized_models"],
        },
        "model": {
            "name": "qwen3_14b",
            "repo": phase1118_protocol["repo"],
            "commit": phase1118_protocol["expected_commit"],
            "manifest_digest": phase1119_prereg["model_manifest_digests"]["qwen3_14b"],
            "expected_parameter_count": EXPECTED_QWEN14B_PARAMETER_COUNT,
            "precision": "fp16",
            "quantization": "none",
            "placement": "frozen Phase1118 CUDA-plus-disk device map",
            "phase1118_protocol_digest": phase1118_protocol["protocol_digest"],
            "phase1118_audit_digest": phase1118_audit["audit_digest"],
            "device_map": phase1118_protocol["device_map"],
        },
        "carrier": {
            "case_count": len(source_cases),
            "case_digest": source_protocol.digest(source_cases),
            "partitions": list(PARTITIONS),
            "routes": list(ROUTES),
            "components": list(COMPONENTS),
            "batch_size": BATCH_SIZE,
            "qwen_tokenizer_sha256": EXPECTED_QWEN_TOKENIZER_SHA256,
            "hidden_holdout_scored": False,
        },
        "score_identity": {
            "case": "S_total = S_candidate + S_suffix",
            "interaction": "z_total = z_candidate + z_suffix",
            "additivity_tolerance": 1e-6,
            "component_gate_note": (
                "candidate and suffix components reuse the directional and matched-null gates but omit the lexical "
                "gate because character coverage is not commensurate with either isolated log-probability component"
            ),
        },
        "thresholds": thresholds,
        "component_thresholds": component_thresholds,
        "predictions": {
            "P1": "all source, tokenizer, model-carrier, and protocol identity checks pass",
            "P2": "stored total-score interactions equal candidate plus suffix interactions within 1e-6",
            "P3": "Qwen3-4B candidate-only component passes both frozen behavior partitions",
            "P4": "Qwen3-4B suffix-only component passes both frozen behavior partitions",
            "P5": "Qwen3-14B is numerically qualified and passes all six unchanged total-score gates in both partitions",
            "P6": "Qwen3-4B and Qwen3-14B provide a same-family behavior replication on the unchanged carrier",
            "P7": "Phase1126 cross-model gate remains closed regardless of the Qwen3-14B result",
            "P8": "hidden_holdout and all hidden/component/causal scans remain unauthorized",
        },
        "forbidden": [
            "no change to Phase1126 panels, partitions, scores, controls, thresholds, or source model results",
            "no reinterpretation of Qwen3-14B as a member of the frozen Phase1126 three-model gate",
            "no hidden_holdout scoring",
            "no hidden-state, attention, MLP, head, SAE, neuron, patching, or causal scan",
            "no BF16, FP32, quantization, retry-after-result, case deletion, or threshold change",
            "no claim that score anatomy isolates pure semantics",
            "no automatic continuation from a same-family replication",
        ],
        "auto_continue": False,
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)

    checks = {
        "source_protocol_digest": source_prereg["protocol_digest"] == EXPECTED_SOURCE_PROTOCOL_DIGEST,
        "source_final_digest": source_final["final_digest"] == EXPECTED_SOURCE_FINAL_DIGEST,
        "source_audit_digest": source_audit["audit_digest"] == EXPECTED_SOURCE_AUDIT_DIGEST,
        "source_audit_passed": source_audit["passed"] is True,
        "source_gate_remains_failed": source_final["predictions"]["P2_cross_resource_behavior"] is False,
        "source_auto_continue_false": source_final["auto_continue"]["value"] is False,
        "source_score_digests_linked": all(
            source_summaries[model]["detail_digest"] == source_score_digests[model]
            for model in SOURCE_MODELS
        ),
        "source_cases_match": source_protocol.digest(source_cases) == source_prereg["case_digests"]["qwen3"],
        "case_count_320": len(source_cases) == 320,
        "only_behavior_partitions": sorted({row["partition"] for row in source_cases}) == sorted(PARTITIONS),
        "hidden_holdout_absent": all(row["partition"] != "hidden_holdout" for row in source_cases),
        "phase1118_audit_passed": phase1118_audit["all_checks_passed"] is True,
        "phase1118_audit_digest": phase1118_audit["audit_digest"] == EXPECTED_PHASE1118_AUDIT_DIGEST,
        "phase1119_audit_passed": phase1119_audit["all_checks_passed"] is True,
        "qwen14_manifest": phase1119_prereg["model_manifest_digests"]["qwen3_14b"] == EXPECTED_QWEN14B_MANIFEST_DIGEST,
        "qwen14_parameter_count": phase1119_prereg["expected_parameter_counts"]["qwen3_14b"] == EXPECTED_QWEN14B_PARAMETER_COUNT,
        "tokenizers_identical": len(set(tokenizer_hashes.values())) == 1,
        "tokenizer_expected": tokenizer_hashes["qwen3_14b"] == EXPECTED_QWEN_TOKENIZER_SHA256,
        "model_files_present": MODEL_ROOT.exists() and (MODEL_ROOT / "model.safetensors.index.json").exists(),
        "auto_continue_frozen_false": prereg["auto_continue"] is False,
    }
    audit_core = {
        "schema_version": "phase1127_semeval_score_anatomy_qwen14b_protocol_audit.v1",
        "phase": PHASE,
        "checks": checks,
        "passed_count": sum(checks.values()),
        "total_count": len(checks),
        "passed": all(checks.values()),
        "protocol_digest": prereg["protocol_digest"],
        "tokenizer_hashes": tokenizer_hashes,
    }
    audit = dict(audit_core)
    audit["audit_digest"] = digest(audit_core)

    write_jsonl(OUT_ROOT / "protocol" / "cases.qwen3_14b.jsonl", source_cases)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", prereg)
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["passed"]:
        raise RuntimeError("Phase1127 protocol audit failed")


if __name__ == "__main__":
    main()
