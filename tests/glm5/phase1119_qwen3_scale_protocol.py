#!/usr/bin/env python3
"""Freeze the Qwen3-4B/14B matched contextual-modulation scale protocol."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1119_qwen3_4b_14b_scale"
SOURCE_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1117_pythia_training_dynamics_verified_safetensors_v4"
)
PHASE1118_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1118_qwen3_14b_fp16_offload_smoke"
)
MODEL_ROOTS = {
    "qwen3_4b": ROOT / "models" / "hf" / "qwen3-4b",
    "qwen3_14b": ROOT / "models" / "hf" / "Qwen3-14B",
}
EXPECTED_PARAMETER_COUNTS = {
    "qwen3_4b": 4_022_468_096,
    "qwen3_14b": 14_768_307_200,
}
PHASE = 1119
SPLITS = ("discovery", "independent_confirmation", "heldout")
TEMPLATE_COUNT = 6

ABSOLUTE_THRESHOLDS = {
    "minimum_finite_fraction": 0.99,
    "minimum_overall_direction_accuracy": 0.80,
    "minimum_split_direction_accuracy": 0.75,
    "minimum_template_direction_accuracy": 0.65,
    "minimum_overall_control_advantage": 0.15,
    "minimum_split_control_advantage": 0.10,
    "minimum_template_control_advantage": 0.05,
    "minimum_concept_positive_fraction": 0.80,
    "minimum_positive_concepts_per_split": 15,
}
SCALE_THRESHOLDS = {
    "minimum_direction_gain": 0.03,
    "minimum_control_advantage_gain": 0.03,
    "minimum_bidirectional_gain": 0.03,
    "minimum_concept_fraction_gain": 0.0,
    "maximum_split_direction_regression": 0.05,
    "maximum_split_control_advantage_regression": 0.05,
}
PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "The Phase1117 source audit, the exact Qwen3 tokenizer identity, all 57 concepts, "
        "684 cases, 2736 candidate continuations, weight manifests, and protocol digests pass "
        "before either scale result is analyzed."
    ),
    "P2": (
        "Both models run in FP16 without quantization, produce finite candidate scores for at "
        "least 99 percent of cases, and use the frozen shared tokenizer and prompts."
    ),
    "P3": (
        "Qwen3-14B passes every frozen absolute direction, split, template, matched-control, "
        "and concept gate."
    ),
    "P4": (
        "Relative to the freshly rerun Qwen3-4B baseline, Qwen3-14B gains at least 0.03 in "
        "direction accuracy, control advantage, and bidirectional use; concept-positive "
        "fraction does not fall, and no split regresses by more than 0.05."
    ),
    "P5": (
        "A P3/P4 pass is one same-family scale data point, not an identified universal effect "
        "of parameter count and not evidence for hidden or causal semantic structure."
    ),
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def continuation_token_id(tokenizer: Any, prompt: str, token: str) -> int:
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    full = tokenizer.encode(prompt + " " + token, add_special_tokens=False)
    if full[: len(prefix)] != prefix or len(full) != len(prefix) + 1:
        raise RuntimeError(f"candidate is not one stable continuation token: {token!r}")
    return int(full[-1])


def weight_manifest(model_root: Path) -> list[dict[str, Any]]:
    files = sorted(model_root.glob("model-*-of-*.safetensors"))
    if not files:
        raise RuntimeError(f"no sharded safetensors weights under {model_root}")
    return [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in files
    ]


def build_protocol() -> dict[str, Any]:
    source_audit = read_json(SOURCE_ROOT / "audit" / "result_audit.json")
    source_prereg = read_json(SOURCE_ROOT / "protocol" / "preregistration.json")
    source_cases = list(read_jsonl(SOURCE_ROOT / "protocol" / "cases.jsonl"))
    phase1118_audit = read_json(PHASE1118_ROOT / "audit" / "result_audit.json")
    if not source_audit["all_checks_passed"] or not phase1118_audit["all_checks_passed"]:
        raise RuntimeError("source audit failed")
    if digest(source_cases) != source_prereg["case_digest"]:
        raise RuntimeError("Phase1117 source case digest mismatch")

    tokenizers = {
        name: AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        for name, path in MODEL_ROOTS.items()
    }
    tokenizer_manifests = {
        name: {
            "tokenizer_json_size": (path / "tokenizer.json").stat().st_size,
            "tokenizer_json_sha256": file_sha256(path / "tokenizer.json"),
            "tokenizer_config_sha256": file_sha256(path / "tokenizer_config.json"),
            "vocab_size": len(tokenizers[name]),
        }
        for name, path in MODEL_ROOTS.items()
    }
    if tokenizer_manifests["qwen3_4b"] != tokenizer_manifests["qwen3_14b"]:
        raise RuntimeError("Qwen3 tokenizer manifests are not identical")
    tokenizer = tokenizers["qwen3_4b"]

    rows: list[dict[str, Any]] = []
    for source in source_cases:
        prompt = source["raw_prompt"]
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        true_ids = [continuation_token_id(tokenizer, prompt, label) for label in source["true_candidate_labels"]]
        control_ids = [
            continuation_token_id(tokenizer, prompt, label)
            for label in source["control_candidate_labels"]
        ]
        rows.append(
            {
                "schema_version": "phase1119_qwen3_scale_case.v1",
                "phase": PHASE,
                "case_index": int(source["case_index"]),
                "record_id": source["record_id"].replace("phase1117", "phase1119"),
                "pair_id": source["pair_id"].replace("phase1117", "phase1119"),
                "concept_id": source["concept_id"],
                "source_phase": source["source_phase"],
                "split": source["split"],
                "template": int(source["template"]),
                "sense": int(source["sense"]),
                "base": source["base"],
                "native_example": source["native_example"],
                "raw_prompt": prompt,
                "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "input_ids": [int(value) for value in input_ids],
                "true_candidate_labels": list(source["true_candidate_labels"]),
                "true_candidate_ids": true_ids,
                "control_concept_id": source["control_concept_id"],
                "control_candidate_labels": list(source["control_candidate_labels"]),
                "control_candidate_ids": control_ids,
            }
        )

    model_manifests = {name: weight_manifest(path) for name, path in MODEL_ROOTS.items()}
    phase1118_shards = {
        row["name"]: (row["actual_size"], row["actual_sha256"])
        for row in phase1118_audit["shards"]
    }
    if any(
        phase1118_shards.get(row["name"]) != (row["size"], row["sha256"])
        for row in model_manifests["qwen3_14b"]
    ):
        raise RuntimeError("Qwen3-14B manifest differs from the Phase1118 verified carrier")

    split_counts = Counter(row["split"] for row in rows)
    pair_counts = Counter(row["pair_id"] for row in rows)
    checks = {
        "source_audit": source_audit["all_checks_passed"],
        "phase1118_carrier_audit": phase1118_audit["all_checks_passed"],
        "source_case_count": len(source_cases) == 684,
        "case_count": len(rows) == 684,
        "case_indices": [row["case_index"] for row in rows] == list(range(684)),
        "pair_structure": len(pair_counts) == 342 and set(pair_counts.values()) == {2},
        "concept_count": len({row["concept_id"] for row in rows}) == 57,
        "split_balance": split_counts == Counter({split: 228 for split in SPLITS}),
        "template_balance": Counter(row["template"] for row in rows)
        == Counter({template: 114 for template in range(TEMPLATE_COUNT)}),
        "candidate_boundaries": all(
            len(row["true_candidate_ids"]) == len(row["control_candidate_ids"]) == 2
            for row in rows
        ),
        "tokenizer_identity": tokenizer_manifests["qwen3_4b"] == tokenizer_manifests["qwen3_14b"],
        "model_manifest_counts": len(model_manifests["qwen3_4b"]) == 3
        and len(model_manifests["qwen3_14b"]) == 8,
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")

    case_path = OUT_ROOT / "protocol" / "cases.jsonl"
    write_jsonl(case_path, rows)
    core = {
        "schema_version": "phase1119_qwen3_scale_preregistration.v1",
        "phase": PHASE,
        "models": list(MODEL_ROOTS),
        "precision": "fp16",
        "quantization": "none",
        "source_phase1117_protocol_digest": source_prereg["protocol_digest"],
        "source_phase1117_audit_digest": source_audit["audit_digest"],
        "source_phase1118_audit_digest": phase1118_audit["audit_digest"],
        "tokenizer_manifest": tokenizer_manifests["qwen3_4b"],
        "model_manifests": model_manifests,
        "model_manifest_digests": {
            name: digest(manifest) for name, manifest in model_manifests.items()
        },
        "expected_parameter_counts": EXPECTED_PARAMETER_COUNTS,
        "case_count": len(rows),
        "pair_count": len(pair_counts),
        "concept_count": 57,
        "case_digest": digest(rows),
        "case_sha256": file_sha256(case_path),
        "absolute_thresholds": ABSOLUTE_THRESHOLDS,
        "scale_thresholds": SCALE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "checks": checks,
        "hidden_or_causal_authorized": False,
        "interpretive_limits": [
            "This is one 4B-to-14B same-family scale interval, not a full scale law.",
            "Training data, architecture details, and alignment can still vary with model size.",
            "Context effects retain lexical, topical, syntactic, and candidate interaction terms.",
            "No hidden-state, component, neuron, or causal scan is authorized.",
        ],
    }
    prereg = dict(core)
    prereg["protocol_digest"] = digest(core)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", prereg)
    audit_core = {
        "schema_version": "phase1119_qwen3_scale_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    audit = dict(audit_core)
    audit["audit_digest"] = digest(audit_core)
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    return prereg


if __name__ == "__main__":
    print(json.dumps(build_protocol(), ensure_ascii=False, indent=2))
