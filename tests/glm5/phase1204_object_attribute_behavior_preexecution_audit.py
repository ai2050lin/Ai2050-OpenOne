#!/usr/bin/env python3
"""Independent zero-output audit for the sealed Phase1204 runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
PHASE = 1204
OUT_ROOT = TEST_ROOT / "result/phase1204_object_attribute_behavior_execution"
CONTRACT_PATH = OUT_ROOT / "protocol/execution_contract.json"
AUDIT_PATH = OUT_ROOT / "audit/preexecution_audit.json"
UPSTREAM_ROOT = TEST_ROOT / "result/phase1203_object_attribute_behavior_protocol"
UPSTREAM_FINAL = UPSTREAM_ROOT / "analysis/final.json"
UPSTREAM_PROTOCOL = UPSTREAM_ROOT / "protocol/behavior_protocol.json"
UPSTREAM_AUDIT = UPSTREAM_ROOT / "audit/independent_protocol_audit.json"
MANIFEST_DIR = UPSTREAM_ROOT / "protocol/model_manifests"

MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
EXPECTED_BATCH = {"qwen3": 16, "glm4": 2, "deepseek7b": 4}
EXPECTED_CASES = 4608
EXPECTED_FINAL_DIGEST = "ef1c8825f190682f165f4b7080130cf043fabd1cd6a6be30a2cb0199eca2f198"
EXPECTED_PROTOCOL_DIGEST = "62ff69b41c7de1407beb9b26ccdaf9e4eed8ea342959356e575a1dd1080434a6"
EXPECTED_UPSTREAM_AUDIT_DIGEST = "a4cc6e3668c7a5dccaf45e2bf22293cb72e12747bd038c04edf710e2e24dbc3d"

SOURCE_PATHS = {
    "execution": TEST_ROOT / "phase1204_object_attribute_behavior_execution.py",
    "preexecution_audit": Path(__file__).resolve(),
    "sequential_runner": TEST_ROOT / "phase1204_run_sequential.py",
    "finalize": TEST_ROOT / "phase1204_object_attribute_behavior_finalize.py",
    "result_audit": TEST_ROOT / "phase1204_object_attribute_behavior_result_audit.py",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def validate_digest(payload: dict[str, Any], field: str) -> bool:
    return digest({key: value for key, value in payload.items() if key != field}) == payload.get(field)


def audit(write: bool) -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    upstream_final = read_json(UPSTREAM_FINAL)
    upstream_protocol = read_json(UPSTREAM_PROTOCOL)
    upstream_audit = read_json(UPSTREAM_AUDIT)
    checks: list[dict[str, Any]] = []

    add(checks, "phase", contract.get("phase") == PHASE)
    add(checks, "contract_digest", validate_digest(contract, "contract_digest"))
    add(checks, "source_set", set(contract.get("source_hashes", {})) == set(SOURCE_PATHS))
    actual_source_hashes = {name: file_sha256(path) for name, path in SOURCE_PATHS.items()}
    add(checks, "source_hashes", contract.get("source_hashes") == actual_source_hashes)
    add(checks, "upstream_final_digest", upstream_final.get("final_digest") == EXPECTED_FINAL_DIGEST)
    add(checks, "upstream_final_integrity", validate_digest(upstream_final, "final_digest"))
    add(checks, "upstream_protocol_digest", upstream_protocol.get("protocol_digest") == EXPECTED_PROTOCOL_DIGEST)
    add(checks, "upstream_protocol_integrity", validate_digest(upstream_protocol, "protocol_digest"))
    add(
        checks,
        "upstream_audit",
        upstream_audit.get("gate_pass") is True
        and upstream_audit.get("checks_passed") == upstream_audit.get("checks_total") == 67
        and upstream_audit.get("audit_digest") == EXPECTED_UPSTREAM_AUDIT_DIGEST,
    )
    add(
        checks,
        "upstream_authorization",
        upstream_final.get("authorized_next", {}).get("phase1204_sequential_fp16_behavior_execution") is True
        and upstream_final.get("authorized_next", {}).get("hidden_state_scan") is False,
    )

    execution = contract.get("execution", {})
    add(checks, "model_order", tuple(execution.get("model_order", [])) == MODEL_ORDER)
    add(checks, "fixed_batch", execution.get("fixed_batch_size") == EXPECTED_BATCH)
    add(checks, "fp16", execution.get("precision") == "FP16")
    add(checks, "no_quantization", execution.get("quantization") == "none")
    add(checks, "cuda_required", execution.get("cuda_required") is True)
    add(checks, "one_model_process", execution.get("one_model_per_process") is True)
    add(checks, "release_between_models", execution.get("release_before_next_model") is True)
    add(checks, "exact_manifest_inputs", execution.get("exact_manifest_input_ids_only") is True)
    add(checks, "no_runtime_retokenization", execution.get("runtime_retokenization") is False)
    add(checks, "length_bucketing", execution.get("exact_length_bucketing") is True)
    add(checks, "no_adaptive_oom", execution.get("adaptive_oom_fallback") is False)
    add(checks, "no_generation", execution.get("generation") is False)
    add(checks, "no_hidden", execution.get("output_hidden_states") is False)
    add(checks, "no_attention", execution.get("output_attentions") is False)
    add(checks, "tie_tolerance", abs(float(execution.get("tie_tolerance", -1)) - 1e-7) <= 1e-15)
    add(checks, "behavior_only_claim", contract.get("claim_boundary", {}).get("behavior_only") is True)
    add(checks, "no_mechanism_claim", contract.get("claim_boundary", {}).get("mechanism_claim") is False)

    execution_source = SOURCE_PATHS["execution"].read_text(encoding="utf-8")
    sequential_source = SOURCE_PATHS["sequential_runner"].read_text(encoding="utf-8")
    add(checks, "source_loads_fp16", "load_fp16(model_name)" in execution_source)
    add(checks, "source_audits_quantization", "quantization_audit(model)" in execution_source)
    add(checks, "source_checks_vocab_finite", "torch.isfinite(raw_logits).all(dim=-1)" in execution_source)
    add(checks, "source_fp32_logsoftmax", "torch.log_softmax(raw_logits.float(), dim=-1)" in execution_source)
    add(checks, "source_disables_cache", "use_cache=False" in execution_source)
    add(checks, "source_disables_hidden", "output_hidden_states=False" in execution_source)
    add(checks, "source_disables_attention", "output_attentions=False" in execution_source)
    add(checks, "source_no_generate_call", ".generate(" not in execution_source)
    add(checks, "source_no_tokenizer_encode", "tokenizer.encode(" not in execution_source)
    add(checks, "sequential_uses_frozen_order", "for model_name in execution.MODEL_ORDER" in sequential_source)
    add(checks, "sequential_subprocess", "subprocess.run" in sequential_source)

    manifest_summaries: dict[str, Any] = {}
    expected_manifest_digests = contract["upstream"]["manifest_digests"]
    for model_name in MODEL_ORDER:
        rows = read_jsonl(MANIFEST_DIR / f"{model_name}.jsonl")
        labels_single = all(len(ids) == 1 for row in rows for ids in row["candidate_token_ids"].values())
        labels_distinct = all(
            len({ids[0] for ids in row["candidate_token_ids"].values()}) == 3 for row in rows
        )
        add(checks, f"{model_name}_count", len(rows) == EXPECTED_CASES, len(rows))
        add(checks, f"{model_name}_digest", digest(rows) == expected_manifest_digests[model_name])
        add(checks, f"{model_name}_unique_items", len({row["item_id"] for row in rows}) == EXPECTED_CASES)
        add(checks, f"{model_name}_indices", [row["execution_index"] for row in rows] == list(range(EXPECTED_CASES)))
        add(checks, f"{model_name}_identity", all(row["model"] == model_name for row in rows))
        add(checks, f"{model_name}_single_token_candidates", labels_single)
        add(checks, f"{model_name}_distinct_candidate_ids", labels_distinct)
        add(checks, f"{model_name}_input_lengths", all(len(row["input_ids"]) == row["input_length"] for row in rows))
        manifest_summaries[model_name] = {
            "case_count": len(rows),
            "manifest_digest": digest(rows),
            "lengths": sorted({row["input_length"] for row in rows}),
        }

    add(checks, "no_behavior_outputs_before_audit", not (OUT_ROOT / "behavior").exists())
    add(checks, "no_analysis_outputs_before_audit", not (OUT_ROOT / "analysis").exists())
    gate = all(check["pass"] for check in checks)
    output: dict[str, Any] = {
        "phase": PHASE,
        "kind": "independent_zero_output_runner_audit",
        "contract_digest": contract["contract_digest"],
        "gate_pass": gate,
        "checks_passed": sum(check["pass"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "manifest_summaries": manifest_summaries,
        "behavior_cases_scored": 0,
        "model_weights_loaded": False,
    }
    output["audit_digest"] = digest(output)
    if write:
        if AUDIT_PATH.exists():
            raise RuntimeError("preexecution audit already exists")
        write_json(AUDIT_PATH, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    output = audit(args.write)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not output["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
