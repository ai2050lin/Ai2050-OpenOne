#!/usr/bin/env python3
"""Independent zero-model audit for the Phase1230 Qwen3 behavior protocol."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
PHASE = 1230
SCRIPT = TEST_ROOT / "phase1230_qwen3_clock_compass_behavior_protocol.py"
AUDIT_SCRIPT = Path(__file__).resolve()

UPSTREAM_ROOT = TEST_ROOT / "result/phase1229_deanswer_clock_compass_material_contract"
UPSTREAM_FINAL = UPSTREAM_ROOT / "analysis/final.json"
UPSTREAM_FINAL_AUDIT = UPSTREAM_ROOT / "audit/independent_final_audit.json"
UPSTREAM_CONTRACT = UPSTREAM_ROOT / "protocol/material_contract.json"
UPSTREAM_MATERIAL = UPSTREAM_ROOT / "material/clock_compass_binding.jsonl"

EXPECTED_UPSTREAM_FINAL_DIGEST = "ade2d188d55e206a94330806fd9c81d280fe9429a288206e460d28e772bdc5a2"
EXPECTED_UPSTREAM_FINAL_AUDIT_DIGEST = "b20c6d5edb8655bfa5986d5237e693fe9bac8b5af8e354a9ce9dbcd946d5873d"
EXPECTED_UPSTREAM_CONTRACT_DIGEST = "e14563b9c25904de64911166b0bfb505921a4f19a44fcc9f2e7f0667cc718358"

OUT_ROOT = TEST_ROOT / "result/phase1230_qwen3_clock_compass_behavior_protocol"
CONTRACT_PATH = OUT_ROOT / "protocol/preregistration.json"
MANIFEST_PATH = OUT_ROOT / "protocol/qwen3_manifest.jsonl"
TOKEN_AUDIT_PATH = OUT_ROOT / "audit/tokenizer_interface_audit.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

SPLITS = ("discovery", "confirmation", "natural_use")
PANELS = ("active", "matched_null", "surface_order")
CANDIDATES = ("north", "east", "south", "west")
EXPECTED_ROWS = 9216
EXPECTED_ACTIVE = 3072
EXPECTED_BUNDLES = 2304
MAX_INPUT_LENGTH = 160


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_digest(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def imported_names(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def preaudit() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    source = SCRIPT.read_text(encoding="utf-8")
    imports = imported_names(source)
    upstream_final = read_json(UPSTREAM_FINAL)
    upstream_final_audit = read_json(UPSTREAM_FINAL_AUDIT)
    upstream_contract = read_json(UPSTREAM_CONTRACT)
    checks: dict[str, bool] = {
        "phase": contract.get("phase") == PHASE,
        "contract_self_digest": contract.get("contract_digest") == digest(strip_digest(contract, "contract_digest")),
        "compiler_hash": contract["source_hashes"]["protocol_compiler"] == file_sha256(SCRIPT),
        "audit_hash": contract["source_hashes"]["independent_audit"] == file_sha256(AUDIT_SCRIPT),
        "upstream_final_digest": upstream_final.get("final_digest") == EXPECTED_UPSTREAM_FINAL_DIGEST,
        "upstream_final_audit_digest": upstream_final_audit.get("audit_digest") == EXPECTED_UPSTREAM_FINAL_AUDIT_DIGEST,
        "upstream_final_audit_pass": upstream_final_audit.get("all_checks_passed") is True,
        "upstream_contract_digest": upstream_contract.get("contract_digest") == EXPECTED_UPSTREAM_CONTRACT_DIGEST,
        "upstream_file_hashes": contract["upstream"]["file_hashes"]["material"] == file_sha256(UPSTREAM_MATERIAL),
        "tokenizer_only_scope": contract["execution_scope"]["tokenizer_only"] is True,
        "no_model_weight_scope": contract["execution_scope"]["model_weights_loaded"] is False,
        "no_model_import": not any(name.startswith("transformers.models") or "AutoModel" in name for name in imports),
        "no_torch_import": "torch" not in imports,
        "no_model_loader_symbol": not any(
            name.endswith(("AutoModelForCausalLM", "load_fp16", "load_model"))
            for name in imports
        ),
        "typed_ledgers": set(contract["typed_behavior_ledgers"]) == {
            "Q0_numerical", "Q1_candidate_semantics", "Q2_active_counterfactual",
            "Q3_controls", "Q4_interface_invariance", "Q5_natural_first_token",
        },
        "typed_authorization": contract["typed_authorization_rule"]["Q5_does_not_veto"].startswith("a candidate-scored"),
        "candidate_registry": tuple(contract["interface"]["candidates"]) == CANDIDATES,
        "candidate_absence_required": contract["interface"]["candidate_token_absent_from_input"] is True,
        "fp16_future_only": contract["future_execution"]["precision"].startswith("FP16"),
        "exact_manifest_future": contract["interface"]["exact_manifest_input_only"] is True,
        "no_automatic_execution": contract["authorization"]["automatic_model_execution"] is False,
        "no_hidden_authorization": contract["authorization"]["hidden_scan"] is False,
        "formal_outputs_absent": not any(path.exists() for path in (MANIFEST_PATH, TOKEN_AUDIT_PATH, RESULT_AUDIT_PATH, FINAL_PATH, FINAL_AUDIT_PATH)),
    }
    result: dict[str, Any] = {
        "phase": PHASE,
        "audit_type": "independent_preaudit",
        "created_at_utc": utc_now(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
        "contract_digest": contract["contract_digest"],
    }
    result["audit_digest"] = digest(result)
    write_json(PREAUDIT_PATH, result)
    return result


def valid_span_registry(case: dict[str, Any]) -> bool:
    required = {
        "record_full", "record_object", "record_anchor", "record_relation", "record_value",
        "query_full", "query_subject", "query_anchor", "query_relation", "answer_boundary",
    }
    spans = case.get("role_token_spans", {})
    if set(spans) != required:
        return False
    length = int(case["input_length"])
    for role, entries in spans.items():
        if not entries:
            return False
        for start, end in entries:
            if not (0 <= int(start) < int(end) <= length):
                return False
        if role == "answer_boundary" and entries != [[length - 1, length]]:
            return False
    return int(case["prediction_token_index"]) == length - 1


def result_audit() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    preaudit_result = read_json(PREAUDIT_PATH)
    upstream_rows = read_jsonl(UPSTREAM_MATERIAL)
    manifest = read_jsonl(MANIFEST_PATH)
    token_audit = read_json(TOKEN_AUDIT_PATH)
    upstream_by_id = {row["item_id"]: row for row in upstream_rows}
    manifest_by_id = {row["item_id"]: row for row in manifest}

    split_counts = Counter(row["split"] for row in manifest)
    panel_counts = Counter(row["panel"] for row in manifest)
    split_panel_counts = Counter((row["split"], row["panel"]) for row in manifest)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        groups[row["bundle_id"]].append(row)
    active_groups = [group for group in groups.values() if group[0]["panel"] == "active"]

    candidate_registries = {
        canonical_json(row["candidate_token_ids"]) for row in manifest
    }
    candidate_registry = manifest[0]["candidate_token_ids"] if manifest else {}
    candidate_ids = [candidate_registry[name] for name in CANDIDATES] if manifest else []
    candidate_single_distinct = bool(
        manifest
        and all(len(ids) == 1 for ids in candidate_ids)
        and len({ids[0] for ids in candidate_ids}) == len(CANDIDATES)
    )
    candidate_input_overlap = sum(
        int(any(ids[0] in row["input_ids"] for ids in row["candidate_token_ids"].values()))
        for row in manifest
    )
    upstream_links_ok = all(
        row["item_id"] in upstream_by_id
        and row["phase1229_row_digest"] == upstream_by_id[row["item_id"]]["row_digest"]
        and row["gold_candidate"] == upstream_by_id[row["item_id"]]["gold_candidate"]
        and row["bundle_id"] == upstream_by_id[row["item_id"]]["bundle_id"]
        for row in manifest
    )
    row_digests_ok = all(
        row["manifest_row_digest"] == digest(strip_digest(row, "manifest_row_digest"))
        for row in manifest
    )
    input_digests_ok = all(
        row["input_ids_digest"] == digest(row["input_ids"])
        and row["input_id_multiset_digest"] == digest(sorted(row["input_ids"]))
        for row in manifest
    )
    active_length_match = all(len({row["input_length"] for row in group}) == 1 for group in active_groups)
    active_multiset_match = all(
        len({row["input_id_multiset_digest"] for row in group}) == 1 for group in active_groups
    )
    bundle_structure = all(
        len(group) == 4 and {int(row["binding_state"]) for row in group} == {0, 1, 2, 3}
        for group in groups.values()
    )
    role_spans_ok = all(valid_span_registry(row) for row in manifest)
    lengths = [int(row["input_length"]) for row in manifest]

    checks: dict[str, bool] = {
        "preaudit_pass": preaudit_result.get("all_checks_passed") is True,
        "preaudit_self_digest": preaudit_result.get("audit_digest") == digest(strip_digest(preaudit_result, "audit_digest")),
        "contract_self_digest": contract.get("contract_digest") == digest(strip_digest(contract, "contract_digest")),
        "source_immutability": contract["source_hashes"]["protocol_compiler"] == file_sha256(SCRIPT) and contract["source_hashes"]["independent_audit"] == file_sha256(AUDIT_SCRIPT),
        "upstream_immutability": contract["upstream"]["file_hashes"]["material"] == file_sha256(UPSTREAM_MATERIAL),
        "row_count": len(manifest) == EXPECTED_ROWS,
        "active_count": sum(row["panel"] == "active" for row in manifest) == EXPECTED_ACTIVE,
        "item_ids_unique": len(manifest_by_id) == len(manifest),
        "item_id_set_exact": set(manifest_by_id) == set(upstream_by_id),
        "execution_indices_exact": [row["execution_index"] for row in manifest] == list(range(EXPECTED_ROWS)),
        "upstream_links": upstream_links_ok,
        "manifest_row_digests": row_digests_ok,
        "input_digests": input_digests_ok,
        "split_counts": set(split_counts) == set(SPLITS) and set(split_counts.values()) == {3072},
        "panel_counts": set(panel_counts) == set(PANELS) and set(panel_counts.values()) == {3072},
        "split_panel_balance": set(split_panel_counts.values()) == {1024},
        "bundle_count": len(groups) == EXPECTED_BUNDLES,
        "bundle_structure": bundle_structure,
        "active_length_match": active_length_match,
        "active_id_multiset_match": active_multiset_match,
        "candidate_registry_constant": len(candidate_registries) == 1,
        "candidate_single_distinct": candidate_single_distinct,
        "candidate_input_overlap_zero": candidate_input_overlap == 0,
        "gold_token_link": all(row["gold_candidate_token_id"] == row["candidate_token_ids"][row["gold_candidate"]][0] for row in manifest),
        "role_span_registry": role_spans_ok,
        "input_length_bound": bool(lengths) and min(lengths) > 0 and max(lengths) <= MAX_INPUT_LENGTH,
        "token_audit_self_digest": token_audit.get("tokenizer_audit_digest") == digest(strip_digest(token_audit, "tokenizer_audit_digest")),
        "token_audit_manifest_digest": token_audit.get("manifest_digest") == digest(manifest),
        "token_audit_interface_gate": token_audit.get("interface_gate") is True,
        "token_audit_no_model": token_audit.get("model_weights_loaded") is False and token_audit.get("behavior_cases_scored") == 0,
        "token_audit_no_cuda": token_audit.get("cuda_used") is False,
        "token_audit_candidates": token_audit.get("candidate_token_ids") == candidate_registry,
        "token_audit_lengths": token_audit.get("input_length_min") == min(lengths) and token_audit.get("input_length_max") == max(lengths),
        "token_audit_active_matches": token_audit.get("active_length_match_rate") == 1.0 and token_audit.get("active_input_id_multiset_match_rate") == 1.0,
        "slow_fast_exact": token_audit.get("slow_fast_id_mismatch_count") == 0,
        "typed_spans_exact": token_audit.get("role_span_failure_count") == 0 and token_audit.get("prompt_embedding_failure_count") == 0,
        "claim_scope_still_zero_model": contract["execution_scope"]["behavior_cases_scored"] == 0 and contract["execution_scope"]["hidden_states"] is False,
    }
    metrics = {
        "row_count": len(manifest),
        "active_count": sum(row["panel"] == "active" for row in manifest),
        "bundle_count": len(groups),
        "candidate_token_ids": candidate_registry,
        "candidate_input_overlap_count": candidate_input_overlap,
        "input_length_min": min(lengths),
        "input_length_max": max(lengths),
        "input_length_mean": sum(lengths) / len(lengths),
        "active_length_match_rate": sum(len({row["input_length"] for row in group}) == 1 for group in active_groups) / len(active_groups),
        "active_id_multiset_match_rate": sum(len({row["input_id_multiset_digest"] for row in group}) == 1 for group in active_groups) / len(active_groups),
    }
    result: dict[str, Any] = {
        "phase": PHASE,
        "audit_type": "independent_result_audit",
        "created_at_utc": utc_now(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
        "metrics": metrics,
        "contract_digest": contract["contract_digest"],
        "manifest_digest": digest(manifest),
    }
    result["audit_digest"] = digest(result)
    write_json(RESULT_AUDIT_PATH, result)
    return result


def final_audit() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    preaudit_result = read_json(PREAUDIT_PATH)
    result = read_json(RESULT_AUDIT_PATH)
    token_audit = read_json(TOKEN_AUDIT_PATH)
    final = read_json(FINAL_PATH)
    checks = {
        "preaudit_pass": preaudit_result.get("all_checks_passed") is True,
        "result_audit_pass": result.get("all_checks_passed") is True,
        "final_self_digest": final.get("final_digest") == digest(strip_digest(final, "final_digest")),
        "contract_link": final.get("contract_digest") == contract.get("contract_digest"),
        "tokenizer_audit_link": final.get("tokenizer_audit_digest") == token_audit.get("tokenizer_audit_digest"),
        "preaudit_link": final.get("independent_preaudit_digest") == preaudit_result.get("audit_digest"),
        "result_audit_link": final.get("independent_result_audit_digest") == result.get("audit_digest"),
        "manifest_link": final.get("manifest_digest") == result.get("manifest_digest"),
        "status": final.get("status") == "qwen3_behavior_protocol_ready_not_executed",
        "no_new_k": final.get("k_ledger", {}).get("new_item") is None,
        "no_automatic_execution": final.get("authorization", {}).get("auto_continue") == 0 and final.get("authorization", {}).get("automatic_execution") is False,
        "no_hidden_authorization": final.get("authorization", {}).get("hidden_scan") is False,
        "no_causal_authorization": final.get("authorization", {}).get("causal_intervention") is False,
    }
    output: dict[str, Any] = {
        "phase": PHASE,
        "audit_type": "independent_final_audit",
        "created_at_utc": utc_now(),
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
        "final_digest": final["final_digest"],
    }
    output["audit_digest"] = digest(output)
    write_json(FINAL_AUDIT_PATH, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("preaudit", "result", "final"))
    stage = parser.parse_args().stage
    payload = {"preaudit": preaudit, "result": result_audit, "final": final_audit}[stage]()
    print(
        canonical_json(
            {
                "stage": stage,
                "all_checks_passed": payload["all_checks_passed"],
                "passed": payload["passed_count"],
                "total": payload["check_count"],
                "audit_digest": payload["audit_digest"],
            }
        )
    )


if __name__ == "__main__":
    main()
