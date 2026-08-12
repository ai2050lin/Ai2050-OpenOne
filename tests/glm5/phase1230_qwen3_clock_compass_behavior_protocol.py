#!/usr/bin/env python3
"""Freeze the Qwen3 behavior interface for the Phase1229 clock-compass family.

This phase may load the local Qwen3 tokenizer, but it never loads model weights,
scores behavior, captures hidden states, or performs an intervention.  It seals
the exact native-chat token IDs, answer continuations, token-role spans,
behavior ledgers, and claim-specific gates for a later execution phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
PHASE = 1230
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1230_qwen3_clock_compass_behavior_protocol_audit.py"

UPSTREAM_ROOT = TEST_ROOT / "result/phase1229_deanswer_clock_compass_material_contract"
UPSTREAM_FINAL = UPSTREAM_ROOT / "analysis/final.json"
UPSTREAM_FINAL_AUDIT = UPSTREAM_ROOT / "audit/independent_final_audit.json"
UPSTREAM_CONTRACT = UPSTREAM_ROOT / "protocol/material_contract.json"
UPSTREAM_MATERIAL = UPSTREAM_ROOT / "material/clock_compass_binding.jsonl"
UPSTREAM_DONORS = UPSTREAM_ROOT / "material/donor_registry.jsonl"
UPSTREAM_MATERIAL_AUDIT = UPSTREAM_ROOT / "audit/independent_material_audit.json"

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

MODEL_NAME = "qwen3"
MODEL_PATH = ROOT / "models/hf/qwen3-4b"
SYSTEM_PROMPT = (
    "Use only the supplied records. Return exactly one lowercase direction word "
    "and no explanation."
)
SPLITS = ("discovery", "confirmation", "natural_use")
PANELS = ("active", "matched_null", "surface_order")
CANDIDATES = ("north", "east", "south", "west")
EXPECTED_ROWS = 9216
EXPECTED_ACTIVE = 3072
MAX_INPUT_LENGTH = 160
TIE_TOLERANCE = 1e-7

FINITE_RATE_MIN = 1.0
PANEL_ACCURACY_MIN = 0.90
ACTIVE_WORST_MARGINAL_MIN = 0.80
ACTIVE_QUARTET_MIN = 0.75
CONTROL_INVARIANT_BUNDLE_MIN = 0.80
TEMPLATE_PAIR_MIN = 0.85
NATURAL_FIRST_TOKEN_MIN = 0.80
HIDDEN_BEHAVIOR_AXES = (
    "world_id",
    "template_id",
    "target_entity",
    "gold_candidate",
    "order_variant",
    "mapping_variant",
)


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
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not an object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def strip_digest(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def source_hashes() -> dict[str, str]:
    return {
        "protocol_compiler": file_sha256(SCRIPT),
        "independent_audit": file_sha256(AUDIT_SCRIPT),
    }


def upstream_hashes() -> dict[str, str]:
    final = read_json(UPSTREAM_FINAL)
    final_audit = read_json(UPSTREAM_FINAL_AUDIT)
    contract = read_json(UPSTREAM_CONTRACT)
    if final.get("final_digest") != EXPECTED_UPSTREAM_FINAL_DIGEST:
        raise RuntimeError("Phase1229 final digest mismatch")
    if final_audit.get("audit_digest") != EXPECTED_UPSTREAM_FINAL_AUDIT_DIGEST:
        raise RuntimeError("Phase1229 final audit digest mismatch")
    if contract.get("contract_digest") != EXPECTED_UPSTREAM_CONTRACT_DIGEST:
        raise RuntimeError("Phase1229 contract digest mismatch")
    if final_audit.get("all_checks_passed") is not True:
        raise RuntimeError("Phase1229 final audit did not pass")
    return {
        "final": file_sha256(UPSTREAM_FINAL),
        "final_audit": file_sha256(UPSTREAM_FINAL_AUDIT),
        "contract": file_sha256(UPSTREAM_CONTRACT),
        "material": file_sha256(UPSTREAM_MATERIAL),
        "donors": file_sha256(UPSTREAM_DONORS),
        "material_audit": file_sha256(UPSTREAM_MATERIAL_AUDIT),
    }


def build_contract() -> dict[str, Any]:
    contract: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1230.qwen3.clock_compass.behavior_protocol.v1",
        "created_at_utc": utc_now(),
        "objective": (
            "Freeze an exact Qwen3 native-chat behavior interface for the Phase1229 "
            "de-answer-loaded clock-to-compass family without loading model weights."
        ),
        "source_hashes": source_hashes(),
        "upstream": {
            "phase1229_final_digest": EXPECTED_UPSTREAM_FINAL_DIGEST,
            "phase1229_final_audit_digest": EXPECTED_UPSTREAM_FINAL_AUDIT_DIGEST,
            "phase1229_contract_digest": EXPECTED_UPSTREAM_CONTRACT_DIGEST,
            "file_hashes": upstream_hashes(),
        },
        "execution_scope": {
            "tokenizer_only": True,
            "model_weights_loaded": False,
            "behavior_cases_scored": 0,
            "hidden_states": False,
            "causal_intervention": False,
            "cuda_required": False,
        },
        "interface": {
            "model": MODEL_NAME,
            "model_path": str(MODEL_PATH),
            "system_prompt": SYSTEM_PROMPT,
            "native_chat_template": True,
            "enable_thinking": False,
            "candidate_continuation": "one ASCII space followed by one lowercase candidate",
            "candidates": list(CANDIDATES),
            "candidate_token_count_required": 1,
            "candidate_ids_distinct": True,
            "candidate_token_absent_from_input": True,
            "slow_fast_tokenizer_ids_equal": True,
            "maximum_input_length": MAX_INPUT_LENGTH,
            "exact_manifest_input_only": True,
        },
        "future_execution": {
            "phase": 1231,
            "precision": "FP16 weights and FP16 forward path; no quantization",
            "device": "CUDA",
            "one_model_per_process": True,
            "batching": "exact input-length buckets with frozen batch size 16",
            "adaptive_batch_fallback": False,
            "dropout": "model.eval()",
            "inference_mode": True,
            "generation": False,
            "hidden_states": False,
            "attentions": False,
            "score": "FP32 log_softmax of the final-position logits at each frozen one-token candidate ID",
            "prediction": "argmax over four candidate scores",
            "unconstrained_first_token": "also record the full-vocabulary argmax; do not use it to redefine candidate prediction",
            "finite_case": "all final-position vocabulary logits and all candidate scores are finite",
            "tie_rule": f"UNRESOLVED_TIE when top two candidate scores differ by <= {TIE_TOLERANCE}",
            "ties_and_nonfinite": "retain in every denominator and count as incorrect",
        },
        "typed_behavior_ledgers": {
            "Q0_numerical": {
                "finite_rate_min": FINITE_RATE_MIN,
                "scope": "overall and every split",
            },
            "Q1_candidate_semantics": {
                "panel_accuracy_min": PANEL_ACCURACY_MIN,
                "scope": "every split x panel",
                "active_worst_one_factor_marginal_min": ACTIVE_WORST_MARGINAL_MIN,
                "marginal_axes": list(HIDDEN_BEHAVIOR_AXES),
            },
            "Q2_active_counterfactual": {
                "quartet_success_min": ACTIVE_QUARTET_MIN,
                "success": "all four binding states finite and correct with predictions covering all four compass answers",
                "scope": "every split",
            },
            "Q3_controls": {
                "invariant_bundle_success_min": CONTROL_INVARIANT_BUNDLE_MIN,
                "panels": ["matched_null", "surface_order"],
                "success": "all four states finite and correct with one invariant predicted answer",
                "scope": "every split x control panel",
            },
            "Q4_interface_invariance": {
                "template_pair_success_min": TEMPLATE_PAIR_MIN,
                "success": "paired template variants are finite, correct, and predict the same answer",
                "scope": "every split x panel",
            },
            "Q5_natural_first_token": {
                "full_vocab_top1_equals_gold_candidate_id_min": NATURAL_FIRST_TOKEN_MIN,
                "scope": "natural_use split separately",
                "claim": "first-token interface use only; not open-ended generation",
            },
        },
        "typed_authorization_rule": {
            "candidate_hidden_eligibility": "Q0 and Q1 and Q2 and Q3 and Q4",
            "natural_first_token_claim": "candidate_hidden_eligibility and Q5",
            "Q5_does_not_veto": "a candidate-scored Qwen3 hidden-state study if Q0-Q4 pass",
            "failure_scope": "a failed ledger denies only the claim that ledger identifies",
        },
        "future_raw_case_schema": [
            "item_id",
            "all_vocab_logits_finite",
            "candidate_scores",
            "prediction",
            "gold_candidate",
            "correct",
            "gold_margin",
            "full_vocab_top1_id",
            "full_vocab_top1_text",
            "full_vocab_top1_is_gold_candidate",
            "input_length",
            "runtime_batch_size",
        ],
        "future_group_keys": {
            "quartet": "bundle_id and panel",
            "template_pair": (
                "split, panel, world_id, target_entity, order_variant, mapping_variant, "
                "binding_state, with template_id omitted"
            ),
        },
        "claim_boundary": [
            "This phase loads a tokenizer but no model weights.",
            "No behavior, hidden state, causal necessity, or natural language mechanism is measured.",
            "The clock-to-compass ontology is familiar parametric knowledge; the new object binding is contextual.",
            "Candidate scoring is not equivalent to unconstrained multi-token generation.",
            "A future Qwen3 result remains model- and interface-specific.",
        ],
        "forbidden": [
            "load AutoModelForCausalLM or any model weights",
            "run CUDA",
            "score any behavior outcome",
            "change prompts, candidates, thresholds, ledgers, or tie handling after materialization",
            "drop any split, panel, direction, template, world, state, order, or mapping variant",
            "merge the five typed ledgers into an untyped post-hoc pass",
            "scan hidden states before a separately audited behavior execution",
        ],
        "authorization": {
            "pass_authorizes": "a separately named Phase1231 Qwen3 FP16 behavior execution using only the frozen manifest",
            "automatic_model_execution": False,
            "hidden_scan": False,
            "causal_intervention": False,
        },
    }
    contract["contract_digest"] = digest(contract)
    return contract


def verify_frozen() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    if contract.get("contract_digest") != digest(strip_digest(contract, "contract_digest")):
        raise RuntimeError("Phase1230 contract digest mismatch")
    if contract.get("source_hashes") != source_hashes():
        raise RuntimeError("Phase1230 source changed after preregistration")
    if contract["upstream"]["file_hashes"] != upstream_hashes():
        raise RuntimeError("Phase1229 upstream changed after preregistration")
    return contract


def render_native(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def continuation_ids(tokenizer: Any, rendered: str, candidate: str) -> list[int]:
    base = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
    extended = [
        int(value)
        for value in tokenizer.encode(rendered + " " + candidate, add_special_tokens=False)
    ]
    if extended[: len(base)] != base:
        raise RuntimeError(f"candidate {candidate!r} retokenized the prompt")
    suffix = extended[len(base) :]
    if not suffix:
        raise RuntimeError(f"candidate {candidate!r} has an empty continuation")
    return suffix


def token_span_for_chars(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> list[int]:
    if char_end <= char_start:
        raise ValueError("zero-length character spans require a typed boundary rule")
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and end > char_start and start < char_end
    ]
    if not indices:
        raise RuntimeError(f"no tokens overlap character span {char_start}:{char_end}")
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise RuntimeError("character span maps to non-contiguous tokens")
    if offsets[indices[0]][0] > char_start or offsets[indices[-1]][1] < char_end:
        raise RuntimeError("token span does not cover character span")
    return [indices[0], indices[-1] + 1]


def role_token_spans(
    row: dict[str, Any], rendered: str, offsets: list[tuple[int, int]], input_length: int
) -> tuple[dict[str, list[list[int]]], int]:
    prompt = row["prompt"]
    prompt_start = rendered.find(prompt)
    if prompt_start < 0 or rendered.find(prompt, prompt_start + 1) >= 0:
        raise RuntimeError("raw prompt is not uniquely embedded in native chat rendering")
    output: dict[str, list[list[int]]] = {}
    for role, spans in row["spans"].items():
        if role == "answer_boundary":
            output[role] = [[input_length - 1, input_length]]
            continue
        output[role] = [
            token_span_for_chars(offsets, prompt_start + int(start), prompt_start + int(end))
            for start, end in spans
        ]
    return output, prompt_start


def build_manifest(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from transformers import AutoTokenizer, __version__ as transformers_version

    slow = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    fast = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )
    if not getattr(fast, "is_fast", False):
        raise RuntimeError("Qwen3 fast tokenizer is unavailable; typed offset spans cannot be frozen")

    manifest: list[dict[str, Any]] = []
    candidate_registry: dict[str, list[int]] | None = None
    slow_fast_mismatches = 0
    role_span_failures = 0
    candidate_input_overlap = 0
    prompt_embedding_failures = 0

    for execution_index, row in enumerate(rows):
        rendered = render_native(slow, str(row["prompt"]))
        slow_ids = [int(value) for value in slow.encode(rendered, add_special_tokens=False)]
        fast_encoded = fast(rendered, add_special_tokens=False, return_offsets_mapping=True)
        fast_ids = [int(value) for value in fast_encoded["input_ids"]]
        offsets = [(int(start), int(end)) for start, end in fast_encoded["offset_mapping"]]
        if slow_ids != fast_ids:
            slow_fast_mismatches += 1
        candidate_token_ids = {
            candidate: continuation_ids(slow, rendered, candidate) for candidate in CANDIDATES
        }
        if candidate_registry is None:
            candidate_registry = candidate_token_ids
        if candidate_token_ids != candidate_registry:
            raise RuntimeError("candidate continuation IDs depend on the prompt")
        singleton_ids = [values[0] for values in candidate_token_ids.values() if len(values) == 1]
        if any(token_id in slow_ids for token_id in singleton_ids):
            candidate_input_overlap += 1
        try:
            token_roles, prompt_start = role_token_spans(row, rendered, offsets, len(slow_ids))
        except RuntimeError:
            role_span_failures += 1
            token_roles = {}
            prompt_start = -1
        if prompt_start < 0:
            prompt_embedding_failures += 1
        manifest_row: dict[str, Any] = {
            "schema_version": "phase1230.qwen3.behavior_case.v1",
            "execution_index": execution_index,
            "model": MODEL_NAME,
            "item_id": row["item_id"],
            "phase1229_row_digest": row["row_digest"],
            "input_ids": slow_ids,
            "input_ids_digest": digest(slow_ids),
            "input_id_multiset_digest": digest(sorted(slow_ids)),
            "input_length": len(slow_ids),
            "candidate_labels": list(CANDIDATES),
            "candidate_token_ids": candidate_token_ids,
            "gold_candidate": row["gold_candidate"],
            "gold_candidate_token_id": candidate_token_ids[row["gold_candidate"]][0],
            "prediction_token_index": len(slow_ids) - 1,
            "native_prompt_digest": digest(rendered),
            "prompt_char_start_in_native": prompt_start,
            "role_token_spans": token_roles,
            "split": row["split"],
            "panel": row["panel"],
            "bundle_id": row["bundle_id"],
            "world_id": row["world_id"],
            "template_id": row["template_id"],
            "target_entity": row["target_entity"],
            "binding_state": row["binding_state"],
            "order_variant": row["order_variant"],
            "mapping_variant": row["mapping_variant"],
            "target_record_position": row["target_record_position"],
        }
        manifest_row["manifest_row_digest"] = digest(manifest_row)
        manifest.append(manifest_row)

    if candidate_registry is None:
        raise RuntimeError("empty Phase1229 material")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in manifest:
        groups[case["bundle_id"]].append(case)
    active_groups = [group for group in groups.values() if group[0]["panel"] == "active"]
    all_group_length_match = [len({case["input_length"] for case in group}) == 1 for group in groups.values()]
    all_group_multiset_match = [
        len({case["input_id_multiset_digest"] for case in group}) == 1 for group in groups.values()
    ]
    active_length_match = [
        len({case["input_length"] for case in group}) == 1 for group in active_groups
    ]
    active_multiset_match = [
        len({case["input_id_multiset_digest"] for case in group}) == 1
        for group in active_groups
    ]
    lengths = [case["input_length"] for case in manifest]
    candidate_lengths = [len(ids) for ids in candidate_registry.values()]
    candidate_ids = [ids[0] for ids in candidate_registry.values() if len(ids) == 1]
    summary: dict[str, Any] = {
        "phase": PHASE,
        "kind": "qwen3_tokenizer_only_native_interface",
        "created_at_utc": utc_now(),
        "model": MODEL_NAME,
        "model_path": str(MODEL_PATH),
        "transformers_version": transformers_version,
        "slow_tokenizer_class": type(slow).__name__,
        "fast_tokenizer_class": type(fast).__name__,
        "model_weights_loaded": False,
        "cuda_used": False,
        "behavior_cases_scored": 0,
        "row_count": len(manifest),
        "active_count": sum(case["panel"] == "active" for case in manifest),
        "split_counts": dict(Counter(case["split"] for case in manifest)),
        "panel_counts": dict(Counter(case["panel"] for case in manifest)),
        "candidate_token_ids": candidate_registry,
        "candidate_token_count_min": min(candidate_lengths),
        "candidate_token_count_max": max(candidate_lengths),
        "candidate_ids_distinct": len(candidate_ids) == len(CANDIDATES) and len(set(candidate_ids)) == len(CANDIDATES),
        "candidate_input_overlap_count": candidate_input_overlap,
        "slow_fast_id_mismatch_count": slow_fast_mismatches,
        "role_span_failure_count": role_span_failures,
        "prompt_embedding_failure_count": prompt_embedding_failures,
        "input_length_min": min(lengths),
        "input_length_max": max(lengths),
        "input_length_mean": sum(lengths) / len(lengths),
        "input_length_bucket_count": len(set(lengths)),
        "bundle_count": len(groups),
        "active_bundle_count": len(active_groups),
        "active_length_match_rate": sum(active_length_match) / len(active_length_match),
        "active_input_id_multiset_match_rate": sum(active_multiset_match) / len(active_multiset_match),
        "all_bundle_length_match_rate": sum(all_group_length_match) / len(all_group_length_match),
        "all_bundle_input_id_multiset_match_rate": sum(all_group_multiset_match) / len(all_group_multiset_match),
        "manifest_digest": digest(manifest),
    }
    summary["interface_gate"] = bool(
        summary["row_count"] == EXPECTED_ROWS
        and summary["active_count"] == EXPECTED_ACTIVE
        and summary["candidate_token_count_min"] == 1
        and summary["candidate_token_count_max"] == 1
        and summary["candidate_ids_distinct"]
        and summary["candidate_input_overlap_count"] == 0
        and summary["slow_fast_id_mismatch_count"] == 0
        and summary["role_span_failure_count"] == 0
        and summary["prompt_embedding_failure_count"] == 0
        and summary["input_length_max"] <= MAX_INPUT_LENGTH
        and summary["active_length_match_rate"] == 1.0
        and summary["active_input_id_multiset_match_rate"] == 1.0
    )
    summary["tokenizer_audit_digest"] = digest(summary)
    return manifest, summary


def preregister() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError("Phase1230 output directory already exists")
    contract = build_contract()
    write_json(CONTRACT_PATH, contract)
    print(canonical_json({"status": "preregistered", "contract_digest": contract["contract_digest"]}))


def materialize() -> None:
    contract = verify_frozen()
    preaudit = read_json(PREAUDIT_PATH)
    if preaudit.get("all_checks_passed") is not True:
        raise RuntimeError("independent preaudit did not pass")
    if MANIFEST_PATH.exists() or TOKEN_AUDIT_PATH.exists():
        raise RuntimeError("Phase1230 interface outputs already exist")
    rows = read_jsonl(UPSTREAM_MATERIAL)
    manifest, token_audit = build_manifest(rows)
    token_audit["contract_digest"] = contract["contract_digest"]
    token_audit["phase1229_material_sha256"] = file_sha256(UPSTREAM_MATERIAL)
    token_audit["tokenizer_audit_digest"] = digest(strip_digest(token_audit, "tokenizer_audit_digest"))
    write_jsonl(MANIFEST_PATH, manifest)
    write_json(TOKEN_AUDIT_PATH, token_audit)
    print(
        canonical_json(
            {
                "status": "tokenizer_interface_materialized_pending_independent_audit",
                "rows": len(manifest),
                "interface_gate": token_audit["interface_gate"],
            }
        )
    )


def finalize() -> None:
    contract = verify_frozen()
    preaudit = read_json(PREAUDIT_PATH)
    token_audit = read_json(TOKEN_AUDIT_PATH)
    result_audit = read_json(RESULT_AUDIT_PATH)
    if preaudit.get("all_checks_passed") is not True:
        raise RuntimeError("independent preaudit failed")
    if token_audit.get("interface_gate") is not True:
        raise RuntimeError("tokenizer interface gate failed")
    if result_audit.get("all_checks_passed") is not True:
        raise RuntimeError("independent result audit failed")
    final: dict[str, Any] = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "status": "qwen3_behavior_protocol_ready_not_executed",
        "contract_digest": contract["contract_digest"],
        "tokenizer_audit_digest": token_audit["tokenizer_audit_digest"],
        "independent_preaudit_digest": preaudit["audit_digest"],
        "independent_result_audit_digest": result_audit["audit_digest"],
        "manifest_digest": token_audit["manifest_digest"],
        "result": {
            "interface_gate": True,
            "row_count": token_audit["row_count"],
            "candidate_token_ids": token_audit["candidate_token_ids"],
            "candidate_input_overlap_count": token_audit["candidate_input_overlap_count"],
            "input_length_range": [token_audit["input_length_min"], token_audit["input_length_max"]],
            "active_length_match_rate": token_audit["active_length_match_rate"],
            "active_input_id_multiset_match_rate": token_audit["active_input_id_multiset_match_rate"],
        },
        "k_ledger": {
            "new_item": None,
            "reason": "Tokenizer/interface qualification is a protocol asset, not behavior or mechanism evidence.",
        },
        "claim_boundary": contract["claim_boundary"],
        "authorization": {
            "next_experiment": "Phase1231 Qwen3 FP16 behavior execution from the frozen manifest",
            "automatic_execution": False,
            "auto_continue": 0,
            "hidden_scan": False,
            "causal_intervention": False,
            "reason": "Model execution is a separately named evidence phase; this tokenizer-only phase stops before loading weights.",
        },
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": final["status"], "final_digest": final["final_digest"], "auto_continue": 0}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("preregister", "materialize", "finalize"))
    stage = parser.parse_args().stage
    {"preregister": preregister, "materialize": materialize, "finalize": finalize}[stage]()


if __name__ == "__main__":
    main()
