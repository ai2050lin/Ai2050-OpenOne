#!/usr/bin/env python3
"""Freeze the Phase1203 object-attribute behavior qualification protocol.

The phase is zero-model by construction. It seals scoring, numerical gates,
behavior ledgers, native chat interfaces, and exact tokenizer input manifests.
No model weights are loaded and no behavior outcome is observed here.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

PHASE = 1203
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1203_object_attribute_behavior_protocol_audit.py"
PHASE1202_ROOT = TEST_ROOT / "result/phase1202_object_attribute_mother_contract"
UPSTREAM_FINAL = PHASE1202_ROOT / "analysis/final.json"
UPSTREAM_CONTRACT = PHASE1202_ROOT / "protocol/mother_family_contract.json"
UPSTREAM_PACKAGE = PHASE1202_ROOT / "material/object_attribute_binding.jsonl"
UPSTREAM_TOKEN_AUDIT = PHASE1202_ROOT / "audit/tokenizer_audit.json"
UPSTREAM_AUDIT = PHASE1202_ROOT / "audit/independent_audit.json"
UPSTREAM_SUMMARY = PHASE1202_ROOT / "analysis/readiness_summary.json"

OUT_ROOT = TEST_ROOT / "result/phase1203_object_attribute_behavior_protocol"
PROTOCOL_PATH = OUT_ROOT / "protocol/behavior_protocol.json"
MANIFEST_DIR = OUT_ROOT / "protocol/model_manifests"
INTERFACE_AUDIT_PATH = OUT_ROOT / "audit/interface_tokenizer_audit.json"
SUMMARY_PATH = OUT_ROOT / "analysis/protocol_summary.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_protocol_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

EXPECTED_PHASE1202_FINAL_DIGEST = (
    "c1d8a986074bc87daa3301d3048936104342f4b56a9f663890958eef1b62e2b5"
)
EXPECTED_PACKAGE_DIGEST = (
    "b6e52c726a010d721a131a9de9148fa5b8fc30e10d1eeef5149641b60dfdb5cc"
)
MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
MODEL_BATCH_SIZE = {"qwen3": 16, "glm4": 2, "deepseek7b": 4}
SYSTEM_PROMPT = (
    "Use only the supplied profiles. Return exactly one allowed name and no explanation."
)

FINITE_RATE_MIN = 0.99
OVERALL_ACCURACY_MIN = 0.90
WORST_MARGINAL_CELL_MIN = 0.85
PANEL_PAIR_MIN = 0.85
CANDIDATE_ORDER_INVARIANCE_MIN = 0.85
TEMPLATE_INVARIANCE_MIN = 0.85
TIE_TOLERANCE = 1e-7
MIN_CROSS_MODEL_PASSES = 2
SPLITS = ("discovery", "confirmation", "unseen_composition")
PANELS = ("active", "matched_null", "surface_only", "semantic_neighbor")
WORST_CELL_AXES = ("attribute", "world", "template", "gold_position", "panel")
EXPECTED_CASES_PER_MODEL = 4608


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


def manifest_path(model_name: str) -> Path:
    return MANIFEST_DIR / f"{model_name}.jsonl"


def source_hashes() -> dict[str, str]:
    return {
        "protocol_compiler": file_sha256(SCRIPT),
        "independent_audit": file_sha256(AUDIT_SCRIPT),
    }


def upstream_hashes() -> dict[str, str]:
    upstream = read_json(UPSTREAM_FINAL)
    if upstream.get("final_digest") != EXPECTED_PHASE1202_FINAL_DIGEST:
        raise RuntimeError("Phase1202 final digest mismatch")
    summary = read_json(UPSTREAM_SUMMARY)
    if summary.get("package_digest") != EXPECTED_PACKAGE_DIGEST:
        raise RuntimeError("Phase1202 package digest mismatch")
    return {
        "phase1202_final": file_sha256(UPSTREAM_FINAL),
        "phase1202_contract": file_sha256(UPSTREAM_CONTRACT),
        "phase1202_package": file_sha256(UPSTREAM_PACKAGE),
        "phase1202_tokenizer_audit": file_sha256(UPSTREAM_TOKEN_AUDIT),
        "phase1202_independent_audit": file_sha256(UPSTREAM_AUDIT),
        "phase1202_summary": file_sha256(UPSTREAM_SUMMARY),
    }


def render_native(tokenizer: Any, model_name: str, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if model_name == "qwen3":
        kwargs["enable_thinking"] = False
    rendered = str(tokenizer.apply_chat_template(messages, **kwargs))
    if model_name == "deepseek7b" and rendered.endswith("<think>\n"):
        rendered += "</think>\n\n"
    return rendered


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


def build_protocol() -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1203.object_attribute.behavior_protocol.v1",
        "created_at": utc_now(),
        "purpose": "freeze behavior qualification before any Qwen3, GLM4, or DS7B score is observed",
        "source_hashes": source_hashes(),
        "upstream": {
            "phase1202_final_digest": EXPECTED_PHASE1202_FINAL_DIGEST,
            "phase1202_package_digest": EXPECTED_PACKAGE_DIGEST,
            "upstream_hashes": upstream_hashes(),
        },
        "interface": {
            "system_prompt": SYSTEM_PROMPT,
            "native_chat_template": True,
            "qwen3_enable_thinking": False,
            "deepseek7b_close_empty_thinking_prefix": True,
            "candidate_continuation": "one ASCII space followed by the candidate name",
            "model_input": "exact token IDs frozen in the Phase1203 model manifest; no runtime retokenization",
            "candidate_constraint": "all candidate continuations must contain exactly one token in every frozen manifest",
            "state_pair_tokenization_boundary": (
                "Phase1202 lexical token multisets are matched, but native-chat BPE lengths and token-ID multisets "
                "may differ because record order changes token boundaries; exact residual rates are recorded and must "
                "be treated as a carrier covariate in any later hidden-state study."
            ),
        },
        "candidate_scoring": {
            "formula": "mean_j log P(candidate_token_j | frozen_prompt, earlier_candidate_tokens)",
            "full_sequence_not_first_token_switching": True,
            "length_normalization": "arithmetic mean over candidate continuation token count",
            "current_manifest_token_count": 1,
            "raw_model_precision": "FP16 weights and FP16 forward path",
            "normalization_precision": "cast final-position logits to FP32, then apply log_softmax",
            "finite_case": "every final-position vocabulary logit and every candidate score is finite",
            "tie_rule": f"UNRESOLVED_TIE if the two largest candidate scores differ by <= {TIE_TOLERANCE}",
            "nonfinite_and_ties": "included in every denominator and counted as incorrect",
        },
        "execution": {
            "models": list(MODEL_ORDER),
            "model_order": list(MODEL_ORDER),
            "precision": "FP16",
            "quantization": "none",
            "cuda_required": True,
            "one_model_per_process": True,
            "release_before_next_model": True,
            "exact_input_length_bucketing": True,
            "fixed_batch_size": MODEL_BATCH_SIZE,
            "adaptive_oom_batch_fallback": False,
            "dropout": "model.eval()",
            "inference_mode": True,
            "save_hidden_states": False,
            "save_attentions": False,
            "generation": False,
        },
        "five_behavior_ledgers": {
            "L1_numerical": {
                "metric": "finite case rate",
                "threshold": FINITE_RATE_MIN,
                "scope": "overall and separately in every split",
            },
            "L2_identity": {
                "overall_accuracy_threshold": OVERALL_ACCURACY_MIN,
                "worst_marginal_cell_threshold": WORST_MARGINAL_CELL_MIN,
                "worst_cell_axes": list(WORST_CELL_AXES),
                "cell_definition": "one-factor marginal cells, not the sparse full Cartesian product",
                "scope": "each split separately",
            },
            "L3_panel_logic": {
                "pair_success_threshold": PANEL_PAIR_MIN,
                "active": "both states correct and predicted entity identity flips",
                "matched_null": "both states correct and predicted entity identity remains the anchor",
                "surface_only": "both states correct and predicted entity identity remains fixed",
                "semantic_neighbor": "both states correct and predicted entity identity remains fixed",
                "scope": "every panel in every split",
            },
            "L4_interface_invariance": {
                "candidate_order_triple_threshold": CANDIDATE_ORDER_INVARIANCE_MIN,
                "template_pair_threshold": TEMPLATE_INVARIANCE_MIN,
                "success": "all variants finite, correct, and predict the same entity identity",
                "scope": "each split separately",
            },
            "L5_unseen_composition": {
                "object": "held-out profile x attribute combinations",
                "overall_accuracy_threshold": OVERALL_ACCURACY_MIN,
                "worst_marginal_cell_threshold": WORST_MARGINAL_CELL_MIN,
                "all_other_ledgers_required": True,
                "not_claimed": "new lexical atoms, new attributes, or natural-use generalization",
            },
        },
        "model_pass_rule": (
            "A model passes only if all five ledgers pass in discovery, confirmation, and unseen_composition "
            "with no threshold refit and no template or attribute exclusion."
        ),
        "cross_model_rule": {
            "minimum_passing_models": MIN_CROSS_MODEL_PASSES,
            "two_or_more": "authorize a separately preregistered cross-model hidden-specificity phase",
            "exactly_one": "record only a model-specific behavioral domain; no cross-model hidden claim",
            "zero": "stop this interface; no hidden scan and no post-hoc template replacement",
        },
        "failure_scope": {
            "numerical_failure": "denies semantic judgment for that model under this FP16 interface",
            "worst_cell_failure": "denies whole-model authorization under this contract",
            "control_failure": "denies content-specific mechanism scanning",
            "unseen_composition_failure": "denies compositional mechanism claims but does not prove binding absence",
            "behavior_failure": "does not prove the internal mechanism is absent",
        },
        "forbidden_after_scoring": [
            "switch first-token versus sequence score",
            "switch length normalization",
            "change the system prompt or chat template",
            "remove a failing attribute, world, template, candidate position, panel, or split",
            "relax any frozen threshold",
            "scan hidden states for a failed model",
            "promote this controlled package to the natural-use U gate",
        ],
        "future_raw_case_schema": [
            "model", "item_id", "all_vocab_logits_finite", "candidate_scores",
            "prediction", "gold_candidate", "correct", "gold_margin",
            "runtime_batch_size", "input_length",
        ],
        "claim_boundary": {
            "this_phase_model_weights_loaded": False,
            "this_phase_behavior_cases_scored": 0,
            "this_phase_new_k_item": False,
            "this_phase_hidden_or_causal_evidence": False,
            "maximum_claim": "the behavior experiment is now executable without outcome-dependent scoring choices",
        },
    }
    protocol["protocol_digest"] = digest(protocol)
    return protocol


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    candidate = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    if digest(candidate) != protocol["protocol_digest"]:
        raise RuntimeError("protocol digest mismatch")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("source changed after preregistration")
    if protocol["upstream"]["upstream_hashes"] != upstream_hashes():
        raise RuntimeError("Phase1202 upstream changed after preregistration")
    return protocol


def build_model_manifest(model_name: str, package_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from model_utils import MODEL_CONFIGS
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[model_name]["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    manifest: list[dict[str, Any]] = []
    for execution_index, row in enumerate(package_rows):
        rendered = render_native(tokenizer, model_name, row["prompt"])
        input_ids = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
        candidate_token_ids = {
            candidate: continuation_ids(tokenizer, rendered, candidate)
            for candidate in row["candidates"]
        }
        if any(len(ids) != 1 for ids in candidate_token_ids.values()):
            raise RuntimeError(f"{model_name} has a multi-token candidate in {row['item_id']}")
        singleton_ids = [ids[0] for ids in candidate_token_ids.values()]
        if len(set(singleton_ids)) != len(singleton_ids):
            raise RuntimeError(f"{model_name} candidate collision in {row['item_id']}")
        manifest.append(
            {
                "schema_version": "phase1203.behavior_execution_case.v1",
                "execution_index": execution_index,
                "model": model_name,
                "item_id": row["item_id"],
                "phase1202_row_digest": digest(row),
                "input_ids": input_ids,
                "input_ids_digest": digest(input_ids),
                "input_id_multiset_digest": digest(sorted(input_ids)),
                "input_length": len(input_ids),
                "candidate_labels": list(row["candidates"]),
                "candidate_token_ids": candidate_token_ids,
                "gold_candidate": row["gold_candidate"],
                "split": row["split"],
                "panel": row["panel"],
                "world": row["world"],
                "attribute": row["attribute"],
                "template": row["template"],
                "gold_position": row["gold_position"],
                "candidate_order": row["candidate_order"],
                "binding_state": row["binding_state"],
                "combination_id": row["combination_id"],
            }
        )

    state_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for case in manifest:
        key = (
            case["combination_id"], case["panel"], case["template"], case["candidate_order"]
        )
        state_groups[key].append(case)
    state_groups_complete = all(
        len(group) == 2 and {case["binding_state"] for case in group} == {0, 1}
        for group in state_groups.values()
    )
    length_deltas = [
        abs(group[0]["input_length"] - group[1]["input_length"])
        for group in state_groups.values()
        if len(group) == 2
    ]
    id_multiset_matches = [
        len({group[0]["input_id_multiset_digest"], group[1]["input_id_multiset_digest"]}) == 1
        for group in state_groups.values()
        if len(group) == 2
    ]
    lengths = [case["input_length"] for case in manifest]
    summary = {
        "model": model_name,
        "tokenizer_path": str(MODEL_CONFIGS[model_name]["path"]),
        "case_count": len(manifest),
        "unique_item_count": len({case["item_id"] for case in manifest}),
        "minimum_input_length": min(lengths),
        "maximum_input_length": max(lengths),
        "mean_input_length": sum(lengths) / len(lengths),
        "length_bucket_count": len(set(lengths)),
        "all_candidates_single_token": all(
            len(ids) == 1
            for case in manifest
            for ids in case["candidate_token_ids"].values()
        ),
        "all_candidate_ids_distinct_within_case": all(
            len({ids[0] for ids in case["candidate_token_ids"].values()}) == 3
            for case in manifest
        ),
        "state_groups_complete": state_groups_complete,
        "state_input_length_match_rate": sum(delta == 0 for delta in length_deltas) / len(length_deltas),
        "maximum_state_input_length_delta": max(length_deltas),
        "state_input_id_multiset_match_rate": sum(id_multiset_matches) / len(id_multiset_matches),
        "manifest_digest": digest(manifest),
    }
    summary["pass"] = bool(
        summary["case_count"] == EXPECTED_CASES_PER_MODEL
        and summary["unique_item_count"] == EXPECTED_CASES_PER_MODEL
        and summary["maximum_input_length"] <= 256
        and summary["all_candidates_single_token"]
        and summary["all_candidate_ids_distinct_within_case"]
        and summary["state_groups_complete"]
    )
    return manifest, summary


def selftest() -> None:
    package_rows = read_jsonl(UPSTREAM_PACKAGE)
    if len(package_rows) != EXPECTED_CASES_PER_MODEL:
        raise AssertionError(len(package_rows))
    if digest(package_rows) != EXPECTED_PACKAGE_DIGEST:
        raise AssertionError("package digest")
    if set(Counter(row["split"] for row in package_rows)) != set(SPLITS):
        raise AssertionError("split levels")
    protocol = build_protocol()
    if protocol["candidate_scoring"]["full_sequence_not_first_token_switching"] is not True:
        raise AssertionError("scoring policy")
    print(canonical_json({"status": "selftest_pass", "cases": len(package_rows)}))


def preregister() -> None:
    if PROTOCOL_PATH.exists() or SUMMARY_PATH.exists():
        raise RuntimeError("Phase1203 protocol or outcomes already exist")
    upstream = read_json(UPSTREAM_FINAL)
    if not upstream["authorized_next"]["phase1203_behavior_protocol_preregistration"]:
        raise RuntimeError("Phase1202 did not authorize Phase1203 protocol preregistration")
    if upstream["authorized_next"]["automatic_model_execution"]:
        raise RuntimeError("Phase1202 unexpectedly authorized automatic execution")
    protocol = build_protocol()
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"protocol_digest": protocol["protocol_digest"]}))


def materialize() -> None:
    protocol = verify_protocol()
    if SUMMARY_PATH.exists() or INTERFACE_AUDIT_PATH.exists() or MANIFEST_DIR.exists():
        raise RuntimeError("Phase1203 interface outputs already exist")
    package_rows = read_jsonl(UPSTREAM_PACKAGE)
    if digest(package_rows) != EXPECTED_PACKAGE_DIGEST:
        raise RuntimeError("package content digest mismatch")
    model_summaries: dict[str, Any] = {}
    for model_name in MODEL_ORDER:
        manifest, model_summary = build_model_manifest(model_name, package_rows)
        write_jsonl(manifest_path(model_name), manifest)
        model_summaries[model_name] = model_summary
    interface_audit = {
        "phase": PHASE,
        "kind": "tokenizer_only_native_interface_materialization",
        "model_weights_loaded": False,
        "models": model_summaries,
        "overall_pass": all(summary["pass"] for summary in model_summaries.values()),
    }
    interface_audit["interface_audit_digest"] = digest(interface_audit)
    summary = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": "behavior_protocol_materialized" if interface_audit["overall_pass"] else "interface_materialization_failed",
        "protocol_digest": protocol["protocol_digest"],
        "phase1202_package_digest": EXPECTED_PACKAGE_DIGEST,
        "interface_audit_digest": interface_audit["interface_audit_digest"],
        "model_manifest_digests": {
            model: payload["manifest_digest"] for model, payload in model_summaries.items()
        },
        "model_weights_loaded": False,
        "behavior_cases_scored": 0,
        "new_k_item": None,
    }
    write_json(INTERFACE_AUDIT_PATH, interface_audit)
    write_json(SUMMARY_PATH, summary)
    print(canonical_json({"status": summary["status"], "models": model_summaries}))


def finalize() -> None:
    protocol = verify_protocol()
    summary = read_json(SUMMARY_PATH)
    interface_audit = read_json(INTERFACE_AUDIT_PATH)
    audit = read_json(AUDIT_PATH)
    if not interface_audit.get("overall_pass", False):
        raise RuntimeError("interface audit failed")
    if not audit.get("gate_pass", False):
        raise RuntimeError("independent protocol audit failed")
    final = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": "behavior_protocol_ready_not_executed",
        "protocol_digest": protocol["protocol_digest"],
        "interface_audit_digest": interface_audit["interface_audit_digest"],
        "independent_audit_digest": audit["audit_digest"],
        "model_manifest_digests": summary["model_manifest_digests"],
        "evidence_scope": {
            "kind": "zero-model behavior protocol preregistration",
            "model_weights_loaded": False,
            "behavior_cases_scored": 0,
            "new_k_item": None,
            "canonical_k_range": "K1-K183",
            "behavior_evidence": False,
            "hidden_state_evidence": False,
            "causal_evidence": False,
        },
        "authorized_next": {
            "phase1204_sequential_fp16_behavior_execution": True,
            "automatic_model_execution": False,
            "hidden_state_scan": False,
            "causal_intervention": False,
            "new_mechanism_algebra": False,
        },
        "stop_reason": (
            "Scoring and exact model inputs are now sealed. Phase1202 authorized preregistration only, "
            "so this phase stops before loading model weights."
        ),
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": final["status"], "authorized_next": final["authorized_next"], "final_digest": final["final_digest"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("selftest", "preregister", "materialize", "finalize"))
    command = parser.parse_args().command
    {
        "selftest": selftest,
        "preregister": preregister,
        "materialize": materialize,
        "finalize": finalize,
    }[command]()


if __name__ == "__main__":
    main()
