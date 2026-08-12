#!/usr/bin/env python3
"""Independent audit for the sealed Phase1203 behavior protocol."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

PHASE1202_ROOT = TEST_ROOT / "result/phase1202_object_attribute_mother_contract"
OUT_ROOT = TEST_ROOT / "result/phase1203_object_attribute_behavior_protocol"
PROTOCOL_PATH = OUT_ROOT / "protocol/behavior_protocol.json"
MANIFEST_DIR = OUT_ROOT / "protocol/model_manifests"
INTERFACE_AUDIT_PATH = OUT_ROOT / "audit/interface_tokenizer_audit.json"
SUMMARY_PATH = OUT_ROOT / "analysis/protocol_summary.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_protocol_audit.json"
PACKAGE_PATH = PHASE1202_ROOT / "material/object_attribute_binding.jsonl"
UPSTREAM_FINAL = PHASE1202_ROOT / "analysis/final.json"

EXPECTED_PHASE = 1203
EXPECTED_UPSTREAM_DIGEST = "c1d8a986074bc87daa3301d3048936104342f4b56a9f663890958eef1b62e2b5"
EXPECTED_PACKAGE_DIGEST = "b6e52c726a010d721a131a9de9148fa5b8fc30e10d1eeef5149641b60dfdb5cc"
MODELS = ("qwen3", "glm4", "deepseek7b")
EXPECTED_CASES = 4608
SYSTEM_PROMPT = "Use only the supplied profiles. Return exactly one allowed name and no explanation."
EXPECTED_THRESHOLDS = {
    "finite": 0.99,
    "overall": 0.90,
    "worst": 0.85,
    "panel": 0.85,
    "order": 0.85,
    "template": 0.85,
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


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


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
        return []
    return extended[len(base) :]


def audit(write: bool) -> dict[str, Any]:
    from model_utils import MODEL_CONFIGS
    from transformers import AutoTokenizer

    checks: list[dict[str, Any]] = []
    protocol = read_json(PROTOCOL_PATH)
    summary = read_json(SUMMARY_PATH)
    interface = read_json(INTERFACE_AUDIT_PATH)
    package_rows = read_jsonl(PACKAGE_PATH)
    package_by_id = {row["item_id"]: row for row in package_rows}
    upstream = read_json(UPSTREAM_FINAL)

    protocol_body = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    add(checks, "phase", protocol.get("phase") == EXPECTED_PHASE)
    add(checks, "protocol_digest", digest(protocol_body) == protocol.get("protocol_digest"))
    add(checks, "upstream_digest", upstream.get("final_digest") == EXPECTED_UPSTREAM_DIGEST)
    add(checks, "package_digest", digest(package_rows) == EXPECTED_PACKAGE_DIGEST)
    add(
        checks,
        "source_hashes",
        protocol["source_hashes"]
        == {
            "protocol_compiler": file_sha256(TEST_ROOT / "phase1203_object_attribute_behavior_protocol.py"),
            "independent_audit": file_sha256(Path(__file__).resolve()),
        },
    )
    add(checks, "zero_model_scope", protocol["claim_boundary"]["this_phase_model_weights_loaded"] is False)
    add(checks, "zero_behavior_cases", protocol["claim_boundary"]["this_phase_behavior_cases_scored"] == 0)
    add(checks, "no_k_item", protocol["claim_boundary"]["this_phase_new_k_item"] is False)
    add(checks, "no_hidden_or_causal", protocol["claim_boundary"]["this_phase_hidden_or_causal_evidence"] is False)

    scoring = protocol["candidate_scoring"]
    add(checks, "full_sequence_score", scoring["full_sequence_not_first_token_switching"] is True)
    add(checks, "length_normalization_frozen", scoring["length_normalization"] == "arithmetic mean over candidate continuation token count")
    add(checks, "fp16_forward", scoring["raw_model_precision"] == "FP16 weights and FP16 forward path")
    add(checks, "fp32_log_softmax", scoring["normalization_precision"] == "cast final-position logits to FP32, then apply log_softmax")
    add(checks, "ties_and_nonfinite_fail", scoring["nonfinite_and_ties"] == "included in every denominator and counted as incorrect")

    ledgers = protocol["five_behavior_ledgers"]
    add(checks, "finite_threshold", ledgers["L1_numerical"]["threshold"] == EXPECTED_THRESHOLDS["finite"])
    add(checks, "overall_threshold", ledgers["L2_identity"]["overall_accuracy_threshold"] == EXPECTED_THRESHOLDS["overall"])
    add(checks, "worst_threshold", ledgers["L2_identity"]["worst_marginal_cell_threshold"] == EXPECTED_THRESHOLDS["worst"])
    add(checks, "panel_threshold", ledgers["L3_panel_logic"]["pair_success_threshold"] == EXPECTED_THRESHOLDS["panel"])
    add(checks, "order_threshold", ledgers["L4_interface_invariance"]["candidate_order_triple_threshold"] == EXPECTED_THRESHOLDS["order"])
    add(checks, "template_threshold", ledgers["L4_interface_invariance"]["template_pair_threshold"] == EXPECTED_THRESHOLDS["template"])
    add(
        checks,
        "worst_axes",
        set(ledgers["L2_identity"]["worst_cell_axes"])
        == {"attribute", "world", "template", "gold_position", "panel"},
    )
    add(checks, "all_splits_required", "discovery, confirmation, and unseen_composition" in protocol["model_pass_rule"])
    add(checks, "cross_model_minimum", protocol["cross_model_rule"]["minimum_passing_models"] == 2)
    add(checks, "one_model_scope", "model-specific" in protocol["cross_model_rule"]["exactly_one"])
    add(checks, "no_adaptive_batch", protocol["execution"]["adaptive_oom_batch_fallback"] is False)
    add(checks, "sequential_models", protocol["execution"]["one_model_per_process"] is True and protocol["execution"]["release_before_next_model"] is True)
    add(checks, "no_quantization", protocol["execution"]["precision"] == "FP16" and protocol["execution"]["quantization"] == "none")
    add(checks, "no_hidden_save", protocol["execution"]["save_hidden_states"] is False and protocol["execution"]["save_attentions"] is False)
    add(checks, "forbidden_threshold_relaxation", "relax any frozen threshold" in protocol["forbidden_after_scoring"])
    add(checks, "natural_use_not_promoted", "promote this controlled package to the natural-use U gate" in protocol["forbidden_after_scoring"])

    add(checks, "interface_scope", interface.get("model_weights_loaded") is False)
    interface_body = {key: value for key, value in interface.items() if key != "interface_audit_digest"}
    add(checks, "interface_digest", digest(interface_body) == interface["interface_audit_digest"])
    add(checks, "interface_overall_pass", interface.get("overall_pass") is True)
    add(checks, "manifest_models", set(interface["models"]) == set(MODELS))

    recomputed_models: dict[str, Any] = {}
    for model_name in MODELS:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIGS[model_name]["path"],
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        manifest = read_jsonl(MANIFEST_DIR / f"{model_name}.jsonl")
        case_by_id = {case["item_id"]: case for case in manifest}
        add(checks, f"{model_name}_case_count", len(manifest) == EXPECTED_CASES, len(manifest))
        add(checks, f"{model_name}_unique_cases", len(case_by_id) == EXPECTED_CASES)
        add(checks, f"{model_name}_package_coverage", set(case_by_id) == set(package_by_id))
        exact_inputs = True
        exact_candidates = True
        exact_factors = True
        for item_id, row in package_by_id.items():
            case = case_by_id[item_id]
            rendered = render_native(tokenizer, model_name, row["prompt"])
            input_ids = [int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)]
            exact_inputs &= case["input_ids"] == input_ids
            exact_inputs &= case["input_ids_digest"] == digest(input_ids)
            exact_inputs &= case["input_length"] == len(input_ids)
            own_candidate_ids = {
                candidate: continuation_ids(tokenizer, rendered, candidate)
                for candidate in row["candidates"]
            }
            exact_candidates &= case["candidate_token_ids"] == own_candidate_ids
            exact_candidates &= all(len(ids) == 1 for ids in own_candidate_ids.values())
            exact_candidates &= len({ids[0] for ids in own_candidate_ids.values()}) == 3
            exact_factors &= all(
                case[field] == row[field]
                for field in (
                    "split", "panel", "world", "attribute", "template", "gold_position",
                    "candidate_order", "binding_state", "combination_id", "gold_candidate",
                )
            )
            exact_factors &= case["candidate_labels"] == row["candidates"]
            exact_factors &= case["phase1202_row_digest"] == digest(row)
        add(checks, f"{model_name}_exact_inputs", exact_inputs)
        add(checks, f"{model_name}_exact_candidates", exact_candidates)
        add(checks, f"{model_name}_exact_factors", exact_factors)

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
        add(checks, f"{model_name}_state_groups_complete", state_groups_complete)
        add(
            checks,
            f"{model_name}_state_tokenization_residue_recorded",
            abs(
                interface["models"][model_name]["state_input_length_match_rate"]
                - sum(delta == 0 for delta in length_deltas) / len(length_deltas)
            ) <= 1e-12
            and interface["models"][model_name]["maximum_state_input_length_delta"]
            == max(length_deltas)
            and abs(
                interface["models"][model_name]["state_input_id_multiset_match_rate"]
                - sum(id_multiset_matches) / len(id_multiset_matches)
            ) <= 1e-12,
            {
                "length_match_rate": sum(delta == 0 for delta in length_deltas) / len(length_deltas),
                "maximum_length_delta": max(length_deltas),
                "id_multiset_match_rate": sum(id_multiset_matches) / len(id_multiset_matches),
            },
        )
        lengths = [case["input_length"] for case in manifest]
        own_summary = {
            "case_count": len(manifest),
            "unique_item_count": len(case_by_id),
            "minimum_input_length": min(lengths),
            "maximum_input_length": max(lengths),
            "mean_input_length": sum(lengths) / len(lengths),
            "length_bucket_count": len(set(lengths)),
            "manifest_digest": digest(manifest),
        }
        frozen = interface["models"][model_name]
        summary_match = all(
            abs(float(own_summary[key]) - float(frozen[key])) <= 1e-12
            if isinstance(own_summary[key], float)
            else own_summary[key] == frozen[key]
            for key in own_summary
        )
        add(checks, f"{model_name}_summary_recompute", summary_match)
        recomputed_models[model_name] = own_summary

    add(checks, "summary_protocol_link", summary["protocol_digest"] == protocol["protocol_digest"])
    add(checks, "summary_interface_link", summary["interface_audit_digest"] == interface["interface_audit_digest"])
    add(checks, "summary_no_model", summary["model_weights_loaded"] is False)
    add(checks, "summary_no_behavior", summary["behavior_cases_scored"] == 0)
    add(checks, "summary_no_k", summary["new_k_item"] is None)
    add(
        checks,
        "summary_manifest_links",
        summary["model_manifest_digests"]
        == {model: recomputed_models[model]["manifest_digest"] for model in MODELS},
    )

    gate = all(check["pass"] for check in checks)
    output = {
        "phase": EXPECTED_PHASE,
        "kind": "independent_zero_model_behavior_protocol_audit",
        "gate_pass": gate,
        "checks_passed": sum(check["pass"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "recomputed_models": recomputed_models,
    }
    output["audit_digest"] = digest(output)
    if write:
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
