#!/usr/bin/env python3
"""CPU-only preregistration for Phase 979 external boundary diagnostics.

The protocol seals two separate diagnostic tracks before any Phase 979 model
forward pass:

1. a 128-item, 4-control x 2-decoding x 2-stream natural-rollout factorial;
2. a deterministic prompt-relative-truth x punctuation output-head cross.

Neither track may open the Phase 977 holdout or authorize internal mechanism
work.  The truth replication block is source-committed and pre-audited here,
but may receive model evaluation only after the frozen development gate passes.
It is therefore a conditionally executed replication block, not an
analyst-blind holdout.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

import phase979_boundary_core as core  # noqa: E402
import phase979_diagnostic_dataset as natural_data  # noqa: E402
import phase979_truth_punctuation_dataset as truth_data  # noqa: E402
from model_utils import MODEL_CONFIGS  # noqa: E402


PHASE = 979
SCHEMA_VERSION = 1
EXPERIMENT = "three_boundary_factorial_and_truth_punctuation"
MODEL_NAME = "qwen3"

OUT = ROOT / "tests" / "glm5" / "result" / "phase979_three_boundary_factorial"
PROTOCOL_PATH = OUT / "protocol_preregistration.json"

PHASE978_DIR = ROOT / "tests" / "glm5" / "result" / "phase978_legal_budget_stabilization"
PHASE978_PROTOCOL = PHASE978_DIR / "protocol_preregistration.json"
PHASE978_ADMISSION = PHASE978_DIR / "admission_development.json"
PHASE978_POSTMORTEM = PHASE978_DIR / "postmortem_development.json"
PHASE978_OPEN_RECEIPT = PHASE978_DIR / "holdout_open_receipt.json"

SCRIPT_PATHS = {
    "protocol": GLM5 / "phase979_protocol.py",
    "boundary_core": GLM5 / "phase979_boundary_core.py",
    "natural_dataset": GLM5 / "phase979_diagnostic_dataset.py",
    "truth_dataset": GLM5 / "phase979_truth_punctuation_dataset.py",
    "natural_runner": GLM5 / "phase979_natural_runner.py",
    "natural_auditor": GLM5 / "phase979_natural_audit.py",
    "truth_runner": GLM5 / "phase979_truth_punctuation.py",
    "truth_auditor": GLM5 / "phase979_truth_audit.py",
}

EXPECTED_NATURAL_ROWS = 128 * 4 * 2 * 2
EXPECTED_TRUTH_ROWS_PER_SPLIT = 64 * 2 * 2 * 2

TRUTH_THRESHOLDS = {
    "development_and_replication_each": {
        "truth_effect_D0_and_D1": {
            "mean_min_logits": 2.0,
            "positive_pairs_min": 48,
            "pair_denominator": 64,
            "tasks_meeting_positive_6_of_8_min": 6,
            "any_task_mean_failure_at_or_below": -2.0,
        },
        "punctuation_effect_QC_and_QW": {
            "mean_max_logits": -2.0,
            "negative_pairs_min": 48,
            "pair_denominator": 64,
            "tasks_with_negative_mean_min": 6,
        },
    },
    "replication_combined_confirmation": {
        "truth_effect_D0_and_D1_positive_pairs_min": 96,
        "pair_denominator": 128,
        "tasks_meeting_positive_12_of_16_min": 6,
    },
}

NATURAL_SCREEN_THRESHOLDS = {
    "denominator_per_cell_stream": 128,
    "valid_stop_net_improvement_min": 13,
    "censored_net_reduction_min": 13,
    "eos_invalid_increase_max": 6,
    "tasks_with_valid_stop_improvement_min": 6,
    "must_pass_both_frozen_streams": True,
    "interpretation": (
        "A passing contrast is only a candidate for a fresh independent "
        "confirmation; it cannot open the old holdout or authorize a mechanism scan."
    ),
}


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_full": sys.version,
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "version_source": "installed_distribution_metadata_only",
    }


def assert_no_old_holdout_import() -> None:
    loaded = [name for name in sys.modules
              if name == "phase977_holdout_dataset"
              or name.endswith(".phase977_holdout_dataset")]
    core.require(not loaded, f"old sealed holdout module imported: {loaded}")


def verify_self_hash(
    document: dict[str, Any], hash_field: str, time_field: str, label: str,
) -> None:
    payload = core.without_fields(document, hash_field, time_field)
    core.require(document.get(hash_field) == core.sha256_json(payload),
                 f"{label} self-hash invalid")


def authenticate_phase978() -> tuple[dict[str, Any], dict[str, Any]]:
    assert_no_old_holdout_import()
    core.require(not PHASE978_OPEN_RECEIPT.exists(),
                 "Phase978 holdout OPEN receipt exists; Phase979 must stop")
    old_protocol = core.load_json(PHASE978_PROTOCOL, "Phase978 protocol")
    admission = core.load_json(PHASE978_ADMISSION, "Phase978 development admission")
    postmortem = core.load_json(PHASE978_POSTMORTEM, "Phase978 postmortem")
    verify_self_hash(old_protocol, "protocol_sha256", "created_at_utc",
                     "Phase978 protocol")
    verify_self_hash(admission, "admission_sha256", "audited_at_utc",
                     "Phase978 admission")
    verify_self_hash(postmortem, "postmortem_sha256", "generated_at_utc",
                     "Phase978 postmortem")
    core.require(old_protocol.get("phase") == 978, "wrong Phase978 protocol phase")
    core.require(admission.get("phase") == 978, "wrong Phase978 admission phase")
    core.require(admission.get("protocol_sha256") == old_protocol["protocol_sha256"],
                 "Phase978 admission/protocol mismatch")
    core.require(admission.get("decision_gate", {}).get("passed") is False,
                 "Phase978 development decision is not NO-GO")
    core.require(admission.get("holdout_authorized") is False
                 and admission.get("holdout_loaded") is False,
                 "Phase978 admission crossed holdout boundary")
    frozen = postmortem.get("frozen_decision", {})
    core.require(frozen.get("passed") is False
                 and frozen.get("holdout_authorized") is False
                 and frozen.get("holdout_loaded") is False
                 and frozen.get("mechanism_authorized") is False
                 and frozen.get("decision_unchanged") is True,
                 "Phase978 postmortem did not preserve NO-GO")
    commitments = {
        "development_gate_passed": False,
        "holdout_authorized": False,
        "holdout_loaded": False,
        "mechanism_authorized": False,
        "open_receipt_exists": False,
        "protocol": {
            "path": relative(PHASE978_PROTOCOL),
            "sha256": core.sha256_file(PHASE978_PROTOCOL),
            "protocol_sha256": old_protocol["protocol_sha256"],
        },
        "development_admission": {
            "path": relative(PHASE978_ADMISSION),
            "sha256": core.sha256_file(PHASE978_ADMISSION),
            "admission_sha256": admission["admission_sha256"],
        },
        "development_postmortem": {
            "path": relative(PHASE978_POSTMORTEM),
            "sha256": core.sha256_file(PHASE978_POSTMORTEM),
            "postmortem_sha256": postmortem["postmortem_sha256"],
        },
    }
    assert_no_old_holdout_import()
    return commitments, old_protocol


def verify_model_identity(old_protocol: dict[str, Any]) -> dict[str, Any]:
    identity = old_protocol.get("local_model_artifact_identity")
    core.require(isinstance(identity, dict), "Phase978 protocol lacks model identity")
    model_root = ROOT / str(identity.get("path", ""))
    configured = Path(MODEL_CONFIGS[MODEL_NAME]["path"]).resolve()
    core.require(model_root.resolve() == configured and model_root.is_dir(),
                 "Qwen3 model path changed")
    files = identity.get("files")
    core.require(isinstance(files, dict) and files, "model identity lacks files")
    for name, expected in files.items():
        path = model_root / str(name)
        core.require(path.is_file(), f"missing model artifact: {name}")
        core.require(path.stat().st_size == int(expected["bytes"]),
                     f"model artifact size changed: {name}")
        core.require(core.sha256_file(path) == str(expected["sha256"]),
                     f"model artifact changed: {name}")
    return json.loads(json.dumps(identity))


def audit_datasets() -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    natural_items = natural_data.build_items()
    natural_audit = natural_data.audit_items(natural_items)
    core.require(natural_audit.get("passed") is True
                 and natural_audit.get("holdout_accessed") is False,
                 "natural diagnostic dataset audit failed")
    core.require(len(natural_items) == 128, "natural diagnostic is not 128 items")

    truth_by_split = {
        split: truth_data.build_pairs(split) for split in truth_data.SPLITS
    }
    for split, pairs in truth_by_split.items():
        audit = truth_data.audit_pairs(pairs)
        core.require(audit.get("passed") is True and len(pairs) == 64,
                     f"truth dataset audit failed: {split}")
        core.require(audit.get("split_counts") == {split: 64},
                     f"truth split leakage: {split}")
    combined = truth_data.audit_pairs(
        truth_by_split["development"] + truth_by_split["replication"])
    core.require(combined.get("passed") is True and combined.get("n_pairs") == 128,
                 "combined truth dataset audit failed")
    core.require(truth_data.dataset_identity() == truth_data.STABLE_IDENTITY,
                 "truth identity is not stable")
    return natural_items, natural_audit, truth_by_split, combined


def load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[MODEL_NAME]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def tokenizer_audit(
    tok, natural_items: list[dict[str, Any]],
    truth_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    answer_ids = {label: core.single_token_id(tok, label) for label in ("A", "B")}
    period_id = core.single_token_id(tok, ".")
    think_open_id = core.single_token_id(tok, "<think>")
    think_close_id = core.single_token_id(tok, "</think>")
    core.require(answer_ids == {"A": 32, "B": 33},
                 f"Qwen3 A/B token identity changed: {answer_ids}")
    core.require(period_id == 13, f"Qwen3 period token identity changed: {period_id}")

    natural_prefix_hashes: list[dict[str, Any]] = []
    for item in natural_items:
        for control in core.CONTROL_POLICIES:
            user, rendered, ids = core.render_prefix(tok, item, control)
            opens = core.positions_of(ids, {think_open_id})
            closes = core.positions_of(ids, {think_close_id})
            if control == "hard_no_think":
                core.require(len(opens) == len(closes) == 1 and opens[0] < closes[0],
                             "hard-no-think lacks prefilled empty think block")
            else:
                core.require(not opens and not closes,
                             f"{control} unexpectedly prefills think tags")
            natural_prefix_hashes.append({
                "id": item["id"], "control": control,
                "effective_user_prompt_sha256": core.sha256_json(user),
                "rendered_prefix_sha256": core.sha256_json(rendered),
                "input_ids_sha256": core.sha256_json(ids),
                "prompt_len": len(ids),
            })

    truth_contexts = 0
    truth_prefix_hashes: list[dict[str, Any]] = []
    for split in truth_data.SPLITS:
        for pair in truth_by_split[split]:
            for side in ("qA", "qB"):
                probe = {"prompt": pair["prompts"][side]}
                _user, rendered, input_ids = core.render_prefix(
                    tok, probe, "hard_no_think")
                for candidate in ("A", "B"):
                    bare = list(tok(
                        rendered + candidate, add_special_tokens=False,
                        return_attention_mask=False,
                    ).input_ids)
                    punctuated = list(tok(
                        rendered + candidate + ".", add_special_tokens=False,
                        return_attention_mask=False,
                    ).input_ids)
                    core.require(bare[:len(input_ids)] == input_ids
                                 and punctuated[:len(input_ids)] == input_ids,
                                 "truth answer changed official prefix tokenization")
                    core.require(bare[len(input_ids):] == [answer_ids[candidate]],
                                 "truth bare answer is not the frozen one-token suffix")
                    core.require(punctuated == bare + [period_id],
                                 "truth period is not the same pure one-token suffix")
                    truth_contexts += 1
                    truth_prefix_hashes.append({
                        "pair_id": pair["id"], "side": side,
                        "candidate": candidate,
                        "input_ids_sha256": core.sha256_json(input_ids),
                        "bare_ids_sha256": core.sha256_json(bare),
                        "period_ids_sha256": core.sha256_json(punctuated),
                    })
    core.require(truth_contexts == 512,
                 f"truth tokenizer denominator changed: {truth_contexts}")
    return {
        "tokenizer_class": type(tok).__name__,
        "tokenizer_length": len(tok),
        "chat_template_sha256": core.sha256_json(
            str(getattr(tok, "chat_template", ""))),
        "eos_token_id": int(tok.eos_token_id),
        "pad_token_id": int(tok.pad_token_id),
        "special_token_ids": {
            "A": answer_ids["A"], "B": answer_ids["B"],
            "period": period_id, "think_open": think_open_id,
            "think_close": think_close_id,
        },
        "natural_prefix_contexts": len(natural_prefix_hashes),
        "natural_prefixes_sha256": core.sha256_json(natural_prefix_hashes),
        "truth_bare_period_context_pairs": truth_contexts,
        "truth_all_periods_are_same_pure_one_token_suffix": True,
        "truth_prefixes_sha256": core.sha256_json(truth_prefix_hashes),
    }


def script_commitments() -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for label, path in SCRIPT_PATHS.items():
        core.require(path.is_file(), f"missing Phase979 script before seal: {path}")
        output[label] = {"path": relative(path), "sha256": core.sha256_file(path)}
    return output


def assert_clean_first_seal() -> None:
    if PROTOCOL_PATH.exists():
        return
    forbidden = [
        OUT / "manifest_natural.json", OUT / "rows_natural.jsonl",
        OUT / "generator_status_natural.json", OUT / "audit_natural.json",
        OUT / "manifest_truth_development.json",
        OUT / "rows_truth_development.jsonl",
        OUT / "generator_status_truth_development.json",
        OUT / "truth_admission_development.json",
        OUT / "manifest_truth_replication.json",
        OUT / "rows_truth_replication.jsonl",
        OUT / "generator_status_truth_replication.json",
        OUT / "truth_audit_replication.json",
    ]
    existing = [relative(path) for path in forbidden if path.exists()]
    core.require(not existing,
                 f"Phase979 output exists before protocol seal: {existing}")


def build_protocol() -> dict[str, Any]:
    core.require(core.PHASE == PHASE and core.SCHEMA_VERSION == SCHEMA_VERSION,
                 "Phase979 core identity mismatch")
    core.require(core.MAX_NEW_TOKENS == 2048 and core.BATCH_SIZE == 8,
                 "Phase979 natural budget/batch changed")
    assert_no_old_holdout_import()
    phase978, old_protocol = authenticate_phase978()
    current_runtime = runtime_versions()
    core.require(current_runtime == old_protocol.get("runtime_versions"),
                 "runtime differs from authenticated Phase978 runtime")
    model_identity = verify_model_identity(old_protocol)
    natural_items, natural_audit, truth_by_split, truth_audit = audit_datasets()
    tok = load_tokenizer()
    try:
        token_audit = tokenizer_audit(tok, natural_items, truth_by_split)
    finally:
        del tok
        gc.collect()
    scripts = script_commitments()
    natural_identity = dict(natural_audit["identity"])
    truth_identity = dict(truth_data.STABLE_IDENTITY)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "model_name": MODEL_NAME,
        "runtime_versions": current_runtime,
        "local_model_artifact_identity": model_identity,
        "phase978_commitments": phase978,
        "phase979_script_hashes": scripts,
        "natural_dataset_identity": {
            "path": relative(SCRIPT_PATHS["natural_dataset"]),
            "sha256": scripts["natural_dataset"]["sha256"],
            **natural_identity,
        },
        "truth_dataset_identity": truth_identity,
        "truth_dataset_module": {
            "path": relative(SCRIPT_PATHS["truth_dataset"]),
            "sha256": scripts["truth_dataset"]["sha256"],
        },
        "dataset_audits": {
            "natural_audit_sha256": core.sha256_json(natural_audit),
            "truth_combined_audit_sha256": core.sha256_json(truth_audit),
        },
        "tokenizer_audit": token_audit,
        "controls": core.CONTROL_POLICIES,
        "decoding_policies": core.DECODING_POLICIES,
        "official_cells": sorted([list(value) for value in core.OFFICIAL_CELLS]),
        "checkpoints": list(core.CHECKPOINTS),
        "decision_checkpoint": core.MAX_NEW_TOKENS,
        "max_new_tokens": core.MAX_NEW_TOKENS,
        "batch_size": core.BATCH_SIZE,
        "replicates": list(core.REPLICATES),
        "expected_natural_rows": EXPECTED_NATURAL_ROWS,
        "expected_truth_rows": {
            "development": EXPECTED_TRUTH_ROWS_PER_SPLIT,
            "replication": EXPECTED_TRUTH_ROWS_PER_SPLIT,
        },
        "natural_contract": {
            "new_diagnostic_cap_not_phase978_revision": True,
            "max_new_tokens": core.MAX_NEW_TOKENS,
            "single_rollout_per_row": True,
            "checkpoint_snapshots_are_prefixes_not_reruns": True,
            "per_row_independent_generator": True,
            "two_streams_are_seed_dependence_screen_not_variance_estimate": True,
            "terminal_states": list(core.TERMINAL_STATES),
            "cap_categories_are_right_censored_snapshots_not_terminal_failures": True,
            "screen_thresholds": NATURAL_SCREEN_THRESHOLDS,
            "holdout_loaded": False,
            "mechanism_authorized": False,
        },
        "truth_contract": {
            "control_policy": "hard_no_think",
            "teacher_forced": True,
            "sampling": False,
            "random_seed": None,
            "gap_formula": "g*=max_{j not in EOS} z_j - max_{e in EOS} z_e",
            "D_formula": (
                "D_r=.5*((G(qA,B,r)-G(qA,A,r))"
                "+(G(qB,A,r)-G(qB,B,r)))"
            ),
            "Q_correct_formula": (
                "Q_C=.5*((G(qA,A,period)-G(qA,A,bare))"
                "+(G(qB,B,period)-G(qB,B,bare)))"
            ),
            "Q_wrong_formula": (
                "Q_W=.5*((G(qA,B,period)-G(qA,B,bare))"
                "+(G(qB,A,period)-G(qB,A,bare)))"
            ),
            "interaction_identity": "I=D_period-D_bare=Q_W-Q_C",
            "positive_D_meaning": "correct answer has lower EOS gap / stronger EOS",
            "negative_Q_meaning": "period has lower EOS gap / stronger EOS",
            "thresholds": TRUTH_THRESHOLDS,
            "development_precedes_replication": True,
            "replication_source_precommitted_and_preaudited": True,
            "replication_is_not_analyst_blind_holdout": True,
            "replication_model_evaluation_requires_development_admission": True,
            "eos_top1_is_secondary_only": True,
            "natural_rollout_claim_authorized": False,
            "holdout_loaded": False,
            "mechanism_authorized": False,
        },
        "holdout_loaded": False,
        "mechanism_authorized": False,
        "decision_boundary": (
            "Phase979 is an external design diagnostic. No result revises the "
            "Phase978 NO-GO, opens the Phase977 holdout, or directly authorizes "
            "layer/span/cross-time mechanism experiments."
        ),
        "execution_contract": {
            "cpu_only_protocol_freeze": True,
            "model_weights_loaded": False,
            "generation_performed": False,
            "truth_forward_pass_performed": False,
            "old_holdout_module_imported": False,
            "old_holdout_module_parsed": False,
        },
    }
    assert_no_old_holdout_import()
    return {
        **payload,
        "protocol_sha256": core.sha256_json(payload),
        "created_at_utc": core.utc_now(),
    }


def install_or_validate(document: dict[str, Any], freeze: bool) -> None:
    verify_self_hash(document, "protocol_sha256", "created_at_utc",
                     "new Phase979 protocol")
    if PROTOCOL_PATH.exists():
        prior = core.load_json(PROTOCOL_PATH, "existing Phase979 protocol")
        verify_self_hash(prior, "protocol_sha256", "created_at_utc",
                         "existing Phase979 protocol")
        core.require(prior["protocol_sha256"] == document["protocol_sha256"],
                     "existing Phase979 protocol differs from current sources")
        return
    core.require(freeze, "protocol is not sealed; rerun with --freeze")
    OUT.mkdir(parents=True, exist_ok=True)
    core.atomic_write_json(PROTOCOL_PATH, document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true",
                        help="atomically install the protocol if no result exists")
    args = parser.parse_args()
    assert_clean_first_seal()
    protocol = build_protocol()
    install_or_validate(protocol, bool(args.freeze))
    print(json.dumps({
        "phase": PHASE,
        "protocol_sha256": protocol["protocol_sha256"],
        "sealed": PROTOCOL_PATH.exists(),
        "expected_natural_rows": EXPECTED_NATURAL_ROWS,
        "expected_truth_rows_per_split": EXPECTED_TRUTH_ROWS_PER_SPLIT,
        "holdout_loaded": False,
        "mechanism_authorized": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
