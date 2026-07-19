#!/usr/bin/env python3
"""CPU-only independent audit for Phase 979 truth x punctuation rows.

The development audit authenticates only the development manifest/rows and
emits the conditional replication admission.  The replication source is
already committed by the Phase 979 protocol, but replication model output
must not exist before that admission.  The replication audit re-authenticates
both split artifacts and applies the frozen per-split and combined gates.

No model weights, natural trajectories, old holdout module, or internal
mechanism artifact is loaded by this script.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

import phase979_boundary_core as core  # noqa: E402
import phase979_truth_punctuation_dataset as dataset  # noqa: E402
from model_utils import MODEL_CONFIGS  # noqa: E402


PHASE = 979
SCHEMA_VERSION = 1
EXPERIMENT = "three_boundary_factorial_and_truth_punctuation"
RUNNER_EXPERIMENT = "truth_punctuation_teacher_forcing"
AUDIT_EXPERIMENT = "truth_punctuation_cross_independent_cpu_audit"
SPLITS = ("development", "replication")
PROMPT_SIDES = ("qA", "qB")
CANDIDATES = ("A", "B")
PUNCTUATIONS = ("bare", "period")
EXPECTED_PAIRS = 64
EXPECTED_ROWS = 512
EXPECTED_TASKS = 8

OUT = ROOT / "tests" / "glm5" / "result" / "phase979_three_boundary_factorial"
PROTOCOL_PATH = OUT / "protocol_preregistration.json"
DATASET_PATH = GLM5 / "phase979_truth_punctuation_dataset.py"
CORE_PATH = GLM5 / "phase979_boundary_core.py"
AUDIT_SCRIPT_PATH = Path(__file__).resolve()
DEVELOPMENT_ADMISSION_PATH = OUT / "truth_admission_development.json"
REPLICATION_AUDIT_PATH = OUT / "truth_audit_replication.json"
PHASE978_OPEN_RECEIPT_PATH = (
    GLM5 / "result" / "phase978_legal_budget_stabilization"
    / "holdout_open_receipt.json"
)
PHASE978_PUBLIC_ARTIFACTS = {
    "protocol": (
        GLM5 / "result" / "phase978_legal_budget_stabilization"
        / "protocol_preregistration.json"
    ),
    "development_admission": (
        GLM5 / "result" / "phase978_legal_budget_stabilization"
        / "admission_development.json"
    ),
    "development_postmortem": (
        GLM5 / "result" / "phase978_legal_budget_stabilization"
        / "postmortem_development.json"
    ),
}

TRUTH_MEAN_MIN = 2.0
TRUTH_POSITIVE_MIN = 48
TRUTH_TASKS_MIN = 6
TRUTH_TASK_POSITIVE_MIN = 6
TRUTH_TASK_MEAN_FLOOR = -2.0
PUNCTUATION_MEAN_MAX = -2.0
PUNCTUATION_NEGATIVE_MIN = 48
PUNCTUATION_TASKS_MIN = 6
COMBINED_POSITIVE_MIN = 96
COMBINED_TASKS_MIN = 6
COMBINED_TASK_POSITIVE_MIN = 12

FLOAT_ABS_TOL = 1e-5
FLOAT_REL_TOL = 1e-6

REQUIRED_ROW_FIELDS = {
    "schema_version", "phase", "experiment", "protocol_sha256",
    "pair_id", "task", "split", "prompt_side", "prompt_id", "candidate",
    "punctuation", "is_correct", "answer_text", "input_ids",
    "answer_prefix_ids", "answer_suffix_ids", "eos_ids", "eos_logit",
    "selected_eos_id", "max_non_eos_id", "top1_id", "max_non_eos_logit",
    "gap", "eos_rank", "eos_top1", "eos_probability",
    "teacher_forcing", "sampling_performed", "holdout_loaded",
    "phase978_holdout_loaded", "mechanism_authorized",
    "manifest_sha256", "row_sha256",
}


def require(condition: bool, message: str) -> None:
    core.require(condition, message)


def assert_no_old_holdout_import() -> None:
    forbidden = [
        name for name in sys.modules
        if name == "phase977_holdout_dataset"
        or name.endswith(".phase977_holdout_dataset")
    ]
    require(not forbidden, f"old sealed holdout module imported: {forbidden}")
    require(
        not PHASE978_OPEN_RECEIPT_PATH.exists(),
        "Phase978 holdout OPEN receipt exists; Phase979 truth audit must stop",
    )


def manifest_path(split: str) -> Path:
    require(split in SPLITS, f"unknown truth split: {split}")
    return OUT / f"manifest_truth_{split}.json"


def rows_path(split: str) -> Path:
    require(split in SPLITS, f"unknown truth split: {split}")
    return OUT / f"rows_truth_{split}.jsonl"


def status_path(split: str) -> Path:
    require(split in SPLITS, f"unknown truth split: {split}")
    return OUT / f"generator_status_truth_{split}.json"


def authenticate_safe_pre_admission_replication_status() -> dict[str, Any]:
    """Allow only a fail-closed receipt from a rejected premature invocation."""
    path = status_path("replication")
    if not path.exists():
        return {"exists": False, "safe_rejection_receipt": False}
    status = core.load_json(path, "pre-admission truth replication status")
    _verify_self_hash(
        status, "status_sha256", "updated_at_utc",
        "pre-admission truth replication status",
    )
    require(
        status.get("schema_version") == SCHEMA_VERSION
        and status.get("phase") == PHASE
        and status.get("experiment") == RUNNER_EXPERIMENT
        and status.get("split") == "replication",
        "pre-admission replication status identity mismatch",
    )
    require(
        status.get("state") == "FAILED"
        and status.get("complete") is False
        and status.get("completed_rows") == 0
        and status.get("manifest_sha256") is None
        and status.get("replication_authenticated") is False
        and status.get("dataset_block_built") is False
        and status.get("model_weights_loaded") is False
        and status.get("model_forward_performed") is False
        and status.get("replication_model_evaluated") is False
        and status.get("sampling_performed") is False
        and status.get("holdout_loaded") is False
        and status.get("phase978_open_receipt_exists") is False
        and status.get("mechanism_authorized") is False,
        "replication status is not a safe pre-admission rejection receipt",
    )
    require(isinstance(status.get("error_type"), str),
            "pre-admission rejection status lacks an error type")
    return {
        "exists": True,
        "safe_rejection_receipt": True,
        "status_sha256": status["status_sha256"],
        "status_file_sha256": core.sha256_file(path),
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_list(value: Any, label: str, *, nonempty: bool = True) -> list[int]:
    require(isinstance(value, list), f"{label} is not a list")
    require(not nonempty or bool(value), f"{label} is empty")
    require(all(_is_int(item) for item in value), f"{label} has non-integer IDs")
    return [int(item) for item in value]


def _finite(value: Any, label: str) -> float:
    require(not isinstance(value, bool), f"{label} is Boolean")
    return core.finite_number(value, label)


def _close(left: float, right: float) -> bool:
    return math.isclose(
        left, right, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL
    )


def _verify_self_hash(
    document: dict[str, Any], hash_field: str, time_field: str, label: str,
) -> None:
    claimed = document.get(hash_field)
    require(isinstance(claimed, str) and len(claimed) == 64,
            f"{label} lacks {hash_field}")
    payload = core.without_fields(document, hash_field, time_field)
    require(claimed == core.sha256_json(payload), f"{label} self-hash invalid")


def _resolve_workspace_path(path_text: str, label: str) -> Path:
    raw = Path(path_text)
    require(not raw.is_absolute(), f"{label} path must be workspace-relative")
    resolved = (ROOT / raw).resolve()
    root = ROOT.resolve()
    require(
        resolved == root or root in resolved.parents,
        f"{label} path escapes the workspace: {path_text}",
    )
    return resolved


def _verify_script_commitments(protocol: dict[str, Any]) -> dict[str, str]:
    entries = protocol.get("phase979_script_hashes")
    require(isinstance(entries, dict) and entries,
            "protocol lacks Phase979 script hashes")
    verified: dict[str, str] = {}
    by_path: dict[Path, str] = {}
    for label, entry in entries.items():
        require(isinstance(entry, dict)
                and isinstance(entry.get("path"), str)
                and isinstance(entry.get("sha256"), str),
                f"invalid Phase979 script commitment: {label}")
        path = _resolve_workspace_path(entry["path"], f"Phase979 script {label}")
        require(path.is_file(), f"missing sealed Phase979 script: {entry['path']}")
        actual = core.sha256_file(path)
        require(actual == entry["sha256"],
                f"sealed Phase979 script changed: {entry['path']}")
        verified[str(label)] = actual
        by_path[path] = actual
    for path in (CORE_PATH, DATASET_PATH, AUDIT_SCRIPT_PATH):
        resolved = path.resolve()
        require(resolved in by_path,
                f"protocol does not seal required truth audit file: {path.name}")
        require(by_path[resolved] == core.sha256_file(path),
                f"required truth audit file differs from seal: {path.name}")
    return verified


def _verify_phase978_commitments(commitments: Any) -> None:
    require(isinstance(commitments, dict), "protocol lacks Phase978 commitments")
    require(
        commitments.get("development_gate_passed") is False
        and commitments.get("holdout_authorized") is False
        and commitments.get("holdout_loaded") is False
        and commitments.get("mechanism_authorized") is False
        and commitments.get("open_receipt_exists") is False,
        "protocol does not preserve Phase978 NO-GO",
    )
    for label, expected_path in PHASE978_PUBLIC_ARTIFACTS.items():
        entry = commitments.get(label)
        require(
            isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
            and isinstance(entry.get("sha256"), str),
            f"invalid Phase978 public-artifact commitment: {label}",
        )
        path = _resolve_workspace_path(
            str(entry["path"]), f"Phase978 public artifact {label}"
        )
        require(
            path == expected_path.resolve(),
            f"Phase978 public-artifact path mismatch: {label}",
        )
        require(path.is_file(), f"missing Phase978 public artifact: {path}")
        require(
            core.sha256_file(path) == entry["sha256"],
            f"Phase978 public artifact changed: {label}",
        )


def _frozen_thresholds() -> dict[str, Any]:
    return {
        "development_and_replication_each": {
            "truth_effect_D0_and_D1": {
                "mean_min_logits": TRUTH_MEAN_MIN,
                "positive_pairs_min": TRUTH_POSITIVE_MIN,
                "pair_denominator": EXPECTED_PAIRS,
                "tasks_meeting_positive_6_of_8_min": TRUTH_TASKS_MIN,
                "any_task_mean_failure_at_or_below": TRUTH_TASK_MEAN_FLOOR,
            },
            "punctuation_effect_QC_and_QW": {
                "mean_max_logits": PUNCTUATION_MEAN_MAX,
                "negative_pairs_min": PUNCTUATION_NEGATIVE_MIN,
                "pair_denominator": EXPECTED_PAIRS,
                "tasks_with_negative_mean_min": PUNCTUATION_TASKS_MIN,
            },
        },
        "replication_combined_confirmation": {
            "truth_effect_D0_and_D1_positive_pairs_min": COMBINED_POSITIVE_MIN,
            "pair_denominator": EXPECTED_PAIRS * 2,
            "tasks_meeting_positive_12_of_16_min": COMBINED_TASKS_MIN,
        },
    }


def authenticate_protocol() -> dict[str, Any]:
    assert_no_old_holdout_import()
    protocol = core.load_json(PROTOCOL_PATH, "Phase979 protocol preregistration")
    _verify_self_hash(
        protocol, "protocol_sha256", "created_at_utc",
        "Phase979 protocol preregistration",
    )
    require(protocol.get("phase") == PHASE, "wrong protocol phase")
    require(protocol.get("schema_version") == SCHEMA_VERSION,
            "wrong protocol schema")
    require(protocol.get("experiment") == EXPERIMENT,
            "wrong protocol experiment")
    require(protocol.get("expected_truth_rows") == {
        "development": EXPECTED_ROWS, "replication": EXPECTED_ROWS,
    }, "truth row denominator changed")
    identity_core = core.without_fields(
        dataset.STABLE_IDENTITY, "identity_sha256"
    )
    require(
        dataset.STABLE_IDENTITY.get("identity_sha256")
        == core.sha256_json(identity_core),
        "runtime truth dataset identity self-hash invalid",
    )
    require(protocol.get("truth_dataset_identity") == dataset.STABLE_IDENTITY,
            "protocol truth dataset identity mismatch")
    module = protocol.get("truth_dataset_module")
    require(isinstance(module, dict)
            and _resolve_workspace_path(
                str(module.get("path", "")), "truth dataset module"
            )
            == DATASET_PATH.resolve()
            and module.get("sha256") == core.sha256_file(DATASET_PATH),
            "protocol truth dataset module commitment mismatch")
    contract = protocol.get("truth_contract")
    require(isinstance(contract, dict), "protocol lacks truth_contract")
    require(contract.get("control_policy") == "hard_no_think"
            and contract.get("teacher_forced") is True
            and contract.get("sampling") is False
            and contract.get("random_seed") is None,
            "truth execution contract changed")
    require(
        contract.get("gap_formula")
        == "g*=max_{j not in EOS} z_j - max_{e in EOS} z_e"
        and contract.get("D_formula")
        == "D_r=.5*((G(qA,B,r)-G(qA,A,r))+(G(qB,A,r)-G(qB,B,r)))"
        and contract.get("Q_correct_formula")
        == (
            "Q_C=.5*((G(qA,A,period)-G(qA,A,bare))"
            "+(G(qB,B,period)-G(qB,B,bare)))"
        )
        and contract.get("Q_wrong_formula")
        == (
            "Q_W=.5*((G(qA,B,period)-G(qA,B,bare))"
            "+(G(qB,A,period)-G(qB,A,bare)))"
        )
        and contract.get("interaction_identity")
        == "I=D_period-D_bare=Q_W-Q_C",
        "truth metric formulas changed",
    )
    require(contract.get("thresholds") == _frozen_thresholds(),
            "truth thresholds changed")
    require(contract.get("development_precedes_replication") is True
            and contract.get("replication_source_precommitted_and_preaudited") is True
            and contract.get("replication_is_not_analyst_blind_holdout") is True
            and contract.get(
                "replication_model_evaluation_requires_development_admission") is True,
            "truth replication contract changed")
    require(contract.get("eos_top1_is_secondary_only") is True
            and contract.get("natural_rollout_claim_authorized") is False
            and contract.get("holdout_loaded") is False
            and contract.get("mechanism_authorized") is False,
            "truth claim boundary changed")
    token_audit = protocol.get("tokenizer_audit")
    require(isinstance(token_audit, dict)
            and token_audit.get("special_token_ids", {}).get("A") == 32
            and token_audit.get("special_token_ids", {}).get("B") == 33
            and token_audit.get("special_token_ids", {}).get("period") == 13
            and token_audit.get("truth_bare_period_context_pairs") == 512
            and token_audit.get(
                "truth_all_periods_are_same_pure_one_token_suffix") is True,
            "protocol tokenizer truth seal changed")
    _verify_phase978_commitments(protocol.get("phase978_commitments"))
    require(protocol.get("holdout_loaded") is False
            and protocol.get("mechanism_authorized") is False,
            "protocol crosses forbidden boundary")
    _verify_script_commitments(protocol)
    assert_no_old_holdout_import()
    return protocol


def authenticate_manifest(
    split: str, protocol: dict[str, Any], expected_identity: dict[str, Any],
) -> dict[str, Any]:
    path = manifest_path(split)
    manifest = core.load_json(path, f"Phase979 truth {split} manifest")
    _verify_self_hash(
        manifest, "manifest_sha256", "created_at_utc",
        f"Phase979 truth {split} manifest",
    )
    require(manifest.get("phase") == PHASE, "wrong truth manifest phase")
    require(manifest.get("schema_version") == SCHEMA_VERSION,
            "wrong truth manifest schema")
    require(manifest.get("experiment") == RUNNER_EXPERIMENT,
            "wrong truth manifest experiment")
    require(manifest.get("split") == split, "truth manifest split mismatch")
    require(manifest.get("protocol_sha256") == protocol["protocol_sha256"],
            "truth manifest protocol mismatch")
    require(manifest.get("expected_rows") == EXPECTED_ROWS,
            "truth manifest denominator mismatch")
    require(manifest.get("dataset_identity") == dataset.STABLE_IDENTITY,
            "truth manifest dataset identity mismatch")
    require(manifest.get("dataset_split_sha256") == expected_identity["split_sha256"],
            "truth manifest split dataset hash mismatch")
    require(manifest.get("dataset_module_sha256") == core.sha256_file(DATASET_PATH),
            "truth manifest dataset module hash mismatch")
    require(
        _resolve_workspace_path(
            str(manifest.get("dataset_module_path", "")),
            "truth manifest dataset module",
        ) == DATASET_PATH.resolve(),
        "truth manifest dataset module path mismatch",
    )
    runner_path = GLM5 / "phase979_truth_punctuation.py"
    require(
        _resolve_workspace_path(
            str(manifest.get("runner_path", "")), "truth manifest runner"
        ) == runner_path.resolve()
        and manifest.get("runner_sha256") == core.sha256_file(runner_path),
        "truth manifest runner commitment mismatch",
    )
    require(
        _resolve_workspace_path(
            str(manifest.get("boundary_core_path", "")),
            "truth manifest boundary core",
        ) == CORE_PATH.resolve()
        and manifest.get("boundary_core_sha256") == core.sha256_file(CORE_PATH),
        "truth manifest boundary-core commitment mismatch",
    )
    require(
        manifest.get("n_pairs") == EXPECTED_PAIRS
        and manifest.get("actual_case_count") == EXPECTED_ROWS,
        "truth manifest pair/case denominator mismatch",
    )
    require(manifest.get("control_policy") == "hard_no_think"
            and manifest.get("teacher_forced") is True
            and manifest.get("teacher_forcing") is True
            and manifest.get("sampling") is False
            and manifest.get("sampling_performed") is False,
            "truth manifest execution mode changed")
    require(manifest.get("holdout_loaded") is False
            and manifest.get("phase978_holdout_loaded") is False
            and manifest.get("mechanism_authorized") is False,
            "truth manifest crosses forbidden boundary")
    require(
        manifest.get("batch_size") == 8
        and manifest.get("device_type") == "cuda"
        and manifest.get("left_padding") is True
        and manifest.get("explicit_attention_mask") is True
        and manifest.get("explicit_position_ids") is True
        and manifest.get("tf32_enabled") is False,
        "truth manifest CUDA/batch/padding contract changed",
    )
    require(
        manifest.get("model_vocab_size") == runtime_model_vocab_size(),
        "truth manifest output-head vocabulary size differs from local config",
    )
    if split == "development":
        require(
            manifest.get("replication_authorized") is False
            and manifest.get("truth_admission_sha256") is None
            and manifest.get("truth_admission_file_sha256") is None,
            "development manifest unexpectedly crosses replication admission",
        )
    else:
        require(
            manifest.get("replication_authorized") is True
            and isinstance(manifest.get("truth_admission_sha256"), str)
            and isinstance(manifest.get("truth_admission_file_sha256"), str),
            "replication manifest lacks authenticated development admission",
        )
    eos_ids = _int_list(manifest.get("eos_token_ids"),
                        "truth manifest eos_token_ids")
    require(len(set(eos_ids)) == len(eos_ids), "manifest EOS IDs are duplicated")
    special = manifest.get("special_token_ids")
    frozen_special = protocol["tokenizer_audit"]["special_token_ids"]
    require(isinstance(special, dict)
            and special.get("A") == frozen_special["A"]
            and special.get("B") == frozen_special["B"]
            and special.get("period") == frozen_special["period"],
            "truth manifest special-token identity mismatch")
    token_runtime = manifest.get("token_audit")
    require(
        isinstance(token_runtime, dict)
        and token_runtime.get("official_control") == "hard_no_think"
        and token_runtime.get("enable_thinking") is False
        and token_runtime.get("n_official_prefixes") == EXPECTED_PAIRS * 2
        and token_runtime.get("n_teacher_forced_states") == EXPECTED_ROWS
        and token_runtime.get("bare_labels_are_one_token") is True
        and token_runtime.get(
            "period_is_same_pure_one_token_suffix_everywhere"
        ) is True
        and token_runtime.get("answer_states_preserve_official_prefix") is True,
        "truth manifest runtime tokenizer audit is incomplete",
    )
    return manifest


def load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def runtime_eos_ids(tok) -> list[int]:
    """Recover the frozen EOS union without loading model weights."""
    values: list[Any] = [getattr(tok, "eos_token_id", None)]
    model_root = Path(MODEL_CONFIGS["qwen3"]["path"])
    for name in ("generation_config.json", "config.json"):
        path = model_root / name
        require(path.is_file(), f"missing local model config: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid local model config: {path}") from exc
        require(isinstance(value, dict), f"model config is not an object: {path}")
        values.append(value.get("eos_token_id"))
    output: list[int] = []
    for value in values:
        if value is None:
            continue
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            require(_is_int(candidate), f"invalid EOS token ID: {candidate!r}")
            if int(candidate) not in output:
                output.append(int(candidate))
    require(bool(output), "no runtime EOS token IDs found")
    return output


def runtime_model_vocab_size() -> int:
    """Read the output-head vocabulary width without loading model weights."""
    path = Path(MODEL_CONFIGS["qwen3"]["path"]) / "config.json"
    require(path.is_file(), f"missing local Qwen3 config: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid local Qwen3 config: {path}") from exc
    require(isinstance(config, dict), "local Qwen3 config is not an object")
    value = config.get("vocab_size")
    require(_is_int(value) and int(value) > 0,
            "local Qwen3 config lacks a valid vocab_size")
    return int(value)


def expected_split_identity(split: str, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    require(split in SPLITS, f"unknown split: {split}")
    field = f"{split}_pairs_sha256"
    split_sha = dataset.STABLE_IDENTITY.get(field)
    require(isinstance(split_sha, str) and len(split_sha) == 64,
            f"stable identity lacks {field}")
    require(core.sha256_json(pairs) == split_sha,
            f"runtime {split} pairs differ from stable identity")
    return {"split": split, "split_sha256": split_sha}


def expected_keys(
    pairs: list[dict[str, Any]],
) -> set[tuple[str, str, str, str]]:
    return {
        (str(pair["id"]), side, candidate, punctuation)
        for pair in pairs
        for side in PROMPT_SIDES
        for candidate in CANDIDATES
        for punctuation in PUNCTUATIONS
    }


def read_rows(
    split: str, manifest_sha256: str,
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    path = rows_path(split)
    require(path.is_file(), f"missing truth {split} rows: {path}")
    payload = path.read_bytes()
    require(payload.endswith(b"\n"), f"truth {split} JSONL lacks final newline")
    records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for line_number, raw in enumerate(payload.splitlines(), 1):
        require(bool(raw.strip()), f"truth {split} row {line_number} is blank")
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"malformed truth {split} row {line_number}"
            ) from exc
        require(isinstance(row, dict),
                f"truth {split} row {line_number} is not an object")
        missing = sorted(REQUIRED_ROW_FIELDS - set(row))
        require(not missing,
                f"truth {split} row {line_number} missing fields {missing}")
        claimed = row.get("row_sha256")
        require(claimed == core.sha256_json(core.without_fields(row, "row_sha256")),
                f"truth {split} row self-hash mismatch {line_number}")
        require(row.get("manifest_sha256") == manifest_sha256,
                f"truth {split} row manifest mismatch {line_number}")
        key = core.truth_key(row)
        require(key not in records, f"duplicate truth factorial key {key}")
        records[key] = row
    require(len(records) == EXPECTED_ROWS,
            f"expected {EXPECTED_ROWS} truth {split} rows, got {len(records)}")
    return records


def _validate_primitive_metrics(
    row: dict[str, Any], key: tuple[str, ...], vocab_n: int,
    eos_ids: set[int],
) -> None:
    selected_eos_id = row["selected_eos_id"]
    max_non_eos_id = row["max_non_eos_id"]
    top1_id = row["top1_id"]
    for field, value in (
        ("selected_eos_id", selected_eos_id),
        ("max_non_eos_id", max_non_eos_id),
        ("top1_id", top1_id),
    ):
        require(
            _is_int(value) and 0 <= int(value) < vocab_n,
            f"{key} invalid vocabulary ID in {field}",
        )
    selected_eos_id = int(selected_eos_id)
    max_non_eos_id = int(max_non_eos_id)
    top1_id = int(top1_id)
    require(selected_eos_id in eos_ids, f"{key} selected EOS ID is not sealed")
    require(max_non_eos_id not in eos_ids,
            f"{key} max_non_eos_id is an EOS token")

    eos_logit = _finite(row["eos_logit"], f"{key} eos_logit")
    max_non_eos = _finite(
        row["max_non_eos_logit"], f"{key} max_non_eos_logit"
    )
    gap = _finite(row["gap"], f"{key} gap")
    require(_close(gap, max_non_eos - eos_logit),
            f"{key} gap algebra mismatch")
    # An exact EOS/non-EOS top-logit tie is not a stable top1 diagnostic.
    require(abs(gap) > FLOAT_ABS_TOL, f"{key} ambiguous EOS/non-EOS tie")
    rank = row["eos_rank"]
    require(_is_int(rank) and 1 <= int(rank) <= vocab_n,
            f"{key} invalid 1-based EOS rank")
    top1 = row["eos_top1"]
    require(isinstance(top1, bool), f"{key} eos_top1 is not Boolean")
    require(top1 == (top1_id in eos_ids),
            f"{key} eos_top1/top1_id mismatch")
    require(top1 == (gap < 0.0), f"{key} eos_top1/gap sign mismatch")
    require((int(rank) == 1) == top1, f"{key} eos_rank/top1 mismatch")
    if not top1:
        require(top1_id == max_non_eos_id,
                f"{key} non-EOS top1 differs from max_non_eos_id")
    probability = _finite(row["eos_probability"], f"{key} eos_probability")
    require(0.0 <= probability <= 1.0,
            f"{key} EOS probability is outside [0,1]")
    if "eos_probability_total" in row:
        probability_total = _finite(
            row["eos_probability_total"], f"{key} eos_probability_total"
        )
        require(
            probability <= probability_total <= 1.0 + 1e-7,
            f"{key} selected/total EOS probability relationship is invalid",
        )


def validate_rows(
    split: str,
    manifest: dict[str, Any],
    records: dict[tuple[str, str, str, str], dict[str, Any]],
    pairs: list[dict[str, Any]],
    tok,
) -> dict[str, Any]:
    pair_by_id = {str(pair["id"]): pair for pair in pairs}
    require(len(pair_by_id) == EXPECTED_PAIRS, "truth pair IDs are duplicated")
    require(set(records) == expected_keys(pairs),
            f"truth {split} factorial key set mismatch")
    require(
        manifest.get("tokenizer_length") == len(tok),
        f"truth {split} manifest/runtime tokenizer length differs",
    )
    model_vocab_n = runtime_model_vocab_size()
    require(
        manifest.get("model_vocab_size") == model_vocab_n,
        f"truth {split} manifest/runtime model vocabulary differs",
    )
    runtime_eos = runtime_eos_ids(tok)
    manifest_eos = [int(value) for value in manifest["eos_token_ids"]]
    require(manifest_eos == runtime_eos,
            f"truth {split} manifest/runtime EOS IDs differ")
    require(
        all(0 <= value < model_vocab_n for value in manifest_eos),
        f"truth {split} EOS ID is outside the model output vocabulary",
    )
    answer_token_ids = {
        label: core.single_token_id(tok, label) for label in CANDIDATES
    }
    period_id = core.single_token_id(tok, ".")
    frozen_special = manifest["special_token_ids"]
    require(answer_token_ids == {"A": 32, "B": 33}
            and period_id == 13
            and frozen_special.get("A") == answer_token_ids["A"]
            and frozen_special.get("B") == answer_token_ids["B"]
            and frozen_special.get("period") == period_id,
            "runtime/frozen A/B/period token identity differs")

    prefix_cache: dict[tuple[str, str], tuple[str, list[int]]] = {}
    expected_suffix_cache: dict[tuple[str, str, str, str], list[int]] = {}
    for pair in pairs:
        for side in PROMPT_SIDES:
            probe = {"prompt": pair["prompts"][side]}
            _user, rendered, prefix_ids = core.render_prefix(
                tok, probe, "hard_no_think"
            )
            prefix_cache[(pair["id"], side)] = (rendered, prefix_ids)
            for candidate in CANDIDATES:
                for punctuation in PUNCTUATIONS:
                    text = pair["answer_states"][candidate][punctuation]
                    full_ids = list(tok(
                        rendered + text, add_special_tokens=False,
                        return_attention_mask=False,
                    ).input_ids)
                    full_ids = [int(value) for value in full_ids]
                    require(full_ids[:len(prefix_ids)] == prefix_ids,
                            "answer state changed official prefix tokenization")
                    suffix = full_ids[len(prefix_ids):]
                    expected_suffix_cache[(
                        pair["id"], side, candidate, punctuation
                    )] = suffix
            for candidate in CANDIDATES:
                bare = expected_suffix_cache[(pair["id"], side, candidate, "bare")]
                period = expected_suffix_cache[(
                    pair["id"], side, candidate, "period"
                )]
                require(bare == [answer_token_ids[candidate]],
                        "bare A/B answer is not the frozen one-token suffix")
                require(period == bare + [period_id],
                        "period answer is not the same pure one-token suffix")

    for key, row in records.items():
        pair_id, side, candidate, punctuation = key
        pair = pair_by_id[pair_id]
        prompt_id = pair["prompt_ids"][side]
        expected_correct = bool(pair["truth_table"][side][candidate])
        expected_text = pair["answer_states"][candidate][punctuation]
        rendered, expected_input = prefix_cache[(pair_id, side)]
        expected_full = [int(value) for value in tok(
            rendered + expected_text, add_special_tokens=False,
            return_attention_mask=False,
        ).input_ids]
        expected_suffix = expected_suffix_cache[key]

        input_ids = _int_list(row["input_ids"], f"{key} input_ids")
        answer_prefix = _int_list(
            row["answer_prefix_ids"], f"{key} answer_prefix_ids"
        )
        answer_suffix = _int_list(
            row["answer_suffix_ids"], f"{key} answer_suffix_ids"
        )
        row_eos = _int_list(row["eos_ids"], f"{key} eos_ids")
        require(
            row["schema_version"] == SCHEMA_VERSION
            and row["phase"] == PHASE
            and row["experiment"] == RUNNER_EXPERIMENT
            and row["protocol_sha256"] == manifest["protocol_sha256"]
            and row["manifest_sha256"] == manifest["manifest_sha256"]
            and row["pair_id"] == pair_id
            and row["task"] == pair["task"]
            and row["split"] == split
            and row["prompt_side"] == side
            and row["prompt_id"] == prompt_id
            and row["candidate"] == candidate
            and row["punctuation"] == punctuation
            and row["is_correct"] is expected_correct
            and row["answer_text"] == expected_text,
            f"truth design fields mismatch {key}",
        )
        require(
            row["teacher_forcing"] is True
            and row["sampling_performed"] is False
            and row["holdout_loaded"] is False
            and row["phase978_holdout_loaded"] is False
            and row["mechanism_authorized"] is False,
            f"truth execution/boundary metadata mismatch {key}",
        )
        require(input_ids == expected_input,
                f"official hard-no-think input prefix mismatch {key}")
        require(answer_prefix == expected_full,
                f"teacher-forced answer prefix mismatch {key}")
        require(answer_prefix[:len(input_ids)] == input_ids,
                f"answer prefix does not preserve official input prefix {key}")
        require(answer_suffix == answer_prefix[len(input_ids):] == expected_suffix,
                f"answer suffix mismatch {key}")
        require(row_eos == manifest_eos,
                f"row EOS IDs differ from manifest {key}")
        _validate_primitive_metrics(row, key, model_vocab_n, set(manifest_eos))

    if "rows_file_sha256" in manifest:
        require(manifest["rows_file_sha256"] == core.sha256_file(rows_path(split)),
                f"truth {split} manifest rows hash mismatch")
    return {
        "passed": True,
        "expected_factorial_keys_exact": True,
        "rows_n": len(records),
        "all_rows_self_hashed": True,
        "all_design_fields_reconstructed": True,
        "all_official_prefixes_recomputed": True,
        "all_answer_prefixes_recomputed": True,
        "all_answer_suffixes_recomputed": True,
        "all_bare_labels_single_token": True,
        "all_periods_same_pure_one_token_suffix": True,
        "answer_token_ids": answer_token_ids,
        "period_token_id": period_id,
        "eos_ids": manifest_eos,
        "model_vocab_size": model_vocab_n,
        "all_primitive_metric_algebra_valid": True,
        "rows_file_sha256": core.sha256_file(rows_path(split)),
    }


def pair_metrics(
    records: dict[tuple[str, str, str, str], dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    def gap(pair_id: str, side: str, candidate: str, punctuation: str) -> float:
        return float(records[(pair_id, side, candidate, punctuation)]["gap"])

    for pair in pairs:
        pair_id = str(pair["id"])
        g_aa0 = gap(pair_id, "qA", "A", "bare")
        g_ab0 = gap(pair_id, "qA", "B", "bare")
        g_ba0 = gap(pair_id, "qB", "A", "bare")
        g_bb0 = gap(pair_id, "qB", "B", "bare")
        g_aa1 = gap(pair_id, "qA", "A", "period")
        g_ab1 = gap(pair_id, "qA", "B", "period")
        g_ba1 = gap(pair_id, "qB", "A", "period")
        g_bb1 = gap(pair_id, "qB", "B", "period")
        d0 = 0.5 * ((g_ab0 - g_aa0) + (g_ba0 - g_bb0))
        d1 = 0.5 * ((g_ab1 - g_aa1) + (g_ba1 - g_bb1))
        q_correct = 0.5 * ((g_aa1 - g_aa0) + (g_bb1 - g_bb0))
        q_wrong = 0.5 * ((g_ab1 - g_ab0) + (g_ba1 - g_ba0))
        interaction = d1 - d0
        interaction_alt = q_wrong - q_correct
        require(_close(interaction, interaction_alt),
                f"interaction identity failed for {pair_id}")
        output.append({
            "pair_id": pair_id,
            "task": pair["task"],
            "split": pair["split"],
            "D_bare": d0,
            "D_period": d1,
            "Q_correct": q_correct,
            "Q_wrong": q_wrong,
            "interaction": interaction,
            "interaction_via_Q": interaction_alt,
        })
    return output


def _mean(values: list[float]) -> float:
    require(bool(values), "cannot average an empty effect list")
    return float(sum(values) / len(values))


def summarize_effect(
    metrics: list[dict[str, Any]], field: str,
) -> dict[str, Any]:
    values = [float(item[field]) for item in metrics]
    by_task: dict[str, Any] = {}
    for task in dataset.TASKS:
        task_values = [
            float(item[field]) for item in metrics if item["task"] == task
        ]
        require(bool(task_values), f"effect {field} lacks task {task}")
        by_task[task] = {
            "n": len(task_values),
            "mean": _mean(task_values),
            "positive_n": sum(value > 0.0 for value in task_values),
            "negative_n": sum(value < 0.0 for value in task_values),
            "zero_n": sum(value == 0.0 for value in task_values),
        }
    return {
        "field": field,
        "n": len(values),
        "mean": _mean(values),
        "positive_n": sum(value > 0.0 for value in values),
        "negative_n": sum(value < 0.0 for value in values),
        "zero_n": sum(value == 0.0 for value in values),
        "by_task": by_task,
    }


def truth_effect_gate(summary: dict[str, Any]) -> dict[str, Any]:
    task_positive = sum(
        block["positive_n"] >= TRUTH_TASK_POSITIVE_MIN
        for block in summary["by_task"].values()
    )
    task_floor_violations = sorted(
        task for task, block in summary["by_task"].items()
        if block["mean"] <= TRUTH_TASK_MEAN_FLOOR
    )
    checks = {
        "mean_at_least_2": summary["mean"] >= TRUTH_MEAN_MIN,
        "positive_pairs_at_least_48_of_64": (
            summary["positive_n"] >= TRUTH_POSITIVE_MIN
        ),
        "tasks_with_positive_at_least_6_of_8_at_least_6": (
            task_positive >= TRUTH_TASKS_MIN
        ),
        "no_task_mean_at_or_below_minus_2": not task_floor_violations,
    }
    return {
        "effect": summary["field"],
        "thresholds": {
            "mean_min": TRUTH_MEAN_MIN,
            "positive_pairs_min": TRUTH_POSITIVE_MIN,
            "tasks_min": TRUTH_TASKS_MIN,
            "positive_per_task_min": TRUTH_TASK_POSITIVE_MIN,
            "task_mean_must_be_above": TRUTH_TASK_MEAN_FLOOR,
        },
        "tasks_meeting_positive_threshold_n": task_positive,
        "task_mean_floor_violations": task_floor_violations,
        "checks": checks,
        "passed": all(checks.values()),
    }


def punctuation_effect_gate(summary: dict[str, Any]) -> dict[str, Any]:
    tasks_negative = sum(
        block["mean"] < 0.0 for block in summary["by_task"].values()
    )
    checks = {
        "mean_at_or_below_minus_2": summary["mean"] <= PUNCTUATION_MEAN_MAX,
        "negative_pairs_at_least_48_of_64": (
            summary["negative_n"] >= PUNCTUATION_NEGATIVE_MIN
        ),
        "tasks_with_negative_mean_at_least_6": (
            tasks_negative >= PUNCTUATION_TASKS_MIN
        ),
    }
    return {
        "effect": summary["field"],
        "thresholds": {
            "mean_max": PUNCTUATION_MEAN_MAX,
            "negative_pairs_min": PUNCTUATION_NEGATIVE_MIN,
            "tasks_with_negative_mean_min": PUNCTUATION_TASKS_MIN,
        },
        "tasks_with_negative_mean_n": tasks_negative,
        "checks": checks,
        "passed": all(checks.values()),
    }


def eos_top1_secondary(
    records: dict[tuple[str, str, str, str], dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for punctuation in PUNCTUATIONS:
        correct_n = 0
        wrong_n = 0
        correct_only = 0
        wrong_only = 0
        both = 0
        neither = 0
        comparisons = 0
        by_task: dict[str, Counter[str]] = defaultdict(Counter)
        for pair in pairs:
            pair_id = str(pair["id"])
            task = str(pair["task"])
            for side in PROMPT_SIDES:
                correct = str(pair["correct_label"][side])
                wrong = "B" if correct == "A" else "A"
                c = bool(records[(pair_id, side, correct, punctuation)]["eos_top1"])
                w = bool(records[(pair_id, side, wrong, punctuation)]["eos_top1"])
                comparisons += 1
                correct_n += c
                wrong_n += w
                label = (
                    "both" if c and w else
                    "correct_only" if c else
                    "wrong_only" if w else "neither"
                )
                by_task[task][label] += 1
                correct_only += label == "correct_only"
                wrong_only += label == "wrong_only"
                both += label == "both"
                neither += label == "neither"
        output[punctuation] = {
            "prompt_comparisons_n": comparisons,
            "correct_eos_top1_n": correct_n,
            "wrong_eos_top1_n": wrong_n,
            "correct_only_discordant_n": correct_only,
            "wrong_only_discordant_n": wrong_only,
            "both_n": both,
            "neither_n": neither,
            "by_task": {
                task: {
                    label: counts[label]
                    for label in ("correct_only", "wrong_only", "both", "neither")
                }
                for task, counts in sorted(by_task.items())
            },
            "role": "secondary_threshold_diagnostic_only",
        }
    return output


def build_split_metrics(
    records: dict[tuple[str, str, str, str], dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = pair_metrics(records, pairs)
    summaries = {
        field: summarize_effect(metrics, field)
        for field in (
            "D_bare", "D_period", "Q_correct", "Q_wrong", "interaction"
        )
    }
    truth_gates = {
        field: truth_effect_gate(summaries[field])
        for field in ("D_bare", "D_period")
    }
    punctuation_gates = {
        field: punctuation_effect_gate(summaries[field])
        for field in ("Q_correct", "Q_wrong")
    }
    truth_passed = all(block["passed"] for block in truth_gates.values())
    punctuation_passed = all(
        block["passed"] for block in punctuation_gates.values()
    )
    return {
        "pair_metrics": metrics,
        "effect_summaries": summaries,
        "truth_gates": truth_gates,
        "punctuation_gates": punctuation_gates,
        "truth_gate_passed": truth_passed,
        "punctuation_gate_passed": punctuation_passed,
        "both_effect_gates_passed": truth_passed and punctuation_passed,
        "eos_top1_secondary": eos_top1_secondary(records, pairs),
    }


def combined_truth_gate(
    development_metrics: list[dict[str, Any]],
    replication_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    combined = development_metrics + replication_metrics
    require(len(combined) == 128, "combined truth denominator is not 128 pairs")
    output: dict[str, Any] = {}
    for field in ("D_bare", "D_period"):
        summary = summarize_effect(combined, field)
        tasks_passing = sum(
            block["positive_n"] >= COMBINED_TASK_POSITIVE_MIN
            for block in summary["by_task"].values()
        )
        checks = {
            "positive_pairs_at_least_96_of_128": (
                summary["positive_n"] >= COMBINED_POSITIVE_MIN
            ),
            "tasks_with_positive_at_least_12_of_16_at_least_6": (
                tasks_passing >= COMBINED_TASKS_MIN
            ),
        }
        output[field] = {
            "summary": summary,
            "tasks_meeting_positive_threshold_n": tasks_passing,
            "checks": checks,
            "passed": all(checks.values()),
        }
    return {
        "effects": output,
        "passed": all(block["passed"] for block in output.values()),
    }


def load_and_validate_split(
    split: str, protocol: dict[str, Any], tok,
) -> dict[str, Any]:
    pairs = dataset.build_pairs(split)
    data_audit = dataset.audit_pairs(pairs)
    require(data_audit.get("passed") is True
            and data_audit.get("split_counts") == {split: EXPECTED_PAIRS},
            f"truth dataset audit failed for {split}")
    identity = expected_split_identity(split, pairs)
    manifest = authenticate_manifest(split, protocol, identity)
    records = read_rows(split, manifest["manifest_sha256"])
    integrity = validate_rows(split, manifest, records, pairs, tok)
    metrics = build_split_metrics(records, pairs)
    return {
        "split": split,
        "pairs": pairs,
        "dataset_audit": data_audit,
        "identity": identity,
        "manifest": manifest,
        "records": records,
        "integrity": integrity,
        "metrics": metrics,
    }


def _development_report(
    protocol: dict[str, Any], result: dict[str, Any],
    pre_admission_replication_status: dict[str, Any],
) -> dict[str, Any]:
    metrics = result["metrics"]
    integrity_passed = result["integrity"].get("passed") is True
    authorized = (
        integrity_passed
        and metrics["truth_gate_passed"]
        and metrics["punctuation_gate_passed"]
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": AUDIT_EXPERIMENT,
        "split": "development",
        "role": "conditional_replication_admission",
        "protocol_sha256": protocol["protocol_sha256"],
        "manifest_sha256": result["manifest"]["manifest_sha256"],
        "manifest_file_sha256": core.sha256_file(manifest_path("development")),
        "rows_file_sha256": core.sha256_file(rows_path("development")),
        "dataset_identity": dataset.STABLE_IDENTITY,
        "dataset_split_sha256": result["identity"]["split_sha256"],
        "dataset_audit_sha256": core.sha256_json(result["dataset_audit"]),
        "integrity": result["integrity"],
        "integrity_passed": integrity_passed,
        "metrics": metrics,
        "truth_gate_passed": metrics["truth_gate_passed"],
        "punctuation_gate_passed": metrics["punctuation_gate_passed"],
        "replication_authorized": authorized,
        "replication_model_evaluated": False,
        "pre_admission_replication_status": pre_admission_replication_status,
        "replication_source_status": (
            "precommitted_and_preaudited_in_protocol_but_not_model_evaluated"
        ),
        "replication_is_analyst_blind_holdout": False,
        "phase977_holdout_authorized": False,
        "mechanism_authorized": False,
        "holdout_loaded": False,
        "model_weights_loaded": False,
        "natural_rollout_claim_authorized": False,
        "decision_boundary": (
            "Replication authorization applies only to the precommitted Phase979 "
            "truth replication block. It does not open the old Phase977 holdout, "
            "revise Phase978, or authorize mechanism work."
        ),
    }
    return {
        **payload,
        "admission_sha256": core.sha256_json(payload),
        "audited_at_utc": core.utc_now(),
    }


def authenticate_development_admission(
    protocol: dict[str, Any], *, require_authorized: bool,
) -> dict[str, Any]:
    admission = core.load_json(
        DEVELOPMENT_ADMISSION_PATH, "Phase979 truth development admission"
    )
    _verify_self_hash(
        admission, "admission_sha256", "audited_at_utc",
        "Phase979 truth development admission",
    )
    require(admission.get("phase") == PHASE
            and admission.get("split") == "development"
            and admission.get("protocol_sha256") == protocol["protocol_sha256"],
            "development admission identity mismatch")
    require(
        admission.get("dataset_identity") == dataset.STABLE_IDENTITY
        and admission.get("dataset_split_sha256")
        == dataset.STABLE_IDENTITY["development_pairs_sha256"],
        "development admission dataset identity mismatch",
    )
    if require_authorized:
        require(
            admission.get("replication_authorized") is True
            and admission.get("integrity_passed") is True
            and admission.get("truth_gate_passed") is True
            and admission.get("punctuation_gate_passed") is True,
            "truth replication is not authorized by all development gates",
        )
    require(admission.get("replication_model_evaluated") is False
            and admission.get("phase977_holdout_authorized") is False
            and admission.get("mechanism_authorized") is False
            and admission.get("holdout_loaded") is False,
            "development admission crosses forbidden boundary")
    return admission


def _replication_report(
    protocol: dict[str, Any], admission: dict[str, Any],
    development: dict[str, Any], replication: dict[str, Any],
) -> dict[str, Any]:
    require(
        admission["manifest_sha256"]
        == development["manifest"]["manifest_sha256"]
        and admission["manifest_file_sha256"]
        == core.sha256_file(manifest_path("development"))
        and admission["rows_file_sha256"]
        == core.sha256_file(rows_path("development")),
        "development artifacts changed after replication admission",
    )
    require(
        replication["manifest"]["truth_admission_sha256"]
        == admission["admission_sha256"]
        and replication["manifest"]["truth_admission_file_sha256"]
        == core.sha256_file(DEVELOPMENT_ADMISSION_PATH),
        "replication manifest development-admission commitment mismatch",
    )
    combined = combined_truth_gate(
        development["metrics"]["pair_metrics"],
        replication["metrics"]["pair_metrics"],
    )
    replication_split_passed = bool(
        replication["integrity"].get("passed") is True
        and replication["metrics"]["truth_gate_passed"]
        and replication["metrics"]["punctuation_gate_passed"]
    )
    passed = bool(
        admission["replication_authorized"]
        and development["metrics"]["both_effect_gates_passed"]
        and replication_split_passed
        and combined["passed"]
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": AUDIT_EXPERIMENT,
        "split": "replication",
        "role": "precommitted_replication_confirmation",
        "protocol_sha256": protocol["protocol_sha256"],
        "development_admission_sha256": admission["admission_sha256"],
        "development": {
            "manifest_sha256": development["manifest"]["manifest_sha256"],
            "manifest_file_sha256": core.sha256_file(
                manifest_path("development")),
            "rows_file_sha256": core.sha256_file(rows_path("development")),
            "dataset_split_sha256": development["identity"]["split_sha256"],
            "metrics": development["metrics"],
        },
        "replication": {
            "manifest_sha256": replication["manifest"]["manifest_sha256"],
            "manifest_file_sha256": core.sha256_file(
                manifest_path("replication")),
            "rows_file_sha256": core.sha256_file(rows_path("replication")),
            "dataset_split_sha256": replication["identity"]["split_sha256"],
            "dataset_audit_sha256": core.sha256_json(
                replication["dataset_audit"]),
            "integrity": replication["integrity"],
            "metrics": replication["metrics"],
            "split_gate_passed": replication_split_passed,
        },
        "combined_truth_gate": combined,
        "truth_replication_confirmation_passed": passed,
        "replication_model_evaluated": True,
        "replication_is_analyst_blind_holdout": False,
        "phase977_holdout_authorized": False,
        "mechanism_authorized": False,
        "holdout_loaded": False,
        "model_weights_loaded": False,
        "natural_rollout_claim_authorized": False,
        "decision_boundary": (
            "A PASS is restricted to teacher-forced prompt-relative truth x "
            "punctuation behavior. It does not open the old Phase977 holdout or "
            "authorize any layer/span/cross-time mechanism experiment."
        ),
    }
    return {
        **payload,
        "audit_sha256": core.sha256_json(payload),
        "audited_at_utc": core.utc_now(),
    }


def install_or_validate(path: Path, report: dict[str, Any], hash_field: str) -> None:
    if path.exists():
        prior = core.load_json(path, f"existing {path.name}")
        _verify_self_hash(prior, hash_field, "audited_at_utc", path.name)
        require(prior.get(hash_field) == report.get(hash_field),
                f"existing {path.name} differs from recomputed audit")
        return
    core.atomic_write_json(path, report)


def audit(split: str, *, write: bool = True) -> dict[str, Any]:
    require(split in SPLITS, f"unknown truth audit split: {split}")
    assert_no_old_holdout_import()
    protocol = authenticate_protocol()

    if split == "development":
        forbidden = [
            manifest_path("replication"), rows_path("replication"),
            REPLICATION_AUDIT_PATH,
        ]
        existing = [str(path) for path in forbidden if path.exists()]
        require(not existing,
                f"replication model artifacts exist before admission: {existing}")
        pre_admission_status = authenticate_safe_pre_admission_replication_status()
        tok = load_tokenizer()
        try:
            development = load_and_validate_split("development", protocol, tok)
        finally:
            del tok
            gc.collect()
        report = _development_report(
            protocol, development, pre_admission_status
        )
        if write:
            install_or_validate(
                DEVELOPMENT_ADMISSION_PATH, report, "admission_sha256"
            )
    else:
        admission = authenticate_development_admission(
            protocol, require_authorized=True
        )
        tok = load_tokenizer()
        try:
            development = load_and_validate_split("development", protocol, tok)
            replication = load_and_validate_split("replication", protocol, tok)
        finally:
            del tok
            gc.collect()
        report = _replication_report(
            protocol, admission, development, replication
        )
        if write:
            install_or_validate(
                REPLICATION_AUDIT_PATH, report, "audit_sha256"
            )
    assert_no_old_holdout_import()
    return report


def synthetic_unit_test() -> dict[str, Any]:
    """Exercise formulas/gates with 64 synthetic complete factorial pairs."""
    primitive = {
        "selected_eos_id": 2,
        "max_non_eos_id": 10,
        "top1_id": 10,
        "eos_logit": 1.0,
        "max_non_eos_logit": 4.0,
        "gap": 3.0,
        "eos_rank": 2,
        "eos_top1": False,
        "eos_probability": 0.10,
        "eos_probability_total": 0.12,
    }
    _validate_primitive_metrics(
        primitive, ("synthetic", "qA", "A", "bare"), 12, {2}
    )
    rejected_bad_rank = False
    try:
        _validate_primitive_metrics(
            {**primitive, "eos_rank": 13},
            ("synthetic_bad_rank", "qA", "A", "bare"), 12, {2},
        )
    except RuntimeError:
        rejected_bad_rank = True
    require(rejected_bad_rank, "synthetic out-of-model-vocabulary rank was accepted")
    rejected_bad_probability = False
    try:
        _validate_primitive_metrics(
            {**primitive, "eos_probability_total": 0.05},
            ("synthetic_bad_probability", "qA", "A", "bare"), 12, {2},
        )
    except RuntimeError:
        rejected_bad_probability = True
    require(
        rejected_bad_probability,
        "synthetic selected/total EOS probability violation was accepted",
    )

    pairs = dataset.build_pairs("development")
    records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for pair in pairs:
        pair_id = str(pair["id"])
        for side in PROMPT_SIDES:
            correct = str(pair["correct_label"][side])
            for candidate in CANDIDATES:
                is_correct = candidate == correct
                for punctuation in PUNCTUATIONS:
                    base = 10.0 if is_correct else 13.0
                    value = base - (4.0 if punctuation == "period" else 0.0)
                    records[(pair_id, side, candidate, punctuation)] = {
                        "gap": value,
                        "eos_top1": value < 0.0,
                    }
    metrics = build_split_metrics(records, pairs)
    require(metrics["truth_gate_passed"] is True,
            "synthetic truth gate should pass")
    require(metrics["punctuation_gate_passed"] is True,
            "synthetic punctuation gate should pass")
    require(all(_close(item["D_bare"], 3.0)
                and _close(item["D_period"], 3.0)
                and _close(item["Q_correct"], -4.0)
                and _close(item["Q_wrong"], -4.0)
                and _close(item["interaction"], 0.0)
                for item in metrics["pair_metrics"]),
            "synthetic per-pair formulas changed")
    failing = {
        key: {**row, "gap": 10.0}
        for key, row in records.items()
    }
    failing_metrics = build_split_metrics(failing, pairs)
    require(failing_metrics["truth_gate_passed"] is False
            and failing_metrics["punctuation_gate_passed"] is False,
            "synthetic null effects should fail both gates")
    return {
        "passed": True,
        "pairs": len(pairs),
        "factorial_rows": len(records),
        "D_bare": 3.0,
        "D_period": 3.0,
        "Q_correct": -4.0,
        "Q_wrong": -4.0,
        "interaction": 0.0,
        "null_effects_fail_closed": True,
        "primitive_row_checks_passed": True,
        "output_head_ids_may_exceed_tokenizer_length": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--no-write", action="store_true",
        help="authenticate and recompute without installing the audit artifact",
    )
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(synthetic_unit_test(), indent=2))
        return
    require(args.split in SPLITS, "--split development|replication is required")
    report = audit(str(args.split), write=not bool(args.no_write))
    hash_field = (
        "admission_sha256" if args.split == "development" else "audit_sha256"
    )
    print(json.dumps({
        "phase": PHASE,
        "split": args.split,
        hash_field: report[hash_field],
        "replication_authorized": report.get("replication_authorized"),
        "truth_replication_confirmation_passed": report.get(
            "truth_replication_confirmation_passed"),
        "replication_model_evaluated": report["replication_model_evaluated"],
        "phase977_holdout_authorized": False,
        "mechanism_authorized": False,
        "holdout_loaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
