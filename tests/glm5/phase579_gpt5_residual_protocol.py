#!/usr/bin/env python3
"""CPU-only preregistration and freeze for Phase579 residual discovery.

This module deliberately uses only the Python standard library.  It seals an
observation contract and a complete development-case/replay manifest; it does
not import torch/transformers, load a checkpoint, touch CUDA, collect a trace,
select a coordinate, or introduce a mechanism formula.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# Hide CUDA before any repository-local module could be imported.  In fact this
# protocol imports no repository module at all; the environment guard makes the
# CPU-only boundary explicit and independently auditable.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase579"
MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
ELIGIBLE_MODELS = ("qwen3", "glm4")
BLOCKED_MODELS = ("deepseek7b",)
REPEATS = ("repeat1", "repeat2")

OUT_DIR = ROOT / "tests/glm5/result/phase579_gpt5_residual_protocol"
ENGINEERING_DIR = ROOT / "tests/glm5/result/phase579_gpt5_residual_engineering"
TRACE_DIR = ROOT / "tests/glm5/result/phase579_gpt5_residual_trace"
INVENTORY_DIR = ROOT / "tests/glm5/result/phase579_gpt5_residual_inventory"
EXECUTION_AUDIT_DIR = (
    ROOT / "tests/glm5/result/phase579_gpt5_residual_independent_audit"
)
FUTURE_RESULT_ROOTS = (
    ENGINEERING_DIR, TRACE_DIR, INVENTORY_DIR, EXECUTION_AUDIT_DIR,
)

MANIFEST_NAME = "phase579_development_residual_manifest.jsonl"
PROTOCOL_NAME = "phase579_preregistered_residual_protocol.json"
SELF_TEST_NAME = "phase579_protocol_self_test.json"
STAGE_COMMIT_NAME = "phase579_stage_commit.json"
AUDIT_NAME = "phase579_independent_freeze_audit.json"
FREEZE_NAME = "phase579_freeze_commit.json"

INITIAL_FILES = {MANIFEST_NAME, PROTOCOL_NAME, SELF_TEST_NAME, STAGE_COMMIT_NAME}
FINAL_FILES = INITIAL_FILES | {AUDIT_NAME, FREEZE_NAME}

FORMAL_PYTHON = Path(
    r"C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe"
)
FORMAL_PYTHON_SHA256 = (
    "0f11fb7422fa347b7609ba0964ceccef3c8fa9f15230c37b9ec27668e68e8a8a"
)
FORMAL_PACKAGES = {
    "torch": "2.11.0+cu128",
    "transformers": "5.12.0",
    "bitsandbytes": "0.49.2",
    "accelerate": "1.14.0",
}

SOURCE_RELATIVES = (
    "tests/glm5/phase579_gpt5_residual_runner.py",
    "tests/glm5/phase579_gpt5_residual_inventory.py",
    "tests/glm5/phase579_gpt5_residual_audit.py",
    "tests/glm5/phase579_gpt5_residual_protocol.py",
)

PHASE577_DIR = ROOT / "tests/glm5/result/phase577_gpt5_natural_behavior_protocol"
PHASE578_PROTOCOL_DIR = (
    ROOT / "tests/glm5/result/phase578_gpt5_runner_scorer_protocol"
)
PHASE578_RAW_DIR = (
    ROOT / "tests/glm5/result/phase578_gpt5_development_behavior_raw"
)
PHASE578_ANALYSIS_DIR = (
    ROOT / "tests/glm5/result/phase578_gpt5_development_behavior_analysis"
)
PHASE578_EXECUTION_AUDIT_DIR = (
    ROOT
    / "tests/glm5/result/phase578_gpt5_development_behavior_independent_audit"
)

DEVELOPMENT_PATH = PHASE577_DIR / "phase577_development_cases.jsonl"
SUMMARY_PATH = (
    PHASE578_ANALYSIS_DIR / "phase578_development_behavior_summary.json"
)
DECISION_PATHS = {
    "qwen3": PHASE578_ANALYSIS_DIR / "phase578_qwen3_development_decision.json",
    "glm4": PHASE578_ANALYSIS_DIR / "phase578_glm4_development_decision.json",
    "deepseek7b": (
        PHASE578_ANALYSIS_DIR / "phase578_deepseek7b_development_decision.json"
    ),
}
RAW_PATHS = {
    model: PHASE578_RAW_DIR / f"{index:02d}_{model}/raw_rows.jsonl.gz"
    for index, model in enumerate(MODEL_ORDER)
}
RAW_STATUS_PATHS = {
    model: PHASE578_RAW_DIR / f"{index:02d}_{model}/status.json"
    for index, model in enumerate(MODEL_ORDER)
}


# Completed upstream evidence is byte-pinned.  No confirmation, heldout, or
# sealed path is named or opened anywhere in this protocol.
UPSTREAM_EXPECTED: dict[str, tuple[Path, str]] = {
    "phase577_development": (
        DEVELOPMENT_PATH,
        "4c40ea882e1c0e2994441f64fe37dc531b326ab01edc8f24934866b127fd8a5c",
    ),
    "phase577_protocol": (
        PHASE577_DIR / "phase577_preregistered_protocol.json",
        "aaad6a29ae537255aa04df51c50c62a3f22c35943e05d30ab9897ea362d6df84",
    ),
    "phase577_freeze": (
        PHASE577_DIR / "phase577_freeze_commit.json",
        "4690654ac42259adcbb733028d147ab2ac8cd211d5e29eedecfa545c0c6b4533",
    ),
    "phase578_protocol": (
        PHASE578_PROTOCOL_DIR / "phase578_preregistered_runner_protocol.json",
        "a26e934d9f06b1955e92947dbd7b848e059fdfa2e612f74974dd7004b29c9722",
    ),
    "phase578_prompt_manifest": (
        PHASE578_PROTOCOL_DIR / "phase578_development_prompt_manifest.jsonl",
        "47dc8bef8b0d851097586ce48e933cd4b4008035297b07c9b4344f7ec2c08104",
    ),
    "phase578_freeze_audit": (
        PHASE578_PROTOCOL_DIR / "phase578_independent_audit.json",
        "1e32a291e716c0959f385b5509232115a0e42240413bcce8eb64c6353c9394ba",
    ),
    "phase578_freeze": (
        PHASE578_PROTOCOL_DIR / "phase578_freeze_commit.json",
        "01903b7c11d59b91a9ffe022297c011ff3a41f21c0c1ac2a7cdf6d3ef31fa49c",
    ),
    "phase578_raw_receipt": (
        PHASE578_RAW_DIR / "execution_receipt.json",
        "00b9572625f80955aea62fb475805e538232c1b2cf53be52c13b35817605cfe7",
    ),
    "phase578_qwen3_raw_status": (
        RAW_STATUS_PATHS["qwen3"],
        "6d1bc5b491b7f72d5c4784f0ca9b5a19e7b2307e45729a8d002604c7804215f0",
    ),
    "phase578_qwen3_raw_rows": (
        RAW_PATHS["qwen3"],
        "06fcf79b933b5efd217996e58d16758b796da9539b0919d55f318923f2d0ec60",
    ),
    "phase578_glm4_raw_status": (
        RAW_STATUS_PATHS["glm4"],
        "27d79e3386a95c538eff3a7a9d25bcad9588ac22ea3bd13fb38370ab16e7925a",
    ),
    "phase578_glm4_raw_rows": (
        RAW_PATHS["glm4"],
        "4bb3602c66ad2e37343bbfa4627f6e19b3ef43ad33426eee9548ad864df6f0e6",
    ),
    "phase578_deepseek7b_raw_status": (
        RAW_STATUS_PATHS["deepseek7b"],
        "171e7d58b76400c98be7035ee4d3d121406388d26ce6a12f978d20dbf96a77ed",
    ),
    "phase578_deepseek7b_raw_rows": (
        RAW_PATHS["deepseek7b"],
        "b8f7668ba9f9d2bb914af184f11f4427ea4d5e452db285a03b66d5ede1b3ee2a",
    ),
    "phase578_summary": (
        SUMMARY_PATH,
        "4d3d28ccf84b6ceb21aeb71daf7ed94d41f82611bd13c72cb59e2d2029b211d3",
    ),
    "phase578_qwen3_decision": (
        DECISION_PATHS["qwen3"],
        "ec9e4c0756a051b1a8b72d83a568232f456601eea71c7b31fbfcd914f02be058",
    ),
    "phase578_glm4_decision": (
        DECISION_PATHS["glm4"],
        "a9673a3ba71b9d12693fd950046e12ba793ea87ff3cc5a49ef2cd81b48e40004",
    ),
    "phase578_deepseek7b_decision": (
        DECISION_PATHS["deepseek7b"],
        "f568fb8c012accec1ed1c7cecb8093b5c30ef645614593335553a6f5d730f820",
    ),
    "phase578_analysis_receipt": (
        PHASE578_ANALYSIS_DIR / "phase578_analysis_receipt.json",
        "ef452a064c989defc076959a7c52538d2ef36da88a1731d9c2e20a075eff1380",
    ),
    "phase578_execution_audit": (
        PHASE578_EXECUTION_AUDIT_DIR
        / "phase578_development_independent_audit.json",
        "3803b9568e1c11b3ce8b39b1f9269a4e537a59d9526a13ebd2f0af80a792fb3e",
    ),
    "cross_model_engine": (
        ROOT / "tests/glm5/phase983_cross_model_engine.py",
        "e345daf3c3eae289eb7a71b8a741eeaf3a11c6897d009a5f9d90a386b23eef6f",
    ),
    "model_registry": (
        ROOT / "tests/gpt5/model_registry.py",
        "84c30398a9effa47791635fd25662426164460e036e59bb83a38c855db370864",
    ),
}


OBSERVATION_CANDIDATE_GATE: dict[str, Any] = {
    "registration_only_after_immutable_trace_and_independent_inventory_audit": True,
    "evidence_unit_by_axis": {
        "relation": "relation_focus_object_unit",
        "query_polarity": "analysis_unit_id",
        "selection_order": "analysis_unit_id",
        "output_contract": "analysis_unit_id",
        "paraphrase": "analysis_unit_id",
    },
    "minimum_replay_valid_cases": 24,
    "minimum_distinct_analysis_units": 6,
    "minimum_unit_vectors": 6,
    "minimum_distinct_focus_objects": 4,
    "minimum_distinct_surface_ids": 2,
    "minimum_distinct_paraphrase_ids": 2,
    "minimum_output_contracts": 2,
    "all_claimed_repeat_occurrences_must_match_frozen_token_replay": True,
    "finite_and_shape_valid_required": True,
    "scope_dimensions_must_be_explicit": [
        "relation", "interface", "query_polarity", "order",
        "output_contract", "surface_id", "paraphrase_id",
        "target_truth_polarity", "target", "foil", "focus_object",
        "focus_object_class", "comparison_object", "comparison_object_class",
        "left_option", "right_option", "relation_contract_id",
        "positive_object", "negative_object", "analysis_unit_id",
        "token_role", "layer",
    ],
    "tested_invariant_definition": (
        "non-contrast, non-fixed-coordinate, non-axis-coupled scope "
        "dimensions with at least two observed levels"
    ),
    "minimum_tested_invariant_dimensions": 2,
    "every_axis_claimed_invariant_requires_at_least_two_levels": True,
    "matched_control_observations_required": True,
    "axis_coupled_control_fields": {
        "relation": [
            "target_truth_polarity", "target", "foil", "positive_object",
            "negative_object", "analysis_unit_id", "relation_contract_id",
        ],
        "query_polarity": [
            "target_truth_polarity", "target", "foil", "focus_object",
            "focus_object_class", "comparison_object",
            "comparison_object_class",
        ],
        "selection_order": ["left_option", "right_option"],
        "output_contract": ["surface_id"],
        "paraphrase": ["surface_id"],
    },
    "matched_pair_allowed_differences": {
        "relation": [
            "relation", "target_truth_polarity", "target", "foil",
            "positive_object", "negative_object", "analysis_unit_id",
            "relation_contract_id", "raw_prompt",
        ],
        "query_polarity": [
            "query_polarity", "target_truth_polarity", "target", "foil",
            "focus_object", "focus_object_class", "comparison_object",
            "comparison_object_class", "raw_prompt",
        ],
        "selection_order": [
            "order", "left_option", "right_option", "raw_prompt",
        ],
        "output_contract": [
            "output_contract", "surface_id", "raw_prompt",
        ],
        "paraphrase": ["paraphrase_id", "surface_id", "raw_prompt"],
    },
    "minimum_distinct_control_slices": 2,
    "minimum_reproducing_control_slices": 2,
    "cross_unit_positive_dot_fraction_numerator": 4,
    "cross_unit_positive_dot_fraction_denominator": 5,
    "post_discovery_cross_unit_direction_check": {
        "evidence_counting_level": "cross_frozen_evidence_unit_pair",
        "positive_dot_definition": (
            "elementary paired displacement dot product is strictly positive"
        ),
        "integer_gate": (
            "5 * positive_pairwise_dot_count >= 4 * pairwise_dot_total"
        ),
        "minimum_eligible_unit_count": 6,
        "equivalent_fraction_floor": "4/5",
        "mean_is_diagnostic_only": True,
        "gate_applies_only_after_axis_role_layer_observation_is_discovered": True,
    },
    "average_only_or_single_case_registration_forbidden": True,
    "statistical_significance_claimed": False,
    "cross_model_support_allowed": False,
    "causal_mechanism_label_allowed": False,
    "formula_registration_allowed": False,
    "initial_label": "observation_candidate_only",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path, root: Path = ROOT) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"regular non-symlink file required: {path}")
    resolved = path.resolve(strict=True)
    try:
        label = str(resolved.relative_to(root.resolve(strict=True))).replace(
            "\\", "/"
        )
    except ValueError:
        label = str(resolved).replace("\\", "/")
    stat = path.stat()
    return {
        "path": label,
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
        "is_symlink": False,
        "hardlink_count": stat.st_nlink,
    }


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def read_jsonl_raw(path: Path) -> list[tuple[bytes, dict[str, Any]]]:
    rows: list[tuple[bytes, dict[str, Any]]] = []
    with path.open("rb") as handle:
        for raw_with_newline in handle:
            raw = raw_with_newline.rstrip(b"\r\n")
            if raw:
                value = json.loads(raw.decode("utf-8"))
                if not isinstance(value, dict):
                    raise RuntimeError(f"JSONL object required: {path}")
                rows.append((raw, value))
    return rows


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (canonical_json(row) + "\n").encode("utf-8") for row in rows
    )


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    if temporary.exists() or path.exists():
        raise RuntimeError(f"no-overwrite publication refused: {path}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def source_identities() -> dict[str, Any]:
    return {
        relative: identity(ROOT / relative) for relative in SOURCE_RELATIVES
    }


def verify_formal_interpreter() -> dict[str, Any]:
    observed = identity(FORMAL_PYTHON, FORMAL_PYTHON.parent)
    if observed["sha256"] != FORMAL_PYTHON_SHA256:
        raise RuntimeError("formal Python executable drift")
    packages = {
        name: importlib.metadata.version(name) for name in FORMAL_PACKAGES
    }
    if packages != FORMAL_PACKAGES:
        raise RuntimeError(f"formal package environment drift: {packages}")
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError("Phase579 freeze requires the frozen formal interpreter")
    return {
        "python_executable": str(FORMAL_PYTHON),
        "python_executable_identity": observed,
        "python_version": sys.version.split()[0],
        "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "side_environment_not_authorized": str(ROOT / ".venv/Scripts/python.exe"),
    }


def assert_future_roots_absent() -> None:
    present = [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in FUTURE_RESULT_ROOTS if path.exists()
    ]
    if present:
        raise RuntimeError(f"future Phase579 result roots already exist: {present}")


def _validate_artifact_verification(model: str, value: Any) -> None:
    if not isinstance(value, dict):
        raise RuntimeError(f"missing full model artifact verification: {model}")
    files = value.get("files")
    if (
        value.get("model") != model
        or not isinstance(files, list)
        or value.get("file_count") != len(files)
        or len(files) < 1
        or len({item.get("relative_path") for item in files}) != len(files)
    ):
        raise RuntimeError(f"invalid full model artifact registry: {model}")
    for item in files:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("relative_path"), str)
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] < 0
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
        ):
            raise RuntimeError(f"invalid model artifact file identity: {model}")
    if any(
        not isinstance(value.get(field), str) or len(value[field]) != 64
        for field in ("frozen_identity_sha256", "verification_payload_sha256")
    ):
        raise RuntimeError(f"invalid model artifact payload identity: {model}")


def frozen_model_artifact_identities() -> dict[str, Any]:
    phase578_protocol = read_json(
        UPSTREAM_EXPECTED["phase578_protocol"][0]
    )
    historical = phase578_protocol.get("frozen_model_artifact_identities")
    if not isinstance(historical, dict) or set(historical) != set(MODEL_ORDER):
        raise RuntimeError("Phase578 frozen model identity registry drift")
    output: dict[str, Any] = {}
    for model in MODEL_ORDER:
        status = read_json(RAW_STATUS_PATHS[model])
        verification = status.get("model_artifact_verification")
        _validate_artifact_verification(model, verification)
        if verification["frozen_identity_sha256"] != historical[model].get(
            "identity_sha256"
        ):
            raise RuntimeError(f"full model identity bridge mismatch: {model}")
        output[model] = verification
    return output


def verify_upstream() -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for name, (path, expected_hash) in UPSTREAM_EXPECTED.items():
        observed = identity(path)
        if observed["sha256"] != expected_hash:
            raise RuntimeError(f"upstream identity drift: {name}")
        reports[name] = observed

    phase578_freeze = read_json(UPSTREAM_EXPECTED["phase578_freeze"][0])
    if not all((
        phase578_freeze.get("freeze_complete") is True,
        phase578_freeze.get("candidate_coordinates") == [],
        phase578_freeze.get("candidate_mechanism_formulas") == [],
        phase578_freeze.get("confirmation_authorized") is False,
        phase578_freeze.get("heldout_authorized") is False,
        phase578_freeze.get("sealed_authorized") is False,
    )):
        raise RuntimeError("Phase578 freeze boundary drift")

    summary = read_json(SUMMARY_PATH)
    if not all((
        summary.get("phase_id") == "Phase578",
        summary.get("models_in_required_order") == list(MODEL_ORDER),
        summary.get("behavior_passed_models") == list(ELIGIBLE_MODELS),
        summary.get("future_single_model_natural_trace_eligible_models")
        == list(ELIGIBLE_MODELS),
        summary.get("behavior_blocked_models") == list(BLOCKED_MODELS),
        summary.get("cross_model_internal_comparison_authorized") is False,
        summary.get("internal_trace_run_count") == 0,
        summary.get("activation_collected") is False,
        summary.get("mechanism_claim_authorized") is False,
        summary.get("candidate_coordinates") == [],
        summary.get("candidate_mechanism_formulas") == [],
    )):
        raise RuntimeError("Phase578 eligibility boundary drift")

    raw_receipt = read_json(UPSTREAM_EXPECTED["phase578_raw_receipt"][0])
    attempts = raw_receipt.get("attempts")
    if not all((
        raw_receipt.get("phase_id") == "Phase578",
        raw_receipt.get("mode") == "development",
        raw_receipt.get("behavior_raw_execution_complete") is True,
        raw_receipt.get("attempted_models_in_order") == list(MODEL_ORDER),
        raw_receipt.get("completed_models") == list(MODEL_ORDER),
        raw_receipt.get("failed_models") == [],
        raw_receipt.get("not_attempted_models") == [],
        raw_receipt.get("fatal_cleanup_failure") is False,
        raw_receipt.get("activation_collected") is False,
        raw_receipt.get("hidden_states_requested") is False,
        raw_receipt.get("attentions_requested") is False,
        raw_receipt.get("scores_requested") is False,
        raw_receipt.get("causal_intervention") is False,
        raw_receipt.get("confirmation_accessed") is False,
        raw_receipt.get("heldout_accessed") is False,
        raw_receipt.get("sealed_accessed") is False,
        isinstance(attempts, list),
        len(attempts) == 3,
        [item.get("model") for item in attempts] == list(MODEL_ORDER),
        all(
            item.get("status") == "complete"
            and item.get("child_exit_code") == 0
            and item.get("cleanup_pass") is True
            for item in attempts
        ),
    )):
        raise RuntimeError("Phase578 raw execution receipt drift")

    analysis_receipt = read_json(
        UPSTREAM_EXPECTED["phase578_analysis_receipt"][0]
    )
    if not all((
        analysis_receipt.get("phase_id") == "Phase578",
        analysis_receipt.get("analysis_complete") is True,
        analysis_receipt.get("full_development_access") is True,
        analysis_receipt.get(
            "full_development_access_occurred_after_raw_publication"
        ) is True,
        analysis_receipt.get("activation_collected") is False,
        analysis_receipt.get("model_weights_loaded") is False,
        analysis_receipt.get("gpu_used") is False,
        analysis_receipt.get("confirmation_accessed") is False,
        analysis_receipt.get("heldout_accessed") is False,
        analysis_receipt.get("sealed_accessed") is False,
    )):
        raise RuntimeError("Phase578 analysis receipt drift")

    expected_gate = {"qwen3": True, "glm4": True, "deepseek7b": False}
    for model in MODEL_ORDER:
        decision = read_json(DECISION_PATHS[model])
        if not all((
            decision.get("phase_id") == "Phase578",
            decision.get("model") == model,
            decision.get("case_count") == 336,
            decision.get("repeat_row_count") == 672,
            decision.get("full_generated_identity_case_count") == 336,
            decision.get("behavior_gate_pass") is expected_gate[model],
            isinstance(decision.get("case_reports"), dict),
            len(decision["case_reports"]) == 336,
        )):
            raise RuntimeError(f"Phase578 decision boundary drift: {model}")

        status = read_json(RAW_STATUS_PATHS[model])
        cleanup = status.get("cleanup", {})
        if not all((
            status.get("phase_id") == "Phase578",
            status.get("model") == model,
            status.get("mode") == "development",
            status.get("error") is None,
            status.get("activation_collected") is False,
            status.get("hidden_states_requested") is False,
            status.get("attentions_requested") is False,
            status.get("causal_intervention") is False,
            cleanup.get("cleanup_pass") is True,
            cleanup.get("allocated_after_release") == 0,
            cleanup.get("reserved_after_release") == 0,
        )):
            raise RuntimeError(f"Phase578 raw status boundary drift: {model}")

    execution_audit = read_json(
        UPSTREAM_EXPECTED["phase578_execution_audit"][0]
    )
    if not all((
        execution_audit.get("passed") is True,
        all(execution_audit.get("checks", {}).values()),
        execution_audit.get("analysis_summary_sha256")
        == UPSTREAM_EXPECTED["phase578_summary"][1],
        execution_audit.get("raw_execution_receipt_sha256")
        == UPSTREAM_EXPECTED["phase578_raw_receipt"][1],
        execution_audit.get("model_weights_loaded") is False,
        execution_audit.get("gpu_used") is False,
    )):
        raise RuntimeError("Phase578 independent execution audit drift")

    # Forces validation of all recursively inventoried model artifact entries.
    frozen_model_artifact_identities()
    return reports


def _raw_replay_rows(model: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_path = RAW_PATHS[model]
    with gzip.open(raw_path, "rb") as handle:
        for raw_with_newline in handle:
            raw = raw_with_newline.rstrip(b"\r\n")
            if not raw:
                continue
            row = json.loads(raw.decode("utf-8"))
            if not isinstance(row, dict):
                raise RuntimeError(f"Phase578 raw row is not an object: {model}")
            if not all((
                row.get("schema_version") == "phase578_development_behavior_row.v1",
                row.get("phase_id") == "Phase578",
                row.get("model") == model,
                row.get("mode") == "development",
                row.get("split") == "development",
                row.get("execution_repeat") in REPEATS,
                row.get("activation_collected") is False,
                row.get("hidden_states_requested") is False,
                row.get("attentions_requested") is False,
                row.get("causal_intervention") is False,
                isinstance(row.get("case_id"), str),
                isinstance(row.get("input_token_ids"), list),
                isinstance(row.get("generated_token_ids_before_eos"), list),
                isinstance(row.get("full_generated_suffix_token_ids"), list),
            )):
                raise RuntimeError(f"invalid Phase578 raw replay row: {model}")
            grouped[row["case_id"]].append({
                "schema_version": row["schema_version"],
                "source_raw_row_sha256": sha256_bytes(raw),
                "execution_repeat": row["execution_repeat"],
                "model": model,
                "source_case_record_sha256": row["source_case_record_sha256"],
                "rendered_prompt_sha256": row["rendered_prompt_sha256"],
                "input_token_ids": row["input_token_ids"],
                "input_token_ids_sha256": row["input_token_ids_sha256"],
                "input_token_count": row["input_token_count"],
                "attention_mask_valid_tokens": row["attention_mask_valid_tokens"],
                "batch_index": row["batch_index"],
                "batch_row_index": row["batch_row_index"],
                "batch_padded_prompt_width": row["batch_padded_prompt_width"],
                "generation_contract_sha256": row["generation_contract_sha256"],
                "effective_eos_token_ids": row["effective_eos_token_ids"],
                "pad_token_id": row["pad_token_id"],
                "generated_token_ids_before_eos": row[
                    "generated_token_ids_before_eos"
                ],
                "full_generated_suffix_token_ids": row[
                    "full_generated_suffix_token_ids"
                ],
                "generated_token_count_before_eos": row[
                    "generated_token_count_before_eos"
                ],
                "termination_event": row["termination_event"],
                "eos_seen": row["eos_seen"],
                "first_eos_index": row["first_eos_index"],
                "first_eos_token_id": row["first_eos_token_id"],
                "budget_truncated": row["budget_truncated"],
                "post_eos_token_ids": row["post_eos_token_ids"],
                "post_eos_tokens_all_pad": row["post_eos_tokens_all_pad"],
            })
    if len(grouped) != 336 or sum(map(len, grouped.values())) != 672:
        raise RuntimeError(f"Phase578 raw replay denominator drift: {model}")
    for case_id, records in grouped.items():
        records.sort(key=lambda item: REPEATS.index(item["execution_repeat"]))
        if [item["execution_repeat"] for item in records] != list(REPEATS):
            raise RuntimeError(f"repeat closure drift: {model}/{case_id}")
        replay_cores = [
            {
                key: value for key, value in record.items()
                if key not in {"source_raw_row_sha256", "execution_repeat"}
            }
            for record in records
        ]
        if replay_cores[0] != replay_cores[1]:
            raise RuntimeError(
                f"Phase578 deterministic replay mismatch: {model}/{case_id}"
            )
    return dict(grouped)


def build_manifest() -> tuple[list[dict[str, Any]], str]:
    source_rows = read_jsonl_raw(DEVELOPMENT_PATH)
    if len(source_rows) != 336:
        raise RuntimeError("Phase577 development denominator drift")

    decisions = {model: read_json(DECISION_PATHS[model]) for model in MODEL_ORDER}
    raw_replay = {model: _raw_replay_rows(model) for model in ELIGIBLE_MODELS}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, (raw, source) in enumerate(source_rows):
        case_id = source.get("case_id")
        if not all((
            source.get("schema_version")
            == "phase577_gpt5_natural_behavior_case.v2",
            source.get("phase_id") == "Phase577",
            source.get("split") == "development",
            source.get("sealed") is False,
            isinstance(case_id, str),
            case_id not in seen,
            isinstance(source.get("raw_prompt"), str),
            isinstance(source.get("raw_role_char_spans"), dict),
        )):
            raise RuntimeError(f"invalid Phase577 development row: {ordinal}")
        seen.add(case_id)
        behavior: dict[str, Any] = {}
        for model in MODEL_ORDER:
            report = decisions[model]["case_reports"].get(case_id)
            if not isinstance(report, dict) or report.get("case_id") != case_id:
                raise RuntimeError(f"missing audited behavior row: {model}/{case_id}")
            behavior[model] = {
                "behavior_gate_pass": decisions[model]["behavior_gate_pass"],
                "decision_sha256": UPSTREAM_EXPECTED[
                    f"phase578_{model}_decision"
                ][1],
                "case_report": report,
            }
        replay = {model: raw_replay[model][case_id] for model in ELIGIBLE_MODELS}
        for model, records in replay.items():
            if any(
                item["source_case_record_sha256"] != sha256_bytes(raw)
                for item in records
            ):
                raise RuntimeError(f"raw/source case bridge drift: {model}/{case_id}")

        rows.append({
            "schema_version": "phase579_development_residual_manifest_row.v1",
            "phase_id": PHASE,
            "source_phase_id": "Phase577",
            "source_behavior_phase_id": "Phase578",
            "split": "development",
            "ordinal": ordinal,
            "case_id": case_id,
            "analysis_unit_id": source["analysis_unit_id"],
            "raw_prompt": source["raw_prompt"],
            "normalized_prompt_sha256": source["normalized_prompt_sha256"],
            "source_case_record_sha256": sha256_bytes(raw),
            "raw_role_char_spans": source["raw_role_char_spans"],
            "relation": source["relation"],
            "relation_contract_id": source["relation_contract_id"],
            "interface": source["interface"],
            "surface_id": source["surface_id"],
            "paraphrase_id": source["paraphrase_id"],
            "order": source["order"],
            "output_contract": source["output_contract"],
            "query_polarity": source["query_polarity"],
            "target_truth_polarity": source["target_truth_polarity"],
            "focus_object": source["focus_object"],
            "focus_object_class": source["focus_object_class"],
            "comparison_object": source["comparison_object"],
            "comparison_object_class": source["comparison_object_class"],
            "positive_object": source["positive_object"],
            "negative_object": source["negative_object"],
            "target": source["target"],
            "foil": source["foil"],
            "candidate_groups": source["candidate_groups"],
            "left_option": source["left_option"],
            "right_option": source["right_option"],
            "source_case_metadata": source,
            "phase578_behavior": behavior,
            "raw_replay": replay,
            "candidate_coordinates": [],
            "candidate_mechanism_formulas": [],
        })

    if set(raw_replay["qwen3"]) != seen or set(raw_replay["glm4"]) != seen:
        raise RuntimeError("eligible-model replay case closure drift")
    payload = jsonl_bytes(rows)
    return rows, sha256_bytes(payload)


def build_protocol(created: str) -> dict[str, Any]:
    upstream = verify_upstream()
    runtime = verify_formal_interpreter()
    sources = source_identities()
    artifacts = frozen_model_artifact_identities()
    manifest, manifest_hash = build_manifest()
    engineering_rows = manifest[:8]
    phase578_protocol = read_json(
        UPSTREAM_EXPECTED["phase578_protocol"][0]
    )
    return {
        "schema_version": "phase579_preregistered_residual_protocol.v1",
        "phase_id": PHASE,
        "created_at_utc": created,
        "research_role": (
            "single-model, observer-only, hypothesis-free development residual "
            "trace discovery preregistration"
        ),
        "source_identities": sources,
        "upstream_identities": upstream,
        "formal_runtime_identity": runtime,
        "engine_identity": upstream["cross_model_engine"],
        "model_registry_identity": upstream["model_registry"],
        "frozen_model_artifact_identities": artifacts,
        "frozen_tokenizer_input_identities": phase578_protocol[
            "frozen_tokenizer_input_identities"
        ],
        "development_residual_manifest": {
            "filename": MANIFEST_NAME,
            "row_count": len(manifest),
            "sha256": manifest_hash,
            "source_development_sha256": upstream[
                "phase577_development"
            ]["sha256"],
            "contains_all_source_case_metadata": True,
            "contains_two_repeat_raw_replay_for_each_eligible_model": True,
            "eligible_model_replay_row_count_each": 672,
            "candidate_coordinates": [],
            "candidate_mechanism_formulas": [],
            "top_level_fields": list(manifest[0]),
            "source_case_metadata_fields": list(
                manifest[0]["source_case_metadata"]
            ),
        },
        "models_in_historical_order": list(MODEL_ORDER),
        "eligible_models": list(ELIGIBLE_MODELS),
        "models_in_required_order": list(ELIGIBLE_MODELS),
        "future_single_model_trace_eligible_models": list(ELIGIBLE_MODELS),
        "models_in_required_future_order": list(ELIGIBLE_MODELS),
        "behavior_blocked_models": list(BLOCKED_MODELS),
        "blocked_model_receipt": {
            "model": "deepseek7b",
            "reason": "Phase578 development behavior gate failed",
            "behavior_gate_pass": False,
            "trace_authorized": False,
            "decision_sha256": upstream[
                "phase578_deepseek7b_decision"
            ]["sha256"],
        },
        "cross_model_internal_comparison_authorized": False,
        "split_access_policy": {
            "development_case_count": 336,
            "development_only": True,
            "confirmation_accessed": False,
            "confirmation_authorized": False,
            "heldout_accessed": False,
            "heldout_authorized": False,
            "sealed_accessed": False,
            "sealed_authorized": False,
            "future_split_paths_named_by_protocol": False,
        },
        "phase578_raw_replay_contract": {
            "models": list(ELIGIBLE_MODELS),
            "behavior_replay_records_available": list(REPEATS),
            "behavior_replay_record_used_for_trace": "repeat1",
            "behavior_replay_repeat_count": len(REPEATS),
            "input_token_ids_must_match": True,
            "rendered_prompt_sha256_must_match": True,
            "generated_token_ids_before_eos_must_match": True,
            "full_generated_suffix_token_ids_must_match": True,
            "eos_and_budget_event_must_match": True,
            "source_raw_rows_sha256": {
                model: upstream[f"phase578_{model}_raw_rows"]["sha256"]
                for model in ELIGIBLE_MODELS
            },
            "trace_blocked_on_any_replay_mismatch": True,
            "mismatched_trace_may_not_enter_inventory": True,
        },
        "engineering_qualification_contract": {
            "authorized_by_freeze": True,
            "required_before_full_development_trace": True,
            "models_in_serial_order": list(ELIGIBLE_MODELS),
            "case_ordinals": list(range(8)),
            "case_ids": [row["case_id"] for row in engineering_rows],
            "case_count_per_model": 8,
            "phase578_behavior_reference_repeat": "repeat1",
            "internal_reexecution_count": 2,
            "serialized_trace_copy_count": 1,
            "phase578_generation_exact_replay_required": True,
            "all_layer_hidden_state_presence_required": True,
            "tensor_shape_validation_required": True,
            "all_values_finite_required": True,
            "bfloat16_serialization_roundtrip_required": True,
            "strict_cuda_cleanup_between_models_required": True,
            "allocated_bytes_after_cleanup_required": 0,
            "reserved_bytes_after_cleanup_required": 0,
            "automatic_fallback": False,
            "full_trace_authorized_without_qualification": False,
        },
        "full_development_trace_contract": {
            "conditionally_authorized_only_after_engineering_passes": True,
            "models_in_serial_order": list(ELIGIBLE_MODELS),
            "case_count_per_model": 336,
            "include_behavior_success_cases": True,
            "include_behavior_failure_cases": True,
            "phase578_behavior_reference_repeat": "repeat1",
            "internal_reexecution_count": 1,
            "serialized_trace_copy_count": 1,
            "observer_only": True,
            "causal_intervention": False,
            "hooks_that_modify_values": False,
            "scores_or_logits_persisted": False,
            "attentions_persisted": False,
            "all_residual_layers_persisted": True,
            "all_prompt_token_positions_persisted": True,
            "every_generated_token_actually_fed_back_as_input_persisted": True,
            "unfed_terminal_position_may_not_be_fabricated": True,
            "maximum_generated_token_count": 24,
            "feedback_axis_length": 23,
            "feedback_axis_semantics": (
                "generated token f as input to the next executed forward; "
                "the final emitted but unfed token has no residual slot"
            ),
            "raw_trace_published_before_inventory": True,
            "strict_cuda_cleanup_between_models_required": True,
            "trace_replay_gate": "phase578_raw_replay_contract",
            "preselected_layer": None,
            "preselected_head": None,
            "preselected_neuron": None,
            "preselected_direction": None,
            "preselected_token_coordinate": None,
        },
        "inventory_contract": {
            "begins_only_after_immutable_raw_trace_publication": True,
            "basic_direct_case_and_analysis_unit_counts_first": True,
            "prespecified_pca_probe_cluster_or_mechanism_formula": False,
            "candidate_registration_gate": OBSERVATION_CANDIDATE_GATE,
            "empty_result_is_valid": True,
            "empty_result_label": "empty_within_frozen_scope",
            "candidate_is_observation_not_mechanism": True,
        },
        "observation_candidate_registration_gate": OBSERVATION_CANDIDATE_GATE,
        "engineering_qualification_authorized": True,
        "full_development_trace_authorized": False,
        "full_development_trace_authorized_if_and_only_if_engineering_passes": True,
        "inventory_authorized_before_immutable_trace": False,
        "confirmation_authorized": False,
        "heldout_authorized": False,
        "sealed_authorized": False,
        "cross_model_internal_comparison_authorized": False,
        "causal_intervention_authorized": False,
        "mechanism_claim_authorized": False,
        "candidate_coordinates": [],
        "candidate_mechanism_formulas": [],
        "gpu_used": False,
        "model_weights_loaded": False,
        "internal_trace_run_count": 0,
        "scientific_limits": [
            "Phase579 freeze contains no internal observation and no candidate",
            "behavior eligibility is necessary but does not identify an internal mechanism",
            "development discovery cannot serve as independent confirmation",
            "repeat replay establishes deterministic reproducibility, not statistical independence",
            "residual coordinates are model-local and cross-model comparison is forbidden",
            "candidate thresholds are observation admission rules, not language laws",
            "a registered observation candidate is neither causal nor a mechanism formula",
        ],
        "next_required_stage": "phase579_residual_engineering_qualification",
    }


def self_test() -> dict[str, Any]:
    assert_future_roots_absent()
    upstream = verify_upstream()
    runtime = verify_formal_interpreter()
    sources = source_identities()
    manifest, manifest_hash = build_manifest()
    first_eight = manifest[:8]
    checks = {
        "upstream_count": len(upstream) == len(UPSTREAM_EXPECTED),
        "source_count": len(sources) == len(SOURCE_RELATIVES),
        "formal_interpreter": runtime["python_executable_identity"]["sha256"]
        == FORMAL_PYTHON_SHA256,
        "formal_packages": runtime["packages"] == FORMAL_PACKAGES,
        "manifest_rows": len(manifest) == 336,
        "manifest_ordinals": [row["ordinal"] for row in manifest]
        == list(range(336)),
        "manifest_unique_cases": len({row["case_id"] for row in manifest}) == 336,
        "development_only": all(
            row["split"] == "development"
            and row["source_case_metadata"]["sealed"] is False
            for row in manifest
        ),
        "all_case_metadata_present": all(
            row["source_case_metadata"]["case_id"] == row["case_id"]
            and row["raw_prompt"] == row["source_case_metadata"]["raw_prompt"]
            for row in manifest
        ),
        "two_repeats_each_eligible_model": all(
            all(
                [item["execution_repeat"] for item in row["raw_replay"][model]]
                == list(REPEATS)
                for model in ELIGIBLE_MODELS
            )
            for row in manifest
        ),
        "eligibility_exact": all(
            row["phase578_behavior"]["qwen3"]["behavior_gate_pass"] is True
            and row["phase578_behavior"]["glm4"]["behavior_gate_pass"] is True
            and row["phase578_behavior"]["deepseek7b"]["behavior_gate_pass"]
            is False
            for row in manifest
        ),
        "no_candidates": all(
            row["candidate_coordinates"] == []
            and row["candidate_mechanism_formulas"] == []
            for row in manifest
        ),
        "engineering_case_lock": [row["ordinal"] for row in first_eight]
        == list(range(8)),
        "candidate_gate_integer_contract": all((
            OBSERVATION_CANDIDATE_GATE["minimum_unit_vectors"] == 6,
            OBSERVATION_CANDIDATE_GATE[
                "cross_unit_positive_dot_fraction_numerator"
            ] == 4,
            OBSERVATION_CANDIDATE_GATE[
                "cross_unit_positive_dot_fraction_denominator"
            ] == 5,
            OBSERVATION_CANDIDATE_GATE[
                "minimum_reproducing_control_slices"
            ] == 2,
            OBSERVATION_CANDIDATE_GATE[
                "minimum_distinct_control_slices"
            ] == 2,
            OBSERVATION_CANDIDATE_GATE[
                "minimum_tested_invariant_dimensions"
            ] == 2,
            OBSERVATION_CANDIDATE_GATE[
                "post_discovery_cross_unit_direction_check"
            ]["minimum_eligible_unit_count"]
            == OBSERVATION_CANDIDATE_GATE["minimum_unit_vectors"],
            OBSERVATION_CANDIDATE_GATE[
                "post_discovery_cross_unit_direction_check"
            ]["equivalent_fraction_floor"] == "4/5",
        )),
        "candidate_scope_and_matched_pair_contract": all((
            OBSERVATION_CANDIDATE_GATE["evidence_unit_by_axis"] == {
                "relation": "relation_focus_object_unit",
                "query_polarity": "analysis_unit_id",
                "selection_order": "analysis_unit_id",
                "output_contract": "analysis_unit_id",
                "paraphrase": "analysis_unit_id",
            },
            OBSERVATION_CANDIDATE_GATE[
                "scope_dimensions_must_be_explicit"
            ] == [
                "relation", "interface", "query_polarity", "order",
                "output_contract", "surface_id", "paraphrase_id",
                "target_truth_polarity", "target", "foil", "focus_object",
                "focus_object_class", "comparison_object",
                "comparison_object_class", "left_option", "right_option",
                "relation_contract_id",
                "positive_object", "negative_object", "analysis_unit_id",
                "token_role", "layer",
            ],
            OBSERVATION_CANDIDATE_GATE[
                "matched_pair_allowed_differences"
            ] == {
                "relation": [
                    "relation", "target_truth_polarity", "target", "foil",
                    "positive_object", "negative_object", "analysis_unit_id",
                    "relation_contract_id", "raw_prompt",
                ],
                "query_polarity": [
                    "query_polarity", "target_truth_polarity", "target",
                    "foil", "focus_object", "focus_object_class",
                    "comparison_object", "comparison_object_class", "raw_prompt",
                ],
                "selection_order": [
                    "order", "left_option", "right_option", "raw_prompt",
                ],
                "output_contract": [
                    "output_contract", "surface_id", "raw_prompt",
                ],
                "paraphrase": [
                    "paraphrase_id", "surface_id", "raw_prompt",
                ],
            },
        )),
        "phase578_behavior_repeat_count": len(REPEATS) == 2,
        "manifest_hash": len(manifest_hash) == 64,
        "future_result_roots_absent": True,
        "torch_not_imported": "torch" not in sys.modules,
        "transformers_not_imported": "transformers" not in sys.modules,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase579 protocol self-test failed: {checks}")
    return {
        "schema_version": "phase579_protocol_self_test.v1",
        "phase_id": PHASE,
        "passed": True,
        "checks": checks,
        "manifest_sha256": manifest_hash,
        "gpu_used": False,
        "model_weights_loaded": False,
        "files_written": False,
    }


def write_stage() -> dict[str, Any]:
    if OUT_DIR.exists():
        raise RuntimeError("Phase579 protocol output already exists")
    assert_future_roots_absent()
    created = now()
    protocol = build_protocol(created)
    manifest, manifest_hash = build_manifest()
    test = self_test()
    pending = OUT_DIR.with_name(
        f".{OUT_DIR.name}.pending-{os.getpid()}-{uuid.uuid4().hex}"
    )
    pending.mkdir(parents=True, exist_ok=False)
    try:
        write_exclusive(pending / MANIFEST_NAME, jsonl_bytes(manifest))
        write_exclusive(pending / PROTOCOL_NAME, json_bytes(protocol))
        write_exclusive(pending / SELF_TEST_NAME, json_bytes(test))
        initial = {
            name: identity(pending / name, pending)
            for name in (MANIFEST_NAME, PROTOCOL_NAME, SELF_TEST_NAME)
        }
        stage = {
            "schema_version": "phase579_stage_commit.v1",
            "phase_id": PHASE,
            "created_at_utc": created,
            "stage_complete": True,
            "artifact_identities": initial,
            "development_manifest_sha256": manifest_hash,
            "source_identities": protocol["source_identities"],
            "upstream_identities": protocol["upstream_identities"],
            "formal_runtime_identity": protocol["formal_runtime_identity"],
            "frozen_model_artifact_identities": protocol[
                "frozen_model_artifact_identities"
            ],
            "future_result_roots_absent_before_freeze": True,
            "gpu_used": False,
            "model_weights_loaded": False,
            "engineering_qualification_run_count": 0,
            "internal_trace_run_count": 0,
            "candidate_coordinates": [],
            "candidate_mechanism_formulas": [],
        }
        write_exclusive(pending / STAGE_COMMIT_NAME, json_bytes(stage))
        pending.rename(OUT_DIR)
    except BaseException:
        if pending.exists():
            if pending.parent.resolve(strict=True) != OUT_DIR.parent.resolve(strict=True):
                raise RuntimeError("Phase579 pending quarantine escaped result root")
            pending.rename(
                pending.with_name(f".{OUT_DIR.name}.failed-{uuid.uuid4().hex}")
            )
        raise
    return verify_stage(require_final=False)


def _exact_files(expected: set[str]) -> bool:
    if not OUT_DIR.is_dir() or OUT_DIR.is_symlink():
        return False
    members = list(OUT_DIR.rglob("*"))
    files = {
        str(path.relative_to(OUT_DIR)).replace("\\", "/")
        for path in members if path.is_file()
    }
    return (
        files == expected
        and not any(path.is_dir() for path in members)
        and not any(path.is_symlink() for path in members)
    )


def independent_audit_verification() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    process = subprocess.run(
        [
            str(FORMAL_PYTHON),
            str(ROOT / "tests/glm5/phase579_gpt5_residual_audit.py"),
            "--verify-freeze-audit",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "Phase579 independent freeze-audit verifier failed: "
            + process.stderr
        )
    payload = json.loads(process.stdout)
    if not all((
        isinstance(payload, dict),
        payload.get("passed") is True,
        payload.get("gpu_used") is False,
        payload.get("model_weights_loaded") is False,
    )):
        raise RuntimeError("Phase579 independent audit verifier did not pass")
    return payload


def verify_stage(
    require_final: bool,
    allow_audit: bool = False,
) -> dict[str, Any]:
    expected = (
        FINAL_FILES if require_final
        else INITIAL_FILES | ({AUDIT_NAME} if allow_audit else set())
    )
    if not _exact_files(expected):
        raise RuntimeError(f"Phase579 exact artifact closure failed: {expected}")
    protocol = read_json(OUT_DIR / PROTOCOL_NAME)
    stage = read_json(OUT_DIR / STAGE_COMMIT_NAME)
    test = read_json(OUT_DIR / SELF_TEST_NAME)
    manifest, manifest_hash = build_manifest()
    sources = source_identities()
    upstream = verify_upstream()
    runtime = verify_formal_interpreter()
    artifacts = frozen_model_artifact_identities()
    checks = {
        "protocol_schema": protocol.get("schema_version")
        == "phase579_preregistered_residual_protocol.v1",
        "phase": protocol.get("phase_id") == PHASE,
        "sources": protocol.get("source_identities") == sources,
        "upstream": protocol.get("upstream_identities") == upstream,
        "formal_runtime": protocol.get("formal_runtime_identity") == runtime,
        "model_artifacts": protocol.get("frozen_model_artifact_identities")
        == artifacts,
        "eligibility": protocol.get("future_single_model_trace_eligible_models")
        == list(ELIGIBLE_MODELS)
        and protocol.get("eligible_models") == list(ELIGIBLE_MODELS)
        and protocol.get("models_in_required_order") == list(ELIGIBLE_MODELS),
        "blocked": protocol.get("behavior_blocked_models")
        == list(BLOCKED_MODELS),
        "candidate_gate": protocol.get(
            "observation_candidate_registration_gate"
        ) == OBSERVATION_CANDIDATE_GATE
        and protocol.get("inventory_contract", {}).get(
            "candidate_registration_gate"
        ) == OBSERVATION_CANDIDATE_GATE,
        "cross_model_false": protocol.get(
            "cross_model_internal_comparison_authorized"
        ) is False,
        "manifest_bytes": (OUT_DIR / MANIFEST_NAME).read_bytes()
        == jsonl_bytes(manifest),
        "manifest_hash": protocol["development_residual_manifest"]["sha256"]
        == manifest_hash,
        "saved_self_test": test.get("schema_version")
        == "phase579_protocol_self_test.v1"
        and test.get("passed") is True
        and all(test.get("checks", {}).values()),
        "stage_schema": stage.get("schema_version") == "phase579_stage_commit.v1",
        "stage_complete": stage.get("stage_complete") is True,
        "stage_artifact_identities": stage.get("artifact_identities") == {
            name: identity(OUT_DIR / name, OUT_DIR)
            for name in (MANIFEST_NAME, PROTOCOL_NAME, SELF_TEST_NAME)
        },
        "stage_manifest_hash": stage.get("development_manifest_sha256")
        == sha256_file(OUT_DIR / MANIFEST_NAME),
        "stage_sources": stage.get("source_identities") == sources,
        "stage_upstream": stage.get("upstream_identities") == upstream,
        "stage_formal_runtime": stage.get("formal_runtime_identity") == runtime,
        "stage_model_artifacts": stage.get("frozen_model_artifact_identities")
        == artifacts,
        "historical_future_absence": stage.get(
            "future_result_roots_absent_before_freeze"
        ) is True,
        "no_stage_runs_or_candidates": stage.get(
            "engineering_qualification_run_count"
        ) == 0
        and stage.get("internal_trace_run_count") == 0
        and stage.get("candidate_coordinates") == []
        and stage.get("candidate_mechanism_formulas") == [],
        "no_model_modules": "torch" not in sys.modules
        and "transformers" not in sys.modules,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase579 stage verification failed: {checks}")

    if require_final:
        audit = read_json(OUT_DIR / AUDIT_NAME)
        freeze = read_json(OUT_DIR / FREEZE_NAME)
        audit_verification = independent_audit_verification()
        final_checks = {
            "audit_schema": audit.get("schema_version")
            == "phase579_independent_freeze_audit.v1",
            "audit_passed": audit.get("passed") is True
            and all(audit.get("checks", {}).values()),
            "freeze_schema": freeze.get("schema_version")
            == "phase579_freeze_commit.v1",
            "freeze_complete": freeze.get("freeze_complete") is True,
            "freeze_protocol_hash": freeze.get("protocol_sha256")
            == sha256_file(OUT_DIR / PROTOCOL_NAME),
            "freeze_stage_hash": freeze.get("stage_commit_sha256")
            == sha256_file(OUT_DIR / STAGE_COMMIT_NAME),
            "freeze_manifest_hash": freeze.get("development_manifest_sha256")
            == sha256_file(OUT_DIR / MANIFEST_NAME),
            "freeze_self_test_hash": freeze.get("self_test_sha256")
            == sha256_file(OUT_DIR / SELF_TEST_NAME),
            "freeze_audit_hash": freeze.get("independent_audit_sha256")
            == sha256_file(OUT_DIR / AUDIT_NAME),
            "audit_verifier_payload": freeze.get(
                "independent_audit_verification_payload_sha256"
            ) == sha256_bytes(
                canonical_json(audit_verification).encode("utf-8")
            ),
            "freeze_sources": freeze.get("source_identities") == sources,
            "freeze_upstream": freeze.get("upstream_identities") == upstream,
            "freeze_runtime": freeze.get("formal_runtime_identity") == runtime,
            "freeze_model_artifacts": freeze.get(
                "frozen_model_artifact_identities"
            ) == artifacts,
            "eligible_models": freeze.get("eligible_models")
            == list(ELIGIBLE_MODELS),
            "blocked_models": freeze.get("behavior_blocked_models")
            == list(BLOCKED_MODELS),
            "engineering_authorized": freeze.get(
                "engineering_qualification_authorized"
            ) is True,
            "full_trace_conditional": freeze.get(
                "full_development_trace_authorized"
            ) is False
            and freeze.get(
                "full_development_trace_authorized_if_and_only_if_engineering_passes"
            ) is True,
            "future_split_and_cross_model_false": all(
                freeze.get(field) is False for field in (
                    "confirmation_authorized", "heldout_authorized",
                    "sealed_authorized",
                    "cross_model_internal_comparison_authorized",
                    "causal_intervention_authorized",
                    "mechanism_claim_authorized",
                )
            ),
            "no_runs_or_candidates": freeze.get(
                "engineering_qualification_run_count"
            ) == 0
            and freeze.get("internal_trace_run_count") == 0
            and freeze.get("candidate_coordinates") == []
            and freeze.get("candidate_mechanism_formulas") == [],
            "next_stage": freeze.get("next_required_stage")
            == "phase579_residual_engineering_qualification",
        }
        checks.update(final_checks)
        if not all(final_checks.values()):
            raise RuntimeError(f"Phase579 final verification failed: {final_checks}")
    return {
        "schema_version": "phase579_protocol_verification.v1",
        "phase_id": PHASE,
        "passed": True,
        "checks": checks,
        "gpu_used": False,
        "model_weights_loaded": False,
        "files_written": False,
    }


def finalize() -> dict[str, Any]:
    assert_future_roots_absent()
    verify_stage(require_final=False, allow_audit=True)
    audit_path = OUT_DIR / AUDIT_NAME
    if not audit_path.is_file():
        raise RuntimeError("Phase579 independent freeze audit is required")
    audit = read_json(audit_path)
    if not all((
        audit.get("schema_version") == "phase579_independent_freeze_audit.v1",
        audit.get("passed") is True,
        all(audit.get("checks", {}).values()),
        audit.get("gpu_used") is False,
        audit.get("model_weights_loaded") is False,
    )):
        raise RuntimeError("Phase579 independent freeze audit did not pass")
    audit_verification = independent_audit_verification()
    freeze_path = OUT_DIR / FREEZE_NAME
    if freeze_path.exists():
        raise RuntimeError("Phase579 final freeze already exists")
    protocol = read_json(OUT_DIR / PROTOCOL_NAME)
    freeze = {
        "schema_version": "phase579_freeze_commit.v1",
        "phase_id": PHASE,
        "created_at_utc": now(),
        "freeze_complete": True,
        "protocol_sha256": sha256_file(OUT_DIR / PROTOCOL_NAME),
        "stage_commit_sha256": sha256_file(OUT_DIR / STAGE_COMMIT_NAME),
        "development_manifest_sha256": sha256_file(OUT_DIR / MANIFEST_NAME),
        "self_test_sha256": sha256_file(OUT_DIR / SELF_TEST_NAME),
        "independent_audit_sha256": sha256_file(audit_path),
        "independent_audit_verification_payload_sha256": sha256_bytes(
            canonical_json(audit_verification).encode("utf-8")
        ),
        "source_identities": protocol["source_identities"],
        "upstream_identities": protocol["upstream_identities"],
        "formal_runtime_identity": protocol["formal_runtime_identity"],
        "frozen_model_artifact_identities": protocol[
            "frozen_model_artifact_identities"
        ],
        "eligible_models": list(ELIGIBLE_MODELS),
        "behavior_blocked_models": list(BLOCKED_MODELS),
        "models_in_required_future_order": list(ELIGIBLE_MODELS),
        "engineering_case_ordinals": list(range(8)),
        "engineering_case_ids": protocol[
            "engineering_qualification_contract"
        ]["case_ids"],
        "engineering_qualification_authorized": True,
        "engineering_qualification_run_count": 0,
        "full_development_trace_authorized": False,
        "full_development_trace_authorized_if_and_only_if_engineering_passes": True,
        "internal_trace_run_count": 0,
        "inventory_authorized_before_immutable_trace": False,
        "confirmation_authorized": False,
        "heldout_authorized": False,
        "sealed_authorized": False,
        "cross_model_internal_comparison_authorized": False,
        "causal_intervention_authorized": False,
        "mechanism_claim_authorized": False,
        "candidate_coordinates": [],
        "candidate_mechanism_formulas": [],
        "gpu_used": False,
        "model_weights_loaded": False,
        "next_required_stage": "phase579_residual_engineering_qualification",
    }
    write_exclusive(freeze_path, json_bytes(freeze))
    return verify_stage(require_final=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify-stage", action="store_true")
    group.add_argument("--finalize", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.write:
        result = write_stage()
    elif args.verify_stage:
        result = verify_stage(require_final=False)
    elif args.finalize:
        result = finalize()
    else:
        result = verify_stage(require_final=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
