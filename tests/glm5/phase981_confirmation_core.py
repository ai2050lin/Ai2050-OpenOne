#!/usr/bin/env python3
"""Shared, holdout-free definitions for the Phase 981 fresh confirmation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"

import phase979_boundary_core as boundary


PHASE = 981
SCHEMA_VERSION = 1
EXPERIMENT = "fresh256_soft_configuration_semantic_confirmation"
MODEL_NAME = "qwen3"
ITEM_COUNT = 256
TASK_COUNT = 8
ITEMS_PER_TASK = 32
ITEMS_PER_DIFFICULTY = 128
STREAMS = (0, 1, 2)
BATCH_SIZE = 8
MAX_NEW_TOKENS = 2048
CHECKPOINTS = (256, 512, 1024, 1536, 2048)
DECISION_CHECKPOINT = 2048
EXPECTED_ROWS = ITEM_COUNT * 2 * len(STREAMS)

ARM_A = "A"
ARM_B = "B"
ARMS: dict[str, dict[str, str]] = {
    ARM_A: {
        "control_policy": "soft_no_think",
        "decoding_policy": "thinking_sampling",
        "role": "baseline",
    },
    ARM_B: {
        "control_policy": "soft_thinking",
        "decoding_policy": "thinking_sampling",
        "role": "candidate",
    },
}
PRIMARY_DIRECTION = "B_minus_A"
SAMPLING = dict(boundary.DECODING_POLICIES["thinking_sampling"])

SIX_STATES = tuple(boundary.TERMINAL_STATES)
FOUR_CHANNELS = ("V", "C", "I_mode", "I_sem")
THREE_CHANNELS = ("V", "C", "I")
DIFFICULTIES = ("easy", "hard")

OUT = ROOT / "tests" / "glm5" / "result" / "phase981_fresh256_confirmation"
DATASET_ARTIFACT_PATH = OUT / "dataset.json"
DATASET_AUDIT_PATH = OUT / "audit.json"
PROTOCOL_PATH = OUT / "protocol_preregistration.json"
ADMISSION_PATH = OUT / "generation_admission.json"
MANIFEST_PATH = OUT / "manifest_confirmation.json"
ROWS_PATH = OUT / "rows_confirmation.jsonl"
STATUS_PATH = OUT / "generator_status_confirmation.json"
AUDIT_PATH = OUT / "confirmation_audit.json"
RUN_LOCK_PATH = OUT / "confirmation_runner.lock"

PHASE981_SCRIPT_PATHS = {
    "core": "tests/glm5/phase981_confirmation_core.py",
    "dataset": "tests/glm5/phase981_fresh_dataset.py",
    "semantic_gate": "tests/glm5/phase981_semantic_gate.py",
    "protocol": "tests/glm5/phase981_confirmation_protocol.py",
    "admission": "tests/glm5/phase981_confirmation_admission.py",
    "runner": "tests/glm5/phase981_confirmation_runner.py",
    "audit": "tests/glm5/phase981_confirmation_audit.py",
}

# Complete repository-local import closure that can affect Phase981 protocol
# construction, prompt/classification semantics, or Qwen model selection/loading.
RUNTIME_DEPENDENCY_PATHS = {
    "phase979_boundary_core": "tests/glm5/phase979_boundary_core.py",
    "phase979_diagnostic_dataset": "tests/glm5/phase979_diagnostic_dataset.py",
    "phase980_rescue_gate_feasibility": "tests/glm5/phase980_rescue_gate_feasibility.py",
    "model_utils": "tests/glm5/model_utils.py",
    "model_registry": "tests/gpt5/model_registry.py",
}

PHASE979_SCRIPT_PATHS = {
    "boundary_core": "tests/glm5/phase979_boundary_core.py",
    "natural_auditor": "tests/glm5/phase979_natural_audit.py",
    "natural_dataset": "tests/glm5/phase979_diagnostic_dataset.py",
    "natural_runner": "tests/glm5/phase979_natural_runner.py",
    "protocol": "tests/glm5/phase979_protocol.py",
    "truth_auditor": "tests/glm5/phase979_truth_audit.py",
    "truth_dataset": "tests/glm5/phase979_truth_punctuation_dataset.py",
    "truth_runner": "tests/glm5/phase979_truth_punctuation.py",
}

MODEL_ARTIFACT_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
EXPECTED_EOS_TOKEN_IDS = (151643, 151645)
EXPECTED_TOKENIZER_EOS_ID = 151645
EXPECTED_THINK_OPEN_ID = 151667
EXPECTED_THINK_CLOSE_ID = 151668
EXPECTED_A_ID = 32
EXPECTED_B_ID = 33

AUTHORIZATION_SCOPE = {
    "model": MODEL_NAME,
    "experiment": EXPERIMENT,
    "expected_rows": EXPECTED_ROWS,
    "arms": ARMS,
    "streams": list(STREAMS),
    "internal_activations": False,
    "layer_span_cross_time_interventions": False,
}

PROTOCOL_EXECUTION_CONTRACT = {
    "cpu_only_protocol_freeze": True,
    "model_weights_loaded": False,
    "generation_performed": False,
    "gpu_authorized": False,
    "generation_requires_independent_admission": True,
}

GENERATION_CONTRACT = {
    "single_rollout_prefix_checkpoints": True,
    "first_eos_absorbing": True,
    "private_generator_per_row": True,
    "same_item_stream_A_B_seed": True,
    "arm_excluded_from_seed": True,
    "left_padding_with_explicit_attention_mask": True,
    "decision_withheld_for_independent_cpu_audit": True,
}

ROW_GENERATION_CONTRACT = {
    "generation_performed": True,
    "first_actual_eos_absorbing": True,
    "single_rollout_prefix_checkpoints": True,
    "private_generator_per_row": True,
    "same_pair_seed_across_arms": True,
    "holdout": False,
    "holdout_loaded": False,
    "mechanism": False,
    "mechanism_authorized": False,
}

INTEGRITY_NEGATIVE_TEST_KEYS = {
    "empty_phase981_script_seals_rejected",
    "changed_dependency_hash_rejected",
    "empty_phase979_script_hashes_rejected",
    "empty_model_file_registry_rejected",
    "expanded_intervention_scope_rejected",
    "changed_eos_identity_rejected",
    "protocol_holdout_loaded_true_rejected",
    "protocol_gpu_authorized_true_rejected",
    "admission_holdout_loaded_true_rejected",
}

MODEL_SCOPE_CONTRACT = {
    "qwen3_only": True,
    "configuration_bundle_not_component_isolation": True,
    "direct_generalization_to_GLM4_forbidden": True,
    "direct_generalization_to_DeepSeek7B_forbidden": True,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_file_seals(
    expected_paths: dict[str, str], label: str,
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for name, relative_path in expected_paths.items():
        path = ROOT / relative_path
        require(path.is_file(), f"missing {label}: {relative_path}")
        output[name] = {"path": relative_path, "sha256": sha256_file(path)}
    return output


def verify_file_seals(
    seals: Any, expected_paths: dict[str, str], label: str,
) -> dict[str, str]:
    require(isinstance(seals, dict), f"{label} registry missing")
    require(set(seals) == set(expected_paths),
            f"{label} key registry changed")
    verified: dict[str, str] = {}
    for name, relative_path in expected_paths.items():
        seal = seals.get(name)
        require(isinstance(seal, dict) and set(seal) == {"path", "sha256"},
                f"{label} seal malformed: {name}")
        require(seal.get("path") == relative_path,
                f"{label} path changed: {name}")
        path = ROOT / relative_path
        expected = seal.get("sha256")
        require(isinstance(expected, str) and len(expected) == 64
                and path.is_file() and sha256_file(path) == expected,
                f"{label} file changed: {name}")
        verified[name] = expected
    return verified


def verify_protocol_file_seals(protocol: dict[str, Any]) -> dict[str, Any]:
    phase979_source = protocol.get("phase979_source")
    require(isinstance(phase979_source, dict), "Phase979 source registry missing")
    scripts = verify_file_seals(
        protocol.get("script_seals"), PHASE981_SCRIPT_PATHS,
        "Phase981 script",
    )
    dependencies = verify_file_seals(
        protocol.get("dependency_seals"), RUNTIME_DEPENDENCY_PATHS,
        "Phase981 runtime dependency",
    )
    phase979 = verify_file_seals(
        phase979_source.get("phase979_script_hashes"), PHASE979_SCRIPT_PATHS,
        "Phase979 authenticated script",
    )
    return {
        "script_seals_sha256": canonical_sha256(protocol["script_seals"]),
        "dependency_seals_sha256": canonical_sha256(protocol["dependency_seals"]),
        "phase979_script_hashes_sha256": canonical_sha256(
            phase979_source["phase979_script_hashes"]),
        "verified_script_count": len(scripts),
        "verified_dependency_count": len(dependencies),
        "verified_phase979_script_count": len(phase979),
    }


def verify_model_artifact_identity(identity: Any) -> dict[str, Any]:
    require(isinstance(identity, dict), "model artifact identity missing")
    require(identity.get("logical_name") == MODEL_NAME
            and identity.get("path") == "models/hf/qwen3-4b"
            and identity.get("weights_loaded") is False
            and identity.get("gpu_accessed") is False,
            "Qwen model identity boundary changed")
    files = identity.get("files")
    require(isinstance(files, dict) and set(files) == set(MODEL_ARTIFACT_FILES),
            "Qwen model artifact file registry changed")
    require(identity.get("identity_sha256") == canonical_sha256(files),
            "Qwen model artifact identity self-hash invalid")
    root = ROOT / str(identity["path"])
    require(root.resolve() == (ROOT / "models/hf/qwen3-4b").resolve()
            and root.is_dir(), "Qwen model root changed")
    for name in MODEL_ARTIFACT_FILES:
        seal = files.get(name)
        path = root / name
        require(isinstance(seal, dict) and set(seal) == {"bytes", "sha256"}
                and isinstance(seal.get("bytes"), int)
                and isinstance(seal.get("sha256"), str)
                and len(seal["sha256"]) == 64
                and path.is_file()
                and path.stat().st_size == seal["bytes"]
                and sha256_file(path) == seal["sha256"],
                f"Qwen model artifact changed: {name}")
    return identity


def token_identity_from_artifacts(tokenizer_eos_id: Any) -> dict[str, Any]:
    model_root = ROOT / "models/hf/qwen3-4b"
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    generation = json.loads(
        (model_root / "generation_config.json").read_text(encoding="utf-8"))

    def values(value: Any) -> set[int]:
        if isinstance(value, int) and not isinstance(value, bool):
            return {int(value)}
        require(isinstance(value, list) and value
                and all(isinstance(item, int) and not isinstance(item, bool)
                        for item in value), "malformed EOS token registry")
        return {int(item) for item in value}

    require(tokenizer_eos_id == EXPECTED_TOKENIZER_EOS_ID,
            "tokenizer EOS identity changed")
    config_ids = values(config.get("eos_token_id"))
    generation_ids = values(generation.get("eos_token_id"))
    union = sorted({int(tokenizer_eos_id)} | config_ids | generation_ids)
    require(config_ids == {EXPECTED_TOKENIZER_EOS_ID}
            and generation_ids == set(EXPECTED_EOS_TOKEN_IDS)
            and union == list(EXPECTED_EOS_TOKEN_IDS),
            "Qwen config/generation EOS identities changed")
    return {
        "tokenizer_eos_token_id": int(tokenizer_eos_id),
        "model_config_eos_token_ids": sorted(config_ids),
        "generation_config_eos_token_ids": sorted(generation_ids),
        "effective_eos_token_ids": union,
        "identity_source": (
            "independently parsed tokenizer, config.json, and generation_config.json"
        ),
    }


def verify_think_and_answer_token_ids(
    think_open_id: Any, think_close_id: Any, a_id: Any, b_id: Any,
) -> None:
    require((think_open_id, think_close_id, a_id, b_id) == (
        EXPECTED_THINK_OPEN_ID, EXPECTED_THINK_CLOSE_ID,
        EXPECTED_A_ID, EXPECTED_B_ID,
    ), "Qwen think/A/B token identity changed")


def verify_protocol_token_identity(
    tokenizer_audit: Any, tokenizer_eos_id: Any, think_open_id: Any,
    think_close_id: Any, a_id: Any, b_id: Any,
) -> dict[str, Any]:
    require(isinstance(tokenizer_audit, dict),
            "protocol tokenizer audit missing")
    artifact_identity = token_identity_from_artifacts(tokenizer_eos_id)
    verify_think_and_answer_token_ids(think_open_id, think_close_id, a_id, b_id)
    require(tokenizer_audit.get("token_identity") == artifact_identity
            and tokenizer_audit.get("think_open_id") == think_open_id
            and tokenizer_audit.get("think_close_id") == think_close_id
            and tokenizer_audit.get("A_id") == a_id
            and tokenizer_audit.get("B_id") == b_id,
            "protocol token identity differs from independent derivation")
    return artifact_identity


def verify_authorization_scope(scope: Any) -> None:
    require(scope == AUTHORIZATION_SCOPE,
            "Qwen-only admission authorization scope changed")


def verify_protocol_boundary_contract(protocol: dict[str, Any]) -> None:
    require(protocol.get("holdout") is False
            and protocol.get("holdout_loaded") is False
            and protocol.get("holdout_authorized") is False
            and protocol.get("mechanism") is False
            and protocol.get("mechanism_authorized") is False,
            "protocol holdout/mechanism boundary changed")
    require(protocol.get("execution_contract") == PROTOCOL_EXECUTION_CONTRACT,
            "protocol CPU-freeze/pre-generation boundary changed")


def verify_admission_boundary_contract(admission: dict[str, Any]) -> None:
    require(admission.get("decision") == "ADMIT_QWEN_EXTERNAL_GENERATION"
            and admission.get("admitted") is True
            and admission.get("qwen_external_generation_authorized") is True
            and admission.get("gpu_authorized") is True,
            "admission authorization decision changed")
    verify_authorization_scope(admission.get("authorization_scope"))
    require(admission.get("model_weights_loaded") is False
            and admission.get("generation_performed") is False
            and admission.get("gpu_used") is False,
            "admission pre-generation state changed")
    require(admission.get("holdout") is False
            and admission.get("holdout_loaded") is False
            and admission.get("holdout_authorized") is False
            and admission.get("mechanism") is False
            and admission.get("mechanism_authorized") is False,
            "admission holdout/mechanism boundary changed")


def verify_protocol_integrity_metadata(protocol: dict[str, Any]) -> None:
    integrity = protocol.get("integrity_contract")
    require(isinstance(integrity, dict) and set(integrity) == {
        "script_seals_exact",
        "runtime_dependency_seals_exact",
        "phase979_script_hashes_persisted_and_exact",
        "model_artifact_file_registry_exact_and_nonempty",
        "negative_tamper_tests",
    }, "protocol integrity metadata registry changed")
    tests = integrity.get("negative_tamper_tests")
    require(integrity.get("script_seals_exact") is True
            and integrity.get("runtime_dependency_seals_exact") is True
            and integrity.get("phase979_script_hashes_persisted_and_exact") is True
            and integrity.get("model_artifact_file_registry_exact_and_nonempty") is True
            and isinstance(tests, dict)
            and set(tests) == INTEGRITY_NEGATIVE_TEST_KEYS
            and all(value is True for value in tests.values()),
            "protocol integrity negative tests changed")
    require(protocol.get("model_scope_contract") == MODEL_SCOPE_CONTRACT,
            "Qwen-only model scope contract changed")


def verify_manifest_dependency_contract(
    manifest: dict[str, Any], protocol: dict[str, Any],
) -> None:
    seal_audit = verify_protocol_file_seals(protocol)
    phase979_scripts = protocol["phase979_source"]["phase979_script_hashes"]
    require(manifest.get("script_seals") == protocol.get("script_seals")
            and manifest.get("dependency_seals") == protocol.get("dependency_seals")
            and manifest.get("phase979_script_hashes") == phase979_scripts,
            "manifest file seal registries differ from protocol")
    require(manifest.get("script_seals_sha256")
            == seal_audit["script_seals_sha256"]
            and manifest.get("dependency_seals_sha256")
            == seal_audit["dependency_seals_sha256"]
            and manifest.get("phase979_script_hashes_sha256")
            == seal_audit["phase979_script_hashes_sha256"],
            "manifest file seal registry hashes differ from protocol")
    require(manifest.get("runner_sha256")
            == protocol["script_seals"]["runner"]["sha256"]
            and manifest.get("boundary_core_sha256")
            == protocol["dependency_seals"]["phase979_boundary_core"]["sha256"],
            "manifest runner/boundary hashes differ from protocol")
    require(manifest.get("generation_contract") == GENERATION_CONTRACT,
            "manifest generation contract changed")
    require(manifest.get("model_weights_loaded") is True
            and manifest.get("gpu_used") is True
            and manifest.get("generation_performed") is False
            and manifest.get("holdout") is False
            and manifest.get("holdout_loaded") is False
            and manifest.get("mechanism") is False
            and manifest.get("mechanism_authorized") is False,
            "manifest creation-state flags changed")


def verify_row_generation_contract(row: dict[str, Any]) -> None:
    for key, expected in ROW_GENERATION_CONTRACT.items():
        require(row.get(key) is expected,
                f"row generation contract changed: {key}")
    require(row.get("max_new_tokens") == MAX_NEW_TOKENS,
            "row max_new_tokens changed")


def verify_complete_status_generation_contract(status: dict[str, Any]) -> None:
    require(status.get("complete") is True
            and status.get("completed_rows") == EXPECTED_ROWS
            and status.get("generation_performed") is True
            and status.get("model_weights_loaded") is True
            and status.get("decision_computed") is False
            and status.get("holdout") is False
            and status.get("holdout_loaded") is False
            and status.get("mechanism") is False
            and status.get("mechanism_authorized") is False,
            "complete generator status contract changed")


def assert_contract() -> None:
    require(tuple(boundary.CHECKPOINTS) == CHECKPOINTS, "checkpoint dependency changed")
    require(boundary.MAX_NEW_TOKENS == MAX_NEW_TOKENS, "trajectory budget changed")
    require(boundary.BATCH_SIZE == BATCH_SIZE, "batch dependency changed")
    require(tuple(boundary.REPLICATES) == (0, 1),
            "Phase979 replicate registry changed unexpectedly")
    require(ARMS[ARM_A] == {
        "control_policy": "soft_no_think",
        "decoding_policy": "thinking_sampling",
        "role": "baseline",
    }, "formal arm A changed")
    require(ARMS[ARM_B] == {
        "control_policy": "soft_thinking",
        "decoding_policy": "thinking_sampling",
        "role": "candidate",
    }, "formal arm B changed")
    for spec in ARMS.values():
        require(spec["control_policy"] in boundary.CONTROL_POLICIES,
                "unknown control policy")
        require(spec["decoding_policy"] in boundary.DECODING_POLICIES,
                "unknown decoding policy")
        require(boundary.DECODING_POLICIES[spec["decoding_policy"]] == SAMPLING,
                "arms do not share frozen sampling")
    require(EXPECTED_ROWS == 1536, "formal row denominator changed")


def stable_pair_seed(dataset_identity_sha256: str, item_id: str, stream: int) -> int:
    """Common-random-number seed shared by A/B; arm is deliberately absent."""
    require(len(str(dataset_identity_sha256)) == 64, "invalid dataset identity hash")
    require(int(stream) in STREAMS, f"unknown stream: {stream}")
    payload = (
        f"phase981|fresh256|{dataset_identity_sha256}|"
        f"item={item_id}|stream={int(stream)}"
    )
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    return int(value % (2**31 - 1))


def pair_id(item_id: str, stream: int) -> str:
    require(int(stream) in STREAMS, f"unknown stream: {stream}")
    return f"{item_id}|stream_{int(stream)}"


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row.get("id")), str(row.get("arm")), int(row.get("stream", -1))


def expected_keys(items: Iterable[dict[str, Any]]) -> set[tuple[str, str, int]]:
    return {
        (str(item["id"]), arm, stream)
        for stream in STREAMS for arm in ARMS for item in items
    }


def canonical_grid(
    items: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], str, int]]:
    """Stream-major, arm-major, item-major grid used by resume/batch checks."""
    require(len(items) == ITEM_COUNT, "fresh dataset is not 256 items")
    output = [
        (item, arm, stream)
        for stream in STREAMS for arm in ARMS for item in items
    ]
    require(len(output) == EXPECTED_ROWS, "canonical grid denominator changed")
    require(len(output) % BATCH_SIZE == 0, "canonical grid does not divide into batches")
    return output


def chunks(values: list[Any], size: int = BATCH_SIZE) -> Iterable[list[Any]]:
    require(size == BATCH_SIZE and len(values) % size == 0,
            "formal grid requires full batches of eight")
    for start in range(0, len(values), size):
        yield values[start:start + size]


def render_prefix(tok, item: dict[str, Any], arm: str) -> tuple[str, str, list[int]]:
    require(arm in ARMS, f"unknown arm: {arm}")
    return boundary.render_prefix(tok, item, ARMS[arm]["control_policy"])


def analyze_checkpoints(
    tok, item: dict[str, Any], arm: str, generated_ids: list[int],
    eos_ids: list[int], think_open_id: int, think_close_id: int,
) -> dict[str, Any]:
    require(arm in ARMS, f"unknown arm: {arm}")
    return boundary.analyze_checkpoints(
        tok, item, ARMS[arm]["control_policy"], generated_ids,
        eos_ids, think_open_id, think_close_id,
    )


def four_channel(terminal_state: str) -> str:
    require(terminal_state in SIX_STATES, f"unknown terminal state: {terminal_state}")
    if terminal_state == "VALID_STOP":
        return "V"
    if terminal_state.startswith("CENSORED_"):
        return "C"
    if terminal_state == "EOS_INVALID_MODE":
        return "I_mode"
    require(terminal_state == "EOS_INVALID_SEMANTIC", "unmapped terminal state")
    return "I_sem"


def three_channel(terminal_state: str) -> str:
    value = four_channel(terminal_state)
    return "I" if value in {"I_mode", "I_sem"} else value


def matrix(
    labels: tuple[str, ...], pairs: Iterable[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    output = {left: {right: 0 for right in labels} for left in labels}
    for left, right in pairs:
        require(left in output and right in output[left], "matrix label outside registry")
        output[left][right] += 1
    return output


assert_contract()
