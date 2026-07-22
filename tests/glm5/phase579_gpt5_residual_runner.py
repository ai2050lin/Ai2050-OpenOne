#!/usr/bin/env python3
"""Frozen Phase579 full-residual acquisition runner.

This file deliberately does not search for a layer, head, neuron, direction,
formula, or score.  After a separate eight-case engineering qualification it
replays every one of the 336 Phase578 development prompts, in the original
batch order, for qwen3 and then GLM4.  The Phase578 repeat-1 token trajectory
is an exact admission condition.  DS7B is represented only by a blocking
receipt because it failed the preregistered Phase578 behavior gate.

The parent process never imports torch.  Each admitted model runs in a fresh,
strictly ordered child process.  A child persists BF16 model-API hidden-state
shards for every returned layer ordinal (whose final entry may include a
model-specific final normalization), all
left-padded and valid prompt positions, and every feedback forward actually
executed by batched greedy generation (including absorbing pads while peers
remain active).  No attention tensor, score,
hook, intervention, or model-selected coordinate is collected.
"""

from __future__ import annotations

import argparse
import builtins
import gc
import gzip
import hashlib
import importlib.metadata
import io
import json
import msvcrt
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests/glm5"
GPT5 = ROOT / "tests/gpt5"
for _entry in (GLM5, GPT5):
    if str(_entry) in sys.path:
        sys.path.remove(str(_entry))
sys.path.insert(0, str(GLM5))
sys.path.insert(1, str(GPT5))

PHASE = "Phase579"
ELIGIBLE_MODELS = ("qwen3", "glm4")
BLOCKED_MODELS = ("deepseek7b",)
ALL_PHASE578_MODELS = (*ELIGIBLE_MODELS, *BLOCKED_MODELS)
BATCH_SIZE = 8
MAX_NEW_TOKENS = 24
# Hugging Face generation returns one prefill hidden-state step followed by at
# most MAX_NEW_TOKENS - 1 forwards in which an emitted token is fed back.  The
# 24th emitted token has no following forward and therefore has no residual.
FEEDBACK_WIDTH = MAX_NEW_TOKENS - 1
ENGINEERING_ORDINALS = tuple(range(8))

PROTOCOL_DIR = ROOT / "tests/glm5/result/phase579_gpt5_residual_protocol"
MANIFEST_PATH = PROTOCOL_DIR / "phase579_development_residual_manifest.jsonl"
PROTOCOL_PATH = PROTOCOL_DIR / "phase579_preregistered_residual_protocol.json"
SELF_TEST_PATH = PROTOCOL_DIR / "phase579_protocol_self_test.json"
STAGE_COMMIT_PATH = PROTOCOL_DIR / "phase579_stage_commit.json"
AUDIT_PATH = PROTOCOL_DIR / "phase579_independent_freeze_audit.json"
FREEZE_PATH = PROTOCOL_DIR / "phase579_freeze_commit.json"

ENGINEERING_DIR = ROOT / "tests/glm5/result/phase579_gpt5_residual_engineering"
TRACE_DIR = ROOT / "tests/glm5/result/phase579_gpt5_residual_trace"
PHASE578_RAW_DIR = (
    ROOT / "tests/glm5/result/phase578_gpt5_development_behavior_raw"
)
PHASE578_ANALYSIS_DIR = (
    ROOT / "tests/glm5/result/phase578_gpt5_development_behavior_analysis"
)
PHASE578_SUMMARY_PATH = (
    PHASE578_ANALYSIS_DIR / "phase578_development_behavior_summary.json"
)
PHASE578_RECEIPT_PATH = PHASE578_RAW_DIR / "execution_receipt.json"

FORMAL_PYTHON = Path(
    r"C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe"
)
FORMAL_PYTHON_SHA256 = (
    "0f11fb7422fa347b7609ba0964ceccef3c8fa9f15230c37b9ec27668e68e8a8a"
)
ENGINE_PATH = ROOT / "tests/glm5/phase983_cross_model_engine.py"
ENGINE_EXPECTED_SHA256 = (
    "e345daf3c3eae289eb7a71b8a741eeaf3a11c6897d009a5f9d90a386b23eef6f"
)
MODEL_REGISTRY_PATH = ROOT / "tests/gpt5/model_registry.py"
SOURCE_RELATIVE = "tests/glm5/phase579_gpt5_residual_runner.py"
MODEL_LOGICAL_DIRS = {
    "qwen3": ROOT / "models/hf/qwen3-4b",
    "glm4": ROOT / "models/hf/glm4-9b-chat-hf",
    "deepseek7b": ROOT / "models/hf/deepseek-r1-distill-qwen-7b",
}
PHASE578_RAW_PATHS = {
    model: PHASE578_RAW_DIR / f"{index:02d}_{model}/raw_rows.jsonl.gz"
    for index, model in enumerate(ALL_PHASE578_MODELS)
}

GENERATION_CONTRACT = {
    "batch_size": BATCH_SIZE,
    "replay_source": "Phase578 development repeat1 raw rows",
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "use_cache": True,
    "pad_token_id": "adapter.pad_token_id",
    "eos_token_id": "adapter.effective_eos_token_ids",
    "return_dict_in_generate": True,
    "output_scores": False,
    "output_attentions": False,
    "output_hidden_states": True,
    "tokenizer_padding_side": "left",
    "tokenizer_padding": True,
    "tokenizer_truncation": False,
    "tokenizer_add_special_tokens": False,
    "decode_skip_special_tokens": False,
    "decode_clean_up_tokenization_spaces": False,
    "qwen3_enable_thinking": False,
    "quantization": "bitsandbytes_int8",
    "nonquantized_dtype": "torch.bfloat16",
    "attention_implementation": "sdpa",
    "cpu_or_disk_offload": False,
    "automatic_fallback": False,
    "phase578_behavior_replay_repeats_available": ["repeat1", "repeat2"],
    "phase578_behavior_replay_repeat_used": "repeat1",
    "engineering_internal_reexecution_count": 2,
    "full_trace_internal_reexecution_count": 1,
}
GENERATION_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(GENERATION_CONTRACT, sort_keys=True, separators=(",", ":"))
    .encode("utf-8")
).hexdigest()

SHARD_KEYS = (
    "metadata_rows",
    "prefill_residual",
    "prompt_mask",
    "feedback_residual",
    "feedback_executed_mask",
    "feedback_pre_eos_mask",
)
LAYER_INDEX_SEMANTICS = {
    "0": (
        "first model-API representation returned as hidden_states[0]; its "
        "architectural interpretation is defined by the frozen model code"
    ),
    "1..N": (
        "subsequent model-API representations in returned order; the final "
        "entry may include a model-specific final normalization"
    ),
    "selection": "none; all returned layers are persisted in original order",
}
FEEDBACK_INDEX_SEMANTICS = {
    "axis_length": FEEDBACK_WIDTH,
    "slot_f": (
        "residual produced when generated suffix token f is fed back to predict "
        "token f+1; generate prefill is not duplicated on this axis"
    ),
    "executed_mask": (
        "true when Hugging Face actually executed that feedback forward; this "
        "includes absorbing pad forwards for already-finished rows while other "
        "rows in the same batch remain active"
    ),
    "pre_eos_mask": (
        "true only for feedback positions strictly before the row's EOS (or "
        "before the available feedback-width boundary when EOS is absent)"
    ),
    "invalid_fill": "exact BF16 zero",
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


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
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


def write_json(path: Path, payload: Any) -> None:
    write_exclusive(path, json_bytes(payload))


def _identity_path(entry: dict[str, Any]) -> Path | None:
    raw = entry.get("path", entry.get("resolved_path"))
    if not isinstance(raw, str) or not raw:
        return None
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _verify_file_identity(path: Path, entry: dict[str, Any], label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing/aliased frozen file: {label}: {path}")
    expected_size = entry.get("size_bytes")
    expected_hash = entry.get("sha256")
    if (
        not isinstance(expected_size, int)
        or not isinstance(expected_hash, str)
        or path.stat().st_size != expected_size
        or sha256_file(path) != expected_hash
    ):
        raise RuntimeError(f"frozen file identity drift: {label}: {path}")


def _future_split_path(path: Path) -> bool:
    folded = str(path).replace("\\", "/").casefold()
    return any(term in folded for term in ("confirmation", "heldout", "sealed"))


def read_manifest() -> list[dict[str, Any]]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 336:
        raise RuntimeError("Phase579 development denominator is not exactly 336")
    if [row.get("ordinal") for row in rows] != list(range(336)):
        raise RuntimeError("Phase579 development manifest order drift")
    if len({row.get("case_id") for row in rows}) != 336:
        raise RuntimeError("Phase579 case registry is not one-to-one")
    for row in rows:
        if not all((
            row.get("phase_id") == PHASE,
            row.get("split") == "development",
            isinstance(row.get("case_id"), str),
            isinstance(row.get("raw_prompt"), str),
            isinstance(row.get("raw_role_char_spans"), dict),
            row.get("candidate_coordinates") == [],
            row.get("candidate_mechanism_formulas") == [],
        )):
            raise RuntimeError(f"invalid Phase579 manifest row: {row.get('ordinal')}")
        source = row.get("source_case_metadata")
        if isinstance(source, dict) and source.get("split") != "development":
            raise RuntimeError("non-development source metadata entered Phase579")
        replay = row.get("raw_replay")
        if not isinstance(replay, dict):
            raise RuntimeError("Phase579 raw replay registry missing")
        for model in ELIGIBLE_MODELS:
            records = replay.get(model)
            if not isinstance(records, list) or [
                item.get("execution_repeat") if isinstance(item, dict) else None
                for item in records
            ] != ["repeat1", "repeat2"]:
                raise RuntimeError(
                    f"Phase579 two-repeat manifest closure drift: "
                    f"{row.get('case_id')}/{model}"
                )
    return rows


def _required_protocol_files() -> tuple[Path, ...]:
    return (
        MANIFEST_PATH, PROTOCOL_PATH, SELF_TEST_PATH, STAGE_COMMIT_PATH,
        AUDIT_PATH, FREEZE_PATH,
    )


def verify_bridge() -> dict[str, Any]:
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError("Phase579 requires the frozen formal Python interpreter")
    if (
        not FORMAL_PYTHON.is_file()
        or sha256_file(FORMAL_PYTHON) != FORMAL_PYTHON_SHA256
    ):
        raise RuntimeError("formal Python executable identity drift")
    for path in _required_protocol_files():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing/aliased Phase579 freeze artifact: {path}")
    freeze = read_json(FREEZE_PATH)
    protocol = read_json(PROTOCOL_PATH)
    audit = read_json(AUDIT_PATH)
    self_test = read_json(SELF_TEST_PATH)
    stage_commit = read_json(STAGE_COMMIT_PATH)
    expected_commitments = {
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "stage_commit_sha256": sha256_file(STAGE_COMMIT_PATH),
        "development_manifest_sha256": sha256_file(MANIFEST_PATH),
        "self_test_sha256": sha256_file(SELF_TEST_PATH),
        "independent_audit_sha256": sha256_file(AUDIT_PATH),
    }
    if not all((
        freeze.get("schema_version") == "phase579_freeze_commit.v1",
        freeze.get("freeze_complete") is True,
        all(freeze.get(key) == value for key, value in expected_commitments.items()),
        freeze.get("eligible_models") == list(ELIGIBLE_MODELS),
        freeze.get("behavior_blocked_models") == list(BLOCKED_MODELS),
        freeze.get("models_in_required_future_order") == list(ELIGIBLE_MODELS),
        freeze.get("engineering_qualification_authorized") is True,
        freeze.get("full_development_trace_authorized") is False,
        freeze.get(
            "full_development_trace_authorized_if_and_only_if_engineering_passes"
        ) is True,
        freeze.get("cross_model_internal_comparison_authorized") is False,
        freeze.get("confirmation_authorized") is False,
        freeze.get("heldout_authorized") is False,
        freeze.get("sealed_authorized") is False,
        freeze.get("candidate_coordinates") == [],
        freeze.get("candidate_mechanism_formulas") == [],
        freeze.get("next_required_stage")
        == "phase579_residual_engineering_qualification",
        audit.get("passed") is True,
        all(audit.get("checks", {}).values()),
        self_test.get("passed") is True,
        stage_commit.get("stage_complete") is True,
    )):
        raise RuntimeError("Phase579 frozen bridge contract failed")
    protocol_order = protocol.get(
        "models_in_required_future_order",
        protocol.get("eligible_models"),
    )
    if protocol_order != list(ELIGIBLE_MODELS):
        raise RuntimeError("Phase579 protocol eligible-model order drift")
    if protocol.get("candidate_coordinates") not in (None, []):
        raise RuntimeError("Phase579 protocol preselected candidate coordinates")
    if protocol.get("candidate_mechanism_formulas") not in (None, []):
        raise RuntimeError("Phase579 protocol preselected a mechanism formula")
    manifest_rows = read_manifest()
    frozen_engineering_ids = freeze.get("engineering_case_ids")
    protocol_engineering_ids = protocol.get(
        "engineering_qualification_contract", {}
    ).get("case_ids")
    observed_engineering_ids = [row["case_id"] for row in manifest_rows[:8]]
    if not all((
        frozen_engineering_ids == observed_engineering_ids,
        protocol_engineering_ids == observed_engineering_ids,
        all(
            _manifest_replay_entry(manifest_rows[0], model).get(
                "execution_repeat"
            ) == "repeat1"
            for model in ELIGIBLE_MODELS
        ),
    )):
        raise RuntimeError("Phase579 real-manifest runner bridge self-test failed")
    if sha256_file(ENGINE_PATH) != ENGINE_EXPECTED_SHA256:
        raise RuntimeError("cross-model engine source drift")

    source_registry = freeze.get("source_identities", {})
    if not isinstance(source_registry, dict) or SOURCE_RELATIVE not in source_registry:
        raise RuntimeError("Phase579 runner is absent from frozen source identities")
    runner_entry = source_registry[SOURCE_RELATIVE]
    runner_path = _identity_path(runner_entry)
    if runner_path is None or runner_path.resolve() != Path(__file__).resolve():
        raise RuntimeError("frozen Phase579 runner path drift")
    _verify_file_identity(runner_path, runner_entry, SOURCE_RELATIVE)

    # Reopen only implementation/runtime and already-public development evidence.
    # Future split paths are never opened merely because a historical freeze
    # contains their commitments.
    verified_upstream: dict[str, str] = {}
    upstream = freeze.get("upstream_identities", {})
    if not isinstance(upstream, dict):
        raise RuntimeError("Phase579 upstream identity registry missing")
    admitted_roots = (
        PHASE578_RAW_DIR.resolve(), PHASE578_ANALYSIS_DIR.resolve(),
        (ROOT / "tests/glm5/result/phase578_gpt5_runner_scorer_protocol").resolve(),
        (ROOT / "tests/glm5/result/phase577_gpt5_natural_behavior_protocol").resolve(),
    )
    admitted_exact = {ENGINE_PATH.resolve(), MODEL_REGISTRY_PATH.resolve()}
    for name, entry in upstream.items():
        if not isinstance(entry, dict):
            continue
        path = _identity_path(entry)
        if path is None or _future_split_path(path):
            continue
        resolved = path.resolve(strict=False)
        inside_admitted = resolved in admitted_exact or any(
            resolved == root or root in resolved.parents for root in admitted_roots
        )
        if not inside_admitted:
            continue
        # Within the Phase577 protocol directory only development is admissible.
        if (
            "phase577_gpt5_natural_behavior_protocol" in str(resolved)
            and "development" not in resolved.name.casefold()
        ):
            continue
        _verify_file_identity(path, entry, f"upstream:{name}")
        verified_upstream[name] = entry["sha256"]

    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "bitsandbytes", "accelerate")
    }
    frozen_runtime = freeze.get(
        "formal_runtime_identity", protocol.get("formal_runtime_identity", {})
    )
    expected_packages = frozen_runtime.get("packages", frozen_runtime)
    if packages != expected_packages:
        raise RuntimeError(f"Phase579 formal package drift: {packages}")
    if protocol.get("generation_contract") not in (None, GENERATION_CONTRACT):
        raise RuntimeError("Phase579 generation contract drift")
    return {
        "freeze_sha256": sha256_file(FREEZE_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "stage_commit_sha256": sha256_file(STAGE_COMMIT_PATH),
        "audit_sha256": sha256_file(AUDIT_PATH),
        "self_test_sha256": sha256_file(SELF_TEST_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "engine_sha256": sha256_file(ENGINE_PATH),
        "formal_python_sha256": FORMAL_PYTHON_SHA256,
        "formal_packages": packages,
        "verified_admitted_upstream": verified_upstream,
        "confirmation_accessed": False,
        "heldout_accessed": False,
        "sealed_accessed": False,
    }


def _frozen_model_entry(protocol: dict[str, Any], model: str) -> dict[str, Any]:
    for container in (
        protocol.get("frozen_model_artifact_identities"),
        read_json(FREEZE_PATH).get("frozen_model_artifact_identities"),
    ):
        if isinstance(container, dict) and isinstance(container.get(model), dict):
            return container[model]
    raise RuntimeError(f"missing frozen Phase579 model identity: {model}")


def verify_model_artifacts(model: str, protocol: dict[str, Any]) -> dict[str, Any]:
    expected = _frozen_model_entry(protocol, model)
    logical = MODEL_LOGICAL_DIRS[model]
    resolved = logical.resolve(strict=True)
    expected_root = expected.get("resolved_path", expected.get("local_dir"))
    if not isinstance(expected_root, str) or (
        str(resolved).casefold()
        != str(Path(expected_root).resolve(strict=True)).casefold()
    ):
        raise RuntimeError(f"{model}: resolved model path drift")
    expected_files: dict[str, dict[str, Any]] = {}
    for group in ("tokenizer_and_config_files", "weight_files", "files"):
        values = expected.get(group, [])
        if isinstance(values, list):
            for item in values:
                if not isinstance(item, dict):
                    continue
                relative = item.get("relative_path", item.get("name"))
                if isinstance(relative, str):
                    expected_files[relative.replace("\\", "/")] = item
    if not expected_files:
        raise RuntimeError(f"{model}: frozen recursive artifact list is empty")
    actual_files: dict[str, Path] = {}
    for path in resolved.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"{model}: nested model symlink forbidden: {path}")
        if path.is_file():
            relative = str(path.relative_to(resolved)).replace("\\", "/")
            actual_files[relative] = path
    if set(actual_files) != set(expected_files):
        raise RuntimeError(f"{model}: recursive model artifact registry drift")
    reports = []
    for relative in sorted(expected_files):
        path = actual_files[relative]
        expected_file = expected_files[relative]
        observed = sha256_file(path)
        if (
            path.stat().st_size != expected_file.get("size_bytes")
            or observed != expected_file.get("sha256")
        ):
            raise RuntimeError(f"{model}: model artifact drift: {relative}")
        reports.append({
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": observed,
        })
    payload = {
        "model": model,
        "resolved_path": str(resolved),
        "file_count": len(reports),
        "files": reports,
        "frozen_identity_sha256": expected.get(
            "frozen_identity_sha256", expected.get("identity_sha256")
        ),
    }
    payload["verification_payload_sha256"] = sha256_bytes(
        canonical_json(payload).encode("utf-8")
    )
    return payload


def _phase578_registry_entry(path: Path) -> dict[str, Any]:
    receipt = read_json(PHASE578_RECEIPT_PATH)
    relative = str(path.relative_to(PHASE578_RAW_DIR)).replace("\\", "/")
    matches = [
        item for item in receipt.get("artifact_registry_before_receipt", [])
        if item.get("path") == relative
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Phase578 raw artifact is not uniquely receipted: {relative}")
    entry = matches[0]
    _verify_file_identity(path, entry, f"Phase578 raw:{relative}")
    return entry


def read_phase578_repeat1(model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = PHASE578_RAW_PATHS[model]
    registry = _phase578_registry_entry(path)
    selected: list[dict[str, Any]] = []
    with gzip.open(path, "rb") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw.decode("utf-8"))
            if row.get("execution_repeat") == "repeat1":
                row["_source_raw_row_sha256"] = sha256_bytes(raw)
                selected.append(row)
    if len(selected) != 336:
        raise RuntimeError(f"{model}: Phase578 repeat1 denominator drift")
    if len({row.get("case_id") for row in selected}) != 336:
        raise RuntimeError(f"{model}: Phase578 repeat1 case registry drift")
    if any(row.get("split") != "development" for row in selected):
        raise RuntimeError(f"{model}: non-development Phase578 raw row entered replay")
    return selected, {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "phase578_receipt_entry": registry,
    }


def _manifest_replay_entry(row: dict[str, Any], model: str) -> dict[str, Any]:
    replay = row.get("raw_replay")
    if not isinstance(replay, dict) or not isinstance(
        replay.get(model), (dict, list)
    ):
        raise RuntimeError(f"manifest raw replay missing: {row.get('case_id')}/{model}")
    model_replay = replay[model]
    if isinstance(model_replay, list):
        matches = [
            item for item in model_replay
            if isinstance(item, dict) and item.get("execution_repeat") == "repeat1"
        ]
        entry = matches[0] if len(matches) == 1 else None
    else:
        entry = model_replay.get("repeat1")
    if not isinstance(entry, dict):
        raise RuntimeError(f"manifest repeat1 replay missing: {row.get('case_id')}/{model}")
    return entry


def _validate_manifest_replay(
    manifest_row: dict[str, Any], reference: dict[str, Any], model: str,
) -> None:
    entry = _manifest_replay_entry(manifest_row, model)
    aliases = {
        "input_token_ids": "input_token_ids",
        "input_token_ids_sha256": "input_token_ids_sha256",
        "rendered_prompt_sha256": "rendered_prompt_sha256",
        "generated_token_ids_before_eos": "generated_token_ids_before_eos",
        "full_generated_suffix_token_ids": "full_generated_suffix_token_ids",
        "effective_eos_token_ids": "effective_eos_token_ids",
        "pad_token_id": "pad_token_id",
        "first_eos_index": "first_eos_index",
        "eos_seen": "eos_seen",
        "termination_event": "termination_event",
    }
    for manifest_key, raw_key in aliases.items():
        if manifest_key in entry and entry[manifest_key] != reference.get(raw_key):
            raise RuntimeError(
                f"manifest/Phase578 replay drift: {manifest_row['case_id']}/"
                f"{model}/{manifest_key}"
            )
    row_hash = entry.get(
        "source_raw_row_sha256",
        entry.get("row_sha256", entry.get("raw_row_sha256")),
    )
    if not isinstance(row_hash, str) or row_hash != reference.get(
        "_source_raw_row_sha256"
    ):
        raise RuntimeError(
            f"manifest Phase578 row hash drift: {manifest_row['case_id']}/{model}"
        )


def render_chat(tokenizer: Any, model: str, content: str) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if model == "qwen3":
        kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], **kwargs
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("chat template returned an empty prompt")
    return rendered


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(ids), skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _token_character_intervals(
    tokenizer: Any, rendered: str, input_ids: list[int],
) -> list[tuple[int, int]]:
    boundaries = [0]
    for end in range(1, len(input_ids) + 1):
        decoded = _decode(tokenizer, input_ids[:end])
        if not rendered.startswith(decoded):
            raise RuntimeError("tokenizer prefix decode is not a rendered-text prefix")
        if len(decoded) < boundaries[-1]:
            raise RuntimeError("tokenizer prefix decode boundary is non-monotonic")
        boundaries.append(len(decoded))
    if boundaries[-1] != len(rendered) or _decode(tokenizer, input_ids) != rendered:
        raise RuntimeError("rendered prompt lacks an exact reversible token decode")
    return list(zip(boundaries[:-1], boundaries[1:]))


def _role_token_positions(
    tokenizer: Any, row: dict[str, Any], rendered: str,
    input_ids: list[int], prompt_width: int,
) -> dict[str, Any]:
    raw_prompt = row["raw_prompt"]
    raw_start = rendered.find(raw_prompt)
    if raw_start < 0 or rendered.find(raw_prompt, raw_start + 1) >= 0:
        raise RuntimeError("raw prompt is not a unique rendered-text substring")
    intervals = _token_character_intervals(tokenizer, rendered, input_ids)
    pad_left = prompt_width - len(input_ids)
    if pad_left < 0:
        raise RuntimeError("negative prompt left padding")
    output: dict[str, Any] = {}
    spans = row["raw_role_char_spans"]
    for role in ("focus", "comparison", "query_anchor"):
        span = spans.get(role)
        if span is None:
            output[role] = {
                "raw_char_span": None,
                "rendered_char_span": None,
                "unpadded_token_positions": [],
                "padded_token_positions": [],
                "token_ids": [],
            }
            continue
        if not isinstance(span, dict):
            raise RuntimeError(f"invalid raw role span: {role}")
        start, end, text = span.get("start"), span.get("end"), span.get("text")
        if not all((
            isinstance(start, int), isinstance(end, int),
            isinstance(text, str), 0 <= start < end <= len(raw_prompt),
            raw_prompt[start:end] == text,
        )):
            raise RuntimeError(f"invalid raw role character identity: {role}")
        rendered_start, rendered_end = raw_start + start, raw_start + end
        positions = [
            index for index, (token_start, token_end) in enumerate(intervals)
            if token_start < rendered_end and token_end > rendered_start
        ]
        if not positions:
            raise RuntimeError(f"role span mapped to no prompt token: {role}")
        padded = [pad_left + index for index in positions]
        output[role] = {
            "raw_char_span": {"start": start, "end": end, "text": text},
            "rendered_char_span": {
                "start": rendered_start, "end": rendered_end, "text": text,
            },
            "unpadded_token_positions": positions,
            "padded_token_positions": padded,
            "token_ids": [input_ids[index] for index in positions],
        }
    return output


def _forbidden_research_path(
    value: Any, model: str, pending_root: Path,
) -> bool:
    if isinstance(value, int):
        return False
    try:
        candidate = Path(os.fsdecode(value)).resolve(strict=False)
    except (TypeError, ValueError, OSError):
        return False
    current_model = MODEL_LOGICAL_DIRS[model].resolve(strict=True)
    all_models = [path.resolve(strict=True) for path in MODEL_LOGICAL_DIRS.values()]
    if any(candidate == root or root in candidate.parents for root in all_models):
        return not (candidate == current_model or current_model in candidate.parents)
    repo = ROOT.resolve(strict=True)
    if not (candidate == repo or repo in candidate.parents):
        return False
    if _future_split_path(candidate):
        return True
    allowed_exact = {
        *[path.resolve(strict=False) for path in _required_protocol_files()],
        ENGINE_PATH.resolve(), MODEL_REGISTRY_PATH.resolve(),
        Path(__file__).resolve(), PHASE578_RECEIPT_PATH.resolve(),
        PHASE578_SUMMARY_PATH.resolve(), PHASE578_RAW_PATHS[model].resolve(),
    }
    pycache_allowed = (
        candidate.parent.name == "__pycache__"
        and any(candidate.name.startswith(prefix) for prefix in (
            "phase983_cross_model_engine.", "model_registry.",
        ))
    )
    inside_pending = candidate == pending_root or pending_root in candidate.parents
    return not (candidate in allowed_exact or pycache_allowed or inside_pending)


def install_research_access_guard(
    model: str, pending_root: Path,
) -> tuple[dict[str, int], Callable[[], None]]:
    attempts = {
        "forbidden_research_open_attempts": 0,
        "confirmation_open_attempts": 0,
        "heldout_open_attempts": 0,
        "sealed_open_attempts": 0,
    }
    original_builtin, original_io, original_os = builtins.open, io.open, os.open

    def guard(original: Callable[..., Any]) -> Callable[..., Any]:
        def guarded(file: Any, *args: Any, **kwargs: Any) -> Any:
            if _forbidden_research_path(file, model, pending_root):
                attempts["forbidden_research_open_attempts"] += 1
                folded = str(file).casefold()
                for term in ("confirmation", "heldout", "sealed"):
                    if term in folded:
                        attempts[f"{term}_open_attempts"] += 1
                raise RuntimeError(f"Phase579 development-only firewall: {file}")
            return original(file, *args, **kwargs)
        return guarded

    builtins.open = guard(original_builtin)
    io.open = guard(original_io)
    os.open = guard(original_os)

    def restore() -> None:
        builtins.open, io.open, os.open = original_builtin, original_io, original_os

    return attempts, restore


def strict_release(engine: Any, adapter: Any) -> dict[str, Any]:
    import torch

    release_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        engine.release_model_adapter(adapter)
    except BaseException as exc:
        release_error = exc
        exc.__traceback__ = None
    steps: dict[str, bool] = {}
    try:
        gc.collect()
        torch.cuda.synchronize()
        steps["synchronize_before_cublas_clear"] = True
        clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        if not callable(clear):
            raise RuntimeError("required cuBLAS workspace clear API unavailable")
        clear()
        steps["cublas_workspaces_cleared"] = True
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        steps["final_allocator_cleanup"] = True
    except BaseException as exc:
        cleanup_error = exc
        exc.__traceback__ = None
    allocated = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())
    report = {
        "steps": steps,
        "allocated_after_release": allocated,
        "reserved_after_release": reserved,
        "cleanup_pass": (
            release_error is None and cleanup_error is None
            and allocated == 0 and reserved == 0
        ),
        "release_error_type": (
            None if release_error is None else type(release_error).__name__
        ),
        "strict_cleanup_error_type": (
            None if cleanup_error is None else type(cleanup_error).__name__
        ),
    }
    if release_error is not None:
        raise release_error
    if cleanup_error is not None:
        raise cleanup_error
    if allocated != 0 or reserved != 0:
        raise RuntimeError(
            f"strict CUDA cleanup retained allocated={allocated}, reserved={reserved}"
        )
    return report


def _tensor_finite(tensor: Any) -> bool:
    import torch

    return bool(torch.isfinite(tensor).all().item())


def _shard_tensor_contract(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    if tuple(payload) != SHARD_KEYS:
        raise RuntimeError(f"trace shard key/order drift: {tuple(payload)}")
    metadata = payload["metadata_rows"]
    prefill = payload["prefill_residual"]
    prompt_mask = payload["prompt_mask"]
    feedback = payload["feedback_residual"]
    executed = payload["feedback_executed_mask"]
    pre_eos = payload["feedback_pre_eos_mask"]
    if not isinstance(metadata, list) or not metadata:
        raise RuntimeError("trace shard metadata is empty/non-list")
    batch = len(metadata)
    if (
        not isinstance(prefill, torch.Tensor)
        or not isinstance(prompt_mask, torch.Tensor)
        or not isinstance(feedback, torch.Tensor)
        or not isinstance(executed, torch.Tensor)
        or not isinstance(pre_eos, torch.Tensor)
        or prefill.ndim != 4 or feedback.ndim != 4
        or prompt_mask.ndim != 2 or executed.ndim != 2 or pre_eos.ndim != 2
    ):
        raise RuntimeError("trace shard tensor rank/type drift")
    b, layers, prompt_width, hidden = prefill.shape
    if (
        b != batch
        or tuple(prompt_mask.shape) != (batch, prompt_width)
        or tuple(feedback.shape) != (batch, layers, FEEDBACK_WIDTH, hidden)
        or tuple(executed.shape) != (batch, FEEDBACK_WIDTH)
        or tuple(pre_eos.shape) != (batch, FEEDBACK_WIDTH)
        or prefill.dtype != torch.bfloat16
        or feedback.dtype != torch.bfloat16
        or prompt_mask.dtype != torch.bool
        or executed.dtype != torch.bool
        or pre_eos.dtype != torch.bool
        or any(tensor.device.type != "cpu" for tensor in (
            prefill, prompt_mask, feedback, executed, pre_eos,
        ))
    ):
        raise RuntimeError("trace shard shape/dtype/device contract drift")
    if not _tensor_finite(prefill) or not _tensor_finite(feedback):
        raise RuntimeError("trace shard contains non-finite residual values")
    if bool((pre_eos & ~executed).any().item()):
        raise RuntimeError("pre-EOS feedback mask escaped executed mask")
    invalid = ~executed[:, None, :, None]
    if bool(feedback.masked_select(invalid.expand_as(feedback)).ne(0).any().item()):
        raise RuntimeError("invalid feedback residual slots are not exact zero")
    for row_index, row in enumerate(metadata):
        prompt_count = row.get("prompt_token_count")
        if not isinstance(prompt_count, int) or not 0 < prompt_count <= prompt_width:
            raise RuntimeError("invalid metadata prompt token count")
        expected_prompt_mask = torch.zeros(prompt_width, dtype=torch.bool)
        expected_prompt_mask[prompt_width - prompt_count:] = True
        if not torch.equal(prompt_mask[row_index], expected_prompt_mask):
            raise RuntimeError("prompt mask is not exact left padding")
        if row.get("prompt_padded_width") != prompt_width:
            raise RuntimeError("metadata/tensor prompt width drift")
        if row.get("answer_boundary_position") != prompt_width - 1:
            raise RuntimeError("answer boundary is not the final prompt position")
        if row.get("generation_start_position") != prompt_width:
            raise RuntimeError("generation start boundary drift")
        for role_entry in row.get("role_token_positions", {}).values():
            for position in role_entry.get("padded_token_positions", []):
                if not 0 <= position < prompt_width or not bool(
                    prompt_mask[row_index, position].item()
                ):
                    raise RuntimeError("role position points outside valid prompt")
    return {
        "batch_size": batch,
        "layer_count": layers,
        "prompt_width": prompt_width,
        "hidden_size": hidden,
        "prefill_shape": list(prefill.shape),
        "feedback_shape": list(feedback.shape),
        "bf16": True,
        "finite": True,
        "invalid_feedback_exact_zero": True,
    }


def write_torch_shard_exclusive(
    path: Path, payload: dict[str, Any],
) -> dict[str, Any]:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"no-overwrite shard publication refused: {path}")
    contract = _shard_tensor_contract(payload)
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        loaded_contract = _shard_tensor_contract(loaded)
        if loaded_contract != contract or loaded["metadata_rows"] != payload[
            "metadata_rows"
        ]:
            raise RuntimeError("trace shard round-trip metadata/contract drift")
        for key in SHARD_KEYS[1:]:
            if not torch.equal(loaded[key], payload[key]):
                raise RuntimeError(f"trace shard round-trip tensor drift: {key}")
        del loaded
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        **contract,
        "path": path.name,
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "torch_save_load_roundtrip_exact": True,
    }


def _trace_payloads_exact(
    left: dict[str, Any], right: dict[str, Any],
) -> bool:
    """Exact engineering repeat gate; no tolerance or statistic is admitted."""
    import torch

    if tuple(left) != SHARD_KEYS or tuple(right) != SHARD_KEYS:
        return False
    if left["metadata_rows"] != right["metadata_rows"]:
        return False
    return all(torch.equal(left[key], right[key]) for key in SHARD_KEYS[1:])


def _expected_feedback_masks(
    suffix: list[int], first_eos_index: int | None,
) -> tuple[list[bool], list[bool]]:
    emitted = len(suffix)
    pre_eos_end = min(
        max(emitted - 1, 0) if first_eos_index is None else first_eos_index,
        FEEDBACK_WIDTH,
    )
    pre_eos = [index < pre_eos_end for index in range(FEEDBACK_WIDTH)]
    # Generation performs one feedback forward for every generation step after
    # prefill.  Finished rows still participate with an absorbing pad while a
    # peer row remains active, so execution is batch-wide rather than per-EOS.
    executable_end = min(max(emitted - 1, 0), FEEDBACK_WIDTH)
    executed = [index < executable_end for index in range(FEEDBACK_WIDTH)]
    return executed, pre_eos


def collect_batch(
    adapter: Any, model: str, batch_index: int,
    rows: list[dict[str, Any]], references: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    if len(rows) != len(references) or not rows:
        raise RuntimeError("trace batch row/reference cardinality drift")
    rendered = [render_chat(adapter.tokenizer, model, row["raw_prompt"]) for row in rows]
    input_ids = [
        [int(value) for value in adapter.tokenizer(
            text, add_special_tokens=False, return_attention_mask=False,
        ).input_ids]
        for text in rendered
    ]
    encoded = adapter.tokenizer(
        rendered, return_tensors="pt", padding=True, truncation=False,
        add_special_tokens=False,
    )
    prompt_width = int(encoded["input_ids"].shape[1])
    if prompt_width <= 0:
        raise RuntimeError("empty padded prompt batch")
    attention_lengths = [
        int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()
    ]
    if attention_lengths != [len(value) for value in input_ids]:
        raise RuntimeError("batch attention mask/input length mismatch")
    for row_index, ids in enumerate(input_ids):
        mask = encoded["attention_mask"][row_index].to(dtype=torch.bool)
        observed = [int(value) for value in encoded["input_ids"][row_index][mask].tolist()]
        if observed != ids:
            raise RuntimeError("padded batch token matrix differs from unpadded prompt")
    eos_ids = [int(value) for value in adapter.eos_identity[
        "effective_eos_token_ids"
    ]]
    for row, reference, text, ids in zip(rows, references, rendered, input_ids):
        _validate_manifest_replay(row, reference, model)
        if not all((
            reference.get("case_id") == row.get("case_id"),
            reference.get("execution_repeat") == "repeat1",
            reference.get("split") == "development",
            reference.get("input_token_ids") == ids,
            reference.get("input_token_ids_sha256")
            == sha256_bytes(canonical_json(ids).encode("utf-8")),
            reference.get("rendered_prompt_sha256")
            == sha256_bytes(text.encode("utf-8")),
            reference.get("batch_padded_prompt_width") == prompt_width,
            reference.get("attention_mask_valid_tokens") == len(ids),
            reference.get("effective_eos_token_ids") == eos_ids,
            reference.get("pad_token_id") == int(adapter.pad_token_id),
        )):
            raise RuntimeError(f"Phase578 input replay mismatch: {row.get('case_id')}")
    gpu_inputs = {key: value.to(adapter.input_device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = adapter.model.generate(
            **gpu_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
            pad_token_id=adapter.pad_token_id,
            eos_token_id=eos_ids,
            return_dict_in_generate=True,
            output_scores=False,
            output_attentions=False,
            output_hidden_states=True,
        )
    sequences = generated.sequences
    suffix_width = int(sequences.shape[1]) - prompt_width
    if not 0 < suffix_width <= MAX_NEW_TOKENS:
        raise RuntimeError("generated suffix width outside frozen budget")
    suffixes = [
        [int(value) for value in sequences[index, prompt_width:].detach().cpu().tolist()]
        for index in range(len(rows))
    ]
    hidden_steps = generated.hidden_states
    if not isinstance(hidden_steps, (tuple, list)) or len(hidden_steps) != suffix_width:
        raise RuntimeError("generate hidden-state step count differs from suffix width")
    if not hidden_steps or not isinstance(hidden_steps[0], (tuple, list)):
        raise RuntimeError("generate did not return per-layer hidden states")
    layer_count = len(hidden_steps[0])
    expected_layers = int(getattr(adapter.config, "num_hidden_layers")) + 1
    if layer_count != expected_layers:
        raise RuntimeError("hidden-state layer count differs from model config")
    if any(len(step) != layer_count for step in hidden_steps):
        raise RuntimeError("hidden-state layer count changed across generation")
    first_shapes = [tuple(value.shape) for value in hidden_steps[0]]
    if any(shape[:2] != (len(rows), prompt_width) for shape in first_shapes):
        raise RuntimeError("prefill hidden-state batch/prompt shape drift")
    hidden_size = first_shapes[0][-1]
    if hidden_size != int(getattr(adapter.config, "hidden_size")):
        raise RuntimeError("hidden-state width differs from model config")
    if any(shape[-1] != hidden_size for shape in first_shapes):
        raise RuntimeError("prefill hidden width changed across layers")

    prefill = torch.stack([
        value.detach().to(device="cpu", dtype=torch.bfloat16)
        for value in hidden_steps[0]
    ], dim=1).contiguous()
    feedback = torch.zeros(
        (len(rows), layer_count, FEEDBACK_WIDTH, hidden_size),
        dtype=torch.bfloat16, device="cpu",
    )
    executed_mask = torch.zeros(
        (len(rows), FEEDBACK_WIDTH), dtype=torch.bool, device="cpu"
    )
    pre_eos_mask = torch.zeros_like(executed_mask)
    metadata: list[dict[str, Any]] = []
    exact_rows = 0
    for row_index, (row, reference, text, ids, suffix) in enumerate(zip(
        rows, references, rendered, input_ids, suffixes,
    )):
        if suffix != reference.get("full_generated_suffix_token_ids"):
            raise RuntimeError(
                f"Phase578 full suffix replay mismatch: {row.get('case_id')}"
            )
        first_eos = next(
            (index for index, token_id in enumerate(suffix) if token_id in eos_ids),
            None,
        )
        post_eos = [] if first_eos is None else suffix[first_eos + 1:]
        if not all((
            first_eos == reference.get("first_eos_index"),
            (first_eos is not None) == reference.get("eos_seen"),
            (None if first_eos is None else suffix[first_eos])
            == reference.get("first_eos_token_id"),
            post_eos == reference.get("post_eos_token_ids"),
            all(value == adapter.pad_token_id for value in post_eos),
            reference.get("post_eos_tokens_all_pad") is True,
        )):
            raise RuntimeError(f"Phase578 EOS/pad replay mismatch: {row.get('case_id')}")
        executed, pre_eos = _expected_feedback_masks(suffix, first_eos)
        executed_mask[row_index] = torch.tensor(executed, dtype=torch.bool)
        pre_eos_mask[row_index] = torch.tensor(pre_eos, dtype=torch.bool)
        roles = _role_token_positions(
            adapter.tokenizer, row, text, ids, prompt_width
        )
        metadata.append({
            "schema_version": "phase579_residual_trace_metadata_row.v1",
            "phase_id": PHASE,
            "split": "development",
            "model": model,
            "model_order_index": ELIGIBLE_MODELS.index(model),
            "ordinal": row["ordinal"],
            "case_id": row["case_id"],
            "analysis_unit_id": row.get("analysis_unit_id"),
            "batch_index": batch_index,
            "batch_row_index": row_index,
            "source_case_record_sha256": row.get("source_case_record_sha256"),
            "raw_prompt_sha256": sha256_bytes(row["raw_prompt"].encode("utf-8")),
            "rendered_prompt_sha256": sha256_bytes(text.encode("utf-8")),
            "input_token_ids": ids,
            "input_token_ids_sha256": sha256_bytes(
                canonical_json(ids).encode("utf-8")
            ),
            "pad_token_id": int(adapter.pad_token_id),
            "prompt_token_count": len(ids),
            "prompt_padded_width": prompt_width,
            "prompt_left_pad_count": prompt_width - len(ids),
            "answer_boundary_position": prompt_width - 1,
            "generation_start_position": prompt_width,
            "generation_token_ids": suffix,
            "generation_token_ids_sha256": sha256_bytes(
                canonical_json(suffix).encode("utf-8")
            ),
            "generation_suffix_width": len(suffix),
            "effective_eos_token_ids": eos_ids,
            "first_eos_index": first_eos,
            "eos_seen": first_eos is not None,
            "post_eos_token_ids": post_eos,
            "post_eos_tokens_all_pad": True,
            "role_token_positions": roles,
            "feedback_executed_positions": [
                index for index, value in enumerate(executed) if value
            ],
            "feedback_pre_eos_positions": [
                index for index, value in enumerate(pre_eos) if value
            ],
            "phase578_repeat1_exact_input_replay": True,
            "phase578_repeat1_exact_suffix_eos_pad_replay": True,
            "phase578_reference_row_sha256": reference[
                "_source_raw_row_sha256"
            ],
            "candidate_coordinates": [],
            "candidate_mechanism_formulas": [],
        })
        exact_rows += 1
    # hidden_steps[f+1] is the forward in which suffix token f was fed back.
    for feedback_index in range(min(suffix_width - 1, FEEDBACK_WIDTH)):
        step = hidden_steps[feedback_index + 1]
        if any(tuple(value.shape) != (len(rows), 1, hidden_size) for value in step):
            raise RuntimeError("feedback hidden-state shape drift")
        stacked = torch.stack([
            value[:, -1, :].detach().to(device="cpu", dtype=torch.bfloat16)
            for value in step
        ], dim=1).contiguous()
        if not bool(executed_mask[:, feedback_index].all().item()):
            raise RuntimeError("generate feedback execution is not batch-uniform")
        feedback[:, :, feedback_index, :] = stacked
        del stacked
    prompt_mask = encoded["attention_mask"].to(device="cpu", dtype=torch.bool).contiguous()
    payload = {
        "metadata_rows": metadata,
        "prefill_residual": prefill,
        "prompt_mask": prompt_mask,
        "feedback_residual": feedback,
        "feedback_executed_mask": executed_mask,
        "feedback_pre_eos_mask": pre_eos_mask,
    }
    report = {
        "batch_index": batch_index,
        "case_count": len(rows),
        "first_ordinal": rows[0]["ordinal"],
        "last_ordinal": rows[-1]["ordinal"],
        "exact_replay_rows": exact_rows,
        "suffix_width": suffix_width,
        "layer_count": layer_count,
        "hidden_size": hidden_size,
        "prompt_width": prompt_width,
    }
    del generated, sequences, hidden_steps, gpu_inputs, encoded
    return payload, report


def _authorization_path(
    pending_root: Path, model: str,
) -> Path:
    return pending_root / (
        f"worker_authorization_{ELIGIBLE_MODELS.index(model):02d}_{model}.json"
    )


def verify_worker_authorization(
    model: str, mode: str, pending_root: Path,
    authorization_path: Path, authorization_nonce: str,
) -> dict[str, Any]:
    expected = _authorization_path(pending_root, model)
    if authorization_path.resolve(strict=True) != expected.resolve(strict=True):
        raise RuntimeError("worker authorization path drift")
    authorization = read_json(expected)
    stage_start = pending_root / "stage_start.json"
    if not all((
        authorization.get("schema_version") == "phase579_worker_authorization.v1",
        authorization.get("phase_id") == PHASE,
        authorization.get("mode") == mode,
        authorization.get("model") == model,
        authorization.get("model_order_index") == ELIGIBLE_MODELS.index(model),
        authorization.get("authorization_nonce") == authorization_nonce,
        secrets.compare_digest(
            str(authorization.get("authorization_nonce", "")),
            authorization_nonce,
        ),
        authorization.get("parent_pid") == os.getppid(),
        authorization.get("pending_root") == str(pending_root),
        authorization.get("runner_source_sha256")
        == sha256_file(Path(__file__).resolve()),
        stage_start.is_file(),
        authorization.get("stage_start_sha256") == sha256_file(stage_start),
    )):
        raise RuntimeError("Phase579 worker authorization contract failed")
    stage_payload = read_json(stage_start)
    prior = authorization.get("prior_terminal_model_statuses")
    if not isinstance(prior, list) or [item.get("model") for item in prior] != list(
        ELIGIBLE_MODELS[:ELIGIBLE_MODELS.index(model)]
    ):
        raise RuntimeError("Phase579 worker predecessor order drift")
    for item in prior:
        status_path = pending_root / item["relative_path"]
        status = read_json(status_path)
        if not all((
            sha256_file(status_path) == item.get("sha256"),
            status.get("status") in {"complete", "failed"},
            status.get("cleanup", {}).get("cleanup_pass") is True,
            status.get("cleanup", {}).get("allocated_after_release") == 0,
            status.get("cleanup", {}).get("reserved_after_release") == 0,
        )):
            raise RuntimeError("Phase579 predecessor is not clean and terminal")
    if mode == "trace":
        engineering = authorization.get("engineering_verification")
        if not isinstance(engineering, dict):
            raise RuntimeError("Phase579 trace engineering payload is absent")
        if not all((
            engineering == stage_payload.get("engineering_verification"),
            engineering.get("schema_version")
            == "phase579_execution_verification.v1",
            engineering.get("phase_id") == PHASE,
            engineering.get("mode") == "engineering",
            engineering.get("passed") is True,
            all(engineering.get("checks", {}).values()),
            engineering.get("case_count_per_model") == 8,
            engineering.get("shard_count_per_model") == 1,
            set(engineering.get("model_status_checks", {}))
            == set(ELIGIBLE_MODELS),
            all(engineering.get("model_status_checks", {}).values()),
            isinstance(engineering.get("execution_receipt_sha256"), str),
            len(engineering.get("execution_receipt_sha256", "")) == 64,
        )):
            raise RuntimeError("Phase579 trace engineering authorization drift")
    elif authorization.get("engineering_verification") is not None:
        raise RuntimeError("engineering worker received recursive authorization")
    return authorization


def worker(
    model: str, mode: str, pending_root: Path,
    authorization_path: Path, authorization_nonce: str,
) -> int:
    if model not in ELIGIBLE_MODELS or mode not in {"engineering", "trace"}:
        raise RuntimeError("invalid Phase579 worker model/mode")
    bridge = verify_bridge()
    pending_root = pending_root.resolve(strict=True)
    final_root = ENGINEERING_DIR if mode == "engineering" else TRACE_DIR
    if (
        pending_root.parent != final_root.parent.resolve(strict=True)
        or not pending_root.name.startswith(f".{final_root.name}.pending-")
    ):
        raise RuntimeError("worker pending root escaped frozen result namespace")
    authorization = verify_worker_authorization(
        model, mode, pending_root, authorization_path, authorization_nonce
    )
    model_dir = pending_root / f"{ELIGIBLE_MODELS.index(model):02d}_{model}"
    model_dir.mkdir(parents=False, exist_ok=False)
    attempts, restore_guard = install_research_access_guard(model, pending_root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    adapter = None
    started = now()
    failure: BaseException | None = None
    cleanup_report: dict[str, Any] | None = None
    model_identity: dict[str, Any] | None = None
    artifact_verification: dict[str, Any] | None = None
    raw_identity: dict[str, Any] | None = None
    shard_registry: list[dict[str, Any]] = []
    trace_manifest: dict[str, Any] | None = None
    peak_allocated = 0
    try:
        if any(name in sys.modules for name in (
            "phase578_retrieval_closure", "model_utils",
            "phase983_cross_model_engine",
        )):
            raise RuntimeError("forbidden/preloaded execution module detected")
        import torch
        import phase983_cross_model_engine as engine

        imported_engine = Path(engine.__file__).resolve(strict=True)
        registry_module = sys.modules.get("model_registry")
        if not all((
            imported_engine == ENGINE_PATH.resolve(strict=True),
            sha256_file(imported_engine) == ENGINE_EXPECTED_SHA256,
            registry_module is not None,
            Path(registry_module.__file__).resolve(strict=True)
            == MODEL_REGISTRY_PATH.resolve(strict=True),
        )):
            raise RuntimeError("execution engine/model registry import shadow detected")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one CUDA device is required")
        if int(torch.cuda.memory_allocated()) != 0 or int(torch.cuda.memory_reserved()) != 0:
            raise RuntimeError("dirty CUDA allocator baseline")
        torch.cuda.reset_peak_memory_stats()
        manifest = read_manifest()
        references, raw_identity = read_phase578_repeat1(model)
        if [row["case_id"] for row in references] != [
            row["case_id"] for row in manifest
        ]:
            raise RuntimeError("Phase578/Phase579 case order mismatch")
        if mode == "engineering":
            manifest = [manifest[index] for index in ENGINEERING_ORDINALS]
            references = [references[index] for index in ENGINEERING_ORDINALS]
        protocol = read_json(PROTOCOL_PATH)
        artifact_verification = verify_model_artifacts(model, protocol)
        adapter = engine.load_model_adapter(model)
        adapter.tokenizer.padding_side = "left"
        model_identity = adapter.identity
        expected_count = 8 if mode == "engineering" else 336
        if len(manifest) != expected_count or len(references) != expected_count:
            raise RuntimeError("Phase579 selected trace denominator drift")
        layer_count: int | None = None
        hidden_size: int | None = None
        total_bytes = 0
        for start in range(0, expected_count, BATCH_SIZE):
            batch_index = start // BATCH_SIZE
            payload, batch_report = collect_batch(
                adapter, model, batch_index,
                manifest[start:start + BATCH_SIZE],
                references[start:start + BATCH_SIZE],
            )
            engineering_reexecution_exact: bool | None = None
            if mode == "engineering":
                repeated_payload, repeated_report = collect_batch(
                    adapter, model, batch_index,
                    manifest[start:start + BATCH_SIZE],
                    references[start:start + BATCH_SIZE],
                )
                engineering_reexecution_exact = (
                    repeated_report == batch_report
                    and _trace_payloads_exact(payload, repeated_payload)
                )
                del repeated_payload
                if not engineering_reexecution_exact:
                    raise RuntimeError(
                        "engineering hidden-state reexecution was not bit exact"
                    )
            if layer_count is None:
                layer_count = batch_report["layer_count"]
                hidden_size = batch_report["hidden_size"]
            elif (
                layer_count != batch_report["layer_count"]
                or hidden_size != batch_report["hidden_size"]
            ):
                raise RuntimeError("trace residual shape changed between batches")
            shard_path = model_dir / f"trace_shard_{batch_index:04d}.pt"
            shard_entry = write_torch_shard_exclusive(shard_path, payload)
            shard_entry.update(batch_report)
            shard_entry["engineering_reexecution_exact"] = (
                engineering_reexecution_exact
            )
            shard_registry.append(shard_entry)
            total_bytes += shard_entry["size_bytes"]
            del payload
            gc.collect()
            peak_allocated = max(
                peak_allocated, int(torch.cuda.max_memory_allocated())
            )
            done = min(start + BATCH_SIZE, expected_count)
            if batch_index == 0 or done == expected_count or batch_index % 4 == 3:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {mode}/{model} "
                    f"{done}/{expected_count}", flush=True,
                )
        trace_manifest = {
            "schema_version": "phase579_model_trace_manifest.v1",
            "phase_id": PHASE,
            "created_at_utc": now(),
            "mode": mode,
            "model": model,
            "model_order_index": ELIGIBLE_MODELS.index(model),
            "split": "development",
            "case_count": expected_count,
            "first_ordinal": 0,
            "last_ordinal": 7 if mode == "engineering" else 335,
            "batch_size": BATCH_SIZE,
            "shard_count": len(shard_registry),
            "layer_count": layer_count,
            "hidden_size": hidden_size,
            "layer_index_semantics": LAYER_INDEX_SEMANTICS,
            "feedback_index_semantics": FEEDBACK_INDEX_SEMANTICS,
            "shard_keys": list(SHARD_KEYS),
            "storage_dtype": "torch.bfloat16",
            "phase578_repeat1_raw_identity": raw_identity,
            "phase578_repeat1_exact_replay_case_count": expected_count,
            "internal_reexecution_count": 2 if mode == "engineering" else 1,
            "engineering_reexecution_exact": (
                all(item["engineering_reexecution_exact"] is True
                    for item in shard_registry)
                if mode == "engineering" else None
            ),
            "all_shards_finite": True,
            "all_shards_roundtrip_exact": all(
                item["torch_save_load_roundtrip_exact"] for item in shard_registry
            ),
            "shard_total_bytes": total_bytes,
            "shards": shard_registry,
            "attention_collected": False,
            "scores_collected": False,
            "hooks_registered": 0,
            "causal_intervention": False,
            "candidate_coordinates": [],
            "candidate_mechanism_formulas": [],
            "confirmation_accessed": False,
            "heldout_accessed": False,
            "sealed_accessed": False,
        }
        write_json(model_dir / "trace_manifest.json", trace_manifest)
    except BaseException as exc:
        failure = exc
        exc.__traceback__ = None
    finally:
        try:
            import phase983_cross_model_engine as cleanup_engine
            if Path(cleanup_engine.__file__).resolve(strict=True) != ENGINE_PATH.resolve(
                strict=True
            ):
                raise RuntimeError("cleanup engine binding drift")
            cleanup_report = strict_release(cleanup_engine, adapter)
        except BaseException as cleanup_exc:
            if failure is None:
                failure = cleanup_exc
            cleanup_exc.__traceback__ = None
            cleanup_report = {
                "cleanup_pass": False,
                "allocated_after_release": None,
                "reserved_after_release": None,
                "error_type": type(cleanup_exc).__name__,
                "error": str(cleanup_exc),
            }
        adapter = None
        if failure is not None:
            failure.__traceback__ = None
        restore_guard()
    expected_count = 8 if mode == "engineering" else 336
    completed = all((
        failure is None,
        cleanup_report is not None,
        cleanup_report.get("cleanup_pass") is True,
        len(shard_registry) == expected_count // BATCH_SIZE,
        trace_manifest is not None,
    ))
    status = {
        "schema_version": "phase579_model_worker_status.v1",
        "phase_id": PHASE,
        "created_at_utc": now(),
        "started_at_utc": started,
        "mode": mode,
        "model": model,
        "model_order_index": ELIGIBLE_MODELS.index(model),
        "status": "complete" if completed else "failed",
        "bridge_identity": bridge,
        "generation_contract": GENERATION_CONTRACT,
        "generation_contract_sha256": GENERATION_CONTRACT_SHA256,
        "case_count": expected_count if trace_manifest is not None else 0,
        "expected_case_count": expected_count,
        "shard_count": len(shard_registry),
        "expected_shard_count": expected_count // BATCH_SIZE,
        "internal_reexecution_count": 2 if mode == "engineering" else 1,
        "engineering_reexecution_exact": (
            trace_manifest.get("engineering_reexecution_exact")
            if trace_manifest is not None else None
        ),
        "trace_manifest_sha256": (
            sha256_file(model_dir / "trace_manifest.json")
            if (model_dir / "trace_manifest.json").is_file() else None
        ),
        "phase578_repeat1_raw_identity": raw_identity,
        "model_identity": model_identity,
        "model_artifact_verification": artifact_verification,
        "worker_authorization_sha256": sha256_file(authorization_path),
        "worker_authorization_parent_pid": authorization.get("parent_pid"),
        "peak_cuda_memory_allocated": peak_allocated,
        "cleanup": cleanup_report,
        "research_access_attempts": attempts,
        "error_type": None if failure is None else type(failure).__name__,
        "error": None if failure is None else str(failure),
        "traceback_persisted": False,
        "automatic_fallback_used": False,
        "activation_collected": completed,
        "hidden_states_requested": True,
        "all_layers_collected": completed,
        "all_prompt_positions_collected": completed,
        "feedback_residuals_collected": completed,
        "attentions_requested": False,
        "scores_requested": False,
        "hooks_registered": 0,
        "causal_intervention": False,
        "candidate_coordinates": [],
        "candidate_mechanism_formulas": [],
        "confirmation_accessed": False,
        "heldout_accessed": False,
        "sealed_accessed": False,
    }
    write_json(model_dir / "status.json", status)
    print(json.dumps({
        "mode": mode,
        "model": model,
        "status": status["status"],
        "case_count": status["case_count"],
        "shard_count": status["shard_count"],
        "cleanup_pass": cleanup_report.get("cleanup_pass") if cleanup_report else False,
    }, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if completed else 2


def acquire_lease(mode: str) -> Any:
    path = ROOT / f"tests/glm5/result/.phase579_residual_{mode}.lease"
    handle = path.open("a+b", buffering=0)
    if path.stat().st_size == 0:
        handle.write(b"0")
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"Phase579 residual {mode} execution is already leased") from exc
    return handle


def release_lease(handle: Any) -> None:
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


def _quarantine_pending(final_dir: Path) -> list[str]:
    quarantined: list[str] = []
    prefix = f".{final_dir.name}.pending-"
    for path in sorted(final_dir.parent.iterdir()):
        if path.is_dir() and path.name.startswith(prefix):
            destination = path.with_name(
                f".{final_dir.name}.aborted-"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            )
            path.rename(destination)
            quarantined.append(destination.name)
    return quarantined


def _artifact_registry(root: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("Phase579 execution artifact symlink forbidden")
        if path.is_file():
            output.append({
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return output


def _blocked_model_receipt(bridge: dict[str, Any], mode: str) -> dict[str, Any]:
    summary = read_json(PHASE578_SUMMARY_PATH)
    passed = summary.get("behavior_passed_models")
    blocked = summary.get(
        "behavior_failed_models",
        summary.get("behavior_blocked_models"),
    )
    if passed != list(ELIGIBLE_MODELS) or (
        blocked is not None and "deepseek7b" not in blocked
    ):
        raise RuntimeError("Phase578 behavior eligibility summary drift")
    return {
        "schema_version": "phase579_blocked_model_receipt.v1",
        "phase_id": PHASE,
        "created_at_utc": now(),
        "mode": mode,
        "model": "deepseek7b",
        "phase578_behavior_gate_passed": False,
        "block_reason": "failed preregistered Phase578 development behavior gate",
        "phase578_summary_sha256": sha256_file(PHASE578_SUMMARY_PATH),
        "worker_authorized": False,
        "worker_started": False,
        "cuda_model_loaded": False,
        "hidden_states_requested": False,
        "trace_artifacts_written": False,
        "bridge_identity": bridge,
        "candidate_coordinates": [],
        "candidate_mechanism_formulas": [],
    }


def parent_run(mode: str) -> dict[str, Any]:
    if mode not in {"engineering", "trace"}:
        raise RuntimeError("invalid Phase579 parent mode")
    bridge = verify_bridge()
    final_dir = ENGINEERING_DIR if mode == "engineering" else TRACE_DIR
    if final_dir.exists():
        raise RuntimeError(f"terminal Phase579 {mode} result already exists")
    engineering_verification: dict[str, Any] | None = None
    if mode == "trace":
        engineering_verification = verify_execution(
            "engineering", rehash_models=True
        )
        if engineering_verification.get("passed") is not True:
            raise RuntimeError("Phase579 engineering did not authorize full trace")
    lease = acquire_lease(mode)
    pending: Path | None = None
    try:
        quarantined = _quarantine_pending(final_dir)
        pending = final_dir.with_name(
            f".{final_dir.name}.pending-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            f"-pid{os.getpid()}"
        )
        pending.mkdir(parents=True, exist_ok=False)
        write_json(pending / "stage_start.json", {
            "schema_version": "phase579_execution_stage_start.v1",
            "phase_id": PHASE,
            "created_at_utc": now(),
            "mode": mode,
            "required_model_order": list(ELIGIBLE_MODELS),
            "blocked_models": list(BLOCKED_MODELS),
            "bridge_identity": bridge,
            "generation_contract": GENERATION_CONTRACT,
            "generation_contract_sha256": GENERATION_CONTRACT_SHA256,
            "engineering_verification": engineering_verification,
            "quarantined_prior_pending": quarantined,
            "candidate_coordinates": [],
            "candidate_mechanism_formulas": [],
        })
        write_json(
            pending / "blocked_model_receipt.json",
            _blocked_model_receipt(bridge, mode),
        )
        attempts: list[dict[str, Any]] = []
        prior_terminal: list[dict[str, Any]] = []
        fatal_cleanup = False
        for model in ELIGIBLE_MODELS:
            nonce = secrets.token_hex(32)
            authorization_path = _authorization_path(pending, model)
            authorization = {
                "schema_version": "phase579_worker_authorization.v1",
                "phase_id": PHASE,
                "created_at_utc": now(),
                "mode": mode,
                "model": model,
                "model_order_index": ELIGIBLE_MODELS.index(model),
                "authorization_nonce": nonce,
                "parent_pid": os.getpid(),
                "pending_root": str(pending.resolve(strict=True)),
                "runner_source_sha256": sha256_file(Path(__file__).resolve()),
                "stage_start_sha256": sha256_file(pending / "stage_start.json"),
                "prior_terminal_model_statuses": prior_terminal,
                "engineering_verification": engineering_verification,
                "blocked_models": list(BLOCKED_MODELS),
            }
            write_json(authorization_path, authorization)
            command = [
                str(FORMAL_PYTHON), str(Path(__file__).resolve()),
                "--worker", "--mode", mode, "--model", model,
                "--pending-root", str(pending),
                "--authorization", str(authorization_path),
                "--authorization-nonce", nonce,
            ]
            started = time.time()
            process = subprocess.run(command, cwd=str(ROOT), check=False)
            status_path = (
                pending / f"{ELIGIBLE_MODELS.index(model):02d}_{model}/status.json"
            )
            status = read_json(status_path) if status_path.is_file() else {}
            cleanup_pass = status.get("cleanup", {}).get("cleanup_pass") is True
            attempt = {
                "model": model,
                "model_order_index": ELIGIBLE_MODELS.index(model),
                "child_exit_code": process.returncode,
                "elapsed_seconds": round(time.time() - started, 6),
                "status": status.get("status", "missing"),
                "cleanup_pass": cleanup_pass,
                "status_sha256": (
                    sha256_file(status_path) if status_path.is_file() else None
                ),
                "authorization_sha256": sha256_file(authorization_path),
            }
            attempts.append(attempt)
            if status.get("status") in {"complete", "failed"} and cleanup_pass:
                prior_terminal.append({
                    "model": model,
                    "relative_path": str(status_path.relative_to(pending)).replace(
                        "\\", "/"
                    ),
                    "sha256": sha256_file(status_path),
                })
            if not cleanup_pass:
                fatal_cleanup = True
                break
        completed = [
            item["model"] for item in attempts
            if item["status"] == "complete" and item["child_exit_code"] == 0
            and item["cleanup_pass"] is True
        ]
        attempted = [item["model"] for item in attempts]
        failed = [model for model in attempted if model not in completed]
        not_attempted = [model for model in ELIGIBLE_MODELS if model not in attempted]
        passed = (
            completed == list(ELIGIBLE_MODELS)
            and not failed and not not_attempted and not fatal_cleanup
        )
        registry = _artifact_registry(pending)
        receipt = {
            "schema_version": "phase579_execution_receipt.v1",
            "phase_id": PHASE,
            "created_at_utc": now(),
            "mode": mode,
            "required_model_order": list(ELIGIBLE_MODELS),
            "attempted_models_in_order": attempted,
            "completed_models": completed,
            "failed_models": failed,
            "not_attempted_models": not_attempted,
            "blocked_models": list(BLOCKED_MODELS),
            "blocked_model_receipt_sha256": sha256_file(
                pending / "blocked_model_receipt.json"
            ),
            "attempts": attempts,
            "fatal_cleanup_failure": fatal_cleanup,
            "engineering_qualification_passed": mode == "engineering" and passed,
            "full_development_trace_authorized": mode == "engineering" and passed,
            "full_development_trace_complete": mode == "trace" and passed,
            "bridge_identity": bridge,
            "engineering_verification": engineering_verification,
            "artifact_registry_before_receipt": registry,
            "artifact_registry_sha256": sha256_bytes(
                canonical_json(registry).encode("utf-8")
            ),
            "development_case_count_per_completed_model": (
                8 if mode == "engineering" else 336
            ),
            "all_layers_all_prompt_and_feedback_positions": passed,
            "attention_collected": False,
            "scores_collected": False,
            "hooks_registered": 0,
            "causal_intervention": False,
            "confirmation_accessed": False,
            "heldout_accessed": False,
            "sealed_accessed": False,
            "candidate_coordinates": [],
            "candidate_mechanism_formulas": [],
        }
        write_json(pending / "execution_receipt.json", receipt)
        pending.rename(final_dir)
        pending = None
        return receipt
    finally:
        release_lease(lease)


def _expected_model_files(model_root: Path) -> set[str]:
    return {
        "status.json", "trace_manifest.json",
        *{
            path.name for path in model_root.glob("trace_shard_*.pt")
            if path.is_file()
        },
    }


def _verify_metadata_replay(
    metadata: dict[str, Any], manifest_row: dict[str, Any],
    reference: dict[str, Any], model: str, prompt_mask_row: Any,
    executed_row: Any, pre_eos_row: Any,
) -> None:
    import torch

    ids = metadata.get("input_token_ids")
    suffix = metadata.get("generation_token_ids")
    if not all((
        metadata.get("case_id") == manifest_row.get("case_id")
        == reference.get("case_id"),
        metadata.get("ordinal") == manifest_row.get("ordinal"),
        metadata.get("model") == model,
        metadata.get("split") == "development",
        ids == reference.get("input_token_ids"),
        metadata.get("input_token_ids_sha256")
        == sha256_bytes(canonical_json(ids).encode("utf-8")),
        metadata.get("rendered_prompt_sha256")
        == reference.get("rendered_prompt_sha256"),
        suffix == reference.get("full_generated_suffix_token_ids"),
        metadata.get("generation_token_ids_sha256")
        == sha256_bytes(canonical_json(suffix).encode("utf-8")),
        metadata.get("effective_eos_token_ids")
        == reference.get("effective_eos_token_ids"),
        metadata.get("pad_token_id") == reference.get("pad_token_id"),
        metadata.get("first_eos_index") == reference.get("first_eos_index"),
        metadata.get("eos_seen") == reference.get("eos_seen"),
        metadata.get("post_eos_token_ids") == reference.get("post_eos_token_ids"),
        metadata.get("post_eos_tokens_all_pad") is True,
        metadata.get("phase578_reference_row_sha256")
        == reference.get("_source_raw_row_sha256"),
        metadata.get("phase578_repeat1_exact_input_replay") is True,
        metadata.get("phase578_repeat1_exact_suffix_eos_pad_replay") is True,
        metadata.get("candidate_coordinates") == [],
        metadata.get("candidate_mechanism_formulas") == [],
    )):
        raise RuntimeError(f"trace metadata replay drift: {model}/{metadata.get('case_id')}")
    expected_executed, expected_pre_eos = _expected_feedback_masks(
        suffix, reference.get("first_eos_index")
    )
    if not torch.equal(executed_row, torch.tensor(expected_executed, dtype=torch.bool)):
        raise RuntimeError("trace feedback executed mask replay drift")
    if not torch.equal(pre_eos_row, torch.tensor(expected_pre_eos, dtype=torch.bool)):
        raise RuntimeError("trace feedback pre-EOS mask replay drift")
    prompt_count = len(ids)
    if int(prompt_mask_row.sum().item()) != prompt_count:
        raise RuntimeError("trace prompt mask valid-token count drift")
    for role in ("focus", "comparison", "query_anchor"):
        role_entry = metadata.get("role_token_positions", {}).get(role)
        if not isinstance(role_entry, dict):
            raise RuntimeError("trace role-token inventory missing")


def _verify_shard_file(
    path: Path, registry: dict[str, Any], manifest_rows: list[dict[str, Any]],
    references_by_case: dict[str, dict[str, Any]], model: str,
) -> list[str]:
    import torch

    if (
        path.is_symlink()
        or path.stat().st_size != registry.get("size_bytes")
        or sha256_file(path) != registry.get("sha256")
    ):
        raise RuntimeError(f"trace shard file identity drift: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    contract = _shard_tensor_contract(payload)
    for key in (
        "batch_size", "layer_count", "prompt_width", "hidden_size",
        "prefill_shape", "feedback_shape",
    ):
        if contract[key] != registry.get(key):
            raise RuntimeError(f"trace shard registry contract drift: {path.name}/{key}")
    manifest_by_case = {row["case_id"]: row for row in manifest_rows}
    case_ids: list[str] = []
    for index, metadata in enumerate(payload["metadata_rows"]):
        case_id = metadata.get("case_id")
        if case_id not in manifest_by_case or case_id not in references_by_case:
            raise RuntimeError("trace shard contains an unknown development case")
        _verify_metadata_replay(
            metadata, manifest_by_case[case_id], references_by_case[case_id], model,
            payload["prompt_mask"][index],
            payload["feedback_executed_mask"][index],
            payload["feedback_pre_eos_mask"][index],
        )
        case_ids.append(case_id)
    del payload
    return case_ids


def verify_execution(mode: str, *, rehash_models: bool = True) -> dict[str, Any]:
    if mode not in {"engineering", "trace"}:
        raise RuntimeError("invalid Phase579 verification mode")
    bridge = verify_bridge()
    protocol = read_json(PROTOCOL_PATH)
    root = ENGINEERING_DIR if mode == "engineering" else TRACE_DIR
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"missing/aliased Phase579 {mode} execution")
    receipt_path = root / "execution_receipt.json"
    receipt = read_json(receipt_path)
    registry = receipt.get("artifact_registry_before_receipt")
    if not isinstance(registry, list):
        raise RuntimeError("Phase579 receipt artifact registry missing")
    expected_paths = {item["path"] for item in registry}
    actual_paths = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*") if path.is_file()
    }
    if actual_paths != expected_paths | {"execution_receipt.json"}:
        raise RuntimeError("Phase579 execution artifact closure drift")
    for item in registry:
        path = root / item["path"]
        if (
            path.is_symlink()
            or path.stat().st_size != item["size_bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise RuntimeError(f"Phase579 execution artifact drift: {item['path']}")
    if receipt.get("artifact_registry_sha256") != sha256_bytes(
        canonical_json(registry).encode("utf-8")
    ):
        raise RuntimeError("Phase579 receipt artifact registry hash drift")
    blocked = read_json(root / "blocked_model_receipt.json")
    blocked_ok = all((
        blocked.get("model") == "deepseek7b",
        blocked.get("phase578_behavior_gate_passed") is False,
        blocked.get("worker_authorized") is False,
        blocked.get("worker_started") is False,
        blocked.get("cuda_model_loaded") is False,
        blocked.get("hidden_states_requested") is False,
        blocked.get("trace_artifacts_written") is False,
        receipt.get("blocked_model_receipt_sha256")
        == sha256_file(root / "blocked_model_receipt.json"),
    ))
    all_manifest_rows = read_manifest()
    selected_manifest = (
        [all_manifest_rows[index] for index in ENGINEERING_ORDINALS]
        if mode == "engineering" else all_manifest_rows
    )
    expected_count = 8 if mode == "engineering" else 336
    attempts_by_model = {item["model"]: item for item in receipt.get("attempts", [])}
    model_checks: dict[str, bool] = {}
    model_trace_hashes: dict[str, str] = {}
    for model in ELIGIBLE_MODELS:
        model_root = root / f"{ELIGIBLE_MODELS.index(model):02d}_{model}"
        status_path = model_root / "status.json"
        trace_manifest_path = model_root / "trace_manifest.json"
        status = read_json(status_path)
        trace_manifest = read_json(trace_manifest_path)
        references, raw_identity = read_phase578_repeat1(model)
        if mode == "engineering":
            references = [references[index] for index in ENGINEERING_ORDINALS]
        references_by_case = {row["case_id"]: row for row in references}
        shard_registry = trace_manifest.get("shards")
        if not isinstance(shard_registry, list):
            raise RuntimeError("Phase579 trace shard registry missing")
        observed_case_ids: list[str] = []
        for shard in shard_registry:
            shard_path = model_root / shard["path"]
            observed_case_ids.extend(_verify_shard_file(
                shard_path, shard, selected_manifest, references_by_case, model
            ))
        expected_case_ids = [row["case_id"] for row in selected_manifest]
        current_artifact = (
            verify_model_artifacts(model, protocol) if rehash_models else None
        )
        artifact_report = status.get("model_artifact_verification")
        artifact_payload = dict(artifact_report or {})
        reported_payload_hash = artifact_payload.pop(
            "verification_payload_sha256", None
        )
        loaded = status.get("model_identity", {})
        quant = loaded.get("loaded_quantization", {})
        attempt = attempts_by_model.get(model, {})
        authorization_path = _authorization_path(root, model)
        access = status.get("research_access_attempts", {})
        status_ok = all((
            status.get("status") == "complete",
            status.get("mode") == mode,
            status.get("model") == model,
            status.get("model_order_index") == ELIGIBLE_MODELS.index(model),
            status.get("case_count") == expected_count,
            status.get("expected_case_count") == expected_count,
            status.get("shard_count") == expected_count // BATCH_SIZE,
            status.get("expected_shard_count") == expected_count // BATCH_SIZE,
            status.get("trace_manifest_sha256") == sha256_file(trace_manifest_path),
            status.get("phase578_repeat1_raw_identity") == raw_identity,
            status.get("cleanup", {}).get("cleanup_pass") is True,
            status.get("cleanup", {}).get("allocated_after_release") == 0,
            status.get("cleanup", {}).get("reserved_after_release") == 0,
            all(value == 0 for value in access.values()),
            loaded.get("weights_loaded") is True,
            loaded.get("gpu_used") is True,
            loaded.get("loaded_attn_implementation") == "sdpa",
            loaded.get("cuda_only_no_cpu_or_disk_offload") is True,
            quant.get("load_in_8bit") is True,
            quant.get("non_quantized_dtype") == "torch.bfloat16",
            status.get("hidden_states_requested") is True,
            status.get("all_layers_collected") is True,
            status.get("all_prompt_positions_collected") is True,
            status.get("feedback_residuals_collected") is True,
            status.get("attentions_requested") is False,
            status.get("scores_requested") is False,
            status.get("hooks_registered") == 0,
            status.get("causal_intervention") is False,
            status.get("candidate_coordinates") == [],
            status.get("candidate_mechanism_formulas") == [],
            status.get("automatic_fallback_used") is False,
            status.get("confirmation_accessed") is False,
            status.get("heldout_accessed") is False,
            status.get("sealed_accessed") is False,
            isinstance(artifact_report, dict),
            reported_payload_hash == sha256_bytes(
                canonical_json(artifact_payload).encode("utf-8")
            ),
            current_artifact is None or current_artifact == artifact_report,
            authorization_path.is_file(),
            status.get("worker_authorization_sha256")
            == sha256_file(authorization_path),
            attempt.get("child_exit_code") == 0,
            attempt.get("status_sha256") == sha256_file(status_path),
            attempt.get("authorization_sha256") == sha256_file(authorization_path),
            observed_case_ids == expected_case_ids,
            trace_manifest.get("case_count") == expected_count,
            trace_manifest.get("shard_count") == expected_count // BATCH_SIZE,
            trace_manifest.get("storage_dtype") == "torch.bfloat16",
            trace_manifest.get("shard_keys") == list(SHARD_KEYS),
            trace_manifest.get("layer_index_semantics") == LAYER_INDEX_SEMANTICS,
            trace_manifest.get("feedback_index_semantics")
            == FEEDBACK_INDEX_SEMANTICS,
            trace_manifest.get("phase578_repeat1_raw_identity") == raw_identity,
            trace_manifest.get("phase578_repeat1_exact_replay_case_count")
            == expected_count,
            trace_manifest.get("internal_reexecution_count")
            == (2 if mode == "engineering" else 1),
            trace_manifest.get("engineering_reexecution_exact")
            == (True if mode == "engineering" else None),
            status.get("internal_reexecution_count")
            == (2 if mode == "engineering" else 1),
            status.get("engineering_reexecution_exact")
            == (True if mode == "engineering" else None),
            trace_manifest.get("all_shards_finite") is True,
            trace_manifest.get("all_shards_roundtrip_exact") is True,
            trace_manifest.get("attention_collected") is False,
            trace_manifest.get("scores_collected") is False,
            trace_manifest.get("hooks_registered") == 0,
            trace_manifest.get("causal_intervention") is False,
            trace_manifest.get("candidate_coordinates") == [],
            trace_manifest.get("candidate_mechanism_formulas") == [],
        ))
        model_checks[model] = status_ok
        model_trace_hashes[model] = sha256_file(trace_manifest_path)
    receipt_checks = {
        "bridge": receipt.get("bridge_identity") == bridge,
        "mode": receipt.get("mode") == mode,
        "model_order": receipt.get("required_model_order")
        == receipt.get("attempted_models_in_order") == list(ELIGIBLE_MODELS),
        "all_completed": receipt.get("completed_models") == list(ELIGIBLE_MODELS),
        "none_failed": receipt.get("failed_models") == [],
        "none_not_attempted": receipt.get("not_attempted_models") == [],
        "blocked_only_deepseek": receipt.get("blocked_models") == list(BLOCKED_MODELS)
        and blocked_ok,
        "cleanup": all(
            item.get("cleanup_pass") is True for item in receipt.get("attempts", [])
        ),
        "child_exit_codes": all(
            item.get("child_exit_code") == 0 for item in receipt.get("attempts", [])
        ),
        "model_statuses_and_shards": all(model_checks.values()),
        "mode_gate": (
            receipt.get("engineering_qualification_passed") is True
            and receipt.get("full_development_trace_authorized") is True
            and receipt.get("full_development_trace_complete") is False
            if mode == "engineering"
            else receipt.get("full_development_trace_complete") is True
            and receipt.get("engineering_qualification_passed") is False
            and receipt.get("full_development_trace_authorized") is False
        ),
        "no_preselection_or_intervention": receipt.get("candidate_coordinates") == []
        and receipt.get("candidate_mechanism_formulas") == []
        and receipt.get("attention_collected") is False
        and receipt.get("scores_collected") is False
        and receipt.get("hooks_registered") == 0
        and receipt.get("causal_intervention") is False,
        "development_only": receipt.get("confirmation_accessed") is False
        and receipt.get("heldout_accessed") is False
        and receipt.get("sealed_accessed") is False,
    }
    if mode == "trace":
        receipt_checks["engineering_bridge"] = receipt.get(
            "engineering_verification"
        ) == verify_execution("engineering", rehash_models=False)
    elif receipt.get("engineering_verification") is not None:
        receipt_checks["engineering_bridge"] = False
    if not all(receipt_checks.values()):
        raise RuntimeError(
            f"Phase579 {mode} execution verification failed: {receipt_checks}"
        )
    return {
        "schema_version": "phase579_execution_verification.v1",
        "phase_id": PHASE,
        "mode": mode,
        "passed": True,
        "checks": receipt_checks,
        "execution_receipt_sha256": sha256_file(receipt_path),
        "model_status_checks": model_checks,
        "model_trace_manifest_sha256": model_trace_hashes,
        "case_count_per_model": expected_count,
        "shard_count_per_model": expected_count // BATCH_SIZE,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--engineering", action="store_true")
    group.add_argument("--verify-engineering", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--verify", action="store_true")
    group.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("engineering", "trace"))
    parser.add_argument("--model", choices=ELIGIBLE_MODELS)
    parser.add_argument("--pending-root", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-nonce")
    args = parser.parse_args()
    if args.worker:
        if any(value is None for value in (
            args.mode, args.model, args.pending_root,
            args.authorization, args.authorization_nonce,
        )):
            raise RuntimeError("worker requires frozen mode/model/pending authorization")
        raise SystemExit(worker(
            args.model, args.mode, args.pending_root,
            args.authorization, args.authorization_nonce,
        ))
    if any(value is not None for value in (
        args.mode, args.model, args.pending_root,
        args.authorization, args.authorization_nonce,
    )):
        raise RuntimeError("worker-only arguments supplied to Phase579 parent")
    if args.engineering:
        result = parent_run("engineering")
    elif args.verify_engineering:
        result = verify_execution("engineering", rehash_models=True)
    elif args.run:
        result = parent_run("trace")
    else:
        result = verify_execution("trace", rehash_models=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
