#!/usr/bin/env python3
"""Shared definitions for the Phase 983 cross-model external-contract study.

The study deliberately does *not* compare native ``thinking`` switches.  Its
only intervention is a pair of model-independent natural-language response
instructions, serialized with each model's native chat template.  All states
defined here are externally parsed terminal outcomes, not neural channels.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable


PHASE = 983
SCHEMA_VERSION = 1
EXPERIMENT = "fresh256_option_swap_cross_model_external_contract"

ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
OUT = GLM5 / "result" / "phase983_cross_model_external_contract"

DATASET_PATH = OUT / "dataset.json"
DATASET_AUDIT_PATH = OUT / "dataset_audit.json"
PROTOCOL_PATH = OUT / "protocol_preregistration.json"
ADMISSION_PATH = OUT / "generation_admission.json"
QUALIFICATION_PATH = OUT / "engineering_qualification.json"
QUALIFICATION_LOCK_PATH = OUT / "engineering_qualification.lock"
COMBINED_AUDIT_PATH = OUT / "cross_model_audit.json"
ORCHESTRATOR_STATUS_PATH = OUT / "orchestrator_status.json"
ORCHESTRATOR_LOCK_PATH = OUT / "orchestrator.lock"

MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
MODEL_PATHS = {
    "qwen3": "models/hf/qwen3-4b",
    "glm4": "models/hf/glm4-9b-chat-hf",
    "deepseek7b": "models/hf/deepseek-r1-distill-qwen-7b",
}

ITEM_COUNT = 256
SEMANTIC_INSTANCE_COUNT = 128
TASK_COUNT = 8
STREAMS = (0, 1, 2)
BATCH_SIZE = 8
CHECKPOINTS = (256, 512, 1024, 1536, 2048)
DECISION_CHECKPOINT = 2048
MAX_NEW_TOKENS = 2048
EXPECTED_ROWS_PER_MODEL = ITEM_COUNT * 2 * len(STREAMS)
EXPECTED_ROWS_ALL_MODELS = EXPECTED_ROWS_PER_MODEL * len(MODEL_ORDER)

ARM_A = "A"
ARM_B = "B"
ARMS: dict[str, dict[str, str]] = {
    ARM_A: {
        "name": "external_direct_instruction",
        "role": "baseline",
        "instruction": (
            "Solve the task and answer as directly as possible. Do not show your "
            "reasoning. End with exactly one standalone line FINAL: X, replacing X "
            "with A or B. Output nothing after that line."
        ),
    },
    ARM_B: {
        "name": "external_deliberate_instruction",
        "role": "candidate",
        "instruction": (
            "Work through the task step by step before answering. Then end with "
            "exactly one standalone line FINAL: X, replacing X with A or B. Output "
            "nothing after that line."
        ),
    },
}
PRIMARY_DIRECTION = "B_minus_A"
SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
}
QUANTIZATION = {
    "load_in_8bit": True,
    "non_quantized_compute_dtype": "bfloat16",
    "attn_implementation": "sdpa",
    "same_policy_all_models": True,
}
ENGINE_NAMESPACE = "phase983-cross-model-engine-v1"

TERMINAL_STATES = ("V", "C", "I_protocol", "I_sem")
DIFFICULTIES = ("easy", "hard")
LABELS = ("A", "B")
SWAP_SIDES = ("original", "swapped")

SCRIPT_PATHS = {
    "core": "tests/glm5/phase983_cross_model_core.py",
    "dataset": "tests/glm5/phase983_option_swap_dataset.py",
    "gate": "tests/glm5/phase983_cross_model_gate.py",
    "engine": "tests/glm5/phase983_cross_model_engine.py",
    "protocol": "tests/glm5/phase983_cross_model_protocol.py",
    "qualification": "tests/glm5/phase983_cross_model_qualification.py",
    "admission": "tests/glm5/phase983_cross_model_admission.py",
    "runner": "tests/glm5/phase983_cross_model_runner.py",
    "orchestrator": "tests/glm5/phase983_cross_model_orchestrator.py",
    "audit": "tests/glm5/phase983_cross_model_audit.py",
}
DEPENDENCY_PATHS = {
    "model_registry": "tests/gpt5/model_registry.py",
}

_FINAL_RE = re.compile(r"(?:\A|\n)FINAL: ([AB])\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    require(isinstance(value, str), "text hash input must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def without_fields(value: dict[str, Any], *fields: str) -> dict[str, Any]:
    blocked = set(fields)
    return {key: item for key, item in value.items() if key not in blocked}


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_publish_lock(path: Path, value: dict[str, Any]) -> int:
    """Publish a complete lock document without exposing an empty lock path.

    The candidate is fully written and fsynced in the target directory before
    an atomic, no-overwrite hard-link publishes ``path``.  A hard kill before
    publication leaves no lock path; a hard kill after publication leaves a
    complete, independently recoverable document.  The returned descriptor is
    only a lifetime handle; ownership is authenticated by the JSON PID/start
    token rather than by an inherited advisory lock.
    """
    require(isinstance(value, dict), "lock document must be a JSON object")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode("utf-8")
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".candidate", dir=str(path.parent),
    )
    temporary = Path(temporary_text)
    published = False
    try:
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published = True
        temporary.unlink()
        return os.open(path, os.O_RDONLY)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if published:
            path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def verify_self_hash(
    document: dict[str, Any], field: str, timestamp_field: str | None, label: str,
) -> None:
    omitted = [field]
    if timestamp_field is not None:
        omitted.append(timestamp_field)
        timestamp = document.get(timestamp_field)
        require(isinstance(timestamp, str) and timestamp,
                f"{label} timestamp missing")
        try:
            from datetime import datetime, timezone
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RuntimeError(f"{label} timestamp is not ISO-8601") from exc
        require(parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed),
                f"{label} timestamp is not UTC")
    expected = sha256_json(without_fields(document, *omitted))
    require(document.get(field) == expected, f"{label} self-hash invalid")


def build_file_seals(paths: dict[str, str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, relative in paths.items():
        path = ROOT / relative
        require(path.is_file(), f"missing sealed file: {relative}")
        output[name] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return output


def verify_file_seals(
    seals: Any, paths: dict[str, str], label: str,
) -> None:
    require(isinstance(seals, dict) and set(seals) == set(paths),
            f"{label} registry changed")
    for name, relative in paths.items():
        seal = seals.get(name)
        path = ROOT / relative
        require(
            isinstance(seal, dict)
            and seal.get("path") == relative
            and seal.get("bytes") == path.stat().st_size
            and seal.get("sha256") == sha256_file(path),
            f"{label} changed: {name}",
        )


def model_output_dir(model_key: str) -> Path:
    require(model_key in MODEL_ORDER, f"unknown model: {model_key}")
    return OUT / model_key


def manifest_path(model_key: str) -> Path:
    return model_output_dir(model_key) / "manifest.json"


def rows_path(model_key: str) -> Path:
    return model_output_dir(model_key) / "rows.jsonl"


def status_path(model_key: str) -> Path:
    return model_output_dir(model_key) / "status.json"


def run_lock_path(model_key: str) -> Path:
    return model_output_dir(model_key) / "runner.lock"


def effective_user_prompt(item: dict[str, Any], arm: str) -> str:
    require(arm in ARMS, f"unknown arm: {arm}")
    problem = str(item.get("problem_prompt", item.get("prompt", ""))).strip()
    require(problem, "item lacks problem prompt")
    return f"{problem}\n\nResponse contract:\n{ARMS[arm]['instruction']}"


def render_prefix(tok: Any, item: dict[str, Any], arm: str) -> tuple[str, str, list[int]]:
    user = effective_user_prompt(item, arm)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    require(isinstance(rendered, str) and rendered, "chat template returned no prefix")
    ids = list(tok(
        rendered, add_special_tokens=False, return_attention_mask=False,
    ).input_ids)
    require(ids and all(isinstance(value, int) for value in ids),
            "tokenizer returned invalid prefix IDs")
    return user, rendered, [int(value) for value in ids]


def stable_pair_seed(
    protocol_sha256: str, model_key: str, seed_key: str, stream: int,
    arm: str | None = None,
) -> int:
    require(len(protocol_sha256) == 64, "invalid protocol hash for seed")
    require(model_key in MODEL_ORDER, "unknown model seed namespace")
    require(isinstance(seed_key, str) and seed_key.strip(), "empty seed key")
    require(isinstance(stream, int) and not isinstance(stream, bool)
            and stream in STREAMS, "unknown stream")
    if arm is not None:
        require(arm in ARMS, "unknown arm for paired seed")
    payload = {
        "dataset_namespace": protocol_sha256,
        "engine_namespace": ENGINE_NAMESPACE,
        "item_id": seed_key,
        "model_key": model_key,
        "stream": int(stream),
    }
    value = int.from_bytes(
        hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()[:8], "big")
    return int(value % (2**31 - 1))


def pair_id(model_key: str, item_id: str, stream: int) -> str:
    require(model_key in MODEL_ORDER
            and isinstance(item_id, str) and item_id
            and isinstance(stream, int) and not isinstance(stream, bool)
            and stream in STREAMS, "invalid pair ID")
    return f"{model_key}|{item_id}|stream_{int(stream)}"


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    item_id = row.get("id")
    arm = row.get("arm")
    stream = row.get("stream")
    require(isinstance(item_id, str) and item_id,
            "row item ID is empty or not a string")
    require(isinstance(arm, str) and arm in ARMS, "row arm is invalid")
    require(isinstance(stream, int) and not isinstance(stream, bool)
            and stream in STREAMS, "row stream is invalid")
    return item_id, arm, int(stream)


def canonical_grid(
    items: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], str, int]]:
    require(len(items) == ITEM_COUNT, "dataset item denominator changed")
    output = [
        (item, arm, stream)
        for stream in STREAMS for arm in ARMS for item in items
    ]
    require(len(output) == EXPECTED_ROWS_PER_MODEL, "row denominator changed")
    require(len(output) % BATCH_SIZE == 0, "grid is not full-batch divisible")
    return output


def chunks(values: list[Any], size: int = BATCH_SIZE) -> Iterable[list[Any]]:
    require(size == BATCH_SIZE and len(values) % size == 0,
            "formal batches must be complete batches of eight")
    for start in range(0, len(values), size):
        yield values[start:start + size]


def trim_at_first_eos(ids: list[int], eos_ids: Iterable[int]) -> list[int]:
    eos = {int(value) for value in eos_ids}
    require(eos, "empty EOS registry")
    for index, value in enumerate(ids):
        if int(value) in eos:
            return [int(item) for item in ids[:index + 1]]
    return [int(item) for item in ids]


def checkpoint_prefix(ids: list[int], checkpoint: int, eos_ids: list[int]) -> list[int]:
    require(checkpoint in CHECKPOINTS, "unknown checkpoint")
    trimmed = trim_at_first_eos(ids, eos_ids)
    return trimmed if len(trimmed) <= checkpoint else trimmed[:checkpoint]


def parse_final_contract(text: str) -> dict[str, Any]:
    stripped = str(text).strip()
    marker_like_count = len(re.findall(
        r"FINAL\s*:", stripped, flags=re.IGNORECASE))
    exact = _FINAL_RE.search(stripped)
    return {
        "plain_text": stripped,
        "marker_like_count": marker_like_count,
        "exact_terminal_marker": exact is not None,
        "parsed_label": exact.group(1) if exact is not None else None,
        "protocol_valid": exact is not None and marker_like_count == 1,
    }


def analyze_ids(
    tok: Any, item: dict[str, Any], ids: list[int], eos_ids: list[int], budget: int,
) -> dict[str, Any]:
    require(isinstance(budget, int) and not isinstance(budget, bool)
            and budget in CHECKPOINTS, "unknown analysis budget")
    require(isinstance(ids, list)
            and all(isinstance(value, int) and not isinstance(value, bool)
                    and value >= 0 for value in ids),
            "trajectory token IDs are invalid")
    require(isinstance(eos_ids, list) and eos_ids
            and all(isinstance(value, int) and not isinstance(value, bool)
                    and value >= 0 for value in eos_ids),
            "EOS token registry is invalid")
    require(0 < len(ids) <= budget, f"invalid trajectory length {len(ids)}/{budget}")
    eos_set = {int(value) for value in eos_ids}
    positions = [index for index, value in enumerate(ids) if int(value) in eos_set]
    require(len(positions) <= 1, "trajectory contains multiple EOS tokens")
    has_eos = bool(positions)
    if has_eos:
        require(positions[0] == len(ids) - 1, "EOS is not absorbing/terminal")
    content_ids = ids[:-1] if has_eos else ids
    raw_special_ids = getattr(tok, "all_special_ids", [])
    require(isinstance(raw_special_ids, (list, tuple, set)),
            "tokenizer all_special_ids registry is invalid")
    special_ids: set[int] = set()
    for value in raw_special_ids:
        require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                "tokenizer special token ID is invalid")
        special_ids.add(int(value))
    unexpected_special_positions = [
        index for index, value in enumerate(content_ids)
        if int(value) in special_ids and int(value) not in eos_set
    ]
    unexpected_special_token_ids = sorted({
        int(content_ids[index]) for index in unexpected_special_positions
    })
    # Never decode with ``skip_special_tokens=True`` for the scientific
    # contract.  Doing so can silently erase a generated assistant/BOS/PAD
    # token and turn an invalid terminal trajectory into an apparent V row.
    parsed = parse_final_contract(tok.decode(
        content_ids, skip_special_tokens=False,
    ))
    if unexpected_special_positions:
        parsed["protocol_valid"] = False
    semantic_match = (
        bool(parsed["protocol_valid"])
        and parsed["parsed_label"] == str(item["answer"])
    )
    if not has_eos:
        terminal = "C"
    elif not bool(parsed["protocol_valid"]):
        terminal = "I_protocol"
    elif semantic_match:
        terminal = "V"
    else:
        terminal = "I_sem"
    require(terminal in TERMINAL_STATES, "unknown terminal state")
    if terminal != "I_protocol":
        protocol_subtype = None
    elif unexpected_special_positions:
        protocol_subtype = "EOS_WITH_UNEXPECTED_SPECIAL_TOKEN"
    else:
        protocol_subtype = "EOS_WITH_INVALID_FINAL_CONTRACT"
    if terminal != "C":
        censor_subtype = None
    elif unexpected_special_positions:
        censor_subtype = "CENSORED_WITH_UNEXPECTED_SPECIAL_TOKEN"
    elif bool(parsed["protocol_valid"]):
        censor_subtype = "CENSORED_AFTER_EXACT_FINAL"
    elif parsed["marker_like_count"]:
        censor_subtype = "CENSORED_WITH_MALFORMED_OR_NONTERMINAL_FINAL"
    else:
        censor_subtype = "CENSORED_BEFORE_FINAL"
    return {
        "budget": int(budget),
        "n_tokens": len(ids),
        "has_eos": has_eos,
        "eos_positions": positions,
        "hit_budget": (not has_eos and len(ids) == budget),
        "unexpected_special_count": len(unexpected_special_positions),
        "unexpected_special_positions": unexpected_special_positions,
        "unexpected_special_token_ids": unexpected_special_token_ids,
        "terminal_state": terminal,
        "protocol_subtype": protocol_subtype,
        "valid_stop": terminal == "V",
        "semantic_match": bool(semantic_match),
        "censor_subtype": censor_subtype,
        **parsed,
    }


def analyze_checkpoints(
    tok: Any, item: dict[str, Any], ids: list[int], eos_ids: list[int],
) -> dict[str, Any]:
    require(0 < len(ids) <= MAX_NEW_TOKENS, "invalid full generated trajectory")
    output: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        prefix = checkpoint_prefix(ids, checkpoint, eos_ids)
        output[str(checkpoint)] = analyze_ids(tok, item, prefix, eos_ids, checkpoint)
    return output


def matrix(
    pairs: Iterable[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    output = {
        left: {right: 0 for right in TERMINAL_STATES}
        for left in TERMINAL_STATES
    }
    for left, right in pairs:
        require(left in output and right in output[left], "matrix state outside registry")
        output[left][right] += 1
    return output


def finite_number(value: Any, label: str) -> float:
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def assert_static_contract() -> None:
    require(tuple(ARMS) == (ARM_A, ARM_B), "arm order changed")
    require(ARMS[ARM_A]["name"] == "external_direct_instruction", "A changed")
    require(ARMS[ARM_B]["name"] == "external_deliberate_instruction", "B changed")
    require("/think" not in canonical_json(ARMS)
            and "enable_thinking" not in canonical_json(ARMS),
            "native Qwen thinking control entered cross-model arms")
    require(MAX_NEW_TOKENS == DECISION_CHECKPOINT == 2048,
            "unique decision horizon changed")
    require(EXPECTED_ROWS_PER_MODEL == 1536 and EXPECTED_ROWS_ALL_MODELS == 4608,
            "formal row denominator changed")
    require(QUANTIZATION["same_policy_all_models"] is True,
            "precision policy differs across models")
    require(parse_final_contract("FINAL: A")["protocol_valid"] is True,
            "exact terminal parser rejected the positive control")
    for invalid in (
        "FINAL:A", "FINAL: C", "FINAL: A\nextra", "FINAL:C\nFINAL: A",
        "FINAL: X\nFINAL: A", "FINAL:\nFINAL: A",
    ):
        require(parse_final_contract(invalid)["protocol_valid"] is False,
                f"terminal parser accepted invalid control: {invalid!r}")


assert_static_contract()
