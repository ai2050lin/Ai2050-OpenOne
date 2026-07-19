#!/usr/bin/env python3
"""Shared, holdout-free core for Phase 979 external boundary diagnostics."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


PHASE = 979
SCHEMA_VERSION = 1
CHECKPOINTS = (256, 512, 1024, 1536, 2048)
MAX_NEW_TOKENS = 2048
BATCH_SIZE = 8
REPLICATES = (0, 1)

CONTROL_POLICIES: dict[str, dict[str, Any]] = {
    "hard_no_think": {
        "enable_thinking": False,
        "prompt_suffix": "",
        "requires_generated_close": False,
        "requires_nonempty_thinking": False,
    },
    "hard_thinking": {
        "enable_thinking": True,
        "prompt_suffix": "",
        "requires_generated_close": True,
        "requires_nonempty_thinking": True,
    },
    "soft_no_think": {
        "enable_thinking": True,
        "prompt_suffix": " /no_think",
        "requires_generated_close": True,
        "requires_nonempty_thinking": False,
    },
    "soft_thinking": {
        "enable_thinking": True,
        "prompt_suffix": " /think",
        "requires_generated_close": True,
        "requires_nonempty_thinking": True,
    },
}

DECODING_POLICIES: dict[str, dict[str, Any]] = {
    "no_think_sampling": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
    },
    "thinking_sampling": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    },
}

OFFICIAL_CELLS = {
    ("hard_no_think", "no_think_sampling"),
    ("hard_thinking", "thinking_sampling"),
    ("soft_no_think", "thinking_sampling"),
    ("soft_thinking", "thinking_sampling"),
}

TERMINAL_STATES = (
    "CENSORED_BEFORE_VALID_CLOSE",
    "CENSORED_AFTER_FINAL_START_NO_ANSWER",
    "CENSORED_AFTER_ANSWER_OBSERVED",
    "EOS_INVALID_MODE",
    "EOS_INVALID_SEMANTIC",
    "VALID_STOP",
)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode("utf-8")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def without_fields(value: dict[str, Any], *fields: str) -> dict[str, Any]:
    blocked = set(fields)
    return {key: item for key, item in value.items() if key not in blocked}


def stable_seed(item_id: str, control: str, decoding: str, replicate: int) -> int:
    payload = f"phase979|{item_id}|{control}|{decoding}|replicate={replicate}"
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    return int(value % (2**31 - 1))


def natural_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row.get("id")), str(row.get("control_policy")),
        str(row.get("decoding_policy")), int(row.get("replicate", -1)),
    )


def truth_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("pair_id")), str(row.get("prompt_side")),
        str(row.get("candidate")), str(row.get("punctuation")),
    )


def effective_user_prompt(item: dict[str, Any], control: str) -> str:
    require(control in CONTROL_POLICIES, f"unknown control: {control}")
    return str(item["prompt"]) + str(CONTROL_POLICIES[control]["prompt_suffix"])


def render_prefix(tok, item: dict[str, Any], control: str) -> tuple[str, str, list[int]]:
    spec = CONTROL_POLICIES[control]
    user_prompt = effective_user_prompt(item, control)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": user_prompt}], tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(spec["enable_thinking"]),
    )
    ids = list(tok(
        rendered, add_special_tokens=False, return_attention_mask=False,
    ).input_ids)
    return user_prompt, rendered, [int(value) for value in ids]


def single_token_id(tok, text: str) -> int:
    ids = list(tok(
        text, add_special_tokens=False, return_attention_mask=False,
    ).input_ids)
    require(len(ids) == 1, f"expected one token for {text!r}, got {ids}")
    return int(ids[0])


def positions_of(ids: list[int], token_ids: set[int]) -> list[int]:
    return [index for index, value in enumerate(ids) if int(value) in token_ids]


def allowed_answer_texts(answer: str) -> tuple[str, str]:
    require(answer in {"A", "B"}, f"unexpected Phase979 primary answer: {answer}")
    return answer, answer + "."


def exact_answer_match(answer: str, text: str) -> bool:
    return text.strip() in allowed_answer_texts(answer)


def mode_analysis(
    tok, ids: list[int], control: str, think_open_id: int, think_close_id: int,
) -> dict[str, Any]:
    spec = CONTROL_POLICIES[control]
    opens = positions_of(ids, {think_open_id})
    closes = positions_of(ids, {think_close_id})
    unique_ordered = len(opens) == 1 and len(closes) == 1 and opens[0] < closes[0]
    thinking_ids = ids[opens[0] + 1:closes[0]] if unique_ordered else []
    thinking_text = tok.decode(thinking_ids, skip_special_tokens=True).strip()
    thinking_nonempty = bool(thinking_text)

    if control == "hard_no_think":
        mode_valid = not opens and not closes
        final_start = 0 if mode_valid else None
        close_applicable = False
        close_observed = False
        close_step = None
        reason = ("no_generated_think_tags" if mode_valid
                  else "generated_think_tag_under_hard_no_think")
        structure = "no_generated_tags" if mode_valid else "unexpected_generated_tags"
    else:
        close_applicable = True
        close_observed = unique_ordered
        close_step = closes[0] + 1 if unique_ordered else None
        final_start = closes[0] + 1 if unique_ordered else None
        expected_nonempty = bool(spec["requires_nonempty_thinking"])
        mode_valid = unique_ordered and thinking_nonempty == expected_nonempty
        if not unique_ordered:
            reason = "missing_repeated_reversed_or_unclosed_generated_tags"
            structure = ("missing_required_tags" if not opens and not closes
                         else "invalid_repeated_reversed_or_unclosed_tags")
        else:
            structure = "unique_ordered_pair"
            reason = ("expected_thinking_content" if mode_valid and expected_nonempty
                      else "expected_empty_thinking_block" if mode_valid
                      else "thinking_emptiness_mismatch")

    final_ids = ids[final_start:] if final_start is not None else []
    return {
        "generated_think_open_positions": opens,
        "generated_think_close_positions": closes,
        "think_structure_status": structure,
        "thinking_nonempty": thinking_nonempty,
        "mode_valid": bool(mode_valid),
        "mode_valid_reason": reason,
        "final_start_position": final_start,
        "final_region_valid": final_start is not None,
        "final_text": tok.decode(final_ids, skip_special_tokens=True).strip(),
        "close_applicable": close_applicable,
        "close_observed": close_observed,
        "t_close_step": close_step,
    }


def first_answer_step(
    tok, ids: list[int], final_start: int | None, answer: str,
    eos_ids: set[int],
) -> int | None:
    if final_start is None:
        return None
    stop = next((index for index in range(final_start, len(ids))
                 if int(ids[index]) in eos_ids), len(ids))
    for end in range(final_start + 1, stop + 1):
        text = tok.decode(ids[final_start:end], skip_special_tokens=True).strip()
        if exact_answer_match(answer, text):
            return end
        if text and not any(value.startswith(text) for value in allowed_answer_texts(answer)):
            return None
    return None


def terminal_state(
    control: str, has_eos: bool, mode: dict[str, Any],
    semantic_match: bool, t_answer_step: int | None,
) -> str:
    if has_eos:
        if not bool(mode["mode_valid"]):
            return "EOS_INVALID_MODE"
        if not semantic_match:
            return "EOS_INVALID_SEMANTIC"
        return "VALID_STOP"
    if (control != "hard_no_think" and not bool(mode["close_observed"])):
        return "CENSORED_BEFORE_VALID_CLOSE"
    if t_answer_step is None:
        return "CENSORED_AFTER_FINAL_START_NO_ANSWER"
    return "CENSORED_AFTER_ANSWER_OBSERVED"


def analyze_ids(
    tok, item: dict[str, Any], control: str, ids: list[int], eos_ids: list[int],
    think_open_id: int, think_close_id: int, budget: int,
) -> dict[str, Any]:
    require(0 < len(ids) <= budget, f"invalid trajectory length {len(ids)}/{budget}")
    eos_set = {int(value) for value in eos_ids}
    eos_positions = positions_of(ids, eos_set)
    require(len(eos_positions) <= 1, "trajectory has multiple EOS before trimming")
    first_eos_position = eos_positions[0] if eos_positions else None
    has_eos = first_eos_position is not None
    if has_eos:
        require(first_eos_position == len(ids) - 1, "EOS is not terminal")
    mode = mode_analysis(tok, ids, control, think_open_id, think_close_id)
    t_answer = first_answer_step(
        tok, ids, mode["final_start_position"], str(item["answer"]), eos_set,
    )
    semantic = exact_answer_match(str(item["answer"]), str(mode["final_text"]))
    state = terminal_state(control, has_eos, mode, semantic, t_answer)
    require(state in TERMINAL_STATES, f"unknown terminal state {state}")
    t_eos = first_eos_position + 1 if first_eos_position is not None else None
    if state == "VALID_STOP":
        require(t_answer is not None and t_eos is not None and t_answer < t_eos,
                "valid stop lacks ordered answer/EOS")
        if bool(mode["close_applicable"]):
            require(mode["t_close_step"] is not None
                    and int(mode["t_close_step"]) < t_answer,
                    "valid thinking stop lacks T_C<T_A")
    return {
        "budget": int(budget),
        "n_tokens": len(ids),
        "has_eos": has_eos,
        "eos_positions": eos_positions,
        "first_eos_position": first_eos_position,
        "t_eos_step": t_eos,
        "hit_budget": (not has_eos and len(ids) == budget),
        **mode,
        "t_answer_step": t_answer,
        "answer_observed": t_answer is not None,
        "semantic_match_at_snapshot": semantic,
        "valid_stop": state == "VALID_STOP",
        "terminal_state": state,
        "event_vector": {
            "close_applicable": bool(mode["close_applicable"]),
            "close_observed": bool(mode["close_observed"]),
            "answer_observed": t_answer is not None,
            "eos_observed": has_eos,
            "mode_valid_at_snapshot": bool(mode["mode_valid"]),
            "semantic_match_at_snapshot": semantic,
        },
    }


def trim_at_first_eos(ids: list[int], eos_ids: list[int]) -> list[int]:
    eos_set = {int(value) for value in eos_ids}
    for index, value in enumerate(ids):
        if int(value) in eos_set:
            return [int(item) for item in ids[:index + 1]]
    return [int(item) for item in ids]


def checkpoint_prefix(ids: list[int], checkpoint: int, eos_ids: list[int]) -> list[int]:
    trimmed = trim_at_first_eos(ids, eos_ids)
    if len(trimmed) <= checkpoint:
        return trimmed
    return trimmed[:checkpoint]


def analyze_checkpoints(
    tok, item: dict[str, Any], control: str, ids: list[int], eos_ids: list[int],
    think_open_id: int, think_close_id: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        prefix = checkpoint_prefix(ids, checkpoint, eos_ids)
        output[str(checkpoint)] = analyze_ids(
            tok, item, control, prefix, eos_ids,
            think_open_id, think_close_id, checkpoint,
        )
    return output


def finite_number(value: Any, label: str) -> float:
    number = float(value)
    require(math.isfinite(number), f"{label} is not finite")
    return number
