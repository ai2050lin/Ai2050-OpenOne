#!/usr/bin/env python3
"""Self-contained frozen primitives for the Phase978 legal-trajectory protocol.

This module intentionally has no project-local imports.  It copies the
reviewed Phase977 strict-v2 parser, matcher, official condition registry,
same-seed sampler, and durable JSON helpers needed by Phase978.  Keeping this
small dependency surface prevents unrelated historical experiment modules
from executing transitively during a confirmatory run.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = 2
GENERATED_MODE_PARSER_VERSION = "strict_final_region_v2"
DEFAULT_BASE_SEED = 977_000

CONDITIONS: dict[str, dict[str, Any]] = {
    "hard_no_think": {
        "enable_thinking": False,
        "prompt_suffix": "",
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
    },
    "hard_thinking": {
        "enable_thinking": True,
        "prompt_suffix": "",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    },
    "soft_no_think": {
        "enable_thinking": True,
        "prompt_suffix": " /no_think",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    },
    "soft_thinking": {
        "enable_thinking": True,
        "prompt_suffix": " /think",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2,
                         allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open("rb+") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    payload = (canonical_json(row) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["id"]), str(row["condition"]), str(row["stage"])


def normalize_item(raw: dict[str, Any]) -> dict[str, Any]:
    alias_groups = raw.get("alias_groups")
    if alias_groups is None:
        aliases = raw.get("aliases") or [raw["answer"]]
        alias_groups = [aliases]
    if not isinstance(alias_groups, list) or not alias_groups:
        raise ValueError(f"{raw.get('id')}: alias_groups must be nonempty")
    normalized_groups: list[list[str]] = []
    for group in alias_groups:
        if not isinstance(group, list) or not group:
            raise ValueError(f"{raw.get('id')}: invalid alias group")
        values = [str(value).strip() for value in group if str(value).strip()]
        if not values:
            raise ValueError(f"{raw.get('id')}: empty normalized alias group")
        normalized_groups.append(values)
    exact = bool(raw.get("exact", False))
    if exact and len(normalized_groups) != 1:
        raise ValueError(f"{raw.get('id')}: exact item needs one alias group")
    item = {
        "id": str(raw["id"]),
        "task": str(raw["task"]),
        "prompt": str(raw["prompt"]),
        "answer": str(raw["answer"]),
        "alias_groups": normalized_groups,
        "exact": exact,
    }
    if "prompt_template" in raw:
        item["prompt_template"] = str(raw["prompt_template"])
    return item


def audit_local_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [item["id"] for item in items]
    duplicate_ids = sorted(item_id for item_id in set(ids)
                           if ids.count(item_id) > 1)
    switch_collisions = [item["id"] for item in items if re.search(
        r"/(?:no_)?think\b", item["prompt"], flags=re.IGNORECASE)]
    empty_prompts = [item["id"] for item in items if not item["prompt"].strip()]
    if duplicate_ids or switch_collisions or empty_prompts:
        raise ValueError({
            "duplicate_ids": duplicate_ids,
            "preexisting_soft_switches": switch_collisions,
            "empty_prompts": empty_prompts,
        })
    return {
        "n": len(items),
        "n_tasks": len({item["task"] for item in items}),
        "duplicate_ids": duplicate_ids,
        "preexisting_soft_switches": switch_collisions,
        "empty_prompts": empty_prompts,
        "exact_n": sum(bool(item["exact"]) for item in items),
    }


def dataset_hash(items: list[dict[str, Any]]) -> str:
    stable = [{
        "id": item["id"], "task": item["task"], "prompt": item["prompt"],
        "answer": item["answer"], "alias_groups": item["alias_groups"],
        "exact": item["exact"],
    } for item in items]
    return sha256_json(stable)


def _substring_normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", value).strip()


LEGACY_DISCOVERY_STEMS = {"condens", "refract", "dissolv", "reflect", "magnet"}


def _substring_alias_matches(alias: str, value: str) -> bool:
    alias_value = _substring_normalize(alias)
    if not alias_value:
        return False
    escaped = re.escape(alias_value)
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\d+:\d+)", alias_value):
        pattern = r"(?<![\w.+-])" + escaped + r"(?!\w|\.\d)"
        return re.search(pattern, value) is not None
    if alias_value in LEGACY_DISCOVERY_STEMS:
        return re.search(r"(?<!\w)" + escaped + r"\w*", value) is not None
    return re.search(r"(?<!\w)" + escaped + r"(?!\w)", value) is not None


def _exact_candidates(text: str) -> set[str]:
    value = text.strip().casefold()
    values = {value}
    if value and value[-1] in ".?!":
        values.add(value[:-1].rstrip())
    return values


def semantic_match(alias_groups: list[list[str]], text: str, exact: bool) -> bool:
    if exact:
        if len(alias_groups) != 1:
            raise ValueError("exact matching requires one alias group")
        candidates = _exact_candidates(text)
        return any(alias.strip().casefold() in candidates
                   for alias in alias_groups[0])
    value = _substring_normalize(text)
    return all(any(_substring_alias_matches(alias, value) for alias in group)
               for group in alias_groups)


def stable_item_seed(base_seed: int, split: str, item_id: str) -> int:
    raw = f"phase977|{base_seed}|{split}|{item_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31 - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def effective_user_prompt(item: dict[str, Any], condition: str) -> str:
    return item["prompt"].rstrip() + str(CONDITIONS[condition]["prompt_suffix"])


def render_prefix(tok, item: dict[str, Any], condition: str) -> tuple[str, str, list[int]]:
    spec = CONDITIONS[condition]
    user_prompt = effective_user_prompt(item, condition)
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(spec["enable_thinking"]),
    )
    ids = list(tok(rendered, add_special_tokens=False,
                   return_attention_mask=False).input_ids)
    return user_prompt, rendered, [int(value) for value in ids]


def single_token_id(tok, text: str) -> int:
    ids = list(tok(text, add_special_tokens=False,
                   return_attention_mask=False).input_ids)
    if len(ids) != 1:
        raise RuntimeError(f"expected one token for {text!r}, got {ids}")
    return int(ids[0])


def positions_of(ids: list[int], token_ids: set[int]) -> list[int]:
    return [index for index, value in enumerate(ids) if int(value) in token_ids]


def build_template_tokens(tok, probe: dict[str, Any], think_open_id: int,
                          think_close_id: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in CONDITIONS:
        prompt, rendered, ids = render_prefix(tok, probe, condition)
        output[condition] = {
            "effective_user_prompt": prompt,
            "rendered_prefix": rendered,
            "input_ids": ids,
            "tokens": tok.convert_ids_to_tokens(ids),
            "prompt_len": len(ids),
            "prefilled_think_open_positions": positions_of(ids, {think_open_id}),
            "prefilled_think_close_positions": positions_of(ids, {think_close_id}),
        }
    hard = output["hard_no_think"]
    if not hard["prefilled_think_open_positions"] or not hard["prefilled_think_close_positions"]:
        raise RuntimeError("hard_no_think did not prefill an empty think block")
    for condition in ("hard_thinking", "soft_no_think", "soft_thinking"):
        if (output[condition]["prefilled_think_open_positions"]
                or output[condition]["prefilled_think_close_positions"]):
            raise RuntimeError(f"{condition} unexpectedly prefilled think tags")
    return output


def generated_mode_analysis(tok, ids: list[int], condition: str,
                            think_open_id: int, think_close_id: int) -> dict[str, Any]:
    opens = positions_of(ids, {think_open_id})
    closes = positions_of(ids, {think_close_id})
    well_formed = len(opens) == 1 and len(closes) == 1 and opens[0] < closes[0]
    thinking_ids = ids[opens[0] + 1:closes[0]] if well_formed else []
    thinking_text = tok.decode(thinking_ids, skip_special_tokens=True).strip()
    thinking_nonempty = bool(thinking_text)
    if condition == "hard_no_think":
        mode_valid = not opens and not closes
        reason = ("no_generated_think_tags" if mode_valid
                  else "generated_think_tag_under_hard_switch")
    elif condition == "soft_no_think":
        mode_valid = well_formed and not thinking_nonempty
        reason = ("well_formed_empty_generated_block" if mode_valid
                  else "expected_empty_generated_block")
    else:
        mode_valid = well_formed and thinking_nonempty
        reason = ("well_formed_nonempty_generated_block" if mode_valid
                  else "expected_nonempty_generated_block")
    if well_formed:
        final_start = closes[0] + 1
        final_ids = ids[final_start:]
        final_source = "after_unique_ordered_generated_think_close"
        final_region_valid = True
        structure = "unique_ordered_pair"
    elif condition == "hard_no_think" and not opens and not closes:
        final_start = 0
        final_ids = ids
        final_source = "full_generated_output_hard_no_think_no_tags"
        final_region_valid = True
        structure = "no_generated_tags"
    else:
        final_start = None
        final_ids = []
        final_source = "empty_invalid_missing_or_unclosed_think_structure"
        final_region_valid = False
        structure = ("missing_required_tags" if not opens and not closes
                     else "invalid_repeated_reversed_or_unclosed_tags")
    return {
        "generated_mode_parser_version": GENERATED_MODE_PARSER_VERSION,
        "generated_think_open_positions": opens,
        "generated_think_close_positions": closes,
        "think_well_formed": well_formed,
        "think_structure_status": structure,
        "thinking_text": thinking_text,
        "thinking_nonempty": thinking_nonempty,
        "mode_valid": mode_valid,
        "mode_valid_reason": reason,
        "final_region_valid": final_region_valid,
        "final_start_position": final_start,
        "final_source": final_source,
        "final_text": tok.decode(final_ids, skip_special_tokens=True).strip(),
    }


def analyze_generation(tok, item: dict[str, Any], condition: str, ids: list[int],
                       eos_ids: list[int], think_open_id: int,
                       think_close_id: int, max_new_tokens: int) -> dict[str, Any]:
    eos_positions = positions_of(ids, {int(value) for value in eos_ids})
    first = eos_positions[0] if eos_positions else None
    mode = generated_mode_analysis(tok, ids, condition, think_open_id, think_close_id)
    matched = semantic_match(item["alias_groups"], mode["final_text"], item["exact"])
    has_eos = bool(eos_positions)
    return {
        "generated_ids": [int(value) for value in ids],
        "raw": tok.decode(ids, skip_special_tokens=False),
        "plain": tok.decode(ids, skip_special_tokens=True),
        **mode,
        "semantic_match": matched,
        "eos_positions": eos_positions,
        "first_eos_position": first,
        "first_eos_step": None if first is None else first + 1,
        "first_eos_id": None if first is None else int(ids[first]),
        "has_eos": has_eos,
        "valid_eos": bool(has_eos and matched),
        "valid_mode_eos": bool(has_eos and matched and mode["mode_valid"]),
        "n_tokens": len(ids),
        "hit_budget": bool(len(ids) >= max_new_tokens and not has_eos),
    }


def generate_stage(model, tok, device, eos_ids: list[int], item: dict[str, Any],
                   condition: str, seed: int,
                   max_new_tokens: int) -> tuple[list[int], list[int], str]:
    spec = CONDITIONS[condition]
    user_prompt, rendered, input_ids = render_prefix(tok, item, condition)
    encoded = tok(rendered, add_special_tokens=False, return_tensors="pt",
                  return_attention_mask=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    if encoded["input_ids"][0].tolist() != input_ids:
        raise RuntimeError(f"{item['id']}/{condition}: render/encode mismatch")
    seed_everything(seed)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=float(spec["temperature"]),
            top_p=float(spec["top_p"]),
            top_k=int(spec["top_k"]),
            min_p=float(spec["min_p"]),
            use_cache=True,
            pad_token_id=tok.pad_token_id,
            eos_token_id=eos_ids,
            return_dict_in_generate=True,
        )
    generated = output.sequences[0, len(input_ids):].tolist()
    return input_ids, [int(value) for value in generated], user_prompt


def get_eos_ids(model, tokenizer) -> list[int]:
    ids: list[int] = []
    values = (
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
    )
    for value in values:
        if value is None:
            continue
        candidates = value if isinstance(value, (list, tuple, set)) else [value]
        for candidate in candidates:
            if candidate is not None and int(candidate) not in ids:
                ids.append(int(candidate))
    if not ids:
        raise RuntimeError("no EOS token IDs found")
    return ids
