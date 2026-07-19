#!/usr/bin/env python3
"""Phase 977: legal Qwen3 mode trajectories with long, sampled rollouts.

This experiment compares exactly four *official* Qwen3 mode controls:

* ``hard_no_think``: ``enable_thinking=False``;
* ``hard_thinking``: ``enable_thinking=True``;
* ``soft_no_think``: ``enable_thinking=True`` plus ``/no_think``;
* ``soft_thinking``: ``enable_thinking=True`` plus ``/think``.

There are no token ablations, residual patches, EOS biases, greedy decoding, or
condition-selection gates in this script.  Every item/condition first receives
a reproducible 256-token sampled rollout.  A trajectory that has not emitted an
EOS token is rerun from the original prompt with the same seed and a 512-token
budget.  Both stages are retained as separate recoverable JSONL records.

The hard-no-think template contains a prefilled empty ``<think>...</think>``
block.  Think-tag positions are measured only in generated token IDs, never in
the prompt IDs, so that prefilled block cannot be mistaken for generated
thinking.  Schema v2 also makes the scored final region strict: a generated
tag structure is scorable only when it is one unique ordered pair, except that
hard-no-think with no generated tags scores its whole generated output.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, release_model
from phase951_protocol_atlas import ensure_dir
from phase966_natural_stop import log
from phase973_conditional_trajectory import get_eos_ids


PHASE = 977
MODEL_NAME = "qwen3"
OUT = Path("tests/glm5/result/phase977_legal_mode_trajectories")
INITIAL_BUDGET = 256
EXTENDED_BUDGET = 512
SCHEMA_VERSION = 2
GENERATED_MODE_PARSER_VERSION = "strict_final_region_v2"
DEFAULT_BASE_SEED = 977_000


# Keep this as the sole condition registry.  The three enable_thinking=True
# conditions use the local Qwen3 README's thinking-mode sampling parameters;
# hard_no_think uses its separately recommended non-thinking parameters.
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


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    os.replace(temporary, path)


def normalize_item(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize Phase976 aliases and Phase977 grouped aliases to one schema."""
    alias_groups = raw.get("alias_groups")
    if alias_groups is None:
        aliases = raw.get("aliases") or [raw["answer"]]
        # Phase976 used a single OR-list; retain exactly that semantics.
        alias_groups = [aliases]

    if not isinstance(alias_groups, list) or not alias_groups:
        raise ValueError(f"{raw.get('id')}: alias_groups must be a nonempty list")
    normalized_groups: list[list[str]] = []
    for group in alias_groups:
        if not isinstance(group, list) or not group:
            raise ValueError(f"{raw.get('id')}: every alias group must be nonempty")
        values = [str(x).strip() for x in group if str(x).strip()]
        if not values:
            raise ValueError(f"{raw.get('id')}: empty alias group after normalization")
        normalized_groups.append(values)

    exact = bool(raw.get("exact", False))
    if exact and len(normalized_groups) != 1:
        raise ValueError(f"{raw.get('id')}: exact=True requires exactly one alias group")

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


def _discovery_items() -> list[dict[str, Any]]:
    module = importlib.import_module("phase976_qwen_mode_external")
    return [normalize_item(x) for x in module.build_external_dataset()]


def _development_items() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Deliberately lazy: this module is not imported for a discovery run.
    module = importlib.import_module("phase977_dev_dataset")
    previous = [x["prompt"] for x in _discovery_items()]
    audit = module.audit_dataset(previous_prompts=previous)
    return [normalize_item(x) for x in module.build_dataset()], audit


def _holdout_items() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Deliberately lazy and reachable only through explicit --split holdout.
    module = importlib.import_module("phase977_holdout_dataset")
    dev_module = importlib.import_module("phase977_dev_dataset")
    previous = [x["prompt"] for x in _discovery_items()]
    previous.extend(str(x["prompt"]) for x in dev_module.build_dataset())
    audit = module.audit_dataset(previous_prompts=previous)
    return [normalize_item(x) for x in module.build_dataset()], audit


def audit_local_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [x["id"] for x in items]
    duplicate_ids = sorted({x for x in ids if ids.count(x) > 1})
    switch_collisions = [x["id"] for x in items if re.search(
        r"/(?:no_)?think\b", x["prompt"], flags=re.IGNORECASE)]
    empty_prompts = [x["id"] for x in items if not x["prompt"].strip()]
    if duplicate_ids or switch_collisions or empty_prompts:
        raise ValueError({
            "duplicate_ids": duplicate_ids,
            "preexisting_soft_switches": switch_collisions,
            "empty_prompts": empty_prompts,
        })
    return {
        "n": len(items),
        "n_tasks": len({x["task"] for x in items}),
        "duplicate_ids": duplicate_ids,
        "preexisting_soft_switches": switch_collisions,
        "empty_prompts": empty_prompts,
        "exact_n": sum(x["exact"] for x in items),
    }


def load_split(split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split == "discovery":
        items = _discovery_items()
        external_audit: dict[str, Any] = {
            "source": "phase976_qwen_mode_external.build_external_dataset",
            "note": "Previously used texts; Phase977 mode trajectories are new interventions.",
        }
    elif split == "development":
        items, external_audit = _development_items()
    elif split == "holdout":
        items, external_audit = _holdout_items()
    else:
        raise ValueError(split)

    for audit_flag in ("ok", "passed"):
        if audit_flag in external_audit and not bool(external_audit[audit_flag]):
            raise ValueError(f"{split}: external dataset audit failed: {external_audit}")
    if external_audit.get("errors"):
        raise ValueError(f"{split}: external dataset audit reported errors: {external_audit}")
    local_audit = audit_local_items(items)
    return items, {"external": external_audit, "local": local_audit}


def dataset_hash(items: list[dict[str, Any]]) -> str:
    stable = [{
        "id": x["id"], "task": x["task"], "prompt": x["prompt"],
        "answer": x["answer"], "alias_groups": x["alias_groups"],
        "exact": x["exact"],
    } for x in items]
    return sha256_json(stable)


def _substring_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


LEGACY_DISCOVERY_STEMS = {"condens", "refract", "dissolv", "reflect", "magnet"}


def _substring_alias_matches(alias: str, value: str) -> bool:
    alias_value = _substring_normalize(alias)
    if not alias_value:
        return False
    escaped = re.escape(alias_value)

    # Numeric answers need stronger left boundaries: ordinary word boundaries
    # would incorrectly count ``1`` inside ``-1``.  Also reject decimal tails.
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\d+:\d+)", alias_value):
        # A sentence-final period is allowed, but a period followed by a digit
        # means the alias is only a prefix of a decimal (1 versus 1.5).
        pattern = r"(?<![\w.+-])" + escaped + r"(?!\w|\.\d)"
        return re.search(pattern, value) is not None

    # Phase976 intentionally used six truncated causal aliases.  Preserve those
    # exact legacy stems, but still require a word start so e.g. ``reflect`` is
    # not found in the middle of an unrelated token.
    if alias_value in LEGACY_DISCOVERY_STEMS:
        return re.search(r"(?<!\w)" + escaped + r"\w*", value) is not None

    # New Phase977 corpora require whole words or whole phrases.  This prevents
    # metal/nonmetal and vertebrate/invertebrate false positives.
    return re.search(r"(?<!\w)" + escaped + r"(?!\w)", value) is not None


def _exact_candidates(text: str) -> set[str]:
    value = text.strip().casefold()
    values = {value}
    # The dataset contract permits one trailing sentence punctuation mark.
    if value and value[-1] in ".?!":
        values.add(value[:-1].rstrip())
    return values


def semantic_match(alias_groups: list[list[str]], text: str, exact: bool) -> bool:
    """Group-wise AND, within-group OR; exact items use whole-string equality."""
    if exact:
        if len(alias_groups) != 1:
            raise ValueError("exact semantic matching requires exactly one group")
        candidates = _exact_candidates(text)
        return any(alias.strip().casefold() in candidates for alias in alias_groups[0])

    value = _substring_normalize(text)
    return all(any(_substring_alias_matches(alias, value) for alias in group)
               for group in alias_groups)


def stable_item_seed(base_seed: int, split: str, item_id: str) -> int:
    # The same item seed is intentionally reused across all four conditions.
    raw = f"phase977|{base_seed}|{split}|{item_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31 - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def effective_user_prompt(item: dict[str, Any], condition: str) -> str:
    suffix = str(CONDITIONS[condition]["prompt_suffix"])
    return item["prompt"].rstrip() + suffix


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
    return user_prompt, rendered, ids


def single_token_id(tok, text: str) -> int:
    ids = list(tok(text, add_special_tokens=False,
                   return_attention_mask=False).input_ids)
    if len(ids) != 1:
        raise RuntimeError(f"expected one token for {text!r}, got {ids}")
    return int(ids[0])


def positions_of(ids: list[int], token_ids: set[int]) -> list[int]:
    return [i for i, value in enumerate(ids) if int(value) in token_ids]


def build_template_tokens(tok, probe: dict[str, Any], think_open_id: int,
                          think_close_id: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition in CONDITIONS:
        prompt, rendered, ids = render_prefix(tok, probe, condition)
        out[condition] = {
            "effective_user_prompt": prompt,
            "rendered_prefix": rendered,
            "input_ids": ids,
            "tokens": tok.convert_ids_to_tokens(ids),
            "prompt_len": len(ids),
            "prefilled_think_open_positions": positions_of(ids, {think_open_id}),
            "prefilled_think_close_positions": positions_of(ids, {think_close_id}),
        }

    # The official hard switch must contain the empty block; the three
    # enable_thinking=True prefixes must not prefill special think tags.
    hard = out["hard_no_think"]
    if not hard["prefilled_think_open_positions"] or not hard["prefilled_think_close_positions"]:
        raise RuntimeError("hard_no_think template did not prefill an empty think block")
    for condition in ("hard_thinking", "soft_no_think", "soft_thinking"):
        if (out[condition]["prefilled_think_open_positions"]
                or out[condition]["prefilled_think_close_positions"]):
            raise RuntimeError(f"{condition} unexpectedly prefilled think tags")
    return out


def make_manifest(split: str, items: list[dict[str, Any]], data_audit: dict[str, Any],
                  base_seed: int, model, tok, eos_ids: list[int],
                  think_open_id: int, think_close_id: int) -> dict[str, Any]:
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    core = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": "qwen3_official_legal_mode_trajectories",
        "model": MODEL_NAME,
        "model_class": type(model).__name__,
        "model_name_or_path": str(getattr(model.config, "_name_or_path", "")),
        "tokenizer_class": type(tok).__name__,
        "split": split,
        "n_items": len(items),
        "item_ids": [x["id"] for x in items],
        "dataset_sha256": dataset_hash(items),
        "dataset_audit": data_audit,
        "script_sha256": script_hash,
        "conditions": CONDITIONS,
        "main_condition_order": list(CONDITIONS),
        "budgets": {"initial": INITIAL_BUDGET, "extended": EXTENDED_BUDGET},
        "extension_rule": "rerun from original prompt with the same item seed iff initial256 has no EOS",
        "extension_strategy": "full_rerun_same_seed; retain initial256 and extended512 JSONL stages",
        "sampling_source": "local Qwen3-4B README: thinking 0.6/0.95/20; hard no-think 0.7/0.8/20; min_p=0",
        "decoding": "sampling only; do_sample=True; no greedy main result",
        "base_seed": int(base_seed),
        "seed_rule": "sha256(phase977|base_seed|split|item_id), identical across the four conditions",
        "eos_token_ids": [int(x) for x in eos_ids],
        "special_token_ids": {
            "think_open": int(think_open_id),
            "think_close": int(think_close_id),
        },
        "template_tokens": build_template_tokens(
            tok, items[0], think_open_id, think_close_id),
        "generated_tag_rule": "think positions are searched only in generated_ids, never prompt input_ids",
        "generated_mode_parser_version": GENERATED_MODE_PARSER_VERSION,
        "generated_final_region_rule": (
            "score text after the close token only for exactly one ordered "
            "generated <think>/</think> pair; score the full generated text "
            "only for hard_no_think with no generated think tags; every other "
            "missing, unclosed, repeated, reversed, or malformed tag structure "
            "has an empty final_text"
        ),
        "schema_v2_change": (
            "schema v1 could score reasoning as final text when an open think "
            "tag had no unique valid close; schema v2 makes that region unscorable"
        ),
        "torch_version": torch.__version__,
    }
    digest = sha256_json(core)
    return {**core, "manifest_sha256": digest, "created_at_utc": utc_now()}


def install_or_validate_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError(
                f"manifest mismatch at {path}; refusing to mix runs\n"
                f"existing={prior.get('manifest_sha256')}\n"
                f"current={manifest['manifest_sha256']}"
            )
        return
    atomic_write_json(path, manifest)


def _ensure_append_boundary(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        last = handle.read(1)
    if last != b"\n":
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    _ensure_append_boundary(path)
    payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":"))
               + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["id"]), str(row["condition"]), str(row["stage"])


def load_jsonl(path: Path, manifest_sha256: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    # A process can be killed between opening a JSONL append and finishing its
    # final line.  Retain every complete fsynced record and truncate only that
    # malformed tail, so a resumed append cannot turn it into a bad middle line.
    lines = path.read_bytes().splitlines(keepends=True)
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicates = 0
    byte_offset = 0
    for index, raw_line in enumerate(lines):
        line_start = byte_offset
        byte_offset += len(raw_line)
        if not raw_line.strip():
            continue
        try:
            line = raw_line.decode("utf-8")
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            if index == len(lines) - 1:
                with path.open("r+b") as handle:
                    handle.truncate(line_start)
                    handle.flush()
                    os.fsync(handle.fileno())
                log(f"Removed a truncated final JSONL record in {path}")
                break
            raise
        if row.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError(f"row/manifest mismatch in {path} line {index+1}")
        key = row_key(row)
        if key in out:
            duplicates += 1
        out[key] = row
    if duplicates:
        log(f"Warning: {path} contained {duplicates} duplicate stage keys; latest rows used")
    return out


def generated_mode_analysis(tok, ids: list[int], condition: str,
                            think_open_id: int, think_close_id: int) -> dict[str, Any]:
    opens = positions_of(ids, {think_open_id})
    closes = positions_of(ids, {think_close_id})
    well_formed = (len(opens) == 1 and len(closes) == 1 and opens[0] < closes[0])
    thinking_ids: list[int] = []
    if well_formed:
        thinking_ids = ids[opens[0] + 1:closes[0]]
    thinking_text = tok.decode(thinking_ids, skip_special_tokens=True).strip()
    thinking_nonempty = bool(thinking_text)

    if condition == "hard_no_think":
        mode_valid = not opens and not closes
        reason = "no_generated_think_tags" if mode_valid else "generated_think_tag_under_hard_switch"
    elif condition == "soft_no_think":
        mode_valid = well_formed and not thinking_nonempty
        reason = "well_formed_empty_generated_block" if mode_valid else "expected_empty_generated_block"
    else:
        mode_valid = well_formed and thinking_nonempty
        reason = "well_formed_nonempty_generated_block" if mode_valid else "expected_nonempty_generated_block"

    if well_formed:
        final_start = closes[0] + 1
        final_ids = ids[final_start:]
        final_source = "after_unique_ordered_generated_think_close"
        final_region_valid = True
        think_structure_status = "unique_ordered_pair"
    elif condition == "hard_no_think" and not opens and not closes:
        final_start = 0
        final_ids = ids
        final_source = "full_generated_output_hard_no_think_no_tags"
        final_region_valid = True
        think_structure_status = "no_generated_tags"
    else:
        final_start = None
        final_ids = []
        final_source = "empty_invalid_missing_or_unclosed_think_structure"
        final_region_valid = False
        think_structure_status = (
            "missing_required_tags" if not opens and not closes
            else "invalid_repeated_reversed_or_unclosed_tags"
        )

    return {
        "generated_mode_parser_version": GENERATED_MODE_PARSER_VERSION,
        "generated_think_open_positions": opens,
        "generated_think_close_positions": closes,
        "think_well_formed": well_formed,
        "think_structure_status": think_structure_status,
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
    eos_set = {int(x) for x in eos_ids}
    eos_positions = positions_of(ids, eos_set)
    first_eos_position = eos_positions[0] if eos_positions else None
    first_eos_id = int(ids[first_eos_position]) if first_eos_position is not None else None
    mode = generated_mode_analysis(tok, ids, condition, think_open_id, think_close_id)
    matched = semantic_match(item["alias_groups"], mode["final_text"], item["exact"])
    has_eos = bool(eos_positions)
    raw = tok.decode(ids, skip_special_tokens=False)
    plain = tok.decode(ids, skip_special_tokens=True)
    return {
        "generated_ids": [int(x) for x in ids],
        "raw": raw,
        "plain": plain,
        **mode,
        "semantic_match": matched,
        "eos_positions": eos_positions,
        "first_eos_position": first_eos_position,
        "first_eos_step": None if first_eos_position is None else first_eos_position + 1,
        "first_eos_id": first_eos_id,
        "has_eos": has_eos,
        "valid_eos": bool(has_eos and matched),
        "valid_mode_eos": bool(has_eos and matched and mode["mode_valid"]),
        "n_tokens": len(ids),
        "hit_budget": bool(len(ids) >= max_new_tokens and not has_eos),
    }


def generate_stage(model, tok, device, eos_ids: list[int], item: dict[str, Any],
                   condition: str, seed: int, max_new_tokens: int) -> tuple[list[int], list[int], str]:
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
    return input_ids, [int(x) for x in generated], user_prompt


def build_stage_row(manifest: dict[str, Any], tok, item: dict[str, Any],
                    condition: str, seed: int, stage: str, input_ids: list[int],
                    generated_ids: list[int], user_prompt: str,
                    eos_ids: list[int], think_open_id: int, think_close_id: int,
                    max_new_tokens: int, initial_row: dict[str, Any] | None) -> dict[str, Any]:
    analysis = analyze_generation(tok, item, condition, generated_ids, eos_ids,
                                  think_open_id, think_close_id, max_new_tokens)
    prefilled_open = positions_of(input_ids, {think_open_id})
    prefilled_close = positions_of(input_ids, {think_close_id})
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "manifest_sha256": manifest["manifest_sha256"],
        "split": manifest["split"],
        "id": item["id"],
        "task": item["task"],
        "condition": condition,
        "stage": stage,
        "seed": int(seed),
        "prompt": item["prompt"],
        "effective_user_prompt": user_prompt,
        "answer": item["answer"],
        "alias_groups": item["alias_groups"],
        "exact": item["exact"],
        "enable_thinking": bool(CONDITIONS[condition]["enable_thinking"]),
        "sampling": {key: CONDITIONS[condition][key] for key in
                     ("temperature", "top_p", "top_k", "min_p")},
        "max_new_tokens": int(max_new_tokens),
        "input_ids": [int(x) for x in input_ids],
        "prompt_len": len(input_ids),
        "prefilled_think_open_positions": prefilled_open,
        "prefilled_think_close_positions": prefilled_close,
        **analysis,
        "hit256": None,
        "hit512": None,
        "extension_strategy": None,
        "extension_replayed_initial256_exact": None,
        "recorded_at_utc": utc_now(),
    }
    if stage == "initial256":
        row["hit256"] = bool(analysis["hit_budget"])
    elif stage == "extended512":
        if initial_row is None:
            raise ValueError("extended512 requires its initial256 row")
        row["hit256"] = bool(initial_row["hit256"])
        row["hit512"] = bool(analysis["hit_budget"])
        row["extension_strategy"] = "rerun_from_original_prompt_same_seed"
        initial_ids = [int(x) for x in initial_row["generated_ids"]]
        row["extension_replayed_initial256_exact"] = (
            generated_ids[:len(initial_ids)] == initial_ids)
        row["initial256_n_tokens"] = len(initial_ids)
    else:
        raise ValueError(stage)
    return row


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(np.mean([bool(x.get(key, False)) for x in rows]))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    extended = [x for x in rows if x["stage"] == "extended512"]
    replay_values = [x["extension_replayed_initial256_exact"] for x in extended
                     if x.get("extension_replayed_initial256_exact") is not None]
    return {
        "n": len(rows),
        "initial_final_n": sum(x["stage"] == "initial256" for x in rows),
        "extended_final_n": len(extended),
        "semantic_rate": _mean_bool(rows, "semantic_match"),
        "mode_valid_rate": _mean_bool(rows, "mode_valid"),
        "eos_rate": _mean_bool(rows, "has_eos"),
        "valid_eos_rate": _mean_bool(rows, "valid_eos"),
        "valid_mode_eos_rate": _mean_bool(rows, "valid_mode_eos"),
        "think_open_rate": float(np.mean([
            bool(x["generated_think_open_positions"]) for x in rows])),
        "think_close_rate": float(np.mean([
            bool(x["generated_think_close_positions"]) for x in rows])),
        "hit256_rate": _mean_bool(rows, "hit256"),
        "hit512_rate": _mean_bool(rows, "hit512"),
        "mean_tokens_final_stage": float(np.mean([x["n_tokens"] for x in rows])),
        "extension_replay_exact_rate": (
            float(np.mean(replay_values)) if replay_values else None),
    }


def final_rows(records: dict[tuple[str, str, str], dict[str, Any]],
               items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        for condition in CONDITIONS:
            initial = records.get((item["id"], condition, "initial256"))
            if initial is None:
                continue
            extended = records.get((item["id"], condition, "extended512"))
            rows.append(extended if extended is not None else initial)
    return rows


def legal_trajectory_gate(split: str, complete: bool,
                          by_condition: dict[str, Any]) -> dict[str, Any]:
    """Frozen endpoint gate; it never selects conditions or changes thresholds."""
    overall = {name: by_condition[name]["overall"] for name in CONDITIONS}
    mode_threshold = {
        "hard_no_think": 0.95,
        "hard_thinking": 0.80,
        "soft_no_think": 0.80,
        "soft_thinking": 0.80,
    }
    if split == "holdout":
        endpoint_threshold = {
            "hard_no_think": 0.90,
            "hard_thinking": 0.75,
            "soft_no_think": 0.80,
            "soft_thinking": 0.75,
        }
    else:
        endpoint_threshold = {
            "hard_no_think": 0.75,
            "hard_thinking": 0.50,
            "soft_no_think": 0.65,
            "soft_thinking": 0.50,
        }
    condition_checks = {}
    for name in CONDITIONS:
        task_rows = by_condition[name]["by_task"]
        task_coverage = sum(
            values.get("mode_valid_rate", 0.0) >= 0.75
            and values.get("valid_mode_eos_rate", 0.0) >= 0.25
            for values in task_rows.values())
        replay = overall[name].get("extension_replay_exact_rate")
        condition_checks[name] = {
            "mode_valid_passed": bool(
                overall[name].get("mode_valid_rate", 0.0) >= mode_threshold[name]),
            "valid_mode_eos_passed": bool(
                overall[name].get("valid_mode_eos_rate", 0.0)
                >= endpoint_threshold[name]),
            "task_coverage_n": int(task_coverage),
            "task_coverage_passed": task_coverage >= 6,
            "extension_replay_passed": replay is None or replay == 1.0,
        }
    passed = bool(complete and all(
        all(value[key] for key in (
            "mode_valid_passed", "valid_mode_eos_passed",
            "task_coverage_passed", "extension_replay_passed"))
        for value in condition_checks.values()))
    return {
        "passed": passed,
        "complete": complete,
        "condition_checks": condition_checks,
        "rule": "official modes only; mode-valid and valid-mode-EOS thresholds plus >=6/8 task coverage and exact 256-prefix replay",
    }


def write_summary(path: Path, manifest: dict[str, Any],
                  records: dict[tuple[str, str, str], dict[str, Any]],
                  items: list[dict[str, Any]]) -> None:
    selected = final_rows(records, items)
    expected = len(items) * len(CONDITIONS)
    complete_trajectories = 0
    for item in items:
        for condition in CONDITIONS:
            initial = records.get((item["id"], condition, "initial256"))
            if initial is None:
                continue
            if initial["has_eos"] or ((item["id"], condition, "extended512") in records):
                complete_trajectories += 1

    by_condition: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_rows = [x for x in selected if x["condition"] == condition]
        by_task = {}
        for task in sorted({x["task"] for x in condition_rows}):
            by_task[task] = summarize_rows([x for x in condition_rows if x["task"] == task])
        by_condition[condition] = {
            "overall": summarize_rows(condition_rows),
            "by_task": by_task,
        }

    complete = complete_trajectories == expected
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "split": manifest["split"],
        "manifest_sha256": manifest["manifest_sha256"],
        "generated_mode_parser_version": GENERATED_MODE_PARSER_VERSION,
        "generated_final_region_rule": manifest["generated_final_region_rule"],
        "expected_trajectories": expected,
        "complete_trajectories": complete_trajectories,
        "complete": complete,
        "jsonl_stage_rows": len(records),
        "final_rows_available": len(selected),
        "conditions": by_condition,
        "decision_gate": legal_trajectory_gate(
            manifest["split"], complete, by_condition),
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(path, summary)


def run(split: str, base_seed: int) -> None:
    if split == "holdout":
        log("Phase977: explicit holdout run requested; no tuning or condition selection is performed")
    ensure_dir(OUT)
    items, data_audit = load_split(split)
    if not items:
        raise RuntimeError(f"{split}: empty dataset")

    t0 = time.time()
    model = None
    manifest_path = OUT / f"manifest_{split}.json"
    rows_path = OUT / f"rows_{split}.jsonl"
    summary_path = OUT / f"summary_{split}.json"
    try:
        model, tok, device = load_model(MODEL_NAME)
        eos_ids = get_eos_ids(model, tok)
        think_open_id = single_token_id(tok, "<think>")
        think_close_id = single_token_id(tok, "</think>")
        manifest = make_manifest(split, items, data_audit, base_seed, model, tok,
                                 eos_ids, think_open_id, think_close_id)
        install_or_validate_manifest(manifest_path, manifest)
        records = load_jsonl(rows_path, manifest["manifest_sha256"])
        write_summary(summary_path, manifest, records, items)

        total = len(items) * len(CONDITIONS)
        completed = 0
        for item_index, item in enumerate(items):
            seed = stable_item_seed(base_seed, split, item["id"])
            for condition in CONDITIONS:
                initial_key = (item["id"], condition, "initial256")
                initial = records.get(initial_key)
                if initial is None:
                    input_ids, generated, user_prompt = generate_stage(
                        model, tok, device, eos_ids, item, condition, seed,
                        INITIAL_BUDGET)
                    initial = build_stage_row(
                        manifest, tok, item, condition, seed, "initial256",
                        input_ids, generated, user_prompt, eos_ids,
                        think_open_id, think_close_id, INITIAL_BUDGET, None)
                    append_jsonl(rows_path, initial)
                    records[initial_key] = initial

                if not initial["has_eos"]:
                    extended_key = (item["id"], condition, "extended512")
                    if extended_key not in records:
                        input_ids, generated, user_prompt = generate_stage(
                            model, tok, device, eos_ids, item, condition, seed,
                            EXTENDED_BUDGET)
                        extended = build_stage_row(
                            manifest, tok, item, condition, seed, "extended512",
                            input_ids, generated, user_prompt, eos_ids,
                            think_open_id, think_close_id, EXTENDED_BUDGET, initial)
                        append_jsonl(rows_path, extended)
                        records[extended_key] = extended

                completed += 1
                write_summary(summary_path, manifest, records, items)
                if completed % 8 == 0:
                    log(f"  Phase977 {split} trajectories {completed}/{total}")
            log(f"  Phase977 {split} items {item_index+1}/{len(items)}")

        elapsed = time.time() - t0
        write_summary(summary_path, manifest, records, items)
        log(f"Phase977 {split} complete; elapsed={elapsed/60:.1f} min; rows={rows_path}")
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", required=True,
        choices=("discovery", "development", "holdout"),
        help="Run exactly one split. Holdout is loaded only when explicitly requested.",
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.split, args.base_seed)
