#!/usr/bin/env python3
"""Phase 977 CPU-only, read-only re-audit of the frozen discovery rollout.

This migration does not generate tokens and never loads model weights.  It
strictly validates the schema-v1 discovery manifest, summary, JSONL protocol,
official chat prefixes, seeds, EOS accounting, and 256->512 replay.  It then
keeps every raw ``generated_ids`` sequence unchanged and re-derives every
generation field with the schema-v2 strict final-region parser.

The three source artifacts are never repaired or overwritten.  The only
output is the independent ``reaudit_discovery.json`` report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Make accidental CUDA use impossible before importing torch through the
# Phase977 module.  This script only instantiates a local tokenizer.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

import phase977_legal_mode_trajectories as legal  # noqa: E402
from model_utils import MODEL_CONFIGS  # noqa: E402


PHASE = 977
REAUDIT_SCHEMA_VERSION = 1
SOURCE_SCHEMA_VERSION = 1
TARGET_SCHEMA_VERSION = 2
SPLIT = "discovery"
EXPERIMENT = "qwen3_official_legal_mode_trajectories_reaudit"
DEFAULT_SOURCE_DIR = ROOT / legal.OUT
DEFAULT_OUTPUT = DEFAULT_SOURCE_DIR / "reaudit_discovery.json"

# The source manifest binds the generated rows to this immutable schema-v1
# implementation.  The implementation has now correctly moved to schema v2,
# so migration validation must compare against the frozen old identity rather
# than the current script bytes.
EXPECTED_LEGACY_SCRIPT_SHA256 = (
    "bf2780ac094d3af4c447e30543ef3eeae567b0048527bcc993a180fe5a76decb"
)
EXPECTED_LEGACY_MANIFEST_SHA256 = (
    "b526c223863e59647be9d0c727aa9a5cb8c3211fc131465bfb8b7bed5a4854de"
)

SOURCE_NAMES = (
    "manifest_discovery.json",
    "rows_discovery.jsonl",
    "summary_discovery.json",
)

LEGACY_ANALYSIS_FIELDS = (
    "generated_ids",
    "raw",
    "plain",
    "generated_think_open_positions",
    "generated_think_close_positions",
    "think_well_formed",
    "thinking_text",
    "thinking_nonempty",
    "mode_valid",
    "mode_valid_reason",
    "final_start_position",
    "final_source",
    "final_text",
    "semantic_match",
    "eos_positions",
    "first_eos_position",
    "first_eos_step",
    "first_eos_id",
    "has_eos",
    "valid_eos",
    "valid_mode_eos",
    "n_tokens",
    "hit_budget",
)

COMPARABLE_ANALYSIS_FIELDS = tuple(
    field for field in LEGACY_ANALYSIS_FIELDS if field != "generated_ids"
)

SUMMARY_METRICS = (
    "semantic_rate",
    "mode_valid_rate",
    "eos_rate",
    "valid_eos_rate",
    "valid_mode_eos_rate",
    "think_open_rate",
    "think_close_rate",
    "hit256_rate",
    "hit512_rate",
    "mean_tokens_final_stage",
    "extension_replay_exact_rate",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_json_equal(actual: Any, expected: Any, context: str) -> None:
    if legal.canonical_json(actual) != legal.canonical_json(expected):
        raise RuntimeError(
            f"{context}: mismatch\n"
            f"actual={legal.canonical_json(actual)}\n"
            f"expected={legal.canonical_json(expected)}"
        )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def load_jsonl_strict(path: Path, manifest_sha256: str) -> tuple[
        dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    """Read source JSONL without the repair/truncation behavior of run code."""
    payload = path.read_bytes()
    require(payload.endswith(b"\n"), f"{path}: source lacks a final newline")
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        require(bool(raw_line.strip()), f"{path}:{line_number}: blank line")
        try:
            row = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{path}:{line_number}: malformed source JSONL") from exc
        require(isinstance(row, dict), f"{path}:{line_number}: row is not an object")
        require(
            row.get("manifest_sha256") == manifest_sha256,
            f"{path}:{line_number}: row/manifest mismatch",
        )
        key = legal.row_key(row)
        require(key not in records, f"{path}:{line_number}: duplicate stage key {key}")
        records[key] = row
        ordered.append(row)
    return records, ordered


def manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in manifest.items()
        if key not in {"manifest_sha256", "created_at_utc"}
    }


def validate_manifest(manifest: dict[str, Any], items: list[dict[str, Any]]) -> None:
    require(manifest.get("schema_version") == SOURCE_SCHEMA_VERSION,
            "source manifest is not frozen schema v1")
    require(manifest.get("phase") == PHASE, "source manifest phase mismatch")
    require(manifest.get("experiment") == "qwen3_official_legal_mode_trajectories",
            "source experiment mismatch")
    require(manifest.get("model") == legal.MODEL_NAME, "source model mismatch")
    require(manifest.get("split") == SPLIT, "source split mismatch")
    require(manifest.get("n_items") == len(items), "source n_items mismatch")
    require(manifest.get("item_ids") == [item["id"] for item in items],
            "source item order mismatch")
    require(manifest.get("dataset_sha256") == legal.dataset_hash(items),
            "source dataset hash no longer matches discovery data")
    require(manifest.get("script_sha256") == EXPECTED_LEGACY_SCRIPT_SHA256,
            "source is not the frozen schema-v1 legal script artifact")
    require(manifest.get("manifest_sha256") == EXPECTED_LEGACY_MANIFEST_SHA256,
            "unexpected schema-v1 discovery manifest identity")
    require(manifest.get("manifest_sha256") == legal.sha256_json(manifest_core(manifest)),
            "source manifest self-hash is invalid")
    require_json_equal(manifest.get("conditions"), legal.CONDITIONS,
                       "source condition registry")
    require(manifest.get("main_condition_order") == list(legal.CONDITIONS),
            "source condition order mismatch")
    require_json_equal(
        manifest.get("budgets"),
        {"initial": legal.INITIAL_BUDGET, "extended": legal.EXTENDED_BUDGET},
        "source budgets",
    )
    require(isinstance(manifest.get("base_seed"), int), "invalid base seed")
    eos_ids = manifest.get("eos_token_ids")
    require(isinstance(eos_ids, list) and eos_ids and
            all(isinstance(value, int) for value in eos_ids), "invalid EOS IDs")
    require(len(eos_ids) == len(set(eos_ids)), "duplicate EOS IDs")
    special = manifest.get("special_token_ids")
    require(isinstance(special, dict), "missing special token IDs")
    require(all(isinstance(special.get(key), int)
                for key in ("think_open", "think_close")),
            "invalid generated think IDs")
    require(special["think_open"] != special["think_close"],
            "think open/close IDs collide")
    require(legal.SCHEMA_VERSION == TARGET_SCHEMA_VERSION,
            "current legal trajectory implementation is not schema v2")
    require(legal.GENERATED_MODE_PARSER_VERSION == "strict_final_region_v2",
            "current legal parser is not the reviewed strict parser")


def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[legal.MODEL_NAME]["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def validate_tokenizer_and_templates(tok, manifest: dict[str, Any],
                                     items: list[dict[str, Any]]) -> dict[str, Any]:
    require(type(tok).__name__ == manifest.get("tokenizer_class"),
            "runtime tokenizer class differs from source manifest")
    special = manifest["special_token_ids"]
    think_open_id = legal.single_token_id(tok, "<think>")
    think_close_id = legal.single_token_id(tok, "</think>")
    require(think_open_id == special["think_open"], "think-open ID mismatch")
    require(think_close_id == special["think_close"], "think-close ID mismatch")
    require(max(manifest["eos_token_ids"] + [think_open_id, think_close_id]) < len(tok),
            "manifest token ID exceeds runtime tokenizer vocabulary")

    expected_templates = legal.build_template_tokens(
        tok, items[0], think_open_id, think_close_id)
    require_json_equal(manifest.get("template_tokens"), expected_templates,
                       "manifest official template audit")
    return {
        "tokenizer_class": type(tok).__name__,
        "tokenizer_name_or_path": str(getattr(tok, "name_or_path", "")),
        "tokenizer_vocab_size": len(tok),
        "think_open_id": think_open_id,
        "think_close_id": think_close_id,
        "official_template_tokens_exact": True,
    }


def legacy_generated_mode_analysis_v1(tok, ids: list[int], condition: str,
                                      think_open_id: int,
                                      think_close_id: int) -> dict[str, Any]:
    """Frozen schema-v1 parser, used only to authenticate stored derivations."""
    opens = legal.positions_of(ids, {think_open_id})
    closes = legal.positions_of(ids, {think_close_id})
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

    if closes:
        final_start = closes[-1] + 1
        final_source = "after_last_generated_think_close"
    else:
        final_start = 0
        final_source = "full_generated_output_no_think_close"
    return {
        "generated_think_open_positions": opens,
        "generated_think_close_positions": closes,
        "think_well_formed": well_formed,
        "thinking_text": thinking_text,
        "thinking_nonempty": thinking_nonempty,
        "mode_valid": mode_valid,
        "mode_valid_reason": reason,
        "final_start_position": final_start,
        "final_source": final_source,
        "final_text": tok.decode(ids[final_start:], skip_special_tokens=True).strip(),
    }


def legacy_analyze_generation_v1(tok, item: dict[str, Any], condition: str,
                                 ids: list[int], eos_ids: list[int],
                                 think_open_id: int, think_close_id: int,
                                 max_new_tokens: int) -> dict[str, Any]:
    eos_positions = legal.positions_of(ids, {int(value) for value in eos_ids})
    first_eos_position = eos_positions[0] if eos_positions else None
    mode = legacy_generated_mode_analysis_v1(
        tok, ids, condition, think_open_id, think_close_id)
    matched = legal.semantic_match(item["alias_groups"], mode["final_text"],
                                   item["exact"])
    has_eos = bool(eos_positions)
    return {
        "generated_ids": [int(value) for value in ids],
        "raw": tok.decode(ids, skip_special_tokens=False),
        "plain": tok.decode(ids, skip_special_tokens=True),
        **mode,
        "semantic_match": matched,
        "eos_positions": eos_positions,
        "first_eos_position": first_eos_position,
        "first_eos_step": (None if first_eos_position is None
                           else first_eos_position + 1),
        "first_eos_id": (None if first_eos_position is None
                         else int(ids[first_eos_position])),
        "has_eos": has_eos,
        "valid_eos": bool(has_eos and matched),
        "valid_mode_eos": bool(has_eos and matched and mode["mode_valid"]),
        "n_tokens": len(ids),
        "hit_budget": bool(len(ids) >= max_new_tokens and not has_eos),
    }


def validate_row_metadata(row: dict[str, Any], item: dict[str, Any],
                          condition: str, stage: str,
                          manifest: dict[str, Any], tok,
                          expected_prefix: tuple[str, str, list[int]]) -> None:
    key_text = f"{item['id']}/{condition}/{stage}"
    require(row.get("schema_version") == SOURCE_SCHEMA_VERSION,
            f"{key_text}: source row schema mismatch")
    require(row.get("phase") == PHASE, f"{key_text}: phase mismatch")
    require(row.get("split") == SPLIT, f"{key_text}: split mismatch")
    for field in ("id", "task", "prompt", "answer", "alias_groups", "exact"):
        require_json_equal(row.get(field), item[field], f"{key_text}: {field}")
    require(row.get("condition") == condition, f"{key_text}: condition mismatch")
    require(row.get("stage") == stage, f"{key_text}: stage mismatch")
    expected_seed = legal.stable_item_seed(
        manifest["base_seed"], SPLIT, item["id"])
    require(row.get("seed") == expected_seed, f"{key_text}: seed mismatch")
    require(row.get("enable_thinking") ==
            bool(legal.CONDITIONS[condition]["enable_thinking"]),
            f"{key_text}: enable_thinking mismatch")
    require_json_equal(
        row.get("sampling"),
        {name: legal.CONDITIONS[condition][name]
         for name in ("temperature", "top_p", "top_k", "min_p")},
        f"{key_text}: sampling",
    )
    budget = (legal.INITIAL_BUDGET if stage == "initial256"
              else legal.EXTENDED_BUDGET)
    require(row.get("max_new_tokens") == budget, f"{key_text}: budget mismatch")

    expected_user_prompt, _rendered, expected_input_ids = expected_prefix
    require(row.get("effective_user_prompt") == expected_user_prompt,
            f"{key_text}: effective prompt mismatch")
    require_json_equal(row.get("input_ids"), expected_input_ids,
                       f"{key_text}: official prefix IDs")
    require(row.get("prompt_len") == len(expected_input_ids),
            f"{key_text}: prompt length mismatch")
    special = manifest["special_token_ids"]
    require_json_equal(
        row.get("prefilled_think_open_positions"),
        legal.positions_of(expected_input_ids, {special["think_open"]}),
        f"{key_text}: prefilled open positions",
    )
    require_json_equal(
        row.get("prefilled_think_close_positions"),
        legal.positions_of(expected_input_ids, {special["think_close"]}),
        f"{key_text}: prefilled close positions",
    )

    ids = row.get("generated_ids")
    require(isinstance(ids, list) and ids, f"{key_text}: empty generated IDs")
    require(all(isinstance(value, int) and not isinstance(value, bool)
                for value in ids), f"{key_text}: non-integer generated ID")
    require(min(ids) >= 0 and max(ids) < len(tok),
            f"{key_text}: generated ID outside tokenizer vocabulary")
    require(len(ids) <= budget, f"{key_text}: generated length exceeds budget")


def validate_and_rescore(
        records: dict[tuple[str, str, str], dict[str, Any]],
        ordered_rows: list[dict[str, Any]], items: list[dict[str, Any]],
        manifest: dict[str, Any], tok) -> tuple[
            dict[tuple[str, str, str], dict[str, Any]],
            list[dict[str, Any]], dict[str, Any]]:
    item_by_id = {item["id"]: item for item in items}
    expected_keys: set[tuple[str, str, str]] = set()
    prefix_cache: dict[tuple[str, str], tuple[str, str, list[int]]] = {}
    eos_ids = [int(value) for value in manifest["eos_token_ids"]]
    special = manifest["special_token_ids"]
    new_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    output_rows: list[dict[str, Any]] = []
    changed_common_counts: Counter[str] = Counter()
    structure_counts: Counter[str] = Counter()
    legacy_source_counts: Counter[str] = Counter()
    stage_semantic_true_to_false: list[str] = []
    stage_semantic_false_to_true: list[str] = []
    score_region_changed: list[str] = []

    for item in items:
        for condition in legal.CONDITIONS:
            prefix_cache[(item["id"], condition)] = legal.render_prefix(
                tok, item, condition)
            expected_keys.add((item["id"], condition, "initial256"))

    require(set(records).issuperset(expected_keys),
            "source is missing one or more initial256 rows")

    # Establish the exact protocol key set from each initial row's EOS status.
    for item in items:
        for condition in legal.CONDITIONS:
            initial_key = (item["id"], condition, "initial256")
            initial = records[initial_key]
            if not bool(initial.get("has_eos")):
                expected_keys.add((item["id"], condition, "extended512"))
    require(set(records) == expected_keys,
            f"stage key set mismatch: extra={sorted(set(records) - expected_keys)} "
            f"missing={sorted(expected_keys - set(records))}")

    for source_row in ordered_rows:
        key = legal.row_key(source_row)
        item_id, condition, stage = key
        require(item_id in item_by_id, f"{key}: unknown item")
        require(condition in legal.CONDITIONS, f"{key}: unknown condition")
        require(stage in {"initial256", "extended512"}, f"{key}: unknown stage")
        item = item_by_id[item_id]
        validate_row_metadata(
            source_row, item, condition, stage, manifest, tok,
            prefix_cache[(item_id, condition)],
        )
        generated_ids = [int(value) for value in source_row["generated_ids"]]
        budget = int(source_row["max_new_tokens"])

        legacy = legacy_analyze_generation_v1(
            tok, item, condition, generated_ids, eos_ids,
            special["think_open"], special["think_close"], budget,
        )
        for field in LEGACY_ANALYSIS_FIELDS:
            require_json_equal(source_row.get(field), legacy[field],
                               f"{item_id}/{condition}/{stage}: legacy {field}")

        rescored = legal.analyze_generation(
            tok, item, condition, generated_ids, eos_ids,
            special["think_open"], special["think_close"], budget,
        )
        require_json_equal(rescored["generated_ids"], generated_ids,
                           f"{key}: raw generated IDs were not preserved")
        require_json_equal(rescored["raw"], legacy["raw"], f"{key}: raw decode")
        require_json_equal(rescored["plain"], legacy["plain"], f"{key}: plain decode")
        for field in (
                "generated_think_open_positions", "generated_think_close_positions",
                "think_well_formed", "thinking_text", "thinking_nonempty",
                "mode_valid", "mode_valid_reason", "eos_positions",
                "first_eos_position", "first_eos_step", "first_eos_id", "has_eos",
                "n_tokens", "hit_budget"):
            require_json_equal(rescored[field], legacy[field],
                               f"{key}: parser-independent {field}")

        eos_positions = rescored["eos_positions"]
        if eos_positions:
            require(len(eos_positions) == 1 and eos_positions[0] == len(generated_ids) - 1,
                    f"{key}: EOS is not a unique terminal token")
        else:
            require(len(generated_ids) == budget and rescored["hit_budget"],
                    f"{key}: no-EOS generation did not exhaust its budget")

        changed_fields = [
            field for field in COMPARABLE_ANALYSIS_FIELDS
            if legal.canonical_json(legacy[field]) != legal.canonical_json(rescored[field])
        ]
        changed_common_counts.update(changed_fields)
        structure_counts[rescored["think_structure_status"]] += 1
        legacy_source_counts[legacy["final_source"]] += 1
        key_text = "/".join(key)
        if legacy["semantic_match"] and not rescored["semantic_match"]:
            stage_semantic_true_to_false.append(key_text)
        if not legacy["semantic_match"] and rescored["semantic_match"]:
            stage_semantic_false_to_true.append(key_text)
        if legacy["final_text"] != rescored["final_text"]:
            score_region_changed.append(key_text)

        migrated = {
            **source_row,
            "schema_version": TARGET_SCHEMA_VERSION,
            **rescored,
        }
        new_records[key] = migrated
        output_rows.append({
            "id": item_id,
            "task": item["task"],
            "condition": condition,
            "stage": stage,
            "source_row_sha256": legal.sha256_json(source_row),
            "raw_generated_ids_sha256": legal.sha256_json(generated_ids),
            "raw_generated_ids": generated_ids,
            "raw_generated_ids_preserved_exact": True,
            "legacy_parser_derived": {
                field: legacy[field] for field in COMPARABLE_ANALYSIS_FIELDS
            },
            "strict_v2_derived": {
                key_name: value for key_name, value in rescored.items()
                if key_name != "generated_ids"
            },
            "changed_common_derived_fields": changed_fields,
            "strict_v2_new_fields": {
                field: rescored[field] for field in (
                    "generated_mode_parser_version", "think_structure_status",
                    "final_region_valid")
            },
        })

    # Cross-stage extension validation is performed only after every raw row
    # has independently passed legacy authentication and v2 re-derivation.
    extension_count = 0
    for item in items:
        expected_seed = legal.stable_item_seed(
            manifest["base_seed"], SPLIT, item["id"])
        for condition in legal.CONDITIONS:
            initial_key = (item["id"], condition, "initial256")
            extended_key = (item["id"], condition, "extended512")
            initial = records[initial_key]
            extended = records.get(extended_key)
            require(initial["seed"] == expected_seed,
                    f"{initial_key}: initial seed mismatch")
            require(initial.get("hit256") == bool(initial["hit_budget"]),
                    f"{initial_key}: hit256 mismatch")
            require(initial.get("hit512") is None and
                    initial.get("extension_strategy") is None and
                    initial.get("extension_replayed_initial256_exact") is None,
                    f"{initial_key}: invalid initial extension metadata")
            if initial["has_eos"]:
                require(extended is None, f"{extended_key}: illegal extension after EOS")
                continue
            require(initial["n_tokens"] == legal.INITIAL_BUDGET,
                    f"{initial_key}: no-EOS initial length mismatch")
            require(extended is not None, f"{extended_key}: required extension missing")
            extension_count += 1
            require(extended["seed"] == expected_seed,
                    f"{extended_key}: extension seed mismatch")
            require_json_equal(extended["input_ids"], initial["input_ids"],
                               f"{extended_key}: replay prefix")
            require_json_equal(
                extended["generated_ids"][:legal.INITIAL_BUDGET],
                initial["generated_ids"],
                f"{extended_key}: exact replay of initial 256",
            )
            require(extended.get("extension_replayed_initial256_exact") is True,
                    f"{extended_key}: replay flag mismatch")
            require(extended.get("extension_strategy") ==
                    "rerun_from_original_prompt_same_seed",
                    f"{extended_key}: extension strategy mismatch")
            require(extended.get("initial256_n_tokens") == legal.INITIAL_BUDGET,
                    f"{extended_key}: initial length metadata mismatch")
            require(extended.get("hit256") is True,
                    f"{extended_key}: extension hit256 mismatch")
            require(extended.get("hit512") == bool(extended["hit_budget"]),
                    f"{extended_key}: hit512 mismatch")

    audit = {
        "source_stage_rows": len(records),
        "initial_rows": len(items) * len(legal.CONDITIONS),
        "extended_rows": extension_count,
        "all_stage_keys_exact": True,
        "all_source_schema_v1_rows_authenticated": True,
        "all_official_prefixes_exact": True,
        "all_seeds_exact": True,
        "all_eos_accounting_exact": True,
        "all_required_extensions_present": True,
        "all_extension_prefix_replays_exact": True,
        "all_raw_generated_ids_preserved_exact": True,
        "legacy_final_source_counts": dict(sorted(legacy_source_counts.items())),
        "strict_structure_counts": dict(sorted(structure_counts.items())),
        "changed_common_derived_field_counts": dict(sorted(changed_common_counts.items())),
        "score_region_changed_stage_n": len(score_region_changed),
        "score_region_changed_stage_keys": score_region_changed,
        "semantic_true_to_false_stage_n": len(stage_semantic_true_to_false),
        "semantic_true_to_false_stage_keys": stage_semantic_true_to_false,
        "semantic_false_to_true_stage_n": len(stage_semantic_false_to_true),
        "semantic_false_to_true_stage_keys": stage_semantic_false_to_true,
    }
    return new_records, output_rows, audit


def summarize(records: dict[tuple[str, str, str], dict[str, Any]],
              items: list[dict[str, Any]]) -> dict[str, Any]:
    selected = legal.final_rows(records, items)
    expected = len(items) * len(legal.CONDITIONS)
    complete_trajectories = 0
    for item in items:
        for condition in legal.CONDITIONS:
            initial = records[(item["id"], condition, "initial256")]
            if initial["has_eos"] or ((item["id"], condition, "extended512") in records):
                complete_trajectories += 1
    complete = complete_trajectories == expected
    by_condition: dict[str, Any] = {}
    for condition in legal.CONDITIONS:
        condition_rows = [row for row in selected if row["condition"] == condition]
        by_task = {
            task: legal.summarize_rows(
                [row for row in condition_rows if row["task"] == task])
            for task in sorted({row["task"] for row in condition_rows})
        }
        by_condition[condition] = {
            "overall": legal.summarize_rows(condition_rows),
            "by_task": by_task,
        }
    return {
        "expected_trajectories": expected,
        "complete_trajectories": complete_trajectories,
        "complete": complete,
        "jsonl_stage_rows": len(records),
        "final_rows_available": len(selected),
        "conditions": by_condition,
        "decision_gate": legal.legal_trajectory_gate(SPLIT, complete, by_condition),
    }


def validate_stored_summary(summary: dict[str, Any], manifest: dict[str, Any],
                            recomputed: dict[str, Any]) -> None:
    require(summary.get("schema_version") == SOURCE_SCHEMA_VERSION,
            "stored summary schema mismatch")
    require(summary.get("phase") == PHASE, "stored summary phase mismatch")
    require(summary.get("split") == SPLIT, "stored summary split mismatch")
    require(summary.get("manifest_sha256") == manifest["manifest_sha256"],
            "stored summary manifest mismatch")
    for field in (
            "expected_trajectories", "complete_trajectories", "complete",
            "jsonl_stage_rows", "final_rows_available", "conditions",
            "decision_gate"):
        require_json_equal(summary.get(field), recomputed[field],
                           f"stored legacy summary {field}")


def metric_delta(old: Any, new: Any) -> float | None:
    if old is None or new is None:
        return None
    return round(float(new) - float(old), 12)


def summary_difference(old: dict[str, Any], new: dict[str, Any],
                       old_records: dict[tuple[str, str, str], dict[str, Any]],
                       new_records: dict[tuple[str, str, str], dict[str, Any]],
                       items: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for condition in legal.CONDITIONS:
        old_condition = old["conditions"][condition]
        new_condition = new["conditions"][condition]
        overall = {
            metric: {
                "old": old_condition["overall"].get(metric),
                "new": new_condition["overall"].get(metric),
                "delta": metric_delta(old_condition["overall"].get(metric),
                                      new_condition["overall"].get(metric)),
            }
            for metric in SUMMARY_METRICS
        }
        by_task: dict[str, Any] = {}
        for task in old_condition["by_task"]:
            by_task[task] = {
                metric: {
                    "old": old_condition["by_task"][task].get(metric),
                    "new": new_condition["by_task"][task].get(metric),
                    "delta": metric_delta(
                        old_condition["by_task"][task].get(metric),
                        new_condition["by_task"][task].get(metric),
                    ),
                }
                for metric in (
                    "semantic_rate", "mode_valid_rate", "eos_rate",
                    "valid_eos_rate", "valid_mode_eos_rate")
            }
        by_condition[condition] = {"overall": overall, "by_task": by_task}

    old_final = {legal.row_key(row): row for row in legal.final_rows(old_records, items)}
    new_final = {legal.row_key(row): row for row in legal.final_rows(new_records, items)}
    require(set(old_final) == set(new_final), "old/new final row keys differ")
    semantic_true_to_false = [
        "/".join(key) for key in old_final
        if old_final[key]["semantic_match"] and not new_final[key]["semantic_match"]
    ]
    semantic_false_to_true = [
        "/".join(key) for key in old_final
        if not old_final[key]["semantic_match"] and new_final[key]["semantic_match"]
    ]
    final_text_changed = [
        "/".join(key) for key in old_final
        if old_final[key]["final_text"] != new_final[key]["final_text"]
    ]
    valid_eos_changed = [
        "/".join(key) for key in old_final
        if old_final[key]["valid_eos"] != new_final[key]["valid_eos"]
    ]
    valid_mode_eos_changed = [
        "/".join(key) for key in old_final
        if old_final[key]["valid_mode_eos"] != new_final[key]["valid_mode_eos"]
    ]
    old_gate = old["decision_gate"]
    new_gate = new["decision_gate"]
    return {
        "by_condition": by_condition,
        "final_rows": {
            "n": len(old_final),
            "final_text_changed_n": len(final_text_changed),
            "final_text_changed_keys": final_text_changed,
            "semantic_true_to_false_n": len(semantic_true_to_false),
            "semantic_true_to_false_keys": semantic_true_to_false,
            "semantic_false_to_true_n": len(semantic_false_to_true),
            "semantic_false_to_true_keys": semantic_false_to_true,
            "valid_eos_changed_n": len(valid_eos_changed),
            "valid_eos_changed_keys": valid_eos_changed,
            "valid_mode_eos_changed_n": len(valid_mode_eos_changed),
            "valid_mode_eos_changed_keys": valid_mode_eos_changed,
        },
        "decision_gate": {
            "old_passed": bool(old_gate["passed"]),
            "new_passed": bool(new_gate["passed"]),
            "passed_status_changed": bool(old_gate["passed"] != new_gate["passed"]),
            "old_condition_checks": old_gate["condition_checks"],
            "new_condition_checks": new_gate["condition_checks"],
        },
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(source_dir: Path, output: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output = output.resolve()
    source_paths = {name: source_dir / name for name in SOURCE_NAMES}
    for name, path in source_paths.items():
        require(path.is_file(), f"missing required source artifact: {name}")
        require(output != path.resolve(), f"refusing to overwrite source artifact: {path}")
    source_hashes_before = {name: sha256_file(path)
                            for name, path in source_paths.items()}

    manifest = load_json(source_paths["manifest_discovery.json"])
    stored_summary = load_json(source_paths["summary_discovery.json"])
    items, dataset_audit = legal.load_split(SPLIT)
    validate_manifest(manifest, items)
    records, ordered_rows = load_jsonl_strict(
        source_paths["rows_discovery.jsonl"], manifest["manifest_sha256"])

    tok = load_tokenizer()
    tokenizer_audit = validate_tokenizer_and_templates(tok, manifest, items)
    new_records, reaudit_rows, row_audit = validate_and_rescore(
        records, ordered_rows, items, manifest, tok)

    legacy_summary = summarize(records, items)
    validate_stored_summary(stored_summary, manifest, legacy_summary)
    strict_summary = summarize(new_records, items)
    differences = summary_difference(
        legacy_summary, strict_summary, records, new_records, items)

    source_hashes_after = {name: sha256_file(path)
                           for name, path in source_paths.items()}
    require_json_equal(source_hashes_after, source_hashes_before,
                       "source artifacts changed during read-only re-audit")
    current_legal_script = Path(legal.__file__).resolve()
    no_go = not bool(strict_summary["decision_gate"]["passed"])
    report = {
        "schema_version": REAUDIT_SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "split": SPLIT,
        "created_at_utc": utc_now(),
        "execution_contract": {
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "model_weights_loaded": False,
            "generation_performed": False,
            "source_artifacts_read_only": True,
            "output_is_independent": True,
        },
        "migration": {
            "source_schema_version": SOURCE_SCHEMA_VERSION,
            "target_schema_version": TARGET_SCHEMA_VERSION,
            "source_parser_version": "legacy_permissive_final_region_v1",
            "target_parser_version": legal.GENERATED_MODE_PARSER_VERSION,
            "reason": (
                "schema v1 scored the whole generated reasoning whenever no "
                "generated close tag existed; schema v2 leaves every tagged "
                "missing/unclosed/repeated/reversed/malformed structure unscored"
            ),
            "hard_no_think_no_generated_tags_rule":
                "the entire generated output remains the final region",
            "unique_ordered_pair_rule":
                "only text strictly after the unique generated close is final",
        },
        "source_artifacts": {
            name: {
                "path": str(path),
                "sha256_before": source_hashes_before[name],
                "sha256_after": source_hashes_after[name],
                "unchanged": source_hashes_before[name] == source_hashes_after[name],
            }
            for name, path in source_paths.items()
        },
        "source_identity": {
            "legacy_script_sha256": manifest["script_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "dataset_sha256": manifest["dataset_sha256"],
            "stored_summary_sha256": source_hashes_before["summary_discovery.json"],
            "rows_jsonl_sha256": source_hashes_before["rows_discovery.jsonl"],
            "manifest_self_hash_valid": True,
            "stored_summary_reproduced_exact": True,
        },
        "target_identity": {
            "legal_script_path": str(current_legal_script),
            "legal_script_sha256": sha256_file(current_legal_script),
            "legal_schema_version": legal.SCHEMA_VERSION,
            "generated_mode_parser_version": legal.GENERATED_MODE_PARSER_VERSION,
        },
        "dataset_audit": dataset_audit,
        "tokenizer_audit": tokenizer_audit,
        "row_and_protocol_audit": row_audit,
        "legacy_summary_recomputed": legacy_summary,
        "strict_v2_summary_recomputed": strict_summary,
        "old_new_differences": differences,
        "downstream_gate": {
            "status": "NO-GO" if no_go else "GO",
            "no_go": no_go,
            "reason": (
                "strict-v2 discovery decision gate failed; span and OOD runs "
                "must remain closed"
                if no_go else
                "strict-v2 discovery decision gate passed; downstream work may "
                "apply its remaining independent gates"
            ),
            "old_gate_passed": bool(legacy_summary["decision_gate"]["passed"]),
            "strict_v2_gate_passed": bool(strict_summary["decision_gate"]["passed"]),
            "gate_status_changed": bool(
                legacy_summary["decision_gate"]["passed"] !=
                strict_summary["decision_gate"]["passed"]),
            "required_downstream_reference": (
                "Pin this report's source manifest/rows hashes and require "
                "downstream_gate.no_go == false before span or OOD execution."
            ),
        },
        "stage_rows": reaudit_rows,
    }
    atomic_write_json(output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
        help="directory containing the frozen schema-v1 discovery artifacts",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="independent re-audit JSON path (source artifacts are never overwritten)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.source_dir, args.output)
    compact = {
        "output": str(args.output.resolve()),
        "stage_rows": report["row_and_protocol_audit"]["source_stage_rows"],
        "final_rows": report["strict_v2_summary_recomputed"]["final_rows_available"],
        "semantic_true_to_false_final_n": report[
            "old_new_differences"]["final_rows"]["semantic_true_to_false_n"],
        "valid_mode_eos_changed_final_n": report[
            "old_new_differences"]["final_rows"]["valid_mode_eos_changed_n"],
        "old_gate_passed": report["downstream_gate"]["old_gate_passed"],
        "strict_v2_gate_passed": report["downstream_gate"]["strict_v2_gate_passed"],
        "downstream_status": report["downstream_gate"]["status"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
