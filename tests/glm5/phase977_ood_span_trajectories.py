#!/usr/bin/env python3
"""Phase 977: long-horizon OOD diagnostics for a frozen Qwen3 mode span.

This development-only experiment reads the frozen ``span_group`` from the
completed Phase-977 causal-decomposition artifact.  It then compares three
trajectories that all start from the *same official hard no-thinking token
prefix*:

* ``clean`` keeps the official prefix unchanged and is the sole legal mode;
* ``selected_span_zero`` zeroes the frozen span's prefill embeddings;
* ``selected_span_neutral`` replaces those embeddings with the embedding of a
  single newline token.

The latter two conditions are equal-length, out-of-distribution embedding
diagnostics.  They are not legal Qwen templates.  Generated think-tag patterns
under either corruption are recorded only as OOD behavioral features; they are
never equated with normal official thinking mode.

Before CUDA is touched, the script verifies the completed legal development
trajectory manifest/JSONL/summary, the causal-decomposition artifact hashes,
the frozen span selection, and the replicated development span gate.  It runs
only the frozen 64-item Qwen3 development corpus.  Every item shares one seed
across all three conditions.  A no-EOS 256-token trajectory is rerun from the
original prefix with the same seed for 512 tokens, and the first 256 generated
IDs must replay exactly.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS, load_model, release_model
from phase951_protocol_atlas import ensure_dir
from phase966_natural_stop import log
from phase973_conditional_trajectory import get_eos_ids
from phase977_dev_dataset import audit_dataset, build_dataset
from phase977_legal_mode_trajectories import (
    CONDITIONS as LEGAL_CONDITIONS,
    SCHEMA_VERSION as LEGAL_SCHEMA_VERSION,
    analyze_generation as legal_analyze_generation,
    dataset_hash,
    final_rows as legal_final_rows,
    generated_mode_analysis,
    legal_trajectory_gate,
    normalize_item,
    positions_of,
    render_prefix as legal_render_prefix,
    seed_everything,
    semantic_match,
    single_token_id,
    stable_item_seed,
    summarize_rows as legal_summarize_rows,
)


PHASE = 977
SCHEMA_VERSION = 1
MODEL_NAME = "qwen3"
SPLIT = "development"
N_ITEMS = 64
INITIAL_BUDGET = 256
EXTENDED_BUDGET = 512
BASE_SEED = 977_000

OUT = ROOT / "tests" / "glm5" / "result" / "phase977_ood_span_trajectories"
SPAN_RESULT = (
    ROOT / "tests" / "glm5" / "result"
    / "phase977_span_causal_decomposition" / "qwen3_result.json"
)
LEGAL_DIR = ROOT / "tests" / "glm5" / "result" / "phase977_legal_mode_trajectories"
LEGAL_MANIFEST = LEGAL_DIR / "manifest_development.json"
LEGAL_ROWS = LEGAL_DIR / "rows_development.jsonl"
LEGAL_SUMMARY = LEGAL_DIR / "summary_development.json"
LEGAL_DISCOVERY_REAUDIT = LEGAL_DIR / "reaudit_discovery.json"
SPAN_SCRIPT = ROOT / "tests" / "glm5" / "phase977_span_causal_decomposition.py"
LEGAL_SCRIPT = ROOT / "tests" / "glm5" / "phase977_legal_mode_trajectories.py"


CONDITION_ORDER = (
    "clean",
    "selected_span_zero",
    "selected_span_neutral",
)

CONDITIONS: dict[str, dict[str, Any]] = {
    "clean": {
        "embedding_intervention": None,
        "template_legality": "official_legal_hard_no_think",
        "interpretation": "legal clean reference",
    },
    "selected_span_zero": {
        "embedding_intervention": "zero",
        "template_legality": "OOD_equal_length_embedding_diagnostic_not_legal_template",
        "interpretation": "frozen-span OOD diagnostic only",
    },
    "selected_span_neutral": {
        "embedding_intervention": "neutral_newline_embedding",
        "template_legality": "OOD_equal_length_embedding_diagnostic_not_legal_template",
        "interpretation": "frozen-span OOD diagnostic only",
    },
}

SAMPLING = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
}

# This is fixed before any trajectory from this script is observed.  The
# effect criteria deliberately use only final, freely sampled endpoints from
# the staged 256/512 protocol.  Teacher-forced logits and OOD think-tag feature
# rates are not direct scientific pass/fail thresholds.  The strict generated
# mode parser still defines which decoded region is scored as the final answer.
FROZEN_GATE_SPEC: dict[str, Any] = {
    "name": "persistent_long_budget_ood_endpoint_effect",
    "population": "frozen Qwen3 Phase977 development64",
    "final_row_rule": (
        "use initial256 if it emitted EOS; otherwise use the same-seed "
        "extended512 full rerun"
    ),
    "natural_outcome": (
        "strict-parser final-answer semantic_match AND EOS on the final staged trajectory"
    ),
    "integrity_prerequisites": {
        "complete_trajectories": N_ITEMS * len(CONDITION_ORDER),
        "complete_pairs_per_ood_condition": N_ITEMS,
        "all_extended_prefixes_replay_exactly": True,
        "all_upstream_prerequisites_passed": True,
    },
    "effect_thresholds": {
        "clean_valid_eos_rate_min": 0.75,
        "clean_hard_no_think_adherence_rate_min": 0.95,
        "clean_minus_ood_valid_eos_rate_min_each": 0.20,
        "task_drop_min": 0.125,
        "tasks_with_drop_min_each": 6,
    },
    "threshold_source": (
        "frozen Phase977 span-causal natural-trajectory criterion: at least "
        "20 percentage points overall and the same direction in at least 6/8 tasks"
    ),
    "excluded_from_effect_gate": [
        "teacher_forced_next_token_logits",
        "EOS_logit_gap",
        "OOD_generated_think_tag_feature_rates_as_standalone_thresholds",
        "trajectory_token_identity_rate",
    ],
    "interpretation_limit": (
        "passing means only that both frozen equal-length OOD embedding "
        "corruptions retain a distributed natural valid-EOS cost through the "
        "staged long budget; it does not make either corruption a legal mode, "
        "normal thinking, a useful controller, a span-specific mechanism, or "
        "a write-layer localization. The joint endpoint does not by itself "
        "separate semantic damage from raw termination damage"
    ),
    "parser_dependency": (
        "OOD think-tag rates are not thresholded directly, but the reused strict "
        "parser uses generated close tags to delimit the final-answer region"
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain one JSON object: {path}")
    return value


def require_upstream_files() -> None:
    for required in (
        LEGAL_MANIFEST, LEGAL_ROWS, LEGAL_SUMMARY, LEGAL_DISCOVERY_REAUDIT,
        SPAN_RESULT, LEGAL_SCRIPT, SPAN_SCRIPT,
    ):
        if not required.is_file():
            raise RuntimeError(f"upstream prerequisite is absent: {required}")


def load_preflight_tokenizer():
    """Load only the local Qwen3 tokenizer; this never allocates a model/GPU."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[MODEL_NAME]["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def verify_embedded_manifest_hash(manifest: dict[str, Any]) -> bool:
    expected = manifest.get("manifest_sha256")
    core = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_sha256", "created_at_utc"}
    }
    return isinstance(expected, str) and sha256_json(core) == expected


def raw_development_hash(items: list[dict[str, Any]]) -> str:
    # This intentionally reproduces phase977_span_causal_decomposition.py.
    payload = json.dumps(items, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_upstream_jsonl(path: Path) -> tuple[
        dict[tuple[str, str, str], dict[str, Any]], int]:
    """Read an upstream artifact without repairing or mutating it."""
    if not path.is_file():
        raise RuntimeError(f"missing legal development JSONL: {path}")
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicate_keys = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid upstream JSONL line {line_number}: {path}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"upstream JSONL line {line_number} is not an object"
                )
            try:
                key = (str(row["id"]), str(row["condition"]), str(row["stage"]))
            except KeyError as exc:
                raise RuntimeError(
                    f"upstream JSONL line {line_number} misses {exc}"
                ) from exc
            if key in records:
                duplicate_keys += 1
            records[key] = row
    return records, duplicate_keys


def legal_rows_integrity(
    records: dict[tuple[str, str, str], dict[str, Any]],
    items: list[dict[str, Any]],
    manifest: dict[str, Any],
    tok,
) -> dict[str, Any]:
    conditions = tuple(LEGAL_CONDITIONS)
    errors: list[str] = []
    complete = 0
    exact_replays = 0
    extended_n = 0
    item_by_id = {item["id"]: item for item in items}
    item_ids = [item["id"] for item in items]
    allowed_ids = set(item_by_id)
    allowed_conditions = set(conditions)
    manifest_sha256 = str(manifest.get("manifest_sha256", ""))
    eos_ids = [int(value) for value in manifest.get("eos_token_ids", [])]
    try:
        think_open_id = single_token_id(tok, "<think>")
        think_close_id = single_token_id(tok, "</think>")
    except Exception as exc:
        return {
            "passed": False,
            "expected_trajectories": len(items) * len(conditions),
            "complete_trajectories": 0,
            "unique_stage_rows": len(records),
            "extended_trajectories": 0,
            "exact_extension_replays": 0,
            "errors": [f"runtime tokenizer think-token parsing failed: {exc}"],
            "error_count": 1,
        }
    if not eos_ids:
        errors.append("legal manifest has no EOS token IDs")
    if manifest.get("tokenizer_class") != type(tok).__name__:
        errors.append("legal manifest tokenizer class differs from runtime tokenizer")
    if manifest.get("special_token_ids") != {
        "think_open": think_open_id,
        "think_close": think_close_id,
    }:
        errors.append("legal manifest think token IDs differ from runtime tokenizer")

    prefix_cache: dict[tuple[str, str], tuple[str, str, list[int]]] = {}
    for item in items:
        for condition in conditions:
            try:
                prefix_cache[(item["id"], condition)] = legal_render_prefix(
                    tok, item, condition
                )
            except Exception as exc:
                errors.append(
                    f"legal runtime prefix failed: {item['id']}/{condition}: {exc}"
                )

    for key, row in records.items():
        item_id, condition, stage = key
        if item_id not in allowed_ids:
            errors.append(f"unexpected item id in legal rows: {item_id}")
            continue
        if condition not in allowed_conditions:
            errors.append(f"unexpected legal condition: {condition}")
            continue
        if stage not in {"initial256", "extended512"}:
            errors.append(f"unexpected legal stage: {stage}")
            continue
        if row.get("manifest_sha256") != manifest_sha256:
            errors.append(f"row/manifest mismatch: {key}")

        item = item_by_id[item_id]
        budget = INITIAL_BUDGET if stage == "initial256" else EXTENDED_BUDGET
        expected_seed = stable_item_seed(
            int(manifest.get("base_seed", -1)), SPLIT, item_id
        )
        expected_sampling = {
            name: LEGAL_CONDITIONS[condition][name]
            for name in ("temperature", "top_p", "top_k", "min_p")
        }
        prefix_value = prefix_cache.get((item_id, condition))
        if prefix_value is None:
            continue
        user_prompt, _rendered, input_ids = prefix_value
        expected_prefilled_open = positions_of(input_ids, {think_open_id})
        expected_prefilled_close = positions_of(input_ids, {think_close_id})
        metadata_ok = bool(
            row.get("schema_version") == LEGAL_SCHEMA_VERSION
            and row.get("phase") == PHASE
            and row.get("split") == SPLIT
            and row.get("task") == item["task"]
            and row.get("seed") == expected_seed
            and row.get("prompt") == item["prompt"]
            and row.get("effective_user_prompt") == user_prompt
            and row.get("answer") == item["answer"]
            and row.get("alias_groups") == item["alias_groups"]
            and row.get("exact") == item["exact"]
            and row.get("enable_thinking")
            == bool(LEGAL_CONDITIONS[condition]["enable_thinking"])
            and row.get("sampling") == expected_sampling
            and row.get("max_new_tokens") == budget
            and row.get("input_ids") == input_ids
            and row.get("prompt_len") == len(input_ids)
            and row.get("prefilled_think_open_positions")
            == expected_prefilled_open
            and row.get("prefilled_think_close_positions")
            == expected_prefilled_close
        )
        if not metadata_ok:
            errors.append(f"legal row metadata/prefix mismatch: {key}")

        raw_generated = row.get("generated_ids")
        if not isinstance(raw_generated, list) or not raw_generated:
            errors.append(f"legal row has empty/non-list generated_ids: {key}")
            continue
        if any(
            type(value) is not int or value < 0 or value >= len(tok)
            for value in raw_generated
        ):
            errors.append(f"legal row generated_ids outside tokenizer: {key}")
            continue
        if len(raw_generated) > budget:
            errors.append(f"legal row exceeds stage budget: {key}")
            continue
        try:
            recomputed = legal_analyze_generation(
                tok, item, condition, raw_generated, eos_ids,
                think_open_id, think_close_id, budget,
            )
        except Exception as exc:
            errors.append(f"legal row reparse failed: {key}: {exc}")
            continue
        for field, expected in recomputed.items():
            if row.get(field) != expected:
                errors.append(f"legal derived field mismatch: {key}/{field}")
        if recomputed["has_eos"]:
            if (
                len(recomputed["eos_positions"]) != 1
                or recomputed["eos_positions"][0] != len(raw_generated) - 1
            ):
                errors.append(f"legal row has nonterminal/multiple EOS: {key}")
        elif len(raw_generated) != budget:
            errors.append(f"legal no-EOS row did not reach its stage budget: {key}")
        if stage == "initial256":
            if not (
                row.get("hit256") == recomputed["hit_budget"]
                and row.get("hit512") is None
                and row.get("extension_strategy") is None
                and row.get("extension_replayed_initial256_exact") is None
            ):
                errors.append(f"legal initial-stage metadata mismatch: {key}")
        elif not (
            row.get("hit256") is True
            and row.get("hit512") == recomputed["hit_budget"]
            and row.get("extension_strategy")
            == "rerun_from_original_prompt_same_seed"
            and row.get("extension_replayed_initial256_exact") is True
            and row.get("initial256_n_tokens") == INITIAL_BUDGET
        ):
            errors.append(f"legal extended-stage metadata mismatch: {key}")

    for item_id in item_ids:
        for condition in conditions:
            initial = records.get((item_id, condition, "initial256"))
            if initial is None:
                errors.append(f"missing legal initial row: {item_id}/{condition}")
                continue
            initial_ids = [int(x) for x in initial.get("generated_ids", [])]
            if bool(initial.get("has_eos")):
                if (item_id, condition, "extended512") in records:
                    errors.append(
                        f"unexpected legal extension after EOS: {item_id}/{condition}"
                    )
                complete += 1
                continue
            if len(initial_ids) != INITIAL_BUDGET:
                errors.append(
                    f"no-EOS legal initial row is not 256 tokens: {item_id}/{condition}"
                )
            extended = records.get((item_id, condition, "extended512"))
            if extended is None:
                errors.append(f"missing legal extension: {item_id}/{condition}")
                continue
            extended_n += 1
            extended_ids = [int(x) for x in extended.get("generated_ids", [])]
            replay = extended_ids[:len(initial_ids)] == initial_ids
            if replay and extended.get("extension_replayed_initial256_exact") is True:
                exact_replays += 1
            else:
                errors.append(f"legal extension replay mismatch: {item_id}/{condition}")
            complete += 1

    expected = len(item_ids) * len(conditions)
    return {
        "passed": not errors and complete == expected and exact_replays == extended_n,
        "expected_trajectories": expected,
        "complete_trajectories": complete,
        "unique_stage_rows": len(records),
        "extended_trajectories": extended_n,
        "exact_extension_replays": exact_replays,
        "errors": errors[:20],
        "error_count": len(errors),
    }


def recompute_legal_summary_core(
    records: dict[tuple[str, str, str], dict[str, Any]],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute the legal summary fields that feed its development gate."""
    selected = legal_final_rows(records, items)
    expected = len(items) * len(LEGAL_CONDITIONS)
    complete_trajectories = 0
    for item in items:
        for condition in LEGAL_CONDITIONS:
            initial = records.get((item["id"], condition, "initial256"))
            if initial is None:
                continue
            if initial.get("has_eos") or (
                (item["id"], condition, "extended512") in records
            ):
                complete_trajectories += 1

    by_condition: dict[str, Any] = {}
    for condition in LEGAL_CONDITIONS:
        rows = [row for row in selected if row.get("condition") == condition]
        by_task = {}
        for task in sorted({row["task"] for row in rows}):
            by_task[task] = legal_summarize_rows([
                row for row in rows if row["task"] == task
            ])
        by_condition[condition] = {
            "overall": legal_summarize_rows(rows),
            "by_task": by_task,
        }
    complete = complete_trajectories == expected
    return {
        "expected_trajectories": expected,
        "complete_trajectories": complete_trajectories,
        "complete": complete,
        "jsonl_stage_rows": len(records),
        "final_rows_available": len(selected),
        "conditions": by_condition,
        "decision_gate": legal_trajectory_gate(SPLIT, complete, by_condition),
    }


def validate_upstream(items: list[dict[str, Any]], tok) -> dict[str, Any]:
    """Validate every prerequisite before model loading or output mutation."""
    require_upstream_files()

    legal_manifest = load_json_object(LEGAL_MANIFEST, "legal development manifest")
    legal_summary = load_json_object(LEGAL_SUMMARY, "legal development summary")
    legal_reaudit = load_json_object(
        LEGAL_DISCOVERY_REAUDIT, "strict-v2 discovery legal re-audit")
    span_result = load_json_object(SPAN_RESULT, "span causal Qwen3 result")
    legal_records, duplicate_keys = read_upstream_jsonl(LEGAL_ROWS)

    item_ids = [item["id"] for item in items]
    current_dataset_hash = dataset_hash(items)
    legal_manifest_hash = str(legal_manifest.get("manifest_sha256", ""))
    row_integrity = legal_rows_integrity(
        legal_records, items, legal_manifest, tok
    )
    recomputed_legal = recompute_legal_summary_core(legal_records, items)
    stored_legal_core = {
        key: legal_summary.get(key) for key in recomputed_legal
    }

    selected_group = (
        span_result.get("frozen_candidates", {}).get("span_group")
    )
    discovery_selected = span_result.get("span_discovery", {}).get("selected", {})
    span_dev = span_result.get("development", {})
    span_gate = span_dev.get("gates", {})
    span_summary = span_dev.get("span_summary", {})
    zero_ap = span_summary.get("zero", {}).get("answer_period", {})
    neutral_ap = span_summary.get("neutral", {}).get("answer_period", {})
    example_groups = span_result.get("template_manifest_example", {}).get("groups", {})
    legal_gate = legal_summary.get("decision_gate", {})
    legal_summary_file_hash = sha256_file(LEGAL_SUMMARY)

    checks: dict[str, bool] = {
        "strict_v2_discovery_gate_authorizes_downstream": bool(
            legal_reaudit.get("phase") == PHASE
            and legal_reaudit.get("split") == "discovery"
            and legal_reaudit.get("migration", {}).get("target_schema_version")
            == LEGAL_SCHEMA_VERSION
            and legal_reaudit.get("migration", {}).get("target_parser_version")
            == "strict_final_region_v2"
            and legal_reaudit.get("target_identity", {}).get("legal_script_sha256")
            == sha256_file(LEGAL_SCRIPT)
            and legal_reaudit.get("strict_v2_summary_recomputed", {}).get(
                "complete") is True
            and legal_reaudit.get("strict_v2_summary_recomputed", {}).get(
                "decision_gate", {}).get("passed") is True
            and legal_reaudit.get("downstream_gate", {}).get("no_go") is False
            and legal_reaudit.get("downstream_gate", {}).get("status") == "GO"
            and legal_reaudit.get("downstream_gate", {}).get(
                "strict_v2_gate_passed") is True
        ),
        "legal_manifest_internal_hash": verify_embedded_manifest_hash(legal_manifest),
        "legal_manifest_script_hash": (
            legal_manifest.get("script_sha256") == sha256_file(LEGAL_SCRIPT)
        ),
        "legal_manifest_qwen3_development64": bool(
            legal_manifest.get("schema_version") == LEGAL_SCHEMA_VERSION
            and legal_manifest.get("phase") == PHASE
            and legal_manifest.get("model") == MODEL_NAME
            and legal_manifest.get("split") == SPLIT
            and legal_manifest.get("n_items") == N_ITEMS
            and legal_manifest.get("item_ids") == item_ids
            and legal_manifest.get("dataset_sha256") == current_dataset_hash
            and legal_manifest.get("conditions") == LEGAL_CONDITIONS
            and legal_manifest.get("budgets") == {
                "initial": INITIAL_BUDGET,
                "extended": EXTENDED_BUDGET,
            }
        ),
        "legal_summary_manifest_link": (
            legal_summary.get("manifest_sha256") == legal_manifest_hash
        ),
        "legal_summary_complete": bool(
            legal_summary.get("schema_version") == LEGAL_SCHEMA_VERSION
            and legal_summary.get("phase") == PHASE
            and legal_summary.get("split") == SPLIT
            and legal_summary.get("complete") is True
            and legal_summary.get("expected_trajectories") == N_ITEMS * len(LEGAL_CONDITIONS)
            and legal_summary.get("complete_trajectories") == N_ITEMS * len(LEGAL_CONDITIONS)
            and legal_summary.get("final_rows_available") == N_ITEMS * len(LEGAL_CONDITIONS)
            and legal_summary.get("jsonl_stage_rows") == len(legal_records)
            and set(legal_summary.get("conditions", {})) == set(LEGAL_CONDITIONS)
        ),
        "legal_summary_recomputed_from_rows": (
            stored_legal_core == recomputed_legal
        ),
        "legal_development_gate": bool(
            legal_gate.get("passed") is True
            and legal_gate.get("complete") is True
        ),
        "legal_rows_complete_and_replay_exact": bool(row_integrity["passed"]),
        "legal_rows_no_duplicate_stage_keys": duplicate_keys == 0,
        "span_artifact_identity": bool(
            span_result.get("phase") == PHASE
            and span_result.get("schema_version") == 2
            and span_result.get("model") == MODEL_NAME
            and isinstance(span_result.get("n_layers"), int)
            and span_result.get("n_layers", 0) > 0
            and isinstance(span_result.get("eos_token_ids"), list)
            and bool(span_result.get("eos_token_ids"))
        ),
        "span_artifact_script_hash": (
            span_result.get("script_sha256") == sha256_file(SPAN_SCRIPT)
        ),
        "span_artifact_development_hash": (
            span_result.get("development_sha256") == raw_development_hash(items)
        ),
        "span_artifact_legal_summary_hash": (
            span_result.get("legal_development_summary_sha256")
            == legal_summary_file_hash
        ),
        "span_artifact_legal_reaudit_hash": (
            span_result.get("legal_discovery_reaudit_sha256")
            == sha256_file(LEGAL_DISCOVERY_REAUDIT)
        ),
        "span_and_legal_eos_token_sets_match": (
            span_result.get("eos_token_ids")
            == legal_manifest.get("eos_token_ids")
        ),
        "span_artifact_complete": bool(
            all(key in span_result for key in (
                "span_discovery", "marker_layer_scan", "legal_mode_layer_scan",
                "development", "frozen_candidates", "elapsed_seconds",
            ))
            and isinstance(span_result.get("elapsed_seconds"), (int, float))
            and span_result.get("elapsed_seconds", 0) > 0
            and span_result.get("split", {}).get("dev_n") == N_ITEMS
        ),
        "span_development_audit": bool(
            span_result.get("dev_audit", {}).get("passed") is True
            and span_result.get("dev_audit", {}).get("n_items") == N_ITEMS
            and not span_result.get("dev_audit", {}).get("errors")
            and not span_result.get("dev_audit", {}).get("schema_issues")
            and not span_result.get("dev_audit", {}).get("cross_set_overlap")
        ),
        "span_group_frozen_and_discovery_selected": bool(
            isinstance(selected_group, str)
            and bool(selected_group)
            and discovery_selected.get("group") == selected_group
            and discovery_selected.get("passes") is True
            and discovery_selected.get("selection_level") != "fallback_full"
            and selected_group in example_groups
            and isinstance(example_groups.get(selected_group), list)
            and bool(example_groups.get(selected_group))
        ),
        "span_development_span_replicated": bool(
            span_gate.get("span_replicated") is True
            and zero_ap.get("n") == N_ITEMS
            and neutral_ap.get("n") == N_ITEMS
            and isinstance(zero_ap.get("mean_delta_gap"), (int, float))
            and isinstance(neutral_ap.get("mean_delta_gap"), (int, float))
            and zero_ap.get("mean_delta_gap", -np.inf) >= 2.0
            and neutral_ap.get("mean_delta_gap", -np.inf) >= 2.0
            and zero_ap.get("positive_rate", -np.inf) >= 0.75
            and neutral_ap.get("positive_rate", -np.inf) >= 0.75
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "upstream integrity/development gate failed before model load: "
            + ", ".join(failed)
            + f"; legal_rows={row_integrity}"
        )

    return {
        "passed": True,
        "checks": checks,
        "selected_span_group": selected_group,
        "span_result_path": str(SPAN_RESULT),
        "span_result_sha256": sha256_file(SPAN_RESULT),
        "span_script_sha256": sha256_file(SPAN_SCRIPT),
        "legal_manifest_path": str(LEGAL_MANIFEST),
        "legal_manifest_file_sha256": sha256_file(LEGAL_MANIFEST),
        "legal_manifest_sha256": legal_manifest_hash,
        "legal_rows_path": str(LEGAL_ROWS),
        "legal_rows_sha256": sha256_file(LEGAL_ROWS),
        "legal_rows_duplicate_keys_latest_used": duplicate_keys,
        "legal_rows_integrity": row_integrity,
        "legal_runtime_identity": {
            "model_class": legal_manifest.get("model_class"),
            "model_name_or_path": legal_manifest.get("model_name_or_path"),
            "tokenizer_class": legal_manifest.get("tokenizer_class"),
            "eos_token_ids": legal_manifest.get("eos_token_ids"),
            "special_token_ids": legal_manifest.get("special_token_ids"),
        },
        "legal_summary_path": str(LEGAL_SUMMARY),
        "legal_summary_sha256": legal_summary_file_hash,
        "legal_discovery_reaudit_path": str(LEGAL_DISCOVERY_REAUDIT),
        "legal_discovery_reaudit_sha256": sha256_file(LEGAL_DISCOVERY_REAUDIT),
        "legal_gate_snapshot": legal_gate,
        "span_development_span_gate_snapshot": {
            "span_replicated": span_gate.get("span_replicated"),
            "zero_answer_period": zero_ap,
            "neutral_answer_period": neutral_ap,
        },
        "span_development_dataset_audit": span_result.get("dev_audit"),
    }


def token_ids(tok, text: str) -> list[int]:
    return list(tok(
        text, add_special_tokens=False, return_attention_mask=False
    ).input_ids)


def build_hard_no_think_prefix(
    tok, item: dict[str, Any], selected_group: str,
    think_open_id: int, think_close_id: int,
) -> dict[str, Any]:
    messages = [{"role": "user", "content": item["prompt"]}]
    thinking_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    no_think_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    thinking_ids = token_ids(tok, thinking_text)
    no_think_ids = token_ids(tok, no_think_text)
    suffix_ids = token_ids(tok, "<think>\n\n</think>\n\n")
    if len(suffix_ids) != 4:
        raise RuntimeError(
            f"{item['id']}: official no-think suffix is not four tokens: {suffix_ids}"
        )
    if suffix_ids[0] != think_open_id or suffix_ids[2] != think_close_id:
        raise RuntimeError(
            f"{item['id']}: runtime think token IDs disagree with official suffix"
        )
    if no_think_ids != thinking_ids + suffix_ids:
        raise RuntimeError(
            f"{item['id']}: official hard no-think prefix decomposition changed"
        )

    start = len(thinking_ids)
    groups = {
        "open_tag": [start],
        "inner_blank": [start + 1],
        "close_tag": [start + 2],
        "answer_separator": [start + 3],
        "open_half": [start, start + 1],
        "close_half": [start + 2, start + 3],
        "full_mode_block": list(range(start, start + 4)),
    }
    if selected_group not in groups:
        raise RuntimeError(f"unknown frozen span group: {selected_group}")
    positions = groups[selected_group]
    if not positions or any(pos < 0 or pos >= len(no_think_ids) for pos in positions):
        raise RuntimeError(f"{item['id']}: invalid selected positions {positions}")
    return {
        "rendered_prefix": no_think_text,
        "input_ids": no_think_ids,
        "thinking_prefix_ids": thinking_ids,
        "official_suffix_ids": suffix_ids,
        "official_suffix_tokens": tok.convert_ids_to_tokens(suffix_ids),
        "groups": groups,
        "selected_positions": positions,
    }


class PrefillEmbeddingIntervention:
    """Apply one equal-length embedding replacement, then remove its hook."""

    def __init__(self, embedding, condition: str, expected_ids: list[int],
                 positions: list[int], neutral_token_id: int):
        if condition not in {"selected_span_zero", "selected_span_neutral"}:
            raise ValueError(condition)
        self.embedding = embedding
        self.condition = condition
        self.expected_ids = [int(x) for x in expected_ids]
        self.positions = [int(x) for x in positions]
        self.neutral_token_id = int(neutral_token_id)
        self.handle = None
        self.hook_calls = 0
        self.applications = 0
        self.input_ids_exact = False
        self.input_length = None
        self.output_shape_before = None
        self.output_shape_after = None
        self.removed_inside_prefill = False

    def install(self) -> None:
        if self.handle is not None:
            raise RuntimeError("embedding hook was already installed")
        self.handle = self.embedding.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        self.hook_calls += 1
        if self.hook_calls != 1 or self.applications != 0:
            raise RuntimeError("embedding intervention was invoked more than once")
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise RuntimeError("unexpected embedding output type/shape")
        if not args or not isinstance(args[0], torch.Tensor) or args[0].ndim != 2:
            raise RuntimeError("embedding hook did not receive 2D input IDs")

        actual_ids = [int(x) for x in args[0][0].detach().to("cpu").tolist()]
        self.input_ids_exact = actual_ids == self.expected_ids
        self.input_length = len(actual_ids)
        self.output_shape_before = list(output.shape)
        if not self.input_ids_exact:
            raise RuntimeError("embedding hook did not fire on the exact official prefill")
        if output.shape[0] != 1 or output.shape[1] != len(self.expected_ids):
            raise RuntimeError("embedding hook fired outside the batch-one prefill")
        if any(pos < 0 or pos >= output.shape[1] for pos in self.positions):
            raise RuntimeError("selected embedding position is outside the prefill")

        replaced = output.clone()
        if self.condition == "selected_span_zero":
            replaced[:, self.positions, :] = 0
        else:
            neutral = module.weight[self.neutral_token_id].detach().to(
                device=replaced.device, dtype=replaced.dtype
            )
            if neutral.ndim != 1 or neutral.shape[0] != replaced.shape[-1]:
                raise RuntimeError("neutral embedding dimensionality mismatch")
            replaced[:, self.positions, :] = neutral

        self.output_shape_after = list(replaced.shape)
        if self.output_shape_after != self.output_shape_before:
            raise RuntimeError("embedding intervention changed tensor length/shape")
        self.applications += 1
        # Removing here guarantees decode-step embeddings are never touched.
        if self.handle is None:
            raise RuntimeError("embedding hook handle vanished before prefill")
        self.handle.remove()
        self.handle = None
        self.removed_inside_prefill = True
        return replaced

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def validate(self) -> None:
        if not (
            self.hook_calls == 1
            and self.applications == 1
            and self.input_ids_exact
            and self.input_length == len(self.expected_ids)
            and self.output_shape_before == self.output_shape_after
            and self.removed_inside_prefill
            and self.handle is None
        ):
            raise RuntimeError(f"prefill embedding hook audit failed: {self.stats()}")

    def stats(self) -> dict[str, Any]:
        return {
            "installed": True,
            "hook_calls": self.hook_calls,
            "applications": self.applications,
            "prefill_input_ids_exact": self.input_ids_exact,
            "prefill_input_length": self.input_length,
            "expected_prefill_length": len(self.expected_ids),
            "selected_positions": self.positions,
            "output_shape_before": self.output_shape_before,
            "output_shape_after": self.output_shape_after,
            "equal_length": self.output_shape_before == self.output_shape_after,
            "removed_inside_prefill": self.removed_inside_prefill,
            "decode_steps_intervened": 0 if self.removed_inside_prefill else None,
        }


def clean_hook_stats(prefix_len: int, positions: list[int]) -> dict[str, Any]:
    return {
        "installed": False,
        "hook_calls": 0,
        "applications": 0,
        "prefill_input_ids_exact": True,
        "prefill_input_length": prefix_len,
        "expected_prefill_length": prefix_len,
        "selected_positions": positions,
        "output_shape_before": None,
        "output_shape_after": None,
        "equal_length": True,
        "removed_inside_prefill": None,
        "decode_steps_intervened": 0,
    }


def analyze_generation(
    tok,
    item: dict[str, Any],
    generated_ids: list[int],
    eos_ids: list[int],
    think_open_id: int,
    think_close_id: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Reuse the legal script's strict semantics and generated-tag parser."""
    eos_set = {int(x) for x in eos_ids}
    eos_positions = positions_of(generated_ids, eos_set)
    first_eos_position = eos_positions[0] if eos_positions else None
    mode = generated_mode_analysis(
        tok, generated_ids, "hard_no_think", think_open_id, think_close_id
    )
    matched = semantic_match(
        item["alias_groups"], mode["final_text"], item["exact"]
    )
    has_eos = bool(eos_positions)
    return {
        "generated_ids": [int(x) for x in generated_ids],
        "raw": tok.decode(generated_ids, skip_special_tokens=False),
        "plain": tok.decode(generated_ids, skip_special_tokens=True),
        **mode,
        "semantic_match": bool(matched),
        "eos_positions": eos_positions,
        "first_eos_position": first_eos_position,
        "first_eos_step": (
            None if first_eos_position is None else first_eos_position + 1
        ),
        "first_eos_id": (
            None if first_eos_position is None
            else int(generated_ids[first_eos_position])
        ),
        "has_eos": has_eos,
        "valid_eos": bool(has_eos and matched),
        # This means adherence to the original hard-no-think expectation.  For
        # OOD rows it is not a claim that the corruption defines another mode.
        "valid_mode_eos": bool(has_eos and matched and mode["mode_valid"]),
        "n_tokens": len(generated_ids),
        "hit_budget": bool(
            len(generated_ids) >= max_new_tokens and not has_eos
        ),
    }


def generate_stage(
    model,
    tok,
    device,
    eos_ids: list[int],
    prefix: dict[str, Any],
    condition: str,
    seed: int,
    neutral_token_id: int,
    max_new_tokens: int,
) -> tuple[list[int], dict[str, Any]]:
    input_ids = [int(x) for x in prefix["input_ids"]]
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(tensor)
    controller = None
    if condition != "clean":
        controller = PrefillEmbeddingIntervention(
            model.get_input_embeddings(), condition, input_ids,
            prefix["selected_positions"], neutral_token_id,
        )
        controller.install()

    seed_everything(seed)
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=tensor,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=bool(SAMPLING["do_sample"]),
                temperature=float(SAMPLING["temperature"]),
                top_p=float(SAMPLING["top_p"]),
                top_k=int(SAMPLING["top_k"]),
                min_p=float(SAMPLING["min_p"]),
                use_cache=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos_ids,
                return_dict_in_generate=True,
            )
    finally:
        if controller is not None:
            controller.remove()

    if controller is not None:
        controller.validate()
        hook_stats = controller.stats()
    else:
        hook_stats = clean_hook_stats(
            len(input_ids), prefix["selected_positions"]
        )
    generated = [int(x) for x in output.sequences[0, len(input_ids):].tolist()]
    return generated, hook_stats


def make_manifest(
    items: list[dict[str, Any]],
    data_audit: dict[str, Any],
    upstream: dict[str, Any],
    model,
    tok,
    eos_ids: list[int],
    think_open_id: int,
    think_close_id: int,
    neutral_token_id: int,
) -> dict[str, Any]:
    selected_group = str(upstream["selected_span_group"])
    example = build_hard_no_think_prefix(
        tok, items[0], selected_group, think_open_id, think_close_id
    )
    core = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": "qwen3_frozen_span_long_horizon_ood_diagnostics",
        "model": MODEL_NAME,
        "model_class": type(model).__name__,
        "model_name_or_path": str(getattr(model.config, "_name_or_path", "")),
        "tokenizer_class": type(tok).__name__,
        "split": SPLIT,
        "n_items": len(items),
        "item_ids": [item["id"] for item in items],
        "dataset_sha256": dataset_hash(items),
        "dataset_audit": {
            "current_development_local": data_audit,
            "cross_set_audit_source": (
                "upstream span causal dev_audit, validated before model loading"
            ),
        },
        "script_sha256": sha256_file(Path(__file__)),
        "upstream_prerequisites": upstream,
        "frozen_span_group": selected_group,
        "conditions": CONDITIONS,
        "condition_order": list(CONDITION_ORDER),
        "template_protocol": (
            "all conditions use identical official enable_thinking=False token "
            "IDs; only zero/neutral prefill embeddings are OOD"
        ),
        "sampling": SAMPLING,
        "sampling_source": (
            "local Qwen3 README hard no-thinking recommendation: "
            "temperature=0.7, top_p=0.8, top_k=20, min_p=0"
        ),
        "budgets": {"initial": INITIAL_BUDGET, "extended": EXTENDED_BUDGET},
        "extension_rule": (
            "rerun from the original official prefix with the same item seed "
            "iff initial256 contains no EOS"
        ),
        "extension_replay_rule": "extended512 generated_ids[:256] must equal initial256 exactly",
        "base_seed": BASE_SEED,
        "seed_rule": (
            "phase977_legal_mode_trajectories.stable_item_seed; identical for "
            "clean/selected_span_zero/selected_span_neutral within each item"
        ),
        "embedding_hook_rule": (
            "OOD hook must match exact official input IDs, apply exactly once "
            "during batch-one prefill, preserve shape, remove itself before decode"
        ),
        "eos_token_ids": [int(x) for x in eos_ids],
        "special_token_ids": {
            "think_open_runtime": int(think_open_id),
            "think_close_runtime": int(think_close_id),
            "neutral_newline_runtime": int(neutral_token_id),
        },
        "template_example": {
            "item_id": items[0]["id"],
            "input_ids": example["input_ids"],
            "thinking_prefix_ids": example["thinking_prefix_ids"],
            "official_suffix_ids": example["official_suffix_ids"],
            "official_suffix_tokens": example["official_suffix_tokens"],
            "groups": example["groups"],
            "selected_positions": example["selected_positions"],
        },
        "strict_scoring_reuse": {
            "semantic_match": "phase977_legal_mode_trajectories.semantic_match",
            "generated_mode_analysis": (
                "phase977_legal_mode_trajectories.generated_mode_analysis "
                "with hard_no_think expectation"
            ),
        },
        "generated_feature_interpretation": (
            "think tags/nonempty spans after an OOD corruption are only OOD "
            "features, never evidence of normal official thinking"
        ),
        "frozen_gate_spec": FROZEN_GATE_SPEC,
        "torch_version": torch.__version__,
    }
    return {
        **core,
        "manifest_sha256": sha256_json(core),
        "created_at_utc": utc_now(),
    }


def install_or_validate_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        prior = load_json_object(path, "existing OOD trajectory manifest")
        if not verify_embedded_manifest_hash(prior):
            raise RuntimeError(f"existing manifest has an invalid digest: {path}")
        if prior.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError(
                f"manifest mismatch; refusing to mix runs at {path}\n"
                f"existing={prior.get('manifest_sha256')}\n"
                f"current={manifest['manifest_sha256']}"
            )
        return
    atomic_write_json(path, manifest)


def ensure_append_boundary(path: Path) -> None:
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
    ensure_append_boundary(path)
    payload = (
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["id"]), str(row["condition"]), str(row["stage"])


def load_jsonl(
    path: Path, manifest_sha256: str,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    lines = path.read_bytes().splitlines(keepends=True)
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    byte_offset = 0
    duplicates = 0
    for index, raw_line in enumerate(lines):
        line_start = byte_offset
        byte_offset += len(raw_line)
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if index == len(lines) - 1:
                with path.open("r+b") as handle:
                    handle.truncate(line_start)
                    handle.flush()
                    os.fsync(handle.fileno())
                log(f"Removed truncated final JSONL record in {path}")
                break
            raise
        if row.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError(f"row/manifest mismatch in {path} line {index + 1}")
        key = row_key(row)
        if key in records:
            duplicates += 1
        records[key] = row
    if duplicates:
        raise RuntimeError(
            f"{path} contains {duplicates} duplicate stage keys; refusing "
            "ambiguous recovery data"
        )
    return records


def build_stage_row(
    manifest: dict[str, Any],
    tok,
    item: dict[str, Any],
    prefix: dict[str, Any],
    condition: str,
    seed: int,
    stage: str,
    generated_ids: list[int],
    hook_stats: dict[str, Any],
    eos_ids: list[int],
    think_open_id: int,
    think_close_id: int,
    max_new_tokens: int,
    initial_row: dict[str, Any] | None,
) -> dict[str, Any]:
    analysis = analyze_generation(
        tok, item, generated_ids, eos_ids, think_open_id,
        think_close_id, max_new_tokens,
    )
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "manifest_sha256": manifest["manifest_sha256"],
        "split": SPLIT,
        "id": item["id"],
        "task": item["task"],
        "condition": condition,
        "condition_template_legality": CONDITIONS[condition]["template_legality"],
        "condition_interpretation": CONDITIONS[condition]["interpretation"],
        "stage": stage,
        "seed": int(seed),
        "shared_item_seed_across_conditions": True,
        "prompt": item["prompt"],
        "answer": item["answer"],
        "alias_groups": item["alias_groups"],
        "exact": item["exact"],
        "base_template": "official_hard_no_think_enable_thinking_false",
        "rendered_prefix": prefix["rendered_prefix"],
        "input_ids": [int(x) for x in prefix["input_ids"]],
        "prompt_len": len(prefix["input_ids"]),
        "selected_span_group": manifest["frozen_span_group"],
        "selected_positions": [int(x) for x in prefix["selected_positions"]],
        "embedding_hook_audit": hook_stats,
        "sampling": SAMPLING,
        "max_new_tokens": int(max_new_tokens),
        **analysis,
        "hit256": None,
        "hit512": None,
        "extension_strategy": None,
        "extension_replayed_initial256_exact": None,
        "generated_think_feature_interpretation": (
            "parser feature only; OOD rows are not normal thinking mode"
        ),
        "recorded_at_utc": utc_now(),
    }
    if stage == "initial256":
        row["hit256"] = bool(analysis["hit_budget"])
        if not analysis["has_eos"] and len(generated_ids) != INITIAL_BUDGET:
            raise RuntimeError(
                f"{item['id']}/{condition}: no-EOS initial rollout did not reach 256"
            )
    elif stage == "extended512":
        if initial_row is None:
            raise ValueError("extended512 requires initial256")
        if initial_row.get("has_eos"):
            raise RuntimeError("an EOS-complete initial row must not be extended")
        initial_ids = [int(x) for x in initial_row["generated_ids"]]
        if len(initial_ids) != INITIAL_BUDGET:
            raise RuntimeError("extended row's initial reference is not 256 tokens")
        if row["input_ids"] != initial_row["input_ids"]:
            raise RuntimeError("extended rerun did not use the original prefix")
        if int(seed) != int(initial_row["seed"]):
            raise RuntimeError("extended rerun did not reuse the original seed")
        replay = generated_ids[:INITIAL_BUDGET] == initial_ids
        if not replay:
            raise RuntimeError(
                f"{item['id']}/{condition}: extended512 failed exact first-256 replay"
            )
        row["hit256"] = True
        row["hit512"] = bool(analysis["hit_budget"])
        if not analysis["has_eos"] and len(generated_ids) != EXTENDED_BUDGET:
            raise RuntimeError(
                f"{item['id']}/{condition}: no-EOS extended rollout did not reach 512"
            )
        row["extension_strategy"] = "full_rerun_original_prefix_same_seed"
        row["extension_replayed_initial256_exact"] = True
        row["initial256_n_tokens"] = len(initial_ids)
    else:
        raise ValueError(stage)
    return row


def validate_resume_records(
    records: dict[tuple[str, str, str], dict[str, Any]],
    items: list[dict[str, Any]],
    manifest: dict[str, Any],
    tok,
    eos_ids: list[int],
    think_open_id: int,
    think_close_id: int,
) -> None:
    item_by_id = {item["id"]: item for item in items}
    prefixes = {
        item["id"]: build_hard_no_think_prefix(
            tok, item, manifest["frozen_span_group"],
            think_open_id, think_close_id,
        )
        for item in items
    }
    for key, row in records.items():
        item_id, condition, stage = key
        if item_id not in item_by_id or condition not in CONDITION_ORDER:
            raise RuntimeError(f"unexpected resumed row key: {key}")
        if stage not in {"initial256", "extended512"}:
            raise RuntimeError(f"unexpected resumed stage: {key}")
        item = item_by_id[item_id]
        prefix = prefixes[item_id]
        budget = INITIAL_BUDGET if stage == "initial256" else EXTENDED_BUDGET
        expected_seed = stable_item_seed(BASE_SEED, SPLIT, item_id)
        if int(row.get("seed", -1)) != expected_seed:
            raise RuntimeError(f"resumed row seed mismatch: {key}")
        if not (
            row.get("schema_version") == SCHEMA_VERSION
            and row.get("phase") == PHASE
            and row.get("split") == SPLIT
            and row.get("task") == item["task"]
            and row.get("prompt") == item["prompt"]
            and row.get("answer") == item["answer"]
            and row.get("alias_groups") == item["alias_groups"]
            and row.get("exact") == item["exact"]
            and row.get("shared_item_seed_across_conditions") is True
            and row.get("sampling") == SAMPLING
            and row.get("max_new_tokens") == budget
            and row.get("base_template")
            == "official_hard_no_think_enable_thinking_false"
        ):
            raise RuntimeError(f"resumed row metadata mismatch: {key}")
        if row.get("selected_span_group") != manifest["frozen_span_group"]:
            raise RuntimeError(f"resumed row span-group mismatch: {key}")
        legality = CONDITIONS[condition]["template_legality"]
        if not (
            row.get("condition_template_legality") == legality
            and row.get("condition_interpretation")
            == CONDITIONS[condition]["interpretation"]
            and row.get("rendered_prefix") == prefix["rendered_prefix"]
            and row.get("input_ids") == prefix["input_ids"]
            and row.get("prompt_len") == len(prefix["input_ids"])
            and row.get("selected_positions") == prefix["selected_positions"]
        ):
            raise RuntimeError(f"resumed row legality-label mismatch: {key}")

        raw_generated = row.get("generated_ids")
        if not isinstance(raw_generated, list) or not raw_generated:
            raise RuntimeError(f"resumed row has invalid generated_ids: {key}")
        if any(
            type(value) is not int or value < 0 or value >= len(tok)
            for value in raw_generated
        ):
            raise RuntimeError(f"resumed generated_ids are outside the tokenizer: {key}")
        try:
            generated_ids = [int(value) for value in raw_generated]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"resumed generated_ids are not integers: {key}") from exc
        if generated_ids != raw_generated or len(generated_ids) > budget:
            raise RuntimeError(f"resumed generated_ids type/length mismatch: {key}")
        recomputed = analyze_generation(
            tok, item, generated_ids, eos_ids, think_open_id,
            think_close_id, budget,
        )
        for field, expected in recomputed.items():
            if row.get(field) != expected:
                raise RuntimeError(
                    f"resumed row derived field mismatch: {key}/{field}; "
                    f"stored={row.get(field)!r}, recomputed={expected!r}"
                )
        if recomputed["has_eos"]:
            if (
                len(recomputed["eos_positions"]) != 1
                or recomputed["eos_positions"][0] != len(generated_ids) - 1
            ):
                raise RuntimeError(f"resumed row has nonterminal/multiple EOS: {key}")
        elif len(generated_ids) != budget:
            raise RuntimeError(f"resumed no-EOS row did not reach its budget: {key}")

        hook = row.get("embedding_hook_audit", {})
        if condition == "clean":
            hook_ok = bool(
                hook.get("applications") == 0
                and hook.get("hook_calls") == 0
                and hook.get("installed") is False
                and hook.get("decode_steps_intervened") == 0
                and hook.get("selected_positions") == prefix["selected_positions"]
                and hook.get("expected_prefill_length") == len(prefix["input_ids"])
            )
        else:
            before_shape = hook.get("output_shape_before")
            after_shape = hook.get("output_shape_after")
            hook_ok = bool(
                hook.get("applications") == 1
                and hook.get("hook_calls") == 1
                and hook.get("prefill_input_ids_exact") is True
                and hook.get("equal_length") is True
                and hook.get("decode_steps_intervened") == 0
                and hook.get("removed_inside_prefill") is True
                and hook.get("selected_positions") == prefix["selected_positions"]
                and hook.get("prefill_input_length") == len(prefix["input_ids"])
                and hook.get("expected_prefill_length") == len(prefix["input_ids"])
                and isinstance(before_shape, list)
                and len(before_shape) == 3
                and before_shape == after_shape
                and before_shape[0] == 1
                and before_shape[1] == len(prefix["input_ids"])
            )
        if not hook_ok:
            raise RuntimeError(f"resumed row embedding-hook audit failed: {key}")

        if stage == "initial256":
            if not (
                row.get("hit256") == recomputed["hit_budget"]
                and row.get("hit512") is None
                and row.get("extension_strategy") is None
                and row.get("extension_replayed_initial256_exact") is None
            ):
                raise RuntimeError(f"resumed initial-stage metadata mismatch: {key}")
        elif not (
            row.get("hit256") is True
            and row.get("hit512") == recomputed["hit_budget"]
            and row.get("extension_strategy")
            == "full_rerun_original_prefix_same_seed"
            and row.get("extension_replayed_initial256_exact") is True
            and row.get("initial256_n_tokens") == INITIAL_BUDGET
        ):
            raise RuntimeError(f"resumed extended-stage metadata mismatch: {key}")

    for item in items:
        available_prefixes = []
        for condition in CONDITION_ORDER:
            initial = records.get((item["id"], condition, "initial256"))
            extended = records.get((item["id"], condition, "extended512"))
            if extended is not None and initial is None:
                raise RuntimeError(f"orphan extension: {item['id']}/{condition}")
            if initial is not None:
                available_prefixes.append(initial["input_ids"])
                if not initial.get("has_eos"):
                    ids = [int(x) for x in initial.get("generated_ids", [])]
                    if len(ids) != INITIAL_BUDGET:
                        raise RuntimeError(
                            f"invalid resumed no-EOS initial length: {item['id']}/{condition}"
                        )
                elif extended is not None:
                    raise RuntimeError(
                        f"unexpected extension after EOS: {item['id']}/{condition}"
                    )
            if extended is not None:
                initial_ids = [int(x) for x in initial["generated_ids"]]
                extended_ids = [int(x) for x in extended.get("generated_ids", [])]
                if not (
                    extended.get("extension_replayed_initial256_exact") is True
                    and extended_ids[:len(initial_ids)] == initial_ids
                    and extended.get("input_ids") == initial.get("input_ids")
                ):
                    raise RuntimeError(
                        f"resumed extension replay mismatch: {item['id']}/{condition}"
                    )
        if available_prefixes and any(
            ids != available_prefixes[0] for ids in available_prefixes[1:]
        ):
            raise RuntimeError(f"conditions did not share token prefix: {item['id']}")


def final_rows(
    records: dict[tuple[str, str, str], dict[str, Any]],
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        for condition in CONDITION_ORDER:
            initial = records.get((item["id"], condition, "initial256"))
            if initial is None:
                continue
            if initial.get("has_eos"):
                rows.append(initial)
                continue
            extended = records.get((item["id"], condition, "extended512"))
            if extended is not None:
                rows.append(extended)
    return rows


def mean_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(np.mean([bool(row.get(key, False)) for row in rows]))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "semantic_rate": None,
            "eos_rate": None,
            "valid_eos_rate": None,
            "hard_no_think_adherence_rate": None,
            "valid_hard_no_think_eos_rate": None,
            "generated_think_open_feature_rate": None,
            "generated_think_close_feature_rate": None,
            "generated_nonempty_think_span_feature_rate": None,
            "hit256_rate": None,
            "hit512_rate_among_extended": None,
            "mean_tokens": None,
            "extension_replay_exact_rate": None,
        }
    extended = [row for row in rows if row["stage"] == "extended512"]
    replay = [
        bool(row["extension_replayed_initial256_exact"]) for row in extended
    ]
    return {
        "n": len(rows),
        "semantic_rate": mean_bool(rows, "semantic_match"),
        "eos_rate": mean_bool(rows, "has_eos"),
        "valid_eos_rate": mean_bool(rows, "valid_eos"),
        "hard_no_think_adherence_rate": mean_bool(rows, "mode_valid"),
        "valid_hard_no_think_eos_rate": mean_bool(rows, "valid_mode_eos"),
        "generated_think_open_feature_rate": float(np.mean([
            bool(row.get("generated_think_open_positions")) for row in rows
        ])),
        "generated_think_close_feature_rate": float(np.mean([
            bool(row.get("generated_think_close_positions")) for row in rows
        ])),
        "generated_nonempty_think_span_feature_rate": mean_bool(
            rows, "thinking_nonempty"
        ),
        "hit256_rate": mean_bool(rows, "hit256"),
        "hit512_rate_among_extended": (
            mean_bool(extended, "hit512") if extended else None
        ),
        "mean_tokens": float(np.mean([row["n_tokens"] for row in rows])),
        "extension_replay_exact_rate": (
            float(np.mean(replay)) if replay else None
        ),
        "interpretation_guard": (
            "generated think-tag fields are OOD features, not normal-thinking rates"
        ),
    }


def paired_summary(
    clean_rows: list[dict[str, Any]],
    ood_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    clean_by_id = {row["id"]: row for row in clean_rows}
    ood_by_id = {row["id"]: row for row in ood_rows}
    ids = sorted(set(clean_by_id) & set(ood_by_id))
    pairs = [(clean_by_id[item_id], ood_by_id[item_id]) for item_id in ids]
    if not pairs:
        return {
            "n_pairs": 0,
            "valid_eos_rate_delta_ood_minus_clean": None,
            "clean_minus_ood_valid_eos_rate": None,
            "paired_valid_eos_degradation_rate": None,
            "paired_valid_eos_improvement_rate": None,
            "paired_degradation_minus_improvement_rate": None,
            "natural_endpoint_changed_rate": None,
            "generated_ids_exact_same_rate": None,
            "mean_token_delta_ood_minus_clean": None,
            "tasks_with_valid_eos_drop_at_least_0_125": 0,
            "by_task": {},
        }

    def endpoint(row: dict[str, Any]) -> tuple[bool, ...]:
        return (
            bool(row["semantic_match"]), bool(row["has_eos"]),
            bool(row["valid_eos"]), bool(row["mode_valid"]),
            bool(row["hit_budget"]),
        )

    degraded = [bool(clean["valid_eos"]) and not bool(ood["valid_eos"])
                for clean, ood in pairs]
    improved = [not bool(clean["valid_eos"]) and bool(ood["valid_eos"])
                for clean, ood in pairs]
    clean_valid = float(np.mean([bool(clean["valid_eos"]) for clean, _ in pairs]))
    ood_valid = float(np.mean([bool(ood["valid_eos"]) for _, ood in pairs]))

    by_task: dict[str, Any] = {}
    for task in sorted({clean["task"] for clean, _ in pairs}):
        task_pairs = [(clean, ood) for clean, ood in pairs if clean["task"] == task]
        task_clean = float(np.mean([
            bool(clean["valid_eos"]) for clean, _ in task_pairs
        ]))
        task_ood = float(np.mean([
            bool(ood["valid_eos"]) for _, ood in task_pairs
        ]))
        by_task[task] = {
            "n_pairs": len(task_pairs),
            "clean_valid_eos_rate": task_clean,
            "ood_valid_eos_rate": task_ood,
            "clean_minus_ood_valid_eos_rate": task_clean - task_ood,
        }
    task_drop_n = sum(
        values["clean_minus_ood_valid_eos_rate"]
        >= FROZEN_GATE_SPEC["effect_thresholds"]["task_drop_min"]
        for values in by_task.values()
    )
    degraded_rate = float(np.mean(degraded))
    improved_rate = float(np.mean(improved))
    return {
        "n_pairs": len(pairs),
        "valid_eos_rate_delta_ood_minus_clean": ood_valid - clean_valid,
        "clean_minus_ood_valid_eos_rate": clean_valid - ood_valid,
        "paired_valid_eos_degradation_rate": degraded_rate,
        "paired_valid_eos_improvement_rate": improved_rate,
        "paired_degradation_minus_improvement_rate": degraded_rate - improved_rate,
        "natural_endpoint_changed_rate": float(np.mean([
            endpoint(clean) != endpoint(ood) for clean, ood in pairs
        ])),
        "generated_ids_exact_same_rate": float(np.mean([
            clean["generated_ids"] == ood["generated_ids"] for clean, ood in pairs
        ])),
        "mean_token_delta_ood_minus_clean": float(np.mean([
            int(ood["n_tokens"]) - int(clean["n_tokens"])
            for clean, ood in pairs
        ])),
        "tasks_with_valid_eos_drop_at_least_0_125": int(task_drop_n),
        "by_task": by_task,
    }


def decision_gate(
    complete: bool,
    by_condition: dict[str, Any],
    paired: dict[str, Any],
    upstream_passed: bool,
) -> dict[str, Any]:
    thresholds = FROZEN_GATE_SPEC["effect_thresholds"]
    clean = by_condition["clean"]["overall"]
    replay_values = [
        by_condition[name]["overall"].get("extension_replay_exact_rate")
        for name in CONDITION_ORDER
    ]
    replay_passed = all(value is None or value == 1.0 for value in replay_values)
    pair_counts_passed = all(
        paired[name].get("n_pairs") == N_ITEMS
        for name in ("selected_span_zero", "selected_span_neutral")
    )
    integrity_checks = {
        "upstream_prerequisites_passed": bool(upstream_passed),
        "complete_trajectories": bool(complete),
        "complete_pairs": bool(pair_counts_passed),
        "all_extended_prefixes_replay_exactly": bool(replay_passed),
    }

    effect_checks: dict[str, bool] = {
        "clean_valid_eos_baseline": bool(
            clean.get("valid_eos_rate") is not None
            and clean["valid_eos_rate"] >= thresholds["clean_valid_eos_rate_min"]
        ),
        "clean_hard_no_think_adherence": bool(
            clean.get("hard_no_think_adherence_rate") is not None
            and clean["hard_no_think_adherence_rate"]
            >= thresholds["clean_hard_no_think_adherence_rate_min"]
        ),
    }
    for name in ("selected_span_zero", "selected_span_neutral"):
        values = paired[name]
        effect_checks[f"{name}_persistent_valid_eos_drop"] = bool(
            values.get("clean_minus_ood_valid_eos_rate") is not None
            and values["clean_minus_ood_valid_eos_rate"]
            >= thresholds["clean_minus_ood_valid_eos_rate_min_each"]
        )
        effect_checks[f"{name}_task_coverage"] = bool(
            values.get("tasks_with_valid_eos_drop_at_least_0_125", 0)
            >= thresholds["tasks_with_drop_min_each"]
        )

    integrity_passed = all(integrity_checks.values())
    natural_effect_passed = all(effect_checks.values())
    return {
        "passed": bool(integrity_passed and natural_effect_passed),
        "integrity_passed": bool(integrity_passed),
        "natural_effect_passed": bool(natural_effect_passed),
        "integrity_checks": integrity_checks,
        "natural_long_budget_effect_checks": effect_checks,
        "frozen_spec": FROZEN_GATE_SPEC,
        "interpretation_limit": FROZEN_GATE_SPEC["interpretation_limit"],
    }


def write_summary(
    path: Path,
    manifest: dict[str, Any],
    records: dict[tuple[str, str, str], dict[str, Any]],
    items: list[dict[str, Any]],
) -> None:
    selected = final_rows(records, items)
    by_condition: dict[str, Any] = {}
    condition_rows: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITION_ORDER:
        rows = [row for row in selected if row["condition"] == condition]
        condition_rows[condition] = rows
        by_task = {}
        for task in sorted({item["task"] for item in items}):
            by_task[task] = summarize_rows([
                row for row in rows if row["task"] == task
            ])
        by_condition[condition] = {
            "template_legality": CONDITIONS[condition]["template_legality"],
            "overall": summarize_rows(rows),
            "by_task": by_task,
        }

    paired = {
        condition: paired_summary(condition_rows["clean"], condition_rows[condition])
        for condition in ("selected_span_zero", "selected_span_neutral")
    }
    expected = N_ITEMS * len(CONDITION_ORDER)
    complete = len(selected) == expected
    gate = decision_gate(
        complete, by_condition, paired,
        bool(manifest["upstream_prerequisites"].get("passed")),
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": manifest["experiment"],
        "split": SPLIT,
        "manifest_sha256": manifest["manifest_sha256"],
        "frozen_span_group": manifest["frozen_span_group"],
        "expected_trajectories": expected,
        "complete_trajectories": len(selected),
        "complete": complete,
        "jsonl_unique_stage_rows": len(records),
        "conditions": by_condition,
        "paired_ood_vs_clean": paired,
        "decision_gate": gate,
        "upstream_frozen_gate": manifest["upstream_prerequisites"],
        "interpretation_guard": {
            "clean": "official legal hard no-thinking template",
            "selected_span_zero": "OOD embedding diagnostic; not a legal template",
            "selected_span_neutral": "OOD embedding diagnostic; not a legal template",
            "generated_think_features": (
                "special-tag features under OOD corruption; not normal thinking; "
                "the strict parser may still use a close tag to delimit final text"
            ),
            "span_specificity": (
                "no matched non-selected-span long rollout is included, so a pass "
                "does not exclude generic template-embedding fragility"
            ),
            "joint_endpoint": (
                "valid_eos combines semantics and EOS; a pass alone does not isolate "
                "persistent termination damage from semantic damage"
            ),
        },
        "updated_at_utc": utc_now(),
    }
    atomic_write_json(path, summary)


def assert_forbidden_dataset_module_absent() -> None:
    forbidden = "phase977_" + "holdout_dataset"
    if forbidden in sys.modules:
        raise RuntimeError(
            f"development-only protocol violation: {forbidden} was imported"
        )


def run() -> None:
    legal_reaudit = load_json_object(
        LEGAL_DISCOVERY_REAUDIT, "strict-v2 discovery legal re-audit")
    discovery_gate = legal_reaudit.get("downstream_gate", {})
    if not (
        legal_reaudit.get("phase") == PHASE
        and legal_reaudit.get("split") == "discovery"
        and legal_reaudit.get("migration", {}).get("target_schema_version")
        == LEGAL_SCHEMA_VERSION
        and legal_reaudit.get("migration", {}).get("target_parser_version")
        == "strict_final_region_v2"
        and legal_reaudit.get("target_identity", {}).get("legal_script_sha256")
        == sha256_file(LEGAL_SCRIPT)
        and legal_reaudit.get("strict_v2_summary_recomputed", {}).get("complete") is True
        and legal_reaudit.get("strict_v2_summary_recomputed", {}).get(
            "decision_gate", {}).get("passed") is True
        and discovery_gate.get("no_go") is False
        and discovery_gate.get("status") == "GO"
        and discovery_gate.get("strict_v2_gate_passed") is True
    ):
        raise RuntimeError(
            "strict-v2 discovery legal gate is NO-GO; OOD span trajectories remain closed")
    assert_forbidden_dataset_module_absent()
    raw_items = build_dataset()
    items = [normalize_item(item) for item in raw_items]
    data_audit = audit_dataset(previous_prompts=None)
    if not (
        data_audit.get("passed") is True
        and data_audit.get("n_items") == N_ITEMS
        and not data_audit.get("errors")
        and not data_audit.get("schema_issues")
    ):
        raise RuntimeError(f"Phase977 development dataset audit failed: {data_audit}")
    if len(items) != N_ITEMS or len({item["id"] for item in items}) != N_ITEMS:
        raise RuntimeError("Phase977 development corpus is not exactly 64 unique items")

    # These gates intentionally precede ensure_dir and model loading.  A local
    # CPU tokenizer is sufficient to reparse every upstream generated row and
    # verify official templates; no model weights or CUDA are touched here.
    require_upstream_files()
    preflight_tok = load_preflight_tokenizer()
    try:
        upstream = validate_upstream(items, preflight_tok)
    finally:
        del preflight_tok
        gc.collect()
    assert_forbidden_dataset_module_absent()
    if not torch.cuda.is_available():
        raise RuntimeError("this formal Qwen3 trajectory test requires CUDA")

    manifest_path = OUT / "manifest_development.json"
    rows_path = OUT / "rows_development.jsonl"
    summary_path = OUT / "summary_development.json"
    t0 = time.time()
    model = None
    try:
        model, tok, device = load_model(MODEL_NAME)
        if getattr(device, "type", str(device).split(":")[0]) != "cuda":
            raise RuntimeError(f"Qwen3 was not loaded on CUDA: {device}")
        eos_ids = [int(x) for x in get_eos_ids(model, tok)]
        if not eos_ids:
            raise RuntimeError("Qwen3 EOS token set is empty")
        # Runtime tokenization is authoritative; no numerical think IDs are
        # hardcoded in this experiment.
        think_open_id = single_token_id(tok, "<think>")
        think_close_id = single_token_id(tok, "</think>")
        neutral_token_id = single_token_id(tok, "\n")
        runtime_identity = {
            "model_class": type(model).__name__,
            "model_name_or_path": str(getattr(model.config, "_name_or_path", "")),
            "tokenizer_class": type(tok).__name__,
            "eos_token_ids": eos_ids,
            "special_token_ids": {
                "think_open": think_open_id,
                "think_close": think_close_id,
            },
        }
        if runtime_identity != upstream["legal_runtime_identity"]:
            raise RuntimeError(
                "runtime Qwen3/tokenizer identity differs from the gated legal "
                f"development run: runtime={runtime_identity}, "
                f"upstream={upstream['legal_runtime_identity']}"
            )
        ensure_dir(OUT)
        manifest = make_manifest(
            items, data_audit, upstream, model, tok, eos_ids,
            think_open_id, think_close_id, neutral_token_id,
        )
        install_or_validate_manifest(manifest_path, manifest)
        records = load_jsonl(rows_path, manifest["manifest_sha256"])
        validate_resume_records(
            records, items, manifest, tok, eos_ids,
            think_open_id, think_close_id,
        )
        write_summary(summary_path, manifest, records, items)

        total = N_ITEMS * len(CONDITION_ORDER)
        completed = 0
        selected_group = manifest["frozen_span_group"]
        for item_index, item in enumerate(items):
            seed = stable_item_seed(BASE_SEED, SPLIT, item["id"])
            prefix = build_hard_no_think_prefix(
                tok, item, selected_group, think_open_id, think_close_id
            )
            for condition in CONDITION_ORDER:
                initial_key = (item["id"], condition, "initial256")
                initial = records.get(initial_key)
                if initial is not None and (
                    initial.get("input_ids") != prefix["input_ids"]
                    or initial.get("selected_positions")
                    != prefix["selected_positions"]
                ):
                    raise RuntimeError(
                        f"{item['id']}/{condition}: resumed row disagrees with "
                        "the runtime official prefix or frozen span positions"
                    )
                if initial is None:
                    generated, hook_stats = generate_stage(
                        model, tok, device, eos_ids, prefix, condition, seed,
                        neutral_token_id, INITIAL_BUDGET,
                    )
                    initial = build_stage_row(
                        manifest, tok, item, prefix, condition, seed,
                        "initial256", generated, hook_stats, eos_ids,
                        think_open_id, think_close_id, INITIAL_BUDGET, None,
                    )
                    append_jsonl(rows_path, initial)
                    records[initial_key] = initial

                if not initial["has_eos"]:
                    extended_key = (item["id"], condition, "extended512")
                    if extended_key not in records:
                        generated, hook_stats = generate_stage(
                            model, tok, device, eos_ids, prefix, condition,
                            seed, neutral_token_id, EXTENDED_BUDGET,
                        )
                        extended = build_stage_row(
                            manifest, tok, item, prefix, condition, seed,
                            "extended512", generated, hook_stats, eos_ids,
                            think_open_id, think_close_id, EXTENDED_BUDGET,
                            initial,
                        )
                        append_jsonl(rows_path, extended)
                        records[extended_key] = extended

                completed += 1
                write_summary(summary_path, manifest, records, items)
                if completed % 8 == 0:
                    log(f"  Phase977 OOD span trajectories {completed}/{total}")
            log(f"  Phase977 OOD span dev items {item_index + 1}/{N_ITEMS}")

        validate_resume_records(
            records, items, manifest, tok, eos_ids,
            think_open_id, think_close_id,
        )
        write_summary(summary_path, manifest, records, items)
        elapsed = time.time() - t0
        log(
            f"Phase977 OOD span development complete; "
            f"elapsed={elapsed / 60:.1f} min; rows={rows_path}"
        )
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        assert_forbidden_dataset_module_absent()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # There is intentionally no split/model/condition/threshold CLI: this is a
    # single frozen development protocol, not an adaptive trajectory browser.
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    run()
