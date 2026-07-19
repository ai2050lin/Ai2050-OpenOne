#!/usr/bin/env python3
"""Phase 978 development budget stabilization for official Qwen3 modes.

The frozen Phase977 development trajectories at 256/512 tokens are design
data.  This runner authenticates those files, selects every trajectory that
exhausted 512 tokens without EOS, and reruns it from the original official
prefix with the same seed at 1024 and, only when needed, 1536 tokens.

This file never imports the sealed holdout dataset.  It only generates raw
development extensions; the separate CPU admission auditor owns the decision.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch


sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

from model_utils import MODEL_CONFIGS, load_model, release_model  # noqa: E402
import phase978_legal_core as legal  # noqa: E402

log = legal.log
get_eos_ids = legal.get_eos_ids


PHASE = 978
SCHEMA_VERSION = 1
MODEL_NAME = "qwen3"
SPLIT = "development"
SOURCE_BUDGET = 512
CHECKPOINTS = (256, 512, 1024, 1536)
NEW_STAGES = {1024: "extended1024", 1536: "extended1536"}
EXPECTED_ITEMS = 64
EXPECTED_SOURCE_ROWS = 408
EXPECTED_SOURCE_INITIAL = 256
EXPECTED_SOURCE_EXTENDED = 152
EXPECTED_HIT512 = 112
EXPECTED_HIT512_BY_CONDITION = {
    "hard_no_think": 3,
    "hard_thinking": 55,
    "soft_no_think": 2,
    "soft_thinking": 52,
}

SOURCE_DIR = ROOT / "tests" / "glm5" / "result" / "phase977_legal_mode_trajectories"
SOURCE_MANIFEST = SOURCE_DIR / "manifest_development.json"
SOURCE_ROWS = SOURCE_DIR / "rows_development.jsonl"
SOURCE_SUMMARY = SOURCE_DIR / "summary_development.json"
SOURCE_DATASET_SCRIPT = GLM5 / "phase977_dev_dataset.py"
SOURCE_LEGAL_SCRIPT = GLM5 / "phase977_legal_mode_trajectories.py"
HOLDOUT_SCRIPT = GLM5 / "phase977_holdout_dataset.py"
PROTOCOL_SCRIPT = GLM5 / "phase978_budget_protocol.py"

OUT = ROOT / "tests" / "glm5" / "result" / "phase978_legal_budget_stabilization"
PROTOCOL_PATH = OUT / "protocol_preregistration.json"
MANIFEST_PATH = OUT / "manifest_development.json"
ROWS_PATH = OUT / "rows_development.jsonl"
RUN_SUMMARY_PATH = OUT / "generator_status_development.json"

PINNED_SHA256 = {
    "phase977_dev_dataset.py": "ac28e7d0b1a806653564f8f9e330c59ab3134062b45d5c578a2616e2d6997399",
    "phase977_holdout_dataset.py": "d4d630f00a7c0197f6e7ba83704fdcf13121d67b5b09d3a77d649cb3fdff4755",
    "phase977_legal_mode_trajectories.py": "9b725cbac0cb5c975c4e588ee7f6924e60004154f0ac4cf2dbdcb9aa34a28481",
    "manifest_development.json": "de25a435eee181ebaa7219c4f5d8bb722cac948695cadd78c28243b5eb77bcb0",
    "rows_development.jsonl": "8b7c9b4d2f8a1d6e8e5bf0c6a9575a8545169f6b86053a1b7fe4fa83be3fe426",
    "summary_development.json": "48ae80112682e7f8b6dceab1202c2d7ade4b99d492deb9925432dabddc8d2968",
}
PINNED_SOURCE_MANIFEST_SHA256 = (
    "378a28959759177883ef72f6ad8f4c903c21f1762f2258f57b478e5545102544"
)
PINNED_DATASET_SHA256 = (
    "ff2703b07868f1b440b21e067385c9a3cdd56810c04b02670f607f151368e72d"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} is not one JSON object")
    return value


def manifest_core(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item for key, item in value.items()
        if key not in {"manifest_sha256", "created_at_utc"}
    }


def assert_holdout_not_imported() -> None:
    require(
        "phase977_holdout_dataset" not in sys.modules,
        "sealed holdout module was imported in the development process",
    )


def verify_pinned_files() -> dict[str, str]:
    paths = {
        SOURCE_DATASET_SCRIPT.name: SOURCE_DATASET_SCRIPT,
        HOLDOUT_SCRIPT.name: HOLDOUT_SCRIPT,
        SOURCE_LEGAL_SCRIPT.name: SOURCE_LEGAL_SCRIPT,
        SOURCE_MANIFEST.name: SOURCE_MANIFEST,
        SOURCE_ROWS.name: SOURCE_ROWS,
        SOURCE_SUMMARY.name: SOURCE_SUMMARY,
    }
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        require(path.is_file(), f"missing pinned artifact: {path}")
        hashes[name] = sha256_file(path)
        require(
            hashes[name] == PINNED_SHA256[name],
            f"pinned artifact changed: {name}: {hashes[name]}",
        )
    return hashes


def validate_protocol_lock() -> dict[str, Any]:
    protocol = load_json(PROTOCOL_PATH, "Phase978 protocol lock")
    core = {
        key: value for key, value in protocol.items()
        if key not in {"protocol_sha256", "created_at_utc"}
    }
    require(
        protocol.get("protocol_sha256") == legal.sha256_json(core),
        "Phase978 protocol self-hash is invalid",
    )
    require(protocol.get("phase") == PHASE, "protocol phase mismatch")
    require(protocol.get("checkpoints") == list(CHECKPOINTS), "checkpoint mismatch")
    require(protocol.get("decision_checkpoint") == 1536, "decision checkpoint mismatch")
    execution = protocol.get("execution_contract", {})
    require(execution.get("holdout_module_imported") is False and
            execution.get("holdout_module_parsed") is False,
            "protocol freeze reports holdout access")

    script_hashes = protocol.get("phase978_script_hashes", {})
    require(isinstance(script_hashes, dict), "protocol lacks script hashes")
    for entry in script_hashes.values():
        require(isinstance(entry, dict) and "path" in entry and "sha256" in entry,
                "invalid Phase978 script commitment")
        path = ROOT / str(entry["path"])
        require(path.is_file() and sha256_file(path) == entry["sha256"],
                f"Phase978 script changed after freeze: {entry.get('path')}")
    by_name = {
        Path(value["path"]).name: value["sha256"]
        for value in script_hashes.values()
        if isinstance(value, dict) and "path" in value and "sha256" in value
    }
    current = sha256_file(Path(__file__))
    require(by_name.get(Path(__file__).name) == current,
            "development runner differs from frozen protocol")
    require(by_name.get(PROTOCOL_SCRIPT.name) == sha256_file(PROTOCOL_SCRIPT),
            "protocol implementation differs from frozen protocol")
    for commitment in protocol.get("phase977_frozen_sources", {}).values():
        require(isinstance(commitment, dict) and "path" in commitment and
                "sha256" in commitment, "invalid frozen source commitment")
        source_path = ROOT / str(commitment["path"])
        require(source_path.is_file() and
                sha256_file(source_path) == commitment["sha256"],
                f"frozen execution/source artifact changed: {commitment.get('path')}")
    expected_runtime = protocol.get("runtime_versions", {})
    current_runtime = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_full": sys.version,
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "version_source": "installed_distribution_metadata_only",
    }
    require(current_runtime == expected_runtime,
            "Python/torch/transformers runtime differs from frozen protocol")
    model_identity = protocol.get("local_model_artifact_identity", {})
    model_root = ROOT / str(model_identity.get("path", ""))
    files = model_identity.get("files", {})
    require(model_root.is_dir() and isinstance(files, dict) and files,
            "protocol lacks local model identity")
    require(Path(MODEL_CONFIGS[MODEL_NAME]["path"]).resolve() == model_root.resolve(),
            "model loader registry path differs from frozen model path")
    for name, expected in files.items():
        path = model_root / name
        require(path.is_file() and path.stat().st_size == expected.get("bytes") and
                sha256_file(path) == expected.get("sha256"),
                f"local model artifact changed after protocol freeze: {name}")
    return protocol


def read_source_rows(manifest_sha256: str) -> tuple[
        dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    payload = SOURCE_ROWS.read_bytes()
    require(payload.endswith(b"\n"), "source development JSONL lacks final newline")
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.splitlines(), 1):
        require(bool(raw.strip()), f"source JSONL blank line {line_number}")
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"malformed source JSONL line {line_number}") from exc
        require(isinstance(row, dict), f"source line {line_number} is not an object")
        require(row.get("manifest_sha256") == manifest_sha256,
                f"source line {line_number} manifest mismatch")
        key = legal.row_key(row)
        require(key not in records, f"duplicate source stage key: {key}")
        records[key] = row
        ordered.append(row)
    require(len(ordered) == EXPECTED_SOURCE_ROWS,
            f"expected {EXPECTED_SOURCE_ROWS} source rows, got {len(ordered)}")
    return records, ordered


def validate_source_without_tokenizer() -> dict[str, Any]:
    """Authenticate immutable files and the exact 112 trajectory selection."""
    assert_holdout_not_imported()
    hashes_before = verify_pinned_files()
    manifest = load_json(SOURCE_MANIFEST, "Phase977 development manifest")
    summary = load_json(SOURCE_SUMMARY, "Phase977 development summary")
    require(manifest.get("manifest_sha256") == PINNED_SOURCE_MANIFEST_SHA256,
            "unexpected source manifest identity")
    require(manifest.get("manifest_sha256") == legal.sha256_json(manifest_core(manifest)),
            "source manifest self-hash invalid")
    require(manifest.get("schema_version") == 2, "source schema is not v2")
    require(manifest.get("phase") == 977, "source phase mismatch")
    require(manifest.get("split") == SPLIT, "source split mismatch")
    require(manifest.get("n_items") == EXPECTED_ITEMS, "source item count mismatch")
    require(manifest.get("dataset_sha256") == PINNED_DATASET_SHA256,
            "source dataset identity mismatch")
    require(manifest.get("script_sha256") == PINNED_SHA256[SOURCE_LEGAL_SCRIPT.name],
            "source script identity mismatch")
    require(manifest.get("generated_mode_parser_version") ==
            legal.GENERATED_MODE_PARSER_VERSION == "strict_final_region_v2",
            "strict-v2 parser mismatch")
    require(manifest.get("conditions") == legal.CONDITIONS,
            "official condition registry mismatch")
    require(manifest.get("main_condition_order") == list(legal.CONDITIONS),
            "official condition order mismatch")
    require(manifest.get("budgets") == {"initial": 256, "extended": 512},
            "source budget mismatch")
    require(manifest.get("base_seed") == legal.DEFAULT_BASE_SEED == 977000,
            "source seed mismatch")
    require(summary.get("manifest_sha256") == manifest["manifest_sha256"],
            "source summary manifest mismatch")
    require(summary.get("complete") is True and
            summary.get("complete_trajectories") == 256 and
            summary.get("jsonl_stage_rows") == EXPECTED_SOURCE_ROWS,
            "source summary is incomplete")

    records, ordered = read_source_rows(manifest["manifest_sha256"])
    initial = [row for row in ordered if row.get("stage") == "initial256"]
    extended = [row for row in ordered if row.get("stage") == "extended512"]
    require(len(initial) == EXPECTED_SOURCE_INITIAL, "source initial count mismatch")
    require(len(extended) == EXPECTED_SOURCE_EXTENDED, "source extension count mismatch")
    selected = [row for row in extended if row.get("hit512") is True]
    selected.sort(key=lambda row: (str(row["id"]), str(row["condition"])))
    require(len(selected) == EXPECTED_HIT512, "source hit512 selection count mismatch")
    counts = Counter(str(row["condition"]) for row in selected)
    require(dict(counts) == EXPECTED_HIT512_BY_CONDITION,
            f"source hit512 condition counts changed: {dict(counts)}")
    for row in selected:
        require(row.get("has_eos") is False and row.get("hit_budget") is True,
                f"selected source row is not a genuine hit512: {legal.row_key(row)}")
        require(row.get("n_tokens") == SOURCE_BUDGET and
                len(row.get("generated_ids", [])) == SOURCE_BUDGET,
                f"selected source row length mismatch: {legal.row_key(row)}")

    hashes_after = verify_pinned_files()
    require(hashes_after == hashes_before, "source changed during preflight")
    assert_holdout_not_imported()
    return {
        "manifest": manifest,
        "summary": summary,
        "records": records,
        "ordered": ordered,
        "selected": selected,
        "source_hashes": hashes_before,
    }


def load_dev_items() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    module = importlib.import_module("phase977_dev_dataset")
    external = module.audit_dataset()
    items = [legal.normalize_item(item) for item in module.build_dataset()]
    audit = {
        "external": external,
        "local": legal.audit_local_items(items),
        "cross_set_overlap_contract": (
            "authenticated from the frozen Phase977 development manifest; "
            "Phase978 does not re-import the historical discovery builder"
        ),
    }
    require(len(items) == EXPECTED_ITEMS, "runtime dev dataset count mismatch")
    require(legal.dataset_hash(items) == PINNED_DATASET_SHA256,
            "runtime dev dataset hash mismatch")
    require(audit["external"].get("ok") is True, "runtime dev dataset audit failed")
    assert_holdout_not_imported()
    return items, audit


def validate_source_with_tokenizer(
    source: dict[str, Any], tok, items: list[dict[str, Any]]
) -> dict[str, Any]:
    manifest = source["manifest"]
    records = source["records"]
    item_by_id = {item["id"]: item for item in items}
    eos_ids = [int(value) for value in manifest["eos_token_ids"]]
    special = manifest["special_token_ids"]
    require(legal.single_token_id(tok, "<think>") == special["think_open"],
            "runtime think-open token mismatch")
    require(legal.single_token_id(tok, "</think>") == special["think_close"],
            "runtime think-close token mismatch")
    require(legal.build_template_tokens(
        tok, items[0], special["think_open"], special["think_close"]
    ) == manifest["template_tokens"], "official template token mismatch")

    expected_keys: set[tuple[str, str, str]] = set()
    extension_n = 0
    for item in items:
        for condition in legal.CONDITIONS:
            initial_key = (item["id"], condition, "initial256")
            expected_keys.add(initial_key)
            initial = records.get(initial_key)
            require(initial is not None, f"missing source row {initial_key}")
            if not bool(initial.get("has_eos")):
                expected_keys.add((item["id"], condition, "extended512"))
                extension_n += 1
    require(set(records) == expected_keys, "source stage key set is not exact")
    require(extension_n == EXPECTED_SOURCE_EXTENDED, "source extension rule changed")

    analysis_fields = tuple(legal.analyze_generation(
        tok, items[0], "hard_no_think", [eos_ids[0]], eos_ids,
        special["think_open"], special["think_close"], 256,
    ).keys())

    for row in source["ordered"]:
        key = legal.row_key(row)
        item_id, condition, stage = key
        require(item_id in item_by_id and condition in legal.CONDITIONS,
                f"unknown source key {key}")
        item = item_by_id[item_id]
        budget = 256 if stage == "initial256" else 512
        require(stage in {"initial256", "extended512"}, f"unknown source stage {key}")
        require(row.get("schema_version") == 2 and row.get("phase") == 977 and
                row.get("split") == SPLIT and row.get("max_new_tokens") == budget,
                f"source protocol metadata mismatch {key}")
        for field in ("id", "task", "prompt", "answer", "alias_groups", "exact"):
            require(row.get(field) == item[field], f"source {field} mismatch {key}")
        require(row.get("seed") == legal.stable_item_seed(
            manifest["base_seed"], SPLIT, item_id), f"source seed mismatch {key}")
        expected_user, _rendered, input_ids = legal.render_prefix(tok, item, condition)
        require(row.get("effective_user_prompt") == expected_user and
                row.get("input_ids") == input_ids and
                row.get("prompt_len") == len(input_ids),
                f"source official prefix mismatch {key}")
        expected_sampling = {
            name: legal.CONDITIONS[condition][name]
            for name in ("temperature", "top_p", "top_k", "min_p")
        }
        require(row.get("sampling") == expected_sampling,
                f"source sampling mismatch {key}")
        generated = row.get("generated_ids")
        require(isinstance(generated, list) and generated and
                all(isinstance(value, int) and not isinstance(value, bool)
                    for value in generated), f"invalid source IDs {key}")
        rescored = legal.analyze_generation(
            tok, item, condition, generated, eos_ids,
            special["think_open"], special["think_close"], budget,
        )
        for field in analysis_fields:
            require(row.get(field) == rescored[field],
                    f"source strict-v2 field {field} mismatch {key}")
        eos_positions = rescored["eos_positions"]
        require(
            (len(eos_positions) == 1 and eos_positions[0] == len(generated) - 1)
            or (not eos_positions and len(generated) == budget and rescored["hit_budget"]),
            f"source termination invalid {key}",
        )
        if stage == "extended512":
            first = records[(item_id, condition, "initial256")]
            require(first["has_eos"] is False and
                    generated[:256] == first["generated_ids"] and
                    row.get("extension_replayed_initial256_exact") is True,
                    f"source 256 replay mismatch {key}")
    return {
        "eos_ids": eos_ids,
        "think_open_id": int(special["think_open"]),
        "think_close_id": int(special["think_close"]),
        "all_source_rows_strict_v2_recomputed": True,
        "all_source_terminations_exact": True,
        "all_source_256_replays_exact": True,
    }


def make_manifest(
    protocol: dict[str, Any], source: dict[str, Any], data_audit: dict[str, Any],
    tokenizer_audit: dict[str, Any], model, tok,
) -> dict[str, Any]:
    core = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": "qwen3_official_budget_stabilization_development",
        "split": SPLIT,
        "model": MODEL_NAME,
        "model_class": type(model).__name__,
        "model_name_or_path": str(getattr(model.config, "_name_or_path", "")),
        "tokenizer_class": type(tok).__name__,
        "protocol_sha256": protocol["protocol_sha256"],
        "runner_sha256": sha256_file(Path(__file__)),
        "source_artifact_sha256": source["source_hashes"],
        "source_manifest_sha256": source["manifest"]["manifest_sha256"],
        "source_dataset_sha256": source["manifest"]["dataset_sha256"],
        "source_stage_rows": EXPECTED_SOURCE_ROWS,
        "selected_hit512_trajectories": EXPECTED_HIT512,
        "selected_hit512_by_condition": EXPECTED_HIT512_BY_CONDITION,
        "conditions": legal.CONDITIONS,
        "condition_order": list(legal.CONDITIONS),
        "checkpoints": list(CHECKPOINTS),
        "new_stages": {str(key): value for key, value in NEW_STAGES.items()},
        "decision_checkpoint": 1536,
        "extension_rule": (
            "only source hit512 trajectories rerun to 1024; only hit1024 "
            "trajectories rerun to 1536; always original official prefix and same seed"
        ),
        "replay_rule": "all prior checkpoint generated token IDs must match exactly",
        "natural_endpoint_rule": "actual sampled terminal EOS; top1/logit rank is not the gate",
        "base_seed": source["manifest"]["base_seed"],
        "seed_rule": source["manifest"]["seed_rule"],
        "eos_token_ids": tokenizer_audit["eos_ids"],
        "special_token_ids": source["manifest"]["special_token_ids"],
        "generated_mode_parser_version": legal.GENERATED_MODE_PARSER_VERSION,
        "data_audit": data_audit,
        "source_tokenizer_audit": tokenizer_audit,
        "holdout_module_imported": False,
        "decision_owned_by": "phase978_dev_admission_audit.py",
    }
    return {
        **core,
        "manifest_sha256": legal.sha256_json(core),
        "created_at_utc": legal.utc_now(),
    }


def install_or_validate_manifest(manifest: dict[str, Any]) -> None:
    if MANIFEST_PATH.exists():
        prior = load_json(MANIFEST_PATH, "Phase978 development manifest")
        require(prior.get("manifest_sha256") == legal.sha256_json(manifest_core(prior)),
                "existing Phase978 dev manifest self-hash invalid")
        require(prior.get("manifest_sha256") == manifest["manifest_sha256"],
                "Phase978 dev manifest mismatch; refusing to mix runs")
        return
    legal.atomic_write_json(MANIFEST_PATH, manifest)


def _truncate_malformed_final_line(path: Path, offset: int) -> None:
    with path.open("r+b") as handle:
        handle.truncate(offset)
        handle.flush()
        os.fsync(handle.fileno())


def load_output_rows(manifest_sha256: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not ROWS_PATH.exists():
        return {}
    payload = ROWS_PATH.read_bytes()
    lines = payload.splitlines(keepends=True)
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    offset = 0
    for index, raw in enumerate(lines):
        start = offset
        offset += len(raw)
        require(bool(raw.strip()), f"blank output JSONL line {index + 1}")
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index == len(lines) - 1:
                _truncate_malformed_final_line(ROWS_PATH, start)
                log("Removed one malformed final Phase978 dev JSONL fragment")
                break
            raise RuntimeError(f"malformed output JSONL line {index + 1}") from exc
        require(isinstance(row, dict), f"output line {index + 1} is not an object")
        require(row.get("manifest_sha256") == manifest_sha256,
                f"output line {index + 1} manifest mismatch")
        key = legal.row_key(row)
        require(key not in records, f"duplicate output row key: {key}")
        records[key] = row
    return records


def build_extension_row(
    manifest: dict[str, Any], tok, item: dict[str, Any], condition: str,
    seed: int, stage: str, max_new_tokens: int, input_ids: list[int],
    generated_ids: list[int], user_prompt: str, eos_ids: list[int],
    think_open_id: int, think_close_id: int, source_row: dict[str, Any],
) -> dict[str, Any]:
    analysis = legal.analyze_generation(
        tok, item, condition, generated_ids, eos_ids,
        think_open_id, think_close_id, max_new_tokens,
    )
    frozen_ids = [int(value) for value in source_row["generated_ids"]]
    source_budget = int(source_row["max_new_tokens"])
    replay_exact = generated_ids[:source_budget] == frozen_ids
    row = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "manifest_sha256": manifest["manifest_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "split": SPLIT,
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
        "enable_thinking": bool(legal.CONDITIONS[condition]["enable_thinking"]),
        "sampling": {
            name: legal.CONDITIONS[condition][name]
            for name in ("temperature", "top_p", "top_k", "min_p")
        },
        "max_new_tokens": int(max_new_tokens),
        "input_ids": [int(value) for value in input_ids],
        "prompt_len": len(input_ids),
        "prefilled_think_open_positions": legal.positions_of(
            input_ids, {think_open_id}),
        "prefilled_think_close_positions": legal.positions_of(
            input_ids, {think_close_id}),
        **analysis,
        "source_stage": source_row["stage"],
        "source_budget": source_budget,
        "source_generated_ids_sha256": legal.sha256_json(frozen_ids),
        "extension_strategy": "full_rerun_original_official_prefix_same_seed",
        "replay_prior_checkpoint_exact": replay_exact,
        "hit1024": bool(analysis["hit_budget"]) if max_new_tokens == 1024 else True,
        "hit1536": bool(analysis["hit_budget"]) if max_new_tokens == 1536 else None,
        "decision_checkpoint": 1536,
        "natural_endpoint_is_actual_sampled_eos": True,
        "holdout_module_imported": False,
        "recorded_at_utc": legal.utc_now(),
    }
    return row


def validate_extension_row(
    row: dict[str, Any], manifest: dict[str, Any], tok, item: dict[str, Any],
    source_row: dict[str, Any], eos_ids: list[int], think_open_id: int,
    think_close_id: int,
) -> None:
    condition = str(source_row["condition"])
    stage = str(row.get("stage"))
    budget = 1024 if stage == "extended1024" else 1536
    require(stage == NEW_STAGES[budget], f"invalid extension stage: {stage}")
    generated = row.get("generated_ids")
    require(isinstance(generated, list) and generated and
            all(isinstance(value, int) and not isinstance(value, bool)
                for value in generated), f"invalid output IDs {item['id']}/{condition}/{stage}")
    frozen = [int(value) for value in source_row["generated_ids"]]
    source_budget = int(source_row["max_new_tokens"])
    recomputed = legal.analyze_generation(
        tok, item, condition, generated, eos_ids,
        think_open_id, think_close_id, budget,
    )
    expected_user, _rendered, expected_input_ids = legal.render_prefix(tok, item, condition)
    expected_sampling = {
        name: legal.CONDITIONS[condition][name]
        for name in ("temperature", "top_p", "top_k", "min_p")
    }
    require(
        row.get("manifest_sha256") == manifest["manifest_sha256"]
        and row.get("protocol_sha256") == manifest["protocol_sha256"]
        and row.get("schema_version") == SCHEMA_VERSION
        and row.get("phase") == PHASE
        and row.get("split") == SPLIT
        and row.get("id") == item["id"]
        and row.get("task") == item["task"]
        and row.get("condition") == condition
        and row.get("seed") == source_row["seed"]
        and row.get("prompt") == item["prompt"]
        and row.get("effective_user_prompt") == expected_user
        and row.get("answer") == item["answer"]
        and row.get("alias_groups") == item["alias_groups"]
        and row.get("exact") == item["exact"]
        and row.get("sampling") == expected_sampling
        and row.get("max_new_tokens") == budget
        and row.get("input_ids") == expected_input_ids
        and row.get("prompt_len") == len(expected_input_ids)
        and row.get("source_stage") == source_row["stage"]
        and row.get("source_budget") == source_budget
        and row.get("source_generated_ids_sha256") == legal.sha256_json(frozen)
        and row.get("replay_prior_checkpoint_exact") is True
        and generated[:source_budget] == frozen
        and row.get("natural_endpoint_is_actual_sampled_eos") is True
        and row.get("holdout_module_imported") is False,
        f"invalid extension metadata/replay {item['id']}/{condition}/{stage}",
    )
    for key, value in recomputed.items():
        require(row.get(key) == value,
                f"output derived field {key} mismatch {item['id']}/{condition}/{stage}")
    eos_positions = recomputed["eos_positions"]
    require(
        (len(eos_positions) == 1 and eos_positions[0] == len(generated) - 1
         and source_budget < len(generated) <= budget)
        or (not eos_positions and len(generated) == budget and recomputed["hit_budget"]),
        f"invalid output termination {item['id']}/{condition}/{stage}",
    )
    require(row.get("hit1024") == (recomputed["hit_budget"] if budget == 1024 else True),
            f"hit1024 flag mismatch {item['id']}/{condition}/{stage}")
    require(row.get("hit1536") == (recomputed["hit_budget"] if budget == 1536 else None),
            f"hit1536 flag mismatch {item['id']}/{condition}/{stage}")


def write_generator_status(
    manifest: dict[str, Any], records: dict[tuple[str, str, str], dict[str, Any]],
    selected: list[dict[str, Any]], elapsed: float,
) -> None:
    completed1024 = sum(
        (str(row["id"]), str(row["condition"]), "extended1024") in records
        for row in selected
    )
    hit1024_keys = [
        (str(row["id"]), str(row["condition"]), "extended1024")
        for row in selected
        if (str(row["id"]), str(row["condition"]), "extended1024") in records
        and records[(str(row["id"]), str(row["condition"]), "extended1024")]["hit1024"]
    ]
    completed1536 = sum(
        (item_id, condition, "extended1536") in records
        for item_id, condition, _stage in hit1024_keys
    )
    complete = completed1024 == len(selected) and completed1536 == len(hit1024_keys)
    status = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": manifest["experiment"],
        "split": SPLIT,
        "manifest_sha256": manifest["manifest_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "expected_extended1024": len(selected),
        "completed_extended1024": completed1024,
        "required_extended1536_so_far": len(hit1024_keys),
        "completed_extended1536": completed1536,
        "complete": complete,
        "raw_stage_rows": len(records),
        "elapsed_seconds_this_invocation": float(elapsed),
        "decision_computed": False,
        "decision_owned_by": "phase978_dev_admission_audit.py",
        "condition_metrics_withheld_until_complete_independent_audit": True,
        "holdout_loaded": False,
        "updated_at_utc": legal.utc_now(),
    }
    legal.atomic_write_json(RUN_SUMMARY_PATH, status)


def cpu_preflight() -> dict[str, Any]:
    protocol = validate_protocol_lock()
    source = validate_source_without_tokenizer()
    items, data_audit = load_dev_items()
    from transformers import AutoTokenizer
    from model_utils import MODEL_CONFIGS

    tok = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[MODEL_NAME]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        tokenizer_audit = validate_source_with_tokenizer(source, tok, items)
    finally:
        del tok
        gc.collect()
    assert_holdout_not_imported()
    return {
        "protocol_sha256": protocol["protocol_sha256"],
        "source_artifacts_authenticated": True,
        "source_stage_rows": len(source["ordered"]),
        "selected_hit512": len(source["selected"]),
        "selected_by_condition": dict(Counter(
            str(row["condition"]) for row in source["selected"])),
        "dataset_audit_passed": data_audit["external"].get("ok") is True,
        **tokenizer_audit,
        "model_weights_loaded": False,
        "generation_performed": False,
        "holdout_loaded": False,
    }


def run() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("formal Phase978 development run requires local CUDA")
    t0 = time.time()
    protocol = validate_protocol_lock()
    source = validate_source_without_tokenizer()
    items, data_audit = load_dev_items()
    item_by_id = {item["id"]: item for item in items}
    model = None
    try:
        model, tok, device = load_model(MODEL_NAME)
        require(getattr(device, "type", str(device).split(":")[0]) == "cuda",
                f"Qwen3 did not load on CUDA: {device}")
        tokenizer_audit = validate_source_with_tokenizer(source, tok, items)
        runtime_eos = [int(value) for value in get_eos_ids(model, tok)]
        require(runtime_eos == tokenizer_audit["eos_ids"], "runtime EOS IDs changed")
        think_open_id = legal.single_token_id(tok, "<think>")
        think_close_id = legal.single_token_id(tok, "</think>")
        manifest = make_manifest(
            protocol, source, data_audit, tokenizer_audit, model, tok)
        install_or_validate_manifest(manifest)
        records = load_output_rows(manifest["manifest_sha256"])

        selected = source["selected"]
        allowed1024 = {
            (str(row["id"]), str(row["condition"]), "extended1024")
            for row in selected
        }
        allowed1536: set[tuple[str, str, str]] = set()
        for source_row in selected:
            key1024 = (str(source_row["id"]), str(source_row["condition"]), "extended1024")
            if key1024 in records and records[key1024].get("hit1024") is True:
                allowed1536.add((key1024[0], key1024[1], "extended1536"))
        require(set(records).issubset(allowed1024 | allowed1536),
                "output contains a non-protocol stage key")

        # Validate every resumable row before generating anything new.
        selected_by_key = {
            (str(row["id"]), str(row["condition"])): row for row in selected
        }
        for key, row in records.items():
            base = selected_by_key[(key[0], key[1])]
            parent = base
            if key[2] == "extended1536":
                parent = records[(key[0], key[1], "extended1024")]
                require(parent.get("hit1024") is True,
                        f"illegal 1536 child after EOS: {key}")
            validate_extension_row(
                row, manifest, tok, item_by_id[key[0]], parent,
                runtime_eos, think_open_id, think_close_id,
            )

        total = len(selected)
        for index, source_row in enumerate(selected, 1):
            item = item_by_id[str(source_row["id"])]
            condition = str(source_row["condition"])
            seed = int(source_row["seed"])
            key1024 = (item["id"], condition, "extended1024")
            row1024 = records.get(key1024)
            if row1024 is None:
                input_ids, generated, user_prompt = legal.generate_stage(
                    model, tok, device, runtime_eos, item, condition, seed, 1024)
                row1024 = build_extension_row(
                    manifest, tok, item, condition, seed, "extended1024", 1024,
                    input_ids, generated, user_prompt, runtime_eos,
                    think_open_id, think_close_id, source_row,
                )
                validate_extension_row(
                    row1024, manifest, tok, item, source_row,
                    runtime_eos, think_open_id, think_close_id,
                )
                legal.append_jsonl(ROWS_PATH, row1024)
                records[key1024] = row1024

            if row1024["hit1024"]:
                key1536 = (item["id"], condition, "extended1536")
                if key1536 not in records:
                    input_ids, generated, user_prompt = legal.generate_stage(
                        model, tok, device, runtime_eos, item, condition, seed, 1536)
                    row1536 = build_extension_row(
                        manifest, tok, item, condition, seed, "extended1536", 1536,
                        input_ids, generated, user_prompt, runtime_eos,
                        think_open_id, think_close_id, row1024,
                    )
                    validate_extension_row(
                        row1536, manifest, tok, item, row1024,
                        runtime_eos, think_open_id, think_close_id,
                    )
                    legal.append_jsonl(ROWS_PATH, row1536)
                    records[key1536] = row1536

            write_generator_status(
                manifest, records, selected, time.time() - t0)
            if index % 8 == 0 or index == total:
                log(f"  Phase978 dev source-hit512 trajectories {index}/{total}")

        write_generator_status(manifest, records, selected, time.time() - t0)
        assert_holdout_not_imported()
        log(f"Phase978 development extensions complete; elapsed={(time.time()-t0)/60:.1f} min")
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-preflight", action="store_true",
        help="Authenticate source/protocol/tokenizer without loading weights.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cpu_preflight:
        print(json.dumps(cpu_preflight(), ensure_ascii=False, indent=2))
    else:
        run()
