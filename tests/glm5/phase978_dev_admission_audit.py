#!/usr/bin/env python3
"""Independent CPU admission audit for Phase978 development trajectories.

This script ignores the generator's status file.  It authenticates and
re-scores the frozen Phase977 256/512 rows plus every Phase978 1024/1536 raw
row, constructs full-denominator cumulative checkpoints, and applies the
preregistered development gate at 1536.  It never imports the holdout module.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

import phase978_legal_core as legal  # noqa: E402
import phase978_budget_protocol as frozen_protocol  # noqa: E402
from model_utils import MODEL_CONFIGS  # noqa: E402


PHASE = 978
SCHEMA_VERSION = 1
SPLIT = "development"
CHECKPOINTS = (256, 512, 1024, 1536)
CONDITIONS = tuple(legal.CONDITIONS)
EXPECTED_ITEMS = 64
EXPECTED_TASKS = 8
EXPECTED_SOURCE_ROWS = 408
EXPECTED_HIT512 = 112

SOURCE_DIR = ROOT / "tests" / "glm5" / "result" / "phase977_legal_mode_trajectories"
SOURCE_MANIFEST = SOURCE_DIR / "manifest_development.json"
SOURCE_ROWS = SOURCE_DIR / "rows_development.jsonl"
SOURCE_SUMMARY = SOURCE_DIR / "summary_development.json"
SOURCE_DATASET_SCRIPT = GLM5 / "phase977_dev_dataset.py"
SOURCE_LEGAL_SCRIPT = GLM5 / "phase977_legal_mode_trajectories.py"
HOLDOUT_SCRIPT = GLM5 / "phase977_holdout_dataset.py"
PROTOCOL_SCRIPT = GLM5 / "phase978_budget_protocol.py"
RUNNER_SCRIPT = GLM5 / "phase978_dev_budget_stabilization.py"

OUT = ROOT / "tests" / "glm5" / "result" / "phase978_legal_budget_stabilization"
PROTOCOL_PATH = OUT / "protocol_preregistration.json"
DEV_MANIFEST_PATH = OUT / "manifest_development.json"
DEV_ROWS_PATH = OUT / "rows_development.jsonl"
ADMISSION_PATH = OUT / "admission_development.json"

PINNED_FILES = {
    SOURCE_DATASET_SCRIPT: "ac28e7d0b1a806653564f8f9e330c59ab3134062b45d5c578a2616e2d6997399",
    HOLDOUT_SCRIPT: "d4d630f00a7c0197f6e7ba83704fdcf13121d67b5b09d3a77d649cb3fdff4755",
    SOURCE_LEGAL_SCRIPT: "9b725cbac0cb5c975c4e588ee7f6924e60004154f0ac4cf2dbdcb9aa34a28481",
    SOURCE_MANIFEST: "de25a435eee181ebaa7219c4f5d8bb722cac948695cadd78c28243b5eb77bcb0",
    SOURCE_ROWS: "8b7c9b4d2f8a1d6e8e5bf0c6a9575a8545169f6b86053a1b7fe4fa83be3fe426",
    SOURCE_SUMMARY: "48ae80112682e7f8b6dceab1202c2d7ade4b99d492deb9925432dabddc8d2968",
}
PINNED_SOURCE_MANIFEST = "378a28959759177883ef72f6ad8f4c903c21f1762f2258f57b478e5545102544"
PINNED_DATASET = "ff2703b07868f1b440b21e067385c9a3cdd56810c04b02670f607f151368e72d"

MODE_THRESHOLDS = {
    "hard_no_think": 0.95,
    "hard_thinking": 0.80,
    "soft_no_think": 0.80,
    "soft_thinking": 0.80,
}
DEV_ENDPOINT_THRESHOLDS = {
    "hard_no_think": 0.75,
    "hard_thinking": 0.50,
    "soft_no_think": 0.65,
    "soft_thinking": 0.50,
}
TASK_MODE_THRESHOLD = 0.75
TASK_ENDPOINT_THRESHOLD = 0.25
TASK_COVERAGE_MIN = 6
HIT_CAP_OVERALL_MAX = 0.10
HIT_CAP_TASK_MAX = 0.25


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def without_fields(value: dict[str, Any], *excluded: str) -> dict[str, Any]:
    return {
        key: item for key, item in value.items()
        if key not in set(excluded)
    }


def assert_no_holdout_import() -> None:
    require("phase977_holdout_dataset" not in sys.modules,
            "holdout module was imported by the development auditor")


def read_jsonl_strict(
    path: Path, label: str, expected_manifest: str,
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    require(path.is_file(), f"missing {label}: {path}")
    payload = path.read_bytes()
    require(payload.endswith(b"\n"), f"{label} lacks a final newline")
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.splitlines(), 1):
        require(bool(raw.strip()), f"{label} line {line_number} is blank")
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{label} line {line_number} malformed") from exc
        require(isinstance(row, dict), f"{label} line {line_number} not object")
        require(row.get("manifest_sha256") == expected_manifest,
                f"{label} line {line_number} manifest mismatch")
        key = legal.row_key(row)
        require(key not in records, f"{label} duplicate key {key}")
        records[key] = row
        ordered.append(row)
    return records, ordered


def load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
        local_files_only=True, use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def authenticate_protocol() -> dict[str, Any]:
    protocol = load_json(PROTOCOL_PATH, "Phase978 preregistration")
    core = without_fields(protocol, "protocol_sha256", "created_at_utc")
    require(protocol.get("protocol_sha256") == legal.sha256_json(core),
            "protocol self-hash invalid")
    require(protocol.get("phase") == PHASE, "protocol phase mismatch")
    require(protocol.get("checkpoints") == list(CHECKPOINTS), "protocol checkpoints changed")
    require(protocol.get("decision_checkpoint") == 1536, "protocol endpoint changed")
    execution = protocol.get("execution_contract", {})
    require(execution.get("holdout_module_imported") is False and
            execution.get("holdout_module_parsed") is False,
            "protocol freeze accessed holdout")
    script_hashes = protocol.get("phase978_script_hashes", {})
    for value in script_hashes.values():
        require(isinstance(value, dict) and "path" in value and "sha256" in value,
                "invalid Phase978 script commitment")
        script_path = ROOT / str(value["path"])
        require(script_path.is_file() and sha256_file(script_path) == value["sha256"],
                f"Phase978 script changed after freeze: {value.get('path')}")
    code_hashes = {
        Path(value["path"]).name: value["sha256"]
        for value in script_hashes.values()
        if isinstance(value, dict) and "path" in value and "sha256" in value
    }
    for path in (PROTOCOL_SCRIPT, RUNNER_SCRIPT, Path(__file__)):
        require(code_hashes.get(path.name) == sha256_file(path),
                f"code differs from protocol lock: {path.name}")
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
    frozen_model_root = ROOT / str(model_identity.get("path", ""))
    require(Path(MODEL_CONFIGS["qwen3"]["path"]).resolve() ==
            frozen_model_root.resolve(),
            "auditor tokenizer registry path differs from frozen model path")
    model_files = model_identity.get("files", {})
    require(frozen_model_root.is_dir() and isinstance(model_files, dict) and model_files,
            "protocol lacks frozen tokenizer/model files")
    for name, expected in model_files.items():
        path = frozen_model_root / name
        require(path.is_file() and path.stat().st_size == expected.get("bytes") and
                sha256_file(path) == expected.get("sha256"),
                f"auditor tokenizer/model artifact changed after freeze: {name}")
    gate = protocol.get("gate", {})
    require(gate.get("mode_valid_thresholds", {}).get(SPLIT) == MODE_THRESHOLDS,
            "protocol mode-valid thresholds differ")
    require(gate.get("valid_mode_eos_thresholds", {}).get(SPLIT) ==
            DEV_ENDPOINT_THRESHOLDS,
            "protocol development endpoint thresholds differ")
    task_gate = gate.get("task_qualification", {})
    budget_gate = gate.get("budget_stability_at_1536", {})
    require(task_gate.get("mode_valid_rate_min") == TASK_MODE_THRESHOLD and
            task_gate.get("valid_mode_eos_rate_min") == TASK_ENDPOINT_THRESHOLD and
            task_gate.get("qualified_tasks_required_per_condition") == TASK_COVERAGE_MIN and
            budget_gate.get("overall_max_per_condition") == HIT_CAP_OVERALL_MAX and
            budget_gate.get("any_task_max") == HIT_CAP_TASK_MAX,
            "protocol task/budget thresholds differ")
    return protocol


def authenticate_source(tok) -> tuple[
    dict[str, Any], dict[tuple[str, str, str], dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    hashes_before: dict[str, str] = {}
    for path, expected in PINNED_FILES.items():
        require(path.is_file(), f"missing frozen source {path}")
        current = sha256_file(path)
        require(current == expected, f"frozen source changed: {path.name}")
        hashes_before[path.name] = current

    manifest = load_json(SOURCE_MANIFEST, "Phase977 dev manifest")
    require(manifest.get("manifest_sha256") == PINNED_SOURCE_MANIFEST,
            "Phase977 dev manifest identity changed")
    require(manifest.get("manifest_sha256") == legal.sha256_json(
        without_fields(manifest, "manifest_sha256", "created_at_utc")),
        "Phase977 dev manifest self-hash invalid")
    require(manifest.get("schema_version") == 2 and manifest.get("phase") == 977 and
            manifest.get("split") == SPLIT and manifest.get("n_items") == EXPECTED_ITEMS,
            "Phase977 dev manifest protocol mismatch")
    require(manifest.get("dataset_sha256") == PINNED_DATASET,
            "Phase977 dev dataset hash mismatch")
    require(manifest.get("conditions") == legal.CONDITIONS and
            manifest.get("generated_mode_parser_version") == "strict_final_region_v2",
            "Phase977 conditions/parser mismatch")

    dataset_module = importlib.import_module("phase977_dev_dataset")
    external_audit = dataset_module.audit_dataset()
    items = [legal.normalize_item(item) for item in dataset_module.build_dataset()]
    data_audit = {
        "external": external_audit,
        "local": legal.audit_local_items(items),
        "cross_set_overlap_contract": (
            "authenticated from the frozen Phase977 development manifest; "
            "historical discovery builder is not imported"
        ),
    }
    require(len(items) == EXPECTED_ITEMS and legal.dataset_hash(items) == PINNED_DATASET,
            "runtime dev data mismatch")
    require(data_audit["external"].get("ok") is True, "runtime dev audit failed")
    assert_no_holdout_import()
    item_by_id = {item["id"]: item for item in items}
    require(len(item_by_id) == EXPECTED_ITEMS, "duplicate runtime dev item")

    records, ordered = read_jsonl_strict(
        SOURCE_ROWS, "Phase977 source rows", manifest["manifest_sha256"])
    require(len(ordered) == EXPECTED_SOURCE_ROWS, "source row count mismatch")
    eos_ids = [int(value) for value in manifest["eos_token_ids"]]
    special = manifest["special_token_ids"]
    require(legal.single_token_id(tok, "<think>") == special["think_open"] and
            legal.single_token_id(tok, "</think>") == special["think_close"],
            "runtime think token mismatch")
    require(legal.build_template_tokens(
        tok, items[0], special["think_open"], special["think_close"]
    ) == manifest["template_tokens"], "runtime official template mismatch")

    expected_keys: set[tuple[str, str, str]] = set()
    for item in items:
        for condition in CONDITIONS:
            first_key = (item["id"], condition, "initial256")
            expected_keys.add(first_key)
            require(first_key in records, f"missing source initial {first_key}")
            if records[first_key].get("has_eos") is False:
                expected_keys.add((item["id"], condition, "extended512"))
    require(set(records) == expected_keys, "source stage key set mismatch")

    for key, row in records.items():
        item_id, condition, stage = key
        require(item_id in item_by_id and condition in CONDITIONS and
                stage in {"initial256", "extended512"}, f"unknown source key {key}")
        item = item_by_id[item_id]
        budget = 256 if stage == "initial256" else 512
        for field in ("id", "task", "prompt", "answer", "alias_groups", "exact"):
            require(row.get(field) == item[field], f"source {field} mismatch {key}")
        require(row.get("schema_version") == 2 and row.get("phase") == 977 and
                row.get("split") == SPLIT and row.get("max_new_tokens") == budget,
                f"source metadata mismatch {key}")
        require(row.get("seed") == legal.stable_item_seed(
            manifest["base_seed"], SPLIT, item_id), f"source seed mismatch {key}")
        user_prompt, _rendered, input_ids = legal.render_prefix(tok, item, condition)
        require(row.get("effective_user_prompt") == user_prompt and
                row.get("input_ids") == input_ids and row.get("prompt_len") == len(input_ids),
                f"source prefix mismatch {key}")
        expected_sampling = {name: legal.CONDITIONS[condition][name]
                             for name in ("temperature", "top_p", "top_k", "min_p")}
        require(row.get("sampling") == expected_sampling, f"source sampling mismatch {key}")
        ids = row.get("generated_ids")
        require(isinstance(ids, list) and ids and
                all(isinstance(value, int) and not isinstance(value, bool) for value in ids),
                f"invalid source IDs {key}")
        rescored = legal.analyze_generation(
            tok, item, condition, ids, eos_ids,
            special["think_open"], special["think_close"], budget,
        )
        for field, value in rescored.items():
            require(row.get(field) == value, f"source derived {field} mismatch {key}")
        eos = rescored["eos_positions"]
        require((len(eos) == 1 and eos[0] == len(ids) - 1) or
                (not eos and len(ids) == budget and rescored["hit_budget"]),
                f"source termination mismatch {key}")
        if stage == "initial256":
            require(row.get("hit256") == rescored["hit_budget"],
                    f"source hit256 mismatch {key}")
        else:
            parent = records[(item_id, condition, "initial256")]
            require(parent["has_eos"] is False and ids[:256] == parent["generated_ids"] and
                    row.get("extension_replayed_initial256_exact") is True and
                    row.get("hit512") == rescored["hit_budget"],
                    f"source 256 replay mismatch {key}")

    selected = sorted(
        (row for row in records.values()
         if row["stage"] == "extended512" and row.get("hit512") is True),
        key=lambda row: (str(row["id"]), str(row["condition"])),
    )
    require(len(selected) == EXPECTED_HIT512, "source hit512 count mismatch")
    require(Counter(str(row["condition"]) for row in selected) == Counter({
        "hard_no_think": 3, "hard_thinking": 55,
        "soft_no_think": 2, "soft_thinking": 52,
    }), "source hit512 condition balance mismatch")
    for path, expected in PINNED_FILES.items():
        require(sha256_file(path) == hashes_before[path.name] == expected,
                f"source changed during independent audit: {path.name}")
    return manifest, records, selected, items, {
        "source_file_sha256": hashes_before,
        "dataset_audit": data_audit,
        "all_408_source_rows_recomputed": True,
        "all_source_terminations_valid": True,
        "all_152_source_replays_exact": True,
        "all_source_256_replays_exact": True,
    }


def authenticate_extensions(
    protocol: dict[str, Any], tok, source_manifest: dict[str, Any],
    source_records: dict[tuple[str, str, str], dict[str, Any]],
    selected: list[dict[str, Any]], items: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    manifest = load_json(DEV_MANIFEST_PATH, "Phase978 dev manifest")
    core = without_fields(manifest, "manifest_sha256", "created_at_utc")
    require(manifest.get("manifest_sha256") == legal.sha256_json(core),
            "Phase978 dev manifest self-hash invalid")
    require(manifest.get("phase") == PHASE and manifest.get("split") == SPLIT and
            manifest.get("protocol_sha256") == protocol["protocol_sha256"] and
            manifest.get("runner_sha256") == sha256_file(RUNNER_SCRIPT),
            "Phase978 dev manifest protocol mismatch")
    require(manifest.get("source_manifest_sha256") == source_manifest["manifest_sha256"] and
            manifest.get("selected_hit512_trajectories") == EXPECTED_HIT512 and
            manifest.get("checkpoints") == list(CHECKPOINTS) and
            manifest.get("decision_checkpoint") == 1536 and
            manifest.get("generated_mode_parser_version") == "strict_final_region_v2" and
            manifest.get("holdout_module_imported") is False,
            "Phase978 dev manifest selection/rule mismatch")
    records, ordered = read_jsonl_strict(
        DEV_ROWS_PATH, "Phase978 dev extensions", manifest["manifest_sha256"])
    item_by_id = {item["id"]: item for item in items}
    selected_keys = {(str(row["id"]), str(row["condition"])) for row in selected}
    eos_ids = [int(value) for value in source_manifest["eos_token_ids"]]
    special = source_manifest["special_token_ids"]
    expected1024 = {(item_id, condition, "extended1024")
                    for item_id, condition in selected_keys}
    require(expected1024.issubset(records), "Phase978 dev is missing required 1024 rows")
    expected1536 = {
        (item_id, condition, "extended1536")
        for item_id, condition, _stage in expected1024
        if records[(item_id, condition, "extended1024")].get("hit1024") is True
    }
    require(set(records) == expected1024 | expected1536,
            "Phase978 dev extension key set is not exact/complete")

    replay_checks: list[bool] = []
    for key, row in records.items():
        item_id, condition, stage = key
        require((item_id, condition) in selected_keys and item_id in item_by_id,
                f"unexpected Phase978 extension key {key}")
        item = item_by_id[item_id]
        budget = 1024 if stage == "extended1024" else 1536
        require(stage in {"extended1024", "extended1536"}, f"invalid stage {key}")
        parent = (source_records[(item_id, condition, "extended512")]
                  if budget == 1024 else records[(item_id, condition, "extended1024")])
        require(parent.get("hit_budget") is True,
                f"illegal extension after an EOS endpoint {key}")
        ids = row.get("generated_ids")
        require(isinstance(ids, list) and ids and
                all(isinstance(value, int) and not isinstance(value, bool) for value in ids),
                f"invalid Phase978 IDs {key}")
        parent_ids = [int(value) for value in parent["generated_ids"]]
        parent_budget = int(parent["max_new_tokens"])
        item_seed = legal.stable_item_seed(source_manifest["base_seed"], SPLIT, item_id)
        user_prompt, _rendered, input_ids = legal.render_prefix(tok, item, condition)
        expected_sampling = {name: legal.CONDITIONS[condition][name]
                             for name in ("temperature", "top_p", "top_k", "min_p")}
        require(row.get("schema_version") == SCHEMA_VERSION and row.get("phase") == PHASE and
                row.get("protocol_sha256") == protocol["protocol_sha256"] and
                row.get("split") == SPLIT and row.get("id") == item_id and
                row.get("condition") == condition and row.get("task") == item["task"] and
                row.get("seed") == item_seed and row.get("max_new_tokens") == budget and
                row.get("prompt") == item["prompt"] and
                row.get("effective_user_prompt") == user_prompt and
                row.get("answer") == item["answer"] and
                row.get("alias_groups") == item["alias_groups"] and
                row.get("exact") == item["exact"] and
                row.get("sampling") == expected_sampling and
                row.get("input_ids") == input_ids and row.get("prompt_len") == len(input_ids) and
                row.get("source_stage") == parent["stage"] and
                row.get("source_budget") == parent_budget and
                row.get("source_generated_ids_sha256") == legal.sha256_json(parent_ids) and
                row.get("extension_strategy") ==
                    "full_rerun_original_official_prefix_same_seed" and
                row.get("replay_prior_checkpoint_exact") is True and
                row.get("natural_endpoint_is_actual_sampled_eos") is True and
                row.get("holdout_module_imported") is False,
                f"Phase978 extension metadata mismatch {key}")
        exact_replay = ids[:parent_budget] == parent_ids
        replay_checks.append(exact_replay)
        require(exact_replay, f"Phase978 exact replay failed {key}")
        rescored = legal.analyze_generation(
            tok, item, condition, ids, eos_ids,
            special["think_open"], special["think_close"], budget,
        )
        for field, value in rescored.items():
            require(row.get(field) == value, f"Phase978 derived {field} mismatch {key}")
        eos = rescored["eos_positions"]
        require((len(eos) == 1 and eos[0] == len(ids) - 1 and
                 parent_budget < len(ids) <= budget) or
                (not eos and len(ids) == budget and rescored["hit_budget"]),
                f"Phase978 termination mismatch {key}")
        require(row.get("hit1024") == (rescored["hit_budget"] if budget == 1024 else True) and
                row.get("hit1536") == (rescored["hit_budget"] if budget == 1536 else None),
                f"Phase978 hit checkpoint flag mismatch {key}")

    require(all(replay_checks), "not all Phase978 replay checks passed")
    return manifest, records, {
        "extension_rows": len(ordered),
        "extended1024_rows": len(expected1024),
        "extended1536_rows": len(expected1536),
        "all_extension_keys_exact": True,
        "all_extension_rows_recomputed": True,
        "all_extension_terminations_valid": True,
        "all_extension_replays_exact": True,
        "rows_sha256": sha256_file(DEV_ROWS_PATH),
        "manifest_file_sha256": sha256_file(DEV_MANIFEST_PATH),
    }


def resolve_endpoint(
    source: dict[tuple[str, str, str], dict[str, Any]],
    extended: dict[tuple[str, str, str], dict[str, Any]],
    item_id: str, condition: str, checkpoint: int,
) -> dict[str, Any]:
    row = source[(item_id, condition, "initial256")]
    if checkpoint == 256 or row["has_eos"]:
        return row
    row = source[(item_id, condition, "extended512")]
    if checkpoint == 512 or row["has_eos"]:
        return row
    row = extended[(item_id, condition, "extended1024")]
    if checkpoint == 1024 or row["has_eos"]:
        return row
    return extended[(item_id, condition, "extended1536")]


def metric_block(rows: list[dict[str, Any]], checkpoint: int) -> dict[str, Any]:
    require(bool(rows), "cannot summarize an empty row set")
    n = len(rows)
    bool_keys = (
        "semantic_match", "mode_valid", "has_eos", "valid_eos",
        "valid_mode_eos", "think_well_formed", "final_region_valid",
    )
    out: dict[str, Any] = {"n": n, "checkpoint": checkpoint}
    for key in bool_keys:
        count = sum(bool(row.get(key, False)) for row in rows)
        out[f"{key}_n"] = count
        out[f"{key}_rate"] = count / n
    hit = sum(not bool(row.get("has_eos")) and
              int(row.get("n_tokens", 0)) == checkpoint for row in rows)
    out["hit_cap_n"] = hit
    out["hit_cap_rate"] = hit / n
    # Source and Phase978 extension validators fail closed on any mismatch.
    # Non-extended trajectories are vacuously exact, so this is a full-split
    # trajectory denominator rather than a survivor/extension denominator.
    out["extension_replay_exact_n"] = n
    out["extension_replay_exact_rate"] = 1.0
    if checkpoint == 1536:
        out["hit1536_rate"] = hit / n
    steps = [int(row["first_eos_step"]) for row in rows
             if row.get("first_eos_step") is not None]
    out["mean_first_eos_step_among_eos"] = (sum(steps) / len(steps) if steps else None)
    return out


def build_checkpoints(
    source: dict[tuple[str, str, str], dict[str, Any]],
    extended: dict[tuple[str, str, str], dict[str, Any]],
    items: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoints: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        condition_blocks: dict[str, Any] = {}
        for condition in CONDITIONS:
            rows = [resolve_endpoint(source, extended, item["id"], condition, checkpoint)
                    for item in items]
            by_task = {
                task: metric_block([row for row in rows if row["task"] == task], checkpoint)
                for task in sorted({str(row["task"]) for row in rows})
            }
            require(len(rows) == EXPECTED_ITEMS and len(by_task) == EXPECTED_TASKS,
                    f"checkpoint denominator mismatch {checkpoint}/{condition}")
            condition_blocks[condition] = {
                "overall": metric_block(rows, checkpoint),
                "by_task": by_task,
            }
        checkpoints[str(checkpoint)] = {
            "full_denominator_per_condition": EXPECTED_ITEMS,
            "early_eos_is_absorbing": True,
            "conditions": condition_blocks,
        }

    intervals: dict[str, Any] = {}
    for condition in CONDITIONS:
        counts = Counter()
        for item in items:
            final = resolve_endpoint(source, extended, item["id"], condition, 1536)
            step = final.get("first_eos_step")
            if step is None:
                counts[">1536_censored"] += 1
            elif step <= 256:
                counts["<=256"] += 1
            elif step <= 512:
                counts["257-512"] += 1
            elif step <= 1024:
                counts["513-1024"] += 1
            elif step <= 1536:
                counts["1025-1536"] += 1
            else:
                raise RuntimeError(f"EOS step exceeds final cap: {step}")
        intervals[condition] = {
            key: counts[key] for key in
            ("<=256", "257-512", "513-1024", "1025-1536", ">1536_censored")
        }
        require(sum(intervals[condition].values()) == EXPECTED_ITEMS,
                f"EOS interval denominator mismatch {condition}")
    return checkpoints, intervals


def evaluate_development_gate(
    checkpoint1536: dict[str, Any], all_replays_exact: bool,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for condition in CONDITIONS:
        block = checkpoint1536["conditions"][condition]
        overall = block["overall"]
        by_task = block["by_task"]
        task_coverage = sum(
            values["mode_valid_rate"] >= TASK_MODE_THRESHOLD
            and values["valid_mode_eos_rate"] >= TASK_ENDPOINT_THRESHOLD
            for values in by_task.values()
        )
        max_task_hit = max(values["hit_cap_rate"] for values in by_task.values())
        checks[condition] = {
            "mode_valid_rate": overall["mode_valid_rate"],
            "mode_valid_threshold": MODE_THRESHOLDS[condition],
            "mode_valid_passed": overall["mode_valid_rate"] >= MODE_THRESHOLDS[condition],
            "valid_mode_eos_rate": overall["valid_mode_eos_rate"],
            "valid_mode_eos_threshold": DEV_ENDPOINT_THRESHOLDS[condition],
            "valid_mode_eos_passed": (
                overall["valid_mode_eos_rate"] >= DEV_ENDPOINT_THRESHOLDS[condition]),
            "task_coverage_n": task_coverage,
            "task_coverage_min": TASK_COVERAGE_MIN,
            "task_coverage_passed": task_coverage >= TASK_COVERAGE_MIN,
            "hit1536_rate": overall["hit_cap_rate"],
            "hit1536_overall_max": HIT_CAP_OVERALL_MAX,
            "hit1536_overall_passed": overall["hit_cap_rate"] <= HIT_CAP_OVERALL_MAX,
            "max_task_hit1536_rate": max_task_hit,
            "hit1536_per_task_max": HIT_CAP_TASK_MAX,
            "hit1536_per_task_passed": max_task_hit <= HIT_CAP_TASK_MAX,
            "all_replays_exact": all_replays_exact,
        }
    keys = (
        "mode_valid_passed", "valid_mode_eos_passed", "task_coverage_passed",
        "hit1536_overall_passed", "hit1536_per_task_passed", "all_replays_exact",
    )
    passed = all(all(bool(block[key]) for key in keys) for block in checks.values())
    return {
        "passed": passed,
        "decision_checkpoint": 1536,
        "all_four_conditions_required": True,
        "condition_checks": checks,
        "rule": (
            "frozen official modes; dev mode-valid and valid-mode-EOS thresholds; "
            ">=6/8 task coverage; hit1536 overall<=0.10 and every task<=0.25; "
            "all staged prefix replays exact"
        ),
    }


def install_admission(report: dict[str, Any]) -> None:
    if ADMISSION_PATH.exists():
        prior = load_json(ADMISSION_PATH, "existing Phase978 admission")
        require(prior.get("admission_sha256") == legal.sha256_json(
            without_fields(prior, "admission_sha256", "audited_at_utc")),
            "existing admission self-hash invalid")
        require(prior.get("admission_sha256") == report["admission_sha256"],
                "existing admission differs; refusing to overwrite")
        return
    legal.atomic_write_json(ADMISSION_PATH, report)


def audit(write: bool) -> dict[str, Any]:
    assert_no_holdout_import()
    protocol = authenticate_protocol()
    tok = load_tokenizer()
    try:
        source_manifest, source_records, selected, items, source_audit = (
            authenticate_source(tok))
        dev_manifest, extensions, extension_audit = authenticate_extensions(
            protocol, tok, source_manifest, source_records, selected, items)
        checkpoints, intervals = build_checkpoints(source_records, extensions, items)
        all_replays_exact = (
            extension_audit["all_extension_replays_exact"]
            and source_audit["all_source_256_replays_exact"]
        )
        independent_gate = evaluate_development_gate(
            checkpoints["1536"], extension_audit["all_extension_replays_exact"]
            and source_audit["all_source_256_replays_exact"])
        canonical_summaries = checkpoints["1536"]["conditions"]
        for block in canonical_summaries.values():
            if not all_replays_exact:
                block["overall"]["extension_replay_exact_n"] = 0
                block["overall"]["extension_replay_exact_rate"] = 0.0
        gate = frozen_protocol.evaluate_gate(
            split=SPLIT,
            checkpoint=1536,
            complete=True,
            condition_summaries=canonical_summaries,
        )
        require(gate["passed"] == independent_gate["passed"],
                "independent and canonical development gates disagree")
        for condition in CONDITIONS:
            require(
                gate["condition_checks"][condition]["passed"] == all((
                    independent_gate["condition_checks"][condition]["mode_valid_passed"],
                    independent_gate["condition_checks"][condition]["valid_mode_eos_passed"],
                    independent_gate["condition_checks"][condition]["task_coverage_passed"],
                    independent_gate["condition_checks"][condition]["hit1536_overall_passed"],
                    independent_gate["condition_checks"][condition]["hit1536_per_task_passed"],
                    independent_gate["condition_checks"][condition]["all_replays_exact"],
                )),
                f"independent/canonical condition gate disagreement: {condition}",
            )
    finally:
        del tok
        gc.collect()
    assert_no_holdout_import()

    core = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": "qwen3_official_budget_stabilization_development_admission",
        "split": SPLIT,
        "protocol_sha256": protocol["protocol_sha256"],
        "auditor_sha256": sha256_file(Path(__file__)),
        "development_manifest_sha256": dev_manifest["manifest_sha256"],
        "development_manifest_file_sha256": sha256_file(DEV_MANIFEST_PATH),
        "development_rows_sha256": sha256_file(DEV_ROWS_PATH),
        "source_audit": source_audit,
        "extension_audit": extension_audit,
        "checkpoints": checkpoints,
        "eos_time_intervals": intervals,
        "decision_gate": gate,
        "independent_gate_crosscheck": independent_gate,
        "holdout_authorized": bool(gate["passed"]),
        "holdout_opening_rule": (
            "only a self-hashed PASS from this frozen independent CPU auditor "
            "may authorize one atomic holdout OPEN receipt"
        ),
        "holdout_loaded": False,
        "model_weights_loaded": False,
        "natural_endpoint_is_actual_sampled_eos": True,
        "greedy_or_logit_gap_used_for_gate": False,
        "wrong_answer_auxiliary_used_for_gate": False,
        "mechanism_authorized": False,
    }
    report = {
        **core,
        "admission_sha256": legal.sha256_json(core),
        "audited_at_utc": legal.utc_now(),
    }
    if write:
        install_admission(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="Atomically install the immutable admission artifact.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = audit(args.write)
    print(json.dumps({
        "admission_sha256": result["admission_sha256"],
        "passed": result["decision_gate"]["passed"],
        "holdout_authorized": result["holdout_authorized"],
        "development_rows_sha256": result["development_rows_sha256"],
        "holdout_loaded": result["holdout_loaded"],
    }, ensure_ascii=False, indent=2))
