#!/usr/bin/env python3
"""Independent static/freeze and execution audit for Phase579 residual traces.

The execution audit intentionally does not import the primary inventory.  It
reopens every trace shard on CPU, independently verifies token replay and
tensor closure, and recomputes the integer observer-candidate evidence.
"""

from __future__ import annotations

import argparse
import ast
import gc
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase579"
MODELS = ("qwen3", "glm4")
MODEL_DIRS = {"qwen3": "00_qwen3", "glm4": "01_glm4"}
CASE_COUNT = 336
MAX_NEW_TOKENS = 24
FEEDBACK_BUDGET = MAX_NEW_TOKENS - 1
PREFILL_ROLES = ("focus", "comparison", "query_anchor", "answer_boundary")
AXES = ("relation", "query_polarity", "selection_order", "output_contract", "paraphrase")
AXIS_VARYING_FIELD = {
    "relation": "relation", "query_polarity": "query_polarity",
    "selection_order": "order", "output_contract": "output_contract",
    "paraphrase": "paraphrase_id",
}
PAIR_SEMANTIC_FIELDS = (
    "analysis_unit_id", "candidate_groups", "comparison_object",
    "comparison_object_class", "focus_object", "focus_object_class", "foil",
    "interface", "left_option", "negative_object", "order", "output_contract",
    "paraphrase_id", "positive_object", "query_polarity", "raw_prompt",
    "relation", "relation_contract_id", "right_option", "surface_id", "target",
    "target_truth_polarity",
)
EXPECTED_AXIS_PAIR_COUNTS = {
    "relation": 36, "query_polarity": 96, "selection_order": 96,
    "output_contract": 168, "paraphrase": 240,
}

PROTOCOL_DIR = ROOT / "tests/glm5/result/phase579_gpt5_residual_protocol"
MANIFEST_PATH = PROTOCOL_DIR / "phase579_development_residual_manifest.jsonl"
PROTOCOL_PATH = PROTOCOL_DIR / "phase579_preregistered_residual_protocol.json"
SELF_TEST_PATH = PROTOCOL_DIR / "phase579_protocol_self_test.json"
STAGE_PATH = PROTOCOL_DIR / "phase579_stage_commit.json"
FREEZE_AUDIT_PATH = PROTOCOL_DIR / "phase579_independent_freeze_audit.json"
FREEZE_PATH = PROTOCOL_DIR / "phase579_freeze_commit.json"
ENGINEERING_DIR = ROOT / "tests/glm5/result/phase579_gpt5_residual_engineering"
TRACE_DIR = ROOT / "tests/glm5/result/phase579_gpt5_residual_trace"
INVENTORY_DIR = ROOT / "tests/glm5/result/phase579_gpt5_residual_inventory"
EXECUTION_AUDIT_DIR = ROOT / "tests/glm5/result/phase579_gpt5_residual_independent_audit"
TEMP_ROOT = ROOT / "tests/glm5_temp"

RUNNER_PATH = ROOT / "tests/glm5/phase579_gpt5_residual_runner.py"
INVENTORY_PATH = ROOT / "tests/glm5/phase579_gpt5_residual_inventory.py"
PROTOCOL_SOURCE_PATH = ROOT / "tests/glm5/phase579_gpt5_residual_protocol.py"
AUDIT_SOURCE_PATH = Path(__file__).resolve()
SOURCE_RELATIVES = {
    "tests/glm5/phase579_gpt5_residual_protocol.py": PROTOCOL_SOURCE_PATH,
    "tests/glm5/phase579_gpt5_residual_runner.py": RUNNER_PATH,
    "tests/glm5/phase579_gpt5_residual_inventory.py": INVENTORY_PATH,
    "tests/glm5/phase579_gpt5_residual_audit.py": AUDIT_SOURCE_PATH,
}
FUTURE_ROOTS = (ENGINEERING_DIR, TRACE_DIR, INVENTORY_DIR, EXECUTION_AUDIT_DIR)

SUMMARY_NAME = "phase579_residual_inventory_summary.json"
RECEIPT_NAME = "phase579_inventory_receipt.json"
MODEL_NAMES = {
    "qwen3": "phase579_qwen3_residual_inventory.json",
    "glm4": "phase579_glm4_residual_inventory.json",
}
EXECUTION_AUDIT_NAME = "phase579_residual_execution_independent_audit.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids_sha256(values: Sequence[int]) -> str:
    return sha256_bytes(canonical_json([int(value) for value in values]).encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"JSONL object required: {path}")
                rows.append(value)
    return rows


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"gzip JSONL object rows required: {path}")
    return rows


def regular(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def require(checks: Mapping[str, Any], label: str) -> None:
    failed = [name for name, value in checks.items() if value is not True]
    if failed:
        raise RuntimeError(f"{label} failed: {failed}")


def self_hashed(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    return result


def verify_self_hash(payload: Mapping[str, Any], field: str) -> bool:
    core = dict(payload)
    claimed = core.pop(field, None)
    return isinstance(claimed, str) and claimed == sha256_bytes(canonical_json(core).encode("utf-8"))


def exclusive_write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise RuntimeError(f"no-overwrite audit publication refused: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any) -> None:
    exclusive_write(path, json_bytes(value))


def case_value(row: Mapping[str, Any], name: str, default: Any = None) -> Any:
    if name in row:
        return row[name]
    nested = row.get("source_case_metadata")
    return nested.get(name, default) if isinstance(nested, Mapping) else default


class SourceFacts(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.true_keywords: list[str] = []
        self.assignments: dict[str, list[Any]] = defaultdict(list)
        self.imports: list[str] = []

    @staticmethod
    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            left = SourceFacts.dotted(node.value)
            return f"{left}.{node.attr}" if left else node.attr
        return ""

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(self.dotted(node.func))
        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                self.true_keywords.append(keyword.arg or "**")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            value = None
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assignments[target.id].append(value)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.extend(alias.name for alias in node.names)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)


def parse_source(path: Path) -> tuple[str, SourceFacts]:
    if not regular(path):
        raise RuntimeError(f"source missing or aliased: {path}")
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    facts = SourceFacts(); facts.visit(tree)
    return text, facts


def source_identity_checks(protocol: Mapping[str, Any] | None) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    frozen = protocol.get("source_identities", {}) if isinstance(protocol, Mapping) else {}
    for name, path in SOURCE_RELATIVES.items():
        checks[f"source_regular::{name}"] = regular(path)
        identity = frozen.get(name, {}) if isinstance(frozen, Mapping) else {}
        checks[f"source_frozen::{name}"] = (
            isinstance(identity, Mapping)
            and identity.get("sha256") == sha256_file(path)
            and identity.get("size_bytes") == path.stat().st_size
        )
    return checks


def static_source_checks(protocol: Mapping[str, Any]) -> dict[str, bool]:
    runner_text, runner = parse_source(RUNNER_PATH)
    inventory_text, inventory = parse_source(INVENTORY_PATH)
    protocol_text, protocol_facts = parse_source(PROTOCOL_SOURCE_PATH)
    audit_text, audit = parse_source(AUDIT_SOURCE_PATH)
    all_calls = runner.calls + inventory.calls + protocol_facts.calls + audit.calls
    hook_calls = [name for name in all_calls if name.endswith((
        "register_forward_hook", "register_forward_pre_hook", "register_full_backward_hook",
        "register_backward_hook",
    ))]
    intervention_markers = (
        "phase579_true_edge_closure", "activation_patch", "causal_patch",
        "candidate_neuron_indices", "preselected_layers", "selected_heads",
    )
    return {
        "all_sources_ast_parse": True,
        "no_hook_registration_calls": not hook_calls,
        "no_attention_request_true": "output_attentions" not in runner.true_keywords,
        "runner_requests_full_hidden_states": "output_hidden_states" in runner_text
        and "prefill_residual" in runner_text and "feedback_residual" in runner_text,
        "runner_publishes_all_required_tensors": all(name in runner_text for name in (
            "prompt_mask", "feedback_executed_mask", "feedback_pre_eos_mask",
        )),
        "no_legacy_or_intervention_markers": not any(
            marker in runner_text or marker in inventory_text for marker in intervention_markers
        ),
        "inventory_models_are_only_eligible_models": inventory.assignments.get("MODELS") == [("qwen3", "glm4")],
        "inventory_has_no_model_from_pretrained": not any(
            name.endswith("from_pretrained") for name in inventory.calls
        ),
        "inventory_hides_cuda": 'os.environ["CUDA_VISIBLE_DEVICES"] = ""' in inventory_text,
        "inventory_has_all_control_axes": all(axis in inventory_text for axis in AXES),
        "inventory_formula_registry_empty": '"candidate_mechanism_formulas": []' in inventory_text,
        "audit_does_not_import_inventory": "phase579_gpt5_residual_inventory" not in audit.imports,
        "protocol_no_initial_coordinates": protocol.get("candidate_coordinates") == [],
        "protocol_no_initial_formulas": protocol.get("candidate_mechanism_formulas") == [],
        "protocol_cross_model_forbidden": protocol.get("cross_model_internal_comparison_authorized") is False,
    }


def freeze_artifact_core() -> dict[str, Any]:
    for path in (PROTOCOL_PATH, MANIFEST_PATH, SELF_TEST_PATH, STAGE_PATH):
        if not regular(path):
            raise RuntimeError(f"freeze input missing or aliased: {path}")
    protocol, stage, self_test = read_json(PROTOCOL_PATH), read_json(STAGE_PATH), read_json(SELF_TEST_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    checks: dict[str, bool] = {
        "protocol_schema": protocol.get("schema_version") == "phase579_preregistered_residual_protocol.v1",
        "protocol_phase": protocol.get("phase_id") == PHASE,
        "manifest_count": len(manifest) == CASE_COUNT,
        "manifest_case_unique": len({case_value(row, "case_id") for row in manifest}) == CASE_COUNT,
        "manifest_development_only": all(case_value(row, "split") == "development" for row in manifest),
        "manifest_hash": protocol.get("development_residual_manifest", {}).get("sha256") == sha256_file(MANIFEST_PATH),
        "eligible_models": protocol.get("future_single_model_trace_eligible_models") == list(MODELS),
        "blocked_deepseek": protocol.get("behavior_blocked_models") == ["deepseek7b"],
        "stage_complete": stage.get("stage_complete") is True,
        "stage_historical_absence": stage.get("future_result_roots_absent_before_freeze") is True,
        "self_test_passed": self_test.get("passed") is True and all(self_test.get("checks", {}).values()),
        "no_model_modules_loaded": "torch" not in sys.modules and "transformers" not in sys.modules,
    }
    checks.update(source_identity_checks(protocol))
    checks.update(static_source_checks(protocol))
    for row in manifest:
        if any(case_value(row, name) is not None for name in (
            "candidate_layer", "candidate_neuron", "candidate_direction",
        )):
            checks["manifest_no_preselected_coordinates"] = False
            break
    else:
        checks["manifest_no_preselected_coordinates"] = True
    return {
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "self_test_sha256": sha256_file(SELF_TEST_PATH),
        "stage_commit_sha256": sha256_file(STAGE_PATH),
        "source_sha256": {name: sha256_file(path) for name, path in SOURCE_RELATIVES.items()},
        "checks": checks,
    }


def run_freeze_audit() -> Path:
    if FREEZE_AUDIT_PATH.exists():
        raise RuntimeError("no-overwrite freeze audit refused")
    if FREEZE_PATH.exists():
        raise RuntimeError("freeze audit must precede freeze commit")
    absence = {str(path.relative_to(ROOT)).replace("\\", "/"): not path.exists() for path in FUTURE_ROOTS}
    if not all(absence.values()):
        raise RuntimeError(f"future result root exists before freeze: {absence}")
    core = freeze_artifact_core()
    checks = dict(core["checks"])
    checks["freeze_commit_absent_before_audit"] = True
    checks["future_result_roots_absent_before_audit"] = all(absence.values())
    require(checks, "independent freeze audit")
    payload = self_hashed({
        "schema_version": "phase579_independent_freeze_audit.v1",
        "phase_id": PHASE, "created_at_utc": now(), "passed": True,
        "gpu_used": False, "model_weights_loaded": False,
        "checks": checks,
        "historical_witness": {
            "freeze_commit_absent_before_audit": True,
            "future_result_roots_absent_before_audit": absence,
        },
        "input_identities": {key: value for key, value in core.items() if key != "checks"},
        "candidate_coordinates": [], "candidate_mechanism_formulas": [],
        "causal_intervention_authorized": False,
    }, "freeze_audit_payload_sha256")
    write_json(FREEZE_AUDIT_PATH, payload)
    verify_freeze_audit()
    return FREEZE_AUDIT_PATH


def verify_freeze_audit() -> dict[str, Any]:
    if not regular(FREEZE_AUDIT_PATH):
        raise RuntimeError("freeze audit artifact missing or aliased")
    stored = read_json(FREEZE_AUDIT_PATH)
    core = freeze_artifact_core()
    dynamic_checks = dict(core["checks"])
    checks = {
        "schema": stored.get("schema_version") == "phase579_independent_freeze_audit.v1",
        "phase": stored.get("phase_id") == PHASE,
        "self_hash": verify_self_hash(stored, "freeze_audit_payload_sha256"),
        "stored_passed": stored.get("passed") is True and all(stored.get("checks", {}).values()),
        "cpu_only": stored.get("gpu_used") is False and stored.get("model_weights_loaded") is False,
        "historical_freeze_absence": stored.get("historical_witness", {}).get("freeze_commit_absent_before_audit") is True,
        "historical_future_absence": all(
            stored.get("historical_witness", {}).get("future_result_roots_absent_before_audit", {}).values()
        ),
        "current_immutable_inputs": stored.get("input_identities") == {
            key: value for key, value in core.items() if key != "checks"
        },
        "current_static_checks": all(dynamic_checks.values()),
        "no_candidates": stored.get("candidate_coordinates") == []
        and stored.get("candidate_mechanism_formulas") == [],
    }
    if FREEZE_PATH.exists():
        freeze = read_json(FREEZE_PATH)
        checks["freeze_references_audit"] = freeze.get("independent_audit_sha256") == sha256_file(FREEZE_AUDIT_PATH)
    require(checks, "freeze audit verification")
    # Stable payload: this exact object is identical before and after freeze.
    return {
        "phase_id": PHASE, "passed": True, "gpu_used": False,
        "model_weights_loaded": False,
        "freeze_audit_sha256": sha256_file(FREEZE_AUDIT_PATH),
        "freeze_audit_payload_sha256": stored["freeze_audit_payload_sha256"],
        "verified_static_identity_sha256": sha256_bytes(canonical_json({
            "inputs": stored["input_identities"], "source_checks": dynamic_checks,
        }).encode("utf-8")),
    }


# ---- Independent execution audit -------------------------------------------------


def load_replay(model: str, protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    path = ROOT / "tests/glm5/result/phase578_gpt5_development_behavior_raw" / MODEL_DIRS[model] / "raw_rows.jsonl.gz"
    frozen = protocol.get("upstream_identities", {}).get(f"phase578_{model}_raw_rows")
    if not regular(path) or not isinstance(frozen, Mapping) or not all((
        frozen.get("sha256") == sha256_file(path), frozen.get("size_bytes") == path.stat().st_size,
        frozen.get("is_symlink") is False,
    )):
        raise RuntimeError(f"independent frozen replay identity failed: {model}")
    rows = read_jsonl_gz(path)
    bank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bank[str(row.get("case_id"))].append(row)
    if len(rows) != CASE_COUNT * 2 or len(bank) != CASE_COUNT:
        raise RuntimeError(f"independent replay closure failed: {model}")
    result: dict[str, dict[str, Any]] = {}
    for cid, values in bank.items():
        by_repeat = {value.get("execution_repeat"): value for value in values}
        if set(by_repeat) != {"repeat1", "repeat2"}:
            raise RuntimeError(f"independent repeat closure failed: {model}/{cid}")
        left, right = by_repeat["repeat1"], by_repeat["repeat2"]
        fields = (
            "input_token_ids", "input_token_ids_sha256", "rendered_prompt_sha256",
            "full_generated_suffix_token_ids", "effective_eos_token_ids", "eos_seen",
            "first_eos_index", "first_eos_token_id", "pad_token_id",
        )
        if any(left.get(field) != right.get(field) for field in fields):
            raise RuntimeError(f"independent repeat mismatch: {model}/{cid}")
        if token_ids_sha256(left["input_token_ids"]) != left["input_token_ids_sha256"]:
            raise RuntimeError(f"independent replay token hash failed: {model}/{cid}")
        result[cid] = left
    return result


def metadata_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def independent_role_positions(row: Mapping[str, Any], role: str) -> list[int]:
    aliases = {"focus": ("focus", "focus_object"), "comparison": ("comparison", "comparison_object"),
               "query_anchor": ("query_anchor", "query")}[role]
    for key in ("role_token_positions", "raw_role_token_positions", "token_positions_by_role", "role_positions"):
        bank = row.get(key)
        if not isinstance(bank, Mapping):
            continue
        for alias in aliases:
            value = bank.get(alias)
            if isinstance(value, Mapping):
                value = value.get(
                    "padded_token_positions",
                    value.get("token_positions", value.get("positions")),
                )
            if isinstance(value, list) and all(isinstance(item, int) for item in value):
                return list(map(int, value))
    return []


def trace_registry(model_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(model_dir / "trace_manifest.json")
    raw = manifest.get("shards", manifest.get("trace_shards"))
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"independent shard registry missing: {model_dir}")
    shards: list[dict[str, Any]] = []
    for index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise RuntimeError("independent shard registry row invalid")
        name = row.get(
            "filename", row.get("name", row.get("relative_path", row.get("path")))
        )
        if name != f"trace_shard_{index:04d}.pt":
            raise RuntimeError(f"independent shard order invalid: {name}")
        shards.append(dict(row, filename=name))
    actual = sorted(path.name for path in model_dir.glob("trace_shard_*.pt"))
    if actual != [row["filename"] for row in shards]:
        raise RuntimeError(f"independent shard exact closure failed: {model_dir}")
    return manifest, shards


def tensor_imports() -> tuple[Any, Any]:
    import numpy as np
    import torch
    return np, torch


def independent_tokenizer_reconstruction(
    model: str, cases: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any],
    replay: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from transformers import AutoTokenizer

    frozen = protocol.get("frozen_tokenizer_input_identities", {}).get(model)
    if not isinstance(frozen, Mapping) or not isinstance(frozen.get("files"), Mapping):
        raise RuntimeError(f"independent frozen tokenizer identity absent: {model}")
    tokenizer_files: dict[str, Any] = {}
    for name, identity in sorted(frozen["files"].items()):
        if not isinstance(identity, Mapping):
            raise RuntimeError(f"independent tokenizer file identity invalid: {model}/{name}")
        path = Path(str(identity.get("resolved_path")))
        if not regular(path) or not all((
            sha256_file(path) == identity.get("sha256"),
            path.stat().st_size == identity.get("size_bytes"),
            identity.get("leaf_is_symlink") is False,
        )):
            raise RuntimeError(f"independent tokenizer file drift: {model}/{name}")
        tokenizer_files[name] = {
            "resolved_path": str(path.resolve()), "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    model_dir = Path(str(frozen.get("resolved_directory")))
    if not model_dir.is_dir():
        raise RuntimeError(f"independent tokenizer directory absent: {model}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, use_fast=True, trust_remote_code=False,
    )
    if getattr(tokenizer, "is_fast", False) is not True:
        raise RuntimeError(f"independent fast tokenizer required: {model}")
    output: dict[str, dict[str, Any]] = {}
    for case in cases:
        cid, raw_prompt = str(case_value(case, "case_id")), str(case_value(case, "raw_prompt"))
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if model == "qwen3":
            kwargs["enable_thinking"] = False
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}], **kwargs,
        )
        if not isinstance(rendered, str) or not rendered:
            raise RuntimeError(f"independent chat render invalid: {model}/{cid}")
        encoded = tokenizer(
            rendered, add_special_tokens=False, return_attention_mask=False,
            return_offsets_mapping=True,
        )
        input_ids = [int(value) for value in encoded.input_ids]
        offsets = [tuple(map(int, value)) for value in encoded.offset_mapping]
        if len(input_ids) != len(offsets) or input_ids != replay[cid].get("input_token_ids"):
            raise RuntimeError(f"independent tokenizer input replay failed: {model}/{cid}")
        rendered_hash = sha256_bytes(rendered.encode("utf-8"))
        if rendered_hash != replay[cid].get("rendered_prompt_sha256"):
            raise RuntimeError(f"independent rendered prompt replay failed: {model}/{cid}")
        raw_start = rendered.find(raw_prompt)
        if raw_start < 0 or rendered.find(raw_prompt, raw_start + 1) >= 0:
            raise RuntimeError(f"independent raw prompt uniqueness failed: {model}/{cid}")
        source_spans = case_value(case, "raw_role_char_spans")
        if not isinstance(source_spans, Mapping):
            raise RuntimeError(f"independent source role spans absent: {model}/{cid}")
        roles: dict[str, Any] = {}
        for role in PREFILL_ROLES[:-1]:
            span = source_spans.get(role)
            if span is None:
                roles[role] = {
                    "raw_char_span": None, "rendered_char_span": None,
                    "unpadded_token_positions": [], "token_ids": [],
                }
                continue
            if not isinstance(span, Mapping):
                raise RuntimeError(f"independent source role span invalid: {model}/{cid}/{role}")
            start, end, text = span.get("start"), span.get("end"), span.get("text")
            if not all((isinstance(start, int), isinstance(end, int), isinstance(text, str),
                        0 <= start < end <= len(raw_prompt), raw_prompt[start:end] == text)):
                raise RuntimeError(f"independent raw role span failed: {model}/{cid}/{role}")
            rendered_start, rendered_end = raw_start + start, raw_start + end
            positions = [index for index, (token_start, token_end) in enumerate(offsets)
                         if token_start < rendered_end and token_end > rendered_start]
            if not positions:
                raise RuntimeError(f"independent offset role mapping empty: {model}/{cid}/{role}")
            roles[role] = {
                "raw_char_span": dict(span),
                "rendered_char_span": {"start": rendered_start, "end": rendered_end, "text": text},
                "unpadded_token_positions": positions,
                "token_ids": [input_ids[index] for index in positions],
            }
        output[cid] = {
            "input_token_ids": input_ids, "rendered_prompt_sha256": rendered_hash,
            "raw_prompt_rendered_start": raw_start, "roles": roles,
        }
    report = {
        "model": model, "tokenizer_class": type(tokenizer).__name__, "is_fast": True,
        "use_fast": True, "offset_mapping_route": True,
        "qwen3_enable_thinking": False if model == "qwen3" else None,
        "case_count": len(output), "all_input_ids_match_phase578": True,
        "all_rendered_hashes_match_phase578": True,
        "all_role_offsets_reconstructed": True,
        "frozen_tokenizer_files": tokenizer_files,
        "frozen_tokenizer_identity_sha256": sha256_bytes(
            canonical_json(tokenizer_files).encode("utf-8")
        ),
    }
    del tokenizer
    gc.collect()
    return output, report


def extract_independent_store(
    model: str, cases: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], work: Path,
) -> tuple[dict[str, Any], Any, Any, Any]:
    np, torch = tensor_imports()
    replay = load_replay(model, protocol)
    independent_roles, tokenizer_report = independent_tokenizer_reconstruction(
        model, cases, protocol, replay,
    )
    model_dir = TRACE_DIR / MODEL_DIRS[model]
    status_path, trace_manifest_path = model_dir / "status.json", model_dir / "trace_manifest.json"
    if not regular(status_path) or not regular(trace_manifest_path):
        raise RuntimeError(f"independent trace model files missing: {model}")
    status = read_json(status_path)
    if status.get("model") != model or status.get("status") != "complete":
        raise RuntimeError(f"independent trace status failed: {model}")
    trace_manifest, shards = trace_registry(model_dir)
    expected_model_files = {"status.json", "trace_manifest.json", *[row["filename"] for row in shards]}
    actual_model_files = {path.name for path in model_dir.iterdir() if path.is_file()}
    if actual_model_files != expected_model_files or any(path.is_dir() or path.is_symlink() for path in model_dir.iterdir()):
        raise RuntimeError(f"independent model artifact closure failed: {model}")
    if not all((
        status.get("trace_manifest_sha256") == sha256_file(trace_manifest_path),
        status.get("case_count") == CASE_COUNT,
        status.get("shard_count") == len(shards),
        trace_manifest.get("case_count") == CASE_COUNT,
        trace_manifest.get("shard_count") == len(shards),
        trace_manifest.get("all_shards_finite") is True,
        trace_manifest.get("all_shards_roundtrip_exact") is True,
        trace_manifest.get("hooks_registered") == 0,
        trace_manifest.get("causal_intervention") is False,
        trace_manifest.get("candidate_coordinates") == [],
        trace_manifest.get("candidate_mechanism_formulas") == [],
    )):
        raise RuntimeError(f"independent trace status/manifest failed: {model}")
    index_by_case = {str(case_value(case, "case_id")): index for index, case in enumerate(cases)}
    role_count = len(PREFILL_ROLES) + FEEDBACK_BUDGET
    features = valid = tokens = None
    layer_count = hidden_size = None
    seen: set[str] = set()
    checked_shards: list[dict[str, Any]] = []
    for shard_row in shards:
        path = model_dir / shard_row["filename"]
        if not regular(path) or sha256_file(path) != shard_row.get("sha256"):
            raise RuntimeError(f"independent shard identity failed: {path}")
        shard = torch.load(path, map_location="cpu", weights_only=True)
        expected_keys = {"metadata_rows", "prefill_residual", "prompt_mask", "feedback_residual",
                         "feedback_executed_mask", "feedback_pre_eos_mask"}
        if not isinstance(shard, Mapping) or set(shard) != expected_keys:
            raise RuntimeError(f"independent shard keys failed: {path}")
        metadata = shard["metadata_rows"]
        prefill, prompt = shard["prefill_residual"], shard["prompt_mask"]
        feedback = shard["feedback_residual"]
        executed, pre_eos = shard["feedback_executed_mask"], shard["feedback_pre_eos_mask"]
        if not isinstance(metadata, list) or prefill.ndim != 4 or feedback.ndim != 4:
            raise RuntimeError(f"independent shard rank failed: {path}")
        b, layers, width, hidden = map(int, prefill.shape)
        fb, fl, fw, fh = map(int, feedback.shape)
        shape_ok = (
            b == len(metadata) == fb and layers == fl and hidden == fh
            and fw == FEEDBACK_BUDGET and tuple(prompt.shape) == (b, width)
            and tuple(executed.shape) == (b, fw) and tuple(pre_eos.shape) == (b, fw)
        )
        if not shape_ok or prefill.dtype != torch.bfloat16 or feedback.dtype != torch.bfloat16:
            raise RuntimeError(f"independent shape/dtype failed: {path}")
        if any(mask.dtype != torch.bool for mask in (prompt, executed, pre_eos)):
            raise RuntimeError(f"independent mask dtype failed: {path}")
        if not torch.isfinite(prefill).all().item() or not torch.isfinite(feedback).all().item():
            raise RuntimeError(f"independent finite check failed: {path}")
        if (pre_eos & ~executed).any().item():
            raise RuntimeError(f"independent pre-EOS subset failed: {path}")
        invalid_feedback = ~executed[:, None, :, None]
        if feedback.masked_select(invalid_feedback.expand_as(feedback)).ne(0).any().item():
            raise RuntimeError(f"independent unexecuted feedback is nonzero: {path}")
        if layer_count is None:
            layer_count, hidden_size = layers, hidden
            features = np.memmap(work / f"audit-{model}.bf16", mode="w+", dtype=np.uint16,
                                 shape=(layers, role_count, CASE_COUNT, hidden))
            valid = np.memmap(work / f"audit-{model}.valid", mode="w+", dtype=np.uint8,
                              shape=(role_count, CASE_COUNT))
            tokens = np.memmap(work / f"audit-{model}.tokens", mode="w+", dtype=np.int64,
                               shape=(FEEDBACK_BUDGET, CASE_COUNT))
            features[:] = 0; valid[:] = 0; tokens[:] = -1
        elif (layers, hidden) != (layer_count, hidden_size):
            raise RuntimeError(f"independent cross-shard shape drift: {model}")
        for local, meta in enumerate(metadata):
            if not isinstance(meta, Mapping):
                raise RuntimeError("independent metadata row invalid")
            cid = str(meta.get("case_id"))
            if cid not in index_by_case or cid in seen:
                raise RuntimeError(f"independent case closure failed: {model}/{cid}")
            replay_row = replay[cid]
            input_ids = metadata_value(meta, "input_token_ids", "prompt_input_token_ids")
            generated = metadata_value(meta, "full_generated_suffix_token_ids", "generated_suffix_token_ids",
                                       "generation_token_ids", "generated_token_ids")
            checks = {
                "input": input_ids == replay_row.get("input_token_ids"),
                "input_hash": metadata_value(meta, "input_token_ids_sha256") == replay_row.get("input_token_ids_sha256"),
                "rendered": metadata_value(meta, "rendered_prompt_sha256") == replay_row.get("rendered_prompt_sha256"),
                "generated": generated == replay_row.get("full_generated_suffix_token_ids"),
                "eos": metadata_value(meta, "eos_seen") == replay_row.get("eos_seen"),
                "eos_index": metadata_value(meta, "first_eos_index") == replay_row.get("first_eos_index"),
                "eos_ids": metadata_value(meta, "effective_eos_token_ids") == replay_row.get("effective_eos_token_ids"),
                "pad": metadata_value(meta, "pad_token_id") == replay_row.get("pad_token_id"),
            }
            require(checks, f"independent replay {model}/{cid}")
            left_pad = width - len(input_ids)
            expected_prompt = torch.tensor([False] * left_pad + [True] * len(input_ids), dtype=torch.bool)
            if not torch.equal(prompt[local], expected_prompt):
                raise RuntimeError(f"independent prompt mask failed: {model}/{cid}")
            generated_count = min(max(len(generated) - 1, 0), fw)
            expected_executed = torch.tensor([True] * generated_count + [False] * (fw - generated_count), dtype=torch.bool)
            if not torch.equal(executed[local], expected_executed):
                raise RuntimeError(f"independent execution mask failed: {model}/{cid}")
            first_eos = replay_row.get("first_eos_index")
            before = min(len(generated) if first_eos is None else int(first_eos), fw)
            expected_pre = torch.tensor([True] * before + [False] * (fw - before), dtype=torch.bool)
            if not torch.equal(pre_eos[local], expected_pre):
                raise RuntimeError(f"independent pre-EOS mask failed: {model}/{cid}")
            global_index = index_by_case[cid]
            role_banks = {role: independent_role_positions(meta, role) for role in PREFILL_ROLES[:-1]}
            role_banks["answer_boundary"] = [width - 1]
            if not role_banks["focus"] or not role_banks["query_anchor"]:
                raise RuntimeError(f"independent required roles absent: {model}/{cid}")
            if bool(role_banks["comparison"]) != (case_value(cases[global_index], "comparison_object") is not None):
                raise RuntimeError(f"independent comparison role failed: {model}/{cid}")
            source_spans = case_value(cases[global_index], "raw_role_char_spans", {})
            entries = meta.get("role_token_positions")
            if not isinstance(source_spans, Mapping) or not isinstance(entries, Mapping):
                raise RuntimeError(f"independent role contract absent: {model}/{cid}")
            for role in PREFILL_ROLES[:-1]:
                entry = entries.get(role)
                if not isinstance(entry, Mapping):
                    raise RuntimeError(f"independent role entry absent: {model}/{cid}/{role}")
                unpadded = entry.get("unpadded_token_positions")
                padded = entry.get("padded_token_positions")
                role_ids = entry.get("token_ids")
                if not all(isinstance(value, list) for value in (unpadded, padded, role_ids)):
                    raise RuntimeError(f"independent role token identity invalid: {model}/{cid}/{role}")
                if padded != role_banks[role] or padded != [left_pad + int(value) for value in unpadded]:
                    raise RuntimeError(f"independent role position identity failed: {model}/{cid}/{role}")
                if role_ids != [input_ids[int(value)] for value in unpadded]:
                    raise RuntimeError(f"independent role token ids failed: {model}/{cid}/{role}")
                if entry.get("raw_char_span") != source_spans.get(role):
                    raise RuntimeError(f"independent raw span identity failed: {model}/{cid}/{role}")
                rendered_span = entry.get("rendered_char_span")
                if source_spans.get(role) is None:
                    if rendered_span is not None or padded:
                        raise RuntimeError(f"independent absent role gained identity: {model}/{cid}/{role}")
                elif not isinstance(rendered_span, Mapping) or rendered_span.get("text") != source_spans[role].get("text"):
                    raise RuntimeError(f"independent rendered span failed: {model}/{cid}/{role}")
                independently_rebuilt = independent_roles[cid]["roles"][role]
                if not all((
                    unpadded == independently_rebuilt["unpadded_token_positions"],
                    padded == [left_pad + value for value in independently_rebuilt["unpadded_token_positions"]],
                    role_ids == independently_rebuilt["token_ids"],
                    entry.get("raw_char_span") == independently_rebuilt["raw_char_span"],
                    entry.get("rendered_char_span") == independently_rebuilt["rendered_char_span"],
                )):
                    raise RuntimeError(f"independent fast-offset reconstruction disagreement: {model}/{cid}/{role}")
            for role_index, role in enumerate(PREFILL_ROLES):
                positions = role_banks[role]
                if any(position < left_pad or position >= width for position in positions):
                    raise RuntimeError(f"independent role bounds failed: {model}/{cid}/{role}")
                if positions:
                    vector = prefill[local, :, positions, :].float().mean(dim=1).to(torch.bfloat16)
                    features[:, role_index, global_index, :] = vector.contiguous().view(torch.uint16).numpy()
                    valid[role_index, global_index] = 1
            for step in range(fw):
                role_index = len(PREFILL_ROLES) + step
                if pre_eos[local, step].item():
                    features[:, role_index, global_index, :] = feedback[local, :, step, :].contiguous().view(torch.uint16).numpy()
                    valid[role_index, global_index] = 1
                    tokens[step, global_index] = int(generated[step])
            seen.add(cid)
        checked_shards.append({"filename": path.name, "sha256": sha256_file(path), "case_count": b,
                               "shape": [b, layers, width, hidden], "feedback_shape": [b, fl, fw, fh]})
        del shard, prefill, feedback, prompt, executed, pre_eos
        gc.collect()
    if seen != set(index_by_case):
        raise RuntimeError(f"independent trace cases incomplete: {model}")
    features.flush(); valid.flush(); tokens.flush()
    report = {
        "model": model, "case_count": len(seen), "layer_count_including_embedding": layer_count,
        "hidden_size": hidden_size, "shards": checked_shards,
        "status_sha256": sha256_file(status_path), "trace_manifest_sha256": sha256_file(trace_manifest_path),
        "finite": True, "shape": True, "masks": True, "token_replay": True,
        "independent_tokenizer_reconstruction": tokenizer_report,
        "layer_index_semantics": trace_manifest.get("layer_index_semantics"),
        "layer_index_semantics_sha256": sha256_bytes(
            canonical_json(trace_manifest.get("layer_index_semantics")).encode("utf-8")
        ),
    }
    return report, features, valid, tokens


def independent_axis_pairs(cases: Sequence[Mapping[str, Any]], axis: str) -> list[dict[str, Any]]:
    varying = AXIS_VARYING_FIELD[axis]
    levels = {"relation": ("citrus_membership", "fruit_membership"),
              "query_polarity": ("negative", "positive"), "selection_order": (0, 1),
              "output_contract": ("semantic_label_first", "exact_short"), "paraphrase": None}[axis]
    grouped: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        interface = case_value(case, "interface")
        if axis in ("query_polarity", "selection_order") and interface != "selection": continue
        if axis == "relation" and interface != "direct": continue
        if axis == "relation":
            key = (case_value(case, "focus_object"), case_value(case, "surface_id"),
                   case_value(case, "paraphrase_id"), case_value(case, "output_contract"))
        else:
            fields = ("analysis_unit_id", "relation", "interface", "surface_id", "paraphrase_id",
                      "order", "output_contract", "query_polarity")
            excluded = {varying}
            if axis in ("output_contract", "paraphrase"):
                excluded.add("surface_id")
            key = tuple(case_value(case, field) for field in fields if field not in excluded)
        grouped[key].append(index)
    result: list[dict[str, Any]] = []
    for _, indices in sorted(grouped.items(), key=lambda item: canonical_json(item[0])):
        bank: dict[Any, list[int]] = defaultdict(list)
        for index in indices: bank[case_value(cases[index], varying)].append(index)
        if levels is None:
            ordered = sorted(bank, key=canonical_json)
            pairs = [(ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i + 1, len(ordered))]
        else:
            pairs = [levels] if set(bank) == set(levels) else []
        for low, high in pairs:
            if len(bank[low]) != 1 or len(bank[high]) != 1: continue
            left, right = bank[low][0], bank[high][0]
            source = cases[left]
            unit = str(case_value(source, "analysis_unit_id")) if axis != "relation" else f"relation::{case_value(source, 'focus_object')}"
            slice_id = canonical_json({
                "interface": case_value(source, "interface"),
                "relation": None if axis == "relation" else case_value(source, "relation"),
                "surface_id": None if axis in ("output_contract", "paraphrase") else case_value(source, "surface_id"),
                "paraphrase_id": None if axis == "paraphrase" else case_value(source, "paraphrase_id"),
                "order": None if axis == "selection_order" else case_value(source, "order"),
                "output_contract": None if axis == "output_contract" else case_value(source, "output_contract"),
                "query_polarity": None if axis == "query_polarity" else case_value(source, "query_polarity"),
                "levels": [low, high],
            })
            result.append({"left": left, "right": right, "unit": unit, "slice": slice_id})
    return result


def independent_pair_configuration(protocol: Mapping[str, Any]) -> dict[str, Any]:
    gate = protocol.get("observation_candidate_registration_gate")
    if not isinstance(gate, Mapping):
        raise RuntimeError("independent frozen candidate gate absent")
    allowed = gate.get("matched_pair_allowed_differences")
    coupled = gate.get("axis_coupled_control_fields")
    units = gate.get("evidence_unit_by_axis")
    if not all(isinstance(value, Mapping) and set(value) == set(AXES)
               for value in (allowed, coupled, units)):
        raise RuntimeError("independent per-axis pair contract absent")
    expected_units = {
        "relation": "relation_focus_object_unit", "query_polarity": "analysis_unit_id",
        "selection_order": "analysis_unit_id", "output_contract": "analysis_unit_id",
        "paraphrase": "analysis_unit_id",
    }
    if dict(units) != expected_units:
        raise RuntimeError("independent evidence-unit mapping drift")
    return {
        "allowed": {axis: list(allowed[axis]) for axis in AXES},
        "coupled": {axis: list(coupled[axis]) for axis in AXES},
        "units": dict(units),
    }


def independent_contractualize_pairs(
    cases: Sequence[Mapping[str, Any]], raw: Mapping[str, Sequence[Mapping[str, Any]]],
    protocol: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    config = independent_pair_configuration(protocol)
    output: dict[str, list[dict[str, Any]]] = {}
    for axis in AXES:
        primary = AXIS_VARYING_FIELD[axis]
        allowed = config["allowed"][axis]
        if primary not in allowed:
            raise RuntimeError(f"independent primary field not allowed: {axis}")
        rows: list[dict[str, Any]] = []
        for pair in raw[axis]:
            li, ri = int(pair["left"]), int(pair["right"])
            left, right = cases[li], cases[ri]
            changed = [field for field in PAIR_SEMANTIC_FIELDS
                       if case_value(left, field) != case_value(right, field)]
            unallowed = [field for field in changed if field not in allowed]
            definition = config["units"][axis]
            expected_unit = (
                f"relation::{case_value(left, 'focus_object')}"
                if definition == "relation_focus_object_unit"
                else str(case_value(left, "analysis_unit_id"))
            )
            checks = {
                "case_ids_distinct": case_value(left, "case_id") != case_value(right, "case_id"),
                "primary_contrast_changes": primary in changed
                and case_value(left, primary) != case_value(right, primary),
                "observed_changes_nonempty": bool(changed),
                "no_change_outside_frozen_allowed_bundle": unallowed == [],
                "evidence_unit_matches_frozen_definition": str(pair["unit"]) == expected_unit,
                "slice_id_nonempty": isinstance(pair["slice"], str) and bool(pair["slice"]),
            }
            core = {
                "axis": axis, "primary_contrast_field": primary,
                "left_index": li, "right_index": ri,
                "left_case_id": case_value(left, "case_id"),
                "right_case_id": case_value(right, "case_id"),
                "left_level": case_value(left, primary), "right_level": case_value(right, primary),
                "evidence_unit": str(pair["unit"]), "evidence_unit_definition": definition,
                "control_slice_id": str(pair["slice"]),
                "observed_varying_fields": changed, "contrast_bundle": changed,
                "axis_coupled_control_fields": config["coupled"][axis],
                "frozen_allowed_varying_fields": allowed,
                "unallowed_varying_fields": unallowed,
                "fixed_equal_semantic_fields": [field for field in PAIR_SEMANTIC_FIELDS if field not in changed],
                "contract_checks": checks,
                "matched_control_contract_pass": all(checks.values()),
            }
            rows.append({"pair_id": sha256_bytes(canonical_json(core).encode("utf-8")), **core})
        if not rows or not all(row["matched_control_contract_pass"] for row in rows):
            raise RuntimeError(f"independent matched-pair contract failed: {axis}")
        output[axis] = rows
    return output


def gate_values(protocol: Mapping[str, Any]) -> dict[str, Any]:
    gate = protocol.get("observation_candidate_registration_gate")
    if not isinstance(gate, Mapping):
        raise RuntimeError("independent candidate gate missing")
    direction = gate.get("post_discovery_cross_unit_direction_check")
    keys = (
        "minimum_replay_valid_cases", "minimum_distinct_analysis_units",
        "minimum_distinct_focus_objects", "minimum_distinct_surface_ids",
        "minimum_distinct_paraphrase_ids", "minimum_output_contracts",
        "minimum_unit_vectors", "cross_unit_positive_dot_fraction_numerator",
        "cross_unit_positive_dot_fraction_denominator", "minimum_distinct_control_slices",
        "minimum_reproducing_control_slices", "minimum_tested_invariant_dimensions",
    )
    if not isinstance(direction, Mapping) or any(type(gate.get(key)) is not int for key in keys):
        raise RuntimeError("independent top-level threshold contract missing")
    values: dict[str, Any] = {
        "cases": gate["minimum_replay_valid_cases"],
        "analysis_units": gate["minimum_distinct_analysis_units"],
        "objects": gate["minimum_distinct_focus_objects"],
        "surfaces": gate["minimum_distinct_surface_ids"],
        "paraphrases": gate["minimum_distinct_paraphrase_ids"],
        "contracts": gate["minimum_output_contracts"],
        "unit_vectors": gate["minimum_unit_vectors"],
        "num": gate["cross_unit_positive_dot_fraction_numerator"],
        "den": gate["cross_unit_positive_dot_fraction_denominator"],
        "distinct_slices": gate["minimum_distinct_control_slices"],
        "reproducing_slices": gate["minimum_reproducing_control_slices"],
        "tested_invariants": gate["minimum_tested_invariant_dimensions"],
        "scope_dimensions": list(gate.get("scope_dimensions_must_be_explicit", [])),
        "coupled": independent_pair_configuration(protocol)["coupled"],
        "unit_definitions": independent_pair_configuration(protocol)["units"],
    }
    if not all((
        all(isinstance(item, str) for item in values["scope_dimensions"]),
        len(values["scope_dimensions"]) == len(set(values["scope_dimensions"])),
        direction.get("minimum_eligible_unit_count") == values["unit_vectors"],
        direction.get("evidence_counting_level") == "cross_frozen_evidence_unit_pair",
        direction.get("equivalent_fraction_floor") == f"{values['num']}/{values['den']}",
        direction.get("integer_gate") == "5 * positive_pairwise_dot_count >= 4 * pairwise_dot_total",
        direction.get("mean_is_diagnostic_only") is True,
        values["num"] == 4, values["den"] == 5,
    )):
        raise RuntimeError("independent nested/top-level 4/5 gate disagreement")
    return values


def positive_counts(matrix: Any, torch: Any) -> tuple[int, int]:
    n = int(matrix.shape[0]); total = n * (n - 1) // 2
    if not total: return 0, 0
    gram = matrix @ matrix.T
    upper = torch.triu_indices(n, n, offset=1)
    return int((gram[upper[0], upper[1]] > 0).sum().item()), total


def independent_distinct(values: Sequence[Any] | Any) -> list[Any]:
    bank = {canonical_json(value): value for value in values}
    return [bank[key] for key in sorted(bank)]


def recompute_descriptor(
    differences: Any, pairs: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any], model: str, layer: int, role: str, axis: str,
    global_checks: Mapping[str, bool], feedback_ids: Sequence[int] | None,
    layer_index_semantics_sha256: str, torch: Any,
) -> dict[str, Any]:
    units: dict[str, list[Any]] = defaultdict(list)
    slices: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for pair, vector in zip(pairs, differences):
        units[str(pair["evidence_unit"])].append(vector)
        slices[str(pair["control_slice_id"])][str(pair["evidence_unit"])].append(vector)
    unit_matrix = torch.stack([torch.stack(bank).mean(dim=0) for _, bank in sorted(units.items())])
    positive, total = positive_counts(unit_matrix, torch)
    reproducing = 0
    for _, by_unit in sorted(slices.items()):
        matrix = torch.stack([torch.stack(bank).mean(dim=0) for _, bank in sorted(by_unit.items())])
        spos, stotal = positive_counts(matrix, torch)
        reproducing += int(len(by_unit) >= thresholds["unit_vectors"] and stotal > 0
                           and spos * thresholds["den"] >= stotal * thresholds["num"])
    involved = {int(pair[side]) for pair in pairs for side in ("left_index", "right_index")}
    analysis_units = {case_value(cases[index], "analysis_unit_id") for index in involved}
    coverage = {
        "case_count": len(involved), "evidence_unit_count": len(units),
        "analysis_unit_count": len(analysis_units),
        "focus_object_count": len({case_value(cases[i], "focus_object") for i in involved}),
        "surface_id_count": len({case_value(cases[i], "surface_id") for i in involved}),
        "paraphrase_id_count": len({case_value(cases[i], "paraphrase_id") for i in involved}),
        "output_contract_count": len({case_value(cases[i], "output_contract") for i in involved}),
        "order_levels": independent_distinct([case_value(cases[i], "order") for i in involved]),
        "query_polarity_levels": sorted({case_value(cases[i], "query_polarity") for i in involved}),
        "relation_levels": sorted({case_value(cases[i], "relation") for i in involved}),
        "control_slice_count": len(slices), "reproducing_control_slice_count": reproducing,
    }
    scope_levels: dict[str, list[Any]] = {}
    for dimension in thresholds["scope_dimensions"]:
        if dimension == "token_role": raw_values = [role]
        elif dimension == "layer": raw_values = [layer]
        else: raw_values = [case_value(cases[index], dimension) for index in involved]
        scope_levels[dimension] = independent_distinct(raw_values)
    if feedback_ids is not None:
        scope_levels["matched_feedback_token_id"] = independent_distinct(
            [int(value) for value in feedback_ids]
        )
    contrast = AXIS_VARYING_FIELD[axis]
    excluded = {contrast, "token_role", "layer", *thresholds["coupled"][axis]}
    tested = [dimension for dimension in thresholds["scope_dimensions"]
              if dimension not in excluded and len(scope_levels[dimension]) >= 2]
    bundles = independent_distinct([pair["contrast_bundle"] for pair in pairs])
    changed = independent_distinct([
        field for pair in pairs for field in pair["observed_varying_fields"]
    ])
    pair_ids = [str(pair["pair_id"]) for pair in pairs]
    recurrence = total > 0 and positive * thresholds["den"] >= total * thresholds["num"]
    gate_checks = {
        "global_replay_case_gate": global_checks.get("replay_case_count_exact") is True,
        "global_finite_gate": global_checks.get("all_finite") is True,
        "global_shape_gate": global_checks.get("all_shapes_valid") is True,
        "global_mask_gate": global_checks.get("all_masks_valid") is True,
        "all_usable_pair_contracts_pass": all(pair["matched_control_contract_pass"] is True for pair in pairs),
        "scope_dimensions_complete": set(thresholds["scope_dimensions"]).issubset(scope_levels),
        "primary_contrast_has_two_levels": len(scope_levels[contrast]) >= 2,
        "tested_invariants_are_real_multilevel_controls": all(len(scope_levels[item]) >= 2 for item in tested),
        "enough_tested_invariant_dimensions": len(tested) >= thresholds["tested_invariants"],
        "enough_cases": coverage["case_count"] >= thresholds["cases"],
        "enough_distinct_analysis_units": coverage["analysis_unit_count"] >= thresholds["analysis_units"],
        "enough_evidence_unit_vectors": coverage["evidence_unit_count"] >= thresholds["unit_vectors"],
        "enough_focus_objects": coverage["focus_object_count"] >= thresholds["objects"],
        "enough_surfaces": coverage["surface_id_count"] >= thresholds["surfaces"],
        "enough_paraphrases": coverage["paraphrase_id_count"] >= thresholds["paraphrases"],
        "enough_output_contracts": coverage["output_contract_count"] >= thresholds["contracts"],
        "cross_unit_direction_recurrence": recurrence,
        "enough_distinct_control_slices": coverage["control_slice_count"] >= thresholds["distinct_slices"],
        "enough_reproducing_control_slices": reproducing >= thresholds["reproducing_slices"],
    }
    return {"matched_control_pair_count": len(pairs), "unit_mean_displacement_count": len(units),
            "positive_pairwise_dot_count": positive, "pairwise_dot_total": total,
            "observer_candidate_gate_pass": all(gate_checks.values()), "coverage": coverage,
            "observer_candidate_gate_checks": gate_checks,
            "fixed_coordinates": {"model": model, "representation": "model_api_hidden_state",
                                  "token_role": role, "layer": layer,
                                  "layer_index_semantics_sha256": layer_index_semantics_sha256},
            "evidence_unit_definition": thresholds["unit_definitions"][axis],
            "scope_levels": scope_levels, "tested_invariant_dimensions": tested,
            "tested_invariant_levels": {item: scope_levels[item] for item in tested},
            "observed_changed_fields": changed,
            "contrast_bundle": {"primary_contrast_field": contrast,
                                "observed_bundle_variants": bundles,
                                "pure_single_field_contrast": bundles == [[contrast]]},
            "usable_pair_ids_sha256": sha256_bytes(canonical_json(pair_ids).encode("utf-8")),
            "matched_feedback_token_ids_sha256": None if feedback_ids is None else sha256_bytes(
                canonical_json([int(value) for value in feedback_ids]).encode("utf-8")),
            }


def audit_model_inventory(
    model: str, cases: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], work: Path,
) -> dict[str, Any]:
    np, torch = tensor_imports()
    trace, features, valid, tokens = extract_independent_store(model, cases, protocol, work)
    primary = read_json(INVENTORY_DIR / MODEL_NAMES[model])
    primary_rows = {(row["layer"], row["token_role"], row["axis"]): row for row in primary.get("descriptors", [])}
    if len(primary_rows) != len(primary.get("descriptors", [])):
        raise RuntimeError(f"primary descriptor duplicate: {model}")
    thresholds = gate_values(protocol)
    raw_pairs = {axis: independent_axis_pairs(cases, axis) for axis in AXES}
    pairs_by_axis = independent_contractualize_pairs(cases, raw_pairs, protocol)
    pair_counts = {axis: len(bank) for axis, bank in pairs_by_axis.items()}
    if pair_counts != EXPECTED_AXIS_PAIR_COUNTS:
        raise RuntimeError(f"independent control-pair registry drift: {pair_counts}")
    if primary.get("control_pair_registry_by_axis") != pairs_by_axis:
        raise RuntimeError(f"independent pair registry disagreement: {model}")
    if primary.get("control_pair_registry_sha256") != sha256_bytes(
        canonical_json(pairs_by_axis).encode("utf-8")
    ):
        raise RuntimeError(f"independent pair registry hash disagreement: {model}")
    global_checks = {
        "replay_case_count_exact": trace["case_count"] == CASE_COUNT
        and trace["token_replay"] is True
        and trace["independent_tokenizer_reconstruction"]["case_count"] == CASE_COUNT,
        "all_finite": trace["finite"] is True,
        "all_shapes_valid": trace["shape"] is True,
        "all_masks_valid": trace["masks"] is True,
    }
    if primary.get("global_trace_gate_checks") != global_checks:
        raise RuntimeError(f"independent global trace gate disagreement: {model}")
    roles = [*PREFILL_ROLES, *(f"feedback_token_{i:02d}" for i in range(FEEDBACK_BUDGET))]
    independent_candidates: list[tuple[int, str, str]] = []
    compared = 0
    expected_keys: set[tuple[int, str, str]] = set()
    for layer in range(int(trace["layer_count_including_embedding"])):
        for role_index, role in enumerate(roles):
            matrix = torch.from_numpy(np.asarray(features[layer, role_index])).view(torch.bfloat16).float()
            role_valid = np.asarray(valid[role_index], dtype=np.uint8)
            for axis in AXES:
                usable: list[dict[str, Any]] = []; left: list[int] = []; right: list[int] = []
                feedback_ids: list[int] | None = [] if role.startswith("feedback_token_") else None
                for pair in pairs_by_axis[axis]:
                    a, b = int(pair["left_index"]), int(pair["right_index"])
                    if not role_valid[a] or not role_valid[b]: continue
                    if role.startswith("feedback_token_"):
                        step = role_index - len(PREFILL_ROLES)
                        if int(tokens[step, a]) != int(tokens[step, b]): continue
                        assert feedback_ids is not None
                        feedback_ids.append(int(tokens[step, a]))
                    usable.append(pair); left.append(a); right.append(b)
                if not usable: continue
                key = (layer, role, axis); expected_keys.add(key)
                result = recompute_descriptor(
                    matrix[right] - matrix[left], usable, cases, thresholds,
                    model, layer, role, axis, global_checks, feedback_ids,
                    trace["layer_index_semantics_sha256"], torch,
                )
                stored = primary_rows.get(key)
                if stored is None:
                    raise RuntimeError(f"primary descriptor absent: {model}/{key}")
                if not verify_self_hash(stored, "descriptor_sha256"):
                    raise RuntimeError(f"primary descriptor self-hash failed: {model}/{key}")
                for field in ("matched_control_pair_count", "unit_mean_displacement_count",
                              "positive_pairwise_dot_count", "pairwise_dot_total",
                              "observer_candidate_gate_pass", "observer_candidate_gate_checks",
                              "fixed_coordinates", "evidence_unit_definition", "scope_levels",
                              "tested_invariant_dimensions", "tested_invariant_levels",
                              "observed_changed_fields", "contrast_bundle",
                              "usable_pair_ids_sha256", "matched_feedback_token_ids_sha256"):
                    if stored.get(field) != result[field]:
                        raise RuntimeError(f"independent descriptor disagreement: {model}/{key}/{field}")
                for field, value in result["coverage"].items():
                    if stored.get("coverage", {}).get(field) != value:
                        raise RuntimeError(f"independent coverage disagreement: {model}/{key}/{field}")
                if result["observer_candidate_gate_pass"]:
                    independent_candidates.append(key)
                compared += 1
            del matrix
        gc.collect()
    if expected_keys != set(primary_rows):
        raise RuntimeError(f"independent descriptor closure disagreement: {model}")
    primary_candidate_rows = primary.get("provisional_observer_candidates", [])
    primary_candidates = {(int(row["layer"]), str(row["token_role"]), str(row["contrast_axis"]))
                          for row in primary_candidate_rows}
    if set(independent_candidates) != primary_candidates:
        raise RuntimeError(f"independent candidate registry disagreement: {model}")
    for candidate in primary_candidate_rows:
        key = (int(candidate["layer"]), str(candidate["token_role"]), str(candidate["contrast_axis"]))
        descriptor = primary_rows[key]
        candidate_checks = {
            "descriptor_hash": candidate.get("descriptor_sha256") == descriptor.get("descriptor_sha256"),
            "full_gate": candidate.get("observer_candidate_gate_checks")
            == descriptor.get("observer_candidate_gate_checks"),
            "gate_pass": candidate.get("observer_candidate_gate_pass") is True
            and descriptor.get("observer_candidate_gate_pass") is True,
            "scope": candidate.get("scope_levels") == descriptor.get("scope_levels"),
            "fixed": candidate.get("fixed_coordinates") == descriptor.get("fixed_coordinates"),
            "invariants": candidate.get("tested_invariant_dimensions")
            == descriptor.get("tested_invariant_dimensions"),
            "bundle": candidate.get("contrast_bundle") == descriptor.get("contrast_bundle"),
            "pair_hash": candidate.get("usable_pair_ids_sha256")
            == descriptor.get("usable_pair_ids_sha256"),
            "noncausal": candidate.get("causal") is False
            and candidate.get("mechanism_claimed") is False and candidate.get("formula") is None,
        }
        require(candidate_checks, f"independent candidate payload {model}/{key}")
    return {
        "model": model, "trace_validation": trace,
        "descriptor_count_compared": compared,
        "independently_confirmed_observer_candidates": [
            {"layer": layer, "token_role": role, "contrast_axis": axis,
             "descriptor_sha256": primary_rows[(layer, role, axis)]["descriptor_sha256"],
             "observer_candidate_gate_checks": primary_rows[(layer, role, axis)]["observer_candidate_gate_checks"],
             "fixed_coordinates": primary_rows[(layer, role, axis)]["fixed_coordinates"],
             "scope_levels": primary_rows[(layer, role, axis)]["scope_levels"],
             "tested_invariant_dimensions": primary_rows[(layer, role, axis)]["tested_invariant_dimensions"],
             "contrast_bundle": primary_rows[(layer, role, axis)]["contrast_bundle"],
             "label": "observation_candidate_only", "causal": False}
            for layer, role, axis in independent_candidates
        ],
        "primary_model_inventory_sha256": sha256_file(INVENTORY_DIR / MODEL_NAMES[model]),
        "all_integer_metrics_match": True,
    }


def verify_inventory_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    expected = {SUMMARY_NAME, RECEIPT_NAME, *MODEL_NAMES.values()}
    if not INVENTORY_DIR.is_dir() or INVENTORY_DIR.is_symlink():
        raise RuntimeError("primary inventory directory missing or aliased")
    if {path.name for path in INVENTORY_DIR.iterdir() if path.is_file()} != expected:
        raise RuntimeError("primary inventory exact closure failed")
    receipt, summary = read_json(INVENTORY_DIR / RECEIPT_NAME), read_json(INVENTORY_DIR / SUMMARY_NAME)
    require({
        "receipt_hash": verify_self_hash(receipt, "inventory_receipt_sha256"),
        "summary_hash": verify_self_hash(summary, "summary_sha256"),
        "complete": receipt.get("state") == "complete",
        "cpu": receipt.get("gpu_used") is False and receipt.get("model_weights_loaded") is False,
        "no_cross_model": summary.get("cross_model_alignment_performed") is False,
        "no_formula": summary.get("candidate_mechanism_formulas") == [],
        "no_causal": summary.get("causal_mechanism_claimed") is False,
    }, "primary inventory boundary")
    return receipt, summary


def independent_trace_receipt() -> str:
    path = TRACE_DIR / "execution_receipt.json"
    if not regular(path) or TRACE_DIR.is_symlink():
        raise RuntimeError("independent trace receipt missing or aliased")
    expected_entries = {
        "execution_receipt.json", "stage_start.json", "blocked_model_receipt.json",
        "worker_authorization_00_qwen3.json", "worker_authorization_01_glm4.json",
        "00_qwen3", "01_glm4",
    }
    if {item.name for item in TRACE_DIR.iterdir()} != expected_entries or any(
        item.is_symlink() for item in TRACE_DIR.rglob("*")
    ):
        raise RuntimeError("independent trace root exact closure failed")
    registry = [
        {"path": str(item.relative_to(TRACE_DIR)).replace("\\", "/"),
         "size_bytes": item.stat().st_size, "sha256": sha256_file(item)}
        for item in sorted(TRACE_DIR.rglob("*")) if item.is_file() and item != path
    ]
    receipt = read_json(path)
    checks = {
        "phase": receipt.get("phase_id") == PHASE,
        "mode": receipt.get("mode") == "trace",
        "order": receipt.get("required_model_order") == list(MODELS),
        "attempted": receipt.get("attempted_models_in_order") == list(MODELS),
        "completed": receipt.get("completed_models") == list(MODELS),
        "no_failed": receipt.get("failed_models") == [] and receipt.get("not_attempted_models") == [],
        "full_trace": receipt.get("full_development_trace_complete") is True,
        "blocked": receipt.get("blocked_models") == ["deepseek7b"],
        "registry": receipt.get("artifact_registry_before_receipt") == registry,
        "registry_hash": receipt.get("artifact_registry_sha256")
        == sha256_bytes(canonical_json(registry).encode("utf-8")),
        "no_hooks": receipt.get("hooks_registered") == 0,
        "no_causal": receipt.get("causal_intervention") is False,
        "no_candidates": receipt.get("candidate_coordinates") == []
        and receipt.get("candidate_mechanism_formulas") == [],
    }
    require(checks, "independent trace receipt")
    return sha256_file(path)


def independent_model_qualification_chain(protocol: Mapping[str, Any]) -> dict[str, Any]:
    frozen_artifacts = protocol.get("frozen_model_artifact_identities")
    if not isinstance(frozen_artifacts, Mapping) or set(frozen_artifacts) < set(MODELS):
        raise RuntimeError("independent frozen model-artifact registry absent")
    current_artifact_rehash: dict[str, Any] = {}
    for model in MODELS:
        frozen = frozen_artifacts[model]
        if not isinstance(frozen, Mapping) or not isinstance(frozen.get("files"), list):
            raise RuntimeError(f"independent frozen artifact payload invalid: {model}")
        model_root = Path(str(frozen.get("resolved_path")))
        if not model_root.is_dir():
            raise RuntimeError(f"independent frozen model root absent: {model}")
        expected = {str(item.get("relative_path")): item for item in frozen["files"]
                    if isinstance(item, Mapping)}
        actual = {str(path.relative_to(model_root)).replace("\\", "/"): path
                  for path in model_root.rglob("*") if path.is_file()}
        if set(expected) != set(actual) or len(expected) != frozen.get("file_count"):
            raise RuntimeError(f"independent recursive model closure failed: {model}")
        files: list[dict[str, Any]] = []
        for relative in sorted(expected):
            path, identity = actual[relative], expected[relative]
            observed = {"relative_path": relative, "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path)}
            if path.is_symlink() or observed != {
                "relative_path": relative, "size_bytes": identity.get("size_bytes"),
                "sha256": identity.get("sha256"),
            }:
                raise RuntimeError(f"independent recursive model hash failed: {model}/{relative}")
            files.append(observed)
        rehash_core = {"model": model, "resolved_path": str(model_root.resolve()),
                       "file_count": len(files), "files": files,
                       "frozen_verification_payload_sha256": frozen.get("verification_payload_sha256")}
        current_artifact_rehash[model] = {
            "passed": True, "file_count": len(files),
            "total_bytes": sum(item["size_bytes"] for item in files),
            "rehash_payload_sha256": sha256_bytes(canonical_json(rehash_core).encode("utf-8")),
        }
    modes = (("engineering", ENGINEERING_DIR, 8), ("trace", TRACE_DIR, CASE_COUNT))
    mode_reports: dict[str, Any] = {}
    engineering_receipt_hash = engineering_verification = None
    for mode, root, expected_count in modes:
        receipt_path = root / "execution_receipt.json"
        if not regular(receipt_path) or root.is_symlink():
            raise RuntimeError(f"independent {mode} receipt absent")
        receipt = read_json(receipt_path)
        registry = receipt.get("artifact_registry_before_receipt")
        if not isinstance(registry, list):
            raise RuntimeError(f"independent {mode} artifact registry absent")
        actual_paths = {str(path.relative_to(root)).replace("\\", "/")
                        for path in root.rglob("*") if path.is_file()}
        if actual_paths != {item.get("path") for item in registry} | {"execution_receipt.json"}:
            raise RuntimeError(f"independent {mode} artifact closure failed")
        if any(not isinstance(item, Mapping) or not regular(root / str(item.get("path")))
               or (root / str(item.get("path"))).stat().st_size != item.get("size_bytes")
               or sha256_file(root / str(item.get("path"))) != item.get("sha256")
               for item in registry):
            raise RuntimeError(f"independent {mode} artifact identity failed")
        if receipt.get("artifact_registry_sha256") != sha256_bytes(
            canonical_json(registry).encode("utf-8")
        ):
            raise RuntimeError(f"independent {mode} registry hash failed")
        attempts = {item.get("model"): item for item in receipt.get("attempts", [])
                    if isinstance(item, Mapping)}
        stage_start_path = root / "stage_start.json"
        stage_start = read_json(stage_start_path)
        model_reports: dict[str, Any] = {}
        for index, model in enumerate(MODELS):
            model_root = root / MODEL_DIRS[model]
            status_path, manifest_path = model_root / "status.json", model_root / "trace_manifest.json"
            auth_path = root / f"worker_authorization_{index:02d}_{model}.json"
            status, manifest, authorization = read_json(status_path), read_json(manifest_path), read_json(auth_path)
            attempt = attempts.get(model, {})
            artifact = status.get("model_artifact_verification")
            artifact_core = dict(artifact) if isinstance(artifact, Mapping) else {}
            artifact_hash = artifact_core.pop("verification_payload_sha256", None)
            loaded = status.get("model_identity", {})
            quant = loaded.get("loaded_quantization", {}) if isinstance(loaded, Mapping) else {}
            access = status.get("research_access_attempts", {})
            checks = {
                "status_complete": status.get("status") == "complete" and status.get("mode") == mode,
                "model_order": status.get("model") == model and status.get("model_order_index") == index,
                "case_count": status.get("case_count") == expected_count
                and status.get("expected_case_count") == expected_count,
                "shard_count": status.get("shard_count") == expected_count // 8
                and status.get("expected_shard_count") == expected_count // 8
                and manifest.get("shard_count") == expected_count // 8,
                "manifest_identity": status.get("trace_manifest_sha256") == sha256_file(manifest_path),
                "phase578_raw_identity": status.get("phase578_repeat1_raw_identity")
                == manifest.get("phase578_repeat1_raw_identity"),
                "frozen_artifact": artifact == frozen_artifacts[model],
                "artifact_payload_hash": artifact_hash
                == sha256_bytes(canonical_json(artifact_core).encode("utf-8")),
                "weights_cuda": loaded.get("weights_loaded") is True and loaded.get("gpu_used") is True
                and loaded.get("cuda_only_no_cpu_or_disk_offload") is True,
                "int8_bf16": quant.get("load_in_8bit") is True
                and quant.get("non_quantized_dtype") == "torch.bfloat16",
                "sdpa": loaded.get("loaded_attn_implementation") == "sdpa",
                "cleanup_zero": status.get("cleanup", {}).get("cleanup_pass") is True
                and status.get("cleanup", {}).get("allocated_after_release") == 0
                and status.get("cleanup", {}).get("reserved_after_release") == 0,
                "no_forbidden_access": isinstance(access, Mapping) and all(value == 0 for value in access.values()),
                "observer_only": status.get("hooks_registered") == 0
                and status.get("causal_intervention") is False
                and status.get("attentions_requested") is False
                and status.get("scores_requested") is False,
                "complete_hidden_collection": status.get("hidden_states_requested") is True
                and status.get("all_layers_collected") is True
                and status.get("all_prompt_positions_collected") is True
                and status.get("feedback_residuals_collected") is True,
                "no_fallback_or_future_split": status.get("automatic_fallback_used") is False
                and status.get("confirmation_accessed") is False
                and status.get("heldout_accessed") is False
                and status.get("sealed_accessed") is False,
                "authorization_hash": status.get("worker_authorization_sha256") == sha256_file(auth_path)
                and attempt.get("authorization_sha256") == sha256_file(auth_path),
                "authorization_contract": authorization.get("phase_id") == PHASE
                and authorization.get("mode") == mode and authorization.get("model") == model
                and authorization.get("model_order_index") == index
                and authorization.get("runner_source_sha256") == sha256_file(RUNNER_PATH)
                and authorization.get("stage_start_sha256") == sha256_file(stage_start_path),
                "child_exit_status": attempt.get("child_exit_code") == 0
                and attempt.get("status_sha256") == sha256_file(status_path)
                and attempt.get("cleanup_pass") is True,
                "trace_manifest_contract": manifest.get("case_count") == expected_count
                and manifest.get("storage_dtype") == "torch.bfloat16"
                and manifest.get("shard_keys") == [
                    "metadata_rows", "prefill_residual", "prompt_mask", "feedback_residual",
                    "feedback_executed_mask", "feedback_pre_eos_mask",
                ]
                and isinstance(manifest.get("layer_index_semantics"), Mapping)
                and manifest.get("layer_index_semantics", {}).get("selection")
                == "none; all returned layers are persisted in original order"
                and manifest.get("feedback_index_semantics", {}).get("axis_length") == FEEDBACK_BUDGET
                and manifest.get("feedback_index_semantics", {}).get("invalid_fill")
                == "exact BF16 zero"
                and manifest.get("phase578_repeat1_exact_replay_case_count") == expected_count
                and manifest.get("internal_reexecution_count") == (2 if mode == "engineering" else 1)
                and manifest.get("engineering_reexecution_exact")
                == (True if mode == "engineering" else None)
                and status.get("internal_reexecution_count") == (2 if mode == "engineering" else 1)
                and status.get("engineering_reexecution_exact")
                == (True if mode == "engineering" else None)
                and manifest.get("all_shards_finite") is True
                and manifest.get("all_shards_roundtrip_exact") is True,
            }
            require(checks, f"independent {mode} model qualification {model}")
            model_reports[model] = {
                "checks": checks, "status_sha256": sha256_file(status_path),
                "trace_manifest_sha256": sha256_file(manifest_path),
                "authorization_sha256": sha256_file(auth_path),
                "model_artifact_verification_sha256": sha256_bytes(
                    canonical_json(artifact).encode("utf-8")
                ),
                "loaded_model_identity_sha256": sha256_bytes(
                    canonical_json(loaded).encode("utf-8")
                ),
            }
        receipt_checks = {
            "mode": receipt.get("mode") == mode,
            "order": receipt.get("required_model_order") == list(MODELS)
            and receipt.get("attempted_models_in_order") == list(MODELS),
            "completed": receipt.get("completed_models") == list(MODELS),
            "none_failed": receipt.get("failed_models") == []
            and receipt.get("not_attempted_models") == [],
            "no_cleanup_failure": receipt.get("fatal_cleanup_failure") is False,
            "blocked_deepseek": receipt.get("blocked_models") == ["deepseek7b"],
            "stage_start": stage_start.get("schema_version") == "phase579_execution_stage_start.v1"
            and stage_start.get("phase_id") == PHASE and stage_start.get("mode") == mode
            and stage_start.get("required_model_order") == list(MODELS)
            and stage_start.get("blocked_models") == ["deepseek7b"]
            and stage_start.get("bridge_identity") == receipt.get("bridge_identity")
            and stage_start.get("bridge_identity", {}).get("freeze_sha256") == sha256_file(FREEZE_PATH)
            and stage_start.get("bridge_identity", {}).get("protocol_sha256") == sha256_file(PROTOCOL_PATH)
            and stage_start.get("bridge_identity", {}).get("runner_sha256") == sha256_file(RUNNER_PATH)
            and stage_start.get("bridge_identity", {}).get("manifest_sha256") == sha256_file(MANIFEST_PATH)
            and stage_start.get("candidate_coordinates") == []
            and stage_start.get("candidate_mechanism_formulas") == [],
            "mode_gate": (
                receipt.get("engineering_qualification_passed") is True
                and receipt.get("full_development_trace_authorized") is True
                and receipt.get("full_development_trace_complete") is False
                if mode == "engineering" else
                receipt.get("engineering_qualification_passed") is False
                and receipt.get("full_development_trace_authorized") is False
                and receipt.get("full_development_trace_complete") is True
            ),
        }
        require(receipt_checks, f"independent {mode} receipt qualification")
        if mode == "engineering":
            engineering_receipt_hash = sha256_file(receipt_path)
        else:
            engineering_verification = receipt.get("engineering_verification")
            if not isinstance(engineering_verification, Mapping) or not all((
                engineering_verification.get("passed") is True,
                engineering_verification.get("mode") == "engineering",
                engineering_verification.get("execution_receipt_sha256") == engineering_receipt_hash,
                all(engineering_verification.get("model_status_checks", {}).values()),
            )):
                raise RuntimeError("independent trace-to-engineering bridge failed")
            for model in MODELS:
                authorization = read_json(root / f"worker_authorization_{MODELS.index(model):02d}_{model}.json")
                if authorization.get("engineering_verification") != engineering_verification:
                    raise RuntimeError(f"independent engineering authorization bridge failed: {model}")
        mode_reports[mode] = {
            "receipt_sha256": sha256_file(receipt_path),
            "stage_start_sha256": sha256_file(stage_start_path),
            "receipt_checks": receipt_checks, "models": model_reports,
        }
    return {
        "passed": True, "modes": mode_reports,
        "independent_recursive_model_artifact_rehash": current_artifact_rehash,
        "engineering_verification_payload_sha256": sha256_bytes(
            canonical_json(engineering_verification).encode("utf-8")
        ),
    }


def run_execution_audit() -> Path:
    if EXECUTION_AUDIT_DIR.exists():
        raise RuntimeError("no-overwrite execution audit refused")
    verify_freeze_audit()
    if not regular(FREEZE_PATH):
        raise RuntimeError("final freeze required before execution audit")
    protocol, cases = read_json(PROTOCOL_PATH), read_jsonl(MANIFEST_PATH)
    if len(cases) != CASE_COUNT:
        raise RuntimeError("execution audit manifest closure failed")
    receipt, summary = verify_inventory_artifacts()
    raw_receipt_hash = independent_trace_receipt()
    qualification = independent_model_qualification_chain(protocol)
    work = TEMP_ROOT / f"phase579_audit_{os.getpid()}_{uuid.uuid4().hex}"
    pending = EXECUTION_AUDIT_DIR.with_name(f".{EXECUTION_AUDIT_DIR.name}.pending-{uuid.uuid4().hex}")
    TEMP_ROOT.mkdir(parents=True, exist_ok=True); work.mkdir(); pending.mkdir()
    try:
        reports = [audit_model_inventory(model, cases, protocol, work) for model in MODELS]
        checks = {
            "models_exact": [report["model"] for report in reports] == list(MODELS),
            "all_cases_replayed": all(report["trace_validation"]["case_count"] == CASE_COUNT for report in reports),
            "all_descriptors_match": all(report["all_integer_metrics_match"] for report in reports),
            "no_cross_model_alignment": True,
            "formula_registry_empty": summary.get("candidate_mechanism_formulas") == [],
            "causal_claim_false": summary.get("causal_mechanism_claimed") is False,
        }
        require(checks, "independent execution audit")
        payload = self_hashed({
            "schema_version": "phase579_residual_execution_independent_audit.v1",
            "phase_id": PHASE, "created_at_utc": now(), "passed": True,
            "gpu_used": False, "model_weights_loaded": False,
            "checks": checks, "model_reports": reports,
            "independent_model_qualification_chain": qualification,
            "confirmed_observer_candidates_by_model": {
                report["model"]: report["independently_confirmed_observer_candidates"] for report in reports
            },
            "candidate_mechanism_formulas": [], "causal_mechanism_claimed": False,
            "cross_model_alignment_performed": False,
            "input_identities": {
                "protocol_sha256": sha256_file(PROTOCOL_PATH), "manifest_sha256": sha256_file(MANIFEST_PATH),
                "freeze_sha256": sha256_file(FREEZE_PATH),
                "trace_execution_receipt_sha256": raw_receipt_hash,
                "inventory_receipt_sha256": sha256_file(INVENTORY_DIR / RECEIPT_NAME),
                "inventory_summary_sha256": sha256_file(INVENTORY_DIR / SUMMARY_NAME),
                "audit_source_sha256": sha256_file(AUDIT_SOURCE_PATH),
                "engineering_execution_receipt_sha256": qualification["modes"]["engineering"]["receipt_sha256"],
                "engineering_stage_start_sha256": qualification["modes"]["engineering"]["stage_start_sha256"],
                "trace_stage_start_sha256": qualification["modes"]["trace"]["stage_start_sha256"],
                "engineering_model_status_sha256": {
                    model: qualification["modes"]["engineering"]["models"][model]["status_sha256"]
                    for model in MODELS
                },
                "engineering_model_authorization_sha256": {
                    model: qualification["modes"]["engineering"]["models"][model]["authorization_sha256"]
                    for model in MODELS
                },
                "trace_model_status_sha256": {
                    model: qualification["modes"]["trace"]["models"][model]["status_sha256"]
                    for model in MODELS
                },
                "trace_model_authorization_sha256": {
                    model: qualification["modes"]["trace"]["models"][model]["authorization_sha256"]
                    for model in MODELS
                },
                "trace_model_manifest_sha256": {
                    model: qualification["modes"]["trace"]["models"][model]["trace_manifest_sha256"]
                    for model in MODELS
                },
                "independent_tokenizer_identity_sha256": {
                    report["model"]: report["trace_validation"][
                        "independent_tokenizer_reconstruction"
                    ]["frozen_tokenizer_identity_sha256"] for report in reports
                },
                "independent_model_artifact_rehash_payload_sha256": {
                    model: qualification["independent_recursive_model_artifact_rehash"][model][
                        "rehash_payload_sha256"
                    ] for model in MODELS
                },
            },
        }, "execution_audit_payload_sha256")
        write_json(pending / EXECUTION_AUDIT_NAME, payload)
        os.replace(pending, EXECUTION_AUDIT_DIR)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if pending.exists(): shutil.rmtree(pending, ignore_errors=True)
    verify_execution_audit()
    return EXECUTION_AUDIT_DIR


def verify_execution_audit() -> dict[str, Any]:
    verify_freeze_audit()
    path = EXECUTION_AUDIT_DIR / EXECUTION_AUDIT_NAME
    if not regular(path) or EXECUTION_AUDIT_DIR.is_symlink():
        raise RuntimeError("execution audit artifact missing or aliased")
    if {item.name for item in EXECUTION_AUDIT_DIR.iterdir()} != {EXECUTION_AUDIT_NAME}:
        raise RuntimeError("execution audit exact closure failed")
    payload = read_json(path)
    protocol = read_json(PROTOCOL_PATH)
    current_qualification = independent_model_qualification_chain(protocol)
    cases = read_jsonl(MANIFEST_PATH)
    work = TEMP_ROOT / f"phase579_audit_verify_{os.getpid()}_{uuid.uuid4().hex}"
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    work.mkdir()
    try:
        recomputed_model_reports = [
            audit_model_inventory(model, cases, protocol, work) for model in MODELS
        ]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    checks = {
        "schema": payload.get("schema_version") == "phase579_residual_execution_independent_audit.v1",
        "phase": payload.get("phase_id") == PHASE,
        "self_hash": verify_self_hash(payload, "execution_audit_payload_sha256"),
        "passed": payload.get("passed") is True and all(payload.get("checks", {}).values()),
        "cpu": payload.get("gpu_used") is False and payload.get("model_weights_loaded") is False,
        "no_cross_model": payload.get("cross_model_alignment_performed") is False,
        "no_formula": payload.get("candidate_mechanism_formulas") == [],
        "no_causal": payload.get("causal_mechanism_claimed") is False,
        "protocol": payload.get("input_identities", {}).get("protocol_sha256") == sha256_file(PROTOCOL_PATH),
        "manifest": payload.get("input_identities", {}).get("manifest_sha256") == sha256_file(MANIFEST_PATH),
        "freeze": payload.get("input_identities", {}).get("freeze_sha256") == sha256_file(FREEZE_PATH),
        "trace": payload.get("input_identities", {}).get("trace_execution_receipt_sha256") == independent_trace_receipt(),
        "inventory_receipt": payload.get("input_identities", {}).get("inventory_receipt_sha256") == sha256_file(INVENTORY_DIR / RECEIPT_NAME),
        "inventory_summary": payload.get("input_identities", {}).get("inventory_summary_sha256") == sha256_file(INVENTORY_DIR / SUMMARY_NAME),
        "audit_source": payload.get("input_identities", {}).get("audit_source_sha256") == sha256_file(AUDIT_SOURCE_PATH),
        "model_reports": [row.get("model") for row in payload.get("model_reports", [])] == list(MODELS),
        "full_model_reports_recomputed": payload.get("model_reports") == recomputed_model_reports,
        "qualification_chain": payload.get("independent_model_qualification_chain")
        == current_qualification and current_qualification.get("passed") is True,
        "engineering_receipt": payload.get("input_identities", {}).get(
            "engineering_execution_receipt_sha256"
        ) == current_qualification["modes"]["engineering"]["receipt_sha256"],
        "engineering_statuses": payload.get("input_identities", {}).get(
            "engineering_model_status_sha256"
        ) == {model: current_qualification["modes"]["engineering"]["models"][model]["status_sha256"]
             for model in MODELS},
        "engineering_authorizations": payload.get("input_identities", {}).get(
            "engineering_model_authorization_sha256"
        ) == {model: current_qualification["modes"]["engineering"]["models"][model]["authorization_sha256"]
             for model in MODELS},
        "trace_statuses": payload.get("input_identities", {}).get("trace_model_status_sha256")
        == {model: current_qualification["modes"]["trace"]["models"][model]["status_sha256"]
            for model in MODELS},
        "trace_authorizations": payload.get("input_identities", {}).get(
            "trace_model_authorization_sha256"
        ) == {model: current_qualification["modes"]["trace"]["models"][model]["authorization_sha256"]
              for model in MODELS},
        "trace_manifests": payload.get("input_identities", {}).get("trace_model_manifest_sha256")
        == {model: current_qualification["modes"]["trace"]["models"][model]["trace_manifest_sha256"]
            for model in MODELS},
        "tokenizer_identities": payload.get("input_identities", {}).get(
            "independent_tokenizer_identity_sha256"
        ) == {report["model"]: report["trace_validation"]["independent_tokenizer_reconstruction"][
            "frozen_tokenizer_identity_sha256"] for report in recomputed_model_reports},
        "recursive_model_artifacts": payload.get("input_identities", {}).get(
            "independent_model_artifact_rehash_payload_sha256"
        ) == {model: current_qualification["independent_recursive_model_artifact_rehash"][model][
            "rehash_payload_sha256"] for model in MODELS},
    }
    require(checks, "execution audit verification")
    return {"phase_id": PHASE, "passed": True, "gpu_used": False,
            "model_weights_loaded": False, "checks": checks,
            "execution_audit_sha256": sha256_file(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-freeze-audit", action="store_true")
    group.add_argument("--verify-freeze-audit", action="store_true")
    group.add_argument("--run-execution-audit", action="store_true")
    group.add_argument("--verify-execution-audit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_freeze_audit: result: Any = run_freeze_audit()
    elif args.verify_freeze_audit: result = verify_freeze_audit()
    elif args.run_execution_audit: result = run_execution_audit()
    else: result = verify_execution_audit()
    print(json.dumps(result if isinstance(result, dict) else {"published": str(result)},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
