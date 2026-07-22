#!/usr/bin/env python3
"""CPU-only, observer-only inventory of immutable Phase579 residual traces.

This file deliberately contains no model loading, no intervention and no
cross-model alignment.  It validates the complete trace publication and then
describes matched, within-model residual displacements.  Coordinates are
registered only as *observer candidates*; formulas and causal claims remain
empty/false.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
# generate() returns a prefill hidden-state tuple plus at most max_new_tokens-1
# feedback tuples.  The terminal 24th token is not fed back and must never be
# represented by a fabricated residual.
FEEDBACK_BUDGET = MAX_NEW_TOKENS - 1

PROTOCOL_DIR = ROOT / "tests/glm5/result/phase579_gpt5_residual_protocol"
MANIFEST_PATH = PROTOCOL_DIR / "phase579_development_residual_manifest.jsonl"
PROTOCOL_PATH = PROTOCOL_DIR / "phase579_preregistered_residual_protocol.json"
FREEZE_PATH = PROTOCOL_DIR / "phase579_freeze_commit.json"
TRACE_DIR = ROOT / "tests/glm5/result/phase579_gpt5_residual_trace"
INVENTORY_DIR = ROOT / "tests/glm5/result/phase579_gpt5_residual_inventory"
TEMP_ROOT = ROOT / "tests/glm5_temp"
SOURCE_RELATIVE = "tests/glm5/phase579_gpt5_residual_inventory.py"

SUMMARY_NAME = "phase579_residual_inventory_summary.json"
RECEIPT_NAME = "phase579_inventory_receipt.json"
MODEL_NAMES = {
    "qwen3": "phase579_qwen3_residual_inventory.json",
    "glm4": "phase579_glm4_residual_inventory.json",
}
EXPECTED_INVENTORY_FILES = frozenset((*MODEL_NAMES.values(), SUMMARY_NAME, RECEIPT_NAME))

PREFILL_ROLES = ("focus", "comparison", "query_anchor", "answer_boundary")
AXES = ("relation", "query_polarity", "selection_order", "output_contract", "paraphrase")
AXIS_VARYING_FIELD = {
    "relation": "relation",
    "query_polarity": "query_polarity",
    "selection_order": "order",
    "output_contract": "output_contract",
    "paraphrase": "paraphrase_id",
}
# Identity/hash/span and preregistration placeholders are deliberately absent.
# Every remaining source-case field is part of the independently checked
# semantic matched-pair contract.
PAIR_SEMANTIC_FIELDS = (
    "analysis_unit_id", "candidate_groups", "comparison_object",
    "comparison_object_class", "focus_object", "focus_object_class", "foil",
    "interface", "left_option", "negative_object", "order", "output_contract",
    "paraphrase_id", "positive_object", "query_polarity", "raw_prompt",
    "relation", "relation_contract_id", "right_option", "surface_id", "target",
    "target_truth_polarity",
)
EXPECTED_AXIS_PAIR_COUNTS = {
    "relation": 36,
    "query_polarity": 96,
    "selection_order": 96,
    "output_contract": 168,
    "paraphrase": 240,
}


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


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    return sha256_bytes(canonical_json([int(x) for x in token_ids]).encode("utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"object JSON required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"object JSONL row required: {path}:{line_number}")
            rows.append(value)
    return rows


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"object gzip JSONL row required: {path}:{line_number}")
            rows.append(value)
    return rows


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def regular_unaliased(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def self_hashed(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = sha256_bytes(canonical_json(payload).encode("utf-8"))
    return result


def verify_self_hash(payload: Mapping[str, Any], field: str) -> bool:
    if not isinstance(payload.get(field), str):
        return False
    core = dict(payload)
    claimed = core.pop(field)
    return claimed == sha256_bytes(canonical_json(core).encode("utf-8"))


def exclusive_write(path: Path, data: bytes) -> None:
    if path.exists():
        raise RuntimeError(f"no-overwrite publication refused: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
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
    if isinstance(nested, Mapping) and name in nested:
        return nested[name]
    return default


def require_bool_checks(checks: Mapping[str, Any], label: str) -> None:
    failed = [name for name, value in checks.items() if value is not True]
    if failed:
        raise RuntimeError(f"{label} failed: {failed}")


def protocol_source_identity(freeze: Mapping[str, Any], relative_name: str) -> Mapping[str, Any]:
    identities = freeze.get("source_identities")
    if not isinstance(identities, Mapping) or relative_name not in identities:
        raise RuntimeError(f"frozen source identity missing: {relative_name}")
    identity = identities[relative_name]
    if not isinstance(identity, Mapping):
        raise RuntimeError(f"invalid source identity: {relative_name}")
    return identity


def verify_frozen_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    for path in (PROTOCOL_PATH, MANIFEST_PATH, FREEZE_PATH, TRACE_DIR / "execution_receipt.json"):
        if not regular_unaliased(path):
            raise RuntimeError(f"required immutable input missing or aliased: {path}")
    protocol = read_json(PROTOCOL_PATH)
    freeze = read_json(FREEZE_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    checks = {
        "phase": protocol.get("phase_id") == PHASE and freeze.get("phase_id") == PHASE,
        "freeze_complete": freeze.get("freeze_complete") is True,
        "protocol_hash": freeze.get("protocol_sha256") == sha256_file(PROTOCOL_PATH),
        "manifest_hash": (
            freeze.get("manifest_sha256", freeze.get("development_manifest_sha256"))
            == sha256_file(MANIFEST_PATH)
        ),
        "manifest_count": len(manifest) == CASE_COUNT,
        "manifest_unique": len({case_value(row, "case_id") for row in manifest}) == CASE_COUNT,
        "eligible_models": protocol.get("future_single_model_trace_eligible_models") == list(MODELS)
        and protocol.get("models_in_required_future_order") == list(MODELS),
        "cross_model_forbidden": protocol.get("cross_model_internal_comparison_authorized") is False,
        "no_frozen_coordinates": protocol.get("candidate_coordinates") == [],
        "no_frozen_formulas": protocol.get("candidate_mechanism_formulas") == [],
    }
    source = protocol_source_identity(freeze, SOURCE_RELATIVE)
    checks["inventory_source"] = (
        source.get("sha256") == sha256_file(Path(__file__).resolve())
        and source.get("size_bytes") == Path(__file__).stat().st_size
    )
    require_bool_checks(checks, "frozen inventory bridge")
    for row in manifest:
        cid = case_value(row, "case_id")
        if not isinstance(cid, str) or not cid:
            raise RuntimeError("manifest case_id invalid")
        if case_value(row, "split") != "development":
            raise RuntimeError(f"non-development row refused: {cid}")
        if case_value(row, "candidate_layer") is not None:
            raise RuntimeError(f"preselected candidate layer refused: {cid}")
        if case_value(row, "candidate_neuron") is not None:
            raise RuntimeError(f"preselected candidate neuron refused: {cid}")
        if case_value(row, "candidate_direction") is not None:
            raise RuntimeError(f"preselected candidate direction refused: {cid}")
    return protocol, manifest, freeze


def load_phase578_replay(
    model: str, protocol: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    path = ROOT / "tests/glm5/result/phase578_gpt5_development_behavior_raw" / MODEL_DIRS[model] / "raw_rows.jsonl.gz"
    if not regular_unaliased(path):
        raise RuntimeError(f"Phase578 replay source unavailable: {path}")
    frozen = protocol.get("upstream_identities", {}).get(f"phase578_{model}_raw_rows")
    if not isinstance(frozen, Mapping) or not all((
        frozen.get("sha256") == sha256_file(path),
        frozen.get("size_bytes") == path.stat().st_size,
        frozen.get("is_symlink") is False,
    )):
        raise RuntimeError(f"Phase578 frozen replay identity drift: {model}")
    rows = read_jsonl_gz(path)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("model") != model or row.get("split") != "development":
            raise RuntimeError(f"Phase578 replay scope mismatch: {model}")
        by_case[str(row.get("case_id"))].append(row)
    if len(rows) != CASE_COUNT * 2 or len(by_case) != CASE_COUNT:
        raise RuntimeError(f"Phase578 replay closure mismatch: {model}")
    result: dict[str, dict[str, Any]] = {}
    for case_id, bank in by_case.items():
        repeats = {row.get("execution_repeat"): row for row in bank}
        if set(repeats) != {"repeat1", "repeat2"}:
            raise RuntimeError(f"Phase578 repeat closure mismatch: {model}/{case_id}")
        left, right = repeats["repeat1"], repeats["repeat2"]
        identity_fields = (
            "input_token_ids", "input_token_ids_sha256", "rendered_prompt_sha256",
            "full_generated_suffix_token_ids", "full_generated_suffix_decode",
            "effective_eos_token_ids", "eos_seen", "first_eos_index",
            "first_eos_token_id", "pad_token_id", "termination_event",
        )
        if any(left.get(key) != right.get(key) for key in identity_fields):
            raise RuntimeError(f"Phase578 greedy repeats diverged: {model}/{case_id}")
        if token_ids_sha256(left["input_token_ids"]) != left.get("input_token_ids_sha256"):
            raise RuntimeError(f"Phase578 token hash invalid: {model}/{case_id}")
        result[case_id] = left
    return result


def trace_manifest_shards(model_dir: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = manifest.get("shards", manifest.get("trace_shards"))
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"trace shard registry missing: {model_dir}")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"invalid shard registry row: {model_dir}/{index}")
        filename = item.get(
            "filename", item.get("name", item.get("relative_path", item.get("path")))
        )
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise RuntimeError(f"unsafe shard filename: {filename!r}")
        result.append(dict(item, filename=filename))
    expected = [f"trace_shard_{index:04d}.pt" for index in range(len(result))]
    if [item["filename"] for item in result] != expected:
        raise RuntimeError(f"non-canonical shard sequence: {model_dir}")
    actual = sorted(path.name for path in model_dir.glob("trace_shard_*.pt") if path.is_file())
    if actual != expected:
        raise RuntimeError(f"trace shard exact closure failed: {model_dir}")
    return result


def metadata_value(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return default


def role_positions(row: Mapping[str, Any], role: str) -> list[int]:
    banks = (
        row.get("role_token_positions"), row.get("raw_role_token_positions"),
        row.get("token_positions_by_role"), row.get("role_positions"),
    )
    aliases = {
        "focus": ("focus", "focus_object"),
        "comparison": ("comparison", "comparison_object"),
        "query_anchor": ("query_anchor", "query"),
    }[role]
    for bank in banks:
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
                return [int(item) for item in value]
    return []


def validate_metadata_row(
    row: Mapping[str, Any], case: Mapping[str, Any], replay: Mapping[str, Any],
    padded_width: int, feedback_width: int,
) -> dict[str, Any]:
    case_id = metadata_value(row, "case_id")
    input_ids = metadata_value(row, "input_token_ids", "prompt_input_token_ids")
    generated = metadata_value(
        row, "full_generated_suffix_token_ids", "generated_suffix_token_ids",
        "generation_token_ids", "generated_token_ids",
    )
    if not isinstance(input_ids, list) or not all(isinstance(x, int) for x in input_ids):
        raise RuntimeError(f"trace input tokens invalid: {case_id}")
    if not isinstance(generated, list) or not all(isinstance(x, int) for x in generated):
        raise RuntimeError(f"trace feedback tokens invalid: {case_id}")
    checks = {
        "case": case_id == case_value(case, "case_id"),
        "input_ids": input_ids == replay.get("input_token_ids"),
        "input_hash": (
            metadata_value(row, "input_token_ids_sha256") == replay.get("input_token_ids_sha256")
            and token_ids_sha256(input_ids) == replay.get("input_token_ids_sha256")
        ),
        "rendered_hash": metadata_value(row, "rendered_prompt_sha256") == replay.get("rendered_prompt_sha256"),
        "suffix": generated == replay.get("full_generated_suffix_token_ids"),
        "eos_ids": metadata_value(row, "effective_eos_token_ids") == replay.get("effective_eos_token_ids"),
        "eos_seen": metadata_value(row, "eos_seen") == replay.get("eos_seen"),
        "first_eos": metadata_value(row, "first_eos_index") == replay.get("first_eos_index"),
        "pad": metadata_value(row, "pad_token_id") == replay.get("pad_token_id"),
        "prompt_width": len(input_ids) <= padded_width,
        "feedback_width": len(generated) <= MAX_NEW_TOKENS and feedback_width == FEEDBACK_BUDGET,
    }
    require_bool_checks(checks, f"Phase578 token replay {case_id}")
    expected_boundary = padded_width - 1
    boundary = metadata_value(row, "answer_boundary_position", "generation_start_position")
    if metadata_value(row, "generation_start_position") is not None:
        generation_start = int(metadata_value(row, "generation_start_position"))
        if generation_start not in (padded_width, expected_boundary):
            raise RuntimeError(f"generation boundary mismatch: {case_id}")
    if boundary is None:
        boundary = expected_boundary
    boundary = int(boundary)
    if boundary == padded_width:
        boundary = expected_boundary
    if boundary != expected_boundary:
        raise RuntimeError(f"answer boundary mismatch: {case_id}")
    positions = {role: role_positions(row, role) for role in PREFILL_ROLES[:-1]}
    positions["answer_boundary"] = [boundary]
    for role in ("focus", "query_anchor"):
        if not positions[role]:
            raise RuntimeError(f"required role positions absent: {case_id}/{role}")
    comparison_expected = case_value(case, "comparison_object") is not None
    if bool(positions["comparison"]) != comparison_expected:
        raise RuntimeError(f"comparison role presence mismatch: {case_id}")
    left_pad = padded_width - len(input_ids)
    for role, values in positions.items():
        if any(value < left_pad or value >= padded_width for value in values):
            raise RuntimeError(f"role position outside valid prompt: {case_id}/{role}")
        if len(set(values)) != len(values):
            raise RuntimeError(f"duplicate role position: {case_id}/{role}")
    role_bank = row.get("role_token_positions")
    source_spans = case_value(case, "raw_role_char_spans", {})
    if not isinstance(role_bank, Mapping) or not isinstance(source_spans, Mapping):
        raise RuntimeError(f"role metadata contract absent: {case_id}")
    for role in PREFILL_ROLES[:-1]:
        entry = role_bank.get(role)
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"role metadata entry absent: {case_id}/{role}")
        unpadded = entry.get("unpadded_token_positions")
        padded = entry.get("padded_token_positions")
        ids = entry.get("token_ids")
        if not all(isinstance(value, list) for value in (unpadded, padded, ids)):
            raise RuntimeError(f"role token identity invalid: {case_id}/{role}")
        if padded != positions[role] or padded != [left_pad + int(value) for value in unpadded]:
            raise RuntimeError(f"role padded/unpadded identity mismatch: {case_id}/{role}")
        if ids != [input_ids[int(value)] for value in unpadded]:
            raise RuntimeError(f"role token-id identity mismatch: {case_id}/{role}")
        if entry.get("raw_char_span") != source_spans.get(role):
            raise RuntimeError(f"role raw-span identity mismatch: {case_id}/{role}")
        rendered_span = entry.get("rendered_char_span")
        if source_spans.get(role) is None:
            if rendered_span is not None or padded:
                raise RuntimeError(f"absent role gained rendered identity: {case_id}/{role}")
        elif not isinstance(rendered_span, Mapping) or rendered_span.get("text") != source_spans[role].get("text"):
            raise RuntimeError(f"role rendered-span identity mismatch: {case_id}/{role}")
    return {
        "case_id": case_id,
        "input_ids": input_ids,
        "generated_ids": generated,
        "positions": positions,
        "left_pad": left_pad,
    }


def tensor_imports() -> tuple[Any, Any]:
    import numpy as np  # CPU-only delayed imports
    import torch
    if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("inventory must hide CUDA")
    return np, torch


def torch_load_cpu(path: Path, torch: Any) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("weights_only torch.load support required") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"trace shard must be a mapping: {path}")
    return value


def build_feature_store(
    model: str, cases: Sequence[Mapping[str, Any]], replay: Mapping[str, Mapping[str, Any]],
    work_dir: Path,
) -> tuple[dict[str, Any], Any, Any, Any, list[list[int]]]:
    np, torch = tensor_imports()
    model_dir = TRACE_DIR / MODEL_DIRS[model]
    status_path, manifest_path = model_dir / "status.json", model_dir / "trace_manifest.json"
    if not regular_unaliased(status_path) or not regular_unaliased(manifest_path):
        raise RuntimeError(f"trace model publication incomplete: {model}")
    status, trace_manifest = read_json(status_path), read_json(manifest_path)
    if status.get("model") != model or status.get("status") != "complete":
        raise RuntimeError(f"trace status invalid: {model}")
    shards = trace_manifest_shards(model_dir, trace_manifest)
    expected_model_files = {"status.json", "trace_manifest.json", *[row["filename"] for row in shards]}
    actual_model_files = {path.name for path in model_dir.iterdir() if path.is_file()}
    if actual_model_files != expected_model_files or any(path.is_dir() or path.is_symlink() for path in model_dir.iterdir()):
        raise RuntimeError(f"trace model exact artifact closure failed: {model}")
    if not all((
        status.get("trace_manifest_sha256") == sha256_file(manifest_path),
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
        raise RuntimeError(f"trace status/manifest contract failed: {model}")
    case_index = {str(case_value(row, "case_id")): index for index, row in enumerate(cases)}
    seen: set[str] = set()
    layer_count = hidden_size = None
    role_count = len(PREFILL_ROLES) + FEEDBACK_BUDGET
    feature_path = work_dir / f"{model}.features.bf16"
    valid_path = work_dir / f"{model}.valid.u1"
    token_path = work_dir / f"{model}.tokens.i8"
    features = valid = feedback_tokens = None
    model_shards: list[dict[str, Any]] = []
    for shard_index, registry in enumerate(shards):
        path = model_dir / registry["filename"]
        if not regular_unaliased(path):
            raise RuntimeError(f"trace shard missing or aliased: {path}")
        actual_hash = sha256_file(path)
        if registry.get("sha256") != actual_hash:
            raise RuntimeError(f"trace shard hash mismatch: {path}")
        if registry.get("size_bytes") not in (None, path.stat().st_size):
            raise RuntimeError(f"trace shard size mismatch: {path}")
        shard = torch_load_cpu(path, torch)
        required = {
            "metadata_rows", "prefill_residual", "prompt_mask", "feedback_residual",
            "feedback_executed_mask", "feedback_pre_eos_mask",
        }
        if set(shard) != required:
            raise RuntimeError(f"trace shard key closure failed: {path}: {sorted(shard)}")
        metadata = shard["metadata_rows"]
        prefill, prompt_mask = shard["prefill_residual"], shard["prompt_mask"]
        feedback = shard["feedback_residual"]
        executed, pre_eos = shard["feedback_executed_mask"], shard["feedback_pre_eos_mask"]
        if not isinstance(metadata, list) or not all(isinstance(x, Mapping) for x in metadata):
            raise RuntimeError(f"metadata rows invalid: {path}")
        b = len(metadata)
        shape_checks = {
            "prefill_rank": getattr(prefill, "ndim", None) == 4,
            "prompt_mask_rank": getattr(prompt_mask, "ndim", None) == 2,
            "feedback_rank": getattr(feedback, "ndim", None) == 4,
            "executed_rank": getattr(executed, "ndim", None) == 2,
            "pre_eos_rank": getattr(pre_eos, "ndim", None) == 2,
        }
        require_bool_checks(shape_checks, f"trace shape rank {path}")
        sb, sl, sp, sh = map(int, prefill.shape)
        fb, fl, ff, fh = map(int, feedback.shape)
        if not (sb == b == fb and sl == fl and sh == fh and ff == FEEDBACK_BUDGET):
            raise RuntimeError(f"trace tensor shape disagreement: {path}")
        if tuple(prompt_mask.shape) != (b, sp) or tuple(executed.shape) != (b, ff) or tuple(pre_eos.shape) != (b, ff):
            raise RuntimeError(f"trace mask shape disagreement: {path}")
        if prefill.dtype != torch.bfloat16 or feedback.dtype != torch.bfloat16:
            raise RuntimeError(f"trace residual dtype must be BF16: {path}")
        if prompt_mask.dtype != torch.bool or executed.dtype != torch.bool or pre_eos.dtype != torch.bool:
            raise RuntimeError(f"trace masks must be bool: {path}")
        if not torch.isfinite(prefill).all().item() or not torch.isfinite(feedback).all().item():
            raise RuntimeError(f"non-finite trace refused: {path}")
        if (pre_eos & ~executed).any().item():
            raise RuntimeError(f"pre-EOS mask exceeds executed mask: {path}")
        invalid_feedback = ~executed[:, None, :, None]
        if feedback.masked_select(invalid_feedback.expand_as(feedback)).ne(0).any().item():
            raise RuntimeError(f"unexecuted feedback residual is nonzero: {path}")
        prompt_values = prompt_mask.to(torch.int8)
        if prompt_values.shape[1] > 1 and (
            (prompt_values[:, 1:] - prompt_values[:, :-1]) < 0
        ).any().item():
            raise RuntimeError(f"prompt mask is not left-padded: {path}")
        for mask_name, mask in (("executed", executed), ("pre_eos", pre_eos)):
            values = mask.to(torch.int8)
            if values.shape[1] > 1 and (
                (values[:, 1:] - values[:, :-1]) > 0
            ).any().item():
                raise RuntimeError(f"feedback {mask_name} mask is not a prefix: {path}")
        if layer_count is None:
            layer_count, hidden_size = sl, sh
            features = np.memmap(feature_path, mode="w+", dtype=np.uint16,
                                 shape=(layer_count, role_count, CASE_COUNT, hidden_size))
            valid = np.memmap(valid_path, mode="w+", dtype=np.uint8,
                              shape=(role_count, CASE_COUNT))
            feedback_tokens = np.memmap(token_path, mode="w+", dtype=np.int64,
                                        shape=(FEEDBACK_BUDGET, CASE_COUNT))
            features[:] = 0
            valid[:] = 0
            feedback_tokens[:] = -1
        elif (layer_count, hidden_size) != (sl, sh):
            raise RuntimeError(f"model trace shape changed across shards: {model}")
        for local_index, meta in enumerate(metadata):
            cid = str(meta.get("case_id"))
            if cid not in case_index or cid in seen:
                raise RuntimeError(f"trace case closure failed: {model}/{cid}")
            global_index = case_index[cid]
            detail = validate_metadata_row(meta, cases[global_index], replay[cid], sp, ff)
            expected_prompt = torch.tensor(
                [False] * detail["left_pad"] + [True] * len(detail["input_ids"]),
                dtype=torch.bool,
            )
            if not torch.equal(prompt_mask[local_index], expected_prompt):
                raise RuntimeError(f"prompt mask/token replay mismatch: {model}/{cid}")
            generated_count = min(max(len(detail["generated_ids"]) - 1, 0), ff)
            expected_executed = torch.tensor(
                [True] * generated_count + [False] * (ff - generated_count), dtype=torch.bool,
            )
            if not torch.equal(executed[local_index], expected_executed):
                raise RuntimeError(f"feedback executed mask mismatch: {model}/{cid}")
            first_eos = replay[cid].get("first_eos_index")
            pre_eos_count = min(
                len(detail["generated_ids"]) if first_eos is None else int(first_eos), ff,
            )
            expected_pre_eos = torch.tensor(
                [True] * pre_eos_count + [False] * (ff - pre_eos_count), dtype=torch.bool,
            )
            if not torch.equal(pre_eos[local_index], expected_pre_eos):
                raise RuntimeError(f"feedback pre-EOS mask mismatch: {model}/{cid}")
            for role_index, role in enumerate(PREFILL_ROLES):
                positions = detail["positions"][role]
                if positions:
                    vector = prefill[local_index, :, positions, :].float().mean(dim=1).to(torch.bfloat16)
                    features[:, role_index, global_index, :] = vector.contiguous().view(torch.uint16).numpy()
                    valid[role_index, global_index] = 1
            for step in range(ff):
                role_index = len(PREFILL_ROLES) + step
                if bool(pre_eos[local_index, step].item()):
                    vector = feedback[local_index, :, step, :].contiguous()
                    features[:, role_index, global_index, :] = vector.view(torch.uint16).numpy()
                    valid[role_index, global_index] = 1
                    feedback_tokens[step, global_index] = int(detail["generated_ids"][step])
            seen.add(cid)
        model_shards.append({
            "filename": registry["filename"], "sha256": actual_hash,
            "size_bytes": path.stat().st_size, "case_count": b,
            "shape": [b, sl, sp, sh], "feedback_shape": [b, fl, ff, fh],
        })
        del shard, prefill, feedback, prompt_mask, executed, pre_eos
        gc.collect()
    if seen != set(case_index) or len(seen) != CASE_COUNT:
        raise RuntimeError(f"trace case set incomplete: {model}")
    assert features is not None and valid is not None and feedback_tokens is not None
    features.flush(); valid.flush(); feedback_tokens.flush()
    trace_report = {
        "model": model, "case_count": len(seen), "layer_count_including_embedding": layer_count,
        "hidden_size": hidden_size, "feedback_budget": FEEDBACK_BUDGET,
        "shard_count": len(model_shards), "shards": model_shards,
        "status_sha256": sha256_file(status_path), "trace_manifest_sha256": sha256_file(manifest_path),
        "all_finite": True, "all_shapes_valid": True, "all_masks_valid": True,
        "phase578_token_replay_valid_case_count": len(seen),
        "layer_index_semantics": trace_manifest.get("layer_index_semantics"),
        "layer_index_semantics_sha256": sha256_bytes(
            canonical_json(trace_manifest.get("layer_index_semantics")).encode("utf-8")
        ),
    }
    generated_by_case = [list(map(int, replay[str(case_value(case, "case_id"))]["full_generated_suffix_token_ids"])) for case in cases]
    return trace_report, features, valid, feedback_tokens, generated_by_case


def axis_pairs(cases: Sequence[Mapping[str, Any]], axis: str) -> list[dict[str, Any]]:
    varying = AXIS_VARYING_FIELD[axis]
    expected_levels: dict[str, tuple[Any, Any] | None] = {
        "relation": ("citrus_membership", "fruit_membership"),
        "query_polarity": ("negative", "positive"),
        "selection_order": (0, 1),
        "output_contract": ("semantic_label_first", "exact_short"),
        "paraphrase": None,
    }
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(cases):
        interface = case_value(row, "interface")
        if axis in ("query_polarity", "selection_order") and interface != "selection":
            continue
        if axis == "relation" and interface != "direct":
            continue
        if axis == "relation":
            key = (
                case_value(row, "focus_object"), case_value(row, "surface_id"),
                case_value(row, "paraphrase_id"), case_value(row, "output_contract"),
            )
        else:
            fields = (
                "analysis_unit_id", "relation", "interface", "surface_id", "paraphrase_id",
                "order", "output_contract", "query_polarity",
            )
            excluded = {varying}
            if axis in ("output_contract", "paraphrase"):
                excluded.add("surface_id")
            key = tuple(case_value(row, field) for field in fields if field not in excluded)
        groups[key].append(index)
    result: list[dict[str, Any]] = []
    for key, indices in sorted(groups.items(), key=lambda item: canonical_json(item[0])):
        by_level: dict[Any, list[int]] = defaultdict(list)
        for index in indices:
            by_level[case_value(cases[index], varying)].append(index)
        level_pairs: list[tuple[Any, Any]] = []
        fixed = expected_levels[axis]
        if fixed is not None:
            if set(by_level) == set(fixed) and all(len(by_level[level]) == 1 for level in fixed):
                level_pairs = [fixed]
        else:
            levels = sorted(by_level, key=lambda value: canonical_json(value))
            if all(len(by_level[level]) == 1 for level in levels):
                level_pairs = [(levels[i], levels[j]) for i in range(len(levels)) for j in range(i + 1, len(levels))]
        for low, high in level_pairs:
            left, right = by_level[low][0], by_level[high][0]
            source = cases[left]
            unit = (
                str(case_value(source, "analysis_unit_id")) if axis != "relation"
                else f"relation::{case_value(source, 'focus_object')}"
            )
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


def frozen_gate(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    gate = protocol.get("observation_candidate_registration_gate")
    if not isinstance(gate, Mapping):
        raise RuntimeError("frozen observation candidate gate missing")
    return gate


def pair_contract_configuration(protocol: Mapping[str, Any]) -> dict[str, Any]:
    gate = frozen_gate(protocol)
    allowed = gate.get("matched_pair_allowed_differences")
    coupled = gate.get("axis_coupled_control_fields")
    units = gate.get("evidence_unit_by_axis")
    if not all(isinstance(value, Mapping) and set(value) == set(AXES)
               for value in (allowed, coupled, units)):
        raise RuntimeError("per-axis matched-pair/evidence-unit contract missing")
    expected_units = {
        "relation": "relation_focus_object_unit",
        "query_polarity": "analysis_unit_id",
        "selection_order": "analysis_unit_id",
        "output_contract": "analysis_unit_id",
        "paraphrase": "analysis_unit_id",
    }
    if dict(units) != expected_units:
        raise RuntimeError(f"evidence-unit definition drift: {units}")
    output = {"allowed": {}, "coupled": {}, "units": dict(units)}
    for axis in AXES:
        for name, source in (("allowed", allowed), ("coupled", coupled)):
            values = source.get(axis)
            if (not isinstance(values, list) or not all(isinstance(item, str) for item in values)
                    or len(values) != len(set(values))):
                raise RuntimeError(f"invalid {name} fields: {axis}")
            output[name][axis] = list(values)
        if AXIS_VARYING_FIELD[axis] not in output["allowed"][axis]:
            raise RuntimeError(f"primary contrast absent from allowed fields: {axis}")
    return output


def contractualize_pairs(
    cases: Sequence[Mapping[str, Any]], raw_pairs: Mapping[str, Sequence[Mapping[str, Any]]],
    protocol: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    config = pair_contract_configuration(protocol)
    registries: dict[str, list[dict[str, Any]]] = {}
    for axis in AXES:
        primary = AXIS_VARYING_FIELD[axis]
        allowed = config["allowed"][axis]
        bank: list[dict[str, Any]] = []
        for pair in raw_pairs[axis]:
            left_index, right_index = int(pair["left"]), int(pair["right"])
            left, right = cases[left_index], cases[right_index]
            changed = [
                field for field in PAIR_SEMANTIC_FIELDS
                if case_value(left, field) != case_value(right, field)
            ]
            unallowed = [field for field in changed if field not in allowed]
            fixed_equal = [field for field in PAIR_SEMANTIC_FIELDS if field not in changed]
            low, high = case_value(left, primary), case_value(right, primary)
            unit_definition = config["units"][axis]
            expected_unit = (
                f"relation::{case_value(left, 'focus_object')}"
                if unit_definition == "relation_focus_object_unit"
                else str(case_value(left, "analysis_unit_id"))
            )
            contract_checks = {
                "case_ids_distinct": case_value(left, "case_id") != case_value(right, "case_id"),
                "primary_contrast_changes": primary in changed and low != high,
                "observed_changes_nonempty": bool(changed),
                "no_change_outside_frozen_allowed_bundle": unallowed == [],
                "evidence_unit_matches_frozen_definition": str(pair["unit"]) == expected_unit,
                "slice_id_nonempty": isinstance(pair["slice"], str) and bool(pair["slice"]),
            }
            pair_core = {
                "axis": axis, "primary_contrast_field": primary,
                "left_index": left_index, "right_index": right_index,
                "left_case_id": case_value(left, "case_id"),
                "right_case_id": case_value(right, "case_id"),
                "left_level": low, "right_level": high,
                "evidence_unit": str(pair["unit"]),
                "evidence_unit_definition": unit_definition,
                "control_slice_id": str(pair["slice"]),
                "observed_varying_fields": changed,
                "contrast_bundle": changed,
                "axis_coupled_control_fields": config["coupled"][axis],
                "frozen_allowed_varying_fields": allowed,
                "unallowed_varying_fields": unallowed,
                "fixed_equal_semantic_fields": fixed_equal,
                "contract_checks": contract_checks,
                "matched_control_contract_pass": all(contract_checks.values()),
            }
            pair_id = sha256_bytes(canonical_json(pair_core).encode("utf-8"))
            bank.append({"pair_id": pair_id, **pair_core})
        if not bank or not all(item["matched_control_contract_pass"] for item in bank):
            failed = [item for item in bank if not item["matched_control_contract_pass"]][:3]
            raise RuntimeError(f"matched-control pair contract failed: {axis}: {failed}")
        if len({item["pair_id"] for item in bank}) != len(bank):
            raise RuntimeError(f"duplicate matched-control pair identity: {axis}")
        registries[axis] = bank
    return registries


def pairwise_positive(vectors: Any, torch: Any) -> tuple[int, int, float | None]:
    count = int(vectors.shape[0])
    total = count * (count - 1) // 2
    if total == 0:
        return 0, 0, None
    gram = vectors @ vectors.T
    upper = torch.triu_indices(count, count, offset=1)
    values = gram[upper[0], upper[1]]
    positive = int((values > 0).sum().item())
    mean = float(values.mean().item())
    if not (mean == mean and abs(mean) != float("inf")):
        raise RuntimeError("non-finite pairwise dot diagnostic")
    return positive, total, mean


def distinct_json_values(values: Iterable[Any]) -> list[Any]:
    bank = {canonical_json(value): value for value in values}
    return [bank[key] for key in sorted(bank)]


def gate_thresholds(protocol: Mapping[str, Any]) -> dict[str, Any]:
    gate = frozen_gate(protocol)
    # Protocol coverage conditions remain mandatory.  The integer recurrence
    # conditions are a discovery gate, not a formula or mechanism test.
    direction = gate.get("post_discovery_cross_unit_direction_check", {})
    if not isinstance(direction, Mapping):
        raise RuntimeError("post-discovery direction gate missing")
    integer_keys = (
        "minimum_replay_valid_cases", "minimum_distinct_analysis_units",
        "minimum_distinct_focus_objects", "minimum_distinct_surface_ids",
        "minimum_distinct_paraphrase_ids", "minimum_output_contracts",
        "minimum_unit_vectors", "cross_unit_positive_dot_fraction_numerator",
        "cross_unit_positive_dot_fraction_denominator",
        "minimum_distinct_control_slices", "minimum_reproducing_control_slices",
        "minimum_tested_invariant_dimensions",
    )
    if any(type(gate.get(key)) is not int for key in integer_keys):
        raise RuntimeError("top-level integer candidate threshold missing")
    result: dict[str, Any] = {
        "minimum_replay_valid_cases": gate["minimum_replay_valid_cases"],
        "minimum_distinct_analysis_units": gate["minimum_distinct_analysis_units"],
        "minimum_distinct_focus_objects": gate["minimum_distinct_focus_objects"],
        "minimum_distinct_surface_ids": gate["minimum_distinct_surface_ids"],
        "minimum_distinct_paraphrase_ids": gate["minimum_distinct_paraphrase_ids"],
        "minimum_output_contracts": gate["minimum_output_contracts"],
        "minimum_unit_vectors": gate["minimum_unit_vectors"],
        "positive_dot_numerator": gate["cross_unit_positive_dot_fraction_numerator"],
        "positive_dot_denominator": gate["cross_unit_positive_dot_fraction_denominator"],
        "minimum_distinct_control_slices": gate["minimum_distinct_control_slices"],
        "minimum_reproducing_control_slices": gate["minimum_reproducing_control_slices"],
        "minimum_tested_invariant_dimensions": gate["minimum_tested_invariant_dimensions"],
    }
    if any(value <= 0 for value in result.values()):
        raise RuntimeError("non-positive candidate threshold refused")
    if result["positive_dot_numerator"] > result["positive_dot_denominator"]:
        raise RuntimeError("candidate positive-dot fraction exceeds one")
    if not all((
        direction.get("minimum_eligible_unit_count") == result["minimum_unit_vectors"],
        direction.get("equivalent_fraction_floor")
        == f"{result['positive_dot_numerator']}/{result['positive_dot_denominator']}",
        direction.get("evidence_counting_level") == "cross_frozen_evidence_unit_pair",
        direction.get("mean_is_diagnostic_only") is True,
        direction.get("integer_gate")
        == "5 * positive_pairwise_dot_count >= 4 * pairwise_dot_total",
        result["positive_dot_numerator"] == 4,
        result["positive_dot_denominator"] == 5,
    )):
        raise RuntimeError("nested/top-level 4/5 recurrence contract disagreement")
    scope = gate.get("scope_dimensions_must_be_explicit")
    if not isinstance(scope, list) or len(scope) != len(set(scope)) or not all(
        isinstance(item, str) for item in scope
    ):
        raise RuntimeError("frozen explicit scope dimensions invalid")
    for required in ("relation", "interface", "query_polarity", "order", "output_contract",
                     "surface_id", "paraphrase_id", "target_truth_polarity",
                     "token_role", "layer"):
        if required not in scope:
            raise RuntimeError(f"required explicit scope dimension absent: {required}")
    result["scope_dimensions"] = list(scope)
    result["axis_coupled_control_fields"] = pair_contract_configuration(protocol)["coupled"]
    result["evidence_unit_by_axis"] = pair_contract_configuration(protocol)["units"]
    forbidden = {
        "formula_registration_allowed": False,
        "causal_mechanism_label_allowed": False,
        "cross_model_support_allowed": False,
        "average_only_or_single_case_registration_forbidden": True,
    }
    if any(gate.get(key) is not expected for key, expected in forbidden.items()):
        raise RuntimeError("candidate epistemic boundary not frozen")
    return result


def descriptor_for(
    model: str, layer: int, role: str, axis: str, pairs: Sequence[Mapping[str, Any]],
    vectors: Any, cases: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any],
    global_trace_checks: Mapping[str, bool], feedback_token_ids: Sequence[int] | None,
    layer_index_semantics_sha256: str, torch: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    by_unit: dict[str, list[Any]] = defaultdict(list)
    by_slice_unit: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    used_pairs: list[Mapping[str, Any]] = []
    for pair, vector in zip(pairs, vectors):
        by_unit[str(pair["evidence_unit"])].append(vector)
        by_slice_unit[str(pair["control_slice_id"])][str(pair["evidence_unit"])].append(vector)
        used_pairs.append(pair)
    unit_vectors = torch.stack([
        torch.stack(bank).mean(dim=0) for _, bank in sorted(by_unit.items())
    ]) if by_unit else torch.empty((0, vectors.shape[-1]), dtype=torch.float32)
    positive, total, mean_dot = pairwise_positive(unit_vectors, torch)
    slice_reports: list[dict[str, Any]] = []
    reproducing = 0
    numerator = thresholds["positive_dot_numerator"]
    denominator = thresholds["positive_dot_denominator"]
    minimum_units = thresholds["minimum_unit_vectors"]
    for slice_id, units in sorted(by_slice_unit.items()):
        matrix = torch.stack([
            torch.stack(bank).mean(dim=0) for _, bank in sorted(units.items())
        ])
        spos, stotal, smean = pairwise_positive(matrix, torch)
        passed = len(units) >= minimum_units and stotal > 0 and spos * denominator >= stotal * numerator
        reproducing += int(passed)
        slice_reports.append({
            "slice_id": slice_id, "unit_vector_count": len(units),
            "positive_pairwise_dot_count": spos, "pairwise_dot_total": stotal,
            "positive_fraction_gate_pass": passed,
            "mean_pairwise_dot_diagnostic": smean,
        })
    involved = sorted({
        int(pair[side]) for pair in used_pairs for side in ("left_index", "right_index")
    })
    analysis_units = {
        case_value(cases[index], "analysis_unit_id") for index in involved
    }
    coverage = {
        "case_count": len(involved),
        "evidence_unit_count": len(by_unit),
        "analysis_unit_count": len(analysis_units),
        "focus_object_count": len({case_value(cases[i], "focus_object") for i in involved}),
        "surface_id_count": len({case_value(cases[i], "surface_id") for i in involved}),
        "paraphrase_id_count": len({case_value(cases[i], "paraphrase_id") for i in involved}),
        "output_contract_count": len({case_value(cases[i], "output_contract") for i in involved}),
        "order_levels": sorted({case_value(cases[i], "order") for i in involved}, key=canonical_json),
        "query_polarity_levels": sorted({case_value(cases[i], "query_polarity") for i in involved}),
        "relation_levels": sorted({case_value(cases[i], "relation") for i in involved}),
        "control_slice_count": len(by_slice_unit),
        "reproducing_control_slice_count": reproducing,
    }
    scope_levels: dict[str, list[Any]] = {}
    for dimension in thresholds["scope_dimensions"]:
        if dimension == "token_role":
            values = [role]
        elif dimension == "layer":
            values = [layer]
        else:
            values = [case_value(cases[index], dimension) for index in involved]
        scope_levels[dimension] = distinct_json_values(values)
    if feedback_token_ids is not None:
        scope_levels["matched_feedback_token_id"] = distinct_json_values(
            int(value) for value in feedback_token_ids
        )
    fixed_coordinates = {
        "model": model, "representation": "model_api_hidden_state",
        "token_role": role, "layer": layer,
        "layer_index_semantics_sha256": layer_index_semantics_sha256,
    }
    contrast_dimension = AXIS_VARYING_FIELD[axis]
    excluded_invariants = {
        contrast_dimension, "token_role", "layer",
        *thresholds["axis_coupled_control_fields"][axis],
    }
    tested_invariants = [
        dimension for dimension in thresholds["scope_dimensions"]
        if dimension not in excluded_invariants
        and len(scope_levels[dimension]) >= 2
    ]
    changed_fields = distinct_json_values(
        field for pair in used_pairs for field in pair["observed_varying_fields"]
    )
    contrast_bundles = distinct_json_values(
        pair["contrast_bundle"] for pair in used_pairs
    )
    usable_pair_ids = [str(pair["pair_id"]) for pair in used_pairs]
    usable_pair_ids_sha256 = sha256_bytes(
        canonical_json(usable_pair_ids).encode("utf-8")
    )
    recurrence = total > 0 and positive * denominator >= total * numerator
    gate_checks = {
        "global_replay_case_gate": global_trace_checks.get("replay_case_count_exact") is True,
        "global_finite_gate": global_trace_checks.get("all_finite") is True,
        "global_shape_gate": global_trace_checks.get("all_shapes_valid") is True,
        "global_mask_gate": global_trace_checks.get("all_masks_valid") is True,
        "all_usable_pair_contracts_pass": all(
            pair["matched_control_contract_pass"] is True for pair in used_pairs
        ),
        "scope_dimensions_complete": set(thresholds["scope_dimensions"]).issubset(scope_levels),
        "primary_contrast_has_two_levels": len(scope_levels[contrast_dimension]) >= 2,
        "tested_invariants_are_real_multilevel_controls": all(
            len(scope_levels[dimension]) >= 2 for dimension in tested_invariants
        ),
        "enough_tested_invariant_dimensions": len(tested_invariants)
        >= thresholds["minimum_tested_invariant_dimensions"],
        "enough_cases": coverage["case_count"] >= thresholds["minimum_replay_valid_cases"],
        "enough_distinct_analysis_units": coverage["analysis_unit_count"]
        >= thresholds["minimum_distinct_analysis_units"],
        "enough_evidence_unit_vectors": coverage["evidence_unit_count"] >= minimum_units,
        "enough_focus_objects": coverage["focus_object_count"] >= thresholds["minimum_distinct_focus_objects"],
        "enough_surfaces": coverage["surface_id_count"] >= thresholds["minimum_distinct_surface_ids"],
        "enough_paraphrases": coverage["paraphrase_id_count"] >= thresholds["minimum_distinct_paraphrase_ids"],
        "enough_output_contracts": coverage["output_contract_count"] >= thresholds["minimum_output_contracts"],
        "cross_unit_direction_recurrence": recurrence,
        "enough_distinct_control_slices": coverage["control_slice_count"]
        >= thresholds["minimum_distinct_control_slices"],
        "enough_reproducing_control_slices": reproducing
        >= thresholds["minimum_reproducing_control_slices"],
    }
    descriptor_core = {
        "axis": axis, "token_role": role, "layer": layer,
        "fixed_coordinates": fixed_coordinates,
        "evidence_unit_definition": thresholds["evidence_unit_by_axis"][axis],
        "scope_levels": scope_levels,
        "tested_invariant_dimensions": tested_invariants,
        "tested_invariant_levels": {
            dimension: scope_levels[dimension] for dimension in tested_invariants
        },
        "observed_changed_fields": changed_fields,
        "contrast_bundle": {
            "primary_contrast_field": contrast_dimension,
            "observed_bundle_variants": contrast_bundles,
            "pure_single_field_contrast": contrast_bundles == [[contrast_dimension]],
        },
        "matched_control_pair_count": len(used_pairs),
        "usable_pair_ids_sha256": usable_pair_ids_sha256,
        "matched_feedback_token_ids_sha256": (
            None if feedback_token_ids is None else sha256_bytes(
                canonical_json([int(value) for value in feedback_token_ids]).encode("utf-8")
            )
        ),
        "unit_mean_displacement_count": len(by_unit),
        "positive_pairwise_dot_count": positive,
        "pairwise_dot_total": total,
        "mean_pairwise_dot_diagnostic": mean_dot,
        "mean_unit_displacement_l2_diagnostic": (
            float(torch.linalg.vector_norm(unit_vectors.mean(dim=0)).item()) if len(by_unit) else None
        ),
        "coverage": coverage, "control_slice_reports": slice_reports,
        "observer_candidate_gate_checks": gate_checks,
        "observer_candidate_gate_pass": all(gate_checks.values()),
        "mean_used_as_registration_gate": False,
        "statistical_significance_claimed": False,
    }
    descriptor = self_hashed(descriptor_core, "descriptor_sha256")
    candidate = None
    if descriptor["observer_candidate_gate_pass"]:
        candidate = {
            "model": model, "coordinate_type": "layer_token_role_observation",
            "layer": layer, "token_role": role, "contrast_axis": axis,
            "descriptor_sha256": descriptor["descriptor_sha256"],
            "fixed_coordinates": fixed_coordinates,
            "evidence_unit_definition": thresholds["evidence_unit_by_axis"][axis],
            "scope_levels": scope_levels,
            "tested_invariant_dimensions": tested_invariants,
            "observed_changed_fields": changed_fields,
            "contrast_bundle": descriptor["contrast_bundle"],
            "usable_pair_ids_sha256": usable_pair_ids_sha256,
            "observer_candidate_gate_checks": dict(gate_checks),
            "observer_candidate_gate_pass": True,
            "label": "observation_candidate_only", "causal": False,
            "mechanism_claimed": False, "formula": None,
            "supporting_evidence_unit_count": len(by_unit),
            "supporting_analysis_unit_count": len(analysis_units),
            "supporting_control_slice_count": reproducing,
            "positive_pairwise_dot_count": positive, "pairwise_dot_total": total,
        }
    return descriptor, candidate


def scan_model(
    model: str, cases: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], work_dir: Path,
) -> dict[str, Any]:
    np, torch = tensor_imports()
    replay = load_phase578_replay(model, protocol)
    trace_report, features, valid, feedback_tokens, _ = build_feature_store(
        model, cases, replay, work_dir,
    )
    thresholds = gate_thresholds(protocol)
    raw_pairs = {axis: axis_pairs(cases, axis) for axis in AXES}
    all_pairs = contractualize_pairs(cases, raw_pairs, protocol)
    pair_counts = {axis: len(bank) for axis, bank in all_pairs.items()}
    if pair_counts != EXPECTED_AXIS_PAIR_COUNTS:
        raise RuntimeError(f"control-pair registry drift: {pair_counts}")
    descriptors: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    global_trace_checks = {
        "replay_case_count_exact": trace_report["case_count"] == CASE_COUNT
        and trace_report["phase578_token_replay_valid_case_count"] == CASE_COUNT,
        "all_finite": trace_report["all_finite"] is True,
        "all_shapes_valid": trace_report["all_shapes_valid"] is True,
        "all_masks_valid": trace_report["all_masks_valid"] is True,
    }
    if not all(global_trace_checks.values()):
        raise RuntimeError(f"global trace gate failed before scan: {model}")
    layer_count = int(trace_report["layer_count_including_embedding"])
    hidden_size = int(trace_report["hidden_size"])
    role_names = [*PREFILL_ROLES, *(f"feedback_token_{i:02d}" for i in range(FEEDBACK_BUDGET))]
    for layer in range(layer_count):
        for role_index, role in enumerate(role_names):
            bits = np.asarray(features[layer, role_index, :, :])
            matrix = torch.from_numpy(bits).view(torch.bfloat16).float()
            role_valid = np.asarray(valid[role_index, :], dtype=np.uint8)
            for axis in AXES:
                usable: list[dict[str, Any]] = []
                left_indices: list[int] = []
                right_indices: list[int] = []
                matched_feedback_ids: list[int] | None = (
                    [] if role.startswith("feedback_token_") else None
                )
                for pair in all_pairs[axis]:
                    left, right = int(pair["left_index"]), int(pair["right_index"])
                    if not role_valid[left] or not role_valid[right]:
                        continue
                    if role.startswith("feedback_token_"):
                        step = role_index - len(PREFILL_ROLES)
                        if int(feedback_tokens[step, left]) != int(feedback_tokens[step, right]):
                            continue
                        assert matched_feedback_ids is not None
                        matched_feedback_ids.append(int(feedback_tokens[step, left]))
                    usable.append(pair); left_indices.append(left); right_indices.append(right)
                if not usable:
                    continue
                differences = matrix[right_indices, :] - matrix[left_indices, :]
                descriptor, candidate = descriptor_for(
                    model, layer, role, axis, usable, differences, cases, thresholds,
                    global_trace_checks, matched_feedback_ids,
                    trace_report["layer_index_semantics_sha256"], torch,
                )
                descriptors.append(descriptor)
                if candidate is not None:
                    candidates.append(candidate)
                del differences
            del matrix
        gc.collect()
    report = {
        "schema_version": "phase579_residual_inventory_model.v1",
        "phase_id": PHASE, "model": model, "created_at_utc": now(),
        "observer_only": True, "cpu_only_analysis": True,
        "cross_model_alignment_performed": False,
        "trace_validation": trace_report,
        "global_trace_gate_checks": global_trace_checks,
        "control_axes": list(AXES), "token_roles": role_names,
        "pair_registry_counts": pair_counts,
        "control_pair_registry_by_axis": all_pairs,
        "control_pair_registry_sha256": sha256_bytes(
            canonical_json(all_pairs).encode("utf-8")
        ),
        "descriptor_count": len(descriptors), "descriptors": descriptors,
        # Registration is a two-step boundary: these discoveries remain
        # provisional until the independent execution audit reproduces them.
        "provisional_observer_candidates": candidates,
        "observer_candidates": [],
        "candidate_coordinates": [],
        "candidate_mechanism_formulas": [], "causal_mechanism_claimed": False,
        "candidate_gate_thresholds": thresholds,
    }
    return self_hashed(report, "model_inventory_sha256")


def trace_receipt_hash() -> str:
    path = TRACE_DIR / "execution_receipt.json"
    receipt = read_json(path)
    expected_entries = {
        "execution_receipt.json", "stage_start.json", "blocked_model_receipt.json",
        "worker_authorization_00_qwen3.json", "worker_authorization_01_glm4.json",
        "00_qwen3", "01_glm4",
    }
    if {item.name for item in TRACE_DIR.iterdir()} != expected_entries or any(
        item.is_symlink() for item in TRACE_DIR.rglob("*")
    ):
        raise RuntimeError("trace root exact artifact closure failed")
    actual_registry = []
    for item in sorted(TRACE_DIR.rglob("*")):
        if item.is_file() and item != path:
            actual_registry.append({
                "path": str(item.relative_to(TRACE_DIR)).replace("\\", "/"),
                "size_bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            })
    checks = {
        "phase": receipt.get("phase_id") == PHASE,
        "mode": receipt.get("mode") == "trace",
        "order": receipt.get("required_model_order") == list(MODELS),
        "attempted": receipt.get("attempted_models_in_order") == list(MODELS),
        "completed": receipt.get("completed_models") == list(MODELS),
        "no_failed": receipt.get("failed_models") == [] and receipt.get("not_attempted_models") == [],
        "full_trace": receipt.get("full_development_trace_complete") is True,
        "blocked": receipt.get("blocked_models") == ["deepseek7b"],
        "no_hooks": receipt.get("hooks_registered") == 0,
        "no_causal": receipt.get("causal_intervention") is False,
        "no_candidates": receipt.get("candidate_coordinates") == []
        and receipt.get("candidate_mechanism_formulas") == [],
        "artifact_registry": receipt.get("artifact_registry_before_receipt") == actual_registry,
        "artifact_registry_hash": receipt.get("artifact_registry_sha256")
        == sha256_bytes(canonical_json(actual_registry).encode("utf-8")),
    }
    require_bool_checks(checks, "trace execution receipt")
    return sha256_file(path)


def build_summary(
    protocol: Mapping[str, Any], freeze: Mapping[str, Any], model_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    provisional = {report["model"]: report["provisional_observer_candidates"] for report in model_reports}
    return self_hashed({
        "schema_version": "phase579_residual_inventory_summary.v1",
        "phase_id": PHASE, "created_at_utc": now(),
        "models_in_required_order": list(MODELS),
        "models_analyzed_independently": list(MODELS),
        "cross_model_alignment_performed": False,
        "cross_model_internal_comparison_performed": False,
        "model_reports": [{
            "model": report["model"],
            "model_inventory_filename": MODEL_NAMES[report["model"]],
            "model_inventory_sha256": report["model_inventory_sha256"],
            "case_count": report["trace_validation"]["case_count"],
            "layer_count_including_embedding": report["trace_validation"]["layer_count_including_embedding"],
            "descriptor_count": report["descriptor_count"],
            "provisional_observer_candidate_count": len(report["provisional_observer_candidates"]),
            "registered_observer_candidate_count_before_independent_audit": 0,
        } for report in model_reports],
        "provisional_observer_candidates_by_model": provisional,
        "observer_candidates_by_model": {model: [] for model in MODELS},
        "candidate_coordinates": [],
        "candidate_mechanism_formulas": [],
        "causal_mechanism_claimed": False,
        "theory_formula_registration_authorized": False,
        "statistical_significance_claimed": False,
        "input_identities": {
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "freeze_sha256": sha256_file(FREEZE_PATH),
            "trace_execution_receipt_sha256": trace_receipt_hash(),
            "inventory_source_sha256": sha256_file(Path(__file__).resolve()),
        },
    }, "summary_sha256")


def run_inventory() -> Path:
    protocol, cases, freeze = verify_frozen_inputs()
    if INVENTORY_DIR.exists():
        raise RuntimeError(f"no-overwrite inventory refused: {INVENTORY_DIR}")
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    work = TEMP_ROOT / f"phase579_inventory_{os.getpid()}_{uuid.uuid4().hex}"
    pending = INVENTORY_DIR.with_name(f".{INVENTORY_DIR.name}.pending-{uuid.uuid4().hex}")
    work.mkdir(parents=False)
    pending.mkdir(parents=False)
    try:
        reports = [scan_model(model, cases, protocol, work) for model in MODELS]
        for report in reports:
            write_json(pending / MODEL_NAMES[report["model"]], report)
        summary = build_summary(protocol, freeze, reports)
        write_json(pending / SUMMARY_NAME, summary)
        outputs = {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(pending.iterdir()) if path.is_file()
        }
        receipt = self_hashed({
            "schema_version": "phase579_residual_inventory_receipt.v1",
            "phase_id": PHASE, "created_at_utc": now(), "state": "complete",
            "gpu_used": False, "model_weights_loaded": False,
            "models_analyzed_serially": list(MODELS),
            "cross_model_alignment_performed": False,
            "candidate_mechanism_formulas": [], "causal_mechanism_claimed": False,
            "summary_sha256": summary["summary_sha256"], "outputs": outputs,
        }, "inventory_receipt_sha256")
        write_json(pending / RECEIPT_NAME, receipt)
        actual = {path.name for path in pending.iterdir() if path.is_file()}
        if actual != EXPECTED_INVENTORY_FILES:
            raise RuntimeError(f"inventory pending closure failed: {sorted(actual)}")
        os.replace(pending, INVENTORY_DIR)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if pending.exists():
            shutil.rmtree(pending, ignore_errors=True)
    verify_inventory()
    return INVENTORY_DIR


def verify_inventory() -> dict[str, Any]:
    protocol, cases, freeze = verify_frozen_inputs()
    if not INVENTORY_DIR.is_dir() or INVENTORY_DIR.is_symlink():
        raise RuntimeError("inventory directory missing or aliased")
    actual = {path.name for path in INVENTORY_DIR.iterdir() if path.is_file()}
    if actual != EXPECTED_INVENTORY_FILES:
        raise RuntimeError(f"inventory exact closure failed: {sorted(actual)}")
    receipt, summary = read_json(INVENTORY_DIR / RECEIPT_NAME), read_json(INVENTORY_DIR / SUMMARY_NAME)
    checks = {
        "receipt_self_hash": verify_self_hash(receipt, "inventory_receipt_sha256"),
        "summary_self_hash": verify_self_hash(summary, "summary_sha256"),
        "receipt_complete": receipt.get("state") == "complete",
        "cpu_only": receipt.get("gpu_used") is False and receipt.get("model_weights_loaded") is False,
        "model_order": receipt.get("models_analyzed_serially") == list(MODELS),
        "no_cross_model": receipt.get("cross_model_alignment_performed") is False
        and summary.get("cross_model_alignment_performed") is False,
        "no_formulas": receipt.get("candidate_mechanism_formulas") == []
        and summary.get("candidate_mechanism_formulas") == [],
        "no_causal": receipt.get("causal_mechanism_claimed") is False
        and summary.get("causal_mechanism_claimed") is False,
        "protocol_identity": summary.get("input_identities", {}).get("protocol_sha256") == sha256_file(PROTOCOL_PATH),
        "manifest_identity": summary.get("input_identities", {}).get("manifest_sha256") == sha256_file(MANIFEST_PATH),
        "freeze_identity": summary.get("input_identities", {}).get("freeze_sha256") == sha256_file(FREEZE_PATH),
        "trace_identity": summary.get("input_identities", {}).get("trace_execution_receipt_sha256") == trace_receipt_hash(),
        "source_identity": summary.get("input_identities", {}).get("inventory_source_sha256") == sha256_file(Path(__file__).resolve()),
    }
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != EXPECTED_INVENTORY_FILES - {RECEIPT_NAME}:
        checks["output_registry"] = False
    else:
        checks["output_registry"] = all(
            isinstance(identity, Mapping)
            and identity.get("sha256") == sha256_file(INVENTORY_DIR / name)
            and identity.get("size_bytes") == (INVENTORY_DIR / name).stat().st_size
            for name, identity in outputs.items()
        )
    model_summaries = {item.get("model"): item for item in summary.get("model_reports", []) if isinstance(item, Mapping)}
    checks["model_summary_closure"] = set(model_summaries) == set(MODELS)
    for model in MODELS:
        report = read_json(INVENTORY_DIR / MODEL_NAMES[model])
        model_checks = {
            "self_hash": verify_self_hash(report, "model_inventory_sha256"),
            "model": report.get("model") == model,
            "case_count": report.get("trace_validation", {}).get("case_count") == CASE_COUNT,
            "replay_count": report.get("trace_validation", {}).get("phase578_token_replay_valid_case_count") == CASE_COUNT,
            "finite": report.get("trace_validation", {}).get("all_finite") is True,
            "shape": report.get("trace_validation", {}).get("all_shapes_valid") is True,
            "mask": report.get("trace_validation", {}).get("all_masks_valid") is True,
            "no_cross_model": report.get("cross_model_alignment_performed") is False,
            "no_formulas": report.get("candidate_mechanism_formulas") == [],
            "no_causal": report.get("causal_mechanism_claimed") is False,
            "descriptor_count": report.get("descriptor_count") == len(report.get("descriptors", [])),
            "global_trace_gate": all(report.get("global_trace_gate_checks", {}).values()),
            "pair_registry_hash": report.get("control_pair_registry_sha256")
            == sha256_bytes(canonical_json(report.get("control_pair_registry_by_axis")).encode("utf-8")),
            "pair_registry_counts": report.get("pair_registry_counts") == {
                axis: len(report.get("control_pair_registry_by_axis", {}).get(axis, [])) for axis in AXES
            } == EXPECTED_AXIS_PAIR_COUNTS,
            "registered_candidates_still_empty": report.get("observer_candidates") == []
            and report.get("candidate_coordinates") == [],
        }
        for descriptor in report.get("descriptors", []):
            if not isinstance(descriptor, Mapping):
                model_checks["descriptor_schema"] = False; break
            positive, total = descriptor.get("positive_pairwise_dot_count"), descriptor.get("pairwise_dot_total")
            if not isinstance(positive, int) or not isinstance(total, int) or not 0 <= positive <= total:
                model_checks["descriptor_counts"] = False; break
            if not verify_self_hash(descriptor, "descriptor_sha256"):
                model_checks["descriptor_hash"] = False; break
            if set(descriptor.get("observer_candidate_gate_checks", {})) != {
                "global_replay_case_gate", "global_finite_gate", "global_shape_gate",
                "global_mask_gate", "all_usable_pair_contracts_pass",
                "scope_dimensions_complete", "primary_contrast_has_two_levels",
                "tested_invariants_are_real_multilevel_controls",
                "enough_tested_invariant_dimensions", "enough_cases",
                "enough_distinct_analysis_units", "enough_evidence_unit_vectors",
                "enough_focus_objects", "enough_surfaces", "enough_paraphrases",
                "enough_output_contracts", "cross_unit_direction_recurrence",
                "enough_distinct_control_slices", "enough_reproducing_control_slices",
            }:
                model_checks["descriptor_gate_schema"] = False; break
            if descriptor.get("observer_candidate_gate_pass") is not all(
                descriptor["observer_candidate_gate_checks"].values()
            ):
                model_checks["descriptor_gate_value"] = False; break
        descriptors_by_hash = {
            row.get("descriptor_sha256"): row for row in report.get("descriptors", [])
        }
        for candidate in report.get("provisional_observer_candidates", []):
            source = descriptors_by_hash.get(candidate.get("descriptor_sha256"))
            if not isinstance(source, Mapping) or not all((
                source.get("observer_candidate_gate_pass") is True,
                candidate.get("observer_candidate_gate_checks")
                == source.get("observer_candidate_gate_checks"),
                candidate.get("scope_levels") == source.get("scope_levels"),
                candidate.get("contrast_bundle") == source.get("contrast_bundle"),
                candidate.get("causal") is False,
                candidate.get("mechanism_claimed") is False,
                candidate.get("formula") is None,
            )):
                model_checks["provisional_candidate_contract"] = False; break
        require_bool_checks(model_checks, f"stored model inventory {model}")
        if model in model_summaries:
            checks[f"{model}_summary_hash"] = (
                model_summaries[model].get("model_inventory_sha256") == report.get("model_inventory_sha256")
            )
    require_bool_checks(checks, "inventory verification")
    return {"phase_id": PHASE, "verified": True, "checks": checks,
            "inventory_receipt_sha256": sha256_file(INVENTORY_DIR / RECEIPT_NAME)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result: Any = run_inventory() if args.run else verify_inventory()
    print(json.dumps(result if isinstance(result, dict) else {"published": str(result)},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
