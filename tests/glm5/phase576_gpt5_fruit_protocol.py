#!/usr/bin/env python3
"""Freeze the evidence-first Phase576 natural fruit knowledge denominator.

This file defines external cases and operational gates only.  It deliberately
does not name a layer, head, neuron, component, representation direction, or
mechanism formula.  The sealed split is generated before any model execution
and must not be read until an open-data decision authorizes it.
"""

from __future__ import annotations

import argparse
import ast
import gc
import gzip
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
GPT5 = ROOT / "tests" / "gpt5"
if str(GPT5) not in sys.path:
    sys.path.insert(0, str(GPT5))

from phase548_shared_attention_compute_protocol import (  # noqa: E402
    render_chat,
    tokenizer_for,
)
from model_registry import get_model_spec  # noqa: E402


PHASE = "Phase576"
SCHEMA_VERSION = "phase576_gpt5_fruit_protocol.v2"
CASE_SCHEMA_VERSION = "phase576_gpt5_fruit_case.v2"
DISCOVERY_REGISTRY_SCHEMA_VERSION = "phase576_discovered_structure_registry.v2"
CONFIRMATION_DECISION_SCHEMA_VERSION = "phase576_structure_confirmation_decision.v2"
HELDOUT_DECISION_SCHEMA_VERSION = "phase576_heldout_replication_decision.v1"
TRACE_MANIFEST_SCHEMA_VERSION = "phase576_natural_trace_manifest.v2"
TRACE_RECEIPT_SCHEMA_VERSION = "phase576_trace_execution_receipt.v1"
TRACE_STAGE_RECEIPT_SCHEMA_VERSION = "phase576_trace_execution_receipt.v2"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("discovery", "confirmation", "heldout_recombination", "sealed")
OPEN_SPLITS = SPLITS[:-1]
RELATIONS = ("category", "color")
INTERFACES = ("direct", "selection")
DIRECT_SURFACES = tuple(range(6))
SELECTION_SURFACES = tuple(range(4))
SELECTION_ORDERS = (0, 1)
BEHAVIOR_REPEATS = ("repeat1", "repeat2")
BEHAVIOR_BATCH_SIZE = 4
TRACE_BATCH_SIZE = 4
MAX_NEW_TOKENS = 8
CATEGORY_RELATION_CONTRACT_ID = "fruit_membership_binary.v1"
COLOR_RELATION_CONTRACT_ID = "most_common_color_association.v1"
POST_TRACE_ALLOWED_IMPORT_ROOTS = (
    "__future__", "argparse", "collections", "datetime", "gc", "gzip",
    "hashlib", "json", "math", "os", "pathlib", "torch", "typing",
)

OUT_DIR = ROOT / "tests/glm5/result/phase576_gpt5_fruit_structure"
OPEN_CASES_PATH = OUT_DIR / "phase576_open_cases.jsonl"
OPEN_SPLIT_CASE_PATHS = {
    split: OUT_DIR / f"phase576_{split}_cases.jsonl"
    for split in OPEN_SPLITS
}
SEALED_CASES_PATH = OUT_DIR / "protocol/private/phase576_sealed_cases.jsonl"
SEALED_COMMITMENT_PATH = OUT_DIR / "phase576_sealed_commitment.json"
PROTOCOL_PATH = OUT_DIR / "phase576_frozen_protocol.json"
STATIC_AUDIT_PATH = OUT_DIR / "phase576_static_audit.json"
ENGINEERING_QUALIFICATION_PATH = OUT_DIR / "phase576_engineering_qualification.json"
BEHAVIOR_DECISION_PATHS = {
    split: OUT_DIR / f"phase576_{split}_behavior_decision.json"
    for split in OPEN_SPLITS
}
BEHAVIOR_DECISION_PATH = BEHAVIOR_DECISION_PATHS["discovery"]
DISCOVERY_REGISTRY_PATH = OUT_DIR / "phase576_discovered_structure_registry.json"
CONFIRMATION_DECISION_PATH = OUT_DIR / "phase576_structure_confirmation_decision.json"
HELDOUT_DECISION_PATH = OUT_DIR / "phase576_heldout_replication_decision.json"
SEALED_OPEN_RECEIPT_PATH = OUT_DIR / "phase576_sealed_open_receipt.json"
FREEZE_COMMIT_PATH = OUT_DIR / "phase576_freeze_commit.json"
FREEZE_LOCK_PATH = OUT_DIR.parent / f".{OUT_DIR.name}.freeze.lock"


def trace_model_dir(stage: str, model: str) -> Path:
    if stage not in OPEN_SPLITS or model not in MODELS:
        raise ValueError(f"invalid Phase576 trace coordinate: {stage}/{model}")
    return OUT_DIR / "natural_trace" / stage / model


def trace_manifest_path(stage: str, model: str) -> Path:
    return trace_model_dir(stage, model) / (
        f"phase576_{model}_{stage}_trace_manifest.json"
    )


def trace_receipt_path(stage: str, model: str) -> Path:
    return trace_model_dir(stage, model) / (
        f"phase576_{model}_{stage}_trace_execution_receipt.json"
    )


def trace_contract_path(stage: str, model: str) -> Path:
    return trace_model_dir(stage, model) / "phase576_generation_trace_contract.json"


def trace_completed_path(stage: str, model: str) -> Path:
    return trace_model_dir(stage, model) / "phase576_generation_trace_completed.json"


def trace_stage_receipt_path(stage: str) -> Path:
    if stage not in OPEN_SPLITS:
        raise ValueError(f"invalid Phase576 trace stage: {stage}")
    return OUT_DIR / "natural_trace" / stage / (
        f"phase576_{stage}_trace_execution_receipt.json"
    )

PRIOR_OPEN_CASE_PATHS = (
    ROOT / "tests/gpt5/result/phase556_fruit_encoding/phase556_open_cases.jsonl",
    ROOT / "tests/gpt5/result/phase557_fruit_composite/phase557_open_cases.jsonl",
)
PRIOR_SEALED_OBJECT_IDS = {
    "quince", "lime", "raspberry", "pineapple", "plum", "honeydew",
    "cucumber", "cheese", "traffic_cone", "frog", "grapefruit",
    "blackberry", "papaya", "apricot", "fig", "celery", "sapphire",
}


def obj(
    object_id: str,
    label: str,
    is_fruit: bool,
    category: str,
    category_aliases: tuple[str, ...],
    color: str,
    color_aliases: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "id": object_id,
        "label": label,
        "is_fruit": is_fruit,
        "category": category,
        "category_aliases": list(dict.fromkeys((category, *category_aliases))),
        "color": color,
        "color_aliases": list(dict.fromkeys((color, *color_aliases))),
    }


# Each split contains six fruits and six non-fruits with exactly matched colors.
# The sealed objects are new to the Phase556/557 sealed object lists.
OBJECTS: dict[str, tuple[dict[str, Any], ...]] = {
    "discovery": (
        obj("banana", "banana", True, "fruit", (), "yellow"),
        obj("orange", "orange", True, "fruit", (), "orange"),
        obj("strawberry", "strawberry", True, "fruit", (), "red"),
        obj("blueberry", "blueberry", True, "fruit", (), "blue"),
        obj("avocado", "avocado", True, "fruit", (), "green"),
        obj("coconut", "coconut", True, "fruit", (), "brown"),
        obj("canary", "canary", False, "bird", ("animal",), "yellow"),
        obj("carrot", "carrot", False, "vegetable", (), "orange"),
        obj("ruby", "ruby", False, "mineral", ("gem", "gemstone"), "red"),
        obj("blue_jay", "blue jay", False, "bird", ("animal",), "blue"),
        obj("emerald", "emerald", False, "mineral", ("gem", "gemstone"), "green"),
        obj("chocolate", "chocolate", False, "food", (), "brown"),
    ),
    "confirmation": (
        obj("lemon", "lemon", True, "fruit", (), "yellow"),
        obj("tangerine", "tangerine", True, "fruit", (), "orange"),
        obj("cherry", "cherry", True, "fruit", (), "red"),
        obj("elderberry", "elderberry", True, "fruit", (), "purple"),
        obj("kiwi", "kiwi", True, "fruit", (), "green"),
        obj("date", "date fruit", True, "fruit", (), "brown"),
        obj("sunflower", "sunflower", False, "plant", ("flower",), "yellow"),
        obj("basketball", "basketball", False, "object", ("sports equipment",), "orange"),
        obj("cardinal", "cardinal", False, "bird", ("animal",), "red"),
        obj("amethyst", "amethyst", False, "mineral", ("gem", "gemstone"), "purple"),
        obj("green_pepper", "green pepper", False, "vegetable", (), "green"),
        obj("soil", "soil", False, "material", ("earth",), "brown"),
    ),
    "heldout_recombination": (
        obj("mango", "mango", True, "fruit", (), "yellow"),
        obj("persimmon", "persimmon", True, "fruit", (), "orange"),
        obj("pomegranate", "pomegranate", True, "fruit", (), "red"),
        obj("blackcurrant", "blackcurrant", True, "fruit", (), "purple"),
        obj("guava", "guava", True, "fruit", (), "green"),
        obj("tamarind", "tamarind", True, "fruit", (), "brown"),
        obj("school_bus", "school bus", False, "vehicle", (), "yellow"),
        obj("pumpkin", "pumpkin", False, "vegetable", ("gourd",), "orange"),
        obj("fire_engine", "fire engine", False, "vehicle", (), "red"),
        obj("lavender", "lavender flower", False, "plant", ("flower",), "purple"),
        obj("broccoli", "broccoli", False, "vegetable", (), "green"),
        obj("cinnamon", "cinnamon", False, "spice", ("food",), "brown"),
    ),
    "sealed": (
        obj("starfruit", "starfruit", True, "fruit", (), "yellow"),
        obj("loquat", "loquat", True, "fruit", (), "orange"),
        obj("lychee", "lychee", True, "fruit", (), "red"),
        obj("passionfruit", "passion fruit", True, "fruit", (), "purple"),
        obj("jackfruit", "jackfruit", True, "fruit", (), "green"),
        obj("sapodilla", "sapodilla", True, "fruit", (), "brown"),
        obj("gold", "gold", False, "metal", ("mineral",), "yellow"),
        obj("copper", "copper", False, "metal", (), "orange"),
        obj("firetruck", "firetruck", False, "vehicle", (), "red"),
        obj("violet", "violet flower", False, "plant", ("flower",), "purple"),
        obj("grass", "grass", False, "plant", (), "green"),
        obj("wood", "wood", False, "material", (), "brown"),
    ),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    before_identity = (before.st_size, before.st_mtime_ns)
    after_identity = (after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def current_runtime_identity() -> dict[str, Any]:
    import torch

    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "bitsandbytes": package_version("bitsandbytes"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
    }


def normalize_prompt(text: str) -> str:
    return " ".join(text.casefold().split())


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _parse_jsonl_bytes(payload: bytes, source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise RuntimeError(f"non-object JSONL row at {source}:{line_number}")
        rows.append(row)
    return rows


def prior_open_file_snapshots() -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Read each required historical open file once and bind that exact byte view."""
    identities: list[dict[str, Any]] = []
    row_banks: list[list[dict[str, Any]]] = []
    for path in PRIOR_OPEN_CASE_PATHS:
        if not path.is_file():
            raise RuntimeError(f"required prior-open case file is missing: {path}")
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"prior-open file changed while reading: {path}")
        rows = _parse_jsonl_bytes(payload, path)
        if not rows:
            raise RuntimeError(f"required prior-open case file is empty: {path}")
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        identities.append({
            "path": relative,
            "resolved_path": str(path.resolve(strict=True)),
            "path_is_symlink": path.is_symlink(),
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "row_count": len(rows),
        })
        row_banks.append(rows)
    if len(identities) != len(PRIOR_OPEN_CASE_PATHS):
        raise RuntimeError("prior-open identity cardinality drift")
    return identities, row_banks


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(payload))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jsonl_bytes(rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def model_artifact_identity(
    models: tuple[str, ...] = MODELS,
) -> dict[str, Any]:
    """Recursively seal every regular file under each local model root."""
    identities: dict[str, Any] = {}
    for model in models:
        spec = get_model_spec(model)
        logical = Path(spec.local_dir)
        resolved = logical.resolve(strict=True)
        entries = sorted(
            resolved.rglob("*"),
            key=lambda path: path.relative_to(resolved).as_posix(),
        )
        for path in entries:
            if not path.is_symlink():
                continue
            target = path.resolve(strict=True)
            if target.is_dir():
                raise RuntimeError(
                    f"{model}: nested directory symlinks are not allowed in the "
                    f"frozen model root: {path.relative_to(resolved).as_posix()}"
                )
        files = [path for path in entries if path.is_file()]
        if not files:
            raise RuntimeError(f"{model}: local model root contains no regular files")

        def file_row(path: Path) -> dict[str, Any]:
            before = path.stat()
            digest = sha256_file(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise RuntimeError(f"{model}: artifact changed while sealing: {path}")
            return {
                "relative_path": path.relative_to(resolved).as_posix(),
                "resolved_path": str(path.resolve(strict=True)),
                "path_is_symlink": path.is_symlink(),
                "size_bytes": after.st_size,
                "sha256": digest,
            }

        file_rows = [file_row(path) for path in files]
        weights = [
            row
            for path, row in zip(files, file_rows, strict=True)
            if path.suffix.lower() == ".safetensors"
        ]
        if not weights:
            raise RuntimeError(f"{model}: no local safetensors found")
        non_weight_rows = [
            row
            for path, row in zip(files, file_rows, strict=True)
            if path.suffix.lower() != ".safetensors"
        ]
        payload = {
            "logical_path": str(logical),
            "resolved_path": str(resolved),
            "logical_path_is_symlink": logical.is_symlink(),
            "repo_id": spec.repo_id,
            "artifact_inventory_mode": "recursive_all_regular_files.v1",
            "nested_directory_symlinks_allowed": False,
            "artifact_file_count": len(file_rows),
            "artifact_total_bytes": sum(row["size_bytes"] for row in file_rows),
            "weight_files": weights,
            "weight_file_count": len(weights),
            "weight_total_bytes": sum(row["size_bytes"] for row in weights),
            "tokenizer_and_config_files": non_weight_rows,
            "non_weight_file_count": len(non_weight_rows),
            "non_weight_total_bytes": sum(
                row["size_bytes"] for row in non_weight_rows
            ),
        }
        identities[model] = {
            **payload,
            "identity_sha256": stable_hash(payload),
        }
    return identities


def stage_source_seals() -> dict[str, dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "tests/glm5/phase576_gpt5_fruit_behavior.py",
        ROOT / "tests/glm5/phase576_gpt5_fruit_behavior_analysis.py",
        ROOT / "tests/glm5/phase576_gpt5_fruit_engineering_qualification.py",
        ROOT / "tests/glm5/phase576_gpt5_fruit_natural_trace.py",
        ROOT / "tests/glm5/phase983_cross_model_engine.py",
        ROOT / "tests/gpt5/phase548_shared_attention_compute_protocol.py",
        ROOT / "tests/gpt5/model_registry.py",
    )
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            raise RuntimeError(f"missing Phase576 stage source before freeze: {path}")
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        result[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def verify_frozen_source_seals(frozen: dict[str, Any]) -> None:
    current = stage_source_seals()
    if frozen.get("stage_source_seals") != current:
        raise RuntimeError("Phase576 executable source/dependency seal drift")


def verify_frozen_model_artifacts(
    frozen: dict[str, Any], models: tuple[str, ...] = MODELS
) -> None:
    current = model_artifact_identity(models)
    expected = {
        model: frozen["model_artifact_identities"][model] for model in models
    }
    if current != expected:
        raise RuntimeError("Phase576 full model/tokenizer artifact identity drift")


def single_model_trace_authorized(stage: str, model: str, stage_pass: Any) -> bool:
    return stage in OPEN_SPLITS and model in MODELS and stage_pass is True


def cross_model_observational_comparison_authorized(
    stage: str,
    model_stage_pass: dict[str, Any],
) -> bool:
    """Derive authorization to compare complete same-stage observations."""
    if stage not in OPEN_SPLITS or not isinstance(model_stage_pass, dict):
        return False
    if set(model_stage_pass) != set(MODELS):
        return False
    return all(model_stage_pass[model] is True for model in MODELS)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.casefold())
    )


def _require_exact_keys(payload: dict[str, Any], expected: set[str], label: str) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise RuntimeError(f"{label} schema keys differ; missing={missing}, extra={extra}")


def _verify_analysis_source_identity(
    identity: Any,
    frozen: dict[str, Any],
) -> None:
    if not isinstance(identity, dict):
        raise RuntimeError("analysis source identity must be an object")
    _require_exact_keys(identity, {"path", "size_bytes", "sha256"}, "analysis source")
    relative = identity["path"]
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise RuntimeError("analysis source path must be repository-relative")
    path = (ROOT / relative).resolve(strict=True)
    analysis_root = (ROOT / "tests/glm5").resolve(strict=True)
    try:
        path.relative_to(analysis_root)
    except ValueError as exc:
        raise RuntimeError("analysis source is outside tests/glm5") from exc
    normalized = str(path.relative_to(ROOT)).replace("\\", "/")
    if normalized != relative.replace("\\", "/"):
        raise RuntimeError("analysis source path resolves through an alias or symlink")
    if normalized in frozen["stage_source_seals"]:
        raise RuntimeError(
            "post-trace discovery analysis source was included in initial source seals"
        )
    if (
        not path.is_file()
        or identity["size_bytes"] != path.stat().st_size
        or not _is_sha256(identity["sha256"])
        or identity["sha256"] != sha256_file(path)
    ):
        raise RuntimeError("post-trace analysis source identity drift")
    source_text = path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(path))
    allowed_import_roots = set(POST_TRACE_ALLOWED_IMPORT_ROOTS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            if not roots <= allowed_import_roots:
                raise RuntimeError("post-trace source has an unsealed import dependency")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if node.level != 0 or root not in allowed_import_roots:
                raise RuntimeError("post-trace source has a relative/unsealed import")
    forbidden_sealed_literals = (
        "phase576_sealed_cases", "protocol/private", "SEALED_CASES_PATH",
    )
    if any(value in source_text for value in forbidden_sealed_literals):
        raise RuntimeError("post-trace source names a forbidden sealed payload")


def _load_post_trace_analysis_source(identity: dict[str, Any]) -> Any:
    relative = identity["path"]
    path = (ROOT / relative).resolve(strict=True)
    module_name = f"phase576_post_trace_{identity['sha256']}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen post-trace analysis source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace_behavior_rows_path(stage: str, model: str) -> Path:
    return OUT_DIR / "open_behavior" / stage / (
        f"phase576_{model}_{stage}_behavior_rows.jsonl.gz"
    )


def _trace_expected_behavior_capsules(
    stage: str,
    model: str,
    behavior_decision: dict[str, Any],
) -> dict[str, dict[str, list[int]]]:
    path = _trace_behavior_rows_path(stage, model)
    reports = behavior_decision.get("reports", [])
    report = next(
        (item for item in reports if item.get("model") == model), None
    )
    if (
        report is None
        or not path.is_file()
        or report.get("behavior_rows_sha256") != sha256_file(path)
    ):
        raise RuntimeError(f"{stage}/{model}: behavior-row trace dependency drift")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    repeat1 = [row for row in rows if row.get("execution_repeat") == "repeat1"]
    expected_case_ids = {
        row["case_id"] for row in read_jsonl(OPEN_SPLIT_CASE_PATHS[stage])
    }
    actual_case_ids = [row.get("case_id") for row in repeat1]
    if (
        len(repeat1) != 336
        or len(actual_case_ids) != len(set(actual_case_ids))
        or set(actual_case_ids) != expected_case_ids
    ):
        raise RuntimeError(f"{stage}/{model}: behavior repeat1 registry is not exact")
    tokenizer = tokenizer_for(model)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(pad_token_id, int) or isinstance(pad_token_id, bool):
        raise RuntimeError(f"{stage}/{model}: tokenizer pad identity is invalid")
    generation: dict[str, dict[str, list[int]]] = {}
    for row in repeat1:
        content = row.get("generated_token_ids_before_eos")
        full_suffix = row.get("full_generated_suffix_token_ids")
        first_eos = row.get("first_eos_token_id")
        eos_seen = row.get("eos_seen") is True
        if (
            row.get("schema_version") != "phase576_open_behavior_row.v2"
            or row.get("phase_id") != PHASE
            or row.get("model") != model
            or row.get("stage") != stage
            or row.get("split") != stage
            or row.get("sealed_model_access") is not False
            or not isinstance(content, list)
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in content
            )
            or eos_seen != (isinstance(first_eos, int) and not isinstance(first_eos, bool))
            or not isinstance(full_suffix, list)
            or not full_suffix
            or len(full_suffix) > MAX_NEW_TOKENS
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in full_suffix
            )
            or row.get("generation_suffix_width") != len(full_suffix)
        ):
            raise RuntimeError(f"{stage}/{model}: behavior repeat1 capsule is invalid")
        capsule = list(content) + ([int(first_eos)] if eos_seen else [])
        if (
            not capsule
            or len(capsule) > MAX_NEW_TOKENS
            or (not eos_seen and len(capsule) != MAX_NEW_TOKENS)
            or full_suffix[:len(capsule)] != capsule
            or (
                eos_seen
                and (
                    row.get("first_eos_index") != len(content)
                    or row.get("post_eos_token_ids") != full_suffix[len(capsule):]
                    or any(
                        value != pad_token_id for value in full_suffix[len(capsule):]
                    )
                    or row.get("post_eos_tokens_all_pad") is not True
                )
            )
            or (not eos_seen and full_suffix != capsule)
        ):
            raise RuntimeError(f"{stage}/{model}: behavior capsule length is invalid")
        generation[row["case_id"]] = {
            "capsule": capsule,
            "full_suffix": list(full_suffix),
        }
    del tokenizer
    gc.collect()
    return generation


def _independent_trace_prompt_registry(
    model: str,
    cases: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    tokenizer = tokenizer_for(model)
    if getattr(tokenizer, "is_fast", False) is not True:
        raise RuntimeError(f"{model}: independent trace locator requires fast offsets")
    registry: dict[str, dict[str, Any]] = {}
    for case in cases:
        raw_prompt = case["raw_prompt"]
        rendered = render_chat(tokenizer, model, raw_prompt)
        raw_start = rendered.find(raw_prompt)
        if raw_start < 0 or rendered.find(raw_prompt, raw_start + 1) >= 0:
            raise RuntimeError(f"{case['case_id']}: raw prompt rendering is not unique")
        encoded = tokenizer(
            rendered,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )
        input_ids = [int(value) for value in encoded["input_ids"]]
        offsets = [
            (int(pair[0]), int(pair[1])) for pair in encoded["offset_mapping"]
        ]
        attention = [int(value) for value in encoded["attention_mask"]]
        if len(input_ids) != len(offsets) or attention != [1] * len(input_ids):
            raise RuntimeError(f"{case['case_id']}: independent offset mapping failed")
        spans = case["raw_role_char_spans"]
        role_positions: list[int] = []
        for role in ("focus", "comparison", "query_anchor"):
            span = spans[role]
            if role == "comparison" and span is None:
                role_positions.append(0)
                continue
            if (
                not isinstance(span, dict)
                or set(span) != {"start", "end", "text"}
                or not isinstance(span["start"], int)
                or isinstance(span["start"], bool)
                or not isinstance(span["end"], int)
                or isinstance(span["end"], bool)
                or not isinstance(span["text"], str)
                or span["start"] < 0
                or span["end"] <= span["start"]
                or span["end"] > len(raw_prompt)
                or raw_prompt[span["start"]:span["end"]].casefold()
                != span["text"].casefold()
            ):
                raise RuntimeError(f"{case['case_id']}: invalid frozen {role} span")
            absolute_start = raw_start + span["start"]
            absolute_end = raw_start + span["end"]
            overlaps = [
                index for index, (start, end) in enumerate(offsets)
                if end > start and start < absolute_end and end > absolute_start
            ]
            if not overlaps:
                raise RuntimeError(f"{case['case_id']}: {role} span has no token")
            role_positions.append(overlaps[-1])
        role_positions.append(len(input_ids) - 1)
        registry[case["case_id"]] = {
            "rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
            "rendered_prompt_token_ids": input_ids,
            "unpad_prompt_role_positions": role_positions,
            "prompt_role_mask": [
                True,
                case["comparison_object_label"] is not None,
                True,
                True,
            ],
        }
    del tokenizer
    gc.collect()
    return registry


def _verify_trace_shard_closure(
    stage: str,
    model: str,
    manifest: dict[str, Any],
    frozen: dict[str, Any],
    behavior_decision: dict[str, Any],
    qualification: dict[str, Any],
) -> None:
    """Reopen every persisted shard and verify the full raw-evidence closure."""
    import torch

    expected_cases = read_jsonl(OPEN_SPLIT_CASE_PATHS[stage])
    case_by_id = {row["case_id"]: row for row in expected_cases}
    expected_case_ids = [row["case_id"] for row in expected_cases]
    expected_unit_ids = {row["independent_unit_id"] for row in expected_cases}
    expected_generation = _trace_expected_behavior_capsules(
        stage, model, behavior_decision,
    )
    expected_prompts = _independent_trace_prompt_registry(model, expected_cases)
    expected_feedback_slots = [
        f"generated_feedback_token_{index:02d}"
        for index in range(MAX_NEW_TOKENS - 1)
    ]
    qualification_reports = qualification.get("reports", [])
    qualification_report = next(
        (item for item in qualification_reports if item.get("model") == model), None
    )
    if qualification_report is None:
        raise RuntimeError(f"{stage}/{model}: engineering report is missing")
    expected_hidden_count = qualification_report.get("hidden_state_count")
    expected_hidden_size = qualification_report.get("hidden_size")
    shards = manifest.get("shards")
    if (
        not isinstance(shards, list)
        or len(shards) != 336 // TRACE_BATCH_SIZE
        or manifest.get("shard_count") != len(shards)
    ):
        raise RuntimeError(f"{stage}/{model}: trace shard registry cardinality drift")

    shard_keys = {
        "path", "case_ids", "size_bytes", "sha256", "batch_size",
        "prompt_padded_width", "generation_iteration_count",
        "hidden_state_count", "hidden_size", "runtime_dtype", "stored_dtype",
        "prefill_position_count", "feedback_slot_count",
        "executed_feedback_position_count",
    }
    payload_keys = {
        "schema_version", "phase_id", "model", "stage", "case_rows",
        "prompt_role_labels", "feedback_slots", "prefill_residual",
        "feedback_residual", "prefill_attention_mask", "tensor_identity",
        "all_layers", "all_executed_residual_positions",
        "batch_absorbing_eos_and_pad_feedback_positions_included",
        "complete_component_trajectory", "teacher_forced_replay", "causal",
        "sealed_model_access",
    }
    row_keys = {
        "case_id", "independent_unit_id", "relation", "interface",
        "surface_id", "order", "rendered_prompt_sha256",
        "rendered_prompt_token_ids", "unpad_prompt_role_positions",
        "padded_prompt_role_positions", "prompt_role_mask", "behavior_repeat",
        "generated_capsule_token_ids", "full_generated_suffix_token_ids",
        "feedback_token_ids", "feedback_mask",
    }
    identity_keys = shard_keys - {"path", "case_ids", "size_bytes", "sha256"}
    model_root = trace_model_dir(stage, model)
    if model_root.is_symlink() or model_root.resolve(strict=True) != model_root.absolute():
        raise RuntimeError(f"{stage}/{model}: trace model root is aliased")
    observed_case_ids: list[str] = []
    observed_paths: set[str] = set()
    for shard_index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise RuntimeError(f"{stage}/{model}: shard registry entry is not an object")
        _require_exact_keys(shard, shard_keys, f"{stage}/{model} shard registry")
        expected_name = f"phase576_generation_trace_shard_{shard_index:04d}.pt"
        relative = shard["path"]
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise RuntimeError(f"{stage}/{model}: invalid shard relative path")
        shard_path = OUT_DIR / relative
        normalized = str(shard_path.relative_to(OUT_DIR)).replace("\\", "/")
        if normalized != relative.replace("\\", "/") or relative in observed_paths:
            raise RuntimeError(f"{stage}/{model}: shard path registry is not exact")
        observed_paths.add(relative)
        if (
            shard_path.name != expected_name
            or shard_path.parent != model_root
            or shard_path.is_symlink()
            or not shard_path.is_file()
            or shard_path.resolve(strict=True).parent != model_root.resolve(strict=True)
            or not isinstance(shard["size_bytes"], int)
            or isinstance(shard["size_bytes"], bool)
            or shard["size_bytes"] != shard_path.stat().st_size
            or not _is_sha256(shard["sha256"])
            or shard["sha256"] != sha256_file(shard_path)
        ):
            raise RuntimeError(f"{stage}/{model}: shard file identity mismatch")
        case_ids = shard["case_ids"]
        if (
            not isinstance(case_ids, list)
            or len(case_ids) != TRACE_BATCH_SIZE
            or len(set(case_ids)) != len(case_ids)
            or any(case_id not in case_by_id for case_id in case_ids)
        ):
            raise RuntimeError(f"{stage}/{model}: shard case registry is invalid")
        observed_case_ids.extend(case_ids)
        identity = {key: shard[key] for key in identity_keys}
        if (
            identity["batch_size"] != len(case_ids)
            or identity["hidden_state_count"] != expected_hidden_count
            or identity["hidden_size"] != expected_hidden_size
            or identity["runtime_dtype"] != "torch.bfloat16"
            or identity["stored_dtype"] != "torch.bfloat16"
            or identity["prefill_position_count"] != identity["prompt_padded_width"]
            or identity["feedback_slot_count"] != MAX_NEW_TOKENS - 1
            or identity["executed_feedback_position_count"]
            != identity["generation_iteration_count"] - 1
            or not isinstance(identity["prompt_padded_width"], int)
            or identity["prompt_padded_width"] <= 0
            or not isinstance(identity["generation_iteration_count"], int)
            or not 1 <= identity["generation_iteration_count"] <= MAX_NEW_TOKENS
        ):
            raise RuntimeError(f"{stage}/{model}: shard tensor identity is invalid")

        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise RuntimeError(f"{stage}/{model}: trace shard payload is not an object")
        _require_exact_keys(payload, payload_keys, f"{stage}/{model} shard payload")
        if (
            payload["schema_version"] != "phase576_generation_residual_shard.v2"
            or payload["phase_id"] != PHASE
            or payload["model"] != model
            or payload["stage"] != stage
            or payload["prompt_role_labels"]
            != frozen["trace_policy"]["role_labels"]
            or payload["feedback_slots"] != expected_feedback_slots
            or payload["tensor_identity"] != identity
            or payload["all_layers"] is not True
            or payload["all_executed_residual_positions"] is not True
            or payload[
                "batch_absorbing_eos_and_pad_feedback_positions_included"
            ] is not True
            or payload["complete_component_trajectory"] is not False
            or payload["teacher_forced_replay"] is not False
            or payload["causal"] is not False
            or payload["sealed_model_access"] is not False
        ):
            raise RuntimeError(f"{stage}/{model}: shard payload contract mismatch")
        case_rows = payload["case_rows"]
        masks = payload["prefill_attention_mask"]
        prefill = payload["prefill_residual"]
        feedback = payload["feedback_residual"]
        expected_prefill_shape = (
            len(case_ids), expected_hidden_count,
            identity["prefill_position_count"], expected_hidden_size,
        )
        expected_feedback_shape = (
            len(case_ids), expected_hidden_count,
            MAX_NEW_TOKENS - 1, expected_hidden_size,
        )
        if (
            not isinstance(case_rows, list)
            or len(case_rows) != len(case_ids)
            or not isinstance(masks, list)
            or len(masks) != len(case_ids)
            or not isinstance(prefill, torch.Tensor)
            or tuple(prefill.shape) != expected_prefill_shape
            or prefill.dtype != torch.bfloat16
            or prefill.device.type != "cpu"
            or not prefill.is_contiguous()
            or not isinstance(feedback, torch.Tensor)
            or tuple(feedback.shape) != expected_feedback_shape
            or feedback.dtype != torch.bfloat16
            or feedback.device.type != "cpu"
            or not feedback.is_contiguous()
            or not bool(torch.isfinite(prefill.float()).all().item())
            or not bool(torch.isfinite(feedback.float()).all().item())
        ):
            raise RuntimeError(f"{stage}/{model}: shard tensor payload is invalid")
        full_suffix_lengths: list[int] = []
        for row_index, (case_id, row) in enumerate(zip(case_ids, case_rows)):
            if not isinstance(row, dict):
                raise RuntimeError(f"{stage}/{model}: shard case row is not an object")
            _require_exact_keys(row, row_keys, f"{stage}/{model}/{case_id} shard row")
            case = case_by_id[case_id]
            expected_prompt = expected_prompts[case_id]
            tokens = row["rendered_prompt_token_ids"]
            unpadded_roles = row["unpad_prompt_role_positions"]
            padded_roles = row["padded_prompt_role_positions"]
            role_mask = row["prompt_role_mask"]
            capsule = row["generated_capsule_token_ids"]
            full_suffix = row["full_generated_suffix_token_ids"]
            feedback_ids = row["feedback_token_ids"]
            feedback_mask = row["feedback_mask"]
            padding = identity["prefill_position_count"] - len(tokens)
            expected_attention = [False] * padding + [True] * len(tokens)
            if (
                row["case_id"] != case_id
                or row["independent_unit_id"] != case["independent_unit_id"]
                or row["relation"] != case["relation"]
                or row["interface"] != case["interface"]
                or row["surface_id"] != case["surface_id"]
                or row["order"] != case["order"]
                or row["behavior_repeat"] != "repeat1"
                or row["rendered_prompt_sha256"]
                != expected_prompt["rendered_prompt_sha256"]
                or tokens != expected_prompt["rendered_prompt_token_ids"]
                or padding < 0
                or masks[row_index] != expected_attention
                or unpadded_roles
                != expected_prompt["unpad_prompt_role_positions"]
                or padded_roles != [padding + value for value in unpadded_roles]
                or role_mask != expected_prompt["prompt_role_mask"]
                or capsule != expected_generation[case_id]["capsule"]
                or full_suffix != expected_generation[case_id]["full_suffix"]
                or feedback_ids != full_suffix[:-1]
                or feedback_mask != [
                    index < identity["executed_feedback_position_count"]
                    for index in range(MAX_NEW_TOKENS - 1)
                ]
            ):
                raise RuntimeError(f"{stage}/{model}/{case_id}: shard row mismatch")
            if padding and bool(torch.count_nonzero(prefill[row_index, :, :padding, :]).item()):
                raise RuntimeError(f"{stage}/{model}/{case_id}: nonzero prefill padding")
            for feedback_index, valid in enumerate(feedback_mask):
                if not valid and bool(torch.count_nonzero(
                    feedback[row_index, :, feedback_index, :]
                ).item()):
                    raise RuntimeError(
                        f"{stage}/{model}/{case_id}: nonzero feedback padding"
                    )
            full_suffix_lengths.append(len(full_suffix))
        if (
            len(set(full_suffix_lengths)) != 1
            or identity["generation_iteration_count"] != full_suffix_lengths[0]
        ):
            raise RuntimeError(f"{stage}/{model}: generation iteration count drift")
        del payload, prefill, feedback
        gc.collect()
    if (
        observed_case_ids != expected_case_ids
        or len(set(observed_case_ids)) != 336
        or manifest.get("case_count") != 336
        or manifest.get("independent_unit_count") != len(expected_unit_ids)
        or len(expected_unit_ids) != 36
    ):
        raise RuntimeError(f"{stage}/{model}: trace denominator closure failed")


def _verify_behavior_decision_for_trace(
    stage: str,
    behavior_decision: dict[str, Any],
    frozen: dict[str, Any],
    qualification_sha: str,
    qualification_receipt_sha: str,
    qualification: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    expected_keys = {
        "schema_version", "phase_id", "created_at_utc", "stage",
        "models_in_required_execution_order", "reports", "qualified_models",
        "single_model_trace_authorized_models",
        "cross_model_observational_comparison_authorized",
        "cross_model_observational_comparison_scope", "blocked_models",
        "stage_cases_sha256", "behavior_execution_receipt_sha256",
        "engineering_qualification_sha256",
        "engineering_execution_receipt_sha256", "runtime_identity",
        "protocol_sha256", "analysis_source_sha256", "analysis_source_seal",
        "analysis_unit_definition", "trace_selection_rule",
        "cross_model_observational_comparison_rule", "sealed_model_access",
        "internal_intervention_authorized", "mechanism_claim_authorized",
    }
    _require_exact_keys(behavior_decision, expected_keys, f"{stage} behavior decision")
    reports = behavior_decision["reports"]
    if (
        not isinstance(reports, list)
        or [item.get("model") for item in reports] != list(MODELS)
    ):
        raise RuntimeError(f"{stage}: behavior report registry is not exact")
    case_ids = {
        row["case_id"] for row in read_jsonl(OPEN_SPLIT_CASE_PATHS[stage])
    }
    report_by_model: dict[str, dict[str, Any]] = {}
    for model, report in zip(MODELS, reports):
        passed = report.get("behavior_gate_pass") is True
        trace_ids = report.get("trace_case_ids")
        report_checks = (
            report.get("model") == model,
            report.get("stage") == stage,
            isinstance(report.get("behavior_gate_pass"), bool),
            report.get("single_model_trace_authorized") is passed,
            report.get("internal_trace_authorized") is passed,
            isinstance(trace_ids, list),
            (
                len(trace_ids) == 336
                and len(set(trace_ids)) == 336
                and set(trace_ids) == case_ids
            ) if passed else trace_ids == [],
            report.get("internal_intervention_authorized") is False,
            report.get("mechanism_claim_authorized") is False,
            report.get("sealed_model_access") is False,
        )
        if not all(report_checks):
            raise RuntimeError(f"{stage}/{model}: behavior trace authorization drift")
        report_by_model[model] = report
    qualified = [
        model for model in MODELS
        if report_by_model[model]["behavior_gate_pass"] is True
    ]
    behavior_receipt_path = OUT_DIR / "open_behavior" / stage / (
        f"phase576_{stage}_behavior_execution_receipt.json"
    )
    decision_checks = (
        behavior_decision["schema_version"] == "phase576_behavior_decision.v2",
        behavior_decision["phase_id"] == PHASE,
        behavior_decision["stage"] == stage,
        behavior_decision["models_in_required_execution_order"] == list(MODELS),
        behavior_decision["qualified_models"] == qualified,
        behavior_decision["single_model_trace_authorized_models"] == qualified,
        behavior_decision["blocked_models"]
        == [model for model in MODELS if model not in qualified],
        behavior_decision["cross_model_observational_comparison_authorized"]
        is (qualified == list(MODELS)),
        behavior_decision["stage_cases_sha256"]
        == frozen["open_case_sha256_by_split"][stage],
        behavior_receipt_path.is_file(),
        behavior_decision["behavior_execution_receipt_sha256"]
        == sha256_file(behavior_receipt_path),
        behavior_decision["engineering_qualification_sha256"]
        == qualification_sha,
        behavior_decision["engineering_execution_receipt_sha256"]
        == qualification_receipt_sha,
        behavior_decision["runtime_identity"] == qualification["runtime_identity"],
        behavior_decision["protocol_sha256"] == sha256_file(PROTOCOL_PATH),
        behavior_decision["analysis_source_sha256"]
        == frozen["stage_source_seals"][
            "tests/glm5/phase576_gpt5_fruit_behavior_analysis.py"
        ]["sha256"],
        behavior_decision["analysis_source_seal"]
        == frozen["stage_source_seals"][
            "tests/glm5/phase576_gpt5_fruit_behavior_analysis.py"
        ],
        behavior_decision["sealed_model_access"] is False,
        behavior_decision["internal_intervention_authorized"] is False,
        behavior_decision["mechanism_claim_authorized"] is False,
    )
    if not all(decision_checks):
        raise RuntimeError(f"{stage}: behavior decision trace contract mismatch")
    return report_by_model


def _verify_trace_stage_receipt(
    stage: str,
    frozen: dict[str, Any],
    behavior_decision: dict[str, Any],
    qualification: dict[str, Any],
    qualification_sha: str,
    qualification_receipt_sha: str,
    *,
    expected_candidate_specification_sha256: str | None,
    expected_discovery_registry_sha256: str | None,
    expected_confirmation_decision_sha256: str | None,
    pass_map: dict[str, bool],
) -> None:
    path = trace_stage_receipt_path(stage)
    if not path.is_file():
        raise RuntimeError(f"{stage}: global trace execution receipt is missing")
    receipt = read_json(path)
    expected_keys = {
        "schema_version", "phase_id", "created_at_utc", "stage",
        "models_considered_in_required_order", "attempts", "qualified_models",
        "completed_models", "failed_models", "not_attempted_models",
        "fatal_error", "terminal_status", "behavior_decision_sha256",
        "stage_cases_sha256", "protocol_sha256", "trace_source_sha256",
        "candidate_specification_sha256", "discovery_registry_sha256",
        "confirmation_decision_sha256", "engineering_qualification_sha256",
        "engineering_execution_receipt_sha256", "runtime_identity",
        "trace_manifest_sha256_by_model", "trace_receipt_sha256_by_model",
        "all_models_behavior_qualified", "all_models_trace_complete",
        "single_model_observation_allowed",
        "cross_model_observational_comparison_authorized",
        "cross_model_common_structure_claim_authorized",
        "internal_intervention_authorized", "mechanism_claim_authorized",
        "final_pytorch_cuda_allocated", "final_pytorch_cuda_reserved",
        "sealed_model_access",
    }
    _require_exact_keys(receipt, expected_keys, f"{stage} global trace receipt")

    qualified = [
        model for model, report in zip(MODELS, behavior_decision["reports"])
        if report["behavior_gate_pass"] is True
    ]
    completed = [model for model in MODELS if pass_map[model]]
    expected_attempts: list[dict[str, Any]] = []
    expected_failures: list[dict[str, Any]] = []
    expected_manifest_hashes: dict[str, str] = {}
    expected_model_receipt_hashes: dict[str, str] = {}
    for model in MODELS:
        model_receipt_path = trace_receipt_path(stage, model)
        model_receipt = read_json(model_receipt_path)
        status = model_receipt["trace_attempt_status"]
        if status == "failed":
            failed_path = trace_model_dir(stage, model) / (
                "phase576_generation_trace_failed.json"
            )
            failed = read_json(failed_path)
            expected_attempts.append({
                "model": model,
                "status": "failed",
                "error_type": failed["error_type"],
            })
            expected_failures.append({
                "model": model,
                "error_type": failed["error_type"],
                "error": failed["error"],
            })
        else:
            expected_attempts.append({"model": model, "status": status})
        if status == "complete":
            expected_manifest_hashes[model] = sha256_file(
                trace_manifest_path(stage, model)
            )
        expected_model_receipt_hashes[model] = sha256_file(model_receipt_path)

    all_behavior_qualified = qualified == list(MODELS)
    all_trace_complete = completed == list(MODELS)
    checks = (
        receipt["schema_version"] == TRACE_STAGE_RECEIPT_SCHEMA_VERSION,
        receipt["phase_id"] == PHASE,
        isinstance(receipt["created_at_utc"], str)
        and bool(receipt["created_at_utc"]),
        receipt["stage"] == stage,
        receipt["models_considered_in_required_order"] == list(MODELS),
        receipt["attempts"] == expected_attempts,
        receipt["qualified_models"] == qualified,
        receipt["completed_models"] == completed,
        receipt["failed_models"] == expected_failures,
        receipt["not_attempted_models"] == [],
        receipt["fatal_error"] is None,
        receipt["terminal_status"] == "complete",
        receipt["behavior_decision_sha256"]
        == sha256_file(BEHAVIOR_DECISION_PATHS[stage]),
        receipt["stage_cases_sha256"]
        == frozen["open_case_sha256_by_split"][stage],
        receipt["protocol_sha256"] == sha256_file(PROTOCOL_PATH),
        receipt["trace_source_sha256"]
        == frozen["stage_source_seals"][
            "tests/glm5/phase576_gpt5_fruit_natural_trace.py"
        ]["sha256"],
        receipt["candidate_specification_sha256"]
        == expected_candidate_specification_sha256,
        receipt["discovery_registry_sha256"]
        == expected_discovery_registry_sha256,
        receipt["confirmation_decision_sha256"]
        == expected_confirmation_decision_sha256,
        receipt["engineering_qualification_sha256"] == qualification_sha,
        receipt["engineering_execution_receipt_sha256"]
        == qualification_receipt_sha,
        receipt["runtime_identity"] == qualification["runtime_identity"],
        receipt["trace_manifest_sha256_by_model"] == expected_manifest_hashes,
        receipt["trace_receipt_sha256_by_model"]
        == expected_model_receipt_hashes,
        receipt["all_models_behavior_qualified"] is all_behavior_qualified,
        receipt["all_models_trace_complete"] is all_trace_complete,
        receipt["single_model_observation_allowed"] is True,
        receipt["cross_model_observational_comparison_authorized"]
        is (all_behavior_qualified and all_trace_complete),
        receipt["cross_model_common_structure_claim_authorized"] is False,
        receipt["internal_intervention_authorized"] is False,
        receipt["mechanism_claim_authorized"] is False,
        receipt["final_pytorch_cuda_allocated"] == 0,
        isinstance(receipt["final_pytorch_cuda_reserved"], int),
        not isinstance(receipt["final_pytorch_cuda_reserved"], bool),
        receipt["final_pytorch_cuda_reserved"] >= 0,
        receipt["sealed_model_access"] is False,
    )
    if not all(checks):
        raise RuntimeError(f"{stage}: global trace execution receipt mismatch")


def _verify_trace_artifact_registry(
    stage: str,
    registry: Any,
    frozen: dict[str, Any],
    *,
    expected_candidate_specification_sha256: str | None,
    expected_discovery_registry_sha256: str | None,
    expected_confirmation_decision_sha256: str | None = None,
) -> dict[str, bool]:
    if not isinstance(registry, dict) or set(registry) != set(MODELS):
        raise RuntimeError(f"{stage}: trace artifact model registry is not exact")
    behavior_path = BEHAVIOR_DECISION_PATHS[stage]
    if not behavior_path.is_file():
        raise RuntimeError(f"{stage}: behavior decision is missing")
    behavior_sha = sha256_file(behavior_path)
    behavior_decision = read_json(behavior_path)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    stage_case_sha = frozen["open_case_sha256_by_split"][stage]
    qualification_path = ENGINEERING_QUALIFICATION_PATH
    if not qualification_path.is_file():
        raise RuntimeError("Phase576 engineering qualification is missing")
    qualification_sha = sha256_file(qualification_path)
    qualification = read_json(qualification_path)
    qualification_receipt_relative = qualification.get("execution_receipt_path")
    if (
        not isinstance(qualification_receipt_relative, str)
        or Path(qualification_receipt_relative).is_absolute()
    ):
        raise RuntimeError("engineering execution receipt path is invalid")
    qualification_receipt_path = (ROOT / qualification_receipt_relative).resolve(
        strict=True
    )
    try:
        qualification_receipt_path.relative_to(OUT_DIR.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError("engineering execution receipt is outside Phase576") from exc
    qualification_receipt_sha = sha256_file(qualification_receipt_path)
    qualification_receipt = read_json(qualification_receipt_path)
    engineering_execution_root = (
        OUT_DIR / "engineering_qualification_execution"
    ).resolve(strict=True)
    engineering_stage_start_path = (
        engineering_execution_root / "stage_start.json"
    )
    engineering_stage_start_regular = (
        engineering_stage_start_path.is_file()
        and not engineering_stage_start_path.is_symlink()
        and engineering_stage_start_path.resolve(strict=True).parent
        == engineering_execution_root
    )
    engineering_stage_start = (
        read_json(engineering_stage_start_path)
        if engineering_stage_start_regular else {}
    )
    qualification_checks = (
        qualification.get("schema_version") == "phase576_engineering_qualification.v2",
        qualification.get("phase_id") == PHASE,
        qualification.get("terminal_status") == "complete",
        qualification.get("passed") is True,
        qualification.get("models_in_execution_order") == list(MODELS),
        qualification.get("qualified_models") == list(MODELS),
        qualification.get("protocol_sha256") == protocol_sha,
        qualification.get("execution_receipt_sha256")
        == qualification_receipt_sha,
        qualification.get("model_artifact_identities")
        == frozen["model_artifact_identities"],
        qualification.get("stage_source_seals") == frozen["stage_source_seals"],
        qualification.get("runtime_identity") == current_runtime_identity(),
        qualification_receipt.get("run_id") == qualification.get("run_id"),
        qualification_receipt.get("execution_contract")
        == qualification.get("execution_contract"),
        qualification_receipt.get("execution_contract_sha256")
        == qualification.get("execution_contract_sha256"),
        engineering_stage_start_regular,
        engineering_stage_start.get("schema_version")
        == "phase576_repeat_forward_stage_start.v1",
        engineering_stage_start.get("phase_id") == PHASE,
        engineering_stage_start.get("run_id") == qualification.get("run_id"),
        engineering_stage_start.get("execution_contract")
        == qualification.get("execution_contract"),
        engineering_stage_start.get("execution_contract_sha256")
        == qualification.get("execution_contract_sha256"),
        engineering_stage_start.get("qualification_source_sha256")
        == qualification.get("qualification_source_sha256"),
        qualification_receipt.get("stage_start_sha256")
        == (
            sha256_file(engineering_stage_start_path)
            if engineering_stage_start_regular else None
        ),
        [item.get("model") for item in qualification.get("reports", [])]
        == list(MODELS),
        behavior_decision.get("engineering_qualification_sha256")
        == qualification_sha,
        behavior_decision.get("engineering_execution_receipt_sha256")
        == qualification_receipt_sha,
        behavior_decision.get("runtime_identity")
        == qualification.get("runtime_identity"),
    )
    if not all(qualification_checks):
        raise RuntimeError("Phase576 engineering/behavior trace chain mismatch")
    behavior_report_by_model = _verify_behavior_decision_for_trace(
        stage,
        behavior_decision,
        frozen,
        qualification_sha,
        qualification_receipt_sha,
        qualification,
    )
    pass_map: dict[str, bool] = {}
    entry_keys = {
        "stage_pass",
        "trace_manifest_sha256",
        "trace_receipt_sha256",
    }
    for model in MODELS:
        entry = registry[model]
        if not isinstance(entry, dict):
            raise RuntimeError(f"{stage}/{model}: trace registry entry is not an object")
        _require_exact_keys(entry, entry_keys, f"{stage}/{model} trace registry")
        stage_pass = entry["stage_pass"]
        if not isinstance(stage_pass, bool):
            raise RuntimeError(f"{stage}/{model}: stage_pass is not boolean")
        pass_map[model] = stage_pass

        receipt_path = trace_receipt_path(stage, model)
        if (
            not receipt_path.is_file()
            or not _is_sha256(entry["trace_receipt_sha256"])
            or entry["trace_receipt_sha256"] != sha256_file(receipt_path)
        ):
            raise RuntimeError(f"{stage}/{model}: trace receipt identity mismatch")
        receipt = read_json(receipt_path)
        receipt_keys = {
            "schema_version", "phase_id", "created_at_utc", "model", "stage",
            "stage_pass", "trace_attempt_status", "protocol_sha256",
            "stage_cases_sha256", "behavior_decision_sha256",
            "candidate_specification_sha256", "discovery_registry_sha256",
            "confirmation_decision_sha256", "engineering_qualification_sha256",
            "engineering_execution_receipt_sha256", "runtime_identity",
            "trace_contract_sha256", "trace_manifest_sha256",
            "completed_status_sha256", "failed_status_sha256",
            "pytorch_cuda_allocated_after_release",
            "sealed_case_payload_parsed_for_analysis", "sealed_model_access",
            "prior_sealed_files_read",
        }
        _require_exact_keys(receipt, receipt_keys, f"{stage}/{model} trace receipt")
        attempt_status = receipt["trace_attempt_status"]
        receipt_checks = (
            receipt.get("schema_version") == TRACE_RECEIPT_SCHEMA_VERSION,
            receipt.get("phase_id") == PHASE,
            receipt.get("stage") == stage,
            receipt.get("model") == model,
            receipt.get("protocol_sha256") == protocol_sha,
            receipt.get("stage_cases_sha256") == stage_case_sha,
            receipt.get("behavior_decision_sha256") == behavior_sha,
            receipt.get("candidate_specification_sha256")
            == expected_candidate_specification_sha256,
            receipt.get("discovery_registry_sha256")
            == expected_discovery_registry_sha256,
            receipt.get("confirmation_decision_sha256")
            == expected_confirmation_decision_sha256,
            receipt.get("engineering_qualification_sha256") == qualification_sha,
            receipt.get("engineering_execution_receipt_sha256")
            == qualification_receipt_sha,
            receipt.get("runtime_identity") == qualification.get("runtime_identity"),
            receipt.get("stage_pass") is stage_pass,
            attempt_status in {"complete", "behavior_blocked", "failed"},
            stage_pass is (attempt_status == "complete"),
            receipt.get("pytorch_cuda_allocated_after_release") == 0,
            receipt.get("sealed_case_payload_parsed_for_analysis") is False,
            receipt.get("sealed_model_access") is False,
            receipt.get("prior_sealed_files_read") is False,
        )
        if not all(receipt_checks):
            raise RuntimeError(f"{stage}/{model}: trace receipt contract mismatch")
        behavior_pass = behavior_report_by_model[model]["behavior_gate_pass"] is True
        if (
            (not behavior_pass and attempt_status != "behavior_blocked")
            or (behavior_pass and attempt_status == "behavior_blocked")
            or (stage_pass and not behavior_pass)
        ):
            raise RuntimeError(f"{stage}/{model}: behavior/trace authorization mismatch")

        manifest_path = trace_manifest_path(stage, model)
        contract_path = trace_contract_path(stage, model)
        completed_path = trace_completed_path(stage, model)
        failed_path = trace_model_dir(stage, model) / "phase576_generation_trace_failed.json"
        if stage_pass:
            if (
                not manifest_path.is_file()
                or not contract_path.is_file()
                or not completed_path.is_file()
                or failed_path.exists()
                or not _is_sha256(entry["trace_manifest_sha256"])
                or entry["trace_manifest_sha256"] != sha256_file(manifest_path)
                or receipt["trace_manifest_sha256"] != sha256_file(manifest_path)
                or receipt["trace_contract_sha256"] != sha256_file(contract_path)
                or receipt["completed_status_sha256"] != sha256_file(completed_path)
                or receipt["failed_status_sha256"] is not None
            ):
                raise RuntimeError(f"{stage}/{model}: trace manifest identity mismatch")
            manifest = read_json(manifest_path)
            manifest_keys = {
                "schema_version", "phase_id", "created_at_utc", "model", "stage",
                "case_count", "independent_unit_count", "shard_count", "shards",
                "prompt_role_labels", "feedback_slots",
                "all_executed_residual_positions",
                "batch_absorbing_eos_and_pad_feedback_positions_included",
                "complete_component_trajectory",
                "repeat_first_batch_max_abs_delta_bf16",
                "repeat_first_batch_exact_bf16",
                "all_generated_capsules_match_behavior_repeat1",
                "all_values_finite_before_and_after_bf16_conversion",
                "loaded_model_identity", "frozen_model_artifact_identity",
                "contract_sha256", "behavior_decision_sha256", "protocol_sha256",
                "trace_source_sha256", "candidate_specification_sha256",
                "discovery_registry_sha256", "confirmation_decision_sha256",
                "stage_cases_sha256", "engineering_qualification_sha256",
                "engineering_execution_receipt_sha256", "runtime_identity",
                "elapsed_seconds_before_release", "candidate_coordinates",
                "candidate_mechanism_formulas", "trace_complete",
                "internal_intervention", "causal",
                "sealed_case_payload_parsed_for_analysis", "sealed_model_access",
                "prior_sealed_files_read",
            }
            _require_exact_keys(manifest, manifest_keys, f"{stage}/{model} manifest")
            contract = read_json(contract_path)
            contract_keys = {
                "schema_version", "phase_id", "created_at_utc", "model", "stage",
                "model_order_index", "case_count", "case_ids_sha256",
                "stage_cases_sha256", "behavior_rows_sha256",
                "behavior_decision_sha256", "candidate_specification_sha256",
                "discovery_registry_sha256", "confirmation_decision_sha256",
                "protocol_sha256", "engineering_qualification_sha256",
                "engineering_execution_receipt_sha256", "runtime_identity",
                "trace_source_sha256", "frozen_stage_source_seals",
                "model_artifact_identity", "controlled_generation_interface",
                "deterministic_generation_reexecution",
                "behavior_repeat1_capsule_identity_required", "teacher_forced_replay",
                "cached_autoregressive_generation", "all_layers",
                "prompt_role_labels", "feedback_slots",
                "all_rendered_prompt_token_positions",
                "all_actually_executed_feedback_token_positions",
                "batch_absorbing_eos_and_pad_feedback_positions_included",
                "full_vectors_at_every_executed_residual_position",
                "complete_component_trajectory", "candidate_coordinates",
                "candidate_mechanism_formulas", "stored_dtype",
                "finite_values_required", "internal_intervention", "causal",
                "sealed_model_access",
            }
            _require_exact_keys(contract, contract_keys, f"{stage}/{model} trace contract")
            completed = read_json(completed_path)
            completed_keys = {
                "schema_version", "phase_id", "created_at_utc", "model", "stage",
                "status", "manifest_sha256", "contract_sha256",
                "trace_source_sha256", "engineering_qualification_sha256",
                "engineering_execution_receipt_sha256", "runtime_identity",
                "confirmation_decision_sha256",
                "pytorch_cuda_allocated_after_release",
                "pytorch_cuda_reserved_after_release", "sealed_model_access",
            }
            _require_exact_keys(completed, completed_keys, f"{stage}/{model} completed status")
            cases = read_jsonl(OPEN_SPLIT_CASE_PATHS[stage])
            expected_case_ids = [row["case_id"] for row in cases]
            behavior_report = next(
                item for item in behavior_decision["reports"]
                if item["model"] == model
            )
            engineering_report = next(
                item for item in qualification["reports"]
                if item["model"] == model
            )
            manifest_checks = (
                manifest.get("schema_version") == TRACE_MANIFEST_SCHEMA_VERSION,
                manifest.get("phase_id") == PHASE,
                manifest.get("stage") == stage,
                manifest.get("model") == model,
                manifest.get("protocol_sha256") == protocol_sha,
                manifest.get("stage_cases_sha256") == stage_case_sha,
                manifest.get("behavior_decision_sha256") == behavior_sha,
                manifest.get("candidate_specification_sha256")
                == expected_candidate_specification_sha256,
                manifest.get("discovery_registry_sha256")
                == expected_discovery_registry_sha256,
                manifest.get("confirmation_decision_sha256")
                == expected_confirmation_decision_sha256,
                manifest.get("engineering_qualification_sha256")
                == qualification_sha,
                manifest.get("engineering_execution_receipt_sha256")
                == qualification_receipt_sha,
                manifest.get("runtime_identity") == qualification.get("runtime_identity"),
                manifest.get("trace_source_sha256")
                == frozen["stage_source_seals"][
                    "tests/glm5/phase576_gpt5_fruit_natural_trace.py"
                ]["sha256"],
                manifest.get("contract_sha256") == sha256_file(contract_path),
                manifest.get("frozen_model_artifact_identity")
                == frozen["model_artifact_identities"][model],
                manifest.get("loaded_model_identity")
                == engineering_report.get("loaded_model_identity"),
                manifest.get("case_count") == 336,
                manifest.get("independent_unit_count") == 36,
                manifest.get("shard_count") == 336 // TRACE_BATCH_SIZE,
                manifest.get("feedback_slots") == [
                    f"generated_feedback_token_{index:02d}"
                    for index in range(MAX_NEW_TOKENS - 1)
                ],
                manifest.get("all_executed_residual_positions") is True,
                manifest.get(
                    "batch_absorbing_eos_and_pad_feedback_positions_included"
                ) is True,
                manifest.get("complete_component_trajectory") is False,
                manifest.get("repeat_first_batch_max_abs_delta_bf16") == 0.0,
                manifest.get("repeat_first_batch_exact_bf16") is True,
                manifest.get("all_generated_capsules_match_behavior_repeat1") is True,
                manifest.get("all_values_finite_before_and_after_bf16_conversion") is True,
                manifest.get("candidate_coordinates") == [],
                manifest.get("candidate_mechanism_formulas") == [],
                manifest.get("trace_complete") is True,
                manifest.get("internal_intervention") is False,
                manifest.get("causal") is False,
                manifest.get("sealed_case_payload_parsed_for_analysis") is False,
                manifest.get("sealed_model_access") is False,
                manifest.get("prior_sealed_files_read") is False,
                manifest.get("prompt_role_labels")
                == frozen.get("trace_policy", {}).get("role_labels"),
                contract.get("schema_version")
                == "phase576_generation_trace_contract.v2",
                contract.get("phase_id") == PHASE,
                contract.get("model") == model,
                contract.get("stage") == stage,
                contract.get("model_order_index") == MODELS.index(model),
                contract.get("case_count") == 336,
                contract.get("case_ids_sha256") == stable_hash(expected_case_ids),
                contract.get("stage_cases_sha256") == stage_case_sha,
                contract.get("behavior_rows_sha256")
                == behavior_report.get("behavior_rows_sha256"),
                contract.get("behavior_decision_sha256") == behavior_sha,
                contract.get("candidate_specification_sha256")
                == expected_candidate_specification_sha256,
                contract.get("discovery_registry_sha256")
                == expected_discovery_registry_sha256,
                contract.get("confirmation_decision_sha256")
                == expected_confirmation_decision_sha256,
                contract.get("protocol_sha256") == protocol_sha,
                contract.get("engineering_qualification_sha256")
                == qualification_sha,
                contract.get("engineering_execution_receipt_sha256")
                == qualification_receipt_sha,
                contract.get("runtime_identity") == qualification.get("runtime_identity"),
                contract.get("trace_source_sha256")
                == frozen["stage_source_seals"][
                    "tests/glm5/phase576_gpt5_fruit_natural_trace.py"
                ]["sha256"],
                contract.get("frozen_stage_source_seals")
                == frozen["stage_source_seals"],
                contract.get("model_artifact_identity")
                == frozen["model_artifact_identities"][model],
                contract.get("prompt_role_labels")
                == frozen["trace_policy"]["role_labels"],
                contract.get("feedback_slots") == manifest.get("feedback_slots"),
                contract.get("stored_dtype") == "bfloat16",
                contract.get("candidate_coordinates") == [],
                contract.get("candidate_mechanism_formulas") == [],
                contract.get("controlled_generation_interface") is True,
                contract.get("deterministic_generation_reexecution") is True,
                contract.get("behavior_repeat1_capsule_identity_required") is True,
                contract.get("teacher_forced_replay") is False,
                contract.get("cached_autoregressive_generation") is True,
                contract.get("all_layers") is True,
                contract.get("all_rendered_prompt_token_positions") is True,
                contract.get("all_actually_executed_feedback_token_positions") is True,
                contract.get(
                    "batch_absorbing_eos_and_pad_feedback_positions_included"
                ) is True,
                contract.get("full_vectors_at_every_executed_residual_position") is True,
                contract.get("complete_component_trajectory") is False,
                contract.get("finite_values_required") is True,
                contract.get("internal_intervention") is False,
                contract.get("causal") is False,
                contract.get("sealed_model_access") is False,
                completed.get("schema_version")
                == "phase576_generation_trace_completed.v1",
                completed.get("phase_id") == PHASE,
                completed.get("model") == model,
                completed.get("stage") == stage,
                completed.get("status") == "complete",
                completed.get("manifest_sha256") == sha256_file(manifest_path),
                completed.get("contract_sha256") == sha256_file(contract_path),
                completed.get("trace_source_sha256")
                == frozen["stage_source_seals"][
                    "tests/glm5/phase576_gpt5_fruit_natural_trace.py"
                ]["sha256"],
                completed.get("engineering_qualification_sha256")
                == qualification_sha,
                completed.get("engineering_execution_receipt_sha256")
                == qualification_receipt_sha,
                completed.get("runtime_identity") == qualification.get("runtime_identity"),
                completed.get("confirmation_decision_sha256")
                == expected_confirmation_decision_sha256,
                completed.get("pytorch_cuda_allocated_after_release") == 0,
                isinstance(completed.get("pytorch_cuda_reserved_after_release"), int),
                completed.get("sealed_model_access") is False,
            )
            if not all(manifest_checks):
                raise RuntimeError(f"{stage}/{model}: trace manifest contract mismatch")
            _verify_trace_shard_closure(
                stage, model, manifest, frozen, behavior_decision, qualification,
            )
        else:
            if entry["trace_manifest_sha256"] is not None or manifest_path.exists():
                raise RuntimeError(f"{stage}/{model}: blocked trace has a manifest")
            if receipt["trace_manifest_sha256"] is not None:
                raise RuntimeError(f"{stage}/{model}: blocked receipt binds a manifest")
            if attempt_status == "behavior_blocked":
                if any((
                    contract_path.exists(), completed_path.exists(), failed_path.exists(),
                    receipt["trace_contract_sha256"] is not None,
                    receipt["completed_status_sha256"] is not None,
                    receipt["failed_status_sha256"] is not None,
                )):
                    raise RuntimeError(f"{stage}/{model}: blocked trace has run artifacts")
            else:
                if (
                    attempt_status != "failed"
                    or not failed_path.is_file()
                    or completed_path.exists()
                    or receipt["failed_status_sha256"] != sha256_file(failed_path)
                    or receipt["completed_status_sha256"] is not None
                ):
                    raise RuntimeError(f"{stage}/{model}: failed trace closure mismatch")
                failed = read_json(failed_path)
                failed_keys = {
                    "schema_version", "phase_id", "created_at_utc", "model", "stage",
                    "status", "error_type", "error", "trace_contract_sha256",
                    "started_status_sha256", "partial_shards", "trace_source_sha256",
                    "engineering_qualification_sha256",
                    "engineering_execution_receipt_sha256", "runtime_identity",
                    "confirmation_decision_sha256",
                    "pytorch_cuda_allocated_after_release",
                    "pytorch_cuda_reserved_after_release", "sealed_model_access",
                }
                _require_exact_keys(failed, failed_keys, f"{stage}/{model} failed status")
                expected_contract_sha = (
                    sha256_file(contract_path) if contract_path.is_file() else None
                )
                started_path = trace_model_dir(stage, model) / (
                    "phase576_generation_trace_started.json"
                )
                expected_started_sha = (
                    sha256_file(started_path) if started_path.is_file() else None
                )
                partial = failed["partial_shards"]
                actual_partial_paths = sorted(
                    trace_model_dir(stage, model).glob(
                        "phase576_generation_trace_shard_*.pt"
                    )
                )
                if not isinstance(partial, list) or len(partial) != len(actual_partial_paths):
                    raise RuntimeError(f"{stage}/{model}: partial shard registry drift")
                for item, path in zip(partial, actual_partial_paths):
                    _require_exact_keys(
                        item, {"path", "size_bytes", "sha256"},
                        f"{stage}/{model} partial shard",
                    )
                    if (
                        item["path"]
                        != str(path.relative_to(OUT_DIR)).replace("\\", "/")
                        or path.is_symlink()
                        or item["size_bytes"] != path.stat().st_size
                        or item["sha256"] != sha256_file(path)
                    ):
                        raise RuntimeError(f"{stage}/{model}: partial shard identity drift")
                failed_checks = (
                    failed["schema_version"] == "phase576_generation_trace_failed.v1",
                    failed["phase_id"] == PHASE,
                    failed["model"] == model,
                    failed["stage"] == stage,
                    failed["status"] == "failed",
                    isinstance(failed["error_type"], str) and bool(failed["error_type"]),
                    isinstance(failed["error"], str),
                    failed["trace_contract_sha256"] == expected_contract_sha,
                    receipt["trace_contract_sha256"] == expected_contract_sha,
                    failed["started_status_sha256"] == expected_started_sha,
                    failed["trace_source_sha256"]
                    == frozen["stage_source_seals"][
                        "tests/glm5/phase576_gpt5_fruit_natural_trace.py"
                    ]["sha256"],
                    failed["engineering_qualification_sha256"] == qualification_sha,
                    failed["engineering_execution_receipt_sha256"]
                    == qualification_receipt_sha,
                    failed["runtime_identity"] == qualification.get("runtime_identity"),
                    failed["confirmation_decision_sha256"]
                    == expected_confirmation_decision_sha256,
                    failed["pytorch_cuda_allocated_after_release"] == 0,
                    isinstance(failed["pytorch_cuda_reserved_after_release"], int),
                    failed["sealed_model_access"] is False,
                )
                if not all(failed_checks):
                    raise RuntimeError(f"{stage}/{model}: failed status contract mismatch")
    _verify_trace_stage_receipt(
        stage,
        frozen,
        behavior_decision,
        qualification,
        qualification_sha,
        qualification_receipt_sha,
        expected_candidate_specification_sha256=(
            expected_candidate_specification_sha256
        ),
        expected_discovery_registry_sha256=expected_discovery_registry_sha256,
        expected_confirmation_decision_sha256=(
            expected_confirmation_decision_sha256
        ),
        pass_map=pass_map,
    )
    return pass_map


def verify_discovery_registry(
    frozen: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _verify_freeze_commit()
    if frozen is None:
        frozen = read_json(PROTOCOL_PATH)
        verify_frozen_source_seals(frozen)
        verify_frozen_model_artifacts(frozen)
    elif frozen != read_json(PROTOCOL_PATH):
        raise RuntimeError("caller-supplied frozen protocol differs from disk")
    else:
        verify_frozen_source_seals(frozen)
    if not DISCOVERY_REGISTRY_PATH.is_file():
        raise RuntimeError("Phase576 discovery registry is missing")
    registry = read_json(DISCOVERY_REGISTRY_PATH)
    expected_keys = {
        "schema_version",
        "phase_id",
        "stage",
        "created_at_utc",
        "discovery_registry_frozen",
        "protocol_sha256",
        "discovery_cases_sha256",
        "discovery_behavior_decision_sha256",
        "models_in_required_order",
        "trace_artifacts_by_model",
        "analysis_source_identity",
        "analysis_dependency_inventory",
        "analysis_dependency_inventory_sha256",
        "analysis_api_version",
        "analysis_execution_sandboxed",
        "analysis_io_isolation_enforced",
        "analysis_trust_boundary",
        "candidate_claim_type",
        "discovery_candidate_pass",
        "discovery_evaluation",
        "discovery_evaluation_sha256",
        "candidate_specification",
        "candidate_specification_sha256",
        "confirmation_rule",
        "confirmation_rule_sha256",
        "cross_model_observational_comparison_authorized",
        "cross_model_common_structure_claim_authorized",
        "sealed_case_payload_parsed_for_analysis_declared",
        "sealed_model_access_count_declared",
        "sealed_result_analysis_access_count_declared",
        "prior_sealed_files_read_declared",
        "causal_claim_authorized",
    }
    _require_exact_keys(registry, expected_keys, "discovery registry")
    behavior_path = BEHAVIOR_DECISION_PATHS["discovery"]
    if not behavior_path.is_file():
        raise RuntimeError("discovery behavior decision is missing")
    candidate_specification = registry["candidate_specification"]
    confirmation_rule = registry["confirmation_rule"]
    if not isinstance(candidate_specification, dict) or not candidate_specification:
        raise RuntimeError("discovery candidate specification is empty or invalid")
    if not isinstance(confirmation_rule, dict) or not confirmation_rule:
        raise RuntimeError("confirmation rule was not frozen at discovery")
    _verify_analysis_source_identity(registry["analysis_source_identity"], frozen)
    pass_map = _verify_trace_artifact_registry(
        "discovery",
        registry["trace_artifacts_by_model"],
        frozen,
        expected_candidate_specification_sha256=None,
        expected_discovery_registry_sha256=None,
    )
    derived_cross_model_comparison = (
        cross_model_observational_comparison_authorized("discovery", pass_map)
    )
    analysis_module = _load_post_trace_analysis_source(
        registry["analysis_source_identity"]
    )
    recompute_discovery = getattr(
        analysis_module, "recompute_discovery_artifacts", None
    )
    if not callable(recompute_discovery):
        raise RuntimeError("frozen analysis source lacks discovery recomputation API")
    recomputed = recompute_discovery(ROOT, frozen)
    _require_exact_keys(
        recomputed,
        {
            "analysis_api_version", "discovery_evaluation",
            "discovery_candidate_pass", "candidate_specification",
            "confirmation_rule",
        },
        "recomputed discovery artifacts",
    )
    discovery_evaluation = registry["discovery_evaluation"]
    derived_discovery_candidate_pass = recomputed["discovery_candidate_pass"]
    if not isinstance(derived_discovery_candidate_pass, bool):
        raise RuntimeError("recomputed discovery candidate pass is not boolean")
    qualification = read_json(ENGINEERING_QUALIFICATION_PATH)
    expected_dependency_inventory = {
        "local_file_dependencies": [],
        "allowed_import_roots": list(POST_TRACE_ALLOWED_IMPORT_ROOTS),
        "runtime_identity": qualification["runtime_identity"],
    }
    checks = (
        registry["schema_version"] == DISCOVERY_REGISTRY_SCHEMA_VERSION,
        registry["phase_id"] == PHASE,
        registry["stage"] == "discovery",
        registry["discovery_registry_frozen"] is True,
        registry["protocol_sha256"] == sha256_file(PROTOCOL_PATH),
        registry["discovery_cases_sha256"]
        == frozen["open_case_sha256_by_split"]["discovery"],
        registry["discovery_behavior_decision_sha256"]
        == sha256_file(behavior_path),
        registry["models_in_required_order"] == list(MODELS),
        registry["analysis_dependency_inventory"]
        == expected_dependency_inventory,
        _is_sha256(registry["analysis_dependency_inventory_sha256"]),
        registry["analysis_dependency_inventory_sha256"]
        == stable_hash(expected_dependency_inventory),
        registry["analysis_api_version"] == "phase576_post_trace_analysis.v1",
        recomputed["analysis_api_version"] == registry["analysis_api_version"],
        registry["analysis_execution_sandboxed"] is False,
        registry["analysis_io_isolation_enforced"] is False,
        registry["analysis_trust_boundary"]
        == "hash-sealed standalone source plus static import audit and human code review; no OS-level I/O sandbox",
        registry["candidate_claim_type"] == "descriptive_observational_only",
        registry["discovery_candidate_pass"]
        is derived_discovery_candidate_pass,
        not registry["discovery_candidate_pass"]
        or derived_cross_model_comparison,
        isinstance(discovery_evaluation, dict) and bool(discovery_evaluation),
        _is_sha256(registry["discovery_evaluation_sha256"]),
        registry["discovery_evaluation_sha256"]
        == stable_hash(discovery_evaluation),
        recomputed["discovery_evaluation"] == discovery_evaluation,
        recomputed["candidate_specification"] == candidate_specification,
        recomputed["confirmation_rule"] == confirmation_rule,
        _is_sha256(registry["candidate_specification_sha256"]),
        registry["candidate_specification_sha256"]
        == stable_hash(candidate_specification),
        _is_sha256(registry["confirmation_rule_sha256"]),
        registry["confirmation_rule_sha256"] == stable_hash(confirmation_rule),
        registry["cross_model_observational_comparison_authorized"]
        is derived_cross_model_comparison,
        registry["cross_model_common_structure_claim_authorized"] is False,
        registry["sealed_case_payload_parsed_for_analysis_declared"] is False,
        registry["sealed_model_access_count_declared"] == 0,
        registry["sealed_result_analysis_access_count_declared"] == 0,
        registry["prior_sealed_files_read_declared"] is False,
        registry["causal_claim_authorized"] is False,
    )
    if not all(checks):
        raise RuntimeError("Phase576 discovery registry contract mismatch")
    return registry


def verify_confirmation_decision(
    frozen: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if frozen is None:
        frozen = read_json(PROTOCOL_PATH)
        verify_frozen_source_seals(frozen)
        verify_frozen_model_artifacts(frozen)
    elif frozen != read_json(PROTOCOL_PATH):
        raise RuntimeError("caller-supplied frozen protocol differs from disk")
    registry = verify_discovery_registry(frozen)
    if registry["discovery_candidate_pass"] is not True:
        raise RuntimeError("confirmation decision requires passed discovery evidence")
    if not CONFIRMATION_DECISION_PATH.is_file():
        raise RuntimeError("Phase576 confirmation decision is missing")
    decision = read_json(CONFIRMATION_DECISION_PATH)
    expected_keys = {
        "schema_version",
        "phase_id",
        "stage",
        "created_at_utc",
        "structure_confirmation_pass",
        "protocol_sha256",
        "confirmation_cases_sha256",
        "discovery_registry_sha256",
        "confirmation_behavior_decision_sha256",
        "models_in_required_order",
        "trace_artifacts_by_model",
        "analysis_source_identity",
        "analysis_dependency_inventory_sha256",
        "analysis_api_version",
        "analysis_execution_sandboxed",
        "analysis_io_isolation_enforced",
        "analysis_trust_boundary",
        "candidate_specification_sha256",
        "confirmation_rule_sha256",
        "confirmation_evaluation",
        "confirmation_evaluation_sha256",
        "all_models_trace_complete",
        "cross_model_observational_comparison_authorized",
        "cross_model_common_structure_claim_authorized",
        "sealed_case_payload_parsed_for_analysis_declared",
        "sealed_model_access_count_declared",
        "sealed_result_analysis_access_count_declared",
        "prior_sealed_files_read_declared",
        "causal_claim_authorized",
    }
    _require_exact_keys(decision, expected_keys, "confirmation decision")
    behavior_path = BEHAVIOR_DECISION_PATHS["confirmation"]
    if not behavior_path.is_file():
        raise RuntimeError("confirmation behavior decision is missing")
    registry_sha = sha256_file(DISCOVERY_REGISTRY_PATH)
    pass_map = _verify_trace_artifact_registry(
        "confirmation",
        decision["trace_artifacts_by_model"],
        frozen,
        expected_candidate_specification_sha256=registry[
            "candidate_specification_sha256"
        ],
        expected_discovery_registry_sha256=registry_sha,
    )
    all_models_trace_complete = cross_model_observational_comparison_authorized(
        "confirmation", pass_map,
    )
    analysis_module = _load_post_trace_analysis_source(
        registry["analysis_source_identity"]
    )
    recompute_confirmation = getattr(
        analysis_module, "recompute_confirmation_artifacts", None
    )
    if not callable(recompute_confirmation):
        raise RuntimeError("frozen analysis source lacks confirmation recomputation API")
    recomputed = recompute_confirmation(ROOT, frozen, registry)
    _require_exact_keys(
        recomputed,
        {
            "analysis_api_version", "confirmation_evaluation",
            "structure_confirmation_pass",
        },
        "recomputed confirmation artifacts",
    )
    derived_confirmation_pass = recomputed["structure_confirmation_pass"]
    if not isinstance(derived_confirmation_pass, bool):
        raise RuntimeError("recomputed confirmation pass is not boolean")
    structure_claim_authorized = (
        all_models_trace_complete and derived_confirmation_pass
    )
    checks = (
        decision["schema_version"] == CONFIRMATION_DECISION_SCHEMA_VERSION,
        decision["phase_id"] == PHASE,
        decision["stage"] == "confirmation",
        decision["structure_confirmation_pass"] is derived_confirmation_pass,
        decision["protocol_sha256"] == sha256_file(PROTOCOL_PATH),
        decision["confirmation_cases_sha256"]
        == frozen["open_case_sha256_by_split"]["confirmation"],
        decision["discovery_registry_sha256"] == registry_sha,
        decision["confirmation_behavior_decision_sha256"]
        == sha256_file(behavior_path),
        decision["models_in_required_order"] == list(MODELS),
        decision["analysis_source_identity"]
        == registry["analysis_source_identity"],
        decision["analysis_dependency_inventory_sha256"]
        == registry["analysis_dependency_inventory_sha256"],
        decision["analysis_api_version"] == registry["analysis_api_version"]
        == recomputed["analysis_api_version"],
        decision["analysis_execution_sandboxed"] is False,
        decision["analysis_io_isolation_enforced"] is False,
        decision["analysis_trust_boundary"] == registry["analysis_trust_boundary"],
        decision["candidate_specification_sha256"]
        == registry["candidate_specification_sha256"],
        decision["confirmation_rule_sha256"]
        == registry["confirmation_rule_sha256"],
        isinstance(decision["confirmation_evaluation"], dict)
        and bool(decision["confirmation_evaluation"]),
        _is_sha256(decision["confirmation_evaluation_sha256"]),
        decision["confirmation_evaluation_sha256"]
        == stable_hash(decision["confirmation_evaluation"]),
        decision["confirmation_evaluation"]
        == recomputed["confirmation_evaluation"],
        decision["all_models_trace_complete"] is all_models_trace_complete,
        decision["cross_model_observational_comparison_authorized"]
        is all_models_trace_complete,
        decision["cross_model_common_structure_claim_authorized"]
        is structure_claim_authorized,
        not decision["structure_confirmation_pass"] or all_models_trace_complete,
        decision["sealed_case_payload_parsed_for_analysis_declared"] is False,
        decision["sealed_model_access_count_declared"] == 0,
        decision["sealed_result_analysis_access_count_declared"] == 0,
        decision["prior_sealed_files_read_declared"] is False,
        decision["causal_claim_authorized"] is False,
    )
    if not all(checks):
        raise RuntimeError("Phase576 confirmation decision contract mismatch")
    return decision


def verify_heldout_decision(
    frozen: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if frozen is None:
        frozen = read_json(PROTOCOL_PATH)
        verify_frozen_source_seals(frozen)
        verify_frozen_model_artifacts(frozen)
    elif frozen != read_json(PROTOCOL_PATH):
        raise RuntimeError("caller-supplied frozen protocol differs from disk")
    confirmation = verify_confirmation_decision(frozen)
    if confirmation["structure_confirmation_pass"] is not True:
        raise RuntimeError("heldout decision requires a passed confirmation")
    if not HELDOUT_DECISION_PATH.is_file():
        raise RuntimeError("Phase576 heldout replication decision is missing")
    registry = read_json(DISCOVERY_REGISTRY_PATH)
    decision = read_json(HELDOUT_DECISION_PATH)
    expected_keys = {
        "schema_version", "phase_id", "stage", "created_at_utc",
        "heldout_replication_pass", "protocol_sha256", "heldout_cases_sha256",
        "discovery_registry_sha256", "confirmation_decision_sha256",
        "heldout_behavior_decision_sha256", "models_in_required_order",
        "trace_artifacts_by_model", "analysis_source_identity",
        "analysis_dependency_inventory_sha256",
        "analysis_api_version", "candidate_specification_sha256",
        "analysis_execution_sandboxed", "analysis_io_isolation_enforced",
        "analysis_trust_boundary",
        "confirmation_rule_sha256", "heldout_evaluation",
        "heldout_evaluation_sha256", "all_models_trace_complete",
        "cross_model_observational_comparison_authorized",
        "cross_model_common_structure_claim_authorized",
        "sealed_case_payload_parsed_for_analysis_declared",
        "sealed_model_access_count_declared",
        "sealed_result_analysis_access_count_declared",
        "prior_sealed_files_read_declared",
        "causal_claim_authorized",
    }
    _require_exact_keys(decision, expected_keys, "heldout decision")
    registry_sha = sha256_file(DISCOVERY_REGISTRY_PATH)
    confirmation_sha = sha256_file(CONFIRMATION_DECISION_PATH)
    behavior_path = BEHAVIOR_DECISION_PATHS["heldout_recombination"]
    if not behavior_path.is_file():
        raise RuntimeError("heldout behavior decision is missing")
    pass_map = _verify_trace_artifact_registry(
        "heldout_recombination",
        decision["trace_artifacts_by_model"],
        frozen,
        expected_candidate_specification_sha256=registry[
            "candidate_specification_sha256"
        ],
        expected_discovery_registry_sha256=registry_sha,
        expected_confirmation_decision_sha256=confirmation_sha,
    )
    all_models_trace_complete = cross_model_observational_comparison_authorized(
        "heldout_recombination", pass_map,
    )
    analysis_module = _load_post_trace_analysis_source(
        registry["analysis_source_identity"]
    )
    recompute_heldout = getattr(
        analysis_module, "recompute_heldout_artifacts", None
    )
    if not callable(recompute_heldout):
        raise RuntimeError("frozen analysis source lacks heldout recomputation API")
    recomputed = recompute_heldout(ROOT, frozen, registry, confirmation)
    _require_exact_keys(
        recomputed,
        {
            "analysis_api_version", "heldout_evaluation",
            "heldout_replication_pass",
        },
        "recomputed heldout artifacts",
    )
    derived_pass = recomputed["heldout_replication_pass"]
    if not isinstance(derived_pass, bool):
        raise RuntimeError("recomputed heldout pass is not boolean")
    common_structure_authorized = (
        confirmation["cross_model_common_structure_claim_authorized"] is True
        and all_models_trace_complete
        and derived_pass
    )
    checks = (
        decision["schema_version"] == HELDOUT_DECISION_SCHEMA_VERSION,
        decision["phase_id"] == PHASE,
        decision["stage"] == "heldout_recombination",
        decision["heldout_replication_pass"] is derived_pass,
        decision["protocol_sha256"] == sha256_file(PROTOCOL_PATH),
        decision["heldout_cases_sha256"]
        == frozen["open_case_sha256_by_split"]["heldout_recombination"],
        decision["discovery_registry_sha256"] == registry_sha,
        decision["confirmation_decision_sha256"] == confirmation_sha,
        decision["heldout_behavior_decision_sha256"] == sha256_file(behavior_path),
        decision["models_in_required_order"] == list(MODELS),
        decision["analysis_source_identity"] == registry["analysis_source_identity"],
        decision["analysis_dependency_inventory_sha256"]
        == registry["analysis_dependency_inventory_sha256"],
        decision["analysis_api_version"] == registry["analysis_api_version"]
        == recomputed["analysis_api_version"],
        decision["analysis_execution_sandboxed"] is False,
        decision["analysis_io_isolation_enforced"] is False,
        decision["analysis_trust_boundary"] == registry["analysis_trust_boundary"],
        decision["candidate_specification_sha256"]
        == registry["candidate_specification_sha256"],
        decision["confirmation_rule_sha256"]
        == registry["confirmation_rule_sha256"],
        isinstance(decision["heldout_evaluation"], dict)
        and bool(decision["heldout_evaluation"]),
        _is_sha256(decision["heldout_evaluation_sha256"]),
        decision["heldout_evaluation_sha256"]
        == stable_hash(decision["heldout_evaluation"]),
        decision["heldout_evaluation"] == recomputed["heldout_evaluation"],
        decision["all_models_trace_complete"] is all_models_trace_complete,
        decision["cross_model_observational_comparison_authorized"]
        is all_models_trace_complete,
        decision["cross_model_common_structure_claim_authorized"]
        is common_structure_authorized,
        not decision["heldout_replication_pass"] or all_models_trace_complete,
        decision["sealed_case_payload_parsed_for_analysis_declared"] is False,
        decision["sealed_model_access_count_declared"] == 0,
        decision["sealed_result_analysis_access_count_declared"] == 0,
        decision["prior_sealed_files_read_declared"] is False,
        decision["causal_claim_authorized"] is False,
    )
    if not all(checks):
        raise RuntimeError("Phase576 heldout decision contract mismatch")
    return decision


def aliases(item: dict[str, Any], relation: str) -> list[str]:
    return list(item[f"{relation}_aliases"])


def all_alias_groups(split: str, relation: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for item in OBJECTS[split]:
        canonical = item[relation]
        result.setdefault(canonical, [])
        result[canonical].extend(aliases(item, relation))
    return {
        key: list(dict.fromkeys(values))
        for key, values in sorted(result.items())
    }


def _compose_prompt(
    parts: tuple[str | tuple[str, str], ...],
    reply_instruction: str,
) -> tuple[str, dict[str, dict[str, Any]]]:
    chunks: list[str] = []
    spans: dict[str, dict[str, Any]] = {}
    cursor = 0
    for part in parts:
        if isinstance(part, tuple):
            role, text = part
            if role in spans or not text:
                raise RuntimeError(f"invalid or duplicate prompt role: {role}")
            spans[role] = {
                "start": cursor,
                "end": cursor + len(text),
                "text": text,
            }
            chunks.append(text)
            cursor += len(text)
        else:
            chunks.append(part)
            cursor += len(part)
    chunks.append(reply_instruction)
    return "".join(chunks), spans


def _direct_prompt_record(label: str, relation: str, surface: int) -> dict[str, Any]:
    if relation == "category":
        query = "fruit"
        templates: tuple[tuple[str | tuple[str, str], ...], ...] = (
            ("Using common everyday knowledge, is ", ("focus", label), " a ", ("query_anchor", query), "?"),
            ("Fruit membership check: does ", ("focus", label), " count as a ", ("query_anchor", query), "?"),
            ("Answer yes or no: is ", ("focus", label), " ordinarily called a ", ("query_anchor", query), "?"),
            ("Object: ", ("focus", label), ". Requested fact: whether it is a ", ("query_anchor", query), "."),
            ("For ", ("focus", label), ", is the everyday ", ("query_anchor", query), " label applicable?"),
            ("Fruit first: should ", ("focus", label), " be classified as a ", ("query_anchor", query), "?"),
        )
        contract_id = CATEGORY_RELATION_CONTRACT_ID
    elif relation == "color":
        query = "color"
        templates = (
            ("Using common everyday knowledge, what ", ("query_anchor", query), " is most commonly associated with ", ("focus", label), "?"),
            ("Which single ", ("query_anchor", query), " has the most familiar everyday association with ", ("focus", label), "?"),
            ("Name the ", ("query_anchor", query), " most commonly associated with ", ("focus", label), " in everyday descriptions."),
            ("Object: ", ("focus", label), ". Requested field: most commonly associated ", ("query_anchor", query), "."),
            ("For ", ("focus", label), ", give the single most common everyday ", ("query_anchor", query), " association."),
            ("Color association first: which ", ("query_anchor", query), " is most commonly associated with ", ("focus", label), "?"),
        )
        contract_id = COLOR_RELATION_CONTRACT_ID
    else:
        raise RuntimeError(f"unsupported direct relation: {relation}")
    if surface not in DIRECT_SURFACES or len(templates) != len(DIRECT_SURFACES):
        raise RuntimeError(f"invalid direct surface/template registry: {relation}/{surface}")
    raw_prompt, spans = _compose_prompt(
        templates[surface],
        " Reply with only the short answer and no explanation.",
    )
    return {
        "raw_prompt": raw_prompt,
        "prompt_template_id": f"direct.{relation}.s{surface}",
        "relation_contract_id": contract_id,
        "query_value": query,
        "raw_role_char_spans": {
            "focus": spans["focus"],
            "comparison": None,
            "query_anchor": spans["query_anchor"],
        },
    }


def direct_prompt(label: str, relation: str, surface: int) -> str:
    return _direct_prompt_record(label, relation, surface)["raw_prompt"]


def _selection_prompt_record(
    relation: str,
    query_value: str,
    left: str,
    right: str,
    surface: int,
) -> dict[str, Any]:
    query = ("query_anchor", query_value)
    left_part = ("candidate_left", left)
    right_part = ("candidate_right", right)
    if relation == "category":
        if query_value not in {"fruit", "not fruit"}:
            raise RuntimeError(f"unsupported category selection value: {query_value}")
        templates: tuple[tuple[str | tuple[str, str], ...], ...] = (
            ("Query fruit-membership property: ", query, ". Which item matches: ", left_part, " or ", right_part, "?"),
            ("Choose the item for the everyday fruit-membership query ", query, ": ", left_part, "; ", right_part, "."),
            ("Everyday fruit-membership check. Target=", query, ". Compare ", left_part, " versus ", right_part, "."),
            ("Requested fruit property=", query, ". Candidates=", left_part, ", ", right_part, ". Select the matching item."),
        )
        contract_id = CATEGORY_RELATION_CONTRACT_ID
    elif relation == "color":
        templates = (
            ("Query color association: ", query, ". Which item is most commonly associated with that color: ", left_part, " or ", right_part, "?"),
            ("Choose the item whose single most familiar color association is ", query, ": ", left_part, "; ", right_part, "."),
            ("Everyday most-common-color check. Target color=", query, ". Compare ", left_part, " versus ", right_part, "."),
            ("Requested most-commonly-associated color=", query, ". Candidates=", left_part, ", ", right_part, ". Select the matching item."),
        )
        contract_id = COLOR_RELATION_CONTRACT_ID
    else:
        raise RuntimeError(f"unsupported selection relation: {relation}")
    if surface not in SELECTION_SURFACES or len(templates) != len(SELECTION_SURFACES):
        raise RuntimeError(f"invalid selection surface/template registry: {relation}/{surface}")
    raw_prompt, spans = _compose_prompt(
        templates[surface],
        " Reply with only the item name and no explanation.",
    )
    return {
        "raw_prompt": raw_prompt,
        "prompt_template_id": f"selection.{relation}.s{surface}",
        "relation_contract_id": contract_id,
        "query_value": query_value,
        "candidate_spans": {
            "left": spans["candidate_left"],
            "right": spans["candidate_right"],
        },
        "query_anchor_span": spans["query_anchor"],
    }


def selection_prompt(
    relation: str,
    query_value: str,
    left: str,
    right: str,
    surface: int,
) -> str:
    return _selection_prompt_record(
        relation, query_value, left, right, surface
    )["raw_prompt"]


def base_case(
    split: str,
    case_id: str,
    relation: str,
    interface: str,
    raw_prompt: str,
    prompt_template_id: str,
    relation_contract_id: str,
    query_value: str,
    raw_role_char_spans: dict[str, Any],
    target: str,
    target_aliases: list[str],
    candidate_groups: dict[str, list[str]],
    focus: dict[str, Any],
    comparison: dict[str, Any] | None,
    surface: int,
    order: int | None,
    contrast_group_id: str,
    contrast_label: str,
    independent_unit_id: str,
    query_side_object_id: str,
    candidate_left: dict[str, Any] | None,
    candidate_right: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "phase_id": PHASE,
        "split": split,
        "sealed": split == "sealed",
        "case_id": case_id,
        "relation": relation,
        "interface": interface,
        "surface_id": surface,
        "order": order,
        "raw_prompt": raw_prompt,
        "prompt_template_id": prompt_template_id,
        "relation_contract_id": relation_contract_id,
        "normalized_prompt_sha256": stable_hash(normalize_prompt(raw_prompt)),
        "query_value": query_value,
        "raw_role_char_spans": raw_role_char_spans,
        "target": target,
        "target_aliases": target_aliases,
        "candidate_groups": candidate_groups,
        "focus_object_id": focus["id"],
        "focus_object_label": focus["label"],
        "focus_is_fruit": focus["is_fruit"],
        "focus_category": focus["category"],
        "focus_color": focus["color"],
        "comparison_object_id": comparison["id"] if comparison else None,
        "comparison_object_label": comparison["label"] if comparison else None,
        "candidate_left_id": candidate_left["id"] if candidate_left else None,
        "candidate_left_label": candidate_left["label"] if candidate_left else None,
        "candidate_right_id": candidate_right["id"] if candidate_right else None,
        "candidate_right_label": candidate_right["label"] if candidate_right else None,
        "query_side_object_id": query_side_object_id,
        "contrast_group_id": contrast_group_id,
        "contrast_label": contrast_label,
        "independent_unit_id": independent_unit_id,
        "query_anchor_fragment": raw_role_char_spans["query_anchor"]["text"],
        "mechanism_label": None,
        "candidate_layer": None,
        "candidate_head": None,
        "candidate_neuron": None,
        "candidate_direction": None,
    }


def _selection_role_spans(
    prompt_record: dict[str, Any],
    focus: dict[str, Any],
    comparison: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    if {focus["id"], comparison["id"]} != {left["id"], right["id"]}:
        raise RuntimeError("selection role/candidate identity mismatch")
    candidate_spans = prompt_record["candidate_spans"]
    focus_key = "left" if focus["id"] == left["id"] else "right"
    comparison_key = "right" if focus_key == "left" else "left"
    return {
        "focus": candidate_spans[focus_key],
        "comparison": candidate_spans[comparison_key],
        "query_anchor": prompt_record["query_anchor_span"],
    }


def build_split(split: str) -> list[dict[str, Any]]:
    items = OBJECTS[split]
    fruits = [item for item in items if item["is_fruit"]]
    controls = [item for item in items if not item["is_fruit"]]
    if len(fruits) != 6 or len(controls) != 6:
        raise RuntimeError(f"{split}: expected six fruits and six controls")
    by_color_fruit = {item["color"]: item for item in fruits}
    by_color_control = {item["color"]: item for item in controls}
    if set(by_color_fruit) != set(by_color_control) or len(by_color_fruit) != 6:
        raise RuntimeError(f"{split}: primary colors are not one-to-one matched")

    rows: list[dict[str, Any]] = []
    for item in items:
        for relation in RELATIONS:
            if relation == "category":
                target = "yes" if item["is_fruit"] else "no"
                target_aliases = [target]
                groups = {"yes": ["yes"], "no": ["no"]}
            else:
                target = item[relation]
                target_aliases = aliases(item, relation)
                groups = all_alias_groups(split, relation)
            for surface in DIRECT_SURFACES:
                prompt_record = _direct_prompt_record(
                    item["label"], relation, surface
                )
                case_id = f"phase576_{split}_direct_{item['id']}_{relation}_s{surface}"
                rows.append(base_case(
                    split=split,
                    case_id=case_id,
                    relation=relation,
                    interface="direct",
                    raw_prompt=prompt_record["raw_prompt"],
                    prompt_template_id=prompt_record["prompt_template_id"],
                    relation_contract_id=prompt_record["relation_contract_id"],
                    query_value=prompt_record["query_value"],
                    raw_role_char_spans=prompt_record["raw_role_char_spans"],
                    target=target,
                    target_aliases=target_aliases,
                    candidate_groups=groups,
                    focus=item,
                    comparison=None,
                    surface=surface,
                    order=None,
                    contrast_group_id=f"{split}_direct_{item['id']}_s{surface}",
                    contrast_label=relation,
                    independent_unit_id=f"{split}_direct_{item['id']}_{relation}",
                    query_side_object_id=item["id"],
                    candidate_left=None,
                    candidate_right=None,
                ))

    for color in sorted(by_color_fruit):
        fruit = by_color_fruit[color]
        control = by_color_control[color]
        pair_id = f"{split}_category_pair_{color}"
        for queried in (fruit, control):
            query_value = "fruit" if queried is fruit else "not fruit"
            for surface in SELECTION_SURFACES:
                for order in SELECTION_ORDERS:
                    left, right = (fruit, control) if order == 0 else (control, fruit)
                    comparison = control if queried is fruit else fruit
                    prompt_record = _selection_prompt_record(
                        "category",
                        query_value,
                        left["label"],
                        right["label"],
                        surface,
                    )
                    case_id = (
                        f"phase576_{split}_category_select_{color}_"
                        f"target_{queried['id']}_s{surface}_o{order}"
                    )
                    rows.append(base_case(
                        split=split,
                        case_id=case_id,
                        relation="category",
                        interface="selection",
                        raw_prompt=prompt_record["raw_prompt"],
                        prompt_template_id=prompt_record["prompt_template_id"],
                        relation_contract_id=prompt_record["relation_contract_id"],
                        query_value=query_value,
                        raw_role_char_spans=_selection_role_spans(
                            prompt_record, queried, comparison, left, right
                        ),
                        target=queried["label"],
                        target_aliases=[queried["label"]],
                        candidate_groups={
                            fruit["label"]: [fruit["label"]],
                            control["label"]: [control["label"]],
                        },
                        focus=queried,
                        comparison=comparison,
                        surface=surface,
                        order=order,
                        contrast_group_id=f"{pair_id}_s{surface}_o{order}",
                        contrast_label=queried["id"],
                        independent_unit_id=pair_id,
                        query_side_object_id=queried["id"],
                        candidate_left=left,
                        candidate_right=right,
                    ))

    # A six-cycle yields six pair units; each fruit participates in its
    # predecessor and successor pair. Both query sides are fully crossed with
    # every surface/order, so target selection is paired within each unit.
    ordered_fruits = sorted(fruits, key=lambda item: item["id"])
    for index, first in enumerate(ordered_fruits):
        second = ordered_fruits[(index + 1) % len(ordered_fruits)]
        if first["color"] == second["color"]:
            raise RuntimeError(f"{split}: color selection pair is not contrasting")
        pair_key = "__".join(sorted((first["id"], second["id"])))
        for queried in (first, second):
            for surface in SELECTION_SURFACES:
                for order in SELECTION_ORDERS:
                    left, right = (first, second) if order == 0 else (second, first)
                    comparison = second if queried is first else first
                    prompt_record = _selection_prompt_record(
                        "color",
                        queried["color"],
                        left["label"],
                        right["label"],
                        surface,
                    )
                    case_id = (
                        f"phase576_{split}_color_select_{pair_key}_"
                        f"target_{queried['id']}_s{surface}_o{order}"
                    )
                    rows.append(base_case(
                        split=split,
                        case_id=case_id,
                        relation="color",
                        interface="selection",
                        raw_prompt=prompt_record["raw_prompt"],
                        prompt_template_id=prompt_record["prompt_template_id"],
                        relation_contract_id=prompt_record["relation_contract_id"],
                        query_value=queried["color"],
                        raw_role_char_spans=_selection_role_spans(
                            prompt_record, queried, comparison, left, right
                        ),
                        target=queried["label"],
                        target_aliases=[queried["label"]],
                        candidate_groups={
                            first["label"]: [first["label"]],
                            second["label"]: [second["label"]],
                        },
                        focus=queried,
                        comparison=comparison,
                        surface=surface,
                        order=order,
                        contrast_group_id=f"{split}_color_pair_{pair_key}_s{surface}_o{order}",
                        contrast_label=queried["id"],
                        independent_unit_id=f"{split}_color_pair_{pair_key}",
                        query_side_object_id=queried["id"],
                        candidate_left=left,
                        candidate_right=right,
                    ))

    return rows


def _span_matches(
    raw_prompt: str,
    span: Any,
    expected_text: str | None,
) -> bool:
    if expected_text is None:
        return span is None
    if not isinstance(span, dict) or set(span) != {"start", "end", "text"}:
        return False
    start, end, text = span["start"], span["end"], span["text"]
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or text != expected_text
        or not 0 <= start < end <= len(raw_prompt)
    ):
        return False
    return raw_prompt[start:end] == expected_text


def _prompt_template_audit() -> dict[str, Any]:
    registries = {
        "direct|category": [
            _direct_prompt_record("__OBJECT__", "category", surface)["raw_prompt"]
            for surface in DIRECT_SURFACES
        ],
        "direct|color": [
            _direct_prompt_record("__OBJECT__", "color", surface)["raw_prompt"]
            for surface in DIRECT_SURFACES
        ],
        "selection|category|fruit": [
            _selection_prompt_record(
                "category", "fruit", "__LEFT__", "__RIGHT__", surface
            )["raw_prompt"]
            for surface in SELECTION_SURFACES
        ],
        "selection|category|not_fruit": [
            _selection_prompt_record(
                "category", "not fruit", "__LEFT__", "__RIGHT__", surface
            )["raw_prompt"]
            for surface in SELECTION_SURFACES
        ],
        "selection|color": [
            _selection_prompt_record(
                "color", "__COLOR__", "__LEFT__", "__RIGHT__", surface
            )["raw_prompt"]
            for surface in SELECTION_SURFACES
        ],
    }
    expected_counts = {
        key: len(DIRECT_SURFACES) if key.startswith("direct|")
        else len(SELECTION_SURFACES)
        for key in registries
    }
    unique_counts = {
        key: len({normalize_prompt(prompt) for prompt in prompts})
        for key, prompts in registries.items()
    }
    return {
        "expected_surface_counts": expected_counts,
        "actual_surface_counts": {
            key: len(prompts) for key, prompts in registries.items()
        },
        "normalized_unique_template_counts": unique_counts,
        "surface_sets": {
            "direct": list(DIRECT_SURFACES),
            "selection": list(SELECTION_SURFACES),
        },
        "valid": all(
            len(registries[key]) == expected_counts[key]
            and unique_counts[key] == expected_counts[key]
            for key in registries
        ),
    }


def _case_grid_audit(
    by_split: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []

    def fail(label: str) -> None:
        if label not in failures:
            failures.append(label)

    template_audit = _prompt_template_audit()
    if not template_audit["valid"]:
        fail("prompt_template_registry")
    split_audits: dict[str, Any] = {}

    for split in SPLITS:
        rows = by_split[split]
        items = OBJECTS[split]
        object_by_id = {item["id"]: item for item in items}
        fruits = [item for item in items if item["is_fruit"]]
        controls = [item for item in items if not item["is_fruit"]]
        by_color_fruit = {item["color"]: item for item in fruits}
        by_color_control = {item["color"]: item for item in controls}
        unit_rows: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            unit_rows.setdefault(row["independent_unit_id"], []).append(row)

        direct_specs = {
            f"{split}_direct_{item['id']}_{relation}": (item, relation)
            for item in items
            for relation in RELATIONS
        }
        selection_specs: dict[str, dict[str, Any]] = {}
        for color in sorted(by_color_fruit):
            fruit = by_color_fruit[color]
            control = by_color_control[color]
            unit_id = f"{split}_category_pair_{color}"
            selection_specs[unit_id] = {
                "relation": "category",
                "first": fruit,
                "second": control,
                "same_color": True,
            }
        ordered_fruits = sorted(fruits, key=lambda item: item["id"])
        for index, first in enumerate(ordered_fruits):
            second = ordered_fruits[(index + 1) % len(ordered_fruits)]
            pair_key = "__".join(sorted((first["id"], second["id"])))
            unit_id = f"{split}_color_pair_{pair_key}"
            if unit_id in selection_specs:
                fail(f"{split}:selection_unit_collision")
            selection_specs[unit_id] = {
                "relation": "color",
                "first": first,
                "second": second,
                "same_color": False,
            }

        expected_units = set(direct_specs) | set(selection_specs)
        if set(unit_rows) != expected_units:
            fail(f"{split}:independent_unit_registry")

        direct_complete = True
        for unit_id, (item, relation) in direct_specs.items():
            bank = unit_rows.get(unit_id, [])
            expected_surfaces = Counter(DIRECT_SURFACES)
            actual_surfaces = Counter(row.get("surface_id") for row in bank)
            if len(bank) != len(DIRECT_SURFACES) or actual_surfaces != expected_surfaces:
                direct_complete = False
                fail(f"{split}:direct_surface_grid")
                continue
            for row in bank:
                surface = row["surface_id"]
                prompt_record = _direct_prompt_record(item["label"], relation, surface)
                expected_target = (
                    "yes" if item["is_fruit"] else "no"
                ) if relation == "category" else item["color"]
                expected_groups = (
                    {"yes": ["yes"], "no": ["no"]}
                    if relation == "category"
                    else all_alias_groups(split, relation)
                )
                fields_ok = all((
                    row.get("schema_version") == CASE_SCHEMA_VERSION,
                    row.get("split") == split,
                    row.get("interface") == "direct",
                    row.get("relation") == relation,
                    row.get("focus_object_id") == item["id"],
                    row.get("comparison_object_id") is None,
                    row.get("candidate_left_id") is None,
                    row.get("candidate_right_id") is None,
                    row.get("query_side_object_id") == item["id"],
                    row.get("order") is None,
                    row.get("target") == expected_target,
                    row.get("candidate_groups") == expected_groups,
                    row.get("raw_prompt") == prompt_record["raw_prompt"],
                    row.get("prompt_template_id")
                    == prompt_record["prompt_template_id"],
                    row.get("relation_contract_id")
                    == prompt_record["relation_contract_id"],
                    row.get("query_value") == prompt_record["query_value"],
                    row.get("raw_role_char_spans")
                    == prompt_record["raw_role_char_spans"],
                    row.get("contrast_group_id")
                    == f"{split}_direct_{item['id']}_s{surface}",
                    row.get("contrast_label") == relation,
                    row.get("normalized_prompt_sha256")
                    == stable_hash(normalize_prompt(row["raw_prompt"])),
                ))
                spans = row.get("raw_role_char_spans", {})
                span_ok = (
                    isinstance(spans, dict)
                    and set(spans) == {"focus", "comparison", "query_anchor"}
                    and _span_matches(row["raw_prompt"], spans.get("focus"), item["label"])
                    and _span_matches(row["raw_prompt"], spans.get("comparison"), None)
                    and _span_matches(
                        row["raw_prompt"],
                        spans.get("query_anchor"),
                        prompt_record["query_value"],
                    )
                )
                if not fields_ok or not span_ok:
                    direct_complete = False
                    fail(f"{split}:direct_row_contract")

        selection_complete = True
        binary_contrast_count = 0
        for unit_id, spec in selection_specs.items():
            bank = unit_rows.get(unit_id, [])
            first, second = spec["first"], spec["second"]
            pair_ids = (first["id"], second["id"])
            expected_grid = Counter(
                (query_side, surface, order)
                for query_side in pair_ids
                for surface in SELECTION_SURFACES
                for order in SELECTION_ORDERS
            )
            actual_grid = Counter(
                (
                    row.get("query_side_object_id"),
                    row.get("surface_id"),
                    row.get("order"),
                )
                for row in bank
            )
            if len(bank) != 16 or actual_grid != expected_grid:
                selection_complete = False
                fail(f"{split}:selection_query_surface_order_grid")
                continue
            if spec["relation"] == "category":
                if not (
                    first["is_fruit"] is True
                    and second["is_fruit"] is False
                    and first["color"] == second["color"]
                ):
                    selection_complete = False
                    fail(f"{split}:category_pair_contract")
            elif not (
                first["is_fruit"] is True
                and second["is_fruit"] is True
                and first["color"] != second["color"]
            ):
                selection_complete = False
                fail(f"{split}:color_pair_contract")

            for row in bank:
                query_side = row["query_side_object_id"]
                queried = object_by_id[query_side]
                comparison = second if queried["id"] == first["id"] else first
                order = row["order"]
                left, right = (first, second) if order == 0 else (second, first)
                query_value = (
                    "fruit" if queried["is_fruit"] else "not fruit"
                ) if spec["relation"] == "category" else queried["color"]
                prompt_record = _selection_prompt_record(
                    spec["relation"],
                    query_value,
                    left["label"],
                    right["label"],
                    row["surface_id"],
                )
                expected_spans = _selection_role_spans(
                    prompt_record, queried, comparison, left, right
                )
                expected_group = (
                    f"{unit_id}_s{row['surface_id']}_o{order}"
                )
                expected_candidates = {
                    first["label"]: [first["label"]],
                    second["label"]: [second["label"]],
                }
                fields_ok = all((
                    row.get("schema_version") == CASE_SCHEMA_VERSION,
                    row.get("split") == split,
                    row.get("interface") == "selection",
                    row.get("relation") == spec["relation"],
                    row.get("focus_object_id") == queried["id"],
                    row.get("comparison_object_id") == comparison["id"],
                    row.get("candidate_left_id") == left["id"],
                    row.get("candidate_right_id") == right["id"],
                    row.get("target") == queried["label"],
                    row.get("candidate_groups") == expected_candidates,
                    row.get("query_value") == query_value,
                    row.get("query_anchor_fragment") == query_value,
                    row.get("raw_prompt") == prompt_record["raw_prompt"],
                    row.get("prompt_template_id")
                    == prompt_record["prompt_template_id"],
                    row.get("relation_contract_id")
                    == prompt_record["relation_contract_id"],
                    row.get("raw_role_char_spans") == expected_spans,
                    row.get("contrast_group_id") == expected_group,
                    row.get("contrast_label") == queried["id"],
                    row.get("normalized_prompt_sha256")
                    == stable_hash(normalize_prompt(row["raw_prompt"])),
                ))
                spans = row.get("raw_role_char_spans", {})
                span_ok = (
                    isinstance(spans, dict)
                    and set(spans) == {"focus", "comparison", "query_anchor"}
                    and _span_matches(
                        row["raw_prompt"], spans.get("focus"), queried["label"]
                    )
                    and _span_matches(
                        row["raw_prompt"],
                        spans.get("comparison"),
                        comparison["label"],
                    )
                    and _span_matches(
                        row["raw_prompt"], spans.get("query_anchor"), query_value
                    )
                )
                if not fields_ok or not span_ok:
                    selection_complete = False
                    fail(f"{split}:selection_row_contract")

            for surface in SELECTION_SURFACES:
                for order in SELECTION_ORDERS:
                    contrast = [
                        row for row in bank
                        if row["surface_id"] == surface and row["order"] == order
                    ]
                    expected_left, expected_right = (
                        (first, second) if order == 0 else (second, first)
                    )
                    contrast_ok = (
                        len(contrast) == 2
                        and {row["focus_object_id"] for row in contrast} == set(pair_ids)
                        and {row["comparison_object_id"] for row in contrast} == set(pair_ids)
                        and {row["query_side_object_id"] for row in contrast} == set(pair_ids)
                        and {row["contrast_label"] for row in contrast} == set(pair_ids)
                        and {row["target"] for row in contrast}
                        == {first["label"], second["label"]}
                        and len({row["query_value"] for row in contrast}) == 2
                        and all(
                            row["candidate_left_id"] == expected_left["id"]
                            and row["candidate_right_id"] == expected_right["id"]
                            and row["focus_object_id"] != row["comparison_object_id"]
                            for row in contrast
                        )
                    )
                    if contrast_ok:
                        binary_contrast_count += 1
                    else:
                        selection_complete = False
                        fail(f"{split}:selection_binary_complement")

        split_valid = (
            len(direct_specs) == 24
            and len(selection_specs) == 12
            and len(expected_units) == 36
            and direct_complete
            and selection_complete
            and binary_contrast_count == 96
        )
        if not split_valid:
            fail(f"{split}:complete_36_unit_grid")
        split_audits[split] = {
            "valid": split_valid,
            "analysis_unit_count": len(unit_rows),
            "expected_analysis_unit_count": 36,
            "direct_analysis_unit_count": len(direct_specs),
            "selection_analysis_unit_count": len(selection_specs),
            "direct_unit_case_counts": {
                unit_id: len(unit_rows.get(unit_id, []))
                for unit_id in sorted(direct_specs)
            },
            "selection_unit_case_counts": {
                unit_id: len(unit_rows.get(unit_id, []))
                for unit_id in sorted(selection_specs)
            },
            "direct_surface_grid_complete": direct_complete,
            "selection_query_side_surface_order_grid_complete": selection_complete,
            "selection_binary_complement_group_count": binary_contrast_count,
            "expected_selection_binary_complement_group_count": 96,
            "statistical_independence_claimed": False,
        }

    return {
        "valid": not failures,
        "analysis_units_per_split": 36,
        "surface_and_order_are_paired_repeats": True,
        "statistical_independence_claimed": False,
        "prompt_template_audit": template_audit,
        "splits": split_audits,
    }, failures


def build_all() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_split = {split: build_split(split) for split in SPLITS}
    open_rows = [row for split in OPEN_SPLITS for row in by_split[split]]
    sealed_rows = by_split["sealed"]
    all_rows = open_rows + sealed_rows

    failures: list[str] = []
    expected_per_split = 336
    if any(len(rows) != expected_per_split for rows in by_split.values()):
        failures.append("case_count_per_split")
    if len({row["case_id"] for row in all_rows}) != len(all_rows):
        failures.append("case_id_collision")
    if len({row["normalized_prompt_sha256"] for row in all_rows}) != len(all_rows):
        failures.append("normalized_prompt_collision")
    object_sets = {
        split: {item["id"] for item in OBJECTS[split]}
        for split in SPLITS
    }
    prior_sealed_object_overlap = set().union(*object_sets.values()) & PRIOR_SEALED_OBJECT_IDS
    if prior_sealed_object_overlap:
        failures.append("prior_sealed_object_overlap")
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            if object_sets[left] & object_sets[right]:
                failures.append(f"object_overlap:{left}:{right}")
    strata = Counter(
        (row["split"], row["relation"], row["interface"])
        for row in all_rows
    )
    expected_strata = {
        (split, relation, interface): (72 if interface == "direct" else 96)
        for split in SPLITS
        for relation in RELATIONS
        for interface in INTERFACES
    }
    if dict(strata) != expected_strata:
        failures.append("stratum_count")
    case_grid_audit, grid_failures = _case_grid_audit(by_split)
    failures.extend(grid_failures)
    cross_model_rule_audit = {
        "all_three_same_stage_pass_authorizes_comparison": cross_model_observational_comparison_authorized(
            "discovery", {model: True for model in MODELS}
        ),
        "one_model_failure_blocks": not cross_model_observational_comparison_authorized(
            "discovery", {**{model: True for model in MODELS}, "glm4": False}
        ),
        "missing_model_blocks": not cross_model_observational_comparison_authorized(
            "discovery", {model: True for model in MODELS[:-1]}
        ),
        "extra_model_blocks": not cross_model_observational_comparison_authorized(
            "discovery", {**{model: True for model in MODELS}, "extra": True}
        ),
        "invalid_stage_blocks": not cross_model_observational_comparison_authorized(
            "mixed_stage", {model: True for model in MODELS}
        ),
        "truthy_non_boolean_blocks": not cross_model_observational_comparison_authorized(
            "discovery", {**{model: True for model in MODELS}, "glm4": 1}
        ),
    }
    if not all(cross_model_rule_audit.values()):
        failures.append("cross_model_observational_comparison_rule")
    for row in all_rows:
        if row["target"] not in row["candidate_groups"]:
            failures.append("target_missing_from_candidate_groups")
            break
        if row["target"] not in row["target_aliases"]:
            failures.append("canonical_target_missing_from_aliases")
            break
        alias_owners: dict[str, set[str]] = {}
        for canonical, group_aliases in row["candidate_groups"].items():
            for alias in group_aliases:
                alias_owners.setdefault(alias.casefold(), set()).add(canonical)
        if any(len(owners) != 1 for owners in alias_owners.values()):
            failures.append("candidate_alias_collision")
            break

    prior_hashes: set[str] = set()
    prior_open_identities, prior_row_banks = prior_open_file_snapshots()
    for bank in prior_row_banks:
        for row in bank:
            raw = row.get("raw_prompt")
            if isinstance(raw, str):
                prior_hashes.add(stable_hash(normalize_prompt(raw)))
    prior_overlap = sum(
        row["normalized_prompt_sha256"] in prior_hashes for row in all_rows
    )
    if prior_overlap:
        failures.append("prior_open_prompt_overlap")

    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    tokenizer_audit: dict[str, Any] = {}
    for model in MODELS:
        tokenizer = tokenizers[model]
        token_counts: list[int] = []
        target_token_counts: list[int] = []
        empty_targets = 0
        target_too_long_for_eos = 0
        for row in all_rows:
            rendered = render_chat(tokenizer, model, row["raw_prompt"])
            prompt_ids = tokenizer(rendered, add_special_tokens=True)["input_ids"]
            token_counts.append(len(prompt_ids))
            for alias in row["target_aliases"]:
                target_ids = tokenizer(alias, add_special_tokens=False)["input_ids"]
                target_token_counts.append(len(target_ids))
                if not target_ids:
                    empty_targets += 1
                if len(target_ids) + 1 > MAX_NEW_TOKENS:
                    target_too_long_for_eos += 1
        tokenizer_audit[model] = {
            "tokenizer_class": type(tokenizer).__name__,
            "case_count": len(all_rows),
            "prompt_token_min": min(token_counts),
            "prompt_token_max": max(token_counts),
            "empty_target_alias_count": empty_targets,
            "target_token_min": min(target_token_counts),
            "target_token_max": max(target_token_counts),
            "targets_too_long_to_leave_eos_budget": target_too_long_for_eos,
        }
        if empty_targets:
            failures.append(f"empty_target_alias:{model}")
        if target_too_long_for_eos:
            failures.append(f"target_exceeds_eos_budget:{model}")

    audit = {
        "valid": not failures,
        "failures": failures,
        "cases_per_split": {split: len(rows) for split, rows in by_split.items()},
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "object_count_per_split": {
            split: len(OBJECTS[split]) for split in SPLITS
        },
        "fruit_count_per_split": {
            split: sum(item["is_fruit"] for item in OBJECTS[split])
            for split in SPLITS
        },
        "nonfruit_count_per_split": {
            split: sum(not item["is_fruit"] for item in OBJECTS[split])
            for split in SPLITS
        },
        "stratum_counts": {
            "|".join(key): value for key, value in sorted(strata.items())
        },
        "split_object_overlap_counts": {
            f"{left}|{right}": len(object_sets[left] & object_sets[right])
            for left_index, left in enumerate(SPLITS)
            for right in SPLITS[left_index + 1:]
        },
        "normalized_prompt_unique_count": len({
            row["normalized_prompt_sha256"] for row in all_rows
        }),
        "case_grid_audit": case_grid_audit,
        "cross_model_observational_comparison_rule_audit": cross_model_rule_audit,
        "prior_open_file_identities": prior_open_identities,
        "prior_open_files_read": [
            identity["path"] for identity in prior_open_identities
        ],
        "required_prior_open_file_count": len(PRIOR_OPEN_CASE_PATHS),
        "prior_sealed_files_read": [],
        "prior_sealed_object_ids_registered_from_public_protocols": sorted(
            PRIOR_SEALED_OBJECT_IDS
        ),
        "prior_sealed_object_overlap": sorted(prior_sealed_object_overlap),
        "prior_open_prompt_overlap_count": prior_overlap,
        "tokenizer_audit": tokenizer_audit,
        "model_weights_loaded": False,
        "cuda_used": False,
        "sealed_definition_cpu_read_for_static_audit": True,
        "sealed_model_or_result_read_for_analysis": False,
    }
    return open_rows, sealed_rows, audit


def protocol_payload(
    open_rows: list[dict[str, Any]],
    sealed_rows: list[dict[str, Any]],
    audit: dict[str, Any],
    created_at: str,
    model_artifacts: dict[str, Any],
    source_seals: dict[str, dict[str, Any]],
    source_script_sha256: str,
) -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at_utc": created_at,
        "research_route": "gpt5_phase576_natural_fruit_reuse_difference",
        "evidence_order": [
            "freeze_external_denominator",
            "qualify_cuda_repeat_forward_and_cached_generation_hidden_state_API_on_synthetic_nonresearch_input",
            "run_open_behavior_qualification",
            "collect_all-layer residual states at every actually executed rendered-prompt and cached-generation position only for qualified models",
            "discover_repeated_internal_events_without_favored_coordinates",
            "freeze_discovered_event_definitions",
            "confirm_on_independent_and_heldout_open_splits",
            "freeze_only_descriptive_candidate_relations_after_observational_confirmation",
            "require_separate_controlled_intervention_before_any_causal_mechanism_formula",
        ],
        "models_in_required_execution_order": list(MODELS),
        "model_artifact_identities": model_artifacts,
        "prior_open_file_identities": audit["prior_open_file_identities"],
        "splits": list(SPLITS),
        "open_splits": list(OPEN_SPLITS),
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "cases_per_split": audit["cases_per_split"],
        "relations": list(RELATIONS),
        "relation_contracts": {
            "category": CATEGORY_RELATION_CONTRACT_ID,
            "color": COLOR_RELATION_CONTRACT_ID,
        },
        "interfaces": list(INTERFACES),
        "behavior_repeats": list(BEHAVIOR_REPEATS),
        "behavior_batch_size": BEHAVIOR_BATCH_SIZE,
        "trace_batch_size": TRACE_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "behavior_gate": {
            "analysis_unit": "object_x_relation for direct; semantic object pair for selection",
            "analysis_units_per_split": 36,
            "statistical_independence_claimed": False,
            "surface_and_order_are_paired_repeats_not_independent_samples": True,
            "minimum_stable_category_direct_units_of_12": 10,
            "minimum_stable_color_direct_units_of_12": 10,
            "minimum_stable_category_selection_units_of_6": 5,
            "minimum_stable_color_selection_units_of_6": 5,
            "direct_unit_minimum_stable_surfaces_of_6": 5,
            "selection_unit_minimum_stable_cases_of_16": 12,
            "selection_unit_minimum_stable_cases_each_query_side_of_8": 6,
            "minimum_case_level_stable_rate_diagnostic": 0.75,
            "deterministic_generation_required": True,
            "valid_case_requires_exact_short_answer": True,
            "valid_case_requires_eos_before_budget": True,
            "model_failure_is_behavior_blocked_not_mechanism_absence": True,
        },
        "case_grid_contract": audit["case_grid_audit"],
        "cross_model_observational_comparison_policy": {
            "single_model_trace_allowed_after_that_model_same_stage_pass": True,
            "cross_model_comparison_authorized_field_is_derived_not_trusted": True,
            "required_models_in_order": list(MODELS),
            "required_exact_model_set": True,
            "required_same_stage": True,
            "all_three_models_must_pass_same_stage": True,
            "three_model_replication_is_not_statistical_independence": True,
            "common_structure_claim_requires_frozen_rule_confirmation": True,
        },
        "trace_policy": {
            "candidate_coordinates_before_trace": [],
            "candidate_mechanism_formulas_before_trace": [],
            "record_all_model_layers": True,
            "primary_medium_scale_state": "full executed residual trajectory: step-0 rendered-prompt tokens plus every generated EOS or absorbing PAD token actually fed back by batched cached decoding, at all model layers",
            "executed_position_scope": {
                "step0_prefill": "every token in the rendered prompt submitted to the model",
                "cached_generation": "every emitted token that is subsequently fed back and actually executed as cached model input, including EOS and absorbing PAD positions for rows that finish before their batch peers",
                "unexecuted_terminal_token": "an emitted EOS token or the final token at the generation budget has no newly computed residual state when it is not fed back",
            },
            "batch_absorbing_eos_and_pad_feedback_positions_included": True,
            "full_executed_residual_trajectory": True,
            "full_token_trajectory": True,
            "role_labels": [
                "focus_object_last_token",
                "comparison_object_last_token_when_present",
                "query_anchor_last_token",
                "answer_boundary",
            ],
            "raw_role_spans_are_labels_not_sampling_filters": True,
            "full_vectors_at_every_executed_position": True,
            "prefill_only": False,
            "residual_storage_dtype": "bfloat16",
            "finite_value_check_required_before_publish": True,
            "first_batch_exact_repeat_required": True,
            "attention_component_states_recorded": False,
            "mlp_component_states_recorded": False,
            "unexecuted_terminal_token_has_internal_state": False,
            "raw_role_char_spans_frozen_per_case": True,
            "role_token_positions_must_derive_from_frozen_char_spans": True,
            "substring_last_occurrence_inference_allowed": False,
            "sealed_split_allowed": False,
            "causal_intervention_allowed": False,
            "head_scan_allowed": False,
            "neuron_scan_allowed": False,
            "cross_model_raw_coordinate_alignment_allowed": False,
        },
        "staged_analysis_seal_policy": {
            "initial_stage_sources": source_seals,
            "post_trace_discovery_analysis_source_must_not_be_initially_sealed": True,
            "post_trace_discovery_analysis_source_first_sealed_in_registry": True,
            "candidate_specification_first_sealed_in_registry": True,
            "confirmation_rule_first_sealed_in_registry": True,
            "standalone_source_static_import_allowlist_required": True,
            "local_analysis_file_dependencies_allowed": [],
            "analysis_execution_os_sandboxed": False,
            "analysis_io_isolation_mechanically_enforced": False,
            "sealed_access_claim_is_code_reviewed_declaration": True,
        },
        "atomic_freeze_policy": {
            "all_bytes_and_hash_edges_precomputed_before_staging": True,
            "same_filesystem_staging_directory": True,
            "exclusive_freeze_lock": True,
            "exclusive_file_creation_in_staging": True,
            "readback_hash_verification_before_publish": True,
            "single_directory_rename_publish": True,
            "final_commit_marker_required": True,
            "overwrite_allowed": False,
        },
        "sealed_policy": {
            "new_sealed_objects": True,
            "holdout_is_blind": False,
            "sealed_definition_is_public_in_source": True,
            "sealed_definition_cpu_read_during_freeze": True,
            "prior_sealed_files_read": False,
            "stage_specific_behavior_must_pass_before_that_stage_internal_trace": True,
            "open_discovery_and_confirmation_order_is_mechanically_enforced": True,
            "sealed_may_be_opened_once": True,
        },
        "scientific_limits": [
            "natural category and typical-color labels are operational contracts, not exhaustive truth",
            "behavior qualification does not prove an internal mechanism",
            "repeated residual structure is observational until controlled intervention",
            "the trace covers every actually executed residual position, while role spans are labels rather than sampling filters",
            "attention and MLP component states are not recorded, so residual trajectories do not localize a head, neuron, or component computation",
            "an EOS token or final budget token that is emitted but never fed back has no newly executed internal state in the cached trajectory",
            "residual tensors are stored as BF16; finite checks and an exact first-batch repeat are integrity controls, not causal evidence",
            "336 surface cases per split reduce to 36 analysis units; statistical independence is not claimed",
            "a shared three-model pattern is replicated evidence, not three statistically independent samples",
            "int8 weight execution must be reported and is not numerically identical to BF16",
            "this local fruit task does not represent complete language or AGI",
            "post-trace analysis source is hash sealed and import audited but not OS-sandboxed; its no-sealed-access statement remains a code-review trust assumption",
        ],
        "source_script": str(source_path.relative_to(ROOT)).replace("\\", "/"),
        "source_script_sha256": source_script_sha256,
        "stage_source_seals": source_seals,
    }


def _out_relative(path: Path) -> Path:
    try:
        relative = path.relative_to(OUT_DIR)
    except ValueError as exc:
        raise RuntimeError(f"artifact is outside Phase576 output root: {path}") from exc
    if relative == Path(".") or ".." in relative.parts:
        raise RuntimeError(f"invalid Phase576 artifact relative path: {relative}")
    return relative


def _validate_model_artifact_registry(model_artifacts: dict[str, Any]) -> None:
    if not isinstance(model_artifacts, dict) or set(model_artifacts) != set(MODELS):
        raise RuntimeError("model artifact registry does not contain the exact model set")
    for model in MODELS:
        identity = model_artifacts[model]
        if not isinstance(identity, dict) or "identity_sha256" not in identity:
            raise RuntimeError(f"{model}: malformed full artifact identity")
        payload = {
            key: value for key, value in identity.items() if key != "identity_sha256"
        }
        weights = payload.get("weight_files")
        small_files = payload.get("tokenizer_and_config_files")
        rows = (
            weights + small_files
            if isinstance(weights, list) and isinstance(small_files, list)
            else []
        )

        def valid_file_row(row: Any, *, must_be_weight: bool) -> bool:
            if not isinstance(row, dict) or set(row) != {
                "relative_path",
                "resolved_path",
                "path_is_symlink",
                "size_bytes",
                "sha256",
            }:
                return False
            relative_path = row["relative_path"]
            if not isinstance(relative_path, str):
                return False
            parts = relative_path.split("/")
            valid_relative_path = (
                bool(relative_path)
                and "\\" not in relative_path
                and not relative_path.startswith("/")
                and all(part not in {"", ".", ".."} for part in parts)
            )
            has_weight_suffix = relative_path.lower().endswith(".safetensors")
            return all((
                valid_relative_path,
                isinstance(row["resolved_path"], str)
                and bool(row["resolved_path"]),
                isinstance(row["path_is_symlink"], bool),
                isinstance(row["size_bytes"], int)
                and row["size_bytes"] >= 0,
                _is_sha256(row["sha256"]),
                has_weight_suffix is must_be_weight,
                not must_be_weight or row["size_bytes"] > 0,
            ))

        weight_relative_paths = [
            row.get("relative_path")
            for row in (weights or [])
            if isinstance(row, dict)
        ]
        non_weight_relative_paths = [
            row.get("relative_path")
            for row in (small_files or [])
            if isinstance(row, dict)
        ]
        relative_paths = weight_relative_paths + non_weight_relative_paths
        checks = (
            _is_sha256(identity["identity_sha256"]),
            identity["identity_sha256"] == stable_hash(payload),
            isinstance(weights, list) and bool(weights),
            isinstance(small_files, list) and bool(small_files),
            payload.get("artifact_inventory_mode")
            == "recursive_all_regular_files.v1",
            payload.get("nested_directory_symlinks_allowed") is False,
            payload.get("artifact_file_count") == len(rows),
            payload.get("artifact_total_bytes")
            == sum(row.get("size_bytes", -1) for row in rows),
            payload.get("weight_file_count") == len(weights or []),
            payload.get("weight_total_bytes")
            == sum(row.get("size_bytes", -1) for row in (weights or [])),
            payload.get("non_weight_file_count") == len(small_files or []),
            payload.get("non_weight_total_bytes")
            == sum(row.get("size_bytes", -1) for row in (small_files or [])),
            all(valid_file_row(row, must_be_weight=True) for row in (weights or [])),
            all(
                valid_file_row(row, must_be_weight=False)
                for row in (small_files or [])
            ),
            weight_relative_paths == sorted(weight_relative_paths),
            non_weight_relative_paths == sorted(non_weight_relative_paths),
            len(relative_paths) == len(set(relative_paths)),
        )
        if not all(checks):
            raise RuntimeError(f"{model}: internally inconsistent artifact identity")


def _precompute_freeze_artifacts(
    open_rows: list[dict[str, Any]],
    sealed_rows: list[dict[str, Any]],
    audit: dict[str, Any],
    created_at: str,
    model_artifacts: dict[str, Any],
    source_seals: dict[str, dict[str, Any]],
) -> dict[Path, bytes]:
    source_path = Path(__file__).resolve()
    source_sha = sha256_file(source_path)
    source_key = str(source_path.relative_to(ROOT)).replace("\\", "/")
    if source_seals.get(source_key, {}).get("sha256") != source_sha:
        raise RuntimeError("protocol source hash differs from initial source seal")
    _validate_model_artifact_registry(model_artifacts)

    open_blob = jsonl_bytes(open_rows)
    split_blobs = {
        split: jsonl_bytes([row for row in open_rows if row["split"] == split])
        for split in OPEN_SPLITS
    }
    sealed_blob = jsonl_bytes(sealed_rows)
    commitment = {
        "schema_version": "phase576_sealed_commitment.v2",
        "phase_id": PHASE,
        "created_at_utc": created_at,
        "sealed_case_count": len(sealed_rows),
        "sealed_cases_sha256": sha256_bytes(sealed_blob),
        "holdout_is_blind": False,
        "sealed_definition_is_public_in_source": True,
        "sealed_definition_cpu_read_during_freeze": True,
        "sealed_model_opened": False,
        "sealed_model_access_count": 0,
        "sealed_result_analysis_access_count": 0,
        "prior_sealed_files_read": False,
    }
    commitment_blob = json_bytes(commitment)
    payload = protocol_payload(
        open_rows,
        sealed_rows,
        audit,
        created_at,
        model_artifacts,
        source_seals,
        source_sha,
    )
    payload["open_cases_sha256"] = sha256_bytes(open_blob)
    payload["open_case_sha256_by_split"] = {
        split: sha256_bytes(blob) for split, blob in split_blobs.items()
    }
    payload["sealed_commitment_sha256"] = sha256_bytes(commitment_blob)
    protocol_blob = json_bytes(payload)
    audit_payload = {
        "schema_version": "phase576_static_audit.v2",
        "phase_id": PHASE,
        "created_at_utc": created_at,
        **audit,
        "open_cases_sha256": sha256_bytes(open_blob),
        "open_case_sha256_by_split": {
            split: sha256_bytes(blob) for split, blob in split_blobs.items()
        },
        "sealed_cases_sha256": sha256_bytes(sealed_blob),
        "sealed_commitment_sha256": sha256_bytes(commitment_blob),
        "protocol_sha256": sha256_bytes(protocol_blob),
    }
    audit_blob = json_bytes(audit_payload)

    artifacts: dict[Path, bytes] = {
        _out_relative(OPEN_CASES_PATH): open_blob,
        **{
            _out_relative(OPEN_SPLIT_CASE_PATHS[split]): blob
            for split, blob in split_blobs.items()
        },
        _out_relative(SEALED_CASES_PATH): sealed_blob,
        _out_relative(SEALED_COMMITMENT_PATH): commitment_blob,
        _out_relative(PROTOCOL_PATH): protocol_blob,
        _out_relative(STATIC_AUDIT_PATH): audit_blob,
    }
    artifact_hashes = {
        str(relative).replace("\\", "/"): sha256_bytes(blob)
        for relative, blob in sorted(artifacts.items(), key=lambda item: str(item[0]))
    }
    commit_payload = {
        "schema_version": "phase576_freeze_commit.v1",
        "phase_id": PHASE,
        "created_at_utc": created_at,
        "complete": True,
        "overwrite_allowed": False,
        "atomic_directory_publish": True,
        "artifact_count": len(artifacts),
        "artifact_sha256_by_path": artifact_hashes,
        "source_seals_sha256": stable_hash(source_seals),
        "model_artifact_identities_sha256": stable_hash(model_artifacts),
        "prior_open_file_identities_sha256": stable_hash(
            audit["prior_open_file_identities"]
        ),
    }
    artifacts[_out_relative(FREEZE_COMMIT_PATH)] = json_bytes(commit_payload)
    return artifacts


def _validate_precomputed_freeze_artifacts(
    artifacts: dict[Path, bytes],
    open_rows: list[dict[str, Any]],
    sealed_rows: list[dict[str, Any]],
    audit: dict[str, Any],
    model_artifacts: dict[str, Any],
    source_seals: dict[str, dict[str, Any]],
) -> None:
    expected_paths = {
        _out_relative(OPEN_CASES_PATH),
        *(_out_relative(path) for path in OPEN_SPLIT_CASE_PATHS.values()),
        _out_relative(SEALED_CASES_PATH),
        _out_relative(SEALED_COMMITMENT_PATH),
        _out_relative(PROTOCOL_PATH),
        _out_relative(STATIC_AUDIT_PATH),
        _out_relative(FREEZE_COMMIT_PATH),
    }
    if set(artifacts) != expected_paths:
        raise RuntimeError("precomputed freeze artifact path registry drift")
    parsed_open = _parse_jsonl_bytes(
        artifacts[_out_relative(OPEN_CASES_PATH)], OPEN_CASES_PATH
    )
    parsed_sealed = _parse_jsonl_bytes(
        artifacts[_out_relative(SEALED_CASES_PATH)], SEALED_CASES_PATH
    )
    if parsed_open != open_rows or parsed_sealed != sealed_rows:
        raise RuntimeError("precomputed case bytes do not round-trip exactly")
    for split, path in OPEN_SPLIT_CASE_PATHS.items():
        parsed = _parse_jsonl_bytes(artifacts[_out_relative(path)], path)
        expected = [row for row in open_rows if row["split"] == split]
        if parsed != expected:
            raise RuntimeError(f"{split}: precomputed split bytes differ from denominator")

    commitment = json.loads(
        artifacts[_out_relative(SEALED_COMMITMENT_PATH)].decode("utf-8")
    )
    frozen = json.loads(artifacts[_out_relative(PROTOCOL_PATH)].decode("utf-8"))
    frozen_audit = json.loads(
        artifacts[_out_relative(STATIC_AUDIT_PATH)].decode("utf-8")
    )
    commit = json.loads(
        artifacts[_out_relative(FREEZE_COMMIT_PATH)].decode("utf-8")
    )
    artifact_hashes = {
        str(relative).replace("\\", "/"): sha256_bytes(blob)
        for relative, blob in artifacts.items()
        if relative != _out_relative(FREEZE_COMMIT_PATH)
    }
    split_hashes = {
        split: sha256_bytes(artifacts[_out_relative(path)])
        for split, path in OPEN_SPLIT_CASE_PATHS.items()
    }
    created_times = {
        commitment.get("created_at_utc"),
        frozen.get("created_at_utc"),
        frozen_audit.get("created_at_utc"),
        commit.get("created_at_utc"),
    }
    checks = (
        commitment.get("schema_version") == "phase576_sealed_commitment.v2",
        frozen.get("schema_version") == SCHEMA_VERSION,
        frozen_audit.get("schema_version") == "phase576_static_audit.v2",
        len(created_times) == 1 and None not in created_times,
        audit["valid"] is True and not audit["failures"],
        commitment["sealed_cases_sha256"]
        == sha256_bytes(artifacts[_out_relative(SEALED_CASES_PATH)]),
        frozen["open_cases_sha256"]
        == sha256_bytes(artifacts[_out_relative(OPEN_CASES_PATH)]),
        frozen["open_case_sha256_by_split"] == split_hashes,
        frozen["sealed_commitment_sha256"]
        == sha256_bytes(artifacts[_out_relative(SEALED_COMMITMENT_PATH)]),
        frozen["model_artifact_identities"] == model_artifacts,
        frozen["stage_source_seals"] == source_seals,
        frozen["prior_open_file_identities"]
        == audit["prior_open_file_identities"],
        frozen_audit["protocol_sha256"]
        == sha256_bytes(artifacts[_out_relative(PROTOCOL_PATH)]),
        frozen_audit["open_cases_sha256"]
        == sha256_bytes(artifacts[_out_relative(OPEN_CASES_PATH)]),
        frozen_audit["open_case_sha256_by_split"] == split_hashes,
        frozen_audit["sealed_cases_sha256"]
        == sha256_bytes(artifacts[_out_relative(SEALED_CASES_PATH)]),
        frozen_audit["sealed_commitment_sha256"]
        == sha256_bytes(artifacts[_out_relative(SEALED_COMMITMENT_PATH)]),
        frozen_audit["prior_open_file_identities"]
        == audit["prior_open_file_identities"],
        frozen["case_grid_contract"] == audit["case_grid_audit"],
        frozen_audit["case_grid_audit"] == audit["case_grid_audit"],
        commit["schema_version"] == "phase576_freeze_commit.v1",
        commit["phase_id"] == PHASE,
        commit["complete"] is True,
        commit["overwrite_allowed"] is False,
        commit["atomic_directory_publish"] is True,
        commit["artifact_count"] == len(artifact_hashes),
        commit["artifact_sha256_by_path"] == artifact_hashes,
        commit["source_seals_sha256"] == stable_hash(source_seals),
        commit["model_artifact_identities_sha256"]
        == stable_hash(model_artifacts),
        commit["prior_open_file_identities_sha256"]
        == stable_hash(audit["prior_open_file_identities"]),
    )
    if not all(checks):
        raise RuntimeError("precomputed Phase576 freeze hash graph is inconsistent")


def _write_staging_artifacts(staging: Path, artifacts: dict[Path, bytes]) -> None:
    for relative, payload in sorted(artifacts.items(), key=lambda item: str(item[0])):
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.read_bytes() != payload or sha256_file(destination) != sha256_bytes(payload):
            raise RuntimeError(f"staging readback verification failed: {relative}")


def _verify_freeze_commit(*, verify_sealed_payload_bytes: bool = False) -> dict[str, Any]:
    """Verify the atomic freeze without opening sealed-case bytes by default."""
    if not FREEZE_COMMIT_PATH.is_file():
        raise RuntimeError("Phase576 final freeze commit marker is missing")
    commit = read_json(FREEZE_COMMIT_PATH)
    expected_keys = {
        "schema_version",
        "phase_id",
        "created_at_utc",
        "complete",
        "overwrite_allowed",
        "atomic_directory_publish",
        "artifact_count",
        "artifact_sha256_by_path",
        "source_seals_sha256",
        "model_artifact_identities_sha256",
        "prior_open_file_identities_sha256",
    }
    _require_exact_keys(commit, expected_keys, "freeze commit")
    expected_paths = {
        _out_relative(OPEN_CASES_PATH),
        *(_out_relative(path) for path in OPEN_SPLIT_CASE_PATHS.values()),
        _out_relative(SEALED_CASES_PATH),
        _out_relative(SEALED_COMMITMENT_PATH),
        _out_relative(PROTOCOL_PATH),
        _out_relative(STATIC_AUDIT_PATH),
    }
    normalized_expected = {
        str(relative).replace("\\", "/") for relative in expected_paths
    }
    sealed_relative = str(_out_relative(SEALED_CASES_PATH)).replace("\\", "/")
    if set(commit.get("artifact_sha256_by_path", {})) != normalized_expected:
        raise RuntimeError("Phase576 freeze artifact registry is not exact")
    if any(not (OUT_DIR / relative).is_file() for relative in expected_paths):
        raise RuntimeError("Phase576 frozen artifact is missing")
    actual_hashes = {
        str(relative).replace("\\", "/"): sha256_file(OUT_DIR / relative)
        for relative in expected_paths
        if verify_sealed_payload_bytes
        or str(relative).replace("\\", "/") != sealed_relative
    }
    committed_hashes = commit["artifact_sha256_by_path"]
    frozen = read_json(PROTOCOL_PATH)
    audit = read_json(STATIC_AUDIT_PATH)
    commitment = read_json(SEALED_COMMITMENT_PATH)
    created_times = {
        commit.get("created_at_utc"),
        frozen.get("created_at_utc"),
        audit.get("created_at_utc"),
        commitment.get("created_at_utc"),
    }
    checks = (
        commit["schema_version"] == "phase576_freeze_commit.v1",
        commit["phase_id"] == PHASE,
        commit["complete"] is True,
        commit["overwrite_allowed"] is False,
        commit["atomic_directory_publish"] is True,
        commit["artifact_count"] == len(expected_paths),
        all(committed_hashes[path] == digest for path, digest in actual_hashes.items()),
        len(actual_hashes) == len(expected_paths) - (
            0 if verify_sealed_payload_bytes else 1
        ),
        len(created_times) == 1 and None not in created_times,
        commit["source_seals_sha256"] == stable_hash(frozen["stage_source_seals"]),
        commit["model_artifact_identities_sha256"]
        == stable_hash(frozen["model_artifact_identities"]),
        commit["prior_open_file_identities_sha256"]
        == stable_hash(frozen["prior_open_file_identities"]),
        audit["protocol_sha256"] == actual_hashes[
            str(_out_relative(PROTOCOL_PATH)).replace("\\", "/")
        ],
        _is_sha256(committed_hashes[sealed_relative]),
        commitment.get("sealed_cases_sha256") == committed_hashes[sealed_relative],
        audit.get("sealed_cases_sha256") == committed_hashes[sealed_relative],
    )
    if not all(checks):
        raise RuntimeError("Phase576 final freeze commit verification failed")
    return commit


def self_test() -> dict[str, Any]:
    open_rows, sealed_rows, audit = build_all()
    if not audit["valid"]:
        raise RuntimeError(f"Phase576 static self-test failed: {audit['failures']}")
    return {
        "passed": True,
        "open_case_count": len(open_rows),
        "sealed_case_count": len(sealed_rows),
        "cases_per_split": audit["cases_per_split"],
        "analysis_units_per_split": audit["case_grid_audit"][
            "analysis_units_per_split"
        ],
        "complete_case_grid": audit["case_grid_audit"]["valid"],
        "prompt_template_registry_valid": audit["case_grid_audit"][
            "prompt_template_audit"
        ]["valid"],
        "prior_open_file_identities": audit["prior_open_file_identities"],
        "prior_open_prompt_overlap_count": audit["prior_open_prompt_overlap_count"],
        "cross_model_observational_comparison_rule_valid": all(
            audit["cross_model_observational_comparison_rule_audit"].values()
        ),
        "model_weights_loaded": False,
        "cuda_used": False,
    }


def freeze() -> dict[str, Any]:
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    lock_payload = (
        f"{PHASE}|pid={os.getpid()}|started={now()}\n"
    ).encode("utf-8")
    staging: Path | None = None
    published = False
    try:
        with FREEZE_LOCK_PATH.open("xb") as lock_handle:
            lock_handle.write(lock_payload)
            lock_handle.flush()
            os.fsync(lock_handle.fileno())
        if OUT_DIR.exists():
            raise RuntimeError(
                "Phase576 output directory already exists; overwrite is forbidden"
            )

        # All scientific content, source seals, full model hashes, serialized
        # bytes, and hash edges are complete before staging begins.
        open_rows, sealed_rows, audit = build_all()
        if not audit["valid"]:
            raise RuntimeError(f"Phase576 static audit failed: {audit['failures']}")
        created_at = now()
        source_seals = stage_source_seals()
        model_artifacts = model_artifact_identity()
        artifacts = _precompute_freeze_artifacts(
            open_rows,
            sealed_rows,
            audit,
            created_at,
            model_artifacts,
            source_seals,
        )
        _validate_precomputed_freeze_artifacts(
            artifacts,
            open_rows,
            sealed_rows,
            audit,
            model_artifacts,
            source_seals,
        )

        staging = Path(tempfile.mkdtemp(
            prefix=f".{OUT_DIR.name}.staging-",
            dir=OUT_DIR.parent,
        ))
        _write_staging_artifacts(staging, artifacts)

        # Recheck every external snapshot immediately before the one-step
        # publication. This intentionally rehashes full weights once.
        if stage_source_seals() != source_seals:
            raise RuntimeError("source seal drift before atomic freeze publish")
        current_prior_identities, _ = prior_open_file_snapshots()
        if current_prior_identities != audit["prior_open_file_identities"]:
            raise RuntimeError("prior-open identity drift before atomic freeze publish")
        if model_artifact_identity() != model_artifacts:
            raise RuntimeError("model artifact drift before atomic freeze publish")
        _validate_precomputed_freeze_artifacts(
            {
                relative: (staging / relative).read_bytes()
                for relative in artifacts
            },
            open_rows,
            sealed_rows,
            audit,
            model_artifacts,
            source_seals,
        )
        if OUT_DIR.exists():
            raise RuntimeError("Phase576 output appeared during freeze; refusing publish")
        staging.rename(OUT_DIR)
        published = True
        commit = _verify_freeze_commit(verify_sealed_payload_bytes=True)
        frozen_audit = read_json(STATIC_AUDIT_PATH)
        return {
            "passed": True,
            "files_written": True,
            "atomic_directory_publish": True,
            "freeze_commit_complete": commit["complete"],
            "open_case_count": len(open_rows),
            "sealed_case_count": len(sealed_rows),
            "open_cases_sha256": frozen_audit["open_cases_sha256"],
            "sealed_cases_sha256": frozen_audit["sealed_cases_sha256"],
            "protocol_sha256": frozen_audit["protocol_sha256"],
            "model_weights_loaded": False,
            "cuda_used": False,
        }
    finally:
        if staging is not None and not published and staging.exists():
            expected_parent = OUT_DIR.parent.resolve()
            if (
                staging.parent.resolve() != expected_parent
                or not staging.name.startswith(f".{OUT_DIR.name}.staging-")
            ):
                raise RuntimeError("refusing unsafe staging cleanup")
            shutil.rmtree(staging)
        if FREEZE_LOCK_PATH.exists():
            try:
                owned = FREEZE_LOCK_PATH.read_bytes() == lock_payload
            except OSError:
                owned = False
            if owned:
                FREEZE_LOCK_PATH.unlink()


def verify() -> dict[str, Any]:
    for path in (
        OPEN_CASES_PATH,
        *OPEN_SPLIT_CASE_PATHS.values(),
        SEALED_CASES_PATH,
        SEALED_COMMITMENT_PATH,
        PROTOCOL_PATH,
        STATIC_AUDIT_PATH,
        FREEZE_COMMIT_PATH,
    ):
        if not path.exists():
            raise RuntimeError(f"missing Phase576 artifact: {path}")
    frozen = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    audit = json.loads(STATIC_AUDIT_PATH.read_text(encoding="utf-8"))
    commitment = json.loads(SEALED_COMMITMENT_PATH.read_text(encoding="utf-8"))
    commit = _verify_freeze_commit()
    current_model_artifacts = model_artifact_identity()
    current_source_seals = stage_source_seals()
    current_prior_identities, _ = prior_open_file_snapshots()
    current_open_hash = sha256_file(OPEN_CASES_PATH)
    current_protocol_hash = sha256_file(PROTOCOL_PATH)
    current_commitment_hash = sha256_file(SEALED_COMMITMENT_PATH)
    sealed_relative = str(_out_relative(SEALED_CASES_PATH)).replace("\\", "/")
    committed_sealed_hash = commit["artifact_sha256_by_path"][sealed_relative]
    current_split_hashes = {
        split: sha256_file(path)
        for split, path in OPEN_SPLIT_CASE_PATHS.items()
    }
    checks = {
        "schema_versions": (
            frozen["schema_version"] == SCHEMA_VERSION
            and audit["schema_version"] == "phase576_static_audit.v2"
            and commitment["schema_version"] == "phase576_sealed_commitment.v2"
        ),
        "freeze_commit_complete": commit["complete"] is True,
        "freeze_lock_absent": not FREEZE_LOCK_PATH.exists(),
        "static_audit_valid": audit["valid"] is True and not audit["failures"],
        "audit_protocol_hash": audit["protocol_sha256"] == current_protocol_hash,
        "open_hash": frozen["open_cases_sha256"] == current_open_hash,
        "audit_open_hash": audit["open_cases_sha256"] == current_open_hash,
        "sealed_commitment_hash_chain": (
            commitment["sealed_cases_sha256"] == committed_sealed_hash
        ),
        "audit_sealed_commitment_hash_chain": (
            audit["sealed_cases_sha256"] == committed_sealed_hash
        ),
        "commitment_hash": frozen["sealed_commitment_sha256"]
        == current_commitment_hash,
        "audit_commitment_hash": audit["sealed_commitment_sha256"]
        == current_commitment_hash,
        "source_hash": frozen["source_script_sha256"]
        == sha256_file(Path(__file__).resolve()),
        "model_artifact_identity": frozen["model_artifact_identities"]
        == current_model_artifacts,
        "stage_source_seals": frozen["stage_source_seals"]
        == current_source_seals,
        "open_split_hashes": frozen["open_case_sha256_by_split"]
        == current_split_hashes,
        "audit_open_split_hashes": audit["open_case_sha256_by_split"]
        == current_split_hashes,
        "prior_open_identities": frozen["prior_open_file_identities"]
        == current_prior_identities,
        "audit_prior_open_identities": audit["prior_open_file_identities"]
        == current_prior_identities,
        "complete_case_grid": (
            frozen["case_grid_contract"]["valid"] is True
            and audit["case_grid_audit"]["valid"] is True
            and frozen["case_grid_contract"] == audit["case_grid_audit"]
        ),
        "cross_model_observational_comparison_rule": all(
            audit["cross_model_observational_comparison_rule_audit"].values()
        ),
        "sealed_model_unopened": commitment["sealed_model_opened"] is False,
        "sealed_model_access_zero": commitment["sealed_model_access_count"] == 0,
        "sealed_result_analysis_access_zero": commitment[
            "sealed_result_analysis_access_count"
        ] == 0,
        "holdout_not_misreported_blind": commitment["holdout_is_blind"] is False,
        "no_prior_sealed_read": commitment["prior_sealed_files_read"] is False,
        "no_prefrozen_coordinates": not frozen["trace_policy"]["candidate_coordinates_before_trace"],
        "no_prefrozen_mechanism_formulas": not frozen["trace_policy"]["candidate_mechanism_formulas_before_trace"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase576 verification failed: {checks}")
    return {
        "passed": True,
        "files_written": False,
        "checks": checks,
        "protocol_sha256": current_protocol_hash,
        "freeze_commit_sha256": sha256_file(FREEZE_COMMIT_PATH),
        "sealed_case_payload_bytes_read_for_integrity": False,
        "sealed_case_payload_parsed_for_analysis": False,
        "model_weights_loaded": False,
        "cuda_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.write:
        result = freeze()
    else:
        result = verify()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
