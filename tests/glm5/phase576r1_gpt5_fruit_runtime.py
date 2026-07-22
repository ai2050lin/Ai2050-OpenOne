#!/usr/bin/env python3
"""Immutable runtime erratum and CUDA re-entry for frozen Phase576.

The original Phase576 denominator was frozen correctly, but its engineering
stage was accidentally launched with a CPU-only Python environment.  That
terminal failure is preserved in place.  This launcher creates a fresh result
root, binds the original failure by hash, and reuses the exact Phase576 cases,
thresholds, model order, and stage implementations under the intended CUDA
runtime.

This file changes execution routing only.  It must be source-sealed before the
retry freeze and must not be edited after that freeze.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests/glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

PROTOCOL_WAS_PRELOADED = "phase576_gpt5_fruit_protocol" in sys.modules
import phase576_gpt5_fruit_protocol as protocol  # noqa: E402


EXPECTED_PYTHON = Path(
    r"C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe"
)
RETRY_OUT_DIR = ROOT / "tests/glm5/result/phase576r1_gpt5_fruit_structure"
ORIGINAL_OUT_DIR = ROOT / "tests/glm5/result/phase576_gpt5_fruit_structure"
WRAPPER_PATH = Path(__file__).resolve()
WRAPPER_SOURCE_KEY = "tests/glm5/phase576r1_gpt5_fruit_runtime.py"
STAGE_MODULE_NAMES = (
    "phase576_gpt5_fruit_engineering_qualification",
    "phase576_gpt5_fruit_behavior",
    "phase576_gpt5_fruit_behavior_analysis",
    "phase576_gpt5_fruit_natural_trace",
)

ORIGINAL_EVIDENCE = {
    "frozen_protocol": {
        "path": ORIGINAL_OUT_DIR / "phase576_frozen_protocol.json",
        "sha256": "82c478d261a172619191c940e8b5b11520cb02ae1e11b87f325a5b4e8c9f5c7e",
    },
    "freeze_commit": {
        "path": ORIGINAL_OUT_DIR / "phase576_freeze_commit.json",
        "sha256": "3c8911287630160c87c0d61cab9d8a5a8757f253e6e10591cdd44c3a30028dad",
    },
    "engineering_stage_start": {
        "path": ORIGINAL_OUT_DIR / "engineering_qualification_execution/stage_start.json",
        "sha256": "795a690da0721b9a5fde56f50d7f3e0a0ac5a363b189333bb4db0886433120ab",
    },
    "engineering_failure_receipt": {
        "path": ORIGINAL_OUT_DIR / "engineering_qualification_execution/execution_receipt.json",
        "sha256": "ac6d4c537fc283800547be31d7a8ee629f28ff065c4054044300ab203fbdc9ae",
    },
}

ORIGINAL_PROTOCOL_PAYLOAD: Callable[..., dict[str, Any]] = protocol.protocol_payload
ORIGINAL_STAGE_SOURCE_SEALS: Callable[[], dict[str, dict[str, Any]]] = (
    protocol.stage_source_seals
)
PRISTINE_PROTOCOL_FUNCTIONS = {
    name: getattr(protocol, name)
    for name in (
        "build_all",
        "freeze",
        "jsonl_bytes",
        "model_artifact_identity",
        "protocol_payload",
        "sha256_bytes",
        "stage_source_seals",
        "stable_hash",
        "verify",
    )
}
PRISTINE_PROTOCOL_OUT_DIR = protocol.OUT_DIR


def sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(f"file changed while being hashed: {path}")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def relative_path(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing immutable retry evidence: {path}")
    try:
        path_label = relative_path(path)
    except ValueError:
        path_label = str(path.resolve(strict=True))
    return {
        "path": path_label,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve(strict=True))) == os.path.normcase(
        str(right.resolve(strict=True))
    )


def distribution_tree_identity(distribution_name: str, import_name: str) -> dict[str, Any]:
    """Hash registered and package-tree bytes, excluding volatile bytecode caches."""

    distribution = importlib.metadata.distribution(distribution_name)
    anchor = Path(distribution.locate_file("")).resolve(strict=True)
    spec = importlib.util.find_spec(import_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot locate required runtime package: {import_name}")
    origin = Path(spec.origin).resolve(strict=True)
    package_root = origin.parent if origin.name == "__init__.py" else origin

    registered = distribution.files
    if registered is None:
        raise RuntimeError(f"runtime distribution has no file registry: {distribution_name}")

    def collect_paths() -> set[Path]:
        collected: set[Path] = set()
        for entry in registered:
            candidate = Path(distribution.locate_file(entry))
            if "__pycache__" not in candidate.parts and candidate.suffix != ".pyc":
                collected.add(candidate)
        if package_root.is_dir():
            for candidate in package_root.rglob("*"):
                if (
                    candidate.is_file()
                    and "__pycache__" not in candidate.parts
                    and candidate.suffix != ".pyc"
                ):
                    collected.add(candidate)
        else:
            collected.add(package_root)
        return collected

    paths = collect_paths()

    aggregate = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    missing_paths: list[str] = []
    symlink_count = 0
    for path in sorted(paths, key=lambda item: os.path.normcase(str(item))):
        try:
            label = str(path.relative_to(anchor)).replace("\\", "/")
        except ValueError:
            label = str(path.resolve(strict=False))
        if not path.is_file():
            missing_paths.append(label)
            row = {"path": label, "status": "missing"}
        else:
            is_symlink = path.is_symlink()
            symlink_count += int(is_symlink)
            size = path.stat().st_size
            digest = sha256_file(path)
            file_count += 1
            total_bytes += size
            row = {
                "path": label,
                "status": "regular_file",
                "path_is_symlink": is_symlink,
                "resolved_path": str(path.resolve(strict=True)),
                "size_bytes": size,
                "sha256": digest,
            }
        aggregate.update(json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"))
        aggregate.update(b"\n")
    if missing_paths:
        raise RuntimeError(
            f"registered runtime package files are missing for {distribution_name}: "
            f"{missing_paths[:10]}"
        )
    if paths != collect_paths():
        raise RuntimeError(
            f"runtime package inventory changed while hashing: {distribution_name}"
        )
    return {
        "distribution": distribution_name,
        "import_name": import_name,
        "version": distribution.version,
        "inventory_mode": (
            "distribution_registry_union_recursive_package_tree_excluding_pycache.v1"
        ),
        "anchor": str(anchor),
        "entry_origin": str(origin),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "symlink_count": symlink_count,
        "tree_sha256": aggregate.hexdigest(),
    }


@functools.lru_cache(maxsize=1)
def require_intended_cuda_runtime() -> dict[str, Any]:
    if not EXPECTED_PYTHON.is_file():
        raise RuntimeError(f"required Phase576R1 interpreter is missing: {EXPECTED_PYTHON}")
    if not _same_path(Path(sys.executable), EXPECTED_PYTHON):
        raise RuntimeError(
            "Phase576R1 must be launched by the sealed CUDA interpreter; "
            f"observed={sys.executable!r}, required={str(EXPECTED_PYTHON)!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Phase576R1 requires exactly one available CUDA device before any stage"
        )
    runtime = {
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "python_executable_identity": file_identity(Path(sys.executable).resolve(strict=True)),
        "python_version": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torch_cuda_runtime": torch.version.cuda,
        "transformers": package_version("transformers"),
        "bitsandbytes": package_version("bitsandbytes"),
        "cuda_available": True,
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "runtime_distribution_identities": {
            "torch": distribution_tree_identity("torch", "torch"),
            "transformers": distribution_tree_identity(
                "transformers", "transformers"
            ),
            "bitsandbytes": distribution_tree_identity(
                "bitsandbytes", "bitsandbytes"
            ),
            "tokenizers": distribution_tree_identity("tokenizers", "tokenizers"),
            "safetensors": distribution_tree_identity(
                "safetensors", "safetensors"
            ),
            "accelerate": distribution_tree_identity("accelerate", "accelerate"),
            "numpy": distribution_tree_identity("numpy", "numpy"),
            "huggingface_hub": distribution_tree_identity(
                "huggingface-hub", "huggingface_hub"
            ),
            "sentencepiece": distribution_tree_identity(
                "sentencepiece", "sentencepiece"
            ),
        },
    }
    if (
        runtime["python_version"] != "3.11.9"
        or runtime["torch"] != "2.11.0+cu128"
        or runtime["torch_cuda_runtime"] != "12.8"
        or runtime["transformers"] != "5.12.0"
        or runtime["bitsandbytes"] != "0.49.2"
        or runtime["gpu_names"] != ["NVIDIA GeForce RTX 5080"]
        or {
            name: identity["version"]
            for name, identity in runtime[
                "runtime_distribution_identities"
            ].items()
        }
        != {
            "torch": "2.11.0+cu128",
            "transformers": "5.12.0",
            "bitsandbytes": "0.49.2",
            "tokenizers": "0.22.2",
            "safetensors": "0.8.0",
            "accelerate": "1.14.0",
            "numpy": "2.4.4",
            "huggingface_hub": "1.19.0",
            "sentencepiece": "0.2.1",
        }
    ):
        raise RuntimeError(f"Phase576R1 sealed CUDA runtime identity drift: {runtime}")
    return runtime


def revalidate_intended_cuda_runtime() -> dict[str, Any]:
    expected = copy.deepcopy(require_intended_cuda_runtime())
    require_intended_cuda_runtime.cache_clear()
    observed = require_intended_cuda_runtime()
    if observed != expected:
        raise RuntimeError(
            "Phase576R1 runtime package/interpreter tree changed during execution"
        )
    return observed


def strict_root_file_identity(path: Path, root: Path) -> dict[str, Any]:
    logical_root = root.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"evidence must be a non-symlink regular file: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(logical_root)
        logical_relative = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"evidence path escapes its fixed root: {path}") from exc
    return {
        "path": relative_path(path),
        "root_relative_path": str(logical_relative).replace("\\", "/"),
        "resolved_path": str(resolved),
        "path_is_symlink": False,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verified_original_stage_source_seals(
    original_protocol: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    if original_protocol is None:
        protocol_path = ORIGINAL_EVIDENCE["frozen_protocol"]["path"]
        if sha256_file(protocol_path) != ORIGINAL_EVIDENCE["frozen_protocol"]["sha256"]:
            raise RuntimeError("original Phase576 protocol drift before source verification")
        original_protocol = read_json(protocol_path)
    frozen_seals = original_protocol.get("stage_source_seals")
    if not isinstance(frozen_seals, dict) or not frozen_seals:
        raise RuntimeError("original Phase576 source seal registry is malformed")
    current: dict[str, dict[str, Any]] = {}
    for key, expected in frozen_seals.items():
        if (
            not isinstance(key, str)
            or key.startswith("/")
            or "\\" in key
            or any(part in {"", ".", ".."} for part in key.split("/"))
            or not isinstance(expected, dict)
            or set(expected) != {"size_bytes", "sha256"}
        ):
            raise RuntimeError(f"invalid original Phase576 source seal entry: {key!r}")
        path = ROOT / Path(key)
        identity = strict_root_file_identity(path, ROOT)
        observed = {
            "size_bytes": identity["size_bytes"],
            "sha256": identity["sha256"],
        }
        if observed != expected:
            raise RuntimeError(f"original Phase576 source/dependency drift: {key}")
        current[key] = observed
    if current != frozen_seals:
        raise RuntimeError("original Phase576 exact source seal registry drift")
    return copy.deepcopy(frozen_seals)


def verify_original_freeze_closure(
    original_protocol: dict[str, Any],
    freeze_commit: dict[str, Any],
) -> dict[str, Any]:
    registry = freeze_commit.get("artifact_sha256_by_path")
    if (
        not isinstance(registry, dict)
        or freeze_commit.get("artifact_count") != len(registry)
        or freeze_commit.get("complete") is not True
        or freeze_commit.get("overwrite_allowed") is not False
        or freeze_commit.get("atomic_directory_publish") is not True
        or freeze_commit.get("phase_id") != "Phase576"
        or freeze_commit.get("schema_version") != "phase576_freeze_commit.v1"
    ):
        raise RuntimeError("original Phase576 freeze commit is not the audited commit")
    expected_registry = {
        "phase576_confirmation_cases.jsonl",
        "phase576_discovery_cases.jsonl",
        "phase576_frozen_protocol.json",
        "phase576_heldout_recombination_cases.jsonl",
        "phase576_open_cases.jsonl",
        "phase576_sealed_commitment.json",
        "phase576_static_audit.json",
        "protocol/private/phase576_sealed_cases.jsonl",
    }
    if set(registry) != expected_registry:
        raise RuntimeError("original Phase576 freeze artifact registry is not exact")
    identities: dict[str, dict[str, Any]] = {}
    for relative, expected_hash in sorted(registry.items()):
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(part in {"", ".", ".."} for part in relative.split("/"))
            or "\\" in relative
        ):
            raise RuntimeError(f"invalid original freeze registry entry: {relative!r}")
        identity = strict_root_file_identity(
            ORIGINAL_OUT_DIR / Path(relative), ORIGINAL_OUT_DIR
        )
        if identity["sha256"] != expected_hash:
            raise RuntimeError(f"original frozen artifact drift: {relative}")
        identities[relative] = identity
    if (
        registry["phase576_frozen_protocol.json"]
        != ORIGINAL_EVIDENCE["frozen_protocol"]["sha256"]
        or original_protocol.get("open_cases_sha256")
        != registry["phase576_open_cases.jsonl"]
        or original_protocol.get("open_case_sha256_by_split")
        != {
            "discovery": registry["phase576_discovery_cases.jsonl"],
            "confirmation": registry["phase576_confirmation_cases.jsonl"],
            "heldout_recombination": registry[
                "phase576_heldout_recombination_cases.jsonl"
            ],
        }
    ):
        raise RuntimeError("original Phase576 protocol/freeze denominator chain drift")
    commitment = read_json(ORIGINAL_OUT_DIR / "phase576_sealed_commitment.json")
    if (
        original_protocol.get("sealed_commitment_sha256")
        != registry["phase576_sealed_commitment.json"]
        or commitment.get("sealed_cases_sha256")
        != registry["protocol/private/phase576_sealed_cases.jsonl"]
    ):
        raise RuntimeError("original Phase576 sealed hash chain drift")
    return {
        "artifact_count": len(identities),
        "artifact_identities": identities,
        "sealed_case_payload_hashed_for_integrity": True,
        "sealed_case_payload_parsed_for_analysis": False,
    }


def verify_original_failure_evidence() -> dict[str, Any]:
    identities: dict[str, dict[str, Any]] = {}
    for name, expected in ORIGINAL_EVIDENCE.items():
        identity = strict_root_file_identity(expected["path"], ORIGINAL_OUT_DIR)
        if identity["sha256"] != expected["sha256"]:
            raise RuntimeError(f"original Phase576 {name} hash drift")
        identities[name] = identity

    original_protocol = read_json(ORIGINAL_EVIDENCE["frozen_protocol"]["path"])
    freeze_commit = read_json(ORIGINAL_EVIDENCE["freeze_commit"]["path"])
    stage_start = read_json(ORIGINAL_EVIDENCE["engineering_stage_start"]["path"])
    receipt = read_json(ORIGINAL_EVIDENCE["engineering_failure_receipt"]["path"])
    execution_dir = ORIGINAL_OUT_DIR / "engineering_qualification_execution"
    if execution_dir.is_symlink() or not execution_dir.is_dir():
        raise RuntimeError("original engineering failure directory is not canonical")
    execution_inventory = []
    for path in sorted(execution_dir.rglob("*")):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"unexpected non-regular original failure artifact: {path}"
            )
        execution_inventory.append(
            str(path.relative_to(execution_dir)).replace("\\", "/")
        )
    freeze_closure = verify_original_freeze_closure(
        original_protocol, freeze_commit
    )
    source_seals = verified_original_stage_source_seals(original_protocol)
    expected_not_attempted = list(protocol.MODELS)
    checks = {
        "original_phase": original_protocol.get("phase_id") == "Phase576",
        "original_protocol_hash_in_commit": freeze_commit.get(
            "artifact_sha256_by_path", {}
        ).get("phase576_frozen_protocol.json")
        == identities["frozen_protocol"]["sha256"],
        "same_run_id": receipt.get("run_id") == stage_start.get("run_id"),
        "stage_start_hash_bound": receipt.get("stage_start_sha256")
        == identities["engineering_stage_start"]["sha256"],
        "terminal_failure": receipt.get("terminal_status") == "failed"
        and receipt.get("execution_passed") is False,
        "zero_model_attempts": receipt.get("attempted_models_in_order") == []
        and receipt.get("completed_models") == []
        and receipt.get("failed_models") == [],
        "all_models_not_attempted": receipt.get("not_attempted_models")
        == expected_not_attempted,
        "cuda_entry_failure": receipt.get("fatal_error")
        == {
            "error": "CUDA is required for engineering qualification",
            "error_type": "RuntimeError",
        },
        "cpu_only_runtime": receipt.get("runtime_identity")
        == {
            "bitsandbytes": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_runtime": None,
            "gpu_names": [],
            "python": "3.14.6",
            "torch": "2.13.0+cpu",
            "transformers": "5.14.1",
        },
        "no_research_case_parsed": receipt.get("formal_case_content_parsed") is False,
        "no_sealed_split_read": receipt.get("sealed_split_read") is False,
        "qualification_not_published": receipt.get("qualification_publish_authorized")
        is False
        and not (ORIGINAL_OUT_DIR / "phase576_engineering_qualification.json").exists(),
        "freeze_artifact_closure": freeze_closure["artifact_count"] == 8,
        "source_seal_closure": source_seals
        == original_protocol.get("stage_source_seals"),
        "engineering_failure_directory_exact_inventory": execution_inventory
        == ["execution_receipt.json", "stage_start.json"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"original Phase576 failure evidence is not the audited event: {checks}")
    return {
        "artifact_identities": identities,
        "checks": checks,
        "run_id": receipt["run_id"],
        "started_at_utc": stage_start["started_at_utc"],
        "failed_at_utc": receipt["created_at_utc"],
        "fatal_error": receipt["fatal_error"],
        "runtime_identity": receipt["runtime_identity"],
        "attempted_models_in_order": receipt["attempted_models_in_order"],
        "completed_models": receipt["completed_models"],
        "not_attempted_models": receipt["not_attempted_models"],
        "freeze_closure": freeze_closure,
        "stage_source_seals": source_seals,
        "engineering_failure_directory_inventory": execution_inventory,
    }


def runtime_erratum_payload() -> dict[str, Any]:
    failure = verify_original_failure_evidence()
    intended_runtime = require_intended_cuda_runtime()
    return {
        "schema_version": "phase576r1_runtime_erratum.v1",
        "retry_id": "Phase576R1",
        "reason": (
            "the frozen Phase576 engineering entry was accidentally invoked with "
            "a CPU-only repository interpreter; failure occurred before any model attempt"
        ),
        "correction_scope": "interpreter_and_fresh_result_root_only",
        "original_failure": failure,
        "intended_runtime": intended_runtime,
        "retry_result_root": relative_path(RETRY_OUT_DIR),
        "scientific_denominator_changed": False,
        "case_prompts_targets_splits_changed": False,
        "behavior_thresholds_changed": False,
        "trace_scope_changed": False,
        "model_order_changed": False,
        "model_artifacts_changed": False,
        "original_result_overwritten": False,
        "original_failed_receipt_preserved": True,
        "sealed_case_payload_hashed_for_integrity": True,
        "sealed_case_payload_read_for_integrity": True,
        "sealed_case_payload_parsed_for_erratum": False,
    }


def retry_stage_source_seals() -> dict[str, dict[str, Any]]:
    revalidate_intended_cuda_runtime()
    original_protocol_path = ORIGINAL_EVIDENCE["frozen_protocol"]["path"]
    if sha256_file(original_protocol_path) != ORIGINAL_EVIDENCE[
        "frozen_protocol"
    ]["sha256"]:
        raise RuntimeError("original Phase576 protocol drift before retry source seal")
    seals = verified_original_stage_source_seals(read_json(original_protocol_path))
    if ORIGINAL_STAGE_SOURCE_SEALS() != seals:
        raise RuntimeError(
            "current Phase576 stage source registry is not the original frozen registry"
        )
    wrapper_identity = file_identity(WRAPPER_PATH)
    seals[WRAPPER_SOURCE_KEY] = {
        "size_bytes": wrapper_identity["size_bytes"],
        "sha256": wrapper_identity["sha256"],
    }
    return seals


def retry_protocol_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = ORIGINAL_PROTOCOL_PAYLOAD(*args, **kwargs)
    payload["runtime_erratum"] = runtime_erratum_payload()
    open_rows = args[0] if args else kwargs["open_rows"]
    sealed_rows = args[1] if len(args) > 1 else kwargs["sealed_rows"]
    audit = args[2] if len(args) > 2 else kwargs["audit"]
    validate_unpublished_retry_protocol(payload, open_rows, sealed_rows, audit)
    return payload


def configure_retry_namespace() -> None:
    if PROTOCOL_WAS_PRELOADED:
        raise RuntimeError(
            "Phase576R1 refuses a process that preloaded the protocol module"
        )
    if (
        protocol.OUT_DIR != PRISTINE_PROTOCOL_OUT_DIR
        or any(
            getattr(protocol, name, None) is not expected
            for name, expected in PRISTINE_PROTOCOL_FUNCTIONS.items()
        )
    ):
        raise RuntimeError("Phase576 protocol namespace was modified before R1 rebinding")
    prematurely_loaded = [
        name for name in STAGE_MODULE_NAMES if name in sys.modules
    ]
    if prematurely_loaded:
        raise RuntimeError(
            "Phase576R1 refuses a process that imported stage modules before path "
            f"rebinding: {prematurely_loaded}"
        )
    protocol.OUT_DIR = RETRY_OUT_DIR
    protocol.OPEN_CASES_PATH = RETRY_OUT_DIR / "phase576_open_cases.jsonl"
    protocol.OPEN_SPLIT_CASE_PATHS = {
        split: RETRY_OUT_DIR / f"phase576_{split}_cases.jsonl"
        for split in protocol.OPEN_SPLITS
    }
    protocol.SEALED_CASES_PATH = (
        RETRY_OUT_DIR / "protocol/private/phase576_sealed_cases.jsonl"
    )
    protocol.SEALED_COMMITMENT_PATH = RETRY_OUT_DIR / "phase576_sealed_commitment.json"
    protocol.PROTOCOL_PATH = RETRY_OUT_DIR / "phase576_frozen_protocol.json"
    protocol.STATIC_AUDIT_PATH = RETRY_OUT_DIR / "phase576_static_audit.json"
    protocol.ENGINEERING_QUALIFICATION_PATH = (
        RETRY_OUT_DIR / "phase576_engineering_qualification.json"
    )
    protocol.BEHAVIOR_DECISION_PATHS = {
        split: RETRY_OUT_DIR / f"phase576_{split}_behavior_decision.json"
        for split in protocol.OPEN_SPLITS
    }
    protocol.BEHAVIOR_DECISION_PATH = protocol.BEHAVIOR_DECISION_PATHS["discovery"]
    protocol.DISCOVERY_REGISTRY_PATH = (
        RETRY_OUT_DIR / "phase576_discovered_structure_registry.json"
    )
    protocol.CONFIRMATION_DECISION_PATH = (
        RETRY_OUT_DIR / "phase576_structure_confirmation_decision.json"
    )
    protocol.HELDOUT_DECISION_PATH = (
        RETRY_OUT_DIR / "phase576_heldout_replication_decision.json"
    )
    protocol.SEALED_OPEN_RECEIPT_PATH = (
        RETRY_OUT_DIR / "phase576_sealed_open_receipt.json"
    )
    protocol.FREEZE_COMMIT_PATH = RETRY_OUT_DIR / "phase576_freeze_commit.json"
    protocol.FREEZE_LOCK_PATH = (
        RETRY_OUT_DIR.parent / f".{RETRY_OUT_DIR.name}.freeze.lock"
    )
    protocol.stage_source_seals = retry_stage_source_seals
    protocol.protocol_payload = retry_protocol_payload


SCIENTIFIC_EQUIVALENCE_KEYS = (
    "research_route",
    "evidence_order",
    "models_in_required_execution_order",
    "model_artifact_identities",
    "prior_open_file_identities",
    "splits",
    "open_splits",
    "open_case_count",
    "sealed_case_count",
    "cases_per_split",
    "relations",
    "relation_contracts",
    "interfaces",
    "behavior_repeats",
    "behavior_batch_size",
    "trace_batch_size",
    "max_new_tokens",
    "behavior_gate",
    "case_grid_contract",
    "cross_model_observational_comparison_policy",
    "trace_policy",
    "sealed_policy",
    "scientific_limits",
    "open_cases_sha256",
    "open_case_sha256_by_split",
)
PREFREEZE_EQUIVALENCE_KEYS = tuple(
    key for key in SCIENTIFIC_EQUIVALENCE_KEYS
    if key not in {"open_cases_sha256", "open_case_sha256_by_split"}
) + ("atomic_freeze_policy",)


def staged_analysis_policy_equivalent(
    original: dict[str, Any], retry: dict[str, Any]
) -> bool:
    original_policy = copy.deepcopy(original.get("staged_analysis_seal_policy"))
    retry_policy = copy.deepcopy(retry.get("staged_analysis_seal_policy"))
    if not isinstance(original_policy, dict) or not isinstance(retry_policy, dict):
        return False
    original_initial = original_policy.pop("initial_stage_sources", None)
    retry_initial = retry_policy.pop("initial_stage_sources", None)
    expected_retry_seals = copy.deepcopy(original.get("stage_source_seals"))
    if not isinstance(expected_retry_seals, dict):
        return False
    wrapper = file_identity(WRAPPER_PATH)
    expected_retry_seals[WRAPPER_SOURCE_KEY] = {
        "size_bytes": wrapper["size_bytes"],
        "sha256": wrapper["sha256"],
    }
    return all((
        original_initial == original.get("stage_source_seals"),
        retry_initial == retry.get("stage_source_seals"),
        retry_initial == expected_retry_seals,
        original_policy == retry_policy,
    ))


def validate_unpublished_retry_protocol(
    candidate: dict[str, Any],
    open_rows: list[dict[str, Any]],
    sealed_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    """Fail inside protocol serialization, before staging or directory rename."""

    failure = verify_original_failure_evidence()
    original = read_json(ORIGINAL_EVIDENCE["frozen_protocol"]["path"])
    original_registry = read_json(
        ORIGINAL_EVIDENCE["freeze_commit"]["path"]
    )["artifact_sha256_by_path"]
    original_static_audit = read_json(
        ORIGINAL_OUT_DIR / "phase576_static_audit.json"
    )
    expected_raw_audit = copy.deepcopy(original_static_audit)
    for key in (
        "schema_version",
        "phase_id",
        "created_at_utc",
        "open_cases_sha256",
        "open_case_sha256_by_split",
        "sealed_cases_sha256",
        "sealed_commitment_sha256",
        "protocol_sha256",
    ):
        expected_raw_audit.pop(key, None)
    expected_seals = copy.deepcopy(original["stage_source_seals"])
    wrapper = file_identity(WRAPPER_PATH)
    expected_seals[WRAPPER_SOURCE_KEY] = {
        "size_bytes": wrapper["size_bytes"],
        "sha256": wrapper["sha256"],
    }
    open_blob = protocol.jsonl_bytes(open_rows)
    split_hashes = {
        split: protocol.sha256_bytes(protocol.jsonl_bytes([
            row for row in open_rows if row["split"] == split
        ]))
        for split in protocol.OPEN_SPLITS
    }
    original_commitment = read_json(
        ORIGINAL_OUT_DIR / "phase576_sealed_commitment.json"
    )
    expected_commitment_without_time = copy.deepcopy(original_commitment)
    expected_commitment_without_time.pop("created_at_utc", None)
    regenerated_commitment_without_time = {
        "schema_version": "phase576_sealed_commitment.v2",
        "phase_id": protocol.PHASE,
        "sealed_case_count": len(sealed_rows),
        "sealed_cases_sha256": protocol.sha256_bytes(
            protocol.jsonl_bytes(sealed_rows)
        ),
        "holdout_is_blind": False,
        "sealed_definition_is_public_in_source": True,
        "sealed_definition_cpu_read_during_freeze": True,
        "sealed_model_opened": False,
        "sealed_model_access_count": 0,
        "sealed_result_analysis_access_count": 0,
        "prior_sealed_files_read": False,
    }
    checks = {
        "phase_id": candidate.get("phase_id")
        == original.get("phase_id") == "Phase576",
        "schema_version": candidate.get("schema_version")
        == original.get("schema_version") == protocol.SCHEMA_VERSION,
        "source_script": candidate.get("source_script")
        == original.get("source_script"),
        "source_script_sha256": candidate.get("source_script_sha256")
        == original.get("source_script_sha256"),
        "source_seals_original_plus_wrapper": candidate.get(
            "stage_source_seals"
        ) == expected_seals == retry_stage_source_seals(),
        "model_artifacts_identical": candidate.get("model_artifact_identities")
        == original.get("model_artifact_identities"),
        "scientific_and_atomic_fields_identical": all(
            candidate.get(key) == original.get(key)
            for key in PREFREEZE_EQUIVALENCE_KEYS
        ),
        "staged_policy_normalized_identical": staged_analysis_policy_equivalent(
            original, candidate
        ),
        "runtime_erratum_exact": candidate.get("runtime_erratum")
        == runtime_erratum_payload(),
        "raw_static_audit_exact": audit == expected_raw_audit,
        "sealed_commitment_normalized_exact": (
            regenerated_commitment_without_time
            == expected_commitment_without_time
        ),
        "open_bytes_identical": protocol.sha256_bytes(open_blob)
        == original_registry["phase576_open_cases.jsonl"],
        "split_bytes_identical": split_hashes
        == {
            "discovery": original_registry["phase576_discovery_cases.jsonl"],
            "confirmation": original_registry[
                "phase576_confirmation_cases.jsonl"
            ],
            "heldout_recombination": original_registry[
                "phase576_heldout_recombination_cases.jsonl"
            ],
        },
        "sealed_bytes_identical": protocol.sha256_bytes(
            protocol.jsonl_bytes(sealed_rows)
        ) == original_registry["protocol/private/phase576_sealed_cases.jsonl"],
        "original_closure_verified": failure["freeze_closure"][
            "artifact_count"
        ] == 8,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Phase576R1 unpublished protocol equivalence failed: {checks}"
        )


def preflight_retry_freeze() -> dict[str, Any]:
    """Prove retry equivalence before the atomic, non-overwritable publish."""

    if RETRY_OUT_DIR.exists() or protocol.FREEZE_LOCK_PATH.exists():
        raise RuntimeError("Phase576R1 preflight requires an absent result root and lock")
    failure = verify_original_failure_evidence()
    original = read_json(ORIGINAL_EVIDENCE["frozen_protocol"]["path"])
    source_seals = retry_stage_source_seals()
    expected_source_seals = copy.deepcopy(original["stage_source_seals"])
    wrapper = file_identity(WRAPPER_PATH)
    expected_source_seals[WRAPPER_SOURCE_KEY] = {
        "size_bytes": wrapper["size_bytes"],
        "sha256": wrapper["sha256"],
    }
    if source_seals != expected_source_seals:
        raise RuntimeError("Phase576R1 source seals are not original-plus-wrapper exact")

    open_rows, sealed_rows, audit = protocol.build_all()
    if audit.get("valid") is not True or audit.get("failures") != []:
        raise RuntimeError("Phase576R1 regenerated static denominator audit failed")
    open_blob = protocol.jsonl_bytes(open_rows)
    split_blobs = {
        split: protocol.jsonl_bytes([
            row for row in open_rows if row["split"] == split
        ])
        for split in protocol.OPEN_SPLITS
    }
    sealed_blob = protocol.jsonl_bytes(sealed_rows)
    original_registry = read_json(
        ORIGINAL_EVIDENCE["freeze_commit"]["path"]
    )["artifact_sha256_by_path"]
    byte_checks = {
        "open": protocol.sha256_bytes(open_blob)
        == original_registry["phase576_open_cases.jsonl"],
        "discovery": protocol.sha256_bytes(split_blobs["discovery"])
        == original_registry["phase576_discovery_cases.jsonl"],
        "confirmation": protocol.sha256_bytes(split_blobs["confirmation"])
        == original_registry["phase576_confirmation_cases.jsonl"],
        "heldout_recombination": protocol.sha256_bytes(
            split_blobs["heldout_recombination"]
        ) == original_registry["phase576_heldout_recombination_cases.jsonl"],
        "sealed": protocol.sha256_bytes(sealed_blob)
        == original_registry["protocol/private/phase576_sealed_cases.jsonl"],
    }
    if not all(byte_checks.values()):
        raise RuntimeError(f"Phase576R1 regenerated denominator drift: {byte_checks}")

    model_artifacts = protocol.model_artifact_identity()
    if model_artifacts != original.get("model_artifact_identities"):
        raise RuntimeError("Phase576R1 model artifact identity changed before publish")
    candidate = retry_protocol_payload(
        open_rows,
        sealed_rows,
        audit,
        "preflight-only-not-published",
        model_artifacts,
        source_seals,
        sha256_file(Path(protocol.__file__).resolve()),
    )
    checks = {
        "phase_id_identical": candidate.get("phase_id")
        == original.get("phase_id") == "Phase576",
        "schema_version_identical": candidate.get("schema_version")
        == original.get("schema_version") == protocol.SCHEMA_VERSION,
        "source_script_identical": candidate.get("source_script")
        == original.get("source_script"),
        "source_script_sha256_identical": candidate.get("source_script_sha256")
        == original.get("source_script_sha256"),
        "source_registry_original_plus_wrapper": candidate.get(
            "stage_source_seals"
        ) == expected_source_seals,
        "scientific_and_atomic_fields_identical": all(
            candidate.get(key) == original.get(key)
            for key in PREFREEZE_EQUIVALENCE_KEYS
        ),
        "staged_analysis_policy_normalized_identical": (
            staged_analysis_policy_equivalent(original, candidate)
        ),
        "runtime_erratum_exact": candidate.get("runtime_erratum")
        == runtime_erratum_payload(),
        "full_original_freeze_closure_verified": failure["freeze_closure"][
            "artifact_count"
        ] == 8,
        "regenerated_case_bytes_identical": all(byte_checks.values()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase576R1 prepublish equivalence failed: {checks}")
    return {
        "checks": checks,
        "case_byte_checks": byte_checks,
        "model_artifact_identities_sha256": protocol.stable_hash(model_artifacts),
        "source_seals_sha256": protocol.stable_hash(source_seals),
    }


def verify_retry_equivalence() -> dict[str, Any]:
    if not protocol.PROTOCOL_PATH.is_file():
        raise RuntimeError("Phase576R1 frozen protocol is missing")
    original = read_json(ORIGINAL_EVIDENCE["frozen_protocol"]["path"])
    retry = read_json(protocol.PROTOCOL_PATH)
    failure = verify_original_failure_evidence()
    expected_erratum = runtime_erratum_payload()
    expected_source_seals = copy.deepcopy(original["stage_source_seals"])
    wrapper = file_identity(WRAPPER_PATH)
    expected_source_seals[WRAPPER_SOURCE_KEY] = {
        "size_bytes": wrapper["size_bytes"],
        "sha256": wrapper["sha256"],
    }
    retry_commitment = read_json(protocol.SEALED_COMMITMENT_PATH)
    original_commitment = read_json(
        ORIGINAL_OUT_DIR / "phase576_sealed_commitment.json"
    )
    normalized_retry_commitment = copy.deepcopy(retry_commitment)
    normalized_original_commitment = copy.deepcopy(original_commitment)
    normalized_retry_commitment.pop("created_at_utc", None)
    normalized_original_commitment.pop("created_at_utc", None)
    retry_static_audit = read_json(protocol.STATIC_AUDIT_PATH)
    original_static_audit = read_json(
        ORIGINAL_OUT_DIR / "phase576_static_audit.json"
    )
    normalized_retry_audit = copy.deepcopy(retry_static_audit)
    normalized_original_audit = copy.deepcopy(original_static_audit)
    for normalized in (normalized_retry_audit, normalized_original_audit):
        for key in (
            "created_at_utc",
            "protocol_sha256",
            "sealed_commitment_sha256",
        ):
            normalized.pop(key, None)
    retry_sealed_hash = sha256_file(protocol.SEALED_CASES_PATH)
    original_sealed_hash = failure["freeze_closure"]["artifact_identities"][
        "protocol/private/phase576_sealed_cases.jsonl"
    ]["sha256"]
    checks = {
        "phase_id_identical": retry.get("phase_id")
        == original.get("phase_id") == "Phase576",
        "schema_version_identical": retry.get("schema_version")
        == original.get("schema_version") == protocol.SCHEMA_VERSION,
        "runtime_erratum_exact": retry.get("runtime_erratum") == expected_erratum,
        "source_registry_original_plus_wrapper": retry.get(
            "stage_source_seals"
        ) == expected_source_seals == retry_stage_source_seals(),
        "all_scientific_fields_identical": all(
            retry.get(key) == original.get(key)
            for key in SCIENTIFIC_EQUIVALENCE_KEYS
        ),
        "atomic_freeze_policy_identical": retry.get("atomic_freeze_policy")
        == original.get("atomic_freeze_policy"),
        "staged_analysis_policy_normalized_identical": (
            staged_analysis_policy_equivalent(original, retry)
        ),
        "static_audit_normalized_identical": normalized_retry_audit
        == normalized_original_audit,
        "sealed_commitment_normalized_identical": normalized_retry_commitment
        == normalized_original_commitment,
        "case_bytes_identical": sha256_file(protocol.OPEN_CASES_PATH)
        == original["open_cases_sha256"],
        "split_case_bytes_identical": {
            split: sha256_file(path)
            for split, path in protocol.OPEN_SPLIT_CASE_PATHS.items()
        }
        == original["open_case_sha256_by_split"],
        "sealed_case_payload_hash_identical": retry_sealed_hash
        == retry_commitment.get("sealed_cases_sha256")
        == original_sealed_hash,
        "full_original_failure_and_freeze_closure_immutable": failure[
            "freeze_closure"
        ]["artifact_count"] == 8,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase576R1 scientific/runtime equivalence failed: {checks}")
    return checks


def _require_retry_path(path: Path, label: str) -> None:
    retry_root = RETRY_OUT_DIR.resolve(strict=False)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(retry_root)
    except ValueError as exc:
        raise RuntimeError(f"loaded stage path escaped Phase576R1 root: {label}={path}") from exc


def assert_loaded_stage_modules_bound_to_retry() -> None:
    engineering = sys.modules.get(
        "phase576_gpt5_fruit_engineering_qualification"
    )
    if engineering is not None:
        expected = {
            "QUALIFICATION_PATH": protocol.ENGINEERING_QUALIFICATION_PATH,
            "EXECUTION_DIR": RETRY_OUT_DIR / "engineering_qualification_execution",
            "STAGE_START_PATH": RETRY_OUT_DIR
            / "engineering_qualification_execution/stage_start.json",
            "EXECUTION_RECEIPT_PATH": RETRY_OUT_DIR
            / "engineering_qualification_execution/execution_receipt.json",
        }
        for name, value in expected.items():
            observed = getattr(engineering, name, None)
            if not isinstance(observed, Path) or observed != value:
                raise RuntimeError(
                    f"engineering import-time path was not rebound: {name}={observed}"
                )
            _require_retry_path(observed, f"engineering.{name}")
        _require_retry_path(
            engineering.EXECUTION_LEASE_PATH, "engineering.EXECUTION_LEASE_PATH"
        )
    behavior = sys.modules.get("phase576_gpt5_fruit_behavior")
    if behavior is not None:
        for stage in protocol.OPEN_SPLITS:
            _require_retry_path(behavior.stage_dir(stage), f"behavior.{stage}")
    analysis = sys.modules.get("phase576_gpt5_fruit_behavior_analysis")
    if analysis is not None:
        for stage in protocol.OPEN_SPLITS:
            _require_retry_path(analysis.stage_dir(stage), f"analysis.{stage}")
    natural_trace = sys.modules.get("phase576_gpt5_fruit_natural_trace")
    if natural_trace is not None:
        for stage in protocol.OPEN_SPLITS:
            _require_retry_path(
                natural_trace.trace_stage_dir(stage), f"trace.{stage}"
            )


def call_behavior_analysis(stage: str) -> None:
    import phase576_gpt5_fruit_behavior_analysis as analysis

    assert_loaded_stage_modules_bound_to_retry()
    previous = sys.argv
    try:
        sys.argv = [str(Path(analysis.__file__).resolve()), "--stage", stage]
        analysis.main()
    finally:
        sys.argv = previous


def main() -> None:
    configure_retry_namespace()
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    group.add_argument("--engineering", action="store_true")
    group.add_argument("--locator-self-test", action="store_true")
    group.add_argument("--behavior", choices=protocol.OPEN_SPLITS)
    group.add_argument("--behavior-analysis", choices=protocol.OPEN_SPLITS)
    group.add_argument("--trace", choices=protocol.OPEN_SPLITS)
    args = parser.parse_args()

    runtime = require_intended_cuda_runtime()
    verify_original_failure_evidence()
    if args.self_test:
        result = protocol.self_test()
        result["runtime_erratum"] = runtime_erratum_payload()
    elif args.write:
        preflight = preflight_retry_freeze()
        result = protocol.freeze()
        result["prepublish_equivalence"] = preflight
        result["runtime_equivalence_checks"] = verify_retry_equivalence()
    else:
        equivalence = verify_retry_equivalence()
        post_equivalence: dict[str, Any]
        try:
            if args.verify:
                result = protocol.verify()
            elif args.engineering:
                import phase576_gpt5_fruit_engineering_qualification as engineering

                assert_loaded_stage_modules_bound_to_retry()
                engineering.main()
                result = {"passed": True, "stage": "engineering_qualification"}
            elif args.locator_self_test:
                import phase576_gpt5_fruit_natural_trace as natural_trace

                assert_loaded_stage_modules_bound_to_retry()
                result = natural_trace.locator_self_test()
            elif args.behavior is not None:
                import phase576_gpt5_fruit_behavior as behavior

                assert_loaded_stage_modules_bound_to_retry()
                result = behavior.run_stage(args.behavior)
            elif args.behavior_analysis is not None:
                call_behavior_analysis(args.behavior_analysis)
                result = {"passed": True, "stage": args.behavior_analysis}
            else:
                import phase576_gpt5_fruit_natural_trace as natural_trace

                assert_loaded_stage_modules_bound_to_retry()
                result = natural_trace.run_stage(args.trace)
        finally:
            # Recheck both result roots even when a model stage fails.  This
            # closes the long-running-stage mutation window without hiding the
            # stage's own terminal receipt.
            verify_original_failure_evidence()
            post_equivalence = verify_retry_equivalence()
        result["runtime_equivalence_checks_before_stage"] = equivalence
        result["runtime_equivalence_checks"] = post_equivalence
    result["phase576r1_runtime_identity"] = runtime
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
