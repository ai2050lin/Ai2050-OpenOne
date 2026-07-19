#!/usr/bin/env python3
"""Phase 979 CUDA runner for the sealed 4x2 natural-boundary factorial.

Every item/control/decoding/replicate trajectory owns an independent CUDA
``torch.Generator``.  Batching therefore changes compute shape, not RNG state.
Each trajectory is sampled once to at most 2048 tokens; all five checkpoint
snapshots are deterministic prefixes of that one trajectory.

This script never imports or parses the Phase 977 holdout module.  It may run
only after a Phase 979 protocol has sealed this exact file, the shared boundary
core, the diagnostic dataset, the local model/runtime, and the Phase 978
development NO-GO lineage.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

import phase979_boundary_core as core  # noqa: E402
import phase979_diagnostic_dataset as diagnostic_dataset  # noqa: E402
from model_utils import MODEL_CONFIGS, load_model, release_model  # noqa: E402


PHASE = 979
SCHEMA_VERSION = 1
EXPERIMENT = "three_boundary_factorial_and_truth_punctuation"
NATURAL_EXPERIMENT = "three_boundary_natural_factorial"
MODEL_NAME = "qwen3"
EXPECTED_ITEMS = 128
EXPECTED_ROWS = 2048

OUT = ROOT / "tests" / "glm5" / "result" / "phase979_three_boundary_factorial"
PROTOCOL_PATH = OUT / "protocol_preregistration.json"
MANIFEST_PATH = OUT / "manifest_natural.json"
ROWS_PATH = OUT / "rows_natural.jsonl"
STATUS_PATH = OUT / "generator_status_natural.json"
RUN_LOCK_PATH = OUT / "natural_runner.lock"

SCRIPT_PATH = Path(__file__).resolve()
CORE_PATH = GLM5 / "phase979_boundary_core.py"
DATASET_PATH = GLM5 / "phase979_diagnostic_dataset.py"
PHASE978_DIR = GLM5 / "result" / "phase978_legal_budget_stabilization"
PHASE978_ADMISSION_PATH = PHASE978_DIR / "admission_development.json"
PHASE978_OPEN_RECEIPT_PATH = PHASE978_DIR / "holdout_open_receipt.json"
FORBIDDEN_HOLDOUT_MODULE = "phase977_holdout_dataset"

ROW_HASH_FIELDS = {"row_sha256"}
MANIFEST_HASH_FIELDS = {"manifest_sha256", "created_at_utc"}
STATUS_HASH_FIELDS = {"status_sha256", "updated_at_utc"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def assert_holdout_not_imported() -> None:
    loaded = [name for name in sys.modules if
              name == FORBIDDEN_HOLDOUT_MODULE or
              name.endswith("." + FORBIDDEN_HOLDOUT_MODULE)]
    require(not loaded, f"forbidden holdout module imported: {loaded}")


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_full": sys.version,
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "version_source": "installed_distribution_metadata_only",
    }


def load_json(path: Path, label: str) -> dict[str, Any]:
    return core.load_json(path, label)


def verify_self_hash(
    document: dict[str, Any], hash_field: str, excluded: set[str], label: str,
) -> None:
    claimed = document.get(hash_field)
    require(isinstance(claimed, str) and len(claimed) == 64,
            f"{label} lacks {hash_field}")
    payload = {key: value for key, value in document.items()
               if key not in excluded}
    require(claimed == core.sha256_json(payload), f"{label} self-hash invalid")


def normalized_official_cells(value: Any) -> set[tuple[str, str]]:
    require(isinstance(value, (list, tuple, set)),
            "protocol official_cells must be a sequence")
    output: set[tuple[str, str]] = set()
    for item in value:
        require(isinstance(item, (list, tuple)) and len(item) == 2,
                f"invalid official cell: {item!r}")
        output.add((str(item[0]), str(item[1])))
    return output


def iter_path_commitments(value: Any) -> Iterable[dict[str, Any]]:
    """Yield nested objects that are explicit file path/SHA commitments."""
    if isinstance(value, dict):
        if (isinstance(value.get("path"), str)
                and isinstance(value.get("sha256"), str)):
            yield value
        for child in value.values():
            yield from iter_path_commitments(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_path_commitments(child)


def verify_file_commitment(entry: dict[str, Any], label: str) -> Path:
    path = ROOT / str(entry["path"])
    require(path.is_file(), f"missing {label}: {entry['path']}")
    actual = core.sha256_file(path)
    require(actual == entry["sha256"],
            f"{label} changed after seal: {entry['path']}: {actual}")
    return path.resolve()


def authenticate_phase978_no_go(commitments: dict[str, Any]) -> dict[str, Any]:
    """Authenticate the public Phase 978 admission without touching holdout data."""
    assert_holdout_not_imported()
    entries = list(iter_path_commitments(commitments))
    require(entries, "protocol lacks explicit Phase978 file commitments")
    admission_committed = False
    for entry in entries:
        path_text = str(entry["path"]).replace("\\", "/")
        # The old opaque holdout source is deliberately not a Phase979 runtime
        # dependency.  The Phase979 lineage must commit to public results instead.
        require(not path_text.endswith("phase977_holdout_dataset.py"),
                "Phase979 commitments must not expose the old holdout module")
        path = verify_file_commitment(entry, "Phase978 lineage artifact")
        if path == PHASE978_ADMISSION_PATH.resolve():
            admission_committed = True
    require(admission_committed,
            "Phase979 protocol does not commit to Phase978 admission_development.json")

    admission = load_json(PHASE978_ADMISSION_PATH, "Phase978 development admission")
    verify_self_hash(
        admission, "admission_sha256", {"admission_sha256", "audited_at_utc"},
        "Phase978 development admission",
    )
    require(admission.get("phase") == 978, "Phase978 admission phase mismatch")
    require(admission.get("decision_gate", {}).get("passed") is False,
            "Phase978 admission is not the frozen NO-GO")
    require(admission.get("holdout_authorized") is False,
            "Phase978 admission unexpectedly authorizes holdout")
    require(admission.get("holdout_loaded") is False,
            "Phase978 admission reports holdout access")
    require(not PHASE978_OPEN_RECEIPT_PATH.exists(),
            "Phase978 OPEN receipt exists; Phase979 must stop")
    assert_holdout_not_imported()
    return {
        "admission_sha256": admission["admission_sha256"],
        "admission_file_sha256": core.sha256_file(PHASE978_ADMISSION_PATH),
        "decision": "NO-GO",
        "holdout_authorized": False,
        "holdout_loaded": False,
        "open_receipt_exists": False,
    }


def authenticate_model_identity(protocol: dict[str, Any]) -> None:
    expected_runtime = protocol.get("runtime_versions")
    if expected_runtime is not None:
        require(expected_runtime == runtime_versions(),
                "runtime differs from Phase979 protocol")

    identity = protocol.get("local_model_artifact_identity")
    if identity is None:
        identity = protocol.get("model_identity")
    if not isinstance(identity, dict):
        return
    configured = Path(MODEL_CONFIGS[MODEL_NAME]["path"]).resolve()
    if isinstance(identity.get("path"), str):
        expected_root = (ROOT / str(identity["path"])).resolve()
        require(expected_root == configured,
                "model registry path differs from Phase979 protocol")
    files = identity.get("files")
    if isinstance(files, dict) and files:
        for name, expected in files.items():
            require(isinstance(expected, dict), f"invalid model file entry: {name}")
            path = configured / str(name)
            require(path.is_file(), f"missing frozen model artifact: {path}")
            if "bytes" in expected:
                require(path.stat().st_size == int(expected["bytes"]),
                        f"model artifact size changed: {name}")
            if "sha256" in expected:
                require(core.sha256_file(path) == str(expected["sha256"]),
                        f"model artifact hash changed: {name}")


def authenticate_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    assert_holdout_not_imported()
    protocol = load_json(PROTOCOL_PATH, "Phase979 protocol preregistration")
    verify_self_hash(
        protocol, "protocol_sha256", {"protocol_sha256", "created_at_utc"},
        "Phase979 protocol preregistration",
    )
    require(protocol.get("phase") == PHASE, "Phase979 protocol phase mismatch")
    require(protocol.get("experiment") == EXPERIMENT,
            "Phase979 protocol experiment mismatch")
    require(protocol.get("controls") == core.CONTROL_POLICIES,
            "Phase979 control registry mismatch")
    require(protocol.get("decoding_policies") == core.DECODING_POLICIES,
            "Phase979 decoding registry mismatch")
    require(normalized_official_cells(protocol.get("official_cells")) ==
            set(core.OFFICIAL_CELLS), "official cell registry mismatch")
    require(protocol.get("checkpoints") == list(core.CHECKPOINTS),
            "Phase979 checkpoint registry mismatch")
    require(protocol.get("decision_checkpoint") == core.MAX_NEW_TOKENS == 2048,
            "Phase979 decision checkpoint mismatch")
    require(protocol.get("max_new_tokens") == core.MAX_NEW_TOKENS,
            "Phase979 maximum budget mismatch")
    require(protocol.get("batch_size") == core.BATCH_SIZE == 8,
            "Phase979 batch size mismatch")
    require(protocol.get("replicates") == list(core.REPLICATES) == [0, 1],
            "Phase979 replicate registry mismatch")
    require(protocol.get("expected_natural_rows") == EXPECTED_ROWS,
            "Phase979 natural denominator mismatch")

    contract = protocol.get("natural_contract")
    require(isinstance(contract, dict), "protocol lacks natural_contract")
    require(contract.get("holdout_loaded") is False,
            "natural contract reports holdout access")
    require(contract.get("mechanism_authorized") is False,
            "natural contract improperly authorizes mechanism work")
    require(protocol.get("holdout_loaded") is False and
            protocol.get("mechanism_authorized") is False,
            "protocol crosses the holdout/mechanism boundary")
    if "max_new_tokens" in contract:
        require(contract["max_new_tokens"] == core.MAX_NEW_TOKENS,
                "natural contract max_new_tokens mismatch")
    if "single_rollout_per_row" in contract:
        require(contract["single_rollout_per_row"] is True,
                "natural contract is not single-rollout")
    if "per_row_independent_generator" in contract:
        require(contract["per_row_independent_generator"] is True,
                "natural contract does not require independent generators")

    script_entries = protocol.get("phase979_script_hashes")
    require(isinstance(script_entries, dict) and script_entries,
            "protocol lacks Phase979 script hashes")
    verified_paths: dict[Path, str] = {}
    for label, entry in script_entries.items():
        require(isinstance(entry, dict) and "path" in entry and "sha256" in entry,
                f"invalid Phase979 script commitment: {label}")
        path = verify_file_commitment(entry, f"Phase979 script {label}")
        verified_paths[path] = str(entry["sha256"])
    for path in (SCRIPT_PATH, CORE_PATH, DATASET_PATH):
        require(path.resolve() in verified_paths,
                f"protocol does not seal required runtime file: {path.name}")
        require(verified_paths[path.resolve()] == core.sha256_file(path),
                f"required runtime file differs from seal: {path.name}")

    data_identity = protocol.get("natural_dataset_identity")
    require(isinstance(data_identity, dict),
            "protocol lacks natural_dataset_identity")
    # The dataset module itself is already an exact Phase979 script commitment.
    # Protocol builders may store either the stable audit identity directly or
    # wrap it with path/file-SHA metadata; load_items authenticates the identity.
    if "path" in data_identity or "sha256" in data_identity:
        require(isinstance(data_identity.get("path"), str) and
                isinstance(data_identity.get("sha256"), str),
                "wrapped natural_dataset_identity lacks path/SHA")
        require((ROOT / str(data_identity["path"])).resolve() ==
                DATASET_PATH.resolve(), "protocol natural dataset path mismatch")
        require(core.sha256_file(DATASET_PATH) == data_identity["sha256"],
                "natural dataset module differs from seal")

    lineage = protocol.get("phase978_commitments")
    require(isinstance(lineage, dict), "protocol lacks Phase978 commitments")
    require(lineage.get("development_gate_passed") is False and
            lineage.get("holdout_authorized") is False and
            lineage.get("holdout_loaded") is False,
            "protocol does not preserve the Phase978 NO-GO boundary")
    phase978 = authenticate_phase978_no_go(lineage)
    authenticate_model_identity(protocol)
    assert_holdout_not_imported()
    return protocol, phase978


def call_dataset_audit(items: list[dict[str, Any]]) -> dict[str, Any]:
    audit_fn = getattr(diagnostic_dataset, "audit_items", None)
    require(callable(audit_fn), "diagnostic dataset lacks audit_items")
    parameters = inspect.signature(audit_fn).parameters
    audit = audit_fn(items) if parameters else audit_fn()
    require(isinstance(audit, dict), "diagnostic audit must return an object")
    for flag in ("ok", "passed"):
        if flag in audit:
            require(audit[flag] is True, f"diagnostic audit {flag}=false")
    require(not audit.get("errors"), "diagnostic dataset audit reported errors")
    return audit


def load_items(protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    build_fn = getattr(diagnostic_dataset, "build_items", None)
    require(callable(build_fn), "diagnostic dataset lacks build_items")
    raw = build_fn()
    require(isinstance(raw, list), "build_items must return a list")
    require(all(isinstance(item, dict) for item in raw),
            "diagnostic item is not an object")
    items = [dict(item) for item in raw]
    require(len(items) == EXPECTED_ITEMS, "diagnostic dataset is not 128 items")
    ids = [str(item.get("id")) for item in items]
    require(len(set(ids)) == EXPECTED_ITEMS and "None" not in ids,
            "diagnostic item IDs are missing or duplicated")
    require(all(str(item.get("answer")) in {"A", "B"} for item in items),
            "natural diagnostic answers must be exact A/B labels")
    require(all(str(item.get("prompt", "")).strip() for item in items),
            "natural diagnostic contains an empty prompt")
    items.sort(key=lambda item: str(item["id"]))
    audit = call_dataset_audit(items)
    dataset_sha = core.sha256_json(items)
    identity = protocol["natural_dataset_identity"]
    audited_identity = audit.get("identity")
    require(isinstance(audited_identity, dict),
            "diagnostic audit lacks stable identity")
    nested_identity = identity.get("identity")
    if nested_identity is not None:
        protocol_identity = nested_identity
    else:
        # phase979_protocol.py flattens the stable identity beside the committed
        # dataset-module path/SHA.  Those wrapper fields are not part of the
        # dataset's independently audited identity.
        wrapper_fields = {"path", "sha256", "dataset_sha256"}
        protocol_identity = {
            key: value for key, value in identity.items()
            if key not in wrapper_fields
        }
    require(isinstance(protocol_identity, dict),
            "protocol natural dataset stable identity is invalid")
    expected_dataset_sha = (identity.get("dataset_sha256") or
                            protocol_identity.get("items_sha256"))
    require(isinstance(expected_dataset_sha, str),
            "natural_dataset_identity lacks dataset/items SHA")
    require(dataset_sha == expected_dataset_sha,
            "runtime natural items differ from protocol seal")
    require(protocol_identity == audited_identity,
            "protocol/audited natural dataset identities differ")
    if "n_items" in protocol_identity:
        require(protocol_identity["n_items"] == EXPECTED_ITEMS,
                "protocol natural item count mismatch")
    require(len(core.CONTROL_POLICIES) * len(core.DECODING_POLICIES)
            * len(core.REPLICATES) * len(items) == EXPECTED_ROWS,
            "factorial grid is not 2048 rows")
    assert_holdout_not_imported()
    return items, audit, dataset_sha


def get_eos_ids(model, tokenizer) -> list[int]:
    values = (
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
    )
    output: list[int] = []
    for value in values:
        if value is None:
            continue
        candidates = value if isinstance(value, (list, tuple, set)) else [value]
        for candidate in candidates:
            if candidate is not None and int(candidate) not in output:
                output.append(int(candidate))
    require(bool(output), "no EOS token IDs found")
    return output


def make_manifest(
    protocol: dict[str, Any], phase978: dict[str, Any], items: list[dict[str, Any]],
    data_audit: dict[str, Any], dataset_sha: str, model, tok, device,
    eos_ids: list[int], pad_token_id: int, think_open_id: int,
    think_close_id: int,
) -> dict[str, Any]:
    template_probes: dict[str, Any] = {}
    for control in core.CONTROL_POLICIES:
        user_prompt, rendered, input_ids = core.render_prefix(tok, items[0], control)
        template_probes[control] = {
            "effective_user_prompt": user_prompt,
            "rendered_prefix_sha256": core.sha256_json(rendered),
            "input_ids": input_ids,
            "prompt_len": len(input_ids),
        }
    manifest_core = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": NATURAL_EXPERIMENT,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": core.sha256_file(PROTOCOL_PATH),
        "phase978_no_go": phase978,
        "runner_sha256": core.sha256_file(SCRIPT_PATH),
        "boundary_core_sha256": core.sha256_file(CORE_PATH),
        "dataset_module_sha256": core.sha256_file(DATASET_PATH),
        "dataset_sha256": dataset_sha,
        "dataset_audit_sha256": core.sha256_json(data_audit),
        "dataset_identity": data_audit["identity"],
        "n_items": len(items),
        "expected_rows": EXPECTED_ROWS,
        "model_name": MODEL_NAME,
        "model_path": str(Path(MODEL_CONFIGS[MODEL_NAME]["path"]).resolve()),
        "model_class": type(model).__name__,
        "model_dtype": str(getattr(model, "dtype", "unknown")),
        "device_type": getattr(device, "type", str(device).split(":")[0]),
        "tokenizer_class": type(tok).__name__,
        "tokenizer_length": len(tok),
        "tokenizer_chat_template_sha256": hashlib.sha256(
            str(getattr(tok, "chat_template", "")).encode("utf-8")).hexdigest(),
        "eos_token_ids": eos_ids,
        "pad_token_id": int(pad_token_id),
        "think_open_id": int(think_open_id),
        "think_close_id": int(think_close_id),
        "special_token_ids": {
            "think_open": int(think_open_id),
            "think_close": int(think_close_id),
        },
        "controls": core.CONTROL_POLICIES,
        "decoding_policies": core.DECODING_POLICIES,
        "official_cells": sorted([list(value) for value in core.OFFICIAL_CELLS]),
        "replicates": list(core.REPLICATES),
        "checkpoints": list(core.CHECKPOINTS),
        "max_new_tokens": core.MAX_NEW_TOKENS,
        "batch_size": core.BATCH_SIZE,
        "template_probes": template_probes,
        "runtime_versions": runtime_versions(),
        "generation_contract": {
            "single_rollout_per_factorial_row": True,
            "checkpoint_snapshots_are_prefixes_of_single_rollout": True,
            "left_padding": True,
            "explicit_attention_mask": True,
            "per_row_independent_cuda_generator": True,
            "seed_function": "phase979_boundary_core.stable_seed",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "cudnn_benchmark": False,
            "sampling_algorithm": (
                "temperature -> top_k -> top_p -> min_p -> multinomial; "
                "top_k support is sampled with the row-owned torch.Generator"
            ),
            "first_actual_eos_absorbing": True,
            "holdout_loaded": False,
            "mechanism_authorized": False,
        },
        "holdout_loaded": False,
        "mechanism_authorized": False,
    }
    return {
        **manifest_core,
        "manifest_sha256": core.sha256_json(manifest_core),
        "created_at_utc": core.utc_now(),
    }


def install_or_validate_manifest(document: dict[str, Any]) -> dict[str, Any]:
    verify_self_hash(document, "manifest_sha256", MANIFEST_HASH_FIELDS,
                     "new natural manifest")
    if MANIFEST_PATH.exists():
        prior = load_json(MANIFEST_PATH, "existing natural manifest")
        verify_self_hash(prior, "manifest_sha256", MANIFEST_HASH_FIELDS,
                         "existing natural manifest")
        require(prior["manifest_sha256"] == document["manifest_sha256"],
                "existing natural manifest differs; refusing resume")
        return prior
    core.atomic_write_json(MANIFEST_PATH, document)
    return document


def repair_truncated_jsonl_tail(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
        handle.seek(0)
        payload = handle.read()
        boundary = payload.rfind(b"\n")
        handle.seek(0)
        handle.truncate(boundary + 1 if boundary >= 0 else 0)
        handle.flush()
        os.fsync(handle.fileno())


def row_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in ROW_HASH_FIELDS}


def build_row(
    manifest: dict[str, Any], tok, item: dict[str, Any], control: str,
    decoding: str, replicate: int, batch_index: int, input_ids: list[int],
    user_prompt: str, generated_ids: list[int], eos_ids: list[int],
    think_open_id: int, think_close_id: int,
) -> dict[str, Any]:
    trimmed = core.trim_at_first_eos(generated_ids, eos_ids)
    require(trimmed == generated_ids, "batch sampler retained tokens after first EOS")
    seed = core.stable_seed(str(item["id"]), control, decoding, replicate)
    _prompt, rendered, rerendered_ids = core.render_prefix(tok, item, control)
    require(_prompt == user_prompt and rerendered_ids == input_ids,
            "row construction prefix changed within one batch")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": NATURAL_EXPERIMENT,
        "manifest_sha256": manifest["manifest_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "id": str(item["id"]),
        "task": str(item.get("task", "")),
        "prompt": str(item["prompt"]),
        "answer": str(item["answer"]),
        "control_policy": control,
        "decoding_policy": decoding,
        "official_cell": (control, decoding) in core.OFFICIAL_CELLS,
        "replicate": int(replicate),
        "seed": int(seed),
        "batch_index": int(batch_index),
        "effective_user_prompt": user_prompt,
        "rendered_prefix_sha256": core.sha256_json(rendered),
        "input_ids": [int(value) for value in input_ids],
        "prompt_len": len(input_ids),
        "sampling": dict(core.DECODING_POLICIES[decoding]),
        "max_new_tokens": core.MAX_NEW_TOKENS,
        "generated_ids": [int(value) for value in generated_ids],
        "generated_raw": tok.decode(generated_ids, skip_special_tokens=False),
        "generated_plain": tok.decode(generated_ids, skip_special_tokens=True),
        "checkpoints": core.analyze_checkpoints(
            tok, item, control, generated_ids, eos_ids,
            think_open_id, think_close_id,
        ),
        "first_actual_eos_absorbing": True,
        "left_padded_batch_with_explicit_attention_mask": True,
        "per_row_independent_generator": True,
        "holdout_loaded": False,
        "mechanism_authorized": False,
    }
    return {**payload, "row_sha256": core.sha256_json(payload)}


def validate_row(
    row: dict[str, Any], manifest: dict[str, Any], tok,
    item_by_id: dict[str, dict[str, Any]], eos_ids: list[int],
    think_open_id: int, think_close_id: int,
) -> None:
    verify_self_hash(row, "row_sha256", ROW_HASH_FIELDS,
                     f"natural row {core.natural_key(row)}")
    key = core.natural_key(row)
    item_id, control, decoding, replicate = key
    require(item_id in item_by_id, f"unknown natural item: {item_id}")
    require(control in core.CONTROL_POLICIES, f"unknown natural control: {control}")
    require(decoding in core.DECODING_POLICIES,
            f"unknown natural decoding: {decoding}")
    require(replicate in core.REPLICATES, f"unknown natural replicate: {replicate}")
    require(row.get("schema_version") == SCHEMA_VERSION and
            row.get("phase") == PHASE and
            row.get("experiment") == NATURAL_EXPERIMENT,
            f"row identity mismatch: {key}")
    require(row.get("manifest_sha256") == manifest["manifest_sha256"],
            f"row manifest mismatch: {key}")
    require(row.get("protocol_sha256") == manifest["protocol_sha256"],
            f"row protocol mismatch: {key}")
    item = item_by_id[item_id]
    for field in ("id", "task", "prompt", "answer"):
        require(row.get(field) == str(item.get(field, "")),
                f"row {field} mismatch: {key}")
    require(row.get("seed") == core.stable_seed(
        item_id, control, decoding, replicate), f"row seed mismatch: {key}")
    require(row.get("sampling") == core.DECODING_POLICIES[decoding],
            f"row sampling mismatch: {key}")
    require(row.get("official_cell") is
            ((control, decoding) in core.OFFICIAL_CELLS),
            f"row official-cell marker mismatch: {key}")
    item_position = list(item_by_id).index(item_id)
    flat_position = (
        ((list(core.REPLICATES).index(replicate) * len(core.CONTROL_POLICIES)
          + list(core.CONTROL_POLICIES).index(control))
         * len(core.DECODING_POLICIES)
         + list(core.DECODING_POLICIES).index(decoding))
        * len(item_by_id) + item_position
    )
    require(row.get("batch_index") == flat_position // core.BATCH_SIZE + 1,
            f"row canonical batch index mismatch: {key}")
    require(row.get("max_new_tokens") == core.MAX_NEW_TOKENS and
            row.get("first_actual_eos_absorbing") is True and
            row.get("left_padded_batch_with_explicit_attention_mask") is True and
            row.get("per_row_independent_generator") is True,
            f"row generation contract mismatch: {key}")
    user_prompt, rendered, input_ids = core.render_prefix(tok, item, control)
    require(row.get("effective_user_prompt") == user_prompt and
            row.get("rendered_prefix_sha256") == core.sha256_json(rendered) and
            row.get("input_ids") == input_ids and
            row.get("prompt_len") == len(input_ids),
            f"row prefix mismatch: {key}")
    generated = row.get("generated_ids")
    require(isinstance(generated, list) and 0 < len(generated) <= core.MAX_NEW_TOKENS,
            f"invalid generated IDs: {key}")
    require(all(isinstance(value, int) and not isinstance(value, bool)
                for value in generated), f"non-integer generated ID: {key}")
    require(core.trim_at_first_eos(generated, eos_ids) == generated,
            f"row retains tokens after EOS: {key}")
    eos_positions = core.positions_of(generated, set(eos_ids))
    require((len(eos_positions) == 1 and eos_positions[0] == len(generated) - 1)
            or (not eos_positions and len(generated) == core.MAX_NEW_TOKENS),
            f"row termination is neither terminal EOS nor cap censoring: {key}")
    require(row.get("generated_raw") == tok.decode(
        generated, skip_special_tokens=False), f"row raw decode mismatch: {key}")
    require(row.get("generated_plain") == tok.decode(
        generated, skip_special_tokens=True), f"row plain decode mismatch: {key}")
    expected_checkpoints = core.analyze_checkpoints(
        tok, item, control, generated, eos_ids, think_open_id, think_close_id,
    )
    require(row.get("checkpoints") == expected_checkpoints,
            f"row checkpoint analysis mismatch: {key}")
    require(row.get("holdout_loaded") is False and
            row.get("mechanism_authorized") is False,
            f"row violates isolation contract: {key}")


def load_rows(
    manifest: dict[str, Any], tok, items: list[dict[str, Any]],
    eos_ids: list[int], think_open_id: int, think_close_id: int,
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    repair_truncated_jsonl_tail(ROWS_PATH)
    if not ROWS_PATH.exists() or ROWS_PATH.stat().st_size == 0:
        return {}
    payload = ROWS_PATH.read_bytes()
    require(payload.endswith(b"\n"), "natural JSONL lacks final newline")
    item_by_id = {str(item["id"]): item for item in items}
    rows: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for line_number, raw in enumerate(payload.splitlines(), 1):
        require(bool(raw.strip()), f"blank natural JSONL line: {line_number}")
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"malformed natural JSONL line {line_number}") from exc
        require(isinstance(row, dict), f"natural line {line_number} is not an object")
        validate_row(row, manifest, tok, item_by_id, eos_ids,
                     think_open_id, think_close_id)
        key = core.natural_key(row)
        require(key not in rows, f"duplicate natural key: {key}")
        rows[key] = row
    return rows


def expected_grid(items: list[dict[str, Any]]) -> list[
        tuple[dict[str, Any], str, str, int]]:
    output: list[tuple[dict[str, Any], str, str, int]] = []
    for replicate in core.REPLICATES:
        for control in core.CONTROL_POLICIES:
            for decoding in core.DECODING_POLICIES:
                for item in items:
                    output.append((item, control, decoding, int(replicate)))
    require(len(output) == EXPECTED_ROWS, "internal factorial grid size mismatch")
    return output


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    require(size > 0 and len(values) % size == 0,
            "canonical Phase979 grid must divide into full batches")
    for start in range(0, len(values), size):
        yield values[start:start + size]


def make_logits_warpers(spec: dict[str, Any]) -> tuple[Any, ...]:
    """Construct the standard Transformers warpers in the frozen order."""
    from transformers.generation.logits_process import (
        MinPLogitsWarper,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    temperature = core.finite_number(spec["temperature"], "temperature")
    top_p = core.finite_number(spec["top_p"], "top_p")
    min_p = core.finite_number(spec["min_p"], "min_p")
    top_k = int(spec["top_k"])
    require(temperature > 0.0, "temperature must be positive")
    require(0.0 < top_p <= 1.0, "top_p must be in (0,1]")
    require(0.0 <= min_p <= 1.0, "min_p must be in [0,1]")
    require(top_k > 0, "top_k must be positive")
    return (
        TemperatureLogitsWarper(temperature),
        TopKLogitsWarper(top_k),
        TopPLogitsWarper(top_p),
        MinPLogitsWarper(min_p),
    )


def apply_logits_warpers(
    logits: torch.Tensor, warpers: tuple[Any, ...],
) -> torch.Tensor:
    require(logits.ndim == 2, "batch sampling logits must be two-dimensional")
    scores = logits.float()
    # These standard warpers use input_ids only for the common processor API;
    # none of the four reads token history.  A one-column shape is sufficient.
    input_ids = torch.zeros(
        (scores.shape[0], 1), dtype=torch.long, device=scores.device)
    for warper in warpers:
        scores = warper(input_ids, scores)
    return scores


def sample_from_logits(
    logits: torch.Tensor, spec: dict[str, Any], generator: torch.Generator,
) -> torch.Tensor:
    """Sample one row with standard warpers and its private generator."""
    require(logits.ndim == 1, "sampling logits must be one-dimensional")
    processed = apply_logits_warpers(
        logits.unsqueeze(0), make_logits_warpers(spec))[0]
    probabilities = torch.softmax(processed, dim=-1)
    selected = torch.multinomial(probabilities, num_samples=1,
                                 replacement=True, generator=generator)
    return selected.squeeze(0)


def left_pad_prompts(
    prompt_ids: list[list[int]], pad_token_id: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require(len(prompt_ids) == core.BATCH_SIZE,
            "formal natural batches must contain exactly eight rows")
    max_len = max(len(values) for values in prompt_ids)
    require(max_len > 0, "empty rendered prompt")
    input_ids = torch.full(
        (len(prompt_ids), max_len), int(pad_token_id),
        dtype=torch.long, device=device,
    )
    attention_mask = torch.zeros(
        (len(prompt_ids), max_len), dtype=torch.long, device=device,
    )
    for index, values in enumerate(prompt_ids):
        length = len(values)
        input_ids[index, max_len - length:] = torch.tensor(
            values, dtype=torch.long, device=device)
        attention_mask[index, max_len - length:] = 1
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    require(torch.equal(attention_mask.sum(dim=-1).cpu(), torch.tensor(
        [len(values) for values in prompt_ids], dtype=torch.long)),
        "explicit attention mask does not match prompt lengths")
    return input_ids, attention_mask, position_ids


def sample_batch(
    model, tok, device: torch.device,
    jobs: list[tuple[dict[str, Any], str, str, int]],
    eos_ids: list[int], pad_token_id: int,
) -> tuple[list[list[int]], list[list[int]], list[str]]:
    require(len(jobs) == core.BATCH_SIZE, "sample_batch requires batch_size=8")
    prompt_ids: list[list[int]] = []
    user_prompts: list[str] = []
    generators: list[torch.Generator] = []
    for item, control, decoding, replicate in jobs:
        user_prompt, _rendered, ids = core.render_prefix(tok, item, control)
        prompt_ids.append(ids)
        user_prompts.append(user_prompt)
        generator = torch.Generator(device=device)
        generator.manual_seed(core.stable_seed(
            str(item["id"]), control, decoding, replicate))
        generators.append(generator)

    input_ids, attention_mask, position_ids = left_pad_prompts(
        prompt_ids, pad_token_id, device)
    generated: list[list[int]] = [[] for _ in jobs]
    active = [True for _ in jobs]
    eos_set = set(int(value) for value in eos_ids)
    batch_decodings = {decoding for _item, _control, decoding, _replicate in jobs}
    require(len(batch_decodings) == 1,
            "canonical batch unexpectedly mixes decoding policies")
    batch_decoding = next(iter(batch_decodings))
    warpers = make_logits_warpers(core.DECODING_POLICIES[batch_decoding])

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        del outputs, input_ids, position_ids

        for step in range(core.MAX_NEW_TOKENS):
            processed_logits = apply_logits_warpers(logits, warpers)
            probabilities = torch.softmax(processed_logits, dim=-1)
            sampled_tensors: list[torch.Tensor] = []
            for index, (_item, _control, _decoding, _replicate) in enumerate(jobs):
                if active[index]:
                    sampled_tensors.append(torch.multinomial(
                        probabilities[index], num_samples=1, replacement=True,
                        generator=generators[index]).squeeze(0))
                else:
                    sampled_tensors.append(torch.tensor(
                        int(pad_token_id), dtype=torch.long, device=device))
            next_tokens = torch.stack(sampled_tensors).to(dtype=torch.long)
            next_values = [int(value) for value in next_tokens.tolist()]
            next_active: list[bool] = []
            for index, value in enumerate(next_values):
                if active[index]:
                    generated[index].append(value)
                    next_active.append(value not in eos_set)
                else:
                    next_active.append(False)
            active = next_active
            if not any(active) or step + 1 == core.MAX_NEW_TOKENS:
                break

            step_mask = torch.tensor(
                [1 if value else 0 for value in active],
                dtype=attention_mask.dtype, device=device,
            ).unsqueeze(1)
            attention_mask = torch.cat((attention_mask, step_mask), dim=1)
            step_ids = next_tokens.unsqueeze(1)
            step_positions = attention_mask.sum(dim=-1, keepdim=True) - 1
            step_positions.clamp_min_(0)
            outputs = model(
                input_ids=step_ids,
                attention_mask=attention_mask,
                position_ids=step_positions,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            del (outputs, step_ids, step_positions, step_mask, next_tokens,
                 processed_logits, probabilities)

    for ids in generated:
        require(0 < len(ids) <= core.MAX_NEW_TOKENS,
                "sampler produced an invalid trajectory length")
        require(core.trim_at_first_eos(ids, eos_ids) == ids,
                "sampler did not stop a row at first EOS")
    del logits, past_key_values, attention_mask, processed_logits, probabilities
    return prompt_ids, generated, user_prompts


def acquire_run_lock(protocol_sha256: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            RUN_LOCK_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            "Phase979 natural run lock already exists; concurrent run or stale lock"
        ) from exc
    payload = (core.canonical_json({
        "pid": os.getpid(),
        "protocol_sha256": protocol_sha256,
        "acquired_at_utc": core.utc_now(),
    }) + "\n").encode("utf-8")
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, "failed to write Phase979 natural run lock")
            view = view[written:]
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        RUN_LOCK_PATH.unlink(missing_ok=True)
        raise
    return descriptor


def release_run_lock(descriptor: int) -> None:
    os.close(descriptor)
    RUN_LOCK_PATH.unlink(missing_ok=True)


def write_status(
    manifest: dict[str, Any], rows: dict[tuple[str, str, str, int], dict[str, Any]],
    elapsed: float, complete: bool,
) -> dict[str, Any]:
    counts = Counter((key[1], key[2], key[3]) for key in rows)
    status_core = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": NATURAL_EXPERIMENT,
        "protocol_sha256": manifest["protocol_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "dataset_identity": manifest["dataset_identity"],
        "expected_rows": EXPECTED_ROWS,
        "completed_rows": len(rows),
        "completed_by_cell_replicate": {
            f"{control}|{decoding}|r{replicate}": counts.get(
                (control, decoding, replicate), 0)
            for replicate in core.REPLICATES
            for control in core.CONTROL_POLICIES
            for decoding in core.DECODING_POLICIES
        },
        "complete": bool(complete),
        "elapsed_seconds_this_invocation": float(elapsed),
        "single_rollout_checkpoint_contract": True,
        "independent_per_row_generators": True,
        "holdout_loaded": False,
        "mechanism_authorized": False,
        "model_weights_loaded": True,
        "generation_performed": len(rows) > 0,
    }
    document = {
        **status_core,
        "status_sha256": core.sha256_json(status_core),
        "updated_at_utc": core.utc_now(),
    }
    core.atomic_write_json(STATUS_PATH, document)
    return document


def run() -> None:
    assert_holdout_not_imported()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    require(torch.backends.cuda.matmul.allow_tf32 is False and
            torch.backends.cudnn.allow_tf32 is False and
            torch.backends.cudnn.benchmark is False,
            "formal Phase979 natural run requires deterministic backend settings")
    require(torch.cuda.is_available(), "formal Phase979 natural run requires CUDA")
    require(core.PHASE == PHASE and core.SCHEMA_VERSION == SCHEMA_VERSION,
            "Phase979 boundary core identity mismatch")
    require(core.MAX_NEW_TOKENS == 2048 and core.BATCH_SIZE == 8,
            "Phase979 boundary core budget/batch mismatch")
    t0 = time.time()
    protocol, phase978 = authenticate_protocol()
    items, data_audit, dataset_sha = load_items(protocol)
    assert_holdout_not_imported()
    lock = acquire_run_lock(protocol["protocol_sha256"])
    model = None
    try:
        model, tok, device = load_model(MODEL_NAME)
        require(getattr(device, "type", str(device).split(":")[0]) == "cuda",
                f"Qwen3 did not load on CUDA: {device}")
        device = torch.device(device)
        tok.padding_side = "left"
        eos_ids = get_eos_ids(model, tok)
        pad_token_id = tok.pad_token_id
        require(pad_token_id is not None, "tokenizer has no pad token ID")
        think_open_id = core.single_token_id(tok, "<think>")
        think_close_id = core.single_token_id(tok, "</think>")
        manifest = install_or_validate_manifest(make_manifest(
            protocol, phase978, items, data_audit, dataset_sha, model, tok, device,
            eos_ids, int(pad_token_id), think_open_id, think_close_id,
        ))
        rows = load_rows(
            manifest, tok, items, eos_ids, think_open_id, think_close_id)
        grid = expected_grid(items)
        expected_keys = {
            (str(item["id"]), control, decoding, replicate)
            for item, control, decoding, replicate in grid
        }
        require(set(rows).issubset(expected_keys),
                "natural JSONL contains keys outside the sealed factorial grid")
        write_status(manifest, rows, time.time() - t0,
                     complete=len(rows) == EXPECTED_ROWS)

        total_batches = EXPECTED_ROWS // core.BATCH_SIZE
        item_by_id = {str(item["id"]): item for item in items}
        for batch_index, jobs in enumerate(chunks(grid, core.BATCH_SIZE), 1):
            batch_keys = [
                (str(item["id"]), control, decoding, replicate)
                for item, control, decoding, replicate in jobs
            ]
            if all(key in rows for key in batch_keys):
                continue
            prompt_ids, generated, user_prompts = sample_batch(
                model, tok, device, jobs, eos_ids, int(pad_token_id))
            built: list[dict[str, Any]] = []
            for index, (item, control, decoding, replicate) in enumerate(jobs):
                row = build_row(
                    manifest, tok, item, control, decoding, replicate,
                    batch_index, prompt_ids[index], user_prompts[index],
                    generated[index], eos_ids, think_open_id, think_close_id,
                )
                validate_row(row, manifest, tok, item_by_id, eos_ids,
                             think_open_id, think_close_id)
                key = core.natural_key(row)
                if key in rows:
                    require(rows[key]["row_sha256"] == row["row_sha256"],
                            f"partial-batch replay changed existing row: {key}")
                built.append(row)
            for row in built:
                key = core.natural_key(row)
                if key not in rows:
                    core.append_jsonl(ROWS_PATH, row)
                    rows[key] = row
            complete = len(rows) == EXPECTED_ROWS
            write_status(manifest, rows, time.time() - t0, complete=complete)
            if batch_index % 8 == 0 or complete:
                log(f"natural batches {batch_index}/{total_batches}; "
                    f"rows={len(rows)}/{EXPECTED_ROWS}")
            assert_holdout_not_imported()

        require(set(rows) == expected_keys and len(rows) == EXPECTED_ROWS,
                "Phase979 natural factorial did not complete its full denominator")
        final_status = write_status(
            manifest, rows, time.time() - t0, complete=True)
        verify_self_hash(final_status, "status_sha256", STATUS_HASH_FIELDS,
                         "final natural generator status")
        assert_holdout_not_imported()
        require(not PHASE978_OPEN_RECEIPT_PATH.exists(),
                "Phase978 OPEN receipt appeared during Phase979 run")
        log(f"Phase979 natural factorial complete: {EXPECTED_ROWS} rows; "
            f"elapsed={(time.time()-t0)/3600:.2f} h")
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        release_run_lock(lock)
        assert_holdout_not_imported()


if __name__ == "__main__":
    run()
