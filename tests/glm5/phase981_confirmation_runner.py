#!/usr/bin/env python3
"""Sealed CUDA runner for Phase 981 fresh256 soft-configuration confirmation."""
from __future__ import annotations

import argparse
import gc
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


ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

import phase979_boundary_core as boundary  # noqa: E402
import phase981_confirmation_core as core  # noqa: E402
import phase981_fresh_dataset as fresh  # noqa: E402
from model_utils import MODEL_CONFIGS, load_model, release_model  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def log(message: str) -> None:
    print(f"[phase981] {message}", flush=True)


def assert_no_holdout_import() -> None:
    forbidden = [name for name in sys.modules if name == "phase977_holdout_dataset"
                 or name.endswith(".phase977_holdout_dataset")]
    require(not forbidden, f"old holdout module imported: {forbidden}")


def verify_self_hash(
    value: dict[str, Any], hash_field: str, time_field: str, label: str,
) -> None:
    payload = boundary.without_fields(value, hash_field, time_field)
    require(value.get(hash_field) == boundary.sha256_json(payload),
            f"{label} self-hash invalid")


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_full": sys.version,
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "version_source": "installed_distribution_metadata_only",
    }


def dataset_identity(audit: dict[str, Any]) -> dict[str, Any]:
    identity = audit.get("identity")
    if not isinstance(identity, dict) and hasattr(fresh, "dataset_identity"):
        identity = fresh.dataset_identity()
    require(isinstance(identity, dict), "fresh dataset audit lacks identity")
    output = dict(identity)
    output["identity_sha256"] = (
        output.get("identity_sha256") or output.get("dataset_identity_sha256")
    )
    require(isinstance(output["identity_sha256"], str)
            and len(output["identity_sha256"]) == 64,
            "fresh dataset identity hash invalid")
    return output


def authenticate_protocol_and_admission() -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]
]:
    assert_no_holdout_import()
    protocol = boundary.load_json(core.PROTOCOL_PATH, "Phase981 protocol")
    admission = boundary.load_json(core.ADMISSION_PATH, "Phase981 admission")
    seal_audit = core.verify_protocol_file_seals(protocol)
    verify_self_hash(protocol, "protocol_sha256", "created_at_utc", "Phase981 protocol")
    verify_self_hash(admission, "admission_sha256", "admitted_at_utc",
                     "Phase981 admission")
    require(protocol.get("phase") == core.PHASE
            and admission.get("phase") == core.PHASE,
            "wrong Phase981 source phase")
    core.verify_protocol_boundary_contract(protocol)
    core.verify_admission_boundary_contract(admission)
    require(admission.get("protocol_sha256") == protocol["protocol_sha256"],
            "admission/protocol mismatch")
    dependency_audit = admission.get("dependency_audit", {})
    require(dependency_audit.get("script_seals_sha256")
            == seal_audit["script_seals_sha256"]
            and dependency_audit.get("dependency_seals_sha256")
            == seal_audit["dependency_seals_sha256"]
            and dependency_audit.get("phase979_script_hashes_sha256")
            == seal_audit["phase979_script_hashes_sha256"],
            "admission did not revalidate frozen code seals")
    require(protocol.get("arms") == core.ARMS
            and protocol.get("direction") == core.PRIMARY_DIRECTION
            and protocol.get("sampling") == core.SAMPLING,
            "formal A/B configuration changed")
    require(protocol.get("expected_rows") == core.EXPECTED_ROWS
            and protocol.get("batch_size") == core.BATCH_SIZE
            and protocol.get("max_new_tokens") == core.MAX_NEW_TOKENS
            and protocol.get("checkpoints") == list(core.CHECKPOINTS),
            "formal execution denominator changed")
    require(protocol.get("primary_decision", {}).get("route") == "semantic_only"
            and protocol.get("primary_decision", {}).get("censor_route_can_admit") is False,
            "runner protocol is not semantic-only")
    require(seal_audit["verified_script_count"] == len(core.PHASE981_SCRIPT_PATHS)
            and seal_audit["verified_dependency_count"]
            == len(core.RUNTIME_DEPENDENCY_PATHS)
            and seal_audit["verified_phase979_script_count"]
            == len(core.PHASE979_SCRIPT_PATHS),
            "frozen code seal denominator changed")
    core.verify_protocol_integrity_metadata(protocol)
    require(runtime_versions() == protocol.get("runtime_versions"),
            "formal runtime changed")
    model_identity = protocol.get("local_model_artifact_identity", {})
    core.verify_model_artifact_identity(model_identity)
    model_root = ROOT / str(model_identity["path"])
    configured = Path(MODEL_CONFIGS[core.MODEL_NAME]["path"]).resolve()
    require(model_root.resolve() == configured and model_root.is_dir(),
            "formal model path changed")
    core.verify_protocol_token_identity(
        protocol.get("tokenizer_audit"), core.EXPECTED_TOKENIZER_EOS_ID,
        core.EXPECTED_THINK_OPEN_ID, core.EXPECTED_THINK_CLOSE_ID,
        core.EXPECTED_A_ID, core.EXPECTED_B_ID)
    items = fresh.build_items()
    data_audit = fresh.audit_items(items)
    require(data_audit.get("passed") is True or data_audit.get("ok") is True,
            "fresh dataset audit failed")
    identity = dataset_identity(data_audit)
    require(identity["identity_sha256"]
            == protocol.get("dataset_identity", {}).get("identity_sha256"),
            "fresh dataset identity differs from protocol")
    require(boundary.sha256_json(data_audit) == protocol.get("dataset_audit_sha256"),
            "fresh dataset audit differs from protocol")
    artifact_seals = protocol.get("dataset_artifact_seals", {})
    require(boundary.sha256_file(core.DATASET_ARTIFACT_PATH)
            == artifact_seals.get("dataset", {}).get("file_sha256")
            and boundary.sha256_file(core.DATASET_AUDIT_PATH)
            == artifact_seals.get("audit", {}).get("file_sha256"),
            "fresh dataset artifact files differ from protocol")
    dataset_document = boundary.load_json(
        core.DATASET_ARTIFACT_PATH, "fresh dataset artifact")
    artifact_audit = boundary.load_json(
        core.DATASET_AUDIT_PATH, "fresh dataset artifact audit")
    artifact_items = dataset_document.get("items")
    require(dataset_document.get("dataset_sha256")
            == artifact_seals.get("dataset", {}).get("dataset_sha256")
            and artifact_audit.get("audit_sha256")
            == artifact_seals.get("audit", {}).get("audit_sha256")
            and isinstance(artifact_items, list)
            and boundary.sha256_json(sorted(
                artifact_items, key=lambda item: str(item.get("id", ""))))
            == boundary.sha256_json(sorted(
                items, key=lambda item: str(item.get("id", ""))))
            and dataset_document.get("identity", {}).get("items_sha256")
            == identity.get("items_sha256"),
            "fresh dataset artifacts changed")
    require(len(core.expected_keys(items)) == core.EXPECTED_ROWS,
            "expected key grid changed")
    assert_no_holdout_import()
    return protocol, admission, items, data_audit


def preflight() -> dict[str, Any]:
    """CPU-only final authentication; never loads a model or writes output."""
    protocol, admission, items, data_audit = authenticate_protocol_and_admission()
    require(not core.RUN_LOCK_PATH.exists(), "runner lock already exists")
    require(not core.AUDIT_PATH.exists(),
            "independent audit already exists; generation is frozen")
    if not core.MANIFEST_PATH.exists():
        require(not core.ROWS_PATH.exists() and not core.STATUS_PATH.exists(),
                "rows/status exist without a manifest")
    if core.MANIFEST_PATH.exists():
        manifest = boundary.load_json(core.MANIFEST_PATH, "existing Phase981 manifest")
        verify_self_hash(manifest, "manifest_sha256", "created_at_utc",
                         "existing Phase981 manifest")
        require(manifest.get("protocol_sha256") == protocol["protocol_sha256"]
                and manifest.get("admission_sha256") == admission["admission_sha256"],
                "existing manifest lineage mismatch")
        core.verify_manifest_dependency_contract(manifest, protocol)
    return {
        "phase": core.PHASE,
        "protocol_sha256": protocol["protocol_sha256"],
        "admission_sha256": admission["admission_sha256"],
        "dataset_identity_sha256": dataset_identity(data_audit)["identity_sha256"],
        "item_count": len(items),
        "expected_rows": len(core.expected_keys(items)),
        "output_state": {
            "manifest_exists": core.MANIFEST_PATH.exists(),
            "rows_exists": core.ROWS_PATH.exists(),
            "status_exists": core.STATUS_PATH.exists(),
            "audit_exists": core.AUDIT_PATH.exists(),
        },
        "cpu_only": True,
        "model_weights_loaded": False,
        "generation_performed": False,
        "files_written": False,
        "gpu_authorized_by_admission": True,
        "holdout": False, "mechanism": False,
    }


def get_eos_ids(model, tok) -> list[int]:
    values: set[int] = set()
    for value in (
        getattr(tok, "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ):
        if isinstance(value, int):
            values.add(int(value))
        elif isinstance(value, (list, tuple, set)):
            values.update(int(item) for item in value)
    require(values, "runtime has no EOS token IDs")
    return sorted(values)


def make_logits_warpers(spec: dict[str, Any]) -> tuple[Any, ...]:
    from transformers.generation.logits_process import (
        MinPLogitsWarper, TemperatureLogitsWarper,
        TopKLogitsWarper, TopPLogitsWarper,
    )
    temperature = boundary.finite_number(spec["temperature"], "temperature")
    top_p = boundary.finite_number(spec["top_p"], "top_p")
    min_p = boundary.finite_number(spec["min_p"], "min_p")
    top_k = int(spec["top_k"])
    require(temperature > 0 and 0 < top_p <= 1 and 0 <= min_p <= 1 and top_k > 0,
            "invalid frozen sampling spec")
    return (
        TemperatureLogitsWarper(temperature), TopKLogitsWarper(top_k),
        TopPLogitsWarper(top_p), MinPLogitsWarper(min_p),
    )


def apply_warpers(logits: torch.Tensor, warpers: tuple[Any, ...]) -> torch.Tensor:
    require(logits.ndim == 2, "sampler logits must be rank two")
    scores = logits.float()
    dummy_ids = torch.zeros((scores.shape[0], 1), dtype=torch.long,
                            device=scores.device)
    for warper in warpers:
        scores = warper(dummy_ids, scores)
    return scores


def left_pad(
    prompts: list[list[int]], pad_id: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require(len(prompts) == core.BATCH_SIZE, "formal batch is not eight")
    width = max(len(values) for values in prompts)
    require(width > 0, "empty rendered prefix")
    input_ids = torch.full((len(prompts), width), int(pad_id),
                           dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    for index, values in enumerate(prompts):
        input_ids[index, width - len(values):] = torch.tensor(
            values, dtype=torch.long, device=device)
        attention[index, width - len(values):] = 1
    positions = attention.cumsum(dim=-1) - 1
    positions.masked_fill_(attention == 0, 0)
    return input_ids, attention, positions


def sample_batch(
    model, tok, device: torch.device,
    jobs: list[tuple[dict[str, Any], str, int]], identity_sha: str,
    eos_ids: list[int], pad_id: int,
) -> tuple[list[list[int]], list[list[int]], list[str]]:
    require(len(jobs) == core.BATCH_SIZE, "formal batch is not eight")
    require(len({arm for _item, arm, _stream in jobs}) == 1,
            "canonical batch mixes arms")
    prompts: list[list[int]] = []
    users: list[str] = []
    generators: list[torch.Generator] = []
    for item, arm, stream in jobs:
        user, _rendered, ids = core.render_prefix(tok, item, arm)
        prompts.append(ids)
        users.append(user)
        generator = torch.Generator(device=device)
        generator.manual_seed(core.stable_pair_seed(identity_sha, item["id"], stream))
        generators.append(generator)
    input_ids, attention, positions = left_pad(prompts, pad_id, device)
    generated = [[] for _ in jobs]
    active = [True] * len(jobs)
    eos_set = set(eos_ids)
    warpers = make_logits_warpers(core.SAMPLING)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention,
                        position_ids=positions, use_cache=True,
                        logits_to_keep=1, return_dict=True)
        cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        del outputs, input_ids, positions
        for step in range(core.MAX_NEW_TOKENS):
            probabilities = torch.softmax(apply_warpers(logits, warpers), dim=-1)
            sampled = []
            for index in range(len(jobs)):
                if active[index]:
                    sampled.append(torch.multinomial(
                        probabilities[index], 1, replacement=True,
                        generator=generators[index]).squeeze(0))
                else:
                    sampled.append(torch.tensor(pad_id, dtype=torch.long, device=device))
            next_tokens = torch.stack(sampled).long()
            values = [int(value) for value in next_tokens.tolist()]
            next_active: list[bool] = []
            for index, value in enumerate(values):
                if active[index]:
                    generated[index].append(value)
                    next_active.append(value not in eos_set)
                else:
                    next_active.append(False)
            active = next_active
            if not any(active) or step + 1 == core.MAX_NEW_TOKENS:
                break
            step_mask = torch.tensor(active, dtype=attention.dtype,
                                     device=device).unsqueeze(1)
            attention = torch.cat((attention, step_mask), dim=1)
            step_ids = next_tokens.unsqueeze(1)
            step_positions = attention.sum(dim=-1, keepdim=True) - 1
            step_positions.clamp_min_(0)
            outputs = model(input_ids=step_ids, attention_mask=attention,
                            position_ids=step_positions, past_key_values=cache,
                            use_cache=True, logits_to_keep=1, return_dict=True)
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            del outputs, step_ids, step_positions, step_mask, next_tokens, probabilities
    for ids in generated:
        require(0 < len(ids) <= core.MAX_NEW_TOKENS, "invalid generated length")
        require(boundary.trim_at_first_eos(ids, eos_ids) == ids,
                "tokens retained after first EOS")
    del logits, cache, attention
    return prompts, generated, users


def make_manifest(
    protocol: dict[str, Any], admission: dict[str, Any], data_audit: dict[str, Any],
    model, tok, device: torch.device, eos_ids: list[int], pad_id: int,
    think_open: int, think_close: int,
) -> dict[str, Any]:
    identity = dataset_identity(data_audit)
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE,
        "experiment": core.EXPERIMENT,
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": boundary.sha256_file(core.PROTOCOL_PATH),
        "admission_sha256": admission["admission_sha256"],
        "admission_file_sha256": boundary.sha256_file(core.ADMISSION_PATH),
        "dataset_identity": identity,
        "dataset_audit_sha256": boundary.sha256_json(data_audit),
        "expected_rows": core.EXPECTED_ROWS,
        "arms": core.ARMS,
        "direction": core.PRIMARY_DIRECTION,
        "streams": list(core.STREAMS),
        "sampling": core.SAMPLING,
        "checkpoints": list(core.CHECKPOINTS),
        "max_new_tokens": core.MAX_NEW_TOKENS,
        "batch_size": core.BATCH_SIZE,
        "model_name": core.MODEL_NAME,
        "model_class": type(model).__name__,
        "model_dtype": str(next(model.parameters()).dtype),
        "device_type": device.type,
        "tokenizer_class": type(tok).__name__,
        "tokenizer_length": len(tok),
        "eos_token_ids": eos_ids,
        "pad_token_id": pad_id,
        "think_open_id": think_open,
        "think_close_id": think_close,
        "runner_sha256": boundary.sha256_file(Path(__file__).resolve()),
        "boundary_core_sha256": boundary.sha256_file(
            GLM5 / "phase979_boundary_core.py"),
        "script_seals": json.loads(json.dumps(protocol["script_seals"])),
        "dependency_seals": json.loads(json.dumps(protocol["dependency_seals"])),
        "phase979_script_hashes": json.loads(json.dumps(
            protocol["phase979_source"]["phase979_script_hashes"])),
        "script_seals_sha256": core.canonical_sha256(protocol["script_seals"]),
        "dependency_seals_sha256": core.canonical_sha256(
            protocol["dependency_seals"]),
        "phase979_script_hashes_sha256": core.canonical_sha256(
            protocol["phase979_source"]["phase979_script_hashes"]),
        "generation_contract": json.loads(json.dumps(core.GENERATION_CONTRACT)),
        "model_weights_loaded": True,
        "gpu_used": True,
        "generation_performed": False,
        "holdout": False, "holdout_loaded": False,
        "mechanism": False, "mechanism_authorized": False,
    }
    core.verify_manifest_dependency_contract(payload, protocol)
    return {**payload, "manifest_sha256": boundary.sha256_json(payload),
            "created_at_utc": boundary.utc_now()}


def install_manifest(document: dict[str, Any]) -> dict[str, Any]:
    verify_self_hash(document, "manifest_sha256", "created_at_utc", "new manifest")
    if core.MANIFEST_PATH.exists():
        prior = boundary.load_json(core.MANIFEST_PATH, "existing manifest")
        verify_self_hash(prior, "manifest_sha256", "created_at_utc", "existing manifest")
        require(prior["manifest_sha256"] == document["manifest_sha256"],
                "existing manifest differs")
        return prior
    boundary.atomic_write_json(core.MANIFEST_PATH, document)
    return document


def repair_truncated_tail(path: Path) -> None:
    if not path.exists():
        return
    payload = path.read_bytes()
    if not payload or payload.endswith(b"\n"):
        return
    boundary_index = payload.rfind(b"\n")
    prefix = payload[:boundary_index + 1] if boundary_index >= 0 else b""
    tail = payload[boundary_index + 1:]
    try:
        json.loads(tail.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        with path.open("r+b") as handle:
            handle.truncate(len(prefix))
            handle.flush()
            os.fsync(handle.fileno())
        return
    raise RuntimeError("rows JSONL has a complete final object without newline")


def build_row(
    manifest: dict[str, Any], tok, item: dict[str, Any], arm: str, stream: int,
    batch_index: int, input_ids: list[int], user_prompt: str,
    generated_ids: list[int], eos_ids: list[int], think_open: int, think_close: int,
) -> dict[str, Any]:
    rendered_user, rendered, expected_ids = core.render_prefix(tok, item, arm)
    require(rendered_user == user_prompt and expected_ids == input_ids,
            "row prefix changed within batch")
    trimmed = boundary.trim_at_first_eos(generated_ids, eos_ids)
    require(trimmed == generated_ids, "row contains tokens after EOS")
    identity_sha = manifest["dataset_identity"]["identity_sha256"]
    pair_seed = core.stable_pair_seed(identity_sha, item["id"], stream)
    checkpoints = core.analyze_checkpoints(
        tok, item, arm, generated_ids, eos_ids, think_open, think_close)
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE,
        "experiment": core.EXPERIMENT,
        "protocol_sha256": manifest["protocol_sha256"],
        "admission_sha256": manifest["admission_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "id": item["id"], "task": item["task"],
        "difficulty": item["difficulty"], "prompt": item["prompt"],
        "answer": item["answer"],
        "arm": arm, "arm_spec": core.ARMS[arm], "stream": stream,
        "pair_id": core.pair_id(item["id"], stream),
        "pair_seed": pair_seed,
        "sampling": core.SAMPLING,
        "batch_index": batch_index,
        "effective_user_prompt": user_prompt,
        "rendered_prefix_sha256": boundary.sha256_json(rendered),
        "input_ids": input_ids, "prompt_len": len(input_ids),
        "generated_ids": generated_ids,
        "generated_raw": tok.decode(generated_ids, skip_special_tokens=False),
        "generated_plain": tok.decode(generated_ids, skip_special_tokens=True).strip(),
        "checkpoints": checkpoints,
        "max_new_tokens": core.MAX_NEW_TOKENS,
        "first_actual_eos_absorbing": True,
        "single_rollout_prefix_checkpoints": True,
        "private_generator_per_row": True,
        "same_pair_seed_across_arms": True,
        "generation_performed": True,
        "holdout": False, "holdout_loaded": False,
        "mechanism": False, "mechanism_authorized": False,
    }
    return {**payload, "row_sha256": boundary.sha256_json(payload)}


def validate_row(
    row: dict[str, Any], manifest: dict[str, Any], tok,
    item_by_id: dict[str, dict[str, Any]], eos_ids: list[int],
    think_open: int, think_close: int, grid_positions: dict[tuple[str, str, int], int],
) -> None:
    require(row.get("row_sha256")
            == boundary.sha256_json(boundary.without_fields(row, "row_sha256")),
            "row self-hash invalid")
    key = core.row_key(row)
    require(key in grid_positions, f"row outside frozen grid: {key}")
    item = item_by_id[key[0]]
    arm, stream = key[1], key[2]
    user, rendered, ids = core.render_prefix(tok, item, arm)
    generated = row.get("generated_ids")
    require(isinstance(generated, list) and generated
            and all(isinstance(value, int) and not isinstance(value, bool)
                    for value in generated), f"invalid generated IDs: {key}")
    eos_positions = boundary.positions_of(generated, set(eos_ids))
    require((len(eos_positions) == 1 and eos_positions[0] == len(generated) - 1)
            or (not eos_positions and len(generated) == core.MAX_NEW_TOKENS),
            f"invalid termination: {key}")
    recomputed = core.analyze_checkpoints(
        tok, item, arm, generated, eos_ids, think_open, think_close)
    identity_sha = manifest["dataset_identity"]["identity_sha256"]
    require(
        row.get("phase") == core.PHASE
        and row.get("protocol_sha256") == manifest["protocol_sha256"]
        and row.get("admission_sha256") == manifest["admission_sha256"]
        and row.get("manifest_sha256") == manifest["manifest_sha256"]
        and row.get("task") == item["task"]
        and row.get("difficulty") == item["difficulty"]
        and row.get("prompt") == item["prompt"]
        and row.get("answer") == item["answer"]
        and row.get("arm_spec") == core.ARMS[arm]
        and row.get("pair_id") == core.pair_id(item["id"], stream)
        and row.get("pair_seed") == core.stable_pair_seed(identity_sha, item["id"], stream)
        and row.get("sampling") == core.SAMPLING
        and row.get("batch_index") == grid_positions[key] // core.BATCH_SIZE + 1
        and row.get("effective_user_prompt") == user
        and row.get("rendered_prefix_sha256") == boundary.sha256_json(rendered)
        and row.get("input_ids") == ids
        and row.get("prompt_len") == len(ids)
        and row.get("checkpoints") == recomputed
        and row.get("holdout") is False
        and row.get("mechanism") is False,
        f"row metadata/derivation mismatch: {key}",
    )
    core.verify_row_generation_contract(row)


def load_rows(
    manifest: dict[str, Any], tok, items: list[dict[str, Any]],
    eos_ids: list[int], think_open: int, think_close: int,
    grid_positions: dict[tuple[str, str, int], int],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    repair_truncated_tail(core.ROWS_PATH)
    if not core.ROWS_PATH.exists():
        return {}
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    item_by_id = {str(item["id"]): item for item in items}
    for line_number, raw in enumerate(core.ROWS_PATH.read_bytes().splitlines(), 1):
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"malformed row {line_number}") from exc
        validate_row(row, manifest, tok, item_by_id, eos_ids,
                     think_open, think_close, grid_positions)
        key = core.row_key(row)
        require(key not in records, f"duplicate row key: {key}")
        records[key] = row
    return records


def write_status(
    manifest: dict[str, Any], rows: dict[tuple[str, str, int], dict[str, Any]],
    elapsed: float, complete: bool,
) -> dict[str, Any]:
    counts = Counter((key[1], key[2]) for key in rows)
    payload = {
        "schema_version": core.SCHEMA_VERSION,
        "phase": core.PHASE, "experiment": core.EXPERIMENT,
        "protocol_sha256": manifest["protocol_sha256"],
        "admission_sha256": manifest["admission_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "expected_rows": core.EXPECTED_ROWS,
        "completed_rows": len(rows),
        "completed_by_arm_stream": {
            f"{arm}|stream_{stream}": counts[(arm, stream)]
            for stream in core.STREAMS for arm in core.ARMS
        },
        "complete": bool(complete),
        "elapsed_seconds_this_invocation": float(elapsed),
        "generation_performed": bool(rows),
        "model_weights_loaded": True,
        "decision_computed": False,
        "holdout": False, "holdout_loaded": False,
        "mechanism": False, "mechanism_authorized": False,
    }
    document = {**payload, "status_sha256": boundary.sha256_json(payload),
                "updated_at_utc": boundary.utc_now()}
    boundary.atomic_write_json(core.STATUS_PATH, document)
    return document


def acquire_lock(protocol_sha: str) -> int:
    core.OUT.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(core.RUN_LOCK_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                             0o600)
    except FileExistsError as exc:
        raise RuntimeError("Phase981 runner lock exists") from exc
    payload = (json.dumps({"pid": os.getpid(), "protocol_sha256": protocol_sha},
                          sort_keys=True) + "\n").encode()
    os.write(descriptor, payload)
    os.fsync(descriptor)
    return descriptor


def release_lock(descriptor: int) -> None:
    os.close(descriptor)
    core.RUN_LOCK_PATH.unlink(missing_ok=True)


def run() -> None:
    assert_no_holdout_import()
    protocol, admission, items, data_audit = authenticate_protocol_and_admission()
    require(torch.cuda.is_available(), "formal Phase981 generation requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    require(not core.AUDIT_PATH.exists(),
            "independent audit already exists; refusing generation/resume")
    lock = acquire_lock(protocol["protocol_sha256"])
    t0 = time.time()
    model = None
    try:
        model, tok, device_value = load_model(core.MODEL_NAME)
        device = torch.device(device_value)
        require(device.type == "cuda", f"Qwen did not load on CUDA: {device}")
        tok.padding_side = "left"
        eos_ids = get_eos_ids(model, tok)
        require(tok.pad_token_id is not None, "tokenizer lacks pad token")
        pad_id = int(tok.pad_token_id)
        think_open = boundary.single_token_id(tok, "<think>")
        think_close = boundary.single_token_id(tok, "</think>")
        a_id = boundary.single_token_id(tok, "A")
        b_id = boundary.single_token_id(tok, "B")
        token_identity = core.verify_protocol_token_identity(
            protocol.get("tokenizer_audit"), tok.eos_token_id,
            think_open, think_close, a_id, b_id)
        require(eos_ids == token_identity["effective_eos_token_ids"],
                "runtime EOS registry differs from frozen independent identity")
        manifest = install_manifest(make_manifest(
            protocol, admission, data_audit, model, tok, device,
            eos_ids, pad_id, think_open, think_close))
        grid = core.canonical_grid(items)
        grid_positions = {
            (str(item["id"]), arm, stream): index
            for index, (item, arm, stream) in enumerate(grid)
        }
        rows = load_rows(manifest, tok, items, eos_ids,
                         think_open, think_close, grid_positions)
        require(set(rows).issubset(set(grid_positions)), "rows outside frozen grid")
        write_status(manifest, rows, time.time() - t0,
                     len(rows) == core.EXPECTED_ROWS)
        identity_sha = manifest["dataset_identity"]["identity_sha256"]
        item_by_id = {str(item["id"]): item for item in items}
        total_batches = core.EXPECTED_ROWS // core.BATCH_SIZE
        for batch_index, jobs in enumerate(core.chunks(grid), 1):
            keys = [(str(item["id"]), arm, stream) for item, arm, stream in jobs]
            if all(key in rows for key in keys):
                continue
            prompt_ids, generated, users = sample_batch(
                model, tok, device, jobs, identity_sha, eos_ids, pad_id)
            built = [build_row(
                manifest, tok, item, arm, stream, batch_index,
                prompt_ids[index], users[index], generated[index],
                eos_ids, think_open, think_close,
            ) for index, (item, arm, stream) in enumerate(jobs)]
            for row in built:
                validate_row(row, manifest, tok, item_by_id, eos_ids,
                             think_open, think_close, grid_positions)
                key = core.row_key(row)
                if key in rows:
                    require(rows[key]["row_sha256"] == row["row_sha256"],
                            f"partial batch replay changed row: {key}")
                else:
                    boundary.append_jsonl(core.ROWS_PATH, row)
                    rows[key] = row
            complete = len(rows) == core.EXPECTED_ROWS
            write_status(manifest, rows, time.time() - t0, complete)
            if batch_index % 8 == 0 or complete:
                log(f"batches={batch_index}/{total_batches}; rows={len(rows)}/{core.EXPECTED_ROWS}")
            assert_no_holdout_import()
        require(set(rows) == set(grid_positions) and len(rows) == core.EXPECTED_ROWS,
                "Phase981 did not complete the frozen grid")
        status = write_status(manifest, rows, time.time() - t0, True)
        verify_self_hash(status, "status_sha256", "updated_at_utc", "final status")
        log(f"complete: {core.EXPECTED_ROWS} rows; elapsed={(time.time()-t0)/3600:.2f}h")
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        release_lock(lock)
        assert_no_holdout_import()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true",
                        help="CPU-only authentication; no model load and no writes")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(), ensure_ascii=False, indent=2))
        return
    run()


if __name__ == "__main__":
    main()
