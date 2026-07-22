from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import platform
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import phase983_cross_model_engine as engine
import phase992_delayed_binding_runner as p992_runner


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = Path(__file__).resolve()
PROTOCOL_ROOT = ROOT / "tests/glm5/result/phase994_external_failure_localization_protocol"
ACTIVATION_PATH = PROTOCOL_ROOT / "activation.json"
EXECUTION_ROOT = ROOT / "tests/glm5/result/phase994_external_failure_localization_execution"
MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
SCOPES = {"engineering": 16, "public": 3072}
PHASE = 994
EXPERIMENT = "external_interface_depth_budget_localization"
MAX_NEW_TOKENS = 64
BATCH_SIZE = 8
MIN_FREE_DISK_GIB = 80


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def runtime_identity() -> dict[str, Any]:
    distributions = ("torch", "transformers", "bitsandbytes", "accelerate", "tokenizers")
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "distributions": {name: importlib.metadata.version(name) for name in distributions},
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON is not an object: {path}")
    return value


def verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    unsigned = dict(value)
    expected = unsigned.pop(field, None)
    require(isinstance(expected, str) and expected == sha256_json(unsigned), f"{label} self-hash mismatch")


def sealed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result[field] = sha256_json(result)
    return result


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if path.exists():
            path.unlink()
        raise


def file_seal(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(base)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_activation(require_formal_python: bool = True) -> dict[str, Any]:
    activation = load_json(ACTIVATION_PATH)
    verify_self_hash(activation, "activation_sha256", "Phase994 activation")
    require(activation.get("phase") == PHASE and activation.get("gpu_execution_authorized") is True,
            "Phase994 execution is not authorized")
    require(tuple(activation.get("model_order", ())) == MODEL_ORDER, "model order drift")
    require(activation.get("max_new_tokens") == MAX_NEW_TOKENS
            and activation.get("batch_size") == BATCH_SIZE, "generation contract drift")
    require(activation.get("internal_trace_authorized") is False
            and activation.get("phase992_holdout_authorized") is False, "scope authority drift")
    if require_formal_python:
        require(Path(sys.executable).resolve() == Path(str(activation["formal_python"])).resolve(),
                "formal Python executable drift")
    require(runtime_identity() == activation.get("runtime_identity"), "formal runtime package identity drift")
    protocol_seal = activation["protocol"]
    protocol_path = PROTOCOL_ROOT / protocol_seal["path"]
    require(
        protocol_path.is_file() and protocol_path.stat().st_size == protocol_seal["bytes"]
        and sha256_file(protocol_path) == protocol_seal["sha256"],
        "protocol preregistration artifact drift",
    )
    preregistration = load_json(protocol_path)
    verify_self_hash(preregistration, "protocol_sha256", "Phase994 preregistration")
    require(
        preregistration["protocol_sha256"] == activation["protocol_self_sha256"]
        and preregistration["thresholds"] == activation["thresholds"],
        "protocol/activation binding drift",
    )
    for role, seal in activation["source_seals"].items():
        path = ROOT / seal["path"]
        require(path.is_file() and path.stat().st_size == seal["bytes"]
                and sha256_file(path) == seal["sha256"], f"source seal drift: {role}")
    for name, seal in activation["dataset_seals"].items():
        path = PROTOCOL_ROOT / seal["path"]
        require(path.is_file() and path.stat().st_size == seal["bytes"]
                and sha256_file(path) == seal["sha256"], f"dataset seal drift: {name}")
    return activation


def read_manifest(path: Path, scope: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            require(isinstance(row, dict), "manifest row is not an object")
            prompt = str(row.get("prompt", ""))
            require(row.get("phase") == PHASE and prompt
                    and row.get("prompt_sha256") == sha256_bytes(prompt.encode("utf-8")),
                    "manifest prompt identity drift")
            require(row.get("serialization") in ("raw_text", "native_default_chat")
                    and row.get("depth") in ("copy_control", "one_hop", "two_hop"),
                    "manifest factor drift")
            forbidden = {"gold", "gold_value", "gold_object", "foil", "target"}
            require(not (forbidden & set(row)), "truth field leaked into runner manifest")
            rows.append(row)
    require(len(rows) == SCOPES[scope] and len({row["record_id"] for row in rows}) == len(rows),
            f"{scope} manifest count/identity drift")
    return rows


def token_ids(tokenizer: Any, text: str) -> list[int]:
    ids = list(tokenizer(text, add_special_tokens=False, return_attention_mask=False).input_ids)
    require(ids and all(isinstance(value, int) and value >= 0 for value in ids), "invalid tokenizer IDs")
    return [int(value) for value in ids]


def render_input(adapter: Any, row: Mapping[str, Any]) -> tuple[str, list[int]]:
    prompt = str(row["prompt"])
    if row["serialization"] == "raw_text":
        rendered = prompt
        ids = token_ids(adapter.tokenizer, rendered)
    else:
        prefix = adapter.render_user(prompt)
        rendered = prefix.rendered_text
        ids = list(prefix.input_ids)
        require(prefix.rendered_sha256 == sha256_bytes(rendered.encode("utf-8")), "native render hash drift")
    return rendered, ids


def left_pad(torch: Any, sequences: Sequence[Sequence[int]], pad: int, device: Any) -> tuple[Any, Any]:
    require(sequences and all(sequences), "cannot pad empty sequences")
    width = max(len(row) for row in sequences)
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    for index, row in enumerate(sequences):
        values = torch.tensor(row, dtype=torch.long, device=device)
        ids[index, -len(row):] = values
        mask[index, -len(row):] = 1
    return ids, mask


def scientific_rows(
    adapter: Any, torch: Any, prompts: Sequence[Mapping[str, Any]], scope: str,
    run_id: str, activation: Mapping[str, Any], manifest_sha256: str,
) -> Iterator[dict[str, Any]]:
    tokenizer = adapter.tokenizer
    tokenizer.padding_side = "left"
    require(tokenizer.padding_side == "left", "tokenizer left-padding drift")
    effective_eos = sorted(int(value) for value in adapter.eos_identity["effective_eos_token_ids"])
    require(effective_eos, "effective EOS union is empty")
    pad = int(adapter.pad_token_id)
    serialization_index = {value: index for index, value in enumerate(("raw_text", "native_default_chat"))}
    depth_index = {value: index for index, value in enumerate(("copy_control", "one_hop", "two_hop"))}
    transform_index = {
        value: index for index, value in enumerate(("original", "value_swap", "binding_swap", "relation_swap"))
    }
    ordered = sorted(
        prompts,
        key=lambda row: (
            serialization_index[str(row["serialization"])],
            depth_index[str(row["depth"])],
            int(row["world_ordinal"]),
            transform_index[str(row["semantic_transform"])],
        ),
    )
    for start in range(0, len(ordered), BATCH_SIZE):
        batch = ordered[start : start + BATCH_SIZE]
        require(len(batch) == BATCH_SIZE, "formal batch is not full")
        if scope == "public":
            require(
                len({str(row["serialization"]) for row in batch}) == 1
                and len({str(row["depth"]) for row in batch}) == 1,
                "public batch crossed a frozen factor-cell boundary",
            )
            factor_local_batch_index: int | None = (start % 512) // BATCH_SIZE
        else:
            factor_local_batch_index = None
        batch_units = [
            [int(row["world_ordinal"]), str(row["semantic_transform"])] for row in batch
        ]
        batch_units_sha256 = sha256_json(batch_units)
        rendered_and_ids = [render_input(adapter, row) for row in batch]
        rendered_texts = [item[0] for item in rendered_and_ids]
        raw_ids = [item[1] for item in rendered_and_ids]
        input_ids, attention_mask = left_pad(torch, raw_ids, pad, adapter.input_device)
        with torch.inference_mode():
            generated = adapter.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                use_cache=True,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=pad,
                eos_token_id=effective_eos,
                return_dict_in_generate=True,
                output_scores=False,
                output_attentions=False,
                output_hidden_states=False,
            )
        suffix_tensor = generated.sequences[:, input_ids.shape[1] :]
        suffixes = [[int(value) for value in row] for row in suffix_tensor.detach().cpu().tolist()]
        del generated, suffix_tensor, input_ids, attention_mask
        padded_input_width = max(len(ids) for ids in raw_ids)
        for batch_position, (row, rendered, ids, suffix) in enumerate(
            zip(batch, rendered_texts, raw_ids, suffixes, strict=True)
        ):
            first_eos = next((index for index, value in enumerate(suffix) if value in effective_eos), None)
            before = suffix if first_eos is None else suffix[:first_eos]
            if first_eos is None:
                require(len(suffix) == MAX_NEW_TOKENS, "generation ended without EOS before max budget")
            text = tokenizer.decode(before, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            yield {
                "schema_version": "phase994_external_localization_raw.v1",
                "phase": PHASE,
                "experiment": EXPERIMENT,
                "scope": scope,
                "model": adapter.model_key,
                "model_order_index": MODEL_ORDER.index(adapter.model_key),
                "run_id": run_id,
                "record_id": row["record_id"],
                "paired_cell_id": row["paired_cell_id"],
                "semantic_world_id": row["semantic_world_id"],
                "world_ordinal": row["world_ordinal"],
                "nuisance_id": row["nuisance_id"],
                "split": row["split"],
                "semantic_transform": row["semantic_transform"],
                "paraphrase_id": row["paraphrase_id"],
                "fact_order_id": row["fact_order_id"],
                "horizon_id": row["horizon_id"],
                "depth": row["depth"],
                "serialization": row["serialization"],
                "prompt_sha256": row["prompt_sha256"],
                "rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
                "input_manifest_sha256": manifest_sha256,
                "input_token_ids": ids,
                "input_token_ids_sha256": sha256_json(ids),
                "input_token_count": len(ids),
                "execution_batch_index": start // BATCH_SIZE,
                "factor_local_batch_index": factor_local_batch_index,
                "batch_position": batch_position,
                "batch_unit_members_sha256": batch_units_sha256,
                "padded_input_width": padded_input_width,
                "generated_suffix_token_ids": suffix,
                "generated_token_ids_before_eos": before,
                "generated_text": text,
                "effective_eos_token_ids": effective_eos,
                "first_eos_index": first_eos,
                "first_eos_token_id": None if first_eos is None else suffix[first_eos],
                "eos_seen": first_eos is not None,
                "budget_exhausted_64": first_eos is None,
                "termination_reason": "effective_eos" if first_eos is not None else "max_new_tokens_64",
                "activation_sha256": activation["activation_sha256"],
            }


def gzip_rows_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"raw output already exists before streaming: {path}")
    descriptor, pending_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
    pending = Path(pending_name)
    count = 0
    canonical_digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as target:
            with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
                for row in rows:
                    line = canonical_bytes(dict(row))
                    compressed.write(line)
                    canonical_digest.update(line)
                    count += 1
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(pending, path)
        except FileExistsError as exc:
            raise RuntimeError(f"raw output appeared during exclusive commit: {path}") from exc
        pending.unlink()
    except BaseException:
        if pending.exists():
            pending.unlink()
        raise
    return count, canonical_digest.hexdigest()


def worker(scope: str, model: str, manifest_path: Path, raw_path: Path, status_path: Path, run_id: str) -> dict[str, Any]:
    activation = verify_activation()
    require(scope in SCOPES and model in MODEL_ORDER, "worker scope/model drift")
    manifest_path = manifest_path.resolve(strict=True)
    prompts = read_manifest(manifest_path, scope)
    manifest = file_seal(manifest_path, EXECUTION_ROOT)
    expected_manifest_name = "engineering_manifest.jsonl" if scope == "engineering" else "public_manifest.jsonl"
    expected_manifest_path = (EXECUTION_ROOT / "manifests" / expected_manifest_name).resolve(strict=True)
    expected_manifest = activation["dataset_seals"][expected_manifest_name]
    require(
        manifest_path == expected_manifest_path
        and manifest["bytes"] == expected_manifest["bytes"]
        and manifest["sha256"] == expected_manifest["sha256"],
        "worker manifest differs from the frozen scope manifest",
    )
    require(raw_path.resolve(strict=False).is_relative_to(EXECUTION_ROOT.resolve()), "raw path escaped execution root")
    require(status_path.resolve(strict=False).is_relative_to(EXECUTION_ROOT.resolve()), "status path escaped execution root")
    require(not raw_path.exists() and not status_path.exists(), "worker output already exists before model load")
    import torch

    adapter = None
    loaded_identity: dict[str, Any] | None = None
    repeat_exact: bool | None = None
    raw_canonical_sha: str | None = None
    artifact_verification = p992_runner.verify_model_artifacts(activation, model)
    release_report: dict[str, Any] | None = None
    started = utc_now()
    try:
        adapter = engine.load_model_adapter(model)
        p992_runner.validate_loaded_identity(adapter.identity, model)
        loaded_artifact_root = Path(
            str(adapter.identity["artifact_identity"]["local_dir"])
        ).resolve(strict=True)
        verified_artifact_root = Path(
            str(artifact_verification["resolved_root"])
        ).resolve(strict=True)
        require(
            loaded_artifact_root == verified_artifact_root,
            "loaded model root differs from the fully hashed artifact root",
        )
        loaded_identity = deepcopy(adapter.identity)
        if scope == "engineering":
            first = list(scientific_rows(adapter, torch, prompts, scope, run_id, activation, manifest["sha256"]))
            second = list(scientific_rows(adapter, torch, prompts, scope, run_id, activation, manifest["sha256"]))
            repeat_exact = canonical_bytes(first) == canonical_bytes(second)
            require(repeat_exact, "engineering deterministic repeat mismatch")
            count, raw_canonical_sha = gzip_rows_exclusive(raw_path, first)
        else:
            count, raw_canonical_sha = gzip_rows_exclusive(
                raw_path, scientific_rows(adapter, torch, prompts, scope, run_id, activation, manifest["sha256"])
            )
        require(count == SCOPES[scope], "worker raw row count drift")
    finally:
        release_report = p992_runner.strict_cuda_release(engine, adapter, torch)
        adapter = None
    require(release_report is not None and release_report.get("cleanup_pass") is True,
            "strict CUDA cleanup did not pass")
    require(loaded_identity is not None and raw_canonical_sha is not None, "worker did not finish")
    raw = file_seal(raw_path, EXECUTION_ROOT)
    status = sealed({
        "schema_version": "phase994_external_localization_worker_status.v1",
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "created_at_utc": utc_now(),
        "started_at_utc": started,
        "scope": scope,
        "model": model,
        "model_order_index": MODEL_ORDER.index(model),
        "run_id": run_id,
        "status": "success",
        "activation_sha256": activation["activation_sha256"],
        "runner_source_sha256": sha256_file(RUNNER_PATH),
        "engine_source_sha256": sha256_file(ROOT / activation["source_seals"]["phase983_engine"]["path"]),
        "input_manifest": manifest,
        "raw_artifact": raw,
        "raw_row_count": SCOPES[scope],
        "raw_canonical_lines_sha256": raw_canonical_sha,
        "record_ids_sha256": sha256_json(sorted(row["record_id"] for row in prompts)),
        "model_artifact_verification": artifact_verification,
        "loaded_artifact_resolved_root": str(loaded_artifact_root),
        "loaded_model_identity": loaded_identity,
        "engineering_repeat_exact": repeat_exact,
        "model_released": True,
        "strict_cuda_release": release_report,
        "cuda_allocated_after": int(release_report["allocated_after_release"]),
        "cuda_reserved_after": int(release_report["reserved_after_release"]),
        "truth_opened": False,
        "internal_trace_authorized": False,
    }, "worker_status_sha256")
    write_exclusive(status_path, canonical_bytes(status))
    return {
        "passed": True,
        "scope": scope,
        "model": model,
        "row_count": SCOPES[scope],
        "worker_status_sha256": status["worker_status_sha256"],
        "truth_opened": False,
    }


def gpu_baseline() -> dict[str, Any]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    require(completed.returncode == 0, f"nvidia-smi failed: {completed.stderr[-500:]}")
    devices = []
    for line in completed.stdout.splitlines():
        if line.strip():
            left, right = [part.strip() for part in line.split(",", 1)]
            devices.append({"index": int(left), "used_mib": int(right)})
    require(devices, "no CUDA device reported")
    return {"devices": devices, "used_mib_total": sum(item["used_mib"] for item in devices)}


def resource_preflight() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    require(usage.free >= MIN_FREE_DISK_GIB * 1024**3, "free disk below frozen minimum")
    return {"disk_free_bytes": usage.free, "gpu": gpu_baseline()}


def create_lease(run_id: str, scope: str) -> tuple[Path, str]:
    path = EXECUTION_ROOT / "execution.lease.json"
    lease = sealed({
        "schema_version": "phase994_execution_lease.v1",
        "run_id": run_id,
        "scope": scope,
        "pid": os.getpid(),
        "process_instance": uuid.uuid4().hex,
        "wall_clock_ns": time.time_ns(),
        "created_at_utc": utc_now(),
    }, "lease_sha256")
    write_exclusive(path, canonical_bytes(lease))
    return path, str(lease["lease_sha256"])


def release_lease(path: Path, expected: str) -> None:
    value = load_json(path)
    verify_self_hash(value, "lease_sha256", "execution lease")
    require(value["lease_sha256"] == expected, "lease ownership drift")
    path.unlink()


def copy_manifest(scope: str) -> Path:
    source_name = "engineering_manifest.jsonl" if scope == "engineering" else "public_manifest.jsonl"
    source = PROTOCOL_ROOT / "dataset" / source_name
    target = EXECUTION_ROOT / "manifests" / source_name
    write_exclusive(target, source.read_bytes())
    activation = verify_activation()
    expected = activation["dataset_seals"][source_name]
    require(target.stat().st_size == expected["bytes"] and sha256_file(target) == expected["sha256"],
            "copied manifest seal drift")
    return target


def run_child(
    activation: Mapping[str, Any], scope: str, model: str, manifest: Path,
    raw_path: Path, status_path: Path, run_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(not raw_path.exists() and not status_path.exists(), "child output exists before resource preflight")
    before = resource_preflight()
    command = [
        str(Path(str(activation["formal_python"])).resolve()), "-B", str(RUNNER_PATH),
        "--worker", "--scope", scope, "--model", model, "--manifest", str(manifest),
        "--raw-output", str(raw_path), "--status-output", str(status_path), "--run-id", run_id,
    ]
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)
    if completed.returncode != 0:
        failure = sealed({
            "schema_version": "phase994_worker_failure_receipt.v1",
            "phase": PHASE,
            "scope": scope,
            "model": model,
            "run_id": run_id,
            "created_at_utc": utc_now(),
            "returncode": completed.returncode,
            "stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
            "stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
            "stderr_tail": completed.stderr[-4000:],
            "truth_opened": False,
        }, "failure_sha256")
        write_exclusive(EXECUTION_ROOT / "failures" / f"{scope}_{model}.json", canonical_bytes(failure))
        raise RuntimeError(f"{scope}/{model} worker failed: {completed.stderr[-4000:]}")
    child = json.loads(completed.stdout)
    require(child.get("passed") is True, "worker stdout report did not pass")
    status = load_json(status_path)
    verify_self_hash(status, "worker_status_sha256", "worker status")
    require(status.get("scope") == scope and status.get("model") == model
            and status.get("run_id") == run_id and status.get("status") == "success",
            "worker status identity drift")
    after = gpu_baseline()
    recovered = after["used_mib_total"] <= before["gpu"]["used_mib_total"] + 512
    require(recovered and status.get("cuda_allocated_after") == 0
            and status.get("cuda_reserved_after") == 0, "GPU baseline/allocator did not recover")
    cleanup = {
        "baseline_before": before,
        "baseline_after": {"gpu": after},
        "baseline_recovered": recovered,
        "model_released": status.get("model_released") is True,
        "cuda_allocated_zero": status.get("cuda_allocated_after") == 0,
        "cuda_reserved_zero": status.get("cuda_reserved_after") == 0,
        "child_exit_zero": True,
        "stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
        "stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
        "stderr_tail": completed.stderr[-2000:],
        "cleanup_pass": True,
    }
    return status, cleanup


def receipts(
    activation: Mapping[str, Any], status: Mapping[str, Any], cleanup: Mapping[str, Any],
    scope: str, model: str, run_id: str, previous: str | None, status_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    execution = sealed({
        "schema_version": "phase994_execution_receipt.v1",
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "created_at_utc": utc_now(),
        "scope": scope,
        "model": model,
        "model_order_index": MODEL_ORDER.index(model),
        "run_id": run_id,
        "status": "sealed",
        "execution_status": "success",
        "previous_model_receipt_sha256": previous,
        "activation_sha256": activation["activation_sha256"],
        "worker_status_sha256": status["worker_status_sha256"],
        "worker_status_artifact": file_seal(status_path, EXECUTION_ROOT),
        "raw_artifact": deepcopy(status["raw_artifact"]),
        "input_manifest": deepcopy(status["input_manifest"]),
        "row_count": status["raw_row_count"],
        "record_ids_sha256": status["record_ids_sha256"],
        "truth_opened": False,
        "internal_trace_authorized": False,
    }, "receipt_sha256")
    cleanup_receipt = sealed({
        "schema_version": "phase994_cleanup_receipt.v1",
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "created_at_utc": utc_now(),
        "scope": scope,
        "model": model,
        "model_order_index": MODEL_ORDER.index(model),
        "run_id": run_id,
        "status": "sealed",
        "activation_sha256": activation["activation_sha256"],
        "worker_status_sha256": status["worker_status_sha256"],
        **deepcopy(dict(cleanup)),
    }, "receipt_sha256")
    return execution, cleanup_receipt


def verify_engineering(activation: Mapping[str, Any]) -> dict[str, Any]:
    value = load_json(EXECUTION_ROOT / "engineering/qualification.json")
    verify_self_hash(value, "qualification_sha256", "engineering qualification")
    require(value.get("passed") is True and value.get("activation_sha256") == activation["activation_sha256"]
            and value.get("runner_source_sha256") == sha256_file(RUNNER_PATH)
            and tuple(value.get("model_order", ())) == MODEL_ORDER, "engineering gate stale")
    return value


def parent_engineering() -> dict[str, Any]:
    activation = verify_activation()
    require(not EXECUTION_ROOT.exists(), "engineering requires fresh execution root")
    EXECUTION_ROOT.mkdir(parents=True)
    run_id = f"phase994-engineering-{uuid.uuid4().hex}"
    lease_path, lease_sha = create_lease(run_id, "engineering")
    reports: dict[str, Any] = {}
    try:
        manifest = copy_manifest("engineering")
        for model in MODEL_ORDER:
            raw_path = EXECUTION_ROOT / f"engineering/raw/{model}.jsonl.gz"
            status_path = EXECUTION_ROOT / f"engineering/worker_status/{model}.json"
            status, cleanup = run_child(activation, "engineering", model, manifest, raw_path, status_path, run_id)
            require(status.get("engineering_repeat_exact") is True, f"engineering repeat failed: {model}")
            execution, cleanup_receipt = receipts(
                activation, status, cleanup, "engineering", model, run_id, None, status_path,
            )
            write_exclusive(EXECUTION_ROOT / f"engineering/receipts/{model}.json", canonical_bytes(execution))
            write_exclusive(
                EXECUTION_ROOT / f"engineering/receipts/cleanup_{model}.json", canonical_bytes(cleanup_receipt)
            )
            reports[model] = {
                "repeat_exact": True,
                "cleanup_pass": True,
                "receipt_sha256": execution["receipt_sha256"],
                "cleanup_receipt_sha256": cleanup_receipt["receipt_sha256"],
            }
        qualification = sealed({
            "schema_version": "phase994_engineering_qualification.v1",
            "phase": PHASE,
            "experiment": EXPERIMENT,
            "created_at_utc": utc_now(),
            "passed": True,
            "run_id": run_id,
            "activation_sha256": activation["activation_sha256"],
            "runner_source_sha256": sha256_file(RUNNER_PATH),
            "model_order": list(MODEL_ORDER),
            "models": reports,
            "scientific_accuracy_evidence": False,
            "truth_opened": False,
        }, "qualification_sha256")
        write_exclusive(EXECUTION_ROOT / "engineering/qualification.json", canonical_bytes(qualification))
        return {"passed": True, "run_id": run_id, "qualification_sha256": qualification["qualification_sha256"]}
    finally:
        if lease_path.exists():
            release_lease(lease_path, lease_sha)


def parent_public() -> dict[str, Any]:
    activation = verify_activation()
    require(EXECUTION_ROOT.exists(), "execution root missing")
    engineering = verify_engineering(activation)
    require(not (EXECUTION_ROOT / "receipts").exists(), "public execution already started")
    expected_absent = [
        EXECUTION_ROOT / "public_raw_stage.json",
        EXECUTION_ROOT / "manifests/public_manifest.jsonl",
        *[EXECUTION_ROOT / f"raw/public/{model}.jsonl.gz" for model in MODEL_ORDER],
        *[EXECUTION_ROOT / f"worker_status/public/{model}.json" for model in MODEL_ORDER],
    ]
    require(not any(path.exists() for path in expected_absent), "stale public artifact exists before GPU run")
    run_id = f"phase994-public-{uuid.uuid4().hex}"
    lease_path, lease_sha = create_lease(run_id, "public")
    reports: dict[str, Any] = {}
    previous: str | None = None
    try:
        manifest = copy_manifest("public")
        for model in MODEL_ORDER:
            raw_path = EXECUTION_ROOT / f"raw/public/{model}.jsonl.gz"
            status_path = EXECUTION_ROOT / f"worker_status/public/{model}.json"
            status, cleanup = run_child(activation, "public", model, manifest, raw_path, status_path, run_id)
            execution, cleanup_receipt = receipts(
                activation, status, cleanup, "public", model, run_id, previous, status_path,
            )
            write_exclusive(EXECUTION_ROOT / f"receipts/public_{model}.json", canonical_bytes(execution))
            write_exclusive(
                EXECUTION_ROOT / f"receipts/cleanup_public_{model}.json", canonical_bytes(cleanup_receipt)
            )
            previous = str(execution["receipt_sha256"])
            reports[model] = {
                "row_count": SCOPES["public"],
                "receipt_sha256": execution["receipt_sha256"],
                "cleanup_receipt_sha256": cleanup_receipt["receipt_sha256"],
                "cleanup_pass": True,
            }
        stage = sealed({
            "schema_version": "phase994_public_raw_stage.v1",
            "phase": PHASE,
            "experiment": EXPERIMENT,
            "created_at_utc": utc_now(),
            "passed": True,
            "run_id": run_id,
            "activation_sha256": activation["activation_sha256"],
            "engineering_qualification_sha256": engineering["qualification_sha256"],
            "model_order": list(MODEL_ORDER),
            "models": reports,
            "all_raw_status_and_cleanup_sealed_before_scoring": True,
            "truth_opened": False,
        }, "stage_sha256")
        write_exclusive(EXECUTION_ROOT / "public_raw_stage.json", canonical_bytes(stage))
        return {"passed": True, "run_id": run_id, "stage_sha256": stage["stage_sha256"], "models": reports,
                "truth_opened": False}
    finally:
        if lease_path.exists():
            release_lease(lease_path, lease_sha)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--engineering", action="store_true")
    mode.add_argument("--public", action="store_true")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--scope")
    parser.add_argument("--model")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--status-output", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.worker:
        require(all((args.scope, args.model, args.manifest, args.raw_output, args.status_output, args.run_id)),
                "worker arguments incomplete")
        result = worker(args.scope, args.model, args.manifest, args.raw_output, args.status_output, args.run_id)
    elif args.engineering:
        result = parent_engineering()
    else:
        result = parent_public()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
