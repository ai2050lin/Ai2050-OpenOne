#!/usr/bin/env python3
"""Frozen raw-only runner for Phase578 engineering and development behavior.

The parent process never imports torch.  It launches qwen3, GLM4, and DS7B as
strictly serial single-model subprocesses.  Development workers can read only
the truth-free Phase578 prompt manifest and persist no scores or activations.
"""

from __future__ import annotations

import argparse
import builtins
import gc
import gzip
import hashlib
import importlib.metadata
import io
import json
import msvcrt
import os
import secrets
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests/glm5"
GPT5 = ROOT / "tests/gpt5"
for item in (GLM5, GPT5):
    if str(item) in sys.path:
        sys.path.remove(str(item))
sys.path.insert(0, str(GLM5))
sys.path.insert(1, str(GPT5))

PHASE = "Phase578"
MODELS = ("qwen3", "glm4", "deepseek7b")
REPEATS = ("repeat1", "repeat2")
BATCH_SIZE = 8
MAX_NEW_TOKENS = 24
PREFIX_TOKEN_BUDGET = 8
PROTOCOL_DIR = ROOT / "tests/glm5/result/phase578_gpt5_runner_scorer_protocol"
MANIFEST_PATH = PROTOCOL_DIR / "phase578_development_prompt_manifest.jsonl"
PROTOCOL_PATH = PROTOCOL_DIR / "phase578_preregistered_runner_protocol.json"
AUDIT_PATH = PROTOCOL_DIR / "phase578_independent_audit.json"
FREEZE_PATH = PROTOCOL_DIR / "phase578_freeze_commit.json"
ENGINEERING_DIR = ROOT / "tests/glm5/result/phase578_gpt5_engineering_qualification"
DEVELOPMENT_DIR = ROOT / "tests/glm5/result/phase578_gpt5_development_behavior_raw"
FORMAL_PYTHON = Path(
    r"C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe"
)
FORMAL_PYTHON_SHA256 = (
    "0f11fb7422fa347b7609ba0964ceccef3c8fa9f15230c37b9ec27668e68e8a8a"
)
ENGINE_PATH = ROOT / "tests/glm5/phase983_cross_model_engine.py"
ENGINE_EXPECTED_SHA256 = (
    "e345daf3c3eae289eb7a71b8a741eeaf3a11c6897d009a5f9d90a386b23eef6f"
)
SOURCE_RELATIVE = "tests/glm5/phase578_gpt5_development_runner.py"
MODEL_LOGICAL_DIRS = {
    "qwen3": ROOT / "models/hf/qwen3-4b",
    "glm4": ROOT / "models/hf/glm4-9b-chat-hf",
    "deepseek7b": ROOT / "models/hf/deepseek-r1-distill-qwen-7b",
}

GENERATION_CONTRACT = {
    "batch_size": BATCH_SIZE,
    "repeats": list(REPEATS),
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "use_cache": True,
    "pad_token_id": "adapter.pad_token_id",
    "eos_token_id": "adapter.effective_eos_token_ids",
    "return_dict_in_generate": False,
    "output_scores": False,
    "output_attentions": False,
    "output_hidden_states": False,
    "tokenizer_padding_side": "left",
    "tokenizer_padding": True,
    "tokenizer_truncation": False,
    "tokenizer_add_special_tokens": False,
    "decode_skip_special_tokens": False,
    "decode_clean_up_tokenization_spaces": False,
    "qwen3_enable_thinking": False,
    "deepseek_empty_think_prefill_closed": True,
    "quantization": "bitsandbytes_int8",
    "nonquantized_dtype": "torch.bfloat16",
    "attention_implementation": "sdpa",
    "cpu_or_disk_offload": False,
    "automatic_fallback": False,
}
GENERATION_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(GENERATION_CONTRACT, sort_keys=True, separators=(",", ":"))
    .encode("utf-8")
).hexdigest()

ENGINEERING_PROMPTS = tuple(
    {
        "schema_version": "phase578_engineering_prompt.v1",
        "phase_id": PHASE,
        "split": "synthetic_engineering_only",
        "ordinal": index,
        "case_id": f"phase578_engineering_{index:02d}",
        "raw_prompt": prompt,
        "source_case_record_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }
    for index, prompt in enumerate((
        "ENGINEERING QUALIFICATION ONLY. Return exactly READY.",
        "Synthetic runtime check; answer with READY and nothing else.",
        "This is not a research case. Emit the single word READY.",
        "GPU generation-path check. The requested response is READY.",
        "Tokenizer and batching check number five: return READY.",
        "Deterministic decoding check number six: return READY.",
        "CUDA cleanup qualification item seven: return READY.",
        "Final synthetic item. Reply with READY.",
    ))
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n"
    ).encode("utf-8")


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or path.exists():
        raise RuntimeError(f"no-overwrite publication refused: {path}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def write_json(path: Path, payload: Any) -> None:
    write_exclusive(path, json_bytes(payload))


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    raw = b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)
    write_exclusive(path, gzip.compress(raw, compresslevel=6, mtime=0))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest() -> list[dict[str, Any]]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 336 or [row.get("ordinal") for row in rows] != list(range(336)):
        raise RuntimeError("Phase578 prompt manifest denominator/order drift")
    if len({row.get("case_id") for row in rows}) != 336:
        raise RuntimeError("Phase578 prompt manifest case registry drift")
    forbidden = {
        "target", "foil", "candidate_groups", "focus_object_class",
        "comparison_object_class", "target_truth_polarity",
    }
    if any(set(row) & forbidden for row in rows):
        raise RuntimeError("truth-bearing field entered GPU prompt manifest")
    return rows


def verify_bridge() -> dict[str, Any]:
    for path in (PROTOCOL_PATH, AUDIT_PATH, FREEZE_PATH, MANIFEST_PATH):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing/aliased Phase578 bridge artifact: {path}")
    if Path(sys.executable).resolve() != FORMAL_PYTHON.resolve():
        raise RuntimeError("runner invoked under a non-frozen Python interpreter")
    if sha256_file(FORMAL_PYTHON) != FORMAL_PYTHON_SHA256:
        raise RuntimeError("formal Python executable identity drift")
    if sha256_file(ENGINE_PATH) != ENGINE_EXPECTED_SHA256:
        raise RuntimeError("cross-model engine source drift")
    freeze = read_json(FREEZE_PATH)
    protocol = read_json(PROTOCOL_PATH)
    audit = read_json(AUDIT_PATH)
    if not all((
        freeze.get("freeze_complete") is True,
        freeze.get("gpu_behavior_authorized") is False,
        freeze.get("engineering_qualification_run_count") == 0,
        freeze.get("gpu_behavior_run_count") == 0,
        freeze.get("next_required_stage")
        == "phase578_separate_engineering_qualification",
        freeze.get("protocol_sha256") == sha256_file(PROTOCOL_PATH),
        freeze.get("independent_audit_sha256") == sha256_file(AUDIT_PATH),
        freeze.get("development_manifest_sha256") == sha256_file(MANIFEST_PATH),
        audit.get("passed") is True,
        all(audit.get("checks", {}).values()),
        protocol.get("generation_contract") == GENERATION_CONTRACT,
        protocol.get("models_in_required_order") == list(MODELS),
        protocol.get("gpu_behavior_authorized_by_this_protocol") is False,
        protocol.get("development_prompt_manifest", {}).get("truth_fields_present")
        is False,
    )):
        raise RuntimeError("Phase578 frozen bridge verification failed")
    for registry_name, registry in (
        ("source", freeze.get("source_identities", {})),
        ("upstream", protocol.get("upstream_identities", {})),
    ):
        if not isinstance(registry, dict) or not registry:
            raise RuntimeError(f"Phase578 {registry_name} identity registry missing")
        for name, expected in registry.items():
            if registry_name == "source" and name != SOURCE_RELATIVE:
                continue
            if registry_name == "upstream" and name not in {
                "cross_model_engine", "model_registry",
            }:
                # All historical/data evidence is already committed by the
                # Phase578 freeze.  The execution process reopens no Phase577
                # or Phase576 research artifact, including full development.
                continue
            raw_path = expected.get("path")
            if not isinstance(raw_path, str):
                raise RuntimeError(f"Phase578 {registry_name} path invalid: {name}")
            candidate = Path(raw_path)
            path = candidate if candidate.is_absolute() else ROOT / candidate
            if (
                not path.is_file() or path.is_symlink()
                or path.stat().st_size != expected.get("size_bytes")
                or sha256_file(path) != expected.get("sha256")
            ):
                raise RuntimeError(f"Phase578 {registry_name} identity drift: {name}")
    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "bitsandbytes", "accelerate")
    }
    if packages != protocol.get("formal_runtime_identity", {}).get("packages"):
        raise RuntimeError(f"Phase578 formal package drift: {packages}")
    for model, entry in protocol.get("frozen_tokenizer_input_identities", {}).items():
        if model not in MODELS:
            raise RuntimeError("Phase578 tokenizer model registry drift")
        for filename, expected in entry.get("files", {}).items():
            path = Path(expected["resolved_path"])
            if (
                not path.is_file() or path.is_symlink()
                or path.stat().st_size != expected["size_bytes"]
                or sha256_file(path) != expected["sha256"]
            ):
                raise RuntimeError(f"Phase578 tokenizer input drift: {model}/{filename}")
    legacy = protocol.get("legacy_phase578_collision", {})
    if legacy.get("status") != "excluded_not_imported_not_executed":
        raise RuntimeError("legacy Phase578 exclusion drift")
    return {
        "freeze_sha256": sha256_file(FREEZE_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "audit_sha256": sha256_file(AUDIT_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "engine_sha256": sha256_file(ENGINE_PATH),
        "formal_python_sha256": FORMAL_PYTHON_SHA256,
        "formal_packages": packages,
        "truth_bearing_upstream_reopened": False,
    }


def verify_model_artifacts(model: str, protocol: dict[str, Any]) -> dict[str, Any]:
    expected = protocol.get("frozen_model_artifact_identities", {}).get(model)
    if not isinstance(expected, dict):
        raise RuntimeError(f"missing frozen model identity: {model}")
    logical = MODEL_LOGICAL_DIRS[model]
    resolved = logical.resolve(strict=True)
    if str(resolved).casefold() != str(Path(expected["resolved_path"]).resolve(strict=True)).casefold():
        raise RuntimeError(f"{model}: resolved model path drift")
    expected_files = {
        item["relative_path"]: item
        for group in ("tokenizer_and_config_files", "weight_files")
        for item in expected.get(group, [])
    }
    actual_files: dict[str, Path] = {}
    for path in resolved.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"{model}: nested model symlink forbidden: {path}")
        if path.is_file():
            actual_files[str(path.relative_to(resolved)).replace("\\", "/")] = path
    if set(actual_files) != set(expected_files):
        raise RuntimeError(f"{model}: recursive model artifact registry drift")
    reports = []
    for relative, expected_file in sorted(expected_files.items()):
        path = actual_files[relative]
        observed_hash = sha256_file(path)
        if (
            path.stat().st_size != expected_file["size_bytes"]
            or observed_hash != expected_file["sha256"]
        ):
            raise RuntimeError(f"{model}: model artifact drift: {relative}")
        reports.append({
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": observed_hash,
        })
    payload = {
        "model": model,
        "resolved_path": str(resolved),
        "file_count": len(reports),
        "files": reports,
        "frozen_identity_sha256": expected.get("identity_sha256"),
    }
    payload["verification_payload_sha256"] = sha256_bytes(
        canonical_json(payload).encode("utf-8")
    )
    return payload


def _forbidden_research_path(
    value: Any, model: str, pending_root: Path,
) -> bool:
    if isinstance(value, int):
        return False
    try:
        candidate = Path(os.fsdecode(value)).resolve(strict=False)
    except (TypeError, ValueError, OSError):
        return False
    repo = ROOT.resolve(strict=True)
    current_model = MODEL_LOGICAL_DIRS[model].resolve(strict=True)
    all_models = [path.resolve(strict=True) for path in MODEL_LOGICAL_DIRS.values()]
    allowed_exact = {
        path.resolve(strict=False) for path in (
            MANIFEST_PATH, PROTOCOL_PATH, AUDIT_PATH, FREEZE_PATH,
            ENGINE_PATH, ROOT / "tests/gpt5/model_registry.py",
            Path(__file__).resolve(),
        )
    }
    pycache_allowed = (
        candidate.parent.name == "__pycache__"
        and any(candidate.name.startswith(prefix) for prefix in (
            "phase983_cross_model_engine.", "model_registry.",
        ))
    )
    inside_pending = candidate == pending_root or pending_root in candidate.parents
    inside_current_model = (
        candidate == current_model or current_model in candidate.parents
    )
    inside_any_model = any(
        candidate == root or root in candidate.parents for root in all_models
    )
    if inside_any_model and not inside_current_model:
        return True
    if candidate == repo or repo in candidate.parents:
        return not (
            candidate in allowed_exact or pycache_allowed
            or inside_pending or inside_current_model
        )
    return False


def install_research_access_guard(
    model: str, pending_root: Path,
) -> tuple[dict[str, int], Callable[[], None]]:
    attempts = {"forbidden_research_open_attempts": 0}
    original_builtin, original_io, original_os = builtins.open, io.open, os.open

    def guard(original: Callable[..., Any]) -> Callable[..., Any]:
        def guarded(file: Any, *args: Any, **kwargs: Any) -> Any:
            if _forbidden_research_path(file, model, pending_root):
                attempts["forbidden_research_open_attempts"] += 1
                raise RuntimeError(f"GPU worker research-path firewall: {file}")
            return original(file, *args, **kwargs)
        return guarded

    builtins.open, io.open, os.open = guard(original_builtin), guard(original_io), guard(original_os)

    def restore() -> None:
        builtins.open, io.open, os.open = original_builtin, original_io, original_os

    return attempts, restore


def render_chat(tokenizer: Any, model: str, content: str) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if model == "qwen3":
        kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], **kwargs
    )
    if model == "deepseek7b" and rendered.endswith("<think>\n"):
        rendered += "</think>\n\n"
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("chat template returned an empty prompt")
    return rendered


def strict_release(engine: Any, adapter: Any) -> dict[str, Any]:
    import torch

    base_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        engine.release_model_adapter(adapter)
    except BaseException as exc:
        base_error = exc
        exc.__traceback__ = None
    steps: dict[str, bool] = {}
    try:
        gc.collect()
        torch.cuda.synchronize()
        steps["synchronize_before_cublas_clear"] = True
        clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        if not callable(clear):
            raise RuntimeError("required cuBLAS workspace clear API is unavailable")
        clear()
        steps["cublas_workspaces_cleared"] = True
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        steps["final_allocator_cleanup"] = True
    except BaseException as exc:
        cleanup_error = exc
        exc.__traceback__ = None
    allocated = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())
    report = {
        "steps": steps,
        "allocated_after_release": allocated,
        "reserved_after_release": reserved,
        "cleanup_pass": (
            base_error is None and cleanup_error is None
            and allocated == 0 and reserved == 0
        ),
        "base_release_error": None if base_error is None else type(base_error).__name__,
        "strict_cleanup_error": (
            None if cleanup_error is None else type(cleanup_error).__name__
        ),
    }
    if base_error is not None:
        raise base_error
    if cleanup_error is not None:
        raise cleanup_error
    if allocated != 0 or reserved != 0:
        raise RuntimeError(
            f"strict CUDA cleanup retained allocated={allocated}, reserved={reserved}"
        )
    return report


def verify_worker_authorization(
    model: str, mode: str, pending_root: Path,
    authorization_path: Path, authorization_nonce: str,
) -> dict[str, Any]:
    expected_path = pending_root / f"worker_authorization_{MODELS.index(model):02d}_{model}.json"
    if authorization_path.resolve(strict=True) != expected_path.resolve(strict=True):
        raise RuntimeError("worker authorization path drift")
    authorization = read_json(expected_path)
    stage_start_path = pending_root / "stage_start.json"
    if not all((
        authorization.get("schema_version") == "phase578_worker_authorization.v1",
        authorization.get("phase_id") == PHASE,
        authorization.get("mode") == mode,
        authorization.get("model") == model,
        authorization.get("model_order_index") == MODELS.index(model),
        authorization.get("authorization_nonce") == authorization_nonce,
        secrets.compare_digest(
            str(authorization.get("authorization_nonce", "")),
            authorization_nonce,
        ),
        authorization.get("parent_pid") == os.getppid(),
        authorization.get("pending_root") == str(pending_root),
        authorization.get("runner_source_sha256")
        == sha256_file(Path(__file__).resolve()),
        stage_start_path.is_file(),
        authorization.get("stage_start_sha256") == sha256_file(stage_start_path),
    )):
        raise RuntimeError("worker authorization contract failed")
    prior = authorization.get("prior_terminal_model_statuses")
    if not isinstance(prior, list) or [item.get("model") for item in prior] != list(
        MODELS[:MODELS.index(model)]
    ):
        raise RuntimeError("worker predecessor authorization order drift")
    for item in prior:
        path = pending_root / item["relative_path"]
        status = read_json(path)
        if not all((
            sha256_file(path) == item["sha256"],
            status.get("status") in {"complete", "failed"},
            status.get("cleanup", {}).get("cleanup_pass") is True,
            status.get("cleanup", {}).get("allocated_after_release") == 0,
            status.get("cleanup", {}).get("reserved_after_release") == 0,
        )):
            raise RuntimeError("worker predecessor status is not clean/terminal")
    if mode == "development":
        if authorization.get("engineering_verification") != verify_execution(
            "engineering"
        ):
            raise RuntimeError("worker engineering authorization drift")
    elif authorization.get("engineering_verification") is not None:
        raise RuntimeError("engineering worker received recursive authorization")
    return authorization


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def generate_batch(
    adapter: Any, model: str, mode: str, repeat: str,
    batch_index: int, rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import torch

    rendered = [render_chat(adapter.tokenizer, model, row["raw_prompt"]) for row in rows]
    unpadded_ids = [
        [int(value) for value in adapter.tokenizer(
            text, add_special_tokens=False, return_attention_mask=False,
        ).input_ids]
        for text in rendered
    ]
    encoded = adapter.tokenizer(
        rendered, return_tensors="pt", padding=True, truncation=False,
        add_special_tokens=False,
    )
    prompt_width = int(encoded["input_ids"].shape[1])
    attention_lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
    if attention_lengths != [len(value) for value in unpadded_ids]:
        raise RuntimeError("batch attention mask/input length mismatch")
    for index, expected_ids in enumerate(unpadded_ids):
        mask = encoded["attention_mask"][index].to(dtype=torch.bool)
        observed_ids = [
            int(value) for value in encoded["input_ids"][index][mask].tolist()
        ]
        if observed_ids != expected_ids:
            raise RuntimeError("batch padded token matrix differs from unpadded prompt")
    encoded = {key: value.to(adapter.input_device) for key, value in encoded.items()}
    eos_ids = [int(value) for value in adapter.eos_identity["effective_eos_token_ids"]]
    with torch.inference_mode():
        generated = adapter.model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            use_cache=True,
            pad_token_id=adapter.pad_token_id,
            eos_token_id=eos_ids,
            return_dict_in_generate=False,
            output_scores=False,
            output_attentions=False,
            output_hidden_states=False,
        )
    suffixes = [
        [int(value) for value in generated[index, prompt_width:].detach().cpu().tolist()]
        for index in range(len(rows))
    ]
    del generated, encoded
    output = []
    for batch_row_index, (row, rendered_text, input_ids, suffix) in enumerate(
        zip(rows, rendered, unpadded_ids, suffixes)
    ):
        first_eos_index = next(
            (index for index, token_id in enumerate(suffix) if token_id in eos_ids), None
        )
        content_ids = suffix if first_eos_index is None else suffix[:first_eos_index]
        post_eos = [] if first_eos_index is None else suffix[first_eos_index + 1:]
        prefix_texts = [
            _decode(adapter.tokenizer, content_ids[:index])
            for index in range(1, min(PREFIX_TOKEN_BUDGET, len(content_ids)) + 1)
        ]
        token_pieces = [
            str(value) for value in adapter.tokenizer.convert_ids_to_tokens(content_ids)
        ]
        eos_seen = first_eos_index is not None
        budget = not eos_seen and len(suffix) == MAX_NEW_TOKENS
        output.append({
            "schema_version": "phase578_development_behavior_row.v1",
            "phase_id": PHASE,
            "mode": mode,
            "model": model,
            "model_order_index": MODELS.index(model),
            "split": row["split"],
            "execution_repeat": repeat,
            "case_id": row["case_id"],
            "source_case_record_sha256": row["source_case_record_sha256"],
            "batch_index": batch_index,
            "batch_row_index": batch_row_index,
            "rendered_prompt_sha256": sha256_bytes(rendered_text.encode("utf-8")),
            "input_token_ids": input_ids,
            "input_token_count": len(input_ids),
            "input_token_ids_sha256": sha256_bytes(canonical_json(input_ids).encode("utf-8")),
            "batch_padded_prompt_width": prompt_width,
            "attention_mask_valid_tokens": len(input_ids),
            "full_generated_suffix_token_ids": suffix,
            "full_generated_suffix_decode": _decode(adapter.tokenizer, suffix),
            "generated_token_ids_before_eos": content_ids,
            "generated_token_pieces_before_eos": token_pieces,
            "generated_token_count_before_eos": len(content_ids),
            "generated_text": _decode(adapter.tokenizer, content_ids),
            "prefix_text_by_generated_token": prefix_texts,
            "effective_eos_token_ids": eos_ids,
            "pad_token_id": int(adapter.pad_token_id),
            "first_eos_index": first_eos_index,
            "first_eos_token_id": (
                None if first_eos_index is None else suffix[first_eos_index]
            ),
            "post_eos_token_ids": post_eos,
            "post_eos_tokens_all_pad": all(
                value == adapter.pad_token_id for value in post_eos
            ),
            "eos_seen": eos_seen,
            "budget_truncated": budget,
            "termination_event": "eos" if eos_seen else "budget" if budget else "other",
            "generation_contract_sha256": GENERATION_CONTRACT_SHA256,
            "observer_only": True,
            "activation_collected": False,
            "hidden_states_requested": False,
            "attentions_requested": False,
            "scores_requested": False,
            "hooks_registered": 0,
            "causal_intervention": False,
            "sealed_model_access": False,
        })
    return output


def worker(
    model: str, mode: str, pending_root: Path,
    authorization_path: Path, authorization_nonce: str,
) -> int:
    bridge = verify_bridge()
    if model not in MODELS or mode not in {"engineering", "development"}:
        raise RuntimeError("invalid worker model/mode")
    pending_root = pending_root.resolve(strict=True)
    expected_parent = ENGINEERING_DIR.parent.resolve(strict=True)
    if pending_root.parent != expected_parent or not pending_root.name.startswith(
        f".{(ENGINEERING_DIR if mode == 'engineering' else DEVELOPMENT_DIR).name}.pending-"
    ):
        raise RuntimeError("worker pending root escaped the frozen result namespace")
    authorization = verify_worker_authorization(
        model, mode, pending_root, authorization_path, authorization_nonce
    )
    model_dir = pending_root / f"{MODELS.index(model):02d}_{model}"
    model_dir.mkdir(parents=False, exist_ok=False)
    attempts, restore_guard = install_research_access_guard(model, pending_root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    adapter = None
    raw_rows: list[dict[str, Any]] = []
    started = now()
    status: dict[str, Any]
    failure: BaseException | None = None
    cleanup_report: dict[str, Any] | None = None
    model_identity: dict[str, Any] | None = None
    model_artifact_verification: dict[str, Any] | None = None
    try:
        if any(name in sys.modules for name in (
            "phase578_retrieval_closure", "model_utils",
            "phase983_cross_model_engine",
        )):
            raise RuntimeError("forbidden/preloaded execution module detected")
        import torch
        import phase983_cross_model_engine as engine
        imported_engine_path = Path(engine.__file__).resolve(strict=True)
        registry_module = sys.modules.get("model_registry")
        if (
            imported_engine_path != ENGINE_PATH.resolve(strict=True)
            or sha256_file(imported_engine_path) != ENGINE_EXPECTED_SHA256
            or registry_module is None
            or Path(registry_module.__file__).resolve(strict=True)
            != (ROOT / "tests/gpt5/model_registry.py").resolve(strict=True)
        ):
            raise RuntimeError("execution engine/model registry import shadow detected")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one CUDA device is required")
        if int(torch.cuda.memory_allocated()) != 0 or int(torch.cuda.memory_reserved()) != 0:
            raise RuntimeError("dirty CUDA allocator baseline")
        torch.cuda.reset_peak_memory_stats()
        rows = list(ENGINEERING_PROMPTS) if mode == "engineering" else read_manifest()
        model_artifact_verification = verify_model_artifacts(
            model, read_json(PROTOCOL_PATH)
        )
        adapter = engine.load_model_adapter(model)
        adapter.tokenizer.padding_side = "left"
        model_identity = adapter.identity
        for repeat in REPEATS:
            for start in range(0, len(rows), BATCH_SIZE):
                batch_index = start // BATCH_SIZE
                raw_rows.extend(generate_batch(
                    adapter, model, mode, repeat, batch_index,
                    rows[start:start + BATCH_SIZE],
                ))
                done = min(start + BATCH_SIZE, len(rows))
                if batch_index == 0 or done == len(rows) or batch_index % 8 == 7:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] {mode}/{model}/{repeat} "
                        f"{done}/{len(rows)}", flush=True,
                    )
        expected_rows = len(rows) * len(REPEATS)
        if len(raw_rows) != expected_rows:
            raise RuntimeError("worker raw-row denominator drift")
    except BaseException as exc:
        failure = exc
        exc.__traceback__ = None
    finally:
        try:
            import phase983_cross_model_engine as cleanup_engine
            if Path(cleanup_engine.__file__).resolve(strict=True) != ENGINE_PATH.resolve(strict=True):
                raise RuntimeError("cleanup engine binding drift")
            cleanup_report = strict_release(cleanup_engine, adapter)
        except BaseException as cleanup_exc:
            if failure is None:
                failure = cleanup_exc
            cleanup_report = {
                "cleanup_pass": False,
                "error_type": type(cleanup_exc).__name__,
                "error": str(cleanup_exc),
            }
        adapter = None
        if failure is not None:
            failure.__traceback__ = None
        restore_guard()
    completed = failure is None and cleanup_report is not None and cleanup_report.get(
        "cleanup_pass"
    ) is True
    if completed:
        if mode == "development":
            write_jsonl_gz(model_dir / "raw_rows.jsonl.gz", raw_rows)
        else:
            repeat_hashes = {
                repeat: sha256_bytes(canonical_json([
                    row["full_generated_suffix_token_ids"] for row in raw_rows
                    if row["execution_repeat"] == repeat
                ]).encode("utf-8"))
                for repeat in REPEATS
            }
            write_json(model_dir / "engineering_generation_capsule.json", {
                "schema_version": "phase578_engineering_generation_capsule.v1",
                "phase_id": PHASE, "model": model,
                "row_count": len(raw_rows), "repeat_hashes": repeat_hashes,
                "repeat_outputs_exact": len(set(repeat_hashes.values())) == 1,
                "termination_counts": {
                    event: sum(row["termination_event"] == event for row in raw_rows)
                    for event in ("eos", "budget", "other")
                },
                "raw_rows_payload_sha256": sha256_bytes(
                    canonical_json(raw_rows).encode("utf-8")
                ),
                "activation_collected": False,
            })
            if len(set(repeat_hashes.values())) != 1:
                completed = False
                failure = RuntimeError("engineering greedy repeats were not exact")
    status = {
        "schema_version": "phase578_model_worker_status.v1",
        "phase_id": PHASE, "created_at_utc": now(), "started_at_utc": started,
        "mode": mode, "model": model, "model_order_index": MODELS.index(model),
        "status": "complete" if completed else "failed",
        "bridge_identity": bridge,
        "generation_contract": GENERATION_CONTRACT,
        "generation_contract_sha256": GENERATION_CONTRACT_SHA256,
        "raw_row_count": len(raw_rows),
        "expected_raw_row_count": (16 if mode == "engineering" else 672),
        "model_identity": model_identity,
        "model_artifact_verification": model_artifact_verification,
        "worker_authorization_sha256": sha256_file(authorization_path),
        "worker_authorization_parent_pid": authorization.get("parent_pid"),
        "cleanup": cleanup_report,
        "research_access_attempts": attempts,
        "error_type": None if failure is None else type(failure).__name__,
        "error": None if failure is None else str(failure),
        "traceback_persisted": False,
        "automatic_fallback_used": False,
        "activation_collected": False,
        "hidden_states_requested": False,
        "attentions_requested": False,
        "scores_requested": False,
        "hooks_registered": 0,
        "causal_intervention": False,
        "sealed_model_access": False,
    }
    write_json(model_dir / "status.json", status)
    print(json.dumps({
        "mode": mode, "model": model, "status": status["status"],
        "cleanup_pass": cleanup_report.get("cleanup_pass") if cleanup_report else False,
        "raw_row_count": len(raw_rows),
    }, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if completed else 2


def acquire_lease(mode: str) -> Any:
    path = ROOT / f"tests/glm5/result/.phase578_{mode}.lease"
    handle = path.open("a+b", buffering=0)
    if path.stat().st_size == 0:
        handle.write(b"0")
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"Phase578 {mode} execution is already leased") from exc
    return handle


def release_lease(handle: Any) -> None:
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


def _quarantine_pending(final_dir: Path) -> list[str]:
    quarantined = []
    prefix = f".{final_dir.name}.pending-"
    for path in sorted(final_dir.parent.iterdir()):
        if path.is_dir() and path.name.startswith(prefix):
            destination = path.with_name(
                f".{final_dir.name}.aborted-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            )
            path.rename(destination)
            quarantined.append(destination.name)
    return quarantined


def _artifact_registry(root: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("execution artifact symlink forbidden")
        if path.is_file():
            output.append({
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return output


def parent_run(mode: str) -> dict[str, Any]:
    bridge = verify_bridge()
    final_dir = ENGINEERING_DIR if mode == "engineering" else DEVELOPMENT_DIR
    if final_dir.exists():
        raise RuntimeError(f"terminal Phase578 {mode} result already exists")
    engineering_verification: dict[str, Any] | None = None
    if mode == "development":
        engineering_verification = verify_execution("engineering")
        qualification_path = ENGINEERING_DIR / "execution_receipt.json"
        if not qualification_path.is_file():
            raise RuntimeError("development requires terminal engineering qualification")
        qualification = read_json(qualification_path)
        if not all((
            qualification.get("qualification_passed") is True,
            qualification.get("development_gpu_authorized") is True,
            qualification.get("attempted_models_in_order") == list(MODELS),
        )):
            raise RuntimeError("engineering qualification did not authorize development")
    lease = acquire_lease(mode)
    pending: Path | None = None
    try:
        quarantined = _quarantine_pending(final_dir)
        pending = final_dir.with_name(
            f".{final_dir.name}.pending-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-pid{os.getpid()}"
        )
        pending.mkdir(parents=True, exist_ok=False)
        write_json(pending / "stage_start.json", {
            "schema_version": "phase578_execution_stage_start.v1",
            "phase_id": PHASE, "created_at_utc": now(), "mode": mode,
            "required_model_order": list(MODELS),
            "bridge_identity": bridge,
            "generation_contract": GENERATION_CONTRACT,
            "generation_contract_sha256": GENERATION_CONTRACT_SHA256,
            "engineering_verification": engineering_verification,
            "quarantined_prior_pending": quarantined,
            "activation_collected": False,
        })
        attempts: list[dict[str, Any]] = []
        prior_terminal_statuses: list[dict[str, Any]] = []
        fatal_cleanup = False
        for model in MODELS:
            nonce = secrets.token_hex(32)
            authorization_path = pending / (
                f"worker_authorization_{MODELS.index(model):02d}_{model}.json"
            )
            authorization = {
                "schema_version": "phase578_worker_authorization.v1",
                "phase_id": PHASE, "created_at_utc": now(), "mode": mode,
                "model": model, "model_order_index": MODELS.index(model),
                "authorization_nonce": nonce,
                "parent_pid": os.getpid(),
                "pending_root": str(pending.resolve(strict=True)),
                "runner_source_sha256": sha256_file(Path(__file__).resolve()),
                "stage_start_sha256": sha256_file(pending / "stage_start.json"),
                "prior_terminal_model_statuses": prior_terminal_statuses,
                "engineering_verification": engineering_verification,
            }
            write_json(authorization_path, authorization)
            command = [
                str(FORMAL_PYTHON), str(Path(__file__).resolve()),
                "--worker", "--mode", mode, "--model", model,
                "--pending-root", str(pending),
                "--authorization", str(authorization_path),
                "--authorization-nonce", nonce,
            ]
            started = time.time()
            process = subprocess.run(command, cwd=str(ROOT), check=False)
            status_path = pending / f"{MODELS.index(model):02d}_{model}/status.json"
            status = read_json(status_path) if status_path.is_file() else {}
            cleanup_pass = status.get("cleanup", {}).get("cleanup_pass") is True
            attempts.append({
                "model": model, "model_order_index": MODELS.index(model),
                "child_exit_code": process.returncode,
                "elapsed_seconds": round(time.time() - started, 6),
                "status": status.get("status", "missing"),
                "cleanup_pass": cleanup_pass,
                "status_sha256": sha256_file(status_path) if status_path.is_file() else None,
                "authorization_sha256": sha256_file(authorization_path),
            })
            if status.get("status") in {"complete", "failed"} and cleanup_pass:
                prior_terminal_statuses.append({
                    "model": model,
                    "relative_path": str(status_path.relative_to(pending)).replace("\\", "/"),
                    "sha256": sha256_file(status_path),
                })
            if not cleanup_pass:
                fatal_cleanup = True
                break
        completed = [
            item["model"] for item in attempts
            if item["status"] == "complete" and item["child_exit_code"] == 0
            and item["cleanup_pass"] is True
        ]
        failed = [item["model"] for item in attempts if item["model"] not in completed]
        not_attempted = [model for model in MODELS if model not in {
            item["model"] for item in attempts
        }]
        qualification_pass = (
            mode == "engineering" and completed == list(MODELS)
            and not failed and not not_attempted and not fatal_cleanup
        )
        behavior_execution_complete = (
            mode == "development" and completed == list(MODELS)
            and not failed and not not_attempted and not fatal_cleanup
        )
        pre_receipt_registry = _artifact_registry(pending)
        receipt = {
            "schema_version": "phase578_execution_receipt.v1",
            "phase_id": PHASE, "created_at_utc": now(), "mode": mode,
            "required_model_order": list(MODELS),
            "attempted_models_in_order": [item["model"] for item in attempts],
            "completed_models": completed, "failed_models": failed,
            "not_attempted_models": not_attempted, "attempts": attempts,
            "fatal_cleanup_failure": fatal_cleanup,
            "qualification_passed": qualification_pass,
            "development_gpu_authorized": qualification_pass,
            "behavior_raw_execution_complete": behavior_execution_complete,
            "behavior_scoring_performed": False,
            "bridge_identity": bridge,
            "engineering_verification": engineering_verification,
            "artifact_registry_before_receipt": pre_receipt_registry,
            "artifact_registry_sha256": sha256_bytes(
                canonical_json(pre_receipt_registry).encode("utf-8")
            ),
            "activation_collected": False,
            "hidden_states_requested": False,
            "attentions_requested": False,
            "scores_requested": False,
            "hooks_registered": 0,
            "causal_intervention": False,
            "confirmation_accessed": False,
            "heldout_accessed": False,
            "sealed_accessed": False,
            "candidate_coordinates": [],
            "candidate_mechanism_formulas": [],
        }
        write_json(pending / "execution_receipt.json", receipt)
        pending.rename(final_dir)
        pending = None
        return receipt
    finally:
        release_lease(lease)


def verify_execution(mode: str) -> dict[str, Any]:
    bridge = verify_bridge()
    protocol = read_json(PROTOCOL_PATH)
    root = ENGINEERING_DIR if mode == "engineering" else DEVELOPMENT_DIR
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"missing Phase578 {mode} execution")
    receipt = read_json(root / "execution_receipt.json")
    registry = receipt.get("artifact_registry_before_receipt")
    expected_paths = {item["path"] for item in registry}
    actual_paths = {
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*") if path.is_file()
    }
    if actual_paths != expected_paths | {"execution_receipt.json"}:
        raise RuntimeError("execution artifact closure drift")
    for item in registry:
        path = root / item["path"]
        if path.is_symlink() or sha256_file(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            raise RuntimeError(f"execution artifact drift: {item['path']}")
    model_status_checks: dict[str, bool] = {}
    attempts_by_model = {item["model"]: item for item in receipt.get("attempts", [])}
    for model in MODELS:
        model_root = root / f"{MODELS.index(model):02d}_{model}"
        status = read_json(model_root / "status.json")
        loaded = status.get("model_identity", {})
        quant = loaded.get("loaded_quantization", {})
        artifact_report = status.get("model_artifact_verification", {})
        expected_artifact = protocol.get("frozen_model_artifact_identities", {}).get(
            model, {}
        )
        expected_artifact_files = sorted((
            {
                "relative_path": item["relative_path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for group in ("tokenizer_and_config_files", "weight_files")
            for item in expected_artifact.get(group, [])
        ), key=lambda item: item["relative_path"])
        authorization_path = root / f"worker_authorization_{MODELS.index(model):02d}_{model}.json"
        attempt = attempts_by_model.get(model, {})
        artifact_payload = dict(artifact_report)
        reported_artifact_payload_hash = artifact_payload.pop(
            "verification_payload_sha256", None
        )
        status_ok = all((
            status.get("status") == "complete",
            status.get("model") == model,
            status.get("model_order_index") == MODELS.index(model),
            status.get("raw_row_count") == (16 if mode == "engineering" else 672),
            status.get("expected_raw_row_count")
            == (16 if mode == "engineering" else 672),
            status.get("cleanup", {}).get("cleanup_pass") is True,
            status.get("cleanup", {}).get("allocated_after_release") == 0,
            status.get("cleanup", {}).get("reserved_after_release") == 0,
            status.get("research_access_attempts", {}).get(
                "forbidden_research_open_attempts"
            ) == 0,
            loaded.get("weights_loaded") is True,
            loaded.get("gpu_used") is True,
            loaded.get("loaded_attn_implementation") == "sdpa",
            loaded.get("cuda_only_no_cpu_or_disk_offload") is True,
            quant.get("load_in_8bit") is True,
            quant.get("non_quantized_dtype") == "torch.bfloat16",
            status.get("activation_collected") is False,
            status.get("hidden_states_requested") is False,
            status.get("attentions_requested") is False,
            status.get("scores_requested") is False,
            status.get("hooks_registered") == 0,
            status.get("causal_intervention") is False,
            status.get("sealed_model_access") is False,
            status.get("automatic_fallback_used") is False,
            artifact_report.get("model") == model,
            artifact_report.get("file_count") == expected_artifact.get(
                "artifact_file_count"
            ),
            artifact_report.get("frozen_identity_sha256")
            == expected_artifact.get("identity_sha256"),
            artifact_report.get("files") == expected_artifact_files,
            isinstance(artifact_report.get("verification_payload_sha256"), str),
            reported_artifact_payload_hash == sha256_bytes(
                canonical_json(artifact_payload).encode("utf-8")
            ),
            authorization_path.is_file(),
            status.get("worker_authorization_sha256")
            == sha256_file(authorization_path),
            attempt.get("child_exit_code") == 0,
            attempt.get("status_sha256") == sha256_file(model_root / "status.json"),
            attempt.get("authorization_sha256") == sha256_file(authorization_path),
        ))
        if mode == "engineering":
            capsule = read_json(model_root / "engineering_generation_capsule.json")
            status_ok = status_ok and all((
                capsule.get("row_count") == 16,
                capsule.get("repeat_outputs_exact") is True,
                capsule.get("activation_collected") is False,
            ))
        model_status_checks[model] = status_ok
    checks = {
        "bridge": receipt.get("bridge_identity") == bridge,
        "model_order": receipt.get("attempted_models_in_order") == list(MODELS),
        "all_completed": receipt.get("completed_models") == list(MODELS),
        "none_failed": receipt.get("failed_models") == [],
        "none_not_attempted": receipt.get("not_attempted_models") == [],
        "cleanup": all(item.get("cleanup_pass") is True for item in receipt["attempts"]),
        "child_exit_codes_zero": all(
            item.get("child_exit_code") == 0 for item in receipt["attempts"]
        ),
        "model_statuses": all(model_status_checks.values()),
        "no_internal": receipt.get("activation_collected") is False
        and receipt.get("candidate_coordinates") == []
        and receipt.get("candidate_mechanism_formulas") == [],
        "no_future_split": receipt.get("confirmation_accessed") is False
        and receipt.get("heldout_accessed") is False
        and receipt.get("sealed_accessed") is False,
        "mode_gate": (
            receipt.get("qualification_passed") is True
            and receipt.get("development_gpu_authorized") is True
            if mode == "engineering"
            else receipt.get("behavior_raw_execution_complete") is True
            and receipt.get("behavior_scoring_performed") is False
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase578 {mode} execution verification failed: {checks}")
    return {
        "schema_version": "phase578_execution_verification.v1",
        "phase_id": PHASE, "mode": mode, "passed": True, "checks": checks,
        "execution_receipt_sha256": sha256_file(root / "execution_receipt.json"),
        "model_status_checks": model_status_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--engineering", action="store_true")
    group.add_argument("--development", action="store_true")
    group.add_argument("--verify-engineering", action="store_true")
    group.add_argument("--verify-development", action="store_true")
    group.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("engineering", "development"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--pending-root", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-nonce")
    args = parser.parse_args()
    if args.worker:
        if any(value is None for value in (
            args.mode, args.model, args.pending_root,
            args.authorization, args.authorization_nonce,
        )):
            raise RuntimeError("worker requires frozen mode/model/pending root")
        raise SystemExit(worker(
            args.model, args.mode, args.pending_root,
            args.authorization, args.authorization_nonce,
        ))
    if any(value is not None for value in (
        args.mode, args.model, args.pending_root,
        args.authorization, args.authorization_nonce,
    )):
        raise RuntimeError("worker-only arguments supplied to parent")
    if args.engineering:
        result = parent_run("engineering")
    elif args.development:
        result = parent_run("development")
    elif args.verify_engineering:
        result = verify_execution("engineering")
    else:
        result = verify_execution("development")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
