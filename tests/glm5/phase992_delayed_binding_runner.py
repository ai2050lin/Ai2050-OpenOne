#!/usr/bin/env python3
"""Fail-closed Phase992 delayed two-hop external-behavior runner.

The public parent process creates truth-free manifests and launches exactly
one formal-Python worker per model.  A worker replays raw Phase991 prompts,
performs greedy natural generation plus the frozen four-candidate
teacher-forced diagnostic, seals a gzip JSONL artifact, releases CUDA, and
exits.  This source never opens scoring truth and never requests hidden
states, attentions, hooks, gradients, or generation scores.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence
import uuid


PHASE = 992
EXPERIMENT = "delayed_two_hop_gpu_behavior"
ACTIVATION_SCHEMA = "phase992_gpu_behavior_activation.v1"
RAW_SCHEMA = "phase992_delayed_binding_raw.v1"
ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
P991_ROOT = GLM5 / "result" / "phase991_delayed_binding_gpu_admission"
PROTOCOL_ROOT = GLM5 / "result" / "phase992_delayed_binding_behavior_protocol"
DEFAULT_ACTIVATION = PROTOCOL_ROOT / "activation.json"
DEFAULT_EXECUTION_ROOT = GLM5 / "result" / "phase992_delayed_binding_behavior_execution"
ENGINE_PATH = GLM5 / "phase983_cross_model_engine.py"
RUNNER_PATH = Path(__file__).resolve()
MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
VALUES = ("red", "blue", "green", "black")
PUBLIC_SPLITS = ("discovery", "confirmation", "adversarial")
PROMPT_KEYS = {
    "schema_version", "input_mode", "prompt", "prompt_sha256", "record_id",
    "semantic_world_id", "split", "split_ordinal", "variant_id",
}
FORBIDDEN_PROMPT_KEYS = {
    "gold", "gold_value", "gold_object", "answer_value", "target", "foil",
    "foil_values", "query_entity", "query_relation", "semantic_peer_record_ids",
}
EXPECTED_COUNTS = {"engineering": 8, "primary": 8192, "holdout": 2048}
MAX_NEW_TOKENS = 24
BATCH_SIZE = 8
MIN_FREE_DISK_GIB = 80
FORMAL_PYTHON_SHA256 = "0f11fb7422fa347b7609ba0964ceccef3c8fa9f15230c37b9ec27668e68e8a8a"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def verify_self_hash(document: Mapping[str, Any], field: str, label: str) -> None:
    observed = document.get(field)
    require(isinstance(observed, str) and len(observed) == 64, f"{label} hash missing")
    body = {key: value for key, value in document.items() if key != field}
    require(sha256_json(body) == observed, f"{label} self-hash mismatch")


def sealed(document: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = deepcopy(dict(document))
    require(field not in result, f"reserved self-hash field: {field}")
    result[field] = sha256_json(result)
    return result


def json_bytes(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(
        document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n").encode("utf-8")


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def file_seal(path: Path, base: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing/aliased artifact: {path}")
    try:
        display = str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except ValueError:
        display = str(path.resolve())
    return {"path": display, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def resolve_seal_path(seal: Mapping[str, Any], base: Path) -> Path:
    raw = Path(str(seal.get("path", "")))
    path = raw if raw.is_absolute() else base / raw
    path = path.resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), f"sealed source invalid: {path}")
    require(path.stat().st_size == seal.get("bytes") and sha256_file(path) == seal.get("sha256"),
            f"sealed source drift: {path}")
    return path


def verify_activation(path: Path, *, require_formal_python: bool = True) -> dict[str, Any]:
    require(path.resolve(strict=True) == DEFAULT_ACTIVATION.resolve(strict=True),
            "activation must be the unique frozen default artifact")
    activation = read_json(path)
    verify_self_hash(activation, "activation_sha256", "activation")
    require(activation.get("schema_version") == ACTIVATION_SCHEMA, "activation schema drift")
    require(activation.get("phase") == PHASE and activation.get("experiment") == EXPERIMENT,
            "activation identity drift")
    require(activation.get("gpu_behavior_execution_authorized") is True
            and activation.get("behavior_only_authorized") is True
            and activation.get("runner_fail_closed_unless_all_identity_and_authority_checks_pass") is True,
            "behavior execution is not authorized")
    for key in (
        "internal_trace_authorized", "hidden_states_authorized", "attentions_authorized",
        "scoring_authorized", "causal_intervention_authorized",
        "mechanism_formula_authorized", "expanded_confirmation_authorized",
    ):
        require(activation.get(key) is False, f"unsafe activation flag: {key}")
    require(tuple(activation.get("model_order", ())) == MODEL_ORDER, "model order drift")
    require(activation.get("execution_root") ==
            "tests/glm5/result/phase992_delayed_binding_behavior_execution",
            "execution root drift")
    formal = activation.get("formal_python")
    require(isinstance(formal, Mapping) and formal.get("sha256") == FORMAL_PYTHON_SHA256,
            "formal Python activation drift")
    formal_path = Path(str(formal["path"])).resolve(strict=True)
    require(sha256_file(formal_path) == FORMAL_PYTHON_SHA256, "formal Python file drift")
    if require_formal_python:
        require(Path(sys.executable).resolve(strict=True) == formal_path,
                "runner is not executing under frozen formal Python")
    sources = activation.get("source_seals")
    require(isinstance(sources, Mapping) and set(sources) ==
            {"protocol", "broker", "runner", "scorer", "audit"},
            "activation source registry drift")
    for role, seal in sources.items():
        resolved = resolve_seal_path(seal, ROOT)
        if role == "runner":
            require(resolved == RUNNER_PATH and seal.get("sha256") == sha256_file(RUNNER_PATH),
                    "runner is not activation-sealed")
    engine = activation.get("phase983_engine")
    require(isinstance(engine, Mapping), "engine seal absent")
    require(resolve_seal_path(engine, ROOT) == ENGINE_PATH.resolve(strict=True), "engine path drift")
    generation = activation.get("generation_contract")
    require(isinstance(generation, Mapping)
            and activation.get("generation_contract_sha256") == sha256_json(generation),
            "generation contract hash drift")
    required_generation = {
        "input_mode": "raw_text_no_chat_template_add_special_tokens_false",
        "chat_template": False, "add_special_tokens": False, "padding_side": "left",
        "truncation": False, "batch_size": BATCH_SIZE, "do_sample": False,
        "num_beams": 1, "num_return_sequences": 1, "use_cache": True,
        "max_new_tokens": MAX_NEW_TOKENS, "output_scores": False,
        "output_attentions": False, "output_hidden_states": False,
        "return_dict_in_generate": True, "load_in_8bit": True,
        "attention_implementation": "sdpa", "cpu_or_disk_offload": False,
        "automatic_fallback": False, "one_model_resident_at_a_time": True,
    }
    require(all(generation.get(key) == value for key, value in required_generation.items()),
            "generation contract values drift")
    teacher = activation.get("teacher_forced_contract")
    require(isinstance(teacher, Mapping)
            and activation.get("teacher_forced_contract_sha256") == sha256_json(teacher)
            and tuple(teacher.get("candidate_values_in_order", ())) == VALUES,
            "teacher-forced contract drift")
    return activation


def validate_prompt(row: Mapping[str, Any]) -> dict[str, Any]:
    require(set(row) == PROMPT_KEYS, f"prompt manifest field drift: {sorted(set(row) ^ PROMPT_KEYS)}")
    require(not (set(row) & FORBIDDEN_PROMPT_KEYS), "truth leaked into prompt manifest")
    require(row.get("schema_version") == "phase991_runtime_prompt.v1", "prompt schema drift")
    require(row.get("input_mode") == "raw_text_no_chat_template_add_special_tokens_false",
            "prompt input mode drift")
    for key in ("prompt", "prompt_sha256", "record_id", "semantic_world_id", "split", "variant_id"):
        require(isinstance(row.get(key), str) and row[key], f"invalid prompt field: {key}")
    require(isinstance(row.get("split_ordinal"), int) and row["split_ordinal"] >= 0,
            "invalid split ordinal")
    require(sha256_bytes(row["prompt"].encode("utf-8")) == row["prompt_sha256"],
            "prompt text hash mismatch")
    return dict(row)


def phase991_public_prompt_source(split: str) -> Path:
    require(split in PUBLIC_SPLITS, "requested prompt source is not public")
    admission_path = P991_ROOT / "gpu_admission_preregistration.json"
    admission = read_json(admission_path)
    verify_self_hash(admission, "gpu_admission_sha256", "Phase991 admission")
    relative = f"runtime_prompts/public/{split}.jsonl"
    seal = admission.get("artifact_seals", {}).get(relative)
    require(isinstance(seal, Mapping), f"Phase991 prompt seal missing: {split}")
    source = resolve_seal_path(seal, P991_ROOT)
    require(source == (P991_ROOT / relative).resolve(strict=True),
            f"Phase991 prompt source path drift: {split}")
    return source


def read_manifest(path: Path, expected_scope: str) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), "manifest missing or aliased")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(validate_prompt(json.loads(line)))
    require(len(rows) == EXPECTED_COUNTS[expected_scope], "manifest row-count mismatch")
    ids = [row["record_id"] for row in rows]
    require(len(ids) == len(set(ids)), "manifest record IDs are not unique")
    if expected_scope == "primary":
        require({row["split"] for row in rows} == set(PUBLIC_SPLITS), "public split set drift")
    return rows


def write_manifest(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for row in rows:
            handle.write((canonical_json(row) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    return file_seal(path, DEFAULT_EXECUTION_ROOT)


def build_public_manifest(execution_root: Path) -> tuple[Path, dict[str, Any]]:
    path = execution_root / "manifests" / "primary.jsonl"
    all_rows: list[dict[str, Any]] = []
    expected_bytes = bytearray()
    for split in PUBLIC_SPLITS:
        source = phase991_public_prompt_source(split)
        expected_bytes.extend(source.read_bytes())
        with source.open("r", encoding="utf-8") as handle:
            all_rows.extend(validate_prompt(json.loads(line)) for line in handle if line.strip())
    require(len(all_rows) == EXPECTED_COUNTS["primary"], "public manifest count drift")
    require(len({row["record_id"] for row in all_rows}) == len(all_rows), "public IDs overlap")
    seal = write_manifest(path, all_rows)
    require(seal["bytes"] == len(expected_bytes)
            and seal["sha256"] == sha256_bytes(bytes(expected_bytes)),
            "combined public manifest differs from exact sealed Phase991 prompt bytes")
    return path, seal


def build_engineering_manifest(execution_root: Path, activation: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    frozen = activation.get("engineering_contract", {}).get("engineering_records")
    require(isinstance(frozen, list) and len(frozen) == 8, "engineering contract drift")
    wanted = {str(item["record_id"]): dict(item) for item in frozen}
    require(len(wanted) == 8, "engineering IDs duplicate")
    source = phase991_public_prompt_source("discovery")
    found: dict[str, dict[str, Any]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = validate_prompt(json.loads(line))
            if row["record_id"] in wanted:
                found[row["record_id"]] = row
    require(set(found) == set(wanted), "engineering records missing")
    ordered = [found[str(item["record_id"])] for item in frozen]
    for row, item in zip(ordered, frozen, strict=True):
        require(all(row[key] == item[key] for key in
                    ("record_id", "semantic_world_id", "variant_id", "prompt_sha256")),
                "engineering record identity drift")
    path = execution_root / "manifests" / "engineering.jsonl"
    return path, write_manifest(path, ordered)


def gzip_rows_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    require(not path.exists(), f"refusing to overwrite raw artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
    os.close(fd)
    pending = Path(raw_name)
    count = 0
    digest = hashlib.sha256()
    try:
        with pending.open("wb") as target:
            with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
                for row in rows:
                    line = (canonical_json(row) + "\n").encode("utf-8")
                    compressed.write(line)
                    digest.update(line)
                    count += 1
            target.flush()
            os.fsync(target.fileno())
        require(not path.exists(), "raw artifact appeared during execution")
        os.replace(pending, path)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise
    return count, digest.hexdigest()


def float_entry(number: float) -> dict[str, Any]:
    value = float(number)
    require(math.isfinite(value), "non-finite teacher-forced score")
    return {"value": value, "hex": value.hex()}


def token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text, add_special_tokens=False, return_attention_mask=False,
    )
    ids = list(encoded.input_ids)
    require(ids and all(isinstance(item, int) and item >= 0 for item in ids),
            "tokenizer produced invalid raw IDs")
    return [int(item) for item in ids]


def left_pad(
    torch: Any, sequences: Sequence[Sequence[int]], pad: int, device: Any,
) -> tuple[Any, Any, Any]:
    require(sequences and all(sequences), "cannot pad empty token sequences")
    width = max(len(row) for row in sequences)
    ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    for index, row in enumerate(sequences):
        values = torch.tensor(row, dtype=torch.long, device=device)
        ids[index, -len(row):] = values
        mask[index, -len(row):] = 1
    positions = mask.cumsum(dim=-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions


def candidate_identity(
    tokenizer: Any, contexts: Sequence[str], *, verify_boundaries: bool,
) -> tuple[dict[str, int], list[list[int]]]:
    context_ids = [token_ids(tokenizer, context) for context in contexts]
    candidates: dict[str, int] = {}
    for value in VALUES:
        continuation = " " + value
        ids = token_ids(tokenizer, continuation)
        require(len(ids) == 1, f"candidate is not one token: {value}")
        candidates[value] = ids[0]
        if verify_boundaries:
            for context, prefix in zip(contexts, context_ids, strict=True):
                full = token_ids(tokenizer, context + continuation)
                require(full == prefix + ids, f"candidate boundary drift: {value}")
    return candidates, context_ids


def scientific_rows(
    adapter: Any, torch: Any, prompts: Sequence[Mapping[str, Any]], scope: str,
    run_id: str, activation: Mapping[str, Any], manifest_sha256: str,
) -> Iterator[dict[str, Any]]:
    tokenizer = adapter.tokenizer
    tokenizer.padding_side = "left"
    require(tokenizer.padding_side == "left", "tokenizer did not retain left padding")
    effective_eos = sorted(int(item) for item in adapter.eos_identity["effective_eos_token_ids"])
    pad = int(adapter.pad_token_id)
    require(effective_eos, "effective EOS union is empty")
    generation_sha = str(activation["generation_contract_sha256"])
    activation_sha = str(activation["activation_sha256"])
    for start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[start:start + BATCH_SIZE]
        texts = [str(row["prompt"]) for row in batch]
        raw_ids = [token_ids(tokenizer, text) for text in texts]
        input_ids, attention_mask, _generation_positions = left_pad(
            torch, raw_ids, pad, adapter.input_device,
        )
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
        suffix_tensor = generated.sequences[:, input_ids.shape[1]:]
        suffixes = [[int(item) for item in row] for row in suffix_tensor.detach().cpu().tolist()]
        del generated, suffix_tensor, input_ids, attention_mask, _generation_positions
        contexts = [text + "\nThe retrieved marker is" for text in texts]
        candidate_ids, context_ids = candidate_identity(
            tokenizer, contexts, verify_boundaries=(scope == "engineering" or start == 0),
        )
        tf_input, tf_mask, tf_positions = left_pad(
            torch, context_ids, pad, adapter.input_device,
        )
        with torch.inference_mode():
            outputs = adapter.model(
                input_ids=tf_input,
                attention_mask=tf_mask,
                position_ids=tf_positions,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            require(outputs.logits.ndim == 3 and outputs.logits.shape[1] == 1,
                    "teacher-forced logits_to_keep=1 contract changed")
            last_logits = outputs.logits[:, -1, :].float()
            last_logprobs = torch.log_softmax(last_logits, dim=-1)
            candidate_logits = {
                value: last_logits[:, identifier].detach().cpu().tolist()
                for value, identifier in candidate_ids.items()
            }
            candidate_logprobs = {
                value: last_logprobs[:, identifier].detach().cpu().tolist()
                for value, identifier in candidate_ids.items()
            }
        del outputs, last_logits, last_logprobs, tf_input, tf_mask, tf_positions
        for local_index, (prompt, ids, suffix, context) in enumerate(
            zip(batch, raw_ids, suffixes, contexts, strict=True)
        ):
            first_eos_index = next((i for i, item in enumerate(suffix) if item in effective_eos), None)
            eos_seen = first_eos_index is not None
            before = suffix if first_eos_index is None else suffix[:first_eos_index]
            if not eos_seen:
                require(len(suffix) == MAX_NEW_TOKENS, "generation ended without EOS before budget")
            text = tokenizer.decode(
                before, skip_special_tokens=False, clean_up_tokenization_spaces=False,
            )
            candidates: dict[str, Any] = {}
            for value in VALUES:
                logit = float(candidate_logits[value][local_index])
                logprob = float(candidate_logprobs[value][local_index])
                require(math.isfinite(logit) and math.isfinite(logprob), "non-finite candidate score")
                candidates[value] = {
                    "continuation": " " + value,
                    "token_id": candidate_ids[value],
                    "logit": logit,
                    "logit_hex": logit.hex(),
                    "logprob": logprob,
                    "logprob_hex": logprob.hex(),
                }
            yield {
                "schema_version": RAW_SCHEMA,
                "phase": PHASE,
                "experiment": EXPERIMENT,
                "scope": scope,
                "model": adapter.model_key,
                "model_order_index": MODEL_ORDER.index(adapter.model_key),
                "run_id": run_id,
                "record_id": prompt["record_id"],
                "semantic_world_id": prompt["semantic_world_id"],
                "split": prompt["split"],
                "split_ordinal": prompt["split_ordinal"],
                "variant_id": prompt["variant_id"],
                "prompt_sha256": prompt["prompt_sha256"],
                "input_manifest_sha256": manifest_sha256,
                "input_token_ids": ids,
                "input_token_count": len(ids),
                "input_token_ids_sha256": sha256_json(ids),
                "generated_suffix_token_ids": suffix,
                "generated_token_ids_before_eos": before,
                "generated_text": text,
                "effective_eos_token_ids": effective_eos,
                "first_eos_index": first_eos_index,
                "first_eos_token_id": suffix[first_eos_index] if eos_seen else None,
                "eos_seen": eos_seen,
                "budget_exhausted": not eos_seen,
                "termination_reason": "effective_eos" if eos_seen else "max_new_tokens",
                "teacher_forced_context_sha256": sha256_bytes(context.encode("utf-8")),
                "teacher_forced_candidate_order": list(VALUES),
                "teacher_forced_candidates": candidates,
                "generation_contract_sha256": generation_sha,
                "activation_sha256": activation_sha,
            }


def validate_loaded_identity(identity: Mapping[str, Any], model: str) -> None:
    require(identity.get("model_key") == model and identity.get("weights_loaded") is True
            and identity.get("gpu_used") is True, "loaded model identity drift")
    require(identity.get("loaded_attn_implementation") == "sdpa"
            and identity.get("cuda_only_no_cpu_or_disk_offload") is True,
            "loaded attention/device identity drift")
    quant = identity.get("loaded_quantization")
    require(isinstance(quant, Mapping) and quant.get("load_in_8bit") is True
            and quant.get("backend") == "bitsandbytes"
            and quant.get("non_quantized_dtype") == "torch.bfloat16"
            and quant.get("device_map") == "auto",
            "loaded model is not frozen bitsandbytes INT8")
    device_map = identity.get("hf_device_map")
    require(isinstance(device_map, Mapping) and device_map
            and all(str(value).startswith("cuda:") for value in device_map.values()),
            "loaded device map is not CUDA-only")


def verify_model_artifacts(activation: Mapping[str, Any], model: str) -> dict[str, Any]:
    anchors = activation.get("phase991_anchors")
    require(isinstance(anchors, Mapping), "Phase991 anchor registry missing")
    seal = anchors.get("model_manifests")
    require(isinstance(seal, Mapping), "Phase991 model-manifest seal missing")
    manifest_path = resolve_seal_path(seal, ROOT)
    manifest = read_json(manifest_path)
    verify_self_hash(manifest, "model_manifest_sha256", "Phase991 model manifest")
    require(tuple(manifest.get("model_order", ())) == MODEL_ORDER,
            "Phase991 model manifest order drift")
    entries = manifest.get("models_in_required_order")
    require(isinstance(entries, list) and len(entries) == len(MODEL_ORDER),
            "Phase991 model manifest entries drift")
    entry = entries[MODEL_ORDER.index(model)]
    require(entry.get("model") == model, "Phase991 model manifest index drift")
    resolved_root = Path(str(entry.get("resolved_root", ""))).resolve(strict=True)
    require(resolved_root.is_dir() and not resolved_root.is_symlink(),
            "resolved model root is missing or aliased")
    expected_files = entry.get("files")
    require(isinstance(expected_files, list) and len(expected_files) == entry.get("file_count"),
            "Phase991 per-file model manifest malformed")
    observed_relatives = sorted(
        str(path.relative_to(resolved_root)).replace("\\", "/")
        for path in resolved_root.rglob("*") if path.is_file()
    )
    expected_relatives = [str(item.get("relative_path", "")).replace("\\", "/")
                          for item in expected_files]
    require(observed_relatives == sorted(expected_relatives),
            f"{model} model file population drift")
    verified: list[dict[str, Any]] = []
    total = 0
    weight_total = 0
    weight_count = 0
    for item in expected_files:
        relative = str(item["relative_path"])
        path = (resolved_root / relative).resolve(strict=True)
        require(path.is_file() and not path.is_symlink()
                and path == Path(str(item["resolved_path"])).resolve(strict=True),
                f"{model} model file path/symlink drift: {relative}")
        size = path.stat().st_size
        digest = sha256_file(path)
        require(size == item.get("bytes") and digest == item.get("sha256"),
                f"{model} model file bytes/hash drift: {relative}")
        current = deepcopy(dict(item))
        verified.append(current)
        total += size
        if item.get("is_weight_shard") is True:
            weight_total += size
            weight_count += 1
    require(sha256_json(verified) == entry.get("files_manifest_sha256")
            and total == sum(int(item["bytes"]) for item in expected_files)
            and weight_total == entry.get("weight_bytes")
            and weight_count == entry.get("weight_shard_count"),
            f"{model} full artifact manifest aggregate drift")
    return {
        "passed": True,
        "model": model,
        "model_manifest_sha256": manifest["model_manifest_sha256"],
        "files_manifest_sha256": entry["files_manifest_sha256"],
        "file_count": len(verified),
        "all_file_bytes": total,
        "weight_shard_count": weight_count,
        "weight_bytes": weight_total,
        "resolved_root": str(resolved_root),
        "all_files_sha256_verified_immediately_before_load": True,
    }


def strict_cuda_release(engine: Any, adapter: Any, torch: Any) -> dict[str, Any]:
    base_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        engine.release_model_adapter(adapter)
    except BaseException as error:
        base_error = error
        error.__traceback__ = None
    steps: dict[str, bool] = {}
    try:
        gc.collect()
        torch.cuda.synchronize()
        steps["synchronize_before_cublas_clear"] = True
        clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        require(callable(clear), "required cuBLAS workspace clear API is unavailable")
        clear()
        steps["cublas_workspaces_cleared"] = True
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        steps["final_allocator_cleanup"] = True
    except BaseException as error:
        cleanup_error = error
        error.__traceback__ = None
    allocated = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())
    report = {
        "steps": steps,
        "allocated_after_release": allocated,
        "reserved_after_release": reserved,
        "base_release_error": None if base_error is None else type(base_error).__name__,
        "strict_cleanup_error": None if cleanup_error is None else type(cleanup_error).__name__,
        "cleanup_pass": (base_error is None and cleanup_error is None
                         and allocated == 0 and reserved == 0),
    }
    if base_error is not None:
        raise base_error
    if cleanup_error is not None:
        raise cleanup_error
    require(allocated == 0 and reserved == 0,
            f"strict CUDA cleanup retained allocated={allocated}, reserved={reserved}")
    return report


def worker(
    activation_path: Path, execution_root: Path, scope: str, model: str,
    manifest_path: Path, raw_path: Path, status_path: Path, run_id: str,
) -> dict[str, Any]:
    activation = verify_activation(activation_path)
    require(scope in EXPECTED_COUNTS and model in MODEL_ORDER, "invalid worker scope/model")
    require(execution_root.resolve() == DEFAULT_EXECUTION_ROOT.resolve(), "worker execution root drift")
    manifest_path = manifest_path.resolve(strict=True)
    prompts = read_manifest(manifest_path, scope)
    manifest = file_seal(manifest_path, execution_root)
    require(raw_path.resolve(strict=False).is_relative_to(execution_root.resolve()),
            "raw output escaped execution root")
    require(status_path.resolve(strict=False).is_relative_to(execution_root.resolve()),
            "worker status escaped execution root")
    import torch
    import phase983_cross_model_engine as engine

    adapter = None
    loaded_identity: dict[str, Any] | None = None
    repeat_exact: bool | None = None
    raw_canonical_sha: str | None = None
    started = utc_now()
    artifact_verification = verify_model_artifacts(activation, model)
    release_report: dict[str, Any] | None = None
    try:
        adapter = engine.load_model_adapter(model)
        validate_loaded_identity(adapter.identity, model)
        loaded_identity = deepcopy(adapter.identity)
        if scope == "engineering":
            first = list(scientific_rows(
                adapter, torch, prompts, scope, run_id, activation, manifest["sha256"],
            ))
            second = list(scientific_rows(
                adapter, torch, prompts, scope, run_id, activation, manifest["sha256"],
            ))
            repeat_exact = canonical_json(first) == canonical_json(second)
            require(repeat_exact, "engineering deterministic repeat mismatch")
            count, raw_canonical_sha = gzip_rows_exclusive(raw_path, first)
        else:
            count, raw_canonical_sha = gzip_rows_exclusive(
                raw_path,
                scientific_rows(adapter, torch, prompts, scope, run_id, activation,
                                manifest["sha256"]),
            )
        require(count == EXPECTED_COUNTS[scope], "worker output row-count mismatch")
    finally:
        release_report = strict_cuda_release(engine, adapter, torch)
        adapter = None
    require(release_report is not None and release_report["cleanup_pass"] is True,
            "strict CUDA release report did not pass")
    allocated = int(release_report["allocated_after_release"])
    reserved = int(release_report["reserved_after_release"])
    require(loaded_identity is not None and raw_canonical_sha is not None, "worker did not finish")
    raw = file_seal(raw_path, execution_root)
    status = sealed({
        "schema_version": "phase992_delayed_binding_worker_status.v1",
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
        "generation_contract_sha256": activation["generation_contract_sha256"],
        "runner_source_sha256": sha256_file(RUNNER_PATH),
        "engine_source_sha256": sha256_file(ENGINE_PATH),
        "input_manifest": manifest,
        "raw_artifact": raw,
        "raw_row_count": EXPECTED_COUNTS[scope],
        "raw_canonical_lines_sha256": raw_canonical_sha,
        "record_ids_sha256": sha256_json(sorted(row["record_id"] for row in prompts)),
        "model_artifact_verification": artifact_verification,
        "loaded_model_identity": loaded_identity,
        "engineering_repeat_exact": repeat_exact,
        "model_released": True,
        "strict_cuda_release": release_report,
        "cuda_allocated_after": allocated,
        "cuda_reserved_after": reserved,
        "truth_opened": False,
        "internal_trace_authorized": False,
    }, "worker_status_sha256")
    write_exclusive(status_path, json_bytes(status))
    return {
        "passed": True, "scope": scope, "model": model,
        "worker_status_sha256": status["worker_status_sha256"],
        "row_count": EXPECTED_COUNTS[scope], "cuda_released": True,
        "truth_opened": False,
    }


def gpu_baseline() -> dict[str, Any]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    require(completed.returncode == 0, f"nvidia-smi failed: {completed.stderr[-500:]}")
    devices: list[dict[str, int]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        left, right = [part.strip() for part in line.split(",", 1)]
        devices.append({"index": int(left), "used_mib": int(right)})
    require(devices, "nvidia-smi reported no GPU")
    return {"devices": devices, "used_mib_total": sum(item["used_mib"] for item in devices)}


def resource_preflight() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    free_gib = usage.free / 1024**3
    require(free_gib >= MIN_FREE_DISK_GIB, "free disk is below frozen 80 GiB minimum")
    return {"disk_free_bytes": usage.free, "disk_free_gib": free_gib, "gpu": gpu_baseline()}


def create_lease(execution_root: Path, activation: Mapping[str, Any], run_id: str, scope: str) -> tuple[Path, str]:
    path = execution_root / "execution.lease.json"
    token = sha256_json({
        "pid": os.getpid(), "process_instance": uuid.uuid4().hex,
        "wall_clock_ns": time.time_ns(), "executable": str(Path(sys.executable).resolve()),
    })
    lease = sealed({
        "schema_version": "phase992_execution_lease.v1",
        "activation_sha256": activation["activation_sha256"],
        "run_id": run_id, "pid": os.getpid(), "process_start_token": token,
        "scope": scope, "model": None, "created_at_utc": utc_now(),
    }, "lease_sha256")
    write_exclusive(path, json_bytes(lease))
    return path, str(lease["lease_sha256"])


def release_lease(path: Path, expected_sha: str) -> None:
    lease = read_json(path)
    verify_self_hash(lease, "lease_sha256", "execution lease")
    require(lease["lease_sha256"] == expected_sha, "execution lease ownership drift")
    path.unlink()


def run_child(
    activation: Mapping[str, Any], activation_path: Path, execution_root: Path,
    scope: str, model: str, manifest_path: Path, raw_path: Path,
    status_path: Path, run_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    before = resource_preflight()
    formal = Path(str(activation["formal_python"]["path"]))
    command = [
        str(formal), "-B", str(RUNNER_PATH), "--worker",
        "--activation-path", str(activation_path),
        "--execution-root", str(execution_root), "--scope", scope,
        "--model", model, "--manifest", str(manifest_path),
        "--raw-output", str(raw_path), "--status-output", str(status_path),
        "--run-id", run_id,
    ]
    environment = {
        **os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False, env=environment,
    )
    require(completed.returncode == 0,
            f"{scope}/{model} worker failed: {completed.stderr[-4000:]}")
    try:
        child_report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{scope}/{model} worker stdout is not one JSON document") from error
    require(child_report.get("passed") is True, "worker report did not pass")
    status = read_json(status_path)
    verify_self_hash(status, "worker_status_sha256", "worker status")
    require(status.get("status") == "success" and status.get("scope") == scope
            and status.get("model") == model and status.get("run_id") == run_id,
            "worker status identity drift")
    after_gpu = gpu_baseline()
    recovered = after_gpu["used_mib_total"] <= before["gpu"]["used_mib_total"] + 512
    require(recovered, f"GPU baseline not recovered after {model}")
    cleanup = {
        "baseline_before": before,
        "baseline_after": {"gpu": after_gpu},
        "baseline_recovered": recovered,
        "allocated_after": int(status["cuda_allocated_after"]),
        "reserved_after": int(status["cuda_reserved_after"]),
        "model_released": status.get("model_released") is True,
        "child_exit_zero": completed.returncode == 0,
        "cuda_allocated_zero": status.get("cuda_allocated_after") == 0,
        "cuda_reserved_zero": status.get("cuda_reserved_after") == 0,
        "stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
        "stderr_sha256": sha256_bytes(completed.stderr.encode("utf-8")),
        "stderr_tail": completed.stderr[-2000:],
    }
    cleanup["cleanup_pass"] = all((
        cleanup["baseline_recovered"], cleanup["model_released"], cleanup["child_exit_zero"],
        cleanup["cuda_allocated_zero"], cleanup["cuda_reserved_zero"],
    ))
    require(cleanup["cleanup_pass"], f"cleanup contract failed after {model}")
    return status, cleanup, child_report


def execution_receipt(
    status: Mapping[str, Any], scope: str, model: str, run_id: str,
    previous: str | None, status_path: Path, execution_root: Path,
) -> dict[str, Any]:
    return sealed({
        "schema_version": "phase992_delayed_binding_execution_receipt.v1",
        "phase": PHASE, "experiment": EXPERIMENT, "created_at_utc": utc_now(),
        "scope": scope, "model": model, "model_order_index": MODEL_ORDER.index(model),
        "run_id": run_id, "status": "sealed", "execution_status": "success",
        "previous_model_receipt_sha256": previous,
        "worker_status_sha256": status["worker_status_sha256"],
        "worker_status_artifact": file_seal(status_path, execution_root),
        "raw_artifact": deepcopy(status["raw_artifact"]),
        "input_manifest": deepcopy(status["input_manifest"]),
        "row_count": status["raw_row_count"],
        "record_ids_sha256": status["record_ids_sha256"],
        "activation_sha256": status["activation_sha256"],
        "generation_contract_sha256": status["generation_contract_sha256"],
        "truth_opened": False, "internal_trace_authorized": False,
    }, "receipt_sha256")


def cleanup_receipt(
    cleanup: Mapping[str, Any], activation: Mapping[str, Any], scope: str,
    model: str, run_id: str, worker_sha: str,
) -> dict[str, Any]:
    return sealed({
        "schema_version": "phase992_delayed_binding_cleanup_receipt.v1",
        "phase": PHASE, "experiment": EXPERIMENT, "created_at_utc": utc_now(),
        "scope": scope, "model": model, "model_order_index": MODEL_ORDER.index(model),
        "run_id": run_id, "status": "sealed",
        "activation_sha256": activation["activation_sha256"],
        "worker_status_sha256": worker_sha,
        **deepcopy(dict(cleanup)),
    }, "receipt_sha256")


def verify_engineering_gate(execution_root: Path, activation: Mapping[str, Any]) -> dict[str, Any]:
    path = execution_root / "engineering" / "qualification.json"
    document = read_json(path)
    verify_self_hash(document, "qualification_sha256", "engineering qualification")
    require(document.get("passed") is True
            and document.get("activation_sha256") == activation["activation_sha256"]
            and document.get("runner_source_sha256") == sha256_file(RUNNER_PATH)
            and tuple(document.get("model_order", ())) == MODEL_ORDER,
            "engineering qualification missing or stale")
    return document


def parent_engineering(activation_path: Path, execution_root: Path) -> dict[str, Any]:
    activation = verify_activation(activation_path)
    require(execution_root.resolve() == DEFAULT_EXECUTION_ROOT.resolve(), "execution root drift")
    require(not execution_root.exists(), "engineering requires a fresh Phase992 execution root")
    execution_root.mkdir(parents=True)
    run_id = f"phase992-engineering-{uuid.uuid4().hex}"
    lease_path, lease_sha = create_lease(execution_root, activation, run_id, "engineering")
    model_reports: dict[str, Any] = {}
    try:
        manifest_path, _ = build_engineering_manifest(execution_root, activation)
        for model in MODEL_ORDER:
            raw_path = execution_root / "engineering" / "raw" / f"{model}.jsonl.gz"
            status_path = execution_root / "engineering" / "worker_status" / f"{model}.json"
            status, cleanup, _ = run_child(
                activation, activation_path, execution_root, "engineering", model,
                manifest_path, raw_path, status_path, run_id,
            )
            require(status.get("engineering_repeat_exact") is True,
                    f"engineering repeat failed: {model}")
            receipt = execution_receipt(
                status, "engineering", model, run_id, None, status_path, execution_root,
            )
            cleanup_doc = cleanup_receipt(
                cleanup, activation, "engineering", model, run_id,
                str(status["worker_status_sha256"]),
            )
            write_exclusive(
                execution_root / "engineering" / "receipts" / f"{model}.json",
                json_bytes(receipt),
            )
            write_exclusive(
                execution_root / "engineering" / "receipts" / f"cleanup_{model}.json",
                json_bytes(cleanup_doc),
            )
            model_reports[model] = {
                "worker_status_sha256": status["worker_status_sha256"],
                "receipt_sha256": receipt["receipt_sha256"],
                "cleanup_receipt_sha256": cleanup_doc["receipt_sha256"],
                "repeat_exact": True, "cleanup_pass": True,
                "loaded_model_identity": status["loaded_model_identity"],
            }
        qualification = sealed({
            "schema_version": "phase992_engineering_qualification.v1",
            "phase": PHASE, "experiment": EXPERIMENT, "created_at_utc": utc_now(),
            "passed": True, "run_id": run_id, "model_order": list(MODEL_ORDER),
            "activation_sha256": activation["activation_sha256"],
            "runner_source_sha256": sha256_file(RUNNER_PATH),
            "generation_contract_sha256": activation["generation_contract_sha256"],
            "models": model_reports, "truth_opened": False,
            "scientific_accuracy_evidence": False,
        }, "qualification_sha256")
        write_exclusive(execution_root / "engineering" / "qualification.json",
                        json_bytes(qualification))
        return {"passed": True, "run_id": run_id,
                "qualification_sha256": qualification["qualification_sha256"],
                "models": list(MODEL_ORDER), "truth_opened": False}
    finally:
        if lease_path.exists():
            release_lease(lease_path, lease_sha)


def parent_public(activation_path: Path, execution_root: Path) -> dict[str, Any]:
    activation = verify_activation(activation_path)
    require(execution_root.resolve(strict=True) == DEFAULT_EXECUTION_ROOT.resolve(strict=True),
            "execution root drift")
    engineering = verify_engineering_gate(execution_root, activation)
    require(not (execution_root / "receipts").exists(), "public execution already started")
    run_id = f"phase992-public-{uuid.uuid4().hex}"
    lease_path, lease_sha = create_lease(execution_root, activation, run_id, "primary")
    previous: str | None = None
    reports: dict[str, Any] = {}
    try:
        manifest_path, _ = build_public_manifest(execution_root)
        for model in MODEL_ORDER:
            raw_path = execution_root / "raw" / "primary" / f"{model}.jsonl.gz"
            status_path = execution_root / "worker_status" / "primary" / f"{model}.json"
            status, cleanup, _ = run_child(
                activation, activation_path, execution_root, "primary", model,
                manifest_path, raw_path, status_path, run_id,
            )
            receipt = execution_receipt(
                status, "primary", model, run_id, previous, status_path, execution_root,
            )
            cleanup_doc = cleanup_receipt(
                cleanup, activation, "primary", model, run_id,
                str(status["worker_status_sha256"]),
            )
            receipt_path = execution_root / "receipts" / f"primary_{model}.json"
            cleanup_path = execution_root / "receipts" / f"cleanup_primary_{model}.json"
            write_exclusive(receipt_path, json_bytes(receipt))
            write_exclusive(cleanup_path, json_bytes(cleanup_doc))
            previous = str(receipt["receipt_sha256"])
            reports[model] = {
                "receipt_sha256": previous,
                "cleanup_receipt_sha256": cleanup_doc["receipt_sha256"],
                "row_count": EXPECTED_COUNTS["primary"], "cleanup_pass": True,
            }
        stage = sealed({
            "schema_version": "phase992_public_raw_stage.v1", "phase": PHASE,
            "experiment": EXPERIMENT, "created_at_utc": utc_now(), "passed": True,
            "run_id": run_id, "activation_sha256": activation["activation_sha256"],
            "engineering_qualification_sha256": engineering["qualification_sha256"],
            "model_order": list(MODEL_ORDER), "models": reports,
            "all_raw_and_cleanup_sealed_before_scoring": True, "truth_opened": False,
        }, "stage_sha256")
        write_exclusive(execution_root / "public_raw_stage.json", json_bytes(stage))
        return {"passed": True, "run_id": run_id, "stage_sha256": stage["stage_sha256"],
                "models": reports, "truth_opened": False}
    finally:
        if lease_path.exists():
            release_lease(lease_path, lease_sha)


def parent_holdout(activation_path: Path, execution_root: Path) -> dict[str, Any]:
    activation = verify_activation(activation_path)
    require(execution_root.resolve(strict=True) == DEFAULT_EXECUTION_ROOT.resolve(strict=True),
            "execution root drift")
    verify_engineering_gate(execution_root, activation)
    admission_path = execution_root / "public_behavior_admission.json"
    admission = read_json(admission_path)
    verify_self_hash(admission, "admission_sha256", "public admission")
    require(admission.get("all_models_public_pass") is True
            and admission.get("sealed_holdout_model_access_authorized") is True,
            "public behavior gate did not authorize holdout")
    run_id = str(admission.get("run_id", ""))
    require(run_id, "public admission run ID missing")
    require(not (execution_root / "raw" / "holdout").exists(), "holdout already started")
    (execution_root / "temporary_holdout").mkdir(parents=True, exist_ok=False)
    (execution_root / "holdout_access" / "events").mkdir(parents=True, exist_ok=False)
    lease_path, lease_sha = create_lease(execution_root, activation, run_id, "holdout")
    previous: str | None = None
    reports: dict[str, Any] = {}
    try:
        import phase992_holdout_broker as broker
        runner_sha = sha256_file(RUNNER_PATH)
        for model in MODEL_ORDER:
            temporary = execution_root / "temporary_holdout" / f"{run_id}_{model}.jsonl"
            grant = broker.grant(
                activation_path, admission_path, execution_root, run_id, model,
                temporary, runner_sha,
            )
            grant_path = Path(str(grant["receipt_path"]))
            try:
                raw_path = execution_root / "raw" / "holdout" / f"{model}.jsonl.gz"
                status_path = execution_root / "worker_status" / "holdout" / f"{model}.json"
                status, cleanup, _ = run_child(
                    activation, activation_path, execution_root, "holdout", model,
                    temporary, raw_path, status_path, run_id,
                )
                receipt = execution_receipt(
                    status, "holdout", model, run_id, previous, status_path, execution_root,
                )
                cleanup_doc = cleanup_receipt(
                    cleanup, activation, "holdout", model, run_id,
                    str(status["worker_status_sha256"]),
                )
                receipt_path = execution_root / "receipts" / f"holdout_{model}.json"
                cleanup_path = execution_root / "receipts" / f"cleanup_holdout_{model}.json"
                write_exclusive(receipt_path, json_bytes(receipt))
                write_exclusive(cleanup_path, json_bytes(cleanup_doc))
                seal = broker.finalize(
                    activation_path, admission_path, execution_root, run_id, model,
                    grant_path, receipt_path, cleanup_path,
                )
            except BaseException as original:
                try:
                    broker.abort(
                        activation_path, admission_path, execution_root, run_id, model,
                        grant_path, "worker_failure",
                    )
                except Exception as abort_error:
                    if temporary.exists():
                        raise RuntimeError(
                            f"holdout failure and emergency revoke both failed: {abort_error}"
                        ) from original
                raise
            previous = str(receipt["receipt_sha256"])
            reports[model] = {
                "receipt_sha256": previous,
                "cleanup_receipt_sha256": cleanup_doc["receipt_sha256"],
                "grant_receipt_sha256": grant["receipt_sha256"],
                "seal_receipt_sha256": seal["receipt_sha256"],
                "row_count": EXPECTED_COUNTS["holdout"], "cleanup_pass": True,
            }
        chain = broker.publish_final_chain(execution_root, run_id)
        stage = sealed({
            "schema_version": "phase992_holdout_raw_stage.v1", "phase": PHASE,
            "experiment": EXPERIMENT, "created_at_utc": utc_now(), "passed": True,
            "run_id": run_id, "activation_sha256": activation["activation_sha256"],
            "model_order": list(MODEL_ORDER), "models": reports,
            "holdout_chain_receipt_sha256": chain["receipt_sha256"],
            "all_raw_cleanup_and_revocation_sealed_before_scoring": True,
            "truth_opened": False,
        }, "stage_sha256")
        write_exclusive(execution_root / "holdout_raw_stage.json", json_bytes(stage))
        return {"passed": True, "run_id": run_id, "stage_sha256": stage["stage_sha256"],
                "models": reports, "truth_opened": False}
    finally:
        if lease_path.exists():
            release_lease(lease_path, lease_sha)


def self_test() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    example = {"b": 2, "a": 1}
    checks["canonical_hash_order_independent"] = sha256_json(example) == sha256_json({"a": 1, "b": 2})
    document = sealed({"schema": "x", "value": 1}, "receipt_sha256")
    try:
        verify_self_hash(document, "receipt_sha256", "synthetic")
        checks["self_hash_round_trip"] = True
    except RuntimeError:
        checks["self_hash_round_trip"] = False
    checks["frozen_model_order"] = MODEL_ORDER == ("qwen3", "glm4", "deepseek7b")
    checks["frozen_values"] = VALUES == ("red", "blue", "green", "black")
    checks["frozen_batch_and_budget"] = BATCH_SIZE == 8 and MAX_NEW_TOKENS == 24
    checks["truth_keys_rejected"] = bool(FORBIDDEN_PROMPT_KEYS)
    checks["raw_schema"] = RAW_SCHEMA == "phase992_delayed_binding_raw.v1"
    checks["behavior_only_literals"] = (
        ACTIVATION_SCHEMA == "phase992_gpu_behavior_activation.v1"
        and EXPERIMENT == "delayed_two_hop_gpu_behavior"
    )
    with tempfile.TemporaryDirectory(prefix="phase992-runner-selftest-") as raw:
        path = Path(raw) / "rows.jsonl.gz"
        count, digest = gzip_rows_exclusive(path, [{"x": 1}, {"x": 2}])
        checks["deterministic_gzip_writer"] = count == 2 and len(digest) == 64 and path.is_file()
    require(all(checks.values()), f"runner self-test failed: {checks}")
    return {
        "schema_version": "phase992_delayed_binding_runner_self_test.v1",
        "passed": True, "checks": checks, "cuda_used": False,
        "truth_opened": False, "files_written": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--engineering", action="store_true")
    modes.add_argument("--public", action="store_true")
    modes.add_argument("--holdout", action="store_true")
    modes.add_argument("--worker", action="store_true")
    parser.add_argument("--activation-path", type=Path, default=DEFAULT_ACTIVATION)
    parser.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    parser.add_argument("--scope", choices=tuple(EXPECTED_COUNTS))
    parser.add_argument("--model", choices=MODEL_ORDER)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--status-output", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.self_test:
        result = self_test()
    elif args.engineering:
        result = parent_engineering(args.activation_path.resolve(), args.execution_root.resolve())
    elif args.public:
        result = parent_public(args.activation_path.resolve(), args.execution_root.resolve())
    elif args.holdout:
        result = parent_holdout(args.activation_path.resolve(), args.execution_root.resolve())
    else:
        require(all((args.scope, args.model, args.manifest, args.raw_output,
                     args.status_output, args.run_id)), "worker arguments missing")
        result = worker(
            args.activation_path.resolve(), args.execution_root.resolve(), args.scope,
            args.model, args.manifest.resolve(), args.raw_output.resolve(),
            args.status_output.resolve(), args.run_id,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
