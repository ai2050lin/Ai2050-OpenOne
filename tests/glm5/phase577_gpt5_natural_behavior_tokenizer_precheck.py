#!/usr/bin/env python3
"""Tokenizer-only Phase577 precheck; never opens model weight files."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/glm5/result/phase577_gpt5_natural_behavior_protocol"
PROTOCOL_PATH = OUT_DIR / "phase577_preregistered_protocol.json"
STAGE_COMMIT_PATH = OUT_DIR / "phase577_stage_commit.json"
OUTPUT_PATH = OUT_DIR / "phase577_tokenizer_precheck.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
MODEL_DIRS = {
    "qwen3": ROOT / "models/hf/qwen3-4b",
    "glm4": ROOT / "models/hf/glm4-9b-chat-hf",
    "deepseek7b": ROOT / "models/hf/deepseek-r1-distill-qwen-7b",
}
OPEN_SPLITS = ("development", "confirmation", "heldout_novel_entities")
STAGE_PUBLIC_RELATIVES = (
    "phase577_development_cases.jsonl",
    "phase577_confirmation_cases.jsonl",
    "phase577_heldout_novel_entities_cases.jsonl",
    "phase577_preregistered_protocol.json",
    "phase577_dataset_audit.json",
    "phase577_sealed_commitment.json",
)
BASE_RESULT_RELATIVES = set(STAGE_PUBLIC_RELATIVES) | {
    "phase577_stage_commit.json",
    "protocol/private/phase577_sealed_cases.jsonl",
}
WEIGHT_SUFFIXES = (
    ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx",
)
PRIVATE_FORBIDDEN_ROOTS = (
    OUT_DIR / "protocol/private",
    ROOT / "tests/glm5/result/phase576_gpt5_fruit_structure/protocol/private",
    ROOT / "tests/glm5/result/phase576r1_gpt5_fruit_structure/protocol/private",
    ROOT / "tests/glm5/result/phase576r2_gpt5_fruit_structure/protocol/private",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def confined_path(base: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path must be a non-empty string")
    pure = PurePosixPath(relative)
    if (
        "\\" in relative
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or str(pure) != relative
    ):
        raise ValueError(f"unsafe artifact path: {relative!r}")
    base_resolved = base.resolve(strict=True)
    candidate = base.joinpath(*pure.parts)
    candidate.resolve(strict=True).relative_to(base_resolved)
    cursor = candidate
    while cursor != base:
        if cursor.is_symlink():
            raise ValueError(f"symlink is forbidden in result path: {relative!r}")
        cursor = cursor.parent
    return candidate


def result_file_relatives() -> set[str]:
    if not OUT_DIR.is_dir() or OUT_DIR.is_symlink():
        raise RuntimeError("Phase577 result root is missing or is a symlink")
    result = set()
    allowed_directories = {"protocol", "protocol/private"}
    for path in OUT_DIR.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Phase577 result symlink is forbidden: {path}")
        if path.is_file():
            result.add(str(path.relative_to(OUT_DIR)).replace("\\", "/"))
        elif path.is_dir():
            relative = str(path.relative_to(OUT_DIR)).replace("\\", "/")
            if relative not in allowed_directories:
                raise RuntimeError(f"unexpected Phase577 result directory: {path}")
        else:
            raise RuntimeError(f"unsupported Phase577 result entry: {path}")
    return result


def verify_public_identity(
    identity: Any, expected_relative: str
) -> bool:
    if not isinstance(identity, dict) or set(identity) != {
        "path", "size_bytes", "sha256",
    } or identity.get("path") != expected_relative:
        return False
    try:
        path = confined_path(OUT_DIR, expected_relative)
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        path.is_file()
        and path.stat().st_size == identity.get("size_bytes")
        and sha256_file(path) == identity.get("sha256")
    )


def verify_tokenizer_inputs(protocol: dict[str, Any]) -> dict[str, Any]:
    registry = protocol.get("tokenizer_input_identities")
    if not isinstance(registry, dict) or set(registry) != set(MODELS):
        raise RuntimeError("Phase577 tokenizer input registry model set drift")
    for model in MODELS:
        entry = registry.get(model)
        if not isinstance(entry, dict) or set(entry) != {
            "entry_path", "entry_is_symlink", "resolved_directory", "files",
        }:
            raise RuntimeError(f"{model}: tokenizer registry schema drift")
        directory = MODEL_DIRS[model]
        if (
            entry["entry_path"]
            != str(directory.relative_to(ROOT)).replace("\\", "/")
            or entry["entry_is_symlink"] is not directory.is_symlink()
            or entry["resolved_directory"]
            != str(directory.resolve(strict=True)).replace("\\", "/")
        ):
            raise RuntimeError(f"{model}: tokenizer directory target drift")
        files = entry["files"]
        if not isinstance(files, dict) or not {
            "tokenizer_config.json", "tokenizer.json",
        }.issubset(files):
            raise RuntimeError(f"{model}: tokenizer registry file set drift")
        for name, identity in files.items():
            if not isinstance(name, str) or "/" in name or "\\" in name:
                raise RuntimeError(f"{model}: unsafe tokenizer file name")
            path = directory / name
            expected_path = f"{entry['entry_path']}/{name}"
            if (
                not isinstance(identity, dict)
                or set(identity) != {
                    "path", "resolved_path", "size_bytes", "sha256",
                    "leaf_is_symlink",
                }
                or identity.get("path") != expected_path
                or identity.get("resolved_path")
                != str(path.resolve(strict=True)).replace("\\", "/")
                or identity.get("leaf_is_symlink") is not path.is_symlink()
                or identity.get("size_bytes") != path.stat().st_size
                or identity.get("sha256") != sha256_file(path)
            ):
                raise RuntimeError(f"{model}/{name}: tokenizer input drift")
    registry_hash = hashlib.sha256(canonical_bytes(registry)).hexdigest()
    if protocol.get("tokenizer_input_registry_sha256") != registry_hash:
        raise RuntimeError("Phase577 tokenizer registry hash drift")
    return registry


def verify_stage_before_run(protocol: dict[str, Any]) -> dict[str, Any]:
    if result_file_relatives() != BASE_RESULT_RELATIVES:
        raise RuntimeError("Phase577 tokenizer precheck requires an exact base stage")
    stage = read_json(STAGE_COMMIT_PATH)
    public = stage.get("public_artifact_identities")
    if (
        stage.get("schema_version") != "phase577_stage_commit.v2"
        or stage.get("phase_id") != "Phase577"
        or stage.get("stage_complete") is not True
        or not isinstance(public, dict)
        or set(public) != set(STAGE_PUBLIC_RELATIVES)
        or not all(
            verify_public_identity(public.get(relative), relative)
            for relative in STAGE_PUBLIC_RELATIVES
        )
        or stage.get("source_identities") != protocol.get("source_identities")
        or stage.get("candidate_coordinates") != []
        or stage.get("candidate_mechanism_formulas") != []
        or stage.get("gpu_used") is not False
        or stage.get("model_weights_loaded") is not False
    ):
        raise RuntimeError("Phase577 tokenizer stage-chain verification failed")
    return public


def write_new(path: Path, payload: Any) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite tokenizer artifact: {path}")
    data = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def verify_source_chain(protocol: dict[str, Any]) -> None:
    relative = "tests/glm5/phase577_gpt5_natural_behavior_tokenizer_precheck.py"
    expected = protocol["source_identities"][relative]
    path = ROOT / relative
    if (
        expected["sha256"] != sha256_file(path)
        or expected["size_bytes"] != path.stat().st_size
        or expected["is_symlink"] is not False
    ):
        raise RuntimeError("Phase577 tokenizer source identity drift")
    if protocol.get("candidate_coordinates_before_trace") != [] or protocol.get(
        "candidate_mechanism_formulas_before_trace"
    ) != []:
        raise RuntimeError("Phase577 protocol contains premature internal candidates")


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
        raise RuntimeError(f"{model}: chat template returned an empty prompt")
    return rendered


def compute_tokenizer_evidence(
    rows: list[dict[str, Any]], tokenizer_registry: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    weight_attempts: list[str] = []
    private_content_open_attempts: list[str] = []
    unregistered_model_input_attempts: list[str] = []
    allowed_model_inputs = {
        Path(identity["resolved_path"]).resolve(strict=True)
        for entry in tokenizer_registry.values()
        for identity in entry["files"].values()
    }
    resolved_model_directories = {
        Path(entry["resolved_directory"]).resolve(strict=True)
        for entry in tokenizer_registry.values()
    }
    original_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open

    def forbidden_weight(file: Any) -> bool:
        if isinstance(file, int):
            return False
        try:
            return Path(file).suffix.casefold() in WEIGHT_SUFFIXES
        except TypeError:
            return False

    def forbidden_private(file: Any) -> bool:
        if isinstance(file, int):
            return False
        try:
            candidate = Path(file).resolve(strict=False)
        except (OSError, TypeError, ValueError):
            return False
        return any(
            candidate == root.resolve(strict=False)
            or root.resolve(strict=False) in candidate.parents
            for root in PRIVATE_FORBIDDEN_ROOTS
        )

    def unregistered_model_input(file: Any) -> bool:
        if isinstance(file, int):
            return False
        try:
            candidate = Path(file).resolve(strict=False)
        except (OSError, TypeError, ValueError):
            return False
        inside_model = any(
            candidate == directory or directory in candidate.parents
            for directory in resolved_model_directories
        )
        return inside_model and candidate not in allowed_model_inputs

    def guard(original: Any) -> Any:
        def guarded(file: Any, *args: Any, **kwargs: Any) -> Any:
            label = os.fspath(file) if isinstance(file, (str, os.PathLike)) else ""
            if forbidden_private(file):
                private_content_open_attempts.append(label)
                raise RuntimeError(f"private sealed content open forbidden: {label}")
            if forbidden_weight(file):
                weight_attempts.append(label)
                raise RuntimeError(
                    f"weight file open forbidden in tokenizer precheck: {label}"
                )
            if unregistered_model_input(file):
                unregistered_model_input_attempts.append(label)
                raise RuntimeError(f"unregistered tokenizer input open: {label}")
            return original(file, *args, **kwargs)
        return guarded

    builtins.open = guard(original_open)
    io.open = guard(original_io_open)
    os.open = guard(original_os_open)
    reports = []
    try:
        from transformers import AutoTokenizer, __version__ as transformers_version

        for model in MODELS:
            started = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(
                str(MODEL_DIRS[model]),
                trust_remote_code=False,
                local_files_only=True,
                use_fast=False,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            prompt_lengths = []
            candidate_lengths = []
            rendered_hashes = []
            split_counts = {split: 0 for split in OPEN_SPLITS}
            for row in rows:
                rendered = render_chat(tokenizer, model, row["raw_prompt"])
                token_ids = tokenizer(
                    rendered,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
                if (
                    not token_ids
                    or not all(isinstance(value, int) and value >= 0 for value in token_ids)
                    or len(token_ids) > 512
                ):
                    raise RuntimeError(f"{model}/{row['case_id']}: invalid prompt tokens")
                for candidate in row["candidate_groups"]:
                    candidate_ids = tokenizer(
                        candidate,
                        add_special_tokens=False,
                        return_attention_mask=False,
                    )["input_ids"]
                    if not candidate_ids or len(candidate_ids) > 8:
                        raise RuntimeError(
                            f"{model}/{row['case_id']}: invalid candidate tokenization"
                        )
                    candidate_lengths.append(len(candidate_ids))
                prompt_lengths.append(len(token_ids))
                rendered_hashes.append(
                    hashlib.sha256(rendered.encode("utf-8")).hexdigest()
                )
                split_counts[row["split"]] += 1
            if len(set(rendered_hashes)) != len(rendered_hashes):
                raise RuntimeError(f"{model}: rendered prompts are not unique")
            reports.append({
                "model": model,
                "model_order_index": MODELS.index(model),
                "tokenizer_class": type(tokenizer).__name__,
                "tokenizer_length": len(tokenizer),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "case_count": len(rows),
                "split_counts": split_counts,
                "prompt_token_min": min(prompt_lengths),
                "prompt_token_max": max(prompt_lengths),
                "candidate_token_min": min(candidate_lengths),
                "candidate_token_max": max(candidate_lengths),
                "rendered_prompt_registry_sha256": hashlib.sha256(
                    canonical_bytes(rendered_hashes)
                ).hexdigest(),
                "elapsed_seconds": time.perf_counter() - started,
                "weight_file_open_attempt_count": len(weight_attempts),
            })
            del tokenizer
    finally:
        builtins.open = original_open
        io.open = original_io_open
        os.open = original_os_open
    if weight_attempts:
        raise RuntimeError("tokenizer precheck attempted to open model weights")
    if private_content_open_attempts:
        raise RuntimeError("tokenizer precheck attempted to open sealed content")
    if unregistered_model_input_attempts:
        raise RuntimeError("tokenizer precheck opened an unregistered model input")
    return reports, transformers_version


def run() -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file() or not STAGE_COMMIT_PATH.is_file():
        raise RuntimeError("Phase577 stage must be frozen before tokenizer precheck")
    protocol = read_json(PROTOCOL_PATH)
    verify_source_chain(protocol)
    public_identities = verify_stage_before_run(protocol)
    tokenizer_registry = verify_tokenizer_inputs(protocol)
    if protocol.get("models_in_required_order") != list(MODELS):
        raise RuntimeError("Phase577 model order drift")
    rows = []
    input_case_identities = {}
    for split in OPEN_SPLITS:
        relative = f"phase577_{split}_cases.jsonl"
        path = confined_path(OUT_DIR, relative)
        input_case_identities[split] = public_identities[relative]
        split_rows = read_jsonl(path)
        if len(split_rows) != 336 or any(row.get("sealed") is not False for row in split_rows):
            raise RuntimeError(f"{split}: tokenizer denominator invalid")
        rows.extend(split_rows)
    if len(rows) != 1008:
        raise RuntimeError("Phase577 open tokenizer denominator is not 1008")

    reports, transformers_version = compute_tokenizer_evidence(
        rows, tokenizer_registry
    )
    payload = {
        "schema_version": "phase577_tokenizer_precheck.v2",
        "phase_id": "Phase577",
        "created_at_utc": now(),
        "passed": True,
        "models_in_observed_order": [report["model"] for report in reports],
        "open_case_count": len(rows),
        "input_case_artifact_identities": input_case_identities,
        "input_case_id_registry_sha256": hashlib.sha256(canonical_bytes(
            [row["case_id"] for row in rows]
        )).hexdigest(),
        "reports": reports,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "stage_commit_sha256": sha256_file(STAGE_COMMIT_PATH),
        "tokenizer_source_sha256": sha256_file(Path(__file__).resolve()),
        "tokenizer_input_registry_sha256": hashlib.sha256(
            canonical_bytes(tokenizer_registry)
        ).hexdigest(),
        "tokenizer_inputs_verified": True,
        "transformers_version": transformers_version,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_used": False,
        "model_weights_loaded": False,
        "weight_file_open_attempts": [],
        "unregistered_model_input_attempts": [],
        "private_content_open_attempts": [],
        "sealed_payload_read": False,
        "old_phase576_sealed_payload_read": False,
    }
    write_new(OUTPUT_PATH, payload)
    return payload


def validate_payload(payload: dict[str, Any]) -> dict[str, bool]:
    protocol = read_json(PROTOCOL_PATH)
    verify_source_chain(protocol)
    tokenizer_registry = verify_tokenizer_inputs(protocol)
    stage = read_json(STAGE_COMMIT_PATH)
    public = stage.get("public_artifact_identities", {})
    input_identities = {
        split: public.get(f"phase577_{split}_cases.jsonl")
        for split in OPEN_SPLITS
    }
    rows = [
        row
        for split in OPEN_SPLITS
        for row in read_jsonl(OUT_DIR / f"phase577_{split}_cases.jsonl")
    ]
    reports = payload.get("reports")
    report_schema = {
        "model", "model_order_index", "tokenizer_class", "tokenizer_length",
        "pad_token_id", "eos_token_id", "case_count", "split_counts",
        "prompt_token_min", "prompt_token_max", "candidate_token_min",
        "candidate_token_max", "rendered_prompt_registry_sha256",
        "elapsed_seconds", "weight_file_open_attempt_count",
    }
    reports_valid = (
        isinstance(reports, list)
        and len(reports) == 3
        and all(isinstance(item, dict) and set(item) == report_schema for item in reports)
        and [item.get("model") for item in reports] == list(MODELS)
        and all(
            item.get("model_order_index") == index
            and item.get("case_count") == 1008
            and item.get("split_counts") == {split: 336 for split in OPEN_SPLITS}
            and isinstance(item.get("tokenizer_class"), str)
            and bool(item.get("tokenizer_class"))
            and isinstance(item.get("tokenizer_length"), int)
            and item.get("tokenizer_length", 0) > 0
            and isinstance(item.get("prompt_token_min"), int)
            and 0 < item.get("prompt_token_min", 0)
            <= item.get("prompt_token_max", 0) <= 512
            and isinstance(item.get("candidate_token_min"), int)
            and 0 < item.get("candidate_token_min", 0)
            <= item.get("candidate_token_max", 0) <= 8
            and isinstance(item.get("rendered_prompt_registry_sha256"), str)
            and len(item["rendered_prompt_registry_sha256"]) == 64
            and isinstance(item.get("elapsed_seconds"), (int, float))
            and item.get("elapsed_seconds", -1) >= 0
            and item.get("weight_file_open_attempt_count") == 0
            for index, item in enumerate(reports)
        )
    )
    recomputed_reports, recomputed_transformers_version = compute_tokenizer_evidence(
        rows, tokenizer_registry
    )

    def deterministic_report(report: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in report.items() if key != "elapsed_seconds"}

    reports_recomputed = (
        reports_valid
        and [deterministic_report(item) for item in reports]
        == [deterministic_report(item) for item in recomputed_reports]
    )
    expected_keys = {
        "schema_version", "phase_id", "created_at_utc", "passed",
        "models_in_observed_order", "open_case_count",
        "input_case_artifact_identities", "input_case_id_registry_sha256",
        "reports", "protocol_sha256", "stage_commit_sha256",
        "tokenizer_source_sha256", "tokenizer_input_registry_sha256",
        "tokenizer_inputs_verified", "transformers_version",
        "cuda_visible_devices", "cuda_used", "model_weights_loaded",
        "weight_file_open_attempts", "private_content_open_attempts",
        "unregistered_model_input_attempts",
        "sealed_payload_read", "old_phase576_sealed_payload_read",
    }
    return {
        "exact_schema": set(payload) == expected_keys,
        "schema": payload.get("schema_version") == "phase577_tokenizer_precheck.v2",
        "phase": payload.get("phase_id") == "Phase577",
        "passed": payload.get("passed") is True,
        "model_order": payload.get("models_in_observed_order") == list(MODELS),
        "case_count": payload.get("open_case_count") == 1008 and len(rows) == 1008,
        "input_case_chain": payload.get("input_case_artifact_identities")
        == input_identities,
        "input_case_id_chain": payload.get("input_case_id_registry_sha256")
        == hashlib.sha256(canonical_bytes(
            [row["case_id"] for row in rows]
        )).hexdigest(),
        "reports": reports_valid,
        "reports_recomputed": reports_recomputed,
        "transformers_version_recomputed": payload.get("transformers_version")
        == recomputed_transformers_version,
        "protocol_chain": payload.get("protocol_sha256") == sha256_file(PROTOCOL_PATH),
        "stage_chain": payload.get("stage_commit_sha256")
        == sha256_file(STAGE_COMMIT_PATH),
        "source_chain": payload.get("tokenizer_source_sha256")
        == sha256_file(Path(__file__).resolve()),
        "tokenizer_input_chain": payload.get("tokenizer_input_registry_sha256")
        == protocol.get("tokenizer_input_registry_sha256")
        and payload.get("tokenizer_inputs_verified") is True,
        "no_cuda": payload.get("cuda_visible_devices") == ""
        and payload.get("cuda_used") is False,
        "no_weights": payload.get("model_weights_loaded") is False
        and payload.get("weight_file_open_attempts") == []
        and payload.get("unregistered_model_input_attempts") == [],
        "sealed_unread": payload.get("private_content_open_attempts") == []
        and payload.get("sealed_payload_read") is False
        and payload.get("old_phase576_sealed_payload_read") is False,
    }


def verify() -> dict[str, Any]:
    payload = read_json(OUTPUT_PATH)
    checks = validate_payload(payload)
    if not all(checks.values()):
        raise RuntimeError(f"Phase577 tokenizer verification failed: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "files_written": False,
        "cuda_used": False,
        "model_weights_loaded": False,
        "sealed_payload_read": False,
        "tokenizer_precheck_sha256": sha256_file(OUTPUT_PATH),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = run() if args.run else verify()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
