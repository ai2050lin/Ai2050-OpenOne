#!/usr/bin/env python3
"""CPU/tokenizer-only analysis of immutable Phase578 development raw output."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import importlib
import importlib.util
import importlib.metadata
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Scoring is CPU-only and follows immutable raw publication.  Hiding CUDA keeps
# this process from accidentally becoming an internal-state collection stage.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[2]

PHASE = "Phase578"
MODELS = ("qwen3", "glm4", "deepseek7b")
REPEATS = ("repeat1", "repeat2")
PROTOCOL_DIR = ROOT / "tests/glm5/result/phase578_gpt5_runner_scorer_protocol"
RAW_DIR = ROOT / "tests/glm5/result/phase578_gpt5_development_behavior_raw"
ANALYSIS_DIR = ROOT / "tests/glm5/result/phase578_gpt5_development_behavior_analysis"
DEVELOPMENT_PATH = (
    ROOT / "tests/glm5/result/phase577_gpt5_natural_behavior_protocol/phase577_development_cases.jsonl"
)
MANIFEST_PATH = PROTOCOL_DIR / "phase578_development_prompt_manifest.jsonl"
PROTOCOL_PATH = PROTOCOL_DIR / "phase578_preregistered_runner_protocol.json"
FREEZE_PATH = PROTOCOL_DIR / "phase578_freeze_commit.json"
SOURCE_RELATIVE = "tests/glm5/phase578_gpt5_behavior_analysis.py"
SCORER_PATH = ROOT / "tests/glm5/phase578_gpt5_behavior_scorer.py"
MODEL_REGISTRY_PATH = ROOT / "tests/gpt5/model_registry.py"
MAX_NEW_TOKENS = 24
PREFIX_TOKEN_BUDGET = 8


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n"
    ).encode("utf-8")


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or path.exists():
        raise RuntimeError(f"no-overwrite analysis publication refused: {path}")
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


def load_sealed_module(unique_name: str, path: Path, expected_sha256: str) -> Any:
    """Load one exact frozen source file without module-name path resolution."""
    resolved = path.resolve(strict=True)
    if path.is_symlink() or sha256_file(resolved) != expected_sha256:
        raise RuntimeError(f"sealed module identity drift: {path}")
    spec = importlib.util.spec_from_file_location(unique_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct sealed module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(unique_name)
    sys.modules[unique_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(unique_name, None)
        else:
            sys.modules[unique_name] = previous
        raise
    if (
        Path(module.__file__).resolve(strict=True) != resolved
        or sha256_file(Path(module.__file__).resolve(strict=True)) != expected_sha256
    ):
        raise RuntimeError(f"loaded sealed module identity mismatch: {path}")
    return module


def verify_inputs() -> dict[str, Any]:
    # Truth-bearing Phase577 files are deliberately absent from this first
    # stage.  A premature analysis attempt must fail on the raw receipt before
    # opening or hashing any full development case record.
    for path in (FREEZE_PATH, PROTOCOL_PATH, MANIFEST_PATH):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"analysis input missing or aliased: {path}")
    freeze, protocol = read_json(FREEZE_PATH), read_json(PROTOCOL_PATH)
    if not all((
        freeze.get("freeze_complete") is True,
        freeze.get("protocol_sha256") == sha256_file(PROTOCOL_PATH),
        freeze.get("development_manifest_sha256") == sha256_file(MANIFEST_PATH),
        protocol.get("models_in_required_order") == list(MODELS),
    )):
        raise RuntimeError("analysis bridge identity failed")
    source = freeze.get("source_identities", {}).get(SOURCE_RELATIVE, {})
    if source.get("sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("analysis source drift")
    for name, expected in freeze.get("source_identities", {}).items():
        candidate = Path(expected.get("path"))
        path = candidate if candidate.is_absolute() else ROOT / candidate
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != expected.get("size_bytes")
            or sha256_file(path) != expected.get("sha256")
        ):
            raise RuntimeError(f"analysis source identity drift: {name}")
    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "bitsandbytes", "accelerate")
    }
    if packages != protocol.get("formal_runtime_identity", {}).get("packages"):
        raise RuntimeError("analysis formal package identity drift")
    formal = Path(protocol["formal_runtime_identity"]["python_executable"])
    if (
        Path(sys.executable).resolve() != formal.resolve(strict=True)
        or sha256_file(formal)
        != protocol["formal_runtime_identity"]["python_executable_identity"]["sha256"]
    ):
        raise RuntimeError("analysis formal interpreter identity drift")
    for model, entry in protocol.get("frozen_tokenizer_input_identities", {}).items():
        for filename, expected in entry.get("files", {}).items():
            path = Path(expected["resolved_path"])
            if (
                not path.is_file() or path.is_symlink()
                or path.stat().st_size != expected["size_bytes"]
                or sha256_file(path) != expected["sha256"]
            ):
                raise RuntimeError(f"analysis tokenizer input drift: {model}/{filename}")
    receipt_path = RAW_DIR / "execution_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError("immutable raw execution receipt is missing")
    receipt = read_json(receipt_path)
    expected_bridge = {
        "freeze_sha256": sha256_file(FREEZE_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "audit_sha256": sha256_file(PROTOCOL_DIR / "phase578_independent_audit.json"),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "runner_sha256": freeze["source_identities"][
            "tests/glm5/phase578_gpt5_development_runner.py"
        ]["sha256"],
        "engine_sha256": protocol["upstream_identities"]["cross_model_engine"]["sha256"],
        "formal_python_sha256": protocol["formal_runtime_identity"][
            "python_executable_identity"
        ]["sha256"],
        "formal_packages": packages,
        "truth_bearing_upstream_reopened": False,
    }
    if not all((
        receipt.get("behavior_raw_execution_complete") is True,
        receipt.get("behavior_scoring_performed") is False,
        receipt.get("attempted_models_in_order") == list(MODELS),
        receipt.get("completed_models") == list(MODELS),
        receipt.get("failed_models") == [],
        receipt.get("bridge_identity") == expected_bridge,
        all(item.get("child_exit_code") == 0 for item in receipt.get("attempts", [])),
        all(item.get("cleanup_pass") is True for item in receipt.get("attempts", [])),
    )):
        raise RuntimeError("raw development execution is not scoreable")
    registry = receipt.get("artifact_registry_before_receipt", [])
    actual_paths = {
        str(path.relative_to(RAW_DIR)).replace("\\", "/")
        for path in RAW_DIR.rglob("*") if path.is_file()
    }
    if actual_paths != {item["path"] for item in registry} | {
        "execution_receipt.json"
    }:
        raise RuntimeError("raw development artifact closure drift")
    for item in registry:
        path = RAW_DIR / item["path"]
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != item["size_bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise RuntimeError(f"raw artifact drift: {item['path']}")
    attempts = {item["model"]: item for item in receipt["attempts"]}
    for model in MODELS:
        status_path = RAW_DIR / f"{MODELS.index(model):02d}_{model}/status.json"
        status = read_json(status_path)
        attempt = attempts[model]
        if not all((
            status.get("status") == "complete",
            status.get("model") == model,
            status.get("model_order_index") == MODELS.index(model),
            status.get("raw_row_count") == 672,
            status.get("cleanup", {}).get("cleanup_pass") is True,
            status.get("cleanup", {}).get("allocated_after_release") == 0,
            status.get("cleanup", {}).get("reserved_after_release") == 0,
            status.get("model_identity", {}).get("weights_loaded") is True,
            status.get("model_identity", {}).get("loaded_attn_implementation") == "sdpa",
            status.get("model_artifact_verification", {}).get("model") == model,
            status.get("activation_collected") is False,
            attempt.get("status_sha256") == sha256_file(status_path),
        )):
            raise RuntimeError(f"raw model status identity drift: {model}")

    # Only a complete, immutable, identity-closed raw publication grants the
    # analysis process access to Phase577 truth-bearing upstream artifacts.
    if not DEVELOPMENT_PATH.is_file() or DEVELOPMENT_PATH.is_symlink():
        raise RuntimeError(f"analysis truth-bearing input missing: {DEVELOPMENT_PATH}")
    for name, expected in protocol.get("upstream_identities", {}).items():
        candidate = Path(expected.get("path"))
        path = candidate if candidate.is_absolute() else ROOT / candidate
        if (
            not path.is_file() or path.is_symlink()
            or path.stat().st_size != expected.get("size_bytes")
            or sha256_file(path) != expected.get("sha256")
        ):
            raise RuntimeError(f"analysis upstream identity drift: {name}")
    if protocol["upstream_identities"]["phase577_development"]["sha256"] != sha256_file(
        DEVELOPMENT_PATH
    ):
        raise RuntimeError("analysis development identity drift")
    return {
        "freeze_sha256": sha256_file(FREEZE_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "development_sha256": sha256_file(DEVELOPMENT_PATH),
        "raw_execution_receipt_sha256": sha256_file(receipt_path),
        "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
        "bridge_identity": expected_bridge,
        "formal_packages": packages,
    }


def _case_record_hashes() -> dict[str, str]:
    output = {}
    with DEVELOPMENT_PATH.open("rb") as handle:
        for raw in handle:
            line = raw.rstrip(b"\r\n")
            if not line:
                continue
            case = json.loads(line.decode("utf-8"))
            output[case["case_id"]] = sha256_bytes(line)
    return output


def render_chat(tokenizer: Any, model: str, content: str) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if model == "qwen3":
        kwargs["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], **kwargs
    )
    if model == "deepseek7b" and rendered.endswith("<think>\n"):
        rendered += "</think>\n\n"
    return rendered


def _plain_eos(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, bool):
        raise RuntimeError("boolean EOS is invalid")
    if isinstance(value, int):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        raise RuntimeError("EOS registry has invalid type")
    if not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0
               for item in values):
        raise RuntimeError("EOS registry has invalid value")
    return [int(item) for item in values]


def tokenizer_bundle(model: str) -> tuple[Any, list[int], int]:
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    protocol = read_json(PROTOCOL_PATH)
    expected = protocol["upstream_identities"]["model_registry"]["sha256"]
    registry = load_sealed_module(
        "_phase578_analysis_model_registry", MODEL_REGISTRY_PATH, expected
    )
    spec = registry.get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
        local_files_only=True, use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
    )
    generation = GenerationConfig.from_pretrained(
        str(spec.local_dir), local_files_only=True,
    )
    eos_ids = sorted(set(
        _plain_eos(getattr(tokenizer, "eos_token_id", None))
        + _plain_eos(getattr(config, "eos_token_id", None))
        + _plain_eos(getattr(generation, "eos_token_id", None))
    ))
    pad = getattr(tokenizer, "pad_token_id", None)
    if not eos_ids or not isinstance(pad, int) or isinstance(pad, bool):
        raise RuntimeError(f"{model}: tokenizer EOS/pad registry invalid")
    return tokenizer, eos_ids, int(pad)


def _decode(tokenizer: Any, values: list[int]) -> str:
    return tokenizer.decode(
        values, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def validate_raw_rows(
    model: str, rows: list[dict[str, Any]], manifest: list[dict[str, Any]],
    case_hashes: dict[str, str],
) -> dict[str, Any]:
    tokenizer, eos_ids, pad_id = tokenizer_bundle(model)
    frozen_generation_hash = sha256_bytes(
        canonical_json(read_json(PROTOCOL_PATH)["generation_contract"]).encode("utf-8")
    )
    manifest_by_id = {row["case_id"]: row for row in manifest}
    expected_keys = {
        (case_id, repeat) for case_id in manifest_by_id for repeat in REPEATS
    }
    actual_keys = [(row.get("case_id"), row.get("execution_repeat")) for row in rows]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        raise RuntimeError(f"{model}: raw case x repeat registry drift")
    termination = Counter()
    for observed in rows:
        case_id = observed["case_id"]
        projected = manifest_by_id[case_id]
        immutable = {
            "schema_version": "phase578_development_behavior_row.v1",
            "phase_id": PHASE, "mode": "development", "model": model,
            "model_order_index": MODELS.index(model), "split": "development",
            "source_case_record_sha256": case_hashes[case_id],
            "generation_contract_sha256": frozen_generation_hash,
            "observer_only": True, "activation_collected": False,
            "hidden_states_requested": False, "attentions_requested": False,
            "scores_requested": False, "hooks_registered": 0,
            "causal_intervention": False, "sealed_model_access": False,
        }
        # The generation hash field above is checked explicitly against the
        # frozen stage-start contract below; keeping it in immutable asserts type.
        for key, expected in immutable.items():
            if observed.get(key) != expected:
                raise RuntimeError(f"{model}/{case_id}: raw drift in {key}")
        rendered = render_chat(tokenizer, model, projected["raw_prompt"])
        input_ids = [int(value) for value in tokenizer(
            rendered, add_special_tokens=False, return_attention_mask=False,
        ).input_ids]
        if not all((
            observed.get("rendered_prompt_sha256")
            == sha256_bytes(rendered.encode("utf-8")),
            observed.get("input_token_ids") == input_ids,
            observed.get("input_token_count") == len(input_ids),
            observed.get("attention_mask_valid_tokens") == len(input_ids),
            observed.get("input_token_ids_sha256")
            == sha256_bytes(canonical_json(input_ids).encode("utf-8")),
            observed.get("effective_eos_token_ids") == eos_ids,
            observed.get("pad_token_id") == pad_id,
        )):
            raise RuntimeError(f"{model}/{case_id}: prompt/tokenizer identity drift")
        suffix = observed.get("full_generated_suffix_token_ids")
        content = observed.get("generated_token_ids_before_eos")
        if not isinstance(suffix, list) or not suffix or len(suffix) > MAX_NEW_TOKENS:
            raise RuntimeError(f"{model}/{case_id}: suffix registry invalid")
        if not all(isinstance(value, int) and not isinstance(value, bool)
                   and 0 <= value < len(tokenizer) for value in suffix):
            raise RuntimeError(f"{model}/{case_id}: suffix token invalid")
        first = next((index for index, value in enumerate(suffix) if value in eos_ids), None)
        rebuilt_content = suffix if first is None else suffix[:first]
        post = [] if first is None else suffix[first + 1:]
        eos_seen = first is not None
        budget = not eos_seen and len(suffix) == MAX_NEW_TOKENS
        expected_prefixes = [
            _decode(tokenizer, rebuilt_content[:index])
            for index in range(1, min(PREFIX_TOKEN_BUDGET, len(rebuilt_content)) + 1)
        ]
        expected_pieces = [
            str(value) for value in tokenizer.convert_ids_to_tokens(rebuilt_content)
        ]
        checks = (
            content == rebuilt_content,
            observed.get("generated_token_count_before_eos") == len(rebuilt_content),
            observed.get("generated_text") == _decode(tokenizer, rebuilt_content),
            observed.get("generated_token_pieces_before_eos") == expected_pieces,
            observed.get("prefix_text_by_generated_token") == expected_prefixes,
            observed.get("full_generated_suffix_decode") == _decode(tokenizer, suffix),
            observed.get("first_eos_index") == first,
            observed.get("first_eos_token_id")
            == (None if first is None else suffix[first]),
            observed.get("post_eos_token_ids") == post,
            observed.get("post_eos_tokens_all_pad") == all(value == pad_id for value in post),
            observed.get("eos_seen") is eos_seen,
            observed.get("budget_truncated") is budget,
            observed.get("termination_event")
            == ("eos" if eos_seen else "budget" if budget else "other"),
            (eos_seen and all(value == pad_id for value in post)) or budget,
        )
        if not all(checks):
            raise RuntimeError(f"{model}/{case_id}: generated evidence reconstruction failed")
        termination[observed["termination_event"]] += 1
    del tokenizer
    gc.collect()
    return {
        "row_count": len(rows),
        "case_count": len({row["case_id"] for row in rows}),
        "termination_counts": dict(sorted(termination.items())),
        "tokenizer_reconstruction_passed": True,
    }


def run_analysis() -> dict[str, Any]:
    if ANALYSIS_DIR.exists():
        raise RuntimeError("Phase578 analysis output already exists")
    inputs = verify_inputs()
    freeze = read_json(FREEZE_PATH)
    scorer_expected = freeze["source_identities"][
        "tests/glm5/phase578_gpt5_behavior_scorer.py"
    ]["sha256"]
    scorer_module = load_sealed_module(
        "_phase578_analysis_behavior_scorer", SCORER_PATH, scorer_expected
    )
    cases = read_jsonl(DEVELOPMENT_PATH)
    manifest = read_jsonl(MANIFEST_PATH)
    scorer_module.validate_case_registry(cases)
    case_hashes = _case_record_hashes()
    if any(case_hashes[row["case_id"]] != row["source_case_record_sha256"]
           for row in manifest):
        raise RuntimeError("prompt manifest/full development record hash mismatch")
    pending = ANALYSIS_DIR.with_name(f".{ANALYSIS_DIR.name}.pending-{os.getpid()}")
    pending.mkdir(parents=True, exist_ok=False)
    try:
        model_reports = []
        for model in MODELS:
            raw_path = RAW_DIR / f"{MODELS.index(model):02d}_{model}/raw_rows.jsonl.gz"
            rows = read_jsonl_gz(raw_path)
            reconstruction = validate_raw_rows(model, rows, manifest, case_hashes)
            decision = scorer_module.score_model(cases, rows, model)
            scorer_core_hash = decision.pop("decision_payload_sha256")
            decision["scorer_core_payload_sha256"] = scorer_core_hash
            decision["created_at_utc"] = now()
            decision["raw_rows_sha256"] = sha256_file(raw_path)
            decision["tokenizer_reconstruction"] = reconstruction
            decision["scorer_source_sha256"] = sha256_file(
                SCORER_PATH
            )
            decision["analysis_payload_sha256_excluding_self"] = sha256_bytes(
                canonical_json(decision).encode("utf-8")
            )
            decision_path = pending / f"phase578_{model}_development_decision.json"
            write_json(decision_path, decision)
            model_reports.append({
                "model": model,
                "behavior_gate_pass": decision["behavior_gate_pass"],
                "semantic_stable_case_count": decision["semantic_stable_case_count"],
                "semantic_stable_case_micro_rate": decision[
                    "semantic_stable_case_micro_rate"
                ],
                "passing_analysis_units": decision["passing_analysis_units"],
                "family_passing_units": decision["family_passing_units"],
                "exact_short_stable_case_count": decision[
                    "exact_short_stable_case_count"
                ],
                "exact_short_case_count": decision["exact_short_case_count"],
                "full_generated_identity_case_count": decision[
                    "full_generated_identity_case_count"
                ],
                "both_repeats_eos_case_count": decision[
                    "both_repeats_eos_case_count"
                ],
                "decision_sha256": sha256_file(decision_path),
            })
        eligible = [
            item["model"] for item in model_reports if item["behavior_gate_pass"]
        ]
        summary = {
            "schema_version": "phase578_development_behavior_analysis.v1",
            "phase_id": PHASE, "created_at_utc": now(),
            "input_identities": inputs,
            "models_in_required_order": list(MODELS),
            "model_reports": model_reports,
            "behavior_passed_models": eligible,
            "behavior_blocked_models": [model for model in MODELS if model not in eligible],
            "future_single_model_natural_trace_eligible_models": eligible,
            "cross_model_internal_comparison_authorized": eligible == list(MODELS),
            "internal_trace_run_count": 0,
            "activation_collected": False,
            "candidate_coordinates": [],
            "candidate_mechanism_formulas": [],
            "statistical_independence_claimed": False,
            "mechanism_claim_authorized": False,
        }
        write_json(pending / "phase578_development_behavior_summary.json", summary)
        artifacts = [
            {
                "path": str(path.relative_to(pending)).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(pending.iterdir()) if path.is_file()
        ]
        receipt = {
            "schema_version": "phase578_analysis_receipt.v1",
            "phase_id": PHASE, "created_at_utc": now(),
            "analysis_complete": True,
            "artifact_registry_before_receipt": artifacts,
            "artifact_registry_sha256": sha256_bytes(
                canonical_json(artifacts).encode("utf-8")
            ),
            "full_development_access": True,
            "full_development_access_occurred_after_raw_publication": True,
            "confirmation_accessed": False, "heldout_accessed": False,
            "sealed_accessed": False, "gpu_used": False,
            "model_weights_loaded": False, "activation_collected": False,
        }
        write_json(pending / "phase578_analysis_receipt.json", receipt)
        pending.rename(ANALYSIS_DIR)
        return summary
    except BaseException:
        if pending.exists():
            failed = pending.with_name(
                f".{ANALYSIS_DIR.name}.failed-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            )
            if pending.parent.resolve(strict=True) != ANALYSIS_DIR.parent.resolve(strict=True):
                raise RuntimeError("analysis pending quarantine escaped result root")
            pending.rename(failed)
        raise


def verify_analysis() -> dict[str, Any]:
    inputs = verify_inputs()
    receipt_path = ANALYSIS_DIR / "phase578_analysis_receipt.json"
    summary_path = ANALYSIS_DIR / "phase578_development_behavior_summary.json"
    if not receipt_path.is_file() or not summary_path.is_file():
        raise RuntimeError("Phase578 analysis is missing")
    receipt, summary = read_json(receipt_path), read_json(summary_path)
    registry = receipt.get("artifact_registry_before_receipt", [])
    actual = {
        str(path.relative_to(ANALYSIS_DIR)).replace("\\", "/")
        for path in ANALYSIS_DIR.rglob("*") if path.is_file()
    }
    if actual != {item["path"] for item in registry} | {
        "phase578_analysis_receipt.json"
    }:
        raise RuntimeError("Phase578 analysis artifact closure drift")
    for item in registry:
        path = ANALYSIS_DIR / item["path"]
        if sha256_file(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            raise RuntimeError(f"Phase578 analysis artifact drift: {item['path']}")
    decision_hash_checks = {}
    for model in MODELS:
        decision_path = ANALYSIS_DIR / f"phase578_{model}_development_decision.json"
        decision = read_json(decision_path)
        expected = decision.pop("analysis_payload_sha256_excluding_self", None)
        decision_hash_checks[model] = expected == sha256_bytes(
            canonical_json(decision).encode("utf-8")
        )
    checks = {
        "inputs": summary.get("input_identities") == inputs,
        "model_order": summary.get("models_in_required_order") == list(MODELS),
        "model_count": len(summary.get("model_reports", [])) == 3,
        "no_internal": summary.get("internal_trace_run_count") == 0
        and summary.get("activation_collected") is False,
        "no_candidates": summary.get("candidate_coordinates") == []
        and summary.get("candidate_mechanism_formulas") == [],
        "receipt": receipt.get("analysis_complete") is True
        and receipt.get("gpu_used") is False
        and receipt.get("model_weights_loaded") is False,
        "decision_payload_hashes": all(decision_hash_checks.values()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase578 analysis verification failed: {checks}")
    return {
        "schema_version": "phase578_analysis_verification.v1",
        "phase_id": PHASE, "passed": True, "checks": checks,
        "summary_sha256": sha256_file(summary_path),
        "receipt_sha256": sha256_file(receipt_path),
        "decision_payload_hash_checks": decision_hash_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = run_analysis() if args.run else verify_analysis()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
