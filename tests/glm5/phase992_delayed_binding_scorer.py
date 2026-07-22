#!/usr/bin/env python3
"""Fail-closed Phase992 delayed-binding behavior scorer.

This is a CPU-only observer.  It never imports a model runtime and it never
opens scoring truth until the complete receipt barrier for the requested
scope has been verified.  The three release stages are intentionally
separate:

* public: all three ``primary`` raw/cleanup receipts -> public truth;
* holdout: sealed all-model public PASS, then all three holdout receipts and
  the completed access/seal chain -> sealed-holdout truth;
* extension: sealed all-model cumulative primary PASS, then all three
  extension receipts -> expanded-confirmation truth.

Natural generation is the sole behavior gate.  Teacher-forced target versus
three-foil margins and registered shortcut baselines are diagnostics only;
none of the outputs are internal-structure or mechanism evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


PHASE = 992
SCHEMA_VERSION = "phase992_delayed_binding_score.v1"
EXPERIMENT = "delayed_two_hop_gpu_behavior"
ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
PROTOCOL_ROOT = GLM5 / "result" / "phase992_delayed_binding_behavior_protocol"
RESULT_ROOT = GLM5 / "result" / "phase992_delayed_binding_behavior_execution"
P991_ROOT = GLM5 / "result" / "phase991_delayed_binding_gpu_admission"
ACTIVATION = PROTOCOL_ROOT / "activation.json"
MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
VALUES = ("red", "blue", "green", "black")
PUBLIC_SPLITS = ("discovery", "confirmation", "adversarial")
ALL_PRIMARY_SPLITS = (*PUBLIC_SPLITS, "sealed_holdout")
SEMANTIC_CONDITIONS = ("original", "value_swap", "binding_swap", "relation_swap")
VARIANT_COUNT = 32

SCOPE_CONTRACT = {
    "public": {
        "raw_scope": "primary",
        "splits": PUBLIC_SPLITS,
        "row_count": 8192,
        "output": "scores/public_score.json",
    },
    "holdout": {
        "raw_scope": "holdout",
        "splits": ("sealed_holdout",),
        "row_count": 2048,
        "output": "scores/holdout_score.json",
    },
    "extension": {
        "raw_scope": "extension",
        "splits": ("expanded_confirmation",),
        "row_count": 4096,
        "output": "scores/extension_score.json",
    },
}
THRESHOLDS = {
    "each_major_condition_min_percent": 90,
    "each_split_variant_min_percent": 85,
    "counterfactual_pair_sensitivity_min_percent": 85,
}
MARKER_RE = re.compile(r"(?<![A-Za-z])(red|blue|green|black)(?![A-Za-z])", re.I)
STRICT_REFERENCE_RE = re.compile(
    r"^The retrieved marker is (red|blue|green|black)\.$", re.I
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
RAW_FORBIDDEN_TRUTH_KEYS = {
    "gold", "gold_value", "gold_object", "answer_value", "target", "foil",
    "foil_values", "query_entity", "query_relation", "semantic_peer_record_ids",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing/aliased JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"missing/aliased JSONL: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        for line_no, line in enumerate(handle, 1):
            require(line.endswith("\n"), f"unterminated JSONL line {path}:{line_no}")
            value = json.loads(line)
            require(isinstance(value, dict), f"JSONL row must be object {path}:{line_no}")
            yield value


def verify_self_hash(document: Mapping[str, Any], field: str, label: str) -> None:
    expected = document.get(field)
    require(isinstance(expected, str) and SHA_RE.fullmatch(expected) is not None,
            f"{label} missing valid {field}")
    body = deepcopy(dict(document))
    body.pop(field, None)
    require(sha256_json(body) == expected, f"{label} self hash mismatch")


def _safe_resolve(path_text: str, result_root: Path) -> Path:
    raw = Path(path_text)
    candidates = [raw] if raw.is_absolute() else [result_root / raw, P991_ROOT / raw, ROOT / raw]
    existing = next((candidate for candidate in candidates if candidate.exists()), None)
    require(existing is not None, f"sealed path does not exist: {path_text}")
    resolved = existing.resolve(strict=True)
    require(resolved == ROOT.resolve() or ROOT.resolve() in resolved.parents,
            f"sealed path escaped workspace: {path_text}")
    require(not existing.is_symlink(), f"sealed path is symlink: {path_text}")
    return existing


def verify_file_seal(seal: Mapping[str, Any], result_root: Path, label: str) -> Path:
    require(isinstance(seal, Mapping), f"{label} seal is not object")
    path = _safe_resolve(str(seal.get("path", "")), result_root)
    require(int(seal.get("bytes", -1)) == path.stat().st_size, f"{label} byte mismatch")
    expected = str(seal.get("sha256", ""))
    require(SHA_RE.fullmatch(expected) is not None, f"{label} invalid sha256")
    require(sha256_file(path) == expected, f"{label} file hash mismatch")
    return path


def load_activation(activation_path: Path, result_root: Path) -> dict[str, Any]:
    activation = load_json(activation_path)
    verify_self_hash(activation, "activation_sha256", "activation")
    require(activation.get("schema_version") == "phase992_gpu_behavior_activation.v1",
            "activation schema mismatch")
    require(activation.get("phase") == PHASE and activation.get("experiment") == EXPERIMENT,
            "activation identity mismatch")
    require(tuple(activation.get("model_order", ())) == MODEL_ORDER, "model order drift")
    formal = activation.get("formal_python")
    require(isinstance(formal, Mapping), "formal Python identity missing")
    formal_path = Path(str(formal.get("path", ""))).resolve(strict=True)
    require(Path(sys.executable).resolve(strict=True) == formal_path
            and sha256_file(formal_path) == formal.get("sha256"),
            "formal Python identity drift")
    require(activation.get("gpu_behavior_execution_authorized") is True,
            "GPU behavior was not activated")
    require(activation.get("behavior_only_authorized") is True, "not behavior-only activation")
    for forbidden in ("internal_trace_authorized", "causal_intervention_authorized",
                      "mechanism_formula_authorized"):
        require(activation.get(forbidden) is False, f"forbidden authorization set: {forbidden}")
    freeze = activation.get("qualified_phase991_freeze")
    verify_file_seal(freeze, activation_path.parent, "qualified Phase991 freeze")
    require(str(freeze.get("freeze_commit_sha256", "")) != "", "missing Phase991 freeze hash")
    source_seals = activation.get("source_seals")
    require(isinstance(source_seals, Mapping), "activation source_seals missing")
    for name in ("protocol", "broker", "runner", "scorer", "audit"):
        verify_file_seal(source_seals.get(name), activation_path.parent, f"source:{name}")
    frozen_execution = activation.get("execution_root")
    require(isinstance(frozen_execution, str), "activation execution_root missing")
    frozen_path = Path(frozen_execution)
    if not frozen_path.is_absolute():
        frozen_path = ROOT / frozen_path
    require(frozen_path.resolve() == result_root.resolve(), "execution root drift")
    return activation


def _receipt_paths(result_root: Path, raw_scope: str, model: str) -> tuple[Path, Path]:
    return (
        result_root / "receipts" / f"{raw_scope}_{model}.json",
        result_root / "receipts" / f"cleanup_{raw_scope}_{model}.json",
    )


def _verify_execution_receipt(
    receipt: Mapping[str, Any], result_root: Path, raw_scope: str,
    model: str, model_index: int, previous_receipt_sha: str | None,
) -> tuple[Path, dict[str, Any], str]:
    verify_self_hash(receipt, "receipt_sha256", f"{raw_scope}/{model} receipt")
    require(receipt.get("phase") == PHASE and receipt.get("experiment") == EXPERIMENT,
            f"{raw_scope}/{model} receipt identity mismatch")
    require(receipt.get("scope") == raw_scope and receipt.get("model") == model,
            f"{raw_scope}/{model} receipt scope/model mismatch")
    require(receipt.get("status") == "sealed", f"{raw_scope}/{model} is not sealed")
    require(receipt.get("execution_status") == "success",
            f"{raw_scope}/{model} execution did not succeed")
    require(receipt.get("model_order_index") == model_index,
            f"{raw_scope}/{model} order index mismatch")
    require(receipt.get("previous_model_receipt_sha256") == previous_receipt_sha,
            f"{raw_scope}/{model} receipt chain mismatch")
    require(isinstance(receipt.get("run_id"), str) and receipt["run_id"], "missing run_id")
    require(SHA_RE.fullmatch(str(receipt.get("worker_status_sha256", ""))) is not None,
            f"{raw_scope}/{model} missing worker status seal")
    worker_path = verify_file_seal(
        receipt.get("worker_status_artifact"), result_root,
        f"worker-status:{raw_scope}/{model}",
    )
    worker = load_json(worker_path)
    verify_self_hash(worker, "worker_status_sha256", f"worker status {raw_scope}/{model}")
    require(worker.get("worker_status_sha256") == receipt.get("worker_status_sha256")
            and worker.get("phase") == PHASE and worker.get("experiment") == EXPERIMENT
            and worker.get("scope") == raw_scope and worker.get("model") == model
            and worker.get("model_order_index") == model_index
            and worker.get("run_id") == receipt.get("run_id")
            and worker.get("status") == "success",
            f"{raw_scope}/{model} worker status identity mismatch")
    require(worker.get("activation_sha256") == receipt.get("activation_sha256")
            and worker.get("generation_contract_sha256")
                == receipt.get("generation_contract_sha256")
            and worker.get("raw_artifact") == receipt.get("raw_artifact")
            and worker.get("input_manifest") == receipt.get("input_manifest")
            and worker.get("raw_row_count") == receipt.get("row_count")
            and worker.get("record_ids_sha256") == receipt.get("record_ids_sha256"),
            f"{raw_scope}/{model} worker/receipt evidence mismatch")
    require(worker.get("runner_source_sha256") == sha256_file(
                GLM5 / "phase992_delayed_binding_runner.py")
            and worker.get("engine_source_sha256") == sha256_file(
                GLM5 / "phase983_cross_model_engine.py"),
            f"{raw_scope}/{model} worker source identity mismatch")
    identity = worker.get("loaded_model_identity")
    artifact = worker.get("model_artifact_verification")
    strict_release = worker.get("strict_cuda_release")
    quant = identity.get("loaded_quantization") if isinstance(identity, Mapping) else None
    device_map = identity.get("hf_device_map") if isinstance(identity, Mapping) else None
    require(isinstance(artifact, Mapping) and artifact.get("passed") is True
            and artifact.get("model") == model
            and artifact.get("all_files_sha256_verified_immediately_before_load") is True
            and isinstance(artifact.get("file_count"), int) and artifact["file_count"] > 0
            and isinstance(artifact.get("weight_bytes"), int) and artifact["weight_bytes"] > 0
            and SHA_RE.fullmatch(str(artifact.get("model_manifest_sha256", ""))) is not None
            and SHA_RE.fullmatch(str(artifact.get("files_manifest_sha256", ""))) is not None
            and isinstance(identity, Mapping) and identity.get("model_key") == model
            and identity.get("weights_loaded") is True and identity.get("gpu_used") is True
            and identity.get("loaded_attn_implementation") == "sdpa"
            and identity.get("cuda_only_no_cpu_or_disk_offload") is True
            and isinstance(quant, Mapping) and quant.get("backend") == "bitsandbytes"
            and quant.get("load_in_8bit") is True
            and quant.get("non_quantized_dtype") == "torch.bfloat16"
            and quant.get("device_map") == "auto"
            and isinstance(device_map, Mapping) and bool(device_map)
            and all(str(value).startswith("cuda:") for value in device_map.values())
            and worker.get("model_released") is True
            and isinstance(strict_release, Mapping)
            and strict_release.get("cleanup_pass") is True
            and strict_release.get("allocated_after_release") == 0
            and strict_release.get("reserved_after_release") == 0
            and all(strict_release.get("steps", {}).get(step) is True for step in (
                "synchronize_before_cublas_clear", "cublas_workspaces_cleared",
                "final_allocator_cleanup"))
            and worker.get("cuda_allocated_after") == 0
            and worker.get("cuda_reserved_after") == 0
            and worker.get("truth_opened") is False
            and worker.get("internal_trace_authorized") is False,
            f"{raw_scope}/{model} loaded-model or release identity mismatch")
    raw_path = verify_file_seal(receipt.get("raw_artifact"), result_root, f"raw:{raw_scope}/{model}")
    manifest_seal = receipt.get("input_manifest")
    require(isinstance(manifest_seal, Mapping)
            and isinstance(manifest_seal.get("path"), str)
            and isinstance(manifest_seal.get("bytes"), int)
            and SHA_RE.fullmatch(str(manifest_seal.get("sha256", ""))) is not None,
            f"manifest:{raw_scope}/{model} seal malformed")
    require(int(receipt.get("row_count", -1)) == SCOPE_CONTRACT[
        {"primary": "public", "holdout": "holdout", "extension": "extension"}[raw_scope]
    ]["row_count"], f"{raw_scope}/{model} receipt row count mismatch")
    require(SHA_RE.fullmatch(str(receipt.get("record_ids_sha256", ""))) is not None,
            f"{raw_scope}/{model} missing record ID seal")
    return raw_path, deepcopy(dict(manifest_seal)), str(receipt["receipt_sha256"])


def _verify_cleanup_receipt(
    receipt: Mapping[str, Any], execution: Mapping[str, Any], raw_scope: str,
    model: str, model_index: int,
) -> None:
    verify_self_hash(receipt, "receipt_sha256", f"cleanup {raw_scope}/{model}")
    require(receipt.get("phase") == PHASE and receipt.get("experiment") == EXPERIMENT,
            f"cleanup {raw_scope}/{model} identity mismatch")
    require(receipt.get("scope") == raw_scope and receipt.get("model") == model,
            f"cleanup {raw_scope}/{model} scope/model mismatch")
    require(receipt.get("status") == "sealed" and receipt.get("model_order_index") == model_index,
            f"cleanup {raw_scope}/{model} not sealed/in order")
    require(receipt.get("cleanup_pass") is True and receipt.get("baseline_recovered") is True,
            f"cleanup {raw_scope}/{model} failed")
    require(receipt.get("run_id") == execution.get("run_id")
            and receipt.get("activation_sha256") == execution.get("activation_sha256")
            and receipt.get("worker_status_sha256") == execution.get("worker_status_sha256"),
            f"cleanup {raw_scope}/{model} execution binding mismatch")
    for field in ("model_released", "child_exit_zero", "cuda_allocated_zero",
                  "cuda_reserved_zero"):
        require(receipt.get(field) is True, f"cleanup {raw_scope}/{model} missing {field}")
    baseline = receipt.get("baseline_before")
    require(isinstance(baseline, Mapping), f"cleanup {raw_scope}/{model} baseline missing")
    for field in ("allocated_after", "reserved_after"):
        require(isinstance(receipt.get(field), int) and int(receipt[field]) >= 0,
                f"cleanup {raw_scope}/{model} invalid {field}")


def _manifest_ids(path: Path) -> tuple[list[str], Counter[str]]:
    ids: list[str] = []
    splits: Counter[str] = Counter()
    for row in jsonl_rows(path):
        record_id = row.get("record_id")
        require(isinstance(record_id, str) and record_id, f"manifest invalid record_id: {path}")
        ids.append(record_id)
        splits[str(row.get("split", ""))] += 1
    require(len(ids) == len(set(ids)), f"duplicate manifest IDs: {path}")
    return ids, splits


def _manifest_prompt_hashes(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in jsonl_rows(path):
        record_id = row.get("record_id")
        prompt = row.get("prompt")
        prompt_sha = row.get("prompt_sha256")
        require(isinstance(record_id, str) and record_id not in result,
                f"manifest prompt identity duplicate: {path}")
        require(isinstance(prompt, str) and isinstance(prompt_sha, str)
                and hashlib.sha256(prompt.encode("utf-8")).hexdigest() == prompt_sha,
                f"manifest prompt text/hash mismatch: {path}")
        result[record_id] = prompt_sha
    return result


def _expected_prompt_path(split: str) -> Path:
    relative = (f"runtime_prompts/public/{split}.jsonl" if split in PUBLIC_SPLITS
                else f"runtime_prompts/private/{split}.jsonl")
    admission = load_json(P991_ROOT / "gpu_admission_preregistration.json")
    verify_self_hash(admission, "gpu_admission_sha256", "Phase991 admission")
    seal = admission.get("artifact_seals", {}).get(relative)
    path = verify_file_seal(seal, P991_ROOT, f"Phase991 prompt:{split}")
    require(path.resolve() == (P991_ROOT / relative).resolve(), "Phase991 prompt path drift")
    return path


def expected_scope_ids(logical_scope: str) -> tuple[list[str], Counter[str]]:
    ids: list[str] = []
    counts: Counter[str] = Counter()
    for split in SCOPE_CONTRACT[logical_scope]["splits"]:
        rows, row_counts = _manifest_ids(_expected_prompt_path(split))
        ids.extend(rows)
        counts.update(row_counts)
    require(len(ids) == len(set(ids)), f"expected {logical_scope} prompt IDs overlap")
    return ids, counts


def _verify_holdout_chain(result_root: Path, receipts: Mapping[str, Mapping[str, Any]]) -> None:
    chain_path = result_root / "holdout_access" / "final_chain_receipt.json"
    chain = load_json(chain_path)
    verify_self_hash(chain, "receipt_sha256", "holdout access chain")
    require(chain.get("schema_version") == "phase992_holdout_access_chain_receipt.v1"
            and chain.get("phase") == PHASE and chain.get("status") == "complete",
            "holdout access chain identity/incomplete")
    require(chain.get("model_order") == list(MODEL_ORDER)
            and chain.get("event_count") == 2 * len(MODEL_ORDER)
            and chain.get("all_temporary_copies_revoked") is True,
            "holdout final-chain closure mismatch")
    require(not any((result_root / "temporary_holdout").glob("*.jsonl")),
            "holdout temporary copies still exist")
    event_paths = sorted((result_root / "holdout_access" / "events").glob("*.json"))
    require(len(event_paths) == 2 * len(MODEL_ORDER), "holdout event count mismatch")
    head = str(chain.get("genesis_head", ""))
    require(isinstance(head, str) and SHA_RE.fullmatch(head) is not None,
            "invalid holdout chain genesis")
    for model_index, model in enumerate(MODEL_ORDER):
        for offset, action in enumerate(("grant", "seal_and_revoke")):
            ordinal = 2 * model_index + offset
            event_path = event_paths[ordinal]
            require(event_path.name.startswith(f"{ordinal:04d}_"), "holdout event ordinal gap")
            event = load_json(event_path)
            require(event.get("model") == model and event.get("action") == action,
                    "holdout chain order/action mismatch")
            require(event.get("schema_version") == "phase992_holdout_access_event.v1"
                    and event.get("ordinal") == ordinal
                    and event.get("model_order_index") == model_index
                    and event.get("previous_head") == head,
                    "holdout event identity/link mismatch")
            body = dict(event)
            body.pop("new_head", None)
            expected_head = sha256_json(body)
            require(event.get("new_head") == expected_head, "holdout event head mismatch")
            if action == "seal_and_revoke":
                receipt_file_sha = sha256_file(
                    result_root / "receipts" / f"holdout_{model}.json"
                )
                require(event.get("output_receipt_sha256") == receipt_file_sha,
                        "holdout chain output receipt mismatch")
            head = expected_head
    require(chain.get("final_head") == head, "holdout final chain head mismatch")
    run_id = str(chain.get("run_id", ""))
    require(run_id and all(receipts[model].get("run_id") == run_id for model in MODEL_ORDER),
            "holdout chain/run receipt identity mismatch")
    for model_index, model in enumerate(MODEL_ORDER):
        grant = load_json(result_root / "holdout_access" / f"grant_{model_index:02d}_{model}.json")
        seal = load_json(result_root / "holdout_access" / f"seal_{model_index:02d}_{model}.json")
        verify_self_hash(grant, "receipt_sha256", f"holdout grant {model}")
        verify_self_hash(seal, "receipt_sha256", f"holdout seal {model}")
        require(grant.get("schema_version") == "phase992_holdout_grant_receipt.v1"
                and grant.get("run_id") == run_id and grant.get("model") == model
                and grant.get("model_order_index") == model_index
                and grant.get("status") == "granted", "holdout grant receipt mismatch")
        require(seal.get("schema_version") == "phase992_holdout_seal_receipt.v1"
                and seal.get("run_id") == run_id and seal.get("model") == model
                and seal.get("model_order_index") == model_index
                and seal.get("status") == "sealed"
                and seal.get("temporary_copy_revoked") is True
                and seal.get("output_receipt_sha256") == sha256_file(
                    result_root / "receipts" / f"holdout_{model}.json")
                and seal.get("cleanup_receipt_sha256") == sha256_file(
                    result_root / "receipts" / f"cleanup_holdout_{model}.json"),
                "holdout seal receipt mismatch")


def verify_release_barrier(
    result_root: Path, logical_scope: str,
) -> tuple[dict[str, Path], dict[str, Mapping[str, Any]], list[str], dict[str, str]]:
    """Verify all receipts before the caller can open any truth for scope."""
    require(logical_scope in SCOPE_CONTRACT, f"unknown scope: {logical_scope}")
    raw_scope = str(SCOPE_CONTRACT[logical_scope]["raw_scope"])
    raw_paths: dict[str, Path] = {}
    receipts: dict[str, Mapping[str, Any]] = {}
    manifest_seals: dict[str, dict[str, Any]] = {}
    previous: str | None = None
    for index, model in enumerate(MODEL_ORDER):
        receipt_path, cleanup_path = _receipt_paths(result_root, raw_scope, model)
        receipt = load_json(receipt_path)
        cleanup = load_json(cleanup_path)
        raw_path, manifest_seal, previous = _verify_execution_receipt(
            receipt, result_root, raw_scope, model, index, previous
        )
        _verify_cleanup_receipt(cleanup, receipt, raw_scope, model, index)
        raw_paths[model] = raw_path
        receipts[model] = receipt
        manifest_seals[model] = manifest_seal
    # No scope manifest is opened until every model execution+cleanup receipt
    # above has passed.  Holdout broker copies have already been revoked, so
    # their seals are compared to the immutable Phase991 source instead.
    manifest_id_reference: list[str] | None = None
    manifest_prompt_reference: dict[str, str] | None = None
    for model in MODEL_ORDER:
        if logical_scope in ("holdout", "extension"):
            split = str(SCOPE_CONTRACT[logical_scope]["splits"][0])
            admission = load_json(P991_ROOT / "gpu_admission_preregistration.json")
            verify_self_hash(admission, "gpu_admission_sha256", "Phase991 admission")
            expected_seal = admission["artifact_seals"][
                f"runtime_prompts/private/{split}.jsonl"
            ]
            require(manifest_seals[model]["bytes"] == expected_seal["bytes"]
                    and manifest_seals[model]["sha256"] == expected_seal["sha256"],
                    f"{raw_scope}/{model} private manifest seal mismatch")
            manifest_path = _expected_prompt_path(split)
        else:
            manifest_path = verify_file_seal(
                manifest_seals[model], result_root, f"manifest:{raw_scope}/{model}"
            )
        manifest_ids, manifest_splits = _manifest_ids(manifest_path)
        manifest_prompts = _manifest_prompt_hashes(manifest_path)
        require(len(manifest_ids) == SCOPE_CONTRACT[logical_scope]["row_count"],
                f"{raw_scope}/{model} manifest count mismatch")
        require(set(manifest_splits) == set(SCOPE_CONTRACT[logical_scope]["splits"]),
                f"{raw_scope}/{model} manifest split mismatch")
        require(sha256_json(sorted(manifest_ids)) == receipts[model].get("record_ids_sha256"),
                f"{raw_scope}/{model} receipt record ID hash mismatch")
        if manifest_id_reference is None:
            manifest_id_reference = manifest_ids
            manifest_prompt_reference = manifest_prompts
        else:
            require(set(manifest_ids) == set(manifest_id_reference),
                    f"{raw_scope} model manifests differ")
            require(manifest_prompts == manifest_prompt_reference,
                    f"{raw_scope} model prompt identities differ")
    require(manifest_id_reference is not None, "empty release barrier")
    expected_ids, expected_splits = expected_scope_ids(logical_scope)
    require(set(manifest_id_reference) == set(expected_ids),
            f"{logical_scope} manifest IDs differ from sealed Phase991 prompts")
    require(sum(expected_splits.values()) == SCOPE_CONTRACT[logical_scope]["row_count"],
            f"{logical_scope} expected prompt count mismatch")
    expected_prompt_hashes: dict[str, str] = {}
    for split in SCOPE_CONTRACT[logical_scope]["splits"]:
        expected_prompt_hashes.update(_manifest_prompt_hashes(_expected_prompt_path(split)))
    require(manifest_prompt_reference == expected_prompt_hashes,
            f"{logical_scope} manifest prompt identities differ from sealed Phase991 prompts")
    if logical_scope == "holdout":
        _verify_holdout_chain(result_root, receipts)
        public_ids, _ = expected_scope_ids("public")
        require(not (set(public_ids) & set(expected_ids)), "public/holdout prompt ID overlap")
        require(len(set(public_ids) | set(expected_ids)) == 10240,
                "cumulative primary manifest is not exact 10240 IDs")
    return raw_paths, receipts, expected_ids, expected_prompt_hashes


def verify_prior_score(result_root: Path, logical_scope: str) -> dict[str, Any]:
    if logical_scope == "holdout":
        path = result_root / SCOPE_CONTRACT["public"]["output"]
        required_field = "scope_behavior_pass"
    elif logical_scope == "extension":
        path = result_root / SCOPE_CONTRACT["holdout"]["output"]
        required_field = "primary_behavior_pass"
    else:
        raise RuntimeError("public scope has no prior score")
    report = load_json(path)
    verify_self_hash(report, "score_sha256", f"prior {logical_scope} score")
    require(report.get("passed") is True, f"prior score did not pass: {path}")
    models = report.get("models")
    require(isinstance(models, Mapping) and set(models) == set(MODEL_ORDER),
            "prior score model set mismatch")
    require(all(models[m].get(required_field) is True for m in MODEL_ORDER),
            f"prior score missing all-model {required_field}")
    return report


def parse_marker(text: str) -> dict[str, Any]:
    matches = [match.group(1).casefold() for match in MARKER_RE.finditer(text.casefold())]
    distinct = list(dict.fromkeys(matches))
    prediction = matches[0] if matches else None
    strict = STRICT_REFERENCE_RE.fullmatch(text.strip())
    return {
        "prediction": prediction,
        "markers": matches,
        "distinct_markers": distinct,
        "unparsed": prediction is None,
        "ambiguous": len(distinct) > 1,
        "strict_format": strict is not None and strict.group(1).casefold() == prediction,
    }


def _truth_rows(logical_scope: str) -> list[dict[str, Any]]:
    # This function is deliberately called only after verify_release_barrier.
    rows: list[dict[str, Any]] = []
    admission = load_json(P991_ROOT / "gpu_admission_preregistration.json")
    verify_self_hash(admission, "gpu_admission_sha256", "Phase991 admission")
    for split in SCOPE_CONTRACT[logical_scope]["splits"]:
        relative = f"scoring_truth/private/{split}.jsonl"
        path = verify_file_seal(admission.get("artifact_seals", {}).get(relative), P991_ROOT,
                                f"Phase991 truth:{split}")
        require(path.resolve() == (P991_ROOT / relative).resolve(), "Phase991 truth path drift")
        rows.extend(jsonl_rows(path))
    return rows


def _truth_seals(logical_scope: str) -> dict[str, Any]:
    admission = load_json(P991_ROOT / "gpu_admission_preregistration.json")
    verify_self_hash(admission, "gpu_admission_sha256", "Phase991 admission")
    return {split: deepcopy(admission["artifact_seals"][f"scoring_truth/private/{split}.jsonl"])
            for split in SCOPE_CONTRACT[logical_scope]["splits"]}


def _validate_raw_row(row: Mapping[str, Any], model: str, raw_scope: str) -> None:
    require(not (RAW_FORBIDDEN_TRUTH_KEYS & set(row)), "truth field leaked into raw row")
    require(row.get("schema_version") == "phase992_delayed_binding_raw.v1",
            "raw row schema mismatch")
    require(row.get("phase") == PHASE and row.get("experiment") == EXPERIMENT,
            "raw row identity mismatch")
    require(row.get("scope") == raw_scope and row.get("model") == model,
            "raw row scope/model mismatch")
    require(row.get("model_order_index") == MODEL_ORDER.index(model), "raw row model order mismatch")
    for field in ("record_id", "semantic_world_id", "split", "variant_id",
                  "generated_text", "termination_reason", "run_id"):
        require(isinstance(row.get(field), str), f"invalid raw field: {field}")
    for field in ("prompt_sha256", "input_manifest_sha256", "input_token_ids_sha256",
                  "teacher_forced_context_sha256", "generation_contract_sha256",
                  "activation_sha256"):
        require(isinstance(row.get(field), str) and SHA_RE.fullmatch(str(row[field])) is not None,
                f"invalid raw SHA field: {field}")
    input_ids = row.get("input_token_ids")
    require(isinstance(input_ids, list) and input_ids
            and all(isinstance(token, int) for token in input_ids)
            and row.get("input_token_count") == len(input_ids)
            and row.get("input_token_ids_sha256") == sha256_json(input_ids),
            "input token identity mismatch")
    require(isinstance(row.get("eos_seen"), bool) and
            isinstance(row.get("budget_exhausted"), bool), "invalid EOS/budget flags")
    ids = row.get("generated_token_ids_before_eos")
    require(isinstance(ids, list) and all(isinstance(token, int) for token in ids),
            "invalid generated_token_ids_before_eos")
    require(len(ids) <= 24, "generation exceeded frozen budget")
    suffix = row.get("generated_suffix_token_ids")
    effective_eos = row.get("effective_eos_token_ids")
    require(isinstance(suffix, list) and len(suffix) <= 24
            and all(isinstance(token, int) for token in suffix),
            "invalid generated suffix")
    require(isinstance(effective_eos, list) and effective_eos
            and all(isinstance(token, int) for token in effective_eos),
            "invalid effective EOS set")
    if row["eos_seen"]:
        first_index = row.get("first_eos_index")
        require(isinstance(first_index, int) and 0 <= first_index < len(suffix),
                "invalid first EOS index")
        require(suffix[first_index] in effective_eos
                and row.get("first_eos_token_id") == suffix[first_index]
                and ids == suffix[:first_index], "EOS prefix accounting mismatch")
        require(row["budget_exhausted"] is False, "EOS and budget cannot both terminate")
    else:
        require(row.get("first_eos_index") is None and row.get("first_eos_token_id") is None
                and ids == suffix and len(ids) == 24 and row["budget_exhausted"] is True,
                "non-EOS budget accounting mismatch")
    candidates = row.get("teacher_forced_candidates")
    require(isinstance(candidates, Mapping) and tuple(row.get("teacher_forced_candidate_order", ())) == VALUES,
            "teacher-forced candidate contract mismatch")
    require(set(candidates) == set(VALUES), "teacher-forced candidate set mismatch")
    for value in VALUES:
        item = candidates[value]
        require(isinstance(item, Mapping), "teacher candidate must be object")
        logit = item.get("logit")
        require(isinstance(logit, (int, float)) and math.isfinite(float(logit)), "nonfinite logit")
        require(item.get("logit_hex") == float(logit).hex(), "logit/hex mismatch")
        logprob = item.get("logprob")
        require(isinstance(logprob, (int, float)) and math.isfinite(float(logprob)), "nonfinite logprob")
        require(item.get("logprob_hex") == float(logprob).hex(), "logprob/hex mismatch")


def _variant_parts(variant_id: str) -> tuple[str, str, str, str]:
    parts = tuple(variant_id.split("__"))
    require(len(parts) == 4 and parts[0] in SEMANTIC_CONDITIONS,
            f"invalid variant_id: {variant_id}")
    return parts  # type: ignore[return-value]


def _teacher_diagnostic(raw: Mapping[str, Any], gold: str) -> dict[str, Any]:
    logits = {value: float(raw["teacher_forced_candidates"][value]["logit"]) for value in VALUES}
    target = logits[gold]
    best_foil = max(logits[value] for value in VALUES if value != gold)
    margin = target - best_foil
    return {
        "target_logit_hex": target.hex(),
        "best_foil_logit_hex": best_foil.hex(),
        "margin_hex": margin.hex(),
        "positive": margin > 0.0,
        "tie": margin == 0.0,
    }


def score_model_rows(
    model: str, logical_scope: str, raw_rows: Sequence[Mapping[str, Any]],
    truth_rows: Sequence[Mapping[str, Any]], expected_ids: Sequence[str],
) -> dict[str, Any]:
    raw_scope = str(SCOPE_CONTRACT[logical_scope]["raw_scope"])
    require(len(raw_rows) == len(truth_rows) == len(expected_ids), f"{model} row count mismatch")
    raw_by_id: dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        _validate_raw_row(row, model, raw_scope)
        record_id = str(row["record_id"])
        require(record_id not in raw_by_id, f"{model} duplicate raw ID")
        raw_by_id[record_id] = row
    truth_by_id = {str(row["record_id"]): row for row in truth_rows}
    require(len(truth_by_id) == len(truth_rows), "duplicate truth ID")
    require(set(raw_by_id) == set(truth_by_id) == set(expected_ids),
            f"{model} raw/truth/manifest ID mismatch")

    total = correct = unparsed = ambiguous = strict_format = eos = budget = 0
    major: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    cells: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    predictions: dict[tuple[str, str, str, str], dict[str, str | None]] = defaultdict(dict)
    teacher_margins: list[float] = []
    teacher_positive = teacher_tie = 0
    evidence: list[dict[str, Any]] = []
    for record_id in sorted(raw_by_id):
        raw = raw_by_id[record_id]
        truth = truth_by_id[record_id]
        for field in ("record_id", "semantic_world_id", "split", "variant_id", "split_ordinal"):
            require(raw.get(field) == truth.get(field), f"{model}/{record_id} metadata mismatch: {field}")
        parsed = parse_marker(str(raw["generated_text"]))
        gold = str(truth["gold_value"])
        require(gold in VALUES, "invalid truth value")
        semantic, paraphrase, order, horizon = _variant_parts(str(truth["variant_id"]))
        is_correct = parsed["prediction"] == gold
        total += 1
        correct += int(is_correct)
        unparsed += int(parsed["unparsed"])
        ambiguous += int(parsed["ambiguous"])
        strict_format += int(parsed["strict_format"])
        eos += int(raw["eos_seen"])
        budget += int(raw["budget_exhausted"])
        major[semantic][0] += int(is_correct)
        major[semantic][1] += 1
        cell_key = f"{truth['split']}|{truth['variant_id']}"
        cells[cell_key][0] += int(is_correct)
        cells[cell_key][1] += 1
        nuisance_key = (str(truth["semantic_world_id"]), paraphrase, order, horizon)
        require(semantic not in predictions[nuisance_key], "duplicate counterfactual cell")
        predictions[nuisance_key][semantic] = parsed["prediction"]
        teacher = _teacher_diagnostic(raw, gold)
        margin = float.fromhex(teacher["margin_hex"])
        teacher_margins.append(margin)
        teacher_positive += int(teacher["positive"])
        teacher_tie += int(teacher["tie"])
        evidence.append({
            "record_id": record_id,
            "prediction": parsed["prediction"],
            "markers": parsed["markers"],
            "correct": is_correct,
            "unparsed": parsed["unparsed"],
            "ambiguous": parsed["ambiguous"],
            "strict_format": parsed["strict_format"],
            "eos_seen": raw["eos_seen"],
            "budget_exhausted": raw["budget_exhausted"],
            "termination_reason": raw["termination_reason"],
            "teacher": teacher,
        })

    pair_denominator = pair_sensitive = pair_joint_correct = 0
    pair_axis: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    correct_by_id = {row["record_id"]: row["correct"] for row in evidence}
    for nuisance_key, condition_predictions in predictions.items():
        require(set(condition_predictions) == set(SEMANTIC_CONDITIONS),
                f"incomplete counterfactual closure: {nuisance_key}")
        for left_index, left in enumerate(SEMANTIC_CONDITIONS):
            for right in SEMANTIC_CONDITIONS[left_index + 1:]:
                left_prediction = condition_predictions[left]
                right_prediction = condition_predictions[right]
                sensitive = (left_prediction is not None and right_prediction is not None
                             and left_prediction != right_prediction)
                pair_denominator += 1
                pair_sensitive += int(sensitive)
                pair_axis[f"{left}|{right}"][0] += int(sensitive)
                pair_axis[f"{left}|{right}"][1] += 1
        # Joint correctness is reconstructed from truth peer IDs, independently
        # of prediction-change sensitivity.  It is diagnostic, not the gate.
    pair_joint_correct = 0  # retained explicitly; computed below by peer sets
    seen_pairs: set[tuple[str, str]] = set()
    for truth in truth_rows:
        peers = truth.get("semantic_peer_record_ids")
        require(isinstance(peers, Mapping) and set(peers) == set(SEMANTIC_CONDITIONS),
                "truth counterfactual peer registry invalid")
        for peer_id in peers.values():
            pair = tuple(sorted((str(truth["record_id"]), str(peer_id))))
            if pair[0] == pair[1] or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            pair_joint_correct += int(bool(correct_by_id[pair[0]]) and bool(correct_by_id[pair[1]]))
    require(len(seen_pairs) == pair_denominator, "peer registry/pair grouping disagree")

    def ratio(correct_count: int, denominator: int, threshold: int | None = None) -> dict[str, Any]:
        require(denominator > 0, "zero denominator")
        result = {
            "correct": correct_count,
            "denominator": denominator,
            "percent": 100.0 * correct_count / denominator,
        }
        if threshold is not None:
            result["threshold_percent"] = threshold
            result["passed"] = 100 * correct_count >= threshold * denominator
        return result

    major_report = {
        key: ratio(value[0], value[1], THRESHOLDS["each_major_condition_min_percent"])
        for key, value in sorted(major.items())
    }
    cell_report = {
        key: ratio(value[0], value[1], THRESHOLDS["each_split_variant_min_percent"])
        for key, value in sorted(cells.items())
    }
    require(set(major_report) == set(SEMANTIC_CONDITIONS), "major-condition coverage mismatch")
    require(len(cell_report) == len(SCOPE_CONTRACT[logical_scope]["splits"]) * VARIANT_COUNT,
            "split x 32-variant coverage mismatch")
    counterfactual = ratio(
        pair_sensitive, pair_denominator,
        THRESHOLDS["counterfactual_pair_sensitivity_min_percent"],
    )
    counterfactual["definition"] = "both predictions parsed and different over all six unordered semantic pairs per matched nuisance cell"
    counterfactual["joint_semantic_correct_pairs"] = pair_joint_correct
    counterfactual["joint_semantic_correct_percent"] = 100.0 * pair_joint_correct / pair_denominator
    counterfactual["by_condition_pair"] = {
        key: ratio(value[0], value[1]) for key, value in sorted(pair_axis.items())
    }
    teacher_sorted = sorted(teacher_margins)
    behavior_pass = (
        all(row["passed"] for row in major_report.values())
        and all(row["passed"] for row in cell_report.values())
        and counterfactual["passed"]
    )
    gate_counts = {
        "major_conditions": {key: {"correct": row["correct"], "denominator": row["denominator"]}
                             for key, row in major_report.items()},
        "split_variants": {key: {"correct": row["correct"], "denominator": row["denominator"]}
                           for key, row in cell_report.items()},
        "counterfactual": {"correct": pair_sensitive, "denominator": pair_denominator},
    }
    return {
        "model": model,
        "scope": logical_scope,
        "scope_behavior_pass": behavior_pass,
        "natural_generation": {
            "overall": ratio(correct, total),
            "major_conditions": major_report,
            "split_variants": cell_report,
            "counterfactual_pair_sensitivity": counterfactual,
            "separate_accounts": {
                "unparsed": unparsed, "ambiguity": ambiguous,
                "strict_reference_format": strict_format,
                "eos_seen": eos, "budget_exhausted": budget,
                "denominator": total,
            },
        },
        "teacher_forced_diagnostic": {
            "role": "diagnostic_only_not_a_natural_generation_gate",
            "definition": "target first-continuation logit minus maximum of three foil logits",
            "denominator": total,
            "positive": teacher_positive,
            "tie": teacher_tie,
            "positive_percent": 100.0 * teacher_positive / total,
            "mean_margin": math.fsum(teacher_margins) / total,
            "median_margin": teacher_sorted[total // 2] if total % 2 else
                (teacher_sorted[total // 2 - 1] + teacher_sorted[total // 2]) / 2.0,
            "minimum_margin": teacher_sorted[0],
            "maximum_margin": teacher_sorted[-1],
        },
        "gate_counts": gate_counts,
        "case_evidence_count": len(evidence),
        "case_evidence_sha256": sha256_json(evidence),
        "scientific_scope": {
            "behavior_evidence_only": True,
            "internal_structure_evidence": False,
            "causal_mechanism_evidence": False,
        },
    }


def _merge_gate_counts(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    def merge_map(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
        keys = set(a) | set(b)
        return {key: {
            "correct": int(a.get(key, {}).get("correct", 0)) + int(b.get(key, {}).get("correct", 0)),
            "denominator": int(a.get(key, {}).get("denominator", 0)) + int(b.get(key, {}).get("denominator", 0)),
        } for key in sorted(keys)}
    return {
        "major_conditions": merge_map(left["major_conditions"], right["major_conditions"]),
        "split_variants": merge_map(left["split_variants"], right["split_variants"]),
        "counterfactual": {
            "correct": int(left["counterfactual"]["correct"]) + int(right["counterfactual"]["correct"]),
            "denominator": int(left["counterfactual"]["denominator"]) + int(right["counterfactual"]["denominator"]),
        },
    }


def _gate_from_counts(counts: Mapping[str, Any]) -> dict[str, Any]:
    def convert(items: Mapping[str, Any], threshold: int) -> dict[str, Any]:
        return {key: {
            **value,
            "percent": 100.0 * int(value["correct"]) / int(value["denominator"]),
            "threshold_percent": threshold,
            "passed": 100 * int(value["correct"]) >= threshold * int(value["denominator"]),
        } for key, value in sorted(items.items())}
    major = convert(counts["major_conditions"], 90)
    cells = convert(counts["split_variants"], 85)
    cf = counts["counterfactual"]
    cf_report = {
        **cf, "percent": 100.0 * int(cf["correct"]) / int(cf["denominator"]),
        "threshold_percent": 85,
        "passed": 100 * int(cf["correct"]) >= 85 * int(cf["denominator"]),
    }
    return {
        "major_conditions": major,
        "split_variants": cells,
        "counterfactual_pair_sensitivity": cf_report,
        "passed": all(v["passed"] for v in major.values())
            and all(v["passed"] for v in cells.values()) and cf_report["passed"],
    }


def discovery_fit_baseline(all_primary_truth: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    representative = [row for row in all_primary_truth
                      if row["paraphrase_id"] == "standard"
                      and row["fact_order_id"] == "order_a"
                      and row["horizon_id"] == "near"]
    training = [row for row in representative if row["split"] == "discovery"]
    counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for row in training:
        key = (str(row["query_entity"]), str(row["gold_object"]), str(row["query_relation"]))
        counts[key][str(row["gold_value"])] += 1
        global_counts[str(row["gold_value"])] += 1
    order = {value: index for index, value in enumerate(VALUES)}
    choose = lambda counter: min(VALUES, key=lambda v: (-counter[v], order[v]))
    fallback = choose(global_counts)
    table = {key: choose(counter) for key, counter in counts.items()}
    evaluations: dict[str, Any] = {}
    for split in ALL_PRIMARY_SPLITS:
        rows = [row for row in representative if row["split"] == split]
        correct = seen = 0
        for row in rows:
            key = (str(row["query_entity"]), str(row["gold_object"]), str(row["query_relation"]))
            prediction = table.get(key, fallback)
            correct += int(prediction == row["gold_value"])
            seen += int(key in table)
        evaluations[split] = {
            "correct": correct, "denominator": len(rows),
            "accuracy_percent": 100.0 * correct / len(rows),
            "seen_feature_rows": seen, "unseen_feature_rows": len(rows) - seen,
        }
    return {
        "role": "first-hop-resolved selected-object plus relation lookup baseline",
        "not_a_pure_surface_baseline": True,
        "fit_split": "discovery",
        "training_rows": len(training),
        "feature": "query_entity+resolved_selected_object+query_relation",
        "tie_break": "fixed VALUES order red,blue,green,black",
        "unseen_fallback": fallback,
        "table_sha256": sha256_json({"|".join(k): v for k, v in sorted(table.items())}),
        "evaluations": evaluations,
        "not_mechanism_evidence": True,
    }


def shortcut_baselines(all_primary_truth: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frozen = _frozen_phase991_baselines()
    return {
        "oracle_structure_baseline": deepcopy(frozen["oracle_structure_baseline"]),
        "discovery_fitted_lookup_baseline": discovery_fit_baseline(all_primary_truth),
        "behavior_above_either_baseline_does_not_prove_two_hop_mechanism": True,
        "source_phase991_gpu_admission_sha256": frozen["source_phase991_gpu_admission_sha256"],
    }


def _seal_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = deepcopy(dict(payload))
    body["score_sha256"] = sha256_json(body)
    return body


def _exclusive_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite score: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _frozen_phase991_baselines() -> dict[str, Any]:
    admission = load_json(P991_ROOT / "gpu_admission_preregistration.json")
    verify_self_hash(admission, "gpu_admission_sha256", "Phase991 admission")
    baselines = admission.get("shortcut_baselines")
    require(isinstance(baselines, Mapping)
            and "oracle_structure_baseline" in baselines
            and "discovery_fitted_lookup_baseline" in baselines,
            "Phase991 frozen baselines missing")
    result = deepcopy(dict(baselines))
    result["source_phase991_gpu_admission_sha256"] = admission["gpu_admission_sha256"]
    result["reported_as_behavior_comparator_not_mechanism_evidence"] = True
    return result


def _verify_extension_release(result_root: Path, activation: Mapping[str, Any],
                              prior: Mapping[str, Any]) -> dict[str, Any]:
    path = result_root / "extension_behavior_release.json"
    release = load_json(path)
    verify_self_hash(release, "receipt_sha256", "extension behavior release")
    require(release.get("schema_version") == "phase992_extension_behavior_release.v1"
            and release.get("phase") == PHASE
            and release.get("experiment") == EXPERIMENT,
            "extension release identity mismatch")
    require(release.get("extension_behavior_execution_authorized") is True
            and release.get("all_models_primary_pass") is True,
            "extension execution remains fail-closed")
    require(release.get("activation_sha256") == activation["activation_sha256"]
            and release.get("holdout_score_sha256") == prior["score_sha256"],
            "extension release binding mismatch")
    return release


def _public_admission(
    result_root: Path, report: Mapping[str, Any], activation: Mapping[str, Any],
    run_id: str,
) -> dict[str, Any]:
    score_path = result_root / SCOPE_CONTRACT["public"]["output"]
    require(score_path.is_file(), "public score must seal before admission")
    all_pass = bool(report["passed"])
    payload = {
        "schema_version": "phase992_public_behavior_admission.v1",
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "role": "public_behavior_gate_to_sealed_holdout_access",
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "model_order": list(MODEL_ORDER),
        "all_models_public_pass": all_pass,
        "sealed_holdout_model_access_authorized": all_pass,
        "activation_sha256": activation["activation_sha256"],
        "truth_artifact_seals": _truth_seals("public"),
        "public_score": {
            "path": str(score_path.relative_to(result_root)).replace("\\", "/"),
            "bytes": score_path.stat().st_size,
            "sha256": sha256_file(score_path),
            "score_sha256": report["score_sha256"],
        },
        "behavior_only": True,
        "internal_trace_authorized": False,
    }
    payload["admission_sha256"] = sha256_json(payload)
    return payload


def score_scope(
    result_root: Path, logical_scope: str, write: bool,
    activation_path: Path = ACTIVATION,
) -> dict[str, Any]:
    activation = load_activation(activation_path, result_root)
    prior: dict[str, Any] | None = None
    if logical_scope != "public":
        prior = verify_prior_score(result_root, logical_scope)
    if logical_scope == "extension":
        assert prior is not None
        _verify_extension_release(result_root, activation, prior)
    # The barrier completes before the first call to _truth_rows.
    raw_paths, receipts, expected_ids, expected_prompt_hashes = verify_release_barrier(
        result_root, logical_scope
    )
    run_ids = {str(receipts[model].get("run_id", "")) for model in MODEL_ORDER}
    require(len(run_ids) == 1 and "" not in run_ids, "scope receipts do not share one run_id")
    run_id = next(iter(run_ids))
    truth = _truth_rows(logical_scope)
    require(len(truth) == SCOPE_CONTRACT[logical_scope]["row_count"], "truth count mismatch")
    models: dict[str, Any] = {}
    for model in MODEL_ORDER:
        raw_rows = list(jsonl_rows(raw_paths[model]))
        require(receipts[model].get("activation_sha256") == activation["activation_sha256"]
                and receipts[model].get("generation_contract_sha256")
                    == activation["generation_contract_sha256"],
                f"{model} receipt activation/generation binding mismatch")
        for row in raw_rows:
            require(row.get("activation_sha256") == activation["activation_sha256"]
                    and row.get("generation_contract_sha256")
                        == activation["generation_contract_sha256"]
                    and row.get("input_manifest_sha256")
                        == receipts[model]["input_manifest"]["sha256"],
                    f"{model} raw activation/manifest binding mismatch")
            require(row.get("run_id") == receipts[model]["run_id"]
                    and row.get("prompt_sha256")
                        == expected_prompt_hashes.get(str(row.get("record_id", ""))),
                    f"{model} raw run/prompt identity mismatch")
        models[model] = score_model_rows(model, logical_scope, raw_rows, truth, expected_ids)
        models[model]["raw_receipt_sha256"] = receipts[model]["receipt_sha256"]

    if logical_scope == "public":
        for model in MODEL_ORDER:
            models[model]["primary_behavior_pass"] = False
        # Report the already-frozen Phase991 baselines.  This reads no raw
        # holdout output and opens no private truth.
        baselines: Mapping[str, Any] = _frozen_phase991_baselines()
    elif logical_scope == "holdout":
        assert prior is not None
        for model in MODEL_ORDER:
            merged = _merge_gate_counts(prior["models"][model]["gate_counts"],
                                        models[model]["gate_counts"])
            models[model]["cumulative_primary_gate"] = _gate_from_counts(merged)
            models[model]["primary_behavior_pass"] = models[model]["cumulative_primary_gate"]["passed"]
        public_truth = _truth_rows("public")
        baselines = shortcut_baselines([*public_truth, *truth])
    else:
        for model in MODEL_ORDER:
            models[model]["primary_behavior_pass"] = True
        baselines = {
            "status": "inherited_from_sealed_holdout_score",
            "source_score_sha256": prior["score_sha256"] if prior else None,
            "not_mechanism_evidence": True,
        }
    passed = all(model["scope_behavior_pass"] for model in models.values())
    if logical_scope == "holdout":
        passed = passed and all(model["primary_behavior_pass"] for model in models.values())
    report = _seal_report({
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "role": "sealed_truth_released_behavior_score",
        "created_at_utc": utc_now(),
        "scope": logical_scope,
        "run_id": run_id,
        "passed": passed,
        "activation_sha256": activation["activation_sha256"],
        "truth_artifact_seals": _truth_seals(logical_scope),
        "thresholds": deepcopy(THRESHOLDS),
        "equivalence_rule": {
            "normalization": "casefold; frozen ASCII word-boundary marker regex",
            "marker_regex": MARKER_RE.pattern,
            "primary_prediction": "first complete marker in generated text before EOS",
            "multiple_distinct_markers": "first scores semantics; ambiguity separate",
            "strict_format_eos_budget": "separate accounts",
        },
        "models": models,
        "shortcut_baselines": baselines,
        "scientific_adjudication": {
            "behavior_only": True,
            "teacher_forcing_is_diagnostic_only": True,
            "baseline_outperformance_is_not_mechanism_evidence": True,
            "internal_structure_discovered": False,
            "mechanism_formula_authorized": False,
        },
    })
    if write:
        _exclusive_write_json(result_root / SCOPE_CONTRACT[logical_scope]["output"], report)
        if logical_scope == "public":
            admission = _public_admission(result_root, report, activation, run_id)
            _exclusive_write_json(result_root / "public_behavior_admission.json", admission)
    return report


def self_test() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["first_complete_marker"] = parse_marker("Blue, then red.")["prediction"] == "blue"
    checks["distinct_ambiguity"] = parse_marker("Blue, then red.")["ambiguous"] is True
    checks["repeat_not_ambiguous"] = parse_marker("blue BLUE")["ambiguous"] is False
    checks["ascii_boundary_rejects_embedded"] = parse_marker("blueberry") ["unparsed"] is True
    checks["strict_format"] = parse_marker("The retrieved marker is green.")["strict_format"] is True
    checks["format_separate"] = parse_marker("green")["strict_format"] is False
    counts = {
        "major_conditions": {condition: {"correct": 9, "denominator": 10}
                             for condition in SEMANTIC_CONDITIONS},
        "split_variants": {f"s|v{i}": {"correct": 17, "denominator": 20}
                           for i in range(32)},
        "counterfactual": {"correct": 17, "denominator": 20},
    }
    checks["inclusive_integer_thresholds"] = _gate_from_counts(counts)["passed"] is True
    bad = deepcopy(counts)
    bad["major_conditions"]["original"] = {"correct": 89, "denominator": 100}
    checks["major_fail_closed"] = _gate_from_counts(bad)["passed"] is False
    checks["scope_counts"] = (SCOPE_CONTRACT["public"]["row_count"]
                              + SCOPE_CONTRACT["holdout"]["row_count"] == 10240)
    checks["extension_count"] = SCOPE_CONTRACT["extension"]["row_count"] == 4096
    checks["three_stage_release"] = tuple(SCOPE_CONTRACT) == ("public", "holdout", "extension")
    synthetic_raw: list[dict[str, Any]] = []
    synthetic_truth: list[dict[str, Any]] = []
    for split_ordinal, split in enumerate(PUBLIC_SPLITS):
        world = f"self_{split}"
        for paraphrase in ("standard", "paraphrase"):
            for order in ("order_a", "order_b"):
                for horizon in ("near", "far"):
                    peer_ids = {
                        condition: f"{world}_{condition}_{paraphrase}_{order}_{horizon}"
                        for condition in SEMANTIC_CONDITIONS
                    }
                    for condition_index, condition in enumerate(SEMANTIC_CONDITIONS):
                        record_id = peer_ids[condition]
                        gold = VALUES[condition_index]
                        variant = f"{condition}__{paraphrase}__{order}__{horizon}"
                        candidates = {}
                        for value in VALUES:
                            logit = 1.0 if value == gold else 0.0
                            candidates[value] = {
                                "continuation": f" {value}", "token_id": VALUES.index(value),
                                "logit": logit, "logit_hex": logit.hex(),
                                "logprob": logit, "logprob_hex": logit.hex(),
                            }
                        synthetic_raw.append({
                            "schema_version": "phase992_delayed_binding_raw.v1",
                            "phase": PHASE, "experiment": EXPERIMENT, "scope": "primary",
                            "model": "qwen3", "model_order_index": 0, "run_id": "self",
                            "record_id": record_id, "semantic_world_id": world, "split": split,
                            "split_ordinal": split_ordinal, "variant_id": variant,
                            "prompt_sha256": "a" * 64, "input_manifest_sha256": "b" * 64,
                            "input_token_ids": [7, 8], "input_token_count": 2,
                            "input_token_ids_sha256": sha256_json([7, 8]),
                            "teacher_forced_context_sha256": "c" * 64,
                            "generation_contract_sha256": "d" * 64,
                            "activation_sha256": "e" * 64,
                            "generated_text": f"The retrieved marker is {gold}.",
                            "termination_reason": "eos", "eos_seen": True,
                            "budget_exhausted": False,
                            "generated_suffix_token_ids": [1, 2],
                            "generated_token_ids_before_eos": [1],
                            "effective_eos_token_ids": [2], "first_eos_index": 1,
                            "first_eos_token_id": 2,
                            "teacher_forced_candidate_order": list(VALUES),
                            "teacher_forced_candidates": candidates,
                        })
                        synthetic_truth.append({
                            "record_id": record_id, "semantic_world_id": world, "split": split,
                            "split_ordinal": split_ordinal, "variant_id": variant,
                            "gold_value": gold, "semantic_peer_record_ids": peer_ids,
                        })
    synthetic_ids = [row["record_id"] for row in synthetic_truth]
    synthetic_report = score_model_rows(
        "qwen3", "public", synthetic_raw, synthetic_truth, synthetic_ids
    )
    checks["full_scoring_path"] = (
        synthetic_report["scope_behavior_pass"] is True
        and synthetic_report["case_evidence_count"] == 96
        and synthetic_report["natural_generation"]["counterfactual_pair_sensitivity"]["percent"]
            == 100.0
    )
    with tempfile.TemporaryDirectory(prefix="phase992-scorer-selftest-") as raw:
        temporary_root = Path(raw)
        score_path = temporary_root / SCOPE_CONTRACT["public"]["output"]
        score_path.parent.mkdir(parents=True)
        score_path.write_text("{}\n", encoding="utf-8")
        admission = _public_admission(
            temporary_root,
            {"passed": True, "score_sha256": "f" * 64},
            {"activation_sha256": "e" * 64},
            "self-test",
        )
        checks["public_admission_path"] = (
            admission["schema_version"] == "phase992_public_behavior_admission.v1"
            and admission["all_models_public_pass"] is True
            and set(admission["truth_artifact_seals"]) == set(PUBLIC_SPLITS)
        )
    checks["no_model_runtime_import"] = "torch" not in sys.modules and "transformers" not in sys.modules
    require(all(checks.values()), f"self-test failed: {checks}")
    return {
        "phase": PHASE, "schema_version": SCHEMA_VERSION,
        "role": "pure_cpu_no_truth_no_write_self_test", "passed": True,
        "checks": checks, "cuda_used": False, "truth_opened": False,
        "files_written": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--score-public", action="store_true")
    group.add_argument("--score-holdout", action="store_true")
    group.add_argument("--score-extension", action="store_true")
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--activation-path", type=Path, default=ACTIVATION)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        report = self_test()
    else:
        scope = "public" if args.score_public else "holdout" if args.score_holdout else "extension"
        report = score_scope(
            args.result_root.resolve(), scope, write=not args.no_write,
            activation_path=args.activation_path.resolve(),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
