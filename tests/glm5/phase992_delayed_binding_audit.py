#!/usr/bin/env python3
"""Independent CPU audit for Phase992 delayed-binding behavior scores.

The implementation intentionally does not import the Phase992 scorer and does
not call any of its functions.  It independently reopens sealed raw artifacts
and private truth only after rebuilding the requested scope's receipt barrier,
then recomputes every case, every gate, the counterfactual pairs, teacher-forced
margin diagnostics, baseline hashes, and report seals.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import ast
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


PHASE = 992
SCHEMA_VERSION = "phase992_delayed_binding_independent_audit.v1"
EXPERIMENT = "delayed_two_hop_gpu_behavior"
ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
PROTOCOL_ROOT = GLM5 / "result" / "phase992_delayed_binding_behavior_protocol"
EXECUTION_ROOT = GLM5 / "result" / "phase992_delayed_binding_behavior_execution"
P991_ROOT = GLM5 / "result" / "phase991_delayed_binding_gpu_admission"
ACTIVATION = PROTOCOL_ROOT / "activation.json"
MODEL_ORDER = ("qwen3", "glm4", "deepseek7b")
VALUES = ("red", "blue", "green", "black")
CONDITIONS = ("original", "value_swap", "binding_swap", "relation_swap")
PUBLIC_SPLITS = ("discovery", "confirmation", "adversarial")
SCOPE = {
    "public": {"raw": "primary", "splits": PUBLIC_SPLITS, "count": 8192,
               "score": "scores/public_score.json"},
    "holdout": {"raw": "holdout", "splits": ("sealed_holdout",), "count": 2048,
                "score": "scores/holdout_score.json"},
    "extension": {"raw": "extension", "splits": ("expanded_confirmation",), "count": 4096,
                  "score": "scores/extension_score.json"},
}
MARKER = re.compile(r"(?<![A-Za-z])(red|blue|green|black)(?![A-Za-z])", re.I)
STRICT = re.compile(r"^The retrieved marker is (red|blue|green|black)\.$", re.I)
SHA = re.compile(r"^[0-9a-f]{64}$")
TRUTH_FORBIDDEN = {"gold", "gold_value", "gold_object", "answer_value", "target",
                   "foil", "foil_values", "query_entity", "query_relation",
                   "semantic_peer_record_ids"}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canon(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def json_hash(value: Any) -> str:
    return hashlib.sha256(canon(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    need(path.is_file() and not path.is_symlink(), f"missing/aliased JSON: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(result, dict), f"non-object JSON: {path}")
    return result


def rows(path: Path) -> Iterable[dict[str, Any]]:
    need(path.is_file() and not path.is_symlink(), f"missing/aliased JSONL: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            need(line.endswith("\n"), f"unterminated line {path}:{line_number}")
            item = json.loads(line)
            need(isinstance(item, dict), f"non-object row {path}:{line_number}")
            yield item


def self_hash(document: Mapping[str, Any], field: str, label: str) -> None:
    expected = document.get(field)
    need(isinstance(expected, str) and SHA.fullmatch(expected) is not None,
         f"{label} invalid {field}")
    body = deepcopy(dict(document))
    body.pop(field, None)
    need(json_hash(body) == expected, f"{label} self-hash mismatch")


def locate(path_text: str, execution_root: Path, protocol_root: Path) -> Path:
    path = Path(path_text)
    options = [path] if path.is_absolute() else [execution_root / path, protocol_root / path,
                                                 P991_ROOT / path, ROOT / path]
    selected = next((item for item in options if item.exists()), None)
    need(selected is not None, f"sealed path absent: {path_text}")
    resolved = selected.resolve(strict=True)
    need(resolved == ROOT.resolve() or ROOT.resolve() in resolved.parents, "path escaped workspace")
    need(not selected.is_symlink(), f"sealed path is symlink: {selected}")
    return selected


def seal_path(seal: Mapping[str, Any], execution_root: Path, protocol_root: Path,
              label: str) -> Path:
    need(isinstance(seal, Mapping), f"{label} missing seal")
    path = locate(str(seal.get("path", "")), execution_root, protocol_root)
    need(int(seal.get("bytes", -1)) == path.stat().st_size, f"{label} bytes mismatch")
    need(SHA.fullmatch(str(seal.get("sha256", ""))) is not None, f"{label} SHA invalid")
    need(file_hash(path) == seal["sha256"], f"{label} SHA mismatch")
    return path


def activation_check(activation_path: Path, execution_root: Path) -> dict[str, Any]:
    value = read_json(activation_path)
    self_hash(value, "activation_sha256", "activation")
    need(value.get("schema_version") == "phase992_gpu_behavior_activation.v1", "activation schema")
    need(value.get("phase") == PHASE and value.get("experiment") == EXPERIMENT, "activation identity")
    need(tuple(value.get("model_order", ())) == MODEL_ORDER, "activation model order")
    formal = value.get("formal_python")
    need(isinstance(formal, Mapping), "formal Python identity absent")
    formal_path = Path(str(formal.get("path", ""))).resolve(strict=True)
    need(Path(sys.executable).resolve(strict=True) == formal_path
         and file_hash(formal_path) == formal.get("sha256"), "formal Python identity drift")
    need(value.get("gpu_behavior_execution_authorized") is True
         and value.get("behavior_only_authorized") is True, "behavior not authorized")
    need(value.get("internal_trace_authorized") is False
         and value.get("causal_intervention_authorized") is False
         and value.get("mechanism_formula_authorized") is False, "forbidden activation")
    frozen_execution = Path(str(value.get("execution_root", "")))
    if not frozen_execution.is_absolute():
        frozen_execution = ROOT / frozen_execution
    need(frozen_execution.resolve() == execution_root.resolve(),
         "execution root differs from activation")
    seal_path(value["qualified_phase991_freeze"], execution_root, activation_path.parent,
              "Phase991 freeze")
    sources = value.get("source_seals")
    need(isinstance(sources, Mapping), "source seals missing")
    for key in ("protocol", "broker", "runner", "scorer", "audit"):
        seal_path(sources[key], execution_root, activation_path.parent, f"source {key}")
    return value


def prompt_path(split: str) -> Path:
    kind = "public" if split in PUBLIC_SPLITS else "private"
    relative = f"runtime_prompts/{kind}/{split}.jsonl"
    admission = read_json(P991_ROOT / "gpu_admission_preregistration.json")
    self_hash(admission, "gpu_admission_sha256", "Phase991 admission")
    path = seal_path(admission["artifact_seals"][relative], P991_ROOT, P991_ROOT,
                     f"Phase991 prompt {split}")
    need(path.resolve() == (P991_ROOT / relative).resolve(), "Phase991 prompt path drift")
    return path


def prompt_ids(split_names: Sequence[str]) -> tuple[list[str], Counter[str]]:
    identifiers: list[str] = []
    counts: Counter[str] = Counter()
    for split in split_names:
        for item in rows(prompt_path(split)):
            need(item.get("split") == split, "sealed prompt split mismatch")
            identifiers.append(str(item["record_id"]))
            counts[split] += 1
    need(len(identifiers) == len(set(identifiers)), "sealed prompt IDs duplicate")
    return identifiers, counts


def manifest_ids(path: Path) -> tuple[list[str], Counter[str]]:
    identifiers: list[str] = []
    counts: Counter[str] = Counter()
    for item in rows(path):
        identifiers.append(str(item["record_id"]))
        counts[str(item["split"])] += 1
    need(len(identifiers) == len(set(identifiers)), "manifest IDs duplicate")
    return identifiers, counts


def manifest_prompt_hashes(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in rows(path):
        identifier = item.get("record_id")
        prompt = item.get("prompt")
        prompt_sha = item.get("prompt_sha256")
        need(isinstance(identifier, str) and identifier not in result,
             "manifest prompt identity duplicate")
        need(isinstance(prompt, str) and isinstance(prompt_sha, str)
             and hashlib.sha256(prompt.encode("utf-8")).hexdigest() == prompt_sha,
             "manifest prompt text/hash mismatch")
        result[identifier] = prompt_sha
    return result


def holdout_chain_check(execution_root: Path, execution_receipts: Mapping[str, Any]) -> str:
    path = execution_root / "holdout_access/final_chain_receipt.json"
    chain = read_json(path)
    self_hash(chain, "receipt_sha256", "holdout chain")
    need(chain.get("schema_version") == "phase992_holdout_access_chain_receipt.v1"
         and chain.get("phase") == PHASE and chain.get("status") == "complete"
         and chain.get("model_order") == list(MODEL_ORDER)
         and chain.get("event_count") == 6
         and chain.get("all_temporary_copies_revoked") is True,
         "holdout chain incomplete")
    need(not any((execution_root / "temporary_holdout").glob("*.jsonl")),
         "holdout temporary copies remain")
    event_paths = sorted((execution_root / "holdout_access/events").glob("*.json"))
    need(len(event_paths) == 6, "holdout chain event count")
    head = str(chain.get("genesis_head", ""))
    need(SHA.fullmatch(head) is not None, "holdout genesis invalid")
    for index, model in enumerate(MODEL_ORDER):
        for offset, action in enumerate(("grant", "seal_and_revoke")):
            ordinal = 2 * index + offset
            event_path = event_paths[ordinal]
            need(event_path.name.startswith(f"{ordinal:04d}_"), "holdout event ordinal gap")
            event = read_json(event_path)
            need(event.get("model") == model and event.get("action") == action,
                 "holdout event order")
            need(event.get("schema_version") == "phase992_holdout_access_event.v1"
                 and event.get("ordinal") == ordinal
                 and event.get("model_order_index") == index
                 and event.get("previous_head") == head, "holdout chain link/identity")
            body = dict(event)
            body.pop("new_head", None)
            head = json_hash(body)
            need(event.get("new_head") == head, "holdout event hash")
            if action == "seal_and_revoke":
                need(event.get("output_receipt_sha256") == file_hash(
                    execution_root / f"receipts/holdout_{model}.json"
                ), "holdout receipt link")
    need(chain.get("final_head") == head, "holdout final head")
    run_id = str(chain.get("run_id", ""))
    need(run_id and all(execution_receipts[model].get("run_id") == run_id
                        for model in MODEL_ORDER), "holdout chain run mismatch")
    for index, model in enumerate(MODEL_ORDER):
        grant = read_json(execution_root / f"holdout_access/grant_{index:02d}_{model}.json")
        seal = read_json(execution_root / f"holdout_access/seal_{index:02d}_{model}.json")
        self_hash(grant, "receipt_sha256", f"holdout grant {model}")
        self_hash(seal, "receipt_sha256", f"holdout seal {model}")
        need(grant.get("schema_version") == "phase992_holdout_grant_receipt.v1"
             and grant.get("run_id") == run_id and grant.get("model") == model
             and grant.get("model_order_index") == index
             and grant.get("status") == "granted", "holdout grant mismatch")
        need(seal.get("schema_version") == "phase992_holdout_seal_receipt.v1"
             and seal.get("run_id") == run_id and seal.get("model") == model
             and seal.get("model_order_index") == index
             and seal.get("status") == "sealed"
             and seal.get("temporary_copy_revoked") is True
             and seal.get("output_receipt_sha256") == file_hash(
                 execution_root / f"receipts/holdout_{model}.json")
             and seal.get("cleanup_receipt_sha256") == file_hash(
                 execution_root / f"receipts/cleanup_holdout_{model}.json"),
             "holdout seal mismatch")
    return file_hash(path)


def receipt_barrier(execution_root: Path, protocol_root: Path, logical_scope: str
                    ) -> tuple[dict[str, Path], dict[str, Any], list[str],
                               dict[str, str], dict[str, Any]]:
    contract = SCOPE[logical_scope]
    raw_scope = str(contract["raw"])
    raw_paths: dict[str, Path] = {}
    receipts: dict[str, Any] = {}
    cleanups: dict[str, Any] = {}
    manifest_seals: dict[str, dict[str, Any]] = {}
    receipt_evidence: dict[str, Any] = {}
    previous: str | None = None
    for index, model in enumerate(MODEL_ORDER):
        receipt = read_json(execution_root / "receipts" / f"{raw_scope}_{model}.json")
        cleanup = read_json(execution_root / "receipts" / f"cleanup_{raw_scope}_{model}.json")
        self_hash(receipt, "receipt_sha256", f"execution {raw_scope}/{model}")
        self_hash(cleanup, "receipt_sha256", f"cleanup {raw_scope}/{model}")
        for document, role in ((receipt, "execution"), (cleanup, "cleanup")):
            need(document.get("phase") == PHASE and document.get("experiment") == EXPERIMENT,
                 f"{role} identity")
            need(document.get("scope") == raw_scope and document.get("model") == model,
                 f"{role} scope/model")
            need(document.get("status") == "sealed"
                 and document.get("model_order_index") == index, f"{role} status/order")
        need(receipt.get("execution_status") == "success", "execution status not success")
        need(SHA.fullmatch(str(receipt.get("activation_sha256", ""))) is not None
             and SHA.fullmatch(str(receipt.get("generation_contract_sha256", ""))) is not None,
             "execution activation/generation binding missing")
        need(receipt.get("previous_model_receipt_sha256") == previous, "execution receipt chain")
        need(SHA.fullmatch(str(receipt.get("worker_status_sha256", ""))) is not None,
             "worker status not sealed")
        worker_path = seal_path(
            receipt.get("worker_status_artifact"), execution_root, protocol_root,
            f"worker status {raw_scope}/{model}",
        )
        worker = read_json(worker_path)
        self_hash(worker, "worker_status_sha256", f"worker status {raw_scope}/{model}")
        need(worker.get("worker_status_sha256") == receipt.get("worker_status_sha256")
             and worker.get("phase") == PHASE and worker.get("experiment") == EXPERIMENT
             and worker.get("scope") == raw_scope and worker.get("model") == model
             and worker.get("model_order_index") == index
             and worker.get("run_id") == receipt.get("run_id")
             and worker.get("status") == "success", "worker status identity")
        need(worker.get("activation_sha256") == receipt.get("activation_sha256")
             and worker.get("generation_contract_sha256")
                 == receipt.get("generation_contract_sha256")
             and worker.get("raw_artifact") == receipt.get("raw_artifact")
             and worker.get("input_manifest") == receipt.get("input_manifest")
             and worker.get("raw_row_count") == receipt.get("row_count")
             and worker.get("record_ids_sha256") == receipt.get("record_ids_sha256"),
             "worker/receipt evidence binding")
        need(worker.get("runner_source_sha256") == file_hash(
                 GLM5 / "phase992_delayed_binding_runner.py")
             and worker.get("engine_source_sha256") == file_hash(
                 GLM5 / "phase983_cross_model_engine.py"), "worker source identity")
        identity = worker.get("loaded_model_identity")
        artifact = worker.get("model_artifact_verification")
        strict_release = worker.get("strict_cuda_release")
        quant = identity.get("loaded_quantization") if isinstance(identity, Mapping) else None
        device_map = identity.get("hf_device_map") if isinstance(identity, Mapping) else None
        need(isinstance(artifact, Mapping) and artifact.get("passed") is True
             and artifact.get("model") == model
             and artifact.get("all_files_sha256_verified_immediately_before_load") is True
             and isinstance(artifact.get("file_count"), int) and artifact["file_count"] > 0
             and isinstance(artifact.get("weight_bytes"), int) and artifact["weight_bytes"] > 0
             and SHA.fullmatch(str(artifact.get("model_manifest_sha256", ""))) is not None
             and SHA.fullmatch(str(artifact.get("files_manifest_sha256", ""))) is not None
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
             "loaded-model/release identity")
        previous = str(receipt["receipt_sha256"])
        need(cleanup.get("cleanup_pass") is True and cleanup.get("baseline_recovered") is True,
             "cleanup/baseline failed")
        need(cleanup.get("run_id") == receipt.get("run_id")
             and cleanup.get("activation_sha256") == receipt.get("activation_sha256")
             and cleanup.get("worker_status_sha256") == receipt.get("worker_status_sha256"),
             "cleanup/execution binding")
        need(all(cleanup.get(field) is True for field in (
                 "model_released", "child_exit_zero", "cuda_allocated_zero",
                 "cuda_reserved_zero")), "cleanup lifecycle evidence")
        need(isinstance(cleanup.get("baseline_before"), Mapping), "cleanup baseline missing")
        need(isinstance(cleanup.get("allocated_after"), int)
             and isinstance(cleanup.get("reserved_after"), int), "cleanup counters invalid")
        raw_path = seal_path(receipt["raw_artifact"], execution_root, protocol_root,
                             f"raw {raw_scope}/{model}")
        manifest_seal = receipt.get("input_manifest")
        need(isinstance(manifest_seal, Mapping)
             and isinstance(manifest_seal.get("path"), str)
             and isinstance(manifest_seal.get("bytes"), int)
             and SHA.fullmatch(str(manifest_seal.get("sha256", ""))) is not None,
             "input manifest seal malformed")
        raw_paths[model] = raw_path
        receipts[model] = receipt
        cleanups[model] = cleanup
        manifest_seals[model] = deepcopy(dict(manifest_seal))
    # Only now, after all execution and cleanup receipts pass, open the scope
    # manifests and compare their exact ID population.
    expected, expected_counts = prompt_ids(contract["splits"])
    need(len(expected) == contract["count"], "sealed prompt count")
    reference_manifest: set[str] | None = None
    reference_prompts: dict[str, str] | None = None
    for model in MODEL_ORDER:
        if logical_scope in ("holdout", "extension"):
            split = str(contract["splits"][0])
            admission = read_json(P991_ROOT / "gpu_admission_preregistration.json")
            self_hash(admission, "gpu_admission_sha256", "Phase991 admission")
            expected_seal = admission["artifact_seals"][f"runtime_prompts/private/{split}.jsonl"]
            need(manifest_seals[model]["bytes"] == expected_seal["bytes"]
                 and manifest_seals[model]["sha256"] == expected_seal["sha256"],
                 "private input manifest seal mismatch")
            manifest_path = prompt_path(split)
        else:
            manifest_path = seal_path(manifest_seals[model], execution_root, protocol_root,
                                      f"manifest {raw_scope}/{model}")
        identifiers, split_counts = manifest_ids(manifest_path)
        manifest_prompts = manifest_prompt_hashes(manifest_path)
        need(len(identifiers) == contract["count"] == int(receipts[model].get("row_count", -1)),
             "manifest/receipt count mismatch")
        need(split_counts == expected_counts, "manifest split counts mismatch")
        need(set(identifiers) == set(expected), "manifest differs from Phase991 sealed prompts")
        need(json_hash(sorted(identifiers)) == receipts[model].get("record_ids_sha256"),
             "receipt ID hash mismatch")
        if reference_manifest is None:
            reference_manifest = set(identifiers)
            reference_prompts = manifest_prompts
        else:
            need(reference_manifest == set(identifiers), "cross-model manifest mismatch")
            need(reference_prompts == manifest_prompts,
                 "cross-model prompt identity mismatch")
        receipt_evidence[model] = {
            "execution_receipt_sha256": receipts[model]["receipt_sha256"],
            "cleanup_receipt_sha256": cleanups[model]["receipt_sha256"],
            "raw_sha256": file_hash(raw_paths[model]),
            "manifest_sha256": manifest_seals[model]["sha256"],
        }
    if logical_scope == "holdout":
        public, _ = prompt_ids(PUBLIC_SPLITS)
        need(not (set(public) & set(expected)) and len(set(public) | set(expected)) == 10240,
             "cumulative primary IDs not exact 10240")
        receipt_evidence["holdout_chain_sha256"] = holdout_chain_check(execution_root, receipts)
    expected_prompts: dict[str, str] = {}
    for split in contract["splits"]:
        expected_prompts.update(manifest_prompt_hashes(prompt_path(str(split))))
    need(reference_prompts == expected_prompts,
         "manifest prompts differ from sealed Phase991 prompt identities")
    return raw_paths, receipts, expected, expected_prompts, receipt_evidence


def prior_gate(execution_root: Path, logical_scope: str) -> dict[str, Any] | None:
    if logical_scope == "public":
        return None
    prior_scope = "public" if logical_scope == "holdout" else "holdout"
    prior = read_json(execution_root / SCOPE[prior_scope]["score"])
    self_hash(prior, "score_sha256", f"prior {prior_scope} score")
    need(prior.get("passed") is True, f"prior {prior_scope} did not pass")
    field = "scope_behavior_pass" if logical_scope == "holdout" else "primary_behavior_pass"
    need(all(prior["models"][model].get(field) is True for model in MODEL_ORDER),
         f"prior all-model {field} absent")
    return prior


def extension_release_check(execution_root: Path, activation: Mapping[str, Any],
                            prior: Mapping[str, Any]) -> None:
    release = read_json(execution_root / "extension_behavior_release.json")
    self_hash(release, "receipt_sha256", "extension release")
    need(release.get("schema_version") == "phase992_extension_behavior_release.v1"
         and release.get("phase") == PHASE and release.get("experiment") == EXPERIMENT,
         "extension release identity")
    need(release.get("extension_behavior_execution_authorized") is True
         and release.get("all_models_primary_pass") is True, "extension release unauthorized")
    need(release.get("activation_sha256") == activation["activation_sha256"]
         and release.get("holdout_score_sha256") == prior["score_sha256"],
         "extension release binding")


def private_truth(logical_scope: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    admission = read_json(P991_ROOT / "gpu_admission_preregistration.json")
    self_hash(admission, "gpu_admission_sha256", "Phase991 admission")
    for split in SCOPE[logical_scope]["splits"]:
        relative = f"scoring_truth/private/{split}.jsonl"
        path = seal_path(admission["artifact_seals"][relative], P991_ROOT, P991_ROOT,
                         f"Phase991 truth {split}")
        need(path.resolve() == (P991_ROOT / relative).resolve(), "Phase991 truth path drift")
        result.extend(rows(path))
    return result


def truth_seals(logical_scope: str) -> dict[str, Any]:
    admission = read_json(P991_ROOT / "gpu_admission_preregistration.json")
    self_hash(admission, "gpu_admission_sha256", "Phase991 admission")
    return {split: deepcopy(admission["artifact_seals"][f"scoring_truth/private/{split}.jsonl"])
            for split in SCOPE[logical_scope]["splits"]}


def parsed(text: str) -> dict[str, Any]:
    markers = [match.group(1).casefold() for match in MARKER.finditer(text.casefold())]
    unique = list(dict.fromkeys(markers))
    prediction = markers[0] if markers else None
    strict = STRICT.fullmatch(text.strip())
    return {"prediction": prediction, "markers": markers, "correctable": prediction is not None,
            "unparsed": prediction is None, "ambiguous": len(unique) > 1,
            "strict_format": strict is not None and strict.group(1).casefold() == prediction}


def raw_check(item: Mapping[str, Any], model: str, raw_scope: str) -> None:
    need(not (TRUTH_FORBIDDEN & set(item)), "raw truth leak")
    need(item.get("schema_version") == "phase992_delayed_binding_raw.v1", "raw schema")
    need(item.get("phase") == PHASE and item.get("experiment") == EXPERIMENT, "raw identity")
    need(item.get("scope") == raw_scope and item.get("model") == model, "raw scope/model")
    need(item.get("model_order_index") == MODEL_ORDER.index(model), "raw model order")
    need(isinstance(item.get("generated_text"), str), "generated text type")
    for field in ("prompt_sha256", "input_manifest_sha256", "input_token_ids_sha256",
                  "teacher_forced_context_sha256", "generation_contract_sha256",
                  "activation_sha256"):
        need(isinstance(item.get(field), str) and SHA.fullmatch(str(item[field])) is not None,
             f"raw SHA field {field}")
    input_ids = item.get("input_token_ids")
    need(isinstance(input_ids, list) and input_ids
         and all(isinstance(token, int) for token in input_ids)
         and item.get("input_token_count") == len(input_ids)
         and item.get("input_token_ids_sha256") == json_hash(input_ids),
         "input token identity")
    need(isinstance(item.get("eos_seen"), bool)
         and isinstance(item.get("budget_exhausted"), bool), "EOS/budget type")
    token_ids = item.get("generated_token_ids_before_eos")
    need(isinstance(token_ids, list) and len(token_ids) <= 24
         and all(isinstance(token, int) for token in token_ids), "generated token IDs")
    suffix = item.get("generated_suffix_token_ids")
    effective_eos = item.get("effective_eos_token_ids")
    need(isinstance(suffix, list) and len(suffix) <= 24
         and all(isinstance(token, int) for token in suffix), "generated suffix")
    need(isinstance(effective_eos, list) and effective_eos
         and all(isinstance(token, int) for token in effective_eos), "effective EOS set")
    if item["eos_seen"]:
        first = item.get("first_eos_index")
        need(isinstance(first, int) and 0 <= first < len(suffix)
             and suffix[first] in effective_eos
             and item.get("first_eos_token_id") == suffix[first]
             and token_ids == suffix[:first]
             and item["budget_exhausted"] is False, "EOS accounting")
    else:
        need(item.get("first_eos_index") is None and item.get("first_eos_token_id") is None
             and token_ids == suffix and len(token_ids) == 24
             and item["budget_exhausted"] is True, "budget accounting")
    candidates = item.get("teacher_forced_candidates")
    need(isinstance(candidates, Mapping) and set(candidates) == set(VALUES), "candidate set")
    need(tuple(item.get("teacher_forced_candidate_order", ())) == VALUES, "candidate order")
    for value in VALUES:
        candidate = candidates[value]
        for field, hex_field in (("logit", "logit_hex"), ("logprob", "logprob_hex")):
            number = candidate.get(field)
            need(isinstance(number, (int, float)) and math.isfinite(float(number)), "candidate finite")
            need(candidate.get(hex_field) == float(number).hex(), "candidate hex mismatch")


def independent_recompute(model: str, logical_scope: str, raw_data: Sequence[Mapping[str, Any]],
                          truth_data: Sequence[Mapping[str, Any]], expected: Sequence[str]) -> dict[str, Any]:
    raw_scope = str(SCOPE[logical_scope]["raw"])
    raw_by_id: dict[str, Mapping[str, Any]] = {}
    for item in raw_data:
        raw_check(item, model, raw_scope)
        identifier = str(item["record_id"])
        need(identifier not in raw_by_id, "duplicate raw ID")
        raw_by_id[identifier] = item
    truth_by_id = {str(item["record_id"]): item for item in truth_data}
    need(len(truth_by_id) == len(truth_data), "duplicate truth ID")
    need(set(raw_by_id) == set(truth_by_id) == set(expected), "raw/truth/manifest ID mismatch")
    major: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    cells: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    closures: dict[tuple[str, str, str, str], dict[str, str | None]] = defaultdict(dict)
    correct_by_id: dict[str, bool] = {}
    evidence: list[dict[str, Any]] = []
    margins: list[float] = []
    total_correct = unparsed = ambiguity = strict_count = eos = budget = positive = ties = 0
    for identifier in sorted(raw_by_id):
        raw = raw_by_id[identifier]
        truth = truth_by_id[identifier]
        for field in ("record_id", "semantic_world_id", "split", "variant_id", "split_ordinal"):
            need(raw.get(field) == truth.get(field), f"case metadata {field}")
        result = parsed(str(raw["generated_text"]))
        gold = str(truth["gold_value"])
        condition, paraphrase, order, horizon = str(truth["variant_id"]).split("__")
        need(condition in CONDITIONS and gold in VALUES, "truth/variant invalid")
        correct = result["prediction"] == gold
        correct_by_id[identifier] = correct
        total_correct += int(correct)
        unparsed += int(result["unparsed"]); ambiguity += int(result["ambiguous"])
        strict_count += int(result["strict_format"]); eos += int(raw["eos_seen"])
        budget += int(raw["budget_exhausted"])
        major[condition][0] += int(correct); major[condition][1] += 1
        key = f"{truth['split']}|{truth['variant_id']}"
        cells[key][0] += int(correct); cells[key][1] += 1
        closure = (str(truth["semantic_world_id"]), paraphrase, order, horizon)
        need(condition not in closures[closure], "duplicate closure condition")
        closures[closure][condition] = result["prediction"]
        logits = {value: float(raw["teacher_forced_candidates"][value]["logit"])
                  for value in VALUES}
        target = logits[gold]; best = max(logits[value] for value in VALUES if value != gold)
        margin = target - best
        margins.append(margin); positive += int(margin > 0); ties += int(margin == 0)
        teacher = {"target_logit_hex": target.hex(), "best_foil_logit_hex": best.hex(),
                   "margin_hex": margin.hex(), "positive": margin > 0, "tie": margin == 0}
        evidence.append({"record_id": identifier, "prediction": result["prediction"],
                         "markers": result["markers"], "correct": correct,
                         "unparsed": result["unparsed"], "ambiguous": result["ambiguous"],
                         "strict_format": result["strict_format"], "eos_seen": raw["eos_seen"],
                         "budget_exhausted": raw["budget_exhausted"],
                         "termination_reason": raw["termination_reason"], "teacher": teacher})
    sensitive = pair_count = 0
    by_pair: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for key, predictions in closures.items():
        need(set(predictions) == set(CONDITIONS), f"incomplete closure {key}")
        for left_index, left in enumerate(CONDITIONS):
            for right in CONDITIONS[left_index + 1:]:
                passed = predictions[left] is not None and predictions[right] is not None \
                    and predictions[left] != predictions[right]
                sensitive += int(passed); pair_count += 1
                by_pair[f"{left}|{right}"][0] += int(passed)
                by_pair[f"{left}|{right}"][1] += 1
    peers: set[tuple[str, str]] = set()
    joint = 0
    for truth in truth_data:
        registry = truth.get("semantic_peer_record_ids")
        need(isinstance(registry, Mapping) and set(registry) == set(CONDITIONS), "peer registry")
        for peer in registry.values():
            pair = tuple(sorted((str(truth["record_id"]), str(peer))))
            if pair[0] != pair[1] and pair not in peers:
                peers.add(pair); joint += int(correct_by_id[pair[0]] and correct_by_id[pair[1]])
    need(len(peers) == pair_count, "peer/group pair disagreement")

    def passed(count: int, denominator: int, threshold: int) -> bool:
        return 100 * count >= threshold * denominator
    major_counts = {key: {"correct": value[0], "denominator": value[1]}
                    for key, value in sorted(major.items())}
    cell_counts = {key: {"correct": value[0], "denominator": value[1]}
                   for key, value in sorted(cells.items())}
    scope_pass = (all(passed(v[0], v[1], 90) for v in major.values())
                  and all(passed(v[0], v[1], 85) for v in cells.values())
                  and passed(sensitive, pair_count, 85))
    margins.sort(); n = len(margins)
    median = margins[n // 2] if n % 2 else (margins[n // 2 - 1] + margins[n // 2]) / 2
    return {
        "gate_counts": {"major_conditions": major_counts, "split_variants": cell_counts,
                        "counterfactual": {"correct": sensitive, "denominator": pair_count}},
        "scope_behavior_pass": scope_pass,
        "overall": {"correct": total_correct, "denominator": n},
        "separate": {"unparsed": unparsed, "ambiguity": ambiguity,
                     "strict_reference_format": strict_count, "eos_seen": eos,
                     "budget_exhausted": budget, "denominator": n},
        "counterfactual": {"correct": sensitive, "denominator": pair_count,
                           "joint": joint, "by_pair": dict(sorted(by_pair.items()))},
        "teacher": {"denominator": n, "positive": positive, "tie": ties,
                    "mean_margin": math.fsum(margins) / n, "median_margin": median,
                    "minimum_margin": margins[0], "maximum_margin": margins[-1]},
        "case_evidence_count": n, "case_evidence_sha256": json_hash(evidence),
    }


def merge_counts(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    def merged(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
        return {key: {"correct": int(a.get(key, {}).get("correct", 0))
                                + int(b.get(key, {}).get("correct", 0)),
                      "denominator": int(a.get(key, {}).get("denominator", 0))
                                    + int(b.get(key, {}).get("denominator", 0))}
                for key in sorted(set(a) | set(b))}
    return {"major_conditions": merged(left["major_conditions"], right["major_conditions"]),
            "split_variants": merged(left["split_variants"], right["split_variants"]),
            "counterfactual": {"correct": int(left["counterfactual"]["correct"])
                                          + int(right["counterfactual"]["correct"]),
                               "denominator": int(left["counterfactual"]["denominator"])
                                              + int(right["counterfactual"]["denominator"])}}


def cumulative_pass(counts: Mapping[str, Any]) -> bool:
    return (all(100 * row["correct"] >= 90 * row["denominator"]
                for row in counts["major_conditions"].values())
            and all(100 * row["correct"] >= 85 * row["denominator"]
                    for row in counts["split_variants"].values())
            and 100 * counts["counterfactual"]["correct"]
                >= 85 * counts["counterfactual"]["denominator"])


def detailed_gate(counts: Mapping[str, Any]) -> dict[str, Any]:
    def detail(items: Mapping[str, Any], threshold: int) -> dict[str, Any]:
        return {key: {**row,
                      "percent": 100.0 * row["correct"] / row["denominator"],
                      "threshold_percent": threshold,
                      "passed": 100 * row["correct"] >= threshold * row["denominator"]}
                for key, row in sorted(items.items())}
    major = detail(counts["major_conditions"], 90)
    variants = detail(counts["split_variants"], 85)
    counterfactual = counts["counterfactual"]
    cf = {**counterfactual,
          "percent": 100.0 * counterfactual["correct"] / counterfactual["denominator"],
          "threshold_percent": 85,
          "passed": 100 * counterfactual["correct"] >= 85 * counterfactual["denominator"]}
    return {"major_conditions": major, "split_variants": variants,
            "counterfactual_pair_sensitivity": cf,
            "passed": all(row["passed"] for row in major.values())
                      and all(row["passed"] for row in variants.values()) and cf["passed"]}


def baseline_recompute(all_truth: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    representatives = [item for item in all_truth if item["paraphrase_id"] == "standard"
                       and item["fact_order_id"] == "order_a" and item["horizon_id"] == "near"]
    training = [item for item in representatives if item["split"] == "discovery"]
    table_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for item in training:
        key = (str(item["query_entity"]), str(item["gold_object"]), str(item["query_relation"]))
        table_counts[key][str(item["gold_value"])] += 1
        global_counts[str(item["gold_value"])] += 1
    order = {value: index for index, value in enumerate(VALUES)}
    choose = lambda counter: min(VALUES, key=lambda value: (-counter[value], order[value]))
    fallback = choose(global_counts)
    table = {key: choose(counter) for key, counter in table_counts.items()}
    evaluations: dict[str, Any] = {}
    for split in (*PUBLIC_SPLITS, "sealed_holdout"):
        selected = [item for item in representatives if item["split"] == split]
        correct = seen = 0
        for item in selected:
            key = (str(item["query_entity"]), str(item["gold_object"]), str(item["query_relation"]))
            correct += int(table.get(key, fallback) == item["gold_value"]); seen += int(key in table)
        evaluations[split] = {"correct": correct, "denominator": len(selected),
                              "accuracy_percent": 100.0 * correct / len(selected),
                              "seen_feature_rows": seen,
                              "unseen_feature_rows": len(selected) - seen}
    return {"role": "first-hop-resolved selected-object plus relation lookup baseline",
            "not_a_pure_surface_baseline": True, "fit_split": "discovery",
            "training_rows": len(training),
            "feature": "query_entity+resolved_selected_object+query_relation",
            "tie_break": "fixed VALUES order red,blue,green,black",
            "unseen_fallback": fallback,
            "table_sha256": json_hash({"|".join(key): value for key, value in sorted(table.items())}),
            "evaluations": evaluations, "not_mechanism_evidence": True}


def compare_report_model(report: Mapping[str, Any], recomputed: Mapping[str, Any]) -> dict[str, bool]:
    natural = report["natural_generation"]
    teacher = report["teacher_forced_diagnostic"]
    def gate_rows_match(actual: Mapping[str, Any], counts: Mapping[str, Any], threshold: int) -> bool:
        if set(actual) != set(counts):
            return False
        for key, count in counts.items():
            denominator = int(count["denominator"]); correct = int(count["correct"])
            row = actual[key]
            if not (row.get("correct") == correct and row.get("denominator") == denominator
                    and row.get("threshold_percent") == threshold
                    and row.get("percent") == 100.0 * correct / denominator
                    and row.get("passed") == (100 * correct >= threshold * denominator)):
                return False
        return True
    cf_counts = recomputed["gate_counts"]["counterfactual"]
    cf_actual = natural["counterfactual_pair_sensitivity"]
    checks = {
        "case_count": report.get("case_evidence_count") == recomputed["case_evidence_count"],
        "case_hash": report.get("case_evidence_sha256") == recomputed["case_evidence_sha256"],
        "gate_counts": report.get("gate_counts") == recomputed["gate_counts"],
        "scope_pass": report.get("scope_behavior_pass") == recomputed["scope_behavior_pass"],
        "major_gate_rows": gate_rows_match(
            natural["major_conditions"], recomputed["gate_counts"]["major_conditions"], 90),
        "split_variant_gate_rows": gate_rows_match(
            natural["split_variants"], recomputed["gate_counts"]["split_variants"], 85),
        "counterfactual_gate": (
            cf_actual.get("correct") == cf_counts["correct"]
            and cf_actual.get("denominator") == cf_counts["denominator"]
            and cf_actual.get("threshold_percent") == 85
            and cf_actual.get("percent") == 100.0 * cf_counts["correct"] / cf_counts["denominator"]
            and cf_actual.get("passed") == (
                100 * cf_counts["correct"] >= 85 * cf_counts["denominator"])),
        "overall_counts": (natural["overall"]["correct"], natural["overall"]["denominator"])
                          == (recomputed["overall"]["correct"], recomputed["overall"]["denominator"]),
        "separate_accounts": natural["separate_accounts"] == recomputed["separate"],
        "counterfactual_counts": (
            natural["counterfactual_pair_sensitivity"]["correct"],
            natural["counterfactual_pair_sensitivity"]["denominator"],
            natural["counterfactual_pair_sensitivity"]["joint_semantic_correct_pairs"],
        ) == (recomputed["counterfactual"]["correct"],
              recomputed["counterfactual"]["denominator"], recomputed["counterfactual"]["joint"]),
        "teacher_counts": (teacher["denominator"], teacher["positive"], teacher["tie"])
                          == (recomputed["teacher"]["denominator"],
                              recomputed["teacher"]["positive"], recomputed["teacher"]["tie"]),
        "teacher_moments": all(teacher[key] == recomputed["teacher"][key]
                               for key in ("mean_margin", "median_margin",
                                           "minimum_margin", "maximum_margin")),
    }
    need(all(checks.values()), f"score mismatch: {checks}")
    return checks


def audit_scope(execution_root: Path, protocol_root: Path, activation_path: Path,
                logical_scope: str, write: bool) -> dict[str, Any]:
    activation = activation_check(activation_path, execution_root)
    prior = prior_gate(execution_root, logical_scope)
    if logical_scope == "extension":
        need(prior is not None, "extension prior absent")
        extension_release_check(execution_root, activation, prior)
    # Complete independent barrier before first private_truth call.
    raw_paths, receipts, expected, expected_prompts, receipt_evidence = receipt_barrier(
        execution_root, protocol_root, logical_scope)
    truth = private_truth(logical_scope)
    need(len(truth) == SCOPE[logical_scope]["count"], "truth count")
    score_path = execution_root / SCOPE[logical_scope]["score"]
    score = read_json(score_path)
    self_hash(score, "score_sha256", f"{logical_scope} score")
    need(score.get("scope") == logical_scope and score.get("activation_sha256") ==
         activation["activation_sha256"], "score scope/activation mismatch")
    need(score.get("truth_artifact_seals") == truth_seals(logical_scope),
         "score truth artifact seals mismatch")
    run_ids = {str(receipts[model].get("run_id", "")) for model in MODEL_ORDER}
    need(len(run_ids) == 1 and "" not in run_ids and score.get("run_id") in run_ids,
         "score/receipt run identity mismatch")
    model_reports: dict[str, Any] = {}
    for model in MODEL_ORDER:
        raw_data = list(rows(raw_paths[model]))
        need(all(item.get("run_id") == receipts[model]["run_id"]
                 and item.get("prompt_sha256")
                     == expected_prompts.get(str(item.get("record_id", "")))
                 and item.get("activation_sha256") == activation["activation_sha256"]
                 and item.get("generation_contract_sha256")
                     == receipts[model]["generation_contract_sha256"]
                 and item.get("input_manifest_sha256")
                     == receipts[model]["input_manifest"]["sha256"]
                 for item in raw_data), "raw run/prompt/activation binding")
        recomputed = independent_recompute(model, logical_scope, raw_data, truth, expected)
        checks = compare_report_model(score["models"][model], recomputed)
        if logical_scope == "holdout":
            need(prior is not None, "missing public prior")
            merged = merge_counts(prior["models"][model]["gate_counts"], recomputed["gate_counts"])
            expected_cumulative = detailed_gate(merged)
            need(score["models"][model]["cumulative_primary_gate"] == expected_cumulative
                 and score["models"][model]["primary_behavior_pass"] == expected_cumulative["passed"],
                 "cumulative primary gate mismatch")
        model_reports[model] = {"passed": True, "checks": checks,
                                "recomputed_case_evidence_sha256": recomputed["case_evidence_sha256"],
                                "receipt_evidence": receipt_evidence[model]}
    expected_top_pass = all(score["models"][model]["scope_behavior_pass"]
                            for model in MODEL_ORDER)
    if logical_scope == "holdout":
        expected_top_pass = expected_top_pass and all(
            score["models"][model]["primary_behavior_pass"] for model in MODEL_ORDER)
    need(score.get("passed") == expected_top_pass, "top-level score gate mismatch")
    baseline_checks: dict[str, bool] = {"non_mechanism_label": True}
    if logical_scope == "public":
        frozen = read_json(P991_ROOT / "gpu_admission_preregistration.json")
        self_hash(frozen, "gpu_admission_sha256", "Phase991 admission")
        expected_baselines = deepcopy(frozen["shortcut_baselines"])
        expected_baselines["source_phase991_gpu_admission_sha256"] = frozen["gpu_admission_sha256"]
        expected_baselines["reported_as_behavior_comparator_not_mechanism_evidence"] = True
        baseline_checks["frozen_phase991_baselines_exact"] = (
            score["shortcut_baselines"] == expected_baselines)
        admission_path = execution_root / "public_behavior_admission.json"
        admission = read_json(admission_path)
        self_hash(admission, "admission_sha256", "public behavior admission")
        baseline_checks["public_admission_bound"] = (
            admission.get("schema_version") == "phase992_public_behavior_admission.v1"
            and admission.get("run_id") == score["run_id"]
            and admission.get("model_order") == list(MODEL_ORDER)
            and admission.get("all_models_public_pass") == score["passed"]
            and admission.get("sealed_holdout_model_access_authorized") == score["passed"]
            and admission.get("activation_sha256") == activation["activation_sha256"]
            and admission.get("public_score", {}).get("sha256") == file_hash(score_path)
            and admission.get("public_score", {}).get("score_sha256") == score["score_sha256"]
        )
    elif logical_scope == "holdout":
        public_truth = private_truth("public")
        expected_baseline = baseline_recompute([*public_truth, *truth])
        actual = score["shortcut_baselines"]["discovery_fitted_lookup_baseline"]
        baseline_checks["discovery_fit_exact"] = actual == expected_baseline
        frozen = read_json(P991_ROOT / "gpu_admission_preregistration.json")
        self_hash(frozen, "gpu_admission_sha256", "Phase991 admission")
        oracle = score["shortcut_baselines"]["oracle_structure_baseline"]
        baseline_checks["oracle_frozen_exact"] = (
            oracle == frozen["shortcut_baselines"]["oracle_structure_baseline"]
            and score["shortcut_baselines"].get("source_phase991_gpu_admission_sha256")
                == frozen["gpu_admission_sha256"])
    need(all(baseline_checks.values()), f"baseline mismatch: {baseline_checks}")
    payload = {
        "schema_version": SCHEMA_VERSION, "phase": PHASE, "experiment": EXPERIMENT,
        "role": "independent_truth_and_raw_recomputation_audit", "scope": logical_scope,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
        "activation_sha256": activation["activation_sha256"],
        "score_file": {"path": str(score_path.relative_to(execution_root)).replace("\\", "/"),
                       "bytes": score_path.stat().st_size, "sha256": file_hash(score_path),
                       "score_sha256": score["score_sha256"]},
        "models": model_reports, "baseline_checks": baseline_checks,
        "holdout_chain_sha256": receipt_evidence.get("holdout_chain_sha256"),
        "independence": {"imports_scorer": False, "calls_primary_score_function": False,
                         "all_cases_recomputed": True, "gates_recomputed_from_integer_counts": True,
                         "raw_and_truth_hashes_rechecked": True},
        "scientific_scope": {"behavior_only": True, "internal_structure_evidence": False,
                             "mechanism_evidence": False},
    }
    payload["audit_sha256"] = json_hash(payload)
    if write:
        output = execution_root / "scores" / f"{logical_scope}_independent_audit.json"
        need(not output.exists(), f"refusing overwrite: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                              allow_nan=False) + "\n").encode("utf-8")
        with output.open("xb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
    return payload


def self_test() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
               for alias in node.names}
    from_imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    checks["scorer_not_imported"] = all(
        name != "phase992_delayed_binding_scorer" for name in imports | from_imports)
    checks["first_marker"] = parsed("red and then blue")["prediction"] == "red"
    checks["ambiguity_separate"] = parsed("red and then blue")["ambiguous"] is True
    checks["embedded_rejected"] = parsed("blackbird")["unparsed"] is True
    checks["strict_format"] = parsed("The retrieved marker is black.")["strict_format"] is True
    synthetic = {"major_conditions": {key: {"correct": 90, "denominator": 100}
                                       for key in CONDITIONS},
                 "split_variants": {str(i): {"correct": 85, "denominator": 100}
                                    for i in range(32)},
                 "counterfactual": {"correct": 85, "denominator": 100}}
    checks["integer_gate_inclusive"] = cumulative_pass(synthetic) is True
    synthetic["counterfactual"]["correct"] = 84
    checks["counterfactual_fail_closed"] = cumulative_pass(synthetic) is False
    checks["scope_counts"] = SCOPE["public"]["count"] + SCOPE["holdout"]["count"] == 10240
    checks["extension_conditional"] = SCOPE["extension"]["count"] == 4096
    synthetic_raw: list[dict[str, Any]] = []
    synthetic_truth: list[dict[str, Any]] = []
    for split_ordinal, split in enumerate(PUBLIC_SPLITS):
        world = f"audit_self_{split}"
        for paraphrase in ("standard", "paraphrase"):
            for order_name in ("order_a", "order_b"):
                for horizon in ("near", "far"):
                    peer_ids = {condition: f"{world}_{condition}_{paraphrase}_{order_name}_{horizon}"
                                for condition in CONDITIONS}
                    for condition_index, condition in enumerate(CONDITIONS):
                        identifier = peer_ids[condition]; gold = VALUES[condition_index]
                        variant = f"{condition}__{paraphrase}__{order_name}__{horizon}"
                        candidates: dict[str, Any] = {}
                        for value in VALUES:
                            logit = 1.0 if value == gold else 0.0
                            candidates[value] = {"continuation": f" {value}", "token_id": VALUES.index(value),
                                                 "logit": logit, "logit_hex": logit.hex(),
                                                 "logprob": logit, "logprob_hex": logit.hex()}
                        synthetic_raw.append({
                            "schema_version": "phase992_delayed_binding_raw.v1", "phase": PHASE,
                            "experiment": EXPERIMENT, "scope": "primary", "model": "qwen3",
                            "model_order_index": 0, "record_id": identifier,
                            "semantic_world_id": world, "split": split,
                            "split_ordinal": split_ordinal, "variant_id": variant,
                            "prompt_sha256": "a" * 64,
                            "input_manifest_sha256": "b" * 64,
                            "input_token_ids": [7, 8], "input_token_count": 2,
                            "input_token_ids_sha256": json_hash([7, 8]),
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
                        synthetic_truth.append({"record_id": identifier, "semantic_world_id": world,
                                                "split": split, "split_ordinal": split_ordinal,
                                                "variant_id": variant, "gold_value": gold,
                                                "semantic_peer_record_ids": peer_ids})
    recomputed = independent_recompute(
        "qwen3", "public", synthetic_raw, synthetic_truth,
        [row["record_id"] for row in synthetic_truth],
    )
    checks["full_independent_recompute"] = (
        recomputed["scope_behavior_pass"] is True
        and recomputed["case_evidence_count"] == 96
        and recomputed["counterfactual"]["correct"] == recomputed["counterfactual"]["denominator"])
    self_node = next(node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name == "self_test")
    self_calls = {node.func.id for node in ast.walk(self_node)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    checks["no_truth_read_by_self_test"] = "private_truth" not in self_calls
    need(all(checks.values()), f"audit self-test failed: {checks}")
    return {"phase": PHASE, "schema_version": SCHEMA_VERSION,
            "role": "independent_cpu_no_truth_no_write_self_test", "passed": True,
            "checks": checks, "cuda_used": False, "truth_opened": False, "files_written": 0}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--audit-public", action="store_true")
    modes.add_argument("--audit-holdout", action="store_true")
    modes.add_argument("--audit-extension", action="store_true")
    parser.add_argument("--execution-root", type=Path, default=EXECUTION_ROOT)
    parser.add_argument("--protocol-root", type=Path, default=PROTOCOL_ROOT)
    parser.add_argument("--activation-path", type=Path, default=ACTIVATION)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        report = self_test()
    else:
        scope = "public" if args.audit_public else "holdout" if args.audit_holdout else "extension"
        report = audit_scope(args.execution_root.resolve(), args.protocol_root.resolve(),
                             args.activation_path.resolve(), scope, not args.no_write)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
