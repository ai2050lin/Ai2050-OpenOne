#!/usr/bin/env python3
"""Phase 1225: typed FP16 numerical applicability calibration for Qwen3.

This is an instrument experiment, not a language-mechanism search.  It keeps
the Phase1224 final-layer readout fixed and separates four numerical contracts:
same-shape repeat, same-load shape change, cross-load same-shape replay, and
cross-load stored-state intervention replay.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1224_final_layer_patch_construct_audit as p1224
from model_utils import get_layers
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1225
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = TEST_ROOT / "phase1225_qwen3_fp16_numerical_applicability_audit.py"
SOURCE_ROOT = TEST_ROOT / "result/phase1224_final_layer_patch_construct_audit"
SOURCE_FINAL = SOURCE_ROOT / "analysis/final.json"
SOURCE_AUDIT = SOURCE_ROOT / "audit/independent_result_audit.json"
SOURCE_PROTOCOL = SOURCE_ROOT / "protocol/preregistration.json"
SOURCE_MANIFEST = SOURCE_ROOT / "protocol/construct_manifest.jsonl"
SOURCE_RECORDS = SOURCE_ROOT / "runs/construct_records.jsonl"
SOURCE_STATES = p1224.SOURCE_STATES
EXPECTED_SOURCE_FINAL_DIGEST = "f1ef7ef3669d4b3838b331e72b2135ab6548ecefdf0d30fe30eba99bb05f0edb"
EXPECTED_SOURCE_AUDIT_DIGEST = "a94373f4c84e9fbd338b9959034935b3be0afda3d80bde36f0f5ba57cced9805"

OUT_ROOT = TEST_ROOT / "result/phase1225_qwen3_fp16_numerical_applicability"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
MANIFEST_PATH = OUT_ROOT / "protocol/numerical_manifest.jsonl"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
REFERENCE_RECORD_PATH = OUT_ROOT / "runs/reference_records.jsonl"
REFERENCE_ARRAY_PATH = OUT_ROOT / "runs/reference_arrays.npz"
REFERENCE_SUMMARY_PATH = OUT_ROOT / "runs/reference_summary.json"
RELOAD_RECORD_PATH = OUT_ROOT / "runs/reload_records.jsonl"
RELOAD_SUMMARY_PATH = OUT_ROOT / "runs/reload_summary.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"

SPLITS = p1224.SPLITS
HOLDOUT_SPLITS = p1224.HOLDOUT_SPLITS
LAYER_COUNT = p1224.LAYER_COUNT
MAX_CONTINUATION = 6
EPSILON = 1e-12
EXACT_VARIANTS = ("immediate_repeat", "delayed_repeat")
SHAPE_VARIANTS = ("batch1", "batch8", "suffix4", "cache_true", "prompt_only")
ENVELOPE_MULTIPLIER = 2.0
FUNCTIONAL_CAPS = {
    "hidden_relative_rms": 0.05,
    "logit_max_abs": 1.0,
    "probability_max_abs": 0.02,
    "margin_drift_ratio": 0.05,
    "score_drift_ratio": 0.05,
    "top1_agreement": 1.0,
    "stored_completion_median": 0.99,
    "stored_positive_fraction": 1.0,
    "exact_abs": 1e-4,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else float("nan")


def fraction(values: list[bool]) -> float:
    return float(sum(bool(value) for value in values) / len(values)) if values else float("nan")


def verify_source() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    final = read_json(SOURCE_FINAL)
    audit = read_json(SOURCE_AUDIT)
    if final.get("final_digest") != EXPECTED_SOURCE_FINAL_DIGEST:
        raise RuntimeError("Phase1224 final digest drift")
    if audit.get("audit_digest") != EXPECTED_SOURCE_AUDIT_DIGEST or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1224 result audit drift")
    manifest = read_jsonl(SOURCE_MANIFEST)
    records = read_jsonl(SOURCE_RECORDS)
    states = read_jsonl(SOURCE_STATES)
    if len(manifest) != 160 or len(records) != 160 or len(states) != 640:
        raise RuntimeError("Phase1224 source cardinality drift")
    return manifest, records, states


def build_manifest() -> list[dict[str, Any]]:
    source, records, _states = verify_source()
    records_by_id = {row["pair_id"]: row for row in records}
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(source):
        source_record = records_by_id[item["pair_id"]]
        row: dict[str, Any] = {
            "schema_version": "phase1225.numerical-manifest.v1",
            "phase": PHASE,
            "index": index,
            "pair_id": item["pair_id"],
            "scope": item["scope"],
            "split": item["split"],
            "donor_state_id": item["donor_state_id"],
            "recipient_state_id": item["recipient_state_id"],
            "candidates": item["candidates"],
            "candidate_token_ids": item["candidate_token_ids"],
            "donor_gold": item["donor_gold"],
            "recipient_gold": item["recipient_gold"],
            "continuation_length": int(item["continuation_length"]),
            "generation_boundary": int(item["generation_boundary"]),
            "target_shift_abs": float(source_record["target_shift_abs"]),
        }
        if row["continuation_length"] > MAX_CONTINUATION:
            raise RuntimeError("continuation length exceeds frozen storage")
        row["row_digest"] = digest(row)
        rows.append(row)
    return rows


def build_protocol(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    source_hashes = {
        "script": file_sha256(SCRIPT),
        "audit_script": file_sha256(AUDIT_SCRIPT),
        "phase1224_final": file_sha256(SOURCE_FINAL),
        "phase1224_result_audit": file_sha256(SOURCE_AUDIT),
        "phase1224_protocol": file_sha256(SOURCE_PROTOCOL),
        "phase1224_manifest": file_sha256(SOURCE_MANIFEST),
        "phase1224_records": file_sha256(SOURCE_RECORDS),
        "phase1223_states": file_sha256(SOURCE_STATES),
    }
    protocol: dict[str, Any] = {
        "schema_version": "phase1225.preregistration.v1",
        "phase": PHASE,
        "created_at": utc_now(),
        "objective": "calibrate typed numerical applicability of the Phase1224 final-layer intervention camera",
        "claim_boundary": {
            "instrument_only": True,
            "not_language_mechanism": True,
            "not_cross_model": True,
            "not_new_mathematics": True,
        },
        "model": {
            "name": "qwen3",
            "precision": "float16",
            "quantization": "none",
            "attention_backend": "eager",
            "placement": "full_cuda",
            "loads": 2,
        },
        "material": {
            "pair_count": len(manifest),
            "split_counts": {split: sum(row["split"] == split for row in manifest) for split in SPLITS},
            "manifest_digest": digest(manifest),
            "discovery_split": "discovery",
            "holdout_splits": list(HOLDOUT_SPLITS),
            "all_pairs_no_selection": True,
        },
        "contracts": {
            "C0": "same load, same batch and sequence geometry, immediate and delayed exact repeats",
            "C1": "same load, changed batch/sequence/cache geometry, functional equivalence under discovery-derived held-out envelope",
            "C2": "independent model reload, identical geometry, functional equivalence under discovery-derived held-out envelope",
            "C3": "independent reload of stored donor scoring states into recipient, compared with live donor, live patch, and identity patch",
        },
        "reference_variants": {
            "exact": list(EXACT_VARIANTS),
            "shape": list(SHAPE_VARIANTS),
            "suffix_tokens": 4,
            "batch8_policy": "duplicate the four candidate rows once and compare the first four",
            "batch1_policy": "run each candidate row separately and concatenate",
            "prompt_only_policy": "compare only the generation-boundary state and full-vocabulary logits",
        },
        "readouts": [
            "hidden relative RMS and max absolute drift",
            "full-vocabulary logit RMS and max absolute drift",
            "maximum probability drift",
            "top-1 agreement",
            "per-token candidate log probability",
            "complete candidate score and fixed margin drift",
            "drift relative to the Phase1224 donor-recipient target shift",
        ],
        "envelope": {
            "rule": "for each variant and metric, holdout maximum must not exceed two times discovery maximum plus 1e-12",
            "multiplier": ENVELOPE_MULTIPLIER,
            "epsilon": EPSILON,
            "functional_caps": FUNCTIONAL_CAPS,
            "discovery_does_not_change_source_or_variant_registry": True,
        },
        "authorization": {
            "phase1226_known_truth_library_if_all_contracts_pass": True,
            "qwen_mechanism_scan": False,
            "head_or_neuron_search": False,
            "threshold_relaxation_after_run": False,
        },
        "source_hashes": source_hashes,
    }
    protocol["protocol_digest"] = digest(protocol)
    return protocol


def materialize() -> None:
    if PROTOCOL_PATH.exists() or MANIFEST_PATH.exists():
        raise RuntimeError("Phase1225 protocol already materialized")
    manifest = build_manifest()
    protocol = build_protocol(manifest)
    write_jsonl(MANIFEST_PATH, manifest)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"status": "materialized", "pairs": len(manifest), "protocol_digest": protocol["protocol_digest"]}))


def verify_formal_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    protocol = read_json(PROTOCOL_PATH)
    payload = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    if protocol.get("protocol_digest") != digest(payload):
        raise RuntimeError("protocol digest drift")
    if not read_json(PREAUDIT_PATH).get("all_checks_passed"):
        raise RuntimeError("independent preaudit did not pass")
    manifest = read_jsonl(MANIFEST_PATH)
    if digest(manifest) != protocol["material"]["manifest_digest"]:
        raise RuntimeError("manifest digest drift")
    _source, _records, states = verify_source()
    return protocol, manifest, {row["state_id"]: row for row in states}


def model_readout(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    if not hasattr(model, "model") or not hasattr(model.model, "norm") or not hasattr(model, "lm_head"):
        raise RuntimeError("Qwen3 final readout modules not found")
    return model.lm_head(model.model.norm(hidden))


def capture_selected(
    model: Any,
    module: Any,
    input_ids: torch.Tensor,
    positions: list[int],
    use_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode(), p1224.CaptureLastLayer(module) as capture:
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=use_cache,
            logits_to_keep=1,
            return_dict=True,
        )
    if capture.calls != 1 or capture.value is None:
        raise RuntimeError("final-layer capture drift")
    selected = capture.value[:, positions, :].detach()
    with torch.inference_mode():
        logits = model_readout(model, selected).detach()
    del output
    return selected, logits


def patched_selected(
    model: Any,
    module: Any,
    input_ids: torch.Tensor,
    positions: list[int],
    replacement: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    with (
        torch.inference_mode(),
        p1224.MultiPositionPatch(module, positions, replacement) as patch,
        p1224.CaptureLastLayer(module) as capture,
    ):
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
    if patch.calls != 1 or capture.calls != 1 or capture.value is None:
        raise RuntimeError("patched capture drift")
    selected = capture.value[:, positions, :].detach()
    with torch.inference_mode():
        logits = model_readout(model, selected).detach()
    del output
    return selected, logits, float(patch.write_max_abs)


def score_bundle(
    logits: torch.Tensor,
    candidates: list[str],
    candidate_ids: dict[str, list[int]],
    donor_gold: str,
    recipient_gold: str,
) -> dict[str, Any]:
    scores, token_scores, finite = p1224.score_from_logits(logits, candidates, candidate_ids)
    return {
        "scores": scores,
        "token_scores": token_scores,
        "fixed_margin": p1224.fixed_margin(scores, donor_gold, recipient_gold),
        "finite": bool(finite),
        "top1_ids": logits.argmax(dim=-1).detach().cpu().tolist(),
    }


def comparison_metrics(
    hidden: torch.Tensor,
    logits: torch.Tensor,
    reference_hidden: torch.Tensor,
    reference_logits: torch.Tensor,
    bundle: dict[str, Any] | None,
    reference_bundle: dict[str, Any] | None,
    target_shift_abs: float,
) -> dict[str, Any]:
    hidden_float = hidden.float()
    ref_hidden_float = reference_hidden.float()
    logit_float = logits.float()
    ref_logit_float = reference_logits.float()
    hidden_diff = hidden_float - ref_hidden_float
    logit_diff = logit_float - ref_logit_float
    probability_diff = torch.softmax(logit_float, dim=-1) - torch.softmax(ref_logit_float, dim=-1)
    result: dict[str, Any] = {
        "finite": bool(torch.isfinite(hidden).all().item() and torch.isfinite(logits).all().item()),
        "hidden_relative_rms": float(
            torch.sqrt(torch.mean(hidden_diff * hidden_diff)).item()
            / (torch.sqrt(torch.mean(ref_hidden_float * ref_hidden_float)).item() + EPSILON)
        ),
        "hidden_max_abs": float(hidden_diff.abs().max().item()),
        "logit_rms": float(torch.sqrt(torch.mean(logit_diff * logit_diff)).item()),
        "logit_max_abs": float(logit_diff.abs().max().item()),
        "probability_max_abs": float(probability_diff.abs().max().item()),
        "top1_agreement": float((logits.argmax(dim=-1) == reference_logits.argmax(dim=-1)).float().mean().item()),
    }
    if bundle is not None and reference_bundle is not None:
        candidates = list(reference_bundle["scores"])
        score_drift = max(abs(bundle["scores"][key] - reference_bundle["scores"][key]) for key in candidates)
        margin_drift = abs(bundle["fixed_margin"] - reference_bundle["fixed_margin"])
        token_drift = max(
            abs(left - right)
            for key in candidates
            for left, right in zip(bundle["token_scores"][key], reference_bundle["token_scores"][key])
        )
        result.update({
            "score_max_abs": float(score_drift),
            "token_logprob_max_abs": float(token_drift),
            "fixed_margin_abs": float(margin_drift),
            "score_drift_ratio": float(score_drift / (target_shift_abs + EPSILON)),
            "margin_drift_ratio": float(margin_drift / (target_shift_abs + EPSILON)),
        })
    return result


def run_reference() -> None:
    protocol, manifest, state_by_id = verify_formal_inputs()
    if REFERENCE_RECORD_PATH.exists() or REFERENCE_ARRAY_PATH.exists() or REFERENCE_SUMMARY_PATH.exists():
        raise RuntimeError("Phase1225 reference output already exists")
    started = time.time()
    records: list[dict[str, Any]] = []
    hidden_array: np.ndarray | None = None
    boundary_logits_array: np.ndarray | None = None
    model = tokenizer = None
    try:
        model, tokenizer, device, placement = load_fp16("qwen3")
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or set(precision["parameter_dtypes"]) != {"float16"}:
            raise RuntimeError("Qwen3 is not pure FP16")
        layers = get_layers(model)
        if len(layers) != LAYER_COUNT or getattr(model.config, "_attn_implementation", None) != "eager":
            raise RuntimeError("Qwen3 numerical domain drift")
        module = layers[-1]
        hidden_size = int(model.config.hidden_size)
        vocab_size = int(model.config.vocab_size)
        hidden_array = np.zeros((len(manifest), 4, MAX_CONTINUATION, hidden_size), dtype=np.float16)
        boundary_logits_array = np.zeros((len(manifest), vocab_size), dtype=np.float16)
        eos_id = int(tokenizer.eos_token_id)

        for offset, item in enumerate(manifest):
            state = state_by_id[item["donor_state_id"]]
            candidates = list(item["candidates"])
            candidate_ids = {key: [int(value) for value in item["candidate_token_ids"][key]] for key in candidates}
            input_ids, continuation, boundary = p1224.make_batch(state, candidates, device)
            positions = list(range(boundary, boundary + continuation))
            ref_hidden, ref_logits = capture_selected(model, module, input_ids, positions)
            ref_bundle = score_bundle(ref_logits, candidates, candidate_ids, item["donor_gold"], item["recipient_gold"])

            variants: dict[str, Any] = {}
            immediate_hidden, immediate_logits = capture_selected(model, module, input_ids, positions)
            immediate_bundle = score_bundle(immediate_logits, candidates, candidate_ids, item["donor_gold"], item["recipient_gold"])
            variants["immediate_repeat"] = comparison_metrics(
                immediate_hidden, immediate_logits, ref_hidden, ref_logits, immediate_bundle, ref_bundle, item["target_shift_abs"]
            )

            batch1_hidden: list[torch.Tensor] = []
            batch1_logits: list[torch.Tensor] = []
            for row_index in range(input_ids.shape[0]):
                row_hidden, row_logits = capture_selected(model, module, input_ids[row_index : row_index + 1], positions)
                batch1_hidden.append(row_hidden)
                batch1_logits.append(row_logits)
            shape_hidden = torch.cat(batch1_hidden, dim=0)
            shape_logits = torch.cat(batch1_logits, dim=0)
            shape_bundle = score_bundle(shape_logits, candidates, candidate_ids, item["donor_gold"], item["recipient_gold"])
            variants["batch1"] = comparison_metrics(
                shape_hidden, shape_logits, ref_hidden, ref_logits, shape_bundle, ref_bundle, item["target_shift_abs"]
            )

            batch8_ids = torch.cat([input_ids, input_ids], dim=0)
            shape_hidden8, shape_logits8 = capture_selected(model, module, batch8_ids, positions)
            shape_hidden = shape_hidden8[:4]
            shape_logits = shape_logits8[:4]
            shape_bundle = score_bundle(shape_logits, candidates, candidate_ids, item["donor_gold"], item["recipient_gold"])
            variants["batch8"] = comparison_metrics(
                shape_hidden, shape_logits, ref_hidden, ref_logits, shape_bundle, ref_bundle, item["target_shift_abs"]
            )

            suffix = torch.full((input_ids.shape[0], 4), eos_id, dtype=torch.long, device=device)
            suffix_ids = torch.cat([input_ids, suffix], dim=1)
            shape_hidden, shape_logits = capture_selected(model, module, suffix_ids, positions)
            shape_bundle = score_bundle(shape_logits, candidates, candidate_ids, item["donor_gold"], item["recipient_gold"])
            variants["suffix4"] = comparison_metrics(
                shape_hidden, shape_logits, ref_hidden, ref_logits, shape_bundle, ref_bundle, item["target_shift_abs"]
            )

            shape_hidden, shape_logits = capture_selected(model, module, input_ids, positions, use_cache=True)
            shape_bundle = score_bundle(shape_logits, candidates, candidate_ids, item["donor_gold"], item["recipient_gold"])
            variants["cache_true"] = comparison_metrics(
                shape_hidden, shape_logits, ref_hidden, ref_logits, shape_bundle, ref_bundle, item["target_shift_abs"]
            )

            prompt_ids = torch.tensor([state["input_ids"]], dtype=torch.long, device=device)
            prompt_hidden, prompt_logits = capture_selected(model, module, prompt_ids, [boundary])
            variants["prompt_only"] = comparison_metrics(
                prompt_hidden,
                prompt_logits,
                ref_hidden[0:1, 0:1],
                ref_logits[0:1, 0:1],
                None,
                None,
                item["target_shift_abs"],
            )

            delayed_hidden, delayed_logits = capture_selected(model, module, input_ids, positions)
            delayed_bundle = score_bundle(delayed_logits, candidates, candidate_ids, item["donor_gold"], item["recipient_gold"])
            variants["delayed_repeat"] = comparison_metrics(
                delayed_hidden, delayed_logits, ref_hidden, ref_logits, delayed_bundle, ref_bundle, item["target_shift_abs"]
            )

            hidden_array[offset, :, :continuation] = ref_hidden.detach().cpu().numpy().astype(np.float16)
            boundary_logits_array[offset] = ref_logits[0, 0].detach().cpu().numpy().astype(np.float16)
            record: dict[str, Any] = {
                "schema_version": "phase1225.reference-record.v1",
                "phase": PHASE,
                "protocol_digest": protocol["protocol_digest"],
                "pair_id": item["pair_id"],
                "index": offset,
                "scope": item["scope"],
                "split": item["split"],
                "continuation_length": continuation,
                "target_shift_abs": item["target_shift_abs"],
                "reference": ref_bundle,
                "variants": variants,
            }
            record["record_digest"] = digest(record)
            records.append(record)
            del (
                input_ids, ref_hidden, ref_logits, immediate_hidden, immediate_logits,
                batch1_hidden, batch1_logits, shape_hidden, shape_logits,
                shape_hidden8, shape_logits8, batch8_ids, suffix_ids, suffix,
                prompt_ids, prompt_hidden, prompt_logits, delayed_hidden, delayed_logits,
            )
            if (offset + 1) % 16 == 0:
                print(f"[phase1225/reference] {offset + 1}/{len(manifest)}", flush=True)

        write_jsonl(REFERENCE_RECORD_PATH, records)
        REFERENCE_ARRAY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REFERENCE_ARRAY_PATH.open("wb") as handle:
            np.savez_compressed(handle, hidden=hidden_array, boundary_logits=boundary_logits_array)
        summary: dict[str, Any] = {
            "phase": PHASE,
            "stage": "reference",
            "created_at": utc_now(),
            "protocol_digest": protocol["protocol_digest"],
            "record_count": len(records),
            "record_digest": digest(records),
            "array_sha256": file_sha256(REFERENCE_ARRAY_PATH),
            "array_shapes": {"hidden": list(hidden_array.shape), "boundary_logits": list(boundary_logits_array.shape)},
            "precision_audit": precision,
            "placement": placement,
            "attention_backend": getattr(model.config, "_attn_implementation", None),
            "elapsed_seconds": time.time() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        }
        summary["summary_digest"] = digest(summary)
        write_json(REFERENCE_SUMMARY_PATH, summary)
        print(canonical_json({"status": "reference_complete", "records": len(records), "summary_digest": summary["summary_digest"]}))
    finally:
        if model is not None:
            release_fp16(model)
        del model, tokenizer, hidden_array, boundary_logits_array
        gc.collect()


def patch_condition(
    logits: torch.Tensor,
    candidates: list[str],
    candidate_ids: dict[str, list[int]],
    donor_gold: str,
    recipient_gold: str,
    donor_bundle: dict[str, Any],
    recipient_bundle: dict[str, Any],
    write_max_abs: float,
) -> dict[str, Any]:
    bundle = score_bundle(logits, candidates, candidate_ids, donor_gold, recipient_gold)
    target = donor_bundle["fixed_margin"] - recipient_bundle["fixed_margin"]
    completion = (bundle["fixed_margin"] - recipient_bundle["fixed_margin"]) / target if abs(target) > EPSILON else 0.0
    score_error = max(abs(bundle["scores"][key] - donor_bundle["scores"][key]) for key in candidates)
    top1_ref = np.asarray(donor_bundle["top1_ids"], dtype=np.int64)
    top1_now = np.asarray(bundle["top1_ids"], dtype=np.int64)
    return {
        "bundle": bundle,
        "completion": float(completion),
        "score_max_abs_vs_donor": float(score_error),
        "score_drift_ratio": float(score_error / (abs(target) + EPSILON)),
        "top1_agreement_vs_donor": float(np.mean(top1_now == top1_ref)),
        "write_max_abs": float(write_max_abs),
    }


def run_reload() -> None:
    protocol, manifest, state_by_id = verify_formal_inputs()
    if not REFERENCE_RECORD_PATH.exists() or not REFERENCE_ARRAY_PATH.exists() or not REFERENCE_SUMMARY_PATH.exists():
        raise RuntimeError("reference stage is incomplete")
    if RELOAD_RECORD_PATH.exists() or RELOAD_SUMMARY_PATH.exists():
        raise RuntimeError("Phase1225 reload output already exists")
    reference_records = read_jsonl(REFERENCE_RECORD_PATH)
    reference_by_id = {row["pair_id"]: row for row in reference_records}
    reference_summary = read_json(REFERENCE_SUMMARY_PATH)
    if reference_summary["record_digest"] != digest(reference_records) or reference_summary["array_sha256"] != file_sha256(REFERENCE_ARRAY_PATH):
        raise RuntimeError("reference output drift")
    arrays = np.load(REFERENCE_ARRAY_PATH, mmap_mode="r")
    started = time.time()
    records: list[dict[str, Any]] = []
    model = None
    try:
        model, _tokenizer, device, placement = load_fp16("qwen3")
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or set(precision["parameter_dtypes"]) != {"float16"}:
            raise RuntimeError("Qwen3 is not pure FP16")
        layers = get_layers(model)
        if len(layers) != LAYER_COUNT or getattr(model.config, "_attn_implementation", None) != "eager":
            raise RuntimeError("Qwen3 reload numerical domain drift")
        module = layers[-1]

        for offset, item in enumerate(manifest):
            ref_record = reference_by_id[item["pair_id"]]
            donor = state_by_id[item["donor_state_id"]]
            recipient = state_by_id[item["recipient_state_id"]]
            candidates = list(item["candidates"])
            candidate_ids = {key: [int(value) for value in item["candidate_token_ids"][key]] for key in candidates}
            donor_input, continuation, boundary = p1224.make_batch(donor, candidates, device)
            recipient_input, recipient_continuation, recipient_boundary = p1224.make_batch(recipient, candidates, device)
            if continuation != recipient_continuation or boundary != recipient_boundary:
                raise RuntimeError("pair geometry drift")
            positions = list(range(boundary, boundary + continuation))
            donor_hidden, donor_logits = capture_selected(model, module, donor_input, positions)
            recipient_hidden, recipient_logits = capture_selected(model, module, recipient_input, positions)
            donor_bundle = score_bundle(donor_logits, candidates, candidate_ids, item["donor_gold"], item["recipient_gold"])
            recipient_bundle = score_bundle(recipient_logits, candidates, candidate_ids, item["donor_gold"], item["recipient_gold"])

            stored_hidden = torch.tensor(
                arrays["hidden"][offset, :, :continuation].astype(np.float32),
                device=device,
                dtype=donor_hidden.dtype,
            )
            stored_boundary_logits = torch.tensor(
                arrays["boundary_logits"][offset].astype(np.float32),
                device=device,
                dtype=donor_logits.dtype,
            )[None, None, :]
            stored_bundle = ref_record["reference"]
            cross_load = comparison_metrics(
                donor_hidden,
                donor_logits[:, 0:1],
                stored_hidden,
                stored_boundary_logits.expand(donor_logits.shape[0], -1, -1),
                donor_bundle,
                stored_bundle,
                item["target_shift_abs"],
            )

            stored_replacement = recipient_hidden.new_zeros((recipient_hidden.shape[0], recipient_input.shape[1], recipient_hidden.shape[-1]))
            stored_replacement[:, positions, :] = stored_hidden
            live_replacement = recipient_hidden.new_zeros((recipient_hidden.shape[0], recipient_input.shape[1], recipient_hidden.shape[-1]))
            live_replacement[:, positions, :] = donor_hidden
            zero_replacement = recipient_hidden.new_zeros((recipient_hidden.shape[0], recipient_input.shape[1], recipient_hidden.shape[-1]))
            zero_replacement[:, positions, :] = recipient_hidden

            _stored_out, stored_logits, stored_write = patched_selected(
                model, module, recipient_input, positions, stored_replacement
            )
            _live_out, live_logits, live_write = patched_selected(
                model, module, recipient_input, positions, live_replacement
            )
            _zero_out, zero_logits, zero_write = patched_selected(
                model, module, recipient_input, positions, zero_replacement
            )
            conditions = {
                "stored": patch_condition(
                    stored_logits, candidates, candidate_ids, item["donor_gold"], item["recipient_gold"],
                    donor_bundle, recipient_bundle, stored_write,
                ),
                "live": patch_condition(
                    live_logits, candidates, candidate_ids, item["donor_gold"], item["recipient_gold"],
                    donor_bundle, recipient_bundle, live_write,
                ),
                "zero": patch_condition(
                    zero_logits, candidates, candidate_ids, item["donor_gold"], item["recipient_gold"],
                    donor_bundle, recipient_bundle, zero_write,
                ),
            }
            zero_error = max(abs(conditions["zero"]["bundle"]["scores"][key] - recipient_bundle["scores"][key]) for key in candidates)
            conditions["zero"]["score_max_abs_vs_recipient"] = float(zero_error)

            record: dict[str, Any] = {
                "schema_version": "phase1225.reload-record.v1",
                "phase": PHASE,
                "protocol_digest": protocol["protocol_digest"],
                "pair_id": item["pair_id"],
                "index": offset,
                "scope": item["scope"],
                "split": item["split"],
                "continuation_length": continuation,
                "target_shift_abs": item["target_shift_abs"],
                "cross_load": cross_load,
                "live_donor": donor_bundle,
                "live_recipient": recipient_bundle,
                "conditions": conditions,
            }
            record["record_digest"] = digest(record)
            records.append(record)
            del (
                donor_input, recipient_input, donor_hidden, donor_logits, recipient_hidden, recipient_logits,
                stored_hidden, stored_boundary_logits, stored_replacement, live_replacement, zero_replacement,
                stored_logits, live_logits, zero_logits, _stored_out, _live_out, _zero_out,
            )
            if (offset + 1) % 16 == 0:
                print(f"[phase1225/reload] {offset + 1}/{len(manifest)}", flush=True)

        write_jsonl(RELOAD_RECORD_PATH, records)
        summary: dict[str, Any] = {
            "phase": PHASE,
            "stage": "reload",
            "created_at": utc_now(),
            "protocol_digest": protocol["protocol_digest"],
            "reference_summary_digest": reference_summary["summary_digest"],
            "record_count": len(records),
            "record_digest": digest(records),
            "precision_audit": precision,
            "placement": placement,
            "attention_backend": getattr(model.config, "_attn_implementation", None),
            "elapsed_seconds": time.time() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        }
        summary["summary_digest"] = digest(summary)
        write_json(RELOAD_SUMMARY_PATH, summary)
        print(canonical_json({"status": "reload_complete", "records": len(records), "summary_digest": summary["summary_digest"]}))
    finally:
        arrays.close()
        if model is not None:
            release_fp16(model)
        del model
        gc.collect()


def envelope(discovery: list[float]) -> float:
    return float(ENVELOPE_MULTIPLIER * max(discovery, default=0.0) + EPSILON)


def evaluate(reference: list[dict[str, Any]], reload: list[dict[str, Any]]) -> dict[str, Any]:
    discovery_reference = [row for row in reference if row["split"] == "discovery"]
    holdout_reference = [row for row in reference if row["split"] in HOLDOUT_SPLITS]
    discovery_reload = [row for row in reload if row["split"] == "discovery"]
    holdout_reload = [row for row in reload if row["split"] in HOLDOUT_SPLITS]
    caps = FUNCTIONAL_CAPS

    c0_details: dict[str, Any] = {}
    c0_pass = True
    for variant in EXACT_VARIANTS:
        rows = [row["variants"][variant] for row in reference]
        detail = {
            "hidden_max_abs": max(row["hidden_max_abs"] for row in rows),
            "logit_max_abs": max(row["logit_max_abs"] for row in rows),
            "score_max_abs": max(row["score_max_abs"] for row in rows),
            "top1_min": min(row["top1_agreement"] for row in rows),
        }
        detail["passed"] = (
            detail["hidden_max_abs"] <= caps["exact_abs"]
            and detail["logit_max_abs"] <= caps["exact_abs"]
            and detail["score_max_abs"] <= caps["exact_abs"]
            and detail["top1_min"] >= caps["top1_agreement"]
        )
        c0_details[variant] = detail
        c0_pass = c0_pass and detail["passed"]

    c1_details: dict[str, Any] = {}
    c1_pass = True
    for variant in SHAPE_VARIANTS:
        disc = [row["variants"][variant] for row in discovery_reference]
        hold = [row["variants"][variant] for row in holdout_reference]
        metric_names = ["hidden_relative_rms", "logit_max_abs", "probability_max_abs"]
        if variant != "prompt_only":
            metric_names.extend(["margin_drift_ratio", "score_drift_ratio"])
        metric_detail: dict[str, Any] = {}
        variant_pass = True
        for name in metric_names:
            disc_values = [float(row[name]) for row in disc]
            hold_values = [float(row[name]) for row in hold]
            limit = envelope(disc_values)
            cap_name = name
            cap = float(caps[cap_name])
            observed = max(hold_values)
            passed = observed <= limit and observed <= cap
            metric_detail[name] = {
                "discovery_max": max(disc_values),
                "holdout_max": observed,
                "envelope": limit,
                "functional_cap": cap,
                "passed": passed,
            }
            variant_pass = variant_pass and passed
        top1_min = min(float(row["top1_agreement"]) for row in hold)
        top1_pass = top1_min >= caps["top1_agreement"]
        metric_detail["top1"] = {"holdout_min": top1_min, "passed": top1_pass}
        variant_pass = variant_pass and top1_pass
        c1_details[variant] = {"metrics": metric_detail, "passed": variant_pass}
        c1_pass = c1_pass and variant_pass

    c2_metric_names = [
        "hidden_relative_rms", "logit_max_abs", "probability_max_abs",
        "margin_drift_ratio", "score_drift_ratio",
    ]
    c2_details: dict[str, Any] = {}
    c2_pass = True
    for name in c2_metric_names:
        disc_values = [float(row["cross_load"][name]) for row in discovery_reload]
        hold_values = [float(row["cross_load"][name]) for row in holdout_reload]
        limit = envelope(disc_values)
        cap = float(caps[name])
        observed = max(hold_values)
        passed = observed <= limit and observed <= cap
        c2_details[name] = {
            "discovery_max": max(disc_values),
            "holdout_max": observed,
            "envelope": limit,
            "functional_cap": cap,
            "passed": passed,
        }
        c2_pass = c2_pass and passed
    c2_top1 = min(float(row["cross_load"]["top1_agreement"]) for row in holdout_reload)
    c2_details["top1"] = {"holdout_min": c2_top1, "passed": c2_top1 >= caps["top1_agreement"]}
    c2_pass = c2_pass and c2_details["top1"]["passed"]

    live = [row["conditions"]["live"] for row in reload]
    zero = [row["conditions"]["zero"] for row in reload]
    stored_discovery = [row["conditions"]["stored"] for row in discovery_reload]
    stored_holdout = [row["conditions"]["stored"] for row in holdout_reload]
    stored_limit = envelope([float(row["score_drift_ratio"]) for row in stored_discovery])
    c3_details = {
        "live_score_error_max": max(row["score_max_abs_vs_donor"] for row in live),
        "live_completion_median": median([row["completion"] for row in live]),
        "live_write_error_max": max(row["write_max_abs"] for row in live),
        "zero_score_error_max": max(row["score_max_abs_vs_recipient"] for row in zero),
        "zero_completion_abs_max": max(abs(row["completion"]) for row in zero),
        "zero_write_error_max": max(row["write_max_abs"] for row in zero),
        "stored_holdout_completion_median": median([row["completion"] for row in stored_holdout]),
        "stored_holdout_positive_fraction": fraction([row["completion"] > 0 for row in stored_holdout]),
        "stored_holdout_score_ratio_max": max(row["score_drift_ratio"] for row in stored_holdout),
        "stored_score_ratio_envelope": stored_limit,
        "stored_holdout_top1_min": min(row["top1_agreement_vs_donor"] for row in stored_holdout),
    }
    c3_pass = (
        c3_details["live_score_error_max"] <= caps["exact_abs"]
        and c3_details["live_completion_median"] >= 0.999
        and c3_details["live_write_error_max"] <= caps["exact_abs"]
        and c3_details["zero_score_error_max"] <= caps["exact_abs"]
        and c3_details["zero_completion_abs_max"] <= caps["exact_abs"]
        and c3_details["zero_write_error_max"] <= caps["exact_abs"]
        and c3_details["stored_holdout_completion_median"] >= caps["stored_completion_median"]
        and c3_details["stored_holdout_positive_fraction"] >= caps["stored_positive_fraction"]
        and c3_details["stored_holdout_score_ratio_max"] <= stored_limit
        and c3_details["stored_holdout_score_ratio_max"] <= caps["score_drift_ratio"]
        and c3_details["stored_holdout_top1_min"] >= caps["top1_agreement"]
    )
    c3_details["passed"] = c3_pass

    finite = all(
        row["reference"]["finite"] and all(value["finite"] for value in row["variants"].values())
        for row in reference
    ) and all(
        row["live_donor"]["finite"] and row["live_recipient"]["finite"]
        and all(value["bundle"]["finite"] for value in row["conditions"].values())
        for row in reload
    )
    contracts = {"C0": c0_pass, "C1": c1_pass, "C2": c2_pass, "C3": c3_pass}
    return {
        "finite": finite,
        "contracts": contracts,
        "details": {"C0": c0_details, "C1": c1_details, "C2": c2_details, "C3": c3_details},
        "passed": bool(finite and all(contracts.values())),
    }


def analyze() -> None:
    if FINAL_PATH.exists():
        raise RuntimeError("Phase1225 final already exists")
    protocol = read_json(PROTOCOL_PATH)
    reference = read_jsonl(REFERENCE_RECORD_PATH)
    reload = read_jsonl(RELOAD_RECORD_PATH)
    reference_summary = read_json(REFERENCE_SUMMARY_PATH)
    reload_summary = read_json(RELOAD_SUMMARY_PATH)
    if reference_summary["record_digest"] != digest(reference) or reload_summary["record_digest"] != digest(reload):
        raise RuntimeError("run record digest drift")
    result = evaluate(reference, reload)
    passed = bool(result["passed"])
    final: dict[str, Any] = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": "numerical_domain_confirmed" if passed else "numerical_domain_not_confirmed",
        "protocol_digest": protocol["protocol_digest"],
        "reference_summary_digest": reference_summary["summary_digest"],
        "reload_summary_digest": reload_summary["summary_digest"],
        "result": result,
        "k_item": {
            "identifier": "K202",
            "evidence_grade": "E3-NUMERICAL-DOMAIN" if passed else "E3-NUMERICAL-BOUNDARY",
            "statement": (
                "Qwen3 eager-FP16 final-layer intervention contracts C0-C3 passed on held-out pairs."
                if passed else
                "At least one typed Qwen3 eager-FP16 final-layer numerical contract failed on held-out pairs."
            ),
            "scope": "Qwen3-4B; eager FP16; Phase1224 final layer; 160 generated pairs; instrument only",
        },
        "authorization": {
            "automatic_execution": passed,
            "next_experiment": "Phase1226 known-truth finite mechanism library" if passed else None,
            "qwen_mechanism_scan": False,
            "reason": "all numerical contracts passed" if passed else "repair or narrow the failed typed numerical contract",
        },
        "new_mathematics_required": False,
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": final["status"], "contracts": result["contracts"], "final_digest": final["final_digest"]}))


def selftest() -> None:
    ref_hidden = torch.tensor([[[1.0, 2.0]]])
    ref_logits = torch.tensor([[[0.0, 1.0, 2.0]]])
    metrics = comparison_metrics(ref_hidden, ref_logits, ref_hidden.clone(), ref_logits.clone(), None, None, 2.0)
    if any(metrics[key] != 0.0 for key in ("hidden_relative_rms", "hidden_max_abs", "logit_rms", "logit_max_abs", "probability_max_abs")):
        raise RuntimeError("identity comparison selftest failed")
    if metrics["top1_agreement"] != 1.0 or envelope([0.5]) != 1.000000000001:
        raise RuntimeError("envelope selftest failed")
    print("phase1225 selftest passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("selftest", "materialize", "run-reference", "run-reload", "analyze"))
    args = parser.parse_args()
    if args.stage == "selftest":
        selftest()
    elif args.stage == "materialize":
        materialize()
    elif args.stage == "run-reference":
        run_reference()
    elif args.stage == "run-reload":
        run_reload()
    else:
        analyze()


if __name__ == "__main__":
    main()
