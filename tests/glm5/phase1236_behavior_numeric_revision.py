#!/usr/bin/env python3
"""Phase1236 Revision 1: auditable non-finite behavior runner.

The frozen Phase1236 scientific protocol is unchanged.  This runner repairs a
serialization failure discovered after the GLM4 forward pass: non-finite FP16
candidate scores are represented as null, marked numerically ineligible, and
never assigned a winner.  Stage checkpoints make the long behavior run
resumable without changing any model call, material, decoder, or gate.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1236_global_functional_structure_identification as base


SCRIPT = Path(__file__).resolve()
REVISION_AUDIT_SCRIPT = TEST_ROOT / "phase1236_behavior_numeric_revision_audit.py"
REVISION_PATH = base.OUT_ROOT / "protocol/behavior_numeric_revision1.json"
REVISION_AUDIT_PATH = base.OUT_ROOT / "audit/behavior_numeric_revision1_audit.json"
CHECKPOINT_ROOT = base.OUT_ROOT / "runtime/revision1_checkpoints"
FAILURE_LOG = base.OUT_ROOT / "runtime/glm4_20260811_191219.stderr.log"
REVISION_ID = "phase1236.behavior_numeric_revision1"


def checkpoint_path(model_name: str, stage: str) -> Path:
    return CHECKPOINT_ROOT / model_name / f"{stage}.json"


def write_strict_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_checkpoint(model_name: str, stage: str) -> tuple[Any, dict[str, Any]] | None:
    path = checkpoint_path(model_name, stage)
    if not path.exists():
        return None
    value = base.read_json(path)
    if value.get("revision_id") != REVISION_ID or value.get("model") != model_name:
        raise RuntimeError(f"invalid {stage} checkpoint identity")
    payload = value["payload"]
    if value.get("payload_digest") != base.digest(payload):
        raise RuntimeError(f"invalid {stage} checkpoint digest")
    return payload, value["runtime"]


def save_checkpoint(model_name: str, stage: str, payload: Any, runtime: dict[str, Any]) -> None:
    value = {
        "revision_id": REVISION_ID,
        "model": model_name,
        "stage": stage,
        "contract_digest": base.read_json(base.CONTRACT_PATH)["contract_digest"],
        "payload": payload,
        "payload_digest": base.digest(payload),
        "runtime": runtime,
    }
    write_strict_json(checkpoint_path(model_name, stage), value)


def preregister_revision() -> None:
    if REVISION_PATH.exists():
        raise RuntimeError("numeric revision already registered")
    contract = base.read_json(base.CONTRACT_PATH)
    preaudit = base.read_json(base.PREAUDIT_PATH)
    if preaudit.get("all_checks_passed") is not True:
        raise RuntimeError("base preaudit did not pass")
    if not FAILURE_LOG.exists():
        raise RuntimeError("the triggering GLM4 failure log is missing")
    value: dict[str, Any] = {
        "phase": base.PHASE,
        "revision_id": REVISION_ID,
        "created_at_utc": base.utc_now(),
        "base_contract_digest": contract["contract_digest"],
        "base_source_hashes": contract["source_hashes"],
        "revision_source_hashes": {
            "runner": base.file_sha256(SCRIPT),
            "independent_audit": base.file_sha256(REVISION_AUDIT_SCRIPT),
        },
        "trigger": {
            "model": "glm4",
            "failure_type": "strict_json_rejected_nonfinite_fp16_candidate_score",
            "failure_log": str(FAILURE_LOG.relative_to(ROOT)).replace("\\", "/"),
            "failure_log_sha256": base.file_sha256(FAILURE_LOG),
        },
        "allowed_changes": [
            "serialize non-finite candidate aggregates as JSON null",
            "mark any row containing a non-finite candidate score numerically ineligible",
            "return no candidate winner when any compared aggregate is non-finite",
            "write strict stage checkpoints after each unchanged inference lane",
        ],
        "frozen_invariants": {
            "material_digest": contract["material"]["material_digest"],
            "manifest_digests": {
                model: contract["manifest_summaries"][model]["manifest_digest"] for model in base.MODELS
            },
            "thresholds": contract["thresholds"],
            "models": list(base.MODELS),
            "precision": contract["execution"]["precision"],
            "quantization": contract["execution"]["quantization"],
            "generation_budget": base.GENERATION_BUDGET,
        },
        "scientific_claim": "instrument repair only; no scientific gate or material revision",
    }
    value["revision_digest"] = base.digest(value)
    base.write_json(REVISION_PATH, value)
    print(base.canonical_json({"status": "numeric_revision_registered", "revision_digest": value["revision_digest"]}))


def verify_revision() -> dict[str, Any]:
    contract, _material, _manifests = base.verify_frozen()
    revision = base.read_json(REVISION_PATH)
    if revision.get("revision_digest") != base.digest(base.strip_digest(revision, "revision_digest")):
        raise RuntimeError("numeric revision digest mismatch")
    if revision.get("base_contract_digest") != contract["contract_digest"]:
        raise RuntimeError("numeric revision contract mismatch")
    expected = {
        "runner": base.file_sha256(SCRIPT),
        "independent_audit": base.file_sha256(REVISION_AUDIT_SCRIPT),
    }
    if revision.get("revision_source_hashes") != expected:
        raise RuntimeError("numeric revision source hash mismatch")
    audit = base.read_json(REVISION_AUDIT_PATH)
    if audit.get("all_checks_passed") is not True or audit.get("revision_digest") != revision["revision_digest"]:
        raise RuntimeError("numeric revision independent audit missing or failed")
    return revision


def direct_candidate_scores(
    model: Any,
    device: torch.device,
    manifest: list[dict[str, Any]],
    field: str,
    batch_rows: int,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    entries = []
    for row in manifest:
        suffixes = row[field]
        continuation_length = len(next(iter(suffixes.values())))
        entries.append({**row, "total_length": len(row["input_ids"]) + continuation_length})
    result: dict[str, dict[str, dict[str, Any]]] = {}
    started = time.time()
    batches = 0
    all_finite = True
    nonfinite_candidates = 0
    for batch in base.grouped_batches(entries, batch_rows, "total_length"):
        sequences = []
        metadata = []
        for row in batch:
            for candidate in row["candidates"]:
                continuation = [int(value) for value in row[field][candidate]]
                sequences.append([int(value) for value in row["input_ids"]] + continuation)
                metadata.append((row, candidate, continuation))
        input_ids = torch.tensor(sequences, dtype=torch.long, device=device)
        continuation_length = len(metadata[0][2])
        with torch.inference_mode():
            output = base.model_forward(
                model,
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=continuation_length + 1,
                return_dict=True,
            )
        output_start = input_ids.shape[1] - output.logits.shape[1]
        for index, (row, candidate, continuation) in enumerate(metadata):
            token_scores: list[float] = []
            vocab_finite = True
            prompt_length = len(row["input_ids"])
            for offset, token_id in enumerate(continuation):
                logits = output.logits[index, prompt_length + offset - 1 - output_start].float()
                vocab_finite = vocab_finite and bool(torch.isfinite(logits).all().item())
                score = logits[int(token_id)] - torch.logsumexp(logits, dim=-1)
                token_scores.append(float(score.item()))
            candidate_finite = vocab_finite and all(math.isfinite(value) for value in token_scores)
            aggregate = float(sum(token_scores)) if candidate_finite else None
            result.setdefault(row["item_id"], {})[candidate] = {
                "sum_log_probability": aggregate,
                "mean_log_probability": aggregate / len(token_scores) if aggregate is not None else None,
                "token_count": len(token_scores),
                "all_vocab_logits_finite": candidate_finite,
            }
            all_finite = all_finite and candidate_finite
            nonfinite_candidates += int(not candidate_finite)
        del output, input_ids
        batches += 1
        if batches % 50 == 0:
            print(f"[phase1236-revision1/{field}] batches={batches}", flush=True)
    runtime = {
        "field": field,
        "batch_count": batches,
        "elapsed_seconds": time.time() - started,
        "all_finite": all_finite,
        "nonfinite_candidate_count": nonfinite_candidates,
    }
    return result, runtime


def unique_winner(scores: dict[str, dict[str, Any]]) -> str | None:
    values = [entry.get("sum_log_probability") for entry in scores.values()]
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
        return None
    ordered = sorted(scores, key=lambda key: float(scores[key]["sum_log_probability"]), reverse=True)
    if len(ordered) > 1 and abs(
        float(scores[ordered[0]]["sum_log_probability"]) - float(scores[ordered[1]]["sum_log_probability"])
    ) <= base.TIE_TOLERANCE:
        return None
    return ordered[0]


def load_or_run_lanes(model_name: str, manifest: list[dict[str, Any]]) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    stages = {
        "content_scores": read_checkpoint(model_name, "content_scores"),
        "contract_scores": read_checkpoint(model_name, "contract_scores"),
        "generations": read_checkpoint(model_name, "generations"),
    }
    model = None
    placement = None
    precision = None
    try:
        if any(value is None for value in stages.values()):
            model, tokenizer, device, placement = base.load_fp16(model_name)
            precision = base.quantization_audit(model)
            if precision["has_quantized_modules"] or set(precision["parameter_dtypes"]) != {"float16"}:
                raise RuntimeError(f"{model_name} numerical contract failed")
            write_strict_json(
                checkpoint_path(model_name, "execution_metadata"),
                {
                    "revision_id": REVISION_ID,
                    "model": model_name,
                    "precision": precision,
                    "placement": placement,
                },
            )
            batch_rows = {"qwen3": 8, "glm4": 2, "deepseek7b": 4}[model_name]
            generation_batch = {"qwen3": 16, "glm4": 2, "deepseek7b": 4}[model_name]
            if stages["content_scores"] is None:
                payload, runtime = direct_candidate_scores(
                    model, device, manifest, "content_candidate_token_ids", batch_rows
                )
                save_checkpoint(model_name, "content_scores", payload, runtime)
                stages["content_scores"] = (payload, runtime)
            if stages["contract_scores"] is None:
                payload, runtime = direct_candidate_scores(
                    model, device, manifest, "contract_candidate_token_ids", batch_rows
                )
                save_checkpoint(model_name, "contract_scores", payload, runtime)
                stages["contract_scores"] = (payload, runtime)
            if stages["generations"] is None:
                payload, runtime = base.greedy_generation(model, tokenizer, device, manifest, generation_batch)
                save_checkpoint(model_name, "generations", payload, runtime)
                stages["generations"] = (payload, runtime)
    finally:
        if model is not None:
            base.release_fp16(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if precision is None or placement is None:
        metadata_path = checkpoint_path(model_name, "execution_metadata")
        if not metadata_path.exists():
            raise RuntimeError("resumed checkpoints require execution metadata")
        metadata = base.read_json(metadata_path)
        precision, placement = metadata["precision"], metadata["placement"]
    return stages["content_scores"], stages["contract_scores"], stages["generations"], precision, placement


def run_behavior(model_name: str) -> None:
    revision = verify_revision()
    if base.behavior_raw_path(model_name).exists() or base.behavior_summary_path(model_name).exists():
        raise RuntimeError(f"{model_name} behavior output already exists")
    contract, material, manifests = base.verify_frozen()
    manifest = manifests[model_name]
    material_by_id = {row["item_id"]: row for row in material}
    started = time.time()
    (content_scores, content_runtime), (contract_scores, contract_runtime), (generations, generation_runtime), precision, placement = load_or_run_lanes(model_name, manifest)
    raw = []
    for manifest_row in manifest:
        item_id = manifest_row["item_id"]
        row = material_by_id[item_id]
        content_prediction = unique_winner(content_scores[item_id])
        contract_prediction = unique_winner(contract_scores[item_id])
        generation = generations[item_id]
        parsed = base.parse_output(generation["generated_text"], row["candidates"], row["expected_exact"], row["protocol"])
        value: dict[str, Any] = {
            "phase": base.PHASE,
            "schema_version": "phase1236.behavior_row.v1",
            "model": model_name,
            "contract_digest": contract["contract_digest"],
            "numeric_revision_digest": revision["revision_digest"],
            "item_id": item_id,
            "pair_id": row["pair_id"],
            "base_pair_id": row["base_pair_id"],
            "world_id": row["world_id"],
            "world_index": row["world_index"],
            "partition": row["partition"],
            "topology": row["topology"],
            "template_index": row["template_index"],
            "protocol": row["protocol"],
            "binding_state": row["binding_state"],
            "query_index": row["query_index"],
            "gold": row["gold"],
            "candidates": row["candidates"],
            "expected_exact": row["expected_exact"],
            "content_candidate_scores": content_scores[item_id],
            "contract_candidate_scores": contract_scores[item_id],
            "content_prediction": content_prediction,
            "contract_prediction": contract_prediction,
            "content_score_correct": content_prediction == row["gold"],
            "contract_score_correct": contract_prediction == row["gold"],
            "candidate_scores_finite": all(
                candidate["all_vocab_logits_finite"]
                for scores in (content_scores[item_id], contract_scores[item_id])
                for candidate in scores.values()
            ),
            "generation": generation,
            "generation_parse": parsed,
            "generation_content_correct": parsed["prediction"] == row["gold"],
            "generation_exact": parsed["exact"],
            "generation_format_valid": parsed["format_valid"],
        }
        value["behavior_row_digest"] = base.digest(value)
        raw.append(value)
    raw.sort(key=lambda row: row["item_id"])
    summary: dict[str, Any] = {
        "phase": base.PHASE,
        "schema_version": "phase1236.behavior_summary.v1",
        "created_at_utc": base.utc_now(),
        "model": model_name,
        "contract_digest": contract["contract_digest"],
        "numeric_revision_digest": revision["revision_digest"],
        "case_count": len(raw),
        "raw_digest": base.digest(raw),
        "runtimes": {
            "content_score": content_runtime,
            "contract_score": contract_runtime,
            "generation": generation_runtime,
        },
        "precision_audit": precision,
        "placement": placement,
        "elapsed_seconds": time.time() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    summary["summary_digest"] = base.digest(summary)
    base.write_jsonl(base.behavior_raw_path(model_name), raw)
    base.write_json(base.behavior_summary_path(model_name), summary)
    print(base.canonical_json({"status": "behavior_complete_revision1", "model": model_name, "rows": len(raw), "summary_digest": summary["summary_digest"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preregister-revision", "run-behavior"), required=True)
    parser.add_argument("--model", choices=base.MODELS)
    args = parser.parse_args()
    if args.stage == "preregister-revision":
        preregister_revision()
    elif args.model is None:
        raise RuntimeError("--model is required for run-behavior")
    else:
        run_behavior(args.model)


if __name__ == "__main__":
    main()
