#!/usr/bin/env python3
"""Execute the sealed Phase1203 object-attribute behavior protocol.

This module has two deliberately separate entry points. ``preflight`` freezes
the Phase1204 runner before any model output exists. ``run`` consumes only the
exact token-ID manifest for one model and writes candidate scores; it never
retokenizes prompts, generates text, or requests hidden states or attentions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1203_object_attribute_behavior_protocol as phase1203
from model_utils import MODEL_CONFIGS
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


PHASE = 1204
SCRIPT = Path(__file__).resolve()
PREEXECUTION_AUDIT_SCRIPT = TEST_ROOT / "phase1204_object_attribute_behavior_preexecution_audit.py"
SEQUENTIAL_SCRIPT = TEST_ROOT / "phase1204_run_sequential.py"
FINALIZE_SCRIPT = TEST_ROOT / "phase1204_object_attribute_behavior_finalize.py"
RESULT_AUDIT_SCRIPT = TEST_ROOT / "phase1204_object_attribute_behavior_result_audit.py"

UPSTREAM_ROOT = TEST_ROOT / "result/phase1203_object_attribute_behavior_protocol"
UPSTREAM_FINAL = UPSTREAM_ROOT / "analysis/final.json"
UPSTREAM_PROTOCOL = UPSTREAM_ROOT / "protocol/behavior_protocol.json"
UPSTREAM_AUDIT = UPSTREAM_ROOT / "audit/independent_protocol_audit.json"
UPSTREAM_MANIFEST_DIR = UPSTREAM_ROOT / "protocol/model_manifests"

OUT_ROOT = TEST_ROOT / "result/phase1204_object_attribute_behavior_execution"
CONTRACT_PATH = OUT_ROOT / "protocol/execution_contract.json"
PREEXECUTION_AUDIT_PATH = OUT_ROOT / "audit/preexecution_audit.json"

EXPECTED_PHASE1203_FINAL_DIGEST = "ef1c8825f190682f165f4b7080130cf043fabd1cd6a6be30a2cb0199eca2f198"
EXPECTED_PHASE1203_PROTOCOL_DIGEST = "62ff69b41c7de1407beb9b26ccdaf9e4eed8ea342959356e575a1dd1080434a6"
EXPECTED_PHASE1203_AUDIT_DIGEST = "a4cc6e3668c7a5dccaf45e2bf22293cb72e12747bd038c04edf710e2e24dbc3d"

MODEL_ORDER = tuple(phase1203.MODEL_ORDER)
BATCH_SIZE = dict(phase1203.MODEL_BATCH_SIZE)
EXPECTED_CASES = phase1203.EXPECTED_CASES_PER_MODEL
TIE_TOLERANCE = phase1203.TIE_TOLERANCE


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} in {path} is not an object")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    if pending.exists():
        pending.unlink()
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(pending, path)


def source_paths() -> dict[str, Path]:
    return {
        "execution": SCRIPT,
        "preexecution_audit": PREEXECUTION_AUDIT_SCRIPT,
        "sequential_runner": SEQUENTIAL_SCRIPT,
        "finalize": FINALIZE_SCRIPT,
        "result_audit": RESULT_AUDIT_SCRIPT,
    }


def source_hashes() -> dict[str, str]:
    return {name: file_sha256(path) for name, path in source_paths().items()}


def behavior_dir(model_name: str) -> Path:
    return OUT_ROOT / "behavior" / model_name


def raw_path(model_name: str) -> Path:
    return behavior_dir(model_name) / "raw_scores.jsonl"


def summary_path(model_name: str) -> Path:
    return behavior_dir(model_name) / "run_summary.json"


def manifest_path(model_name: str) -> Path:
    return UPSTREAM_MANIFEST_DIR / f"{model_name}.jsonl"


def validate_embedded_digest(payload: dict[str, Any], field: str) -> None:
    expected = payload[field]
    candidate = {key: value for key, value in payload.items() if key != field}
    if digest(candidate) != expected:
        raise RuntimeError(f"embedded digest mismatch for {field}")


def verify_upstream() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    final = read_json(UPSTREAM_FINAL)
    protocol = phase1203.verify_protocol()
    audit = read_json(UPSTREAM_AUDIT)
    validate_embedded_digest(final, "final_digest")
    if final["final_digest"] != EXPECTED_PHASE1203_FINAL_DIGEST:
        raise RuntimeError("Phase1203 final digest mismatch")
    if protocol["protocol_digest"] != EXPECTED_PHASE1203_PROTOCOL_DIGEST:
        raise RuntimeError("Phase1203 protocol digest mismatch")
    if audit.get("audit_digest") != EXPECTED_PHASE1203_AUDIT_DIGEST or not audit.get("gate_pass"):
        raise RuntimeError("Phase1203 independent audit mismatch")
    if not final["authorized_next"]["phase1204_sequential_fp16_behavior_execution"]:
        raise RuntimeError("Phase1203 did not authorize Phase1204")
    if final["authorized_next"]["hidden_state_scan"]:
        raise RuntimeError("Phase1203 unexpectedly authorized hidden-state access")
    return final, protocol, audit


def model_artifact_fingerprint(model_name: str) -> dict[str, Any]:
    root = Path(MODEL_CONFIGS[model_name]["path"])
    files = sorted(path for path in root.iterdir() if path.is_file())
    small_hashes = {
        path.name: file_sha256(path)
        for path in files
        if path.name in {
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer_config.json",
        }
    }
    weight_files = [path for path in files if path.suffix in {".safetensors", ".bin"}]
    return {
        "model_path": str(root),
        "small_metadata_hashes": small_hashes,
        "weight_file_names_and_sizes": [[path.name, path.stat().st_size] for path in weight_files],
        "total_weight_bytes": sum(path.stat().st_size for path in weight_files),
    }


def build_contract() -> dict[str, Any]:
    final, protocol, audit = verify_upstream()
    contract: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1204.object_attribute.execution_contract.v1",
        "created_at": utc_now(),
        "purpose": "execute the exact Phase1203 manifests without outcome-dependent adaptation",
        "source_hashes": source_hashes(),
        "upstream": {
            "phase1203_final_digest": final["final_digest"],
            "phase1203_protocol_digest": protocol["protocol_digest"],
            "phase1203_audit_digest": audit["audit_digest"],
            "manifest_digests": final["model_manifest_digests"],
        },
        "execution": {
            "model_order": list(MODEL_ORDER),
            "fixed_batch_size": BATCH_SIZE,
            "precision": "FP16",
            "normalization": "final-position FP16 vocab logits cast to FP32 then log_softmax",
            "quantization": "none",
            "cuda_required": True,
            "one_model_per_process": True,
            "release_before_next_model": True,
            "exact_manifest_input_ids_only": True,
            "runtime_retokenization": False,
            "exact_length_bucketing": True,
            "adaptive_oom_fallback": False,
            "generation": False,
            "output_hidden_states": False,
            "output_attentions": False,
            "tie_tolerance": TIE_TOLERANCE,
        },
        "model_artifacts": {model: model_artifact_fingerprint(model) for model in MODEL_ORDER},
        "raw_case_schema": [
            "model",
            "execution_index",
            "item_id",
            "all_vocab_logits_finite",
            "candidate_scores",
            "prediction",
            "gold_candidate",
            "correct",
            "gold_margin",
            "runtime_batch_size",
            "input_length",
        ],
        "forbidden": list(protocol["forbidden_after_scoring"]),
        "claim_boundary": {
            "behavior_only": True,
            "hidden_state_evidence": False,
            "causal_evidence": False,
            "natural_use_evidence": False,
            "mechanism_claim": False,
        },
    }
    contract["contract_digest"] = digest(contract)
    return contract


def preflight() -> None:
    if CONTRACT_PATH.exists() or PREEXECUTION_AUDIT_PATH.exists() or (OUT_ROOT / "behavior").exists():
        raise RuntimeError("Phase1204 preflight or behavior output already exists")
    contract = build_contract()
    write_json(CONTRACT_PATH, contract)
    print(canonical_json({"status": "execution_contract_frozen", "contract_digest": contract["contract_digest"]}))


def verify_contract() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    validate_embedded_digest(contract, "contract_digest")
    if contract["source_hashes"] != source_hashes():
        raise RuntimeError("Phase1204 source changed after preflight")
    final, protocol, audit = verify_upstream()
    expected_upstream = {
        "phase1203_final_digest": final["final_digest"],
        "phase1203_protocol_digest": protocol["protocol_digest"],
        "phase1203_audit_digest": audit["audit_digest"],
        "manifest_digests": final["model_manifest_digests"],
    }
    if contract["upstream"] != expected_upstream:
        raise RuntimeError("Phase1204 upstream link changed")
    return contract


def verify_preexecution_audit(contract: dict[str, Any]) -> dict[str, Any]:
    audit = read_json(PREEXECUTION_AUDIT_PATH)
    if not audit.get("gate_pass"):
        raise RuntimeError("Phase1204 preexecution audit failed")
    if audit.get("contract_digest") != contract["contract_digest"]:
        raise RuntimeError("Phase1204 preexecution audit contract mismatch")
    validate_embedded_digest(audit, "audit_digest")
    return audit


def load_manifest(model_name: str, contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_jsonl(manifest_path(model_name))
    if len(rows) != EXPECTED_CASES:
        raise RuntimeError(f"{model_name} manifest case count mismatch")
    if phase1203.digest(rows) != contract["upstream"]["manifest_digests"][model_name]:
        raise RuntimeError(f"{model_name} manifest digest mismatch")
    if [row["execution_index"] for row in rows] != list(range(EXPECTED_CASES)):
        raise RuntimeError(f"{model_name} execution index mismatch")
    if any(row["model"] != model_name for row in rows):
        raise RuntimeError(f"{model_name} manifest model mismatch")
    if any(len(ids) != 1 for row in rows for ids in row["candidate_token_ids"].values()):
        raise RuntimeError(f"{model_name} manifest contains a multi-token candidate")
    return rows


def enforce_model_order(model_name: str) -> None:
    index = MODEL_ORDER.index(model_name)
    for predecessor in MODEL_ORDER[:index]:
        if not summary_path(predecessor).exists():
            raise RuntimeError(f"{model_name} cannot run before {predecessor}")
    for successor in MODEL_ORDER[index + 1 :]:
        if summary_path(successor).exists():
            raise RuntimeError(f"out-of-order output already exists for {successor}")


def finite_json_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def run_model(model_name: str) -> None:
    contract = verify_contract()
    preaudit = verify_preexecution_audit(contract)
    enforce_model_order(model_name)
    if raw_path(model_name).exists() or summary_path(model_name).exists():
        raise RuntimeError(f"Phase1204 output already exists for {model_name}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen Phase1203 protocol")

    rows = load_manifest(model_name, contract)
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_length[int(row["input_length"])].append(row)

    started = time.time()
    model = None
    details: list[dict[str, Any]] = []
    try:
        model, _tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
            or set(precision["parameter_dtypes"]) != {"float16"}
        ):
            raise RuntimeError("FP16/no-quantization audit failed")

        with torch.inference_mode():
            for input_length in sorted(by_length):
                bucket = by_length[input_length]
                for start in range(0, len(bucket), BATCH_SIZE[model_name]):
                    batch = bucket[start : start + BATCH_SIZE[model_name]]
                    input_ids = torch.tensor(
                        [row["input_ids"] for row in batch],
                        dtype=torch.long,
                        device=device,
                    )
                    attention_mask = torch.ones_like(input_ids)
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                        output_hidden_states=False,
                        output_attentions=False,
                    )
                    raw_logits = output.logits[:, -1, :]
                    vocab_finite = torch.isfinite(raw_logits).all(dim=-1)
                    log_probs = torch.log_softmax(raw_logits.float(), dim=-1)

                    for slot, row in enumerate(batch):
                        labels = list(row["candidate_labels"])
                        scores = {
                            label: float(log_probs[slot, row["candidate_token_ids"][label][0]].item())
                            for label in labels
                        }
                        all_vocab_finite = bool(vocab_finite[slot].item())
                        score_finite = all(math.isfinite(value) for value in scores.values())
                        finite = all_vocab_finite and score_finite
                        sorted_scores = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
                        top_gap = sorted_scores[0][1] - sorted_scores[1][1] if score_finite else math.nan
                        tie = finite and top_gap <= TIE_TOLERANCE
                        if not finite:
                            prediction = "NONFINITE"
                        elif tie:
                            prediction = "UNRESOLVED_TIE"
                        else:
                            prediction = sorted_scores[0][0]
                        gold = str(row["gold_candidate"])
                        other_best = max(value for label, value in scores.items() if label != gold)
                        gold_margin = scores[gold] - other_best if score_finite else math.nan
                        details.append(
                            {
                                "schema_version": "phase1204.object_attribute.raw_score.v1",
                                "model": model_name,
                                "execution_index": int(row["execution_index"]),
                                "item_id": row["item_id"],
                                "all_vocab_logits_finite": all_vocab_finite,
                                "candidate_scores": {
                                    label: finite_json_float(value) for label, value in scores.items()
                                },
                                "prediction": prediction,
                                "gold_candidate": gold,
                                "correct": bool(finite and not tie and prediction == gold),
                                "gold_margin": finite_json_float(gold_margin),
                                "top_two_gap": finite_json_float(top_gap),
                                "unresolved_tie": bool(tie),
                                "runtime_batch_size": len(batch),
                                "frozen_batch_size": BATCH_SIZE[model_name],
                                "input_length": int(row["input_length"]),
                            }
                        )
                    del output, raw_logits, vocab_finite, log_probs, input_ids, attention_mask

                print(
                    canonical_json(
                        {
                            "phase": PHASE,
                            "model": model_name,
                            "completed_input_length": input_length,
                            "completed_cases": len(details),
                        }
                    ),
                    flush=True,
                )

        details.sort(key=lambda row: row["execution_index"])
        if len(details) != EXPECTED_CASES:
            raise RuntimeError(f"{model_name} output count mismatch")
        run_summary: dict[str, Any] = {
            "phase": PHASE,
            "schema_version": "phase1204.object_attribute.run_summary.v1",
            "created_at": utc_now(),
            "model": model_name,
            "contract_digest": contract["contract_digest"],
            "preexecution_audit_digest": preaudit["audit_digest"],
            "manifest_digest": contract["upstream"]["manifest_digests"][model_name],
            "raw_digest": digest(details),
            "case_count": len(details),
            "all_vocab_finite_rate": sum(row["all_vocab_logits_finite"] for row in details) / len(details),
            "unresolved_tie_rate": sum(row["unresolved_tie"] for row in details) / len(details),
            "preliminary_accuracy": sum(row["correct"] for row in details) / len(details),
            "fixed_batch_size": BATCH_SIZE[model_name],
            "precision_audit": precision,
            "placement": placement,
            "model_artifact_fingerprint": contract["model_artifacts"][model_name],
            "runtime": {
                "elapsed_seconds": time.time() - started,
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            "claim_boundary": contract["claim_boundary"],
        }
        run_summary["summary_digest"] = digest(run_summary)
        write_jsonl_atomic(raw_path(model_name), details)
        write_json(summary_path(model_name), run_summary)
        print(
            json.dumps(
                {
                    "phase": PHASE,
                    "model": model_name,
                    "case_count": run_summary["case_count"],
                    "all_vocab_finite_rate": run_summary["all_vocab_finite_rate"],
                    "unresolved_tie_rate": run_summary["unresolved_tie_rate"],
                    "preliminary_accuracy": run_summary["preliminary_accuracy"],
                    "elapsed_seconds": run_summary["runtime"]["elapsed_seconds"],
                    "summary_digest": run_summary["summary_digest"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    finally:
        if model is not None:
            release_fp16(model)
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("model", choices=MODEL_ORDER)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
    else:
        run_model(args.model)


if __name__ == "__main__":
    main()
