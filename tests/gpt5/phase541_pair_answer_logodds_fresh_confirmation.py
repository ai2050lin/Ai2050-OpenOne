#!/usr/bin/env python3
"""Confirm Phase539 pair-answer log-odds observers without refitting."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase539_pair_answer_logodds_observer as observer  # noqa: E402
from model_utils import load_model, release_model  # noqa: E402


MODELS = ("qwen3", "glm4", "deepseek7b")
OPEN_SPLITS = ("fresh_vocabulary_confirmation", "fresh_relation_confirmation")
PROTOCOL_DIR = ROOT / "tests/gpt5/result/phase540_pair_answer_logodds_fresh_protocol"
CONTRACT_PATH = PROTOCOL_DIR / "phase540_frozen_contract.json"
STATIC_PATH = PROTOCOL_DIR / "phase540_static_audit.json"
PROTOCOL_SOURCE = ROOT / "tests/gpt5/phase540_pair_answer_logodds_fresh_protocol.py"
OUT_DIR = ROOT / "tests/gpt5/result/phase541_pair_answer_logodds_fresh_confirmation"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def verify() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    static = read_json(STATIC_PATH)
    if static["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase540 static audit failed")
    if observer.sha256_file(PROTOCOL_SOURCE) != contract["source_sha256"]:
        raise RuntimeError("Phase540 protocol source drift")
    if observer.sha256_file(CONTRACT_PATH) != static["contract_sha256"]:
        raise RuntimeError("Phase540 contract drift")
    phase539_auth = ROOT / contract["phase539_authorization_path"]
    if observer.sha256_file(phase539_auth) != contract["phase539_authorization_sha256"]:
        raise RuntimeError("Phase539 authorization drift")
    for split in OPEN_SPLITS:
        spec = contract["split_files"][split]
        if spec["sealed"] or observer.sha256_file(ROOT / spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"Phase540 open split drift: {split}")
    return contract


def excluded_summary(model_name: str) -> Path:
    path = OUT_DIR / f"phase541_{model_name}_summary.json"
    payload = {
        "schema_version": "phase541_pair_answer_logodds_fresh_confirmation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "not_phase539_candidate",
        "model": model_name,
        "cuda_used": False,
        "model_weights_loaded": False,
        "open_confirmation_splits_read": False,
        "sealed_split_read": False,
        "physical_collection_authorized": False,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return path


def run_model(model_name: str, batch_size: int, use_8bit: bool) -> Path:
    contract = verify()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if model_name not in contract["fresh_confirmation_candidate_models"]:
        return excluded_summary(model_name)

    ledger_spec = contract["frozen_observer_ledgers"][model_name]
    ledger_path = ROOT / ledger_spec["path"]
    if observer.sha256_file(ledger_path) != ledger_spec["sha256"]:
        raise RuntimeError(f"Phase539 {model_name} observer ledger drift")
    ledger = read_json(ledger_path)
    model = None
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase541 requires CUDA for qualified models")
        model, tokenizer, device = load_model(model_name, use_8bit=True if use_8bit else None)
        if observer.candidate_token_ids(tokenizer) != ledger["continuation_token_ids"]:
            raise RuntimeError("candidate continuation tokenization drift")
        split_reports: dict[str, Any] = {}
        output_rows: list[dict[str, Any]] = []
        for split in OPEN_SPLITS:
            rows = read_jsonl(ROOT / contract["split_files"][split]["path"])
            candidate_scores, token_ids = observer.score_sequences(
                model,
                tokenizer,
                device,
                [row["natural_prompt"] for row in rows],
                batch_size,
                model_name,
                split,
            )
            if token_ids != ledger["continuation_token_ids"]:
                raise RuntimeError("candidate token IDs changed during confirmation")
            raw = candidate_scores[:, 0] - candidate_scores[:, 1]
            predictions = observer.apply_threshold(raw, ledger["threshold_fit"])
            split_reports[split] = observer.split_report(rows, raw, predictions, contract["behavior_gate"])
            for row, scores, prediction in zip(rows, candidate_scores, predictions, strict=True):
                output_rows.append({
                    "sample_id": row["sample_id"],
                    "source_group_id": row["source_group_id"],
                    "world_surface_id": row["world_surface_id"],
                    "pair_flip_id": row["pair_flip_id"],
                    "split": split,
                    "surface": row["surface"],
                    "world_id": row["world_id"],
                    "candidate_index": row["candidate_index"],
                    "candidate_slot": row["candidate_slot"],
                    "truth_value": bool(row["truth_value"]),
                    "supported_mean_logprob": float(scores[0]),
                    "contradicted_mean_logprob": float(scores[1]),
                    "raw_logodds_score": float(scores[0] - scores[1]),
                    "predicted_truth": bool(prediction),
                    "correct": bool(prediction) == bool(row["truth_value"]),
                })
        all_pass = all(report["gate_pass"] for report in split_reports.values())
        rows_path = OUT_DIR / f"phase541_{model_name}_score_rows.jsonl"
        write_jsonl(rows_path, output_rows)
        summary_path = OUT_DIR / f"phase541_{model_name}_summary.json"
        payload = {
            "schema_version": "phase541_pair_answer_logodds_fresh_confirmation.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "fresh_confirmed" if all_pass else "fresh_confirmation_failed",
            "model": model_name,
            "split_reports": split_reports,
            "all_open_confirmation_pass": all_pass,
            "threshold_refit": False,
            "continuation_refit": False,
            "physical_collection_authorized": all_pass,
            "frozen_observer_ledger_path": str(ledger_path.relative_to(ROOT)),
            "frozen_observer_ledger_sha256": ledger_spec["sha256"],
            "score_rows_path": str(rows_path.relative_to(ROOT)),
            "score_rows_sha256": observer.sha256_file(rows_path),
            "row_count": len(output_rows),
            "cuda_used": True,
            "model_weights_loaded": True,
            "sealed_split_read": False,
            "runtime_seconds": time.monotonic() - started,
            "evidence_boundary": "Fresh interface observer confirmation only; no hidden-state, component, neuron, compute-edge, or causal claim.",
        }
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(summary_path)
        return summary_path
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def aggregate() -> Path:
    contract = verify()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for model in MODELS:
        path = OUT_DIR / f"phase541_{model}_summary.json"
        if not path.exists():
            raise RuntimeError(f"missing Phase541 summary: {model}")
        summaries[model] = read_json(path)
    authorized = [model for model in MODELS if summaries[model].get("physical_collection_authorized", False)]
    path = OUT_DIR / "phase541_physical_collection_authorization.json"
    payload = {
        "schema_version": "phase541_pair_answer_logodds_physical_collection_authorization.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_after_required_model_order",
        "models_in_required_order": list(MODELS),
        "phase539_candidates": contract["fresh_confirmation_candidate_models"],
        "fresh_confirmed_models": authorized,
        "physical_collection_authorized_models": authorized,
        "contract_path": str(CONTRACT_PATH.relative_to(ROOT)),
        "contract_sha256": observer.sha256_file(CONTRACT_PATH),
        "sealed_split_read": False,
        "causal_authorized": False,
        "evidence_boundary": "Authorization covers observational hidden-state collection only.",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS + ("aggregate",))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()
    if args.model == "aggregate":
        aggregate()
    else:
        run_model(args.model, args.batch_size, args.use_8bit)


if __name__ == "__main__":
    main()
