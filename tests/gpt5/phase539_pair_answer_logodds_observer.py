#!/usr/bin/env python3
"""Explore a frozen two-answer log-odds observer on Phase535 open data."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from model_utils import load_model, release_model  # noqa: E402


MODELS = ("qwen3", "glm4", "deepseek7b")
OPEN_SPLITS = ("discovery", "entity_prediction", "relation_prediction")
PROTOCOL_DIR = ROOT / "tests/gpt5/result/phase535_pair_addressed_binding_protocol"
CONTRACT_PATH = PROTOCOL_DIR / "phase535_frozen_contract.json"
STATIC_PATH = PROTOCOL_DIR / "phase535_static_audit.json"
PROTOCOL_SOURCE = ROOT / "tests/gpt5/phase535_pair_addressed_binding_protocol.py"
OUT_DIR = ROOT / "tests/gpt5/result/phase539_pair_answer_logodds_observer"
CONTINUATIONS = {
    "supported": "\nThe statement is supported.",
    "contradicted": "\nThe statement is contradicted.",
}
Z = 1.96


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def wilson(count: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = count / n
    denominator = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denominator
    radius = Z * math.sqrt((p * (1 - p) + Z * Z / (4 * n)) / n) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def rate(values: list[bool] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=bool)
    n = int(array.size)
    count = int(array.sum())
    lower, upper = wilson(count, n)
    return {"n": n, "count": count, "rate": count / n if n else 0.0, "lcb95": lower, "ucb95": upper}


def verify() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    static = read_json(STATIC_PATH)
    if static["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase535 static audit failed")
    if sha256_file(PROTOCOL_SOURCE) != contract["source_sha256"]:
        raise RuntimeError("Phase535 protocol source drift")
    for split in OPEN_SPLITS:
        spec = contract["split_files"][split]
        if spec["sealed"] or sha256_file(ROOT / spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"Phase535 split drift: {split}")
    return contract


def candidate_token_ids(tokenizer: Any) -> dict[str, list[int]]:
    result = {
        name: tokenizer.encode(text, add_special_tokens=False)
        for name, text in CONTINUATIONS.items()
    }
    if any(not tokens for tokens in result.values()):
        raise RuntimeError("empty candidate continuation")
    return result


def score_sequences(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    batch_size: int,
    model_name: str,
    stage: str,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    candidates = candidate_token_ids(tokenizer)
    scores = np.zeros((len(prompts), 2), dtype=np.float32)
    names = ("supported", "contradicted")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        sequences = []
        metadata = []
        for local_index, prompt in enumerate(batch_prompts):
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            for candidate_index, name in enumerate(names):
                continuation = candidates[name]
                sequences.append(prompt_ids + continuation)
                metadata.append((local_index, candidate_index, len(prompt_ids), len(continuation)))
        width = max(len(sequence) for sequence in sequences)
        input_ids = torch.full((len(sequences), width), int(pad_id), dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        pads = []
        for index, sequence in enumerate(sequences):
            pad = width - len(sequence)
            pads.append(pad)
            input_ids[index, pad:] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[index, pad:] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        for sequence_index, (local_index, candidate_index, prompt_length, continuation_length) in enumerate(metadata):
            first_target = pads[sequence_index] + prompt_length
            token_ids = input_ids[sequence_index, first_target : first_target + continuation_length]
            prediction_positions = torch.arange(
                first_target - 1,
                first_target + continuation_length - 1,
                device=device,
            )
            token_logits = logits[sequence_index, prediction_positions]
            selected = token_logits.gather(1, token_ids.unsqueeze(1)).squeeze(1)
            log_probs = selected - torch.logsumexp(token_logits, dim=1)
            scores[start + local_index, candidate_index] = float(log_probs.mean().item())
        del logits, input_ids, attention_mask
        if start == 0 or start + len(batch_prompts) == len(prompts) or (start // batch_size) % 16 == 15:
            log(f"{model_name} {stage} {min(start + len(batch_prompts), len(prompts))}/{len(prompts)}")
    return scores, candidates


def fold_id(group_id: str, fold_count: int = 4) -> int:
    return int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % fold_count


def fit_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    positive = float(scores[labels].mean())
    negative = float(scores[~labels].mean())
    direction = 1.0 if positive >= negative else -1.0
    threshold = (positive * direction + negative * direction) / 2
    return direction, threshold


def discovery_oof(rows: list[dict[str, Any]], raw_scores: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.asarray([bool(row["truth_value"]) for row in rows], dtype=bool)
    folds = np.asarray([fold_id(row["source_group_id"]) for row in rows], dtype=np.int8)
    predictions = np.zeros(len(rows), dtype=bool)
    fold_ledgers = []
    for fold in range(4):
        test = folds == fold
        train = ~test
        direction, threshold = fit_threshold(raw_scores[train], labels[train])
        predictions[test] = direction * raw_scores[test] > threshold
        fold_ledgers.append({
            "fold": fold,
            "train_row_count": int(train.sum()),
            "test_row_count": int(test.sum()),
            "direction": direction,
            "threshold": threshold,
        })
    direction, threshold = fit_threshold(raw_scores, labels)
    return predictions, {
        "fold_count": 4,
        "fold_ledgers": fold_ledgers,
        "full_discovery_direction": direction,
        "full_discovery_threshold": threshold,
        "positive_raw_score_mean": float(raw_scores[labels].mean()),
        "negative_raw_score_mean": float(raw_scores[~labels].mean()),
    }


def apply_threshold(raw_scores: np.ndarray, ledger: dict[str, Any]) -> np.ndarray:
    return (
        float(ledger["full_discovery_direction"]) * raw_scores
        > float(ledger["full_discovery_threshold"])
    )


def exact_group_rate(rows: list[dict[str, Any]], predictions: np.ndarray, key: str, size: int) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row[key]].append(index)
    labels = np.asarray([bool(row["truth_value"]) for row in rows], dtype=bool)
    return rate([
        len(indices) == size and bool(np.all(predictions[indices] == labels[indices]))
        for indices in groups.values()
    ])


def split_report(
    rows: list[dict[str, Any]],
    raw_scores: np.ndarray,
    predictions: np.ndarray,
    gate: dict[str, Any],
) -> dict[str, Any]:
    labels = np.asarray([bool(row["truth_value"]) for row in rows], dtype=bool)
    correct = predictions == labels
    by_surface = {}
    by_truth = {}
    by_candidate = {}
    for surface in sorted({row["surface"] for row in rows}):
        mask = np.asarray([row["surface"] == surface for row in rows], dtype=bool)
        by_surface[surface] = rate(correct[mask])
    for truth in (False, True):
        by_truth[str(truth).lower()] = rate(correct[labels == truth])
    for candidate in range(4):
        mask = np.asarray([row["candidate_index"] == candidate for row in rows], dtype=bool)
        by_candidate[str(candidate)] = rate(correct[mask])
    overall = rate(correct)
    world_exact = exact_group_rate(rows, predictions, "world_surface_id", 4)
    pair_flip_exact = exact_group_rate(rows, predictions, "pair_flip_id", 2)
    source_group_exact = exact_group_rate(rows, predictions, "source_group_id", 16)
    gate_pass = (
        overall["lcb95"] >= float(gate["overall_lcb95_min"])
        and all(item["lcb95"] >= float(gate["surface_lcb95_min"]) for item in by_surface.values())
        and world_exact["lcb95"] >= float(gate["world_exact_lcb95_min"])
        and pair_flip_exact["lcb95"] >= float(gate["pair_flip_exact_lcb95_min"])
    )
    return {
        "overall": overall,
        "by_surface": by_surface,
        "by_truth": by_truth,
        "by_candidate_index": by_candidate,
        "world_exact": world_exact,
        "pair_flip_exact": pair_flip_exact,
        "source_group_exact": source_group_exact,
        "raw_score": {
            "mean": float(raw_scores.mean()),
            "std": float(raw_scores.std()),
            "positive_mean": float(raw_scores[labels].mean()),
            "negative_mean": float(raw_scores[~labels].mean()),
        },
        "gate_pass": gate_pass,
    }


def run_model(model_name: str, batch_size: int, use_8bit: bool) -> Path:
    contract = verify()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = None
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase539 requires CUDA")
        model, tokenizer, device = load_model(model_name, use_8bit=True if use_8bit else None)
        discovery_rows = read_jsonl(ROOT / contract["split_files"]["discovery"]["path"])
        discovery_candidates, token_ids = score_sequences(
            model,
            tokenizer,
            device,
            [row["natural_prompt"] for row in discovery_rows],
            batch_size,
            model_name,
            "discovery",
        )
        discovery_raw = discovery_candidates[:, 0] - discovery_candidates[:, 1]
        discovery_predictions, ledger = discovery_oof(discovery_rows, discovery_raw)
        ledger_payload = {
            "schema_version": "phase539_pair_answer_logodds_frozen_ledger.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "frozen_after_discovery_before_holdout_scores",
            "model": model_name,
            "continuations": CONTINUATIONS,
            "continuation_token_ids": token_ids,
            "score_definition": "mean conditional token log-probability(supported) minus contradicted",
            "threshold_fit": ledger,
            "discovery_row_count": len(discovery_rows),
            "entity_prediction_scores_read": False,
            "relation_prediction_scores_read": False,
            "sealed_split_read": False,
        }
        ledger_path = OUT_DIR / f"phase539_{model_name}_frozen_discovery_ledger.json"
        ledger_path.write_text(
            json.dumps(ledger_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        all_rows = []
        all_candidates = []
        all_predictions = []
        split_reports = {
            "discovery": split_report(
                discovery_rows,
                discovery_raw,
                discovery_predictions,
                contract["behavior_gate"],
            )
        }
        for row, candidates, prediction in zip(
            discovery_rows,
            discovery_candidates,
            discovery_predictions,
            strict=True,
        ):
            all_rows.append(row)
            all_candidates.append(candidates)
            all_predictions.append(bool(prediction))

        for split in ("entity_prediction", "relation_prediction"):
            rows = read_jsonl(ROOT / contract["split_files"][split]["path"])
            candidates, _ = score_sequences(
                model,
                tokenizer,
                device,
                [row["natural_prompt"] for row in rows],
                batch_size,
                model_name,
                split,
            )
            raw = candidates[:, 0] - candidates[:, 1]
            predictions = apply_threshold(raw, ledger)
            split_reports[split] = split_report(rows, raw, predictions, contract["behavior_gate"])
            for row, local_candidates, prediction in zip(rows, candidates, predictions, strict=True):
                all_rows.append(row)
                all_candidates.append(local_candidates)
                all_predictions.append(bool(prediction))

        score_rows = []
        for row, candidates, prediction in zip(all_rows, all_candidates, all_predictions, strict=True):
            score_rows.append({
                "sample_id": row["sample_id"],
                "source_group_id": row["source_group_id"],
                "world_surface_id": row["world_surface_id"],
                "pair_flip_id": row["pair_flip_id"],
                "split": row["split"],
                "surface": row["surface"],
                "world_id": row["world_id"],
                "candidate_index": row["candidate_index"],
                "candidate_slot": row["candidate_slot"],
                "truth_value": row["truth_value"],
                "supported_mean_logprob": float(candidates[0]),
                "contradicted_mean_logprob": float(candidates[1]),
                "raw_logodds_score": float(candidates[0] - candidates[1]),
                "predicted_truth": prediction,
                "correct": prediction == bool(row["truth_value"]),
            })
        rows_path = OUT_DIR / f"phase539_{model_name}_score_rows.jsonl"
        write_jsonl(rows_path, score_rows)
        all_open_pass = all(report["gate_pass"] for report in split_reports.values())
        summary_path = OUT_DIR / f"phase539_{model_name}_summary.json"
        payload = {
            "schema_version": "phase539_pair_answer_logodds_observer.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete_exploratory_requires_fresh_confirmation" if all_open_pass else "complete_exploratory_gate_fail",
            "model": model_name,
            "row_count": len(score_rows),
            "split_reports": split_reports,
            "exploratory_all_open_pass": all_open_pass,
            "fresh_confirmation_required": all_open_pass,
            "physical_authorized": False,
            "frozen_discovery_ledger_path": str(ledger_path.relative_to(ROOT)),
            "score_rows_path": str(rows_path.relative_to(ROOT)),
            "score_rows_sha256": sha256_file(rows_path),
            "cuda_used": True,
            "sealed_split_read": False,
            "runtime_seconds": time.monotonic() - started,
        }
        summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(summary_path)
        return summary_path
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def aggregate() -> Path:
    verify()
    summaries = {}
    for model in MODELS:
        path = OUT_DIR / f"phase539_{model}_summary.json"
        if not path.exists():
            raise RuntimeError(f"missing Phase539 summary: {model}")
        summaries[model] = read_json(path)
    candidates = [model for model in MODELS if summaries[model]["fresh_confirmation_required"]]
    output = OUT_DIR / "phase539_fresh_confirmation_authorization.json"
    payload = {
        "schema_version": "phase539_pair_answer_logodds_fresh_confirmation_authorization.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_after_required_model_order",
        "models_in_required_order": list(MODELS),
        "fresh_confirmation_required_models": candidates,
        "physical_authorized_models": [],
        "sealed_split_read": False,
        "evidence_boundary": "Phase539 is exploratory interface calibration; even a passing model requires newly generated confirmation data.",
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return output


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
