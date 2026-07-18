#!/usr/bin/env python3
"""Run one Phase509 behavior stage for one CUDA model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

from model_utils import load_model, release_model  # noqa: E402


PROTOCOL_DIR = ROOT / "tests/gpt5/result/phase509_dual_contract_protocol"
CONTRACT_PATH = PROTOCOL_DIR / "phase509_frozen_contract.json"
AUDIT_PATH = PROTOCOL_DIR / "phase509_static_audit.json"
PROTOCOL_SOURCE = ROOT / "tests/gpt5/phase509_dual_contract_protocol.py"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = ("identity", "native_plain_candidate")
LABEL_SYSTEMS = ("mapped_ab", "mapped_01")
Z = 1.96

STAGES = {
    "calibration": {
        "phase": 510,
        "contracts": ("relation", "binding"),
        "split": "calibration",
        "authorization": None,
        "out": ROOT / "tests/gpt5/result/phase510_relation_binding_calibration",
    },
    "confirmation": {
        "phase": 512,
        "contracts": ("relation", "binding"),
        "split": "confirmation",
        "authorization": ROOT / "tests/gpt5/result/phase511_calibration_authorization/phase511_calibration_authorization.json",
        "out": ROOT / "tests/gpt5/result/phase512_relation_binding_confirmation",
    },
    "joint": {
        "phase": 514,
        "contracts": ("joint",),
        "split": "joint_confirmation",
        "authorization": ROOT / "tests/gpt5/result/phase513_confirmation_authorization/phase513_confirmation_authorization.json",
        "out": ROOT / "tests/gpt5/result/phase514_joint_confirmation",
    },
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def rate(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    n = len(rows)
    k = sum(bool(row[field]) for row in rows)
    lcb, ucb = wilson(k, n)
    return {"n": n, "count": k, "rate": k / n if n else 0.0, "lcb95": lcb, "ucb95": ucb}


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0


def split_path(split: str, contract: str) -> Path:
    return PROTOCOL_DIR / f"phase509_{split}_{contract}.jsonl"


def allowed_contracts(stage: str, model: str) -> set[str]:
    if stage == "calibration":
        return {"relation", "binding"}
    authorization = read_json(STAGES[stage]["authorization"])
    if stage == "confirmation":
        allowed = set()
        if model in authorization["relation_models"]:
            allowed.add("relation")
        if model in authorization["binding_models"]:
            allowed.add("binding")
        return allowed
    return {"joint"} if model in authorization["joint_models"] else set()


def flatten(samples: list[dict[str, Any]], contract: str, split: str) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        if sample["split"] != split or sample["sealed"]:
            raise RuntimeError(f"invalid {split} sample")
        for variant in sample["variants"]:
            row = {
                key: value
                for key, value in sample.items()
                if key not in {"variants", "rules", "facts", "target_facts", "distractors", "state_cue", "mapping_line", "claim"}
            }
            row.update({
                "contract": contract,
                "surface": variant["surface"],
                "prompt": variant["prompt"],
                "true_candidate": variant["true_candidate"],
                "false_candidate": variant["false_candidate"],
            })
            rows.append(row)
    return rows


def single_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"candidate {text!r} is not one token: {ids}")
    return int(ids[0])


def score_rows(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    batch_size: int,
) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["true_candidate"], row["false_candidate"])].append(row)
    tokenizer.padding_side = "left"
    completed = 0
    for (true_candidate, false_candidate), group in sorted(groups.items()):
        true_id = single_token_id(tokenizer, true_candidate)
        false_id = single_token_id(tokenizer, false_candidate)
        for start in range(0, len(group), batch_size):
            batch = group[start:start + batch_size]
            encoded = tokenizer(
                [row["prompt"] for row in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded, use_cache=False, return_dict=True).logits[:, -1, :].float()
            true_scores = logits[:, true_id]
            false_scores = logits[:, false_id]
            log_normalizer = torch.logsumexp(logits, dim=-1)
            true_probs = torch.exp(true_scores - log_normalizer)
            false_probs = torch.exp(false_scores - log_normalizer)
            global_ids = logits.argmax(dim=-1)
            for index, row in enumerate(batch):
                prediction = bool(true_scores[index] > false_scores[index])
                expected_id = true_id if row["truth_value"] else false_id
                row.update({
                    "semantic_margin_true_minus_false": float(true_scores[index] - false_scores[index]),
                    "semantic_prediction": prediction,
                    "candidate_correct": prediction == row["truth_value"],
                    "free_event_correct": int(global_ids[index]) == expected_id,
                    "global_argmax_token_id": int(global_ids[index]),
                    "candidate_probability_mass": float(true_probs[index] + false_probs[index]),
                    "non_candidate_probability_mass": float(1.0 - true_probs[index] - false_probs[index]),
                })
            completed += len(batch)
            if completed == len(rows) or completed % 512 < len(batch):
                log(f"candidate scoring {completed}/{len(rows)}")


def surface_metrics(rows: list[dict[str, Any]], gate: dict[str, Any], pair_key: str | None) -> dict[str, Any]:
    by_surface = {
        surface: rate([row for row in rows if row["surface"] == surface], "candidate_correct")
        for surface in SURFACES
    }
    sample_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_groups[row["sample_id"]].append(row)
    intersections = []
    for key, items in sample_groups.items():
        if len(items) != 2 or {item["surface"] for item in items} != set(SURFACES):
            raise RuntimeError(f"incomplete surface group {key}")
        intersections.append({"candidate_correct": all(item["candidate_correct"] for item in items)})
    intersection = rate(intersections, "candidate_correct")
    payload: dict[str, Any] = {
        "by_surface": by_surface,
        "surface_intersection": intersection,
        "free_event": rate(rows, "free_event_correct"),
        "mean_candidate_probability_mass": mean(rows, "candidate_probability_mass"),
        "mean_non_candidate_probability_mass": mean(rows, "non_candidate_probability_mass"),
    }
    if pair_key is not None:
        pair_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            extra = tuple(row.get(name) for name in ("label_system", "mapping_flip")) if row["contract"] == "joint" else ()
            pair_groups[(row[pair_key], *extra)].append(row)
        pairs = []
        for key, items in pair_groups.items():
            if (
                len(items) != 4
                or {item["truth_value"] for item in items} != {False, True}
                or {item["surface"] for item in items} != set(SURFACES)
            ):
                raise RuntimeError(f"incomplete paired world {key}")
            pairs.append({"candidate_correct": all(item["candidate_correct"] for item in items)})
        payload["paired_world"] = rate(pairs, "candidate_correct")
    return payload


def relation_summary(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    gate = contract["gates"]["R"]
    metrics = surface_metrics(rows, gate, "source_pair_id")
    by_relation = {
        verb: rate([row for row in rows if row["relation_verb"] == verb], "candidate_correct")
        for verb in sorted({row["relation_verb"] for row in rows})
    }
    passed = (
        metrics["by_surface"]["identity"]["lcb95"] >= gate["identity_lcb95_min"]
        and metrics["by_surface"]["native_plain_candidate"]["lcb95"] >= gate["native_plain_lcb95_min"]
        and metrics["surface_intersection"]["lcb95"] >= gate["surface_intersection_lcb95_min"]
        and metrics["paired_world"]["lcb95"] >= gate["paired_world_lcb95_min"]
    )
    return {**metrics, "by_relation_verb": by_relation, "gate_pass": passed}


def reversal_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["mapping_reversal_id"], row["surface"])].append(row)
    records = []
    for key, items in groups.items():
        if len(items) != 2 or {item["mapping_flip"] for item in items} != {False, True}:
            raise RuntimeError(f"incomplete mapping reversal {key}")
        records.append({"candidate_correct": all(item["candidate_correct"] for item in items)})
    return rate(records, "candidate_correct")


def binding_summary(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    gate = contract["gates"]["B"]
    by_label = {}
    for label_system in LABEL_SYSTEMS:
        selected = [row for row in rows if row["label_system"] == label_system]
        metrics = surface_metrics(selected, gate, None)
        metrics["by_mapping_flip"] = {
            str(flip).lower(): rate(
                [row for row in selected if row["mapping_flip"] is flip],
                "candidate_correct",
            )
            for flip in (False, True)
        }
        metrics["mapping_reversal"] = reversal_rate(selected)
        metrics["gate_pass"] = (
            all(item["lcb95"] >= gate["label_system_lcb95_min"] for item in metrics["by_surface"].values())
            and metrics["surface_intersection"]["lcb95"] >= gate["surface_intersection_lcb95_min"]
            and all(item["lcb95"] >= gate["mapping_flip_lcb95_min"] for item in metrics["by_mapping_flip"].values())
            and metrics["mapping_reversal"]["lcb95"] >= gate["mapping_reversal_lcb95_min"]
        )
        by_label[label_system] = metrics
    free_gate = contract["gates"]["S"]
    free_event = rate(rows, "free_event_correct")
    non_candidate_mass = mean(rows, "non_candidate_probability_mass")
    return {
        "by_label_system": by_label,
        "gate_pass": all(item["gate_pass"] for item in by_label.values()),
        "free_output_event": {
            "accuracy": free_event,
            "mean_non_candidate_probability_mass": non_candidate_mass,
            "gate_pass": (
                free_event["lcb95"] >= free_gate["free_event_lcb95_min"]
                and non_candidate_mass <= free_gate["mean_non_candidate_mass_max"]
            ),
        },
    }


def joint_summary(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    gate = contract["gates"]["J"]
    by_label = {}
    for label_system in LABEL_SYSTEMS:
        selected = [row for row in rows if row["label_system"] == label_system]
        metrics = surface_metrics(selected, gate, "source_pair_id")
        metrics["mapping_reversal"] = reversal_rate(selected)
        metrics["gate_pass"] = (
            all(item["lcb95"] >= gate["label_system_lcb95_min"] for item in metrics["by_surface"].values())
            and metrics["surface_intersection"]["lcb95"] >= gate["surface_intersection_lcb95_min"]
            and metrics["paired_world"]["lcb95"] >= gate["paired_world_lcb95_min"]
            and metrics["mapping_reversal"]["lcb95"] >= gate["mapping_reversal_lcb95_min"]
        )
        by_label[label_system] = metrics
    return {
        "by_label_system": by_label,
        "gate_pass": all(item["gate_pass"] for item in by_label.values()),
        "free_output_event": {
            "accuracy": rate(rows, "free_event_correct"),
            "mean_non_candidate_probability_mass": mean(rows, "non_candidate_probability_mass"),
        },
    }


def verify(stage: str) -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    audit = read_json(AUDIT_PATH)
    if audit["status"] != "static_pass_no_model_run":
        raise RuntimeError("Phase509 static audit failed")
    if sha256_file(PROTOCOL_SOURCE) != contract["source_sha256"]:
        raise RuntimeError("Phase509 protocol source changed after freeze")
    split = STAGES[stage]["split"]
    for contract_name in STAGES[stage]["contracts"]:
        key = f"{split}_{contract_name}"
        path = split_path(split, contract_name)
        if sha256_file(path) != contract["split_files"][key]["sha256"]:
            raise RuntimeError(f"Phase509 {key} hash drift")
    auth_path = STAGES[stage]["authorization"]
    if auth_path is not None and not auth_path.exists():
        raise RuntimeError(f"missing authorization {auth_path}")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--reanalyze-only", action="store_true")
    args = parser.parse_args()
    frozen = verify(args.stage)
    allowed = allowed_contracts(args.stage, args.model)
    split = STAGES[args.stage]["split"]
    phase = STAGES[args.stage]["phase"]
    out_dir = STAGES[args.stage]["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / f"phase{phase}_{args.model}_rows.jsonl"
    summary_path = out_dir / f"phase{phase}_{args.model}_summary.json"
    previous_summary = read_json(summary_path) if summary_path.exists() else None
    if args.reanalyze_only:
        if not rows_path.exists() or previous_summary is None:
            raise RuntimeError("reanalyze-only requires existing rows and summary")
        rows = read_jsonl(rows_path)
    else:
        rows = []
        for contract_name in STAGES[args.stage]["contracts"]:
            if contract_name not in allowed:
                continue
            rows.extend(flatten(read_jsonl(split_path(split, contract_name)), contract_name, split))

    model = None
    loaded = False
    started = time.monotonic()
    if rows and not args.reanalyze_only:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase509 behavior requires CUDA")
        try:
            model, tokenizer, device = load_model(args.model, use_8bit=True if args.use_8bit else None)
            loaded = True
            score_rows(model, tokenizer, device, rows, args.batch_size)
        finally:
            if model is not None:
                release_model(model)
            gc.collect()
            torch.cuda.empty_cache()
    runtime = time.monotonic() - started
    if args.reanalyze_only and previous_summary is not None:
        loaded = bool(previous_summary["model_weights_loaded"])

    contract_rows = {
        name: [row for row in rows if row["contract"] == name]
        for name in ("relation", "binding", "joint")
    }
    summaries: dict[str, Any] = {}
    if contract_rows["relation"]:
        summaries["R"] = relation_summary(contract_rows["relation"], frozen)
    if contract_rows["binding"]:
        summaries["B"] = binding_summary(contract_rows["binding"], frozen)
    if contract_rows["joint"]:
        summaries["J"] = joint_summary(contract_rows["joint"], frozen)

    if not args.reanalyze_only:
        for row in rows:
            row["model"] = args.model
            row.pop("prompt", None)
        write_jsonl(rows_path, rows)
    summary = {
        "schema_version": f"phase{phase}_dual_contract_behavior.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if rows else "no_authorized_contracts",
        "stage": args.stage,
        "split": split,
        "model": args.model,
        "cuda_used": loaded,
        "model_weights_loaded": loaded,
        "runtime_seconds": runtime,
        "row_count": len(rows),
        "sealed_split_read": False,
        "reanalyzed_from_existing_logits": bool(args.reanalyze_only),
        "contract_summaries": summaries,
        "passed_contracts": [name for name, item in summaries.items() if item["gate_pass"]],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(rows_path)
    print(summary_path)


if __name__ == "__main__":
    main()
