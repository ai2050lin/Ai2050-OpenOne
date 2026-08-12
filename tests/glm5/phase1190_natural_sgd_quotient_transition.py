from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402
import phase1187_typed_evidence_compiler as p1187  # noqa: E402
import phase1188_terminal_three_evidence_confirmation as p1188  # noqa: E402
import phase1189_quotient_formation_operator_calibration as p1189  # noqa: E402


PHASE = 1190
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1190_natural_sgd_quotient_transition_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1190_natural_sgd_quotient_transition"
DEVELOPMENT_ROWS = OUT_ROOT / "development/events.jsonl"
DEVELOPMENT_SUMMARY = OUT_ROOT / "development/summary.json"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
TRAINING_METRICS = OUT_ROOT / "runs/training/training_metrics.jsonl"
TRAINING_SEAL = OUT_ROOT / "runs/training/seal.json"
CHECKPOINT_ROOT = OUT_ROOT / "runs/training/checkpoints"
EVENT_ROWS = OUT_ROOT / "analysis/events.jsonl"
BEHAVIOR_ROWS = OUT_ROOT / "analysis/endpoint_behavior.jsonl"
SUMMARY_PATH = OUT_ROOT / "analysis/summary.json"
CLAIMS_PATH = OUT_ROOT / "analysis/typed_claims.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"

DEVELOPMENT_SOURCE = p1171.OUT_ROOT / "runs/training/checkpoints"
MODULUS = 61
WIDTH = 128
REPLICATES = 8
TASK_COUNT = 8
TASK_SELECTION_SEED = 1_190_001_017
PANEL_SEED_OFFSET = 1_190
STEPS = (25, 50, 75, 100, 150, 200, 350, 500, 750, 1000, 10_000)
FORMATION_STEPS = STEPS[:-1]
INTERVALS = tuple(zip(FORMATION_STEPS[:-1], FORMATION_STEPS[1:]))
TIME_NULL = {
    (25, 50): (50, 75),
    (50, 75): (75, 100),
    (75, 100): (25, 50),
    (100, 150): (150, 200),
    (150, 200): (100, 150),
    (200, 350): (350, 500),
    (350, 500): (200, 350),
    (500, 750): (750, 1000),
    (750, 1000): (500, 750),
}
TRAINING = {
    "learning_rate": 0.001,
    "weight_decay": 1.0,
    "precision": "bfloat16",
    "batching": "full_batch",
    "maximum_step": max(STEPS),
}
THRESHOLDS = {
    "event_norm_min": 0.02,
    "eligible_event_fraction_min": 0.90,
    "true_cosine_mean_min": 0.50,
    "replicate_null_advantage_mean_min": 0.15,
    "time_null_advantage_mean_min": 0.20,
    "replicate_positive_fraction_min": 0.65,
    "time_positive_fraction_min": 0.65,
    "positive_task_count_per_split_min": 3,
    "system_count_per_split": 32,
    "task_count_per_split": 4,
    "endpoint_train_accuracy_min": 0.99,
    "endpoint_holdout_accuracy_min": 0.90,
    "qualified_system_count_per_split_min": 24,
    "qualified_task_count_per_split_min": 3,
    "qualified_system_count_per_task_min": 6,
    "all_logits_finite_required": True,
}


def eligible_operations() -> list[tuple[int, int, int]]:
    excluded = set(p1171.OPERATION_SAMPLE) | set(p1188.OPERATIONS) | {
        p1171.PILOT_OPERATION,
        (19, 23, 7),
        (29, 31, 11),
    }
    return [operation for operation in p1171.eligible_operations() if operation not in excluded]


OPERATIONS = tuple(random.Random(TASK_SELECTION_SEED).sample(eligible_operations(), TASK_COUNT))
TASKS = {
    f"formation_affine_{index:02d}_a{operation[0]}_b{operation[1]}_g{operation[2]}": operation
    for index, operation in enumerate(OPERATIONS)
}


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_seed(task_index: int, replicate: int) -> int:
    return 1_190_000_000 + task_index * 100_003 + replicate * 1_009


def checkpoint_path(task_name: str, replicate: int, seed: int, step: int) -> Path:
    return CHECKPOINT_ROOT / f"{task_name}_r{replicate}_s{seed}_step{step:05d}.pt"


def trajectory_id(payload: dict[str, Any]) -> str:
    return f"{payload['task_name']}_r{payload['replicate']}_s{payload['seed']}"


def checkpoint_payload(
    model: p1171.RoleSquareNetwork,
    task_name: str,
    task_index: int,
    operation: tuple[int, int, int],
    replicate: int,
    seed: int,
    step: int,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "task_name": task_name,
        "task_index": task_index,
        "operation": operation,
        "replicate": replicate,
        "seed": seed,
        "step": step,
        "config": {"modulus": MODULUS, "width": WIDTH},
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }


def development_endpoints() -> list[Path]:
    return sorted(DEVELOPMENT_SOURCE.glob("*step10000.pt"))


def formal_endpoints() -> list[Path]:
    return sorted(CHECKPOINT_ROOT.glob("*step10000.pt"))


def path_at(endpoint: Path, step: int) -> Path:
    return endpoint.with_name(endpoint.name.replace("step10000", f"step{step:05d}"))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12)
    return float(np.dot(left, right) / denominator)


def build_transition_vectors(
    endpoints: list[Path], corpus: str, device: torch.device
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for endpoint_index, endpoint in enumerate(endpoints):
        payload = p1189.load_payload(endpoint)
        panel = p1189.panel_from_payload(payload)
        states: dict[int, dict[str, np.ndarray]] = {}
        for step in FORMATION_STEPS:
            current = p1189.load_model(p1189.load_payload(path_at(endpoint, step)), device)
            states[step] = {
                "calibration": p1189.response_unit_shape(current, panel, panel.train_mask, device),
                "evaluation": p1189.response_unit_shape(current, panel, panel.holdout_mask, device),
            }
            del current
        for left, right in INTERVALS:
            calibration = states[right]["calibration"] - states[left]["calibration"]
            evaluation = states[right]["evaluation"] - states[left]["evaluation"]
            rows.append(
                {
                    "corpus": corpus,
                    "task_name": str(payload["task_name"]),
                    "task_index": int(payload["task_index"]),
                    "operation": [int(value) for value in payload["operation"]],
                    "replicate": int(payload["replicate"]),
                    "seed": int(payload["seed"]),
                    "trajectory_id": trajectory_id(payload),
                    "split": "development" if corpus == "development" else (
                        "discovery" if int(payload["task_index"]) < 4 else "confirmation"
                    ),
                    "left_step": left,
                    "right_step": right,
                    "interval": f"{left}-{right}",
                    "calibration_delta": calibration.tolist(),
                    "evaluation_delta": evaluation.tolist(),
                    "calibration_norm": float(np.linalg.norm(calibration)),
                    "evaluation_norm": float(np.linalg.norm(evaluation)),
                    "true_cosine": cosine(calibration, evaluation),
                }
            )
        print(
            canonical_json(
                {
                    "corpus": corpus,
                    "trajectory": endpoint_index + 1,
                    "total": len(endpoints),
                    "checkpoint": endpoint.name,
                }
            ),
            flush=True,
        )
        torch.cuda.empty_cache()
    return rows


def add_matched_nulls(rows: list[dict[str, Any]]) -> None:
    lookup = {
        (row["task_name"], row["replicate"], row["left_step"], row["right_step"]): row
        for row in rows
    }
    for row in rows:
        replicate_null = lookup[
            (
                row["task_name"],
                (row["replicate"] + 1) % REPLICATES,
                row["left_step"],
                row["right_step"],
            )
        ]
        null_left, null_right = TIME_NULL[(row["left_step"], row["right_step"])]
        time_null = lookup[(row["task_name"], row["replicate"], null_left, null_right)]
        calibration = np.asarray(row["calibration_delta"], dtype=np.float64)
        row["replicate_null_trajectory_id"] = replicate_null["trajectory_id"]
        row["replicate_null_cosine"] = cosine(
            calibration, np.asarray(replicate_null["evaluation_delta"], dtype=np.float64)
        )
        row["time_null_interval"] = time_null["interval"]
        row["time_null_cosine"] = cosine(
            calibration, np.asarray(time_null["evaluation_delta"], dtype=np.float64)
        )
        row["replicate_advantage"] = row["true_cosine"] - row["replicate_null_cosine"]
        row["time_advantage"] = row["true_cosine"] - row["time_null_cosine"]
        row["eligible"] = bool(
            row["calibration_norm"] >= THRESHOLDS["event_norm_min"]
            and row["evaluation_norm"] >= THRESHOLDS["event_norm_min"]
        )


def task_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for task_name in sorted({row["task_name"] for row in rows}):
        selected = [row for row in rows if row["task_name"] == task_name and row["eligible"]]
        result.append(
            {
                "task_name": task_name,
                "eligible_count": len(selected),
                "replicate_advantage_mean": float(np.mean([row["replicate_advantage"] for row in selected])),
                "time_advantage_mean": float(np.mean([row["time_advantage"] for row in selected])),
                "positive": bool(
                    np.mean([row["replicate_advantage"] for row in selected]) > 0.0
                    and np.mean([row["time_advantage"] for row in selected]) > 0.0
                ),
            }
        )
    return result


def summarize_events(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = rows if split in ("development", "all") else [row for row in rows if row["split"] == split]
    eligible = [row for row in selected if row["eligible"]]
    tasks = task_summaries(selected)
    expected_systems = 64 if split == "development" else THRESHOLDS["system_count_per_split"]
    expected_tasks = 8 if split == "development" else THRESHOLDS["task_count_per_split"]
    expected_events = expected_systems * len(INTERVALS)
    result = {
        "split": split,
        "event_count": len(selected),
        "expected_event_count": expected_events,
        "eligible_event_count": len(eligible),
        "eligible_event_fraction": len(eligible) / max(len(selected), 1),
        "system_count": len({row["trajectory_id"] for row in selected}),
        "task_count": len({row["task_name"] for row in selected}),
        "true_cosine_mean": float(np.mean([row["true_cosine"] for row in eligible])),
        "replicate_null_cosine_mean": float(np.mean([row["replicate_null_cosine"] for row in eligible])),
        "time_null_cosine_mean": float(np.mean([row["time_null_cosine"] for row in eligible])),
        "replicate_null_advantage_mean": float(np.mean([row["replicate_advantage"] for row in eligible])),
        "time_null_advantage_mean": float(np.mean([row["time_advantage"] for row in eligible])),
        "replicate_positive_fraction": float(np.mean([row["replicate_advantage"] > 0 for row in eligible])),
        "time_positive_fraction": float(np.mean([row["time_advantage"] > 0 for row in eligible])),
        "positive_task_count": sum(task["positive"] for task in tasks),
        "task_summaries": tasks,
        "event_norm_mean": float(np.mean([row["evaluation_norm"] for row in eligible])),
        "event_norm_min": min(row["evaluation_norm"] for row in eligible),
    }
    minimum_positive_tasks = 6 if split == "development" else THRESHOLDS["positive_task_count_per_split_min"]
    result["gate_pass"] = bool(
        result["event_count"] == expected_events
        and result["system_count"] == expected_systems
        and result["task_count"] == expected_tasks
        and result["eligible_event_fraction"] >= THRESHOLDS["eligible_event_fraction_min"]
        and result["true_cosine_mean"] >= THRESHOLDS["true_cosine_mean_min"]
        and result["replicate_null_advantage_mean"] >= THRESHOLDS["replicate_null_advantage_mean_min"]
        and result["time_null_advantage_mean"] >= THRESHOLDS["time_null_advantage_mean_min"]
        and result["replicate_positive_fraction"] >= THRESHOLDS["replicate_positive_fraction_min"]
        and result["time_positive_fraction"] >= THRESHOLDS["time_positive_fraction_min"]
        and result["positive_task_count"] >= minimum_positive_tasks
    )
    return result


def source_hashes() -> dict[str, str]:
    paths = [SCRIPT, AUDIT_SCRIPT, Path(p1171.__file__), Path(p1187.__file__), Path(p1189.__file__)]
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def develop() -> None:
    endpoints = development_endpoints()
    if len(endpoints) != 64:
        raise RuntimeError("Phase1171 development trajectory count changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = build_transition_vectors(endpoints, "development", torch.device("cuda"))
    add_matched_nulls(rows)
    write_jsonl(DEVELOPMENT_ROWS, rows)
    summary = summarize_events(rows, "development")
    summary.update(
        {
            "phase": PHASE,
            "created_at_utc": utc_now(),
            "rows_sha256": file_sha256(DEVELOPMENT_ROWS),
            "thresholds": THRESHOLDS,
            "pilot_scope": "Thresholds were frozen from task 0 only before this eight-task development run.",
            "formal_data_read": False,
        }
    )
    summary["summary_digest"] = digest({key: value for key, value in summary.items() if key != "summary_digest"})
    write_json(DEVELOPMENT_SUMMARY, summary)
    if not summary["gate_pass"]:
        raise RuntimeError("development formation event gate failed")


def preregister() -> None:
    development = read_json(DEVELOPMENT_SUMMARY)
    phase1189 = read_json(p1189.FINAL_PATH)
    if not development["gate_pass"] or not phase1189["main_gate_pass"]:
        raise RuntimeError("upstream authorization failed")
    if TRAINING_SEAL.exists() or EVENT_ROWS.exists():
        raise RuntimeError("formal outcomes already exist")
    allocation = {
        "discovery": sorted(TASKS)[:4],
        "confirmation": sorted(TASKS)[4:],
    }
    protocol = {
        "phase": PHASE,
        "title": "Natural SGD quotient-transition external validity",
        "created_at_utc": utc_now(),
        "scientific_question": (
            "Do actual free-network SGD updates produce gauge-invariant response transitions that transfer "
            "from one data half to another and exceed same-progress cross-implementation and matched-duration "
            "cross-time controls on all-new tasks?"
        ),
        "upstream": {
            "phase1189_final_sha256": file_sha256(p1189.FINAL_PATH),
            "phase1189_final_digest": phase1189["final_digest"],
            "development_summary_sha256": file_sha256(DEVELOPMENT_SUMMARY),
            "development_rows_sha256": file_sha256(DEVELOPMENT_ROWS),
            "development_summary_digest": development["summary_digest"],
        },
        "tasks": TASKS,
        "task_selection_seed": TASK_SELECTION_SEED,
        "replicates": REPLICATES,
        "trajectory_count": TASK_COUNT * REPLICATES,
        "steps": STEPS,
        "formation_intervals": INTERVALS,
        "time_null_mapping": {f"{a}-{b}": f"{c}-{d}" for (a, b), (c, d) in TIME_NULL.items()},
        "allocation": allocation,
        "allocation_digest": digest(allocation),
        "training": TRAINING,
        "thresholds": THRESHOLDS,
        "source_hashes": source_hashes(),
        "evidence_contract_sha256": file_sha256(p1187.CONTRACT_PATH),
        "hard_stops": [
            "All tasks, seeds, trajectories, intervals, and low-scoring events remain in the report.",
            "Only a fixed event-norm eligibility threshold may prevent undefined near-zero cosines.",
            "No ridge, feature search, endpoint prediction, layer search, or threshold retuning is allowed.",
            "A failure closes this natural-SGD quotient-transition registry.",
            "A pass is confined to RoleSquare affine tasks and does not authorize Transformer transfer.",
        ],
        "authorization": {
            "phase1191_accumulation_or_causal_preregistration": "both splits plus independent audit must pass",
            "transformer_or_llm_transfer": False,
            "theory_closure": False,
        },
    }
    protocol["protocol_digest"] = digest({key: value for key, value in protocol.items() if key != "protocol_digest"})
    write_json(PROTOCOL_PATH, protocol)


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    expected = digest({key: value for key, value in protocol.items() if key != "protocol_digest"})
    if expected != protocol["protocol_digest"]:
        raise RuntimeError("protocol digest mismatch")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("source code changed after preregistration")
    if file_sha256(DEVELOPMENT_SUMMARY) != protocol["upstream"]["development_summary_sha256"]:
        raise RuntimeError("development summary changed")
    if file_sha256(DEVELOPMENT_ROWS) != protocol["upstream"]["development_rows_sha256"]:
        raise RuntimeError("development rows changed")
    if file_sha256(p1189.FINAL_PATH) != protocol["upstream"]["phase1189_final_sha256"]:
        raise RuntimeError("Phase1189 final changed")
    return protocol


def train() -> None:
    protocol = verify_protocol()
    if TRAINING_SEAL.exists() or EVENT_ROWS.exists():
        raise RuntimeError("training or analysis already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    metric_rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    for task_index, (task_name, operation) in enumerate(TASKS.items()):
        for replicate in range(REPLICATES):
            seed = model_seed(task_index, replicate)
            set_seed(seed)
            data = p1171.make_data(operation, seed + 17)
            model = p1171.RoleSquareNetwork(p1171.RoleSquareConfig(MODULUS, WIDTH)).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=TRAINING["learning_rate"],
                weight_decay=TRAINING["weight_decay"],
            )
            train_x = data["train_x"].to(device)
            train_y = data["train_y"].to(device)
            for step in range(1, max(STEPS) + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(train_x).float()
                    loss = F.cross_entropy(logits, train_y)
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(f"nonfinite loss: {task_name}/{replicate}/{step}")
                loss.backward()
                optimizer.step()
                if step not in STEPS:
                    continue
                model.eval()
                metrics = p1171.evaluate(model, data["train_x"], data["train_y"], device)
                path = checkpoint_path(task_name, replicate, seed, step)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    checkpoint_payload(model, task_name, task_index, operation, replicate, seed, step), path
                )
                checkpoint_hashes[path.name] = file_sha256(path)
                metric_rows.append(
                    {
                        "task_name": task_name,
                        "task_index": task_index,
                        "operation": list(operation),
                        "replicate": replicate,
                        "seed": seed,
                        "step": step,
                        "loss": float(loss.item()),
                        "train": metrics,
                        "train_pair_digest": digest(data["train_x"].tolist()),
                        "sealed_holdout_pair_digest": digest(data["holdout_x"].tolist()),
                        "holdout_evaluated_during_training": False,
                        "checkpoint_sha256": checkpoint_hashes[path.name],
                    }
                )
            print(canonical_json({"trained": task_name, "replicate": replicate}), flush=True)
            del model, optimizer, train_x, train_y
            gc.collect()
            torch.cuda.empty_cache()
    write_jsonl(TRAINING_METRICS, metric_rows)
    seal = {
        "phase": PHASE,
        "protocol_digest": protocol["protocol_digest"],
        "trajectory_count": TASK_COUNT * REPLICATES,
        "checkpoint_count": len(metric_rows),
        "training_metrics_sha256": file_sha256(TRAINING_METRICS),
        "checkpoint_hashes": checkpoint_hashes,
        "holdout_outcomes_absent": not EVENT_ROWS.exists() and not BEHAVIOR_ROWS.exists(),
    }
    seal["seal_digest"] = digest({key: value for key, value in seal.items() if key != "seal_digest"})
    write_json(TRAINING_SEAL, seal)


def verify_seal() -> dict[str, Any]:
    seal = read_json(TRAINING_SEAL)
    if file_sha256(TRAINING_METRICS) != seal["training_metrics_sha256"]:
        raise RuntimeError("training metrics changed")
    for name, expected in seal["checkpoint_hashes"].items():
        if file_sha256(CHECKPOINT_ROOT / name) != expected:
            raise RuntimeError(f"checkpoint changed: {name}")
    return seal


def endpoint_behavior(endpoint: Path, device: torch.device) -> dict[str, Any]:
    payload = p1189.load_payload(endpoint)
    data = p1171.make_data(tuple(payload["operation"]), int(payload["seed"]) + 17)
    model = p1189.load_model(payload, device)
    train = p1171.evaluate(model, data["train_x"], data["train_y"], device)
    holdout = p1171.evaluate(model, data["holdout_x"], data["holdout_y"], device)
    result = {
        "task_name": str(payload["task_name"]),
        "task_index": int(payload["task_index"]),
        "replicate": int(payload["replicate"]),
        "seed": int(payload["seed"]),
        "trajectory_id": trajectory_id(payload),
        "split": "discovery" if int(payload["task_index"]) < 4 else "confirmation",
        "train": train,
        "holdout": holdout,
    }
    result["qualified"] = bool(
        train["accuracy"] >= THRESHOLDS["endpoint_train_accuracy_min"]
        and holdout["accuracy"] >= THRESHOLDS["endpoint_holdout_accuracy_min"]
        and train["exact_all_finite"]
        and holdout["exact_all_finite"]
    )
    del model
    return result


def summarize_behavior(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    tasks = sorted({row["task_name"] for row in selected})
    passing_tasks = [
        task for task in tasks
        if sum(row["task_name"] == task and row["qualified"] for row in selected)
        >= THRESHOLDS["qualified_system_count_per_task_min"]
    ]
    result = {
        "split": split,
        "system_count": len(selected),
        "qualified_system_count": sum(row["qualified"] for row in selected),
        "task_count": len(tasks),
        "qualified_task_count": len(passing_tasks),
        "qualified_tasks": passing_tasks,
        "minimum_train_accuracy": min(row["train"]["accuracy"] for row in selected),
        "minimum_holdout_accuracy": min(row["holdout"]["accuracy"] for row in selected),
        "all_logits_finite": all(
            row["train"]["exact_all_finite"] and row["holdout"]["exact_all_finite"] for row in selected
        ),
    }
    result["gate_pass"] = bool(
        result["system_count"] == THRESHOLDS["system_count_per_split"]
        and result["task_count"] == THRESHOLDS["task_count_per_split"]
        and result["qualified_system_count"] >= THRESHOLDS["qualified_system_count_per_split_min"]
        and result["qualified_task_count"] >= THRESHOLDS["qualified_task_count_per_split_min"]
        and result["all_logits_finite"]
    )
    return result


def bounded(value: float, threshold: float, comparator: str) -> dict[str, Any]:
    return {
        "claim_type": "bounded_float",
        "gating": True,
        "value": float(value),
        "threshold": float(threshold),
        "comparator": comparator,
        "dtype": "float64",
    }


def compile_claims(summary: dict[str, Any]) -> dict[str, Any]:
    contract = read_json(p1187.CONTRACT_PATH)
    raw: dict[str, dict[str, Any]] = {}
    for split in ("discovery", "confirmation"):
        event = summary["events"][split]
        behavior = summary["behavior"][split]
        raw[split + ".eligible"] = bounded(
            event["eligible_event_fraction"], THRESHOLDS["eligible_event_fraction_min"], ">="
        )
        raw[split + ".true"] = bounded(
            event["true_cosine_mean"], THRESHOLDS["true_cosine_mean_min"], ">="
        )
        raw[split + ".replicate_advantage"] = bounded(
            event["replicate_null_advantage_mean"],
            THRESHOLDS["replicate_null_advantage_mean_min"],
            ">=",
        )
        raw[split + ".time_advantage"] = bounded(
            event["time_null_advantage_mean"], THRESHOLDS["time_null_advantage_mean_min"], ">="
        )
        raw[split + ".replicate_fraction"] = bounded(
            event["replicate_positive_fraction"], THRESHOLDS["replicate_positive_fraction_min"], ">="
        )
        raw[split + ".time_fraction"] = bounded(
            event["time_positive_fraction"], THRESHOLDS["time_positive_fraction_min"], ">="
        )
        raw[split + ".positive_tasks"] = bounded(
            event["positive_task_count"], THRESHOLDS["positive_task_count_per_split_min"], ">="
        )
        raw[split + ".behavior_systems"] = bounded(
            behavior["qualified_system_count"],
            THRESHOLDS["qualified_system_count_per_split_min"],
            ">=",
        )
        raw[split + ".behavior_tasks"] = bounded(
            behavior["qualified_task_count"], THRESHOLDS["qualified_task_count_per_split_min"], ">="
        )
    compiled = {name: p1187.compile_claim(claim, contract) for name, claim in raw.items()}
    conjunction = p1187.compile_claim(
        {
            "claim_type": "conjunction",
            "gating": True,
            "values": [bool(claim["authorizes"]) for claim in compiled.values()],
        },
        contract,
    )
    return {
        "raw": raw,
        "compiled": compiled,
        "conjunction": conjunction,
        "gate_pass": bool(conjunction["authorizes"]),
    }


def analyze() -> None:
    protocol = verify_protocol()
    seal = verify_seal()
    endpoints = formal_endpoints()
    if len(endpoints) != TASK_COUNT * REPLICATES:
        raise RuntimeError("formal endpoint count mismatch")
    device = torch.device("cuda")
    event_rows = build_transition_vectors(endpoints, "formal", device)
    add_matched_nulls(event_rows)
    write_jsonl(EVENT_ROWS, event_rows)
    behavior_rows = [endpoint_behavior(path, device) for path in endpoints]
    write_jsonl(BEHAVIOR_ROWS, behavior_rows)
    summary = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "training_seal_digest": seal["seal_digest"],
        "event_rows_sha256": file_sha256(EVENT_ROWS),
        "behavior_rows_sha256": file_sha256(BEHAVIOR_ROWS),
        "events": {
            split: summarize_events(event_rows, split) for split in ("discovery", "confirmation")
        },
        "behavior": {
            split: summarize_behavior(behavior_rows, split) for split in ("discovery", "confirmation")
        },
    }
    summary["formal_gate_pass"] = bool(
        all(summary["events"][split]["gate_pass"] for split in ("discovery", "confirmation"))
        and all(summary["behavior"][split]["gate_pass"] for split in ("discovery", "confirmation"))
    )
    summary["summary_digest"] = digest({key: value for key, value in summary.items() if key != "summary_digest"})
    write_json(SUMMARY_PATH, summary)
    write_json(CLAIMS_PATH, compile_claims(summary))


def finalize() -> None:
    protocol = verify_protocol()
    summary = read_json(SUMMARY_PATH)
    claims = read_json(CLAIMS_PATH)
    audit_pass = bool(AUDIT_PATH.exists() and read_json(AUDIT_PATH).get("gate_pass"))
    main_pass = bool(summary["formal_gate_pass"] and claims["gate_pass"] and audit_pass)
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "status": "natural_sgd_quotient_transition_confirmed" if main_pass else (
            "awaiting_independent_audit" if summary["formal_gate_pass"] and claims["gate_pass"] and not AUDIT_PATH.exists()
            else "natural_sgd_quotient_transition_failed"
        ),
        "protocol_digest": protocol["protocol_digest"],
        "summary_digest": summary["summary_digest"],
        "claims_sha256": file_sha256(CLAIMS_PATH),
        "audit_digest": read_json(AUDIT_PATH).get("audit_digest") if AUDIT_PATH.exists() else None,
        "formal_gate_pass": summary["formal_gate_pass"],
        "typed_claim_gate_pass": claims["gate_pass"],
        "independent_audit_pass": audit_pass,
        "main_gate_pass": main_pass,
        "evidence_grade": "E3_KT_free_network_transition" if main_pass else "no_upgrade",
        "authorized_next": {
            "phase1191_accumulation_or_causal_preregistration": main_pass,
            "automatic_unpreregistered_execution": False,
            "transformer_or_language_model_transfer": False,
            "theory_closure": False,
        },
        "claim_scope": (
            "A natural SGD update in all-new free RoleSquare networks has a model- and time-specific quotient "
            "response transition that transfers across measurement halves beyond two matched controls. This is "
            "a formation event, not yet an accumulated endpoint law or causal optimizer mechanism."
        ),
        "final_digest": None,
    }
    final["final_digest"] = digest({key: value for key, value in final.items() if key != "final_digest"})
    write_json(FINAL_PATH, final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("develop", "preregister", "train", "analyze", "finalize"))
    args = parser.parse_args()
    {
        "develop": develop,
        "preregister": preregister,
        "train": train,
        "analyze": analyze,
        "finalize": finalize,
    }[args.command]()


if __name__ == "__main__":
    main()
