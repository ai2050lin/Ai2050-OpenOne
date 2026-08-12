#!/usr/bin/env python3
"""Seal Phase1162 after its preregistered behavior gate stopped execution."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT_ROOT = ROOT / "tests/glm5/result/phase1162_modular_task_response_transfer"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1162_modular_task_response_transfer as phase  # noqa: E402


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    protocol = phase.verify_protocol()
    final_path = OUT_ROOT / "analysis/final.json"
    if final_path.exists():
        raise RuntimeError("refusing to overwrite final")
    forbidden = [
        OUT_ROOT / "predictions/predictions.npz",
        OUT_ROOT / "runs/models/holdout_responses.npz",
        OUT_ROOT / "analysis/score.json",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("forbidden downstream artifacts exist")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    architecture = "compact"
    replicate = 0
    seed = phase.model_seed(architecture, replicate)
    identifier = phase.model_id(seed)
    lexicon = phase.make_lexicon(seed + 17)
    model, metrics = phase.train_model(
        phase.ARCHITECTURES[architecture], seed, lexicon, torch.device("cuda")
    )
    if metrics["qualified"]:
        raise RuntimeError("behavior unexpectedly qualified during stop sealing")
    stop_root = OUT_ROOT / "runs/behavior_stop"
    stop_root.mkdir(parents=True, exist_ok=True)
    checkpoint = stop_root / "failed_model.pt"
    torch.save(
        phase.checkpoint_payload(model, phase.ARCHITECTURES[architecture], lexicon),
        checkpoint,
    )
    behavior = {
        "phase": phase.PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_digest": protocol["protocol_digest"],
        "architecture": architecture,
        "replicate": replicate,
        "seed": seed,
        "model_id": identifier,
        "metrics": metrics,
        "failed_checks": {
            "accuracy": metrics["accuracy"] < phase.THRESHOLDS["behavior_accuracy_min"],
            "minimum_probability": metrics["minimum_probability"]
            < phase.THRESHOLDS["behavior_min_probability_min"],
            "finite": metrics["finite_fraction"] < phase.THRESHOLDS["finite_fraction_min"],
        },
        "checkpoint_sha256": sha256_file(checkpoint),
        "stop_finalizer_sha256": sha256_file(SCRIPT),
    }
    behavior["behavior_stop_digest"] = digest(behavior)
    write_json(stop_root / "behavior_stop.json", behavior)
    final = {
        "phase": phase.PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": protocol["title"],
        "protocol_digest": protocol["protocol_digest"],
        "decision": "behavior_gate_stop",
        "behavior_model_id": identifier,
        "behavior_stop_digest": behavior["behavior_stop_digest"],
        "global_transfer_tested": False,
        "high_order_stress_tested": False,
        "global_transfer_passed": None,
        "high_order_stress_passed": None,
        "full_independent_task_transfer_passed": None,
        "causal_graph_recovered": False,
        "physical_hyperedges_recovered": False,
        "full_mechanism_recovery_complete": False,
        "claim_scope": "The frozen modular-task extension stopped at behavior qualification; response-transfer endpoints were not tested.",
        "auto_continue": False,
        "auto_continue_reason": "The only authorized independent task-family extension failed its frozen behavior confidence gate; changing training or task now would start a new research program.",
        "non_implications": [
            "Behavior confidence failure is not evidence that the Phase1161 estimator fails on this task.",
            "Accuracy 1.0 with a low worst-case probability does not satisfy the frozen high-confidence substrate.",
            "Unrun prediction and stress endpoints must remain unknown, not negative.",
        ],
    }
    final["final_digest"] = digest(final)
    write_json(final_path, final)
    print(canonical(final))


if __name__ == "__main__":
    main()
