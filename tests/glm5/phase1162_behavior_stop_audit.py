#!/usr/bin/env python3
"""Independent audit of the Phase1162 behavior-stop branch."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
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
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    behavior = read_json(OUT_ROOT / "runs/behavior_stop/behavior_stop.json")
    final = read_json(OUT_ROOT / "analysis/final.json")
    checkpoint = OUT_ROOT / "runs/behavior_stop/failed_model.pt"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    model, _config, lexicon = phase.load_checkpoint(checkpoint, torch.device("cuda"))
    inputs, targets = phase.all_training_examples(lexicon)
    recomputed = phase.evaluate_behavior(model, inputs, targets, lexicon)
    stored = behavior["metrics"]
    protocol_body = dict(protocol)
    stored_protocol_digest = protocol_body.pop("protocol_digest")
    checks = {
        "protocol_digest": digest(protocol_body) == stored_protocol_digest,
        "primary_source_frozen": sha256_file(phase.SCRIPT) == protocol["source_hashes"]["primary_script"],
        "checkpoint_hash": sha256_file(checkpoint) == behavior["checkpoint_sha256"],
        "model_id": behavior["model_id"] == phase.model_id(phase.model_seed("compact", 0)),
        "stored_behavior_not_qualified": not stored["qualified"],
        "stored_accuracy_complete": stored["accuracy"] == 1.0,
        "stored_probability_failed": stored["minimum_probability"]
        < phase.THRESHOLDS["behavior_min_probability_min"],
        "recomputed_accuracy": recomputed["accuracy"] == stored["accuracy"],
        "recomputed_minimum_probability": abs(
            recomputed["minimum_probability"] - stored["minimum_probability"]
        ) <= 1e-6,
        "recomputed_finite": recomputed["finite_fraction"] == 1.0,
        "behavior_stop_digest": digest(
            {key: value for key, value in behavior.items() if key != "behavior_stop_digest"}
        ) == behavior["behavior_stop_digest"],
        "predictions_absent": not (OUT_ROOT / "predictions/predictions.npz").exists(),
        "holdout_absent": not (OUT_ROOT / "runs/models/holdout_responses.npz").exists(),
        "score_absent": not (OUT_ROOT / "analysis/score.json").exists(),
        "final_unknown_not_negative": final["global_transfer_tested"] is False
        and final["high_order_stress_tested"] is False
        and final["global_transfer_passed"] is None
        and final["high_order_stress_passed"] is None,
        "final_auto_stop": not final["auto_continue"],
        "final_no_overclaim": not final["causal_graph_recovered"]
        and not final["physical_hyperedges_recovered"]
        and not final["full_mechanism_recovery_complete"],
        "final_digest": digest({key: value for key, value in final.items() if key != "final_digest"})
        == final["final_digest"],
    }
    audit = {
        "phase": phase.PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": "behavior_gate_stop",
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "recomputed_behavior": recomputed,
        "primary_final_digest": final["final_digest"],
    }
    audit["audit_digest"] = digest(audit)
    write_json(OUT_ROOT / "audit/behavior_stop_audit.json", audit)
    print(canonical(audit))
    if not audit["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
