#!/usr/bin/env python3
"""Record and independently reproduce the preregistered Phase1179 engineering abort."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1179_free_dual_path_response_camera as phase1179  # noqa: E402
import phase1179_free_training_library as lib  # noqa: E402


OUTPUT = phase1179.OUT_ROOT / "analysis/execution_abort.json"


def main() -> None:
    protocol = phase1179.read_json(phase1179.PROTOCOL_PATH)
    task = phase1179.SPLITS["discovery"].tasks[0]
    seed = phase1179.system_seed("discovery", 0, 0, 0, 0)
    observed: dict[str, Any]
    try:
        lib.train_system(task, "endpoint", seed, 0, torch.device("cuda"))
        observed = {"raised": False}
    except RuntimeError as exc:
        observed = {
            "raised": True,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

    checks = {
        "cuda_available": torch.cuda.is_available(),
        "protocol_digest_valid": lib.digest(
            {key: value for key, value in protocol.items() if key != "protocol_digest"}
        ) == protocol["protocol_digest"],
        "frozen_main_hash_valid": phase1179.sha256_file(phase1179.SCRIPT_PATH)
        == protocol["scripts"]["main_sha256"],
        "frozen_library_hash_valid": phase1179.sha256_file(phase1179.LIBRARY_PATH)
        == protocol["scripts"]["library_sha256"],
        "frozen_audit_hash_valid": phase1179.sha256_file(phase1179.AUDIT_PATH)
        == protocol["scripts"]["audit_sha256"],
        "no_discovery_artifacts": not (phase1179.OUT_ROOT / "runs/discovery").exists(),
        "first_system_reproduces_abort": observed.get("raised", False)
        and "functional parameter norm exceeded ballast target" in observed.get("message", ""),
    }
    payload = {
        "phase": phase1179.PHASE,
        "status": "preregistered_engineering_abort",
        "classification": "instrument_feasibility_failure; no scientific camera result",
        "failed_stage": "first discovery system before any split artifact was written",
        "first_system": {
            "task": task.name,
            "modulus": task.modulus,
            "cohort": "endpoint",
            "block": 0,
            "config_index": 0,
            "seed": seed,
        },
        "observed": observed,
        "checks": checks,
        "passed": all(checks.values()),
        "phase1179_closed": True,
        "scientific_claim": "untested",
        "auto_continue_within_phase1179": False,
    }
    payload["abort_digest"] = lib.digest(payload)
    phase1179.write_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
