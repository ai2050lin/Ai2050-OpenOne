#!/usr/bin/env python3
"""Independent audit for Phase1357/C054."""
from __future__ import annotations

import json
import py_compile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1357_c054_same_batch_causal_contract"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    protocol = load(OUT / "protocol/preregistration.json")
    preaudit = load(OUT / "audit/pre_model_semantic_naturalness_zero_model_audit.json")
    final = load(OUT / "analysis/final.json")
    calibration = rows(OUT / "material/calibration_cases.jsonl")
    replay = rows(OUT / "material/causal_replay_manifest.jsonl")
    checks = {
        "preaudit": preaudit["all_checks_passed"],
        "calibration_count": len(calibration) == 48,
        "calibration_balance": Counter(x["span_length"] for x in calibration) == {1: 24, 2: 24},
        "replay_count": len(replay) == 324,
        "partition_balance": Counter(x["partition"] for x in replay) == {"confirmation": 162, "lockbox": 162},
        "surface_balance": Counter(x["surface"] for x in replay) == {"ordinary": 108, "dictionary": 108, "claim": 108},
        "single_token_contract": protocol["material"]["replay_tested_families"] == ["currency", "language", "emotion"],
        "frozen_layer": protocol["causal_replay"]["layer"] == 27,
        "finite_routes": set(protocol["causal_replay"]["routes"]) == {"state_transport", "paired_delta_transport"},
        "contract_link": final["contract_sha256"] == protocol["contract_sha256"],
        "authorization": final["authorization"] == "run_phase1358_c054_camera_calibration",
        "script_compiles": True,
    }
    try:
        py_compile.compile(str(TESTS / "phase1357_c054_same_batch_causal_contract.py"), doraise=True)
    except Exception:
        checks["script_compiles"] = False
    result = {"phase": 1357, "campaign": "C054", "checks": checks,
              "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "audit/independent_final_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
