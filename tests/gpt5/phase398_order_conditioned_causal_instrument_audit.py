#!/usr/bin/env python3
"""Audit Phase398 query-end causal instrumentation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase398_joint_binding"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    rows = []
    completes = []
    for model in MODELS:
        root = OUT / f"causal/instrument/private/models/{model}"
        completes.append(read_json(root / "complete.json"))
        rows.extend(read_jsonl(root / "direction_rows.jsonl"))
    scenario_rows = [scenario for row in rows for scenario in row["scenario_rows"]]
    no_patch = [row for row in scenario_rows if row["scenario"] == "no_intervention"]
    identity = [row for row in scenario_rows if row["scenario"] == "identity_candidate_parent"]
    patched = [row for row in scenario_rows if row["scenario"] != "no_intervention"]
    report = {
        "schema_version": "72.12.0",
        "phase_id": "Phase398-OrderConditionedCausalInstrumentAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "model_count": 3,
            "group_count": sum(row["group_count"] for row in completes),
            "direction_count": len(rows),
            "scenario_count": len(scenario_rows),
        },
        "gates": {
            "all_models_complete": all(row["valid"] for row in completes),
            "all_144_natural_recipient_answers_present": len(no_patch) == 144 and all(row["recipient_target_present"] for row in no_patch),
            "all_144_identity_recipient_answers_present": len(identity) == 144 and all(row["recipient_target_present"] for row in identity),
            "all_720_patches_called_once": len(patched) == 720 and all(row["patch_audit"]["patch_call_count"] == 1 for row in patched),
            "all_patch_errors_zero": all(row["patch_audit"]["max_patch_error"] == 0 and row["patch_audit"]["max_outside_error"] == 0 for row in patched),
        },
    }
    authorized = all(report["gates"].values())
    report["authorization"] = {"run_causal_test": authorized, "single_neuron_scan": False}
    path = OUT / "phase398_order_conditioned_causal_instrument_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not authorized:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
