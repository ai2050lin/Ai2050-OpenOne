#!/usr/bin/env python3
"""Independent audit for Phase1367 C056 exact-shape identity camera."""
from __future__ import annotations

import json
import math
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1367_c056_qwen_path_identity_camera"
CONTRACT = TESTS / "result/phase1364_c056_hidden_path_contract"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    protocol = load(CONTRACT / "protocol/preregistration.json")
    manifest = load(OUT / "protocol/execution_manifest.json")
    summary = load(OUT / "analysis/qwen3_path_identity_camera.json")
    final = load(OUT / "analysis/final.json")
    material = rows(OUT / "material/calibration_cases.jsonl")
    raw = rows(OUT / "raw/qwen3_path_identity_camera.jsonl")
    expected_records = sum(len(path["checkpoints"]) for path in protocol["paths"].values()) * len(manifest["arms"]) * len(material)
    recomputed = {}
    for name in protocol["paths"]:
        values = [row for row in raw if row["path"] == name]
        recomputed[name] = {
            "output": max(abs(row["margin_diff"]) for row in values) <= protocol["camera"]["output_margin_max_abs_diff"],
            "checkpoint": max(row["checkpoint_relative_l2"] for row in values) <= protocol["camera"]["checkpoint_relative_l2_max"],
        }
    qualified = all(all(value.values()) for value in recomputed.values())
    cells = {(row["partition"], row["surface"]): 0 for row in material}
    for row in material:
        cells[(row["partition"], row["surface"])] += 1
    checks = {
        "contract_hash": manifest["contract_sha256"] == protocol["contract_sha256"],
        "frozen_paths": manifest["paths"] == protocol["paths"],
        "exact_shape": manifest["rows_per_case"] == 24 and manifest["same_execution_shape_as_phase1368"],
        "balanced_material": len(material) == 48 and set(cells.values()) == {4},
        "record_count": len(raw) == expected_records,
        "finite": all(math.isfinite(row["margin_diff"]) and math.isfinite(row["checkpoint_relative_l2"])
                      for row in raw),
        "checks_recomputed": recomputed == summary["checks"],
        "qualification_recomputed": qualified == summary["camera_qualified"] == final["camera_qualified"],
        "authorization": final["authorization"] == ("run_phase1368_c056_all_path_causal_competition"
                                                       if qualified else
                                                       "close_c056_camera_unqualified_without_mechanism_claim"),
        "observation_not_rehabilitated": manifest["observation_audit_passed"] is False,
    }
    py_compile.compile(str(TESTS / "phase1367_c056_qwen_path_identity_camera.py"), doraise=True)
    py_compile.compile(str(TESTS / "phase1367_c056_qwen_path_identity_camera_audit.py"), doraise=True)
    checks["scripts_compile"] = True
    result = {
        "phase": 1367, "campaign": "C056", "checks": checks,
        "passed": sum(checks.values()), "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    path = OUT / "audit/independent_final_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
