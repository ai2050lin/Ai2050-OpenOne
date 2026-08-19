#!/usr/bin/env python3
"""Independent audit for Phase1368 C056 all-path causal competition."""
from __future__ import annotations

import json
import math
import py_compile
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1368_c056_all_path_causal_competition"
CONTRACT = TESTS / "result/phase1364_c056_hidden_path_contract"
CAMERA = TESTS / "result/phase1367_c056_qwen_path_identity_camera"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    protocol = load(CONTRACT / "protocol/preregistration.json")
    camera = load(CAMERA / "analysis/final.json")
    camera_audit = load(CAMERA / "audit/independent_final_audit.json")
    manifest = load(OUT / "protocol/execution_manifest.json")
    summary = load(OUT / "analysis/qwen3_all_path_causal.json")
    final = load(OUT / "analysis/final.json")
    raw = rows(OUT / "raw/qwen3_all_path_causal.jsonl")
    gate = protocol["causal"]
    recomputed_checks, recomputed_qualified = {}, {}
    for name, path in protocol["paths"].items():
        values = [row for row in raw if row["path"] == name]
        cp_checks = {}
        for checkpoint in path["checkpoints"]:
            key = f'{checkpoint["role"]}@{checkpoint["layer"]}'
            correct = [row["checkpoint_alpha"][key]["correct_clean"] for row in values]
            wrong = [row["checkpoint_alpha"][key]["wrong_identity_true"] for row in values]
            status = [row["checkpoint_alpha"][key]["status_true"] for row in values]
            advantage = [c - max(w, s) for c, w, s in zip(correct, wrong, status)]
            cp_checks[key] = {
                "projection": statistics.median(correct) >= gate["checkpoint_recovery_projection_median_min"],
                "advantage": statistics.median(advantage) >= gate["checkpoint_correct_over_controls_median_min"],
                "win": sum(c > max(w, s) for c, w, s in zip(correct, wrong, status)) / len(values)
                       >= gate["checkpoint_correct_over_controls_win_min"],
                "self": max(row["self_checkpoint_relative_l2"][key] for row in values)
                        <= gate["self_checkpoint_relative_l2_max"],
            }
        correct = [row["output_gain"]["correct_clean"] for row in values]
        wrong = [row["output_gain"]["wrong_identity_true"] for row in values]
        status = [row["output_gain"]["status_true"] for row in values]
        advantage = [c - max(w, s) for c, w, s in zip(correct, wrong, status)]
        out_checks = {
            "gain": statistics.median(correct) >= gate["output_gain_median_min"],
            "advantage": statistics.median(advantage) >= gate["output_correct_over_controls_median_min"],
            "win": sum(c > max(w, s) for c, w, s in zip(correct, wrong, status)) / len(values)
                   >= gate["output_correct_over_controls_win_min"],
            "self": max(abs(row["output_gain"]["self"]) for row in values) <= gate["self_output_max_abs_diff"],
        }
        recomputed_checks[name] = {"checkpoints": cp_checks, "output": out_checks}
        recomputed_qualified[name] = all(all(v.values()) for v in cp_checks.values()) and all(out_checks.values())
    checks = {
        "camera_parent": camera["camera_qualified"] and camera_audit["all_checks_passed"],
        "contract_hash": manifest["contract_sha256"] == protocol["contract_sha256"],
        "frozen_paths": manifest["paths"] == protocol["paths"],
        "exact_shape": manifest["rows_per_case"] == 24,
        "all_cases_paths": len(raw) == 96 * 5 and {row["path"] for row in raw} == set(protocol["paths"]),
        "finite": all(math.isfinite(value) for row in raw for value in row["output_gain"].values()) and
                  all(math.isfinite(value) for row in raw for cp in row["checkpoint_alpha"].values()
                      for value in cp.values()),
        "checks_recomputed": recomputed_checks == summary["path_checks"],
        "qualification_recomputed": recomputed_qualified == summary["path_qualified"],
        "qualified_list": sorted(name for name, value in recomputed_qualified.items() if value)
                          == sorted(summary["qualified_paths"]) == sorted(final["qualified_paths"]),
        "all_paths_executed": manifest["all_paths_run_even_after_failures"] and len(recomputed_qualified) == 5,
        "closed": final["authorization"] == "close_c056_after_frozen_all_path_competition"
                  and final["campaign_closed"],
    }
    py_compile.compile(str(TESTS / "phase1368_c056_all_path_causal_competition.py"), doraise=True)
    py_compile.compile(str(TESTS / "phase1368_c056_all_path_causal_competition_audit.py"), doraise=True)
    checks["scripts_compile"] = True
    result = {
        "phase": 1368, "campaign": "C056", "checks": checks,
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
