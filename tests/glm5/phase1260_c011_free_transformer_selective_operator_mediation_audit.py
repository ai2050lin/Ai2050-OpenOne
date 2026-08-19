"""Independent artifact audit for Phase1260."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1260_c011_free_transformer_selective_operator_mediation"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_factorial_worlds.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
MODELS = OUT / "raw/model_results.jsonl"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def preaudit() -> dict[str, Any]:
    protocol = read(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    counts = {name: sum(row["partition"] == name for row in rows) for name in protocol["world_counts"]}
    checks = {
        "phase_campaign_contract": protocol.get("phase") == 1260 and protocol.get("campaign") == "C011" and protocol.get("contract_id") == "EXP-C011-WP02-001",
        "phase1259_dependency_frozen": set(protocol.get("phase1259_dependency", {})) == {"final", "audit"},
        "six_models_three_depths": len(protocol.get("model_seeds", {})) == 6 and set(protocol.get("architectures", {})) == {"shallow4", "middle6", "deep8"},
        "large_disjoint_material": len(rows) == 512 and counts == {"discovery": 128, "selection": 128, "confirmation": 256},
        "row_digests": all(digest({key: value for key, value in row.items() if key != "row_digest"}) == row["row_digest"] for row in rows),
        "factorial_panels": all(all(f"{panel}_ids" in row for panel in ("base", "target", "wrong", "null")) for row in rows),
        "same_executor_frozen": "same explicit" in protocol.get("executor_invariant", ""),
        "three_camera_outcomes": set(protocol.get("camera", {}).get("candidates", [])) == {"global_oblique_operator", "shift_conditioned_oblique_operator", "typed_abstention"},
        "selection_confirmation_separated": "no reselection" in protocol.get("camera", {}).get("confirmation", ""),
        "conjunctive_gates": set(protocol.get("gates", [])) >= {"behavior", "target_rescue", "wrong_rejection", "matched_null", "path_mediation", "context_probe_preservation", "manifold_proximity"},
        "abstention_in_denominator": any("abstentions remain" in item for item in protocol.get("hard_stops", [])),
        "qwen_not_authorized": any("Failure blocks Qwen3" in item for item in protocol.get("hard_stops", [])),
        "environment_present": ENVIRONMENT.exists() and "fp64_operator_fit" in read(ENVIRONMENT).get("precision", ""),
    }
    result = {"mode": "pre", "created_at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks, "passed_checks": sum(checks.values()), "total_checks": len(checks), "all_checks_passed": all(checks.values())}
    write(PREAUDIT, result)
    return result


def final_audit() -> dict[str, Any]:
    protocol = read(PROTOCOL)
    models = read_jsonl(MODELS)
    summary = read(SUMMARY)
    marker = read(COMPLETE)
    analysis = read(ANALYSIS)
    final = read(FINAL)
    per_depth = analysis["summary"]["per_depth"]
    recomputed_qualified = sum(row["behavior_qualified"] for row in models)
    recomputed_actionable = sum(row.get("selected_event_pair") is not None and row.get("context_probe_qualified") for row in models)
    recomputed_passed = sum(row["passed"] for row in models)
    expected_hashes = {
        "protocol": sha(PROTOCOL), "environment": sha(ENVIRONMENT), "material": sha(MATERIAL), "preaudit": sha(PREAUDIT),
        "models": sha(MODELS), "summary": sha(SUMMARY), "complete": sha(COMPLETE), "analysis": sha(ANALYSIS),
    }
    checks = {
        "formal_marker": marker.get("status") == "formal_run_complete",
        "six_unique_models": len(models) == len({row["model_key"] for row in models}) == protocol.get("replicates") * len(protocol.get("architectures", {})) == 6,
        "run_digest": digest(models) == summary.get("run_digest"),
        "qualified_recomputed": recomputed_qualified == analysis["summary"]["behavior_qualified"],
        "actionable_recomputed": recomputed_actionable == analysis["summary"]["camera_actionable"],
        "passed_recomputed": recomputed_passed == analysis["summary"]["passed_models"],
        "depth_counts": all(value["models"] == 2 for value in per_depth.values()),
        "confirmation_not_used_for_selection": all(row.get("selected_event_pair") is None or len(row["selected_event_pair"]) == 2 for row in models),
        "registered_gate_set": set(analysis["summary"]["gates"]) == {"G-BEHAVIOR", "G-CAMERA-BREADTH", "G-TARGET-RESCUE", "G-WRONG-REJECTION", "G-MATCHED-NULL", "G-PATH-MEDIATION", "G-CONTEXT-PROBE", "G-MANIFOLD", "G-CONTROLS", "G-BREADTH"},
        "verdict_matches_gates": analysis["passed"] == all(analysis["summary"]["gates"].values()) and final["passed"] == analysis["passed"],
        "qwen_still_denied": not final.get("authorization", {}).get("qwen3"),
        "hashes_match": final.get("artifact_hashes") == expected_hashes,
        "final_digest": final.get("final_digest") == digest({key: value for key, value in final.items() if key != "final_digest"}),
        "claim_scope_narrow": "no language or pretrained-model mechanism claim" in final.get("claim_boundary", ""),
    }
    result = {"mode": "final", "created_at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks, "passed_checks": sum(checks.values()), "total_checks": len(checks), "all_checks_passed": all(checks.values()), "recomputed": {"qualified": recomputed_qualified, "actionable": recomputed_actionable, "passed": recomputed_passed}}
    write(FINAL_AUDIT, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pre", "final"), required=True)
    args = parser.parse_args()
    result = preaudit() if args.mode == "pre" else final_audit()
    print(canonical({"mode": result["mode"], "passed": result["all_checks_passed"], "checks": f"{result['passed_checks']}/{result['total_checks']}"}))
    raise SystemExit(0 if result["all_checks_passed"] else 1)


if __name__ == "__main__":
    main()
