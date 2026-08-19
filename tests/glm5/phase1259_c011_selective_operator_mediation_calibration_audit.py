"""Independent artifact audit for Phase1259."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1259_c011_selective_operator_mediation_calibration"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PUBLIC = OUT / "material/public_systems.jsonl"
TRUTH = OUT / "material/private_mechanism_truth.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/system_results.jsonl"
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
    public = read_jsonl(PUBLIC)
    truth = read_jsonl(TRUTH)
    public_ids = {row["system_id"] for row in public}
    truth_ids = {row["system_id"] for row in truth}
    checks = {
        "phase_and_campaign": protocol.get("phase") == 1259 and protocol.get("campaign") == "C011",
        "known_truth_scope": protocol.get("claim_type") == "known_truth_instrument_calibration",
        "system_count": len(public) == len(truth) == protocol.get("systems") == 384,
        "id_bijection": public_ids == truth_ids and len(public_ids) == len(public),
        "public_has_no_family": all("family" not in row and "expected_camera" not in row and "seed" not in row for row in public),
        "truth_is_typed": all(row.get("expected_camera") in {"global", "conditioned", "abstain"} for row in truth),
        "material_hashes": digest(public) == protocol.get("public_digest") and digest(truth) == protocol.get("truth_digest"),
        "three_camera_candidates": set(protocol.get("camera_candidates", [])) == {"global_oblique_projector", "control_conditioned_oblique_projector", "typed_abstention"},
        "selection_is_frozen": str(protocol.get("selection_order", "")).startswith("choose global if it passes"),
        "conjunctive_gates_present": set(protocol.get("gates", [])) >= {"target_rescue", "context_preservation", "wrong_identity_rejection", "matched_null_rejection", "path_block_and_rescue", "on_manifold_confirmation"},
        "negative_controls_present": set(protocol.get("controls", [])) == {"full_state_patch", "orthogonal_projector", "random_projector", "public_truth_leak_audit"},
        "hard_stop_present": "deny free-network extrapolation" in protocol.get("stopping_rule", ""),
        "qwen_not_authorized": protocol.get("authorized_next_step", "").startswith("A pass authorizes one free-trained") and any("Qwen3" in item for item in protocol.get("forbidden_claims", [])),
        "environment_present": ENVIRONMENT.exists() and read(ENVIRONMENT).get("precision") == "float64 deterministic known-truth tensor algebra",
    }
    result = {
        "mode": "pre",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    write(PREAUDIT, result)
    return result


def final_audit() -> dict[str, Any]:
    protocol = read(PROTOCOL)
    truth = read_jsonl(TRUTH)
    rows = read_jsonl(RAW)
    summary = read(SUMMARY)
    marker = read(COMPLETE)
    analysis = read(ANALYSIS)
    final = read(FINAL)
    truth_by_id = {row["system_id"]: row for row in truth}
    expected = {row["system_id"]: row["expected_camera"] for row in truth}
    type_accuracy = sum(row["selected_camera"] == expected[row["system_id"]] for row in rows) / len(rows)
    abstention = [row for row in rows if expected[row["system_id"]] == "abstain"]
    abstention_accuracy = sum(row["selected_camera"] == "abstain" for row in abstention) / len(abstention)
    artifact_hashes = final.get("artifact_hashes", {})
    expected_hashes = {
        "protocol": sha(PROTOCOL), "environment": sha(ENVIRONMENT), "public": sha(PUBLIC), "truth": sha(TRUTH),
        "preaudit": sha(PREAUDIT), "raw": sha(RAW), "summary": sha(SUMMARY), "complete": sha(COMPLETE), "analysis": sha(ANALYSIS),
    }
    checks = {
        "formal_marker": marker.get("status") == "formal_run_complete",
        "raw_count": len(rows) == protocol.get("systems") == 384,
        "raw_digest": digest(rows) == summary.get("raw_digest"),
        "all_ids_known": len({row["system_id"] for row in rows}) == len(rows) and all(row["system_id"] in truth_by_id for row in rows),
        "type_accuracy_recomputed": abs(type_accuracy - analysis.get("camera_type_accuracy", -1)) <= 1.0e-12,
        "abstention_recomputed": abs(abstention_accuracy - analysis.get("abstention_accuracy", -1)) <= 1.0e-12,
        "row_type_flags": all(row["camera_type_correct"] == (row["selected_camera"] == expected[row["system_id"]]) for row in rows),
        "abstention_has_no_intervention_result": all(row["confirmation"] is None for row in abstention),
        "actionable_has_confirmation": all(row["confirmation"] is not None for row in rows if expected[row["system_id"]] != "abstain"),
        "all_registered_gates": set(analysis.get("gates", {})) == {"G-CAMERA-TYPE", "G-ABSTENTION", "G-TARGET-RESCUE", "G-CONTEXT-PRESERVATION", "G-WRONG-REJECTION", "G-MATCHED-NULL", "G-PATH-MEDIATION", "G-ON-MANIFOLD", "G-CONTROLS", "G-BREADTH"},
        "verdict_matches_gates": analysis.get("passed") == all(analysis.get("gates", {}).values()) and final.get("passed") == analysis.get("passed"),
        "authorization_matches_verdict": final.get("authorization", {}).get("free_transformer_cross_depth") == final.get("passed") and not final.get("authorization", {}).get("qwen3"),
        "hashes_match": artifact_hashes == expected_hashes,
        "final_digest": final.get("final_digest") == digest({key: value for key, value in final.items() if key != "final_digest"}),
        "scope_is_narrow": "No natural-network or language-mechanism claim" in final.get("claim_boundary", ""),
    }
    result = {
        "mode": "final",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "all_checks_passed": all(checks.values()),
        "recomputed": {"camera_type_accuracy": type_accuracy, "abstention_accuracy": abstention_accuracy},
    }
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
