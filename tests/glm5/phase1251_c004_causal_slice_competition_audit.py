#!/usr/bin/env python3
"""Independent protocol and result audit for Phase1251."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "tests/glm5/phase1251_c004_causal_slice_competition.py"
AUDITOR = Path(__file__).resolve()
DEPENDENCY = ROOT / "tests/glm5/phase1146_learned_composition_benchmark.py"
OUT = ROOT / "tests/glm5/result/phase1251_c004_causal_slice_competition"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_worlds.jsonl"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
RAW = OUT / "raw/run_summary.json"
ARRAYS = OUT / "raw/camera_arrays.npz"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/causal_slice_adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"

EXPECTED_COUNTS = {"discovery": 64, "selection": 32, "confirmation": 64}
EXPECTED_DEPTHS = {"shallow4": 4, "middle6": 6, "deep8": 8}
EXPECTED_REPLICATES = 2
EXPECTED_FAMILIES = {
    "source_only",
    "condition_only",
    "source_condition",
    "precut_interaction",
    "shift_early",
    "map_receiver_middle",
    "map_donor_middle",
    "boundary_early",
    "boundary_middle",
    "boundary_late",
    "boundary_final",
    "multievent_full",
    "multievent_no_boundary",
    "loo_source",
    "loo_shift",
    "loo_map_receiver",
    "loo_map_donor",
    "loo_boundary",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    output = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            output.update(chunk)
    return output.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_row(row: dict[str, Any]) -> bool:
    value = dict(row)
    stored = value.pop("row_digest", None)
    if stored != digest(value):
        return False
    receiver = row["receiver_ids"]
    target = row["target_ids"]
    null = row["null_ids"]
    if len(receiver) != 23 or len(target) != 23 or len(null) != 23:
        return False
    if row["receiver_answer"] != (row["codes"][0] + row["shift"]) % 4:
        return False
    if row["target_answer"] != (row["target_code"] + row["shift"]) % 4:
        return False
    if row["null_answer"] != row["receiver_answer"]:
        return False
    if receiver[:4] != target[:4] or receiver[:8] != null[:8]:
        return False
    return True


def preaudit() -> None:
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    groups_by_partition = {
        name: {int(row["group"]) for row in rows if row["partition"] == name}
        for name in EXPECTED_COUNTS
    }
    checks = {
        "schema_and_contract": protocol.get("phase") == 1251 and protocol.get("contract_id") == "EXP-C004-WP01-001",
        "source_hashes": protocol.get("source_hashes") == {
            "main": file_sha256(MAIN),
            "auditor": file_sha256(AUDITOR),
            "model_dependency": file_sha256(DEPENDENCY),
        },
        "row_and_group_counts": len(rows) == sum(EXPECTED_COUNTS.values()) * 8
        and {name: len(groups) for name, groups in groups_by_partition.items()} == EXPECTED_COUNTS,
        "group_partitions_disjoint": not (
            groups_by_partition["discovery"] & groups_by_partition["selection"]
            or groups_by_partition["discovery"] & groups_by_partition["confirmation"]
            or groups_by_partition["selection"] & groups_by_partition["confirmation"]
        ),
        "material_rows_valid": all(verify_row(row) for row in rows),
        "depth_breadth_and_seeds": {
            name: int(config["layers"]) for name, config in protocol.get("architectures", {}).items()
        } == EXPECTED_DEPTHS
        and protocol.get("replicates") == EXPECTED_REPLICATES
        and len(set(protocol.get("model_seeds", {}).values())) == len(EXPECTED_DEPTHS) * EXPECTED_REPLICATES,
        "camera_object_ledger": set(protocol.get("camera", {}).get("families", [])) == EXPECTED_FAMILIES
        and protocol.get("camera", {}).get("input_dimension_each") == 80,
        "sealed_confirmation_and_group_bootstrap": protocol.get("camera", {}).get("confirmation") == "sealed groups at alpha=1"
        and protocol.get("camera", {}).get("bootstrap", {}).get("unit") == "world group"
        and protocol.get("camera", {}).get("bootstrap", {}).get("replicates") == 4000,
        "typed_hard_stops": len(protocol.get("hard_stops", [])) >= 6
        and any("not a causal-path claim" in value for value in protocol.get("hard_stops", []))
        and any("No pretrained" in value for value in protocol.get("hard_stops", [])),
        "environment_recorded": ENVIRONMENT.exists() and read_json(ENVIRONMENT).get("cuda_available") is True,
    }
    report = {
        "phase": 1251,
        "audit_stage": "preaudit",
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    report["audit_digest"] = digest(report)
    write_json(PREAUDIT, report)
    print(canonical_json(report))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


def recompute_breadth(models: list[dict[str, Any]], field: str, minimum: int, per_depth_minimum: int) -> tuple[bool, dict[str, int]]:
    per_depth = {
        architecture: sum(bool(row.get(field)) for row in models if row["architecture"] == architecture)
        for architecture in EXPECTED_DEPTHS
    }
    return sum(bool(row.get(field)) for row in models) >= minimum and all(
        value >= per_depth_minimum for value in per_depth.values()
    ), per_depth


def final_audit() -> None:
    protocol = read_json(PROTOCOL)
    raw = read_json(RAW)
    complete = read_json(COMPLETE)
    analysis = read_json(ANALYSIS)
    final = read_json(FINAL)
    thresholds = protocol["thresholds"]
    models = raw["models"]
    behavior, behavior_depth = recompute_breadth(
        models, "behavior_gate", thresholds["breadth_models_min"], thresholds["breadth_per_depth_min"]
    )
    full, full_depth = recompute_breadth(
        models, "full_camera_quality", thresholds["breadth_models_min"], thresholds["breadth_per_depth_min"]
    )
    single, single_depth = recompute_breadth(
        models, "late_single_sufficient", thresholds["breadth_models_min"], thresholds["breadth_per_depth_min"]
    )
    exante, exante_depth = recompute_breadth(
        models, "exante_sufficient", thresholds["breadth_models_min"], thresholds["breadth_per_depth_min"]
    )
    distributed, distributed_depth = recompute_breadth(
        models, "distributed_predictive_advantage", thresholds["breadth_models_min"], thresholds["breadth_per_depth_min"]
    )
    gates = {
        "G-BEHAVIOR-BREADTH": behavior,
        "G-FULL-CAMERA-BREADTH": full,
        "G-LATE-SINGLE-SUFFICIENCY": single,
        "G-EXANTE-SUFFICIENCY": exante,
        "G-DISTRIBUTED-PREDICTIVE-ADVANTAGE": distributed,
    }
    if not behavior:
        verdict = "behavior_qualification_failed"
    elif not full:
        verdict = "trajectory_camera_not_reproduced"
    elif single or exante:
        verdict = "distributed_observation_necessity_rejected"
    elif distributed:
        verdict = "distributed_multievent_predictive_advantage_confirmed_not_causal"
    else:
        verdict = "causal_slice_object_competition_unresolved"
    model_structure = all(
        set(row.get("camera_families", {})) == EXPECTED_FAMILIES
        and row.get("selected_single_family") in {
            "shift_early", "map_receiver_middle", "map_donor_middle", "boundary_early", "boundary_middle", "boundary_late", "boundary_final"
        }
        and row.get("selected_exante_family") in {"condition_only", "source_condition", "precut_interaction"}
        and all(
            summary.get("effective_degrees_of_freedom", 1000) <= 81.000001
            for summary in row.get("camera_families", {}).values()
        )
        for row in models if row.get("behavior_gate")
    )
    bootstrap_structure = all(
        comparison.get("independent_groups") == EXPECTED_COUNTS["confirmation"]
        and comparison.get("bootstrap_replicates") == 4000
        for row in models if row.get("behavior_gate")
        for comparison in row.get("comparisons", {}).values()
    )
    expected_depths = {
        "behavior": behavior_depth,
        "full_camera": full_depth,
        "late_single": single_depth,
        "exante": exante_depth,
        "distributed": distributed_depth,
    }
    checks = {
        "one_shot_completion_marker": complete.get("status") == "formal_run_complete"
        and complete.get("run_digest") == raw.get("run_digest")
        and complete.get("raw_sha256") == file_sha256(RAW)
        and complete.get("arrays_sha256") == file_sha256(ARRAYS),
        "protocol_and_array_integrity": raw.get("protocol_digest") == protocol.get("protocol_digest")
        and raw.get("array_sha256") == file_sha256(ARRAYS),
        "six_frozen_models_preserved": len(models) == 6
        and {row["architecture"] for row in models} == set(EXPECTED_DEPTHS)
        and all(sum(row["architecture"] == name for row in models) == 2 for name in EXPECTED_DEPTHS),
        "no_pretrained_model": raw.get("pretrained_model_loaded") is False,
        "camera_and_effective_df_structure": model_structure,
        "group_bootstrap_structure": bootstrap_structure,
        "gates_recomputed": analysis.get("gates") == gates and final.get("gates") == gates,
        "depth_breadth_recomputed": analysis.get("per_depth") == expected_depths and final.get("per_depth") == expected_depths,
        "verdict_recomputed": analysis.get("verdict") == verdict and final.get("verdict") == verdict,
        "authorization_scope": analysis.get("authorization", {}).get("pretrained_language_model_phase") is False
        and analysis.get("authorization", {}).get("semantic_mechanism_claim") is False
        and analysis.get("authorization", {}).get("causal_path_claim") is False,
        "final_artifact_hashes": final.get("artifact_hashes", {}).get("raw") == file_sha256(RAW)
        and final.get("artifact_hashes", {}).get("arrays") == file_sha256(ARRAYS)
        and final.get("artifact_hashes", {}).get("complete") == file_sha256(COMPLETE)
        and final.get("artifact_hashes", {}).get("analysis") == file_sha256(ANALYSIS),
    }
    report = {
        "phase": 1251,
        "audit_stage": "final",
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "recomputed_verdict": verdict,
        "recomputed_gates": gates,
    }
    report["audit_digest"] = digest(report)
    write_json(FINAL_AUDIT, report)
    print(canonical_json(report))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "final"))
    args = parser.parse_args()
    if args.stage == "preaudit":
        preaudit()
    else:
        final_audit()


if __name__ == "__main__":
    main()
