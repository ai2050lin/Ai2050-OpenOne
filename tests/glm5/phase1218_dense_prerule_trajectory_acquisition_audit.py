#!/usr/bin/env python3
"""Independent preaudit and result audit for Phase 1218."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1218_dense_prerule_trajectory_acquisition as experiment  # noqa: E402
import phase1217_factorial_free_formation_clock_transfer_audit as source_audit  # noqa: E402


OUT_ROOT = experiment.OUT_ROOT
PROTOCOL_PATH = experiment.PROTOCOL_PATH
PREAUDIT_PATH = experiment.PREAUDIT_PATH
FINAL_PATH = experiment.FINAL_PATH
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def valid_embedded_digest(value: dict[str, Any], field: str) -> bool:
    clean = dict(value)
    stored = clean.pop(field, None)
    return isinstance(stored, str) and digest(clean) == stored


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def preaudit() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    checks: list[dict[str, Any]] = []
    add(checks, "protocol_digest_valid", valid_embedded_digest(protocol, "protocol_digest"))
    add(checks, "phase_exact", protocol["phase"] == 1218)
    add(checks, "main_hash_current", protocol["script_hashes"]["phase1218_main"] == sha256_file(experiment.SCRIPT))
    add(checks, "audit_hash_current", protocol["script_hashes"]["phase1218_audit"] == sha256_file(Path(__file__)))
    add(
        checks,
        "measurement_source_hash_current",
        protocol["script_hashes"]["phase1217_measurement_source"] == sha256_file(Path(experiment.core.__file__)),
    )
    add(checks, "source_gate_all_true", all(protocol["source_gate"].values()), protocol["source_gate"])
    add(checks, "run_count_exact", protocol["formal_run_count"] == 32 and protocol["runs_per_split"] == 16)
    add(checks, "replicates_exact", protocol["replicates"] == 2)
    add(
        checks,
        "full_factor_dimensions",
        all(
            len(protocol[key][split]) == 2
            for key in ("tasks", "lexicons", "architectures")
            for split in ("discovery", "confirmation")
        ),
    )
    task_maps = [
        tuple(task["source_roles"][role] for role in experiment.core.FUNCTION_QUERIES)
        for split in protocol["tasks"].values()
        for task in split
    ]
    add(checks, "task_maps_unique", len(task_maps) == len(set(task_maps)) == 4)
    lexicon_seeds = [item["seed"] for split in protocol["lexicons"].values() for item in split]
    add(checks, "lexicon_seeds_unique", len(lexicon_seeds) == len(set(lexicon_seeds)) == 4)
    add(checks, "lexicons_new_vs_1217", all(seed // 1000 == 1218 for seed in lexicon_seeds))
    dense = tuple(protocol["observation_contract"]["dense_grid"])
    anchors = tuple(protocol["observation_contract"]["anchor_grid"])
    add(checks, "dense_grid_exact", dense == experiment.OBSERVATION_STEPS)
    add(checks, "dense_grid_count", len(dense) == 65)
    add(checks, "anchor_grid_exact", anchors == experiment.ANCHOR_STEPS and len(anchors) == 25)
    add(checks, "all_anchors_observed", set(anchors).issubset(set(dense)))
    add(checks, "landmark_exact", protocol["training"]["landmark_step"] == 50)
    add(checks, "landmark_has_eleven_points", sum(step <= 50 for step in dense) == 11)
    add(checks, "horizon_unchanged", protocol["training"]["maximum_steps"] == 2400)
    add(checks, "anchor_interval_unchanged", protocol["training"]["evaluation_interval"] == 100)
    add(checks, "clock_anchor_only", protocol["observation_contract"]["clock_outcomes_use_anchor_grid_only"] is True)
    add(checks, "predictor_forbidden_here", "fitting or selecting a precursor predictor in Phase 1218" in protocol["forbidden"])
    add(checks, "checkpoint_unit_forbidden", "treating checkpoints as independent prediction samples" in protocol["forbidden"])
    add(checks, "language_claim_forbidden", "claiming pretrained-language or human-brain external validity" in protocol["forbidden"])
    add(checks, "eight_baseline_families", len(protocol["frozen_targets_for_separate_phase1219"]["baseline_families"]) == 8)
    add(checks, "six_mechanism_families", len(protocol["frozen_targets_for_separate_phase1219"]["mechanistic_candidate_families"]) == 6)
    all_passed = all(row["passed"] for row in checks)
    result = {
        "phase": 1218,
        "mode": "preaudit",
        "protocol_digest": protocol["protocol_digest"],
        "check_count": len(checks),
        "passed_count": sum(row["passed"] for row in checks),
        "all_checks_passed": all_passed,
        "checks": checks,
    }
    result["audit_digest"] = digest(result)
    write_json(PREAUDIT_PATH, result)
    if not all_passed:
        raise RuntimeError(f"preaudit failed: {[row for row in checks if not row['passed']]}")
    return result


def independent_summary(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    anchors = [row for row in trajectory if int(row["step"]) in set(experiment.ANCHOR_STEPS)]
    profiles: dict[str, Any] = {}
    for profile in experiment.core.THRESHOLD_PROFILES:
        clocks = {
            clock: source_audit.independent_onset(anchors, profile, clock)
            for clock in experiment.CLOCKS
        }
        profiles[profile] = clocks
    r_clock = profiles["primary"]["R"]
    dense_count = (
        sum(int(row["step"]) < int(r_clock["step"]) for row in trajectory)
        if r_clock["status"] == "observed"
        else sum(int(row["step"]) < experiment.TRAINING["maximum_steps"] for row in trajectory)
    )
    landmark = [row for row in trajectory if int(row["step"]) <= experiment.LANDMARK_STEP]
    return {
        "profiles": profiles,
        "dense_pre_behavior_prefix_count": int(dense_count),
        "landmark_pre_rule": bool(
            all(
                not source_audit.independent_checkpoint_gates(
                    row, experiment.core.THRESHOLD_PROFILES["primary"]
                )["R"]
                for row in landmark
            )
        ),
        "landmark_observation_count": len(landmark),
    }


def result_audit() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    pre = read_json(PREAUDIT_PATH)
    final = read_json(FINAL_PATH)
    checks: list[dict[str, Any]] = []
    add(checks, "protocol_digest_valid", valid_embedded_digest(protocol, "protocol_digest"))
    add(checks, "preaudit_digest_valid", valid_embedded_digest(pre, "audit_digest"))
    add(checks, "preaudit_passed", pre["all_checks_passed"] is True)
    add(checks, "final_digest_valid", valid_embedded_digest(final, "final_digest"))
    add(checks, "protocol_link_final", final["protocol_digest"] == protocol["protocol_digest"])
    add(checks, "manifest_count", len(final["run_manifest"]) == 32)
    add(checks, "manifest_unique", len({row["run_id"] for row in final["run_manifest"]}) == 32)

    rows_by_split: dict[str, list[dict[str, Any]]] = {"discovery": [], "confirmation": []}
    for item in final["run_manifest"]:
        path = ROOT / item["path"]
        add(checks, f"run_file_exists_{item['run_id']}", path.exists())
        add(checks, f"run_file_hash_{item['run_id']}", path.exists() and sha256_file(path) == item["sha256"])
        row = read_json(path)
        add(checks, f"metrics_digest_{item['run_id']}", valid_embedded_digest(row, "metrics_digest"))
        add(checks, f"metrics_manifest_digest_{item['run_id']}", row["metrics_digest"] == item["metrics_digest"])
        add(checks, f"protocol_link_{item['run_id']}", row["protocol_digest"] == protocol["protocol_digest"])
        add(checks, f"phase_link_{item['run_id']}", row["phase"] == 1218)
        add(
            checks,
            f"trajectory_grid_{item['run_id']}",
            [point["step"] for point in row["trajectory"]] == list(experiment.OBSERVATION_STEPS),
        )
        add(checks, f"checkpoint_count_{item['run_id']}", len(row["checkpoint_manifest"]) == 65)
        for checkpoint in row["checkpoint_manifest"]:
            checkpoint_path = ROOT / checkpoint["path"]
            add(checks, f"checkpoint_exists_{item['run_id']}_{checkpoint['step']}", checkpoint_path.exists())
            add(
                checks,
                f"checkpoint_hash_{item['run_id']}_{checkpoint['step']}",
                checkpoint_path.exists() and sha256_file(checkpoint_path) == checkpoint["sha256"],
            )
        finite = all(
            point["train_behavior"]["finite_fraction"] == 1.0
            and point["holdout_behavior"]["finite_fraction"] == 1.0
            and all(
                value is None or (isinstance(value, (int, float)) and math.isfinite(float(value)))
                for value in (point["loss"], point["gradient_norm"], point["parameter_norm"])
            )
            for point in row["trajectory"]
        )
        add(checks, f"finite_and_baselines_{item['run_id']}", finite)
        gates_recomputed = all(
            source_audit.independent_checkpoint_gates(
                point, experiment.core.THRESHOLD_PROFILES[profile]
            )
            == point["gates"][profile]
            for point in row["trajectory"]
            for profile in experiment.core.THRESHOLD_PROFILES
        )
        add(checks, f"checkpoint_gates_recompute_{item['run_id']}", gates_recomputed)
        recomputed = independent_summary(row["trajectory"])
        formation = row["formation"]
        onset_match = all(
            recomputed["profiles"][profile][clock]["status"]
            == formation["profiles"][profile]["clocks"][clock]["status"]
            and (
                recomputed["profiles"][profile][clock].get("step")
                == formation["profiles"][profile]["clocks"][clock].get("step")
            )
            for profile in experiment.core.THRESHOLD_PROFILES
            for clock in experiment.CLOCKS
        )
        add(checks, f"onsets_recompute_{item['run_id']}", onset_match)
        add(
            checks,
            f"dense_prefix_recompute_{item['run_id']}",
            recomputed["dense_pre_behavior_prefix_count"]
            == formation["dense_pre_behavior_prefix_count"],
        )
        add(
            checks,
            f"landmark_recompute_{item['run_id']}",
            recomputed["landmark_pre_rule"] == formation["landmark_pre_rule"]
            and recomputed["landmark_observation_count"] == formation["landmark_observation_count"],
        )
        rows_by_split[row["split"]].append(row)

    for split, rows in rows_by_split.items():
        add(checks, f"split_count_{split}", len(rows) == 16)
        cells: dict[tuple[int, int, int, int], int] = {}
        for row in rows:
            cell = tuple(int(row[key]) for key in ("task_index", "lexicon_index", "architecture_index", "replicate"))
            cells[cell] = cells.get(cell, 0) + 1
        add(checks, f"full_factorial_{split}", len(cells) == 16 and set(cells.values()) == {1})
        recomputed_summary = experiment.group_summary(split, rows)
        add(checks, f"summary_recompute_{split}", recomputed_summary == final["summaries"][split])

    expected_acquisition = all(final["summaries"][split]["dense_acquisition_gate"] for split in rows_by_split)
    expected_targets = []
    for target in ("formed_by_800", "formed_by_2400"):
        if all(final["summaries"][split]["prediction_target_counts"][target]["balanced"] for split in rows_by_split):
            expected_targets.append(target)
    if all(final["summaries"][split]["prediction_target_counts"]["primary_onset"]["identifiable"] for split in rows_by_split):
        expected_targets.append("primary_onset")
    expected_authorized = bool(expected_acquisition and expected_targets)
    add(checks, "acquisition_claim_recompute", final["claims"]["dense_early_data_geometry_confirmed"] == expected_acquisition)
    add(checks, "target_list_recompute", final["authorized_next"]["common_identifiable_targets"] == expected_targets)
    add(checks, "authorization_recompute", final["authorized_next"]["automatic_execution"] == expected_authorized)
    add(checks, "predictor_not_fitted", final["claims"]["precursor_predictor_fitted"] is False)
    add(checks, "anchor_only_claim", final["claims"]["clock_outcomes_recomputed_on_anchor_grid_only"] is True)
    add(checks, "semantic_not_tested", final["claims"]["semantic_mechanism"] == "not_tested")
    add(checks, "new_math_false", final["new_mathematics_required"] is False)

    all_passed = all(row["passed"] for row in checks)
    result = {
        "phase": 1218,
        "mode": "result",
        "protocol_digest": protocol["protocol_digest"],
        "final_digest": final["final_digest"],
        "check_count": len(checks),
        "passed_count": sum(row["passed"] for row in checks),
        "all_checks_passed": all_passed,
        "checks": checks,
    }
    result["audit_digest"] = digest(result)
    write_json(RESULT_AUDIT_PATH, result)
    if not all_passed:
        raise RuntimeError(f"result audit failed: {[row for row in checks if not row['passed']][:20]}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preaudit", "result"), required=True)
    args = parser.parse_args()
    value = preaudit() if args.mode == "preaudit" else result_audit()
    print(json.dumps({key: value[key] for key in ("mode", "check_count", "passed_count", "all_checks_passed", "audit_digest")}, indent=2))


if __name__ == "__main__":
    main()
