#!/usr/bin/env python3
"""Independent preaudit and result audit for Phase 1217."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1217_factorial_free_formation_clock_transfer as experiment  # noqa: E402


OUT_ROOT = TEST_ROOT / "result/phase1217_factorial_free_formation_clock_transfer"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    os.replace(pending, path)


def valid_embedded_digest(value: dict[str, Any], field: str) -> bool:
    clean = dict(value)
    stored = clean.pop(field, None)
    return stored is not None and digest(clean) == stored


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def source_checks(checks: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    source_final_path = experiment.SOURCE1216 / "analysis/final.json"
    source_audit_path = experiment.SOURCE1216 / "audit/independent_audit.json"
    add(checks, "source_final_exists", source_final_path.exists())
    add(checks, "source_audit_exists", source_audit_path.exists())
    source_final = read_json(source_final_path)
    source_audit = read_json(source_audit_path)
    add(checks, "source_final_digest_valid", valid_embedded_digest(source_final, "final_digest"))
    add(checks, "source_audit_digest_valid", valid_embedded_digest(source_audit, "audit_digest"))
    add(checks, "source_final_digest_frozen", source_final["final_digest"] == experiment.EXPECTED_1216_FINAL)
    add(checks, "source_audit_digest_frozen", source_audit["audit_digest"] == experiment.EXPECTED_1216_AUDIT)
    add(checks, "source_calibration_passed", source_final["summary"]["overall_pass"] is True)
    add(
        checks,
        "source_authorized_t02",
        source_final["authorized_next"]["experiment"] == "T02_FACTORIAL_FREE_FORMATION",
    )
    add(checks, "embedded_source_gate_all_true", all(protocol["source_gate"].values()))


def preaudit() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    checks: list[dict[str, Any]] = []
    add(checks, "protocol_digest_valid", valid_embedded_digest(protocol, "protocol_digest"))
    add(checks, "phase_exact", protocol["phase"] == 1217)
    add(checks, "main_hash_current", protocol["script_hashes"]["phase1217_main"] == sha256_file(experiment.SCRIPT))
    add(checks, "audit_hash_current", protocol["script_hashes"]["phase1217_audit"] == sha256_file(Path(__file__)))
    add(
        checks,
        "material_hash_current",
        protocol["script_hashes"]["phase1213_material_source"] == sha256_file(Path(experiment.p1213.__file__)),
    )
    add(
        checks,
        "transformer_hash_current",
        protocol["script_hashes"]["tiny_transformer_source"]
        == sha256_file(TEST_ROOT / "phase1146_learned_composition_benchmark.py"),
    )
    source_checks(checks, protocol)
    add(checks, "run_count_exact", protocol["formal_run_count"] == 32)
    add(checks, "runs_per_split_exact", protocol["runs_per_split"] == 16)
    add(checks, "two_tasks_each_split", all(len(protocol["tasks"][split]) == 2 for split in ("discovery", "confirmation")))
    add(checks, "two_lexicons_each_split", all(len(protocol["lexicons"][split]) == 2 for split in ("discovery", "confirmation")))
    add(checks, "two_architectures_each_split", all(len(protocol["architectures"][split]) == 2 for split in ("discovery", "confirmation")))
    add(checks, "two_seeds", protocol["replicates"] == 2)
    task_names = [task["name"] for split in protocol["tasks"].values() for task in split]
    lexicon_names = [item["name"] for split in protocol["lexicons"].values() for item in split]
    lexicon_seeds = [item["seed"] for split in protocol["lexicons"].values() for item in split]
    architecture_names = [name for split in protocol["architectures"].values() for name in split]
    add(checks, "task_names_unique", len(task_names) == len(set(task_names)))
    add(checks, "lexicon_names_unique", len(lexicon_names) == len(set(lexicon_names)))
    add(checks, "lexicon_seeds_unique", len(lexicon_seeds) == len(set(lexicon_seeds)))
    add(checks, "architecture_names_unique", len(architecture_names) == len(set(architecture_names)))
    source_maps = [tuple(task["source_roles"][role] for role in experiment.FUNCTION_QUERIES) for split in protocol["tasks"].values() for task in split]
    add(checks, "task_rules_are_distinct", len(source_maps) == len(set(source_maps)))
    add(checks, "task_rules_are_permutations", all(sorted(mapping) == sorted(experiment.ROLES) for mapping in source_maps))
    for split in ("discovery", "confirmation"):
        for task_index, lexicon_index, architecture_index, replicate in itertools.product(range(2), repeat=4):
            condition = experiment.make_condition(split, task_index, lexicon_index)
            train, holdout = experiment.split_combinations(condition)
            add(
                checks,
                f"balanced_material_{split}_{task_index}_{lexicon_index}_{architecture_index}_{replicate}",
                len(train) == 384 and len(holdout) == 128 and not set(train).intersection(holdout),
            )
    profiles = protocol["threshold_profiles"]
    for clock in experiment.CLOCKS:
        add(
            checks,
            f"threshold_monotone_{clock}",
            profiles["lenient"][clock] <= profiles["primary"][clock] <= profiles["strict"][clock],
        )
    add(checks, "no_early_stopping", protocol["training"]["no_early_stopping"] is True)
    add(checks, "checkpoint_grid_exact", protocol["training"]["maximum_steps"] == 2400 and protocol["training"]["evaluation_interval"] == 100)
    add(checks, "full_response_frozen", protocol["frozen_analysis"]["full_response"].startswith("all scalar"))
    add(checks, "posthoc_order_forbidden", protocol["frozen_analysis"]["no_posthoc_clock_order"] is True)
    add(checks, "language_claim_forbidden", "claiming language-model external validity" in protocol["forbidden"])
    all_passed = all(row["passed"] for row in checks)
    result = {
        "phase": 1217,
        "created_at": utc_now(),
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
        raise RuntimeError("preaudit failed")
    return result


def independent_checkpoint_gates(row: dict[str, Any], profile: dict[str, float]) -> dict[str, bool]:
    d_layers = [
        layer
        for layer in row["layers"]
        if layer["initial_validation"]["combined_accuracy"] <= experiment.FIXED_CONTROLS["initial_decode_accuracy_max"]
        and layer["validation"]["combined_accuracy"] >= profile["D"]
        and layer["holdout"]["combined_accuracy"] >= profile["D"]
    ]
    e_layers = [
        layer
        for layer in d_layers
        if layer["patch"] is not None
        and layer["patch"]["same_baseline_match"] >= experiment.FIXED_CONTROLS["same_baseline_match_min"]
        and layer["patch"]["same_preservation"] >= profile["E"]
        and layer["patch"]["wrong_eligible_fraction"] >= experiment.FIXED_CONTROLS["wrong_eligible_fraction_min"]
        and layer["patch"]["wrong_transfer"] >= profile["E"]
    ]
    return {
        "R": row["metrics"]["rule_accuracy"] >= profile["R"],
        "C": row["metrics"]["minimum_correct_probability"] >= profile["C"],
        "D": bool(d_layers),
        "E": bool(e_layers),
        "U1": row["metrics"]["single_necessity"] >= profile["U1"],
        "UJ": row["metrics"]["joint_necessity"] >= profile["UJ"],
    }


def independent_onset(trajectory: list[dict[str, Any]], profile: str, clock: str) -> dict[str, Any]:
    gates = [independent_checkpoint_gates(row, experiment.THRESHOLD_PROFILES[profile])[clock] for row in trajectory]
    for index in range(len(gates) - experiment.TRAINING["required_consecutive_passes"] + 1):
        if all(gates[index : index + experiment.TRAINING["required_consecutive_passes"]]):
            tail = float(np.mean(gates[index:]))
            if gates[-1] and tail >= experiment.TRAINING["post_formation_stability_min"]:
                return {"status": "observed", "step": int(trajectory[index]["step"])}
    return {"status": "right_censored"}


def independent_factor_effect(rows: list[dict[str, Any]], factor: str, clock: str) -> dict[str, Any]:
    factors = ("task_index", "lexicon_index", "architecture_index", "replicate")
    others = tuple(value for value in factors if value != factor)
    groups: dict[tuple[int, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[tuple(int(row[value]) for value in others)][int(row[factor])] = row
    pairs = [group for group in groups.values() if set(group) == {0, 1}]
    discordant = 0
    differences = []
    for pair in pairs:
        left = pair[0]["formation"]["primary_clocks"][clock]
        right = pair[1]["formation"]["primary_clocks"][clock]
        left_step = int(left["step"]) if left["status"] == "observed" else None
        right_step = int(right["step"]) if right["status"] == "observed" else None
        discordant += int((left_step is None) != (right_step is None))
        if left_step is not None and right_step is not None:
            differences.append(right_step - left_step)
    return {
        "matched_pair_count": len(pairs),
        "status_discordant_count": discordant,
        "both_observed_count": len(differences),
        "differences": differences,
    }


def result_audit() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    preaudit_value = read_json(PREAUDIT_PATH)
    final = read_json(FINAL_PATH)
    checks: list[dict[str, Any]] = []
    add(checks, "protocol_digest_valid", valid_embedded_digest(protocol, "protocol_digest"))
    add(checks, "preaudit_digest_valid", valid_embedded_digest(preaudit_value, "audit_digest"))
    add(checks, "preaudit_passed", preaudit_value["all_checks_passed"] is True)
    add(checks, "final_digest_valid", valid_embedded_digest(final, "final_digest"))
    add(checks, "protocol_link_final", final["protocol_digest"] == protocol["protocol_digest"])
    add(checks, "manifest_count", len(final["run_manifest"]) == 32)
    add(checks, "manifest_unique", len({row["run_id"] for row in final["run_manifest"]}) == 32)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"discovery": [], "confirmation": []}
    expected_steps = list(range(0, 2401, 100))
    for item in final["run_manifest"]:
        path = ROOT / item["path"]
        add(checks, f"run_file_exists_{item['run_id']}", path.exists())
        add(checks, f"run_file_hash_{item['run_id']}", sha256_file(path) == item["sha256"])
        row = read_json(path)
        rows_by_split[row["split"]].append(row)
        add(checks, f"metrics_digest_{row['run_id']}", valid_embedded_digest(row, "metrics_digest"))
        add(checks, f"metrics_manifest_digest_{row['run_id']}", row["metrics_digest"] == item["metrics_digest"])
        add(checks, f"protocol_link_{row['run_id']}", row["protocol_digest"] == protocol["protocol_digest"])
        add(checks, f"trajectory_grid_{row['run_id']}", [value["step"] for value in row["trajectory"]] == expected_steps)
        add(checks, f"checkpoint_count_{row['run_id']}", len(row["checkpoint_manifest"]) == len(expected_steps))
        for checkpoint in row["checkpoint_manifest"]:
            checkpoint_path = ROOT / checkpoint["path"]
            add(checks, f"checkpoint_exists_{row['run_id']}_{checkpoint['step']}", checkpoint_path.exists())
            add(checks, f"checkpoint_hash_{row['run_id']}_{checkpoint['step']}", sha256_file(checkpoint_path) == checkpoint["sha256"])
        for index, trajectory in enumerate(row["trajectory"]):
            add(
                checks,
                f"layer_shape_{row['run_id']}_{index}",
                len(trajectory["layers"]) == row["config"]["layers"] + 1
                and len(trajectory["necessity"]["layers"]) == row["config"]["layers"],
            )
            add(
                checks,
                f"finite_{row['run_id']}_{index}",
                trajectory["train_behavior"]["finite_fraction"] == 1.0
                and trajectory["holdout_behavior"]["finite_fraction"] == 1.0,
            )
            add(
                checks,
                f"zero_drift_{row['run_id']}_{index}",
                trajectory["necessity"]["zero_drift_max"] <= experiment.FIXED_CONTROLS["zero_drift_max"],
                trajectory["necessity"]["zero_drift_max"],
            )
            for profile, thresholds in experiment.THRESHOLD_PROFILES.items():
                recomputed = independent_checkpoint_gates(trajectory, thresholds)
                add(
                    checks,
                    f"gates_{row['run_id']}_{index}_{profile}",
                    recomputed == trajectory["gates"][profile],
                )
        for profile in experiment.THRESHOLD_PROFILES:
            for clock in experiment.CLOCKS:
                expected = independent_onset(row["trajectory"], profile, clock)
                actual = row["formation"]["profiles"][profile]["clocks"][clock]
                add(
                    checks,
                    f"onset_{row['run_id']}_{profile}_{clock}",
                    expected["status"] == actual["status"]
                    and (expected["status"] != "observed" or expected["step"] == actual["step"]),
                )
        add(
            checks,
            f"threshold_stability_{row['run_id']}",
            all(
                abs(
                    row["formation"]["threshold_status_stability"][clock]
                    - np.mean(
                        [
                            row["formation"]["profiles"][name]["clocks"][clock]["status"]
                            == row["formation"]["primary_clocks"][clock]["status"]
                            for name in experiment.THRESHOLD_PROFILES
                        ]
                    )
                )
                < 1.0e-12
                for clock in experiment.CLOCKS
            ),
        )
    for split, rows in rows_by_split.items():
        add(checks, f"split_count_{split}", len(rows) == 16)
        cells = Counter(
            (row["task_index"], row["lexicon_index"], row["architecture_index"], row["replicate"])
            for row in rows
        )
        add(checks, f"full_factorial_{split}", len(cells) == 16 and set(cells.values()) == {1})
        summary = final["summaries"][split]
        for clock in experiment.CLOCKS:
            observed = sum(row["formation"]["primary_clocks"][clock]["status"] == "observed" for row in rows)
            add(checks, f"summary_count_{split}_{clock}", summary["per_clock"][clock]["observed"] == observed)
        for factor in ("task_index", "lexicon_index", "architecture_index", "replicate"):
            for clock in experiment.CLOCKS:
                expected = independent_factor_effect(rows, factor, clock)
                actual = summary["factor_effects"][factor][clock]
                add(checks, f"factor_pairs_{split}_{factor}_{clock}", expected["matched_pair_count"] == actual["matched_pair_count"] == 8)
                add(checks, f"factor_discordance_{split}_{factor}_{clock}", expected["status_discordant_count"] == actual["status_discordant_count"])
                add(checks, f"factor_differences_{split}_{factor}_{clock}", expected["differences"] == actual["signed_step_differences_level1_minus_level0"])
        behavior = summary["per_clock"]["R"]["observed"] >= experiment.GROUP_GATES["behavior_observed_per_split_min"]
        level_gate = all(
            summary["per_level"][factor][str(level)]["R_observed"]
            >= experiment.GROUP_GATES["behavior_observed_per_binary_level_min"]
            for factor in summary["per_level"]
            for level in (0, 1)
        )
        decode = summary["per_clock"]["D"]["observed"] >= experiment.GROUP_GATES["decode_observed_per_split_min"]
        interface = summary["per_clock"]["E"]["observed"] >= experiment.GROUP_GATES["interface_observed_per_split_min"]
        threshold = all(
            summary["per_clock"][clock]["threshold_status_stability_mean"]
            >= experiment.GROUP_GATES["threshold_status_stability_min"]
            for clock in experiment.CLOCKS
        )
        expected_gate = behavior and level_gate and decode and interface and threshold and summary["gates"]["all_finite"] and summary["gates"]["zero_drift"]
        add(checks, f"group_gate_{split}", summary["clock_construct_transfer_gate"] == expected_gate)
    expected_transfer = all(final["summaries"][split]["clock_construct_transfer_gate"] for split in rows_by_split)
    add(
        checks,
        "overall_transfer_claim",
        final["claims"]["known_truth_clock_construct_transferred_to_free_networks"] == expected_transfer,
    )
    add(
        checks,
        "status_consistent",
        final["status"]
        == ("factorial_free_clock_transfer_passed" if expected_transfer else "factorial_free_clock_transfer_not_confirmed"),
    )
    add(checks, "universal_order_not_claimed", final["claims"]["universal_clock_order"] == "not_claimed")
    add(checks, "pretrained_not_tested", final["claims"]["pretrained_language_external_validity"] == "not_tested")
    add(checks, "new_math_false", final["new_mathematics_required"] is False)
    all_passed = all(row["passed"] for row in checks)
    result = {
        "phase": 1217,
        "created_at": utc_now(),
        "mode": "result_audit",
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
        failed = [row for row in checks if not row["passed"]]
        raise RuntimeError(f"result audit failed: {failed[:10]}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preaudit", "result"), required=True)
    args = parser.parse_args()
    value = preaudit() if args.mode == "preaudit" else result_audit()
    print(json.dumps({key: value[key] for key in ("mode", "check_count", "passed_count", "all_checks_passed", "audit_digest")}, indent=2))


if __name__ == "__main__":
    main()
