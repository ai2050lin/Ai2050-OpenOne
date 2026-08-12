#!/usr/bin/env python3
"""Independent integrity and arithmetic audit for Phase1178 artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

import phase1178_dual_path_implementation_library as main
import phase1178_implementation_library as lib


AUDIT_OUTPUT = main.OUT_ROOT / "audit/independent_audit.json"


def run_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    protocol = main.read_json(main.PROTOCOL_PATH)
    clean_protocol = dict(protocol)
    stored_protocol_digest = clean_protocol.pop("protocol_digest")
    add("protocol_digest", lib.digest(clean_protocol) == stored_protocol_digest)
    add("main_script_hash", protocol["scripts"]["main_sha256"] == main.sha256_file(main.SCRIPT_PATH))
    add("library_script_hash", protocol["scripts"]["library_sha256"] == main.sha256_file(main.LIBRARY_PATH))
    add("audit_script_hash", protocol["scripts"]["audit_sha256"] == main.sha256_file(Path(__file__).resolve()))
    add("frozen_thresholds", protocol["thresholds"] == main.THRESHOLDS)
    add("frozen_neutral_set", tuple(protocol["neutral_interventions"]) == lib.NEUTRAL_INTERVENTIONS)
    add("frozen_diagnostic_set", tuple(protocol["diagnostic_interventions"]) == lib.DIAGNOSTIC_INTERVENTIONS)
    add(
        "neutral_diagnostic_disjoint",
        set(protocol["neutral_interventions"]).isdisjoint(protocol["diagnostic_interventions"]),
    )

    split_ids: dict[str, set[str]] = {}
    split_passes = {}
    for split, config in main.SPLITS.items():
        run_root = main.OUT_ROOT / f"runs/{split}"
        public = main.read_jsonl(run_root / "public_manifest.jsonl")
        truth = main.read_jsonl(run_root / "sealed_truth.jsonl")
        rescue = main.read_jsonl(run_root / "rescue_responses.jsonl")
        states_file = np.load(run_root / "public_states.npz")
        states = {key: states_file[key] for key in states_file.files}
        expected = len(config.tasks) * main.BLOCKS_PER_TASK * 4
        add(f"{split}_public_count", len(public) == expected, len(public))
        add(f"{split}_truth_count", len(truth) == expected, len(truth))
        add(f"{split}_rescue_count", len(rescue) == expected, len(rescue))
        ids = {row["system_id"] for row in public}
        add(f"{split}_unique_ids", len(ids) == expected)
        add(f"{split}_joined_ids", ids == {row["system_id"] for row in truth} == {row["system_id"] for row in rescue})
        add(f"{split}_state_ids", ids == set(states))
        add(
            f"{split}_public_schema_sealed",
            all(key not in row for row in public for key in protocol["public_schema_excludes"]),
        )
        add(
            f"{split}_exact_family_slot_balance",
            all(
                sum(row["implementation_family"] == family and row["active_slot"] == slot for row in truth)
                == expected // 4
                for family in lib.IMPLEMENTATIONS for slot in (0, 1)
            ),
        )
        add(
            f"{split}_matched_state_digest_quartets",
            all(
                len({
                    public_row["observation_digest"]
                    for public_row in public
                    if public_row["task_name"] == task.name and public_row["block"] == block
                }) == 1
                for task in config.tasks for block in range(main.BLOCKS_PER_TASK)
            ),
        )
        recomputed = main.summarize_split(split, public, truth, rescue, states)
        stored = main.read_json(run_root / "summary.json")
        stored_clean = dict(stored)
        stored_digest = stored_clean.pop("summary_digest")
        recomputed_clean = dict(recomputed)
        recomputed_digest = recomputed_clean.pop("summary_digest")
        add(f"{split}_stored_summary_digest", lib.digest(stored_clean) == stored_digest)
        add(f"{split}_summary_exact_recompute", lib.digest(recomputed_clean) == lib.digest(stored_clean))
        add(f"{split}_recomputed_digest", recomputed_digest == lib.digest(recomputed_clean))
        add(f"{split}_all_component_checks", recomputed["passed"], recomputed["checks"])
        split_ids[split] = ids
        split_passes[split] = bool(recomputed["passed"])

    add("split_ids_disjoint", split_ids["discovery"].isdisjoint(split_ids["confirmation"]))
    add(
        "split_tasks_and_moduli_disjoint",
        {task.name for task in main.SPLITS["discovery"].tasks}.isdisjoint(
            {task.name for task in main.SPLITS["confirmation"].tasks}
        ) and {task.modulus for task in main.SPLITS["discovery"].tasks}.isdisjoint(
            {task.modulus for task in main.SPLITS["confirmation"].tasks}
        ),
    )
    final = main.read_json(main.OUT_ROOT / "analysis/final.json")
    final_clean = dict(final)
    final_digest = final_clean.pop("final_digest")
    add("final_digest", lib.digest(final_clean) == final_digest)
    add("final_matches_split_gates", final["development_package_complete"] == all(split_passes.values()))
    add("scope_boundary_present", "does not establish" in final["evidence_scope"])
    add("auto_continue_false", final["auto_continue"] is False)

    payload = {
        "phase": main.PHASE,
        "audit": "independent artifact integrity, sealed/public separation, and metric recomputation",
        "check_count": len(checks),
        "passed_count": sum(row["passed"] for row in checks),
        "failed_count": sum(not row["passed"] for row in checks),
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
    }
    payload["audit_digest"] = lib.digest(payload)
    main.write_json(AUDIT_OUTPUT, payload)
    return payload


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.parse_args()
    result = run_audit()
    print(lib.canonical({
        "passed": result["passed"],
        "passed_count": result["passed_count"],
        "check_count": result["check_count"],
        "audit_digest": result["audit_digest"],
    }))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main_cli()

