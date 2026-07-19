#!/usr/bin/env python3
"""CPU-only feasibility audit for complementary Phase 980 rescue gates.

This module authenticates the completed Phase 979 natural diagnostic, checks
that its terminal partition is exhaustive, and tests the arithmetic
feasibility of two *future* N=256 confirmation gates.  It does not load model
weights, generate trajectories, open a holdout, or authorize mechanism work.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
GLM5 = ROOT / "tests" / "glm5"
if str(GLM5) not in sys.path:
    sys.path.insert(0, str(GLM5))

import phase979_boundary_core as core  # noqa: E402


PHASE = 980
SOURCE_PHASE = 979
SCHEMA_VERSION = 1

SOURCE_DIR = ROOT / "tests" / "glm5" / "result" / "phase979_three_boundary_factorial"
PROTOCOL_PATH = SOURCE_DIR / "protocol_preregistration.json"
MANIFEST_PATH = SOURCE_DIR / "manifest_natural.json"
AUDIT_PATH = SOURCE_DIR / "audit_natural.json"
STATUS_PATH = SOURCE_DIR / "generator_status_natural.json"
ROWS_PATH = SOURCE_DIR / "rows_natural.jsonl"

OUT_DIR = ROOT / "tests" / "glm5" / "result" / "phase980_rescue_gate_design"
REPORT_PATH = OUT_DIR / "feasibility_report.json"

SOURCE_N = 128
FUTURE_N = 256
TASK_N = 32
FUTURE_STREAMS = ("stream_0", "stream_1", "stream_2")
DIFFICULTY_STRATA = ("easy", "hard")
TASKS = (
    "boolean_logic",
    "constraint_order",
    "modular_arithmetic",
    "multistep_arithmetic",
    "relation_path",
    "sequence_rule",
    "state_machine",
    "string_transform",
)

CENSORED_STATES = {
    "CENSORED_BEFORE_VALID_CLOSE",
    "CENSORED_AFTER_FINAL_START_NO_ANSWER",
    "CENSORED_AFTER_ANSWER_OBSERVED",
}
INVALID_STATES = {"EOS_INVALID_MODE", "EOS_INVALID_SEMANTIC"}
VALID_STATES = {"VALID_STOP"}
ALL_STATES = CENSORED_STATES | INVALID_STATES | VALID_STATES

GATE_THRESHOLDS = {
    "N_per_stream": FUTURE_N,
    "task_n_per_stream": TASK_N,
    "frozen_streams": list(FUTURE_STREAMS),
    "valid_stop_improvement_min": 26,
    "target_reservoir_reduction_min": 26,
    "non_target_increase_max": 12,
    "task_valid_stop_improvement_min": 3,
    "tasks_passing_min": 6,
    "task_denominator": 8,
    "per_task_valid_stop_regression_floor": -2,
    "difficulty_strata": list(DIFFICULTY_STRATA),
    "difficulty_stratum_valid_stop_improvement_min": 0,
    "task_rule": (
        "at_least_6 identical tasks must have delta_V>=3 in the intersection "
        "across all 3 streams"
    ),
    "route_rule": (
        "(all3_semantic_reservoir_route) OR (all3_censor_reservoir_route); "
        "per-stream route mixing is forbidden"
    ),
}


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def assert_no_holdout_import() -> None:
    loaded = [
        name for name in sys.modules
        if name == "phase977_holdout_dataset"
        or name.endswith(".phase977_holdout_dataset")
    ]
    require(not loaded, f"forbidden old holdout module imported: {loaded}")


def verify_self_hash(
    document: dict[str, Any], hash_field: str, time_field: str, label: str,
) -> None:
    payload = core.without_fields(document, hash_field, time_field)
    require(
        document.get(hash_field) == core.sha256_json(payload),
        f"{label} self-hash invalid",
    )


def authenticate_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the Phase 979 chain and return protocol plus audit."""
    assert_no_holdout_import()
    protocol = core.load_json(PROTOCOL_PATH, "Phase979 protocol")
    verify_self_hash(
        protocol, "protocol_sha256", "created_at_utc", "Phase979 protocol"
    )
    require(protocol.get("phase") == SOURCE_PHASE, "wrong source protocol phase")
    require(protocol.get("expected_natural_rows") == 2048, "source row count changed")
    require(protocol.get("decision_checkpoint") == 2048, "decision checkpoint changed")
    require(protocol.get("replicates") == [0, 1], "source replicate contract changed")
    require(
        protocol.get("holdout_loaded") is False
        and protocol.get("mechanism_authorized") is False,
        "Phase979 protocol crossed a forbidden boundary",
    )
    natural_contract = protocol.get("natural_contract", {})
    require(
        natural_contract.get("holdout_loaded") is False
        and natural_contract.get("mechanism_authorized") is False,
        "Phase979 natural contract crossed a forbidden boundary",
    )
    commitments = protocol.get("phase978_commitments", {})
    require(
        commitments.get("development_gate_passed") is False
        and commitments.get("holdout_authorized") is False
        and commitments.get("holdout_loaded") is False
        and commitments.get("mechanism_authorized") is False,
        "Phase978 NO-GO lineage is not intact",
    )
    script_seals = protocol.get("phase979_script_hashes", {})
    require(isinstance(script_seals, dict) and script_seals, "missing source script seals")
    for label, seal in script_seals.items():
        require(isinstance(seal, dict), f"invalid source script seal: {label}")
        path = ROOT / str(seal.get("path", ""))
        require(
            path.is_file() and core.sha256_file(path) == seal.get("sha256"),
            f"sealed Phase979 script changed: {label}",
        )

    manifest = core.load_json(MANIFEST_PATH, "Phase979 natural manifest")
    verify_self_hash(
        manifest, "manifest_sha256", "created_at_utc", "Phase979 natural manifest"
    )
    require(manifest.get("phase") == SOURCE_PHASE, "wrong natural manifest phase")
    require(
        manifest.get("protocol_sha256") == protocol["protocol_sha256"],
        "manifest/protocol mismatch",
    )
    require(
        manifest.get("protocol_file_sha256") == core.sha256_file(PROTOCOL_PATH),
        "manifest protocol-file hash mismatch",
    )
    require(manifest.get("expected_rows") == 2048, "manifest row count changed")
    require(
        manifest.get("holdout_loaded") is False
        and manifest.get("mechanism_authorized") is False,
        "natural manifest crossed a forbidden boundary",
    )

    status = core.load_json(STATUS_PATH, "Phase979 natural status")
    verify_self_hash(
        status, "status_sha256", "updated_at_utc", "Phase979 natural status"
    )
    require(status.get("phase") == SOURCE_PHASE, "wrong natural status phase")
    require(
        status.get("protocol_sha256") == protocol["protocol_sha256"]
        and status.get("manifest_sha256") == manifest["manifest_sha256"],
        "status lineage mismatch",
    )
    require(
        status.get("complete") is True
        and status.get("expected_rows") == 2048
        and status.get("completed_rows") == 2048,
        "natural run is incomplete",
    )
    require(
        status.get("holdout_loaded") is False
        and status.get("mechanism_authorized") is False,
        "natural status crossed a forbidden boundary",
    )

    audit = core.load_json(AUDIT_PATH, "Phase979 natural audit")
    verify_self_hash(audit, "audit_sha256", "audited_at_utc", "Phase979 natural audit")
    require(audit.get("phase") == SOURCE_PHASE, "wrong natural audit phase")
    require(
        audit.get("protocol_sha256") == protocol["protocol_sha256"]
        and audit.get("manifest_sha256") == manifest["manifest_sha256"],
        "audit lineage mismatch",
    )
    require(
        audit.get("holdout_loaded") is False
        and audit.get("phase977_holdout_authorized") is False
        and audit.get("mechanism_authorized") is False,
        "natural audit crossed a forbidden boundary",
    )
    require(
        audit.get("passed_candidate_screens") == []
        and audit.get("new_independent_confirmation_candidate_exists") is False,
        "P is not empty in the source natural audit",
    )
    require(
        all(
            screen.get("passed") is False
            for screen in audit.get("candidate_effect_screens", {}).values()
        ),
        "source candidate screen detail contradicts P=empty",
    )

    row_chain = authenticate_rows(protocol, manifest, status, audit)
    evidence = {
        "protocol": {
            "path": relative(PROTOCOL_PATH),
            "file_sha256": core.sha256_file(PROTOCOL_PATH),
            "protocol_sha256": protocol["protocol_sha256"],
            "self_hash_valid": True,
            "all_phase979_script_seals_valid": True,
        },
        "manifest": {
            "path": relative(MANIFEST_PATH),
            "file_sha256": core.sha256_file(MANIFEST_PATH),
            "manifest_sha256": manifest["manifest_sha256"],
            "self_hash_valid": True,
        },
        "natural_status": {
            "path": relative(STATUS_PATH),
            "file_sha256": core.sha256_file(STATUS_PATH),
            "status_sha256": status["status_sha256"],
            "self_hash_valid": True,
            "complete": True,
        },
        "natural_audit": {
            "path": relative(AUDIT_PATH),
            "file_sha256": core.sha256_file(AUDIT_PATH),
            "audit_sha256": audit["audit_sha256"],
            "self_hash_valid": True,
        },
        "natural_rows": row_chain,
        "lineage_consistent": True,
        "holdout": False,
        "holdout_loaded": False,
        "holdout_authorized": False,
        "mechanism": False,
        "mechanism_authorized": False,
    }
    assert_no_holdout_import()
    return evidence, audit


def authenticate_rows(
    protocol: dict[str, Any], manifest: dict[str, Any],
    status: dict[str, Any], audit: dict[str, Any],
) -> dict[str, Any]:
    """Verify file/row hashes and independently rebuild the 2048 partition."""
    require(ROWS_PATH.is_file(), f"missing Phase979 natural rows: {ROWS_PATH}")
    payload = ROWS_PATH.read_bytes()
    require(payload.endswith(b"\n"), "natural rows lack a final newline")
    rows_file_sha256 = core.sha256_file(ROWS_PATH)
    require(
        rows_file_sha256 == audit.get("rows_file_sha256")
        == audit.get("row_audit", {}).get("rows_file_sha256"),
        "rows file hash does not match the authenticated natural audit",
    )

    controls = tuple(protocol.get("controls", {}).keys())
    decodings = tuple(protocol.get("decoding_policies", {}).keys())
    replicates = tuple(int(value) for value in protocol.get("replicates", []))
    expected_cells = {
        (control, decoding, replicate)
        for control in controls for decoding in decodings for replicate in replicates
    }
    require(len(expected_cells) == 16, "source cell/stream count is not 16")

    counts: dict[tuple[str, str, int], Counter[str]] = defaultdict(Counter)
    by_task: dict[tuple[str, str, int], dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    seen_keys: set[tuple[str, str, str, int]] = set()
    n_rows = 0
    for line_number, raw in enumerate(payload.splitlines(), 1):
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"malformed natural row {line_number}") from exc
        require(isinstance(row, dict), f"natural row {line_number} is not an object")
        require(
            row.get("row_sha256")
            == core.sha256_json(core.without_fields(row, "row_sha256")),
            f"natural row self-hash mismatch at line {line_number}",
        )
        require(
            row.get("protocol_sha256") == protocol["protocol_sha256"]
            and row.get("manifest_sha256") == manifest["manifest_sha256"],
            f"natural row lineage mismatch at line {line_number}",
        )
        require(
            row.get("holdout_loaded") is False
            and row.get("mechanism_authorized") is False,
            f"natural row crossed a forbidden boundary at line {line_number}",
        )
        key = (
            str(row.get("id")), str(row.get("control_policy")),
            str(row.get("decoding_policy")), int(row.get("replicate", -1)),
        )
        require(key not in seen_keys, f"duplicate natural row key: {key}")
        seen_keys.add(key)
        cell = (key[1], key[2], key[3])
        require(cell in expected_cells, f"unexpected natural cell: {cell}")
        task = str(row.get("task"))
        require(task in TASKS, f"unexpected source task: {task}")
        snapshot = row.get("checkpoints", {}).get("2048", {})
        state = str(snapshot.get("terminal_state"))
        require(state in ALL_STATES, f"unknown terminal state at line {line_number}: {state}")
        counts[cell][state] += 1
        by_task[cell][task][state] += 1
        n_rows += 1

    require(n_rows == 2048 and len(seen_keys) == 2048, "natural row denominator changed")
    status_counts = status.get("completed_by_cell_replicate", {})
    audit_checkpoint = audit.get("checkpoints", {}).get("2048", {})
    cells: dict[str, Any] = {}
    for control, decoding, replicate in sorted(expected_cells):
        cell = (control, decoding, replicate)
        counter = counts[cell]
        v = sum(counter[state] for state in VALID_STATES)
        c = sum(counter[state] for state in CENSORED_STATES)
        i = sum(counter[state] for state in INVALID_STATES)
        n = v + c + i
        require(n == SOURCE_N, f"V+C+I != N for source cell {cell}")
        status_key = f"{control}|{decoding}|r{replicate}"
        require(status_counts.get(status_key) == SOURCE_N, f"status count mismatch: {cell}")
        source_summary = audit_checkpoint[control][decoding][str(replicate)]["overall"]
        require(
            source_summary.get("n") == n
            and source_summary.get("valid_stop_n") == v
            and source_summary.get("censored_n") == c
            and source_summary.get("eos_invalid_n") == i,
            f"source audit partition mismatch: {cell}",
        )
        task_partition: dict[str, Any] = {}
        for task in TASKS:
            task_counter = by_task[cell][task]
            tv = sum(task_counter[state] for state in VALID_STATES)
            tc = sum(task_counter[state] for state in CENSORED_STATES)
            ti = sum(task_counter[state] for state in INVALID_STATES)
            require(tv + tc + ti == 16, f"source task denominator mismatch: {cell}/{task}")
            task_partition[task] = {"V": tv, "C": tc, "I": ti, "N": 16}
        cells[status_key] = {
            "V": v, "C": c, "I": i, "N": n,
            "V_plus_C_plus_I_equals_N": True,
            "by_task": task_partition,
        }

    return {
        "path": relative(ROWS_PATH),
        "file_sha256": rows_file_sha256,
        "file_hash_matches_natural_audit": True,
        "row_self_hashes_verified": n_rows,
        "unique_rows_verified": n_rows,
        "all_row_lineages_valid": True,
        "all_row_firewalls_valid": True,
        "decision_checkpoint": 2048,
        "partition_definition": {
            "V": sorted(VALID_STATES),
            "C": sorted(CENSORED_STATES),
            "I": sorted(INVALID_STATES),
        },
        "source_N_per_cell_stream": SOURCE_N,
        "cell_stream_count": len(cells),
        "cells": cells,
        "all_cells_V_plus_C_plus_I_equals_N": True,
        "P_definition": "set of Phase979 passed candidate-effect screens",
        "P": [],
        "P_is_empty": True,
    }


def split_difficulty_margins(overall: Counter[str]) -> dict[str, dict[str, int]]:
    """Make two N=128 margins without implying a task/difficulty joint table."""
    easy = {key: int(overall[key] // 2) for key in ("V", "C", "I")}
    slots = FUTURE_N // 2 - sum(easy.values())
    for key in ("V", "C", "I"):
        if slots and overall[key] % 2:
            easy[key] += 1
            slots -= 1
    require(slots == 0 and sum(easy.values()) == FUTURE_N // 2,
            "could not construct difficulty margin")
    hard = {key: int(overall[key] - easy[key]) for key in ("V", "C", "I")}
    require(sum(hard.values()) == FUTURE_N // 2, "hard margin denominator changed")
    return {"easy": easy, "hard": hard}


def arm_from_tasks(task_counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    require(set(task_counts) == set(TASKS), "future arm task set changed")
    overall = Counter()
    normalized: dict[str, dict[str, int]] = {}
    for task in TASKS:
        values = task_counts[task]
        require(set(values) == {"V", "C", "I"}, f"invalid count fields for {task}")
        require(
            all(isinstance(values[key], int) and not isinstance(values[key], bool)
                and values[key] >= 0 for key in ("V", "C", "I")),
            f"invalid future count for {task}",
        )
        require(sum(values.values()) == TASK_N, f"future task denominator changed: {task}")
        normalized[task] = {key: int(values[key]) for key in ("V", "C", "I")}
        overall.update(normalized[task])
    require(sum(overall.values()) == FUTURE_N, "future arm denominator changed")
    return {
        "V": overall["V"], "C": overall["C"], "I": overall["I"],
        "N": FUTURE_N, "by_task": normalized,
        "by_difficulty": split_difficulty_margins(overall),
    }


def make_uniform_arm(v: int, c: int, i: int) -> dict[str, Any]:
    require(v + c + i == TASK_N, "uniform task arm does not sum to 32")
    return arm_from_tasks({task: {"V": v, "C": c, "I": i} for task in TASKS})


def candidate_from_deltas(
    baseline: dict[str, Any], delta_v: list[int], delta_c: list[int],
) -> dict[str, Any]:
    require(len(delta_v) == len(delta_c) == len(TASKS), "delta vector length changed")
    tasks: dict[str, dict[str, int]] = {}
    for index, task in enumerate(TASKS):
        base = baseline["by_task"][task]
        v = base["V"] + int(delta_v[index])
        c = base["C"] + int(delta_c[index])
        i = TASK_N - v - c
        tasks[task] = {"V": v, "C": c, "I": i}
    return arm_from_tasks(tasks)


def with_difficulty_margins(
    arm: dict[str, Any], easy: dict[str, int], hard: dict[str, int],
) -> dict[str, Any]:
    """Return an arm with explicit, aggregate-consistent easy/hard margins."""
    result = {
        **arm,
        "by_task": {task: dict(values) for task, values in arm["by_task"].items()},
        "by_difficulty": {"easy": dict(easy), "hard": dict(hard)},
    }
    for stratum in DIFFICULTY_STRATA:
        values = result["by_difficulty"][stratum]
        require(set(values) == {"V", "C", "I"}, f"invalid {stratum} margin")
        require(all(isinstance(value, int) and value >= 0 for value in values.values()),
                f"invalid {stratum} count")
        require(sum(values.values()) == FUTURE_N // 2,
                f"{stratum} margin denominator changed")
    for key in ("V", "C", "I"):
        require(
            sum(result["by_difficulty"][stratum][key]
                for stratum in DIFFICULTY_STRATA) == result[key],
            f"difficulty margins do not recover overall {key}",
        )
    return result


def validate_future_arm(arm: dict[str, Any]) -> None:
    require(arm["V"] + arm["C"] + arm["I"] == FUTURE_N, "invalid arm partition")
    require(set(arm["by_task"]) == set(TASKS), "invalid future task margin")
    for task in TASKS:
        require(sum(arm["by_task"][task].values()) == TASK_N,
                f"invalid task partition: {task}")
    require(set(arm["by_difficulty"]) == set(DIFFICULTY_STRATA),
            "invalid difficulty strata")
    for stratum in DIFFICULTY_STRATA:
        values = arm["by_difficulty"][stratum]
        require(sum(values.values()) == FUTURE_N // 2,
                f"invalid difficulty denominator: {stratum}")
    for key in ("V", "C", "I"):
        require(sum(arm["by_difficulty"][value][key] for value in DIFFICULTY_STRATA)
                == arm[key], f"difficulty aggregate mismatch: {key}")


def gate_stream(
    baseline: dict[str, Any], candidate: dict[str, Any], gate: str,
) -> dict[str, Any]:
    require(gate in {"semantic_rescue", "censor_rescue"}, f"unknown gate: {gate}")
    for arm in (baseline, candidate):
        validate_future_arm(arm)
    delta_v = candidate["V"] - baseline["V"]
    rescue_c = baseline["C"] - candidate["C"]
    rescue_i = baseline["I"] - candidate["I"]
    delta_c = candidate["C"] - baseline["C"]
    delta_i = candidate["I"] - baseline["I"]
    task_deltas = {
        task: candidate["by_task"][task]["V"] - baseline["by_task"][task]["V"]
        for task in TASKS
    }
    qualifying_tasks = sorted(task for task, value in task_deltas.items() if value >= 3)
    tasks_passing = len(qualifying_tasks)
    difficulty_deltas = {
        stratum: (
            candidate["by_difficulty"][stratum]["V"]
            - baseline["by_difficulty"][stratum]["V"]
        )
        for stratum in DIFFICULTY_STRATA
    }
    common = {
        "delta_V_at_least_26": delta_v >= 26,
        "at_least_6_of_8_tasks_have_delta_V_at_least_3": tasks_passing >= 6,
        "all_8_tasks_have_delta_V_at_least_minus_2": min(task_deltas.values()) >= -2,
        "easy_delta_V_at_least_0": difficulty_deltas["easy"] >= 0,
        "hard_delta_V_at_least_0": difficulty_deltas["hard"] >= 0,
    }
    if gate == "semantic_rescue":
        checks = {
            **common,
            "R_I_at_least_26": rescue_i >= 26,
            "C_increase_at_most_12": delta_c <= 12,
        }
    else:
        checks = {
            **common,
            "R_C_at_least_26": rescue_c >= 26,
            "I_increase_at_most_12": delta_i <= 12,
        }
    return {
        "gate": gate,
        "baseline": {key: baseline[key] for key in ("V", "C", "I", "N")},
        "candidate": {key: candidate[key] for key in ("V", "C", "I", "N")},
        "delta_V": delta_v,
        "R_C": rescue_c,
        "R_I": rescue_i,
        "R_C_definition": "marginal censored-reservoir reduction C_baseline-C_candidate",
        "R_I_definition": "marginal invalid-reservoir reduction I_baseline-I_candidate",
        "not_item_level_rescue": True,
        "C_increase": delta_c,
        "I_increase": delta_i,
        "conservation_identity": "delta_V=R_C+R_I",
        "conservation_identity_verified": delta_v == rescue_c + rescue_i,
        "by_task_delta_V": task_deltas,
        "qualifying_tasks_delta_V_at_least_3": qualifying_tasks,
        "tasks_passing": tasks_passing,
        "by_difficulty_delta_V": difficulty_deltas,
        "checks": checks,
        "passed": all(checks.values()),
    }


def gate_three_streams(
    stream_pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]], gate: str,
) -> dict[str, Any]:
    require(set(stream_pairs) == set(FUTURE_STREAMS), "future gate requires exactly 3 streams")
    results = {
        stream: gate_stream(*stream_pairs[stream], gate)
        for stream in FUTURE_STREAMS
    }
    common_tasks = set(TASKS)
    for result in results.values():
        common_tasks &= set(result["qualifying_tasks_delta_V_at_least_3"])
    common_tasks_sorted = sorted(common_tasks)
    individual_pass = all(result["passed"] for result in results.values())
    common_task_pass = len(common_tasks_sorted) >= 6
    return {
        "gate": gate,
        "stream_results": results,
        "all_three_streams_pass_individually": individual_pass,
        "common_qualifying_task_intersection": common_tasks_sorted,
        "common_qualifying_task_count": len(common_tasks_sorted),
        "common_task_intersection_at_least_6": common_task_pass,
        "all_three_streams_pass": individual_pass and common_task_pass,
        "combination": "stream_0 AND stream_1 AND stream_2 AND common-task-intersection>=6",
    }


def overall_gate(
    stream_pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    """Apply route-level OR only after a single route passes all three streams."""
    semantic = gate_three_streams(stream_pairs, "semantic_rescue")
    censor = gate_three_streams(stream_pairs, "censor_rescue")
    forbidden_mixed_route_value = all(
        semantic["stream_results"][stream]["passed"]
        or censor["stream_results"][stream]["passed"]
        for stream in FUTURE_STREAMS
    )
    passed = semantic["all_three_streams_pass"] or censor["all_three_streams_pass"]
    return {
        "semantic_route": semantic,
        "censor_route": censor,
        "passed": passed,
        "official_formula": "(all3_semantic) OR (all3_censor)",
        "forbidden_formula": "all3(each_stream_semantic_OR_censor)",
        "forbidden_mixed_route_value_for_diagnostic_only": forbidden_mixed_route_value,
        "mixed_routes_allowed": False,
    }


def run_feasibility_self_tests() -> dict[str, Any]:
    """Exercise boundary witnesses and deliberately failing counterexamples."""
    delta_26 = [4, 4, 4, 4, 3, 3, 2, 2]
    delta_25 = [4, 4, 4, 4, 3, 3, 2, 1]
    only_five_tasks = [5, 5, 5, 5, 4, 1, 1, 0]
    different_six_tasks = [4, 4, 4, 4, 2, 2, 3, 3]
    one_task_minus_3 = [5, 5, 5, 5, 4, 4, -3, 1]
    nuisance_12 = [2, 2, 2, 2, 1, 1, 1, 1]
    nuisance_13 = [2, 2, 2, 2, 2, 1, 1, 1]
    zeros = [0] * len(TASKS)

    # C_b=0: semantic rescue can sit exactly on both 26-count boundaries.
    semantic_baseline = make_uniform_arm(16, 0, 16)
    semantic_candidate = candidate_from_deltas(semantic_baseline, delta_26, zeros)
    semantic_witness = gate_stream(
        semantic_baseline, semantic_candidate, "semantic_rescue"
    )
    require(semantic_witness["passed"], "semantic boundary witness did not pass")
    require(
        semantic_witness["delta_V"] == semantic_witness["R_I"] == 26
        and semantic_witness["C_increase"] == 0
        and semantic_witness["tasks_passing"] == 6,
        "semantic witness is not on the intended boundary",
    )
    semantic_as_censor = gate_stream(
        semantic_baseline, semantic_candidate, "censor_rescue"
    )
    require(not semantic_as_censor["passed"], "C_b=0 unexpectedly passed censor rescue")
    require(
        semantic_baseline["C"] == 0 and semantic_as_censor["R_C"] <= 0,
        "C_b=0 reachability proof failed",
    )

    # I_b=0: censor rescue can sit exactly on both 26-count boundaries.
    censor_baseline = make_uniform_arm(16, 16, 0)
    censor_candidate = candidate_from_deltas(censor_baseline, delta_26, [-x for x in delta_26])
    censor_witness = gate_stream(censor_baseline, censor_candidate, "censor_rescue")
    require(censor_witness["passed"], "censor boundary witness did not pass")
    require(
        censor_witness["delta_V"] == censor_witness["R_C"] == 26
        and censor_witness["I_increase"] == 0
        and censor_witness["tasks_passing"] == 6,
        "censor witness is not on the intended boundary",
    )
    censor_as_semantic = gate_stream(
        censor_baseline, censor_candidate, "semantic_rescue"
    )
    require(not censor_as_semantic["passed"], "I_b=0 unexpectedly passed semantic rescue")
    require(
        censor_baseline["I"] == 0 and censor_as_semantic["R_I"] <= 0,
        "I_b=0 reachability proof failed",
    )

    # Counterexamples just across each decision boundary.
    semantic_delta_v_fail = gate_stream(
        semantic_baseline,
        candidate_from_deltas(semantic_baseline, delta_25, [1] + [0] * 7),
        "semantic_rescue",
    )
    require(
        not semantic_delta_v_fail["passed"]
        and semantic_delta_v_fail["delta_V"] == 25
        and semantic_delta_v_fail["R_I"] == 26,
        "semantic delta-V counterexample failed",
    )
    semantic_ri_baseline = make_uniform_arm(15, 1, 16)
    semantic_ri_fail = gate_stream(
        semantic_ri_baseline,
        candidate_from_deltas(semantic_ri_baseline, delta_26, [-1] + [0] * 7),
        "semantic_rescue",
    )
    require(
        not semantic_ri_fail["passed"] and semantic_ri_fail["R_I"] == 25,
        "semantic R_I counterexample failed",
    )
    semantic_cap_baseline = make_uniform_arm(14, 2, 16)
    semantic_cap_fail = gate_stream(
        semantic_cap_baseline,
        candidate_from_deltas(semantic_cap_baseline, delta_26, nuisance_13),
        "semantic_rescue",
    )
    require(
        not semantic_cap_fail["passed"] and semantic_cap_fail["C_increase"] == 13,
        "semantic nuisance-cap counterexample failed",
    )
    semantic_task_fail = gate_stream(
        semantic_baseline,
        candidate_from_deltas(semantic_baseline, only_five_tasks, zeros),
        "semantic_rescue",
    )
    require(
        not semantic_task_fail["passed"]
        and semantic_task_fail["delta_V"] == 26
        and semantic_task_fail["tasks_passing"] == 5,
        "semantic task-spread counterexample failed",
    )

    censor_delta_v_fail = gate_stream(
        censor_baseline,
        candidate_from_deltas(
            censor_baseline,
            delta_25,
            [
                -(delta_25[index] + (1 if index == 0 else 0))
                for index in range(8)
            ],
        ),
        "censor_rescue",
    )
    require(
        not censor_delta_v_fail["passed"]
        and censor_delta_v_fail["delta_V"] == 25
        and censor_delta_v_fail["R_C"] == 26,
        "censor delta-V counterexample failed",
    )
    censor_rc_baseline = make_uniform_arm(15, 16, 1)
    censor_rc_fail = gate_stream(
        censor_rc_baseline,
        candidate_from_deltas(censor_rc_baseline, delta_26, [-3, -4, -4, -4, -3, -3, -2, -2]),
        "censor_rescue",
    )
    require(
        not censor_rc_fail["passed"] and censor_rc_fail["R_C"] == 25,
        "censor R_C counterexample failed",
    )
    censor_cap_baseline = make_uniform_arm(14, 16, 2)
    censor_cap_fail = gate_stream(
        censor_cap_baseline,
        candidate_from_deltas(
            censor_cap_baseline, delta_26,
            [-(delta_26[index] + nuisance_13[index]) for index in range(8)],
        ),
        "censor_rescue",
    )
    require(
        not censor_cap_fail["passed"] and censor_cap_fail["I_increase"] == 13,
        "censor nuisance-cap counterexample failed",
    )
    censor_task_fail = gate_stream(
        censor_baseline,
        candidate_from_deltas(censor_baseline, only_five_tasks, [-x for x in only_five_tasks]),
        "censor_rescue",
    )
    require(
        not censor_task_fail["passed"]
        and censor_task_fail["delta_V"] == 26
        and censor_task_fail["tasks_passing"] == 5,
        "censor task-spread counterexample failed",
    )

    # Explicitly exercise the nuisance cap at its inclusive boundary.
    semantic_cap_witness = gate_stream(
        semantic_cap_baseline,
        candidate_from_deltas(semantic_cap_baseline, delta_26, nuisance_12),
        "semantic_rescue",
    )
    require(
        semantic_cap_witness["passed"] and semantic_cap_witness["C_increase"] == 12,
        "semantic inclusive nuisance boundary failed",
    )
    censor_cap_witness = gate_stream(
        censor_cap_baseline,
        candidate_from_deltas(
            censor_cap_baseline, delta_26,
            [-(delta_26[index] + nuisance_12[index]) for index in range(8)],
        ),
        "censor_rescue",
    )
    require(
        censor_cap_witness["passed"] and censor_cap_witness["I_increase"] == 12,
        "censor inclusive nuisance boundary failed",
    )

    semantic_streams = {
        stream: (semantic_baseline, semantic_candidate) for stream in FUTURE_STREAMS
    }
    censor_streams = {
        stream: (censor_baseline, censor_candidate) for stream in FUTURE_STREAMS
    }
    semantic_three = gate_three_streams(semantic_streams, "semantic_rescue")
    censor_three = gate_three_streams(censor_streams, "censor_rescue")
    require(
        semantic_three["all_three_streams_pass"]
        and censor_three["all_three_streams_pass"],
        "three-stream boundary witness did not pass",
    )
    one_stream_fails = dict(semantic_streams)
    one_stream_fails["stream_2"] = (
        semantic_baseline,
        candidate_from_deltas(semantic_baseline, delta_25, [1] + [0] * 7),
    )
    and_counterexample = gate_three_streams(one_stream_fails, "semantic_rescue")
    require(
        not and_counterexample["all_three_streams_pass"]
        and [
            and_counterexample["stream_results"][stream]["passed"]
            for stream in FUTURE_STREAMS
        ] == [True, True, False],
        "three-stream AND counterexample failed",
    )

    # Each stream can have six qualifying tasks while their common set is too small.
    different_task_sets = {
        "stream_0": (semantic_baseline, semantic_candidate),
        "stream_1": (
            semantic_baseline,
            candidate_from_deltas(semantic_baseline, different_six_tasks, zeros),
        ),
        "stream_2": (semantic_baseline, semantic_candidate),
    }
    task_intersection_counterexample = gate_three_streams(
        different_task_sets, "semantic_rescue"
    )
    require(
        task_intersection_counterexample["all_three_streams_pass_individually"]
        and task_intersection_counterexample["common_qualifying_task_count"] == 4
        and not task_intersection_counterexample["all_three_streams_pass"],
        "common-task-intersection counterexample failed",
    )

    task_regression_counterexample = gate_stream(
        semantic_baseline,
        candidate_from_deltas(semantic_baseline, one_task_minus_3, zeros),
        "semantic_rescue",
    )
    require(
        task_regression_counterexample["delta_V"] == 26
        and task_regression_counterexample["tasks_passing"] == 6
        and min(task_regression_counterexample["by_task_delta_V"].values()) == -3
        and not task_regression_counterexample["passed"],
        "per-task regression-floor counterexample failed",
    )

    difficulty_regression_candidate = with_difficulty_margins(
        semantic_candidate,
        easy={"V": 63, "C": 0, "I": 65},
        hard={"V": 91, "C": 0, "I": 37},
    )
    difficulty_regression_counterexample = gate_stream(
        semantic_baseline, difficulty_regression_candidate, "semantic_rescue"
    )
    require(
        difficulty_regression_counterexample["delta_V"] == 26
        and difficulty_regression_counterexample["by_difficulty_delta_V"]
        == {"easy": -1, "hard": 27}
        and not difficulty_regression_counterexample["passed"],
        "difficulty-stratum regression counterexample failed",
    )

    # Route selection is global: three streams cannot choose different routes.
    mixed_route_streams = {
        "stream_0": (semantic_baseline, semantic_candidate),
        "stream_1": (censor_baseline, censor_candidate),
        "stream_2": (semantic_baseline, semantic_candidate),
    }
    mixed_route_counterexample = overall_gate(mixed_route_streams)
    require(
        mixed_route_counterexample["forbidden_mixed_route_value_for_diagnostic_only"]
        and not mixed_route_counterexample["passed"],
        "cross-stream mixed-route counterexample failed",
    )

    # A 13+13 marginal split is deliberately not promoted to a third route.
    mixed_reservoir_baseline = make_uniform_arm(16, 8, 8)
    mixed_reservoir_candidate = candidate_from_deltas(
        mixed_reservoir_baseline, delta_26, [-value for value in nuisance_13]
    )
    mixed_reservoir_streams = {
        stream: (mixed_reservoir_baseline, mixed_reservoir_candidate)
        for stream in FUTURE_STREAMS
    }
    mixed_reservoir_counterexample = overall_gate(mixed_reservoir_streams)
    for stream in FUTURE_STREAMS:
        semantic_result = mixed_reservoir_counterexample["semantic_route"][
            "stream_results"
        ][stream]
        censor_result = mixed_reservoir_counterexample["censor_route"][
            "stream_results"
        ][stream]
        require(
            semantic_result["delta_V"] == 26
            and semantic_result["R_I"] == 13
            and censor_result["R_C"] == 13,
            "13+13 reservoir construction changed",
        )
    require(
        not mixed_reservoir_counterexample["passed"],
        "13+13 mixed reservoir reduction unexpectedly passed",
    )

    semantic_overall = overall_gate(semantic_streams)
    censor_overall = overall_gate(censor_streams)
    require(
        semantic_overall["passed"] and censor_overall["passed"],
        "official route-level OR rejected a valid full route",
    )

    perfect = make_uniform_arm(32, 0, 0)
    perfect_semantic = gate_stream(perfect, perfect, "semantic_rescue")
    perfect_censor = gate_stream(perfect, perfect, "censor_rescue")
    require(
        not perfect_semantic["passed"] and not perfect_censor["passed"]
        and perfect["V"] == FUTURE_N,
        "perfect-baseline counterexample failed",
    )

    return {
        "passed": True,
        "arithmetic_identity": {
            "partition": "V+C+I=N",
            "contrast": "delta_V=R_C+R_I",
            "verified_for_all_constructed_arms": True,
            "R_C_and_R_I_are_marginal_reservoir_reductions_only": True,
            "item_level_transition_proved": False,
        },
        "boundary_witnesses": {
            "semantic_rescue_with_C_baseline_zero": semantic_witness,
            "censor_rescue_with_I_baseline_zero": censor_witness,
            "semantic_C_increase_exactly_12": semantic_cap_witness,
            "censor_I_increase_exactly_12": censor_cap_witness,
            "semantic_three_stream_AND": semantic_three,
            "censor_three_stream_AND": censor_three,
            "semantic_official_overall_route": semantic_overall,
            "censor_official_overall_route": censor_overall,
        },
        "reachability": {
            "C_baseline_zero": {
                "censor_rescue_reachable": False,
                "proof": "R_C=C_b-C_c=-C_c<=0<26 because C_c>=0",
                "semantic_rescue_reachable": True,
                "witness_delta_V": semantic_witness["delta_V"],
                "witness_R_I": semantic_witness["R_I"],
            },
            "I_baseline_zero": {
                "semantic_rescue_reachable": False,
                "proof": "R_I=I_b-I_c=-I_c<=0<26 because I_c>=0",
                "censor_rescue_reachable": True,
                "witness_delta_V": censor_witness["delta_V"],
                "witness_R_C": censor_witness["R_C"],
            },
            "perfect_baseline": {
                "either_gate_reachable": False,
                "proof": "If V_b=N, then V_c<=N, so delta_V=V_c-V_b<=0<26",
                "counterexample_semantic_passed": perfect_semantic["passed"],
                "counterexample_censor_passed": perfect_censor["passed"],
            },
        },
        "counterexamples": {
            "semantic_delta_V_25": semantic_delta_v_fail,
            "semantic_R_I_25": semantic_ri_fail,
            "semantic_C_increase_13": semantic_cap_fail,
            "semantic_only_5_tasks": semantic_task_fail,
            "censor_delta_V_25": censor_delta_v_fail,
            "censor_R_C_25": censor_rc_fail,
            "censor_I_increase_13": censor_cap_fail,
            "censor_only_5_tasks": censor_task_fail,
            "two_of_three_streams_is_not_enough": and_counterexample,
            "each_stream_has_6_but_common_task_intersection_has_4": (
                task_intersection_counterexample
            ),
            "one_task_delta_V_minus_3": task_regression_counterexample,
            "easy_stratum_delta_V_minus_1": difficulty_regression_counterexample,
            "cross_stream_route_mixing_forbidden": mixed_route_counterexample,
            "mixed_R_I_13_plus_R_C_13_is_not_a_route": (
                mixed_reservoir_counterexample
            ),
        },
    }


def build_report() -> dict[str, Any]:
    source_authentication, audit = authenticate_sources()
    self_tests = run_feasibility_self_tests()
    require(self_tests["passed"] is True, "feasibility self-test failed")
    source_partition = source_authentication["natural_rows"]
    require(
        source_partition["all_cells_V_plus_C_plus_I_equals_N"] is True
        and source_partition["P_is_empty"] is True,
        "source partition/P audit failed",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": "complementary_rescue_gate_feasibility",
        "role": "design_only",
        "design_only": True,
        "source_phase": SOURCE_PHASE,
        "source_authentication": source_authentication,
        "source_decision": {
            "P": audit["passed_candidate_screens"],
            "P_is_empty": True,
            "phase979_confirmation_candidate_exists": False,
            "phase979_decision_unchanged": True,
        },
        "future_confirmation_gate_design": {
            "population": {
                "N_per_stream": FUTURE_N,
                "task_count": len(TASKS),
                "items_per_task_per_stream": TASK_N,
                "difficulty_strata": list(DIFFICULTY_STRATA),
                "items_per_difficulty_stratum_per_stream": FUTURE_N // 2,
                "stream_count": len(FUTURE_STREAMS),
                "streams": list(FUTURE_STREAMS),
            },
            "partition": {
                "V": "VALID_STOP count",
                "C": "right-censored count",
                "I": "EOS-invalid count",
                "identity": "V+C+I=N",
            },
            "contrasts": {
                "delta_V": "V_candidate-V_baseline",
                "R_C": (
                    "marginal censored-reservoir reduction "
                    "C_baseline-C_candidate"
                ),
                "R_I": (
                    "marginal invalid-reservoir reduction "
                    "I_baseline-I_candidate"
                ),
                "identity": "delta_V=R_C+R_I",
                "item_level_transition_claim": False,
            },
            "thresholds": GATE_THRESHOLDS,
            "semantic_rescue": {
                "per_stream": (
                    "delta_V>=26 AND R_I>=26 AND "
                    "signed(C_candidate-C_baseline)<=12 AND at least 6/8 tasks have "
                    "delta_V>=3 AND every task has delta_V>=-2 AND "
                    "delta_V_easy>=0 AND delta_V_hard>=0"
                ),
                "three_stream_route": (
                    "all 3 streams pass the same semantic-reservoir route AND the "
                    "intersection of their delta_V>=3 task sets has size at least 6"
                ),
            },
            "censor_rescue": {
                "per_stream": (
                    "delta_V>=26 AND R_C>=26 AND "
                    "signed(I_candidate-I_baseline)<=12 AND at least 6/8 tasks have "
                    "delta_V>=3 AND every task has delta_V>=-2 AND "
                    "delta_V_easy>=0 AND delta_V_hard>=0"
                ),
                "three_stream_route": (
                    "all 3 streams pass the same censor-reservoir route AND the "
                    "intersection of their delta_V>=3 task sets has size at least 6"
                ),
            },
            "official_overall_gate": (
                "(all3_semantic_reservoir_route) OR "
                "(all3_censor_reservoir_route)"
            ),
            "explicitly_forbidden": {
                "cross_stream_route_mixing": "all3(each_stream_semantic_OR_censor)",
                "post_hoc_mixed_reservoir_route": (
                    "R_I=13 plus R_C=13 is deliberately rejected; no third route "
                    "may be added after outcomes are seen"
                ),
                "absolute_value_cap": (
                    "The nuisance condition is a signed increase<=12, not abs(increase)<=12"
                ),
            },
            "interpretation": (
                "These are complementary feasibility definitions for a future, newly frozen "
                "confirmation. They are not evaluated on model outputs here."
            ),
            "measurement_limit": (
                "R_I and R_C compare marginal reservoir counts. They do not show that the "
                "same item moved I->V or C->V and therefore do not prove item-level rescue."
            ),
            "future_preregistration_requirements": {
                "freeze_easy_hard_assignment_before_model_evaluation": True,
                "freeze_all_three_streams_before_model_evaluation": True,
                "report_paired_item_transition_matrix_per_stream": (
                    "baseline_state x candidate_state over V/C/I"
                ),
                "item_rescue_wording_allowed_only_after_paired_transition_evidence": True,
                "mixed_route_may_not_be_added_post_hoc": True,
            },
        },
        "feasibility_self_tests": self_tests,
        "execution_contract": {
            "cpu_only": True,
            "gpu_used": False,
            "model_weights_loaded": False,
            "generation_performed": False,
            "source_files_modified": False,
        },
        "gpu_authorized": False,
        "holdout": False,
        "holdout_loaded": False,
        "holdout_authorized": False,
        "mechanism": False,
        "mechanism_authorized": False,
        "decision_boundary": (
            "This feasibility result only proves arithmetic coherence and edge-case "
            "reachability. It does not pre-register or run Phase980 confirmation, revise "
            "Phase979, open any holdout, or authorize mechanism experiments."
        ),
    }
    return {
        **payload,
        "report_sha256": core.sha256_json(payload),
        "created_at_utc": core.utc_now(),
    }


def install_or_validate(report: dict[str, Any]) -> None:
    verify_self_hash(report, "report_sha256", "created_at_utc", "Phase980 report")
    if REPORT_PATH.exists():
        prior = core.load_json(REPORT_PATH, "existing Phase980 feasibility report")
        verify_self_hash(prior, "report_sha256", "created_at_utc", "existing Phase980 report")
        require(
            prior["report_sha256"] == report["report_sha256"],
            "existing Phase980 feasibility report differs from current authenticated inputs",
        )
        return
    core.atomic_write_json(REPORT_PATH, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test", action="store_true",
        help="run authentication and feasibility tests in memory; implies no write",
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="build and validate the report without installing it",
    )
    args = parser.parse_args()
    report = build_report()
    write = not args.no_write and not args.self_test
    if write:
        install_or_validate(report)
    print(json.dumps({
        "phase": PHASE,
        "report_sha256": report["report_sha256"],
        "self_test_passed": report["feasibility_self_tests"]["passed"],
        "source_partition_valid": report["source_authentication"]["natural_rows"][
            "all_cells_V_plus_C_plus_I_equals_N"
        ],
        "P_is_empty": report["source_decision"]["P_is_empty"],
        "written": write,
        "report_path": relative(REPORT_PATH),
        "design_only": True,
        "gpu_authorized": False,
        "holdout": False,
        "mechanism": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
