#!/usr/bin/env python3
"""CPU-only Phase 981 four-channel semantic confirmation gate.

This module contains no model, tokenizer, CUDA, dataset-generation, or result
writing path.  It evaluates paired terminal-state rows for exactly two frozen
arms and three frozen streams.  The primary decision is semantic-only:
secondary censored-reservoir diagnostics can never make the primary pass.

The four terminal channels are mutually exclusive and exhaustive at the
decision checkpoint::

    V + C + I_mode + I_sem = N.

The marginal semantic gate establishes a population-level reservoir reduction.
A separately reported, stricter transition subgate is required before using
item-level ``I_sem -> V`` wording.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


# Make accidental downstream accelerator discovery impossible in this process.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

PHASE = 981
SCHEMA_VERSION = 1
EXPERIMENT = "fresh256_soft_semantic_gate"

STATES = ("V", "C", "I_mode", "I_sem")
TASKS = (
    "multistep_arithmetic",
    "modular_arithmetic",
    "boolean_logic",
    "relation_path",
    "state_machine",
    "sequence_rule",
    "string_transform",
    "constraint_order",
)
DIFFICULTIES = ("easy", "hard")
STREAMS = ("stream_0", "stream_1", "stream_2")

N_PER_STREAM = 256
N_PER_TASK = 32
N_PER_DIFFICULTY = 128
N_PER_TASK_DIFFICULTY = 16

DELTA_V_MIN = 26
R_I_SEM_MIN = 26
NON_TARGET_INCREASE_MAX = 12
TASK_DELTA_V_MIN = 3
TASKS_REQUIRED = 6
TASK_FLOOR = -2
DIFFICULTY_FLOOR = 0

# A direct-item claim is deliberately harder than the marginal primary gate.
# At least 26 actual I_sem->V moves must remain after subtracting the reverse
# V->I_sem moves, and at most 12 gross moves may enter C or I_mode from outside
# that destination channel.  The exact boundary 26/0/0 is attainable.
DIRECT_GROSS_I_SEM_TO_V_MIN = 26
DIRECT_NET_I_SEM_TO_V_MIN = 26
DIRECT_GROSS_NEW_NON_TARGET_MAX = 12

ROW_FIELDS = ("id", "task", "difficulty", "state")

GATE_CONTRACT: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "phase": PHASE,
    "experiment": EXPERIMENT,
    "cpu_only": True,
    # Stable top-level integration fields consumed by the Phase981 protocol.
    "N_per_stream": N_PER_STREAM,
    "direction": "B_minus_A",
    "channels": list(STATES),
    "primary_route": "semantic_only",
    "censor_route_role": "secondary_descriptive_only",
    "states": list(STATES),
    "partition": "V+C+I_mode+I_sem=N",
    "arms": {
        "baseline_A": "soft_no_think+thinking_sampling",
        "candidate_B": "soft_thinking+thinking_sampling",
        "contrast": "B-A",
    },
    "denominators": {
        "per_stream": N_PER_STREAM,
        "per_task": N_PER_TASK,
        "per_difficulty": N_PER_DIFFICULTY,
        "per_task_difficulty": N_PER_TASK_DIFFICULTY,
    },
    "streams": list(STREAMS),
    "primary_semantic": {
        "delta_V_min": DELTA_V_MIN,
        "R_I_sem_min": R_I_SEM_MIN,
        "delta_C_max": NON_TARGET_INCREASE_MAX,
        "delta_I_mode_max": NON_TARGET_INCREASE_MAX,
        "delta_C_plus_delta_I_mode_max": NON_TARGET_INCREASE_MAX,
        "same_route_required_in_all_streams": True,
        "censor_route_can_pass_primary": False,
    },
    "coverage": {
        "common_tasks_delta_V_min": TASK_DELTA_V_MIN,
        "common_tasks_required": TASKS_REQUIRED,
        "every_task_delta_V_floor": TASK_FLOOR,
        "easy_delta_V_floor": DIFFICULTY_FLOOR,
        "hard_delta_V_floor": DIFFICULTY_FLOOR,
    },
    "direct_item_transition_subgate": {
        "claim_only_not_primary_decision": True,
        "gross_I_sem_to_V_min": DIRECT_GROSS_I_SEM_TO_V_MIN,
        "net_I_sem_to_V_minus_V_to_I_sem_min": DIRECT_NET_I_SEM_TO_V_MIN,
        "gross_new_C_or_I_mode_max": DIRECT_GROSS_NEW_NON_TARGET_MAX,
    },
    "secondary_censor_descriptive": {
        "delta_V_min": DELTA_V_MIN,
        "R_C_min": DELTA_V_MIN,
        "delta_I_sem_max": NON_TARGET_INCREASE_MAX,
        "delta_I_mode_max": NON_TARGET_INCREASE_MAX,
        "delta_I_sem_plus_delta_I_mode_max": NON_TARGET_INCREASE_MAX,
        "can_change_primary": False,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_int(value: Any, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool),
            f"{label} is not a strict integer")
    return int(value)


def _empty_matrix() -> dict[str, dict[str, int]]:
    return {source: {target: 0 for target in STATES} for source in STATES}


def _matrix_total(matrix: Mapping[str, Mapping[str, Any]]) -> int:
    require(set(matrix) == set(STATES), "transition matrix source states changed")
    total = 0
    for source in STATES:
        require(set(matrix[source]) == set(STATES),
                f"transition matrix target states changed for {source}")
        for target in STATES:
            value = _strict_int(matrix[source][target],
                                f"matrix[{source}][{target}]")
            require(value >= 0, "transition count is negative")
            total += value
    return total


def _validate_arm(
    rows: Sequence[Mapping[str, Any]], label: str,
) -> dict[str, dict[str, Any]]:
    require(isinstance(rows, (list, tuple)), f"{label} rows are not a sequence")
    require(len(rows) == N_PER_STREAM,
            f"{label} requires exactly {N_PER_STREAM} rows, got {len(rows)}")
    output: dict[str, dict[str, Any]] = {}
    task_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    joint_counts: Counter[tuple[str, str]] = Counter()
    for index, raw in enumerate(rows):
        require(isinstance(raw, Mapping), f"{label} row {index} is not an object")
        missing = [field for field in ROW_FIELDS if field not in raw]
        require(not missing, f"{label} row {index} lacks fields {missing}")
        item_id = str(raw["id"])
        task = str(raw["task"])
        difficulty = str(raw["difficulty"])
        state = str(raw["state"])
        require(item_id and item_id not in output,
                f"{label} item ID is empty or duplicated: {item_id!r}")
        require(task in TASKS, f"{label} unknown task: {task}")
        require(difficulty in DIFFICULTIES,
                f"{label} unknown difficulty: {difficulty}")
        require(state in STATES, f"{label} unknown state: {state}")
        output[item_id] = {
            "id": item_id,
            "task": task,
            "difficulty": difficulty,
            "state": state,
        }
        task_counts[task] += 1
        difficulty_counts[difficulty] += 1
        joint_counts[(task, difficulty)] += 1
    require(task_counts == Counter({task: N_PER_TASK for task in TASKS}),
            f"{label} task denominators changed: {dict(task_counts)}")
    require(
        difficulty_counts
        == Counter({difficulty: N_PER_DIFFICULTY for difficulty in DIFFICULTIES}),
        f"{label} difficulty denominators changed: {dict(difficulty_counts)}",
    )
    require(
        joint_counts == Counter({
            (task, difficulty): N_PER_TASK_DIFFICULTY
            for task in TASKS for difficulty in DIFFICULTIES
        }),
        f"{label} task-by-difficulty denominators changed",
    )
    return output


def _paired_matrices(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = _validate_arm(baseline_rows, "baseline")
    candidate = _validate_arm(candidate_rows, "candidate")
    require(set(baseline) == set(candidate),
            "baseline/candidate paired item keys differ")
    overall = _empty_matrix()
    by_task = {task: _empty_matrix() for task in TASKS}
    by_difficulty = {difficulty: _empty_matrix() for difficulty in DIFFICULTIES}
    for item_id in sorted(baseline):
        left = baseline[item_id]
        right = candidate[item_id]
        require(
            left["task"] == right["task"]
            and left["difficulty"] == right["difficulty"],
            f"paired metadata changed between arms: {item_id}",
        )
        source = left["state"]
        target = right["state"]
        task = left["task"]
        difficulty = left["difficulty"]
        overall[source][target] += 1
        by_task[task][source][target] += 1
        by_difficulty[difficulty][source][target] += 1
    require(_matrix_total(overall) == N_PER_STREAM,
            "overall transition matrix denominator changed")
    require(all(_matrix_total(by_task[task]) == N_PER_TASK for task in TASKS),
            "task transition matrix denominator changed")
    require(all(
        _matrix_total(by_difficulty[difficulty]) == N_PER_DIFFICULTY
        for difficulty in DIFFICULTIES
    ), "difficulty transition matrix denominator changed")
    return {
        "overall": overall,
        "by_task": by_task,
        "by_difficulty": by_difficulty,
    }


def _marginals(matrix: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    total = _matrix_total(matrix)
    baseline = {
        state: sum(int(matrix[state][target]) for target in STATES)
        for state in STATES
    }
    candidate = {
        state: sum(int(matrix[source][state]) for source in STATES)
        for state in STATES
    }
    require(sum(baseline.values()) == sum(candidate.values()) == total,
            "transition marginals changed denominator")
    delta_v = candidate["V"] - baseline["V"]
    r_c = baseline["C"] - candidate["C"]
    r_i_mode = baseline["I_mode"] - candidate["I_mode"]
    r_i_sem = baseline["I_sem"] - candidate["I_sem"]
    delta_c = -r_c
    delta_i_mode = -r_i_mode
    delta_i_sem = -r_i_sem
    require(delta_v == r_c + r_i_mode + r_i_sem,
            "four-channel conservation identity failed")
    return {
        "N": total,
        "baseline": baseline,
        "candidate": candidate,
        "delta_V": delta_v,
        "R_C": r_c,
        "R_I_mode": r_i_mode,
        "R_I_sem": r_i_sem,
        "delta_C": delta_c,
        "delta_I_mode": delta_i_mode,
        "delta_I_sem": delta_i_sem,
        "delta_C_plus_delta_I_mode": delta_c + delta_i_mode,
        "delta_I_sem_plus_delta_I_mode": delta_i_sem + delta_i_mode,
        "identity": "delta_V=R_C+R_I_mode+R_I_sem",
        "identity_verified": True,
    }


def _direct_transition_subgate(
    matrix: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gross = int(matrix["I_sem"]["V"])
    reverse = int(matrix["V"]["I_sem"])
    net = gross - reverse
    gross_new_c = sum(
        int(matrix[source]["C"]) for source in STATES if source != "C"
    )
    gross_new_i_mode = sum(
        int(matrix[source]["I_mode"])
        for source in STATES if source != "I_mode"
    )
    gross_new_non_target = gross_new_c + gross_new_i_mode
    checks = {
        "gross_I_sem_to_V_at_least_26": (
            gross >= DIRECT_GROSS_I_SEM_TO_V_MIN
        ),
        "net_I_sem_to_V_minus_V_to_I_sem_at_least_26": (
            net >= DIRECT_NET_I_SEM_TO_V_MIN
        ),
        "gross_new_C_or_I_mode_at_most_12": (
            gross_new_non_target <= DIRECT_GROSS_NEW_NON_TARGET_MAX
        ),
    }
    return {
        "gross_I_sem_to_V": gross,
        "gross_V_to_I_sem": reverse,
        "net_I_sem_to_V_minus_V_to_I_sem": net,
        "gross_new_C": gross_new_c,
        "gross_new_I_mode": gross_new_i_mode,
        "gross_new_C_or_I_mode": gross_new_non_target,
        "checks": checks,
        "passed": all(checks.values()),
        "claim_authorized_if_three_stream_primary_also_passes": (
            "direct_item_I_sem_to_V_evidence"
        ),
    }


def evaluate_stream(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate one paired stream without allowing any denominator exclusion."""
    matrices = _paired_matrices(baseline_rows, candidate_rows)
    overall = _marginals(matrices["overall"])
    task_delta_v = {
        task: _marginals(matrices["by_task"][task])["delta_V"]
        for task in TASKS
    }
    difficulty_delta_v = {
        difficulty: _marginals(matrices["by_difficulty"][difficulty])["delta_V"]
        for difficulty in DIFFICULTIES
    }
    qualifying_tasks = sorted(
        task for task, value in task_delta_v.items()
        if value >= TASK_DELTA_V_MIN
    )

    shared_coverage_checks = {
        "at_least_6_tasks_delta_V_at_least_3": (
            len(qualifying_tasks) >= TASKS_REQUIRED
        ),
        "every_task_delta_V_at_least_minus_2": all(
            value >= TASK_FLOOR for value in task_delta_v.values()
        ),
        "easy_delta_V_at_least_0": (
            difficulty_delta_v["easy"] >= DIFFICULTY_FLOOR
        ),
        "hard_delta_V_at_least_0": (
            difficulty_delta_v["hard"] >= DIFFICULTY_FLOOR
        ),
    }
    primary_checks = {
        "delta_V_at_least_26": overall["delta_V"] >= DELTA_V_MIN,
        "R_I_sem_at_least_26": overall["R_I_sem"] >= R_I_SEM_MIN,
        "delta_C_at_most_12": (
            overall["delta_C"] <= NON_TARGET_INCREASE_MAX
        ),
        "delta_I_mode_at_most_12": (
            overall["delta_I_mode"] <= NON_TARGET_INCREASE_MAX
        ),
        "delta_C_plus_delta_I_mode_at_most_12": (
            overall["delta_C_plus_delta_I_mode"]
            <= NON_TARGET_INCREASE_MAX
        ),
        **shared_coverage_checks,
    }
    primary_stream_passed = all(primary_checks.values())

    # Secondary only.  Even three-stream censor PASS is deliberately unable to
    # modify primary_semantic_passed in evaluate_three_streams.
    censor_checks = {
        "delta_V_at_least_26": overall["delta_V"] >= DELTA_V_MIN,
        "R_C_at_least_26": overall["R_C"] >= DELTA_V_MIN,
        "delta_I_sem_at_most_12": (
            overall["delta_I_sem"] <= NON_TARGET_INCREASE_MAX
        ),
        "delta_I_mode_at_most_12": (
            overall["delta_I_mode"] <= NON_TARGET_INCREASE_MAX
        ),
        "delta_I_sem_plus_delta_I_mode_at_most_12": (
            overall["delta_I_sem_plus_delta_I_mode"]
            <= NON_TARGET_INCREASE_MAX
        ),
        **shared_coverage_checks,
    }
    censor_secondary_passed = all(censor_checks.values())
    direct = _direct_transition_subgate(matrices["overall"])
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "overall": overall,
        "by_task_delta_V": task_delta_v,
        "by_difficulty_delta_V": difficulty_delta_v,
        "qualifying_tasks": qualifying_tasks,
        "primary_semantic_checks": primary_checks,
        "primary_semantic_stream_passed": primary_stream_passed,
        "direct_I_sem_to_V_subgate": direct,
        "direct_claim_stream_passed": (
            primary_stream_passed and bool(direct["passed"])
        ),
        "secondary_censor_checks": censor_checks,
        "secondary_censor_stream_passed": censor_secondary_passed,
        "secondary_censor_can_set_primary": False,
        "transition_matrix_4x4": matrices["overall"],
        "transition_matrix_sha256": sha256_json(matrices["overall"]),
    }


def evaluate_three_streams(
    stream_pairs: Mapping[
        str, tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]
    ],
) -> dict[str, Any]:
    """Apply three-stream AND to semantic primary; censor remains descriptive."""
    require(isinstance(stream_pairs, Mapping), "stream_pairs is not a mapping")
    require(set(stream_pairs) == set(STREAMS),
            f"exactly three frozen streams are required: {STREAMS}")
    results: dict[str, Any] = {}
    for stream in STREAMS:
        pair = stream_pairs[stream]
        require(isinstance(pair, (tuple, list)) and len(pair) == 2,
                f"{stream} does not contain baseline/candidate rows")
        results[stream] = evaluate_stream(pair[0], pair[1])

    common_tasks = set(TASKS)
    for stream in STREAMS:
        common_tasks &= set(results[stream]["qualifying_tasks"])
    common_tasks_sorted = sorted(common_tasks)
    common_task_gate = len(common_tasks_sorted) >= TASKS_REQUIRED
    all_primary_streams = all(
        results[stream]["primary_semantic_stream_passed"]
        for stream in STREAMS
    )
    primary_passed = all_primary_streams and common_task_gate
    direct_passed = primary_passed and all(
        results[stream]["direct_claim_stream_passed"] for stream in STREAMS
    )
    secondary_censor_passed = common_task_gate and all(
        results[stream]["secondary_censor_stream_passed"]
        for stream in STREAMS
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "stream_results": results,
        "common_qualifying_tasks": common_tasks_sorted,
        "common_qualifying_task_count": len(common_tasks_sorted),
        "common_task_gate_passed": common_task_gate,
        "all_three_semantic_streams_passed": all_primary_streams,
        "primary_semantic_passed": primary_passed,
        "direct_item_I_sem_to_V_evidence_passed": direct_passed,
        "direct_claim_authorized": direct_passed,
        "secondary_censor_descriptive_passed": secondary_censor_passed,
        "secondary_censor_can_set_primary": False,
        "primary_formula": (
            "all3(primary_semantic_stream) AND common_tasks>=6"
        ),
        "no_route_or_in_primary": True,
    }


# ---------------------------------------------------------------------------
# Synthetic fail-closed tests
# ---------------------------------------------------------------------------

Plan = dict[tuple[str, str], Counter[tuple[str, str]]]


def _new_plan() -> Plan:
    return {
        (task, difficulty): Counter()
        for task in TASKS for difficulty in DIFFICULTIES
    }


def _add(
    plan: Plan, task: str, difficulty: str,
    source: str, target: str, count: int,
) -> None:
    require(task in TASKS and difficulty in DIFFICULTIES,
            "synthetic plan metadata invalid")
    require(source in STATES and target in STATES,
            "synthetic transition state invalid")
    count = _strict_int(count, "synthetic transition count")
    require(count >= 0, "synthetic transition count is negative")
    plan[(task, difficulty)][(source, target)] += count
    require(sum(plan[(task, difficulty)].values()) <= N_PER_TASK_DIFFICULTY,
            "synthetic task-by-difficulty cell exceeds 16 rows")


def _rows_from_plan(plan: Plan) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    require(set(plan) == {
        (task, difficulty) for task in TASKS for difficulty in DIFFICULTIES
    }, "synthetic plan cells changed")
    baseline: list[dict[str, str]] = []
    candidate: list[dict[str, str]] = []
    for task in TASKS:
        for difficulty in DIFFICULTIES:
            transitions: list[tuple[str, str]] = []
            for (source, target), count in sorted(plan[(task, difficulty)].items()):
                transitions.extend([(source, target)] * count)
            require(len(transitions) <= N_PER_TASK_DIFFICULTY,
                    "synthetic transition list is too long")
            transitions.extend(
                [("V", "V")] * (N_PER_TASK_DIFFICULTY - len(transitions))
            )
            for index, (source, target) in enumerate(transitions):
                item_id = f"p981_syn_{task}_{difficulty}_{index:02d}"
                shared = {"id": item_id, "task": task, "difficulty": difficulty}
                baseline.append({**shared, "state": source})
                candidate.append({**shared, "state": target})
    require(len(baseline) == len(candidate) == N_PER_STREAM,
            "synthetic row denominator changed")
    return baseline, candidate


def _split_positive_deltas(
    task_deltas: Sequence[int],
) -> tuple[list[int], list[int]]:
    require(len(task_deltas) == len(TASKS), "task delta vector length changed")
    require(all(_strict_int(value, "task delta") >= 0 for value in task_deltas),
            "positive delta splitter received a negative value")
    total = sum(task_deltas)
    easy = [value // 2 for value in task_deltas]
    target_easy = total // 2
    for index, value in enumerate(task_deltas):
        if sum(easy) >= target_easy:
            break
        if easy[index] < value:
            easy[index] += 1
    require(sum(easy) == target_easy, "could not balance synthetic difficulty delta")
    hard = [value - easy[index] for index, value in enumerate(task_deltas)]
    return easy, hard


def _positive_pair(
    task_deltas: Sequence[int], sources: Sequence[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], Plan]:
    easy, hard = _split_positive_deltas(task_deltas)
    total = sum(task_deltas)
    source_values = list(sources) if sources is not None else ["I_sem"] * total
    require(len(source_values) == total and all(value in STATES for value in source_values),
            "synthetic source sequence length/state changed")
    plan = _new_plan()
    cursor = 0
    for task_index, task in enumerate(TASKS):
        for difficulty, counts in (("easy", easy), ("hard", hard)):
            for _ in range(counts[task_index]):
                _add(plan, task, difficulty, source_values[cursor], "V", 1)
                cursor += 1
    require(cursor == total, "synthetic source sequence was not consumed")
    baseline, candidate = _rows_from_plan(plan)
    return baseline, candidate, plan


def _add_greedy_transition(
    plan: Plan, source: str, target: str, count: int,
) -> None:
    remaining = count
    for task in TASKS:
        for difficulty in DIFFICULTIES:
            capacity = (
                N_PER_TASK_DIFFICULTY
                - sum(plan[(task, difficulty)].values())
            )
            take = min(capacity, remaining)
            if take:
                _add(plan, task, difficulty, source, target, take)
                remaining -= take
            if remaining == 0:
                return
    raise RuntimeError("synthetic plan lacks transition capacity")


def _three(
    pair: tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]
) -> dict[str, tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]]:
    return {stream: pair for stream in STREAMS}


def _expect_runtime_error(callable_value, label: str) -> str:
    try:
        callable_value()
    except RuntimeError as exc:
        return str(exc)
    raise RuntimeError(f"fail-closed test did not fail: {label}")


def self_test() -> dict[str, Any]:
    """Exercise every threshold boundary and known gate-confusion failure."""
    pass_vector = [4, 4, 4, 4, 3, 3, 2, 2]
    baseline, candidate, _plan = _positive_pair(pass_vector)
    boundary = evaluate_three_streams(_three((baseline, candidate)))
    require(boundary["primary_semantic_passed"],
            "exact primary semantic boundary should pass")
    require(boundary["direct_item_I_sem_to_V_evidence_passed"],
            "exact direct I_sem->V boundary should pass")
    first_stream = boundary["stream_results"]["stream_0"]
    require(first_stream["overall"]["delta_V"] == 26
            and first_stream["overall"]["R_I_sem"] == 26,
            "exact 26 boundary changed")
    require(_matrix_total(first_stream["transition_matrix_4x4"]) == 256,
            "4x4 transition matrix denominator changed")

    below_vector = [4, 4, 4, 4, 3, 3, 2, 1]
    below_pair = _positive_pair(below_vector)[:2]
    below = evaluate_three_streams(_three(below_pair))
    require(not below["primary_semantic_passed"],
            "delta_V/R_I_sem=25 should fail")

    # Aggregate invalid reduction caused only by mode compliance is not semantic.
    mode_pair = _positive_pair(pass_vector, ["I_mode"] * 26)[:2]
    mode_confusion = evaluate_three_streams(_three(mode_pair))
    require(
        not mode_confusion["primary_semantic_passed"]
        and mode_confusion["stream_results"]["stream_0"]["overall"]["R_I_sem"] == 0,
        "I_mode reduction was incorrectly promoted to semantic primary",
    )

    # Exact non-target boundary 12 passes; one count more fails.
    cross12_plan = _positive_pair(pass_vector)[2]
    _add_greedy_transition(cross12_plan, "I_sem", "C", 12)
    cross12_pair = _rows_from_plan(cross12_plan)
    cross12 = evaluate_three_streams(_three(cross12_pair))
    require(cross12["primary_semantic_passed"],
            "delta_C=12 semantic boundary should pass")
    require(cross12["direct_item_I_sem_to_V_evidence_passed"],
            "gross new non-target=12 direct boundary should pass")
    cross13_plan = _positive_pair(pass_vector)[2]
    _add_greedy_transition(cross13_plan, "I_sem", "C", 13)
    cross13 = evaluate_three_streams(_three(_rows_from_plan(cross13_plan)))
    require(not cross13["primary_semantic_passed"],
            "delta_C=13 should fail primary")

    mode12_plan = _positive_pair(pass_vector)[2]
    _add_greedy_transition(mode12_plan, "I_sem", "I_mode", 12)
    mode12 = evaluate_three_streams(_three(_rows_from_plan(mode12_plan)))
    require(mode12["primary_semantic_passed"],
            "delta_I_mode=12 semantic boundary should pass")
    mode13_plan = _positive_pair(pass_vector)[2]
    _add_greedy_transition(mode13_plan, "I_sem", "I_mode", 13)
    mode13 = evaluate_three_streams(_three(_rows_from_plan(mode13_plan)))
    require(not mode13["primary_semantic_passed"],
            "delta_I_mode=13 should fail primary")

    combined13_plan = _positive_pair(pass_vector)[2]
    _add_greedy_transition(combined13_plan, "I_sem", "C", 7)
    _add_greedy_transition(combined13_plan, "I_sem", "I_mode", 6)
    combined13 = evaluate_three_streams(
        _three(_rows_from_plan(combined13_plan))
    )
    require(
        not combined13["primary_semantic_passed"]
        and combined13["stream_results"]["stream_0"]["overall"][
            "delta_C_plus_delta_I_mode"
        ] == 13,
        "combined non-target increase=13 should fail when components <=12",
    )

    # 13 semantic + 13 censored reductions are deliberately not a semantic pass.
    mixed_sources = ["I_sem"] * 13 + ["C"] * 13
    mixed_pair = _positive_pair(pass_vector, mixed_sources)[:2]
    mixed_13_13 = evaluate_three_streams(_three(mixed_pair))
    require(not mixed_13_13["primary_semantic_passed"],
            "R_I_sem=13 plus R_C=13 must not create a semantic route")

    # A pure censored-reservoir improvement may be descriptively strong but can
    # never set the semantic primary decision.
    censor_pair = _positive_pair(pass_vector, ["C"] * 26)[:2]
    censor_only = evaluate_three_streams(_three(censor_pair))
    require(
        censor_only["secondary_censor_descriptive_passed"]
        and not censor_only["primary_semantic_passed"]
        and censor_only["secondary_censor_can_set_primary"] is False,
        "secondary censor route leaked into primary",
    )

    # Cross-stream semantic/censor mixing cannot pass either all-stream result.
    cross_stream = evaluate_three_streams({
        "stream_0": (baseline, candidate),
        "stream_1": (baseline, candidate),
        "stream_2": censor_pair,
    })
    require(
        not cross_stream["primary_semantic_passed"]
        and not cross_stream["secondary_censor_descriptive_passed"],
        "cross-stream route mixing should fail",
    )

    # Six qualifying tasks per stream is insufficient if they are not the same.
    intersection_vectors = (
        [5, 5, 4, 4, 4, 4, 0, 0],
        [0, 0, 5, 5, 4, 4, 4, 4],
        [5, 5, 0, 0, 4, 4, 4, 4],
    )
    intersection_pairs = [_positive_pair(vector)[:2] for vector in intersection_vectors]
    task_intersection = evaluate_three_streams({
        stream: intersection_pairs[index]
        for index, stream in enumerate(STREAMS)
    })
    require(
        all(
            len(task_intersection["stream_results"][stream]["qualifying_tasks"])
            >= 6 for stream in STREAMS
        )
        and task_intersection["common_qualifying_task_count"] < 6
        and not task_intersection["primary_semantic_passed"],
        "different six-task sets should fail the common intersection gate",
    )

    # Six good tasks cannot hide one task at -3.
    floor_plan = _new_plan()
    for task_index, task in enumerate(TASKS[:6]):
        if task_index % 2 == 0:
            _add(floor_plan, task, "easy", "I_sem", "V", 3)
            _add(floor_plan, task, "hard", "I_sem", "V", 2)
        else:
            _add(floor_plan, task, "easy", "I_sem", "V", 2)
            _add(floor_plan, task, "hard", "I_sem", "V", 3)
    _add(floor_plan, TASKS[6], "easy", "V", "I_sem", 1)
    _add(floor_plan, TASKS[7], "easy", "V", "I_sem", 1)
    _add(floor_plan, TASKS[7], "hard", "V", "I_sem", 2)
    floor_case = evaluate_three_streams(_three(_rows_from_plan(floor_plan)))
    require(
        floor_case["stream_results"]["stream_0"]["overall"]["delta_V"] == 26
        and not floor_case["primary_semantic_passed"],
        "task delta_V=-3 should fail despite total delta_V=26",
    )

    # Total improvement cannot hide easy=-1, hard=+27.
    difficulty_plan = _new_plan()
    hard_deltas = [5, 5, 5, 4, 4, 4, 0, 0]
    for task, count in zip(TASKS, hard_deltas):
        _add(difficulty_plan, task, "hard", "I_sem", "V", count)
    _add(difficulty_plan, TASKS[-1], "easy", "V", "I_sem", 1)
    difficulty_case = evaluate_three_streams(
        _three(_rows_from_plan(difficulty_plan))
    )
    require(
        difficulty_case["stream_results"]["stream_0"]["overall"]["delta_V"] == 26
        and not difficulty_case["primary_semantic_passed"],
        "easy difficulty regression should fail",
    )

    # Marginal primary may pass while the strict direct-flow claim fails.
    flow_plan = _positive_pair(pass_vector)[2]
    _add_greedy_transition(flow_plan, "V", "I_sem", 1)
    _add_greedy_transition(flow_plan, "I_sem", "C", 1)
    _add_greedy_transition(flow_plan, "C", "V", 1)
    flow_exchange = evaluate_three_streams(_three(_rows_from_plan(flow_plan)))
    require(
        flow_exchange["primary_semantic_passed"]
        and not flow_exchange["direct_item_I_sem_to_V_evidence_passed"],
        "net direct=25 flow exchange should retain marginal PASS but reject direct claim",
    )

    # The marginal identity can also pass through an indirect C->V leg while
    # gross direct I_sem->V is only 25; direct wording must still be rejected.
    gross25_plan = _positive_pair(below_vector)[2]
    _add_greedy_transition(gross25_plan, "I_sem", "C", 1)
    _add_greedy_transition(gross25_plan, "C", "V", 1)
    gross25 = evaluate_three_streams(_three(_rows_from_plan(gross25_plan)))
    require(
        gross25["primary_semantic_passed"]
        and not gross25["direct_item_I_sem_to_V_evidence_passed"]
        and gross25["stream_results"]["stream_0"][
            "direct_I_sem_to_V_subgate"
        ]["gross_I_sem_to_V"] == 25,
        "gross direct=25 should retain marginal PASS but reject direct claim",
    )

    # Gross non-target churn is capped conservatively even when reverse flow
    # makes the marginal non-target increase look acceptable (13-1=12).
    gross_non_target_plan = _positive_pair(pass_vector)[2]
    _add_greedy_transition(gross_non_target_plan, "I_sem", "C", 13)
    _add_greedy_transition(gross_non_target_plan, "C", "I_sem", 1)
    gross_non_target = evaluate_three_streams(
        _three(_rows_from_plan(gross_non_target_plan))
    )
    require(
        gross_non_target["primary_semantic_passed"]
        and not gross_non_target["direct_item_I_sem_to_V_evidence_passed"]
        and gross_non_target["stream_results"]["stream_0"][
            "direct_I_sem_to_V_subgate"
        ]["gross_new_C_or_I_mode"] == 13,
        "gross non-target=13 should reject direct claim despite marginal delta=12",
    )

    missing_stream_error = _expect_runtime_error(
        lambda: evaluate_three_streams({
            "stream_0": (baseline, candidate),
            "stream_1": (baseline, candidate),
        }),
        "missing stream",
    )
    missing_row_error = _expect_runtime_error(
        lambda: evaluate_stream(baseline, candidate[:-1]),
        "missing row",
    )

    cases = {
        "exact_26_primary_and_direct_pass": True,
        "delta_25_fails": True,
        "I_mode_does_not_count_as_I_sem": True,
        "non_target_12_passes_13_fails": True,
        "delta_I_mode_12_passes_13_fails": True,
        "combined_non_target_13_fails": True,
        "R_I_sem_13_plus_R_C_13_fails": True,
        "censor_secondary_cannot_set_primary": True,
        "cross_stream_route_mixing_fails": True,
        "common_task_intersection_required": True,
        "task_minus_3_fails": True,
        "difficulty_minus_1_fails": True,
        "marginal_pass_does_not_imply_direct_pass": True,
        "gross_direct_25_rejects_direct_claim": True,
        "gross_non_target_13_rejects_direct_claim": True,
        "missing_stream_fails_closed": bool(missing_stream_error),
        "missing_row_fails_closed": bool(missing_row_error),
    }
    require(all(cases.values()), "one or more Phase981 self-tests failed")
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "passed": True,
        "cpu_only": True,
        "gpu_used": False,
        "model_weights_loaded": False,
        "generation_performed": False,
        "gate_contract_sha256": sha256_json(GATE_CONTRACT),
        "tests": cases,
        "fail_closed_messages": {
            "missing_stream": missing_stream_error,
            "missing_row": missing_row_error,
        },
        "design_judgment": {
            "primary": "marginal semantic-reservoir confirmation only",
            "direct_subgate": (
                "strict paired I_sem->V evidence; required only for direct-item wording"
            ),
            "censor": "secondary descriptive only; cannot change primary",
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test", action="store_true",
        help="run the complete synthetic fail-closed gate audit (also the default)",
    )
    parser.add_argument(
        "--contract", action="store_true",
        help="include the complete frozen gate contract in printed output",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = self_test()
    report["script_sha256"] = sha256_file(Path(__file__).resolve())
    if args.contract:
        report["gate_contract"] = GATE_CONTRACT
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
