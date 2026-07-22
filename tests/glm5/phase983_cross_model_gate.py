#!/usr/bin/env python3
"""CPU-only Phase 983 cross-model external-bundle gate.

The scientific object in this module is a paired *external configuration
bundle* contrast driven by the same natural-language instruction contract in
Qwen3, GLM4, and DS7B.  It is not a claim about an internal thinking circuit,
an EOS switch, or a shared cross-model mechanism.

For every model and frozen stream, exactly 256 paired items are classified into
four mutually exclusive and exhaustive external terminal categories::

    V + C + I_protocol + I_sem = 256.

All decisions are made within a model and stream before conjunction.  Pooling
rows, counts, streams, or models is deliberately unsupported and fails closed.
The thresholds encode a pre-fixed operational utility of roughly 10% benefit
and 5% tolerated harm.  They are neither significance tests nor natural
constants.

This file has no dataset generation, model/tokenizer import, CUDA path,
holdout access, mechanism authorization, or result-writing path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


# Hide accelerators even if this module is imported by a broader process.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PHASE = 983
SCHEMA_VERSION = 1
EXPERIMENT = "cross_model_external_configuration_bundle_gate"

MODELS = ("qwen3", "glm4", "deepseek7b")
MODEL_DISPLAY_NAMES = {
    "qwen3": "Qwen3",
    "glm4": "GLM4",
    "deepseek7b": "DS7B",
}
STREAMS = ("stream_0", "stream_1", "stream_2")
STATES = ("V", "C", "I_protocol", "I_sem")
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
GOLD_LABELS = ("A", "B")
SWAP_VARIANTS = ("original", "swapped")

N_PER_MODEL_STREAM = 256
N_PER_TASK = 32
N_PER_DIFFICULTY = 128
N_PER_GOLD_LABEL = 128
N_PER_SWAP_VARIANT = 128
N_PER_TASK_DIFFICULTY = 16
N_PER_GOLD_SWAP = 64
N_PER_TASK_DIFFICULTY_GOLD_SWAP = 4
N_SEMANTIC_PAIRS = 128

DELTA_V_MIN = 26
R_I_SEM_MIN = 26
NON_TARGET_INCREASE_MAX = 12
PAIRED_I_SEM_TO_V_MIN = 26
PAIRED_NET_MIN = 26
PAIRED_LEAKAGE_MAX = 12
TASK_DELTA_V_MIN = 3
TASKS_REQUIRED = 6
TASK_FLOOR = -2
DIFFICULTY_FLOOR = 0
STRATUM_DELTA_V_MIN = 13
STRATUM_NON_TARGET_INCREASE_MAX = 6
B_GOLD_C_ABS_DIFF_MAX = 12

ROW_FIELDS = (
    "id",
    "semantic_id",
    "task",
    "difficulty",
    "gold_label",
    "swap_variant",
    "state",
)
METADATA_FIELDS = ROW_FIELDS[:-1]

GATE_CONTRACT: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "phase": PHASE,
    "experiment": EXPERIMENT,
    "cpu_only": True,
    "scientific_object": "cross-model external configuration bundle contrast",
    "same_natural_language_instruction_contract": True,
    "internal_thinking_mechanism_claim": False,
    "direction": "external_bundle_B_minus_external_bundle_A",
    "models": list(MODELS),
    "model_display_names": MODEL_DISPLAY_NAMES,
    "streams": list(STREAMS),
    "states": list(STATES),
    "partition": "V+C+I_protocol+I_sem=N",
    "denominators": {
        "per_model_stream_arm": N_PER_MODEL_STREAM,
        "per_task": N_PER_TASK,
        "per_difficulty": N_PER_DIFFICULTY,
        "per_gold_label": N_PER_GOLD_LABEL,
        "per_swap_variant": N_PER_SWAP_VARIANT,
        "per_task_difficulty": N_PER_TASK_DIFFICULTY,
        "per_gold_swap_cell": N_PER_GOLD_SWAP,
        "per_task_difficulty_gold_swap_cell": (
            N_PER_TASK_DIFFICULTY_GOLD_SWAP
        ),
        "semantic_pairs": N_SEMANTIC_PAIRS,
    },
    "pairing": {
        "one_original_and_one_swapped_surface_per_semantic_id": True,
        "opposite_gold_labels_within_semantic_pair": True,
        "twins_are_not_independent_samples": True,
        "baseline_candidate_item_keys_and_metadata_must_match": True,
    },
    "operational_threshold_interpretation": {
        "benefit": "pre-fixed approximately 10 percent of N=256",
        "harm": "pre-fixed approximately 5 percent of N=256",
        "gold_and_swap_strata": "proportionally halved for N=128",
        "statistical_significance_test": False,
        "natural_constant": False,
        "caller_overrides_allowed": False,
    },
    "per_model_stream_primary": {
        "delta_V_min": DELTA_V_MIN,
        "R_I_sem_min": R_I_SEM_MIN,
        "delta_C_max": NON_TARGET_INCREASE_MAX,
        "delta_I_protocol_max": NON_TARGET_INCREASE_MAX,
        "delta_C_plus_I_protocol_max": NON_TARGET_INCREASE_MAX,
        "paired_I_sem_to_V_min": PAIRED_I_SEM_TO_V_MIN,
        "paired_net_I_sem_to_V_minus_V_to_I_sem_min": PAIRED_NET_MIN,
        "paired_bad_leakage_max": PAIRED_LEAKAGE_MAX,
        "paired_bad_leakage_formula": (
            "I_sem->C + I_sem->I_protocol + V->C + V->I_protocol"
        ),
    },
    "coverage": {
        "qualifying_task_delta_V_min": TASK_DELTA_V_MIN,
        "qualifying_tasks_required": TASKS_REQUIRED,
        "every_task_delta_V_floor": TASK_FLOOR,
        "easy_and_hard_delta_V_floor": DIFFICULTY_FLOOR,
        "same_tasks_required_within_model_stream_intersection": True,
        "same_tasks_required_across_all_model_stream_cells": True,
    },
    "gold_and_swap_safety": {
        "each_N": N_PER_GOLD_LABEL,
        "delta_V_min": STRATUM_DELTA_V_MIN,
        "delta_C_plus_I_protocol_max": STRATUM_NON_TARGET_INCREASE_MAX,
        "candidate_B_abs_C_gold_A_minus_gold_B_max": B_GOLD_C_ABS_DIFF_MAX,
    },
    "decision": {
        "pooling_allowed": False,
        "model_pass": "all3_streams AND common_tasks>=6",
        "cross_model_pass": "all3_models AND all9_common_tasks>=6",
        "secondary_can_set_primary": False,
    },
}


def require(condition: bool, message: str) -> None:
    """Raise a stable fail-closed error when an invariant is not met."""
    if not condition:
        raise RuntimeError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
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
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} is not a strict integer",
    )
    return int(value)


def _assert_cpu_import_boundary() -> None:
    forbidden_roots = {"torch", "transformers"}
    imported = sorted(
        name for name in sys.modules
        if name.split(".", 1)[0] in forbidden_roots
        or "phase977_holdout" in name.lower()
        or "holdout_dataset" in name.lower()
    )
    require(not imported, f"forbidden model/holdout modules imported: {imported}")


def _empty_matrix() -> dict[str, dict[str, int]]:
    return {source: {target: 0 for target in STATES} for source in STATES}


def _matrix_total(matrix: Mapping[str, Mapping[str, Any]]) -> int:
    require(set(matrix) == set(STATES), "transition matrix source states changed")
    total = 0
    for source in STATES:
        targets = matrix[source]
        require(
            isinstance(targets, Mapping) and set(targets) == set(STATES),
            f"transition matrix target states changed for {source}",
        )
        for target in STATES:
            value = _strict_int(targets[target], f"matrix[{source}][{target}]")
            require(value >= 0, "transition matrix contains a negative count")
            total += value
    return total


def _new_matrix_map(keys: Sequence[str]) -> dict[str, dict[str, dict[str, int]]]:
    return {key: _empty_matrix() for key in keys}


def _validate_arm(
    rows: Sequence[Mapping[str, Any]], label: str,
) -> dict[str, dict[str, str]]:
    require(
        isinstance(rows, (list, tuple)),
        f"{label} rows must be a concrete list or tuple; pooling is forbidden",
    )
    require(
        len(rows) == N_PER_MODEL_STREAM,
        f"{label} requires exactly {N_PER_MODEL_STREAM} rows, got {len(rows)}; "
        "stream/model pooling is forbidden",
    )

    output: dict[str, dict[str, str]] = {}
    task_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    gold_counts: Counter[str] = Counter()
    swap_counts: Counter[str] = Counter()
    task_difficulty_counts: Counter[tuple[str, str]] = Counter()
    gold_swap_counts: Counter[tuple[str, str]] = Counter()
    full_grid_counts: Counter[tuple[str, str, str, str]] = Counter()
    semantic_groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)

    for index, raw in enumerate(rows):
        require(isinstance(raw, Mapping), f"{label} row {index} is not an object")
        missing = [field for field in ROW_FIELDS if field not in raw]
        require(not missing, f"{label} row {index} lacks fields {missing}")
        row = {field: str(raw[field]) for field in ROW_FIELDS}
        item_id = row["id"]
        semantic_id = row["semantic_id"]
        task = row["task"]
        difficulty = row["difficulty"]
        gold = row["gold_label"]
        swap = row["swap_variant"]
        state = row["state"]
        require(item_id, f"{label} row {index} has an empty item ID")
        require(semantic_id, f"{label} row {index} has an empty semantic ID")
        require(item_id not in output, f"{label} duplicate item ID: {item_id}")
        require(task in TASKS, f"{label} unknown task: {task}")
        require(difficulty in DIFFICULTIES, f"{label} unknown difficulty: {difficulty}")
        require(gold in GOLD_LABELS, f"{label} unknown gold label: {gold}")
        require(swap in SWAP_VARIANTS, f"{label} unknown swap variant: {swap}")
        require(state in STATES, f"{label} unknown external terminal state: {state}")
        output[item_id] = row
        semantic_groups[semantic_id].append(row)
        task_counts[task] += 1
        difficulty_counts[difficulty] += 1
        gold_counts[gold] += 1
        swap_counts[swap] += 1
        task_difficulty_counts[(task, difficulty)] += 1
        gold_swap_counts[(gold, swap)] += 1
        full_grid_counts[(task, difficulty, gold, swap)] += 1

    require(
        task_counts == Counter({task: N_PER_TASK for task in TASKS}),
        f"{label} task denominators/zero cells changed: {dict(task_counts)}",
    )
    require(
        difficulty_counts
        == Counter({difficulty: N_PER_DIFFICULTY for difficulty in DIFFICULTIES}),
        f"{label} difficulty denominators changed: {dict(difficulty_counts)}",
    )
    require(
        gold_counts == Counter({gold: N_PER_GOLD_LABEL for gold in GOLD_LABELS}),
        f"{label} gold-label denominators changed: {dict(gold_counts)}",
    )
    require(
        swap_counts
        == Counter({swap: N_PER_SWAP_VARIANT for swap in SWAP_VARIANTS}),
        f"{label} swap denominators changed: {dict(swap_counts)}",
    )
    require(
        task_difficulty_counts
        == Counter({
            (task, difficulty): N_PER_TASK_DIFFICULTY
            for task in TASKS for difficulty in DIFFICULTIES
        }),
        f"{label} task-by-difficulty denominators/zero cells changed",
    )
    require(
        gold_swap_counts
        == Counter({
            (gold, swap): N_PER_GOLD_SWAP
            for gold in GOLD_LABELS for swap in SWAP_VARIANTS
        }),
        f"{label} gold-by-swap denominators/zero cells changed",
    )
    require(
        full_grid_counts
        == Counter({
            (task, difficulty, gold, swap): N_PER_TASK_DIFFICULTY_GOLD_SWAP
            for task in TASKS
            for difficulty in DIFFICULTIES
            for gold in GOLD_LABELS
            for swap in SWAP_VARIANTS
        }),
        f"{label} task×difficulty×gold×swap denominators/zero cells changed",
    )
    require(
        len(semantic_groups) == N_SEMANTIC_PAIRS,
        f"{label} requires exactly {N_SEMANTIC_PAIRS} semantic twin pairs",
    )
    for semantic_id, pair in semantic_groups.items():
        require(
            len(pair) == 2,
            f"{label} semantic twin {semantic_id!r} does not have exactly 2 rows",
        )
        require(
            {row["swap_variant"] for row in pair} == set(SWAP_VARIANTS),
            f"{label} semantic twin {semantic_id!r} lacks original/swapped surfaces",
        )
        require(
            {row["gold_label"] for row in pair} == set(GOLD_LABELS),
            f"{label} semantic twin {semantic_id!r} does not flip gold A/B",
        )
        require(
            len({row["task"] for row in pair}) == 1
            and len({row["difficulty"] for row in pair}) == 1,
            f"{label} semantic twin {semantic_id!r} changes task/difficulty",
        )
    return output


def _paired_matrices(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = _validate_arm(baseline_rows, "external_bundle_A")
    candidate = _validate_arm(candidate_rows, "external_bundle_B")
    require(
        set(baseline) == set(candidate),
        "external bundle A/B paired item keys differ",
    )

    overall = _empty_matrix()
    by_task = _new_matrix_map(TASKS)
    by_difficulty = _new_matrix_map(DIFFICULTIES)
    by_gold = _new_matrix_map(GOLD_LABELS)
    by_swap = _new_matrix_map(SWAP_VARIANTS)
    by_gold_swap = {
        gold: {swap: _empty_matrix() for swap in SWAP_VARIANTS}
        for gold in GOLD_LABELS
    }

    for item_id in sorted(baseline):
        left = baseline[item_id]
        right = candidate[item_id]
        require(
            all(left[field] == right[field] for field in METADATA_FIELDS),
            f"paired external-bundle metadata changed between arms: {item_id}",
        )
        source = left["state"]
        target = right["state"]
        task = left["task"]
        difficulty = left["difficulty"]
        gold = left["gold_label"]
        swap = left["swap_variant"]
        overall[source][target] += 1
        by_task[task][source][target] += 1
        by_difficulty[difficulty][source][target] += 1
        by_gold[gold][source][target] += 1
        by_swap[swap][source][target] += 1
        by_gold_swap[gold][swap][source][target] += 1

    require(_matrix_total(overall) == N_PER_MODEL_STREAM, "overall N changed")
    require(
        all(_matrix_total(by_task[key]) == N_PER_TASK for key in TASKS),
        "task matrix denominator changed",
    )
    require(
        all(
            _matrix_total(by_difficulty[key]) == N_PER_DIFFICULTY
            for key in DIFFICULTIES
        ),
        "difficulty matrix denominator changed",
    )
    require(
        all(_matrix_total(by_gold[key]) == N_PER_GOLD_LABEL for key in GOLD_LABELS),
        "gold-label matrix denominator changed",
    )
    require(
        all(
            _matrix_total(by_swap[key]) == N_PER_SWAP_VARIANT
            for key in SWAP_VARIANTS
        ),
        "swap matrix denominator changed",
    )
    require(
        all(
            _matrix_total(by_gold_swap[gold][swap]) == N_PER_GOLD_SWAP
            for gold in GOLD_LABELS for swap in SWAP_VARIANTS
        ),
        "gold-by-swap matrix denominator changed",
    )
    return {
        "overall": overall,
        "by_task": by_task,
        "by_difficulty": by_difficulty,
        "by_gold_label": by_gold,
        "by_swap_variant": by_swap,
        "by_gold_and_swap": by_gold_swap,
    }


def _accounting(matrix: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    total = _matrix_total(matrix)
    baseline = {
        state: sum(int(matrix[state][target]) for target in STATES)
        for state in STATES
    }
    candidate = {
        state: sum(int(matrix[source][state]) for source in STATES)
        for state in STATES
    }
    require(
        sum(baseline.values()) == sum(candidate.values()) == total,
        "transition marginals changed the denominator",
    )
    delta_v = candidate["V"] - baseline["V"]
    r_c = baseline["C"] - candidate["C"]
    r_i_protocol = baseline["I_protocol"] - candidate["I_protocol"]
    r_i_sem = baseline["I_sem"] - candidate["I_sem"]
    delta_c = -r_c
    delta_i_protocol = -r_i_protocol
    delta_i_sem = -r_i_sem
    require(
        delta_v == r_c + r_i_protocol + r_i_sem,
        "four-category external accounting identity failed",
    )
    require(
        sum(baseline.values()) == total and sum(candidate.values()) == total,
        "V+C+I_protocol+I_sem=N failed",
    )
    return {
        "N": total,
        "external_bundle_A_counts": baseline,
        "external_bundle_B_counts": candidate,
        "delta_V": delta_v,
        "R_C": r_c,
        "R_I_protocol": r_i_protocol,
        "R_I_sem": r_i_sem,
        "delta_C": delta_c,
        "delta_I_protocol": delta_i_protocol,
        "delta_I_sem": delta_i_sem,
        "delta_C_plus_I_protocol": delta_c + delta_i_protocol,
        "partition_identity": "V+C+I_protocol+I_sem=N",
        "contrast_identity": "delta_V=R_C+R_I_protocol+R_I_sem",
        "identities_verified": True,
    }


def _paired_metrics(matrix: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    _matrix_total(matrix)
    i_sem_to_v = int(matrix["I_sem"]["V"])
    v_to_i_sem = int(matrix["V"]["I_sem"])
    net = i_sem_to_v - v_to_i_sem
    leakage_terms = {
        "I_sem_to_C": int(matrix["I_sem"]["C"]),
        "I_sem_to_I_protocol": int(matrix["I_sem"]["I_protocol"]),
        "V_to_C": int(matrix["V"]["C"]),
        "V_to_I_protocol": int(matrix["V"]["I_protocol"]),
    }
    bad_leakage = sum(leakage_terms.values())
    return {
        "I_sem_A_to_V_B": i_sem_to_v,
        "V_A_to_I_sem_B": v_to_i_sem,
        "net_I_sem_to_V_minus_V_to_I_sem": net,
        "bad_leakage_terms": leakage_terms,
        "bad_paired_leakage": bad_leakage,
        "bad_paired_leakage_formula": (
            "I_sem->C + I_sem->I_protocol + V->C + V->I_protocol"
        ),
    }


def _stratum_checks(
    matrices: Mapping[str, Mapping[str, Mapping[str, Any]]],
    expected_n: int,
    label: str,
) -> tuple[dict[str, Any], bool]:
    results: dict[str, Any] = {}
    for key, matrix in matrices.items():
        accounting = _accounting(matrix)
        require(accounting["N"] == expected_n, f"{label} {key} N changed")
        checks = {
            "delta_V_at_least_13": (
                accounting["delta_V"] >= STRATUM_DELTA_V_MIN
            ),
            "delta_C_plus_I_protocol_at_most_6": (
                accounting["delta_C_plus_I_protocol"]
                <= STRATUM_NON_TARGET_INCREASE_MAX
            ),
        }
        results[key] = {
            "accounting": accounting,
            "checks": checks,
            "passed": all(checks.values()),
        }
    return results, all(result["passed"] for result in results.values())


def evaluate_model_stream(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate one model×stream without accepting caller thresholds."""
    _assert_cpu_import_boundary()
    matrices = _paired_matrices(baseline_rows, candidate_rows)
    overall = _accounting(matrices["overall"])
    paired = _paired_metrics(matrices["overall"])
    task_accounting = {
        task: _accounting(matrices["by_task"][task]) for task in TASKS
    }
    difficulty_accounting = {
        difficulty: _accounting(matrices["by_difficulty"][difficulty])
        for difficulty in DIFFICULTIES
    }
    qualifying_tasks = sorted(
        task for task in TASKS
        if task_accounting[task]["delta_V"] >= TASK_DELTA_V_MIN
    )

    primary_checks = {
        "delta_V_at_least_26": overall["delta_V"] >= DELTA_V_MIN,
        "R_I_sem_at_least_26": overall["R_I_sem"] >= R_I_SEM_MIN,
        "delta_C_at_most_12": overall["delta_C"] <= NON_TARGET_INCREASE_MAX,
        "delta_I_protocol_at_most_12": (
            overall["delta_I_protocol"] <= NON_TARGET_INCREASE_MAX
        ),
        "delta_C_plus_I_protocol_at_most_12": (
            overall["delta_C_plus_I_protocol"] <= NON_TARGET_INCREASE_MAX
        ),
        "paired_I_sem_to_V_at_least_26": (
            paired["I_sem_A_to_V_B"] >= PAIRED_I_SEM_TO_V_MIN
        ),
        "paired_net_at_least_26": (
            paired["net_I_sem_to_V_minus_V_to_I_sem"] >= PAIRED_NET_MIN
        ),
        "bad_paired_leakage_at_most_12": (
            paired["bad_paired_leakage"] <= PAIRED_LEAKAGE_MAX
        ),
    }
    coverage_checks = {
        "at_least_6_tasks_delta_V_at_least_3": (
            len(qualifying_tasks) >= TASKS_REQUIRED
        ),
        "every_task_delta_V_at_least_minus_2": all(
            task_accounting[task]["delta_V"] >= TASK_FLOOR for task in TASKS
        ),
        "easy_delta_V_at_least_0": (
            difficulty_accounting["easy"]["delta_V"] >= DIFFICULTY_FLOOR
        ),
        "hard_delta_V_at_least_0": (
            difficulty_accounting["hard"]["delta_V"] >= DIFFICULTY_FLOOR
        ),
    }
    gold_results, gold_passed = _stratum_checks(
        matrices["by_gold_label"], N_PER_GOLD_LABEL, "gold label",
    )
    swap_results, swap_passed = _stratum_checks(
        matrices["by_swap_variant"], N_PER_SWAP_VARIANT, "swap variant",
    )
    b_gold_c = {
        gold: gold_results[gold]["accounting"]["external_bundle_B_counts"]["C"]
        for gold in GOLD_LABELS
    }
    b_gold_c_abs_diff = abs(b_gold_c["A"] - b_gold_c["B"])
    label_balance_checks = {
        "both_gold_strata_pass": gold_passed,
        "both_swap_strata_pass": swap_passed,
        "bundle_B_abs_C_gold_A_minus_B_at_most_12": (
            b_gold_c_abs_diff <= B_GOLD_C_ABS_DIFF_MAX
        ),
    }
    passed = (
        all(primary_checks.values())
        and all(coverage_checks.values())
        and all(label_balance_checks.values())
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "scientific_object": "external_configuration_bundle_contrast",
        "internal_mechanism_claim": False,
        "overall_accounting": overall,
        "paired_metrics": paired,
        "task_accounting": task_accounting,
        "difficulty_accounting": difficulty_accounting,
        "gold_strata": gold_results,
        "swap_strata": swap_results,
        "bundle_B_C_by_gold_label": b_gold_c,
        "bundle_B_abs_C_gold_A_minus_B": b_gold_c_abs_diff,
        "qualifying_tasks": qualifying_tasks,
        "primary_checks": primary_checks,
        "coverage_checks": coverage_checks,
        "label_and_swap_checks": label_balance_checks,
        "external_bundle_stream_passed": passed,
        "secondary_can_set_primary": False,
        "pooling_used": False,
        "transition_matrices": matrices,
        "transition_matrix_sha256": sha256_json(matrices),
    }


def evaluate_model(
    stream_pairs: Mapping[
        str, tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]
    ],
) -> dict[str, Any]:
    """Conjoin three unpooled streams for one model."""
    require(isinstance(stream_pairs, Mapping), "model streams are not a mapping")
    require(
        set(stream_pairs) == set(STREAMS),
        f"exact frozen streams required: {STREAMS}; pooling/overrides forbidden",
    )
    results: dict[str, Any] = {}
    for stream in STREAMS:
        pair = stream_pairs[stream]
        require(
            isinstance(pair, (tuple, list)) and len(pair) == 2,
            f"{stream} must contain exactly external bundle A/B rows; "
            "threshold metadata and pooling are forbidden",
        )
        results[stream] = evaluate_model_stream(pair[0], pair[1])

    common_tasks = set(TASKS)
    for stream in STREAMS:
        common_tasks &= set(results[stream]["qualifying_tasks"])
    common_tasks_sorted = sorted(common_tasks)
    all_streams_passed = all(
        results[stream]["external_bundle_stream_passed"] for stream in STREAMS
    )
    common_task_gate = len(common_tasks_sorted) >= TASKS_REQUIRED
    return {
        "stream_results": results,
        "all_three_streams_passed": all_streams_passed,
        "common_qualifying_tasks": common_tasks_sorted,
        "common_qualifying_task_count": len(common_tasks_sorted),
        "common_task_gate_passed": common_task_gate,
        "model_pass": all_streams_passed and common_task_gate,
        "model_formula": "all3(stream_primary) AND common_tasks>=6",
        "pooling_used": False,
        "secondary_can_set_primary": False,
    }


def evaluate_cross_models(
    model_stream_pairs: Mapping[
        str,
        Mapping[
            str, tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]
        ],
    ],
) -> dict[str, Any]:
    """Conjoin three separately passing models; never pool their rows/counts."""
    _assert_cpu_import_boundary()
    require(isinstance(model_stream_pairs, Mapping), "models are not a mapping")
    require(
        set(model_stream_pairs) == set(MODELS),
        f"exact frozen models required: {MODELS}; model pooling/overrides forbidden",
    )
    model_results = {
        model: evaluate_model(model_stream_pairs[model]) for model in MODELS
    }
    common_tasks = set(TASKS)
    for model in MODELS:
        for stream in STREAMS:
            common_tasks &= set(
                model_results[model]["stream_results"][stream]["qualifying_tasks"]
            )
    common_tasks_sorted = sorted(common_tasks)
    all_models_passed = all(model_results[model]["model_pass"] for model in MODELS)
    common_task_gate = len(common_tasks_sorted) >= TASKS_REQUIRED
    cross_passed = all_models_passed and common_task_gate
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "model_results": model_results,
        "all_three_models_passed": all_models_passed,
        "cross_model_common_qualifying_tasks": common_tasks_sorted,
        "cross_model_common_qualifying_task_count": len(common_tasks_sorted),
        "cross_model_common_task_gate_passed": common_task_gate,
        "cross_model_pass": cross_passed,
        "cross_model_formula": "all3(model_pass) AND all9_common_tasks>=6",
        "pooling_used": False,
        "secondary_can_set_primary": False,
        "claim_scope": (
            "external configuration bundle behavior only; no shared internal "
            "thinking mechanism is inferred"
        ),
    }


# ---------------------------------------------------------------------------
# Synthetic fail-closed audit
# ---------------------------------------------------------------------------


Pair = tuple[list[dict[str, str]], list[dict[str, str]]]


def _metadata_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task in TASKS:
        for difficulty in DIFFICULTIES:
            for pair_index in range(8):
                semantic_id = f"{task}|{difficulty}|semantic_{pair_index:02d}"
                original_gold = "A" if pair_index % 2 == 0 else "B"
                swapped_gold = "B" if original_gold == "A" else "A"
                for swap, gold in (
                    ("original", original_gold),
                    ("swapped", swapped_gold),
                ):
                    rows.append({
                        "id": f"{semantic_id}|{swap}",
                        "semantic_id": semantic_id,
                        "task": task,
                        "difficulty": difficulty,
                        "gold_label": gold,
                        "swap_variant": swap,
                        "state": "V",
                    })
    require(len(rows) == N_PER_MODEL_STREAM, "synthetic metadata N changed")
    return rows


def _balanced_improvement_ids(task_targets: Sequence[int]) -> list[str]:
    require(len(task_targets) == len(TASKS), "synthetic task vector length changed")
    require(sum(task_targets) == DELTA_V_MIN, "synthetic benefit must total 26")
    require(all(0 <= value <= 5 for value in task_targets), "bad task target")
    rows = _metadata_rows()
    by_cell: defaultdict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for row in rows:
        by_cell[
            (row["task"], row["difficulty"], row["swap_variant"], row["gold_label"])
        ].append(row["id"])

    odd_seen = 0
    cell_targets: list[tuple[str, str, int]] = []
    for task, target in zip(TASKS, task_targets):
        easy = target // 2
        if target % 2:
            if odd_seen % 2 == 0:
                easy += 1
            odd_seen += 1
        cell_targets.append((task, "easy", easy))
        cell_targets.append((task, "hard", target - easy))
    require(sum(count for _, _, count in cell_targets) == 26, "cell target sum changed")
    require(
        sum(count for _, difficulty, count in cell_targets if difficulty == "easy")
        == 13,
        "synthetic easy benefit is not 13",
    )

    # Across 26 selections this cycle yields original/swapped=13/13 and A/B=13/13.
    combo_cycle = (
        ("original", "A"),
        ("swapped", "B"),
        ("original", "B"),
        ("swapped", "A"),
    )
    selected: list[str] = []
    cursor = 0
    for task, difficulty, count in cell_targets:
        used_in_cell: Counter[tuple[str, str]] = Counter()
        for _ in range(count):
            swap, gold = combo_cycle[cursor % len(combo_cycle)]
            cursor += 1
            key = (task, difficulty, swap, gold)
            index = used_in_cell[(swap, gold)]
            require(index < len(by_cell[key]), "synthetic combo exhausted")
            selected.append(by_cell[key][index])
            used_in_cell[(swap, gold)] += 1
    require(len(selected) == 26 and len(set(selected)) == 26, "selection invalid")
    selected_rows = [next(row for row in rows if row["id"] == item_id) for item_id in selected]
    require(Counter(row["gold_label"] for row in selected_rows) == Counter(A=13, B=13),
            "synthetic gold benefit is not balanced")
    require(
        Counter(row["swap_variant"] for row in selected_rows)
        == Counter(original=13, swapped=13),
        "synthetic swap benefit is not balanced",
    )
    return selected


def _positive_pair(
    task_targets: Sequence[int] = (5, 5, 4, 4, 4, 4, 0, 0),
    source_state: str = "I_sem",
) -> Pair:
    require(source_state in STATES, "synthetic source state invalid")
    baseline = _metadata_rows()
    candidate = copy.deepcopy(baseline)
    selected = set(_balanced_improvement_ids(task_targets))
    for row in baseline:
        if row["id"] in selected:
            row["state"] = source_state
    # Candidate remains V for the selected improvement rows.
    return baseline, candidate


def _clone_pair(pair: Pair) -> Pair:
    return copy.deepcopy(pair[0]), copy.deepcopy(pair[1])


def _rows_by_id(pair: Pair) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    return (
        {row["id"]: row for row in pair[0]},
        {row["id"]: row for row in pair[1]},
    )


def _set_transition(pair: Pair, item_id: str, source: str, target: str) -> None:
    require(source in STATES and target in STATES, "synthetic transition invalid")
    baseline, candidate = _rows_by_id(pair)
    require(item_id in baseline and item_id in candidate, "synthetic item missing")
    baseline[item_id]["state"] = source
    candidate[item_id]["state"] = target


def _eligible_ids(
    pair: Pair,
    predicate: Callable[[Mapping[str, str]], bool] | None = None,
    transition: tuple[str, str] = ("V", "V"),
) -> list[str]:
    baseline, candidate = _rows_by_id(pair)
    predicate = predicate or (lambda _: True)
    return sorted(
        item_id for item_id, row in baseline.items()
        if row["state"] == transition[0]
        and candidate[item_id]["state"] == transition[1]
        and predicate(row)
    )


def _add_balanced_leakage(pair: Pair, count: int, target: str = "C") -> None:
    require(target in ("C", "I_protocol"), "bad leakage target")
    combo_cycle = (
        ("original", "A"),
        ("swapped", "B"),
        ("original", "B"),
        ("swapped", "A"),
    )
    used: set[str] = set()
    for index in range(count):
        swap, gold = combo_cycle[index % 4]
        choices = _eligible_ids(
            pair,
            lambda row, swap=swap, gold=gold: (
                row["swap_variant"] == swap and row["gold_label"] == gold
            ),
        )
        item_id = next(item for item in choices if item not in used)
        used.add(item_id)
        _set_transition(pair, item_id, "I_sem", target)


def _three_streams(pair: Pair) -> dict[str, Pair]:
    return {stream: _clone_pair(pair) for stream in STREAMS}


def _three_models(pair: Pair) -> dict[str, dict[str, Pair]]:
    return {model: _three_streams(pair) for model in MODELS}


def _expect_error(action: Callable[[], Any], label: str) -> str:
    try:
        action()
    except RuntimeError as exc:
        require(str(exc), f"{label} produced an empty fail-closed message")
        return str(exc)
    raise RuntimeError(f"{label} unexpectedly passed")


def _move_improvement_between_strata(
    pair: Pair, source_field: str, source_value: str, target_value: str,
) -> None:
    baseline, candidate = _rows_by_id(pair)
    selected = [
        row for row in baseline.values()
        if row["state"] == "I_sem"
        and candidate[row["id"]]["state"] == "V"
        and row[source_field] == source_value
    ]
    for old in selected:
        choices = [
            row for row in baseline.values()
            if row["state"] == "V"
            and candidate[row["id"]]["state"] == "V"
            and row[source_field] == target_value
            and row["task"] == old["task"]
            and row["difficulty"] == old["difficulty"]
            and all(
                row[field] == old[field]
                for field in ("gold_label", "swap_variant")
                if field != source_field
            )
        ]
        if choices:
            _set_transition(pair, old["id"], "V", "V")
            _set_transition(pair, choices[0]["id"], "I_sem", "V")
            return
    raise RuntimeError(f"unable to move synthetic improvement across {source_field}")


def _candidate_gold_c_difference_pair(candidate_difference: int) -> Pair:
    require(candidate_difference in (12, 13), "synthetic label difference unsupported")
    pair = _positive_pair()
    # All candidate C rows use gold A.  Existing C->C rows keep marginal harm
    # at six while permitting candidate-B C label difference 12 or 13.
    original_total = (candidate_difference + 1) // 2
    swapped_total = candidate_difference // 2
    baseline_c_total = candidate_difference - 6
    original_baseline_c = (baseline_c_total + 1) // 2
    swapped_baseline_c = baseline_c_total // 2
    for swap, total, stable in (
        ("original", original_total, original_baseline_c),
        ("swapped", swapped_total, swapped_baseline_c),
    ):
        ids = _eligible_ids(
            pair,
            lambda row, swap=swap: (
                row["gold_label"] == "A" and row["swap_variant"] == swap
            ),
        )
        require(len(ids) >= total, "not enough synthetic gold-A rows")
        for item_id in ids[:stable]:
            _set_transition(pair, item_id, "C", "C")
        for item_id in ids[stable:total]:
            _set_transition(pair, item_id, "I_sem", "C")
    return pair


def self_test() -> dict[str, Any]:
    """Run synthetic positive and adversarial cases without reading/writing data."""
    _assert_cpu_import_boundary()
    cases: dict[str, bool] = {}
    errors: dict[str, str] = {}

    positive = _positive_pair()
    positive_stream = evaluate_model_stream(*positive)
    positive_cross = evaluate_cross_models(_three_models(positive))
    cases["exact_primary_boundary_passes"] = (
        positive_stream["external_bundle_stream_passed"]
        and positive_stream["overall_accounting"]["delta_V"] == 26
        and positive_stream["overall_accounting"]["R_I_sem"] == 26
        and positive_stream["paired_metrics"]["I_sem_A_to_V_B"] == 26
        and positive_stream["paired_metrics"][
            "net_I_sem_to_V_minus_V_to_I_sem"
        ] == 26
    )
    cases["positive_model_and_cross_model_pass"] = positive_cross["cross_model_pass"]
    cases["matrix_preserves_all_zero_cells"] = all(
        set(positive_stream["transition_matrices"]["overall"][state]) == set(STATES)
        for state in STATES
    ) and positive_stream["transition_matrices"]["overall"]["C"]["C"] == 0
    cases["accounting_identities_verified"] = bool(
        positive_stream["overall_accounting"]["identities_verified"]
    )

    benefit25 = _clone_pair(positive)
    baseline25, candidate25 = _rows_by_id(benefit25)
    removed = next(
        item_id for item_id, row in baseline25.items()
        if row["state"] == "I_sem" and candidate25[item_id]["state"] == "V"
    )
    _set_transition(benefit25, removed, "V", "V")
    cases["delta_V_25_rejected"] = not evaluate_model_stream(
        *benefit25
    )["external_bundle_stream_passed"]

    harm12 = _clone_pair(positive)
    _add_balanced_leakage(harm12, 12, "C")
    harm12_result = evaluate_model_stream(*harm12)
    cases["harm_and_leakage_12_boundary_passes"] = (
        harm12_result["external_bundle_stream_passed"]
        and harm12_result["overall_accounting"]["delta_C"] == 12
        and harm12_result["paired_metrics"]["bad_paired_leakage"] == 12
    )
    harm13 = _clone_pair(positive)
    _add_balanced_leakage(harm13, 13, "C")
    cases["delta_C_13_rejected"] = not evaluate_model_stream(
        *harm13
    )["external_bundle_stream_passed"]

    protocol12 = _clone_pair(positive)
    _add_balanced_leakage(protocol12, 12, "I_protocol")
    cases["I_protocol_harm_12_boundary_passes"] = evaluate_model_stream(
        *protocol12
    )["external_bundle_stream_passed"]
    protocol13 = _clone_pair(positive)
    _add_balanced_leakage(protocol13, 13, "I_protocol")
    cases["I_protocol_harm_13_rejected"] = not evaluate_model_stream(
        *protocol13
    )["external_bundle_stream_passed"]

    combined13 = _clone_pair(positive)
    _add_balanced_leakage(combined13, 7, "C")
    # Add six protocol leakages in still-neutral rows, balanced by construction.
    _add_balanced_leakage(combined13, 6, "I_protocol")
    combined13_result = evaluate_model_stream(*combined13)
    cases["combined_non_target_13_rejected"] = (
        not combined13_result["external_bundle_stream_passed"]
        and combined13_result["overall_accounting"][
            "delta_C_plus_I_protocol"
        ] == 13
    )

    net25 = _clone_pair(positive)
    b_net, c_net = _rows_by_id(net25)
    anchor = next(
        row for row in b_net.values()
        if row["state"] == "V" and c_net[row["id"]]["state"] == "V"
    )
    same_meta = lambda row: all(
        row[field] == anchor[field]
        for field in ("task", "difficulty", "gold_label", "swap_variant")
    )
    ids = _eligible_ids(net25, same_meta)
    require(len(ids) >= 3, "not enough rows for net=25 case")
    _set_transition(net25, ids[0], "V", "I_sem")
    _set_transition(net25, ids[1], "I_sem", "C")
    _set_transition(net25, ids[2], "C", "V")
    net25_result = evaluate_model_stream(*net25)
    cases["paired_net_25_rejected_despite_marginal_26"] = (
        not net25_result["external_bundle_stream_passed"]
        and net25_result["overall_accounting"]["delta_V"] == 26
        and net25_result["overall_accounting"]["R_I_sem"] == 26
        and net25_result["paired_metrics"][
            "net_I_sem_to_V_minus_V_to_I_sem"
        ] == 25
    )

    gross25 = _clone_pair(positive)
    b_gross, c_gross = _rows_by_id(gross25)
    selected = next(
        row for row in b_gross.values()
        if row["state"] == "I_sem" and c_gross[row["id"]]["state"] == "V"
    )
    replacement = next(
        row for row in b_gross.values()
        if row["state"] == "V"
        and c_gross[row["id"]]["state"] == "V"
        and all(
            row[field] == selected[field]
            for field in ("task", "difficulty", "gold_label", "swap_variant")
        )
    )
    _set_transition(gross25, selected["id"], "I_sem", "C")
    _set_transition(gross25, replacement["id"], "C", "V")
    gross25_result = evaluate_model_stream(*gross25)
    cases["paired_gross_25_rejected_despite_marginal_26"] = (
        not gross25_result["external_bundle_stream_passed"]
        and gross25_result["overall_accounting"]["delta_V"] == 26
        and gross25_result["paired_metrics"]["I_sem_A_to_V_B"] == 25
    )

    gross_leak13 = _clone_pair(positive)
    _add_balanced_leakage(gross_leak13, 13, "C")
    # Cycle item 0 is original/gold-A, the overfull gold and swap strata.
    offset = next(
        item_id for item_id in _eligible_ids(
            gross_leak13,
            lambda row: row["gold_label"] == "A"
            and row["swap_variant"] == "original",
        )
    )
    _set_transition(gross_leak13, offset, "C", "I_sem")
    gross_leak13_result = evaluate_model_stream(*gross_leak13)
    cases["gross_leakage_13_rejected_when_marginal_harm_12"] = (
        not gross_leak13_result["external_bundle_stream_passed"]
        and gross_leak13_result["overall_accounting"][
            "delta_C_plus_I_protocol"
        ] == 12
        and gross_leak13_result["paired_metrics"]["bad_paired_leakage"] == 13
    )

    protocol_source = _positive_pair(source_state="I_protocol")
    cases["I_protocol_rescue_cannot_impersonate_semantic_rescue"] = (
        not evaluate_model_stream(*protocol_source)["external_bundle_stream_passed"]
    )
    censor_source = _positive_pair(source_state="C")
    censor_result = evaluate_model_stream(*censor_source)
    cases["secondary_censor_like_success_cannot_set_primary"] = (
        censor_result["overall_accounting"]["delta_V"] == 26
        and censor_result["overall_accounting"]["R_C"] == 26
        and not censor_result["external_bundle_stream_passed"]
        and censor_result["secondary_can_set_primary"] is False
    )

    gold_benefit12 = _clone_pair(positive)
    _move_improvement_between_strata(gold_benefit12, "gold_label", "A", "B")
    gold_benefit_result = evaluate_model_stream(*gold_benefit12)
    cases["gold_stratum_delta_V_12_rejected"] = (
        not gold_benefit_result["external_bundle_stream_passed"]
        and gold_benefit_result["gold_strata"]["A"]["accounting"]["delta_V"] == 12
    )
    swap_benefit12 = _clone_pair(positive)
    _move_improvement_between_strata(
        swap_benefit12, "swap_variant", "original", "swapped",
    )
    swap_benefit_result = evaluate_model_stream(*swap_benefit12)
    cases["swap_stratum_delta_V_12_rejected"] = (
        not swap_benefit_result["external_bundle_stream_passed"]
        and swap_benefit_result["swap_strata"]["original"]["accounting"][
            "delta_V"
        ] == 12
    )

    gold_harm7 = _clone_pair(positive)
    ids = _eligible_ids(gold_harm7, lambda row: row["gold_label"] == "A")
    # Four original and three swapped rows: gold fails at 7, swap remains <=4.
    chosen = [
        *[item for item in ids if "|original" in item][:4],
        *[item for item in ids if "|swapped" in item][:3],
    ]
    for item_id in chosen:
        _set_transition(gold_harm7, item_id, "I_sem", "C")
    gold_harm_result = evaluate_model_stream(*gold_harm7)
    cases["gold_stratum_harm_7_rejected"] = (
        not gold_harm_result["external_bundle_stream_passed"]
        and gold_harm_result["gold_strata"]["A"]["accounting"][
            "delta_C_plus_I_protocol"
        ] == 7
    )

    swap_harm7 = _clone_pair(positive)
    ids = _eligible_ids(
        swap_harm7, lambda row: row["swap_variant"] == "original",
    )
    chosen = [
        *[item for item in ids if "semantic_00" in item or "semantic_02" in item][:4],
        *[item for item in ids if "semantic_01" in item or "semantic_03" in item][:3],
    ]
    require(len(chosen) == 7, "synthetic swap harm selection failed")
    for item_id in chosen:
        _set_transition(swap_harm7, item_id, "I_sem", "C")
    swap_harm_result = evaluate_model_stream(*swap_harm7)
    cases["swap_stratum_harm_7_rejected"] = (
        not swap_harm_result["external_bundle_stream_passed"]
        and swap_harm_result["swap_strata"]["original"]["accounting"][
            "delta_C_plus_I_protocol"
        ] == 7
    )

    label12 = evaluate_model_stream(*_candidate_gold_c_difference_pair(12))
    cases["candidate_B_gold_C_difference_12_boundary_passes"] = (
        label12["external_bundle_stream_passed"]
        and label12["bundle_B_abs_C_gold_A_minus_B"] == 12
    )
    label13 = evaluate_model_stream(*_candidate_gold_c_difference_pair(13))
    cases["candidate_B_gold_C_difference_13_rejected"] = (
        not label13["external_bundle_stream_passed"]
        and label13["bundle_B_abs_C_gold_A_minus_B"] == 13
    )

    task_floor = _clone_pair(positive)
    base_floor, cand_floor = _rows_by_id(task_floor)
    add_ids = _eligible_ids(task_floor, lambda row: row["task"] == TASKS[6])[:3]
    reverse_ids = _eligible_ids(task_floor, lambda row: row["task"] == TASKS[7])[:3]
    require(len(add_ids) == len(reverse_ids) == 3, "task-floor rows missing")
    available_reverse_ids = list(reverse_ids)
    for add_id in add_ids:
        # Match metadata strata across tasks so all global strata remain balanced.
        add_row = base_floor[add_id]
        match = next(
            item_id for item_id in available_reverse_ids
            if all(
                base_floor[item_id][field] == add_row[field]
                for field in ("difficulty", "gold_label", "swap_variant")
            )
        )
        _set_transition(task_floor, add_id, "I_sem", "V")
        _set_transition(task_floor, match, "V", "I_sem")
        available_reverse_ids.remove(match)
    task_floor_result = evaluate_model_stream(*task_floor)
    cases["task_delta_V_minus_3_rejected"] = (
        not task_floor_result["external_bundle_stream_passed"]
        and task_floor_result["task_accounting"][TASKS[7]]["delta_V"] == -3
    )

    # Each stream passes, but its six qualifying tasks differ; common intersection fails.
    stream_vectors = (
        (5, 5, 4, 4, 4, 4, 0, 0),
        (0, 0, 5, 5, 4, 4, 4, 4),
        (5, 5, 0, 0, 4, 4, 4, 4),
    )
    stream_mix = {
        stream: _positive_pair(vector)
        for stream, vector in zip(STREAMS, stream_vectors)
    }
    stream_mix_result = evaluate_model(stream_mix)
    cases["stream_common_task_intersection_required"] = (
        stream_mix_result["all_three_streams_passed"]
        and stream_mix_result["common_qualifying_task_count"] < 6
        and not stream_mix_result["model_pass"]
    )

    model_mix = {
        model: _three_streams(_positive_pair(vector))
        for model, vector in zip(MODELS, stream_vectors)
    }
    model_mix_result = evaluate_cross_models(model_mix)
    cases["cross_model_common_task_intersection_required"] = (
        model_mix_result["all_three_models_passed"]
        and model_mix_result["cross_model_common_qualifying_task_count"] < 6
        and not model_mix_result["cross_model_pass"]
    )

    pooled_stream = _clone_pair(positive)
    errors["stream_pooling"] = _expect_error(
        lambda: evaluate_model({
            "stream_0": (pooled_stream[0] * 3, pooled_stream[1] * 3),
        }),
        "stream pooling",
    )
    errors["model_pooling"] = _expect_error(
        lambda: evaluate_cross_models({"pooled": _three_streams(positive)}),
        "model pooling",
    )
    errors["threshold_override"] = _expect_error(
        lambda: evaluate_model({
            "stream_0": {
                "A": positive[0], "B": positive[1], "delta_V_min": 25,
            },
            "stream_1": positive,
            "stream_2": positive,
        }),
        "threshold override",
    )
    cases["stream_pooling_rejected"] = bool(errors["stream_pooling"])
    cases["model_pooling_rejected"] = bool(errors["model_pooling"])
    cases["threshold_relaxation_input_rejected"] = bool(errors["threshold_override"])

    missing_row = _clone_pair(positive)
    missing_row[1].pop()
    errors["missing_row"] = _expect_error(
        lambda: evaluate_model_stream(*missing_row), "missing row",
    )
    cases["missing_row_rejected"] = bool(errors["missing_row"])

    missing_zero_grid = _clone_pair(positive)
    for arm in missing_zero_grid:
        for row in arm:
            if row["task"] == TASKS[-1]:
                row["task"] = TASKS[-2]
    errors["missing_zero_grid"] = _expect_error(
        lambda: evaluate_model_stream(*missing_zero_grid), "missing task zero grid",
    )
    cases["missing_task_zero_grid_rejected"] = bool(errors["missing_zero_grid"])

    broken_twin = _clone_pair(positive)
    broken_twin[0][0]["semantic_id"] = broken_twin[0][2]["semantic_id"]
    broken_twin[1][0]["semantic_id"] = broken_twin[1][2]["semantic_id"]
    errors["broken_twin"] = _expect_error(
        lambda: evaluate_model_stream(*broken_twin), "broken semantic twin",
    )
    cases["broken_option_swap_twin_rejected"] = bool(errors["broken_twin"])

    metadata_drift = _clone_pair(positive)
    metadata_drift[1][0]["gold_label"] = "B"
    errors["metadata_drift"] = _expect_error(
        lambda: evaluate_model_stream(*metadata_drift), "arm metadata drift",
    )
    cases["arm_metadata_drift_rejected"] = bool(errors["metadata_drift"])

    unknown_state = _clone_pair(positive)
    unknown_state[1][0]["state"] = "UNKNOWN"
    errors["unknown_state"] = _expect_error(
        lambda: evaluate_model_stream(*unknown_state), "unknown state",
    )
    cases["unknown_terminal_state_rejected"] = bool(errors["unknown_state"])

    duplicate_id = _clone_pair(positive)
    duplicate_id[0][1]["id"] = duplicate_id[0][0]["id"]
    errors["duplicate_id"] = _expect_error(
        lambda: evaluate_model_stream(*duplicate_id), "duplicate ID",
    )
    cases["duplicate_item_id_rejected"] = bool(errors["duplicate_id"])

    require(len(cases) >= 20, "fewer than 20 synthetic cases were installed")
    failed = sorted(name for name, passed in cases.items() if not passed)
    require(not failed, f"Phase983 self-tests failed: {failed}")
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "passed": True,
        "cpu_only": True,
        "gpu_used": False,
        "model_weights_loaded": False,
        "generation_performed": False,
        "result_written": False,
        "old_holdout_imported": False,
        "mechanism_authorized": False,
        "scientific_object": "external configuration bundle behavior",
        "internal_thinking_mechanism_claim": False,
        "gate_contract_sha256": sha256_json(GATE_CONTRACT),
        "synthetic_case_count": len(cases),
        "tests": cases,
        "fail_closed_messages": errors,
        "stratification_compatibility": {
            "compatible_only_if": (
                "each arm has 128 semantic IDs with exactly original/swapped "
                "surfaces, opposite gold labels, exact 128/128 gold and swap "
                "strata, and complete task×difficulty×gold×swap cells"
            ),
            "incompatible_inputs_fail_closed": True,
            "twins_counted_as_independent_samples": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the synthetic fail-closed audit and print JSON (default)",
    )
    parser.add_argument(
        "--contract",
        action="store_true",
        help="include the full immutable gate contract in the JSON output",
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
