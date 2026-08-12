#!/usr/bin/env python3
"""Exact rule-extension identifiability audit after Phase1167.

This phase trains no network.  It counts held-out extensions under four
explicit hypothesis/constraint classes and keeps identifiability conditional
on the declared class.  The goal is to separate an underdetermined training
specification from a model that fails to learn a uniquely extending rule.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import deque
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1168_rule_extension_identifiability_audit_recompute.py"
P1167_SCRIPT = ROOT / "tests/glm5/phase1167_compositional_formation_axis.py"
P1167_FINAL = ROOT / "tests/glm5/result/phase1167_compositional_formation_axis/analysis/final.json"
P1167_AUDIT = ROOT / "tests/glm5/result/phase1167_compositional_formation_axis/audit/report.json"
OUT_ROOT = ROOT / "tests/glm5/result/phase1168_rule_extension_identifiability_audit"
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1167_compositional_formation_axis as p1167  # noqa: E402


p1163 = p1167.p1163
PHASE = 1168
CONSTRAINT_CLASSES = (
    "data_only",
    "train_internal_equivariance",
    "partition_internal_equivariance",
    "global_equivariance",
    "separable_additive",
)


def inverse_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(permutation)
    for source, target in enumerate(permutation):
        result[target] = source
    return tuple(result)


def cyclic_shift_permutation(order: int, shift: int = 1) -> tuple[int, ...]:
    return tuple((value + shift) % order for value in range(order))


def context_flip_permutation() -> tuple[int, ...]:
    return tuple(value ^ 4 for value in range(8))


def current_task() -> dict[str, Any]:
    nodes = tuple(
        (row, col, context)
        for context in range(2)
        for row in range(4)
        for col in range(4)
    )
    holdout = {
        node for node in nodes if node[0] in (0, 2) and node[1] in (0, 2)
    }
    shift = tuple(((value % 4) + 1) % 4 + 4 * (value // 4) for value in range(8))
    generators = (
        ("row_plus_one", lambda node: ((node[0] + 1) % 4, node[1], node[2]), shift),
        ("col_plus_one", lambda node: (node[0], (node[1] + 1) % 4, node[2]), shift),
        ("context_flip", lambda node: (node[0], node[1], 1 - node[2]), context_flip_permutation()),
    )
    return {
        "name": "z4_context_checkerboard",
        "nodes": nodes,
        "output_size": 8,
        "target": lambda node: (node[0] + node[1]) % 4 + 4 * node[2],
        "holdout": holdout,
        "generators": generators,
        "templates": len(p1167.p1166.source.TEMPLATES),
        "family": "z4_context_addition",
    }


def cyclic_task(order: int, geometry: str) -> dict[str, Any]:
    nodes = tuple((row, col) for row in range(order) for col in range(order))
    if geometry == "diagonal":
        holdout = {node for node in nodes if node[0] == node[1]}
    elif geometry == "double_diagonal":
        holdout = {
            node for node in nodes if (node[0] - node[1]) % order in (0, 1)
        }
    else:
        raise KeyError(geometry)
    shift = cyclic_shift_permutation(order)
    generators = (
        ("row_plus_one", lambda node: ((node[0] + 1) % order, node[1]), shift),
        ("col_plus_one", lambda node: (node[0], (node[1] + 1) % order), shift),
    )
    return {
        "name": f"z{order}_{geometry}",
        "nodes": nodes,
        "output_size": order,
        "target": lambda node: (node[0] + node[1]) % order,
        "holdout": holdout,
        "generators": generators,
        "templates": 1,
        "family": "cyclic_addition",
    }


def task_panel() -> tuple[dict[str, Any], ...]:
    return (
        current_task(),
        cyclic_task(5, "diagonal"),
        cyclic_task(7, "double_diagonal"),
    )


def build_edges(task: dict[str, Any], constraint_class: str) -> list[tuple[Any, Any, tuple[int, ...]]]:
    if constraint_class in ("data_only", "separable_additive"):
        return []
    nodes = set(task["nodes"])
    holdout = task["holdout"]
    edges = []
    for source in task["nodes"]:
        for _, transform, permutation in task["generators"]:
            target = transform(source)
            if target not in nodes:
                raise RuntimeError("generator left the finite domain")
            source_train = source not in holdout
            target_train = target not in holdout
            include = False
            if constraint_class == "train_internal_equivariance":
                include = source_train and target_train
            elif constraint_class == "partition_internal_equivariance":
                include = source_train == target_train
            elif constraint_class == "global_equivariance":
                include = True
            else:
                raise KeyError(constraint_class)
            if include:
                edges.append((source, target, permutation))
    return edges


def graph_extension_count(task: dict[str, Any], constraint_class: str) -> dict[str, Any]:
    nodes = task["nodes"]
    holdout = task["holdout"]
    output_size = task["output_size"]
    observed = {node: task["target"](node) for node in nodes if node not in holdout}
    adjacency: dict[Any, list[tuple[Any, tuple[int, ...]]]] = {node: [] for node in nodes}
    for source, target, permutation in build_edges(task, constraint_class):
        adjacency[source].append((target, permutation))
        adjacency[target].append((source, inverse_permutation(permutation)))

    visited: set[Any] = set()
    holdout_extension_count = 1
    component_rows = []
    for root in nodes:
        if root in visited:
            continue
        component = []
        queue = deque([root])
        visited.add(root)
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor, _ in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        holdout_nodes = tuple(sorted(node for node in component if node in holdout))
        allowed_holdout_assignments = set()
        consistent_roots = 0
        for root_value in range(output_size):
            assignment = {root: root_value}
            propagate = deque([root])
            consistent = True
            while propagate and consistent:
                node = propagate.popleft()
                for neighbor, permutation in adjacency[node]:
                    proposed = permutation[assignment[node]]
                    if neighbor in assignment and assignment[neighbor] != proposed:
                        consistent = False
                        break
                    if neighbor not in assignment:
                        assignment[neighbor] = proposed
                        propagate.append(neighbor)
            if consistent:
                consistent = all(
                    assignment[node] == value
                    for node, value in observed.items()
                    if node in assignment
                )
            if consistent:
                consistent_roots += 1
                if holdout_nodes:
                    allowed_holdout_assignments.add(
                        tuple(assignment[node] for node in holdout_nodes)
                    )
        if consistent_roots == 0:
            raise RuntimeError(
                f"inconsistent constraint system: {task['name']}/{constraint_class}"
            )
        if holdout_nodes:
            holdout_extension_count *= len(allowed_holdout_assignments)
        component_rows.append(
            {
                "node_count": len(component),
                "holdout_node_count": len(holdout_nodes),
                "observed_node_count": len(component) - len(holdout_nodes),
                "consistent_root_values": consistent_roots,
                "distinct_holdout_assignments": (
                    len(allowed_holdout_assignments) if holdout_nodes else 1
                ),
            }
        )
    return {
        "edge_count": len(build_edges(task, constraint_class)),
        "component_count": len(component_rows),
        "holdout_extension_count": holdout_extension_count,
        "identifiable": holdout_extension_count == 1,
        "components": component_rows,
    }


def separable_extension_count(task: dict[str, Any]) -> dict[str, Any]:
    """Count distinct holdout predictions in f(r,c,z)=(a_r+b_c mod n, h_z)."""
    name = task["name"]
    nodes = task["nodes"]
    holdout = task["holdout"]
    output_size = task["output_size"]
    if name == "z4_context_checkerboard":
        order = 4
        contexts = 2
        low_target = lambda node: task["target"](node) % 4
        high_target = lambda node: task["target"](node) // 4
    else:
        order = output_size
        contexts = 1
        low_target = task["target"]
        high_target = lambda node: 0

    observed = [node for node in nodes if node not in holdout]
    # The bipartite observed graph fixes all a_r+b_c values up to one gauge per component.
    variable_nodes = [("r", value) for value in range(order)] + [
        ("c", value) for value in range(order)
    ]
    adjacency: dict[tuple[str, int], list[tuple[tuple[str, int], int]]] = {
        node: [] for node in variable_nodes
    }
    seen_pairs = set()
    for node in observed:
        row, col = node[0], node[1]
        if (row, col) in seen_pairs:
            continue
        seen_pairs.add((row, col))
        target = low_target(node)
        adjacency[("r", row)].append((("c", col), target))
        adjacency[("c", col)].append((("r", row), target))

    components = []
    variable_component: dict[tuple[str, int], int] = {}
    offsets: dict[tuple[str, int], tuple[int, int]] = {}
    # Store each variable as sign*root + offset modulo order.
    for start in variable_nodes:
        if start in variable_component:
            continue
        component_index = len(components)
        variable_component[start] = component_index
        offsets[start] = (1, 0)
        queue = deque([start])
        members = []
        consistent = True
        while queue:
            current = queue.popleft()
            members.append(current)
            sign, offset = offsets[current]
            for neighbor, target in adjacency[current]:
                proposed = (-sign, (target - offset) % order)
                if neighbor in offsets and offsets[neighbor] != proposed:
                    consistent = False
                if neighbor not in offsets:
                    offsets[neighbor] = proposed
                    variable_component[neighbor] = component_index
                    queue.append(neighbor)
        if not consistent:
            raise RuntimeError(f"separable class inconsistent: {name}")
        components.append(members)

    high_options = []
    for context in range(contexts):
        values = {
            high_target(node)
            for node in observed
            if (node[2] if len(node) == 3 else 0) == context
        }
        if len(values) > 1:
            raise RuntimeError("inconsistent context output")
        high_options.append(tuple(values) if values else tuple(range(2)))

    distinct_holdout = set()
    for roots in itertools.product(range(order), repeat=len(components)):
        variable_values = {}
        for variable in variable_nodes:
            sign, offset = offsets[variable]
            root = roots[variable_component[variable]]
            variable_values[variable] = (sign * root + offset) % order
        for highs in itertools.product(*high_options):
            predictions = []
            for node in sorted(holdout):
                row, col = node[0], node[1]
                low = (
                    variable_values[("r", row)] + variable_values[("c", col)]
                ) % order
                context = node[2] if len(node) == 3 else 0
                predictions.append(low + (4 * highs[context] if contexts == 2 else 0))
            # Verify the represented function fits all observed points.
            valid = True
            for node in observed:
                row, col = node[0], node[1]
                low = (
                    variable_values[("r", row)] + variable_values[("c", col)]
                ) % order
                context = node[2] if len(node) == 3 else 0
                prediction = low + (4 * highs[context] if contexts == 2 else 0)
                if prediction != task["target"](node):
                    valid = False
                    break
            if valid:
                distinct_holdout.add(tuple(predictions))
    if not distinct_holdout:
        raise RuntimeError(f"no separable function fits {name}")
    return {
        "edge_count": len(seen_pairs),
        "component_count": len(components),
        "holdout_extension_count": len(distinct_holdout),
        "identifiable": len(distinct_holdout) == 1,
        "parameterization_note": "distinct holdout functions counted after quotienting additive gauge duplicates",
    }


def analyze_task(task: dict[str, Any]) -> dict[str, Any]:
    train = set(task["nodes"]) - task["holdout"]
    results = {}
    for constraint_class in CONSTRAINT_CLASSES:
        if constraint_class == "separable_additive":
            results[constraint_class] = separable_extension_count(task)
        else:
            results[constraint_class] = graph_extension_count(task, constraint_class)
    output_size = task["output_size"]
    expected_unrestricted = output_size ** len(task["holdout"])
    if results["data_only"]["holdout_extension_count"] != expected_unrestricted:
        raise RuntimeError("unrestricted extension count mismatch")
    return {
        "name": task["name"],
        "family": task["family"],
        "node_count": len(task["nodes"]),
        "train_count": len(train),
        "holdout_count": len(task["holdout"]),
        "output_size": output_size,
        "target_holdout_digest": p1163.digest(
            [task["target"](node) for node in sorted(task["holdout"])]
        ),
        "constraint_results": results,
        "token_level_unrestricted_extension_count": (
            output_size ** (len(task["holdout"]) * task["templates"])
        ),
        "templates": task["templates"],
    }


def protocol_command() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError("refusing to overwrite Phase1168")
    prior_final = p1163.read_json(P1167_FINAL)
    prior_audit = p1163.read_json(P1167_AUDIT)
    tasks = task_panel()
    checks = {
        "phase1167_closed": prior_final["branch_status"]
        == "closed_after_finite_formation_panel",
        "phase1167_no_auto_continue": not prior_final["auto_continue"],
        "phase1167_audit_passed": prior_audit["all_checks_passed"],
        "zero_model_phase": True,
        "three_frozen_task_geometries": len(tasks) == 3,
        "constraint_classes_frozen": len(CONSTRAINT_CLASSES) == 5,
        "all_holdouts_nonempty": all(task["holdout"] for task in tasks),
        "all_training_sets_nonempty": all(
            len(task["nodes"]) > len(task["holdout"]) for task in tasks
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol checks failed: {checks}")
    protocol = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "title": "exact version-space audit of rule-extension identifiability",
        "source_digests": {
            "phase1167_final": prior_final["final_digest"],
            "phase1167_audit": prior_audit["audit_digest"],
        },
        "source_hashes": {
            "primary_script": p1163.sha256_file(SCRIPT),
            "audit_script": p1163.sha256_file(AUDIT_SCRIPT),
            "phase1167_script": p1163.sha256_file(P1167_SCRIPT),
        },
        "task_panel": [
            {
                "name": task["name"],
                "family": task["family"],
                "node_count": len(task["nodes"]),
                "holdout_nodes": [list(node) for node in sorted(task["holdout"])],
                "output_size": task["output_size"],
                "templates": task["templates"],
                "generators": [row[0] for row in task["generators"]],
            }
            for task in tasks
        ],
        "constraint_classes": list(CONSTRAINT_CLASSES),
        "definitions": {
            "data_only": "training labels fixed; every held-out semantic node is otherwise free",
            "train_internal_equivariance": "only generator edges whose two endpoints are training nodes",
            "partition_internal_equivariance": "generator edges internal to training or held-out partitions; no crossing edge",
            "global_equivariance": "all generator edges, including train-to-holdout crossings",
            "separable_additive": "f(r,c,z)=(a_r+b_c mod n,h_z), counted after quotienting parameter gauge",
        },
        "primary_endpoint": "for every frozen task, local constraints admit multiple held-out extensions while global equivariance and separable-additive classes admit exactly one",
        "authorization_rule": "success authorizes a separate trajectory-formation protocol; it does not authorize hidden-state scanning or claim that a network learned either global class",
        "hard_stops": [
            "Identifiability is always reported conditional on the declared hypothesis class.",
            "Global constraints may be used for mathematical audit but not silently injected as unlabeled held-out training data.",
            "A unique extension under a hand-specified class does not show that gradient training selects that class.",
            "No new neural mechanism operator or K-level empirical mechanism is inferred from enumeration.",
            "The Phase1167 task, holdout, model outcomes, and branch status remain unchanged.",
        ],
        "checks": checks,
    }
    protocol["protocol_digest"] = p1163.digest(protocol)
    p1163.write_json(OUT_ROOT / "protocol/preregistration.json", protocol)
    print(p1163.canonical({"protocol_digest": protocol["protocol_digest"], "checks": checks}))


def verify_protocol() -> dict[str, Any]:
    protocol = p1163.read_json(OUT_ROOT / "protocol/preregistration.json")
    body = dict(protocol)
    stored = body.pop("protocol_digest")
    if p1163.digest(body) != stored:
        raise RuntimeError("protocol digest mismatch")
    for key, path in (
        ("primary_script", SCRIPT),
        ("audit_script", AUDIT_SCRIPT),
        ("phase1167_script", P1167_SCRIPT),
    ):
        if p1163.sha256_file(path) != protocol["source_hashes"][key]:
            raise RuntimeError(f"frozen source changed: {key}")
    return protocol


def run_command() -> None:
    protocol = verify_protocol()
    root = OUT_ROOT / "analysis"
    if root.exists():
        raise RuntimeError("refusing to overwrite analysis")
    rows = [analyze_task(task) for task in task_panel()]
    primary_cells = []
    for row in rows:
        results = row["constraint_results"]
        passed = bool(
            results["data_only"]["holdout_extension_count"] > 1
            and results["train_internal_equivariance"]["holdout_extension_count"] > 1
            and results["partition_internal_equivariance"]["holdout_extension_count"] > 1
            and results["global_equivariance"]["holdout_extension_count"] == 1
            and results["separable_additive"]["holdout_extension_count"] == 1
        )
        primary_cells.append({"task": row["name"], "passed": passed})
    primary_passed = all(cell["passed"] for cell in primary_cells)
    result = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "protocol_digest": protocol["protocol_digest"],
        "tasks": rows,
        "primary_cells": primary_cells,
        "primary_endpoint_passed": primary_passed,
        "trajectory_protocol_authorized": primary_passed,
        "hidden_state_scan_authorized": False,
        "conclusions": [
            "The Phase1167 training labels and implemented local equivariance constraints do not uniquely determine held-out behavior.",
            "Unique extension appears only after a global rule class is supplied; that is a class-conditional fact, not evidence that the network learned the class.",
            "Cross-partition generator edges are exactly what connect observed and held-out components, but using held-out inputs in training would change the generalization protocol.",
        ],
        "non_implications": [
            "Underdetermination does not imply that neural generalization is impossible.",
            "Global mathematical identifiability does not imply learnability by gradient descent.",
            "A unique separable extension does not imply a separable internal representation.",
            "This audit does not identify a hidden mechanism.",
        ],
    }
    result["analysis_digest"] = p1163.digest(result)
    p1163.write_json(root / "result.json", result)
    print(p1163.canonical(result))


def finalize_command() -> None:
    protocol = verify_protocol()
    result = p1163.read_json(OUT_ROOT / "analysis/result.json")
    auto_continue = bool(result["trajectory_protocol_authorized"])
    final = {
        "phase": PHASE,
        "created_at_utc": p1163.now(),
        "title": protocol["title"],
        "protocol_digest": protocol["protocol_digest"],
        "analysis_digest": result["analysis_digest"],
        "primary_endpoint_passed": result["primary_endpoint_passed"],
        "trajectory_protocol_authorized": auto_continue,
        "hidden_state_scan_authorized": False,
        "natural_mechanism_recovered": False,
        "branch_status": (
            "complete_authorizing_new_trajectory_protocol"
            if auto_continue
            else "complete_without_trajectory_authorization"
        ),
        "auto_continue": auto_continue,
        "auto_continue_reason": (
            "The finite audit found the preregistered local/global identifiability separation in every task; only a separate sealed trajectory protocol is authorized."
            if auto_continue
            else "The finite task panel did not establish the preregistered identifiability separation."
        ),
        "conclusions": result["conclusions"],
        "non_implications": result["non_implications"],
    }
    final["final_digest"] = p1163.digest(final)
    p1163.write_json(OUT_ROOT / "analysis/final.json", final)
    print(p1163.canonical(final))


def smoke_command() -> None:
    print(
        p1163.canonical(
            {
                "tasks": [task["name"] for task in task_panel()],
                "constraint_classes": list(CONSTRAINT_CLASSES),
                "phase1167_final_exists": P1167_FINAL.exists(),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("protocol", "run", "finalize", "smoke"))
    command = parser.parse_args().command
    {
        "protocol": protocol_command,
        "run": run_command,
        "finalize": finalize_command,
        "smoke": smoke_command,
    }[command]()


if __name__ == "__main__":
    main()
