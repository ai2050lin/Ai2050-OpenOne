#!/usr/bin/env python3
"""Apply frozen controls to the complete Phase380 causal layout scan."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase380_independent_layout_validation"
MODELS = ("qwen3", "glm4", "deepseek7b")
DEPTHS = ("early", "middle_early", "middle", "middle_late", "late")
ROLES = ("source", "query", "current")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    freeze = read_json(OUT / "phase380_causal_scan_freeze.json")
    gates = freeze["frozen_gates"]
    rows = []
    for model in MODELS:
        rows.extend(
            read_jsonl(OUT / "causal/private/models" / model / "phase380_causal_rows.jsonl")
        )
    by_condition = {
        (
            row["model"], row["mechanism_id"], row["contrast_axis"],
            row["anonymous_parallel_group_id"], row["transfer_name"],
            row["depth_name"], row["component_type"], row["position_role"],
            row["condition"],
        ): row
        for row in rows
    }
    direction_rows = []
    for key, natural in sorted(by_condition.items()):
        if key[-1] != "natural_swap":
            continue
        model, mechanism, axis, group, transfer, depth, component, role, _ = key
        energy = by_condition[(*key[:-1], "equal_energy_permutation")]
        wrong_depth = DEPTHS[(DEPTHS.index(depth) + 1) % len(DEPTHS)]
        wrong_role = ROLES[(ROLES.index(role) + 1) % len(ROLES)]
        depth_control = by_condition[
            (model, mechanism, axis, group, transfer, wrong_depth, component, role, "natural_swap")
        ]
        role_control = by_condition[
            (model, mechanism, axis, group, transfer, depth, component, wrong_role, "natural_swap")
        ]
        passed = bool(
            natural["transfer_gain"] >= gates["minimum_natural_transfer_gain"]
            and natural["transfer_gain"]
            >= energy["transfer_gain"] + gates["minimum_gain_over_equal_energy"]
            and natural["terminal_transfer_share"]
            >= gates["minimum_terminal_transfer_share"]
            and natural["terminal_transfer_share"]
            >= energy["terminal_transfer_share"]
            + gates["minimum_share_over_equal_energy"]
            and natural["transfer_gain"]
            >= depth_control["transfer_gain"]
            + gates["minimum_gain_over_cyclic_wrong_depth"]
            and natural["transfer_gain"]
            >= role_control["transfer_gain"]
            + gates["minimum_gain_over_cyclic_wrong_role"]
        )
        direction_rows.append(
            {
                "schema_version": "53.10.0",
                "phase_id": "Phase380-CausalLayoutAnalysis",
                "model": model,
                "mechanism_id": mechanism,
                "contrast_axis": axis,
                "anonymous_parallel_group_id": group,
                "transfer_name": transfer,
                "depth_name": depth,
                "component_type": component,
                "position_role": role,
                "natural_transfer_gain": natural["transfer_gain"],
                "equal_energy_transfer_gain": energy["transfer_gain"],
                "wrong_depth_transfer_gain": depth_control["transfer_gain"],
                "wrong_role_transfer_gain": role_control["transfer_gain"],
                "natural_terminal_transfer_share": natural[
                    "terminal_transfer_share"
                ],
                "equal_energy_terminal_transfer_share": energy[
                    "terminal_transfer_share"
                ],
                "direction_gate_pass": passed,
            }
        )
    transfer_names: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in direction_rows:
        transfer_names[(row["mechanism_id"], row["contrast_axis"])].add(
            row["transfer_name"]
        )
    group_cells: dict[tuple[Any, ...], dict[str, bool]] = defaultdict(dict)
    for row in direction_rows:
        key = (
            row["model"], row["mechanism_id"], row["contrast_axis"],
            row["anonymous_parallel_group_id"], row["depth_name"],
            row["component_type"], row["position_role"],
        )
        group_cells[key][row["transfer_name"]] = row["direction_gate_pass"]
    cell_groups: dict[tuple[Any, ...], list[bool]] = defaultdict(list)
    for key, transfer_passes in group_cells.items():
        model, mechanism, axis, _group, depth, component, role = key
        expected = transfer_names[(mechanism, axis)]
        group_pass = set(transfer_passes) == expected and all(transfer_passes.values())
        cell_groups[(model, mechanism, axis, depth, component, role)].append(group_pass)
    model_cells = []
    for key, values in sorted(cell_groups.items()):
        model, mechanism, axis, depth, component, role = key
        pass_count = sum(values)
        model_cells.append(
            {
                "schema_version": "53.10.0",
                "phase_id": "Phase380-CausalLayoutAnalysis",
                "model": model,
                "mechanism_id": mechanism,
                "contrast_axis": axis,
                "depth_name": depth,
                "component_type": component,
                "position_role": role,
                "group_count": len(values),
                "all_four_direction_group_pass_count": pass_count,
                "model_cell_pass": pass_count
                >= gates["minimum_groups_all_four_directions"],
                "terminal_interface_cell": depth == "late"
                and component in {"layer_input", "layer_output"}
                and role == "current",
                "terminal_endpoint_cell": depth == "late"
                and component == "layer_output"
                and role == "current",
            }
        )
    crossmodel_cells = []
    cell_models: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in model_cells:
        if row["model_cell_pass"]:
            cell_models[
                (
                    row["mechanism_id"], row["contrast_axis"], row["depth_name"],
                    row["component_type"], row["position_role"],
                )
            ].add(row["model"])
    for key, models in sorted(cell_models.items()):
        mechanism, axis, depth, component, role = key
        level2 = "glm4" in models and bool(models & {"qwen3", "deepseek7b"})
        crossmodel_cells.append(
            {
                "schema_version": "53.10.0",
                "phase_id": "Phase380-CausalLayoutAnalysis",
                "mechanism_id": mechanism,
                "contrast_axis": axis,
                "depth_name": depth,
                "component_type": component,
                "position_role": role,
                "models": sorted(models),
                "heterogeneous_level2": level2,
                "level3": models == set(MODELS),
                "terminal_interface_cell": depth == "late"
                and component in {"layer_input", "layer_output"}
                and role == "current",
                "terminal_endpoint_cell": depth == "late"
                and component == "layer_output"
                and role == "current",
                "upstream_path_claimed": False,
            }
        )
    level2 = [row for row in crossmodel_cells if row["heterogeneous_level2"]]
    terminal_interface = [row for row in level2 if row["terminal_interface_cell"]]
    upstream = [row for row in level2 if not row["terminal_interface_cell"]]
    shared_cells: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in upstream:
        shared_cells[
            (row["depth_name"], row["component_type"], row["position_role"])
        ].add(row["mechanism_id"])
    shared = [
        {
            "depth_name": key[0],
            "component_type": key[1],
            "position_role": key[2],
            "mechanisms": sorted(mechanisms),
            "cross_mechanism_causal_territory_candidate": len(mechanisms) >= 2,
            "same_neurons_established": False,
        }
        for key, mechanisms in sorted(shared_cells.items())
        if len(mechanisms) >= 2
    ]
    shared_terminal_cells: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in terminal_interface:
        shared_terminal_cells[
            (row["depth_name"], row["component_type"], row["position_role"])
        ].add(row["mechanism_id"])
    shared_terminal = [
        {
            "depth_name": key[0],
            "component_type": key[1],
            "position_role": key[2],
            "mechanisms": sorted(mechanisms),
            "cross_mechanism_terminal_interface": len(mechanisms) >= 2,
            "upstream_language_rule_established": False,
            "same_neurons_established": False,
        }
        for key, mechanisms in sorted(shared_terminal_cells.items())
        if len(mechanisms) >= 2
    ]
    write_jsonl(OUT / "causal/phase380_direction_gate_rows.jsonl", direction_rows)
    write_jsonl(OUT / "causal/phase380_model_cell_rows.jsonl", model_cells)
    write_jsonl(OUT / "causal/phase380_crossmodel_cell_rows.jsonl", crossmodel_cells)
    write_jsonl(OUT / "causal/phase380_shared_territory_rows.jsonl", shared)
    write_jsonl(
        OUT / "causal/phase380_shared_terminal_interface_rows.jsonl",
        shared_terminal,
    )
    natural_gains = [row["natural_transfer_gain"] for row in direction_rows]
    summary = {
        "schema_version": "53.10.0",
        "phase_id": "Phase380-CausalLayoutAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "test_whether_independently_stable_residual_profiles_contain_control_specific_causal_layout_cells",
        "denominator": {
            "condition_row_count": len(rows),
            "direction_gate_row_count": len(direction_rows),
            "model_cell_count": len(model_cells),
            "crossmodel_passing_cell_count": len(crossmodel_cells),
        },
        "descriptive": {
            "natural_transfer_gain_minimum": min(natural_gains),
            "natural_transfer_gain_median": median(natural_gains),
            "natural_transfer_gain_mean": mean(natural_gains),
            "natural_transfer_gain_maximum": max(natural_gains),
        },
        "results": {
            "direction_gate_pass_count": sum(
                row["direction_gate_pass"] for row in direction_rows
            ),
            "model_cell_pass_count": sum(row["model_cell_pass"] for row in model_cells),
            "heterogeneous_level2_cell_count": len(level2),
            "level3_cell_count": sum(row["level3"] for row in crossmodel_cells),
            "heterogeneous_terminal_interface_cell_count": len(terminal_interface),
            "heterogeneous_upstream_cell_count": len(upstream),
            "shared_cross_mechanism_territory_count": len(shared),
            "shared_cross_mechanism_territories": shared,
            "shared_terminal_interface_territory_count": len(shared_terminal),
            "shared_terminal_interface_territories": shared_terminal,
            "complete_upstream_language_path_count": 0,
            "single_neuron_causal_count": 0,
            "language_encoding_mechanism_closed": False,
        },
        "claim_boundary": {
            "crossmodel_cell_is_a_complete_path": False,
            "shared_normalized_cell_is_same_physical_neurons": False,
            "terminal_endpoint_is_upstream_encoding_rule": False,
            "late_layer_input_is_upstream_encoding_rule": False,
            "physical_holdout_opened": False,
        },
    }
    write_json(OUT / "phase380_causal_layout_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
