#!/usr/bin/env python3
"""Apply the frozen Phase381 joint-state gates after all model scans finish."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase381_joint_state_formation"
MODELS = ("qwen3", "glm4", "deepseek7b")
DEPTHS = ("early", "middle_early", "middle", "middle_late", "late")
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")
SINGLES = ("source", "query", "current")
JOINT = "source_query_current"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def main() -> None:
    freeze = read_json(OUT / "phase381_joint_scan_freeze.json")
    gates = freeze["frozen_joint_gates"]
    rows: list[dict[str, Any]] = []
    run_summaries = []
    for model in MODELS:
        rows.extend(
            read_jsonl(
                OUT / "causal/private/models" / model / "phase381_joint_rows.jsonl"
            )
        )
        run_summaries.append(read_json(OUT / "causal/models" / model / "complete.json"))
    expected = freeze["denominator"]["condition_rows_total"]
    if len(rows) != expected or not all(row["valid"] for row in run_summaries):
        raise RuntimeError(f"Incomplete Phase381 causal denominator: {len(rows)}/{expected}")
    by_condition = {
        (
            row["model"],
            row["mechanism_id"],
            row["contrast_axis"],
            row["anonymous_parallel_group_id"],
            row["transfer_name"],
            row["depth_name"],
            row["component_type"],
            row["role_set_name"],
            row["condition"],
        ): row
        for row in rows
    }
    direction_rows: list[dict[str, Any]] = []
    for key, natural in sorted(by_condition.items()):
        if key[-2:] != (JOINT, "natural_swap"):
            continue
        model, mechanism, axis, group, transfer, depth, component, _role_set, _ = key
        equal_energy = by_condition[(*key[:-1], "equal_energy_permutation")]
        singles = [
            by_condition[
                (model, mechanism, axis, group, transfer, depth, component, role, "natural_swap")
            ]
            for role in SINGLES
        ]
        best_single = max(row["transfer_gain"] for row in singles)
        wrong_depth = DEPTHS[(DEPTHS.index(depth) + 1) % len(DEPTHS)]
        wrong_component = COMPONENTS[(COMPONENTS.index(component) + 1) % len(COMPONENTS)]
        depth_control = by_condition[
            (model, mechanism, axis, group, transfer, wrong_depth, component, JOINT, "natural_swap")
        ]
        component_control = by_condition[
            (model, mechanism, axis, group, transfer, depth, wrong_component, JOINT, "natural_swap")
        ]
        gain_gate = natural["transfer_gain"] >= gates["minimum_natural_transfer_gain"]
        energy_gate = natural["transfer_gain"] >= (
            equal_energy["transfer_gain"] + gates["minimum_gain_over_equal_energy"]
        )
        share_gate = natural["terminal_transfer_share"] >= gates[
            "minimum_terminal_transfer_share"
        ]
        share_control_gate = natural["terminal_transfer_share"] >= (
            equal_energy["terminal_transfer_share"]
            + gates["minimum_share_over_equal_energy"]
        )
        synergy_gate = natural["transfer_gain"] >= (
            best_single + gates["minimum_gain_over_best_single_position"]
        )
        depth_gate = natural["transfer_gain"] >= (
            depth_control["transfer_gain"] + gates["minimum_gain_over_cyclic_wrong_depth"]
        )
        component_gate = natural["transfer_gain"] >= (
            component_control["transfer_gain"]
            + gates["minimum_gain_over_cyclic_wrong_component"]
        )
        side_effect_gate = natural["transfer_to_offtarget_rms_ratio"] >= gates[
            "minimum_transfer_to_offtarget_rms_ratio"
        ]
        passed = all(
            (
                gain_gate,
                energy_gate,
                share_gate,
                share_control_gate,
                synergy_gate,
                depth_gate,
                component_gate,
                side_effect_gate,
            )
        )
        direction_rows.append(
            {
                "schema_version": "54.6.0",
                "phase_id": "Phase381-JointStateAnalysis",
                "model": model,
                "mechanism_id": mechanism,
                "contrast_axis": axis,
                "anonymous_parallel_group_id": group,
                "transfer_name": transfer,
                "depth_name": depth,
                "component_type": component,
                "role_set_name": JOINT,
                "joint_transfer_gain": natural["transfer_gain"],
                "best_single_position_transfer_gain": best_single,
                "joint_synergy_gain": natural["transfer_gain"] - best_single,
                "equal_energy_transfer_gain": equal_energy["transfer_gain"],
                "wrong_depth_transfer_gain": depth_control["transfer_gain"],
                "wrong_component_transfer_gain": component_control["transfer_gain"],
                "joint_terminal_transfer_share": natural["terminal_transfer_share"],
                "equal_energy_terminal_transfer_share": equal_energy[
                    "terminal_transfer_share"
                ],
                "joint_transfer_to_offtarget_rms_ratio": natural[
                    "transfer_to_offtarget_rms_ratio"
                ],
                "gain_gate": gain_gate,
                "equal_energy_gate": energy_gate,
                "terminal_share_gate": share_gate,
                "terminal_share_control_gate": share_control_gate,
                "joint_over_best_single_gate": synergy_gate,
                "wrong_depth_gate": depth_gate,
                "wrong_component_gate": component_gate,
                "side_effect_gate": side_effect_gate,
                "direction_gate_pass": passed,
            }
        )
    expected_transfers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in direction_rows:
        expected_transfers[(row["mechanism_id"], row["contrast_axis"])].add(
            row["transfer_name"]
        )
    group_cells: dict[tuple[Any, ...], dict[str, bool]] = defaultdict(dict)
    for row in direction_rows:
        key = (
            row["model"],
            row["mechanism_id"],
            row["contrast_axis"],
            row["anonymous_parallel_group_id"],
            row["depth_name"],
            row["component_type"],
        )
        group_cells[key][row["transfer_name"]] = row["direction_gate_pass"]
    cell_groups: dict[tuple[Any, ...], list[bool]] = defaultdict(list)
    for key, transfer_passes in group_cells.items():
        model, mechanism, axis, _group, depth, component = key
        expected_names = expected_transfers[(mechanism, axis)]
        group_pass = set(transfer_passes) == expected_names and all(
            transfer_passes.values()
        )
        cell_groups[(model, mechanism, axis, depth, component)].append(group_pass)
    model_cells: list[dict[str, Any]] = []
    for key, values in sorted(cell_groups.items()):
        model, mechanism, axis, depth, component = key
        pass_count = sum(values)
        terminal_interface = depth == "late" and component in {
            "layer_input",
            "layer_output",
        }
        model_cells.append(
            {
                "schema_version": "54.6.0",
                "phase_id": "Phase381-JointStateAnalysis",
                "model": model,
                "mechanism_id": mechanism,
                "contrast_axis": axis,
                "depth_name": depth,
                "component_type": component,
                "role_set_name": JOINT,
                "group_count": len(values),
                "all_four_direction_group_pass_count": pass_count,
                "model_cell_pass": pass_count
                >= gates["minimum_groups_all_four_directions"],
                "terminal_interface_cell": terminal_interface,
                "upstream_joint_state_cell": not terminal_interface,
            }
        )
    cell_models: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in model_cells:
        if row["model_cell_pass"]:
            cell_models[
                (
                    row["mechanism_id"],
                    row["contrast_axis"],
                    row["depth_name"],
                    row["component_type"],
                )
            ].add(row["model"])
    crossmodel_cells: list[dict[str, Any]] = []
    for key, models in sorted(cell_models.items()):
        mechanism, axis, depth, component = key
        terminal_interface = depth == "late" and component in {
            "layer_input",
            "layer_output",
        }
        crossmodel_cells.append(
            {
                "schema_version": "54.6.0",
                "phase_id": "Phase381-JointStateAnalysis",
                "mechanism_id": mechanism,
                "contrast_axis": axis,
                "depth_name": depth,
                "component_type": component,
                "role_set_name": JOINT,
                "models": sorted(models),
                "heterogeneous_level2": "glm4" in models
                and bool(models & {"qwen3", "deepseek7b"}),
                "level3": models == set(MODELS),
                "terminal_interface_cell": terminal_interface,
                "upstream_joint_state_cell": not terminal_interface,
                "complete_language_path": False,
                "same_neurons_established": False,
            }
        )
    level2 = [row for row in crossmodel_cells if row["heterogeneous_level2"]]
    upstream = [row for row in level2 if row["upstream_joint_state_cell"]]
    terminal = [row for row in level2 if row["terminal_interface_cell"]]
    territories: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in upstream:
        territories[(row["depth_name"], row["component_type"])].add(
            row["mechanism_id"]
        )
    shared = [
        {
            "depth_name": key[0],
            "component_type": key[1],
            "role_set_name": JOINT,
            "mechanisms": sorted(mechanisms),
            "cross_mechanism_joint_state_territory": len(mechanisms) >= 2,
            "complete_language_path": False,
            "same_neurons_established": False,
        }
        for key, mechanisms in sorted(territories.items())
        if len(mechanisms) >= 2
    ]
    write_jsonl(OUT / "causal/phase381_joint_direction_rows.jsonl", direction_rows)
    write_jsonl(OUT / "causal/phase381_joint_model_cells.jsonl", model_cells)
    write_jsonl(OUT / "causal/phase381_joint_crossmodel_cells.jsonl", crossmodel_cells)
    write_jsonl(OUT / "causal/phase381_shared_upstream_territories.jsonl", shared)
    gains = [row["joint_transfer_gain"] for row in direction_rows]
    synergies = [row["joint_synergy_gain"] for row in direction_rows]
    gate_fields = (
        "gain_gate",
        "equal_energy_gate",
        "terminal_share_gate",
        "terminal_share_control_gate",
        "joint_over_best_single_gate",
        "wrong_depth_gate",
        "wrong_component_gate",
        "side_effect_gate",
    )
    group_pass_distribution = Counter(
        row["all_four_direction_group_pass_count"] for row in model_cells
    )
    near_cells = sorted(
        model_cells,
        key=lambda row: (
            -row["all_four_direction_group_pass_count"],
            row["model"],
            row["mechanism_id"],
            row["contrast_axis"],
            row["depth_name"],
            row["component_type"],
        ),
    )[:15]
    summary = {
        "schema_version": "54.6.0",
        "phase_id": "Phase381-JointStateAnalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "test_whether_upstream_state_requires_joint_source_query_current_intervention",
        "denominator": {
            "condition_row_count": len(rows),
            "joint_direction_gate_row_count": len(direction_rows),
            "model_cell_count": len(model_cells),
            "crossmodel_passing_cell_count": len(crossmodel_cells),
        },
        "descriptive": {
            "joint_transfer_gain_minimum": min(gains),
            "joint_transfer_gain_median": median(gains),
            "joint_transfer_gain_mean": mean(gains),
            "joint_transfer_gain_maximum": max(gains),
            "joint_synergy_gain_minimum": min(synergies),
            "joint_synergy_gain_median": median(synergies),
            "joint_synergy_gain_mean": mean(synergies),
            "joint_synergy_gain_maximum": max(synergies),
        },
        "results": {
            "direction_gate_pass_counts": {
                field: sum(row[field] for row in direction_rows)
                for field in gate_fields
            },
            "joint_direction_gate_pass_count": sum(
                row["direction_gate_pass"] for row in direction_rows
            ),
            "model_cell_pass_count": sum(row["model_cell_pass"] for row in model_cells),
            "maximum_all_four_direction_group_pass_count_in_any_model_cell": max(
                row["all_four_direction_group_pass_count"] for row in model_cells
            ),
            "model_cell_group_pass_distribution": {
                str(count): cell_count
                for count, cell_count in sorted(group_pass_distribution.items())
            },
            "top_near_cells_posthoc_diagnostic": near_cells,
            "heterogeneous_level2_cell_count": len(level2),
            "level3_cell_count": sum(row["level3"] for row in crossmodel_cells),
            "heterogeneous_terminal_interface_cell_count": len(terminal),
            "heterogeneous_upstream_joint_state_cell_count": len(upstream),
            "shared_cross_mechanism_upstream_territory_count": len(shared),
            "shared_cross_mechanism_upstream_territories": shared,
            "joint_distributed_upstream_state_established": bool(upstream),
            "complete_upstream_language_path_count": 0,
            "single_neuron_causal_count": 0,
            "language_encoding_mechanism_closed": False,
        },
        "claim_boundary": {
            "joint_cell_is_complete_path": False,
            "normalized_cell_is_same_neurons_across_models": False,
            "terminal_interface_is_upstream_rule": False,
            "single_neuron_scan_opened": False,
            "nine_family_layout_completed": False,
        },
    }
    write_json(OUT / "phase381_joint_state_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
