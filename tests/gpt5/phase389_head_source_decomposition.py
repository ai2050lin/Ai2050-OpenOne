#!/usr/bin/env python3
"""Decompose the Phase386 relation-binding trajectory by head and source role."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
P386 = ROOT / "tests/gpt5/result/phase386_multitime_relation_atlas"
OUT = ROOT / "tests/gpt5/result/phase389_head_source_decomposition"
MODELS = ("qwen3", "glm4", "deepseek7b")
LAYERS = {"qwen3": 31, "glm4": 34, "deepseek7b": 24}
CONDITIONS = (
    "A_operation_lex_x",
    "B_control_lex_x",
    "C_operation_lex_y",
    "D_control_lex_y",
)
ROLES = (
    "source_anchor",
    "source_local_context",
    "query_local_context",
    "other_prior_context",
)
WINDOW = 8


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float()
    right = right.float()
    denom = float(
        torch.linalg.vector_norm(left).item()
        * torch.linalg.vector_norm(right).item()
    )
    return float(torch.dot(left, right).item()) / max(denom, 1e-12)


def share(child: torch.Tensor, parent: torch.Tensor) -> float:
    parent = parent.float()
    denom = float(torch.dot(parent, parent).item())
    return float(torch.dot(child.float(), parent).item()) / max(denom, 1e-12)


def rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def repeat_kv(values: torch.Tensor, head_count: int) -> torch.Tensor:
    if values.shape[1] == head_count:
        return values
    if head_count % values.shape[1]:
        raise RuntimeError("Attention head count is not divisible by K/V head count")
    return values.repeat_interleave(head_count // values.shape[1], dim=1)


def role_indices(source: int, query: int, sequence_length: int) -> dict[str, list[int]]:
    source_local = list(range(max(0, source - WINDOW + 1), source))
    query_local = list(range(max(source + 1, query - WINDOW + 1), query + 1))
    claimed = {source, *source_local, *query_local}
    other = [index for index in range(query + 1) if index not in claimed]
    roles = {
        "source_anchor": [source],
        "source_local_context": source_local,
        "query_local_context": query_local,
        "other_prior_context": other,
    }
    flattened = [index for indices in roles.values() for index in indices]
    if sorted(flattened) != list(range(query + 1)):
        raise RuntimeError("Source-role partition does not conserve the causal prefix")
    if query >= sequence_length:
        raise RuntimeError("Query position exceeds attention source sequence")
    return roles


def case_events(split: str, model: str, case: dict[str, Any]) -> dict[str, Any]:
    layer = LAYERS[model]
    path = (
        P386
        / "collection"
        / split
        / "private/models"
        / model
        / case["blind_case_id"]
        / f"layer_{layer:03d}.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    frame = next(
        (
            item
            for item in payload["attention"]["frames"]
            if "source_encoded" in item["coordinate_names"]
            and "query_integrated" in item["coordinate_names"]
        ),
        None,
    )
    if frame is None:
        raise RuntimeError(f"Missing source/query frame in {path}")
    source_receiver = frame["coordinate_names"].index("source_encoded")
    query_receiver = frame["coordinate_names"].index("query_integrated")
    probabilities = frame["probabilities_receivers_all_sources"].float()[0]
    values = repeat_kv(frame["value_states_all_sources"].float(), int(frame["head_count"]))[0]
    contributions = probabilities[:, :, :, None] * values[:, None, :, :]
    source_position = int(frame["global_positions"][source_receiver])
    query_position = int(frame["global_positions"][query_receiver])
    roles = role_indices(source_position, query_position, int(values.shape[1]))
    source_heads = contributions[:, source_receiver].sum(dim=1)
    query_source_contributions = contributions[:, query_receiver]
    query_heads = query_source_contributions.sum(dim=1)
    role_heads = {
        role: query_source_contributions[:, indices].sum(dim=1)
        if indices
        else torch.zeros_like(query_heads)
        for role, indices in roles.items()
    }
    reconstructed = sum(role_heads.values(), torch.zeros_like(query_heads))
    error = float((reconstructed - query_heads).abs().max().item())
    return {
        "source_heads": source_heads,
        "query_heads": query_heads,
        "role_heads": role_heads,
        "role_conservation_max_abs_error": error,
        "head_count": int(frame["head_count"]),
        "source_position": source_position,
        "query_position": query_position,
    }


def group_rows(split: str) -> list[dict[str, Any]]:
    cases = [
        row
        for row in read_jsonl(P386 / f"protocol/private/phase386_{split}_cases.jsonl")
        if row["mechanism_id"] == "relation_binding"
    ]
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        grouped[(case["private_execution_model"], case["phase386_public_parallel_group_id"])][
            case["contrast_condition"]
        ] = case
    rows: list[dict[str, Any]] = []
    for (model, group_id), conditions in sorted(grouped.items()):
        if set(conditions) != set(CONDITIONS):
            raise RuntimeError(f"Incomplete Phase389 four-condition group {model}/{group_id}")
        events = {
            condition: case_events(split, model, conditions[condition])
            for condition in CONDITIONS
        }
        head_count = events[CONDITIONS[0]]["head_count"]
        for head in range(head_count):
            source_x = (
                events["A_operation_lex_x"]["source_heads"][head]
                - events["B_control_lex_x"]["source_heads"][head]
            )
            source_y = (
                events["C_operation_lex_y"]["source_heads"][head]
                - events["D_control_lex_y"]["source_heads"][head]
            )
            query_x = (
                events["A_operation_lex_x"]["query_heads"][head]
                - events["B_control_lex_x"]["query_heads"][head]
            )
            query_y = (
                events["C_operation_lex_y"]["query_heads"][head]
                - events["D_control_lex_y"]["query_heads"][head]
            )
            role_metrics = {}
            for role in ROLES:
                role_x = (
                    events["A_operation_lex_x"]["role_heads"][role][head]
                    - events["B_control_lex_x"]["role_heads"][role][head]
                )
                role_y = (
                    events["C_operation_lex_y"]["role_heads"][role][head]
                    - events["D_control_lex_y"]["role_heads"][role][head]
                )
                role_metrics[role] = {
                    "share_lex_x": share(role_x, query_x),
                    "share_lex_y": share(role_y, query_y),
                    "lexical_replication": cosine(role_x, role_y),
                }
            rows.append(
                {
                    "schema_version": "63.0.0",
                    "phase_id": "Phase389-HeadSourceDecomposition",
                    "split": split,
                    "model": model,
                    "parallel_group_id": group_id,
                    "layer_index": LAYERS[model],
                    "head_index": head,
                    "source_effect_norm_lex_x": float(torch.linalg.vector_norm(source_x).item()),
                    "source_effect_norm_lex_y": float(torch.linalg.vector_norm(source_y).item()),
                    "query_effect_norm_lex_x": float(torch.linalg.vector_norm(query_x).item()),
                    "query_effect_norm_lex_y": float(torch.linalg.vector_norm(query_y).item()),
                    "source_to_query_relation_lex_x": cosine(source_x, query_x),
                    "source_to_query_relation_lex_y": cosine(source_y, query_y),
                    "source_lexical_replication": cosine(source_x, source_y),
                    "query_lexical_replication": cosine(query_x, query_y),
                    "role_metrics": role_metrics,
                    "role_conservation_max_abs_error": max(
                        event["role_conservation_max_abs_error"]
                        for event in events.values()
                    ),
                }
            )
    return rows


def head_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "group_count": len(rows),
        "nonzero_source_rate": rate(
            [
                row["source_effect_norm_lex_x"] > 1e-6
                and row["source_effect_norm_lex_y"] > 1e-6
                for row in rows
            ]
        ),
        "nonzero_query_rate": rate(
            [
                row["query_effect_norm_lex_x"] > 1e-6
                and row["query_effect_norm_lex_y"] > 1e-6
                for row in rows
            ]
        ),
        "median_relation_lex_x": median(
            row["source_to_query_relation_lex_x"] for row in rows
        ),
        "median_relation_lex_y": median(
            row["source_to_query_relation_lex_y"] for row in rows
        ),
        "positive_relation_rate_lex_x": rate(
            [row["source_to_query_relation_lex_x"] > 0 for row in rows]
        ),
        "positive_relation_rate_lex_y": rate(
            [row["source_to_query_relation_lex_y"] > 0 for row in rows]
        ),
        "median_source_lexical_replication": median(
            row["source_lexical_replication"] for row in rows
        ),
        "median_query_lexical_replication": median(
            row["query_lexical_replication"] for row in rows
        ),
    }


def gate_pass(metrics: dict[str, Any]) -> bool:
    return (
        metrics["nonzero_source_rate"] >= 0.875
        and metrics["nonzero_query_rate"] >= 0.875
        and metrics["median_relation_lex_x"] >= 0.15
        and metrics["median_relation_lex_y"] >= 0.15
        and metrics["positive_relation_rate_lex_x"] >= 0.75
        and metrics["positive_relation_rate_lex_y"] >= 0.75
        and metrics["median_source_lexical_replication"] >= 0.10
        and metrics["median_query_lexical_replication"] >= 0.10
    )


def role_gate(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
    metrics = {
        "median_share_lex_x": median(
            row["role_metrics"][role]["share_lex_x"] for row in rows
        ),
        "median_share_lex_y": median(
            row["role_metrics"][role]["share_lex_y"] for row in rows
        ),
        "positive_share_rate_lex_x": rate(
            [row["role_metrics"][role]["share_lex_x"] > 0 for row in rows]
        ),
        "positive_share_rate_lex_y": rate(
            [row["role_metrics"][role]["share_lex_y"] > 0 for row in rows]
        ),
        "median_lexical_replication": median(
            row["role_metrics"][role]["lexical_replication"] for row in rows
        ),
    }
    metrics["gate_pass"] = (
        metrics["median_share_lex_x"] >= 0.10
        and metrics["median_share_lex_y"] >= 0.10
        and metrics["positive_share_rate_lex_x"] >= 0.75
        and metrics["positive_share_rate_lex_y"] >= 0.75
        and metrics["median_lexical_replication"] >= 0.10
    )
    return metrics


def main() -> None:
    discovery = group_rows("discovery")
    calibration = group_rows("calibration")
    write_jsonl(OUT / "phase389_discovery_head_group_rows.jsonl", discovery)
    write_jsonl(OUT / "phase389_calibration_head_group_rows.jsonl", calibration)
    discovery_by_head: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    calibration_by_head: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in discovery:
        discovery_by_head[(row["model"], row["head_index"])].append(row)
    for row in calibration:
        calibration_by_head[(row["model"], row["head_index"])].append(row)

    head_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    for key, rows in sorted(discovery_by_head.items()):
        model, head = key
        discovery_metrics = head_gate(rows)
        calibration_metrics = head_gate(calibration_by_head[key])
        discovery_pass = gate_pass(discovery_metrics)
        calibration_pass = gate_pass(calibration_metrics)
        head_rows.append(
            {
                "schema_version": "63.0.0",
                "phase_id": "Phase389-HeadSourceDecomposition",
                "model": model,
                "layer_index": LAYERS[model],
                "head_index": head,
                "discovery_metrics": discovery_metrics,
                "calibration_metrics": calibration_metrics,
                "discovery_gate_pass": discovery_pass,
                "calibration_gate_pass": calibration_pass,
                "replicated_head_relation": discovery_pass and calibration_pass,
                "causal_claim": False,
            }
        )
        if discovery_pass and calibration_pass:
            for role in ROLES:
                discovery_role = role_gate(rows, role)
                calibration_role = role_gate(calibration_by_head[key], role)
                role_rows.append(
                    {
                        "schema_version": "63.0.0",
                        "phase_id": "Phase389-HeadSourceDecomposition",
                        "model": model,
                        "layer_index": LAYERS[model],
                        "head_index": head,
                        "source_role": role,
                        "discovery_metrics": discovery_role,
                        "calibration_metrics": calibration_role,
                        "replicated_role_route": discovery_role["gate_pass"]
                        and calibration_role["gate_pass"],
                        "causal_claim": False,
                    }
                )
    write_jsonl(OUT / "phase389_head_candidate_rows.jsonl", head_rows)
    write_jsonl(OUT / "phase389_role_candidate_rows.jsonl", role_rows)
    replicated_heads = [row for row in head_rows if row["replicated_head_relation"]]
    replicated_roles = [row for row in role_rows if row["replicated_role_route"]]
    role_by_head: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in role_rows:
        role_by_head[(row["model"], row["head_index"])][row["source_role"]] = row
    specificity_rows: list[dict[str, Any]] = []
    for (model, head), roles in sorted(role_by_head.items()):
        source = roles["source_anchor"]
        other_roles = [row for role, row in roles.items() if role != "source_anchor"]
        discovery_specific = source["discovery_metrics"]["gate_pass"] and all(
            source["discovery_metrics"][axis] > other["discovery_metrics"][axis]
            for other in other_roles
            for axis in ("median_share_lex_x", "median_share_lex_y")
        )
        calibration_specific = source["calibration_metrics"]["gate_pass"] and all(
            source["calibration_metrics"][axis] > other["calibration_metrics"][axis]
            for other in other_roles
            for axis in ("median_share_lex_x", "median_share_lex_y")
        )
        specificity_rows.append(
            {
                "schema_version": "63.1.0",
                "phase_id": "Phase389-SourceAnchorSpecificity",
                "model": model,
                "layer_index": LAYERS[model],
                "head_index": head,
                "source_anchor_discovery_specific": discovery_specific,
                "source_anchor_calibration_specific": calibration_specific,
                "replicated_source_anchor_specificity": discovery_specific
                and calibration_specific,
                "post_decomposition_exploratory_rule": True,
                "causal_claim": False,
            }
        )
    write_jsonl(
        OUT / "phase389_source_anchor_specificity_rows.jsonl", specificity_rows
    )
    specific_rows = [
        row for row in specificity_rows if row["replicated_source_anchor_specificity"]
    ]
    roles_by_model = {
        model: sorted(
            {
                row["source_role"]
                for row in replicated_roles
                if row["model"] == model
            }
        )
        for model in MODELS
    }
    common_roles = sorted(set.intersection(*(set(value) for value in roles_by_model.values())))
    summary = {
        "schema_version": "63.0.0",
        "phase_id": "Phase389-HeadSourceDecomposition",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "models": list(MODELS),
            "discovery_groups_per_model": 8,
            "calibration_groups_per_model": 4,
            "all_heads_evaluated": len(head_rows),
            "discovery_head_group_rows": len(discovery),
            "calibration_head_group_rows": len(calibration),
            "source_role_partition_count": len(ROLES),
            "source_role_window_width": WINDOW,
            "physical_holdout_reused": False,
            "new_model_run_count": 0,
        },
        "results": {
            "discovery_head_candidate_count": sum(
                row["discovery_gate_pass"] for row in head_rows
            ),
            "replicated_head_relation_count": len(replicated_heads),
            "replicated_head_counts_by_model": {
                model: sum(row["model"] == model for row in replicated_heads)
                for model in MODELS
            },
            "replicated_role_route_count": len(replicated_roles),
            "replicated_role_counts_by_model": {
                model: sum(row["model"] == model for row in replicated_roles)
                for model in MODELS
            },
            "replicated_roles_by_model": roles_by_model,
            "common_replicated_source_roles": common_roles,
            "replicated_source_anchor_specificity_count": len(specific_rows),
            "replicated_source_anchor_specificity_by_model": {
                model: sum(row["model"] == model for row in specific_rows)
                for model in MODELS
            },
            "max_role_conservation_abs_error": max(
                row["role_conservation_max_abs_error"]
                for row in discovery + calibration
            ),
            "causal_head_source_route_established": False,
            "single_neuron_path_established": False,
        },
        "claim_boundary": {
            "discovery_and_calibration_were_previously_viewed_at_aggregate_level": True,
            "head_source_analysis_is_fully_independent_holdout": False,
            "source_anchor_specificity_rule_added_after_broad_head_result": True,
            "replicated_head_relation_is_causal_transport": False,
            "small_model_head_numbers_are_crossmodel_equivalent": False,
        },
        "authorization": {
            "register_descriptive_head_source_candidates": bool(replicated_roles),
            "run_new_head_specific_intervention": all(
                any(row["model"] == model for row in specific_rows)
                for model in MODELS
            ),
            "run_single_neuron_scan": False,
            "reuse_phase386_physical_holdout": False,
        },
    }
    write_json(OUT / "phase389_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
