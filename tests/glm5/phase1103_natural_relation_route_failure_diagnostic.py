#!/usr/bin/env python3
"""Describe the frozen Phase1103 behavior stop without changing authorization."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any

import phase1103_natural_relation_route_protocol as protocol


def template_from_unit(unit_id: str) -> int:
    match = re.search(r"\.t([0-3])\.i\d+$", unit_id)
    if not match:
        raise ValueError(unit_id)
    return int(match.group(1))


def main() -> None:
    behavior = protocol.read_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json"
    )
    models = {}
    route_pair_models: dict[str, dict[str, list[str]]] = {
        route: {} for route in protocol.ROUTE_TYPES
    }
    pair_minimums_by_model: dict[str, dict[str, float]] = {}
    for model in protocol.MODELS:
        model_auth = behavior["models"][model]
        false_gates: Counter = Counter()
        route_both_split_pairs = {route: [] for route in protocol.ROUTE_TYPES}
        all_conflict_both = []
        congruent_both = []
        generation_both = []
        pair_rows = {}
        pair_minimums = {}
        for pair, pair_result in model_auth["pair_results"].items():
            splits = pair_result["splits"]
            for split, row in splits.items():
                false_gates.update(
                    key for key, value in row["gates"].items() if not value
                )
            route_passes = {
                route: all(
                    splits[split]["gates"][f"{route}_route"]
                    for split in protocol.SPLITS
                )
                for route in protocol.ROUTE_TYPES
            }
            for route, passed in route_passes.items():
                if passed:
                    route_both_split_pairs[route].append(pair)
            all_conflict = all(
                splits[split]["gates"]["all_conflict_cells"]
                for split in protocol.SPLITS
            )
            congruent = all(
                splits[split]["gates"]["congruent"]
                for split in protocol.SPLITS
            )
            generation = all(
                splits[split]["gates"]["all_generation_cells"]
                for split in protocol.SPLITS
            )
            if all_conflict:
                all_conflict_both.append(pair)
            if congruent:
                congruent_both.append(pair)
            if generation:
                generation_both.append(pair)
            minimum = min(
                float(splits[split]["minimum_conflict_cell_accuracy"])
                for split in protocol.SPLITS
            )
            pair_minimums[pair] = minimum
            pair_rows[pair] = {
                "minimum_conflict_cell_accuracy": minimum,
                "route_passes_both_splits": route_passes,
                "all_conflict_cells_both_splits": all_conflict,
                "congruent_both_splits": congruent,
                "all_generation_cells_both_splits": generation,
                "passed_gate_count": sum(
                    value for split in protocol.SPLITS
                    for value in splits[split]["gates"].values()
                ),
                "total_gate_count": sum(
                    len(splits[split]["gates"])
                    for split in protocol.SPLITS
                ),
            }

        details = protocol.read_jsonl(
            protocol.OUT_ROOT / "behavior" / model / "candidate_detail.jsonl"
        )
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in details:
            grouped[(
                str(row["surface"]), str(row["split"]),
                str(row["route_type"]), str(row["congruence"]),
            )].append(row)
        global_cells = {}
        for key, records in sorted(grouped.items()):
            finite = [row for row in records if row["finite"]]
            global_cells["|".join(key)] = {
                "count": len(records),
                "finite_fraction": len(finite) / max(len(records), 1),
                "accuracy": sum(row["hit"] for row in records)
                / max(len(records), 1),
            }
        by_template: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in details:
            if row["congruence"] != "conflict":
                continue
            by_template[(
                template_from_unit(str(row["unit_id"])),
                str(row["surface"]), str(row["route_type"]),
            )].append(row)
        template_cells = {
            "|".join((str(template), surface, route)): {
                "count": len(records),
                "accuracy": sum(row["hit"] for row in records)
                / max(len(records), 1),
            }
            for (template, surface, route), records in sorted(by_template.items())
        }
        ranked_pairs = sorted(
            pair_rows,
            key=lambda pair: (
                pair_rows[pair]["passed_gate_count"],
                pair_rows[pair]["minimum_conflict_cell_accuracy"],
                pair,
            ),
            reverse=True,
        )
        models[model] = {
            "false_gate_counts_across_30_pair_splits": dict(false_gates),
            "route_pairs_passing_both_splits": route_both_split_pairs,
            "route_pair_counts_passing_both_splits": {
                route: len(pairs)
                for route, pairs in route_both_split_pairs.items()
            },
            "all_conflict_cell_pairs_passing_both_splits": all_conflict_both,
            "congruent_pairs_passing_both_splits": congruent_both,
            "all_generation_cell_pairs_passing_both_splits": generation_both,
            "global_cells": global_cells,
            "template_cells": template_cells,
            "pair_diagnostic": pair_rows,
            "best_five_pairs_descriptive_only": ranked_pairs[:5],
        }
        pair_minimums_by_model[model] = pair_minimums

    for route in protocol.ROUTE_TYPES:
        for pair in protocol.RELATION_PAIRS:
            passing_models = [
                model for model in protocol.MODELS
                if pair in models[model][
                    "route_pairs_passing_both_splits"
                ][route]
            ]
            route_pair_models[route][pair] = passing_models
    common_route_pairs = {
        route: {
            pair: passing_models
            for pair, passing_models in mapping.items()
            if len(passing_models) >= 2
        }
        for route, mapping in route_pair_models.items()
    }
    result = {
        "schema_version": "phase1103_failure_diagnostic.v1",
        "phase": protocol.PHASE,
        "authorization_digest": behavior["authorization_digest"],
        "frozen_authorization_unchanged": True,
        "models": models,
        "cross_model_route_pairs_passing_both_splits": common_route_pairs,
        "interpretation_limits": [
            "This diagnostic cannot authorize hidden-state access.",
            "Best-pair rankings are post-hoc descriptions and cannot be selected for Phase1103 causality.",
            "A paraphrase failure may reflect semantic mismatch, prompt naturalness, tokenization, or model capacity; it is not a direct neural-code result.",
            "A passing exact or ordinal route does not establish a semantic relation address.",
        ],
    }
    result["diagnostic_digest"] = protocol.digest(result)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "failure_diagnostic.json",
        result,
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "route_pair_counts": {
            model: row["route_pair_counts_passing_both_splits"]
            for model, row in models.items()
        },
        "cross_model_route_pairs": common_route_pairs,
        "diagnostic_digest": result["diagnostic_digest"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
