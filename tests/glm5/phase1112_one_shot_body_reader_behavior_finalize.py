#!/usr/bin/env python3
"""Freeze Phase1112 pair-specific behavior authorization."""

from __future__ import annotations

import json

import phase1112_one_shot_body_reader_protocol as protocol


def cell(summary: dict, pair: str, surface: str, split: str, regime: str, route: str, congruence: str) -> dict:
    return summary["per_cell"]["|".join((
        pair, surface, split, regime, route, congruence,
    ))]


def weighted_accuracy(rows: list[dict]) -> float:
    count = sum(int(row["candidate_count"]) for row in rows)
    return sum(
        float(row["candidate_accuracy"]) * int(row["candidate_count"])
        for row in rows
    ) / max(count, 1)


def evaluate_split(summary: dict, pair: str, split: str) -> dict:
    thresholds = protocol.THRESHOLDS
    exact_conflict = []
    exact_congruent = []
    ordinal_conflict = []
    regime_rows = {regime: [] for regime in protocol.LABEL_REGIMES}
    cells = {}
    for surface in protocol.SURFACES:
        for regime in protocol.LABEL_REGIMES:
            exact = cell(summary, pair, surface, split, regime, "exact", "conflict")
            congruent = cell(summary, pair, surface, split, regime, "exact", "congruent")
            ordinal = cell(summary, pair, surface, split, regime, "ordinal", "conflict")
            exact_conflict.append(exact)
            exact_congruent.append(congruent)
            ordinal_conflict.append(ordinal)
            regime_rows[regime].append(exact)
            cells[f"{surface}|{regime}|exact|conflict"] = {
                **exact,
                "passed": (
                    exact["candidate_finite_fraction"]
                    >= thresholds["minimum_candidate_finite_fraction"]
                    and exact["candidate_accuracy"]
                    >= thresholds["minimum_conflict_cell_accuracy"]
                ),
            }
    regime_accuracy = {
        regime: weighted_accuracy(rows) for regime, rows in regime_rows.items()
    }
    congruent_accuracy = weighted_accuracy(exact_congruent)
    gates = {
        "all_exact_conflict_cells": all(row["passed"] for row in cells.values()),
        "relation_label_accuracy": (
            regime_accuracy["relation_label"] >= thresholds["minimum_regime_accuracy"]
        ),
        "neutral_label_accuracy": (
            regime_accuracy["neutral_label"] >= thresholds["minimum_regime_accuracy"]
        ),
        "exact_congruent": congruent_accuracy >= thresholds["minimum_congruent_accuracy"],
    }
    return {
        "split": split,
        "exact_accuracy": weighted_accuracy(exact_conflict),
        "regime_accuracy": regime_accuracy,
        "congruent_accuracy": congruent_accuracy,
        "ordinal_accuracy_descriptive": weighted_accuracy(ordinal_conflict),
        "cells": cells,
        "gates": gates,
        "passed": all(gates.values()),
    }


def evaluate_model(model: str) -> dict:
    summary = protocol.read_json(protocol.OUT_ROOT / "behavior" / model / "summary.json")
    pair_results = {}
    passing = []
    for pair in protocol.RELATION_PAIRS:
        splits = {split: evaluate_split(summary, pair, split) for split in protocol.SPLITS}
        passed = all(row["passed"] for row in splits.values())
        pair_results[pair] = {"splits": splits, "passed": passed}
        if passed:
            passing.append(pair)
    return {
        "model": model,
        "summary_digest": summary["summary_digest"],
        "candidate_finite_fraction": summary["candidate_finite_fraction"],
        "candidate_accuracy": summary["candidate_accuracy"],
        "precision_gate": (
            summary["candidate_finite_fraction"]
            >= protocol.THRESHOLDS["minimum_candidate_finite_fraction"]
        ),
        "pair_results": pair_results,
        "passing_pairs": passing,
        "passing_pair_count": len(passing),
    }


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1112 protocol audit failed")
    models = {model: evaluate_model(model) for model in protocol.MODELS}
    models_by_pair = {
        pair: [
            model for model, result in models.items() if pair in result["passing_pairs"]
        ]
        for pair in protocol.RELATION_PAIRS
    }
    cross_model_pairs = [
        pair for pair, passing_models in models_by_pair.items()
        if len(passing_models) >= protocol.THRESHOLDS["minimum_models_per_pair"]
    ]
    authorized_models = [
        model for model, result in models.items()
        if sum(pair in result["passing_pairs"] for pair in cross_model_pairs)
        >= protocol.THRESHOLDS["minimum_cross_model_pairs"]
    ]
    hidden_scan_authorized = (
        len(cross_model_pairs) >= protocol.THRESHOLDS["minimum_cross_model_pairs"]
        and len(authorized_models) >= protocol.THRESHOLDS["minimum_models"]
    )
    result = {
        "schema_version": "phase1112_behavior_authorization.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "protocol_audit_digest": audit["audit_digest"],
        "models": models,
        "models_by_pair": models_by_pair,
        "cross_model_pairs": cross_model_pairs,
        "authorized_models": authorized_models,
        "hidden_scan_authorized": hidden_scan_authorized,
        "decision": (
            "run_one_shot_body_attention_map"
            if hidden_scan_authorized else "close_registry_at_behavior_gate"
        ),
        "authorization_scope": (
            "Only model-pair cells passing both independent splits enter hidden analysis."
        ),
    }
    result["authorization_digest"] = protocol.digest(result)
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "behavior_authorization.json", result)
    print(json.dumps({
        "phase": protocol.PHASE,
        "passing_pairs_by_model": {
            model: row["passing_pairs"] for model, row in models.items()
        },
        "cross_model_pairs": cross_model_pairs,
        "authorized_models": authorized_models,
        "hidden_scan_authorized": hidden_scan_authorized,
        "authorization_digest": result["authorization_digest"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
