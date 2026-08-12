#!/usr/bin/env python3
"""Freeze model-specific Phase1104 lexical behavior authorization."""

from __future__ import annotations

import json

import phase1104_lexical_address_execution_protocol as protocol


def cell(
    summary: dict,
    pair: str,
    surface: str,
    split: str,
    regime: str,
    route: str,
    congruence: str,
) -> dict:
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
            exact = cell(
                summary, pair, surface, split, regime, "exact", "conflict"
            )
            congruent = cell(
                summary, pair, surface, split, regime, "exact", "congruent"
            )
            ordinal = cell(
                summary, pair, surface, split, regime, "ordinal", "conflict"
            )
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
            regime_accuracy["relation_label"]
            >= thresholds["minimum_regime_accuracy"]
        ),
        "neutral_label_accuracy": (
            regime_accuracy["neutral_label"]
            >= thresholds["minimum_regime_accuracy"]
        ),
        "exact_congruent": (
            congruent_accuracy >= thresholds["minimum_congruent_accuracy"]
        ),
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
    summary = protocol.read_json(
        protocol.OUT_ROOT / "behavior" / model / "summary.json"
    )
    pair_results = {}
    passing = []
    for pair in protocol.CANDIDATE_PAIRS:
        splits = {
            split: evaluate_split(summary, pair, split)
            for split in protocol.SPLITS
        }
        passed = all(row["passed"] for row in splits.values())
        pair_results[pair] = {"splits": splits, "passed": passed}
        if passed:
            passing.append(pair)
    ranked = sorted(
        passing,
        key=lambda pair: (
            -min(
                float(row["candidate_accuracy"])
                for row in pair_results[pair]["splits"]["qualification"]
                ["cells"].values()
            ),
            protocol.CANDIDATE_PAIRS.index(pair),
        ),
    )
    selected = ranked[:protocol.MAX_CAUSAL_PAIRS_PER_MODEL]
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
        "model_specific_passing_pairs": passing,
        "model_specific_passing_pair_count": len(passing),
        "qualification_ranked_passing_pairs": ranked,
        "causal_selected_pairs": selected,
        "causal_selected_pair_count": len(selected),
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1104 protocol audit failed")
    models = {model: evaluate_model(model) for model in protocol.MODELS}
    models_by_pair = {
        pair: [
            model for model, row in models.items()
            if pair in row["model_specific_passing_pairs"]
        ]
        for pair in protocol.CANDIDATE_PAIRS
    }
    causal_models_by_pair = {
        pair: [
            model for model, row in models.items()
            if pair in row["causal_selected_pairs"]
        ]
        for pair in protocol.CANDIDATE_PAIRS
    }
    cross_model_pairs = [
        pair for pair, passing_models in models_by_pair.items()
        if len(passing_models)
        >= protocol.THRESHOLDS["minimum_models_for_cross_model_upgrade"]
    ]
    model_specific_scan_authorized = any(
        row["causal_selected_pairs"] for row in models.values()
    )
    wrong_pair = {
        pair: protocol.CANDIDATE_PAIRS[
            (index + 1) % len(protocol.CANDIDATE_PAIRS)
        ]
        for index, pair in enumerate(protocol.CANDIDATE_PAIRS)
    }
    result = {
        "schema_version": "phase1104_behavior_authorization.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "protocol_audit_digest": audit["audit_digest"],
        "models": models,
        "behavior_models_by_pair": models_by_pair,
        "causal_models_by_pair": causal_models_by_pair,
        "model_specific_causal_scan_authorized": model_specific_scan_authorized,
        "cross_model_behavior_pairs": cross_model_pairs,
        "wrong_pair_controls": wrong_pair,
        "decision": (
            "authorize_model_specific_lexical_causal_scan"
            if model_specific_scan_authorized
            else "stop_at_behavior_gate"
        ),
        "authorization_scope": (
            "A one-model pass authorizes only model/pair-specific lexical "
            "execution analysis. Cross-model claims require the separately "
            "listed cross-model behavior pairs."
        ),
        "phase1103_decision_unchanged": True,
        "generation_excluded_from_gate": True,
        "paraphrase_excluded_to_separate_semantic_ledger": True,
    }
    result["authorization_digest"] = protocol.digest(result)
    protocol.write_json(
        protocol.OUT_ROOT / "analysis" / "behavior_authorization.json", result
    )
    print(json.dumps({
        "phase": protocol.PHASE,
        "passing_pairs_by_model": {
            model: row["model_specific_passing_pairs"]
            for model, row in models.items()
        },
        "cross_model_behavior_pairs": cross_model_pairs,
        "model_specific_causal_scan_authorized": model_specific_scan_authorized,
        "authorization_digest": result["authorization_digest"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
