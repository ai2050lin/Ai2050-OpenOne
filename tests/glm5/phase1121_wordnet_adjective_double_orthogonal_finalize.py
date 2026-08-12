#!/usr/bin/env python3
"""Finalize the frozen Phase1121 behavior gate from raw candidate scores."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1121_wordnet_adjective_double_orthogonal_protocol as protocol


def rate(rows: list[dict[str, Any]], key: str = "candidate_hit") -> float:
    return sum(bool(row[key]) for row in rows) / max(len(rows), 1)


def grouped_rates(rows: list[dict[str, Any]], field: str, key: str = "candidate_hit") -> dict[str, float]:
    values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        values[str(row[field])].append(row)
    return {name: rate(panel, key) for name, panel in sorted(values.items())}


def minimum(values: dict[str, float]) -> float:
    return min(values.values()) if values else 0.0


def interaction_rows(detail: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        grouped[row["interaction_id"]].append(row)
    values: list[dict[str, Any]] = []
    for interaction_id, panel in sorted(grouped.items()):
        if len(panel) != 4 or not all(row["finite"] for row in panel):
            continue
        cells = {(int(row["context_sense"]), int(row["definition_sense"])): float(row["z_true_minus_false"]) for row in panel}
        if set(cells) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            continue
        interaction = 0.5 * ((cells[(0, 0)] - cells[(0, 1)]) - (cells[(1, 0)] - cells[(1, 1)]))
        first = panel[0]
        values.append({
            "interaction_id": interaction_id,
            "concept_id": first["concept_id"],
            "split": first["split"],
            "template": first["template"],
            "surface": first["surface"],
            "interaction": interaction,
            "direction_hit": interaction > 0.0,
        })
    return values


def cross_surface_rows(interactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in interactions:
        grouped[(row["concept_id"], int(row["template"]))].append(row)
    values: list[dict[str, Any]] = []
    for (concept_id, template), panel in sorted(grouped.items()):
        by_surface = {row["surface"]: row for row in panel}
        if set(by_surface) != set(protocol.SURFACES):
            continue
        values.append({
            "concept_id": concept_id,
            "template": template,
            "split": panel[0]["split"],
            "base_interaction": by_surface["base"]["interaction"],
            "synonym_interaction": by_surface["synonym"]["interaction"],
            "pair_hit": by_surface["base"]["direction_hit"] and by_surface["synonym"]["direction_hit"],
        })
    return values


def evaluate_model(model_name: str, prereg: dict[str, Any]) -> dict[str, Any]:
    summary = protocol.read_json(protocol.OUT_ROOT / "behavior" / model_name / "summary.json")
    detail = list(protocol.read_jsonl(protocol.OUT_ROOT / "behavior" / model_name / "candidate_detail.jsonl"))
    if summary["protocol_digest"] != prereg["protocol_digest"] or summary["case_digest"] != prereg["case_digests"][model_name]:
        raise RuntimeError(f"digest mismatch for {model_name}")
    if protocol.digest(detail) != summary["detail_digest"]:
        raise RuntimeError(f"detail digest mismatch for {model_name}")
    finite = [row for row in detail if row["finite"]]
    interactions = interaction_rows(detail)
    surface_pairs = cross_surface_rows(interactions)
    candidate = {
        "overall": rate(finite),
        "by_split": grouped_rates(finite, "split"),
        "by_surface": grouped_rates(finite, "surface"),
        "by_context_sense": grouped_rates(finite, "context_sense"),
        "by_definition_sense": grouped_rates(finite, "definition_sense"),
        "by_template": grouped_rates(finite, "template"),
    }
    direction = {
        "overall": rate(interactions, "direction_hit"),
        "by_split": grouped_rates(interactions, "split", "direction_hit"),
        "by_surface": grouped_rates(interactions, "surface", "direction_hit"),
        "by_template": grouped_rates(interactions, "template", "direction_hit"),
    }
    cross_surface = {
        "overall": rate(surface_pairs, "pair_hit"),
        "by_split": grouped_rates(surface_pairs, "split", "pair_hit"),
    }
    thresholds = prereg["thresholds"]
    surface_gap = abs(direction["by_surface"].get("base", 0.0) - direction["by_surface"].get("synonym", 0.0))
    gates = {
        "finite_fraction": summary["finite_fraction"] >= thresholds["minimum_candidate_finite_fraction"],
        "overall_candidate_accuracy": candidate["overall"] >= thresholds["minimum_overall_candidate_accuracy"],
        "split_candidate_accuracy": minimum(candidate["by_split"]) >= thresholds["minimum_split_candidate_accuracy"],
        "surface_candidate_accuracy": minimum(candidate["by_surface"]) >= thresholds["minimum_surface_candidate_accuracy"],
        "sense_candidate_accuracy": min(minimum(candidate["by_context_sense"]), minimum(candidate["by_definition_sense"])) >= thresholds["minimum_sense_candidate_accuracy"],
        "template_candidate_accuracy": minimum(candidate["by_template"]) >= thresholds["minimum_template_candidate_accuracy"],
        "interaction_direction_accuracy": direction["overall"] >= thresholds["minimum_interaction_direction_accuracy"],
        "split_interaction_direction_accuracy": minimum(direction["by_split"]) >= thresholds["minimum_split_interaction_direction_accuracy"],
        "surface_interaction_direction_accuracy": minimum(direction["by_surface"]) >= thresholds["minimum_surface_interaction_direction_accuracy"],
        "template_interaction_direction_accuracy": minimum(direction["by_template"]) >= thresholds["minimum_template_interaction_direction_accuracy"],
        "cross_surface_pair_accuracy": cross_surface["overall"] >= thresholds["minimum_cross_surface_pair_accuracy"],
        "split_cross_surface_pair_accuracy": minimum(cross_surface["by_split"]) >= thresholds["minimum_split_cross_surface_pair_accuracy"],
        "surface_direction_gap": surface_gap <= thresholds["maximum_surface_direction_accuracy_gap"],
    }
    return {
        "model": model_name,
        "case_count": len(detail),
        "finite_case_count": len(finite),
        "interaction_count": len(interactions),
        "cross_surface_pair_count": len(surface_pairs),
        "candidate": candidate,
        "interaction_direction": direction,
        "cross_surface": cross_surface,
        "surface_direction_accuracy_gap": surface_gap,
        "gates": gates,
        "qualified": all(gates.values()),
        "behavior_summary_digest": summary["summary_digest"],
        "detail_digest": summary["detail_digest"],
    }


def compute_final() -> dict[str, Any]:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not protocol_audit["all_checks_passed"]:
        raise RuntimeError("Phase1121 protocol audit failed")
    models = {model: evaluate_model(model, prereg) for model in protocol.MODELS}
    qualified_reference = [model for model in protocol.REFERENCE_MODELS if models[model]["qualified"]]
    authorization = models["pythia"]["qualified"] and len(qualified_reference) >= prereg["thresholds"]["minimum_qualified_reference_models"]
    predictions = {
        "P1_protocol_integrity": protocol_audit["all_checks_passed"],
        "P2_cross_model_behavior_authorization": authorization,
        "P3_truth_balanced_interaction": authorization and all(models[model]["gates"]["interaction_direction_accuracy"] for model in ["pythia", *qualified_reference]),
        "P4_cross_surface_pairing": authorization and all(models[model]["gates"]["cross_surface_pair_accuracy"] for model in ["pythia", *qualified_reference]),
        "P5_hidden_replication_authorized": authorization,
        "P6_scope_limit_preserved": True,
    }
    core = {
        "schema_version": "phase1121_adjective_double_orthogonal_final_summary.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "protocol_audit_digest": protocol_audit["audit_digest"],
        "models": models,
        "qualified_reference_models": qualified_reference,
        "pythia_qualified": models["pythia"]["qualified"],
        "hidden_trajectory_authorized": authorization,
        "predictions": predictions,
        "interpretation": {
            "authorized": "The new adjective material supports a separately frozen Pythia hidden-formation replication; no layer or component is selected here.",
            "not_authorized": "The independent material failed its frozen behavior qualification. Hidden scanning is denied; this is a behavior/material boundary, not proof that semantic geometry is absent.",
        }["authorized" if authorization else "not_authorized"],
    }
    final = dict(core)
    final["final_digest"] = protocol.digest(core)
    return final


def main() -> None:
    final = compute_final()
    protocol.write_json(protocol.OUT_ROOT / "analysis" / "final_summary.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
