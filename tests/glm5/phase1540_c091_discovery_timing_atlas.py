#!/usr/bin/env python3
"""Phase1540: discovery atlas for behavior-grounded truth and factorial timing responses."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1536_c091_human_validated_chinese_relation_contract"
SCOPE = RESULT / "phase1538_c091_behavior_gate_adjudication"
PARENT = RESULT / "phase1539_c091_canonical_all_state_capture"
OUT = RESULT / "phase1540_c091_discovery_timing_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

ROLES = ("source_word", "target_word", "relation_anchor", "boundary")
SURFACES = ("prequery", "postquery")
CONTROLS = ("similarity", "class_inclusion")
DISCOVERY = "response_discovery"


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def metrics(vectors: list[np.ndarray], labels: list[tuple[str, str]]) -> dict:
    stack = np.stack(vectors).astype(np.float64)
    centroid = stack.mean(axis=0)
    by_control = {control: stack[[i for i, value in enumerate(labels) if value[0] == control]].mean(axis=0) for control in CONTROLS}
    by_concreteness = {value: stack[[i for i, label in enumerate(labels) if label[1] == value]].mean(axis=0) for value in ("concrete", "abstract")}
    alignments = [cosine(vector, centroid) for vector in stack]
    return {
        "centroid_norm": float(np.linalg.norm(centroid)),
        "control_cosine": cosine(by_control[CONTROLS[0]], by_control[CONTROLS[1]]),
        "concreteness_cosine": cosine(by_concreteness["concrete"], by_concreteness["abstract"]),
        "median_individual_alignment": float(np.median(alignments)),
        "minimum_individual_alignment": float(np.min(alignments)),
        "centroid": centroid.astype(np.float32),
    }


def matched_pairs(pairs: list[dict], family: str, concreteness: str) -> list[dict]:
    return sorted(
        [row for row in pairs if row["partition"] == DISCOVERY and row["family"] == family and row["concreteness"] == concreteness],
        key=lambda row: row["pair_id"],
    )


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1540 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    scope = core.load(SCOPE / "protocol/frozen_behavior_routes_and_hidden_scope.json")
    if parent["authorization"] != "run_phase1540_c091_discovery_timing_atlas" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1539 authorization missing")
    if scope["qualified_families"] != ["whole_part"]:
        raise RuntimeError("unexpected semantic scope")
    field = np.load(PARENT / "raw/canonical_all_role_field.float16.npy", mmap_mode="r")
    index = core.rows(PARENT / "raw/canonical_all_role_field_index.jsonl")
    pairs = core.rows(CONTRACT / "material/frozen_pairs.jsonl")
    lookup = {(row["pair_id"], row["surface"], row["query_family"]): row for row in index}

    interaction_vectors: dict[str, list[np.ndarray]] = {surface: [] for surface in SURFACES}
    interaction_labels: dict[str, list[tuple[str, str]]] = {surface: [] for surface in SURFACES}
    truth_vectors: dict[str, list[np.ndarray]] = {surface: [] for surface in SURFACES}
    truth_labels: dict[str, list[tuple[str, str]]] = {surface: [] for surface in SURFACES}
    for surface in SURFACES:
        for concreteness in ("concrete", "abstract"):
            whole = matched_pairs(pairs, "whole_part", concreteness)
            similarity = matched_pairs(pairs, "similarity", concreteness)
            category = matched_pairs(pairs, "class_inclusion", concreteness)
            for control, control_pairs in (("similarity", similarity), ("class_inclusion", category)):
                for whole_pair, control_pair in zip(whole, control_pairs, strict=True):
                    pw_qw = field[lookup[(whole_pair["pair_id"], surface, "whole_part")]["row_index"]].astype(np.float32)
                    pg_qg = field[lookup[(control_pair["pair_id"], surface, control)]["row_index"]].astype(np.float32)
                    pw_qg = field[lookup[(whole_pair["pair_id"], surface, control)]["row_index"]].astype(np.float32)
                    pg_qw = field[lookup[(control_pair["pair_id"], surface, "whole_part")]["row_index"]].astype(np.float32)
                    interaction_vectors[surface].append(0.5 * (pw_qw + pg_qg - pw_qg - pg_qw))
                    interaction_labels[surface].append((control, concreteness))
            for whole_pair, sim_pair, class_pair in zip(whole, similarity, category, strict=True):
                hw = field[lookup[(whole_pair["pair_id"], surface, "whole_part")]["row_index"]].astype(np.float32)
                hs = field[lookup[(sim_pair["pair_id"], surface, "whole_part")]["row_index"]].astype(np.float32)
                hc = field[lookup[(class_pair["pair_id"], surface, "whole_part")]["row_index"]].astype(np.float32)
                raw = hw - 0.5 * (hs + hc)
                dynamic = raw - raw[0:1]
                truth_vectors[surface].append(dynamic)
                truth_labels[surface].append(("truth", concreteness))

    interaction_atlas = []
    truth_atlas = []
    interaction_centroids = {}
    truth_centroids = {}
    for surface in SURFACES:
        for state in range(37):
            for role_index, role in enumerate(ROLES):
                values = [vector[state, role_index] for vector in interaction_vectors[surface]]
                item = metrics(values, interaction_labels[surface])
                key = f"{surface}|{state}|{role}"
                interaction_centroids[key] = item.pop("centroid")
                interaction_atlas.append({"surface": surface, "state": state, "role": role, **item})
                truth_values = [vector[state, role_index] for vector in truth_vectors[surface]]
                truth_stack = np.stack(truth_values).astype(np.float64)
                truth_centroid = truth_stack.mean(axis=0)
                concrete = truth_stack[[i for i, label in enumerate(truth_labels[surface]) if label[1] == "concrete"]].mean(axis=0)
                abstract = truth_stack[[i for i, label in enumerate(truth_labels[surface]) if label[1] == "abstract"]].mean(axis=0)
                alignments = [cosine(vector, truth_centroid) for vector in truth_stack]
                truth_centroids[key] = truth_centroid.astype(np.float32)
                truth_atlas.append({
                    "surface": surface,
                    "state": state,
                    "role": role,
                    "centroid_norm": float(np.linalg.norm(truth_centroid)),
                    "concreteness_cosine": cosine(concrete, abstract),
                    "median_individual_alignment": float(np.median(alignments)),
                    "minimum_individual_alignment": float(np.min(alignments)),
                })

    causal_nulls = {
        "prequery_relation_anchor_max_norm": max(row["centroid_norm"] for row in interaction_atlas if row["surface"] == "prequery" and row["role"] == "relation_anchor"),
        "postquery_source_max_norm": max(row["centroid_norm"] for row in interaction_atlas if row["surface"] == "postquery" and row["role"] == "source_word"),
        "postquery_target_max_norm": max(row["centroid_norm"] for row in interaction_atlas if row["surface"] == "postquery" and row["role"] == "target_word"),
    }

    eligible_roles = {
        "prequery": {"target_word", "boundary"},
        "postquery": {"relation_anchor", "boundary"},
    }
    interaction_candidates = {}
    truth_candidates = {}
    candidate_vectors = []
    for surface in SURFACES:
        eligible = [row for row in interaction_atlas if row["surface"] == surface and row["state"] >= 1 and row["role"] in eligible_roles[surface]]
        for row in eligible:
            row["selection_score"] = min(row["control_cosine"], row["concreteness_cosine"], row["median_individual_alignment"])
        winner = max(eligible, key=lambda row: (row["selection_score"], row["centroid_norm"], -row["state"], row["role"]))
        winner = {**winner, "discovery_candidate_pass": winner["centroid_norm"] > 1e-6 and winner["control_cosine"] >= 0.5 and winner["concreteness_cosine"] >= 0.5 and winner["median_individual_alignment"] >= 0.0}
        interaction_candidates[surface] = winner
        candidate_vectors.append(interaction_centroids[f"{surface}|{winner['state']}|{winner['role']}"])

        eligible_truth = [row for row in truth_atlas if row["surface"] == surface and row["state"] >= 1 and row["role"] in eligible_roles[surface]]
        for row in eligible_truth:
            row["selection_score"] = min(row["concreteness_cosine"], row["median_individual_alignment"])
        truth_winner = max(eligible_truth, key=lambda row: (row["selection_score"], row["centroid_norm"], -row["state"], row["role"]))
        truth_winner = {**truth_winner, "discovery_candidate_pass": truth_winner["centroid_norm"] > 1e-6 and truth_winner["concreteness_cosine"] >= 0.5 and truth_winner["median_individual_alignment"] >= 0.0}
        truth_candidates[surface] = truth_winner
        candidate_vectors.append(truth_centroids[f"{surface}|{truth_winner['state']}|{truth_winner['role']}"])

    cross_surface = {
        "interaction_candidate_cosine": cosine(candidate_vectors[0], candidate_vectors[2]),
        "truth_candidate_cosine": cosine(candidate_vectors[1], candidate_vectors[3]),
    }
    interaction_path = OUT / "analysis/discovery_factorial_interaction_atlas.jsonl"
    truth_path = OUT / "analysis/discovery_behavior_grounded_truth_atlas.jsonl"
    vector_path = OUT / "raw/discovery_candidate_centroids.float32.npy"
    core.write_rows(interaction_path, interaction_atlas)
    core.write_rows(truth_path, truth_atlas)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vector_path, np.stack(candidate_vectors).astype(np.float32), allow_pickle=False)
    checks = {
        "parent_audited": True,
        "interaction_coverage": len(interaction_atlas) == 2 * 37 * 4,
        "truth_coverage": len(truth_atlas) == 2 * 37 * 4,
        "vector_shape": list(np.load(vector_path).shape) == [4, 2560],
        "prequery_anchor_null": causal_nulls["prequery_relation_anchor_max_norm"] == 0.0,
        "postquery_source_null": causal_nulls["postquery_source_max_norm"] == 0.0,
        "postquery_target_null": causal_nulls["postquery_target_max_norm"] == 0.0,
        "interaction_candidates": all(row["discovery_candidate_pass"] for row in interaction_candidates.values()),
        "truth_candidates": all(row["discovery_candidate_pass"] for row in truth_candidates.values()),
    }
    status = "discovery_candidates_frozen" if checks["interaction_candidates"] and checks["truth_candidates"] else "discovery_candidate_gate_failed"
    report = {
        "phase": 1540,
        "campaign": "C091",
        "status": status,
        "causal_nulls": causal_nulls,
        "factorial_interaction_candidates": interaction_candidates,
        "behavior_grounded_truth_candidates": truth_candidates,
        "cross_surface_discovery": cross_surface,
        "candidate_vector_order": [
            "prequery_factorial_interaction",
            "prequery_behavior_grounded_truth",
            "postquery_factorial_interaction",
            "postquery_behavior_grounded_truth",
        ],
        "frozen_holdout_gate": {
            "partitions": ["response_confirmation", "lockbox"],
            "same_formula_state_and_role": True,
            "centroid_cosine_to_discovery_min": 0.5,
            "concreteness_cosine_min": 0.3,
            "median_individual_alignment_min": 0.0,
            "factorial_control_cosine_min": 0.3,
            "all_candidates_all_partitions_required": True,
            "failure_action": "close_C091_without_hidden_mechanism_claim",
        },
        "checks": checks,
        "files": {
            "interaction_atlas": {"sha256": core.sha(interaction_path), "rows": len(interaction_atlas)},
            "truth_atlas": {"sha256": core.sha(truth_path), "rows": len(truth_atlas)},
            "candidate_centroids": {"sha256": core.sha(vector_path), "shape": [4, 2560]},
        },
        "claim_boundary": {
            "factorial": "lexical/query main effects canceled but two control queries lack behavior qualification",
            "truth": "whole-part query behavior-qualified but lexical family identity is not exactly canceled",
            "forbidden": ["causal transport", "semantic neuron", "universal timing law", "new mathematics"],
        },
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "protocol/frozen_discovery_candidates.json", report)
    core.save(OUT / "analysis/discovery_timing_atlas_summary.json", report)
    authorization = "run_phase1541_c091_dual_holdout_timing_validation" if status == "discovery_candidates_frozen" else "run_phase1542_c091_route_closure"
    core.save(OUT / "analysis/final.json", {"phase": 1540, "campaign": "C091", "status": status, "authorization": authorization})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
