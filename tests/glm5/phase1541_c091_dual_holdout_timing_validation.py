#!/usr/bin/env python3
"""Phase1541: reveal confirmation and lockbox for the four frozen C091 candidates."""
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
FIELD_DIR = RESULT / "phase1539_c091_canonical_all_state_capture"
PARENT = RESULT / "phase1540_c091_discovery_timing_atlas"
OUT = RESULT / "phase1541_c091_dual_holdout_timing_validation"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

ROLES = ("source_word", "target_word", "relation_anchor", "boundary")
SURFACES = ("prequery", "postquery")
CONTROLS = ("similarity", "class_inclusion")
PARTITIONS = ("confirmation", "lockbox")


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def selected_pairs(pairs: list[dict], partition: str, family: str, concreteness: str) -> list[dict]:
    return sorted(
        [row for row in pairs if row["partition"] == partition and row["family"] == family and row["concreteness"] == concreteness],
        key=lambda row: row["pair_id"],
    )


def interaction_vectors(field: np.ndarray, lookup: dict, pairs: list[dict], partition: str, surface: str, state: int, role: str) -> tuple[list[np.ndarray], list[tuple[str, str]]]:
    role_index = ROLES.index(role)
    vectors: list[np.ndarray] = []
    labels: list[tuple[str, str]] = []
    for concreteness in ("concrete", "abstract"):
        whole = selected_pairs(pairs, partition, "whole_part", concreteness)
        for control in CONTROLS:
            control_pairs = selected_pairs(pairs, partition, control, concreteness)
            for whole_pair, control_pair in zip(whole, control_pairs, strict=True):
                pw_qw = field[lookup[(whole_pair["pair_id"], surface, "whole_part")]["row_index"], state, role_index].astype(np.float32)
                pg_qg = field[lookup[(control_pair["pair_id"], surface, control)]["row_index"], state, role_index].astype(np.float32)
                pw_qg = field[lookup[(whole_pair["pair_id"], surface, control)]["row_index"], state, role_index].astype(np.float32)
                pg_qw = field[lookup[(control_pair["pair_id"], surface, "whole_part")]["row_index"], state, role_index].astype(np.float32)
                vectors.append(0.5 * (pw_qw + pg_qg - pw_qg - pg_qw))
                labels.append((control, concreteness))
    return vectors, labels


def truth_vectors(field: np.ndarray, lookup: dict, pairs: list[dict], partition: str, surface: str, state: int, role: str) -> tuple[list[np.ndarray], list[str]]:
    role_index = ROLES.index(role)
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    for concreteness in ("concrete", "abstract"):
        whole = selected_pairs(pairs, partition, "whole_part", concreteness)
        similarity = selected_pairs(pairs, partition, "similarity", concreteness)
        category = selected_pairs(pairs, partition, "class_inclusion", concreteness)
        for whole_pair, sim_pair, class_pair in zip(whole, similarity, category, strict=True):
            hw = field[lookup[(whole_pair["pair_id"], surface, "whole_part")]["row_index"]].astype(np.float32)
            hs = field[lookup[(sim_pair["pair_id"], surface, "whole_part")]["row_index"]].astype(np.float32)
            hc = field[lookup[(class_pair["pair_id"], surface, "whole_part")]["row_index"]].astype(np.float32)
            raw = hw - 0.5 * (hs + hc)
            vectors.append((raw[state, role_index] - raw[0, role_index]).astype(np.float32))
            labels.append(concreteness)
    return vectors, labels


def summarize(vectors: list[np.ndarray], concreteness: list[str], discovery: np.ndarray, controls: list[str] | None = None) -> tuple[dict, np.ndarray]:
    stack = np.stack(vectors).astype(np.float64)
    centroid = stack.mean(axis=0)
    by_concreteness = {
        value: stack[[i for i, label in enumerate(concreteness) if label == value]].mean(axis=0)
        for value in ("concrete", "abstract")
    }
    alignments = [cosine(vector, centroid) for vector in stack]
    result = {
        "n_vectors": len(vectors),
        "centroid_norm": float(np.linalg.norm(centroid)),
        "centroid_cosine_to_discovery": cosine(centroid, discovery),
        "concreteness_cosine": cosine(by_concreteness["concrete"], by_concreteness["abstract"]),
        "median_individual_alignment": float(np.median(alignments)),
        "minimum_individual_alignment": float(np.min(alignments)),
    }
    if controls is not None:
        by_control = {
            value: stack[[i for i, label in enumerate(controls) if label == value]].mean(axis=0)
            for value in CONTROLS
        }
        result["control_cosine"] = cosine(by_control[CONTROLS[0]], by_control[CONTROLS[1]])
    return result, centroid.astype(np.float32)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1541 exists")
    final = core.load(PARENT / "analysis/final.json")
    audit = core.load(PARENT / "audit/independent_final_audit.json")
    frozen = core.load(PARENT / "protocol/frozen_discovery_candidates.json")
    if final["authorization"] != "run_phase1541_c091_dual_holdout_timing_validation" or not audit["all_checks_passed"]:
        raise RuntimeError("Phase1540 authorization missing")
    gate = frozen["frozen_holdout_gate"]
    discovery_vectors = np.load(PARENT / "raw/discovery_candidate_centroids.float32.npy")
    field = np.load(FIELD_DIR / "raw/canonical_all_role_field.float16.npy", mmap_mode="r")
    index = core.rows(FIELD_DIR / "raw/canonical_all_role_field_index.jsonl")
    pairs = core.rows(CONTRACT / "material/frozen_pairs.jsonl")
    lookup = {(row["pair_id"], row["surface"], row["query_family"]): row for row in index}
    vector_index = {
        ("factorial_interaction", "prequery"): 0,
        ("behavior_grounded_truth", "prequery"): 1,
        ("factorial_interaction", "postquery"): 2,
        ("behavior_grounded_truth", "postquery"): 3,
    }

    rows = []
    holdout_centroids = []
    for partition in PARTITIONS:
        for surface in SURFACES:
            candidate = frozen["factorial_interaction_candidates"][surface]
            vectors, labels = interaction_vectors(field, lookup, pairs, partition, surface, candidate["state"], candidate["role"])
            summary, centroid = summarize(
                vectors,
                [label[1] for label in labels],
                discovery_vectors[vector_index[("factorial_interaction", surface)]],
                [label[0] for label in labels],
            )
            summary["gate_pass"] = (
                summary["centroid_cosine_to_discovery"] >= gate["centroid_cosine_to_discovery_min"]
                and summary["concreteness_cosine"] >= gate["concreteness_cosine_min"]
                and summary["median_individual_alignment"] >= gate["median_individual_alignment_min"]
                and summary["control_cosine"] >= gate["factorial_control_cosine_min"]
            )
            rows.append({"partition": partition, "object": "factorial_interaction", "surface": surface, "state": candidate["state"], "role": candidate["role"], **summary})
            holdout_centroids.append(centroid)

            candidate = frozen["behavior_grounded_truth_candidates"][surface]
            truth, truth_labels = truth_vectors(field, lookup, pairs, partition, surface, candidate["state"], candidate["role"])
            summary, centroid = summarize(
                truth,
                truth_labels,
                discovery_vectors[vector_index[("behavior_grounded_truth", surface)]],
            )
            summary["gate_pass"] = (
                summary["centroid_cosine_to_discovery"] >= gate["centroid_cosine_to_discovery_min"]
                and summary["concreteness_cosine"] >= gate["concreteness_cosine_min"]
                and summary["median_individual_alignment"] >= gate["median_individual_alignment_min"]
            )
            rows.append({"partition": partition, "object": "behavior_grounded_truth", "surface": surface, "state": candidate["state"], "role": candidate["role"], **summary})
            holdout_centroids.append(centroid)

    result_path = OUT / "analysis/dual_holdout_candidate_validation.jsonl"
    vector_path = OUT / "raw/dual_holdout_candidate_centroids.float32.npy"
    core.write_rows(result_path, rows)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vector_path, np.stack(holdout_centroids).astype(np.float32), allow_pickle=False)
    all_passed = all(row["gate_pass"] for row in rows) and len(rows) == 8
    status = "dual_holdout_gate_passed" if all_passed else "dual_holdout_gate_failed"
    report = {
        "phase": 1541,
        "campaign": "C091",
        "status": status,
        "frozen_gate": gate,
        "partition_alias_resolution": {
            "frozen_name": "response_confirmation",
            "contract_name": "confirmation",
            "resolved_before_first_vector_or_statistic": True,
            "materials_candidates_formulas_and_thresholds_changed": False,
        },
        "results": rows,
        "checks": {
            "parent_audited": True,
            "eight_frozen_validations": len(rows) == 8,
            "all_holdout_gates_passed": all_passed,
            "no_candidate_reselection": all(
                row["state"] == frozen[("factorial_interaction_candidates" if row["object"] == "factorial_interaction" else "behavior_grounded_truth_candidates")][row["surface"]]["state"]
                and row["role"] == frozen[("factorial_interaction_candidates" if row["object"] == "factorial_interaction" else "behavior_grounded_truth_candidates")][row["surface"]]["role"]
                for row in rows
            ),
        },
        "files": {
            "validation": {"sha256": core.sha(result_path), "rows": len(rows)},
            "centroids": {"sha256": core.sha(vector_path), "shape": list(np.load(vector_path).shape)},
        },
        "claim_boundary": {
            "supported_if_passed": "frozen late-boundary group response repeats across confirmation and lockbox in one Qwen3 contract",
            "not_supported": ["causal transport", "minimal neural mechanism", "universal relation code", "cross-model invariance", "new mathematics"],
        },
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/dual_holdout_summary.json", report)
    authorization = "run_phase1542_c091_final_adjudication" if all_passed else "run_phase1542_c091_route_closure"
    core.save(OUT / "analysis/final.json", {"phase": 1541, "campaign": "C091", "status": status, "authorization": authorization})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
