"""Independent audit for Phase1200 conditional causal trace mapping."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1200_conditional_causal_trace_mapper as p  # noqa: E402


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return np.zeros_like(vector) if norm <= 1e-15 else vector / norm


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(unit(left), unit(right)))


def center(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    return np.mean(np.asarray([row[field] for row in rows], dtype=np.float64), axis=0)


def vector_accuracy(
    support: list[dict[str, Any]], evaluation: list[dict[str, Any]], field: str, label_field: str
) -> tuple[float, dict[int, np.ndarray]]:
    labels = sorted({int(row[label_field]) for row in support})
    centers = {
        label: center([row for row in support if int(row[label_field]) == label], field)
        for label in labels
    }
    correct = 0
    for row in evaluation:
        feature = np.asarray(row[field], dtype=np.float64)
        prediction = labels[int(np.argmax([cosine(feature, centers[label]) for label in labels]))]
        correct += prediction == int(row[label_field])
    return correct / max(len(evaluation), 1), centers


def scalar_accuracy(
    support: list[dict[str, Any]], evaluation: list[dict[str, Any]], field: str, label_field: str
) -> float:
    labels = sorted({int(row[label_field]) for row in support})
    centers = {
        label: float(np.mean([row[field] for row in support if int(row[label_field]) == label]))
        for label in labels
    }
    correct = 0
    for row in evaluation:
        prediction = min(labels, key=lambda label: (abs(float(row[field]) - centers[label]), label))
        correct += prediction == int(row[label_field])
    return correct / max(len(evaluation), 1)


def independent_summary(payload: dict[str, Any]) -> dict[str, Any]:
    family_rows = payload["family_rows"]
    support = [row for row in family_rows if row["role"] == "support"]
    evaluation = [row for row in family_rows if row["role"] == "evaluation"]
    invariant, _ = vector_accuracy(support, evaluation, "signature", "family")
    raw, _ = vector_accuracy(support, evaluation, "raw", "family")
    null, _ = vector_accuracy(support, evaluation, "null_signature", "family")
    shuffled, _ = vector_accuracy(support, evaluation, "shuffled_signature", "family")
    sentinel, _ = vector_accuracy(support, evaluation, "sentinel", "family")
    norm_only = scalar_accuracy(support, evaluation, "norm_only", "family")
    baseline = max(raw, null, shuffled, norm_only)

    shell_center = unit(center(support, "null_signature"))
    shell_cosines = [cosine(np.asarray(row["null_signature"]), shell_center) for row in evaluation]

    base_grams = {
        family: center([row for row in support if row["family"] == family], "gram_core")
        for family in range(p.N_PRIMITIVES)
    }
    predictions = {pair: base_grams[pair[0]] + base_grams[pair[1]] for pair in p.ALL_PAIRS}
    additive_scores: list[float] = []
    additive_top1: list[bool] = []
    additive_residuals: list[float] = []
    interaction_residuals: list[float] = []
    truth: list[bool] = []
    predicted_interaction: list[bool] = []
    for row in payload["pair_rows"]:
        observed = np.asarray(row["gram_core"], dtype=np.float64)
        pair = tuple(int(value) for value in row["pair"])
        scores = {candidate: cosine(observed, value) for candidate, value in predictions.items()}
        score = scores[pair]
        residual = 1.0 - score
        is_interaction = bool(row["interacting"])
        truth.append(is_interaction)
        predicted_interaction.append(residual >= 0.10)
        if is_interaction:
            interaction_residuals.append(residual)
        else:
            additive_scores.append(score)
            additive_top1.append(max(scores, key=scores.get) == pair)
            additive_residuals.append(residual)
    class_accuracies = []
    for target in (False, True):
        indices = [index for index, value in enumerate(truth) if value == target]
        class_accuracies.append(float(np.mean([predicted_interaction[index] == target for index in indices])))
    interaction_balanced = float(np.mean(class_accuracies))

    twin_support = [row for row in payload["twin_rows"] if row["role"] == "support"]
    twin_evaluation = [row for row in payload["twin_rows"] if row["role"] == "evaluation"]
    core_accuracy, core_centers = vector_accuracy(twin_support, twin_evaluation, "core_signature", "label")
    extended_accuracy, extended_centers = vector_accuracy(
        twin_support, twin_evaluation, "extended_signature", "label"
    )

    metrics = {
        "family_invariant_accuracy": invariant,
        "family_raw_coordinate_accuracy": raw,
        "family_matched_null_accuracy": null,
        "family_shuffled_registry_accuracy": shuffled,
        "family_norm_only_accuracy": norm_only,
        "family_leakage_sentinel_accuracy": sentinel,
        "family_invariant_advantage": invariant - baseline,
        "shared_shell_cosine_mean": float(np.mean(shell_cosines)),
        "shared_shell_cosine_min": float(np.min(shell_cosines)),
        "additive_composition_top1_accuracy": float(np.mean(additive_top1)),
        "additive_composition_cosine_mean": float(np.mean(additive_scores)),
        "interaction_balanced_accuracy": interaction_balanced,
        "additive_residual_mean": float(np.mean(additive_residuals)),
        "interaction_residual_mean": float(np.mean(interaction_residuals)),
        "interaction_residual_gap": float(np.mean(interaction_residuals) - np.mean(additive_residuals)),
        "core_twin_accuracy": core_accuracy,
        "core_twin_centroid_distance": float(np.linalg.norm(core_centers[0] - core_centers[1])),
        "extended_twin_accuracy": extended_accuracy,
        "extended_twin_centroid_distance": float(np.linalg.norm(extended_centers[0] - extended_centers[1])),
    }
    t = p.THRESHOLDS
    gates = {
        "family_mapping": bool(
            metrics["family_invariant_accuracy"] >= t["family_invariant_accuracy_min"]
            and metrics["family_invariant_advantage"] >= t["family_invariant_advantage_min"]
            and metrics["family_raw_coordinate_accuracy"] <= t["raw_coordinate_accuracy_max"]
            and metrics["family_matched_null_accuracy"] <= t["matched_null_accuracy_max"]
            and metrics["family_shuffled_registry_accuracy"] <= t["shuffled_registry_accuracy_max"]
            and metrics["family_norm_only_accuracy"] <= t["norm_only_accuracy_max"]
            and metrics["family_leakage_sentinel_accuracy"] >= t["leakage_sentinel_accuracy_min"]
        ),
        "shared_reuse": bool(
            metrics["shared_shell_cosine_mean"] >= t["shared_shell_cosine_mean_min"]
            and metrics["shared_shell_cosine_min"] >= t["shared_shell_cosine_min_min"]
        ),
        "composition": bool(
            metrics["additive_composition_top1_accuracy"] >= t["additive_composition_top1_min"]
            and metrics["additive_composition_cosine_mean"] >= t["additive_composition_cosine_mean_min"]
            and metrics["interaction_balanced_accuracy"] >= t["interaction_balanced_accuracy_min"]
            and metrics["interaction_residual_gap"] >= t["interaction_residual_gap_min"]
        ),
        "identifiability_boundary": bool(
            metrics["core_twin_accuracy"] <= t["core_twin_accuracy_max"]
            and metrics["core_twin_centroid_distance"] <= t["core_twin_centroid_distance_max"]
            and metrics["extended_twin_accuracy"] >= t["extended_twin_accuracy_min"]
            and metrics["extended_twin_centroid_distance"] >= t["extended_twin_centroid_distance_min"]
        ),
    }
    return {
        "corpus": payload["corpus"],
        "seed": payload["seed"],
        "counts": {
            "family_rows": len(payload["family_rows"]),
            "pair_rows": len(payload["pair_rows"]),
            "twin_rows": len(payload["twin_rows"]),
            "pair_families": len(payload["pair_assignment"]),
            "additive_pair_rows": sum(not row["interacting"] for row in payload["pair_rows"]),
            "interaction_pair_rows": sum(row["interacting"] for row in payload["pair_rows"]),
        },
        "metrics": metrics,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def common_payload_checks(checks: list[dict[str, Any]], payload: dict[str, Any], corpus: str) -> None:
    expected_pairs = len(p.pair_assignment()[corpus])
    add(checks, f"{corpus}_family_rows", len(payload["family_rows"]) == p.N_PRIMITIVES * p.CONTEXTS * p.REPLICATES)
    add(checks, f"{corpus}_pair_rows", len(payload["pair_rows"]) == expected_pairs * len(p.EVALUATION_CONTEXTS) * p.REPLICATES)
    add(checks, f"{corpus}_twin_rows", len(payload["twin_rows"]) == p.CONTEXTS * p.TWIN_REPLICATES * 2)
    add(checks, f"{corpus}_pair_assignment", payload["pair_assignment"] == [list(pair) for pair in p.pair_assignment()[corpus]])
    family_balance = {
        family: sum(row["family"] == family for row in payload["family_rows"])
        for family in range(p.N_PRIMITIVES)
    }
    add(checks, f"{corpus}_family_balance", len(set(family_balance.values())) == 1, family_balance)
    finite = True
    for row in payload["family_rows"]:
        for field in ("gram_core", "signature", "raw", "null_signature", "shuffled_signature", "sentinel"):
            finite &= bool(np.all(np.isfinite(np.asarray(row[field], dtype=np.float64))))
    for row in payload["pair_rows"]:
        finite &= bool(np.all(np.isfinite(np.asarray(row["gram_core"], dtype=np.float64))))
    for row in payload["twin_rows"]:
        finite &= bool(np.all(np.isfinite(np.asarray(row["core_signature"], dtype=np.float64))))
        finite &= bool(np.all(np.isfinite(np.asarray(row["extended_signature"], dtype=np.float64))))
    add(checks, f"{corpus}_all_finite", finite)

    twin_lookup = {
        (row["context"], row["replicate"], row["label"]): row for row in payload["twin_rows"]
    }
    core_equal = True
    extended_different = True
    for context in range(p.CONTEXTS):
        for replicate in range(p.TWIN_REPLICATES):
            left = twin_lookup[(context, replicate, 0)]
            right = twin_lookup[(context, replicate, 1)]
            core_equal &= np.array_equal(
                np.asarray(left["core_signature"]), np.asarray(right["core_signature"])
            )
            extended_different &= not np.array_equal(
                np.asarray(left["extended_signature"]), np.asarray(right["extended_signature"])
            )
    add(checks, f"{corpus}_core_twins_exact", core_equal)
    add(checks, f"{corpus}_extended_twins_separate", extended_different)


def metric_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        if isinstance(left[key], dict):
            if not metric_match(left[key], right[key]):
                return False
        elif isinstance(left[key], float):
            if not close(left[key], right[key]):
                return False
        else:
            if left[key] != right[key]:
                return False
    return True


def audit_development(write: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    payload = p.read_json(p.DEVELOPMENT_ROWS)
    summary_file = p.read_json(p.DEVELOPMENT_SUMMARY)
    regenerated = p.build_corpus("development", p.DEVELOPMENT_SEED)
    add(checks, "development_regeneration", p.digest(payload) == p.digest(regenerated))
    common_payload_checks(checks, payload, "development")
    independent = independent_summary(payload)
    add(checks, "development_summary_recompute", metric_match(independent, summary_file["summary"]))
    add(checks, "development_decision_recompute", summary_file["development_gate_pass"] == independent["overall_pass"])
    gate = all(check["pass"] for check in checks)
    output = {
        "phase": p.PHASE,
        "kind": "independent_development_audit",
        "gate_pass": gate,
        "checks_passed": sum(check["pass"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    output["audit_digest"] = p.digest(output)
    if write:
        p.write_json(p.DEVELOPMENT_AUDIT, output)
    return output


def audit_formal(write: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    protocol = p.read_json(p.PROTOCOL_PATH)
    protocol_candidate = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    add(checks, "protocol_digest", p.digest(protocol_candidate) == protocol["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == p.source_hashes())
    add(
        checks,
        "development_assets",
        protocol["development"]
        == {
            "rows_sha256": p.file_sha256(p.DEVELOPMENT_ROWS),
            "summary_sha256": p.file_sha256(p.DEVELOPMENT_SUMMARY),
            "audit_sha256": p.file_sha256(p.DEVELOPMENT_AUDIT),
        },
    )
    seal = p.read_json(p.FORMAL_SEAL)
    seal_candidate = {key: value for key, value in seal.items() if key != "seal_digest"}
    add(checks, "seal_digest", p.digest(seal_candidate) == seal["seal_digest"])
    add(checks, "seal_protocol", seal["protocol_digest"] == protocol["protocol_digest"])
    add(
        checks,
        "formal_row_hashes",
        seal["row_hashes"]
        == {
            "discovery": p.file_sha256(p.FORMAL_DISCOVERY_ROWS),
            "confirmation": p.file_sha256(p.FORMAL_CONFIRMATION_ROWS),
        },
    )

    summary_file = p.read_json(p.SUMMARY_PATH)
    claims = p.read_json(p.CLAIMS_PATH)
    summaries = {}
    for corpus, path in (
        ("discovery", p.FORMAL_DISCOVERY_ROWS),
        ("confirmation", p.FORMAL_CONFIRMATION_ROWS),
    ):
        payload = p.read_json(path)
        regenerated = p.build_corpus(corpus, p.FORMAL_SEEDS[corpus])
        add(checks, f"{corpus}_regeneration", p.digest(payload) == p.digest(regenerated))
        common_payload_checks(checks, payload, corpus)
        independent = independent_summary(payload)
        summaries[corpus] = independent
        add(checks, f"{corpus}_summary_recompute", metric_match(independent, summary_file[corpus]))

    discovery_pairs = {tuple(pair) for pair in p.pair_assignment()["discovery"]}
    confirmation_pairs = {tuple(pair) for pair in p.pair_assignment()["confirmation"]}
    add(checks, "formal_pair_splits_disjoint", discovery_pairs.isdisjoint(confirmation_pairs))
    add(checks, "formal_pair_splits_complete", discovery_pairs | confirmation_pairs == set(p.ALL_PAIRS))
    positive = summaries["discovery"]["overall_pass"] and summaries["confirmation"]["overall_pass"]
    add(checks, "decision_recompute", summary_file["formal_decision"] == ("positive" if positive else "not_confirmed"))
    expected_type = "E3-KT" if positive else "E3-KT-scope-boundary"
    add(checks, "claim_type", claims["conditional_causal_trace_mapper"]["type"] == expected_type)
    gate = all(check["pass"] for check in checks)
    output = {
        "phase": p.PHASE,
        "kind": "independent_formal_audit",
        "gate_pass": gate,
        "checks_passed": sum(check["pass"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "independent_summaries": summaries,
    }
    output["audit_digest"] = p.digest(output)
    if write:
        p.write_json(p.AUDIT_PATH, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    output = audit_development(args.write) if args.development else audit_formal(args.write)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not output["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
