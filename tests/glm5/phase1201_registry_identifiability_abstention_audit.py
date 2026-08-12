"""Independent audit for the Phase1201 registry abstention compiler."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1200_conditional_causal_trace_mapper as p1200  # noqa: E402
import phase1201_registry_identifiability_abstention as p  # noqa: E402


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return np.zeros_like(vector) if norm <= 1e-15 else vector / norm


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(unit(left), unit(right)))


def own_compile(
    rows: list[dict[str, Any]], feature_field: str, label_field: str, task_name: str
) -> dict[str, Any]:
    support = [row for row in rows if row["role"] == "support"]
    evaluation = [row for row in rows if row["role"] == "evaluation"]
    labels = sorted({int(row[label_field]) for row in support})
    centers = {
        label: np.mean(
            np.asarray([row[feature_field] for row in support if int(row[label_field]) == label], dtype=np.float64),
            axis=0,
        )
        for label in labels
    }
    distance = min(
        float(np.linalg.norm(centers[left] - centers[right]))
        for left, right in itertools.combinations(labels, 2)
    )
    identifiable = distance > p.SEPARATION_TOLERANCE
    predictions: list[int | str] = []
    if identifiable:
        for row in evaluation:
            feature = np.asarray(row[feature_field], dtype=np.float64)
            scores = [cosine(feature, centers[label]) for label in labels]
            predictions.append(labels[int(np.argmax(scores))])
    else:
        predictions = [p.ABSTENTION_TOKEN] * len(evaluation)
    claims = sum(prediction != p.ABSTENTION_TOKEN for prediction in predictions)
    correct = sum(prediction == int(row[label_field]) for prediction, row in zip(predictions, evaluation))
    return {
        "task": task_name,
        "support_count": len(support),
        "evaluation_count": len(evaluation),
        "class_count": len(labels),
        "minimum_centroid_distance": distance,
        "separation_tolerance": p.SEPARATION_TOLERANCE,
        "identifiable": identifiable,
        "decision": "IDENTITY_AUTHORIZED" if identifiable else p.ABSTENTION_TOKEN,
        "identity_claim_count": claims,
        "abstention_count": len(evaluation) - claims,
        "accuracy_when_authorized": correct / max(len(evaluation), 1) if identifiable else None,
        "prediction_digest": p.digest(predictions),
    }


def own_corpus(payload: dict[str, Any]) -> dict[str, Any]:
    family = own_compile(payload["family_rows"], "signature", "family", "family_core_registry")
    core = own_compile(payload["twin_rows"], "core_signature", "label", "twin_core_registry")
    extended = own_compile(payload["twin_rows"], "extended_signature", "label", "twin_extended_registry")
    gates = {
        "family_identity_authorized": bool(
            family["identifiable"]
            and family["accuracy_when_authorized"] is not None
            and family["accuracy_when_authorized"] >= p.IDENTIFIABLE_ACCURACY_MIN
        ),
        "core_twin_identity_denied": bool(
            not core["identifiable"]
            and core["identity_claim_count"] == 0
            and core["abstention_count"] == core["evaluation_count"]
        ),
        "extended_twin_identity_authorized": bool(
            extended["identifiable"]
            and extended["accuracy_when_authorized"] is not None
            and extended["accuracy_when_authorized"] >= p.IDENTIFIABLE_ACCURACY_MIN
        ),
        "registry_extension_monotonicity": bool(
            extended["minimum_centroid_distance"] > core["minimum_centroid_distance"]
        ),
    }
    return {
        "corpus": payload["corpus"],
        "family": family,
        "twin_core": core,
        "twin_extended": extended,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def audit(write: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    protocol = p.read_json(p.PROTOCOL_PATH)
    candidate = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    add(checks, "protocol_digest", p.digest(candidate) == protocol["protocol_digest"])
    add(checks, "source_hashes", protocol["source_hashes"] == p.source_hashes())
    add(checks, "upstream_hashes", protocol["upstream_hashes"] == p.upstream_hashes())
    add(checks, "no_new_data", protocol["scope"]["new_data"] is False)
    add(checks, "no_new_model_run", protocol["scope"]["new_model_run"] is False)
    add(checks, "no_new_k_item", protocol["scope"]["new_k_item"] is False)

    summary = p.read_json(p.SUMMARY_PATH)
    claims = p.read_json(p.CLAIMS_PATH)
    recomputed: dict[str, Any] = {}
    for corpus, path in (
        ("discovery", p1200.FORMAL_DISCOVERY_ROWS),
        ("confirmation", p1200.FORMAL_CONFIRMATION_ROWS),
    ):
        payload = p.read_json(path)
        own = own_corpus(payload)
        recomputed[corpus] = own
        add(checks, f"{corpus}_recompute", equal(own, summary[corpus]))
        add(checks, f"{corpus}_family_authorized", own["family"]["decision"] == "IDENTITY_AUTHORIZED")
        add(checks, f"{corpus}_core_abstains", own["twin_core"]["decision"] == p.ABSTENTION_TOKEN)
        add(checks, f"{corpus}_core_zero_claims", own["twin_core"]["identity_claim_count"] == 0)
        add(checks, f"{corpus}_extended_authorized", own["twin_extended"]["decision"] == "IDENTITY_AUTHORIZED")
        add(checks, f"{corpus}_extended_accuracy", own["twin_extended"]["accuracy_when_authorized"] == 1.0)
    positive = recomputed["discovery"]["overall_pass"] and recomputed["confirmation"]["overall_pass"]
    add(checks, "decision_recompute", summary["formal_decision"] == ("positive" if positive else "not_confirmed"))
    add(
        checks,
        "claim_scope",
        claims["registry_relative_abstention"]["type"] == "methodological-consolidation"
        and "a new empirical K item" in claims["registry_relative_abstention"]["not_claimed"],
    )
    gate = all(check["pass"] for check in checks)
    output = {
        "phase": p.PHASE,
        "kind": "independent_registry_abstention_audit",
        "gate_pass": gate,
        "checks_passed": sum(check["pass"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "recomputed": recomputed,
    }
    output["audit_digest"] = p.digest(output)
    if write:
        p.write_json(p.AUDIT_PATH, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    output = audit(args.write)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not output["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
