"""Compile Phase1200 separability evidence into explicit claim abstention.

Phase1200 showed that the core-registry twins have identical observable
centroids, but its nearest-centroid helper still emits a label on ties. This
script adds a registry-level identifiability gate: identity predictions are
authorized only when every support centroid pair is separated by more than a
frozen tolerance. Otherwise the whole identity claim is marked
``UNIDENTIFIABLE``. No new model or synthetic outcome is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1200_conditional_causal_trace_mapper as p1200  # noqa: E402


PHASE = 1201
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1201_registry_identifiability_abstention_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1201_registry_identifiability_abstention"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
SUMMARY_PATH = OUT_ROOT / "analysis/summary.json"
CLAIMS_PATH = OUT_ROOT / "analysis/typed_claims.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"

SEPARATION_TOLERANCE = 1e-8
IDENTIFIABLE_ACCURACY_MIN = 0.99
ABSTENTION_TOKEN = "UNIDENTIFIABLE"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return np.zeros_like(vector) if norm <= 1e-15 else vector / norm


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(unit(left), unit(right)))


def source_hashes() -> dict[str, str]:
    return {"phase1201": file_sha256(SCRIPT), "phase1201_audit": file_sha256(AUDIT_SCRIPT)}


def upstream_hashes() -> dict[str, str]:
    return {
        "phase1200_final": file_sha256(p1200.FINAL_PATH),
        "phase1200_discovery": file_sha256(p1200.FORMAL_DISCOVERY_ROWS),
        "phase1200_confirmation": file_sha256(p1200.FORMAL_CONFIRMATION_ROWS),
        "phase1200_audit": file_sha256(p1200.AUDIT_PATH),
    }


def centroids(rows: list[dict[str, Any]], field: str, label_field: str) -> dict[int, np.ndarray]:
    labels = sorted({int(row[label_field]) for row in rows})
    return {
        label: np.mean(
            np.asarray([row[field] for row in rows if int(row[label_field]) == label], dtype=np.float64),
            axis=0,
        )
        for label in labels
    }


def minimum_centroid_distance(centers: dict[int, np.ndarray]) -> float:
    return min(
        float(np.linalg.norm(centers[left] - centers[right]))
        for left, right in itertools.combinations(sorted(centers), 2)
    )


def compile_task(
    rows: list[dict[str, Any]],
    feature_field: str,
    label_field: str,
    task_name: str,
) -> dict[str, Any]:
    support = [row for row in rows if row["role"] == "support"]
    evaluation = [row for row in rows if row["role"] == "evaluation"]
    centers = centroids(support, feature_field, label_field)
    minimum_distance = minimum_centroid_distance(centers)
    identifiable = minimum_distance > SEPARATION_TOLERANCE
    predictions: list[int | str] = []
    if identifiable:
        labels = sorted(centers)
        for row in evaluation:
            feature = np.asarray(row[feature_field], dtype=np.float64)
            scores = [cosine(feature, centers[label]) for label in labels]
            predictions.append(labels[int(np.argmax(scores))])
    else:
        predictions = [ABSTENTION_TOKEN] * len(evaluation)
    identity_claim_count = sum(prediction != ABSTENTION_TOKEN for prediction in predictions)
    correct = sum(
        prediction == int(row[label_field]) for prediction, row in zip(predictions, evaluation)
    )
    accuracy = correct / max(len(evaluation), 1) if identifiable else None
    return {
        "task": task_name,
        "support_count": len(support),
        "evaluation_count": len(evaluation),
        "class_count": len(centers),
        "minimum_centroid_distance": minimum_distance,
        "separation_tolerance": SEPARATION_TOLERANCE,
        "identifiable": identifiable,
        "decision": "IDENTITY_AUTHORIZED" if identifiable else ABSTENTION_TOKEN,
        "identity_claim_count": identity_claim_count,
        "abstention_count": len(evaluation) - identity_claim_count,
        "accuracy_when_authorized": accuracy,
        "prediction_digest": digest(predictions),
    }


def compile_corpus(payload: dict[str, Any]) -> dict[str, Any]:
    family = compile_task(payload["family_rows"], "signature", "family", "family_core_registry")
    twin_core = compile_task(payload["twin_rows"], "core_signature", "label", "twin_core_registry")
    twin_extended = compile_task(
        payload["twin_rows"], "extended_signature", "label", "twin_extended_registry"
    )
    gates = {
        "family_identity_authorized": bool(
            family["identifiable"]
            and family["accuracy_when_authorized"] is not None
            and family["accuracy_when_authorized"] >= IDENTIFIABLE_ACCURACY_MIN
        ),
        "core_twin_identity_denied": bool(
            not twin_core["identifiable"]
            and twin_core["identity_claim_count"] == 0
            and twin_core["abstention_count"] == twin_core["evaluation_count"]
        ),
        "extended_twin_identity_authorized": bool(
            twin_extended["identifiable"]
            and twin_extended["accuracy_when_authorized"] is not None
            and twin_extended["accuracy_when_authorized"] >= IDENTIFIABLE_ACCURACY_MIN
        ),
        "registry_extension_monotonicity": bool(
            twin_extended["minimum_centroid_distance"] > twin_core["minimum_centroid_distance"]
        ),
    }
    return {
        "corpus": payload["corpus"],
        "family": family,
        "twin_core": twin_core,
        "twin_extended": twin_extended,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def preregister() -> None:
    if PROTOCOL_PATH.exists() or SUMMARY_PATH.exists():
        raise RuntimeError("Phase1201 protocol or outcomes already exist")
    phase1200_final = read_json(p1200.FINAL_PATH)
    if not phase1200_final["authorized_next"]["theory_and_measurement_consolidation"]:
        raise RuntimeError("Phase1200 did not authorize measurement consolidation")
    protocol = {
        "phase": PHASE,
        "schema_version": "phase1201.registry_abstention.protocol.v1",
        "created_at": utc_now(),
        "purpose": "convert registry-relative non-separability into explicit identity-claim abstention",
        "separation_tolerance": SEPARATION_TOLERANCE,
        "identifiable_accuracy_min": IDENTIFIABLE_ACCURACY_MIN,
        "abstention_token": ABSTENTION_TOKEN,
        "source_hashes": source_hashes(),
        "upstream_hashes": upstream_hashes(),
        "scope": {
            "new_data": False,
            "new_model_run": False,
            "new_k_item": False,
            "natural_language_claim": False,
        },
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"protocol_digest": protocol["protocol_digest"]}))


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    candidate = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    if digest(candidate) != protocol["protocol_digest"]:
        raise RuntimeError("protocol digest mismatch")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("source changed after preregistration")
    if protocol["upstream_hashes"] != upstream_hashes():
        raise RuntimeError("Phase1200 upstream changed after preregistration")
    return protocol


def run() -> None:
    if SUMMARY_PATH.exists() or CLAIMS_PATH.exists():
        raise RuntimeError("Phase1201 outcomes already exist")
    protocol = verify_protocol()
    discovery = compile_corpus(read_json(p1200.FORMAL_DISCOVERY_ROWS))
    confirmation = compile_corpus(read_json(p1200.FORMAL_CONFIRMATION_ROWS))
    positive = discovery["overall_pass"] and confirmation["overall_pass"]
    summary = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": "explicit_registry_abstention_confirmed" if positive else "explicit_registry_abstention_not_confirmed",
        "protocol_digest": protocol["protocol_digest"],
        "discovery": discovery,
        "confirmation": confirmation,
        "formal_decision": "positive" if positive else "not_confirmed",
    }
    claims = {
        "registry_relative_abstention": {
            "type": "methodological-consolidation",
            "confirmed_if_positive": "Phase1200 exact non-separability can be compiled into an explicit registry-level UNIDENTIFIABLE decision before any identity prediction.",
            "not_claimed": [
                "sample-level calibrated uncertainty",
                "near-collision abstention beyond the frozen exact tolerance",
                "natural-language mechanism identification",
                "a new empirical K item",
            ],
        }
    }
    write_json(SUMMARY_PATH, summary)
    write_json(CLAIMS_PATH, claims)
    print(canonical_json({"status": summary["status"], "discovery": discovery, "confirmation": confirmation}))


def finalize() -> None:
    protocol = verify_protocol()
    summary = read_json(SUMMARY_PATH)
    claims = read_json(CLAIMS_PATH)
    audit = read_json(AUDIT_PATH)
    if not audit.get("gate_pass", False):
        raise RuntimeError("independent audit did not pass")
    final = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": summary["status"],
        "evidence": claims,
        "protocol_digest": protocol["protocol_digest"],
        "audit_digest": audit["audit_digest"],
        "summary": summary,
        "authorized_next": {
            "theory_and_measurement_consolidation": True,
            "natural_language_trace_scan": False,
            "new_mechanism_algebra": False,
        },
        "scope": "derived measurement semantics only; K1-K183 unchanged",
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": final["status"], "authorized_next": final["authorized_next"], "final_digest": final["final_digest"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run", "finalize"))
    command = parser.parse_args().command
    {"preregister": preregister, "run": run, "finalize": finalize}[command]()


if __name__ == "__main__":
    main()
