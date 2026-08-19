#!/usr/bin/env python3
"""Phase1249: zero-GPU source-observation identifiability audit.

This analysis does not refit the Phase1248 camera and cannot upgrade its
evidence.  It asks a narrower question: can the frozen source-only observation
be a single-valued predictor when later codebook and interface tokens differ?
Exact causal-prefix collisions provide a deterministic answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PHASE = 1249
SCRIPT = Path(__file__).resolve()
AUDITOR = ROOT / "tests/glm5/phase1249_c003_source_observation_identifiability_audit.py"
UPSTREAM_ROOT = ROOT / "tests/glm5/result/phase1248_c002_qwen_self_response_atlas"
MATERIAL_PATH = UPSTREAM_ROOT / "material/frozen_worlds.jsonl"
ARRAY_PATH = UPSTREAM_ROOT / "raw/response_arrays.npz"
ATLAS_PATH = UPSTREAM_ROOT / "analysis/model_self_response_atlas.json"
SOURCE_1247 = ROOT / "tests/glm5/phase1247_c002_hidden_response_imaging_camera.py"
SOURCE_1248 = ROOT / "tests/glm5/phase1248_c002_qwen_self_response_atlas.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1249_c003_source_observation_identifiability"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
PREAUDIT_PATH = OUT_ROOT / "audit/independent_preaudit.json"
ANALYSIS_PATH = OUT_ROOT / "analysis/observation_sufficiency.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
FINAL_AUDIT_PATH = OUT_ROOT / "audit/independent_final_audit.json"

DONORS = ("target", "null")
ALPHAS = (0.25, 0.5, 0.75, 1.0)
EVENT_ID = "residual_source_d06"
EVENT_INDEX = 0
DONOR_INDEX = 0
ALPHA = 1.0
ALPHA_INDEX = ALPHAS.index(ALPHA)
FEATURE_TOLERANCE = 1.0e-5
RESPONSE_SEPARATION_MIN = 1.0


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    output = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            output.update(chunk)
    return output.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in MATERIAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def protocol_payload() -> dict[str, Any]:
    payload = {
        "phase": PHASE,
        "schema_version": "phase1249.c003.source_identifiability.protocol.v1",
        "created_at_utc": utc_now(),
        "claim_type": "posthoc_deterministic_observation_identifiability_audit",
        "upstream_verdict_preserved": "bounded_external_validity_failure",
        "question": "Does the frozen source-only observation remain single-valued when later codebook mapping and output interface change?",
        "fixed_object": {
            "event_id": EVENT_ID,
            "event_index": EVENT_INDEX,
            "donor": "target",
            "alpha": ALPHA,
            "feature": "alpha * (target_projected - receiver_projected)",
            "response": "centered eight-candidate actual patch response",
        },
        "exact_collision_group": ["world_id", "representation", "receiver_state"],
        "within_group_axes": ["mapping", "interface"],
        "thresholds": {
            "feature_max_pair_distance": FEATURE_TOLERANCE,
            "response_pair_distance_min": RESPONSE_SEPARATION_MIN,
        },
        "tests": [
            "alpha_implementation_audit",
            "causal_prefix_identity",
            "exact_feature_collision",
            "mapping_response_separation",
            "interface_response_separation",
            "oracle_collision_lower_bound",
            "world_cluster_accounting",
        ],
        "interpretation": {
            "can_close": "a context-free source-only response map f(O_source)",
            "cannot_close": [
                "a context-conditioned map g(O_source,C)",
                "a multi-event observation bundle",
                "future-response quotient theory",
                "natural-language semantic mechanism",
            ],
            "authorization_on_exact_collision": "known-truth one-hop/two-hop multi-event camera calibration only",
        },
        "source_hashes": {
            "main": file_sha256(SCRIPT),
            "auditor": file_sha256(AUDITOR),
            "phase1247": file_sha256(SOURCE_1247),
            "phase1248": file_sha256(SOURCE_1248),
            "material": file_sha256(MATERIAL_PATH),
            "arrays": file_sha256(ARRAY_PATH),
            "atlas": file_sha256(ATLAS_PATH),
        },
        "hard_stops": [
            "No model or GPU may be loaded.",
            "No event, threshold, projection or sample may be reselected.",
            "No fitted model from this posthoc audit upgrades Phase1248.",
            "Failure or collision authorizes only a new known-truth calibration contract.",
        ],
    }
    payload["protocol_digest"] = digest({key: value for key, value in payload.items() if key != "created_at_utc"})
    return payload


def prepare() -> None:
    if ANALYSIS_PATH.exists() or FINAL_PATH.exists():
        raise RuntimeError("formal analysis already exists")
    write_json(PROTOCOL_PATH, protocol_payload())
    print(canonical_json({"status": "prepared", "protocol": str(PROTOCOL_PATH)}))


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    expected = protocol_payload()
    if protocol["source_hashes"] != expected["source_hashes"]:
        raise RuntimeError("source or upstream artifact changed after freeze")
    if protocol["protocol_digest"] != expected["protocol_digest"]:
        raise RuntimeError("protocol digest mismatch")
    preaudit = read_json(PREAUDIT_PATH)
    if not preaudit.get("all_checks_passed"):
        raise RuntimeError("independent preaudit did not pass")
    return protocol


def pair_distance(values: np.ndarray, left: int, right: int) -> float:
    return float(np.linalg.norm(values[left] - values[right]))


def group_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["world_id"], row["representation"], int(row["receiver_state"])


def summarize_groups(rows: list[dict[str, Any]], features: np.ndarray, responses: np.ndarray) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], list[int]] = {}
    for index, row in enumerate(rows):
        if row["partition"] == "confirmation":
            groups.setdefault(group_key(row), []).append(index)
    records: list[dict[str, Any]] = []
    oracle_prediction = np.zeros_like(responses)
    confirmation_indices: list[int] = []
    for key, indices in sorted(groups.items()):
        if len(indices) != 4:
            raise RuntimeError(f"incomplete collision group {key}: {len(indices)}")
        feature_distances = [pair_distance(features, a, b) for a, b in combinations(indices, 2)]
        response_distances = [pair_distance(responses, a, b) for a, b in combinations(indices, 2)]
        mapping_pairs = []
        interface_pairs = []
        for a, b in combinations(indices, 2):
            ra, rb = rows[a], rows[b]
            if ra["interface"] == rb["interface"] and ra["mapping"] != rb["mapping"]:
                mapping_pairs.append(pair_distance(responses, a, b))
            if ra["mapping"] == rb["mapping"] and ra["interface"] != rb["interface"]:
                interface_pairs.append(pair_distance(responses, a, b))
        mean_response = responses[indices].mean(axis=0)
        oracle_prediction[indices] = mean_response
        confirmation_indices.extend(indices)
        records.append({
            "world_id": key[0],
            "representation": key[1],
            "receiver_state": key[2],
            "count": len(indices),
            "feature_max_pair_distance": max(feature_distances),
            "response_max_pair_distance": max(response_distances),
            "mapping_response_distance_mean": float(np.mean(mapping_pairs)),
            "interface_response_distance_mean": float(np.mean(interface_pairs)),
            "exact_collision": max(feature_distances) <= FEATURE_TOLERANCE,
            "separated_collision": max(feature_distances) <= FEATURE_TOLERANCE and max(response_distances) >= RESPONSE_SEPARATION_MIN,
        })
    selected = np.asarray(sorted(confirmation_indices), dtype=np.int64)
    residual = responses[selected] - oracle_prediction[selected]
    irreducible_rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    response_rms = float(np.sqrt(np.mean(np.sum(responses[selected] * responses[selected], axis=1))))
    by_representation: dict[str, Any] = {}
    for representation in ("direct", "code"):
        subset = [record for record in records if record["representation"] == representation]
        by_representation[representation] = {
            "groups": len(subset),
            "exact_collision_fraction": float(np.mean([record["exact_collision"] for record in subset])),
            "separated_collision_fraction": float(np.mean([record["separated_collision"] for record in subset])),
            "feature_max_pair_distance_max": float(max(record["feature_max_pair_distance"] for record in subset)),
            "response_max_pair_distance_mean": float(np.mean([record["response_max_pair_distance"] for record in subset])),
            "mapping_response_distance_mean": float(np.mean([record["mapping_response_distance_mean"] for record in subset])),
            "interface_response_distance_mean": float(np.mean([record["interface_response_distance_mean"] for record in subset])),
        }
    return {
        "group_count": len(records),
        "rows": len(selected),
        "by_representation": by_representation,
        "all_groups_exact_collision_fraction": float(np.mean([record["exact_collision"] for record in records])),
        "all_groups_separated_collision_fraction": float(np.mean([record["separated_collision"] for record in records])),
        "oracle_source_only_irreducible_rms": irreducible_rms,
        "response_rms": response_rms,
        "oracle_irreducible_fraction": irreducible_rms / max(response_rms, 1.0e-12),
        "records": records,
    }


def causal_prefix_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row["partition"] == "confirmation":
            groups.setdefault(group_key(row), []).append(row)
    checks = []
    for key, values in sorted(groups.items()):
        for variant in ("receiver", "target"):
            prefixes = []
            for row in values:
                prompt = row["variants"][variant]["prompt"]
                end = int(row["variants"][variant]["source_span"][1])
                prefixes.append(prompt[:end])
            checks.append({"key": list(key), "variant": variant, "identical": len(set(prefixes)) == 1})
    return {
        "checks": len(checks),
        "all_identical": all(item["identical"] for item in checks),
        "failure_count": sum(not item["identical"] for item in checks),
    }


def run() -> None:
    protocol = verify_protocol()
    if ANALYSIS_PATH.exists() or FINAL_PATH.exists():
        raise RuntimeError("one-shot analysis already exists")
    rows = read_rows()
    with np.load(ARRAY_PATH) as arrays:
        features = ALPHA * (arrays["target_projected"][:, EVENT_INDEX] - arrays["receiver_projected"][:, EVENT_INDEX])
        responses = arrays["responses"][:, EVENT_INDEX, DONOR_INDEX, ALPHA_INDEX]
    source_1247 = SOURCE_1247.read_text(encoding="utf-8")
    source_1248 = SOURCE_1248.read_text(encoding="utf-8")
    alpha_audit = {
        "phase1247_feature_scales_alpha": "train_x.append(float(alpha) * delta[discovery_index])" in source_1247 and "features = float(alpha) * delta[indices]" in source_1247,
        "phase1248_feature_scales_alpha": "return float(alpha) * (arrays[donor_key][indices, event] - arrays[\"receiver_projected\"][indices, event])" in source_1248,
        "formula": "x(alpha)=alpha*P*(h_donor-h_receiver)",
    }
    prefix = causal_prefix_audit(rows)
    collisions = summarize_groups(rows, features, responses)
    direct = collisions["by_representation"]["direct"]
    code = collisions["by_representation"]["code"]
    gates = {
        "G-ALPHA-IMPLEMENTATION": all(alpha_audit[key] for key in ("phase1247_feature_scales_alpha", "phase1248_feature_scales_alpha")),
        "G-CAUSAL-PREFIX": prefix["all_identical"],
        "G-EXACT-SOURCE-COLLISION": collisions["all_groups_exact_collision_fraction"] == 1.0,
        "G-CODE-RESPONSE-SEPARATION": code["mapping_response_distance_mean"] >= RESPONSE_SEPARATION_MIN,
        "G-DIRECT-CODE-DIFFERENCE": code["mapping_response_distance_mean"] > direct["mapping_response_distance_mean"],
    }
    verdict = "source_only_observation_nonidentifiable_across_later_context" if all(gates.values()) else "collision_hypothesis_not_confirmed"
    result = {
        "phase": PHASE,
        "schema_version": "phase1249.c003.source_identifiability.analysis.v1",
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "alpha_audit": alpha_audit,
        "causal_prefix": prefix,
        "collisions": collisions,
        "gates": gates,
        "verdict": verdict,
        "authorization": {
            "known_truth_multi_event_calibration": all(gates.values()),
            "qwen_rerun": False,
            "semantic_mechanism_claim": False,
            "phase1248_evidence_upgrade": False,
        },
        "claim_boundary": "The result closes context-free source-only prediction. It does not prove that a multi-event bundle is sufficient or that code processing is exactly two neural hops.",
    }
    result["analysis_digest"] = digest({key: value for key, value in result.items() if key != "created_at_utc"})
    write_json(ANALYSIS_PATH, result)
    final = {
        "phase": PHASE,
        "verdict": verdict,
        "gates": gates,
        "known_truth_multi_event_calibration_authorized": all(gates.values()),
        "qwen_rerun_authorized": False,
        "analysis_digest": result["analysis_digest"],
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": "complete", "verdict": verdict, "gates": gates}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
