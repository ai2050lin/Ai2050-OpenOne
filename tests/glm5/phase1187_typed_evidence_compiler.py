#!/usr/bin/env python3
"""Phase1187: one-shot typed evidence compiler calibration.

This phase does not repair or re-adjudicate Phase1186.  It calibrates a
machine-readable claim contract on known-truth fixtures.  Gating predicates,
descriptive floating ratios, bounded measurements, digests, and conjunctions
remain distinct types from production through serialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1187_typed_evidence_compiler_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1187_typed_evidence_compiler"
CONTRACT_PATH = OUT_ROOT / "protocol/evidence_contract.json"
FIXTURES_PATH = OUT_ROOT / "protocol/known_truth_fixtures.jsonl"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
RESULT_ROWS_PATH = OUT_ROOT / "analysis/compiled_claims.jsonl"
MIGRATION_PATH = OUT_ROOT / "analysis/phase1186_non_gating_migration.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
PHASE1186_ROOT = ROOT / "tests/glm5/result/phase1186_reducer_safe_numerical_qualification"
PHASE = 1187
FIXTURE_SEED = 11870017


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_contract() -> dict[str, Any]:
    return {
        "schema_version": "1187.2",
        "principle": "Logical claim type is preserved across production, serialization, audit, and authorization.",
        "claim_types": {
            "universal_boolean": {
                "primitive_fields": ["agree_count:int", "eligible_count:int", "abstained:bool"],
                "gating": "required_true",
                "value": "null when abstained, otherwise agree_count == eligible_count",
                "abstain": "eligible_count == 0",
                "floating_surrogate_forbidden": True,
            },
            "descriptive_ratio": {
                "primitive_fields": [
                    "numerator:int",
                    "denominator:int",
                    "dtype:enum(float32,float64)",
                    "serialized_value:finite_float",
                ],
                "gating": "required_false",
                "value": "descriptive_only",
            },
            "bounded_float": {
                "primitive_fields": [
                    "value:finite_float",
                    "threshold:finite_float",
                    "comparator:enum(<=,>=,<,>)",
                    "dtype:enum(float32,float64)",
                ],
                "gating": "required_true",
                "value": "declared comparator without hidden tolerance",
            },
            "exact_digest": {
                "primitive_fields": ["observed:sha256", "expected:sha256"],
                "gating": "required_true",
                "value": "observed == expected",
            },
            "conjunction": {
                "primitive_fields": ["values:list[bool]"],
                "gating": "required_true",
                "value": "all(values)",
            },
        },
        "authorization": {
            "only_gating_types": [
                "universal_boolean",
                "bounded_float",
                "exact_digest",
                "conjunction",
            ],
            "descriptive_values_never_gate": True,
            "authorization_states": ["pass", "fail", "abstain", "non_gating", "invalid"],
            "only_pass_authorizes": True,
            "raw_primitives_required": True,
            "nonfinite_values_rejected": True,
            "unknown_fields_rejected": False,
        },
        "claim_exclusions": [
            "A compiler pass does not repair Phase1186.",
            "A compiler pass does not confirm K165.",
            "The compiler is evidence infrastructure, not a neural mechanism.",
            "Legacy descriptive values without raw integer primitives are quarantined, not reconstructed.",
        ],
    }


def valid_int(value: Any) -> bool:
    return type(value) is int


def valid_float(value: Any) -> bool:
    return type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(float(value))


def compile_claim(raw: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    try:
        claim_type = raw["claim_type"]
        gating = raw["gating"]
        if claim_type not in contract["claim_types"] or type(gating) is not bool:
            raise ValueError("unknown_type_or_invalid_gating")
        if claim_type == "universal_boolean":
            agree = raw["agree_count"]
            eligible = raw["eligible_count"]
            abstained = raw["abstained"]
            if not valid_int(agree) or not valid_int(eligible) or type(abstained) is not bool:
                raise ValueError("invalid_universal_primitive_type")
            if eligible < 0 or agree < 0 or agree > eligible or abstained != (eligible == 0):
                raise ValueError("invalid_universal_counts")
            if not gating:
                raise ValueError("universal_must_gate")
            value = None if abstained else agree == eligible
            decision_state = "abstain" if abstained else ("pass" if value else "fail")
            witness = {
                "agree_count": agree,
                "eligible_count": eligible,
                "abstained": abstained,
            }
        elif claim_type == "descriptive_ratio":
            numerator = raw["numerator"]
            denominator = raw["denominator"]
            dtype = raw["dtype"]
            serialized = raw["serialized_value"]
            if not valid_int(numerator) or not valid_int(denominator):
                raise ValueError("invalid_ratio_count_type")
            if denominator <= 0 or numerator < 0 or numerator > denominator:
                raise ValueError("invalid_ratio_counts")
            if dtype not in ("float32", "float64") or not valid_float(serialized):
                raise ValueError("invalid_ratio_representation")
            if gating:
                raise ValueError("descriptive_ratio_cannot_gate")
            expected = (
                float(np.float32(numerator / denominator))
                if dtype == "float32"
                else float(numerator / denominator)
            )
            if float(serialized) != expected:
                raise ValueError("ratio_dtype_roundtrip_mismatch")
            value = float(serialized)
            decision_state = "non_gating"
            witness = {
                "numerator": numerator,
                "denominator": denominator,
                "dtype": dtype,
                "serialized_value": value,
            }
        elif claim_type == "bounded_float":
            value_raw = raw["value"]
            threshold = raw["threshold"]
            comparator = raw["comparator"]
            dtype = raw["dtype"]
            if not valid_float(value_raw) or not valid_float(threshold):
                raise ValueError("nonfinite_or_invalid_bounded_float")
            if comparator not in ("<=", ">=", "<", ">") or dtype not in ("float32", "float64"):
                raise ValueError("invalid_bounded_float_contract")
            if not gating:
                raise ValueError("bounded_float_must_gate")
            left = float(value_raw)
            right = float(threshold)
            value = {
                "<=": left <= right,
                ">=": left >= right,
                "<": left < right,
                ">": left > right,
            }[comparator]
            decision_state = "pass" if value else "fail"
            witness = {
                "value": left,
                "threshold": right,
                "comparator": comparator,
                "dtype": dtype,
            }
        elif claim_type == "exact_digest":
            observed = raw["observed"]
            expected = raw["expected"]
            if not isinstance(observed, str) or not isinstance(expected, str):
                raise ValueError("invalid_digest_type")
            if len(observed) != 64 or len(expected) != 64:
                raise ValueError("invalid_digest_length")
            int(observed, 16)
            int(expected, 16)
            if not gating:
                raise ValueError("digest_must_gate")
            value = observed == expected
            decision_state = "pass" if value else "fail"
            witness = {"observed": observed, "expected": expected}
        else:
            values = raw["values"]
            if not isinstance(values, list) or not all(type(item) is bool for item in values):
                raise ValueError("invalid_conjunction_values")
            if not gating:
                raise ValueError("conjunction_must_gate")
            value = all(values)
            decision_state = "pass" if value else "fail"
            witness = {"values": values}
        return {
            "accepted": True,
            "claim_type": claim_type,
            "gating": gating,
            "value": value,
            "decision_state": decision_state,
            "authorizes": decision_state == "pass",
            "witness": witness,
            "error": None,
        }
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return {
            "accepted": False,
            "claim_type": raw.get("claim_type"),
            "gating": raw.get("gating"),
            "value": None,
            "decision_state": "invalid",
            "authorizes": False,
            "witness": None,
            "error": str(error),
        }


def build_fixtures() -> list[dict[str, Any]]:
    rng = random.Random(FIXTURE_SEED)
    rows: list[dict[str, Any]] = []

    def add(raw: dict[str, Any], accepted: bool, value: Any = None) -> None:
        if not accepted:
            decision_state = "invalid"
        elif not raw["gating"]:
            decision_state = "non_gating"
        elif value is None:
            decision_state = "abstain"
        else:
            decision_state = "pass" if value else "fail"
        rows.append(
            {
                "case_id": f"case_{len(rows):05d}",
                "raw": raw,
                "expected_accepted": accepted,
                "expected_value": value if accepted else None,
                "expected_decision_state": decision_state,
                "expected_authorizes": decision_state == "pass",
            }
        )

    for length in (0, 1, 3, 7, 61, 227, 3721, 8191):
        add(
            {
                "claim_type": "universal_boolean",
                "gating": True,
                "agree_count": length,
                "eligible_count": length,
                "abstained": length == 0,
            },
            True,
            None if length == 0 else True,
        )
        if length:
            add(
                {
                    "claim_type": "universal_boolean",
                    "gating": True,
                    "agree_count": length - 1,
                    "eligible_count": length,
                    "abstained": False,
                },
                True,
                False,
            )
    for _ in range(256):
        denominator = rng.randint(1, 10000)
        numerator = rng.randint(0, denominator)
        all_equal = rng.choice((True, False))
        agree = denominator if all_equal else max(0, denominator - rng.randint(1, denominator))
        add(
            {
                "claim_type": "universal_boolean",
                "gating": True,
                "agree_count": agree,
                "eligible_count": denominator,
                "abstained": False,
            },
            True,
            agree == denominator,
        )
        for dtype in ("float32", "float64"):
            value = (
                float(np.float32(numerator / denominator))
                if dtype == "float32"
                else float(numerator / denominator)
            )
            add(
                {
                    "claim_type": "descriptive_ratio",
                    "gating": False,
                    "numerator": numerator,
                    "denominator": denominator,
                    "dtype": dtype,
                    "serialized_value": value,
                },
                True,
                value,
            )
    comparators = ("<=", ">=", "<", ">")
    for index in range(256):
        threshold = rng.uniform(-100.0, 100.0)
        offset = (index % 5 - 2) * max(abs(threshold), 1.0) * 1e-7
        value = threshold + offset
        comparator = comparators[index % len(comparators)]
        expected = {
            "<=": value <= threshold,
            ">=": value >= threshold,
            "<": value < threshold,
            ">": value > threshold,
        }[comparator]
        add(
            {
                "claim_type": "bounded_float",
                "gating": True,
                "value": value,
                "threshold": threshold,
                "comparator": comparator,
                "dtype": "float64",
            },
            True,
            expected,
        )
    for index in range(128):
        expected_digest = hashlib.sha256(f"expected:{index}".encode()).hexdigest()
        observed = expected_digest if index % 2 == 0 else hashlib.sha256(f"observed:{index}".encode()).hexdigest()
        add(
            {
                "claim_type": "exact_digest",
                "gating": True,
                "observed": observed,
                "expected": expected_digest,
            },
            True,
            observed == expected_digest,
        )
    for index in range(128):
        values = [rng.choice((True, False)) for _ in range(index % 9)]
        add(
            {"claim_type": "conjunction", "gating": True, "values": values},
            True,
            all(values),
        )

    invalid = [
        {"claim_type": "universal_boolean", "gating": True, "agree_count": 1.0, "eligible_count": 1, "abstained": False},
        {"claim_type": "universal_boolean", "gating": True, "agree_count": True, "eligible_count": 1, "abstained": False},
        {"claim_type": "universal_boolean", "gating": True, "agree_count": 2, "eligible_count": 1, "abstained": False},
        {"claim_type": "universal_boolean", "gating": True, "agree_count": 0, "eligible_count": 0, "abstained": False},
        {"claim_type": "universal_boolean", "gating": False, "agree_count": 1, "eligible_count": 1, "abstained": False},
        {"claim_type": "descriptive_ratio", "gating": True, "numerator": 1, "denominator": 3, "dtype": "float32", "serialized_value": float(np.float32(1 / 3))},
        {"claim_type": "descriptive_ratio", "gating": False, "numerator": 1, "denominator": 0, "dtype": "float32", "serialized_value": 0.0},
        {"claim_type": "descriptive_ratio", "gating": False, "numerator": 1, "denominator": 3, "dtype": "float32", "serialized_value": float(1 / 3)},
        {"claim_type": "descriptive_ratio", "gating": False, "denominator": 3, "dtype": "float32", "serialized_value": float(np.float32(1 / 3))},
        {"claim_type": "bounded_float", "gating": True, "value": "nan", "threshold": 1.0, "comparator": "<=", "dtype": "float64"},
        {"claim_type": "bounded_float", "gating": False, "value": 0.0, "threshold": 1.0, "comparator": "<=", "dtype": "float64"},
        {"claim_type": "bounded_float", "gating": True, "value": 0.0, "threshold": 1.0, "comparator": "~=", "dtype": "float64"},
        {"claim_type": "exact_digest", "gating": True, "observed": "abc", "expected": "def"},
        {"claim_type": "exact_digest", "gating": False, "observed": "0" * 64, "expected": "0" * 64},
        {"claim_type": "conjunction", "gating": True, "values": [True, 1]},
        {"claim_type": "conjunction", "gating": False, "values": [True]},
        {"claim_type": "unknown", "gating": True},
        {"gating": True},
    ]
    for raw in invalid:
        add(raw, False)
    return rows


def preregister() -> None:
    if PROTOCOL_PATH.exists():
        raise RuntimeError("Phase1187 already preregistered")
    if not AUDIT_SCRIPT.exists():
        raise RuntimeError("audit script must exist before registration")
    phase1186_final = read_json(PHASE1186_ROOT / "analysis/final.json")
    phase1186_audit = read_json(PHASE1186_ROOT / "audit/independent_audit.json")
    if phase1186_audit["audit_pass"] or phase1186_audit["phase1187_authorized_after_audit"]:
        raise RuntimeError("Phase1186 frozen failure boundary changed")
    contract = build_contract()
    fixtures = build_fixtures()
    contract["contract_digest"] = digest(contract)
    write_json(CONTRACT_PATH, contract)
    write_jsonl(FIXTURES_PATH, fixtures)
    protocol = {
        "phase": PHASE,
        "registered_at_utc": utc_now(),
        "scientific_object": "One-shot calibration of a typed evidence compiler; no neural mechanism is tested.",
        "phase1186_boundary": {
            "final_digest": phase1186_final["final_digest"],
            "audit_digest": phase1186_audit["audit_digest"],
            "audit_pass": False,
            "phase1187_old_authorization": False,
            "verdict_unchanged": True,
        },
        "fixture_seed": FIXTURE_SEED,
        "fixture_count": len(fixtures),
        "valid_fixture_count": sum(row["expected_accepted"] for row in fixtures),
        "invalid_fixture_count": sum(not row["expected_accepted"] for row in fixtures),
        "contract_sha256": file_sha256(CONTRACT_PATH),
        "fixtures_sha256": file_sha256(FIXTURES_PATH),
        "decision": {
            "main": "all known-truth valid fixtures compile exactly; all invalid fixtures are rejected; Phase1186 migration is non-gating",
            "audit": "independent implementation reproduces every fixture and serialized main result",
            "pass_action": "freeze qualification development and authorize one new terminal three-evidence registry",
            "failure_action": "stop; do not patch individual evidence fields",
        },
        "scripts": {
            "runner": file_sha256(SCRIPT),
            "audit": file_sha256(AUDIT_SCRIPT),
            "phase1186_final": file_sha256(PHASE1186_ROOT / "analysis/final.json"),
            "phase1186_audit": file_sha256(PHASE1186_ROOT / "audit/independent_audit.json"),
            "phase1186_gauge_rows": file_sha256(PHASE1186_ROOT / "analysis/gauge_rows.jsonl"),
            "phase1186_positive_rows": file_sha256(PHASE1186_ROOT / "analysis/positive_control_rows.jsonl"),
        },
        "claim_exclusions": contract["claim_exclusions"],
    }
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"registered": str(PROTOCOL_PATH), "digest": protocol["protocol_digest"]}))


def validate_protocol() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = read_json(PROTOCOL_PATH)
    copy = dict(protocol)
    stored = copy.pop("protocol_digest")
    if digest(copy) != stored:
        raise RuntimeError("protocol digest mismatch")
    paths = {
        "runner": SCRIPT,
        "audit": AUDIT_SCRIPT,
        "phase1186_final": PHASE1186_ROOT / "analysis/final.json",
        "phase1186_audit": PHASE1186_ROOT / "audit/independent_audit.json",
        "phase1186_gauge_rows": PHASE1186_ROOT / "analysis/gauge_rows.jsonl",
        "phase1186_positive_rows": PHASE1186_ROOT / "analysis/positive_control_rows.jsonl",
    }
    for name, path in paths.items():
        if file_sha256(path) != protocol["scripts"][name]:
            raise RuntimeError(f"frozen source changed: {name}")
    if file_sha256(CONTRACT_PATH) != protocol["contract_sha256"]:
        raise RuntimeError("contract changed")
    if file_sha256(FIXTURES_PATH) != protocol["fixtures_sha256"]:
        raise RuntimeError("fixtures changed")
    contract = read_json(CONTRACT_PATH)
    contract_copy = dict(contract)
    contract_digest = contract_copy.pop("contract_digest")
    if digest(contract_copy) != contract_digest:
        raise RuntimeError("contract digest mismatch")
    return protocol, contract


def migrate_phase1186(contract: dict[str, Any]) -> dict[str, Any]:
    gauge_rows = read_jsonl(PHASE1186_ROOT / "analysis/gauge_rows.jsonl")
    positive_rows = read_jsonl(PHASE1186_ROOT / "analysis/positive_control_rows.jsonl")
    claims: list[dict[str, Any]] = []
    for row in gauge_rows:
        for name in ("decision", "margin_sign"):
            source = row["exact_decision"][name]
            claims.append(
                compile_claim(
                    {
                        "claim_type": "universal_boolean",
                        "gating": True,
                        "agree_count": source["agree_count"],
                        "eligible_count": source["eligible_count"],
                        "abstained": source["abstained"],
                    },
                    contract,
                )
            )
    legacy_descriptive = [
        {
            "case": row["case"],
            "serialized_value": row["decision_agreement"],
            "declared_dtype": "float32",
            "status": "quarantined_missing_raw_numerator",
        }
        for row in positive_rows
    ]
    frozen_audit = read_json(PHASE1186_ROOT / "audit/independent_audit.json")
    return {
        "status": "non_gating_migration_fixture_only",
        "source_field_count": len(claims) + len(legacy_descriptive),
        "typed_claim_count": len(claims),
        "accepted_count": sum(claim["accepted"] for claim in claims),
        "gating_claim_count": sum(claim["gating"] is True for claim in claims),
        "descriptive_claim_count": 0,
        "legacy_untyped_descriptive_count": len(legacy_descriptive),
        "legacy_untyped_descriptive_digest": digest(legacy_descriptive),
        "descriptive_gate_count": sum(
            claim["claim_type"] == "descriptive_ratio" and claim["gating"] is True
            for claim in claims
        ),
        "decision_state_counts": {
            state: sum(claim["decision_state"] == state for claim in claims)
            for state in ("pass", "fail", "abstain", "non_gating", "invalid")
        },
        "authorization_count": sum(claim["authorizes"] for claim in claims),
        "phase1186_audit_pass_unchanged": frozen_audit["audit_pass"],
        "phase1186_authorization_unchanged": frozen_audit["phase1187_authorized_after_audit"],
    }


def run() -> None:
    protocol, contract = validate_protocol()
    if FINAL_PATH.exists():
        raise RuntimeError("Phase1187 already finalized")
    fixtures = read_jsonl(FIXTURES_PATH)
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        compiled = compile_claim(fixture["raw"], contract)
        expected_match = bool(
            compiled["accepted"] == fixture["expected_accepted"]
            and (
                not fixture["expected_accepted"]
                or (
                    compiled["value"] == fixture["expected_value"]
                    and compiled["decision_state"] == fixture["expected_decision_state"]
                    and compiled["authorizes"] == fixture["expected_authorizes"]
                )
            )
        )
        rows.append(
            {
                "case_id": fixture["case_id"],
                "expected_accepted": fixture["expected_accepted"],
                "expected_value": fixture["expected_value"],
                "compiled": compiled,
                "expected_match": expected_match,
            }
        )
    write_jsonl(RESULT_ROWS_PATH, rows)
    migration = migrate_phase1186(contract)
    migration["migration_digest"] = digest(migration)
    write_json(MIGRATION_PATH, migration)
    valid_rows = [row for row in rows if row["expected_accepted"]]
    invalid_rows = [row for row in rows if not row["expected_accepted"]]
    main_pass = bool(
        all(row["expected_match"] for row in rows)
        and all(row["compiled"]["accepted"] for row in valid_rows)
        and all(not row["compiled"]["accepted"] for row in invalid_rows)
        and migration["accepted_count"] == migration["typed_claim_count"] == 512
        and migration["legacy_untyped_descriptive_count"] == 64
        and migration["descriptive_gate_count"] == 0
        and migration["phase1186_audit_pass_unchanged"] is False
        and migration["phase1186_authorization_unchanged"] is False
    )
    final = {
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "protocol_digest": protocol["protocol_digest"],
        "contract_digest": contract["contract_digest"],
        "fixture_count": len(rows),
        "valid_fixture_count": len(valid_rows),
        "invalid_fixture_count": len(invalid_rows),
        "exact_fixture_match_count": sum(row["expected_match"] for row in rows),
        "invalid_rejection_count": sum(not row["compiled"]["accepted"] for row in invalid_rows),
        "decision_state_counts": {
            state: sum(row["compiled"]["decision_state"] == state for row in rows)
            for state in ("pass", "fail", "abstain", "non_gating", "invalid")
        },
        "authorization_count": sum(row["compiled"]["authorizes"] for row in rows),
        "phase1186_migration": migration,
        "main_pass": main_pass,
        "qualification_development_status": "awaiting_independent_audit",
        "phase1188_authorized_before_audit": False,
        "claim_scope": "typed_evidence_infrastructure_only",
        "artifacts": {
            "compiled_claims_sha256": file_sha256(RESULT_ROWS_PATH),
            "migration_sha256": file_sha256(MIGRATION_PATH),
        },
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json(final))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run"))
    args = parser.parse_args()
    if args.command == "preregister":
        preregister()
    else:
        run()


if __name__ == "__main__":
    main()
