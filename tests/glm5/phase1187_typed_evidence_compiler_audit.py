#!/usr/bin/env python3
"""Independent implementation audit for the Phase1187 evidence compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
OUT_ROOT = ROOT / "tests/glm5/result/phase1187_typed_evidence_compiler"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
CONTRACT_PATH = OUT_ROOT / "protocol/evidence_contract.json"
FIXTURES_PATH = OUT_ROOT / "protocol/known_truth_fixtures.jsonl"
ROWS_PATH = OUT_ROOT / "analysis/compiled_claims.jsonl"
MIGRATION_PATH = OUT_ROOT / "analysis/phase1186_non_gating_migration.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"
RUNNER = ROOT / "tests/glm5/phase1187_typed_evidence_compiler.py"
PHASE1186_ROOT = ROOT / "tests/glm5/result/phase1186_reducer_safe_numerical_qualification"


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def exact_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def independent_compile(raw: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    claim_type = raw.get("claim_type")
    gating = raw.get("gating")
    rejected = {
        "accepted": False,
        "claim_type": claim_type,
        "gating": gating,
        "value": None,
        "decision_state": "invalid",
        "authorizes": False,
        "witness": None,
    }
    if claim_type not in contract["claim_types"] or not isinstance(gating, bool):
        return rejected
    try:
        if claim_type == "universal_boolean":
            agree, eligible, abstained = raw["agree_count"], raw["eligible_count"], raw["abstained"]
            if not exact_integer(agree) or not exact_integer(eligible) or not isinstance(abstained, bool):
                return rejected
            if min(agree, eligible) < 0 or agree > eligible or abstained != (eligible == 0) or not gating:
                return rejected
            value = None if abstained else agree == eligible
            decision_state = "abstain" if abstained else ("pass" if value else "fail")
            witness = {"agree_count": agree, "eligible_count": eligible, "abstained": abstained}
        elif claim_type == "descriptive_ratio":
            numerator, denominator = raw["numerator"], raw["denominator"]
            dtype, serialized = raw["dtype"], raw["serialized_value"]
            if not exact_integer(numerator) or not exact_integer(denominator):
                return rejected
            if denominator <= 0 or numerator < 0 or numerator > denominator or gating:
                return rejected
            if dtype not in ("float32", "float64") or not finite_number(serialized):
                return rejected
            rebuilt = float(np.float32(numerator / denominator)) if dtype == "float32" else numerator / denominator
            if float(serialized) != float(rebuilt):
                return rejected
            value = float(serialized)
            decision_state = "non_gating"
            witness = {
                "numerator": numerator,
                "denominator": denominator,
                "dtype": dtype,
                "serialized_value": value,
            }
        elif claim_type == "bounded_float":
            measured, threshold = raw["value"], raw["threshold"]
            comparator, dtype = raw["comparator"], raw["dtype"]
            if not finite_number(measured) or not finite_number(threshold) or not gating:
                return rejected
            if comparator not in ("<=", ">=", "<", ">") or dtype not in ("float32", "float64"):
                return rejected
            left, right = float(measured), float(threshold)
            if comparator == "<=":
                value = left <= right
            elif comparator == ">=":
                value = left >= right
            elif comparator == "<":
                value = left < right
            else:
                value = left > right
            witness = {"value": left, "threshold": right, "comparator": comparator, "dtype": dtype}
            decision_state = "pass" if value else "fail"
        elif claim_type == "exact_digest":
            observed, expected = raw["observed"], raw["expected"]
            if not isinstance(observed, str) or not isinstance(expected, str) or not gating:
                return rejected
            if len(observed) != 64 or len(expected) != 64:
                return rejected
            int(observed, 16)
            int(expected, 16)
            value = observed == expected
            decision_state = "pass" if value else "fail"
            witness = {"observed": observed, "expected": expected}
        else:
            values = raw["values"]
            if not isinstance(values, list) or not gating:
                return rejected
            if any(not isinstance(value, bool) for value in values):
                return rejected
            value = all(values)
            decision_state = "pass" if value else "fail"
            witness = {"values": values}
    except (KeyError, TypeError, ValueError, OverflowError):
        return rejected
    return {
        "accepted": True,
        "claim_type": claim_type,
        "gating": gating,
        "value": value,
        "decision_state": decision_state,
        "authorizes": decision_state == "pass",
        "witness": witness,
    }


def append(checks: list[dict[str, Any]], name: str, passed: bool) -> None:
    checks.append({"name": name, "pass": bool(passed)})


def audit() -> None:
    if AUDIT_PATH.exists():
        raise RuntimeError("Phase1187 audit already exists")
    protocol = read_json(PROTOCOL_PATH)
    contract = read_json(CONTRACT_PATH)
    fixtures = read_jsonl(FIXTURES_PATH)
    main_rows = read_jsonl(ROWS_PATH)
    migration = read_json(MIGRATION_PATH)
    final = read_json(FINAL_PATH)
    checks: list[dict[str, Any]] = []

    protocol_copy = dict(protocol)
    protocol_digest = protocol_copy.pop("protocol_digest")
    append(checks, "protocol_digest", digest(protocol_copy) == protocol_digest)
    contract_copy = dict(contract)
    contract_digest = contract_copy.pop("contract_digest")
    append(checks, "contract_digest", digest(contract_copy) == contract_digest)
    append(checks, "runner_hash", file_sha256(RUNNER) == protocol["scripts"]["runner"])
    append(checks, "audit_hash", file_sha256(SCRIPT) == protocol["scripts"]["audit"])
    append(checks, "contract_hash", file_sha256(CONTRACT_PATH) == protocol["contract_sha256"])
    append(checks, "fixtures_hash", file_sha256(FIXTURES_PATH) == protocol["fixtures_sha256"])
    append(checks, "rows_hash", file_sha256(ROWS_PATH) == final["artifacts"]["compiled_claims_sha256"])
    append(checks, "migration_hash", file_sha256(MIGRATION_PATH) == final["artifacts"]["migration_sha256"])
    for name, relative in (
        ("phase1186_final", "analysis/final.json"),
        ("phase1186_audit", "audit/independent_audit.json"),
        ("phase1186_gauge_rows", "analysis/gauge_rows.jsonl"),
        ("phase1186_positive_rows", "analysis/positive_control_rows.jsonl"),
    ):
        append(checks, f"{name}_hash", file_sha256(PHASE1186_ROOT / relative) == protocol["scripts"][name])

    gauge_source = read_jsonl(PHASE1186_ROOT / "analysis/gauge_rows.jsonl")
    positive_source = read_jsonl(PHASE1186_ROOT / "analysis/positive_control_rows.jsonl")
    migrated_typed: list[dict[str, Any]] = []
    for row in gauge_source:
        for name in ("decision", "margin_sign"):
            source = row["exact_decision"][name]
            migrated_typed.append(
                independent_compile(
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
        for row in positive_source
    ]

    independent: list[dict[str, Any]] = []
    for fixture in fixtures:
        compiled = independent_compile(fixture["raw"], contract)
        expected_match = compiled["accepted"] == fixture["expected_accepted"]
        if fixture["expected_accepted"]:
            expected_match = bool(
                expected_match
                and compiled["value"] == fixture["expected_value"]
                and compiled["decision_state"] == fixture["expected_decision_state"]
                and compiled["authorizes"] == fixture["expected_authorizes"]
            )
        independent.append({"case_id": fixture["case_id"], "compiled": compiled, "expected_match": expected_match})
    append(checks, "fixture_count", len(fixtures) == len(main_rows) == protocol["fixture_count"])
    append(checks, "independent_known_truth", all(row["expected_match"] for row in independent))
    append(
        checks,
        "main_independent_semantic_match",
        all(
            left["case_id"] == right["case_id"]
            and left["compiled"]["accepted"] == right["compiled"]["accepted"]
            and left["compiled"]["claim_type"] == right["compiled"]["claim_type"]
            and left["compiled"]["gating"] == right["compiled"]["gating"]
            and left["compiled"]["value"] == right["compiled"]["value"]
            and left["compiled"]["decision_state"] == right["compiled"]["decision_state"]
            and left["compiled"]["authorizes"] == right["compiled"]["authorizes"]
            and left["compiled"]["witness"] == right["compiled"]["witness"]
            for left, right in zip(main_rows, independent)
        ),
    )
    invalid = [row for row, fixture in zip(independent, fixtures) if not fixture["expected_accepted"]]
    append(checks, "all_invalid_rejected", all(not row["compiled"]["accepted"] for row in invalid))
    append(
        checks,
        "descriptive_never_gates",
        all(
            not row["compiled"]["gating"]
            for row in independent
            if row["compiled"]["accepted"] and row["compiled"]["claim_type"] == "descriptive_ratio"
        ),
    )
    append(
        checks,
        "abstention_never_authorizes",
        all(
            not row["compiled"]["authorizes"]
            for row in independent
            if row["compiled"]["decision_state"] == "abstain"
        ),
    )
    append(
        checks,
        "only_pass_authorizes",
        all(
            row["compiled"]["authorizes"] == (row["compiled"]["decision_state"] == "pass")
            for row in independent
        ),
    )
    append(checks, "migration_digest", digest({key: value for key, value in migration.items() if key != "migration_digest"}) == migration["migration_digest"])
    append(
        checks,
        "migration_all_typed_accepted",
        migration["accepted_count"] == migration["typed_claim_count"] == 512,
    )
    append(
        checks,
        "migration_type_split",
        migration["source_field_count"] == 576
        and migration["gating_claim_count"] == 512
        and migration["descriptive_claim_count"] == 0
        and migration["legacy_untyped_descriptive_count"] == 64
        and migration["descriptive_gate_count"] == 0,
    )
    append(checks, "migration_independent_typed_acceptance", all(row["accepted"] for row in migrated_typed))
    append(
        checks,
        "migration_independent_state_counts",
        migration["decision_state_counts"]
        == {
            state: sum(row["decision_state"] == state for row in migrated_typed)
            for state in ("pass", "fail", "abstain", "non_gating", "invalid")
        },
    )
    append(
        checks,
        "migration_independent_authorization_count",
        migration["authorization_count"] == sum(row["authorizes"] for row in migrated_typed),
    )
    append(
        checks,
        "legacy_descriptive_quarantine_digest",
        migration["legacy_untyped_descriptive_digest"] == digest(legacy_descriptive),
    )
    append(checks, "phase1186_verdict_unchanged", migration["phase1186_audit_pass_unchanged"] is False and migration["phase1186_authorization_unchanged"] is False)
    final_copy = dict(final)
    final_digest = final_copy.pop("final_digest")
    append(checks, "final_digest", digest(final_copy) == final_digest)
    expected_main = bool(
        all(row["expected_match"] for row in independent)
        and all(not row["compiled"]["accepted"] for row in invalid)
        and migration["accepted_count"] == migration["typed_claim_count"] == 512
        and migration["legacy_untyped_descriptive_count"] == 64
        and migration["descriptive_gate_count"] == 0
    )
    append(checks, "main_decision_recompute", final["main_pass"] == expected_main)
    append(checks, "claim_scope", final["claim_scope"] == "typed_evidence_infrastructure_only")

    audit_pass = all(check["pass"] for check in checks)
    authorized = bool(audit_pass and final["main_pass"])
    result = {
        "phase": 1187,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_digest": protocol_digest,
        "contract_digest": contract_digest,
        "final_digest": final_digest,
        "check_count": len(checks),
        "pass_count": sum(check["pass"] for check in checks),
        "checks": checks,
        "audit_pass": audit_pass,
        "qualification_development_frozen": audit_pass,
        "phase1188_authorized_after_audit": authorized,
        "phase1186_verdict_unchanged": True,
        "k165_status": "untested",
        "auto_continue": {
            "authorized": authorized,
            "next": "one_new_terminal_three_evidence_registry" if authorized else None,
        },
    }
    result["audit_digest"] = digest(result)
    write_json(AUDIT_PATH, result)
    print(canonical_json(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit",))
    args = parser.parse_args()
    if args.command == "audit":
        audit()


if __name__ == "__main__":
    main()
