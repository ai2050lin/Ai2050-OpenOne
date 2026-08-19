#!/usr/bin/env python3
"""Audit that Phase1327 stopped for the frozen zero-model failure only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1327_c040_cross_model_noun_relation_contract"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run() -> None:
    protocol = load(OUT / "protocol/preregistration.json")
    final = load(OUT / "analysis/final.json")
    machine = load(OUT / "audit/tokenizer_zero_model_audit.json")
    original = load(OUT / "audit/independent_final_audit.json")
    timeless = {key: value for key, value in protocol.items()
                if key not in {"contract_sha256", "script_sha256", "auditor_sha256", "created_at_utc"}}
    checks = {
        "contract_immutable": digest(timeless) == protocol["contract_sha256"],
        "original_audit_localizes_failure": (
            original["all_checks_passed"] is False
            and {name for name, passed in original["checks"].items() if not passed} == {"zero_models", "final"}
        ),
        "one_frozen_zero_model_failed": (
            machine["zero_models"]["candidate_identity_majority"] == 0.5833333333333334
            and machine["zero_models"]["candidate_identity_majority"] > protocol["zero_models"]["max_identity"]
        ),
        "other_zero_models_pass": (
            machine["zero_models"]["candidate_position"] == 0.5
            and machine["zero_models"]["lexicographic"] <= protocol["zero_models"]["max_lexicographic"]
            and machine["zero_models"]["target_char_bigram_overlap"] <= protocol["zero_models"]["max_char_overlap"]
            and max(machine["zero_models"]["per_model_shorter_token"].values())
            <= protocol["zero_models"]["max_token_length"]
        ),
        "formal_stop": final["all_gates_passed"] is False and final["authorization"] == "stop_c040_before_model",
        "no_model_or_hidden_artifacts": not (OUT / "raw").exists() and not (OUT / "field").exists(),
    }
    output = {"phase": 1327, "campaign": "C040", "audit_type": "pre_model_failure_adjudication",
              "checks": checks, "passed": sum(checks.values()), "total": len(checks),
              "all_checks_passed": all(checks.values()),
              "authorization": "close_c040_and_permit_new_independent_contract" if all(checks.values()) else "none"}
    save(OUT / "audit/independent_failure_audit.json", output)
    print(json.dumps(output, indent=2))
    if not output["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
