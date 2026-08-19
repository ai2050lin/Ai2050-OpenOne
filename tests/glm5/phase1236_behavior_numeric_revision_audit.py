#!/usr/bin/env python3
"""Independent audit of the Phase1236 non-finite serialization revision."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
OUT_ROOT = TEST_ROOT / "result/phase1236_global_functional_structure_identification"
BASE_MAIN = TEST_ROOT / "phase1236_global_functional_structure_identification.py"
BASE_AUDIT = TEST_ROOT / "phase1236_global_functional_structure_identification_audit.py"
RUNNER = TEST_ROOT / "phase1236_behavior_numeric_revision.py"
AUDIT = Path(__file__).resolve()
CONTRACT = OUT_ROOT / "protocol/preregistration.json"
PREAUDIT = OUT_ROOT / "audit/independent_preaudit.json"
REVISION = OUT_ROOT / "protocol/behavior_numeric_revision1.json"
OUTPUT = OUT_ROOT / "audit/behavior_numeric_revision1_audit.json"
MATERIAL = OUT_ROOT / "material/frozen_response_worlds.jsonl"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def strip(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: item for name, item in value.items() if name != key}


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("numeric revision audit already exists")
    contract = read(CONTRACT)
    preaudit = read(PREAUDIT)
    revision = read(REVISION)
    failure_log = ROOT / revision["trigger"]["failure_log"]
    checks = {
        "base_contract_digest": contract["contract_digest"] == digest(strip(contract, "contract_digest")),
        "base_preaudit": preaudit.get("all_checks_passed") is True and preaudit.get("contract_digest") == contract["contract_digest"],
        "base_sources_unchanged": revision["base_source_hashes"] == {
            "main": file_sha256(BASE_MAIN), "audit": file_sha256(BASE_AUDIT)
        },
        "revision_digest": revision["revision_digest"] == digest(strip(revision, "revision_digest")),
        "revision_sources": revision["revision_source_hashes"] == {
            "runner": file_sha256(RUNNER), "independent_audit": file_sha256(AUDIT)
        },
        "failure_log_exists": failure_log.exists(),
        "failure_log_digest": failure_log.exists() and revision["trigger"]["failure_log_sha256"] == file_sha256(failure_log),
        "failure_is_nonfinite_serialization": failure_log.exists() and "Out of range float values" in failure_log.read_text(encoding="utf-8", errors="replace"),
        "material_frozen": revision["frozen_invariants"]["material_digest"] == contract["material"]["material_digest"],
        "manifests_frozen": revision["frozen_invariants"]["manifest_digests"] == {
            model: contract["manifest_summaries"][model]["manifest_digest"] for model in ("qwen3", "glm4", "deepseek7b")
        },
        "thresholds_frozen": revision["frozen_invariants"]["thresholds"] == contract["thresholds"],
        "execution_frozen": revision["frozen_invariants"]["precision"] == "float16" and revision["frozen_invariants"]["quantization"] == "none",
        "narrow_allowed_changes": len(revision["allowed_changes"]) == 4 and revision["scientific_claim"].startswith("instrument repair only"),
        "material_file_still_matches": contract["material"]["material_digest"] == digest([
            json.loads(line) for line in MATERIAL.read_text(encoding="utf-8").splitlines() if line.strip()
        ]),
    }
    value: dict[str, Any] = {
        "phase": 1236,
        "schema_version": "phase1236.behavior_numeric_revision1_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "revision_digest": revision["revision_digest"],
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    value["audit_digest"] = digest(value)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical(value))
    if not value["all_checks_passed"]:
        raise RuntimeError([name for name, passed in checks.items() if not passed])


if __name__ == "__main__":
    main()
