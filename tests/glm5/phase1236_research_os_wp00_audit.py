#!/usr/bin/env python3
"""Independent, no-model audit for C001-WP00 research-OS migration."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[2]
OS_ROOT = WORKSPACE / "research" / "ai2050_research_os"
RESULT_DIR = WORKSPACE / "tests" / "glm5" / "result" / "phase1236_research_os_wp00"
RESULT_PATH = RESULT_DIR / "wp00_verification.json"
PYTHON = WORKSPACE / ".venv" / "Scripts" / "python.exe"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_researchctl():
    path = OS_ROOT / "scripts" / "researchctl.py"
    spec = importlib.util.spec_from_file_location("ai2050_researchctl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load researchctl")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(*args: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [str(PYTHON), str(OS_ROOT / "scripts" / "researchctl.py"), *args],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=120,
    )
    return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def main() -> int:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    researchctl = load_researchctl()
    data = researchctl.load_all()
    validation_errors = researchctl.validate(data)
    record("registry_validation", not validation_errors, validation_errors)

    sources = data["sources"]
    source_ids = {item["id"] for item in sources}
    source_paths_exist = all((WORKSPACE / item["path"]).is_file() for item in sources)
    no_upload_refs = all("../upload" not in json.dumps(item, ensure_ascii=False) for item in data["evidence"])
    record("source_registry_22_git_snapshots", len(sources) == 22 and len(source_ids) == 22 and source_paths_exist, {"count": len(sources), "paths_exist": source_paths_exist})
    record("legacy_upload_refs_removed", no_upload_refs, "evidence contains no ../upload reference")
    record("nul_sources_typed", all(item["nul_count"] == 0 or item["authority"] != "primary_evidence" for item in sources), [{"id": item["id"], "nul_count": item["nul_count"], "authority": item["authority"]} for item in sources if item["nul_count"]])

    phase_records = data["phases"]
    record_ids = [item["record_id"] for item in phase_records]
    record("composite_phase_identity", len(record_ids) == len(set(record_ids)) and all("phase_label" in item and "occurrence" in item for item in phase_records), {"records": len(record_ids)})

    duplicate_data = copy.deepcopy(data)
    duplicate = copy.deepcopy(duplicate_data["phases"][-1])
    duplicate["record_id"] = "PHREC-AUDIT-DUPLICATE-LABEL"
    duplicate["occurrence"] = 2
    duplicate["evidence_refs"] = []
    duplicate["object_ids"] = []
    duplicate["construct_ids"] = []
    duplicate_data["phases"].append(duplicate)
    duplicate_errors = researchctl.validate(duplicate_data)
    duplicate_key_errors = [item for item in duplicate_errors if "主键重复" in item or "latest_recorded_phase" in item]
    record("duplicate_phase_label_allowed", not duplicate_key_errors, duplicate_key_errors)

    contracts = data["contracts"]
    contract_index = contracts[0]
    contract_path = OS_ROOT / contract_index["path"]
    contract = load_json(contract_path)
    contract_hash_ok = sha256(contract_path) == contract_index["contract_sha256"]
    typed_constructs = set(contract["construct_types"])
    expected_constructs = {"CON-CONTENT-SELECTION", "CON-EXACT-FORMAT", "CON-NATURAL-GENERATION", "CON-STOP-CACHE"}
    record("contract_frozen", contract_hash_ok and contract_index["frozen_at"] is not None, {"sha256": contract_index["contract_sha256"], "frozen_at": contract_index["frozen_at"]})
    record("four_typed_constructs", typed_constructs == expected_constructs and len(contract["typed_gates"]) == 4, sorted(typed_constructs))
    record("contract_not_run_ready", contract_index["run_ready"] is False and contract["frozen_artifacts"]["readiness"] == "contract_frozen", contract["frozen_artifacts"])
    record("no_hidden_authorization", all(term not in contract["scope"]["allowed_observations"] for term in ("hidden_state", "attention", "mlp", "neuron")), contract["scope"]["allowed_observations"])
    record("sealed_confirmation", contract["data_contract"]["confirmation"]["sealed"] is True, contract["data_contract"]["confirmation"])
    record("zero_adaptive_rounds", contract["budget"]["max_adaptive_rounds"] == 0 and contract["budget"]["max_runs"] == 1, contract["budget"])

    schema = load_json(OS_ROOT / contract_index["schema_path"])
    mutated = copy.deepcopy(contract)
    mutated["posthoc_override"] = True
    schema_errors: list[str] = []
    researchctl.validate_schema(mutated, schema, "mutated_contract", schema_errors)
    record("schema_rejects_extra_fields", any("未声明字段 posthoc_override" in item for item in schema_errors), schema_errors)

    campaign = data["campaigns"][0]
    stage_budget_sum = sum(stage["gpu_hour_budget"] for stage in campaign["stages"])
    active_stages = [stage["id"] for stage in campaign["stages"] if stage["status"] == "active"]
    record("campaign_budget_frozen", stage_budget_sum == campaign["max_gpu_hours"] == 24, {"stage_sum": stage_budget_sum, "campaign": campaign["max_gpu_hours"]})
    record("wp00_complete_wp01_active", campaign["stages"][0]["status"] == "completed" and active_stages == ["WP01"], active_stages)

    manifest_cli = run_cli("verify-manifest", "manifests/EXP-C001-WP01-001.manifest.json")
    validate_cli = run_cli("validate")
    clean_cli = run_cli("build", "--check-clean")
    record("manifest_cli", manifest_cli["returncode"] == 0, manifest_cli)
    record("validate_cli", validate_cli["returncode"] == 0, validate_cli)
    record("generated_views_clean", clean_cli["returncode"] == 0, clean_cli)

    passed = all(item["passed"] for item in checks)
    result = {
        "phase": 1236,
        "campaign_id": "C001",
        "work_package_id": "WP00",
        "audit_type": "no_model_independent_engineering_audit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_runs": 0,
        "gpu_hours": 0,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "passed": passed,
        "checks": checks,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "checks": f"{result['checks_passed']}/{result['checks_total']}", "result": str(RESULT_PATH)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
