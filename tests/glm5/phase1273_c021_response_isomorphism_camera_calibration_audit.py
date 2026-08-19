#!/usr/bin/env python3
"""Independent audit for Phase1273 known-truth response-isomorphism calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1273_c021_response_isomorphism_camera_calibration as phase


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def preaudit() -> None:
    protocol, rows = phase.verify_protocol()
    checks: list[dict[str, Any]] = []
    check(checks, "contract_exists", phase.CONTRACT.exists())
    check(checks, "source_hashes_match", protocol["source_hashes"] == phase.protocol_payload(rows)["source_hashes"])
    check(checks, "system_count", len(rows) == len(phase.FAMILIES) * len(phase.DEPTHS) * len(phase.IMPLEMENTATIONS), len(rows))
    check(checks, "all_typed_qualified", all(row["typed_status"]["qualification"] == phase.RunStatus.QUALIFIED.value for row in rows))
    check(checks, "all_typed_measured", all(row["typed_status"]["measurement"] == phase.RunStatus.MEASURED.value for row in rows))
    check(checks, "all_claims_initially_abstain", all(row["typed_status"]["claim"] == phase.RunStatus.ABSTAINED.value for row in rows))
    check(checks, "confirmation_sealed", protocol["selection"].startswith("thresholds are optimized on discovery"))
    payload = {"phase": phase.PHASE, "mode": "preaudit", "checks": checks, "passed": all(item["passed"] for item in checks), "protocol_hash": file_sha256(phase.PROTOCOL)}
    write_json(phase.PREAUDIT, payload)
    print(json.dumps({"passed": payload["passed"], "checks": len(checks)}))
    if not payload["passed"]: raise SystemExit(1)


def final_audit() -> None:
    protocol, rows = phase.verify_protocol()
    ledger, final, summary = phase.read_jsonl(phase.RAW), read_json(phase.FINAL), read_json(phase.SUMMARY)
    recomputed = phase.analyze_ledger(ledger)
    checks: list[dict[str, Any]] = []
    check(checks, "formal_complete", phase.COMPLETE.exists() and read_json(phase.COMPLETE)["complete"])
    check(checks, "pair_count", len(ledger) == len(rows) * (len(rows) - 1) // 2, len(ledger))
    check(checks, "raw_hash", summary["pair_hash"] == file_sha256(phase.RAW))
    check(checks, "material_hash", summary["material_hash"] == file_sha256(phase.SYSTEMS))
    check(checks, "camera_registry", set(final["camera_results"]) == set(phase.CAMERAS))
    check(checks, "decision_recomputed", final["decision"] == recomputed["decision"])
    check(checks, "gates_recomputed", final["gates"] == recomputed["gates"])
    check(checks, "scores_recomputed", all(abs(final["camera_results"][camera]["confirmation_balanced_accuracy"] - recomputed["camera_results"][camera]["confirmation_balanced_accuracy"]) < 1.0e-12 for camera in phase.CAMERAS))
    check(checks, "discovery_only_thresholds", all(abs(final["camera_results"][camera]["threshold"] - recomputed["camera_results"][camera]["threshold"]) < 1.0e-12 for camera in phase.CAMERAS))
    check(checks, "random_sentinel_bounded", final["random_label_sentinel"]["balanced_accuracy"] <= protocol["thresholds"]["random_sentinel_balanced_accuracy_max"])
    check(checks, "physical_identity_refused", not final["gauge_policy"]["response_equivalent_physical_identity_authorized"])
    check(checks, "pretrained_denied", not final["pretrained_authorized"])
    check(checks, "typed_status_enum", {status.value for status in phase.RunStatus} == set(protocol["typed_evidence_contract"]["statuses"]))
    check(checks, "protocol_digest", protocol["protocol_digest"] == phase.protocol_payload(rows)["protocol_digest"])
    payload = {"phase": phase.PHASE, "mode": "final", "checks": checks, "passed": all(item["passed"] for item in checks), "passed_count": sum(item["passed"] for item in checks), "check_count": len(checks), "final_hash": file_sha256(phase.FINAL), "raw_hash": file_sha256(phase.RAW)}
    write_json(phase.FINAL_AUDIT, payload)
    print(json.dumps({"passed": payload["passed"], "passed_count": payload["passed_count"], "check_count": payload["check_count"]}))
    if not payload["passed"]: raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preaudit", "final"))
    args = parser.parse_args()
    preaudit() if args.mode == "preaudit" else final_audit()


if __name__ == "__main__":
    main()
