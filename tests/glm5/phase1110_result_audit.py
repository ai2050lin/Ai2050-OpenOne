#!/usr/bin/env python3
"""Independent integrity audit for Phase1110 frozen value-read results."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1110_frozen_value_read_protocol as protocol


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    protocol_root = protocol.OUT_ROOT / "protocol"
    analysis_root = protocol.OUT_ROOT / "analysis"
    prereg = protocol.read_json(protocol_root / "preregistration.json")
    protocol_audit = protocol.read_json(protocol_root / "audit.json")
    decisions = protocol.read_json(analysis_root / "model_value_decisions.json")
    final = protocol.read_json(analysis_root / "final_summary.json")
    checks = {
        "protocol_digest": protocol.digest({key: value for key, value in prereg.items() if key != "protocol_digest"}) == prereg["protocol_digest"],
        "protocol_audit_digest": protocol.digest({key: value for key, value in protocol_audit.items() if key != "audit_digest"}) == protocol_audit["audit_digest"],
        "protocol_audit_passed": bool(protocol_audit["all_checks_passed"]),
        "final_digest": protocol.digest({key: value for key, value in final.items() if key != "final_summary_digest"}) == final["final_summary_digest"],
        "phase_exact": int(final["phase"]) == protocol.PHASE,
        "authorized_models_exact": tuple(final["authorized_models"]) == protocol.AUTHORIZED_MODELS,
        "denied_models_exact": tuple(final["denied_models"]) == protocol.DENIED_MODELS,
        "decision_models_exact": set(decisions) == set(protocol.AUTHORIZED_MODELS),
        "causal_stop_preserved": not final["causal_staircase_authorized"] and not final["component_head_qkv_neuron_localization_authorized"],
        "automatic_stop_preserved": not final["automatic_next_required"],
        "canonical_theory_name_preserved": final["canonical_theory_name_unchanged"] == "条件化输出场闭合理论",
        "P8_false": final["prospective_predictions"]["P8"] is False,
        "denial_exists": (protocol.OUT_ROOT / "atlas" / "deepseek7b" / "denial.json").exists(),
    }
    file_hashes = {}
    for model in protocol.AUTHORIZED_MODELS:
        atlas_root = protocol.OUT_ROOT / "atlas" / model
        summary = protocol.read_json(atlas_root / "summary.json")
        rows = list(protocol.read_jsonl(protocol_root / f"cases.{model}.jsonl"))
        with np.load(atlas_root / "frozen_value_read_fields.npz") as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        checks[f"{model}_case_digest"] = protocol.digest(rows) == prereg["case_digests"][model]
        checks[f"{model}_summary_digest"] = protocol.digest({key: value for key, value in summary.items() if key != "summary_digest"}) == summary["summary_digest"]
        checks[f"{model}_summary_link"] = decisions[model]["model_summary_digest"] == summary["summary_digest"] == final["model_summary_digests"][model]
        checks[f"{model}_instrument"] = bool(summary["all_checks_passed"])
        checks[f"{model}_arrays_finite"] = all(np.isfinite(value).all() for value in arrays.values())
        checks[f"{model}_shape"] = arrays["attention_mass"].shape == (48, 64, 4, 5)
        checks[f"{model}_reconstruction"] = float(np.max(arrays["reconstruction_relative_error"])) <= prereg["thresholds"]["maximum_head_reconstruction_relative_error"]
        file_hashes[model] = {
            "cases": file_sha256(protocol_root / f"cases.{model}.jsonl"),
            "fields": file_sha256(atlas_root / "frozen_value_read_fields.npz"),
            "summary": file_sha256(atlas_root / "summary.json"),
            "units": file_sha256(atlas_root / "units.json"),
        }
    result = {
        "schema_version": "phase1110_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "final_summary_digest": final["final_summary_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_check_count": sum(checks.values()),
        "all_checks_passed": all(checks.values()),
        "file_hashes": file_hashes,
    }
    result["audit_digest"] = protocol.digest(result)
    audit_root = protocol.OUT_ROOT / "audit"
    audit_root.mkdir(parents=True, exist_ok=True)
    protocol.write_json(audit_root / "result_audit.json", result)
    print(json.dumps({
        "phase": protocol.PHASE,
        "passed": result["passed_check_count"],
        "total": result["check_count"],
        "all_checks_passed": result["all_checks_passed"],
        "audit_digest": result["audit_digest"],
    }, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
