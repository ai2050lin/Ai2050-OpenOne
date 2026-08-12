#!/usr/bin/env python3
"""Independently audit Phase1117 protocol, behavior, and final decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import phase1117_pythia_training_dynamics_finalize as finalize
import phase1117_pythia_training_dynamics_protocol as protocol


def audit() -> dict[str, Any]:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    selected = protocol.read_json(protocol.OUT_ROOT / "protocol" / "selected_concepts.json")
    cases = list(protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / "cases.jsonl"))
    final_summary = protocol.read_json(protocol.OUT_ROOT / "analysis" / "final_summary.json")
    authorization = protocol.read_json(protocol.OUT_ROOT / "analysis" / "trajectory_authorization.json")
    integrity = protocol.read_json(protocol.OUT_ROOT / "protocol" / "checkpoint_integrity.json")

    prereg_core = dict(prereg)
    prereg_digest = prereg_core.pop("protocol_digest")
    final_core = dict(final_summary)
    final_digest = final_core.pop("final_digest")
    authorization_core = dict(authorization)
    authorization_digest = authorization_core.pop("authorization_digest")
    integrity_core = dict(integrity)
    integrity_digest = integrity_core.pop("integrity_digest")

    global_checks = {
        "protocol_audit_passed": protocol_audit["all_checks_passed"],
        "protocol_digest_recomputed": protocol.digest(prereg_core) == prereg_digest,
        "case_digest_recomputed": protocol.digest(cases) == prereg["case_digest"],
        "selected_digest_recomputed": protocol.digest(selected["selected"]) == prereg["selected_digest"],
        "final_digest_recomputed": protocol.digest(final_core) == final_digest,
        "authorization_digest_recomputed": protocol.digest(authorization_core) == authorization_digest,
        "integrity_audit_passed": integrity["all_checks_passed"],
        "integrity_digest_recomputed": protocol.digest(integrity_core) == integrity_digest,
        "hidden_state_not_authorized": not authorization["hidden_state_authorized"] and not final_summary["hidden_state_authorized"],
        "no_hidden_or_causal_artifacts": not any(
            any(token in path.name.lower() for token in ("hidden", "activation", "attention", "causal", "neuron"))
            for path in protocol.OUT_ROOT.rglob("*")
            if path.is_file()
        ),
    }

    checkpoint_audits: dict[str, Any] = {}
    recomputed_metrics: dict[str, Any] = {}
    for checkpoint in final_summary["present_checkpoints"]:
        root = protocol.OUT_ROOT / "behavior" / checkpoint
        detail = list(protocol.read_jsonl(root / "candidate_detail.jsonl"))
        summary = protocol.read_json(root / "summary.json")
        summary_core = dict(summary)
        summary_digest = summary_core.pop("summary_digest")
        metrics = finalize.compute_checkpoint(checkpoint, detail)
        recomputed_metrics[checkpoint] = metrics
        local_model_root = protocol.MODEL_ROOT / checkpoint
        manifest_checks = []
        for entry in summary["model_manifest"]:
            path = local_model_root / entry["path"]
            manifest_checks.append(
                path.exists()
                and path.stat().st_size == entry["size"]
                and protocol.file_sha256(path) == entry["sha256"]
            )
        checks = {
            "detail_count": len(detail) == prereg["case_count"] == summary["case_count"],
            "detail_digest": protocol.digest(detail) == summary["detail_digest"],
            "summary_digest": protocol.digest(summary_core) == summary_digest,
            "protocol_digest": summary["protocol_digest"] == prereg["protocol_digest"],
            "case_digest": summary["case_digest"] == prereg["case_digest"],
            "precision_fp16": summary["precision"]["has_fp16_parameters"],
            "precision_not_bf16": not summary["precision"]["has_bf16_parameters"],
            "precision_not_quantized": not summary["precision"]["has_quantized_modules"],
            "declared_weight_format": summary.get("weight_format") == protocol.WEIGHT_FORMAT,
            "parameter_probe_present": bool(summary.get("parameter_probe", {}).get("digest")),
            "parameter_probe_matches_preflight": summary.get("parameter_probe") == integrity["checkpoints"][checkpoint]["parameter_probe"],
            "manifest_digest": protocol.digest(summary["model_manifest"]) == summary["model_manifest_digest"],
            "manifest_files": bool(manifest_checks) and all(manifest_checks),
            "metrics_recomputed": protocol.digest(metrics) == protocol.digest(final_summary["checkpoint_metrics"][checkpoint]),
        }
        checkpoint_audits[checkpoint] = {"checks": checks, "all_checks_passed": all(checks.values())}

    probe_groups: dict[str, list[str]] = {}
    for checkpoint in prereg["checkpoints"]:
        summary = protocol.read_json(protocol.OUT_ROOT / "behavior" / checkpoint / "summary.json")
        probe_groups.setdefault(summary["parameter_probe"]["digest"], []).append(checkpoint)
    nonadjacent_probe_collisions = [
        group for group in probe_groups.values() if len(group) > 1 and set(group) != {"step0", "step1"}
    ]
    global_checks["no_nonadjacent_parameter_probe_collision"] = not nonadjacent_probe_collisions

    global_checks["checkpoint_set_matches"] = set(checkpoint_audits) == set(final_summary["present_checkpoints"])
    global_checks["checkpoint_metrics_recomputed"] = protocol.digest(recomputed_metrics) == protocol.digest(final_summary["checkpoint_metrics"])
    all_checks_passed = all(global_checks.values()) and all(value["all_checks_passed"] for value in checkpoint_audits.values())
    result_core = {
        "schema_version": "phase1117_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "final_digest": final_summary["final_digest"],
        "global_checks": global_checks,
        "checkpoint_audits": checkpoint_audits,
        "parameter_probe_groups": list(probe_groups.values()),
        "nonadjacent_parameter_probe_collisions": nonadjacent_probe_collisions,
        "all_checks_passed": all_checks_passed,
    }
    result = dict(result_core)
    result["audit_digest"] = protocol.digest(result_core)
    protocol.write_json(protocol.OUT_ROOT / "audit" / "result_audit.json", result)
    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)
