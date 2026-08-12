#!/usr/bin/env python3
"""Independently audit Phase1101 artifacts and frozen decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import phase1101_relation_identity_routing_protocol as protocol


TEST_ROOT = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def digest_without(row: dict, key: str) -> str:
    copied = dict(row)
    copied.pop(key, None)
    return protocol.digest(copied)


def check(name: str, passed: bool, detail=None) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def main() -> None:
    root = protocol.OUT_ROOT
    prereg = protocol.read_json(root / "protocol" / "preregistration.json")
    common_audit = protocol.read_json(root / "protocol" / "audit.json")
    behavior = protocol.read_json(root / "analysis" / "behavior_authorization.json")
    final = protocol.read_json(root / "analysis" / "final_summary.json")
    diagnostic = protocol.read_json(root / "analysis" / "failure_diagnostic.json")
    source_final = protocol.read_json(protocol.SOURCE_PHASE1100)
    source_audit = protocol.read_json(protocol.SOURCE_PHASE1100_AUDIT)
    checks = []
    checks.append(check(
        "protocol_digest_recomputes",
        digest_without(prereg, "protocol_digest") == prereg["protocol_digest"],
    ))
    checks.append(check(
        "protocol_audit_digest_recomputes",
        digest_without(common_audit, "audit_digest") == common_audit["audit_digest"],
    ))
    checks.append(check("protocol_audit_passed", common_audit["all_checks_passed"]))
    checks.append(check(
        "source_phase1100_final_frozen",
        source_final["final_digest"] == prereg["source_phase1100_final_digest"],
    ))
    checks.append(check(
        "source_phase1100_audit_frozen",
        source_audit["audit_digest"] == prereg["source_phase1100_audit_digest"],
    ))
    checks.append(check(
        "behavior_authorization_digest_recomputes",
        digest_without(behavior, "authorization_digest") == behavior["authorization_digest"],
    ))
    checks.append(check(
        "final_digest_recomputes",
        digest_without(final, "final_digest") == final["final_digest"],
    ))
    checks.append(check(
        "diagnostic_digest_recomputes",
        digest_without(diagnostic, "diagnostic_digest") == diagnostic["diagnostic_digest"],
    ))
    expected_cases = (
        len(protocol.RELATION_PAIRS) * len(protocol.SURFACES)
        * len(protocol.TEMPLATES) * protocol.ITEMS_PER_TEMPLATE
        * len(protocol.STATES)
    )
    expected_superunits = (
        len(protocol.SURFACES) * len(protocol.TEMPLATES)
        * protocol.ITEMS_PER_TEMPLATE
    )
    for model in protocol.MODELS:
        case_path = root / "protocol" / f"cases.{model}.jsonl"
        case_rows = protocol.read_jsonl(case_path)
        model_audit = protocol.read_json(root / "protocol" / f"audit.{model}.json")
        behavior_summary = protocol.read_json(root / "behavior" / model / "summary.json")
        candidate_detail = protocol.read_jsonl(
            root / "behavior" / model / "candidate_detail.jsonl"
        )
        generation_detail = protocol.read_jsonl(
            root / "behavior" / model / "generation_detail.jsonl"
        )
        atlas_root = root / "atlas" / model
        atlas_summary = protocol.read_json(atlas_root / "summary.json")
        atlas_index = protocol.read_jsonl(atlas_root / "superunit_index.jsonl")
        checks.extend((
            check(f"{model}_case_count", len(case_rows) == expected_cases, len(case_rows)),
            check(f"{model}_case_digest", protocol.digest(case_rows) == prereg["model_case_digests"][model]),
            check(f"{model}_model_audit_passed", model_audit["all_checks_passed"]),
            check(f"{model}_behavior_summary_digest", digest_without(behavior_summary, "summary_digest") == behavior_summary["summary_digest"]),
            check(f"{model}_candidate_detail_count", len(candidate_detail) == expected_cases, len(candidate_detail)),
            check(f"{model}_generation_detail_count", len(generation_detail) == behavior_summary["generation_count"], len(generation_detail)),
            check(f"{model}_fp16", behavior_summary["precision"]["has_fp16_parameters"]),
            check(f"{model}_not_bf16", not behavior_summary["precision"]["has_bf16_parameters"]),
            check(f"{model}_not_quantized", not behavior_summary["precision"]["has_quantized_modules"]),
            check(f"{model}_atlas_summary_digest", digest_without(atlas_summary, "summary_digest") == atlas_summary["summary_digest"]),
            check(f"{model}_atlas_index_count", len(atlas_index) == expected_superunits, len(atlas_index)),
            check(f"{model}_exact_gram", atlas_summary["exact_full_d_model_gram"]),
            check(f"{model}_output_excluded", atlas_summary["primary_signature_excludes_output_gram"]),
        ))
        with np.load(
            atlas_root / "relation_identity_routing_geometry.npz",
            allow_pickle=False,
        ) as archive:
            gram = archive["pair_gram"]
            shared = archive["shared_energy"]
            differential = archive["differential_energy"]
            norms = archive["centered_norm"]
            output = archive["output_gram"]
            event_count = atlas_summary["event_count"]
            pair_count = len(protocol.RELATION_PAIRS)
            field_count = len(protocol.FIELDS)
            role_count = len(protocol.CAPTURE_ROLES)
            checks.extend((
                check(f"{model}_gram_shape", gram.shape == (expected_superunits, event_count, field_count, role_count, pair_count, pair_count), list(gram.shape)),
                check(f"{model}_shared_shape", shared.shape == (expected_superunits, event_count, field_count, role_count), list(shared.shape)),
                check(f"{model}_differential_shape", differential.shape == shared.shape, list(differential.shape)),
                check(f"{model}_norm_shape", norms.shape == (expected_superunits, event_count, field_count, role_count, pair_count), list(norms.shape)),
                check(f"{model}_output_shape", output.shape == (expected_superunits, field_count, pair_count, pair_count), list(output.shape)),
                check(f"{model}_gram_symmetric", bool(np.nanmax(np.abs(gram - np.swapaxes(gram, -1, -2))) <= 1e-5)),
                check(f"{model}_shared_finite", bool(np.isfinite(shared).mean() >= 0.97), float(np.isfinite(shared).mean())),
                check(f"{model}_differential_finite", bool(np.isfinite(differential).mean() >= 0.97), float(np.isfinite(differential).mean())),
            ))
    recomputed_gates = dict(final["gates"])
    checks.append(check(
        "automatic_next_matches_all_gates",
        final["automatic_next_required"] == all(recomputed_gates.values()),
    ))
    checks.append(check(
        "behavior_gate_matches_authorization",
        final["gates"]["P2"] == (
            sum(row["model_behavior_passed"] for row in behavior["models"].values())
            >= protocol.THRESHOLDS["minimum_behavior_models"]
        ),
    ))
    checks.append(check(
        "permutation_counts",
        final["registered_family_permutations"] == 119
        and final["registered_within_family_permutations"] == 7775,
    ))
    checks.append(check(
        "diagnostic_is_non_upgrading",
        diagnostic["evidence_status"] == "non_upgrading_post_frozen_diagnostic",
    ))
    script_paths = (
        TEST_ROOT / "phase1101_relation_identity_routing_protocol.py",
        TEST_ROOT / "phase1101_relation_identity_routing_behavior.py",
        TEST_ROOT / "phase1101_relation_identity_routing_behavior_finalize.py",
        TEST_ROOT / "phase1101_relation_identity_routing_scan.py",
        TEST_ROOT / "phase1101_relation_identity_routing_finalize.py",
        TEST_ROOT / "phase1101_relation_identity_routing_diagnostic.py",
        TEST_ROOT / "phase1101_relation_identity_routing_result_audit.py",
        TEST_ROOT / "phase1101_run_sequential.py",
    )
    checks.append(check("all_scripts_exist", all(path.exists() for path in script_paths)))
    failed = [row for row in checks if not row["passed"]]
    result = {
        "schema_version": "phase1101_result_audit.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "final_digest": final["final_digest"],
        "check_count": len(checks),
        "passed_count": len(checks) - len(failed),
        "failed_count": len(failed),
        "all_checks_passed": not failed,
        "checks": checks,
        "artifact_sha256": {
            "preregistration": file_sha256(root / "protocol" / "preregistration.json"),
            "behavior_authorization": file_sha256(root / "analysis" / "behavior_authorization.json"),
            "final_summary": file_sha256(root / "analysis" / "final_summary.json"),
            "failure_diagnostic": file_sha256(root / "analysis" / "failure_diagnostic.json"),
            **{
                f"atlas_{model}": file_sha256(
                    root / "atlas" / model / "relation_identity_routing_geometry.npz"
                ) for model in protocol.MODELS
            },
            **{
                f"script_{path.stem}": file_sha256(path) for path in script_paths
                if path.exists()
            },
        },
    }
    result["audit_digest"] = protocol.digest(result)
    protocol.write_json(root / "audit" / "result_audit.json", result)
    print(json.dumps({
        "phase": protocol.PHASE,
        "check_count": len(checks),
        "failed_count": len(failed),
        "all_checks_passed": not failed,
        "failed": failed[:5],
        "audit_digest": result["audit_digest"],
    }, ensure_ascii=False), flush=True)
    if failed:
        raise RuntimeError(f"Phase1101 result audit failed: {failed[:5]}")


if __name__ == "__main__":
    main()
