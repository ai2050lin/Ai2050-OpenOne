#!/usr/bin/env python3
"""Audit the Phase1089-1091 truth-control and cross-surface evidence chain."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1089_truth_matched_color_binding_protocol as p1089
import phase1090_cross_surface_color_behavior_protocol as p1090
import phase1091_cross_surface_color_signed_protocol as p1091


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest_without(data: dict[str, Any], key: str, module: Any) -> str:
    body = dict(data)
    body.pop(key, None)
    return module.digest(body)


def audit_protocol(module: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    root = module.OUT_ROOT
    prereg = module.read_json(root / "protocol" / "preregistration.json")
    global_audit = module.read_json(root / "protocol" / "audit.json")
    model_rows: dict[str, Any] = {}

    for model in module.MODELS:
        cases_path = root / "protocol" / f"cases.{model}.jsonl"
        model_audit_path = root / "protocol" / f"audit.{model}.json"
        cases = module.read_jsonl(cases_path)
        model_audit = module.read_json(model_audit_path)
        checks = {
            "case_file_present": cases_path.exists(),
            "model_audit_present": model_audit_path.exists(),
            "case_count_matches": len(cases) == int(prereg["case_count_per_model"]),
            "unit_count_matches": len({row["unit_id"] for row in cases})
            == int(prereg["unit_count_per_model"]),
            "case_digest_matches": module.digest(cases)
            == prereg["model_case_digests"][model]
            == model_audit["case_digest"],
            "model_static_audit_passed": model_audit["all_checks_passed"],
        }
        model_rows[model] = {
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "case_count": len(cases),
            "unit_count": len({row["unit_id"] for row in cases}),
        }

    checks = {
        "phase_matches": int(prereg["phase"]) == int(module.PHASE),
        "protocol_digest_recomputes": digest_without(
            prereg, "protocol_digest", module
        ) == prereg["protocol_digest"],
        "global_static_audit_passed": global_audit["all_checks_passed"],
        "model_order_qwen_glm_deepseek": tuple(
            prereg["sequential_model_order"]
        ) == ("qwen3", "glm4", "deepseek7b"),
        "fp16_preregistered": prereg["precision"] == "fp16",
        "quantization_forbidden": prereg["quantization"] == "none",
        "all_model_protocols_passed": all(
            row["all_checks_passed"] for row in model_rows.values()
        ),
    }
    return prereg, {
        "checks": checks,
        "models": model_rows,
        "all_checks_passed": all(checks.values()),
    }


def precision_checks(summary: dict[str, Any]) -> dict[str, bool]:
    precision = summary["precision"]
    return {
        "fp16_parameters_present": precision["has_fp16_parameters"],
        "bf16_parameters_absent": not precision["has_bf16_parameters"],
        "quantized_modules_absent": not precision["has_quantized_modules"],
        "placement_quantization_none": summary["placement"]["quantization"]
        == "none",
    }


def audit_signed_phase(
    module: Any,
    prereg: dict[str, Any],
    final_name: str = "final_summary.json",
) -> dict[str, Any]:
    root = module.OUT_ROOT
    final_path = root / "analysis" / final_name
    final = module.read_json(final_path)
    model_rows: dict[str, Any] = {}
    required = [
        root / "protocol" / "preregistration.json",
        root / "protocol" / "audit.json",
        final_path,
        root / "analysis" / "numeric_audit.json",
        root / "analysis" / "projection_audit.json",
        root / "analysis" / "automatic_next.json",
    ]

    for model in module.MODELS:
        summary_path = root / "atlas" / model / "summary.json"
        npz_path = root / "atlas" / model / "signed_fields.npz"
        projection_path = root / "atlas" / model / "projection_audit.json"
        required.extend([summary_path, npz_path, projection_path])
        summary = module.read_json(summary_path)
        digest_key = "summary_digest"
        checks = {
            "protocol_digest_matches": summary["protocol_digest"]
            == prereg["protocol_digest"],
            "case_digest_matches": summary["case_digest"]
            == prereg["model_case_digests"][model],
            "summary_digest_recomputes": digest_without(
                summary, digest_key, module
            ) == summary[digest_key],
            "hidden_states_finite": float(
                summary["hidden_finite_fraction_lower_bound"]
            ) >= float(
                prereg["evidence_thresholds"]["minimum_hidden_finite_fraction"]
            ),
            "prequery_zero": float(summary["pre_query_global_max_abs"])
            <= float(prereg["evidence_thresholds"]["pre_query_tolerance"]),
            "identity_repeat_zero": float(summary["identity_maximum"]) == 0.0,
            "npz_present": npz_path.exists() and npz_path.stat().st_size > 0,
            "projection_audit_present": projection_path.exists(),
            **precision_checks(summary),
        }
        expected_npz = final["models"][model].get("npz_sha256")
        if expected_npz is not None:
            checks["npz_hash_matches_final"] = sha256(npz_path) == expected_npz
        model_rows[model] = {
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "summary_digest": summary[digest_key],
            "npz_sha256": sha256(npz_path),
            "candidate_finite_fraction": summary["candidate_finite_fraction"],
            "hidden_finite_fraction": summary[
                "hidden_finite_fraction_lower_bound"
            ],
        }

    checks = {
        "all_required_files_present": all(path.exists() for path in required),
        "final_protocol_digest_matches": final["protocol_digest"]
        == prereg["protocol_digest"],
        "final_summary_digest_recomputes": digest_without(
            final, "summary_digest", module
        ) == final["summary_digest"],
        "all_model_artifacts_passed": all(
            row["all_checks_passed"] for row in model_rows.values()
        ),
    }
    return {
        "checks": checks,
        "models": model_rows,
        "final_decision": final["decision"],
        "final_summary_digest": final["summary_digest"],
        "all_checks_passed": all(checks.values()),
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in required
            if path.exists()
        ],
    }


def audit_behavior_phase(
    module: Any, prereg: dict[str, Any]
) -> dict[str, Any]:
    root = module.OUT_ROOT
    final_path = root / "analysis" / "final_summary.json"
    final = module.read_json(final_path)
    model_rows: dict[str, Any] = {}
    required = [
        root / "protocol" / "preregistration.json",
        root / "protocol" / "audit.json",
        final_path,
    ]

    for model in module.MODELS:
        summary_path = root / "pilot" / f"{model}.json"
        candidate_path = root / "pilot" / f"candidate.{model}.jsonl"
        generation_path = root / "pilot" / f"generation.{model}.jsonl"
        required.extend([summary_path, candidate_path, generation_path])
        summary = module.read_json(summary_path)
        candidates = module.read_jsonl(candidate_path)
        generations = module.read_jsonl(generation_path)
        checks = {
            "protocol_digest_matches": summary["protocol_digest"]
            == prereg["protocol_digest"],
            "case_digest_matches": summary["case_digest"]
            == prereg["model_case_digests"][model],
            "result_digest_recomputes": digest_without(
                summary, "result_digest", module
            ) == summary["result_digest"],
            "candidate_count_matches": len(candidates)
            == int(prereg["case_count_per_model"])
            == int(summary["candidate_case_count"]),
            "generation_count_matches": len(generations)
            == int(summary["generation_case_count"]),
            **precision_checks(summary),
        }
        model_rows[model] = {
            "checks": checks,
            "all_checks_passed": all(checks.values()),
            "result_digest": summary["result_digest"],
            "candidate_finite_fraction": summary["candidate_finite_fraction"],
        }

    checks = {
        "all_required_files_present": all(path.exists() for path in required),
        "final_protocol_digest_matches": final["protocol_digest"]
        == prereg["protocol_digest"],
        "final_summary_digest_recomputes": digest_without(
            final, "summary_digest", module
        ) == final["summary_digest"],
        "selected_routes_frozen": final["selected_routes_for_phase1091"]
        == ["en_en", "zh_zh", "en_zh", "zh_en"],
        "hidden_map_only_authorized": final["hidden_protocol_authorized"]
        and not final["causal_authorized"],
        "all_model_artifacts_passed": all(
            row["all_checks_passed"] for row in model_rows.values()
        ),
    }
    return {
        "checks": checks,
        "models": model_rows,
        "final_decision": final["decision"],
        "final_summary_digest": final["summary_digest"],
        "all_checks_passed": all(checks.values()),
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in required
            if path.exists()
        ],
    }


def main() -> None:
    prereg1089, protocol1089 = audit_protocol(p1089)
    prereg1090, protocol1090 = audit_protocol(p1090)
    prereg1091, protocol1091 = audit_protocol(p1091)
    phase1089 = audit_signed_phase(p1089, prereg1089)
    phase1090 = audit_behavior_phase(p1090, prereg1090)
    phase1091 = audit_signed_phase(p1091, prereg1091)

    auto1089 = p1089.read_json(
        p1089.OUT_ROOT / "analysis" / "automatic_next.json"
    )
    auto1091 = p1091.read_json(
        p1091.OUT_ROOT / "analysis" / "automatic_next.json"
    )
    chain_checks = {
        "phase1089_protocol_passed": protocol1089["all_checks_passed"],
        "phase1089_artifacts_passed": phase1089["all_checks_passed"],
        "phase1090_protocol_passed": protocol1090["all_checks_passed"],
        "phase1090_artifacts_passed": phase1090["all_checks_passed"],
        "phase1091_protocol_passed": protocol1091["all_checks_passed"],
        "phase1091_artifacts_passed": phase1091["all_checks_passed"],
        "phase1091_consumes_phase1090_summary": prereg1091[
            "source_phase1090_summary_digest"
        ] == phase1090["final_summary_digest"],
        "phase1091_routes_match_phase1090": prereg1091["surface_routes"]
        == ["en_en", "zh_zh", "en_zh", "zh_en"],
        "phase1089_forbids_causal_escalation": not auto1089[
            "local_causal_authorized"
        ],
        "phase1091_forbids_causal_escalation": not auto1091[
            "local_causal_authorized"
        ],
        "phase1091_forbids_automatic_hidden_extension": not auto1091[
            "automatic_hidden_extension_authorized"
        ],
        "phase1091_semantic_surface_not_claimed": not auto1091[
            "semantic_surface_evidence"
        ],
    }
    all_checks = all(chain_checks.values())
    audit = {
        "schema_version": "phase1089_1091_result_integrity_audit.v1",
        "phases": [1089, 1090, 1091],
        "protocols": {
            "phase1089": protocol1089,
            "phase1090": protocol1090,
            "phase1091": protocol1091,
        },
        "artifacts": {
            "phase1089": phase1089,
            "phase1090": phase1090,
            "phase1091": phase1091,
        },
        "chain_checks": chain_checks,
        "scientific_scope": {
            "retained": "model-internal lexical/context-conditioned color-pair map",
            "not_established": [
                "cross-surface color semantic geometry",
                "cross-model conserved physical coordinates",
                "causal components",
                "causal neurons",
            ],
        },
        "all_checks_passed": all_checks,
    }
    audit["audit_digest"] = p1091.digest(audit)
    output = p1091.OUT_ROOT / "analysis" / "result_integrity_audit.json"
    p1091.write_json(output, audit)
    if not all_checks:
        raise RuntimeError(json.dumps(chain_checks, indent=2))
    print({
        "status": "audit_passed",
        "phases": audit["phases"],
        "automatic_hidden_extension": False,
        "causal_localization": False,
        "audit_digest": audit["audit_digest"],
    })


if __name__ == "__main__":
    main()
