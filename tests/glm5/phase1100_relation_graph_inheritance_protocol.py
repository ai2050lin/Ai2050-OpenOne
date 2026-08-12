#!/usr/bin/env python3
"""Freeze the Phase1100 lexical-inheritance and interface-boundary audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
PHASE = 1100
MODELS = ("qwen3", "glm4", "deepseek7b")
FORMAL_MODELS = ("qwen3", "glm4")
PRECISION = "fp16"
QUANTIZATION = "none"
SURFACES = ("en", "zh")
SAMPLE_SPLITS = ("discovery", "confirmation")
RELATION_SPLITS = ("discovery", "confirmation")
PRIMARY_SOURCE = "input_query_polarity"
ALTERNATIVE_SOURCE = "output_query_polarity"
FORM_SOURCE = "query_token_form"
PRIMARY_TARGET_FIELD = "relational_execution"
PRIMARY_TARGET_ROLE = "answer_boundary"
MATCHED_TARGET_CONTROLS = ("lookup_execution", "relational_carrier")
DIAGNOSTIC_TARGET_FIELDS = ("relational_representation", "lookup_representation")
SOURCE_ROOT = TEST_ROOT / "result" / "phase1099_relation_family_atlas"
OUT_ROOT = TEST_ROOT / "result" / "phase1100_relation_graph_inheritance"


THRESHOLDS = {
    "minimum_source_finite_fraction": 1.0,
    "minimum_graph_finite_fraction": 0.95,
    "minimum_inheritance_cosine": 0.50,
    "minimum_family_permutation_margin": 0.02,
    "minimum_within_family_permutation_margin": 0.02,
    "minimum_execution_specificity_advantage": 0.05,
    "minimum_confirmation_cells_per_surface": 2,
    "minimum_surface_passes_per_formal_model": 2,
    "minimum_cross_model_curve_cosine": 0.80,
    "maximum_cross_model_curve_mean_absolute_error": 0.15,
    "minimum_cross_model_curve_cells": 4,
}


GATES = {
    "P1": "Every Phase1099 source artifact and the frozen Phase1100 protocol pass digest, shape, and provenance audits.",
    "P2": "Signed lexical polarity vectors are extracted from actual query spans under sequential FP16, non-quantized loading for all three models.",
    "P3": "Discovery-only event selection yields a preregistered inheritance candidate for both surfaces in both formal models.",
    "P4": "The lexical graph predicts unseen-relation and independent-template execution graphs above both exact family and within-family label permutations in both surfaces of both formal models.",
    "P5": "The same execution inheritance exceeds lookup-execution, carrier, and token-form controls in both surfaces of both formal models.",
    "P6": "Qwen3 and GLM4 show a repeated functional inheritance trajectory in at least four of six surface-component cells.",
    "P7": "No output margin, generated score, learned probe, clustering label, component intervention, or post-hoc event selection enters the primary gates.",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_paths() -> list[Path]:
    paths = [
        SOURCE_ROOT / "protocol" / "preregistration.json",
        SOURCE_ROOT / "analysis" / "behavior_authorization.json",
        SOURCE_ROOT / "analysis" / "final_summary.json",
        SOURCE_ROOT / "analysis" / "failure_diagnostic.json",
        SOURCE_ROOT / "audit" / "result_audit.json",
    ]
    for model in MODELS:
        paths.extend(
            [
                SOURCE_ROOT / "protocol" / f"cases.{model}.jsonl",
                SOURCE_ROOT / "atlas" / model / "summary.json",
                SOURCE_ROOT / "atlas" / model / "superunit_index.jsonl",
                SOURCE_ROOT / "atlas" / model / "relative_relation_geometry.npz",
            ]
        )
    return paths


def build_preregistration() -> dict[str, Any]:
    source_audit = read_json(SOURCE_ROOT / "audit" / "result_audit.json")
    source_final = read_json(SOURCE_ROOT / "analysis" / "final_summary.json")
    payload = {
        "schema_version": "phase1100_preregistration.v1",
        "phase": PHASE,
        "objective": "Test whether the downstream relation-execution graph is specifically inherited from the signed lexical polarity geometry observed at the actual query interface, before naming an execution interface or a computation primitive.",
        "models": list(MODELS),
        "formal_models": list(FORMAL_MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "surfaces": list(SURFACES),
        "sample_splits": list(SAMPLE_SPLITS),
        "relation_splits": list(RELATION_SPLITS),
        "primary_source": PRIMARY_SOURCE,
        "alternative_source": ALTERNATIVE_SOURCE,
        "form_source": FORM_SOURCE,
        "primary_target_field": PRIMARY_TARGET_FIELD,
        "primary_target_role": PRIMARY_TARGET_ROLE,
        "matched_target_controls": list(MATCHED_TARGET_CONTROLS),
        "diagnostic_target_fields": list(DIAGNOSTIC_TARGET_FIELDS),
        "event_selection": "For each model and surface, select exactly one event using only discovery-sample/discovery-relation data by maximizing the minimum registered gate slack; freeze that event before evaluating the other three split cells.",
        "confirmation_cells": [
            {"sample_split": "confirmation", "relation_split": "discovery"},
            {"sample_split": "discovery", "relation_split": "confirmation"},
            {"sample_split": "confirmation", "relation_split": "confirmation"},
        ],
        "graph_metric": "Mean-center the strict upper triangle of each 15-relation Gram graph, unit-normalize it, and take its dot product.",
        "permutation_nulls": {
            "family": "All 5! family-block permutations while preserving within-family relation slots.",
            "within_family": "All (3!)^5 independent within-family permutations while preserving family blocks.",
        },
        "specificity_controls": [
            "input lexical graph to lookup-execution target",
            "input lexical graph to relational-carrier target",
            "query token-form graph to relational-execution target",
        ],
        "thresholds": THRESHOLDS,
        "gates": GATES,
        "automatic_next_rule": "Only P1-P7 jointly authorize a new interface or causal phase. Descriptive inheritance without specificity does not authorize it.",
        "source_phase": 1099,
        "source_phase_audit_digest": source_audit["audit_digest"],
        "source_phase_final_digest": source_final["summary_digest"],
        "source_files": {str(path.relative_to(ROOT)).replace("\\", "/"): file_digest(path) for path in source_paths()},
        "epistemic_constraints": [
            "Phase1099 did not locate semantic content at the embedding layer; lexical inheritance is a falsifiable hypothesis, not a premise.",
            "A lexical query-span embedding graph is a source proxy, not the full contextual input state.",
            "The inherited graph may be task or syntax structure; only specificity over matched controls can promote it.",
            "No unlabeled primitive, interface, or causal mechanism is named when a registered gate fails.",
        ],
    }
    payload["protocol_digest"] = digest(payload)
    return payload


def audit_preregistration(preregistration: dict[str, Any]) -> dict[str, Any]:
    source_audit = read_json(SOURCE_ROOT / "audit" / "result_audit.json")
    source_final = read_json(SOURCE_ROOT / "analysis" / "final_summary.json")
    source_authorization = read_json(SOURCE_ROOT / "analysis" / "behavior_authorization.json")
    checks = {
        "phase1099_audit_passed": bool(source_audit["all_checks_passed"]),
        "phase1099_automatic_stop_preserved": not bool(source_final["automatic_next_required"]),
        "phase1099_hidden_scan_authorized": bool(source_authorization["hidden_scan_authorized"]),
        "source_files_exist": all(path.exists() for path in source_paths()),
        "source_file_digests_match": all(
            preregistration["source_files"][str(path.relative_to(ROOT)).replace("\\", "/")] == file_digest(path)
            for path in source_paths()
        ),
        "formal_models_exact": tuple(preregistration["formal_models"]) == FORMAL_MODELS,
        "sequential_fp16_exact": preregistration["precision"] == "fp16" and preregistration["quantization"] == "none",
        "discovery_confirmation_separated": len(preregistration["confirmation_cells"]) == 3,
        "exact_permutation_nulls_registered": set(preregistration["permutation_nulls"]) == {"family", "within_family"},
        "controls_registered": len(preregistration["specificity_controls"]) == 3,
        "protocol_digest": preregistration["protocol_digest"] == digest({key: value for key, value in preregistration.items() if key != "protocol_digest"}),
    }
    result = {
        "schema_version": "phase1100_protocol_audit.v1",
        "phase": PHASE,
        "checks": checks,
        "passed_checks": sum(bool(value) for value in checks.values()),
        "check_count": len(checks),
        "all_checks_passed": all(checks.values()),
    }
    result["audit_digest"] = digest(result)
    return result


def main() -> None:
    preregistration = build_preregistration()
    audit = audit_preregistration(preregistration)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", preregistration)
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1100 protocol audit failed")
    print(json.dumps({"phase": PHASE, "protocol_digest": preregistration["protocol_digest"], "audit_digest": audit["audit_digest"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
