#!/usr/bin/env python3
"""Independent audit for Phase1215's typed elimination engine."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MAIN_SCRIPT = ROOT / "tests/glm5/phase1215_mechanism_hypothesis_elimination_engine.py"
AUDIT_SCRIPT = Path(__file__).resolve()
OUT_ROOT = ROOT / "tests/glm5/result/phase1215_mechanism_hypothesis_elimination_engine"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
EVIDENCE_PATH = OUT_ROOT / "analysis/evidence_bundles.json"
REGISTRY_PATH = OUT_ROOT / "analysis/hypothesis_registry.json"
MATRIX_PATH = OUT_ROOT / "analysis/falsification_matrix.json"
FAILURE_LEDGER_PATH = OUT_ROOT / "analysis/failure_type_ledger.json"
SELECTOR_PATH = OUT_ROOT / "analysis/experiment_selector.json"
SUMMARY_PATH = OUT_ROOT / "analysis/summary.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"
PHASE1214_FINAL = ROOT / "tests/glm5/result/phase1214_functional_event_formation_dynamics/analysis/final.json"

CURRENT_CAPABILITIES = {
    "typed_evidence_compiler",
    "known_truth_causal_use_camera",
    "free_transformer_local_transition",
    "right_censoring_ledger",
}
OPEN_MEASUREMENT_GAPS = {
    "three_clock_construct",
    "readability_use_separation_over_time",
    "censoring_clock_validation",
}


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expected_status(hypothesis: dict[str, Any]) -> str:
    predictions = hypothesis["predictions"]
    fatal = any(
        item["fatal_if_decisively_contradicted"]
        and item["verdict"] == "contradicted"
        and item["strength"] == "decisive"
        for item in predictions
    )
    if fatal:
        return "CLOSED_STRONG_FORM"
    if any(item["verdict"] == "supported" for item in predictions):
        return "ACTIVE_CONSTRAINED"
    if any(item["verdict"] == "unidentifiable" for item in predictions):
        return "UNIDENTIFIABLE"
    return "OPEN_UNTESTED"


def main() -> None:
    protocol = read_json(PROTOCOL_PATH)
    evidence = read_json(EVIDENCE_PATH)
    registry_payload = read_json(REGISTRY_PATH)
    matrix = read_json(MATRIX_PATH)
    failure_ledger = read_json(FAILURE_LEDGER_PATH)
    selector = read_json(SELECTOR_PATH)
    summary = read_json(SUMMARY_PATH)
    upstream = read_json(PHASE1214_FINAL)

    checks: list[dict[str, Any]] = []

    def check(identifier: str, condition: bool, detail: Any = None) -> None:
        checks.append({"id": identifier, "pass": bool(condition), "detail": detail})

    candidate = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    check("protocol_digest", digest(candidate) == protocol["protocol_digest"])
    check("main_source_hash", protocol["source_hashes"]["main"] == file_sha256(MAIN_SCRIPT))
    check("audit_source_hash", protocol["source_hashes"]["audit"] == file_sha256(AUDIT_SCRIPT))
    check("upstream_sha256", protocol["upstream"]["phase1214_final_sha256"] == file_sha256(PHASE1214_FINAL))
    check("upstream_digest", protocol["upstream"]["phase1214_final_digest"] == upstream["final_digest"])
    check("upstream_stopped", protocol["upstream"]["phase1214_auto_continue"] is False)
    check("zero_model_scope", protocol["scope"]["new_model_run"] is False)
    check("zero_neural_data_scope", protocol["scope"]["new_neural_data"] is False)
    check("no_new_k_scope", protocol["scope"]["new_empirical_k_item"] is False)
    check("nonexclusive_logic", protocol["logic"]["hypotheses_mutually_exclusive"] is False)
    check("entropy_forbidden", protocol["logic"]["entropy_claim_forbidden"] is True)
    check("evidence_output_exact", evidence["bundles"] == protocol["evidence_bundles"])

    observed_k: list[int] = []
    for bundle in protocol["evidence_bundles"]:
        check(f"bundle_id_{bundle['id']}", isinstance(bundle["id"], str) and bundle["id"].startswith("E"))
        for reference in bundle.get("k_refs", []):
            valid = reference.startswith("K") and reference[1:].isdigit()
            check(f"valid_ref_{bundle['id']}_{reference}", valid)
            if valid:
                observed_k.append(int(reference[1:]))
    check("k_coverage_exact", sorted(observed_k) == list(range(1, 193)))
    check("k_coverage_unique", len(observed_k) == len(set(observed_k)) == 192)
    for index in range(1, 193):
        check(f"K{index}_covered_once", observed_k.count(index) == 1)
    check("phase1214_boundary_has_no_k", next(row for row in protocol["evidence_bundles"] if row["id"] == "E26")["k_refs"] == [])

    protocol_hypotheses = {row["id"]: row for row in protocol["hypotheses"]}
    output_hypotheses = {row["id"]: row for row in registry_payload["hypotheses"]}
    check("hypothesis_id_set", set(protocol_hypotheses) == set(output_hypotheses))
    expected_rows = []
    status_counts: dict[str, int] = {}
    unresolved: set[str] = set()
    for hypothesis_id, source in protocol_hypotheses.items():
        output = output_hypotheses[hypothesis_id]
        status = expected_status(source)
        check(f"status_{hypothesis_id}", output["status"] == status, {"expected": status, "observed": output["status"]})
        check(f"prediction_copy_{hypothesis_id}", output["predictions"] == source["predictions"])
        status_counts[status] = status_counts.get(status, 0) + 1
        for item in source["predictions"]:
            full_id = f"{hypothesis_id}.{item['id']}"
            if status != "CLOSED_STRONG_FORM" and item["verdict"] in {"untested", "unidentifiable"}:
                unresolved.add(full_id)
            expected_rows.append(
                {
                    "hypothesis": hypothesis_id,
                    "hypothesis_status": status,
                    "prediction": item["id"],
                    "statement": item["statement"],
                    "verdict": item["verdict"],
                    "strength": item["strength"],
                    "fatal": item["fatal_if_decisively_contradicted"],
                    "evidence_refs": item["evidence_refs"],
                    "evidence_bundles": item["evidence_bundles"],
                }
            )
    check("matrix_exact", matrix["rows"] == expected_rows)
    check("status_counts", summary["status_counts"] == status_counts, {"expected": status_counts, "observed": summary["status_counts"]})
    check("hypothesis_count", summary["hypothesis_count"] == len(protocol_hypotheses) == 18)
    check("closed_count", status_counts.get("CLOSED_STRONG_FORM") == 10)
    check("active_count", status_counts.get("ACTIVE_CONSTRAINED") == 6)
    check("unidentifiable_count", status_counts.get("UNIDENTIFIABLE") == 2)
    check("open_untested_count", status_counts.get("OPEN_UNTESTED", 0) == 0)
    check("no_posterior_claim", summary["probability_or_entropy_claimed"] is False)
    check("registry_relative_only", summary["ontology_complete_version_space_claimed"] is False)
    check("version_space_size", summary["registry_relative_version_space_size"] == 8)
    check("no_new_k_summary", summary["new_k_item"] is False)
    check("fixtures_all_pass", all(row["pass"] for row in summary["compiler_fixtures"]))
    check("fixture_count", len(summary["compiler_fixtures"]) == 6)

    failure_ids = [row["id"] for row in failure_ledger["types"]]
    check("failure_type_count", len(failure_ids) == len(set(failure_ids)) == 10)
    check("phase1214_failure_primary", failure_ledger["phase1214"]["primary_failure_types"] == ["F01_BEHAVIOR_OBJECT_NOT_FORMED", "F07_RIGHT_CENSORED"])
    check("phase1214_not_mechanism_failure", failure_ledger["phase1214"]["not_classified_as"] == "F05_MECHANISM_PREDICTION_CONTRADICTED")
    check("phase1214_measurement_warning", failure_ledger["phase1214"]["measurement_warning"] == "F08_MEASUREMENT_NOT_CALIBRATED")

    recomputed_experiments = []
    for experiment in protocol["candidate_experiments"]:
        prerequisites = set(experiment["prerequisites"])
        eligible = prerequisites.issubset(CURRENT_CAPABILITIES)
        prerequisite_gain = len(set(experiment["closes"]) & OPEN_MEASUREMENT_GAPS)
        targets = sorted(set(experiment["targets"]) & unresolved)
        rank_key = [
            1 if eligible and prerequisite_gain > 0 else 0,
            prerequisite_gain if eligible else -1,
            len(targets) if eligible else -1,
            -int(experiment["cost_units"]),
        ]
        recomputed_experiments.append(
            {
                **experiment,
                "eligible": eligible,
                "missing_prerequisites": sorted(prerequisites - CURRENT_CAPABILITIES),
                "prerequisite_gain": prerequisite_gain,
                "open_prediction_targets": targets,
                "guaranteed_resolution_count": len(targets),
                "rank_key": rank_key,
            }
        )
    expected_sorted = sorted(recomputed_experiments, key=lambda row: tuple(row["rank_key"]), reverse=True)
    check("selector_rows_exact", selector["experiments"] == expected_sorted)
    check("selector_nonprobabilistic", selector["probabilistic_information_gain_used"] is False)
    check("selected_experiment", selector["selected_experiment"] == "T01_KNOWN_TRUTH_THREE_CLOCK_ZOO")
    check("selected_auto", selector["selected_auto_executable"] is True)
    check("summary_selected", summary["selected_next_experiment"] == selector["selected_experiment"])
    check("summary_auto_candidate", summary["auto_continue_candidate"] is True)

    gate_pass = all(row["pass"] for row in checks)
    audit = {
        "phase": 1215,
        "created_at": utc_now(),
        "gate_pass": gate_pass,
        "checks_passed": sum(row["pass"] for row in checks),
        "checks_total": len(checks),
        "checks": checks,
        "scope": "compilation integrity only; expert semantic adjudications are not independently re-proven",
    }
    audit["audit_digest"] = digest(audit)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical_json({"gate_pass": gate_pass, "checks_passed": audit["checks_passed"], "checks_total": audit["checks_total"], "audit_digest": audit["audit_digest"]}))
    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
