"""Independent audit for Phase1264 C014 WP01."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
MAIN_PATH = ROOT / "tests/glm5/phase1264_c014_population_risk_certificate_calibration.py"
SPEC = importlib.util.spec_from_file_location("phase1264_main", MAIN_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Phase1264 main module")
main = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = main
SPEC.loader.exec_module(main)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def preaudit() -> dict[str, Any]:
    protocol = main.read_json(main.PROTOCOL)
    public = main.read_jsonl(main.PUBLIC)
    truth = main.read_jsonl(main.TRUTH)
    expected = main.protocol_payload(public, truth)
    public_ids = {row["system_id"] for row in public}
    truth_ids = {row["system_id"] for row in truth}
    expected_radius = math.sqrt(
        math.log(2.0 * main.SYSTEM_COUNT * len(main.CANDIDATES) / main.GLOBAL_ERROR_BUDGET)
        / (2.0 * main.SELECTION_DRAWS)
    )
    boundary_rows = [row for row in truth if row["boundary_target"] is not None]
    target_errors = [
        abs(row["population_risks"][row["boundary_target"]["candidate"]] - row["boundary_target"]["risk"])
        for row in boundary_rows
    ]
    checks = {
        "protocol_digest": protocol.get("protocol_digest") == expected.get("protocol_digest"),
        "source_hashes": protocol.get("source_hashes") == expected.get("source_hashes"),
        "public_digest": protocol.get("public_digest") == main.digest(public),
        "truth_digest": protocol.get("truth_digest") == main.digest(truth),
        "system_count": len(public) == main.SYSTEM_COUNT == len(truth),
        "unique_and_aligned_ids": len(public_ids) == len(public) and public_ids == truth_ids,
        "private_truth_absent_public": all(
            "source_profile" not in row
            and "seed" not in row
            and "amplitude" not in row
            and "exact_class" not in row
            and "population_risks" not in row
            for row in public
        ),
        "finite_population_registered": protocol.get("registered_finite_population") == main.ORACLE_UNIVERSE_COUNT,
        "selection_draws_registered": protocol.get("selection_draws_with_replacement") == main.SELECTION_DRAWS,
        "candidate_order_frozen": protocol.get("candidate_order") == list(main.CANDIDATES) + ["abstain"],
        "thresholds_frozen": protocol.get("thresholds") == main.THRESHOLDS,
        "radius_recomputed": abs(protocol.get("certificate", {}).get("radius", -1.0) - expected_radius) < 1.0e-15,
        "bounded_loss_registered": protocol.get("certificate", {}).get("loss_range") == [0.0, 1.0],
        "global_union_registered": "all registered systems" in protocol.get("certificate", {}).get("bound", ""),
        "boundary_gradient_present": len(boundary_rows) == len(main.REGISTRY_SPLITS) * 2 * main.TASKS * main.REPLICATES,
        "boundary_targets_exact": max(target_errors) <= 1.0e-6,
        "robust_class_breadth": all(
            protocol.get("robust_class_counts", {}).get(name, 0) >= main.THRESHOLDS["minimum_robust_class_count"]
            for name in main.CANDIDATES
        ),
        "always_abstain_baseline": "always_abstain" in protocol.get("baselines", []),
        "two_level_authorization": protocol.get("rescue_provenance", {}).get("authorization", "").endswith("<= 0.025"),
        "approximate_not_quotient": "not asserted to be equivalence" in protocol.get("estimand", ""),
        "free_network_conditional": any("Failure blocks free-Transformer" in item for item in protocol.get("hard_stops", [])),
        "pretrained_denied": all(name in protocol.get("forbidden_claims", []) for name in ("qwen3", "natural-language mechanism")),
        "memo_not_in_instrument": all(not path.endswith(".md") for path in protocol.get("source_hashes", {})),
    }
    return {
        "mode": "pre",
        "created_at_utc": utc_now(),
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "all_checks_passed": all(checks.values()),
    }


def recompute_population_truth(public: list[dict[str, Any]], truth_rows: list[dict[str, Any]]) -> tuple[bool, float]:
    if not torch.cuda.is_available():
        return False, float("inf")
    public_by_id = {row["system_id"]: row for row in public}
    worst = 0.0
    for truth in truth_rows:
        public_row = public_by_id[truth["system_id"]]
        system = main.CertifiedSystem(
            str(truth["source_profile"]),
            int(truth["seed"]),
            int(public_row["task_id"]),
            torch.device("cuda"),
            float(truth["amplitude"]),
        )
        discovery = system.make_partition("discovery", main.DISCOVERY_COUNT)
        universe = system.make_partition("reference_a", main.ORACLE_UNIVERSE_COUNT)
        risks = main.mean_risks(main.population_losses(main.fit_models(discovery), universe))
        for name in main.CANDIDATES:
            worst = max(worst, abs(risks[name] - truth["population_risks"][name]))
        exact_class, reason = main.select_point(risks)
        if exact_class != truth["exact_class"] or reason != truth["truth_reason"]:
            return False, worst
    return worst <= 1.0e-12, worst


def final_audit() -> dict[str, Any]:
    protocol = main.read_json(main.PROTOCOL)
    public = main.read_jsonl(main.PUBLIC)
    truth_rows = main.read_jsonl(main.TRUTH)
    truth = {row["system_id"]: row for row in truth_rows}
    rows = main.read_jsonl(main.RAW)
    summary = main.read_json(main.SUMMARY)
    final = main.read_json(main.FINAL)
    final_without_digest = dict(final)
    expected_final_digest = final_without_digest.pop("final_digest")

    row_math = True
    for row in rows:
        expected_point, expected_point_reason = main.select_point(row["sample_risks"])
        expected_bounds = main.confidence_bounds(row["sample_risks"])
        expected_certificate, expected_certificate_reason = main.select_certificate(expected_bounds)
        expected_rescue = (
            expected_certificate != "abstain"
            and expected_bounds[expected_certificate]["upper"] <= main.THRESHOLDS["rescue_authorization_upper_max"]
        )
        row_math = row_math and (
            row["point_class"] == expected_point
            and row["point_reason"] == expected_point_reason
            and row["confidence_bounds"] == expected_bounds
            and row["certificate_class"] == expected_certificate
            and row["certificate_reason"] == expected_certificate_reason
            and row["rescue_authorized"] == expected_rescue
        )

    false_authorizations = sum(
        row["certificate_class"] != "abstain"
        and row["certificate_class"] != truth[row["system_id"]]["exact_class"]
        for row in rows
    )
    point_false = sum(
        row["point_class"] != "abstain" and row["point_class"] != truth[row["system_id"]]["exact_class"]
        for row in rows
    )
    robust_rows = [row for row in rows if truth[row["system_id"]]["robust_actionable"]]
    robust_coverage = sum(
        row["certificate_class"] == truth[row["system_id"]]["exact_class"] for row in robust_rows
    ) / max(1, len(robust_rows))
    ambiguous_rows = [row for row in rows if truth[row["system_id"]]["exact_class"] == "abstain"]
    ambiguous_abstention = sum(row["certificate_class"] == "abstain" for row in ambiguous_rows) / max(1, len(ambiguous_rows))
    population_ok, worst_population_delta = recompute_population_truth(public, truth_rows)
    expected_gate_set = {
        "G-EXACT-FINITE-POPULATION",
        "G-ZERO-FALSE-AUTHORIZATION",
        "G-ROBUST-COVERAGE",
        "G-AMBIGUOUS-ABSTENTION",
        "G-NONTRIVIAL-VS-ALWAYS-ABSTAIN",
        "G-BOUNDARY-GRADIENT",
        "G-ROBUST-CLASS-BREADTH",
        "G-SPLIT-BREADTH",
        "G-DUAL-DONOR-RESCUE",
        "G-NEGATIVE-CONTROLS",
    }
    checks = {
        "formal_marker": main.COMPLETE.exists() and main.read_json(main.COMPLETE).get("status") == "formal_run_complete",
        "system_count": len(rows) == protocol.get("systems") == main.SYSTEM_COUNT,
        "raw_digest": main.digest(rows) == summary.get("raw_digest") == final.get("raw_digest"),
        "all_ids_known": {row["system_id"] for row in rows} == set(truth),
        "population_truth_recomputed": population_ok,
        "row_certificate_math": row_math,
        "false_authorizations_recomputed": false_authorizations == final.get("certificate_false_authorizations"),
        "point_baseline_recomputed": point_false == final.get("point_false_authorizations"),
        "robust_coverage_recomputed": abs(robust_coverage - final.get("certificate_robust_coverage", -1.0)) < 1.0e-12,
        "ambiguous_abstention_recomputed": abs(ambiguous_abstention - final.get("ambiguous_abstention", -1.0)) < 1.0e-12,
        "rescue_scope_typed": all((row["confirmation"] is not None) == row["rescue_authorized"] for row in rows),
        "confirmation_flags_recomputed": all(
            row["confirmation_passed"]
            == (main.confirmation_passes(row["confirmation"]) if row["confirmation"] is not None else True)
            for row in rows
        ),
        "registered_gate_set": set(final.get("gates", {})) == expected_gate_set,
        "verdict_matches_gates": final.get("passed") == all(final.get("gates", {}).values()),
        "authorization_matches": final.get("authorization", {}).get("free_transformer_population_certificate") == final.get("passed"),
        "pretrained_not_authorized": not any(final.get("authorization", {}).get(name) for name in ("qwen3", "glm4", "ds7b")),
        "always_abstain_not_success": final.get("always_abstain_robust_coverage") == 0.0 and final.get("certificate_robust_coverage", 0.0) > 0.0,
        "final_digest": main.digest(final_without_digest) == expected_final_digest,
        "scope_narrow": "registered finite populations" in final.get("claim_boundary", "") and "not quotient" in final.get("claim_boundary", ""),
    }
    return {
        "mode": "final",
        "created_at_utc": utc_now(),
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "all_checks_passed": all(checks.values()),
        "recomputed": {
            "certificate_false_authorizations": false_authorizations,
            "point_false_authorizations": point_false,
            "robust_coverage": robust_coverage,
            "ambiguous_abstention": ambiguous_abstention,
            "worst_population_risk_delta": worst_population_delta,
        },
    }


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pre", "final"), required=True)
    args = parser.parse_args()
    result = preaudit() if args.mode == "pre" else final_audit()
    main.atomic_json(main.PREAUDIT if args.mode == "pre" else main.FINAL_AUDIT, result)
    print(json.dumps({"mode": args.mode, "checks": f"{result['passed_checks']}/{result['total_checks']}", "passed": result["all_checks_passed"]}, separators=(",", ":")))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
