"""Independent audit for Phase1263 C013 WP01."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MAIN_PATH = ROOT / "tests/glm5/phase1263_c013_predictive_equivalence_independent_rescue_calibration.py"
SPEC = importlib.util.spec_from_file_location("phase1263_main", MAIN_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Phase1263 main module")
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
    checks = {
        "protocol_digest": protocol.get("protocol_digest") == expected.get("protocol_digest"),
        "source_hashes": protocol.get("source_hashes") == expected.get("source_hashes"),
        "public_digest": protocol.get("public_digest") == main.digest(public),
        "truth_digest": protocol.get("truth_digest") == main.digest(truth),
        "system_count": len(public) == len(main.REGISTRY_SPLITS) * len(main.SOURCE_FAMILIES) * main.TASKS * main.REPLICATES,
        "unique_and_aligned_ids": len(public_ids) == len(public) and public_ids == truth_ids,
        "private_truth_absent_public": all("source_family" not in row and "predictive_class" not in row and "seed" not in row for row in public),
        "candidate_order_frozen": protocol.get("candidate_order") == list(main.CANDIDATES) + ["abstain"],
        "reference_definition_frozen": protocol.get("reference_thresholds") == {"pass_max": main.REFERENCE_PASS_MAX, "earlier_fail_min": main.REFERENCE_EARLIER_FAIL_MIN},
        "thresholds_frozen": protocol.get("thresholds") == main.THRESHOLDS,
        "all_partitions_present": protocol.get("partitions") == main.PARTITION_COUNTS,
        "abstention_present": protocol.get("class_counts", {}).get("abstain", 0) >= main.THRESHOLDS["minimum_class_count"],
        "actionable_classes_present": all(protocol.get("class_counts", {}).get(name, 0) >= main.THRESHOLDS["minimum_class_count"] for name in main.CANDIDATES),
        "independent_rescue_registered": "donor_discovery" in protocol.get("rescue_provenance", {}).get("correct_rescue", ""),
        "replay_excluded": "excluded" in protocol.get("rescue_provenance", {}).get("algebraic_replay_sentinel", ""),
        "free_network_conditional": any("failure blocks free-Transformer" in item for item in protocol.get("hard_stops", [])),
        "qwen_denied": "qwen3" in protocol.get("forbidden_claims", []),
        "memo_not_in_instrument": all(not path.endswith(".md") for path in protocol.get("source_hashes", {})),
    }
    return {"mode": "pre", "created_at_utc": utc_now(), "checks": checks, "passed_checks": sum(checks.values()), "total_checks": len(checks), "all_checks_passed": all(checks.values())}


def final_audit() -> dict[str, Any]:
    protocol = main.read_json(main.PROTOCOL)
    rows = main.read_jsonl(main.RAW)
    truth = {row["system_id"]: row for row in main.read_jsonl(main.TRUTH)}
    summary = main.read_json(main.SUMMARY)
    final = main.read_json(main.FINAL)
    expected_digest = final.pop("final_digest")
    type_accuracy = sum(row["selected_class"] == truth[row["system_id"]]["predictive_class"] for row in rows) / len(rows)
    abstain_rows = [row for row in rows if truth[row["system_id"]]["predictive_class"] == "abstain"]
    abstention_accuracy = sum(row["selected_class"] == "abstain" for row in abstain_rows) / len(abstain_rows)
    expected_gate_set = {"G-EQUIVALENCE-CLASS", "G-ABSTENTION", "G-CLASS-BREADTH", "G-CROSS-SOURCE-ALIAS", "G-CONFIRMATION", "G-INDEPENDENT-RESCUE", "G-ALGEBRAIC-SENTINEL", "G-WRONG-IDENTITY", "G-CONTROLS", "G-SPLIT-BREADTH"}
    checks = {
        "formal_marker": main.COMPLETE.exists() and main.read_json(main.COMPLETE).get("status") == "formal_run_complete",
        "system_count": len(rows) == protocol.get("systems"),
        "raw_digest": main.digest(rows) == summary.get("raw_digest"),
        "all_ids_known": {row["system_id"] for row in rows} == set(truth),
        "type_accuracy_recomputed": abs(type_accuracy - final.get("type_accuracy", -1)) < 1.0e-12,
        "abstention_recomputed": abs(abstention_accuracy - final.get("abstention_accuracy", -1)) < 1.0e-12,
        "abstention_has_no_confirmation": all(row["confirmation"] is None for row in rows if row["selected_class"] == "abstain"),
        "actionable_has_confirmation": all(row["confirmation"] is not None for row in rows if row["selected_class"] != "abstain"),
        "confirmation_flags_recomputed": all(row["confirmation_passed"] == (main.confirmation_passes(row["confirmation"]) if row["confirmation"] is not None else True) for row in rows),
        "replay_is_only_sentinel": "zero mediation evidence" in final.get("claim_boundary", ""),
        "donor_gap_nonzero": all(row["confirmation"] is None or row["confirmation"]["donor_source_gap"] >= main.THRESHOLDS["donor_source_gap_min"] for row in rows),
        "registered_gate_set": set(final.get("gates", {})) == expected_gate_set,
        "verdict_matches_gates": final.get("passed") == all(final.get("gates", {}).values()),
        "authorization_matches": final.get("authorization", {}).get("free_transformer_predictive_equivalence") == final.get("passed") and not final.get("authorization", {}).get("qwen3"),
        "final_digest": main.digest(final) == expected_digest,
        "scope_narrow": "Known-truth" in final.get("claim_boundary", "") and "no free-network" in final.get("claim_boundary", ""),
    }
    return {"mode": "final", "created_at_utc": utc_now(), "checks": checks, "passed_checks": sum(checks.values()), "total_checks": len(checks), "all_checks_passed": all(checks.values()), "recomputed": {"type_accuracy": type_accuracy, "abstention_accuracy": abstention_accuracy}}


def write(path: Path, value: Any) -> None:
    main.atomic_json(path, value)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pre", "final"), required=True)
    args = parser.parse_args()
    result = preaudit() if args.mode == "pre" else final_audit()
    write(main.PREAUDIT if args.mode == "pre" else main.FINAL_AUDIT, result)
    print(json.dumps({"mode": args.mode, "checks": f"{result['passed_checks']}/{result['total_checks']}", "passed": result["all_checks_passed"]}, separators=(",", ":")))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
