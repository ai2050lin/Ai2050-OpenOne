"""Independent audit for Phase1261."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MAIN_PATH = ROOT / "tests/glm5/phase1261_c012_factorial_response_compiler_calibration.py"
SPEC = importlib.util.spec_from_file_location("phase1261_main", MAIN_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Phase1261 main module")
main = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = main
SPEC.loader.exec_module(main)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value: Any) -> None:
    main.atomic_json(path, value)


def preaudit() -> dict[str, Any]:
    protocol = main.read_json(main.PROTOCOL)
    public = main.read_jsonl(main.PUBLIC)
    truth = main.read_jsonl(main.TRUTH)
    truth_ids = {row["system_id"] for row in truth}
    public_ids = {row["system_id"] for row in public}
    expected = main.protocol_payload(public, truth)
    checks = {
        "protocol_digest": protocol.get("protocol_digest") == expected.get("protocol_digest"),
        "source_hashes": protocol.get("source_hashes") == expected.get("source_hashes"),
        "public_digest": protocol.get("public_digest") == main.digest(public),
        "truth_digest": protocol.get("truth_digest") == main.digest(truth),
        "system_count": len(public) == len(main.REGISTRY_SPLITS) * len(main.FAMILIES) * main.TASKS * main.REPLICATES,
        "unique_ids": len(public_ids) == len(public),
        "truth_alignment": public_ids == truth_ids,
        "private_labels_absent_public": all("family" not in row and "expected_compiler" not in row and "seed" not in row for row in public),
        "compiler_order_frozen": protocol.get("compiler_order") == list(main.COMPILER_ORDER) + ["abstain"],
        "thresholds_frozen": protocol.get("thresholds") == main.THRESHOLDS,
        "three_partitions": protocol.get("partitions") == main.PARTITION_COUNTS,
        "hidden_collision_registered": main.EXPECTED_COMPILER.get("hidden_collision") == "abstain",
        "family_holdout_registered": set(protocol.get("registry_splits", [])) == set(main.REGISTRY_SPLITS),
        "qwen_denied": "qwen3" in protocol.get("forbidden_claims", []),
        "memo_not_part_of_instrument": all(not str(path).endswith(".md") for path in protocol.get("source_hashes", {})),
    }
    return {"mode": "pre", "created_at_utc": utc_now(), "checks": checks, "passed_checks": sum(checks.values()), "total_checks": len(checks), "all_checks_passed": all(checks.values())}


def final_audit() -> dict[str, Any]:
    protocol = main.read_json(main.PROTOCOL)
    rows = main.read_jsonl(main.RAW)
    truth = {row["system_id"]: row for row in main.read_jsonl(main.TRUTH)}
    summary = main.read_json(main.SUMMARY)
    final = main.read_json(main.FINAL)
    type_accuracy = sum(row["selected_compiler"] == truth[row["system_id"]]["expected_compiler"] for row in rows) / len(rows)
    abstain_rows = [row for row in rows if truth[row["system_id"]]["expected_compiler"] == "abstain"]
    abstention_accuracy = sum(row["selected_compiler"] == "abstain" for row in abstain_rows) / len(abstain_rows)
    expected_digest = final.pop("final_digest")
    earliest_selection_ok = all(
        row["selected_compiler"] == next((item["compiler"] for item in row.get("selection", []) if item.get("passed")), "abstain")
        for row in rows
    )
    checks = {
        "formal_marker": main.COMPLETE.exists() and main.read_json(main.COMPLETE).get("status") == "formal_run_complete",
        "system_count": len(rows) == protocol.get("systems"),
        "raw_digest": main.digest(rows) == summary.get("raw_digest"),
        "all_ids_known": {row["system_id"] for row in rows} == set(truth),
        "type_accuracy_recomputed": abs(type_accuracy - final.get("type_accuracy", -1)) < 1.0e-12,
        "abstention_recomputed": abs(abstention_accuracy - final.get("abstention_accuracy", -1)) < 1.0e-12,
        "candidate_order_respected": earliest_selection_ok,
        "confirmation_not_used_for_selection": all("confirmation" not in item for row in rows for item in row.get("selection", [])),
        "abstention_has_no_confirmation": all(row["confirmation"] is None for row in rows if row["selected_compiler"] == "abstain"),
        "actionable_has_confirmation": all(row["confirmation"] is not None for row in rows if row["selected_compiler"] != "abstain"),
        "confirmation_flags_recomputed": all(
            row["confirmation_passed"] == (main.confirmation_passes(row["confirmation"]) if row["confirmation"] is not None else True)
            for row in rows
        ),
        "registered_gate_set": set(final.get("gates", {})) == {"G-COMPILER-TYPE", "G-ABSTENTION", "G-A1-PREDICTION", "G-B-PRESERVATION", "G-J-PREDICTION", "G-WRONG", "G-NUISANCE", "G-PATH", "G-MANIFOLD", "G-FAMILY-HOLDOUT", "G-CONTROLS"},
        "verdict_matches_gates": final.get("passed") == all(final.get("gates", {}).values()),
        "authorization_matches": final.get("authorization", {}).get("free_transformer_factorial_compiler") == final.get("passed") and not final.get("authorization", {}).get("qwen3"),
        "artifact_hashes": all(main.file_sha256(getattr(main, key.upper() if key != "public" else "PUBLIC")) == value for key, value in final.get("artifact_hashes", {}).items() if key in {"protocol", "environment", "public", "truth", "preaudit", "raw", "summary", "complete", "analysis"}),
        "final_digest": main.digest(final) == expected_digest,
        "scope_narrow": "Known-truth" in final.get("claim_boundary", "") and "no free-network" in final.get("claim_boundary", ""),
    }
    return {"mode": "final", "created_at_utc": utc_now(), "checks": checks, "passed_checks": sum(checks.values()), "total_checks": len(checks), "all_checks_passed": all(checks.values()), "recomputed": {"type_accuracy": type_accuracy, "abstention_accuracy": abstention_accuracy}}


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pre", "final"), required=True)
    args = parser.parse_args()
    result = preaudit() if args.mode == "pre" else final_audit()
    path = main.PREAUDIT if args.mode == "pre" else main.FINAL_AUDIT
    write(path, result)
    print(json.dumps({"mode": args.mode, "checks": f"{result['passed_checks']}/{result['total_checks']}", "passed": result["all_checks_passed"]}, separators=(",", ":")))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main_cli()
