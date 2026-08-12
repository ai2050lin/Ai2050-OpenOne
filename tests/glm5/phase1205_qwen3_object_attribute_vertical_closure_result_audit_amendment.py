#!/usr/bin/env python3
"""Narrow result-audit amendment for the Phase1205 placement schema.

The frozen independent audit checked a nonexistent boolean placement field.
This amendment is allowed to repair only that one engineering check.  It first
requires every other independently recomputed check to pass, then validates the
actual placement ledger emitted by the frozen FP16 loader.  It does not alter
the protocol, arrays, thresholds, selected depth, or scientific verdict.
"""

from __future__ import annotations

import argparse
import json

import phase1205_qwen3_object_attribute_vertical_closure_audit as base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write and base.RESULT_AUDIT_PATH.exists():
        raise RuntimeError("Phase1205 independent result audit already exists")

    output = base.result_audit(False)
    failed = [item["name"] for item in output["checks"] if not item["pass"]]
    if failed != ["full_cuda"]:
        raise RuntimeError(f"amendment scope exceeded: {failed}")

    summary = base.read_json(base.RUN_SUMMARY_PATH)
    base.validate_embedded_digest(summary, "summary_digest")
    placement = summary["placement"]
    placement_pass = bool(
        placement.get("placement") == "full_cuda"
        and placement.get("quantization") == "none"
        and set(placement.get("parameter_dtypes", {})) == {"float16"}
        and int(placement["parameter_dtypes"]["float16"]) > 0
    )
    for item in output["checks"]:
        if item["name"] == "full_cuda":
            item["pass"] = placement_pass
            item["detail"] = {
                "amended_check": (
                    "placement == full_cuda, quantization == none, and all recorded parameters are float16"
                ),
                "observed": placement,
            }

    output["audit_schema_amendment"] = {
        "scope": "full_cuda check only",
        "reason": (
            "The frozen audit expected all_parameters_on_cuda, but the frozen run ledger records the "
            "same qualification as placement='full_cuda' plus dtype and quantization ledgers."
        ),
        "protocol_changed": False,
        "hidden_arrays_changed": False,
        "thresholds_changed": False,
        "selected_depth_changed": False,
        "scientific_recomputation_changed": False,
        "base_audit_script_sha256": base.sha256_file(base.AUDIT_SCRIPT),
    }
    output["passed_checks"] = sum(item["pass"] for item in output["checks"])
    output["total_checks"] = len(output["checks"])
    output["gate_pass"] = all(item["pass"] for item in output["checks"])
    output.pop("audit_digest", None)
    output["audit_digest"] = base.digest(output)
    if not output["gate_pass"]:
        raise RuntimeError("Phase1205 amended result audit still failed")
    if args.write:
        base.write_json(base.RESULT_AUDIT_PATH, output)
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
