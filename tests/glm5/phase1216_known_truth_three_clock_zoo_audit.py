#!/usr/bin/env python3
"""Independent audit for Phase1216 known-truth clock calibration."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MAIN_SCRIPT = ROOT / "tests/glm5/phase1216_known_truth_three_clock_zoo.py"
AUDIT_SCRIPT = Path(__file__).resolve()
OUT_ROOT = ROOT / "tests/glm5/result/phase1216_known_truth_three_clock_zoo"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
SUMMARY_PATH = OUT_ROOT / "analysis/summary.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"
PHASE1215_FINAL = ROOT / "tests/glm5/result/phase1215_mechanism_hypothesis_elimination_engine/analysis/final.json"
CLOCKS = ["R", "C", "D", "E", "U1", "UJ"]


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_onset(gates: list[bool], steps: list[int], consecutive_count: int, tail_min: float) -> int | None:
    for start in range(len(gates) - consecutive_count + 1):
        consecutive = all(gates[start : start + consecutive_count])
        tail_fraction = sum(gates[start:]) / len(gates[start:])
        if consecutive and gates[-1] and tail_fraction >= tail_min:
            return steps[start]
    return None


def infer_clocks(trace: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, int | None]:
    rows = sorted(trace, key=lambda row: row["checkpoint_index"])
    thresholds = protocol["thresholds"]
    gates = {
        "R": [row["metrics"]["rule_accuracy"] >= thresholds["rule_accuracy"] for row in rows],
        "C": [row["metrics"]["min_correct_probability"] >= thresholds["min_correct_probability"] for row in rows],
        "D": [row["metrics"]["decode_accuracy"] >= thresholds["decode_accuracy"] for row in rows],
        "E": [
            row["metrics"]["decode_accuracy"] >= thresholds["decode_accuracy"]
            and row["metrics"]["transfer_success"] >= thresholds["transfer_success"]
            and row["metrics"]["preservation_success"] >= thresholds["preservation_success"]
            for row in rows
        ],
        "U1": [row["metrics"]["single_necessity"] >= thresholds["single_necessity"] for row in rows],
        "UJ": [row["metrics"]["joint_necessity"] >= thresholds["joint_necessity"] for row in rows],
    }
    consecutive = int(protocol["stable_onset"]["consecutive"])
    tail_min = float(protocol["stable_onset"]["tail_fraction"])
    return {
        clock: stable_onset(gates[clock], protocol["checkpoint_steps"], consecutive, tail_min)
        for clock in CLOCKS
    }


def signature(clocks: dict[str, int | None]) -> str:
    finite = sorted({value for value in clocks.values() if value is not None})
    ranks = {value: index for index, value in enumerate(finite)}
    return "|".join(
        f"{clock}:{'X' if clocks[clock] is None else ranks[clocks[clock]]}" for clock in CLOCKS
    )


def public_trace_digest(trace: list[dict[str, Any]]) -> str:
    rows = sorted(trace, key=lambda row: row["checkpoint_index"])
    public = [
        {
            "checkpoint_index": row["checkpoint_index"],
            "step": row["step"],
            "metrics": row["metrics"],
            "zero_drift": row["zero_drift"],
        }
        for row in rows
    ]
    return digest(public)


def main() -> None:
    protocol = read_json(PROTOCOL_PATH)
    summary = read_json(SUMMARY_PATH)
    upstream = read_json(PHASE1215_FINAL)
    checks: list[dict[str, Any]] = []

    def check(identifier: str, condition: bool, detail: Any = None) -> None:
        checks.append({"id": identifier, "pass": bool(condition), "detail": detail})

    candidate = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    check("protocol_digest", digest(candidate) == protocol["protocol_digest"])
    check("main_source_hash", protocol["source_hashes"]["main"] == file_sha256(MAIN_SCRIPT))
    check("audit_source_hash", protocol["source_hashes"]["audit"] == file_sha256(AUDIT_SCRIPT))
    check("upstream_hash", protocol["upstream"]["phase1215_final_sha256"] == file_sha256(PHASE1215_FINAL))
    check("upstream_digest", protocol["upstream"]["phase1215_final_digest"] == upstream["final_digest"])
    check("upstream_selection", protocol["upstream"]["selected_experiment"] == "T01_KNOWN_TRUTH_THREE_CLOCK_ZOO")
    check("known_truth_scope", protocol["scope"]["known_truth_only"] is True)
    check("no_trained_network", protocol["scope"]["trained_neural_network"] is False)
    check("no_pretrained_model", protocol["scope"]["pretrained_language_model"] is False)
    check("clock_set", set(protocol["clocks"]) == set(CLOCKS))
    check("archetype_count", len(protocol["archetypes"]) == 12)
    check("checkpoint_grid", protocol["checkpoint_steps"] == list(range(0, 2500, 100)))

    recomputed_summaries = {}
    all_outputs_valid = True
    for split in ("discovery", "confirmation"):
        manifest = summary["manifests"][split]
        systems_path = ROOT / manifest["systems_path"]
        traces_path = ROOT / manifest["traces_path"]
        check(f"{split}_systems_hash", file_sha256(systems_path) == manifest["systems_sha256"])
        check(f"{split}_traces_hash", file_sha256(traces_path) == manifest["traces_sha256"])
        systems = read_jsonl(systems_path)
        traces = read_jsonl(traces_path)
        check(f"{split}_system_count", len(systems) == 12 * 48)
        check(f"{split}_trace_count", len(traces) == 12 * 48 * 25)

        by_system: dict[str, list[dict[str, Any]]] = {}
        for row in traces:
            by_system.setdefault(row["system_id"], []).append(row)
        check(f"{split}_system_trace_keys", set(by_system) == {row["system_id"] for row in systems})
        finite = all(
            math.isfinite(float(value))
            for row in traces
            for value in list(row["metrics"].values()) + [row["zero_drift"]]
        )
        zero_drift = all(row["zero_drift"] == 0.0 for row in traces)
        check(f"{split}_all_finite", finite)
        check(f"{split}_zero_drift", zero_drift)

        clock_cells = 0
        exact_clock_cells = 0
        exact_systems = 0
        exact_signatures = 0
        per_archetype: dict[str, dict[str, Any]] = {}
        twin_groups: dict[str, list[dict[str, Any]]] = {}
        for system in systems:
            trace = by_system[system["system_id"]]
            check(f"trace_len_{system['system_id']}", len(trace) == 25)
            inferred = infer_clocks(trace, protocol)
            inferred_signature = signature(inferred)
            trace_digest = public_trace_digest(trace)
            check(f"clock_recompute_{system['system_id']}", inferred == system["inferred_clocks"])
            check(f"signature_recompute_{system['system_id']}", inferred_signature == system["inferred_signature"])
            check(f"trace_digest_{system['system_id']}", trace_digest == system["observable_trace_digest"])
            exact = inferred == system["truth_clocks"]
            sig_exact = inferred_signature == system["truth_signature"]
            exact_systems += int(exact)
            exact_signatures += int(sig_exact)
            for clock in CLOCKS:
                clock_cells += 1
                exact_clock_cells += int(inferred[clock] == system["truth_clocks"][clock])
            stats = per_archetype.setdefault(system["archetype"], {"count": 0, "clock_exact": 0, "signature_exact": 0})
            stats["count"] += 1
            stats["clock_exact"] += int(exact)
            stats["signature_exact"] += int(sig_exact)
            if system["twin_id"] is not None:
                twin_groups.setdefault(system["twin_id"], []).append(system)

        twin_pass = []
        for twin_id, rows in twin_groups.items():
            passed = (
                len(rows) == 2
                and rows[0]["observable_trace_digest"] == rows[1]["observable_trace_digest"]
                and rows[0]["latent_truth"]["censor_cause"] != rows[1]["latent_truth"]["censor_cause"]
                and all(row["censor_cause_decision"] == "UNIDENTIFIABLE" for row in rows)
            )
            twin_pass.append(passed)
            check(f"twin_{twin_id}", passed)
        check(f"{split}_twin_count", len(twin_pass) == 48)

        recomputed_per_archetype = {
            archetype: {
                "count": values["count"],
                "clock_exact_fraction": values["clock_exact"] / values["count"],
                "signature_exact_fraction": values["signature_exact"] / values["count"],
            }
            for archetype, values in per_archetype.items()
        }
        gates = {
            "clock_exact": exact_systems == len(systems),
            "signature_exact": exact_signatures == len(systems),
            "all_archetypes_exact": all(
                values["clock_exact_fraction"] >= 0.99
                and values["signature_exact_fraction"] >= 0.99
                for values in recomputed_per_archetype.values()
            ),
            "censor_twins_unidentifiable": all(twin_pass) and len(twin_pass) == 48,
            "all_finite": finite,
            "zero_drift": zero_drift,
        }
        recomputed = {
            "system_count": len(systems),
            "trace_row_count": len(traces),
            "checkpoint_count_per_system": 25,
            "clock_cell_count": clock_cells,
            "clock_cell_exact_count": exact_clock_cells,
            "clock_cell_exact_fraction": exact_clock_cells / clock_cells,
            "signature_exact_fraction": exact_signatures / len(systems),
            "censor_twin_pair_count": len(twin_pass),
            "censor_twin_pass_count": sum(twin_pass),
            "per_archetype": recomputed_per_archetype,
            "gates": gates,
            "overall_pass": all(gates.values()),
        }
        recomputed_summaries[split] = recomputed
        check(f"{split}_summary_exact", recomputed == summary["summaries"][split])
        all_outputs_valid = all_outputs_valid and recomputed["overall_pass"]

    check("both_splits_pass", all_outputs_valid and summary["overall_pass"] is True)
    check("status", summary["status"] == "known_truth_three_clock_calibration_passed")
    check("three_clock_claim", summary["claims"]["three_clock_construct_calibrated"] is True)
    check("read_use_claim", summary["claims"]["readability_use_separation_calibrated"] is True)
    check("censor_logic_claim", summary["claims"]["censoring_clock_logic_calibrated"] is True)
    check("censor_cause_abstention", summary["claims"]["censor_cause_identifiable_from_public_trace"] is False)
    check("free_external_not_tested", summary["claims"]["free_network_external_validity"] == "not_tested")
    check("pretrained_external_not_tested", summary["claims"]["pretrained_language_external_validity"] == "not_tested")

    gate_pass = all(row["pass"] for row in checks)
    audit = {
        "phase": 1216,
        "created_at": utc_now(),
        "gate_pass": gate_pass,
        "checks_passed": sum(row["pass"] for row in checks),
        "checks_total": len(checks),
        "checks": checks,
        "scope": "known-truth clock and abstention calibration only",
    }
    audit["audit_digest"] = digest(audit)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical_json({"gate_pass": gate_pass, "checks_passed": audit["checks_passed"], "checks_total": audit["checks_total"], "audit_digest": audit["audit_digest"]}))
    if not gate_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
