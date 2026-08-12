#!/usr/bin/env python3
"""Phase1216: known-truth calibration for rule, calibration, and interface clocks.

The core clocks are tau_R (stable top-1 rule), tau_C (stable high-confidence
calibration), and tau_E (stable predictive-transfer interface).  Auxiliary
clocks tau_D, tau_U1, and tau_UJ separate decodability, single-path use, and
joint-path use.  Scripted known-truth trajectories include isolated false
positives, right censoring, readable-but-unused states, direct behavior,
interface-first and behavior-first orders, and redundant paths.

This phase calibrates measurement semantics only.  It does not establish that
free or pretrained neural networks use any particular archetype.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
AUDIT_SCRIPT = ROOT / "tests/glm5/phase1216_known_truth_three_clock_zoo_audit.py"
OUT_ROOT = ROOT / "tests/glm5/result/phase1216_known_truth_three_clock_zoo"
PROTOCOL_PATH = OUT_ROOT / "protocol/preregistration.json"
SUMMARY_PATH = OUT_ROOT / "analysis/summary.json"
AUDIT_PATH = OUT_ROOT / "audit/independent_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"
PHASE1215_FINAL = ROOT / "tests/glm5/result/phase1215_mechanism_hypothesis_elimination_engine/analysis/final.json"
PHASE = 1216

CHECKPOINT_STEPS = list(range(0, 2500, 100))
CLOCKS = ["R", "C", "D", "E", "U1", "UJ"]
REPLICATES_PER_ARCHETYPE = 48
STABLE_CONSECUTIVE = 2
STABLE_TAIL_FRACTION = 0.80
THRESHOLDS = {
    "rule_accuracy": 0.99,
    "min_correct_probability": 0.95,
    "decode_accuracy": 0.95,
    "transfer_success": 0.90,
    "preservation_success": 0.90,
    "single_necessity": 0.15,
    "joint_necessity": 0.15,
}
ARCHETYPES = [
    "simultaneous",
    "confidence_lag",
    "interface_first",
    "rule_first",
    "readable_unused",
    "behavior_without_interface",
    "interface_without_qualified_behavior",
    "redundant_joint_necessary",
    "use_lag",
    "decode_before_transfer",
    "transient_false_positive",
    "right_censored_late",
]
SPLIT_SEEDS = {"discovery": 12160017, "confirmation": 12160119}


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def source_hashes() -> dict[str, str]:
    return {"main": file_sha256(SCRIPT), "audit": file_sha256(AUDIT_SCRIPT)}


def output_paths(split: str) -> tuple[Path, Path]:
    root = OUT_ROOT / "runs" / split
    return root / "systems.jsonl", root / "traces.jsonl"


def schedule_for(archetype: str, base: int) -> tuple[dict[str, int | None], dict[str, Any]]:
    none = None
    if archetype == "simultaneous":
        clocks = {clock: base for clock in CLOCKS}
    elif archetype == "confidence_lag":
        clocks = {"R": base, "C": base + 3, "D": base, "E": base, "U1": base, "UJ": base}
    elif archetype == "interface_first":
        clocks = {"R": base + 4, "C": base + 6, "D": base, "E": base + 1, "U1": base + 4, "UJ": base + 4}
    elif archetype == "rule_first":
        clocks = {"R": base, "C": base + 1, "D": base + 4, "E": base + 5, "U1": base + 5, "UJ": base + 5}
    elif archetype == "readable_unused":
        clocks = {"R": base + 2, "C": base + 4, "D": base, "E": none, "U1": none, "UJ": none}
    elif archetype == "behavior_without_interface":
        clocks = {"R": base, "C": base + 2, "D": none, "E": none, "U1": none, "UJ": none}
    elif archetype == "interface_without_qualified_behavior":
        clocks = {"R": none, "C": none, "D": base, "E": base + 1, "U1": base + 1, "UJ": base + 1}
    elif archetype == "redundant_joint_necessary":
        clocks = {"R": base, "C": base + 2, "D": base, "E": base, "U1": none, "UJ": base}
    elif archetype == "use_lag":
        clocks = {"R": base, "C": base + 2, "D": base, "E": base, "U1": base + 5, "UJ": base + 5}
    elif archetype == "decode_before_transfer":
        clocks = {"R": base + 2, "C": base + 4, "D": base, "E": base + 5, "U1": base + 5, "UJ": base + 5}
    elif archetype == "transient_false_positive":
        clocks = {clock: none for clock in CLOCKS}
    elif archetype == "right_censored_late":
        clocks = {clock: none for clock in CLOCKS}
    else:
        raise ValueError(archetype)
    latent = {
        "censor_cause": (
            "transient_only"
            if archetype == "transient_false_positive"
            else "forms_after_horizon"
            if archetype == "right_censored_late"
            else "not_censored_archetype"
        ),
        "latent_after_horizon_index": base + 22 if archetype == "right_censored_late" else None,
    }
    return clocks, latent


def isolated_flickers(clock_index: int | None, observable_seed: int) -> set[int]:
    rng = random.Random(observable_seed + 7919)
    candidates = [2, 5, 8]
    rng.shuffle(candidates)
    selected = candidates[:2] if clock_index is None else candidates[:1]
    return {index for index in selected if clock_index is None or index + 1 < clock_index}


def gate_value(clock_index: int | None, index: int, flickers: set[int]) -> bool:
    return bool((clock_index is not None and index >= clock_index) or index in flickers)


def numeric_value(kind: str, passing: bool, seed: int) -> float:
    rng = random.Random(seed)
    ranges = {
        "rule_accuracy": ((0.992, 1.0), (0.55, 0.975)),
        "min_correct_probability": ((0.965, 0.995), (0.51, 0.94)),
        "decode_accuracy": ((0.965, 0.995), (0.50, 0.93)),
        "transfer_success": ((0.93, 0.99), (0.05, 0.86)),
        "preservation_success": ((0.95, 0.995), (0.20, 0.87)),
        "single_necessity": ((0.20, 0.60), (0.0, 0.11)),
        "joint_necessity": ((0.25, 0.70), (0.0, 0.11)),
    }
    low, high = ranges[kind][0 if passing else 1]
    return low + (high - low) * rng.random()


def make_trace(
    system_id: str,
    clocks: dict[str, int | None],
    observable_seed: int,
) -> list[dict[str, Any]]:
    flickers = {clock: isolated_flickers(clocks[clock], observable_seed + index * 101) for index, clock in enumerate(CLOCKS)}
    # E requires D at the same checkpoint.  Synchronizing isolated E/D
    # flickers creates a genuine one-checkpoint false positive that the stable
    # onset rule must reject.
    flickers["E"] = set(flickers["D"])
    rows = []
    for index, step in enumerate(CHECKPOINT_STEPS):
        gates = {clock: gate_value(clocks[clock], index, flickers[clock]) for clock in CLOCKS}
        metric_pass = {
            "rule_accuracy": gates["R"],
            "min_correct_probability": gates["C"],
            "decode_accuracy": gates["D"],
            "transfer_success": gates["E"],
            "preservation_success": gates["E"],
            "single_necessity": gates["U1"],
            "joint_necessity": gates["UJ"],
        }
        metrics = {
            kind: numeric_value(kind, passing, observable_seed + index * 1009 + offset * 37)
            for offset, (kind, passing) in enumerate(metric_pass.items())
        }
        rows.append(
            {
                "system_id": system_id,
                "checkpoint_index": index,
                "step": step,
                "metrics": metrics,
                "zero_drift": 0.0,
            }
        )
    return rows


def observed_gates(trace: list[dict[str, Any]]) -> dict[str, list[bool]]:
    rows = sorted(trace, key=lambda row: row["checkpoint_index"])
    return {
        "R": [row["metrics"]["rule_accuracy"] >= THRESHOLDS["rule_accuracy"] for row in rows],
        "C": [row["metrics"]["min_correct_probability"] >= THRESHOLDS["min_correct_probability"] for row in rows],
        "D": [row["metrics"]["decode_accuracy"] >= THRESHOLDS["decode_accuracy"] for row in rows],
        "E": [
            row["metrics"]["decode_accuracy"] >= THRESHOLDS["decode_accuracy"]
            and row["metrics"]["transfer_success"] >= THRESHOLDS["transfer_success"]
            and row["metrics"]["preservation_success"] >= THRESHOLDS["preservation_success"]
            for row in rows
        ],
        "U1": [row["metrics"]["single_necessity"] >= THRESHOLDS["single_necessity"] for row in rows],
        "UJ": [row["metrics"]["joint_necessity"] >= THRESHOLDS["joint_necessity"] for row in rows],
    }


def stable_onset(gates: list[bool]) -> int | None:
    for start in range(len(gates) - STABLE_CONSECUTIVE + 1):
        consecutive = all(gates[start : start + STABLE_CONSECUTIVE])
        tail_fraction = sum(gates[start:]) / len(gates[start:])
        if consecutive and gates[-1] and tail_fraction >= STABLE_TAIL_FRACTION:
            return CHECKPOINT_STEPS[start]
    return None


def infer_clocks(trace: list[dict[str, Any]]) -> dict[str, int | None]:
    gates = observed_gates(trace)
    return {clock: stable_onset(gates[clock]) for clock in CLOCKS}


def relation_signature(clocks: dict[str, int | None]) -> str:
    finite = sorted({value for value in clocks.values() if value is not None})
    ranks = {value: index for index, value in enumerate(finite)}
    return "|".join(
        f"{clock}:{'X' if clocks[clock] is None else ranks[clocks[clock]]}" for clock in CLOCKS
    )


def observable_trace_digest(trace: list[dict[str, Any]]) -> str:
    public_rows = [
        {
            "checkpoint_index": row["checkpoint_index"],
            "step": row["step"],
            "metrics": row["metrics"],
            "zero_drift": row["zero_drift"],
        }
        for row in trace
    ]
    return digest(public_rows)


def generate_split(split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    systems: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    split_rng = random.Random(SPLIT_SEEDS[split])
    base_low, base_high = (4, 10) if split == "discovery" else (8, 14)
    for archetype_index, archetype in enumerate(ARCHETYPES):
        for replicate in range(REPLICATES_PER_ARCHETYPE):
            base_rng = random.Random(SPLIT_SEEDS[split] + replicate * 313 + archetype_index * 10007)
            base = base_rng.randint(base_low, base_high)
            clocks_index, latent = schedule_for(archetype, base)
            is_censor_twin = archetype in {"transient_false_positive", "right_censored_late"}
            observable_seed = (
                SPLIT_SEEDS[split] + replicate * 104729 + 17
                if is_censor_twin
                else SPLIT_SEEDS[split] + archetype_index * 1000003 + replicate * 104729
            )
            system_id = f"{split}__{archetype}__r{replicate:03d}"
            trace = make_trace(system_id, clocks_index, observable_seed)
            truth_clocks = {
                clock: None if clocks_index[clock] is None else CHECKPOINT_STEPS[clocks_index[clock]]
                for clock in CLOCKS
            }
            inferred = infer_clocks(trace)
            truth_signature = relation_signature(truth_clocks)
            inferred_signature = relation_signature(inferred)
            all_censored = all(value is None for value in inferred.values())
            systems.append(
                {
                    "system_id": system_id,
                    "split": split,
                    "archetype": archetype,
                    "replicate": replicate,
                    "twin_id": f"{split}__censor_twin__r{replicate:03d}" if is_censor_twin else None,
                    "latent_truth": latent,
                    "truth_clocks": truth_clocks,
                    "inferred_clocks": inferred,
                    "truth_signature": truth_signature,
                    "inferred_signature": inferred_signature,
                    "clock_exact": truth_clocks == inferred,
                    "signature_exact": truth_signature == inferred_signature,
                    "censor_cause_decision": "UNIDENTIFIABLE" if all_censored else "NOT_APPLICABLE",
                    "observable_trace_digest": observable_trace_digest(trace),
                }
            )
            all_traces.extend(trace)
    split_rng.shuffle(systems)
    split_rng.shuffle(all_traces)
    return systems, all_traces


def summarize_split(systems: list[dict[str, Any]], traces: list[dict[str, Any]]) -> dict[str, Any]:
    per_archetype = {}
    for archetype in ARCHETYPES:
        rows = [row for row in systems if row["archetype"] == archetype]
        per_archetype[archetype] = {
            "count": len(rows),
            "clock_exact_fraction": sum(row["clock_exact"] for row in rows) / len(rows),
            "signature_exact_fraction": sum(row["signature_exact"] for row in rows) / len(rows),
        }
    twin_groups: dict[str, list[dict[str, Any]]] = {}
    for row in systems:
        if row["twin_id"] is not None:
            twin_groups.setdefault(row["twin_id"], []).append(row)
    twin_pass = []
    for rows in twin_groups.values():
        twin_pass.append(
            len(rows) == 2
            and rows[0]["observable_trace_digest"] == rows[1]["observable_trace_digest"]
            and rows[0]["latent_truth"]["censor_cause"] != rows[1]["latent_truth"]["censor_cause"]
            and all(row["censor_cause_decision"] == "UNIDENTIFIABLE" for row in rows)
        )
    finite_values = [
        value
        for row in traces
        for value in list(row["metrics"].values()) + [row["zero_drift"]]
    ]
    gates = {
        "clock_exact": all(row["clock_exact"] for row in systems),
        "signature_exact": all(row["signature_exact"] for row in systems),
        "all_archetypes_exact": all(
            item["clock_exact_fraction"] >= 0.99 and item["signature_exact_fraction"] >= 0.99
            for item in per_archetype.values()
        ),
        "censor_twins_unidentifiable": all(twin_pass) and len(twin_pass) == REPLICATES_PER_ARCHETYPE,
        "all_finite": all(math.isfinite(float(value)) for value in finite_values),
        "zero_drift": all(row["zero_drift"] == 0.0 for row in traces),
    }
    clock_cells = len(systems) * len(CLOCKS)
    exact_clock_cells = sum(
        row["truth_clocks"][clock] == row["inferred_clocks"][clock]
        for row in systems
        for clock in CLOCKS
    )
    return {
        "system_count": len(systems),
        "trace_row_count": len(traces),
        "checkpoint_count_per_system": len(CHECKPOINT_STEPS),
        "clock_cell_count": clock_cells,
        "clock_cell_exact_count": exact_clock_cells,
        "clock_cell_exact_fraction": exact_clock_cells / clock_cells,
        "signature_exact_fraction": sum(row["signature_exact"] for row in systems) / len(systems),
        "censor_twin_pair_count": len(twin_pass),
        "censor_twin_pass_count": sum(twin_pass),
        "per_archetype": per_archetype,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def build_protocol() -> dict[str, Any]:
    upstream = read_json(PHASE1215_FINAL)
    if upstream["authorized_next"]["experiment"] != "T01_KNOWN_TRUTH_THREE_CLOCK_ZOO":
        raise RuntimeError("Phase1215 did not select the three-clock zoo")
    return {
        "phase": PHASE,
        "schema_version": "phase1216.known_truth_three_clock_zoo.protocol.v1",
        "created_at": utc_now(),
        "purpose": "calibrate typed rule, confidence, interface, readability, and necessity formation clocks",
        "source_hashes": source_hashes(),
        "upstream": {
            "phase1215_final_sha256": file_sha256(PHASE1215_FINAL),
            "phase1215_final_digest": upstream["final_digest"],
            "selected_experiment": upstream["authorized_next"]["experiment"],
        },
        "splits": {split: {"seed": seed, "replicates_per_archetype": REPLICATES_PER_ARCHETYPE} for split, seed in SPLIT_SEEDS.items()},
        "archetypes": ARCHETYPES,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "clocks": {
            "R": "stable top-1 rule",
            "C": "stable minimum correct probability at least 0.95",
            "D": "stable decodability",
            "E": "stable decodability plus preservation and counterfactual transfer",
            "U1": "stable single-path necessity",
            "UJ": "stable joint-path necessity",
        },
        "thresholds": THRESHOLDS,
        "stable_onset": {
            "consecutive": STABLE_CONSECUTIVE,
            "tail_fraction": STABLE_TAIL_FRACTION,
            "endpoint_required": True,
        },
        "gates": {
            "clock_cell_exact_fraction_min": 0.99,
            "signature_exact_fraction_min": 0.99,
            "every_archetype_every_split_min": 0.99,
            "censor_twins_unidentifiable_fraction": 1.0,
            "all_finite": True,
            "zero_drift": True,
            "both_splits_required": True,
        },
        "scope": {
            "known_truth_only": True,
            "trained_neural_network": False,
            "pretrained_language_model": False,
            "natural_language_claim": False,
            "new_mechanism_claim": False,
        },
        "exclusions": [
            "Exact clock recovery is construct calibration, not neural external validity.",
            "Scripted trajectories may be easier than learned trajectories.",
            "Observable censor twins prove only contract-relative non-identifiability.",
            "No Qwen3, GLM4, or DS7B run is authorized by this phase.",
        ],
    }


def verify_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    candidate = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    if digest(candidate) != protocol["protocol_digest"]:
        raise RuntimeError("protocol digest mismatch")
    if protocol["source_hashes"] != source_hashes():
        raise RuntimeError("source changed after preregistration")
    upstream = read_json(PHASE1215_FINAL)
    if protocol["upstream"]["phase1215_final_sha256"] != file_sha256(PHASE1215_FINAL):
        raise RuntimeError("Phase1215 upstream changed")
    if protocol["upstream"]["phase1215_final_digest"] != upstream["final_digest"]:
        raise RuntimeError("Phase1215 final digest changed")
    return protocol


def preregister() -> None:
    if PROTOCOL_PATH.exists() or SUMMARY_PATH.exists():
        raise RuntimeError("Phase1216 protocol or outcomes already exist")
    protocol = build_protocol()
    protocol["protocol_digest"] = digest(protocol)
    write_json(PROTOCOL_PATH, protocol)
    print(canonical_json({"phase": PHASE, "protocol_digest": protocol["protocol_digest"]}))


def run() -> None:
    if SUMMARY_PATH.exists():
        raise RuntimeError("Phase1216 outcomes already exist")
    protocol = verify_protocol()
    summaries = {}
    manifests = {}
    for split in ("discovery", "confirmation"):
        systems, traces = generate_split(split)
        systems_path, traces_path = output_paths(split)
        write_jsonl(systems_path, systems)
        write_jsonl(traces_path, traces)
        summaries[split] = summarize_split(systems, traces)
        manifests[split] = {
            "systems_path": str(systems_path.relative_to(ROOT)).replace("\\", "/"),
            "systems_sha256": file_sha256(systems_path),
            "traces_path": str(traces_path.relative_to(ROOT)).replace("\\", "/"),
            "traces_sha256": file_sha256(traces_path),
        }
    overall_pass = all(summary["overall_pass"] for summary in summaries.values())
    summary = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": "known_truth_three_clock_calibration_passed" if overall_pass else "known_truth_three_clock_calibration_failed",
        "protocol_digest": protocol["protocol_digest"],
        "summaries": summaries,
        "manifests": manifests,
        "overall_pass": overall_pass,
        "claims": {
            "three_clock_construct_calibrated": overall_pass,
            "readability_use_separation_calibrated": overall_pass,
            "censoring_clock_logic_calibrated": overall_pass,
            "censor_cause_identifiable_from_public_trace": False,
            "free_network_external_validity": "not_tested",
            "pretrained_language_external_validity": "not_tested",
        },
    }
    write_json(SUMMARY_PATH, summary)
    print(canonical_json(summary))


def finalize() -> None:
    protocol = verify_protocol()
    summary = read_json(SUMMARY_PATH)
    audit = read_json(AUDIT_PATH)
    if not audit.get("gate_pass", False):
        raise RuntimeError("independent audit did not pass")
    positive = bool(summary["overall_pass"])
    final = {
        "phase": PHASE,
        "created_at": utc_now(),
        "status": summary["status"],
        "protocol_digest": protocol["protocol_digest"],
        "audit_digest": audit["audit_digest"],
        "summary": summary,
        "k_item": {
            "id": "K193" if positive else None,
            "grade": "E3-KT" if positive else None,
            "claim": (
                "A typed rule-calibration-interface clock instrument, augmented by decode and single/joint-use clocks, "
                "recovers scripted known-truth order and censoring while abstaining on observationally identical censor causes."
                if positive
                else "not_registered"
            ),
        },
        "authorized_next": {
            "experiment": "T02_FACTORIAL_FREE_FORMATION" if positive else None,
            "scope": "new preregistered free-Transformer construct-validity test" if positive else "stop",
            "automatic_execution": False,
            "reason": "The known-truth camera is calibrated, but free learned trajectories are a new scientific object requiring a separate frozen protocol.",
            "pretrained_model_run": False,
        },
        "new_mathematics_required": False,
    }
    final["final_digest"] = digest(final)
    write_json(FINAL_PATH, final)
    print(canonical_json({"status": final["status"], "k_item": final["k_item"], "authorized_next": final["authorized_next"], "final_digest": final["final_digest"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preregister", "run", "finalize"))
    command = parser.parse_args().command
    {"preregister": preregister, "run": run, "finalize": finalize}[command]()


if __name__ == "__main__":
    main()
