#!/usr/bin/env python3
"""Capture compact exact Phase397 factor traces at two frozen depths."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase392_parent_boundary_intervention import run_path  # noqa: E402
from phase397_multitask_protocol import MODELS  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase397_multitask_binding"
MODES = ("instrument", "discovery", "calibration", "physical_holdout")
PAIR_SPECS = (
    ("relation_x", "A", "B", "literal_identity"),
    ("relation_y", "F", "G", "literal_identity"),
    ("content", "A", "F", "same_position"),
    ("order_x", "A", "C", "literal_identity"),
    ("order_y", "F", "H", "literal_identity"),
    ("syntax_x", "A", "D", "literal_identity"),
    ("syntax_y", "F", "I", "literal_identity"),
    ("query_x", "A", "E", "literal_identity"),
    ("query_y", "F", "J", "literal_identity"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def case_file(mode: str) -> Path:
    private = OUT / "factor_trace/protocol/private"
    if mode == "instrument":
        return private / "phase397_instrument_trace_cases.jsonl"
    return private / f"phase397_{mode}_trace_cases.jsonl"


def authorized(mode: str) -> None:
    if mode == "instrument":
        allowed = read_json(OUT / "phase397_factor_trace_protocol.json")["authorization"]["instrument_trace"]
    elif mode == "discovery":
        gate = OUT / "phase397_factor_trace_instrument_audit.json"
        allowed = gate.is_file() and read_json(gate)["authorization"]["discovery_trace"]
    elif mode == "calibration":
        gate = OUT / "phase397_factor_discovery_analysis.json"
        allowed = gate.is_file() and read_json(gate)["authorization"]["calibration_trace"]
    else:
        gate = OUT / "phase397_factor_calibration_analysis.json"
        allowed = gate.is_file() and read_json(gate)["authorization"]["physical_holdout_trace"]
    if not allowed:
        raise RuntimeError(f"Phase397 factor trace mode not authorized: {mode}")


def positions(case: dict[str, Any], mapping: str) -> list[int]:
    values = case["literal_value_positions_private"]
    if mapping == "same_position":
        return sorted(int(position) for item in values.values() for position in item)
    return [int(position) for key in ("value_a", "value_b") for position in values[key]]


def state(outcome: dict[str, Any], case: dict[str, Any], layer: int, mapping: str) -> torch.Tensor:
    mapped = positions(case, mapping)
    return outcome["layer_inputs"][layer][0, mapped].float().reshape(-1)


def relative_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = float(torch.linalg.vector_norm(left - right).item())
    denominator = 0.5 * (
        float(torch.linalg.vector_norm(left).item()) + float(torch.linalg.vector_norm(right).item())
    )
    return numerator / max(denominator, 1e-12)


def pair_row(
    name: str,
    left_code: str,
    right_code: str,
    mapping: str,
    cases: dict[str, dict[str, Any]],
    natural: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    left_case, right_case = cases[left_code], cases[right_code]
    candidate = int(left_case["candidate_layer"])
    wrong = int(left_case["wrong_depth_layer"])
    left_candidate = state(natural[left_code], left_case, candidate, mapping)
    right_candidate = state(natural[right_code], right_case, candidate, mapping)
    left_wrong = state(natural[left_code], left_case, wrong, mapping)
    right_wrong = state(natural[right_code], right_case, wrong, mapping)
    if left_candidate.shape != right_candidate.shape or left_wrong.shape != right_wrong.shape:
        raise RuntimeError(f"Phase397 pair shape mismatch: {name}")
    candidate_delta = relative_delta(left_candidate, right_candidate)
    wrong_delta = relative_delta(left_wrong, right_wrong)
    return {
        "factor": name,
        "left_condition": left_code,
        "right_condition": right_code,
        "mapping": mapping,
        "candidate_layer": candidate,
        "wrong_depth_layer": wrong,
        "state_width": int(left_candidate.numel()),
        "candidate_relative_delta": candidate_delta,
        "wrong_depth_relative_delta": wrong_delta,
        "candidate_minus_wrong_delta": candidate_delta - wrong_delta,
    }


def run(model: str, mode: str) -> dict[str, Any]:
    authorized(mode)
    cases = [row for row in read_jsonl(case_file(mode)) if row["private_execution_model"] == model]
    expected_groups = {"instrument": 3, "discovery": 24, "calibration": 12, "physical_holdout": 12}[mode]
    if len(cases) != expected_groups * 10:
        raise RuntimeError(f"Expected {expected_groups * 10} {mode} cases for {model}, got {len(cases)}")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        grouped[case["phase397_public_parallel_group_id"]][case["condition_code_private"]] = case
    if len(grouped) != expected_groups or any(set(items) != set("ABCDEFGHIJ") for items in grouped.values()):
        raise RuntimeError(f"Invalid Phase397 {mode} groups for {model}")
    loaded = None
    factor_rows: list[dict[str, Any]] = []
    instrument_rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        for group_index, (group_id, conditions) in enumerate(sorted(grouped.items()), 1):
            layers = tuple(sorted({int(conditions["A"]["candidate_layer"]), int(conditions["A"]["wrong_depth_layer"])}))
            natural = {code: run_path(loaded, case, layers) for code, case in conditions.items()}
            failed_replays = [
                code for code, outcome in natural.items()
                if not outcome["all_prefix_transitions_match"] or not outcome["target_decision_argmax_match"]
            ]
            if failed_replays:
                raise RuntimeError(f"Phase397 natural replay failed {model}/{group_id}: {failed_replays}")
            surface = conditions["A"]["task_surface_private"]
            group_pairs = [
                {
                    "schema_version": "71.4.0",
                    "phase_id": "Phase397-FactorTrace",
                    "created_at": now(),
                    "model": model,
                    "mode": mode,
                    "task_surface": surface,
                    "parallel_group_id": group_id,
                    **pair_row(*spec, conditions, natural),
                }
                for spec in PAIR_SPECS
            ]
            factor_rows.extend(group_pairs)
            if mode == "instrument":
                case_a = conditions["A"]
                candidate = int(case_a["candidate_layer"])
                literal_positions = positions(case_a, "literal_identity")
                identity = run_path(
                    loaded,
                    case_a,
                    layers,
                    patch_layer=candidate,
                    recipient_positions=literal_positions,
                    patch_vectors=natural["A"]["layer_inputs"][candidate][0, literal_positions],
                )
                logit_error = float(
                    (identity["decision_logits"] - natural["A"]["decision_logits"]).abs().max().item()
                )
                query_rows = [row for row in group_pairs if row["factor"] in {"query_x", "query_y"}]
                instrument_rows.append(
                    {
                        "model": model,
                        "task_surface": surface,
                        "parallel_group_id": group_id,
                        "natural_replay_count": len(natural),
                        "identity_patch_logit_max_error": logit_error,
                        "identity_patch_audit": identity["audit"],
                        "maximum_query_source_relative_delta": max(row["candidate_relative_delta"] for row in query_rows),
                        "valid": (
                            identity["audit"]["patch_call_count"] == 1
                            and identity["audit"]["max_patch_error"] <= 0.01
                            and identity["audit"]["max_outside_error"] <= 0.01
                            and logit_error <= 0.01
                            and max(row["candidate_relative_delta"] for row in query_rows) <= 1e-6
                        ),
                    }
                )
            print(f"[{model}/{mode}] groups {group_index}/{len(grouped)} pairs={len(factor_rows)}", flush=True)

        root = OUT / "factor_trace" / mode / model
        write_jsonl(root / "factor_rows.jsonl", factor_rows)
        if instrument_rows:
            write_jsonl(root / "instrument_rows.jsonl", instrument_rows)
        summary = {
            "schema_version": "71.4.0",
            "phase_id": "Phase397-FactorTrace",
            "created_at": now(),
            "model": model,
            "mode": mode,
            "group_count": len(grouped),
            "natural_case_count": len(cases),
            "factor_pair_count": len(factor_rows),
            "instrument_group_count": len(instrument_rows),
            "instrument_valid_count": sum(row["valid"] for row in instrument_rows),
            "all_natural_transitions_replayed": True,
            "valid": len(factor_rows) == len(grouped) * len(PAIR_SPECS) and all(row["valid"] for row in instrument_rows),
        }
        write_json(root / "complete.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args()
    run(args.model, args.mode)
