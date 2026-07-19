#!/usr/bin/env python3
"""Analyze replicated Qwen3 Phase556 layer-input causal boundaries."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase556_fruit_encoding"
MODEL = "qwen3"
BOUNDARY_DIR = OUT_DIR / "layer_input_boundary"
SPLITS = ("boundary_discovery", "boundary_confirmation")
SCENARIOS = ("matched_factor_delta", "channel_roll_delta")
OUTPUT = OUT_DIR / "phase556_layer_input_boundary_analysis.json"
EXPECTED_READOUT_CONTRACT = "first_non_whitespace_candidate_content_token_v2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(statistics.median(finite)) if finite else None


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def analyze() -> dict[str, Any]:
    rows_by_split = {
        split: read_jsonl(
            BOUNDARY_DIR / split / "phase556_boundary_rows.jsonl"
        )
        for split in SPLITS
    }
    readout_contracts = {
        row.get("restricted_readout_contract")
        for rows in rows_by_split.values() for row in rows
    }
    if readout_contracts != {EXPECTED_READOUT_CONTRACT}:
        raise RuntimeError(f"Phase556 stale or mixed boundary readout contracts: {readout_contracts}")
    torch_dtypes = {
        row.get("torch_dtype") for rows in rows_by_split.values() for row in rows
    }
    if torch_dtypes != {"torch.bfloat16"}:
        raise RuntimeError(f"Phase556 boundary dtype drift: {torch_dtypes}")
    mechanisms = sorted({row["mechanism"] for rows in rows_by_split.values() for row in rows})
    layers = sorted({int(row["layer"]) for rows in rows_by_split.values() for row in rows})
    mechanism_reports: dict[str, Any] = {}
    for mechanism in mechanisms:
        split_reports: dict[str, Any] = {}
        pass_by_split: dict[str, set[int]] = {}
        for split, all_rows in rows_by_split.items():
            layer_reports: dict[str, Any] = {}
            passed: set[int] = set()
            for layer in layers:
                rows = [
                    row for row in all_rows
                    if row["mechanism"] == mechanism and int(row["layer"]) == layer
                ]
                scenario_reports: dict[str, Any] = {}
                for scenario in SCENARIOS:
                    scenario_rows = [row for row in rows if row["scenario"] == scenario]
                    valid = [row for row in scenario_rows if row["numerical_valid"]]
                    target = [row for row in valid if row["pair_role"] == "target"]
                    control = [row for row in valid if row["pair_role"] == "specificity_control"]
                    scenario_reports[scenario] = {
                        "total_n": len(scenario_rows),
                        "numerical_valid_n": len(valid),
                        "numerical_valid_rate": len(valid) / len(scenario_rows) if scenario_rows else 0.0,
                        "target_n": len(target),
                        "control_n": len(control),
                        "target_donor_selection_rate": rate(target, "patched_donor_selected"),
                        "control_recipient_preservation_rate": rate(control, "patched_recipient_preserved"),
                        "median_target_transfer_fraction": median([
                            row["transfer_fraction"] for row in target
                        ]),
                    }
                matched = scenario_reports["matched_factor_delta"]
                rolled = scenario_reports["channel_roll_delta"]
                natural = [
                    row for row in rows
                    if row["scenario"] == "matched_factor_delta" and row["numerical_valid"]
                ]
                baseline_accuracy = rate(natural, "baseline_semantic_correct_restricted")
                donor_accuracy = rate(natural, "natural_donor_semantic_correct_restricted")
                transfer = matched["median_target_transfer_fraction"]
                criteria = {
                    "target_n_at_least_96": matched["target_n"] >= 96,
                    "control_n_at_least_96": matched["control_n"] >= 96,
                    "numerical_valid_at_least_95pct": min(
                        report["numerical_valid_rate"] for report in scenario_reports.values()
                    ) >= 0.95,
                    "natural_endpoints_at_least_95pct": min(baseline_accuracy, donor_accuracy) >= 0.95,
                    "matched_donor_selection_at_least_50pct": matched["target_donor_selection_rate"] >= 0.50,
                    "matched_over_roll_at_least_15pp": (
                        matched["target_donor_selection_rate"] - rolled["target_donor_selection_rate"] >= 0.15
                    ),
                    "median_transfer_at_least_0_20": transfer is not None and transfer >= 0.20,
                    "irrelevant_query_preservation_at_least_90pct": (
                        matched["control_recipient_preservation_rate"] >= 0.90
                    ),
                }
                qualified = all(criteria.values())
                if qualified:
                    passed.add(layer)
                layer_reports[str(layer)] = {
                    "layer": layer,
                    "relative_depth": rows[0]["relative_depth"],
                    "baseline_accuracy": baseline_accuracy,
                    "donor_accuracy": donor_accuracy,
                    "criteria": criteria,
                    "causal_boundary_pass": qualified,
                    "scenario_reports": scenario_reports,
                }
            split_reports[split] = {
                "pass_layers": sorted(passed),
                "layer_reports": layer_reports,
            }
            pass_by_split[split] = passed
        replicated = sorted(set.intersection(*(pass_by_split[split] for split in SPLITS)))
        earliest = replicated[0] if replicated else None
        mechanism_reports[mechanism] = {
            "replicated_pass_layers": replicated,
            "earliest_replicated_layer": earliest,
            "earliest_replicated_relative_depth": (
                split_reports[SPLITS[0]]["layer_reports"][str(earliest)]["relative_depth"]
                if earliest is not None else None
            ),
            "parent_decomposition_authorized": earliest is not None and earliest > 0,
            "split_reports": split_reports,
        }
    payload = {
        "schema_version": "phase556_layer_input_boundary_analysis.v1",
        "phase_id": "Phase556",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "restricted_readout_contract": EXPECTED_READOUT_CONTRACT,
        "torch_dtypes": sorted(torch_dtypes),
        "mechanism_reports": mechanism_reports,
        "parent_decomposition_authorized_mechanisms": [
            mechanism for mechanism, report in mechanism_reports.items()
            if report["parent_decomposition_authorized"]
        ],
        "sealed_split_read": False,
        "closure_claim_authorized": False,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        mechanism: {
            "replicated_pass_layers": report["replicated_pass_layers"],
            "earliest_replicated_layer": report["earliest_replicated_layer"],
        }
        for mechanism, report in mechanism_reports.items()
    }, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    analyze()
