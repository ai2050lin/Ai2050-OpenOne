#!/usr/bin/env python3
"""Analyze Phase556 direct-parent response cards without forcing a writer."""

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
ROWS = OUT_DIR / "direct_parent_decomposition/phase556_direct_parent_rows.jsonl"
OUTPUT = OUT_DIR / "phase556_direct_parent_analysis.json"
EXPECTED_READOUT_CONTRACT = "first_non_whitespace_candidate_content_token_v2"
EXPECTED_INTERVENTION_SEMANTICS = "additive_parent_component_delta_at_child_state"
CONTROL = "channel_roll_all_parent"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def median(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(statistics.median(finite)) if finite else None


def analyze() -> dict[str, Any]:
    rows = read_jsonl(ROWS)
    readout_contracts = {row.get("restricted_readout_contract") for row in rows}
    if readout_contracts != {EXPECTED_READOUT_CONTRACT}:
        raise RuntimeError(f"Phase556 stale or mixed parent readout contracts: {readout_contracts}")
    torch_dtypes = {row.get("torch_dtype") for row in rows}
    if torch_dtypes != {"torch.bfloat16"}:
        raise RuntimeError(f"Phase556 parent dtype drift: {torch_dtypes}")
    intervention_semantics = {row.get("parent_intervention_semantics") for row in rows}
    if intervention_semantics != {EXPECTED_INTERVENTION_SEMANTICS}:
        raise RuntimeError(
            f"Phase556 stale or mixed parent intervention semantics: {intervention_semantics}"
        )
    if {row.get("intervention_location") for row in rows} != {"boundary_layer_input"}:
        raise RuntimeError("Phase556 parent intervention location drift")
    if any(row.get("compute_edge") for row in rows):
        raise RuntimeError("Phase556 additive child-state attribution cannot claim a compute edge")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["mechanism"], row["condition"])].append(row)
    mechanisms = sorted({key[0] for key in grouped})
    mechanism_reports: dict[str, Any] = {}
    for mechanism in mechanisms:
        control_rows = [row for row in grouped[(mechanism, CONTROL)] if row["numerical_valid"]]
        control_target = [row for row in control_rows if row["pair_role"] == "target"]
        control_donor_rate = rate(control_target, "patched_donor_selected")
        control_total = grouped[(mechanism, CONTROL)]
        condition_reports: dict[str, Any] = {}
        for (row_mechanism, condition), condition_rows in sorted(grouped.items()):
            if row_mechanism != mechanism:
                continue
            valid = [row for row in condition_rows if row["numerical_valid"]]
            target = [row for row in valid if row["pair_role"] == "target"]
            specificity = [row for row in valid if row["pair_role"] == "specificity_control"]
            donor_rate = rate(target, "patched_donor_selected")
            transfer = median([row["transfer_fraction"] for row in target])
            criteria = {
                "target_n_at_least_64": len(target) >= 64,
                "specificity_n_at_least_64": len(specificity) >= 64,
                "numerical_valid_at_least_95pct": len(valid) / len(condition_rows) >= 0.95,
                "roll_control_target_n_at_least_64": len(control_target) >= 64,
                "roll_control_numerically_valid_at_least_95pct": (
                    len(control_rows) / len(control_total) >= 0.95
                ),
                "natural_endpoints_at_least_95pct": min(
                    rate(valid, "baseline_semantic_correct_restricted"),
                    rate(valid, "natural_donor_semantic_correct_restricted"),
                ) >= 0.95,
                "donor_selection_at_least_50pct": donor_rate >= 0.50,
                "over_roll_control_at_least_15pp": donor_rate - control_donor_rate >= 0.15,
                "median_transfer_at_least_0_20": transfer is not None and transfer >= 0.20,
                "irrelevant_query_preservation_at_least_90pct": (
                    rate(specificity, "patched_recipient_preserved") >= 0.90
                ),
            }
            qualified = condition != CONTROL and all(criteria.values())
            condition_reports[condition] = {
                "total_n": len(condition_rows),
                "valid_n": len(valid),
                "target_n": len(target),
                "specificity_n": len(specificity),
                "target_donor_selection_rate": donor_rate,
                "roll_control_donor_selection_rate": control_donor_rate,
                "specificity_recipient_preservation_rate": rate(
                    specificity, "patched_recipient_preserved"
                ),
                "median_target_transfer_fraction": transfer,
                "criteria": criteria,
                "causal_parent_qualified": qualified,
            }
        qualified_conditions = sorted(
            condition for condition, report in condition_reports.items()
            if report["causal_parent_qualified"]
        )
        component_contribution_conditions = sorted(set(qualified_conditions) & {
            "attention_write_delta", "mlp_write_delta", "attention_mlp_joint_delta"
        })
        residual_only = (
            "residual_carry_delta" in qualified_conditions
            and not component_contribution_conditions
        )
        exemplar = next(row for row in rows if row["mechanism"] == mechanism)
        mechanism_reports[mechanism] = {
            "boundary_layer": exemplar["boundary_layer"],
            "parent_layer": exemplar["parent_layer"],
            "qualified_conditions": qualified_conditions,
            "qualified_component_contribution_conditions": component_contribution_conditions,
            "qualified_writer_conditions": [],
            "residual_carry_without_local_writer": residual_only,
            "direct_compute_edge_recovered": False,
            "parameter_localization_authorized": False,
            "condition_reports": condition_reports,
        }
    payload = {
        "schema_version": "phase556_direct_parent_analysis.v2",
        "phase_id": "Phase556",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "restricted_readout_contract": EXPECTED_READOUT_CONTRACT,
        "torch_dtypes": sorted(torch_dtypes),
        "parent_intervention_semantics": EXPECTED_INTERVENTION_SEMANTICS,
        "mechanism_reports": mechanism_reports,
        "parameter_localization_authorized_mechanisms": [
            mechanism for mechanism, report in mechanism_reports.items()
            if report["parameter_localization_authorized"]
        ],
        "evidence_boundary": {
            "independent_open_parent_holdout": True,
            "additive_parent_contribution_at_child_state": True,
            "direct_compute_edge_recovered": False,
            "single_model": True,
            "natural_fruit_parameter_storage": False,
            "sealed_split_read": False,
            "closure_claim_authorized": False,
        },
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        mechanism: {
            "qualified_conditions": report["qualified_conditions"],
            "parameter_localization_authorized": report["parameter_localization_authorized"],
        }
        for mechanism, report in mechanism_reports.items()
    }, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    analyze()
