#!/usr/bin/env python3
"""Audit the Phase421 behavior boundary before any physical collection."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase421_balanced_boundary_case_bank import MODELS, OUT, SCHEMA_VERSION  # noqa: E402


PHASE_ID = "Phase421-BalancedBoundaryBehaviorAudit"
SPLITS = ("discovery", "calibration", "behavior_holdout", "physical_holdout")
DEVELOPMENT_SPLITS = ("discovery", "calibration", "behavior_holdout")
NEAR_ZERO = 0.25
MIN_POSITIVE_RATE = 0.20
MIN_NEGATIVE_RATE = 0.20
MIN_NEAR_ZERO_RATE = 0.02
MAX_DOMINANT_RATE = 0.70


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase421 non-finite analysis scalar: {value}")
    return round(float(value), 10)


def effect_class(value: float) -> str:
    if value > NEAR_ZERO:
        return "positive"
    if value < -NEAR_ZERO:
        return "negative"
    return "near_zero"


def build_effect_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            row["model"],
            row["group_id"],
            row["current_identity"],
            row["interface"],
            row["current_support_count"],
            row["history_reliability_score"],
        )
        groups[key][row["history_relation"]] = row
    output = []
    for key, cells in sorted(groups.items()):
        if set(cells) != {"compatible", "conflict", "irrelevant"}:
            raise RuntimeError(f"Incomplete Phase421 relation cells: {key}/{set(cells)}")
        baseline = cells["irrelevant"]
        for relation in ("compatible", "conflict"):
            cell = cells[relation]
            effect = float(cell["target_branch_margin"]) - float(
                baseline["target_branch_margin"]
            )
            output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase421-BehaviorRelationEffect",
                    "created_at": now(),
                    "model": cell["model"],
                    "group_id": cell["group_id"],
                    "group_index": cell["group_index"],
                    "split": cell["split"],
                    "family_id": cell["family_id"],
                    "mechanism_id": cell["mechanism_id"],
                    "interface": cell["interface"],
                    "current_identity": cell["current_identity"],
                    "current_support_count": cell["current_support_count"],
                    "history_reliability_score": cell["history_reliability_score"],
                    "history_relation": relation,
                    "relation_margin_effect_vs_irrelevant": clean(effect),
                    "relation_effect_class": effect_class(effect),
                    "relation_prompt_token_count_delta": int(cell["prompt_token_count"])
                    - int(baseline["prompt_token_count"]),
                    "physical_development_panel": cell["physical_development_panel"],
                    "physical_holdout_sealed": cell["physical_holdout_sealed"],
                    "physical": False,
                    "predictive": False,
                    "causal": False,
                }
            )
    return output


def cell_audit(rows: list[dict[str, Any]], model: str, split: str) -> dict[str, Any]:
    values = [row for row in rows if row["model"] == model and row["split"] == split]
    counts = Counter(row["relation_effect_class"] for row in values)
    total = len(values)
    rates = {
        label: counts[label] / total for label in ("positive", "negative", "near_zero")
    }
    compatible = [
        row["relation_margin_effect_vs_irrelevant"]
        for row in values
        if row["history_relation"] == "compatible"
    ]
    conflict = [
        row["relation_margin_effect_vs_irrelevant"]
        for row in values
        if row["history_relation"] == "conflict"
    ]
    support_medians = {
        str(level): clean(
            median(
                row["relation_margin_effect_vs_irrelevant"]
                for row in values
                if row["current_support_count"] == level
            )
        )
        for level in (1, 2, 3)
    }
    reliability_medians = {
        str(level): clean(
            median(
                row["relation_margin_effect_vs_irrelevant"]
                for row in values
                if row["history_reliability_score"] == level
            )
        )
        for level in (1, 2, 3)
    }
    balance_pass = bool(
        rates["positive"] >= MIN_POSITIVE_RATE
        and rates["negative"] >= MIN_NEGATIVE_RATE
        and rates["near_zero"] >= MIN_NEAR_ZERO_RATE
        and max(rates.values()) <= MAX_DOMINANT_RATE
        and median(compatible) > median(conflict)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": model,
        "split": split,
        "effect_count": total,
        "effect_class_count": dict(counts),
        "effect_class_rate": {key: clean(value) for key, value in rates.items()},
        "compatible_effect_median": clean(median(compatible)),
        "conflict_effect_median": clean(median(conflict)),
        "support_level_effect_median": support_medians,
        "reliability_level_effect_median": reliability_medians,
        "positive_and_negative_boundary_present": rates["positive"] >= MIN_POSITIVE_RATE
        and rates["negative"] >= MIN_NEGATIVE_RATE,
        "near_zero_boundary_present": rates["near_zero"] >= MIN_NEAR_ZERO_RATE,
        "largest_class_below_ceiling": max(rates.values()) <= MAX_DOMINANT_RATE,
        "compatible_median_exceeds_conflict_median": median(compatible) > median(conflict),
        "behavior_boundary_gate_pass": balance_pass,
    }


def generation_audit(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    selected = [row for row in rows if row["model"] == model]
    by_relation = {}
    for relation in ("compatible", "conflict", "irrelevant"):
        values = [row for row in selected if row["history_relation"] == relation]
        by_relation[relation] = {
            "count": len(values),
            "target_event_count": sum(row["target_event_match"] for row in values),
            "opposite_event_count": sum(row["opposite_event_match"] for row in values),
            "right_censored_count": sum(row["right_censored"] for row in values),
        }
    return {
        "model": model,
        "condition_count": len(selected),
        "by_relation": by_relation,
    }


def analyze() -> dict[str, Any]:
    qualification = read_json(OUT / "phase421_denominator_qualification.json")
    if not qualification["valid"]:
        raise RuntimeError("Phase421 denominator invalid")
    score_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    model_summaries = []
    for model in MODELS:
        model_root = OUT / "models" / model
        summary = read_json(model_root / "phase421_behavior_complete.json")
        if not summary["all_behavior_rows_pass"]:
            raise RuntimeError(f"Phase421 behavior incomplete for {model}")
        model_summaries.append(summary)
        score_rows.extend(read_jsonl(model_root / "phase421_behavior_margin_rows.jsonl"))
        generation_rows.extend(read_jsonl(model_root / "phase421_behavior_generation_rows.jsonl"))
    effects = build_effect_rows(score_rows)
    audits = [
        cell_audit(effects, model, split)
        for model in MODELS
        for split in SPLITS
    ]
    development_audits = [row for row in audits if row["split"] in DEVELOPMENT_SPLITS]
    behavior_gate = all(row["behavior_boundary_gate_pass"] for row in development_audits)
    generation = [generation_audit(generation_rows, model) for model in MODELS]
    authorization = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase421-PhysicalDevelopmentAuthorization",
        "created_at": now(),
        "denominator_valid": qualification["valid"],
        "all_model_behavior_traces_valid": all(
            row["all_behavior_rows_pass"] for row in model_summaries
        ),
        "development_behavior_boundary_gate_pass": behavior_gate,
        "development_audit_cell_count": len(development_audits),
        "development_audit_pass_count": sum(
            row["behavior_boundary_gate_pass"] for row in development_audits
        ),
        "physical_development_collection_authorized": behavior_gate,
        "physical_holdout_collection_authorized": False,
        "causal_intervention_authorized": False,
        "single_neuron_scan_authorized": False,
        "stop_reason": None
        if behavior_gate
        else "behavior_boundary_did_not_cover_registered_positive_negative_near_zero_regions",
    }
    write_jsonl(OUT / "phase421_behavior_effect_rows.jsonl", effects)
    write_jsonl(OUT / "phase421_behavior_boundary_audit.jsonl", audits)
    write_json(OUT / "phase421_generation_panel_audit.json", generation)
    write_json(OUT / "phase421_physical_development_authorization.json", authorization)
    return authorization


def main() -> None:
    authorization = analyze()
    print(json.dumps(authorization, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
