#!/usr/bin/env python3
"""Decompose Phase536 errors without running or fitting another model."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("discovery", "entity_prediction", "relation_prediction")
SOURCE_DIR = ROOT / "tests/gpt5/result/phase536_pair_addressed_binding_behavior"
AUTH_PATH = SOURCE_DIR / "phase536_physical_authorization.json"
OUT_DIR = ROOT / "tests/gpt5/result/phase537_pair_addressed_behavior_diagnostics"
OUT_PATH = OUT_DIR / "phase537_pair_addressed_behavior_diagnostics.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def simple_rate(values: list[bool]) -> dict[str, Any]:
    return {
        "n": len(values),
        "count": sum(values),
        "rate": sum(values) / len(values) if values else 0.0,
    }


def grouped_rates(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {
        value: simple_rate([bool(row["first_event_correct"]) for row in local])
        for value, local in sorted(groups.items())
    }


def pair_flip_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["pair_flip_id"]].append(row)
    categories = Counter()
    event_flip = []
    correct = []
    for group in groups.values():
        if len(group) != 2:
            raise RuntimeError("invalid pair-flip group")
        true_row = next(row for row in group if row["truth_value"])
        false_row = next(row for row in group if not row["truth_value"])
        true_correct = bool(true_row["first_event_correct"])
        false_correct = bool(false_row["first_event_correct"])
        if true_correct and false_correct:
            categories["both_correct"] += 1
        elif true_correct:
            categories["true_only"] += 1
        elif false_correct:
            categories["false_only"] += 1
        else:
            categories["neither"] += 1
        recoverable = true_row["first_event_value"] is not None and false_row["first_event_value"] is not None
        event_flip.append(recoverable and true_row["first_event_value"] != false_row["first_event_value"])
        correct.append(true_correct and false_correct)
    return {
        "group_count": len(groups),
        "outcome_counts": dict(categories),
        "exact_pair_flip": simple_rate(correct),
        "observed_event_value_flip": simple_rate(event_flip),
    }


def event_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(
        "supported" if row["first_event_value"] is True
        else "contradicted" if row["first_event_value"] is False
        else "unrecoverable"
        for row in rows
    )
    return {
        key: {"count": counts[key], "rate": counts[key] / len(rows) if rows else 0.0}
        for key in ("supported", "contradicted", "unrecoverable")
    }


def split_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair = pair_flip_report(rows)
    truth_surface = {}
    for truth in (False, True):
        for surface in sorted({row["surface"] for row in rows}):
            local = [
                row for row in rows
                if bool(row["truth_value"]) == truth and row["surface"] == surface
            ]
            truth_surface[f"{str(truth).lower()}__{surface}"] = simple_rate(
                [bool(row["first_event_correct"]) for row in local]
            )
    return {
        "row_count": len(rows),
        "by_truth": grouped_rates(rows, "truth_value"),
        "by_surface": grouped_rates(rows, "surface"),
        "by_truth_surface": truth_surface,
        "by_world": grouped_rates(rows, "world_id"),
        "by_candidate_index": grouped_rates(rows, "candidate_index"),
        "by_candidate_slot": grouped_rates(rows, "candidate_slot"),
        "by_relation": grouped_rates(rows, "relation_active"),
        "event_distribution": event_distribution(rows),
        "pair_flip": pair,
    }


def main() -> None:
    authorization = read_json(AUTH_PATH)
    if authorization["physical_authorized_models"]:
        raise RuntimeError("Phase537 stop audit expects no authorized physical model")
    model_reports = {}
    for model in MODELS:
        summary = read_json(SOURCE_DIR / f"phase536_{model}_behavior_summary.json")
        rows = read_jsonl(SOURCE_DIR / f"phase536_{model}_behavior_rows.jsonl")
        model_reports[model] = {
            "runtime_seconds": summary["runtime_seconds"],
            "physical_authorized": summary["physical_authorized"],
            "splits": {
                split: split_diagnostics([row for row in rows if row["split"] == split])
                for split in SPLITS
            },
        }

    payload = {
        "schema_version": "phase537_pair_addressed_behavior_diagnostics.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_physical_stop",
        "models_in_required_order": list(MODELS),
        "physical_authorized_models": [],
        "model_reports": model_reports,
        "stage_findings": {
            "pair_address_shortcut_detected": False,
            "all_models_behavior_qualified": False,
            "truth_polarity_bias_present": True,
            "qwen3_relation_holdout_only_pass": True,
            "physical_collection_run": False,
            "prediction_split_hidden_states_read": False,
            "pipeline_permutations_run": 0,
            "sealed_split_read": False,
        },
        "evidence_boundary": {
            "behavior_only": True,
            "hidden_state_observed": False,
            "predictive_pair_binding": False,
            "causal": False,
            "sealed": False,
        },
        "next_algorithmic_question": (
            "Separate latent truth discrimination from the supported/contradicted generation interface "
            "using a frozen two-answer logit contrast, then require fresh confirmation before physical collection."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(OUT_PATH)


if __name__ == "__main__":
    main()
