#!/usr/bin/env python3
"""Audit Phase405 finite-panel and full-vocabulary natural future gates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase404_direct_state_analysis import (  # noqa: E402
    BASELINE_MARGIN_MIN,
    REQUIRED_CASES,
    REQUIRED_GROUPS,
    matched_state_blind_baseline,
    model_family_audit,
)
from phase405_natural_future_protocol import (  # noqa: E402
    FAMILIES,
    MODELS,
    OUT,
    QUERIES,
    SPLIT_GROUP_COUNTS,
    STATE_IDS,
    SURFACE_REPLICAS,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def natural_group_audit(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    expected_cases = (
        len(STATE_IDS[family]) * len(SURFACE_REPLICAS) * len(QUERIES[family])
    )
    state_rows = []
    all_states_pass = True
    for state_id in STATE_IDS[family]:
        selected = [row for row in rows if row["state_id_private"] == state_id]
        surface_pass_count = 0
        for surface in SURFACE_REPLICAS:
            surface_rows = [
                row
                for row in selected
                if row["surface_id_private"] == surface["surface_id"]
            ]
            surface_pass_count += int(
                len(surface_rows) == len(QUERIES[family])
                and all(row["global_top_is_target_token"] for row in surface_rows)
            )
        state_pass = surface_pass_count >= 3
        all_states_pass = all_states_pass and state_pass
        state_rows.append(
            {
                "state_id": state_id,
                "natural_surface_truth_pass_count": surface_pass_count,
                "natural_state_pass": state_pass,
            }
        )
    natural_correct = sum(row["global_top_is_target_token"] for row in rows)
    return {
        "case_count": len(rows),
        "expected_case_count": expected_cases,
        "natural_top_correct_count": natural_correct,
        "required_natural_top_correct_count": REQUIRED_CASES[family],
        "all_natural_state_units_pass": all_states_pass,
        "natural_group_pass": len(rows) == expected_cases
        and natural_correct >= REQUIRED_CASES[family]
        and all_states_pass,
        "states": state_rows,
    }


def natural_model_family_audit(
    rows: list[dict[str, Any]], family: str, split: str
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["anonymous_parallel_group_id"]].append(row)
    group_rows = [
        {
            "anonymous_parallel_group_id": group_id,
            **natural_group_audit(selected, family),
        }
        for group_id, selected in sorted(groups.items())
    ]
    natural_correct = sum(row["global_top_is_target_token"] for row in rows)
    natural_accuracy = natural_correct / len(rows) if rows else 0.0
    baseline = matched_state_blind_baseline(rows)
    pass_count = sum(row["natural_group_pass"] for row in group_rows)
    return {
        "natural_top_correct_count": natural_correct,
        "natural_top_accuracy": natural_accuracy,
        "natural_accuracy_minus_state_blind_baseline": natural_accuracy - baseline,
        "natural_baseline_margin_pass": natural_accuracy - baseline
        >= BASELINE_MARGIN_MIN,
        "natural_group_pass_count": pass_count,
        "required_natural_group_pass_count": REQUIRED_GROUPS[split],
        "natural_model_family_pass": len(group_rows) == SPLIT_GROUP_COUNTS[split]
        and pass_count >= REQUIRED_GROUPS[split]
        and natural_accuracy - baseline >= BASELINE_MARGIN_MIN,
        "natural_groups": group_rows,
    }


def authorized_families(stage: str) -> tuple[str, ...]:
    if stage == "discovery":
        return FAMILIES
    if stage == "calibration":
        return tuple(
            read_json(OUT / "phase405_discovery_analysis.json")[
                "crossmodel_candidate_families"
            ]
        )
    return tuple(
        read_json(OUT / "phase405_calibration_analysis.json")[
            "crossmodel_candidate_families"
        ]
    )


def main(stage: str) -> None:
    families = authorized_families(stage)
    summaries = []
    group_details = []
    all_rows = []
    for model in MODELS:
        complete = read_json(OUT / "collection" / stage / model / "complete.json")
        if not complete["valid"]:
            raise RuntimeError(f"Invalid Phase405 collection: {model}/{stage}")
        path = OUT / "collection" / stage / "private" / model / "rows.jsonl"
        rows = read_jsonl(path) if path.is_file() else []
        all_rows.extend(rows)
        for family in families:
            selected = [row for row in rows if row["family_id"] == family]
            candidate = model_family_audit(selected, family, stage)
            natural = natural_model_family_audit(selected, family, stage)
            candidate_groups = candidate.pop("groups")
            natural_groups = natural.pop("natural_groups")
            natural_by_group = {
                row["anonymous_parallel_group_id"]: row for row in natural_groups
            }
            for group in candidate_groups:
                natural_group = natural_by_group[group["anonymous_parallel_group_id"]]
                group_details.append(
                    {
                        "model": model,
                        "family_id": family,
                        "split": stage,
                        "anonymous_parallel_group_id": group[
                            "anonymous_parallel_group_id"
                        ],
                        "candidate_group": group,
                        "natural_group": natural_group,
                    }
                )
            summary = {
                "model": model,
                **candidate,
                **natural,
            }
            summary["model_family_pass"] = (
                candidate["model_family_pass"]
                and natural["natural_model_family_pass"]
            )
            summaries.append(summary)

    crossmodel_candidates = []
    for family in families:
        selected = [row for row in summaries if row["family_id"] == family]
        if len(selected) == len(MODELS) and all(
            row["model_family_pass"] for row in selected
        ):
            crossmodel_candidates.append(family)

    payload = {
        "schema_version": "79.2.0",
        "phase_id": "Phase405-NaturalFutureAnalysis",
        "created_at": now(),
        "stage": stage,
        "authorized_families": list(families),
        "models": list(MODELS),
        "case_count": len(all_rows),
        "finite_candidate_correct_count": sum(
            row["finite_candidate_correct"] for row in all_rows
        ),
        "global_top_is_target_count": sum(
            row["global_top_is_target_token"] for row in all_rows
        ),
        "global_top_in_candidate_set_count": sum(
            row["global_top_in_candidate_set"] for row in all_rows
        ),
        "model_family_rows": summaries,
        "crossmodel_candidate_families": crossmodel_candidates,
        "authorization": {
            "run_calibration": stage == "discovery" and bool(crossmodel_candidates),
            "run_behavioral_holdout": stage == "calibration"
            and bool(crossmodel_candidates),
            "run_physical_holdout_mapping": stage == "behavioral_holdout"
            and bool(crossmodel_candidates),
            "run_causal_intervention": False,
            "run_neuron_scan": False,
        },
        "claim_boundary": {
            "candidate_name": "natural_future_predictive_state_candidate",
            "finite_branch_panel_is_exhaustive": False,
            "predictive_state_is_causal_state": False,
            "semantic_transition_graph_is_internal_operator": False,
        },
    }
    write_json(OUT / f"phase405_{stage}_analysis.json", payload)
    write_jsonl(
        OUT / "analysis" / f"phase405_{stage}_group_details.jsonl",
        group_details,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("discovery", "calibration", "behavioral_holdout"),
        required=True,
    )
    args = parser.parse_args()
    main(args.stage)
