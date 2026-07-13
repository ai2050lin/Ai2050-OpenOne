#!/usr/bin/env python3
"""Run the frozen Phase396 field-extraction physical intervention."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase392_parent_boundary_intervention import projection_metrics, run_path  # noqa: E402
from phase395_binding_intervention import (  # noqa: E402
    GENERATION_SCENARIOS,
    SCENARIOS,
    generate_patch,
    scenario_spec,
)


OUT = ROOT / "tests/gpt5/result/phase396_field_binding_physical"
CASES = OUT / "protocol/private/phase396_physical_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def run(model: str, max_new_tokens: int) -> dict[str, Any]:
    cases = [row for row in read_jsonl(CASES) if row["private_execution_model"] == model]
    if len(cases) != 24:
        raise RuntimeError(f"Expected 24 Phase396 cases for {model}, got {len(cases)}")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        grouped[case["phase395_public_parallel_group_id"]][case["condition_code_private"]] = case
    if len(grouped) != 6 or any(set(items) != {"A", "B", "C", "D"} for items in grouped.values()):
        raise RuntimeError(f"Invalid Phase396 groups for {model}")
    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        for group_index, (group_id, conditions) in enumerate(sorted(grouped.items()), 1):
            capture_layers = tuple(sorted({
                int(next(iter(conditions.values()))["candidate_layer"]),
                int(next(iter(conditions.values()))["wrong_depth_layer"]),
            }))
            natural = {
                name: run_path(loaded, case, capture_layers)
                for name, case in conditions.items()
            }
            for name, outcome in natural.items():
                if not outcome["all_prefix_transitions_match"] or not outcome["target_decision_argmax_match"]:
                    raise RuntimeError(f"Phase396 natural replay failed for {model}/{group_id}/{name}")
            for recipient_name, donor_name in (("A", "B"), ("B", "A"), ("C", "D"), ("D", "C")):
                recipient_case, donor_case = conditions[recipient_name], conditions[donor_name]
                recipient, donor = natural[recipient_name], natural[donor_name]
                recipient_token = int(recipient_case["target_first_token_id_private"])
                donor_token = int(donor_case["target_first_token_id_private"])
                recipient_margin = float(
                    recipient["decision_logits"][donor_token]
                    - recipient["decision_logits"][recipient_token]
                )
                donor_margin = float(
                    donor["decision_logits"][donor_token]
                    - donor["decision_logits"][recipient_token]
                )
                scenario_rows = []
                generation_rows = []
                for scenario in SCENARIOS:
                    if scenario == "no_intervention":
                        outcome = recipient
                    else:
                        layer, positions, vectors = scenario_spec(
                            scenario, recipient_case, donor_case, recipient, donor
                        )
                        outcome = run_path(
                            loaded,
                            recipient_case,
                            capture_layers,
                            patch_layer=layer,
                            recipient_positions=positions,
                            patch_vectors=vectors,
                        )
                    margin = float(
                        outcome["decision_logits"][donor_token]
                        - outcome["decision_logits"][recipient_token]
                    )
                    scenario_rows.append({
                        "scenario": scenario,
                        **projection_metrics(
                            recipient["query_outputs"][int(recipient_case["candidate_layer"])],
                            donor["query_outputs"][int(recipient_case["candidate_layer"])],
                            outcome["query_outputs"][int(recipient_case["candidate_layer"])],
                        ),
                        "donor_vs_recipient_margin": margin,
                        "donor_margin_shift": margin - recipient_margin,
                        "normalized_margin_mediation": (margin - recipient_margin)
                        / max(abs(donor_margin - recipient_margin), 1e-8),
                        "argmax_token_id": int(torch.argmax(outcome["decision_logits"]).item()),
                        "argmax_is_donor": int(torch.argmax(outcome["decision_logits"]).item()) == donor_token,
                        "patch_audit": outcome["audit"],
                    })
                    if scenario in GENERATION_SCENARIOS:
                        generation_rows.append({
                            "scenario": scenario,
                            **generate_patch(
                                loaded, recipient_case, donor_case, recipient, donor,
                                scenario, max_new_tokens,
                            ),
                        })
                rows.append({
                    "schema_version": "70.1.0",
                    "phase_id": "Phase396-FieldPhysicalIntervention",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "task_surface": "field_extraction",
                    "parallel_group_id": group_id,
                    "direction_id": f"{group_id}:{recipient_name}<-{donor_name}",
                    "recipient_condition": recipient_name,
                    "donor_condition": donor_name,
                    "candidate_layer": int(recipient_case["candidate_layer"]),
                    "wrong_depth_layer": int(recipient_case["wrong_depth_layer"]),
                    "recipient_natural_margin": recipient_margin,
                    "donor_natural_margin": donor_margin,
                    "natural_margin_separation": donor_margin - recipient_margin,
                    "scenario_rows": scenario_rows,
                    "generation_rows": generation_rows,
                    "physical_binding_claim": False,
                    "single_neuron_claim": False,
                })
            print(f"[{model}/physical] groups {group_index}/6 directions={len(rows)}", flush=True)
        root = OUT / "collection" / model
        write_jsonl(root / "direction_rows.jsonl", rows)
        summary = {
            "schema_version": "70.1.0",
            "phase_id": "Phase396-FieldPhysicalIntervention",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "group_count": len(grouped),
            "direction_count": len(rows),
            "scenario_count": len(rows) * len(SCENARIOS),
            "generation_count": len(rows) * len(GENERATION_SCENARIOS),
            "valid": len(rows) == 24,
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
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()
    run(args.model, args.max_new_tokens)
