#!/usr/bin/env python3
"""Describe the frozen Phase375 negative result without changing any gate."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase375_finite_exact_subgraphs"
DISCOVERY = OUT / "phase375_discovery"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = int(round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def basic(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values) if values else 0.0,
        "q25": quantile(values, 0.25),
        "median": quantile(values, 0.5),
        "q75": quantile(values, 0.75),
        "maximum": max(values) if values else 0.0,
        "mean": sum(values) / len(values) if values else 0.0,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lexical = [item for row in rows for item in row["lexical_pairs"]]
    gates = Counter()
    all_except_absolute = 0
    all_except_vocab = 0
    for item in lexical:
        for name, passed in item["component_gates"].items():
            gates[name] += int(passed)
        all_except_absolute += int(
            all(
                passed
                for name, passed in item["component_gates"].items()
                if name != "absolute_error"
            )
        )
        all_except_vocab += int(
            all(
                passed
                for name, passed in item["component_gates"].items()
                if name != "vocab_context"
            )
        )
    current = [float(item["current_error"]) for item in lexical]
    return {
        "lexical_count": len(lexical),
        "gate_pass_counts": dict(sorted(gates.items())),
        "all_except_absolute_error_count": all_except_absolute,
        "all_except_vocab_context_count": all_except_vocab,
        "current_error": basic(current),
        "best_single_advantage": basic(
            [float(item["best_single_error"] - item["current_error"]) for item in lexical]
        ),
        "past_margin": basic(
            [float(item["past_error"] - item["current_error"]) for item in lexical]
        ),
        "wrong_depth_margin": basic(
            [float(item["wrong_depth_error"] - item["current_error"]) for item in lexical]
        ),
        "wrong_role_margin": basic(
            [float(item["wrong_role_error"] - item["current_error"]) for item in lexical]
        ),
        "wrong_group_margin": basic(
            [float(item["wrong_group_error"] - item["current_error"]) for item in lexical]
        ),
        "history_gain": basic([float(item["history_gain"]) for item in lexical]),
        "error_threshold_counts": {
            "le_0_75": sum(value <= 0.75 for value in current),
            "le_0_90": sum(value <= 0.90 for value in current),
            "le_0_95": sum(value <= 0.95 for value in current),
            "le_0_99": sum(value <= 0.99 for value in current),
        },
    }


def main() -> None:
    all_rows = []
    by_model = {}
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_mechanism: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model in MODELS:
        rows = read_jsonl(
            DISCOVERY
            / "models"
            / model
            / "private/phase375_group_rows.jsonl"
        )
        all_rows.extend(rows)
        by_model[model] = summarize(rows)
        for row in rows:
            by_template[row["template"]].append(row)
            by_mechanism[row["mechanism_id"]].append(row)
    overall = summarize(all_rows)
    absolute_values = overall["current_error"]
    interpretation = {
        "frozen_phase375_gate_result": "rejected_all_templates",
        "dominant_observation": (
            "exact_multi_vector_spans_remain_nearly_orthogonal_to_the_next_generation_"
            "receiver_difference"
        ),
        "supported": [
            "adding_three_to_six_local_exact_vectors_does_not_make_the_current_"
            "linear_future_readout_sufficient",
            "failure_is_shared_by_all_three_models_and_both_mechanisms",
            "no_discovery_causal_replay_is_authorized_under_the_frozen_protocol",
        ],
        "not_supported": [
            "finite_physical_subgraphs_do_not_exist",
            "the_same_subgraphs_have_no_causal_task_effect",
            "a_local_subgraph_must_be_a_global_markov_state",
        ],
        "algorithmic_diagnosis": (
            "same_coordinate_projection_across_a_nonlinear_layer_transition_and_an_"
            "autoregressive_generation_step_is_not_a_valid_primary_causal_readout"
        ),
        "next_required_readout": (
            "preregistered_deterministic_transition_or_activation_swap_on_natural_"
            "subgraph_boundaries_without_predictive_prefiltering"
        ),
        "minimum_current_error_observed": absolute_values["minimum"],
    }
    summary = {
        "schema_version": "48.5.0",
        "phase_id": "Phase375-NegativeDiagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "posthoc": True,
        "gates_changed": False,
        "candidate_rescue_allowed": False,
        "denominator": {
            "group_candidate_count": len(all_rows),
            "lexical_evaluation_count": 2 * len(all_rows),
        },
        "overall": overall,
        "by_model": by_model,
        "by_template": {
            name: summarize(rows) for name, rows in sorted(by_template.items())
        },
        "by_mechanism": {
            name: summarize(rows) for name, rows in sorted(by_mechanism.items())
        },
        "interpretation": interpretation,
        "authorization": {
            "revive_phase375_candidate": False,
            "run_phase375_model_intervention": False,
            "design_new_causal_readout_protocol": True,
            "open_calibration": False,
            "open_physical": False,
        },
    }
    path = OUT / "phase375_discovery/phase375_negative_result_diagnostic.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
