#!/usr/bin/env python3
"""Split Phase416 instrument qualification by actual execution domain."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase416_dual_track_case_bank import (
    MODELS,
    OUT,
    PHASE_ID,
    SCHEMA_VERSION,
    read_json,
    read_jsonl,
    write_json,
)


PREFILL_GATES = (
    "direct_hook_logit_pass",
    "direct_hook_js_pass",
    "layer_output_pass",
    "component_ledger_pass",
    "checkpoint_replay_logit_pass",
    "checkpoint_replay_js_pass",
)
CACHE_GATES = (
    "chunked_cache_logit_pass",
    "chunked_cache_js_pass",
    "chunked_cache_top1_pass",
    "chunked_cache_shape_pass",
    "chunked_cache_value_pass",
)
GENERATION_GATES = (
    "greedy_token_pass",
    "greedy_score_pass",
    "finite_generation_pass",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gate_pass(row: dict[str, Any], names: tuple[str, ...]) -> bool:
    return all(row["gates"].get(name, False) for name in names)


def analyze() -> dict[str, Any]:
    model_rows = []
    total_prefill = 0
    total_cache = 0
    total_generation = 0
    behavior_cells = []
    for model in MODELS:
        root = OUT / "models" / model
        rows = read_jsonl(root / "phase416_collector_case_rows.jsonl")
        complete = read_json(root / "phase416_collector_complete.json")
        prefill = sum(gate_pass(row, PREFILL_GATES) for row in rows)
        cache = sum(gate_pass(row, CACHE_GATES) for row in rows)
        generation = sum(gate_pass(row, GENERATION_GATES) for row in rows)
        total_prefill += prefill
        total_cache += cache
        total_generation += generation
        behavior_cells.extend(complete["behavior_cells"])
        model_rows.append(
            {
                "model": model,
                "case_count": len(rows),
                "prefill_collector_pass_count": prefill,
                "prefill_collector_qualification_pass": prefill == len(rows) == 55,
                "incremental_cache_pass_count": cache,
                "incremental_cache_qualification_pass": cache == len(rows) == 55,
                "greedy_generation_pass_count": generation,
                "greedy_generation_qualification_pass": generation == len(rows) == 55,
                "target_behavior_pass_count": complete["target_behavior_pass_count"],
                "exact_answer_pass_count": complete["exact_answer_pass_count"],
                "qualified_formal_family_count": complete["qualified_formal_family_count"],
                "prefill_physical_collection_authorized": prefill == len(rows) == 55,
                "generation_time_physical_collection_authorized": False,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase416-InstrumentDomainQualification",
        "created_at": now(),
        "valid": all(row["case_count"] == 55 for row in model_rows),
        "assessment": {
            "phase414_measurement_ontology_correct": True,
            "external_review_should_block_formal_prefill_collection": False,
            "observer_qualification_should_block_unlabeled_prefill_collection": False,
            "single_collector_gate_across_prefill_cache_and_generation_valid": False,
            "prefill_and_incremental_execution_must_be_separate_instrument_domains": True,
        },
        "denominators": {
            "model_count": len(MODELS),
            "case_count_per_model": 55,
            "collector_case_count": 165,
            "prefill_gate_count": len(PREFILL_GATES),
            "cache_gate_count": len(CACHE_GATES),
            "generation_gate_count": len(GENERATION_GATES),
        },
        "results": {
            "prefill_collector_pass_count": total_prefill,
            "incremental_cache_pass_count": total_cache,
            "greedy_generation_pass_count": total_generation,
            "prefill_qualified_model_count": sum(row["prefill_collector_qualification_pass"] for row in model_rows),
            "incremental_cache_qualified_model_count": sum(row["incremental_cache_qualification_pass"] for row in model_rows),
            "greedy_generation_qualified_model_count": sum(row["greedy_generation_qualification_pass"] for row in model_rows),
            "qualified_formal_behavior_cell_count": sum(row["formal_behavior_qualified"] for row in behavior_cells),
            "formal_behavior_cell_count": len(behavior_cells),
        },
        "models": model_rows,
        "behavior_cells": behavior_cells,
        "authorization": {
            "collect_observer_free_prefill_physical_atlas": total_prefill == 165,
            "collect_generation_time_physical_atlas": False,
            "attach_functional_labels_only_to_behavior_qualified_formal_cells": True,
            "claim_natural_language_external_validity": False,
            "train_or_publish_qualified_observer": False,
            "run_causal_intervention": False,
            "run_neuron_scan": False,
        },
        "hard_limits": [
            "prefill qualification does not qualify KV-cached incremental execution",
            "formal target behavior does not establish natural-language external validity",
            "raw physical nodes receive no semantic-probability or causal labels",
            "DS7B knowledge behavior failed the frozen 0.75 target-event gate",
            "no generation-time stop or sentence-boundary topology may be inferred",
        ],
        "claim_boundary": "prefill_instrument_domain_only_not_generation_time_or_causal_mechanism",
    }


def main() -> None:
    summary = analyze()
    write_json(OUT / "phase416_instrument_domain_qualification.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
