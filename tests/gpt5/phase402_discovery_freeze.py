#!/usr/bin/env python3
"""Freeze Phase402 discovery execution after the instrument gate passes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase402_multiparent_graph"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    protocol_path = OUT / "phase402_multiparent_protocol.json"
    behavior_path = OUT / "phase402_behavior_freeze_summary.json"
    trace_path = OUT / "phase402_trace_protocol.json"
    instrument_path = OUT / "phase402_instrument_audit.json"
    protocol = read_json(protocol_path)
    behavior = read_json(behavior_path)
    instrument = read_json(instrument_path)
    if not instrument["valid"] or not instrument["authorization"]["run_discovery"]:
        raise RuntimeError("Phase402 instrument gate did not authorize discovery")
    eligible = behavior["eligible_surfaces"]
    selected = protocol["fresh_behavior_denominator"][
        "selected_split_group_counts"
    ]["discovery"]
    payload = {
        "schema_version": "76.6.0",
        "phase_id": "Phase402-DiscoveryExecutionFreeze",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {
            "multiparent_protocol": digest(protocol_path),
            "behavior_freeze": digest(behavior_path),
            "trace_protocol": digest(trace_path),
            "instrument_audit": digest(instrument_path),
        },
        "discovery_denominator": {
            "eligible_surfaces": eligible,
            "groups_per_surface": selected,
            "conditions_per_group": 16,
            "models": protocol["models_in_execution_order"],
            "case_count": len(eligible) * selected * 16 * 3,
            "subset_count_including_empty": 16,
            "controls_including_true": protocol["control_applicability"],
        },
        "subset_control_applicability": {
            "empty_subset": "instrument_only_not_a_candidate",
            "true_relation": "all_nonempty_subsets",
            "same_target_wrong_order": "all_nonempty_subsets",
            "wrong_receiver_role": "all_nonempty_subsets",
            "wrong_semantic_time": "all_nonempty_subsets",
            "wrong_depth_quarter_shift": "all_nonempty_subsets",
            "source_content_role_permutation": (
                "only_subsets_containing_source_content"
            ),
            "same_content_wrong_structure": (
                "only_subsets_containing_source_structure"
            ),
            "deterministic_random_natural_donor": "all_nonempty_subsets",
            "same_absolute_mass_sign_permuted": (
                "all_nonempty_subsets_measurement_control_not_legal_kv_patch"
            ),
            "not_applicable_controls_are_recorded_and_excluded_from_the_gate": True,
        },
        "frozen_gates": {
            "pair_gate": protocol["pair_gate"],
            "group_layer_subset_gate": protocol["group_layer_subset_gate"],
            "discovery_candidate_gate": protocol["discovery_candidate_gate"],
        },
        "storage_contract": {
            "private_pair_metrics": "gzip_jsonl",
            "group_layer_subset_metrics": "jsonl",
            "raw_hidden_vectors_retained": False,
            "all_pair_scalar_metrics_retained": True,
        },
        "counterfactual_execution_contract": {
            "base_control_groups_per_vectorized_chunk": 16,
            "subsets_per_base_control_group": 16,
            "maximum_counterfactual_rows_per_projection_batch": 256,
            "same_chunk_shape_required_for_all_three_models": True,
            "smoke_outputs_are_not_formal_denominator": True,
        },
        "authorization": {
            "run_discovery_models_sequentially": True,
            "run_calibration": False,
            "run_physical_holdout": False,
            "run_propagation_or_terminal": False,
            "run_neuron_scan": False,
        },
        "claim_boundary": {
            "discovery_candidate_is_a_language_path": False,
            "remaining_prefix_is_generated_history": False,
            "synthetic_sign_control_is_a_compute_graph_parent_patch": False,
        },
    }
    write_json(OUT / "phase402_discovery_execution_freeze.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
