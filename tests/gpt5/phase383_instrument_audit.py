#!/usr/bin/env python3
"""Audit Phase383 target-decision instruments before opening discovery."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase383_exact_component_event_map"
COLLECTION = OUT / "collection"
MODELS = ("qwen3", "glm4", "deepseek7b")
THRESHOLD = 0.01


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    protocol = read_json(OUT / "phase383_protocol.json")
    expected_dtypes = protocol["instrument_contract"]["runtime_dtype_by_model"]
    expected_per_model = (
        protocol["frozen_denominator"]["split_group_counts_per_mechanism"][
            "instrument_audit"
        ]
        * len(protocol["frozen_denominator"]["mechanisms"])
        * len(protocol["frozen_denominator"]["conditions"])
    )
    manifests = [
        read_json(
            COLLECTION
            / "instrument_audit/models"
            / model
            / "manifest.json"
        )
        for model in MODELS
    ]
    model_rows = []
    exact_attention_head_source_event_count = 0
    exact_mlp_channel_event_count = 0
    for manifest in manifests:
        layer_files = [row for row in manifest["files"] if row["kind"] == "layer"]
        for row in layer_files:
            payload = torch.load(
                COLLECTION / row["relative_path"],
                map_location="cpu",
                weights_only=True,
            )
            attention = payload["attention"]
            mlp = payload["mlp"]
            receiver_count = len(payload["role_names"])
            source_count = int(attention["value_states_all_sources"].shape[2])
            exact_attention_head_source_event_count += (
                receiver_count * source_count * int(attention["head_count"])
            )
            exact_mlp_channel_event_count += (
                receiver_count * int(mlp["channel_count"])
            )
        all_maxima_pass = all(
            float(value) <= THRESHOLD for value in manifest["gate_maxima"].values()
        )
        model_rows.append(
            {
                "model": manifest["model"],
                "case_count": manifest["case_count"],
                "layer_count": manifest["layer_count"],
                "runtime_dtype": manifest.get("runtime_dtype"),
                "baseline_replay_match_count": manifest[
                    "baseline_replay_match_count"
                ],
                "gate_maxima": manifest["gate_maxima"],
                "all_maxima_pass": all_maxima_pass,
                "manifest_valid": manifest["valid"],
            }
        )
    valid = all(
        row["case_count"] == expected_per_model
        and row["runtime_dtype"] == expected_dtypes[row["model"]]
        and row["baseline_replay_match_count"] == expected_per_model
        and row["all_maxima_pass"]
        and row["manifest_valid"]
        for row in model_rows
    )
    summary = {
        "schema_version": "57.2.0",
        "phase_id": "Phase383-InstrumentAudit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "model_count": len(MODELS),
            "case_count": sum(row["case_count"] for row in model_rows),
            "layer_observation_count": sum(
                row["case_count"] * row["layer_count"] for row in model_rows
            ),
            "exact_lazy_attention_head_source_event_count": (
                exact_attention_head_source_event_count
            ),
            "exact_lazy_mlp_channel_event_count": exact_mlp_channel_event_count,
        },
        "frozen_thresholds": {
            "component_relative_error_max": THRESHOLD,
            "probability_sum_error_max": THRESHOLD,
            "retuned_after_observation": False,
        },
        "models": model_rows,
        "results": {
            "baseline_replay_match_count": sum(
                row["baseline_replay_match_count"] for row in model_rows
            ),
            "all_three_model_instruments_valid": valid,
            "top_k_used": False,
            "attention_head_source_events_exactly_replayable": valid,
            "mlp_channel_events_exactly_replayable": valid,
        },
        "claim_boundary": {
            "instrument_valid_is_language_mechanism": False,
            "instrument_valid_is_upstream_layout": False,
            "four_receiver_roles_cover_all_token_positions": False,
            "event_families_are_lazy_exact_not_materialized_top_k": True,
        },
        "authorization": {
            "discovery_collection": valid,
            "calibration_collection": False,
            "physical_holdout_collection": False,
            "causal_intervention": False,
        },
        "next_decision": (
            "collect_balanced_discovery_target_decision_events"
            if valid
            else "stop_and_repair_instrument"
        ),
    }
    write_json(OUT / "phase383_instrument_audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
