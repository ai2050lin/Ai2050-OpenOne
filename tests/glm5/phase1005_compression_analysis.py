#!/usr/bin/env python3
"""Aggregate the preregistered Phase1005 compression search."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PHASE = 1005
ROOT = (
    Path(__file__).resolve().parent
    / "result"
    / "phase1005_blind_layerwise_source_compression"
)
OUT_ROOT = ROOT / "analysis"
RUNS = (
    ("8bit", "qwen3"),
    ("8bit", "glm4"),
    ("8bit", "deepseek7b"),
    ("bf16", "qwen3"),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                + "\n"
            )


def maximum(values: Iterable[float]) -> float | None:
    values = list(values)
    return max(values) if values else None


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(statistics.fmean(values)) if values else None


def dominant_role(audit: dict[str, Any]) -> str:
    rates = audit["role_match_rate"]
    if not rates:
        return "unmatched"
    best = max(rates.values())
    roles = sorted(role for role, rate in rates.items() if rate == best)
    return "+".join(roles)


def main() -> None:
    model_rows = []
    domain_rows = []
    event_rows = []
    discovery_interventions = 0
    confirmation_interventions = 0
    for precision, model in RUNS:
        model_summary = read_json(
            ROOT / precision / model / "summary.json"
        )
        model_rows.append({
            "precision": precision,
            "model": model,
            "parent_qualified_domain_count": int(
                model_summary["parent_qualified_domain_count"]
            ),
            "compressed_domain_count": int(
                model_summary["compressed_domain_count"]
            ),
            "compressed_event_pass_count": int(
                model_summary["compressed_event_pass_count"]
            ),
            "elapsed_seconds": float(
                model_summary["elapsed_seconds"]
            ),
        })
        for domain, value in model_summary["domains"].items():
            row = {
                "schema_version": (
                    "phase1005_compression_domain_aggregate.v1"
                ),
                "phase": PHASE,
                "precision": precision,
                "model": model,
                "domain": domain,
                "status": value["status"],
                "parent_qualified": value["status"] == "complete",
                "compressed_event_pass_count": int(
                    value["compressed_event_pass_count"]
                ),
                "compressed_single_position_found": bool(
                    value["compressed_single_position_found"]
                ),
            }
            if value["status"] == "complete":
                domain_root = ROOT / precision / model / domain
                discovery_count = count_jsonl(
                    domain_root / "discovery_rows.jsonl"
                )
                confirmation_count = count_jsonl(
                    domain_root / "confirmation_rows.jsonl"
                )
                discovery_interventions += discovery_count
                confirmation_interventions += confirmation_count
                row.update({
                    "discovery_n": int(value["discovery_n"]),
                    "confirmation_n": int(value["confirmation_n"]),
                    "event_universe_count": int(
                        value["event_universe_count"]
                    ),
                    "frozen_event_count": int(
                        value["frozen_event_count"]
                    ),
                    "discovery_intervention_row_count": (
                        discovery_count
                    ),
                    "confirmation_intervention_row_count": (
                        confirmation_count
                    ),
                    "maximum_frozen_discovery_donor_rate": maximum(
                        event["discovery_metrics"]["donor_rate"]
                        for event in value["frozen_events"]
                    ),
                    "maximum_frozen_confirmation_donor_rate": maximum(
                        event["confirmation_different_answer"][
                            "donor_rate"
                        ]
                        for event in value["frozen_events"]
                    ),
                    "minimum_frozen_noop_target_rate": min(
                        event["confirmation_target_noop"][
                            "target_rate"
                        ]
                        for event in value["frozen_events"]
                    ),
                })
                for event in value["frozen_events"]:
                    event_rows.append({
                        "schema_version": (
                            "phase1005_compression_event_aggregate.v1"
                        ),
                        "phase": PHASE,
                        "precision": precision,
                        "model": model,
                        "domain": domain,
                        "event_id": event["event_id"],
                        "discovery_rank": int(
                            event["discovery_rank"]
                        ),
                        "relative_depth": float(
                            event["relative_depth"]
                        ),
                        "end_offset": int(event["end_offset"]),
                        "discovery_donor_rate": float(
                            event["discovery_metrics"]["donor_rate"]
                        ),
                        "discovery_median_transfer": float(
                            event["discovery_metrics"][
                                "median_normalized_transfer"
                            ]
                        ),
                        "confirmation_donor_rate": float(
                            event["confirmation_different_answer"][
                                "donor_rate"
                            ]
                        ),
                        "confirmation_median_transfer": float(
                            event["confirmation_different_answer"][
                                "median_normalized_transfer"
                            ]
                        ),
                        "confirmation_same_answer_target_rate": float(
                            event["confirmation_same_answer"][
                                "target_rate"
                            ]
                        ),
                        "confirmation_noop_target_rate": float(
                            event["confirmation_target_noop"][
                                "target_rate"
                            ]
                        ),
                        "compressed_event_gate_pass": bool(
                            event["compressed_event_gate_pass"]
                        ),
                        "dominant_revealed_role": dominant_role(
                            event[
                                "semantic_reconstruction_audit"
                            ]
                        ),
                        "role_match_rate": event[
                            "semantic_reconstruction_audit"
                        ]["role_match_rate"],
                    })
            domain_rows.append(row)

    qualified = [row for row in domain_rows if row["parent_qualified"]]
    role_counts = Counter(
        event["dominant_revealed_role"] for event in event_rows
    )
    top_events = [
        min(
            (
                event
                for event in event_rows
                if (
                    event["precision"] == domain["precision"]
                    and event["model"] == domain["model"]
                    and event["domain"] == domain["domain"]
                )
            ),
            key=lambda event: event["discovery_rank"],
        )
        for domain in qualified
    ]

    phase1004_path = (
        ROOT.parent
        / "phase1004_blind_causal_state_basis"
        / "analysis"
        / "summary.json"
    )
    phase1004 = read_json(phase1004_path)
    summary = {
        "schema_version": "phase1005_compression_analysis.v1",
        "phase": PHASE,
        "protocol_revision": 2,
        "formal_run_count": len(RUNS),
        "parent_qualified_domain_count": len(qualified),
        "discovery_intervention_row_count": discovery_interventions,
        "confirmation_and_control_intervention_row_count": (
            confirmation_interventions
        ),
        "total_intervention_row_count": (
            discovery_interventions + confirmation_interventions
        ),
        "frozen_event_count": len(event_rows),
        "compressed_event_pass_count": sum(
            event["compressed_event_gate_pass"]
            for event in event_rows
        ),
        "compressed_domain_count": sum(
            row["compressed_single_position_found"]
            for row in qualified
        ),
        "maximum_discovery_donor_rate_over_frozen_events": maximum(
            event["discovery_donor_rate"] for event in event_rows
        ),
        "maximum_confirmation_donor_rate_over_frozen_events": maximum(
            event["confirmation_donor_rate"] for event in event_rows
        ),
        "mean_noop_target_rate_over_frozen_events": mean(
            event["confirmation_noop_target_rate"]
            for event in event_rows
        ),
        "dominant_revealed_role_counts": dict(
            sorted(role_counts.items())
        ),
        "top_event_per_qualified_domain": top_events,
        "model_summaries": model_rows,
        "domain_summaries": domain_rows,
        "combined_phase1004_phase1005_inference": {
            "phase1004_repeated_prompt_source_topology": (
                phase1004["evidence_classification"][
                    "repeated_label_blind_source_role_topology_found"
                ]
            ),
            "phase1004_cross_precision_late_answer_boundary_residual_chain": (
                phase1004["evidence_classification"][
                    "cross_precision_late_residual_chain_found"
                ]
            ),
            "phase1005_single_raw_prompt_position_compression": False,
            "cautious_inference": (
                "The tested state remains distributed over existing "
                "prompt positions. In Qwen3 color, the intervention "
                "effect later appears at the newly constructed "
                "answer-boundary residual state. This favors a "
                "temporal aggregation hypothesis over a static "
                "single-prompt-position hypothesis, but the "
                "answer-boundary chain is not yet cross-model."
            ),
        },
        "claim_boundary": {
            "supported": [
                "No scanned raw-prompt single-position event passed.",
                "The NO-GO repeats across three models, two domains, and BF16.",
                "Existing prompt-token state is not singly sufficient under this intervention.",
            ],
            "not_supported": [
                "No single neuron can be important.",
                "No small multi-position state exists.",
                "The answer-boundary state is the native unique mechanism.",
                "A complete language encoding formula has been found.",
            ],
        },
        "next_stage_decision": {
            "automatically_decompose_heads_or_neurons": False,
            "reason": (
                "Neither Phase1004 nor Phase1005 produced a repeated "
                "cross-model attention/MLP parent event."
            ),
            "next_scientific_target": (
                "Preregister a new held-out test of multi-position "
                "prompt state converging into the autoregressively "
                "created answer-boundary state, with full-layer "
                "transport, restoration, and natural-rollout gates."
            ),
            "requires_new_independent_holdout_before_execution": True,
        },
    }
    write_jsonl(OUT_ROOT / "event_rows.jsonl", event_rows)
    write_jsonl(OUT_ROOT / "domain_rows.jsonl", domain_rows)
    write_json(OUT_ROOT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
