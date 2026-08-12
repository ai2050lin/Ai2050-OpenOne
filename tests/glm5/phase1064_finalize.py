#!/usr/bin/env python3
"""Aggregate Phase1064 without promoting rate deltas to a scaling law."""

from __future__ import annotations

from collections import defaultdict
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1064_cross_panel_transport_protocol as protocol


def panel_delta(summary: dict, field: str, key: str) -> float:
    novel = summary["panel_results"]["novel_noun"][field][key]
    anchor = summary["panel_results"]["anchor_common"][field][key]
    return float(novel) - float(anchor)


def condition_records(model_name: str) -> dict[str, dict[int, bool]]:
    grouped: dict[str, dict[int, bool]] = defaultdict(dict)
    rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "atlas"
        / model_name
        / "condition_records.jsonl"
    )
    for row in rows:
        grouped[str(row["condition"])][
            int(row["target_index"])
        ] = bool(row["both_match_other_clean_text"])
    return dict(grouped)


def matched_selected_subset() -> dict:
    models = ("qwen3", "glm4")
    records = {
        model_name: condition_records(model_name)
        for model_name in models
    }
    conditions = sorted(
        set(records["qwen3"]) & set(records["glm4"])
    )
    output = {}
    for condition in conditions:
        common = sorted(
            set(records["qwen3"][condition])
            & set(records["glm4"][condition])
        )
        output[condition] = {
            "common_selected_target_count": len(common),
            "rates": {
                model_name: (
                    sum(
                        records[model_name][condition][index]
                        for index in common
                    )
                    / len(common)
                    if common else 0.0
                )
                for model_name in models
            },
        }
    return {
        "status": (
            "posthoc_descriptive_only_not_used_by_any_gate"
        ),
        "models": list(models),
        "conditions": output,
    }


def main() -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    summaries = {
        model_name: protocol.read_json(
            protocol.OUT_ROOT
            / "atlas"
            / model_name
            / "summary.json"
        )
        for model_name in protocol.MODELS
    }
    for model_name, summary in summaries.items():
        if summary["protocol_digest"] != prereg["protocol_digest"]:
            raise RuntimeError(
                f"protocol digest drift for {model_name}"
            )
        if summary["clean_replay_parity_rate"] < 1.0:
            raise RuntimeError(
                f"clean replay drift for {model_name}"
            )
    passing = [
        model_name
        for model_name, summary in summaries.items()
        if summary["cross_panel_transport_gate_passed"]
    ]
    repeated = (
        len(passing)
        >= prereg["gates"]["minimum_repeated_models"]
    )
    contrasts = {}
    for model_name, summary in summaries.items():
        if not summary["source_behavior_gate_passed"]:
            continue
        contrasts[model_name] = {
            "novel_minus_anchor_phrase": panel_delta(
                summary, "component_text_rates", "phrase"
            ),
            "novel_minus_anchor_color": panel_delta(
                summary, "component_text_rates", "color"
            ),
            "novel_minus_anchor_noun": panel_delta(
                summary, "component_text_rates", "noun"
            ),
            "novel_minus_anchor_early": panel_delta(
                summary, "phase_text_rates", "early"
            ),
            "novel_minus_anchor_post": panel_delta(
                summary, "phase_text_rates", "post"
            ),
            "novel_minus_anchor_all": panel_delta(
                summary, "phase_text_rates", "all"
            ),
            "novel_minus_anchor_k_only": panel_delta(
                summary, "channel_text_rates", "k_only"
            ),
            "novel_minus_anchor_v_only": panel_delta(
                summary, "channel_text_rates", "v_only"
            ),
            "novel_minus_anchor_kv": panel_delta(
                summary, "channel_text_rates", "kv"
            ),
        }
    if repeated:
        decision = {
            "route": "stop_at_cross_lexicon_transport_milestone",
            "should_continue_automatically": False,
            "rationale": (
                "Cross-panel transport repeated in two models. The next "
                "language pattern requires an independent protocol rather "
                "than automatic reuse of the translation intervention."
            ),
        }
    else:
        decision = {
            "route": "stop_with_cross_panel_transport_mismatch",
            "should_continue_automatically": False,
            "rationale": (
                "Fewer than two behavior-qualified models passed both "
                "transport panels; no cross-model mechanism claim is made."
            ),
        }
    aggregate = {
        "schema_version": "phase1064_aggregate.v1",
        "phase": protocol.PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": list(protocol.MODELS),
        "model_results": summaries,
        "passing_models": passing,
        "cross_model_cross_panel_repetition": repeated,
        "descriptive_panel_contrasts": contrasts,
        "matched_selected_subset_analysis": (
            matched_selected_subset()
        ),
        "contrast_warning": (
            "These paired rate differences are descriptive and are not "
            "a lexical difficulty scaling law."
        ),
        "automatic_next_decision": decision,
        "interpretation_limits": prereg["interpretation_limits"],
    }
    protocol.write_json(
        protocol.OUT_ROOT / "aggregate.json", aggregate
    )
    print(
        f"Phase{protocol.PHASE} finalized: repeated={repeated} "
        f"passing={passing} route={decision['route']}"
    )


if __name__ == "__main__":
    main()
