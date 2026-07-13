#!/usr/bin/env python3
"""Parse Phase407 event times and evaluate independent functional gates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase407_event_horizon_protocol import (  # noqa: E402
    FAMILIES,
    HISTORY_MODES,
    INTERFACES,
    MODELS,
    OUT,
    SPLIT_GROUP_COUNTS,
    STATE_IDS,
    SURFACE_REPLICAS,
    semantic_transition_table,
)


SEQUENCE_GROUP_MIN = {
    "knowledge_binding": 56,
    "rule_reasoning": 28,
    "grammar_constraint": 56,
}
COMPLETION_GROUP_MIN = {
    "knowledge_binding": 48,
    "rule_reasoning": 24,
    "grammar_constraint": 48,
}
MODEL_FAMILY_GROUP_MIN = {
    "discovery": 9,
    "calibration": 5,
    "behavioral_holdout": 5,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
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


def _first_match(
    text: str, patterns: list[tuple[str, re.Pattern[str]]]
) -> tuple[str | None, str, int | None, int | None]:
    for method, pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1), method, match.start(1), match.end(1)
    return None, "no_match", None, None


def parse_semantic_state(text: str, row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one interface-specific literal response to a frozen state."""

    family = row["family_id"]
    lowered = text.lower()
    if family == "knowledge_binding":
        both = re.search(r"\bboth\s+(green|yellow)\b", lowered)
        if both:
            state = "green_green" if both.group(1) == "green" else "yellow_yellow"
            return {
                "semantic_state_private": state,
                "semantic_parse_method": "both_color",
                "semantic_span_start_private": both.start(1),
                "semantic_span_end_private": both.end(1),
                "semantic_parse_ambiguous": False,
            }
        colors = list(re.finditer(r"\b(green|yellow)\b", lowered))
        if len(colors) >= 2:
            state = f"{colors[0].group(1)}_{colors[1].group(1)}"
            return {
                "semantic_state_private": state,
                "semantic_parse_method": "ordered_color_pair",
                "semantic_span_start_private": colors[0].start(1),
                "semantic_span_end_private": colors[1].end(1),
                "semantic_parse_ambiguous": len(colors) > 2
                and any(
                    item.group(1) != colors[index % 2].group(1)
                    for index, item in enumerate(colors[2:])
                ),
            }
        return {
            "semantic_state_private": None,
            "semantic_parse_method": "incomplete_color_pair",
            "semantic_span_start_private": None,
            "semantic_span_end_private": None,
            "semantic_parse_ambiguous": False,
        }

    interface = row["interface_private"]
    aliases = row["semantic_aliases_by_state_private"]
    alias_to_state = {
        alias.lower(): state
        for state, values in aliases.items()
        for alias in values
    }
    if family == "rule_reasoning" and interface == "truth_condition":
        raw, method, start, end = _first_match(
            text,
            [
                (
                    "response_initial_truth",
                    re.compile(r"^\s*[\[({\"'`*_#:\-]*\s*(yes|no|true|false)\b", re.I),
                ),
                (
                    "answer_slot_truth",
                    re.compile(r"\b(?:answer|result|conclusion)\s*(?:is|:)\s*[\"'`]*\s*(yes|no|true|false)\b", re.I),
                ),
            ],
        )
    elif family == "rule_reasoning":
        entity_a = re.escape(aliases["holder_a"][-1])
        entity_b = re.escape(aliases["holder_b"][-1])
        raw, method, start, end = _first_match(
            text,
            [
                (
                    "response_initial_label",
                    re.compile(r"^\s*[\[({\"'`*_#:\-]*\s*([AB])(?:\b|[.])", re.I),
                ),
                ("person_label", re.compile(r"\bperson\s+([AB])\b", re.I)),
                (
                    "answer_slot_label",
                    re.compile(r"\b(?:answer|person|label|result)\s*(?:is|:)\s*[\"'`]*\s*([AB])\b", re.I),
                ),
                ("entity_a", re.compile(rf"\b({entity_a})\b", re.I)),
                ("entity_b", re.compile(rf"\b({entity_b})\b", re.I)),
            ],
        )
    else:
        raw, method, start, end = _first_match(
            text,
            [
                (
                    "response_initial_be_form",
                    re.compile(r"^\s*[\[({\"'`*_#:\-]*\s*(is|are|was|were)\b", re.I),
                ),
                (
                    "required_form_slot",
                    re.compile(r"\b(?:form|word|answer|auxiliary)\s*(?:is|:|should\s+be|would\s+be)\s*[\"'`]*\s*(is|are|was|were)\b", re.I),
                ),
                (
                    "quoted_be_form",
                    re.compile(r"[\"'`]\s*(is|are|was|were)\s*[\"'`]", re.I),
                ),
            ],
        )

    state = alias_to_state.get(raw.lower()) if raw is not None else None
    return {
        "semantic_state_private": state,
        "semantic_parse_method": method,
        "semantic_span_start_private": start,
        "semantic_span_end_private": end,
        "semantic_parse_ambiguous": False,
    }


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    observations = []
    for step in row["step_ledger_private"]:
        parsed = parse_semantic_state(step["decoded_prefix_private"], row)
        if parsed["semantic_state_private"] is not None:
            observations.append(
                {
                    "step": step["step"],
                    "prefix": step["decoded_prefix_private"],
                    **parsed,
                }
            )
    first = observations[0] if observations else None
    first_state = first["semantic_state_private"] if first else None
    distinct_states = list(
        dict.fromkeys(item["semantic_state_private"] for item in observations)
    )
    semantic_parse_ambiguous = any(
        item["semantic_parse_ambiguous"] for item in observations
    )
    semantic_reversal = len(distinct_states) > 1 or semantic_parse_ambiguous
    target = row["target_semantic_state_private"]
    semantic_correct = (
        first_state == target
        and not semantic_reversal
        and row["all_generated_step_logits_valid"]
    )
    tau_semantic = first["step"] if first else None
    tau_target_semantic = next(
        (
            item["step"]
            for item in observations
            if item["semantic_state_private"] == target
        ),
        None,
    )
    tau_boundary = None
    if first is not None:
        for item in observations:
            if item["step"] < first["step"]:
                continue
            suffix = item["prefix"][item["semantic_span_end_private"] :]
            if re.search(r"[.!?\n]", suffix):
                tau_boundary = item["step"]
                break
    tau_stop = row["eos_step_private"]
    boundary_after_correct_semantic = semantic_correct and tau_boundary is not None
    complete_response = semantic_correct and (
        tau_boundary is not None or tau_stop is not None
    )
    return {
        **row,
        "normalized_semantic_state_private": first_state,
        "semantic_parse_method": (
            first["semantic_parse_method"] if first else "no_complete_state"
        ),
        "semantic_observed_state_sequence_private": distinct_states,
        "semantic_parse_ambiguous": semantic_parse_ambiguous,
        "semantic_reversal": semantic_reversal,
        "semantic_correct": semantic_correct,
        "tau_semantic_private": tau_semantic,
        "tau_target_semantic_private": tau_target_semantic,
        "tau_boundary_private": tau_boundary,
        "tau_stop_private": tau_stop,
        "semantic_right_censored_at_H48": tau_semantic is None
        and row["H48_right_edge_reached"],
        "boundary_right_censored_at_H48": tau_boundary is None
        and row["H48_right_edge_reached"],
        "stop_right_censored_at_H48": tau_stop is None
        and row["H48_right_edge_reached"],
        "boundary_after_correct_semantic": boundary_after_correct_semantic,
        "complete_response": complete_response,
        "tokens_after_boundary_private": (
            row["generated_token_count"] - tau_boundary
            if tau_boundary is not None
            else None
        ),
    }


def _selected(
    rows: list[dict[str, Any]], **conditions: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if all(row[key] == value for key, value in conditions.items())
    ]


def _surface_pair_count(
    rows: list[dict[str, Any]], pair_key: str, pair_values: tuple[str, str], **fixed: str
) -> int:
    count = 0
    for surface in SURFACE_REPLICAS:
        selected = _selected(rows, surface_id_private=surface["surface_id"], **fixed)
        values = {row[pair_key]: row for row in selected}
        if len(selected) == 2 and all(
            value in values and values[value]["semantic_correct"]
            for value in pair_values
        ):
            count += 1
    return count


def operator_group_audit(
    rows: list[dict[str, Any]], family: str
) -> dict[str, Any]:
    edge_rows = []
    for edge in semantic_transition_table()[family]:
        condition_passes = []
        for interface in INTERFACES[family]:
            for history in HISTORY_MODES:
                paired_surface_count = 0
                for surface in SURFACE_REPLICAS:
                    source = _selected(
                        rows,
                        state_id_private=edge["source"],
                        interface_private=interface,
                        history_mode_private=history,
                        surface_id_private=surface["surface_id"],
                    )
                    target = _selected(
                        rows,
                        state_id_private=edge["target"],
                        interface_private=interface,
                        history_mode_private=history,
                        surface_id_private=surface["surface_id"],
                    )
                    if (
                        len(source) == 1
                        and len(target) == 1
                        and source[0]["semantic_correct"]
                        and target[0]["semantic_correct"]
                    ):
                        paired_surface_count += 1
                condition_passes.append(paired_surface_count >= 3)
        edge_rows.append(
            {
                **edge,
                "condition_pass_count": sum(condition_passes),
                "condition_count": len(condition_passes),
                "edge_pass": all(condition_passes),
            }
        )
    return {
        "direct_endpoint_edge_pass_count": sum(row["edge_pass"] for row in edge_rows),
        "direct_endpoint_edge_count": len(edge_rows),
        "direct_endpoint_operator_group_pass": all(row["edge_pass"] for row in edge_rows),
        "direct_endpoint_edges": edge_rows,
    }


def group_audit(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    expected = (
        len(STATE_IDS[family])
        * len(SURFACE_REPLICAS)
        * len(INTERFACES[family])
        * len(HISTORY_MODES)
    )
    surface_units = []
    for state in STATE_IDS[family]:
        for interface in INTERFACES[family]:
            for history in HISTORY_MODES:
                selected = _selected(
                    rows,
                    state_id_private=state,
                    interface_private=interface,
                    history_mode_private=history,
                )
                count = sum(row["semantic_correct"] for row in selected)
                surface_units.append(
                    {
                        "state": state,
                        "interface": interface,
                        "history": history,
                        "semantic_correct_count": count,
                        "case_count": len(selected),
                        "unit_pass": len(selected) == 4 and count >= 3,
                    }
                )
    surface_group_pass = all(row["unit_pass"] for row in surface_units)

    interface_units = []
    for state in STATE_IDS[family]:
        for history in HISTORY_MODES:
            paired = _surface_pair_count(
                rows,
                "interface_private",
                INTERFACES[family],
                state_id_private=state,
                history_mode_private=history,
            )
            interface_units.append(
                {
                    "state": state,
                    "history": history,
                    "paired_surface_pass_count": paired,
                    "unit_pass": paired >= 3,
                }
            )
    interface_group_pass = all(row["unit_pass"] for row in interface_units)

    history_units = []
    for state in STATE_IDS[family]:
        for interface in INTERFACES[family]:
            paired = _surface_pair_count(
                rows,
                "history_mode_private",
                HISTORY_MODES,
                state_id_private=state,
                interface_private=interface,
            )
            history_units.append(
                {
                    "state": state,
                    "interface": interface,
                    "paired_surface_pass_count": paired,
                    "unit_pass": paired >= 3,
                }
            )
    history_group_pass = all(row["unit_pass"] for row in history_units)

    semantic_correct_count = sum(row["semantic_correct"] for row in rows)
    complete_response_count = sum(row["complete_response"] for row in rows)
    sequence_group_pass = (
        len(rows) == expected
        and semantic_correct_count >= SEQUENCE_GROUP_MIN[family]
    )
    completion_group_pass = (
        len(rows) == expected
        and complete_response_count >= COMPLETION_GROUP_MIN[family]
    )
    operator = operator_group_audit(rows, family)
    return {
        "case_count": len(rows),
        "expected_case_count": expected,
        "semantic_correct_count": semantic_correct_count,
        "required_semantic_correct_count": SEQUENCE_GROUP_MIN[family],
        "complete_response_count": complete_response_count,
        "required_complete_response_count": COMPLETION_GROUP_MIN[family],
        "eos_observed_count": sum(row["eos_observed"] for row in rows),
        "boundary_observed_count": sum(
            row["tau_boundary_private"] is not None for row in rows
        ),
        "semantic_reversal_count": sum(row["semantic_reversal"] for row in rows),
        "canonical_target_preferred_count": sum(
            row["canonical_target_preferred_to_foil"] for row in rows
        ),
        "surface_group_pass": surface_group_pass,
        "interface_group_pass": interface_group_pass,
        "history_group_pass": history_group_pass,
        "sequence_group_pass": sequence_group_pass,
        "completion_group_pass": completion_group_pass,
        "surface_units": surface_units,
        "interface_units": interface_units,
        "history_units": history_units,
        **operator,
    }


def model_family_audit(
    rows: list[dict[str, Any]], family: str, split: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["anonymous_parallel_group_id"]].append(row)
    groups = []
    for group_id, selected in sorted(grouped.items()):
        groups.append(
            {
                "anonymous_parallel_group_id": group_id,
                **group_audit(selected, family),
            }
        )
    required = MODEL_FAMILY_GROUP_MIN[split]
    gate_counts = {
        gate: sum(group[f"{gate}_group_pass"] for group in groups)
        for gate in ("surface", "interface", "history", "sequence", "completion")
    }
    gate_passes = {
        gate: len(groups) == SPLIT_GROUP_COUNTS[split] and count >= required
        for gate, count in gate_counts.items()
    }
    semantic_candidate = all(
        gate_passes[gate]
        for gate in ("surface", "interface", "history", "sequence")
    )
    complete_candidate = semantic_candidate and gate_passes["completion"]
    operator_group_pass_count = sum(
        group["direct_endpoint_operator_group_pass"] for group in groups
    )
    operator_candidate = semantic_candidate and operator_group_pass_count >= required
    summary = {
        "family_id": family,
        "case_count": len(rows),
        "semantic_correct_count": sum(row["semantic_correct"] for row in rows),
        "complete_response_count": sum(row["complete_response"] for row in rows),
        "eos_observed_count": sum(row["eos_observed"] for row in rows),
        "semantic_right_censored_count": sum(
            row["semantic_right_censored_at_H48"] for row in rows
        ),
        "stop_right_censored_count": sum(
            row["stop_right_censored_at_H48"] for row in rows
        ),
        "canonical_target_preferred_count": sum(
            row["canonical_target_preferred_to_foil"] for row in rows
        ),
        "group_count": len(groups),
        "required_group_pass_count": required,
        "gate_group_pass_counts": gate_counts,
        "gate_model_family_pass": gate_passes,
        "semantic_state_candidate": semantic_candidate,
        "complete_generation_candidate": complete_candidate,
        "direct_endpoint_operator_group_pass_count": operator_group_pass_count,
        "direct_endpoint_operator_candidate": operator_candidate,
        "operator_is_model_executed_instruction": False,
    }
    return summary, groups


def authorized_families(stage: str) -> tuple[str, ...]:
    if stage == "discovery":
        return FAMILIES
    prior = {
        "calibration": "phase407_discovery_analysis.json",
        "behavioral_holdout": "phase407_calibration_analysis.json",
    }[stage]
    return tuple(
        read_json(OUT / prior)["strict_crossmodel_semantic_candidate_families"]
    )


def pair_candidates(
    rows: list[dict[str, Any]], models: tuple[str, str]
) -> list[str]:
    result = []
    for family in FAMILIES:
        selected = [
            row
            for row in rows
            if row["family_id"] == family and row["model"] in models
        ]
        if len(selected) == 2 and all(
            row["semantic_state_candidate"] for row in selected
        ):
            result.append(family)
    return result


def main(stage: str) -> None:
    families = authorized_families(stage)
    summaries = []
    group_details = []
    all_enriched = []
    for model in MODELS:
        complete = read_json(OUT / "collection" / stage / model / "complete.json")
        if not complete.get("valid"):
            raise RuntimeError(f"Invalid Phase407 collection {model}/{stage}")
        raw_rows = read_jsonl(
            OUT / "collection" / stage / "private" / model / "rows.jsonl"
        )
        enriched = [enrich_row(row) for row in raw_rows]
        all_enriched.extend(enriched)
        write_jsonl(
            OUT / "analysis" / stage / "private" / model / "semantic_rows.jsonl",
            enriched,
        )
        for family in families:
            selected = [row for row in enriched if row["family_id"] == family]
            summary, groups = model_family_audit(selected, family, stage)
            summaries.append({"model": model, **summary})
            group_details.extend(
                {
                    "model": model,
                    "family_id": family,
                    "stage": stage,
                    **group,
                }
                for group in groups
            )

    strict_semantic = []
    strict_complete = []
    strict_endpoint_operator = []
    for family in families:
        selected = [row for row in summaries if row["family_id"] == family]
        if len(selected) == len(MODELS) and all(
            row["semantic_state_candidate"] for row in selected
        ):
            strict_semantic.append(family)
        if len(selected) == len(MODELS) and all(
            row["complete_generation_candidate"] for row in selected
        ):
            strict_complete.append(family)
        if len(selected) == len(MODELS) and all(
            row["direct_endpoint_operator_candidate"] for row in selected
        ):
            strict_endpoint_operator.append(family)

    payload = {
        "schema_version": "81.3.0",
        "phase_id": "Phase407-EventHorizonAnalysis",
        "created_at": now(),
        "stage": stage,
        "models": list(MODELS),
        "authorized_families": list(families),
        "case_count": len(all_enriched),
        "semantic_correct_count": sum(
            row["semantic_correct"] for row in all_enriched
        ),
        "complete_response_count": sum(
            row["complete_response"] for row in all_enriched
        ),
        "semantic_reversal_count": sum(
            row["semantic_reversal"] for row in all_enriched
        ),
        "eos_observed_count": sum(row["eos_observed"] for row in all_enriched),
        "semantic_right_censored_count": sum(
            row["semantic_right_censored_at_H48"] for row in all_enriched
        ),
        "stop_right_censored_count": sum(
            row["stop_right_censored_at_H48"] for row in all_enriched
        ),
        "canonical_target_preferred_count": sum(
            row["canonical_target_preferred_to_foil"] for row in all_enriched
        ),
        "model_family_rows": summaries,
        "single_model_semantic_candidate_families": {
            model: [
                row["family_id"]
                for row in summaries
                if row["model"] == model and row["semantic_state_candidate"]
            ]
            for model in MODELS
        },
        "glm4_pair_semantic_candidate_families": {
            "qwen3_glm4": pair_candidates(summaries, ("qwen3", "glm4")),
            "glm4_deepseek7b": pair_candidates(
                summaries, ("glm4", "deepseek7b")
            ),
        },
        "strict_crossmodel_semantic_candidate_families": strict_semantic,
        "strict_crossmodel_complete_generation_families": strict_complete,
        "strict_crossmodel_direct_endpoint_operator_families": strict_endpoint_operator,
        "authorization": {
            "run_calibration": stage == "discovery" and bool(strict_semantic),
            "run_behavioral_holdout": stage == "calibration"
            and bool(strict_semantic),
            "promote_direct_endpoint_operator": stage != "discovery"
            and bool(strict_endpoint_operator),
            "run_physical_mapping": False,
            "run_causal_intervention": False,
            "run_neuron_scan": False,
        },
        "claim_boundary": {
            "surface_interface_history_sequence_gates_are_independent": True,
            "completion_gate_is_separate_from_semantic_state_gate": True,
            "eos_is_not_required_for_sentence_boundary_completion": True,
            "direct_endpoint_operator_is_model_executed_instruction": False,
            "finite_condition_state_is_causal_state": False,
        },
    }
    write_jsonl(
        OUT / "analysis" / f"phase407_{stage}_group_details.jsonl",
        group_details,
    )
    write_json(OUT / f"phase407_{stage}_analysis.json", payload)
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
