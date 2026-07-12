#!/usr/bin/env python3
"""Physically confirm decision-aligned terminal residual transfer in Phase378."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase371c_blind_vector_contrast import model_pairs  # noqa: E402
from phase376_decision_aligned_intervention import (  # noqa: E402
    CONDITIONS,
    TRANSFER_PAIRS,
    capture_event,
    condition_metrics,
    direction_pass,
    run_patch_batch,
)


OUT = ROOT / "tests/gpt5/result/phase378_physical_confirmation"
PHASE376 = ROOT / "tests/gpt5/result/phase376_decision_aligned_subgraphs"
PROTOCOL = OUT / "phase378_physical_protocol.json"
ANALYSIS = OUT / "phase378_physical_behavior_analysis_summary.json"
CASES = OUT / "private/phase378_physical_intervention_cases.jsonl"
FREEZE = OUT / "phase378_intervention_execution_freeze.json"
DISCOVERY_PROTOCOL = PHASE376 / "phase376_intervention_protocol.json"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
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
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )


def model_groups(model: str) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, str]]:
    rows = [row for row in read_jsonl(CASES) if row["private_execution_model"] == model]
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    mechanisms = {}
    for row in rows:
        parallel = row["anonymous_parallel_group_id"]
        groups[parallel][row["contrast_condition"].split("_", 1)[0]] = row
        mechanisms[parallel] = row["mechanism_id"]
    if len(rows) != 32 or len(groups) != 8 or any(
        set(values) != {"A", "B", "C", "D"} for values in groups.values()
    ):
        raise RuntimeError(f"Invalid Phase378 cases for {model}")
    return dict(groups), mechanisms


def process_model(model: str) -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    analysis = read_json(ANALYSIS)
    freeze = read_json(FREEZE)
    discovery_protocol = read_json(DISCOVERY_PROTOCOL)
    if not analysis["authorization"]["run_physical_interventions"] or not freeze["valid"]:
        raise RuntimeError("Physical intervention not authorized")
    templates = {
        name: discovery_protocol["natural_templates"][name]
        for name in protocol["scope"]["templates"]
    }
    physical_gate = protocol["intervention_gate"]
    gates = {
        "minimum_correct_transfer_gain": physical_gate["minimum_correct_transfer_gain"],
        "minimum_gain_over_wrong_depth": physical_gate["minimum_control_margin"],
        "minimum_gain_over_wrong_role": physical_gate["minimum_control_margin"],
        "minimum_gain_over_wrong_time_when_distinct": physical_gate[
            "minimum_control_margin"
        ],
        "minimum_independent_groups_per_model_mechanism_template": physical_gate[
            "minimum_common_groups_per_model_mechanism_template"
        ],
    }
    groups, mechanisms = model_groups(model)
    pair_rows, _ = model_pairs(model)
    depth_layers = {row["name"]: int(row["source_layer"]) for row in pair_rows}
    selected_layer = depth_layers["late"]
    wrong_layer = depth_layers["early"]
    loaded = None
    rows = []
    baseline_total = 0
    baseline_match = 0
    try:
        loaded = load_probe_model(model)
        for group_index, parallel in enumerate(sorted(groups), 1):
            cases = groups[parallel]
            natural = {}
            for condition, case in cases.items():
                decision = capture_event(loaded, case, "target_decision", depth_layers)
                entry = (
                    decision
                    if int(case["target_decision_step"]) == 0
                    else capture_event(loaded, case, "answer_entry", depth_layers)
                )
                target_token = int(
                    case["generated_token_ids"][int(case["target_decision_step"])]
                )
                baseline_total += 1
                baseline_match += int(decision["argmax_token_id"] == target_token)
                natural[condition] = {
                    "decision": decision,
                    "entry": entry,
                    "target_token": target_token,
                }
            for donor_condition, recipient_condition, transfer_name, transfer_class in TRANSFER_PAIRS:
                donor = cases[donor_condition]
                recipient = cases[recipient_condition]
                donor_natural = natural[donor_condition]
                recipient_natural = natural[recipient_condition]
                for template_name, template in templates.items():
                    patched, reached = run_patch_batch(
                        loaded,
                        recipient,
                        donor_natural["decision"],
                        donor_natural["entry"],
                        selected_layer,
                        wrong_layer,
                        template_name,
                        template,
                    )
                    condition_rows = {
                        condition: condition_metrics(
                            logits,
                            recipient_natural["decision"]["logits"],
                            donor_natural["target_token"],
                            recipient_natural["target_token"],
                        )
                        for condition, logits in patched.items()
                    }
                    row = {
                        "schema_version": "51.3.0",
                        "phase_id": "Phase378-PhysicalIntervention",
                        "model": model,
                        "mechanism_id": mechanisms[parallel],
                        "semantic_group_id": recipient["semantic_group_id"],
                        "anonymous_parallel_group_id": parallel,
                        "transfer": transfer_name,
                        "transfer_class": transfer_class,
                        "donor_condition": donor_condition,
                        "recipient_condition": recipient_condition,
                        "donor_decision_step": int(donor["target_decision_step"]),
                        "recipient_decision_step": int(recipient["target_decision_step"]),
                        "donor_target_token_id": donor_natural["target_token"],
                        "recipient_target_token_id": recipient_natural["target_token"],
                        "relative_depth": "late",
                        "selected_layer": selected_layer,
                        "wrong_depth_control": "early",
                        "wrong_layer_control": wrong_layer,
                        "template": template_name,
                        "baseline_replay_token_id": recipient_natural["decision"][
                            "argmax_token_id"
                        ],
                        "baseline_replay_matches_recipient_token": (
                            recipient_natural["decision"]["argmax_token_id"]
                            == recipient_natural["target_token"]
                        ),
                        "wrong_time_control_distinct": int(donor["target_decision_step"])
                        > 0,
                        "patch_reached": reached,
                        "conditions": condition_rows,
                        "winner_transfer_under_correct_patch": condition_rows["correct"][
                            "donor_token_is_argmax"
                        ],
                    }
                    row["direction_gate_pass"] = direction_pass(row, gates)
                    rows.append(row)
            print(f"[{model}] physical intervention groups {group_index}/8", flush=True)
        private_dir = OUT / "phase378_intervention/models" / model / "private"
        write_jsonl(private_dir / "phase378_intervention_rows.jsonl", rows)
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[
                (
                    row["mechanism_id"],
                    row["anonymous_parallel_group_id"],
                    row["template"],
                )
            ].append(row)
        group_gates = []
        pass_counts: Counter[tuple[str, str]] = Counter()
        winner_counts: Counter[tuple[str, str]] = Counter()
        for key, selected in sorted(grouped.items()):
            by_transfer = {row["transfer"]: row for row in selected}
            treatment = [by_transfer["A_to_C"], by_transfer["C_to_A"]]
            group_pass = all(row["direction_gate_pass"] for row in treatment)
            winner_both = all(row["winner_transfer_under_correct_patch"] for row in treatment)
            canonical = (key[0], key[2])
            if group_pass:
                pass_counts[canonical] += 1
            if group_pass and winner_both:
                winner_counts[canonical] += 1
            group_gates.append(
                {
                    "model": model,
                    "mechanism_id": key[0],
                    "anonymous_parallel_group_id": key[1],
                    "relative_depth": "late",
                    "template": key[2],
                    "group_gate_pass": group_pass,
                    "winner_flip_both_directions": winner_both,
                }
            )
        minimum = int(gates["minimum_independent_groups_per_model_mechanism_template"])
        candidates = [
            {
                "model": model,
                "mechanism_id": key[0],
                "relative_depth": "late",
                "template": key[1],
                "group_pass_count": count,
                "winner_flip_group_count": winner_counts[key],
                "minimum_group_count": minimum,
                "physical_transfer_gate_pass": count >= minimum,
                "physical_winner_flip_pass": winner_counts[key] >= minimum,
            }
            for key, count in sorted(pass_counts.items())
            if count >= minimum
        ]
        write_jsonl(private_dir / "phase378_group_gate_rows.jsonl", group_gates)
        write_jsonl(
            OUT / "phase378_intervention/models" / model / "phase378_model_candidates.jsonl",
            candidates,
        )
        summary = {
            "schema_version": "51.3.0",
            "phase_id": "Phase378-PhysicalIntervention",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "execution": {
                "device": str(loaded.input_device),
                "model_execution": True,
                "physical_opened": True,
                "other_mechanisms_opened": False,
            },
            "denominator": {
                "case_count": 32,
                "parallel_group_count": 8,
                "template_depth_transfer_count": len(rows),
                "patched_forward_condition_count": len(rows) * len(CONDITIONS),
                "group_gate_count": len(group_gates),
            },
            "quality": {
                "baseline_replay_count": baseline_total,
                "baseline_replay_match_count": baseline_match,
                "baseline_replay_match_rate": baseline_match / baseline_total,
                "all_patch_hooks_reached": all(
                    all(row["patch_reached"].values()) for row in rows
                ),
            },
            "results": {
                "model_candidate_count": len(candidates),
                "physical_winner_flip_candidate_count": sum(
                    row["physical_winner_flip_pass"] for row in candidates
                ),
                "model_candidates": candidates,
            },
            "claim_boundary": protocol["claim_boundary"],
        }
        write_json(
            OUT / "phase378_intervention/models" / model / "phase378_model_summary.json",
            summary,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def merge_models() -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    summaries = [
        read_json(
            OUT / "phase378_intervention/models" / model / "phase378_model_summary.json"
        )
        for model in MODELS
    ]
    canonical: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        for row in summary["results"]["model_candidates"]:
            canonical[(row["mechanism_id"], row["template"])].append(row)
    cross_rows = []
    for key, rows in sorted(canonical.items()):
        models = {row["model"] for row in rows}
        winner_models = {
            row["model"] for row in rows if row["physical_winner_flip_pass"]
        }
        level2 = "glm4" in winner_models and bool(
            winner_models & {"qwen3", "deepseek7b"}
        )
        level3 = winner_models == set(MODELS)
        cross_rows.append(
            {
                "mechanism_id": key[0],
                "relative_depth": "late",
                "template": key[1],
                "physical_transfer_models": sorted(models),
                "physical_winner_flip_models": sorted(winner_models),
                "heterogeneous_level2_physical_pass": level2,
                "level3_physical_pass": level3,
                "evidence_class": (
                    "physically_confirmed_terminal_content_carrier"
                    if level2
                    else "model_specific_terminal_content_carrier"
                ),
                "language_mechanism_claimed": False,
            }
        )
    physical = [row for row in cross_rows if row["heterogeneous_level2_physical_pass"]]
    summary = {
        "schema_version": "51.4.0",
        "phase_id": "Phase378-PhysicalMerge",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "model_count": 3,
            "physical_case_count": 96,
            "template_depth_transfer_count": sum(
                row["denominator"]["template_depth_transfer_count"] for row in summaries
            ),
            "patched_forward_condition_count": sum(
                row["denominator"]["patched_forward_condition_count"] for row in summaries
            ),
            "canonical_candidate_count": len(cross_rows),
        },
        "quality": {
            "behavior_strict_correct_count": 96,
            "baseline_replay_match_counts": {
                row["model"]: row["quality"]["baseline_replay_match_count"]
                for row in summaries
            },
            "all_patch_hooks_reached": all(
                row["quality"]["all_patch_hooks_reached"] for row in summaries
            ),
            "failed_groups_replaced": False,
        },
        "results": {
            "physically_confirmed_terminal_carrier_count": len(physical),
            "level3_terminal_carrier_count": sum(
                row["level3_physical_pass"] for row in cross_rows
            ),
            "upstream_encoding_rule_count": 0,
            "natural_necessity_count": 0,
            "full_generation_sufficiency_count": 0,
            "language_path_candidate_count": 0,
            "language_mechanism_claimed": False,
        },
        "model_results": [
            {
                "model": row["model"],
                "model_candidate_count": row["results"]["model_candidate_count"],
                "physical_winner_flip_candidate_count": row["results"][
                    "physical_winner_flip_candidate_count"
                ],
            }
            for row in summaries
        ],
        "cross_model_rows": cross_rows,
        "claim_boundary": protocol["claim_boundary"],
        "next_stage": {
            "priority": "trace_where_terminal_content_residual_is_formed_before_late_readout",
            "continue_terminal_residual_swaps": False,
            "open_other_language_families": False,
            "single_neuron_scan": False,
        },
    }
    write_jsonl(OUT / "phase378_intervention/phase378_cross_model_rows.jsonl", cross_rows)
    write_json(OUT / "phase378_intervention/phase378_physical_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge_models()
    elif args.model:
        process_model(args.model)
    else:
        raise SystemExit("Use --model MODEL or --merge")


if __name__ == "__main__":
    main()
