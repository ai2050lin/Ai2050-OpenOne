#!/usr/bin/env python3
"""Run decision-aligned natural-boundary activation swaps for Phase376."""

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

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase334_natural_contrast_survey import component_tensor  # noqa: E402
from phase371c_blind_vector_contrast import model_pairs, static_roles  # noqa: E402


PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
OUT = ROOT / "tests/gpt5/result/phase376_decision_aligned_subgraphs"
PROTOCOL = OUT / "phase376_intervention_protocol.json"
FREEZE = OUT / "phase376_intervention_execution_freeze.json"
COLLECTOR = (
    PHASE371
    / "phase371c_behavior_analysis/private/phase371c_discovery_collector_cases.jsonl"
)
BEHAVIOR = PHASE371 / "phase371c_behavior_qualification/private/models"
DECISION_ROWS = OUT / "private/phase376_decision_time_rows.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
CONDITIONS = ("correct", "wrong_depth", "wrong_role", "wrong_time")
TRANSFER_PAIRS = (
    ("A", "C", "A_to_C", "treatment"),
    ("C", "A", "C_to_A", "treatment"),
    ("B", "D", "B_to_D", "direct_route_control"),
    ("D", "B", "D_to_B", "direct_route_control"),
)
ROLE_CYCLE = {"source": "query", "query": "current", "current": "source"}


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


def replace_output(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    raise TypeError(type(output).__name__)


def condition_letter(value: str) -> str:
    letter = value.split("_", 1)[0]
    if letter not in {"A", "B", "C", "D"}:
        raise ValueError(value)
    return letter


def build_cases(model: str) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, str]]:
    collector = {
        row["blind_case_id"]: row
        for row in read_jsonl(COLLECTOR)
        if row["private_execution_model"] == model
    }
    behavior = {
        row["blind_case_id"]: row
        for row in read_jsonl(BEHAVIOR / model / "phase371c_behavior_rows.jsonl")
        if row["blind_case_id"] in collector
    }
    decisions = {
        row["blind_case_id"]: row
        for row in read_jsonl(DECISION_ROWS)
        if row["model"] == model
    }
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    mechanisms: dict[str, str] = {}
    for case_id, base in collector.items():
        row = {**base, **behavior[case_id], **decisions[case_id]}
        if row["target_decision_step"] is None:
            raise RuntimeError(f"Missing target decision for {model}/{case_id}")
        parallel = row["anonymous_parallel_group_id"]
        groups[parallel][condition_letter(row["contrast_condition"])] = row
        mechanisms[parallel] = row["mechanism_id"]
    if len(groups) != 22 or any(set(rows) != {"A", "B", "C", "D"} for rows in groups.values()):
        raise RuntimeError(f"Invalid Phase376 groups for {model}")
    return dict(groups), mechanisms


def context_inputs(loaded: Any, case: dict[str, Any], event: str) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    encoded = loaded.tokenizer(
        case["prompt"],
        add_special_tokens=bool(case["tokenization_add_special_tokens"]),
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    base_ids = encoded["input_ids"][0]
    decision_step = int(case["target_decision_step"])
    prefix = [] if event == "answer_entry" else case["generated_token_ids"][:decision_step]
    if prefix:
        suffix = torch.tensor(prefix, dtype=base_ids.dtype)
        input_ids = torch.cat([base_ids, suffix], dim=0).unsqueeze(0)
    else:
        input_ids = base_ids.unsqueeze(0)
    input_ids = input_ids.to(loaded.input_device)
    attention_mask = torch.ones_like(input_ids)
    static, _base_length = static_roles(loaded.tokenizer, case)
    positions = {
        "source": int(static[0]),
        "query": int(static[1]),
        "current": int(input_ids.shape[1]) - 1,
    }
    return {"input_ids": input_ids, "attention_mask": attention_mask}, positions


@torch.inference_mode()
def capture_event(
    loaded: Any,
    case: dict[str, Any],
    event: str,
    depth_layers: dict[str, int],
) -> dict[str, Any]:
    inputs, positions = context_inputs(loaded, case, event)
    layers = get_layers(loaded.model)
    captures: dict[int, dict[str, dict[str, torch.Tensor]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    handles = []
    selected = set(depth_layers.values())
    for layer_index in selected:
        layer = layers[layer_index]

        def save_component(component: str, idx: int, output: Any) -> None:
            tensor = component_tensor(output)
            for role, position in positions.items():
                captures[idx][component][role] = tensor[0, position].detach().cpu()

        def attention_hook(
            _module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> None:
            save_component("attention_output", idx, output)

        def mlp_hook(
            _module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> None:
            save_component("mlp_output", idx, output)

        def layer_hook(
            _module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> None:
            save_component("residual_output", idx, output)

        handles.extend(
            [
                layer.self_attn.register_forward_hook(attention_hook),
                layer.mlp.register_forward_hook(mlp_hook),
                layer.register_forward_hook(layer_hook),
            ]
        )
    try:
        output = loaded.model(**inputs, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    logits = output.logits[0, -1].detach().float().cpu()
    return {
        "event": event,
        "sequence_length": int(inputs["input_ids"].shape[1]),
        "positions": positions,
        "logits": logits,
        "argmax_token_id": int(torch.argmax(logits).item()),
        "captures": {
            layer: {
                component: dict(role_values)
                for component, role_values in components.items()
            }
            for layer, components in captures.items()
        },
    }


def template_specs(
    template_name: str,
    template: dict[str, Any],
    donor_event: dict[str, Any],
    donor_layer: int,
    recipient_positions: dict[str, int],
    wrong_roles: bool,
) -> list[dict[str, Any]]:
    components = (
        ("attention_output", "mlp_output")
        if template["component"] == "attention_mlp_output"
        else (template["component"],)
    )
    specs = []
    for component in components:
        for recipient_role in template["roles"]:
            donor_role = ROLE_CYCLE[recipient_role] if wrong_roles else recipient_role
            specs.append(
                {
                    "template": template_name,
                    "component": component,
                    "recipient_position": recipient_positions[recipient_role],
                    "donor_role": donor_role,
                    "value": donor_event["captures"][donor_layer][component][donor_role],
                }
            )
    return specs


@torch.inference_mode()
def run_patch_batch(
    loaded: Any,
    recipient: dict[str, Any],
    donor_decision: dict[str, Any],
    donor_entry: dict[str, Any],
    selected_layer: int,
    wrong_layer: int,
    template_name: str,
    template: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, bool]]:
    inputs, recipient_positions = context_inputs(loaded, recipient, "target_decision")
    batch_inputs = {
        key: value.repeat(len(CONDITIONS), 1) for key, value in inputs.items()
    }
    event_by_condition = {
        "correct": (donor_decision, selected_layer, False),
        "wrong_depth": (donor_decision, wrong_layer, False),
        "wrong_role": (donor_decision, selected_layer, True),
        "wrong_time": (donor_entry, selected_layer, False),
    }
    specs = {
        condition: template_specs(
            template_name,
            template,
            event,
            donor_layer,
            recipient_positions,
            wrong_roles,
        )
        for condition, (event, donor_layer, wrong_roles) in event_by_condition.items()
    }
    layers = get_layers(loaded.model)
    layer = layers[selected_layer]
    reached = {condition: False for condition in CONDITIONS}

    def patch_component(component: str, output: Any) -> Any:
        tensor = component_tensor(output)
        modified = tensor.clone()
        changed = False
        for batch_index, condition in enumerate(CONDITIONS):
            for spec in specs[condition]:
                if spec["component"] != component:
                    continue
                position = int(spec["recipient_position"])
                modified[batch_index, position] = spec["value"].to(
                    modified.device, dtype=modified.dtype
                )
                reached[condition] = True
                changed = True
        return replace_output(output, modified) if changed else output

    def attention_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        return patch_component("attention_output", output)

    def mlp_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        return patch_component("mlp_output", output)

    def layer_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        return patch_component("residual_output", output)

    handles = [
        layer.self_attn.register_forward_hook(attention_hook),
        layer.mlp.register_forward_hook(mlp_hook),
        layer.register_forward_hook(layer_hook),
    ]
    try:
        output = loaded.model(**batch_inputs, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    logits = output.logits[:, -1].detach().float().cpu()
    return {condition: logits[index] for index, condition in enumerate(CONDITIONS)}, reached


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    value = logits[token_id]
    return 1 + int((logits > value).sum().item())


def condition_metrics(
    logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    donor_token: int,
    recipient_token: int,
) -> dict[str, Any]:
    baseline_margin = float(
        baseline_logits[donor_token].item() - baseline_logits[recipient_token].item()
    )
    margin = float(logits[donor_token].item() - logits[recipient_token].item())
    return {
        "donor_vs_recipient_margin": margin,
        "baseline_donor_vs_recipient_margin": baseline_margin,
        "transfer_gain": margin - baseline_margin,
        "donor_token_rank": token_rank(logits, donor_token),
        "recipient_token_rank": token_rank(logits, recipient_token),
        "argmax_token_id": int(torch.argmax(logits).item()),
        "donor_token_is_argmax": int(torch.argmax(logits).item()) == donor_token,
        "recipient_token_is_argmax": int(torch.argmax(logits).item()) == recipient_token,
    }


def direction_pass(row: dict[str, Any], gates: dict[str, Any]) -> bool:
    values = row["conditions"]
    correct = float(values["correct"]["transfer_gain"])
    passed = (
        row["baseline_replay_matches_recipient_token"]
        and all(row["patch_reached"].values())
        and correct >= float(gates["minimum_correct_transfer_gain"])
        and correct
        >= float(values["wrong_depth"]["transfer_gain"])
        + float(gates["minimum_gain_over_wrong_depth"])
        and correct
        >= float(values["wrong_role"]["transfer_gain"])
        + float(gates["minimum_gain_over_wrong_role"])
    )
    if row["wrong_time_control_distinct"]:
        passed = passed and (
            correct
            >= float(values["wrong_time"]["transfer_gain"])
            + float(gates["minimum_gain_over_wrong_time_when_distinct"])
        )
    return bool(passed)


def process_model(model: str) -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    freeze = read_json(FREEZE)
    if not protocol["authorization"]["run_all_preregistered_discovery_interventions"]:
        raise RuntimeError("Protocol does not authorize interventions")
    if not freeze["valid"]:
        raise RuntimeError("Execution freeze invalid")
    groups, mechanisms = build_cases(model)
    pair_rows, _ = model_pairs(model)
    depth_layers = {row["name"]: int(row["source_layer"]) for row in pair_rows}
    templates = protocol["natural_templates"]
    gates = protocol["frozen_numeric_gates"]
    loaded = None
    rows = []
    baseline_total = 0
    baseline_match = 0
    try:
        loaded = load_probe_model(model)
        for group_index, parallel in enumerate(sorted(groups), 1):
            cases = groups[parallel]
            natural: dict[str, dict[str, Any]] = {}
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
                for depth_index, depth in enumerate(("early", "middle", "late")):
                    wrong_depth = ("early", "middle", "late")[(depth_index + 1) % 3]
                    selected_layer = depth_layers[depth]
                    wrong_layer = depth_layers[wrong_depth]
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
                            "schema_version": "49.2.0",
                            "phase_id": "Phase376-Intervention",
                            "model": model,
                            "mechanism_id": mechanisms[parallel],
                            "anonymous_parallel_group_id": parallel,
                            "transfer": transfer_name,
                            "transfer_class": transfer_class,
                            "donor_condition": donor_condition,
                            "recipient_condition": recipient_condition,
                            "donor_case_id": donor["blind_case_id"],
                            "recipient_case_id": recipient["blind_case_id"],
                            "donor_decision_step": int(donor["target_decision_step"]),
                            "recipient_decision_step": int(recipient["target_decision_step"]),
                            "donor_target_token_id": donor_natural["target_token"],
                            "recipient_target_token_id": recipient_natural["target_token"],
                            "relative_depth": depth,
                            "selected_layer": selected_layer,
                            "wrong_depth_control": wrong_depth,
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
            print(f"[{model}] decision-aligned groups {group_index}/22", flush=True)
        private_dir = OUT / "phase376_intervention/models" / model / "private"
        write_jsonl(private_dir / "phase376_intervention_rows.jsonl", rows)
        grouped_rows: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped_rows[
                (
                    row["mechanism_id"],
                    row["anonymous_parallel_group_id"],
                    row["relative_depth"],
                    row["template"],
                )
            ].append(row)
        group_gates = []
        pass_counts: Counter[tuple[str, str, str]] = Counter()
        winner_counts: Counter[tuple[str, str, str]] = Counter()
        for key, selected in sorted(grouped_rows.items()):
            by_transfer = {row["transfer"]: row for row in selected}
            treatment = [by_transfer["A_to_C"], by_transfer["C_to_A"]]
            controls = [by_transfer["B_to_D"], by_transfer["D_to_B"]]
            group_pass = all(row["direction_gate_pass"] for row in treatment)
            winner_flip_both = all(row["winner_transfer_under_correct_patch"] for row in treatment)
            canonical = (key[0], key[2], key[3])
            if group_pass:
                pass_counts[canonical] += 1
            if group_pass and winner_flip_both:
                winner_counts[canonical] += 1
            group_gates.append(
                {
                    "model": model,
                    "mechanism_id": key[0],
                    "anonymous_parallel_group_id": key[1],
                    "relative_depth": key[2],
                    "template": key[3],
                    "treatment_direction_passes": {
                        row["transfer"]: row["direction_gate_pass"] for row in treatment
                    },
                    "treatment_correct_gains": {
                        row["transfer"]: row["conditions"]["correct"]["transfer_gain"]
                        for row in treatment
                    },
                    "direct_route_control_correct_gains": {
                        row["transfer"]: row["conditions"]["correct"]["transfer_gain"]
                        for row in controls
                    },
                    "group_gate_pass": group_pass,
                    "winner_flip_both_directions": winner_flip_both,
                }
            )
        minimum = int(gates["minimum_independent_groups_per_model_mechanism_template"])
        candidates = [
            {
                "model": model,
                "mechanism_id": key[0],
                "relative_depth": key[1],
                "template": key[2],
                "group_pass_count": count,
                "winner_flip_group_count": winner_counts[key],
                "minimum_group_count": minimum,
                "model_gate_pass": count >= minimum,
                "replicated_winner_flip_pass": winner_counts[key] >= minimum,
            }
            for key, count in sorted(pass_counts.items())
            if count >= minimum
        ]
        write_jsonl(private_dir / "phase376_group_gate_rows.jsonl", group_gates)
        write_jsonl(
            OUT
            / "phase376_intervention/models"
            / model
            / "phase376_model_candidates.jsonl",
            candidates,
        )
        summary = {
            "schema_version": "49.2.0",
            "phase_id": "Phase376-Intervention",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "execution": {
                "device": str(loaded.input_device),
                "model_execution": True,
                "discovery_only": True,
                "calibration_opened": False,
                "physical_opened": False,
            },
            "denominator": {
                "case_count": 88,
                "parallel_group_count": 22,
                "transfer_count": 88,
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
                "replicated_winner_flip_candidate_count": sum(
                    row["replicated_winner_flip_pass"] for row in candidates
                ),
                "model_candidates": candidates,
            },
            "claim_boundary": {
                "full_generation_tested": False,
                "natural_necessity_tested": False,
                "single_neuron_causality_tested": False,
                "language_mechanism_claimed": False,
            },
        }
        write_json(
            OUT
            / "phase376_intervention/models"
            / model
            / "phase376_model_summary.json",
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
    summaries = [
        read_json(
            OUT
            / "phase376_intervention/models"
            / model
            / "phase376_model_summary.json"
        )
        for model in MODELS
    ]
    canonical: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        for row in summary["results"]["model_candidates"]:
            canonical[
                (row["mechanism_id"], row["relative_depth"], row["template"])
            ].append(row)
    cross_rows = []
    for key, rows in sorted(canonical.items()):
        models = {row["model"] for row in rows}
        level2 = "glm4" in models and bool(models & {"qwen3", "deepseek7b"})
        level3 = models == set(MODELS)
        winner_models = {
            row["model"] for row in rows if row["replicated_winner_flip_pass"]
        }
        winner_level2 = "glm4" in winner_models and bool(
            winner_models & {"qwen3", "deepseek7b"}
        )
        cross_rows.append(
            {
                "mechanism_id": key[0],
                "relative_depth": key[1],
                "template": key[2],
                "models": sorted(models),
                "heterogeneous_level2_transfer_pass": level2,
                "level3_transfer_pass": level3,
                "winner_flip_models": sorted(winner_models),
                "heterogeneous_level2_winner_flip_pass": winner_level2,
                "language_mechanism_claimed": False,
            }
        )
    level2 = [row for row in cross_rows if row["heterogeneous_level2_transfer_pass"]]
    winner_level2 = [
        row for row in cross_rows if row["heterogeneous_level2_winner_flip_pass"]
    ]
    summary = {
        "schema_version": "49.3.0",
        "phase_id": "Phase376-Intervention-Merge",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominator": {
            "model_count": 3,
            "case_count": 264,
            "template_depth_transfer_count": sum(
                row["denominator"]["template_depth_transfer_count"] for row in summaries
            ),
            "patched_forward_condition_count": sum(
                row["denominator"]["patched_forward_condition_count"] for row in summaries
            ),
            "model_candidate_count": sum(
                row["results"]["model_candidate_count"] for row in summaries
            ),
            "canonical_candidate_count": len(cross_rows),
        },
        "quality": {
            "baseline_replay_match_counts": {
                row["model"]: row["quality"]["baseline_replay_match_count"]
                for row in summaries
            },
            "all_patch_hooks_reached": all(
                row["quality"]["all_patch_hooks_reached"] for row in summaries
            ),
        },
        "results": {
            "heterogeneous_level2_transfer_count": len(level2),
            "level3_transfer_count": sum(
                row["level3_transfer_pass"] for row in cross_rows
            ),
            "heterogeneous_level2_winner_flip_count": len(winner_level2),
            "language_path_candidate_count": 0,
            "language_mechanism_claimed": False,
        },
        "model_results": [
            {
                "model": row["model"],
                "model_candidate_count": row["results"]["model_candidate_count"],
                "replicated_winner_flip_candidate_count": row["results"][
                    "replicated_winner_flip_candidate_count"
                ],
            }
            for row in summaries
        ],
        "cross_model_rows": cross_rows,
        "authorization": {
            "open_calibration": bool(winner_level2),
            "open_physical": False,
            "single_neuron_scan": False,
        },
        "next_decision": (
            "freeze_separate_calibration_replication_protocol"
            if winner_level2
            else "close_current_templates_and_analyze_direct_causal_effects"
        ),
    }
    write_jsonl(
        OUT / "phase376_intervention/phase376_cross_model_rows.jsonl", cross_rows
    )
    write_json(
        OUT / "phase376_intervention/phase376_intervention_summary.json", summary
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge:
        merge_models()
    elif args.model:
        process_model(args.model)
    else:
        raise SystemExit("Use --model MODEL or --merge")


if __name__ == "__main__":
    main()
