#!/usr/bin/env python3
"""Run the frozen Phase381 single-position versus joint-state scan."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase334_natural_contrast_survey import component_tensor  # noqa: E402
from phase376_decision_aligned_intervention import replace_output, token_rank  # noqa: E402
from phase379_decision_aligned_trace import decision_input  # noqa: E402
from phase380_causal_layout_scan import terminal_share, transfer_letters  # noqa: E402
from phase381_joint_state_case_bank import read_jsonl, write_json, write_jsonl  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase381_joint_state_formation"
CASES = OUT / "private/phase381_qualified_trace_cases.jsonl"
FREEZE = OUT / "phase381_joint_scan_freeze.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
DEPTHS = ("early", "middle_early", "middle", "middle_late", "late")
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")
ROLE_INDEX = {"source": 0, "query": 1, "current": 2}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def off_target_rms(
    patched: torch.Tensor,
    baseline: torch.Tensor,
    donor_token: int,
    recipient_token: int,
) -> float:
    delta = patched.float() - baseline.float()
    delta[donor_token] = 0.0
    delta[recipient_token] = 0.0
    return float(torch.linalg.vector_norm(delta).item() / math.sqrt(max(delta.numel() - 2, 1)))


@torch.inference_mode()
def run_batch(
    loaded: Any,
    selected_layer: int,
    component: str,
    examples: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = {len(example["sequence"]) for example in examples}
    role_counts = {len(example["patch_positions"]) for example in examples}
    if len(lengths) != 1 or len(role_counts) != 1:
        raise RuntimeError("Joint causal batches require equal lengths and role counts")
    input_ids = torch.tensor(
        [example["sequence"] for example in examples],
        dtype=torch.long,
        device=loaded.input_device,
    )
    attention_mask = torch.ones_like(input_ids)
    positions = [example["patch_positions"] for example in examples]
    values = [example["patch_values"] for example in examples]
    layers = get_layers(loaded.model)
    selected = layers[selected_layer]
    terminal_layer = layers[-1]
    terminal_capture: torch.Tensor | None = None
    patch_reached = False

    def patch_tensor(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal patch_reached
        modified = tensor.clone()
        for batch_index, (case_positions, case_values) in enumerate(
            zip(positions, values, strict=True)
        ):
            modified[batch_index, case_positions] = case_values.to(
                modified.device, dtype=modified.dtype
            )
        patch_reached = True
        return modified

    def input_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        return (patch_tensor(inputs[0]), *inputs[1:])

    def output_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        return replace_output(output, patch_tensor(component_tensor(output)))

    def terminal_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal terminal_capture
        terminal_capture = component_tensor(output)[:, -1].detach().float().cpu()

    handles = []
    if component == "layer_input":
        handles.append(selected.register_forward_pre_hook(input_hook))
    elif component == "attention_output":
        handles.append(selected.self_attn.register_forward_hook(output_hook))
    elif component == "mlp_output":
        handles.append(selected.mlp.register_forward_hook(output_hook))
    elif component == "layer_output":
        handles.append(selected.register_forward_hook(output_hook))
    else:
        raise KeyError(component)
    handles.append(terminal_layer.register_forward_hook(terminal_hook))
    try:
        output = loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    if not patch_reached or terminal_capture is None:
        raise RuntimeError("Joint patch or terminal capture did not execute")
    return output.logits[:, -1].detach().float().cpu(), terminal_capture


def process(model: str, batch_size: int) -> dict[str, Any]:
    freeze = read_json(FREEZE)
    if not freeze["authorization"]["run_joint_scan_sequentially"]:
        raise RuntimeError("Phase381 freeze did not authorize the joint scan")
    cases = [row for row in read_jsonl(CASES) if row["private_execution_model"] == model]
    case_by_group: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        case_by_group[case["anonymous_parallel_group_id"]][
            case["contrast_condition"][0]
        ] = case
    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        depth_layers = {
            name: int(round(fraction * (len(layers) - 1)))
            for name, fraction in zip(
                DEPTHS,
                freeze["scan_grid"]["relative_depth_fractions"],
                strict=True,
            )
        }
        selected_case_ids: set[str] = set()
        for groups in freeze["selected_replay_qualified_groups"].values():
            for group in groups:
                selected_case_ids.update(
                    case["blind_case_id"] for case in case_by_group[group].values()
                )
        case_by_id = {case["blind_case_id"]: case for case in cases}
        prepared: dict[str, dict[str, Any]] = {}
        payloads: dict[str, dict[str, Any]] = {}
        for case_id in selected_case_ids:
            case = case_by_id[case_id]
            sequence, positions = decision_input(loaded, case)
            prepared[case_id] = {"sequence": sequence, "positions": positions}
            payloads[case_id] = torch.load(
                OUT / "trace/private/models" / model / "cases" / f"{case_id}.pt",
                map_location="cpu",
                weights_only=True,
            )
        tasks = []
        for stable in freeze["stable_objects"]:
            mechanism = stable["mechanism_id"]
            axis = stable["contrast_axis"]
            for group in freeze["selected_replay_qualified_groups"][mechanism]:
                slots = case_by_group[group]
                for transfer in freeze["transfer_pairs_by_axis"][axis]:
                    donor_letter, recipient_letter = transfer_letters(transfer)
                    tasks.append(
                        {
                            "mechanism_id": mechanism,
                            "contrast_axis": axis,
                            "parallel_group_id": group,
                            "transfer_name": transfer,
                            "donor": slots[donor_letter],
                            "recipient": slots[recipient_letter],
                        }
                    )
        expected_tasks = freeze["denominator"]["transfer_task_count_per_model"]
        if len(tasks) != expected_tasks:
            raise RuntimeError(f"Expected {expected_tasks} tasks, got {len(tasks)}")
        completed_examples = 0
        for depth_name in DEPTHS:
            layer_index = depth_layers[depth_name]
            for component_index, component in enumerate(COMPONENTS):
                for role_set_name, role_names in freeze["scan_grid"]["role_sets"].items():
                    role_indices = [ROLE_INDEX[name] for name in role_names]
                    examples: list[dict[str, Any]] = []
                    for task in tasks:
                        donor = task["donor"]
                        recipient = task["recipient"]
                        donor_payload = payloads[donor["blind_case_id"]]
                        recipient_payload = payloads[recipient["blind_case_id"]]
                        donor_values = donor_payload["vectors"][
                            layer_index, component_index, role_indices
                        ]
                        recipient_values = recipient_payload["vectors"][
                            layer_index, component_index, role_indices
                        ]
                        delta = donor_values.float() - recipient_values.float()
                        permuted = []
                        for index, value in enumerate(delta):
                            shift = max(1, (index + 1) * value.numel() // 4)
                            permuted.append(torch.roll(value, shifts=shift))
                        values = {
                            "natural_swap": donor_values,
                            "equal_energy_permutation": recipient_values.float()
                            + torch.stack(permuted),
                        }
                        for condition, patch_values in values.items():
                            examples.append(
                                {
                                    **task,
                                    "depth_name": depth_name,
                                    "layer_index": layer_index,
                                    "component_type": component,
                                    "role_set_name": role_set_name,
                                    "position_roles": role_names,
                                    "condition": condition,
                                    "sequence": prepared[recipient["blind_case_id"]]["sequence"],
                                    "patch_positions": [
                                        prepared[recipient["blind_case_id"]]["positions"][index]
                                        for index in role_indices
                                    ],
                                    "patch_values": patch_values,
                                }
                            )
                    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
                    for example in examples:
                        buckets[len(example["sequence"])].append(example)
                    for _length, bucket in sorted(buckets.items()):
                        for start in range(0, len(bucket), batch_size):
                            selected_examples = bucket[start : start + batch_size]
                            logits, terminal = run_batch(
                                loaded, layer_index, component, selected_examples
                            )
                            for index, example in enumerate(selected_examples):
                                donor_payload = payloads[example["donor"]["blind_case_id"]]
                                recipient_payload = payloads[
                                    example["recipient"]["blind_case_id"]
                                ]
                                donor_token = int(donor_payload["target_completion_token_id"])
                                recipient_token = int(
                                    recipient_payload["target_completion_token_id"]
                                )
                                baseline_logits = recipient_payload[
                                    "full_vocabulary_logits"
                                ].float()
                                baseline_margin = float(
                                    baseline_logits[donor_token]
                                    - baseline_logits[recipient_token]
                                )
                                patched_margin = float(
                                    logits[index, donor_token]
                                    - logits[index, recipient_token]
                                )
                                donor_terminal = donor_payload["vectors"][-1, -1, -1]
                                recipient_terminal = recipient_payload["vectors"][-1, -1, -1]
                                transfer_gain = patched_margin - baseline_margin
                                side_effect = off_target_rms(
                                    logits[index],
                                    baseline_logits,
                                    donor_token,
                                    recipient_token,
                                )
                                rows.append(
                                    {
                                        "schema_version": "54.5.0",
                                        "phase_id": "Phase381-JointStateScan",
                                        "created_at": datetime.now(timezone.utc).isoformat(),
                                        "model": model,
                                        "mechanism_id": example["mechanism_id"],
                                        "contrast_axis": example["contrast_axis"],
                                        "anonymous_parallel_group_id": example[
                                            "parallel_group_id"
                                        ],
                                        "transfer_name": example["transfer_name"],
                                        "depth_name": depth_name,
                                        "layer_index": layer_index,
                                        "relative_depth": layer_index / max(len(layers) - 1, 1),
                                        "component_type": component,
                                        "role_set_name": role_set_name,
                                        "position_roles": example["position_roles"],
                                        "condition": example["condition"],
                                        "donor_token_id": donor_token,
                                        "recipient_token_id": recipient_token,
                                        "baseline_donor_vs_recipient_margin": baseline_margin,
                                        "patched_donor_vs_recipient_margin": patched_margin,
                                        "transfer_gain": transfer_gain,
                                        "donor_token_rank": token_rank(logits[index], donor_token),
                                        "recipient_token_rank": token_rank(
                                            logits[index], recipient_token
                                        ),
                                        "terminal_transfer_share": terminal_share(
                                            terminal[index], recipient_terminal, donor_terminal
                                        ),
                                        "off_target_logit_rms": side_effect,
                                        "transfer_to_offtarget_rms_ratio": transfer_gain
                                        / max(side_effect, 1e-6),
                                        "single_neuron_intervention": False,
                                        "language_path_claimed": False,
                                    }
                                )
                            completed_examples += len(selected_examples)
                    print(
                        f"[{model}] Phase381 joint {depth_name}/{component}/{role_set_name} "
                        f"{completed_examples}/{freeze['denominator']['condition_rows_per_model']}",
                        flush=True,
                    )
        expected_rows = freeze["denominator"]["condition_rows_per_model"]
        if len(rows) != expected_rows:
            raise RuntimeError(f"Expected {expected_rows} rows, got {len(rows)}")
        write_jsonl(
            OUT / "causal/private/models" / model / "phase381_joint_rows.jsonl",
            rows,
        )
        summary = {
            "schema_version": "54.5.0",
            "phase_id": "Phase381-JointStateScan",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "condition_row_count": len(rows),
            "transfer_task_count": len(tasks),
            "depth_layers": depth_layers,
            "all_patch_hooks_reached": True,
            "top_k_used": False,
            "single_neuron_scan": False,
            "valid": True,
        }
        write_json(OUT / "causal/models" / model / "complete.json", summary)
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(process(args.model, args.batch_size), ensure_ascii=False, indent=2))
