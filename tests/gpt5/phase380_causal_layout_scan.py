#!/usr/bin/env python3
"""Run the registered Phase380 natural-boundary causal layout scan."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase334_natural_contrast_survey import component_tensor  # noqa: E402
from phase376_decision_aligned_intervention import replace_output, token_rank  # noqa: E402
from phase379_decision_aligned_trace import decision_input  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase380_independent_layout_validation"
CASES = OUT / "private/phase380_qualified_trace_cases.jsonl"
FREEZE = OUT / "phase380_causal_scan_freeze.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
DEPTH_NAMES = ("early", "middle_early", "middle", "middle_late", "late")
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")
ROLES = ("source", "query", "current")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def transfer_letters(name: str) -> tuple[str, str]:
    left, right = name.split("_to_")
    return left, right


def terminal_share(
    patched: torch.Tensor, recipient: torch.Tensor, donor: torch.Tensor
) -> float:
    delta = donor.float() - recipient.float()
    denominator = float(torch.dot(delta, delta).item())
    if denominator <= 1e-12:
        return 0.0
    return float(torch.dot(patched.float() - recipient.float(), delta).item() / denominator)


@torch.inference_mode()
def run_batch(
    loaded: Any,
    selected_layer: int,
    component: str,
    examples: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = {len(example["sequence"]) for example in examples}
    if len(lengths) != 1:
        raise RuntimeError("Causal batches must have equal sequence lengths")
    input_ids = torch.tensor(
        [example["sequence"] for example in examples],
        dtype=torch.long,
        device=loaded.input_device,
    )
    attention_mask = torch.ones_like(input_ids)
    positions = torch.tensor(
        [example["patch_position"] for example in examples],
        dtype=torch.long,
        device=loaded.input_device,
    )
    values = torch.stack([example["patch_value"] for example in examples]).to(
        loaded.input_device
    )
    batch_indices = torch.arange(len(examples), device=loaded.input_device)
    layers = get_layers(loaded.model)
    selected = layers[selected_layer]
    terminal_layer = layers[-1]
    terminal_capture: torch.Tensor | None = None
    patch_reached = False

    def patch_tensor(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal patch_reached
        modified = tensor.clone()
        modified[batch_indices, positions] = values.to(
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
        raise RuntimeError("Causal patch or terminal hook did not execute")
    logits = output.logits[:, -1].detach().float().cpu()
    return logits, terminal_capture


def process(model: str, batch_size: int) -> dict[str, Any]:
    freeze = read_json(FREEZE)
    cases = [row for row in read_jsonl(CASES) if row["private_execution_model"] == model]
    case_by_group: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        case_by_group[case["anonymous_parallel_group_id"]][
            case["contrast_condition"][0]
        ] = case
    loaded = None
    rows = []
    try:
        loaded = load_probe_model(model)
        layers = get_layers(loaded.model)
        depth_layers = {
            name: int(round(fraction * (len(layers) - 1)))
            for name, fraction in zip(
                DEPTH_NAMES,
                freeze["scan_grid"]["relative_depth_fractions"],
                strict=True,
            )
        }
        prepared = {}
        payloads = {}
        selected_case_ids = set()
        for mechanism, groups in freeze["selected_parallel_groups"].items():
            for group in groups:
                for case in case_by_group[group].values():
                    selected_case_ids.add(case["blind_case_id"])
        case_by_id = {case["blind_case_id"]: case for case in cases}
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
            for group in freeze["selected_parallel_groups"][mechanism]:
                slots = case_by_group[group]
                for transfer in freeze["transfer_pairs_by_axis"][axis]:
                    donor_letter, recipient_letter = transfer_letters(transfer)
                    donor = slots[donor_letter]
                    recipient = slots[recipient_letter]
                    tasks.append(
                        {
                            "mechanism_id": mechanism,
                            "contrast_axis": axis,
                            "parallel_group_id": group,
                            "transfer_name": transfer,
                            "donor": donor,
                            "recipient": recipient,
                        }
                    )
        expected_tasks = freeze["denominator"]["selected_group_object_count"] * 4
        if len(tasks) != expected_tasks:
            raise RuntimeError(f"Expected {expected_tasks} transfer tasks, got {len(tasks)}")
        completed_examples = 0
        for depth_name in DEPTH_NAMES:
            layer_index = depth_layers[depth_name]
            for component_index, component in enumerate(COMPONENTS):
                examples = []
                for task in tasks:
                    donor = task["donor"]
                    recipient = task["recipient"]
                    donor_payload = payloads[donor["blind_case_id"]]
                    recipient_payload = payloads[recipient["blind_case_id"]]
                    for role_index, role in enumerate(ROLES):
                        donor_value = donor_payload["vectors"][
                            layer_index, component_index, role_index
                        ]
                        recipient_value = recipient_payload["vectors"][
                            layer_index, component_index, role_index
                        ]
                        delta = donor_value.float() - recipient_value.float()
                        shift = max(1, delta.numel() // 3)
                        values = {
                            "natural_swap": donor_value,
                            "equal_energy_permutation": recipient_value.float()
                            + torch.roll(delta, shifts=shift),
                        }
                        for condition, patch_value in values.items():
                            examples.append(
                                {
                                    **task,
                                    "depth_name": depth_name,
                                    "layer_index": layer_index,
                                    "component_type": component,
                                    "position_role": role,
                                    "condition": condition,
                                    "sequence": prepared[recipient["blind_case_id"]]["sequence"],
                                    "patch_position": prepared[recipient["blind_case_id"]]["positions"][role_index],
                                    "patch_value": patch_value,
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
                            rows.append(
                                {
                                    "schema_version": "53.9.0",
                                    "phase_id": "Phase380-CausalLayoutScan",
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
                                    "relative_depth": layer_index
                                    / max(len(layers) - 1, 1),
                                    "component_type": component,
                                    "position_role": example["position_role"],
                                    "condition": example["condition"],
                                    "donor_token_id": donor_token,
                                    "recipient_token_id": recipient_token,
                                    "baseline_donor_vs_recipient_margin": baseline_margin,
                                    "patched_donor_vs_recipient_margin": patched_margin,
                                    "transfer_gain": patched_margin - baseline_margin,
                                    "donor_token_rank": token_rank(logits[index], donor_token),
                                    "recipient_token_rank": token_rank(
                                        logits[index], recipient_token
                                    ),
                                    "terminal_transfer_share": terminal_share(
                                        terminal[index], recipient_terminal, donor_terminal
                                    ),
                                    "terminal_effect_norm": float(
                                        torch.linalg.vector_norm(
                                            terminal[index].float()
                                            - recipient_terminal.float()
                                        ).item()
                                    ),
                                    "single_neuron_intervention": False,
                                    "language_path_claimed": False,
                                }
                            )
                        completed_examples += len(selected_examples)
                print(
                    f"[{model}] Phase380 causal {depth_name}/{component} {completed_examples}/{freeze['denominator']['condition_rows_per_model']}",
                    flush=True,
                )
        if len(rows) != freeze["denominator"]["condition_rows_per_model"]:
            raise RuntimeError(
                f"Expected {freeze['denominator']['condition_rows_per_model']} rows, got {len(rows)}"
            )
        path = OUT / "causal/private/models" / model / "phase380_causal_rows.jsonl"
        write_jsonl(path, rows)
        summary = {
            "schema_version": "53.9.0",
            "phase_id": "Phase380-CausalLayoutScan",
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
