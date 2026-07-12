#!/usr/bin/env python3
"""Collect full-layer exact boundary vectors at Phase379 semantic decisions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase334_natural_contrast_survey import component_tensor  # noqa: E402
from phase371c_blind_vector_contrast import static_roles  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase379_global_reuse_difference_layout"
CASES = OUT / "private/phase379_execution_cases.jsonl"
PROTOCOL = OUT / "phase379_protocol.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("fresh_discovery", "fresh_calibration")
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")
ROLES = ("source", "query", "current")


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
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_rank(logits: torch.Tensor, token_id: int) -> int:
    return 1 + int((logits > logits[token_id]).sum().item())


def decision_input(
    loaded: Any, case: dict[str, Any]
) -> tuple[list[int], tuple[int, int, int]]:
    base = loaded.tokenizer(
        case["prompt"],
        add_special_tokens=bool(case["tokenization_add_special_tokens"]),
        truncation=True,
        max_length=256,
    )["input_ids"]
    step = int(case["target_decision_step"])
    prefix = case["generated_token_ids"][:step]
    values = [int(value) for value in [*base, *prefix]]
    static, base_length = static_roles(loaded.tokenizer, case)
    if base_length != len(base):
        raise RuntimeError(f"Role/base token mismatch for {case['blind_case_id']}")
    return values, (int(static[0]), int(static[1]), len(values) - 1)


@torch.inference_mode()
def trace_batch(
    loaded: Any,
    cases: list[dict[str, Any]],
    sequences: list[list[int]],
    positions: list[tuple[int, int, int]],
    destination: Path,
    artifact_root: Path = OUT,
) -> list[dict[str, Any]]:
    lengths = {len(values) for values in sequences}
    if len(lengths) != 1:
        raise RuntimeError("Trace batches must have equal decision-context lengths")
    input_ids = torch.tensor(
        sequences, dtype=torch.long, device=loaded.input_device
    )
    attention_mask = torch.ones_like(input_ids)
    position_tensor = torch.tensor(
        positions, dtype=torch.long, device=loaded.input_device
    )
    batch_indices = torch.arange(len(cases), device=loaded.input_device).unsqueeze(1)
    layers = get_layers(loaded.model)
    captures: dict[str, list[torch.Tensor | None]] = {
        component: [None] * len(layers) for component in COMPONENTS
    }
    reached = Counter()
    handles = []

    def save(name: str, layer_index: int, output: Any) -> None:
        tensor = component_tensor(output)
        if tensor.ndim != 3:
            raise RuntimeError(
                f"Unexpected {name} shape at layer {layer_index}: {tuple(tensor.shape)}"
            )
        selected = tensor[batch_indices, position_tensor]
        captures[name][layer_index] = selected.detach().cpu()
        reached[(name, layer_index)] += 1

    for layer_index, layer in enumerate(layers):
        def layer_pre(
            _module: Any, inputs: tuple[Any, ...], idx: int = layer_index
        ) -> None:
            if not inputs:
                raise RuntimeError(f"Missing layer input at {idx}")
            save("layer_input", idx, inputs[0])

        def attention_post(
            _module: Any,
            _inputs: tuple[Any, ...],
            output: Any,
            idx: int = layer_index,
        ) -> None:
            save("attention_output", idx, output)

        def mlp_post(
            _module: Any,
            _inputs: tuple[Any, ...],
            output: Any,
            idx: int = layer_index,
        ) -> None:
            save("mlp_output", idx, output)

        def layer_post(
            _module: Any,
            _inputs: tuple[Any, ...],
            output: Any,
            idx: int = layer_index,
        ) -> None:
            save("layer_output", idx, output)

        handles.extend(
            [
                layer.register_forward_pre_hook(layer_pre),
                layer.self_attn.register_forward_hook(attention_post),
                layer.mlp.register_forward_hook(mlp_post),
                layer.register_forward_hook(layer_post),
            ]
        )
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
    expected = len(layers) * len(COMPONENTS)
    if len(reached) != expected or any(value != 1 for value in reached.values()):
        raise RuntimeError(
            f"Incomplete hook ledger: reached={len(reached)} expected={expected}"
        )
    component_stacks = []
    for component in COMPONENTS:
        values = captures[component]
        if any(value is None for value in values):
            raise RuntimeError(f"Missing {component} layer")
        component_stacks.append(torch.stack(values, dim=1))
    vectors = torch.stack(component_stacks, dim=2)
    logits = output.logits[:, -1].detach().float().cpu()
    result_rows = []
    destination.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases):
        path = destination / f"{case['blind_case_id']}.pt"
        target_token = int(
            case["generated_token_ids"][int(case["target_decision_step"])]
        )
        case_logits = logits[index]
        payload = {
            "schema_version": "52.1.0",
            "phase_id": "Phase379-ExactDecisionTrace",
            "model": loaded.key,
            "blind_case_id": case["blind_case_id"],
            "sequence_length": len(sequences[index]),
            "positions": torch.tensor(positions[index], dtype=torch.int64),
            "component_names": list(COMPONENTS),
            "role_names": list(ROLES),
            "vectors": vectors[index].contiguous(),
            "full_vocabulary_logits": case_logits.contiguous(),
            "target_completion_token_id": target_token,
        }
        torch.save(payload, path)
        argmax = int(torch.argmax(case_logits).item())
        result_rows.append(
            {
                "schema_version": "52.1.0",
                "phase_id": "Phase379-ExactDecisionTrace",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": loaded.key,
                "blind_case_id": case["blind_case_id"],
                "anonymous_parallel_group_id": case[
                    "anonymous_parallel_group_id"
                ],
                "anonymous_group_id": case["anonymous_group_id"],
                "anonymous_condition_slot": case["anonymous_condition_slot"],
                "phase379_split": case.get(
                    "phase379_split", case.get("phase380_split", "unspecified")
                ),
                "sequence_length": len(sequences[index]),
                "target_decision_step": int(case["target_decision_step"]),
                "layer_count": len(layers),
                "hidden_size": int(vectors.shape[-1]),
                "exact_vector_shape": list(vectors[index].shape),
                "exact_vector_dtype": str(vectors[index].dtype),
                "target_completion_token_id": target_token,
                "target_completion_token_rank": token_rank(case_logits, target_token),
                "argmax_token_id": argmax,
                "baseline_replay_matches_observed_target_token": argmax
                == target_token,
                "exact_vector_path": str(path.relative_to(artifact_root)),
                "exact_vector_sha256": sha256(path),
                "semantic_labels_available_to_trace": False,
                "target_specific_competition_available_to_trace": False,
            }
        )
    del output, logits, vectors, component_stacks, captures, input_ids, attention_mask
    return result_rows


def process_model(model: str, split: str, batch_size: int) -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    if split == "fresh_discovery":
        authorized = True
    else:
        freeze = OUT / "phase379_discovery_mapping_freeze.json"
        authorized = freeze.is_file() and read_json(freeze)["authorization"][
            "open_calibration_exact_trace"
        ]
    if not authorized:
        raise RuntimeError(f"{split} trace is not authorized")
    cases = [
        row
        for row in read_jsonl(CASES)
        if row["private_execution_model"] == model
        and row["phase379_split"] == split
    ]
    expected = {"fresh_discovery": 112, "fresh_calibration": 60}[split]
    if len(cases) != expected:
        raise RuntimeError(f"Expected {expected} {model}/{split} cases, got {len(cases)}")
    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        prepared = []
        for case in cases:
            sequence, positions = decision_input(loaded, case)
            prepared.append((case, sequence, positions))
        buckets: dict[int, list[tuple[dict[str, Any], list[int], tuple[int, int, int]]]] = defaultdict(list)
        for row in prepared:
            buckets[len(row[1])].append(row)
        completed = 0
        destination = OUT / split / "private/models" / model / "cases"
        for _length, bucket in sorted(buckets.items()):
            for start in range(0, len(bucket), batch_size):
                selected = bucket[start : start + batch_size]
                rows.extend(
                    trace_batch(
                        loaded,
                        [row[0] for row in selected],
                        [row[1] for row in selected],
                        [row[2] for row in selected],
                        destination,
                    )
                )
                completed += len(selected)
                print(
                    f"[{model}] Phase379 {split} exact trace {completed}/{len(cases)}",
                    flush=True,
                )
        metadata_path = OUT / split / "models" / model / "phase379_trace_rows.jsonl"
        write_jsonl(metadata_path, sorted(rows, key=lambda row: row["blind_case_id"]))
        summary = {
            "schema_version": "52.1.0",
            "phase_id": "Phase379-ExactDecisionTrace",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "split": split,
            "case_count": len(rows),
            "layer_count": len(get_layers(loaded.model)),
            "component_count": len(COMPONENTS),
            "position_role_count": len(ROLES),
            "exact_event_vector_count": len(rows)
            * len(get_layers(loaded.model))
            * len(COMPONENTS)
            * len(ROLES),
            "baseline_replay_match_count": sum(
                row["baseline_replay_matches_observed_target_token"] for row in rows
            ),
            "baseline_replay_mismatch_count": sum(
                not row["baseline_replay_matches_observed_target_token"] for row in rows
            ),
            "semantic_labels_available_to_trace": False,
            "top_k_used": False,
            "full_vocabulary_logits_retained": True,
            "all_exact_vector_files_exist": all(
                (OUT / row["exact_vector_path"]).is_file() for row in rows
            ),
            "protocol_sha256": sha256(PROTOCOL),
            "metadata_sha256": sha256(metadata_path),
            "valid": len(rows) == expected,
        }
        write_json(OUT / split / "models" / model / "complete.json", summary)
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            process_model(args.model, args.split, args.batch_size),
            ensure_ascii=False,
            indent=2,
        )
    )
