#!/usr/bin/env python3
"""Run the Phase570 late answer-boundary attention causal screen."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
from phase569_relation_competition_behavior import classify  # noqa: E402
from phase569_role_position_utils import role_positions  # noqa: E402


PHASE = "Phase570"
MODELS = ("qwen3", "glm4", "deepseek7b")
CONDITIONS = (
    "baseline",
    "target_projection_remove",
    "random_matched_remove",
    "wrong_layer_projection_remove",
)
CAUSAL_CASE_CAP = 64
MINIMUM_CASES_PER_PHENOTYPE = 48
OUT_DIR = ROOT / "tests/gpt5/result/phase570_answer_bridge_causal"
CASES_PATH = OUT_DIR / "phase570_registered_cases.jsonl.gz"
PROTOCOL_PATH = OUT_DIR / "phase570_frozen_protocol.json"
AUDIT_PATH = OUT_DIR / "phase570_static_audit.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_from_output(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(value):
        raise TypeError(f"Unexpected attention output: {type(value).__name__}")
    return value


def replace_tensor_output(output: Any, value: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (value, *output[1:])
    return value


def screen_path(model: str) -> Path:
    return OUT_DIR / f"phase570_{model}_baseline_screen_rows.jsonl"


def causal_path(model: str) -> Path:
    return OUT_DIR / f"phase570_{model}_causal_rows.jsonl"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase570_{model}_execution_summary.json"


def contract_path(model: str) -> Path:
    return OUT_DIR / f"phase570_{model}_run_contract.json"


def load_cases(model: str) -> list[dict[str, Any]]:
    cases = [row for row in iter_jsonl(CASES_PATH) if row["model"] == model]
    if len(cases) != 768 or any(row["sealed"] for row in cases):
        raise RuntimeError(f"Phase570 case denominator drift for {model}: {len(cases)}")
    return sorted(cases, key=lambda row: row["case_id"])


def prepare(model: str, restart: bool) -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    audit = read_json(AUDIT_PATH)
    if not audit["valid"] or audit["cases_sha256"] != sha256_file(CASES_PATH):
        raise RuntimeError("Phase570 static protocol failed or drifted")
    payload = {
        "schema_version": "phase570_run_contract.v1",
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "cases_sha256": sha256_file(CASES_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "conditions": list(CONDITIONS),
        "causal_case_cap_per_phenotype": CAUSAL_CASE_CAP,
        "minimum_cases_per_phenotype": MINIMUM_CASES_PER_PHENOTYPE,
        "do_sample": False,
        "torch_dtype_requested": "torch.bfloat16",
        "use_8bit": False,
        "sealed_split_read": False,
    }
    if restart:
        for path in (screen_path(model), causal_path(model), summary_path(model), contract_path(model)):
            path.unlink(missing_ok=True)
    if contract_path(model).exists():
        existing = read_json(contract_path(model))
        for key in (
            "model", "cases_sha256", "protocol_sha256", "conditions",
            "causal_case_cap_per_phenotype", "minimum_cases_per_phenotype",
            "do_sample", "torch_dtype_requested", "use_8bit", "sealed_split_read",
        ):
            if existing[key] != payload[key]:
                raise RuntimeError(f"Phase570 run contract drift: {model}/{key}")
    else:
        write_json(contract_path(model), payload)
    return protocol


def balanced(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted(rows, key=lambda item: item["case_id"]):
        strata[(
            row["source_factorial_cell"], row["target"], row["other_relation_target"]
        )].append(row)
    keys = sorted(strata)
    selected = []
    while keys and len(selected) < cap:
        next_keys = []
        for key in keys:
            if len(selected) >= cap:
                break
            if strata[key]:
                selected.append(strata[key].popleft())
            if strata[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def deterministic_random_directions(
    rows: list[dict[str, Any]], direction_unit: torch.Tensor
) -> torch.Tensor:
    hidden = direction_unit.shape[-1]
    vectors = []
    for row in rows:
        seed = int(hashlib.sha256(row["case_id"].encode("utf-8")).hexdigest()[:16], 16)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        vectors.append(torch.randn(hidden, generator=generator, dtype=torch.float32))
    random = torch.stack(vectors).to(direction_unit.device)
    random = random - (random * direction_unit).sum(dim=-1, keepdim=True) * direction_unit
    return random / random.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def batch_coordinates(tokenizer: Any, model: str, rows: list[dict[str, Any]]) -> tuple[list[str], list[list[int]], list[int]]:
    prompts = [render_chat(tokenizer, model, row["raw_prompt"]) for row in rows]
    individual = [role_positions(tokenizer, prompt, row) for prompt, row in zip(prompts, rows)]
    return prompts, [ids for ids, _groups in individual], [
        groups["answer_boundary"][-1] for _ids, groups in individual
    ]


def run_generation_batch(
    loaded: Any,
    layers: list[Any],
    model: str,
    rows: list[dict[str, Any]],
    condition: str,
    target_layer: int,
    wrong_layer: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    prompts, individual_ids, local_positions = batch_coordinates(
        loaded.tokenizer, model, rows
    )
    encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    prompt_width = int(encoded["input_ids"].shape[1])
    positions = []
    for row_index, ids in enumerate(individual_ids):
        mask_ids = encoded["input_ids"][row_index][
            encoded["attention_mask"][row_index].bool()
        ].tolist()
        if [int(value) for value in mask_ids] != ids:
            raise RuntimeError("Phase570 individual/batch tokenization drift")
        positions.append(prompt_width - len(ids) + local_positions[row_index])
    output_embeddings = loaded.model.get_output_embeddings()
    target_ids = torch.tensor([
        row["candidate_token_ids"][row["target"]][0] for row in rows
    ], dtype=torch.long, device=output_embeddings.weight.device)
    other_ids = torch.tensor([
        row["candidate_token_ids"][row["other_relation_target"]][0] for row in rows
    ], dtype=torch.long, device=output_embeddings.weight.device)
    direction = (
        output_embeddings.weight[target_ids] - output_embeddings.weight[other_ids]
    ).detach().float()
    direction_unit = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    random_unit = deterministic_random_directions(rows, direction_unit)
    positions_tensor = torch.tensor(positions, dtype=torch.long, device=loaded.input_device)
    batch_indices = torch.arange(len(rows), device=loaded.input_device)
    handle = None
    if condition != "baseline":
        selected_layer = wrong_layer if condition == "wrong_layer_projection_remove" else target_layer

        def intervention(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            value = tensor_from_output(output)
            if value.shape[1] <= int(positions_tensor.max().item()):
                return output
            modified = value.clone()
            vectors = modified[batch_indices, positions_tensor, :].float()
            projection = (vectors * direction_unit).sum(dim=-1, keepdim=True)
            removal_direction = (
                random_unit if condition == "random_matched_remove" else direction_unit
            )
            vectors = vectors - projection * removal_direction
            modified[batch_indices, positions_tensor, :] = vectors.to(modified.dtype)
            return replace_tensor_output(output, modified)

        handle = layers[selected_layer].self_attn.register_forward_hook(intervention)
    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    try:
        with torch.inference_mode():
            generated = loaded.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=loaded.tokenizer.pad_token_id,
                eos_token_id=loaded.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
    finally:
        if handle is not None:
            handle.remove()
    if not generated.scores:
        raise RuntimeError("Phase570 generation returned no first-step scores")
    first_scores = generated.scores[0].float()
    margins = first_scores[batch_indices, target_ids] - first_scores[batch_indices, other_ids]
    results = []
    for index, row in enumerate(rows):
        text = loaded.tokenizer.decode(
            generated.sequences[index, prompt_width:], skip_special_tokens=True
        )
        results.append({
            **row,
            **classify(row, text),
            "condition": condition,
            "first_step_target_minus_other_margin": float(margins[index].item()),
            "target_layer": target_layer,
            "wrong_layer_control": wrong_layer,
            "intervention_component": "attention_output",
            "intervention_role": "answer_boundary",
            "observer_or_causal": "observer" if condition == "baseline" else "causal_screen",
        })
    del generated, encoded, first_scores, margins
    return results


def run(model: str, behavior_batch_size: int, causal_batch_size: int, max_new_tokens: int, restart: bool) -> Path:
    protocol = prepare(model, restart)
    cases = load_cases(model)
    layer_spec = protocol["selected_layers_by_model"][model]
    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        loaded.tokenizer.padding_side = "left"
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase570 requires BF16, got {run_dtype}")
        layers = get_layers(loaded.model)
        if len(layers) != layer_spec["layer_count"]:
            raise RuntimeError("Phase570 layer count drift")
        screen_rows = []
        for start in range(0, len(cases), behavior_batch_size):
            batch = cases[start:start + behavior_batch_size]
            screen_rows.extend(run_generation_batch(
                loaded, layers, model, batch, "baseline",
                layer_spec["target_layer"], layer_spec["wrong_layer_control"],
                max_new_tokens,
            ))
            done = min(start + behavior_batch_size, len(cases))
            if start == 0 or done == len(cases) or (start // behavior_batch_size) % 8 == 7:
                print(f"[{time.strftime('%H:%M:%S')}] {model} Phase570 screen {done}/{len(cases)}", flush=True)
        write_jsonl(screen_path(model), screen_rows)
        eligible_by_phenotype = {
            "stable_correct": [
                row for row in screen_rows
                if row["intended_phenotype"] == "stable_correct" and row["semantic_correct"]
            ],
            "stable_relation_confusion": [
                row for row in screen_rows
                if row["intended_phenotype"] == "stable_relation_confusion"
                and row["relation_confusion"]
            ],
        }
        selected = []
        selected_counts = {}
        for phenotype, eligible in eligible_by_phenotype.items():
            chosen = balanced(eligible, CAUSAL_CASE_CAP)
            selected_counts[phenotype] = len(chosen)
            if len(chosen) < MINIMUM_CASES_PER_PHENOTYPE:
                raise RuntimeError(
                    f"Phase570 independent phenotype failed for {model}/{phenotype}: {len(chosen)}"
                )
            selected.extend(chosen)
        selected = sorted(selected, key=lambda row: (row["intended_phenotype"], row["case_id"]))
        causal_rows = []
        for condition in CONDITIONS:
            for start in range(0, len(selected), causal_batch_size):
                batch = selected[start:start + causal_batch_size]
                causal_rows.extend(run_generation_batch(
                    loaded, layers, model, batch, condition,
                    layer_spec["target_layer"], layer_spec["wrong_layer_control"],
                    max_new_tokens,
                ))
            print(
                f"[{time.strftime('%H:%M:%S')}] {model} Phase570 causal {condition} "
                f"{len(selected)}/{len(selected)}",
                flush=True,
            )
        baseline_screen = {row["case_id"]: row for row in screen_rows}
        baseline_causal = {
            row["case_id"]: row for row in causal_rows if row["condition"] == "baseline"
        }
        mismatch = sum(
            baseline_screen[case_id]["normalized_generated"]
            != row["normalized_generated"]
            for case_id, row in baseline_causal.items()
        )
        retained_ids = {
            case_id
            for case_id, row in baseline_causal.items()
            if (
                row["intended_phenotype"] == "stable_correct"
                and row["semantic_correct"]
            ) or (
                row["intended_phenotype"] == "stable_relation_confusion"
                and row["relation_confusion"]
            )
        }
        retained_counts = {
            phenotype: sum(
                row["case_id"] in retained_ids
                and row["intended_phenotype"] == phenotype
                for row in baseline_causal.values()
            )
            for phenotype in ("stable_correct", "stable_relation_confusion")
        }
        if any(count < MINIMUM_CASES_PER_PHENOTYPE for count in retained_counts.values()):
            raise RuntimeError(
                f"Phase570 causal-batch phenotype failed for {model}: {retained_counts}"
            )
        causal_rows = [row for row in causal_rows if row["case_id"] in retained_ids]
        write_jsonl(causal_path(model), causal_rows)
        summary = {
            "schema_version": "phase570_execution_summary.v1",
            "phase_id": PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model,
            "torch_dtype": run_dtype,
            "registered_case_count": len(cases),
            "baseline_screen_count": len(screen_rows),
            "eligible_case_counts": {
                phenotype: len(rows) for phenotype, rows in eligible_by_phenotype.items()
            },
            "selected_causal_case_counts": selected_counts,
            "selected_causal_case_count": len(selected),
            "retained_paired_case_counts": retained_counts,
            "retained_paired_case_count": len(retained_ids),
            "condition_count": len(CONDITIONS),
            "causal_row_count": len(causal_rows),
            "deterministic_baseline_mismatch_count": mismatch,
            "deterministic_baseline_mismatch_rate": mismatch / len(selected),
            "target_layer": layer_spec["target_layer"],
            "wrong_layer_control": layer_spec["wrong_layer_control"],
            "runtime_seconds": time.monotonic() - started,
            "screen_rows_sha256": sha256_file(screen_path(model)),
            "causal_rows_sha256": sha256_file(causal_path(model)),
            "causal_scope": "late answer competition bridge only",
            "upstream_relation_encoding_claimed": False,
            "sealed_split_read": False,
        }
        write_json(summary_path(model), summary)
        print(summary_path(model), flush=True)
        return summary_path(model)
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--behavior-batch-size", type=int, default=24)
    parser.add_argument("--causal-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(
        args.model, args.behavior_batch_size, args.causal_batch_size,
        args.max_new_tokens, args.restart,
    )


if __name__ == "__main__":
    main()
