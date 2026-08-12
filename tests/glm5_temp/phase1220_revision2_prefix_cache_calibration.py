"""Numerically compare Phase 1220 prefix-cache scoring with brute-force scoring."""

from __future__ import annotations

import importlib.util
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))


def load_phase_module() -> Any:
    path = TEST_ROOT / "phase1220_object_relation_value_master_task.py"
    spec = importlib.util.spec_from_file_location("phase1220_calibration_target", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Phase 1220 module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def brute_force_scores(
    model: Any, device: torch.device, rows: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    entries = []
    for row in rows:
        prompt = [int(value) for value in row["input_ids"]]
        for candidate in row["candidates"]:
            continuation = [int(value) for value in row["candidate_token_ids"][candidate]]
            entries.append(
                {
                    "item_id": row["item_id"],
                    "candidate": candidate,
                    "prompt_length": len(prompt),
                    "continuation": continuation,
                    "sequence": prompt + continuation,
                    "sequence_length": len(prompt) + len(continuation),
                }
            )
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[entry["sequence_length"]].append(entry)
    scores: dict[str, dict[str, float]] = {}
    for length in sorted(groups):
        values = groups[length]
        for start_index in range(0, len(values), 12):
            batch = values[start_index : start_index + 12]
            sequence = torch.tensor([entry["sequence"] for entry in batch], dtype=torch.long, device=device)
            keep = max(len(entry["continuation"]) + 1 for entry in batch)
            with torch.inference_mode():
                output = model(
                    input_ids=sequence,
                    use_cache=False,
                    logits_to_keep=keep,
                    return_dict=True,
                )
            output_start = sequence.shape[1] - output.logits.shape[1]
            for batch_index, entry in enumerate(batch):
                token_scores = []
                for offset, token_id in enumerate(entry["continuation"]):
                    absolute_position = entry["prompt_length"] + offset - 1
                    logits = output.logits[batch_index, absolute_position - output_start].float()
                    score = logits[token_id] - torch.logsumexp(logits, dim=-1)
                    token_scores.append(float(score.item()))
                key = f"{entry['item_id']}::{entry['candidate']}"
                scores[key] = {
                    "sum": sum(token_scores),
                    "mean": sum(token_scores) / len(token_scores),
                }
            del output, sequence
    return scores


def main() -> None:
    phase = load_phase_module()
    all_materials = phase.build_materials()
    sample_indices = [
        round(index * (len(all_materials) - 1) / 31)
        for index in range(32)
    ]
    materials = [all_materials[index] for index in sample_indices]
    manifest, _ = phase.build_manifest(materials)
    model, _tokenizer, device, _placement = phase.load_fp16("qwen3")
    try:
        shared, _stats = phase.candidate_scores(model, device, manifest)
        brute = brute_force_scores(model, device, manifest)
        errors = []
        details = []
        winner_matches = []
        for row in manifest:
            cached_means = {}
            direct_means = {}
            for candidate in row["candidates"]:
                cached = shared[row["item_id"]][candidate]["sum_log_probability"]
                direct = brute[f"{row['item_id']}::{candidate}"]["sum"]
                error = abs(cached - direct)
                errors.append(error)
                cached_means[candidate] = shared[row["item_id"]][candidate]["mean_log_probability"]
                direct_means[candidate] = brute[f"{row['item_id']}::{candidate}"]["mean"]
                details.append(
                    {
                        "item_id": row["item_id"],
                        "candidate": candidate,
                        "token_count": len(row["candidate_token_ids"][candidate]),
                        "cached": cached,
                        "direct": direct,
                        "abs_error": error,
                    }
                )
            winner_matches.append(
                max(cached_means, key=cached_means.get) == max(direct_means, key=direct_means.get)
            )
        maximum = max(errors)
        mean_error = sum(errors) / len(errors)
        winner_rate = sum(winner_matches) / len(winner_matches)
        if not math.isfinite(maximum) or maximum > 0.25 or winner_rate != 1.0:
            print(sorted(details, key=lambda item: item["abs_error"], reverse=True)[:16])
            raise RuntimeError(
                "prefix-cache calibration failed: "
                f"max_abs_error={maximum}, winner_rate={winner_rate}"
            )
        print(
            {
                "status": "passed",
                "row_count": len(manifest),
                "candidate_count": len(errors),
                "max_abs_error": maximum,
                "mean_abs_error": mean_error,
                "winner_agreement_rate": winner_rate,
            }
        )
    finally:
        phase.release_fp16(model)


if __name__ == "__main__":
    main()
