#!/usr/bin/env python3
"""Sequential cross-model translation behavior and full-coordinate topology worker."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase2163_c629_model_specific_worker as loader
import phase2176_c641_c645_translation_relative_encoding_campaign as campaign


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def capture(model, device, compiled: list[dict], output_dir: Path) -> dict:
    core = model.model
    modules = [core.embed_tokens, *list(core.layers), core.norm]
    coordinates = int(core.embed_tokens.weight.shape[1])
    relative = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    q_indices = [int(round(value * (len(modules) - 1))) for value in relative]
    field_path = output_dir / "translation_role_field.float16.npy"
    fields = np.lib.format.open_memmap(
        field_path, mode="w+", dtype=np.float16,
        shape=(len(compiled), len(q_indices), len(campaign.ROLES), coordinates))
    captured = []
    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)
    handles = [module.register_forward_hook(hook) for module in modules]
    try:
        for row_i, item in enumerate(compiled):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=device)[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      use_cache=False, return_dict=True)
            for qi, q in enumerate(q_indices):
                tensor = captured[q][0]
                for role_i, role in enumerate(campaign.ROLES):
                    fields[row_i, qi, role_i] = tensor[
                        int(item["role_positions"][role][-1])].float().cpu().numpy().astype(np.float16)
            print(f"[cross hidden] {row_i + 1}/{len(compiled)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    fields.flush()
    by_key = {(row["concept_index"], row["source_language"], row["target_language"]): i
              for i, row in enumerate(compiled)}
    response_sums = np.zeros((len(q_indices), len(campaign.ROLES)), dtype=np.float64)
    response_count = 0
    triangle_num = 0.0; triangle_den = 0.0; triangle_count = 0
    for concept in sorted({row["concept_index"] for row in compiled}):
        required = [(concept, "zh", "en"), (concept, "zh", "fr"),
                    (concept, "en", "en"), (concept, "en", "fr")]
        if not all(key in by_key for key in required):
            continue
        ze, zf, ee, ef = [np.asarray(fields[by_key[key]], dtype=np.float32) for key in required]
        switch = zf - ze
        response_sums += np.mean(np.square(switch), axis=2)
        response_count += 1
        residual = switch - (ef - ee)
        triangle_num += float(np.square(residual).sum())
        triangle_den += float(np.square(switch).sum())
        triangle_count += 1
    topology = np.sqrt(response_sums / max(response_count, 1))
    topology /= np.maximum(np.sqrt(np.square(topology).sum(axis=1, keepdims=True)), 1e-12)
    topology_rows = [{"relative_depth": relative[i], "checkpoint": q_indices[i],
                      "role_rms_normalized": topology[i].tolist()}
                     for i in range(len(q_indices))]
    save(output_dir / "translation_response_topology.json", topology_rows)
    fields.flush(); campaign.close_mmap(fields)
    return {"hiddenstate_ran": True, "hidden_rows": len(compiled),
            "checkpoints_total": len(modules), "sampled_checkpoints": q_indices,
            "coordinates": coordinates, "field": str(field_path.relative_to(ROOT)),
            "topology": str((output_dir / "translation_response_topology.json").relative_to(ROOT)),
            "translation_triangle_nrmse": float(np.sqrt(triangle_num / max(triangle_den, 1e-12))),
            "translation_triangle_concepts": triangle_count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("glm4", "deepseek7b", "qwen3_14b"), required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for row in campaign.read_rows(args.material) if row["cross_model_subset"]]
    model = None
    try:
        model, tokenizer, device, placement, loader_name = loader.load_model(args.model)
        compiled = campaign.compile_rows(tokenizer, rows)
        scores_all = campaign.base.old.previous.c607.batch_candidate_scores(
            model, device, compiled, batch_size=2 if args.model == "qwen3_14b" else 8)
        behavior = []
        for i, (item, scores) in enumerate(zip(compiled, scores_all)):
            text = campaign.base.old.previous.c607.greedy_text(
                model, tokenizer, device, item["prompt_ids"], max_new_tokens=8)
            prediction, generated_correct = campaign.evaluate_generation(text, item)
            behavior.append({"case_id": item["case_id"], "route": f'{item["source_language"]}-{item["target_language"]}',
                             "candidate_correct": int(np.argmax(scores)) == item["gold_position"],
                             "generated_text": text, "generated_prediction": prediction,
                             "generated_correct": generated_correct})
            print(f"[{args.model}] behavior {i + 1}/{len(compiled)}", flush=True)
        campaign.write_rows(args.output.parent / "behavior.jsonl", behavior)
        groups = defaultdict(list)
        for row in behavior:
            groups[row["route"]].append(row)
        route_metrics = {}
        for route, values in groups.items():
            ca = float(np.mean([row["candidate_correct"] for row in values]))
            ga = float(np.mean([row["generated_correct"] for row in values]))
            route_metrics[route] = {"rows": len(values), "candidate_accuracy": ca,
                                    "generated_accuracy": ga,
                                    "qualified": ca >= campaign.BEHAVIOR_GATE and ga >= campaign.BEHAVIOR_GATE}
        qualified = len(route_metrics) == 4 and all(row["qualified"] for row in route_metrics.values())
        hidden = {"hiddenstate_ran": False}
        if qualified:
            correct_ids = {row["case_id"] for row in behavior
                           if row["candidate_correct"] and row["generated_correct"]}
            hidden = capture(model, device, [row for row in compiled if row["case_id"] in correct_ids],
                             args.output.parent)
        save(args.output, {"status": "closed" if qualified else "behavior_unqualified",
                           "model": args.model, "rows": len(rows), "route_metrics": route_metrics,
                           "placement": placement, "loader": loader_name, **hidden,
                           "strict_interpretation": "Only relative-depth role topology is compared; physical coordinate IDs remain model-specific."})
    except Exception as error:
        save(args.output, {"status": "worker_error", "model": args.model,
                           "error_type": type(error).__name__, "error": str(error),
                           "hiddenstate_ran": False})
        raise
    finally:
        loader.release_model(args.model, model); gc.collect()


if __name__ == "__main__":
    main()
