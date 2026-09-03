#!/usr/bin/env python3
"""Sequential model worker for the C650 fixed-target composition panel."""
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
import phase2183_c647_c650_fixed_target_concept_bridge_campaign as campaign


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _capture(model, device, compiled: list[dict], output_dir: Path) -> dict:
    core = model.model
    modules = [core.embed_tokens, *list(core.layers), core.norm]
    coordinates = int(core.embed_tokens.weight.shape[1])
    relative_depths = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    checkpoints = [int(round(depth * (len(modules) - 1))) for depth in relative_depths]
    field_path = output_dir / "temporary_role_field.float16.npy"
    field = np.lib.format.open_memmap(
        field_path, mode="w+", dtype=np.float16,
        shape=(len(compiled), len(checkpoints), len(campaign.ROLES), coordinates))
    captured = []
    handles = [module.register_forward_hook(lambda _m, _a, output: captured.append(
        output[0] if isinstance(output, tuple) else output)) for module in modules]
    try:
        for row_i, item in enumerate(compiled):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=device)[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      use_cache=False, return_dict=True)
            for qi, q in enumerate(checkpoints):
                values = captured[q][0]
                for role_i, role in enumerate(campaign.ROLES):
                    field[row_i, qi, role_i] = values[
                        int(item["role_positions"][role][-1])].float().cpu().numpy().astype(np.float16)
            print(f"[cross hidden] {row_i + 1}/{len(compiled)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    field.flush()
    by_key = {(row["concept_uid"], row["target_language"]): i for i, row in enumerate(compiled)}
    pair_rms = np.zeros((2, len(checkpoints), len(campaign.ROLES)), np.float64)
    pair_counts = np.zeros(2, np.int32)
    by_family = defaultdict(list)
    for row in compiled:
        by_family[row["concept_family"]].append(row["concept_uid"])
    for family in campaign.FAMILIES:
        uids = sorted(set(by_family[family]), key=lambda uid: campaign.RECORD_BY_UID[uid]["family_rank"])
        for start in range(0, len(uids), 2):
            a, b = uids[start:start + 2]
            for target_i, target in enumerate(("en", "fr")):
                if (a, target) not in by_key or (b, target) not in by_key:
                    continue
                delta = (field[by_key[(b, target)]].astype(np.float32) -
                         field[by_key[(a, target)]].astype(np.float32))
                pair_rms[target_i] += np.mean(np.square(delta), axis=2)
                pair_counts[target_i] += 1
    pair_rms = np.sqrt(pair_rms / np.maximum(pair_counts[:, None, None], 1))
    pair_rms /= np.maximum(np.sqrt(np.square(pair_rms).sum(axis=2, keepdims=True)), 1e-12)
    topology = {
        target: [{"relative_depth": relative_depths[qi], "checkpoint": checkpoints[qi],
                  "role_rms_normalized": pair_rms[target_i, qi].tolist()}
                 for qi in range(len(checkpoints))]
        for target_i, target in enumerate(("en", "fr"))
    }
    save(output_dir / "concept_pair_relative_topology.json", topology)
    field.flush(); campaign.close_mmap(field)
    deleted = field_path.stat().st_size
    field_path.unlink()
    return {"hiddenstate_ran": True, "hidden_rows": len(compiled),
            "checkpoints_total": len(modules), "sampled_checkpoints": checkpoints,
            "coordinates": coordinates,
            "relative_topology": str((output_dir / "concept_pair_relative_topology.json").relative_to(ROOT)),
            "temporary_field_bytes_deleted": deleted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("glm4", "deepseek7b", "qwen3_14b"), required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for row in campaign.read_rows(args.material) if row.get("cross_model_subset")]
    model = None
    try:
        model, tokenizer, device, placement, loader_name = loader.load_model(args.model)
        compiled = campaign.compile_rows(tokenizer, rows)
        scores_all = campaign.translation.base.old.previous.c607.batch_candidate_scores(
            model, device, compiled, batch_size=2 if args.model == "qwen3_14b" else 8)
        behavior = []
        for row_i, (item, scores) in enumerate(zip(compiled, scores_all)):
            text = campaign.translation.base.old.previous.c607.greedy_text(
                model, tokenizer, device, item["prompt_ids"], max_new_tokens=10)
            prediction, generated_correct = campaign.translation.evaluate_generation(text, item)
            behavior.append({"case_id": item["case_id"], "target_language": item["target_language"],
                             "candidate_correct": int(np.argmax(scores)) == item["gold_position"],
                             "generated_text": text, "generated_prediction": prediction,
                             "generated_correct": generated_correct})
            print(f"[{args.model}] behavior {row_i + 1}/{len(compiled)}", flush=True)
        campaign.write_rows(args.output.parent / "behavior.jsonl", behavior)
        routes = {}
        for target in ("en", "fr"):
            values = [row for row in behavior if row["target_language"] == target]
            ca = float(np.mean([row["candidate_correct"] for row in values]))
            ga = float(np.mean([row["generated_correct"] for row in values]))
            routes[target] = {"rows": len(values), "candidate_accuracy": ca,
                              "generated_accuracy": ga,
                              "qualified": ca >= campaign.BEHAVIOR_GATE and ga >= campaign.BEHAVIOR_GATE}
        qualified = all(value["qualified"] for value in routes.values())
        hidden = {"hiddenstate_ran": False}
        if qualified:
            hidden = _capture(model, device, compiled, args.output.parent)
        save(args.output, {"status": "closed" if qualified else "behavior_unqualified",
                           "model": args.model, "rows": len(rows), "route_metrics": routes,
                           "placement": placement, "loader": loader_name, **hidden,
                           "strict_interpretation": (
                               "Only model-relative role topology is reported. Physical coordinate IDs are not aligned across models.")})
    except Exception as error:
        save(args.output, {"status": "worker_error", "model": args.model,
                           "error_type": type(error).__name__, "error": str(error),
                           "hiddenstate_ran": False})
        raise
    finally:
        loader.release_model(args.model, model); gc.collect()


if __name__ == "__main__":
    main()
