#!/usr/bin/env python3
"""Sequential model-specific behavior and HiddenState topology worker for C634."""
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

import phase2165_c630_c634_conditional_gear_identification_campaign as campaign
import phase2163_c629_model_specific_worker as old_worker


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def hidden_capture(model, device, compiled: list[dict], behavior: dict[str, dict], output_dir: Path) -> dict:
    selected = [row for row in compiled if behavior[row["case_id"]]["candidate_correct"]
                and behavior[row["case_id"]]["generated_correct"]]
    if not selected:
        return {"hiddenstate_ran": False, "reason": "no_dual_correct_rows"}
    coordinates = int(model.config.hidden_size)
    relative = (0.0, 0.25, 0.50, 0.67, 0.83, 1.0)
    target = output_dir / "relative_role_field.float16.npy"
    field = np.lib.format.open_memmap(target, mode="w+", dtype=np.float16,
                                      shape=(len(selected), len(relative), len(campaign.ROLES), coordinates))
    ledger = []
    for row_i, item in enumerate(selected):
        ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
        mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=device)[None]
        with torch.inference_mode():
            result = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                           output_hidden_states=True, return_dict=True)
        hidden = result.hidden_states
        chosen = [int(round(depth * (len(hidden) - 1))) for depth in relative]
        for q_i, hidden_i in enumerate(chosen):
            tensor = hidden[hidden_i]
            for role_i, role in enumerate(campaign.ROLES):
                position = int(item["role_positions"][role][-1])
                field[row_i, q_i, role_i] = tensor[0, position].float().cpu().numpy().astype(np.float16)
        ledger.append({"row": row_i, "case_id": item["case_id"], "family": item["family"],
                       "language": item["language"], "unit": item["unit"],
                       "semantic": item["semantic"], "code_shift": item["code_shift"],
                       "relative_hidden_indices": chosen})
        print(f"[C634 hidden] {row_i + 1}/{len(selected)}", flush=True)
    field.flush(); campaign.write_rows(output_dir / "relative_role_index.jsonl", ledger)
    values = np.asarray(field, np.float32); by_id = {row["case_id"]: row for row in ledger}
    topology = {}
    for family in campaign.FAMILIES:
        responses = []
        for language, unit in ((l, u) for l in campaign.LANGUAGES for u in (0, 6)):
            left = f"c630|{family}|{language}|canonical|u{unit:02d}|s0|k0"
            right = f"c630|{family}|{language}|canonical|u{unit:02d}|s1|k0"
            if left in by_id and right in by_id:
                responses.append(values[by_id[right]["row"]] - values[by_id[left]["row"]])
        if responses:
            rms = np.sqrt(np.mean(np.stack(responses) ** 2, axis=(0, 3)))
            normalized = rms / (np.sqrt(np.sum(rms * rms, axis=1, keepdims=True)) + 1e-12)
            topology[family] = {"pairs": len(responses), "relative_depths": list(relative),
                                "role_rms_normalized": normalized.tolist()}
        else:
            topology[family] = {"pairs": 0, "status": "NA_no_dual_correct_pairs"}
    save(output_dir / "relative_role_topology.json", topology)
    field.flush(); del values, field
    return {"hiddenstate_ran": True, "hidden_rows": len(selected), "relative_checkpoints": len(relative),
            "coordinates": coordinates, "role_field": str(target.relative_to(ROOT)),
            "topology": str((output_dir / "relative_role_topology.json").relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("glm4", "deepseek7b", "qwen3_14b"), required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for row in campaign.read_rows(args.material) if row["cross_model_subset"]]
    model = None
    try:
        model, tokenizer, device, placement, loader = old_worker.load_model(args.model)
        compiled = campaign.compile_rows(tokenizer, rows)
        scores_all = campaign.old.previous.c607.batch_candidate_scores(
            model, device, compiled, batch_size=2 if args.model == "qwen3_14b" else 8)
        behavior = []
        for i, (item, scores) in enumerate(zip(compiled, scores_all)):
            text = campaign.old.previous.c607.greedy_text(model, tokenizer, device, item["prompt_ids"], max_new_tokens=4)
            candidate = int(np.argmax(scores)); generated = campaign.generated_prediction(text)
            behavior.append({"case_id": item["case_id"], "family": item["family"],
                             "candidate_correct": candidate == item["gold_position"],
                             "generated_correct": generated == item["gold_position"], "generated_text": text})
            print(f"[{args.model}] behavior {i + 1}/{len(compiled)}", flush=True)
        campaign.write_rows(args.output.parent / "behavior.jsonl", behavior)
        grouped = defaultdict(list)
        for row in behavior:
            grouped[row["family"]].append(row)
        families = {family: {"rows": len(values),
                             "candidate_accuracy": float(np.mean([v["candidate_correct"] for v in values])),
                             "generated_accuracy": float(np.mean([v["generated_correct"] for v in values]))}
                    for family, values in grouped.items()}
        for value in families.values():
            value["qualified"] = (value["candidate_accuracy"] >= campaign.BEHAVIOR_GATE
                                  and value["generated_accuracy"] >= campaign.BEHAVIOR_GATE)
        qualified_ids = {row["case_id"] for row in behavior if families[row["family"]]["qualified"]}
        hidden = hidden_capture(model, device, [row for row in compiled if row["case_id"] in qualified_ids],
                                {row["case_id"]: row for row in behavior}, args.output.parent) if qualified_ids else {
                                    "hiddenstate_ran": False, "reason": "no_qualified_family"}
        save(args.output, {"status": "closed", "model": args.model, "rows": len(rows),
                           "families": families, "qualified_families": sum(v["qualified"] for v in families.values()),
                           "placement": placement, "loader": loader, **hidden,
                           "strict_interpretation": "Physical coordinate IDs are model-specific; only relative checkpoint-role response topology is compared."})
    except Exception as error:
        save(args.output, {"status": "worker_error", "model": args.model,
                           "error_type": type(error).__name__, "error": str(error), "hiddenstate_ran": False})
        raise
    finally:
        old_worker.release_model(args.model, model); gc.collect()


if __name__ == "__main__":
    main()
