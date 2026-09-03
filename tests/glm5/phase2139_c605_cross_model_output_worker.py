#!/usr/bin/env python3
"""Isolated GLM4/DeepSeek behavior-first worker for C605."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase2134_c600_c605_language_transport_campaign as campaign


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("glm4", "deepseek7b"), required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_dir = args.output.parent.parent / "raw" / args.model
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_rows = campaign.read_rows(args.material)
    units = {0, 1, 2, 3, 12, 13, 14, 15}
    rows = [r for r in all_rows if r["unit"] in units and r["surface"] == "ledger" and (
        r["panel"] == "atomic" or
        (r["panel"] == "readout_factorial" and r["operation_domain"] == "badge")
    )]
    model = None
    hooks = []
    captured = []
    states = None
    try:
        model, tokenizer, device, placement = campaign.passport.previous.model_base().load_bf16(args.model)
        compiled = campaign.compile_rows(tokenizer, rows)
        write_rows(raw_dir / "compiled.jsonl", compiled)
        behavior = []
        for i, item in enumerate(compiled):
            scores = campaign.candidate_sequence_scores(model, device, item["prompt_ids"], item["candidate_ids"])
            pred = int(np.argmax(scores))
            one_token = all(len(ids) == 1 for ids in item["candidate_ids"])
            open_correct = False
            if one_token:
                ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
                mask = torch.ones_like(ids)
                pos = torch.arange(ids.shape[1], device=device)[None]
                with torch.inference_mode():
                    logits = model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                   use_cache=False, return_dict=True).logits[0, -1]
                open_correct = int(torch.argmax(logits)) == int(item["candidate_ids"][item["gold_position"]][0])
            behavior.append({"case_id": item["case_id"], "candidate_correct": pred == item["gold_position"],
                             "open_correct": open_correct, "scores": scores})
            if i % 32 == 0 or i + 1 == len(compiled):
                print(f"[{args.model} behavior] {i + 1}/{len(compiled)}", flush=True)
        write_rows(raw_dir / "behavior.jsonl", behavior)
        accuracy = float(np.mean([v["candidate_correct"] for v in behavior]))
        open_accuracy = float(np.mean([v["open_correct"] for v in behavior]))
        if accuracy < campaign.BEHAVIOR_GATE:
            save(args.output, {"status": "behavior_unqualified", "model": args.model, "rows": len(rows),
                               "behavior_accuracy": accuracy, "open_accuracy": open_accuracy,
                               "hiddenstate_ran": False, "functional_candidate": False, "placement": placement})
            return

        base = model.model
        layers = list(base.layers)
        checkpoints = len(layers) + 2
        coordinates = int(model.get_input_embeddings().weight.shape[1])
        n = len(compiled)
        states = np.lib.format.open_memmap(raw_dir / "role_last.float16.npy", mode="w+", dtype=np.float16,
                                           shape=(n, checkpoints, len(campaign.ROLES), coordinates))
        representative = None
        def hook(_module, _args, output):
            captured.append(output[0] if isinstance(output, tuple) else output)
        hooks.append(base.embed_tokens.register_forward_hook(hook))
        hooks.extend(layer.register_forward_hook(hook) for layer in layers)
        hooks.append(base.norm.register_forward_hook(hook))
        index = []
        for i, item in enumerate(compiled):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            pos = torch.arange(ids.shape[1], device=device)[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            if len(captured) != checkpoints:
                raise RuntimeError((len(captured), checkpoints))
            for q, hidden in enumerate(captured):
                for role_i, role in enumerate(campaign.ROLES):
                    at = int(item["role_positions"][role][-1])
                    states[i, q, role_i] = hidden[0, at].float().cpu().numpy().astype(np.float16)
            if representative is None and item["partition"] == "lockbox":
                relative_q = sorted(set((0, round((checkpoints - 1) * .25), round((checkpoints - 1) * .5),
                                                 round((checkpoints - 1) * .75), checkpoints - 1)))
                representative = {
                    "case_id": item["case_id"], "qpoints": relative_q,
                    "roles": list(campaign.ROLES), "coordinates": coordinates,
                    "role_last_states": [
                        [captured[q][0, int(item["role_positions"][role][-1])].float().cpu().numpy().astype(np.float16).tolist()
                         for role in campaign.ROLES]
                        for q in relative_q
                    ],
                }
            index.append({"hidden_index": i, "case_id": item["case_id"], "panel": item["panel"],
                          "family": item["family"], "operation_domain": item["operation_domain"],
                          "surface": item["surface"], "unit": item["unit"], "partition": item["partition"],
                          "cell": item["cell"], "behavior_correct": behavior[i]["candidate_correct"]})
            if i % 16 == 0 or i + 1 == n:
                states.flush()
                print(f"[{args.model} field] {i + 1}/{n}", flush=True)
        write_rows(raw_dir / "hidden_index.jsonl", index)
        pairs = campaign.transition_pairs(index)
        qpoints = sorted(set((0, round((checkpoints - 1) * .25), round((checkpoints - 1) * .5),
                              round((checkpoints - 1) * .75), checkpoints - 1)))
        metrics = {}
        representatives = {}
        grouped = defaultdict(list)
        for pair in pairs:
            grouped[pair["operation"]].append(pair)
        for operation, values in grouped.items():
            train = [p for p in values if p["partition"] == "discovery"]
            test = [p for p in values if p["partition"] == "lockbox"]
            if len(train) < 2 or len(test) < 2:
                continue
            for q in qpoints:
                tr = np.stack([np.asarray(states[p["right"]["hidden_index"], q], np.float32) - np.asarray(states[p["left"]["hidden_index"], q], np.float32) for p in train])
                te = np.stack([np.asarray(states[p["right"]["hidden_index"], q], np.float32) - np.asarray(states[p["left"]["hidden_index"], q], np.float32) for p in test])
                proto = tr.mean(axis=0)
                correct = campaign.metric(np.broadcast_to(proto, te.shape), te)
                zero = campaign.metric(np.zeros_like(te), te)
                key = f"{operation}|q{q}"
                metrics[key] = {"samples": len(test), "correct": correct, "zero": zero,
                                "gate": correct["nrmse"] <= zero["nrmse"] - campaign.CONTROL_MARGIN}
                representatives[key] = proto.tolist()
        by_operation = {}
        for operation in sorted(grouped):
            values = [v["gate"] for k, v in metrics.items() if k.startswith(operation + "|q")]
            by_operation[operation] = {"passed": int(sum(values)), "total": len(values),
                                       "pass_rate": float(np.mean(values)) if values else 0.0}
        functional = any(v["pass_rate"] >= campaign.PREDICTION_GATE for v in by_operation.values())
        raw_path = raw_dir / "role_last.float16.npy"
        save(args.output, {"status": "closed", "model": args.model, "rows": n,
                           "behavior_accuracy": accuracy, "open_accuracy": open_accuracy,
                           "hiddenstate_ran": True, "checkpoints": checkpoints, "coordinates": coordinates,
                           "qpoints": qpoints, "shape": list(states.shape),
                           "raw_path": str(raw_path.relative_to(ROOT)), "raw_bytes": raw_path.stat().st_size,
                           "metrics": metrics, "operation_summary": by_operation,
                           "representative_full_coordinates": representatives,
                           "representative_role_field": representative,
                           "functional_candidate": functional, "placement": placement,
                           "strict_interpretation": "model-internal functional topology only; physical coordinates are not aligned"})
    except Exception as error:
        save(args.output, {"status": "worker_error", "model": args.model, "error_type": type(error).__name__,
                           "error": str(error), "hiddenstate_ran": False, "functional_candidate": False})
        raise
    finally:
        for handle in hooks:
            handle.remove()
        if states is not None:
            states.flush(); del states
        campaign.passport.previous.model_base().release_bf16(model)
        gc.collect()


if __name__ == "__main__":
    main()
