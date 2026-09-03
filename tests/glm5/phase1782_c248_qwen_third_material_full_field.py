#!/usr/bin/env python3
"""C248: capture the third material's complete Qwen embedding/HiddenState field."""
from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1780_c246_c255_event_hypergraph_common as common

core = common.core
OUT = common.OUTS["C248"]
PARENT = common.OUTS["C247"]
BATCH = 2


def batch_inputs(rows: list[dict], pad: int, device):
    ids = torch.full((len(rows), common.WIDTH), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    positions = torch.zeros_like(ids)
    lengths = []
    for i, row in enumerate(rows):
        values = row["prompt_ids"]
        lengths.append(len(values))
        ids[i, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        mask[i, :len(values)] = 1
        positions[i, :len(values)] = torch.arange(len(values), device=device)
    return ids, mask, positions, lengths


@torch.inference_mode()
def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(PARENT / "audit/independent_final_audit.json")
    rows = core.rows(PARENT / "compiled/qwen3.jsonl")
    checks = {"authorization": parent["all_checks_passed"] and parent["authorization"] == "C248_Qwen_full_field_capture_once", "rows": len(rows) == 768, "cuda": torch.cuda.is_available()}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1782, "campaign": "C248", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "capture_frozen", "model": "Qwen3-4B BF16 CUDA unquantized", "shape": [768, 37, 128, 2560],
        "checkpoints": "embedding plus every block output/pre-next-block-norm input, including the final model hidden-state checkpoint",
        "all_tokens_all_coordinates": True, "batch": BATCH, "behavior_gate": core.load(PARENT / "protocol/preregistration.json")["behavior_gate"],
        "producer_sha256": core.sha(Path(__file__)), "authorization": "capture_once_then_C249_and_C250_reuse_without_reload",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    fields = np.lib.format.open_memmap(OUT / "raw/full_fields.float16.npy", mode="w+", dtype=np.float16, shape=(768, 37, common.WIDTH, common.DIM))
    logits = np.zeros((768, 2), np.float32)
    index = []
    model = None
    started = time.time()
    try:
        model, tokenizer, device, placement = common.previous.load_bf16("qwen3")
        quant = common.previous.quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            ids, mask, positions, lengths = batch_inputs(batch, pad, device)
            output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True, output_hidden_states=True)
            if len(output.hidden_states) != 37:
                raise RuntimeError(len(output.hidden_states))
            for local, row in enumerate(batch):
                i = start + local
                length = lengths[local]
                for q, state in enumerate(output.hidden_states):
                    fields[i, q, :length] = state[local, :length].float().cpu().numpy().astype(np.float16)
                logits[i] = [float(output.logits[local, length - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(logits[i, 1] > logits[i, 0])
                index.append({
                    "hidden_index": i, "case_id": row["case_id"], "panel": row["panel"], "family": row["family"],
                    "surface": row["surface"], "unit": row["unit"], "factor_a": row["factor_a"], "factor_b": row["factor_b"],
                    "order": row["order"], "length": length, "gold_position": row["gold_position"], "prediction": prediction,
                    "correct": prediction == row["gold_position"], "role_positions": row["role_positions"],
                })
            del output, ids, mask, positions
            if start % 64 == 0 or start + len(batch) == len(rows):
                fields.flush()
                print(f"[C248] full fields {start + len(batch)}/{len(rows)}", flush=True)
        fields.flush()
        np.save(OUT / "raw/behavior_logits.float32.npy", logits)
        core.write_rows(OUT / "raw/hidden_index.jsonl", index)
        by_family = {family: float(np.mean([row["correct"] for row in index if row["family"] == family])) for family in (*common.FAMILIES, "nested_attitude")}
        global_accuracy = float(np.mean([row["correct"] for row in index]))
        gate = protocol["behavior_gate"]
        eligible = global_accuracy >= gate["global_min"] and min(by_family[f] for f in common.FAMILIES) >= gate["each_core_family_min"] and by_family["nested_attitude"] >= gate["nested_min"]

        # Numerical replay is deliberately tiny; it checks execution identity, not semantics.
        replay_rows = rows[:2]
        ids, mask, positions, lengths = batch_inputs(replay_rows, pad, device)
        replay = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True, output_hidden_states=True)
        replay_max = 0.0
        for local in range(2):
            length = lengths[local]
            for q, state in enumerate(replay.hidden_states):
                replay_max = max(replay_max, float(np.max(np.abs(state[local, :length].float().cpu().numpy() - np.asarray(fields[local, q, :length], np.float32)))))
        report = {
            "global_accuracy": global_accuracy, "by_family_accuracy": by_family, "behavior_eligible": eligible,
            "placement": placement, "quantization": quant, "elapsed_seconds": time.time() - started,
            "float16_values": int(fields.size), "field_bytes": int(fields.nbytes), "numerical_replay_max_abs_after_float16": replay_max,
            "strict_boundary": "The archive is a complete activation field. It contains no attention, MLP, gradient, weight, or causal-edge observation.",
        }
        core.save(OUT / "analysis/behavior_capture.json", report)
        capture_checks = {
            "rows": len(index) == 768, "shape": list(fields.shape) == [768, 37, 128, 2560], "finite": bool(np.isfinite(logits).all()),
            "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"], "replay_within_float16": replay_max <= 0.0625,
        }
        core.save(OUT / "audit/internal_capture_audit.json", {"checks": capture_checks, "all_checks_passed": all(capture_checks.values())})
        final_checks = {"contract": True, "capture": all(capture_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
        final = {"phase": 1782, "campaign": "C248", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": "C249_prospective_core_and_C250_full_token_observation"}
        core.save(OUT / "analysis/final.json", final)
        core.save(OUT / "audit/independent_final_audit.json", {"checks": final_checks, "all_checks_passed": all(final_checks.values()), "authorization": final["next_authorization"]})
        print(json.dumps(final, indent=2))
    finally:
        common.previous.release(model)
        gc.collect()


if __name__ == "__main__":
    main()
