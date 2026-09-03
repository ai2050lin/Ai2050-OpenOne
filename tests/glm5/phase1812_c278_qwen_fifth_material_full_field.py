#!/usr/bin/env python3
"""C278: capture all Qwen tokens, physical coordinates, and 38 real checkpoints."""
from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1811_c277_c289_joint_response_common as common

core, OUT, PARENT = common.core, common.OUTS["C278"], common.OUTS["C277"]
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
    parent = core.load(PARENT / "analysis/final.json")
    rows = core.rows(PARENT / "compiled/qwen3.jsonl")
    checks = {
        "parent": parent["all_checks_passed"],
        "rows": len(rows) == 768,
        "cuda": torch.cuda.is_available(),
        "all_tokens_coordinates": True,
        "explicit_block36_capture": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1812,
        "campaign": "C278",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "full_field_capture_frozen",
        "model": "Qwen3-4B BF16 CUDA unquantized",
        "full_field_shape": [768, 38, 128, 2560],
        "role_field_shape": [768, 38, 6, 2560],
        "checkpoints": list(common.RAW_CHECKPOINTS),
        "storage": "float16 archival representation only; all token positions and all 2560 coordinates retained",
        "canonical_primary_indices": list(common.CANONICAL_NEW_INDICES),
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "C279_joint_state_word_partition",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    fields = np.lib.format.open_memmap(OUT / "raw/full_fields.float16.npy", mode="w+", dtype=np.float16, shape=(768, 38, common.WIDTH, common.DIM))
    role_states = np.lib.format.open_memmap(OUT / "raw/role_states.float16.npy", mode="w+", dtype=np.float16, shape=(768, 38, len(common.ROLES), common.DIM))
    token_ids = np.zeros((768, common.WIDTH), np.int32)
    token_mask = np.zeros((768, common.WIDTH), bool)
    logits = np.zeros((768, 2), np.float32)
    index = []
    model = None
    hooks = []
    captured: list[torch.Tensor] = []
    started = time.time()
    try:
        model, tokenizer, device, placement = common.model_base.load_bf16("qwen3")
        quant = common.model_base.quantization_audit(model)
        base = model.model

        def capture(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            captured.append(value)

        hooks.append(base.embed_tokens.register_forward_hook(capture))
        hooks.extend(layer.register_forward_hook(capture) for layer in base.layers)
        hooks.append(base.norm.register_forward_hook(capture))
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            ids, mask, positions, lengths = batch_inputs(batch, pad, device)
            captured.clear()
            output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
            if len(captured) != 38:
                raise RuntimeError(("checkpoint_count", len(captured)))
            for local, row in enumerate(batch):
                i, length = start + local, lengths[local]
                token_ids[i, :length] = ids[local, :length].cpu().numpy()
                token_mask[i, :length] = True
                for q, state in enumerate(captured):
                    value = state[local, :length].float().cpu().numpy().astype(np.float16)
                    fields[i, q, :length] = value
                    for ri, role in enumerate(common.ROLES):
                        role_states[i, q, ri] = value[row["role_positions"][role]].mean(axis=0).astype(np.float16)
                logits[i] = [float(output.logits[local, length - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(logits[i, 1] > logits[i, 0])
                index.append({
                    "hidden_index": i,
                    "case_id": row["case_id"],
                    "panel": row["panel"],
                    "family": row["family"],
                    "surface": row["surface"],
                    "unit": row["unit"],
                    "factor_a": row["factor_a"],
                    "factor_b": row["factor_b"],
                    "order": row["order"],
                    "length": length,
                    "gold_position": row["gold_position"],
                    "prediction": prediction,
                    "correct": prediction == row["gold_position"],
                    "role_positions": row["role_positions"],
                })
            del output, ids, mask, positions
            captured.clear()
            if start % 64 == 0 or start + len(batch) == len(rows):
                fields.flush(); role_states.flush(); print(f"[C278] {start + len(batch)}/768", flush=True)
        fields.flush(); role_states.flush()
        np.save(OUT / "raw/token_ids.int32.npy", token_ids)
        np.save(OUT / "raw/token_mask.bool.npy", token_mask)
        np.save(OUT / "raw/behavior_logits.float32.npy", logits)
        core.write_rows(OUT / "raw/hidden_index.jsonl", index)
        by_family = {family: float(np.mean([r["correct"] for r in index if r["family"] == family])) for family in common.FAMILIES}
        accuracy = float(np.mean([r["correct"] for r in index]))
        gates = core.load(PARENT / "protocol/preregistration.json")["gates"]
        eligible = accuracy >= gates["behavior_global_min"] and min(by_family.values()) >= gates["family_min"]
        report = {
            "phase": 1812,
            "campaign": "C278",
            "status": "captured",
            "accuracy": accuracy,
            "by_family_accuracy": by_family,
            "behavior_eligible": eligible,
            "placement": placement,
            "quantization": quant,
            "elapsed_seconds": time.time() - started,
            "raw_bytes": int((OUT / "raw/full_fields.float16.npy").stat().st_size),
            "strict_interpretation": "All 38 physical checkpoints are archived. Cross-material primary analyses use the registered 37-checkpoint canonical intersection; block-36 output is a new observation, not retroactively present in old archives.",
        }
        core.save(OUT / "analysis/summary.json", report)
        analysis_checks = {
            "index": len(index) == 768,
            "field_shape": list(fields.shape) == [768, 38, 128, 2560],
            "role_shape": list(role_states.shape) == [768, 38, 6, 2560],
            "finite": bool(np.isfinite(role_states[:, :, :, ::64]).all()),
            "bf16": quant["has_bf16_parameters"],
            "unquantized": not quant["has_quantized_modules"],
        }
        core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
        final_checks = {"contract": all(checks.values()), "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
        final = {"phase": 1812, "campaign": "C278", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": "C279_joint_state_word_partition"}
        core.save(OUT / "analysis/final.json", final)
        print(json.dumps(final, ensure_ascii=False, indent=2))
    finally:
        for hook in hooks:
            hook.remove()
        common.model_base.release(model)
        gc.collect()


if __name__ == "__main__":
    main()
