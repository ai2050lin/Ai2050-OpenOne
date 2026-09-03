#!/usr/bin/env python3
"""C264: capture every token and coordinate for a fourth Qwen material system."""
from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1797_c263_c272_state_operator_common as common

core, OUT, PARENT = common.core, common.OUTS["C264"], common.OUTS["C263"]
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
    checks = {"authorization": parent["all_checks_passed"], "rows": len(rows) == 768, "cuda": torch.cuda.is_available(), "all_tokens_coordinates": True}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1798, "campaign": "C264", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "capture_frozen",
        "model": "Qwen3-4B BF16 CUDA unquantized", "full_field_shape": [768, 37, 128, 2560], "role_field_shape": [768, 37, 6, 2560],
        "checkpoints": "embedding plus all 36 block outputs/pre-next-block-norm states", "compression": "float16 storage only; no dimension selection or projection",
        "behavior_policy": core.load(PARENT / "protocol/preregistration.json")["behavior_policy"], "producer_sha256": core.sha(Path(__file__)), "authorization": "C265_coordinate_passports",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    fields = np.lib.format.open_memmap(OUT / "raw/full_fields.float16.npy", mode="w+", dtype=np.float16, shape=(768, 37, common.WIDTH, common.DIM))
    role_states = np.lib.format.open_memmap(OUT / "raw/role_states.float16.npy", mode="w+", dtype=np.float16, shape=(768, 37, len(common.ROLES), common.DIM))
    token_ids = np.zeros((768, common.WIDTH), np.int32)
    token_mask = np.zeros((768, common.WIDTH), bool)
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
                i, length = start + local, lengths[local]
                token_ids[i, :length] = ids[local, :length].cpu().numpy()
                token_mask[i, :length] = True
                for q, state in enumerate(output.hidden_states):
                    value = state[local, :length].float().cpu().numpy().astype(np.float16)
                    fields[i, q, :length] = value
                    for ri, role in enumerate(common.ROLES):
                        role_states[i, q, ri] = value[row["role_positions"][role]].mean(axis=0).astype(np.float16)
                logits[i] = [float(output.logits[local, length - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                prediction = int(logits[i, 1] > logits[i, 0])
                index.append({
                    "hidden_index": i, "case_id": row["case_id"], "panel": row["panel"], "family": row["family"], "surface": row["surface"],
                    "unit": row["unit"], "factor_a": row["factor_a"], "factor_b": row["factor_b"], "order": row["order"], "length": length,
                    "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"], "role_positions": row["role_positions"],
                })
            del output, ids, mask, positions
            if start % 64 == 0 or start + len(batch) == len(rows):
                fields.flush(); role_states.flush(); print(f"[C264] {start + len(batch)}/768", flush=True)
        fields.flush(); role_states.flush()
        np.save(OUT / "raw/token_ids.int32.npy", token_ids)
        np.save(OUT / "raw/token_mask.bool.npy", token_mask)
        np.save(OUT / "raw/behavior_logits.float32.npy", logits)
        core.write_rows(OUT / "raw/hidden_index.jsonl", index)
        core_rows = [r for r in index if r["panel"] == "core"]
        nested = [r for r in index if r["panel"] == "nested_composition"]
        by_family = {family: float(np.mean([r["correct"] for r in core_rows if r["family"] == family])) for family in common.FAMILIES}
        accuracy = float(np.mean([r["correct"] for r in index]))
        nested_accuracy = float(np.mean([r["correct"] for r in nested]))
        gate = core.load(PARENT / "protocol/preregistration.json")["gates"]
        eligible = accuracy >= gate["behavior_global_min"] and min(by_family.values()) >= gate["family_min"] and nested_accuracy >= gate["family_min"]
        report = {"phase": 1798, "campaign": "C264", "status": "captured", "accuracy": accuracy, "by_family_accuracy": by_family, "nested_accuracy": nested_accuracy, "behavior_eligible": eligible, "placement": placement, "quantization": quant, "elapsed_seconds": time.time() - started, "raw_bytes": int((OUT / "raw/full_fields.float16.npy").stat().st_size), "strict_interpretation": "The archive preserves all token positions and 2560 activation coordinates. Behavior errors remain archived; downstream mechanism claims are stratified by complete correct factorial groups."}
        core.save(OUT / "analysis/summary.json", report)
        analysis_checks = {"index": len(index) == 768, "field_shape": list(fields.shape) == [768, 37, 128, 2560], "role_shape": list(role_states.shape) == [768, 37, 6, 2560], "finite": bool(np.isfinite(role_states[:, :, :, ::64]).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}
        core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
        final_checks = {"contract": all(checks.values()), "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
        final = {"phase": 1798, "campaign": "C264", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": "C265_coordinate_passports"}
        core.save(OUT / "analysis/final.json", final)
        print(json.dumps(final, indent=2))
    finally:
        common.previous.release(model)
        gc.collect()


if __name__ == "__main__":
    main()
