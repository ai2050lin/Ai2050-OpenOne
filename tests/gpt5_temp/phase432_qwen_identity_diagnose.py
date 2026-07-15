#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase431_position_time_collect as p431
from hf_probe_env import load_probe_model, release_loaded
from phase432_prechoice_terminal_protocol import OUT, read_jsonl


BAD_IDS = {
    "phase432__independent_confirmation__pair_060__language_action_stable_result_control__rb__route_conflict__qwen3",
    "phase432__independent_confirmation__pair_120__language_action_stable_result_control__rb__route_conflict__qwen3",
}


@torch.inference_mode()
def main() -> None:
    loaded = load_probe_model("qwen3")
    try:
        rows = [
            row
            for row in read_jsonl(
                OUT / "open/qwen3/phase432_materialized_conditions.jsonl"
            )
            if row["condition_id"] in BAD_IDS
        ]
        final_norm, output_head = p431.final_norm_and_head(loaded)
        output = []
        for row in rows:
            ids = p431.prompt_ids(loaded, row)
            input_ids = torch.tensor([ids], dtype=torch.long, device=loaded.input_device)
            native = loaded.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = native.hidden_states[-1][:, -1]
            reconstructed = output_head(final_norm(hidden))
            native_logits = native.logits[:, -1]
            native_top = torch.topk(native_logits.float(), 4, dim=-1)
            reconstructed_top = torch.topk(reconstructed.float(), 4, dim=-1)
            output.append(
                {
                    "condition_id": row["condition_id"],
                    "native_dtype": str(native_logits.dtype),
                    "reconstructed_dtype": str(reconstructed.dtype),
                    "max_abs_error": float(
                        (native_logits.float() - reconstructed.float()).abs().max().item()
                    ),
                    "native_ids": native_top.indices[0].cpu().tolist(),
                    "native_values": native_top.values[0].cpu().tolist(),
                    "reconstructed_ids": reconstructed_top.indices[0].cpu().tolist(),
                    "reconstructed_values": reconstructed_top.values[0].cpu().tolist(),
                }
            )
        print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        release_loaded(loaded)


if __name__ == "__main__":
    main()
