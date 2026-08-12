#!/usr/bin/env python3
"""Temporary observational probe for physical channels in the frozen 6 heads."""
from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
from phase1000_factorial_binding_protocol import COLORS, MODEL
from phase1000_scpg_discovery import (
    batches_by_template,
    capture_residuals,
    source_patch_spec,
)
from phase1001_attention_head_discovery import (
    HEAD_DIM,
    RESULT_ROOT,
    SOURCE_DEPTH,
    capture_attention_states,
    read_json,
)
from phase1001_attention_source_path_decomposition import selected_inputs


def main():
    protocol, _, directional, _ = selected_inputs("discovery")
    candidate_ids = {
        color: int(protocol["candidate_token_ids"][color])
        for color in COLORS
    }
    frozen = read_json(
        RESULT_ROOT / "minimum_head_cut/discovery/frozen_spec.json"
    )
    event_ids = frozen["frozen_event_ids"]
    metrics = defaultdict(lambda: defaultdict(list))
    model = tokenizer = None
    try:
        model, tokenizer, device = load_model(
            MODEL, dtype=torch.bfloat16, use_8bit=False
        )
        layers = get_layers(model)
        output_weight = model.get_output_embeddings().weight.detach().float()
        color_index = {color: index for index, color in enumerate(COLORS)}
        candidate_unembed = output_weight[
            torch.tensor(
                [candidate_ids[color] for color in COLORS],
                device=device,
            )
        ]
        for batch in batches_by_template(directional, 32):
            source_cases = [item["source"] for item in batch]
            target_cases = [item["target"] for item in batch]
            _, source_residuals = capture_residuals(
                model,
                device,
                source_cases,
                (SOURCE_DEPTH,),
                candidate_ids,
            )
            source_patch = source_patch_spec(
                SOURCE_DEPTH,
                target_cases,
                source_residuals[SOURCE_DEPTH],
                "joint",
            )
            _, target_heads, _ = capture_attention_states(
                model, layers, device, target_cases, candidate_ids
            )
            _, do_heads, _ = capture_attention_states(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=source_patch,
            )
            source_index = torch.tensor(
                [
                    color_index[item["source"]["gold"]]
                    for item in batch
                ],
                device=device,
            )
            target_index = torch.tensor(
                [
                    color_index[item["target"]["gold"]]
                    for item in batch
                ],
                device=device,
            )
            unembed_direction = (
                candidate_unembed[source_index]
                - candidate_unembed[target_index]
            )
            for event_id in event_ids:
                layer_number = int(event_id.split(".")[0][1:])
                head_index = int(event_id.split(".")[1][1:])
                delta = (
                    do_heads[layer_number][:, head_index, :].float()
                    - target_heads[layer_number][:, head_index, :].float()
                )
                w_o = (
                    layers[layer_number - 1]
                    .self_attn.o_proj.weight.detach()
                    .float()
                )
                start = head_index * HEAD_DIM
                projection = w_o[:, start : start + HEAD_DIM]
                channel_logit = delta * (
                    unembed_direction @ projection
                )
                for channel in range(HEAD_DIM):
                    key = f"{event_id}.c{channel:03d}"
                    metrics[key]["delta"].extend(
                        delta[:, channel].detach().cpu().tolist()
                    )
                    metrics[key]["direct"].extend(
                        channel_logit[:, channel].detach().cpu().tolist()
                    )
                    metrics[key]["template"].extend(
                        [int(item["target"]["template"]) for item in batch]
                    )
                    metrics[key]["source_gold"].extend(
                        [item["source"]["gold"] for item in batch]
                    )
        summary = []
        for channel_id, values in metrics.items():
            direct = np.asarray(values["direct"])
            delta = np.asarray(values["delta"])
            templates = np.asarray(values["template"])
            source_gold = np.asarray(values["source_gold"])
            item = {
                "channel_id": channel_id,
                "event_id": channel_id.split(".c")[0],
                "channel_index": int(channel_id.rsplit("c", 1)[1]),
                "mean_direct_effect": float(np.mean(direct)),
                "median_direct_effect": float(np.median(direct)),
                "mean_abs_direct_effect": float(np.mean(np.abs(direct))),
                "positive_direct_rate": float(np.mean(direct > 0)),
                "mean_abs_paired_delta": float(np.mean(np.abs(delta))),
                "template_mean_direct": {
                    str(t): float(np.mean(direct[templates == t]))
                    for t in range(4)
                },
                "color_mean_direct": {
                    color: float(np.mean(direct[source_gold == color]))
                    for color in COLORS
                },
            }
            item["min_template_mean_direct"] = min(
                item["template_mean_direct"].values()
            )
            item["min_color_mean_direct"] = min(
                item["color_mean_direct"].values()
            )
            summary.append(item)
        ordered = sorted(
            summary,
            key=lambda item: (
                -item["min_template_mean_direct"],
                -item["mean_direct_effect"],
            ),
        )
        out = (
            RESULT_ROOT
            / "channel_sparsification"
            / "observation_probe.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "channel_count": len(summary),
                    "channels": summary,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        positive_total = sum(
            max(0.0, item["mean_direct_effect"]) for item in summary
        )
        for count in (8, 16, 32, 64, 96, 128, 192, 256):
            captured = sum(
                max(0.0, item["mean_direct_effect"])
                for item in ordered[:count]
            )
            print(
                f"top{count}: positive-direct coverage "
                f"{captured / max(positive_total, 1e-8):.4f}"
            )
        print(json.dumps(ordered[:12], ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
