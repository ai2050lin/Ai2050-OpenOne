"""Development-only probe for the C019 microevent timing contract."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
import phase1266_c014_free_transformer_population_certificate as p1266
import phase1268_c016_distributed_causal_support_ladder as p1268
import phase1269_c017_causal_support_funnel_confirmation as p1269
import phase1270_c018_answer_excluded_causal_recomputation as p1270


def forward_pre(model, ids, actions=None, capture=False):
    hidden = model.embed(ids)
    inputs = []
    length = ids.shape[1]
    causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=ids.device), diagonal=1)
    for layer, block in enumerate(model.blocks):
        if capture:
            inputs.append(hidden.detach().clone())
        if actions and layer in actions:
            hidden = actions[layer](hidden)
        normalized = block.attn_norm(hidden)
        batch, _, width = normalized.shape
        qkv = block.attn.qkv(normalized).view(batch, length, 3, block.attn.heads, block.attn.head_dim)
        query, key, value = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)
        weights = torch.softmax(scores.masked_fill(causal[None, None], float("-inf")), dim=-1)
        attended = torch.matmul(weights, value).transpose(1, 2).contiguous().view(batch, length, width)
        hidden = hidden + block.attn.out(attended)
        hidden = hidden + block.mlp(block.mlp_norm(hidden))
    logits = model.lm_head(model.final_norm(hidden))
    return logits, torch.stack(inputs, dim=1) if capture else None


def score(model, rows, mode, family, start_layer, device):
    values = []
    with torch.inference_mode():
        for offset in range(0, len(rows), 256):
            batch = rows[offset : offset + 256]
            ids01 = torch.tensor([row["h01_ids"] for row in batch], device=device)
            ids11 = torch.tensor([row["h11_ids"] for row in batch], device=device)
            target = torch.tensor([row["answers"]["h11"] for row in batch], device=device)
            base = torch.tensor([row["answers"]["h01"] for row in batch], device=device)
            _l01, pre01 = forward_pre(model, ids01, capture=True)
            _l11, pre11 = forward_pre(model, ids11, capture=True)
            assert pre01 is not None and pre11 is not None
            mask = p1270.support_mask(batch, family, device)
            if mode == "pre_once":
                actions01 = {start_layer: p1268.patch_action(pre11[:, start_layer], mask)}
                actions11 = {start_layer: p1268.patch_action(pre01[:, start_layer], mask)}
            elif mode == "pre_sustained":
                actions01 = {layer: p1268.patch_action(pre11[:, layer], mask) for layer in range(start_layer, len(model.blocks))}
                actions11 = {layer: p1268.patch_action(pre01[:, layer], mask) for layer in range(start_layer, len(model.blocks))}
            else:
                raise ValueError(mode)
            logits01, _ = forward_pre(model, ids01, actions01)
            logits11, _ = forward_pre(model, ids11, actions11)
            pred01 = torch.argmax(logits01[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
            pred11 = torch.argmax(logits11[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
            values.extend(((pred01 == target) & (pred11 == base)).cpu().tolist())
    return float(np.mean(values))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    config = p1266.ARCHITECTURES["middle6"]
    seed = 1_271_699_991
    p1266.set_seed(seed)
    model, training = p1266.task_module.train_model(config, seed, device)
    rows = p1269.sample_worlds("probe", 1024, 1_271_999_001)
    natural, gap = p1268.evaluate_behavior(model, rows, device)
    results = []
    for family in ("source_only", "semantic_chain_no_answer", "causal_prefix_no_answer"):
        for layer in range(config.layers):
            for mode in ("pre_once", "pre_sustained"):
                results.append({"family": family, "layer": layer, "mode": mode, "paired_switch_accuracy": score(model, rows, mode, family, layer, device)})
    payload = {"status": "development_only", "seed": seed, "training": training, "natural_accuracy": natural, "executor_gap": gap, "results": results}
    output = ROOT / "tests/glm5_temp/phase1271_microevent_timing_probe.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"natural": natural, "max": max(item["paired_switch_accuracy"] for item in results), "output": str(output)}))


if __name__ == "__main__":
    main()
