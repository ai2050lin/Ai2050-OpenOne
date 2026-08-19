"""Development-only probe of first-block answer-position writes."""

from __future__ import annotations

import json
import gc
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


def block_parts(block, hidden, causal):
    normalized = block.attn_norm(hidden)
    batch, length, width = normalized.shape
    qkv = block.attn.qkv(normalized).view(batch, length, 3, block.attn.heads, block.attn.head_dim)
    query, key, value = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)
    weights = torch.softmax(scores.masked_fill(causal[None, None], float("-inf")), dim=-1)
    attended = torch.matmul(weights, value).transpose(1, 2).contiguous().view(batch, length, width)
    attn_write = block.attn.out(attended)
    after_attn = hidden + attn_write
    mlp_write = block.mlp(block.mlp_norm(after_attn))
    return attn_write, after_attn, mlp_write


def capture(model, ids):
    hidden = model.embed(ids)
    causal = torch.triu(torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool, device=ids.device), diagonal=1)
    traces = []
    for block in model.blocks:
        attn_write, after_attn, mlp_write = block_parts(block, hidden, causal)
        hidden = after_attn + mlp_write
        traces.append({"attn_write": attn_write.detach().clone(), "mlp_write": mlp_write.detach().clone(), "after_block": hidden.detach().clone()})
    return traces


def forward_patch(model, ids, donor, layer, stage):
    hidden = model.embed(ids)
    causal = torch.triu(torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool, device=ids.device), diagonal=1)
    for index, block in enumerate(model.blocks):
        attn_write, after_attn, mlp_write = block_parts(block, hidden, causal)
        if index == layer and stage == "attn_write_answer":
            attn_write = attn_write.clone()
            attn_write[:, 22] = donor[index]["attn_write"][:, 22]
            after_attn = hidden + attn_write
            mlp_write = block.mlp(block.mlp_norm(after_attn))
        elif index == layer and stage == "mlp_write_answer":
            mlp_write = mlp_write.clone()
            mlp_write[:, 22] = donor[index]["mlp_write"][:, 22]
        hidden = after_attn + mlp_write
        if index == layer and stage == "after_block_answer":
            hidden = hidden.clone()
            hidden[:, 22] = donor[index]["after_block"][:, 22]
    return model.lm_head(model.final_norm(hidden))


def forward_patch_all(model, ids, donor, program, selected_layers=None):
    hidden = model.embed(ids)
    causal = torch.triu(torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool, device=ids.device), diagonal=1)
    selected = set(range(len(model.blocks))) if selected_layers is None else set(selected_layers)
    for index, block in enumerate(model.blocks):
        attn_write, after_attn, mlp_write = block_parts(block, hidden, causal)
        if index in selected and program in ("all_attn_writes", "all_attn_and_mlp_writes"):
            attn_write = attn_write.clone()
            attn_write[:, 22] = donor[index]["attn_write"][:, 22]
            after_attn = hidden + attn_write
            mlp_write = block.mlp(block.mlp_norm(after_attn))
        if index in selected and program in ("all_mlp_writes", "all_attn_and_mlp_writes"):
            mlp_write = mlp_write.clone()
            mlp_write[:, 22] = donor[index]["mlp_write"][:, 22]
        hidden = after_attn + mlp_write
        if index in selected and program == "all_after_block_states":
            hidden = hidden.clone()
            hidden[:, 22] = donor[index]["after_block"][:, 22]
    return model.lm_head(model.final_norm(hidden))


def score(model, rows, stage, layer, donor_panel, expected_name, device):
    patch_ok, reverse_ok, false_target = [], [], []
    with torch.inference_mode():
        for offset in range(0, len(rows), 256):
            batch = rows[offset : offset + 256]
            ids01 = torch.tensor([row["h01_ids"] for row in batch], device=device)
            ids11 = torch.tensor([row["h11_ids"] for row in batch], device=device)
            donor_ids = torch.tensor([row[f"{donor_panel}_ids"] for row in batch], device=device)
            trace01 = capture(model, ids01)
            trace11 = capture(model, ids11)
            donor_trace = capture(model, donor_ids)
            patch = forward_patch(model, ids01, donor_trace, layer, stage)
            reverse = forward_patch(model, ids11, trace01, layer, stage)
            patch_pred = torch.argmax(patch[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
            reverse_pred = torch.argmax(reverse[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
            expected = torch.tensor([row["answers"][expected_name] for row in batch], device=device)
            base = torch.tensor([row["answers"]["h01"] for row in batch], device=device)
            target = torch.tensor([row["answers"]["h11"] for row in batch], device=device)
            patch_ok.extend((patch_pred == expected).cpu().tolist())
            reverse_ok.extend((reverse_pred == base).cpu().tolist())
            false_target.extend((patch_pred == target).cpu().tolist())
    return {"patch_expected": float(np.mean(patch_ok)), "reverse_base": float(np.mean(reverse_ok)), "patch_false_target": float(np.mean(false_target))}


def score_all(model, rows, program, donor_panel, expected_name, device, selected_layers=None):
    patch_ok, reverse_ok, false_target = [], [], []
    with torch.inference_mode():
        for offset in range(0, len(rows), 256):
            batch = rows[offset : offset + 256]
            ids01 = torch.tensor([row["h01_ids"] for row in batch], device=device)
            ids11 = torch.tensor([row["h11_ids"] for row in batch], device=device)
            donor_ids = torch.tensor([row[f"{donor_panel}_ids"] for row in batch], device=device)
            trace01 = capture(model, ids01)
            donor_trace = capture(model, donor_ids)
            patch = forward_patch_all(model, ids01, donor_trace, program, selected_layers)
            reverse = forward_patch_all(model, ids11, trace01, program, selected_layers)
            patch_pred = torch.argmax(patch[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
            reverse_pred = torch.argmax(reverse[:, -1, p1266.CANDIDATE_SLICE], dim=-1)
            expected = torch.tensor([row["answers"][expected_name] for row in batch], device=device)
            base = torch.tensor([row["answers"]["h01"] for row in batch], device=device)
            target = torch.tensor([row["answers"]["h11"] for row in batch], device=device)
            patch_ok.extend((patch_pred == expected).cpu().tolist())
            reverse_ok.extend((reverse_pred == base).cpu().tolist())
            false_target.extend((patch_pred == target).cpu().tolist())
    return {"patch_expected": float(np.mean(patch_ok)), "reverse_base": float(np.mean(reverse_ok)), "patch_false_target": float(np.mean(false_target))}


def main():
    device = torch.device("cuda")
    config = p1266.ARCHITECTURES["middle6"]
    rows = p1269.sample_worlds("probe", 1024, 1_271_999_002)
    attempts = []
    model = None
    training = None
    natural = 0.0
    gap = float("inf")
    seed = -1
    for candidate in (1_271_699_992, 1_271_699_993, 1_271_699_994, 1_271_699_995, 1_271_699_996):
        p1266.set_seed(candidate)
        candidate_model, candidate_training = p1266.task_module.train_model(config, candidate, device)
        candidate_natural, candidate_gap = p1268.evaluate_behavior(candidate_model, rows, device)
        passed = min(candidate_training["accuracy_overall"], candidate_training["accuracy_direct"], candidate_training["accuracy_code"], candidate_natural) >= 0.995 and candidate_gap <= 2.0e-4
        attempts.append({"seed": candidate, "natural": candidate_natural, "training": candidate_training, "gap": candidate_gap, "passed": passed})
        if passed:
            model, training, natural, gap, seed = candidate_model, candidate_training, candidate_natural, candidate_gap, candidate
            break
        del candidate_model
        gc.collect()
        torch.cuda.empty_cache()
    if model is None or training is None:
        raise RuntimeError("development behavior pool exhausted")
    results = []
    for layer in (0, 1):
        for stage in ("attn_write_answer", "mlp_write_answer", "after_block_answer"):
            results.append({"layer": layer, "stage": stage, "donor": "h11", **score(model, rows, stage, layer, "h11", "h11", device)})
            results.append({"layer": layer, "stage": stage, "donor": "hwrong11", **score(model, rows, stage, layer, "hwrong11", "hwrong11", device)})
    coalitions = []
    for program in ("all_attn_writes", "all_mlp_writes", "all_attn_and_mlp_writes", "all_after_block_states"):
        coalitions.append({"program": program, "donor": "h11", **score_all(model, rows, program, "h11", "h11", device)})
        coalitions.append({"program": program, "donor": "hwrong11", **score_all(model, rows, program, "hwrong11", "hwrong11", device)})
    prefixes = []
    for end in range(config.layers):
        selected = list(range(end + 1))
        prefixes.append({"end_layer": end, "layers": selected, **score_all(model, rows, "all_attn_writes", "h11", "h11", device, selected)})
    payload = {"status": "development_only", "seed": seed, "attempts": attempts, "training": training, "natural_accuracy": natural, "executor_gap": gap, "results": results, "coalitions": coalitions, "attention_prefixes": prefixes}
    output = ROOT / "tests/glm5_temp/phase1271_microcomponent_handoff_probe.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"natural": natural, "output": str(output)}))


if __name__ == "__main__":
    main()
