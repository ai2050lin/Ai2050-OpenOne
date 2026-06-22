#!/usr/bin/env python3
"""
Phase 570: Final Norm Gate and L20-L22 Writer Audit
最终归一化门与L20-L22写入者审计

Exp1: Norm decomposition (raw → rms → norm+weight) — which stage causes sign flip?
Exp2: Dimension contribution — which dims contribute most to the flip?
Exp3: Module writer at key layers — attn_out swap vs mlp_out swap vs h_in/h_out swap
Exp4: Norm-normalized semantic alignment — cos(h, d_TC) vs cos(Norm(h), d_TC)
Exp5: Cross-category validation (GLM4 only) — clothing_tool, furniture_tool

Run:
  python tests/glm5/phase570_norm_gate_writer.py qwen3 --smoke
  python tests/glm5/phase570_norm_gate_writer.py qwen3
  python tests/glm5/phase570_norm_gate_writer.py glm4
  python tests/glm5/phase570_norm_gate_writer.py deepseek7b
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_W_U, get_layers, get_model_info, release_model  # noqa: E402
from phase530_state_pair_decomposition import load_model_bf16_flash  # noqa: E402
from phase536_pair_quality_selectivity import CATEGORY_BANK  # noqa: E402
from phase539_interface_cluster_mechanism import PAIR_SPECS, layer_windows  # noqa: E402
import phase544_natural_decode_policy_gate_audit as p544  # noqa: E402
import phase545_sampling_stability_cross_category as p545  # noqa: E402
import phase548_paraphrase_candidate_robustness as p548  # noqa: E402
import phase558_prototype_object_binding_audit as p558  # noqa: E402
import phase559_prototype_generation_closure as p559  # noqa: E402
import phase568_locked_prefix_swap as p568  # noqa: E402
import phase569_pre_layer_source_tracing as p569  # noqa: E402


OUT_ROOT = Path("results/glm5_phase570_norm_gate_writer")
DEFAULT_ROUTES = ["forbidden_sentence_completion:temperature<-forbidden_definition"]

# Module writer trace layers
WRITER_LAYERS = {
    "qwen3": [8, 10, 12],
    "glm4": [18, 20, 22],
    "deepseek7b": [14, 16, 18],
}

# Best h_in swap layer from Phase 569 (for Norm decomposition)
BEST_H_IN_LAYER = {"qwen3": 8, "glm4": 22, "deepseek7b": 20}

# Cross-category pairs (GLM4 only)
CROSS_CATEGORY_PAIRS = ["clothing_tool", "furniture_tool"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


# ============================================================================
# Module cache collection: attn_out and mlp_out
# ============================================================================

def collect_module_caches(
    model: Any,
    layers: list[Any],
    donor_batch: dict[str, torch.Tensor],
    donor_pos: torch.Tensor,
    layer_ids: list[int],
) -> dict[str, dict[int, torch.Tensor]]:
    """Collect donor attn_out and mlp_out at answer position.

    attn_out: output of self_attn (the delta added to residual)
    mlp_out:  output of mlp (the delta added to residual)
    """
    caches: dict[str, dict[int, torch.Tensor]] = {"attn_out": {}, "mlp_out": {}}
    handles = []
    pos_cpu = donor_pos.cpu()

    for lid in layer_ids:
        layer = layers[lid]

        def make_attn_hook(layer_id: int, p_cpu: torch.Tensor):
            def hook(_module, _inp, output):
                hs = p559.tensor_from_output(output)
                p_dev = p_cpu.to(hs.device)
                bidx = torch.arange(hs.shape[0], device=hs.device)
                caches["attn_out"][layer_id] = hs[bidx, p_dev, :].detach()
            return hook

        def make_mlp_hook(layer_id: int, p_cpu: torch.Tensor):
            def hook(_module, _inp, output):
                hs = p559.tensor_from_output(output)
                p_dev = p_cpu.to(hs.device)
                bidx = torch.arange(hs.shape[0], device=hs.device)
                caches["mlp_out"][layer_id] = hs[bidx, p_dev, :].detach()
            return hook

        handles.append(layer.self_attn.register_forward_hook(make_attn_hook(lid, pos_cpu)))
        handles.append(layer.mlp.register_forward_hook(make_mlp_hook(lid, pos_cpu)))

    with torch.inference_mode():
        model(**donor_batch, return_dict=True, use_cache=False)
    for h in handles:
        h.remove()
    return caches


# ============================================================================
# Module swap hooks: attn_out swap, mlp_out swap
# ============================================================================

def install_module_swap_hooks(
    layers: list[Any],
    layer_ids: list[int],
    batch_size: int,
    answer_pos: int,
    donor_module_caches: dict[str, dict[int, torch.Tensor]] | None,
    swap_spec: str,  # "attn_out" or "mlp_out"
) -> list[Any]:
    handles = []
    if donor_module_caches is None or swap_spec not in ("attn_out", "mlp_out"):
        return handles

    pos_cpu = torch.full((batch_size,), answer_pos, dtype=torch.long)

    for lid in layer_ids:
        layer = layers[lid]
        site = swap_spec  # "attn_out" or "mlp_out"
        donor_vec = donor_module_caches[site].get(lid)
        if donor_vec is None:
            continue

        def make_hook(donor_vec: torch.Tensor, p_cpu: torch.Tensor):
            def hook(_module, _inp, output):
                hs = p559.tensor_from_output(output)
                new_hs = hs.clone()
                p_dev = p_cpu.to(hs.device)
                bidx = torch.arange(hs.shape[0], device=hs.device)
                new_hs[bidx, p_dev, :] = donor_vec.to(hs.device, dtype=hs.dtype)
                return p559.replace_output(output, new_hs)
            return hook

        target_module = layer.self_attn if swap_spec == "attn_out" else layer.mlp
        handles.append(target_module.register_forward_hook(make_hook(donor_vec, pos_cpu)))

    return handles


# ============================================================================
# Unified hook installation for any swap type
# ============================================================================

def install_any_swap_hooks(
    layers: list[Any],
    layer_ids: list[int],
    batch_size: int,
    answer_pos: int,
    swap_spec: str,
    donor_caches: dict[str, dict[int, torch.Tensor]] | None,
    donor_module_caches: dict[str, dict[int, torch.Tensor]] | None,
) -> list[Any]:
    if swap_spec == "baseline":
        return []
    if swap_spec in ("h_in", "h_attn", "h_out"):
        return p568.install_state_swap_hooks(
            layers, layer_ids, batch_size, answer_pos, donor_caches, swap_spec
        )
    if swap_spec in ("attn_out", "mlp_out"):
        return install_module_swap_hooks(
            layers, layer_ids, batch_size, answer_pos, donor_module_caches, swap_spec
        )
    return []


# ============================================================================
# Generation with any swap type + detailed logit recording
# ============================================================================

def generate_with_any_swap(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    swap_layer_ids: list[int],
    prompts: list[str],
    swap_spec: str,
    donor_caches: dict[str, dict[int, torch.Tensor]] | None,
    donor_module_caches: dict[str, dict[int, torch.Tensor]] | None,
    groups: dict[str, list[int]],
    mode: str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_length: int,
    logit_steps: int = 4,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    batch_size = len(prompts)

    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    answer_pos = input_ids.shape[1] - 1
    tokenizer.padding_side = old_padding

    handles = install_any_swap_hooks(
        layers, swap_layer_ids, batch_size, answer_pos,
        swap_spec, donor_caches, donor_module_caches
    )

    generated: list[list[int]] = [[] for _ in prompts]
    step_stats: list[list[dict[str, float]]] = [[] for _ in prompts]
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        logits0 = out.logits[:, answer_pos, :].float().cpu().numpy()
    for h in handles:
        h.remove()

    toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits0]
    for i, tok in enumerate(toks):
        generated[i].append(int(tok))
        if logit_steps > 0:
            step_stats[i].append(p569.detailed_logit_stats(logits0[i], groups))

    full_mask = attention_mask
    for step in range(1, max_new_tokens):
        new_ids = torch.tensor([[int(t)] for t in toks], dtype=torch.long, device=device)
        full_mask = torch.cat(
            [full_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)], dim=1
        )
        with torch.inference_mode():
            out = model(
                input_ids=new_ids, attention_mask=full_mask,
                past_key_values=past_kv, use_cache=True, return_dict=True,
            )
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :].float().cpu().numpy()
        toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits]
        for i, tok in enumerate(toks):
            generated[i].append(int(tok))
            if step < logit_steps:
                step_stats[i].append(p569.detailed_logit_stats(logits[i], groups))

    suffixes = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    del past_kv, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"generated_ids": generated, "generated_suffix": suffixes, "step_stats": step_stats}


# ============================================================================
# Exp1+2+4: Norm decomposition + dimension contribution + cosine alignment
# ============================================================================

def norm_decomposition_audit(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    swap_layer_ids: list[int],
    prompts: list[str],
    swap_spec: str,
    donor_caches: dict[str, dict[int, torch.Tensor]] | None,
    donor_module_caches: dict[str, dict[int, torch.Tensor]] | None,
    groups: dict[str, list[int]],
    W_U_np: np.ndarray,
    eps: float,
    max_length: int,
) -> dict[str, Any]:
    """Exp1+2+4: Decompose margin into raw → rms → norm+weight stages.

    Also computes per-dimension contribution and cosine alignment.
    Captures h_norm from forward pass (avoids meta tensor issues).
    """
    batch_size = len(prompts)
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    answer_pos = input_ids.shape[1] - 1
    tokenizer.padding_side = old_padding

    # Capture final layer output (h_raw) and final norm output (h_norm)
    final_layer = layers[-1]
    final_norm_module = model.model.norm
    captured: dict[str, torch.Tensor] = {}

    def make_final_hook():
        def hook(_module, _inp, output):
            hs = p559.tensor_from_output(output)
            bidx = torch.arange(hs.shape[0], device=hs.device)
            p_dev = torch.full((hs.shape[0],), answer_pos, dtype=torch.long, device=hs.device)
            captured["h_raw"] = hs[bidx, p_dev, :].detach().float().cpu()
        return hook

    def make_norm_hook():
        def hook(_module, _inp, output):
            hs = p559.tensor_from_output(output)
            bidx = torch.arange(hs.shape[0], device=hs.device)
            p_dev = torch.full((hs.shape[0],), answer_pos, dtype=torch.long, device=hs.device)
            captured["h_norm"] = hs[bidx, p_dev, :].detach().float().cpu()
        return hook

    swap_handles = install_any_swap_hooks(
        layers, swap_layer_ids, batch_size, answer_pos,
        swap_spec, donor_caches, donor_module_caches
    )
    final_handle = final_layer.register_forward_hook(make_final_hook())
    norm_handle = final_norm_module.register_forward_hook(make_norm_hook())

    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, return_dict=True)
    finally:
        for h in swap_handles:
            h.remove()
        final_handle.remove()
        norm_handle.remove()

    # h_raw and h_norm: [batch, d_model]
    h_raw = captured["h_raw"].numpy().astype(np.float64)
    h_norm_captured = captured["h_norm"].numpy().astype(np.float64)

    # Compute h_rms = h_raw / sqrt(mean(h_raw^2) + eps)
    variance = np.mean(h_raw ** 2, axis=-1, keepdims=True)
    h_rms = h_raw / np.sqrt(variance + eps)

    # Derive w_norm from h_norm / h_rms (element-wise, per batch then average)
    # h_norm = w_norm * h_rms, so w_norm = h_norm / h_rms
    # This should be the same for all batch elements
    w_norm_derived = (h_norm_captured / h_rms).mean(0)  # [d_model]

    # Logits at each stage
    z_raw = h_raw @ W_U_np.T.astype(np.float64)
    z_rms = h_rms @ W_U_np.T.astype(np.float64)
    z_norm = h_norm_captured @ W_U_np.T.astype(np.float64)

    # Margins at each stage
    def avg_margin(z):
        stats = [p569.detailed_logit_stats(z[i], groups) for i in range(batch_size)]
        return float(np.mean([s["target_minus_competitor"] for s in stats]))

    def avg_rank(z):
        stats = [p569.detailed_logit_stats(z[i], groups) for i in range(batch_size)]
        return float(np.mean([s["target_best_rank"] for s in stats]))

    def avg_mass(z, group_key):
        stats = [p569.detailed_logit_stats(z[i], groups) for i in range(batch_size)]
        return float(np.mean([s[f"{group_key}_group_mass"] for s in stats]))

    def avg_entropy(z):
        stats = [p569.detailed_logit_stats(z[i], groups) for i in range(batch_size)]
        return float(np.mean([s["entropy"] for s in stats]))

    result = {
        "raw_margin": avg_margin(z_raw),
        "rms_margin": avg_margin(z_rms),
        "norm_margin": avg_margin(z_norm),
        "raw_rank": avg_rank(z_raw),
        "rms_rank": avg_rank(z_rms),
        "norm_rank": avg_rank(z_norm),
        "raw_target_mass": avg_mass(z_raw, "target"),
        "rms_target_mass": avg_mass(z_rms, "target"),
        "norm_target_mass": avg_mass(z_norm, "target"),
        "raw_entropy": avg_entropy(z_raw),
        "rms_entropy": avg_entropy(z_rms),
        "norm_entropy": avg_entropy(z_norm),
        "n_examples": batch_size,
    }

    # Exp2: Dimension contribution
    tgt_ids = [i for i in groups["target"] if 0 <= i < W_U_np.shape[0]]
    cmp_ids = [i for i in groups["competitor"] if 0 <= i < W_U_np.shape[0]]
    d_TC = W_U_np[tgt_ids].mean(0).astype(np.float64) - W_U_np[cmp_ids].mean(0).astype(np.float64)

    # Per-dimension contribution (averaged over batch)
    raw_contrib = (h_raw * d_TC).mean(0)  # [d_model]
    rms_contrib = (h_rms * d_TC).mean(0)
    norm_contrib = (h_norm_captured * d_TC).mean(0)

    # Find dimensions with largest change from raw to norm
    delta_contrib = norm_contrib - raw_contrib
    top_dims = np.argsort(np.abs(delta_contrib))[::-1][:20]

    result["top_flip_dims"] = top_dims.tolist()
    result["raw_contrib_top20"] = raw_contrib[top_dims].tolist()
    result["norm_contrib_top20"] = norm_contrib[top_dims].tolist()
    result["delta_contrib_top20"] = delta_contrib[top_dims].tolist()

    # Total contribution sums (approximation of margin)
    result["raw_contrib_sum"] = float(raw_contrib.sum())
    result["rms_contrib_sum"] = float(rms_contrib.sum())
    result["norm_contrib_sum"] = float(norm_contrib.sum())

    # Exp4: Norm-normalized alignment
    def cos_sim(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    A_raw = np.array([cos_sim(h_raw[i], d_TC) for i in range(batch_size)])
    A_rms = np.array([cos_sim(h_rms[i], d_TC) for i in range(batch_size)])
    A_norm = np.array([cos_sim(h_norm_captured[i], d_TC) for i in range(batch_size)])

    result["A_TC_raw"] = float(np.mean(A_raw))
    result["A_TC_rms"] = float(np.mean(A_rms))
    result["A_TC_norm"] = float(np.mean(A_norm))

    # Norm of h at each stage
    result["h_raw_norm_mean"] = float(np.mean(np.linalg.norm(h_raw, axis=-1)))
    result["h_rms_norm_mean"] = float(np.mean(np.linalg.norm(h_rms, axis=-1)))
    result["h_norm_norm_mean"] = float(np.mean(np.linalg.norm(h_norm_captured, axis=-1)))

    return result


# ============================================================================
# Main run per model
# ============================================================================

def run_pair(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    info: Any,
    window: list[int],
    W_U_np: np.ndarray,
    eps: float,
    pair: str,
    routes: list[dict],
    test_n: int,
    seeds: list[int],
    writer_layers: list[int],
    best_h_in_L: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_length: int,
    logit_steps: int,
    do_norm_audit: bool = True,
    do_writer: bool = True,
    do_cross_category_subset: bool = False,
) -> dict[str, Any]:
    """Run all experiments for one pair."""
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pos_label, neg_label = PAIR_SPECS[pair]
    groups = p544.token_groups(tokenizer, pair)
    scaffolds = sorted(set([r["recipient_scaffold"] for r in routes] + [r["donor_scaffold"] for r in routes]))
    prompt_sets = p548.build_prompts(pair, test_n, scaffolds)

    all_cache_layers = sorted(set(writer_layers + window + [best_h_in_L]))
    pair_result: dict[str, Any] = {"pair": pair, "routes": {}}

    for route in routes:
        route_name = route["name"]
        prompt_rows = prompt_sets[route["recipient_scaffold"]]
        prompts = [r["prompt"] for r in prompt_rows]

        # Collect donor caches
        donor_scaffold = route["donor_scaffold"]
        donor_objs = CATEGORY_BANK[pos_label][-test_n:]
        repeat_idx = min(4, len(donor_objs) - 1)
        donor_repeat4_prompts = [
            p548.forbidden_prompt(donor_scaffold, donor_objs[repeat_idx], pos_label, neg_label)
        ] * test_n
        donor_enc = tokenizer(donor_repeat4_prompts, return_tensors="pt",
                              padding=True, truncation=True, max_length=max_length)
        donor_batch = {k: v.to(device) for k, v in donor_enc.items()}
        donor_pos = donor_batch["attention_mask"].sum(dim=1) - 1

        donor_caches = p568.collect_three_state_cache(
            model, layers, donor_batch, donor_pos, all_cache_layers
        )
        donor_module_caches = collect_module_caches(
            model, layers, donor_batch, donor_pos, writer_layers
        )
        log(f"  [{pair}/{route_name}] Collected donor caches (3-state + module)")

        route_result: dict[str, Any] = {"norm_audit": {}, "writer": {}, "compact": []}

        # === Exp1+2+4: Norm decomposition ===
        if do_norm_audit:
            norm_conds = [
                ("baseline", "baseline", window, None, None),
                ("repeat4_h_out", "h_out", window, donor_caches, None),
                ("best_h_in_swap", "h_in", [best_h_in_L], donor_caches, None),
            ]
            for cond_name, sw_spec, sw_layers, d_caches, d_mod in norm_conds:
                na = norm_decomposition_audit(
                    model, tokenizer, device, layers, sw_layers, prompts,
                    sw_spec, d_caches, d_mod, groups, W_U_np, eps, max_length,
                )
                route_result["norm_audit"][cond_name] = na
                log(f"  [{pair}/{route_name}] Norm audit {cond_name}: "
                    f"raw={na['raw_margin']:+.2f} rms={na['rms_margin']:+.2f} "
                    f"norm={na['norm_margin']:+.2f} | "
                    f"A_raw={na['A_TC_raw']:.4f} A_norm={na['A_TC_norm']:.4f} | "
                    f"|h|_raw={na['h_raw_norm_mean']:.1f} |h|_norm={na['h_norm_norm_mean']:.1f}")

        # === Exp3: Module writer generation ===
        if do_writer:
            all_records: dict[str, list[dict[str, Any]]] = {}

            for seed in seeds:
                # baseline
                res = generate_with_any_swap(
                    model, tokenizer, device, layers, window, prompts,
                    "baseline", None, None, groups, route["mode"], seed,
                    max_new_tokens, temperature, top_p, max_length, logit_steps,
                )
                all_records.setdefault("baseline_free", []).extend([
                    {"prompt_index": i, "object": r["object"], "seed": seed,
                     "condition": "baseline_free",
                     **{k: res[k][i] for k in ["generated_suffix", "generated_ids", "step_stats"]}}
                    for i, r in enumerate(prompt_rows)
                ])

                # repeat4 (h_out swap on window)
                res = generate_with_any_swap(
                    model, tokenizer, device, layers, window, prompts,
                    "h_out", donor_caches, None, groups, route["mode"], seed,
                    max_new_tokens, temperature, top_p, max_length, logit_steps,
                )
                all_records.setdefault("repeat4_free", []).extend([
                    {"prompt_index": i, "object": r["object"], "seed": seed,
                     "condition": "repeat4_free",
                     **{k: res[k][i] for k in ["generated_suffix", "generated_ids", "step_stats"]}}
                    for i, r in enumerate(prompt_rows)
                ])

                # Module writer: for each writer layer, test h_in, attn_out, mlp_out, h_out
                for L in writer_layers:
                    for sw_spec, label in [("h_in", "h_in"), ("attn_out", "attn_out"),
                                           ("mlp_out", "mlp_out"), ("h_out", "h_out")]:
                        cond_name = f"L{L}_{label}"
                        res = generate_with_any_swap(
                            model, tokenizer, device, layers, [L], prompts,
                            sw_spec, donor_caches, donor_module_caches,
                            groups, route["mode"], seed,
                            max_new_tokens, temperature, top_p, max_length, logit_steps,
                        )
                        all_records.setdefault(cond_name, []).extend([
                            {"prompt_index": i, "object": r["object"], "seed": seed,
                             "condition": cond_name,
                             **{k: res[k][i] for k in ["generated_suffix", "generated_ids", "step_stats"]}}
                            for i, r in enumerate(prompt_rows)
                        ])

                elapsed_hint = f"seed={seed} done"
                log(f"  [{pair}/{route_name}] Writer {elapsed_hint}")

            # Aggregate
            cond_order = ["baseline_free", "repeat4_free"]
            for L in writer_layers:
                cond_order += [f"L{L}_h_in", f"L{L}_attn_out", f"L{L}_mlp_out", f"L{L}_h_out"]

            for cond in cond_order:
                recs = all_records.get(cond, [])
                if not recs:
                    continue
                classified = [
                    {**r, **p548.classify_suffix(r["generated_suffix"], r["object"], pos_label, neg_label)}
                    for r in recs
                ]
                agg = p548.aggregate(classified)
                for step in range(logit_steps):
                    vals = [r["step_stats"][step] for r in recs if len(r.get("step_stats", [])) > step]
                    if vals:
                        for key in ["target_minus_competitor", "target_best_rank",
                                    "target_group_mass", "entropy"]:
                            agg[f"step{step}_{key}"] = float(np.mean([v[key] for v in vals]))

                route_result["writer"][cond] = agg
                route_result["compact"].append({
                    "condition": cond,
                    "free_clean": agg["clean_non_object_rate"],
                    "step0_margin": agg.get("step0_target_minus_competitor", 0),
                    "step0_rank": agg.get("step0_target_best_rank", 0),
                    "step0_entropy": agg.get("step0_entropy", 0),
                })
                log(f"  [{pair}/{route_name}] {cond}: clean={agg['clean_non_object_rate']:.2f}, "
                    f"s0_margin={agg.get('step0_target_minus_competitor', 0):.2f}, "
                    f"s0_rank={agg.get('step0_target_best_rank', 0):.0f}")

        pair_result["routes"][route_name] = route_result

    return pair_result


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    routes = p558.parse_routes(args.routes)
    seeds = parse_int_csv(args.sample_seeds)

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        _, window = next(iter(windows.items()))
        W_U = get_W_U(model, args.model).astype(np.float32)

        # Get final norm eps (weight is captured from forward pass to avoid meta tensor issues)
        final_norm = model.model.norm
        eps = getattr(final_norm, 'variance_epsilon', None)
        if eps is None:
            eps = getattr(final_norm, 'eps', None)
        if eps is None:
            eps = getattr(model.config, 'rms_norm_eps', 1e-6)

        writer_layers = [L for L in WRITER_LAYERS[args.model] if 0 <= L < info.n_layers]
        best_h_in_L = BEST_H_IN_LAYER[args.model]

        log(f"{args.model}: phase570 window={window}, writer_layers={writer_layers}, "
            f"eps={eps}, best_h_in_L={best_h_in_L}")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / f"phase570_{args.model}_checkpoint.json"

        result: dict[str, Any] = {
            "phase": 570, "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "window": window, "writer_layers": writer_layers,
            "best_h_in_layer": best_h_in_L, "eps": eps,
            "routes": routes, "sample_seeds": seeds,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "pairs": {},
        }

        # === Main pair ===
        log(f"=== Main pair: {args.pair} ===")
        main_result = run_pair(
            model, tokenizer, device, layers, info, window,
            W_U, eps, args.pair, routes, args.test_n, seeds,
            writer_layers, best_h_in_L, args.max_new_tokens, args.temperature, args.top_p,
            args.max_length, args.logit_steps,
            do_norm_audit=True, do_writer=True,
        )
        result["pairs"][args.pair] = main_result

        # Checkpoint
        checkpoint_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")

        # === Cross-category (GLM4 only) ===
        if args.model == "glm4" and not args.skip_cross_category:
            cross_seeds = parse_int_csv(args.cross_seeds)
            for pair in CROSS_CATEGORY_PAIRS:
                log(f"=== Cross-category pair: {pair} ===")
                cross_result = run_pair(
                    model, tokenizer, device, layers, info, window,
                    W_U, eps, pair, routes, args.test_n, cross_seeds,
                    writer_layers, best_h_in_L, args.max_new_tokens, args.temperature, args.top_p,
                    args.max_length, args.logit_steps,
                    do_norm_audit=True, do_writer=True,
                )
                result["pairs"][pair] = cross_result

                # Checkpoint after each pair
                checkpoint_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                           encoding="utf-8")

        return result
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--windows", default=None)
    parser.add_argument("--pair", default="vehicle_tool")
    parser.add_argument("--test-n", type=int, default=24)
    parser.add_argument("--sample-seeds", default="101,103,107,109,113,127")
    parser.add_argument("--cross-seeds", default="101,103,107,109")
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--logit-steps", type=int, default=4)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--skip-cross-category", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.test_n = 4
        args.sample_seeds = "101,103"
        args.cross_seeds = "101"
        args.max_new_tokens = 8
        args.logit_steps = 3
        log("SMOKE TEST MODE: test_n=4, seeds=2, max_tokens=8")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out_path = out_dir / f"phase570_{args.model}_norm_gate_writer{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
