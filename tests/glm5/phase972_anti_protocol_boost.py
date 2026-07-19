#!/usr/bin/env python3
"""Phase 972: anti-protocol causal validation and semantic-path control.

All discovery uses a 20-prompt set; the selected interventions are evaluated on
the full 65-prompt set.  The script deliberately separates discovery from the
large evaluation to reduce the small-sample bias seen in phases 969--971.
"""
from __future__ import annotations

import gc, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import load_model, get_layers, get_model_info, release_model
from phase951_protocol_atlas import ensure_dir
from phase964_forward_diff import get_head_dims
from phase966_natural_stop import get_boundary_ids, log, register_multi_head_ablation
from phase970_completion_audit import (
    SAFE_HEADS, BoundaryDynamicProcessor, EN_PROMPTS_65, evaluate_clean_v5,
    generate_with_processor,
)
from phase971_protocol_field import get_mlp_down_proj

PHASE = 972
RESULT_DIR = Path("tests/glm5/result/phase972_anti_protocol_boost")


def output_scale_hook(scale):
    """Scale only the last-position MLP residual contribution."""
    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        y = output[0] if is_tuple else output
        z = y.clone()
        z[:, -1, :] *= scale
        return (z,) + output[1:] if is_tuple else z
    return hook


def channel_scale_hook(channels, scale):
    channels = list(map(int, channels))
    def hook(module, args):
        x = args[0].clone()
        x[:, -1, channels] *= scale
        return (x,) + tuple(args[1:])
    return hook


def logits_gap(logits, eos_id):
    v = logits[0, -1].float()
    return float(v.max() - v[eos_id]), int(v.argmax()), float(v[eos_id])


def measure(model, tokenizer, layers, eos_id, prompts, interventions):
    rows = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            base = model(ids, use_cache=False).logits
        bg, bt, be = logits_gap(base, eos_id)
        handles = interventions(layers)
        with torch.no_grad():
            patched = model(ids, use_cache=False).logits
        for h in handles: h.remove()
        pg, pt, pe = logits_gap(patched, eos_id)
        rows.append({"prompt": prompt, "base_gap": bg, "patched_gap": pg,
                     "delta_gap": pg-bg, "delta_eos": pe-be,
                     "top1_changed": pt != bt})
    return {"n": len(rows), "mean_base_gap": float(np.mean([r["base_gap"] for r in rows])),
            "mean_delta_gap": float(np.mean([r["delta_gap"] for r in rows])),
            "mean_delta_eos": float(np.mean([r["delta_eos"] for r in rows])),
            "top1_changed_rate": float(np.mean([r["top1_changed"] for r in rows])),
            "rows": rows}


def discover_channels(model, tokenizer, layers, eos_id, prompts, layer_idx, mode, topk=32):
    """Mean direct channel gap contribution using each prompt's actual top1.

    mode=anti selects negative contributors to boost; mode=protocol selects
    positive contributors to ablate.
    """
    layer = layers[layer_idx]
    Wd = get_mlp_down_proj(layer).detach().float().cpu()
    Wu = model.lm_head.weight.detach().float().cpu()
    scores = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        cap = {}
        def capture(module, args): cap["x"] = args[0].detach()
        h = layer.mlp.down_proj.register_forward_pre_hook(capture)
        with torch.no_grad(): out = model(ids, use_cache=False).logits
        h.remove()
        top = int(out[0, -1].argmax())
        direction = Wd.T @ (Wu[top] - Wu[eos_id])
        scores.append(cap["x"][0, -1].float().cpu() * direction)
    mean = torch.stack(scores).mean(0)
    order = torch.argsort(mean, descending=(mode == "protocol"))[:topk]
    return [{"channel": int(c), "mean_gap_contrib": float(mean[c])} for c in order]


def generation_eval(model, tokenizer, layers, eos_id, boundary_ids, d_head,
                    prompts, name, intervention, semantic_prompt=False):
    rows = []
    safe_heads = SAFE_HEADS.get("glm4", [])
    for prompt in prompts:
        effective = prompt + " Answer with only the shortest correct answer:" if semantic_prompt else prompt
        ids = tokenizer.encode(effective, return_tensors="pt").to(model.device)
        handles = register_multi_head_ablation(layers, safe_heads, d_head) + intervention(layers)
        with torch.no_grad(): logits = model(ids, use_cache=False).logits
        gap, _, _ = logits_gap(logits, eos_id)
        for h in handles: h.remove()
        b = max(1, int(gap + 2))
        handles = register_multi_head_ablation(layers, safe_heads, d_head) + intervention(layers)
        proc = BoundaryDynamicProcessor(eos_id, b, min_delay=2, boundary_ids=boundary_ids)
        gen, has_eos, n_tok = generate_with_processor(model, tokenizer, ids, proc, max_new=30, pad_id=eos_id)
        for h in handles: h.remove()
        ev = evaluate_clean_v5(prompt, gen, has_eos, n_tok)
        rows.append({"prompt": prompt, "effective_prompt": effective, "b": b, "gap": gap,
                     "generated": gen, "has_eos": has_eos, "n_tokens": n_tok, **ev})
    n = len(rows)
    return {"name": name, "n": n,
            "clean_rate": sum(r["strict_clean"] for r in rows)/n,
            "eos_rate": sum(r["has_eos"] for r in rows)/n,
            "expected_rate": sum(r["has_expected"] for r in rows)/n,
            "mean_b": float(np.mean([r["b"] for r in rows])), "rows": rows}


def run_glm4():
    ensure_dir(RESULT_DIR)
    model, tok, _ = load_model("glm4")
    layers = get_layers(model); info = get_model_info(model, "glm4")
    _, d_head = get_head_dims(model, info)
    eos_id = tok.eos_token_id; boundary_ids = get_boundary_ids(tok)
    discovery = EN_PROMPTS_65[:20]
    result = {"phase": PHASE, "model": "glm4", "discovery_n": 20, "evaluation_n": 65}

    # Whole-layer causal dose curve.
    dose = {}
    for scale in [0.0, 0.5, 1.25, 1.5, 2.0]:
        dose[str(scale)] = measure(model, tok, layers, eos_id, discovery,
            lambda ls, s=scale: [ls[39].mlp.register_forward_hook(output_scale_hook(s))])
        log(f"L39 scale={scale}: dg={dose[str(scale)]['mean_delta_gap']:+.3f}, "
            f"changed={dose[str(scale)]['top1_changed_rate']:.1%}")
    result["l39_dose_curve_20p"] = dose

    anti = discover_channels(model, tok, layers, eos_id, discovery, 39, "anti", 32)
    protocol = discover_channels(model, tok, layers, eos_id, discovery, 38, "protocol", 32)
    result["l39_anti_candidates"] = anti; result["l38_protocol_candidates"] = protocol

    group_tests = {}
    for k in [4, 8, 16, 32]:
        ac = [x["channel"] for x in anti[:k]]; pc = [x["channel"] for x in protocol[:k]]
        group_tests[f"l39_boost_{k}"] = measure(model, tok, layers, eos_id, discovery,
            lambda ls, c=ac: [ls[39].mlp.down_proj.register_forward_pre_hook(channel_scale_hook(c, 2.0))])
        group_tests[f"l38_ablate_{k}"] = measure(model, tok, layers, eos_id, discovery,
            lambda ls, c=pc: [ls[38].mlp.down_proj.register_forward_pre_hook(channel_scale_hook(c, 0.0))])
    result["channel_group_tests_20p"] = group_tests

    # Pick strongest safe-ish group by gap reduction penalized for top1 changes.
    candidates = [(n, r) for n, r in group_tests.items() if r["top1_changed_rate"] <= 0.15]
    best_name, best = min(candidates, key=lambda x: x[1]["mean_delta_gap"]) if candidates else ("none", None)
    if best_name.startswith("l39"):
        k = int(best_name.rsplit("_", 1)[1]); chans = [x["channel"] for x in anti[:k]]; L=39; scale=2.0
    elif best_name.startswith("l38"):
        k = int(best_name.rsplit("_", 1)[1]); chans = [x["channel"] for x in protocol[:k]]; L=38; scale=0.0
    else:
        chans=[]; L=39; scale=1.0
    intervention = lambda ls: ([] if not chans else [ls[L].mlp.down_proj.register_forward_pre_hook(channel_scale_hook(chans, scale))])
    result["selected"] = {"name": best_name, "layer": L, "scale": scale, "channels": chans,
                          "discovery_metrics": best}
    result["eval_baseline_65p"] = generation_eval(model, tok, layers, eos_id, boundary_ids, d_head,
                                                    EN_PROMPTS_65, "baseline", lambda ls: [])
    result["eval_selected_65p"] = generation_eval(model, tok, layers, eos_id, boundary_ids, d_head,
                                                    EN_PROMPTS_65, best_name, intervention)
    result["eval_answer_only_65p"] = generation_eval(model, tok, layers, eos_id, boundary_ids, d_head,
                                                    EN_PROMPTS_65, best_name+"+answer_only", intervention, True)
    out = RESULT_DIR / "glm4_result.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    release_model(model); gc.collect(); torch.cuda.empty_cache()
    log(f"Saved {out}")


def cross_model_scan(model_name):
    """Locate the most anti-gap MLP layer, then causally test its scaling.

    This is a mechanism-transfer test, not a clean-generation optimization.
    It uses 20 prompts for both discovery and causal validation.
    """
    ensure_dir(RESULT_DIR)
    model, tok, _ = load_model(model_name)
    layers = get_layers(model); eos_id = tok.eos_token_id
    prompts = EN_PROMPTS_65[:20]; Wu = model.lm_head.weight.detach().float().cpu()
    per_prompt = []
    for prompt in prompts:
        ids = tok.encode(prompt, return_tensors="pt").to(model.device)
        cap = {}; handles = []
        for L, layer in enumerate(layers):
            def hook(module, args, output, idx=L):
                y = output[0] if isinstance(output, tuple) else output
                cap[idx] = y.detach()
            handles.append(layer.mlp.register_forward_hook(hook))
        with torch.no_grad(): logits = model(ids, use_cache=False).logits
        for h in handles: h.remove()
        top = int(logits[0, -1].argmax()); direction = Wu[top] - Wu[eos_id]
        per_prompt.append([float(torch.dot(cap[L][0, -1].float().cpu(), direction))
                           for L in range(len(layers))])
    means = np.mean(per_prompt, axis=0).tolist()
    anti_layer = int(np.argmin(means))
    dose = {}
    for scale in [0.0, 0.5, 1.25, 1.5, 2.0]:
        dose[str(scale)] = measure(model, tok, layers, eos_id, prompts,
            lambda ls, s=scale, L=anti_layer: [ls[L].mlp.register_forward_hook(output_scale_hook(s))])
        log(f"{model_name} L{anti_layer} scale={scale}: dg={dose[str(scale)]['mean_delta_gap']:+.3f}, "
            f"changed={dose[str(scale)]['top1_changed_rate']:.1%}")
    result = {"phase": PHASE, "model": model_name, "n": 20,
              "mean_mlp_gap_contrib_by_layer": means, "selected_anti_layer": anti_layer,
              "selected_dla_gap_contrib": means[anti_layer], "dose_curve": dose}
    out = RESULT_DIR / f"{model_name}_cross_model.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    release_model(model); gc.collect(); torch.cuda.empty_cache(); log(f"Saved {out}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "glm4"
    run_glm4() if target == "glm4" else cross_model_scan(target)
