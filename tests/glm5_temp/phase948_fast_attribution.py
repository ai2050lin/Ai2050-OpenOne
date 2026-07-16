#!/usr/bin/env python3
"""
Phase 948 快速归因: 从已有协议发现结果出发, 仅做最少的归因分析
"""
import json, sys, gc, time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
from model_utils import load_model, get_layers, get_model_info, release_model, get_sample_layers

RESULT_DIR = Path("results/phase948_protocol_field_audit")
MAX_ATTRIBUTION_TIME = 240  # 4 min max for attribution


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_decode(tokenizer, tid):
    try:
        r = tokenizer.decode([tid], skip_special_tokens=False)
        return r if r else f"<tok_{tid}>"
    except:
        return f"<tok_{tid}>"


def zero_last(output):
    if isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
            return output
        p = output[0].clone()
        if p.ndim >= 3:
            p[:, -1, :] = 0
        return (p, *output[1:])
    if torch.is_tensor(output):
        p = output.clone()
        if p.ndim >= 3:
            p[:, -1, :] = 0
        return p
    return output


def run_attribution_fast(model, tokenizer, device, prompts, layer_indices,
                          protocol_ids, max_prompts=8):
    """Fast attribution: limited prompts and layers."""
    sel_prompts = prompts[:max_prompts]
    layers_list = get_layers(model)
    attributions = defaultdict(lambda: defaultdict(list))
    t_start = time.time()

    for pi, prompt in enumerate(sel_prompts):
        if time.time() - t_start > MAX_ATTRIBUTION_TIME:
            log(f"  stopping attribution at {pi}/{max_prompts} prompts (timeout)")
            break

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            base_out = model(input_ids=input_ids, use_cache=False)
            base_logits = base_out.logits[0, -1].detach().float().cpu()
        
        for li in layer_indices:
            if time.time() - t_start > MAX_ATTRIBUTION_TIME:
                break
            layer = layers_list[li]
            
            for comp in ["attention", "mlp"]:
                module = getattr(layer, "self_attn" if comp == "attention" else "mlp", None)
                if module is None:
                    continue
                
                handle = module.register_forward_hook(
                    lambda _m, _in, out: zero_last(out)
                )
                with torch.no_grad():
                    try:
                        patched = model(input_ids=input_ids, use_cache=False)
                        patched_logits = patched.logits[0, -1].detach().float().cpu()
                    except:
                        patched_logits = base_logits.clone()
                handle.remove()
                
                key = f"L{li}_{comp}"
                for tid in protocol_ids:
                    delta = float(patched_logits[tid].item() - base_logits[tid].item())
                    attributions[str(tid)][key].append(delta)
        
        if (pi + 1) % 3 == 0:
            elapsed = time.time() - t_start
            log(f"  attr {pi+1}/{max_prompts} ({elapsed:.0f}s)")
    
    # Aggregate
    agg = {}
    for tid_str, deltas in attributions.items():
        by_key = {}
        for key, vals in deltas.items():
            by_key[key] = {
                "mean_delta": float(np.mean(vals)),
                "std_delta": float(np.std(vals)),
                "neg_count": sum(1 for v in vals if v < 0),
                "total": len(vals),
                "neg_ratio": sum(1 for v in vals if v < 0) / max(len(vals), 1),
            }
        agg[tid_str] = by_key
    
    return {"n_prompts": pi + 1, "n_layers": len(layer_indices), "attributions": agg}


def main():
    for model_name in ["glm4", "deepseek7b"]:
        log(f"{'='*50}")
        log(f"Phase 948 Fast: {model_name}")
        
        # Load protocol discovery
        disc_path = RESULT_DIR / model_name / "protocol_discovery.json"
        if not disc_path.exists():
            log(f"  No protocol discovery for {model_name}, creating first...")
            # Need to run full distribution first
            log("  Running distribution capture...")
            cmd = f'python tests/glm5/phase948_protocol_field_audit.py --model {model_name} --min_cross_prompt 5 --attr_prompts 0 --top_protocol_tokens 5'
            import subprocess
            subprocess.run(cmd, shell=True, cwd=str(ROOT))
        
        disc = json.loads(disc_path.read_text(encoding="utf-8"))
        candidates = disc["protocol_candidates"]
        top_ids = [c["token_id"] for c in candidates[:30]]
        log(f"  Loaded {len(candidates)} candidates, tracing top {len(top_ids)}")
        
        # Load model
        model, tokenizer, device = load_model(model_name)
        info = get_model_info(model, model_name)
        log(f"  Model: {info.model_class}, {info.n_layers} layers")
        
        # Get prompts from distribution
        dist_path = RESULT_DIR / model_name / "logit_distribution.json"
        dist = json.loads(dist_path.read_text(encoding="utf-8"))
        all_prompts = [p["prompt"] for p in dist["prompts"]]
        
        # Layer indices
        layer_indices = get_sample_layers(info.n_layers, n_samples=max(1, info.n_layers // 5))
        log(f"  Attribution layers: {layer_indices} ({len(layer_indices)})")
        
        # Run fast attribution
        attr = run_attribution_fast(model, tokenizer, device,
                                     all_prompts, layer_indices, top_ids,
                                     max_prompts=6)
        
        attr_path = RESULT_DIR / model_name / "component_attribution.json"
        attr_path.write_text(json.dumps(attr, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  Saved attribution: {attr_path}")
        
        # Summary stats
        agg = attr["attributions"]
        attn_vals, mlp_vals = [], []
        for tid_str, comp_data in agg.items():
            for k, v in comp_data.items():
                if "mean_delta" in v:
                    if "attention" in k:
                        attn_vals.append(abs(v["mean_delta"]))
                    elif "mlp" in k:
                        mlp_vals.append(abs(v["mean_delta"]))
        
        log(f"  Mean |attn_delta| = {np.mean(attn_vals):.4f}")
        log(f"  Mean |mlp_delta|  = {np.mean(mlp_vals):.4f}")
        log(f"  Attn/MLP ratio   = {np.mean(attn_vals)/max(np.mean(mlp_vals),1e-10):.3f}")
        
        # Simple clustering
        tok_ids = sorted(agg.keys())
        if len(tok_ids) >= 2:
            all_keys = sorted(next(iter(agg.values())).keys())
            X = np.array([[agg[tid][k]["mean_delta"] for k in all_keys] for tid in tok_ids])
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            Xn = X / norms
            cos = Xn @ Xn.T
            
            clusters = []
            used = set()
            for i in range(len(tok_ids)):
                if i in used:
                    continue
                mem = [tok_ids[i]]
                used.add(i)
                for j in range(i+1, len(tok_ids)):
                    if j not in used and cos[i,j] > 0.7:
                        mem.append(tok_ids[j])
                        used.add(j)
                if len(mem) >= 2:
                    clusters.append({"members": mem, "size": len(mem)})
            
            cluster_path = RESULT_DIR / model_name / "token_clusters.json"
            cluster_path.write_text(json.dumps({"n_clusters": len(clusters), "clusters": sorted(clusters, key=lambda c: -c["size"])}, ensure_ascii=False, indent=2), encoding="utf-8")
            log(f"  Clusters found: {len(clusters)}")
        
        # Summary
        summary = {
            "model": model_name,
            "protocol_candidates": len(candidates),
            "top_tokens": [{"token": safe_decode(tokenizer, c["token_id"]),
                           "cross_ratio": c["cross_prompt_ratio"],
                           "mean_rank": c["mean_rank"]}
                          for c in candidates[:15]],
            "attribution": {
                "mean_attn_delta": float(np.mean(attn_vals)),
                "mean_mlp_delta": float(np.mean(mlp_vals)),
                "attn_mlp_ratio": float(np.mean(attn_vals)/max(np.mean(mlp_vals),1e-10)),
            },
            "clusters": len(clusters),
        }
        summary_path = RESULT_DIR / model_name / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  Summary saved")
        
        release_model(model)
        log(f"  Done {model_name}")


if __name__ == "__main__":
    main()
