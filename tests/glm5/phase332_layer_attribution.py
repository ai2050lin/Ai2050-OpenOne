"""
Phase 332: Layer Attribution — Decomposing Binding by Layer Contribution
========================================================================

Output replacement patching (Phase 332 initial) showed frac_destroyed ≈ 1.0
at ALL layers — because replacing any layer's output lets the model
recompute from source, destroying ALL binding regardless.

This script uses a more surgical approach: LAYER ATTRIBUTION (logit attribution).

In a transformer with residual connections:
  h[N] = h[0] + Σ_L (attn_out_L + mlp_out_L)

The binding at the final layer:
  binding = (W_U[target] - W_U[competitor]) @ h[N]
          = gap_embedding + Σ_L Δ_gap_L

Where Δ_gap_L = (W_U[target] - W_U[competitor]) @ (h[L+1] - h[L])
is the contribution of layer L to the gap.

For binding_item:
  binding_item = gap_obj - gap_item
  Δ_binding_item_L = Δ_gap_obj_L - Δ_gap_item_L

This directly measures: "How much does layer L contribute to binding_item,
above and beyond what it contributes to the item baseline?"

Expected:
- Early layers: small contributions
- Late layers (binding growth region): large positive for HC, negative for NI
- Cumulative sum = total binding_item

This is MUCH more informative than output replacement!

Usage:
  python tests/glm5/phase332_layer_attribution.py qwen3
  python tests/glm5/phase332_layer_attribution.py glm4
  python tests/glm5/phase332_layer_attribution.py deepseek7b
"""
import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


MODEL_CONFIGS = {
    "qwen3": {
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "n_layers": 36, "d_model": 2560,
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
    },
}

# Expanded test pairs
TEST_PAIRS = [
    # HC (12)
    ("apple", "red", "blue", "color", "high_compatible"),
    ("banana", "yellow", "purple", "color", "high_compatible"),
    ("snow", "white", "black", "color", "high_compatible"),
    ("sky", "blue", "green", "color", "high_compatible"),
    ("cherry", "red", "blue", "color", "high_compatible"),
    ("leaf", "green", "red", "color", "high_compatible"),
    ("stone", "rough", "soft", "texture", "high_compatible"),
    ("silk", "smooth", "rough", "texture", "high_compatible"),
    ("ice", "cold", "hot", "temperature", "high_compatible"),
    ("fire", "hot", "cold", "temperature", "high_compatible"),
    ("oven", "hot", "cold", "temperature", "high_compatible"),
    ("fridge", "cold", "hot", "temperature", "high_compatible"),
    # NI (6)
    ("apple", "blue", "black", "color", "near_incompatible"),
    ("snow", "pink", "orange", "color", "near_incompatible"),
    ("banana", "white", "black", "color", "near_incompatible"),
    ("grass", "yellow", "purple", "color", "near_incompatible"),
    ("sky", "red", "brown", "color", "near_incompatible"),
    ("fire", "blue", "green", "color", "near_incompatible"),
    # CT (4)
    ("apple", "sharp", "soft", "texture", "cross_type"),
    ("snow", "sweet", "bitter", "taste", "cross_type"),
    ("stone", "sweet", "sour", "taste", "cross_type"),
    ("fire", "quiet", "loud", "sound", "cross_type"),
    # AA (5)
    ("idea", "red", "blue", "color", "abstract_absurd"),
    ("concept", "green", "yellow", "color", "abstract_absurd"),
    ("justice", "blue", "red", "color", "abstract_absurd"),
    ("democracy", "hot", "cold", "temperature", "abstract_absurd"),
    ("freedom", "rough", "smooth", "texture", "abstract_absurd"),
]

BASELINE_PROMPTS = ["The item"]  # For binding_item


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=impl,
            )
            log(f"  Loaded {model_name} with attn_impl={impl}")
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Model: {type(model).__name__}, device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def get_W_U(model, model_name):
    if hasattr(model, "lm_head"):
        w = model.lm_head.weight
        if not w.is_meta:
            return w.detach().cpu().float().numpy()
    import glob
    from safetensors import safe_open
    model_path = MODEL_CONFIGS[model_name]["path"]
    sf_files = glob.glob(os.path.join(model_path, '*.safetensors'))
    for sf_file in sf_files:
        with safe_open(sf_file, framework='pt', device='cpu') as sf:
            if 'lm_head.weight' in sf.keys():
                w = sf.get_tensor('lm_head.weight')
                log(f"  Loaded lm_head from {os.path.basename(sf_file)}")
                return w.float().numpy()
    raise ValueError(f"Cannot load lm_head for {model_name}")


def get_token_id(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        return None
    if len(ids) > 1:
        log(f"    WARN: '{word}' → {len(ids)} tokens, using first")
    return ids[0]


def get_hidden_states_all(model, tokenizer, device, prompt, n_layers):
    """Get hidden states at ALL layers for last token position"""
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    # List of numpy arrays: (d_model,) — last token at each layer
    hs_list = [hs[0, -1].float().cpu().numpy() for hs in out.hidden_states]
    return hs_list


def run_all(model_name):
    log(f"Phase 332: Layer Attribution — {model_name}")
    log("=" * 60)
    
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    
    W_U = get_W_U(model, model_name)
    log(f"  W_U shape: {W_U.shape}")
    
    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # ============================================================
    # STEP 0: Compute baseline hidden states
    # ============================================================
    log(f"\n=== STEP 0: Computing baseline hidden states ===")
    
    baseline_hs = {}
    for bp in BASELINE_PROMPTS:
        baseline_hs[bp] = get_hidden_states_all(model, tokenizer, device, bp, n_layers)
        log(f"  Baseline '{bp}': {len(baseline_hs[bp])} layers")
    
    # ============================================================
    # STEP 1: Per-pair layer attribution
    # ============================================================
    log(f"\n=== STEP 1: Per-pair layer attribution ===")
    
    level_order = ["high_compatible", "near_incompatible", "cross_type", "abstract_absurd"]
    all_pair_data = []
    
    for idx, (obj, target_val, competitor_val, attr_type, compat_level) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target_val)
        tid_c = get_token_id(tokenizer, competitor_val)
        if tid_t is None or tid_c is None:
            continue
        
        obj_prompt = f"The {obj}"
        
        # Direction vector for this pair
        binding_dir = W_U[tid_t] - W_U[tid_c]  # (d_model,)
        
        # Object hidden states
        hs_obj = get_hidden_states_all(model, tokenizer, device, obj_prompt, n_layers)
        
        # Item hidden states (for binding_item)
        hs_item = baseline_hs["The item"]
        
        # Compute per-layer contributions
        layer_attributions = []
        cumul_binding_obj = 0.0
        cumul_binding_item = 0.0
        
        for l in range(n_layers):
            # Layer contribution = h[L+1] - h[L]
            diff_obj = hs_obj[l + 1] - hs_obj[l]
            diff_item = hs_item[l + 1] - hs_item[l]
            
            # Gap contribution of this layer
            delta_gap_obj = float(binding_dir @ diff_obj)
            delta_gap_item = float(binding_dir @ diff_item)
            
            # Binding_item contribution of this layer
            delta_binding_item = delta_gap_obj - delta_gap_item
            
            # Cumulative
            cumul_binding_obj += delta_gap_obj
            cumul_binding_item += delta_binding_item
            
            layer_attributions.append({
                "layer": l,
                "rel_depth": round(l / n_layers, 3),
                "delta_gap_obj": round(delta_gap_obj, 4),
                "delta_gap_item": round(delta_gap_item, 4),
                "delta_binding_item": round(delta_binding_item, 4),
                "cumul_binding_item": round(cumul_binding_item, 4),
            })
        
        # Final binding values (for verification)
        final_gap_obj = float(binding_dir @ hs_obj[-1])
        final_gap_item = float(binding_dir @ hs_item[-1])
        final_binding_item = final_gap_obj - final_gap_item
        
        # Embedding contribution
        embed_gap_obj = float(binding_dir @ hs_obj[0])
        embed_gap_item = float(binding_dir @ hs_item[0])
        embed_binding_item = embed_gap_obj - embed_gap_item
        
        result = {
            "obj": obj,
            "target_val": target_val,
            "competitor_val": competitor_val,
            "compat_level": compat_level,
            "final_gap_obj": round(final_gap_obj, 4),
            "final_gap_item": round(final_gap_item, 4),
            "final_binding_item": round(final_binding_item, 4),
            "embed_gap_obj": round(embed_gap_obj, 4),
            "embed_binding_item": round(embed_binding_item, 4),
            "cumul_from_layers": round(cumul_binding_item, 4),
            "layer_attributions": layer_attributions,
        }
        all_pair_data.append(result)
        
        # Verification: cumulative should match final
        embed_plus_cumul = embed_binding_item + cumul_binding_item
        mismatch = abs(embed_plus_cumul - final_binding_item)
        
        if (idx + 1) % 5 == 0 or idx < 3:
            log(f"  [{idx+1}/{len(TEST_PAIRS)}] {obj}-{target_val} ({compat_level}): "
                f"final_binding={final_binding_item:+.3f}, "
                f"embed={embed_binding_item:+.3f}, layers={cumul_binding_item:+.3f}, "
                f"mismatch={mismatch:.4f}")
        
        if (idx + 1) % 8 == 0 and torch.cuda.is_available():
            log(f"    GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB, elapsed={time.time()-t0:.0f}s")
    
    # ============================================================
    # STEP 2: Aggregate by compat_level
    # ============================================================
    log(f"\n=== STEP 2: Aggregation by compat_level ===")
    
    level_trajectories = {}
    for cl in level_order:
        pair_results = [r for r in all_pair_data if r["compat_level"] == cl]
        if not pair_results:
            continue
        
        # Average per-layer contributions
        layer_aggs = []
        for l in range(n_layers):
            deltas = [r["layer_attributions"][l]["delta_binding_item"] for r in pair_results]
            cumuls = [r["layer_attributions"][l]["cumul_binding_item"] for r in pair_results]
            deltas_obj = [r["layer_attributions"][l]["delta_gap_obj"] for r in pair_results]
            
            layer_aggs.append({
                "layer": l,
                "rel_depth": round(l / n_layers, 3),
                "avg_delta_binding_item": round(float(np.mean(deltas)), 4),
                "std_delta_binding_item": round(float(np.std(deltas)), 4),
                "avg_cumul_binding_item": round(float(np.mean(cumuls)), 4),
                "avg_delta_gap_obj": round(float(np.mean(deltas_obj)), 4),
            })
        
        # Summary stats
        avg_final = float(np.mean([r["final_binding_item"] for r in pair_results]))
        avg_embed = float(np.mean([r["embed_binding_item"] for r in pair_results]))
        
        level_trajectories[cl] = {
            "n_pairs": len(pair_results),
            "avg_final_binding_item": round(avg_final, 4),
            "avg_embed_binding_item": round(avg_embed, 4),
            "layer_aggs": layer_aggs,
        }
        
        log(f"  {cl}: n={len(pair_results)}, final={avg_final:+.3f}, embed={avg_embed:+.3f}")
    
    # ============================================================
    # STEP 3: Layer attribution table (main output)
    # ============================================================
    log(f"\n{'='*100}")
    log(f"LAYER ATTRIBUTION TABLE — {model_name} ({n_layers} layers)")
    log(f"Δ_binding_item_L = contribution of layer L to binding_item")
    log(f"Cumulative = running sum from L0 (excludes embedding)")
    log(f"{'='*100}")
    
    # Sample layers for display
    sample_layers = list(range(0, n_layers, max(1, n_layers // 20)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))
    
    # Embedding contribution
    log(f"\n  Embedding contribution (before any transformer layer):")
    for cl in level_order:
        if cl in level_trajectories:
            log(f"    {cl}: {level_trajectories[cl]['avg_embed_binding_item']:+.4f}")
    
    # Per-layer table
    header = f"  {'Layer':>5} {'RelD':>6}"
    for cl in level_order:
        short = {"high_compatible": "HC", "near_incompatible": "NI",
                 "cross_type": "CT", "abstract_absurd": "AA"}
        header += f"  {'Δ_'+short[cl]:>8} {'Cum_'+short[cl]:>9}"
    log(header)
    log("  " + "-" * (12 + 18 * len(level_order)))
    
    for l in sample_layers:
        rel = f"{l/n_layers:.2f}"
        row = f"  L{l:>4} {rel:>6}"
        for cl in level_order:
            if cl in level_trajectories:
                la = level_trajectories[cl]["layer_aggs"][l]
                row += f"  {la['avg_delta_binding_item']:>+8.3f} {la['avg_cumul_binding_item']:>+9.3f}"
            else:
                row += f"  {'N/A':>8} {'N/A':>9}"
        log(row)
    
    # Final layer
    log(f"\n  Final layer values:")
    for cl in level_order:
        if cl in level_trajectories:
            la = level_trajectories[cl]["layer_aggs"][-1]
            log(f"    {cl}: cumul_from_layers={la['avg_cumul_binding_item']:+.3f}, "
                f"final_binding={level_trajectories[cl]['avg_final_binding_item']:+.3f}")
    
    # ============================================================
    # STEP 4: Identify key binding formation layers
    # ============================================================
    log(f"\n=== STEP 4: Key binding formation layers ===")
    
    for cl in level_order:
        if cl not in level_trajectories:
            continue
        la = level_trajectories[cl]["layer_aggs"]
        
        # Top 5 layers by absolute contribution
        sorted_by_contrib = sorted(la, key=lambda x: abs(x["avg_delta_binding_item"]), reverse=True)
        log(f"\n  {cl} — Top 5 layers by |Δ_binding_item|:")
        for e in sorted_by_contrib[:5]:
            log(f"    L{e['layer']} (rel={e['rel_depth']:.2f}): "
                f"Δ={e['avg_delta_binding_item']:+.4f} ±{e['std_delta_binding_item']:.3f}")
        
        # First layer where cumulative exceeds 50% of final
        final_val = level_trajectories[cl]["avg_final_binding_item"]
        embed_val = level_trajectories[cl]["avg_embed_binding_item"]
        target_cumul = 0.5 * (final_val - embed_val)  # 50% of layer contribution
        
        for e in la:
            if (final_val - embed_val) > 0 and e["avg_cumul_binding_item"] >= target_cumul:
                log(f"  {cl}: 50% of layer binding at L{e['layer']} (rel={e['rel_depth']:.2f})")
                break
        
        # First layer where cumulative turns positive (for HC)
        if cl == "high_compatible":
            for e in la:
                if e["avg_cumul_binding_item"] > 0.1:
                    log(f"  {cl}: Cumulative > 0.1 at L{e['layer']} (rel={e['rel_depth']:.2f})")
                    break
    
    # ============================================================
    # STEP 5: HC vs AA cumulative comparison
    # ============================================================
    log(f"\n=== STEP 5: HC vs AA cumulative binding_item ===")
    
    hc_la = level_trajectories.get("high_compatible", {}).get("layer_aggs", [])
    aa_la = level_trajectories.get("abstract_absurd", {}).get("layer_aggs", [])
    
    if hc_la and aa_la:
        header = f"  {'Layer':>5} {'RelD':>6} {'HC_cumul':>10} {'AA_cumul':>10} {'HC>AA':>6} {'HC-AA_gap':>10}"
        log(header)
        log("  " + "-" * 52)
        
        first_hc_gt_aa = None
        for l in sample_layers:
            if l < len(hc_la) and l < len(aa_la):
                hc_c = hc_la[l]["avg_cumul_binding_item"]
                aa_c = aa_la[l]["avg_cumul_binding_item"]
                hc_gt_aa = hc_c > aa_c
                gap = hc_c - aa_c
                
                # Include embedding
                hc_embed = level_trajectories["high_compatible"]["avg_embed_binding_item"]
                aa_embed = level_trajectories["abstract_absurd"]["avg_embed_binding_item"]
                hc_total = hc_embed + hc_c
                aa_total = aa_embed + aa_c
                
                if first_hc_gt_aa is None and hc_total > aa_total and hc_total > 0.1:
                    first_hc_gt_aa = l
                
                rel = f"{l/n_layers:.2f}"
                log(f"  L{l:>4} {rel:>6} {hc_total:>+10.3f} {aa_total:>+10.3f} "
                    f"{str(hc_gt_aa):>6} {gap:>+10.3f}")
        
        if first_hc_gt_aa is not None:
            log(f"\n  First HC>AA (with embed, thr 0.1): L{first_hc_gt_aa} (rel={first_hc_gt_aa/n_layers:.2f})")
    
    # ============================================================
    # STEP 6: NI suppression layer attribution
    # ============================================================
    log(f"\n=== STEP 6: NI suppression — which layers make NI negative? ===")
    
    ni_la = level_trajectories.get("near_incompatible", {}).get("layer_aggs", [])
    if ni_la:
        ni_embed = level_trajectories["near_incompatible"]["avg_embed_binding_item"]
        
        header = f"  {'Layer':>5} {'RelD':>6} {'Δ_NI':>8} {'Cum_NI':>8} {'NI_total':>10}"
        log(header)
        log("  " + "-" * 36)
        
        for l in sample_layers:
            if l < len(ni_la):
                e = ni_la[l]
                ni_total = ni_embed + e["avg_cumul_binding_item"]
                rel = f"{l/n_layers:.2f}"
                log(f"  L{l:>4} {rel:>6} {e['avg_delta_binding_item']:>+8.3f} "
                    f"{e['avg_cumul_binding_item']:>+8.3f} {ni_total:>+10.3f}")
    
    # ============================================================
    # STEP 7: Key per-pair attribution curves
    # ============================================================
    log(f"\n=== STEP 7: Key per-pair attribution curves ===")
    
    key_pairs = ["apple_red", "banana_yellow", "snow_white", "ice_cold",
                 "idea_red", "justice_blue"]
    
    for r in all_pair_data:
        key = f"{r['obj']}_{r['target_val']}"
        if key not in key_pairs:
            continue
        
        log(f"\n  {key} ({r['compat_level']}): final_binding={r['final_binding_item']:+.3f}")
        header = f"    {'Layer':>5} {'RelD':>6} {'Δ_bind':>8} {'Cumul':>8} {'Δ_gap_obj':>10}"
        log(header)
        log("    " + "-" * 38)
        
        for l in sample_layers:
            if l < len(r["layer_attributions"]):
                la = r["layer_attributions"][l]
                rel = f"{l/n_layers:.2f}"
                log(f"    L{l:>4} {rel:>6} {la['delta_binding_item']:>+8.3f} "
                    f"{la['cumul_binding_item']:>+8.3f} {la['delta_gap_obj']:>+10.3f}")
    
    # Release model
    del model, W_U
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save results
    save_data = {
        "model": model_name,
        "n_layers": n_layers,
        "approach": "layer_attribution_binding_item",
        "level_trajectories": level_trajectories,
        "details": all_pair_data,
    }
    
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj
    
    save_data = convert(save_data)
    
    os.makedirs("results/phase332_layer_attribution", exist_ok=True)
    out_path = f"results/phase332_layer_attribution/{model_name}_phase332.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")
    
    # Final summary
    log(f"\n{'='*60}")
    log(f"SUMMARY — {model_name}")
    log(f"{'='*60}")
    
    for cl in level_order:
        if cl in level_trajectories:
            lt = level_trajectories[cl]
            log(f"  {cl}: final_binding={lt['avg_final_binding_item']:+.3f}, "
                f"embed={lt['avg_embed_binding_item']:+.3f}, "
                f"layers={lt['avg_final_binding_item']-lt['avg_embed_binding_item']:+.3f}")
            
            # Top 3 contributing layers
            la = lt["layer_aggs"]
            top3 = sorted(la, key=lambda x: x["avg_delta_binding_item"], reverse=True)[:3]
            for e in top3:
                log(f"    L{e['layer']} (rel={e['rel_depth']:.2f}): "
                    f"Δ={e['avg_delta_binding_item']:+.4f}")
    
    log(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}")
        sys.exit(1)
    
    run_all(model_name)
    log("Phase 332 Layer Attribution complete!")
