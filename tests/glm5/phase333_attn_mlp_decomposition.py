"""
Phase 333: Component-level Binding Decomposition (Attention vs MLP)
===================================================================

Phase 332 showed layer attribution: which layers contribute most to binding_item.
This script decomposes each layer's contribution into attention vs MLP:

  h[L+1] = h[L] + attn_out_L + mlp_out_L
  Δ_binding_item_L = binding_dir @ (attn_out_L_obj + mlp_out_L_obj)
                    - binding_dir @ (attn_out_L_item + mlp_out_L_item)
                  = Δ_binding_item_L_attn + Δ_binding_item_L_mlp

Where:
  Δ_binding_item_L_attn = (binding_dir @ attn_out_obj_L) - (binding_dir @ attn_out_item_L)
  Δ_binding_item_L_mlp = (binding_dir @ mlp_out_obj_L) - (binding_dir @ mlp_out_item_L)

This answers: "Is binding computed by attention (context routing) or MLP (knowledge retrieval)?"

Expected outcomes:
- If MLP dominates → binding is knowledge retrieval (object→attribute mapping stored in MLP weights)
- If Attention dominates → binding is context routing (attention selects relevant attributes)
- If both contribute → binding is a distributed computation

Usage:
  python tests/glm5/phase333_attn_mlp_decomposition.py qwen3
  python tests/glm5/phase333_attn_mlp_decomposition.py glm4
  python tests/glm5/phase333_attn_mlp_decomposition.py deepseek7b
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

# Same test pairs as Phase 332
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

BASELINE_PROMPT = "The item"


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
    
    # Show layer distribution for device_map="auto"
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        layer_devices = {}
        for k, v in dmap.items():
            if k.startswith('model.layers.'):
                lid = k.split('.')[2]
                if lid not in layer_devices:
                    layer_devices[lid] = str(v)
        gpu_layers = sum(1 for v in layer_devices.values() if 'cuda' in v)
        cpu_layers = sum(1 for v in layer_devices.values() if 'cpu' in v)
        log(f"  Layer distribution: {gpu_layers} GPU + {cpu_layers} CPU")
    
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


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Cannot find transformer layers in {type(model).__name__}")


def run_with_hooks(model, tokenizer, device, prompt, n_layers):
    """Run model with hooks on self_attn and mlp for ALL layers.
    
    Returns:
        attn_outs: list of numpy arrays (d_model,) — attention output at last token position
        mlp_outs: list of numpy arrays (d_model,) — MLP output at last token position
        hidden_states: list of numpy arrays (d_model,) — hidden state at last token position
    """
    captured = {}
    layers = get_layers(model)
    
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook
    
    hooks = []
    for li in range(n_layers):
        layer = layers[li]
        # Hook on self_attn
        if hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(make_hook(f"attn_{li}")))
        # Hook on mlp
        if hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(make_hook(f"mlp_{li}")))
    
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    
    for h in hooks:
        h.remove()
    
    # Extract last token position for each component
    attn_outs = []
    mlp_outs = []
    hidden_states = [hs[0, -1].float().cpu().numpy() for hs in out.hidden_states]
    
    for li in range(n_layers):
        attn_key = f"attn_{li}"
        mlp_key = f"mlp_{li}"
        
        if attn_key in captured:
            attn_outs.append(captured[attn_key][0, -1].numpy())
        else:
            attn_outs.append(None)
        
        if mlp_key in captured:
            mlp_outs.append(captured[mlp_key][0, -1].numpy())
        else:
            mlp_outs.append(None)
    
    return attn_outs, mlp_outs, hidden_states


def run_all(model_name):
    log(f"Phase 333: Component-level Binding Decomposition — {model_name}")
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
    # STEP 0: Get baseline (item) component outputs
    # ============================================================
    log(f"\n=== STEP 0: Computing baseline component outputs ===")
    
    attn_item, mlp_item, hs_item = run_with_hooks(model, tokenizer, device, BASELINE_PROMPT, n_layers)
    log(f"  Baseline '{BASELINE_PROMPT}': {len(hs_item)} layers, "
        f"attn captured={sum(1 for a in attn_item if a is not None)}, "
        f"mlp captured={sum(1 for m in mlp_item if m is not None)}")
    
    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # ============================================================
    # STEP 1: Per-pair component-level attribution
    # ============================================================
    log(f"\n=== STEP 1: Per-pair component-level attribution ===")
    
    level_order = ["high_compatible", "near_incompatible", "cross_type", "abstract_absurd"]
    all_pair_data = []
    
    for idx, (obj, target_val, competitor_val, attr_type, compat_level) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target_val)
        tid_c = get_token_id(tokenizer, competitor_val)
        if tid_t is None or tid_c is None:
            continue
        
        obj_prompt = f"The {obj}"
        binding_dir = W_U[tid_t] - W_U[tid_c]  # (d_model,)
        
        # Get object component outputs
        attn_obj, mlp_obj, hs_obj = run_with_hooks(model, tokenizer, device, obj_prompt, n_layers)
        
        # Per-layer component contributions
        layer_decomps = []
        cumul_binding_attn = 0.0
        cumul_binding_mlp = 0.0
        cumul_binding_total = 0.0
        
        for l in range(n_layers):
            # Attention contribution
            if attn_obj[l] is not None and attn_item[l] is not None:
                gap_obj_attn = float(binding_dir @ attn_obj[l])
                gap_item_attn = float(binding_dir @ attn_item[l])
                delta_binding_attn = gap_obj_attn - gap_item_attn
            else:
                gap_obj_attn = 0.0
                gap_item_attn = 0.0
                delta_binding_attn = 0.0
            
            # MLP contribution
            if mlp_obj[l] is not None and mlp_item[l] is not None:
                gap_obj_mlp = float(binding_dir @ mlp_obj[l])
                gap_item_mlp = float(binding_dir @ mlp_item[l])
                delta_binding_mlp = gap_obj_mlp - gap_item_mlp
            else:
                gap_obj_mlp = 0.0
                gap_item_mlp = 0.0
                delta_binding_mlp = 0.0
            
            delta_binding_total = delta_binding_attn + delta_binding_mlp
            
            # Verification: compare with hidden-state-based computation
            diff_obj = hs_obj[l + 1] - hs_obj[l]
            diff_item = hs_item[l + 1] - hs_item[l]
            delta_gap_obj = float(binding_dir @ diff_obj)
            delta_gap_item = float(binding_dir @ diff_item)
            delta_binding_hs = delta_gap_obj - delta_gap_item
            
            cumul_binding_attn += delta_binding_attn
            cumul_binding_mlp += delta_binding_mlp
            cumul_binding_total += delta_binding_total
            
            layer_decomps.append({
                "layer": l,
                "rel_depth": round(l / n_layers, 3),
                "delta_binding_attn": round(delta_binding_attn, 4),
                "delta_binding_mlp": round(delta_binding_mlp, 4),
                "delta_binding_total": round(delta_binding_total, 4),
                "delta_binding_hs": round(delta_binding_hs, 4),
                "mismatch": round(abs(delta_binding_total - delta_binding_hs), 4),
                "gap_obj_attn": round(gap_obj_attn, 4),
                "gap_obj_mlp": round(gap_obj_mlp, 4),
                "cumul_binding_attn": round(cumul_binding_attn, 4),
                "cumul_binding_mlp": round(cumul_binding_mlp, 4),
                "cumul_binding_total": round(cumul_binding_total, 4),
            })
        
        # Final values for verification
        final_gap_obj = float(binding_dir @ hs_obj[-1])
        final_gap_item = float(binding_dir @ hs_item[-1])
        final_binding_item = final_gap_obj - final_gap_item
        
        embed_gap_obj = float(binding_dir @ hs_obj[0])
        embed_gap_item = float(binding_dir @ hs_item[0])
        embed_binding_item = embed_gap_obj - embed_gap_item
        
        result = {
            "obj": obj,
            "target_val": target_val,
            "competitor_val": competitor_val,
            "compat_level": compat_level,
            "final_binding_item": round(final_binding_item, 4),
            "embed_binding_item": round(embed_binding_item, 4),
            "total_attn": round(cumul_binding_attn, 4),
            "total_mlp": round(cumul_binding_mlp, 4),
            "attn_pct": round(100 * cumul_binding_attn / max(abs(cumul_binding_attn + cumul_binding_mlp), 1e-10), 1),
            "mlp_pct": round(100 * cumul_binding_mlp / max(abs(cumul_binding_attn + cumul_binding_mlp), 1e-10), 1),
            "layer_decomps": layer_decomps,
        }
        all_pair_data.append(result)
        
        if (idx + 1) % 5 == 0 or idx < 3:
            log(f"  [{idx+1}/{len(TEST_PAIRS)}] {obj}-{target_val} ({compat_level}): "
                f"final_binding={final_binding_item:+.3f}, "
                f"attn={cumul_binding_attn:+.3f} ({result['attn_pct']:.0f}%), "
                f"mlp={cumul_binding_mlp:+.3f} ({result['mlp_pct']:.0f}%)")
        
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
            attn_deltas = [r["layer_decomps"][l]["delta_binding_attn"] for r in pair_results]
            mlp_deltas = [r["layer_decomps"][l]["delta_binding_mlp"] for r in pair_results]
            total_deltas = [r["layer_decomps"][l]["delta_binding_total"] for r in pair_results]
            hs_deltas = [r["layer_decomps"][l]["delta_binding_hs"] for r in pair_results]
            mismatches = [r["layer_decomps"][l]["mismatch"] for r in pair_results]
            
            attn_cumuls = [r["layer_decomps"][l]["cumul_binding_attn"] for r in pair_results]
            mlp_cumuls = [r["layer_decomps"][l]["cumul_binding_mlp"] for r in pair_results]
            total_cumuls = [r["layer_decomps"][l]["cumul_binding_total"] for r in pair_results]
            
            layer_aggs.append({
                "layer": l,
                "rel_depth": round(l / n_layers, 3),
                "avg_delta_attn": round(float(np.mean(attn_deltas)), 4),
                "avg_delta_mlp": round(float(np.mean(mlp_deltas)), 4),
                "avg_delta_total": round(float(np.mean(total_deltas)), 4),
                "avg_delta_hs": round(float(np.mean(hs_deltas)), 4),
                "avg_mismatch": round(float(np.mean(mismatches)), 4),
                "std_delta_attn": round(float(np.std(attn_deltas)), 4),
                "std_delta_mlp": round(float(np.std(mlp_deltas)), 4),
                "avg_cumul_attn": round(float(np.mean(attn_cumuls)), 4),
                "avg_cumul_mlp": round(float(np.mean(mlp_cumuls)), 4),
                "avg_cumul_total": round(float(np.mean(total_cumuls)), 4),
            })
        
        # Summary stats
        avg_final = float(np.mean([r["final_binding_item"] for r in pair_results]))
        avg_attn = float(np.mean([r["total_attn"] for r in pair_results]))
        avg_mlp = float(np.mean([r["total_mlp"] for r in pair_results]))
        avg_attn_pct = float(np.mean([r["attn_pct"] for r in pair_results]))
        avg_mlp_pct = float(np.mean([r["mlp_pct"] for r in pair_results]))
        
        level_trajectories[cl] = {
            "n_pairs": len(pair_results),
            "avg_final_binding_item": round(avg_final, 4),
            "avg_total_attn": round(avg_attn, 4),
            "avg_total_mlp": round(avg_mlp, 4),
            "avg_attn_pct": round(avg_attn_pct, 1),
            "avg_mlp_pct": round(avg_mlp_pct, 1),
            "layer_aggs": layer_aggs,
        }
        
        log(f"  {cl}: n={len(pair_results)}, final={avg_final:+.3f}, "
            f"attn={avg_attn:+.3f} ({avg_attn_pct:.0f}%), mlp={avg_mlp:+.3f} ({avg_mlp_pct:.0f}%)")
    
    # ============================================================
    # STEP 3: Component decomposition table
    # ============================================================
    log(f"\n{'='*120}")
    log(f"COMPONENT DECOMPOSITION TABLE — {model_name} ({n_layers} layers)")
    log(f"Δ_binding_item_L = Δ_binding_item_L_attn + Δ_binding_item_L_mlp")
    log(f"{'='*120}")
    
    # Sample layers for display
    sample_layers = list(range(0, n_layers, max(1, n_layers // 20)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))
    
    # HC table
    if "high_compatible" in level_trajectories:
        log(f"\n  HC (High Compatible) — Per-layer Attention vs MLP decomposition:")
        log(f"  {'Layer':>5} {'RelD':>6} {'Δ_attn':>8} {'Δ_mlp':>8} {'Δ_total':>8} {'Cum_attn':>9} {'Cum_mlp':>9} {'Cum_tot':>9} {'Attn%':>6}")
        log("  " + "-" * 82)
        
        for l in sample_layers:
            la = level_trajectories["high_compatible"]["layer_aggs"][l]
            total_cum = la["avg_cumul_attn"] + la["avg_cumul_mlp"]
            if abs(total_cum) > 0.01:
                attn_pct = 100 * la["avg_cumul_attn"] / abs(total_cum)
            else:
                attn_pct = 0.0
            rel = f"{l/n_layers:.2f}"
            log(f"  L{l:>4} {rel:>6} {la['avg_delta_attn']:>+8.3f} {la['avg_delta_mlp']:>+8.3f} "
                f"{la['avg_delta_total']:>+8.3f} {la['avg_cumul_attn']:>+9.3f} {la['avg_cumul_mlp']:>+9.3f} "
                f"{la['avg_cumul_total']:>+9.3f} {attn_pct:>5.1f}%")
    
    # ============================================================
    # STEP 4: Key binding layers — detailed decomposition
    # ============================================================
    log(f"\n=== STEP 4: Key binding layers — detailed decomposition ===")
    
    for cl in level_order:
        if cl not in level_trajectories:
            continue
        la = level_trajectories[cl]["layer_aggs"]
        
        # Top 5 layers by total |Δ_binding|
        sorted_by_total = sorted(la, key=lambda x: abs(x["avg_delta_total"]), reverse=True)
        log(f"\n  {cl} — Top 5 layers by |Δ_binding_item| (attn vs mlp):")
        for e in sorted_by_total[:5]:
            total = abs(e["avg_delta_attn"]) + abs(e["avg_delta_mlp"])
            if total > 0.01:
                attn_frac = abs(e["avg_delta_attn"]) / total * 100
                mlp_frac = abs(e["avg_delta_mlp"]) / total * 100
            else:
                attn_frac = mlp_frac = 0
            log(f"    L{e['layer']} (rel={e['rel_depth']:.2f}): "
                f"attn={e['avg_delta_attn']:+.4f} ({attn_frac:.0f}%), "
                f"mlp={e['avg_delta_mlp']:+.4f} ({mlp_frac:.0f}%), "
                f"total={e['avg_delta_total']:+.4f}, mismatch={e['avg_mismatch']:.4f}")
    
    # ============================================================
    # STEP 5: Cumulative decomposition at key depth fractions
    # ============================================================
    log(f"\n=== STEP 5: Cumulative attn vs mlp at key depth fractions ===")
    
    depth_fractions = [0.25, 0.50, 0.75, 1.0]
    for cl in level_order:
        if cl not in level_trajectories:
            continue
        la = level_trajectories[cl]["layer_aggs"]
        log(f"\n  {cl}:")
        log(f"    {'Depth':>6} {'Cum_attn':>10} {'Cum_mlp':>10} {'Cum_total':>10} {'Attn%':>6}")
        
        for frac in depth_fractions:
            target_l = int(frac * n_layers)
            target_l = min(target_l, n_layers - 1)
            e = la[target_l]
            total_cum = e["avg_cumul_attn"] + e["avg_cumul_mlp"]
            if abs(total_cum) > 0.01:
                attn_pct = 100 * e["avg_cumul_attn"] / abs(total_cum)
            else:
                attn_pct = 0.0
            log(f"    {frac:>5.0f}% {e['avg_cumul_attn']:>+10.3f} {e['avg_cumul_mlp']:>+10.3f} "
                f"{e['avg_cumul_total']:>+10.3f} {attn_pct:>5.1f}%")
    
    # ============================================================
    # STEP 6: Verification — attn+mlp should match hidden-state decomposition
    # ============================================================
    log(f"\n=== STEP 6: Verification — attn+mlp vs hidden-state decomposition ===")
    
    max_mismatch = 0.0
    for cl in level_order:
        if cl not in level_trajectories:
            continue
        la = level_trajectories[cl]["layer_aggs"]
        for e in la:
            max_mismatch = max(max_mismatch, e["avg_mismatch"])
    log(f"  Max avg mismatch across all levels/layers: {max_mismatch:.4f}")
    if max_mismatch < 0.5:
        log(f"  ✓ Mismatch < 0.5 — decomposition is consistent!")
    else:
        log(f"  ⚠ Mismatch > 0.5 — some layers have hook capture issues")
    
    # ============================================================
    # STEP 7: Per-pair key examples
    # ============================================================
    log(f"\n=== STEP 7: Key per-pair examples ===")
    
    key_pairs = ["apple_red", "banana_yellow", "snow_white", "ice_cold",
                 "idea_red", "justice_blue"]
    
    for r in all_pair_data:
        key = f"{r['obj']}_{r['target_val']}"
        if key not in key_pairs:
            continue
        
        log(f"\n  {key} ({r['compat_level']}): final={r['final_binding_item']:+.3f}, "
            f"attn={r['total_attn']:+.3f} ({r['attn_pct']:.0f}%), "
            f"mlp={r['total_mlp']:+.3f} ({r['mlp_pct']:.0f}%)")
        header = (f"    {'Layer':>5} {'RelD':>6} {'Δ_attn':>8} {'Δ_mlp':>8} "
                  f"{'Δ_total':>8} {'Cum_attn':>9} {'Cum_mlp':>9}")
        log(header)
        log("    " + "-" * 54)
        
        for l in sample_layers:
            if l < len(r["layer_decomps"]):
                ld = r["layer_decomps"][l]
                rel = f"{l/n_layers:.2f}"
                log(f"    L{l:>4} {rel:>6} {ld['delta_binding_attn']:>+8.3f} "
                    f"{ld['delta_binding_mlp']:>+8.3f} {ld['delta_binding_total']:>+8.3f} "
                    f"{ld['cumul_binding_attn']:>+9.3f} {ld['cumul_binding_mlp']:>+9.3f}")
    
    # Release model
    del model, W_U
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save results
    save_data = {
        "model": model_name,
        "n_layers": n_layers,
        "approach": "attn_mlp_decomposition_binding_item",
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
    
    os.makedirs("results/phase333_attn_mlp_decomposition", exist_ok=True)
    out_path = f"results/phase333_attn_mlp_decomposition/{model_name}_phase333.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")
    
    # ============================================================
    # Final summary
    # ============================================================
    log(f"\n{'='*60}")
    log(f"SUMMARY — {model_name}")
    log(f"{'='*60}")
    
    for cl in level_order:
        if cl in level_trajectories:
            lt = level_trajectories[cl]
            log(f"  {cl}: final={lt['avg_final_binding_item']:+.3f}, "
                f"attn={lt['avg_total_attn']:+.3f} ({lt['avg_attn_pct']:.0f}%), "
                f"mlp={lt['avg_total_mlp']:+.3f} ({lt['avg_mlp_pct']:.0f}%)")
            
            # Top 3 contributing layers (by total)
            la = lt["layer_aggs"]
            top3 = sorted(la, key=lambda x: abs(x["avg_delta_total"]), reverse=True)[:3]
            for e in top3:
                total = abs(e["avg_delta_attn"]) + abs(e["avg_delta_mlp"])
                if total > 0.01:
                    attn_frac = abs(e["avg_delta_attn"]) / total * 100
                else:
                    attn_frac = 0
                log(f"    L{e['layer']} (rel={e['rel_depth']:.2f}): "
                    f"attn={e['avg_delta_attn']:+.4f} ({attn_frac:.0f}%), "
                    f"mlp={e['avg_delta_mlp']:+.4f} ({100-attn_frac:.0f}%)")
    
    log(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}")
        sys.exit(1)
    
    run_all(model_name)
    log("Phase 333 complete!")
