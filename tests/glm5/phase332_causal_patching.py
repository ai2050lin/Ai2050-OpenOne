"""
Phase 332: Causal Patching — Key Layer Intervention
====================================================

Goal: From "binding trajectory" to "binding causal mechanism"

Method: Activation patching (output replacement at layer L)
- Clean run: "The apple" → binding_clean
- Source run: "The item" → capture all hidden states
- Patched run at layer L: "The apple" but layer L's output replaced with source
  → binding_patched
- Effect: Δ = binding_patched - binding_clean

Semantics: "What if the model processed 'The item' from layer L+1 onward?"
If Δ << 0 at layer L, the object-specific information up to layer L is
causally necessary for binding.

Key insight: Δ = gap(patched) - gap(clean) [gap(item) cancels out]
Max possible Δ = -binding_clean (complete destruction)

Expected: Δ becomes more negative at later layers, with sharp drops at
key binding formation layers identified in Phase 331b.

Usage:
  python tests/glm5/phase332_causal_patching.py qwen3
  python tests/glm5/phase332_causal_patching.py glm4
  python tests/glm5/phase332_causal_patching.py deepseek7b
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

# Test pairs — focused set for causal patching
TEST_PAIRS = [
    # HC (8) — core binding pairs
    ("apple", "red", "blue", "color", "high_compatible"),
    ("banana", "yellow", "purple", "color", "high_compatible"),
    ("snow", "white", "black", "color", "high_compatible"),
    ("sky", "blue", "green", "color", "high_compatible"),
    ("stone", "rough", "soft", "texture", "high_compatible"),
    ("ice", "cold", "hot", "temperature", "high_compatible"),
    ("fire", "hot", "cold", "temperature", "high_compatible"),
    ("silk", "smooth", "rough", "texture", "high_compatible"),
    # NI (4)
    ("apple", "blue", "black", "color", "near_incompatible"),
    ("snow", "pink", "orange", "color", "near_incompatible"),
    ("banana", "white", "black", "color", "near_incompatible"),
    ("sky", "red", "brown", "color", "near_incompatible"),
    # AA (4)
    ("idea", "red", "blue", "color", "abstract_absurd"),
    ("justice", "blue", "red", "color", "abstract_absurd"),
    ("democracy", "hot", "cold", "temperature", "abstract_absurd"),
    ("freedom", "rough", "smooth", "texture", "abstract_absurd"),
]

# Patch relative depths — dense in the key binding region
PATCH_REL_DEPTHS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                    0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

# Source prompt for patching
SRC_PROMPT = "The item"


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


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Cannot find layers in {type(model).__name__}")


def capture_source_hs(model, tokenizer, device, src_prompt):
    """Capture source hidden states (all positions, all layers) on CPU"""
    inp = tokenizer(src_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    # List of numpy arrays: (seq_len, d_model), one per layer boundary
    src_hs = [hs[0].float().cpu().numpy() for hs in out.hidden_states]
    src_seq_len = src_hs[0].shape[0]
    log(f"  Source '{src_prompt}': seq_len={src_seq_len}, hs_entries={len(src_hs)}")
    return src_hs, src_seq_len


def compute_gap(W_U, hs_last, tid_t, tid_c):
    """Compute gap = logit_target - logit_competitor"""
    logit_t = float(W_U[tid_t] @ hs_last)
    logit_c = float(W_U[tid_c] @ hs_last)
    return logit_t - logit_c


def run_clean(model, tokenizer, device, prompt):
    """Run clean forward pass, return final hidden state at last token"""
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    final_hs = out.hidden_states[-1][0, -1].float().cpu().numpy()
    seq_len = out.hidden_states[0].shape[1]
    return final_hs, seq_len


def run_patched(model, tokenizer, device, obj_prompt, src_hs, patch_layer):
    """
    Run forward pass with output patching at patch_layer.
    
    Replaces the OUTPUT of transformer layer patch_layer with source
    hidden states. Subsequent layers process source context.
    
    Returns final hidden state at last token position.
    """
    obj_inp = tokenizer(obj_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    # Source hidden state at layer boundary patch_layer+1
    # (hs[0] = embedding, hs[L+1] = output of layer L)
    src_np = src_hs[patch_layer + 1]  # (seq_src, d_model) numpy
    src_tensor_cpu = torch.tensor(src_np, dtype=torch.float32)
    
    final_hs = None
    
    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            out_hs = output[0]  # (1, seq_obj, d_model)
            target_device = out_hs.device
            target_dtype = out_hs.dtype
            
            src_t = src_tensor_cpu.to(device=target_device, dtype=target_dtype)
            src_t = src_t.unsqueeze(0)  # (1, seq_src, d_model)
            
            if src_t.shape[1] == out_hs.shape[1]:
                # Full replacement — seq lengths match
                new_hs = src_t
            else:
                # Length mismatch: replace only last token position
                new_hs = out_hs.clone()
                new_hs[:, -1:, :] = src_t[:, -1:, :]
            
            return (new_hs,) + output[1:]
        else:
            target_device = output.device
            target_dtype = output.dtype
            src_t = src_tensor_cpu.to(device=target_device, dtype=target_dtype).unsqueeze(0)
            return src_t
    
    layers = get_layers(model)
    hook = layers[patch_layer].register_forward_hook(patch_hook)
    
    try:
        with torch.no_grad():
            out = model(**obj_inp, output_hidden_states=True)
            final_hs = out.hidden_states[-1][0, -1].float().cpu().numpy()
    finally:
        hook.remove()
    
    return final_hs


def run_all(model_name):
    log(f"Phase 332: Causal Patching — {model_name}")
    log("=" * 60)
    
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    
    W_U = get_W_U(model, model_name)
    log(f"  W_U shape: {W_U.shape}")
    
    # Compute patch layers from relative depths
    patch_layers = sorted(set(
        min(int(rd * n_layers), n_layers - 1)
        for rd in PATCH_REL_DEPTHS
    ))
    log(f"  Patch layers: {patch_layers}")
    log(f"  Patch rel depths: {[f'{l/n_layers:.2f}' for l in patch_layers]}")
    
    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # ============================================================
    # STEP 0: Capture source hidden states
    # ============================================================
    log(f"\n=== STEP 0: Capturing source hidden states ===")
    src_hs, src_seq_len = capture_source_hs(model, tokenizer, device, SRC_PROMPT)
    # Also get item's final hidden state for binding_item computation
    item_final_hs = src_hs[-1][-1]  # last token, last layer
    
    # ============================================================
    # STEP 1: Run all test pairs — clean + patched
    # ============================================================
    log(f"\n=== STEP 1: Clean + Patched runs ===")
    
    level_order = ["high_compatible", "near_incompatible", "abstract_absurd"]
    all_results = []
    
    for idx, (obj, target_val, competitor_val, attr_type, compat_level) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target_val)
        tid_c = get_token_id(tokenizer, competitor_val)
        if tid_t is None or tid_c is None:
            log(f"  [{idx+1}] SKIP {obj}-{target_val}: token not found")
            continue
        
        obj_prompt = f"The {obj}"
        log(f"  [{idx+1}/{len(TEST_PAIRS)}] {obj_prompt} → {target_val}/{competitor_val} ({compat_level})")
        
        # Clean run
        clean_final_hs, obj_seq_len = run_clean(model, tokenizer, device, obj_prompt)
        clean_gap = compute_gap(W_U, clean_final_hs, tid_t, tid_c)
        item_gap = compute_gap(W_U, item_final_hs, tid_t, tid_c)
        clean_binding_item = clean_gap - item_gap
        
        if obj_seq_len != src_seq_len:
            log(f"    WARN: seq_len mismatch: obj={obj_seq_len}, src={src_seq_len}")
        
        log(f"    Clean: gap={clean_gap:+.3f}, item_gap={item_gap:+.3f}, binding_item={clean_binding_item:+.3f}")
        
        # Patched runs
        patched_results = []
        for pi, patch_l in enumerate(patch_layers):
            patched_final_hs = run_patched(
                model, tokenizer, device, obj_prompt, src_hs, patch_l
            )
            patched_gap = compute_gap(W_U, patched_final_hs, tid_t, tid_c)
            patched_binding_item = patched_gap - item_gap
            delta = patched_binding_item - clean_binding_item
            frac_destroyed = -delta / clean_binding_item if abs(clean_binding_item) > 0.01 else 0.0
            
            patched_results.append({
                "patch_layer": patch_l,
                "rel_depth": round(patch_l / n_layers, 3),
                "patched_binding_item": round(patched_binding_item, 4),
                "delta": round(delta, 4),
                "frac_destroyed": round(frac_destroyed, 4),
            })
            
            if (pi + 1) % 5 == 0:
                log(f"    Patched {pi+1}/{len(patch_layers)} layers done")
        
        result = {
            "obj": obj,
            "target_val": target_val,
            "competitor_val": competitor_val,
            "compat_level": compat_level,
            "obj_seq_len": obj_seq_len,
            "src_seq_len": src_seq_len,
            "clean_gap": round(clean_gap, 4),
            "item_gap": round(item_gap, 4),
            "clean_binding_item": round(clean_binding_item, 4),
            "patched_results": patched_results,
        }
        all_results.append(result)
        
        # Log key patching effects
        for pr in patched_results:
            rd = pr["rel_depth"]
            if rd in [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
                log(f"    L{pr['patch_layer']} (rel={rd:.2f}): "
                    f"Δ={pr['delta']:+.3f}, frac={pr['frac_destroyed']:.2f}")
        
        # Memory management
        del clean_final_hs
        if (idx + 1) % 4 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                log(f"    GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB, elapsed={time.time()-t0:.0f}s")
    
    # ============================================================
    # STEP 2: Aggregate by compat_level
    # ============================================================
    log(f"\n=== STEP 2: Aggregation by compat_level ===")
    
    level_aggregated = {}
    for cl in level_order:
        pair_results = [r for r in all_results if r["compat_level"] == cl]
        if not pair_results:
            continue
        
        avg_clean = float(np.mean([r["clean_binding_item"] for r in pair_results]))
        
        patch_effects = defaultdict(list)
        for r in pair_results:
            for pr in r["patched_results"]:
                patch_effects[pr["patch_layer"]].append(pr["delta"])
        
        patch_fracs = defaultdict(list)
        for r in pair_results:
            for pr in r["patched_results"]:
                patch_fracs[pr["patch_layer"]].append(pr["frac_destroyed"])
        
        aggregated = {
            "clean_binding_item": round(avg_clean, 4),
            "n_pairs": len(pair_results),
            "layer_effects": [],
        }
        
        for patch_l in patch_layers:
            deltas = patch_effects.get(patch_l, [])
            fracs = patch_fracs.get(patch_l, [])
            avg_delta = float(np.mean(deltas)) if deltas else 0.0
            avg_frac = float(np.mean(fracs)) if fracs else 0.0
            std_delta = float(np.std(deltas)) if len(deltas) > 1 else 0.0
            aggregated["layer_effects"].append({
                "patch_layer": patch_l,
                "rel_depth": round(patch_l / n_layers, 3),
                "avg_delta": round(avg_delta, 4),
                "std_delta": round(std_delta, 4),
                "avg_frac_destroyed": round(avg_frac, 4),
            })
        
        level_aggregated[cl] = aggregated
        log(f"  {cl}: clean_binding={avg_clean:+.3f}, n={len(pair_results)}")
    
    # ============================================================
    # STEP 3: Patching effect table (main output)
    # ============================================================
    log(f"\n{'='*90}")
    log(f"PATCHING EFFECT TABLE — {model_name} ({n_layers} layers)")
    log(f"Source: '{SRC_PROMPT}' | Metric: Δ_binding_item = patched - clean")
    log(f"{'='*90}")
    
    header = f"  {'Layer':>5} {'RelD':>6}"
    for cl in level_order:
        short = {"high_compatible": "HC", "near_incompatible": "NI", "abstract_absurd": "AA"}
        header += f"  {'Δ_'+short[cl]:>8} {'±':>4} {'Frac':>6}"
    header += f"  {'HC_patched':>10} {'AA_patched':>10} {'HC>AA':>6}"
    log(header)
    log("  " + "-" * (12 + 19 * len(level_order) + 27))
    
    for patch_l in patch_layers:
        rel = f"{patch_l/n_layers:.2f}"
        row = f"  L{patch_l:>4} {rel:>6}"
        patched_values = {}
        for cl in level_order:
            if cl in level_aggregated:
                le = level_aggregated[cl]["layer_effects"]
                effect = next((e for e in le if e["patch_layer"] == patch_l), None)
                if effect:
                    row += f"  {effect['avg_delta']:>+8.3f} {effect['std_delta']:>4.2f} {effect['avg_frac_destroyed']:>6.2f}"
                    patched_values[cl] = level_aggregated[cl]["clean_binding_item"] + effect["avg_delta"]
                else:
                    row += f"  {'N/A':>8} {'N/A':>4} {'N/A':>6}"
            else:
                row += f"  {'N/A':>8} {'N/A':>4} {'N/A':>6}"
        
        hc_pv = patched_values.get("high_compatible", None)
        aa_pv = patched_values.get("abstract_absurd", None)
        if hc_pv is not None and aa_pv is not None:
            hc_gt_aa = hc_pv > aa_pv
            row += f"  {hc_pv:>+10.3f} {aa_pv:>+10.3f} {str(hc_gt_aa):>6}"
        else:
            row += f"  {'N/A':>10} {'N/A':>10} {'N/A':>6}"
        log(row)
    
    # Clean baseline for reference
    log(f"\n  Clean baseline (no patching):")
    for cl in level_order:
        if cl in level_aggregated:
            log(f"    {cl}: binding_item={level_aggregated[cl]['clean_binding_item']:+.3f}")
    
    # ============================================================
    # STEP 4: Causal critical layers
    # ============================================================
    log(f"\n=== STEP 4: Causal critical layers ===")
    
    for cl in level_order:
        if cl not in level_aggregated:
            continue
        le = level_aggregated[cl]["layer_effects"]
        
        # Top 5 by fraction destroyed
        sorted_by_frac = sorted(le, key=lambda x: x["avg_frac_destroyed"], reverse=True)
        log(f"\n  {cl} — Top 5 causal critical layers:")
        for e in sorted_by_frac[:5]:
            log(f"    L{e['patch_layer']} (rel={e['rel_depth']:.2f}): "
                f"Δ={e['avg_delta']:+.3f} ±{e['std_delta']:.2f}, frac={e['avg_frac_destroyed']:.2f}")
        
        # 50% destruction layer
        for e in le:
            if e["avg_frac_destroyed"] >= 0.5:
                log(f"  {cl}: 50% binding destroyed at L{e['patch_layer']} (rel={e['rel_depth']:.2f})")
                break
    
    # ============================================================
    # STEP 5: Growth region analysis
    # ============================================================
    log(f"\n=== STEP 5: Growth vs non-growth region patching ===")
    
    growth_regions = {
        "qwen3": (28, 35),    # Phase 331b: max gain at L30
        "glm4": (25, 39),     # Phase 331b: first HC>AA at L27, max at L39
        "deepseek7b": (23, 27),  # Phase 331b: max gain at L24
    }
    
    if model_name in growth_regions:
        g_start, g_end = growth_regions[model_name]
        growth_layers = [l for l in patch_layers if g_start <= l <= g_end]
        non_growth_layers = [l for l in patch_layers if l < g_start]
        
        log(f"  Growth region: L{g_start}-L{g_end}")
        log(f"  Growth patch layers: {growth_layers}")
        log(f"  Non-growth patch layers: {non_growth_layers}")
        
        for cl in level_order:
            if cl not in level_aggregated:
                continue
            le = level_aggregated[cl]["layer_effects"]
            
            growth_fracs = [e["avg_frac_destroyed"] for e in le if e["patch_layer"] in growth_layers]
            non_growth_fracs = [e["avg_frac_destroyed"] for e in le if e["patch_layer"] in non_growth_layers]
            
            avg_g = float(np.mean(growth_fracs)) if growth_fracs else 0
            avg_ng = float(np.mean(non_growth_fracs)) if non_growth_fracs else 0
            
            log(f"  {cl}:")
            log(f"    Growth region:     avg frac destroyed = {avg_g:.3f} (n={len(growth_fracs)})")
            log(f"    Non-growth region: avg frac destroyed = {avg_ng:.3f} (n={len(non_growth_fracs)})")
            log(f"    Growth more causal: {avg_g > avg_ng}")
    
    # ============================================================
    # STEP 6: Sanity check — patching at L0 should give ≈0 binding
    # ============================================================
    log(f"\n=== STEP 6: Sanity check — patching at first layer ===")
    
    for cl in level_order:
        if cl not in level_aggregated:
            continue
        le = level_aggregated[cl]["layer_effects"]
        first_effect = next((e for e in le if e["patch_layer"] == patch_layers[0]), None)
        if first_effect:
            patched_binding = level_aggregated[cl]["clean_binding_item"] + first_effect["avg_delta"]
            log(f"  {cl}: clean={level_aggregated[cl]['clean_binding_item']:+.3f}, "
                f"patched_L0={patched_binding:+.3f}, "
                f"frac_destroyed={first_effect['avg_frac_destroyed']:.2f}")
    
    # ============================================================
    # STEP 7: Per-pair detail for key pairs
    # ============================================================
    log(f"\n=== STEP 7: Key per-pair patching curves ===")
    
    key_pairs = ["apple_red", "banana_yellow", "snow_white", "ice_cold", "idea_red", "justice_blue"]
    
    for r in all_results:
        key = f"{r['obj']}_{r['target_val']}"
        if key not in key_pairs:
            continue
        
        log(f"\n  {key} ({r['compat_level']}): clean_binding={r['clean_binding_item']:+.3f}")
        header = f"    {'Layer':>5} {'RelD':>6} {'Patched':>10} {'Delta':>8} {'Frac':>6}"
        log(header)
        log("    " + "-" * 38)
        
        for pr in r["patched_results"]:
            log(f"    L{pr['patch_layer']:>4} {pr['rel_depth']:>6.2f} "
                f"{pr['patched_binding_item']:>+10.3f} {pr['delta']:>+8.3f} {pr['frac_destroyed']:>6.2f}")
    
    # Release model
    del model, W_U, src_hs
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save results
    save_data = {
        "model": model_name,
        "n_layers": n_layers,
        "approach": "causal_patching_output_replacement",
        "source_prompt": SRC_PROMPT,
        "patch_layers": patch_layers,
        "patch_rel_depths": [round(l / n_layers, 3) for l in patch_layers],
        "level_aggregated": level_aggregated,
        "details": all_results,
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
    
    os.makedirs("results/phase332_causal_patching", exist_ok=True)
    out_path = f"results/phase332_causal_patching/{model_name}_phase332.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")
    
    # Final summary
    log(f"\n{'='*60}")
    log(f"SUMMARY — {model_name}")
    log(f"{'='*60}")
    
    for cl in level_order:
        if cl in level_aggregated:
            la = level_aggregated[cl]
            log(f"  {cl}: clean_binding={la['clean_binding_item']:+.3f}")
            le = la["layer_effects"]
            top3 = sorted(le, key=lambda x: x["avg_frac_destroyed"], reverse=True)[:3]
            for e in top3:
                log(f"    L{e['patch_layer']} (rel={e['rel_depth']:.2f}): "
                    f"Δ={e['avg_delta']:+.3f}, frac={e['avg_frac_destroyed']:.2f}")
    
    log(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}")
        sys.exit(1)
    
    run_all(model_name)
    log("Phase 332 complete!")
