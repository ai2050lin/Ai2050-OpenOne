"""
Phase 404: Component Attribution - MLP vs Attention
====================================================

核心问题:
1. 深层符号反转是RMSNorm还是MLP导致的?
2. vehicle→vehicle强传递来自哪个组件?
3. animal→animal弱传递是因为attention主导?
4. SPEED成分和TYPE成分分别在哪里产生/被放大/被抑制?

测试设计:
- 对每一层,分别patch: attention_output, mlp_output, post-RMSNorm
- 检查速度方向的odd/even效应在哪个组件上最强
- 追踪分布压缩发生在哪里

关键指标:
- per_component odd (方向效应)
- per_component even (范数效应)
- entropy change per component
- fast/slow candidate odd per component

Usage:
  python tests/glm5/phase404_component_attribution.py qwen3
  python tests/glm5/phase404_component_attribution.py deepseek7b
  python tests/glm5/phase404_component_attribution.py glm4
"""

import sys
import os
import json
import time
import gc
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS, get_layers, get_model_info, release_model

# 6 objects
SPEED_OBJECTS = {
    "snail":      {"type": "animal",     "speed_level": 1, "target": "slow",   "comp": "fast"},
    "cheetah":    {"type": "animal",     "speed_level": 5, "target": "fast",   "comp": "slow"},
    "bicycle":    {"type": "vehicle",    "speed_level": 2, "target": "slow",   "comp": "fast"},
    "rocket":     {"type": "vehicle",    "speed_level": 5, "target": "fast",   "comp": "slow"},
    "glacier":    {"type": "phenomenon", "speed_level": 1, "target": "slow",   "comp": "fast"},
    "lightning":  {"type": "phenomenon", "speed_level": 5, "target": "fast",   "comp": "slow"},
}

SPEED_CANDIDATES = {
    "sluggish": 1, "slow": 2, "steady": 3, "moderate": 4,
    "quick": 5, "fast": 6, "rapid": 7, "swift": 8,
}

FRAMES = [
    "The {obj} is {attr}.",
    "An {obj} is {attr}.",
]

CORRUPT_FRAMES = [
    "The item is {attr}.",
    "An item is {attr}.",
]

LAYER_CONFIGS = {
    "qwen3": [4, 28],
    "deepseek7b": [4, 20],
    "glm4": [5, 35],
}


def log_memory():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU: {alloc:.2f}GB alloc, {reserved:.2f}GB reserved"
    return "GPU not available"


def load_model_bf16_safe(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name} (BF16+auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    for impl in ["eager", "sdpa"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=impl
            )
            break
        except Exception as e:
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    print(f"  Loaded. {log_memory()}")
    return model, tokenizer


def compute_speed_direction(model, tokenizer, layers_list, device, li, obj_name, obj_data, token_ids):
    """Compute speed direction at residual stream level."""
    target = obj_data["target"]
    comp = obj_data["comp"]
    tid = token_ids.get(target)
    cid = token_ids.get(comp)
    
    h_correct_list = []
    h_corrupt_list = []
    baseline_diffs = []
    
    captured = {}
    def make_hook(key):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook_fn
    
    handle = layers_list[li].register_forward_hook(make_hook('h'))
    
    for f_idx in range(2):
        correct_clean = FRAMES[f_idx].format(obj=obj_name, attr=target)
        correct_corrupt = CORRUPT_FRAMES[f_idx].format(attr=target)
        
        captured.clear()
        inputs = tokenizer(correct_clean, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            model(input_ids=inputs["input_ids"].to(device),
                  attention_mask=inputs["attention_mask"].to(device))
        h_correct_list.append(captured['h'][0, -1].numpy())
        
        captured.clear()
        inputs = tokenizer(correct_corrupt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out = model(input_ids=inputs["input_ids"].to(device),
                       attention_mask=inputs["attention_mask"].to(device))
        h_corrupt_list.append(captured['h'][0, -1].numpy())
        diff = float(out.logits[0, -1, tid].float().cpu() - out.logits[0, -1, cid].float().cpu()) if tid is not None and cid is not None else 0.0
        baseline_diffs.append(diff)
    
    handle.remove()
    dh = np.mean(np.array(h_correct_list) - np.array(h_corrupt_list), axis=0)
    baseline = float(np.mean(baseline_diffs))
    return dh, baseline


def get_submodules(layer, mlp_type):
    """Get attention, MLP, and layernorm submodules from a transformer layer."""
    # Attention output
    attn_out = None
    for name in ["self_attn", "attention", "attn"]:
        if hasattr(layer, name):
            sa = getattr(layer, name)
            for oname in ["o_proj", "dense", "out_proj"]:
                if hasattr(sa, oname):
                    attn_out = getattr(sa, oname)
                    break
            break
    
    # MLP output (down_proj)
    mlp_down = None
    mlp_gate_up = None
    mlp_gate = None
    mlp_up = None
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        for dname in ["down_proj", "dense_4h_to_h"]:
            if hasattr(mlp, dname):
                mlp_down = getattr(mlp, dname)
                break
        if mlp_type == "merged_gate_up":
            for gname in ["gate_up_proj", "dense_h_to_4h"]:
                if hasattr(mlp, gname):
                    mlp_gate_up = getattr(mlp, gname)
                    break
        else:
            for gname in ["gate_proj"]:
                if hasattr(mlp, gname):
                    mlp_gate = getattr(mlp, gname)
                    break
            for uname in ["up_proj"]:
                if hasattr(mlp, uname):
                    mlp_up = getattr(mlp, uname)
                    break
    
    # LayerNorm
    input_ln = None
    post_attn_ln = None
    for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
        if hasattr(layer, ln_name):
            input_ln = getattr(layer, ln_name)
            break
    for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
        if hasattr(layer, ln_name):
            post_attn_ln = getattr(layer, ln_name)
            break
    
    return {
        'attn_out': attn_out,
        'mlp_down': mlp_down,
        'mlp_gate_up': mlp_gate_up,
        'mlp_gate': mlp_gate,
        'mlp_up': mlp_up,
        'input_ln': input_ln,
        'post_attn_ln': post_attn_ln,
    }


def component_patch_test(model, tokenizer, device, submodules, comp_name, submodule,
                         direction, obj_name, obj_data, token_ids, obj_baseline):
    """Test patching a specific submodule with a direction.
    
    Note: For submodules with different output dimensions (e.g., gate_proj, up_proj),
    we skip direct injection and only inject at the residual stream level.
    For attn_out and mlp_down, the output dimension matches d_model.
    """
    tgt_data = SPEED_OBJECTS[obj_name]
    tgt_tid = token_ids.get(tgt_data["target"])
    tgt_cid = token_ids.get(tgt_data["comp"])
    tgt_prompt = CORRUPT_FRAMES[0].format(attr=tgt_data["target"])
    
    if tgt_tid is None or tgt_cid is None or submodule is None:
        return None
    
    # Check if output dimension matches direction dimension
    # Get output dimension from weight shape
    out_dim = direction.shape[0]
    if hasattr(submodule, 'weight'):
        weight_shape = submodule.weight.shape
        # For linear layers: weight is [out_features, in_features]
        # output of the module has shape [batch, seq, out_features]
        module_out_dim = weight_shape[0]
        if module_out_dim != out_dim:
            # Skip submodules with mismatched dimensions
            return None
    
    # +direction
    delta = torch.tensor(direction, dtype=torch.bfloat16, device=device)
    def make_add_hook(dv):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0].clone()
                hs[0, -1, :] += dv
                return (hs,) + output[1:]
            else:
                hs = output.clone()
                hs[0, -1, :] += dv
                return hs
        return hook_fn
    
    h = submodule.register_forward_hook(make_add_hook(delta))
    try:
        inputs = tokenizer(tgt_prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out_plus = model(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device))
        diff_plus = float(out_plus.logits[0, -1, tgt_tid].float().cpu() - out_plus.logits[0, -1, tgt_cid].float().cpu())
    finally:
        h.remove()
    
    # -direction
    neg_delta = torch.tensor(-direction, dtype=torch.bfloat16, device=device)
    h2 = submodule.register_forward_hook(make_add_hook(neg_delta))
    try:
        inputs = tokenizer(tgt_prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out_minus = model(input_ids=inputs["input_ids"].to(device),
                             attention_mask=inputs["attention_mask"].to(device))
        diff_minus = float(out_minus.logits[0, -1, tgt_tid].float().cpu() - out_minus.logits[0, -1, tgt_cid].float().cpu())
    finally:
        h2.remove()
    
    plus_eff = diff_plus - obj_baseline
    minus_eff = diff_minus - obj_baseline
    odd = (plus_eff - minus_eff) / 2
    even = (plus_eff + minus_eff) / 2
    
    return {
        'odd': float(odd),
        'even': float(even),
        'plus_eff': float(plus_eff),
        'minus_eff': float(minus_eff),
    }


def run_phase404(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 404: Component Attribution ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")
    
    layer_indices = LAYER_CONFIGS.get(model_name, [4])
    obj_names = sorted(SPEED_OBJECTS.keys())
    print(f"  Layers: {layer_indices}")
    print(f"  Objects: {len(obj_names)}")
    
    # Load model
    model, tokenizer = load_model_bf16_safe(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    mlp_type = info.mlp_type
    device = next(model.parameters()).device
    
    # Resolve token IDs
    token_ids = {}
    for cand_name in SPEED_CANDIDATES:
        ids = tokenizer.encode(cand_name, add_special_tokens=False)
        token_ids[cand_name] = ids[0] if ids else None
    for obj_name, obj_data in SPEED_OBJECTS.items():
        for tok in [obj_data["target"], obj_data["comp"]]:
            if tok not in token_ids:
                ids = tokenizer.encode(tok, add_special_tokens=False)
                token_ids[tok] = ids[0] if ids else None
    
    print(f"  Token IDs: fast={token_ids.get('fast')}, slow={token_ids.get('slow')}")
    
    all_results = {
        'model': model_name,
        'timestamp': timestamp,
        'per_layer': {},
    }
    
    for li in layer_indices:
        t0_layer = time.time()
        print(f"\n{'='*70}")
        print(f"--- Layer {li} ---")
        
        # Step 1: Compute speed directions
        print(f"\n  Step 1: Computing speed directions...")
        obj_dirs = {}
        obj_baselines = {}
        for obj_name in obj_names:
            obj_data = SPEED_OBJECTS[obj_name]
            dh, baseline = compute_speed_direction(model, tokenizer, layers_list, device, li,
                                                    obj_name, obj_data, token_ids)
            obj_dirs[obj_name] = dh
            obj_baselines[obj_name] = baseline
            print(f"    {obj_name}: |dir|={np.linalg.norm(dh):.4f} baseline={baseline:.4f}")
        
        # Step 2: Get submodules
        print(f"\n  Step 2: Getting submodules...")
        submodules = get_submodules(layers_list[li], mlp_type)
        avail = {k: v is not None for k, v in submodules.items()}
        print(f"    Available: {', '.join(k for k, v in avail.items() if v)}")
        
        # Step 3: Component-wise patching
        # Use representative source objects: cheetah(animal-fast), rocket(vehicle-fast), lightning(phenomenon-fast)
        # Target: snail(animal-slow), bicycle(vehicle-slow), glacier(phenomenon-slow)
        print(f"\n  Step 3: Component-wise patching...")
        
        component_names = ['attn_out', 'mlp_down', 'mlp_gate_up', 'mlp_gate', 'mlp_up']
        available_components = [cn for cn in component_names if submodules.get(cn) is not None]
        
        # Representative pairs
        test_pairs = [
            ("cheetah", "snail", True),     # animal→animal (same type)
            ("cheetah", "bicycle", False),   # animal→vehicle (across type)
            ("rocket", "bicycle", True),     # vehicle→vehicle (same type)
            ("rocket", "snail", False),     # vehicle→animal (across type)
            ("lightning", "glacier", True), # phenomenon→phenomenon (same type)
            ("lightning", "snail", False),  # phenomenon→animal (across type)
        ]
        
        component_results = {}
        
        for src_name, tgt_name, is_same_type in test_pairs:
            src_dir = obj_dirs[src_name]
            src_data = SPEED_OBJECTS[src_name]
            tgt_data = SPEED_OBJECTS[tgt_name]
            
            src_result = {}
            
            # First: patch at residual stream (layer output)
            res = component_patch_test(model, tokenizer, device, submodules,
                                      'residual_stream', layers_list[li],
                                      src_dir, tgt_name, tgt_data, token_ids, obj_baselines[tgt_name])
            if res:
                src_result['residual_stream'] = res
            
            # Then: patch each component
            for comp_name in available_components:
                submodule = submodules[comp_name]
                # Scale direction by relative norm
                # Component outputs have different norms, so we scale proportionally
                res = component_patch_test(model, tokenizer, device, submodules,
                                          comp_name, submodule,
                                          src_dir, tgt_name, tgt_data, token_ids, obj_baselines[tgt_name])
                if res:
                    src_result[comp_name] = res
            
            component_results[f"{src_name}->{tgt_name}"] = {
                'same_type': is_same_type,
                'src_type': src_data['type'],
                'tgt_type': tgt_data['type'],
                'src_speed': src_data['speed_level'],
                'tgt_speed': tgt_data['speed_level'],
                'results': src_result,
            }
            
            rs_odd = src_result.get('residual_stream', {}).get('odd', 'N/A')
            attn_odd = src_result.get('attn_out', {}).get('odd', 'N/A')
            mlp_odd = src_result.get('mlp_down', {}).get('odd', 'N/A')
            print(f"    {src_name}->{tgt_name} (same_type={is_same_type}): "
                  f"RS_odd={rs_odd}, Attn_odd={attn_odd}, MLP_odd={mlp_odd}")
        
        # Step 4: Aggregate analysis
        print(f"\n  Step 4: Aggregate analysis...")
        
        # Compare component contributions for within-type vs across-type
        within_by_comp = defaultdict(list)
        across_by_comp = defaultdict(list)
        
        for pair_key, pair_data in component_results.items():
            is_same = pair_data['same_type']
            for comp_name, comp_result in pair_data['results'].items():
                if is_same:
                    within_by_comp[comp_name].append(comp_result['odd'])
                else:
                    across_by_comp[comp_name].append(comp_result['odd'])
        
        print(f"\n    Component Attribution (odd effects):")
        print(f"    {'Component':>20s} {'Within-type odd':>16s} {'Across-type odd':>16s} {'Diff':>10s} {'Ratio':>8s}")
        
        all_components = ['residual_stream'] + available_components
        for comp_name in all_components:
            w_vals = within_by_comp.get(comp_name, [])
            a_vals = across_by_comp.get(comp_name, [])
            mw = np.mean(w_vals) if w_vals else 0
            ma = np.mean(a_vals) if a_vals else 0
            diff = mw - ma
            ratio = mw / ma if abs(ma) > 1e-6 else float('inf')
            print(f"    {comp_name:>20s} {mw:>+16.4f} {ma:>+16.4f} {diff:>+10.4f} {ratio:>+8.2f}")
        
        # Even effects (norm-related)
        print(f"\n    Component Attribution (even effects / norm):")
        within_even_by_comp = defaultdict(list)
        across_even_by_comp = defaultdict(list)
        for pair_key, pair_data in component_results.items():
            is_same = pair_data['same_type']
            for comp_name, comp_result in pair_data['results'].items():
                if is_same:
                    within_even_by_comp[comp_name].append(comp_result['even'])
                else:
                    across_even_by_comp[comp_name].append(comp_result['even'])
        
        print(f"    {'Component':>20s} {'Within-type even':>17s} {'Across-type even':>17s}")
        for comp_name in all_components:
            w_vals = within_even_by_comp.get(comp_name, [])
            a_vals = across_even_by_comp.get(comp_name, [])
            mw = np.mean(w_vals) if w_vals else 0
            ma = np.mean(a_vals) if a_vals else 0
            print(f"    {comp_name:>20s} {mw:>+17.4f} {ma:>+17.4f}")
        
        # Store layer results
        layer_result = {
            'component_results': {},
            'aggregate': {
                'within_odd': {cn: float(np.mean(v)) if v else 0 for cn, v in within_by_comp.items()},
                'across_odd': {cn: float(np.mean(v)) if v else 0 for cn, v in across_by_comp.items()},
                'within_even': {cn: float(np.mean(v)) if v else 0 for cn, v in within_even_by_comp.items()},
                'across_even': {cn: float(np.mean(v)) if v else 0 for cn, v in across_even_by_comp.items()},
            }
        }
        # Simplify component results for storage
        for pair_key, pair_data in component_results.items():
            layer_result['component_results'][pair_key] = {
                'same_type': pair_data['same_type'],
                'results': pair_data['results'],
            }
        
        all_results['per_layer'][str(li)] = layer_result
        print(f"\n  L{li} done in {time.time()-t0_layer:.0f}s. {log_memory()}")
    
    # Cross-layer summary
    print(f"\n{'='*80}")
    print(f"=== Cross-Layer Summary ({model_name}) ===")
    print(f"{'='*80}")
    
    for li in layer_indices:
        lr = all_results['per_layer'].get(str(li), {})
        agg = lr.get('aggregate', {})
        print(f"\n  L{li}:")
        for comp_name in ['residual_stream'] + available_components:
            w_odd = agg.get('within_odd', {}).get(comp_name, 0)
            a_odd = agg.get('across_odd', {}).get(comp_name, 0)
            print(f"    {comp_name:>20s}: within_odd={w_odd:+.4f}, across_odd={a_odd:+.4f}, diff={w_odd-a_odd:+.4f}")
    
    # Save results
    out_dir = ROOT / "results" / "phase404_component_attribution"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_phase404.json"
    
    import copy
    results_to_save = copy.deepcopy(all_results)
    with open(out_path, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    # Cleanup
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Released. {log_memory()}")
    
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase404(model_name)
