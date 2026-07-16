"""
Phase 947 Minimal: GLM4 Color → MLP Channel Gap (Direct Hook)
==============================================================
最简版本：直接用MLP hooks捕获gate/up激活，对比color vs function的通道级差异。
不需要权重提取，不需要完整hidden states，可以在5分钟内存完成。

Usage: python tests/glm5_temp/phase947_minimal.py
"""
import sys, os, gc, time, json, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))
import torch
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

TMP_DIR = Path("tmp"); TMP_DIR.mkdir(parents=True, exist_ok=True)
_log_file = None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file, "a", encoding="utf-8") as f: f.write(line + "\n")
        except: pass

# Minimal stimuli: 3 colors × 3 functions × 3 objects = 18 sentences
COLORS = ["red", "blue", "green"]
FUNCTIONS = ["large", "heavy", "fast"]
OBJECTS = ["apple", "car", "ball"]
FRAME = "the {obj} is {attr}"

def build_sentences():
    sents = []
    for color in COLORS:
        for obj in OBJECTS:
            sents.append({"sentence": FRAME.format(obj=obj, attr=color),
                         "attr_type": "color", "attr": color, "object": obj})
    for func in FUNCTIONS:
        for obj in OBJECTS:
            sents.append({"sentence": FRAME.format(obj=obj, attr=func),
                         "attr_type": "function", "attr": func, "object": obj})
    return sents

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS["glm4"]
    log("Loading glm4...")
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    for attn in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(cfg["path"], torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True, local_files_only=True, attn_implementation=attn)
            log(f"  attn={attn} OK"); break
        except: pass
    model.eval()
    log(f"  GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return model, tok

def get_mlp_activations(model, tokenizer, sent, layer_idx):
    """Get gate*SiLU(gate), up activations at last position for a specific layer."""
    dev = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=24)
    input_ids = inputs["input_ids"].to(dev)
    
    layers = get_layers(model)
    mlp = layers[layer_idx].mlp
    captured = {}
    
    def gate_up_hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        captured["gate_up"] = out.detach().cpu().float()
    
    h = mlp.gate_up_proj.register_forward_hook(gate_up_hook)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    h.remove()
    
    if "gate_up" not in captured:
        return None, None, None, None
    
    gate_up = captured["gate_up"]  # [batch, seq, 2*intermediate]
    inter = gate_up.shape[-1] // 2
    gate_raw = gate_up[:, :, :inter]  # pre-activation
    up_raw = gate_up[:, :, inter:]    # up projection
    
    # Apply SiLU activation: gate * sigmoid(gate)
    gate_act = gate_raw * torch.sigmoid(gate_raw)
    
    # Last position activations
    last_gate = gate_act[0, -1, :].numpy()
    last_up = up_raw[0, -1, :].numpy()
    
    # Also get hidden states at last position
    hs_last = out.hidden_states[layer_idx][0, -1, :].detach().cpu().float().numpy()
    logits_last = out.logits[0, -1, :].detach().cpu().float().numpy()
    
    return last_gate, last_up, hs_last, logits_last

def main():
    global _log_file
    _log_file = str(TMP_DIR / "phase947_minimal.log")
    log("Phase 947 Minimal: GLM4 MLP Channel Gap")
    
    stimuli = build_sentences()
    log(f"Stimuli: {len(stimuli)} sentences")
    
    model, tokenizer = load_model()
    info = get_model_info(model, "glm4")
    nl, d_model, inter = info.n_layers, info.d_model, info.intermediate_size
    mid = nl // 2
    log(f"nl={nl}, d_model={d_model}, inter={inter}, mid={mid}")
    
    # Test layers: mid-5 to mid+10
    test_layers = [mid + d for d in range(-5, 11, 2)]
    log(f"Test layers: {test_layers}")
    
    all_results = {}
    
    for layer in test_layers:
        if layer < 0 or layer >= nl: continue
        log(f"\n--- Testing L{layer} ---")
        
        color_gates, Color_ups = [], []
        func_gates, func_ups = [], []
        color_logits_list, func_logits_list = [], []
        
        for stim in stimuli:
            gate_act, up_act, hs_last, logits = get_mlp_activations(model, tokenizer, stim["sentence"], layer)
            if gate_act is None: continue
            
            if stim["attr_type"] == "color":
                color_gates.append(gate_act)
                Color_ups.append(up_act)
                color_logits_list.append(logits)
            else:
                func_gates.append(gate_act)
                func_ups.append(up_act)
                func_logits_list.append(logits)
        
        if not color_gates or not func_gates: 
            log(f"  Capture failed"); continue
        
        # Mean activations
        color_gate_mean = np.mean(color_gates, axis=0)
        func_gate_mean = np.mean(func_gates, axis=0)
        color_up_mean = np.mean(Color_ups, axis=0)
        func_up_mean = np.mean(func_ups, axis=0)
        
        # Channel-wise differences
        gate_diff = np.abs(color_gate_mean - func_gate_mean)
        up_diff = np.abs(color_up_mean - func_up_mean)
        combined = gate_diff * up_diff  # elementwise product for significance
        
        # Top channels
        top_gate = np.argsort(gate_diff)[-20:][::-1]
        top_up = np.argsort(up_diff)[-20:][::-1]
        top_combined = np.argsort(combined)[-20:][::-1]
        
        # Activation gap metric (like qwen3's activation_gap = +3.70)
        # For GLM4: mean absolute gate activation difference across top channels
        act_gap_gate = float(np.mean(gate_diff[top_gate[:10]]))
        act_gap_up = float(np.mean(up_diff[top_up[:10]]))
        act_gap_combined = float(np.mean(combined[top_combined[:10]]))
        
        log(f"  Activation gaps: gate_top10={act_gap_gate:.4f}, up_top10={act_gap_up:.4f}, combined_top10={act_gap_combined:.4f}")
        
        # Top channels details
        log(f"  Top gate ch: {top_gate[:5].tolist()} vals={[float(gate_diff[ch]) for ch in top_gate[:5]]}")
        log(f"  Top combined ch: {top_combined[:5].tolist()} vals={[float(combined[ch]) for ch in top_combined[:5]]}")
        
        # Also compute: do these channels consistently activate across color pairs?
        # Cross-color-pair consistency
        color_pair_consistency = []
        for i in range(len(color_gates)-1):
            for j in range(i+1, len(color_gates)):
                cdiff = np.abs(color_gates[i] - color_gates[j])
                # correlation between this pair's top channels and the global top
                pair_top = set(np.argsort(cdiff)[-10:][::-1])
                global_top = set(top_gate[:10])
                overlap = len(pair_top & global_top)
                color_pair_consistency.append(overlap)
        
        if color_pair_consistency:
            log(f"  Cross-pair channel overlap: mean={np.mean(color_pair_consistency):.1f}/10")
        
        # Color logit analysis
        color_ids = {}
        for c in COLORS:
            tids = tokenizer.encode(c, add_special_tokens=False)
            if tids: color_ids[c] = tids[0]
        
        logits_gap_data = {}
        if color_logits_list and func_logits_list:
            mean_c_logits = np.mean(color_logits_list, axis=0)
            mean_f_logits = np.mean(func_logits_list, axis=0)
            for cname, cid in color_ids.items():
                logits_gap_data[cname] = float(mean_c_logits[cid] - mean_f_logits[cid])
            log(f"  Color logit gaps: {logits_gap_data}")
        
        all_results[layer] = {
            "act_gap_gate": act_gap_gate,
            "act_gap_up": act_gap_up,
            "act_gap_combined": act_gap_combined,
            "top_gate_channels": top_gate[:10].tolist(),
            "top_combined_channels": top_combined[:10].tolist(),
            "cross_pair_overlap": float(np.mean(color_pair_consistency)) if color_pair_consistency else 0,
            "logit_gaps": logits_gap_data,
            "n_color_samples": len(color_gates),
            "n_func_samples": len(func_gates),
        }
    
    # === SUMMARY ===
    log(f"\n{'='*50}")
    log("PHASE 947 GLM4 SUMMARY")
    log(f"{'='*50}")
    
    # Find best layer (highest combined activation gap)
    best_layer = max(all_results.keys(), key=lambda l: all_results[l]["act_gap_combined"])
    best = all_results[best_layer]
    
    log(f"\nBest layer: L{best_layer}")
    log(f"  gate activation gap: {best['act_gap_gate']:.4f}")
    log(f"  combined activation gap: {best['act_gap_combined']:.4f}")
    log(f"  cross-pair channel overlap: {best['cross_pair_overlap']:.1f}/10")
    log(f"  logit gaps: {best['logit_gaps']}")
    
    # Comparison with qwen3
    log(f"\nComparison with qwen3 Phase 944 results:")
    log(f"  qwen3 activation gap (color): +3.70")
    log(f"  GLM4 activation gap (combined, L{best_layer}): {best['act_gap_combined']:.4f}")
    
    # Act ratio
    log(f"\nKey channels for GLM4 color bridge (L{best_layer}):")
    log(f"  gate: {best['top_gate_channels']}")
    log(f"  combined: {best['top_combined_channels']}")
    
    # Save
    RESULT_DIR = Path("results/phase947_glm4_color_bridge")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / "glm4_minimal.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, ensure_ascii=False, indent=2)
    log(f"\nSaved to {out_path}")
    
    release_model(model)
    log("Done!")

if __name__ == "__main__":
    main()
