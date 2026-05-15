"""
Phase 184: Constraint Boundary & Relative Perturbation Geometry
================================================================

★★★ 核心问题 ★★★
Phase 183发现GLM4浅层R=-62(排斥子), 但这可能是因为绝对扰动ε=0.1
在GLM4浅层||h||≈0.10时, 扰动=100%范数, 不是"小扰动"!

★★★ 四个实验 ★★★

Exp1: Relative Perturbation Attractor Test (★最关键★)
  - 使用 RELATIVE 扰动: ε_rel * ||h_l|| 而非绝对 ε
  - ε_rel = 0.01, 0.05, 0.10 (1%, 5%, 10%范数)
  - 这将确定GLM4的"排斥子"是否真实

Exp2: Constraint Boundary Detection
  - 在h_correct和h_incorrect之间线性插值
  - h(α) = (1-α)*h_correct + α*h_incorrect
  - 二分搜索找到α* = 模型预测翻转的临界点
  - α* → 0: 约束弱(微小偏移就翻转)
  - α* → 1: 约束强(需要大幅偏移才翻转)
  - ★ 这是"约束边界"的直接测量

Exp3: Directional Selectivity (Δ方向 vs 正交方向)
  - 沿Δ方向扰动: δ_margin_Δ = margin(h + ε*Δ/||Δ||) - margin(h)
  - 沿正交方向扰动: δ_margin_⊥ = margin(h + ε*v_⊥) - margin(h)
  - 选择性 S = |δ_margin_Δ| / |δ_margin_⊥|
  - S >> 1: Δ方向对margin有特殊影响 → 约束方向真实
  - S ≈ 1: Δ方向不特殊 → Δ可能只是一般性差异

Exp4: Cross-Template Directional Invariance
  - 5个不同句子模板, 相同语法约束(主谓一致)
  - 计算跨模板的cos(Δ^i, Δ^j)
  - 高相关 → 存在"约束不变方向"
  - 低相关 → 无统一约束方向

Usage: python tests/glm5/phase184_constraint_boundary.py <model_name>
"""

import sys, os, time, json, gc
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'glm5'))
from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[P184] Loading {model_name} (bfloat16 + device_map=auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager")
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[P184] {model_name} loaded: device={device}, class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def find_verb_position(tokenizer, sentence, verb_str):
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    verb_ids = tokenizer.encode(" " + verb_str, add_special_tokens=False)
    if not verb_ids:
        verb_ids = tokenizer.encode(verb_str, add_special_tokens=False)
    if len(verb_ids) >= 1:
        for i in range(len(tokens) - len(verb_ids) + 1):
            if all(tokens[i+j] == verb_ids[j] for j in range(len(verb_ids))):
                return i
    for i, tid in enumerate(tokens):
        decoded = tokenizer.decode([tid]).strip().lower()
        if verb_str.lower() in decoded:
            return i
    return min(3, len(tokens) - 2)


def get_hidden_at_pos(model, tokenizer, device, sentence, target_pos):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                    output_hidden_states=True)
    pos = min(target_pos, out.hidden_states[0].shape[1] - 1)
    result = {}
    n_layers = len(out.hidden_states) - 1
    for li, hs in enumerate(out.hidden_states):
        result[li] = hs[0, pos].detach().cpu().float().numpy().astype(np.float32)
    del out
    return result, n_layers


def compute_margin_at_layer(h_vec, W_U, tok_c_id, tok_i_id):
    """计算单个hidden state的2-token margin"""
    return float(W_U[tok_c_id] @ h_vec - W_U[tok_i_id] @ h_vec)


# =====================================================================
# SENTENCE PAIRS — 大数据量
# =====================================================================

# ★ 反事实对 (same target token, different constraint) — 40对
CF_SINGULAR = [
    ("The cat sleeps", "The cats sleeps", "sleeps"),
    ("The dog runs", "The dogs runs", "runs"),
    ("The bird sings", "The birds sings", "sings"),
    ("The child plays", "The children plays", "plays"),
    ("The sister reads", "The sisters reads", "reads"),
    ("The flower grows", "The flowers grows", "grows"),
    ("The river flows", "The rivers flows", "flows"),
    ("The mother cooks", "The mothers cooks", "cooks"),
    ("The student writes", "The students writes", "writes"),
    ("She walks", "They walks", "walks"),
    ("The horse gallops", "The horses gallops", "gallops"),
    ("The wind blows", "The winds blows", "blows"),
    ("The rabbit hops", "The rabbits hops", "hops"),
    ("He drives", "They drives", "drives"),
    ("The girl dances", "The girls dances", "dances"),
    ("The boat sails", "The boats sails", "sails"),
    ("The plane flies", "The planes flies", "flies"),
    ("The teacher speaks", "The teachers speaks", "speaks"),
    ("The doctor works", "The doctors works", "works"),
    ("The baby cries", "The babies cries", "cries"),
    ("The rain falls", "The rains falls", "falls"),
    ("The man thinks", "The men thinks", "thinks"),
    ("The woman walks", "The women walks", "walks"),
    ("The boy jumps", "The boys jumps", "jumps"),
    ("The tree grows", "The trees grows", "grows"),
    ("The car drives", "The cars drives", "drives"),
    ("The house stands", "The houses stands", "stands"),
    ("The dog barks", "The dogs barks", "barks"),
    ("The cat purrs", "The cats purrs", "purrs"),
    ("The bell rings", "The bells rings", "rings"),
    ("The light shines", "The lights shines", "shines"),
    ("The wheel turns", "The wheels turns", "turns"),
    ("The bird flies", "The birds flies", "flies"),
    ("The clock ticks", "The clocks ticks", "ticks"),
    ("The door opens", "The doors opens", "opens"),
    ("The fire burns", "The fires burns", "burns"),
    ("The star shines", "The stars shines", "shines"),
    ("The song plays", "The songs plays", "plays"),
    ("The story ends", "The stories ends", "ends"),
    ("The page turns", "The pages turns", "turns"),
]

CF_PLURAL = [
    ("The cats sleep", "The cat sleep", "sleep"),
    ("The dogs run", "The dog run", "run"),
    ("The birds sing", "The bird sing", "sing"),
    ("The children play", "The child play", "play"),
    ("The students read", "The student read", "read"),
    ("The flowers grow", "The flower grow", "grow"),
    ("The rivers flow", "The river flow", "flow"),
    ("The mothers cook", "The mother cook", "cook"),
    ("The teachers speak", "The teacher speak", "speak"),
    ("The doctors work", "The doctor work", "work"),
    ("The horses gallop", "The horse gallop", "gallop"),
    ("The dogs bark", "The dog bark", "bark"),
    ("The boys jump", "The boy jump", "jump"),
    ("The girls dance", "The girl dance", "dance"),
    ("The planes fly", "The plane fly", "fly"),
    ("The boats sail", "The boat sail", "sail"),
    ("The bells ring", "The bell ring", "ring"),
    ("The wheels turn", "The wheel turn", "turn"),
    ("The fires burn", "The fire burn", "burn"),
    ("The stars shine", "The star shine", "shine"),
]

# ★ 跨模板句子 — 5个模板 × 8个动词 = 40对
# 模板1: "The {noun} {verb_s}" vs "The {noun}s {verb_s}"
# 模板2: "A {noun} {verb_s}" vs "Some {noun}s {verb_s}"
# 模板3: "This {noun} {verb_s}" vs "These {noun}s {verb_s}"
# 模板4: "My {noun} {verb_s}" vs "My {noun}s {verb_s}"  (注意: "My"不区分单复数)
# 模板5: "One {noun} {verb_s}" vs "Two {noun}s {verb_s}"

CROSS_TEMPLATE_VERBS = ["sleeps", "runs", "sings", "plays", "reads", "grows", "flows", "cooks"]
CROSS_TEMPLATE_NOUNS_S = ["cat", "dog", "bird", "child", "student", "flower", "river", "mother"]

CROSS_TEMPLATE_PAIRS = []  # (template_id, sent_correct, sent_incorrect, verb)
for vi, (verb_s, noun_s) in enumerate(zip(CROSS_TEMPLATE_VERBS, CROSS_TEMPLATE_NOUNS_S)):
    noun_p = noun_s + "s" if not noun_s.endswith("s") else noun_s + "es"
    verb_b = verb_s.rstrip("s") if verb_s.endswith("s") else verb_s
    
    # Template 1: The
    CROSS_TEMPLATE_PAIRS.append((1, f"The {noun_s} {verb_s}", f"The {noun_p} {verb_s}", verb_s))
    # Template 2: A/Some
    CROSS_TEMPLATE_PAIRS.append((2, f"A {noun_s} {verb_s}", f"Some {noun_p} {verb_s}", verb_s))
    # Template 3: This/These
    CROSS_TEMPLATE_PAIRS.append((3, f"This {noun_s} {verb_s}", f"These {noun_p} {verb_s}", verb_s))
    # Template 4: My (same determiner, no number cue)
    CROSS_TEMPLATE_PAIRS.append((4, f"My {noun_s} {verb_s}", f"My {noun_p} {verb_s}", verb_s))
    # Template 5: One/Two
    CROSS_TEMPLATE_PAIRS.append((5, f"One {noun_s} {verb_s}", f"Two {noun_p} {verb_s}", verb_s))


# =====================================================================
# EXP1: RELATIVE PERTURBATION ATTRACTOR TEST
# =====================================================================

def exp1_relative_perturbation(model, tokenizer, device, n_layers, d_model, W_U):
    """
    ★★★ 最关键实验: 用相对扰动修正GLM4排斥子问题 ★★★
    
    Phase 183用绝对扰动ε=0.1, 对GLM4浅层(||h||≈0.10)等于100%范数
    本实验用 ε_rel * ||h_l|| 替代绝对 ε
    
    关键指标:
    - R_rel(ε_rel, l) = margin_perturbed / margin_clean
    - R_rel → 1: 强吸引子 (扰动被修正)
    - R_rel → 0: 弱吸引子
    - R_rel < 0: 排斥子
    """
    print("\n" + "="*60)
    print("Exp1: RELATIVE PERTURBATION ATTRACTOR TEST")
    print("  (Fix GLM4 repeller: use ε_rel * ||h|| instead of absolute ε)")
    print("="*60)
    
    W_U_f32 = W_U.astype(np.float32)
    eps_rels = [0.01, 0.05, 0.10]  # 1%, 5%, 10% of ||h||
    test_layers = sorted(set([1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25,
                               n_layers//2, n_layers-5, n_layers-2, n_layers-1]))
    test_layers = [l for l in test_layers if 1 <= l < n_layers]
    
    # Use 40 counterfactual pairs for robust statistics
    test_pairs = CF_SINGULAR[:20] + CF_PLURAL[:20]
    print(f"  Testing {len(test_pairs)} pairs × {len(test_layers)} layers × {len(eps_rels)} ε_rels")
    
    all_results = {}
    
    for eps_rel in eps_rels:
        print(f"\n  [ε_rel={eps_rel}] Running...", flush=True)
        recovery_rates = defaultdict(list)
        
        for pi, (sent_c, sent_i, verb) in enumerate(test_pairs):
            if pi % 5 == 0:
                print(f"    Pair {pi+1}/{len(test_pairs)}", flush=True)
            
            # Get token IDs
            tok_c_ids = tokenizer.encode(verb, add_special_tokens=False)
            verb_alt = verb.rstrip("s") if verb.endswith("s") else verb + "s"
            tok_i_ids = tokenizer.encode(verb_alt, add_special_tokens=False)
            if not tok_c_ids or not tok_i_ids:
                continue
            
            pos_c = find_verb_position(tokenizer, sent_c, verb)
            pos_i = find_verb_position(tokenizer, sent_i, verb)
            
            # Get clean hidden states
            hs_c, _ = get_hidden_at_pos(model, tokenizer, device, sent_c, pos_c)
            hs_i, _ = get_hidden_at_pos(model, tokenizer, device, sent_i, pos_i)
            
            # Clean margin from final layer
            clean_margin = compute_margin_at_layer(hs_c[n_layers], W_U_f32, tok_c_ids[0], tok_i_ids[0])
            if abs(clean_margin) < 1e-10:
                del hs_c, hs_i
                continue
            
            # Test perturbation at each layer
            layers = get_layers(model)
            inputs_c = tokenizer(sent_c, return_tensors="pt", truncation=True, max_length=128)
            input_ids_c = inputs_c["input_ids"].to(device)
            attn_mask_c = inputs_c["attention_mask"].to(device)
            
            for patch_li in test_layers:
                if patch_li not in hs_c or patch_li not in hs_i:
                    continue
                
                delta_l = hs_c[patch_li] - hs_i[patch_li]
                delta_norm = float(np.linalg.norm(delta_l))
                h_norm = float(np.linalg.norm(hs_c[patch_li]))
                
                if delta_norm < 1e-10 or h_norm < 1e-10:
                    continue
                
                # ★ KEY: Relative perturbation = ε_rel * ||h_l||
                # Direction: move h_correct toward h_incorrect
                perturbation = eps_rel * h_norm * delta_l / delta_norm
                
                hook_handle = None
                
                def make_perturb_hook(perturb_vec, layer_idx, target_pos):
                    perturb_tensor = torch.tensor(perturb_vec, dtype=torch.bfloat16, device=device)
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            new_out = output[0].detach().clone()
                            pos = min(target_pos, new_out.shape[1] - 1)
                            new_out[0, pos] -= perturb_tensor.to(new_out.device)
                            return (new_out,) + output[1:]
                        return output
                    return hook_fn
                
                try:
                    hook_handle = layers[patch_li].register_forward_hook(
                        make_perturb_hook(perturbation, patch_li, pos_c))
                    
                    with torch.no_grad():
                        out_p = model(input_ids=input_ids_c, attention_mask=attn_mask_c)
                    
                    perturbed_margin = float(
                        out_p.logits[0, -1, tok_c_ids[0]] - out_p.logits[0, -1, tok_i_ids[0]])
                    
                    del out_p
                    hook_handle.remove()
                    
                    # Recovery rate
                    recovery = perturbed_margin / clean_margin if abs(clean_margin) > 1e-10 else 0.0
                    recovery_rates[patch_li].append({
                        "R": recovery,
                        "h_norm": h_norm,
                        "delta_norm": delta_norm,
                        "abs_perturb": eps_rel * h_norm,
                    })
                    
                except Exception as e:
                    if hook_handle:
                        hook_handle.remove()
                    recovery_rates[patch_li].append({
                        "R": 0.0, "h_norm": h_norm, "delta_norm": delta_norm,
                        "abs_perturb": eps_rel * h_norm, "error": str(e)[:50]
                    })
                finally:
                    if hook_handle:
                        try: hook_handle.remove()
                        except: pass
            
            del hs_c, hs_i, inputs_c
            force_cleanup()
        
        # Aggregate
        eps_result = {}
        for li in sorted(recovery_rates.keys()):
            rates = [r["R"] for r in recovery_rates[li]]
            h_norms = [r["h_norm"] for r in recovery_rates[li]]
            eps_result[li] = {
                "recovery_mean": float(np.mean(rates)),
                "recovery_std": float(np.std(rates)),
                "recovery_median": float(np.median(rates)),
                "n_pairs": len(rates),
                "is_attractor": float(np.mean(rates)) > 0.5,
                "h_norm_mean": float(np.mean(h_norms)),
                "abs_perturb_mean": float(np.mean([r["abs_perturb"] for r in recovery_rates[li]])),
            }
        all_results[f"eps_rel_{eps_rel}"] = eps_result
    
    del W_U_f32
    return all_results


# =====================================================================
# EXP2: CONSTRAINT BOUNDARY DETECTION
# =====================================================================

def exp2_constraint_boundary(model, tokenizer, device, n_layers, d_model, W_U):
    """
    ★ 约束边界探测
    
    在h_correct和h_incorrect之间线性插值:
    h(α) = (1-α)*h_correct + α*h_incorrect
    
    二分搜索α*使得模型预测翻转
    α* = 约束边界位置
    
    - α* → 0: 约束弱 (微小偏移就翻转)
    - α* → 0.5: 中等约束 (对称边界)
    - α* → 1: 约束强 (需要大幅偏移才翻转)
    """
    print("\n" + "="*60)
    print("Exp2: CONSTRAINT BOUNDARY DETECTION")
    print("  (Binary search for prediction flip point α*)")
    print("="*60)
    
    W_U_f32 = W_U.astype(np.float32)
    test_layers = sorted(set([0, 1, 2, 3, 5, 10, 15, 20, 25,
                               n_layers//2, n_layers-5, n_layers-2, n_layers-1]))
    test_layers = [l for l in test_layers if 0 <= l <= n_layers]
    
    test_pairs = CF_SINGULAR[:15] + CF_PLURAL[:15]
    print(f"  Testing {len(test_pairs)} pairs × {len(test_layers)} layers")
    
    boundary_results = defaultdict(list)  # {layer: [α* values]}
    
    for pi, (sent_c, sent_i, verb) in enumerate(test_pairs):
        if pi % 5 == 0:
            print(f"    Pair {pi+1}/{len(test_pairs)}", flush=True)
        
        tok_c_ids = tokenizer.encode(verb, add_special_tokens=False)
        verb_alt = verb.rstrip("s") if verb.endswith("s") else verb + "s"
        tok_i_ids = tokenizer.encode(verb_alt, add_special_tokens=False)
        if not tok_c_ids or not tok_i_ids:
            continue
        
        pos_c = find_verb_position(tokenizer, sent_c, verb)
        pos_i = find_verb_position(tokenizer, sent_i, verb)
        
        # Get hidden states for both sentences
        hs_c, _ = get_hidden_at_pos(model, tokenizer, device, sent_c, pos_c)
        hs_i, _ = get_hidden_at_pos(model, tokenizer, device, sent_i, pos_i)
        
        # For each test layer, do binary search using FORWARD PASSES
        # (not just linear interpolation in hidden space, but actual model forward)
        layers = get_layers(model)
        
        for patch_li in test_layers:
            if patch_li not in hs_c or patch_li not in hs_i:
                continue
            
            h_c = hs_c[patch_li]
            h_i = hs_i[patch_li]
            delta = h_i - h_c  # Direction from correct to incorrect
            
            # Clean margin (α=0)
            margin_clean = compute_margin_at_layer(hs_c[n_layers], W_U_f32, tok_c_ids[0], tok_i_ids[0])
            if abs(margin_clean) < 1e-10:
                continue
            
            # Binary search: find α* where prediction flips
            alpha_lo, alpha_hi = 0.0, 1.0
            
            # Check endpoint: at α=1, prediction should be wrong
            margin_alpha1 = compute_margin_at_layer(
                hs_c[n_layers] + (hs_i[n_layers] - hs_c[n_layers]),
                W_U_f32, tok_c_ids[0], tok_i_ids[0])
            
            # If even at α=1 the margin doesn't flip, skip
            if margin_clean * margin_alpha1 > 0:
                boundary_results[patch_li].append(1.0)  # Boundary beyond α=1
                continue
            
            # Binary search in hidden space at the FINAL layer
            # (This is the simplest approach: interpolate final hidden states)
            for _ in range(20):  # 20 iterations = precision of ~1e-6
                alpha_mid = (alpha_lo + alpha_hi) / 2
                h_mid = hs_c[n_layers] + alpha_mid * (hs_i[n_layers] - hs_c[n_layers])
                margin_mid = compute_margin_at_layer(h_mid, W_U_f32, tok_c_ids[0], tok_i_ids[0])
                
                if margin_clean * margin_mid > 0:
                    alpha_lo = alpha_mid
                else:
                    alpha_hi = alpha_mid
            
            boundary_results[patch_li].append(alpha_mid)
        
        del hs_c, hs_i
        force_cleanup()
    
    # Also do forward-pass binary search for a subset of layers
    # (This tests the ACTUAL model dynamics, not just final-layer interpolation)
    print("\n  [B] Forward-pass binary search (subset of layers)...")
    fw_boundary_results = defaultdict(list)
    fw_test_layers = test_layers[:5]  # Only first 5 layers for speed
    fw_test_pairs = CF_SINGULAR[:10]
    
    for pi, (sent_c, sent_i, verb) in enumerate(fw_test_pairs):
        if pi % 3 == 0:
            print(f"    Pair {pi+1}/{len(fw_test_pairs)}", flush=True)
        
        tok_c_ids = tokenizer.encode(verb, add_special_tokens=False)
        verb_alt = verb.rstrip("s") if verb.endswith("s") else verb + "s"
        tok_i_ids = tokenizer.encode(verb_alt, add_special_tokens=False)
        if not tok_c_ids or not tok_i_ids:
            continue
        
        pos_c = find_verb_position(tokenizer, sent_c, verb)
        pos_i = find_verb_position(tokenizer, sent_i, verb)
        
        hs_c, _ = get_hidden_at_pos(model, tokenizer, device, sent_c, pos_c)
        hs_i, _ = get_hidden_at_pos(model, tokenizer, device, sent_i, pos_i)
        
        layers = get_layers(model)
        inputs_c = tokenizer(sent_c, return_tensors="pt", truncation=True, max_length=128)
        input_ids_c = inputs_c["input_ids"].to(device)
        attn_mask_c = inputs_c["attention_mask"].to(device)
        
        # Clean margin
        clean_logits = None
        with torch.no_grad():
            out_clean = model(input_ids=input_ids_c, attention_mask=attn_mask_c)
            clean_logits = out_clean.logits[0, -1].detach().cpu().float().numpy()
        del out_clean
        
        clean_margin = float(clean_logits[tok_c_ids[0]] - clean_logits[tok_i_ids[0]])
        if abs(clean_margin) < 1e-10:
            del hs_c, hs_i, inputs_c, clean_logits
            continue
        
        for patch_li in fw_test_layers:
            if patch_li not in hs_c or patch_li not in hs_i:
                continue
            
            delta_l = hs_i[patch_li] - hs_c[patch_li]
            delta_norm = float(np.linalg.norm(delta_l))
            h_norm = float(np.linalg.norm(hs_c[patch_li]))
            if delta_norm < 1e-10 or h_norm < 1e-10:
                continue
            
            # Binary search via forward pass
            alpha_lo, alpha_hi = 0.0, 1.0
            
            for _ in range(12):  # 12 iterations (balance speed and precision)
                alpha_mid = (alpha_lo + alpha_hi) / 2
                
                # Interpolate: h_mid = h_correct + α * (h_incorrect - h_correct)
                interp_perturb = alpha_mid * delta_l  # numpy
                
                hook_handle = None
                
                def make_interp_hook(perturb_vec, target_pos):
                    perturb_tensor = torch.tensor(perturb_vec, dtype=torch.bfloat16, device=device)
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            new_out = output[0].detach().clone()
                            pos = min(target_pos, new_out.shape[1] - 1)
                            new_out[0, pos] += perturb_tensor.to(new_out.device)
                            return (new_out,) + output[1:]
                        return output
                    return hook_fn
                
                try:
                    hook_handle = layers[patch_li].register_forward_hook(
                        make_interp_hook(interp_perturb, patch_li, pos_c))
                    
                    with torch.no_grad():
                        out_p = model(input_ids=input_ids_c, attention_mask=attn_mask_c)
                    
                    perturbed_margin = float(
                        out_p.logits[0, -1, tok_c_ids[0]] - out_p.logits[0, -1, tok_i_ids[0]])
                    
                    del out_p
                    hook_handle.remove()
                    
                    if clean_margin * perturbed_margin > 0:
                        alpha_lo = alpha_mid
                    else:
                        alpha_hi = alpha_mid
                    
                except Exception as e:
                    if hook_handle:
                        hook_handle.remove()
                    break
                finally:
                    if hook_handle:
                        try: hook_handle.remove()
                        except: pass
            
            fw_boundary_results[patch_li].append(alpha_mid)
        
        del hs_c, hs_i, inputs_c, clean_logits
        force_cleanup()
    
    # Aggregate
    result = {}
    for li in sorted(boundary_results.keys()):
        alphas = boundary_results[li]
        entry = {
            "alpha_star_mean": float(np.mean(alphas)),
            "alpha_star_std": float(np.std(alphas)),
            "alpha_star_median": float(np.median(alphas)),
            "n_pairs": len(alphas),
        }
        if li in fw_boundary_results:
            fw_alphas = fw_boundary_results[li]
            entry["fw_alpha_star_mean"] = float(np.mean(fw_alphas))
            entry["fw_alpha_star_std"] = float(np.std(fw_alphas))
        result[li] = entry
    
    del W_U_f32
    return result


# =====================================================================
# EXP3: DIRECTIONAL SELECTIVITY (Δ vs orthogonal)
# =====================================================================

def exp3_directional_selectivity(model, tokenizer, device, n_layers, d_model, W_U):
    """
    ★ 方向选择性: Δ方向 vs 正交方向对margin的影响
    
    方法:
    1. 沿Δ方向扰动: h' = h + ε_rel * ||h|| * Δ/||Δ||
    2. 沿正交方向扰动: h'' = h + ε_rel * ||h|| * v_⊥
    3. 测量两种扰动对margin的影响
    
    选择性 S = |δ_margin_Δ| / |δ_margin_⊥|
    - S >> 1: Δ方向对margin有特殊影响 → 约束方向真实
    - S ≈ 1: Δ方向不特殊 → Δ只是一般性差异
    """
    print("\n" + "="*60)
    print("Exp3: DIRECTIONAL SELECTIVITY (Δ vs orthogonal)")
    print("="*60)
    
    W_U_f32 = W_U.astype(np.float32)
    eps_rel = 0.05  # 5% of norm
    test_layers = sorted(set([1, 2, 3, 5, 10, 15, 20, n_layers//2, n_layers-5, n_layers-1]))
    test_layers = [l for l in test_layers if 1 <= l < n_layers]
    
    test_pairs = CF_SINGULAR[:15] + CF_PLURAL[:10]
    print(f"  Testing {len(test_pairs)} pairs × {len(test_layers)} layers")
    
    selectivity_results = defaultdict(list)  # {layer: [S values]}
    
    for pi, (sent_c, sent_i, verb) in enumerate(test_pairs):
        if pi % 5 == 0:
            print(f"    Pair {pi+1}/{len(test_pairs)}", flush=True)
        
        tok_c_ids = tokenizer.encode(verb, add_special_tokens=False)
        verb_alt = verb.rstrip("s") if verb.endswith("s") else verb + "s"
        tok_i_ids = tokenizer.encode(verb_alt, add_special_tokens=False)
        if not tok_c_ids or not tok_i_ids:
            continue
        
        pos_c = find_verb_position(tokenizer, sent_c, verb)
        pos_i = find_verb_position(tokenizer, sent_i, verb)
        
        hs_c, _ = get_hidden_at_pos(model, tokenizer, device, sent_c, pos_c)
        hs_i, _ = get_hidden_at_pos(model, tokenizer, device, sent_i, pos_i)
        
        # Clean margin
        clean_margin = compute_margin_at_layer(hs_c[n_layers], W_U_f32, tok_c_ids[0], tok_i_ids[0])
        if abs(clean_margin) < 1e-10:
            del hs_c, hs_i
            continue
        
        layers = get_layers(model)
        inputs_c = tokenizer(sent_c, return_tensors="pt", truncation=True, max_length=128)
        input_ids_c = inputs_c["input_ids"].to(device)
        attn_mask_c = inputs_c["attention_mask"].to(device)
        
        for patch_li in test_layers:
            if patch_li not in hs_c or patch_li not in hs_i:
                continue
            
            delta_l = hs_c[patch_li] - hs_i[patch_li]
            delta_norm = float(np.linalg.norm(delta_l))
            h_norm = float(np.linalg.norm(hs_c[patch_li]))
            
            if delta_norm < 1e-10 or h_norm < 1e-10:
                continue
            
            # Δ direction (unit vector)
            delta_dir = delta_l / delta_norm
            
            # Orthogonal direction: remove Δ component from a random vector
            rng = np.random.RandomState(42 + pi)
            v_rand = rng.randn(d_model).astype(np.float32)
            v_rand = v_rand - np.dot(v_rand, delta_dir) * delta_dir  # Gram-Schmidt
            v_rand_norm = float(np.linalg.norm(v_rand))
            if v_rand_norm < 1e-10:
                continue
            v_perp = v_rand / v_rand_norm
            
            # Perturbation magnitude
            eps_abs = eps_rel * h_norm
            
            # --- Test Δ direction perturbation ---
            perturb_delta = eps_abs * delta_dir
            margin_delta = None
            hook_handle = None
            
            def make_hook_delta(perturb_vec, target_pos):
                perturb_tensor = torch.tensor(perturb_vec, dtype=torch.bfloat16, device=device)
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        new_out = output[0].detach().clone()
                        pos = min(target_pos, new_out.shape[1] - 1)
                        new_out[0, pos] -= perturb_tensor.to(new_out.device)
                        return (new_out,) + output[1:]
                    return output
                return hook_fn
            
            try:
                hook_handle = layers[patch_li].register_forward_hook(
                    make_hook_delta(perturb_delta, pos_c))
                with torch.no_grad():
                    out_p = model(input_ids=input_ids_c, attention_mask=attn_mask_c)
                margin_delta = float(out_p.logits[0, -1, tok_c_ids[0]] - out_p.logits[0, -1, tok_i_ids[0]])
                del out_p
                hook_handle.remove()
            except:
                if hook_handle: hook_handle.remove()
                continue
            finally:
                if hook_handle:
                    try: hook_handle.remove()
                    except: pass
            
            # --- Test orthogonal direction perturbation ---
            perturb_perp = eps_abs * v_perp
            margin_perp = None
            hook_handle = None
            
            def make_hook_perp(perturb_vec, target_pos):
                perturb_tensor = torch.tensor(perturb_vec, dtype=torch.bfloat16, device=device)
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        new_out = output[0].detach().clone()
                        pos = min(target_pos, new_out.shape[1] - 1)
                        new_out[0, pos] += perturb_tensor.to(new_out.device)
                        return (new_out,) + output[1:]
                    return output
                return hook_fn
            
            try:
                hook_handle = layers[patch_li].register_forward_hook(
                    make_hook_perp(perturb_perp, pos_c))
                with torch.no_grad():
                    out_p = model(input_ids=input_ids_c, attention_mask=attn_mask_c)
                margin_perp = float(out_p.logits[0, -1, tok_c_ids[0]] - out_p.logits[0, -1, tok_i_ids[0]])
                del out_p
                hook_handle.remove()
            except:
                if hook_handle: hook_handle.remove()
                continue
            finally:
                if hook_handle:
                    try: hook_handle.remove()
                    except: pass
            
            # Compute selectivity
            if margin_delta is not None and margin_perp is not None:
                delta_margin_delta = abs(margin_delta - clean_margin)
                delta_margin_perp = abs(margin_perp - clean_margin)
                
                selectivity = delta_margin_delta / max(delta_margin_perp, 1e-10)
                
                selectivity_results[patch_li].append({
                    "S": selectivity,
                    "delta_margin_delta": delta_margin_delta,
                    "delta_margin_perp": delta_margin_perp,
                    "margin_delta": margin_delta,
                    "margin_perp": margin_perp,
                })
        
        del hs_c, hs_i, inputs_c
        force_cleanup()
    
    # Aggregate
    result = {}
    for li in sorted(selectivity_results.keys()):
        S_vals = [r["S"] for r in selectivity_results[li]]
        dmd = [r["delta_margin_delta"] for r in selectivity_results[li]]
        dmp = [r["delta_margin_perp"] for r in selectivity_results[li]]
        result[li] = {
            "selectivity_mean": float(np.mean(S_vals)),
            "selectivity_median": float(np.median(S_vals)),
            "selectivity_std": float(np.std(S_vals)),
            "delta_margin_delta_mean": float(np.mean(dmd)),
            "delta_margin_perp_mean": float(np.mean(dmp)),
            "n_pairs": len(S_vals),
        }
    
    del W_U_f32
    return result


# =====================================================================
# EXP4: CROSS-TEMPLATE DIRECTIONAL INVARIANCE
# =====================================================================

def exp4_cross_template_invariance(model, tokenizer, device, n_layers, d_model):
    """
    ★ 跨模板方向不变性测试
    
    对于相同语法约束(主谓一致), 使用5个不同句子模板:
    1. "The cat sleeps" vs "The cats sleeps"
    2. "A cat sleeps" vs "Some cats sleeps"
    3. "This cat sleeps" vs "These cats sleeps"
    4. "My cat sleeps" vs "My cats sleeps"
    5. "One cat sleeps" vs "Two cats sleeps"
    
    计算不同模板间Δ方向的cos相似度:
    - 高相关(>0.7): 存在"约束不变方向"
    - 低相关(<0.3): 无统一约束方向
    """
    print("\n" + "="*60)
    print("Exp4: CROSS-TEMPLATE DIRECTIONAL INVARIANCE")
    print("  (Test if Δ direction is stable across sentence templates)")
    print("="*60)
    
    # Group pairs by template
    template_deltas = defaultdict(lambda: defaultdict(list))  # {template_id: {layer: [Δ vectors]}}
    
    for ti, (tmpl_id, sent_c, sent_i, verb) in enumerate(CROSS_TEMPLATE_PAIRS):
        if ti % 10 == 0:
            print(f"    Pair {ti+1}/{len(CROSS_TEMPLATE_PAIRS)}", flush=True)
        
        pos_c = find_verb_position(tokenizer, sent_c, verb)
        pos_i = find_verb_position(tokenizer, sent_i, verb)
        
        hs_c, _ = get_hidden_at_pos(model, tokenizer, device, sent_c, pos_c)
        hs_i, _ = get_hidden_at_pos(model, tokenizer, device, sent_i, pos_i)
        
        for li in range(n_layers + 1):
            if li in hs_c and li in hs_i:
                delta = hs_c[li] - hs_i[li]
                delta_norm = float(np.linalg.norm(delta))
                if delta_norm > 1e-10:
                    template_deltas[tmpl_id][li].append(delta / delta_norm)  # Unit vector
        
        del hs_c, hs_i
        force_cleanup()
    
    # Compute cross-template cosine similarity
    result = {}
    key_layers = sorted(set([0, 1, 2, 5, 10, 15, 20, 25, n_layers//2, n_layers-5, n_layers-2, n_layers-1]))
    key_layers = [l for l in key_layers if 0 <= l <= n_layers]
    
    for li in key_layers:
        entry = {}
        
        # Within-template stability: average cos between pairs of same template
        within_cos = []
        for tmpl_id in sorted(template_deltas.keys()):
            deltas = template_deltas[tmpl_id].get(li, [])
            if len(deltas) >= 2:
                for i in range(len(deltas)):
                    for j in range(i+1, min(len(deltas), 5)):  # Limit comparisons
                        cos_val = float(np.dot(deltas[i], deltas[j]))
                        within_cos.append(cos_val)
        
        if within_cos:
            entry["within_template_cos_mean"] = float(np.mean(within_cos))
            entry["within_template_cos_std"] = float(np.std(within_cos))
        
        # Cross-template similarity: cos between Δ from different templates
        cross_cos = []
        tmpl_ids = sorted(template_deltas.keys())
        for ti in range(len(tmpl_ids)):
            for tj in range(ti+1, len(tmpl_ids)):
                deltas_i = template_deltas[tmpl_ids[ti]].get(li, [])
                deltas_j = template_deltas[tmpl_ids[tj]].get(li, [])
                if deltas_i and deltas_j:
                    # Average Δ for each template
                    avg_i = np.mean(deltas_i[:5], axis=0)
                    avg_j = np.mean(deltas_j[:5], axis=0)
                    ni = float(np.linalg.norm(avg_i))
                    nj = float(np.linalg.norm(avg_j))
                    if ni > 1e-10 and nj > 1e-10:
                        cos_val = float(np.dot(avg_i, avg_j) / (ni * nj))
                        cross_cos.append(cos_val)
        
        if cross_cos:
            entry["cross_template_cos_mean"] = float(np.mean(cross_cos))
            entry["cross_template_cos_std"] = float(np.std(cross_cos))
            entry["cross_template_cos_min"] = float(np.min(cross_cos))
            entry["cross_template_cos_max"] = float(np.max(cross_cos))
        
        # Key verdict
        if cross_cos:
            mean_cos = np.mean(cross_cos)
            if mean_cos > 0.7:
                entry["invariance_verdict"] = "STRONG invariant direction"
            elif mean_cos > 0.4:
                entry["invariance_verdict"] = "MODERATE invariant direction"
            elif mean_cos > 0.2:
                entry["invariance_verdict"] = "WEAK invariant direction"
            else:
                entry["invariance_verdict"] = "NO invariant direction"
        
        if entry:
            result[li] = entry
    
    return result


# =====================================================================
# MAIN
# =====================================================================

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    t_start = time.time()
    
    print(f"\n{'#'*70}")
    print(f"# Phase 184: CONSTRAINT BOUNDARY & RELATIVE PERTURBATION — {model_name}")
    print(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers, d_model, vocab_size = info.n_layers, info.d_model, info.vocab_size
    print(f"\nModel: {info.model_class}, Layers={n_layers}, d_model={d_model}, vocab={vocab_size}")
    
    W_U = get_W_U(model, model_name).astype(np.float32)
    print(f"W_U shape: {W_U.shape}, dtype: {W_U.dtype}")
    
    # ===== Exp1: Relative Perturbation (MOST CRITICAL) =====
    print(f"\n{'='*70}")
    print("Running Exp1: Relative Perturbation Attractor Test...")
    print("  ★★★ This fixes the GLM4 repeller question ★★★")
    exp1_results = exp1_relative_perturbation(model, tokenizer, device, n_layers, d_model, W_U)
    force_cleanup()
    
    # ===== Exp2: Constraint Boundary =====
    print(f"\n{'='*70}")
    print("Running Exp2: Constraint Boundary Detection...")
    exp2_results = exp2_constraint_boundary(model, tokenizer, device, n_layers, d_model, W_U)
    force_cleanup()
    
    # ===== Exp3: Directional Selectivity =====
    print(f"\n{'='*70}")
    print("Running Exp3: Directional Selectivity (Δ vs orthogonal)...")
    exp3_results = exp3_directional_selectivity(model, tokenizer, device, n_layers, d_model, W_U)
    force_cleanup()
    
    # ===== Exp4: Cross-Template Invariance =====
    print(f"\n{'='*70}")
    print("Running Exp4: Cross-Template Directional Invariance...")
    exp4_results = exp4_cross_template_invariance(model, tokenizer, device, n_layers, d_model)
    force_cleanup()
    
    # ===== Save =====
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = f"tests/glm5_temp/phase184_{model_name}_{timestamp}.json"
    
    full_results = {
        "model": model_name, "n_layers": n_layers, "d_model": d_model, "vocab_size": vocab_size,
        "timestamp": timestamp, "elapsed_sec": round(time.time() - t_start, 1),
        "exp1_relative_perturbation": exp1_results,
        "exp2_constraint_boundary": {str(k): v for k, v in exp2_results.items()},
        "exp3_directional_selectivity": {str(k): v for k, v in exp3_results.items()},
        "exp4_cross_template_invariance": {str(k): v for k, v in exp4_results.items()},
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    
    # ===== Summary =====
    print(f"\n{'#'*70}")
    print("PHASE 184 SUMMARY")
    print(f"{'#'*70}")
    
    # Exp1 Summary
    print("\n★★★ Exp1: Relative Perturbation Attractor Test ★★★")
    for eps_key in sorted(exp1_results.keys()):
        eps_data = exp1_results[eps_key]
        eps_val = eps_key.replace("eps_rel_", "")
        
        attractor_layers = []
        non_attractor_layers = []
        repeller_layers = []
        
        for li_str in sorted(eps_data.keys(), key=lambda x: int(x)):
            ld = eps_data[li_str]
            rm = ld.get("recovery_mean", 0)
            if rm > 0.5:
                attractor_layers.append(int(li_str))
            elif rm > 0:
                non_attractor_layers.append(int(li_str))
            else:
                repeller_layers.append(int(li_str))
        
        print(f"\n  ε_rel={eps_val}:")
        print(f"    Attractor layers (R>0.5): {sorted(attractor_layers)}")
        print(f"    Non-attractor layers (0<R≤0.5): {sorted(non_attractor_layers)}")
        print(f"    Repeller layers (R≤0): {sorted(repeller_layers)}")
        
        # Print key layers
        for li in [1, 2, 3, 5, 10, n_layers//2, n_layers-1]:
            li_str = str(li)
            if li_str in eps_data:
                ld = eps_data[li_str]
                rm = ld.get("recovery_mean", 0)
                hnorm = ld.get("h_norm_mean", 0)
                tag = "★ATTRACTOR" if rm > 0.5 else "NO-ATTRACTOR" if rm > 0 else "★REPELLER"
                print(f"    L{li}: R_rel={rm:.3f} ||h||={hnorm:.2f} [{tag}]")
    
    # Key comparison: was GLM4 repeller real?
    print("\n  ★★★ GLM4 Repeller Verdict ★★★")
    eps_001 = exp1_results.get("eps_rel_0.01", {})
    all_R = [eps_001[str(li)]["recovery_mean"] for li in range(1, n_layers) if str(li) in eps_001]
    if all_R:
        print(f"    ε_rel=0.01: mean(R)={np.mean(all_R):.3f}, min(R)={np.min(all_R):.3f}")
        if np.min(all_R) > 0:
            print(f"    ★ ALL layers R>0 with relative perturbation → GLM4 repeller was ARTIFACT!")
        else:
            print(f"    ★ Some layers still R<0 → GLM4 repeller may be REAL")
    
    # Exp2 Summary
    print("\n★★★ Exp2: Constraint Boundary ★★★")
    for li in [0, 1, 5, 10, n_layers//2, n_layers-1, n_layers]:
        li_str = str(li)
        if li_str in exp2_results:
            d = exp2_results[li_str]
            alpha = d.get("alpha_star_mean", None)
            alpha_std = d.get("alpha_star_std", 0)
            fw_alpha = d.get("fw_alpha_star_mean", None)
            print(f"  L{li}: α*={alpha:.3f}±{alpha_std:.3f}" if alpha is not None else f"  L{li}: N/A", end="")
            if fw_alpha is not None:
                print(f" (forward: {fw_alpha:.3f})")
            else:
                print()
    
    # Exp3 Summary
    print("\n★★★ Exp3: Directional Selectivity ★★★")
    for li in [1, 2, 5, 10, n_layers//2, n_layers-1]:
        li_str = str(li)
        if li_str in exp3_results:
            d = exp3_results[li_str]
            S = d.get("selectivity_mean", 0)
            dmd = d.get("delta_margin_delta_mean", 0)
            dmp = d.get("delta_margin_perp_mean", 0)
            tag = "HIGH selectivity" if S > 2.0 else "MODERATE" if S > 1.2 else "NO selectivity"
            print(f"  L{li}: S={S:.2f} (δ_margin_Δ={dmd:.4f}, δ_margin_⊥={dmp:.4f}) [{tag}]")
    
    # Exp4 Summary
    print("\n★★★ Exp4: Cross-Template Invariance ★★★")
    for li in [0, 1, 5, 10, n_layers//2, n_layers-1, n_layers]:
        li_str = str(li)
        if li_str in exp4_results:
            d = exp4_results[li_str]
            within = d.get("within_template_cos_mean", None)
            cross = d.get("cross_template_cos_mean", None)
            verdict = d.get("invariance_verdict", "N/A")
            print(f"  L{li}: within_cos={within:.3f}" if within is not None else f"  L{li}: N/A", end="")
            print(f", cross_cos={cross:.3f}" if cross is not None else ", cross=N/A", end="")
            print(f" [{verdict}]")
    
    release_model(model)
    elapsed = time.time() - t_start
    print(f"\n{'#'*70}")
    print(f"Phase 184 COMPLETE! Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
