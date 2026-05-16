"""
Phase 185: Transport Topology & Jacobian Spectrum
==================================================

★★★ 核心问题 ★★★
Phase 184发现了约束边界和两种编码哲学(attractor vs barrier), 但:
1. 边界形成过程未知: α*(l)如何在层间逐层变化?
2. 排斥子的根源未明: Jacobian谱半径λ_max是否>1?
3. 边界可能是decoder-induced: 需要证明边界存在于内部表示空间
4. 不同约束类型是否共享同一拓扑?

★★★ 四个实验 ★★★

Exp1: Per-Layer Boundary Formation (★最关键★)
  - 在每个采样层做前向传播二分搜索
  - α*(l)轨迹显示约束边界如何在层间形成
  - 若α*(浅层)<<α*(深层) → 边界是内部形成的，不是decoder-induced
  - 若α*(l)≈常数 → 边界可能只是最终层的分类面

Exp2: Jacobian Directional Amplification (★机制核心★)
  - 在层l添加扰动ε*Δ_hat, 测量对层l+1的影响
  - g_Δ(l) = ||J_l · Δ_hat|| = 扰动沿约束方向的放大率
  - g_perp(l) = ||J_l · v_perp|| = 正交方向的放大率
  - g_Δ > 1 → barrier (扰动被放大)
  - g_Δ < 1 → attractor (扰动被衰减)
  - ★ 也做power iteration估计λ_max(J_l^T J_l)

Exp3: Multi-Step Propagation Profile
  - 在层l注入扰动, 测量在所有后续层的传播
  - ||δh_k|| vs k 的曲线显示全局传播动力学
  - 与Exp2的局部Jacobian互补: 局部+全局

Exp4: Constraint Type Comparison
  - 语法约束(S-V agreement) vs 语义约束(selectional restriction) vs 事实约束
  - 不同约束类型是否有不同的边界拓扑和Jacobian谱?

Usage: python tests/glm5/phase185_transport_topology.py <model_name>
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
    print(f"[P185] Loading {model_name} (bfloat16 + device_map=auto)...")
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
    print(f"[P185] {model_name} loaded: device={device}, class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
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


def get_all_hidden_states(model, tokenizer, device, sentence, target_pos):
    """获取所有层的hidden states"""
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


def compute_margin(h_vec, W_U, tok_c_id, tok_i_id):
    return float(W_U[tok_c_id] @ h_vec - W_U[tok_i_id] @ h_vec)


# =====================================================================
# SENTENCE PAIRS
# =====================================================================

# ★ 语法约束: 主谓一致 (syntactic)
SYNTACTIC_SINGULAR = [
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
]

SYNTACTIC_PLURAL = [
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
]

# ★ 语义约束: 选择限制 (selectional restriction)
# 动词需要animate主语, inanimate主语是语义异常
SEMANTIC_PAIRS = [
    ("The cat sleeps", "The rock sleeps", "sleeps"),     # animate vs inanimate
    ("The dog barks", "The stone barks", "barks"),
    ("The man thinks", "The table thinks", "thinks"),
    ("The bird flies", "The chair flies", "flies"),
    ("The baby cries", "The cloud cries", "cries"),
    ("The woman reads", "The mountain reads", "reads"),
    ("The teacher speaks", "The wall speaks", "speaks"),
    ("The doctor works", "The river works", "works"),
    ("The child runs", "The building runs", "runs"),
    ("The student writes", "The tree writes", "writes"),
]

# ★ 事实约束: 世界知识 (world knowledge)
# correct vs incorrect factual statement
FACTUAL_PAIRS = [
    ("Paris is in France", "Paris is in Germany", "France", "Germany"),
    ("The sun is a star", "The sun is a planet", "star", "planet"),
    ("Dogs are mammals", "Dogs are reptiles", "mammals", "reptiles"),
    ("Water freezes at zero", "Water freezes at hundred", "zero", "hundred"),
    ("Two plus two equals four", "Two plus two equals five", "four", "five"),
    ("Birds can fly", "Birds can swim", "fly", "swim"),
    ("The earth is round", "The earth is flat", "round", "flat"),
    ("Fish live in water", "Fish live in fire", "water", "fire"),
]


# =====================================================================
# EXP1: PER-LAYER BOUNDARY FORMATION
# =====================================================================

def exp1_per_layer_boundary(model, tokenizer, device, n_layers, d_model, W_U):
    """
    ★★★ 最关键实验: 逐层约束边界形成 ★★★
    
    在每个采样层做前向传播二分搜索:
    h_l(α) = h_l^correct + α * (h_l^incorrect - h_l^correct)
    
    找到α*(l) = 预测翻转的临界点
    
    关键预期:
    - α*(浅层) ≈ 0.3-0.5 → 约束弱, 容易翻转
    - α*(深层) ≈ 0.9+ → 约束强, 需要大幅偏移才翻转
    - 若α*(l)单调递增 → 约束在层间逐步形成 (内部形成, 非decoder-induced)
    - 若α*(l)≈常数 → 约束可能只是最终层的分类面
    """
    print("\n" + "="*60)
    print("Exp1: PER-LAYER BOUNDARY FORMATION")
    print("  (Binary search α*(l) at every sample layer)")
    print("="*60)
    
    W_U_f32 = W_U.astype(np.float32)
    # Sample layers: evenly spaced + key positions
    n_sample = min(12, n_layers)
    test_layers = sorted(set(
        [0, 1, 2] +
        list(range(0, n_layers, max(1, n_layers // n_sample))) +
        [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
    ))
    test_layers = [l for l in test_layers if 0 <= l < n_layers]
    test_layers = sorted(set(test_layers))
    
    # Use syntactic pairs (20 for robust statistics)
    test_pairs = SYNTACTIC_SINGULAR[:15] + SYNTACTIC_PLURAL[:5]
    print(f"  Testing {len(test_pairs)} pairs × {len(test_layers)} layers × 10 bisection steps")
    
    boundary_by_layer = defaultdict(list)  # {layer: [α* values]}
    
    for pi, (sent_c, sent_i, verb) in enumerate(test_pairs):
        if pi % 3 == 0:
            print(f"    Pair {pi+1}/{len(test_pairs)}: '{sent_c}' vs '{sent_i}'", flush=True)
        
        # Get token IDs
        tok_c_ids = tokenizer.encode(verb, add_special_tokens=False)
        verb_alt = verb.rstrip("s") if verb.endswith("s") else verb + "s"
        tok_i_ids = tokenizer.encode(verb_alt, add_special_tokens=False)
        if not tok_c_ids or not tok_i_ids:
            continue
        
        pos_c = find_verb_position(tokenizer, sent_c, verb)
        pos_i = find_verb_position(tokenizer, sent_i, verb)
        
        # Get clean hidden states
        hs_c, _ = get_all_hidden_states(model, tokenizer, device, sent_c, pos_c)
        hs_i, _ = get_all_hidden_states(model, tokenizer, device, sent_i, pos_i)
        
        # Clean margin (for reference)
        clean_margin = compute_margin(hs_c[n_layers], W_U_f32, tok_c_ids[0], tok_i_ids[0])
        if abs(clean_margin) < 1e-10:
            del hs_c, hs_i
            continue
        
        # Get layer list for hooks
        layers = get_layers(model)
        inputs_c = tokenizer(sent_c, return_tensors="pt", truncation=True, max_length=128)
        input_ids_c = inputs_c["input_ids"].to(device)
        attn_mask_c = inputs_c["attention_mask"].to(device)
        
        # Per-layer binary search
        for patch_li in test_layers:
            if patch_li not in hs_c or patch_li not in hs_i:
                continue
            
            delta_l = hs_i[patch_li] - hs_c[patch_li]  # Direction from correct to incorrect
            delta_norm = float(np.linalg.norm(delta_l))
            if delta_norm < 1e-10:
                continue
            
            alpha_lo, alpha_hi = 0.0, 1.0
            
            # Check endpoint: at α=1, prediction should be wrong
            hook_handle = None
            try:
                perturb_full = delta_l.copy()  # α=1 perturbation
                perturb_tensor = torch.tensor(perturb_full, dtype=torch.bfloat16, device=device)
                
                def make_hook(pvec, tpos):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            new_out = output[0].detach().clone()
                            p = min(tpos, new_out.shape[1] - 1)
                            new_out[0, p] += pvec.to(new_out.device)
                            return (new_out,) + output[1:]
                        return output
                    return hook_fn
                
                hook_handle = layers[patch_li].register_forward_hook(
                    make_hook(perturb_tensor, pos_c))
                
                with torch.no_grad():
                    out_p = model(input_ids=input_ids_c, attention_mask=attn_mask_c)
                margin_alpha1 = float(
                    out_p.logits[0, -1, tok_c_ids[0]] - out_p.logits[0, -1, tok_i_ids[0]])
                del out_p
                hook_handle.remove()
                hook_handle = None
                
                # If even at α=1 the margin doesn't flip, boundary is beyond α=1
                if clean_margin * margin_alpha1 > 0:
                    boundary_by_layer[patch_li].append(1.0)
                    continue
                
            except Exception as e:
                if hook_handle:
                    hook_handle.remove()
                boundary_by_layer[patch_li].append(-1.0)  # Error
                continue
            
            # Binary search
            for bisect_step in range(10):
                alpha_mid = (alpha_lo + alpha_hi) / 2
                perturb_vec = alpha_mid * delta_l
                perturb_tensor = torch.tensor(perturb_vec, dtype=torch.bfloat16, device=device)
                
                hook_handle = None
                try:
                    hook_handle = layers[patch_li].register_forward_hook(
                        make_hook(perturb_tensor, pos_c))
                    
                    with torch.no_grad():
                        out_p = model(input_ids=input_ids_c, attention_mask=attn_mask_c)
                    
                    perturbed_margin = float(
                        out_p.logits[0, -1, tok_c_ids[0]] - out_p.logits[0, -1, tok_i_ids[0]])
                    
                    del out_p
                    hook_handle.remove()
                    hook_handle = None
                    
                    if clean_margin * perturbed_margin > 0:
                        alpha_lo = alpha_mid
                    else:
                        alpha_hi = alpha_mid
                        
                except Exception as e:
                    if hook_handle:
                        hook_handle.remove()
                    break
            
            boundary_by_layer[patch_li].append(alpha_mid)
        
        del hs_c, hs_i, inputs_c
        force_cleanup()
    
    # Aggregate
    result = {}
    for li in sorted(boundary_by_layer.keys()):
        alphas = [a for a in boundary_by_layer[li] if a >= 0]
        if alphas:
            result[li] = {
                "alpha_star_mean": float(np.mean(alphas)),
                "alpha_star_std": float(np.std(alphas)),
                "alpha_star_median": float(np.median(alphas)),
                "alpha_star_min": float(np.min(alphas)),
                "alpha_star_max": float(np.max(alphas)),
                "n_pairs": len(alphas),
            }
    
    del W_U_f32
    return result


# =====================================================================
# EXP2: JACOBIAN DIRECTIONAL AMPLIFICATION
# =====================================================================

def exp2_jacobian_amplification(model, tokenizer, device, n_layers, d_model, W_U):
    """
    ★★★ 机制核心: Jacobian方向放大率 ★★★
    
    在层l添加扰动ε*v, 测量对层l+1的影响:
    J_l · v ≈ (T_l(h_l + ε*v) - T_l(h_l)) / ε
    
    关键指标:
    - g_Δ(l) = ||J_l · Δ_hat|| = 约束方向放大率
    - g_perp(l) = ||J_l · v_perp|| = 正交方向放大率
    - g_Δ > 1 → barrier (扰动被放大)
    - g_Δ < 1 → attractor (扰动被衰减)
    - g_Δ / g_perp >> 1 → Δ方向是特殊输运通道
    
    也做3步power iteration估计λ_max:
    v_0 = Δ_hat → v_1 = J_l·v_0/||J_l·v_0|| → ... → λ_max ≈ ||v_k||
    """
    print("\n" + "="*60)
    print("Exp2: JACOBIAN DIRECTIONAL AMPLIFICATION")
    print("  (One-step transport: J_l · v analysis)")
    print("="*60)
    
    W_U_f32 = W_U.astype(np.float32)
    eps_rel = 0.01  # 1% perturbation for finite differences
    n_power_iter = 3  # Power iteration steps
    
    # Sample layers (not too many to keep runtime manageable)
    test_layers = sorted(set(
        [1, 2, 3, 5, 10, 15, 20, n_layers//2, n_layers-5, n_layers-2]
    ))
    test_layers = [l for l in test_layers if 1 <= l < n_layers]
    
    test_pairs = SYNTACTIC_SINGULAR[:10] + SYNTACTIC_PLURAL[:5]
    print(f"  Testing {len(test_pairs)} pairs × {len(test_layers)} layers")
    print(f"  ε_rel={eps_rel}, power_iter_steps={n_power_iter}")
    
    jacobian_results = defaultdict(list)  # {layer: [{g_delta, g_perp, lambda_max, ...}]}
    
    for pi, (sent_c, sent_i, verb) in enumerate(test_pairs):
        if pi % 3 == 0:
            print(f"    Pair {pi+1}/{len(test_pairs)}", flush=True)
        
        tok_c_ids = tokenizer.encode(verb, add_special_tokens=False)
        verb_alt = verb.rstrip("s") if verb.endswith("s") else verb + "s"
        tok_i_ids = tokenizer.encode(verb_alt, add_special_tokens=False)
        if not tok_c_ids or not tok_i_ids:
            continue
        
        pos_c = find_verb_position(tokenizer, sent_c, verb)
        pos_i = find_verb_position(tokenizer, sent_i, verb)
        
        # Get ALL clean hidden states
        hs_c, _ = get_all_hidden_states(model, tokenizer, device, sent_c, pos_c)
        hs_i, _ = get_all_hidden_states(model, tokenizer, device, sent_i, pos_i)
        
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
            
            # Unit direction vectors
            delta_hat = delta_l / delta_norm  # Constraint direction
            
            # Orthogonal direction (Gram-Schmidt)
            rng = np.random.RandomState(42 + pi)
            v_rand = rng.randn(d_model).astype(np.float32)
            v_perp = v_rand - np.dot(v_rand, delta_hat) * delta_hat
            v_perp_norm = float(np.linalg.norm(v_perp))
            if v_perp_norm < 1e-10:
                continue
            v_perp = v_perp / v_perp_norm
            
            # Perturbation magnitude
            eps_abs = eps_rel * h_norm
            
            # --- Compute J_l · v for Δ direction ---
            g_delta = _compute_jacobian_action(
                model, layers, patch_li, pos_c, input_ids_c, attn_mask_c,
                hs_c, delta_hat, eps_abs, device)
            
            # --- Compute J_l · v for ⊥ direction ---
            g_perp = _compute_jacobian_action(
                model, layers, patch_li, pos_c, input_ids_c, attn_mask_c,
                hs_c, v_perp, eps_abs, device)
            
            # --- Power iteration for λ_max ---
            lambda_max = _compute_lambda_max(
                model, layers, patch_li, pos_c, input_ids_c, attn_mask_c,
                hs_c, delta_hat, eps_abs, device, n_power_iter)
            
            if g_delta is not None and g_perp is not None:
                jacobian_results[patch_li].append({
                    "g_delta": g_delta,
                    "g_perp": g_perp,
                    "lambda_max": lambda_max if lambda_max is not None else 0.0,
                    "delta_norm": delta_norm,
                    "h_norm": h_norm,
                    "eps_abs": eps_abs,
                })
        
        del hs_c, hs_i, inputs_c
        force_cleanup()
    
    # Aggregate
    result = {}
    for li in sorted(jacobian_results.keys()):
        gd = [r["g_delta"] for r in jacobian_results[li]]
        gp = [r["g_perp"] for r in jacobian_results[li]]
        lm = [r["lambda_max"] for r in jacobian_results[li]]
        result[li] = {
            "g_delta_mean": float(np.mean(gd)),
            "g_delta_std": float(np.std(gd)),
            "g_delta_median": float(np.median(gd)),
            "g_perp_mean": float(np.mean(gp)),
            "g_perp_std": float(np.std(gp)),
            "g_perp_median": float(np.median(gp)),
            "lambda_max_mean": float(np.mean(lm)),
            "lambda_max_std": float(np.std(lm)),
            "lambda_max_median": float(np.median(lm)),
            "selectivity_mean": float(np.mean([r["g_delta"]/max(r["g_perp"],1e-10) for r in jacobian_results[li]])),
            "is_barrier": float(np.mean([1 if g > 1.0 else 0 for g in gd])),
            "n_pairs": len(gd),
        }
    
    del W_U_f32
    return result


def _compute_jacobian_action(model, layers, patch_li, target_pos,
                              input_ids, attn_mask, hs_clean,
                              direction, eps_abs, device):
    """
    计算 J_l · v 的范数 (单步Jacobian作用)
    
    方法:
    1. 在层l-1的输出添加 ε*v
    2. 捕获层l的输出
    3. J_l · v ≈ (h_{l+1}' - h_{l+1}) / ε
    4. 返回 ||J_l · v||
    """
    if patch_li < 1 or patch_li >= len(layers):
        return None
    
    # Perturbation to add to output of layer patch_li-1
    perturb_vec = eps_abs * direction
    
    hook_inject = None
    hook_capture = None
    captured_output = {}
    
    def make_inject_hook(pvec, tpos):
        pt = torch.tensor(pvec, dtype=torch.bfloat16, device=device)
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                new_out = output[0].detach().clone()
                p = min(tpos, new_out.shape[1] - 1)
                new_out[0, p] += pt.to(new_out.device)
                return (new_out,) + output[1:]
            return output
        return hook_fn
    
    def make_capture_hook(key):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured_output[key] = output[0][0, min(target_pos, output[0].shape[1]-1)].detach().cpu().float().numpy()
            else:
                captured_output[key] = output[0, min(target_pos, output[0].shape[1]-1)].detach().cpu().float().numpy()
        return hook_fn
    
    try:
        # Inject perturbation at layer patch_li-1's output
        hook_inject = layers[patch_li - 1].register_forward_hook(
            make_inject_hook(perturb_vec, target_pos))
        
        # Capture output at layer patch_li
        hook_capture = layers[patch_li].register_forward_hook(
            make_capture_hook("perturbed"))
        
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn_mask)
        
        hook_inject.remove()
        hook_inject = None
        hook_capture.remove()
        hook_capture = None
        
        if "perturbed" not in captured_output:
            return None
        
        # Clean output at layer patch_li
        clean_h = hs_clean.get(patch_li + 1)  # h_{l+1} = output of layer l
        if clean_h is None:
            return None
        
        # Jacobian action
        delta_h = captured_output["perturbed"] - clean_h
        g = float(np.linalg.norm(delta_h)) / eps_abs
        
        return g
        
    except Exception as e:
        if hook_inject:
            hook_inject.remove()
        if hook_capture:
            hook_capture.remove()
        return None


def _compute_lambda_max(model, layers, patch_li, target_pos,
                         input_ids, attn_mask, hs_clean,
                         init_direction, eps_abs, device, n_iter=3):
    """
    Power iteration估计λ_max(J_l^T J_l)
    
    v_0 = init_direction
    v_{k+1} = J_l · v_k / ||J_l · v_k||
    λ_max ≈ ||J_l · v_{k}||
    """
    if patch_li < 1 or patch_li >= len(layers):
        return None
    
    v = init_direction.copy()
    lambda_estimates = []
    
    for k in range(n_iter):
        # Compute J_l · v
        delta_h = _compute_jacobian_action_raw(
            model, layers, patch_li, target_pos, input_ids, attn_mask,
            hs_clean, v, eps_abs, device)
        
        if delta_h is None:
            break
        
        sigma = float(np.linalg.norm(delta_h))
        lambda_estimates.append(sigma)
        
        if sigma < 1e-10:
            break
        
        # Update direction
        v = delta_h / sigma
    
    if lambda_estimates:
        return lambda_estimates[-1]  # Last estimate is most accurate
    return None


def _compute_jacobian_action_raw(model, layers, patch_li, target_pos,
                                  input_ids, attn_mask, hs_clean,
                                  direction, eps_abs, device):
    """返回J_l · direction (向量, 非范数)"""
    if patch_li < 1 or patch_li >= len(layers):
        return None
    
    perturb_vec = eps_abs * direction
    
    hook_inject = None
    hook_capture = None
    captured_output = {}
    
    def make_inject_hook(pvec, tpos):
        pt = torch.tensor(pvec, dtype=torch.bfloat16, device=device)
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                new_out = output[0].detach().clone()
                p = min(tpos, new_out.shape[1] - 1)
                new_out[0, p] += pt.to(new_out.device)
                return (new_out,) + output[1:]
            return output
        return hook_fn
    
    def make_capture_hook(key):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured_output[key] = output[0][0, min(target_pos, output[0].shape[1]-1)].detach().cpu().float().numpy()
            else:
                captured_output[key] = output[0, min(target_pos, output[0].shape[1]-1)].detach().cpu().float().numpy()
        return hook_fn
    
    try:
        hook_inject = layers[patch_li - 1].register_forward_hook(
            make_inject_hook(perturb_vec, target_pos))
        hook_capture = layers[patch_li].register_forward_hook(
            make_capture_hook("perturbed"))
        
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn_mask)
        
        hook_inject.remove()
        hook_inject = None
        hook_capture.remove()
        hook_capture = None
        
        if "perturbed" not in captured_output:
            return None
        
        clean_h = hs_clean.get(patch_li + 1)
        if clean_h is None:
            return None
        
        delta_h = (captured_output["perturbed"] - clean_h) / eps_abs
        return delta_h
        
    except Exception as e:
        if hook_inject:
            hook_inject.remove()
        if hook_capture:
            hook_capture.remove()
        return None


# =====================================================================
# EXP3: MULTI-STEP PROPAGATION PROFILE
# =====================================================================

def exp3_propagation_profile(model, tokenizer, device, n_layers, d_model, W_U):
    """
    ★ 多步传播剖面: 扰动在层间的全局传播 ★
    
    在层l注入扰动ε*Δ_hat, 测量在所有后续层的传播:
    ||δh_k|| for k = l+1, l+2, ..., L
    
    关键:
    - 递减曲线 → attractor (扰动被逐步吸收)
    - 递增曲线 → barrier (扰动被逐步放大)
    - 振荡曲线 → 混合动力学
    """
    print("\n" + "="*60)
    print("Exp3: MULTI-STEP PROPAGATION PROFILE")
    print("  (Inject at layer l, measure at ALL subsequent layers)")
    print("="*60)
    
    W_U_f32 = W_U.astype(np.float32)
    eps_rel = 0.05  # 5% perturbation
    
    # Injection layers
    inject_layers = sorted(set([1, 5, 10, n_layers//2, n_layers-5, n_layers-2]))
    inject_layers = [l for l in inject_layers if 1 <= l < n_layers]
    
    test_pairs = SYNTACTIC_SINGULAR[:8] + SYNTACTIC_PLURAL[:4]
    print(f"  Testing {len(test_pairs)} pairs × {len(inject_layers)} injection layers")
    
    propagation_results = defaultdict(list)  # {inject_layer: [{norm_by_layer: {k: ||δh_k||}}]}
    
    for pi, (sent_c, sent_i, verb) in enumerate(test_pairs):
        if pi % 3 == 0:
            print(f"    Pair {pi+1}/{len(test_pairs)}", flush=True)
        
        tok_c_ids = tokenizer.encode(verb, add_special_tokens=False)
        verb_alt = verb.rstrip("s") if verb.endswith("s") else verb + "s"
        tok_i_ids = tokenizer.encode(verb_alt, add_special_tokens=False)
        if not tok_c_ids or not tok_i_ids:
            continue
        
        pos_c = find_verb_position(tokenizer, sent_c, verb)
        
        # Get clean hidden states
        hs_c, _ = get_all_hidden_states(model, tokenizer, device, sent_c, pos_c)
        
        layers = get_layers(model)
        inputs_c = tokenizer(sent_c, return_tensors="pt", truncation=True, max_length=128)
        input_ids_c = inputs_c["input_ids"].to(device)
        attn_mask_c = inputs_c["attention_mask"].to(device)
        
        for inject_li in inject_layers:
            if inject_li not in hs_c:
                continue
            
            delta_l = hs_c.get(inject_li) - hs_c.get(inject_li, np.zeros_like(hs_c[0]))
            # We need Δ from the incorrect sentence
            hs_i_temp, _ = get_all_hidden_states(model, tokenizer, device, sent_i, 
                                                  find_verb_position(tokenizer, sent_i, verb))
            
            if inject_li not in hs_c or inject_li not in hs_i_temp:
                del hs_i_temp
                continue
            
            delta_l = hs_c[inject_li] - hs_i_temp[inject_li]
            delta_norm = float(np.linalg.norm(delta_l))
            h_norm = float(np.linalg.norm(hs_c[inject_li]))
            
            if delta_norm < 1e-10 or h_norm < 1e-10:
                del hs_i_temp
                continue
            
            delta_hat = delta_l / delta_norm
            eps_abs = eps_rel * h_norm
            perturb_vec = eps_abs * delta_hat
            
            # Inject at inject_li and capture at ALL subsequent layers
            captured_perturbed = {}
            
            def make_inject_hook(pvec, tpos):
                pt = torch.tensor(pvec, dtype=torch.bfloat16, device=device)
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        new_out = output[0].detach().clone()
                        p = min(tpos, new_out.shape[1] - 1)
                        new_out[0, p] -= pt.to(new_out.device)  # Move toward incorrect
                        return (new_out,) + output[1:]
                    return output
                return hook_fn
            
            capture_hooks = []
            
            def make_capture_hook(key, tpos):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        captured_perturbed[key] = output[0][0, min(tpos, output[0].shape[1]-1)].detach().cpu().float().numpy()
                return hook_fn
            
            try:
                # Inject hook
                inject_hook = layers[inject_li].register_forward_hook(
                    make_inject_hook(perturb_vec, pos_c))
                
                # Capture hooks on ALL subsequent layers
                for cap_li in range(inject_li + 1, n_layers + 1):
                    if cap_li < n_layers:
                        h = layers[cap_li].register_forward_hook(
                            make_capture_hook(f"L{cap_li}", pos_c))
                        capture_hooks.append(h)
                
                with torch.no_grad():
                    model(input_ids=input_ids_c, attention_mask=attn_mask_c)
                
                inject_hook.remove()
                for h in capture_hooks:
                    h.remove()
                
                # Compute propagation norms
                norm_by_layer = {}
                for cap_li in range(inject_li + 1, n_layers + 1):
                    key = f"L{cap_li}"
                    if key in captured_perturbed and cap_li in hs_c:
                        diff = captured_perturbed[key] - hs_c[cap_li]
                        norm_by_layer[cap_li] = float(np.linalg.norm(diff))
                
                if norm_by_layer:
                    propagation_results[inject_li].append({
                        "norm_by_layer": norm_by_layer,
                        "inject_layer": inject_li,
                        "delta_norm": delta_norm,
                        "h_norm": h_norm,
                    })
                
            except Exception as e:
                try:
                    inject_hook.remove()
                except:
                    pass
                for h in capture_hooks:
                    try:
                        h.remove()
                    except:
                        pass
            
            del hs_i_temp
            force_cleanup()
        
        del hs_c, inputs_c
        force_cleanup()
    
    # Aggregate
    result = {}
    for inject_li in sorted(propagation_results.keys()):
        entries = propagation_results[inject_li]
        if not entries:
            continue
        
        # Average norm at each capture layer
        all_cap_layers = sorted(set(
            k for e in entries for k in e["norm_by_layer"].keys()
        ))
        
        avg_norms = {}
        for cap_li in all_cap_layers:
            norms = [e["norm_by_layer"][cap_li] for e in entries if cap_li in e["norm_by_layer"]]
            if norms:
                avg_norms[cap_li] = {
                    "mean": float(np.mean(norms)),
                    "std": float(np.std(norms)),
                    "n": len(norms),
                }
        
        # Compute propagation slope
        if len(all_cap_layers) >= 2:
            first_norm = avg_norms[all_cap_layers[0]]["mean"]
            last_norm = avg_norms[all_cap_layers[-1]]["mean"]
            n_steps = all_cap_layers[-1] - all_cap_layers[0]
            if n_steps > 0 and first_norm > 1e-10:
                slope = (last_norm - first_norm) / (n_steps * first_norm)
            else:
                slope = 0.0
        else:
            slope = 0.0
        
        result[inject_li] = {
            "avg_norms": {str(k): v for k, v in avg_norms.items()},
            "propagation_slope": slope,
            "n_entries": len(entries),
            "verdict": "GROWING (barrier)" if slope > 0.01 else "DECAYING (attractor)" if slope < -0.01 else "STABLE",
        }
    
    del W_U_f32
    return result


# =====================================================================
# EXP4: CONSTRAINT TYPE COMPARISON
# =====================================================================

def exp4_constraint_type_comparison(model, tokenizer, device, n_layers, d_model, W_U):
    """
    ★ 约束类型拓扑对比 ★
    
    三种约束类型:
    1. 语法约束: "The cat sleeps" vs "The cats sleeps" (S-V agreement)
    2. 语义约束: "The cat sleeps" vs "The rock sleeps" (selectional restriction)
    3. 事实约束: "Paris is in France" vs "Paris is in Germany" (world knowledge)
    
    对比:
    - α*(l) 边界形成轨迹
    - g_Δ(l) Jacobian放大率
    - 内部分离度: ||Δ_l||/||h_l|| 随层变化
    
    核心问题: 不同约束类型是否有不同的边界拓扑?
    """
    print("\n" + "="*60)
    print("Exp4: CONSTRAINT TYPE COMPARISON")
    print("  (Syntactic vs Semantic vs Factual)")
    print("="*60)
    
    W_U_f32 = W_U.astype(np.float32)
    key_layers = sorted(set([1, 3, 5, 10, n_layers//2, n_layers-5, n_layers-1]))
    key_layers = [l for l in key_layers if 1 <= l < n_layers]
    
    results = {}
    
    # --- Type 1: Syntactic (S-V agreement) ---
    print("\n  [A] Syntactic constraint (S-V agreement)...")
    synt_pairs = SYNTACTIC_SINGULAR[:10]
    synt_internal = defaultdict(list)  # {layer: [||Δ||/||h|| values]}
    
    for pi, (sent_c, sent_i, verb) in enumerate(synt_pairs):
        tok_c_ids = tokenizer.encode(verb, add_special_tokens=False)
        verb_alt = verb.rstrip("s") if verb.endswith("s") else verb + "s"
        tok_i_ids = tokenizer.encode(verb_alt, add_special_tokens=False)
        if not tok_c_ids or not tok_i_ids:
            continue
        
        pos_c = find_verb_position(tokenizer, sent_c, verb)
        pos_i = find_verb_position(tokenizer, sent_i, verb)
        
        hs_c, _ = get_all_hidden_states(model, tokenizer, device, sent_c, pos_c)
        hs_i, _ = get_all_hidden_states(model, tokenizer, device, sent_i, pos_i)
        
        for li in key_layers:
            if li in hs_c and li in hs_i:
                delta = hs_c[li] - hs_i[li]
                h_norm = float(np.linalg.norm(hs_c[li]))
                d_norm = float(np.linalg.norm(delta))
                if h_norm > 1e-10:
                    synt_internal[li].append(d_norm / h_norm)
        
        del hs_c, hs_i
        force_cleanup()
    
    # --- Type 2: Semantic (selectional restriction) ---
    print("  [B] Semantic constraint (selectional restriction)...")
    sem_internal = defaultdict(list)
    
    for pi, (sent_c, sent_i, verb) in enumerate(SEMANTIC_PAIRS):
        # For semantic pairs, both sentences have the SAME verb
        # The constraint is about whether the subject is compatible with the verb
        tok_c_ids = tokenizer.encode(verb, add_special_tokens=False)
        verb_alt = verb.rstrip("s") if verb.endswith("s") else verb + "s"
        tok_i_ids = tokenizer.encode(verb_alt, add_special_tokens=False)
        if not tok_c_ids or not tok_i_ids:
            continue
        
        pos_c = find_verb_position(tokenizer, sent_c, verb)
        pos_i = find_verb_position(tokenizer, sent_i, verb)
        
        hs_c, _ = get_all_hidden_states(model, tokenizer, device, sent_c, pos_c)
        hs_i, _ = get_all_hidden_states(model, tokenizer, device, sent_i, pos_i)
        
        for li in key_layers:
            if li in hs_c and li in hs_i:
                delta = hs_c[li] - hs_i[li]
                h_norm = float(np.linalg.norm(hs_c[li]))
                d_norm = float(np.linalg.norm(delta))
                if h_norm > 1e-10:
                    sem_internal[li].append(d_norm / h_norm)
        
        del hs_c, hs_i
        force_cleanup()
    
    # --- Type 3: Factual (world knowledge) ---
    print("  [C] Factual constraint (world knowledge)...")
    fact_internal = defaultdict(list)
    
    for pi, (sent_c, sent_i, tok_c_str, tok_i_str) in enumerate(FACTUAL_PAIRS):
        tok_c_ids = tokenizer.encode(tok_c_str, add_special_tokens=False)
        tok_i_ids = tokenizer.encode(tok_i_str, add_special_tokens=False)
        if not tok_c_ids or not tok_i_ids:
            continue
        
        # Use last position for factual statements
        hs_c, _ = get_all_hidden_states(model, tokenizer, device, sent_c, 
                                         len(tokenizer.encode(sent_c)) - 1)
        hs_i, _ = get_all_hidden_states(model, tokenizer, device, sent_i,
                                         len(tokenizer.encode(sent_i)) - 1)
        
        for li in key_layers:
            if li in hs_c and li in hs_i:
                delta = hs_c[li] - hs_i[li]
                h_norm = float(np.linalg.norm(hs_c[li]))
                d_norm = float(np.linalg.norm(delta))
                if h_norm > 1e-10:
                    fact_internal[li].append(d_norm / h_norm)
        
        del hs_c, hs_i
        force_cleanup()
    
    # Aggregate
    for ctype, data in [("syntactic", synt_internal), ("semantic", sem_internal), ("factual", fact_internal)]:
        ctype_result = {}
        for li in sorted(data.keys()):
            vals = data[li]
            if vals:
                ctype_result[li] = {
                    "relative_delta_mean": float(np.mean(vals)),
                    "relative_delta_std": float(np.std(vals)),
                    "n_pairs": len(vals),
                }
        results[ctype] = ctype_result
    
    # Compute "boundary formation slope" for each type
    for ctype in ["syntactic", "semantic", "factual"]:
        ctype_data = results[ctype]
        layers_sorted = sorted([int(k) for k in ctype_data.keys() if k != "_meta"])
        if len(layers_sorted) >= 2:
            first_key = layers_sorted[0]
            last_key = layers_sorted[-1]
            # Keys may be int or str
            if first_key not in ctype_data:
                first_key = str(first_key)
            if last_key not in ctype_data:
                last_key = str(last_key)
            first_val = ctype_data[first_key]["relative_delta_mean"]
            last_val = ctype_data[last_key]["relative_delta_mean"]
            n_steps = layers_sorted[-1] - layers_sorted[0]
            if n_steps > 0:
                slope = (last_val - first_val) / n_steps
            else:
                slope = 0.0
            results[ctype]["_meta"] = {
                "formation_slope": slope,
                "verdict": "INCREASING (boundary forms)" if slope > 0.001 else "DECREASING" if slope < -0.001 else "FLAT",
                "first_layer_delta": first_val,
                "last_layer_delta": last_val,
            }
    
    del W_U_f32
    return results


# =====================================================================
# MAIN
# =====================================================================

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    t_start = time.time()
    
    print(f"\n{'#'*70}")
    print(f"# Phase 185: TRANSPORT TOPOLOGY & JACOBIAN SPECTRUM — {model_name}")
    print(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers, d_model, vocab_size = info.n_layers, info.d_model, info.vocab_size
    print(f"\nModel: {info.model_class}, Layers={n_layers}, d_model={d_model}, vocab={vocab_size}")
    
    W_U = get_W_U(model, model_name).astype(np.float32)
    print(f"W_U shape: {W_U.shape}, dtype: {W_U.dtype}")
    
    # ===== Exp1: Per-Layer Boundary Formation =====
    print(f"\n{'='*70}")
    print("Running Exp1: Per-Layer Boundary Formation...")
    print("  ★★★ Shows how constraint boundary forms layer by layer ★★★")
    exp1_results = exp1_per_layer_boundary(model, tokenizer, device, n_layers, d_model, W_U)
    force_cleanup()
    
    # ===== Exp2: Jacobian Directional Amplification =====
    print(f"\n{'='*70}")
    print("Running Exp2: Jacobian Directional Amplification...")
    print("  ★★★ Mechanism: attractor vs barrier via Jacobian spectrum ★★★")
    exp2_results = exp2_jacobian_amplification(model, tokenizer, device, n_layers, d_model, W_U)
    force_cleanup()
    
    # ===== Exp3: Multi-Step Propagation Profile =====
    print(f"\n{'='*70}")
    print("Running Exp3: Multi-Step Propagation Profile...")
    print("  ★★★ Global propagation dynamics from injection to output ★★★")
    exp3_results = exp3_propagation_profile(model, tokenizer, device, n_layers, d_model, W_U)
    force_cleanup()
    
    # ===== Exp4: Constraint Type Comparison =====
    print(f"\n{'='*70}")
    print("Running Exp4: Constraint Type Comparison...")
    print("  ★★★ Syntactic vs Semantic vs Factual boundary topology ★★★")
    exp4_results = exp4_constraint_type_comparison(model, tokenizer, device, n_layers, d_model, W_U)
    force_cleanup()
    
    # ===== Save =====
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = f"tests/glm5_temp/phase185_{model_name}_{timestamp}.json"
    
    full_results = {
        "model": model_name, "n_layers": n_layers, "d_model": d_model, "vocab_size": vocab_size,
        "timestamp": timestamp, "elapsed_sec": round(time.time() - t_start, 1),
        "exp1_per_layer_boundary": {str(k): v for k, v in exp1_results.items()},
        "exp2_jacobian_amplification": {str(k): v for k, v in exp2_results.items()},
        "exp3_propagation_profile": {str(k): v for k, v in exp3_results.items()},
        "exp4_constraint_type_comparison": exp4_results,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    
    # ===== Summary =====
    print(f"\n{'#'*70}")
    print("PHASE 185 SUMMARY")
    print(f"{'#'*70}")
    
    # Exp1: Boundary Formation
    print("\n★★★ Exp1: Per-Layer Boundary Formation ★★★")
    layers_sorted = sorted([int(k) for k in exp1_results.keys()])
    for li in layers_sorted:
        d = exp1_results[li]
        alpha = d.get("alpha_star_mean", 0)
        alpha_std = d.get("alpha_star_std", 0)
        print(f"  L{li}: α*={alpha:.3f}±{alpha_std:.3f} (n={d.get('n_pairs',0)})")
    
    # Boundary formation trend
    if len(layers_sorted) >= 2:
        first_alpha = exp1_results[layers_sorted[0]]["alpha_star_mean"]
        last_alpha = exp1_results[layers_sorted[-1]]["alpha_star_mean"]
        trend = "INCREASING (boundary forms in depth)" if last_alpha > first_alpha + 0.05 else \
                "DECREASING" if last_alpha < first_alpha - 0.05 else "FLAT"
        print(f"\n  ★ Trend: α*({layers_sorted[0]})={first_alpha:.3f} → α*({layers_sorted[-1]})={last_alpha:.3f} [{trend}]")
        if trend.startswith("INCREASING"):
            print(f"  ★★★ BOUNDARY IS INTERNALLY FORMED, NOT DECODER-INDUCED ★★★")
        else:
            print(f"  ⚠️ Boundary may be decoder-induced (flat or decreasing α*)")
    
    # Exp2: Jacobian
    print("\n★★★ Exp2: Jacobian Directional Amplification ★★★")
    for li in sorted([int(k) for k in exp2_results.keys()]):
        d = exp2_results[li]
        gd = d.get("g_delta_mean", 0)
        gp = d.get("g_perp_mean", 0)
        lm = d.get("lambda_max_mean", 0)
        sel = d.get("selectivity_mean", 0)
        is_barrier = d.get("is_barrier", 0)
        tag = "BARRIER" if is_barrier > 0.5 else "ATTRACTOR"
        print(f"  L{li}: g_Δ={gd:.3f}, g_⊥={gp:.3f}, λ_max={lm:.3f}, S={sel:.2f} [{tag}]")
    
    # Exp3: Propagation
    print("\n★★★ Exp3: Multi-Step Propagation Profile ★★★")
    for inject_li in sorted([int(k) for k in exp3_results.keys()]):
        d = exp3_results[inject_li]
        slope = d.get("propagation_slope", 0)
        verdict = d.get("verdict", "N/A")
        print(f"  Inject L{inject_li}: slope={slope:.4f} [{verdict}]")
    
    # Exp4: Constraint Types
    print("\n★★★ Exp4: Constraint Type Comparison ★★★")
    for ctype in ["syntactic", "semantic", "factual"]:
        if ctype in exp4_results:
            meta = exp4_results[ctype].get("_meta", {})
            slope = meta.get("formation_slope", 0)
            verdict = meta.get("verdict", "N/A")
            first_d = meta.get("first_layer_delta", 0)
            last_d = meta.get("last_layer_delta", 0)
            print(f"  {ctype}: slope={slope:.5f}, first_δ={first_d:.4f}, last_δ={last_d:.4f} [{verdict}]")
    
    release_model(model)
    elapsed = time.time() - t_start
    print(f"\n{'#'*70}")
    print(f"Phase 185 COMPLETE! Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
