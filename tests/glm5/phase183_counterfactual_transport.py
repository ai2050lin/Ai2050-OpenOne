"""
Phase 183: Counterfactual Transport Geometry
=============================================

核心理论:
  Δ_l = h_correct - h_incorrect 混合了"token嵌入差异"和"约束信号"
  需要反事实输运(counterfactual transport)来分离

★★★ 三个实验 ★★★

Exp1: Same-Token Counterfactual Analysis
  - 构造"目标token相同但约束不同"的句子对
  - 例如: "The cat sleeps" (correct) vs "The cats sleeps" (incorrect)
    → 两个句子的动词位置都是 "sleeps", 没有token嵌入污染
  - 与原始对比较, 量化token嵌入污染比例
  - 与context-only对比较, 分离纯约束信号

Exp2: Attractor Stability Test (Perturbation Recovery)
  - 对正确句子的hidden state施加朝错误方向的扰动
  - 测量模型是否能恢复 → 吸引子存在
  - 扰动尺度: ε = 0.1, 0.5, 1.0
  - 恢复率 R(ε, l) = margin(perturbed) / margin(clean)

Exp3: Jacobian Constraint Direction
  - 因果约束方向: v_c = W_U[tok_correct] - W_U[tok_incorrect]
  - 测量观测Δ与因果方向的对齐度: cos(Δ_l, v_c)
  - 关键: 反事实Δ对齐 vs 原始Δ对齐

Usage: python tests/glm5/phase183_counterfactual_transport.py <model_name>
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
    print(f"[P183] Loading {model_name} (bfloat16 + device_map=auto)...")
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
    print(f"[P183] {model_name} loaded: device={device}, class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def find_verb_position(tokenizer, sentence, verb_str):
    """找到动词token在句子中的位置"""
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    # 尝试找到动词token
    verb_ids = tokenizer.encode(" " + verb_str, add_special_tokens=False)
    if not verb_ids:
        verb_ids = tokenizer.encode(verb_str, add_special_tokens=False)
    if len(verb_ids) >= 1:
        for i in range(len(tokens) - len(verb_ids) + 1):
            if all(tokens[i+j] == verb_ids[j] for j in range(len(verb_ids))):
                return i  # 返回第一个token的位置
    # 后备: 逐个解码找
    for i, tid in enumerate(tokens):
        decoded = tokenizer.decode([tid]).strip().lower()
        if verb_str.lower() in decoded:
            return i
    # 最后手段: 假设短句中动词在位置3
    return min(3, len(tokens) - 2)


def get_hidden_at_pos(model, tokenizer, device, sentence, target_pos):
    """获取句子在目标位置所有层的hidden states"""
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


# =====================================================================
# SAME-TOKEN COUNTERFACTUAL PAIRS (40+ pairs)
# =====================================================================
# 格式: (correct_sentence, incorrect_sentence, verb_token)
# 关键: 两个句子的目标动词token完全相同, 只是主语数不同

SAME_TOKEN_SINGULAR_VERB = [
    # Singular subject + singular verb (correct) vs Plural subject + singular verb (incorrect)
    # 目标token = 单数动词, 两个句子中完全相同
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
    ("The fish swims", "The fishs swims", "swims"),  # grammatically both wrong but tests signal
    ("The clock ticks", "The clocks ticks", "ticks"),
    ("The door opens", "The doors opens", "opens"),
    ("The fire burns", "The fires burns", "burns"),
    ("The star shines", "The stars shines", "shines"),
    ("The song plays", "The songs plays", "plays"),
    ("The story ends", "The stories ends", "ends"),
]

SAME_TOKEN_PLURAL_VERB = [
    # Plural subject + plural verb (correct) vs Singular subject + plural verb (incorrect)
    # 目标token = 复数动词, 两个句子中完全相同
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

# Context-only control pairs (same verb, different subject, both correct → no constraint difference)
CONTEXT_ONLY_PAIRS = [
    ("The cat sleeps", "The dog sleeps", "sleeps"),
    ("The dog runs", "The cat runs", "runs"),
    ("The bird sings", "The girl sings", "sings"),
    ("The child plays", "The dog plays", "plays"),
    ("The man walks", "The woman walks", "walks"),
    ("The boy jumps", "The girl jumps", "jumps"),
    ("The teacher speaks", "The doctor speaks", "speaks"),
    ("The horse gallops", "The dog gallops", "gallops"),
    ("The river flows", "The wind flows", "flows"),
    ("The tree grows", "The flower grows", "grows"),
    ("The baby cries", "The child cries", "cries"),
    ("The rain falls", "The snow falls", "falls"),
    ("The car drives", "The bus drives", "drives"),
    ("The fire burns", "The sun burns", "burns"),
    ("The star shines", "The light shines", "shines"),
]

# Original pairs (different target tokens) for comparison
ORIGINAL_PAIRS = [
    ("The cat sleeps", "The cat sleep", "sleeps", "sleep"),
    ("The dog runs", "The dog run", "runs", "run"),
    ("The bird sings", "The bird sing", "sings", "sing"),
    ("The child plays", "The child play", "plays", "play"),
    ("The sister reads", "The sister read", "reads", "read"),
    ("The flower grows", "The flower grow", "grows", "grow"),
    ("The river flows", "The river flow", "flows", "flow"),
    ("The mother cooks", "The mother cook", "cooks", "cook"),
    ("The student writes", "The student write", "writes", "write"),
    ("The horse gallops", "The horse gallop", "gallops", "gallop"),
    ("The wind blows", "The wind blow", "blows", "blow"),
    ("The rabbit hops", "The rabbit hop", "hops", "hop"),
    ("The girl dances", "The girl dance", "dances", "dance"),
    ("The boat sails", "The boat sail", "sails", "sail"),
    ("The plane flies", "The plane fly", "flies", "fly"),
    ("The teacher speaks", "The teacher speak", "speaks", "speak"),
    ("The doctor works", "The doctor work", "works", "work"),
    ("The baby cries", "The baby cry", "cries", "cry"),
    ("The rain falls", "The rain fall", "falls", "fall"),
    ("The man thinks", "The man think", "thinks", "think"),
    ("The dog barks", "The dog bark", "barks", "bark"),
    ("The bell rings", "The bell ring", "rings", "ring"),
    ("The fire burns", "The fire burn", "burns", "burn"),
    ("The star shines", "The star shine", "shines", "shine"),
    ("The tree grows", "The tree grow", "grows", "grow"),
    ("The cats sleep", "The cats sleeps", "sleep", "sleeps"),
    ("The dogs run", "The dogs runs", "run", "runs"),
    ("The birds sing", "The birds sings", "sing", "sings"),
    ("The children play", "The children plays", "play", "plays"),
    ("The students read", "The students reads", "read", "reads"),
]


# =====================================================================
# EXP1: SAME-TOKEN COUNTERFACTUAL ANALYSIS
# =====================================================================

def exp1_counterfactual_analysis(model, tokenizer, device, n_layers, d_model):
    """
    ★ 核心实验: 反事实输运分析
    
    比较三种Δ:
    1. Δ_original = h("cat sleeps") - h("cat sleep")  [不同token]
    2. Δ_counterfactual = h("cat sleeps") - h("cats sleeps")  [相同token, 不同约束]
    3. Δ_context = h("cat sleeps") - h("dog sleeps")  [相同token, 无约束差异]
    
    关键指标:
    - Token污染率 = 1 - ||Δ_counterfactual||/||Δ_original||
    - 约束隔离度 = ||Δ_counterfactual||/||h|| - ||Δ_context||/||h||
    """
    print("\n" + "="*60)
    print("Exp1: SAME-TOKEN COUNTERFACTUAL ANALYSIS")
    print("="*60)
    
    all_results = {}
    
    # --- Part A: Same-token counterfactual pairs ---
    print("\n  [A] Same-token counterfactual (singular verb)...")
    cf_delta_norms = defaultdict(list)
    cf_delta_rel = defaultdict(list)
    cf_cos_next = defaultdict(list)
    
    for pi, (sent_c, sent_i, verb) in enumerate(SAME_TOKEN_SINGULAR_VERB):
        if pi % 10 == 0:
            print(f"    Pair {pi+1}/{len(SAME_TOKEN_SINGULAR_VERB)}", flush=True)
        
        pos_c = find_verb_position(tokenizer, sent_c, verb)
        pos_i = find_verb_position(tokenizer, sent_i, verb)
        
        hs_c, _ = get_hidden_at_pos(model, tokenizer, device, sent_c, pos_c)
        hs_i, _ = get_hidden_at_pos(model, tokenizer, device, sent_i, pos_i)
        
        deltas = {}
        for li in range(n_layers + 1):
            if li in hs_c and li in hs_i:
                deltas[li] = hs_c[li] - hs_i[li]
                dn = float(np.linalg.norm(deltas[li]))
                hn = (float(np.linalg.norm(hs_c[li])) + float(np.linalg.norm(hs_i[li]))) / 2
                cf_delta_norms[li].append(dn)
                cf_delta_rel[li].append(dn / max(hn, 1e-10))
        
        for li in sorted(deltas.keys()):
            if li + 1 in deltas:
                dn = float(np.linalg.norm(deltas[li]))
                dn1 = float(np.linalg.norm(deltas[li+1]))
                if dn > 1e-10 and dn1 > 1e-10:
                    cf_cos_next[li].append(float(np.dot(deltas[li], deltas[li+1]) / (dn * dn1)))
        
        del hs_c, hs_i, deltas
        force_cleanup()
    
    # --- Part B: Original pairs (different target tokens) ---
    print("\n  [B] Original pairs (different tokens)...")
    orig_delta_norms = defaultdict(list)
    orig_delta_rel = defaultdict(list)
    orig_cos_next = defaultdict(list)
    
    for pi, (sent_c, sent_i, verb_c, verb_i) in enumerate(ORIGINAL_PAIRS):
        if pi % 10 == 0:
            print(f"    Pair {pi+1}/{len(ORIGINAL_PAIRS)}", flush=True)
        
        pos_c = find_verb_position(tokenizer, sent_c, verb_c)
        pos_i = find_verb_position(tokenizer, sent_i, verb_i)
        
        hs_c, _ = get_hidden_at_pos(model, tokenizer, device, sent_c, pos_c)
        hs_i, _ = get_hidden_at_pos(model, tokenizer, device, sent_i, pos_i)
        
        deltas = {}
        for li in range(n_layers + 1):
            if li in hs_c and li in hs_i:
                deltas[li] = hs_c[li] - hs_i[li]
                dn = float(np.linalg.norm(deltas[li]))
                hn = (float(np.linalg.norm(hs_c[li])) + float(np.linalg.norm(hs_i[li]))) / 2
                orig_delta_norms[li].append(dn)
                orig_delta_rel[li].append(dn / max(hn, 1e-10))
        
        for li in sorted(deltas.keys()):
            if li + 1 in deltas:
                dn = float(np.linalg.norm(deltas[li]))
                dn1 = float(np.linalg.norm(deltas[li+1]))
                if dn > 1e-10 and dn1 > 1e-10:
                    orig_cos_next[li].append(float(np.dot(deltas[li], deltas[li+1]) / (dn * dn1)))
        
        del hs_c, hs_i, deltas
        force_cleanup()
    
    # --- Part C: Context-only control pairs ---
    print("\n  [C] Context-only control (same verb, no constraint difference)...")
    ctx_delta_norms = defaultdict(list)
    ctx_delta_rel = defaultdict(list)
    
    for pi, (sent_a, sent_b, verb) in enumerate(CONTEXT_ONLY_PAIRS):
        if pi % 5 == 0:
            print(f"    Pair {pi+1}/{len(CONTEXT_ONLY_PAIRS)}", flush=True)
        
        pos_a = find_verb_position(tokenizer, sent_a, verb)
        pos_b = find_verb_position(tokenizer, sent_b, verb)
        
        hs_a, _ = get_hidden_at_pos(model, tokenizer, device, sent_a, pos_a)
        hs_b, _ = get_hidden_at_pos(model, tokenizer, device, sent_b, pos_b)
        
        for li in range(n_layers + 1):
            if li in hs_a and li in hs_b:
                delta = hs_a[li] - hs_b[li]
                dn = float(np.linalg.norm(delta))
                hn = (float(np.linalg.norm(hs_a[li])) + float(np.linalg.norm(hs_b[li]))) / 2
                ctx_delta_norms[li].append(dn)
                ctx_delta_rel[li].append(dn / max(hn, 1e-10))
        
        del hs_a, hs_b
        force_cleanup()
    
    # --- Part D: Same-token counterfactual (plural verb) ---
    print("\n  [D] Same-token counterfactual (plural verb)...")
    cf_pl_delta_norms = defaultdict(list)
    cf_pl_delta_rel = defaultdict(list)
    
    for pi, (sent_c, sent_i, verb) in enumerate(SAME_TOKEN_PLURAL_VERB):
        if pi % 5 == 0:
            print(f"    Pair {pi+1}/{len(SAME_TOKEN_PLURAL_VERB)}", flush=True)
        
        pos_c = find_verb_position(tokenizer, sent_c, verb)
        pos_i = find_verb_position(tokenizer, sent_i, verb)
        
        hs_c, _ = get_hidden_at_pos(model, tokenizer, device, sent_c, pos_c)
        hs_i, _ = get_hidden_at_pos(model, tokenizer, device, sent_i, pos_i)
        
        for li in range(n_layers + 1):
            if li in hs_c and li in hs_i:
                delta = hs_c[li] - hs_i[li]
                dn = float(np.linalg.norm(delta))
                hn = (float(np.linalg.norm(hs_c[li])) + float(np.linalg.norm(hs_i[li]))) / 2
                cf_pl_delta_norms[li].append(dn)
                cf_pl_delta_rel[li].append(dn / max(hn, 1e-10))
        
        del hs_c, hs_i
        force_cleanup()
    
    # --- Aggregate ---
    # Merge singular and plural counterfactual
    for li in cf_pl_delta_norms:
        cf_delta_norms[li].extend(cf_pl_delta_norms[li])
        cf_delta_rel[li].extend(cf_pl_delta_rel[li])
    
    result = {}
    for li in range(n_layers + 1):
        entry = {}
        if li in orig_delta_norms:
            entry["orig_delta_norm_mean"] = float(np.mean(orig_delta_norms[li]))
            entry["orig_delta_rel_mean"] = float(np.mean(orig_delta_rel[li]))
        if li in cf_delta_norms:
            entry["cf_delta_norm_mean"] = float(np.mean(cf_delta_norms[li]))
            entry["cf_delta_rel_mean"] = float(np.mean(cf_delta_rel[li]))
        if li in ctx_delta_norms:
            entry["ctx_delta_norm_mean"] = float(np.mean(ctx_delta_norms[li]))
            entry["ctx_delta_rel_mean"] = float(np.mean(ctx_delta_rel[li]))
        
        # ★ Key metrics
        if li in orig_delta_norms and li in cf_delta_norms:
            orig_m = np.mean(orig_delta_norms[li])
            cf_m = np.mean(cf_delta_norms[li])
            # Token contamination ratio
            entry["token_contamination"] = 1.0 - cf_m / max(orig_m, 1e-10)
            entry["token_contamination"] = max(0.0, min(1.0, entry["token_contamination"]))
        
        if li in cf_delta_rel and li in ctx_delta_rel:
            cf_rel = np.mean(cf_delta_rel[li])
            ctx_rel = np.mean(ctx_delta_rel[li])
            # Constraint isolation: counterfactual - context_only
            entry["constraint_isolation"] = cf_rel - ctx_rel
        
        if li in orig_cos_next:
            entry["orig_cos_next_mean"] = float(np.mean(orig_cos_next[li]))
        if li in cf_cos_next:
            entry["cf_cos_next_mean"] = float(np.mean(cf_cos_next[li]))
        
        if entry:
            result[li] = entry
    
    all_results["per_layer"] = result
    return all_results


# =====================================================================
# EXP2: ATTRACTOR STABILITY TEST (Perturbation Recovery)
# =====================================================================

def exp2_attractor_test(model, tokenizer, device, n_layers, d_model, W_U):
    """
    ★ 吸引子稳定性测试
    
    方法:
    1. 获取正确/错误句子的hidden states
    2. 对正确hidden state施加朝错误方向的扰动
    3. 让模型继续处理, 测量输出是否恢复
    4. 恢复率 R(ε, l) = margin(perturbed) / margin(clean)
    
    ★ R → 1: 强吸引子 (扰动被修正)
    ★ R → 0: 弱吸引子 (扰动被放大)
    """
    print("\n" + "="*60)
    print("Exp2: ATTRACTOR STABILITY TEST (Perturbation Recovery)")
    print("="*60)
    
    W_U_f32 = W_U.astype(np.float32)
    epsilons = [0.1, 0.5, 1.0]
    test_layers = sorted(set([1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 
                               n_layers//2, n_layers-5, n_layers-2, n_layers-1]))
    test_layers = [l for l in test_layers if 1 <= l < n_layers]
    
    # Use a subset of same-token pairs for speed
    test_pairs = SAME_TOKEN_SINGULAR_VERB[:15]
    
    all_results = {}
    
    for eps in epsilons:
        print(f"\n  [ε={eps}] Testing {len(test_layers)} layers, {len(test_pairs)} pairs...")
        recovery_rates = defaultdict(list)  # {layer: [recovery_rates]}
        
        for pi, (sent_c, sent_i, verb) in enumerate(test_pairs):
            if pi % 5 == 0:
                print(f"    Pair {pi+1}/{len(test_pairs)}", flush=True)
            
            # Get token IDs for margin computation
            tok_c_ids = tokenizer.encode(verb, add_special_tokens=False)
            # For same-token pairs, the "incorrect" version is the same verb but wrong agreement
            # The correct continuation after singular subject is the singular verb
            # We'll use the verb itself and a common incorrect form
            verb_plural = verb.rstrip("s") if verb.endswith("s") else verb + "s"
            tok_i_ids = tokenizer.encode(verb_plural, add_special_tokens=False)
            
            pos_c = find_verb_position(tokenizer, sent_c, verb)
            pos_i = find_verb_position(tokenizer, sent_i, verb)
            
            # Get clean hidden states
            hs_c, _ = get_hidden_at_pos(model, tokenizer, device, sent_c, pos_c)
            hs_i, _ = get_hidden_at_pos(model, tokenizer, device, sent_i, pos_i)
            
            # Get clean margin (from final layer)
            if len(tok_c_ids) > 0 and len(tok_i_ids) > 0:
                clean_margin = float(W_U_f32[tok_c_ids[0]] @ hs_c[n_layers] - 
                                     W_U_f32[tok_i_ids[0]] @ hs_c[n_layers])
            else:
                clean_margin = 0.0
            
            if abs(clean_margin) < 1e-10:
                del hs_c, hs_i
                continue
            
            # Test perturbation recovery at each layer
            layers = get_layers(model)
            inputs_c = tokenizer(sent_c, return_tensors="pt", truncation=True, max_length=128)
            input_ids_c = inputs_c["input_ids"].to(device)
            attn_mask_c = inputs_c["attention_mask"].to(device)
            
            for patch_li in test_layers:
                if patch_li not in hs_c or patch_li not in hs_i:
                    continue
                
                delta_l = hs_c[patch_li] - hs_i[patch_li]
                delta_norm = float(np.linalg.norm(delta_l))
                
                if delta_norm < 1e-10:
                    continue
                
                # Compute perturbation: move h_correct toward h_incorrect
                perturbation = eps * delta_l / delta_norm  # numpy [d_model]
                
                # Use hook to inject perturbation at patch_li
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
                    
                    if len(tok_c_ids) > 0 and len(tok_i_ids) > 0:
                        perturbed_margin = float(
                            out_p.logits[0, -1, tok_c_ids[0]] - out_p.logits[0, -1, tok_i_ids[0]])
                    else:
                        perturbed_margin = 0.0
                    
                    del out_p
                    hook_handle.remove()
                    
                    # Recovery rate
                    recovery = perturbed_margin / clean_margin if abs(clean_margin) > 1e-10 else 0.0
                    recovery_rates[patch_li].append(recovery)
                    
                except Exception as e:
                    if hook_handle:
                        hook_handle.remove()
                    recovery_rates[patch_li].append(0.0)
                finally:
                    if hook_handle:
                        try: hook_handle.remove()
                        except: pass
            
            del hs_c, hs_i, inputs_c
            force_cleanup()
        
        # Aggregate for this epsilon
        eps_result = {}
        for li in sorted(recovery_rates.keys()):
            rates = recovery_rates[li]
            eps_result[li] = {
                "recovery_mean": float(np.mean(rates)),
                "recovery_std": float(np.std(rates)),
                "n_pairs": len(rates),
                "is_attractor": float(np.mean(rates)) > 0.5,
            }
        all_results[f"eps_{eps}"] = eps_result
    
    del W_U_f32
    return all_results


# =====================================================================
# EXP3: JACOBIAN CONSTRAINT DIRECTION
# =====================================================================

def exp3_jacobian_direction(model, tokenizer, device, n_layers, d_model, W_U):
    """
    ★ 因果约束方向分析
    
    因果方向: v_c = W_U[tok_correct] - W_U[tok_incorrect]
    - 这是margin对h的梯度方向
    - 沿此方向移动h, 最大化margin变化
    
    测量:
    - cos(Δ_original, v_c): 原始Δ与因果方向的对齐
    - cos(Δ_counterfactual, v_c): 反事实Δ与因果方向的对齐
    - 差异揭示token嵌入污染
    """
    print("\n" + "="*60)
    print("Exp3: JACOBIAN CONSTRAINT DIRECTION")
    print("="*60)
    
    W_U_f32 = W_U.astype(np.float32)
    
    # --- Original pairs alignment ---
    print("\n  [A] Original pairs alignment...")
    orig_align = defaultdict(list)
    
    for pi, (sent_c, sent_i, verb_c, verb_i) in enumerate(ORIGINAL_PAIRS):
        if pi % 10 == 0:
            print(f"    Pair {pi+1}/{len(ORIGINAL_PAIRS)}", flush=True)
        
        tok_c_ids = tokenizer.encode(verb_c, add_special_tokens=False)
        tok_i_ids = tokenizer.encode(verb_i, add_special_tokens=False)
        if not tok_c_ids or not tok_i_ids:
            continue
        
        # Causal direction: v_c = W_U[tok_c] - W_U[tok_i]
        v_c = W_U_f32[tok_c_ids[0]] - W_U_f32[tok_i_ids[0]]
        v_c_norm = float(np.linalg.norm(v_c))
        if v_c_norm < 1e-10:
            continue
        
        pos_c = find_verb_position(tokenizer, sent_c, verb_c)
        pos_i = find_verb_position(tokenizer, sent_i, verb_i)
        
        hs_c, _ = get_hidden_at_pos(model, tokenizer, device, sent_c, pos_c)
        hs_i, _ = get_hidden_at_pos(model, tokenizer, device, sent_i, pos_i)
        
        for li in range(n_layers + 1):
            if li in hs_c and li in hs_i:
                delta = hs_c[li] - hs_i[li]
                dn = float(np.linalg.norm(delta))
                if dn > 1e-10:
                    cos_align = float(np.dot(delta, v_c) / (dn * v_c_norm))
                    orig_align[li].append(cos_align)
        
        del hs_c, hs_i
        force_cleanup()
    
    # --- Same-token counterfactual alignment ---
    print("\n  [B] Counterfactual pairs alignment...")
    cf_align = defaultdict(list)
    
    all_cf_pairs = SAME_TOKEN_SINGULAR_VERB + SAME_TOKEN_PLURAL_VERB
    
    for pi, (sent_c, sent_i, verb) in enumerate(all_cf_pairs):
        if pi % 10 == 0:
            print(f"    Pair {pi+1}/{len(all_cf_pairs)}", flush=True)
        
        tok_c_ids = tokenizer.encode(verb, add_special_tokens=False)
        # For same-token pairs, we need a different "incorrect" token
        # Singular verb "sleeps" → incorrect form would be "sleep" (or vice versa)
        verb_alt = verb.rstrip("s") if verb.endswith("s") else verb + "s"
        tok_i_ids = tokenizer.encode(verb_alt, add_special_tokens=False)
        if not tok_c_ids or not tok_i_ids:
            continue
        
        v_c = W_U_f32[tok_c_ids[0]] - W_U_f32[tok_i_ids[0]]
        v_c_norm = float(np.linalg.norm(v_c))
        if v_c_norm < 1e-10:
            continue
        
        pos_c = find_verb_position(tokenizer, sent_c, verb)
        pos_i = find_verb_position(tokenizer, sent_i, verb)
        
        hs_c, _ = get_hidden_at_pos(model, tokenizer, device, sent_c, pos_c)
        hs_i, _ = get_hidden_at_pos(model, tokenizer, device, sent_i, pos_i)
        
        for li in range(n_layers + 1):
            if li in hs_c and li in hs_i:
                delta = hs_c[li] - hs_i[li]
                dn = float(np.linalg.norm(delta))
                if dn > 1e-10:
                    cos_align = float(np.dot(delta, v_c) / (dn * v_c_norm))
                    cf_align[li].append(cos_align)
        
        del hs_c, hs_i
        force_cleanup()
    
    # --- Context-only alignment (should be near 0) ---
    print("\n  [C] Context-only alignment (baseline)...")
    ctx_align = defaultdict(list)
    
    for pi, (sent_a, sent_b, verb) in enumerate(CONTEXT_ONLY_PAIRS):
        tok_c_ids = tokenizer.encode(verb, add_special_tokens=False)
        verb_alt = verb.rstrip("s") if verb.endswith("s") else verb + "s"
        tok_i_ids = tokenizer.encode(verb_alt, add_special_tokens=False)
        if not tok_c_ids or not tok_i_ids:
            continue
        
        v_c = W_U_f32[tok_c_ids[0]] - W_U_f32[tok_i_ids[0]]
        v_c_norm = float(np.linalg.norm(v_c))
        if v_c_norm < 1e-10:
            continue
        
        pos_a = find_verb_position(tokenizer, sent_a, verb)
        pos_b = find_verb_position(tokenizer, sent_b, verb)
        
        hs_a, _ = get_hidden_at_pos(model, tokenizer, device, sent_a, pos_a)
        hs_b, _ = get_hidden_at_pos(model, tokenizer, device, sent_b, pos_b)
        
        for li in range(n_layers + 1):
            if li in hs_a and li in hs_b:
                delta = hs_a[li] - hs_b[li]
                dn = float(np.linalg.norm(delta))
                if dn > 1e-10:
                    cos_align = float(np.dot(delta, v_c) / (dn * v_c_norm))
                    ctx_align[li].append(cos_align)
        
        del hs_a, hs_b
        force_cleanup()
    
    # Aggregate
    result = {}
    for li in range(n_layers + 1):
        entry = {}
        if li in orig_align:
            entry["orig_jacobian_alignment_mean"] = float(np.mean(orig_align[li]))
            entry["orig_jacobian_alignment_std"] = float(np.std(orig_align[li]))
        if li in cf_align:
            entry["cf_jacobian_alignment_mean"] = float(np.mean(cf_align[li]))
            entry["cf_jacobian_alignment_std"] = float(np.std(cf_align[li]))
        if li in ctx_align:
            entry["ctx_jacobian_alignment_mean"] = float(np.mean(ctx_align[li]))
        if entry:
            result[li] = entry
    
    del W_U_f32
    return result


# =====================================================================
# MAIN
# =====================================================================

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    t_start = time.time()
    
    print(f"\n{'#'*70}")
    print(f"# Phase 183: COUNTERFACTUAL TRANSPORT GEOMETRY — {model_name}")
    print(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers, d_model, vocab_size = info.n_layers, info.d_model, info.vocab_size
    print(f"\nModel: {info.model_class}, Layers={n_layers}, d_model={d_model}, vocab={vocab_size}")
    
    W_U = get_W_U(model, model_name).astype(np.float32)
    print(f"W_U shape: {W_U.shape}, dtype: {W_U.dtype}")
    
    # ===== Exp1: Counterfactual Analysis =====
    print(f"\n{'='*70}")
    print("Running Exp1: Same-Token Counterfactual Analysis...")
    exp1_results = exp1_counterfactual_analysis(model, tokenizer, device, n_layers, d_model)
    force_cleanup()
    
    # ===== Exp2: Attractor Stability =====
    print(f"\n{'='*70}")
    print("Running Exp2: Attractor Stability Test...")
    exp2_results = exp2_attractor_test(model, tokenizer, device, n_layers, d_model, W_U)
    force_cleanup()
    
    # ===== Exp3: Jacobian Direction =====
    print(f"\n{'='*70}")
    print("Running Exp3: Jacobian Constraint Direction...")
    exp3_results = exp3_jacobian_direction(model, tokenizer, device, n_layers, d_model, W_U)
    force_cleanup()
    
    # ===== Save =====
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = f"tests/glm5_temp/phase183_{model_name}_{timestamp}.json"
    
    full_results = {
        "model": model_name, "n_layers": n_layers, "d_model": d_model, "vocab_size": vocab_size,
        "timestamp": timestamp, "elapsed_sec": round(time.time() - t_start, 1),
        "exp1_counterfactual": {k: {str(kk): vv for kk, vv in v.items()} for k, v in exp1_results.items()},
        "exp2_attractor": exp2_results,
        "exp3_jacobian": {str(k): v for k, v in exp3_results.items()},
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    
    # ===== Summary =====
    print(f"\n{'#'*70}")
    print("PHASE 183 SUMMARY")
    print(f"{'#'*70}")
    
    # Exp1 Summary
    print("\n★★★ Exp1: Counterfactual Analysis ★★★")
    data = exp1_results.get("per_layer", {})
    layers_sorted = sorted([int(k) for k in data.keys()])
    
    # Find token contamination
    tc_values = []
    ci_values = []
    for li in layers_sorted:
        if li in data:
            tc = data[li].get("token_contamination", None)
            ci = data[li].get("constraint_isolation", None)
            if tc is not None:
                tc_values.append((li, tc))
            if ci is not None:
                ci_values.append((li, ci))
    
    if tc_values:
        early_tc = [tc for li, tc in tc_values if li <= n_layers // 3]
        late_tc = [tc for li, tc in tc_values if li > 2 * n_layers // 3]
        print(f"\n  Token contamination (1 - ||Δ_cf||/||Δ_orig||):")
        if early_tc:
            print(f"    Early layers (L0-L{n_layers//3}): mean={np.mean(early_tc):.3f}")
        if late_tc:
            print(f"    Late layers (L{2*n_layers//3+1}-L{n_layers}): mean={np.mean(late_tc):.3f}")
        all_tc = [tc for _, tc in tc_values]
        print(f"    All layers: mean={np.mean(all_tc):.3f}")
        print(f"    ★ {'HIGH contamination' if np.mean(all_tc) > 0.5 else 'MODERATE contamination' if np.mean(all_tc) > 0.2 else 'LOW contamination'}")
        print(f"    ★ Meaning: {'Most of ||Δ|| was token embedding difference!' if np.mean(all_tc) > 0.5 else 'Token embedding is a significant part of Δ' if np.mean(all_tc) > 0.2 else 'Δ captures real constraint signal'}")
    
    if ci_values:
        early_ci = [ci for li, ci in ci_values if li <= n_layers // 3]
        late_ci = [ci for li, ci in ci_values if li > 2 * n_layers // 3]
        print(f"\n  Constraint isolation (Δ_cf/||h|| - Δ_ctx/||h||):")
        if early_ci:
            print(f"    Early layers: mean={np.mean(early_ci):.6f}")
        if late_ci:
            print(f"    Late layers: mean={np.mean(late_ci):.6f}")
        all_ci = [ci for _, ci in ci_values]
        print(f"    All layers: mean={np.mean(all_ci):.6f}")
        print(f"    ★ {'Constraint signal IS separable from context' if np.mean(all_ci) > 0.01 else 'Constraint and context are NOT separable by this method'}")
    
    # Δ relative comparison
    print(f"\n  Δ relative (||Δ||/||h||) comparison at final layer:")
    if n_layers in data:
        d = data[n_layers]
        if "orig_delta_rel_mean" in d:
            print(f"    Original (diff tokens): {d['orig_delta_rel_mean']:.6f}")
        if "cf_delta_rel_mean" in d:
            print(f"    Counterfactual (same token): {d['cf_delta_rel_mean']:.6f}")
        if "ctx_delta_rel_mean" in d:
            print(f"    Context-only (no constraint): {d['ctx_delta_rel_mean']:.6f}")
    
    # Exp2 Summary
    print("\n★★★ Exp2: Attractor Stability ★★★")
    for eps_key in sorted(exp2_results.keys()):
        eps_data = exp2_results[eps_key]
        eps_val = eps_key.replace("eps_", "")
        # Find layers with strongest/weakest attractor
        attractor_layers = []
        non_attractor_layers = []
        for li_str, ld in eps_data.items():
            if ld.get("is_attractor", False):
                attractor_layers.append(int(li_str))
            else:
                non_attractor_layers.append(int(li_str))
        
        print(f"\n  ε={eps_val}:")
        print(f"    Attractor layers (R>0.5): {sorted(attractor_layers)}")
        print(f"    Non-attractor layers (R≤0.5): {sorted(non_attractor_layers)}")
        
        # Print recovery rates for key layers
        for li_str in sorted(eps_data.keys(), key=lambda x: int(x)):
            ld = eps_data[li_str]
            rm = ld.get("recovery_mean", 0)
            tag = "★ATTRACTOR" if ld.get("is_attractor") else "no-attractor"
            if int(li_str) in [1, 2, 3, 5, 10, n_layers//2, n_layers-1]:
                print(f"    L{li_str}: R={rm:.3f} [{tag}]")
    
    # Exp3 Summary
    print("\n★★★ Exp3: Jacobian Constraint Direction ★★★")
    for li in [0, 1, 5, 10, n_layers//2, n_layers-1, n_layers]:
        li_str = str(li)
        if li_str in exp3_results:
            d = exp3_results[li_str]
            orig_a = d.get("orig_jacobian_alignment_mean", None)
            cf_a = d.get("cf_jacobian_alignment_mean", None)
            ctx_a = d.get("ctx_jacobian_alignment_mean", None)
            print(f"  L{li}: orig_alignment={orig_a:.3f}" if orig_a is not None else f"  L{li}: orig=N/A", end="")
            print(f", cf_alignment={cf_a:.3f}" if cf_a is not None else ", cf=N/A", end="")
            print(f", ctx_alignment={ctx_a:.3f}" if ctx_a is not None else ", ctx=N/A")
    
    # Key insight
    cf_align_late = [exp3_results[str(li)].get("cf_jacobian_alignment_mean", 0) 
                     for li in range(2*n_layers//3, n_layers+1) if str(li) in exp3_results]
    orig_align_late = [exp3_results[str(li)].get("orig_jacobian_alignment_mean", 0) 
                       for li in range(2*n_layers//3, n_layers+1) if str(li) in exp3_results]
    if cf_align_late and orig_align_late:
        print(f"\n  Late-layer alignment: orig={np.mean(orig_align_late):.3f}, cf={np.mean(cf_align_late):.3f}")
        print(f"  ★ {'Counterfactual Δ is MORE aligned with causal direction → genuine constraint signal!' if np.mean(cf_align_late) > np.mean(orig_align_late) else 'Original Δ has higher alignment → token embedding contamination includes causal direction'}")
    
    release_model(model)
    elapsed = time.time() - t_start
    print(f"\n{'#'*70}")
    print(f"Phase 183 COMPLETE! Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
