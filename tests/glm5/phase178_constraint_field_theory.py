"""
Phase 178: ★★★ 约束场理论 — 从观察到方程 ★★★
=================================================

用户核心洞察 (完全正确):
  1. 约束必须满足: 可验证/可传播/可组合/可闭合
  2. Transformer本质是约束满足引擎: P(x_t|x_{<t}) = 约束满足后的残余自由度
  3. 概念是约束流形/吸引盆, 不是向量
  4. 应研究 C_l(pos) 约束状态场, 不是 h_l 隐藏状态
  5. 能量应该是约束违反惩罚: E_C = -margin_C

Phase 177 → Phase 178 关键升级:
  ★ 不再在LAST TOKEN测量 → 在PRE-VERB POSITION测量
  ★ 不再用通用词表定义信号 → 用具体动词token对定义信号
  ★ 不只观察violation → 测试传播动力学 C_{l+1} = F(C_l)
  ★ 不用ad hoc能量 → 用proper constraint violation penalty
  ★ 不只测单个约束 → 测试约束组合和闭合

★★★ 内存优化 (v2) ★★★
  旧版: W_U.astype(np.float64) @ h → 全词表logits → ~5GB内存 → OOM卡死
  新版: (W_U[sg_id] - W_U[pl_id]) · h → 2个向量点积 → <1KB
  关键: 只计算需要的2个token的logit差值，不计算全词表

★★★ 五大实验 ★★★

Phase A: 约束传播动力学 (C_{l+1} = F_l(C_l))
  - 核心问题: 能否从层l的约束状态预测层l+1的约束状态?
  - 方法: 在pre-verb位置测量σ_N跨所有层, 拟合线性模型
  - 如果R²高 → 约束传播是确定性动力学
  - 如果R²低 → 存在非线性/随机效应

Phase B: 正确的约束违反能量
  - 定义: margin_C(l) = expected_sign * (logit(correct_form) - logit(wrong_form))
  - E_C(l) = -margin_C(l)
  - 验证: dE/dl < 0 for correct sentences → 约束闭合
  - 验证: E(correct) < E(violated) → 能量景观区分

Phase C: 约束状态场 C_l(pos) — 空间传播
  - 在每个(layer, position)对测量约束信号
  - 追踪: 约束信息如何从subject位置传播到verb位置?
  - 这是"约束场"的空间结构

Phase D: 约束组合
  - 句子同时有number + gender约束
  - 测试: E_total ≈ w1*E_number + w2*E_gender?
  - 如果是 → 约束可组合 (composable)

Phase E: 约束闭合
  - 在每层计算: 有多少约束被满足?
  - closure(l) = fraction of constraints satisfied at layer l
  - 验证: closure increases with depth → 约束逐层闭合

Usage: python tests/glm5/phase178_constraint_field_theory.py <model_name>
  model_name: qwen3, glm4, deepseek7b
"""

import sys
import os
import time
import json
import gc
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'glm5'))

from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS


# =====================================================================
# MODEL LOADING (BF16 + device_map="auto")
# =====================================================================

def load_model_bf16(model_name):
    """BF16 + device_map=auto loading for all models"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[bf16] Loading {model_name} (bfloat16 + device_map=auto)...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[bf16] {model_name} loaded: GPU={gpu_mem:.2f}GB", flush=True)

    return model, tokenizer, device


# =====================================================================
# ★★★ 句子模板 — 精确控制subject/verb位置 ★★★
# =====================================================================

# 动词对: (singular_form, plural_form)
VERB_PAIRS = [
    ("sleeps", "sleep"), ("runs", "run"), ("walks", "walk"),
    ("flies", "fly"), ("swims", "swim"), ("jumps", "jump"),
    ("sings", "sing"), ("reads", "read"), ("writes", "write"),
    ("eats", "eat"), ("drinks", "drink"), ("thinks", "think"),
    ("knows", "know"), ("grows", "grow"), ("moves", "move"),
    ("works", "work"), ("plays", "play"), ("talks", "talk"),
    ("sits", "sit"), ("stands", "stand"), ("falls", "fall"),
    ("rises", "rise"), ("shines", "shine"), ("drifts", "drift"),
    ("breaks", "break"), ("drives", "drive"), ("hides", "hide"),
    ("cries", "cry"), ("smiles", "smile"), ("waits", "wait"),
]

# 名词对: (singular, plural)
NOUN_PAIRS = [
    ("cat", "cats"), ("dog", "dogs"), ("bird", "birds"),
    ("child", "children"), ("man", "men"), ("woman", "women"),
    ("horse", "horses"), ("student", "students"), ("teacher", "teachers"),
    ("flower", "flowers"), ("river", "rivers"), ("cloud", "clouds"),
    ("lamp", "lamps"), ("clock", "clocks"), ("bell", "bells"),
    ("door", "doors"), ("tree", "trees"), ("book", "books"),
    ("car", "cars"), ("star", "stars"),
]

# Gender agreement templates
GENDER_TEMPLATES = [
    ("actor", "actress", "he", "she"),
    ("king", "queen", "he", "she"),
    ("boy", "girl", "he", "she"),
    ("man", "woman", "he", "she"),
    ("father", "mother", "he", "she"),
    ("brother", "sister", "he", "she"),
    ("uncle", "aunt", "he", "she"),
    ("hero", "heroine", "he", "she"),
    ("prince", "princess", "he", "she"),
    ("gentleman", "lady", "he", "she"),
    ("monk", "nun", "he", "she"),
    ("knight", "witch", "he", "she"),
    ("wizard", "witch", "he", "she"),
    ("emperor", "empress", "he", "she"),
    ("soldier", "nurse", "he", "she"),
]

# Dual constraint templates (number + gender)
DUAL_CONSTRAINT_TEMPLATES = [
    ("The actor sleeps", "The actors sleep", "The actress sleeps", "The actresses sleep"),
    ("The king runs", "The kings run", "The queen runs", "The queens run"),
    ("The boy walks", "The boys walk", "The girl walks", "The girls walk"),
    ("The man sings", "The men sing", "The woman sings", "The women sing"),
    ("The father reads", "The fathers read", "The mother reads", "The mothers read"),
    ("The prince waits", "The princes wait", "The princess waits", "The princesses wait"),
    ("The hero stands", "The heroes stand", "The heroine stands", "The heroines stand"),
    ("The uncle smiles", "The uncles smile", "The aunt smiles", "The aunts smile"),
    ("The brother cries", "The brothers cry", "The sister cries", "The sisters cry"),
    ("The gentleman talks", "The gentlemen talk", "The lady talks", "The ladies talk"),
    ("The emperor thinks", "The emperors think", "The empress thinks", "The empresses think"),
    ("The soldier works", "The soldiers work", "The nurse works", "The nurses work"),
]


# =====================================================================
# ★★★ 核心工具函数 — 内存优化版 ★★★
# =====================================================================

def get_final_norm_weight(model):
    """Get the final layer norm weight from the model (handles meta device offload)"""
    norm = None
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        norm = model.model.norm
    elif hasattr(model, 'model') and hasattr(model.model, 'final_layernorm'):
        norm = model.model.final_layernorm
    
    if norm is None:
        print("  [WARN] Could not find final norm, using identity", flush=True)
        return None
    
    w = norm.weight
    if not w.is_meta:
        return w.detach().cpu().float().numpy()
    
    # Weight is on meta device (offloaded), load from safetensors
    print("  [INFO] Final norm on meta device, loading from safetensors...", flush=True)
    try:
        import glob, os
        from safetensors import safe_open
        model_path = None
        # Try to get model path from config
        if hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
            model_path = model.config._name_or_path
        if model_path is None:
            print("  [WARN] Cannot find model path for safetensors, using identity norm", flush=True)
            return None
        
        sf_files = glob.glob(os.path.join(model_path, '*.safetensors'))
        norm_key = None
        for name in ['model.norm.weight', 'model.final_layernorm.weight']:
            for sf_file in sf_files:
                with safe_open(sf_file, framework='pt', device='cpu') as sf:
                    if name in sf.keys():
                        w = sf.get_tensor(name)
                        print(f"  [INFO] Loaded norm from {os.path.basename(sf_file)}, key={name}", flush=True)
                        return w.float().numpy()
        
        print("  [WARN] Could not find norm weight in safetensors, using identity", flush=True)
        return None
    except Exception as e:
        print(f"  [WARN] Failed to load norm from safetensors: {e}, using identity", flush=True)
        return None


def get_token_id(tokenizer, word):
    """Get the token ID for a word (first token if multi-token)"""
    ids = tokenizer.encode(word, add_special_tokens=False)
    return ids[0] if ids else None


def compute_margin_from_h(h, w_row_a, w_row_b, norm_weight=None, eps=1e-6):
    """
    ★★★ 直接计算2个token的logit差值, 无需全词表 ★★★
    
    margin = logit(token_a) - logit(token_b)
           = (W_U[a] · h_normed) - (W_U[b] · h_normed)
           = (W_U[a] - W_U[b]) · h_normed
    
    内存: 1次向量点积 ~ d_model*4 bytes (~10KB)
    而非: W_U @ h ~ vocab*d_model*8 bytes (~5GB)
    
    Args:
        h: hidden state [d_model] float16/float32 numpy
        w_row_a: W_U[token_a_id] float32 numpy [d_model]
        w_row_b: W_U[token_b_id] float32 numpy [d_model]
        norm_weight: optional RMSNorm weight [d_model]
    
    Returns:
        float: margin value
    """
    h_float = h.astype(np.float32)
    if norm_weight is not None:
        rms = np.sqrt(np.mean(h_float.astype(np.float64) ** 2) + eps)
        h_normed = (h_float / np.float32(rms)) * norm_weight.astype(np.float32)
    else:
        h_normed = h_float
    
    # margin = (w_a - w_b) · h_normed
    diff = w_row_a.astype(np.float32) - w_row_b.astype(np.float32)
    margin = float(np.dot(diff, h_normed))
    
    if not np.isfinite(margin):
        return 0.0
    return margin


def get_hidden_states(model, tokenizer, sentence, positions, n_layers):
    """
    前向传播, 返回指定位置的hidden states (float32)
    
    ★★★ 不计算全词表logits! 只返回hidden states ★★★
    
    Args:
        positions: dict of {name: pos_index} — 基于tokenizer(sentence)的position,
                   即包含special tokens的position
    
    Returns:
        dict: {pos_name: {layer_idx: h_array_float32}}
    """
    input_device = next(model.parameters()).device
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attn_mask = inputs["attention_mask"].to(input_device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
    
    hs = out.hidden_states  # tuple of (n_layers+1) tensors [1, seq_len, d_model]
    
    result = {}
    for pos_name, pos_idx in positions.items():
        if pos_idx >= hs[0].shape[1]:
            continue
        pos_hs = {}
        for li in range(min(n_layers + 1, len(hs))):
            h = hs[li][0, pos_idx].float().cpu().numpy()  # float32, ~10KB per layer
            pos_hs[li] = h
        result[pos_name] = pos_hs
    
    return result


def find_subject_position(tokenizer, sentence):
    """
    找到subject token在tokenizer(sentence)输出中的position
    
    处理不同tokenizer的BOS token差异:
    - Qwen3: tokenizer("The cat") = [785, 8251] → subject at pos 1
    - GLM4:  tokenizer("The cat") = [151331, 151333, 785, 8250] → subject at pos 3
    
    Returns:
        int: position of the subject token in the tokenized sequence
    """
    full_ids = tokenizer.encode(sentence)
    no_special_ids = tokenizer.encode(sentence, add_special_tokens=False)
    
    if len(full_ids) == len(no_special_ids):
        # No special tokens added (Qwen3)
        return 1  # "The" at 0, subject at 1
    else:
        # Special tokens added (GLM4 has 2 BOS tokens)
        n_special = len(full_ids) - len(no_special_ids)
        return n_special + 1  # skip BOS tokens + "The" token


# =====================================================================
# Phase A: ★★★ 约束传播动力学 (C_{l+1} = F_l(C_l)) ★★★
# =====================================================================

def run_constraint_propagation(model, tokenizer, device, model_info, W_U, norm_weight):
    """
    ★★★ 核心实验: 约束状态在层间如何传播? ★★★
    
    大数据量: 80+ 句子对 (40 singular + 40 plural subjects)
    """
    n_layers = model_info.n_layers
    
    print("\n" + "="*70, flush=True)
    print("Phase A: ★★★ 约束传播动力学 (C_{l+1} = F_l(C_l)) ★★★", flush=True)
    print("="*70, flush=True)
    
    # ★★★ Generate sentence pairs ★★★
    # KEY INSIGHT: 在subject位置(pos=1)测量，模型预测下一个token(动词)
    # 所以正确的对比是: singular subject vs plural subject → 对同一个动词对的margin
    # "The cat sleeps" pos1=" cat" → margin应该>0 (偏好" sleeps")
    # "The cats sleep" pos1=" cats" → margin应该<0 (偏好" sleep")
    # 正确/错误句子在pos1有相同的subject，所以margin相同 → 不是bug，是设计问题
    # 我们只需要: sg_subj句子 vs pl_subj句子
    
    sentences = []  # (sentence, verb_sg, verb_pl, expected_sign, sentence_type)
    
    for i, (noun_sg, noun_pl) in enumerate(NOUN_PAIRS):
        verb_sg, verb_pl = VERB_PAIRS[i % len(VERB_PAIRS)]
        # Singular subject + singular verb (correct)
        sentences.append((f"The {noun_sg} {verb_sg}", verb_sg, verb_pl, +1, "sg_subj"))
        # Plural subject + plural verb (correct)
        sentences.append((f"The {noun_pl} {verb_pl}", verb_sg, verb_pl, -1, "pl_subj"))
    
    print(f"  Total sentences: {len(sentences)}", flush=True)
    
    # ★★★ Pre-extract W_U rows — 必须用带空格的版本! ★★★
    # " sleeps" (1 token=71390) vs "sleeps" (2 tokens=sleep+s)
    # 在句子中，动词前面有空格，所以logit预测的是带空格的token
    verb_pair_rows = {}
    for verb_sg, verb_pl in VERB_PAIRS:
        # 优先尝试带空格版本（句子中动词前有空格）
        sg_ids = tokenizer.encode(" " + verb_sg, add_special_tokens=False)
        pl_ids = tokenizer.encode(" " + verb_pl, add_special_tokens=False)
        
        # 只接受单token的版本（多token的动词对不可靠）
        if len(sg_ids) == 1 and len(pl_ids) == 1:
            sg_id, pl_id = sg_ids[0], pl_ids[0]
            verb_pair_rows[(verb_sg, verb_pl)] = {
                "sg_id": sg_id, "pl_id": pl_id,
                "w_sg": W_U[sg_id].astype(np.float32),
                "w_pl": W_U[pl_id].astype(np.float32),
            }
    
    print(f"  Valid verb pairs: {len(verb_pair_rows)}/{len(VERB_PAIRS)}", flush=True)
    
    all_signals = []
    
    for idx, (sent, v_sg, v_pl, exp_sign, stype) in enumerate(sentences):
        toks = tokenizer.encode(sent, add_special_tokens=False)
        
        if len(toks) < 3:
            continue
        
        subj_pos = find_subject_position(tokenizer, sent)
        positions = {"pre_verb": subj_pos}
        
        try:
            pos_hs = get_hidden_states(model, tokenizer, sent, positions, n_layers)
        except Exception as e:
            continue
        
        if "pre_verb" not in pos_hs:
            continue
        
        # Get W_U rows for this verb pair
        vkey = (v_sg, v_pl)
        if vkey not in verb_pair_rows:
            continue
        rows = verb_pair_rows[vkey]
        
        # Compute number margin at each layer using vector dot product
        layer_margins = {}
        for li, h in pos_hs["pre_verb"].items():
            margin = compute_margin_from_h(h, rows["w_sg"], rows["w_pl"], norm_weight)
            layer_margins[li] = margin
        
        all_signals.append({
            "type": stype,
            "expected_sign": exp_sign,
            "margins": layer_margins,
            "sentence": sent,
        })
        
        if (idx + 1) % 20 == 0:
            print(f"    Processed {idx+1}/{len(sentences)} sentences", flush=True)
    
    print(f"  Successfully processed: {len(all_signals)} sentences", flush=True)
    
    # ====== Analysis ======
    
    # 1. Average margin by type at each layer
    avg_margins = defaultdict(lambda: defaultdict(list))
    for sig in all_signals:
        for li, m in sig["margins"].items():
            avg_margins[sig["type"]][li].append(m)
    
    avg_by_type = {}
    for stype, layer_data in avg_margins.items():
        avg_by_type[stype] = {str(li): round(float(np.mean(vals)), 4) 
                              for li, vals in layer_data.items()}
    
    # 2. ★★★ Violation signal: sg_subj - pl_subj at each layer ★★★
    # sg_subj margin should be > 0 (model prefers singular verb)
    # pl_subj margin should be < 0 (model prefers plural verb)
    # violation = margin(sg_subj) - margin(pl_subj) should be > 0
    violation_signal = {}
    for li in range(n_layers + 1):
        li_str = str(li)
        sg_margin = avg_by_type.get("sg_subj", {}).get(li_str, 0)
        pl_margin = avg_by_type.get("pl_subj", {}).get(li_str, 0)
        violation_signal[li] = sg_margin - pl_margin
    
    # 3. ★★★ Propagation dynamics: σ_N(l+1) = A * σ_N(l) + b ★★★
    r_squared_values = []
    propagation_coeffs = []
    
    for sig in all_signals:
        margins = sig["margins"]
        layers_sorted = sorted(margins.keys())
        
        if len(layers_sorted) < 3:
            continue
        
        X = []
        Y = []
        for i in range(len(layers_sorted) - 1):
            X.append(margins[layers_sorted[i]])
            Y.append(margins[layers_sorted[i + 1]])
        
        if len(X) < 2:
            continue
        
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
            continue
        if np.std(X) < 1e-10 or np.std(Y) < 1e-10:
            r_squared_values.append(1.0)
            propagation_coeffs.append({"A": 1.0, "b": 0.0, "R2": 1.0})
            continue
        
        try:
            A, b = np.polyfit(X, Y, 1)
            Y_pred = A * X + b
            ss_res = np.sum((Y - Y_pred) ** 2)
            ss_tot = np.sum((Y - np.mean(Y)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            
            if np.isfinite(r2):
                r_squared_values.append(max(0, min(1, r2)))
                propagation_coeffs.append({"A": round(float(A), 4), "b": round(float(b), 4), "R2": round(float(r2), 4)})
        except (np.linalg.LinAlgError, ValueError):
            continue
    
    avg_r2 = float(np.mean(r_squared_values)) if r_squared_values else 0
    median_r2 = float(np.median(r_squared_values)) if r_squared_values else 0
    
    print(f"\n  ★★★ Propagation Dynamics Results ★★★", flush=True)
    print(f"    Avg R² = {avg_r2:.4f}, Median R² = {median_r2:.4f}", flush=True)
    print(f"    → {'Linear propagation confirmed!' if avg_r2 > 0.9 else 'Non-linear dynamics detected!' if avg_r2 < 0.5 else 'Partially linear propagation'}", flush=True)
    
    print(f"\n  Violation signal at key layers:", flush=True)
    for li in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]:
        sg_m = avg_by_type.get("sg_subj", {}).get(str(li), 0)
        pl_m = avg_by_type.get("pl_subj", {}).get(str(li), 0)
        vs = violation_signal.get(li, 0)
        print(f"    L{li}: sg_margin={sg_m:.4f}, pl_margin={pl_m:.4f}, violation={vs:.4f}", flush=True)
    
    # Constraint enforcement onset layer
    threshold = 1.0
    onset_layer = None
    for li in sorted(violation_signal.keys()):
        if abs(violation_signal.get(li, 0)) > threshold:
            onset_layer = li
            break
    print(f"\n  Constraint enforcement onset (|violation|>{threshold}): L{onset_layer}", flush=True)
    
    return {
        "n_sentences": len(all_signals),
        "avg_margins_by_type": avg_by_type,
        "violation_signal": {str(li): round(v, 4) for li, v in violation_signal.items()},
        "propagation_r2_avg": round(avg_r2, 4),
        "propagation_r2_median": round(median_r2, 4),
        "propagation_r2_distribution": [round(r, 4) for r in sorted(r_squared_values)],
        "propagation_coeffs_sample": propagation_coeffs[:10],
        "constraint_onset_layer": onset_layer,
    }


# =====================================================================
# Phase B: ★★★ 正确的约束违反能量 ★★★
# =====================================================================

def run_proper_energy(model, tokenizer, device, model_info, W_U, norm_weight):
    """
    ★★★ 正确的能量定义: E_C(l) = -margin_C(l) ★★★
    
    约束满足 → margin > 0 → E < 0 (低能量)
    约束违反 → margin < 0 → E > 0 (高能量)
    """
    n_layers = model_info.n_layers
    
    print("\n" + "="*70, flush=True)
    print("Phase B: ★★★ 正确的约束违反能量 ★★★", flush=True)
    print("="*70, flush=True)
    
    # Pre-extract W_U rows for all verb pairs (带空格版本!)
    verb_pair_rows = {}
    for verb_sg, verb_pl in VERB_PAIRS:
        sg_ids = tokenizer.encode(" " + verb_sg, add_special_tokens=False)
        pl_ids = tokenizer.encode(" " + verb_pl, add_special_tokens=False)
        if len(sg_ids) == 1 and len(pl_ids) == 1:
            verb_pair_rows[(verb_sg, verb_pl)] = {
                "w_sg": W_U[sg_ids[0]].astype(np.float32),
                "w_pl": W_U[pl_ids[0]].astype(np.float32),
            }
    
    test_pairs = []
    for i, (noun_sg, noun_pl) in enumerate(NOUN_PAIRS[:15]):
        verb_sg, verb_pl = VERB_PAIRS[i % len(VERB_PAIRS)]
        # ★★★ 只需要正确句子（在subject位置测量） ★★★
        # sg_subj → expected_sign=+1 (margin should be > 0)
        # pl_subj → expected_sign=-1 (margin should be < 0)
        test_pairs.append({
            "sentence": f"The {noun_sg} {verb_sg}",
            "verb_sg": verb_sg, "verb_pl": verb_pl,
            "expected_sign": +1, "type": "sg_subj",
        })
        test_pairs.append({
            "sentence": f"The {noun_pl} {verb_pl}",
            "verb_sg": verb_sg, "verb_pl": verb_pl,
            "expected_sign": -1, "type": "pl_subj",
        })
    
    print(f"  Test pairs: {len(test_pairs)}", flush=True)
    
    energy_profiles = []
    
    for pair in test_pairs:
        vkey = (pair["verb_sg"], pair["verb_pl"])
        if vkey not in verb_pair_rows:
            continue
        rows = verb_pair_rows[vkey]
        
        sent = pair["sentence"]
        subj_pos = find_subject_position(tokenizer, sent)
        positions = {"pre_verb": subj_pos}
        
        try:
            pos_hs = get_hidden_states(model, tokenizer, sent, positions, n_layers)
        except:
            continue
        
        if "pre_verb" not in pos_hs:
            continue
        
        layer_energies = {}
        for li, h in pos_hs["pre_verb"].items():
            raw_margin = compute_margin_from_h(h, rows["w_sg"], rows["w_pl"], norm_weight)
            proper_margin = pair["expected_sign"] * raw_margin
            energy = -proper_margin
            layer_energies[li] = round(float(energy), 4)
        
        energy_profiles.append({
            "type": pair["type"],
            "energies": layer_energies,
            "expected_sign": pair["expected_sign"],
        })
    
    # ====== Analysis ======
    # Group by type: sg_subj (E should be negative=low energy) vs pl_subj (E should be negative too)
    sg_energies = defaultdict(list)
    pl_energies = defaultdict(list)
    
    for prof in energy_profiles:
        for li, e in prof["energies"].items():
            if prof["type"] == "sg_subj":
                sg_energies[li].append(e)
            else:
                pl_energies[li].append(e)
    
    avg_sg_E = {li: float(np.mean(vals)) for li, vals in sg_energies.items()}
    avg_pl_E = {li: float(np.mean(vals)) for li, vals in pl_energies.items()}
    
    # Proper margin: E = -expected_sign * raw_margin
    # If constraint satisfied: E < 0 for both sg and pl
    # Average energy across both types (both should be negative if constraint satisfied)
    all_energies = defaultdict(list)
    for prof in energy_profiles:
        for li, e in prof["energies"].items():
            all_energies[li].append(e)
    avg_E = {li: float(np.mean(vals)) for li, vals in all_energies.items()}
    
    # Energy derivative: dE/dl for all correct sentences
    dEdl = {}
    sorted_layers = sorted(avg_E.keys())
    for i in range(1, len(sorted_layers)):
        li_prev = sorted_layers[i-1]
        li_curr = sorted_layers[i]
        dE = avg_E[li_curr] - avg_E[li_prev]
        dl = li_curr - li_prev
        dEdl[li_curr] = round(dE / dl, 4)
    
    # Fraction of layers where E decreases (constraint closure)
    closure_fraction = sum(1 for v in dEdl.values() if v < 0) / max(len(dEdl), 1)
    
    print(f"\n  ★★★ Energy Landscape Results ★★★", flush=True)
    print(f"    Avg E(sg_subj, L_last) = {avg_sg_E.get(n_layers, 0):.4f}", flush=True)
    print(f"    Avg E(pl_subj, L_last) = {avg_pl_E.get(n_layers, 0):.4f}", flush=True)
    print(f"    Avg E(all, L_last) = {avg_E.get(n_layers, 0):.4f}", flush=True)
    print(f"    Closure fraction (dE/dl<0): {closure_fraction:.2%}", flush=True)
    
    for li in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]:
        e_sg = avg_sg_E.get(li, 0)
        e_pl = avg_pl_E.get(li, 0)
        e_all = avg_E.get(li, 0)
        print(f"    L{li}: E_sg={e_sg:.4f}, E_pl={e_pl:.4f}, E_avg={e_all:.4f}", flush=True)
    
    return {
        "avg_sg_energy": {str(li): round(v, 4) for li, v in avg_sg_E.items()},
        "avg_pl_energy": {str(li): round(v, 4) for li, v in avg_pl_E.items()},
        "avg_energy": {str(li): round(v, 4) for li, v in avg_E.items()},
        "dEdl": {str(li): v for li, v in dEdl.items()},
        "closure_fraction": round(closure_fraction, 4),
        "n_profiles": len(energy_profiles),
    }


# =====================================================================
# Phase C: ★★★ 约束状态场 C_l(pos) — 空间传播 ★★★
# =====================================================================

def run_constraint_field(model, tokenizer, device, model_info, W_U, norm_weight):
    """
    ★★★ 约束场: C_l(pos) 在 (layer, position) 空间中的结构 ★★★
    """
    n_layers = model_info.n_layers
    
    print("\n" + "="*70, flush=True)
    print("Phase C: ★★★ 约束状态场 C_l(pos) — 空间传播 ★★★", flush=True)
    print("="*70, flush=True)
    
    # Pre-extract W_U rows (带空格版本!)
    verb_pair_rows = {}
    for verb_sg, verb_pl in VERB_PAIRS[:5]:
        sg_ids = tokenizer.encode(" " + verb_sg, add_special_tokens=False)
        pl_ids = tokenizer.encode(" " + verb_pl, add_special_tokens=False)
        if len(sg_ids) == 1 and len(pl_ids) == 1:
            verb_pair_rows[(verb_sg, verb_pl)] = {
                "w_sg": W_U[sg_ids[0]].astype(np.float32),
                "w_pl": W_U[pl_ids[0]].astype(np.float32),
            }
    
    # ★★★ Field sentences — position_info现在只记录语义角色 ★★★
    # 实际position索引会在运行时根据tokenizer动态计算
    field_sentences = [
        ("The cat sleeps", "sleeps", "sleep", +1, ["The", "cat(subj)", "sleeps(verb)"]),
        ("The cats sleep", "sleeps", "sleep", -1, ["The", "cats(subj)", "sleep(verb)"]),
        ("The dog runs", "runs", "run", +1, ["The", "dog(subj)", "runs(verb)"]),
        ("The dogs run", "runs", "run", -1, ["The", "dogs(subj)", "run(verb)"]),
        ("The beautiful cat sleeps", "sleeps", "sleep", +1, ["The", "beautiful", "cat(subj)", "sleeps(verb)"]),
        ("The beautiful cats sleep", "sleeps", "sleep", -1, ["The", "beautiful", "cats(subj)", "sleep(verb)"]),
        ("The cat near the dogs sleeps", "sleeps", "sleep", +1, ["The", "cat(subj)", "near", "the", "dogs(attr)", "sleeps(verb)"]),
        ("The cat near the dogs sleep", "sleeps", "sleep", +1, ["The", "cat(subj)", "near", "the", "dogs(attr)", "sleep(verb)"]),
    ]
    
    field_results = {}
    
    for sent, v_sg, v_pl, exp_sign, role_labels in field_sentences:
        vkey = (v_sg, v_pl)
        if vkey not in verb_pair_rows:
            continue
        rows = verb_pair_rows[vkey]
        
        # Compute actual positions including BOS tokens
        full_ids = tokenizer.encode(sent)
        no_special_ids = tokenizer.encode(sent, add_special_tokens=False)
        n_special = len(full_ids) - len(no_special_ids)
        n_positions = len(full_ids)
        
        # Map positions: pos0..posN in full tokenized sequence
        positions = {f"pos{p}": p for p in range(n_positions)}
        
        # Build role→position mapping
        role_to_pos = {}
        for i, role in enumerate(role_labels):
            role_to_pos[role] = n_special + i
        
        try:
            pos_hs = get_hidden_states(model, tokenizer, sent, positions, n_layers)
        except:
            continue
        
        # Compute number margin at each (layer, position)
        field = {}
        for pos_name, layer_hs in pos_hs.items():
            pos_field = {}
            for li, h in layer_hs.items():
                margin = compute_margin_from_h(h, rows["w_sg"], rows["w_pl"], norm_weight)
                pos_field[li] = round(margin, 4)
            field[pos_name] = pos_field
        
        # Find the "constraint onset" at each position
        onset = {}
        for pos_name, pos_field in field.items():
            for li in sorted(pos_field.keys()):
                if abs(pos_field[li]) > 0.5:
                    onset[pos_name] = li
                    break
        
        # Find subj/verb onset using role mapping
        subj_onset = None
        verb_onset = None
        for role, pos in role_to_pos.items():
            pos_name = f"pos{pos}"
            if "subj" in role and pos_name in onset:
                subj_onset = onset[pos_name]
            if "verb" in role and pos_name in onset:
                verb_onset = onset[pos_name]
        
        field_results[sent[:30]] = {
            "sentence": sent,
            "expected_sign": exp_sign,
            "n_positions": n_positions,
            "role_labels": role_labels,
            "n_special_tokens": n_special,
            "field": {pos: {str(li): v for li, v in pos_field.items()} 
                     for pos, pos_field in field.items()},
            "onset_layers": onset,
            "subj_onset": subj_onset,
            "verb_onset": verb_onset,
            "propagation_delay": (verb_onset - subj_onset) if (subj_onset is not None and verb_onset is not None) else None,
        }
        
        print(f"\n  '{sent}'", flush=True)
        print(f"    Positions: {n_positions}, Subject onset: L{subj_onset}, Verb onset: L{verb_onset}", flush=True)
        if subj_onset is not None and verb_onset is not None:
            print(f"    Propagation delay: {verb_onset - subj_onset} layers", flush=True)
    
    delays = [r["propagation_delay"] for r in field_results.values() 
              if r["propagation_delay"] is not None]
    avg_delay = float(np.mean(delays)) if delays else None
    
    print(f"\n  ★★★ Field Propagation Summary ★★★", flush=True)
    print(f"    Avg propagation delay (subj→verb): {avg_delay} layers" if avg_delay is not None else "    Could not compute propagation delay", flush=True)
    
    return {
        "field_results": field_results,
        "avg_propagation_delay": round(avg_delay, 2) if avg_delay is not None else None,
    }


# =====================================================================
# Phase D: ★★★ 约束组合 ★★★
# =====================================================================

def run_constraint_composition(model, tokenizer, device, model_info, W_U, norm_weight):
    """
    ★★★ 约束组合: 多个约束是否可加性组合? ★★★
    
    E_total ≈ w1*E_number + w2*E_gender ?
    """
    n_layers = model_info.n_layers
    
    print("\n" + "="*70, flush=True)
    print("Phase D: ★★★ 约束组合 ★★★", flush=True)
    print("="*70, flush=True)
    
    # Pre-extract W_U rows for number and gender (带空格版本!)
    verb_pair_rows = {}
    for verb_sg, verb_pl in VERB_PAIRS[:10]:
        sg_ids = tokenizer.encode(" " + verb_sg, add_special_tokens=False)
        pl_ids = tokenizer.encode(" " + verb_pl, add_special_tokens=False)
        if len(sg_ids) == 1 and len(pl_ids) == 1:
            verb_pair_rows[(verb_sg, verb_pl)] = {
                "w_sg": W_U[sg_ids[0]].astype(np.float32),
                "w_pl": W_U[pl_ids[0]].astype(np.float32),
            }
    
    # Gender rows (带空格!)
    he_ids = tokenizer.encode(" he", add_special_tokens=False)
    she_ids = tokenizer.encode(" she", add_special_tokens=False)
    w_he = W_U[he_ids[0]].astype(np.float32) if len(he_ids) == 1 else None
    w_she = W_U[she_ids[0]].astype(np.float32) if len(she_ids) == 1 else None
    
    composition_data = []
    
    for masc_sg, masc_pl, fem_sg, fem_pl in DUAL_CONSTRAINT_TEMPLATES[:10]:
        words_sg = masc_sg.split()
        verb_sg_word = words_sg[-1]
        words_pl = masc_pl.split()
        verb_pl_word = words_pl[-1]
        
        vkey = (verb_sg_word, verb_pl_word)
        if vkey not in verb_pair_rows or w_he is None or w_she is None:
            continue
        rows = verb_pair_rows[vkey]
        
        conditions = [
            ("both_correct", masc_sg, +1, +1),
            ("number_wrong", masc_sg.replace(verb_sg_word, verb_pl_word), +1, +1),
            ("gender_wrong", fem_sg, +1, -1),
            ("both_wrong", fem_sg.replace(verb_sg_word, verb_pl_word), +1, -1),
        ]
        
        for cond_name, sent, num_sign, gen_sign in conditions:
            subj_pos = find_subject_position(tokenizer, sent)
            positions = {"pre_verb": subj_pos}
            
            try:
                pos_hs = get_hidden_states(model, tokenizer, sent, positions, n_layers)
            except:
                continue
            
            if "pre_verb" not in pos_hs:
                continue
            
            layer_margins_num = {}
            layer_margins_gen = {}
            
            for li, h in pos_hs["pre_verb"].items():
                raw_num = compute_margin_from_h(h, rows["w_sg"], rows["w_pl"], norm_weight)
                proper_num = num_sign * raw_num
                
                raw_gen = compute_margin_from_h(h, w_he, w_she, norm_weight)
                proper_gen = gen_sign * raw_gen
                
                layer_margins_num[li] = proper_num
                layer_margins_gen[li] = proper_gen
            
            composition_data.append({
                "condition": cond_name,
                "sentence": sent,
                "number_margins": layer_margins_num,
                "gender_margins": layer_margins_gen,
                "number_energy": {li: round(-m, 4) for li, m in layer_margins_num.items()},
                "gender_energy": {li: round(-m, 4) for li, m in layer_margins_gen.items()},
                "total_energy": {li: round(-(layer_margins_num.get(li, 0) + layer_margins_gen.get(li, 0)), 4) 
                                for li in layer_margins_num.keys()},
            })
    
    # ====== Analysis ======
    additivity_errors = defaultdict(list)
    
    for datum in composition_data:
        for li in datum["number_energy"].keys():
            e_num = datum["number_energy"].get(li, 0)
            e_gen = datum["gender_energy"].get(li, 0)
            e_total = datum["total_energy"].get(li, 0)
            e_sum = e_num + e_gen
            
            if abs(e_sum) > 0.01:
                rel_error = abs(e_total - e_sum) / max(abs(e_sum), 0.01)
                additivity_errors[li].append(rel_error)
    
    avg_additivity_error = {li: round(float(np.mean(errs)), 4) 
                           for li, errs in additivity_errors.items() if errs}
    
    condition_energies = defaultdict(lambda: defaultdict(list))
    for datum in composition_data:
        for li, e in datum["total_energy"].items():
            condition_energies[datum["condition"]][li].append(e)
    
    avg_condition_E = {}
    for cond, layer_data in condition_energies.items():
        avg_condition_E[cond] = {str(li): round(float(np.mean(vals)), 4) 
                                for li, vals in layer_data.items()}
    
    print(f"\n  ★★★ Composition Results ★★★", flush=True)
    
    for li in [0, n_layers//2, n_layers]:
        print(f"    L{li}:", flush=True)
        for cond in ["both_correct", "number_wrong", "gender_wrong", "both_wrong"]:
            e = avg_condition_E.get(cond, {}).get(str(li), 0)
            print(f"      {cond}: E_total={e:.4f}", flush=True)
    
    key_errors = [avg_additivity_error.get(li, 0) for li in [n_layers//2, n_layers]]
    avg_error = float(np.mean(key_errors)) if key_errors else 1.0
    
    print(f"\n    Avg additivity error: {avg_error:.4f}", flush=True)
    print(f"    → {'Constraints compose additively!' if avg_error < 0.2 else 'Non-linear constraint coupling detected!' if avg_error > 0.5 else 'Partially additive composition'}", flush=True)
    
    return {
        "n_data_points": len(composition_data),
        "avg_condition_energy": avg_condition_E,
        "additivity_error": {str(li): v for li, v in avg_additivity_error.items()},
        "avg_additivity_error": round(avg_error, 4),
    }


# =====================================================================
# Phase E: ★★★ 约束闭合 ★★★
# =====================================================================

def run_constraint_closure(model, tokenizer, device, model_info, W_U, norm_weight):
    """
    ★★★ 约束闭合: 约束满足比例是否随深度增加? ★★★
    """
    n_layers = model_info.n_layers
    
    print("\n" + "="*70, flush=True)
    print("Phase E: ★★★ 约束闭合 ★★★", flush=True)
    print("="*70, flush=True)
    
    # Pre-extract W_U rows (带空格版本!)
    verb_pair_rows = {}
    for verb_sg, verb_pl in VERB_PAIRS:
        sg_ids = tokenizer.encode(" " + verb_sg, add_special_tokens=False)
        pl_ids = tokenizer.encode(" " + verb_pl, add_special_tokens=False)
        if len(sg_ids) == 1 and len(pl_ids) == 1:
            verb_pair_rows[(verb_sg, verb_pl)] = {
                "w_sg": W_U[sg_ids[0]].astype(np.float32),
                "w_pl": W_U[pl_ids[0]].astype(np.float32),
            }
    
    # Gender rows (带空格!)
    he_ids = tokenizer.encode(" he", add_special_tokens=False)
    she_ids = tokenizer.encode(" she", add_special_tokens=False)
    w_he = W_U[he_ids[0]].astype(np.float32) if len(he_ids) == 1 else None
    w_she = W_U[she_ids[0]].astype(np.float32) if len(she_ids) == 1 else None
    
    test_items = []
    
    for i, (noun_sg, noun_pl) in enumerate(NOUN_PAIRS[:15]):
        verb_sg, verb_pl = VERB_PAIRS[i % len(VERB_PAIRS)]
        
        # ★★★ 只用正确句子 (sg_subj+sg_verb, pl_subj+pl_verb) ★★★
        # 在subject位置测量，约束是否满足
        test_items.append({
            "sentence": f"The {noun_sg} {verb_sg}",
            "verb_sg": verb_sg, "verb_pl": verb_pl,
            "expected_sign": +1,
            "n_constraints": 1,
            "agreement": "sg_correct",
        })
        
        test_items.append({
            "sentence": f"The {noun_pl} {verb_pl}",
            "verb_sg": verb_sg, "verb_pl": verb_pl,
            "expected_sign": -1,
            "n_constraints": 1,
            "agreement": "pl_correct",
        })
    
    # Add gender constraint sentences (masc_antecedent + "he" = correct)
    for ant_masc, ant_fem, pro_masc, pro_fem in GENDER_TEMPLATES[:10]:
        test_items.append({
            "sentence": f"The {ant_masc} said {pro_masc} was",
            "verb_sg": "was", "verb_pl": "were",
            "expected_sign": +1,
            "gender_expected": +1,
            "n_constraints": 2,
            "agreement": "masc_he_correct",
        })
        
        # masc_antecedent + "she" = gender violation
        test_items.append({
            "sentence": f"The {ant_masc} said {pro_fem} was",
            "verb_sg": "was", "verb_pl": "were",
            "expected_sign": +1,
            "gender_expected": +1,
            "n_constraints": 2,
            "agreement": "masc_she_wrong",
        })
    
    print(f"  Test items: {len(test_items)}", flush=True)
    
    closure_data = []
    
    for item in test_items:
        vkey = (item["verb_sg"], item["verb_pl"])
        if vkey not in verb_pair_rows:
            continue
        rows = verb_pair_rows[vkey]
        
        positions = {"pre_verb": find_subject_position(tokenizer, item["sentence"])}
        
        try:
            pos_hs = get_hidden_states(model, tokenizer, item["sentence"], positions, n_layers)
        except:
            continue
        
        if "pre_verb" not in pos_hs:
            continue
        
        layer_satisfied = {}
        for li, h in pos_hs["pre_verb"].items():
            satisfied = 0
            total = item["n_constraints"]
            
            raw_num = compute_margin_from_h(h, rows["w_sg"], rows["w_pl"], norm_weight)
            proper_num = item["expected_sign"] * raw_num
            if proper_num > 0:
                satisfied += 1
            
            if "gender_expected" in item and w_he is not None and w_she is not None:
                raw_gen = compute_margin_from_h(h, w_he, w_she, norm_weight)
                proper_gen = item["gender_expected"] * raw_gen
                if proper_gen > 0:
                    satisfied += 1
            
            layer_satisfied[li] = satisfied / total
        
        closure_data.append({
            "agreement": item["agreement"],
            "n_constraints": item["n_constraints"],
            "closure": layer_satisfied,
        })
    
    # Group by agreement type
    correct_closure = defaultdict(list)  # sg_correct + pl_correct + masc_he_correct
    wrong_closure = defaultdict(list)    # masc_she_wrong
    
    for datum in closure_data:
        for li, c in datum["closure"].items():
            if "wrong" in datum["agreement"]:
                wrong_closure[li].append(c)
            else:
                correct_closure[li].append(c)
    
    avg_correct_closure = {li: round(float(np.mean(vals)), 4) for li, vals in correct_closure.items()}
    avg_wrong_closure = {li: round(float(np.mean(vals)), 4) for li, vals in wrong_closure.items()}
    
    closure_rate = {}
    sorted_layers = sorted(avg_correct_closure.keys())
    for i in range(1, len(sorted_layers)):
        li_prev = sorted_layers[i-1]
        li_curr = sorted_layers[i]
        dc = avg_correct_closure[li_curr] - avg_correct_closure[li_prev]
        dl = li_curr - li_prev
        closure_rate[li_curr] = round(dc / dl, 4)
    
    increasing_fraction = sum(1 for v in closure_rate.values() if v > 0) / max(len(closure_rate), 1)
    
    print(f"\n  ★★★ Closure Results ★★★", flush=True)
    print(f"    Closure at L0 (correct): {avg_correct_closure.get(0, 0):.4f}", flush=True)
    print(f"    Closure at L_last (correct): {avg_correct_closure.get(n_layers, 0):.4f}", flush=True)
    print(f"    Closure at L_last (wrong): {avg_wrong_closure.get(n_layers, 0):.4f}", flush=True)
    print(f"    Fraction of layers with increasing closure: {increasing_fraction:.2%}", flush=True)
    
    for li in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]:
        cc = avg_correct_closure.get(li, 0)
        wc = avg_wrong_closure.get(li, 0)
        gap = cc - wc
        print(f"    L{li}: correct={cc:.4f}, wrong={wc:.4f}, gap={gap:.4f}", flush=True)
    
    return {
        "avg_correct_closure": {str(li): v for li, v in avg_correct_closure.items()},
        "avg_wrong_closure": {str(li): v for li, v in avg_wrong_closure.items()},
        "closure_rate": {str(li): v for li, v in closure_rate.items()},
        "increasing_fraction": round(increasing_fraction, 4),
        "n_data_points": len(closure_data),
    }


# =====================================================================
# MAIN
# =====================================================================

def run_phase178(model_name):
    print(f"\n{'='*70}", flush=True)
    print(f"Phase 178: ★★★ 约束场理论 — 从观察到方程 ★★★", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"{'='*70}", flush=True)

    t_start = time.time()

    # Load model (BF16 + device_map="auto")
    model, tokenizer, device = load_model_bf16(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    print(f"  Model: {model_info.model_class}, L={n_layers}, d={d_model}", flush=True)

    # Load W_U and final norm weight
    print("  Loading W_U and final norm...", flush=True)
    W_U = get_W_U(model, model_name)
    norm_weight = get_final_norm_weight(model)
    print(f"  W_U shape: {W_U.shape}, Norm weight: {'found' if norm_weight is not None else 'not found'}", flush=True)
    print(f"  ★★★ Memory optimization: W_U rows extracted on-demand, NO full matrix multiply ★★★", flush=True)

    # =====================================================================
    # Run all experiments
    # =====================================================================

    exp_a = run_constraint_propagation(model, tokenizer, device, model_info, W_U, norm_weight)
    exp_b = run_proper_energy(model, tokenizer, device, model_info, W_U, norm_weight)
    exp_c = run_constraint_field(model, tokenizer, device, model_info, W_U, norm_weight)
    exp_d = run_constraint_composition(model, tokenizer, device, model_info, W_U, norm_weight)
    exp_e = run_constraint_closure(model, tokenizer, device, model_info, W_U, norm_weight)

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "timestamp": timestamp,
        "phase_A_propagation": exp_a,
        "phase_B_energy": exp_b,
        "phase_C_field": exp_c,
        "phase_D_composition": exp_d,
        "phase_E_closure": exp_e,
    }

    out_path = f"tests/glm5_temp/phase178_{model_name}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    # Release model
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\nPhase 178 ({model_name}) completed in {elapsed:.1f}s", flush=True)

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase178_constraint_field_theory.py <model_name>")
        print("  model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    run_phase178(model_name)
