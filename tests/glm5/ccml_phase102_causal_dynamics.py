"""
Phase 102: Causal Dynamics — 从观察到干预
==========================================

Phase 101的进步:
  1. 最后token ≠ 语义对象 (ctx ratio=193)
  2. 翻译方向一致性0.94 (is-a只有0.40)
  3. 开始从实体转向关系

Phase 101的硬伤 (用户批判):
  1. 方向一致性 ≠ 计算原语 — 可能只是decoder alignment / embedding geometry
  2. 高维集中效应未解决 — cosine/distance/CKA的解释不可靠
  3. CKA≈0.003 ≠ "完全正交" — CKA度量的是表示相似性，不是子空间正交性
  4. "概念流形不存在"证据不足 — 用错了工具
  5. 最后token是序列状态，不是词级语义

核心方法论升级:
  从"观察"到"干预": 不再只看几何模式，而是主动改变内部状态，观察行为变化
  从"方向一致性"到"因果特异性": 改变翻译方向→翻译行为改变，其他行为不变
  从"状态"到"跃迁": Δh = h_{l+1} - h_l 才是真正的计算

实验设计:
  Exp 1: 翻译方向因果干预 — 最关键实验
    1a: 从20+翻译对提取v_trans (上下文化)
    1b: 在中文上下文中添加v_trans，检测是否切换到英文
    1c: 特异性测试 — 是否只增强正确翻译的概率？
    1d: 控制对比 — 随机方向、is-a方向、排列检验
    1e: 反向测试 — 从英文上下文中减去v_trans，是否切换到中文？

  Exp 2: 层间跃迁动力学 (Δh分析)
    2a: 翻译vs中文上下文的Δh轨迹对比
    2b: Δh的范数/方向随层变化 — 找到"计算转折点"
    2c: Δh的秩和结构 — 是否低秩？是否有稳定子空间？

  Exp 3: Jacobian分析
    3a: ∂h_{l+1}/∂h_l 的近似 (数值差分)
    3b: Jacobian的秩和谱 — 局部计算的复杂度
    3c: 翻译vs非翻译的Jacobian差异

Run:
  python tests/glm5/ccml_phase102_causal_dynamics.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase102_causal_dynamics.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase102_causal_dynamics.py --model qwen3 --exp 3
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import gc
import json
import time
from collections import defaultdict

from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U


# ============================================================
# 翻译对定义 — 用于提取翻译方向
# ============================================================
# 训练集: 用于提取v_trans (测试集词不在训练集中)
TRANSLATION_TRAIN = [
    ("猫", "cat"), ("狗", "dog"), ("书", "book"), ("火", "fire"),
    ("花", "flower"), ("鱼", "fish"), ("树", "tree"), ("鸟", "bird"),
    ("马", "horse"), ("铁", "iron"), ("金", "gold"), ("茶", "tea"),
    ("米", "rice"), ("血", "blood"), ("眼", "eye"), ("手", "hand"),
    ("水", "water"), ("风", "wind"), ("雪", "snow"), ("星", "star"),
    ("月", "moon"), ("日", "sun"), ("山", "mountain"), ("河", "river"),
]

# 测试集: 用于干预实验 (不在训练集中)
TRANSLATION_TEST = [
    ("龙", "dragon"), ("云", "cloud"), ("雨", "rain"), ("雷", "thunder"),
    ("石", "stone"), ("草", "grass"), ("沙", "sand"), ("冰", "ice"),
    ("光", "light"), ("影", "shadow"), ("梦", "dream"), ("歌", "song"),
]

# Is-a关系对 — 用于提取is-a方向
ISA_TRAIN = [
    ("苹果", "水果"), ("狗", "动物"), ("玫瑰", "花"), ("老虎", "猫科"),
    ("桌子", "家具"), ("钢笔", "文具"), ("汽车", "交通工具"), ("米饭", "食物"),
    ("铁", "金属"), ("香蕉", "水果"), ("熊猫", "动物"), ("白菜", "蔬菜"),
    ("飞机", "交通工具"), ("猫", "动物"), ("红", "颜色"), ("北京", "城市"),
]

# 中文上下文模板 — 不涉及翻译
CHINESE_ONLY_PROMPTS = [
    "{word}是一种{cat}，它",
    "{word}在很多地方都能看到，",
    "关于{word}，我想说的是",
    "{word}的特点是",
]


def get_last_token_hidden(model, input_ids, device):
    """获取最后token的所有层hidden state"""
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hiddens = []
    for l in range(len(outputs.hidden_states)):
        h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
        hiddens.append(h)
    return hiddens


def get_word_position_hidden(model, tokenizer, text, word, device):
    """获取目标词位置的所有层hidden state"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        outputs = model(input_ids, output_hidden_states=True)
    
    # 找到目标词的token位置
    word_tokens = tokenizer.encode(word, add_special_tokens=False)
    full_tokens = input_ids[0].tolist()
    
    # 寻找word_tokens在full_tokens中的起始位置
    word_start = -1
    for i in range(len(full_tokens) - len(word_tokens) + 1):
        if full_tokens[i:i+len(word_tokens)] == word_tokens:
            word_start = i
            break
    
    if word_start == -1:
        # fallback: 用最后token
        word_start = len(full_tokens) - 1
    
    # 取word的最后一个token位置 (更接近完整词义)
    word_end = word_start + len(word_tokens) - 1
    
    hiddens = []
    for l in range(len(outputs.hidden_states)):
        h = outputs.hidden_states[l][0, word_end, :].float().cpu().numpy()
        hiddens.append(h)
    
    return hiddens, word_end


def intervene_at_layer(model, tokenizer, prompt, layer_idx, direction_tensor, alpha, device):
    """在指定层注入方向向量到last token位置，返回干预后的logits"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    layers = get_layers(model)
    
    intervened = [False]
    
    def hook_fn(module, input, output):
        if not intervened[0]:
            if isinstance(output, tuple):
                hidden_states = output[0].clone()
                # 在最后token位置注入方向
                hidden_states[:, -1, :] += alpha * direction_tensor.to(hidden_states.dtype).to(device)
                output = (hidden_states,) + output[1:]
            intervened[0] = True
        return output
    
    handle = layers[layer_idx].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :].float().cpu()  # last token logits
    
    handle.remove()
    return logits


def intervene_at_layer_word_pos(model, tokenizer, prompt, word, layer_idx, 
                                 direction_tensor, alpha, device):
    """在指定层注入方向向量到目标词位置，返回干预后的logits"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # 找到目标词位置
    word_tokens = tokenizer.encode(word, add_special_tokens=False)
    full_tokens = input_ids[0].tolist()
    word_end = len(full_tokens) - 1  # fallback
    for i in range(len(full_tokens) - len(word_tokens) + 1):
        if full_tokens[i:i+len(word_tokens)] == word_tokens:
            word_end = i + len(word_tokens) - 1
            break
    
    layers = get_layers(model)
    intervened = [False]
    
    def hook_fn(module, input, output):
        if not intervened[0]:
            if isinstance(output, tuple):
                hidden_states = output[0].clone()
                hidden_states[:, word_end, :] += alpha * direction_tensor.to(hidden_states.dtype).to(device)
                output = (hidden_states,) + output[1:]
            intervened[0] = True
        return output
    
    handle = layers[layer_idx].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :].float().cpu()
    
    handle.remove()
    return logits


def get_top_k_tokens(logits, tokenizer, k=10):
    """获取top-k token及其概率"""
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, k)
    results = []
    for i in range(k):
        tok_id = topk.indices[i].item()
        prob = topk.values[i].item()
        tok_str = tokenizer.decode([tok_id])
        results.append({"token": tok_str, "token_id": tok_id, "prob": prob})
    return results


def get_token_prob(logits, tokenizer, text):
    """获取特定token的logits概率"""
    tok_ids = tokenizer.encode(text, add_special_tokens=False)
    if not tok_ids:
        return 0.0
    probs = torch.softmax(logits, dim=-1)
    return probs[tok_ids[0]].item()


# ============================================================
# Exp 1: 翻译方向因果干预
# ============================================================
def exp1_translation_intervention(model_name):
    """
    翻译方向因果干预 — 最关键实验
    
    核心逻辑:
    1. 从训练集翻译对提取v_trans
    2. 在测试集中文上下文中注入v_trans
    3. 检测: (a) 是否切换到英文? (b) 是否是正确的英文翻译?
    4. 控制对比: 随机方向、is-a方向、排列检验
    
    判断标准:
    - v_trans是"语言模式切换" → 输出随机英文词
    - v_trans是"翻译计算原语" → 输出正确的英文翻译
    """
    print(f"\n{'='*70}")
    print(f"Exp 1: 翻译方向因果干预 — {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")
    
    results = {}
    
    # ---- Step 1a: 从训练集提取翻译方向 ----
    print(f"\n  === Step 1a: 提取翻译方向 (训练集: {len(TRANSLATION_TRAIN)}对) ===")
    
    # 两种方式提取:
    # 方式A: 孤立词 h(en_word) - h(zh_word)
    # 方式B: 上下文化 h("翻译prompt") - h("中文prompt")
    
    # 方式A: 孤立词
    isolated_deltas = defaultdict(list)  # layer -> [delta_vec]
    for zh, en in TRANSLATION_TRAIN:
        inputs_zh = tokenizer(zh, return_tensors="pt").to(device)
        inputs_en = tokenizer(en, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out_zh = model(inputs_zh["input_ids"], output_hidden_states=True)
            out_en = model(inputs_en["input_ids"], output_hidden_states=True)
        
        for l in range(n_layers + 1):
            h_zh = out_zh.hidden_states[l][0, -1, :].float().cpu().numpy()
            h_en = out_en.hidden_states[l][0, -1, :].float().cpu().numpy()
            delta = h_en - h_zh
            isolated_deltas[l].append(delta)
    
    # 方式B: 上下文化
    contextual_deltas = defaultdict(list)
    for zh, en in TRANSLATION_TRAIN:
        # 翻译上下文
        trans_prompt = f'请把"{zh}"翻译成英文：'
        # 中文补全上下文
        zh_prompt = f"{zh}是一种"
        
        with torch.no_grad():
            out_trans = model(tokenizer(trans_prompt, return_tensors="pt").to(device)["input_ids"],
                            output_hidden_states=True)
            out_zh = model(tokenizer(zh_prompt, return_tensors="pt").to(device)["input_ids"],
                          output_hidden_states=True)
        
        for l in range(n_layers + 1):
            h_trans = out_trans.hidden_states[l][0, -1, :].float().cpu().numpy()
            h_zh = out_zh.hidden_states[l][0, -1, :].float().cpu().numpy()
            delta = h_trans - h_zh
            contextual_deltas[l].append(delta)
    
    # 计算每种方式的平均翻译方向和一致性
    translation_dirs = {}
    for mode, deltas_dict in [("isolated", isolated_deltas), ("contextual", contextual_deltas)]:
        mode_results = {}
        for l in range(n_layers + 1):
            deltas = np.array(deltas_dict[l])  # [n_pairs, d_model]
            mean_delta = np.mean(deltas, axis=0)
            mean_norm = np.linalg.norm(mean_delta)
            
            # 方向一致性: 每个delta与平均方向的余弦
            if mean_norm > 1e-6:
                cosines = []
                for d in deltas:
                    d_norm = np.linalg.norm(d)
                    if d_norm > 1e-6:
                        cos = np.dot(d, mean_delta) / (d_norm * mean_norm)
                        cosines.append(float(cos))
                alignment = float(np.mean(cosines)) if cosines else 0.0
            else:
                alignment = 0.0
            
            # 归一化方向
            if mean_norm > 1e-6:
                direction = mean_delta / mean_norm
            else:
                direction = np.zeros(d_model)
            
            mode_results[str(l)] = {
                "alignment": alignment,
                "mean_norm": float(mean_norm),
                "direction_norm": direction,  # 归一化方向
            }
        
        translation_dirs[mode] = mode_results
        
        # 输出关键层的一致性
        for l in [0, 6, 12, 18, 24, 30]:
            if str(l) in mode_results:
                r = mode_results[str(l)]
                print(f"    {mode} L{l}: alignment={r['alignment']:.3f}, norm={r['mean_norm']:.1f}")
    
    # 也提取is-a方向 (作为控制)
    print(f"\n  提取is-a方向 (训练集: {len(ISA_TRAIN)}对)")
    isa_deltas = defaultdict(list)
    for a, b in ISA_TRAIN:
        inputs_a = tokenizer(a, return_tensors="pt").to(device)
        inputs_b = tokenizer(b, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out_a = model(inputs_a["input_ids"], output_hidden_states=True)
            out_b = model(inputs_b["input_ids"], output_hidden_states=True)
        
        for l in range(n_layers + 1):
            h_a = out_a.hidden_states[l][0, -1, :].float().cpu().numpy()
            h_b = out_b.hidden_states[l][0, -1, :].float().cpu().numpy()
            isa_deltas[l].append(h_b - h_a)
    
    isa_dirs = {}
    for l in range(n_layers + 1):
        deltas = np.array(isa_deltas[l])
        mean_delta = np.mean(deltas, axis=0)
        mean_norm = np.linalg.norm(mean_delta)
        if mean_norm > 1e-6:
            direction = mean_delta / mean_norm
        else:
            direction = np.zeros(d_model)
        isa_dirs[l] = direction
    
    # ---- Step 1b: 因果干预实验 ----
    print(f"\n  === Step 1b: 因果干预实验 (测试集: {len(TRANSLATION_TEST)}对) ===")
    
    # 选择干预层
    intervention_layers = [3, 9, 15, 21, 27, 33]
    intervention_layers = [l for l in intervention_layers if l < n_layers]
    
    # alpha值范围
    alphas = [0.5, 1.0, 2.0, 4.0, 8.0]
    
    # 收集baseline logits (无干预)
    print(f"\n  收集baseline logits...")
    baseline_results = {}
    for zh, en in TRANSLATION_TEST:
        zh_prompt = f"{zh}是一种"
        inputs = tokenizer(zh_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            baseline_logits = model(inputs["input_ids"]).logits[0, -1, :].float().cpu()
        
        # 正确英文翻译的概率
        en_prob = get_token_prob(baseline_logits, tokenizer, en)
        # 中文词的概率 (同类别)
        top10 = get_top_k_tokens(baseline_logits, tokenizer, 10)
        
        baseline_results[zh] = {
            "en_translation": en,
            "en_prob": en_prob,
            "top10": top10,
        }
        print(f"    {zh}({en}): baseline P({en})={en_prob:.6f}, top1={top10[0]['token']}({top10[0]['prob']:.4f})")
    
    # 干预实验: 使用上下文化的翻译方向
    print(f"\n  === 干预实验: 上下文化v_trans ===")
    intervention_results = {}
    
    for l_idx in intervention_layers:
        print(f"\n  --- 干预层: L{l_idx} ---")
        layer_results = {}
        
        # 获取翻译方向 (上下文化)
        v_trans = translation_dirs["contextual"][str(l_idx)]["direction_norm"]
        v_trans_tensor = torch.tensor(v_trans, dtype=torch.float32)
        
        # 获取is-a方向
        v_isa = isa_dirs[l_idx]
        v_isa_tensor = torch.tensor(v_isa, dtype=torch.float32)
        
        # 生成随机方向 (相同范数)
        v_random = np.random.randn(d_model)
        v_random = v_random / np.linalg.norm(v_random)
        v_random_tensor = torch.tensor(v_random, dtype=torch.float32)
        
        for alpha in alphas:
            alpha_key = f"alpha_{alpha}"
            alpha_results = {"translation": {}, "isa_control": {}, "random_control": {}}
            
            for zh, en in TRANSLATION_TEST:
                zh_prompt = f"{zh}是一种"
                
                # 1. 翻译方向干预
                try:
                    intervened_logits = intervene_at_layer(
                        model, tokenizer, zh_prompt, l_idx, v_trans_tensor, alpha, device
                    )
                    en_prob_intervened = get_token_prob(intervened_logits, tokenizer, en)
                    top10_intervened = get_top_k_tokens(intervened_logits, tokenizer, 10)
                    
                    # 统计英文token的概率变化
                    en_tokens_in_top10 = sum(1 for t in top10_intervened 
                                            if all(ord(c) < 128 for c in t['token'].strip()))
                    
                    alpha_results["translation"][zh] = {
                        "en_prob": en_prob_intervened,
                        "en_prob_change": en_prob_intervened - baseline_results[zh]["en_prob"],
                        "top10": top10_intervened,
                        "english_tokens_in_top10": en_tokens_in_top10,
                    }
                except Exception as e:
                    alpha_results["translation"][zh] = {"error": str(e)}
                
                # 2. Is-a方向干预 (控制)
                try:
                    isa_logits = intervene_at_layer(
                        model, tokenizer, zh_prompt, l_idx, v_isa_tensor, alpha, device
                    )
                    en_prob_isa = get_token_prob(isa_logits, tokenizer, en)
                    top10_isa = get_top_k_tokens(isa_logits, tokenizer, 10)
                    
                    alpha_results["isa_control"][zh] = {
                        "en_prob": en_prob_isa,
                        "en_prob_change": en_prob_isa - baseline_results[zh]["en_prob"],
                        "top10": top10_isa,
                    }
                except Exception as e:
                    alpha_results["isa_control"][zh] = {"error": str(e)}
                
                # 3. 随机方向干预 (控制) — 只对第一个alpha做
                if alpha == alphas[0]:
                    try:
                        random_logits = intervene_at_layer(
                            model, tokenizer, zh_prompt, l_idx, v_random_tensor, alpha, device
                        )
                        en_prob_random = get_token_prob(random_logits, tokenizer, en)
                        top10_random = get_top_k_tokens(random_logits, tokenizer, 10)
                        
                        alpha_results["random_control"][zh] = {
                            "en_prob": en_prob_random,
                            "en_prob_change": en_prob_random - baseline_results[zh]["en_prob"],
                            "top10": top10_random,
                        }
                    except Exception as e:
                        alpha_results["random_control"][zh] = {"error": str(e)}
            
            # 汇总alpha结果
            trans_changes = [alpha_results["translation"][zh]["en_prob_change"] 
                           for zh in [z for z, e in TRANSLATION_TEST]
                           if "en_prob_change" in alpha_results["translation"].get(zh, {})]
            isa_changes = [alpha_results["isa_control"][zh]["en_prob_change"]
                         for zh in [z for z, e in TRANSLATION_TEST]  
                         if "en_prob_change" in alpha_results["isa_control"].get(zh, {})]
            
            mean_trans_change = float(np.mean(trans_changes)) if trans_changes else 0
            mean_isa_change = float(np.mean(isa_changes)) if isa_changes else 0
            
            print(f"    α={alpha}: trans_avg_ΔP(en)={mean_trans_change:.6f}, "
                  f"isa_avg_ΔP(en)={mean_isa_change:.6f}")
            
            layer_results[alpha_key] = alpha_results
        
        intervention_results[f"L{l_idx}"] = layer_results
    
    # ---- Step 1c: 特异性测试 ----
    print(f"\n  === Step 1c: 特异性测试 ===")
    print(f"  判断标准: v_trans是否只增强正确翻译的概率？")
    
    specificity_results = {}
    
    # 选择最佳干预层和alpha (基于上面的结果)
    # 使用L15, alpha=4.0作为默认
    best_layer = 15
    best_alpha = 4.0
    if best_layer >= n_layers:
        best_layer = intervention_layers[len(intervention_layers)//2]
    
    v_trans_best = translation_dirs["contextual"][str(best_layer)]["direction_norm"]
    v_trans_best_tensor = torch.tensor(v_trans_best, dtype=torch.float32)
    
    for zh, en in TRANSLATION_TEST:
        zh_prompt = f"{zh}是一种"
        
        # baseline
        inputs = tokenizer(zh_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            baseline_logits = model(inputs["input_ids"]).logits[0, -1, :].float().cpu()
        
        # 干预
        intervened_logits = intervene_at_layer(
            model, tokenizer, zh_prompt, best_layer, v_trans_best_tensor, best_alpha, device
        )
        
        # 计算各种token的概率变化
        baseline_probs = torch.softmax(baseline_logits, dim=-1)
        intervened_probs = torch.softmax(intervened_logits, dim=-1)
        
        # 1. 正确英文翻译的概率变化
        en_tok_ids = tokenizer.encode(en, add_special_tokens=False)
        en_prob_baseline = baseline_probs[en_tok_ids[0]].item() if en_tok_ids else 0
        en_prob_intervened = intervened_probs[en_tok_ids[0]].item() if en_tok_ids else 0
        
        # 2. 其他英文词的概率变化 (5个随机英文词)
        other_en_words = ["the", "and", "is", "of", "a"]
        other_en_changes = []
        for w in other_en_words:
            w_ids = tokenizer.encode(w, add_special_tokens=False)
            if w_ids:
                p_base = baseline_probs[w_ids[0]].item()
                p_int = intervened_probs[w_ids[0]].item()
                other_en_changes.append(p_int - p_base)
        
        # 3. 中文续写词的概率变化
        zh_continue_words = ["很", "是", "有", "能", "在"]
        zh_changes = []
        for w in zh_continue_words:
            w_ids = tokenizer.encode(w, add_special_tokens=False)
            if w_ids:
                p_base = baseline_probs[w_ids[0]].item()
                p_int = intervened_probs[w_ids[0]].item()
                zh_changes.append(p_int - p_base)
        
        specificity_results[zh] = {
            "en_translation": en,
            "en_prob_baseline": en_prob_baseline,
            "en_prob_intervened": en_prob_intervened,
            "en_prob_change": en_prob_intervened - en_prob_baseline,
            "other_en_avg_change": float(np.mean(other_en_changes)),
            "zh_avg_change": float(np.mean(zh_changes)),
            "specificity_ratio": (en_prob_intervened - en_prob_baseline) / (max(np.mean(other_en_changes), 1e-8)),
        }
        
        spec_ratio = specificity_results[zh]["specificity_ratio"]
        print(f"    {zh}({en}): ΔP(correct)={en_prob_intervened-en_prob_baseline:.6f}, "
              f"ΔP(other_en)={np.mean(other_en_changes):.6f}, "
              f"ΔP(zh)={np.mean(zh_changes):.6f}, specificity={spec_ratio:.2f}")
    
    # ---- Step 1d: 排列检验 ----
    print(f"\n  === Step 1d: 排列检验 ===")
    print(f"  随机打乱翻译对，计算'方向一致性'的空分布")
    
    # 用随机配对的词计算方向一致性
    all_zh = [zh for zh, en in TRANSLATION_TRAIN]
    all_en = [en for zh, en in TRANSLATION_TRAIN]
    
    n_permutations = 100
    perm_alignment_l9 = []
    
    # 只在L9做排列检验 (节省时间)
    target_layer = 9
    if target_layer > n_layers:
        target_layer = n_layers // 2
    
    for perm_i in range(n_permutations):
        # 随机打乱en的顺序
        perm_en = np.random.permutation(all_en).tolist()
        perm_deltas = []
        
        for zh, en in zip(all_zh, perm_en):
            inputs_zh = tokenizer(zh, return_tensors="pt").to(device)
            inputs_en = tokenizer(en, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out_zh = model(inputs_zh["input_ids"], output_hidden_states=True)
                out_en = model(inputs_en["input_ids"], output_hidden_states=True)
            
            h_zh = out_zh.hidden_states[target_layer][0, -1, :].float().cpu().numpy()
            h_en = out_en.hidden_states[target_layer][0, -1, :].float().cpu().numpy()
            perm_deltas.append(h_en - h_zh)
        
        # 计算排列后的方向一致性
        perm_deltas = np.array(perm_deltas)
        mean_delta = np.mean(perm_deltas, axis=0)
        mean_norm = np.linalg.norm(mean_delta)
        
        if mean_norm > 1e-6:
            cosines = []
            for d in perm_deltas:
                d_norm = np.linalg.norm(d)
                if d_norm > 1e-6:
                    cos = np.dot(d, mean_delta) / (d_norm * mean_norm)
                    cosines.append(float(cos))
            perm_alignment = float(np.mean(cosines)) if cosines else 0.0
        else:
            perm_alignment = 0.0
        
        perm_alignment_l9.append(perm_alignment)
    
    # 真实的翻译方向一致性
    real_alignment = translation_dirs["isolated"][str(target_layer)]["alignment"]
    
    # p-value
    p_value = float(np.mean([a >= real_alignment for a in perm_alignment_l9]))
    
    print(f"    L{target_layer}: 真实alignment={real_alignment:.3f}, "
          f"排列alignment: mean={np.mean(perm_alignment_l9):.3f}, "
          f"std={np.std(perm_alignment_l9):.3f}, p={p_value:.4f}")
    
    # ---- Step 1e: 反向测试 ----
    print(f"\n  === Step 1e: 反向测试 ===")
    print(f"  从英文上下文中减去v_trans，是否切换到中文？")
    
    reverse_results = {}
    for zh, en in TRANSLATION_TEST[:6]:  # 只测试6对节省时间
        en_prompt = f"The {en} is a"
        
        # baseline (英文上下文)
        inputs = tokenizer(en_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            baseline_logits = model(inputs["input_ids"]).logits[0, -1, :].float().cpu()
        
        # 减去v_trans
        v_trans_l15 = translation_dirs["contextual"][str(best_layer)]["direction_norm"]
        v_trans_l15_tensor = torch.tensor(v_trans_l15, dtype=torch.float32)
        
        reverse_logits = intervene_at_layer(
            model, tokenizer, en_prompt, best_layer, -v_trans_l15_tensor, best_alpha, device
        )
        
        # 检查中文token的概率
        zh_prob_baseline = get_token_prob(baseline_logits, tokenizer, zh)
        zh_prob_reverse = get_token_prob(reverse_logits, tokenizer, zh)
        
        top10_baseline = get_top_k_tokens(baseline_logits, tokenizer, 10)
        top10_reverse = get_top_k_tokens(reverse_logits, tokenizer, 10)
        
        reverse_results[zh] = {
            "en": en,
            "zh_prob_baseline": zh_prob_baseline,
            "zh_prob_reverse": zh_prob_reverse,
            "zh_prob_change": zh_prob_reverse - zh_prob_baseline,
            "top10_baseline": top10_baseline,
            "top10_reverse": top10_reverse,
        }
        
        print(f"    {en}→{zh}: ΔP({zh})={zh_prob_reverse-zh_prob_baseline:.6f}")
    
    # 汇总
    results["translation_dirs"] = {mode: {l: {"alignment": v["alignment"], "mean_norm": v["mean_norm"]} 
                                          for l, v in mode_results.items()} 
                                   for mode, mode_results in translation_dirs.items()}
    results["baseline"] = {zh: {"en_translation": v["en_translation"], "en_prob": v["en_prob"]} 
                          for zh, v in baseline_results.items()}
    results["intervention"] = intervention_results
    results["specificity"] = specificity_results
    results["permutation_test"] = {
        "real_alignment": real_alignment,
        "perm_mean": float(np.mean(perm_alignment_l9)),
        "perm_std": float(np.std(perm_alignment_l9)),
        "p_value": p_value,
        "n_permutations": n_permutations,
    }
    results["reverse_test"] = reverse_results
    
    # 保存
    # 注意: direction_norm是numpy数组，不能直接JSON序列化
    save_path = f"tests/glm5_temp/phase102_exp1_{model_name}_causal_intervention.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  保存到: {save_path}")
    
    release_model(model)
    return results


# ============================================================
# Exp 2: 层间跃迁动力学 (Δh分析)
# ============================================================
def exp2_transition_dynamics(model_name):
    """
    层间跃迁动力学 — Δh = h_{l+1} - h_l 才是真正的计算
    
    核心思路:
    - 不看静态状态h，看状态跃迁Δh
    - 对比不同任务的Δh轨迹: 翻译 vs 中文补全 vs 英文补全
    - Δh的范数变化 → 找到"计算转折点"
    - Δh的方向变化 → 找到稳定的"计算子空间"
    - Δh的秩 → 计算复杂度
    """
    print(f"\n{'='*70}")
    print(f"Exp 2: 层间跃迁动力学 — {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")
    
    results = {}
    
    # 定义三种任务的prompt
    test_pairs = [
        ("猫", "cat"), ("狗", "dog"), ("水", "water"), 
        ("火", "fire"), ("树", "tree"), ("花", "flower"),
        ("鱼", "fish"), ("鸟", "bird"), ("铁", "iron"), ("茶", "tea"),
    ]
    
    task_hiddens = {"zh_continue": {}, "en_continue": {}, "translate": {}}
    
    print(f"\n  收集三种任务的hidden states...")
    for zh, en in test_pairs:
        # 中文补全: "{zh}是一种"
        zh_prompt = f"{zh}是一种"
        # 英文补全: "The {en} is a"
        en_prompt = f"The {en} is a"
        # 翻译: "请翻译：{zh} →"
        trans_prompt = f"请翻译：{zh} →"
        
        for task_name, prompt in [("zh_continue", zh_prompt), 
                                   ("en_continue", en_prompt),
                                   ("translate", trans_prompt)]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(inputs["input_ids"], output_hidden_states=True)
            
            hiddens = []
            for l in range(n_layers + 1):
                h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
                hiddens.append(h)
            
            task_hiddens[task_name][(zh, en)] = hiddens
    
    # ---- 2a: Δh轨迹对比 ----
    print(f"\n  === 2a: Δh轨迹对比 ===")
    
    trajectory_results = {}
    for task_name in ["zh_continue", "en_continue", "translate"]:
        task_deltas = []  # [n_pairs, n_layers, d_model]
        for pair, hiddens in task_hiddens[task_name].items():
            deltas = []
            for l in range(n_layers):
                delta = hiddens[l+1] - hiddens[l]
                deltas.append(delta)
            task_deltas.append(deltas)
        
        task_deltas = np.array(task_deltas)  # [n_pairs, n_layers, d_model]
        
        layer_stats = {}
        for l in range(n_layers):
            deltas_at_l = task_deltas[:, l, :]  # [n_pairs, d_model]
            
            # Δh范数
            norms = np.linalg.norm(deltas_at_l, axis=1)
            mean_norm = float(np.mean(norms))
            
            # Δh方向的pair间一致性
            mean_delta = np.mean(deltas_at_l, axis=0)
            mean_delta_norm = np.linalg.norm(mean_delta)
            if mean_delta_norm > 1e-6:
                cosines = []
                for d in deltas_at_l:
                    d_norm = np.linalg.norm(d)
                    if d_norm > 1e-6:
                        cos = np.dot(d, mean_delta) / (d_norm * mean_delta_norm)
                        cosines.append(float(cos))
                alignment = float(np.mean(cosines)) if cosines else 0.0
            else:
                alignment = 0.0
            
            # Δh方向的范数 (平均方向的范数)
            mean_dir_norm = float(mean_delta_norm)
            
            layer_stats[str(l)] = {
                "mean_delta_norm": mean_norm,
                "alignment": alignment,
                "mean_dir_norm": mean_dir_norm,
                "noise_ratio": 1.0 - alignment,  # 高 = 更diverse
            }
        
        trajectory_results[task_name] = layer_stats
        
        # 找Δh范数最大的层 ("计算最密集"的层)
        norms_by_layer = [(l, layer_stats[str(l)]["mean_delta_norm"]) for l in range(n_layers)]
        top3_norms = sorted(norms_by_layer, key=lambda x: x[1], reverse=True)[:3]
        
        # 找Δh一致性最高的层 ("最稳定的计算"层)
        align_by_layer = [(l, layer_stats[str(l)]["alignment"]) for l in range(n_layers)]
        top3_align = sorted(align_by_layer, key=lambda x: x[1], reverse=True)[:3]
        
        print(f"\n    {task_name}:")
        print(f"      Δh范数最大的层: {[(f'L{l}', f'{n:.1f}') for l, n in top3_norms]}")
        print(f"      Δh一致性最高的层: {[(f'L{l}', f'{a:.3f}') for l, a in top3_align]}")
    
    # ---- 2b: 跨任务Δh对比 ----
    print(f"\n  === 2b: 跨任务Δh对比 ===")
    
    cross_task_results = {}
    for l in range(0, n_layers, 3):
        cross_task = {}
        task_mean_deltas = {}
        for task_name in ["zh_continue", "en_continue", "translate"]:
            task_deltas = []
            for pair, hiddens in task_hiddens[task_name].items():
                delta = hiddens[l+1] - hiddens[l]
                task_deltas.append(delta)
            mean_delta = np.mean(task_deltas, axis=0)
            task_mean_deltas[task_name] = mean_delta
        
        # 计算不同任务间Δh方向的余弦相似度
        task_names = list(task_mean_deltas.keys())
        for i in range(len(task_names)):
            for j in range(i+1, len(task_names)):
                t1, t2 = task_names[i], task_names[j]
                d1, d2 = task_mean_deltas[t1], task_mean_deltas[t2]
                n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                if n1 > 1e-6 and n2 > 1e-6:
                    cos = float(np.dot(d1, d2) / (n1 * n2))
                else:
                    cos = 0.0
                cross_task[f"{t1}_vs_{t2}"] = cos
        
        cross_task_results[str(l)] = cross_task
        
        if l % 9 == 0:
            print(f"    L{l}: " + ", ".join([f"{k}={v:.3f}" for k, v in cross_task.items()]))
    
    # ---- 2c: Δh的SVD分析 — 计算子空间 ----
    print(f"\n  === 2c: Δh的SVD分析 — 计算子空间 ===")
    
    svd_results = {}
    for task_name in ["zh_continue", "translate"]:
        task_deltas = []
        for pair, hiddens in task_hiddens[task_name].items():
            for l in range(n_layers):
                delta = hiddens[l+1] - hiddens[l]
                task_deltas.append(delta)
        
        # 堆叠所有Δh: [n_pairs * n_layers, d_model]
        all_deltas = np.array(task_deltas)
        
        # SVD
        U, S, Vt = np.linalg.svd(all_deltas, full_matrices=False)
        
        # 解释方差比
        total_var = np.sum(S**2)
        cumvar = np.cumsum(S**2) / total_var
        
        # 找到解释90%方差需要的维度
        dim_90 = int(np.searchsorted(cumvar, 0.9)) + 1
        dim_95 = int(np.searchsorted(cumvar, 0.95)) + 1
        dim_99 = int(np.searchsorted(cumvar, 0.99)) + 1
        
        svd_results[task_name] = {
            "dim_90": dim_90,
            "dim_95": dim_95,
            "dim_99": dim_99,
            "top10_singular_values": [float(s) for s in S[:10]],
            "top10_cumvar": [float(c) for c in cumvar[:10]],
        }
        
        print(f"    {task_name}: 有效秩(90%var)={dim_90}, "
              f"(95%var)={dim_95}, (99%var)={dim_99}")
        print(f"      Top10 SVD: {[f'{s:.1f}' for s in S[:10]]}")
        print(f"      Top10 cumvar: {[f'{c:.3f}' for c in cumvar[:10]]}")
    
    # 两个任务的SVD子空间重叠度 (_cka)
    print(f"\n  === Δh子空间重叠 (CKA) ===")
    
    def linear_cka(X, Y):
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
        def hsic(A, B):
            n = A.shape[0]
            H = np.eye(n) - np.ones((n, n)) / n
            return np.trace(A @ H @ B @ H) / (n - 1)**2
        K_X = X @ X.T
        K_Y = Y @ Y.T
        hsic_xy = hsic(K_X, K_Y)
        hsic_xx = hsic(K_X, K_X)
        hsic_yy = hsic(K_Y, K_Y)
        if hsic_xx < 1e-10 or hsic_yy < 1e-10:
            return 0.0
        return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))
    
    # 逐层计算Δh子空间的CKA
    layer_cka = {}
    for l in range(0, n_layers, 3):
        zh_deltas = []
        trans_deltas = []
        for pair, hiddens in task_hiddens["zh_continue"].items():
            zh_deltas.append(hiddens[l+1] - hiddens[l])
        for pair, hiddens in task_hiddens["translate"].items():
            trans_deltas.append(hiddens[l+1] - hiddens[l])
        
        X = np.array(zh_deltas)
        Y = np.array(trans_deltas)
        cka = linear_cka(X, Y)
        layer_cka[str(l)] = cka
    
    # 找CKA最低的层 (翻译计算最独特的层)
    cka_sorted = sorted(layer_cka.items(), key=lambda x: x[1])
    print(f"    Δh子空间最不同的层 (翻译独特): {[(f'L{l}', f'{c:.3f}') for l, c in cka_sorted[:5]]}")
    print(f"    Δh子空间最相似的层: {[(f'L{l}', f'{c:.3f}') for l, c in cka_sorted[-5:]]}")
    
    results["trajectory"] = trajectory_results
    results["cross_task"] = cross_task_results
    results["svd"] = svd_results
    results["layer_cka"] = layer_cka
    
    # 保存
    save_path = f"tests/glm5_temp/phase102_exp2_{model_name}_transition_dynamics.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  保存到: {save_path}")
    
    release_model(model)
    return results


# ============================================================
# Exp 3: Jacobian分析
# ============================================================
def exp3_jacobian_analysis(model_name):
    """
    Jacobian分析 — ∂h_{l+1}/∂h_l 是真正的局部计算
    
    核心思路:
    - Transformer本质是分段线性动力系统
    - Jacobian J_l = ∂h_{l+1}/∂h_l 描述了层l的局部计算
    - J_l的谱 → 局部计算的稳定/不稳定方向
    - J_l的秩 → 有效计算维度
    - 翻译vs非翻译的Jacobian差异 → 翻译计算的具体实现
    
    实现方式: 数值差分
    J[i,j] ≈ (h_{l+1}(h_l + ε*e_j) - h_{l+1}(h_l)) / ε
    
    但全Jacobian太expensive (d_model^2)
    改用: 随机投影 + 有限差分 估计Jacobian的谱
    """
    print(f"\n{'='*70}")
    print(f"Exp 3: Jacobian分析 — {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")
    
    results = {}
    
    # 测试词
    test_words = [("猫", "cat"), ("水", "water"), ("火", "fire"), ("树", "tree")]
    
    # 随机投影维度 ( Hutchinson trace estimator )
    n_probes = 50
    
    print(f"\n  使用 {n_probes} 个随机探针估计Jacobian谱")
    
    jacobian_results = {}
    
    for zh, en in test_words:
        print(f"\n  处理: {zh}({en})")
        
        # 两种上下文
        zh_prompt = f"{zh}是一种"
        trans_prompt = f"请翻译：{zh} →"
        
        word_jacobian = {}
        
        for task_name, prompt in [("zh_continue", zh_prompt), ("translate", trans_prompt)]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            
            # 获取各层hidden state
            with torch.no_grad():
                base_outputs = model(input_ids, output_hidden_states=True)
            
            base_hiddens = [base_outputs.hidden_states[l][0, -1, :].detach().clone() 
                          for l in range(n_layers + 1)]
            
            # 对选定的层估计Jacobian
            target_layers = [3, 9, 15, 21, 27]
            target_layers = [l for l in target_layers if l < n_layers]
            
            layer_jacobians = {}
            
            for l in target_layers:
                # 获取base层的hidden state
                h_l = base_hiddens[l]  # [d_model]
                
                # 使用随机探针估计Jacobian的trace和Frobenius范数
                eps = 0.01
                
                # 随机探针
                probe_vectors = torch.randn(n_probes, d_model, device=device)
                probe_vectors = F.normalize(probe_vectors, dim=1)
                
                # 对每个探针，计算 J @ probe
                jacobian_probes = []
                
                for p_idx in range(n_probes):
                    probe = probe_vectors[p_idx]  # [d_model]
                    
                    # h_l + eps * probe
                    h_l_perturbed = h_l + eps * probe
                    
                    # 需要从层l开始前向传播
                    # 重建: 前面层的输出 + 扰动后的层l
                    layers = get_layers(model)
                    
                    # 方法: 修改层l的输出
                    captured_h_l_plus_1 = [None]
                    
                    def make_hook(captured):
                        def hook_fn(module, input, output):
                            if isinstance(output, tuple):
                                hidden_states = output[0].clone()
                                # 只修改最后token
                                hidden_states[:, -1, :] = h_l_perturbed.to(hidden_states.dtype).to(device)
                                captured[0] = hidden_states[0, -1, :].detach().cpu().float().numpy()
                                output = (hidden_states,) + output[1:]
                            return output
                        return hook_fn
                    
                    # 这里有个问题: 我们需要从层l开始重新前向传播
                    # 但HuggingFace模型不支持从中间层开始
                    # 简化方案: 用hook在层l的输入处注入扰动
                    
                    # 实际方案: 用hook在层l-1的输出(即层l的输入)处修改
                    # 然后看层l+1的输出变化
                    
                    # 更简单的方案: 只看层l→l+1的变换
                    # 用层l的hidden state作为输入，经过层l+1，看输出
                    
                    # 最简方案: 对整个模型注入扰动在层l，看层l+1之后的变化
                    pass
                
                # 简化为: 用数值差分直接估计 Jacobian-vector 乘积
                # J_l @ v ≈ (f(h_l + ε*v) - f(h_l)) / ε
                # 其中f是从层l到层l+1的映射
                
                # 但这需要逐层前向传播，HuggingFace不支持
                # 替代方案: 估计从层l到最后输出的Jacobian
                # J_{l→end} @ v ≈ (logits(h_l + ε*v) - logits(h_l)) / ε
                
                # 这给出的是"层l的hidden state对最终输出的影响"
                # 而不是层间的Jacobian
                
                # 让我用一种更实际的方法:
                # 1. 在层l的residual stream添加扰动 ε*v
                # 2. 看最终logits的变化
                # 3. 这给出的是"层l的扰动如何传播到输出"
                
                pass
            
            # 实际实现: 用hook在层l注入扰动，看最终logits变化
            task_jacobians = {}
            
            for l in target_layers:
                logit_changes = []
                
                for p_idx in range(n_probes):
                    probe = probe_vectors[p_idx]
                    
                    # 干预: 在层l注入 eps*probe
                    try:
                        intervened_logits = intervene_at_layer(
                            model, tokenizer, prompt, l, probe, eps, device
                        )
                    except:
                        continue
                    
                    # baseline logits
                    with torch.no_grad():
                        base_logits = model(input_ids).logits[0, -1, :].float().cpu()
                    
                    # logits变化
                    delta_logits = (intervened_logits - base_logits).numpy()
                    logit_changes.append(delta_logits)
                
                if logit_changes:
                    logit_changes = np.array(logit_changes)  # [n_probes, vocab_size]
                    
                    # 每个探针引起的logits变化范数
                    change_norms = np.linalg.norm(logit_changes, axis=1)
                    mean_change_norm = float(np.mean(change_norms))
                    
                    # trace估计: Σ ||J@v_i|| / n_probes
                    trace_estimate = mean_change_norm
                    
                    task_jacobians[str(l)] = {
                        "mean_logit_change_norm": mean_change_norm,
                        "max_logit_change_norm": float(np.max(change_norms)),
                        "min_logit_change_norm": float(np.min(change_norms)),
                    }
            
            word_jacobian[task_name] = task_jacobians
            
            if task_jacobians:
                print(f"    {task_name}:")
                for l_str, stats in task_jacobians.items():
                    print(f"      L{l_str}: mean_logit_change={stats['mean_logit_change_norm']:.4f}")
        
        jacobian_results[f"{zh}_{en}"] = word_jacobian
    
    results["jacobian"] = jacobian_results
    
    # 保存
    save_path = f"tests/glm5_temp/phase102_exp3_{model_name}_jacobian.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  保存到: {save_path}")
    
    release_model(model)
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    
    if args.exp == 1:
        exp1_translation_intervention(args.model)
    elif args.exp == 2:
        exp2_transition_dynamics(args.model)
    elif args.exp == 3:
        exp3_jacobian_analysis(args.model)
