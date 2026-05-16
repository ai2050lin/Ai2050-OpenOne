"""
Phase 190: 最小语义回路发现与组合编码验证
=============================================

核心转向: 从"向量差异几何" → "组合编码结构"

用户批评的核心洞察:
1. polar ⟂ content 是最重要发现 → 分层语义坐标系
2. EPA可能是统计假象 → 需要因果验证
3. "低维流形"可能是PCA幻觉 → 可能是稀疏组合图
4. 需要找"最小语义回路" → 哪些head负责否定/时间/因果等
5. 需要"受控语义输运" → EPA是否是生成坐标?

实验设计:
- Exp1: 语义回路发现 — 通过head ablation找到负责各语义功能的head
- Exp2: 回路组合性 — 不同回路是否独立可组合?
- Exp3: 受控语义输运 — 沿发现的轴移动, 观察生成变化
- Exp4: 稀疏编码 vs 连续流形 — 验证语义是稀疏组合还是连续插值

用法:
  python tests/glm5/phase190_semantic_circuits.py qwen3
  python tests/glm5/phase190_semantic_circuits.py glm4
  python tests/glm5/phase190_semantic_circuits.py deepseek7b
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import time
import json
import numpy as np
import torch
from collections import defaultdict
from itertools import combinations

from model_utils import (get_layers, get_model_info, release_model, 
                          get_W_U, MODEL_CONFIGS)


# ===== 语义功能测试集 =====
# 每个功能用多个句对测试, 确保统计可靠性
SEMANTIC_FUNCTIONS = {
    "negation": {
        "description": "否定 vs 肯定",
        "pairs": [
            ("The cat is sleeping", "The cat is not sleeping"),
            ("She likes the movie", "She does not like the movie"),
            ("They will come tomorrow", "They will not come tomorrow"),
            ("He can swim well", "He cannot swim well"),
            ("The door was open", "The door was not open"),
            ("Birds can fly high", "Birds cannot fly high"),
            ("The food tastes good", "The food does not taste good"),
            ("She has finished work", "She has not finished work"),
            ("The car is running", "The car is not running"),
            ("We should go now", "We should not go now"),
            ("The sky is blue", "The sky is not blue"),
            ("He knows the answer", "He does not know the answer"),
            ("The water is warm", "The water is not warm"),
            ("I understand the problem", "I do not understand the problem"),
            ("The dog barks loudly", "The dog does not bark loudly"),
        ],
    },
    "tense": {
        "description": "时态变化 (现在 vs 过去)",
        "pairs": [
            ("She walks to school", "She walked to school"),
            ("He eats breakfast", "He ate breakfast"),
            ("They play soccer", "They played soccer"),
            ("The bird sings", "The bird sang"),
            ("She writes letters", "She wrote letters"),
            ("He drives fast", "He drove fast"),
            ("The children laugh", "The children laughed"),
            ("She reads books", "She read books"),
            ("The wind blows hard", "The wind blew hard"),
            ("He runs every morning", "He ran every morning"),
            ("She cooks dinner", "She cooked dinner"),
            ("The river flows east", "The river flowed east"),
            ("They build houses", "They built houses"),
            ("She teaches math", "She taught math"),
            ("The dog chases cats", "The dog chased cats"),
        ],
    },
    "number": {
        "description": "单复数变化",
        "pairs": [
            ("The cat sleeps", "The cats sleep"),
            ("A dog barks", "The dogs bark"),
            ("The book is heavy", "The books are heavy"),
            ("A child plays", "Children play"),
            ("The tree grows tall", "The trees grow tall"),
            ("A bird sings", "Birds sing"),
            ("The flower is red", "The flowers are red"),
            ("A student studies", "Students study"),
            ("The house stands alone", "The houses stand alone"),
            ("A fish swims", "Fish swim"),
            ("The star shines bright", "The stars shine bright"),
            ("A car drives past", "Cars drive past"),
            ("The mountain rises high", "The mountains rise high"),
            ("A leaf falls down", "Leaves fall down"),
            ("The river runs deep", "The rivers run deep"),
        ],
    },
    "polarity": {
        "description": "极性对比 (正 vs 负)",
        "pairs": [
            ("She loves the music", "She hates the music"),
            ("The food is delicious", "The food is disgusting"),
            ("He is very brave", "He is very cowardly"),
            ("The weather is beautiful", "The weather is terrible"),
            ("She is extremely happy", "She is extremely sad"),
            ("The plan is brilliant", "The plan is foolish"),
            ("He is very strong", "He is very weak"),
            ("The result was excellent", "The result was awful"),
            ("She is quite generous", "She is quite stingy"),
            ("The movie is fascinating", "The movie is boring"),
            ("He is very kind", "He is very cruel"),
            ("The gift is precious", "The gift is worthless"),
            ("She is very smart", "She is very foolish"),
            ("The house is luxurious", "The house is shabby"),
            ("He is very honest", "He is very dishonest"),
        ],
    },
    "causation": {
        "description": "因果关系",
        "pairs": [
            ("It rained heavily", "Because it rained heavily"),
            ("She studied hard", "Because she studied hard"),
            ("The ice melted", "The ice melted from heat"),
            ("He arrived late", "Since he arrived late"),
            ("The plant died", "The plant died from drought"),
            ("She smiled broadly", "She smiled because she was happy"),
            ("The bridge collapsed", "The bridge collapsed under weight"),
            ("He won the prize", "He won the prize through effort"),
            ("The car stopped", "The car stopped due to traffic"),
            ("She left early", "She left early since she was tired"),
            ("The glass broke", "The glass broke from the impact"),
            ("He laughed loudly", "He laughed because the joke was funny"),
            ("The fire spread", "The fire spread due to the wind"),
            ("She succeeded quickly", "She succeeded through determination"),
            ("The boat sank", "The boat sank because of the storm"),
        ],
    },
}

# 生成用测试句
GENERATION_TEMPLATES = {
    "neutral": "The apple is",
    "eval_positive": "Something good about the apple is",
    "eval_negative": "Something bad about the apple is",
    "potency_strong": "The powerful apple can",
    "potency_weak": "The weak apple cannot",
    "activity_active": "The active apple keeps",
    "activity_passive": "The passive apple stays",
}


def load_model_bf16_auto(model_name: str):
    """BF16 + device_map=auto 加载 (参考 model_demo_bf16.py)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    print(f"[Phase190] Loading {model_name} (bfloat16 + device_map=auto)...")
    print(f"[Phase190] Path: {cfg['path']}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
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
    print(f"[Phase190] {model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB")
    
    return model, tokenizer, device


def get_device_for_input(model):
    """获取输入应放的设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_hidden_states(model, tokenizer, device, sentences, desc=""):
    """提取所有层的隐藏状态"""
    input_device = get_device_for_input(model)
    results = {}
    
    for i, sent in enumerate(sentences):
        if desc and i % 5 == 0:
            print(f"  [{desc}] {i}/{len(sentences)}: {sent[:40]}...")
        
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        
        # 取最后一个token的隐藏状态
        last_pos = attention_mask.sum().item() - 1
        hs = [layer_hs[0, last_pos].float().cpu().numpy() 
              for layer_hs in out.hidden_states]
        results[sent] = hs
    
    return results


def extract_attn_patterns(model, tokenizer, device, sentences, n_layers, desc=""):
    """提取注意力模式 (每个head对最后一个token的注意力分布)"""
    input_device = get_device_for_input(model)
    results = {}
    
    for i, sent in enumerate(sentences):
        if desc and i % 5 == 0:
            print(f"  [{desc}] {i}/{len(sentences)}: {sent[:40]}...")
        
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_attentions=True)
        
        # out.attentions: tuple of (batch, n_heads, seq_len, seq_len)
        if out.attentions is not None:
            last_pos = attention_mask.sum().item() - 1
            # 取最后一个token对各位置的注意力, shape: (n_layers, n_heads, seq_len)
            attn = torch.stack([a[0, :, last_pos, :].float().cpu() 
                                for a in out.attentions])
            results[sent] = attn.numpy()  # (n_layers, n_heads, seq_len)
        else:
            results[sent] = None
    
    return results


# ===== Exp1: 语义回路发现 =====
def exp1_semantic_circuit_discovery(model, tokenizer, device, model_name):
    """
    通过head ablation找到负责各语义功能的head
    
    方法: 
    1. 对每个语义功能, 提取正/负句的hidden states
    2. 计算每个head的输出对最终差异的贡献 (attribution)
    3. 用head ablation验证: 消除top-k head后差异是否消失
    
    不用真正的ablation (太慢), 而用注意力模式差异 + 输出贡献分解
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n{'='*60}")
    print(f"Exp1: 语义回路发现 — {model_name}")
    print(f"{'='*60}")
    
    results = {}
    
    for func_name, func_data in SEMANTIC_FUNCTIONS.items():
        print(f"\n--- {func_name}: {func_data['description']} ---")
        t0 = time.time()
        
        pairs = func_data["pairs"]
        pos_sents = [p[0] for p in pairs]
        neg_sents = [p[1] for p in pairs]
        
        # 提取hidden states
        pos_hs = extract_hidden_states(model, tokenizer, device, pos_sents, 
                                         desc=func_name)
        neg_hs = extract_hidden_states(model, tokenizer, device, neg_sents, 
                                         desc=func_name)
        
        # 提取attention patterns
        print(f"  Extracting attention patterns for {func_name}...")
        pos_attn = extract_attn_patterns(model, tokenizer, device, pos_sents, 
                                          n_layers, desc=f"{func_name}_pos")
        neg_attn = extract_attn_patterns(model, tokenizer, device, neg_sents, 
                                          n_layers, desc=f"{func_name}_neg")
        
        # 对每一层, 计算差异向量
        layer_deltas = {}
        for layer_idx in range(n_layers + 1):  # +1 for embedding layer
            deltas = []
            for p, n in pairs:
                if p in pos_hs and n in neg_hs:
                    delta = pos_hs[p][layer_idx] - neg_hs[n][layer_idx]
                    deltas.append(delta)
            if deltas:
                mean_delta = np.mean(deltas, axis=0)
                layer_deltas[layer_idx] = {
                    "mean_norm": float(np.linalg.norm(mean_delta)),
                    "individual_norms": [float(np.linalg.norm(d)) for d in deltas],
                    "cosine_sim": float(np.mean([
                        np.dot(deltas[i], deltas[j]) / 
                        (np.linalg.norm(deltas[i]) * np.linalg.norm(deltas[j]) + 1e-10)
                        for i in range(len(deltas)) 
                        for j in range(i+1, len(deltas))
                    ])) if len(deltas) > 1 else 0.0,
                }
        
        # 注意力模式差异分析
        attn_diff = {}
        if pos_attn and neg_attn:
            for layer_idx in range(min(n_layers, len(next(iter(pos_attn.values()))))):
                head_diffs = []
                for p, n in pairs:
                    if (p in pos_attn and pos_attn[p] is not None and 
                        n in neg_attn and neg_attn[n] is not None):
                        pa = pos_attn[p][layer_idx]  # (n_heads, seq_len)
                        na = neg_attn[n][layer_idx]  # (n_heads, seq_len)
                        # 需要对齐序列长度
                        min_len = min(pa.shape[1], na.shape[1])
                        diff = np.abs(pa[:, :min_len] - na[:, :min_len]).mean(axis=1)  # (n_heads,)
                        head_diffs.append(diff)
                
                if head_diffs:
                    mean_diff = np.mean(head_diffs, axis=0)  # (n_heads,)
                    attn_diff[layer_idx] = mean_diff
        
        # 找到注意力差异最大的head (top-5 per function)
        head_ranking = []
        if attn_diff:
            for layer_idx, head_diffs in attn_diff.items():
                for head_idx, diff_val in enumerate(head_diffs):
                    head_ranking.append((layer_idx, head_idx, float(diff_val)))
            head_ranking.sort(key=lambda x: x[2], reverse=True)
        
        # 差异范数的跨层演化
        delta_norms = [layer_deltas.get(l, {}).get("mean_norm", 0) 
                       for l in range(n_layers + 1)]
        peak_layer = np.argmax(delta_norms[1:])  # skip embedding
        peak_norm = delta_norms[peak_layer + 1]
        last_norm = delta_norms[-1]
        
        # 余弦相似度 (个体差异之间的一致性)
        mean_cos = np.mean([layer_deltas.get(l, {}).get("cosine_sim", 0) 
                           for l in range(n_layers + 1) 
                           if layer_deltas.get(l, {}).get("cosine_sim", 0) > 0])
        
        elapsed = time.time() - t0
        print(f"  Peak layer: L{peak_layer}, peak_norm={peak_norm:.2f}, "
              f"last_norm={last_norm:.2f}, mean_cos={mean_cos:.4f}")
        print(f"  Top-5 heads: {[(h[0], h[1], f'{h[2]:.4f}') for h in head_ranking[:5]]}")
        print(f"  Time: {elapsed:.1f}s")
        
        results[func_name] = {
            "layer_deltas": layer_deltas,
            "head_ranking": head_ranking[:20],
            "peak_layer": int(peak_layer),
            "peak_norm": float(peak_norm),
            "last_norm": float(last_norm),
            "mean_cosine_sim": float(mean_cos),
            "n_pairs": len(pairs),
        }
    
    # ===== 跨功能head重叠分析 =====
    print(f"\n{'='*40}")
    print("跨功能head重叠分析:")
    print(f"{'='*40}")
    
    func_top_heads = {}
    for func_name, data in results.items():
        top10 = [(h[0], h[1]) for h in data["head_ranking"][:10]]
        func_top_heads[func_name] = set(top10)
    
    # 两两重叠
    func_names = list(func_top_heads.keys())
    overlap_matrix = np.zeros((len(func_names), len(func_names)))
    for i, f1 in enumerate(func_names):
        for j, f2 in enumerate(func_names):
            overlap = len(func_top_heads[f1] & func_top_heads[f2])
            total = max(len(func_top_heads[f1] | func_top_heads[f2]), 1)
            overlap_matrix[i, j] = overlap / total
    
    print("\nHead重叠率 (Jaccard):")
    header = "         " + "  ".join([f[:7] for f in func_names])
    print(header)
    for i, f1 in enumerate(func_names):
        row = f"{f1[:7]:8s}"
        for j, f2 in enumerate(func_names):
            row += f"  {overlap_matrix[i,j]:.3f}"
        print(row)
    
    # 每个功能特有的head (不与其他任何功能共享)
    all_heads = set()
    for heads in func_top_heads.values():
        all_heads.update(heads)
    
    unique_heads = {}
    for func_name, heads in func_top_heads.items():
        other_heads = set()
        for fn2, h2 in func_top_heads.items():
            if fn2 != func_name:
                other_heads.update(h2)
        unique = heads - other_heads
        unique_heads[func_name] = unique
        print(f"\n{func_name} unique heads (not in any other function): {len(unique)}/{len(heads)}")
        if unique:
            print(f"  Heads: {sorted(unique)}")
    
    # ===== 内容/关系方向正交性验证 (大样本) =====
    print(f"\n{'='*40}")
    print("内容 vs 关系方向正交性 (大样本验证):")
    print(f"{'='*40}")
    
    # 内容差异: category pairs
    category_pairs = [
        ("The cat sleeps", "The dog sleeps"),
        ("The bird flies", "The fish swims"),
        ("The horse runs", "The cow walks"),
        ("The apple is red", "The banana is yellow"),
        ("The car drives", "The boat sails"),
        ("The book opens", "The door closes"),
        ("The sun rises", "The moon appears"),
        ("The chair stands", "The table sits"),
        ("The fire burns", "The ice freezes"),
        ("The rain falls", "The wind blows"),
    ]
    
    # 关系差异: polarity pairs (从上面的polarity中取10对)
    polarity_pairs = SEMANTIC_FUNCTIONS["polarity"]["pairs"][:10]
    
    # 提取最后一层的差异
    content_deltas = []
    relation_deltas = []
    
    cat_s1 = [p[0] for p in category_pairs]
    cat_s2 = [p[1] for p in category_pairs]
    pol_s1 = [p[0] for p in polarity_pairs]
    pol_s2 = [p[1] for p in polarity_pairs]
    
    cat_hs1 = extract_hidden_states(model, tokenizer, device, cat_s1, desc="content_1")
    cat_hs2 = extract_hidden_states(model, tokenizer, device, cat_s2, desc="content_2")
    pol_hs1 = extract_hidden_states(model, tokenizer, device, pol_s1, desc="relation_1")
    pol_hs2 = extract_hidden_states(model, tokenizer, device, pol_s2, desc="relation_2")
    
    last_layer = n_layers  # last hidden state
    for p1, p2 in category_pairs:
        if p1 in cat_hs1 and p2 in cat_hs2:
            delta = cat_hs1[p1][last_layer] - cat_hs2[p2][last_layer]
            content_deltas.append(delta / (np.linalg.norm(delta) + 1e-10))
    
    for p1, p2 in polarity_pairs:
        if p1 in pol_hs1 and p2 in pol_hs2:
            delta = pol_hs1[p1][last_layer] - pol_hs2[p2][last_layer]
            relation_deltas.append(delta / (np.linalg.norm(delta) + 1e-10))
    
    if content_deltas and relation_deltas:
        mean_content = np.mean(content_deltas, axis=0)
        mean_content /= np.linalg.norm(mean_content) + 1e-10
        mean_relation = np.mean(relation_deltas, axis=0)
        mean_relation /= np.linalg.norm(mean_relation) + 1e-10
        
        cross_cos = float(np.abs(np.dot(mean_content, mean_relation)))
        
        # Individual pair cosines
        individual_cos = []
        for cd in content_deltas:
            for rd in relation_deltas:
                individual_cos.append(float(np.abs(np.dot(cd, rd))))
        
        print(f"  Mean content vs mean relation |cos| = {cross_cos:.4f}")
        print(f"  Individual |cos|: mean={np.mean(individual_cos):.4f}, "
              f"std={np.std(individual_cos):.4f}, "
              f"median={np.median(individual_cos):.4f}")
        
        results["content_relation_orthogonality"] = {
            "mean_cos": cross_cos,
            "individual_cos_mean": float(np.mean(individual_cos)),
            "individual_cos_std": float(np.std(individual_cos)),
            "n_content": len(content_deltas),
            "n_relation": len(relation_deltas),
        }
    
    results["overlap_matrix"] = overlap_matrix.tolist()
    results["overlap_labels"] = func_names
    results["unique_heads"] = {k: [str(h) for h in v] for k, v in unique_heads.items()}
    
    return results


# ===== Exp2: 回路组合性测试 =====
def exp2_circuit_composition(model, tokenizer, device, model_name, exp1_results):
    """
    验证不同语义回路是否独立可组合
    
    方法:
    1. 找到negation-specific和tense-specific的head
    2. 构造同时包含两种变化的句子
    3. 检查两种回路是否同时激活, 且独立贡献
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    print(f"\n{'='*60}")
    print(f"Exp2: 回路组合性测试 — {model_name}")
    print(f"{'='*60}")
    
    # 组合测试: 否定 + 时态
    combo_pairs = [
        ("She walks", "She walked", "She does not walk", "She did not walk"),
        ("He runs", "He ran", "He does not run", "He did not run"),
        ("They play", "They played", "They do not play", "They did not play"),
        ("She sings", "She sang", "She does not sing", "She did not sing"),
        ("He writes", "He wrote", "He does not write", "He did not write"),
        ("The bird flies", "The bird flew", "The bird does not fly", "The bird did not fly"),
        ("She cooks", "She cooked", "She does not cook", "She did not cook"),
        ("He reads", "He read", "He does not read", "He did not read"),
        ("They work", "They worked", "They do not work", "They did not work"),
        ("She drives", "She drove", "She does not drive", "She did not drive"),
    ]
    
    all_sents = []
    for quad in combo_pairs:
        all_sents.extend(quad)
    all_sents = list(set(all_sents))
    
    # 提取hidden states
    print("  Extracting hidden states for combo pairs...")
    hs_dict = extract_hidden_states(model, tokenizer, device, all_sents, 
                                     desc="combo")
    
    # 提取attention patterns
    print("  Extracting attention patterns for combo pairs...")
    attn_dict = extract_attn_patterns(model, tokenizer, device, all_sents, 
                                       n_layers, desc="combo_attn")
    
    # 分析组合性
    last_layer = n_layers
    composition_results = []
    
    for quad in combo_pairs:
        base, past, neg, neg_past = quad
        if not all(s in hs_dict for s in quad):
            continue
        
        # 差异向量
        d_tense = hs_dict[past][last_layer] - hs_dict[base][last_layer]
        d_neg = hs_dict[neg][last_layer] - hs_dict[base][last_layer]
        d_combined = hs_dict[neg_past][last_layer] - hs_dict[base][last_layer]
        
        # 如果回路组合, d_combined ≈ d_tense + d_neg
        d_sum = d_tense + d_neg
        
        cos_sum_combined = float(np.dot(d_sum, d_combined) / 
                                  (np.linalg.norm(d_sum) * np.linalg.norm(d_combined) + 1e-10))
        
        # 各个差异的范数
        norm_tense = float(np.linalg.norm(d_tense))
        norm_neg = float(np.linalg.norm(d_neg))
        norm_combined = float(np.linalg.norm(d_combined))
        norm_sum = float(np.linalg.norm(d_sum))
        
        # 差异: combined - sum
        residual_norm = float(np.linalg.norm(d_combined - d_sum))
        
        composition_results.append({
            "quad": quad,
            "cos_sum_combined": cos_sum_combined,
            "norm_tense": norm_tense,
            "norm_neg": norm_neg,
            "norm_combined": norm_combined,
            "norm_sum": norm_sum,
            "residual_norm": residual_norm,
        })
    
    # 汇总
    if composition_results:
        cos_values = [r["cos_sum_combined"] for r in composition_results]
        residual_ratios = [r["residual_norm"] / (r["norm_combined"] + 1e-10) 
                          for r in composition_results]
        
        print(f"\n  Composition test results (n={len(composition_results)}):")
        print(f"  cos(d_tense+d_neg, d_combined): mean={np.mean(cos_values):.4f}, "
              f"std={np.std(cos_values):.4f}")
        print(f"  ||d_combined - d_sum|| / ||d_combined||: mean={np.mean(residual_ratios):.4f}")
        
        # 线性组合质量
        if np.mean(cos_values) > 0.9:
            print("  → 接近线性组合 (cos>0.9), 回路近似独立")
        elif np.mean(cos_values) > 0.7:
            print("  → 部分线性组合 (0.7<cos<0.9), 回路有交互")
        else:
            print("  → 非线性组合 (cos<0.7), 回路强交互/非线性")
    
    return {
        "composition_results": composition_results,
        "mean_cos_sum_combined": float(np.mean(cos_values)) if composition_results else 0,
        "mean_residual_ratio": float(np.mean(residual_ratios)) if composition_results else 0,
    }


# ===== Exp3: 受控语义输运 =====
def exp3_controlled_semantic_transport(model, tokenizer, device, model_name, exp1_results):
    """
    沿发现的语义轴移动, 观察生成是否因果改变
    
    核心验证: 这些轴是否是"生成坐标"?
    
    方法:
    1. 从negation和polarity的对比中提取差异方向
    2. 将基础句的hidden state沿该方向移动
    3. 观察解码后的生成是否改变
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n{'='*60}")
    print(f"Exp3: 受控语义输运 — {model_name}")
    print(f"{'='*60}")
    
    # 获取W_U用于解码
    W_U = get_W_U(model, model_name)  # (d_model, vocab_size) or (vocab_size, d_model)
    if W_U.shape[0] == info.vocab_size:
        W_U = W_U.T  # → (d_model, vocab_size)
    
    transport_results = {}
    
    for func_name in ["negation", "polarity"]:
        if func_name not in SEMANTIC_FUNCTIONS:
            continue
            
        print(f"\n--- Transport along {func_name} axis ---")
        pairs = SEMANTIC_FUNCTIONS[func_name]["pairs"][:10]
        
        # 提取方向向量
        s1_list = [p[0] for p in pairs]
        s2_list = [p[1] for p in pairs]
        hs1 = extract_hidden_states(model, tokenizer, device, s1_list, desc=f"{func_name}_s1")
        hs2 = extract_hidden_states(model, tokenizer, device, s2_list, desc=f"{func_name}_s2")
        
        # 计算平均差异方向 (在最后一层)
        deltas = []
        for p1, p2 in pairs:
            if p1 in hs1 and p2 in hs2:
                delta = hs1[p1][n_layers] - hs2[p2][n_layers]
                deltas.append(delta)
        
        if not deltas:
            print(f"  No deltas for {func_name}, skipping")
            continue
        
        mean_delta = np.mean(deltas, axis=0)
        delta_dir = mean_delta / (np.linalg.norm(mean_delta) + 1e-10)
        
        # 测试句: "The cat is sleeping" (肯定句)
        test_sent = "The cat is sleeping"
        test_hs = extract_hidden_states(model, tokenizer, device, [test_sent])
        
        if test_sent not in test_hs:
            continue
        
        base_h = test_hs[test_sent][n_layers]  # (d_model,)
        
        # 沿方向移动, 用logit lens看top tokens变化
        print(f"  Direction norm: {np.linalg.norm(mean_delta):.2f}")
        print(f"  Base sentence: '{test_sent}'")
        
        # 不做干预, 直接看logit lens: base, base+delta, base-delta
        steps = np.linspace(-2.0, 2.0, 9)
        
        print(f"\n  Logit lens along {func_name} direction:")
        print(f"  step | top-5 tokens")
        print(f"  -----|-----------")
        
        step_results = []
        for alpha in steps:
            h_mod = base_h + alpha * delta_dir * np.linalg.norm(mean_delta)
            logits = h_mod @ W_U  # (vocab_size,)
            top_ids = np.argsort(logits)[-5:][::-1]
            top_tokens = [tokenizer.decode([i]).strip() for i in top_ids]
            top_scores = [float(logits[i]) for i in top_ids]
            
            step_results.append({
                "alpha": float(alpha),
                "top_tokens": top_tokens,
                "top_scores": top_scores,
            })
            print(f"  {alpha:+5.1f} | {', '.join(top_tokens)}")
        
        # 测试更多句子的logit lens变化
        extra_sents = [
            "The weather today is",
            "She thinks that the",
            "The food at the restaurant",
        ]
        
        extra_transport = {}
        for sent in extra_sents:
            sent_hs = extract_hidden_states(model, tokenizer, device, [sent])
            if sent not in sent_hs:
                continue
            
            base = sent_hs[sent][n_layers]
            # 看alpha=0和alpha=±1的top tokens
            for alpha in [-1.0, 0.0, 1.0]:
                h_mod = base + alpha * delta_dir * np.linalg.norm(mean_delta)
                logits = h_mod @ W_U
                top_ids = np.argsort(logits)[-5:][::-1]
                top_tokens = [tokenizer.decode([i]).strip() for i in top_ids]
                key = f"{sent}_{alpha:+.1f}"
                extra_transport[key] = top_tokens
        
        print(f"\n  Extra sentences transport:")
        for key, tokens in extra_transport.items():
            print(f"  {key}: {tokens}")
        
        transport_results[func_name] = {
            "direction_norm": float(np.linalg.norm(mean_delta)),
            "steps": step_results,
            "extra_transport": extra_transport,
        }
    
    return transport_results


# ===== Exp4: 稀疏编码 vs 连续流形 =====
def exp4_sparse_vs_continuous(model, tokenizer, device, model_name):
    """
    验证语义是稀疏组合还是连续插值
    
    方法:
    1. 如果是连续流形: 插值点应该产生合理的语义
    2. 如果是稀疏编码: 插值点应该产生无意义的混合
    
    具体测试:
    - 在两个不相关的概念间插值 (cat ↔ democracy)
    - 在两个相关概念间插值 (cat ↔ dog)
    - 检查中间点的logit lens是否有清晰的token
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n{'='*60}")
    print(f"Exp4: 稀疏编码 vs 连续流形 — {model_name}")
    print(f"{'='*60}")
    
    W_U = get_W_U(model, model_name)
    if W_U.shape[0] == info.vocab_size:
        W_U = W_U.T
    
    # 不相关概念对
    unrelated_pairs = [
        ("The cat", "Democracy is"),
        ("The apple", "Mathematics is"),
        ("She sings", "The parliament"),
    ]
    
    # 相关概念对  
    related_pairs = [
        ("The cat", "The dog"),
        ("The apple", "The banana"),
        ("She sings", "She dances"),
    ]
    
    all_sents = [s for pair in (unrelated_pairs + related_pairs) for s in pair]
    all_sents = list(set(all_sents))
    
    hs_dict = extract_hidden_states(model, tokenizer, device, all_sents, desc="sparse")
    
    def analyze_interpolation(s1, s2, label):
        if s1 not in hs_dict or s2 not in hs_dict:
            return None
        
        h1 = hs_dict[s1][n_layers]
        h2 = hs_dict[s2][n_layers]
        
        # 归一化
        h1_norm = h1 / (np.linalg.norm(h1) + 1e-10)
        h2_norm = h2 / (np.linalg.norm(h2) + 1e-10)
        
        results = []
        alphas = np.linspace(0, 1, 11)
        
        for alpha in alphas:
            h_interp = (1 - alpha) * h1 + alpha * h2
            logits = h_interp @ W_U
            top_ids = np.argsort(logits)[-5:][::-1]
            top_tokens = [tokenizer.decode([i]).strip() for i in top_ids]
            top_scores = [float(logits[i]) for i in top_ids]
            
            # 测量"清晰度" — top-1和top-2的分数差
            clarity = top_scores[0] - top_scores[1] if len(top_scores) > 1 else 0
            
            # 测量top-1 token的熵
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            
            results.append({
                "alpha": float(alpha),
                "top_tokens": top_tokens,
                "clarity": float(clarity),
                "entropy": float(entropy),
            })
        
        # 中间点(alpha=0.5)的清晰度 vs 端点的清晰度
        mid_clarity = results[5]["clarity"]  # alpha=0.5
        end_clarity = (results[0]["clarity"] + results[10]["clarity"]) / 2
        clarity_ratio = mid_clarity / (end_clarity + 1e-10)
        
        mid_entropy = results[5]["entropy"]
        end_entropy = (results[0]["entropy"] + results[10]["entropy"]) / 2
        entropy_ratio = mid_entropy / (end_entropy + 1e-10)
        
        print(f"\n  [{label}] {s1} ↔ {s2}")
        print(f"  Mid clarity: {mid_clarity:.2f}, End clarity: {end_clarity:.2f}, "
              f"ratio: {clarity_ratio:.2f}")
        print(f"  Mid entropy: {mid_entropy:.2f}, End entropy: {end_entropy:.2f}, "
              f"ratio: {entropy_ratio:.2f}")
        for r in results:
            print(f"  alpha={r['alpha']:.1f}: {r['top_tokens']} (clarity={r['clarity']:.2f})")
        
        return {
            "label": label,
            "s1": s1, "s2": s2,
            "clarity_ratio": float(clarity_ratio),
            "entropy_ratio": float(entropy_ratio),
            "mid_top_tokens": results[5]["top_tokens"],
            "steps": results,
        }
    
    results = {"unrelated": [], "related": []}
    
    for s1, s2 in unrelated_pairs:
        r = analyze_interpolation(s1, s2, "unrelated")
        if r:
            results["unrelated"].append(r)
    
    for s1, s2 in related_pairs:
        r = analyze_interpolation(s1, s2, "related")
        if r:
            results["related"].append(r)
    
    # 汇总
    print(f"\n{'='*40}")
    print("Exp4 汇总: 稀疏 vs 连续")
    print(f"{'='*40}")
    
    if results["unrelated"]:
        unrelated_clarity = np.mean([r["clarity_ratio"] for r in results["unrelated"]])
        unrelated_entropy = np.mean([r["entropy_ratio"] for r in results["unrelated"]])
        print(f"  不相关对: clarity_ratio={unrelated_clarity:.3f}, "
              f"entropy_ratio={unrelated_entropy:.3f}")
    
    if results["related"]:
        related_clarity = np.mean([r["clarity_ratio"] for r in results["related"]])
        related_entropy = np.mean([r["entropy_ratio"] for r in results["related"]])
        print(f"  相关对: clarity_ratio={related_clarity:.3f}, "
              f"entropy_ratio={related_entropy:.3f}")
    
    # 判断
    if results["unrelated"] and results["related"]:
        # 如果不相关对的中间点清晰度远低于相关对 → 更像稀疏编码
        # 如果两者差异不大 → 更像连续流形
        clarity_diff = related_clarity - unrelated_clarity
        if clarity_diff > 0.3:
            print(f"  → 差异大(clarity_diff={clarity_diff:.3f}), 更像稀疏组合编码")
        else:
            print(f"  → 差异小(clarity_diff={clarity_diff:.3f}), 更像连续流形")
    
    return results


# ===== 主函数 =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    print(f"\n{'#'*70}")
    print(f"# Phase 190: 最小语义回路发现与组合编码验证")
    print(f"# Model: {model_name}")
    print(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    # 加载模型
    model, tokenizer, device = load_model_bf16_auto(model_name)
    info = get_model_info(model, model_name)
    print(f"  n_layers={info.n_layers}, d_model={info.d_model}, vocab={info.vocab_size}")
    
    all_results = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    try:
        # Exp1: 语义回路发现
        exp1_results = exp1_semantic_circuit_discovery(model, tokenizer, device, model_name)
        all_results["exp1"] = exp1_results
        
        # Exp2: 回路组合性
        exp2_results = exp2_circuit_composition(model, tokenizer, device, model_name, exp1_results)
        all_results["exp2"] = exp2_results
        
        # Exp3: 受控语义输运
        exp3_results = exp3_controlled_semantic_transport(model, tokenizer, device, model_name, exp1_results)
        all_results["exp3"] = exp3_results
        
        # Exp4: 稀疏 vs 连续
        exp4_results = exp4_sparse_vs_continuous(model, tokenizer, device, model_name)
        all_results["exp4"] = exp4_results
        
    except Exception as e:
        print(f"\n!!! Error: {e}")
        import traceback
        traceback.print_exc()
    
    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    
    # 保存结果
    out_path = f"tests/glm5_temp/phase190_{model_name}_{time.strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}")
    
    print(f"\n{'#'*70}")
    print(f"# Phase 190 COMPLETE — {model_name}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
