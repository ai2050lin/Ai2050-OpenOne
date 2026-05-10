"""
Phase 94: 语义结构图谱 — 特殊结构发现
======================================
核心思路转变: 不再问"全局动力学像什么"，而是问"特定语言功能的计算结构是什么"

目标: 寻找Transformer内部反复出现的稳定计算结构(computational motifs)

结构类型:
  Type 1: 字位约束结构 (Character Position Constraint)
          "苹果中的第二个字是__" → "果"
  
  Type 2: 翻译对齐结构 (Translation Alignment)
          "苹果的英文是__" → "apple"
  
  Type 3: 组合结构 (Compositional Structure)
          "红苹果的颜色是__" → "红"
          "红苹果的水果类型是__" → "苹果"
  
  Type 4: 否定结构 (Negation Structure)
          "猫不是一种__" → 什么?

  Type 5: 类比结构 (Analogy Structure)
          "苹果之于水果，如同狗之于__" → "动物"

方法论:
  1. Logit Lens — 每层投影到词汇空间，追踪"答案"何时出现
  2. Information Probe — 线性探针检测特定信息在何时/何地可用
  3. Subspace Analysis — 找到编码特定信息的子空间
  4. 跨示例持续性 — 同类结构是否跨不同输入稳定存在
  5. 架构先验控制 — 训练模型 vs 随机模型对比

Run:
  python tests/glm5/ccml_phase94_structure_atlas.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase94_structure_atlas.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase94_structure_atlas.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase94_structure_atlas.py --model qwen3 --exp 4
  python tests/glm5/ccml_phase94_structure_atlas.py --model qwen3 --exp 5
  python tests/glm5/ccml_phase94_structure_atlas.py --model deepseek7b --exp 1
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F_torch
import numpy as np
import argparse
import gc
import json
import time
from collections import defaultdict

from model_utils import load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS

# ============================================================
# Structure 1: Character Position Constraint
# ============================================================
CHAR_POSITION_TESTS = [
    # (prompt, target_char, word, position)
    # 使用"XX的第N个字是什么？答案是"格式，经测试模型能正确回答
    ("苹果的第二个字是什么？答案是", "果", "苹果", 2),
    ("清华大学的第三个字是什么？答案是", "华", "清华大学", 3),
    ("中华人民共和国的第四个字是什么？答案是", "人", "中华人民共和国", 4),
    ("人工智能的第一个字是什么？答案是", "人", "人工智能", 1),
    ("深度学习的第三个字是什么？答案是", "学", "深度学习", 3),
    ("自然语言处理的第五个字是什么？答案是", "处", "自然语言处理", 5),
    ("机器学习的第二个字是什么？答案是", "器", "机器学习", 2),
    ("数据科学的第四个字是什么？答案是", "科", "数据科学", 4),
    ("计算机的第三个字是什么？答案是", "算", "计算机", 3),
    ("信息技术的第二个字是什么？答案是", "息", "信息技术", 2),
]

# ============================================================
# Structure 2: Translation Alignment
# ============================================================
TRANSLATION_TESTS = [
    # (prompt, target_english, chinese_word)
    ("苹果的英文是", "apple", "苹果"),
    ("猫的英文是", "cat", "猫"),
    ("狗的英文是", "dog", "狗"),
    ("书的英文是", "book", "书"),
    ("水的英文是", "water", "水"),
    ("火的英文是", "fire", "火"),
    ("山的英文是", "mountain", "山"),
    ("花的英文是", "flower", "花"),
    ("鱼的英文是", "fish", "鱼"),
    ("太阳的英文是", "sun", "太阳"),
    ("月亮的英文是", "moon", "月亮"),
    ("学校的英文是", "school", "学校"),
    ("红色的英文是", "red", "红色"),
    ("蓝色的英文是", "blue", "蓝色"),
    ("绿色的英文是", "green", "绿色"),
]

# ============================================================
# Structure 3: Compositional Structure
# ============================================================
COMPOSITION_TESTS = [
    # (prompt, target, modifier, noun)
    ("红苹果的颜色是什么？答案是", "红", "红", "苹果"),
    ("红苹果是什么水果？答案是", "苹果", "红", "苹果"),
    ("大房子的属性是什么？答案是", "大", "大", "房子"),
    ("大房子是什么建筑？答案是", "房子", "大", "房子"),
    ("黑猫的颜色是什么？答案是", "黑", "黑", "猫"),
    ("黑猫是什么动物？答案是", "猫", "黑", "猫"),
    ("高山的特点是什么？答案是", "高", "高", "山"),
    ("高山是什么地形？答案是", "山", "高", "山"),
    ("快车的速度特点是什么？答案是", "快", "快", "车"),
    ("快车是什么交通工具？答案是", "车", "快", "车"),
]

# ============================================================
# Structure 4: Negation Structure
# ============================================================
NEGATION_TESTS = [
    # (affirmative_prompt, negation_prompt, target)
    ("猫是一种动物，这是真的吗？", "猫不是一种植物，这是真的吗？", "动物"),
    ("水是液体，这是真的吗？", "水不是固体，这是真的吗？", "液体"),
    ("铁是金属，这是真的吗？", "铁不是塑料，这是真的吗？", "金属"),
    ("地球是行星，这是真的吗？", "地球不是恒星，这是真的吗？", "行星"),
    ("雪是白色的，这是真的吗？", "雪不是黑色的，这是真的吗？", "白色"),
]

# ============================================================
# Structure 5: Analogy Structure
# ============================================================
ANALOGY_TESTS = [
    # (prompt, target, A, B, C)
    ("苹果属于水果，狗属于什么？答案是", "动物", "苹果", "水果", "狗"),
    ("医生在医院工作，教师在哪里工作？答案是", "学校", "医生", "医院", "教师"),
    ("汽车在公路上行驶，飞机在哪里飞行？答案是", "天空", "汽车", "公路", "飞机"),
    ("书提供知识，食物提供什么？答案是", "营养", "书", "知识", "食物"),
    ("眼睛负责视觉，耳朵负责什么？答案是", "听觉", "眼睛", "视觉", "耳朵"),
]


def get_logit_lens(model, tokenizer, device, prompt, n_layers, top_k=10):
    """
    Logit Lens: 将每层hidden state投影到词汇空间
    返回每层的top-k预测token及其概率
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
    
    # Get unembedding matrix
    W_U = model.lm_head.weight.data.float()  # [vocab_size, d_model]
    
    results = {}
    for layer_idx in range(n_layers + 1):
        h = outputs.hidden_states[layer_idx][0, -1, :].float()  # [d_model]
        
        # Project to vocab space
        logits = h @ W_U.T  # [vocab_size]
        probs = torch.softmax(logits, dim=-1)
        
        top_k_vals, top_k_ids = torch.topk(probs, top_k)
        
        top_tokens = []
        for i in range(top_k):
            token_id = top_k_ids[i].item()
            token_str = tokenizer.decode([token_id])
            prob = top_k_vals[i].item()
            top_tokens.append({
                "token_id": token_id,
                "token": token_str,
                "prob": prob
            })
        
        results[layer_idx] = {
            "top_tokens": top_tokens,
            "logits_norm": torch.norm(logits).item(),
            "hidden_norm": torch.norm(h).item()
        }
    
    return results


def find_target_rank(model, tokenizer, device, prompt, target_str, n_layers):
    """
    在每层的logit分布中找到目标token的排名和概率
    支持多种目标编码方式（带/不带引号、空格等）
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Get target token ids - try multiple formats
    target_variants = [
        target_str,
        f" {target_str}",  # with leading space
        f'"{target_str}',  # with opening quote
        f"\"{target_str}",  # with opening quote
        f"「{target_str}",  # Chinese quote
    ]
    
    target_ids = set()
    for variant in target_variants:
        try:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            target_ids.update(ids)
        except:
            pass
    
    # Also add the bare target
    try:
        bare_ids = tokenizer.encode(target_str, add_special_tokens=False)
        target_ids.update(bare_ids)
    except:
        pass
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
    
    W_U = model.lm_head.weight.data.float()
    
    results = {}
    for layer_idx in range(n_layers + 1):
        h = outputs.hidden_states[layer_idx][0, -1, :].float()
        logits = h @ W_U.T
        probs = torch.softmax(logits, dim=-1)
        
        # For each target token variant
        target_info = []
        for tid in target_ids:
            if 0 <= tid < probs.shape[0]:
                target_prob = probs[tid].item()
                rank = (probs > target_prob).sum().item() + 1
                target_info.append({
                    "token_id": tid,
                    "token": tokenizer.decode([tid]),
                    "prob": target_prob,
                    "rank": rank
                })
        
        # Keep only the best variant per layer
        if target_info:
            best = min(target_info, key=lambda x: x["rank"])
            results[layer_idx] = [best]
        else:
            results[layer_idx] = []
    
    return results


def compute_emergence_layer(ranks_per_layer):
    """
    找到目标token首次进入top-1的层
    返回 emergence_layer, max_prob_layer, max_prob, final_prob
    """
    emergence_layer = None
    max_prob = 0
    max_prob_layer = None
    final_prob = 0
    
    for layer_idx, info_list in sorted(ranks_per_layer.items()):
        if not info_list:
            continue
        # Take the best variant (lowest rank)
        best_info = min(info_list, key=lambda x: x["rank"])
        
        if best_info["rank"] == 1 and emergence_layer is None:
            emergence_layer = layer_idx
        if best_info["prob"] > max_prob:
            max_prob = best_info["prob"]
            max_prob_layer = layer_idx
        final_prob = best_info["prob"]
    
    return emergence_layer, max_prob_layer, max_prob, final_prob


def subspace_analysis(model, tokenizer, device, prompts, target_strs, n_layers, n_components=20):
    """
    分析编码特定信息的子空间
    对同类prompt的hidden states做PCA，分析主成分
    """
    all_hidden = {l: [] for l in range(n_layers + 1)}
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], 
                          attention_mask=inputs["attention_mask"],
                          output_hidden_states=True)
        
        for l in range(n_layers + 1):
            h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
            all_hidden[l].append(h)
    
    # PCA per layer
    from sklearn.decomposition import PCA
    
    results = {}
    for l in range(n_layers + 1):
        H = np.array(all_hidden[l])  # [n_prompts, d_model]
        H_centered = H - H.mean(axis=0, keepdims=True)
        
        # Variance explained by first n_components
        if H_centered.shape[0] > 1:
            pca = PCA(n_components=min(n_components, H_centered.shape[0] - 1))
            pca.fit(H_centered)
            var_explained = pca.explained_variance_ratio_
            cumvar = np.cumsum(var_explained)
            
            # Effective dimension (where cumvar > 0.95)
            eff_dim = np.searchsorted(cumvar, 0.95) + 1
            
            results[l] = {
                "var_explained": var_explained.tolist(),
                "cumvar_5": cumvar[4] if len(cumvar) >= 5 else cumvar[-1],
                "cumvar_10": cumvar[9] if len(cumvar) >= 10 else cumvar[-1],
                "eff_dim_95": int(eff_dim),
                "total_var": float(H_centered.var(axis=0).sum())
            }
        else:
            results[l] = {"eff_dim_95": 0, "total_var": 0, "cumvar_5": 0, "cumvar_10": 0}
    
    return results


def cross_example_persistence(model, tokenizer, device, test_cases, target_key, n_layers):
    """
    测试结构是否跨示例持续存在
    计算同类示例的"涌现层"分布
    """
    emergence_layers = []
    max_probs = []
    
    for case in test_cases:
        prompt = case[0]
        target = case[1]
        
        ranks = find_target_rank(model, tokenizer, device, prompt, target, n_layers)
        emergence, _, max_prob, _ = compute_emergence_layer(ranks)
        
        if emergence is not None:
            emergence_layers.append(emergence)
        max_probs.append(max_prob)
    
    if emergence_layers:
        return {
            "mean_emergence": np.mean(emergence_layers),
            "std_emergence": np.std(emergence_layers),
            "min_emergence": np.min(emergence_layers),
            "max_emergence": np.max(emergence_layers),
            "n_emerged": len(emergence_layers),
            "n_total": len(test_cases),
            "mean_max_prob": np.mean(max_probs) if max_probs else 0
        }
    else:
        return {
            "mean_emergence": None,
            "n_emerged": 0,
            "n_total": len(test_cases),
            "mean_max_prob": np.mean(max_probs) if max_probs else 0
        }


def architecture_control(model, tokenizer, device, prompt, target_str, n_layers):
    """
    架构先验控制: 比较训练模型和随机初始化模型
    检查"答案涌现"是否是训练的结果
    """
    # Trained model results
    trained_ranks = find_target_rank(model, tokenizer, device, prompt, target_str, n_layers)
    
    # Create random model with same architecture
    import copy
    random_model = copy.deepcopy(model)
    
    # Randomize all weights (but keep architecture)
    for param in random_model.parameters():
        param.data.normal_(0, param.data.std() if param.data.std() > 0 else 0.02)
    
    random_model.eval()
    random_model.to(device)
    
    # Random model results
    random_ranks = find_target_rank(random_model, device, prompt, target_str, n_layers)
    
    # Clean up
    del random_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return trained_ranks, random_ranks


# ============================================================
# Experiment 1: Character Position Constraint
# ============================================================
def exp1_char_position(model_name):
    """字位约束结构分析"""
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print(f"\n{'='*70}")
    print(f"实验1: 字位约束结构 (Character Position Constraint)")
    print(f"模型: {model_name}, 层数: {n_layers}, 维度: {d_model}")
    print(f"{'='*70}")
    
    results = {"structure": "char_position", "model": model_name, "cases": []}
    
    for idx, (prompt, target, word, pos) in enumerate(CHAR_POSITION_TESTS):
        print(f"\n--- 案例 {idx+1}: '{prompt}' → 目标: '{target}' (第{pos}字) ---")
        
        # 1. Logit Lens
        ranks = find_target_rank(model, tokenizer, device, prompt, target, n_layers)
        emergence, max_prob_layer, max_prob, final_prob = compute_emergence_layer(ranks)
        
        print(f"  涌现层: {emergence}, 最终概率: {final_prob:.4f}, 最大概率: {max_prob:.4f}")
        
        # 2. Detailed layer-by-layer
        key_layers = list(range(0, n_layers + 1, max(1, n_layers // 8)))
        if n_layers not in key_layers:
            key_layers.append(n_layers)
        
        for l in key_layers:
            info_list = ranks.get(l, [])
            for info_item in info_list:
                if info_item["rank"] <= 5:
                    print(f"  L{l}: rank={info_item['rank']}, prob={info_item['prob']:.6f}, token='{info_item['token']}'")
        
        case_result = {
            "prompt": prompt, "target": target, "word": word, "position": pos,
            "emergence_layer": emergence, "max_prob": max_prob, "final_prob": final_prob
        }
        results["cases"].append(case_result)
    
    # 3. Cross-example persistence
    print(f"\n--- 跨示例持续性 ---")
    persistence = cross_example_persistence(
        model, tokenizer, device, CHAR_POSITION_TESTS, "target", n_layers)
    me = persistence.get('mean_emergence')
    se = persistence.get('std_emergence')
    if me is not None:
        print(f"  涌现层: mean={me:.1f} ± {se:.1f}")
        print(f"  范围: [{persistence.get('min_emergence')}, {persistence.get('max_emergence')}]")
    else:
        print(f"  涌现层: None (无示例涌现)")
    print(f"  成功率: {persistence['n_emerged']}/{persistence['n_total']}")
    print(f"  平均最大概率: {persistence['mean_max_prob']:.4f}")
    results["persistence"] = persistence
    
    # 4. Subspace analysis
    print(f"\n--- 子空间分析 ---")
    prompts_only = [case[0] for case in CHAR_POSITION_TESTS]
    subspace = subspace_analysis(model, tokenizer, device, prompts_only, 
                                 [case[1] for case in CHAR_POSITION_TESTS], n_layers)
    
    for l in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]:
        if l in subspace:
            s = subspace[l]
            print(f"  L{l}: eff_dim={s['eff_dim_95']}, cumvar_5={s['cumvar_5']:.4f}, cumvar_10={s['cumvar_10']:.4f}")
    results["subspace"] = subspace
    
    # 5. Position dependency analysis
    print(f"\n--- 位置依赖性分析 ---")
    pos_groups = defaultdict(list)
    for case_result in results["cases"]:
        pos_groups[case_result["position"]].append(case_result)
    
    for pos, cases in sorted(pos_groups.items()):
        emergences = [c["emergence_layer"] for c in cases if c["emergence_layer"] is not None]
        if emergences:
            print(f"  位置{pos}: mean_emergence={np.mean(emergences):.1f}, n={len(emergences)}")
    
    # Save
    output_path = f"tests/glm5_temp/phase94_{model_name}_exp1_char_position.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {output_path}")
    
    release_model(model)
    return results


# ============================================================
# Experiment 2: Translation Alignment
# ============================================================
def exp2_translation(model_name):
    """翻译对齐结构分析"""
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    print(f"\n{'='*70}")
    print(f"实验2: 翻译对齐结构 (Translation Alignment)")
    print(f"模型: {model_name}, 层数: {n_layers}")
    print(f"{'='*70}")
    
    results = {"structure": "translation", "model": model_name, "cases": []}
    
    for idx, (prompt, target_en, chinese) in enumerate(TRANSLATION_TESTS):
        print(f"\n--- 案例 {idx+1}: '{prompt}' → 目标: '{target_en}' ---")
        
        # Try multiple target formats
        targets_to_try = [target_en, f" {target_en}", target_en.lower(), f" {target_en.lower()}"]
        
        best_emergence = None
        best_final_prob = 0
        best_target_format = None
        best_ranks = None
        
        for target_fmt in targets_to_try:
            try:
                ranks = find_target_rank(model, tokenizer, device, prompt, target_fmt, n_layers)
                emergence, _, max_prob, final_prob = compute_emergence_layer(ranks)
                
                if best_emergence is None or (emergence is not None and 
                    (best_emergence is None or emergence < best_emergence)):
                    best_emergence = emergence
                    best_final_prob = final_prob
                    best_target_format = target_fmt
                    best_ranks = ranks
            except:
                continue
        
        if best_emergence is not None:
            print(f"  涌现层: {best_emergence}, 最终概率: {best_final_prob:.4f}, 格式: '{best_target_format}'")
        else:
            print(f"  未涌现 (最终概率: {best_final_prob:.4f})")
        
        # Logit Lens for key layers
        if best_ranks:
            key_layers = list(range(0, n_layers + 1, max(1, n_layers // 8)))
            if n_layers not in key_layers:
                key_layers.append(n_layers)
            for l in key_layers:
                for info_item in best_ranks.get(l, []):
                    if info_item["rank"] <= 3:
                        print(f"  L{l}: rank={info_item['rank']}, prob={info_item['prob']:.6f}")
        
        results["cases"].append({
            "prompt": prompt, "target_en": target_en, "chinese": chinese,
            "emergence_layer": best_emergence, "final_prob": best_final_prob
        })
    
    # Cross-example persistence
    persistence = cross_example_persistence(
        model, tokenizer, device, 
        [(c[0], c[1]) for c in TRANSLATION_TESTS], 
        "target_en", n_layers)
    
    print(f"\n--- 翻译对齐跨示例持续性 ---")
    print(f"  涌现层: mean={persistence['mean_emergence']:.1f} ± {persistence['std_emergence']:.1f}")
    print(f"  成功率: {persistence['n_emerged']}/{persistence['n_total']}")
    results["persistence"] = persistence
    
    # Subspace analysis
    prompts_only = [case[0] for case in TRANSLATION_TESTS]
    subspace = subspace_analysis(model, tokenizer, device, prompts_only,
                                 [case[1] for case in TRANSLATION_TESTS], n_layers)
    print(f"\n--- 翻译子空间 ---")
    for l in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers]:
        if l in subspace:
            s = subspace[l]
            print(f"  L{l}: eff_dim={s['eff_dim_95']}, cumvar_5={s['cumvar_5']:.4f}")
    results["subspace"] = subspace
    
    # Save
    output_path = f"tests/glm5_temp/phase94_{model_name}_exp2_translation.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {output_path}")
    
    release_model(model)
    return results


# ============================================================
# Experiment 3: Compositional Structure
# ============================================================
def exp3_composition(model_name):
    """组合结构分析"""
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    print(f"\n{'='*70}")
    print(f"实验3: 组合结构 (Compositional Structure)")
    print(f"模型: {model_name}, 层数: {n_layers}")
    print(f"{'='*70}")
    
    results = {"structure": "composition", "model": model_name, "cases": []}
    
    for idx, (prompt, target, modifier, noun) in enumerate(COMPOSITION_TESTS):
        print(f"\n--- 案例 {idx+1}: '{prompt}' → 目标: '{target}' (修饰: '{modifier}', 名词: '{noun}') ---")
        
        ranks = find_target_rank(model, tokenizer, device, prompt, target, n_layers)
        emergence, _, max_prob, final_prob = compute_emergence_layer(ranks)
        
        print(f"  涌现层: {emergence}, 最终概率: {final_prob:.4f}")
        
        # Also track modifier and noun separately
        try:
            modifier_ranks = find_target_rank(model, tokenizer, device, prompt, modifier, n_layers)
            noun_ranks = find_target_rank(model, tokenizer, device, prompt, noun, n_layers)
            
            mod_emergence, _, _, mod_final = compute_emergence_layer(modifier_ranks)
            noun_emergence, _, _, noun_final = compute_emergence_layer(noun_ranks)
            
            print(f"  修饰词'{modifier}': 涌现层={mod_emergence}, 最终概率={mod_final:.4f}")
            print(f"  名词'{noun}': 涌现层={noun_emergence}, 最终概率={noun_final:.4f}")
        except:
            mod_emergence = noun_emergence = None
            mod_final = noun_final = 0
        
        results["cases"].append({
            "prompt": prompt, "target": target, "modifier": modifier, "noun": noun,
            "emergence_layer": emergence, "final_prob": final_prob,
            "modifier_emergence": mod_emergence, "modifier_prob": mod_final,
            "noun_emergence": noun_emergence, "noun_prob": noun_final
        })
    
    # Compare modifier-first vs noun-first emergence
    print(f"\n--- 组合vs分解涌现对比 ---")
    for case in results["cases"]:
        target_em = case["emergence_layer"]
        mod_em = case["modifier_emergence"]
        noun_em = case["noun_emergence"]
        print(f"  '{case['prompt'][:10]}...': target@L{target_em}, "
              f"modifier@L{mod_em}, noun@L{noun_em}")
    
    # Save
    output_path = f"tests/glm5_temp/phase94_{model_name}_exp3_composition.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {output_path}")
    
    release_model(model)
    return results


# ============================================================
# Experiment 4: Negation Structure  
# ============================================================
def exp4_negation(model_name):
    """否定结构分析"""
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    print(f"\n{'='*70}")
    print(f"实验4: 否定结构 (Negation Structure)")
    print(f"模型: {model_name}, 层数: {n_layers}")
    print(f"{'='*70}")
    
    results = {"structure": "negation", "model": model_name, "cases": []}
    
    for idx, (affirm, negate, target) in enumerate(NEGATION_TESTS):
        print(f"\n--- 案例 {idx+1} ---")
        print(f"  肯定: '{affirm}'")
        print(f"  否定: '{negate}' → 目标: '{target}'")
        
        # Compare affirmative vs negation
        try:
            affirm_ranks = find_target_rank(model, tokenizer, device, affirm, target, n_layers)
            negate_ranks = find_target_rank(model, tokenizer, device, negate, target, n_layers)
            
            affirm_em, _, _, affirm_final = compute_emergence_layer(affirm_ranks)
            negate_em, _, _, negate_final = compute_emergence_layer(negate_ranks)
            
            print(f"  肯定句: 涌现层={affirm_em}, 最终概率={affirm_final:.4f}")
            print(f"  否定句: 涌现层={negate_em}, 最终概率={negate_final:.4f}")
            print(f"  概率变化: {affirm_final:.4f} → {negate_final:.4f} (Δ={negate_final-affirm_final:.4f})")
        except:
            affirm_em = negate_em = None
            affirm_final = negate_final = 0
        
        # Layer-by-layer comparison
        key_layers = list(range(0, n_layers + 1, max(1, n_layers // 6)))
        if n_layers not in key_layers:
            key_layers.append(n_layers)
        
        print(f"  逐层概率对比 (肯定 vs 否定):")
        for l in key_layers:
            a_prob = 0
            n_prob = 0
            for info_item in affirm_ranks.get(l, []):
                a_prob = max(a_prob, info_item["prob"])
            for info_item in negate_ranks.get(l, []):
                n_prob = max(n_prob, info_item["prob"])
            if a_prob > 0.001 or n_prob > 0.001:
                marker = " ***" if abs(a_prob - n_prob) > 0.05 else ""
                print(f"    L{l}: {a_prob:.4f} vs {n_prob:.4f}{marker}")
        
        results["cases"].append({
            "affirmative": affirm, "negation": negate, "target": target,
            "affirm_emergence": affirm_em, "negate_emergence": negate_em,
            "affirm_final_prob": affirm_final, "negate_final_prob": negate_final
        })
    
    # Save
    output_path = f"tests/glm5_temp/phase94_{model_name}_exp4_negation.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {output_path}")
    
    release_model(model)
    return results


# ============================================================
# Experiment 5: Analogy Structure
# ============================================================
def exp5_analogy(model_name):
    """类比结构分析"""
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    print(f"\n{'='*70}")
    print(f"实验5: 类比结构 (Analogy Structure)")
    print(f"模型: {model_name}, 层数: {n_layers}")
    print(f"{'='*70}")
    
    results = {"structure": "analogy", "model": model_name, "cases": []}
    
    for idx, (prompt, target, A, B, C) in enumerate(ANALOGY_TESTS):
        print(f"\n--- 案例 {idx+1}: '{prompt}' → 目标: '{target}' ({A}:{B}::{C}:?) ---")
        
        try:
            ranks = find_target_rank(model, tokenizer, device, prompt, target, n_layers)
            emergence, _, max_prob, final_prob = compute_emergence_layer(ranks)
            
            print(f"  涌现层: {emergence}, 最终概率: {final_prob:.4f}")
        except:
            emergence = None
            final_prob = 0
        
        # Also test simpler version: "C的类别是"
        simple_prompt = f"{C}的类别是"
        try:
            simple_ranks = find_target_rank(model, tokenizer, device, simple_prompt, target, n_layers)
            simple_em, _, _, simple_final = compute_emergence_layer(simple_ranks)
            print(f"  简化版'{simple_prompt}': 涌现层={simple_em}, 最终概率={simple_final:.4f}")
        except:
            simple_em = None
            simple_final = 0
        
        results["cases"].append({
            "prompt": prompt, "target": target, "A": A, "B": B, "C": C,
            "analogy_emergence": emergence, "analogy_prob": final_prob,
            "simple_emergence": simple_em, "simple_prob": simple_final
        })
    
    # Save
    output_path = f"tests/glm5_temp/phase94_{model_name}_exp5_analogy.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {output_path}")
    
    release_model(model)
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", 
                       choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--exp", type=str, default="1",
                       help="Experiment number (1-5) or 'all'")
    args = parser.parse_args()
    
    exp_map = {
        "1": exp1_char_position,
        "2": exp2_translation,
        "3": exp3_composition,
        "4": exp4_negation,
        "5": exp5_analogy,
    }
    
    if args.exp == "all":
        for exp_num in ["1", "2", "3", "4", "5"]:
            print(f"\n\n{'#'*70}")
            print(f"# Running Experiment {exp_num}")
            print(f"{'#'*70}")
            exp_map[exp_num](args.model)
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)
    else:
        exp_map[args.exp](args.model)
