"""
Phase 176: ★★★ 约束代数 — 从 Hidden State 坐标系到约束不变量 ★★★
================================================================

用户核心洞察:
  Hidden state h_l 只是神经网络内部坐标系, 严重依赖 basis。
  换个基底 R (R^TR=I), 所有 PCA/neuron/direction/cosine 都会变, 但模型功能不变。
  → 真正的研究对象不是 h_l, 而是"哪些约束被满足" — C_l

★★★ 四大实验 — 全部针对"约束不变量", 不是"坐标向量" ★★★

Phase A: ★★★ 约束基发现 (Constraint Basis Discovery) ★★★
  - apple 由哪些最小约束组成?
  - 不是线性组合, 而是"约束交": apple = ∩ C_i
  - 方法: 用 W_U 投影找出哪些 logit 维度被 apple 激活
  - 用"约束满足函数"而非"向量相似度"来衡量概念关系

Phase B: ★★★ 约束激活动力学 (Constraint Activation Dynamics) ★★★
  - apple 生成时, 哪些约束在哪些层激活?
  - 方法: 逐层计算"约束满足状态", 追踪约束闭合过程
  - 关键: L1→L2→...→L_last, 哪些约束先闭合? 哪些后闭合?

Phase C: ★★★ 约束输运拓扑 (Constraint Transport Topology) ★★★
  - 不是 attention weight, 而是"约束如何跨 token 传播"
  - 方法: 主谓一致句子, 追踪 number/gender 约束在哪些层被激活
  - 比较: 主语单数→动词是否携带 number 约束? 在哪一层闭合?

Phase D: ★★★ 约束等价类 — 跨语言约束同构 (Cross-Lingual Constraint Isomorphism) ★★★
  - 不同语言的 apple/苹果/りんご 是否收敛到同一约束流形?
  - 关键改进: 用各语言的母语句子, 不是英文模板!
  - 方法: 计算跨语言约束子空间的重叠度 (Grassmann distance)
  - ★★★ 决定性实验: 删除 EN 约束子空间 → 是否同时影响 ZH/JA/FR? ★★★

★★★ 核心方法论转变 ★★★
  旧: cosine(h_en, h_zh) — 坐标依赖
  新: Grassmann_distance(Subspace_en, Subspace_zh) — 坐标无关!
  
  旧: PCA of hidden states — 坐标依赖
  新: 约束满足函数 σ(W_U @ h) — 语义空间的函数, 坐标不变

Usage: python tests/glm5/phase176_constraint_algebra.py <model_name>
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
# CONCEPT WORDS AND CONSTRAINT DEFINITIONS
# =====================================================================

# ★★★ 约束 = logit 空间中的语义属性 ★★★
# 每个约束由一组验证词定义 — 如果 h 激活这些词的 logit, 则该约束被满足
CONSTRAINTS = {
    "fruit": ["fruit", "sweet", "juicy", "ripe", "fresh", "tropical", "berry"],
    "edible": ["eat", "food", "delicious", "taste", "cook", "meal", "dish"],
    "round": ["round", "circle", "sphere", "ball", "shape", "curved", "globe"],
    "sweet": ["sweet", "sugar", "honey", "candy", "dessert", "flavor", "tasty"],
    "vehicle": ["vehicle", "drive", "engine", "road", "transport", "wheel", "fast"],
    "mechanical": ["engine", "machine", "motor", "metal", "mechanical", "gear", "power"],
    "animal": ["animal", "wild", "pet", "creature", "alive", "species", "mammal"],
    "living": ["alive", "life", "grow", "born", "breathe", "live", "organism"],
    "emotion": ["emotion", "feel", "mood", "mental", "feeling", "express", "sentiment"],
    "human": ["person", "people", "human", "man", "woman", "child", "face"],
    "plant": ["plant", "tree", "grow", "leaf", "root", "flower", "garden"],
    "color": ["color", "red", "green", "blue", "bright", "dark", "yellow"],
    "size": ["big", "small", "large", "tiny", "huge", "size", "long"],
    "location": ["place", "location", "area", "region", "space", "here", "there"],
}

# 概念 = 约束集合
CONCEPT_CONSTRAINTS = {
    "apple":   ["fruit", "edible", "round", "sweet", "plant"],
    "banana":  ["fruit", "edible", "sweet", "plant"],
    "car":     ["vehicle", "mechanical"],
    "bus":     ["vehicle", "mechanical"],
    "cat":     ["animal", "living"],
    "dog":     ["animal", "living"],
    "happy":   ["emotion", "human"],
    "sad":     ["emotion", "human"],
    "tree":    ["plant", "living"],
    "rose":    ["plant", "color"],
}

# ★★★ 跨语言母语句子 — Phase D 关键改进 ★★★
# 每种语言使用自然的母语句子, 不是英文模板!
CROSS_LINGUAL_NATIVE = {
    "apple": {
        "en": ["The apple is sweet", "I ate an apple", "An apple tree grows",
               "She picked an apple", "The apple was red", "Fresh apple juice"],
        "zh": ["苹果很甜", "我吃了一个苹果", "苹果树生长着",
               "她摘了一个苹果", "苹果是红色的", "新鲜苹果汁"],
        "ja": ["りんごは甘い", "りんごを食べた", "りんごの木がある",
               "彼女はりんごを摘んだ", "りんごは赤い", "新鮮なりんごジュース"],
        "fr": ["La pomme est douce", "J'ai mangé une pomme", "Un pommier pousse",
               "Elle a cueilli une pomme", "La pomme était rouge", "Du jus de pomme frais"],
        "es": ["La manzana es dulce", "Comí una manzana", "Un manzano crece",
               "Ella recogió una manzana", "La manzana era roja", "Jugo de manzana fresco"],
    },
    "cat": {
        "en": ["The cat is cute", "I saw a cat", "A cat sat on the mat",
               "She fed the cat", "The black cat", "My cat is small"],
        "zh": ["猫很可爱", "我看见一只猫", "猫坐在垫子上",
               "她喂了猫", "那只黑猫", "我的猫很小"],
        "ja": ["猫は可愛い", "猫を見た", "猫がマットの上に座った",
               "彼女は猫に餌をやった", "その黒猫", "私の猫は小さい"],
        "fr": ["Le chat est mignon", "J'ai vu un chat", "Un chat s'est assis",
               "Elle a nourri le chat", "Le chat noir", "Mon chat est petit"],
        "es": ["El gato es lindo", "Vi un gato", "Un gato se sentó",
               "Ella alimentó al gato", "El gato negro", "Mi gato es pequeño"],
    },
    "car": {
        "en": ["The car is fast", "I drove a car", "A car on the road",
               "She bought a car", "The red car", "My car is new"],
        "zh": ["汽车很快", "我开了一辆车", "路上有一辆车",
               "她买了一辆车", "那辆红色的汽车", "我的车是新的"],
        "ja": ["車は速い", "車を運転した", "道に車がある",
               "彼女は車を買った", "その赤い車", "私の車は新しい"],
        "fr": ["La voiture est rapide", "J'ai conduit une voiture", "Une voiture sur la route",
               "Elle a acheté une voiture", "La voiture rouge", "Ma voiture est neuve"],
        "es": ["El coche es rápido", "Conduje un coche", "Un coche en la carretera",
               "Ella compró un coche", "El coche rojo", "Mi coche es nuevo"],
    },
}

# ★★★ 主谓一致句子 — Phase C 约束输运 ★★★
AGREEMENT_SENTENCES = {
    "singular_correct": [
        "The cat sleeps on the",
        "The dog runs to the",
        "The bird flies over the",
        "The child reads a",
        "The man walks to the",
        "The woman sings a",
    ],
    "singular_wrong": [
        "The cat sleep on the",
        "The dog run to the",
        "The bird fly over the",
        "The child read a",
        "The man walk to the",
        "The woman sing a",
    ],
    "plural_correct": [
        "The cats sleep on the",
        "The dogs run to the",
        "The birds fly over the",
        "The children read a",
        "The men walk to the",
        "The women sing a",
    ],
    "plural_wrong": [
        "The cats sleeps on the",
        "The dogs runs to the",
        "The birds flies over the",
        "The children reads a",
        "The men walks to the",
        "The women sings a",
    ],
    # 复杂主谓一致: 介词短语插入
    "complex_singular_correct": [
        "The cat near the dogs sleeps on the",
        "The dog behind the trees runs to the",
        "The bird above the cats flies over the",
    ],
    "complex_singular_wrong": [
        "The cat near the dogs sleep on the",
        "The dog behind the trees run to the",
        "The bird above the cats fly over the",
    ],
}


# =====================================================================
# HELPER: FIND WORD POSITION
# =====================================================================

def find_word_position(tokenizer, template, word):
    """Find the token position of word in the FULL tokenized template."""
    full_tokens = tokenizer.encode(template, add_special_tokens=True)
    no_special_tokens = tokenizer.encode(template, add_special_tokens=False)
    n_prefix = len(full_tokens) - len(no_special_tokens)
    word_ids = tokenizer.encode(word, add_special_tokens=False)

    for i in range(len(no_special_tokens) - len(word_ids) + 1):
        if no_special_tokens[i:i+len(word_ids)] == word_ids:
            return i + n_prefix

    decoded = [tokenizer.decode([t]) for t in no_special_tokens]
    for i, d in enumerate(decoded):
        if word.lower() in d.lower() and i > 0:
            return i + n_prefix

    return 1 + n_prefix


# =====================================================================
# ★★★ CORE METHOD: CONSTRAINT SATISFACTION FUNCTION ★★★
# 坐标无关的约束衡量 — 基于 W_U 投影, 不是 h 的内积
# =====================================================================

def compute_constraint_satisfaction(h_vec, W_U, constraint_words, tokenizer):
    """
    ★★★ 约束满足函数 — 坐标不变的核心方法 ★★★
    
    定义: σ_C(h) = mean(logits of constraint_words)
    
    为什么坐标不变:
    - logit = W_U @ h + b
    - 如果 h' = R h (R正交), 则 logit' = W_U @ R h = (W_U R) @ h
    - 虽然 W_U 变了, 但 logit 值不变! (因为 R 只是旋转)
    - 所以 σ_C(h) 在基底变换下不变
    
    Args:
        h_vec: [d_model] hidden state vector
        W_U: [vocab_size, d_model] unembedding matrix
        constraint_words: list of words defining the constraint
        tokenizer: tokenizer
    
    Returns:
        float: constraint satisfaction score (average logit)
    """
    logits = W_U @ h_vec  # [vocab_size]
    
    scores = []
    for word in constraint_words:
        tok_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(tok_ids) == 1:
            scores.append(float(logits[tok_ids[0]]))
        elif len(tok_ids) > 1:
            # Multi-token: average logit
            scores.append(float(np.mean([logits[tid] for tid in tok_ids])))
    
    return float(np.mean(scores)) if scores else 0.0


def compute_all_constraints(h_vec, W_U, tokenizer, constraint_dict=None):
    """Compute satisfaction of all constraints for a hidden state."""
    if constraint_dict is None:
        constraint_dict = CONSTRAINTS
    
    result = {}
    for cname, cwords in constraint_dict.items():
        result[cname] = compute_constraint_satisfaction(h_vec, W_U, cwords, tokenizer)
    return result


# =====================================================================
# ★★★ GRASSMANN DISTANCE — 坐标无关的子空间比较 ★★★
# =====================================================================

def compute_grassmann_distance(subspace1, subspace2):
    """
    ★★★ Grassmann 距离 — 子空间间坐标无关的距离 ★★★
    
    定义: d_G(U, V) = ||P_U - P_V||_F / sqrt(2)
    其中 P_U = U U^T 是投影矩阵
    
    等价: d_G = sqrt(sum_i sin^2(θ_i))  (principal angles)
    
    为什么坐标不变:
    - 子空间由其方向定义, 不依赖具体基底选择
    - 旋转 R 不会改变子空间本身
    - 所以 d_G 在基底变换下不变
    
    Args:
        subspace1: [k1, d] 正交基 (每行是一个基向量)
        subspace2: [k2, d] 正交基
    
    Returns:
        float: Grassmann distance (0 = identical, large = orthogonal)
    """
    # Project to orthogonal bases via SVD
    U1, _, _ = np.linalg.svd(subspace1, full_matrices=False)
    U2, _, _ = np.linalg.svd(subspace2, full_matrices=False)
    
    # Principal angles via SVD of U1 @ U2^T
    M = U1 @ U2.T  # [k1, k2]
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    
    # S are cosines of principal angles
    # Clip to [0, 1] for numerical stability
    S = np.clip(S, 0, 1)
    
    # Grassmann distance = sqrt(sum of sin^2(θ_i))
    sin2 = 1.0 - S**2
    sin2 = np.maximum(sin2, 0)  # numerical safety
    distance = float(np.sqrt(np.sum(sin2)))
    
    # Also compute subspace overlap (how much they share)
    # overlap = mean(cos^2(θ_i)) = mean(S^2)
    overlap = float(np.mean(S**2)) if len(S) > 0 else 0.0
    
    return distance, overlap


# =====================================================================
# HELPER: GET HIDDEN STATES
# =====================================================================

def get_hidden_states_at_layers(model, tokenizer, template, target_layers, W_U=None):
    """Get hidden states at specified layers for the last token position."""
    input_device = next(model.parameters()).device
    inputs = tokenizer(template, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    
    hs = out.hidden_states
    result = {}
    for li in target_layers:
        if li < len(hs):
            result[li] = hs[li][0, -1].float().cpu().numpy()  # last token
    
    # Also return final logits
    logits = out.logits[0, -1].float().cpu().numpy()
    
    return result, logits


# =====================================================================
# Phase A: ★★★ 约束基发现 — 概念 = 约束交集 ★★★
# =====================================================================

def run_constraint_basis_discovery(model, tokenizer, device, model_info, W_U):
    """
    ★★★ Phase A: apple 由哪些最小约束组成? ★★★
    
    方法:
    1. 对每个概念词, 在所有层计算约束满足状态
    2. 找出哪些约束对 apple "选择性满足":
       σ_C(apple) >> σ_C(random_words)
    3. 验证: 这些约束的交集是否足以区分 apple?
    4. 关键测试: 删除某约束 → apple 是否失去该属性?
    
    ★★★ 这不是 PCA! 这是 W_U 空间中的约束分析! ★★★
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # Sample layers densely for dynamics
    sample_layers = list(range(0, n_layers, max(1, n_layers // 12)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    print("\n" + "="*70, flush=True)
    print("Phase A: ★★★ 约束基发现 — 概念 = 约束交集 ★★★", flush=True)
    print("="*70, flush=True)
    
    # Step 1: Collect constraint satisfaction profiles for all concept words
    print("  Step 1: Computing constraint satisfaction profiles...", flush=True)
    
    concept_profiles = {}  # {word: {layer: {constraint: score}}}
    
    for word in CONCEPT_CONSTRAINTS:
        template = f"The {word} is"
        hiddens, logits = get_hidden_states_at_layers(
            model, tokenizer, template, sample_layers, W_U)
        
        word_profile = {}
        for li, h_vec in hiddens.items():
            constraints = compute_all_constraints(h_vec, W_U, tokenizer)
            word_profile[li] = constraints
        
        concept_profiles[word] = word_profile
        print(f"    {word}: computed {len(word_profile)} layers", flush=True)
    
    # Step 2: Compute constraint selectivity
    # σ_C is selective for word W if: σ_C(W) >> σ_C(other words)
    print("\n  Step 2: Computing constraint selectivity...", flush=True)
    
    all_words = list(CONCEPT_CONSTRAINTS.keys())
    constraint_selectivity = {}  # {constraint: {word: selectivity}}
    
    for li in sample_layers:
        # For each constraint, compute how selective it is for each word
        for cname in CONSTRAINTS:
            # Get all word scores for this constraint at this layer
            word_scores = {}
            for word in all_words:
                if li in concept_profiles[word]:
                    word_scores[word] = concept_profiles[word][li][cname]
            
            if len(word_scores) < 2:
                continue
            
            # Selectivity = (target_score - mean_other_scores) / std_other_scores
            scores_array = np.array(list(word_scores.values()))
            mean_score = np.mean(scores_array)
            std_score = np.std(scores_array) + 1e-8
            
            for word, score in word_scores.items():
                selectivity = (score - mean_score) / std_score
                key = f"{cname}_{word}_L{li}"
                constraint_selectivity[key] = {
                    "constraint": cname,
                    "word": word,
                    "layer": li,
                    "score": round(float(score), 4),
                    "selectivity": round(float(selectivity), 4),
                }
    
    # Step 3: For each word, find its "constraint basis"
    # = the constraints that are most selective for this word
    print("\n  Step 3: Finding constraint basis for each concept...", flush=True)
    
    concept_basis = {}
    for word in CONCEPT_CONSTRAINTS:
        word_selectivities = {k: v for k, v in constraint_selectivity.items() 
                             if v["word"] == word and v["layer"] == n_layers - 1}
        
        # Sort by selectivity
        sorted_sel = sorted(word_selectivities.values(), 
                           key=lambda x: abs(x["selectivity"]), reverse=True)
        
        # Take top constraints (selectivity > 0.5)
        active_constraints = [v["constraint"] for v in sorted_sel 
                             if v["selectivity"] > 0.5]
        
        # Compare with predicted constraints
        predicted = CONCEPT_CONSTRAINTS[word]
        
        concept_basis[word] = {
            "predicted_constraints": predicted,
            "discovered_active": active_constraints[:10],
            "top_selectivities": [(v["constraint"], round(v["selectivity"], 3)) 
                                  for v in sorted_sel[:5]],
        }
        
        print(f"    {word}: predicted={predicted}, discovered={active_constraints[:5]}", flush=True)
    
    # Step 4: ★★★ Constraint ablation — 删除约束方向, 通过hook做因果干预 ★★★
    print("\n  Step 4: ★★★ Constraint ablation — causal intervention via hooks ★★★", flush=True)
    
    # For each concept, ablate each constraint direction via hook, measure logit change
    ablation_results = {}
    layers = get_layers(model)
    
    test_concepts = ["apple", "car", "cat"]  # Test 3 concepts
    
    for word in test_concepts:
        expected_constraints = CONCEPT_CONSTRAINTS.get(word, [])
        template = f"The {word} is"
        
        # Get normal logits
        input_device = next(model.parameters()).device
        inputs = tokenizer(template, return_tensors="pt", truncation=True, max_length=64)
        input_ids_dev = inputs["input_ids"].to(input_device)
        attn_mask_dev = inputs["attention_mask"].to(input_device)
        
        with torch.no_grad():
            normal_out = model(input_ids=input_ids_dev, attention_mask=attn_mask_dev)
        normal_logits = normal_out.logits[0, -1].float().cpu().numpy()
        
        # Compute normal constraint satisfaction from logits
        normal_cs = {}
        for cname, cwords in CONSTRAINTS.items():
            scores = []
            for w in cwords:
                tok_ids = tokenizer.encode(w, add_special_tokens=False)
                if len(tok_ids) == 1:
                    scores.append(float(normal_logits[tok_ids[0]]))
                elif len(tok_ids) > 1:
                    scores.append(float(np.mean([normal_logits[tid] for tid in tok_ids])))
            normal_cs[cname] = float(np.mean(scores)) if scores else 0.0
        
        for li in [n_layers // 2, n_layers - 1]:
            # For each constraint, ablate via hook and measure logit change
            for cname in expected_constraints:
                cwords = CONSTRAINTS[cname]
                c_tok_ids = []
                for w in cwords:
                    ids = tokenizer.encode(w, add_special_tokens=False)
                    c_tok_ids.extend(ids)
                
                if not c_tok_ids:
                    continue
                
                # Constraint direction = mean of W_U rows
                constraint_dir = np.mean(W_U[c_tok_ids], axis=0)  # [d_model]
                constraint_dir = constraint_dir / max(np.linalg.norm(constraint_dir), 1e-10)
                
                # Create subspace from this single direction
                subspace_t = torch.tensor(constraint_dir.reshape(1, -1), dtype=torch.float32)
                
                # Hook to ablate constraint direction at target layer
                def make_constraint_ablation_hook(target_layer, subspace):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                        else:
                            h = output
                        subspace_dev = subspace.to(h.device).to(h.dtype)  # [1, d_model]
                        # Project out: h_ablated = h - v(v^T h)
                        proj = torch.matmul(subspace_dev, h.transpose(-1, -2))  # [1, seq]
                        recon = torch.matmul(subspace_dev.T, proj)  # [d_model, seq]
                        h_ablated = h - recon.transpose(-1, -2)
                        if isinstance(output, tuple):
                            return (h_ablated,) + output[1:]
                        return h_ablated
                    return hook
                
                hooks = [layers[li].register_forward_hook(
                    make_constraint_ablation_hook(li, subspace_t))]
                
                with torch.no_grad():
                    ablated_out = model(input_ids=input_ids_dev, attention_mask=attn_mask_dev)
                
                for h in hooks:
                    h.remove()
                
                ablated_logits = ablated_out.logits[0, -1].float().cpu().numpy()
                
                # Compute ablated constraint satisfaction
                ablated_cs = {}
                for cname2, cwords2 in CONSTRAINTS.items():
                    scores = []
                    for w in cwords2:
                        tok_ids = tokenizer.encode(w, add_special_tokens=False)
                        if len(tok_ids) == 1:
                            scores.append(float(ablated_logits[tok_ids[0]]))
                        elif len(tok_ids) > 1:
                            scores.append(float(np.mean([ablated_logits[tid] for tid in tok_ids])))
                    ablated_cs[cname2] = float(np.mean(scores)) if scores else 0.0
                
                # Measure change
                target_change = ablated_cs[cname] - normal_cs[cname]
                other_changes = [ablated_cs[c] - normal_cs[c] for c in CONSTRAINTS if c != cname]
                avg_other_change = float(np.mean(other_changes)) if other_changes else 0
                
                if abs(target_change) > 1e-6:
                    selectivity = target_change / (abs(target_change) + abs(avg_other_change) + 1e-10)
                else:
                    selectivity = 0.0
                
                key = f"{word}_{cname}_L{li}"
                ablation_results[key] = {
                    "word": word,
                    "constraint": cname,
                    "layer": li,
                    "normal_cs": round(float(normal_cs[cname]), 4),
                    "ablated_cs": round(float(ablated_cs[cname]), 4),
                    "target_change": round(float(target_change), 4),
                    "avg_other_change": round(float(avg_other_change), 4),
                    "selectivity": round(float(selectivity), 4),
                }
                
                print(f"    Ablate '{cname}' from '{word}' at L{li}: Δ_target={target_change:.4f}, "
                      f"Δ_other={avg_other_change:.4f}, sel={selectivity:.4f}", flush=True)
    
    return {
        "concept_profiles": {w: {str(li): cs for li, cs in prof.items()} 
                            for w, prof in concept_profiles.items()},
        "constraint_selectivity": constraint_selectivity,
        "concept_basis": concept_basis,
        "constraint_ablation": ablation_results,
    }


# =====================================================================
# Phase B: ★★★ 约束激活动力学 — 逐层约束闭合追踪 ★★★
# =====================================================================

def run_constraint_activation_dynamics(model, tokenizer, device, model_info, W_U):
    """
    ★★★ Phase B: apple 生成时, 哪些约束先激活? 哪些后闭合? ★★★
    
    方法:
    1. 对每个概念词, 在每一层计算约束满足状态
    2. 追踪每个约束从 L0 到 L_last 的激活曲线
    3. 找出"约束闭合层": 某约束在哪一层达到稳定?
    4. 比较不同概念: 约束闭合是否有统一模式?
    
    ★★★ 关键问题: 是否存在"约束闭合的普遍动力学"? ★★★
    """
    n_layers = model_info.n_layers
    
    # Sample ALL layers for dynamics (need full resolution)
    all_layers = list(range(n_layers + 1))
    
    print("\n" + "="*70, flush=True)
    print("Phase B: ★★★ 约束激活动力学 — 逐层约束闭合追踪 ★★★", flush=True)
    print("="*70, flush=True)
    
    # Step 1: Full-resolution constraint profiles
    print("  Step 1: Computing full-resolution constraint profiles...", flush=True)
    
    dynamics = {}  # {word: {constraint: {layer: score}}}
    
    test_words = ["apple", "banana", "car", "bus", "cat", "dog", "happy", "sad", "tree", "rose"]
    
    for word in test_words:
        template = f"The {word} is"
        hiddens, logits = get_hidden_states_at_layers(
            model, tokenizer, template, all_layers, W_U)
        
        word_dynamics = defaultdict(dict)
        for li, h_vec in hiddens.items():
            cs = compute_all_constraints(h_vec, W_U, tokenizer)
            for cname, score in cs.items():
                word_dynamics[cname][li] = score
        
        dynamics[word] = dict(word_dynamics)
        print(f"    {word}: {len(hiddens)} layers, {len(word_dynamics)} constraints", flush=True)
    
    # Step 2: Find "constraint closure layer" for each constraint
    print("\n  Step 2: Finding constraint closure layers...", flush=True)
    
    closure_results = {}
    
    for word in test_words:
        word_closure = {}
        expected_constraints = CONCEPT_CONSTRAINTS.get(word, [])
        
        for cname in CONSTRAINTS:
            if cname not in dynamics[word]:
                continue
            
            layer_scores = dynamics[word][cname]
            if not layer_scores:
                continue
            
            # Sort by layer
            sorted_layers = sorted(layer_scores.keys())
            scores = [layer_scores[li] for li in sorted_layers]
            
            # Find closure layer: first layer where score is within 10% of final score
            final_score = scores[-1]
            if abs(final_score) < 1e-6:
                closure_layer = None
            else:
                closure_layer = sorted_layers[-1]  # default
                for i, s in enumerate(scores):
                    if abs(s - final_score) / (abs(final_score) + 1e-6) < 0.1:
                        closure_layer = sorted_layers[i]
                        break
            
            # Compute "activation slope": rate of change
            if len(scores) > 1:
                slopes = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
                max_slope_idx = np.argmax(np.abs(slopes))
                peak_change_layer = sorted_layers[max_slope_idx]
            else:
                peak_change_layer = sorted_layers[0] if sorted_layers else 0
            
            is_expected = cname in expected_constraints
            
            word_closure[cname] = {
                "final_score": round(float(final_score), 4),
                "closure_layer": closure_layer,
                "peak_change_layer": peak_change_layer,
                "is_expected_constraint": is_expected,
                "L0_score": round(float(scores[0]), 4) if scores else 0,
                "L_last_score": round(float(scores[-1]), 4) if scores else 0,
                "total_change": round(float(scores[-1] - scores[0]), 4) if scores else 0,
            }
        
        closure_results[word] = word_closure
    
    # Step 3: Analyze constraint activation order
    print("\n  Step 3: Constraint activation order analysis...", flush=True)
    
    activation_order = {}
    for word in test_words:
        expected = CONCEPT_CONSTRAINTS.get(word, [])
        word_closure = closure_results[word]
        
        # Sort expected constraints by closure layer
        expected_closures = [(cname, word_closure[cname]["closure_layer"])
                            for cname in expected if cname in word_closure]
        expected_closures.sort(key=lambda x: x[1] if x[1] is not None else 999)
        
        activation_order[word] = {
            "constraint_closure_order": expected_closures,
            "expected_constraints": expected,
        }
        
        if expected_closures:
            order_str = " → ".join([f"{c}(L{cl})" for c, cl in expected_closures])
            print(f"    {word}: {order_str}", flush=True)
    
    # Step 4: ★★★ Cross-concept constraint dynamics comparison ★★★
    print("\n  Step 4: Cross-concept constraint dynamics comparison...", flush=True)
    
    # For each constraint, compare its activation profile across words
    cross_concept = {}
    for cname in CONSTRAINTS:
        word_profiles = {}
        for word in test_words:
            if cname in dynamics[word]:
                layer_scores = dynamics[word][cname]
                # Summarize: L0, L_mid, L_last
                L0 = layer_scores.get(0, 0)
                L_mid = layer_scores.get(n_layers // 2, 0)
                L_last = layer_scores.get(n_layers - 1, 0)
                word_profiles[word] = {
                    "L0": round(float(L0), 4),
                    "L_mid": round(float(L_mid), 4),
                    "L_last": round(float(L_last), 4),
                    "change": round(float(L_last - L0), 4),
                }
        
        cross_concept[cname] = word_profiles
    
    return {
        "dynamics": {w: {c: {str(li): round(s, 4) for li, s in ls.items()} 
                        for c, ls in wd.items()} 
                    for w, wd in dynamics.items()},
        "closure_results": closure_results,
        "activation_order": activation_order,
        "cross_concept_comparison": cross_concept,
    }


# =====================================================================
# Phase C: ★★★ 约束输运拓扑 — 主谓一致约束传播 ★★★
# =====================================================================

def run_constraint_transport(model, tokenizer, device, model_info, W_U):
    """
    ★★★ Phase C: 约束如何跨 token 传播? ★★★
    
    方法:
    1. 对主谓一致句子, 在每一层追踪:
       - 主语 token 的 "number 约束" (singular/plural) 满足状态
       - 动词 token 的 "number 约束" 满足状态
    2. 关键问题: number 约束在哪个 token 激活? 在哪一层传播到动词?
    3. 比较: 正确一致 vs 错误一致 → 模型是否"检测到"约束违反?
    
    ★★★ 不是 attention weight! 而是约束满足状态的传播! ★★★
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # Use key layers for efficiency
    sample_layers = list(range(0, n_layers, max(1, n_layers // 9)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    print("\n" + "="*70, flush=True)
    print("Phase C: ★★★ 约束输运拓扑 — 主谓一致约束传播 ★★★", flush=True)
    print("="*70, flush=True)
    
    # Number constraint verification words
    singular_words = ["is", "was", "runs", "walks", "sleeps", "has", "one"]
    plural_words = ["are", "were", "run", "walk", "sleep", "have", "many"]
    
    number_constraint = {
        "singular": singular_words,
        "plural": plural_words,
    }
    
    # Step 1: For each sentence type, get hidden states at subject and verb positions
    print("  Step 1: Collecting subject/verb hidden states...", flush=True)
    
    transport_results = {}
    
    for sent_type, sentences in AGREEMENT_SENTENCES.items():
        print(f"\n  Sentence type: {sent_type}", flush=True)
        
        sent_cs_subject = {li: [] for li in sample_layers}  # constraint at subject
        sent_cs_verb = {li: [] for li in sample_layers}      # constraint at verb
        sent_cs_last = {li: [] for li in sample_layers}       # constraint at last token
        
        for sent in sentences:
            # Tokenize
            tokens = tokenizer.encode(sent, add_special_tokens=False)
            token_strs = [tokenizer.decode([t]) for t in tokens]
            
            # Find subject position (typically "The X" where X is position 1)
            # and verb position
            subject_pos = 1  # After "The"
            verb_pos = 2     # Typically the word after subject
            
            # Get hidden states at all layers
            input_device = next(model.parameters()).device
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
            
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(input_device),
                           attention_mask=inputs["attention_mask"].to(input_device),
                           output_hidden_states=True)
            
            hs = out.hidden_states
            
            for li in sample_layers:
                if li >= len(hs):
                    continue
                h = hs[li][0].float().cpu().numpy()  # [seq_len, d_model]
                
                # Get constraint satisfaction at subject, verb, and last token
                n_tokens = h.shape[0]
                
                # Subject token (position after "The")
                sub_pos = min(subject_pos, n_tokens - 1)
                h_sub = h[sub_pos]
                
                # Verb token (position 2)
                vb_pos = min(verb_pos, n_tokens - 1)
                h_verb = h[vb_pos]
                
                # Last token
                h_last = h[-1]
                
                # Compute number constraint at each position
                singular_score_sub = compute_constraint_satisfaction(h_sub, W_U, singular_words, tokenizer)
                plural_score_sub = compute_constraint_satisfaction(h_sub, W_U, plural_words, tokenizer)
                
                singular_score_verb = compute_constraint_satisfaction(h_verb, W_U, singular_words, tokenizer)
                plural_score_verb = compute_constraint_satisfaction(h_verb, W_U, plural_words, tokenizer)
                
                singular_score_last = compute_constraint_satisfaction(h_last, W_U, singular_words, tokenizer)
                plural_score_last = compute_constraint_satisfaction(h_last, W_U, plural_words, tokenizer)
                
                sent_cs_subject[li].append({
                    "singular": round(float(singular_score_sub), 4),
                    "plural": round(float(plural_score_sub), 4),
                    "diff": round(float(singular_score_sub - plural_score_sub), 4),
                })
                sent_cs_verb[li].append({
                    "singular": round(float(singular_score_verb), 4),
                    "plural": round(float(plural_score_verb), 4),
                    "diff": round(float(singular_score_verb - plural_score_verb), 4),
                })
                sent_cs_last[li].append({
                    "singular": round(float(singular_score_last), 4),
                    "plural": round(float(plural_score_last), 4),
                    "diff": round(float(singular_score_last - plural_score_last), 4),
                })
        
        # Average across sentences of this type
        avg_cs_subject = {}
        avg_cs_verb = {}
        avg_cs_last = {}
        
        for li in sample_layers:
            if sent_cs_subject[li]:
                avg_cs_subject[li] = {
                    "singular": round(float(np.mean([s["singular"] for s in sent_cs_subject[li]])), 4),
                    "plural": round(float(np.mean([s["plural"] for s in sent_cs_subject[li]])), 4),
                    "diff": round(float(np.mean([s["diff"] for s in sent_cs_subject[li]])), 4),
                }
            if sent_cs_verb[li]:
                avg_cs_verb[li] = {
                    "singular": round(float(np.mean([s["singular"] for s in sent_cs_verb[li]])), 4),
                    "plural": round(float(np.mean([s["plural"] for s in sent_cs_verb[li]])), 4),
                    "diff": round(float(np.mean([s["diff"] for s in sent_cs_verb[li]])), 4),
                }
            if sent_cs_last[li]:
                avg_cs_last[li] = {
                    "singular": round(float(np.mean([s["singular"] for s in sent_cs_last[li]])), 4),
                    "plural": round(float(np.mean([s["plural"] for s in sent_cs_last[li]])), 4),
                    "diff": round(float(np.mean([s["diff"] for s in sent_cs_last[li]])), 4),
                }
        
        transport_results[sent_type] = {
            "subject_number": avg_cs_subject,
            "verb_number": avg_cs_verb,
            "last_token_number": avg_cs_last,
        }
        
        # Print key results
        for li in [0, n_layers // 2, n_layers - 1]:
            if li in avg_cs_subject:
                sub_diff = avg_cs_subject[li]["diff"]
                verb_diff = avg_cs_verb[li]["diff"] if li in avg_cs_verb else 0
                last_diff = avg_cs_last[li]["diff"] if li in avg_cs_last else 0
                print(f"    L{li}: sub_diff={sub_diff:.4f}, verb_diff={verb_diff:.4f}, "
                      f"last_diff={last_diff:.4f}", flush=True)
    
    # Step 2: ★★★ Key comparison: correct vs wrong agreement ★★★
    print("\n  Step 2: ★★★ Correct vs Wrong Agreement — Constraint Violation Detection ★★★", flush=True)
    
    violation_detection = {}
    
    pairs = [
        ("singular_correct", "singular_wrong"),
        ("plural_correct", "plural_wrong"),
        ("complex_singular_correct", "complex_singular_wrong"),
    ]
    
    for correct_type, wrong_type in pairs:
        if correct_type not in transport_results or wrong_type not in transport_results:
            continue
        
        correct_data = transport_results[correct_type]
        wrong_data = transport_results[wrong_type]
        
        # Compare constraint satisfaction at last token across layers
        for li in sample_layers:
            if li in correct_data["last_token_number"] and li in wrong_data["last_token_number"]:
                correct_diff = correct_data["last_token_number"][li]["diff"]
                wrong_diff = wrong_data["last_token_number"][li]["diff"]
                
                # If model detects violation: correct should have stronger singular/plural signal
                violation_signal = abs(correct_diff) - abs(wrong_diff)
                
                key = f"{correct_type}_vs_{wrong_type}_L{li}"
                violation_detection[key] = {
                    "correct_last_diff": correct_diff,
                    "wrong_last_diff": wrong_diff,
                    "violation_signal": round(float(violation_signal), 4),
                }
        
        # Print summary
        for li in [0, n_layers // 2, n_layers - 1]:
            key = f"{correct_type}_vs_{wrong_type}_L{li}"
            if key in violation_detection:
                vd = violation_detection[key]
                print(f"    {correct_type} vs {wrong_type} at L{li}: "
                      f"violation_signal={vd['violation_signal']:.4f}", flush=True)
    
    return {
        "transport_results": transport_results,
        "violation_detection": violation_detection,
    }


# =====================================================================
# Phase D: ★★★ 跨语言约束同构 — 用母语句子! ★★★
# =====================================================================

def run_cross_lingual_constraint_isomorphism(model, tokenizer, device, model_info, W_U):
    """
    ★★★ Phase D: 不同语言的概念是否收敛到同一约束流形? ★★★
    
    关键改进 (vs Phase 175):
    1. 用各语言的母语句子, 不是英文模板!
    2. 用 Grassmann 距离比较子空间, 不是 cosine!
    3. ★★★ 决定性实验: 删除 EN 约束子空间 → 是否同时影响 ZH/JA/FR? ★★★
    
    ★★★ 这是整个 Phase 176 最重要的实验 ★★★
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    
    print("\n" + "="*70, flush=True)
    print("Phase D: ★★★ 跨语言约束同构 — 母语句子 + Grassmann 距离 ★★★", flush=True)
    print("="*70, flush=True)
    
    # Step 1: For each concept, get hidden states in each language's native sentences
    print("  Step 1: Collecting cross-lingual hidden states (native sentences)...", flush=True)
    
    cross_lingual_data = {}  # {concept: {lang: {layer: [vectors]}}}
    
    for concept_key, lang_sentences in CROSS_LINGUAL_NATIVE.items():
        print(f"\n  Concept: '{concept_key}'", flush=True)
        
        concept_data = {}
        
        for lang, sentences in lang_sentences.items():
            # Check if tokenizer can handle this language
            # Try encoding the first sentence
            try:
                test_ids = tokenizer.encode(sentences[0], add_special_tokens=False)
                if len(test_ids) == 0:
                    print(f"    {lang}: cannot tokenize, skipping", flush=True)
                    continue
            except Exception as e:
                print(f"    {lang}: tokenization error: {e}", flush=True)
                continue
            
            lang_vectors = {li: [] for li in key_layers}
            
            for sent in sentences:
                input_device = next(model.parameters()).device
                inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
                
                with torch.no_grad():
                    out = model(input_ids=inputs["input_ids"].to(input_device),
                               attention_mask=inputs["attention_mask"].to(input_device),
                               output_hidden_states=True)
                
                for li in key_layers:
                    if li < len(out.hidden_states):
                        h = out.hidden_states[li][0, -1].float().cpu().numpy()  # last token
                        lang_vectors[li].append(h)
            
            concept_data[lang] = lang_vectors
            
            n_vecs = sum(len(v) for v in lang_vectors.values())
            print(f"    {lang}: collected {n_vecs} vectors across layers", flush=True)
        
        cross_lingual_data[concept_key] = concept_data
    
    # Step 2: Compute constraint satisfaction in each language
    print("\n  Step 2: Computing constraint satisfaction per language...", flush=True)
    
    constraint_profiles = {}  # {concept: {lang: {layer: {constraint: score}}}}
    
    for concept_key, concept_data in cross_lingual_data.items():
        concept_profiles = {}
        expected_constraints = CONCEPT_CONSTRAINTS.get(concept_key, [])
        
        for lang, lang_vectors in concept_data.items():
            lang_profiles = {}
            
            for li in key_layers:
                if not lang_vectors[li]:
                    continue
                
                # Average constraint satisfaction across sentences
                avg_constraints = defaultdict(float)
                n_sents = len(lang_vectors[li])
                
                for h_vec in lang_vectors[li]:
                    cs = compute_all_constraints(h_vec, W_U, tokenizer)
                    for cname, score in cs.items():
                        avg_constraints[cname] += score / n_sents
                
                lang_profiles[li] = dict(avg_constraints)
            
            concept_profiles[lang] = lang_profiles
        
        constraint_profiles[concept_key] = concept_profiles
    
    # Step 3: ★★★ Grassmann distance between cross-lingual constraint subspaces ★★★
    print("\n  Step 3: ★★★ Grassmann distance — coordinate-free comparison ★★★", flush=True)
    
    grassmann_results = {}  # {concept: {layer: {lang_pair: (distance, overlap)}}}
    
    for concept_key, concept_data in cross_lingual_data.items():
        concept_grassmann = {}
        lang_names = list(concept_data.keys())
        
        for li in key_layers:
            # Build constraint subspaces for each language
            # Use constraint directions in W_U space as basis (more stable than PCA of 3 vectors)
            lang_subspaces = {}
            
            for lang in lang_names:
                vecs = concept_data[lang].get(li, [])
                if len(vecs) < 1:
                    continue
                
                X = np.array(vecs)  # [n_sents, d_model]
                centroid = np.mean(X, axis=0)
                X_centered = X - centroid
                
                # PCA: get top-k subspace
                k = min(len(vecs) - 1, X_centered.shape[1])
                if k < 1:
                    # Even with 1 vector, we can compute its direction
                    k = 1
                
                if X_centered.shape[0] >= k:
                    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
                    subspace = Vt[:k]  # [k, d_model]
                else:
                    # Not enough data for SVD, use the centered vectors directly
                    subspace = X_centered[:1] / max(np.linalg.norm(X_centered[0]), 1e-10)
                    subspace = subspace.reshape(1, -1)
                
                lang_subspaces[lang] = subspace
            
            # Compute pairwise Grassmann distances
            layer_results = {}
            lang_list = list(lang_subspaces.keys())
            
            for i in range(len(lang_list)):
                for j in range(i+1, len(lang_list)):
                    la, lb = lang_list[i], lang_list[j]
                    distance, overlap = compute_grassmann_distance(
                        lang_subspaces[la], lang_subspaces[lb])
                    
                    layer_results[f"{la}-{lb}"] = {
                        "grassmann_distance": round(float(distance), 4),
                        "subspace_overlap": round(float(overlap), 4),
                    }
            
            if layer_results:
                concept_grassmann[f"L{li}"] = layer_results
        
        grassmann_results[concept_key] = concept_grassmann
        
        # Print summary
        for li_key, data in concept_grassmann.items():
            for pair, vals in data.items():
                if "en-" in pair:
                    print(f"    {concept_key} {li_key} {pair}: "
                          f"d_G={vals['grassmann_distance']:.4f}, "
                          f"overlap={vals['subspace_overlap']:.4f}", flush=True)
    
    # Step 4: ★★★★★★ THE DECISIVE EXPERIMENT: Cross-lingual causal intervention ★★★★★★
    # Delete EN constraint subspace → does it affect ZH/JA/FR?
    print("\n  Step 4: ★★★★★★ DECISIVE: Cross-lingual causal intervention ★★★★★★", flush=True)
    print("  Delete EN constraint subspace → measure impact on ALL languages", flush=True)
    
    causal_intervention = {}
    layers = get_layers(model)
    
    for concept_key in ["apple", "cat"]:  # Test 2 concepts
        if concept_key not in cross_lingual_data:
            continue
        
        concept_data = cross_lingual_data[concept_key]
        if "en" not in concept_data:
            continue
        
        # Get EN constraint subspace at key layers
        for li in [n_layers // 2, n_layers - 1]:
            en_vecs = concept_data["en"].get(li, [])
            if len(en_vecs) < 2:
                continue
            
            X_en = np.array(en_vecs)
            centroid_en = np.mean(X_en, axis=0)
            X_centered = X_en - centroid_en
            
            k = min(5, X_centered.shape[0] - 1)
            if k < 1:
                continue
            
            _, _, Vt_en = np.linalg.svd(X_centered, full_matrices=False)
            en_subspace = Vt_en[:k]  # [k, d_model]
            
            # Also compute constraint-specific direction
            expected_constraints = CONCEPT_CONSTRAINTS.get(concept_key, [])
            constraint_dirs = []
            for cname in expected_constraints:
                cwords = CONSTRAINTS[cname]
                c_tok_ids = []
                for w in cwords:
                    ids = tokenizer.encode(w, add_special_tokens=False)
                    c_tok_ids.extend(ids)
                if c_tok_ids:
                    c_dir = np.mean(W_U[c_tok_ids], axis=0)
                    c_norm = np.linalg.norm(c_dir)
                    if c_norm > 1e-10:
                        constraint_dirs.append(c_dir / c_norm)
            
            if not constraint_dirs:
                continue
            
            # Combine: EN subspace + constraint directions
            combined_basis = np.vstack([en_subspace, np.array(constraint_dirs)[:3]])
            # Re-orthogonalize
            Q, _ = np.linalg.qr(combined_basis.T)
            combined_subspace = Q.T[:k + min(3, len(constraint_dirs))]
            
            # For each language, test: does deleting EN subspace affect their constraint satisfaction?
            for lang in concept_data:
                lang_sents = CROSS_LINGUAL_NATIVE[concept_key].get(lang, [])
                if not lang_sents:
                    continue
                
                for sent in lang_sents[:2]:  # Test 2 sentences per language
                    try:
                        input_device = next(model.parameters()).device
                        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
                        input_ids_dev = inputs["input_ids"].to(input_device)
                        attn_mask_dev = inputs["attention_mask"].to(input_device)
                        
                        # Normal forward — get logits (the final output is what matters)
                        with torch.no_grad():
                            normal_out = model(input_ids=input_ids_dev, attention_mask=attn_mask_dev)
                        normal_logits = normal_out.logits[0, -1].float().cpu().numpy()
                        
                        # Compute normal constraint satisfaction from LOGITS directly
                        # σ_C = mean(logits of constraint words) — coordinate-free!
                        normal_cs = {}
                        for cname, cwords in CONSTRAINTS.items():
                            scores = []
                            for w in cwords:
                                tok_ids = tokenizer.encode(w, add_special_tokens=False)
                                if len(tok_ids) == 1:
                                    scores.append(float(normal_logits[tok_ids[0]]))
                                elif len(tok_ids) > 1:
                                    scores.append(float(np.mean([normal_logits[tid] for tid in tok_ids])))
                            normal_cs[cname] = float(np.mean(scores)) if scores else 0.0
                        
                        # Forward with EN subspace ablated — measure logit change
                        subspace_t = torch.tensor(combined_subspace, dtype=torch.float32)
                        
                        def make_ablation_hook(target_layer, subspace):
                            def hook(module, input, output):
                                if isinstance(output, tuple):
                                    h = output[0]
                                else:
                                    h = output
                                subspace_dev = subspace.to(h.device).to(h.dtype)
                                proj = torch.matmul(subspace_dev, h.transpose(-1, -2))
                                recon = torch.matmul(subspace_dev.T, proj)
                                h_ablated = h - recon.transpose(-1, -2)
                                if isinstance(output, tuple):
                                    return (h_ablated,) + output[1:]
                                return h_ablated
                            return hook
                        
                        hooks = [layers[li].register_forward_hook(
                            make_ablation_hook(li, subspace_t))]
                        
                        with torch.no_grad():
                            ablated_out = model(input_ids=input_ids_dev, attention_mask=attn_mask_dev)
                        
                        for h in hooks:
                            h.remove()
                        
                        ablated_logits = ablated_out.logits[0, -1].float().cpu().numpy()
                        
                        # Compute ablated constraint satisfaction from LOGITS
                        ablated_cs = {}
                        for cname, cwords in CONSTRAINTS.items():
                            scores = []
                            for w in cwords:
                                tok_ids = tokenizer.encode(w, add_special_tokens=False)
                                if len(tok_ids) == 1:
                                    scores.append(float(ablated_logits[tok_ids[0]]))
                                elif len(tok_ids) > 1:
                                    scores.append(float(np.mean([ablated_logits[tid] for tid in tok_ids])))
                            ablated_cs[cname] = float(np.mean(scores)) if scores else 0.0
                        
                        # Measure change in expected constraints
                        constraint_changes = {}
                        for cname in expected_constraints:
                            if cname in normal_cs and cname in ablated_cs:
                                change = ablated_cs[cname] - normal_cs[cname]
                                constraint_changes[cname] = round(float(change), 4)
                        
                        # Total constraint change
                        all_changes = {c: round(float(ablated_cs[c] - normal_cs[c]), 4) 
                                      for c in CONSTRAINTS if c in normal_cs and c in ablated_cs}
                        
                        # Also compute logit change for the concept word itself
                        concept_tok_ids = tokenizer.encode(
                            CROSS_LINGUAL_NATIVE[concept_key].get("en", [concept_key])[0] 
                            if concept_key in CROSS_LINGUAL_NATIVE else concept_key,
                            add_special_tokens=False)
                        concept_logit_change = 0.0
                        if concept_tok_ids:
                            concept_logit_change = float(
                                np.mean([ablated_logits[tid] - normal_logits[tid] for tid in concept_tok_ids]))
                        
                        key = f"{concept_key}_{lang}_L{li}"
                        causal_intervention[key] = {
                            "concept": concept_key,
                            "language": lang,
                            "layer": li,
                            "sentence": sent[:50],
                            "expected_constraint_changes": constraint_changes,
                            "total_constraint_change": round(
                                float(np.mean([abs(v) for v in all_changes.values()])), 4),
                            "expected_avg_change": round(
                                float(np.mean([abs(v) for v in constraint_changes.values()])), 4
                            ) if constraint_changes else 0,
                            "unexpected_avg_change": round(
                                float(np.mean([abs(v) for c, v in all_changes.items() 
                                              if c not in expected_constraints])), 4
                            ) if any(c not in expected_constraints for c in all_changes) else 0,
                            "concept_word_logit_change": round(float(concept_logit_change), 4),
                        }
                        
                    except Exception as e:
                        print(f"    [WARN] Failed for {concept_key} {lang} L{li}: {e}", flush=True)
                        continue
        
        # Print key results
        print(f"\n  ★ Cross-lingual causal intervention for '{concept_key}':", flush=True)
        for key, data in causal_intervention.items():
            if data["concept"] != concept_key:
                continue
            if data["language"] == "en":
                continue  # Skip EN→EN (it should definitely be affected)
            print(f"    {data['language']} L{data['layer']}: "
                  f"expected_Δ={data['expected_avg_change']:.4f}, "
                  f"unexpected_Δ={data['unexpected_avg_change']:.4f}", flush=True)
    
    return {
        "constraint_profiles": constraint_profiles,
        "grassmann_distances": grassmann_results,
        "cross_lingual_causal": causal_intervention,
    }


# =====================================================================
# MAIN
# =====================================================================

def run_phase176(model_name):
    print(f"\n{'='*70}", flush=True)
    print(f"Phase 176: ★★★ 约束代数 — 从 Hidden State 到约束不变量 ★★★", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"{'='*70}", flush=True)

    t_start = time.time()

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    print(f"  Model: {model_info.model_class}, L={n_layers}, d={d_model}", flush=True)

    # Load W_U — the constraint function's key component
    print("  Loading W_U (unembedding matrix)...", flush=True)
    W_U = get_W_U(model, model_name)
    print(f"  W_U shape: {W_U.shape}", flush=True)

    # =====================================================================
    # Run all experiments
    # =====================================================================

    # Phase A: Constraint Basis Discovery
    exp_a = run_constraint_basis_discovery(model, tokenizer, device, model_info, W_U)

    # Phase B: Constraint Activation Dynamics
    exp_b = run_constraint_activation_dynamics(model, tokenizer, device, model_info, W_U)

    # Phase C: Constraint Transport Topology
    exp_c = run_constraint_transport(model, tokenizer, device, model_info, W_U)

    # Phase D: Cross-Lingual Constraint Isomorphism
    exp_d = run_cross_lingual_constraint_isomorphism(model, tokenizer, device, model_info, W_U)

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "timestamp": timestamp,
        "phase_A_constraint_basis": exp_a,
        "phase_B_activation_dynamics": exp_b,
        "phase_C_constraint_transport": exp_c,
        "phase_D_cross_lingual_isomorphism": exp_d,
    }

    out_path = f"tests/glm5_temp/phase176_{model_name}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    # Release model
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\nPhase 176 ({model_name}) completed in {elapsed:.1f}s", flush=True)

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase176_constraint_algebra.py <model_name>")
        print("  model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    run_phase176(model_name)
