"""
Phase 461: 参数级编码起源 — 基底/差分/纤维束的权重行结构分解
==========================================================
从Phase 460的"残差流编码成分分解"推进到"参数级编码起源"。

核心问题:
1. 类别共享码和私有特征码在MLP权重(W_down行)中有什么结构?
2. 同类别不同对象的差分方向有什么共同模式?
3. 翻译命令如何编码到残差流?
4. 跨语言中间层语义不变量的直接探测(不依赖logit读出)
5. 大beta合成能否突破Phase 460的微弱效果?

Exp1: W_down行级贡献分解 — 哪些行贡献Shared, 哪些贡献Private
Exp2: 跨对象差分结构对比 — 同类别对象的差分方向共性
Exp3: 翻译命令编码 — "translate to English"如何修改残差流
Exp4: 跨语言中间层探针 — 线性探针直接验证语义不变量
Exp5: 大beta合成测试 — beta=20/50/100的因果效果

用法: python tests/glm5/phase461_param_level_encoding.py qwen3 1
      python tests/glm5/phase461_param_level_encoding.py glm4 2
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json, math
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS,
                          load_model as _load_model_utils)

def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 数据定义 ====================
CAT_OBJ = {
    "fruit":   ["apple", "banana", "orange", "grape", "pear", "peach", "lemon", "mango"],
    "animal":  ["dog", "cat", "horse", "lion", "bear", "rabbit", "cow", "tiger"],
    "tool":    ["hammer", "knife", "wrench", "saw", "drill", "axe", "shovel", "scissors"],
    "vehicle": ["car", "bus", "bicycle", "truck", "train", "boat", "plane", "scooter"],
}

# 候选族定义
FAM = {
    "class_fruit":   ["fruit", "produce", "crop", "harvest"],
    "class_animal":  ["animal", "creature", "beast", "mammal"],
    "class_tool":    ["tool", "implement", "instrument", "device"],
    "class_vehicle": ["vehicle", "transport", "conveyance", "automobile"],
}

# 属性族
ATTR_FAM = {
    "attr_color":      ["red", "green", "yellow", "blue", "brown", "black", "white", "orange"],
    "attr_part_bio":   ["seed", "leaf", "stem", "root", "skin", "bone", "leg", "wing"],
    "attr_part_mech":  ["wheel", "blade", "handle", "engine", "gear", "axle", "lever", "spring"],
}

CAT_FAM = {
    "fruit":  {"target": "class_fruit",  "compete": ["class_animal", "class_tool", "class_vehicle"]},
    "animal": {"target": "class_animal", "compete": ["class_fruit", "class_tool", "class_vehicle"]},
    "tool":   {"target": "class_tool",   "compete": ["class_fruit", "class_animal", "class_vehicle"]},
    "vehicle":{"target": "class_vehicle","compete": ["class_fruit", "class_animal", "class_tool"]},
}

# 模板
TEMPLATES = {
    "is_a":     "The {obj} is a kind of",
    "has_color":"The color of a {obj} is",
    "has_part": "A common part of a {obj} is",
}

# 翻译模板(Exp3)
TRANSLATE_TEMPLATES = [
    # 英文→中文
    ("en2zh", "Translate the following to Chinese: The {obj} is a fruit.", "苹果"),
    ("en2zh", "Translate the following to Chinese: The {obj} is an animal.", "狗"),
    ("en2zh", "Translate the following to Chinese: The {obj} is a tool.", "锤子"),
    # 中文→英文
    ("zh2en", "请将以下翻译为英文: {obj}是一种水果", "fruit"),
    ("zh2en", "请将以下翻译为英文: {obj}是一种动物", "animal"),
    ("zh2en", "请将以下翻译为英文: {obj}是一种工具", "tool"),
    # 纯英文对照(不翻译)
    ("en_ref", "The {obj} is a kind of", None),
    # 纯中文对照
    ("zh_ref", "{obj}是一种", None),
]

# 轮次数据量
ROUNDS = {
    1: {k: v[:4] for k, v in CAT_OBJ.items()},   # pilot: 4/类
    2: {k: v[:6] for k, v in CAT_OBJ.items()},   # main: 6/类
}


def load_model_bf16(model_name):
    """BF16加载模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog(f"Loading {model_name} (bfloat16 + device_map=auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager",
    )
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    plog(f"{model_name} loaded: GPU={gpu_mem:.2f}GB, class={type(model).__name__}")
    return model, tokenizer, device


def get_residual_at_layers(model, tokenizer, prompt, target_layers, device):
    """提取指定层的残差流"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    captured = {}
    layers = get_layers(model)
    
    def make_hook(li):
        def hook(module, input, output):
            # 残差流 = input[0] (layer的输入就是残差流)
            if isinstance(input, tuple) and len(input) > 0:
                captured[li] = input[0].detach().float().cpu()
        return hook
    
    hooks = [layers[li].register_forward_hook(make_hook(li)) for li in target_layers]
    
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    
    for h in hooks:
        h.remove()
    
    # 返回每个层最后一个token的残差流
    results = {}
    for li in target_layers:
        if li in captured:
            # 取最后一个非padding位置
            seq_len = attention_mask.sum().item()
            results[li] = captured[li][0, seq_len - 1].numpy()
    return results


def get_mlp_weights(model, model_name, layer_idx):
    """获取MLP权重矩阵 — 兼容meta device"""
    layers = get_layers(model)
    layer = layers[layer_idx]
    info = get_model_info(model, model_name)
    
    def _to_numpy(tensor):
        """安全地将tensor转为numpy, 处理meta device"""
        if tensor.is_meta:
            # meta tensor: 需要从原始文件加载, 先跳过
            return None
        return tensor.detach().cpu().float().numpy()
    
    if info.mlp_type == "split_gate_up":
        W_up = _to_numpy(layer.mlp.up_proj.weight)
        W_down = _to_numpy(layer.mlp.down_proj.weight)
        W_gate = _to_numpy(layer.mlp.gate_proj.weight) if hasattr(layer.mlp, 'gate_proj') else None
    else:  # merged_gate_up
        W_gate_up_tensor = layer.mlp.gate_up_proj.weight
        if W_gate_up_tensor.is_meta:
            W_gate = None
            W_up = None
        else:
            W_gate_up = W_gate_up_tensor.detach().cpu().float().numpy()
            d_inter = W_gate_up.shape[0] // 2
            W_gate = W_gate_up[:d_inter]
            W_up = W_gate_up[d_inter:]
        
        W_down_tensor = layer.mlp.down_proj.weight
        if W_down_tensor.is_meta:
            W_down = None
        else:
            W_down = W_down_tensor.detach().cpu().float().numpy()
    
    return W_up, W_down, W_gate


def get_family_logits(W_U, resid, fam_words, tokenizer):
    """获取候选族logits"""
    vocab = tokenizer.get_vocab()
    logits = resid @ W_U.T
    fam_logits = {}
    for fam_name, words in fam_words.items():
        fam_vals = []
        for w in words:
            w_stripped = w.strip()
            if w_stripped in vocab:
                fam_vals.append(float(logits[vocab[w_stripped]]))
            elif f" {w_stripped}" in vocab:
                fam_vals.append(float(logits[vocab[f" {w_stripped}"]]))
        fam_logits[fam_name] = fam_vals
    return fam_logits


def compute_family_margin(fam_logits, target_fam, compete_fams):
    """计算候选族边际"""
    target_mean = np.mean(fam_logits[target_fam]) if target_fam in fam_logits else 0
    compete_means = [np.mean(fam_logits[f]) for f in compete_fams if f in fam_logits]
    compete_mean = np.mean(compete_means) if compete_means else 0
    return target_mean - compete_mean


# ==================== Exp1: W_down行级贡献分解 ====================
def exp1_wdown_row_contribution(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    分析W_down哪些行对Shared/Private方向贡献最大
    
    方法:
    1. 提取同类别多个对象的残差流
    2. 计算Shared方向 = 类别均值方向, Private方向 = 对象-类别均值
    3. 对每个W_down行, 计算其与Shared/Private方向的点积
    4. 找到对Shared和Private分别贡献最大的行(神经元)
    """
    plog("Exp1: W_down行级贡献分解")
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    
    # 选择关键层
    key_layers = [0, info.n_layers//6, info.n_layers//3, info.n_layers//2,
                  2*info.n_layers//3, 5*info.n_layers//6, info.n_layers-2]
    key_layers = [l for l in key_layers if l < info.n_layers]
    
    results = {}
    
    for cat_name, obj_list in obj_dict.items():
        plog(f"  Category: {cat_name}, objects: {obj_list}")
        # 提取所有对象的is_a模板残差流
        obj_resids = {}
        for obj in obj_list:
            prompt = f"The {obj} is a kind of"
            resids = get_residual_at_layers(model, tokenizer, prompt, key_layers, device)
            obj_resids[obj] = resids
        
        for li in key_layers:
            # 收集该层所有对象的残差流
            vecs = np.array([obj_resids[obj][li] for obj in obj_list if li in obj_resids[obj]])
            if len(vecs) < 2:
                continue
            
            # 类别中心(Shared方向)
            class_center = vecs.mean(axis=0)
            class_center_norm = class_center / (np.linalg.norm(class_center) + 1e-10)
            
            # 每个对象的Private方向
            private_vecs = vecs - class_center
            
            # Shared方向的平均范数和Private方向的平均范数
            shared_norm = np.linalg.norm(class_center)
            private_norms = [np.linalg.norm(pv) for pv in private_vecs]
            avg_private_norm = np.mean(private_norms)
            
            # W_down行级贡献分析
            _, W_down, _ = get_mlp_weights(model, model_name, li)
            if W_down is None:
                # meta device上的层, 跳过权重分析
                layer_key = f"L{li}"
                if cat_name not in results:
                    results[cat_name] = {}
                results[cat_name][layer_key] = {"error": "meta_device_weight"}
                continue
            # W_down: [d_model, d_inter] in PyTorch convention
            # Forward: output = W_down @ intermediate, output=[d_model], intermediate=[d_inter]
            # W_down.T: [d_inter, d_model] maps from residual space to neuron space
            # proj = W_down.T @ direction tells us which neurons contribute to that direction
            d_model_w, d_inter = W_down.shape
            
            # 投影: 哪些中间神经元对Shared方向贡献最大
            # W_down.T @ class_center_norm = [d_inter, d_model] @ [d_model] = [d_inter]
            shared_proj = W_down.T @ class_center_norm  # [d_inter]
            
            # 对每个Private方向
            private_projs = []
            for pv in private_vecs:
                pv_norm = pv / (np.linalg.norm(pv) + 1e-10)
                pp = W_down.T @ pv_norm  # [d_inter]
                private_projs.append(pp)
            avg_private_proj = np.mean(private_projs, axis=0)  # [d_inter]
            
            # Top-k Shared/Private贡献神经元(在d_inter维度上的索引)
            k = min(20, d_inter)
            top_shared_idx = np.argsort(np.abs(shared_proj))[-k:]
            top_private_idx = np.argsort(np.abs(avg_private_proj))[-k:]
            
            # 重叠度
            overlap = len(set(top_shared_idx) & set(top_private_idx))
            
            # Shared和Private神经元的权重范数分布
            # W_down: [d_model, d_inter], 神经元i对应W_down[:, i]
            shared_neuron_norms = np.linalg.norm(W_down[:, top_shared_idx], axis=0)
            private_neuron_norms = np.linalg.norm(W_down[:, top_private_idx], axis=0)
            
            # 相关系数: Shared投影 vs Private投影在所有神经元上的相关性
            if len(shared_proj) > 1 and np.std(shared_proj) > 1e-10 and np.std(avg_private_proj) > 1e-10:
                corr = float(np.corrcoef(shared_proj, avg_private_proj)[0, 1])
            else:
                corr = 0.0
            
            layer_key = f"L{li}"
            if cat_name not in results:
                results[cat_name] = {}
            results[cat_name][layer_key] = {
                "shared_norm": float(shared_norm),
                "avg_private_norm": float(avg_private_norm),
                "shared_private_ratio": float(shared_norm / (avg_private_norm + 1e-10)),
                "top_shared_neuron_ids": [int(x) for x in top_shared_idx[-5:]],
                "top_private_neuron_ids": [int(x) for x in top_private_idx[-5:]],
                "top_shared_proj_vals": [float(shared_proj[i]) for i in top_shared_idx[-5:]],
                "top_private_proj_vals": [float(avg_private_proj[i]) for i in top_private_idx[-5:]],
                "overlap_top_k": int(overlap),
                "shared_neuron_weight_norms_mean": float(np.mean(shared_neuron_norms)),
                "private_neuron_weight_norms_mean": float(np.mean(private_neuron_norms)),
                "shared_private_corr": float(corr),
            }
            
            # Per-object Private方向与Shared方向的角度
            angles = []
            for pv in private_vecs:
                if np.linalg.norm(pv) > 1e-10:
                    cos_angle = np.dot(pv, class_center) / (np.linalg.norm(pv) * np.linalg.norm(class_center) + 1e-10)
                    angles.append(float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))))
            results[cat_name][layer_key]["private_shared_angles"] = angles
            
        plog(f"  {cat_name} done for {len(key_layers)} layers")
    
    return results


# ==================== Exp2: 跨对象差分结构对比 ====================
def exp2_cross_object_differential(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    比较同类别不同对象的差分方向, 寻找共同模式
    
    方法:
    1. 对同类别每个对象, 计算其残差流与类别中心的差(Private方向)
    2. 计算所有Private方向之间的余弦相似度矩阵
    3. 分析Private方向是否聚类(某些对象的差分更相似)
    4. 跨类别对比: 水果的Private结构 vs 动物的Private结构
    5. 差分方向的SVD: 提取共享的差分子空间
    """
    plog("Exp2: 跨对象差分结构对比")
    info = get_model_info(model, model_name)
    
    key_layers = [info.n_layers//4, info.n_layers//2, 3*info.n_layers//4, info.n_layers-2]
    key_layers = [l for l in key_layers if l < info.n_layers]
    
    results = {}
    
    for cat_name, obj_list in obj_dict.items():
        plog(f"  Category: {cat_name}")
        obj_resids = {}
        for obj in obj_list:
            prompt = f"The {obj} is a kind of"
            resids = get_residual_at_layers(model, tokenizer, prompt, key_layers, device)
            obj_resids[obj] = resids
        
        for li in key_layers:
            vecs = {}
            for obj in obj_list:
                if li in obj_resids[obj]:
                    vecs[obj] = obj_resids[obj][li]
            
            if len(vecs) < 3:
                continue
            
            obj_names = list(vecs.keys())
            mat = np.array([vecs[o] for o in obj_names])
            
            # 类别中心
            center = mat.mean(axis=0)
            
            # Private方向矩阵
            priv_mat = mat - center  # [n_obj, d_model]
            
            # Private方向间的余弦相似度矩阵
            norms = np.linalg.norm(priv_mat, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            priv_normed = priv_mat / norms
            cos_sim = priv_normed @ priv_normed.T  # [n_obj, n_obj]
            
            # SVD of Private方向
            # priv_mat: [n_obj, d_model], 但n_obj << d_model
            # 取转置做SVD: priv_mat.T @ priv_mat = [d_model, d_model]
            # 但更高效: SVD of priv_mat直接
            U, S, Vt = np.linalg.svd(priv_mat, full_matrices=False)
            # U: [n_obj, n_obj], S: [min(n_obj, d_model)], Vt: [n_obj, d_model]
            
            # 有效秩: 累积方差解释比
            var_explained = S**2 / (S**2).sum()
            cum_var = np.cumsum(var_explained)
            eff_rank = int(np.searchsorted(cum_var, 0.9) + 1)  # 解释90%方差的维度数
            
            # 第一差分主成分方向
            if len(S) > 0:
                pc1 = Vt[0]  # [d_model]
                pc1_norm = np.linalg.norm(pc1)
                # 各对象在PC1上的投影
                pc1_projs = priv_mat @ pc1 / (pc1_norm**2 + 1e-10)
            else:
                pc1 = np.zeros_like(center)
                pc1_projs = []
            
            # 对象在Private空间中的距离矩阵
            dist_mat = np.zeros((len(obj_names), len(obj_names)))
            for i in range(len(obj_names)):
                for j in range(len(obj_names)):
                    dist_mat[i, j] = np.linalg.norm(priv_mat[i] - priv_mat[j])
            
            layer_key = f"L{li}"
            if cat_name not in results:
                results[cat_name] = {}
            results[cat_name][layer_key] = {
                "private_cosine_matrix": {obj_names[i]: {obj_names[j]: float(cos_sim[i,j]) 
                                                         for j in range(len(obj_names))} 
                                          for i in range(len(obj_names))},
                "singular_values": [float(s) for s in S[:min(8, len(S))]],
                "variance_explained": [float(v) for v in var_explained[:min(8, len(var_explained))]],
                "effective_rank_90pct": int(eff_rank),
                "pc1_projections": {obj_names[i]: float(pc1_projs[i]) for i in range(len(obj_names))},
                "avg_private_cosine_offdiag": float(np.mean(cos_sim[np.triu_indices(len(obj_names), k=1)])),
                "private_norms": {obj_names[i]: float(norms[i, 0]) for i in range(len(obj_names))},
            }
    
    # 跨类别对比
    plog("  Cross-category comparison")
    cross_cat = {}
    all_resids = {}
    for cat_name, obj_list in obj_dict.items():
        for obj in obj_list:
            prompt = f"The {obj} is a kind of"
            resids = get_residual_at_layers(model, tokenizer, prompt, key_layers, device)
            all_resids[f"{cat_name}/{obj}"] = resids
    
    for li in key_layers:
        # 每个类别的中心
        cat_centers = {}
        for cat_name, obj_list in obj_dict.items():
            cat_vecs = [all_resids[f"{cat_name}/{obj}"][li] for obj in obj_list 
                       if f"{cat_name}/{obj}" in all_resids and li in all_resids[f"{cat_name}/{obj}"]]
            if cat_vecs:
                cat_centers[cat_name] = np.mean(cat_vecs, axis=0)
        
        # 跨类别中心的余弦相似度
        cat_names_list = list(cat_centers.keys())
        cross_cos = {}
        for i, c1 in enumerate(cat_names_list):
            for j, c2 in enumerate(cat_names_list):
                if i < j:
                    cos = np.dot(cat_centers[c1], cat_centers[c2]) / (
                        np.linalg.norm(cat_centers[c1]) * np.linalg.norm(cat_centers[c2]) + 1e-10)
                    cross_cos[f"{c1}_vs_{c2}"] = float(cos)
        
        # 跨类别的Private方向是否可区分
        # 取每类前2个对象的Private方向, 计算跨类余弦
        cross_priv_cos = {}
        for c1 in cat_names_list:
            for c2 in cat_names_list:
                if c1 < c2:  # 字母序
                    objs1 = obj_dict[c1][:2]
                    objs2 = obj_dict[c2][:2]
                    priv1 = []
                    priv2 = []
                    for o in objs1:
                        key = f"{c1}/{o}"
                        if key in all_resids and li in all_resids[key]:
                            priv1.append(all_resids[key][li] - cat_centers[c1])
                    for o in objs2:
                        key = f"{c2}/{o}"
                        if key in all_resids and li in all_resids[key]:
                            priv2.append(all_resids[key][li] - cat_centers[c2])
                    if priv1 and priv2:
                        # 计算平均跨类Private余弦
                        cos_vals = []
                        for p1 in priv1:
                            for p2 in priv2:
                                n1, n2 = np.linalg.norm(p1), np.linalg.norm(p2)
                                if n1 > 1e-10 and n2 > 1e-10:
                                    cos_vals.append(float(np.dot(p1, p2) / (n1 * n2)))
                        cross_priv_cos[f"{c1}_vs_{c2}"] = float(np.mean(cos_vals)) if cos_vals else 0
        
        cross_cat[f"L{li}"] = {
            "shared_cosine": cross_cos,
            "private_cross_cosine": cross_priv_cos,
        }
    
    results["_cross_category"] = cross_cat
    return results


# ==================== Exp3: 翻译命令编码 ====================
def exp3_translate_encoding(model, tokenizer, model_name, device, round_num):
    """
    分析翻译命令如何编码到残差流
    
    对比:
    1. 纯英文: "The apple is a kind of" → is_a方向
    2. 翻译命令: "Translate to Chinese: The apple is a kind of" → 命令方向
    3. 差分 = 翻译方向 - is_a方向 = 纯命令编码
    
    以及中文→英文的翻译命令
    """
    plog("Exp3: 翻译命令编码")
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    
    key_layers = list(range(0, info.n_layers, max(1, info.n_layers // 8)))
    if info.n_layers - 1 not in key_layers:
        key_layers.append(info.n_layers - 1)
    
    results = {}
    
    # 测试词
    test_words = ["apple", "dog", "hammer"]
    
    for word in test_words:
        plog(f"  Word: {word}")
        word_results = {}
        
        # 基准: 纯英文is_a
        en_prompt = f"The {word} is a kind of"
        en_resids = get_residual_at_layers(model, tokenizer, en_prompt, key_layers, device)
        
        # 翻译命令: en→zh
        trans_en2zh = f"Translate the following to Chinese: The {word} is a fruit."
        trans_en2zh_resids = get_residual_at_layers(model, tokenizer, trans_en2zh, key_layers, device)
        
        # 翻译命令: zh→en
        trans_zh2en = f"请将以下翻译为英文: {word}是一种水果"
        trans_zh2en_resids = get_residual_at_layers(model, tokenizer, trans_zh2en, key_layers, device)
        
        # 纯中文参照
        zh_prompt = f"{word}是一种"
        zh_resids = get_residual_at_layers(model, tokenizer, zh_prompt, key_layers, device)
        
        for li in key_layers:
            layer_key = f"L{li}"
            layer_data = {}
            
            # 差分向量
            if li in en_resids and li in trans_en2zh_resids:
                diff_en2zh = trans_en2zh_resids[li] - en_resids[li]
                diff_en2zh_norm = np.linalg.norm(diff_en2zh)
                
                # 差分方向与英文is_a方向的角度
                en_norm = np.linalg.norm(en_resids[li])
                if diff_en2zh_norm > 1e-10 and en_norm > 1e-10:
                    cos_en2zh = np.dot(diff_en2zh, en_resids[li]) / (diff_en2zh_norm * en_norm)
                else:
                    cos_en2zh = 0
                
                layer_data["en2zh_diff_norm"] = float(diff_en2zh_norm)
                layer_data["en2zh_diff_cos_with_en"] = float(cos_en2zh)
            else:
                layer_data["en2zh_diff_norm"] = None
                layer_data["en2zh_diff_cos_with_en"] = None
            
            if li in zh_resids and li in trans_zh2en_resids:
                diff_zh2en = trans_zh2en_resids[li] - zh_resids[li]
                diff_zh2en_norm = np.linalg.norm(diff_zh2en)
                
                zh_norm = np.linalg.norm(zh_resids[li])
                if diff_zh2en_norm > 1e-10 and zh_norm > 1e-10:
                    cos_zh2en = np.dot(diff_zh2en, zh_resids[li]) / (diff_zh2en_norm * zh_norm)
                else:
                    cos_zh2en = 0
                
                layer_data["zh2en_diff_norm"] = float(diff_zh2en_norm)
                layer_data["zh2en_diff_cos_with_zh"] = float(cos_zh2en)
            else:
                layer_data["zh2en_diff_norm"] = None
                layer_data["zh2en_diff_cos_with_zh"] = None
            
            # en→zh差分 vs zh→en差分 的余弦
            if (li in en_resids and li in trans_en2zh_resids and 
                li in zh_resids and li in trans_zh2en_resids):
                diff1 = trans_en2zh_resids[li] - en_resids[li]
                diff2 = trans_zh2en_resids[li] - zh_resids[li]
                n1, n2 = np.linalg.norm(diff1), np.linalg.norm(diff2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_12 = np.dot(diff1, diff2) / (n1 * n2)
                else:
                    cos_12 = 0
                layer_data["en2zh_vs_zh2en_diff_cos"] = float(cos_12)
            
            # 残差流范数
            for name, resids_dict in [("en", en_resids), ("trans_en2zh", trans_en2zh_resids),
                                       ("zh", zh_resids), ("trans_zh2en", trans_zh2en_resids)]:
                if li in resids_dict:
                    layer_data[f"{name}_norm"] = float(np.linalg.norm(resids_dict[li]))
            
            # 候选族logits
            for name, resids_dict in [("en", en_resids), ("trans_en2zh", trans_en2zh_resids),
                                       ("zh", zh_resids), ("trans_zh2en", trans_zh2en_resids)]:
                if li in resids_dict:
                    fam_logits = get_family_logits(W_U, resids_dict[li], FAM, tokenizer)
                    for fam_name, vals in fam_logits.items():
                        layer_data[f"{name}_{fam_name}_mean"] = float(np.mean(vals)) if vals else 0
            
            word_results[layer_key] = layer_data
        
        results[word] = word_results
    
    return results


# ==================== Exp4: 跨语言中间层探针 ====================
def exp4_cross_language_probe(model, tokenizer, model_name, device, round_num):
    """
    用线性探针直接验证中间层语义不变量
    
    Phase 460发现中文prompt的logit为负, 但中间层余弦0.85。
    这里我们:
    1. 提取中英文prompt的中间层残差流
    2. 用英文残差流训练线性探针(区分4个类别)
    3. 在中文残差流上测试探针
    4. 如果探针跨语言泛化, 证明语义不变量存在
    """
    plog("Exp4: 跨语言中间层探针")
    info = get_model_info(model, model_name)
    
    key_layers = [info.n_layers//4, info.n_layers//3, info.n_layers//2, 
                  2*info.n_layers//3, 3*info.n_layers//4, info.n_layers-2]
    key_layers = [l for l in key_layers if l < info.n_layers]
    
    results = {}
    
    # 训练数据: 每个类别4个对象
    train_cats = ["fruit", "animal", "tool", "vehicle"]
    train_objs = {c: CAT_OBJ[c][:4] for c in train_cats}
    
    # 提取英文残差流
    plog("  Extracting English residuals...")
    en_data = {}  # {layer: [(vec, cat_idx), ...]}
    for cat_idx, cat in enumerate(train_cats):
        for obj in train_objs[cat]:
            prompt = f"The {obj} is a kind of"
            resids = get_residual_at_layers(model, tokenizer, prompt, key_layers, device)
            for li in key_layers:
                if li not in en_data:
                    en_data[li] = []
                if li in resids:
                    en_data[li].append((resids[li], cat_idx))
    
    # 提取中文残差流
    plog("  Extracting Chinese residuals...")
    zh_data = {}
    zh_objs = {"fruit": ["苹果", "香蕉", "橙子", "葡萄"],
               "animal": ["狗", "猫", "马", "狮子"],
               "tool": ["锤子", "刀", "扳手", "锯"],
               "vehicle": ["汽车", "公交车", "自行车", "卡车"]}
    
    for cat_idx, cat in enumerate(train_cats):
        for obj in zh_objs[cat]:
            prompt = f"{obj}是一种"
            resids = get_residual_at_layers(model, tokenizer, prompt, key_layers, device)
            for li in key_layers:
                if li not in zh_data:
                    zh_data[li] = []
                if li in resids:
                    zh_data[li].append((resids[li], cat_idx))
    
    # 训练和测试
    plog("  Training probes and testing...")
    for li in key_layers:
        if li not in en_data or li not in zh_data:
            continue
        
        en_vecs = np.array([d[0] for d in en_data[li]])
        en_labels = np.array([d[1] for d in en_data[li]])
        zh_vecs = np.array([d[0] for d in zh_data[li]])
        zh_labels = np.array([d[1] for d in zh_data[li]])
        
        if len(en_vecs) < 4 or len(zh_vecs) < 4:
            continue
        
        # 训练: 用英文数据训练one-vs-rest线性分类器(4个类别)
        # 简单方案: 用类别中心方向作为分类器
        cat_centers = []
        for cat_idx in range(4):
            mask = en_labels == cat_idx
            if mask.sum() > 0:
                cat_centers.append(en_vecs[mask].mean(axis=0))
            else:
                cat_centers.append(np.zeros(en_vecs.shape[1]))
        
        # 英文测试: 最近中心分类
        en_preds = []
        for vec in en_vecs:
            dists = [np.linalg.norm(vec - c) for c in cat_centers]
            en_preds.append(np.argmin(dists))
        en_acc = np.mean([p == l for p, l in zip(en_preds, en_labels)])
        
        # 中文测试: 同样用英文训练的中心分类
        zh_preds = []
        for vec in zh_vecs:
            dists = [np.linalg.norm(vec - c) for c in cat_centers]
            zh_preds.append(np.argmin(dists))
        zh_acc = np.mean([p == l for p, l in zip(zh_preds, zh_labels)])
        
        # 随机基线
        random_acc = 0.25
        
        # 中英文残差流的余弦相似度
        cos_vals = []
        for i in range(min(len(en_vecs), len(zh_vecs))):
            n1, n2 = np.linalg.norm(en_vecs[i]), np.linalg.norm(zh_vecs[i])
            if n1 > 1e-10 and n2 > 1e-10:
                cos_vals.append(float(np.dot(en_vecs[i], zh_vecs[i]) / (n1 * n2)))
        avg_cos = np.mean(cos_vals) if cos_vals else 0
        
        # 中英文类别中心的余弦
        zh_cat_centers = []
        for cat_idx in range(4):
            mask = zh_labels == cat_idx
            if mask.sum() > 0:
                zh_cat_centers.append(zh_vecs[mask].mean(axis=0))
            else:
                zh_cat_centers.append(np.zeros(zh_vecs.shape[1]))
        
        center_cos = []
        for i in range(4):
            n1 = np.linalg.norm(cat_centers[i])
            n2 = np.linalg.norm(zh_cat_centers[i])
            if n1 > 1e-10 and n2 > 1e-10:
                center_cos.append(float(np.dot(cat_centers[i], zh_cat_centers[i]) / (n1 * n2)))
        
        results[f"L{li}"] = {
            "en_probe_acc": float(en_acc),
            "zh_probe_acc_cross_lang": float(zh_acc),
            "random_baseline": float(random_acc),
            "avg_cosine_en_zh": float(avg_cos),
            "category_center_cosine": {train_cats[i]: center_cos[i] for i in range(len(center_cos))},
            "n_en_samples": len(en_vecs),
            "n_zh_samples": len(zh_vecs),
        }
    
    return results


# ==================== Exp5: 大beta合成测试 ====================
def exp5_large_beta_synthesis(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    用更大的beta测试Shared/Private方向注入的因果效果
    
    Phase 460用beta=5/10, 效果微弱(0.01-0.11)。
    这里测试beta=20/50/100。
    """
    plog("Exp5: 大beta合成测试")
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    
    # 关键层: Phase 460发现L12和L35最选择性
    key_layers = [info.n_layers//3, info.n_layers//2, 2*info.n_layers//3, info.n_layers-2]
    key_layers = [l for l in key_layers if l < info.n_layers]
    
    betas = [5, 10, 20, 50]
    
    results = {}
    
    # 测试2个对象: apple(fruit→tool), knife(tool→fruit)
    test_cases = [
        ("fruit", "apple", "class_fruit", "class_tool"),
        ("tool", "knife", "class_tool", "class_fruit"),
    ]
    
    for cat, obj, target_fam, comp_fam in test_cases:
        plog(f"  {obj} ({cat}): injecting {comp_fam} direction")
        obj_results = {}
        
        # 基准: 原始prompt
        base_prompt = f"The {obj} is a kind of"
        base_resids = get_residual_at_layers(model, tokenizer, base_prompt, key_layers, device)
        
        # 源类别: 目标类别的多个对象
        src_objs = [o for o in obj_dict[comp_fam.replace("class_", "")][:4]]
        
        # 目标类别中心
        tgt_cat = target_fam.replace("class_", "")
        tgt_objs = obj_dict[tgt_cat][:4]
        
        # 提取源类别和目标类别的中心
        src_resids_list = []
        tgt_resids_list = {}
        for li in key_layers:
            tgt_resids_list[li] = []
        
        for src_obj in src_objs:
            src_prompt = f"The {src_obj} is a kind of"
            src_resids = get_residual_at_layers(model, tokenizer, src_prompt, key_layers, device)
            src_resids_list.append(src_resids)
        
        for tgt_obj in tgt_objs:
            tgt_prompt = f"The {tgt_obj} is a kind of"
            tgt_resids = get_residual_at_layers(model, tokenizer, tgt_prompt, key_layers, device)
            for li in key_layers:
                if li in tgt_resids:
                    tgt_resids_list[li].append(tgt_resids[li])
        
        # 类别差异方向
        for li in key_layers:
            # 源类别中心
            src_vecs = [s[li] for s in src_resids_list if li in s]
            tgt_vecs = tgt_resids_list[li]
            
            if not src_vecs or not tgt_vecs:
                continue
            
            src_center = np.mean(src_vecs, axis=0)
            tgt_center = np.mean(tgt_vecs, axis=0)
            
            # 类别差异方向
            class_diff = src_center - tgt_center  # 从目标类别指向源类别
            class_diff_norm = np.linalg.norm(class_diff)
            
            if class_diff_norm < 1e-10:
                continue
            
            class_diff_dir = class_diff / class_diff_norm
            
            layer_data = {}
            
            # 基准logits
            if li in base_resids:
                base_fam_logits = get_family_logits(W_U, base_resids[li], FAM, tokenizer)
                base_target = np.mean(base_fam_logits[target_fam]) if target_fam in base_fam_logits else 0
                base_comp = np.mean(base_fam_logits[comp_fam]) if comp_fam in base_fam_logits else 0
                base_margin = base_target - base_comp
            else:
                base_margin = 0
                base_target = 0
                base_comp = 0
            
            layer_data["base_target_logit"] = float(base_target)
            layer_data["base_comp_logit"] = float(base_comp)
            layer_data["base_margin"] = float(base_margin)
            layer_data["class_diff_norm"] = float(class_diff_norm)
            
            # 注入不同beta
            for beta in betas:
                # 在base_resid上注入class_diff方向
                if li not in base_resids:
                    continue
                
                patched_resid = base_resids[li] + beta * class_diff_dir
                
                # 计算patched logits
                patched_fam_logits = get_family_logits(W_U, patched_resid, FAM, tokenizer)
                patched_target = np.mean(patched_fam_logits[target_fam]) if target_fam in patched_fam_logits else 0
                patched_comp = np.mean(patched_fam_logits[comp_fam]) if comp_fam in patched_fam_logits else 0
                patched_margin = patched_target - patched_comp
                
                delta_target = patched_target - base_target
                delta_comp = patched_comp - base_comp
                
                layer_data[f"beta{beta}_target_logit"] = float(patched_target)
                layer_data[f"beta{beta}_comp_logit"] = float(patched_comp)
                layer_data[f"beta{beta}_margin"] = float(patched_margin)
                layer_data[f"beta{beta}_delta_target"] = float(delta_target)
                layer_data[f"beta{beta}_delta_comp"] = float(delta_comp)
                layer_data[f"beta{beta}_selectivity"] = float(delta_target - delta_comp)
            
            obj_results[f"L{li}"] = layer_data
        
        results[f"{obj}_{cat}2{comp_fam.replace('class_', '')}"] = obj_results
    
    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    obj_dict = ROUNDS[round_num]
    
    plog(f"Phase 461: 参数级编码起源 — {model_name} R{round_num}")
    plog(f"Objects: {sum(len(v) for v in obj_dict.values())} total")
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    plog(f"Model: {info.model_class}, L={info.n_layers}, d={info.d_model}")
    
    all_results = {
        "model": model_name,
        "round": round_num,
        "model_info": {"class": info.model_class, "n_layers": info.n_layers, "d_model": info.d_model},
    }
    
    # Exp1
    plog("="*50)
    t0 = time.time()
    try:
        all_results["exp1_wdown_row"] = exp1_wdown_row_contribution(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"Exp1 ERROR: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_wdown_row"] = {"error": str(e)}
    plog(f"Exp1 done in {time.time()-t0:.1f}s")
    
    # Exp2
    plog("="*50)
    t0 = time.time()
    try:
        all_results["exp2_cross_object_diff"] = exp2_cross_object_differential(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"Exp2 ERROR: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_cross_object_diff"] = {"error": str(e)}
    plog(f"Exp2 done in {time.time()-t0:.1f}s")
    
    # Exp3
    plog("="*50)
    t0 = time.time()
    try:
        all_results["exp3_translate"] = exp3_translate_encoding(
            model, tokenizer, model_name, device, round_num)
    except Exception as e:
        plog(f"Exp3 ERROR: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_translate"] = {"error": str(e)}
    plog(f"Exp3 done in {time.time()-t0:.1f}s")
    
    # Exp4
    plog("="*50)
    t0 = time.time()
    try:
        all_results["exp4_cross_lang_probe"] = exp4_cross_language_probe(
            model, tokenizer, model_name, device, round_num)
    except Exception as e:
        plog(f"Exp4 ERROR: {e}")
        import traceback; traceback.print_exc()
        all_results["exp4_cross_lang_probe"] = {"error": str(e)}
    plog(f"Exp4 done in {time.time()-t0:.1f}s")
    
    # Exp5
    plog("="*50)
    t0 = time.time()
    try:
        all_results["exp5_large_beta"] = exp5_large_beta_synthesis(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"Exp5 ERROR: {e}")
        import traceback; traceback.print_exc()
        all_results["exp5_large_beta"] = {"error": str(e)}
    plog(f"Exp5 done in {time.time()-t0:.1f}s")
    
    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase461_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    plog(f"Results saved to {out_path}")
    
    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    plog("Model released. Phase 461 complete.")


if __name__ == "__main__":
    main()
