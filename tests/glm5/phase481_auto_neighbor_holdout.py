"""
Phase 481: 自动邻居选择、留出验证与DS7B修复
=============================================

核心目标:
1. 自动邻居选择 — 基于raw direction cosine自动选择最近邻，消除人工指定偏置
2. 留出对象验证 — 用一半对象构造specific方向，另一半对象验证自对齐
3. DS7B多层扫描 — 扫描多个层位+注入强度校准，修复fruit/clothing解耦失败

实验:
1. Exp1: 自动邻居选择 (ALL models) ★★★核心★★★
   - 计算所有类别raw direction两两余弦相似度
   - 自动选择余弦最高(最近邻)的2个类别作为正交化目标
   - 比较自动邻居vs人工邻居的selectivity
   
2. Exp2: 留出对象验证 (Qwen3) ★★★核心★★★
   - 每个类别8个对象分为train(4)+test(4)
   - train对象构造specific方向
   - test对象验证self-rank

3. Exp3: DS7B多层扫描+校准 (DS7B) ★★★修复★★★
   - 扫描L18-L26所有层
   - 校准注入强度: 按raw方向范数比缩放specific向量
   - 找到fruit和clothing的最佳解耦层

用法:
  python tests/glm5/phase481_auto_neighbor_holdout.py qwen3 1
  python tests/glm5/phase481_auto_neighbor_holdout.py glm4 1
  python tests/glm5/phase481_auto_neighbor_holdout.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json, math
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS)


def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 数据定义 ====================
CATEGORIES = {
    "fruit":     ["apple", "banana", "orange", "grape", "pear", "peach", "mango", "plum"],
    "animal":    ["dog", "cat", "horse", "lion", "bear", "rabbit", "eagle", "fish"],
    "tool":      ["hammer", "knife", "wrench", "saw", "drill", "axe", "chisel", "pliers"],
    "vehicle":   ["car", "bus", "bicycle", "truck", "train", "boat", "plane", "motorcycle"],
    "clothing":  ["shirt", "dress", "hat", "coat", "sock", "glove", "jacket", "scarf"],
    "furniture": ["chair", "table", "desk", "sofa", "bed", "shelf", "cabinet", "stool"],
    "food":      ["bread", "rice", "cheese", "pasta", "soup", "steak", "salad", "cake"],
    "plant":     ["tree", "flower", "grass", "bush", "fern", "cactus", "vine", "shrub"],
}

# 留出验证: train对象和test对象
CATEGORIES_TRAIN = {
    "fruit":     ["apple", "banana", "orange", "grape"],
    "animal":    ["dog", "cat", "horse", "lion"],
    "tool":      ["hammer", "knife", "wrench", "saw"],
    "vehicle":   ["car", "bus", "bicycle", "truck"],
    "clothing":  ["shirt", "dress", "hat", "coat"],
    "furniture": ["chair", "table", "desk", "sofa"],
    "food":      ["bread", "rice", "cheese", "pasta"],
    "plant":     ["tree", "flower", "grass", "bush"],
}

CATEGORIES_TEST = {
    "fruit":     ["pear", "peach", "mango", "plum"],
    "animal":    ["bear", "rabbit", "eagle", "fish"],
    "tool":      ["drill", "axe", "chisel", "pliers"],
    "vehicle":   ["train", "boat", "plane", "motorcycle"],
    "clothing":  ["sock", "glove", "jacket", "scarf"],
    "furniture": ["bed", "shelf", "cabinet", "stool"],
    "food":      ["soup", "steak", "salad", "cake"],
    "plant":     ["fern", "cactus", "vine", "shrub"],
}

# Phase 480的人工邻居(作为对照)
MANUAL_NEIGHBORS = {
    "fruit":     ["plant", "food"],
    "animal":    ["food", "clothing"],
    "tool":      ["furniture", "vehicle"],
    "vehicle":   ["tool", "furniture"],
    "clothing":  ["furniture", "tool"],
    "furniture": ["tool", "clothing"],
    "food":      ["plant", "fruit"],
    "plant":     ["food", "fruit"],
}

FAMILY_WORDS_8D = {
    "fruit":     ["fruit", "produce", "crop", "berry"],
    "animal":    ["animal", "creature", "beast", "pet"],
    "tool":      ["tool", "implement", "device", "instrument"],
    "vehicle":   ["vehicle", "transport", "automobile", "car"],
    "clothing":  ["clothing", "attire", "wear", "garment"],
    "furniture": ["furniture", "furnishing", "fixture", "seat"],
    "food":      ["food", "meal", "dish", "snack"],
    "plant":     ["plant", "tree", "vegetation", "flora"],
}

DCF_DIM_NAMES = ["fruit", "animal", "tool", "vehicle", "clothing", "furniture", "food", "plant"]

ATTRIBUTE_WORDS = {
    "edible":       ["edible", "eatable", "food", "meal", "dish", "snack", "eat", "cooked", "taste", "flavor"],
    "plant_grown":  ["plant", "tree", "grown", "vegetation", "flora", "garden", "cultivated", "harvest", "crop", "farm"],
    "seed_bearing": ["seed", "pit", "core", "kernel", "nut", "grain", "bean"],
    "sweet":        ["sweet", "sugar", "honey", "dessert", "candy", "ripe", "juicy", "delicious"],
    "natural":      ["natural", "organic", "wild", "raw", "fresh", "alive", "living", "biological"],
    "objectness":   ["object", "thing", "item", "entity", "substance", "material"],
    "movable":      ["movable", "portable", "lightweight", "carry", "transport", "handheld"],
    "human_made":   ["manufactured", "artificial", "synthetic", "built", "constructed", "engineered"],
    "tool_use":     ["tool", "instrument", "device", "implement", "equipment", "apparatus"],
    "indoor":       ["indoor", "inside", "room", "house", "home", "building", "furniture"],
    "living_being": ["alive", "living", "organism", "animal", "creature", "breathes", "moves"],
    "solid":        ["solid", "hard", "rigid", "firm", "sturdy", "material"],
    "juicy":        ["juicy", "wet", "succulent", "moist", "refreshing", "liquid"],
    "dessert_like": ["dessert", "cake", "pastry", "treat", "pudding", "confection"],
    "metallic":     ["metal", "steel", "iron", "aluminum", "copper", "mechanical"],
    "mechanical":   ["engine", "motor", "machine", "mechanical", "powered", "wheel"],
    "fabric":       ["fabric", "textile", "cloth", "cotton", "wool", "silk", "worn"],
    "seat_like":    ["seat", "sit", "rest", "support", "comfort"],
    "locomotion":   ["move", "travel", "drive", "fly", "ride", "transport"],
    "has_legs":     ["legs", "feet", "walk", "run", "stand", "limb"],
}

ATTR_DIM_NAMES = list(ATTRIBUTE_WORDS.keys())

RELATION_TEMPLATES = {
    "kind_of": "The {obj} is a kind of",
    "belongs_to": "A {obj} belongs to the category",
    "eaten_as": "{obj} is usually eaten as",
    "grown_from": "{obj} is grown from a",
    "found_in": "{obj} is usually found in",
    "used_for": "{obj} is usually used for",
}


# ==================== 模型加载 ====================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog(f"Loading {model_name} (bfloat16 + device_map=auto + flash)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        plog(f"  Flash Attention 2 enabled")
    except Exception as e:
        plog(f"  Flash Attention 2 failed ({e}), falling back to eager")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="eager",
        )
    model.eval()
    layers_list = get_layers(model)
    plog(f"  Loaded {len(layers_list)} transformer layers")
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        layer_devices = {}
        for k, v in dmap.items():
            if k.startswith('model.layers.'):
                lid = k.split('.')[2]
                if lid not in layer_devices:
                    layer_devices[lid] = str(v)
        gpu_layers = sum(1 for v in layer_devices.values() if 'cuda' in v)
        cpu_layers = sum(1 for v in layer_devices.values() if 'cpu' in v)
        plog(f"  Layer distribution: {gpu_layers} GPU + {cpu_layers} CPU")
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    plog(f"  {model_name}: device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


# ==================== 基础工具 ====================
def find_token_id(tokenizer, word):
    vocab = tokenizer.get_vocab()
    for candidate in [word, f" {word}", word.lower(), f" {word.lower()}"]:
        if candidate in vocab:
            return vocab[candidate]
    return None


def compute_dcf(logits, tokenizer, dim_dict, dim_names):
    dcf_vector = []
    for dim_name in dim_names:
        words = dim_dict.get(dim_name, [])
        logit_values = []
        for w in words:
            tid = find_token_id(tokenizer, w)
            if tid is not None and tid < len(logits):
                logit_values.append(float(logits[tid]))
        dcf_vector.append(float(np.mean(logit_values)) if logit_values else 0.0)
    return np.array(dcf_vector)


def logit_lens_dcf(resid, W_U, tokenizer, dim_dict=None, dim_names=None):
    if dim_dict is None:
        dim_dict = FAMILY_WORDS_8D
    if dim_names is None:
        dim_names = DCF_DIM_NAMES
    logits = resid @ W_U.T
    return compute_dcf(logits, tokenizer, dim_dict, dim_names)


def _make_capture_hook(store_dict, key):
    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            store_dict[key] = output[0].detach().float().cpu()
        else:
            store_dict[key] = output.detach().float().cpu()
    return hook_fn


def get_prompt_ids(tokenizer, device, prompt, max_len=128):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = attention_mask.sum().item()
    pos = seq_len - 1
    return input_ids, attention_mask, pos


def make_inject_hook(ivec, position):
    added = [False]
    def hook_fn(module, inp, output):
        if not added[0]:
            if isinstance(output, tuple):
                out = output[0].clone()
            else:
                out = output.clone()
            out[0, position, :] += ivec.to(out.device).to(out.dtype)
            added[0] = True
            if isinstance(output, tuple):
                return (out,) + output[1:]
            return out
        return output
    return hook_fn


def qr_orthogonalize(target_vec, basis_vecs):
    """QR子空间投影法(顺序无关)"""
    if not basis_vecs:
        return target_vec.copy()
    basis = np.array(basis_vecs)
    Q, _ = np.linalg.qr(basis.T)
    proj = Q @ (Q.T @ target_vec)
    return target_vec - proj


def compute_selectivity(dcf_delta, target_idx=0):
    target = abs(dcf_delta[target_idx])
    max_other = max(abs(dcf_delta[i]) for i in range(len(dcf_delta)) if i != target_idx)
    return target / (max_other + 0.01)


def safe_cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def get_test_layers(model_name):
    if model_name == "qwen3":
        return {"mid": 30, "last": 35}
    elif model_name == "glm4":
        return {"mid": 33, "last": 39}
    else:
        return {"mid": 24, "last": 27}


# ==================== 核心: 获取类别方向 ====================
def get_category_residuals_at_layer(model, tokenizer, device, model_name, categories=None,
                                     n_obj=4, target_layer=None, template_key="kind_of"):
    """获取各类别在指定层的residual"""
    if categories is None:
        categories = CATEGORIES
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    if target_layer is None:
        target_layer = get_test_layers(model_name)["mid"]
    
    results = {}
    template = RELATION_TEMPLATES[template_key]
    for cat_name, cat_objs in categories.items():
        resids = []
        for obj in cat_objs[:n_obj]:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap = {}
            h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap:
                resids.append(cap["resid"][0, pos].numpy())
        if resids:
            results[cat_name] = np.mean(resids, axis=0)
    return results


# ==================== Exp1: 自动邻居选择 ====================
def exp1_auto_neighbor(model, tokenizer, device, model_name, W_U):
    """基于raw direction cosine自动选择最近邻，构造specific方向"""
    plog("=== Exp1: 自动邻居选择 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    target_layer = get_test_layers(model_name)["mid"]
    
    # Step1: 获取所有类别的raw direction
    plog(f"  Computing raw directions at L{target_layer}...")
    cat_names = list(CATEGORIES.keys())
    raw_dirs = get_category_residuals_at_layer(model, tokenizer, device, model_name,
                                                 categories=CATEGORIES, n_obj=4,
                                                 target_layer=target_layer)
    
    # Step2: 计算raw direction两两余弦
    raw_cos = {}
    for i, c1 in enumerate(cat_names):
        for j, c2 in enumerate(cat_names):
            if i < j and c1 in raw_dirs and c2 in raw_dirs:
                raw_cos[f"{c1}_vs_{c2}"] = safe_cos(raw_dirs[c1], raw_dirs[c2])
    
    plog(f"  Raw direction cosine matrix computed ({len(raw_cos)} pairs)")
    
    # Step3: 对每个类别自动选择top2最近邻
    auto_neighbors = {}
    for c in cat_names:
        if c not in raw_dirs:
            continue
        cos_scores = []
        for c2 in cat_names:
            if c2 == c or c2 not in raw_dirs:
                continue
            key = f"{c}_vs_{c2}" if f"{c}_vs_{c2}" in raw_cos else f"{c2}_vs_{c}"
            cos_scores.append((c2, raw_cos.get(key, 0.0)))
        cos_scores.sort(key=lambda x: x[1], reverse=True)
        auto_neighbors[c] = [cos_scores[0][0], cos_scores[1][0]]
        plog(f"  {c}: auto_neighbors={[s[0] for s in cos_scores[:3]]}, "
              f"cos={[f'{s[1]:.3f}' for s in cos_scores[:3]]}")
    
    # Step4: 用自动邻居构造specific方向
    results_auto = {}
    results_manual = {}
    
    for cat_name in cat_names:
        if cat_name not in raw_dirs:
            continue
        target_vec = raw_dirs[cat_name]
        target_idx = cat_names.index(cat_name)
        
        # 自动邻居
        auto_basis = [raw_dirs[n] for n in auto_neighbors[cat_name] if n in raw_dirs]
        if auto_basis:
            spec_auto = qr_orthogonalize(target_vec, auto_basis)
        else:
            spec_auto = target_vec.copy()
        
        # 人工邻居(Phase 480)
        manual_basis = [raw_dirs[n] for n in MANUAL_NEIGHBORS[cat_name] if n in raw_dirs]
        if manual_basis:
            spec_manual = qr_orthogonalize(target_vec, manual_basis)
        else:
            spec_manual = target_vec.copy()
        
        # 测试注入
        for label, spec_vec, result_dict in [
            ("auto", spec_auto, results_auto),
            ("manual", spec_manual, results_manual),
        ]:
            # 注入测试: 用5个test对象
            test_objs = CATEGORIES[cat_name][:4]
            template = RELATION_TEMPLATES["kind_of"]
            dcf_deltas = []
            attr_deltas = []
            
            for obj in test_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                
                # Baseline
                cap = {}
                h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                if "resid" not in cap:
                    continue
                baseline_resid = cap["resid"][0, pos].numpy()
                baseline_dcf = logit_lens_dcf(baseline_resid, W_U, tokenizer)
                baseline_attr = logit_lens_dcf(baseline_resid, W_U, tokenizer,
                                                dim_dict=ATTRIBUTE_WORDS, dim_names=ATTR_DIM_NAMES)
                
                # 注入
                ivec = torch.tensor(spec_vec, dtype=torch.float32)
                h = layers_list[target_layer].register_forward_hook(make_inject_hook(ivec, pos))
                cap2 = {}
                h2 = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap2, "resid2"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                h2.remove()
                if "resid2" not in cap2:
                    continue
                inject_resid = cap2["resid2"][0, pos].numpy()
                inject_dcf = logit_lens_dcf(inject_resid, W_U, tokenizer)
                inject_attr = logit_lens_dcf(inject_resid, W_U, tokenizer,
                                             dim_dict=ATTRIBUTE_WORDS, dim_names=ATTR_DIM_NAMES)
                
                dcf_deltas.append(inject_dcf - baseline_dcf)
                attr_deltas.append(inject_attr - baseline_attr)
            
            if dcf_deltas:
                mean_dcf_delta = np.mean(dcf_deltas, axis=0)
                mean_attr_delta = np.mean(attr_deltas, axis=0)
                sel = compute_selectivity(mean_dcf_delta, target_idx)
                
                result_dict[cat_name] = {
                    "dcf_delta": {cat_names[i]: float(mean_dcf_delta[i]) for i in range(len(cat_names))},
                    "attr_delta": {ATTR_DIM_NAMES[i]: float(mean_attr_delta[i]) for i in range(len(ATTR_DIM_NAMES))},
                    "selectivity": float(sel),
                    "target_dcf_delta": float(mean_dcf_delta[target_idx]),
                    "neighbors_used": auto_neighbors[cat_name] if label == "auto" else MANUAL_NEIGHBORS[cat_name],
                }
                plog(f"  {cat_name} ({label}): sel={sel:.2f}, target_Δ={mean_dcf_delta[target_idx]:.2f}, "
                      f"neighbors={auto_neighbors[cat_name] if label == 'auto' else MANUAL_NEIGHBORS[cat_name]}")
    
    return {
        "auto_neighbors": auto_neighbors,
        "raw_cosine_matrix": raw_cos,
        "results_auto": results_auto,
        "results_manual": results_manual,
    }


# ==================== Exp2: 留出对象验证 ====================
def exp2_holdout_validation(model, tokenizer, device, model_name, W_U):
    """用train对象构造specific方向，test对象验证self-rank"""
    plog("=== Exp2: 留出对象验证 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    target_layer = get_test_layers(model_name)["mid"]
    cat_names = list(CATEGORIES.keys())
    
    # Step1: 用train对象构造raw direction
    plog("  Computing train raw directions...")
    train_raw = {}
    template = RELATION_TEMPLATES["kind_of"]
    for cat_name, cat_objs in CATEGORIES_TRAIN.items():
        resids = []
        for obj in cat_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap = {}
            h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap:
                resids.append(cap["resid"][0, pos].numpy())
        if resids:
            train_raw[cat_name] = np.mean(resids, axis=0)
    
    # Step2: 计算raw cosine, 自动选邻居
    raw_cos = {}
    for i, c1 in enumerate(cat_names):
        for j, c2 in enumerate(cat_names):
            if i < j and c1 in train_raw and c2 in train_raw:
                raw_cos[f"{c1}_vs_{c2}"] = safe_cos(train_raw[c1], train_raw[c2])
    
    auto_neighbors = {}
    for c in cat_names:
        if c not in train_raw:
            continue
        cos_scores = []
        for c2 in cat_names:
            if c2 == c or c2 not in train_raw:
                continue
            key = f"{c}_vs_{c2}" if f"{c}_vs_{c2}" in raw_cos else f"{c2}_vs_{c}"
            cos_scores.append((c2, raw_cos.get(key, 0.0)))
        cos_scores.sort(key=lambda x: x[1], reverse=True)
        auto_neighbors[c] = [cos_scores[0][0], cos_scores[1][0]]
    
    # Step3: 用train raw direction构造specific方向
    spec_dirs = {}
    for cat_name in cat_names:
        if cat_name not in train_raw:
            continue
        target_vec = train_raw[cat_name]
        basis = [train_raw[n] for n in auto_neighbors[cat_name] if n in train_raw]
        if basis:
            spec_dirs[cat_name] = qr_orthogonalize(target_vec, basis)
        else:
            spec_dirs[cat_name] = target_vec.copy()
    
    # Step4: 对test对象计算投影
    plog("  Computing test object projections...")
    holdout_results = {}
    for cat_name in cat_names:
        test_objs = CATEGORIES_TEST[cat_name]
        obj_projections = {}
        
        for obj in test_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap = {}
            h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" not in cap:
                continue
            obj_resid = cap["resid"][0, pos].numpy()
            
            # 投影到各类别specific方向
            proj = {}
            for sname, svec in spec_dirs.items():
                svec_norm = np.linalg.norm(svec)
                if svec_norm > 1e-10:
                    proj[sname] = float(np.dot(obj_resid, svec) / svec_norm)
                else:
                    proj[sname] = 0.0
            obj_projections[obj] = proj
        
        # 汇总: 每个test对象在自己类别方向上的排名
        ranks = []
        for obj, proj in obj_projections.items():
            sorted_dirs = sorted(proj.items(), key=lambda x: x[1], reverse=True)
            rank = next(i+1 for i, (name, _) in enumerate(sorted_dirs) if name == cat_name)
            ranks.append(rank)
        
        avg_rank = np.mean(ranks) if ranks else -1
        self_rank_1_count = sum(1 for r in ranks if r == 1)
        
        holdout_results[cat_name] = {
            "avg_rank": float(avg_rank),
            "self_rank_1_count": self_rank_1_count,
            "total_test_objs": len(test_objs),
            "individual_ranks": {obj: int(ranks[i]) for i, obj in enumerate(test_objs) if i < len(ranks)},
            "auto_neighbors": auto_neighbors[cat_name],
        }
        plog(f"  {cat_name}: avg_rank={avg_rank:.1f}, "
              f"self_rank_1={self_rank_1_count}/{len(test_objs)}, "
              f"neighbors={auto_neighbors[cat_name]}")
    
    # Step5: 对比全对象构造的specific方向(Phase 480方法)
    # 用train+test全部8个对象
    all_raw = {}
    for cat_name, cat_objs in CATEGORIES.items():
        resids = []
        for obj in cat_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap = {}
            h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap:
                resids.append(cap["resid"][0, pos].numpy())
        if resids:
            all_raw[cat_name] = np.mean(resids, axis=0)
    
    # 全对象自动邻居+specific方向
    all_cos = {}
    for i, c1 in enumerate(cat_names):
        for j, c2 in enumerate(cat_names):
            if i < j and c1 in all_raw and c2 in all_raw:
                all_cos[f"{c1}_vs_{c2}"] = safe_cos(all_raw[c1], all_raw[c2])
    
    all_auto_neighbors = {}
    for c in cat_names:
        if c not in all_raw:
            continue
        cos_scores = []
        for c2 in cat_names:
            if c2 == c or c2 not in all_raw:
                continue
            key = f"{c}_vs_{c2}" if f"{c}_vs_{c2}" in all_cos else f"{c2}_vs_{c}"
            cos_scores.append((c2, all_cos.get(key, 0.0)))
        cos_scores.sort(key=lambda x: x[1], reverse=True)
        all_auto_neighbors[c] = [cos_scores[0][0], cos_scores[1][0]]
    
    # 对全对象specific方向做注入测试
    all_spec_dirs = {}
    for cat_name in cat_names:
        if cat_name not in all_raw:
            continue
        target_vec = all_raw[cat_name]
        target_idx = cat_names.index(cat_name)
        basis = [all_raw[n] for n in all_auto_neighbors[cat_name] if n in all_raw]
        if basis:
            spec_vec = qr_orthogonalize(target_vec, basis)
        else:
            spec_vec = target_vec.copy()
        all_spec_dirs[cat_name] = spec_vec
        
        # 注入测试
        test_objs = CATEGORIES[cat_name][:4]
        dcf_deltas = []
        for obj in test_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap = {}
            h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" not in cap:
                continue
            baseline_resid = cap["resid"][0, pos].numpy()
            baseline_dcf = logit_lens_dcf(baseline_resid, W_U, tokenizer)
            
            ivec = torch.tensor(spec_vec, dtype=torch.float32)
            h = layers_list[target_layer].register_forward_hook(make_inject_hook(ivec, pos))
            cap2 = {}
            h2 = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap2, "resid2"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            h2.remove()
            if "resid2" not in cap2:
                continue
            inject_resid = cap2["resid2"][0, pos].numpy()
            inject_dcf = logit_lens_dcf(inject_resid, W_U, tokenizer)
            dcf_deltas.append(inject_dcf - baseline_dcf)
        
        if dcf_deltas:
            mean_dcf_delta = np.mean(dcf_deltas, axis=0)
            sel = compute_selectivity(mean_dcf_delta, cat_names.index(cat_name))
            holdout_results[cat_name]["full_inject_selectivity"] = float(sel)
            holdout_results[cat_name]["full_inject_target_delta"] = float(mean_dcf_delta[cat_names.index(cat_name)])
            holdout_results[cat_name]["full_auto_neighbors"] = all_auto_neighbors[cat_name]
    
    return holdout_results


# ==================== Exp3: DS7B多层扫描+校准 ====================
def exp3_ds7b_layer_scan(model, tokenizer, device, model_name, W_U):
    """DS7B多层扫描+注入强度校准"""
    plog("=== Exp3: DS7B多层扫描+校准 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    template = RELATION_TEMPLATES["kind_of"]
    
    # 目标类别: fruit和clothing (Phase 480失败的)
    target_cats = ["fruit", "clothing", "vehicle"]  # vehicle作为对照(成功)
    
    # 扫描层位
    n_layers = len(layers_list)
    scan_layers = list(range(max(0, n_layers - 12), n_layers))  # 最后12层
    plog(f"  Scanning layers: {scan_layers}")
    
    layer_results = {}
    
    for L in scan_layers:
        plog(f"  --- Layer {L} ---")
        
        # 获取该层的raw direction
        raw_dirs = {}
        for cat_name in target_cats:
            cat_objs = CATEGORIES[cat_name][:4]
            resids = []
            for obj in cat_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                cap = {}
                h = layers_list[L].register_forward_hook(_make_capture_hook(cap, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                if "resid" in cap:
                    resids.append(cap["resid"][0, pos].numpy())
            if resids:
                raw_dirs[cat_name] = np.mean(resids, axis=0)
        
        # 计算raw direction cosine
        raw_cos = {}
        for c1 in target_cats:
            for c2 in target_cats:
                if c1 != c2 and c1 in raw_dirs and c2 in raw_dirs:
                    raw_cos[f"{c1}_vs_{c2}"] = safe_cos(raw_dirs[c1], raw_dirs[c2])
        
        # 自动选择邻居(从全部8类别)
        all_raw_dirs = {}
        for cat_name in cat_names:
            cat_objs = CATEGORIES[cat_name][:4]
            resids = []
            for obj in cat_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                cap = {}
                h = layers_list[L].register_forward_hook(_make_capture_hook(cap, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                if "resid" in cap:
                    resids.append(cap["resid"][0, pos].numpy())
            if resids:
                all_raw_dirs[cat_name] = np.mean(resids, axis=0)
        
        # 对fruit和clothing自动选邻居
        for tc in target_cats:
            if tc not in all_raw_dirs:
                continue
            cos_scores = []
            for c2 in cat_names:
                if c2 == tc or c2 not in all_raw_dirs:
                    continue
                cos_scores.append((c2, safe_cos(all_raw_dirs[tc], all_raw_dirs[c2])))
            cos_scores.sort(key=lambda x: x[1], reverse=True)
            auto_neighbors = [cos_scores[0][0], cos_scores[1][0]]
            
            # 构造specific方向
            target_vec = all_raw_dirs[tc]
            basis = [all_raw_dirs[n] for n in auto_neighbors if n in all_raw_dirs]
            if basis:
                spec_vec = qr_orthogonalize(target_vec, basis)
            else:
                spec_vec = target_vec.copy()
            
            # 注入强度校准: 按raw方向范数缩放
            raw_norm = np.linalg.norm(target_vec)
            spec_norm = np.linalg.norm(spec_vec)
            # 标准化注入: 使specific方向注入量 = raw方向注入量的指定比例
            # Phase 480的注入量是1x spec_vec, 这里尝试0.3x和0.5x
            for scale in [1.0, 0.5, 0.3]:
                inject_vec = spec_vec * scale
                
                test_objs = CATEGORIES[tc][:4]
                dcf_deltas = []
                for obj in test_objs:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                    
                    # Baseline
                    cap = {}
                    h = layers_list[L].register_forward_hook(_make_capture_hook(cap, "resid"))
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    h.remove()
                    if "resid" not in cap:
                        continue
                    baseline_resid = cap["resid"][0, pos].numpy()
                    baseline_dcf = logit_lens_dcf(baseline_resid, W_U, tokenizer)
                    
                    # 注入
                    ivec = torch.tensor(inject_vec, dtype=torch.float32)
                    h = layers_list[L].register_forward_hook(make_inject_hook(ivec, pos))
                    cap2 = {}
                    h2 = layers_list[L].register_forward_hook(_make_capture_hook(cap2, "resid2"))
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    h.remove()
                    h2.remove()
                    if "resid2" not in cap2:
                        continue
                    inject_resid = cap2["resid2"][0, pos].numpy()
                    inject_dcf = logit_lens_dcf(inject_resid, W_U, tokenizer)
                    dcf_deltas.append(inject_dcf - baseline_dcf)
                
                if dcf_deltas:
                    mean_dcf = np.mean(dcf_deltas, axis=0)
                    sel = compute_selectivity(mean_dcf, cat_names.index(tc))
                    target_d = float(mean_dcf[cat_names.index(tc)])
                    
                    key = f"L{L}_{tc}_s{scale}"
                    layer_results[key] = {
                        "layer": L,
                        "category": tc,
                        "scale": scale,
                        "selectivity": float(sel),
                        "target_dcf_delta": target_d,
                        "dcf_delta": {cat_names[i]: float(mean_dcf[i]) for i in range(len(cat_names))},
                        "auto_neighbors": auto_neighbors,
                        "spec_norm": float(spec_norm),
                        "raw_norm": float(raw_norm),
                        "norm_ratio": float(spec_norm / raw_norm) if raw_norm > 0 else 0,
                    }
                    plog(f"  L{L} {tc} scale={scale}: sel={sel:.2f}, target_Δ={target_d:.2f}, "
                          f"norm_ratio={spec_norm/raw_norm:.3f}" if raw_norm > 0 else "")
        
        # 定期清理GPU
        if L % 3 == 0:
            torch.cuda.empty_cache()
    
    # 找每个类别的最佳层
    best_results = {}
    for tc in target_cats:
        best_sel = 0
        best_key = None
        for key, val in layer_results.items():
            if val["category"] == tc and val["selectivity"] > best_sel:
                best_sel = val["selectivity"]
                best_key = key
        if best_key:
            best_results[tc] = layer_results[best_key]
            plog(f"  Best for {tc}: L{layer_results[best_key]['layer']}, "
                  f"scale={layer_results[best_key]['scale']}, "
                  f"sel={best_sel:.2f}")
    
    return {
        "layer_results": layer_results,
        "best_results": best_results,
    }


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    plog(f"Phase 481: {model_name} round {round_num}")
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    plog(f"  W_U: shape={W_U.shape}")
    
    results = {
        "phase": 481,
        "model": model_name,
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "core_question": "Can auto-neighbor selection, holdout validation, and DS7B layer scan improve category boundary residuals?",
        "model_info": {
            "class": type(model).__name__,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }
    
    t0 = time.time()
    
    if model_name == "deepseek7b":
        # DS7B: Exp3多层扫描
        plog("Running Exp3: DS7B layer scan...")
        exp3 = exp3_ds7b_layer_scan(model, tokenizer, device, model_name, W_U)
        results["exp3_ds7b_layer_scan"] = exp3
        
        # DS7B也做Exp1(自动邻居)
        plog("Running Exp1: Auto neighbor selection...")
        exp1 = exp1_auto_neighbor(model, tokenizer, device, model_name, W_U)
        results["exp1_auto_neighbor"] = exp1
    else:
        # Qwen3/GLM4: Exp1 + Exp2
        plog("Running Exp1: Auto neighbor selection...")
        exp1 = exp1_auto_neighbor(model, tokenizer, device, model_name, W_U)
        results["exp1_auto_neighbor"] = exp1
        
        plog("Running Exp2: Holdout validation...")
        exp2 = exp2_holdout_validation(model, tokenizer, device, model_name, W_U)
        results["exp2_holdout_validation"] = exp2
    
    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)
    plog(f"Total time: {elapsed:.1f}s")
    
    # 保存
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase481_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    plog(f"Saved to {out_path}")
    
    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    plog("Model released.")


if __name__ == "__main__":
    main()
