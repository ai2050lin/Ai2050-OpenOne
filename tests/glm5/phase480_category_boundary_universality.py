"""
Phase 480: 类别边界残差的普遍性验证
====================================

核心目标:
1. 多类别specific方向构造 — animal/tool/vehicle/clothing/food/plant是否都能解耦出高选择性残差?
2. 类别边界残差属性画像 — 各specific方向的属性DCF profile
3. 自然使用验证 — 自然prompt中category_specific投影是否按预期出现
4. 反向注入测试 — -category_specific是否抑制对应类别
5. 跨模型复现 — GLM4/DS7B是否也有同样的多类别解耦结构
6. DS7B格式覆盖电路行为闭环

实验:
1. Exp1: 多类别specific方向构造+选择性测试 (ALL models) ★★★核心★★★
2. Exp2: 类别边界残差属性画像 (Qwen3)
3. Exp3: 自然使用验证 — 多类别对象的specific投影 (Qwen3)
4. Exp4: 反向注入测试 — -specific是否抑制 (Qwen3)
5. Exp5: 跨模型多类别复现 (GLM4/DS7B, 同Exp1)
6. Exp6: DS7B格式覆盖电路行为闭环 (DS7B)

用法:
  python tests/glm5/phase480_category_boundary_universality.py qwen3 1
  python tests/glm5/phase480_category_boundary_universality.py glm4 1
  python tests/glm5/phase480_category_boundary_universality.py deepseek7b 1
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
    "seat_like":    ["seat", "sit", "sit", "rest", "support", "comfort"],
    "locomotion":   ["move", "travel", "drive", "fly", "ride", "transport"],
    "has_legs":     ["legs", "feet", "walk", "run", "stand", "limb"],
}

ATTR_DIM_NAMES = list(ATTRIBUTE_WORDS.keys())

# 类别解耦定义: 每个类别的"邻居类别"用于构造specific残差
# 关键: 用语义上最近邻的类别来做正交化
CATEGORY_NEIGHBORS = {
    "fruit":     ["plant", "food"],           # fruit和plant/food共享最多
    "animal":    ["food", "clothing"],         # animal和food(肉)/clothing(皮毛)有交叉
    "tool":      ["furniture", "vehicle"],    # tool和furniture/vehicle共享human_made+object
    "vehicle":   ["tool", "furniture"],        # vehicle和tool/furniture共享mechanical+human_made
    "clothing":  ["furniture", "tool"],       # clothing和furniture/tool共享fabric+human_made
    "furniture": ["tool", "clothing"],         # furniture和tool/clothing共享indoor+human_made
    "food":      ["plant", "fruit"],           # food和plant/fruit共享edible+natural
    "plant":     ["food", "fruit"],            # plant和food/fruit共享natural+living
}

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


def gram_schmidt_orth(target_vec, basis_vecs):
    """从target_vec中去除所有basis_vecs方向的投影"""
    residual = target_vec.copy()
    for bv in basis_vecs:
        bv_norm2 = np.dot(bv, bv)
        if bv_norm2 < 1e-10:
            continue
        proj = np.dot(residual, bv) / bv_norm2 * bv
        residual = residual - proj
    return residual


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


def run_injection_test(model, tokenizer, device, model_name, inject_vec, test_objs,
                        template_key="kind_of", target_layer=None):
    """通用注入测试: 返回8D DCF delta和属性DCF delta"""
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)
    if target_layer is None:
        target_layer = get_test_layers(model_name)["mid"]
    last_layer = info.n_layers - 1
    
    inject_tensor = torch.tensor(inject_vec, dtype=torch.float32)
    template = RELATION_TEMPLATES[template_key]
    
    dcf_before = []
    dcf_after = []
    attr_before = []
    attr_after = []
    
    for obj in test_objs:
        prompt = template.format(obj=obj)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        
        # Clean
        cap_clean = {}
        h_clean = layers_list[last_layer].register_forward_hook(_make_capture_hook(cap_clean, "resid"))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h_clean.remove()
        if "resid" not in cap_clean:
            continue
        clean_r = cap_clean["resid"][0, pos].numpy()
        dcf_before.append(logit_lens_dcf(clean_r, W_U, tokenizer))
        attr_before.append(logit_lens_dcf(clean_r, W_U, tokenizer, ATTRIBUTE_WORDS, ATTR_DIM_NAMES))
        
        # Injected
        cap_pert = {}
        h_pert = layers_list[last_layer].register_forward_hook(_make_capture_hook(cap_pert, "resid"))
        h_inj = layers_list[target_layer].register_forward_hook(make_inject_hook(inject_tensor, pos))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h_pert.remove()
        h_inj.remove()
        if "resid" in cap_pert:
            pert_r = cap_pert["resid"][0, pos].numpy()
            dcf_after.append(logit_lens_dcf(pert_r, W_U, tokenizer))
            attr_after.append(logit_lens_dcf(pert_r, W_U, tokenizer, ATTRIBUTE_WORDS, ATTR_DIM_NAMES))
    
    if not dcf_before:
        return None
    
    mean_dcf_delta = np.mean(dcf_after, axis=0) - np.mean(dcf_before, axis=0)
    mean_attr_delta = np.mean(attr_after, axis=0) - np.mean(attr_before, axis=0)
    
    return {
        "dcf_delta": {DCF_DIM_NAMES[i]: float(mean_dcf_delta[i]) for i in range(len(DCF_DIM_NAMES))},
        "attr_delta": {ATTR_DIM_NAMES[i]: float(mean_attr_delta[i]) for i in range(len(ATTR_DIM_NAMES))},
        "selectivity": float(compute_selectivity(mean_dcf_delta, 0)),
        "target_dcf_delta": float(mean_dcf_delta[0]),
    }


# ==================== Exp1: 多类别specific方向构造 (ALL models) ★★★核心★★★ ====================
def exp1_multi_category_specific(model, tokenizer, device, model_name):
    """
    对每个类别构造category_specific = category_raw - Proj(neighbor_categories)
    测试每个specific方向的选择性
    
    核心问题: 类别边界残差是fruit独有的, 还是普遍机制?
    """
    plog(f"=== Exp1: Multi-Category Specific Direction Construction ({model_name}) ===")
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    target_layer = get_test_layers(model_name)["mid"]
    
    # Step1: 获取所有类别的residual方向
    plog(f"  Step1: Getting category residual directions at L{target_layer}...")
    cat_resids = get_category_residuals_at_layer(model, tokenizer, device, model_name)
    
    # 检查哪些类别获取成功
    available_cats = [k for k in CATEGORIES.keys() if k in cat_resids]
    plog(f"  Available categories: {available_cats}")
    
    # Step2: 对每个类别构造specific方向
    plog(f"  Step2: Computing category-specific directions...")
    specific_directions = {}
    raw_directions = {}
    
    for cat_name in available_cats:
        cat_raw = cat_resids[cat_name]
        raw_directions[cat_name] = cat_raw
        
        neighbors = CATEGORY_NEIGHBORS.get(cat_name, [])
        neighbor_vecs = [cat_resids[n] for n in neighbors if n in cat_resids]
        
        if not neighbor_vecs:
            specific_directions[cat_name] = cat_raw.copy()
            plog(f"    {cat_name}: no neighbors available, using raw")
            continue
        
        # 使用QR正交化(顺序无关)
        cat_specific = qr_orthogonalize(cat_raw, neighbor_vecs)
        specific_directions[cat_name] = cat_specific
        
        raw_norm = np.linalg.norm(cat_raw)
        spec_norm = np.linalg.norm(cat_specific)
        ratio = spec_norm / raw_norm if raw_norm > 1e-10 else 0
        plog(f"    {cat_name}: raw_norm={raw_norm:.1f}, spec_norm={spec_norm:.1f}, "
              f"ratio={ratio:.4f}, neighbors={neighbors}")
    
    # Step3: 归一化所有方向到统一范数
    ref_norm = np.linalg.norm(specific_directions.get("fruit", np.ones(info.d_model)))
    if ref_norm < 1e-10:
        ref_norm = 1.0
    
    for cat_name in specific_directions:
        n = np.linalg.norm(specific_directions[cat_name])
        if n > 1e-10:
            specific_directions[cat_name] = specific_directions[cat_name] / n * ref_norm
            raw_directions[cat_name] = raw_directions[cat_name] / np.linalg.norm(raw_directions[cat_name]) * ref_norm
    
    # Step4: 注入测试 — 每个specific方向 + 每个raw方向
    plog(f"  Step3: Injection tests...")
    
    # 测试对象: 用非目标类别的对象作为"基线对象", 然后注入看目标类别提升
    # 也用目标类别对象看效果
    test_objs_per_cat = {}
    for cat_name in available_cats:
        # 使用其他类别的混合对象
        other_objs = []
        for other_cat in ["animal", "tool", "vehicle", "clothing"]:
            if other_cat != cat_name:
                other_objs.extend(CATEGORIES.get(other_cat, [])[:2])
        test_objs_per_cat[cat_name] = other_objs[:6]
    
    results_specific = {}
    results_raw = {}
    
    for cat_name in available_cats:
        target_idx = DCF_DIM_NAMES.index(cat_name) if cat_name in DCF_DIM_NAMES else 0
        test_objs = test_objs_per_cat.get(cat_name, CATEGORIES["animal"][:4])
        
        # Test specific direction
        spec_vec = specific_directions[cat_name]
        plog(f"    Testing {cat_name}_specific...")
        r_spec = run_injection_test(model, tokenizer, device, model_name, spec_vec, test_objs)
        if r_spec:
            # 重新计算selectivity以target_idx为基准
            dcf_delta = np.array([r_spec["dcf_delta"][d] for d in DCF_DIM_NAMES])
            sel = compute_selectivity(dcf_delta, target_idx)
            r_spec["selectivity_target"] = float(sel)
            r_spec["target_idx"] = target_idx
            results_specific[cat_name] = r_spec
            plog(f"      {cat_name}_specific: {cat_name}_Δ={dcf_delta[target_idx]:.3f}, "
                  f"sel={sel:.2f}")
        
        # Test raw direction
        raw_vec = raw_directions[cat_name]
        r_raw = run_injection_test(model, tokenizer, device, model_name, raw_vec, test_objs)
        if r_raw:
            dcf_delta = np.array([r_raw["dcf_delta"][d] for d in DCF_DIM_NAMES])
            sel = compute_selectivity(dcf_delta, target_idx)
            r_raw["selectivity_target"] = float(sel)
            results_raw[cat_name] = r_raw
    
    # Step5: specific方向间余弦矩阵
    plog(f"  Step4: Inter-specific cosine matrix...")
    cos_matrix = {}
    cats_list = list(specific_directions.keys())
    for i in range(len(cats_list)):
        for j in range(i+1, len(cats_list)):
            c1, c2 = cats_list[i], cats_list[j]
            cos_val = safe_cos(specific_directions[c1], specific_directions[c2])
            cos_matrix[f"{c1}_vs_{c2}"] = cos_val
    
    # 也算raw方向的余弦
    raw_cos_matrix = {}
    for i in range(len(cats_list)):
        for j in range(i+1, len(cats_list)):
            c1, c2 = cats_list[i], cats_list[j]
            cos_val = safe_cos(raw_directions[c1], raw_directions[c2])
            raw_cos_matrix[f"{c1}_vs_{c2}"] = cos_val
    
    return {
        "specific_injection": results_specific,
        "raw_injection": results_raw,
        "specific_cosine_matrix": cos_matrix,
        "raw_cosine_matrix": raw_cos_matrix,
        "norm_ratios": {cat: float(np.linalg.norm(specific_directions[cat]) / 
                      (np.linalg.norm(raw_directions[cat]) + 1e-10))
                      for cat in specific_directions},
    }


# ==================== Exp2: 类别边界残差属性画像 (Qwen3) ====================
def exp2_attribute_profiles(model, tokenizer, device, model_name):
    """对每个category_specific方向做完整的属性DCF画像"""
    if model_name != "qwen3":
        return {"skipped": True, "reason": "only for qwen3"}
    
    plog(f"=== Exp2: Category-Specific Attribute Profiles ({model_name}) ===")
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    target_layer = get_test_layers(model_name)["mid"]
    
    # 获取所有类别residual
    cat_resids = get_category_residuals_at_layer(model, tokenizer, device, model_name)
    
    # 构造specific方向
    specific_directions = {}
    for cat_name in CATEGORIES:
        if cat_name not in cat_resids:
            continue
        cat_raw = cat_resids[cat_name]
        neighbors = CATEGORY_NEIGHBORS.get(cat_name, [])
        neighbor_vecs = [cat_resids[n] for n in neighbors if n in cat_resids]
        cat_specific = qr_orthogonalize(cat_raw, neighbor_vecs) if neighbor_vecs else cat_raw.copy()
        specific_directions[cat_name] = cat_specific
    
    # 归一化
    ref_norm = np.linalg.norm(specific_directions.get("fruit", np.ones(info.d_model)))
    for cat_name in specific_directions:
        n = np.linalg.norm(specific_directions[cat_name])
        if n > 1e-10:
            specific_directions[cat_name] = specific_directions[cat_name] / n * ref_norm
    
    # 每个specific方向的属性DCF画像 (用fruit对象的baseline)
    test_objs = CATEGORIES["animal"][:3] + CATEGORIES["tool"][:2]
    
    profiles = {}
    for cat_name, spec_vec in specific_directions.items():
        plog(f"  Profiling {cat_name}_specific...")
        r = run_injection_test(model, tokenizer, device, model_name, spec_vec, test_objs)
        if r:
            profiles[cat_name] = {
                "dcf_delta": r["dcf_delta"],
                "attr_delta": r["attr_delta"],
            }
            # 找出top3正向和top3负向属性
            attr_d = r["attr_delta"]
            sorted_attrs = sorted(attr_d.items(), key=lambda x: -x[1])
            top3_pos = sorted_attrs[:3]
            top3_neg = sorted_attrs[-3:]
            plog(f"    Top3 positive: {[(a, f'{v:.2f}') for a, v in top3_pos]}")
            plog(f"    Top3 negative: {[(a, f'{v:.2f}') for a, v in top3_neg]}")
    
    return profiles


# ==================== Exp3: 自然使用验证 (Qwen3) ====================
def exp3_natural_usage(model, tokenizer, device, model_name):
    """
    测试自然prompt中category_specific方向的投影
    
    如果fruit-specific是自然编码方向, 那么:
    - fruit对象应该在fruit_specific上投影最高
    - animal对象应该在animal_specific上投影最高
    - etc.
    """
    if model_name != "qwen3":
        return {"skipped": True, "reason": "only for qwen3"}
    
    plog(f"=== Exp3: Natural Usage Verification ({model_name}) ===")
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    target_layer = get_test_layers(model_name)["mid"]
    layers_list = get_layers(model)
    
    # 获取所有类别residual
    cat_resids = get_category_residuals_at_layer(model, tokenizer, device, model_name, n_obj=6)
    
    # 构造specific方向
    specific_directions = {}
    for cat_name in CATEGORIES:
        if cat_name not in cat_resids:
            continue
        cat_raw = cat_resids[cat_name]
        neighbors = CATEGORY_NEIGHBORS.get(cat_name, [])
        neighbor_vecs = [cat_resids[n] for n in neighbors if n in cat_resids]
        cat_specific = qr_orthogonalize(cat_raw, neighbor_vecs) if neighbor_vecs else cat_raw.copy()
        # 归一化
        n = np.linalg.norm(cat_specific)
        if n > 1e-10:
            specific_directions[cat_name] = cat_specific / n
        else:
            specific_directions[cat_name] = cat_specific
    
    # 对每个类别的大量对象, 计算它们在各类specific方向上的投影
    plog(f"  Computing projections for {len(CATEGORIES)} categories x 6 objects...")
    
    projection_matrix = {}  # cat_name -> {specific_name: mean_projection}
    
    for cat_name, cat_objs in CATEGORIES.items():
        projections = {spec_name: [] for spec_name in specific_directions}
        
        for obj in cat_objs[:6]:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap = {}
            h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" not in cap:
                continue
            resid = cap["resid"][0, pos].numpy()
            
            # 计算在各类specific方向上的投影
            for spec_name, spec_dir in specific_directions.items():
                proj = np.dot(resid, spec_dir)  # spec_dir已归一化
                projections[spec_name].append(float(proj))
        
        # 取均值
        mean_proj = {}
        for spec_name, vals in projections.items():
            if vals:
                mean_proj[spec_name] = float(np.mean(vals))
        projection_matrix[cat_name] = mean_proj
    
    # 分析: 每个类别在自己的specific方向上是否投影最高
    plog(f"  Analyzing projection alignment...")
    alignment_results = {}
    
    for cat_name in projection_matrix:
        projs = projection_matrix[cat_name]
        if not projs:
            continue
        
        # 找出投影最高的specific方向
        sorted_projs = sorted(projs.items(), key=lambda x: -x[1])
        top1 = sorted_projs[0]
        
        # 自身的specific投影排名
        self_proj = projs.get(cat_name, 0)
        self_rank = next((i+1 for i, (k, _) in enumerate(sorted_projs) if k == cat_name), -1)
        
        alignment_results[cat_name] = {
            "self_projection": float(self_proj),
            "self_rank": self_rank,
            "top1_direction": top1[0],
            "top1_value": float(top1[1]),
            "all_projections": projs,
        }
        
        match = "✓" if top1[0] == cat_name else "✗"
        plog(f"    {cat_name}: self_proj={self_proj:.3f} (rank #{self_rank}), "
              f"top1={top1[0]}={top1[1]:.3f} {match}")
    
    return alignment_results


# ==================== Exp4: 反向注入测试 (Qwen3) ====================
def exp4_reverse_injection(model, tokenizer, device, model_name):
    """
    注入-category_specific方向, 看是否抑制对应类别
    
    如果fruit-specific是因果编码方向:
    - 注入-fruit_specific应该抑制fruit而提升其他类别?
    """
    if model_name != "qwen3":
        return {"skipped": True, "reason": "only for qwen3"}
    
    plog(f"=== Exp4: Reverse Injection Test ({model_name}) ===")
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    target_layer = get_test_layers(model_name)["mid"]
    
    # 获取类别residual
    cat_resids = get_category_residuals_at_layer(model, tokenizer, device, model_name)
    
    # 构造specific方向
    specific_directions = {}
    for cat_name in CATEGORIES:
        if cat_name not in cat_resids:
            continue
        cat_raw = cat_resids[cat_name]
        neighbors = CATEGORY_NEIGHBORS.get(cat_name, [])
        neighbor_vecs = [cat_resids[n] for n in neighbors if n in cat_resids]
        cat_specific = qr_orthogonalize(cat_raw, neighbor_vecs) if neighbor_vecs else cat_raw.copy()
        specific_directions[cat_name] = cat_specific
    
    # 归一化
    ref_norm = np.linalg.norm(specific_directions.get("fruit", np.ones(info.d_model)))
    for cat_name in specific_directions:
        n = np.linalg.norm(specific_directions[cat_name])
        if n > 1e-10:
            specific_directions[cat_name] = specific_directions[cat_name] / n * ref_norm
    
    # 测试: +specific vs -specific vs baseline (用该类别的对象)
    results = {}
    
    test_categories = ["fruit", "animal", "tool", "vehicle"]
    
    for cat_name in test_categories:
        if cat_name not in specific_directions:
            continue
        
        # 用该类别对象做测试(应该看到: +specific提升, -specific抑制)
        test_objs = CATEGORIES.get(cat_name, CATEGORIES["fruit"])[:4]
        
        spec_vec = specific_directions[cat_name]
        neg_spec_vec = -spec_vec
        
        # Positive injection
        r_pos = run_injection_test(model, tokenizer, device, model_name, spec_vec, test_objs)
        # Negative injection
        r_neg = run_injection_test(model, tokenizer, device, model_name, neg_spec_vec, test_objs)
        
        target_idx = DCF_DIM_NAMES.index(cat_name) if cat_name in DCF_DIM_NAMES else 0
        
        results[cat_name] = {
            "positive": r_pos,
            "negative": r_neg,
        }
        
        if r_pos and r_neg:
            pos_target = r_pos["dcf_delta"].get(cat_name, 0)
            neg_target = r_neg["dcf_delta"].get(cat_name, 0)
            plog(f"    {cat_name}: +spec→{cat_name}_Δ={pos_target:.3f}, "
                  f"-spec→{cat_name}_Δ={neg_target:.3f}, "
                  f"asymmetry={pos_target + neg_target:.3f}")
    
    return results


# ==================== Exp5: 跨模型多类别复现 ====================
# 与Exp1相同逻辑, 但对GLM4/DS7B运行
# 在main()中直接调用exp1_multi_category_specific


# ==================== Exp6: DS7B格式覆盖电路行为闭环 ====================
def exp6_ds7b_format_circuit(model, tokenizer, device, model_name):
    """DS7B格式覆盖电路: Head 12+13+10组合的行为验证"""
    if model_name != "deepseek7b":
        return {"skipped": True, "reason": "only for deepseek7b"}
    
    plog(f"=== Exp6: DS7B Format Override Circuit Behavioral Closure ({model_name}) ===")
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    target_layer = get_test_layers(model_name)["mid"]  # L27
    last_layer = info.n_layers - 1
    layers_list = get_layers(model)
    
    # 数学prompt vs 普通prompt
    math_prompts = [
        "Calculate 15 + 27 =",
        "What is 8 times 9?",
        "Solve for x: 2x + 3 = 11, x =",
        "The sum of 45 and 38 is",
    ]
    normal_prompts = [
        "The weather today is",
        "She walked to the",
        "The book was about",
        "He opened the door and",
    ]
    
    # Step1: 获取math vs normal在L27的residual差异
    plog(f"  Step1: Getting math vs normal residuals at L{target_layer}...")
    math_resids = []
    normal_resids = []
    
    for prompt in math_prompts:
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap = {}
        h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h.remove()
        if "resid" in cap:
            math_resids.append(cap["resid"][0, pos].numpy())
    
    for prompt in normal_prompts:
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap = {}
        h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h.remove()
        if "resid" in cap:
            normal_resids.append(cap["resid"][0, pos].numpy())
    
    if not math_resids or not normal_resids:
        return {"error": "Failed to capture residuals"}
    
    math_mean = np.mean(math_resids, axis=0)
    normal_mean = np.mean(normal_resids, axis=0)
    format_direction = math_mean - normal_mean
    
    # Step2: DCF画像 — math vs normal
    plog(f"  Step2: DCF profiles for math vs normal prompts...")
    math_dcfs = []
    normal_dcfs = []
    
    for prompt in math_prompts:
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap = {}
        h = layers_list[last_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h.remove()
        if "resid" in cap:
            math_dcfs.append(logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer))
    
    for prompt in normal_prompts:
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap = {}
        h = layers_list[last_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h.remove()
        if "resid" in cap:
            normal_dcfs.append(logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer))
    
    math_dcf_mean = np.mean(math_dcfs, axis=0) if math_dcfs else np.zeros(len(DCF_DIM_NAMES))
    normal_dcf_mean = np.mean(normal_dcfs, axis=0) if normal_dcfs else np.zeros(len(DCF_DIM_NAMES))
    dcf_diff = math_dcf_mean - normal_dcf_mean
    
    plog(f"    Math vs Normal DCF diff: " +
          ", ".join([f"{DCF_DIM_NAMES[i]}={dcf_diff[i]:.3f}" for i in range(len(DCF_DIM_NAMES))]))
    
    # Step3: format_direction注入测试
    plog(f"  Step3: Format direction injection test...")
    fmt_norm = np.linalg.norm(format_direction)
    if fmt_norm > 1e-10:
        # 用正常prompt测试注入format_direction的效果
        inject_vec = format_direction / fmt_norm * 10.0  # 缩放到适中强度
        test_objs = CATEGORIES["animal"][:3] + CATEGORIES["tool"][:2]
        r = run_injection_test(model, tokenizer, device, model_name, inject_vec, test_objs)
    else:
        r = None
    
    # Step4: Attention head贡献分析
    plog(f"  Step4: Attention head contribution at L{target_layer}...")
    head_results = {}
    
    try:
        attn_layer = layers_list[target_layer]
        n_heads = getattr(attn_layer.self_attn, 'num_heads', 
                         getattr(attn_layer.self_attn, 'n_heads', 8))
        head_dim = info.d_model // n_heads
        
        # 获取各head的输出hook
        for head_idx in [0, 10, 12, 13]:  # 关注的关键头
            if head_idx >= n_heads:
                continue
            
            # 简化测试: 比较math vs normal prompt在各head输出上的差异
            head_math_norms = []
            head_normal_norms = []
            
            for prompt in math_prompts[:2]:
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                cap_attn = {}
                
                def make_attn_hook(hi):
                    def hook_fn(module, inp, output):
                        if isinstance(output, tuple) and len(output) > 0:
                            attn_out = output[0].detach().float().cpu()
                            # 分离各head: [batch, seq, n_heads, head_dim]
                            batch, seq, d = attn_out.shape
                            if d == n_heads * head_dim:
                                attn_out_reshaped = attn_out.view(batch, seq, n_heads, head_dim)
                                cap_attn[f"head_{hi}"] = attn_out_reshaped[0, pos, hi].numpy()
                    return hook_fn
                
                h_attn = attn_layer.self_attn.register_forward_hook(make_attn_hook(head_idx))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_attn.remove()
                
                if f"head_{head_idx}" in cap_attn:
                    head_math_norms.append(np.linalg.norm(cap_attn[f"head_{head_idx}"]))
            
            for prompt in normal_prompts[:2]:
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                cap_attn = {}
                
                def make_attn_hook(hi):
                    def hook_fn(module, inp, output):
                        if isinstance(output, tuple) and len(output) > 0:
                            attn_out = output[0].detach().float().cpu()
                            batch, seq, d = attn_out.shape
                            if d == n_heads * head_dim:
                                attn_out_reshaped = attn_out.view(batch, seq, n_heads, head_dim)
                                cap_attn[f"head_{hi}"] = attn_out_reshaped[0, pos, hi].numpy()
                    return hook_fn
                
                h_attn = attn_layer.self_attn.register_forward_hook(make_attn_hook(head_idx))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_attn.remove()
                
                if f"head_{head_idx}" in cap_attn:
                    head_normal_norms.append(np.linalg.norm(cap_attn[f"head_{head_idx}"]))
            
            math_avg = np.mean(head_math_norms) if head_math_norms else 0
            normal_avg = np.mean(head_normal_norms) if head_normal_norms else 0
            diff_ratio = (math_avg - normal_avg) / (normal_avg + 1e-10)
            
            head_results[f"head_{head_idx}"] = {
                "math_norm": float(math_avg),
                "normal_norm": float(normal_avg),
                "diff_ratio": float(diff_ratio),
            }
            plog(f"    Head {head_idx}: math_norm={math_avg:.3f}, normal_norm={normal_avg:.3f}, "
                  f"diff_ratio={diff_ratio:.3f}")
    
    except Exception as e:
        plog(f"  Head analysis error: {e}")
        head_results = {"error": str(e)}
    
    return {
        "math_vs_normal_dcf_diff": {DCF_DIM_NAMES[i]: float(dcf_diff[i]) for i in range(len(DCF_DIM_NAMES))},
        "format_direction_injection": r,
        "head_contribution": head_results,
        "format_direction_norm": float(fmt_norm),
    }


# ==================== 主流程 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    plog(f"Phase 480: Category Boundary Residual Universality")
    plog(f"Model: {model_name}, Round: {round_num}")
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    plog(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    results = {
        "phase": 480,
        "model": model_name,
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "core_question": "Do category-specific boundary residuals exist universally across categories and models?",
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }
    
    # 运行实验
    t0 = time.time()
    
    # Exp1: 多类别specific方向构造 (ALL models) ★★★核心★★★
    try:
        results["exp1_multi_category_specific"] = exp1_multi_category_specific(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"  Exp1 ERROR: {e}")
        import traceback; traceback.print_exc()
        results["exp1_multi_category_specific"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp2: 类别边界残差属性画像 (Qwen3 only)
    if model_name == "qwen3":
        try:
            results["exp2_attribute_profiles"] = exp2_attribute_profiles(model, tokenizer, device, model_name)
        except Exception as e:
            plog(f"  Exp2 ERROR: {e}")
            import traceback; traceback.print_exc()
            results["exp2_attribute_profiles"] = {"error": str(e)}
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Exp3: 自然使用验证 (Qwen3 only)
    if model_name == "qwen3":
        try:
            results["exp3_natural_usage"] = exp3_natural_usage(model, tokenizer, device, model_name)
        except Exception as e:
            plog(f"  Exp3 ERROR: {e}")
            import traceback; traceback.print_exc()
            results["exp3_natural_usage"] = {"error": str(e)}
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Exp4: 反向注入测试 (Qwen3 only)
    if model_name == "qwen3":
        try:
            results["exp4_reverse_injection"] = exp4_reverse_injection(model, tokenizer, device, model_name)
        except Exception as e:
            plog(f"  Exp4 ERROR: {e}")
            import traceback; traceback.print_exc()
            results["exp4_reverse_injection"] = {"error": str(e)}
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Exp5: 跨模型复现 — 通过在GLM4/DS7B上运行Exp1实现
    # (已由Exp1覆盖, 不需单独写)
    
    # Exp6: DS7B格式覆盖电路行为闭环 (DS7B only)
    if model_name == "deepseek7b":
        try:
            results["exp6_ds7b_format_circuit"] = exp6_ds7b_format_circuit(model, tokenizer, device, model_name)
        except Exception as e:
            plog(f"  Exp6 ERROR: {e}")
            import traceback; traceback.print_exc()
            results["exp6_ds7b_format_circuit"] = {"error": str(e)}
        
        gc.collect()
        torch.cuda.empty_cache()
    
    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)
    plog(f"Total time: {elapsed:.1f}s")
    
    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase480_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    plog(f"Results saved to {out_path}")
    
    # 释放模型
    release_model(model)


if __name__ == "__main__":
    main()
