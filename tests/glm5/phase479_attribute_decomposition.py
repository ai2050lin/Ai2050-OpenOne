"""
Phase 479: 属性原子分解、fruit-specific残差验证与跨模型/跨语言读出接口
====================================================================

核心目标:
1. sweet-specific方向解耦 — fruit-specific是否等价于sweet-specific?
2. Gram-Schmidt顺序稳健性 — 去相关顺序是否影响结果?
3. 属性级神经元定位 — L30中哪些神经元写sweet/edible/plant_grown?
4. 关系槽位读出验证 — kind_of/eaten_as/grown_from是否读出不同成分?
5. GLM4 fruit-specific复现 — GLM4 L33是否有同样的解耦结构?
6. 跨语言语义簇与语言接口分离

实验:
1. Exp1: sweet-specific方向解耦 + fruit-vs-sweet等价性测试 (Qwen3)
2. Exp2: Gram-Schmidt顺序稳健性 + 正交化方法对比 (Qwen3)
3. Exp3: 属性级神经元写入器定位 (Qwen3)
4. Exp4: 关系槽位读出验证 — 5种关系下的簇/特异方向注入 (Qwen3)
5. Exp5: GLM4 fruit-specific解耦 (GLM4)
6. Exp6: 跨语言语义簇与语言接口分离 (Qwen3)

用法:
  python tests/glm5/phase479_attribute_decomposition.py qwen3 1
  python tests/glm5/phase479_attribute_decomposition.py glm4 1
  python tests/glm5/phase479_attribute_decomposition.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json, math
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS,
                          get_layer_weights)


def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 数据定义 ====================
CATEGORIES = {
    "fruit":    ["apple", "banana", "orange", "grape", "pear", "peach", "mango", "plum"],
    "animal":   ["dog", "cat", "horse", "lion", "bear", "rabbit"],
    "tool":     ["hammer", "knife", "wrench", "saw", "drill", "axe"],
    "vehicle":  ["car", "bus", "bicycle", "truck", "train", "boat"],
    "clothing": ["shirt", "dress", "hat", "coat", "sock", "glove"],
    "furniture":["chair", "table", "desk", "sofa", "bed", "shelf"],
}

FAMILY_WORDS_8D = {
    "fruit":    ["fruit", "produce", "crop", "berry"],
    "animal":   ["animal", "creature", "beast", "pet"],
    "tool":     ["tool", "implement", "device", "instrument"],
    "vehicle":  ["vehicle", "transport", "automobile", "car"],
    "clothing": ["clothing", "attire", "wear", "garment"],
    "furniture":["furniture", "furnishing", "fixture", "seat"],
    "food":     ["food", "meal", "dish", "snack"],
    "plant":    ["plant", "tree", "vegetation", "flora"],
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
}

ATTR_DIM_NAMES = list(ATTRIBUTE_WORDS.keys())

RELATION_TEMPLATES = {
    "kind_of":      "The {obj} is a kind of",
    "belongs_to":   "A {obj} belongs to the category",
    "eaten_as":     "{obj} is usually eaten as",
    "grown_from":   "{obj} is grown from a",
    "found_in":     "{obj} is usually found in",
}

TRANSLATION_TEMPLATES = {
    "en_kind_of":   "The {obj} is a kind of",
    "zh_kind_of":   "{obj}是一种",
    "en_eaten_as":  "The {obj} is usually eaten as",
    "zh_eaten_as":  "{obj}通常作为食物被",
    "en_grown_from":"The {obj} is grown from a",
    "zh_grown_from":"{obj}是从一种",
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


def compute_selectivity(dcf_delta, target_idx=0):
    target = abs(dcf_delta[target_idx])
    max_other = max(abs(dcf_delta[i]) for i in range(len(dcf_delta)) if i != target_idx)
    return target / (max_other + 0.01)


def get_attr_wu_direction(W_U, tokenizer, attr_name):
    """获取属性在W_U空间中的方向(多词平均)"""
    words = ATTRIBUTE_WORDS.get(attr_name, [attr_name])
    vecs = []
    for w in words:
        tid = find_token_id(tokenizer, w)
        if tid is not None and tid < W_U.shape[0]:
            vecs.append(W_U[tid])
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def get_test_layers(model_name):
    if model_name == "qwen3":
        return {"mid": 30, "last": 35}
    elif model_name == "glm4":
        return {"mid": 33, "last": 39}
    else:
        return {"mid": 24, "last": 27}


def get_category_residuals(model, tokenizer, device, model_name, categories=None, n_obj=4):
    """批量获取各类别在mid层的residual"""
    if categories is None:
        categories = CATEGORIES
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    target_layer = get_test_layers(model_name)["mid"]
    
    results = {}
    for cat_name, cat_objs in categories.items():
        resids = []
        for obj in cat_objs[:n_obj]:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
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


def get_residual_for_objects(model, tokenizer, device, model_name, objs, template_key="kind_of"):
    """获取一组对象在mid层的residual"""
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    target_layer = get_test_layers(model_name)["mid"]
    template = RELATION_TEMPLATES[template_key]
    
    resids = []
    for obj in objs:
        prompt = template.format(obj=obj)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap = {}
        h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h.remove()
        if "resid" in cap:
            resids.append(cap["resid"][0, pos].numpy())
    return np.mean(resids, axis=0) if resids else np.zeros(info.d_model)


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
        "fruit_dcf_delta": float(mean_dcf_delta[0]),
    }


# ==================== Exp1: sweet-specific方向解耦 ====================
def exp1_sweet_specific(model, tokenizer, device, model_name):
    """解耦sweet-specific方向, 测试fruit-specific是否等价于sweet-specific"""
    if model_name != "qwen3":
        return {"skipped": True, "reason": "only for qwen3"}
    
    plog(f"=== Exp1: Sweet-Specific Direction Decomposition ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)
    target_layer = get_test_layers(model_name)["mid"]
    
    # Step1: 获取类别residual方向
    plog(f"  Step1: Getting category residual directions...")
    cat_resids = get_category_residuals(model, tokenizer, device, model_name)
    plant_resid = get_residual_for_objects(model, tokenizer, device, model_name, ["tree", "flower", "grass", "bush"])
    food_resid = get_residual_for_objects(model, tokenizer, device, model_name, ["bread", "rice", "cheese", "pasta"])
    fruit_resid = cat_resids.get("fruit", np.zeros(info.d_model))
    
    # Step2: 正交化 — fruit_specific_v2 (Phase 478复现)
    plog(f"  Step2: Computing orthogonalized directions...")
    fruit_specific_v2 = gram_schmidt_orth(fruit_resid, [plant_resid, food_resid])
    
    # sweet W_U方向
    sweet_wu = get_attr_wu_direction(W_U, tokenizer, "sweet")
    edible_wu = get_attr_wu_direction(W_U, tokenizer, "edible")
    plant_grown_wu = get_attr_wu_direction(W_U, tokenizer, "plant_grown")
    juicy_wu = get_attr_wu_direction(W_U, tokenizer, "juicy")
    natural_wu = get_attr_wu_direction(W_U, tokenizer, "natural")
    
    # sweet_specific: 从sweet_wu中去fruit+plant+food residual方向
    sweet_specific = gram_schmidt_orth(sweet_wu, [fruit_resid, plant_resid, food_resid]) if sweet_wu is not None else None
    
    # fruit_no_sweet: 从fruit_specific中去sweet方向
    fruit_no_sweet = gram_schmidt_orth(fruit_specific_v2, [sweet_wu]) if sweet_wu is not None else None
    
    # Step3: 注入测试
    plog(f"  Step3: Injection tests...")
    
    test_directions = {}
    fs_norm = np.linalg.norm(fruit_specific_v2)
    if fs_norm < 1e-10:
        return {"error": "fruit_specific_v2 has zero norm"}
    
    for name, vec in [
        ("fruit_specific_v2", fruit_specific_v2),
        ("sweet_specific", sweet_specific),
        ("fruit_no_sweet", fruit_no_sweet),
        ("sweet_wu_raw", sweet_wu),
        ("juicy_wu", juicy_wu),
        ("natural_wu", natural_wu),
    ]:
        if vec is None:
            continue
        n = np.linalg.norm(vec)
        if n > 1e-10:
            test_directions[name] = vec / n * fs_norm  # 统一范数
    
    test_objs = CATEGORIES["animal"][:3] + CATEGORIES["tool"][:2]
    
    results = {}
    for dir_name, dir_vec in test_directions.items():
        plog(f"    Testing {dir_name}...")
        r = run_injection_test(model, tokenizer, device, model_name, dir_vec, test_objs)
        if r:
            results[dir_name] = r
            plog(f"      fruit_Δ={r['fruit_dcf_delta']:.3f}, sel={r['selectivity']:.2f}, "
                 f"sweet_Δ={r['attr_delta'].get('sweet', 0):.3f}")
    
    # Step4: 方向余弦对比
    plog(f"  Step4: Direction cosine comparison...")
    cosine_results = {}
    dir_list = list(test_directions.keys())
    for i in range(len(dir_list)):
        for j in range(i+1, len(dir_list)):
            v1 = test_directions[dir_list[i]]
            v2 = test_directions[dir_list[j]]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            cos = float(np.dot(v1, v2) / (n1 * n2 + 1e-10))
            cosine_results[f"{dir_list[i]}_vs_{dir_list[j]}"] = cos
    
    return {
        "injection_results": results,
        "cosine_comparison": cosine_results,
    }


# ==================== Exp2: Gram-Schmidt顺序稳健性 ====================
def exp2_gs_robustness(model, tokenizer, device, model_name):
    """测试Gram-Schmidt正交化顺序是否影响结果"""
    if model_name != "qwen3":
        return {"skipped": True, "reason": "only for qwen3"}
    
    plog(f"=== Exp2: Gram-Schmidt Order Robustness ({model_name}) ===")
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    
    cat_resids = get_category_residuals(model, tokenizer, device, model_name)
    plant_resid = get_residual_for_objects(model, tokenizer, device, model_name, ["tree", "flower", "grass", "bush"])
    food_resid = get_residual_for_objects(model, tokenizer, device, model_name, ["bread", "rice", "cheese", "pasta"])
    fruit_resid = cat_resids.get("fruit", np.zeros(info.d_model))
    
    # 方法1: 先plant后food
    fs_v1 = gram_schmidt_orth(fruit_resid, [plant_resid, food_resid])
    # 方法2: 先food后plant
    fs_v2 = gram_schmidt_orth(fruit_resid, [food_resid, plant_resid])
    # 方法3: QR子空间投影
    basis = np.array([plant_resid, food_resid])
    Q, _ = np.linalg.qr(basis.T)
    proj = Q @ (Q.T @ fruit_resid)
    fs_qr = fruit_resid - proj
    # 方法4: 去更多属性
    sweet_wu = get_attr_wu_direction(W_U, tokenizer, "sweet")
    natural_wu = get_attr_wu_direction(W_U, tokenizer, "natural")
    edible_wu = get_attr_wu_direction(W_U, tokenizer, "edible")
    fs_multi = gram_schmidt_orth(fruit_resid, [plant_resid, food_resid, natural_wu, edible_wu])
    
    methods = {
        "gs_plant_then_food": fs_v1,
        "gs_food_then_plant": fs_v2,
        "qr_subspace": fs_qr,
        "gs_multi_attr": fs_multi,
    }
    
    # 归一化
    ref_norm = np.linalg.norm(fs_v1)
    for k, v in methods.items():
        n = np.linalg.norm(v)
        if n > 1e-10:
            methods[k] = v / n * ref_norm
    
    test_objs = CATEGORIES["animal"][:3] + CATEGORIES["tool"][:2]
    
    results = {}
    for method_name, dir_vec in methods.items():
        plog(f"    Testing {method_name}...")
        r = run_injection_test(model, tokenizer, device, model_name, dir_vec, test_objs)
        if r:
            results[method_name] = r
            plog(f"      fruit_Δ={r['fruit_dcf_delta']:.3f}, sel={r['selectivity']:.2f}")
    
    # 方法间余弦
    inter_cos = {}
    names = list(methods.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            v1, v2 = methods[names[i]], methods[names[j]]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            cos = float(np.dot(v1, v2) / (n1 * n2 + 1e-10))
            inter_cos[f"{names[i]}_vs_{names[j]}"] = cos
    
    return {
        "method_comparison": results,
        "inter_method_cosine": inter_cos,
    }


# ==================== Exp3: 属性级神经元定位 ====================
def exp3_attribute_neurons(model, tokenizer, device, model_name):
    """在L30中分别找到对sweet/edible/plant_grown/fruit_specific贡献最大的神经元"""
    if model_name != "qwen3":
        return {"skipped": True, "reason": "only for qwen3"}
    
    plog(f"=== Exp3: Attribute-Level Neuron Localization ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)
    target_layer = get_test_layers(model_name)["mid"]
    
    # 获取W_down
    try:
        lw = get_layer_weights(layers_list[target_layer], info.d_model, info.mlp_type)
        W_down = lw.W_down
    except Exception:
        W_down = layers_list[target_layer].mlp.down_proj.weight.detach().float().cpu().numpy().T
    
    # 获取fruit-specific方向
    cat_resids = get_category_residuals(model, tokenizer, device, model_name)
    plant_resid = get_residual_for_objects(model, tokenizer, device, model_name, ["tree", "flower", "grass", "bush"])
    food_resid = get_residual_for_objects(model, tokenizer, device, model_name, ["bread", "rice", "cheese", "pasta"])
    fruit_resid = cat_resids.get("fruit", np.zeros(info.d_model))
    fruit_specific = gram_schmidt_orth(fruit_resid, [plant_resid, food_resid])
    
    # 属性W_U方向
    sweet_wu = get_attr_wu_direction(W_U, tokenizer, "sweet")
    edible_wu = get_attr_wu_direction(W_U, tokenizer, "edible")
    plant_grown_wu = get_attr_wu_direction(W_U, tokenizer, "plant_grown")
    
    target_directions = {
        "fruit_specific": fruit_specific,
        "sweet_wu": sweet_wu,
        "edible_wu": edible_wu,
        "plant_grown_wu": plant_grown_wu,
    }
    
    # 对每个方向, 计算神经元在fruit上下文中的加权贡献
    plog(f"  Computing neuron contributions...")
    neuron_rankings = {}
    n_obj = 4
    
    for dir_name, target_dir in target_directions.items():
        if target_dir is None:
            continue
        dir_norm = np.linalg.norm(target_dir)
        if dir_norm < 1e-10:
            continue
        dir_normed = target_dir / dir_norm
        
        neuron_scores = {}
        for obj in CATEGORIES["fruit"][:n_obj]:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap_mid = {}
            h_mid = layers_list[target_layer].mlp.down_proj.register_forward_hook(
                lambda m, i, o: cap_mid.update({"mid": i[0].detach().float().cpu()}) if isinstance(i, tuple) and len(i) > 0 else None)
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_mid.remove()
            
            if "mid" in cap_mid:
                mid_act = cap_mid["mid"][0, pos].numpy()
                for idx in range(min(len(mid_act), W_down.shape[1])):
                    write_vec = W_down[:, idx]
                    cos_val = np.dot(write_vec, dir_normed) / (np.linalg.norm(write_vec) + 1e-10)
                    weighted_contrib = abs(mid_act[idx]) * cos_val
                    if idx not in neuron_scores:
                        neuron_scores[idx] = 0.0
                    neuron_scores[idx] += weighted_contrib
        
        sorted_neurons = sorted(neuron_scores.items(), key=lambda x: -x[1])
        top50 = [int(n[0]) for n in sorted_neurons[:50]]
        neuron_rankings[dir_name] = {
            "top50_neurons": top50,
            "top10_scores": [float(n[1]) for n in sorted_neurons[:10]],
        }
        plog(f"    {dir_name}: top10 = {top50[:10]}")
    
    # 计算属性写入器之间神经元重叠
    plog(f"  Computing neuron overlap...")
    overlap_results = {}
    dir_names = list(neuron_rankings.keys())
    for i in range(len(dir_names)):
        for j in range(i+1, len(dir_names)):
            set_i = set(neuron_rankings[dir_names[i]]["top50_neurons"][:30])
            set_j = set(neuron_rankings[dir_names[j]]["top50_neurons"][:30])
            overlap = len(set_i & set_j)
            overlap_results[f"{dir_names[i]}_vs_{dir_names[j]}"] = {
                "overlap": overlap,
                "pct": float(overlap / 30.0),
            }
    
    # 注入测试: 各属性top30神经元
    plog(f"  Injection test: per-attribute neuron subset...")
    injection_results = {}
    
    for dir_name in list(neuron_rankings.keys()):
        top_neurons = neuron_rankings[dir_name]["top50_neurons"][:30]
        
        write_vecs = []
        for obj in CATEGORIES["fruit"][:3]:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap_mid = {}
            h_mid = layers_list[target_layer].mlp.down_proj.register_forward_hook(
                lambda m, i, o: cap_mid.update({"mid": i[0].detach().float().cpu()}) if isinstance(i, tuple) and len(i) > 0 else None)
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_mid.remove()
            if "mid" in cap_mid:
                mid_act = cap_mid["mid"][0, pos].numpy()
                wv = np.zeros(info.d_model)
                for idx in top_neurons:
                    if idx < W_down.shape[1]:
                        wv += W_down[:, idx] * mid_act[idx]
                write_vecs.append(wv)
        
        if not write_vecs:
            continue
        mean_wv = np.mean(write_vecs, axis=0)
        
        test_objs = CATEGORIES["animal"][:3]
        r = run_injection_test(model, tokenizer, device, model_name, mean_wv, test_objs)
        if r:
            injection_results[dir_name] = r
            plog(f"    {dir_name} neurons: fruit_Δ={r['fruit_dcf_delta']:.3f}, sel={r['selectivity']:.2f}")
    
    return {
        "neuron_rankings": {k: v["top50_neurons"][:20] for k, v in neuron_rankings.items()},
        "neuron_overlap": overlap_results,
        "injection_results": injection_results,
    }


# ==================== Exp4: 关系槽位读出验证 ====================
def exp4_relation_slot_readout(model, tokenizer, device, model_name):
    """
    测试同一语义簇写入器在不同关系模板下的读出是否不同
    
    假设: cluster_writer在kind_of下读出fruit, eaten_as下读出food, grown_from下读出plant
    """
    if model_name != "qwen3":
        return {"skipped": True, "reason": "only for qwen3"}
    
    plog(f"=== Exp4: Relation Slot Readout Verification ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)
    target_layer = get_test_layers(model_name)["mid"]
    
    # 获取类别residual方向
    cat_resids = get_category_residuals(model, tokenizer, device, model_name)
    plant_resid = get_residual_for_objects(model, tokenizer, device, model_name, ["tree", "flower", "grass", "bush"])
    food_resid = get_residual_for_objects(model, tokenizer, device, model_name, ["bread", "rice", "cheese", "pasta"])
    fruit_resid = cat_resids.get("fruit", np.zeros(info.d_model))
    
    # 解耦方向
    fruit_specific = gram_schmidt_orth(fruit_resid, [plant_resid, food_resid])
    
    # 注入方向: cluster_writer, fruit_specific, plant_resid, food_resid
    inject_directions = {
        "fruit_cluster": fruit_resid,
        "fruit_specific": fruit_specific,
        "plant_resid": plant_resid,
        "food_resid": food_resid,
    }
    
    # 测试关系模板
    relation_keys = ["kind_of", "eaten_as", "grown_from", "found_in"]
    
    # 测试对象: fruit对象在不同关系下
    fruit_objs = CATEGORIES["fruit"][:4]
    
    results = {}
    
    for dir_name, dir_vec in inject_directions.items():
        dir_results = {}
        inject_tensor = torch.tensor(dir_vec, dtype=torch.float32)
        last_layer = info.n_layers - 1
        
        for rel_key in relation_keys:
            template = RELATION_TEMPLATES[rel_key]
            
            dcf_before = []
            dcf_after = []
            
            for obj in fruit_objs:
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
            
            if dcf_before:
                mean_delta = np.mean(dcf_after, axis=0) - np.mean(dcf_before, axis=0)
                dir_results[rel_key] = {
                    "dcf_delta": {DCF_DIM_NAMES[i]: float(mean_delta[i]) for i in range(len(DCF_DIM_NAMES))},
                    "fruit_delta": float(mean_delta[0]),
                    "plant_delta": float(mean_delta[7]),
                    "food_delta": float(mean_delta[6]),
                }
                plog(f"    {dir_name} @ {rel_key}: fruit_Δ={mean_delta[0]:.3f}, "
                     f"plant_Δ={mean_delta[7]:.3f}, food_Δ={mean_delta[6]:.3f}")
        
        results[dir_name] = dir_results
    
    return results


# ==================== Exp5: GLM4 fruit-specific解耦 ====================
def exp5_glm4_fruit_specific(model, tokenizer, device, model_name):
    """在GLM4 L33层复现fruit-specific解耦"""
    if model_name != "glm4":
        return {"skipped": True, "reason": "only for glm4"}
    
    plog(f"=== Exp5: GLM4 Fruit-Specific Decomposition ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)
    target_layer = get_test_layers(model_name)["mid"]  # L33
    last_layer = info.n_layers - 1
    
    # 获取各类别residual (在L33层)
    plog(f"  Getting category residuals at L{target_layer}...")
    
    # fruit对象
    fruit_resids = []
    for obj in CATEGORIES["fruit"][:4]:
        prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap = {}
        h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h.remove()
        if "resid" in cap:
            fruit_resids.append(cap["resid"][0, pos].numpy())
    
    plant_resids = []
    for obj in ["tree", "flower", "grass", "bush"]:
        prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap = {}
        h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h.remove()
        if "resid" in cap:
            plant_resids.append(cap["resid"][0, pos].numpy())
    
    food_resids = []
    for obj in ["bread", "rice", "cheese", "pasta"]:
        prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap = {}
        h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h.remove()
        if "resid" in cap:
            food_resids.append(cap["resid"][0, pos].numpy())
    
    animal_resids = []
    for obj in CATEGORIES["animal"][:4]:
        prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap = {}
        h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h.remove()
        if "resid" in cap:
            animal_resids.append(cap["resid"][0, pos].numpy())
    
    fruit_resid = np.mean(fruit_resids, axis=0) if fruit_resids else np.zeros(info.d_model)
    plant_resid = np.mean(plant_resids, axis=0) if plant_resids else np.zeros(info.d_model)
    food_resid = np.mean(food_resids, axis=0) if food_resids else np.zeros(info.d_model)
    animal_resid = np.mean(animal_resids, axis=0) if animal_resids else np.zeros(info.d_model)
    
    # fruit-specific解耦
    plog(f"  Computing fruit_specific_v2...")
    fruit_specific_v2 = gram_schmidt_orth(fruit_resid, [plant_resid, food_resid])
    
    # 方向余弦分析
    plog(f"  Direction cosine analysis...")
    def safe_cos(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))
    
    dir_cosines = {
        "fruit_specific_vs_fruit": safe_cos(fruit_specific_v2, fruit_resid),
        "fruit_specific_vs_plant": safe_cos(fruit_specific_v2, plant_resid),
        "fruit_specific_vs_food": safe_cos(fruit_specific_v2, food_resid),
        "fruit_specific_vs_animal": safe_cos(fruit_specific_v2, animal_resid),
        "fruit_vs_plant": safe_cos(fruit_resid, plant_resid),
        "fruit_vs_food": safe_cos(fruit_resid, food_resid),
    }
    plog(f"    fruit_specific cos: fruit={dir_cosines['fruit_specific_vs_fruit']:.4f}, "
         f"plant={dir_cosines['fruit_specific_vs_plant']:.4f}, "
         f"food={dir_cosines['fruit_specific_vs_food']:.4f}")
    
    # 注入测试
    plog(f"  Injection tests...")
    test_objs = CATEGORIES["animal"][:3]
    
    inject_directions = {
        "fruit_cluster": fruit_resid,
        "fruit_specific_v2": fruit_specific_v2,
        "plant_resid": plant_resid,
        "food_resid": food_resid,
    }
    
    results = {}
    for dir_name, dir_vec in inject_directions.items():
        n = np.linalg.norm(dir_vec)
        if n < 1e-10:
            continue
        # 归一化到参考范数
        ref_norm = np.linalg.norm(fruit_specific_v2) if np.linalg.norm(fruit_specific_v2) > 1e-10 else 1.0
        inject_vec = dir_vec / n * ref_norm
        
        r = run_injection_test(model, tokenizer, device, model_name, inject_vec, test_objs)
        if r:
            results[dir_name] = r
            plog(f"    {dir_name}: fruit_Δ={r['fruit_dcf_delta']:.3f}, sel={r['selectivity']:.2f}")
    
    return {
        "direction_cosines": dir_cosines,
        "injection_results": results,
    }


# ==================== Exp6: 跨语言语义簇与语言接口分离 ====================
def exp6_cross_language_interface(model, tokenizer, device, model_name):
    """测试语义簇是否跨语言共享, 语言接口是否在不同模板下不同"""
    if model_name != "qwen3":
        return {"skipped": True, "reason": "only for qwen3"}
    
    plog(f"=== Exp6: Cross-Language Semantic Cluster & Language Interface ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)
    target_layer = get_test_layers(model_name)["mid"]
    last_layer = info.n_layers - 1
    
    # Step1: 获取英文和中文模板下的residual
    plog(f"  Step1: Getting residuals for EN/ZH templates...")
    
    en_zh_resids = {}
    # 英文模板
    for rel_key in ["kind_of", "eaten_as", "grown_from"]:
        en_key = f"en_{rel_key}"
        zh_key = f"zh_{rel_key}"
        en_template = TRANSLATION_TEMPLATES[en_key]
        zh_template = TRANSLATION_TEMPLATES[zh_key]
        
        for obj in CATEGORIES["fruit"][:3]:
            # EN
            en_prompt = en_template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, en_prompt)
            cap = {}
            h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap:
                if en_key not in en_zh_resids:
                    en_zh_resids[en_key] = []
                en_zh_resids[en_key].append(cap["resid"][0, pos].numpy())
            
            # ZH
            zh_prompt = zh_template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, zh_prompt)
            cap = {}
            h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap:
                if zh_key not in en_zh_resids:
                    en_zh_resids[zh_key] = []
                en_zh_resids[zh_key].append(cap["resid"][0, pos].numpy())
    
    # 计算mean residuals
    mean_resids = {}
    for k, vs in en_zh_resids.items():
        mean_resids[k] = np.mean(vs, axis=0)
    
    # Step2: 计算跨语言residual相似度
    plog(f"  Step2: Cross-language residual similarity...")
    lang_cosines = {}
    for rel_key in ["kind_of", "eaten_as", "grown_from"]:
        en_key = f"en_{rel_key}"
        zh_key = f"zh_{rel_key}"
        if en_key in mean_resids and zh_key in mean_resids:
            cos = safe_cos(mean_resids[en_key], mean_resids[zh_key])
            lang_cosines[f"en_zh_{rel_key}"] = cos
            plog(f"    cos(en_{rel_key}, zh_{rel_key}) = {cos:.4f}")
    
    # Step3: 注入测试 — 英文fruit writer在中文模板上的效果
    plog(f"  Step3: Fruit writer injection on EN/ZH templates...")
    
    cat_resids = get_category_residuals(model, tokenizer, device, model_name)
    plant_resid = get_residual_for_objects(model, tokenizer, device, model_name, ["tree", "flower", "grass", "bush"])
    food_resid = get_residual_for_objects(model, tokenizer, device, model_name, ["bread", "rice", "cheese", "pasta"])
    fruit_resid = cat_resids.get("fruit", np.zeros(info.d_model))
    fruit_specific = gram_schmidt_orth(fruit_resid, [plant_resid, food_resid])
    
    inject_directions = {
        "fruit_cluster": fruit_resid,
        "fruit_specific": fruit_specific,
    }
    
    test_templates = {
        "en_kind_of": TRANSLATION_TEMPLATES["en_kind_of"],
        "zh_kind_of": TRANSLATION_TEMPLATES["zh_kind_of"],
        "en_eaten_as": TRANSLATION_TEMPLATES["en_eaten_as"],
        "zh_eaten_as": TRANSLATION_TEMPLATES["zh_eaten_as"],
        "en_grown_from": TRANSLATION_TEMPLATES["en_grown_from"],
        "zh_grown_from": TRANSLATION_TEMPLATES["zh_grown_from"],
    }
    
    test_objs = CATEGORIES["animal"][:2]
    
    results = {}
    for dir_name, dir_vec in inject_directions.items():
        inject_tensor = torch.tensor(dir_vec, dtype=torch.float32)
        dir_results = {}
        
        for tmpl_name, tmpl_str in test_templates.items():
            dcf_before = []
            dcf_after = []
            
            for obj in test_objs:
                prompt = tmpl_str.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                
                cap_clean = {}
                h_clean = layers_list[last_layer].register_forward_hook(_make_capture_hook(cap_clean, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_clean.remove()
                if "resid" not in cap_clean:
                    continue
                clean_r = cap_clean["resid"][0, pos].numpy()
                dcf_before.append(logit_lens_dcf(clean_r, W_U, tokenizer))
                
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
            
            if dcf_before:
                mean_delta = np.mean(dcf_after, axis=0) - np.mean(dcf_before, axis=0)
                dir_results[tmpl_name] = {
                    "dcf_delta": {DCF_DIM_NAMES[i]: float(mean_delta[i]) for i in range(len(DCF_DIM_NAMES))},
                    "fruit_delta": float(mean_delta[0]),
                    "plant_delta": float(mean_delta[7]),
                    "food_delta": float(mean_delta[6]),
                }
                plog(f"    {dir_name} @ {tmpl_name}: fruit_Δ={mean_delta[0]:.3f}, "
                     f"plant_Δ={mean_delta[7]:.3f}")
        
        results[dir_name] = dir_results
    
    return {
        "language_cosines": lang_cosines,
        "injection_results": results,
    }


def safe_cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ==================== 主流程 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    plog(f"Phase 479: Attribute Decomposition & Cross-Model/Language Readout")
    plog(f"Model: {model_name}, Round: {round_num}")
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    plog(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    results = {
        "phase": 479,
        "model": model_name,
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "theory": "Attribute Atom Decomposition, fruit-specific Verification & Cross-Language Readout",
        "core_question": "Is fruit-specific equivalent to sweet-specific? How do relation slots read out from semantic clusters?",
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }
    
    # 运行实验
    t0 = time.time()
    
    # Exp1: sweet-specific解耦 (Qwen3 only)
    try:
        results["exp1_sweet_specific"] = exp1_sweet_specific(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"  Exp1 ERROR: {e}")
        import traceback; traceback.print_exc()
        results["exp1_sweet_specific"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp2: GS顺序稳健性 (Qwen3 only)
    try:
        results["exp2_gs_robustness"] = exp2_gs_robustness(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"  Exp2 ERROR: {e}")
        import traceback; traceback.print_exc()
        results["exp2_gs_robustness"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp3: 属性级神经元定位 (Qwen3 only)
    try:
        results["exp3_attribute_neurons"] = exp3_attribute_neurons(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"  Exp3 ERROR: {e}")
        import traceback; traceback.print_exc()
        results["exp3_attribute_neurons"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp4: 关系槽位读出 (Qwen3 only)
    try:
        results["exp4_relation_readout"] = exp4_relation_slot_readout(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"  Exp4 ERROR: {e}")
        import traceback; traceback.print_exc()
        results["exp4_relation_readout"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp5: GLM4 fruit-specific (GLM4 only)
    try:
        results["exp5_glm4_fruit_specific"] = exp5_glm4_fruit_specific(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"  Exp5 ERROR: {e}")
        import traceback; traceback.print_exc()
        results["exp5_glm4_fruit_specific"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp6: 跨语言接口 (Qwen3 only)
    try:
        results["exp6_cross_language"] = exp6_cross_language_interface(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"  Exp6 ERROR: {e}")
        import traceback; traceback.print_exc()
        results["exp6_cross_language"] = {"error": str(e)}
    
    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)
    plog(f"Total time: {elapsed:.1f}s")
    
    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase479_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    plog(f"Results saved to {out_path}")
    
    # 释放模型
    release_model(model)


if __name__ == "__main__":
    main()
