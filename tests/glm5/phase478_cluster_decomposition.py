"""
Phase 478: 语义簇分解、类别特异控制与格式覆盖电路闭环
========================================================
核心目标:
1. 构造Attribute DCF, 解析fruit writer到底激活了哪些属性
2. 从fruit-plant-food簇中解耦fruit-specific方向
3. 验证解耦后的fruit-specific方向是否具有类别选择性
4. 跨模型L30功能定位(GLM4的类别特异写入器在哪层?)
5. DS7B格式覆盖头组合消融(Head 12+13+10)
6. 翻译重构预实验(语义簇是否跨语言共享?)

实验:
1. Exp1: Attribute DCF构造 + fruit writer属性画像 (Qwen3)
2. Exp2: 语义簇分解 - fruit-specific方向 (Qwen3)
3. Exp3: 解耦方向注入测试 (Qwen3)
4. Exp4: GLM4类别特异写入层定位 (GLM4)
5. Exp5: DS7B Head组合消融 (DS7B)
6. Exp6: 翻译重构预实验 (Qwen3)

用法:
  python tests/glm5/phase478_cluster_decomposition.py qwen3 1
  python tests/glm5/phase478_cluster_decomposition.py glm4 1
  python tests/glm5/phase478_cluster_decomposition.py deepseek7b 1
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

# 8D类别DCF维度词汇
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

# 属性DCF词汇 — 每个属性用一组相关词
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
}

ATTR_DIM_NAMES = list(ATTRIBUTE_WORDS.keys())

# 多模板
RELATION_TEMPLATES = {
    "kind_of":      "The {obj} is a kind of",
    "belongs_to":   "A {obj} belongs to the category",
    "classified_as":"{obj} is classified as",
    "eaten_as":     "{obj} is usually eaten as",
    "grown_from":   "{obj} is grown from a",
    "found_in":     "{obj} is usually found in",
}

# 翻译相关
TRANSLATION_TEMPLATES = {
    "en_kind_of":   "The {obj} is a kind of",
    "zh_kind_of":   "{obj}是一种",  # 用拼音测试
    "en_translate": "Translate to English: {obj}是一种水果. The answer is",
    "zh_translate": "Translate to Chinese: The {obj} is a fruit. The answer is",
}

FORMAT_TOKENS = [
    "(", ")", "[", "]", "{", "}", "<", ">", ",", ".", ":", ";", "!", "?",
    "-", "=", "+", "*", "/", "\\", "|", "&", "^", "%", "$", "#", "@", "~",
    "`", "'", "\"", "...", "..", "--", "---",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "to", "for", "with", "on", "at", "by", "from", "as",
    "therefore", "because", "since", "thus", "hence", "so", "consequently",
]

SEMANTIC_TOKENS = [
    "fruit", "apple", "banana", "orange", "grape", "pear", "peach", "produce", "crop", "berry",
    "animal", "dog", "cat", "horse", "lion", "bear", "rabbit", "creature", "beast", "pet",
    "tool", "hammer", "knife", "wrench", "saw", "drill", "axe", "implement", "device", "instrument",
    "vehicle", "car", "bus", "bicycle", "truck", "train", "boat", "transport", "automobile",
    "clothing", "shirt", "dress", "hat", "coat", "sock", "glove", "attire", "wear", "garment",
    "furniture", "chair", "table", "desk", "sofa", "bed", "shelf", "furnishing", "fixture",
    "food", "plant", "tree", "flower", "grass", "leaf", "root", "seed",
]


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
    n_loaded = len(layers_list)
    plog(f"  Loaded {n_loaded} transformer layers")

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
        plog(f"  Layer distribution: {gpu_layers} GPU + {cpu_layers} CPU (total {n_loaded})")
        if cpu_layers > 0:
            gpu_lids = [int(lid) for lid, dev in layer_devices.items() if 'cuda' in dev]
            cpu_lids = [int(lid) for lid, dev in layer_devices.items() if 'cpu' in dev]
            if gpu_lids:
                plog(f"  Last GPU layer: L{max(gpu_lids)}")
            if cpu_lids:
                plog(f"  Last CPU layer: L{max(cpu_lids)}")

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


def compute_dcf_from_logits(logits, tokenizer, dim_dict, dim_names):
    """计算DCF向量（通用版，支持8D类别或属性维度）"""
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
    """logit lens获取DCF向量"""
    if dim_dict is None:
        dim_dict = FAMILY_WORDS_8D
    if dim_names is None:
        dim_names = DCF_DIM_NAMES
    logits = resid @ W_U.T
    return compute_dcf_from_logits(logits, tokenizer, dim_dict, dim_names)


def _make_capture_hook(store_dict, key):
    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            store_dict[key] = output[0].detach().float().cpu()
        else:
            store_dict[key] = output.detach().float().cpu()
    return hook_fn


def compute_entropy(logits):
    max_l = np.max(logits)
    exp_l = np.exp(logits - max_l)
    probs = exp_l / np.sum(exp_l)
    probs = probs[probs > 1e-12]
    return -float(np.sum(probs * np.log(probs)))


def get_prompt_ids(tokenizer, device, prompt, max_len=128):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = attention_mask.sum().item()
    pos = seq_len - 1
    return input_ids, attention_mask, pos


def get_test_layers(model_name):
    if model_name == "qwen3":
        return {"mid": 30, "last": 35, "sample": [24, 27, 30, 33, 35]}
    elif model_name == "glm4":
        return {"mid": 30, "last": 37, "sample": [24, 27, 30, 33, 35, 37]}
    else:  # deepseek7b
        return {"mid": 24, "last": 26, "sample": [24, 25, 26]}


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


def find_fruit_writer_neurons(model, tokenizer, device, model_name, target_layer=None):
    """找到fruit正贡献神经元和对应write vector"""
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)
    
    if target_layer is None:
        target_layer = get_test_layers(model_name)["mid"]
    
    lw = get_layer_weights(layers_list[target_layer], info.d_model, info.mlp_type)
    W_down = lw.W_down
    if W_down is None:
        W_down = layers_list[target_layer].mlp.down_proj.weight.detach().float().cpu().numpy().T

    n_obj = 4
    
    # 找fruit正贡献神经元
    fruit_contributions = {}
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
                if idx not in fruit_contributions:
                    fruit_contributions[idx] = 0.0
                write_vec = W_down[:, idx] * mid_act[idx]
                logits = write_vec @ W_U.T
                dcf_dim0 = 0.0
                for w in FAMILY_WORDS_8D["fruit"]:
                    tid = find_token_id(tokenizer, w)
                    if tid is not None and tid < len(logits):
                        dcf_dim0 += float(logits[tid])
                dcf_dim0 /= len(FAMILY_WORDS_8D["fruit"])
                fruit_contributions[idx] += dcf_dim0

    sorted_neurons = sorted(fruit_contributions.items(), key=lambda x: -x[1])
    top20_pos = [int(n[0]) for n in sorted_neurons[:20]]
    
    # 计算mean write vector
    write_vectors = []
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
            wv = np.zeros(info.d_model)
            for idx in top20_pos:
                if idx < W_down.shape[1]:
                    wv += W_down[:, idx] * mid_act[idx]
            write_vectors.append(wv)

    mean_write_vec = np.mean(write_vectors, axis=0) if write_vectors else np.zeros(info.d_model)
    
    # 同时找plant和food的write vector用于解耦
    # Plant write vector
    plant_write_vectors = []
    for obj in ["tree", "flower", "grass", "bush"]:
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
            for idx in top20_pos:
                if idx < W_down.shape[1]:
                    wv += W_down[:, idx] * mid_act[idx]
            plant_write_vectors.append(wv)
    
    plant_write_vec = np.mean(plant_write_vectors, axis=0) if plant_write_vectors else np.zeros(info.d_model)
    
    return {
        "top20_neurons": top20_pos,
        "mean_write_vec": mean_write_vec,
        "plant_write_vec": plant_write_vec,
        "write_vec_norm": float(np.linalg.norm(mean_write_vec)),
        "target_layer": target_layer,
    }


# ==================== Exp1: Attribute DCF + Fruit Writer属性画像 ====================
def exp1_attribute_dcf(model, tokenizer, device, model_name):
    """
    构造Attribute DCF, 分析fruit writer到底激活了哪些属性
    
    步骤:
    1. 定义12个属性的词汇表
    2. 对每个对象计算attribute DCF向量
    3. 注入fruit writer后看attribute DCF变化
    4. 生成fruit writer的完整属性画像
    """
    if model_name != "qwen3":
        plog(f"  Exp1 skipped (only for qwen3)")
        return {"skipped": True, "reason": "only for qwen3"}

    plog(f"=== Exp1: Attribute DCF + Fruit Writer Profile ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    # Step1: 验证Attribute DCF — 对不同类别对象计算attribute profile
    plog(f"  Step1: Computing attribute profiles for category exemplars...")
    
    category_exemplars = {
        "fruit": ["apple", "banana", "orange"],
        "animal": ["dog", "cat", "horse"],
        "tool": ["hammer", "knife", "wrench"],
        "vehicle": ["car", "bus", "bicycle"],
        "plant": ["tree", "flower", "grass"],
        "food": ["bread", "rice", "cheese"],
        "furniture": ["chair", "table", "desk"],
    }
    
    category_attr_profiles = {}
    for cat, objs in category_exemplars.items():
        attr_vectors = []
        for obj in objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap = {}
            h = layers_list[info.n_layers - 1].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap:
                attr_v = logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer,
                                        ATTRIBUTE_WORDS, ATTR_DIM_NAMES)
                attr_vectors.append(attr_v)
        
        if attr_vectors:
            category_attr_profiles[cat] = {
                "mean_attr": {ATTR_DIM_NAMES[i]: float(np.mean(attr_vectors, axis=0)[i]) 
                              for i in range(len(ATTR_DIM_NAMES))},
                "attr_vector": np.mean(attr_vectors, axis=0),
            }
            plog(f"    {cat}: edible={category_attr_profiles[cat]['mean_attr']['edible']:.2f}, "
                 f"plant_grown={category_attr_profiles[cat]['mean_attr']['plant_grown']:.2f}, "
                 f"sweet={category_attr_profiles[cat]['mean_attr']['sweet']:.2f}")

    # Step2: 注入fruit writer, 看attribute DCF变化
    plog(f"  Step2: Fruit writer attribute profile (injection test)...")
    
    writer_info = find_fruit_writer_neurons(model, tokenizer, device, model_name)
    mean_write_vec = writer_info["mean_write_vec"]
    
    inject_targets = {
        "animal": CATEGORIES["animal"][:3],
        "tool": CATEGORIES["tool"][:3],
        "vehicle": CATEGORIES["vehicle"][:3],
    }
    
    injection_attr_results = {}
    for amp in [1.0]:
        inject_vec = mean_write_vec * amp
        inject_tensor = torch.tensor(inject_vec, dtype=torch.float32)
        target_layer = writer_info["target_layer"]
        
        for target_cat, objs in inject_targets.items():
            attr_before_list = []
            attr_after_list = []
            cat8d_before = []
            cat8d_after = []
            
            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                
                # Clean
                cap_clean = {}
                h_clean = layers_list[info.n_layers - 1].register_forward_hook(
                    _make_capture_hook(cap_clean, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_clean.remove()
                
                if "resid" not in cap_clean:
                    continue
                
                clean_r = cap_clean["resid"][0, pos].numpy()
                attr_before = logit_lens_dcf(clean_r, W_U, tokenizer, ATTRIBUTE_WORDS, ATTR_DIM_NAMES)
                dcf8d_before = logit_lens_dcf(clean_r, W_U, tokenizer)
                attr_before_list.append(attr_before)
                cat8d_before.append(dcf8d_before)
                
                # Injected
                cap_pert = {}
                h_pert = layers_list[info.n_layers - 1].register_forward_hook(
                    _make_capture_hook(cap_pert, "resid"))
                h_inj = layers_list[target_layer].register_forward_hook(make_inject_hook(inject_tensor, pos))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_pert.remove()
                h_inj.remove()
                
                if "resid" in cap_pert:
                    pert_r = cap_pert["resid"][0, pos].numpy()
                    attr_after = logit_lens_dcf(pert_r, W_U, tokenizer, ATTRIBUTE_WORDS, ATTR_DIM_NAMES)
                    dcf8d_after = logit_lens_dcf(pert_r, W_U, tokenizer)
                    attr_after_list.append(attr_after)
                    cat8d_after.append(dcf8d_after)
            
            if attr_before_list:
                mean_attr_before = np.mean(attr_before_list, axis=0)
                mean_attr_after = np.mean(attr_after_list, axis=0)
                mean_attr_delta = mean_attr_after - mean_attr_before
                mean_8d_delta = np.mean(cat8d_after, axis=0) - np.mean(cat8d_before, axis=0)
                
                injection_attr_results[target_cat] = {
                    "attr_delta": {ATTR_DIM_NAMES[i]: float(mean_attr_delta[i]) for i in range(len(ATTR_DIM_NAMES))},
                    "cat8d_delta": {DCF_DIM_NAMES[i]: float(mean_8d_delta[i]) for i in range(len(DCF_DIM_NAMES))},
                }
                
                # 找top属性变化
                sorted_attrs = sorted(enumerate(mean_attr_delta), key=lambda x: -abs(x[1]))
                top3 = [(ATTR_DIM_NAMES[i], float(v)) for i, v in sorted_attrs[:3]]
                plog(f"    {target_cat} attr top3: {top3}")

    results = {
        "category_attr_profiles": {cat: v["mean_attr"] for cat, v in category_attr_profiles.items()},
        "injection_attr_results": injection_attr_results,
        "top20_neurons": writer_info["top20_neurons"],
    }
    return results


# ==================== Exp2: 语义簇分解 - fruit-specific方向 ====================
def exp2_cluster_decomposition(model, tokenizer, device, model_name):
    """
    从fruit-plant-food语义簇中解耦fruit-specific方向
    
    方法:
    1. 获取fruit, plant, food三个类别方向的8D DCF投影
    2. 用正交化方法去除plant/food共享成分
    3. 得到fruit-specific残差方向
    4. 验证该方向的类别选择性
    """
    if model_name != "qwen3":
        plog(f"  Exp2 skipped (only for qwen3)")
        return {"skipped": True, "reason": "only for qwen3"}

    plog(f"=== Exp2: Cluster Decomposition ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    # Step1: 获取各类别在L30层的mean residual
    plog(f"  Step1: Computing category mean residuals at L30...")
    target_layer = 30
    
    category_mean_resids = {}
    for cat in ["fruit", "plant", "food"]:
        if cat == "fruit":
            objs = CATEGORIES["fruit"][:4]
        elif cat == "plant":
            objs = ["tree", "flower", "grass", "bush"]
        elif cat == "food":
            objs = ["bread", "rice", "cheese", "pasta"]
        else:
            continue
            
        resids = []
        for obj in objs:
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
            category_mean_resids[cat] = np.mean(resids, axis=0)
    
    # Step2: 计算fruit vs non-fruit方向
    plog(f"  Step2: Computing category directions...")
    
    # fruit方向: mean(fruit) - mean(nonfruit)
    nonfruit_resids = []
    for cat in ["animal", "tool", "vehicle"]:
        objs = CATEGORIES[cat][:4]
        for obj in objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap = {}
            h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap:
                nonfruit_resids.append(cap["resid"][0, pos].numpy())
    
    nonfruit_mean = np.mean(nonfruit_resids, axis=0) if nonfruit_resids else np.zeros(info.d_model)
    
    # 原始fruit方向
    fruit_direction = category_mean_resids.get("fruit", np.zeros(info.d_model)) - nonfruit_mean
    fruit_dir_norm = np.linalg.norm(fruit_direction)
    if fruit_dir_norm > 0:
        fruit_direction = fruit_direction / fruit_dir_norm
    
    # plant方向
    plant_direction = category_mean_resids.get("plant", np.zeros(info.d_model)) - nonfruit_mean
    plant_dir_norm = np.linalg.norm(plant_direction)
    if plant_dir_norm > 0:
        plant_direction = plant_direction / plant_dir_norm
    
    # food方向
    food_direction = category_mean_resids.get("food", np.zeros(info.d_model)) - nonfruit_mean
    food_dir_norm = np.linalg.norm(food_direction)
    if food_dir_norm > 0:
        food_direction = food_direction / food_dir_norm
    
    # Step3: 正交化 — 从fruit方向去除plant/food共享成分
    plog(f"  Step3: Orthogonalizing fruit-specific direction...")
    
    # 方法1: Gram-Schmidt — fruit方向减去在plant方向上的投影
    fruit_proj_plant = np.dot(fruit_direction, plant_direction) * plant_direction
    fruit_minus_plant = fruit_direction - fruit_proj_plant
    fmp_norm = np.linalg.norm(fruit_minus_plant)
    if fmp_norm > 1e-10:
        fruit_specific_v1 = fruit_minus_plant / fmp_norm
    else:
        fruit_specific_v1 = np.zeros(info.d_model)
    
    # 方法2: 进一步去除food成分
    fruit_proj_food = np.dot(fruit_minus_plant, food_direction) * food_direction
    fruit_minus_plant_food = fruit_minus_plant - fruit_proj_food
    fmpf_norm = np.linalg.norm(fruit_minus_plant_food)
    if fmpf_norm > 1e-10:
        fruit_specific_v2 = fruit_minus_plant_food / fmpf_norm
    else:
        fruit_specific_v2 = np.zeros(info.d_model)
    
    # Step4: 验证各方向的选择性
    plog(f"  Step4: Validating direction selectivity...")
    
    test_objects = {
        "fruit": CATEGORIES["fruit"][:4],
        "animal": CATEGORIES["animal"][:3],
        "tool": CATEGORIES["tool"][:3],
        "vehicle": CATEGORIES["vehicle"][:3],
        "plant": ["tree", "flower", "grass"],
        "food": ["bread", "rice", "cheese"],
    }
    
    direction_names = ["fruit_raw", "fruit_specific_v1", "fruit_specific_v2", "plant", "food"]
    direction_vecs = [fruit_direction, fruit_specific_v1, fruit_specific_v2, plant_direction, food_direction]
    
    direction_selectivity = {}
    for dname, dvec in zip(direction_names, direction_vecs):
        if np.linalg.norm(dvec) < 1e-10:
            direction_selectivity[dname] = {"error": "zero norm"}
            continue
            
        cat_cos_sims = {}
        for cat, objs in test_objects.items():
            cos_sims = []
            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                cap = {}
                h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                if "resid" in cap:
                    r = cap["resid"][0, pos].numpy()
                    r_centered = r - nonfruit_mean
                    cos = float(np.dot(r_centered, dvec) / (np.linalg.norm(r_centered) * np.linalg.norm(dvec) + 1e-12))
                    cos_sims.append(cos)
            cat_cos_sims[cat] = float(np.mean(cos_sims)) if cos_sims else 0.0
        
        # 选择性 = fruit_cos / max(other_cos)
        fruit_cos = cat_cos_sims.get("fruit", 0.0)
        max_other_cos = max(abs(v) for k, v in cat_cos_sims.items() if k != "fruit")
        selectivity = abs(fruit_cos) / (max_other_cos + 0.01)
        
        direction_selectivity[dname] = {
            "category_cos_sims": cat_cos_sims,
            "selectivity": float(selectivity),
            "fruit_cos": float(fruit_cos),
        }
        plog(f"    {dname}: fruit_cos={fruit_cos:.4f}, selectivity={selectivity:.2f}")
    
    # Step5: 各方向之间的余弦相似度
    plog(f"  Step5: Inter-direction cosine similarities...")
    inter_cos = {}
    for i, (n1, v1) in enumerate(zip(direction_names, direction_vecs)):
        for j, (n2, v2) in enumerate(zip(direction_names, direction_vecs)):
            if j > i:
                cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
                inter_cos[f"{n1}_vs_{n2}"] = cos
    
    results = {
        "direction_selectivity": direction_selectivity,
        "inter_direction_cos": inter_cos,
        "fruit_dir_norm": float(fruit_dir_norm),
        "fruit_specific_v1_norm": float(fmp_norm),
        "fruit_specific_v2_norm": float(fmpf_norm),
    }
    return results


# ==================== Exp3: 解耦方向注入测试 ====================
def exp3_decomposed_injection(model, tokenizer, device, model_name):
    """
    分别注入:
    - 原始fruit write vector (簇级)
    - fruit-specific方向 (解耦后)
    - plant方向
    - food方向
    
    比较它们的8D DCF选择性
    """
    if model_name != "qwen3":
        plog(f"  Exp3 skipped (only for qwen3)")
        return {"skipped": True, "reason": "only for qwen3"}

    plog(f"=== Exp3: Decomposed Direction Injection ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)
    
    target_layer = 30
    
    # Step1: 获取各类别方向(与Exp2相同的计算)
    plog(f"  Step1: Computing decomposed directions...")
    
    category_mean_resids = {}
    for cat, objs_list in [("fruit", CATEGORIES["fruit"][:4]),
                            ("plant", ["tree", "flower", "grass", "bush"]),
                            ("food", ["bread", "rice", "cheese", "pasta"])]:
        resids = []
        for obj in objs_list:
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
            category_mean_resids[cat] = np.mean(resids, axis=0)
    
    nonfruit_resids = []
    for cat in ["animal", "tool", "vehicle"]:
        for obj in CATEGORIES[cat][:3]:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap = {}
            h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap:
                nonfruit_resids.append(cap["resid"][0, pos].numpy())
    
    nonfruit_mean = np.mean(nonfruit_resids, axis=0) if nonfruit_resids else np.zeros(info.d_model)
    
    # 原始fruit write vector (从fruit writer获得)
    writer_info = find_fruit_writer_neurons(model, tokenizer, device, model_name, target_layer)
    cluster_write_vec = writer_info["mean_write_vec"]
    
    # fruit方向 (残差空间)
    fruit_resid_dir = category_mean_resids.get("fruit", np.zeros(info.d_model)) - nonfruit_mean
    fruit_resid_norm = np.linalg.norm(fruit_resid_dir)
    
    # plant方向
    plant_resid_dir = category_mean_resids.get("plant", np.zeros(info.d_model)) - nonfruit_mean
    plant_resid_norm = np.linalg.norm(plant_resid_dir)
    
    # food方向
    food_resid_dir = category_mean_resids.get("food", np.zeros(info.d_model)) - nonfruit_mean
    food_resid_norm = np.linalg.norm(food_resid_dir)
    
    # fruit-specific: fruit - proj(fruit, plant) - proj(fruit, food)
    if fruit_resid_norm > 0:
        fruit_unit = fruit_resid_dir / fruit_resid_norm
    else:
        fruit_unit = np.zeros(info.d_model)
    if plant_resid_norm > 0:
        plant_unit = plant_resid_dir / plant_resid_norm
    else:
        plant_unit = np.zeros(info.d_model)
    if food_resid_norm > 0:
        food_unit = food_resid_dir / food_resid_norm
    else:
        food_unit = np.zeros(info.d_model)
    
    # 去除plant和food投影
    fruit_specific = fruit_unit - np.dot(fruit_unit, plant_unit) * plant_unit
    fruit_specific = fruit_specific - np.dot(fruit_specific, food_unit) * food_unit
    fs_norm = np.linalg.norm(fruit_specific)
    if fs_norm > 1e-10:
        fruit_specific = fruit_specific / fs_norm
    else:
        fruit_specific = np.zeros(info.d_model)
        fs_norm = 0.0
    
    plog(f"  Direction norms: cluster={np.linalg.norm(cluster_write_vec):.2f}, "
         f"fruit_resid={fruit_resid_norm:.2f}, plant={plant_resid_norm:.2f}, "
         f"food={food_resid_norm:.2f}, fruit_specific={fs_norm:.4f}")
    
    # Step2: 注入测试
    plog(f"  Step2: Injection test with different directions...")
    
    # 所有方向统一到cluster write vec的范数
    target_norm = np.linalg.norm(cluster_write_vec)
    
    injection_directions = {
        "cluster_writer": cluster_write_vec,
    }
    
    # 只添加非零方向
    if fruit_resid_norm > 1e-10:
        injection_directions["fruit_resid"] = fruit_unit * target_norm
    if plant_resid_norm > 1e-10:
        injection_directions["plant_resid"] = plant_unit * target_norm
    if food_resid_norm > 1e-10:
        injection_directions["food_resid"] = food_unit * target_norm
    if fs_norm > 1e-10:
        injection_directions["fruit_specific"] = fruit_specific * target_norm
    
    test_objects = {
        "animal": CATEGORIES["animal"][:3],
        "tool": CATEGORIES["tool"][:3],
    }
    
    results = {}
    
    for dname, dvec in injection_directions.items():
        if np.linalg.norm(dvec) < 1e-10:
            continue
        
        inject_tensor = torch.tensor(dvec, dtype=torch.float32)
        
        dcf_deltas = []
        attr_deltas = []
        
        for cat, objs in test_objects.items():
            cat_deltas = []
            cat_attr_deltas = []
            
            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                
                # Clean
                cap_clean = {}
                h_clean = layers_list[info.n_layers - 1].register_forward_hook(
                    _make_capture_hook(cap_clean, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_clean.remove()
                
                if "resid" not in cap_clean:
                    continue
                
                clean_r = cap_clean["resid"][0, pos].numpy()
                dcf_before = logit_lens_dcf(clean_r, W_U, tokenizer)
                attr_before = logit_lens_dcf(clean_r, W_U, tokenizer, ATTRIBUTE_WORDS, ATTR_DIM_NAMES)
                
                # Injected
                cap_pert = {}
                h_pert = layers_list[info.n_layers - 1].register_forward_hook(
                    _make_capture_hook(cap_pert, "resid"))
                h_inj = layers_list[target_layer].register_forward_hook(make_inject_hook(inject_tensor, pos))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_pert.remove()
                h_inj.remove()
                
                if "resid" in cap_pert:
                    pert_r = cap_pert["resid"][0, pos].numpy()
                    dcf_after = logit_lens_dcf(pert_r, W_U, tokenizer)
                    attr_after = logit_lens_dcf(pert_r, W_U, tokenizer, ATTRIBUTE_WORDS, ATTR_DIM_NAMES)
                    cat_deltas.append(dcf_after - dcf_before)
                    cat_attr_deltas.append(attr_after - attr_before)
            
            if cat_deltas:
                mean_dcf_delta = np.mean(cat_deltas, axis=0)
                mean_attr_delta = np.mean(cat_attr_deltas, axis=0)
                
                # 选择性: fruit_delta / max(|other_delta|)
                fruit_d = mean_dcf_delta[0]
                max_other = max(abs(mean_dcf_delta[i]) for i in range(1, len(DCF_DIM_NAMES)))
                selectivity = abs(fruit_d) / (max_other + 0.01)
                
                # 属性选择性: edible_delta vs plant_grown_delta vs sweet_delta
                edible_d = mean_attr_delta[ATTR_DIM_NAMES.index("edible")]
                plant_grown_d = mean_attr_delta[ATTR_DIM_NAMES.index("plant_grown")]
                sweet_d = mean_attr_delta[ATTR_DIM_NAMES.index("sweet")]
                
                dcf_deltas.append(mean_dcf_delta)
                attr_deltas.append(mean_attr_delta)
        
        if dcf_deltas:
            overall_dcf_delta = np.mean(dcf_deltas, axis=0)
            overall_attr_delta = np.mean(attr_deltas, axis=0)
            
            fruit_d = overall_dcf_delta[0]
            max_other = max(abs(overall_dcf_delta[i]) for i in range(1, len(DCF_DIM_NAMES)))
            selectivity = abs(fruit_d) / (max_other + 0.01)
            
            results[dname] = {
                "dcf_delta": {DCF_DIM_NAMES[i]: float(overall_dcf_delta[i]) for i in range(len(DCF_DIM_NAMES))},
                "attr_delta": {ATTR_DIM_NAMES[i]: float(overall_attr_delta[i]) for i in range(len(ATTR_DIM_NAMES))},
                "fruit_dcf_delta": float(fruit_d),
                "selectivity": float(selectivity),
            }
            plog(f"    {dname}: fruit_Δ={fruit_d:.3f}, selectivity={selectivity:.2f}, "
                 f"edible_Δ={overall_attr_delta[0]:.3f}, plant_grown_Δ={overall_attr_delta[1]:.3f}")

    return results


# ==================== Exp4: GLM4类别特异写入层定位 ====================
def exp4_glm4_writer_layer(model, tokenizer, device, model_name):
    """
    定位GLM4的类别特异写入层
    方法: 不使用W_down(可能meta device), 而用residual差异法:
    1. 在fruit和animal对象上分别捕获各层residual
    2. 计算层间residual差异: h_{l+1} - h_l (近似MLP+Attn写入)
    3. 测量该差异在fruit方向的投影
    4. 找到最大fruit写入层
    """
    if model_name != "glm4":
        plog(f"  Exp4 skipped (only for glm4)")
        return {"skipped": True, "reason": "only for glm4"}

    plog(f"=== Exp4: GLM4 Category-Specific Writer Layer ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    scan_layers = [24, 27, 30, 33, 35, 37, 39]
    scan_layers = [l for l in scan_layers if l < info.n_layers]
    
    n_obj = 3
    
    # Step1: 获取fruit和animal的各层residual
    plog(f"  Step1: Capturing residuals for fruit/animal at multiple layers...")
    
    fruit_resids = {}  # {layer_idx: mean_residual}
    animal_resids = {}
    
    for cat, store in [("fruit", fruit_resids), ("animal", animal_resids)]:
        if cat == "fruit":
            objs = CATEGORIES["fruit"][:n_obj]
        else:
            objs = CATEGORIES["animal"][:n_obj]
        
        for obj in objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            
            # 捕获多层residual
            cap = {}
            hooks = []
            for li in scan_layers:
                hooks.append(layers_list[li].register_forward_hook(_make_capture_hook(cap, f"L{li}")))
            
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            
            for h in hooks:
                h.remove()
            
            for li in scan_layers:
                key = f"L{li}"
                if key in cap:
                    r = cap[key][0, pos].numpy()
                    if li not in store:
                        store[li] = []
                    store[li].append(r)
    
    # 计算mean residuals
    for li in scan_layers:
        if li in fruit_resids and len(fruit_resids[li]) > 0:
            fruit_resids[li] = np.mean(fruit_resids[li], axis=0)
        else:
            fruit_resids[li] = np.zeros(info.d_model)
        if li in animal_resids and len(animal_resids[li]) > 0:
            animal_resids[li] = np.mean(animal_resids[li], axis=0)
        else:
            animal_resids[li] = np.zeros(info.d_model)
    
    # Step2: 计算层间写入差异和fruit DCF投影
    plog(f"  Step2: Computing layer-wise write and fruit projection...")
    
    # DCF fruit方向的W_U投影(用于测量写入方向)
    fruit_family_vecs = []
    for w in FAMILY_WORDS_8D["fruit"]:
        tid = find_token_id(tokenizer, w)
        if tid is not None and tid < W_U.shape[0]:
            fruit_family_vecs.append(W_U[tid])
    fruit_direction_wu = np.mean(fruit_family_vecs, axis=0) if fruit_family_vecs else np.zeros(info.d_model)
    fruit_dir_norm = np.linalg.norm(fruit_direction_wu)
    if fruit_dir_norm > 0:
        fruit_direction_wu = fruit_direction_wu / fruit_dir_norm
    
    results = {}
    
    for li in scan_layers:
        # 层间差异(近似该层的写入)
        write_diff = fruit_resids[li] - animal_resids[li]
        write_norm = np.linalg.norm(write_diff)
        
        # fruit DCF投影
        fruit_proj = np.dot(write_diff, fruit_direction_wu) if write_norm > 0 else 0.0
        
        # DCF at this layer
        fruit_dcf = logit_lens_dcf(fruit_resids[li], W_U, tokenizer)
        animal_dcf = logit_lens_dcf(animal_resids[li], W_U, tokenizer)
        dcf_separation = fruit_dcf[0] - animal_dcf[0]  # fruit维度差
        
        # 选择性: fruit vs max(other)
        max_other_sep = max(abs(fruit_dcf[i] - animal_dcf[i]) for i in range(1, len(DCF_DIM_NAMES)))
        selectivity = abs(dcf_separation) / (max_other_sep + 0.01)
        
        results[f"L{li}"] = {
            "fruit_dcf": {DCF_DIM_NAMES[i]: float(fruit_dcf[i]) for i in range(len(DCF_DIM_NAMES))},
            "animal_dcf": {DCF_DIM_NAMES[i]: float(animal_dcf[i]) for i in range(len(DCF_DIM_NAMES))},
            "dcf_separation_fruit": float(dcf_separation),
            "selectivity": float(selectivity),
            "write_diff_norm": float(write_norm),
            "fruit_projection": float(fruit_proj),
        }
        plog(f"    L{li}: fruit_sep={dcf_separation:.3f}, selectivity={selectivity:.2f}, "
             f"fruit_proj={fruit_proj:.3f}")
    
    return results


# ==================== Exp5: DS7B Head组合消融 ====================
def exp5_ds7b_head_combination(model, tokenizer, device, model_name):
    """
    联合关闭格式覆盖相关的多个heads
    测试: Head 12单独, Head 12+13, Head 12+13+10, 全部format heads
    
    sign convention: score = clean - ablation
    正值 = head在正常时促进该token
    """
    if model_name != "deepseek7b":
        plog(f"  Exp5 skipped (only for deepseek7b)")
        return {"skipped": True, "reason": "only for deepseek7b"}

    plog(f"=== Exp5: DS7B Head Combination Ablation ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    target_layer = 27
    layer = layers_list[target_layer]

    if hasattr(layer.self_attn, 'num_heads'):
        n_heads = layer.self_attn.num_heads
    elif hasattr(model.config, 'num_attention_heads'):
        n_heads = model.config.num_attention_heads
    else:
        n_heads = info.d_model // layer.self_attn.head_dim
    head_dim = layer.self_attn.head_dim
    plog(f"  L27: n_heads={n_heads}, head_dim={head_dim}")

    # Token IDs
    format_ids = [find_token_id(tokenizer, t) for t in FORMAT_TOKENS]
    format_ids = [tid for tid in format_ids if tid is not None]
    semantic_ids = [find_token_id(tokenizer, t) for t in SEMANTIC_TOKENS]
    semantic_ids = [tid for tid in semantic_ids if tid is not None]

    n_obj = 3
    test_cats = {
        "fruit": CATEGORIES["fruit"][:n_obj],
        "animal": CATEGORIES["animal"][:n_obj],
    }

    # 消融组合
    ablation_combos = {
        "head_12_only": [12],
        "head_13_only": [13],
        "head_12_13": [12, 13],
        "head_12_13_10": [10, 12, 13],
        "head_0_12_13_10": [0, 10, 12, 13],
    }

    results = {}

    for combo_name, head_list in ablation_combos.items():
        plog(f"  Ablating {combo_name}: heads={head_list}")
        
        # 计算需要零掉的范围
        zero_ranges = []
        for h_idx in head_list:
            if h_idx < n_heads:
                start = h_idx * head_dim
                end = start + head_dim
                zero_ranges.append((start, end))

        combo_result = {}
        
        for cat, objs in test_cats.items():
            format_scores = []
            semantic_scores = []
            dcf_before_list = []
            dcf_after_list = []

            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                # Clean
                cap_clean = {}
                h_clean = layers_list[info.n_layers - 1].register_forward_hook(
                    _make_capture_hook(cap_clean, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_clean.remove()

                if "resid" not in cap_clean:
                    continue

                clean_r = cap_clean["resid"][0, pos].numpy()
                logits_clean = clean_r @ W_U.T
                clean_format = float(np.mean([logits_clean[tid] for tid in format_ids if tid < len(logits_clean)]))
                clean_semantic = float(np.mean([logits_clean[tid] for tid in semantic_ids if tid < len(logits_clean)]))
                dcf_before = logit_lens_dcf(clean_r, W_U, tokenizer)

                # Ablate
                cap_abl = {}
                h_abl = layers_list[info.n_layers - 1].register_forward_hook(
                    _make_capture_hook(cap_abl, "resid"))

                def make_multi_head_ablation_hook(ranges, position):
                    def pre_hook(module, args):
                        if isinstance(args, tuple) and len(args) > 0:
                            x = args[0].clone()
                            for start, end in ranges:
                                x[0, position, start:end] = 0.0
                            return (x,) + args[1:] if len(args) > 1 else (x,)
                        return args
                    return pre_hook

                h_pre = layer.self_attn.o_proj.register_forward_pre_hook(
                    make_multi_head_ablation_hook(zero_ranges, pos))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                h_pre.remove()
                h_abl.remove()

                if "resid" in cap_abl:
                    abl_r = cap_abl["resid"][0, pos].numpy()
                    logits_abl = abl_r @ W_U.T
                    abl_format = float(np.mean([logits_abl[tid] for tid in format_ids if tid < len(logits_abl)]))
                    abl_semantic = float(np.mean([logits_abl[tid] for tid in semantic_ids if tid < len(logits_abl)]))
                    dcf_after = logit_lens_dcf(abl_r, W_U, tokenizer)

                    format_scores.append(clean_format - abl_format)
                    semantic_scores.append(clean_semantic - abl_semantic)
                    dcf_before_list.append(dcf_before)
                    dcf_after_list.append(dcf_after)

            if format_scores:
                mean_dcf_delta = np.mean(dcf_after_list, axis=0) - np.mean(dcf_before_list, axis=0)
                combo_result[cat] = {
                    "format_contribution": float(np.mean(format_scores)),
                    "semantic_contribution": float(np.mean(semantic_scores)),
                    "fmt_minus_sem": float(np.mean(format_scores) - np.mean(semantic_scores)),
                    "dcf_delta": {DCF_DIM_NAMES[i]: float(mean_dcf_delta[i]) for i in range(len(DCF_DIM_NAMES))},
                }
                plog(f"    {cat}: format={np.mean(format_scores):.3f}, "
                     f"semantic={np.mean(semantic_scores):.3f}, "
                     f"fmt-sem={np.mean(format_scores)-np.mean(semantic_scores):.3f}")

        results[combo_name] = combo_result

    return results


# ==================== Exp6: 翻译重构预实验 ====================
def exp6_translation_preexperiment(model, tokenizer, device, model_name):
    """
    初步测试语义簇是否跨语言共享
    
    方法:
    1. 英文kind_of模板 vs 中文kind_of模板
    2. 比较DCF和attribute profile
    3. 注入fruit writer看中文模板是否也受影响
    """
    if model_name != "qwen3":
        plog(f"  Exp6 skipped (only for qwen3)")
        return {"skipped": True, "reason": "only for qwen3"}

    plog(f"=== Exp6: Translation Reconstruction Pre-Experiment ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    # 英文对象和模板
    en_templates = {
        "kind_of": "The {obj} is a kind of",
        "eaten_as": "{obj} is usually eaten as",
    }
    en_objects = ["apple", "banana", "dog", "cat"]
    
    # 中文对象和模板
    zh_templates = {
        "kind_of": "{obj}是一种",
        "eaten_as": "{obj}通常被当作",
    }
    zh_objects = ["苹果", "香蕉", "狗", "猫"]

    # Step1: 比较英文vs中文模板的DCF和attribute profile
    plog(f"  Step1: Comparing EN vs ZH attribute profiles...")
    
    en_attr_profiles = {}
    zh_attr_profiles = {}
    
    for obj_en, obj_zh in zip(en_objects[:2], zh_objects[:2]):  # apple, banana
        # EN
        prompt = en_templates["kind_of"].format(obj=obj_en)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap = {}
        h = layers_list[info.n_layers - 1].register_forward_hook(_make_capture_hook(cap, "resid"))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h.remove()
        if "resid" in cap:
            r = cap["resid"][0, pos].numpy()
            en_attr_profiles[obj_en] = {
                "dcf8d": {DCF_DIM_NAMES[i]: float(v) for i, v in enumerate(logit_lens_dcf(r, W_U, tokenizer))},
                "attr": {ATTR_DIM_NAMES[i]: float(v) for i, v in enumerate(logit_lens_dcf(r, W_U, tokenizer, ATTRIBUTE_WORDS, ATTR_DIM_NAMES))},
            }
        
        # ZH
        prompt = zh_templates["kind_of"].format(obj=obj_zh)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap = {}
        h = layers_list[info.n_layers - 1].register_forward_hook(_make_capture_hook(cap, "resid"))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h.remove()
        if "resid" in cap:
            r = cap["resid"][0, pos].numpy()
            zh_attr_profiles[obj_zh] = {
                "dcf8d": {DCF_DIM_NAMES[i]: float(v) for i, v in enumerate(logit_lens_dcf(r, W_U, tokenizer))},
                "attr": {ATTR_DIM_NAMES[i]: float(v) for i, v in enumerate(logit_lens_dcf(r, W_U, tokenizer, ATTRIBUTE_WORDS, ATTR_DIM_NAMES))},
            }
    
    # Step2: 注入fruit writer看中文模板是否受影响
    plog(f"  Step2: Fruit writer injection on ZH template...")
    
    writer_info = find_fruit_writer_neurons(model, tokenizer, device, model_name)
    mean_write_vec = writer_info["mean_write_vec"]
    target_layer = writer_info["target_layer"]
    
    inject_tensor = torch.tensor(mean_write_vec, dtype=torch.float32)
    
    # 中文animal对象
    zh_animal_objects = ["狗", "猫"]
    zh_inject_results = {}
    
    for obj_zh in zh_animal_objects:
        prompt = zh_templates["kind_of"].format(obj=obj_zh)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        
        # Clean
        cap_clean = {}
        h_clean = layers_list[info.n_layers - 1].register_forward_hook(
            _make_capture_hook(cap_clean, "resid"))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h_clean.remove()
        
        if "resid" not in cap_clean:
            continue
        dcf_before = logit_lens_dcf(cap_clean["resid"][0, pos].numpy(), W_U, tokenizer)
        attr_before = logit_lens_dcf(cap_clean["resid"][0, pos].numpy(), W_U, tokenizer, ATTRIBUTE_WORDS, ATTR_DIM_NAMES)
        
        # Injected
        cap_pert = {}
        h_pert = layers_list[info.n_layers - 1].register_forward_hook(
            _make_capture_hook(cap_pert, "resid"))
        h_inj = layers_list[target_layer].register_forward_hook(make_inject_hook(inject_tensor, pos))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h_pert.remove()
        h_inj.remove()
        
        if "resid" in cap_pert:
            dcf_after = logit_lens_dcf(cap_pert["resid"][0, pos].numpy(), W_U, tokenizer)
            attr_after = logit_lens_dcf(cap_pert["resid"][0, pos].numpy(), W_U, tokenizer, ATTRIBUTE_WORDS, ATTR_DIM_NAMES)
            
            dcf_delta = dcf_after - dcf_before
            attr_delta = attr_after - attr_before
            
            zh_inject_results[obj_zh] = {
                "dcf_delta": {DCF_DIM_NAMES[i]: float(dcf_delta[i]) for i in range(len(DCF_DIM_NAMES))},
                "attr_delta": {ATTR_DIM_NAMES[i]: float(attr_delta[i]) for i in range(len(ATTR_DIM_NAMES))},
            }
            plog(f"    {obj_zh}: fruit_Δ={dcf_delta[0]:.3f}, plant_Δ={dcf_delta[7]:.3f}, "
                 f"edible_Δ={attr_delta[0]:.3f}")
    
    # Step3: 比较英文animal对象(对照)
    plog(f"  Step3: EN animal injection (comparison)...")
    en_inject_results = {}
    
    for obj_en in ["dog", "cat"]:
        prompt = en_templates["kind_of"].format(obj=obj_en)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        
        cap_clean = {}
        h_clean = layers_list[info.n_layers - 1].register_forward_hook(
            _make_capture_hook(cap_clean, "resid"))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h_clean.remove()
        
        if "resid" not in cap_clean:
            continue
        dcf_before = logit_lens_dcf(cap_clean["resid"][0, pos].numpy(), W_U, tokenizer)
        
        cap_pert = {}
        h_pert = layers_list[info.n_layers - 1].register_forward_hook(
            _make_capture_hook(cap_pert, "resid"))
        h_inj = layers_list[target_layer].register_forward_hook(make_inject_hook(inject_tensor, pos))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h_pert.remove()
        h_inj.remove()
        
        if "resid" in cap_pert:
            dcf_after = logit_lens_dcf(cap_pert["resid"][0, pos].numpy(), W_U, tokenizer)
            dcf_delta = dcf_after - dcf_before
            
            en_inject_results[obj_en] = {
                "dcf_delta": {DCF_DIM_NAMES[i]: float(dcf_delta[i]) for i in range(len(DCF_DIM_NAMES))},
            }
            plog(f"    {obj_en}: fruit_Δ={dcf_delta[0]:.3f}, plant_Δ={dcf_delta[7]:.3f}")

    results = {
        "en_attr_profiles": en_attr_profiles,
        "zh_attr_profiles": zh_attr_profiles,
        "zh_inject_results": zh_inject_results,
        "en_inject_results": en_inject_results,
    }
    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    plog(f"Phase 478: Cluster Decomposition, Category-Specific Control & Format Head Closure")
    plog(f"Model: {model_name}, Round: {round_num}")

    t_start = time.time()

    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model info: class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    results = {
        "phase": 478,
        "model": model_name,
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "theory": "Semantic Cluster Decomposition, Attribute DCF & Category-Specific Control",
        "core_question": "Can we decompose the fruit-plant-food cluster and find fruit-specific direction?",
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }

    # Exp1: Attribute DCF + Fruit Writer属性画像 (仅qwen3)
    try:
        results["exp1_attribute_dcf"] = exp1_attribute_dcf(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp1_attribute_dcf"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Exp1 done. Elapsed: {time.time()-t_start:.1f}s")

    # Exp2: 语义簇分解 (仅qwen3)
    try:
        results["exp2_cluster_decomposition"] = exp2_cluster_decomposition(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp2_cluster_decomposition"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Exp2 done. Elapsed: {time.time()-t_start:.1f}s")

    # Exp3: 解耦方向注入测试 (仅qwen3)
    try:
        results["exp3_decomposed_injection"] = exp3_decomposed_injection(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp3_decomposed_injection"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Exp3 done. Elapsed: {time.time()-t_start:.1f}s")

    # Exp4: GLM4类别特异写入层定位 (仅glm4)
    try:
        results["exp4_glm4_writer_layer"] = exp4_glm4_writer_layer(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp4 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp4_glm4_writer_layer"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Exp4 done. Elapsed: {time.time()-t_start:.1f}s")

    # Exp5: DS7B Head组合消融 (仅deepseek7b)
    try:
        results["exp5_ds7b_head_combination"] = exp5_ds7b_head_combination(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp5 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp5_ds7b_head_combination"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Exp5 done. Elapsed: {time.time()-t_start:.1f}s")

    # Exp6: 翻译重构预实验 (仅qwen3)
    try:
        results["exp6_translation_preexperiment"] = exp6_translation_preexperiment(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp6 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp6_translation_preexperiment"] = {"error": str(e)}

    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase478_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    plog(f"Results saved to {out_path}")

    # 释放模型
    release_model(model)
    t_total = time.time() - t_start
    plog(f"Phase 478 {model_name} complete. Total: {t_total:.1f}s ({t_total/60:.1f}min)")


if __name__ == "__main__":
    main()
