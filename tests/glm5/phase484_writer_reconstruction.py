"""
Phase 484: 分布式边界写入场重构 + 关系槽位读出 + 异常竞争对解释
================================================================

核心目标(从Phase 483推进):
1. Exp1: 用岭回归重构边界写入场 — 解决Phase 483中cos(Bc)低的问题
2. Exp2: 写入场因果测试 — 消融重构出的神经元，验证能否复现B_c效果
3. Exp3: 关系槽位读出 — 不同关系下注入B_c和M_c，观察读出差异
4. Exp4: 异常竞争对解释 — food→vehicle和animal→clothing的token级分析

关键改进:
- 用ridge regression而非top-k排序来重构B_c
- 测试6种关系模板(kind_of, used_for, found_in, made_of, eaten_as, grown_from)
- 对异常竞争对做token级DCF分解

用法:
  python tests/glm5/phase484_writer_reconstruction.py qwen3 1
  python tests/glm5/phase484_writer_reconstruction.py glm4 1
  python tests/glm5/phase484_writer_reconstruction.py deepseek7b 1
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

BEST_NEIGHBORS = {
    "qwen3": {
        "fruit": ["plant", "food"], "animal": ["food", "clothing"],
        "tool": ["vehicle", "furniture"], "vehicle": ["furniture", "tool"],
        "clothing": ["furniture", "tool"], "furniture": ["vehicle", "clothing"],
        "food": ["plant", "vehicle"], "plant": ["food", "animal"],
    },
    "glm4": {
        "fruit": ["plant", "food"], "animal": ["food", "clothing"],
        "tool": ["furniture", "vehicle"], "vehicle": ["tool", "furniture"],
        "clothing": ["furniture", "plant"], "furniture": ["vehicle", "clothing"],
        "food": ["plant", "fruit"], "plant": ["vehicle", "clothing"],
    },
    "deepseek7b": {
        "fruit": ["plant", "food"], "animal": ["food", "clothing"],
        "tool": ["vehicle", "furniture"], "vehicle": ["furniture", "tool"],
        "clothing": ["furniture", "plant"], "furniture": ["tool", "clothing"],
        "food": ["plant", "fruit"], "plant": ["food", "fruit"],
    },
}

BEST_LAYERS = {
    "qwen3": {
        "fruit": 32, "animal": 33, "tool": 23, "vehicle": 29,
        "clothing": 30, "furniture": 26, "food": 34, "plant": 28,
    },
    "glm4": {
        "fruit": 27, "animal": 38, "tool": 27, "vehicle": 29,
        "clothing": 39, "furniture": 34, "food": 38, "plant": 32,
    },
    "deepseek7b": {
        "fruit": 26, "animal": 27, "tool": 26, "vehicle": 26,
        "clothing": 23, "furniture": 25, "food": 27, "plant": 25,
    },
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

# ===== 关系模板(Phase 484新增) =====
RELATION_TEMPLATES = {
    "kind_of":   "The {obj} is a kind of",
    "used_for":  "The {obj} is used for",
    "found_in":  "The {obj} is found in",
    "made_of":   "The {obj} is made of",
    "eaten_as":  "The {obj} is eaten as",
    "grown_from":"The {obj} is grown from",
}

# 适用于不同关系的关系词(用于DCF测量)
RELATION_DCF_WORDS = {
    "kind_of": FAMILY_WORDS_8D,
    "used_for": {
        "fruit":     ["eat", "cook", "juice", "bake"],
        "animal":    ["pet", "ride", "hunt", "farm"],
        "tool":      ["build", "fix", "cut", "repair"],
        "vehicle":   ["drive", "ride", "transport", "travel"],
        "clothing":  ["wear", "dress", "cover", "protect"],
        "furniture": ["sit", "sleep", "store", "support"],
        "food":      ["eat", "cook", "serve", "taste"],
        "plant":     ["grow", "plant", "decorate", "harvest"],
    },
    "found_in": {
        "fruit":     ["tree", "garden", "market", "kitchen"],
        "animal":    ["zoo", "farm", "forest", "home"],
        "tool":      ["workshop", "garage", "kitchen", "factory"],
        "vehicle":   ["road", "garage", "parking", "highway"],
        "clothing":  ["closet", "store", "wardrobe", "laundry"],
        "furniture": ["house", "office", "room", "apartment"],
        "food":      ["kitchen", "restaurant", "fridge", "table"],
        "plant":     ["garden", "forest", "field", "pot"],
    },
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
    n_layers = len(layers_list)
    
    # 检查并报告层分配情况
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
        plog(f"  Layer allocation: {gpu_layers} GPU + {cpu_layers} CPU (total {n_layers})")
        # 检查深层是否缺失
        last_gpu = max(int(lid) for lid, dev in layer_devices.items() if 'cuda' in dev) if gpu_layers > 0 else -1
        plog(f"  Last GPU layer: L{last_gpu}, total layers: {n_layers}")
    
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


def make_remove_hook(remove_vec, position, scale=1.0):
    added = [False]
    def hook_fn(module, inp, output):
        if not added[0]:
            if isinstance(output, tuple):
                out = output[0].clone()
            else:
                out = output.clone()
            b_hat = remove_vec / (np.linalg.norm(remove_vec) + 1e-10)
            resid_np = out[0, position, :].float().cpu().numpy()
            proj = np.dot(resid_np, b_hat) * b_hat * scale
            out[0, position, :] -= torch.tensor(proj, dtype=out.dtype, device=out.device)
            added[0] = True
            if isinstance(output, tuple):
                return (out,) + output[1:]
            return out
        return output
    return hook_fn


def qr_orthogonalize(target_vec, basis_vecs):
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


# ==================== 安全权重加载 ====================
def safe_load_weight(model, model_name, layer_idx, component_path):
    """
    安全加载权重，处理meta device问题
    
    component_path: 如 "mlp.down_proj" — 相对于layer的属性路径
    """
    layers_list = get_layers(model)
    layer = layers_list[layer_idx]
    
    # 尝试直接获取
    parts = component_path.split('.')
    obj = layer
    for p in parts:
        obj = getattr(obj, p)
    
    w = obj.weight
    if not w.is_meta:
        return w.detach().cpu().float().numpy()
    
    # meta device: 从safetensors加载
    plog(f"    L{layer_idx} {component_path} on meta, loading from safetensors...")
    from safetensors.torch import load_file
    import glob as glob_mod
    model_path = MODEL_CONFIGS[model_name]["path"]
    sf_files = glob_mod.glob(os.path.join(model_path, '*.safetensors'))
    for sf_file in sf_files:
        try:
            st = load_file(sf_file)
            key = f"model.layers.{layer_idx}.{component_path}.weight"
            if key in st:
                result = st[key].float().numpy()
                plog(f"      Loaded from {os.path.basename(sf_file)}")
                return result
        except Exception:
            continue
    
    plog(f"    WARNING: Cannot load {component_path} for L{layer_idx}")
    return None


# ==================== 获取特定层类别方向 ====================
def get_category_residuals_at_layer(model, tokenizer, device, model_name,
                                     categories, n_obj, target_layer, template_key="kind_of"):
    layers_list = get_layers(model)
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


def get_specific_direction(model, tokenizer, device, model_name, cat_name, target_layer, n_obj=4):
    neighbors = BEST_NEIGHBORS[model_name]
    raw_dirs = get_category_residuals_at_layer(
        model, tokenizer, device, model_name,
        categories=CATEGORIES_TRAIN, n_obj=n_obj, target_layer=target_layer
    )
    if cat_name not in raw_dirs:
        return None, 0.0
    target_vec = raw_dirs[cat_name]
    basis_vecs = [raw_dirs[n] for n in neighbors[cat_name] if n in raw_dirs]
    spec_vec = qr_orthogonalize(target_vec, basis_vecs) if basis_vecs else target_vec.copy()
    spec_norm = np.linalg.norm(spec_vec)
    return spec_vec, float(spec_norm)


def get_shared_manifold(model, tokenizer, device, model_name, cat_name, target_layer, n_obj=4):
    """获取共享语义流形方向(类别方向在邻居子空间上的投影)"""
    neighbors = BEST_NEIGHBORS[model_name]
    raw_dirs = get_category_residuals_at_layer(
        model, tokenizer, device, model_name,
        categories=CATEGORIES_TRAIN, n_obj=n_obj, target_layer=target_layer
    )
    if cat_name not in raw_dirs:
        return None, 0.0
    target_vec = raw_dirs[cat_name]
    basis_vecs = [raw_dirs[n] for n in neighbors[cat_name] if n in raw_dirs]
    if not basis_vecs:
        return None, 0.0
    # 共享方向 = target在邻居子空间上的投影
    basis = np.array(basis_vecs)
    Q, _ = np.linalg.qr(basis.T)
    proj = Q @ (Q.T @ target_vec)
    proj_norm = np.linalg.norm(proj)
    return proj, float(proj_norm)


# ==================== Exp1: 边界写入场重构(岭回归) ====================
def exp1_boundary_writer_reconstruction(model, tokenizer, device, model_name, W_U, n_test_obj=4):
    """
    用岭回归重构边界写入场
    
    关键改进(vs Phase 483 top-k):
    - 不用top-k排序，用ridge regression找能合成B_c的最小神经元集合
    - 计算重构cos(Bc), 与Phase 483的top-k方法对比
    
    方法:
    1. 获取MLP激活矩阵 A [n_samples, intermediate_size]
    2. 获取边界方向 B_c [d_model]
    3. 计算 W_down @ B_c 得到目标 y = W_down^T @ B_c [intermediate_size]
       (即：要合成B_c, 需要每个中间神经元的激活 * W_down[i] 投影到B_c)
    4. 用ridge regression: A^T @ w ≈ y, 找最小权重w
    """
    plog("=== Exp1: 边界写入场重构(岭回归) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    
    # 选择3个代表性类别
    if model_name == "qwen3":
        test_cats = ["fruit", "animal", "clothing"]
    elif model_name == "glm4":
        test_cats = ["fruit", "animal", "clothing"]
    else:
        test_cats = ["fruit", "animal", "clothing"]
    
    results = {}
    
    for cat_name in test_cats:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name} @ L{best_layer}...")
        t0 = time.time()
        
        # 获取specific方向和shared manifold
        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"    Skip {cat_name}: spec_norm too small ({spec_norm:.4f})")
            continue
        
        b_hat = spec_vec / spec_norm
        target_idx = cat_names.index(cat_name)
        
        # 获取W_down权重
        W_down = safe_load_weight(model, model_name, best_layer, "mlp.down_proj")
        if W_down is None:
            plog(f"    Skip {cat_name}: Cannot load W_down")
            continue
        # W_down shape: [d_model, intermediate_size]
        
        # 计算目标: 每个中间神经元对B_c方向的贡献权重
        # y[i] = W_down[:, i] · b_hat = 第i个中间神经元通过down_proj对B_c方向的贡献
        y = W_down.T @ b_hat  # [intermediate_size]
        
        # 获取MLP激活矩阵(用所有8个训练对象)
        template = RELATION_TEMPLATES["kind_of"]
        train_objs = CATEGORIES_TRAIN[cat_name]
        
        # 1) 类别内样本
        cat_activations = []
        for obj in train_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            
            cap_mlp = {}
            def make_mlp_input_hook(key):
                done = [False]
                def hook_fn(module, inp, output):
                    if not done[0]:
                        if isinstance(inp, tuple) and len(inp) > 0:
                            t = inp[0]
                            if not t.is_meta:
                                cap_mlp[key] = t.detach().float().cpu()
                        done[0] = True
                return hook_fn
            
            mlp = layers_list[best_layer].mlp
            h_mlp = mlp.down_proj.register_forward_hook(make_mlp_input_hook("mlp_act"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_mlp.remove()
            
            if "mlp_act" in cap_mlp:
                mlp_act = cap_mlp["mlp_act"][0, pos].numpy()
                cat_activations.append(mlp_act)
        
        # 2) 邻居类别样本(提供负样本)
        neighbor_cats = BEST_NEIGHBORS[model_name][cat_name]
        neighbor_activations = []
        for nc in neighbor_cats:
            for obj in CATEGORIES_TRAIN[nc][:2]:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                
                cap_mlp = {}
                def make_mlp_input_hook2(key):
                    done = [False]
                    def hook_fn(module, inp, output):
                        if not done[0]:
                            if isinstance(inp, tuple) and len(inp) > 0:
                                t = inp[0]
                                if not t.is_meta:
                                    cap_mlp[key] = t.detach().float().cpu()
                            done[0] = True
                    return hook_fn
                
                mlp = layers_list[best_layer].mlp
                h_mlp = mlp.down_proj.register_forward_hook(make_mlp_input_hook2("mlp_act"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_mlp.remove()
                
                if "mlp_act" in cap_mlp:
                    mlp_act = cap_mlp["mlp_act"][0, pos].numpy()
                    neighbor_activations.append(mlp_act)
        
        if len(cat_activations) == 0:
            plog(f"    Skip {cat_name}: no MLP activations captured")
            continue
        
        # 构建数据矩阵
        n_cat = len(cat_activations)
        n_neigh = len(neighbor_activations)
        all_acts = np.array(cat_activations + neighbor_activations)  # [n_total, intermediate]
        labels = np.array([1.0]*n_cat + [0.0]*n_neigh)  # 1=类别内, 0=邻居
        
        intermediate_size = all_acts.shape[1]
        plog(f"    Data: {n_cat} cat + {n_neigh} neighbor samples, intermediate={intermediate_size}")
        
        # ---- 方法1: Ridge regression重构B_c ----
        # 目标: 找w使得 sum_i w[j] * A[i,j] ≈ y[j]对所有类别内样本
        # 即: 类别内激活的加权平均应该指向B_c方向
        
        # 更直接: 用类别内vs邻居的差异激活来拟合y
        # diff_activation = mean(cat_act) - mean(neighbor_act)  # [intermediate]
        if n_neigh > 0:
            diff_act = np.mean(cat_activations, axis=0) - np.mean(neighbor_activations, axis=0)
        else:
            diff_act = np.mean(cat_activations, axis=0)
        
        # 差异激活和目标y之间的cos
        cos_diff_y = float(np.dot(diff_act, y) / (np.linalg.norm(diff_act) * np.linalg.norm(y) + 1e-10))
        
        # ---- Ridge回归: 从激活矩阵重构目标y ----
        # 问题: A @ w = y (每个样本的激活加权得到目标方向)
        # 转换: w = (A^T A + λI)^-1 A^T y_per_sample
        # 但这里y是固定的(每个样本都一样), 所以用简单方法:
        # 每个中间神经元对B_c的贡献 = activation_diff[i] * W_down[i,:] · B_c
        
        # 直接计算: 每个神经元对B_c重构的贡献
        # 贡献 = diff_act[j] * y[j] (激活差 × 对B_c的投影权重)
        neuron_contribution = diff_act * y  # [intermediate] — 每个神经元的边界贡献
        
        # 按贡献绝对值排序
        abs_contrib = np.abs(neuron_contribution)
        sorted_idx = np.argsort(abs_contrib)[::-1]
        
        # Top-k重构
        cos_at_k = {}
        energy_at_k = {}
        for k in [10, 20, 50, 100, 200, 500]:
            if k > intermediate_size:
                continue
            top_k_idx = sorted_idx[:k]
            # 用top-k神经元重构B_c
            reconstructed = np.zeros_like(y)
            for idx in top_k_idx:
                reconstructed[idx] = diff_act[idx]
            # 通过W_down映射回d_model空间
            recon_d_model = W_down @ reconstructed  # [d_model]
            # 计算与B_c的cos
            cos_bc = float(np.dot(recon_d_model, b_hat) / (np.linalg.norm(recon_d_model) + 1e-10))
            cos_at_k[k] = cos_bc
            # 能量覆盖率
            total_energy = np.sum(abs_contrib)
            k_energy = np.sum(abs_contrib[top_k_idx])
            energy_at_k[k] = float(k_energy / total_energy)
        
        # ---- Lasso回归: 找稀疏解 ----
        lasso_results = {}
        try:
            from sklearn.linear_model import Lasso, Ridge
            # 用类别内激活的均值与y的差异作为训练目标
            # X: [1, intermediate], y_target: [1, intermediate]
            # 实际上我们需要: 找w使得 A_cat @ w ≈ y (对于类别内样本)
            # 但维度太高, 用正则化方法
            
            # 更好的方法: 直接在d_model空间做
            # 获取类别内和邻居的残差流
            cat_resids = []
            neigh_resids = []
            for obj in CATEGORIES_TRAIN[cat_name]:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                cap = {}
                h = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                if "resid" in cap:
                    cat_resids.append(cap["resid"][0, pos].numpy())
            
            for nc in neighbor_cats:
                for obj in CATEGORIES_TRAIN[nc][:2]:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                    cap = {}
                    h = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    h.remove()
                    if "resid" in cap:
                        neigh_resids.append(cap["resid"][0, pos].numpy())
            
            if cat_resids and neigh_resids:
                mean_cat = np.mean(cat_resids, axis=0)
                mean_neigh = np.mean(neigh_resids, axis=0)
                diff_d_model = mean_cat - mean_neigh  # [d_model]
                
                # diff_d_model应该接近spec_vec
                cos_diff_spec = float(np.dot(diff_d_model, spec_vec) / 
                                     (np.linalg.norm(diff_d_model) * spec_norm + 1e-10))
                
                lasso_results["cos_diff_spec"] = cos_diff_spec
                lasso_results["diff_norm"] = float(np.linalg.norm(diff_d_model))
                lasso_results["spec_norm"] = float(spec_norm)
                
                # Ridge回归: 从MLP激活重构d_model空间的diff
                # A_cat [n_cat, intermediate] -> diff [d_model]
                # 需要: X @ W_down^T ≈ diff (对每个样本)
                # 简化: mean_cat_act @ W_down^T ≈ diff_d_model
                mean_cat_act = np.mean(cat_activations, axis=0)  # [intermediate]
                mean_neigh_act = np.mean(neighbor_activations, axis=0) if neighbor_activations else np.zeros(intermediate_size)
                diff_act_vec = mean_cat_act - mean_neigh_act  # [intermediate]
                
                # diff_act_vec @ W_down^T = 重构的d_model差
                recon_from_act = diff_act_vec @ W_down.T  # [d_model]
                cos_recon = float(np.dot(recon_from_act, spec_vec) / 
                                  (np.linalg.norm(recon_from_act) * spec_norm + 1e-10))
                
                lasso_results["cos_recon_diff_act"] = cos_recon
                lasso_results["recon_norm"] = float(np.linalg.norm(recon_from_act))
                
                # Ridge: 找稀疏w使得 w @ W_down^T ≈ B_c
                # 目标: minimize ||w||^2 + lambda * ||w @ W_down^T - B_c||^2
                # 解: w = B_c @ W_down @ (W_down^T @ W_down + lambda*I)^-1
                # 但维度太大, 用近似方法
                
                # Lasso: 找最小非零元素数的w
                # X = W_down.T [intermediate, d_model], y = b_hat [d_model]
                # w = argmin ||w @ X - y||^2 + alpha * ||w||_1
                # 注意: 样本数=d_model, 特征数=intermediate, 不标准
                # 用转置: X = W_down [d_model, intermediate], y = b_hat
                # w = argmin ||X @ w - y||^2 + alpha * ||w||_1
                # 这才是标准的: n_samples=d_model, n_features=intermediate
                
                from sklearn.linear_model import Lasso
                try:
                    # 目标: 找w[intermediate]使得 W_down @ w ≈ b_hat
                    # 即 minimize ||W_down @ w - b_hat||^2 + alpha * ||w||_1
                    # W_down: [d_model, intermediate], w: [intermediate], b_hat: [d_model]
                    # 标准形式: X = W_down.T [intermediate, d_model], y = b_hat [d_model]
                    # lasso.fit(X.T, y) -> coef_ = [d_model]  不对
                    
                    # 正确: lasso.fit(W_down.T, b_hat) 
                    # X=W_down.T [intermediate, d_model], y=b_hat [d_model]
                    # 但样本数=d_model=2560, 特征数=intermediate=9728, 样本<特征，欠定
                    
                    # 采样d_model维度避免太大
                    n_dim = min(512, info.d_model)
                    sample_dims = np.random.choice(info.d_model, n_dim, replace=False)
                    # X: [n_dim, intermediate] — 每个采样维度一个样本
                    X_sub = W_down[sample_dims]  # [n_dim, intermediate]
                    y_sub = b_hat[sample_dims]   # [n_dim]
                    
                    # Lasso: 找稀疏w[intermediate]使得X_sub @ w ≈ y_sub
                    lasso = Lasso(alpha=0.001, max_iter=5000, tol=1e-4)
                    lasso.fit(X_sub, y_sub)
                    
                    w_lasso = lasso.coef_  # [intermediate]
                    n_nonzero = np.sum(np.abs(w_lasso) > 1e-6)
                    
                    # 用lasso权重重构B_c: recon = W_down @ w_lasso
                    recon_lasso_d = W_down @ w_lasso  # [d_model]
                    cos_lasso = float(np.dot(recon_lasso_d, b_hat) / 
                                      (np.linalg.norm(recon_lasso_d) + 1e-10))
                    
                    lasso_results["lasso_n_nonzero"] = int(n_nonzero)
                    lasso_results["lasso_cos_bc"] = cos_lasso
                    lasso_results["lasso_alpha"] = float(lasso.alpha_)
                    plog(f"    Lasso: n_nonzero={n_nonzero}, cos(Bc)={cos_lasso:.3f}")
                    
                except Exception as e:
                    plog(f"    Lasso failed: {e}")
                    lasso_results["lasso_error"] = str(e)
        except ImportError:
            plog(f"    sklearn not available, skipping Lasso")
            lasso_results["sklearn_missing"] = True
        
        elapsed = time.time() - t0
        plog(f"    Done in {elapsed:.1f}s: cos_diff_y={cos_diff_y:.3f}, "
              f"cos@10={cos_at_k.get(10,0):.3f}, cos@50={cos_at_k.get(50,0):.3f}, "
              f"cos@200={cos_at_k.get(200,0):.3f}")
        
        results[cat_name] = {
            "best_layer": best_layer,
            "spec_norm": float(spec_norm),
            "cos_diff_y": cos_diff_y,
            "cos_at_k": cos_at_k,
            "energy_at_k": energy_at_k,
            "neuron_contribution_stats": {
                "max": float(np.max(abs_contrib)),
                "top10_mean": float(np.mean(abs_contrib[sorted_idx[:10]])),
                "top50_mean": float(np.mean(abs_contrib[sorted_idx[:50]])),
                "total_neurons": int(intermediate_size),
                "n_significant": int(np.sum(abs_contrib > 0.01 * np.max(abs_contrib))),
            },
            "lasso_results": lasso_results,
            "elapsed": round(elapsed, 1),
        }
        
        # 清理GPU
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


# ==================== Exp2: 写入场因果测试 ====================
def exp2_writer_causal_test(model, tokenizer, device, model_name, W_U, n_test_obj=4):
    """
    消融重构出的top神经元，验证能否复现B_c效果
    
    方法:
    1. 对每个类别找到top-20边界写入神经元
    2. 将这些神经元的激活置零(消融)
    3. 测量消融后DCF变化
    4. 与方向级remove B_c的效果对比
    """
    plog("=== Exp2: 写入场因果测试 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    
    if model_name == "qwen3":
        test_cats = ["fruit", "animal", "clothing"]
    elif model_name == "glm4":
        test_cats = ["fruit", "animal", "clothing"]
    else:
        test_cats = ["fruit", "animal", "clothing"]
    
    results = {}
    
    for cat_name in test_cats:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name} @ L{best_layer}...")
        t0 = time.time()
        
        # 获取specific方向
        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"    Skip {cat_name}: spec_norm too small")
            continue
        
        b_hat = spec_vec / spec_norm
        target_idx = cat_names.index(cat_name)
        
        # 获取W_down
        W_down = safe_load_weight(model, model_name, best_layer, "mlp.down_proj")
        if W_down is None:
            continue
        
        # 计算每个神经元的边界贡献权重
        y = W_down.T @ b_hat  # [intermediate]
        
        # 获取类别内平均MLP激活(确定top神经元)
        template = RELATION_TEMPLATES["kind_of"]
        train_objs = CATEGORIES_TRAIN[cat_name]
        
        cat_acts = []
        for obj in train_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            
            cap_mlp = {}
            def make_hook(key):
                done = [False]
                def hook_fn(module, inp, output):
                    if not done[0]:
                        if isinstance(inp, tuple) and len(inp) > 0:
                            t = inp[0]
                            if not t.is_meta:
                                cap_mlp[key] = t.detach().float().cpu()
                        done[0] = True
                return hook_fn
            
            mlp = layers_list[best_layer].mlp
            h_mlp = mlp.down_proj.register_forward_hook(make_hook("mlp_act"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_mlp.remove()
            
            if "mlp_act" in cap_mlp:
                cat_acts.append(cap_mlp["mlp_act"][0, pos].numpy())
        
        if not cat_acts:
            plog(f"    Skip {cat_name}: no MLP activations")
            continue
        
        mean_act = np.mean(cat_acts, axis=0)  # [intermediate]
        neuron_contrib = mean_act * y  # [intermediate] — 每个神经元的边界贡献
        abs_contrib = np.abs(neuron_contrib)
        sorted_idx = np.argsort(abs_contrib)[::-1]
        
        # 测试不同k值的神经元消融
        test_ks = [5, 10, 20, 50]
        
        # 先做baseline DCF
        test_objs = CATEGORIES[cat_name][:n_test_obj]
        baseline_dcfs = []
        for obj in test_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap = {}
            h = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap:
                dcf = logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer)
                baseline_dcfs.append(dcf)
        
        if not baseline_dcfs:
            plog(f"    Skip {cat_name}: no baseline DCFs")
            continue
        
        mean_baseline = np.mean(baseline_dcfs, axis=0)
        
        # 方向级remove B_c的baseline
        remove_dcfs = []
        for obj in test_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            h = layers_list[best_layer].register_forward_hook(
                make_remove_hook(spec_vec, pos, scale=1.0))
            cap = {}
            h2 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            h2.remove()
            if "resid" in cap:
                dcf = logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer)
                remove_dcfs.append(dcf)
        
        mean_remove = np.mean(remove_dcfs, axis=0) if remove_dcfs else mean_baseline
        remove_delta = mean_remove - mean_baseline
        
        # 神经元消融测试 — 计算方式(不用hook消融)
        # 方法: baseline_resid - sum_{j in top_k} (act[j] * W_down[:, j])
        # 即: 从残差流中减去top-k神经元对MLP输出的贡献
        ablation_results = {}
        for k in test_ks:
            top_k_idx = sorted_idx[:k]
            
            # 对每个测试对象，计算消融后DCF
            ablate_dcfs = []
            for obj in test_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                
                # 1) 捕获baseline残差流和MLP激活
                cap_resid = {}
                cap_mlp_act = {}
                done_resid = [False]
                done_mlp = [False]
                
                def make_resid_hook(key):
                    def hook_fn(module, inp, output):
                        if not done_resid[0]:
                            if isinstance(output, tuple):
                                cap_resid[key] = output[0].detach().float().cpu()
                            else:
                                cap_resid[key] = output.detach().float().cpu()
                            done_resid[0] = True
                    return hook_fn
                
                def make_mlp_act_hook(key, position):
                    def hook_fn(module, inp, output):
                        if not done_mlp[0]:
                            if isinstance(inp, tuple) and len(inp) > 0:
                                t = inp[0]
                                if not t.is_meta:
                                    cap_mlp_act[key] = t.detach().float().cpu()
                            done_mlp[0] = True
                    return hook_fn
                
                mlp = layers_list[best_layer].mlp
                h1 = layers_list[best_layer].register_forward_hook(make_resid_hook("resid"))
                h2 = mlp.down_proj.register_forward_hook(make_mlp_act_hook("mlp_act", pos))
                
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                
                h1.remove()
                h2.remove()
                
                if "resid" not in cap_resid or "mlp_act" not in cap_mlp_act:
                    continue
                
                baseline_resid = cap_resid["resid"][0, pos].numpy()  # [d_model]
                mlp_act = cap_mlp_act["mlp_act"][0, pos].numpy()  # [intermediate]
                
                # 2) 计算top-k神经元对残差流的贡献
                # MLP输出贡献 = sum_j (act[j] * W_down[:, j])
                # top-k消融后的残差 = baseline_resid - sum_{j in top_k} (act[j] * W_down[:, j])
                ablated_contribution = np.zeros(info.d_model)
                for j in top_k_idx:
                    ablated_contribution += mlp_act[j] * W_down[:, j]
                
                ablated_resid = baseline_resid - ablated_contribution
                
                # 3) 计算DCF
                dcf = logit_lens_dcf(ablated_resid, W_U, tokenizer)
                ablate_dcfs.append(dcf)
            
            mean_ablate = np.mean(ablate_dcfs, axis=0) if ablate_dcfs else mean_baseline
            ablate_delta = mean_ablate - mean_baseline
            
            # 计算与方向级remove的相似度
            cos_with_remove = float(np.dot(ablate_delta, remove_delta) / 
                                     (np.linalg.norm(ablate_delta) * np.linalg.norm(remove_delta) + 1e-10))
            
            ablation_results[f"k={k}"] = {
                "target_delta": float(ablate_delta[target_idx]),
                "max_competitor_release": float(max(ablate_delta[i] for i in range(8) if i != target_idx)),
                "cos_with_direction_remove": cos_with_remove,
                "dcf_delta": ablate_delta.tolist(),
            }
            plog(f"    k={k}: target_D={ablate_delta[target_idx]:.2f}, "
                  f"cos_with_remove={cos_with_remove:.3f}")
        
        results[cat_name] = {
            "best_layer": best_layer,
            "spec_norm": float(spec_norm),
            "direction_remove_delta": remove_delta.tolist(),
            "direction_remove_target": float(remove_delta[target_idx]),
            "ablation_results": ablation_results,
            "elapsed": round(time.time() - t0, 1),
        }
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


# ==================== Exp3: 关系槽位读出 ====================
def exp3_relation_slot_readout(model, tokenizer, device, model_name, W_U):
    """
    不同关系下注入B_c和M_c，观察读出差异
    
    核心假设:
    - kind_of 主要读 B_c(类别边界残差)
    - used_for 主要读功能共享流形
    - found_in 主要读场景属性
    """
    plog("=== Exp3: 关系槽位读出 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    
    # 选择2个类别做详细分析(控制时间)
    test_cats = ["fruit", "animal"]
    
    # 测试关系
    test_relations = ["kind_of", "used_for", "found_in"]
    
    # 注入尺度
    injection_scales = [0.5, 1.0]
    
    results = {}
    
    for cat_name in test_cats:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name} @ L{best_layer}...")
        t0 = time.time()
        
        # 获取specific方向和shared manifold
        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        shared_vec, shared_norm = get_shared_manifold(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"    Skip {cat_name}: spec_norm too small")
            continue
        
        target_idx = cat_names.index(cat_name)
        
        cat_results = {}
        
        for relation in test_relations:
            template = RELATION_TEMPLATES[relation]
            test_objs = CATEGORIES_TRAIN[cat_name]
            
            relation_results = {}
            
            for scale in injection_scales:
                # baseline
                baseline_dcfs = []
                for obj in test_objs:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                    cap = {}
                    h = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    h.remove()
                    if "resid" in cap:
                        dcf = logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer)
                        baseline_dcfs.append(dcf)
                
                mean_baseline = np.mean(baseline_dcfs, axis=0) if baseline_dcfs else np.zeros(8)
                
                # +B_c 注入
                inject_dcfs = []
                for obj in test_objs:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                    ivec = torch.tensor(spec_vec * scale, dtype=torch.float32)
                    h = layers_list[best_layer].register_forward_hook(make_inject_hook(ivec, pos))
                    cap = {}
                    h2 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    h.remove()
                    h2.remove()
                    if "resid" in cap:
                        dcf = logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer)
                        inject_dcfs.append(dcf)
                
                mean_inject = np.mean(inject_dcfs, axis=0) if inject_dcfs else mean_baseline
                inject_delta = mean_inject - mean_baseline
                inject_sel = compute_selectivity(inject_delta, target_idx)
                
                # +M_c 注入(共享流形)
                shared_inject_dcfs = []
                if shared_vec is not None and shared_norm > 1e-6:
                    for obj in test_objs:
                        prompt = template.format(obj=obj)
                        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                        ivec = torch.tensor(shared_vec * scale, dtype=torch.float32)
                        h = layers_list[best_layer].register_forward_hook(make_inject_hook(ivec, pos))
                        cap = {}
                        h2 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
                        with torch.no_grad():
                            model(input_ids=input_ids, attention_mask=attention_mask)
                        h.remove()
                        h2.remove()
                        if "resid" in cap:
                            dcf = logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer)
                            shared_inject_dcfs.append(dcf)
                    
                    mean_shared_inject = np.mean(shared_inject_dcfs, axis=0) if shared_inject_dcfs else mean_baseline
                    shared_delta = mean_shared_inject - mean_baseline
                    shared_sel = compute_selectivity(shared_delta, target_idx)
                else:
                    shared_delta = np.zeros(8)
                    shared_sel = 0.0
                
                relation_results[f"scale_{scale}"] = {
                    "Bc_inject_target_delta": float(inject_delta[target_idx]),
                    "Bc_inject_selectivity": float(inject_sel),
                    "Bc_inject_dcf_delta": inject_delta.tolist(),
                    "Mc_inject_target_delta": float(shared_delta[target_idx]),
                    "Mc_inject_selectivity": float(shared_sel),
                    "Mc_inject_dcf_delta": shared_delta.tolist(),
                }
            
            cat_results[relation] = relation_results
            
            plog(f"    {relation}: Bc_sel@1.0={relation_results.get('scale_1.0',{}).get('Bc_inject_selectivity',0):.2f}, "
                  f"Mc_sel@1.0={relation_results.get('scale_1.0',{}).get('Mc_inject_selectivity',0):.2f}")
        
        results[cat_name] = {
            "best_layer": best_layer,
            "spec_norm": float(spec_norm),
            "shared_norm": float(shared_norm) if shared_vec is not None else 0,
            "relations": cat_results,
            "elapsed": round(time.time() - t0, 1),
        }
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


# ==================== Exp4: 异常竞争对token级解释 ====================
def exp4_anomalous_competition_explanation(model, tokenizer, device, model_name, W_U):
    """
    解释异常竞争释放对:
    1. food→vehicle (Qwen3: +6.74)
    2. animal→clothing (跨模型一致)
    
    方法:
    - 对每个类别做token级DCF分解
    - 分析food边界压制了哪些vehicle相关token
    - 分析animal边界压制了哪些clothing相关token
    """
    plog("=== Exp4: 异常竞争对token级解释 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    
    results = {}
    
    # 分析2个异常对
    anomalous_pairs = [
        ("food", "vehicle"),      # Qwen3: food移除→vehicle+6.74
        ("animal", "clothing"),   # 跨模型一致
    ]
    
    for removed_cat, released_cat in anomalous_pairs:
        best_layer = BEST_LAYERS[model_name][removed_cat]
        plog(f"  {removed_cat}→{released_cat} @ L{best_layer}...")
        t0 = time.time()
        
        # 获取removed_cat的specific方向
        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, removed_cat, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"    Skip: spec_norm too small")
            continue
        
        b_hat = spec_vec / spec_norm
        target_idx = cat_names.index(removed_cat)
        released_idx = cat_names.index(released_cat)
        
        # 1. Token级DCF分析
        # 对released_cat的8个对象，做边界移除，测量每个token的logit变化
        template = RELATION_TEMPLATES["kind_of"]
        released_objs = CATEGORIES[released_cat]
        
        # baseline logit向量
        token_level_changes = {}
        for obj in released_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            
            # baseline
            cap_base = {}
            h1 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap_base, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h1.remove()
            
            if "resid" not in cap_base:
                continue
            
            base_logits = (cap_base["resid"][0, pos].float().numpy() @ W_U.T)
            
            # remove B_c
            cap_remove = {}
            h2 = layers_list[best_layer].register_forward_hook(
                make_remove_hook(spec_vec, pos, scale=1.0))
            h3 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap_remove, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h2.remove()
            h3.remove()
            
            if "resid" not in cap_remove:
                continue
            
            remove_logits = (cap_remove["resid"][0, pos].float().numpy() @ W_U.T)
            
            # token级变化
            logit_diff = remove_logits - base_logits  # [vocab]
            
            # 找变化最大的tokens
            top_increase_idx = np.argsort(logit_diff)[-20:][::-1]
            top_decrease_idx = np.argsort(logit_diff)[:20]
            
            top_increase_tokens = [(tokenizer.decode([i]).strip(), float(logit_diff[i])) 
                                   for i in top_increase_idx if logit_diff[i] > 0.5]
            top_decrease_tokens = [(tokenizer.decode([i]).strip(), float(logit_diff[i])) 
                                    for i in top_decrease_idx if logit_diff[i] < -0.5]
            
            # released_cat family words的变化
            released_family = FAMILY_WORDS_8D[released_cat]
            released_family_changes = {}
            for w in released_family:
                tid = find_token_id(tokenizer, w)
                if tid is not None and tid < len(logit_diff):
                    released_family_changes[w] = float(logit_diff[tid])
            
            # removed_cat family words的变化
            removed_family = FAMILY_WORDS_8D[removed_cat]
            removed_family_changes = {}
            for w in removed_family:
                tid = find_token_id(tokenizer, w)
                if tid is not None and tid < len(logit_diff):
                    removed_family_changes[w] = float(logit_diff[tid])
            
            token_level_changes[obj] = {
                "released_family_changes": released_family_changes,
                "removed_family_changes": removed_family_changes,
                "top_increase": top_increase_tokens[:10],
                "top_decrease": top_decrease_tokens[:10],
            }
        
        # 2. 分析共享维度
        # 获取removed_cat和released_cat的specific方向
        released_spec_vec, released_spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, released_cat, best_layer
        )
        
        if released_spec_vec is not None and released_spec_norm > 1e-6:
            # 两个边界方向之间的角度
            cos_between = float(np.dot(spec_vec, released_spec_vec) / 
                                (spec_norm * released_spec_norm + 1e-10))
        else:
            cos_between = 0.0
        
        # 3. 共享属性方向分析
        # 检查removed_cat和released_cat是否共享某个属性方向
        # 例如: food和vehicle可能共享"贸易/运输"属性
        shared_attribute_tokens = {
            "commerce": ["buy", "sell", "price", "market", "store", "shop", "trade", "cost"],
            "transport": ["carry", "move", "travel", "ship", "deliver", "load", "route", "drive"],
            "location": ["place", "area", "room", "home", "city", "country", "outside", "inside"],
            "size": ["big", "small", "large", "tiny", "heavy", "light", "long", "short"],
        }
        
        shared_attr_analysis = {}
        for attr_name, attr_tokens in shared_attribute_tokens.items():
            attr_changes = {}
            for obj in CATEGORIES[removed_cat][:4]:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                
                cap_base = {}
                h1 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap_base, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h1.remove()
                if "resid" not in cap_base:
                    continue
                base_logits = cap_base["resid"][0, pos].float().numpy() @ W_U.T
                
                cap_remove = {}
                h2 = layers_list[best_layer].register_forward_hook(
                    make_remove_hook(spec_vec, pos, scale=1.0))
                h3 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap_remove, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h2.remove()
                h3.remove()
                if "resid" not in cap_remove:
                    continue
                remove_logits = cap_remove["resid"][0, pos].float().numpy() @ W_U.T
                
                logit_diff = remove_logits - base_logits
                for w in attr_tokens:
                    tid = find_token_id(tokenizer, w)
                    if tid is not None and tid < len(logit_diff):
                        if w not in attr_changes:
                            attr_changes[w] = []
                        attr_changes[w].append(float(logit_diff[tid]))
            
            # 平均变化
            avg_changes = {w: float(np.mean(v)) for w, v in attr_changes.items() if v}
            if avg_changes:
                shared_attr_analysis[attr_name] = avg_changes
        
        results[f"{removed_cat}->{released_cat}"] = {
            "best_layer": best_layer,
            "cos_between_boundaries": cos_between,
            "token_level_changes": token_level_changes,
            "shared_attribute_analysis": shared_attr_analysis,
            "elapsed": round(time.time() - t0, 1),
        }
        
        plog(f"    cos(boundaries)={cos_between:.3f}, "
              f"shared_attrs={list(shared_attr_analysis.keys())}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    plog(f"Phase 484: model={model_name}, round={round_num}")
    plog(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        plog(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, {info.n_layers} layers, d_model={info.d_model}")
    
    # 加载W_U
    W_U = get_W_U(model, model_name)
    plog(f"W_U: shape={W_U.shape}")
    
    # 运行实验
    all_results = {"phase": 484, "round": round_num, "model": model_name,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    
    try:
        # Exp1: 边界写入场重构
        plog("\n" + "="*60)
        exp1_results = exp1_boundary_writer_reconstruction(model, tokenizer, device, model_name, W_U)
        all_results["exp1_writer_reconstruction"] = exp1_results
        
        # Exp2: 写入场因果测试
        plog("\n" + "="*60)
        exp2_results = exp2_writer_causal_test(model, tokenizer, device, model_name, W_U)
        all_results["exp2_writer_causal_test"] = exp2_results
        
        # Exp3: 关系槽位读出
        plog("\n" + "="*60)
        exp3_results = exp3_relation_slot_readout(model, tokenizer, device, model_name, W_U)
        all_results["exp3_relation_slot_readout"] = exp3_results
        
        # Exp4: 异常竞争对解释
        plog("\n" + "="*60)
        exp4_results = exp4_anomalous_competition_explanation(model, tokenizer, device, model_name, W_U)
        all_results["exp4_anomalous_competition"] = exp4_results
        
    except Exception as e:
        plog(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_results["error"] = str(e)
    
    # 保存
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj
    
    out_path = f"results/glm5/phase484_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
    plog(f"Results saved to {out_path}")
    
    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    
    plog(f"Phase 484 complete for {model_name}")


if __name__ == "__main__":
    main()
