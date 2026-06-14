"""
Phase 485: Attention边界写入器 + MLP幅度闭环 + 关系小尺度 + DS7B格式去除
============================================================================

核心目标(从Phase 484推进):
1. Exp1: Attention头边界写入器定位 — 找fruit/animal边界的真正写入器
2. Exp2: MLP集中边界幅度闭环 — 扩大k值复现方向级remove的幅度
3. Exp3: 关系槽位小尺度测试 — scale=0.05~0.5验证B_c关系不变性
4. Exp4: DS7B格式子空间去除 — 修复food/plant边界污染

关键问题:
- Phase 484发现Qwen3 fruit/animal边界的MLP消融cos_remove为负，
  说明MLP不是主要写入器，需测Attention头
- Qwen3 clothing k=5消融只复现-7.32/-34.18=21%幅度，需扩大k
- B_c注入delta跨关系不变可能因scale=1.0太强
- DS7B food移除非选择性，需去除格式子空间

用法:
  python tests/glm5/phase485_attn_writer_and_amplitude.py qwen3 1
  python tests/glm5/phase485_attn_writer_and_amplitude.py glm4 1
  python tests/glm5/phase485_attn_writer_and_amplitude.py deepseek7b 1
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

RELATION_TEMPLATES = {
    "kind_of":   "The {obj} is a kind of",
    "used_for":  "The {obj} is used for",
    "found_in":  "The {obj} is found in",
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
    
    # 检查层分配
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


def safe_load_weight(model, model_name, layer_idx, component_path):
    """安全加载权重，处理meta device问题"""
    layers_list = get_layers(model)
    layer = layers_list[layer_idx]
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


# ==================== 获取特定方向 ====================
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


# ==================== Exp1: Attention头边界写入器定位 ====================
def exp1_attn_writer_localization(model, tokenizer, device, model_name, W_U, n_test_obj=4):
    """
    对Qwen3 fruit L32和animal L33，定位Attention头对B_c的贡献
    
    方法:
    1. 对每个Attention头，捕获其输出 (W_o @ softmax(...) @ V)
    2. 计算每个头输出与B_c的cos和对B_c的投影
    3. 头消融测试: 将top-k头输出清零，测量DCF变化
    4. 与Phase 484的MLP消融结果对比
    """
    plog("=== Exp1: Attention头边界写入器定位 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    
    # 关键测试类别: Phase 484中MLP非主导的类别
    if model_name == "qwen3":
        test_cats = ["fruit", "animal", "clothing"]  # clothing做对照
    elif model_name == "glm4":
        test_cats = ["animal", "clothing", "fruit"]  # fruit做对照
    else:
        test_cats = ["fruit", "clothing", "animal"]
    
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
            plog(f"    Skip {cat_name}: spec_norm too small ({spec_norm:.4f})")
            continue
        
        b_hat = spec_vec / spec_norm
        target_idx = cat_names.index(cat_name)
        
        # ---- 获取Attention头数 ----
        layer = layers_list[best_layer]
        sa = layer.self_attn
        # 从W_o权重推断头数
        W_o = sa.o_proj.weight.detach().cpu().float().numpy()  # [d_model, d_model]
        d_model = W_o.shape[0]
        n_heads = info.d_model  # will be corrected below
        # 从config获取n_heads
        if hasattr(model.config, 'num_attention_heads'):
            n_heads = model.config.num_attention_heads
        elif hasattr(model.config, 'n_heads'):
            n_heads = model.config.n_heads
        else:
            n_heads = d_model // 128  # 常见假设
        
        head_dim = d_model // n_heads
        plog(f"    n_heads={n_heads}, head_dim={head_dim}, d_model={d_model}")
        
        # ---- 方法1: 每个头输出与B_c的投影 ----
        # 在forward时捕获Attention输出 (attn_output before W_o)
        # 然后计算每个头的贡献: h_i = W_o[:, i*head_dim:(i+1)*head_dim] @ attn_out_i
        
        # 获取类别内和邻居的残差流
        template = RELATION_TEMPLATES["kind_of"]
        train_objs = CATEGORIES_TRAIN[cat_name]
        neighbor_cats = BEST_NEIGHBORS[model_name][cat_name]
        
        # ---- 捕获类别内样本的attn输出 ----
        cat_attn_outs = []  # 每个样本: [n_heads, head_dim] 的attn输出
        cat_resids_attn = []  # attn子层后的残差流
        
        for obj in train_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            
            # 捕获attn子层输出
            cap_attn = {}
            cap_resid_after_attn = {}
            done_attn = [False]
            done_resid = [False]
            
            def make_attn_out_hook(key, position):
                def hook_fn(module, inp, output):
                    if not done_attn[0]:
                        if isinstance(output, tuple):
                            cap_attn[key] = output[0].detach().float().cpu()
                        else:
                            cap_attn[key] = output.detach().float().cpu()
                        done_attn[0] = True
                return hook_fn
            
            def make_resid_after_attn_hook(key, position):
                def hook_fn(module, inp, output):
                    if not done_resid[0]:
                        if isinstance(output, tuple):
                            cap_resid_after_attn[key] = output[0].detach().float().cpu()
                        else:
                            cap_resid_after_attn[key] = output.detach().float().cpu()
                        done_resid[0] = True
                return hook_fn
            
            # hook at self_attn output (after W_o projection)
            h1 = layer.self_attn.register_forward_hook(make_attn_out_hook("attn_out", pos))
            # hook at layer output (after residual add)
            h2 = layer.register_forward_hook(make_resid_after_attn_hook("resid", pos))
            
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            
            h1.remove()
            h2.remove()
            
            if "attn_out" in cap_attn:
                attn_full = cap_attn["attn_out"][0, pos].numpy()  # [d_model]
                cat_attn_outs.append(attn_full)
            if "resid" in cap_resid_after_attn:
                cat_resids_attn.append(cap_resid_after_attn["resid"][0, pos].numpy())
        
        # ---- 捕获邻居类别样本 ----
        neigh_attn_outs = []
        for nc in neighbor_cats:
            for obj in CATEGORIES_TRAIN[nc][:2]:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                
                cap_attn = {}
                done_attn = [False]
                def make_attn_hook2(key, position):
                    def hook_fn(module, inp, output):
                        if not done_attn[0]:
                            if isinstance(output, tuple):
                                cap_attn[key] = output[0].detach().float().cpu()
                            else:
                                cap_attn[key] = output.detach().float().cpu()
                            done_attn[0] = True
                    return hook_fn
                
                h1 = layer.self_attn.register_forward_hook(make_attn_hook2("attn_out", pos))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h1.remove()
                
                if "attn_out" in cap_attn:
                    neigh_attn_outs.append(cap_attn["attn_out"][0, pos].numpy())
        
        if not cat_attn_outs:
            plog(f"    Skip {cat_name}: no attn outputs captured")
            continue
        
        # ---- 计算每个Attention头对B_c的贡献 ----
        # attn_out = W_o @ concat(head_0, head_1, ..., head_{n-1})
        # 拆分attn_out为每个头的贡献:
        # W_o: [d_model, d_model], 按头拆分为 [d_model, n_heads, head_dim]
        # head_i的d_model贡献 = W_o[:, i*head_dim:(i+1)*head_dim] @ head_output_i
        
        # 差异attn输出
        mean_cat_attn = np.mean(cat_attn_outs, axis=0)  # [d_model]
        mean_neigh_attn = np.mean(neigh_attn_outs, axis=0) if neigh_attn_outs else np.zeros(d_model)
        diff_attn = mean_cat_attn - mean_neigh_attn  # [d_model]
        
        # 拆分差异attn输出为每头贡献
        # diff_attn = W_o @ diff_concat_heads
        # 我们需要 diff_concat_heads 使得 W_o @ diff_concat_heads = diff_attn
        # diff_concat_heads = W_o^+ @ diff_attn (pseudo-inverse)
        # 但更直接: 用W_o的行空间分解
        
        # 方法: 直接计算每个头的attn输出在d_model空间中的贡献
        # diff_attn本身是d_model维的，但我们不知道每头的单独贡献
        # 简化: 用W_o的列块来分配
        # head_i对diff_attn的贡献 = W_o[:, i*head_dim:(i+1)*head_dim] @ diff_concat_heads[i*head_dim:(i+1)*head_dim]
        # 但我们不知道diff_concat_heads，只知道W_o @ diff_concat_heads = diff_attn
        
        # 更实际的方法: 计算diff_attn在b_hat方向的投影
        # 然后用W_o的列块来估计每个头的贡献
        
        # 方法1: 简化 — 将diff_attn按d_model维度分成n_heads份
        # 每份对应一个头的输出维度，但这不对因为W_o混合了所有头
        
        # 方法2: 用W_o的伪逆还原每头的concat输出，再计算每头对B_c的贡献
        # diff_concat = W_o^+ @ diff_attn  [d_model]
        # 每头的贡献: head_i_contrib_to_Bc = (diff_concat[i*hd:(i+1)*hd] @ W_o[i*hd:(i+1)*hd, :]) @ b_hat
        # 但W_o^+可能不稳定
        
        # 方法3 (最实用): 直接测量每个头的W_o列块对B_c方向的表达力
        # head_i对B_c方向的表达力 = ||W_o[:, i*hd:(i+1)*hd]^T @ b_hat||
        # 差异贡献 = diff_attn在W_o[:, i*hd:(i+1)*hd]子空间上的投影 × b_hat方向
        
        # 方法4 (最终): 用output_attentions=True获取每头注意力模式
        # 这需要修改forward调用
        
        # ---- 采用方法3: 按W_o列块分析 ----
        head_contributions = []  # [n_heads] — 每头对B_c的贡献
        
        for h_idx in range(n_heads):
            start = h_idx * head_dim
            end = start + head_dim
            if end > d_model:
                break
            # W_o的列块 [d_model, head_dim]
            W_o_block = W_o[:, start:end]  # [d_model, head_dim]
            # 该头对B_c方向的表达力: b_hat^T @ W_o @ head_out = b_hat^T @ W_o_block @ head_out_block
            # 用W_o_block^T @ b_hat得到每头维度对B_c的投影权重
            proj_weights = W_o_block.T @ b_hat  # [head_dim]
            head_importance = float(np.linalg.norm(proj_weights))
            head_contributions.append(head_importance)
        
        head_contributions = np.array(head_contributions)
        sorted_heads = np.argsort(head_contributions)[::-1]
        
        plog(f"    Top-5 heads by B_c projection: {sorted_heads[:5].tolist()}, "
              f"contribs: {head_contributions[sorted_heads[:5]].tolist()}")
        
        # ---- 方法4: 直接测量attn输出的d_model差异在B_c方向的投影 ----
        cos_diff_attn_bc = float(np.dot(diff_attn, b_hat) / 
                                  (np.linalg.norm(diff_attn) + 1e-10))
        proj_diff_attn_bc = float(np.dot(diff_attn, b_hat))
        attn_norm = float(np.linalg.norm(diff_attn))
        
        plog(f"    Attn diff: cos(Bc)={cos_diff_attn_bc:.3f}, proj={proj_diff_attn_bc:.2f}, norm={attn_norm:.2f}")
        
        # ---- Attention头消融测试 ----
        # 方法: 捕获attn子层输出，然后从残差流中减去它，测量DCF变化
        
        # Baseline DCF
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
            h1 = layers_list[best_layer].register_forward_hook(
                make_remove_hook(spec_vec, pos, scale=1.0))
            cap = {}
            h2 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h1.remove()
            h2.remove()
            if "resid" in cap:
                dcf = logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer)
                remove_dcfs.append(dcf)
        
        mean_remove = np.mean(remove_dcfs, axis=0) if remove_dcfs else mean_baseline
        remove_delta = mean_remove - mean_baseline
        direction_remove_target = float(remove_delta[target_idx])
        
        plog(f"    Direction remove: target_D={direction_remove_target:.2f}")
        
        # ---- Attn子层消融(整层) ----
        # 方法: 捕获attn子层输出，从残差流中减去
        attn_ablate_dcfs = []
        for obj in test_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            
            # 捕获attn输出和最终残差
            cap_attn_out = {}
            cap_resid = {}
            done = [False, False]
            
            def make_attn_capture_hook(key, position):
                def hook_fn(module, inp, output):
                    if not done[0]:
                        if isinstance(output, tuple):
                            cap_attn_out[key] = output[0].detach().float().cpu()
                        else:
                            cap_attn_out[key] = output.detach().float().cpu()
                        done[0] = True
                return hook_fn
            
            # 在attn子层捕获输出
            h1 = layer.self_attn.register_forward_hook(make_attn_capture_hook("attn", pos))
            # 在整个layer输出捕获残差
            h2 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap_resid, "resid"))
            
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            
            h1.remove()
            h2.remove()
            
            if "attn" not in cap_attn_out or "resid" not in cap_resid:
                continue
            
            # 从残差流中减去attn输出
            # resid_after_layer = resid_after_attn + mlp_output
            # 如果减去attn输出: new_resid = resid - attn_output
            attn_output = cap_attn_out["attn"][0, pos].numpy()  # [d_model]
            resid = cap_resid["resid"][0, pos].numpy()  # [d_model]
            ablated_resid = resid - attn_output
            
            dcf = logit_lens_dcf(ablated_resid, W_U, tokenizer)
            attn_ablate_dcfs.append(dcf)
        
        if attn_ablate_dcfs:
            mean_attn_ablate = np.mean(attn_ablate_dcfs, axis=0)
            attn_ablate_delta = mean_attn_ablate - mean_baseline
            cos_attn_remove = float(np.dot(attn_ablate_delta, remove_delta) / 
                                     (np.linalg.norm(attn_ablate_delta) * np.linalg.norm(remove_delta) + 1e-10))
            attn_ablate_target = float(attn_ablate_delta[target_idx])
            
            plog(f"    Full attn ablation: target_D={attn_ablate_target:.2f}, "
                  f"cos_remove={cos_attn_remove:.3f}")
        else:
            attn_ablate_target = 0
            cos_attn_remove = 0
            attn_ablate_delta = np.zeros(8)
        
        # ---- MLP子层消融(对比) ----
        # 捕获MLP输出并从残差流中减去
        mlp_ablate_dcfs = []
        for obj in test_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            
            cap_mlp_out = {}
            cap_resid = {}
            
            def make_mlp_capture_hook(key, position):
                done = [False]
                def hook_fn(module, inp, output):
                    if not done[0]:
                        if isinstance(output, tuple):
                            cap_mlp_out[key] = output[0].detach().float().cpu()
                        else:
                            cap_mlp_out[key] = output.detach().float().cpu()
                        done[0] = True
                return hook_fn
            
            h1 = layer.mlp.register_forward_hook(make_mlp_capture_hook("mlp", pos))
            h2 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap_resid, "resid"))
            
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            
            h1.remove()
            h2.remove()
            
            if "mlp" not in cap_mlp_out or "resid" not in cap_resid:
                continue
            
            mlp_output = cap_mlp_out["mlp"][0, pos].numpy()
            resid = cap_resid["resid"][0, pos].numpy()
            ablated_resid = resid - mlp_output
            
            dcf = logit_lens_dcf(ablated_resid, W_U, tokenizer)
            mlp_ablate_dcfs.append(dcf)
        
        if mlp_ablate_dcfs:
            mean_mlp_ablate = np.mean(mlp_ablate_dcfs, axis=0)
            mlp_ablate_delta = mean_mlp_ablate - mean_baseline
            cos_mlp_remove = float(np.dot(mlp_ablate_delta, remove_delta) / 
                                    (np.linalg.norm(mlp_ablate_delta) * np.linalg.norm(remove_delta) + 1e-10))
            mlp_ablate_target = float(mlp_ablate_delta[target_idx])
            
            plog(f"    Full MLP ablation: target_D={mlp_ablate_target:.2f}, "
                  f"cos_remove={cos_mlp_remove:.3f}")
        else:
            mlp_ablate_target = 0
            cos_mlp_remove = 0
            mlp_ablate_delta = np.zeros(8)
        
        # ---- Attn+MLP总消融 ----
        total_attn_target = attn_ablate_target
        total_mlp_target = mlp_ablate_target
        combined_target = total_attn_target + total_mlp_target
        coverage_ratio = combined_target / (direction_remove_target + 1e-10)
        
        plog(f"    Coverage: attn={total_attn_target:.2f} + mlp={total_mlp_target:.2f} = "
              f"{combined_target:.2f} / direction={direction_remove_target:.2f} = {coverage_ratio:.2%}")
        
        elapsed = time.time() - t0
        plog(f"    Done in {elapsed:.1f}s")
        
        results[cat_name] = {
            "best_layer": best_layer,
            "spec_norm": float(spec_norm),
            "n_heads": int(n_heads),
            "head_dim": int(head_dim),
            "cos_diff_attn_bc": cos_diff_attn_bc,
            "proj_diff_attn_bc": proj_diff_attn_bc,
            "attn_diff_norm": attn_norm,
            "top5_heads": sorted_heads[:5].tolist(),
            "top5_head_contribs": head_contributions[sorted_heads[:5]].tolist(),
            "direction_remove_target": direction_remove_target,
            "full_attn_ablation": {
                "target_delta": attn_ablate_target,
                "cos_with_direction_remove": cos_attn_remove,
                "dcf_delta": attn_ablate_delta.tolist(),
            },
            "full_mlp_ablation": {
                "target_delta": mlp_ablate_target,
                "cos_with_direction_remove": cos_mlp_remove,
                "dcf_delta": mlp_ablate_delta.tolist(),
            },
            "coverage_ratio": coverage_ratio,
            "attn_target": total_attn_target,
            "mlp_target": total_mlp_target,
            "elapsed": round(elapsed, 1),
        }
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


# ==================== Exp2: MLP集中边界幅度闭环 ====================
def exp2_mlp_amplitude_closure(model, tokenizer, device, model_name, W_U, n_test_obj=6):
    """
    扩大k值消融，找到复现方向级remove幅度70%+的写入场
    
    关键: Phase 484中k=5只复现21%幅度，需扩大到k=100-500
    """
    plog("=== Exp2: MLP集中边界幅度闭环 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    
    # Phase 484中MLP集中型的类别
    if model_name == "qwen3":
        test_cats = ["clothing", "fruit", "animal"]
    elif model_name == "glm4":
        test_cats = ["fruit", "clothing", "animal"]
    else:
        test_cats = ["fruit", "clothing", "animal"]
    
    results = {}
    
    for cat_name in test_cats:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name} @ L{best_layer}...")
        t0 = time.time()
        
        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"    Skip {cat_name}: spec_norm too small")
            continue
        
        b_hat = spec_vec / spec_norm
        target_idx = cat_names.index(cat_name)
        
        W_down = safe_load_weight(model, model_name, best_layer, "mlp.down_proj")
        if W_down is None:
            plog(f"    Skip {cat_name}: Cannot load W_down")
            continue
        
        # 计算神经元边界贡献
        y = W_down.T @ b_hat  # [intermediate]
        template = RELATION_TEMPLATES["kind_of"]
        train_objs = CATEGORIES_TRAIN[cat_name]
        
        # 获取MLP激活
        cat_acts = []
        for obj in train_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            
            cap_mlp = {}
            done = [False]
            def make_hook(key):
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
        
        mean_act = np.mean(cat_acts, axis=0)
        neuron_contrib = mean_act * y
        abs_contrib = np.abs(neuron_contrib)
        sorted_idx = np.argsort(abs_contrib)[::-1]
        
        # Baseline DCF (增加测试对象)
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
            continue
        mean_baseline = np.mean(baseline_dcfs, axis=0)
        
        # 方向级remove
        remove_dcfs = []
        for obj in test_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            h1 = layers_list[best_layer].register_forward_hook(make_remove_hook(spec_vec, pos, scale=1.0))
            cap = {}
            h2 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h1.remove()
            h2.remove()
            if "resid" in cap:
                dcf = logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer)
                remove_dcfs.append(dcf)
        
        mean_remove = np.mean(remove_dcfs, axis=0) if remove_dcfs else mean_baseline
        remove_delta = mean_remove - mean_baseline
        direction_remove_target = float(remove_delta[target_idx])
        
        plog(f"    Direction remove target_D={direction_remove_target:.2f}")
        
        # 扩大k值消融: k=5,10,20,50,100,200,500
        test_ks = [5, 10, 20, 50, 100, 200, 500]
        ablation_results = {}
        
        for k in test_ks:
            if k > len(sorted_idx):
                continue
            top_k_idx = sorted_idx[:k]
            
            ablate_dcfs = []
            for obj in test_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                
                # 捕获残差流和MLP激活
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
                
                h1 = layers_list[best_layer].register_forward_hook(make_resid_hook("resid"))
                h2 = layers_list[best_layer].mlp.down_proj.register_forward_hook(make_mlp_act_hook("mlp_act", pos))
                
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                
                h1.remove()
                h2.remove()
                
                if "resid" not in cap_resid or "mlp_act" not in cap_mlp_act:
                    continue
                
                baseline_resid = cap_resid["resid"][0, pos].numpy()
                mlp_act = cap_mlp_act["mlp_act"][0, pos].numpy()
                
                # 从残差流中减去top-k神经元的MLP贡献
                ablated_contribution = np.zeros(info.d_model)
                for j in top_k_idx:
                    ablated_contribution += mlp_act[j] * W_down[:, j]
                
                ablated_resid = baseline_resid - ablated_contribution
                dcf = logit_lens_dcf(ablated_resid, W_U, tokenizer)
                ablate_dcfs.append(dcf)
            
            if not ablate_dcfs:
                continue
            
            mean_ablate = np.mean(ablate_dcfs, axis=0)
            ablate_delta = mean_ablate - mean_baseline
            
            cos_with_remove = float(np.dot(ablate_delta, remove_delta) / 
                                     (np.linalg.norm(ablate_delta) * np.linalg.norm(remove_delta) + 1e-10))
            
            ablate_target = float(ablate_delta[target_idx])
            amplitude_ratio = abs(ablate_target / (direction_remove_target + 1e-10))
            
            # 竞争释放
            max_competitor_release = float(max(ablate_delta[i] for i in range(8) if i != target_idx))
            
            plog(f"    k={k}: target_D={ablate_target:.2f}, cos={cos_with_remove:.3f}, "
                  f"amp_ratio={amplitude_ratio:.2%}, max_release={max_competitor_release:.2f}")
            
            ablation_results[f"k={k}"] = {
                "target_delta": ablate_target,
                "cos_with_direction_remove": cos_with_remove,
                "amplitude_ratio": amplitude_ratio,
                "max_competitor_release": max_competitor_release,
                "dcf_delta": ablate_delta.tolist(),
            }
        
        elapsed = time.time() - t0
        plog(f"    Done in {elapsed:.1f}s")
        
        results[cat_name] = {
            "best_layer": best_layer,
            "spec_norm": float(spec_norm),
            "direction_remove_target": direction_remove_target,
            "ablation_results": ablation_results,
            "elapsed": round(elapsed, 1),
        }
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


# ==================== Exp3: 关系槽位小尺度测试 ====================
def exp3_relation_small_scale(model, tokenizer, device, model_name, W_U, n_test_obj=4):
    """
    用scale=0.05~0.5测试B_c注入，验证关系不变性是否在小scale下仍成立
    
    Phase 484发现scale=1.0时delta跨关系完全不变。
    关键问题: 这是真实结构特征，还是强注入的artifact?
    """
    plog("=== Exp3: 关系槽位小尺度测试 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    
    # 3个类别 × 3个关系 × 5个scale
    test_cats = ["fruit", "clothing", "animal"]
    relations = ["kind_of", "used_for", "found_in"]
    scales = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    
    results = {}
    
    for cat_name in test_cats:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name} @ L{best_layer}...")
        t0 = time.time()
        
        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            continue
        
        target_idx = cat_names.index(cat_name)
        
        rel_results = {}
        
        for rel_name in relations:
            template = RELATION_TEMPLATES[rel_name]
            test_objs = CATEGORIES[cat_name][:n_test_obj]
            
            scale_results = {}
            
            for scale in scales:
                # 注入强度 = spec_norm * scale
                inject_vec = spec_vec * scale  # 不是归一化后*scale
                ivec = torch.tensor(inject_vec, dtype=torch.float32)
                
                inject_dcfs = []
                for obj in test_objs:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                    
                    h = layers_list[best_layer].register_forward_hook(
                        make_inject_hook(ivec, pos))
                    cap = {}
                    h2 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    h.remove()
                    h2.remove()
                    
                    if "resid" in cap:
                        dcf = logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer)
                        inject_dcfs.append(dcf)
                
                if not inject_dcfs:
                    continue
                
                mean_inject = np.mean(inject_dcfs, axis=0)
                
                # 也需要baseline
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
                    continue
                
                mean_baseline = np.mean(baseline_dcfs, axis=0)
                delta = mean_inject - mean_baseline
                
                target_dcf = float(delta[target_idx])
                selectivity = compute_selectivity(delta, target_idx)
                
                scale_results[f"scale_{scale}"] = {
                    "baseline_target": float(mean_baseline[target_idx]),
                    "inject_target": float(mean_inject[target_idx]),
                    "target_delta": target_dcf,
                    "selectivity": selectivity,
                    "dcf_delta": delta.tolist(),
                }
            
            rel_results[rel_name] = scale_results
        
        # 计算跨关系delta一致性
        consistency = {}
        for scale_key in [f"scale_{s}" for s in scales]:
            deltas = []
            for rel_name in relations:
                if scale_key in rel_results.get(rel_name, {}):
                    deltas.append(rel_results[rel_name][scale_key]["target_delta"])
            if len(deltas) >= 2:
                delta_range = max(deltas) - min(deltas)
                delta_mean = sum(deltas) / len(deltas)
                consistency[scale_key] = {
                    "delta_mean": delta_mean,
                    "delta_range": delta_range,
                    "relative_range": delta_range / (abs(delta_mean) + 1e-10),
                    "deltas": deltas,
                }
        
        elapsed = time.time() - t0
        plog(f"    Done in {elapsed:.1f}s")
        
        results[cat_name] = {
            "best_layer": best_layer,
            "spec_norm": float(spec_norm),
            "relations": rel_results,
            "cross_relation_consistency": consistency,
            "elapsed": round(elapsed, 1),
        }
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


# ==================== Exp4: DS7B格式子空间去除 ====================
def exp4_ds7b_format_removal(model, tokenizer, device, model_name, W_U, n_test_obj=4):
    """
    DS7B的food/plant边界混入格式控制信号，导致移除非选择性。
    
    方法:
    1. 识别格式子空间: 用不同格式模板提取共同方向
    2. 从类别方向中减去格式子空间
    3. 重测边界移除和竞争释放
    """
    plog("=== Exp4: DS7B格式子空间去除 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    
    # 格式模板: 同一个对象在不同格式下激活的共享方向就是格式方向
    format_templates = {
        "kind_of": "The {obj} is a kind of",
        "used_for": "The {obj} is used for",
        "describe": "Describe the {obj}",
        "question": "What is a {obj}?",
    }
    
    # 测试类别
    test_cats = ["food", "fruit", "animal", "clothing"]
    
    results = {}
    
    for cat_name in test_cats:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name} @ L{best_layer}...")
        t0 = time.time()
        
        # ---- Step 1: 提取格式子空间 ----
        # 用同一组对象在不同模板下的残差流，找共同方向
        format_resids = {}
        for fmt_name, fmt_template in format_templates.items():
            resids = []
            for obj in CATEGORIES_TRAIN[cat_name]:
                prompt = fmt_template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                cap = {}
                h = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                if "resid" in cap:
                    resids.append(cap["resid"][0, pos].numpy())
            if resids:
                format_resids[fmt_name] = np.mean(resids, axis=0)
        
        if len(format_resids) < 2:
            plog(f"    Skip {cat_name}: insufficient format resids")
            continue
        
        # 格式子空间 = 不同模板间差异的主方向
        fmt_vecs = list(format_resids.values())
        fmt_diffs = []
        for i in range(len(fmt_vecs)):
            for j in range(i+1, len(fmt_vecs)):
                fmt_diffs.append(fmt_vecs[i] - fmt_vecs[j])
        
        # SVD提取格式子空间
        fmt_diff_matrix = np.array(fmt_diffs)  # [n_diffs, d_model]
        U_fmt, S_fmt, Vt_fmt = np.linalg.svd(fmt_diff_matrix, full_matrices=False)
        # 取前3个主方向作为格式子空间
        n_format_dims = min(3, len(S_fmt))
        format_subspace = U_fmt[:, :n_format_dims]  # [d_model, n_format_dims] (这是错误的维度)
        # 正确: Vt_fmt[:n_format_dims] 是 [n_format_dims, d_model] — 格式主方向
        format_basis = Vt_fmt[:n_format_dims]  # [n_format_dims, d_model]
        
        format_energy = S_fmt[:n_format_dims].tolist()
        plog(f"    Format subspace: {n_format_dims} dims, energy={format_energy}")
        
        # ---- Step 2: 获取原始类别方向和边界 ----
        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            continue
        
        b_hat = spec_vec / spec_norm
        target_idx = cat_names.index(cat_name)
        
        # 计算B_c在格式子空间上的投影
        fmt_proj = format_basis.T @ (format_basis @ b_hat)  # [d_model]
        fmt_proj_norm = float(np.linalg.norm(fmt_proj))
        cos_bc_fmt = float(np.dot(b_hat, fmt_proj / (fmt_proj_norm + 1e-10)))
        
        plog(f"    cos(Bc, format_proj)={cos_bc_fmt:.3f}, fmt_proj_norm={fmt_proj_norm:.4f}")
        
        # 清洗后的B_c: 去除格式投影
        b_hat_clean = b_hat - fmt_proj
        b_hat_clean_norm = float(np.linalg.norm(b_hat_clean))
        if b_hat_clean_norm > 1e-10:
            b_hat_clean = b_hat_clean / b_hat_clean_norm
        
        cos_clean_original = float(np.dot(b_hat_clean, b_hat))
        plog(f"    cos(Bc_clean, Bc_original)={cos_clean_original:.3f}")
        
        # ---- Step 3: 对比原始vs清洗后B_c的边界移除效果 ----
        template = RELATION_TEMPLATES["kind_of"]
        test_objs = CATEGORIES[cat_name][:n_test_obj]
        
        # Baseline
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
            continue
        mean_baseline = np.mean(baseline_dcfs, axis=0)
        
        # 原始B_c移除
        orig_remove_dcfs = []
        for obj in test_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            h = layers_list[best_layer].register_forward_hook(make_remove_hook(spec_vec, pos, scale=1.0))
            cap = {}
            h2 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            h2.remove()
            if "resid" in cap:
                dcf = logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer)
                orig_remove_dcfs.append(dcf)
        
        mean_orig_remove = np.mean(orig_remove_dcfs, axis=0) if orig_remove_dcfs else mean_baseline
        orig_remove_delta = mean_orig_remove - mean_baseline
        
        # 清洗后B_c移除
        clean_spec_vec = b_hat_clean * spec_norm  # 用清洗后方向×原始范数
        clean_remove_dcfs = []
        for obj in test_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            h = layers_list[best_layer].register_forward_hook(make_remove_hook(clean_spec_vec, pos, scale=1.0))
            cap = {}
            h2 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            h2.remove()
            if "resid" in cap:
                dcf = logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer)
                clean_remove_dcfs.append(dcf)
        
        mean_clean_remove = np.mean(clean_remove_dcfs, axis=0) if clean_remove_dcfs else mean_baseline
        clean_remove_delta = mean_clean_remove - mean_baseline
        
        # 对比
        orig_target_delta = float(orig_remove_delta[target_idx])
        clean_target_delta = float(clean_remove_delta[target_idx])
        
        # 选择性: |target| / max|other|
        orig_selectivity = compute_selectivity(orig_remove_delta, target_idx)
        clean_selectivity = compute_selectivity(clean_remove_delta, target_idx)
        
        # 竞争释放
        orig_max_release = float(max(orig_remove_delta[i] for i in range(8) if i != target_idx))
        clean_max_release = float(max(clean_remove_delta[i] for i in range(8) if i != target_idx))
        
        plog(f"    Original: target_D={orig_target_delta:.2f}, sel={orig_selectivity:.2f}, max_release={orig_max_release:.2f}")
        plog(f"    Clean:    target_D={clean_target_delta:.2f}, sel={clean_selectivity:.2f}, max_release={clean_max_release:.2f}")
        
        elapsed = time.time() - t0
        plog(f"    Done in {elapsed:.1f}s")
        
        results[cat_name] = {
            "best_layer": best_layer,
            "format_energy": format_energy,
            "cos_bc_format": cos_bc_fmt,
            "cos_clean_original": cos_clean_original,
            "n_format_dims": n_format_dims,
            "original_remove": {
                "target_delta": orig_target_delta,
                "selectivity": orig_selectivity,
                "max_competitor_release": orig_max_release,
                "dcf_delta": orig_remove_delta.tolist(),
            },
            "clean_remove": {
                "target_delta": clean_target_delta,
                "selectivity": clean_selectivity,
                "max_competitor_release": clean_max_release,
                "dcf_delta": clean_remove_delta.tolist(),
            },
            "improvement": {
                "selectivity_ratio": clean_selectivity / (orig_selectivity + 0.01),
                "target_preservation": clean_target_delta / (orig_target_delta + 1e-10),
                "release_reduction": (orig_max_release - clean_max_release) / (abs(orig_max_release) + 1e-10),
            },
            "elapsed": round(elapsed, 1),
        }
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


# ==================== 主程序 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    plog(f"Phase 485: {model_name}, round={round_num}")
    plog(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        plog(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    plog(f"Model: {info.model_class}, layers={info.n_layers}, d_model={info.d_model}")
    plog(f"W_U shape: {W_U.shape}")
    
    all_results = {
        "phase": 485,
        "round": round_num,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Exp1: Attention头边界写入器
    try:
        exp1 = exp1_attn_writer_localization(model, tokenizer, device, model_name, W_U)
        all_results["exp1_attn_writer"] = exp1
    except Exception as e:
        plog(f"Exp1 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_attn_writer"] = {"error": str(e)}
    
    # Exp2: MLP幅度闭环
    try:
        exp2 = exp2_mlp_amplitude_closure(model, tokenizer, device, model_name, W_U)
        all_results["exp2_mlp_amplitude"] = exp2
    except Exception as e:
        plog(f"Exp2 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_mlp_amplitude"] = {"error": str(e)}
    
    # Exp3: 关系小尺度测试
    try:
        exp3 = exp3_relation_small_scale(model, tokenizer, device, model_name, W_U)
        all_results["exp3_relation_small_scale"] = exp3
    except Exception as e:
        plog(f"Exp3 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_relation_small_scale"] = {"error": str(e)}
    
    # Exp4: DS7B格式子空间去除
    try:
        exp4 = exp4_ds7b_format_removal(model, tokenizer, device, model_name, W_U)
        all_results["exp4_format_removal"] = exp4
    except Exception as e:
        plog(f"Exp4 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp4_format_removal"] = {"error": str(e)}
    
    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase485_{model_name}_r{round_num}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    plog(f"Results saved to {out_path}")
    
    # 释放模型
    release_model(model)
    plog("Done!")


if __name__ == "__main__":
    main()
