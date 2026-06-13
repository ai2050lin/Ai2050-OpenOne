"""
Phase 483: 类别边界写入器定位 + 竞争释放图谱 + 最佳层位成因
================================================================

核心目标:
1. Exp1: 边界写入器定位 — 找到写入类别边界残差的MLP神经元
2. Exp2: 竞争释放图谱 — 全8类别×8类别边界移除矩阵
3. Exp3: 最佳层位成因 — 类别边界在不同层的形成过程

关键创新:
- 神经元级分析: 不只是方向级, 找到具体哪些神经元写入边界
- 全竞争矩阵: 不只看5个类别, 8个类别全部测试
- 层位形成曲线: 解释为什么不同类别在不同层形成边界

用法:
  python tests/glm5/phase483_boundary_writer_and_competition.py qwen3 1
  python tests/glm5/phase483_boundary_writer_and_competition.py glm4 1
  python tests/glm5/phase483_boundary_writer_and_competition.py deepseek7b 1
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


# ==================== 数据定义(与Phase 482一致) ====================
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
        "fruit": ["plant", "food"],
        "animal": ["food", "clothing"],
        "tool": ["vehicle", "furniture"],
        "vehicle": ["furniture", "tool"],
        "clothing": ["furniture", "tool"],
        "furniture": ["vehicle", "clothing"],
        "food": ["plant", "vehicle"],
        "plant": ["food", "animal"],
    },
    "glm4": {
        "fruit": ["plant", "food"],
        "animal": ["food", "clothing"],
        "tool": ["furniture", "vehicle"],
        "vehicle": ["tool", "furniture"],
        "clothing": ["furniture", "plant"],
        "furniture": ["vehicle", "clothing"],
        "food": ["plant", "fruit"],
        "plant": ["vehicle", "clothing"],
    },
    "deepseek7b": {
        "fruit": ["plant", "food"],
        "animal": ["food", "clothing"],
        "tool": ["vehicle", "furniture"],
        "vehicle": ["furniture", "tool"],
        "clothing": ["furniture", "plant"],
        "furniture": ["tool", "clothing"],
        "food": ["plant", "fruit"],
        "plant": ["food", "fruit"],
    },
}

# Phase 482最优层位(从Phase 482结果中提取)
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
    "kind_of": "The {obj} is a kind of",
}

# 层扫描范围(用于Exp3层位成因)
LAYER_RANGES = {
    "qwen3": list(range(18, 36)),
    "glm4": list(range(22, 40)),
    "deepseek7b": list(range(14, 28)),
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
    """从自然输入中移除B_c分量"""
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


# ==================== 获取特定层类别方向 ====================
def get_category_residuals_at_layer(model, tokenizer, device, model_name,
                                     categories, n_obj, target_layer, template_key="kind_of"):
    info = get_model_info(model, model_name)
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
    """获取某个类别在特定层的specific方向"""
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


# ==================== Exp1: 边界写入器定位 ====================
def exp1_boundary_writers(model, tokenizer, device, model_name, W_U):
    """
    找到写入类别边界残差的MLP神经元
    
    方法: 在每个类别最佳层, 计算每个MLP神经元的边界贡献
    NeuronBoundaryContribution = activation × W_down · B_c
    
    选择3个有代表性的类别做详细分析(控制时间)
    """
    plog("=== Exp1: 边界写入器定位 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    
    # 选择3个代表性类别(覆盖不同层位和不同选择性)
    if model_name == "qwen3":
        test_cats = ["fruit", "animal", "tool"]  # L32, L33, L23 — 跨度大
    elif model_name == "glm4":
        test_cats = ["fruit", "animal", "vehicle"]  # L27, L38, L29
    else:
        test_cats = ["fruit", "animal", "clothing"]  # L26, L27, L23
    
    writer_results = {}
    
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
        
        # 归一化specific方向
        b_hat = spec_vec / spec_norm
        target_idx = cat_names.index(cat_name)
        
        # 获取MLP down_proj权重 [d_model, intermediate_size]
        layer = layers_list[best_layer]
        mlp = layer.mlp
        w = mlp.down_proj.weight
        # 处理meta device(部分层在CPU上, 权重可能在meta device)
        if w.is_meta:
            # 从safetensors加载该层权重
            plog(f"    L{best_layer} down_proj on meta, loading from safetensors...")
            from safetensors.torch import load_file
            import glob
            model_path = MODEL_CONFIGS[model_name]["path"]
            sf_files = glob.glob(os.path.join(model_path, '*.safetensors'))
            W_down = None
            for sf_file in sf_files:
                try:
                    st = load_file(sf_file)
                    key = f"model.layers.{best_layer}.mlp.down_proj.weight"
                    if key in st:
                        W_down = st[key].float().numpy()
                        plog(f"      Loaded from {os.path.basename(sf_file)}")
                        break
                except Exception:
                    continue
            if W_down is None:
                plog(f"    WARNING: Cannot load W_down for L{best_layer}, skipping")
                continue
        else:
            W_down = w.detach().cpu().float().numpy()  # [d_model, intermediate]
        
        # 计算每个神经元对边界方向的贡献
        # W_down[i, :] · b_hat = 第i个中间神经元对边界方向的贡献权重
        neuron_boundary_weight = W_down.T @ b_hat  # [intermediate] — 每个神经元到边界的投影
        
        # 获取MLP激活值(使用4个训练对象)
        template = RELATION_TEMPLATES["kind_of"]
        train_objs = CATEGORIES_TRAIN[cat_name]
        
        all_activations = []
        for obj in train_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            
            # 用hook捕获down_proj的输入(即MLP激活后、投影回d_model前的向量)
            cap_mlp = {}
            def make_mlp_input_hook(key, position):
                """捕获module的input[0]在指定位置"""
                done = [False]
                def hook_fn(module, inp, output):
                    if not done[0]:
                        # inp是tuple, inp[0]是主输入tensor [1, seq, intermediate]
                        if isinstance(inp, tuple) and len(inp) > 0:
                            t = inp[0]
                            if t.is_meta:
                                # meta tensor无法直接读取, 跳过
                                done[0] = True
                                return
                            cap_mlp[key] = t.detach().float().cpu()
                        done[0] = True
                return hook_fn
            
            h_mlp = mlp.down_proj.register_forward_hook(make_mlp_input_hook("mlp_act", pos))
            
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_mlp.remove()
            
            if "mlp_act" in cap_mlp:
                # down_proj的输入: [1, seq_len, intermediate_size]
                mlp_act = cap_mlp["mlp_act"][0, pos].numpy()  # [intermediate]
                all_activations.append(mlp_act)
        
        if not all_activations:
            plog(f"    Skip {cat_name}: no MLP activations captured")
            continue
        
        avg_activation = np.mean(all_activations, axis=0)  # [intermediate]
        
        # 神经元边界贡献 = activation × boundary_weight
        # 即: activation_i × (W_down[:, i] · b_hat)
        neuron_contribution = avg_activation * neuron_boundary_weight  # [intermediate]
        
        # 找top-k边界写入神经元
        abs_contribution = np.abs(neuron_contribution)
        top_k = 50
        top_indices = np.argsort(abs_contribution)[-top_k:][::-1]
        top_contributions = neuron_contribution[top_indices]
        top_activations = avg_activation[top_indices]
        top_weights = neuron_boundary_weight[top_indices]
        
        # 计算top-k神经元的总贡献占比
        total_boundary_signal = np.sum(np.abs(neuron_contribution))
        top_k_signal = np.sum(abs_contribution[top_indices])
        concentration = top_k_signal / (total_boundary_signal + 1e-10)
        
        # 验证: top-k神经元消融效果(用2个test对象)
        test_objs = CATEGORIES[cat_name][4:6]
        
        # Baseline DCF
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
                baseline_dcfs.append(logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer))
        
        # 构造top-k零化hook
        def make_neuron_ablate_hook(top_indices_set, position):
            """将指定神经元在down_proj输入处置零"""
            done = [False]
            def hook_fn(module, inp, output):
                if not done[0]:
                    if isinstance(output, tuple):
                        out = output[0].clone()
                    else:
                        out = output.clone()
                    # out shape: [1, seq_len, d_model] (down_proj输出)
                    # 但我们需要在down_proj输入处操作
                    # 这里hook的是down_proj, input[0]是[1, seq, intermediate]
                    done[0] = True
                return output
            return hook_fn
        
        # 更直接的方法: 注入top-k神经元的组合方向, 看能否模拟+specific效果
        # 构造 top-k 神经元的组合方向
        # 如果我们激活 top-k 神经元, 等价于在residual stream中注入 W_down[:, top_k] @ delta_act
        # 其中 delta_act 只在 top_k 位置非零
        
        # 计算top-k神经元组合产生的residual stream方向
        top_k_direction = np.zeros(info.d_model)
        for i, idx in enumerate(top_indices):
            top_k_direction += top_contributions[i] * W_down[:, idx] / (np.linalg.norm(W_down[:, idx]) + 1e-10)
        
        # 计算 top_k_direction 与 B_c 的余弦
        cos_with_bc = float(np.dot(top_k_direction, b_hat) / (np.linalg.norm(top_k_direction) + 1e-10))
        
        # 也计算top-10, top-20, top-50的集中度
        concentration_10 = np.sum(abs_contribution[top_indices[:10]]) / (total_boundary_signal + 1e-10)
        concentration_20 = np.sum(abs_contribution[top_indices[:20]]) / (total_boundary_signal + 1e-10)
        
        # 正负贡献神经元的分布
        pos_neurons = np.sum(neuron_contribution > 0.01 * np.max(np.abs(neuron_contribution)))
        neg_neurons = np.sum(neuron_contribution < -0.01 * np.max(np.abs(neuron_contribution)))
        
        writer_results[cat_name] = {
            "best_layer": best_layer,
            "spec_norm": float(spec_norm),
            "total_boundary_signal": float(total_boundary_signal),
            "concentration_top50": float(concentration),
            "concentration_top20": float(concentration_20),
            "concentration_top10": float(concentration_10),
            "cos_top50_dir_with_bc": float(cos_with_bc),
            "n_positive_neurons": int(pos_neurons),
            "n_negative_neurons": int(neg_neurons),
            "intermediate_size": int(len(neuron_contribution)),
            "top10_indices": [int(x) for x in top_indices[:10]],
            "top10_contributions": [float(x) for x in top_contributions[:10]],
            "top10_activations": [float(x) for x in top_activations[:10]],
            "top10_boundary_weights": [float(x) for x in top_weights[:10]],
        }
        
        elapsed = time.time() - t0
        plog(f"    {cat_name}: top50 concentration={concentration:.3f}, "
              f"top10={concentration_10:.3f}, cos(Bc)={cos_with_bc:.3f}, "
              f"pos={pos_neurons}, neg={neg_neurons} ({elapsed:.1f}s)")
        gc.collect()
    
    return writer_results


# ==================== Exp2: 竞争释放图谱 ====================
def exp2_competition_release(model, tokenizer, device, model_name, W_U):
    """
    全8类别×8类别边界移除矩阵
    
    对每个类别, 移除其边界残差, 记录所有8个类别的DCF变化。
    得到CompetitionReleaseMatrix[8×8], 行=被移除类别, 列=DCF变化
    """
    plog("=== Exp2: 竞争释放图谱 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    
    release_matrix = np.zeros((8, 8))  # [removed_cat, dcf_dim]
    release_detail = {}
    
    for ci, cat_name in enumerate(cat_names):
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  Removing {cat_name} @ L{best_layer}...")
        t0 = time.time()
        
        # 获取specific方向
        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"    Skip {cat_name}: spec_norm too small")
            continue
        
        target_idx = cat_names.index(cat_name)
        
        # 使用4个test对象
        test_objs = CATEGORIES[cat_name][4:8]
        template = RELATION_TEMPLATES["kind_of"]
        
        dcf_deltas = []
        natural_proj_coeffs = []
        
        for obj in test_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            
            # Baseline
            cap = {}
            h = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" not in cap:
                continue
            baseline_resid = cap["resid"][0, pos].numpy()
            baseline_dcf = logit_lens_dcf(baseline_resid, W_U, tokenizer)
            
            # 记录自然投影
            b_hat = spec_vec / spec_norm
            proj_coeff = float(np.dot(baseline_resid, b_hat))
            natural_proj_coeffs.append(proj_coeff)
            
            # 移除(scale=1.0)
            h2 = layers_list[best_layer].register_forward_hook(
                make_remove_hook(spec_vec, pos, scale=1.0)
            )
            cap2 = {}
            h3 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap2, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h2.remove()
            h3.remove()
            if "resid" not in cap2:
                continue
            remove_dcf = logit_lens_dcf(cap2["resid"][0, pos].numpy(), W_U, tokenizer)
            dcf_deltas.append(remove_dcf - baseline_dcf)
        
        if dcf_deltas:
            avg_delta = np.mean(dcf_deltas, axis=0)
            release_matrix[ci] = avg_delta
            
            # 识别竞争释放: 哪些类别上升了
            competitor_releases = []
            for j, other_cat in enumerate(cat_names):
                if j != target_idx and avg_delta[j] > 0:
                    competitor_releases.append({
                        "category": other_cat,
                        "delta": float(avg_delta[j]),
                    })
            
            release_detail[cat_name] = {
                "best_layer": best_layer,
                "spec_norm": float(spec_norm),
                "target_delta": float(avg_delta[target_idx]),
                "avg_natural_proj": float(np.mean(natural_proj_coeffs)) if natural_proj_coeffs else 0,
                "dcf_delta": {cat_names[j]: float(avg_delta[j]) for j in range(8)},
                "competitor_releases": sorted(competitor_releases, key=lambda x: -x["delta"]),
                "selectivity": float(compute_selectivity(avg_delta, target_idx)),
            }
            
            # 日志
            max_release = max(competitor_releases, key=lambda x: x["delta"]) if competitor_releases else None
            release_str = f"top_release={max_release['category']}+{max_release['delta']:.2f}" if max_release else "no_release"
            plog(f"    {cat_name}: target_Δ={avg_delta[target_idx]:.2f}, {release_str} ({time.time()-t0:.1f}s)")
        
        gc.collect()
    
    return {
        "release_matrix": release_matrix.tolist(),
        "release_detail": release_detail,
        "cat_names": cat_names,
    }


# ==================== Exp3: 最佳层位成因分析 ====================
def exp3_layer_formation(model, tokenizer, device, model_name, W_U):
    """
    分析类别边界在不同层的形成过程
    
    对每个类别, 扫描其最佳层附近的层, 测量:
    1. raw cluster emergence: 类别均值方向与全局均值的差异
    2. specific boundary emergence: specific方向的范数和选择性
    3. DCF readability: 在该层注入specific方向的DCF增益
    4. boundary necessity: 移除specific方向的DCF变化
    """
    plog("=== Exp3: 最佳层位成因分析 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    
    # 选择3个类别(不同层位范围)
    if model_name == "qwen3":
        test_cats = ["fruit", "animal", "tool"]  # L32, L33, L23
    elif model_name == "glm4":
        test_cats = ["fruit", "animal", "vehicle"]
    else:
        test_cats = ["fruit", "animal", "clothing"]
    
    formation_results = {}
    
    for cat_name in test_cats:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name} (best=L{best_layer})...")
        t0 = time.time()
        
        target_idx = cat_names.index(cat_name)
        
        # 扫描最佳层附近的层
        scan_range = list(range(max(0, best_layer - 6), min(info.n_layers, best_layer + 6)))
        # 控制层数不超过12层
        if len(scan_range) > 12:
            step = len(scan_range) // 12
            scan_range = scan_range[::step][:12]
        
        layer_data = []
        
        for li, layer_idx in enumerate(scan_range):
            if layer_idx >= info.n_layers or layer_idx < 0:
                continue
            
            # 获取该层的specific方向
            spec_vec, spec_norm = get_specific_direction(
                model, tokenizer, device, model_name, cat_name, layer_idx
            )
            
            if spec_vec is None or spec_norm < 1e-6:
                layer_data.append({
                    "layer": layer_idx,
                    "spec_norm": 0,
                    "selectivity": 0,
                    "injection_target_delta": 0,
                    "removal_target_delta": 0,
                    "removal_max_competitor_delta": 0,
                })
                continue
            
            b_hat = spec_vec / spec_norm
            
            # 测试注入效果(2个test对象, scale=1.0)
            test_objs = CATEGORIES[cat_name][4:6]
            template = RELATION_TEMPLATES["kind_of"]
            
            inject_deltas = []
            remove_deltas = []
            
            for obj in test_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                
                # Baseline
                cap = {}
                h = layers_list[layer_idx].register_forward_hook(_make_capture_hook(cap, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                if "resid" not in cap:
                    continue
                baseline_dcf = logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer)
                
                # 注入 +specific (scale=1.0)
                ivec = torch.tensor(spec_vec, dtype=torch.float32)
                h2 = layers_list[layer_idx].register_forward_hook(make_inject_hook(ivec, pos))
                cap2 = {}
                h3 = layers_list[layer_idx].register_forward_hook(_make_capture_hook(cap2, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h2.remove()
                h3.remove()
                if "resid" not in cap2:
                    continue
                inject_dcf = logit_lens_dcf(cap2["resid"][0, pos].numpy(), W_U, tokenizer)
                inject_deltas.append(inject_dcf - baseline_dcf)
                
                # 移除 (scale=1.0)
                h4 = layers_list[layer_idx].register_forward_hook(
                    make_remove_hook(spec_vec, pos, scale=1.0)
                )
                cap3 = {}
                h5 = layers_list[layer_idx].register_forward_hook(_make_capture_hook(cap3, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h4.remove()
                h5.remove()
                if "resid" not in cap3:
                    continue
                remove_dcf = logit_lens_dcf(cap3["resid"][0, pos].numpy(), W_U, tokenizer)
                remove_deltas.append(remove_dcf - baseline_dcf)
            
            # 汇总该层结果
            inject_avg = np.mean(inject_deltas, axis=0) if inject_deltas else np.zeros(8)
            remove_avg = np.mean(remove_deltas, axis=0) if remove_deltas else np.zeros(8)
            
            inject_sel = compute_selectivity(inject_avg, target_idx)
            remove_sel = compute_selectivity(remove_avg, target_idx)
            
            # 竞争释放
            remove_competitor_deltas = [remove_avg[j] for j in range(8) if j != target_idx]
            max_competitor_release = max(remove_competitor_deltas) if remove_competitor_deltas else 0
            
            layer_data.append({
                "layer": layer_idx,
                "spec_norm": float(spec_norm),
                "inject_selectivity": float(inject_sel),
                "inject_target_delta": float(inject_avg[target_idx]),
                "remove_selectivity": float(remove_sel),
                "remove_target_delta": float(remove_avg[target_idx]),
                "remove_max_competitor_delta": float(max_competitor_release),
                "inject_dcf_delta": {cat_names[j]: float(inject_avg[j]) for j in range(8)},
                "remove_dcf_delta": {cat_names[j]: float(remove_avg[j]) for j in range(8)},
            })
            
            if (li + 1) % 3 == 0:
                plog(f"    Scanned {li+1}/{len(scan_range)} layers")
            
            gc.collect()
        
        # 找到关键形成指标的变化趋势
        # 1. spec_norm在哪层开始上升?
        # 2. selectivity在哪层达到峰值?
        # 3. 竞争释放在哪层最强?
        
        norms = [d["spec_norm"] for d in layer_data]
        sels = [d["inject_selectivity"] for d in layer_data]
        removals = [d["remove_target_delta"] for d in layer_data]
        competitors = [d["remove_max_competitor_delta"] for d in layer_data]
        
        # 找norm首次超过最大值50%的层
        norm_emergence_layer = None
        if max(norms) > 0:
            threshold = max(norms) * 0.5
            for d in layer_data:
                if d["spec_norm"] >= threshold:
                    norm_emergence_layer = d["layer"]
                    break
        
        formation_results[cat_name] = {
            "best_layer": best_layer,
            "norm_emergence_layer": norm_emergence_layer,
            "max_norm_layer": layer_data[np.argmax(norms)]["layer"] if layer_data else None,
            "max_sel_layer": layer_data[np.argmax(sels)]["layer"] if layer_data else None,
            "max_removal_layer": layer_data[np.argmax([abs(r) for r in removals])]["layer"] if layer_data else None,
            "max_competitor_layer": layer_data[np.argmax(competitors)]["layer"] if layer_data else None,
            "layer_data": layer_data,
        }
        
        plog(f"    {cat_name}: norm_emergence=L{norm_emergence_layer}, "
              f"max_sel=L{formation_results[cat_name]['max_sel_layer']}, "
              f"max_removal=L{formation_results[cat_name]['max_removal_layer']} ({time.time()-t0:.1f}s)")
        
        gc.collect()
    
    return formation_results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    plog(f"Phase 483: Boundary Writers + Competition Release + Layer Formation | "
         f"Model={model_name} | Round={round_num}")
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    plog(f"  W_U: {W_U.shape}")
    
    results = {
        "phase": 483,
        "model": model_name,
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "core_question": "Boundary writers, competition release matrix, layer formation",
        "model_info": {"class": info.model_class, "n_layers": info.n_layers, "d_model": info.d_model},
    }
    
    # Exp1: 边界写入器定位
    t0 = time.time()
    exp1_data = exp1_boundary_writers(model, tokenizer, device, model_name, W_U)
    results["exp1_boundary_writers"] = exp1_data
    plog(f"Exp1 done in {time.time()-t0:.1f}s")
    
    # Exp2: 竞争释放图谱
    t0 = time.time()
    exp2_data = exp2_competition_release(model, tokenizer, device, model_name, W_U)
    results["exp2_competition_release"] = exp2_data
    plog(f"Exp2 done in {time.time()-t0:.1f}s")
    
    # Exp3: 最佳层位成因
    t0 = time.time()
    exp3_data = exp3_layer_formation(model, tokenizer, device, model_name, W_U)
    results["exp3_layer_formation"] = exp3_data
    plog(f"Exp3 done in {time.time()-t0:.1f}s")
    
    # 保存结果
    out_dir = "results/glm5"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"phase483_{model_name}_r{round_num}.json")
    
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj
    
    results = convert(results)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    plog(f"Results saved to {out_path}")
    
    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    plog("Done!")


if __name__ == "__main__":
    main()
