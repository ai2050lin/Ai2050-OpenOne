"""
Phase 486: 跨层边界累积路径追踪
=================================

Phase 485核心发现:
- Qwen3 clothing: 单层MLP+Attn只覆盖39.3%边界幅度
- fruit/animal: 单层Attn+MLP消融覆盖率为负
- B_c关系不变性在小scale下仍成立
- 格式子空间占B_c的30-66%

Phase 486核心目标:
1. Exp1: 跨层边界累积剖面 — 逐层测量attn/MLP对B_c的贡献
2. Exp2: 多层联合消融 — 在多层同时消融MLP，测累积幅度闭环
3. Exp3: 格式子空间逐层分析 — 哪些层格式污染最重
4. Exp4: 关系不变性在跨层视角下的稳定性

关键改进:
- 修复W_o meta device问题(用safe_load_weight)
- 不直接用W_o拆分头，改用hook捕获attn子层输出差异
- 逐层扫描而非只看最佳层

用法:
  python tests/glm5/phase486_cross_layer_boundary.py qwen3
  python tests/glm5/phase486_cross_layer_boundary.py glm4
  python tests/glm5/phase486_cross_layer_boundary.py deepseek7b
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json, math
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS, safe_decode)


def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 数据定义 ====================
CATEGORIES = {
    "fruit":     ["apple", "banana", "orange", "grape"],
    "animal":    ["dog", "cat", "horse", "lion"],
    "clothing":  ["shirt", "dress", "hat", "coat"],
    "food":      ["bread", "rice", "cheese", "pasta"],
    "vehicle":   ["car", "bus", "bicycle", "truck"],
    "plant":     ["tree", "flower", "grass", "bush"],
}

CATEGORIES_R2 = {
    "fruit":     ["apple", "banana", "orange", "grape", "pear", "peach", "mango", "plum"],
    "animal":    ["dog", "cat", "horse", "lion", "bear", "rabbit", "eagle", "fish"],
    "clothing":  ["shirt", "dress", "hat", "coat", "sock", "glove", "jacket", "scarf"],
    "food":      ["bread", "rice", "cheese", "pasta", "soup", "steak", "salad", "cake"],
}

BEST_NEIGHBORS = {
    "qwen3": {
        "fruit": ["plant", "food"], "animal": ["food", "clothing"],
        "clothing": ["furniture", "tool"], "food": ["plant", "fruit"],
        "vehicle": ["furniture", "tool"], "plant": ["food", "animal"],
    },
    "glm4": {
        "fruit": ["plant", "food"], "animal": ["food", "clothing"],
        "clothing": ["furniture", "plant"], "food": ["plant", "fruit"],
        "vehicle": ["tool", "furniture"], "plant": ["vehicle", "clothing"],
    },
    "deepseek7b": {
        "fruit": ["plant", "food"], "animal": ["food", "clothing"],
        "clothing": ["furniture", "plant"], "food": ["plant", "fruit"],
        "vehicle": ["furniture", "tool"], "plant": ["food", "fruit"],
    },
}

BEST_LAYERS = {
    "qwen3": {
        "fruit": 32, "animal": 33, "clothing": 30,
        "food": 34, "vehicle": 29, "plant": 28,
    },
    "glm4": {
        "fruit": 27, "animal": 38, "clothing": 39,
        "food": 38, "vehicle": 29, "plant": 32,
    },
    "deepseek7b": {
        "fruit": 26, "animal": 27, "clothing": 23,
        "food": 27, "vehicle": 26, "plant": 25,
    },
}

FAMILY_WORDS_8D = {
    "fruit":     ["fruit", "produce", "crop", "berry"],
    "animal":    ["animal", "creature", "beast", "pet"],
    "clothing":  ["clothing", "attire", "wear", "garment"],
    "food":      ["food", "meal", "dish", "snack"],
    "vehicle":   ["vehicle", "transport", "automobile", "car"],
    "plant":     ["plant", "tree", "vegetation", "flora"],
    "tool":      ["tool", "implement", "device", "instrument"],
    "furniture": ["furniture", "furnishing", "fixture", "seat"],
}

DCF_DIM_NAMES = ["fruit", "animal", "clothing", "food", "vehicle", "plant", "tool", "furniture"]

RELATION_TEMPLATES = {
    "kind_of":   "The {obj} is a kind of",
    "used_for":  "The {obj} is used for",
    "found_in":  "The {obj} is found in",
}

# Phase 485发现的关键边界: 优先测试
PRIORITY_CATS = {
    "qwen3": ["clothing", "fruit", "animal"],
    "glm4":  ["fruit", "clothing", "animal"],
    "deepseek7b": ["fruit", "food", "animal"],
}


# ==================== 模型加载 ====================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog(f"Loading {model_name} (bfloat16 + device_map=auto + SDPA)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 优先SDPA，回退eager
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="sdpa",
        )
        plog(f"  SDPA enabled")
    except Exception as e:
        plog(f"  SDPA failed ({e}), falling back to eager")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="eager",
        )
    model.eval()
    layers_list = get_layers(model)
    n_layers = len(layers_list)

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

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    plog(f"  {model_name}: device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


# ==================== 基础工具 ====================
def find_token_id(tokenizer, word):
    vocab = tokenizer.get_vocab()
    for c in [word, " " + word, word.lower(), " " + word.lower()]:
        if c in vocab:
            return vocab[c]
    return None


def compute_dcf(logits, tokenizer, dim_dict=None, dim_names=None):
    if dim_dict is None:
        dim_dict = FAMILY_WORDS_8D
    if dim_names is None:
        dim_names = DCF_DIM_NAMES
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


def logit_lens_dcf(resid, W_U, tokenizer):
    logits = resid @ W_U.T
    return compute_dcf(logits, tokenizer)


def get_prompt_ids(tokenizer, device, prompt, max_len=128):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = attention_mask.sum().item()
    pos = seq_len - 1
    return input_ids, attention_mask, pos


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


# ==================== B_c构造 ====================
def get_category_residuals_at_layer(model, tokenizer, device, model_name,
                                     categories, n_obj, target_layer, template_key="kind_of"):
    layers_list = get_layers(model)
    template = RELATION_TEMPLATES[template_key]
    results = {}
    for cat_name, cat_objs in categories.items():
        resids = []
        for obj in cat_objs[:n_obj]:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap = {}
            h = layers_list[target_layer].register_forward_hook(
                lambda m, i, o, k="r": cap.__setitem__(k, o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu())
            )
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "r" in cap:
                resids.append(cap["r"][0, pos].numpy())
        if resids:
            results[cat_name] = np.mean(resids, axis=0)
    return results


def get_specific_direction(model, tokenizer, device, model_name, cat_name, target_layer, n_obj=4):
    neighbors = BEST_NEIGHBORS[model_name]
    raw_dirs = get_category_residuals_at_layer(
        model, tokenizer, device, model_name,
        categories=CATEGORIES, n_obj=n_obj, target_layer=target_layer
    )
    if cat_name not in raw_dirs:
        return None, 0.0
    target_vec = raw_dirs[cat_name]
    basis_vecs = [raw_dirs[n] for n in neighbors.get(cat_name, []) if n in raw_dirs]
    spec_vec = qr_orthogonalize(target_vec, basis_vecs) if basis_vecs else target_vec.copy()
    spec_norm = np.linalg.norm(spec_vec)
    return spec_vec, float(spec_norm)


# ==================== Exp1: 跨层边界累积剖面 ★★★★★ ====================
def exp1_cross_layer_accumulation(model, tokenizer, device, model_name, W_U):
    """
    对每个类别，逐层测量:
    1. 残差流在B_c方向上的投影 (累积信号)
    2. Attn子层输出在B_c方向上的投影 (attn贡献)
    3. MLP子层输出在B_c方向上的投影 (MLP贡献)

    方法: 在最佳层构造B_c方向，然后逐层测量各子层输出与B_c的投影
    """
    plog("=== Exp1: 跨层边界累积剖面 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers
    cat_names = PRIORITY_CATS.get(model_name, ["fruit", "clothing", "animal"])

    results = {}

    for cat_name in cat_names:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name}: B_c from L{best_layer}, scanning all layers...")
        t0 = time.time()

        # 构造B_c方向 (在最佳层)
        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"    Skip {cat_name}: spec_norm={spec_norm:.4f}")
            continue

        b_hat = spec_vec / spec_norm  # 归一化方向
        target_idx = DCF_DIM_NAMES.index(cat_name) if cat_name in DCF_DIM_NAMES else 0

        # 逐层扫描: 采样层 (不全部扫描以节省时间)
        if n_layers <= 40:
            scan_layers = list(range(n_layers))
        else:
            # 均匀采样 + 关键层
            step = max(1, n_layers // 20)
            scan_layers = sorted(set(list(range(0, n_layers, step)) + [best_layer, n_layers-1]))

        plog(f"    Scanning {len(scan_layers)} layers...")

        # 对每个类别，获取4个样本的prompt
        template = RELATION_TEMPLATES["kind_of"]
        cat_objs = CATEGORIES[cat_name]
        neighbor_cats = BEST_NEIGHBORS[model_name].get(cat_name, [])

        layer_profile = {}

        for scan_l in scan_layers:
            layer = layers_list[scan_l]

            # 对每个对象: 捕获attn输出、MLP输出、层输出
            cat_attn_projs = []
            cat_mlp_projs = []
            cat_resid_projs = []
            neigh_attn_projs = []
            neigh_mlp_projs = []
            neigh_resid_projs = []

            # ---- 类别内样本 ----
            for obj in cat_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                cap_attn = {}
                cap_mlp = {}
                cap_resid = {}
                done = [False, False, False]

                def make_hook(store, idx):
                    def hook_fn(module, inp, output):
                        if not done[idx]:
                            if isinstance(output, tuple):
                                store["v"] = output[0].detach().float().cpu()
                            else:
                                store["v"] = output.detach().float().cpu()
                            done[idx] = True
                    return hook_fn

                h1 = layer.self_attn.register_forward_hook(make_hook(cap_attn, 0))
                h2 = layer.mlp.register_forward_hook(make_hook(cap_mlp, 1))
                h3 = layer.register_forward_hook(make_hook(cap_resid, 2))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                h1.remove(); h2.remove(); h3.remove()

                if "v" in cap_resid:
                    resid_vec = cap_resid["v"][0, pos].numpy()
                    cat_resid_projs.append(float(np.dot(resid_vec, b_hat)))
                if "v" in cap_attn:
                    attn_vec = cap_attn["v"][0, pos].numpy()
                    cat_attn_projs.append(float(np.dot(attn_vec, b_hat)))
                if "v" in cap_mlp:
                    mlp_vec = cap_mlp["v"][0, pos].numpy()
                    cat_mlp_projs.append(float(np.dot(mlp_vec, b_hat)))

            # ---- 邻居类别样本 ----
            for nc in neighbor_cats[:2]:
                for obj in CATEGORIES.get(nc, [])[:2]:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                    cap_attn = {}
                    cap_mlp = {}
                    cap_resid = {}
                    done = [False, False, False]

                    def make_hook2(store, idx):
                        def hook_fn(module, inp, output):
                            if not done[idx]:
                                if isinstance(output, tuple):
                                    store["v"] = output[0].detach().float().cpu()
                                else:
                                    store["v"] = output.detach().float().cpu()
                                done[idx] = True
                        return hook_fn

                    h1 = layer.self_attn.register_forward_hook(make_hook2(cap_attn, 0))
                    h2 = layer.mlp.register_forward_hook(make_hook2(cap_mlp, 1))
                    h3 = layer.register_forward_hook(make_hook2(cap_resid, 2))

                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)

                    h1.remove(); h2.remove(); h3.remove()

                    if "v" in cap_resid:
                        resid_vec = cap_resid["v"][0, pos].numpy()
                        neigh_resid_projs.append(float(np.dot(resid_vec, b_hat)))
                    if "v" in cap_attn:
                        attn_vec = cap_attn["v"][0, pos].numpy()
                        neigh_attn_projs.append(float(np.dot(attn_vec, b_hat)))
                    if "v" in cap_mlp:
                        mlp_vec = cap_mlp["v"][0, pos].numpy()
                        neigh_mlp_projs.append(float(np.dot(mlp_vec, b_hat)))

            # 计算差异投影 (类别内 - 邻居)
            cat_resid_mean = np.mean(cat_resid_projs) if cat_resid_projs else 0
            neigh_resid_mean = np.mean(neigh_resid_projs) if neigh_resid_projs else 0
            cat_attn_mean = np.mean(cat_attn_projs) if cat_attn_projs else 0
            neigh_attn_mean = np.mean(neigh_attn_projs) if neigh_attn_projs else 0
            cat_mlp_mean = np.mean(cat_mlp_projs) if cat_mlp_projs else 0
            neigh_mlp_mean = np.mean(neigh_mlp_projs) if neigh_mlp_projs else 0

            layer_profile[str(scan_l)] = {
                "resid_proj_cat": cat_resid_mean,
                "resid_proj_neigh": neigh_resid_mean,
                "resid_proj_diff": cat_resid_mean - neigh_resid_mean,
                "attn_proj_cat": cat_attn_mean,
                "attn_proj_neigh": neigh_attn_mean,
                "attn_proj_diff": cat_attn_mean - neigh_attn_mean,
                "mlp_proj_cat": cat_mlp_mean,
                "mlp_proj_neigh": neigh_mlp_mean,
                "mlp_proj_diff": cat_mlp_mean - neigh_mlp_mean,
            }

        # 找出attn/MLP贡献最大的层
        attn_peak_layer = max(layer_profile.keys(),
                              key=lambda l: abs(layer_profile[l]["attn_proj_diff"]))
        mlp_peak_layer = max(layer_profile.keys(),
                             key=lambda l: abs(layer_profile[l]["mlp_proj_diff"]))
        resid_peak_layer = max(layer_profile.keys(),
                               key=lambda l: abs(layer_profile[l]["resid_proj_diff"]))

        elapsed = time.time() - t0
        plog(f"    {cat_name} done ({elapsed:.0f}s): attn_peak=L{attn_peak_layer}, "
             f"mlp_peak=L{mlp_peak_layer}, resid_peak=L{resid_peak_layer}")

        results[cat_name] = {
            "best_layer": best_layer,
            "spec_norm": spec_norm,
            "layer_profile": layer_profile,
            "attn_peak_layer": attn_peak_layer,
            "mlp_peak_layer": mlp_peak_layer,
            "resid_peak_layer": resid_peak_layer,
            "elapsed": elapsed,
        }

    return results


# ==================== Exp2: 多层联合消融 ====================
def exp2_multi_layer_ablation(model, tokenizer, device, model_name, W_U):
    """
    在多层同时消融MLP对B_c的贡献，测试跨层累积效果

    方法:
    1. 对每个类别，先通过Exp1的profile找到MLP贡献最大的3-5个层
    2. 逐层添加MLP消融hook，测量DCF变化
    3. 对比单层消融 vs 多层联合消融

    消融方式: 将MLP输出在B_c方向上的投影去除
    """
    plog("=== Exp2: 多层联合消融 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers
    cat_names = PRIORITY_CATS.get(model_name, ["fruit", "clothing", "animal"])

    results = {}

    for cat_name in cat_names:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name}: multi-layer MLP ablation...")
        t0 = time.time()

        # 构造B_c方向
        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"    Skip {cat_name}")
            continue

        b_hat = spec_vec / spec_norm
        target_idx = DCF_DIM_NAMES.index(cat_name) if cat_name in DCF_DIM_NAMES else 0

        # 先做方向级remove (baseline)
        template = RELATION_TEMPLATES["kind_of"]
        cat_objs = CATEGORIES[cat_name]
        neighbor_cats = BEST_NEIGHBORS[model_name].get(cat_name, [])

        # 方向级remove baseline
        direction_remove_deltas = []
        for obj in cat_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

            # baseline
            cap_base = {}
            h_base = layers_list[best_layer].register_forward_hook(
                lambda m, i, o: cap_base.__setitem__("v", o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu())
            )
            with torch.no_grad():
                out_base = model(input_ids=input_ids, attention_mask=attention_mask)
            h_base.remove()
            if "v" not in cap_base:
                continue
            logits_base = out_base.logits[0, -1].float().cpu().numpy()
            dcf_base = compute_dcf(logits_base, tokenizer)

            # remove方向
            cap_rm = {}
            def make_remove_hook_fn(b_hat_np, position):
                added = [False]
                def hook_fn(module, inp, output):
                    if not added[0]:
                        if isinstance(output, tuple):
                            out = output[0].clone()
                        else:
                            out = output.clone()
                        resid_np = out[0, position, :].float().cpu().numpy()
                        proj = np.dot(resid_np, b_hat_np) * b_hat_np
                        out[0, position, :] -= torch.tensor(proj, dtype=out.dtype, device=out.device)
                        added[0] = True
                        if isinstance(output, tuple):
                            return (out,) + output[1:]
                        return out
                    return output
                return hook_fn

            h_rm = layers_list[best_layer].register_forward_hook(make_remove_hook_fn(b_hat, pos))
            with torch.no_grad():
                out_rm = model(input_ids=input_ids, attention_mask=attention_mask)
            h_rm.remove()
            logits_rm = out_rm.logits[0, -1].float().cpu().numpy()
            dcf_rm = compute_dcf(logits_rm, tokenizer)
            direction_remove_deltas.append(dcf_rm - dcf_base)

        if not direction_remove_deltas:
            plog(f"    Skip {cat_name}: no direction remove data")
            continue

        dir_remove_mean = np.mean(direction_remove_deltas, axis=0)
        dir_remove_target = dir_remove_mean[target_idx]

        # 选取消融层: best_layer附近 ±5, ±10, 以及首尾
        ablation_layer_sets = [
            [best_layer],  # 单层
            sorted(set([max(0, best_layer-5), best_layer, min(n_layers-1, best_layer+5)])),  # ±5
            sorted(set([max(0, best_layer-10), max(0, best_layer-5), best_layer,
                       min(n_layers-1, best_layer+5), min(n_layers-1, best_layer+10)])),  # ±10
            sorted(set([0, max(0, best_layer-10), best_layer//2, best_layer,
                       min(n_layers-1, best_layer+5), n_layers-1])),  # 全跨度
        ]
        ablation_names = ["single", "pm5", "pm10", "full_span"]

        ablation_results = {}

        for abl_name, abl_layers in zip(ablation_names, ablation_layer_sets):
            plog(f"    Ablation {abl_name}: layers={abl_layers}")
            mlp_remove_deltas = []

            for obj in cat_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                # 在多个层同时添加MLP B_c投影消融
                hooks = []
                for al in abl_layers:
                    layer = layers_list[al]
                    def make_mlp_remove_hook(b_hat_np, position, layer_idx):
                        added = [False]
                        def hook_fn(module, inp, output):
                            if not added[0]:
                                if isinstance(output, tuple):
                                    out = output[0].clone()
                                else:
                                    out = output.clone()
                                mlp_out_np = out[0, position, :].float().cpu().numpy()
                                proj = np.dot(mlp_out_np, b_hat_np) * b_hat_np
                                out[0, position, :] -= torch.tensor(proj, dtype=out.dtype, device=out.device)
                                added[0] = True
                                if isinstance(output, tuple):
                                    return (out,) + output[1:]
                                return out
                            return output
                        return hook_fn
                    hooks.append(layer.mlp.register_forward_hook(make_mlp_remove_hook(b_hat, pos, al)))

                with torch.no_grad():
                    out_abl = model(input_ids=input_ids, attention_mask=attention_mask)

                for h in hooks:
                    h.remove()

                logits_abl = out_abl.logits[0, -1].float().cpu().numpy()
                dcf_abl = compute_dcf(logits_abl, tokenizer)
                mlp_remove_deltas.append(dcf_abl - dcf_base)

            if mlp_remove_deltas:
                mlp_remove_mean = np.mean(mlp_remove_deltas, axis=0)
                mlp_target = mlp_remove_mean[target_idx]
                amplitude_ratio = abs(mlp_target / dir_remove_target) if abs(dir_remove_target) > 0.01 else 0
                cos_with_dir = float(np.dot(mlp_remove_mean, dir_remove_mean) /
                                    (np.linalg.norm(mlp_remove_mean) * np.linalg.norm(dir_remove_mean) + 1e-10))
                ablation_results[abl_name] = {
                    "layers": abl_layers,
                    "target_delta": float(mlp_target),
                    "amplitude_ratio": float(amplitude_ratio),
                    "cos_with_direction_remove": float(cos_with_dir),
                    "dcf_delta": [float(x) for x in mlp_remove_mean],
                }
                plog(f"      target_D={mlp_target:.2f}, amp={amplitude_ratio:.1%}, cos={cos_with_dir:.3f}")

        elapsed = time.time() - t0
        plog(f"    {cat_name} multi-layer done ({elapsed:.0f}s)")

        results[cat_name] = {
            "best_layer": best_layer,
            "direction_remove_target": float(dir_remove_target),
            "ablation_results": ablation_results,
            "elapsed": elapsed,
        }

    return results


# ==================== Exp3: 格式子空间逐层分析 ====================
def exp3_format_subspace_per_layer(model, tokenizer, device, model_name, W_U):
    """
    在每个层构造格式子空间，测量B_c与格式方向的cos

    方法:
    1. 用多种模板获取残差流: "The X is a kind of", "The X is used for",
       "Answer: X", "Question: What is X?"
    2. SVD提取格式主成分
    3. 测量B_c与格式主成分的cos
    """
    plog("=== Exp3: 格式子空间逐层分析 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = PRIORITY_CATS.get(model_name, ["fruit", "clothing", "animal"])

    # 格式模板 — 4种不同句式
    format_templates = [
        "The {obj} is a kind of",
        "The {obj} is used for",
        "Answer: {obj}",
        "Question: What is {obj}?",
    ]

    # 格式对象 — 用同一对象在不同模板下的残差差异提取格式成分
    format_objects = ["apple", "dog", "car", "hat", "bread", "tree"]

    results = {}

    for cat_name in cat_names:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name}: format subspace per layer...")
        t0 = time.time()

        # 构造B_c
        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"    Skip {cat_name}")
            continue

        b_hat = spec_vec / spec_norm

        # 采样层 (不全部扫描)
        if info.n_layers <= 40:
            scan_layers = list(range(0, info.n_layers, 5)) + [info.n_layers - 1]
        else:
            step = max(1, info.n_layers // 10)
            scan_layers = sorted(set(list(range(0, info.n_layers, step)) + [best_layer, info.n_layers - 1]))
        scan_layers = sorted(set(scan_layers))

        format_profile = {}

        for scan_l in scan_layers:
            layer = layers_list[scan_l]

            # 收集不同模板下的残差流
            template_resids = {t: [] for t in format_templates}

            for obj in format_objects[:3]:  # 只用3个对象节省时间
                for tmpl in format_templates:
                    prompt = tmpl.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                    cap = {}
                    h = layer.register_forward_hook(
                        lambda m, i, o: cap.__setitem__("v", o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu())
                    )
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    h.remove()

                    if "v" in cap:
                        template_resids[tmpl].append(cap["v"][0, pos].numpy())

            # 构造格式方向: 对每个对象，不同模板的残差差异
            format_diffs = []
            for obj_idx in range(min(3, len(format_objects[:3]))):
                # 取第obj_idx个对象在所有模板下的残差
                obj_resids = []
                for tmpl in format_templates:
                    if obj_idx < len(template_resids[tmpl]):
                        obj_resids.append(template_resids[tmpl][obj_idx])
                if len(obj_resids) >= 2:
                    # 对象内的模板差异 (去除对象语义，保留格式)
                    for i in range(len(obj_resids)):
                        for j in range(i+1, len(obj_resids)):
                            format_diffs.append(obj_resids[i] - obj_resids[j])

            if len(format_diffs) < 2:
                continue

            # SVD提取格式主成分
            format_matrix = np.array(format_diffs)  # [n_diffs, d_model]
            try:
                U, S, Vt = np.linalg.svd(format_matrix, full_matrices=False)
                # 前3个格式主方向
                format_dirs = Vt[:3]  # [3, d_model]

                # B_c与每个格式方向的cos
                cos_values = []
                for fd in format_dirs:
                    fd_norm = np.linalg.norm(fd)
                    if fd_norm > 1e-10:
                        cos_values.append(float(abs(np.dot(b_hat, fd / fd_norm))))
                    else:
                        cos_values.append(0.0)

                format_profile[str(scan_l)] = {
                    "cos_bc_format_top1": cos_values[0] if len(cos_values) > 0 else 0,
                    "cos_bc_format_top2": cos_values[1] if len(cos_values) > 1 else 0,
                    "cos_bc_format_top3": cos_values[2] if len(cos_values) > 2 else 0,
                    "format_energy_top3": float(np.sum(S[:3]**2) / (np.sum(S**2) + 1e-10)),
                    "n_format_diffs": len(format_diffs),
                }
            except Exception as e:
                plog(f"    L{scan_l} SVD failed: {e}")

        elapsed = time.time() - t0
        plog(f"    {cat_name} format profile done ({elapsed:.0f}s): {len(format_profile)} layers")

        results[cat_name] = {
            "best_layer": best_layer,
            "format_profile": format_profile,
            "elapsed": elapsed,
        }

    return results


# ==================== Exp4: 跨层关系不变性 ====================
def exp4_cross_layer_relation_invariance(model, tokenizer, device, model_name, W_U):
    """
    在多个层(非最佳层)测试B_c注入的关系不变性

    Phase 485已证明最佳层B_c注入delta跨关系稳定。
    现在测试: 在其他层注入B_c时，关系不变性是否仍成立？
    """
    plog("=== Exp4: 跨层关系不变性 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = ["clothing", "fruit"]  # 只测2个类别节省时间

    results = {}

    for cat_name in cat_names:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name}: cross-layer relation invariance...")
        t0 = time.time()

        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"    Skip {cat_name}")
            continue

        b_hat = spec_vec / spec_norm
        target_idx = DCF_DIM_NAMES.index(cat_name) if cat_name in DCF_DIM_NAMES else 0

        # 测试层: 最佳层 ± 5, ± 10, 以及中点
        test_layers = sorted(set([
            max(0, best_layer - 10), max(0, best_layer - 5),
            best_layer,
            min(info.n_layers-1, best_layer + 5),
            min(info.n_layers-1, best_layer + 10),
        ]))

        scale = 0.1  # 小scale
        relations = ["kind_of", "used_for", "found_in"]

        layer_invariance = {}

        for test_l in test_layers:
            rel_deltas = {}

            for rel_key in relations:
                template = RELATION_TEMPLATES[rel_key]
                deltas = []
                for obj in CATEGORIES[cat_name]:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                    # baseline
                    with torch.no_grad():
                        out_base = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits_base = out_base.logits[0, -1].float().cpu().numpy()
                    dcf_base = compute_dcf(logits_base, tokenizer)

                    # 注入B_c
                    inject_vec = torch.tensor(scale * spec_norm * b_hat,
                                               dtype=torch.bfloat16, device=device)
                    added = [False]
                    def make_inject_hook(ivec, position):
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

                    h_inj = layers_list[test_l].register_forward_hook(make_inject_hook(inject_vec, pos))
                    with torch.no_grad():
                        out_inj = model(input_ids=input_ids, attention_mask=attention_mask)
                    h_inj.remove()

                    logits_inj = out_inj.logits[0, -1].float().cpu().numpy()
                    dcf_inj = compute_dcf(logits_inj, tokenizer)
                    deltas.append(dcf_inj - dcf_base)

                if deltas:
                    mean_delta = np.mean(deltas, axis=0)
                    rel_deltas[rel_key] = {
                        "target_delta": float(mean_delta[target_idx]),
                        "dcf_delta": [float(x) for x in mean_delta],
                    }

            # 计算跨关系一致性
            target_deltas = [rel_deltas[r]["target_delta"] for r in relations if r in rel_deltas]
            if len(target_deltas) >= 2:
                delta_mean = float(np.mean(target_deltas))
                delta_range = float(max(target_deltas) - min(target_deltas))
                relative_range = delta_range / (abs(delta_mean) + 0.01)
            else:
                delta_mean = delta_range = relative_range = 0

            layer_invariance[str(test_l)] = {
                "rel_deltas": rel_deltas,
                "delta_mean": delta_mean,
                "delta_range": delta_range,
                "relative_range": relative_range,
            }

        elapsed = time.time() - t0
        plog(f"    {cat_name} done ({elapsed:.0f}s): {len(layer_invariance)} layers tested")

        results[cat_name] = {
            "best_layer": best_layer,
            "scale": scale,
            "layer_invariance": layer_invariance,
            "elapsed": elapsed,
        }

    return results


# ==================== 主流程 ====================
def run_all_experiments(model_name, round_num=1):
    plog(f"Phase 486: {model_name} (R{round_num})")
    plog(f"GPU: {torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'N/A'}")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model info: class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    W_U = get_W_U(model, model_name)
    plog(f"W_U: shape={W_U.shape}")

    all_results = {
        "phase": 486,
        "round": round_num,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Exp1: 跨层边界累积剖面
    try:
        r1 = exp1_cross_layer_accumulation(model, tokenizer, device, model_name, W_U)
        all_results["exp1_cross_layer_accumulation"] = r1
        plog(f"Exp1 done: {list(r1.keys())}")
    except Exception as e:
        plog(f"Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_cross_layer_accumulation"] = {"error": str(e)}

    # 保存部分结果
    save_partial(model_name, round_num, all_results, "exp1")

    # Exp2: 多层联合消融
    try:
        r2 = exp2_multi_layer_ablation(model, tokenizer, device, model_name, W_U)
        all_results["exp2_multi_layer_ablation"] = r2
        plog(f"Exp2 done: {list(r2.keys())}")
    except Exception as e:
        plog(f"Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_multi_layer_ablation"] = {"error": str(e)}

    save_partial(model_name, round_num, all_results, "exp2")

    # Exp3: 格式子空间逐层
    try:
        r3 = exp3_format_subspace_per_layer(model, tokenizer, device, model_name, W_U)
        all_results["exp3_format_subspace_per_layer"] = r3
        plog(f"Exp3 done: {list(r3.keys())}")
    except Exception as e:
        plog(f"Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_format_subspace_per_layer"] = {"error": str(e)}

    save_partial(model_name, round_num, all_results, "exp3")

    # Exp4: 跨层关系不变性
    try:
        r4 = exp4_cross_layer_relation_invariance(model, tokenizer, device, model_name, W_U)
        all_results["exp4_cross_layer_relation_invariance"] = r4
        plog(f"Exp4 done: {list(r4.keys())}")
    except Exception as e:
        plog(f"Exp4 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp4_cross_layer_relation_invariance"] = {"error": str(e)}

    save_partial(model_name, round_num, all_results, "exp4")

    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    # 保存完整结果
    result_path = f"results/glm5/phase486_{model_name}_r{round_num}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    plog(f"Results saved to {result_path}")

    return all_results


def save_partial(model_name, round_num, results, exp_name):
    """保存部分结果防止中断"""
    path = f"results/glm5/phase486_{model_name}_r{round_num}_partial.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        plog(f"  Partial save failed: {e}")


# ==================== R2: 确认测试 ====================
def run_r2_confirmation(model_name):
    """R2确认: 基于R1结果，只确认最重要的发现"""
    plog(f"Phase 486 R2: {model_name}")

    # 读取R1结果
    r1_path = f"results/glm5/phase486_{model_name}_r1.json"
    if not os.path.exists(r1_path):
        plog(f"R1 results not found, running full R1 instead")
        return run_all_experiments(model_name, round_num=1)

    with open(r1_path, "r", encoding="utf-8") as f:
        r1_data = json.load(f)

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    r2_results = {
        "phase": 486, "round": 2, "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # R2确认1: 找到attn/MLP peak层的精确贡献
    # 从R1的Exp1中读取profile，确认peak层
    exp1 = r1_data.get("exp1_cross_layer_accumulation", {})
    if "error" not in exp1:
        confirm_cats = PRIORITY_CATS.get(model_name, ["clothing"])
        confirm_results = {}

        for cat_name in confirm_cats[:2]:  # 只确认2个类别
            best_layer = BEST_LAYERS[model_name][cat_name]
            spec_vec, spec_norm = get_specific_direction(
                model, tokenizer, device, model_name, cat_name, best_layer
            )
            if spec_vec is None or spec_norm < 1e-6:
                continue

            b_hat = spec_vec / spec_norm
            target_idx = DCF_DIM_NAMES.index(cat_name) if cat_name in DCF_DIM_NAMES else 0

            # 获取R1的peak层
            cat_r1 = exp1.get(cat_name, {})
            attn_peak = int(cat_r1.get("attn_peak_layer", best_layer))
            mlp_peak = int(cat_r1.get("mlp_peak_layer", best_layer))

            plog(f"  R2 confirm {cat_name}: attn_peak=L{attn_peak}, mlp_peak=L{mlp_peak}")

            # 用更多对象确认
            template = RELATION_TEMPLATES["kind_of"]
            cat_objs = CATEGORIES_R2.get(cat_name, CATEGORIES[cat_name])
            neighbor_cats = BEST_NEIGHBORS[model_name].get(cat_name, [])

            # 测peak层的精确attn/MLP贡献
            for peak_layer in [attn_peak, mlp_peak]:
                if peak_layer >= info.n_layers:
                    continue
                layer = layers_list[peak_layer]
                cat_attn_projs = []
                cat_mlp_projs = []

                for obj in cat_objs:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                    cap_attn = {}
                    cap_mlp = {}
                    done = [False, False]

                    def make_hook(store, idx):
                        def hook_fn(module, inp, output):
                            if not done[idx]:
                                if isinstance(output, tuple):
                                    store["v"] = output[0].detach().float().cpu()
                                else:
                                    store["v"] = output.detach().float().cpu()
                                done[idx] = True
                        return hook_fn

                    h1 = layer.self_attn.register_forward_hook(make_hook(cap_attn, 0))
                    h2 = layer.mlp.register_forward_hook(make_hook(cap_mlp, 1))
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    h1.remove(); h2.remove()

                    if "v" in cap_attn:
                        cat_attn_projs.append(float(np.dot(cap_attn["v"][0, pos].numpy(), b_hat)))
                    if "v" in cap_mlp:
                        cat_mlp_projs.append(float(np.dot(cap_mlp["v"][0, pos].numpy(), b_hat)))

                # 邻居
                neigh_attn_projs = []
                neigh_mlp_projs = []
                for nc in neighbor_cats[:2]:
                    for obj in CATEGORIES.get(nc, [])[:2]:
                        prompt = template.format(obj=obj)
                        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                        cap_attn = {}
                        cap_mlp = {}
                        done = [False, False]

                        def make_hook2(store, idx):
                            def hook_fn(module, inp, output):
                                if not done[idx]:
                                    if isinstance(output, tuple):
                                        store["v"] = output[0].detach().float().cpu()
                                    else:
                                        store["v"] = output.detach().float().cpu()
                                    done[idx] = True
                            return hook_fn

                        h1 = layer.self_attn.register_forward_hook(make_hook2(cap_attn, 0))
                        h2 = layer.mlp.register_forward_hook(make_hook2(cap_mlp, 1))
                        with torch.no_grad():
                            model(input_ids=input_ids, attention_mask=attention_mask)
                        h1.remove(); h2.remove()

                        if "v" in cap_attn:
                            neigh_attn_projs.append(float(np.dot(cap_attn["v"][0, pos].numpy(), b_hat)))
                        if "v" in cap_mlp:
                            neigh_mlp_projs.append(float(np.dot(cap_mlp["v"][0, pos].numpy(), b_hat)))

                cat_attn_mean = np.mean(cat_attn_projs) if cat_attn_projs else 0
                neigh_attn_mean = np.mean(neigh_attn_projs) if neigh_attn_projs else 0
                cat_mlp_mean = np.mean(cat_mlp_projs) if cat_mlp_projs else 0
                neigh_mlp_mean = np.mean(neigh_mlp_projs) if neigh_mlp_projs else 0

                confirm_results[f"{cat_name}_L{peak_layer}"] = {
                    "attn_proj_diff": float(cat_attn_mean - neigh_attn_mean),
                    "mlp_proj_diff": float(cat_mlp_mean - neigh_mlp_mean),
                    "n_cat_samples": len(cat_attn_projs),
                    "n_neigh_samples": len(neigh_attn_projs),
                }

        r2_results["peak_layer_confirmation"] = confirm_results

    # R2确认2: 关系不变性在peak层 (非best_layer)
    # 从R1的Exp4读取
    exp4 = r1_data.get("exp4_cross_layer_relation_invariance", {})
    if "error" not in exp4:
        invariance_confirm = {}
        for cat_name in ["clothing", "fruit"]:
            best_layer = BEST_LAYERS[model_name][cat_name]
            spec_vec, spec_norm = get_specific_direction(
                model, tokenizer, device, model_name, cat_name, best_layer
            )
            if spec_vec is None or spec_norm < 1e-6:
                continue
            b_hat = spec_vec / spec_norm
            target_idx = DCF_DIM_NAMES.index(cat_name) if cat_name in DCF_DIM_NAMES else 0

            # 在best_layer ± 5测关系不变性
            test_layer = min(info.n_layers - 1, best_layer + 5)
            scale = 0.1
            relations = ["kind_of", "used_for", "found_in"]

            rel_target_deltas = {}
            for rel_key in relations:
                template = RELATION_TEMPLATES[rel_key]
                deltas = []
                for obj in CATEGORIES[cat_name]:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                    with torch.no_grad():
                        out_base = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits_base = out_base.logits[0, -1].float().cpu().numpy()
                    dcf_base = compute_dcf(logits_base, tokenizer)

                    inject_vec = torch.tensor(scale * spec_norm * b_hat,
                                               dtype=torch.bfloat16, device=device)
                    added = [False]
                    def make_inject(ivec, position):
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

                    h_inj = layers_list[test_layer].register_forward_hook(make_inject(inject_vec, pos))
                    with torch.no_grad():
                        out_inj = model(input_ids=input_ids, attention_mask=attention_mask)
                    h_inj.remove()

                    logits_inj = out_inj.logits[0, -1].float().cpu().numpy()
                    dcf_inj = compute_dcf(logits_inj, tokenizer)
                    deltas.append(dcf_inj - dcf_base)

                if deltas:
                    mean_delta = np.mean(deltas, axis=0)
                    rel_target_deltas[rel_key] = float(mean_delta[target_idx])

            if len(rel_target_deltas) >= 2:
                vals = list(rel_target_deltas.values())
                delta_mean = np.mean(vals)
                delta_range = max(vals) - min(vals)
                relative_range = delta_range / (abs(delta_mean) + 0.01)
                invariance_confirm[cat_name] = {
                    "test_layer": test_layer,
                    "scale": scale,
                    "rel_target_deltas": rel_target_deltas,
                    "delta_mean": float(delta_mean),
                    "delta_range": float(delta_range),
                    "relative_range": float(relative_range),
                }

        r2_results["invariance_confirmation"] = invariance_confirm

    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    r2_path = f"results/glm5/phase486_{model_name}_r2.json"
    with open(r2_path, "w", encoding="utf-8") as f:
        json.dump(r2_results, f, indent=2, ensure_ascii=False, default=str)
    plog(f"R2 results saved to {r2_path}")

    return r2_results


# ==================== 入口 ====================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase486_cross_layer_boundary.py <model_name> [round]")
        print("  model_name: qwen3 / glm4 / deepseek7b")
        print("  round: 1 (default) or 2")
        sys.exit(1)

    model_name = sys.argv[1]
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        sys.exit(1)

    os.makedirs("results/glm5", exist_ok=True)

    if round_num == 1:
        run_all_experiments(model_name, round_num=1)
    elif round_num == 2:
        run_r2_confirmation(model_name)
    else:
        print(f"Unknown round: {round_num}")

    plog("Phase 486 complete!")
