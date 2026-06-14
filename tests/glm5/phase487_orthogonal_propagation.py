"""
Phase 487: 正交成分因果测试、连续段消融与反对层验证
======================================================

Phase 486核心发现:
1. 边界信号逐层累积，非单层写入
2. 存在"反对层"——某些层MLP/attn反对边界方向
3. Qwen3 clothing: MLP B_c投影消融仅5%，但完整MLP消融35%
   → 说明MLP通过正交成分经后续层旋转贡献边界
4. GLM4 fruit: 5层MLP投影消融达97.6%幅度闭环
5. DS7B fruit: 多层消融方向反转(cos=0.992但target_D为正)

Phase 487核心目标:
- Exp1: 正交成分因果测试 — 分离MLP的B_c投影成分和正交成分，分别消融
- Exp2: 连续段消融 — 测试L0-L10, L10-L20, L20-L30等连续段的贡献
- Exp3: 反对层因果验证 — 增强/消融反对层，验证其功能
- Exp4: 格式-语义分离重测 — 用更多模板和正交分解

用法:
  python tests/glm5/phase487_orthogonal_propagation.py qwen3
  python tests/glm5/phase487_orthogonal_propagation.py glm4
  python tests/glm5/phase487_orthogonal_propagation.py deepseek7b
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
    },
    "glm4": {
        "fruit": ["plant", "food"], "animal": ["food", "clothing"],
        "clothing": ["furniture", "plant"], "food": ["plant", "fruit"],
    },
    "deepseek7b": {
        "fruit": ["plant", "food"], "animal": ["food", "clothing"],
        "clothing": ["furniture", "plant"], "food": ["plant", "fruit"],
    },
}

BEST_LAYERS = {
    "qwen3": {"fruit": 32, "animal": 33, "clothing": 30},
    "glm4":  {"fruit": 27, "animal": 38, "clothing": 39},
    "deepseek7b": {"fruit": 26, "animal": 27, "clothing": 23, "food": 27},
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

# Phase 486发现的关键实验对象
PRIORITY_CATS = {
    "qwen3": ["clothing", "fruit"],      # clothing: MLP投影5% vs 完整35%; fruit: 后期修正
    "glm4":  ["fruit", "clothing"],       # fruit: 97.6%闭环; clothing: cos高但幅度25%
    "deepseek7b": ["fruit", "food"],      # fruit: 方向反转; food: 104.6%单层闭环
}

# 反对层(来自Phase 486 Exp1)
OPPOSITION_LAYERS = {
    "qwen3": {
        "clothing": {"attn_opp": [10, 15, 17], "mlp_opp": []},  # L10-22 attn反对
        "fruit": {"attn_opp": [], "mlp_opp": [33, 34]},           # L33-34 MLP反对
    },
    "glm4": {
        "fruit": {"attn_opp": [], "mlp_opp": [34, 39]},           # L34,39 MLP反对
        "clothing": {"attn_opp": [], "mlp_opp": []},
    },
    "deepseek7b": {
        "fruit": {"attn_opp": [], "mlp_opp": []},
        "food": {"attn_opp": [27], "mlp_opp": []},                # L27 attn反对
    },
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
                lambda m, i, o: cap.__setitem__("v", o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu())
            )
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "v" in cap:
                resids.append(cap["v"][0, pos].numpy())
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


def get_dcf_baseline(model, tokenizer, device, cat_name, template_key="kind_of"):
    """获取DCF baseline(无注入)"""
    template = RELATION_TEMPLATES[template_key]
    cat_objs = CATEGORIES[cat_name]
    baselines = []
    for obj in cat_objs:
        prompt = template.format(obj=obj)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0, -1].float().cpu().numpy()
        baselines.append(compute_dcf(logits, tokenizer))
    return np.mean(baselines, axis=0)


# ==================== Exp1: 正交成分因果测试 ★★★★★ ====================
def exp1_orthogonal_component_causal(model, tokenizer, device, model_name, W_U):
    """
    核心实验: 分离MLP输出的B_c投影成分和正交成分，分别消融

    Phase 486发现: Qwen3 clothing MLP B_c投影消融仅5%，但完整MLP消融35%
    → 假设: MLP通过正交成分(经后续层旋转)贡献边界

    方法:
    1. 在关键层捕获MLP输出
    2. 分解为: MLP_out = Proj_Bc(MLP_out) + Orth_Bc(MLP_out)
    3. 分别消融三组:
       a. 只消融Proj_Bc (Phase 486已做)
       b. 只消融Orth_Bc (新实验！)
       c. 消融完整MLP_out (Phase 485已做)
    4. 对比三者对DCF的影响

    关键层选择:
    - Qwen3 clothing: L25, L30, L35 (MLP贡献显著的层)
    - GLM4 fruit: L22, L27, L32 (97.6%闭环相关层)
    """
    plog("=== Exp1: 正交成分因果测试 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers
    cat_names = PRIORITY_CATS.get(model_name, ["clothing", "fruit"])

    results = {}

    for cat_name in cat_names:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name}: orthogonal component causal test...")
        t0 = time.time()

        # 构造B_c方向
        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"    Skip {cat_name}: spec_norm={spec_norm:.4f}")
            continue

        b_hat = spec_vec / spec_norm
        target_idx = DCF_DIM_NAMES.index(cat_name) if cat_name in DCF_DIM_NAMES else 0

        template = RELATION_TEMPLATES["kind_of"]
        cat_objs = CATEGORIES[cat_name]

        # ---- 方向级remove baseline ----
        direction_remove_deltas = []
        for obj in cat_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

            with torch.no_grad():
                out_base = model(input_ids=input_ids, attention_mask=attention_mask)
            logits_base = out_base.logits[0, -1].float().cpu().numpy()
            dcf_base = compute_dcf(logits_base, tokenizer)

            # 方向级remove: 在best_layer移除B_c投影
            added = [False]
            def make_remove_hook(b_hat_np, position):
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

            h_rm = layers_list[best_layer].register_forward_hook(make_remove_hook(b_hat, pos))
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

        # ---- 选择测试层 ----
        # Phase 486发现的关键MLP贡献层
        if model_name == "qwen3" and cat_name == "clothing":
            test_layers = [25, 30, 34]  # MLP peak附近
        elif model_name == "glm4" and cat_name == "fruit":
            test_layers = [22, 27, 32]  # 97.6%闭环相关层
        elif model_name == "deepseek7b" and cat_name == "food":
            test_layers = [26, 27]  # 末层爆发
        elif model_name == "deepseek7b" and cat_name == "fruit":
            test_layers = [21, 26]
        else:
            # 通用: best_layer附近
            test_layers = sorted(set([
                max(0, best_layer - 5), best_layer,
                min(n_layers - 1, best_layer + 5)
            ]))

        # 过滤超出范围的层
        test_layers = [l for l in test_layers if l < n_layers]

        ablation_results = {}

        for test_l in test_layers:
            plog(f"    L{test_l}: 3-way ablation (proj/orth/full)...")
            layer_abl = {}

            for abl_type in ["proj_bc", "orth_bc", "full_mlp"]:
                mlp_remove_deltas = []

                for obj in cat_objs:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                    # 先获取baseline
                    with torch.no_grad():
                        out_base = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits_base = out_base.logits[0, -1].float().cpu().numpy()
                    dcf_base = compute_dcf(logits_base, tokenizer)

                    # 在test_l的MLP上添加消融hook
                    added = [False]
                    def make_orth_hook(b_hat_np, position, ablation_mode):
                        def hook_fn(module, inp, output):
                            if not added[0]:
                                if isinstance(output, tuple):
                                    out = output[0].clone()
                                else:
                                    out = output.clone()
                                mlp_out_np = out[0, position, :].float().cpu().numpy()

                                if ablation_mode == "proj_bc":
                                    # 只去除B_c投影成分
                                    proj = np.dot(mlp_out_np, b_hat_np) * b_hat_np
                                    out[0, position, :] -= torch.tensor(proj, dtype=out.dtype, device=out.device)
                                elif ablation_mode == "orth_bc":
                                    # 只去除正交成分: MLP_out - Proj = Orth
                                    proj = np.dot(mlp_out_np, b_hat_np) * b_hat_np
                                    orth = mlp_out_np - proj
                                    out[0, position, :] -= torch.tensor(orth, dtype=out.dtype, device=out.device)
                                elif ablation_mode == "full_mlp":
                                    # 去除完整MLP输出
                                    out[0, position, :] = 0

                                added[0] = True
                                if isinstance(output, tuple):
                                    return (out,) + output[1:]
                                return out
                            return output
                        return hook_fn

                    h = layers_list[test_l].mlp.register_forward_hook(
                        make_orth_hook(b_hat, pos, abl_type)
                    )
                    with torch.no_grad():
                        out_abl = model(input_ids=input_ids, attention_mask=attention_mask)
                    h.remove()

                    logits_abl = out_abl.logits[0, -1].float().cpu().numpy()
                    dcf_abl = compute_dcf(logits_abl, tokenizer)
                    mlp_remove_deltas.append(dcf_abl - dcf_base)

                if mlp_remove_deltas:
                    abl_mean = np.mean(mlp_remove_deltas, axis=0)
                    abl_target = abl_mean[target_idx]
                    amplitude_ratio = abs(abl_target / dir_remove_target) if abs(dir_remove_target) > 0.01 else 0
                    cos_with_dir = float(np.dot(abl_mean, dir_remove_mean) /
                                        (np.linalg.norm(abl_mean) * np.linalg.norm(dir_remove_mean) + 1e-10))
                    selectivity = abs(abl_target) / (max(abs(abl_mean[i]) for i in range(len(abl_mean)) if i != target_idx) + 0.01)

                    layer_abl[abl_type] = {
                        "target_delta": float(abl_target),
                        "amplitude_ratio": float(amplitude_ratio),
                        "cos_with_direction_remove": float(cos_with_dir),
                        "selectivity": float(selectivity),
                        "dcf_delta": [float(x) for x in abl_mean],
                    }
                    plog(f"      {abl_type:10s}: target_D={abl_target:.2f}, amp={amplitude_ratio:.1%}, "
                         f"cos={cos_with_dir:.3f}, sel={selectivity:.2f}")

            ablation_results[f"L{test_l}"] = layer_abl

        elapsed = time.time() - t0
        plog(f"    {cat_name} orthogonal test done ({elapsed:.0f}s)")

        results[cat_name] = {
            "best_layer": best_layer,
            "direction_remove_target": float(dir_remove_target),
            "test_layers": test_layers,
            "ablation_results": ablation_results,
            "elapsed": elapsed,
        }

    return results


# ==================== Exp2: 连续段消融 ★★★★ ====================
def exp2_segment_ablation(model, tokenizer, device, model_name, W_U):
    """
    连续层段消融，而非稀疏点消融

    Phase 486只在稀疏点(L20,L25,L30等)消融，可能遗漏连续段内的累积效应。
    现在测试连续段的完整贡献。

    段划分:
    - Qwen3 (36层): L0-9, L10-17, L18-24, L25-29, L30-33, L34-35
    - GLM4 (40层): L0-9, L10-19, L20-26, L27-32, L33-39
    - DS7B (28层): L0-9, L10-19, L20-26, L27
    """
    plog("=== Exp2: 连续段消融 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers
    cat_names = PRIORITY_CATS.get(model_name, ["clothing", "fruit"])

    results = {}

    for cat_name in cat_names:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name}: segment ablation...")
        t0 = time.time()

        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"    Skip {cat_name}")
            continue

        b_hat = spec_vec / spec_norm
        target_idx = DCF_DIM_NAMES.index(cat_name) if cat_name in DCF_DIM_NAMES else 0

        template = RELATION_TEMPLATES["kind_of"]
        cat_objs = CATEGORIES[cat_name]

        # 方向级remove baseline
        direction_remove_deltas = []
        for obj in cat_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            with torch.no_grad():
                out_base = model(input_ids=input_ids, attention_mask=attention_mask)
            logits_base = out_base.logits[0, -1].float().cpu().numpy()
            dcf_base = compute_dcf(logits_base, tokenizer)

            added = [False]
            def make_remove_hook(b_hat_np, position):
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

            h_rm = layers_list[best_layer].register_forward_hook(make_remove_hook(b_hat, pos))
            with torch.no_grad():
                out_rm = model(input_ids=input_ids, attention_mask=attention_mask)
            h_rm.remove()
            logits_rm = out_rm.logits[0, -1].float().cpu().numpy()
            dcf_rm = compute_dcf(logits_rm, tokenizer)
            direction_remove_deltas.append(dcf_rm - dcf_base)

        dir_remove_mean = np.mean(direction_remove_deltas, axis=0)
        dir_remove_target = dir_remove_mean[target_idx]

        # 定义段
        if n_layers <= 28:
            segments = [
                ("L0-9", 0, 10),
                ("L10-19", 10, 20),
                ("L20-end", 20, n_layers),
            ]
        elif n_layers <= 36:
            segments = [
                ("L0-9", 0, 10),
                ("L10-17", 10, 18),
                ("L18-24", 18, 25),
                ("L25-29", 25, 30),
                ("L30-33", 30, 34),
                ("L34-end", 34, n_layers),
            ]
        else:  # 40层
            segments = [
                ("L0-9", 0, 10),
                ("L10-19", 10, 20),
                ("L20-26", 20, 27),
                ("L27-32", 27, 33),
                ("L33-end", 33, n_layers),
            ]

        # 消融类型: 完整MLP, B_c投影, 正交成分
        abl_types = ["full_mlp", "proj_bc", "orth_bc"]

        segment_results = {}

        for seg_name, seg_start, seg_end in segments:
            seg_layers = list(range(seg_start, min(seg_end, n_layers)))
            plog(f"    Segment {seg_name}: {len(seg_layers)} layers")

            seg_abl = {}

            for abl_type in abl_types:
                mlp_remove_deltas = []

                for obj in cat_objs:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                    with torch.no_grad():
                        out_base = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits_base = out_base.logits[0, -1].float().cpu().numpy()
                    dcf_base = compute_dcf(logits_base, tokenizer)

                    # 在段内所有层的MLP上添加消融hook
                    hooks = []
                    added_flags = [False] * len(seg_layers)

                    def make_seg_hook(b_hat_np, position, ablation_mode, flags_list, idx):
                        def hook_fn(module, inp, output):
                            if not flags_list[idx]:
                                if isinstance(output, tuple):
                                    out = output[0].clone()
                                else:
                                    out = output.clone()
                                mlp_out_np = out[0, position, :].float().cpu().numpy()

                                if ablation_mode == "proj_bc":
                                    proj = np.dot(mlp_out_np, b_hat_np) * b_hat_np
                                    out[0, position, :] -= torch.tensor(proj, dtype=out.dtype, device=out.device)
                                elif ablation_mode == "orth_bc":
                                    proj = np.dot(mlp_out_np, b_hat_np) * b_hat_np
                                    orth = mlp_out_np - proj
                                    out[0, position, :] -= torch.tensor(orth, dtype=out.dtype, device=out.device)
                                elif ablation_mode == "full_mlp":
                                    out[0, position, :] = 0

                                flags_list[idx] = True
                                if isinstance(output, tuple):
                                    return (out,) + output[1:]
                                return out
                            return output
                        return hook_fn

                    for i, al in enumerate(seg_layers):
                        hooks.append(layers_list[al].mlp.register_forward_hook(
                            make_seg_hook(b_hat, pos, abl_type, added_flags, i)
                        ))

                    with torch.no_grad():
                        out_abl = model(input_ids=input_ids, attention_mask=attention_mask)

                    for h in hooks:
                        h.remove()

                    logits_abl = out_abl.logits[0, -1].float().cpu().numpy()
                    dcf_abl = compute_dcf(logits_abl, tokenizer)
                    mlp_remove_deltas.append(dcf_abl - dcf_base)

                if mlp_remove_deltas:
                    abl_mean = np.mean(mlp_remove_deltas, axis=0)
                    abl_target = abl_mean[target_idx]
                    amplitude_ratio = abs(abl_target / dir_remove_target) if abs(dir_remove_target) > 0.01 else 0
                    cos_with_dir = float(np.dot(abl_mean, dir_remove_mean) /
                                        (np.linalg.norm(abl_mean) * np.linalg.norm(dir_remove_mean) + 1e-10))

                    seg_abl[abl_type] = {
                        "target_delta": float(abl_target),
                        "amplitude_ratio": float(amplitude_ratio),
                        "cos_with_direction_remove": float(cos_with_dir),
                        "dcf_delta": [float(x) for x in abl_mean],
                    }
                    plog(f"      {abl_type:10s}: target_D={abl_target:.2f}, amp={amplitude_ratio:.1%}, cos={cos_with_dir:.3f}")

            segment_results[seg_name] = {
                "layers": seg_layers,
                "ablation_results": seg_abl,
            }

        elapsed = time.time() - t0
        plog(f"    {cat_name} segment ablation done ({elapsed:.0f}s)")

        results[cat_name] = {
            "best_layer": best_layer,
            "direction_remove_target": float(dir_remove_target),
            "segments": segment_results,
            "elapsed": elapsed,
        }

    return results


# ==================== Exp3: 反对层因果验证 ★★★★ ====================
def exp3_opposition_layer_causal(model, tokenizer, device, model_name, W_U):
    """
    反对层因果验证: 增强或消融反对层，验证其对边界的真实功能

    Phase 486发现: 某些层的attn/MLP反对B_c方向
    - Qwen3 clothing: L10-22 attn反对 (L17 peak = -5.26)
    - Qwen3 fruit: L33-34 MLP反对 (-12.2, -7.0)
    - GLM4 fruit: L34, L39 MLP反对 (-1.4, -2.3)

    验证方法:
    1. 增强(加倍)反对层MLP/attn输出 → 边界应被削弱
    2. 消融(归零)反对层MLP/attn输出 → 边界应被增强
    3. 增强(加倍)支持层MLP/attn输出 → 边界应被增强
    4. 消融(归零)支持层MLP/attn输出 → 边界应被削弱
    """
    plog("=== Exp3: 反对层因果验证 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers

    opp_config = OPPOSITION_LAYERS.get(model_name, {})
    cat_names = PRIORITY_CATS.get(model_name, ["clothing", "fruit"])

    results = {}

    for cat_name in cat_names:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name}: opposition layer causal test...")
        t0 = time.time()

        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"    Skip {cat_name}")
            continue

        b_hat = spec_vec / spec_norm
        target_idx = DCF_DIM_NAMES.index(cat_name) if cat_name in DCF_DIM_NAMES else 0

        template = RELATION_TEMPLATES["kind_of"]
        cat_objs = CATEGORIES[cat_name]

        # 获取opposition层
        opp_info = opp_config.get(cat_name, {"attn_opp": [], "mlp_opp": []})
        attn_opp_layers = [l for l in opp_info.get("attn_opp", []) if l < n_layers]
        mlp_opp_layers = [l for l in opp_info.get("mlp_opp", []) if l < n_layers]

        # 支持层: 从Phase 486 Exp1读取 (best_layer附近)
        support_layers = sorted(set([
            max(0, best_layer - 2), max(0, best_layer - 1), best_layer
        ]))
        support_layers = [l for l in support_layers if l < n_layers]

        # 如果没有预定义的反对层，从Phase 486结果推断
        if not attn_opp_layers and not mlp_opp_layers and model_name == "qwen3" and cat_name == "clothing":
            attn_opp_layers = [17]  # Phase 486发现L17 attn反对最强
        if not mlp_opp_layers and model_name == "qwen3" and cat_name == "fruit":
            mlp_opp_layers = [33, 34]  # Phase 486发现L33-34 MLP反对
        if not mlp_opp_layers and model_name == "glm4" and cat_name == "fruit":
            mlp_opp_layers = [34, 39]

        plog(f"    Attn opp layers: {attn_opp_layers}")
        plog(f"    MLP opp layers: {mlp_opp_layers}")
        plog(f"    Support layers: {support_layers}")

        # ---- baseline ----
        baseline_deltas = []
        for obj in cat_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[0, -1].float().cpu().numpy()
            baseline_deltas.append(compute_dcf(logits, tokenizer))
        baseline_mean = np.mean(baseline_deltas, axis=0)

        intervention_results = {}

        # ---- 实验1: 消融反对层attn (如果有的话) ----
        for opp_type, opp_layers, component_name in [
            ("attn", attn_opp_layers, "self_attn"),
            ("mlp", mlp_opp_layers, "mlp"),
        ]:
            if not opp_layers:
                continue

            # 消融(归零)反对层
            for action in ["ablate", "double"]:
                action_name = f"{action}_opp_{opp_type}"
                plog(f"    {action_name}: {opp_layers}")

                intervention_deltas = []
                for obj in cat_objs:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                    hooks = []
                    added_flags = [False] * len(opp_layers)

                    def make_opp_hook(position, act, flags_list, idx):
                        def hook_fn(module, inp, output):
                            if not flags_list[idx]:
                                if isinstance(output, tuple):
                                    out = output[0].clone()
                                else:
                                    out = output.clone()

                                if act == "ablate":
                                    out[0, position, :] = 0
                                elif act == "double":
                                    out[0, position, :] *= 2

                                flags_list[idx] = True
                                if isinstance(output, tuple):
                                    return (out,) + output[1:]
                                return out
                            return output
                        return hook_fn

                    for i, ol in enumerate(opp_layers):
                        comp = getattr(layers_list[ol], component_name)
                        hooks.append(comp.register_forward_hook(
                            make_opp_hook(pos, action, added_flags, i)
                        ))

                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attention_mask)
                    for h in hooks:
                        h.remove()

                    logits = out.logits[0, -1].float().cpu().numpy()
                    intervention_deltas.append(compute_dcf(logits, tokenizer) - baseline_mean)

                if intervention_deltas:
                    int_mean = np.mean(intervention_deltas, axis=0)
                    intervention_results[action_name] = {
                        "target_delta": float(int_mean[target_idx]),
                        "dcf_delta": [float(x) for x in int_mean],
                        "layers": opp_layers,
                        "action": action,
                        "component": opp_type,
                    }
                    plog(f"      target_D={int_mean[target_idx]:.2f}")

            # 消融(归零)支持层 (对比)
            for action in ["ablate", "double"]:
                action_name = f"{action}_support_{opp_type}"
                # 用best_layer附近的层作为支持层
                test_layers = support_layers if opp_type == "mlp" else [max(0, best_layer - 3)]
                if not test_layers:
                    continue

                plog(f"    {action_name}: {test_layers}")

                intervention_deltas = []
                for obj in cat_objs:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                    hooks = []
                    added_flags = [False] * len(test_layers)

                    def make_supp_hook(position, act, flags_list, idx):
                        def hook_fn(module, inp, output):
                            if not flags_list[idx]:
                                if isinstance(output, tuple):
                                    out = output[0].clone()
                                else:
                                    out = output.clone()

                                if act == "ablate":
                                    out[0, position, :] = 0
                                elif act == "double":
                                    out[0, position, :] *= 2

                                flags_list[idx] = True
                                if isinstance(output, tuple):
                                    return (out,) + output[1:]
                                return out
                            return output
                        return hook_fn

                    for i, tl in enumerate(test_layers):
                        comp = getattr(layers_list[tl], component_name)
                        hooks.append(comp.register_forward_hook(
                            make_supp_hook(pos, action, added_flags, i)
                        ))

                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attention_mask)
                    for h in hooks:
                        h.remove()

                    logits = out.logits[0, -1].float().cpu().numpy()
                    intervention_deltas.append(compute_dcf(logits, tokenizer) - baseline_mean)

                if intervention_deltas:
                    int_mean = np.mean(intervention_deltas, axis=0)
                    intervention_results[action_name] = {
                        "target_delta": float(int_mean[target_idx]),
                        "dcf_delta": [float(x) for x in int_mean],
                        "layers": test_layers,
                        "action": action,
                        "component": opp_type,
                    }
                    plog(f"      target_D={int_mean[target_idx]:.2f}")

        elapsed = time.time() - t0
        plog(f"    {cat_name} opposition test done ({elapsed:.0f}s)")

        results[cat_name] = {
            "best_layer": best_layer,
            "attn_opp_layers": attn_opp_layers,
            "mlp_opp_layers": mlp_opp_layers,
            "support_layers": support_layers,
            "intervention_results": intervention_results,
            "elapsed": elapsed,
        }

    return results


# ==================== Exp4: 格式-语义分离升级 ★★★ ====================
def exp4_format_semantic_separation(model, tokenizer, device, model_name, W_U):
    """
    用更多模板和正交分解升级格式-语义分离

    Phase 486用4种模板，可能混入语义差异。
    现在用更精心设计的模板:

    1. 同语义不同格式 (纯粹格式差异):
       - "The apple is a kind of" vs "Apple is a type of" vs "An apple belongs to"
    2. 同格式不同语义 (纯粹语义差异):
       - "The apple is a kind of" vs "The dog is a kind of" vs "The shirt is a kind of"

    然后把B_c分解为:
       B_c = B_c^{sem} + B_c^{format} + B_c^{residual}
    """
    plog("=== Exp4: 格式-语义分离升级 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers
    cat_names = PRIORITY_CATS.get(model_name, ["clothing", "fruit"])

    # 同语义不同格式模板
    same_semantic_different_format = [
        "The {obj} is a kind of",
        "A {obj} is a type of",
        "{obj} belongs to the category of",
        "The {obj} can be classified as",
        "The {obj} is considered a",
    ]

    # 测试对象
    test_objects = ["apple", "dog", "car", "hat", "bread", "tree", "book", "chair"]

    results = {}

    for cat_name in cat_names:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name}: format-semantic separation...")
        t0 = time.time()

        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"    Skip {cat_name}")
            continue

        b_hat = spec_vec / spec_norm
        target_idx = DCF_DIM_NAMES.index(cat_name) if cat_name in DCF_DIM_NAMES else 0

        # 采样层
        if n_layers <= 40:
            scan_layers = sorted(set(list(range(0, n_layers, 5)) + [best_layer, n_layers - 1]))
        else:
            step = max(1, n_layers // 10)
            scan_layers = sorted(set(list(range(0, n_layers, step)) + [best_layer, n_layers - 1]))

        format_semantic_profile = {}

        for scan_l in scan_layers:
            layer = layers_list[scan_l]

            # ---- 收集格式差异 (同语义不同格式) ----
            format_diffs = []
            for obj in test_objects[:4]:  # 4个对象
                obj_resids = []
                for tmpl in same_semantic_different_format[:3]:  # 3种格式
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
                        obj_resids.append(cap["v"][0, pos].numpy())

                if len(obj_resids) >= 2:
                    for i in range(len(obj_resids)):
                        for j in range(i+1, len(obj_resids)):
                            format_diffs.append(obj_resids[i] - obj_resids[j])

            # ---- 收集语义差异 (同格式不同语义) ----
            semantic_diffs = []
            tmpl = same_semantic_different_format[0]
            sem_resids = {}
            for obj in test_objects[:6]:
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
                    sem_resids[obj] = cap["v"][0, pos].numpy()

            obj_list = list(sem_resids.keys())
            for i in range(len(obj_list)):
                for j in range(i+1, len(obj_list)):
                    semantic_diffs.append(sem_resids[obj_list[i]] - sem_resids[obj_list[j]])

            # ---- SVD提取格式和语义子空间 ----
            format_dirs = []
            format_energy = 0
            if len(format_diffs) >= 2:
                format_matrix = np.array(format_diffs)
                try:
                    U, S, Vt = np.linalg.svd(format_matrix, full_matrices=False)
                    format_dirs = Vt[:3]  # 前3个格式方向
                    format_energy = float(np.sum(S[:3]**2) / (np.sum(S**2) + 1e-10))
                except Exception:
                    pass

            semantic_dirs = []
            semantic_energy = 0
            if len(semantic_diffs) >= 2:
                semantic_matrix = np.array(semantic_diffs)
                try:
                    U, S, Vt = np.linalg.svd(semantic_matrix, full_matrices=False)
                    semantic_dirs = Vt[:3]  # 前3个语义方向
                    semantic_energy = float(np.sum(S[:3]**2) / (np.sum(S**2) + 1e-10))
                except Exception:
                    pass

            # ---- B_c与格式/语义方向的cos ----
            cos_format = []
            for fd in format_dirs:
                fd_norm = np.linalg.norm(fd)
                if fd_norm > 1e-10:
                    cos_format.append(float(abs(np.dot(b_hat, fd / fd_norm))))
            cos_semantic = []
            for sd in semantic_dirs:
                sd_norm = np.linalg.norm(sd)
                if sd_norm > 1e-10:
                    cos_semantic.append(float(abs(np.dot(b_hat, sd / sd_norm))))

            # ---- B_c在格式子空间中的投影比 ----
            proj_ratio_format = 0.0
            if len(format_dirs) > 0:
                Q_format = np.array(format_dirs).T  # [d_model, k]
                proj_bc_on_format = Q_format @ (Q_format.T @ b_hat)
                proj_ratio_format = float(np.linalg.norm(proj_bc_on_format) / (np.linalg.norm(b_hat) + 1e-10))

            proj_ratio_semantic = 0.0
            if len(semantic_dirs) > 0:
                Q_semantic = np.array(semantic_dirs).T
                proj_bc_on_semantic = Q_semantic @ (Q_semantic.T @ b_hat)
                proj_ratio_semantic = float(np.linalg.norm(proj_bc_on_semantic) / (np.linalg.norm(b_hat) + 1e-10))

            format_semantic_profile[str(scan_l)] = {
                "cos_bc_format_top1": cos_format[0] if cos_format else 0,
                "cos_bc_format_top2": cos_format[1] if len(cos_format) > 1 else 0,
                "cos_bc_semantic_top1": cos_semantic[0] if cos_semantic else 0,
                "cos_bc_semantic_top2": cos_semantic[1] if len(cos_semantic) > 1 else 0,
                "proj_ratio_format": proj_ratio_format,
                "proj_ratio_semantic": proj_ratio_semantic,
                "format_energy_top3": format_energy,
                "semantic_energy_top3": semantic_energy,
            }

        elapsed = time.time() - t0
        plog(f"    {cat_name} format-semantic done ({elapsed:.0f}s): {len(format_semantic_profile)} layers")

        results[cat_name] = {
            "best_layer": best_layer,
            "format_semantic_profile": format_semantic_profile,
            "elapsed": elapsed,
        }

    return results


# ==================== 主流程 ====================
def run_all_experiments(model_name, round_num=1):
    plog(f"Phase 487: {model_name} (R{round_num})")
    plog(f"GPU: {torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'N/A'}")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model info: class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    W_U = get_W_U(model, model_name)
    plog(f"W_U: shape={W_U.shape}")

    all_results = {
        "phase": 487,
        "round": round_num,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Exp1: 正交成分因果测试
    try:
        r1 = exp1_orthogonal_component_causal(model, tokenizer, device, model_name, W_U)
        all_results["exp1_orthogonal_component_causal"] = r1
        plog(f"Exp1 done: {list(r1.keys())}")
    except Exception as e:
        plog(f"Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_orthogonal_component_causal"] = {"error": str(e)}

    save_partial(model_name, round_num, all_results, "exp1")

    # Exp2: 连续段消融
    try:
        r2 = exp2_segment_ablation(model, tokenizer, device, model_name, W_U)
        all_results["exp2_segment_ablation"] = r2
        plog(f"Exp2 done: {list(r2.keys())}")
    except Exception as e:
        plog(f"Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_segment_ablation"] = {"error": str(e)}

    save_partial(model_name, round_num, all_results, "exp2")

    # Exp3: 反对层因果验证
    try:
        r3 = exp3_opposition_layer_causal(model, tokenizer, device, model_name, W_U)
        all_results["exp3_opposition_layer_causal"] = r3
        plog(f"Exp3 done: {list(r3.keys())}")
    except Exception as e:
        plog(f"Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_opposition_layer_causal"] = {"error": str(e)}

    save_partial(model_name, round_num, all_results, "exp3")

    # Exp4: 格式-语义分离升级
    try:
        r4 = exp4_format_semantic_separation(model, tokenizer, device, model_name, W_U)
        all_results["exp4_format_semantic_separation"] = r4
        plog(f"Exp4 done: {list(r4.keys())}")
    except Exception as e:
        plog(f"Exp4 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp4_format_semantic_separation"] = {"error": str(e)}

    save_partial(model_name, round_num, all_results, "exp4")

    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    # 保存完整结果
    result_path = f"results/glm5/phase487_{model_name}_r{round_num}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    plog(f"Results saved to {result_path}")

    return all_results


def save_partial(model_name, round_num, results, exp_name):
    """保存部分结果防止中断"""
    path = f"results/glm5/phase487_{model_name}_r{round_num}_partial.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        plog(f"  Partial save failed: {e}")


# ==================== R2: 确认测试 ====================
def run_r2_confirmation(model_name):
    """
    R2确认: 基于R1结果，确认最重要的发现
    - 增加对象数量到8个
    - 只确认关键类别和关键实验
    """
    plog(f"Phase 487 R2: {model_name}")

    r1_path = f"results/glm5/phase487_{model_name}_r1.json"
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
        "phase": 487, "round": 2, "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ---- R2确认1: 正交成分因果测试 (8对象) ----
    exp1 = r1_data.get("exp1_orthogonal_component_causal", {})
    if "error" not in exp1:
        confirm_cats = PRIORITY_CATS.get(model_name, ["clothing"])
        confirm_results = {}

        for cat_name in confirm_cats[:2]:
            best_layer = BEST_LAYERS[model_name][cat_name]
            spec_vec, spec_norm = get_specific_direction(
                model, tokenizer, device, model_name, cat_name, best_layer, n_obj=4
            )
            if spec_vec is None or spec_norm < 1e-6:
                continue

            b_hat = spec_vec / spec_norm
            target_idx = DCF_DIM_NAMES.index(cat_name) if cat_name in DCF_DIM_NAMES else 0

            # 获取R1中效果最好的层
            cat_r1 = exp1.get(cat_name, {})
            abl_r1 = cat_r1.get("ablation_results", {})
            # 找proj_bc vs orth_bc效果差异最大的层
            best_diff_layer = None
            max_diff = 0
            for layer_key, layer_abl in abl_r1.items():
                proj_amp = layer_abl.get("proj_bc", {}).get("amplitude_ratio", 0)
                orth_amp = layer_abl.get("orth_bc", {}).get("amplitude_ratio", 0)
                diff = abs(orth_amp - proj_amp)
                if diff > max_diff:
                    max_diff = diff
                    best_diff_layer = int(layer_key.replace("L", ""))

            if best_diff_layer is None:
                best_diff_layer = best_layer

            plog(f"  R2 confirm {cat_name}: best_diff_layer=L{best_diff_layer}")

            # 用8对象确认
            template = RELATION_TEMPLATES["kind_of"]
            cat_objs = CATEGORIES_R2.get(cat_name, CATEGORIES[cat_name])

            # 方向级remove baseline
            direction_remove_deltas = []
            for obj in cat_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                with torch.no_grad():
                    out_base = model(input_ids=input_ids, attention_mask=attention_mask)
                logits_base = out_base.logits[0, -1].float().cpu().numpy()
                dcf_base = compute_dcf(logits_base, tokenizer)

                added = [False]
                def make_remove_hook(b_hat_np, position):
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

                h_rm = layers_list[best_layer].register_forward_hook(make_remove_hook(b_hat, pos))
                with torch.no_grad():
                    out_rm = model(input_ids=input_ids, attention_mask=attention_mask)
                h_rm.remove()
                logits_rm = out_rm.logits[0, -1].float().cpu().numpy()
                dcf_rm = compute_dcf(logits_rm, tokenizer)
                direction_remove_deltas.append(dcf_rm - dcf_base)

            dir_remove_mean = np.mean(direction_remove_deltas, axis=0)
            dir_remove_target = dir_remove_mean[target_idx]

            # 3-way消融 (8对象)
            for abl_type in ["proj_bc", "orth_bc", "full_mlp"]:
                mlp_remove_deltas = []
                for obj in cat_objs:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                    with torch.no_grad():
                        out_base = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits_base = out_base.logits[0, -1].float().cpu().numpy()
                    dcf_base = compute_dcf(logits_base, tokenizer)

                    added = [False]
                    def make_orth_hook(b_hat_np, position, ablation_mode):
                        def hook_fn(module, inp, output):
                            if not added[0]:
                                if isinstance(output, tuple):
                                    out = output[0].clone()
                                else:
                                    out = output.clone()
                                mlp_out_np = out[0, position, :].float().cpu().numpy()

                                if ablation_mode == "proj_bc":
                                    proj = np.dot(mlp_out_np, b_hat_np) * b_hat_np
                                    out[0, position, :] -= torch.tensor(proj, dtype=out.dtype, device=out.device)
                                elif ablation_mode == "orth_bc":
                                    proj = np.dot(mlp_out_np, b_hat_np) * b_hat_np
                                    orth = mlp_out_np - proj
                                    out[0, position, :] -= torch.tensor(orth, dtype=out.dtype, device=out.device)
                                elif ablation_mode == "full_mlp":
                                    out[0, position, :] = 0

                                added[0] = True
                                if isinstance(output, tuple):
                                    return (out,) + output[1:]
                                return out
                            return output
                        return hook_fn

                    h = layers_list[best_diff_layer].mlp.register_forward_hook(
                        make_orth_hook(b_hat, pos, abl_type)
                    )
                    with torch.no_grad():
                        out_abl = model(input_ids=input_ids, attention_mask=attention_mask)
                    h.remove()

                    logits_abl = out_abl.logits[0, -1].float().cpu().numpy()
                    dcf_abl = compute_dcf(logits_abl, tokenizer)
                    mlp_remove_deltas.append(dcf_abl - dcf_base)

                if mlp_remove_deltas:
                    abl_mean = np.mean(mlp_remove_deltas, axis=0)
                    abl_target = abl_mean[target_idx]
                    amplitude_ratio = abs(abl_target / dir_remove_target) if abs(dir_remove_target) > 0.01 else 0
                    cos_with_dir = float(np.dot(abl_mean, dir_remove_mean) /
                                        (np.linalg.norm(abl_mean) * np.linalg.norm(dir_remove_mean) + 1e-10))
                    confirm_results[f"{cat_name}_L{best_diff_layer}_{abl_type}"] = {
                        "target_delta": float(abl_target),
                        "amplitude_ratio": float(amplitude_ratio),
                        "cos_with_direction_remove": float(cos_with_dir),
                        "n_samples": len(cat_objs),
                    }
                    plog(f"    {cat_name} L{best_diff_layer} {abl_type}: "
                         f"target_D={abl_target:.2f}, amp={amplitude_ratio:.1%}, cos={cos_with_dir:.3f}")

        r2_results["exp1_confirmation"] = confirm_results

    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    # 保存R2结果
    result_path = f"results/glm5/phase487_{model_name}_r2.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(r2_results, f, indent=2, ensure_ascii=False, default=str)
    plog(f"R2 results saved to {result_path}")

    return r2_results


# ==================== 入口 ====================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase487_orthogonal_propagation.py <model_name> [round_num]")
        print("  model_name: qwen3, glm4, deepseek7b")
        print("  round_num: 1 (default) or 2")
        sys.exit(1)

    model_name = sys.argv[1]
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if round_num == 1:
        run_all_experiments(model_name, round_num=1)
    elif round_num == 2:
        run_r2_confirmation(model_name)
    else:
        print(f"Invalid round: {round_num}")
        sys.exit(1)

    # 进程退出确保GPU释放
    plog(f"Phase 487 {model_name} R{round_num} complete. Hard exit to release GPU.")
    import os
    os._exit(0)
