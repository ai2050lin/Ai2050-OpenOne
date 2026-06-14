"""
Phase 488: 边界前体传播算子与正交空间细分
============================================

Phase 487核心发现:
1. 正交成分是边界因果主要路径 (orth_bc >> proj_bc)
2. Qwen3 fruit L35: orth 165% vs proj 6%
3. GLM4 fruit三层架构: 早层正交 + 中层投影 + 晚层正交
4. 反对层是真实因果机制 (Qwen3 fruit ablate_opp→+2.69, double→-2.91)

Phase 488核心目标:
- Exp1: 扰动传播追踪 — 在早层注入正交方向扰动，追踪其如何传播到目标层
  直接证明: orth_B成分经后续层变换后对齐B_c
- Exp2: 正交空间细分 — 把orth_B分为竞争类别/共享语义/格式/残差等子空间
  找到真正转化为B_c的子成分
- Exp3: 边界轨迹图谱 — 每层测量proj_B/orth_B/attn贡献/DCF可读性
- Exp4: 前体注入测试 — 在早层注入候选前体方向，测量后续层是否自然生成B_c

用法:
  python tests/glm5/phase488_propagation_operator.py qwen3
  python tests/glm5/phase488_propagation_operator.py glm4
  python tests/glm5/phase488_propagation_operator.py deepseek7b
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

PRIORITY_CATS = {
    "qwen3": ["clothing", "fruit"],
    "glm4":  ["fruit", "clothing"],
    "deepseek7b": ["fruit", "food"],
}

# Phase 487发现的关键实验层
KEY_LAYERS = {
    "qwen3": {
        "clothing": [25, 30, 34],     # orth 9.4% at L34
        "fruit": [27, 32, 35],        # orth 140% at L35
    },
    "glm4": {
        "fruit": [22, 27, 32],        # 3-layer architecture
        "clothing": [34, 39],
    },
    "deepseek7b": {
        "fruit": [21, 26],            # orth 148.5% at L26
        "food": [26, 27],             # orth 867.8% at L27
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


# ==================== Exp1: 扰动传播追踪 ★★★★★ ====================
def exp1_perturbation_propagation(model, tokenizer, device, model_name, W_U):
    """
    核心实验: 追踪正交成分的传播路径

    方法:
    1. 在关键层l, 收集MLP输出
    2. 分解: MLP_out = proj_B(MLP_out) + orth_B(MLP_out)
    3. 在层l注入小扰动 ε * orth_B_direction 到残差流
    4. 追踪扰动在后续层的传播:
       - 在层l+5, l+10, ..., L测量残差流变化
       - 计算变化在B_c方向上的投影 → 看正交扰动是否逐步对齐B_c
    5. 同样测试 proj_B_direction 扰动
    6. 对比随机方向扰动 (控制组)

    成功标准:
    - orth_B扰动在后续层逐步对齐B_c (证明正交前体→边界转换)
    - proj_B扰动在后续层保持B_c对齐 (直接路径)
    - 随机扰动不对齐B_c (控制)
    """
    plog("=== Exp1: 扰动传播追踪 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers
    cat_names = PRIORITY_CATS.get(model_name, ["clothing", "fruit"])
    eps = 0.5  # 扰动强度

    results = {}

    for cat_name in cat_names:
        best_layer = BEST_LAYERS[model_name][cat_name]
        key_ls = KEY_LAYERS.get(model_name, {}).get(cat_name, [best_layer])
        plog(f"  {cat_name}: perturbation propagation tracing...")
        t0 = time.time()

        # 构造B_c方向(在best_layer)
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

        # 先收集各类别在best_layer的B_c方向(用于后续层追踪)
        # 需要在多个层收集B_c方向以追踪传播
        all_cat_residuals = {}  # {layer: {cat: vec}}

        propagation_results = {}

        for src_layer in key_ls:
            if src_layer >= n_layers:
                continue
            plog(f"    Source L{src_layer}: collecting MLP output...")

            # ---- 收集MLP输出和正交方向 ----
            mlp_outputs = []  # 每个对象的MLP输出
            for obj in cat_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                # 收集MLP输出
                cap_mlp = {}
                h_mlp = layers_list[src_layer].mlp.register_forward_hook(
                    lambda m, i, o: cap_mlp.__setitem__("v", o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu())
                )
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_mlp.remove()

                if "v" in cap_mlp:
                    mlp_outputs.append(cap_mlp["v"][0, pos].numpy())

            if not mlp_outputs:
                continue

            # 平均MLP输出
            mean_mlp = np.mean(mlp_outputs, axis=0)

            # 在src_layer层获取B_c方向
            spec_vec_l, spec_norm_l = get_specific_direction(
                model, tokenizer, device, model_name, cat_name, src_layer
            )
            if spec_vec_l is None or spec_norm_l < 1e-6:
                plog(f"      Skip L{src_layer}: cannot get B_c direction")
                continue

            b_hat_l = spec_vec_l / spec_norm_l

            # 分解MLP输出
            proj_component = np.dot(mean_mlp, b_hat_l) * b_hat_l
            orth_component = mean_mlp - proj_component

            orth_norm = np.linalg.norm(orth_component)
            proj_norm = np.linalg.norm(proj_component)

            if orth_norm < 1e-8:
                plog(f"      Skip L{src_layer}: orth_component too small")
                continue

            orth_hat = orth_component / orth_norm  # 正交方向(归一化)

            # 生成随机控制方向
            rng = np.random.RandomState(42)
            random_dir = rng.randn(info.d_model)
            random_dir = random_dir - np.dot(random_dir, b_hat_l) * b_hat_l  # 去除B_c投影
            random_dir_norm = np.linalg.norm(random_dir)
            if random_dir_norm > 1e-8:
                random_hat = random_dir / random_dir_norm
            else:
                random_hat = np.zeros_like(b_hat_l)

            # ---- 扰动传播追踪 ----
            # 在src_layer之后注入扰动, 追踪后续层的变化
            trace_layers = []
            for dl in [5, 10, 15, 20]:
                tl = src_layer + dl
                if tl < n_layers and tl != src_layer:
                    trace_layers.append(tl)
            # 确保包含best_layer
            if best_layer > src_layer and best_layer < n_layers:
                trace_layers.append(best_layer)
            # 确保包含最后一层
            if n_layers - 1 > src_layer:
                trace_layers.append(n_layers - 1)
            trace_layers = sorted(set(trace_layers))
            plog(f"      Trace layers: {trace_layers}")

            # 对每种扰动方向测试
            perturb_types = {
                "orth_bc": orth_hat,
                "proj_bc": b_hat_l,
                "random": random_hat,
            }

            layer_propagation = {}

            for perturb_name, perturb_dir in perturb_types.items():
                dir_norm = np.linalg.norm(perturb_dir)
                if dir_norm < 1e-8:
                    continue

                perturb_delta = {}  # {trace_layer: {B_c_alignment, total_norm, ...}}

                for obj in cat_objs[:2]:  # 只用2个对象节省时间
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                    # baseline: 收集各trace层的残差
                    baseline_resids = {}
                    def make_collect_hook(layer_idx, storage):
                        def hook_fn(module, inp, output):
                            if isinstance(output, tuple):
                                storage[layer_idx] = output[0].detach().float().cpu()
                            else:
                                storage[layer_idx] = output.detach().float().cpu()
                        return hook_fn

                    hooks = []
                    for tl in trace_layers:
                        hooks.append(layers_list[tl].register_forward_hook(
                            make_collect_hook(tl, baseline_resids)
                        ))

                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    for h in hooks:
                        h.remove()

                    # perturbed: 在src_layer注入扰动后收集各trace层残差
                    perturbed_resids = {}
                    hooks2 = []
                    added = [False]

                    def make_inject_hook(perturb_dir_np, position, inject_flag):
                        def hook_fn(module, inp, output):
                            if not inject_flag[0]:
                                if isinstance(output, tuple):
                                    out = output[0].clone()
                                else:
                                    out = output.clone()
                                # 注入扰动
                                perturb_tensor = torch.tensor(
                                    eps * perturb_dir_np,
                                    dtype=out.dtype, device=out.device
                                )
                                out[0, position, :] += perturb_tensor
                                inject_flag[0] = True
                                if isinstance(output, tuple):
                                    return (out,) + output[1:]
                                return out
                            return output
                        return hook_fn

                    # 注入hook在src_layer
                    hooks2.append(layers_list[src_layer].register_forward_hook(
                        make_inject_hook(perturb_dir, pos, added)
                    ))
                    # 收集hook在trace_layers
                    for tl in trace_layers:
                        hooks2.append(layers_list[tl].register_forward_hook(
                            make_collect_hook(tl, perturbed_resids)
                        ))

                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    for h in hooks2:
                        h.remove()

                    # 计算各trace层的残差变化
                    for tl in trace_layers:
                        if tl in baseline_resids and tl in perturbed_resids:
                            delta_h = (perturbed_resids[tl] - baseline_resids[tl])[0, pos].numpy()

                            # 获取该层的B_c方向
                            if tl not in all_cat_residuals:
                                # 延迟获取(只在需要时)
                                spec_vec_tl, spec_norm_tl = get_specific_direction(
                                    model, tokenizer, device, model_name, cat_name, tl
                                )
                                if spec_vec_tl is not None and spec_norm_tl > 1e-6:
                                    all_cat_residuals[tl] = spec_vec_tl / spec_norm_tl

                            b_hat_tl = all_cat_residuals.get(tl, b_hat)  # fallback到best_layer方向

                            # 变化在B_c方向上的投影
                            delta_bc_proj = np.dot(delta_h, b_hat_tl)
                            delta_total_norm = np.linalg.norm(delta_h)
                            bc_alignment = delta_bc_proj / delta_total_norm if delta_total_norm > 1e-8 else 0

                            if tl not in perturb_delta:
                                perturb_delta[tl] = {
                                    "bc_projections": [],
                                    "total_norms": [],
                                    "bc_alignments": [],
                                }
                            perturb_delta[tl]["bc_projections"].append(float(delta_bc_proj))
                            perturb_delta[tl]["total_norms"].append(float(delta_total_norm))
                            perturb_delta[tl]["bc_alignments"].append(float(bc_alignment))

                # 汇总
                layer_data = {}
                for tl in trace_layers:
                    if tl in perturb_delta:
                        pd = perturb_delta[tl]
                        layer_data[str(tl)] = {
                            "mean_bc_proj": float(np.mean(pd["bc_projections"])),
                            "mean_total_norm": float(np.mean(pd["total_norms"])),
                            "mean_bc_alignment": float(np.mean(pd["bc_alignments"])),
                            "n_samples": len(pd["bc_projections"]),
                        }
                        plog(f"        {perturb_name} → L{tl}: "
                             f"bc_proj={np.mean(pd['bc_projections']):.4f}, "
                             f"norm={np.mean(pd['total_norms']):.4f}, "
                             f"alignment={np.mean(pd['bc_alignments']):.4f}")

                layer_propagation[perturb_name] = layer_data

            propagation_results[f"L{src_layer}"] = {
                "source_layer": src_layer,
                "orth_component_norm": float(orth_norm),
                "proj_component_norm": float(proj_norm),
                "trace_layers": trace_layers,
                "perturbation_results": layer_propagation,
            }

        elapsed = time.time() - t0
        plog(f"    {cat_name} propagation tracing done ({elapsed:.0f}s)")

        results[cat_name] = {
            "best_layer": best_layer,
            "propagation_results": propagation_results,
            "elapsed": elapsed,
        }

    return results


# ==================== Exp2: 正交空间细分 ★★★★★ ====================
def exp2_orthogonal_subdivision(model, tokenizer, device, model_name, W_U):
    """
    正交空间细分: 把orth_B分解为多个子空间

    orth_B空间包含:
    1. 竞争类别方向 — 与其他类别的B_c方向对齐
    2. 共享语义方向 — 多个类别共享的方向(PCA提取)
    3. 格式方向 — 模板差异导致的方向
    4. 残差方向 — 其余方向

    方法:
    1. 在关键层收集各类别的MLP输出
    2. 计算B_c方向和orth_B
    3. 把orth_B投影到各个子空间
    4. 分别消融每个子空间成分，测量DCF变化
    """
    plog("=== Exp2: 正交空间细分 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers
    cat_names = PRIORITY_CATS.get(model_name, ["clothing", "fruit"])

    results = {}

    for cat_name in cat_names:
        best_layer = BEST_LAYERS[model_name][cat_name]
        key_ls = KEY_LAYERS.get(model_name, {}).get(cat_name, [best_layer])
        plog(f"  {cat_name}: orthogonal subdivision...")
        t0 = time.time()

        # 收集各类别在best_layer的B_c方向
        all_bc_dirs = {}  # {cat: b_hat}
        for c in CATEGORIES.keys():
            sv, sn = get_specific_direction(
                model, tokenizer, device, model_name, c, best_layer
            )
            if sv is not None and sn > 1e-6:
                all_bc_dirs[c] = sv / sn

        if cat_name not in all_bc_dirs:
            plog(f"    Skip {cat_name}: no B_c direction")
            continue

        b_hat = all_bc_dirs[cat_name]
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

        # ---- 构造子空间 ----
        # 1. 竞争类别子空间: 其他类别的B_c方向
        competitor_dirs = []
        for c, bd in all_bc_dirs.items():
            if c != cat_name:
                # 去除目标B_c投影
                bd_orth = bd - np.dot(bd, b_hat) * b_hat
                if np.linalg.norm(bd_orth) > 1e-8:
                    competitor_dirs.append(bd_orth / np.linalg.norm(bd_orth))

        # 2. 共享语义子空间: 所有类别残差的PCA
        all_residuals = []
        for c in CATEGORIES.keys():
            sv, sn = get_specific_direction(
                model, tokenizer, device, model_name, c, best_layer
            )
            if sv is not None and sn > 1e-6:
                all_residuals.append(sv / np.linalg.norm(sv))
        if len(all_residuals) >= 2:
            resid_matrix = np.array(all_residuals)
            try:
                U, S, Vt = np.linalg.svd(resid_matrix, full_matrices=False)
                # 前3个PCA方向(去除B_c投影)
                shared_dirs = []
                for i in range(min(3, len(Vt))):
                    v = Vt[i] - np.dot(Vt[i], b_hat) * b_hat
                    if np.linalg.norm(v) > 1e-8:
                        shared_dirs.append(v / np.linalg.norm(v))
            except Exception:
                shared_dirs = []
        else:
            shared_dirs = []

        # 3. 格式子空间: 同语义不同模板的差异方向
        format_templates = [
            "The {obj} is a kind of",
            "A {obj} is a type of",
            "{obj} belongs to the category of",
        ]
        format_diffs = []
        for obj in ["apple", "dog", "car"][:2]:
            obj_resids = []
            for tmpl in format_templates[:2]:
                prompt = tmpl.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                cap = {}
                h = layers_list[best_layer].register_forward_hook(
                    lambda m, i, o: cap.__setitem__("v", o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu())
                )
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                if "v" in cap:
                    obj_resids.append(cap["v"][0, pos].numpy())
            if len(obj_resids) >= 2:
                fd = obj_resids[0] - obj_resids[1]
                fd_orth = fd - np.dot(fd, b_hat) * b_hat
                if np.linalg.norm(fd_orth) > 1e-8:
                    format_diffs.append(fd_orth / np.linalg.norm(fd_orth))

        plog(f"    Subspaces: competitor={len(competitor_dirs)}, "
             f"shared={len(shared_dirs)}, format={len(format_diffs)}")

        # ---- 在关键层做子空间消融 ----
        subdivision_results = {}

        for src_layer in key_ls:
            if src_layer >= n_layers:
                continue
            plog(f"    L{src_layer}: sub-component ablation...")

            # 获取该层的B_c方向
            spec_vec_l, spec_norm_l = get_specific_direction(
                model, tokenizer, device, model_name, cat_name, src_layer
            )
            if spec_vec_l is None or spec_norm_l < 1e-6:
                continue
            b_hat_l = spec_vec_l / spec_norm_l

            # 定义消融子空间
            subspaces = {
                "proj_bc": [b_hat_l],  # 目标B_c投影(对照)
                "competitor_bc": competitor_dirs[:3],  # 竞争类别方向
                "shared_semantic": shared_dirs[:3],    # 共享语义方向
                "format": format_diffs[:2],            # 格式方向
            }

            # 也测试: "除竞争类别外的正交成分" = orth - competitor - shared - format
            # 即 residual orthogonal

            sub_ablation = {}

            for sub_name, sub_dirs in subspaces.items():
                if not sub_dirs:
                    continue

                ablation_deltas = []

                for obj in cat_objs:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                    with torch.no_grad():
                        out_base = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits_base = out_base.logits[0, -1].float().cpu().numpy()
                    dcf_base = compute_dcf(logits_base, tokenizer)

                    # 消融子空间
                    added = [False]
                    def make_subspace_hook(sub_dirs_list, b_hat_np, position, flag):
                        def hook_fn(module, inp, output):
                            if not flag[0]:
                                if isinstance(output, tuple):
                                    out = output[0].clone()
                                else:
                                    out = output.clone()
                                mlp_out_np = out[0, position, :].float().cpu().numpy()

                                # 投影到子空间并去除
                                removal = np.zeros_like(mlp_out_np)
                                for sd in sub_dirs_list:
                                    removal += np.dot(mlp_out_np, sd) * sd

                                out[0, position, :] -= torch.tensor(
                                    removal, dtype=out.dtype, device=out.device
                                )
                                flag[0] = True
                                if isinstance(output, tuple):
                                    return (out,) + output[1:]
                                return out
                            return output
                        return hook_fn

                    h = layers_list[src_layer].mlp.register_forward_hook(
                        make_subspace_hook(sub_dirs, b_hat_l, pos, added)
                    )
                    with torch.no_grad():
                        out_abl = model(input_ids=input_ids, attention_mask=attention_mask)
                    h.remove()

                    logits_abl = out_abl.logits[0, -1].float().cpu().numpy()
                    dcf_abl = compute_dcf(logits_abl, tokenizer)
                    ablation_deltas.append(dcf_abl - dcf_base)

                if ablation_deltas:
                    abl_mean = np.mean(ablation_deltas, axis=0)
                    abl_target = abl_mean[target_idx]
                    amp_ratio = abs(abl_target / dir_remove_target) if abs(dir_remove_target) > 0.01 else 0
                    cos_dir = float(np.dot(abl_mean, dir_remove_mean) /
                                    (np.linalg.norm(abl_mean) * np.linalg.norm(dir_remove_mean) + 1e-10))

                    sub_ablation[sub_name] = {
                        "target_delta": float(abl_target),
                        "amplitude_ratio": float(amp_ratio),
                        "cos_with_direction_remove": float(cos_dir),
                        "n_subspace_dirs": len(sub_dirs),
                        "dcf_delta": [float(x) for x in abl_mean],
                    }
                    plog(f"      {sub_name:20s}: target_D={abl_target:.2f}, "
                         f"amp={amp_ratio:.1%}, cos={cos_dir:.3f}")

            subdivision_results[f"L{src_layer}"] = sub_ablation

        elapsed = time.time() - t0
        plog(f"    {cat_name} subdivision done ({elapsed:.0f}s)")

        results[cat_name] = {
            "best_layer": best_layer,
            "direction_remove_target": float(dir_remove_target),
            "n_competitor_dirs": len(competitor_dirs),
            "n_shared_dirs": len(shared_dirs),
            "n_format_dirs": len(format_diffs),
            "subdivision_results": subdivision_results,
            "elapsed": elapsed,
        }

    return results


# ==================== Exp3: 边界轨迹图谱 ★★★★ ====================
def exp3_boundary_trajectory(model, tokenizer, device, model_name, W_U):
    """
    边界轨迹图谱: 每层测量proj_B/orth_B/attn/DCF

    对每个类别，在每层测量:
    1. MLP_out在B_c上的投影强度 (显化边界)
    2. MLP_out在B_c正交空间的强度 (前体成分)
    3. Attn_out在B_c上的投影 (注意力贡献)
    4. 残差流在B_c上的投影 (累积边界状态)
    5. DCF可读性 (类别区分度)

    输出: 每个类别的5条曲线
    """
    plog("=== Exp3: 边界轨迹图谱 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers
    cat_names = PRIORITY_CATS.get(model_name, ["clothing", "fruit"])

    # 采样层 (每2-3层测一次,避免太慢)
    sample_step = max(1, n_layers // 15)
    scan_layers = sorted(set(list(range(0, n_layers, sample_step)) + [n_layers - 1]))

    results = {}

    for cat_name in cat_names:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name}: boundary trajectory ({len(scan_layers)} layers)...")
        t0 = time.time()

        # 先在best_layer获取B_c方向
        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"    Skip {cat_name}")
            continue

        b_hat_ref = spec_vec / spec_norm

        template = RELATION_TEMPLATES["kind_of"]
        cat_objs = CATEGORIES[cat_name]

        trajectory = {}

        for scan_l in scan_layers:
            layer_data = {
                "mlp_proj_bc": 0, "mlp_orth_bc": 0,
                "attn_proj_bc": 0, "resid_proj_bc": 0,
                "dcf_readability": 0,
            }

            # 获取该层的B_c方向
            spec_vec_l, spec_norm_l = get_specific_direction(
                model, tokenizer, device, model_name, cat_name, scan_l
            )
            if spec_vec_l is not None and spec_norm_l > 1e-6:
                b_hat_l = spec_vec_l / spec_norm_l
            else:
                b_hat_l = b_hat_ref  # fallback

            for obj in cat_objs[:2]:  # 用2个对象加速
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                # 收集MLP/Attn/残差
                captured = {}
                def make_capture_hook(key):
                    def hook_fn(module, inp, output):
                        if isinstance(output, tuple):
                            captured[key] = output[0].detach().float().cpu()
                        else:
                            captured[key] = output.detach().float().cpu()
                    return hook_fn

                hooks = []
                hooks.append(layers_list[scan_l].register_forward_hook(
                    make_capture_hook("resid")
                ))
                hooks.append(layers_list[scan_l].mlp.register_forward_hook(
                    make_capture_hook("mlp")
                ))
                hooks.append(layers_list[scan_l].self_attn.register_forward_hook(
                    make_capture_hook("attn")
                ))

                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                for h in hooks:
                    h.remove()

                # MLP输出分解
                if "mlp" in captured:
                    mlp_np = captured["mlp"][0, pos].numpy()
                    proj = float(abs(np.dot(mlp_np, b_hat_l)))
                    orth = float(np.linalg.norm(mlp_np - np.dot(mlp_np, b_hat_l) * b_hat_l))
                    layer_data["mlp_proj_bc"] += proj
                    layer_data["mlp_orth_bc"] += orth

                # Attn输出在B_c上的投影
                if "attn" in captured:
                    attn_np = captured["attn"][0, pos].numpy()
                    layer_data["attn_proj_bc"] += float(abs(np.dot(attn_np, b_hat_l)))

                # 残差流在B_c上的投影
                if "resid" in captured:
                    resid_np = captured["resid"][0, pos].numpy()
                    layer_data["resid_proj_bc"] += float(abs(np.dot(resid_np, b_hat_l)))

                # DCF可读性
                logits = out.logits[0, -1].float().cpu().numpy()
                dcf = compute_dcf(logits, tokenizer)
                target_dcf = dcf[DCF_DIM_NAMES.index(cat_name)] if cat_name in DCF_DIM_NAMES else 0
                other_dcf = [dcf[i] for i in range(len(dcf)) if DCF_DIM_NAMES[i] != cat_name]
                readability = target_dcf - np.mean(other_dcf) if other_dcf else 0
                layer_data["dcf_readability"] += readability

            # 平均
            n_samples = min(2, len(cat_objs))
            for k in layer_data:
                layer_data[k] /= n_samples

            trajectory[str(scan_l)] = layer_data

        elapsed = time.time() - t0
        plog(f"    {cat_name} trajectory done ({elapsed:.0f}s): {len(trajectory)} layers")

        results[cat_name] = {
            "best_layer": best_layer,
            "scan_layers": scan_layers,
            "trajectory": trajectory,
            "elapsed": elapsed,
        }

    return results


# ==================== Exp4: 前体注入测试 ★★★★ ====================
def exp4_precursor_injection(model, tokenizer, device, model_name, W_U):
    """
    前体注入测试: 在早层注入候选前体方向，测量后续层是否自然生成B_c

    方法:
    1. 在关键层l, 获取MLP输出的正交方向 orth_hat
    2. 在早层(l-5或l-10)注入 orth_hat 方向
    3. 测量best_layer处残差流是否增加B_c投影
    4. 对比: 在同一早层注入B_c方向(直接注入)
    5. 对比: 在同一早层注入随机方向(控制)

    成功标准:
    - 注入orth_hat后, best_layer的B_c投影增加
    - 说明orth_hat是真实的B_c前体
    """
    plog("=== Exp4: 前体注入测试 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers
    cat_names = PRIORITY_CATS.get(model_name, ["clothing", "fruit"])
    inject_scales = [0.5, 1.0, 2.0]  # 注入强度

    results = {}

    for cat_name in cat_names:
        best_layer = BEST_LAYERS[model_name][cat_name]
        key_ls = KEY_LAYERS.get(model_name, {}).get(cat_name, [best_layer])
        plog(f"  {cat_name}: precursor injection test...")
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

        injection_results = {}

        for src_layer in key_ls:
            if src_layer >= n_layers:
                continue

            # 注入层: src_layer之前的层
            inject_layer = max(0, src_layer - 5)
            if inject_layer == src_layer:
                inject_layer = max(0, src_layer - 1)
            plog(f"    L{src_layer} orth → inject at L{inject_layer}")

            # 获取src_layer的B_c方向和orth方向
            spec_vec_l, spec_norm_l = get_specific_direction(
                model, tokenizer, device, model_name, cat_name, src_layer
            )
            if spec_vec_l is None or spec_norm_l < 1e-6:
                continue
            b_hat_l = spec_vec_l / spec_norm_l

            # 收集MLP输出获取orth方向
            mlp_outputs = []
            for obj in cat_objs[:2]:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                cap = {}
                h = layers_list[src_layer].mlp.register_forward_hook(
                    lambda m, i, o: cap.__setitem__("v", o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu())
                )
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                if "v" in cap:
                    mlp_outputs.append(cap["v"][0, pos].numpy())

            if not mlp_outputs:
                continue

            mean_mlp = np.mean(mlp_outputs, axis=0)
            proj_comp = np.dot(mean_mlp, b_hat_l) * b_hat_l
            orth_comp = mean_mlp - proj_comp
            orth_norm = np.linalg.norm(orth_comp)

            if orth_norm < 1e-8:
                continue

            orth_hat = orth_comp / orth_norm

            # 随机控制方向
            rng = np.random.RandomState(123)
            random_dir = rng.randn(info.d_model)
            random_dir = random_dir - np.dot(random_dir, b_hat_l) * b_hat_l
            if np.linalg.norm(random_dir) > 1e-8:
                random_hat = random_dir / np.linalg.norm(random_dir)
            else:
                random_hat = np.zeros_like(b_hat_l)

            # 注入方向列表
            inject_dirs = {
                "orth_bc": orth_hat,
                "proj_bc": b_hat_l,
                "random": random_hat,
            }

            layer_inject = {}

            for inject_name, inject_dir in inject_dirs.items():
                if np.linalg.norm(inject_dir) < 1e-8:
                    continue

                for scale in inject_scales:
                    bc_increases = []

                    for obj in cat_objs[:2]:
                        prompt = template.format(obj=obj)
                        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                        # baseline: best_layer的B_c投影
                        cap_base = {}
                        h_base = layers_list[best_layer].register_forward_hook(
                            lambda m, i, o: cap_base.__setitem__("v", o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu())
                        )
                        with torch.no_grad():
                            model(input_ids=input_ids, attention_mask=attention_mask)
                        h_base.remove()

                        if "v" not in cap_base:
                            continue
                        base_resid = cap_base["v"][0, pos].numpy()
                        base_bc_proj = float(np.dot(base_resid, b_hat))

                        # 注入: 在inject_layer注入方向
                        cap_inj = {}
                        added = [False]

                        def make_inject_hook2(direction, position, strength, flag):
                            def hook_fn(module, inp, output):
                                if not flag[0]:
                                    if isinstance(output, tuple):
                                        out = output[0].clone()
                                    else:
                                        out = output.clone()
                                    inject_vec = torch.tensor(
                                        strength * direction,
                                        dtype=out.dtype, device=out.device
                                    )
                                    out[0, position, :] += inject_vec
                                    flag[0] = True
                                    if isinstance(output, tuple):
                                        return (out,) + output[1:]
                                    return out
                                return output
                            return hook_fn

                        hooks = []
                        hooks.append(layers_list[inject_layer].register_forward_hook(
                            make_inject_hook2(inject_dir, pos, scale, added)
                        ))
                        hooks.append(layers_list[best_layer].register_forward_hook(
                            lambda m, i, o: cap_inj.__setitem__("v", o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu())
                        ))

                        with torch.no_grad():
                            model(input_ids=input_ids, attention_mask=attention_mask)
                        for h in hooks:
                            h.remove()

                        if "v" not in cap_inj:
                            continue
                        inj_resid = cap_inj["v"][0, pos].numpy()
                        inj_bc_proj = float(np.dot(inj_resid, b_hat))

                        bc_increases.append(inj_bc_proj - base_bc_proj)

                    if bc_increases:
                        key = f"{inject_name}_s{scale}"
                        mean_increase = float(np.mean(bc_increases))
                        layer_inject[key] = {
                            "mean_bc_increase": mean_increase,
                            "inject_layer": inject_layer,
                            "target_layer": best_layer,
                            "scale": scale,
                            "n_samples": len(bc_increases),
                        }
                        plog(f"        {key}: bc_increase={mean_increase:.4f}")

            injection_results[f"L{src_layer}"] = layer_inject

        elapsed = time.time() - t0
        plog(f"    {cat_name} injection done ({elapsed:.0f}s)")

        results[cat_name] = {
            "best_layer": best_layer,
            "injection_results": injection_results,
            "elapsed": elapsed,
        }

    return results


# ==================== 主流程 ====================
def run_all_experiments(model_name, round_num=1):
    plog(f"Phase 488: {model_name} (R{round_num})")
    plog(f"GPU: {torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'N/A'}")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model info: class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    W_U = get_W_U(model, model_name)
    plog(f"W_U: shape={W_U.shape}")

    all_results = {
        "phase": 488,
        "round": round_num,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }

    # Exp1: 扰动传播追踪
    try:
        r1 = exp1_perturbation_propagation(model, tokenizer, device, model_name, W_U)
        all_results["exp1_perturbation_propagation"] = r1
        plog(f"Exp1 done: {list(r1.keys())}")
    except Exception as e:
        plog(f"Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_perturbation_propagation"] = {"error": str(e)}

    save_partial(model_name, round_num, all_results, "exp1")

    # Exp2: 正交空间细分
    try:
        r2 = exp2_orthogonal_subdivision(model, tokenizer, device, model_name, W_U)
        all_results["exp2_orthogonal_subdivision"] = r2
        plog(f"Exp2 done: {list(r2.keys())}")
    except Exception as e:
        plog(f"Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_orthogonal_subdivision"] = {"error": str(e)}

    save_partial(model_name, round_num, all_results, "exp2")

    # Exp3: 边界轨迹图谱
    try:
        r3 = exp3_boundary_trajectory(model, tokenizer, device, model_name, W_U)
        all_results["exp3_boundary_trajectory"] = r3
        plog(f"Exp3 done: {list(r3.keys())}")
    except Exception as e:
        plog(f"Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_boundary_trajectory"] = {"error": str(e)}

    save_partial(model_name, round_num, all_results, "exp3")

    # Exp4: 前体注入测试
    try:
        r4 = exp4_precursor_injection(model, tokenizer, device, model_name, W_U)
        all_results["exp4_precursor_injection"] = r4
        plog(f"Exp4 done: {list(r4.keys())}")
    except Exception as e:
        plog(f"Exp4 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp4_precursor_injection"] = {"error": str(e)}

    save_partial(model_name, round_num, all_results, "exp4")

    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    # 保存完整结果
    result_path = f"results/glm5/phase488_{model_name}_r{round_num}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    plog(f"Results saved to {result_path}")

    return all_results


def save_partial(model_name, round_num, results, exp_name):
    """保存部分结果防止中断"""
    path = f"results/glm5/phase488_{model_name}_r{round_num}_partial.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        plog(f"  Partial save failed: {e}")


# ==================== R2: 确认测试 ====================
def run_r2_confirmation(model_name):
    """
    R2确认: 基于R1结果，用更多对象确认关键发现
    """
    plog(f"Phase 488 R2: {model_name}")

    r1_path = f"results/glm5/phase488_{model_name}_r1.json"
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
        "phase": 488, "round": 2, "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # R2: 用8对象确认Exp1的关键传播结果
    exp1 = r1_data.get("exp1_perturbation_propagation", {})
    if "error" not in exp1:
        cat_names = PRIORITY_CATS.get(model_name, ["clothing", "fruit"])
        confirm_results = {}

        for cat_name in cat_names[:2]:
            best_layer = BEST_LAYERS[model_name][cat_name]
            cat_r1 = exp1.get(cat_name, {})
            prop_results = cat_r1.get("propagation_results", {})

            # 找R1中最强的正交传播效应
            best_src = None
            best_orth_alignment = 0
            for src_key, src_data in prop_results.items():
                perturb_results = src_data.get("perturbation_results", {})
                orth_data = perturb_results.get("orth_bc", {})
                for tl_key, tl_data in orth_data.items():
                    alignment = abs(tl_data.get("mean_bc_alignment", 0))
                    if alignment > best_orth_alignment:
                        best_orth_alignment = alignment
                        best_src = int(src_key.replace("L", ""))

            if best_src is None:
                best_src = best_layer

            plog(f"  R2 confirm {cat_name}: best_src=L{best_src}, R1 alignment={best_orth_alignment:.4f}")

            spec_vec, spec_norm = get_specific_direction(
                model, tokenizer, device, model_name, cat_name, best_layer
            )
            if spec_vec is None or spec_norm < 1e-6:
                continue

            b_hat = spec_vec / spec_norm
            template = RELATION_TEMPLATES["kind_of"]
            cat_objs = CATEGORIES_R2.get(cat_name, CATEGORIES[cat_name])

            # 获取src层的MLP正交方向
            spec_vec_l, spec_norm_l = get_specific_direction(
                model, tokenizer, device, model_name, cat_name, best_src
            )
            if spec_vec_l is None or spec_norm_l < 1e-6:
                continue
            b_hat_l = spec_vec_l / spec_norm_l

            mlp_outputs = []
            for obj in cat_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                cap = {}
                h = layers_list[best_src].mlp.register_forward_hook(
                    lambda m, i, o: cap.__setitem__("v", o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu())
                )
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                if "v" in cap:
                    mlp_outputs.append(cap["v"][0, pos].numpy())

            if not mlp_outputs:
                continue

            mean_mlp = np.mean(mlp_outputs, axis=0)
            proj_comp = np.dot(mean_mlp, b_hat_l) * b_hat_l
            orth_comp = mean_mlp - proj_comp
            orth_norm = np.linalg.norm(orth_comp)
            if orth_norm < 1e-8:
                continue
            orth_hat = orth_comp / orth_norm

            # 用8对象测试正交传播
            eps = 0.5
            trace_layers = [best_layer, info.n_layers - 1]
            if best_layer != info.n_layers - 1:
                trace_layers = sorted(set(trace_layers + [min(info.n_layers - 1, best_src + 10)]))

            for obj in cat_objs[:4]:  # R2用4个对象(比R1的2个多)
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                # baseline
                baseline_resids = {}
                def make_collect_hook(layer_idx, storage):
                    def hook_fn(module, inp, output):
                        if isinstance(output, tuple):
                            storage[layer_idx] = output[0].detach().float().cpu()
                        else:
                            storage[layer_idx] = output.detach().float().cpu()
                    return hook_fn

                hooks = [layers_list[tl].register_forward_hook(make_collect_hook(tl, baseline_resids))
                         for tl in trace_layers]
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                for h in hooks:
                    h.remove()

                # perturbed
                perturbed_resids = {}
                added = [False]
                hooks2 = []

                def make_inject_hook(perturb_dir_np, position, flag):
                    def hook_fn(module, inp, output):
                        if not flag[0]:
                            if isinstance(output, tuple):
                                out = output[0].clone()
                            else:
                                out = output.clone()
                            out[0, position, :] += torch.tensor(
                                eps * perturb_dir_np, dtype=out.dtype, device=out.device
                            )
                            flag[0] = True
                            if isinstance(output, tuple):
                                return (out,) + output[1:]
                            return out
                        return output
                    return hook_fn

                hooks2.append(layers_list[best_src].register_forward_hook(
                    make_inject_hook(orth_hat, pos, added)
                ))
                for tl in trace_layers:
                    hooks2.append(layers_list[tl].register_forward_hook(
                        make_collect_hook(tl, perturbed_resids)
                    ))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                for h in hooks2:
                    h.remove()

                # 计算传播
                for tl in trace_layers:
                    if tl in baseline_resids and tl in perturbed_resids:
                        delta_h = (perturbed_resids[tl] - baseline_resids[tl])[0, pos].numpy()
                        bc_proj = float(np.dot(delta_h, b_hat))
                        total_norm = float(np.linalg.norm(delta_h))
                        alignment = bc_proj / total_norm if total_norm > 1e-8 else 0

                        key = f"{cat_name}_L{best_src}_to_L{tl}"
                        if key not in confirm_results:
                            confirm_results[key] = {"bc_projections": [], "alignments": []}
                        confirm_results[key]["bc_projections"].append(bc_proj)
                        confirm_results[key]["alignments"].append(alignment)

            # 汇总
            for key, data in confirm_results.items():
                data["mean_bc_proj"] = float(np.mean(data["bc_projections"]))
                data["mean_alignment"] = float(np.mean(data["alignments"]))
                plog(f"    R2 {key}: bc_proj={data['mean_bc_proj']:.4f}, "
                     f"alignment={data['mean_alignment']:.4f} (n={len(data['bc_projections'])})")

        r2_results["exp1_confirmation"] = confirm_results

    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    # 保存R2结果
    result_path = f"results/glm5/phase488_{model_name}_r2.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(r2_results, f, indent=2, ensure_ascii=False, default=str)
    plog(f"R2 results saved to {result_path}")

    return r2_results


# ==================== 入口 ====================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase488_propagation_operator.py <model_name> [round_num]")
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
    plog(f"Phase 488 {model_name} R{round_num} complete. Hard exit to release GPU.")
    import os
    os._exit(0)
