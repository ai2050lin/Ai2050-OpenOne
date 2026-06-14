"""
Phase 489: 共享语义抑制机制、末层边界读出前体与竞争释放连接
================================================================

Phase 488核心发现:
1. 中间层orth_bc主要是共享语义抑制项,不是边界前体
2. orth_bc传播后alignment大多为负(反B_c)
3. 中间层orth_bc注入削弱边界(bc_increase为负)
4. 只有最后1-2层orth_bc可能是读出前体
5. shared_semantic是orth_bc最大子成分

Phase 489核心目标:
- Exp1: shared_semantic因果测试 — ablate/inject/double/reverse
- Exp2: 末层orth_bc前体验证 — 注入/消融最后2层orth_bc
- Exp3: 投影写入vs共享抑制剂量曲线 — 2D网格验证drive-brake模型
- Exp4: 共享语义抑制与竞争释放矩阵连接
- Exp5: 跨模型一致性验证

用法:
  python tests/glm5/phase489_shared_semantic_brake.py qwen3
  python tests/glm5/phase489_shared_semantic_brake.py glm4
  python tests/glm5/phase489_shared_semantic_brake.py deepseek7b
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
    "made_of":   "The {obj} is made of",
    "grown_from": "The {obj} is grown from",
    "eaten_as":  "The {obj} is eaten as",
}

PRIORITY_CATS = {
    "qwen3": ["clothing", "fruit"],
    "glm4":  ["fruit", "clothing"],
    "deepseek7b": ["fruit", "food"],
}

# Phase 488发现的关键层位
KEY_LAYERS = {
    "qwen3": {
        "clothing": [25, 30, 34],
        "fruit": [27, 32, 35],
    },
    "glm4": {
        "fruit": [22, 27, 32],
        "clothing": [34, 39],
    },
    "deepseek7b": {
        "fruit": [21, 26],
        "food": [26, 27],
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


# ==================== 子空间构造 ====================
def build_subspaces(model, tokenizer, device, model_name, cat_name, target_layer):
    """构造各类子空间方向"""
    # 收集各类别在target_layer的B_c方向
    all_bc_dirs = {}
    for c in CATEGORIES.keys():
        sv, sn = get_specific_direction(
            model, tokenizer, device, model_name, c, target_layer
        )
        if sv is not None and sn > 1e-6:
            all_bc_dirs[c] = sv / sn

    if cat_name not in all_bc_dirs:
        return None

    b_hat = all_bc_dirs[cat_name]

    # 1. 竞争类别子空间
    competitor_dirs = []
    for c, bd in all_bc_dirs.items():
        if c != cat_name:
            bd_orth = bd - np.dot(bd, b_hat) * b_hat
            if np.linalg.norm(bd_orth) > 1e-8:
                competitor_dirs.append(bd_orth / np.linalg.norm(bd_orth))

    # 2. 共享语义子空间: 所有类别残差的SVD
    all_residuals = []
    for c in CATEGORIES.keys():
        sv, sn = get_specific_direction(
            model, tokenizer, device, model_name, c, target_layer
        )
        if sv is not None and sn > 1e-6:
            all_residuals.append(sv / np.linalg.norm(sv))

    shared_dirs = []
    if len(all_residuals) >= 2:
        resid_matrix = np.array(all_residuals)
        try:
            U, S, Vt = np.linalg.svd(resid_matrix, full_matrices=False)
            for i in range(min(5, len(Vt))):
                v = Vt[i] - np.dot(Vt[i], b_hat) * b_hat
                if np.linalg.norm(v) > 1e-8:
                    shared_dirs.append(v / np.linalg.norm(v))
        except Exception:
            pass

    # 3. 格式子空间
    format_templates = [
        "The {obj} is a kind of",
        "A {obj} is a type of",
    ]
    format_diffs = []
    for obj in ["apple", "dog", "car"][:2]:
        obj_resids = []
        for tmpl in format_templates:
            prompt = tmpl.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap = {}
            h = get_layers(model)[target_layer].register_forward_hook(
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

    return {
        "b_hat": b_hat,
        "competitor_dirs": competitor_dirs[:5],
        "shared_dirs": shared_dirs[:5],
        "format_dirs": format_diffs[:2],
        "all_bc_dirs": all_bc_dirs,
    }


# ==================== Exp1: shared_semantic因果测试 ★★★★★ ====================
def exp1_shared_semantic_causal(model, tokenizer, device, model_name, W_U):
    """
    对shared_semantic做4种操作, 验证它是否是边界抑制项:
    - ablate: 消融shared_semantic → 边界应增强
    - inject: 注入shared_semantic → 边界应削弱
    - double: 加倍shared_semantic → 边界应更弱
    - reverse: 反向注入shared_semantic → 边界应增强(同消融)
    """
    plog("=== Exp1: shared_semantic因果测试 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers
    cat_names = PRIORITY_CATS.get(model_name, ["clothing", "fruit"])

    results = {}

    for cat_name in cat_names:
        best_layer = BEST_LAYERS[model_name][cat_name]
        key_ls = KEY_LAYERS.get(model_name, {}).get(cat_name, [best_layer])
        # 过滤只保留中间层(不是末层)
        mid_layers = [l for l in key_ls if l < n_layers - 2]
        if not mid_layers:
            mid_layers = [key_ls[0]] if key_ls else [best_layer // 2]

        plog(f"  {cat_name}: shared_semantic causal test (mid layers: {mid_layers})...")
        t0 = time.time()

        # 构造子空间
        subspaces = build_subspaces(model, tokenizer, device, model_name, cat_name, best_layer)
        if subspaces is None:
            plog(f"    Skip {cat_name}: cannot build subspaces")
            continue

        b_hat = subspaces["b_hat"]
        shared_dirs = subspaces["shared_dirs"]
        competitor_dirs = subspaces["competitor_dirs"]

        if not shared_dirs:
            plog(f"    Skip {cat_name}: no shared semantic dirs")
            continue

        target_idx = DCF_DIM_NAMES.index(cat_name) if cat_name in DCF_DIM_NAMES else 0
        template = RELATION_TEMPLATES["kind_of"]
        cat_objs = CATEGORIES[cat_name]

        layer_results = {}

        for src_layer in mid_layers[:2]:  # 最多测2个中间层
            plog(f"    L{src_layer}: 4 operations...")

            # 先收集baseline DCF
            baseline_dcfs = []
            for obj in cat_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0, -1].float().cpu().numpy()
                baseline_dcfs.append(compute_dcf(logits, tokenizer))

            baseline_mean = np.mean(baseline_dcfs, axis=0)
            baseline_target = baseline_mean[target_idx]

            # 定义4种操作
            operations = {
                "ablate_shared": ("remove", shared_dirs, 1.0),
                "inject_shared": ("add", shared_dirs, 0.5),
                "double_shared": ("add", shared_dirs, 1.0),
                "reverse_shared": ("add", [-d for d in shared_dirs], 0.5),
            }

            # 也对proj_bc和competitor做对照
            operations["ablate_proj"] = ("remove", [b_hat], 1.0)
            operations["ablate_competitor"] = ("remove", competitor_dirs[:3], 1.0)

            op_results = {}

            for op_name, (op_type, dirs, scale) in operations.items():
                if not dirs:
                    continue

                op_dcfs = []
                for obj in cat_objs:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                    applied = [False]
                    def make_op_hook(dirs_list, b_hat_np, position, op_t, scl, flag):
                        def hook_fn(module, inp, output):
                            if not flag[0]:
                                if isinstance(output, tuple):
                                    out = output[0].clone()
                                else:
                                    out = output.clone()
                                mlp_out_np = out[0, position, :].float().cpu().numpy()

                                # 计算子空间投影
                                removal = np.zeros_like(mlp_out_np)
                                for sd in dirs_list:
                                    removal += np.dot(mlp_out_np, sd) * sd

                                if op_t == "remove":
                                    out[0, position, :] -= torch.tensor(
                                        removal, dtype=out.dtype, device=out.device
                                    )
                                elif op_t == "add":
                                    # 注入固定方向(归一化后乘以scale)
                                    inject_vec = np.zeros_like(mlp_out_np)
                                    for sd in dirs_list:
                                        inject_vec += sd  # 每个方向贡献1.0
                                    inject_norm = np.linalg.norm(inject_vec)
                                    if inject_norm > 1e-8:
                                        inject_vec = scl * inject_vec / inject_norm * np.linalg.norm(mlp_out_np) * 0.1
                                    out[0, position, :] += torch.tensor(
                                        inject_vec, dtype=out.dtype, device=out.device
                                    )

                                flag[0] = True
                                if isinstance(output, tuple):
                                    return (out,) + output[1:]
                                return out
                            return output
                        return hook_fn

                    h = layers_list[src_layer].mlp.register_forward_hook(
                        make_op_hook(dirs, b_hat, pos, op_type, scale, applied)
                    )
                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attention_mask)
                    h.remove()

                    logits = out.logits[0, -1].float().cpu().numpy()
                    op_dcfs.append(compute_dcf(logits, tokenizer))

                op_mean = np.mean(op_dcfs, axis=0)
                delta = op_mean - baseline_mean
                target_delta = delta[target_idx]

                op_results[op_name] = {
                    "target_delta": float(target_delta),
                    "target_baseline": float(baseline_target),
                    "target_after_op": float(op_mean[target_idx]),
                    "dcf_delta": [float(x) for x in delta],
                    "n_dirs": len(dirs),
                }
                plog(f"      {op_name:25s}: target_D={target_delta:+.3f} "
                     f"(baseline={baseline_target:.3f} → {op_mean[target_idx]:.3f})")

            layer_results[f"L{src_layer}"] = op_results

        elapsed = time.time() - t0
        plog(f"    {cat_name} done ({elapsed:.0f}s)")

        results[cat_name] = {
            "best_layer": best_layer,
            "n_shared_dirs": len(shared_dirs),
            "n_competitor_dirs": len(competitor_dirs),
            "layer_results": layer_results,
            "elapsed": elapsed,
        }

    return results


# ==================== Exp2: 末层orth_bc前体验证 ★★★★★ ====================
def exp2_late_orth_precursor(model, tokenizer, device, model_name, W_U):
    """
    只测试最后2层orth_bc:
    - 注入到best_layer, 观察B_c是否增强
    - 消融最后2层orth_bc, 观察B_c是否削弱
    - 对比中间层orth_bc(应削弱B_c)
    """
    plog("=== Exp2: 末层orth_bc前体验证 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers
    cat_names = PRIORITY_CATS.get(model_name, ["clothing", "fruit"])

    results = {}

    for cat_name in cat_names:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name}: late orth_bc precursor test...")
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

        # 收集末层(n_layers-1, n_layers-2)和中间层的MLP正交方向
        test_layers = []
        # 末层
        if n_layers - 1 != best_layer:
            test_layers.append(("late", n_layers - 1))
        if n_layers - 2 != best_layer and n_layers - 2 > best_layer:
            test_layers.append(("late-1", n_layers - 2))
        # 中间层(对照)
        mid_l = best_layer // 2
        if mid_l != best_layer:
            test_layers.append(("mid", mid_l))

        layer_results = {}

        for layer_label, src_layer in test_layers:
            if src_layer >= n_layers:
                continue

            plog(f"    L{src_layer} ({layer_label}): collecting orth_bc...")

            # 获取该层的B_c方向
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

            # 测试1: 注入orth_bc到best_layer前5层, 观察B_c变化
            inject_layer = max(0, best_layer - 5)
            if inject_layer == src_layer:
                inject_layer = max(0, src_layer - 1)

            # 测试2: 消融该层orth_bc, 观察DCF变化
            inject_scales = [0.5, 1.0]

            tests = {}

            # 注入测试
            for scale in inject_scales:
                bc_increases = []
                for obj in cat_objs[:2]:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                    # baseline
                    cap_base = {}
                    h_base = layers_list[best_layer].register_forward_hook(
                        lambda m, i, o: cap_base.__setitem__("v", o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu())
                    )
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    h_base.remove()

                    if "v" not in cap_base:
                        continue
                    base_bc_proj = float(np.dot(cap_base["v"][0, pos].numpy(), b_hat))

                    # 注入
                    cap_inj = {}
                    added = [False]

                    def make_inject_hook(direction, position, strength, flag):
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
                        make_inject_hook(orth_hat, pos, scale, added)
                    ))
                    hooks.append(layers_list[best_layer].register_forward_hook(
                        lambda m, i, o: cap_inj.__setitem__("v", o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu())
                    ))

                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    for h in hooks:
                        h.remove()

                    if "v" in cap_inj:
                        inj_bc_proj = float(np.dot(cap_inj["v"][0, pos].numpy(), b_hat))
                        bc_increases.append(inj_bc_proj - base_bc_proj)

                key = f"inject_orth_s{scale}"
                if bc_increases:
                    tests[key] = {
                        "mean_bc_increase": float(np.mean(bc_increases)),
                        "inject_layer": inject_layer,
                        "target_layer": best_layer,
                        "scale": scale,
                        "is_late_layer": layer_label.startswith("late"),
                    }
                    plog(f"      {key}: bc_increase={np.mean(bc_increases):+.4f} ({layer_label})")

            # 消融测试
            abl_dcfs = []
            base_dcfs = []
            for obj in cat_objs[:2]:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                # baseline
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                base_dcfs.append(compute_dcf(out.logits[0, -1].float().cpu().numpy(), tokenizer))

                # 消融orth_bc
                applied = [False]
                def make_abl_hook(b_hat_np, position, flag):
                    def hook_fn(module, inp, output):
                        if not flag[0]:
                            if isinstance(output, tuple):
                                out = output[0].clone()
                            else:
                                out = output.clone()
                            mlp_np = out[0, position, :].float().cpu().numpy()
                            proj = np.dot(mlp_np, b_hat_np) * b_hat_np
                            orth = mlp_np - proj
                            out[0, position, :] -= torch.tensor(
                                orth, dtype=out.dtype, device=out.device
                            )
                            flag[0] = True
                            if isinstance(output, tuple):
                                return (out,) + output[1:]
                            return out
                        return output
                    return hook_fn

                h = layers_list[src_layer].mlp.register_forward_hook(
                    make_abl_hook(b_hat_l, pos, applied)
                )
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                abl_dcfs.append(compute_dcf(out.logits[0, -1].float().cpu().numpy(), tokenizer))

            base_mean = np.mean(base_dcfs, axis=0)
            abl_mean = np.mean(abl_dcfs, axis=0)
            abl_delta = abl_mean - base_mean

            tests["ablate_orth_bc"] = {
                "target_delta": float(abl_delta[target_idx]),
                "dcf_delta": [float(x) for x in abl_delta],
                "is_late_layer": layer_label.startswith("late"),
            }
            plog(f"      ablate_orth_bc: target_D={abl_delta[target_idx]:+.3f} ({layer_label})")

            layer_results[f"L{src_layer}_{layer_label}"] = tests

        elapsed = time.time() - t0
        plog(f"    {cat_name} done ({elapsed:.0f}s)")

        results[cat_name] = {
            "best_layer": best_layer,
            "layer_results": layer_results,
            "elapsed": elapsed,
        }

    return results


# ==================== Exp3: 投影写入vs共享抑制剂量曲线 ★★★★ ====================
def exp3_dose_curve(model, tokenizer, device, model_name, W_U):
    """
    2D网格: 同时调节proj_bc和shared_semantic的scale
    验证: 边界强度 ≈ 投影写入 - 共享抑制
    """
    plog("=== Exp3: 投影写入vs共享抑制剂量曲线 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers
    cat_names = PRIORITY_CATS.get(model_name, ["clothing", "fruit"])

    # 简化: 只测1个类别, 1个中间层
    cat_name = cat_names[0]
    best_layer = BEST_LAYERS[model_name][cat_name]
    mid_layer = best_layer // 2

    plog(f"  {cat_name} L{mid_layer}: 2D dose curve...")
    t0 = time.time()

    # 构造子空间
    subspaces = build_subspaces(model, tokenizer, device, model_name, cat_name, best_layer)
    if subspaces is None:
        plog(f"    Skip: cannot build subspaces")
        return {"error": "no subspaces"}

    b_hat = subspaces["b_hat"]
    shared_dirs = subspaces["shared_dirs"]

    if not shared_dirs:
        plog(f"    Skip: no shared dirs")
        return {"error": "no shared dirs"}

    target_idx = DCF_DIM_NAMES.index(cat_name) if cat_name in DCF_DIM_NAMES else 0
    template = RELATION_TEMPLATES["kind_of"]
    cat_objs = CATEGORIES[cat_name][:2]  # 只用2个对象节省时间

    # 获取该层的B_c方向
    spec_vec_l, spec_norm_l = get_specific_direction(
        model, tokenizer, device, model_name, cat_name, mid_layer
    )
    if spec_vec_l is None or spec_norm_l < 1e-6:
        return {"error": "no B_c at mid_layer"}
    b_hat_l = spec_vec_l / spec_norm_l

    # 剂量网格
    proj_scales = [-1.0, -0.5, 0, 0.5, 1.0]
    shared_scales = [-1.0, -0.5, 0, 0.5, 1.0]

    # baseline DCF
    baseline_dcfs = []
    for obj in cat_objs:
        prompt = template.format(obj=obj)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        baseline_dcfs.append(compute_dcf(out.logits[0, -1].float().cpu().numpy(), tokenizer))
    baseline_mean = np.mean(baseline_dcfs, axis=0)
    baseline_target = baseline_mean[target_idx]

    dose_grid = {}

    for ps in proj_scales:
        for ss in shared_scales:
            grid_dcfs = []

            for obj in cat_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                applied = [False]
                def make_dose_hook(b_hat_np, shared_dirs_list, position, proj_s, shared_s, flag):
                    def hook_fn(module, inp, output):
                        if not flag[0]:
                            if isinstance(output, tuple):
                                out = output[0].clone()
                            else:
                                out = output.clone()
                            mlp_np = out[0, position, :].float().cpu().numpy()

                            # proj_bc调节
                            proj = np.dot(mlp_np, b_hat_np) * b_hat_np
                            proj_adjust = proj_s * proj  # 正=增加, 负=减少

                            # shared_semantic调节
                            shared_proj = np.zeros_like(mlp_np)
                            for sd in shared_dirs_list:
                                shared_proj += np.dot(mlp_np, sd) * sd
                            shared_adjust = shared_s * shared_proj  # 正=增加(更强抑制), 负=减少(松刹车)

                            out[0, position, :] += torch.tensor(
                                proj_adjust - shared_adjust,  # 注意: 增加shared=增加抑制=减边界
                                dtype=out.dtype, device=out.device
                            )
                            flag[0] = True
                            if isinstance(output, tuple):
                                return (out,) + output[1:]
                            return out
                        return output
                    return hook_fn

                h = layers_list[mid_layer].mlp.register_forward_hook(
                    make_dose_hook(b_hat_l, shared_dirs, pos, ps, ss, applied)
                )
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                grid_dcfs.append(compute_dcf(out.logits[0, -1].float().cpu().numpy(), tokenizer))

            grid_mean = np.mean(grid_dcfs, axis=0)
            key = f"p{ps}_s{ss}"
            dose_grid[key] = {
                "proj_scale": ps,
                "shared_scale": ss,
                "target_dcf": float(grid_mean[target_idx]),
                "target_delta": float(grid_mean[target_idx] - baseline_target),
            }

    # 计算相关性
    proj_effects = []
    shared_effects = []
    predicted = []
    actual = []

    for key, data in dose_grid.items():
        ps = data["proj_scale"]
        ss = data["shared_scale"]
        # 线性预测: delta ≈ a * proj_scale + b * shared_scale
        # 这里只看趋势

    # 专门看: proj_scale=0时, shared_scale对target的影响
    shared_only = {k: v for k, v in dose_grid.items() if v["proj_scale"] == 0}
    proj_only = {k: v for k, v in dose_grid.items() if v["shared_scale"] == 0}

    plog(f"    Shared-only effects (proj=0):")
    for k, v in sorted(shared_only.items()):
        plog(f"      shared_scale={v['shared_scale']:+.1f}: target_delta={v['target_delta']:+.3f}")

    plog(f"    Proj-only effects (shared=0):")
    for k, v in sorted(proj_only.items()):
        plog(f"      proj_scale={v['proj_scale']:+.1f}: target_delta={v['target_delta']:+.3f}")

    elapsed = time.time() - t0
    plog(f"    Dose curve done ({elapsed:.0f}s)")

    results = {
        "cat_name": cat_name,
        "mid_layer": mid_layer,
        "best_layer": best_layer,
        "baseline_target": float(baseline_target),
        "dose_grid": dose_grid,
        "elapsed": elapsed,
    }

    return results


# ==================== Exp4: 共享语义抑制与竞争释放 ★★★★ ====================
def exp4_shared_semantic_competition(model, tokenizer, device, model_name, W_U):
    """
    移除shared_semantic后, 记录8x8 DCF变化
    判断: shared_semantic是压制目标边界, 还是控制竞争类别释放?
    """
    plog("=== Exp4: 共享语义抑制与竞争释放连接 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers
    cat_names = PRIORITY_CATS.get(model_name, ["clothing", "fruit"])

    results = {}

    for cat_name in cat_names[:2]:  # 只测2个类别
        best_layer = BEST_LAYERS[model_name][cat_name]
        mid_layer = best_layer // 2
        plog(f"  {cat_name}: shared_semantic competition release (L{mid_layer})...")
        t0 = time.time()

        subspaces = build_subspaces(model, tokenizer, device, model_name, cat_name, best_layer)
        if subspaces is None:
            continue

        b_hat = subspaces["b_hat"]
        shared_dirs = subspaces["shared_dirs"]
        competitor_dirs = subspaces["competitor_dirs"]
        all_bc_dirs = subspaces["all_bc_dirs"]

        if not shared_dirs:
            continue

        target_idx = DCF_DIM_NAMES.index(cat_name) if cat_name in DCF_DIM_NAMES else 0
        template = RELATION_TEMPLATES["kind_of"]
        cat_objs = CATEGORIES[cat_name]

        # 获取该层的B_c方向
        spec_vec_l, spec_norm_l = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, mid_layer
        )
        if spec_vec_l is None or spec_norm_l < 1e-6:
            continue
        b_hat_l = spec_vec_l / spec_norm_l

        # 3种操作的DCF变化:
        # 1. ablate shared_semantic
        # 2. ablate B_c (对照)
        # 3. ablate competitor_bc (对照)

        ops = {
            "ablate_shared": shared_dirs,
            "ablate_bc": [b_hat_l],
            "ablate_competitor": competitor_dirs[:3],
        }

        # baseline DCF (所有对象平均)
        baseline_dcfs = []
        for obj in cat_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            baseline_dcfs.append(compute_dcf(out.logits[0, -1].float().cpu().numpy(), tokenizer))
        baseline_mean = np.mean(baseline_dcfs, axis=0)

        op_results = {}

        for op_name, dirs in ops.items():
            if not dirs:
                continue

            op_dcfs = []
            for obj in cat_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                applied = [False]
                def make_sub_abl_hook(dirs_list, position, flag):
                    def hook_fn(module, inp, output):
                        if not flag[0]:
                            if isinstance(output, tuple):
                                out = output[0].clone()
                            else:
                                out = output.clone()
                            mlp_np = out[0, position, :].float().cpu().numpy()
                            removal = np.zeros_like(mlp_np)
                            for sd in dirs_list:
                                removal += np.dot(mlp_np, sd) * sd
                            out[0, position, :] -= torch.tensor(
                                removal, dtype=out.dtype, device=out.device
                            )
                            flag[0] = True
                            if isinstance(output, tuple):
                                return (out,) + output[1:]
                            return out
                        return output
                    return hook_fn

                h = layers_list[mid_layer].mlp.register_forward_hook(
                    make_sub_abl_hook(dirs, pos, applied)
                )
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                op_dcfs.append(compute_dcf(out.logits[0, -1].float().cpu().numpy(), tokenizer))

            op_mean = np.mean(op_dcfs, axis=0)
            delta = op_mean - baseline_mean

            # 分析: 哪些DCF维度变化最大?
            dim_deltas = {}
            for i, dim_name in enumerate(DCF_DIM_NAMES):
                dim_deltas[dim_name] = float(delta[i])

            # 竞争类别vs目标类别vs其他
            target_delta = delta[target_idx]
            competitor_deltas = [delta[DCF_DIM_NAMES.index(c)] for c in all_bc_dirs if c != cat_name and c in DCF_DIM_NAMES]
            other_deltas = [delta[i] for i in range(len(delta)) if DCF_DIM_NAMES[i] != cat_name and DCF_DIM_NAMES[i] not in all_bc_dirs]

            op_results[op_name] = {
                "target_delta": float(target_delta),
                "mean_competitor_delta": float(np.mean(competitor_deltas)) if competitor_deltas else 0,
                "mean_other_delta": float(np.mean(other_deltas)) if other_deltas else 0,
                "dim_deltas": dim_deltas,
                "dcf_delta": [float(x) for x in delta],
            }

            plog(f"    {op_name:20s}: target_D={target_delta:+.3f}, "
                 f"competitor_mean={np.mean(competitor_deltas) if competitor_deltas else 0:+.3f}, "
                 f"other_mean={np.mean(other_deltas) if other_deltas else 0:+.3f}")

        elapsed = time.time() - t0
        plog(f"    {cat_name} competition release done ({elapsed:.0f}s)")

        results[cat_name] = {
            "mid_layer": mid_layer,
            "best_layer": best_layer,
            "op_results": op_results,
            "elapsed": elapsed,
        }

    return results


# ==================== Exp5: 跨模型一致性验证 ★★★ ====================
def exp5_cross_model_check(model, tokenizer, device, model_name, W_U):
    """
    在3个模型上验证同一组核心模式:
    1. 中间层orth_bc传播后alignment为负
    2. 中间层orth_bc注入削弱边界
    3. 末层orth_bc注入增强边界(如果存在)
    4. shared_semantic是orth_bc最大子成分
    """
    plog("=== Exp5: 跨模型一致性验证 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = info.n_layers

    # 对2个优先类别做快速验证
    cat_names = PRIORITY_CATS.get(model_name, ["clothing", "fruit"])

    results = {}

    for cat_name in cat_names[:2]:
        best_layer = BEST_LAYERS[model_name][cat_name]
        plog(f"  {cat_name}: cross-model consistency check...")
        t0 = time.time()

        # 检查1: 中间层orth_bc传播alignment
        mid_layer = best_layer // 2
        spec_vec_l, spec_norm_l = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, mid_layer
        )
        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )

        if spec_vec_l is None or spec_vec is None or spec_norm < 1e-6 or spec_norm_l < 1e-6:
            plog(f"    Skip {cat_name}: no B_c direction")
            continue

        b_hat_l = spec_vec_l / spec_norm_l
        b_hat = spec_vec / spec_norm

        template = RELATION_TEMPLATES["kind_of"]
        cat_objs = CATEGORIES[cat_name]

        # 收集中间层MLP的orth方向
        mlp_outputs = []
        for obj in cat_objs[:2]:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap = {}
            h = layers_list[mid_layer].mlp.register_forward_hook(
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

        # 测试: 注入orth_bc到best_layer附近
        inject_layer = max(0, best_layer - 3)
        eps = 1.0
        bc_increases = []

        for obj in cat_objs[:2]:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

            # baseline
            cap_base = {}
            h_base = layers_list[best_layer].register_forward_hook(
                lambda m, i, o: cap_base.__setitem__("v", o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu())
            )
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_base.remove()

            if "v" not in cap_base:
                continue
            base_bc_proj = float(np.dot(cap_base["v"][0, pos].numpy(), b_hat))

            # 注入orth
            cap_inj = {}
            added = [False]

            def make_inject_hook2(direction, position, strength, flag):
                def hook_fn(module, inp, output):
                    if not flag[0]:
                        if isinstance(output, tuple):
                            out = output[0].clone()
                        else:
                            out = output.clone()
                        out[0, position, :] += torch.tensor(
                            strength * direction, dtype=out.dtype, device=out.device
                        )
                        flag[0] = True
                        if isinstance(output, tuple):
                            return (out,) + output[1:]
                        return out
                    return output
                return hook_fn

            hooks = []
            hooks.append(layers_list[inject_layer].register_forward_hook(
                make_inject_hook2(orth_hat, pos, eps, added)
            ))
            hooks.append(layers_list[best_layer].register_forward_hook(
                lambda m, i, o: cap_inj.__setitem__("v", o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu())
            ))

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            for h in hooks:
                h.remove()

            if "v" in cap_inj:
                inj_bc_proj = float(np.dot(cap_inj["v"][0, pos].numpy(), b_hat))
                bc_increases.append(inj_bc_proj - base_bc_proj)

        mid_orth_effect = float(np.mean(bc_increases)) if bc_increases else 0.0

        # 检查末层orth_bc(如果存在)
        late_layer = n_layers - 1
        if late_layer != best_layer:
            spec_vec_late, spec_norm_late = get_specific_direction(
                model, tokenizer, device, model_name, cat_name, late_layer
            )
            if spec_vec_late is not None and spec_norm_late > 1e-6:
                b_hat_late = spec_vec_late / spec_norm_late

                mlp_outputs_late = []
                for obj in cat_objs[:2]:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                    cap = {}
                    h = layers_list[late_layer].mlp.register_forward_hook(
                        lambda m, i, o: cap.__setitem__("v", o[0].detach().float().cpu() if isinstance(o, tuple) else o.detach().float().cpu())
                    )
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    h.remove()
                    if "v" in cap:
                        mlp_outputs_late.append(cap["v"][0, pos].numpy())

                if mlp_outputs_late:
                    mean_mlp_late = np.mean(mlp_outputs_late, axis=0)
                    proj_late = np.dot(mean_mlp_late, b_hat_late) * b_hat_late
                    orth_late = mean_mlp_late - proj_late
                    orth_late_norm = np.linalg.norm(orth_late)

                    if orth_late_norm > 1e-8:
                        orth_late_hat = orth_late / orth_late_norm
                        # 同层传播测试
                        late_orth_alignment = float(np.dot(orth_late_hat, b_hat))

                        plog(f"    Late L{late_layer}: orth_bc alignment with B_c = {late_orth_alignment:.4f}")
                    else:
                        late_orth_alignment = None
                else:
                    late_orth_alignment = None
            else:
                late_orth_alignment = None
        else:
            late_orth_alignment = None

        # 检查shared_semantic占比
        subspaces = build_subspaces(model, tokenizer, device, model_name, cat_name, best_layer)
        shared_semantic_fraction = 0
        if subspaces and subspaces["shared_dirs"]:
            shared_semantic_fraction = len(subspaces["shared_dirs"])
        competitor_fraction = len(subspaces["competitor_dirs"]) if subspaces else 0

        elapsed = time.time() - t0
        plog(f"    {cat_name}: mid_orth_effect={mid_orth_effect:+.4f}, "
             f"late_orth_alignment={late_orth_alignment}, "
             f"n_shared={shared_semantic_fraction}, n_competitor={competitor_fraction}")

        results[cat_name] = {
            "mid_layer": mid_layer,
            "best_layer": best_layer,
            "mid_orth_inject_effect": mid_orth_effect,
            "late_orth_alignment": late_orth_alignment,
            "n_shared_dirs": shared_semantic_fraction,
            "n_competitor_dirs": competitor_fraction,
            "elapsed": elapsed,
        }

    return results


# ==================== 主流程 ====================
def run_all_experiments(model_name, round_num=1):
    plog(f"Phase 489: {model_name} (R{round_num})")
    plog(f"GPU: {torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'N/A'}")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model info: class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    W_U = get_W_U(model, model_name)
    plog(f"W_U: shape={W_U.shape}")

    all_results = {
        "phase": 489,
        "round": round_num,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }

    # Exp1: shared_semantic因果测试
    try:
        r1 = exp1_shared_semantic_causal(model, tokenizer, device, model_name, W_U)
        all_results["exp1_shared_semantic_causal"] = r1
        plog(f"Exp1 done: {list(r1.keys())}")
    except Exception as e:
        plog(f"Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_shared_semantic_causal"] = {"error": str(e)}

    save_partial(model_name, round_num, all_results, "exp1")

    # Exp2: 末层orth_bc前体验证
    try:
        r2 = exp2_late_orth_precursor(model, tokenizer, device, model_name, W_U)
        all_results["exp2_late_orth_precursor"] = r2
        plog(f"Exp2 done: {list(r2.keys())}")
    except Exception as e:
        plog(f"Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_late_orth_precursor"] = {"error": str(e)}

    save_partial(model_name, round_num, all_results, "exp2")

    # Exp3: 投影写入vs共享抑制剂量曲线
    try:
        r3 = exp3_dose_curve(model, tokenizer, device, model_name, W_U)
        all_results["exp3_dose_curve"] = r3
        plog(f"Exp3 done")
    except Exception as e:
        plog(f"Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_dose_curve"] = {"error": str(e)}

    save_partial(model_name, round_num, all_results, "exp3")

    # Exp4: 共享语义抑制与竞争释放
    try:
        r4 = exp4_shared_semantic_competition(model, tokenizer, device, model_name, W_U)
        all_results["exp4_shared_semantic_competition"] = r4
        plog(f"Exp4 done: {list(r4.keys())}")
    except Exception as e:
        plog(f"Exp4 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp4_shared_semantic_competition"] = {"error": str(e)}

    save_partial(model_name, round_num, all_results, "exp4")

    # Exp5: 跨模型一致性验证
    try:
        r5 = exp5_cross_model_check(model, tokenizer, device, model_name, W_U)
        all_results["exp5_cross_model_check"] = r5
        plog(f"Exp5 done: {list(r5.keys())}")
    except Exception as e:
        plog(f"Exp5 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp5_cross_model_check"] = {"error": str(e)}

    save_partial(model_name, round_num, all_results, "exp5")

    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    # 保存完整结果
    result_path = f"results/glm5/phase489_{model_name}_r{round_num}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    plog(f"Results saved to {result_path}")

    return all_results


def save_partial(model_name, round_num, results, exp_name):
    path = f"results/glm5/phase489_{model_name}_r{round_num}_partial.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        plog(f"  Partial save failed: {e}")


# ==================== 入口 ====================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase489_shared_semantic_brake.py <model_name> [round_num]")
        print("  model_name: qwen3, glm4, deepseek7b")
        print("  round_num: 1 (default)")
        sys.exit(1)

    model_name = sys.argv[1]
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    run_all_experiments(model_name, round_num)

    plog(f"Phase 489 {model_name} R{round_num} complete. Hard exit to release GPU.")
    import os
    os._exit(0)
