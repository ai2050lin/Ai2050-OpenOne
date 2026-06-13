"""
Phase 476: DCF分量旋转机制与神经元因果闭环
=============================================
核心问题:
1. DCF分量方向为什么先偏离后回升? 偏离的是什么方向?
2. L30 fruit writer是否有充分性?
3. 正负神经元是否共同构成类别边界?

关键实验:
1. Exp1: DCF分量方向三重追踪 — perturb/clean/target三个对齐角度
2. Exp2: 扰动强度扫描(beta scan) — 修复Phase475 Exp2的扰动过强问题
3. Exp3: 神经元充分性测试 — 在非fruit对象中注入fruit writer的write vector
4. Exp4: 正负神经元协同测试 — 正/负/正+负消融对比
5. Exp5: DS7B L27 head级分解 — 每个head的format/semantic投影

模型加载: bfloat16 + device_map="auto" + flash_attention_2
用法:
  python tests/glm5/phase476_dcf_rotation_causality.py qwen3 1
  python tests/glm5/phase476_dcf_rotation_causality.py glm4 1
  python tests/glm5/phase476_dcf_rotation_causality.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json, math
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS,
                          get_layer_weights, get_sample_layers)


def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 数据定义 ====================
CATEGORIES = {
    "fruit":    ["apple", "banana", "orange", "grape", "pear", "peach"],
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

RELATION_TEMPLATES = {
    "kind_of":    "The {obj} is a kind of",
    "used_for":   "The {obj} is commonly used for",
}

ROUNDS = {
    1: {k: v[:4] for k, v in CATEGORIES.items()},
    2: {k: v[:6] for k, v in CATEGORIES.items()},
}

FORMAT_TOKENS = [
    "(", ")", "[", "]", "{", "}", "<", ">", ",", ".", ":", ";", "!", "?",
    "-", "=", "+", "*", "/", "\\", "|", "&", "^", "%", "$", "#", "@", "~",
    "`", "'", "\"", "...", "..", "--", "---",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "10", "20", "50", "100",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "to", "for", "with", "on", "at", "by", "from", "as",
    "that", "which", "who", "whom", "this", "these", "those",
    "therefore", "because", "since", "thus", "hence", "so", "consequently",
    "however", "but", "although", "yet", "nevertheless",
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


def compute_dcf_from_logits(logits, tokenizer, dim_dict):
    dcf_vector = []
    for dim_name, words in dim_dict.items():
        logit_values = []
        for w in words:
            tid = find_token_id(tokenizer, w)
            if tid is not None and tid < len(logits):
                logit_values.append(float(logits[tid]))
        dcf_vector.append(float(np.mean(logit_values)) if logit_values else 0.0)
    return np.array(dcf_vector)


def logit_lens_dcf(resid, W_U, tokenizer, dim_dict):
    logits = resid @ W_U.T
    return compute_dcf_from_logits(logits, tokenizer, dim_dict)


def _make_capture_hook(store_dict, key):
    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            store_dict[key] = output[0].detach().float().cpu()
        else:
            store_dict[key] = output.detach().float().cpu()
    return hook_fn


def build_dcf_subspace(W_U, tokenizer, dim_dict, d_model):
    """构建DCF敏感子空间的正交基(8维)"""
    basis = []
    for dim_name, words in dim_dict.items():
        vecs = []
        for w in words:
            tid = find_token_id(tokenizer, w)
            if tid is not None and tid < W_U.shape[0]:
                vecs.append(W_U[tid])
        if vecs:
            basis.append(np.mean(vecs, axis=0))
    basis = np.array(basis)  # [8, d_model]

    # Gram-Schmidt正交化
    ortho = []
    for v in basis:
        w = v.copy()
        for u in ortho:
            proj = np.dot(w, u) / (np.dot(u, u) + 1e-12) * u
            w = w - proj
        norm = np.linalg.norm(w)
        if norm > 1e-10:
            ortho.append(w / norm)

    return np.array(ortho)  # [8, d_model]


def project_dcf_subspace(delta, ortho_basis):
    """投影delta到DCF子空间, 返回(dcf_component, null_component)"""
    # delta: [d_model], ortho_basis: [8, d_model]
    dcf_proj = np.zeros_like(delta)
    for basis_vec in ortho_basis:
        proj_len = np.dot(delta, basis_vec)
        dcf_proj += proj_len * basis_vec
    null_proj = delta - dcf_proj
    return dcf_proj, null_proj


# ==================== Exp1: DCF分量方向三重追踪 ====================
def exp1_dcf_direction_triple_alignment(model, tokenizer, device, model_name):
    """
    追踪DCF分量的三个对齐角度:
    1. cos(delta_DCF_l, perturb_direction) — 与初始扰动方向对齐
    2. cos(delta_DCF_l, clean_DCF_direction) — 与干净DCF方向对齐
    3. cos(delta_DCF_l, target_category_direction) — 与目标类别方向对齐
    """
    plog(f"=== Exp1: DCF Direction Triple Alignment ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    cat_list = ["fruit", "animal", "vehicle", "tool"]
    obj_dict = ROUNDS[1]
    n_obj = 4
    betas = [3.0]  # 用中等强度, 避免过度破坏流形

    # 获取sample layers — 只取关键层以提高效率
    if model_name == "qwen3":
        sample_layers = [24, 27, 30, 33, 35]
    elif model_name == "glm4":
        sample_layers = [24, 27, 30, 33, 35, 37]
    else:
        sample_layers = [24, 25, 26]

    # 构建DCF子空间
    ortho_basis = build_dcf_subspace(W_U, tokenizer, FAMILY_WORDS_8D, info.d_model)
    plog(f"  DCF subspace: {ortho_basis.shape[0]} dims")

    results = {}
    perturb_types = ["anti_fruit", "toward_vehicle", "toward_animal"]

    for pt in perturb_types:
        plog(f"  Perturbation type: {pt}")

        # 先计算类别方向(用于构造扰动) — 用第一层的残差
        cat_resids = {}
        for cat in ["fruit", "animal", "vehicle", "tool"]:
            resids = []
            for obj in obj_dict.get(cat, [])[:n_obj]:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                seq_len = attention_mask.sum().item()
                pos = seq_len - 1

                cap = {}
                h = layers_list[0].register_forward_hook(_make_capture_hook(cap, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                if "resid" in cap:
                    resids.append(cap["resid"][0, pos].numpy())
            if resids:
                cat_resids[cat] = np.mean(resids, axis=0)

        # 构造扰动方向
        if pt == "anti_fruit":
            perturb_dir = -cat_resids.get("fruit", np.zeros(info.d_model))
            perturb_dir = perturb_dir / (np.linalg.norm(perturb_dir) + 1e-12)
        elif pt == "toward_vehicle":
            dir_vec = cat_resids.get("vehicle", np.zeros(info.d_model)) - cat_resids.get("fruit", np.zeros(info.d_model))
            perturb_dir = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)
        elif pt == "toward_animal":
            dir_vec = cat_resids.get("animal", np.zeros(info.d_model)) - cat_resids.get("fruit", np.zeros(info.d_model))
            perturb_dir = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)

        # 用fruit对象做测试
        test_cat = "fruit"
        test_objs = obj_dict.get(test_cat, [])[:n_obj]

        beta = betas[0]
        target_norm = beta * math.sqrt(info.d_model)

        triple_alignments = {li: {} for li in sample_layers}

        for oi, obj in enumerate(test_objs):
            plog(f"    {pt} obj={obj} ({oi+1}/{len(test_objs)})")
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = attention_mask.sum().item()
            pos = seq_len - 1

            # Clean baseline: 一次前向传播捕获所有sample layers
            clean_resids = {}
            hooks_clean = []
            for li in sample_layers:
                key = f"L{li}"
                h = layers_list[li].register_forward_hook(_make_capture_hook(clean_resids, key))
                hooks_clean.append(h)
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            for h in hooks_clean:
                h.remove()

            # Perturbed: 在第一层注入扰动, 一次前向捕获所有sample layers
            perturb_vec = perturb_dir * target_norm
            perturb_tensor = torch.tensor(perturb_vec, dtype=torch.float32)

            pert_resids = {}
            hooks_pert = []
            for li in sample_layers:
                key = f"L{li}"
                h = layers_list[li].register_forward_hook(_make_capture_hook(pert_resids, key))
                hooks_pert.append(h)

            # 扰动hook
            added = [False]
            def make_perturb_hook(pvec, position):
                _added = [False]
                def hook_fn(module, inp, output):
                    if not _added[0]:
                        if isinstance(output, tuple):
                            out = output[0].clone()
                        else:
                            out = output.clone()
                        out[0, position, :] += pvec.to(out.device).to(out.dtype)
                        _added[0] = True
                        if isinstance(output, tuple):
                            return (out,) + output[1:]
                        return out
                    return output
                return hook_fn

            h0 = layers_list[0].register_forward_hook(make_perturb_hook(perturb_tensor, pos))

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            h0.remove()
            for h in hooks_pert:
                h.remove()

            # 计算三重对齐
            for li in sample_layers:
                key = f"L{li}"
                if key in clean_resids and key in pert_resids:
                    clean_r = clean_resids[key][0, pos].numpy()
                    pert_r = pert_resids[key][0, pos].numpy()
                    delta = pert_r - clean_r

                    # DCF投影分解
                    dcf_comp, _ = project_dcf_subspace(delta, ortho_basis)
                    dcf_norm = np.linalg.norm(dcf_comp)

                    if dcf_norm > 1e-10:
                        dcf_dir = dcf_comp / dcf_norm

                        # 1. 与扰动方向对齐
                        perturb_dir_normed = perturb_dir / (np.linalg.norm(perturb_dir) + 1e-12)
                        cos_perturb = float(np.dot(dcf_dir, perturb_dir_normed))

                        # 2. 与干净状态的DCF投影方向对齐
                        clean_dcf_comp, _ = project_dcf_subspace(clean_r, ortho_basis)
                        clean_dcf_norm = np.linalg.norm(clean_dcf_comp)
                        if clean_dcf_norm > 1e-10:
                            cos_clean = float(np.dot(dcf_dir, clean_dcf_comp / clean_dcf_norm))
                        else:
                            cos_clean = 0.0

                        # 3. 与目标类别(fruit)方向的DCF投影对齐
                        if "fruit" in cat_resids:
                            fruit_dcf_comp, _ = project_dcf_subspace(cat_resids["fruit"], ortho_basis)
                            fruit_dcf_norm = np.linalg.norm(fruit_dcf_comp)
                            if fruit_dcf_norm > 1e-10:
                                cos_target = float(np.dot(dcf_dir, fruit_dcf_comp / fruit_dcf_norm))
                            else:
                                cos_target = 0.0
                        else:
                            cos_target = 0.0

                        for key_name, val in [("cos_perturb", cos_perturb),
                                             ("cos_clean", cos_clean),
                                             ("cos_target_fruit", cos_target)]:
                            if key_name not in triple_alignments[li]:
                                triple_alignments[li][key_name] = []
                            triple_alignments[li][key_name].append(val)

        # 汇总
        result = {}
        for li in sample_layers:
            r = {}
            for key_name in ["cos_perturb", "cos_clean", "cos_target_fruit"]:
                vals = triple_alignments[li].get(key_name, [])
                r[key_name] = float(np.mean(vals)) if vals else 0.0
            result[f"L{li}"] = r
        results[pt] = result
        plog(f"  {pt} done. L{sample_layers[-1]}: cos_perturb={result[f'L{sample_layers[-1]}']['cos_perturb']:.4f}, "
             f"cos_clean={result[f'L{sample_layers[-1]}']['cos_clean']:.4f}, "
             f"cos_target={result[f'L{sample_layers[-1]}']['cos_target_fruit']:.4f}")

    return results


# ==================== Exp2: 扰动强度扫描 ====================
def exp2_perturbation_beta_scan(model, tokenizer, device, model_name):
    """
    用不同beta值测试方向性扰动的DCF恢复, 找到不破坏流形的扰动强度
    """
    plog(f"=== Exp2: Perturbation Beta Scan ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    cat_list = ["fruit", "animal"]
    obj_dict = ROUNDS[1]
    n_obj = 3
    betas = [0.5, 1.0, 2.0, 3.0, 5.0]

    # 获取中间层和最后层
    if model_name == "qwen3":
        mid_layer = 30
        last_layer = 35
    elif model_name == "glm4":
        mid_layer = 30
        last_layer = 37  # 排除异常L38-39
    else:
        mid_layer = 24
        last_layer = 26  # 排除L27格式覆盖

    # 计算类别方向
    cat_resids = {}
    for cat in ["fruit", "animal", "vehicle"]:
        resids = []
        for obj in obj_dict.get(cat, [])[:n_obj]:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = attention_mask.sum().item()
            pos = seq_len - 1
            cap = {}
            h = layers_list[mid_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap:
                resids.append(cap["resid"][0, pos].numpy())
        if resids:
            cat_resids[cat] = np.mean(resids, axis=0)

    # anti_fruit扰动
    fruit_dir = cat_resids.get("fruit", np.zeros(info.d_model))
    perturb_dir = -fruit_dir / (np.linalg.norm(fruit_dir) + 1e-12)

    results = {}

    for beta in betas:
        plog(f"  Beta={beta}")
        target_norm = beta * math.sqrt(info.d_model)
        perturb_vec = perturb_dir * target_norm

        dcf_changes = {"mid": [], "last": []}

        for obj in obj_dict.get("fruit", [])[:n_obj]:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = attention_mask.sum().item()
            pos = seq_len - 1

            # Clean baseline: 一次前向捕获两层
            clean_data = {}
            hooks_c = []
            hooks_c.append(layers_list[mid_layer].register_forward_hook(_make_capture_hook(clean_data, "mid")))
            hooks_c.append(layers_list[last_layer].register_forward_hook(_make_capture_hook(clean_data, "last")))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            for h in hooks_c:
                h.remove()

            clean_dcfs = {}
            if "mid" in clean_data:
                clean_dcfs[mid_layer] = logit_lens_dcf(clean_data["mid"][0, pos].numpy(), W_U, tokenizer, FAMILY_WORDS_8D)
            if "last" in clean_data:
                clean_dcfs[last_layer] = logit_lens_dcf(clean_data["last"][0, pos].numpy(), W_U, tokenizer, FAMILY_WORDS_8D)

            # Perturbed: 一次前向 + 扰动hook
            perturb_tensor = torch.tensor(perturb_vec, dtype=torch.float32)
            pert_data = {}
            hooks_p = []
            hooks_p.append(layers_list[mid_layer].register_forward_hook(_make_capture_hook(pert_data, "mid")))
            hooks_p.append(layers_list[last_layer].register_forward_hook(_make_capture_hook(pert_data, "last")))

            def make_perturb_hook(pvec, position):
                _added = [False]
                def hook_fn(module, inp, output):
                    if not _added[0]:
                        if isinstance(output, tuple):
                            out = output[0].clone()
                        else:
                            out = output.clone()
                        out[0, position, :] += pvec.to(out.device).to(out.dtype)
                        _added[0] = True
                        if isinstance(output, tuple):
                            return (out,) + output[1:]
                        return out
                    return output
                return hook_fn

            h0 = layers_list[0].register_forward_hook(make_perturb_hook(perturb_tensor, pos))

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            h0.remove()
            for h in hooks_p:
                h.remove()

            for li_key, li in [("mid", mid_layer), ("last", last_layer)]:
                if li_key in pert_data and li in clean_dcfs:
                    pert_dcf = logit_lens_dcf(pert_data[li_key][0, pos].numpy(), W_U, tokenizer, FAMILY_WORDS_8D)
                    cos_sim = float(np.dot(pert_dcf, clean_dcfs[li]) /
                                   (np.linalg.norm(pert_dcf) * np.linalg.norm(clean_dcfs[li]) + 1e-12))
                    dcf_changes[li_key].append(cos_sim)

        results[f"beta_{beta}"] = {
            "mid_recovery": float(np.mean(dcf_changes["mid"])) if dcf_changes["mid"] else 0.0,
            "last_recovery": float(np.mean(dcf_changes["last"])) if dcf_changes["last"] else 0.0,
            "target_norm": target_norm,
        }
        plog(f"    mid_recovery={results[f'beta_{beta}']['mid_recovery']:.4f}, "
             f"last_recovery={results[f'beta_{beta}']['last_recovery']:.4f}")

    return results


# ==================== Exp3: 神经元充分性测试 ====================
def exp3_neuron_sufficiency(model, tokenizer, device, model_name):
    """
    在非fruit对象中注入L30 fruit writer的write vector, 看fruit DCF是否上升
    只对Qwen3执行
    """
    if model_name != "qwen3":
        plog(f"  Exp3 skipped (only for qwen3)")
        return {"skipped": True, "reason": "only for qwen3"}

    plog(f"=== Exp3: Neuron Sufficiency Test ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    target_layer = 30  # L30有fruit特异性(11:1)

    # 获取L30的W_down
    lw = get_layer_weights(layers_list[target_layer], info.d_model, info.mlp_type)
    W_down = lw.W_down  # [d_model, intermediate_size]

    if W_down is None:
        plog(f"  W_down is None, trying direct access...")
        W_down = layers_list[target_layer].mlp.down_proj.weight.detach().float().cpu().numpy().T

    obj_dict = ROUNDS[1]
    n_obj = 4

    # Step1: 找到fruit正贡献神经元(与Phase475相同方法)
    plog(f"  Finding fruit-positive neurons at L{target_layer}...")
    cat_list = ["fruit", "animal", "vehicle", "tool"]
    fruit_contributions = {}  # neuron_idx -> contribution to fruit DCF dim 0

    for obj in obj_dict.get("fruit", [])[:n_obj]:
        prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        seq_len = attention_mask.sum().item()
        pos = seq_len - 1

        cap_mid = {}
        cap_out = {}
        h_mid = layers_list[target_layer].mlp.down_proj.register_forward_hook(
            lambda m, i, o: cap_mid.update({"mid": i[0].detach().float().cpu()}) if isinstance(i, tuple) and len(i) > 0 else None)
        h_out = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap_out, "resid"))

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

        h_mid.remove()
        h_out.remove()

        if "mid" in cap_mid and "resid" in cap_out:
            mid_act = cap_mid["mid"][0, pos].numpy()  # [intermediate_size]
            for idx in range(min(len(mid_act), W_down.shape[1])):
                if idx not in fruit_contributions:
                    fruit_contributions[idx] = 0.0
                # 贡献 = W_down[:, idx] * mid_act[idx] 对DCF维度0(fruit)的影响
                write_vec = W_down[:, idx] * mid_act[idx]
                logits = write_vec @ W_U.T
                dcf_dim0 = 0.0
                for w in FAMILY_WORDS_8D["fruit"]:
                    tid = find_token_id(tokenizer, w)
                    if tid is not None and tid < len(logits):
                        dcf_dim0 += float(logits[tid])
                dcf_dim0 /= len(FAMILY_WORDS_8D["fruit"])
                fruit_contributions[idx] += dcf_dim0

    # 排序获取top-20 fruit-positive神经元
    sorted_neurons = sorted(fruit_contributions.items(), key=lambda x: -x[1])
    top20_pos = [int(n[0]) for n in sorted_neurons[:20]]
    plog(f"  Top-20 fruit-positive neurons: {top20_pos}")

    # Step2: 计算这些神经元的平均write vector
    plog(f"  Computing mean fruit write vector...")
    write_vectors = []
    for obj in obj_dict.get("fruit", [])[:n_obj]:
        prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        seq_len = attention_mask.sum().item()
        pos = seq_len - 1

        cap_mid = {}
        h_mid = layers_list[target_layer].mlp.down_proj.register_forward_hook(
            lambda m, i, o: cap_mid.update({"mid": i[0].detach().float().cpu()}) if isinstance(i, tuple) and len(i) > 0 else None)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h_mid.remove()

        if "mid" in cap_mid:
            mid_act = cap_mid["mid"][0, pos].numpy()
            # 只取top-20神经元的贡献
            wv = np.zeros(info.d_model)
            for idx in top20_pos:
                if idx < W_down.shape[1]:
                    wv += W_down[:, idx] * mid_act[idx]
            write_vectors.append(wv)

    mean_write_vec = np.mean(write_vectors, axis=0) if write_vectors else np.zeros(info.d_model)
    write_vec_norm = np.linalg.norm(mean_write_vec)
    plog(f"  Mean fruit write vector norm: {write_vec_norm:.4f}")

    # Step3: 注入测试 — 在非fruit对象中注入write vector
    plog(f"  Injecting fruit write vector into non-fruit objects...")
    inject_targets = {
        "animal": obj_dict.get("animal", [])[:3],
        "vehicle": obj_dict.get("vehicle", [])[:3],
        "tool": obj_dict.get("tool", [])[:3],
    }

    # 控制注入强度: 使用不同amplification
    amplifications = [0.5, 1.0, 2.0]
    results = {}

    for amp in amplifications:
        plog(f"    Amplification={amp}")
        inject_vec = mean_write_vec * amp
        inject_tensor = torch.tensor(inject_vec, dtype=torch.float32)

        cat_dcf_changes = {}
        for target_cat, objs in inject_targets.items():
            dcf_before_list = []
            dcf_after_list = []
            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                seq_len = attention_mask.sum().item()
                pos = seq_len - 1

                # Clean DCF
                cap_clean = {}
                h_clean = layers_list[info.n_layers - 1].register_forward_hook(
                    _make_capture_hook(cap_clean, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_clean.remove()

                if "resid" in cap_clean:
                    dcf_before = logit_lens_dcf(cap_clean["resid"][0, pos].numpy(), W_U, tokenizer, FAMILY_WORDS_8D)

                    # 注入write vector到L30
                    cap_pert = {}
                    h_pert = layers_list[info.n_layers - 1].register_forward_hook(
                        _make_capture_hook(cap_pert, "resid"))

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

                    h_inj = layers_list[target_layer].register_forward_hook(
                        make_inject_hook(inject_tensor, pos))

                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)

                    h_pert.remove()
                    h_inj.remove()

                    if "resid" in cap_pert:
                        dcf_after = logit_lens_dcf(cap_pert["resid"][0, pos].numpy(), W_U, tokenizer, FAMILY_WORDS_8D)
                        dcf_before_list.append(dcf_before)
                        dcf_after_list.append(dcf_after)

            if dcf_before_list:
                mean_before = np.mean(dcf_before_list, axis=0)
                mean_after = np.mean(dcf_after_list, axis=0)
                cat_dcf_changes[target_cat] = {
                    "before": {cat_list[i]: float(mean_before[i]) for i in range(min(len(cat_list), len(mean_before)))},
                    "after": {cat_list[i]: float(mean_after[i]) for i in range(min(len(cat_list), len(mean_after)))},
                    "delta": {cat_list[i]: float(mean_after[i] - mean_before[i]) for i in range(min(len(cat_list), len(mean_after)))},
                }
                fruit_delta = cat_dcf_changes[target_cat]["delta"].get("fruit", 0.0)
                plog(f"      {target_cat}: fruit DCF delta={fruit_delta:.4f}")

        results[f"amp_{amp}"] = cat_dcf_changes

    results["top20_neurons"] = top20_pos
    results["write_vec_norm"] = float(write_vec_norm)
    return results


# ==================== Exp4: 正负神经元协同 ====================
def exp4_positive_negative_synergy(model, tokenizer, device, model_name):
    """
    测试正写入器+负抑制器的协同效应
    只对Qwen3执行
    """
    if model_name != "qwen3":
        plog(f"  Exp4 skipped (only for qwen3)")
        return {"skipped": True, "reason": "only for qwen3"}

    plog(f"=== Exp4: Positive-Negative Neuron Synergy ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    target_layer = 30
    lw = get_layer_weights(layers_list[target_layer], info.d_model, info.mlp_type)
    W_down = lw.W_down
    if W_down is None:
        plog(f"  W_down is None, trying direct access...")
        W_down = layers_list[target_layer].mlp.down_proj.weight.detach().float().cpu().numpy().T

    cat_list = ["fruit", "animal", "vehicle", "tool"]
    obj_dict = ROUNDS[1]
    n_obj = 4

    # Step1: 找到所有fruit相关神经元(正+负)
    plog(f"  Finding fruit-positive and fruit-negative neurons at L{target_layer}...")
    fruit_contributions = {}

    for obj in obj_dict.get("fruit", [])[:n_obj]:
        prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        seq_len = attention_mask.sum().item()
        pos = seq_len - 1

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

    # 排序
    sorted_neurons = sorted(fruit_contributions.items(), key=lambda x: -x[1])
    top20_pos = set(int(n[0]) for n in sorted_neurons[:20])  # 正贡献
    top20_neg = set(int(n[0]) for n in sorted_neurons[-20:])  # 负贡献(抑制器)
    random_20 = set(int(n[0]) for n in sorted_neurons[len(sorted_neurons)//2:len(sorted_neurons)//2+20])  # 随机

    plog(f"  Top-20 positive: {sorted(top20_pos)}")
    plog(f"  Top-20 negative: {sorted(top20_neg)}")

    # Step2: 四种消融模式
    ablation_modes = {
        "positive_only": top20_pos,
        "negative_only": top20_neg,
        "positive_plus_negative": top20_pos | top20_neg,
        "random_20": random_20,
    }

    results = {}

    for mode_name, neuron_set in ablation_modes.items():
        plog(f"  Ablation mode: {mode_name} ({len(neuron_set)} neurons)")
        neuron_list = [int(i) for i in neuron_set]

        cat_dcfs = {}
        for cat in cat_list:
            objs = obj_dict.get(cat, [])[:n_obj]
            dcfs = []
            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                seq_len = attention_mask.sum().item()
                pos = seq_len - 1

                # 用pre_hook修改down_proj输入: 零掉指定神经元
                def make_pre_ablation_hook(indices, position):
                    def pre_hook(module, args):
                        x = args[0].clone()
                        for idx in indices:
                            if idx < x.shape[-1]:
                                x[0, position, idx] = 0.0
                        return (x,) + args[1:] if len(args) > 1 else (x,)
                    return pre_hook

                h_abl = layers_list[target_layer].mlp.down_proj.register_forward_pre_hook(
                    make_pre_ablation_hook(neuron_list, pos))

                cap = {}
                h_resid = layers_list[info.n_layers - 1].register_forward_hook(
                    _make_capture_hook(cap, "resid"))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                h_abl.remove()
                h_resid.remove()

                if "resid" in cap:
                    dcfs.append(logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer, FAMILY_WORDS_8D))

            if dcfs:
                cat_dcfs[cat] = np.mean(dcfs, axis=0)

        # 计算DCF变化(相对于Phase475的baseline)
        # 这里直接记录绝对DCF值
        mode_result = {}
        for cat in cat_list:
            if cat in cat_dcfs:
                mode_result[cat] = {cat_list[i]: float(cat_dcfs[cat][i]) for i in range(min(len(cat_list), len(cat_dcfs[cat])))}

        results[mode_name] = mode_result

        # 计算fruit特异性
        if "fruit" in cat_dcfs and "animal" in cat_dcfs:
            fruit_dcf = cat_dcfs["fruit"][0] if len(cat_dcfs["fruit"]) > 0 else 0
            animal_dcf = cat_dcfs["animal"][0] if len(cat_dcfs["animal"]) > 0 else 0
            plog(f"    fruit_dcf_dim0={fruit_dcf:.4f}, animal_dcf_dim0={animal_dcf:.4f}, "
                 f"ratio={abs(fruit_dcf)/(abs(animal_dcf)+0.01):.2f}")

    return results


# ==================== Exp5: DS7B L27 Head级分解 ====================
def exp5_ds7b_l27_head_decomposition(model, tokenizer, device, model_name):
    """
    把L27 Attention拆成每个head, 计算各head的format/semantic投影
    只对DS7B执行
    """
    if model_name != "deepseek7b":
        plog(f"  Exp5 skipped (only for deepseek7b)")
        return {"skipped": True, "reason": "only for deepseek7b"}

    plog(f"=== Exp5: DS7B L27 Head-Level Decomposition ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    target_layer = 27
    obj_dict = ROUNDS[1]
    n_obj = 2  # 减少对象数量加快head ablation

    # 获取L27的attention配置
    layer = layers_list[target_layer]
    # Qwen2没有num_heads属性, 从config获取
    if hasattr(layer.self_attn, 'num_heads'):
        n_heads = layer.self_attn.num_heads
    elif hasattr(model.config, 'num_attention_heads'):
        n_heads = model.config.num_attention_heads
    else:
        n_heads = info.d_model // layer.self_attn.head_dim
    head_dim = layer.self_attn.head_dim
    plog(f"  L27: n_heads={n_heads}, head_dim={head_dim}")

    # 获取W_o — 对device_map=auto的模型需要特殊处理
    try:
        W_o = layer.self_attn.o_proj.weight.detach().float().cpu().numpy()  # [d_model, d_model]
    except (NotImplementedError, RuntimeError):
        # meta device, 用ablation方法
        plog(f"  W_o on meta device, using ablation method")
        W_o = None

    # 找到format和semantic token IDs
    format_ids = []
    for t in FORMAT_TOKENS:
        tid = find_token_id(tokenizer, t)
        if tid is not None:
            format_ids.append(tid)

    semantic_ids = []
    for t in SEMANTIC_TOKENS:
        tid = find_token_id(tokenizer, t)
        if tid is not None:
            semantic_ids.append(tid)

    plog(f"  Format tokens: {len(format_ids)}, Semantic tokens: {len(semantic_ids)}")

    # 方法: 用两次前向传播, 一次clean一次关闭单个head, 差值就是head贡献
    # 但更高效: hook o_proj的输入, 按head切分, 如果有W_o则直接计算
    # 如果没有W_o, 就用ablation方法: 逐个head ablation看效果

    results = {}

    if W_o is not None:
        # 直接方法: W_o @ head_output
        for obj in obj_dict.get("fruit", [])[:n_obj]:
            plog(f"  Processing: {obj}")
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = attention_mask.sum().item()
            pos = seq_len - 1

            cap_attn = {}
            def attn_output_hook(module, args):
                if isinstance(args, tuple) and len(args) > 0:
                    cap_attn["attn_output"] = args[0].detach().float().cpu()

            h_attn = layer.self_attn.o_proj.register_forward_pre_hook(attn_output_hook)
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_attn.remove()

            if "attn_output" in cap_attn:
                attn_out = cap_attn["attn_output"][0, pos].numpy()
                for head_idx in range(n_heads):
                    start = head_idx * head_dim
                    end = start + head_dim
                    head_output = attn_out[start:end]
                    head_contribution = W_o[:, start:end] @ head_output
                    logits = head_contribution @ W_U.T
                    format_score = float(np.mean([logits[tid] for tid in format_ids if tid < len(logits)]))
                    semantic_score = float(np.mean([logits[tid] for tid in semantic_ids if tid < len(logits)]))
                    if f"head_{head_idx}" not in results:
                        results[f"head_{head_idx}"] = {"format_scores": [], "semantic_scores": [], "fmt_minus_sem": []}
                    results[f"head_{head_idx}"]["format_scores"].append(format_score)
                    results[f"head_{head_idx}"]["semantic_scores"].append(semantic_score)
                    results[f"head_{head_idx}"]["fmt_minus_sem"].append(format_score - semantic_score)
    else:
        # Ablation方法: 逐个head关闭, 看对format/semantic的影响
        # 更高效: hook o_proj的输入, 手动零掉指定head, 计算残差差值
        plog(f"  Using head ablation method (no W_o access)")
        
        for obj in obj_dict.get("fruit", [])[:n_obj]:
            plog(f"  Processing: {obj}")
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = attention_mask.sum().item()
            pos = seq_len - 1

            # Clean baseline: 捕获L27输出
            cap_clean = {}
            h_clean = layer.register_forward_hook(_make_capture_hook(cap_clean, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_clean.remove()

            if "resid" in cap_clean:
                clean_resid = cap_clean["resid"][0, pos].numpy()
                clean_logits = clean_resid @ W_U.T
                clean_format = float(np.mean([clean_logits[tid] for tid in format_ids if tid < len(clean_logits)]))
                clean_semantic = float(np.mean([clean_logits[tid] for tid in semantic_ids if tid < len(clean_logits)]))

            # 逐个head ablation: 零掉o_proj输入中的指定head维度
            for head_idx in range(n_heads):
                start = head_idx * head_dim
                end = start + head_dim

                cap_abl = {}
                h_abl = layer.register_forward_hook(_make_capture_hook(cap_abl, "resid"))

                def make_head_ablation_hook(h_start, h_end, position):
                    def pre_hook(module, args):
                        if isinstance(args, tuple) and len(args) > 0:
                            x = args[0].clone()
                            x[0, position, h_start:h_end] = 0.0
                            return (x,) + args[1:] if len(args) > 1 else (x,)
                        return args
                    return pre_hook

                h_pre = layer.self_attn.o_proj.register_forward_pre_hook(
                    make_head_ablation_hook(start, end, pos))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                h_pre.remove()
                h_abl.remove()

                if "resid" in cap_abl:
                    abl_resid = cap_abl["resid"][0, pos].numpy()
                    abl_logits = abl_resid @ W_U.T
                    abl_format = float(np.mean([abl_logits[tid] for tid in format_ids if tid < len(abl_logits)]))
                    abl_semantic = float(np.mean([abl_logits[tid] for tid in semantic_ids if tid < len(abl_logits)]))

                    # head贡献 = clean - ablation
                    format_contrib = clean_format - abl_format
                    semantic_contrib = clean_semantic - abl_semantic

                    if f"head_{head_idx}" not in results:
                        results[f"head_{head_idx}"] = {"format_scores": [], "semantic_scores": [], "fmt_minus_sem": []}
                    results[f"head_{head_idx}"]["format_scores"].append(format_contrib)
                    results[f"head_{head_idx}"]["semantic_scores"].append(semantic_contrib)
                    results[f"head_{head_idx}"]["fmt_minus_sem"].append(format_contrib - semantic_contrib)

    # 汇总
    for head_key in results:
        r = results[head_key]
        r["mean_format"] = float(np.mean(r["format_scores"]))
        r["mean_semantic"] = float(np.mean(r["semantic_scores"]))
        r["mean_fmt_minus_sem"] = float(np.mean(r["fmt_minus_sem"]))
        r["format_dominant"] = r["mean_format"] > r["mean_semantic"]
        # 清理原始列表
        del r["format_scores"]
        del r["semantic_scores"]
        del r["fmt_minus_sem"]

    # 找到format覆盖head
    format_heads = []
    semantic_heads = []
    for head_key, r in results.items():
        if r["format_dominant"]:
            format_heads.append(head_key)
        else:
            semantic_heads.append(head_key)

    plog(f"  Format-dominant heads: {format_heads}")
    plog(f"  Semantic-dominant heads: {semantic_heads}")

    # 排序: fmt_minus_sem最大的heads
    sorted_heads = sorted(results.items(), key=lambda x: -x[1]["mean_fmt_minus_sem"])
    top5_info = [(h, round(r["mean_fmt_minus_sem"], 2)) for h, r in sorted_heads[:5]]
    plog(f"  Top-5 format-overwrite heads: {top5_info}")

    results["summary"] = {
        "n_heads": n_heads,
        "format_dominant_heads": format_heads,
        "semantic_dominant_heads": semantic_heads,
        "top5_format_overwrite": [h for h, _ in sorted_heads[:5]],
    }

    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    plog(f"Phase 476: DCF Rotation & Neuron Causality")
    plog(f"Model: {model_name}, Round: {round_num}")

    t_start = time.time()

    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model info: class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    results = {
        "phase": 476,
        "model": model_name,
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "theory": "DCF Rotation Mechanism & Neuron Causal Closure",
        "core_question": "Why does DCF component direction deviate then recover? Do L30 fruit writers have sufficiency?",
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }

    # Exp1: DCF分量方向三重追踪
    try:
        results["exp1_dcf_direction_triple_alignment"] = exp1_dcf_direction_triple_alignment(
            model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp1_dcf_direction_triple_alignment"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Exp1 done. Elapsed: {time.time()-t_start:.1f}s")

    # Exp2: 扰动强度扫描
    try:
        results["exp2_perturbation_beta_scan"] = exp2_perturbation_beta_scan(
            model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp2_perturbation_beta_scan"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Exp2 done. Elapsed: {time.time()-t_start:.1f}s")

    # Exp3: 神经元充分性测试(仅qwen3)
    try:
        results["exp3_neuron_sufficiency"] = exp3_neuron_sufficiency(
            model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp3_neuron_sufficiency"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Exp3 done. Elapsed: {time.time()-t_start:.1f}s")

    # Exp4: 正负神经元协同(仅qwen3)
    try:
        results["exp4_positive_negative_synergy"] = exp4_positive_negative_synergy(
            model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp4 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp4_positive_negative_synergy"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Exp4 done. Elapsed: {time.time()-t_start:.1f}s")

    # Exp5: DS7B L27 head分解(仅deepseek7b)
    try:
        results["exp5_ds7b_l27_head_decomposition"] = exp5_ds7b_l27_head_decomposition(
            model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp5 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp5_ds7b_l27_head_decomposition"] = {"error": str(e)}

    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase476_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    plog(f"Results saved to {out_path}")

    # 释放模型
    release_model(model)
    t_total = time.time() - t_start
    plog(f"Phase 476 {model_name} complete. Total: {t_total:.1f}s ({t_total/60:.1f}min)")


if __name__ == "__main__":
    main()
