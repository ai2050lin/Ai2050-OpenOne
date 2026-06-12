"""
Phase 474: Attractor Mechanism Decomposition, Format Overlay Exclusion & Neuron-Level Writers
==============================================================================================
核心改进(基于Phase 473缺陷修正):
1. 精确定向扰动 — 用真实残差空间类别差异方向替代随机扰动
2. 扰动传播追踪 — 追踪||delta||=||perturbed-clean||在层间的变化(最客观指标)
3. DS7B L27排除验证 — 分别监控L24-L26(不含L27)
4. 格式/语义子空间投影 — 客观测量每层的格式vs语义倾向
5. 神经元级DCF写入贡献 — 对Qwen3关键层定位约束写入神经元

模型加载: bfloat16 + device_map="auto" + flash_attention_2
用法:
  python tests/glm5/phase474_attractor_mechanism.py qwen3 1
  python tests/glm5/phase474_attractor_mechanism.py glm4 1
  python tests/glm5/phase474_attractor_mechanism.py deepseek7b 1
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
    1: {k: v[:4] for k, v in CATEGORIES.items()},  # R1: 4对象/类(基础测试)
    2: {k: v[:6] for k, v in CATEGORIES.items()},  # R2: 6对象/类(确认测试)
}

# 格式token定义 — 用于Exp4
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

    # 确保所有层加载完整 — device_map="auto"会自动分配GPU/CPU
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

    # 验证所有层都加载成功
    layers_list = get_layers(model)
    n_loaded = len(layers_list)
    plog(f"  Loaded {n_loaded} transformer layers")

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
        plog(f"  Layer distribution: {gpu_layers} GPU + {cpu_layers} CPU (total {n_loaded})")
        # 检查深层是否缺失
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


def cluster_quality(vectors, labels):
    from scipy.spatial.distance import pdist, squareform
    if len(vectors) < 3 or len(set(labels)) < 2:
        return 0.0
    vectors = np.array(vectors)
    labels = np.array(labels)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    vectors_norm = vectors / norms
    dist_matrix = squareform(pdist(vectors_norm, metric='cosine'))
    silhouette_values = []
    unique_labels = list(set(labels))
    for i in range(len(vectors)):
        own_label = labels[i]
        own_cluster = [j for j in range(len(vectors)) if labels[j] == own_label and j != i]
        other_clusters = {l: [j for j in range(len(vectors)) if labels[j] == l]
                          for l in unique_labels if l != own_label}
        if not own_cluster:
            continue
        a = np.mean([dist_matrix[i, j] for j in own_cluster])
        b = float('inf')
        for l, indices in other_clusters.items():
            if indices:
                b = min(b, np.mean([dist_matrix[i, j] for j in indices]))
        if b == float('inf'):
            b = 0
        s = (b - a) / max(a, b, 1e-10)
        silhouette_values.append(s)
    return float(np.mean(silhouette_values)) if silhouette_values else 0.0


def logit_lens_dcf(resid, W_U, tokenizer, dim_dict):
    logits = resid @ W_U.T
    return compute_dcf_from_logits(logits, tokenizer, dim_dict)


def collect_residuals(model, tokenizer, model_name, device, prompts, layer_indices):
    """收集指定层的残差向量(批量高效版)"""
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    layers_list = get_layers(model)

    results = {li: {} for li in layer_indices}

    for prompt_key, prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        seq_len = attention_mask.sum().item()
        pos = seq_len - 1

        captured = {}
        hooks = []
        for li in layer_indices:
            def make_hook(key):
                def hook_fn(module, inp, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook_fn
            hooks.append(layers_list[li].register_forward_hook(make_hook(f"L{li}")))

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks:
            h.remove()

        for li in layer_indices:
            key = f"L{li}"
            if key in captured:
                results[li][prompt_key] = captured[key][0, pos].numpy()

    return results


# ==================== Exp1: 精确定向扰动恢复 ====================
def exp1_precise_directional_perturbation(model, tokenizer, model_name, device, obj_dict):
    """
    核心改进: 用真实残差空间类别差异方向替代随机扰动
    
    步骤:
    1. 收集每个类别在perturb_layer的均值残差
    2. 构造精确扰动方向:
       - anti_dcf: -(mean_fruit - mean_all) — 推离fruit
       - cross_category: +(mean_vehicle - mean_all) — 推向vehicle
       - random: 随机方向(对照)
    3. 在perturb_layer注入扰动, 追踪恢复
    
    指标:
    - DCF方向恢复: cos(dcf_perturbed, dcf_clean) at each layer
    - 扰动传播: ||delta||=||perturbed-clean|| at each layer
    """
    plog("=== Exp1: 精确定向扰动恢复 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)
    layers_list = get_layers(model)

    cat_list = ["fruit", "animal", "vehicle", "tool"]
    n_obj = min(4, len(obj_dict.get("fruit", [])))

    # 关键层选择
    if model_name == "qwen3":
        perturb_layer = 24
        monitor_layers = list(range(24, min(36, n_layers)))
    elif model_name == "glm4":
        perturb_layer = 24
        monitor_layers = list(range(24, min(40, n_layers)))
    else:
        perturb_layer = 24
        monitor_layers = list(range(24, min(28, n_layers)))

    perturb_beta = 5.0  # 增大扰动强度以确保效果可见

    # ---- Step 1: 收集每个类别在perturb_layer的均值残差 ----
    plog(f"  Step 1: Collecting mean residuals per category at L{perturb_layer}...")
    cat_mean_resids = {}
    cat_all_resids = {}

    for cat in cat_list:
        objs = obj_dict.get(cat, [])[:n_obj]
        resids = []
        for obj in objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = attention_mask.sum().item()
            pos = seq_len - 1

            captured = {}
            def make_hook(key):
                def hook_fn(module, inp, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook_fn

            h = layers_list[perturb_layer].register_forward_hook(make_hook("resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()

            if "resid" in captured:
                r = captured["resid"][0, pos].numpy()
                resids.append(r)

        if resids:
            cat_mean_resids[cat] = np.mean(resids, axis=0)
            cat_all_resids[cat] = resids
            plog(f"    {cat}: mean_resid_norm={np.linalg.norm(cat_mean_resids[cat]):.2f}, n={len(resids)}")

    # 全局均值
    all_resids = []
    for cat in cat_list:
        all_resids.extend(cat_all_resids.get(cat, []))
    global_mean = np.mean(all_resids, axis=0) if all_resids else np.zeros(info.d_model)

    # ---- Step 2: 构造精确扰动方向 ----
    plog(f"  Step 2: Constructing precise perturbation directions...")

    # fruit方向: fruit的残差相对于全局均值的偏移
    fruit_direction = cat_mean_resids.get("fruit", global_mean) - global_mean
    vehicle_direction = cat_mean_resids.get("vehicle", global_mean) - global_mean
    animal_direction = cat_mean_resids.get("animal", global_mean) - global_mean

    # 归一化
    def normalize(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-10 else v

    fruit_dir_norm = normalize(fruit_direction)
    vehicle_dir_norm = normalize(vehicle_direction)
    anti_fruit_dir = -fruit_dir_norm  # 推离fruit
    cross_cat_dir = vehicle_dir_norm  # 推向vehicle

    # 检查方向有效性
    cos_fruit_vehicle = float(np.dot(fruit_dir_norm, vehicle_dir_norm))
    plog(f"    cos(fruit_dir, vehicle_dir) = {cos_fruit_vehicle:.4f}")
    plog(f"    ||fruit_direction|| = {np.linalg.norm(fruit_direction):.2f}")
    plog(f"    ||vehicle_direction|| = {np.linalg.norm(vehicle_direction):.2f}")

    # ---- Step 3: 扰动恢复测试 ----
    perturb_types = {
        "random": None,  # 运行时生成随机向量
        "anti_fruit": anti_fruit_dir,
        "toward_vehicle": cross_cat_dir,
    }

    results = {}

    for pt_name, perturb_dir_np in perturb_types.items():
        plog(f"  Step 3: Testing perturbation type: {pt_name}")

        # 恢复数据
        recovery_per_layer = {f"L{li}": [] for li in monitor_layers}
        # 扰动传播数据 (||delta||)
        delta_norm_per_layer = {f"L{li}": [] for li in monitor_layers}
        # 扰动方向保持 (cos(delta[l], delta[perturb_layer]))
        delta_alignment_per_layer = {f"L{li}": [] for li in monitor_layers}

        for cat in cat_list:
            objs = obj_dict.get(cat, [])[:n_obj]
            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                seq_len = attention_mask.sum().item()
                pos = seq_len - 1

                # ---- 基线: 收集monitor layers的残差和DCF ----
                base_captured = {}
                base_hooks = []
                for li in monitor_layers:
                    def make_hook(key):
                        def hook_fn(module, inp, output):
                            if isinstance(output, tuple):
                                base_captured[key] = output[0].detach().float().cpu()
                            else:
                                base_captured[key] = output.detach().float().cpu()
                        return hook_fn
                    base_hooks.append(layers_list[li].register_forward_hook(make_hook(f"L{li}")))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                for h in base_hooks:
                    h.remove()

                # 基线DCF和残差
                base_dcf_per_layer = {}
                base_resid_per_layer = {}
                for li in monitor_layers:
                    key = f"L{li}"
                    if key in base_captured:
                        resid = base_captured[key][0, pos].numpy()
                        base_resid_per_layer[key] = resid
                        base_dcf_per_layer[key] = logit_lens_dcf(resid, W_U, tokenizer, FAMILY_WORDS_8D)

                # ---- 扰动: 在perturb_layer注入 ----
                if pt_name == "random":
                    perturb_vec = torch.randn(info.d_model) * perturb_beta
                else:
                    # 方向性扰动: 缩放到与随机扰动相同的期望范数
                    # random: E[||v||] = beta * sqrt(d_model)
                    # directional: ||v|| = target_norm
                    target_norm = perturb_beta * math.sqrt(info.d_model)
                    perturb_vec = torch.tensor(perturb_dir_np, dtype=torch.float32) * target_norm

                def make_perturb_hook(pvec):
                    def hook_fn(module, inp, output):
                        if isinstance(output, tuple):
                            perturbed = output[0].clone()
                            perturbed[0, pos, :] += pvec.to(perturbed.device).to(perturbed.dtype)
                            return (perturbed,) + output[1:]
                        else:
                            perturbed = output.clone()
                            perturbed[0, pos, :] += pvec.to(perturbed.device).to(perturbed.dtype)
                            return perturbed
                    return hook_fn

                perturbed_captured = {}
                perturb_hooks = []
                perturb_hooks.append(layers_list[perturb_layer].register_forward_hook(
                    make_perturb_hook(perturb_vec)))

                for li in monitor_layers:
                    def make_hook2(key):
                        def hook_fn(module, inp, output):
                            if isinstance(output, tuple):
                                perturbed_captured[key] = output[0].detach().float().cpu()
                            else:
                                perturbed_captured[key] = output.detach().float().cpu()
                        return hook_fn
                    perturb_hooks.append(layers_list[li].register_forward_hook(make_hook2(f"L{li}")))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                for h in perturb_hooks:
                    h.remove()

                # ---- 计算恢复和扰动传播 ----
                for li in monitor_layers:
                    key = f"L{li}"
                    if key in perturbed_captured and key in base_dcf_per_layer:
                        perturbed_resid = perturbed_captured[key][0, pos].numpy()
                        perturbed_dcf = logit_lens_dcf(perturbed_resid, W_U, tokenizer, FAMILY_WORDS_8D)

                        # DCF方向恢复
                        base_dir = base_dcf_per_layer[key]
                        pert_dir = perturbed_dcf
                        base_norm = base_dir - np.mean(base_dir)
                        pert_norm = pert_dir - np.mean(pert_dir)
                        bn = np.linalg.norm(base_norm)
                        pn = np.linalg.norm(pert_norm)
                        if bn > 1e-10 and pn > 1e-10:
                            cos_val = float(np.dot(base_norm, pert_norm) / (bn * pn))
                        else:
                            cos_val = 0.0
                        recovery_per_layer[key].append(cos_val)

                        # 扰动传播: ||delta|| and alignment
                        if key in base_resid_per_layer:
                            delta = perturbed_resid - base_resid_per_layer[key]
                            delta_norm = float(np.linalg.norm(delta))
                            delta_norm_per_layer[key].append(delta_norm)

        # ---- 汇总该扰动类型 ----
        mean_recovery = {}
        mean_delta_norm = {}
        for li in monitor_layers:
            key = f"L{li}"
            if recovery_per_layer[key]:
                mean_recovery[key] = round(float(np.mean(recovery_per_layer[key])), 4)
            if delta_norm_per_layer[key]:
                mean_delta_norm[key] = round(float(np.mean(delta_norm_per_layer[key])), 4)

        # Early vs Late
        n_mon = len(monitor_layers)
        early_layers = [f"L{li}" for li in monitor_layers[:n_mon//3]]
        late_layers = [f"L{li}" for li in monitor_layers[2*n_mon//3:]]

        early_rec = [mean_recovery[k] for k in early_layers if k in mean_recovery]
        late_rec = [mean_recovery[k] for k in late_layers if k in mean_recovery]

        early_delta = [mean_delta_norm[k] for k in early_layers if k in mean_delta_norm]
        late_delta = [mean_delta_norm[k] for k in late_layers if k in mean_delta_norm]

        # 扰动传播比率: late/early
        delta_ratio = float(np.mean(late_delta)) / max(float(np.mean(early_delta)), 1e-10) if early_delta and late_delta else 0

        results[pt_name] = {
            "recovery_per_layer": mean_recovery,
            "delta_norm_per_layer": mean_delta_norm,
            "early_mean_recovery": round(float(np.mean(early_rec)), 4) if early_rec else 0,
            "late_mean_recovery": round(float(np.mean(late_rec)), 4) if late_rec else 0,
            "early_mean_delta_norm": round(float(np.mean(early_delta)), 4) if early_delta else 0,
            "late_mean_delta_norm": round(float(np.mean(late_delta)), 4) if late_delta else 0,
            "delta_propagation_ratio": round(delta_ratio, 4),
            "delta_shrinks": delta_ratio < 0.8,
            "delta_grows": delta_ratio > 1.2,
            "attractor_likely": float(np.mean(late_rec)) > 0.7 if late_rec else False,
        }

        plog(f"    {pt_name}: early_rec={results[pt_name]['early_mean_recovery']}, "
             f"late_rec={results[pt_name]['late_mean_recovery']}")
        plog(f"    {pt_name}: early_delta={results[pt_name]['early_mean_delta_norm']}, "
             f"late_delta={results[pt_name]['late_mean_delta_norm']}, ratio={delta_ratio:.4f}")
        plog(f"    {pt_name}: delta_shrinks={results[pt_name]['delta_shrinks']}, "
             f"attractor_likely={results[pt_name]['attractor_likely']}")

    # ---- 汇总 ----
    n_attractor = sum(1 for v in results.values() if v.get("attractor_likely", False))
    n_shrinks = sum(1 for v in results.values() if v.get("delta_shrinks", False))

    summary = {
        "n_perturbation_types": len(perturb_types),
        "n_attractor_types": n_attractor,
        "n_delta_shrink_types": n_shrinks,
        "perturbation_beta": perturb_beta,
        "perturbation_is_directional": True,
        "attractor_hypothesis_supported": n_attractor >= 2,
        "delta_propagation_shrinks": n_shrinks >= 2,
    }
    results["summary"] = summary

    plog(f"  Attractor supported in {n_attractor}/{len(perturb_types)} types")
    plog(f"  Delta shrinks in {n_shrinks}/{len(perturb_types)} types")

    return results


# ==================== Hook辅助函数(避免闭包问题) ====================
def _make_capture_hook(store_dict, key):
    """创建一个将输出存入指定字典的hook, 避免闭包变量捕获问题"""
    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            store_dict[key] = output[0].detach().float().cpu()
        else:
            store_dict[key] = output.detach().float().cpu()
    return hook_fn


# ==================== Exp2: Qwen3吸引子纠正层定位 ====================
def exp2_attractor_correction_layers(model, tokenizer, model_name, device, obj_dict):
    """
    在L24注入扰动, 逐层关闭L25-L35, 看哪个层是"纠正层"
    
    如果关闭某层后恢复明显下降 → 该层是纠正层
    """
    plog("=== Exp2: Qwen3吸引子纠正层定位 ===")
    if model_name != "qwen3":
        plog(f"  Skipping (only for Qwen3)")
        return {"skipped": True, "reason": "only for qwen3"}

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)
    layers_list = get_layers(model)

    perturb_layer = 24
    final_layer = min(35, n_layers - 1)
    correction_candidates = list(range(25, final_layer + 1))

    cat_list = ["fruit", "animal", "vehicle", "tool"]
    n_obj = 4
    perturb_beta = 5.0

    # 先收集基线恢复(不关闭任何层)
    plog(f"  Computing baseline recovery (no ablation)...")

    # 收集fruit方向用于扰动
    fruit_resids = []
    for obj in obj_dict.get("fruit", [])[:n_obj]:
        prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        seq_len = attention_mask.sum().item()
        pos = seq_len - 1

        captured = {}
        h = layers_list[perturb_layer].register_forward_hook(_make_capture_hook(captured, "resid"))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h.remove()
        if "resid" in captured:
            fruit_resids.append(captured["resid"][0, pos].numpy())

    # 全局均值残差
    all_resids_for_mean = []
    for cat in cat_list:
        for obj in obj_dict.get(cat, [])[:n_obj]:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = attention_mask.sum().item()
            pos = seq_len - 1
            cap = {}
            h = layers_list[perturb_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap:
                all_resids_for_mean.append(cap["resid"][0, pos].numpy())

    global_mean = np.mean(all_resids_for_mean, axis=0) if all_resids_for_mean else np.zeros(info.d_model)
    fruit_direction = np.mean(fruit_resids, axis=0) - global_mean if fruit_resids else np.zeros(info.d_model)
    fd_norm = np.linalg.norm(fruit_direction)
    if fd_norm > 1e-10:
        fruit_direction = fruit_direction / fd_norm

    # 缩放扰动向量到与随机扰动相同的期望范数
    target_norm = perturb_beta * math.sqrt(info.d_model)
    perturb_vec = torch.tensor(fruit_direction, dtype=torch.float32) * target_norm

    # 基线恢复
    base_recoveries = []
    for cat in cat_list:
        objs = obj_dict.get(cat, [])[:n_obj]
        for obj in objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = attention_mask.sum().item()
            pos = seq_len - 1

            # Clean baseline
            clean_cap = {}
            h_clean = layers_list[final_layer].register_forward_hook(_make_capture_hook(clean_cap, "L_final"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_clean.remove()

            if "L_final" not in clean_cap:
                continue
            clean_resid = clean_cap["L_final"][0, pos].numpy()
            clean_dcf = logit_lens_dcf(clean_resid, W_U, tokenizer, FAMILY_WORDS_8D)

            # Perturbed (without ablation)
            def make_perturb_hook(pvec, p):
                def hook_fn(module, inp, output):
                    if isinstance(output, tuple):
                        perturbed = output[0].clone()
                        perturbed[0, p, :] += pvec.to(perturbed.device).to(perturbed.dtype)
                        return (perturbed,) + output[1:]
                    else:
                        perturbed = output.clone()
                        perturbed[0, p, :] += pvec.to(perturbed.device).to(perturbed.dtype)
                        return perturbed
                return hook_fn

            pert_cap = {}
            h_p1 = layers_list[perturb_layer].register_forward_hook(make_perturb_hook(perturb_vec, pos))
            h_p2 = layers_list[final_layer].register_forward_hook(_make_capture_hook(pert_cap, "L_final_p"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_p1.remove()
            h_p2.remove()

            if "L_final_p" in pert_cap:
                p_resid = pert_cap["L_final_p"][0, pos].numpy()
                p_dcf = logit_lens_dcf(p_resid, W_U, tokenizer, FAMILY_WORDS_8D)

                base_n = clean_dcf - np.mean(clean_dcf)
                pert_n = p_dcf - np.mean(p_dcf)
                bn = np.linalg.norm(base_n)
                pn = np.linalg.norm(pert_n)
                if bn > 1e-10 and pn > 1e-10:
                    cos_val = float(np.dot(base_n, pert_n) / (bn * pn))
                else:
                    cos_val = 0.0
                base_recoveries.append(cos_val)

    baseline_recovery = float(np.mean(base_recoveries)) if base_recoveries else 0
    plog(f"  Baseline recovery at L{final_layer}: {baseline_recovery:.4f}")

    # ---- 逐层关闭测试 ----
    results = {"baseline_recovery": round(baseline_recovery, 4)}

    for corr_layer in correction_candidates:
        plog(f"  Testing ablation of L{corr_layer}...")
        ablated_recoveries = []

        for cat in cat_list:
            objs = obj_dict.get(cat, [])[:n_obj]
            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                seq_len = attention_mask.sum().item()
                pos = seq_len - 1

                # Perturbed + ablate corr_layer
                hooks = []

                # Perturbation at L24
                def make_perturb_hook2(pvec, p):
                    def hook_fn(module, inp, output):
                        if isinstance(output, tuple):
                            perturbed = output[0].clone()
                            perturbed[0, p, :] += pvec.to(perturbed.device).to(perturbed.dtype)
                            return (perturbed,) + output[1:]
                        else:
                            perturbed = output.clone()
                            perturbed[0, p, :] += pvec.to(perturbed.device).to(perturbed.dtype)
                            return perturbed
                    return hook_fn

                hooks.append(layers_list[perturb_layer].register_forward_hook(make_perturb_hook2(perturb_vec, pos)))

                # Ablate corr_layer (zero out both MLP and Attn)
                def zero_hook(module, inp, output):
                    if isinstance(output, tuple):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    return torch.zeros_like(output)

                hooks.append(layers_list[corr_layer].register_forward_hook(zero_hook))

                # Monitor final layer
                ablated_cap = {}
                hooks.append(layers_list[final_layer].register_forward_hook(_make_capture_hook(ablated_cap, "L_final_a")))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                for h in hooks:
                    h.remove()

                if "L_final_a" in ablated_cap:
                    a_resid = ablated_cap["L_final_a"][0, pos].numpy()
                    a_dcf = logit_lens_dcf(a_resid, W_U, tokenizer, FAMILY_WORDS_8D)

                    # Clean baseline DCF (重新计算)
                    clean_cap2 = {}
                    h_c = layers_list[final_layer].register_forward_hook(_make_capture_hook(clean_cap2, "L_final_c"))
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    h_c.remove()

                    if "L_final_c" in clean_cap2:
                        c_resid = clean_cap2["L_final_c"][0, pos].numpy()
                        c_dcf = logit_lens_dcf(c_resid, W_U, tokenizer, FAMILY_WORDS_8D)

                        c_n = c_dcf - np.mean(c_dcf)
                        a_n = a_dcf - np.mean(a_dcf)
                        cn = np.linalg.norm(c_n)
                        an = np.linalg.norm(a_n)
                        if cn > 1e-10 and an > 1e-10:
                            cos_val = float(np.dot(c_n, a_n) / (cn * an))
                        else:
                            cos_val = 0.0
                        ablated_recoveries.append(cos_val)

        ablated_recovery = float(np.mean(ablated_recoveries)) if ablated_recoveries else 0
        recovery_drop = baseline_recovery - ablated_recovery

        results[f"L{corr_layer}"] = {
            "recovery_with_ablation": round(ablated_recovery, 4),
            "recovery_drop_from_baseline": round(recovery_drop, 4),
            "is_correction_layer": recovery_drop > 0.1,
        }

        plog(f"    L{corr_layer}: recovery={ablated_recovery:.4f}, drop={recovery_drop:.4f}, "
             f"correction={'YES' if recovery_drop > 0.1 else 'no'}")

    # ---- 找到关键纠正层 ----
    correction_layers = [k for k, v in results.items() if isinstance(v, dict) and v.get("is_correction_layer", False)]
    results["summary"] = {
        "baseline_recovery": round(baseline_recovery, 4),
        "correction_layers": correction_layers,
        "n_correction_layers": len(correction_layers),
    }

    plog(f"  Correction layers: {correction_layers}")

    return results


# ==================== Exp3: DS7B L27排除验证 ====================
def exp3_ds7b_l27_exclusion(model, tokenizer, model_name, device, obj_dict):
    """
    分别监控L24-L26(不含L27)的恢复, 判断吸引子是否在L27之前就存在
    
    同时测试: 关闭L27 MLP/Attn后L24-L26的恢复
    """
    plog("=== Exp3: DS7B L27排除验证 ===")
    if model_name != "deepseek7b":
        plog(f"  Skipping (only for DS7B)")
        return {"skipped": True, "reason": "only for deepseek7b"}

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)
    layers_list = get_layers(model)

    perturb_layer = 24
    monitor_layers = [24, 25, 26, 27]  # L24-L27
    cat_list = ["fruit", "animal", "vehicle", "tool"]
    n_obj = 4
    perturb_beta = 5.0

    # 收集fruit方向
    fruit_resids = []
    all_resids_for_mean = []

    for cat in cat_list:
        objs = obj_dict.get(cat, [])[:n_obj]
        for obj in objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = attention_mask.sum().item()
            pos = seq_len - 1

            captured = {}
            def make_hook(key):
                def hook_fn(module, inp, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu()
                    else:
                        captured[key] = output.detach().float().cpu()
                return hook_fn

            h = layers_list[perturb_layer].register_forward_hook(make_hook("resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()

            if "resid" in captured:
                r = captured["resid"][0, pos].numpy()
                all_resids_for_mean.append(r)
                if cat == "fruit":
                    fruit_resids.append(r)

    global_mean = np.mean(all_resids_for_mean, axis=0) if all_resids_for_mean else np.zeros(info.d_model)
    fruit_direction = np.mean(fruit_resids, axis=0) - global_mean if fruit_resids else np.zeros(info.d_model)
    fd_norm = np.linalg.norm(fruit_direction)
    if fd_norm > 1e-10:
        fruit_direction = fruit_direction / fd_norm

    target_norm = perturb_beta * math.sqrt(info.d_model)
    perturb_vec = torch.tensor(fruit_direction, dtype=torch.float32) * target_norm

    # ---- 三种条件: normal, close L27 MLP, close L27 Attn ----
    conditions = ["normal", "no_l27_mlp", "no_l27_attn"]
    results = {}

    for cond in conditions:
        plog(f"  Testing condition: {cond}")

        recovery_per_layer = {f"L{li}": [] for li in monitor_layers}
        delta_norm_per_layer = {f"L{li}": [] for li in monitor_layers}

        for cat in cat_list:
            objs = obj_dict.get(cat, [])[:n_obj]
            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                seq_len = attention_mask.sum().item()
                pos = seq_len - 1

                # ---- Clean baseline ----
                clean_captured = {}
                base_hooks = []
                for li in monitor_layers:
                    def make_hook(key):
                        def hook_fn(module, inp, output):
                            if isinstance(output, tuple):
                                clean_captured[key] = output[0].detach().float().cpu()
                            else:
                                clean_captured[key] = output.detach().float().cpu()
                        return hook_fn
                    base_hooks.append(layers_list[li].register_forward_hook(make_hook(f"L{li}")))

                # 条件性关闭
                cond_hooks = []
                if cond == "no_l27_mlp":
                    def zero_hook(module, inp, output):
                        if isinstance(output, tuple):
                            return (torch.zeros_like(output[0]),) + output[1:]
                        return torch.zeros_like(output)
                    cond_hooks.append(layers_list[27].mlp.register_forward_hook(zero_hook))
                elif cond == "no_l27_attn":
                    def zero_hook(module, inp, output):
                        if isinstance(output, tuple):
                            return (torch.zeros_like(output[0]),) + output[1:]
                        return torch.zeros_like(output)
                    cond_hooks.append(layers_list[27].self_attn.register_forward_hook(zero_hook))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                for h in base_hooks:
                    h.remove()
                for h in cond_hooks:
                    h.remove()

                clean_dcf_per_layer = {}
                clean_resid_per_layer = {}
                for li in monitor_layers:
                    key = f"L{li}"
                    if key in clean_captured:
                        resid = clean_captured[key][0, pos].numpy()
                        clean_resid_per_layer[key] = resid
                        clean_dcf_per_layer[key] = logit_lens_dcf(resid, W_U, tokenizer, FAMILY_WORDS_8D)

                # ---- Perturbed ----
                perturbed_captured = {}
                perturb_hooks = []

                def make_perturb_hook(pvec):
                    def hook_fn(module, inp, output):
                        if isinstance(output, tuple):
                            perturbed = output[0].clone()
                            perturbed[0, pos, :] += pvec.to(perturbed.device).to(perturbed.dtype)
                            return (perturbed,) + output[1:]
                        else:
                            perturbed = output.clone()
                            perturbed[0, pos, :] += pvec.to(perturbed.device).to(perturbed.dtype)
                            return perturbed
                    return hook_fn

                perturb_hooks.append(layers_list[perturb_layer].register_forward_hook(
                    make_perturb_hook(perturb_vec)))

                for li in monitor_layers:
                    def make_hook2(key):
                        def hook_fn(module, inp, output):
                            if isinstance(output, tuple):
                                perturbed_captured[key] = output[0].detach().float().cpu()
                            else:
                                perturbed_captured[key] = output.detach().float().cpu()
                        return hook_fn
                    perturb_hooks.append(layers_list[li].register_forward_hook(make_hook2(f"L{li}")))

                # 条件性关闭
                cond_hooks2 = []
                if cond == "no_l27_mlp":
                    def zero_hook(module, inp, output):
                        if isinstance(output, tuple):
                            return (torch.zeros_like(output[0]),) + output[1:]
                        return torch.zeros_like(output)
                    cond_hooks2.append(layers_list[27].mlp.register_forward_hook(zero_hook))
                elif cond == "no_l27_attn":
                    def zero_hook(module, inp, output):
                        if isinstance(output, tuple):
                            return (torch.zeros_like(output[0]),) + output[1:]
                        return torch.zeros_like(output)
                    cond_hooks2.append(layers_list[27].self_attn.register_forward_hook(zero_hook))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                for h in perturb_hooks:
                    h.remove()
                for h in cond_hooks2:
                    h.remove()

                # ---- 计算恢复和扰动传播 ----
                for li in monitor_layers:
                    key = f"L{li}"
                    if key in perturbed_captured and key in clean_dcf_per_layer:
                        p_resid = perturbed_captured[key][0, pos].numpy()
                        p_dcf = logit_lens_dcf(p_resid, W_U, tokenizer, FAMILY_WORDS_8D)

                        # Recovery
                        base_dir = clean_dcf_per_layer[key]
                        pert_dir = p_dcf
                        bn_vec = base_dir - np.mean(base_dir)
                        pn_vec = pert_dir - np.mean(pert_dir)
                        bn = np.linalg.norm(bn_vec)
                        pn = np.linalg.norm(pn_vec)
                        if bn > 1e-10 and pn > 1e-10:
                            cos_val = float(np.dot(bn_vec, pn_vec) / (bn * pn))
                        else:
                            cos_val = 0.0
                        recovery_per_layer[key].append(cos_val)

                        # Delta norm
                        if key in clean_resid_per_layer:
                            delta = p_resid - clean_resid_per_layer[key]
                            delta_norm_per_layer[key].append(float(np.linalg.norm(delta)))

        # 汇总
        mean_recovery = {k: round(float(np.mean(v)), 4) for k, v in recovery_per_layer.items() if v}
        mean_delta = {k: round(float(np.mean(v)), 4) for k, v in delta_norm_per_layer.items() if v}

        results[cond] = {
            "recovery_per_layer": mean_recovery,
            "delta_norm_per_layer": mean_delta,
        }

        plog(f"    {cond}: recovery={mean_recovery}")
        plog(f"    {cond}: delta_norm={mean_delta}")

    # ---- 判断: 吸引子是否在L27之前就存在 ----
    normal_l26_recovery = results.get("normal", {}).get("recovery_per_layer", {}).get("L26", 0)
    normal_l27_recovery = results.get("normal", {}).get("recovery_per_layer", {}).get("L27", 0)
    normal_l24_recovery = results.get("normal", {}).get("recovery_per_layer", {}).get("L24", 0)

    # L24-L26恢复高 → 吸引子在L27之前就存在
    pre_l27_attractor = normal_l26_recovery > 0.8

    # L27恢复比L26低 → L27在破坏恢复
    l27_degrades = normal_l27_recovery < normal_l26_recovery

    results["summary"] = {
        "normal_L24_recovery": normal_l24_recovery,
        "normal_L26_recovery": normal_l26_recovery,
        "normal_L27_recovery": normal_l27_recovery,
        "pre_L27_attractor_exists": pre_l27_attractor,
        "L27_degrades_recovery": l27_degrades,
        "conclusion": "attractor exists before L27" if pre_l27_attractor else "attractor only through L27 (format overlay)",
    }

    plog(f"  Pre-L27 attractor: {pre_l27_attractor}")
    plog(f"  L27 degrades recovery: {l27_degrades}")

    return results


# ==================== Exp4: 格式/语义子空间投影 ====================
def exp4_format_semantic_subspace(model, tokenizer, model_name, device, obj_dict):
    """
    对每层输出计算:
    - format_score: 在格式token上的平均logit
    - semantic_score: 在语义词上的平均logit
    - ratio: format/semantic
    
    这给出每层"格式倾向"vs"语义倾向"的客观测量
    """
    plog("=== Exp4: 格式/语义子空间投影 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)
    layers_list = get_layers(model)

    # 收集格式和语义词的token ID
    format_ids = []
    for tok in FORMAT_TOKENS:
        tid = find_token_id(tokenizer, tok)
        if tid is not None and tid < info.vocab_size:
            format_ids.append(tid)

    semantic_ids = []
    for tok in SEMANTIC_TOKENS:
        tid = find_token_id(tokenizer, tok)
        if tid is not None and tid < info.vocab_size:
            semantic_ids.append(tid)

    plog(f"  Format tokens found: {len(format_ids)}")
    plog(f"  Semantic tokens found: {len(semantic_ids)}")

    if not format_ids or not semantic_ids:
        plog(f"  WARNING: Insufficient tokens for subspace analysis")
        return {"error": "insufficient tokens"}

    sample_layers = get_sample_layers(n_layers, n_samples=12)
    cat_list = ["fruit", "animal", "vehicle", "tool"]
    n_obj = 4

    # 收集每层的format_score和semantic_score
    format_scores = {f"L{li}": [] for li in sample_layers}
    semantic_scores = {f"L{li}": [] for li in sample_layers}

    for cat in cat_list:
        objs = obj_dict.get(cat, [])[:n_obj]
        for obj in objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = attention_mask.sum().item()
            pos = seq_len - 1

            captured = {}
            hooks = []
            for li in sample_layers:
                def make_hook(key):
                    def hook_fn(module, inp, output):
                        if isinstance(output, tuple):
                            captured[key] = output[0].detach().float().cpu()
                        else:
                            captured[key] = output.detach().float().cpu()
                    return hook_fn
                hooks.append(layers_list[li].register_forward_hook(make_hook(f"L{li}")))

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            for h in hooks:
                h.remove()

            for li in sample_layers:
                key = f"L{li}"
                if key in captured:
                    resid = captured[key][0, pos].numpy()
                    logits = resid @ W_U.T

                    fmt_score = float(np.mean(logits[format_ids])) if format_ids else 0
                    sem_score = float(np.mean(logits[semantic_ids])) if semantic_ids else 0

                    format_scores[key].append(fmt_score)
                    semantic_scores[key].append(sem_score)

    # 汇总
    results = {}
    for li in sample_layers:
        key = f"L{li}"
        fmt_mean = float(np.mean(format_scores[key])) if format_scores[key] else 0
        sem_mean = float(np.mean(semantic_scores[key])) if semantic_scores[key] else 0
        ratio = fmt_mean / max(abs(sem_mean), 1e-10)

        results[key] = {
            "format_score": round(fmt_mean, 4),
            "semantic_score": round(sem_mean, 4),
            "format_semantic_ratio": round(ratio, 4),
        }

    # 找格式倾向最强的层
    max_ratio_layer = max(results.keys(), key=lambda k: results[k]["format_semantic_ratio"])
    max_semantic_layer = max(results.keys(), key=lambda k: results[k]["semantic_score"])

    # 检查最后一层是否格式倾向激增
    last_layer_key = f"L{sample_layers[-1]}"
    second_last_key = f"L{sample_layers[-2]}" if len(sample_layers) >= 2 else None
    last_format_spike = False
    if second_last_key and second_last_key in results and last_layer_key in results:
        ratio_change = results[last_layer_key]["format_semantic_ratio"] - results[second_last_key]["format_semantic_ratio"]
        last_format_spike = ratio_change > 0.5

    results["summary"] = {
        "max_format_ratio_layer": max_ratio_layer,
        "max_format_ratio": results[max_ratio_layer]["format_semantic_ratio"],
        "max_semantic_score_layer": max_semantic_layer,
        "max_semantic_score": results[max_semantic_layer]["semantic_score"],
        "last_layer_format_spike": last_format_spike,
        "n_format_tokens": len(format_ids),
        "n_semantic_tokens": len(semantic_ids),
    }

    plog(f"  Max format ratio at {max_ratio_layer}: {results[max_ratio_layer]['format_semantic_ratio']:.4f}")
    plog(f"  Max semantic score at {max_semantic_layer}: {results[max_semantic_layer]['semantic_score']:.4f}")
    plog(f"  Last layer format spike: {last_format_spike}")

    return results


# ==================== Exp5: 神经元级DCF写入贡献 (Qwen3) ====================
def exp5_neuron_dcf_contribution(model, tokenizer, model_name, device, obj_dict):
    """
    对Qwen3关键层计算每个神经元对DCF的写入贡献
    
    方法:
    1. Hook MLP的down_proj输入, 获取MLP中间激活向量
    2. 计算DCF梯度方向: grad_dcf = W_U^T @ indicator_family_words
    3. 对每个神经元i: contribution_i = activation_i * dot(W_down[i,:], grad_dcf)
    4. 找top-k约束写入神经元
    """
    plog("=== Exp5: 神经元级DCF写入贡献 ===")
    if model_name != "qwen3":
        plog(f"  Skipping (only for Qwen3)")
        return {"skipped": True, "reason": "only for qwen3"}

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)
    layers_list = get_layers(model)

    # 关键层
    key_layers = [24, 30, 33, 35] if n_layers >= 36 else [n_layers//3, n_layers//2, 2*n_layers//3, n_layers-1]

    cat_list = ["fruit", "animal", "vehicle", "tool"]
    n_obj = 4

    # 计算DCF梯度方向 (每个类别)
    plog(f"  Computing DCF gradient directions...")
    dcf_gradients = {}

    for cat in cat_list:
        family_words = FAMILY_WORDS_8D.get(cat, [])
        # indicator vector in logit space
        indicator = np.zeros(info.vocab_size)
        count = 0
        for w in family_words:
            tid = find_token_id(tokenizer, w)
            if tid is not None and tid < info.vocab_size:
                indicator[tid] = 1.0
                count += 1
        if count > 0:
            indicator /= count
        # DCF gradient in residual space: grad = W_U^T @ indicator
        grad = W_U.T @ indicator  # [d_model]
        # 归一化
        gn = np.linalg.norm(grad)
        if gn > 1e-10:
            grad = grad / gn
        dcf_gradients[cat] = grad
        plog(f"    {cat}: dcf_gradient norm={gn:.4f}, n_family_words={count}")

    results = {}

    for li in key_layers:
        plog(f"  Analyzing L{li} neuron-level DCF contribution...")
        layer = layers_list[li]
        lw = get_layer_weights(layer, info.d_model, info.mlp_type)
        W_down = lw.W_down  # [d_model, intermediate_size]

        if W_down is None:
            plog(f"    L{li}: W_down not available, skipping")
            continue

        # 对每个类别收集MLP中间激活
        for cat in cat_list:
            objs = obj_dict.get(cat, [])[:n_obj]

            all_activations = []  # [n_samples, intermediate_size]
            all_cat_labels = []

            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                seq_len = attention_mask.sum().item()
                pos = seq_len - 1

                # Hook MLP down_proj的输入(即MLP中间激活)
                captured_activation = {}

                def activation_hook(module, inp, output):
                    if isinstance(inp, tuple) and len(inp) > 0:
                        captured_activation["mlp_mid"] = inp[0].detach().float().cpu()

                h_act = layer.mlp.down_proj.register_forward_hook(activation_hook)

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                h_act.remove()

                if "mlp_mid" in captured_activation:
                    act = captured_activation["mlp_mid"][0, pos].numpy()  # [intermediate_size]
                    all_activations.append(act)
                    all_cat_labels.append(cat)

            if not all_activations:
                continue

            activations = np.array(all_activations)  # [n_samples, intermediate_size]
            mean_activation = np.mean(activations, axis=0)  # [intermediate_size]

            # 计算每个神经元的DCF贡献
            # contribution_i = activation_i * dot(W_down[:,i], grad_dcf)
            grad_dcf = dcf_gradients.get(cat, np.zeros(info.d_model))
            # W_down shape: [d_model, intermediate_size]
            # dot(W_down[:,i], grad_dcf) = W_down.T[i,:] @ grad_dcf
            neuron_write_direction = W_down.T @ grad_dcf  # [intermediate_size]
            neuron_contribution = mean_activation * neuron_write_direction  # [intermediate_size]

            # Top-k神经元
            top_k = 20
            top_indices = np.argsort(np.abs(neuron_contribution))[-top_k:][::-1]

            key = f"L{li}_{cat}"
            results[key] = {
                "top_neuron_indices": [int(i) for i in top_indices],
                "top_neuron_contributions": [round(float(neuron_contribution[i]), 6) for i in top_indices],
                "top_neuron_activations": [round(float(mean_activation[i]), 6) for i in top_indices],
                "top_neuron_write_alignment": [round(float(neuron_write_direction[i]), 6) for i in top_indices],
                "mean_abs_contribution": round(float(np.mean(np.abs(neuron_contribution))), 6),
                "max_abs_contribution": round(float(np.max(np.abs(neuron_contribution))), 6),
                "n_positive_contributors": int(np.sum(neuron_contribution > 0)),
                "n_negative_contributors": int(np.sum(neuron_contribution < 0)),
            }

            plog(f"    L{li} {cat}: max_contribution={results[key]['max_abs_contribution']:.6f}, "
                 f"positive={results[key]['n_positive_contributors']}, "
                 f"negative={results[key]['n_negative_contributors']}")

    # ---- 交叉验证: 同一神经元是否对不同类别有不同贡献 ----
    plog(f"  Cross-category neuron specificity...")

    # 只对有结果的层做交叉验证
    for li in key_layers:
        cat_results = {}
        for cat in cat_list:
            key = f"L{li}_{cat}"
            if key in results:
                cat_results[cat] = results[key]

        if len(cat_results) < 2:
            continue

        # 检查top-20神经元的重叠
        top_sets = {}
        for cat, r in cat_results.items():
            top_sets[cat] = set(r["top_neuron_indices"][:20])

        # Jaccard similarity between categories
        cat_pairs = []
        cat_list_found = list(top_sets.keys())
        for i in range(len(cat_list_found)):
            for j in range(i+1, len(cat_list_found)):
                c1, c2 = cat_list_found[i], cat_list_found[j]
                intersection = len(top_sets[c1] & top_sets[c2])
                union = len(top_sets[c1] | top_sets[c2])
                jaccard = intersection / max(union, 1)
                cat_pairs.append({
                    "pair": f"{c1}_vs_{c2}",
                    "intersection": intersection,
                    "union": union,
                    "jaccard": round(jaccard, 4),
                })

        results[f"L{li}_neuron_specificity"] = {
            "category_pairs": cat_pairs,
            "neurons_are_category_specific": all(p["jaccard"] < 0.5 for p in cat_pairs),
        }

    results["summary"] = {
        "model": model_name,
        "key_layers": key_layers,
        "categories_analyzed": cat_list,
        "method": "activation * W_down^T @ grad_dcf",
    }

    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        return

    obj_dict = ROUNDS.get(round_num, ROUNDS[1])

    plog(f"Phase 474: Attractor Mechanism Decomposition — {model_name}, Round {round_num}")
    plog(f"Core: Precise directional perturbation, propagation tracking, format/semantic subspace")

    # ---- 1. 加载模型 ----
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    t_load = time.time() - t0
    plog(f"Model loaded in {t_load:.0f}s")

    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    # ---- 2. 运行实验 ----
    all_results = {
        "phase": 474,
        "model": model_name,
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "theory": "Attractor Mechanism Decomposition & Neuron-Level Writers",
        "core_question": "Is Phase3 an attractor or repeated writes? Which layers correct perturbation?",
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }

    # Exp1: 精确定向扰动恢复 (所有模型)
    t1 = time.time()
    all_results["exp1_precise_directional_perturbation"] = exp1_precise_directional_perturbation(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp1 done in {time.time()-t1:.0f}s")

    # Exp2: Qwen3吸引子纠正层定位 (仅Qwen3)
    t2 = time.time()
    all_results["exp2_attractor_correction_layers"] = exp2_attractor_correction_layers(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp2 done in {time.time()-t2:.0f}s")

    # Exp3: DS7B L27排除验证 (仅DS7B)
    t3 = time.time()
    all_results["exp3_ds7b_l27_exclusion"] = exp3_ds7b_l27_exclusion(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp3 done in {time.time()-t3:.0f}s")

    # Exp4: 格式/语义子空间投影 (所有模型)
    t4 = time.time()
    all_results["exp4_format_semantic_subspace"] = exp4_format_semantic_subspace(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp4 done in {time.time()-t4:.0f}s")

    # Exp5: 神经元级DCF写入贡献 (仅Qwen3)
    t5 = time.time()
    all_results["exp5_neuron_dcf_contribution"] = exp5_neuron_dcf_contribution(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp5 done in {time.time()-t5:.0f}s")

    # ---- 3. 保存结果 ----
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase474_{model_name}_r{round_num}.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, (int, float, str)):
            return obj
        return str(obj)

    all_results = convert(all_results)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    plog(f"Results saved to {out_path}")

    # ---- 4. 释放模型 ----
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    total_time = time.time() - t0
    plog(f"Phase 474 {model_name} Round {round_num} complete in {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
