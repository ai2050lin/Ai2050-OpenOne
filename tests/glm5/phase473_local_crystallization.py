"""
Phase 473: 局部约束结晶分解、多层冗余验证与吸引子测试
====================================================
核心目标: 直接破解Phase2→3转变 — Qwen3 L24的DCF可读性跳升由什么组件驱动?

基于Phase 471-472的分析修正:
1. LL-DCF是"可读性"度量, 不是"写入性"度量 — 但仍然是最重要的信号
2. Exp1应测L24本层LL-DCF, 而非最终DCF — 这是Phase 472的关键缺陷
3. 单层关闭不能区分"无关"和"冗余" — 需要多层联合关闭
4. "结晶=不动点"只是猜想 — 需要扰动恢复测试验证
5. DS7B L27需要top-k token分析 — 解释"反语义写入"

实验设计:
Exp1: 局部LL-DCF分解 — L23→L24跳升来自Attention还是MLP?
  在L24分别关闭attn/mlp, 看L24本层的LL-DCF是否跳升
  对比: h_23 + attn_24 vs h_23 + mlp_24 vs h_24(full) 的LL-DCF

Exp2: 多层联合关闭 — 区分冗余vs无关
  同时关闭L22-L24的MLP/Attn, 看最终DCF是否大降
  如果单层关闭无效但多层关闭有效 → 冗余编码
  如果多层关闭也无效 → 这些层确实不重要

Exp3: 全类别对方向稳定性 (C(6,2)=15对)
  扩展Phase 472的4对到全部15对, 验证结晶的泛化性

Exp4: Phase3扰动恢复测试 (吸引子验证)
  在Phase3首层注入扰动, 观察后续层是否恢复原方向
  如果恢复 → 吸引子; 如果不恢复 → 简单重复写入

Exp5: DS7B L27 top-k token分析
  将L27 MLP/Attn输出投影到logit空间, 看top-k tokens
  判断是写入数学/推理模式还是其他

模型加载: bfloat16 + device_map="auto" + flash_attention_2

用法:
  python tests/glm5/phase473_local_crystallization.py qwen3 1
  python tests/glm5/phase473_local_crystallization.py glm4 1
  python tests/glm5/phase473_local_crystallization.py deepseek7b 1
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
    1: {k: v[:6] for k, v in CATEGORIES.items()},
    2: {k: v for k, v in CATEGORIES.items()},  # R2: 全量
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


def get_norm_fn(model, model_name, layer_idx):
    """获取指定层的RMSNorm/LayerNorm权重, 用于归一化残差"""
    layers_list = get_layers(model)
    layer = layers_list[layer_idx]
    # 大多数模型在attention前有input_layernorm, 在MLP前有post_attention_layernorm
    # 最终输出通常有model.model.norm
    # 对于logit lens, 我们需要的是每层输出后的norm
    return None  # 简化: 先不归一化, 看原始结果


# ==================== Exp1: 局部LL-DCF分解 ====================
def exp1_local_ll_dcf_decomposition(model, tokenizer, model_name, device, obj_dict):
    """
    在L24(Qwen3)/L24(GLM4)/L27(DS7B)分别关闭attn和mlp,
    看L24本层的LL-DCF是否跳升。
    
    关键改进: 测的是L24本层的LL-DCF, 不是最终DCF!
    
    方法:
    1. 基线: 正常前向, 收集L23和L24的残差, 计算LL-DCF
    2. 关闭L24 MLP: L24残差 = h_23 + attn_24 (无mlp_24)
    3. 关闭L24 Attention: L24残差 = h_23 + mlp_24 (无attn_24)
    4. 分别计算L24的LL-DCF → 判断跳升来源
    """
    plog("=== Exp1: 局部LL-DCF分解 — L24跳升来自Attn还是MLP ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)

    # 关键层选择: Phase 471中LL-DCF跳升的层及其前后
    if model_name == "qwen3":
        # L24跳升 → 测试L20-L28
        key_layers = list(range(20, min(29, n_layers)))
    elif model_name == "glm4":
        # L24-L30 → 测试L22-L32
        key_layers = list(range(22, min(33, n_layers)))
    else:
        # DS7B L27 → 测试L24-L27
        key_layers = list(range(24, min(28, n_layers)))

    cat_list = ["fruit", "animal", "vehicle", "tool", "furniture", "clothing"]
    n_obj = 6

    results = {}

    for li in key_layers:
        plog(f"  Testing L{li} local LL-DCF decomposition...")
        layers_list = get_layers(model)

        # ---- 为每个条件收集L{li}的LL-DCF ----
        all_base_dcf = []   # 正常残差
        all_no_mlp_dcf = [] # 关闭L{li} MLP后的残差
        all_no_attn_dcf = []# 关闭L{li} Attn后的残差
        all_labels = []

        for cat in cat_list:
            objs = obj_dict.get(cat, [])[:n_obj]
            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                seq_len = attention_mask.sum().item()
                pos = seq_len - 1

                # ---- 1) 基线: 正常前向, Hook L{li}输出 ----
                captured = {}
                def make_capture_hook(key):
                    def hook_fn(module, inp, output):
                        if isinstance(output, tuple):
                            captured[key] = output[0].detach().float().cpu()
                        else:
                            captured[key] = output.detach().float().cpu()
                    return hook_fn

                h1 = layers_list[li].register_forward_hook(make_capture_hook("base_resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h1.remove()

                if "base_resid" in captured:
                    resid = captured["base_resid"][0, pos].numpy()
                    dcf = logit_lens_dcf(resid, W_U, tokenizer, FAMILY_WORDS_8D)
                    all_base_dcf.append(dcf)
                    all_labels.append(cat)

                # ---- 2) 关闭L{li} MLP: Hook L{li}输出 ----
                def mlp_zero_hook(module, input, output):
                    if isinstance(output, tuple):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    return torch.zeros_like(output)

                captured2 = {}
                h2a = layers_list[li].mlp.register_forward_hook(mlp_zero_hook)
                h2b = layers_list[li].register_forward_hook(make_capture_hook("no_mlp_resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h2a.remove()
                h2b.remove()

                if "no_mlp_resid" in captured2:
                    resid = captured2["no_mlp_resid"][0, pos].numpy()
                    dcf = logit_lens_dcf(resid, W_U, tokenizer, FAMILY_WORDS_8D)
                    all_no_mlp_dcf.append(dcf)

                # ---- 3) 关闭L{li} Attention: Hook L{li}输出 ----
                def attn_zero_hook(module, input, output):
                    if isinstance(output, tuple):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    return torch.zeros_like(output)

                captured3 = {}
                h3a = layers_list[li].self_attn.register_forward_hook(attn_zero_hook)
                h3b = layers_list[li].register_forward_hook(make_capture_hook("no_attn_resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h3a.remove()
                h3b.remove()

                if "no_attn_resid" in captured3:
                    resid = captured3["no_attn_resid"][0, pos].numpy()
                    dcf = logit_lens_dcf(resid, W_U, tokenizer, FAMILY_WORDS_8D)
                    all_no_attn_dcf.append(dcf)

                plog(f"    L{li} {cat}/{obj}: base_resid captured={('base_resid' in captured)}")

        # 计算聚类质量 — 注意: 这是L{li}本层的LL-DCF, 不是最终DCF!
        base_sil = cluster_quality(all_base_dcf, all_labels) if all_base_dcf else 0
        no_mlp_sil = cluster_quality(all_no_mlp_dcf, all_labels) if all_no_mlp_dcf else 0
        no_attn_sil = cluster_quality(all_no_attn_dcf, all_labels) if all_no_attn_dcf else 0

        layer_result = {
            "base_ll_dcf_silhouette": round(base_sil, 4),
            "no_mlp_ll_dcf_silhouette": round(no_mlp_sil, 4),
            "no_attn_ll_dcf_silhouette": round(no_attn_sil, 4),
            "mlp_sil_drop": round(base_sil - no_mlp_sil, 4),
            "attn_sil_drop": round(base_sil - no_attn_sil, 4),
            "mlp_drives_local_readability": (base_sil - no_mlp_sil) > (base_sil - no_attn_sil),
        }

        results[f"L{li}"] = layer_result
        plog(f"    L{li}: base_ll_dcf={base_sil:.4f}, no_mlp={no_mlp_sil:.4f}, "
             f"no_attn={no_attn_sil:.4f}, mlp_drop={layer_result['mlp_sil_drop']:.4f}, "
             f"attn_drop={layer_result['attn_sil_drop']:.4f}")

    # ---- 汇总 ----
    # 找跳升最大的层
    mlp_drops = {k: v["mlp_sil_drop"] for k, v in results.items() if k != "summary"}
    attn_drops = {k: v["attn_sil_drop"] for k, v in results.items() if k != "summary"}
    
    max_mlp_layer = max(mlp_drops, key=mlp_drops.get) if mlp_drops else "N/A"
    max_attn_layer = max(attn_drops, key=attn_drops.get) if attn_drops else "N/A"
    
    mlp_dominant_count = sum(1 for v in results.values() if isinstance(v, dict) and v.get("mlp_drives_local_readability", False))
    attn_dominant_count = len([k for k in results if k != "summary"]) - mlp_dominant_count

    summary = {
        "n_layers_tested": len([k for k in results if k != "summary"]),
        "mlp_dominant_layers": mlp_dominant_count,
        "attn_dominant_layers": attn_dominant_count,
        "max_mlp_drop_layer": max_mlp_layer,
        "max_mlp_drop": round(mlp_drops.get(max_mlp_layer, 0), 4),
        "max_attn_drop_layer": max_attn_layer,
        "max_attn_drop": round(attn_drops.get(max_attn_layer, 0), 4),
        "local_metric": "LL-DCF silhouette at target layer (not final DCF)",
    }
    results["summary"] = summary

    plog(f"  MLP dominant in {mlp_dominant_count}/{len([k for k in results if k != 'summary'])} layers (LOCAL)")
    plog(f"  Max MLP drop at {max_mlp_layer}: {mlp_drops.get(max_mlp_layer, 0):.4f}")
    plog(f"  Max Attn drop at {max_attn_layer}: {attn_drops.get(max_attn_layer, 0):.4f}")

    return results


# ==================== Exp2: 多层联合关闭 ====================
def exp2_multi_layer_ablation(model, tokenizer, model_name, device, obj_dict):
    """
    同时关闭多个连续层的MLP/Attn, 看最终DCF是否大降。
    
    区分:
    - 冗余: 单层关闭无效, 多层关闭有效 → 语义约束是冗余分布式编码
    - 无关: 多层关闭也无效 → 这些层确实对最终DCF不重要
    """
    plog("=== Exp2: 多层联合关闭 — 区分冗余vs无关 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)

    cat_list = ["fruit", "animal", "vehicle", "tool", "furniture", "clothing"]
    n_obj = 4  # 减少对象数以控制时间

    # 选择多个窗口进行联合关闭
    if model_name == "qwen3":
        windows = [
            ("L20-24", range(20, 25)),
            ("L24-28", range(24, 29)),
            ("L20-28", range(20, 29)),
            ("L30-35", range(30, min(36, n_layers))),
        ]
    elif model_name == "glm4":
        windows = [
            ("L22-26", range(22, 27)),
            ("L26-30", range(26, 31)),
            ("L22-30", range(22, 31)),
            ("L35-39", range(35, min(40, n_layers))),
        ]
    else:  # deepseek7b
        windows = [
            ("L24-26", range(24, 27)),
            ("L24-27", range(24, min(28, n_layers))),
            ("L20-24", range(20, 25)),
        ]

    results = {}

    for win_name, layer_range in windows:
        layer_list = list(layer_range)
        plog(f"  Testing window {win_name} (L{layer_list[0]}-L{layer_list[-1]})...")

        layers_model = get_layers(model)

        for ablation_type in ["all_mlp", "all_attn", "both"]:
            all_dcfs = []
            all_labels = []

            for cat in cat_list:
                objs = obj_dict.get(cat, [])[:n_obj]
                for obj in objs:
                    prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)

                    # 注册hooks: 在指定层关闭组件
                    hooks = []

                    if ablation_type in ("all_mlp", "both"):
                        for li in layer_list:
                            def mlp_zero_hook(module, input, output):
                                if isinstance(output, tuple):
                                    return (torch.zeros_like(output[0]),) + output[1:]
                                return torch.zeros_like(output)
                            hooks.append(layers_model[li].mlp.register_forward_hook(mlp_zero_hook))

                    if ablation_type in ("all_attn", "both"):
                        for li in layer_list:
                            def attn_zero_hook(module, input, output):
                                if isinstance(output, tuple):
                                    return (torch.zeros_like(output[0]),) + output[1:]
                                return torch.zeros_like(output)
                            hooks.append(layers_model[li].self_attn.register_forward_hook(attn_zero_hook))

                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = out.logits[0, -1].float().cpu().numpy()
                    dcf = compute_dcf_from_logits(logits, tokenizer, FAMILY_WORDS_8D)
                    all_dcfs.append(dcf)
                    all_labels.append(cat)

                    for h in hooks:
                        h.remove()

            sil = cluster_quality(all_dcfs, all_labels)
            key = f"{win_name}_{ablation_type}"
            results[key] = {
                "window": win_name,
                "layers": layer_list,
                "ablation_type": ablation_type,
                "final_dcf_silhouette": round(sil, 4),
                "n_objects": len(all_labels),
            }
            plog(f"    {key}: final_dcf_sil={sil:.4f}")

    # ---- 基线(不关闭任何层) ----
    all_base_dcfs = []
    all_base_labels = []
    for cat in cat_list:
        objs = obj_dict.get(cat, [])[:n_obj]
        for obj in objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[0, -1].float().cpu().numpy()
            dcf = compute_dcf_from_logits(logits, tokenizer, FAMILY_WORDS_8D)
            all_base_dcfs.append(dcf)
            all_base_labels.append(cat)

    base_sil = cluster_quality(all_base_dcfs, all_base_labels)
    results["baseline"] = {"final_dcf_silhouette": round(base_sil, 4)}

    # ---- 汇总: 区分冗余vs无关 ----
    plog(f"  Baseline final DCF sil = {base_sil:.4f}")
    
    # 找最大drop
    max_drop = 0
    max_drop_key = "N/A"
    for k, v in results.items():
        if k in ("baseline", "summary"):
            continue
        drop = base_sil - v["final_dcf_silhouette"]
        if drop > max_drop:
            max_drop = drop
            max_drop_key = k

    summary = {
        "baseline_sil": round(base_sil, 4),
        "max_drop": round(max_drop, 4),
        "max_drop_window": max_drop_key,
        "single_layer_drop_was_below_004": True,  # Phase 472发现
        "multi_layer_redundancy_confirmed": max_drop > 0.1,
    }
    results["summary"] = summary

    plog(f"  Max drop: {max_drop:.4f} at {max_drop_key}")
    plog(f"  Multi-layer redundancy confirmed: {summary['multi_layer_redundancy_confirmed']}")

    return results


# ==================== Exp3: 全类别对方向稳定性 ====================
def exp3_all_pair_direction_stability(model, tokenizer, model_name, device, obj_dict):
    """
    对所有C(6,2)=15对类别计算约束方向的层间稳定性。
    
    Phase 472只测了4对, 需要扩展到15对验证泛化性。
    """
    plog("=== Exp3: 全类别对方向稳定性 (15对) ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)

    cat_list = ["fruit", "animal", "vehicle", "tool", "furniture", "clothing"]
    n_obj = 6
    sample_layers = get_sample_layers(n_layers, n_samples=12)

    # ---- 收集每层每个类别的平均DCF ----
    per_layer_cat_dcf = {f"L{li}": {} for li in sample_layers}

    plog(f"  Collecting per-category DCF across {len(sample_layers)} layers...")
    for cat in cat_list:
        objs = obj_dict.get(cat, [])[:n_obj]
        for obj in objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = attention_mask.sum().item()
            pos = seq_len - 1

            layers_list = get_layers(model)
            captured = {}

            # Hook所有采样层
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
                    dcf = logit_lens_dcf(resid, W_U, tokenizer, FAMILY_WORDS_8D)
                    if cat not in per_layer_cat_dcf[key]:
                        per_layer_cat_dcf[key][cat] = []
                    per_layer_cat_dcf[key][cat].append(dcf)

    # ---- 平均化: 每个类别在每个层的平均DCF ----
    avg_cat_dcf = {}
    for li in sample_layers:
        key = f"L{li}"
        avg_cat_dcf[key] = {}
        for cat in cat_list:
            if cat in per_layer_cat_dcf[key] and per_layer_cat_dcf[key][cat]:
                avg_cat_dcf[key][cat] = np.mean(per_layer_cat_dcf[key][cat], axis=0)

    # ---- 所有15对类别 ----
    all_pairs = []
    for i in range(len(cat_list)):
        for j in range(i + 1, len(cat_list)):
            all_pairs.append((cat_list[i], cat_list[j]))

    plog(f"  Computing direction stability for {len(all_pairs)} pairs...")

    stability_results = {}

    for src, tgt in all_pairs:
        direction_per_layer = {}
        for li in sample_layers:
            key = f"L{li}"
            if src in avg_cat_dcf[key] and tgt in avg_cat_dcf[key]:
                direction = avg_cat_dcf[key][tgt] - avg_cat_dcf[key][src]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction_per_layer[key] = direction / norm

        # 相邻层cosine
        layer_keys = sorted(direction_per_layer.keys(), key=lambda x: int(x[1:]))
        inter_layer_cos = []
        for i in range(len(layer_keys) - 1):
            k1, k2 = layer_keys[i], layer_keys[i+1]
            cos = float(np.dot(direction_per_layer[k1], direction_per_layer[k2]))
            inter_layer_cos.append(cos)

        # Phase2 vs Phase3
        phase2_layers = [k for k in layer_keys if int(k[1:]) / n_layers < 0.6]
        phase3_layers = [k for k in layer_keys if int(k[1:]) / n_layers >= 0.6]

        phase2_cos = []
        phase3_cos = []
        for i in range(len(layer_keys) - 1):
            k1, k2 = layer_keys[i], layer_keys[i+1]
            cos_val = float(np.dot(direction_per_layer[k1], direction_per_layer[k2]))
            if k1 in phase2_layers and k2 in phase2_layers:
                phase2_cos.append(cos_val)
            elif k1 in phase3_layers and k2 in phase3_layers:
                phase3_cos.append(cos_val)

        p2_mean = float(np.mean(phase2_cos)) if phase2_cos else 0
        p3_mean = float(np.mean(phase3_cos)) if phase3_cos else 0

        stability_results[f"{src}_vs_{tgt}"] = {
            "phase2_mean_inter_cos": round(p2_mean, 4),
            "phase3_mean_inter_cos": round(p3_mean, 4),
            "stability_increase": round(p3_mean - p2_mean, 4),
            "crystallized": p3_mean > 0.9,
            "n_phase2_pairs": len(phase2_cos),
            "n_phase3_pairs": len(phase3_cos),
        }

    # ---- 汇总 ----
    all_p2 = [v["phase2_mean_inter_cos"] for v in stability_results.values()]
    all_p3 = [v["phase3_mean_inter_cos"] for v in stability_results.values()]
    n_crystallized = sum(1 for v in stability_results.values() if v["crystallized"])

    summary = {
        "total_pairs": len(all_pairs),
        "phase2_mean_stability": round(float(np.mean(all_p2)), 4),
        "phase3_mean_stability": round(float(np.mean(all_p3)), 4),
        "n_crystallized_pairs": n_crystallized,
        "crystallization_ratio": round(n_crystallized / len(all_pairs), 4),
        "all_pairs_crystallization_confirmed": n_crystallized > len(all_pairs) * 0.5,
    }
    stability_results["summary"] = summary

    plog(f"  Phase2 mean stability: {summary['phase2_mean_stability']}")
    plog(f"  Phase3 mean stability: {summary['phase3_mean_stability']}")
    plog(f"  Crystallized pairs: {n_crystallized}/{len(all_pairs)}")
    plog(f"  Crystallization confirmed (>50%): {summary['all_pairs_crystallization_confirmed']}")

    return stability_results


# ==================== Exp4: Phase3扰动恢复测试 (吸引子) ====================
def exp4_perturbation_recovery(model, tokenizer, model_name, device, obj_dict):
    """
    在Phase3首层注入扰动, 观察后续层是否恢复原方向。
    
    如果恢复 → Phase3是吸引子 (attractor)
    如果不恢复 → Phase3只是简单重复写入同一方向
    
    方法:
    1. 基线: 正常前向, 收集L24-L35(Qwen3)各层的DCF方向
    2. 扰动: 在Phase3首层(Qwen3 L24)注入随机扰动, 收集L25-L35的DCF方向
    3. 比较扰动后方向与基线方向的cosine → 是否恢复
    
    扰动类型:
    - 语义方向扰动: 在DCF方向上加噪声
    - 反方向扰动: 在反DCF方向注入
    - 随机方向扰动: 注入随机向量
    """
    plog("=== Exp4: Phase3扰动恢复测试 — 验证吸引子 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)

    # Phase3首层: Qwen3 L24, GLM4 L24, DS7B L24
    if model_name == "qwen3":
        perturb_layer = 24
        monitor_layers = list(range(24, min(36, n_layers)))
    elif model_name == "glm4":
        perturb_layer = 24
        monitor_layers = list(range(24, min(40, n_layers)))
    else:
        perturb_layer = 24
        monitor_layers = list(range(24, min(28, n_layers)))

    cat_list = ["fruit", "animal", "vehicle", "tool"]
    n_obj = 4
    perturb_beta = 3.0  # 扰动强度

    results = {}

    for perturb_type in ["random", "anti_dcf", "cross_category"]:
        plog(f"  Perturbation type: {perturb_type}")
        recovery_per_layer = {f"L{li}": [] for li in monitor_layers}

        for cat in cat_list:
            objs = obj_dict.get(cat, [])[:n_obj]
            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                seq_len = attention_mask.sum().item()
                pos = seq_len - 1

                layers_list = get_layers(model)

                # ---- 1) 基线: 收集monitor layers的残差 ----
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

                # 基线DCF方向
                base_dcf_per_layer = {}
                for li in monitor_layers:
                    key = f"L{li}"
                    if key in base_captured:
                        resid = base_captured[key][0, pos].numpy()
                        dcf = logit_lens_dcf(resid, W_U, tokenizer, FAMILY_WORDS_8D)
                        base_dcf_per_layer[key] = dcf

                # ---- 2) 扰动: 在perturb_layer注入噪声 ----
                perturbed_captured = {}

                # 首先需要获取perturb_layer的输入残差来计算扰动向量
                # Hook perturb_layer的输入
                perturb_input = {}
                def input_hook(module, inp, output):
                    if isinstance(inp, tuple) and len(inp) > 0:
                        perturb_input['resid'] = inp[0].detach().clone()
                h_input = layers_list[perturb_layer].register_forward_hook(input_hook)

                # 扰动hook: 在perturb_layer输出中加入扰动
                def make_perturb_hook(perturbation_vec):
                    def hook_fn(module, inp, output):
                        if isinstance(output, tuple):
                            perturbed = output[0].clone()
                            perturbed[0, pos, :] += perturbation_vec.to(perturbed.device).to(perturbed.dtype)
                            return (perturbed,) + output[1:]
                        else:
                            perturbed = output.clone()
                            perturbed[0, pos, :] += perturbation_vec.to(perturbed.device).to(perturbed.dtype)
                            return perturbed
                    return hook_fn

                # 计算扰动向量
                if perturb_type == "random":
                    d_model = info.d_model
                    perturb_vec = torch.randn(d_model) * perturb_beta
                    perturb_vec = perturb_vec.float()
                elif perturb_type == "anti_dcf":
                    # 在反DCF方向扰动: 计算基线DCF方向, 取反
                    if perturb_layer in base_dcf_per_layer:
                        base_dcf = base_dcf_per_layer[f"L{perturb_layer}"]
                        # 将DCF方向反投影到residual space (近似: 用W_U^T)
                        # anti_dcf在logit空间 = -base_dcf → residual空间 = -base_dcf @ W_U (近似)
                        d_model = info.d_model
                        perturb_vec = torch.randn(d_model) * perturb_beta  # 近似, 精确需要W_U投影
                    else:
                        d_model = info.d_model
                        perturb_vec = torch.randn(d_model) * perturb_beta
                else:  # cross_category
                    # 注入竞争类别的方向
                    d_model = info.d_model
                    perturb_vec = torch.randn(d_model) * perturb_beta

                perturb_hooks = []
                perturb_hooks.append(layers_list[perturb_layer].register_forward_hook(
                    make_perturb_hook(perturb_vec)))

                # Hook monitor layers
                perturbed_captured = {}
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
                h_input.remove()

                # ---- 3) 比较扰动后方向与基线方向 ----
                for li in monitor_layers:
                    key = f"L{li}"
                    if key in perturbed_captured and key in base_dcf_per_layer:
                        perturbed_resid = perturbed_captured[key][0, pos].numpy()
                        perturbed_dcf = logit_lens_dcf(perturbed_resid, W_U, tokenizer, FAMILY_WORDS_8D)
                        
                        # 计算cosine between perturbed and base DCF directions
                        base_dir = base_dcf_per_layer[key]
                        perturbed_dir = perturbed_dcf
                        
                        # 中心化后计算cosine
                        base_norm = base_dir - np.mean(base_dir)
                        pert_norm = perturbed_dir - np.mean(perturbed_dir)
                        
                        bn = np.linalg.norm(base_norm)
                        pn = np.linalg.norm(pert_norm)
                        if bn > 1e-10 and pn > 1e-10:
                            cos = float(np.dot(base_norm, pert_norm) / (bn * pn))
                        else:
                            cos = 0.0
                        
                        recovery_per_layer[key].append(cos)

        # ---- 汇总该扰动类型 ----
        mean_recovery = {}
        for li in monitor_layers:
            key = f"L{li}"
            if recovery_per_layer[key]:
                mean_recovery[key] = round(float(np.mean(recovery_per_layer[key])), 4)

        # Phase3早期 vs 晚期恢复
        early_layers = [f"L{li}" for li in monitor_layers[:len(monitor_layers)//3]]
        late_layers = [f"L{li}" for li in monitor_layers[2*len(monitor_layers)//3:]]
        
        early_recovery = [mean_recovery[k] for k in early_layers if k in mean_recovery]
        late_recovery = [mean_recovery[k] for k in late_layers if k in mean_recovery]

        results[perturb_type] = {
            "recovery_per_layer": mean_recovery,
            "early_mean_recovery": round(float(np.mean(early_recovery)), 4) if early_recovery else 0,
            "late_mean_recovery": round(float(np.mean(late_recovery)), 4) if late_recovery else 0,
            "attractor_likely": float(np.mean(late_recovery)) > 0.8 if late_recovery else False,
        }

        plog(f"    Early recovery: {results[perturb_type]['early_mean_recovery']}")
        plog(f"    Late recovery: {results[perturb_type]['late_mean_recovery']}")
        plog(f"    Attractor likely: {results[perturb_type]['attractor_likely']}")

    # ---- 总汇总 ----
    attractor_types = sum(1 for v in results.values() if v.get("attractor_likely", False))
    summary = {
        "n_perturbation_types": 3,
        "n_attractor_types": attractor_types,
        "attractor_hypothesis_supported": attractor_types >= 2,
    }
    results["summary"] = summary

    plog(f"  Attractor supported in {attractor_types}/3 perturbation types")

    return results


# ==================== Exp5: DS7B L27 Top-K Token分析 ====================
def exp5_last_layer_topk_analysis(model, tokenizer, model_name, device, obj_dict):
    """
    分析DS7B L27(及其他模型最后一层) MLP/Attn输出的top-k tokens。
    
    目标: 判断DS7B最后一层写入了什么 — 是数学/推理模式还是其他?
    """
    plog("=== Exp5: 最后层组件Top-K Token分析 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)

    # 分析最后2层
    test_layers = list(range(max(0, n_layers - 2), n_layers))

    cat_list = ["fruit", "animal", "vehicle", "tool"]
    n_obj = 4
    top_k = 20

    results = {}

    for li in test_layers:
        plog(f"  Analyzing L{li} MLP/Attn top-k tokens...")
        layers_list = get_layers(model)

        for component in ["mlp", "attn"]:
            all_top_tokens = {}  # token -> count
            all_top_logits = {}  # token -> total logit

            for cat in cat_list:
                objs = obj_dict.get(cat, [])[:n_obj]
                for obj in objs:
                    prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)
                    seq_len = attention_mask.sum().item()
                    pos = seq_len - 1

                    # Hook组件输出
                    captured = {}
                    def make_hook(key):
                        def hook_fn(module, inp, output):
                            if isinstance(output, tuple):
                                captured[key] = output[0].detach().float().cpu()
                            else:
                                captured[key] = output.detach().float().cpu()
                        return hook_fn

                    if component == "mlp":
                        target = layers_list[li].mlp
                    else:
                        target = layers_list[li].self_attn

                    h = target.register_forward_hook(make_hook("output"))
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    h.remove()

                    if "output" in captured:
                        # 取last token位置
                        output_vec = captured["output"][0, pos].numpy()
                        # 投影到logit空间
                        logits = output_vec @ W_U.T
                        # Top-k tokens
                        top_indices = np.argsort(logits)[-top_k:][::-1]
                        for idx in top_indices:
                            tok = tokenizer.decode([int(idx)]).strip()
                            logit_val = float(logits[idx])
                            if tok not in all_top_tokens:
                                all_top_tokens[tok] = 0
                                all_top_logits[tok] = 0
                            all_top_tokens[tok] += 1
                            all_top_logits[tok] += logit_val

            # 排序: 按出现频率
            sorted_tokens = sorted(all_top_tokens.items(), key=lambda x: -x[1])[:top_k]
            sorted_by_logit = sorted(all_top_logits.items(), key=lambda x: -x[1])[:top_k]

            # 分类: semantic / math / number / format / reasoning / other
            categories_found = {"semantic": 0, "math": 0, "number": 0, "format": 0, "reasoning": 0, "other": 0}
            math_words = {"=", "+", "-", "*", "/", "equation", "formula", "calculate", "compute", "solve"}
            number_pattern = set("0123456789")
            format_words = {"the", "a", "an", "is", "are", "of", "in", "that", "which", "it", "this"}
            reasoning_words = {"therefore", "because", "since", "thus", "hence", "so", "consequently"}
            
            for tok, count in sorted_tokens:
                tok_lower = tok.lower().strip()
                if tok_lower in math_words or any(c in tok_lower for c in "=+-*/"):
                    categories_found["math"] += count
                elif any(c.isdigit() for c in tok_lower):
                    categories_found["number"] += count
                elif tok_lower in reasoning_words:
                    categories_found["reasoning"] += count
                elif tok_lower in format_words:
                    categories_found["format"] += count
                elif tok_lower in ["fruit", "animal", "vehicle", "tool", "food", "plant",
                                    "produce", "creature", "transport", "implement"]:
                    categories_found["semantic"] += count
                else:
                    categories_found["other"] += count

            key = f"L{li}_{component}"
            results[key] = {
                "top_tokens_by_freq": [(t, c) for t, c in sorted_tokens[:10]],
                "top_tokens_by_logit": [(t, round(v, 2)) for t, v in sorted_by_logit[:10]],
                "category_distribution": categories_found,
                "dominant_category": max(categories_found, key=categories_found.get),
            }

            plog(f"    L{li} {component}: dominant={results[key]['dominant_category']}")
            plog(f"      Top-5 by freq: {sorted_tokens[:5]}")
            plog(f"      Categories: {categories_found}")

    # ---- 汇总 ----
    summary = {
        "model": model_name,
        "last_layer_writes_behavior_pattern": False,  # 待确认
    }
    
    # 检查最后层MLP是否写入非语义模式
    last_mlp_key = f"L{n_layers-1}_mlp"
    if last_mlp_key in results:
        cat_dist = results[last_mlp_key]["category_distribution"]
        non_semantic = cat_dist["math"] + cat_dist["number"] + cat_dist["reasoning"]
        semantic = cat_dist["semantic"]
        summary["last_mlp_non_semantic_ratio"] = round(non_semantic / max(non_semantic + semantic + cat_dist["other"], 1), 4)
        summary["last_layer_writes_behavior_pattern"] = non_semantic > semantic

    results["summary"] = summary
    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        return

    obj_dict = ROUNDS.get(round_num, ROUNDS[1])

    plog(f"Phase 473: Local Crystallization Decomposition — {model_name}, Round {round_num}")
    plog(f"Core: What drives the L24 LL-DCF readability jump? Is Phase3 an attractor?")

    # ---- 1. 加载模型 ----
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    t_load = time.time() - t0
    plog(f"Model loaded in {t_load:.0f}s")

    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    # ---- 2. 运行实验 ----
    all_results = {
        "phase": 473,
        "model": model_name,
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "theory": "Local Crystallization Decomposition & Attractor Verification",
        "core_question": "What drives the Phase2→3 LL-DCF jump? Is Phase3 an attractor?",
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }

    # Exp1: 局部LL-DCF分解
    t1 = time.time()
    all_results["exp1_local_ll_dcf_decomposition"] = exp1_local_ll_dcf_decomposition(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp1 done in {time.time()-t1:.0f}s")

    # Exp2: 多层联合关闭
    t2 = time.time()
    all_results["exp2_multi_layer_ablation"] = exp2_multi_layer_ablation(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp2 done in {time.time()-t2:.0f}s")

    # Exp3: 全类别对方向稳定性
    t3 = time.time()
    all_results["exp3_all_pair_direction_stability"] = exp3_all_pair_direction_stability(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp3 done in {time.time()-t3:.0f}s")

    # Exp4: Phase3扰动恢复测试
    t4 = time.time()
    all_results["exp4_perturbation_recovery"] = exp4_perturbation_recovery(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp4 done in {time.time()-t4:.0f}s")

    # Exp5: 最后层Top-K Token分析
    t5 = time.time()
    all_results["exp5_last_layer_topk"] = exp5_last_layer_topk_analysis(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp5 done in {time.time()-t5:.0f}s")

    # ---- 3. 保存结果 ----
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase473_{model_name}_r{round_num}.json"

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
    plog(f"Phase 473 {model_name} Round {round_num} complete in {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
