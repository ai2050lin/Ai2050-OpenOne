"""
Phase 472: 约束结晶机制 — Phase2→3跳升的因果分解
=================================================
核心问题: Qwen3在L24的DCF跳升(0.38→0.65)是由什么机制驱动的?

Phase 471发现三阶段模式, 其中Qwen3在L24有显著跳升。
Phase 472用因果方法定位这个跳升的具体写入机制:

Exp1: MLP/Attention贡献分解 — 在L24分别关闭MLP和Attention
  - 在每层, 残差 = input + attn_output + mlp_output
  - DCF跳升是由attn还是mlp驱动的?
  - 在关键层(L22-L27)逐一关闭, 看DCF结构的变化

Exp2: 约束方向的层间稳定性 — 同一语义约束方向在各层是否一致?
  - 计算fruit→vehicle DCF方向在不同层的cosine相似度
  - 验证: Phase3方向是否比Phase2方向更稳定?

Exp3: DS7B的急速写入分析 — L27的MLP输出包含什么?
  - 提取DS7B L27的MLP输出, 看是否有大量语义方向
  - 对比L26和L27的MLP输出差异

模型加载: bfloat16 + device_map="auto"

用法:
  python tests/glm5/phase472_constraint_crystallization.py qwen3 1
  python tests/glm5/phase472_constraint_crystallization.py glm4 1
  python tests/glm5/phase472_constraint_crystallization.py deepseek7b 1
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
}


# ==================== 基础工具 ====================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog(f"Loading {model_name} (bfloat16 + device_map=auto)...")
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
    except:
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


def compute_dcf_centered(dcf_vectors, labels):
    dcf_matrix = np.array(dcf_vectors)
    global_mean = np.mean(dcf_matrix, axis=0)
    centered = dcf_matrix - global_mean
    return centered, global_mean


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
    """从残差向量计算logit-lens DCF"""
    logits = resid @ W_U.T
    return compute_dcf_from_logits(logits, tokenizer, dim_dict)


# ==================== Exp1: MLP/Attention贡献分解 ====================
def exp1_mlp_attn_decomposition(model, tokenizer, model_name, device, obj_dict):
    """
    在关键层(L22-L27 for Qwen3, L23-L30 for GLM4, L24-L27 for DS7B),
    逐一关闭MLP和Attention, 看DCF结构的变化。
    
    方法: 对每层, 用hook捕获attn_output和mlp_output,
    然后分别设为0, 看最终DCF的变化。
    """
    plog("=== Exp1: MLP/Attention贡献分解 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)

    # 关键层选择: Phase 471中DCF跳升的层 ± 2
    # Qwen3: L24跳升 → 测试L22-L27
    # GLM4: L24-L30 → 测试L22-L30
    # DS7B: L27 → 测试L24-L27
    if model_name == "qwen3":
        key_layers = list(range(22, min(28, n_layers)))
    elif model_name == "glm4":
        key_layers = list(range(22, min(31, n_layers)))
    else:
        key_layers = list(range(24, min(28, n_layers)))

    cat_list = ["fruit", "animal", "vehicle", "tool", "furniture", "clothing"]
    n_obj = 6

    results = {}

    for li in key_layers:
        plog(f"  Testing layer L{li}...")
        layer_result = {}

        # ---- 基线: 正常前向, 收集最终logits ----
        all_dcf_base = []
        all_dcf_no_mlp = []
        all_dcf_no_attn = []
        all_labels = []

        layers_list = get_layers(model)

        for cat in cat_list:
            objs = obj_dict.get(cat, [])[:n_obj]
            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                seq_len = attention_mask.sum().item()
                pos = seq_len - 1

                # ---- 基线运行 ----
                with torch.no_grad():
                    base_out = model(input_ids=input_ids, attention_mask=attention_mask)
                base_logits = base_out.logits[0, -1].float().cpu().numpy()
                base_dcf = compute_dcf_from_logits(base_logits, tokenizer, FAMILY_WORDS_8D)
                all_dcf_base.append(base_dcf)
                all_labels.append(cat)

                # ---- 关闭MLP: 将目标层的MLP输出设为0 ----
                captured_mlp = {}
                def mlp_capture_hook(module, input, output):
                    captured_mlp['output'] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
                    # 替换为0
                    if isinstance(output, tuple):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    return torch.zeros_like(output)

                mlp_hook = layers_list[li].mlp.register_forward_hook(mlp_capture_hook)
                with torch.no_grad():
                    no_mlp_out = model(input_ids=input_ids, attention_mask=attention_mask)
                mlp_hook.remove()

                no_mlp_logits = no_mlp_out.logits[0, -1].float().cpu().numpy()
                no_mlp_dcf = compute_dcf_from_logits(no_mlp_logits, tokenizer, FAMILY_WORDS_8D)
                all_dcf_no_mlp.append(no_mlp_dcf)

                # ---- 关闭Attention: 将目标层的Attention输出设为0 ----
                def attn_zero_hook(module, input, output):
                    if isinstance(output, tuple):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    return torch.zeros_like(output)

                # 注意: 对于不同架构, attention输出可能在不同属性上
                sa = layers_list[li].self_attn
                attn_hook = sa.register_forward_hook(attn_zero_hook)
                with torch.no_grad():
                    no_attn_out = model(input_ids=input_ids, attention_mask=attention_mask)
                attn_hook.remove()

                no_attn_logits = no_attn_out.logits[0, -1].float().cpu().numpy()
                no_attn_dcf = compute_dcf_from_logits(no_attn_logits, tokenizer, FAMILY_WORDS_8D)
                all_dcf_no_attn.append(no_attn_dcf)

        # 计算聚类质量
        base_sil = cluster_quality(all_dcf_base, all_labels)
        no_mlp_sil = cluster_quality(all_dcf_no_mlp, all_labels)
        no_attn_sil = cluster_quality(all_dcf_no_attn, all_labels)

        # 计算DCF偏移量
        base_dcfs = np.array(all_dcf_base)
        no_mlp_dcfs = np.array(all_dcf_no_mlp)
        no_attn_dcfs = np.array(all_dcf_no_attn)

        mlp_shift = float(np.mean(np.linalg.norm(no_mlp_dcfs - base_dcfs, axis=1)))
        attn_shift = float(np.mean(np.linalg.norm(no_attn_dcfs - base_dcfs, axis=1)))

        layer_result = {
            "base_silhouette": round(base_sil, 4),
            "no_mlp_silhouette": round(no_mlp_sil, 4),
            "no_attn_silhouette": round(no_attn_sil, 4),
            "mlp_sil_drop": round(base_sil - no_mlp_sil, 4),
            "attn_sil_drop": round(base_sil - no_attn_sil, 4),
            "mlp_shift": round(mlp_shift, 4),
            "attn_shift": round(attn_shift, 4),
            "mlp_more_important": (base_sil - no_mlp_sil) > (base_sil - no_attn_sil),
        }

        results[f"L{li}"] = layer_result
        plog(f"    L{li}: base={base_sil:.4f}, no_mlp={no_mlp_sil:.4f}, no_attn={no_attn_sil:.4f}, "
             f"mlp_drop={layer_result['mlp_sil_drop']:.4f}, attn_drop={layer_result['attn_sil_drop']:.4f}")

    # ---- 汇总: 哪种组件在各关键层更重要 ----
    mlp_dominant_count = sum(1 for v in results.values() if v["mlp_more_important"])
    attn_dominant_count = len(results) - mlp_dominant_count

    summary = {
        "n_layers_tested": len(results),
        "mlp_dominant_layers": mlp_dominant_count,
        "attn_dominant_layers": attn_dominant_count,
        "key_layers": key_layers,
        "mlp_is_primary_constraint_writer": mlp_dominant_count > attn_dominant_count,
    }
    results["summary"] = summary

    plog(f"  MLP dominant: {mlp_dominant_count}/{len(results)} layers")
    plog(f"  Attn dominant: {attn_dominant_count}/{len(results)} layers")

    return results


# ==================== Exp2: 约束方向层间稳定性 ====================
def exp2_constraint_direction_stability(model, tokenizer, model_name, device, obj_dict):
    """
    计算语义约束方向在不同层的稳定性
    
    方法: 在每层提取残差, 计算logit-lens DCF,
    然后比较同一类别的DCF方向在不同层的cosine
    """
    plog("=== Exp2: 约束方向层间稳定性 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)

    cat_list = ["fruit", "animal", "vehicle", "tool", "furniture", "clothing"]
    n_obj = 6
    sample_layers = get_sample_layers(n_layers, n_samples=12)

    # ---- 收集每层每个对象的DCF ----
    per_layer_cat_dcf = {f"L{li}": {} for li in sample_layers}

    for li in sample_layers:
        for cat in cat_list:
            objs = obj_dict.get(cat, [])[:n_obj]
            cat_dcfs = []

            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                # Hook捕获该层残差
                layers_list = get_layers(model)
                captured = {}
                def make_hook(key):
                    def hook_fn(module, inp, output):
                        if isinstance(inp, tuple) and len(inp) > 0:
                            captured[key] = inp[0].detach().float().cpu()
                    return hook_fn

                h = layers_list[li].register_forward_hook(make_hook("resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()

                if "resid" in captured:
                    seq_len = attention_mask.sum().item()
                    pos = seq_len - 1
                    resid = captured["resid"][0, pos].numpy()
                    dcf = logit_lens_dcf(resid, W_U, tokenizer, FAMILY_WORDS_8D)
                    cat_dcfs.append(dcf)

            if cat_dcfs:
                per_layer_cat_dcf[f"L{li}"][cat] = np.mean(cat_dcfs, axis=0)

    # ---- 计算约束方向(类别DCF中心间的差异) ----
    # 对每对类别, 计算DCF差异方向, 然后看各层的cosine
    dim_names = list(FAMILY_WORDS_8D.keys())

    # 选择4个关键对比对
    contrast_pairs = [
        ("fruit", "vehicle"),
        ("animal", "tool"),
        ("clothing", "furniture"),
        ("fruit", "animal"),
    ]

    stability_results = {}

    for src, tgt in contrast_pairs:
        direction_per_layer = {}

        for li in sample_layers:
            key = f"L{li}"
            if src in per_layer_cat_dcf[key] and tgt in per_layer_cat_dcf[key]:
                # 中心化DCF差异方向
                src_dcf = per_layer_cat_dcf[key][src]
                tgt_dcf = per_layer_cat_dcf[key][tgt]
                direction = tgt_dcf - src_dcf
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction_per_layer[key] = direction / norm

        # 计算相邻层之间的方向cosine
        layer_keys = sorted(direction_per_layer.keys(), key=lambda x: int(x[1:]))
        inter_layer_cos = []

        for i in range(len(layer_keys) - 1):
            k1, k2 = layer_keys[i], layer_keys[i+1]
            cos = float(np.dot(direction_per_layer[k1], direction_per_layer[k2]))
            inter_layer_cos.append({
                "from": k1,
                "to": k2,
                "cos": round(cos, 4),
            })

        # 与最终层的cosine
        final_key = layer_keys[-1] if layer_keys else None
        final_cos = []
        if final_key:
            for k in layer_keys[:-1]:
                cos = float(np.dot(direction_per_layer[k], direction_per_layer[final_key]))
                final_cos.append({
                    "layer": k,
                    "cos_to_final": round(cos, 4),
                })

        # Phase2 vs Phase3的平均稳定性
        # Phase2 = 25-60% depth, Phase3 = 60-100% depth
        phase2_layers = [k for k in layer_keys if int(k[1:]) / n_layers < 0.6]
        phase3_layers = [k for k in layer_keys if int(k[1:]) / n_layers >= 0.6]

        phase2_inter_cos = [c["cos"] for c in inter_layer_cos 
                           if c["from"] in phase2_layers and c["to"] in phase2_layers]
        phase3_inter_cos = [c["cos"] for c in inter_layer_cos 
                           if c["from"] in phase3_layers and c["to"] in phase3_layers]

        phase2_final_cos = [c["cos_to_final"] for c in final_cos if c["layer"] in phase2_layers]
        phase3_final_cos = [c["cos_to_final"] for c in final_cos if c["layer"] in phase3_layers]

        stability_results[f"{src}_vs_{tgt}"] = {
            "inter_layer_cos": inter_layer_cos,
            "cos_to_final": final_cos,
            "phase2_mean_inter_cos": round(float(np.mean(phase2_inter_cos)), 4) if phase2_inter_cos else 0,
            "phase3_mean_inter_cos": round(float(np.mean(phase3_inter_cos)), 4) if phase3_inter_cos else 0,
            "phase2_mean_cos_to_final": round(float(np.mean(phase2_final_cos)), 4) if phase2_final_cos else 0,
            "phase3_mean_cos_to_final": round(float(np.mean(phase3_final_cos)), 4) if phase3_final_cos else 0,
            "stability_increase": round(
                (float(np.mean(phase3_inter_cos)) - float(np.mean(phase2_inter_cos)))
                if phase3_inter_cos and phase2_inter_cos else 0, 4),
        }

    # ---- 汇总 ----
    all_phase2_inter = [v["phase2_mean_inter_cos"] for v in stability_results.values()]
    all_phase3_inter = [v["phase3_mean_inter_cos"] for v in stability_results.values()]
    all_phase2_final = [v["phase2_mean_cos_to_final"] for v in stability_results.values()]
    all_phase3_final = [v["phase3_mean_cos_to_final"] for v in stability_results.values()]

    summary = {
        "phase2_mean_inter_stability": round(float(np.mean(all_phase2_inter)), 4) if all_phase2_inter else 0,
        "phase3_mean_inter_stability": round(float(np.mean(all_phase3_inter)), 4) if all_phase3_inter else 0,
        "phase2_mean_alignment_to_final": round(float(np.mean(all_phase2_final)), 4) if all_phase2_final else 0,
        "phase3_mean_alignment_to_final": round(float(np.mean(all_phase3_final)), 4) if all_phase3_final else 0,
        "direction_crystallization_confirmed": float(np.mean(all_phase3_inter)) > float(np.mean(all_phase2_inter)) if all_phase3_inter and all_phase2_inter else False,
    }

    plog(f"  Phase2 inter stability: {summary['phase2_mean_inter_stability']}")
    plog(f"  Phase3 inter stability: {summary['phase3_mean_inter_stability']}")
    plog(f"  Crystallization confirmed: {summary['direction_crystallization_confirmed']}")

    stability_results["summary"] = summary
    return stability_results


# ==================== Exp3: DS7B急速写入分析 ====================
def exp3_ds7b_rapid_write(model, tokenizer, model_name, device, obj_dict):
    """
    分析DS7B的L27 MLP输出 — 为什么语义约束只在最后一层涌现?
    
    方法:
    1. 在L26和L27提取MLP输出(和残差变化)
    2. 将MLP输出投影到logit空间, 计算DCF
    3. 对比L26 MLP vs L27 MLP的DCF结构
    
    对所有模型执行, 但DS7B是重点分析对象
    """
    plog("=== Exp3: MLP输出对DCF的贡献 — 急速写入分析 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)

    # 选择最后4层分析
    test_layers = list(range(max(0, n_layers - 4), n_layers))

    cat_list = ["fruit", "animal", "vehicle", "tool"]
    n_obj = 4

    results = {}

    for li in test_layers:
        plog(f"  Analyzing L{li} MLP contribution...")
        layer_data = {}

        # 收集: 残差和MLP输出
        all_resid_dcf = []
        all_mlp_dcf = []
        all_delta_dcf = []  # mlp_output的DCF
        all_labels = []

        layers_list = get_layers(model)

        for cat in cat_list:
            objs = obj_dict.get(cat, [])[:n_obj]
            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                seq_len = attention_mask.sum().item()
                pos = seq_len - 1

                # Hook: 捕获MLP输出
                captured_mlp = {}
                def mlp_hook(module, input, output):
                    if isinstance(output, tuple):
                        captured_mlp['output'] = output[0].detach().float().cpu()
                    else:
                        captured_mlp['output'] = output.detach().float().cpu()

                mlp_h = layers_list[li].mlp.register_forward_hook(mlp_hook)

                # Hook: 捕获层输出(残差)
                captured_layer = {}
                def layer_hook(module, input, output):
                    if isinstance(output, tuple):
                        captured_layer['output'] = output[0].detach().float().cpu()
                    else:
                        captured_layer['output'] = output.detach().float().cpu()

                layer_h = layers_list[li].register_forward_hook(layer_hook)

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                mlp_h.remove()
                layer_h.remove()

                # 残差DCF
                if 'output' in captured_layer:
                    resid = captured_layer['output'][0, pos].numpy()
                    resid_dcf = logit_lens_dcf(resid, W_U, tokenizer, FAMILY_WORDS_8D)
                    all_resid_dcf.append(resid_dcf)

                # MLP输出DCF
                if 'output' in captured_mlp:
                    mlp_out = captured_mlp['output'][0, pos].numpy()
                    mlp_dcf = logit_lens_dcf(mlp_out, W_U, tokenizer, FAMILY_WORDS_8D)
                    all_mlp_dcf.append(mlp_dcf)
                    all_delta_dcf.append(mlp_dcf)  # MLP输出=残差变化

                all_labels.append(cat)

        # 计算聚类质量
        resid_sil = cluster_quality(all_resid_dcf, all_labels) if all_resid_dcf else 0
        mlp_sil = cluster_quality(all_mlp_dcf, all_labels) if all_mlp_dcf else 0
        delta_sil = cluster_quality(all_delta_dcf, all_labels) if all_delta_dcf else 0

        # MLP输出对最终DCF的贡献比例
        resid_dcfs = np.array(all_resid_dcf) if all_resid_dcf else np.zeros((0, 8))
        delta_dcfs = np.array(all_delta_dcf) if all_delta_dcf else np.zeros((0, 8))

        # delta_dcf / resid_dcf 的比例 (各维度)
        if len(resid_dcfs) > 0 and len(delta_dcfs) > 0:
            ratio_per_dim = []
            for dim_idx in range(8):
                r_vals = resid_dcfs[:, dim_idx]
                d_vals = delta_dcfs[:, dim_idx]
                # 避免分母为0
                r_norm = np.linalg.norm(r_vals)
                d_norm = np.linalg.norm(d_vals)
                ratio_per_dim.append(round(float(d_norm / max(r_norm, 1e-6)), 4))
        else:
            ratio_per_dim = [0.0] * 8

        layer_data = {
            "resid_silhouette": round(resid_sil, 4),
            "mlp_output_silhouette": round(mlp_sil, 4),
            "delta_silhouette": round(delta_sil, 4),
            "mlp_contributes_to_clustering": mlp_sil > 0.1,
            "delta_to_resid_ratio_per_dim": ratio_per_dim,
            "n_objects": len(all_labels),
        }

        results[f"L{li}"] = layer_data
        plog(f"    L{li}: resid_sil={resid_sil:.4f}, mlp_sil={mlp_sil:.4f}, delta_sil={delta_sil:.4f}")

    # ---- 汇总: 最后几层的MLP贡献 ----
    mlp_contributes_count = sum(1 for v in results.values() if v["mlp_contributes_to_clustering"])

    summary = {
        "n_layers_tested": len(results),
        "mlp_contributes_count": mlp_contributes_count,
        "last_layer_mlp_is_semantic_writer": results.get(f"L{n_layers-1}", {}).get("mlp_output_silhouette", 0) > 0.1,
        "model": model_name,
    }
    results["summary"] = summary

    plog(f"  MLP contributes in {mlp_contributes_count}/{len(results)} layers")

    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        return

    obj_dict = ROUNDS[round_num]

    plog(f"Phase 472: Constraint Crystallization — {model_name}, Round {round_num}")
    plog(f"Core: What mechanism drives the Phase2→3 DCF jump?")

    # ---- 1. 加载模型 ----
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    t_load = time.time() - t0
    plog(f"Model loaded in {t_load:.0f}s")

    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    # ---- 2. 运行实验 ----
    all_results = {
        "phase": 472,
        "model": model_name,
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "theory": "Constraint Crystallization Mechanism",
        "core_question": "What drives the Phase2→3 DCF silhouette jump?",
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }

    # Exp1: MLP/Attention贡献分解
    t1 = time.time()
    all_results["exp1_mlp_attn_decomposition"] = exp1_mlp_attn_decomposition(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp1 done in {time.time()-t1:.0f}s")

    # Exp2: 约束方向层间稳定性
    t2 = time.time()
    all_results["exp2_direction_stability"] = exp2_constraint_direction_stability(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp2 done in {time.time()-t2:.0f}s")

    # Exp3: MLP输出对DCF的贡献
    t3 = time.time()
    all_results["exp3_mlp_contribution"] = exp3_ds7b_rapid_write(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp3 done in {time.time()-t3:.0f}s")

    # ---- 3. 保存结果 ----
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase472_{model_name}_r{round_num}.json"

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
    plog(f"Phase 472 {model_name} Round {round_num} complete in {total_time:.0f}s")


if __name__ == "__main__":
    main()