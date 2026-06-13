"""
Phase 475: DCF Projection Robustness & Perturbation Subspace Decomposition
===========================================================================
核心问题: 如果delta永远增长(不是吸引子), 为什么DCF方向仍然鲁棒?

关键实验:
1. Exp1: 扰动投影分解 — 将delta分解为DCF-平行分量和DCF-正交分量
2. Exp2: 修复Qwen3方向扰动norm bug + 纠正层定位
3. Exp3: GLM4重复写入复验 — 控制不同扰动强度
4. Exp4: DS7B L27精确机制 — Attn/MLP分别在格式/语义子空间的投影
5. Exp5: 神经元级因果测试(Qwen3) — top-k神经元ablation

模型加载: bfloat16 + device_map="auto" + flash_attention_2
用法:
  python tests/glm5/phase475_projection_robustness.py qwen3 1
  python tests/glm5/phase475_projection_robustness.py glm4 1
  python tests/glm5/phase475_projection_robustness.py deepseek7b 1
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
    """创建一个将输出存入指定字典的hook, 避免闭包变量捕获问题"""
    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            store_dict[key] = output[0].detach().float().cpu()
        else:
            store_dict[key] = output.detach().float().cpu()
    return hook_fn


def make_perturb_hook(pvec, pos):
    """创建扰动hook, 在指定位置注入扰动向量"""
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


def make_zero_hook():
    """创建零消融hook"""
    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)
    return hook_fn


# ==================== Exp1: 扰动投影分解 ====================
def exp1_perturbation_projection_decomposition(model, tokenizer, model_name, device, obj_dict):
    """
    核心实验: 将delta分解为DCF-平行分量和DCF-正交分量

    步骤:
    1. 构造DCF敏感子空间: 用8D DCF的梯度方向张成的子空间
    2. 在L24注入扰动(随机 + 方向性)
    3. 逐层追踪:
       - ||delta|| (总扰动范数)
       - ||delta_DCF|| (DCF-平行分量范数)
       - ||delta_null|| (DCF-正交分量范数)
       - cos(delta_DCF[l], delta_DCF[L24]) (DCF分量方向保持)

    如果DCF-平行分量缩小而DCF-正交分量增大:
       → 模型把扰动推向DCF无关空间 → DCF投影鲁棒性
    如果两者都增大:
       → 模型整体放大扰动但DCF方向不受影响
    """
    plog("=== Exp1: 扰动投影分解 (DCF-parallel vs DCF-orthogonal) ===")
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

    perturb_beta = 5.0

    # ---- Step 1: 构造DCF敏感子空间 ----
    plog(f"  Step 1: Computing DCF-sensitive subspace...")
    # DCF梯度方向: 对每个类别维度的family words, 在logit空间构造indicator, 投影回残差空间
    dcf_basis = []  # [8, d_model] - 8个DCF基向量
    for dim_name, words in FAMILY_WORDS_8D.items():
        indicator = np.zeros(info.vocab_size)
        count = 0
        for w in words:
            tid = find_token_id(tokenizer, w)
            if tid is not None and tid < info.vocab_size:
                indicator[tid] = 1.0
                count += 1
        if count > 0:
            indicator /= count
        # 梯度: W_U^T @ indicator → [d_model]
        grad = W_U.T @ indicator
        gn = np.linalg.norm(grad)
        if gn > 1e-10:
            grad = grad / gn
        dcf_basis.append(grad)

    dcf_basis = np.array(dcf_basis)  # [8, d_model]
    # Gram-Schmidt正交化
    dcf_ortho = []
    for v in dcf_basis:
        for u in dcf_ortho:
            v = v - np.dot(v, u) * u
        n = np.linalg.norm(v)
        if n > 1e-10:
            dcf_ortho.append(v / n)

    n_dcf_dims = len(dcf_ortho)
    plog(f"    DCF subspace dimension: {n_dcf_dims} (from {dcf_basis.shape[0]} original dims)")
    dcf_ortho = np.array(dcf_ortho)  # [n_dcf_dims, d_model]

    # DCF投影矩阵: P = dcf_ortho^T @ dcf_ortho → [d_model, d_model]
    # delta_DCF = P @ delta, delta_null = delta - delta_DCF

    # ---- Step 2: 收集类别方向 ----
    plog(f"  Step 2: Collecting category directions...")
    cat_mean_resids = {}
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
            cap = {}
            h = layers_list[perturb_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap:
                resids.append(cap["resid"][0, pos].numpy())

        if resids:
            cat_mean_resids[cat] = np.mean(resids, axis=0)

    all_resids = []
    for cat in cat_list:
        if cat in cat_mean_resids:
            all_resids.append(cat_mean_resids[cat])
    global_mean = np.mean(all_resids, axis=0) if all_resids else np.zeros(info.d_model)

    fruit_direction = cat_mean_resids.get("fruit", global_mean) - global_mean
    vehicle_direction = cat_mean_resids.get("vehicle", global_mean) - global_mean
    fd_norm = np.linalg.norm(fruit_direction)
    vd_norm = np.linalg.norm(vehicle_direction)
    if fd_norm > 1e-10:
        fruit_direction = fruit_direction / fd_norm
    if vd_norm > 1e-10:
        vehicle_direction = vehicle_direction / vd_norm

    cos_fv = float(np.dot(fruit_direction, vehicle_direction))
    plog(f"    cos(fruit_dir, vehicle_dir) = {cos_fv:.4f}")

    # ---- Step 3: 扰动类型 ----
    target_norm = perturb_beta * math.sqrt(info.d_model)
    perturb_types = {
        "random": None,
        "anti_fruit": -fruit_direction,
        "toward_vehicle": vehicle_direction,
    }

    results = {}

    for pt_name, perturb_dir_np in perturb_types.items():
        plog(f"  Step 3: Testing perturbation type: {pt_name}")

        # 逐层追踪指标
        total_delta_norm = {f"L{li}": [] for li in monitor_layers}
        dcf_delta_norm = {f"L{li}": [] for li in monitor_layers}
        null_delta_norm = {f"L{li}": [] for li in monitor_layers}
        dcf_fraction = {f"L{li}": [] for li in monitor_layers}  # ||delta_DCF||/||delta||
        dcf_cos_alignment = {f"L{li}": [] for li in monitor_layers}  # cos(delta_DCF[l], delta_DCF[L24])
        recovery_per_layer = {f"L{li}": [] for li in monitor_layers}

        # 存储第一个对象的delta_DCF方向作为参考
        ref_dcf_delta = None

        for cat in cat_list:
            objs = obj_dict.get(cat, [])[:n_obj]
            for oi, obj in enumerate(objs):
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                seq_len = attention_mask.sum().item()
                pos = seq_len - 1

                # ---- Clean baseline ----
                base_cap = {}
                base_hooks = []
                for li in monitor_layers:
                    base_hooks.append(layers_list[li].register_forward_hook(
                        _make_capture_hook(base_cap, f"L{li}")))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                for h in base_hooks:
                    h.remove()

                base_resids = {}
                base_dcfs = {}
                for li in monitor_layers:
                    key = f"L{li}"
                    if key in base_cap:
                        base_resids[key] = base_cap[key][0, pos].numpy()
                        base_dcfs[key] = logit_lens_dcf(base_resids[key], W_U, tokenizer, FAMILY_WORDS_8D)

                # ---- Perturbed ----
                if pt_name == "random":
                    perturb_vec = torch.randn(info.d_model) * perturb_beta
                else:
                    perturb_vec = torch.tensor(perturb_dir_np, dtype=torch.float32) * target_norm

                pert_cap = {}
                pert_hooks = []
                pert_hooks.append(layers_list[perturb_layer].register_forward_hook(
                    make_perturb_hook(perturb_vec, pos)))
                for li in monitor_layers:
                    pert_hooks.append(layers_list[li].register_forward_hook(
                        _make_capture_hook(pert_cap, f"L{li}")))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                for h in pert_hooks:
                    h.remove()

                # ---- 计算分解 ----
                for li in monitor_layers:
                    key = f"L{li}"
                    if key in pert_cap and key in base_resids:
                        pert_resid = pert_cap[key][0, pos].numpy()
                        delta = pert_resid - base_resids[key]

                        # DCF-平行分量: projection onto DCF subspace
                        # delta_DCF = sum_i (delta · dcf_ortho[i]) * dcf_ortho[i]
                        dcf_proj_coeffs = delta @ dcf_ortho.T  # [n_dcf_dims]
                        delta_dcf = dcf_proj_coeffs @ dcf_ortho  # [d_model]
                        delta_null = delta - delta_dcf

                        delta_total_norm = float(np.linalg.norm(delta))
                        delta_dcf_norm = float(np.linalg.norm(delta_dcf))
                        delta_null_norm = float(np.linalg.norm(delta_null))

                        total_delta_norm[key].append(delta_total_norm)
                        dcf_delta_norm[key].append(delta_dcf_norm)
                        null_delta_norm[key].append(delta_null_norm)

                        if delta_total_norm > 1e-10:
                            dcf_fraction[key].append(delta_dcf_norm / delta_total_norm)
                        else:
                            dcf_fraction[key].append(0.0)

                        # DCF分量方向对齐 (与L24的delta_DCF方向比较)
                        if li == monitor_layers[0]:
                            ref_dcf_delta = delta_dcf.copy()

                        if ref_dcf_delta is not None and delta_dcf_norm > 1e-10:
                            ref_norm = np.linalg.norm(ref_dcf_delta)
                            if ref_norm > 1e-10:
                                cos_align = float(np.dot(delta_dcf, ref_dcf_delta) / (delta_dcf_norm * ref_norm))
                            else:
                                cos_align = 0.0
                        else:
                            cos_align = 0.0
                        dcf_cos_alignment[key].append(cos_align)

                        # DCF方向恢复
                        if key in base_dcfs:
                            pert_dcf = logit_lens_dcf(pert_resid, W_U, tokenizer, FAMILY_WORDS_8D)
                            bn_vec = base_dcfs[key] - np.mean(base_dcfs[key])
                            pn_vec = pert_dcf - np.mean(pert_dcf)
                            bn = np.linalg.norm(bn_vec)
                            pn = np.linalg.norm(pn_vec)
                            if bn > 1e-10 and pn > 1e-10:
                                cos_val = float(np.dot(bn_vec, pn_vec) / (bn * pn))
                            else:
                                cos_val = 0.0
                            recovery_per_layer[key].append(cos_val)

        # ---- 汇总 ----
        mean_total_delta = {}
        mean_dcf_delta = {}
        mean_null_delta = {}
        mean_dcf_fraction = {}
        mean_dcf_cos = {}
        mean_recovery = {}

        for li in monitor_layers:
            key = f"L{li}"
            if total_delta_norm[key]:
                mean_total_delta[key] = round(float(np.mean(total_delta_norm[key])), 4)
                mean_dcf_delta[key] = round(float(np.mean(dcf_delta_norm[key])), 4)
                mean_null_delta[key] = round(float(np.mean(null_delta_norm[key])), 4)
                mean_dcf_fraction[key] = round(float(np.mean(dcf_fraction[key])), 6)
                mean_dcf_cos[key] = round(float(np.mean(dcf_cos_alignment[key])), 4)
            if recovery_per_layer[key]:
                mean_recovery[key] = round(float(np.mean(recovery_per_layer[key])), 4)

        # Early vs Late对比
        n_mon = len(monitor_layers)
        early_keys = [f"L{li}" for li in monitor_layers[:n_mon//3]]
        late_keys = [f"L{li}" for li in monitor_layers[2*n_mon//3:]]

        early_dcf_frac = [mean_dcf_fraction[k] for k in early_keys if k in mean_dcf_fraction]
        late_dcf_frac = [mean_dcf_fraction[k] for k in late_keys if k in mean_dcf_fraction]

        early_dcf_norm = [mean_dcf_delta[k] for k in early_keys if k in mean_dcf_delta]
        late_dcf_norm = [mean_dcf_delta[k] for k in late_keys if k in mean_dcf_delta]
        early_null_norm = [mean_null_delta[k] for k in early_keys if k in mean_null_delta]
        late_null_norm = [mean_null_delta[k] for k in late_keys if k in mean_null_delta]

        # DCF分量增长比率
        dcf_ratio = float(np.mean(late_dcf_norm)) / max(float(np.mean(early_dcf_norm)), 1e-10) if early_dcf_norm and late_dcf_norm else 0
        null_ratio = float(np.mean(late_null_norm)) / max(float(np.mean(early_null_norm)), 1e-10) if early_null_norm and late_null_norm else 0

        results[pt_name] = {
            "total_delta_per_layer": mean_total_delta,
            "dcf_delta_per_layer": mean_dcf_delta,
            "null_delta_per_layer": mean_null_delta,
            "dcf_fraction_per_layer": mean_dcf_fraction,
            "dcf_cos_alignment_per_layer": mean_dcf_cos,
            "recovery_per_layer": mean_recovery,
            "early_dcf_fraction": round(float(np.mean(early_dcf_frac)), 6) if early_dcf_frac else 0,
            "late_dcf_fraction": round(float(np.mean(late_dcf_frac)), 6) if late_dcf_frac else 0,
            "dcf_delta_ratio": round(dcf_ratio, 4),
            "null_delta_ratio": round(null_ratio, 4),
            "perturbation_norm": round(target_norm if pt_name != "random" else perturb_beta * math.sqrt(info.d_model), 2),
        }

        plog(f"    {pt_name}: early_dcf_frac={results[pt_name]['early_dcf_fraction']:.6f}, "
             f"late_dcf_frac={results[pt_name]['late_dcf_fraction']:.6f}")
        plog(f"    {pt_name}: dcf_ratio={dcf_ratio:.4f}, null_ratio={null_ratio:.4f}")

    # ---- 总结 ----
    # 判断: DCF分量是否缩小? null分量是否增长更快?
    dcf_shrinks_all = all(results[pt]["dcf_delta_ratio"] < 0.8 for pt in perturb_types if results[pt]["dcf_delta_ratio"] > 0)
    null_grows_faster_all = all(results[pt]["null_delta_ratio"] > results[pt]["dcf_delta_ratio"] for pt in perturb_types if results[pt]["dcf_delta_ratio"] > 0)
    dcf_fraction_decreases = all(results[pt]["late_dcf_fraction"] < results[pt]["early_dcf_fraction"] for pt in perturb_types)

    results["summary"] = {
        "n_perturbation_types": len(perturb_types),
        "dcf_component_shrinks": dcf_shrinks_all,
        "null_grows_faster_than_dcf": null_grows_faster_all,
        "dcf_fraction_decreases_late": dcf_fraction_decreases,
        "n_dcf_subspace_dims": n_dcf_dims,
        "perturb_beta": perturb_beta,
        "perturb_target_norm": round(target_norm, 2),
    }

    plog(f"  DCF component shrinks: {dcf_shrinks_all}")
    plog(f"  Null grows faster than DCF: {null_grows_faster_all}")
    plog(f"  DCF fraction decreases late: {dcf_fraction_decreases}")

    return results


# ==================== Exp2: 修复Qwen3方向扰动 + 纠正层定位 ====================
def exp2_correction_layer_localization(model, tokenizer, model_name, device, obj_dict):
    """
    在L24注入扰动, 逐层关闭L25→末层, 看关闭哪层后DCF恢复下降最大

    修复Phase 474的闭包bug: 使用_make_capture_hook
    修复Phase 474的norm bug: 统一所有扰动类型的范数
    """
    plog("=== Exp2: DCF鲁棒性维持层定位 (修复版) ===")
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

    # ---- 收集fruit方向 ----
    plog(f"  Collecting fruit direction...")
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
            cap = {}
            h = layers_list[perturb_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap:
                r = cap["resid"][0, pos].numpy()
                all_resids_for_mean.append(r)
                if cat == "fruit":
                    fruit_resids.append(r)

    global_mean = np.mean(all_resids_for_mean, axis=0) if all_resids_for_mean else np.zeros(info.d_model)
    fruit_direction = np.mean(fruit_resids, axis=0) - global_mean if fruit_resids else np.zeros(info.d_model)
    fd_norm = np.linalg.norm(fruit_direction)
    if fd_norm > 1e-10:
        fruit_direction = fruit_direction / fd_norm

    # 修复norm: 统一范数为 target_norm
    target_norm = perturb_beta * math.sqrt(info.d_model)
    perturb_vec = torch.tensor(fruit_direction, dtype=torch.float32) * target_norm
    plog(f"  Perturb vector norm: {target_norm:.2f}")

    # ---- 基线恢复(不关闭任何层) ----
    plog(f"  Computing baseline recovery (no ablation)...")
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

            # Clean
            clean_cap = {}
            h_c = layers_list[final_layer].register_forward_hook(_make_capture_hook(clean_cap, "final"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_c.remove()
            if "final" not in clean_cap:
                continue
            clean_resid = clean_cap["final"][0, pos].numpy()
            clean_dcf = logit_lens_dcf(clean_resid, W_U, tokenizer, FAMILY_WORDS_8D)

            # Perturbed
            pert_cap = {}
            h_p1 = layers_list[perturb_layer].register_forward_hook(make_perturb_hook(perturb_vec, pos))
            h_p2 = layers_list[final_layer].register_forward_hook(_make_capture_hook(pert_cap, "final_p"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_p1.remove()
            h_p2.remove()

            if "final_p" in pert_cap:
                p_resid = pert_cap["final_p"][0, pos].numpy()
                p_dcf = logit_lens_dcf(p_resid, W_U, tokenizer, FAMILY_WORDS_8D)
                bn_vec = clean_dcf - np.mean(clean_dcf)
                pn_vec = p_dcf - np.mean(p_dcf)
                bn = np.linalg.norm(bn_vec)
                pn = np.linalg.norm(pn_vec)
                if bn > 1e-10 and pn > 1e-10:
                    cos_val = float(np.dot(bn_vec, pn_vec) / (bn * pn))
                else:
                    cos_val = 0.0
                base_recoveries.append(cos_val)

    baseline_recovery = float(np.mean(base_recoveries)) if base_recoveries else 0
    plog(f"  Baseline recovery at L{final_layer}: {baseline_recovery:.4f}")

    # ---- 逐层关闭测试 ----
    results = {"baseline_recovery": round(baseline_recovery, 4)}
    correction_scores = {}

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

                # Perturbed + Ablate corr_layer
                hooks = []
                hooks.append(layers_list[perturb_layer].register_forward_hook(make_perturb_hook(perturb_vec, pos)))
                hooks.append(layers_list[corr_layer].register_forward_hook(make_zero_hook()))

                ablated_cap = {}
                hooks.append(layers_list[final_layer].register_forward_hook(_make_capture_hook(ablated_cap, "final_a")))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                for h in hooks:
                    h.remove()

                if "final_a" in ablated_cap:
                    a_resid = ablated_cap["final_a"][0, pos].numpy()
                    a_dcf = logit_lens_dcf(a_resid, W_U, tokenizer, FAMILY_WORDS_8D)

                    # Clean baseline (重算)
                    clean_cap2 = {}
                    h_c2 = layers_list[final_layer].register_forward_hook(_make_capture_hook(clean_cap2, "final_c"))
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    h_c2.remove()

                    if "final_c" in clean_cap2:
                        c_resid = clean_cap2["final_c"][0, pos].numpy()
                        c_dcf = logit_lens_dcf(c_resid, W_U, tokenizer, FAMILY_WORDS_8D)
                        cn_vec = c_dcf - np.mean(c_dcf)
                        an_vec = a_dcf - np.mean(a_dcf)
                        cn = np.linalg.norm(cn_vec)
                        an = np.linalg.norm(an_vec)
                        if cn > 1e-10 and an > 1e-10:
                            cos_val = float(np.dot(cn_vec, an_vec) / (cn * an))
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
        correction_scores[f"L{corr_layer}"] = round(recovery_drop, 4)

        plog(f"    L{corr_layer}: recovery={ablated_recovery:.4f}, drop={recovery_drop:.4f}, "
             f"correction={'YES' if recovery_drop > 0.1 else 'no'}")

    # ---- 找关键纠正层 ----
    correction_layers = [k for k, v in results.items() if isinstance(v, dict) and v.get("is_correction_layer", False)]
    # 找drop最大的层
    if correction_scores:
        max_drop_layer = max(correction_scores, key=correction_scores.get)
    else:
        max_drop_layer = "none"

    results["summary"] = {
        "baseline_recovery": round(baseline_recovery, 4),
        "correction_layers": correction_layers,
        "n_correction_layers": len(correction_layers),
        "max_drop_layer": max_drop_layer,
        "max_drop_value": correction_scores.get(max_drop_layer, 0),
    }

    plog(f"  Correction layers: {correction_layers}")
    plog(f"  Max drop at {max_drop_layer}: {correction_scores.get(max_drop_layer, 0):.4f}")

    return results


# ==================== Exp3: GLM4重复写入复验 ====================
def exp3_glm4_repeat_write_verification(model, tokenizer, model_name, device, obj_dict):
    """
    GLM4重复写入验证:
    1. 不同扰动强度测试(beta=3,5,8)
    2. 扰动投影分解(GLM4版)
    3. 检查后续层是否写入baseline方向(独立于前层偏差)
    """
    plog("=== Exp3: GLM4重复写入复验 ===")
    if model_name != "glm4":
        plog(f"  Skipping (only for GLM4)")
        return {"skipped": True, "reason": "only for glm4"}

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)
    layers_list = get_layers(model)

    perturb_layer = 24
    monitor_layers = list(range(24, min(40, n_layers)))
    cat_list = ["fruit", "animal", "vehicle", "tool"]
    n_obj = 4

    # DCF子空间(复用Exp1的逻辑)
    dcf_basis = []
    for dim_name, words in FAMILY_WORDS_8D.items():
        indicator = np.zeros(info.vocab_size)
        count = 0
        for w in words:
            tid = find_token_id(tokenizer, w)
            if tid is not None and tid < info.vocab_size:
                indicator[tid] = 1.0
                count += 1
        if count > 0:
            indicator /= count
        grad = W_U.T @ indicator
        gn = np.linalg.norm(grad)
        if gn > 1e-10:
            grad = grad / gn
        dcf_basis.append(grad)

    dcf_basis = np.array(dcf_basis)
    dcf_ortho = []
    for v in dcf_basis:
        for u in dcf_ortho:
            v = v - np.dot(v, u) * u
        n = np.linalg.norm(v)
        if n > 1e-10:
            dcf_ortho.append(v / n)
    dcf_ortho = np.array(dcf_ortho)
    plog(f"  DCF subspace dims: {len(dcf_ortho)}")

    # 收集类别方向
    cat_mean_resids = {}
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
            cap = {}
            h = layers_list[perturb_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap:
                resids.append(cap["resid"][0, pos].numpy())
        if resids:
            cat_mean_resids[cat] = np.mean(resids, axis=0)

    all_resids = [cat_mean_resids[c] for c in cat_list if c in cat_mean_resids]
    global_mean = np.mean(all_resids, axis=0) if all_resids else np.zeros(info.d_model)

    fruit_direction = cat_mean_resids.get("fruit", global_mean) - global_mean
    fd_norm = np.linalg.norm(fruit_direction)
    if fd_norm > 1e-10:
        fruit_direction = fruit_direction / fd_norm

    # ---- 不同扰动强度 ----
    betas = [3.0, 5.0, 8.0]
    results = {}

    for beta in betas:
        plog(f"  Testing beta={beta}...")
        target_norm = beta * math.sqrt(info.d_model)
        perturb_vec = torch.tensor(-fruit_direction, dtype=torch.float32) * target_norm

        total_delta = {f"L{li}": [] for li in monitor_layers}
        dcf_delta = {f"L{li}": [] for li in monitor_layers}
        null_delta = {f"L{li}": [] for li in monitor_layers}
        dcf_frac = {f"L{li}": [] for li in monitor_layers}
        recovery = {f"L{li}": [] for li in monitor_layers}

        for cat in cat_list:
            objs = obj_dict.get(cat, [])[:n_obj]
            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                seq_len = attention_mask.sum().item()
                pos = seq_len - 1

                # Clean
                clean_cap = {}
                base_hooks = []
                for li in monitor_layers:
                    base_hooks.append(layers_list[li].register_forward_hook(
                        _make_capture_hook(clean_cap, f"L{li}")))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                for h in base_hooks:
                    h.remove()

                # Perturbed
                pert_cap = {}
                pert_hooks = []
                pert_hooks.append(layers_list[perturb_layer].register_forward_hook(
                    make_perturb_hook(perturb_vec, pos)))
                for li in monitor_layers:
                    pert_hooks.append(layers_list[li].register_forward_hook(
                        _make_capture_hook(pert_cap, f"L{li}")))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                for h in pert_hooks:
                    h.remove()

                for li in monitor_layers:
                    key = f"L{li}"
                    if key in pert_cap and key in clean_cap:
                        pert_resid = pert_cap[key][0, pos].numpy()
                        clean_resid = clean_cap[key][0, pos].numpy()
                        delta = pert_resid - clean_resid

                        dcf_proj_coeffs = delta @ dcf_ortho.T
                        delta_dcf = dcf_proj_coeffs @ dcf_ortho
                        delta_null_vec = delta - delta_dcf

                        dn = float(np.linalg.norm(delta))
                        ddn = float(np.linalg.norm(delta_dcf))
                        dnn = float(np.linalg.norm(delta_null_vec))

                        total_delta[key].append(dn)
                        dcf_delta[key].append(ddn)
                        null_delta[key].append(dnn)
                        dcf_frac[key].append(ddn / max(dn, 1e-10))

                        # Recovery
                        clean_dcf = logit_lens_dcf(clean_resid, W_U, tokenizer, FAMILY_WORDS_8D)
                        pert_dcf = logit_lens_dcf(pert_resid, W_U, tokenizer, FAMILY_WORDS_8D)
                        cn = clean_dcf - np.mean(clean_dcf)
                        pn = pert_dcf - np.mean(pert_dcf)
                        cnn = np.linalg.norm(cn)
                        pnn = np.linalg.norm(pn)
                        if cnn > 1e-10 and pnn > 1e-10:
                            recovery[key].append(float(np.dot(cn, pn) / (cnn * pnn)))

        # 汇总
        mean_total = {k: round(float(np.mean(v)), 4) for k, v in total_delta.items() if v}
        mean_dcf = {k: round(float(np.mean(v)), 4) for k, v in dcf_delta.items() if v}
        mean_null = {k: round(float(np.mean(v)), 4) for k, v in null_delta.items() if v}
        mean_frac = {k: round(float(np.mean(v)), 6) for k, v in dcf_frac.items() if v}
        mean_rec = {k: round(float(np.mean(v)), 4) for k, v in recovery.items() if v}

        early_keys = [f"L{li}" for li in monitor_layers[:4]]
        late_keys = [f"L{li}" for li in monitor_layers[-4:]]

        early_dcf = [mean_dcf[k] for k in early_keys if k in mean_dcf]
        late_dcf = [mean_dcf[k] for k in late_keys if k in mean_dcf]
        dcf_ratio = float(np.mean(late_dcf)) / max(float(np.mean(early_dcf)), 1e-10) if early_dcf and late_dcf else 0

        early_null = [mean_null[k] for k in early_keys if k in mean_null]
        late_null = [mean_null[k] for k in late_keys if k in mean_null]
        null_ratio = float(np.mean(late_null)) / max(float(np.mean(early_null)), 1e-10) if early_null and late_null else 0

        results[f"beta_{beta:.0f}"] = {
            "total_delta_per_layer": mean_total,
            "dcf_delta_per_layer": mean_dcf,
            "null_delta_per_layer": mean_null,
            "dcf_fraction_per_layer": mean_frac,
            "recovery_per_layer": mean_rec,
            "dcf_delta_ratio": round(dcf_ratio, 4),
            "null_delta_ratio": round(null_ratio, 4),
        }

        plog(f"    beta={beta}: dcf_ratio={dcf_ratio:.4f}, null_ratio={null_ratio:.4f}")

    # ---- 判断重复写入 ----
    # 如果不同beta下DCF分量都不缩小(null也不缩小) → 重复写入
    # 如果DCF分量缩小而null增长 → 投影鲁棒性
    all_dcf_not_shrink = all(results[f"beta_{b:.0f}"]["dcf_delta_ratio"] >= 0.8 for b in betas)
    all_null_grows = all(results[f"beta_{b:.0f}"]["null_delta_ratio"] > 1.2 for b in betas)

    results["summary"] = {
        "dcf_component_stable_or_grows": all_dcf_not_shrink,
        "null_component_grows": all_null_grows,
        "conclusion": "repeat_write" if all_dcf_not_shrink else "projection_robustness",
    }

    plog(f"  DCF component stable/grows: {all_dcf_not_shrink}")
    plog(f"  Null component grows: {all_null_grows}")

    return results


# ==================== Exp4: DS7B L27精确机制 ====================
def exp4_ds7b_l27_precise_mechanism(model, tokenizer, model_name, device, obj_dict):
    """
    DS7B L27精确机制:
    1. L27 Attention和MLP分别在格式/语义子空间的投影
    2. L27 Attention head-level分析(哪些head贡献格式写入)
    3. 扰动传播: L27 Attn/MLP各自对delta的贡献
    """
    plog("=== Exp4: DS7B L27精确机制 ===")
    if model_name != "deepseek7b":
        plog(f"  Skipping (only for DS7B)")
        return {"skipped": True, "reason": "only for deepseek7b"}

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)
    layers_list = get_layers(model)

    cat_list = ["fruit", "animal", "vehicle", "tool"]
    n_obj = 4

    # 格式/语义词token IDs
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
    plog(f"  Format tokens: {len(format_ids)}, Semantic tokens: {len(semantic_ids)}")

    # ---- 分析L27 MLP和Attn的输出 ----
    plog(f"  Analyzing L27 component outputs...")
    layer27 = layers_list[27]

    # 对L26和L27的残差分别计算格式/语义分数
    # 同时hook L27的Attn输出和MLP输出
    mlp_format_scores = []
    mlp_semantic_scores = []
    attn_format_scores = []
    attn_semantic_scores = []

    # 还需要L26的残差作为对比
    l26_format_scores = []
    l26_semantic_scores = []
    l27_format_scores = []
    l27_semantic_scores = []

    for cat in cat_list:
        objs = obj_dict.get(cat, [])[:n_obj]
        for obj in objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = attention_mask.sum().item()
            pos = seq_len - 1

            # Hook L26输出, L27 Attn输出, L27 MLP输出, L27输出
            captured = {}

            # L26输出
            h_l26 = layers_list[26].register_forward_hook(_make_capture_hook(captured, "L26"))
            # L27 Attn输出 (hook self_attn)
            h_l27_attn = layer27.self_attn.register_forward_hook(_make_capture_hook(captured, "L27_attn"))
            # L27 MLP输出 (hook mlp)
            h_l27_mlp = layer27.mlp.register_forward_hook(_make_capture_hook(captured, "L27_mlp"))
            # L27输出
            h_l27 = layers_list[27].register_forward_hook(_make_capture_hook(captured, "L27"))

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            h_l26.remove()
            h_l27_attn.remove()
            h_l27_mlp.remove()
            h_l27.remove()

            # 计算各组件在格式/语义子空间的投影
            for key, fmt_list, sem_list in [
                ("L26", l26_format_scores, l26_semantic_scores),
                ("L27", l27_format_scores, l27_semantic_scores),
            ]:
                if key in captured:
                    resid = captured[key][0, pos].numpy()
                    logits = resid @ W_U.T
                    fmt_score = float(np.mean(logits[format_ids])) if format_ids else 0
                    sem_score = float(np.mean(logits[semantic_ids])) if semantic_ids else 0
                    fmt_list.append(fmt_score)
                    sem_list.append(sem_score)

            # Attn和MLP输出(注意: 这些是增量, 不是残差)
            for key, fmt_list, sem_list in [
                ("L27_attn", attn_format_scores, attn_semantic_scores),
                ("L27_mlp", mlp_format_scores, mlp_semantic_scores),
            ]:
                if key in captured:
                    out = captured[key][0, pos].numpy()
                    logits = out @ W_U.T
                    fmt_score = float(np.mean(logits[format_ids])) if format_ids else 0
                    sem_score = float(np.mean(logits[semantic_ids])) if semantic_ids else 0
                    fmt_list.append(fmt_score)
                    sem_list.append(sem_score)

    results = {
        "L26_format_score": round(float(np.mean(l26_format_scores)), 4) if l26_format_scores else 0,
        "L26_semantic_score": round(float(np.mean(l26_semantic_scores)), 4) if l26_semantic_scores else 0,
        "L27_format_score": round(float(np.mean(l27_format_scores)), 4) if l27_format_scores else 0,
        "L27_semantic_score": round(float(np.mean(l27_semantic_scores)), 4) if l27_semantic_scores else 0,
        "L27_attn_format_score": round(float(np.mean(attn_format_scores)), 4) if attn_format_scores else 0,
        "L27_attn_semantic_score": round(float(np.mean(attn_semantic_scores)), 4) if attn_semantic_scores else 0,
        "L27_mlp_format_score": round(float(np.mean(mlp_format_scores)), 4) if mlp_format_scores else 0,
        "L27_mlp_semantic_score": round(float(np.mean(mlp_semantic_scores)), 4) if mlp_semantic_scores else 0,
    }

    # 判断哪个组件是格式覆盖主因
    attn_fmt_vs_sem = results["L27_attn_format_score"] - results["L27_attn_semantic_score"]
    mlp_fmt_vs_sem = results["L27_mlp_format_score"] - results["L27_mlp_semantic_score"]

    results["summary"] = {
        "attn_format_dominant": attn_fmt_vs_sem > 0,
        "mlp_format_dominant": mlp_fmt_vs_sem > 0,
        "attn_fmt_minus_sem": round(attn_fmt_vs_sem, 4),
        "mlp_fmt_minus_sem": round(mlp_fmt_vs_sem, 4),
        "format_overlay_source": "attn" if attn_fmt_vs_sem > mlp_fmt_vs_sem else "mlp",
    }

    plog(f"  L26: format={results['L26_format_score']:.4f}, semantic={results['L26_semantic_score']:.4f}")
    plog(f"  L27: format={results['L27_format_score']:.4f}, semantic={results['L27_semantic_score']:.4f}")
    plog(f"  L27 Attn: format={results['L27_attn_format_score']:.4f}, semantic={results['L27_attn_semantic_score']:.4f}")
    plog(f"  L27 MLP: format={results['L27_mlp_format_score']:.4f}, semantic={results['L27_mlp_semantic_score']:.4f}")
    plog(f"  Format overlay source: {results['summary']['format_overlay_source']}")

    return results


# ==================== Exp5: 神经元级因果测试(Qwen3) ====================
def exp5_neuron_causal_test(model, tokenizer, model_name, device, obj_dict):
    """
    Qwen3神经元级因果测试:
    1. 从Phase 474 Exp5的top-k神经元中选择
    2. Ablation测试: 关闭top-k fruit writer, 看DCF是否下降
    3. 检查类别特异性: 关闭fruit writer不应该同等影响animal DCF
    """
    plog("=== Exp5: 神经元级因果测试 (Qwen3) ===")
    if model_name != "qwen3":
        plog(f"  Skipping (only for Qwen3)")
        return {"skipped": True, "reason": "only for qwen3"}

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)
    layers_list = get_layers(model)

    cat_list = ["fruit", "animal", "vehicle", "tool"]
    n_obj = 4

    # 选择关键层和神经元
    # 从Phase 474结果: L30和L33的fruit top neuron
    key_layers = [30, 33, 35]

    # 先收集fruit DCF梯度方向
    plog(f"  Computing DCF gradients...")
    dcf_gradients = {}
    for cat in cat_list:
        family_words = FAMILY_WORDS_8D.get(cat, [])
        indicator = np.zeros(info.vocab_size)
        count = 0
        for w in family_words:
            tid = find_token_id(tokenizer, w)
            if tid is not None and tid < info.vocab_size:
                indicator[tid] = 1.0
                count += 1
        if count > 0:
            indicator /= count
        grad = W_U.T @ indicator
        gn = np.linalg.norm(grad)
        if gn > 1e-10:
            grad = grad / gn
        dcf_gradients[cat] = grad

    results = {}

    for li in key_layers:
        plog(f"  Analyzing L{li} neuron causal effects...")
        layer = layers_list[li]
        lw = get_layer_weights(layer, info.d_model, info.mlp_type)
        W_down = lw.W_down  # [d_model, intermediate_size]

        if W_down is None:
            plog(f"    L{li}: W_down not available, skipping")
            continue

        # 收集MLP中间激活
        all_activations = {}  # {cat: [n_obj, intermediate_size]}
        for cat in cat_list:
            objs = obj_dict.get(cat, [])[:n_obj]
            cat_acts = []
            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                seq_len = attention_mask.sum().item()
                pos = seq_len - 1

                captured_act = {}
                def act_hook(module, inp, output):
                    if isinstance(inp, tuple) and len(inp) > 0:
                        captured_act["mlp_mid"] = inp[0].detach().float().cpu()

                h_act = layer.mlp.down_proj.register_forward_hook(act_hook)
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_act.remove()

                if "mlp_mid" in captured_act:
                    cat_acts.append(captured_act["mlp_mid"][0, pos].numpy())

            if cat_acts:
                all_activations[cat] = np.mean(cat_acts, axis=0)

        if "fruit" not in all_activations:
            continue

        # 找fruit top-20正贡献和负贡献神经元
        grad_dcf = dcf_gradients.get("fruit", np.zeros(info.d_model))
        neuron_write_dir = W_down.T @ grad_dcf  # [intermediate_size]
        fruit_contribution = all_activations["fruit"] * neuron_write_dir

        top_k = 20
        # 正贡献(增强fruit DCF)
        pos_indices = np.argsort(fruit_contribution)[-top_k:][::-1]
        # 负贡献(抑制fruit DCF)
        neg_indices = np.argsort(fruit_contribution)[:top_k]

        # ---- Ablation: 关闭正贡献神经元 ----
        plog(f"    Ablating top-{top_k} positive fruit writers at L{li}...")

        # 收集基线DCF
        baseline_dcfs = {}
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

                cap = {}
                h = layers_list[li].register_forward_hook(_make_capture_hook(cap, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                if "resid" in cap:
                    dcfs.append(logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer, FAMILY_WORDS_8D))
            if dcfs:
                baseline_dcfs[cat] = np.mean(dcfs, axis=0)

        # Ablation: 将正贡献神经元的中间激活设为0
        # 用register_forward_pre_hook修改down_proj的输入
        ablation_dcfs = {}
        pos_indices_list = [int(i) for i in pos_indices[:20]]

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
                        # args is a tuple of input tensors
                        x = args[0].clone()
                        for idx in indices:
                            if idx < x.shape[-1]:
                                x[0, position, idx] = 0.0
                        return (x,) + args[1:]
                    return pre_hook

                h_abl = layer.mlp.down_proj.register_forward_pre_hook(
                    make_pre_ablation_hook(pos_indices_list, pos))
                cap = {}
                h_resid = layers_list[li].register_forward_hook(_make_capture_hook(cap, "resid"))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                h_abl.remove()
                h_resid.remove()
                if "resid" in cap:
                    dcfs.append(logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer, FAMILY_WORDS_8D))
            if dcfs:
                ablation_dcfs[cat] = np.mean(dcfs, axis=0)

        # 计算DCF变化
        dcf_changes = {}
        for cat in cat_list:
            if cat in baseline_dcfs and cat in ablation_dcfs:
                # fruit维度(第0维)的变化
                baseline_fruit_dim = baseline_dcfs[cat][0]
                ablated_fruit_dim = ablation_dcfs[cat][0]
                change = ablated_fruit_dim - baseline_fruit_dim
                dcf_changes[cat] = round(float(change), 6)

        results[f"L{li}"] = {
            "top_positive_neurons": [int(i) for i in pos_indices[:10]],
            "top_negative_neurons": [int(i) for i in neg_indices[:10]],
            "dcf_change_after_ablation": dcf_changes,
            "fruit_specific": dcf_changes.get("fruit", 0) < 0 and
                              abs(dcf_changes.get("fruit", 0)) > abs(dcf_changes.get("animal", 0)),
        }

        plog(f"    L{li} ablation results: {dcf_changes}")
        plog(f"    Fruit-specific: {results[f'L{li}']['fruit_specific']}")

    results["summary"] = {
        "model": model_name,
        "key_layers": key_layers,
        "method": "ablate top-k fruit-positive neurons, check DCF change per category",
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

    plog(f"Phase 475: DCF Projection Robustness & Perturbation Subspace Decomposition")
    plog(f"Model: {model_name}, Round: {round_num}")
    plog(f"Core: Decompose delta into DCF-parallel vs DCF-orthogonal components")

    # ---- 1. 加载模型 ----
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    t_load = time.time() - t0
    plog(f"Model loaded in {t_load:.0f}s")

    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    # ---- 2. 运行实验 ----
    all_results = {
        "phase": 475,
        "model": model_name,
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "theory": "DCF Projection Robustness & Subspace Decomposition",
        "core_question": "If delta grows, why does DCF direction stay robust? Where does perturbation growth happen?",
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }

    # Exp1: 扰动投影分解 (所有模型)
    t1 = time.time()
    all_results["exp1_perturbation_projection_decomposition"] = exp1_perturbation_projection_decomposition(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp1 done in {time.time()-t1:.0f}s")

    # Exp2: Qwen3纠正层定位 (仅Qwen3)
    t2 = time.time()
    all_results["exp2_correction_layer_localization"] = exp2_correction_layer_localization(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp2 done in {time.time()-t2:.0f}s")

    # Exp3: GLM4重复写入复验 (仅GLM4)
    t3 = time.time()
    all_results["exp3_glm4_repeat_write_verification"] = exp3_glm4_repeat_write_verification(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp3 done in {time.time()-t3:.0f}s")

    # Exp4: DS7B L27精确机制 (仅DS7B)
    t4 = time.time()
    all_results["exp4_ds7b_l27_precise_mechanism"] = exp4_ds7b_l27_precise_mechanism(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp4 done in {time.time()-t4:.0f}s")

    # Exp5: 神经元级因果测试 (仅Qwen3)
    t5 = time.time()
    all_results["exp5_neuron_causal_test"] = exp5_neuron_causal_test(
        model, tokenizer, model_name, device, obj_dict)
    plog(f"Exp5 done in {time.time()-t5:.0f}s")

    # ---- 3. 保存结果 ----
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase475_{model_name}_r{round_num}.json"

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
    plog(f"Phase 475 {model_name} Round {round_num} complete in {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
