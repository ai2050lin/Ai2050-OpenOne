"""
Phase 482: 类别-层位图谱、正反向剂量曲线与边界残差必要性
================================================================

核心目标:
1. Exp1: 类别-层位图谱 — 扫描所有类别×关键层，找到每个类别的最佳边界层
2. Exp2: 正反向剂量曲线 — 多类别+specific/-specific多scale注入
3. Exp3: 边界残差必要性 — 从自然输入中移除B_c，测试目标类别是否下降

关键创新:
- 全类别×全层扫描（不仅是DS7B，Qwen3和GLM4也扫描）
- 正反向不对称的定量刻画
- 移除测试(而不仅是注入测试)——这是因果必要性的关键证据

用法:
  python tests/glm5/phase482_layer_map_dose_curve.py qwen3 1
  python tests/glm5/phase482_layer_map_dose_curve.py glm4 1
  python tests/glm5/phase482_layer_map_dose_curve.py deepseek7b 1
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


# ==================== 数据定义 ====================
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

# Phase 481最优邻居（自动+人工融合结果）
BEST_NEIGHBORS = {
    "qwen3": {
        "fruit": ["plant", "food"],
        "animal": ["food", "clothing"],       # 人工更好(1.99 vs 1.24)
        "tool": ["vehicle", "furniture"],
        "vehicle": ["furniture", "tool"],
        "clothing": ["furniture", "tool"],    # 人工更好(4.58 vs 3.61)
        "furniture": ["vehicle", "clothing"], # 自动更好(1.31 vs 1.09)
        "food": ["plant", "vehicle"],         # 自动更好(2.04 vs 0.98)
        "plant": ["food", "animal"],          # 自动更好(2.67 vs 1.00)
    },
    "glm4": {
        "fruit": ["plant", "food"],
        "animal": ["food", "clothing"],       # 人工≈(1.71 vs 1.58)
        "tool": ["furniture", "vehicle"],
        "vehicle": ["tool", "furniture"],     # 人工≈(2.36 vs 2.18)
        "clothing": ["furniture", "plant"],   # 自动更好(2.71 vs 1.24)
        "furniture": ["vehicle", "clothing"], # 自动更好(4.48 vs 2.91)
        "food": ["plant", "fruit"],           # 人工更好(0.74 vs 0.39)
        "plant": ["vehicle", "clothing"],     # 自动更好(2.83 vs 0.95)
    },
    "deepseek7b": {
        "fruit": ["plant", "food"],
        "animal": ["food", "clothing"],
        "tool": ["vehicle", "furniture"],
        "vehicle": ["furniture", "tool"],
        "clothing": ["furniture", "plant"],    # Phase 481 auto
        "furniture": ["tool", "clothing"],
        "food": ["plant", "fruit"],
        "plant": ["food", "fruit"],
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

# 层扫描范围
LAYER_RANGES = {
    "qwen3": list(range(20, 36)),      # L20-L35
    "glm4": list(range(24, 40)),         # L24-L39
    "deepseek7b": list(range(16, 28)),  # L16-L27
}

# 剂量曲线scale
DOSE_SCALES = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

# 注入测试的类别(选代表性的)
DOSE_CATEGORIES = ["fruit", "animal", "vehicle", "food", "plant"]


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
            # 移除: 减去在B_c方向上的投影
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


def safe_cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


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


# ==================== Exp1: 类别-层位图谱 ====================
def exp1_layer_map(model, tokenizer, device, model_name, W_U):
    """扫描所有类别×所有候选层，找到每个类别的最佳边界层"""
    plog("=== Exp1: 类别-层位图谱 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    layer_range = LAYER_RANGES[model_name]
    neighbors = BEST_NEIGHBORS[model_name]
    
    all_results = {}
    best_per_cat = {}
    
    # 先在每个层获取所有类别的raw direction（每层一次前向，复用结果）
    for li, layer_idx in enumerate(layer_range):
        t0 = time.time()
        plog(f"  Layer {layer_idx} ({li+1}/{len(layer_range)})...")
        
        # 获取所有类别的raw direction
        raw_dirs = get_category_residuals_at_layer(
            model, tokenizer, device, model_name,
            categories=CATEGORIES_TRAIN, n_obj=4, target_layer=layer_idx
        )
        
        # 对每个类别构造specific方向并测试
        for cat_name in cat_names:
            if cat_name not in raw_dirs:
                continue
            target_vec = raw_dirs[cat_name]
            target_idx = cat_names.index(cat_name)
            
            # 构造specific方向
            basis_vecs = [raw_dirs[n] for n in neighbors[cat_name] if n in raw_dirs]
            if basis_vecs:
                spec_vec = qr_orthogonalize(target_vec, basis_vecs)
            else:
                spec_vec = target_vec.copy()
            
            spec_norm = np.linalg.norm(spec_vec)
            if spec_norm < 1e-6:
                continue
            
            # 注入测试: 用2个对象测试(控制时间)
            test_objs = CATEGORIES[cat_name][4:6]  # test set前2个
            template = RELATION_TEMPLATES["kind_of"]
            dcf_deltas = []
            
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
                
                # 注入
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
                dcf_deltas.append(inject_dcf - baseline_dcf)
            
            if dcf_deltas:
                avg_delta = np.mean(dcf_deltas, axis=0)
                sel = compute_selectivity(avg_delta, target_idx)
                key = f"L{layer_idx}_{cat_name}"
                all_results[key] = {
                    "layer": layer_idx,
                    "category": cat_name,
                    "selectivity": sel,
                    "target_dcf_delta": float(avg_delta[target_idx]),
                    "dcf_delta": {cat_names[i]: float(avg_delta[i]) for i in range(len(cat_names))},
                    "spec_norm": float(spec_norm),
                }
                
                # 更新best
                if cat_name not in best_per_cat or sel > best_per_cat[cat_name]["selectivity"]:
                    best_per_cat[cat_name] = {
                        "layer": layer_idx,
                        "selectivity": sel,
                        "target_dcf_delta": float(avg_delta[target_idx]),
                        "spec_norm": float(spec_norm),
                        "depth_ratio": layer_idx / info.n_layers,
                    }
        
        elapsed = time.time() - t0
        plog(f"    Done in {elapsed:.1f}s")
        gc.collect()
    
    # 汇总best_per_cat
    plog("\n  === Best Layer per Category ===")
    for cat_name in cat_names:
        if cat_name in best_per_cat:
            b = best_per_cat[cat_name]
            plog(f"    {cat_name}: L{b['layer']} (depth={b['depth_ratio']:.2f}), "
                  f"sel={b['selectivity']:.2f}, Δ={b['target_dcf_delta']:.2f}")
    
    return {"all_results": all_results, "best_per_cat": best_per_cat}


# ==================== Exp2: 正反向剂量曲线 ====================
def exp2_dose_curve(model, tokenizer, device, model_name, W_U, best_per_cat):
    """对代表性类别做正反向注入剂量曲线"""
    plog("=== Exp2: 正反向剂量曲线 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    neighbors = BEST_NEIGHBORS[model_name]
    
    dose_cats = DOSE_CATEGORIES
    dose_results = {}
    
    for cat_name in dose_cats:
        if cat_name not in best_per_cat:
            plog(f"  Skip {cat_name} (no best layer)")
            continue
        
        best_layer = best_per_cat[cat_name]["layer"]
        plog(f"  {cat_name} @ L{best_layer}...")
        
        # 获取该层raw direction
        raw_dirs = get_category_residuals_at_layer(
            model, tokenizer, device, model_name,
            categories=CATEGORIES_TRAIN, n_obj=4, target_layer=best_layer
        )
        
        if cat_name not in raw_dirs:
            continue
        
        target_vec = raw_dirs[cat_name]
        target_idx = cat_names.index(cat_name)
        basis_vecs = [raw_dirs[n] for n in neighbors[cat_name] if n in raw_dirs]
        spec_vec = qr_orthogonalize(target_vec, basis_vecs) if basis_vecs else target_vec.copy()
        
        spec_norm = np.linalg.norm(spec_vec)
        if spec_norm < 1e-6:
            continue
        
        cat_results = {}
        for sign_label, sign in [("+specific", 1.0), ("-specific", -1.0)]:
            for scale in DOSE_SCALES:
                ivec = torch.tensor(spec_vec * sign * scale, dtype=torch.float32)
                
                # 用3个test对象测试
                test_objs = CATEGORIES[cat_name][4:7]
                template = RELATION_TEMPLATES["kind_of"]
                dcf_deltas = []
                
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
                    baseline_dcf = logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer)
                    
                    # 注入
                    h2 = layers_list[best_layer].register_forward_hook(make_inject_hook(ivec, pos))
                    cap2 = {}
                    h3 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap2, "resid"))
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    h2.remove()
                    h3.remove()
                    if "resid" not in cap2:
                        continue
                    inject_dcf = logit_lens_dcf(cap2["resid"][0, pos].numpy(), W_U, tokenizer)
                    dcf_deltas.append(inject_dcf - baseline_dcf)
                
                if dcf_deltas:
                    avg_delta = np.mean(dcf_deltas, axis=0)
                    sel = compute_selectivity(avg_delta, target_idx)
                    
                    # 计算margin: target与最近竞争类别的差距
                    target_abs = abs(avg_delta[target_idx])
                    others_abs = [abs(avg_delta[i]) for i in range(len(cat_names)) if i != target_idx]
                    margin = target_abs - max(others_abs) if others_abs else 0
                    
                    # 计算entropy proxy: DCF的方差
                    dcf_std = float(np.std(avg_delta))
                    
                    key = f"{sign_label}_s{scale}"
                    cat_results[key] = {
                        "sign": sign_label,
                        "scale": scale,
                        "selectivity": sel,
                        "target_dcf_delta": float(avg_delta[target_idx]),
                        "dcf_delta": {cat_names[i]: float(avg_delta[i]) for i in range(len(cat_names))},
                        "margin": float(margin),
                        "dcf_std": dcf_std,
                    }
        
        dose_results[cat_name] = cat_results
        plog(f"    {cat_name}: {len(cat_results)} dose points")
        gc.collect()
    
    return dose_results


# ==================== Exp3: 边界残差必要性 ====================
def exp3_necessity(model, tokenizer, device, model_name, W_U, best_per_cat):
    """从自然输入中移除B_c，测试目标类别是否下降"""
    plog("=== Exp3: 边界残差必要性 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    neighbors = BEST_NEIGHBORS[model_name]
    
    necessity_cats = ["fruit", "vehicle", "food", "plant", "animal"]
    necessity_results = {}
    
    for cat_name in necessity_cats:
        if cat_name not in best_per_cat:
            continue
        best_layer = best_per_cat[cat_name]["layer"]
        plog(f"  {cat_name} @ L{best_layer}...")
        
        # 获取该层specific方向
        raw_dirs = get_category_residuals_at_layer(
            model, tokenizer, device, model_name,
            categories=CATEGORIES_TRAIN, n_obj=4, target_layer=best_layer
        )
        if cat_name not in raw_dirs:
            continue
        
        target_vec = raw_dirs[cat_name]
        target_idx = cat_names.index(cat_name)
        basis_vecs = [raw_dirs[n] for n in neighbors[cat_name] if n in raw_dirs]
        spec_vec = qr_orthogonalize(target_vec, basis_vecs) if basis_vecs else target_vec.copy()
        
        spec_norm = np.linalg.norm(spec_vec)
        if spec_norm < 1e-6:
            continue
        
        # 归一化方向
        b_hat = spec_vec / spec_norm
        
        # 测试3种移除强度
        for remove_scale in [0.5, 1.0, 2.0]:
            # 用test对象
            test_objs = CATEGORIES[cat_name][4:7]
            template = RELATION_TEMPLATES["kind_of"]
            dcf_deltas = []
            
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
                
                # 计算自然投影到B_c上的分量大小
                proj_coeff = float(np.dot(baseline_resid, b_hat))
                
                # 移除: 减去在B_c方向上的投影 × scale
                h2 = layers_list[best_layer].register_forward_hook(
                    make_remove_hook(spec_vec, pos, scale=remove_scale)
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
                key = f"{cat_name}_remove_s{remove_scale}"
                necessity_results[key] = {
                    "category": cat_name,
                    "remove_scale": remove_scale,
                    "best_layer": best_layer,
                    "target_dcf_delta": float(avg_delta[target_idx]),
                    "dcf_delta": {cat_names[i]: float(avg_delta[i]) for i in range(len(cat_names))},
                    "spec_norm": float(spec_norm),
                    "natural_proj_coeff": float(proj_coeff) if 'proj_coeff' in dir() else None,
                }
                plog(f"    remove_s{remove_scale}: target_Δ={avg_delta[target_idx]:.2f}, "
                      f"max_other_Δ={max(abs(avg_delta[i]) for i in range(len(cat_names)) if i != target_idx):.2f}")
        
        gc.collect()
    
    return necessity_results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    plog(f"Phase 482: Layer Map + Dose Curve + Necessity | Model={model_name} | Round={round_num}")
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    plog(f"  W_U: {W_U.shape}")
    
    results = {
        "phase": 482,
        "model": model_name,
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "core_question": "Category-layer map, dose curves, and boundary residual necessity",
        "model_info": {"class": info.model_class, "n_layers": info.n_layers, "d_model": info.d_model},
    }
    
    # Exp1: 类别-层位图谱
    t0 = time.time()
    exp1_data = exp1_layer_map(model, tokenizer, device, model_name, W_U)
    results["exp1_layer_map"] = exp1_data
    plog(f"Exp1 done in {time.time()-t0:.1f}s")
    
    best_per_cat = exp1_data["best_per_cat"]
    
    # Exp2: 正反向剂量曲线
    t0 = time.time()
    exp2_data = exp2_dose_curve(model, tokenizer, device, model_name, W_U, best_per_cat)
    results["exp2_dose_curve"] = exp2_data
    plog(f"Exp2 done in {time.time()-t0:.1f}s")
    
    # Exp3: 边界残差必要性
    t0 = time.time()
    exp3_data = exp3_necessity(model, tokenizer, device, model_name, W_U, best_per_cat)
    results["exp3_necessity"] = exp3_data
    plog(f"Exp3 done in {time.time()-t0:.1f}s")
    
    # 保存结果
    out_dir = "results/glm5"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"phase482_{model_name}_r{round_num}.json")
    
    # 转换numpy类型
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
