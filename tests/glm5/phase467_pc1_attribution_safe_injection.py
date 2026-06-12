"""
Phase 467: PC1功能归因、白化空间新方向、DS7B安全注入与生成质量闭环
===================================================================
核心实验:
1. Exp1: PC1功能归因 — PC1到底对应什么? (位置/范数/语言/类别/熵)
2. Exp2: 白化空间新方向构造 — 在白化空间做PCA差分, 映射回原始空间
3. Exp3: DS7B微扰敏感性地图 — 找到DS7B的安全注入窗口
4. Exp4: 去主轴+去混叠联合方向 — 最优方向组合
5. Exp5: 生成质量系统性验证 — 所有有效注入必须通过生成验证

用法: python tests/glm5/phase467_pc1_attribution_safe_injection.py qwen3 1
      python tests/glm5/phase467_pc1_attribution_safe_injection.py deepseek7b 2
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
    "fruit":   ["apple", "banana", "orange", "grape", "pear", "peach", "lemon", "mango"],
    "animal":  ["dog", "cat", "horse", "lion", "bear", "rabbit", "cow", "tiger"],
    "tool":    ["hammer", "knife", "wrench", "saw", "drill", "axe", "shovel", "scissors"],
    "vehicle": ["car", "bus", "bicycle", "truck", "train", "boat", "plane", "scooter"],
    "clothing":["shirt", "dress", "hat", "coat", "sock", "glove", "scarf", "boot"],
    "furniture":["chair", "table", "desk", "sofa", "bed", "shelf", "lamp", "cabinet"],
}

FAMILIES_EN = {
    "fruit":    ["fruit", "produce", "crop"],
    "animal":   ["animal", "creature", "beast"],
    "tool":     ["tool", "implement", "device"],
    "vehicle":  ["vehicle", "transport", "automobile"],
    "clothing": ["clothing", "attire", "wear"],
    "furniture":["furniture", "furnishing", "fixture"],
}

TEMPLATES_EN = {"is_a": "The {obj} is a kind of"}
TEMPLATES_ZH = {"is_a": "{obj}是一种"}

# 扩展模板, 用于PC1归因
TEMPLATES_VARIANT = [
    "The {obj} is a kind of",
    "A {obj} is considered a",
    "{obj} belongs to the category of",
]

# 不同语言模板
TEMPLATES_BY_LANG = {
    "en": "The {obj} is a kind of",
    "zh": "{obj}是一种",
    "fr": "Le {obj} est une sorte de",
    "de": "Der {obj} ist eine Art von",
}

ROUNDS = {
    1: {k: v[:5] for k, v in CATEGORIES.items()},   # R1: 5对象
    2: {k: v[:8] for k, v in CATEGORIES.items()},   # R2: 8对象(确认)
}


# ==================== 模型加载 ====================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog(f"Loading {model_name} (bfloat16 + device_map=auto + flash_attn)...")
    
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
        plog(f"  flash_attention_2 loaded OK")
    except Exception as e:
        plog(f"  flash_attention_2 failed ({e}), falling back to eager")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="eager",
        )
    
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    plog(f"  {model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


# ==================== 基础工具函数 ====================
def get_residual_at_layer_pos(model, tokenizer, prompt, layer_idx, device, pos=-1):
    """提取指定层指定位置的残差流向量"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    captured = {}
    layers = get_layers(model)
    
    def hook_fn(module, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            captured['resid'] = input[0].detach().float().cpu()
    
    h = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    h.remove()
    
    if 'resid' in captured:
        seq_len = attention_mask.sum().item()
        if pos == -1:
            pos = seq_len - 1
        return captured['resid'][0, pos].numpy(), seq_len
    return None, 0


def get_residual_full_seq(model, tokenizer, prompt, layer_idx, device):
    """提取指定层所有位置的残差流"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    captured = {}
    layers = get_layers(model)
    
    def hook_fn(module, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            captured['resid'] = input[0].detach().float().cpu()
    
    h = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    h.remove()
    
    if 'resid' in captured:
        seq_len = attention_mask.sum().item()
        return captured['resid'][0, :seq_len].numpy(), seq_len
    return None, 0


def get_final_logits(model, tokenizer, prompt, device):
    """获取最后一层的logits"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits[0, -1].float().cpu().numpy()


def run_with_additive_patch(model, tokenizer, prompt, device, patch_layer, delta_vec):
    """加法patch: 在patch_layer的输出中加上delta_vec"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = attention_mask.sum().item()
    
    layers = get_layers(model)
    delta_tensor = torch.tensor(delta_vec, dtype=torch.float32, device=device)
    
    patched = [False]
    def make_hook():
        def hook(module, input, output):
            if not patched[0]:
                patched[0] = True
                if isinstance(output, tuple):
                    out_tensor = output[0].clone()
                    out_tensor[0, seq_len - 1, :] += delta_tensor.to(out_tensor.dtype)
                    return (out_tensor,) + output[1:]
                else:
                    out_tensor = output.clone()
                    out_tensor[0, seq_len - 1, :] += delta_tensor.to(out_tensor.dtype)
                    return out_tensor
            return None
        return hook
    
    h = layers[patch_layer].register_forward_hook(make_hook())
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    h.remove()
    return out.logits[0, -1].float().cpu().numpy()


def run_with_additive_patch_generate(model, tokenizer, prompt, device, patch_layer, delta_vec,
                                      max_new_tokens=20, do_sample=False):
    """加法patch + 生成"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = attention_mask.sum().item()
    
    layers = get_layers(model)
    delta_tensor = torch.tensor(delta_vec, dtype=torch.float32, device=device)
    
    patched = [False]
    def make_hook():
        def hook(module, input, output):
            if not patched[0]:
                patched[0] = True
                if isinstance(output, tuple):
                    out_tensor = output[0].clone()
                    out_tensor[0, seq_len - 1, :] += delta_tensor.to(out_tensor.dtype)
                    return (out_tensor,) + output[1:]
                else:
                    out_tensor = output.clone()
                    out_tensor[0, seq_len - 1, :] += delta_tensor.to(out_tensor.dtype)
                    return out_tensor
            return None
        return hook
    
    h = layers[patch_layer].register_forward_hook(make_hook())
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, repetition_penalty=1.2)
    with torch.no_grad():
        gen_ids = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
    h.remove()
    
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return gen_text


def compute_en_family_margin(logits, tokenizer, target_cat, compete_cats):
    """计算英文候选族边际"""
    target_words = FAMILIES_EN.get(target_cat, [])
    compete_words = []
    for cc in compete_cats:
        compete_words.extend(FAMILIES_EN.get(cc, []))
    
    vocab = tokenizer.get_vocab()
    target_logits, compete_logits = [], []
    for w in target_words:
        w_clean = w.strip()
        if w_clean in vocab:
            target_logits.append(float(logits[vocab[w_clean]]))
        elif f" {w_clean}" in vocab:
            target_logits.append(float(logits[vocab[f" {w_clean}"]]))
    for w in compete_words:
        w_clean = w.strip()
        if w_clean in vocab:
            compete_logits.append(float(logits[vocab[w_clean]]))
        elif f" {w_clean}" in vocab:
            compete_logits.append(float(logits[vocab[f" {w_clean}"]]))
    
    if not target_logits or not compete_logits:
        return 0.0, 0.0, 0.0
    t_mean = float(np.mean(target_logits))
    c_mean = float(np.mean(compete_logits))
    return t_mean - c_mean, t_mean, c_mean


def compute_selectivity(logits_base, logits_patch, tokenizer, target_cat, compete_cats):
    """计算selectivity"""
    margin_base, _, _ = compute_en_family_margin(logits_base, tokenizer, target_cat, compete_cats)
    margin_patch, _, _ = compute_en_family_margin(logits_patch, tokenizer, target_cat, compete_cats)
    return margin_patch - margin_base


def logit_entropy(logits_vec):
    """计算logit分布的熵"""
    log_probs = logits_vec - np.max(logits_vec)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs)))
    return -float(np.sum(np.exp(log_probs) * log_probs))


def logit_kl(logits_p, logits_q):
    """KL散度 KL(p||q)"""
    log_p = logits_p - np.max(logits_p)
    log_p = log_p - np.log(np.sum(np.exp(log_p)))
    log_q = logits_q - np.max(logits_q)
    log_q = log_q - np.log(np.sum(np.exp(log_q)))
    return float(np.sum(np.exp(log_p) * (log_p - log_q)))


def estimate_covariance_and_pcs(model, tokenizer, prompts, layer_idx, device):
    """估计指定层的协方差矩阵和主成分"""
    all_vecs = []
    for prompt in prompts:
        resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, layer_idx, device)
        if resid is not None:
            all_vecs.append(resid)
    
    if len(all_vecs) < 6:
        return None, None, None, None
    
    vecs_matrix = np.array(all_vecs)
    vec_mean = np.mean(vecs_matrix, axis=0)
    vecs_centered = vecs_matrix - vec_mean
    
    cov = np.cov(vecs_centered.T)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        return None, None, None, None
    
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        # 按特征值排序(降序)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        return cov, eigvals, eigvecs, vec_mean
    except:
        return None, None, None, None


def compute_natural_delta_norm(model, tokenizer, obj_dict, layer_idx, device, cat="animal", n=5):
    """计算某层某类别的自然delta范数"""
    objs = obj_dict.get(cat, [])[:n]
    vecs = []
    for obj in objs:
        prompt = TEMPLATES_EN["is_a"].format(obj=obj)
        resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, layer_idx, device)
        if resid is not None:
            vecs.append(resid)
    
    if len(vecs) < 2:
        return 1.0
    
    center = np.mean(vecs, axis=0)
    delta_norms = [np.linalg.norm(v - center) for v in vecs]
    return float(np.mean(delta_norms))


# ==================== Exp1: PC1功能归因 ====================
def exp1_pc1_attribution(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    归因PC1到底对应什么:
    1. 位置相关性: PC1是否沿序列位置系统性变化
    2. 范数相关性: PC1投影是否与残差范数相关
    3. 语言相关性: PC1是否在不同语言模板下变化
    4. 类别相关性: PC1是否在不同类别间变化
    5. logit熵相关性: PC1投影是否与输出不确定性相关
    6. RMSNorm尺度相关性: PC1投影是否与RMSNorm后范数相关
    """
    plog("=== Exp1: PC1功能归因 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    # 选择3个关键层
    key_layers = [
        n_layers // 6,
        n_layers // 3,
        n_layers // 2,
    ]
    key_layers = sorted(set([l for l in key_layers if l < n_layers]))
    
    results = {}
    
    for layer_idx in key_layers:
        plog(f"  Layer L{layer_idx}...")
        layer_result = {}
        
        # ---- 1. 收集大量自然激活 ----
        all_prompts = []
        prompt_labels = []  # (category, template_type, lang)
        
        # 多类别 × 多模板
        for cat in ["fruit", "animal", "vehicle", "tool", "furniture"]:
            objs = obj_dict.get(cat, [])[:4]
            for obj in objs:
                for tmpl_key, tmpl in TEMPLATES_EN.items():
                    p = tmpl.format(obj=obj)
                    all_prompts.append(p)
                    prompt_labels.append((cat, tmpl_key, "en"))
        
        # 收集残差
        all_vecs = []
        for p in all_prompts:
            resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
            if resid is not None:
                all_vecs.append(resid)
        
        if len(all_vecs) < 10:
            plog(f"    L{layer_idx}: Not enough vectors, skip")
            continue
        
        # ---- 2. 估计PCA ----
        vecs_matrix = np.array(all_vecs)
        vec_mean = np.mean(vecs_matrix, axis=0)
        vecs_centered = vecs_matrix - vec_mean
        
        cov = np.cov(vecs_centered.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
        except:
            plog(f"    L{layer_idx}: PCA failed, skip")
            continue
        
        pc1 = eigvecs[:, 0]
        pc2 = eigvecs[:, 1]
        pc3 = eigvecs[:, 2]
        
        layer_result["pc_eigenvalues"] = {
            "pc1": float(eigvals[0]),
            "pc2": float(eigvals[1]),
            "pc3": float(eigvals[2]),
            "pc1_ratio": float(eigvals[0] / max(sum(eigvals[:50]), 1e-10)),
        }
        plog(f"    PC1 eigval={eigvals[0]:.2f}, PC2={eigvals[1]:.2f}, ratio={eigvals[0]/max(sum(eigvals[:50]),1e-10):.4f}")
        
        # ---- 3. PC1投影与类别相关性 ----
        cat_projections = {}
        for cat in ["fruit", "animal", "vehicle", "tool", "furniture"]:
            objs = obj_dict.get(cat, [])[:4]
            proj_vals = []
            for obj in objs:
                p = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
                if resid is not None:
                    centered = resid - vec_mean
                    proj_vals.append(float(np.dot(centered, pc1)))
            if proj_vals:
                cat_projections[cat] = {
                    "mean": float(np.mean(proj_vals)),
                    "std": float(np.std(proj_vals)),
                    "values": [round(v, 4) for v in proj_vals],
                }
        
        layer_result["pc1_vs_category"] = cat_projections
        
        # 检查PC1是否区分类别
        cat_means = [v["mean"] for v in cat_projections.values()]
        if len(cat_means) >= 2:
            cat_spread = float(np.std(cat_means))
            layer_result["pc1_category_spread"] = round(cat_spread, 4)
            plog(f"    PC1 category spread: {cat_spread:.4f}")
        
        # ---- 4. PC1投影与残差范数相关性 ----
        norm_projs = []
        for i, vec in enumerate(all_vecs):
            centered = vec - vec_mean
            pc1_proj = float(np.dot(centered, pc1))
            vec_norm = float(np.linalg.norm(centered))
            norm_projs.append((pc1_proj, vec_norm))
        
        pc1_projs = [x[0] for x in norm_projs]
        vec_norms = [x[1] for x in norm_projs]
        
        # Pearson相关
        if len(pc1_projs) > 2 and np.std(pc1_projs) > 1e-10 and np.std(vec_norms) > 1e-10:
            corr_norm = float(np.corrcoef(pc1_projs, vec_norms)[0, 1])
        else:
            corr_norm = 0.0
        
        layer_result["pc1_vs_norm_correlation"] = round(corr_norm, 4)
        plog(f"    PC1 vs norm correlation: {corr_norm:.4f}")
        
        # ---- 5. PC1投影与序列位置相关性 ----
        # 用一个较长提示检查不同位置的PC1投影
        test_prompt = "The apple is a kind of fruit and the banana is also"
        resid_full, seq_len = get_residual_full_seq(model, tokenizer, test_prompt, layer_idx, device)
        
        if resid_full is not None and seq_len > 3:
            pos_projs = []
            for pos in range(seq_len):
                centered = resid_full[pos] - vec_mean
                pos_projs.append(float(np.dot(centered, pc1)))
            
            # 位置与PC1投影的相关性
            positions = list(range(seq_len))
            if np.std(pos_projs) > 1e-10:
                corr_pos = float(np.corrcoef(positions, pos_projs)[0, 1])
            else:
                corr_pos = 0.0
            
            layer_result["pc1_vs_position"] = {
                "correlation": round(corr_pos, 4),
                "projections": [round(v, 4) for v in pos_projs],
            }
            plog(f"    PC1 vs position correlation: {corr_pos:.4f}")
        
        # ---- 6. PC1投影与logit熵相关性 ----
        entropy_projs = []
        for i, p in enumerate(all_prompts[:15]):  # 限制数量
            logits = get_final_logits(model, tokenizer, p, device)
            ent = logit_entropy(logits)
            centered = all_vecs[i] - vec_mean
            pc1_proj = float(np.dot(centered, pc1))
            entropy_projs.append((pc1_proj, ent))
        
        pc1_projs_ent = [x[0] for x in entropy_projs]
        entropies = [x[1] for x in entropy_projs]
        
        if len(pc1_projs_ent) > 2 and np.std(pc1_projs_ent) > 1e-10 and np.std(entropies) > 1e-10:
            corr_ent = float(np.corrcoef(pc1_projs_ent, entropies)[0, 1])
        else:
            corr_ent = 0.0
        
        layer_result["pc1_vs_entropy_correlation"] = round(corr_ent, 4)
        plog(f"    PC1 vs entropy correlation: {corr_ent:.4f}")
        
        # ---- 7. PC1与W_U读出方向的对齐 ----
        W_U = get_W_U(model, model_name)
        if W_U is not None:
            # W_U的top奇异向量
            try:
                U, S, Vt = np.linalg.svd(W_U, full_matrices=False)
                wu_pc1 = U[:, 0]  # W_U的第一左奇异向量
                wu_pc2 = U[:, 1]
                
                cos_pc1_wu1 = float(abs(np.dot(pc1, wu_pc1)))
                cos_pc1_wu2 = float(abs(np.dot(pc1, wu_pc2)))
                cos_pc2_wu1 = float(abs(np.dot(pc2, wu_pc1)))
                
                layer_result["pc1_vs_W_U_alignment"] = {
                    "cos_pc1_WU_pc1": round(cos_pc1_wu1, 4),
                    "cos_pc1_WU_pc2": round(cos_pc1_wu2, 4),
                    "cos_pc2_WU_pc1": round(cos_pc2_wu1, 4),
                }
                plog(f"    PC1 vs W_U_pc1: cos={cos_pc1_wu1:.4f}, vs W_U_pc2: cos={cos_pc1_wu2:.4f}")
            except:
                layer_result["pc1_vs_W_U_alignment"] = {"error": "SVD failed"}
        
        # ---- 8. PC1去除后的效果: 多类别全面测试 ----
        test_cats = ["fruit", "animal", "vehicle", "tool", "furniture"]
        no_pc1_effects = {}
        
        for cat in test_cats:
            objs = obj_dict.get(cat, [])[:3]
            other_cat = "fruit" if cat != "fruit" else "animal"
            other_objs = obj_dict.get(other_cat, [])[:3]
            
            cat_vecs, other_vecs = [], []
            for obj in objs:
                p = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
                if resid is not None:
                    cat_vecs.append(resid)
            for obj in other_objs:
                p = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
                if resid is not None:
                    other_vecs.append(resid)
            
            if len(cat_vecs) < 2 or len(other_vecs) < 2:
                continue
            
            cat_center = np.mean(cat_vecs, axis=0)
            other_center = np.mean(other_vecs, axis=0)
            diff = cat_center - other_center
            diff_norm = np.linalg.norm(diff)
            if diff_norm < 1e-10:
                continue
            diff_dir = diff / diff_norm
            
            # 去PC1方向
            diff_no_pc1 = diff - np.dot(diff, pc1) * pc1
            diff_no_pc1_norm = np.linalg.norm(diff_no_pc1)
            if diff_no_pc1_norm < 1e-10:
                continue
            diff_no_pc1_dir = diff_no_pc1 / diff_no_pc1_norm
            
            # cos(raw, no_pc1)
            cos_raw_nopc1 = float(np.dot(diff_dir, diff_no_pc1_dir))
            
            # cos(diff, pc1) — 差分方向与PC1的对齐程度
            cos_diff_pc1 = float(np.dot(diff_dir, pc1))
            
            no_pc1_effects[cat] = {
                "cos_raw_nopc1": round(cos_raw_nopc1, 4),
                "cos_diff_pc1": round(cos_diff_pc1, 4),
                "diff_projection_on_pc1": round(float(np.dot(diff_dir, pc1)), 4),
            }
            plog(f"    {cat}: cos(diff,PC1)={cos_diff_pc1:.4f}, cos(raw,nopc1)={cos_raw_nopc1:.4f}")
        
        layer_result["no_pc1_effects"] = no_pc1_effects
        
        # ---- 9. 去除前3个PC的效果 ----
        no_pc3_effects = {}
        for cat in ["fruit", "animal", "vehicle"]:
            objs = obj_dict.get(cat, [])[:3]
            other_cat = "fruit" if cat != "fruit" else "animal"
            other_objs = obj_dict.get(other_cat, [])[:3]
            
            cat_vecs, other_vecs = [], []
            for obj in objs:
                p = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
                if resid is not None:
                    cat_vecs.append(resid)
            for obj in other_objs:
                p = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
                if resid is not None:
                    other_vecs.append(resid)
            
            if len(cat_vecs) < 2 or len(other_vecs) < 2:
                continue
            
            diff = np.mean(cat_vecs, axis=0) - np.mean(other_vecs, axis=0)
            diff_norm = np.linalg.norm(diff)
            if diff_norm < 1e-10:
                continue
            diff_dir = diff / diff_norm
            
            # 去前3个PC
            diff_no_pc3 = diff.copy()
            for k in range(min(3, len(eigvals))):
                diff_no_pc3 -= np.dot(diff_no_pc3, eigvecs[:, k]) * eigvecs[:, k]
            diff_no_pc3_norm = np.linalg.norm(diff_no_pc3)
            if diff_no_pc3_norm < 1e-10:
                no_pc3_effects[cat] = {"error": "zero after removing top3 PCs"}
                continue
            diff_no_pc3_dir = diff_no_pc3 / diff_no_pc3_norm
            
            cos_raw_nopc3 = float(np.dot(diff_dir, diff_no_pc3_dir))
            no_pc3_effects[cat] = {"cos_raw_nopc3": round(cos_raw_nopc3, 4)}
        
        layer_result["no_top3_pc_effects"] = no_pc3_effects
        
        results[f"L{layer_idx}"] = layer_result
    
    return results


# ==================== Exp2: 白化空间新方向构造 ====================
def exp2_whitened_new_directions(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    在白化空间中构造新的差分方向, 映射回原始空间
    
    方法:
    1. 估计白化矩阵 W = Σ^{-1/2}
    2. 在白化空间做类别中心PCA差分
    3. 在白化空间做Gram-Schmidt正交化
    4. 映射回原始空间测试
    
    关键: 不再是"白化已有方向再回映射", 而是"在白化空间中构造新方向"
    """
    plog("=== Exp2: 白化空间新方向构造 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    key_layers = [
        n_layers // 6,
        n_layers // 3,
        n_layers // 2,
    ]
    key_layers = sorted(set([l for l in key_layers if l < n_layers]))
    
    test_cats = ["animal", "vehicle", "fruit"]
    results = {}
    
    for layer_idx in key_layers:
        plog(f"  Layer L{layer_idx}...")
        layer_result = {}
        
        # 收集所有类别的残差
        cat_vecs_dict = {}
        for cat in ["fruit", "animal", "vehicle", "tool", "furniture"]:
            objs = obj_dict.get(cat, [])[:5]
            vecs = []
            for obj in objs:
                p = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
                if resid is not None:
                    vecs.append(resid)
            if len(vecs) >= 2:
                cat_vecs_dict[cat] = vecs
        
        if len(cat_vecs_dict) < 3:
            plog(f"    Not enough categories, skip")
            continue
        
        # 计算所有向量的均值和协方差
        all_vecs = []
        for cat_vecs in cat_vecs_dict.values():
            all_vecs.extend(cat_vecs)
        
        vecs_matrix = np.array(all_vecs)
        grand_mean = np.mean(vecs_matrix, axis=0)
        vecs_centered = vecs_matrix - grand_mean
        
        cov = np.cov(vecs_centered.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
        except:
            continue
        
        n_pca = min(50, len(eigvals))
        eigvals_top = eigvals[:n_pca]
        eigvecs_top = eigvecs[:, :n_pca]
        sqrt_eigvals = np.sqrt(np.maximum(eigvals_top, 1e-10))
        
        # 白化矩阵: W = eigvecs_top @ diag(1/sqrt(eigvals_top))
        # Σ^{-1/2} ≈ W (前n_pca成分)
        whitening_matrix = eigvecs_top / sqrt_eigvals[np.newaxis, :]  # [d_model, n_pca]
        
        # ---- 方法1: 白化空间类别中心PCA ----
        # 在白化空间计算各类别中心
        cat_centers_white = {}
        for cat, vecs in cat_vecs_dict.items():
            center = np.mean(vecs, axis=0)
            center_white = whitening_matrix.T @ (center - grand_mean)  # [n_pca]
            cat_centers_white[cat] = center_white
        
        # 在白化空间做差分
        ref_cat = "fruit"
        if ref_cat not in cat_centers_white:
            continue
        
        ref_center_white = cat_centers_white[ref_cat]
        
        for cat in test_cats:
            if cat not in cat_centers_white or cat == ref_cat:
                continue
            
            plog(f"    Category: {cat}")
            cat_result = {}
            
            # 白化空间差分
            diff_white = cat_centers_white[cat] - ref_center_white
            diff_white_norm = np.linalg.norm(diff_white)
            if diff_white_norm < 1e-10:
                continue
            diff_white_dir = diff_white / diff_white_norm
            
            # ---- 映射回原始空间(使用Σ^{1/2}) ----
            # v_raw = Σ^{1/2} @ diff_white_dir
            Sigma_half = eigvecs_top * sqrt_eigvals[np.newaxis, :]  # [d_model, n_pca]
            v_from_white = Sigma_half @ diff_white_dir  # [d_model]
            v_from_white_norm = np.linalg.norm(v_from_white)
            if v_from_white_norm < 1e-10:
                continue
            v_from_white_dir = v_from_white / v_from_white_norm
            
            # ---- 原始空间差分(对照) ----
            cat_center_raw = np.mean(cat_vecs_dict[cat], axis=0)
            ref_center_raw = np.mean(cat_vecs_dict[ref_cat], axis=0)
            diff_raw = cat_center_raw - ref_center_raw
            diff_raw_norm = np.linalg.norm(diff_raw)
            if diff_raw_norm < 1e-10:
                continue
            diff_raw_dir = diff_raw / diff_raw_norm
            
            # ---- 去PC1方向 ----
            pc1 = eigvecs[:, 0]
            diff_no_pc1 = diff_raw - np.dot(diff_raw, pc1) * pc1
            diff_no_pc1_norm = np.linalg.norm(diff_no_pc1)
            if diff_no_pc1_norm > 1e-10:
                diff_no_pc1_dir = diff_no_pc1 / diff_no_pc1_norm
            else:
                diff_no_pc1_dir = diff_raw_dir
            
            # ---- 白化空间去第1白化主轴方向 ----
            # 在白化空间中, 第1个白化主轴对应原始空间最大方差方向
            # 去掉白化空间第1轴的差分
            diff_white_no1 = diff_white.copy()
            diff_white_no1[0] = 0  # 去掉第1个白化主轴分量
            diff_white_no1_norm = np.linalg.norm(diff_white_no1)
            if diff_white_no1_norm > 1e-10:
                diff_white_no1_dir = diff_white_no1 / diff_white_no1_norm
                v_white_no1 = Sigma_half @ diff_white_no1_dir
                v_white_no1_norm = np.linalg.norm(v_white_no1)
                if v_white_no1_norm > 1e-10:
                    v_white_no1_dir = v_white_no1 / v_white_no1_norm
                else:
                    v_white_no1_dir = None
            else:
                v_white_no1_dir = None
            
            # 计算方向间余弦
            cos_raw_white = float(np.dot(diff_raw_dir, v_from_white_dir))
            cos_raw_nopc1 = float(np.dot(diff_raw_dir, diff_no_pc1_dir))
            
            cat_result["direction_cosines"] = {
                "raw_vs_whitened_new": round(cos_raw_white, 4),
                "raw_vs_nopc1": round(cos_raw_nopc1),
            }
            
            if v_white_no1_dir is not None:
                cos_raw_whiteno1 = float(np.dot(diff_raw_dir, v_white_no1_dir))
                cat_result["direction_cosines"]["raw_vs_white_no1"] = round(cos_raw_whiteno1, 4)
            
            # ---- 注入测试 ----
            test_obj = obj_dict.get(cat, [])[0]
            if not test_obj:
                continue
            
            prompt = TEMPLATES_EN["is_a"].format(obj=test_obj)
            logits_base = get_final_logits(model, tokenizer, prompt, device)
            compete_cats = [c for c in ["animal", "tool", "vehicle", "fruit"] if c != cat][:3]
            
            # 自然delta范数
            mean_delta_norm = compute_natural_delta_norm(
                model, tokenizer, obj_dict, layer_idx, device, cat, 5)
            
            # 测试所有方向, 固定norm_ratio=1
            inject_norm = mean_delta_norm
            
            for method_name, method_dir in [
                ("raw", diff_raw_dir),
                ("whitened_new", v_from_white_dir),
                ("no_pc1", diff_no_pc1_dir),
            ]:
                if method_dir is None:
                    continue
                
                beta = inject_norm / max(np.linalg.norm(method_dir), 1e-10)
                inject_vec = beta * method_dir
                logits_patch = run_with_additive_patch(model, tokenizer, prompt, device, layer_idx, inject_vec)
                
                sel = compute_selectivity(logits_base, logits_patch, tokenizer, cat, compete_cats)
                kl = logit_kl(logits_patch, logits_base)
                
                cat_result[method_name] = {
                    "selectivity": round(sel, 4),
                    "kl_div": round(kl, 4),
                    "beta": round(beta, 4),
                }
                plog(f"      {method_name}: sel={sel:.4f}, kl={kl:.4f}")
            
            # 白化空间去第1轴方向注入
            if v_white_no1_dir is not None:
                beta = inject_norm / max(np.linalg.norm(v_white_no1_dir), 1e-10)
                inject_vec = beta * v_white_no1_dir
                logits_patch = run_with_additive_patch(model, tokenizer, prompt, device, layer_idx, inject_vec)
                
                sel = compute_selectivity(logits_base, logits_patch, tokenizer, cat, compete_cats)
                kl = logit_kl(logits_patch, logits_base)
                
                cat_result["white_no_pc1"] = {
                    "selectivity": round(sel, 4),
                    "kl_div": round(kl, 4),
                    "beta": round(beta, 4),
                }
                plog(f"      white_no_pc1: sel={sel:.4f}, kl={kl:.4f}")
            
            layer_result[cat] = cat_result
        
        results[f"L{layer_idx}"] = layer_result
    
    return results


# ==================== Exp3: DS7B微扰敏感性地图 ====================
def exp3_ds7b_sensitivity_map(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    系统扫描DS7B的安全注入窗口
    
    对所有模型都做, 但DS7B重点关注小ratio:
    - ratio = 0.05, 0.1, 0.25, 0.5, 1.0
    - 在多个层测试
    - 同时检查selectivity和生成质量
    """
    plog("=== Exp3: 微扰敏感性地图 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    # 根据模型选择不同的ratio范围
    if model_name == "deepseek7b":
        target_ratios = [0.05, 0.1, 0.25, 0.5, 1.0]
    else:
        target_ratios = [0.25, 0.5, 1.0, 2.0]
    
    key_layers = [
        n_layers // 6,
        n_layers // 3,
        n_layers // 2,
    ]
    key_layers = sorted(set([l for l in key_layers if l < n_layers]))
    
    test_cases = [
        ("fruit", "apple", ["animal", "tool"]),
        ("animal", "dog", ["fruit", "tool"]),
        ("vehicle", "car", ["fruit", "animal"]),
    ]
    
    results = {}
    
    for layer_idx in key_layers:
        plog(f"  Layer L{layer_idx}...")
        layer_result = {}
        
        for cat, test_obj, compete_cats in test_cases:
            if test_obj not in obj_dict.get(cat, []):
                continue
            
            plog(f"    {cat}/{test_obj}...")
            
            # 构造差分方向
            other_cat = compete_cats[0]
            cat_objs = obj_dict.get(cat, [])[:4]
            other_objs = obj_dict.get(other_cat, [])[:4]
            
            cat_vecs, other_vecs = [], []
            for obj in cat_objs:
                p = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
                if resid is not None:
                    cat_vecs.append(resid)
            for obj in other_objs:
                p = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
                if resid is not None:
                    other_vecs.append(resid)
            
            if len(cat_vecs) < 2 or len(other_vecs) < 2:
                continue
            
            diff = np.mean(cat_vecs, axis=0) - np.mean(other_vecs, axis=0)
            diff_norm = np.linalg.norm(diff)
            if diff_norm < 1e-10:
                continue
            diff_dir = diff / diff_norm
            
            # 自然delta范数
            cat_center = np.mean(cat_vecs, axis=0)
            delta_norms = [np.linalg.norm(v - cat_center) for v in cat_vecs]
            mean_delta_norm = float(np.mean(delta_norms))
            
            prompt = TEMPLATES_EN["is_a"].format(obj=test_obj)
            logits_base = get_final_logits(model, tokenizer, prompt, device)
            
            # 基准生成
            gen_base = run_with_additive_patch_generate(
                model, tokenizer, prompt, device, layer_idx, np.zeros_like(diff_dir), max_new_tokens=15)
            
            cat_ratio_results = {}
            
            for target_ratio in target_ratios:
                inject_norm = target_ratio * mean_delta_norm
                beta = inject_norm / max(diff_norm, 1e-10)
                inject_vec = beta * diff_dir
                
                # logits测试
                logits_patch = run_with_additive_patch(model, tokenizer, prompt, device, layer_idx, inject_vec)
                sel = compute_selectivity(logits_base, logits_patch, tokenizer, cat, compete_cats)
                kl = logit_kl(logits_patch, logits_base)
                
                # 生成测试
                gen_text = run_with_additive_patch_generate(
                    model, tokenizer, prompt, device, layer_idx, inject_vec, max_new_tokens=15)
                
                # 简单生成质量指标
                gen_quality = compute_generation_quality(gen_text, gen_base)
                
                cat_ratio_results[f"ratio_{target_ratio}"] = {
                    "selectivity": round(sel, 4),
                    "kl_div": round(kl, 4),
                    "beta": round(beta, 4),
                    "gen_text": gen_text,
                    "gen_quality": gen_quality,
                }
                
                plog(f"      ratio={target_ratio}: sel={sel:.4f}, kl={kl:.4f}, "
                     f"quality={gen_quality['overall']:.3f}, gen='{gen_text[:60]}...'")
            
            layer_result[cat] = cat_ratio_results
        
        results[f"L{layer_idx}"] = layer_result
    
    return results


def compute_generation_quality(gen_text, gen_base):
    """简单的生成质量评估"""
    # 基本检查
    has_gibberish = any(c.isdigit() and not c.isspace() for c in gen_text[-20:])  # 尾部数字
    has_repeat = gen_text.count(gen_text[-10:]) > 1 if len(gen_text) > 10 else False
    
    # 与基准的差异
    base_words = set(gen_base.lower().split())
    gen_words = set(gen_text.lower().split())
    word_overlap = len(base_words & gen_words) / max(len(base_words), 1)
    
    # 重复词比率
    words = gen_text.split()
    if words:
        unique_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)
    else:
        unique_ratio = 0.0
    
    # 综合评分: 0-1, 1=完美
    quality = 1.0
    if has_gibberish:
        quality -= 0.4
    if has_repeat:
        quality -= 0.3
    if word_overlap < 0.3:
        quality -= 0.2
    if unique_ratio < 0.5:
        quality -= 0.2
    quality = max(quality, 0.0)
    
    return {
        "overall": round(quality, 3),
        "has_gibberish": has_gibberish,
        "has_repeat": has_repeat,
        "word_overlap_with_base": round(word_overlap, 3),
        "unique_word_ratio": round(unique_ratio, 3),
    }


# ==================== Exp4: 去主轴+去混叠联合方向 ====================
def exp4_combined_directions(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    测试最优方向组合:
    1. raw (基线)
    2. no_pc1 (去第1主轴)
    3. disentangle (去类别混叠)
    4. no_pc1 + disentangle (联合)
    5. no_top3_pc + disentangle (联合加强版)
    """
    plog("=== Exp4: 去主轴+去混叠联合方向 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    key_layers = [
        n_layers // 3,
        n_layers // 2,
    ]
    key_layers = sorted(set([l for l in key_layers if l < n_layers]))
    
    test_cats = ["vehicle", "furniture", "animal"]
    results = {}
    
    for layer_idx in key_layers:
        plog(f"  Layer L{layer_idx}...")
        layer_result = {}
        
        # 收集所有类别残差
        cat_vecs_dict = {}
        for cat in ["fruit", "animal", "vehicle", "tool", "furniture"]:
            objs = obj_dict.get(cat, [])[:5]
            vecs = []
            for obj in objs:
                p = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
                if resid is not None:
                    vecs.append(resid)
            if len(vecs) >= 2:
                cat_vecs_dict[cat] = vecs
        
        if len(cat_vecs_dict) < 4:
            continue
        
        # 估计PC1
        all_vecs = []
        for v in cat_vecs_dict.values():
            all_vecs.extend(v)
        vecs_matrix = np.array(all_vecs)
        grand_mean = np.mean(vecs_matrix, axis=0)
        vecs_centered = vecs_matrix - grand_mean
        
        cov = np.cov(vecs_centered.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
        except:
            continue
        
        pc1 = eigvecs[:, 0]
        
        # 参考类别: fruit
        ref_cat = "fruit"
        ref_center = np.mean(cat_vecs_dict[ref_cat], axis=0)
        
        # 计算各类别差分方向
        cat_diff_dirs = {}
        for cat in ["vehicle", "tool", "furniture", "animal"]:
            if cat not in cat_vecs_dict:
                continue
            center = np.mean(cat_vecs_dict[cat], axis=0)
            diff = center - ref_center
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 1e-10:
                cat_diff_dirs[cat] = diff / diff_norm
        
        # 自然delta范数
        all_centers = [np.mean(v, axis=0) for v in cat_vecs_dict.values() if len(v) >= 2]
        grand_center = np.mean(all_centers, axis=0)
        delta_norms = [np.linalg.norm(c - grand_center) for c in all_centers]
        mean_delta_norm = float(np.mean(delta_norms))
        
        for target_cat in test_cats:
            if target_cat not in cat_diff_dirs:
                continue
            
            plog(f"    {target_cat}...")
            
            target_dir = cat_diff_dirs[target_cat]
            
            # 构造各类方向
            # 1. raw
            dir_raw = target_dir
            
            # 2. no_pc1
            diff_no_pc1 = target_dir - np.dot(target_dir, pc1) * pc1
            diff_no_pc1_norm = np.linalg.norm(diff_no_pc1)
            dir_no_pc1 = diff_no_pc1 / diff_no_pc1_norm if diff_no_pc1_norm > 1e-10 else dir_raw
            
            # 3. disentangle (去竞争类别)
            remove_cats = [c for c in ["vehicle", "tool", "furniture", "animal"] 
                         if c != target_cat and c in cat_diff_dirs]
            projected = np.zeros_like(target_dir)
            for rc in remove_cats:
                projected += np.dot(target_dir, cat_diff_dirs[rc]) * cat_diff_dirs[rc]
            dir_disentangle = target_dir - projected
            dir_disentangle_norm = np.linalg.norm(dir_disentangle)
            dir_disentangle = dir_disentangle / dir_disentangle_norm if dir_disentangle_norm > 1e-10 else dir_raw
            
            # 4. no_pc1 + disentangle
            combined = dir_disentangle - np.dot(dir_disentangle, pc1) * pc1
            combined_norm = np.linalg.norm(combined)
            dir_combined = combined / combined_norm if combined_norm > 1e-10 else dir_disentangle
            
            # 5. no_top3_pc + disentangle
            combined2 = dir_disentangle.copy()
            for k in range(min(3, len(eigvals))):
                combined2 -= np.dot(combined2, eigvecs[:, k]) * eigvecs[:, k]
            combined2_norm = np.linalg.norm(combined2)
            dir_combined2 = combined2 / combined2_norm if combined2_norm > 1e-10 else dir_combined
            
            # 随机对照
            np.random.seed(42)
            rand_dir = np.random.randn(len(target_dir))
            rand_norm = np.linalg.norm(rand_dir)
            dir_random = rand_dir / rand_norm if rand_norm > 1e-10 else None
            
            # 测试
            test_obj = obj_dict.get(target_cat, [])[0]
            if not test_obj:
                continue
            
            prompt = TEMPLATES_EN["is_a"].format(obj=test_obj)
            logits_base = get_final_logits(model, tokenizer, prompt, device)
            compete_cats = [c for c in ["animal", "tool", "vehicle", "fruit", "furniture"] if c != target_cat][:3]
            
            inject_norm = mean_delta_norm
            
            cat_result = {}
            
            for method_name, method_dir in [
                ("raw", dir_raw),
                ("no_pc1", dir_no_pc1),
                ("disentangle", dir_disentangle),
                ("no_pc1+disentangle", dir_combined),
                ("no_top3pc+disentangle", dir_combined2),
                ("random", dir_random),
            ]:
                if method_dir is None:
                    continue
                
                beta = inject_norm / max(np.linalg.norm(method_dir), 1e-10)
                inject_vec = beta * method_dir
                logits_patch = run_with_additive_patch(model, tokenizer, prompt, device, layer_idx, inject_vec)
                
                sel = compute_selectivity(logits_base, logits_patch, tokenizer, target_cat, compete_cats)
                kl = logit_kl(logits_patch, logits_base)
                
                cat_result[method_name] = {
                    "selectivity": round(sel, 4),
                    "kl_div": round(kl, 4),
                    "beta": round(beta, 4),
                }
                plog(f"      {method_name}: sel={sel:.4f}, kl={kl:.4f}")
            
            # 方向间余弦
            cosines = {}
            for name1, dir1 in [("raw", dir_raw), ("no_pc1", dir_no_pc1), 
                                ("disentangle", dir_disentangle), ("combined", dir_combined)]:
                for name2, dir2 in [("raw", dir_raw), ("no_pc1", dir_no_pc1),
                                     ("disentangle", dir_disentangle), ("combined", dir_combined)]:
                    if name1 < name2:
                        cosines[f"cos({name1},{name2})"] = round(float(np.dot(dir1, dir2)), 4)
            cat_result["direction_cosines"] = cosines
            
            layer_result[target_cat] = cat_result
        
        results[f"L{layer_idx}"] = layer_result
    
    return results


# ==================== Exp5: 生成质量系统性验证 ====================
def exp5_generation_systematic(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    系统验证注入后生成质量
    
    对3个模型, 在最优方向(no_pc1)和最优ratio下:
    1. 生成短文本
    2. 评估语义/语法/目标类别
    3. 对比DS7B的生成崩坏
    """
    plog("=== Exp5: 生成质量系统性验证 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    layer_idx = n_layers // 2
    
    test_cases = [
        ("fruit", "apple", ["animal", "tool"]),
        ("animal", "dog", ["fruit", "tool"]),
        ("vehicle", "car", ["fruit", "animal"]),
        ("furniture", "chair", ["fruit", "animal"]),
    ]
    
    # DS7B用小ratio, 其他模型用1.0
    if model_name == "deepseek7b":
        test_ratios = [0.1, 0.25, 0.5, 1.0]
    else:
        test_ratios = [0.5, 1.0, 2.0]
    
    results = {}
    
    for cat, test_obj, compete_cats in test_cases:
        if test_obj not in obj_dict.get(cat, []):
            continue
        
        plog(f"  {cat}/{test_obj}...")
        
        # 构造方向(去PC1版 + 原始版)
        other_cat = compete_cats[0]
        cat_objs = obj_dict.get(cat, [])[:4]
        other_objs = obj_dict.get(other_cat, [])[:4]
        
        cat_vecs, other_vecs = [], []
        for obj in cat_objs:
            p = TEMPLATES_EN["is_a"].format(obj=obj)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
            if resid is not None:
                cat_vecs.append(resid)
        for obj in other_objs:
            p = TEMPLATES_EN["is_a"].format(obj=obj)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
            if resid is not None:
                other_vecs.append(resid)
        
        if len(cat_vecs) < 2 or len(other_vecs) < 2:
            continue
        
        diff = np.mean(cat_vecs, axis=0) - np.mean(other_vecs, axis=0)
        diff_norm = np.linalg.norm(diff)
        if diff_norm < 1e-10:
            continue
        diff_dir = diff / diff_norm
        
        # 估计PC1
        all_prompts_for_cov = []
        for c in ["fruit", "animal", "vehicle"]:
            for obj in obj_dict.get(c, [])[:3]:
                all_prompts_for_cov.append(TEMPLATES_EN["is_a"].format(obj=obj))
        
        cov_res = estimate_covariance_and_pcs(model, tokenizer, all_prompts_for_cov, layer_idx, device)
        if cov_res[0] is not None:
            _, eigvals, eigvecs, _ = cov_res
            pc1 = eigvecs[:, 0]
            diff_no_pc1 = diff - np.dot(diff, pc1) * pc1
            diff_no_pc1_norm = np.linalg.norm(diff_no_pc1)
            if diff_no_pc1_norm > 1e-10:
                diff_no_pc1_dir = diff_no_pc1 / diff_no_pc1_norm
            else:
                diff_no_pc1_dir = diff_dir
        else:
            diff_no_pc1_dir = diff_dir
        
        # 自然delta范数
        cat_center = np.mean(cat_vecs, axis=0)
        delta_norms = [np.linalg.norm(v - cat_center) for v in cat_vecs]
        mean_delta_norm = float(np.mean(delta_norms))
        
        prompt = TEMPLATES_EN["is_a"].format(obj=test_obj)
        
        # 基准生成
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        gen_kwargs = dict(max_new_tokens=15, do_sample=False, repetition_penalty=1.2)
        with torch.no_grad():
            gen_base_ids = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        gen_base_text = tokenizer.decode(gen_base_ids[0], skip_special_tokens=True)
        
        cat_gen_results = {}
        
        for dir_name, dir_vec in [("raw", diff_dir), ("no_pc1", diff_no_pc1_dir)]:
            dir_results = {}
            
            for target_ratio in test_ratios:
                inject_norm = target_ratio * mean_delta_norm
                beta = inject_norm / max(np.linalg.norm(dir_vec), 1e-10)
                inject_vec = beta * dir_vec
                
                gen_text = run_with_additive_patch_generate(
                    model, tokenizer, prompt, device, layer_idx, inject_vec, max_new_tokens=15)
                
                quality = compute_generation_quality(gen_text, gen_base_text)
                
                dir_results[f"ratio_{target_ratio}"] = {
                    "gen_text": gen_text,
                    "quality": quality,
                    "beta": round(beta, 4),
                }
                
                plog(f"    {dir_name} ratio={target_ratio}: quality={quality['overall']:.3f}, "
                     f"gen='{gen_text[:60]}...'")
            
            cat_gen_results[dir_name] = dir_results
        
        cat_gen_results["base_gen"] = gen_base_text
        results[cat] = cat_gen_results
    
    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Use: qwen3, glm4, deepseek7b")
        return
    
    obj_dict = ROUNDS[round_num]
    
    plog(f"Phase 467: model={model_name}, round={round_num}, n_objects_per_cat={len(obj_dict.get('fruit',[]))}")
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    all_results = {
        "model": model_name,
        "round": round_num,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
    }
    
    # Exp1: PC1功能归因
    plog("\n" + "="*60)
    try:
        r1 = exp1_pc1_attribution(model, tokenizer, model_name, device, obj_dict, round_num)
        all_results["exp1_pc1_attribution"] = r1
    except Exception as e:
        plog(f"Exp1 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_pc1_attribution"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    plog(f"  After Exp1: GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # Exp2: 白化空间新方向
    plog("\n" + "="*60)
    try:
        r2 = exp2_whitened_new_directions(model, tokenizer, model_name, device, obj_dict, round_num)
        all_results["exp2_whitened_new_directions"] = r2
    except Exception as e:
        plog(f"Exp2 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_whitened_new_directions"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    plog(f"  After Exp2: GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # Exp3: 微扰敏感性地图
    plog("\n" + "="*60)
    try:
        r3 = exp3_ds7b_sensitivity_map(model, tokenizer, model_name, device, obj_dict, round_num)
        all_results["exp3_sensitivity_map"] = r3
    except Exception as e:
        plog(f"Exp3 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_sensitivity_map"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    plog(f"  After Exp3: GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # Exp4: 联合方向
    plog("\n" + "="*60)
    try:
        r4 = exp4_combined_directions(model, tokenizer, model_name, device, obj_dict, round_num)
        all_results["exp4_combined_directions"] = r4
    except Exception as e:
        plog(f"Exp4 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp4_combined_directions"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    plog(f"  After Exp4: GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # Exp5: 生成质量验证
    plog("\n" + "="*60)
    try:
        r5 = exp5_generation_systematic(model, tokenizer, model_name, device, obj_dict, round_num)
        all_results["exp5_generation_systematic"] = r5
    except Exception as e:
        plog(f"Exp5 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp5_generation_systematic"] = {"error": str(e)}
    
    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase467_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    plog(f"Results saved to {out_path}")
    
    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Phase 467 {model_name} R{round_num} complete!")


if __name__ == "__main__":
    main()
