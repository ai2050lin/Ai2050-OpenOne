"""
Phase 466: 白化空间注入、自适应beta校准与类别混叠剥离
=====================================================
核心实验:
1. Exp1: 白化方向构造→映射回原始空间→注入 (vs 原始方向注入)
2. Exp2: 自适应beta校准 — 每层beta使norm_ratio≈1
3. Exp3: vehicle/tool/furniture类别混叠剥离 — 正交化类别方向
4. Exp4: clothing候选族修复 — 检查tokenizer兼容性
5. Exp5: 生成质量验证 — 注入后短文本生成

用法: python tests/glm5/phase466_whitened_injection_disentangle.py qwen3 1
      python tests/glm5/phase466_whitened_injection_disentangle.py deepseek7b 2
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

CATEGORIES_ZH = {
    "fruit":   ["苹果", "香蕉", "橙子", "葡萄", "梨", "桃子", "柠檬", "芒果"],
    "animal":  ["狗", "猫", "马", "狮子", "熊", "兔子", "牛", "老虎"],
    "tool":    ["锤子", "刀", "扳手", "锯子", "钻头", "斧头", "铲子", "剪刀"],
    "vehicle": ["汽车", "公交车", "自行车", "卡车", "火车", "船", "飞机", "滑板车"],
    "clothing":["衬衫", "裙子", "帽子", "外套", "袜子", "手套", "围巾", "靴子"],
    "furniture":["椅子", "桌子", "书桌", "沙发", "床", "书架", "灯", "柜子"],
}

FAMILIES_EN = {
    "fruit":    ["fruit", "produce", "crop", "harvest"],
    "animal":   ["animal", "creature", "beast", "mammal"],
    "tool":     ["tool", "implement", "instrument", "device"],
    "vehicle":  ["vehicle", "transport", "automobile", "conveyance"],
    "clothing": ["clothing", "apparel", "garment", "attire"],
    "furniture":["furniture", "furnishing", "fixture", "seat"],
}

# 扩展clothing候选词: 有些模型可能对clothing/apparel等词tokenize不同
CLOTHING_ALT_FAMILIES = {
    "clothing": ["clothing", "apparel", "garment", "attire", "clothes", "dress", "wear"],
    "furniture": ["furniture", "furnishing", "fixture", "seat", "table", "chair"],
}

ZH_CLASS_WORDS = {
    "fruit": "水果", "animal": "动物", "tool": "工具",
    "vehicle": "交通工具", "clothing": "衣服", "furniture": "家具",
}

TEMPLATES_EN = {"is_a": "The {obj} is a kind of"}
TEMPLATES_ZH = {"is_a": "{obj}是一种"}

ROUNDS = {
    1: {k: v[:4] for k, v in CATEGORIES.items()},   # pilot: 4对象
    2: {k: v[:8] for k, v in CATEGORIES.items()},   # confirm: 8对象
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
    plog(f"  {model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB, class={type(model).__name__}")
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


def compute_en_family_margin(logits, tokenizer, target_cat, compete_cats, use_alt=False):
    """计算英文候选族边际"""
    families = CLOTHING_ALT_FAMILIES if (use_alt and target_cat in CLOTHING_ALT_FAMILIES) else FAMILIES_EN
    target_words = families.get(target_cat, [])
    compete_words = []
    for cc in compete_cats:
        c_families = CLOTHING_ALT_FAMILIES if (use_alt and cc in CLOTHING_ALT_FAMILIES) else FAMILIES_EN
        compete_words.extend(c_families.get(cc, []))
    
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


def compute_selectivity(logits_base, logits_patch, tokenizer, target_cat, compete_cats, use_alt=False):
    """计算selectivity"""
    margin_base_en, _, _ = compute_en_family_margin(logits_base, tokenizer, target_cat, compete_cats, use_alt)
    margin_patch_en, _, _ = compute_en_family_margin(logits_patch, tokenizer, target_cat, compete_cats, use_alt)
    return margin_patch_en - margin_base_en


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


# ==================== Exp1: 白化方向构造与回注入 ====================
def exp1_whitened_injection(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    验证白化后发现的多维结构是否可写
    
    方法:
    1. 收集自然激活, 估计协方差Σ
    2. 在白化空间构造类别差分方向 v_white
    3. 映射回原始空间: v_raw = Σ^{1/2} v_white
    4. 比较: 原始方向注入 vs 白化方向回注入 vs 去主轴方向注入
    """
    plog("=== Exp1: 白化方向构造与回注入 ===")
    info = get_model_info(model, model_name)
    
    key_layers = [
        info.n_layers // 6,
        info.n_layers // 3,
        info.n_layers // 2,
    ]
    key_layers = sorted(set([l for l in key_layers if l < info.n_layers]))
    
    test_cats = ["animal", "vehicle", "fruit"]
    results = {}
    
    for patch_li in key_layers:
        plog(f"  Layer L{patch_li}...")
        layer_results = {}
        
        # 收集自然激活分布(用于估计协方差)
        all_natural_vecs = []
        for cat in test_cats:
            objs = obj_dict.get(cat, [])[:3]
            for obj in objs:
                prompt = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
                if resid is not None:
                    all_natural_vecs.append(resid)
        
        if len(all_natural_vecs) < 6:
            plog(f"    L{patch_li}: Not enough natural vectors, skip")
            continue
        
        # 估计协方差矩阵(用前50个主成分近似)
        vecs_matrix = np.array(all_natural_vecs)
        vec_mean = np.mean(vecs_matrix, axis=0)
        vecs_centered = vecs_matrix - vec_mean
        cov = np.cov(vecs_centered.T)
        
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            n_pca = min(50, len(eigvals))
            idx = np.argsort(eigvals)[::-1][:n_pca]
            eigvals_top = eigvals[idx]
            eigvecs_top = eigvecs[:, idx]
            # Σ^{1/2}的前n_pca成分: eigvecs_top @ diag(sqrt(eigvals_top))
            sqrt_eigvals = np.sqrt(np.maximum(eigvals_top, 1e-10))
            Sigma_half = eigvecs_top * sqrt_eigvals[np.newaxis, :]  # [d_model, n_pca]
            Sigma_inv_half = eigvecs_top / sqrt_eigvals[np.newaxis, :]  # [d_model, n_pca]
            plog(f"    Covariance estimated: {n_pca} PCs, top1_eigval={eigvals_top[0]:.2f}")
        except Exception as e:
            plog(f"    Covariance estimation failed: {e}")
            continue
        
        for cat in test_cats:
            plog(f"    Category: {cat}")
            objs = obj_dict.get(cat, [])[:3]
            other_cat = "fruit" if cat != "fruit" else "animal"
            other_objs = obj_dict.get(other_cat, [])[:3]
            
            cat_vecs, other_vecs = [], []
            for obj in objs:
                prompt = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
                if resid is not None:
                    cat_vecs.append(resid)
            for obj in other_objs:
                prompt = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
                if resid is not None:
                    other_vecs.append(resid)
            
            if len(cat_vecs) < 2 or len(other_vecs) < 2:
                continue
            
            # 原始差分方向
            cat_center = np.mean(cat_vecs, axis=0)
            other_center = np.mean(other_vecs, axis=0)
            diff_raw = cat_center - other_center
            diff_raw_norm = np.linalg.norm(diff_raw)
            if diff_raw_norm < 1e-10:
                continue
            diff_dir_raw = diff_raw / diff_raw_norm
            
            # 白化空间中的差分方向
            diff_white = Sigma_inv_half.T @ diff_raw  # [n_pca]
            diff_white_norm = np.linalg.norm(diff_white)
            if diff_white_norm < 1e-10:
                continue
            diff_dir_white = diff_white / diff_white_norm
            
            # 映射回原始空间: v_raw = Σ^{1/2} v_white (归一化)
            v_back_raw = Sigma_half @ diff_dir_white  # [d_model]
            v_back_norm = np.linalg.norm(v_back_raw)
            if v_back_norm < 1e-10:
                continue
            v_back_dir = v_back_raw / v_back_norm
            
            # 去主轴方向: 去掉第1主成分后的差分
            pc1 = eigvecs_top[:, 0]
            diff_no_pc1 = diff_raw - np.dot(diff_raw, pc1) * pc1
            diff_no_pc1_norm = np.linalg.norm(diff_no_pc1)
            if diff_no_pc1_norm < 1e-10:
                diff_no_pc1_dir = diff_dir_raw  # fallback
            else:
                diff_no_pc1_dir = diff_no_pc1 / diff_no_pc1_norm
            
            # 测量层间自然delta范数(用于beta校准)
            delta_norms = [np.linalg.norm(v - cat_center) for v in cat_vecs]
            mean_delta_norm = float(np.mean(delta_norms)) if delta_norms else 1.0
            
            # 测试对象
            test_obj = objs[0]
            prompt = TEMPLATES_EN["is_a"].format(obj=test_obj)
            logits_base = get_final_logits(model, tokenizer, prompt, device)
            
            compete_cats = ["animal", "tool", "vehicle"] if cat == "fruit" else \
                           ["fruit", "tool", "vehicle"] if cat == "animal" else \
                           ["fruit", "animal", "tool"]
            
            cat_result = {}
            
            # 测试3种注入方向 × 2种beta(固定norm_ratio=1和2)
            for method_name, inject_dir in [("raw", diff_dir_raw), 
                                             ("whitened_back", v_back_dir),
                                             ("no_pc1", diff_no_pc1_dir)]:
                method_result = {}
                
                for target_ratio in [1.0, 2.0]:
                    # 自动计算beta使得norm_ratio = target_ratio
                    inject_norm_for_ratio = target_ratio * mean_delta_norm
                    dir_norm = np.linalg.norm(inject_dir)
                    beta = inject_norm_for_ratio / max(dir_norm, 1e-10)
                    
                    inject_vec = beta * inject_dir
                    logits_patch = run_with_additive_patch(model, tokenizer, prompt, device, patch_li, inject_vec)
                    
                    sel = compute_selectivity(logits_base, logits_patch, tokenizer, cat, compete_cats)
                    kl = logit_kl(logits_patch, logits_base)
                    
                    actual_norm_ratio = np.linalg.norm(inject_vec) / max(mean_delta_norm, 1e-10)
                    
                    method_result[f"ratio_{target_ratio}"] = {
                        "selectivity": round(sel, 4),
                        "kl_div": round(kl, 4),
                        "actual_norm_ratio": round(actual_norm_ratio, 4),
                        "beta": round(beta, 4),
                    }
                    
                    plog(f"      {method_name} ratio={target_ratio}: sel={sel:.4f}, kl={kl:.4f}, "
                         f"actual_ratio={actual_norm_ratio:.3f}, beta={beta:.3f}")
                
                cat_result[method_name] = method_result
            
            # 3种方向的余弦相似度
            cos_raw_white = float(np.dot(diff_dir_raw, v_back_dir))
            cos_raw_nopc1 = float(np.dot(diff_dir_raw, diff_no_pc1_dir))
            cos_white_nopc1 = float(np.dot(v_back_dir, diff_no_pc1_dir))
            
            cat_result["direction_cosine"] = {
                "raw_vs_whitened": round(cos_raw_white, 4),
                "raw_vs_nopc1": round(cos_raw_nopc1, 4),
                "whitened_vs_nopc1": round(cos_white_nopc1, 4),
            }
            
            layer_results[cat] = cat_result
        
        results[f"L{patch_li}"] = layer_results
    
    return results


# ==================== Exp2: 自适应beta校准 ====================
def exp2_adaptive_beta(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    对每层自动设置beta使得norm_ratio≈1
    
    测试:
    - target_norm_ratio = 0.25, 0.5, 1.0, 2.0, 4.0
    - 对每个(层, 类别)组合, 自动计算beta
    - 测量selectivity, KL散度, top5_overlap
    """
    plog("=== Exp2: 自适应beta校准 ===")
    info = get_model_info(model, model_name)
    
    key_layers = [
        info.n_layers // 6,
        info.n_layers // 3,
        info.n_layers // 2,
        2 * info.n_layers // 3,
    ]
    key_layers = sorted(set([l for l in key_layers if l < info.n_layers]))
    
    test_cats = ["animal", "vehicle", "fruit", "clothing"]
    target_ratios = [0.25, 0.5, 1.0, 2.0, 4.0]
    
    results = {}
    
    for patch_li in key_layers:
        plog(f"  Layer L{patch_li}...")
        layer_results = {}
        
        for cat in test_cats:
            objs = obj_dict.get(cat, [])
            if len(objs) < 2:
                continue
            
            other_cat = "fruit" if cat != "fruit" else "animal"
            other_objs = obj_dict.get(other_cat, [])[:3]
            
            cat_vecs, other_vecs = [], []
            for obj in objs[:4]:
                prompt = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
                if resid is not None:
                    cat_vecs.append(resid)
            for obj in other_objs:
                prompt = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
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
            
            # 测量层间自然delta范数
            delta_norms = [np.linalg.norm(v - cat_center) for v in cat_vecs]
            mean_delta_norm = float(np.mean(delta_norms)) if delta_norms else 1.0
            
            test_obj = objs[0]
            prompt = TEMPLATES_EN["is_a"].format(obj=test_obj)
            logits_base = get_final_logits(model, tokenizer, prompt, device)
            
            compete_cats = [c for c in ["animal", "tool", "vehicle", "fruit", "clothing"] if c != cat][:3]
            
            cat_result = {}
            
            for target_ratio in target_ratios:
                # 自动计算beta
                inject_norm_for_ratio = target_ratio * mean_delta_norm
                beta = inject_norm_for_ratio / max(diff_norm, 1e-10)
                
                inject_vec = beta * diff_dir
                logits_patch = run_with_additive_patch(model, tokenizer, prompt, device, patch_li, inject_vec)
                
                sel = compute_selectivity(logits_base, logits_patch, tokenizer, cat, compete_cats)
                kl = logit_kl(logits_patch, logits_base)
                
                # top5 overlap
                top5_base = set(np.argsort(logits_base)[-5:])
                top5_patch = set(np.argsort(logits_patch)[-5:])
                top5_overlap = len(top5_base & top5_patch) / 5.0
                
                actual_norm_ratio = np.linalg.norm(inject_vec) / max(mean_delta_norm, 1e-10)
                
                cat_result[f"ratio_{target_ratio}"] = {
                    "selectivity": round(sel, 4),
                    "kl_div": round(kl, 4),
                    "top5_overlap": round(top5_overlap, 4),
                    "actual_norm_ratio": round(actual_norm_ratio, 4),
                    "beta": round(beta, 4),
                    "mean_delta_norm": round(mean_delta_norm, 2),
                }
                
                plog(f"    {cat} ratio={target_ratio}: sel={sel:.4f}, kl={kl:.4f}, "
                     f"top5={top5_overlap:.2f}, beta={beta:.3f}, delta_norm={mean_delta_norm:.2f}")
            
            layer_results[cat] = cat_result
        
        results[f"L{patch_li}"] = layer_results
    
    return results


# ==================== Exp3: 类别混叠剥离 ====================
def exp3_class_disentangle(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    正交化类别方向, 检查vehicle/tool/furniture混叠
    
    方法:
    1. 计算vehicle, tool, furniture等类别中心(相对fruit)
    2. 构造vehicle_only = vehicle - Proj_tool(vehicle) - Proj_furniture(vehicle)
    3. 比较: 原始vehicle方向 vs vehicle_only方向的selectivity
    4. 也测试tool_only, furniture_only
    """
    plog("=== Exp3: 类别混叠剥离 ===")
    info = get_model_info(model, model_name)
    
    key_layers = [
        info.n_layers // 3,
        info.n_layers // 2,
        2 * info.n_layers // 3,
    ]
    key_layers = sorted(set([l for l in key_layers if l < info.n_layers]))
    
    # 需要剥离的类别组
    disentangle_groups = [
        ("vehicle", ["tool", "furniture"]),
        ("tool", ["vehicle", "furniture"]),
        ("furniture", ["vehicle", "tool"]),
    ]
    
    results = {}
    
    for patch_li in key_layers:
        plog(f"  Layer L{patch_li}...")
        layer_results = {}
        
        # 收集各类别中心(相对fruit的差分)
        ref_cat = "fruit"
        ref_objs = obj_dict.get(ref_cat, [])[:4]
        ref_vecs = []
        for obj in ref_objs:
            prompt = TEMPLATES_EN["is_a"].format(obj=obj)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
            if resid is not None:
                ref_vecs.append(resid)
        
        if len(ref_vecs) < 2:
            continue
        ref_center = np.mean(ref_vecs, axis=0)
        
        cat_diff_dirs = {}  # {cat: diff_direction (normalized)}
        cat_centers = {}
        
        for cat in ["vehicle", "tool", "furniture", "animal", "clothing"]:
            objs = obj_dict.get(cat, [])[:4]
            vecs = []
            for obj in objs:
                prompt = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
                if resid is not None:
                    vecs.append(resid)
            
            if len(vecs) < 2:
                continue
            
            center = np.mean(vecs, axis=0)
            cat_centers[cat] = center
            diff = center - ref_center
            diff_norm = np.linalg.norm(diff)
            if diff_norm > 1e-10:
                cat_diff_dirs[cat] = diff / diff_norm
        
        if len(cat_diff_dirs) < 3:
            plog(f"    L{patch_li}: Not enough cat directions, skip")
            continue
        
        # 测量自然delta范数
        all_vecs = []
        for cat in cat_centers:
            objs = obj_dict.get(cat, [])[:4]
            for obj in objs:
                prompt = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
                if resid is not None:
                    all_vecs.append(resid)
        if all_vecs:
            grand_center = np.mean(all_vecs, axis=0)
            delta_norms = [np.linalg.norm(v - grand_center) for v in all_vecs]
            mean_delta_norm = float(np.mean(delta_norms))
        else:
            mean_delta_norm = 1.0
        
        for target_cat, remove_cats in disentangle_groups:
            if target_cat not in cat_diff_dirs:
                continue
            
            target_dir = cat_diff_dirs[target_cat]
            
            # 原始方向的余弦相似度
            raw_cos = {}
            for other_cat in remove_cats:
                if other_cat in cat_diff_dirs:
                    cos_val = float(np.dot(target_dir, cat_diff_dirs[other_cat]))
                    raw_cos[other_cat] = round(cos_val, 4)
            
            # 构造正交化方向: target_only = target - Σ Proj_{remove_i}(target)
            projected = np.zeros_like(target_dir)
            for remove_cat in remove_cats:
                if remove_cat in cat_diff_dirs:
                    remove_dir = cat_diff_dirs[remove_cat]
                    projected += np.dot(target_dir, remove_dir) * remove_dir
            
            target_only = target_dir - projected
            target_only_norm = np.linalg.norm(target_only)
            if target_only_norm < 1e-10:
                plog(f"    {target_cat}_only is zero after disentangling, skip")
                continue
            target_only_dir = target_only / target_only_norm
            
            # 正交化后的余弦
            disentangle_cos = {}
            for other_cat in remove_cats:
                if other_cat in cat_diff_dirs:
                    cos_val = float(np.dot(target_only_dir, cat_diff_dirs[other_cat]))
                    disentangle_cos[other_cat] = round(cos_val, 4)
            
            # 测试对象
            test_obj = obj_dict.get(target_cat, [])[0]
            if not test_obj:
                continue
            prompt = TEMPLATES_EN["is_a"].format(obj=test_obj)
            logits_base = get_final_logits(model, tokenizer, prompt, device)
            
            compete_cats = [c for c in ["animal", "tool", "vehicle", "fruit", "clothing", "furniture"] 
                          if c != target_cat][:3]
            
            # 注入测试: 用norm_ratio=1
            inject_norm = mean_delta_norm
            
            # 原始方向注入
            beta_raw = inject_norm / max(np.linalg.norm(target_dir), 1e-10)
            logits_raw = run_with_additive_patch(model, tokenizer, prompt, device, patch_li, beta_raw * target_dir)
            sel_raw = compute_selectivity(logits_base, logits_raw, tokenizer, target_cat, compete_cats)
            kl_raw = logit_kl(logits_raw, logits_base)
            
            # 正交化方向注入
            beta_only = inject_norm / max(np.linalg.norm(target_only_dir), 1e-10)
            logits_only = run_with_additive_patch(model, tokenizer, prompt, device, patch_li, beta_only * target_only_dir)
            sel_only = compute_selectivity(logits_base, logits_only, tokenizer, target_cat, compete_cats)
            kl_only = logit_kl(logits_only, logits_base)
            
            # 随机方向对照(与正交化方向正交的随机方向)
            # 生成一个与target_only_dir同维的随机方向, 然后归一化
            np.random.seed(42)
            rand_dir = np.random.randn(len(target_only_dir))
            rand_dir = rand_dir - np.dot(rand_dir, target_only_dir) * target_only_dir  # 去除target_only分量
            rand_norm = np.linalg.norm(rand_dir)
            if rand_norm > 1e-10:
                rand_dir = rand_dir / rand_norm
                beta_rand = inject_norm / max(np.linalg.norm(rand_dir), 1e-10)
                logits_rand = run_with_additive_patch(model, tokenizer, prompt, device, patch_li, beta_rand * rand_dir)
                sel_rand = compute_selectivity(logits_base, logits_rand, tokenizer, target_cat, compete_cats)
                kl_rand = logit_kl(logits_rand, logits_base)
            else:
                sel_rand = 0
                kl_rand = 0
            
            layer_results[target_cat] = {
                "raw_selectivity": round(sel_raw, 4),
                "raw_kl": round(kl_raw, 4),
                "disentangle_selectivity": round(sel_only, 4),
                "disentangle_kl": round(kl_only, 4),
                "random_selectivity": round(sel_rand, 4),
                "random_kl": round(kl_rand, 4),
                "raw_cos_with_others": raw_cos,
                "disentangle_cos_with_others": disentangle_cos,
                "projection_loss_ratio": round(1.0 - target_only_norm / max(np.linalg.norm(target_dir), 1e-10), 4),
                "norm_ratio_used": round(inject_norm / max(mean_delta_norm, 1e-10), 4),
            }
            
            plog(f"    {target_cat}: raw_sel={sel_raw:.4f}, disentangle_sel={sel_only:.4f}, "
                 f"random_sel={sel_rand:.4f}, proj_loss={1.0 - target_only_norm:.4f}")
        
        results[f"L{patch_li}"] = layer_results
    
    return results


# ==================== Exp4: clothing候选族修复 ====================
def exp4_clothing_fix(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    诊断并修复clothing候选族全为0的问题
    
    检查:
    1. 各模型tokenizer中clothing候选词是否存在
    2. 不同候选词的首token ID
    3. clothing对象的残差方向是否存在
    4. 用修复后的候选词重测selectivity
    """
    plog("=== Exp4: clothing候选族修复 ===")
    info = get_model_info(model, model_name)
    
    patch_li = info.n_layers // 2
    
    results = {}
    
    # 1. 检查所有候选词在tokenizer中的状态
    plog("  Checking tokenization...")
    for cat in ["clothing", "fruit", "vehicle", "furniture"]:
        words = FAMILIES_EN.get(cat, [])
        alt_words = CLOTHING_ALT_FAMILIES.get(cat, [])
        all_words = list(set(words + alt_words))
        
        word_status = {}
        for w in all_words:
            vocab = tokenizer.get_vocab()
            w_clean = w.strip()
            if w_clean in vocab:
                tok_id = vocab[w_clean]
                word_status[w] = {"found": True, "key": w_clean, "token_id": tok_id}
            elif f" {w_clean}" in vocab:
                tok_id = vocab[f" {w_clean}"]
                word_status[w] = {"found": True, "key": f" {w_clean}", "token_id": tok_id}
            else:
                # 尝试直接encode
                ids = tokenizer.encode(w, add_special_tokens=False)
                word_status[w] = {"found": len(ids) == 1, "key": "encode", "token_ids": ids, 
                                  "decoded": [tokenizer.decode([i]) for i in ids]}
        
        results[f"{cat}_tokenization"] = word_status
        
        found_count = sum(1 for v in word_status.values() if v.get("found", False))
        plog(f"    {cat}: {found_count}/{len(all_words)} words found in vocab")
    
    # 2. 检查clothing对象的残差方向
    clothing_objs = obj_dict.get("clothing", [])
    fruit_objs = obj_dict.get("fruit", [])
    
    if len(clothing_objs) >= 2 and len(fruit_objs) >= 2:
        cloth_vecs, fruit_vecs = [], []
        for obj in clothing_objs[:4]:
            prompt = TEMPLATES_EN["is_a"].format(obj=obj)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
            if resid is not None:
                cloth_vecs.append(resid)
        for obj in fruit_objs[:4]:
            prompt = TEMPLATES_EN["is_a"].format(obj=obj)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
            if resid is not None:
                fruit_vecs.append(resid)
        
        if len(cloth_vecs) >= 2 and len(fruit_vecs) >= 2:
            cloth_center = np.mean(cloth_vecs, axis=0)
            fruit_center = np.mean(fruit_vecs, axis=0)
            diff = cloth_center - fruit_center
            diff_norm = np.linalg.norm(diff)
            
            results["clothing_residual"] = {
                "diff_norm": round(diff_norm, 4),
                "clothing_center_norm": round(float(np.linalg.norm(cloth_center)), 4),
                "fruit_center_norm": round(float(np.linalg.norm(fruit_center)), 4),
                "n_clothing_vecs": len(cloth_vecs),
                "n_fruit_vecs": len(fruit_vecs),
            }
            
            if diff_norm > 1e-10:
                diff_dir = diff / diff_norm
                
                # 3. 测试clothing的selectivity(使用不同候选词)
                test_obj = clothing_objs[0]
                prompt = TEMPLATES_EN["is_a"].format(obj=test_obj)
                logits_base = get_final_logits(model, tokenizer, prompt, device)
                
                # 注入(固定beta=5)
                logits_patch = run_with_additive_patch(model, tokenizer, prompt, device, patch_li, 5 * diff_dir)
                
                compete_cats = ["fruit", "animal", "vehicle"]
                
                # 标准候选词
                sel_standard = compute_selectivity(logits_base, logits_patch, tokenizer, "clothing", compete_cats)
                # 扩展候选词
                sel_alt = compute_selectivity(logits_base, logits_patch, tokenizer, "clothing", compete_cats, use_alt=True)
                
                results["clothing_selectivity"] = {
                    "standard": round(sel_standard, 4),
                    "alt_families": round(sel_alt, 4),
                    "beta": 5.0,
                    "diff_norm": round(diff_norm, 4),
                }
                
                plog(f"    Clothing sel: standard={sel_standard:.4f}, alt={sel_alt:.4f}")
    
    return results


# ==================== Exp5: 生成质量验证 ====================
def exp5_generation_quality(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    注入后生成短文本, 检查生成质量
    
    指标:
    1. 生成文本是否合理
    2. 目标类别词是否出现
    3. 语法是否崩坏(用启发式: 重复率、特殊字符率)
    4. 与基准生成的差异
    """
    plog("=== Exp5: 生成质量验证 ===")
    info = get_model_info(model, model_name)
    
    patch_li = info.n_layers // 2
    results = {}
    
    test_cases = [
        ("fruit", "apple", ["animal", "tool"]),
        ("animal", "dog", ["fruit", "tool"]),
        ("vehicle", "car", ["fruit", "animal"]),
    ]
    
    for cat, test_obj, compete_cats in test_cases:
        if test_obj not in obj_dict.get(cat, []):
            continue
        
        prompt = TEMPLATES_EN["is_a"].format(obj=test_obj)
        
        # 构造差分方向
        other_cat = compete_cats[0]
        other_objs = obj_dict.get(other_cat, [])[:3]
        cat_objs = obj_dict.get(cat, [])[:3]
        
        cat_vecs, other_vecs = [], []
        for obj in cat_objs:
            p = TEMPLATES_EN["is_a"].format(obj=obj)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, p, patch_li, device)
            if resid is not None:
                cat_vecs.append(resid)
        for obj in other_objs:
            p = TEMPLATES_EN["is_a"].format(obj=obj)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, p, patch_li, device)
            if resid is not None:
                other_vecs.append(resid)
        
        if len(cat_vecs) < 2 or len(other_vecs) < 2:
            continue
        
        diff = np.mean(cat_vecs, axis=0) - np.mean(other_vecs, axis=0)
        diff_norm = np.linalg.norm(diff)
        if diff_norm < 1e-10:
            continue
        diff_dir = diff / diff_norm
        
        # 测量自然delta范数
        cat_center = np.mean(cat_vecs, axis=0)
        delta_norms = [np.linalg.norm(v - cat_center) for v in cat_vecs]
        mean_delta_norm = float(np.mean(delta_norms))
        
        # 基准生成
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        gen_kwargs = dict(max_new_tokens=15, do_sample=False, repetition_penalty=1.2)
        
        with torch.no_grad():
            gen_base_ids = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        gen_base_text = tokenizer.decode(gen_base_ids[0], skip_special_tokens=True)
        
        # 注入后生成 — 使用norm_ratio=1
        inject_norm = mean_delta_norm
        beta = inject_norm / max(diff_norm, 1e-10)
        inject_vec = beta * diff_dir
        
        layers = get_layers(model)
        seq_len = attention_mask.sum().item()
        delta_tensor = torch.tensor(inject_vec, dtype=torch.float32, device=device)
        
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
        
        h = layers[patch_li].register_forward_hook(make_hook())
        with torch.no_grad():
            gen_patch_ids = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        h.remove()
        gen_patch_text = tokenizer.decode(gen_patch_ids[0], skip_special_tokens=True)
        
        # 注入后生成 — 使用norm_ratio=2 (对比)
        inject_vec_2 = 2 * inject_vec
        delta_tensor_2 = torch.tensor(inject_vec_2, dtype=torch.float32, device=device)
        
        patched2 = [False]
        def make_hook2():
            def hook(module, input, output):
                if not patched2[0]:
                    patched2[0] = True
                    if isinstance(output, tuple):
                        out_tensor = output[0].clone()
                        out_tensor[0, seq_len - 1, :] += delta_tensor_2.to(out_tensor.dtype)
                        return (out_tensor,) + output[1:]
                    else:
                        out_tensor = output.clone()
                        out_tensor[0, seq_len - 1, :] += delta_tensor_2.to(out_tensor.dtype)
                        return out_tensor
                return None
            return hook
        
        h2 = layers[patch_li].register_forward_hook(make_hook2())
        with torch.no_grad():
            gen_patch2_ids = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        h2.remove()
        gen_patch2_text = tokenizer.decode(gen_patch2_ids[0], skip_special_tokens=True)
        
        # 启发式质量指标
        def text_quality_metrics(text, cat_words=None):
            """简单的文本质量评估"""
            words = text.split()
            if not words:
                return {"length": 0, "repeat_ratio": 1.0, "has_target_cat": False}
            
            # 重复率
            unique_words = set(w.lower() for w in words)
            repeat_ratio = 1.0 - len(unique_words) / max(len(words), 1)
            
            # 是否包含目标类别词
            has_target = False
            if cat_words:
                has_target = any(cw in text.lower() for cw in cat_words)
            
            return {
                "length": len(words),
                "repeat_ratio": round(repeat_ratio, 3),
                "has_target_cat": has_target,
            }
        
        cat_words = FAMILIES_EN.get(cat, [])[:3]
        
        base_quality = text_quality_metrics(gen_base_text, cat_words)
        patch_quality = text_quality_metrics(gen_patch_text, cat_words)
        patch2_quality = text_quality_metrics(gen_patch2_text, cat_words)
        
        results[cat] = {
            "prompt": prompt,
            "beta_auto": round(beta, 4),
            "norm_ratio_auto": round(inject_norm / max(mean_delta_norm, 1e-10), 4),
            "gen_base": gen_base_text,
            "gen_patch_ratio1": gen_patch_text,
            "gen_patch_ratio2": gen_patch2_text,
            "base_quality": base_quality,
            "patch_quality_ratio1": patch_quality,
            "patch_quality_ratio2": patch2_quality,
        }
        
        plog(f"    {cat}: base='{gen_base_text[:60]}...' → ratio1='{gen_patch_text[:60]}...'")
        plog(f"          ratio2='{gen_patch2_text[:60]}...'")
    
    return results


# ==================== 主函数 ====================
def main():
    if len(sys.argv) < 3:
        print("Usage: python phase466_whitened_injection_disentangle.py <model_name> <round_num>")
        print("  model_name: qwen3 | deepseek7b | glm4")
        print("  round_num: 1 (pilot) | 2 (confirm)")
        sys.exit(1)
    
    model_name = sys.argv[1]
    round_num = int(sys.argv[2])
    
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        sys.exit(1)
    
    obj_dict = ROUNDS.get(round_num, ROUNDS[1])
    
    plog(f"Phase 466: {model_name} round {round_num}")
    plog(f"  Objects per category: {', '.join(f'{k}={len(v)}' for k, v in obj_dict.items())}")
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    plog(f"  Model: {info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    all_results = {
        "model": model_name,
        "round": round_num,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
    }
    
    # ===== Exp1: 白化方向构造与回注入 =====
    try:
        all_results["exp1_whitened_injection"] = exp1_whitened_injection(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"Exp1 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_whitened_injection"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    plog("Exp1 done, GPU cleared")
    
    # ===== Exp2: 自适应beta校准 =====
    try:
        all_results["exp2_adaptive_beta"] = exp2_adaptive_beta(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"Exp2 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_adaptive_beta"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    plog("Exp2 done, GPU cleared")
    
    # ===== Exp3: 类别混叠剥离 =====
    try:
        all_results["exp3_disentangle"] = exp3_class_disentangle(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"Exp3 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_disentangle"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    plog("Exp3 done, GPU cleared")
    
    # ===== Exp4: clothing候选族修复 =====
    try:
        all_results["exp4_clothing_fix"] = exp4_clothing_fix(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"Exp4 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp4_clothing_fix"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    plog("Exp4 done, GPU cleared")
    
    # ===== Exp5: 生成质量验证 =====
    try:
        all_results["exp5_generation_quality"] = exp5_generation_quality(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"Exp5 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp5_generation_quality"] = {"error": str(e)}
    
    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    result_path = f"results/glm5/phase466_{model_name}_r{round_num}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    plog(f"Results saved to {result_path}")
    
    # 释放模型
    release_model(model)
    plog(f"Phase 466 {model_name} R{round_num} complete!")


if __name__ == "__main__":
    main()
