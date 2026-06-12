"""
Phase 468: PC1因果验证、类别净化策略搜索与模板稳健性闭环
================================================================
核心实验:
1. Exp1: PC1因果验证 — 注入/消融PC1, 观察logit entropy/top1_prob/margin变化
2. Exp2: PC1成分分解 — 将PC1拆为entropy/position/template/readout成分
3. Exp3: 类别净化策略搜索 — 6类别 × 6策略, 找每类最优净化方法
4. Exp4: DS7B模板稳健性 — 5模板 × 3模型, 找不触发数学模式的模板

用法:
  python tests/glm5/phase468_pc1_causal_purification_template.py qwen3 1
  python tests/glm5/phase468_pc1_causal_purification_template.py glm4 1
  python tests/glm5/phase468_pc1_causal_purification_template.py deepseek7b 1
  (round 2 for confirmation)
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

# 模板集合 — 包含原模板和新候选模板
TEMPLATES_ALL = {
    "is_a":          "The {obj} is a kind of",
    "category_of":   "The category of {obj} is",
    "belongs_to":    "{obj} belongs to the category of",
    "classified_as": "A {obj} is commonly classified as",
    "simple_answer": "A simple answer: {obj} is a",
}

ROUNDS = {
    1: {k: v[:5] for k, v in CATEGORIES.items()},   # R1: 5对象/类
    2: {k: v[:8] for k, v in CATEGORIES.items()},   # R2: 8对象/类(确认)
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
    """加法patch: 在patch_layer的输出中加上delta_vec(最后token位置)"""
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


def logit_entropy(logits_vec):
    """计算logit分布的熵"""
    log_probs = logits_vec - np.max(logits_vec)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs)))
    return -float(np.sum(np.exp(log_probs) * log_probs))


def top1_probability(logits_vec):
    """最高候选的概率"""
    log_probs = logits_vec - np.max(logits_vec)
    probs = np.exp(log_probs) / np.sum(np.exp(log_probs))
    return float(np.max(probs))


def estimate_pcs_at_layer(model, tokenizer, prompts, layer_idx, device, n_pcs=10):
    """估计指定层的主成分"""
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
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        return cov, eigvals, eigvecs, vec_mean
    except:
        return None, None, None, None


# ==================== Exp1: PC1因果验证 ====================
def exp1_pc1_causal_verification(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    PC1因果验证:
    对PC1做正向注入(+PC1), 反向注入(-PC1), 消融(去除PC1成分),
    观察logit entropy / top1_prob / candidate margin的变化.
    
    如果+PC1增加entropy, -PC1降低entropy → PC1因果控制不确定性
    如果±PC1只改变margin而不改entropy → PC1是类别语义轴
    如果±PC1既改entropy又改margin → PC1是混合轴
    """
    plog("=== Exp1: PC1因果验证 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    # 4个关键层
    key_layers = sorted(set([
        n_layers // 6,
        n_layers // 3,
        n_layers // 2,
        2 * n_layers // 3,
    ]))
    
    n_obj = 5 if round_num == 1 else 8
    results = {}
    
    for layer_idx in key_layers:
        plog(f"  Layer L{layer_idx}...")
        layer_result = {}
        
        # ---- 1. 估计PC1 ----
        all_prompts = []
        for cat in ["fruit", "animal", "vehicle", "tool", "furniture"]:
            objs = obj_dict.get(cat, [])[:n_obj]
            for obj in objs:
                p = TEMPLATES_ALL["is_a"].format(obj=obj)
                all_prompts.append(p)
        
        cov, eigvals, eigvecs, vec_mean = estimate_pcs_at_layer(
            model, tokenizer, all_prompts, layer_idx, device)
        
        if eigvecs is None:
            plog(f"    L{layer_idx}: PCA failed, skip")
            continue
        
        pc1 = eigvecs[:, 0]
        pc1_var_ratio = float(eigvals[0] / max(sum(eigvals[:50]), 1e-10))
        plog(f"    PC1 variance ratio: {pc1_var_ratio:.4f}")
        layer_result["pc1_var_ratio"] = round(pc1_var_ratio, 4)
        
        # ---- 2. 计算自然PC1投影范数(用于确定注入强度) ----
        natural_proj_norms = []
        for p in all_prompts[:10]:
            resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
            if resid is not None:
                centered = resid - vec_mean
                natural_proj_norms.append(abs(float(np.dot(centered, pc1))))
        
        if not natural_proj_norms:
            continue
        
        natural_std = float(np.std(natural_proj_norms))
        if natural_std < 1e-10:
            natural_std = float(np.mean(natural_proj_norms))
        plog(f"    Natural PC1 projection: mean={np.mean(natural_proj_norms):.4f}, std={natural_std:.4f}")
        
        # ---- 3. PC1因果注入测试 ----
        # 用3个对象做测试, 每个对象测试±PC1和消融
        test_objects = [("car", "vehicle"), ("dog", "animal"), ("apple", "fruit")]
        injection_ratios = [0.5, 1.0, 2.0]  # 以natural_std为单位的倍数
        
        causal_results = {}
        
        for obj_name, obj_cat in test_objects:
            prompt = TEMPLATES_ALL["is_a"].format(obj=obj_name)
            other_cats = [c for c in ["fruit", "animal", "vehicle", "tool", "furniture"] if c != obj_cat]
            
            # 基线logits
            base_logits = get_final_logits(model, tokenizer, prompt, device)
            base_entropy = logit_entropy(base_logits)
            base_top1 = top1_probability(base_logits)
            base_margin, _, _ = compute_en_family_margin(base_logits, tokenizer, obj_cat, other_cats)
            
            obj_results = {
                "baseline": {
                    "entropy": round(base_entropy, 4),
                    "top1_prob": round(base_top1, 4),
                    "margin": round(base_margin, 4),
                }
            }
            
            # PC1正负注入
            for ratio in injection_ratios:
                delta_pos = ratio * natural_std * pc1
                delta_neg = -ratio * natural_std * pc1
                
                # +PC1
                logits_pos = run_with_additive_patch(model, tokenizer, prompt, device, layer_idx, delta_pos)
                ent_pos = logit_entropy(logits_pos)
                top1_pos = top1_probability(logits_pos)
                margin_pos, _, _ = compute_en_family_margin(logits_pos, tokenizer, obj_cat, other_cats)
                
                # -PC1
                logits_neg = run_with_additive_patch(model, tokenizer, prompt, device, layer_idx, delta_neg)
                ent_neg = logit_entropy(logits_neg)
                top1_neg = top1_probability(logits_neg)
                margin_neg, _, _ = compute_en_family_margin(logits_neg, tokenizer, obj_cat, other_cats)
                
                obj_results[f"+pc1_{ratio}x"] = {
                    "entropy": round(ent_pos, 4),
                    "top1_prob": round(top1_pos, 4),
                    "margin": round(margin_pos, 4),
                    "delta_entropy": round(ent_pos - base_entropy, 4),
                    "delta_top1": round(top1_pos - base_top1, 4),
                    "delta_margin": round(margin_pos - base_margin, 4),
                }
                obj_results[f"-pc1_{ratio}x"] = {
                    "entropy": round(ent_neg, 4),
                    "top1_prob": round(top1_neg, 4),
                    "margin": round(margin_neg, 4),
                    "delta_entropy": round(ent_neg - base_entropy, 4),
                    "delta_top1": round(top1_neg - base_top1, 4),
                    "delta_margin": round(margin_neg - base_margin, 4),
                }
            
            # PC1消融: 从residual中减去PC1成分
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, layer_idx, device)
            if resid is not None:
                centered = resid - vec_mean
                pc1_component = np.dot(centered, pc1) * pc1
                # 消融 = 减去PC1成分, 即注入-PC1_component
                delta_ablate = -pc1_component
                
                logits_ablate = run_with_additive_patch(model, tokenizer, prompt, device, layer_idx, delta_ablate)
                ent_ablate = logit_entropy(logits_ablate)
                top1_ablate = top1_probability(logits_ablate)
                margin_ablate, _, _ = compute_en_family_margin(logits_ablate, tokenizer, obj_cat, other_cats)
                
                obj_results["ablate_pc1"] = {
                    "entropy": round(ent_ablate, 4),
                    "top1_prob": round(top1_ablate, 4),
                    "margin": round(margin_ablate, 4),
                    "delta_entropy": round(ent_ablate - base_entropy, 4),
                    "delta_top1": round(top1_ablate - base_top1, 4),
                    "delta_margin": round(margin_ablate - base_margin, 4),
                }
            
            # 随机方向对照(5个随机方向)
            random_entropy_deltas = []
            random_margin_deltas = []
            for _ in range(5):
                rand_dir = np.random.randn(len(pc1))
                rand_dir = rand_dir / np.linalg.norm(rand_dir) * natural_std
                
                logits_rand = run_with_additive_patch(model, tokenizer, prompt, device, layer_idx, rand_dir)
                ent_rand = logit_entropy(logits_rand)
                margin_rand, _, _ = compute_en_family_margin(logits_rand, tokenizer, obj_cat, other_cats)
                
                random_entropy_deltas.append(ent_rand - base_entropy)
                random_margin_deltas.append(margin_rand - base_margin)
            
            obj_results["random_control"] = {
                "mean_delta_entropy": round(float(np.mean(random_entropy_deltas)), 4),
                "std_delta_entropy": round(float(np.std(random_entropy_deltas)), 4),
                "mean_delta_margin": round(float(np.mean(random_margin_deltas)), 4),
                "std_delta_margin": round(float(np.std(random_margin_deltas)), 4),
            }
            
            causal_results[obj_name] = obj_results
            plog(f"    {obj_name}: base_ent={base_entropy:.3f}, +pc1_1x_Δent={obj_results['+pc1_1.0x']['delta_entropy']:.4f}, "
                 f"-pc1_1x_Δent={obj_results['-pc1_1.0x']['delta_entropy']:.4f}, "
                 f"random_Δent={obj_results['random_control']['mean_delta_entropy']:.4f}")
        
        layer_result["causal_results"] = causal_results
        
        # ---- 4. 汇总: PC1对entropy的因果效应 ----
        # 对3个对象的1x注入取平均
        pc1_entropy_effect = []
        pc1_margin_effect = []
        random_entropy_effect = []
        random_margin_effect = []
        
        for obj_name in causal_results:
            r = causal_results[obj_name]
            pc1_entropy_effect.append(r["+pc1_1.0x"]["delta_entropy"])
            pc1_margin_effect.append(r["+pc1_1.0x"]["delta_margin"])
            random_entropy_effect.append(r["random_control"]["mean_delta_entropy"])
            random_margin_effect.append(r["random_control"]["mean_delta_margin"])
        
        layer_result["summary"] = {
            "pc1_mean_delta_entropy": round(float(np.mean(pc1_entropy_effect)), 4),
            "pc1_mean_delta_margin": round(float(np.mean(pc1_margin_effect)), 4),
            "random_mean_delta_entropy": round(float(np.mean(random_entropy_effect)), 4),
            "random_mean_delta_margin": round(float(np.mean(random_margin_effect)), 4),
            "pc1_vs_random_entropy_ratio": round(
                float(np.mean(pc1_entropy_effect)) / max(abs(float(np.mean(random_entropy_effect))), 1e-6), 2),
            "is_entropy_axis": abs(float(np.mean(pc1_entropy_effect))) > 2 * abs(float(np.mean(random_entropy_effect))),
        }
        
        plog(f"    Summary: PC1_Δent={layer_result['summary']['pc1_mean_delta_entropy']:.4f}, "
             f"random_Δent={layer_result['summary']['random_mean_delta_entropy']:.4f}, "
             f"ratio={layer_result['summary']['pc1_vs_random_entropy_ratio']:.2f}, "
             f"is_entropy_axis={layer_result['summary']['is_entropy_axis']}")
        
        results[f"L{layer_idx}"] = layer_result
    
    return results


# ==================== Exp2: PC1成分分解 ====================
def exp2_pc1_decomposition(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    将PC1分解为entropy/position/template/readout成分:
    
    方法:
    1. EntropyAxis: 在所有样本上, logit entropy与PC1投影的回归方向
    2. PositionAxis: 序列位置与PC1投影的回归方向
    3. TemplateAxis: 不同模板间PC1投影差异方向
    4. ReadoutAxis: PC1与W_U读出空间的对齐成分
    
    分解: PC1 = a*Entropy + b*Position + c*Template + d*Readout + residual
    """
    plog("=== Exp2: PC1成分分解 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    key_layers = sorted(set([
        n_layers // 6,
        n_layers // 3,
        n_layers // 2,
        2 * n_layers // 3,
    ]))
    
    n_obj = 5 if round_num == 1 else 8
    results = {}
    
    for layer_idx in key_layers:
        plog(f"  Layer L{layer_idx}...")
        layer_result = {}
        
        # ---- 1. 估计PC1 ----
        all_prompts_isa = []
        for cat in ["fruit", "animal", "vehicle", "tool", "furniture"]:
            objs = obj_dict.get(cat, [])[:n_obj]
            for obj in objs:
                all_prompts_isa.append(TEMPLATES_ALL["is_a"].format(obj=obj))
        
        cov, eigvals, eigvecs, vec_mean = estimate_pcs_at_layer(
            model, tokenizer, all_prompts_isa, layer_idx, device)
        
        if eigvecs is None:
            continue
        
        pc1 = eigvecs[:, 0]
        d = len(pc1)
        
        # ---- 2. 构造各成分方向 ----
        
        # (a) Entropy axis: 在所有样本上, 收集PC1投影和logit entropy, 构造回归方向
        ent_data = []
        for p in all_prompts_isa[:15]:
            resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
            if resid is not None:
                logits = get_final_logits(model, tokenizer, p, device)
                ent = logit_entropy(logits)
                centered = resid - vec_mean
                pc1_proj = float(np.dot(centered, pc1))
                ent_data.append((pc1_proj, ent, centered.copy()))
        
        if len(ent_data) > 3:
            # 回归: entropy = a * pc1_projection + b
            projs = np.array([x[0] for x in ent_data])
            entropies = np.array([x[1] for x in ent_data])
            centered_vecs = np.array([x[2] for x in ent_data])
            
            # PC1-entropy相关
            if np.std(projs) > 1e-10 and np.std(entropies) > 1e-10:
                corr_entropy = float(np.corrcoef(projs, entropies)[0, 1])
            else:
                corr_entropy = 0.0
            
            # Entropy方向: 用centered_vecs对entropy做回归
            # v = w * entropy + bias, 用最小二乘
            # 简化: 用PC1方向和entropy相关的比例来近似
            # entropy_axis ≈ sign(corr) * pc1 (因为PC1是最大方差方向)
            entropy_component = corr_entropy * pc1  # 正相关时沿PC1正方向
        else:
            corr_entropy = 0.0
            entropy_component = np.zeros(d)
        
        # (b) Position axis: 用长prompt检查位置与PC1投影
        test_prompt_long = "The apple is a kind of fruit and the banana is also"
        resid_full, seq_len = get_residual_full_seq(model, tokenizer, test_prompt_long, layer_idx, device)
        
        if resid_full is not None and seq_len > 4:
            pos_projs = []
            for pos in range(seq_len):
                centered = resid_full[pos] - vec_mean
                pos_projs.append(float(np.dot(centered, pc1)))
            positions = list(range(seq_len))
            
            if np.std(pos_projs) > 1e-10:
                corr_position = float(np.corrcoef(positions, pos_projs)[0, 1])
            else:
                corr_position = 0.0
            
            position_component = corr_position * pc1
        else:
            corr_position = 0.0
            position_component = np.zeros(d)
        
        # (c) Template axis: 不同模板间PC1投影差异
        template_prompts = {}
        for tmpl_key in ["is_a", "category_of", "classified_as", "simple_answer"]:
            template_prompts[tmpl_key] = []
            for obj in obj_dict.get("fruit", [])[:3]:
                template_prompts[tmpl_key].append(TEMPLATES_ALL[tmpl_key].format(obj=obj))
        
        template_mean_projs = {}
        for tmpl_key, prompts in template_prompts.items():
            projs = []
            for p in prompts:
                resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
                if resid is not None:
                    centered = resid - vec_mean
                    projs.append(float(np.dot(centered, pc1)))
            if projs:
                template_mean_projs[tmpl_key] = float(np.mean(projs))
        
        if len(template_mean_projs) >= 2:
            template_spread = float(np.std(list(template_mean_projs.values())))
        else:
            template_spread = 0.0
        
        # 模板成分: 与is_a模板偏差方向
        if "is_a" in template_mean_projs and len(template_mean_projs) >= 2:
            other_projs = [v for k, v in template_mean_projs.items() if k != "is_a"]
            template_shift = float(np.mean(other_projs)) - template_mean_projs["is_a"]
            template_component = template_shift * pc1
        else:
            template_shift = 0.0
            template_component = np.zeros(d)
        
        # (d) Readout axis: PC1与W_U的对齐
        W_U = get_W_U(model, model_name)
        readout_alignment = 0.0
        readout_component = np.zeros(d)
        
        if W_U is not None:
            try:
                U, S, Vt = np.linalg.svd(W_U, full_matrices=False)
                wu_pc1 = U[:, 0]
                readout_alignment = float(np.dot(pc1, wu_pc1))
                # readout成分: PC1在W_U第一左奇异向量上的投影
                readout_component = readout_alignment * wu_pc1
            except:
                pass
        
        # ---- 3. 分解: 计算各成分占PC1方差的比例 ----
        pc1_norm_sq = float(np.dot(pc1, pc1))
        
        # 正交化各成分(按entropy>position>template>readout顺序Gram-Schmidt)
        components = []
        component_names = ["entropy", "position", "template", "readout"]
        component_vecs = [entropy_component, position_component, template_component, readout_component]
        
        ortho_vecs = []
        residual_pc1 = pc1.copy()
        
        for name, vec in zip(component_names, component_vecs):
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-10:
                ortho_vecs.append(np.zeros(d))
                continue
            # 从residual_pc1中提取与该方向对齐的成分
            proj_on_vec = np.dot(residual_pc1, vec) / max(vec_norm**2, 1e-10) * vec
            proj_norm = np.linalg.norm(proj_on_vec)
            ortho_vecs.append(proj_on_vec if proj_norm > 1e-10 else np.zeros(d))
            # 从residual中减去
            residual_pc1 = residual_pc1 - proj_on_vec
        
        # 计算各成分的方差占比
        component_ratios = {}
        for name, vec in zip(component_names, ortho_vecs):
            vec_norm = np.linalg.norm(vec)
            ratio = (vec_norm**2) / max(pc1_norm_sq, 1e-10)
            component_ratios[name] = round(ratio, 4)
        
        residual_ratio = (np.linalg.norm(residual_pc1)**2) / max(pc1_norm_sq, 1e-10)
        component_ratios["residual"] = round(residual_ratio, 4)
        
        # ---- 4. 记录结果 ----
        layer_result = {
            "pc1_var_ratio": round(float(eigvals[0] / max(sum(eigvals[:50]), 1e-10)), 4),
            "pc1_entropy_correlation": round(corr_entropy, 4),
            "pc1_position_correlation": round(corr_position, 4),
            "template_spread": round(template_spread, 4),
            "template_shift": round(template_shift, 4),
            "readout_alignment": round(readout_alignment, 4),
            "component_ratios": component_ratios,
            "template_mean_projs": {k: round(v, 4) for k, v in template_mean_projs.items()},
        }
        
        plog(f"    PC1 decomposition: entropy={component_ratios['entropy']:.3f}, "
             f"position={component_ratios['position']:.3f}, "
             f"template={component_ratios['template']:.3f}, "
             f"readout={component_ratios['readout']:.3f}, "
             f"residual={component_ratios['residual']:.3f}")
        plog(f"    Correlations: entropy={corr_entropy:.4f}, position={corr_position:.4f}, "
             f"readout={readout_alignment:.4f}")
        
        results[f"L{layer_idx}"] = layer_result
    
    return results


# ==================== Exp3: 类别净化策略搜索 ====================
def exp3_purification_strategy_search(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    对6个类别测试6种净化策略, 找每个类别的最优策略:
    
    策略:
    1. raw: 原始类别中心差分
    2. no_pc1: 去PC1后的方向
    3. no_top3pc: 去前3个PC后的方向
    4. disentangle: 去竞争类混叠
    5. no_pc1+disentangle: 先去PC1再去混叠
    6. no_top3pc+disentangle: 先去前3个PC再去混叠
    
    指标: candidate margin selectivity (相对于random对照)
    """
    plog("=== Exp3: 类别净化策略搜索 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    key_layers = sorted(set([
        n_layers // 3,
        n_layers // 2,
        2 * n_layers // 3,
    ]))
    
    n_obj = 5 if round_num == 1 else 8
    test_cats = ["fruit", "animal", "vehicle", "tool", "furniture", "clothing"]
    results = {}
    
    for layer_idx in key_layers:
        plog(f"  Layer L{layer_idx}...")
        layer_result = {}
        
        # ---- 1. 收集所有类别的残差 ----
        cat_vecs_dict = {}
        for cat in test_cats:
            objs = obj_dict.get(cat, [])[:n_obj]
            vecs = []
            for obj in objs:
                p = TEMPLATES_ALL["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, p, layer_idx, device)
                if resid is not None:
                    vecs.append(resid)
            if len(vecs) >= 2:
                cat_vecs_dict[cat] = vecs
        
        if len(cat_vecs_dict) < 3:
            continue
        
        # ---- 2. 估计PCA ----
        all_prompts = []
        for cat in test_cats:
            objs = obj_dict.get(cat, [])[:n_obj]
            for obj in objs:
                all_prompts.append(TEMPLATES_ALL["is_a"].format(obj=obj))
        
        cov, eigvals, eigvecs, vec_mean = estimate_pcs_at_layer(
            model, tokenizer, all_prompts, layer_idx, device)
        
        if eigvecs is None:
            continue
        
        pc1 = eigvecs[:, 0]
        top3_pcs = eigvecs[:, :3]
        
        # ---- 3. 对每个类别计算6种策略的方向 ----
        for target_cat in test_cats:
            if target_cat not in cat_vecs_dict:
                continue
            
            plog(f"    Testing {target_cat}...")
            cat_result = {}
            
            target_center = np.mean(cat_vecs_dict[target_cat], axis=0)
            compete_cats = [c for c in test_cats if c != target_cat]
            compete_centers = []
            for cc in compete_cats:
                if cc in cat_vecs_dict:
                    compete_centers.append(np.mean(cat_vecs_dict[cc], axis=0))
            
            if not compete_centers:
                continue
            
            compete_mean = np.mean(compete_centers, axis=0)
            
            # ---- 方向计算 ----
            # raw: 目标类别中心 - 竞争类别中心
            diff_raw = target_center - compete_mean
            diff_raw_norm = np.linalg.norm(diff_raw)
            if diff_raw_norm < 1e-10:
                continue
            dir_raw = diff_raw / diff_raw_norm
            
            # no_pc1: 去PC1
            diff_no_pc1 = diff_raw - np.dot(diff_raw, pc1) * pc1
            norm_no_pc1 = np.linalg.norm(diff_no_pc1)
            dir_no_pc1 = diff_no_pc1 / norm_no_pc1 if norm_no_pc1 > 1e-10 else dir_raw
            
            # no_top3pc: 去前3个PC
            diff_no_top3pc = diff_raw.copy()
            for k in range(min(3, eigvecs.shape[1])):
                diff_no_top3pc -= np.dot(diff_no_top3pc, eigvecs[:, k]) * eigvecs[:, k]
            norm_no_top3pc = np.linalg.norm(diff_no_top3pc)
            dir_no_top3pc = diff_no_top3pc / norm_no_top3pc if norm_no_top3pc > 1e-10 else dir_raw
            
            # disentangle: 去竞争类混叠
            # 对每个竞争类别, 去除目标方向在竞争类方向上的投影
            diff_disentangle = diff_raw.copy()
            for cc in compete_cats:
                if cc not in cat_vecs_dict:
                    continue
                cc_center = np.mean(cat_vecs_dict[cc], axis=0)
                cc_diff = cc_center - compete_mean
                cc_norm = np.linalg.norm(cc_diff)
                if cc_norm > 1e-10:
                    cc_dir = cc_diff / cc_norm
                    diff_disentangle -= np.dot(diff_disentangle, cc_dir) * cc_dir
            norm_disentangle = np.linalg.norm(diff_disentangle)
            dir_disentangle = diff_disentangle / norm_disentangle if norm_disentangle > 1e-10 else dir_raw
            
            # no_pc1+disentangle
            diff_combo1 = diff_no_pc1.copy()
            for cc in compete_cats:
                if cc not in cat_vecs_dict:
                    continue
                cc_center = np.mean(cat_vecs_dict[cc], axis=0)
                cc_diff = cc_center - compete_mean
                cc_norm = np.linalg.norm(cc_diff)
                if cc_norm > 1e-10:
                    cc_dir = cc_diff / cc_norm
                    diff_combo1 -= np.dot(diff_combo1, cc_dir) * cc_dir
            norm_combo1 = np.linalg.norm(diff_combo1)
            dir_combo1 = diff_combo1 / norm_combo1 if norm_combo1 > 1e-10 else dir_raw
            
            # no_top3pc+disentangle
            diff_combo2 = diff_no_top3pc.copy()
            for cc in compete_cats:
                if cc not in cat_vecs_dict:
                    continue
                cc_center = np.mean(cat_vecs_dict[cc], axis=0)
                cc_diff = cc_center - compete_mean
                cc_norm = np.linalg.norm(cc_diff)
                if cc_norm > 1e-10:
                    cc_dir = cc_diff / cc_norm
                    diff_combo2 -= np.dot(diff_combo2, cc_dir) * cc_dir
            norm_combo2 = np.linalg.norm(diff_combo2)
            dir_combo2 = diff_combo2 / norm_combo2 if norm_combo2 > 1e-10 else dir_raw
            
            strategies = {
                "raw": (dir_raw, diff_raw_norm),
                "no_pc1": (dir_no_pc1, norm_no_pc1),
                "no_top3pc": (dir_no_top3pc, norm_no_top3pc),
                "disentangle": (dir_disentangle, norm_disentangle),
                "no_pc1+disentangle": (dir_combo1, norm_combo1),
                "no_top3pc+disentangle": (dir_combo2, norm_combo2),
            }
            
            # ---- 4. 测试每种策略 ----
            # 用目标类别的第一个对象做测试
            test_obj = obj_dict[target_cat][0]
            prompt = TEMPLATES_ALL["is_a"].format(obj=test_obj)
            
            base_logits = get_final_logits(model, tokenizer, prompt, device)
            base_margin, _, _ = compute_en_family_margin(base_logits, tokenizer, target_cat, compete_cats)
            
            strategy_results = {}
            
            for strat_name, (direction, dir_norm) in strategies.items():
                # 注入方向, 用自然delta范数缩放
                natural_norm = compute_natural_delta_norm(model, tokenizer, obj_dict, layer_idx, device, target_cat, n_obj)
                delta = direction * natural_norm
                
                logits_patch = run_with_additive_patch(model, tokenizer, prompt, device, layer_idx, delta)
                margin_patch, _, _ = compute_en_family_margin(logits_patch, tokenizer, target_cat, compete_cats)
                
                selectivity = margin_patch - base_margin
                
                strategy_results[strat_name] = {
                    "margin": round(margin_patch, 4),
                    "selectivity": round(selectivity, 4),
                    "dir_norm": round(dir_norm, 4),
                    "cos_with_pc1": round(float(abs(np.dot(direction, pc1))), 4),
                }
            
            # 随机方向对照(5个)
            random_selectivities = []
            for _ in range(5):
                rand_dir = np.random.randn(len(pc1))
                rand_dir = rand_dir / np.linalg.norm(rand_dir) * natural_norm
                
                logits_rand = run_with_additive_patch(model, tokenizer, prompt, device, layer_idx, rand_dir)
                margin_rand, _, _ = compute_en_family_margin(logits_rand, tokenizer, target_cat, compete_cats)
                random_selectivities.append(margin_rand - base_margin)
            
            strategy_results["random_control"] = {
                "mean_selectivity": round(float(np.mean(random_selectivities)), 4),
                "std_selectivity": round(float(np.std(random_selectivities)), 4),
            }
            
            cat_result["base_margin"] = round(base_margin, 4)
            cat_result["strategies"] = strategy_results
            
            # 找最优策略
            best_strat = max(
                [(k, v["selectivity"]) for k, v in strategy_results.items() 
                 if k not in ["random_control"]],
                key=lambda x: x[1]
            )
            cat_result["best_strategy"] = best_strat[0]
            cat_result["best_selectivity"] = best_strat[1]
            
            # selectivity > 2 * random_std ?
            random_std = float(np.std(random_selectivities))
            cat_result["significant"] = best_strat[1] > 2 * random_std if random_std > 1e-6 else best_strat[1] > 0.01
            
            plog(f"      {target_cat}: best={best_strat[0]}({best_strat[1]:.4f}), "
                 f"random_mean={strategy_results['random_control']['mean_selectivity']:.4f}")
            
            layer_result[target_cat] = cat_result
        
        results[f"L{layer_idx}"] = layer_result
    
    return results


def compute_natural_delta_norm(model, tokenizer, obj_dict, layer_idx, device, cat, n=5):
    """计算某层某类别的自然delta范数"""
    objs = obj_dict.get(cat, [])[:n]
    vecs = []
    for obj in objs:
        prompt = TEMPLATES_ALL["is_a"].format(obj=obj)
        resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, layer_idx, device)
        if resid is not None:
            vecs.append(resid)
    
    if len(vecs) < 2:
        return 1.0
    
    center = np.mean(vecs, axis=0)
    delta_norms = [np.linalg.norm(v - center) for v in vecs]
    return float(np.mean(delta_norms))


# ==================== Exp4: DS7B模板稳健性测试 ====================
def exp4_template_robustness(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    测试5个模板在3个模型上的表现:
    - 基线生成质量(是否触发数学模式)
    - 候选边际
    - 注入敏感性
    
    目标: 找到不触发DS7B数学模式的模板
    """
    plog("=== Exp4: 模板稳健性测试 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    n_obj = 3 if round_num == 1 else 5
    test_objects = ["car", "dog", "apple", "hammer", "chair"]
    results = {}
    
    for tmpl_key, tmpl_str in TEMPLATES_ALL.items():
        plog(f"  Template: {tmpl_key} -> '{tmpl_str}'")
        tmpl_result = {}
        
        # ---- 1. 基线生成 ----
        baseline_gens = {}
        for obj in test_objects[:3]:
            prompt = tmpl_str.format(obj=obj)
            
            # 不注入的基线生成
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            gen_kwargs = dict(max_new_tokens=20, do_sample=False, repetition_penalty=1.2)
            with torch.no_grad():
                gen_ids = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            
            # 检测数学模式触发: 包含数字/公式/数学符号
            math_indicators = ["equation", "formula", "matrix", "vector", "function", 
                             "graph", "theorem", "integral", "polynomial", "1", "2", "3",
                             "0.", "0x", "∞", "Σ", "π", "∫"]
            math_trigger = any(ind in gen_text[len(prompt):] for ind in math_indicators)
            
            baseline_gens[obj] = {
                "text": gen_text,
                "math_triggered": math_trigger,
                "gen_part": gen_text[len(prompt):].strip()[:100],
            }
        
        # 统计数学触发率
        math_trigger_rate = sum(1 for v in baseline_gens.values() if v["math_triggered"]) / len(baseline_gens)
        tmpl_result["math_trigger_rate"] = round(math_trigger_rate, 4)
        tmpl_result["baseline_gens"] = baseline_gens
        
        plog(f"    Math trigger rate: {math_trigger_rate:.2%}")
        
        # ---- 2. 基线候选边际 ----
        margin_results = {}
        key_layer = n_layers // 2
        
        for obj, obj_cat in [("car", "vehicle"), ("dog", "animal"), ("apple", "fruit")]:
            prompt = tmpl_str.format(obj=obj)
            other_cats = [c for c in ["fruit", "animal", "vehicle", "tool", "furniture", "clothing"] if c != obj_cat]
            
            logits = get_final_logits(model, tokenizer, prompt, device)
            margin, target_mean, compete_mean = compute_en_family_margin(logits, tokenizer, obj_cat, other_cats)
            entropy = logit_entropy(logits)
            
            margin_results[obj] = {
                "margin": round(margin, 4),
                "target_mean": round(target_mean, 4),
                "compete_mean": round(compete_mean, 4),
                "entropy": round(entropy, 4),
            }
        
        tmpl_result["margins"] = margin_results
        
        # ---- 3. 注入敏感性(只在Qwen3/DS7B上做, 避免GLM4耗时过长) ----
        if model_name in ["qwen3", "deepseek7b"]:
            # 用fruit类别做注入测试
            objs = obj_dict.get("fruit", [])[:n_obj]
            other_cats = ["animal", "vehicle", "tool", "furniture", "clothing"]
            
            cat_vecs = []
            for obj in objs:
                p = tmpl_str.format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, p, key_layer, device)
                if resid is not None:
                    cat_vecs.append(resid)
            
            if len(cat_vecs) >= 2:
                cat_center = np.mean(cat_vecs, axis=0)
                other_vecs = []
                for oc in ["animal", "vehicle"]:
                    for obj in obj_dict.get(oc, [])[:3]:
                        p = tmpl_str.format(obj=obj)
                        resid, _ = get_residual_at_layer_pos(model, tokenizer, p, key_layer, device)
                        if resid is not None:
                            other_vecs.append(resid)
                
                if other_vecs:
                    other_center = np.mean(other_vecs, axis=0)
                    diff = cat_center - other_center
                    diff_norm = np.linalg.norm(diff)
                    
                    if diff_norm > 1e-10:
                        diff_dir = diff / diff_norm
                        test_obj = objs[0]
                        prompt = tmpl_str.format(obj=test_obj)
                        
                        base_logits = get_final_logits(model, tokenizer, prompt, device)
                        base_margin, _, _ = compute_en_family_margin(base_logits, tokenizer, "fruit", other_cats)
                        
                        # 注入
                        delta = diff_dir * compute_natural_delta_norm(model, tokenizer, obj_dict, key_layer, device, "fruit", n_obj)
                        logits_patch = run_with_additive_patch(model, tokenizer, prompt, device, key_layer, delta)
                        margin_patch, _, _ = compute_en_family_margin(logits_patch, tokenizer, "fruit", other_cats)
                        
                        tmpl_result["injection_sensitivity"] = {
                            "base_margin": round(base_margin, 4),
                            "patched_margin": round(margin_patch, 4),
                            "selectivity": round(margin_patch - base_margin, 4),
                        }
        
        results[tmpl_key] = tmpl_result
    
    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        return
    
    obj_dict = ROUNDS[round_num]
    
    plog(f"Phase 468: {model_name}, Round {round_num}")
    plog(f"Objects per category: {len(list(obj_dict.values())[0])}")
    
    # ---- 1. 加载模型 ----
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    t_load = time.time() - t0
    plog(f"Model loaded in {t_load:.0f}s")
    
    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    # ---- 2. 运行实验 ----
    all_results = {
        "model": model_name,
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }
    
    # Exp1: PC1因果验证
    t1 = time.time()
    all_results["exp1_pc1_causal"] = exp1_pc1_causal_verification(
        model, tokenizer, model_name, device, obj_dict, round_num)
    plog(f"Exp1 done in {time.time()-t1:.0f}s")
    
    # Exp2: PC1成分分解
    t2 = time.time()
    all_results["exp2_pc1_decomposition"] = exp2_pc1_decomposition(
        model, tokenizer, model_name, device, obj_dict, round_num)
    plog(f"Exp2 done in {time.time()-t2:.0f}s")
    
    # Exp3: 类别净化策略搜索
    t3 = time.time()
    all_results["exp3_purification_search"] = exp3_purification_strategy_search(
        model, tokenizer, model_name, device, obj_dict, round_num)
    plog(f"Exp3 done in {time.time()-t3:.0f}s")
    
    # Exp4: 模板稳健性
    t4 = time.time()
    all_results["exp4_template_robustness"] = exp4_template_robustness(
        model, tokenizer, model_name, device, obj_dict, round_num)
    plog(f"Exp4 done in {time.time()-t4:.0f}s")
    
    # ---- 3. 保存结果 ----
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase468_{model_name}_r{round_num}.json"
    
    # 将numpy类型转python
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
    plog(f"Phase 468 {model_name} Round {round_num} complete in {total_time:.0f}s")


if __name__ == "__main__":
    main()
