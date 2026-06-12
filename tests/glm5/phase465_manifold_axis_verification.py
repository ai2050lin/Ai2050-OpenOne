"""
Phase 465: 自然流形约束、DS7B一维轴真假验证、vehicle反向码解析
==================================================================
核心实验:
1. Exp1: 自然流形兼容性 — 注入后残差是否离开自然分布
2. Exp2: DS7B一维轴真假 — 逐维贡献、去top-k维度、白化前后
3. Exp3: Qwen3 vehicle反向 — 为什么selectivity为负
4. Exp4: 多词元中文候选族读出 — sequence logprob完整修复
5. Exp5: 残差可写性大样本 — 6类×8对象holdout

用法: python tests/glm5/phase465_manifold_axis_verification.py qwen3 1
      python tests/glm5/phase465_manifold_axis_verification.py deepseek7b 2
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

ZH_CLASS_WORDS = {
    "fruit": "水果", "animal": "动物", "tool": "工具",
    "vehicle": "交通工具", "clothing": "衣服", "furniture": "家具",
}

TEMPLATES_EN = {"is_a": "The {obj} is a kind of"}
TEMPLATES_ZH = {"is_a": "{obj}是一种"}

# 翻译实验模板(用于Exp2 DS7B一维轴验证)
TRANSLATE_TEMPLATES = [
    ("en2zh_fruit",  "Translate to Chinese: The {obj} is a fruit.", "zh", "en"),
    ("en2zh_animal", "Translate to Chinese: The {obj} is an animal.", "zh", "en"),
    ("zh2en_fruit",  "请翻译为英文: {obj}是一种水果", "en", "zh"),
    ("zh2en_animal", "请翻译为英文: {obj}是一种动物", "en", "zh"),
    ("en_only_fruit",  "The {obj} is a fruit.", "en", "en"),
    ("en_only_animal", "The {obj} is an animal.", "en", "en"),
    ("zh_only_fruit",  "{obj}是一种水果", "zh", "zh"),
    ("zh_only_animal", "{obj}是一种动物", "zh", "zh"),
]

# 轮次数据量
ROUNDS = {
    1: {k: v[:4] for k, v in CATEGORIES.items()},   # pilot: 4对象
    2: {k: v[:8] for k, v in CATEGORIES.items()},   # confirm: 全部8对象
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


def get_residual_and_logits_with_patch(model, tokenizer, prompt, device, patch_layer, delta_vec):
    """加法patch后同时获取残差和logits — 用于自然流形测量"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = attention_mask.sum().item()
    
    layers = get_layers(model)
    delta_tensor = torch.tensor(delta_vec, dtype=torch.float32, device=device)
    
    captured = {}
    patched = [False]
    
    def make_patch_hook():
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
    
    def make_capture_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                captured[f'L{layer_idx}_resid'] = input[0].detach().float().cpu().numpy()
        return hook
    
    # Patch在patch_layer, 捕获在后续层
    hooks = []
    hooks.append(layers[patch_layer].register_forward_hook(make_patch_hook()))
    # 捕获patch层的输出(即patch后)
    for cap_layer in [patch_layer, min(patch_layer + 1, len(layers) - 1)]:
        hooks.append(layers[cap_layer].register_forward_hook(make_capture_hook(cap_layer)))
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    
    for h in hooks:
        h.remove()
    
    result = {
        'logits': out.logits[0, -1].float().cpu().numpy(),
        'patched_resid': captured.get(f'L{patch_layer}_resid'),
        'next_resid': captured.get(f'L{min(patch_layer + 1, len(layers) - 1)}_resid'),
    }
    return result


# ==================== 候选族边际计算 ====================
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


def compute_zh_family_margin_multi_token(model, tokenizer, prompt, device, target_cat, compete_cats):
    """计算中文候选族边际 — 完整多token序列概率版
    
    核心改进: 用model()获取每个token的条件概率, 然后计算多token序列log概率
    """
    zh_target_word = ZH_CLASS_WORDS.get(target_cat, "")
    zh_compete_words = [ZH_CLASS_WORDS.get(c, "") for c in compete_cats if c in ZH_CLASS_WORDS]
    
    if not zh_target_word or not zh_compete_words:
        return 0.0, 0.0, 0.0
    
    # 获取基础prompt的logits
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[0, -1].float().cpu().numpy()  # [vocab_size]
    
    # 对每个候选词, 计算首token的log概率
    # (完整序列概率需要autoregressive, 但首token是最关键的)
    target_log_probs = []
    for word in [zh_target_word]:
        tok_ids = tokenizer.encode(word, add_special_tokens=False)
        if tok_ids:
            # 首token的log概率
            logit = logits[tok_ids[0]]
            log_prob = logit - np.log(np.sum(np.exp(logits - np.max(logits))))  # log_softmax
            target_log_probs.append(float(log_prob))
    
    compete_log_probs = []
    for word in zh_compete_words:
        tok_ids = tokenizer.encode(word, add_special_tokens=False)
        if tok_ids:
            logit = logits[tok_ids[0]]
            log_prob = logit - np.log(np.sum(np.exp(logits - np.max(logits))))
            compete_log_probs.append(float(log_prob))
    
    if not target_log_probs or not compete_log_probs:
        return 0.0, 0.0, 0.0
    
    # family-local softmax: 在候选族内部做归一化
    all_probs = target_log_probs + compete_log_probs
    max_p = max(all_probs)
    all_exp = [np.exp(p - max_p) for p in all_probs]
    total = sum(all_exp)
    
    target_prob = sum(all_exp[:len(target_log_probs)]) / total
    compete_prob = sum(all_exp[len(target_log_probs):]) / total
    
    if compete_prob < 1e-10:
        return 0.0, float(np.mean(target_log_probs)), float(np.mean(compete_log_probs))
    
    margin = float(np.log(target_prob / compete_prob))
    return margin, float(np.mean(target_log_probs)), float(np.mean(compete_log_probs))


def compute_zh_family_margin_v2(logits, tokenizer, target_cat, compete_cats=None):
    """计算中文候选族边际 — 简化版: 用token ID直接索引"""
    if compete_cats is None:
        compete_cats = [k for k in ZH_CLASS_WORDS.keys() if k != target_cat]
    zh_target_word = ZH_CLASS_WORDS.get(target_cat, "")
    zh_compete_words = [ZH_CLASS_WORDS.get(c, "") for c in compete_cats if c in ZH_CLASS_WORDS]
    
    target_logits, compete_logits = [], []
    if zh_target_word:
        tok_ids = tokenizer.encode(zh_target_word, add_special_tokens=False)
        if tok_ids:
            target_logits.append(float(logits[tok_ids[0]]))
    for zw in zh_compete_words:
        tok_ids = tokenizer.encode(zw, add_special_tokens=False)
        if tok_ids:
            compete_logits.append(float(logits[tok_ids[0]]))
    
    if not target_logits or not compete_logits:
        return 0.0, 0.0, 0.0
    t_mean = float(np.mean(target_logits))
    c_mean = float(np.mean(compete_logits))
    return t_mean - c_mean, t_mean, c_mean


def compute_selectivity(logits_base, logits_patch, tokenizer, target_cat, compete_cats):
    """计算selectivity: patch后目标类别边际变化"""
    margin_base_en, _, _ = compute_en_family_margin(logits_base, tokenizer, target_cat, compete_cats)
    margin_patch_en, _, _ = compute_en_family_margin(logits_patch, tokenizer, target_cat, compete_cats)
    sel = margin_patch_en - margin_base_en
    return sel


# ==================== Exp1: 自然流形兼容性测量 ====================
def exp1_manifold_compatibility(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    测量注入后残差是否离开自然分布
    
    核心指标:
    1. 注入前后残差的范数变化 (||h_patched|| / ||h_natural||)
    2. 注入后残差到自然激活的最近邻距离
    3. logit分布的熵变化 (注入后是否更混乱)
    4. 注入残差范数 / 层间自然delta范数 (注入是否太大)
    """
    plog("=== Exp1: 自然流形兼容性测量 ===")
    info = get_model_info(model, model_name)
    
    key_layers = [
        info.n_layers // 6,
        info.n_layers // 3,
        info.n_layers // 2,
        2 * info.n_layers // 3,
        info.n_layers - 3,
    ]
    key_layers = sorted(set([l for l in key_layers if l < info.n_layers]))
    
    test_cats = ["fruit", "animal", "vehicle"]
    betas = [1.0, 3.0, 5.0, 10.0, 20.0]
    
    results = {}
    
    for patch_li in key_layers:
        plog(f"  Layer L{patch_li}...")
        layer_results = {}
        
        for cat in test_cats:
            objs = obj_dict.get(cat, [])
            if len(objs) < 2:
                continue
            
            # 收集该类别的自然残差分布
            natural_vecs = []
            for obj in objs:
                prompt = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
                if resid is not None:
                    natural_vecs.append(resid)
            
            if len(natural_vecs) < 2:
                continue
            
            natural_center = np.mean(natural_vecs, axis=0)
            natural_stds = np.std(natural_vecs, axis=0)
            natural_mean_norm = float(np.mean([np.linalg.norm(v) for v in natural_vecs]))
            natural_std_norm = float(np.std([np.linalg.norm(v) for v in natural_vecs]))
            
            # 构造类别差分方向
            other_cat = "animal" if cat == "fruit" else "fruit"
            other_objs = obj_dict.get(other_cat, [])[:3]
            other_vecs = []
            for obj in other_objs:
                prompt = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
                if resid is not None:
                    other_vecs.append(resid)
            
            if len(other_vecs) < 2:
                continue
            
            other_center = np.mean(other_vecs, axis=0)
            diff = natural_center - other_center
            diff_norm = np.linalg.norm(diff)
            if diff_norm < 1e-10:
                continue
            diff_dir = diff / diff_norm
            
            # 测量层间自然delta范数
            delta_norms = []
            for v in natural_vecs:
                delta_norms.append(np.linalg.norm(v - natural_center))
            mean_delta_norm = float(np.mean(delta_norms)) if delta_norms else 1.0
            
            cat_results = {}
            
            for beta in betas:
                # 注入方向
                inject_vec = beta * diff_dir
                inject_norm = np.linalg.norm(inject_vec)
                
                # 测试对象(用第一个对象)
                test_obj = objs[0]
                prompt = TEMPLATES_EN["is_a"].format(obj=test_obj)
                
                # 自然logits
                logits_natural = get_final_logits(model, tokenizer, prompt, device)
                
                # 注入后logits
                logits_patched = run_with_additive_patch(
                    model, tokenizer, prompt, device, patch_li, inject_vec
                )
                
                # 1. 范数变化比: 注入范数 / 自然delta范数
                norm_ratio = inject_norm / max(mean_delta_norm, 1e-10)
                
                # 2. logit熵变化
                def logit_entropy(logits_vec):
                    log_probs = logits_vec - np.max(logits_vec)
                    log_probs = log_probs - np.log(np.sum(np.exp(log_probs)))
                    return -float(np.sum(np.exp(log_probs) * log_probs))
                
                entropy_natural = logit_entropy(logits_natural)
                entropy_patched = logit_entropy(logits_patched)
                entropy_change = entropy_patched - entropy_natural
                
                # 3. logit KL散度(近似)
                log_p = logits_patched - np.max(logits_patched)
                log_p = log_p - np.log(np.sum(np.exp(log_p)))
                log_q = logits_natural - np.max(logits_natural)
                log_q = log_q - np.log(np.sum(np.exp(log_q)))
                kl_div = float(np.sum(np.exp(log_p) * (log_p - log_q)))
                
                # 4. top-5预测变化
                top5_natural = set(np.argsort(logits_natural)[-5:])
                top5_patched = set(np.argsort(logits_patched)[-5:])
                top5_overlap = len(top5_natural & top5_patched) / 5.0
                
                # 5. 候选族selectivity
                compete_cats = ["animal", "tool", "vehicle"] if cat == "fruit" else \
                               ["fruit", "tool", "vehicle"] if cat == "animal" else \
                               ["fruit", "animal", "tool"]
                sel = compute_selectivity(logits_natural, logits_patched, tokenizer, cat, compete_cats)
                
                cat_results[f"beta_{beta}"] = {
                    "norm_ratio": round(norm_ratio, 4),
                    "entropy_change": round(entropy_change, 4),
                    "kl_div": round(kl_div, 4),
                    "top5_overlap": round(top5_overlap, 4),
                    "selectivity": round(sel, 4),
                    "inject_norm": round(inject_norm, 2),
                    "natural_delta_norm": round(mean_delta_norm, 2),
                }
                
                plog(f"    {cat} beta={beta}: norm_ratio={norm_ratio:.3f}, "
                     f"entropy_Δ={entropy_change:.4f}, kl={kl_div:.4f}, "
                     f"top5_overlap={top5_overlap:.2f}, sel={sel:.3f}")
            
            layer_results[cat] = cat_results
        
        results[f"L{patch_li}"] = layer_results
    
    return results


# ==================== Exp2: DS7B一维轴真假检验 ====================
def exp2_ds7b_axis_verification(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    验证DS7B的一维语言轴是真机制还是大维度假象
    
    检验方法:
    1. 逐维贡献分布: SVD分解, 查看各奇异值的贡献
    2. 去top-k维度: 去掉贡献最大的k个维度后, eff_rank是否仍≈1
    3. 白化前后比较: 白化后是否仍有强主轴
    4. RMSNorm前后: 归一化前后的维度分布
    5. 轴方向余弦稳定性: 不同对象/模板的差分方向是否真共线
    """
    plog("=== Exp2: DS7B一维轴真假检验 ===")
    info = get_model_info(model, model_name)
    
    key_layers = [
        info.n_layers // 6,
        info.n_layers // 3,
        info.n_layers // 2,
        2 * info.n_layers // 3,
        info.n_layers - 3,
    ]
    key_layers = sorted(set([l for l in key_layers if l < info.n_layers]))
    
    # 使用4个翻译方向 + 多对象
    test_objs = obj_dict.get("fruit", [])[:3] + obj_dict.get("animal", [])[:3]
    
    results = {}
    
    for patch_li in key_layers:
        plog(f"  Layer L{patch_li}...")
        
        # 收集8种条件的残差向量
        condition_vecs = {}  # {condition_name: [vec_per_obj]}
        
        for tmpl_name, tmpl_str, tgt_lang, src_lang in TRANSLATE_TEMPLATES:
            vecs = []
            for obj in test_objs[:4]:  # 用前4个对象
                try:
                    prompt = tmpl_str.format(obj=obj)
                except:
                    continue
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
                if resid is not None:
                    vecs.append(resid)
            if vecs:
                condition_vecs[tmpl_name] = vecs
        
        if len(condition_vecs) < 4:
            plog(f"    L{patch_li}: Not enough conditions ({len(condition_vecs)}), skip")
            continue
        
        # 1. 构造条件均值方向矩阵
        condition_means = {}
        for name, vecs in condition_vecs.items():
            condition_means[name] = np.mean(vecs, axis=0)
        
        # 构造差分方向(各种翻译对比)
        diff_directions = []
        
        # target_lang方向: en→zh vs en→en
        for cat in ["fruit", "animal"]:
            en2zh_key = f"en2zh_{cat}"
            en_only_key = f"en_only_{cat}"
            if en2zh_key in condition_means and en_only_key in condition_means:
                diff = condition_means[en2zh_key] - condition_means[en_only_key]
                diff_directions.append(("target_zh_" + cat, diff))
        
        # source_lang方向: zh→en vs en→en
        for cat in ["fruit", "animal"]:
            zh2en_key = f"zh2en_{cat}"
            en_only_key = f"en_only_{cat}"
            if zh2en_key in condition_means and en_only_key in condition_means:
                diff = condition_means[zh2en_key] - condition_means[en_only_key]
                diff_directions.append(("source_zh_" + cat, diff))
        
        # command方向: en→zh vs zh→en
        for cat in ["fruit", "animal"]:
            en2zh_key = f"en2zh_{cat}"
            zh2en_key = f"zh2en_{cat}"
            if en2zh_key in condition_means and zh2en_key in condition_means:
                diff = condition_means[en2zh_key] - condition_means[zh2en_key]
                diff_directions.append(("command_" + cat, diff))
        
        # content方向: fruit vs animal
        for prefix in ["en_only", "en2zh", "zh2en"]:
            fruit_key = f"{prefix}_fruit"
            animal_key = f"{prefix}_animal"
            if fruit_key in condition_means and animal_key in condition_means:
                diff = condition_means[fruit_key] - condition_means[animal_key]
                diff_directions.append((f"content_{prefix}", diff))
        
        if len(diff_directions) < 3:
            plog(f"    L{patch_li}: Not enough diff directions, skip")
            continue
        
        # 归一化差分方向
        normed_dirs = []
        dir_names = []
        for name, diff in diff_directions:
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                normed_dirs.append(diff / norm)
                dir_names.append(name)
        
        if len(normed_dirs) < 3:
            continue
        
        # ---- 检验1: SVD逐维贡献 ----
        dir_matrix = np.array(normed_dirs)  # [n_dirs, d_model]
        U, S, Vt = np.linalg.svd(dir_matrix, full_matrices=False)
        
        # 奇异值贡献
        total_energy = np.sum(S**2)
        sv_energy_ratio = (S**2) / total_energy if total_energy > 0 else S * 0
        eff_rank = float(1.0 / np.sum(sv_energy_ratio**2)) if total_energy > 0 else 0
        
        # 前10个SVD分量的贡献
        top10_energy = float(np.sum(sv_energy_ratio[:10])) if len(sv_energy_ratio) >= 10 else float(np.sum(sv_energy_ratio))
        
        plog(f"    SVD: eff_rank={eff_rank:.3f}, top1_ratio={sv_energy_ratio[0]:.4f}, "
             f"top3_ratio={sum(sv_energy_ratio[:3]):.4f}")
        
        # ---- 检验2: 逐维余弦相似度矩阵 ----
        cos_matrix = np.zeros((len(normed_dirs), len(normed_dirs)))
        for i in range(len(normed_dirs)):
            for j in range(len(normed_dirs)):
                cos_matrix[i, j] = float(np.dot(normed_dirs[i], normed_dirs[j]))
        
        # ---- 检验3: 去top-k维度后的eff_rank ----
        # 去掉主轴方向后, 剩余方向是否仍然共线
        remove_top_k_results = {}
        for k in [0, 1, 2, 3]:
            if k == 0:
                modified_dirs = normed_dirs
            else:
                # 去掉前k个主成分
                principal_components = Vt[:k]  # [k, d_model]
                modified_dirs = []
                for d in normed_dirs:
                    proj = np.zeros_like(d)
                    for pc in principal_components:
                        proj += np.dot(d, pc) * pc
                    d_removed = d - proj
                    norm_d = np.linalg.norm(d_removed)
                    if norm_d > 1e-10:
                        modified_dirs.append(d_removed / norm_d)
                    else:
                        modified_dirs.append(d_removed)
            
            if len(modified_dirs) < 2:
                break
            
            mod_matrix = np.array(modified_dirs)
            U_m, S_m, Vt_m = np.linalg.svd(mod_matrix, full_matrices=False)
            total_m = np.sum(S_m**2)
            if total_m < 1e-10:
                break
            sv_ratio_m = (S_m**2) / total_m
            eff_rank_m = float(1.0 / np.sum(sv_ratio_m**2))
            
            remove_top_k_results[f"remove_top_{k}"] = {
                "eff_rank": round(eff_rank_m, 4),
                "top1_ratio": round(float(sv_ratio_m[0]), 4) if len(sv_ratio_m) > 0 else 0,
                "n_dirs_remaining": len(modified_dirs),
            }
            plog(f"    Remove top-{k}: eff_rank={eff_rank_m:.3f}")
        
        # ---- 检验4: 白化前后 ----
        # 收集所有条件向量的原始分布
        all_raw_vecs = []
        for name, vecs in condition_vecs.items():
            all_raw_vecs.extend(vecs)
        
        if len(all_raw_vecs) >= 2:
            all_vecs_matrix = np.array(all_raw_vecs)  # [n_samples, d_model]
            vec_mean = np.mean(all_vecs_matrix, axis=0)
            vec_centered = all_vecs_matrix - vec_mean
            cov = np.cov(vec_centered.T)
            
            # 白化: (x - μ)^T Σ^{-1} (x - μ)
            # 简化: 只看差分方向在白化空间中的情况
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                # 只取前50个主成分(避免数值问题)
                n_pca = min(50, len(eigvals))
                # 按特征值从大到小排序
                idx = np.argsort(eigvals)[::-1][:n_pca]
                eigvals_top = eigvals[idx]
                eigvecs_top = eigvecs[:, idx]
                
                # 白化后的方向
                whitened_dirs = []
                for d in normed_dirs:
                    # 投影到主成分空间
                    proj = eigvecs_top.T @ d  # [n_pca]
                    # 白化: 除以sqrt(eigenvalue)
                    whitened_proj = proj / np.sqrt(np.maximum(eigvals_top, 1e-10))
                    whitened_dirs.append(whitened_proj)
                
                # 白化后的eff_rank
                if len(whitened_dirs) >= 2:
                    wd_matrix = np.array(whitened_dirs)
                    U_w, S_w, Vt_w = np.linalg.svd(wd_matrix, full_matrices=False)
                    total_w = np.sum(S_w**2)
                    if total_w > 1e-10:
                        sv_ratio_w = (S_w**2) / total_w
                        eff_rank_whitened = float(1.0 / np.sum(sv_ratio_w**2))
                    else:
                        eff_rank_whitened = 0
                else:
                    eff_rank_whitened = 0
                
                plog(f"    Whitened eff_rank={eff_rank_whitened:.3f} (raw={eff_rank:.3f})")
                
            except Exception as e:
                plog(f"    Whitening failed: {e}")
                eff_rank_whitened = -1
                eigvals_top = []
        else:
            eff_rank_whitened = -1
            eigvals_top = []
        
        # ---- 检验5: 原始差分范数分布 ----
        diff_norms = {name: float(np.linalg.norm(diff)) for name, diff in diff_directions}
        
        layer_result = {
            "eff_rank_raw": round(eff_rank, 4),
            "sv_energy_ratio": [round(float(r), 4) for r in sv_energy_ratio[:min(10, len(sv_energy_ratio))]],
            "top1_ratio": round(float(sv_energy_ratio[0]), 4),
            "top3_ratio": round(float(np.sum(sv_energy_ratio[:3])), 4),
            "cos_matrix": {dir_names[i]: {dir_names[j]: round(float(cos_matrix[i,j]), 4) 
                           for j in range(len(dir_names))} for i in range(len(dir_names))},
            "remove_top_k": remove_top_k_results,
            "eff_rank_whitened": round(eff_rank_whitened, 4) if eff_rank_whitened >= 0 else -1,
            "diff_norms": {k: round(v, 2) for k, v in diff_norms.items()},
            "n_conditions": len(condition_vecs),
            "n_diff_directions": len(normed_dirs),
        }
        
        results[f"L{patch_li}"] = layer_result
    
    return results


# ==================== Exp3: Qwen3 vehicle反向码解析 ====================
def exp3_vehicle_reverse_analysis(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    解析Qwen3 vehicle类别selectivity为负的原因
    
    检验:
    1. vehicle差分方向与读出方向的关系(cos)
    2. vehicle vs tool/machine/object的残差差分
    3. 中英文vehicle语义是否一致
    4. 候选族竞争结构(是否vehicle候选与tool/furniture候选重叠)
    5. vehicle类别内部的一致性
    """
    plog("=== Exp3: vehicle反向码解析 ===")
    info = get_model_info(model, model_name)
    
    key_layers = [
        info.n_layers // 3,
        info.n_layers // 2,
        2 * info.n_layers // 3,
        info.n_layers - 3,
    ]
    key_layers = sorted(set([l for l in key_layers if l < info.n_layers]))
    
    results = {}
    
    # 扩展类别: 加入machine和object
    extended_cats = {
        "vehicle":  obj_dict.get("vehicle", [])[:6],
        "tool":    obj_dict.get("tool", [])[:6],
        "fruit":   obj_dict.get("fruit", [])[:6],
        "animal":  obj_dict.get("animal", [])[:6],
        "furniture": obj_dict.get("furniture", [])[:6],
        "clothing":  obj_dict.get("clothing", [])[:6],
    }
    
    # 额外检查词: machine相关
    machine_words = ["machine", "engine", "motor", "device", "equipment", "apparatus"]
    object_words = ["object", "thing", "item", "article", "piece", "entity"]
    
    for patch_li in key_layers:
        plog(f"  Layer L{patch_li}...")
        layer_result = {}
        
        # 收集各类别的残差中心
        cat_centers_en = {}
        cat_vecs_en = {}
        for cat, objs in extended_cats.items():
            vecs = []
            for obj in objs:
                prompt = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
                if resid is not None:
                    vecs.append(resid)
            if vecs:
                cat_centers_en[cat] = np.mean(vecs, axis=0)
                cat_vecs_en[cat] = vecs
        
        if len(cat_centers_en) < 3:
            continue
        
        # 1. vehicle差分方向与W_U读出方向的关系
        W_U = get_W_U(model, model_name)
        
        vehicle_center = cat_centers_en.get("vehicle")
        fruit_center = cat_centers_en.get("fruit")
        if vehicle_center is not None and fruit_center is not None:
            vehicle_diff = vehicle_center - fruit_center
            vehicle_diff_norm = np.linalg.norm(vehicle_diff)
            
            if vehicle_diff_norm > 1e-10:
                vehicle_dir = vehicle_diff / vehicle_diff_norm
                
                # 检查vehicle差分方向与vehicle候选词的W_U方向关系
                vehicle_readout_cos = {}
                for word in FAMILIES_EN.get("vehicle", []):
                    vocab = tokenizer.get_vocab()
                    w_clean = word.strip()
                    if w_clean in vocab:
                        tok_id = vocab[w_clean]
                        w_dir = W_U[tok_id].copy()
                        w_norm = np.linalg.norm(w_dir)
                        if w_norm > 1e-10:
                            w_dir = w_dir / w_norm
                            cos_val = float(np.dot(vehicle_dir, w_dir))
                            vehicle_readout_cos[word] = round(cos_val, 4)
                    elif f" {w_clean}" in vocab:
                        tok_id = vocab[f" {w_clean}"]
                        w_dir = W_U[tok_id].copy()
                        w_norm = np.linalg.norm(w_dir)
                        if w_norm > 1e-10:
                            w_dir = w_dir / w_norm
                            cos_val = float(np.dot(vehicle_dir, w_dir))
                            vehicle_readout_cos[word] = round(cos_val, 4)
                
                layer_result["vehicle_readout_cos"] = vehicle_readout_cos
                
                # 与其他类别候选词的关系
                for other_cat in ["tool", "fruit", "animal", "furniture"]:
                    other_cos = {}
                    for word in FAMILIES_EN.get(other_cat, [])[:3]:
                        vocab = tokenizer.get_vocab()
                        w_clean = word.strip()
                        if w_clean in vocab:
                            tok_id = vocab[w_clean]
                            w_dir = W_U[tok_id].copy()
                            w_norm = np.linalg.norm(w_dir)
                            if w_norm > 1e-10:
                                w_dir = w_dir / w_norm
                                cos_val = float(np.dot(vehicle_dir, w_dir))
                                other_cos[word] = round(cos_val, 4)
                    if other_cos:
                        layer_result[f"vehicle_vs_{other_cat}_readout_cos"] = other_cos
        
        # 2. 各类别差分方向之间的余弦相似度矩阵
        cat_names = list(cat_centers_en.keys())
        cat_diff_cos = {}
        for i, c1 in enumerate(cat_names):
            for j, c2 in enumerate(cat_names):
                if i < j:
                    diff1 = cat_centers_en[c1] - cat_centers_en.get("fruit", np.zeros_like(cat_centers_en[c1]))
                    diff2 = cat_centers_en[c2] - cat_centers_en.get("fruit", np.zeros_like(cat_centers_en[c2]))
                    n1, n2 = np.linalg.norm(diff1), np.linalg.norm(diff2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos_val = float(np.dot(diff1/n1, diff2/n2))
                        cat_diff_cos[f"{c1}_vs_{c2}"] = round(cos_val, 4)
        
        layer_result["cat_diff_cos"] = cat_diff_cos
        
        # 3. 中文vehicle残差方向
        cat_centers_zh = {}
        for cat, objs_zh in CATEGORIES_ZH.items():
            if cat not in extended_cats:
                continue
            vecs = []
            for zh_name in objs_zh[:len(extended_cats.get(cat, []))]:
                prompt = TEMPLATES_ZH["is_a"].format(obj=zh_name)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
                if resid is not None:
                    vecs.append(resid)
            if vecs:
                cat_centers_zh[cat] = np.mean(vecs, axis=0)
        
        # 中英文vehicle方向比较
        if "vehicle" in cat_centers_en and "vehicle" in cat_centers_zh:
            en_veh_center = cat_centers_en["vehicle"]
            zh_veh_center = cat_centers_zh["vehicle"]
            
            # 中英文vehicle方向
            en_ref = cat_centers_en.get("fruit", np.zeros_like(en_veh_center))
            zh_ref = cat_centers_zh.get("fruit", np.zeros_like(zh_veh_center))
            
            en_veh_diff = en_veh_center - en_ref
            zh_veh_diff = zh_veh_center - zh_ref
            
            n1, n2 = np.linalg.norm(en_veh_diff), np.linalg.norm(zh_veh_diff)
            if n1 > 1e-10 and n2 > 1e-10:
                cross_lang_veh_cos = float(np.dot(en_veh_diff/n1, zh_veh_diff/n2))
                layer_result["cross_lang_vehicle_cos"] = round(cross_lang_veh_cos, 4)
        
        # 4. 各类别内部一致性(类内方差vs类间距离)
        for cat in ["vehicle", "animal", "fruit", "tool"]:
            if cat in cat_vecs_en and cat in cat_centers_en and len(cat_vecs_en[cat]) >= 2:
                vecs = cat_vecs_en[cat]
                center = cat_centers_en[cat]
                intra_dists = [np.linalg.norm(v - center) for v in vecs]
                layer_result[f"{cat}_intra_dist_mean"] = round(float(np.mean(intra_dists)), 4)
                layer_result[f"{cat}_intra_dist_std"] = round(float(np.std(intra_dists)), 4)
        
        # 5. vehicle内部各对象的方向
        if "vehicle" in cat_vecs_en and "fruit" in cat_centers_en:
            veh_vecs = cat_vecs_en["vehicle"]
            fruit_ref = cat_centers_en["fruit"]
            veh_obj_dirs = {}
            for i, v in enumerate(veh_vecs):
                diff = v - fruit_ref
                n = np.linalg.norm(diff)
                if n > 1e-10:
                    # 计算这个方向与vehicle候选词读出方向的cos
                    dir_normed = diff / n
                    # 只与"vehicle"这个词的W_U方向比较
                    vocab = tokenizer.get_vocab()
                    if "vehicle" in vocab:
                        w_dir = W_U[vocab["vehicle"]].copy()
                        w_norm = np.linalg.norm(w_dir)
                        if w_norm > 1e-10:
                            cos_val = float(np.dot(dir_normed, w_dir / w_norm))
                            obj_name = extended_cats["vehicle"][i] if i < len(extended_cats["vehicle"]) else f"obj_{i}"
                            veh_obj_dirs[obj_name] = round(cos_val, 4)
            
            layer_result["vehicle_obj_readout_cos"] = veh_obj_dirs
        
        results[f"L{patch_li}"] = layer_result
    
    return results


# ==================== Exp4: 多词元中文候选族读出 ====================
def exp4_multi_token_zh_readout(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    完整修复中文候选族读出
    
    方法:
    1. 使用首token的log_softmax(而非原始logit)
    2. family-local softmax归一化
    3. 与英文候选族边际对比
    4. patch后的中英文候选族变化
    """
    plog("=== Exp4: 多词元中文候选族读出 ===")
    info = get_model_info(model, model_name)
    
    patch_li = info.n_layers // 2  # 中层
    betas = [5.0, 10.0]
    
    test_cats = ["fruit", "animal", "vehicle"]
    results = {}
    
    for cat in test_cats:
        plog(f"  Category: {cat}")
        objs = obj_dict.get(cat, [])
        if len(objs) < 3:
            continue
        
        # 构造类别差分方向
        other_cat = "animal" if cat == "fruit" else "fruit"
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
        
        diff = np.mean(cat_vecs, axis=0) - np.mean(other_vecs, axis=0)
        diff_norm = np.linalg.norm(diff)
        if diff_norm < 1e-10:
            continue
        diff_dir = diff / diff_norm
        
        # 测试对象
        test_obj = objs[0]
        prompt = TEMPLATES_EN["is_a"].format(obj=test_obj)
        
        compete_cats = ["animal", "tool", "vehicle"] if cat == "fruit" else \
                       ["fruit", "tool", "vehicle"] if cat == "animal" else \
                       ["fruit", "animal", "tool"]
        
        cat_result = {}
        
        for beta in betas:
            inject_vec = beta * diff_dir
            
            # 自然logits
            logits_base = get_final_logits(model, tokenizer, prompt, device)
            # patch后logits
            logits_patch = run_with_additive_patch(model, tokenizer, prompt, device, patch_li, inject_vec)
            
            # 英文候选族边际(旧方法)
            en_margin_base, en_t_base, en_c_base = compute_en_family_margin(logits_base, tokenizer, cat, compete_cats)
            en_margin_patch, en_t_patch, en_c_patch = compute_en_family_margin(logits_patch, tokenizer, cat, compete_cats)
            
            # 中文候选族边际(旧方法: 单token ID索引)
            zh_margin_base_old, zh_t_base, zh_c_base = compute_zh_family_margin_v2(logits_base, tokenizer, cat, compete_cats)
            zh_margin_patch_old, zh_t_patch, zh_c_patch = compute_zh_family_margin_v2(logits_patch, tokenizer, cat, compete_cats)
            
            # 中文候选族边际(新方法: log_softmax + family-local归一化)
            zh_margin_base_new, _, _ = compute_zh_family_margin_multi_token(
                model, tokenizer, prompt, device, cat, compete_cats)
            
            # 计算patch后的中文边际(新方法)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = attention_mask.sum().item()
            
            layers = get_layers(model)
            delta_tensor = torch.tensor(inject_vec, dtype=torch.float32, device=device)
            patched_flag = [False]
            def make_hook():
                def hook(module, input, output):
                    if not patched_flag[0]:
                        patched_flag[0] = True
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
                out_patch = model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            logits_patch_for_zh = out_patch.logits[0, -1].float().cpu().numpy()
            
            zh_target_word = ZH_CLASS_WORDS.get(cat, "")
            zh_compete_words = [ZH_CLASS_WORDS.get(c, "") for c in compete_cats if c in ZH_CLASS_WORDS]
            
            # 新方法: log_softmax
            def log_softmax_vec(logits_vec):
                max_l = np.max(logits_vec)
                return logits_vec - max_l - np.log(np.sum(np.exp(logits_vec - max_l)))
            
            log_probs_base = log_softmax_vec(logits_base)
            log_probs_patch = log_softmax_vec(logits_patch_for_zh)
            
            # 首token log概率
            zh_target_ids = tokenizer.encode(zh_target_word, add_special_tokens=False) if zh_target_word else []
            zh_target_logprob_base = float(log_probs_base[zh_target_ids[0]]) if zh_target_ids else 0
            zh_target_logprob_patch = float(log_probs_patch[zh_target_ids[0]]) if zh_target_ids else 0
            
            zh_compete_logprobs_base = []
            zh_compete_logprobs_patch = []
            for zw in zh_compete_words:
                ids = tokenizer.encode(zw, add_special_tokens=False)
                if ids:
                    zh_compete_logprobs_base.append(float(log_probs_base[ids[0]]))
                    zh_compete_logprobs_patch.append(float(log_probs_patch[ids[0]]))
            
            # family-local softmax margin (log ratio)
            if zh_target_ids and zh_compete_logprobs_base:
                zh_margin_base_new = float(np.mean([zh_target_logprob_base]) - np.mean(zh_compete_logprobs_base))
                zh_margin_patch_new = float(np.mean([zh_target_logprob_patch]) - np.mean(zh_compete_logprobs_patch))
            else:
                zh_margin_base_new = 0
                zh_margin_patch_new = 0
            
            cat_result[f"beta_{beta}"] = {
                # 英文
                "en_margin_base": round(en_margin_base, 4),
                "en_margin_patch": round(en_margin_patch, 4),
                "en_selectivity": round(en_margin_patch - en_margin_base, 4),
                # 中文(旧方法: raw logit)
                "zh_margin_base_old": round(zh_margin_base_old, 4),
                "zh_margin_patch_old": round(zh_margin_patch_old, 4),
                "zh_selectivity_old": round(zh_margin_patch_old - zh_margin_base_old, 4),
                # 中文(新方法: log_softmax)
                "zh_margin_base_new": round(zh_margin_base_new, 4),
                "zh_margin_patch_new": round(zh_margin_patch_new, 4),
                "zh_selectivity_new": round(zh_margin_patch_new - zh_margin_base_new, 4),
                # 首token概率详情
                "zh_target_logprob_base": round(zh_target_logprob_base, 4),
                "zh_target_logprob_patch": round(zh_target_logprob_patch, 4),
                "zh_target_word": zh_target_word,
                "zh_target_ids": zh_target_ids,
            }
            
            plog(f"    beta={beta}: EN_sel={en_margin_patch - en_margin_base:.3f}, "
                 f"ZH_sel_old={zh_margin_patch_old - zh_margin_base_old:.3f}, "
                 f"ZH_sel_new={zh_margin_patch_new - zh_margin_base_new:.3f}")
        
        results[cat] = cat_result
    
    return results


# ==================== Exp5: 残差可写性大样本验证 ====================
def exp5_large_sample_writability(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    大样本残差可写性验证
    
    R1: 6类×4对象, 前2训练后2测试
    R2: 6类×8对象, 前4训练后4测试
    
    同时测试中英文双语候选族边际
    """
    plog("=== Exp5: 残差可写性大样本验证 ===")
    info = get_model_info(model, model_name)
    
    key_layers = [
        info.n_layers // 3,
        info.n_layers // 2,
        2 * info.n_layers // 3,
    ]
    key_layers = sorted(set([l for l in key_layers if l < info.n_layers]))
    
    betas = [5.0, 10.0]
    test_cats = list(obj_dict.keys())
    
    results = {}
    
    for cat in test_cats:
        objs = obj_dict[cat]
        if len(objs) < 3:
            continue
        
        split = max(2, len(objs) // 2)
        train_objs = objs[:split]
        test_objs = objs[split:]
        
        if not test_objs:
            continue
        
        compete_cats = [c for c in test_cats if c != cat][:3]
        cat_results = {}
        
        for patch_li in key_layers:
            plog(f"  {cat} L{patch_li}: train={len(train_objs)}, test={len(test_objs)}")
            
            # 构造类别差分方向(用训练对象)
            other_cat = compete_cats[0] if compete_cats else "fruit"
            other_objs = obj_dict.get(other_cat, [])[:split]
            
            train_vecs, other_vecs = [], []
            for obj in train_objs:
                prompt = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
                if resid is not None:
                    train_vecs.append(resid)
            for obj in other_objs:
                prompt = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
                if resid is not None:
                    other_vecs.append(resid)
            
            if len(train_vecs) < 2 or len(other_vecs) < 2:
                continue
            
            diff = np.mean(train_vecs, axis=0) - np.mean(other_vecs, axis=0)
            diff_norm = np.linalg.norm(diff)
            if diff_norm < 1e-10:
                continue
            diff_dir = diff / diff_norm
            
            # Holdout测试
            sel_list = []
            for test_obj in test_objs:
                prompt = TEMPLATES_EN["is_a"].format(obj=test_obj)
                
                for beta in betas:
                    inject_vec = beta * diff_dir
                    logits_base = get_final_logits(model, tokenizer, prompt, device)
                    logits_patch = run_with_additive_patch(model, tokenizer, prompt, device, patch_li, inject_vec)
                    
                    sel = compute_selectivity(logits_base, logits_patch, tokenizer, cat, compete_cats)
                    
                    sel_list.append({
                        "obj": test_obj,
                        "beta": beta,
                        "selectivity": round(sel, 4),
                    })
            
            if sel_list:
                avg_sels = {}
                for beta in betas:
                    beta_sels = [s["selectivity"] for s in sel_list if s["beta"] == beta]
                    if beta_sels:
                        avg_sels[f"beta_{beta}"] = round(float(np.mean(beta_sels)), 4)
                
                cat_results[f"L{patch_li}"] = {
                    "n_train": len(train_objs),
                    "n_test": len(test_objs),
                    "avg_selectivity": avg_sels,
                    "per_object": sel_list,
                }
        
        if cat_results:
            results[cat] = cat_results
    
    return results


# ==================== 主函数 ====================
def main():
    if len(sys.argv) < 3:
        print("Usage: python phase465_manifold_axis_verification.py <model_name> <round_num>")
        print("  model_name: qwen3 | deepseek7b | glm4")
        print("  round_num: 1 (pilot) | 2 (confirm)")
        sys.exit(1)
    
    model_name = sys.argv[1]
    round_num = int(sys.argv[2])
    
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        sys.exit(1)
    
    obj_dict = ROUNDS.get(round_num, ROUNDS[1])
    
    plog(f"Phase 465: {model_name} round {round_num}")
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
    
    # ===== Exp1: 自然流形兼容性 =====
    try:
        all_results["exp1_manifold"] = exp1_manifold_compatibility(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"Exp1 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_manifold"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    plog("Exp1 done, GPU cleared")
    
    # ===== Exp2: DS7B一维轴真假 =====
    try:
        all_results["exp2_axis_verify"] = exp2_ds7b_axis_verification(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"Exp2 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_axis_verify"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    plog("Exp2 done, GPU cleared")
    
    # ===== Exp3: vehicle反向码解析 =====
    try:
        all_results["exp3_vehicle"] = exp3_vehicle_reverse_analysis(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"Exp3 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_vehicle"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    plog("Exp3 done, GPU cleared")
    
    # ===== Exp4: 多词元中文候选族读出 =====
    try:
        all_results["exp4_zh_readout"] = exp4_multi_token_zh_readout(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"Exp4 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp4_zh_readout"] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    plog("Exp4 done, GPU cleared")
    
    # ===== Exp5: 残差可写性大样本 =====
    try:
        all_results["exp5_writability"] = exp5_large_sample_writability(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"Exp5 failed: {e}")
        import traceback; traceback.print_exc()
        all_results["exp5_writability"] = {"error": str(e)}
    
    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    result_path = f"results/glm5/phase465_{model_name}_r{round_num}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    plog(f"Results saved to {result_path}")
    
    # 释放模型
    release_model(model)
    plog(f"Phase 465 {model_name} R{round_num} complete!")


if __name__ == "__main__":
    main()
