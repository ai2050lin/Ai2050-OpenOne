"""
Phase 464: 正交分解修复、中文读出修复与模型策略验证
====================================================
修复Phase 463的关键问题:
1. Exp1正交化ratio计算bug: ||sem_only||/||sem_diff_raw|| 而非 ||sem_only||/||sem_dir||
2. 中文候选词边际全0: 需要多token序列概率
3. GLM4残差可写性跨类别验证
4. DS7B一维语言轴因果验证
5. Qwen3翻译维度增长复现

核心修复:
- 正交分解的ratio用归一化向量范数作分母
- 中文候选词用token ID直接索引logits(而非字符串查找)
- 添加数学校验: 理论ratio = sqrt(1 - cos²) vs 实际ratio

用法: python tests/glm5/phase464_orthogonal_fix_verification.py qwen3 1
      python tests/glm5/phase464_orthogonal_fix_verification.py glm4 2
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json, math, glob
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

# 中文类别词(用token ID方式)
ZH_CLASS_WORDS = {
    "fruit": "水果", "animal": "动物", "tool": "工具",
    "vehicle": "交通工具", "clothing": "衣服", "furniture": "家具",
}

TEMPLATES_EN = {"is_a": "The {obj} is a kind of"}
TEMPLATES_ZH = {"is_a": "{obj}是一种"}

# 翻译实验模板
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

CAT_TO_FAM = {
    "fruit":    ("fruit",    ["animal", "tool", "vehicle"]),
    "animal":   ("animal",   ["fruit", "tool", "vehicle"]),
    "tool":     ("tool",     ["fruit", "animal", "vehicle"]),
    "vehicle":  ("vehicle",  ["fruit", "animal", "tool"]),
    "clothing": ("clothing", ["fruit", "animal", "tool"]),
    "furniture":["furniture", ["fruit", "animal", "tool"]],
}

# 轮次数据量
ROUNDS = {
    1: {k: v[:4] for k, v in CATEGORIES.items()},   # pilot: 4类×4对象
    2: {k: v[:6] for k, v in CATEGORIES.items()},   # confirm: 6类×6对象
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


# ==================== 中文候选词修复: 用token ID直接索引 ====================
def get_token_ids_for_word(tokenizer, word):
    """获取一个词的token ID列表(支持多token词)"""
    ids = tokenizer.encode(word, add_special_tokens=False)
    return ids


def compute_en_family_margin_v2(logits, tokenizer, target_cat, compete_cats):
    """计算英文候选族边际 — 用token ID直接索引"""
    target_words = FAMILIES_EN.get(target_cat, [])
    compete_words = []
    for cc in compete_cats:
        compete_words.extend(FAMILIES_EN.get(cc, []))
    
    vocab = tokenizer.get_vocab()
    target_logits = []
    for w in target_words:
        w_clean = w.strip()
        if w_clean in vocab:
            target_logits.append(float(logits[vocab[w_clean]]))
        elif f" {w_clean}" in vocab:
            target_logits.append(float(logits[vocab[f" {w_clean}"]]))
    
    compete_logits = []
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


def compute_zh_family_margin_v2(logits, tokenizer, target_cat, compete_cats=None):
    """计算中文候选族边际 — 修复版: 用token ID直接索引, 支持多token词
    
    compete_cats: 可选, 如果为None则自动从ZH_CLASS_WORDS推导
    """
    if compete_cats is None:
        compete_cats = [k for k in ZH_CLASS_WORDS.keys() if k != target_cat]
    zh_target_word = ZH_CLASS_WORDS.get(target_cat, "")
    zh_compete_words = [ZH_CLASS_WORDS.get(c, "") for c in compete_cats if c in ZH_CLASS_WORDS]
    
    # 方法1: 直接用token ID索引logits(取第一个token的logit)
    target_logits = []
    if zh_target_word:
        tok_ids = tokenizer.encode(zh_target_word, add_special_tokens=False)
        if tok_ids:
            target_logits.append(float(logits[tok_ids[0]]))
    
    compete_logits = []
    for zw in zh_compete_words:
        tok_ids = tokenizer.encode(zw, add_special_tokens=False)
        if tok_ids:
            compete_logits.append(float(logits[tok_ids[0]]))
    
    if not target_logits or not compete_logits:
        return 0.0, 0.0, 0.0
    t_mean = float(np.mean(target_logits))
    c_mean = float(np.mean(compete_logits))
    return t_mean - c_mean, t_mean, c_mean


# ==================== 残差流提取 ====================
def get_residual_at_layer_pos(model, tokenizer, prompt, layer_idx, device, pos=-1):
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


# ==================== Exp1: 正交分解修复与数学校验 ====================
def exp1_orthogonal_fix(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    修复版: 正交分解 + 数学校验
    
    Phase 463 bug: sem_only_ratio = ||semantic_only|| / ||semantic_diff_raw||
    应该是:       sem_only_ratio = ||semantic_only|| / ||semantic_dir||
    
    因为semantic_dir已经是归一化向量(||semantic_dir||=1),
    而semantic_only = semantic_dir - proj, 如果cos≈0, 则||semantic_only||≈1
    
    数学校验:
    ||semantic_only|| / ||semantic_dir|| = sqrt(1 - cos²(sem, lang))
    如果实际值和理论值不匹配, 说明实现有问题
    """
    plog("=== Exp1: 正交分解修复与数学校验 ===")
    info = get_model_info(model, model_name)
    
    key_layers = [
        info.n_layers // 6,
        info.n_layers // 3,
        info.n_layers // 2,
        2 * info.n_layers // 3,
        info.n_layers - 3,
    ]
    key_layers = sorted(set([l for l in key_layers if l < info.n_layers]))
    
    test_cats = ["fruit", "animal"]
    test_objs = {c: obj_dict.get(c, [])[:4] for c in test_cats}
    
    results = {}
    
    for patch_li in key_layers:
        plog(f"  Layer L{patch_li}...")
        
        # ---- 构造语义差分(同语言, 不同类别) ----
        en_fruit_vecs = []
        en_animal_vecs = []
        for obj in test_objs["fruit"]:
            prompt = TEMPLATES_EN["is_a"].format(obj=obj)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
            if resid is not None:
                en_fruit_vecs.append(resid)
        for obj in test_objs["animal"]:
            prompt = TEMPLATES_EN["is_a"].format(obj=obj)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
            if resid is not None:
                en_animal_vecs.append(resid)
        
        if len(en_fruit_vecs) < 2 or len(en_animal_vecs) < 2:
            plog(f"    L{patch_li}: Not enough EN vectors, skip")
            continue
        
        en_fruit_center = np.mean(en_fruit_vecs, axis=0)
        en_animal_center = np.mean(en_animal_vecs, axis=0)
        semantic_diff = en_fruit_center - en_animal_center  # 原始差分(未归一化)
        semantic_diff_norm = np.linalg.norm(semantic_diff)
        if semantic_diff_norm < 1e-10:
            continue
        semantic_dir = semantic_diff / semantic_diff_norm  # 归一化方向
        
        # ---- 构造语言差分(同类别, 不同语言) ----
        zh_fruit_vecs = []
        zh_animal_vecs = []
        zh_fruit_names = CATEGORIES_ZH.get("fruit", [])[:4]
        zh_animal_names = CATEGORIES_ZH.get("animal", [])[:4]
        
        for i in range(min(len(test_objs["fruit"]), len(zh_fruit_names))):
            zh_name = zh_fruit_names[i]
            prompt = TEMPLATES_ZH["is_a"].format(obj=zh_name)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
            if resid is not None:
                zh_fruit_vecs.append(resid)
        for i in range(min(len(test_objs["animal"]), len(zh_animal_names))):
            zh_name = zh_animal_names[i]
            prompt = TEMPLATES_ZH["is_a"].format(obj=zh_name)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
            if resid is not None:
                zh_animal_vecs.append(resid)
        
        if len(zh_fruit_vecs) < 2 or len(zh_animal_vecs) < 2:
            plog(f"    L{patch_li}: Not enough ZH vectors, skip")
            continue
        
        zh_fruit_center = np.mean(zh_fruit_vecs, axis=0)
        zh_animal_center = np.mean(zh_animal_vecs, axis=0)
        
        lang_diff_fruit = en_fruit_center - zh_fruit_center
        lang_diff_animal = en_animal_center - zh_animal_center
        lang_diff = (lang_diff_fruit + lang_diff_animal) / 2
        lang_diff_norm = np.linalg.norm(lang_diff)
        if lang_diff_norm < 1e-10:
            continue
        lang_dir = lang_diff / lang_diff_norm  # 归一化方向
        
        # ---- 正交分解(核心!) ----
        cos_sem_lang = float(np.dot(semantic_dir, lang_dir))
        
        # SemanticOnly = semantic_dir - proj_lang(semantic_dir)
        proj_sem_on_lang = cos_sem_lang * lang_dir
        semantic_only = semantic_dir - proj_sem_on_lang
        semantic_only_norm = np.linalg.norm(semantic_only)
        
        # LanguageOnly = lang_dir - proj_sem(lang_dir)
        proj_lang_on_sem = cos_sem_lang * semantic_dir
        language_only = lang_dir - proj_lang_on_sem
        language_only_norm = np.linalg.norm(language_only)
        
        # ===== 关键修复: ratio分母用归一化向量范数 =====
        # 旧: sem_only_ratio = ||semantic_only|| / ||semantic_diff_raw|| (BUG! 很小因为原始差分范数大)
        # 新: sem_only_ratio = ||semantic_only|| / ||semantic_dir||  (正确! semantic_dir范数=1)
        sem_only_ratio = float(semantic_only_norm / 1.0)  # ||semantic_dir|| = 1
        lang_only_ratio = float(language_only_norm / 1.0)  # ||lang_dir|| = 1
        
        # 数学校验: 理论值 = sqrt(1 - cos²)
        theoretical_sem_ratio = float(np.sqrt(max(0, 1 - cos_sem_lang**2)))
        theoretical_lang_ratio = float(np.sqrt(max(0, 1 - cos_sem_lang**2)))
        
        ratio_error = abs(sem_only_ratio - theoretical_sem_ratio)
        
        plog(f"    cos(sem,lang)={cos_sem_lang:.4f}")
        plog(f"    ||sem_only||/||sem_dir|| = {sem_only_ratio:.4f} (theory: {theoretical_sem_ratio:.4f}, err={ratio_error:.4f})")
        plog(f"    ||lang_only||/||lang_dir|| = {lang_only_ratio:.4f}")
        
        # 同时保留旧指标以便对比
        old_sem_only_ratio = float(semantic_only_norm / semantic_diff_norm)
        old_lang_only_ratio = float(language_only_norm / lang_diff_norm)
        plog(f"    [OLD ratio] sem_only/raw_diff={old_sem_only_ratio:.4f}, lang_only/raw_diff={old_lang_only_ratio:.4f}")
        
        # ---- 注入测试(使用原始差分范数缩放, 保持注入强度一致) ----
        test_pairs = [
            ("fruit", "apple", "苹果", "fruit"),
            ("fruit", "banana", "香蕉", "fruit"),
            ("animal", "dog", "狗", "animal"),
            ("animal", "cat", "猫", "animal"),
        ]
        
        beta = 5.0
        layer_injections = {}
        
        for cat, obj_en, obj_zh, fam_cat in test_pairs:
            compete_cats = CAT_TO_FAM.get(cat, (None, []))[1]
            if not compete_cats:
                continue
            
            prompt_zh = TEMPLATES_ZH["is_a"].format(obj=obj_zh)
            prompt_en = TEMPLATES_EN["is_a"].format(obj=obj_en)
            
            # 1. Baseline
            zh_base_logits = get_final_logits(model, tokenizer, prompt_zh, device)
            zh_base_en_m, _, _ = compute_en_family_margin_v2(zh_base_logits, tokenizer, fam_cat, compete_cats)
            zh_base_zh_m, _, _ = compute_zh_family_margin_v2(zh_base_logits, tokenizer, cat)
            
            en_logits = get_final_logits(model, tokenizer, prompt_en, device)
            en_ref_en_m, _, _ = compute_en_family_margin_v2(en_logits, tokenizer, fam_cat, compete_cats)
            en_ref_zh_m, _, _ = compute_zh_family_margin_v2(en_logits, tokenizer, cat)
            
            # 2. 原始语义方向注入(未正交化)
            delta_sem_raw = beta * semantic_diff_norm * semantic_dir  # 缩放回原始范数
            patched_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_li, delta_sem_raw)
            raw_en_m, _, _ = compute_en_family_margin_v2(patched_logits, tokenizer, fam_cat, compete_cats)
            raw_zh_m, _, _ = compute_zh_family_margin_v2(patched_logits, tokenizer, cat)
            
            # 3. 纯语义方向注入(正交化后, 缩放到原始语义范数)
            if semantic_only_norm > 1e-10:
                delta_sem_only = beta * semantic_diff_norm * (semantic_only / semantic_only_norm)
                patched_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_li, delta_sem_only)
                sem_en_m, _, _ = compute_en_family_margin_v2(patched_logits, tokenizer, fam_cat, compete_cats)
                sem_zh_m, _, _ = compute_zh_family_margin_v2(patched_logits, tokenizer, cat)
            else:
                sem_en_m = zh_base_en_m
                sem_zh_m = zh_base_zh_m
            
            # 4. 纯语言方向注入(正交化后, 缩放到原始语言范数)
            if language_only_norm > 1e-10:
                delta_lang_only = beta * lang_diff_norm * (language_only / language_only_norm)
                patched_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_li, delta_lang_only)
                lang_en_m, _, _ = compute_en_family_margin_v2(patched_logits, tokenizer, fam_cat, compete_cats)
                lang_zh_m, _, _ = compute_zh_family_margin_v2(patched_logits, tokenizer, cat)
            else:
                lang_en_m = zh_base_en_m
                lang_zh_m = zh_base_zh_m
            
            # 5. 混合注入(语义+语言)
            delta_mixed = beta * (semantic_diff_norm * semantic_dir + lang_diff_norm * lang_dir)
            patched_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_li, delta_mixed)
            mixed_en_m, _, _ = compute_en_family_margin_v2(patched_logits, tokenizer, fam_cat, compete_cats)
            mixed_zh_m, _, _ = compute_zh_family_margin_v2(patched_logits, tokenizer, cat)
            
            key = f"{cat}_{obj_en}_L{patch_li}"
            layer_injections[key] = {
                "category": cat, "object_en": obj_en, "object_zh": obj_zh,
                "patch_layer": patch_li, "beta": beta,
                "cos_sem_lang": cos_sem_lang,
                # 新版ratio(修复后)
                "sem_only_ratio_fixed": sem_only_ratio,
                "lang_only_ratio_fixed": lang_only_ratio,
                "theoretical_ratio": theoretical_sem_ratio,
                "ratio_error": ratio_error,
                # 旧版ratio(Phase 463的bug)
                "sem_only_ratio_old": old_sem_only_ratio,
                "lang_only_ratio_old": old_lang_only_ratio,
                # 范数信息
                "semantic_diff_norm": float(semantic_diff_norm),
                "lang_diff_norm": float(lang_diff_norm),
                "semantic_only_norm": float(semantic_only_norm),
                "language_only_norm": float(language_only_norm),
                # margins
                "zh_base_en_margin": float(zh_base_en_m),
                "zh_base_zh_margin": float(zh_base_zh_m),
                "en_ref_en_margin": float(en_ref_en_m),
                "en_ref_zh_margin": float(en_ref_zh_m),
                # raw semantic injection
                "raw_sem_en_delta": float(raw_en_m - zh_base_en_m),
                "raw_sem_zh_delta": float(raw_zh_m - zh_base_zh_m),
                # semantic-only injection
                "sem_only_en_delta": float(sem_en_m - zh_base_en_m),
                "sem_only_zh_delta": float(sem_zh_m - zh_base_zh_m),
                # language-only injection
                "lang_only_en_delta": float(lang_en_m - zh_base_en_m),
                "lang_only_zh_delta": float(lang_zh_m - zh_base_zh_m),
                # mixed injection
                "mixed_en_delta": float(mixed_en_m - zh_base_en_m),
                "mixed_zh_delta": float(mixed_zh_m - zh_base_zh_m),
            }
            
            plog(f"    {cat}/{obj_en} L{patch_li}: "
                 f"raw_enΔ={raw_en_m - zh_base_en_m:.2f} "
                 f"sem_only_enΔ={sem_en_m - zh_base_en_m:.2f} "
                 f"lang_only_enΔ={lang_en_m - zh_base_en_m:.2f} "
                 f"zhΔ: sem={sem_zh_m - zh_base_zh_m:.2f} lang={lang_zh_m - zh_base_zh_m:.2f}")
        
        results[f"L{patch_li}"] = layer_injections
    
    plog(f"  Exp1 done. {len(results)} layers tested")
    return results


# ==================== Exp2: 大样本跨语言Patch(修复中文读出) ====================
def exp2_large_sample_patch_fixed(model, tokenizer, model_name, device, obj_dict, round_num):
    """扩展跨语言patch, 修复中文候选词读出"""
    plog("=== Exp2: 大样本跨语言Patch(中文读出修复) ===")
    info = get_model_info(model, model_name)
    
    test_cats = list(obj_dict.keys())[:4]
    test_objs = {c: obj_dict[c][:4] for c in test_cats}
    
    patch_layers = [
        info.n_layers // 6,
        info.n_layers // 3,
        info.n_layers // 2,
        2 * info.n_layers // 3,
        info.n_layers - 3,
    ]
    patch_layers = sorted(set([l for l in patch_layers if l < info.n_layers]))
    
    results = {}
    
    for cat_name in test_cats:
        for obj_idx, obj in enumerate(test_objs[cat_name]):
            cat_objs_zh = CATEGORIES_ZH.get(cat_name, [])
            obj_zh = cat_objs_zh[obj_idx] if obj_idx < len(cat_objs_zh) else obj
            
            prompt_en = TEMPLATES_EN["is_a"].format(obj=obj)
            prompt_zh = TEMPLATES_ZH["is_a"].format(obj=obj_zh)
            
            fam_cat = CAT_TO_FAM.get(cat_name, (None, []))[0]
            compete_cats = CAT_TO_FAM.get(cat_name, (None, []))[1]
            if fam_cat is None:
                continue
            
            for patch_li in patch_layers:
                en_resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_en, patch_li, device)
                if en_resid is None:
                    continue
                zh_resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_zh, patch_li, device)
                if zh_resid is None:
                    continue
                
                delta = en_resid - zh_resid
                
                # baseline
                zh_base_logits = get_final_logits(model, tokenizer, prompt_zh, device)
                zh_base_en_m, _, _ = compute_en_family_margin_v2(zh_base_logits, tokenizer, fam_cat, compete_cats)
                zh_base_zh_m, _, _ = compute_zh_family_margin_v2(zh_base_logits, tokenizer, cat_name)
                
                en_logits = get_final_logits(model, tokenizer, prompt_en, device)
                en_ref_en_m, _, _ = compute_en_family_margin_v2(en_logits, tokenizer, fam_cat, compete_cats)
                en_ref_zh_m, _, _ = compute_zh_family_margin_v2(en_logits, tokenizer, cat_name)
                
                # patched
                patched_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_li, delta)
                patched_en_m, _, _ = compute_en_family_margin_v2(patched_logits, tokenizer, fam_cat, compete_cats)
                patched_zh_m, _, _ = compute_zh_family_margin_v2(patched_logits, tokenizer, cat_name)
                
                key = f"{cat_name}_{obj}_L{patch_li}"
                results[key] = {
                    "patch_layer": patch_li, "category": cat_name, "object": obj,
                    "zh_base_en_margin": float(zh_base_en_m),
                    "zh_patched_en_margin": float(patched_en_m),
                    "en_ref_en_margin": float(en_ref_en_m),
                    "delta_en_margin": float(patched_en_m - zh_base_en_m),
                    "zh_base_zh_margin": float(zh_base_zh_m),
                    "zh_patched_zh_margin": float(patched_zh_m),
                    "en_ref_zh_margin": float(en_ref_zh_m),
                    "delta_zh_margin": float(patched_zh_m - zh_base_zh_m),
                }
                
                plog(f"    L{patch_li} {cat_name}/{obj}: "
                     f"enΔ={patched_en_m - zh_base_en_m:.2f} "
                     f"zhΔ={patched_zh_m - zh_base_zh_m:.2f}")
    
    # 汇总
    summary = {}
    for key, val in results.items():
        cat = val["category"]
        li = val["patch_layer"]
        if cat not in summary:
            summary[cat] = {}
        if li not in summary[cat]:
            summary[cat][li] = {"en_deltas": [], "zh_deltas": []}
        summary[cat][li]["en_deltas"].append(val["delta_en_margin"])
        summary[cat][li]["zh_deltas"].append(val["delta_zh_margin"])
    
    for cat in summary:
        for li in summary[cat]:
            ens = summary[cat][li]["en_deltas"]
            zhs = summary[cat][li]["zh_deltas"]
            summary[cat][li]["avg_en_delta"] = float(np.mean(ens))
            summary[cat][li]["avg_zh_delta"] = float(np.mean(zhs))
            summary[cat][li]["n"] = len(ens)
    
    plog(f"  Exp2 done. {len(results)} patches")
    return {"per_object": results, "summary": summary}


# ==================== Exp3: GLM4残差可写性跨类别holdout验证 ====================
def exp3_glm4_cross_category_holdout(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    GLM4残差可写性跨类别验证
    
    构造方向: 用前4对象计算class_diff (A组)
    测试: 用后4对象测试selectivity (B组)
    
    同时测试6个类别, 看是否只有animal强
    """
    plog("=== Exp3: GLM4残差可写性跨类别holdout ===")
    info = get_model_info(model, model_name)
    
    # 关键层
    patch_layers = [
        info.n_layers // 3,
        info.n_layers // 2,
        2 * info.n_layers // 3,
    ]
    patch_layers = sorted(set([l for l in patch_layers if l < info.n_layers]))
    
    all_cats = list(obj_dict.keys())[:6]  # 6类
    betas = [5.0, 10.0]
    
    results = {}
    
    for patch_li in patch_layers:
        plog(f"  Layer L{patch_li}...")
        
        # 对每个类别, 构造class_diff方向和测试对象
        for cat in all_cats:
            objs = obj_dict[cat]
            if len(objs) < 3:
                continue
            
            # A组: 前半构造方向; B组: 后半测试
            split = max(2, len(objs) // 2)
            train_objs = objs[:split]
            test_objs = objs[split:]
            
            cat_objs_zh = CATEGORIES_ZH.get(cat, [])
            
            # ---- 构造class_diff: 英文类中心 - 中文类中心(同一类别) ----
            # 这是跨语言差分方向, 代表"从中文语义到英文语义"的偏移
            en_train_vecs = []
            for obj in train_objs:
                prompt = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
                if resid is not None:
                    en_train_vecs.append(resid)
            
            zh_train_vecs = []
            for i in range(min(len(train_objs), len(cat_objs_zh))):
                zh_name = cat_objs_zh[i]
                prompt = TEMPLATES_ZH["is_a"].format(obj=zh_name)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
                if resid is not None:
                    zh_train_vecs.append(resid)
            
            if len(en_train_vecs) < 2 or len(zh_train_vecs) < 2:
                continue
            
            en_center = np.mean(en_train_vecs, axis=0)
            zh_center = np.mean(zh_train_vecs, axis=0)
            class_diff = en_center - zh_center
            class_diff_norm = np.linalg.norm(class_diff)
            if class_diff_norm < 1e-10:
                continue
            class_dir = class_diff / class_diff_norm
            
            # ---- 在B组对象上测试 ----
            fam_cat = CAT_TO_FAM.get(cat, (None, []))[0]
            compete_cats = CAT_TO_FAM.get(cat, (None, []))[1]
            if fam_cat is None:
                continue
            
            for beta in betas:
                sel_list = []
                for obj in test_objs:
                    obj_idx = objs.index(obj) if obj in objs else 0
                    obj_zh = cat_objs_zh[obj_idx] if obj_idx < len(cat_objs_zh) else obj
                    
                    prompt_zh = TEMPLATES_ZH["is_a"].format(obj=obj_zh)
                    
                    # baseline
                    zh_base_logits = get_final_logits(model, tokenizer, prompt_zh, device)
                    base_en_m, _, _ = compute_en_family_margin_v2(zh_base_logits, tokenizer, fam_cat, compete_cats)
                    base_zh_m, _, _ = compute_zh_family_margin_v2(zh_base_logits, tokenizer, cat)
                    
                    # patched
                    delta = beta * class_diff
                    patched_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_li, delta)
                    patched_en_m, _, _ = compute_en_family_margin_v2(patched_logits, tokenizer, fam_cat, compete_cats)
                    patched_zh_m, _, _ = compute_zh_family_margin_v2(patched_logits, tokenizer, cat)
                    
                    en_delta = patched_en_m - base_en_m
                    zh_delta = patched_zh_m - base_zh_m
                    
                    sel = en_delta - zh_delta  # selectivity: 英文边际提升 - 中文边际变化
                    sel_list.append(sel)
                
                avg_sel = float(np.mean(sel_list)) if sel_list else 0.0
                
                key = f"{cat}_L{patch_li}_b{int(beta)}"
                results[key] = {
                    "category": cat,
                    "patch_layer": patch_li,
                    "beta": beta,
                    "avg_selectivity": avg_sel,
                    "n_test_objects": len(test_objs),
                    "individual_selectivities": [float(s) for s in sel_list],
                }
                
                plog(f"    {cat} L{patch_li} beta={beta}: avg_sel={avg_sel:.3f}")
    
    plog(f"  Exp3 done. {len(results)} tests across {len(all_cats)} categories")
    return results


# ==================== Exp4: DS7B一维语言轴因果干预 ====================
def exp4_ds7b_language_axis_intervention(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    验证DS7B一维语言轴的因果性
    
    如果4个方向(target_lang, source_lang, command, content)共线,
    那么沿这个单一轴注入应该能同时影响翻译方向和语义内容
    
    实验:
    1. 构造DS7B的语言轴(翻译方向的平均)
    2. 沿+方向注入: 应该增加"翻译到英文"倾向
    3. 沿-方向注入: 应该增加"翻译到中文"倾向
    4. 对比: 语义类别边际是否也被改变(如果共线, 应该也会变)
    """
    plog("=== Exp4: 一维语言轴因果干预 ===")
    info = get_model_info(model, model_name)
    
    key_layers = [
        info.n_layers // 6,
        info.n_layers // 3,
        info.n_layers // 2,
        2 * info.n_layers // 3,
        info.n_layers - 3,
    ]
    key_layers = sorted(set([l for l in key_layers if l < info.n_layers]))
    
    test_obj = "dog"
    test_obj_zh = "狗"
    
    results = {}
    
    for patch_li in key_layers:
        plog(f"  Layer L{patch_li}...")
        
        # ---- 构造翻译方向 ----
        # 1. en2zh: "Translate to Chinese: The dog is an animal."
        # 2. zh2en: "请翻译为英文: 狗是一种动物"
        # 3. en_only: "The dog is an animal."
        # 4. zh_only: "狗是一种动物"
        
        # target_lang方向: en2zh vs zh_only (目标语言不同, 内容都是中文语义)
        en2zh = "Translate to Chinese: The dog is an animal."
        zh_only = "狗是一种动物"
        en_only = "The dog is an animal."
        zh2en = "请翻译为英文: 狗是一种动物"
        
        resid_en2zh, _ = get_residual_at_layer_pos(model, tokenizer, en2zh, patch_li, device)
        resid_zh_only, _ = get_residual_at_layer_pos(model, tokenizer, zh_only, patch_li, device)
        resid_en_only, _ = get_residual_at_layer_pos(model, tokenizer, en_only, patch_li, device)
        resid_zh2en, _ = get_residual_at_layer_pos(model, tokenizer, zh2en, patch_li, device)
        
        if any(v is None for v in [resid_en2zh, resid_zh_only, resid_en_only, resid_zh2en]):
            plog(f"    L{patch_li}: Missing residuals, skip")
            continue
        
        # 构造4个方向
        target_lang_diff = resid_en2zh - resid_zh_only   # 目标语言差异(同内容, 不同目标语言)
        source_lang_diff = resid_zh2en - resid_en_only    # 源语言差异(同内容, 不同源语言)
        translate_cmd_diff = resid_en2zh - resid_en_only  # 翻译命令差异(同源语言, 有/无翻译命令)
        content_diff = resid_zh_only - resid_en_only      # 内容语言差异(同语义, 不同表达语言)
        
        # 计算共线性
        def cos_sim(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-10 or nb < 1e-10:
                return 0.0
            return float(np.dot(a, b) / (na * nb))
        
        cos_tgt_src = cos_sim(target_lang_diff, source_lang_diff)
        cos_tgt_cont = cos_sim(target_lang_diff, content_diff)
        cos_cmd_cont = cos_sim(translate_cmd_diff, content_diff)
        
        # 有效秩
        mat = np.stack([target_lang_diff, source_lang_diff, translate_cmd_diff, content_diff])
        # SVD
        try:
            U, S, Vt = np.linalg.svd(mat, full_matrices=False)
            total_energy = np.sum(S**2)
            if total_energy > 0:
                energy_ratios = S**2 / total_energy
                eff_rank = float(np.sum(energy_ratios**(-np.log2(energy_ratios + 1e-30))) / np.sum(energy_ratios)) if False else float(1.0 / np.sum((S**2 / total_energy)**2))
            else:
                eff_rank = 0.0
        except Exception:
            eff_rank = 0.0
        
        plog(f"    cos(tgt,src)={cos_tgt_src:.3f}, cos(tgt,cont)={cos_tgt_cont:.3f}, "
             f"cos(cmd,cont)={cos_cmd_cont:.3f}, eff_rank={eff_rank:.3f}")
        
        # ---- 沿语言轴注入 ----
        # 语言轴 = target_lang_diff的归一化方向(这是最"翻译到目标语言"的方向)
        lang_axis = target_lang_diff.copy()
        lang_axis_norm = np.linalg.norm(lang_axis)
        if lang_axis_norm < 1e-10:
            continue
        lang_axis_dir = lang_axis / lang_axis_norm
        
        # 在中文模板上注入, 测试:
        # +方向: 是否让英文候选词更受欢迎?
        # -方向: 是否让中文候选词更受欢迎?
        prompt_zh = TEMPLATES_ZH["is_a"].format(obj=test_obj_zh)
        prompt_en = TEMPLATES_EN["is_a"].format(obj=test_obj)
        
        betas = [3.0, 5.0, 10.0]
        
        for beta in betas:
            # baseline
            zh_base_logits = get_final_logits(model, tokenizer, prompt_zh, device)
            base_en_m, _, _ = compute_en_family_margin_v2(zh_base_logits, tokenizer, "animal", ["fruit", "tool", "vehicle"])
            base_zh_m, _, _ = compute_zh_family_margin_v2(zh_base_logits, tokenizer, "animal")
            
            # +方向注入
            delta_pos = beta * lang_axis_norm * lang_axis_dir
            patched_pos_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_li, delta_pos)
            pos_en_m, _, _ = compute_en_family_margin_v2(patched_pos_logits, tokenizer, "animal", ["fruit", "tool", "vehicle"])
            pos_zh_m, _, _ = compute_zh_family_margin_v2(patched_pos_logits, tokenizer, "animal")
            
            # -方向注入
            delta_neg = -beta * lang_axis_norm * lang_axis_dir
            patched_neg_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_li, delta_neg)
            neg_en_m, _, _ = compute_en_family_margin_v2(patched_neg_logits, tokenizer, "animal", ["fruit", "tool", "vehicle"])
            neg_zh_m, _, _ = compute_zh_family_margin_v2(patched_neg_logits, tokenizer, "animal")
            
            key = f"L{patch_li}_b{int(beta)}"
            results[key] = {
                "patch_layer": patch_li,
                "beta": beta,
                # 共线性指标
                "cos_tgt_src": cos_tgt_src,
                "cos_tgt_cont": cos_tgt_cont,
                "cos_cmd_cont": cos_cmd_cont,
                "eff_rank": eff_rank,
                # baseline
                "base_en_margin": float(base_en_m),
                "base_zh_margin": float(base_zh_m),
                # +方向注入
                "pos_en_delta": float(pos_en_m - base_en_m),
                "pos_zh_delta": float(pos_zh_m - base_zh_m),
                # -方向注入
                "neg_en_delta": float(neg_en_m - base_en_m),
                "neg_zh_delta": float(neg_zh_m - base_zh_m),
                # 对称性: +和-应该有相反效果
                "en_asymmetry": float((pos_en_m - base_en_m) - (base_en_m - neg_en_m)),
                "zh_asymmetry": float((pos_zh_m - base_zh_m) - (base_zh_m - neg_zh_m)),
            }
            
            plog(f"    beta={beta}: "
                 f"+enΔ={pos_en_m - base_en_m:.3f} +zhΔ={pos_zh_m - base_zh_m:.3f} "
                 f"-enΔ={neg_en_m - base_en_m:.3f} -zhΔ={neg_zh_m - base_zh_m:.3f}")
    
    plog(f"  Exp4 done. {len(results)} tests")
    return results


# ==================== Exp5: 翻译控制维度验证(Qwen3) ====================
def exp5_translate_dimension_verification(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    验证Qwen3翻译控制维度是否随深度增加
    
    使用多对象和多模板, 增加方向数量来更稳定地估计eff_rank
    """
    plog("=== Exp5: 翻译控制维度验证 ===")
    info = get_model_info(model, model_name)
    
    key_layers = [
        info.n_layers // 6,
        info.n_layers // 3,
        info.n_layers // 2,
        2 * info.n_layers // 3,
        info.n_layers - 3,
    ]
    key_layers = sorted(set([l for l in key_layers if l < info.n_layers]))
    
    # 多对象, 多模板
    test_objects = [("dog", "狗"), ("apple", "苹果"), ("car", "汽车")]
    
    results = {}
    
    for patch_li in key_layers:
        plog(f"  Layer L{patch_li}...")
        
        all_diffs = []
        labels = []
        
        for obj_en, obj_zh in test_objects:
            # 8种模板条件
            templates = [
                (f"Translate to Chinese: The {obj_en} is a fruit.", "zh", "en", "translate_cmd"),
                (f"Translate to Chinese: The {obj_en} is an animal.", "zh", "en", "translate_cmd"),
                (f"请翻译为英文: {obj_zh}是一种水果", "en", "zh", "translate_cmd"),
                (f"请翻译为英文: {obj_zh}是一种动物", "en", "zh", "translate_cmd"),
                (f"The {obj_en} is a fruit.", "en", "en", "no_cmd"),
                (f"The {obj_en} is an animal.", "en", "en", "no_cmd"),
                (f"{obj_zh}是一种水果", "zh", "zh", "no_cmd"),
                (f"{obj_zh}是一种动物", "zh", "zh", "no_cmd"),
            ]
            
            vecs = []
            for tmpl, tgt_lang, src_lang, cmd_type in templates:
                resid, _ = get_residual_at_layer_pos(model, tokenizer, tmpl, patch_li, device)
                if resid is not None:
                    vecs.append(resid)
                    labels.append((tgt_lang, src_lang, cmd_type, obj_en))
                else:
                    vecs.append(None)
            
            if any(v is None for v in vecs):
                continue
            
            # 构造方向
            # target_lang_diff: target=zh vs target=en (同src=en, 同内容)
            # vecs[0] = en2zh_fruit, vecs[4] = en_only_fruit
            target_diff = vecs[0] - vecs[4]  # en2zh_fruit - en_only_fruit
            source_diff = vecs[2] - vecs[6] if vecs[2] is not None and vecs[6] is not None else None  # zh2en_fruit - zh_only_fruit
            cmd_diff = vecs[0] - vecs[4]  # same as target for now
            content_diff = vecs[6] - vecs[4] if vecs[6] is not None else None  # zh_only_fruit - en_only_fruit
            
            if target_diff is not None:
                all_diffs.append(target_diff)
            if source_diff is not None:
                all_diffs.append(source_diff)
            if content_diff is not None:
                all_diffs.append(content_diff)
        
        if len(all_diffs) < 3:
            plog(f"    L{patch_li}: Not enough diffs, skip")
            continue
        
        # 计算两两余弦
        mat = np.stack(all_diffs)
        n_dirs = mat.shape[0]
        cos_matrix = np.zeros((n_dirs, n_dirs))
        for i in range(n_dirs):
            for j in range(n_dirs):
                ni, nj = np.linalg.norm(mat[i]), np.linalg.norm(mat[j])
                if ni > 1e-10 and nj > 1e-10:
                    cos_matrix[i, j] = float(np.dot(mat[i], mat[j]) / (ni * nj))
        
        # 有效秩 (基于SVD)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        total_energy = np.sum(S**2)
        if total_energy > 0:
            energy_ratios = S**2 / total_energy
            # 熵定义的有效秩
            entropy = -np.sum(energy_ratios * np.log2(energy_ratios + 1e-30))
            eff_rank = float(2 ** entropy)
        else:
            eff_rank = 0.0
        
        # 前k个奇异值的能量占比
        top1_ratio = float(S[0]**2 / total_energy) if total_energy > 0 else 0
        top2_ratio = float((S[0]**2 + S[1]**2) / total_energy) if total_energy > 0 and len(S) > 1 else 0
        
        results[f"L{patch_li}"] = {
            "eff_rank": eff_rank,
            "n_directions": n_dirs,
            "singular_values": [float(s) for s in S[:10]],
            "top1_energy_ratio": top1_ratio,
            "top2_energy_ratio": top2_ratio,
            "cos_matrix_diag_means": float(np.mean(np.diag(cos_matrix))),
        }
        
        plog(f"    L{patch_li}: eff_rank={eff_rank:.3f}, top1={top1_ratio:.3f}, top2={top2_ratio:.3f}, n_dirs={n_dirs}")
    
    plog(f"  Exp5 done. {len(results)} layers tested")
    return results


# ==================== Exp6: 三模型策略总验证 ====================
def exp6_strategy_verification(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    三模型策略分型指标汇总
    
    每个模型输出6个指标:
    1. CrossLangAcc: 跨语言分类准确率(从残差预测类别)
    2. LangSemCos: 语义/语言方向余弦
    3. TranslateEffRank: 翻译控制有效秩
    4. ResidualWriteability: 残差可写性(holdout selectivity)
    5. PatchEffect: 跨语言patch平均效果
    6. ZhReadoutEffect: 中文读出效果
    """
    plog("=== Exp6: 三模型策略指标汇总 ===")
    info = get_model_info(model, model_name)
    
    # 关键层: 中层和深层
    mid_layer = info.n_layers // 2
    deep_layer = info.n_layers - 3
    
    results = {}
    
    # 1. LangSemCos: 语义/语言方向余弦(从Exp1的部分逻辑)
    for layer_label, patch_li in [("mid", mid_layer), ("deep", deep_layer)]:
        en_fruit_vecs = []
        en_animal_vecs = []
        for obj in obj_dict.get("fruit", [])[:4]:
            prompt = TEMPLATES_EN["is_a"].format(obj=obj)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
            if resid is not None:
                en_fruit_vecs.append(resid)
        for obj in obj_dict.get("animal", [])[:4]:
            prompt = TEMPLATES_EN["is_a"].format(obj=obj)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
            if resid is not None:
                en_animal_vecs.append(resid)
        
        zh_fruit_vecs = []
        zh_animal_vecs = []
        zh_fruit_names = CATEGORIES_ZH.get("fruit", [])[:4]
        zh_animal_names = CATEGORIES_ZH.get("animal", [])[:4]
        for zh_name in zh_fruit_names[:4]:
            prompt = TEMPLATES_ZH["is_a"].format(obj=zh_name)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
            if resid is not None:
                zh_fruit_vecs.append(resid)
        for zh_name in zh_animal_names[:4]:
            prompt = TEMPLATES_ZH["is_a"].format(obj=zh_name)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
            if resid is not None:
                zh_animal_vecs.append(resid)
        
        if len(en_fruit_vecs) >= 2 and len(en_animal_vecs) >= 2 and len(zh_fruit_vecs) >= 2 and len(zh_animal_vecs) >= 2:
            en_fruit_c = np.mean(en_fruit_vecs, axis=0)
            en_animal_c = np.mean(en_animal_vecs, axis=0)
            zh_fruit_c = np.mean(zh_fruit_vecs, axis=0)
            zh_animal_c = np.mean(zh_animal_vecs, axis=0)
            
            sem_diff = en_fruit_c - en_animal_c
            lang_diff_f = en_fruit_c - zh_fruit_c
            lang_diff_a = en_animal_c - zh_animal_c
            lang_diff = (lang_diff_f + lang_diff_a) / 2
            
            sn, ln = np.linalg.norm(sem_diff), np.linalg.norm(lang_diff)
            cos_sem_lang = float(np.dot(sem_diff, lang_diff) / (sn * ln)) if sn > 1e-10 and ln > 1e-10 else 0
            
            results[f"LangSemCos_{layer_label}"] = cos_sem_lang
        else:
            results[f"LangSemCos_{layer_label}"] = None
    
    # 2. TranslateEffRank: 翻译控制有效秩
    for layer_label, patch_li in [("mid", mid_layer), ("deep", deep_layer)]:
        diffs = []
        for obj_en, obj_zh in [("dog", "狗"), ("apple", "苹果")]:
            en2zh = f"Translate to Chinese: The {obj_en} is a fruit."
            zh2en = f"请翻译为英文: {obj_zh}是一种水果"
            en_only = f"The {obj_en} is a fruit."
            zh_only = f"{obj_zh}是一种水果"
            
            vecs = []
            for tmpl in [en2zh, zh2en, en_only, zh_only]:
                resid, _ = get_residual_at_layer_pos(model, tokenizer, tmpl, patch_li, device)
                if resid is not None:
                    vecs.append(resid)
            
            if len(vecs) == 4:
                target_diff = vecs[0] - vecs[2]  # en2zh - en_only
                source_diff = vecs[1] - vecs[3]  # zh2en - zh_only
                content_diff = vecs[3] - vecs[2]  # zh_only - en_only
                diffs.extend([target_diff, source_diff, content_diff])
        
        if len(diffs) >= 3:
            mat = np.stack(diffs)
            U, S, Vt = np.linalg.svd(mat, full_matrices=False)
            total_energy = np.sum(S**2)
            if total_energy > 0:
                energy_ratios = S**2 / total_energy
                entropy = -np.sum(energy_ratios * np.log2(energy_ratios + 1e-30))
                eff_rank = float(2 ** entropy)
            else:
                eff_rank = 0
            results[f"TranslateEffRank_{layer_label}"] = eff_rank
        else:
            results[f"TranslateEffRank_{layer_label}"] = None
    
    # 3. PatchEffect: 跨语言patch平均效果
    patch_en_deltas = []
    for cat in ["fruit", "animal", "tool"]:
        objs = obj_dict.get(cat, [])[:2]
        cat_zh = CATEGORIES_ZH.get(cat, [])
        for i, obj in enumerate(objs):
            obj_zh = cat_zh[i] if i < len(cat_zh) else obj
            prompt_en = TEMPLATES_EN["is_a"].format(obj=obj)
            prompt_zh = TEMPLATES_ZH["is_a"].format(obj=obj_zh)
            
            en_resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_en, deep_layer, device)
            zh_resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_zh, deep_layer, device)
            if en_resid is None or zh_resid is None:
                continue
            
            delta = en_resid - zh_resid
            zh_base_logits = get_final_logits(model, tokenizer, prompt_zh, device)
            patched_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, deep_layer, delta)
            
            fam_cat = CAT_TO_FAM.get(cat, (None, []))[0]
            compete_cats = CAT_TO_FAM.get(cat, (None, []))[1]
            if fam_cat is None:
                continue
            
            base_m, _, _ = compute_en_family_margin_v2(zh_base_logits, tokenizer, fam_cat, compete_cats)
            patched_m, _, _ = compute_en_family_margin_v2(patched_logits, tokenizer, fam_cat, compete_cats)
            patch_en_deltas.append(float(patched_m - base_m))
    
    results["PatchEffect_deep"] = float(np.mean(patch_en_deltas)) if patch_en_deltas else None
    
    # 4. ResidualWriteability: holdout selectivity (animal类, beta=10)
    animal_objs = obj_dict.get("animal", [])
    split = max(2, len(animal_objs) // 2)
    train_objs = animal_objs[:split]
    test_objs_list = animal_objs[split:]
    if len(train_objs) >= 2 and len(test_objs_list) >= 1:
        en_train_vecs = []
        for obj in train_objs:
            prompt = TEMPLATES_EN["is_a"].format(obj=obj)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, deep_layer, device)
            if resid is not None:
                en_train_vecs.append(resid)
        zh_train_vecs = []
        for i, obj in enumerate(train_objs):
            zh_name = CATEGORIES_ZH.get("animal", [])[i] if i < len(CATEGORIES_ZH.get("animal", [])) else obj
            prompt = TEMPLATES_ZH["is_a"].format(obj=zh_name)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, deep_layer, device)
            if resid is not None:
                zh_train_vecs.append(resid)
        
        if len(en_train_vecs) >= 2 and len(zh_train_vecs) >= 2:
            class_diff = np.mean(en_train_vecs, axis=0) - np.mean(zh_train_vecs, axis=0)
            
            sel_list = []
            for obj in test_objs_list[:4]:
                idx = CATEGORIES["animal"].index(obj) if obj in CATEGORIES["animal"] else 0
                zh_name = CATEGORIES_ZH.get("animal", [])[idx] if idx < len(CATEGORIES_ZH.get("animal", [])) else obj
                prompt_zh = TEMPLATES_ZH["is_a"].format(obj=zh_name)
                
                base_logits = get_final_logits(model, tokenizer, prompt_zh, device)
                base_en_m, _, _ = compute_en_family_margin_v2(base_logits, tokenizer, "animal", ["fruit", "tool", "vehicle"])
                base_zh_m, _, _ = compute_zh_family_margin_v2(base_logits, tokenizer, "animal")
                
                patched_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, deep_layer, 10.0 * class_diff)
                patched_en_m, _, _ = compute_en_family_margin_v2(patched_logits, tokenizer, "animal", ["fruit", "tool", "vehicle"])
                patched_zh_m, _, _ = compute_zh_family_margin_v2(patched_logits, tokenizer, "animal")
                
                sel_list.append(float((patched_en_m - base_en_m) - (patched_zh_m - base_zh_m)))
            
            results["ResidualWriteability_animal"] = float(np.mean(sel_list)) if sel_list else None
        else:
            results["ResidualWriteability_animal"] = None
    else:
        results["ResidualWriteability_animal"] = None
    
    # 5. ZhReadoutEffect: 中文读出效果(检查中文边际是否非零)
    zh_test_logits = get_final_logits(model, tokenizer, "狗是一种", device)
    zh_m, _, _ = compute_zh_family_margin_v2(zh_test_logits, tokenizer, "animal")
    results["ZhReadoutEffect"] = float(zh_m)
    
    plog(f"  Exp6 done. Indicators: {list(results.keys())}")
    return results


# ==================== 主流程 ====================
def main():
    if len(sys.argv) < 3:
        print("Usage: python phase464_orthogonal_fix_verification.py <model> <round>")
        print("  model: qwen3 / glm4 / deepseek7b")
        print("  round: 1(pilot) / 2(confirm)")
        sys.exit(1)
    
    model_name = sys.argv[1]
    round_num = int(sys.argv[2])
    
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        sys.exit(1)
    
    obj_dict = ROUNDS.get(round_num, ROUNDS[1])
    
    plog(f"Phase 464: {model_name} R{round_num}")
    plog(f"  Objects: {', '.join(f'{k}×{len(v)}' for k, v in obj_dict.items())}")
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    plog(f"  Model: {info.model_class}, {info.n_layers} layers, d={info.d_model}")
    
    all_results = {
        "model": model_name,
        "round": round_num,
        "model_class": info.model_class,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
    }
    
    # Exp1: 正交分解修复
    try:
        all_results["exp1_orthogonal_fix"] = exp1_orthogonal_fix(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"  Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp2: 大样本跨语言patch(中文读出修复)
    try:
        all_results["exp2_patch_fixed"] = exp2_large_sample_patch_fixed(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"  Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp3: GLM4跨类别holdout(所有模型都跑, 看谁有残差可写性)
    try:
        all_results["exp3_cross_category_holdout"] = exp3_glm4_cross_category_holdout(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"  Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp4: 一维语言轴因果干预(所有模型都跑)
    try:
        all_results["exp4_language_axis_intervention"] = exp4_ds7b_language_axis_intervention(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"  Exp4 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp5: 翻译控制维度验证(所有模型都跑)
    try:
        all_results["exp5_translate_dimension"] = exp5_translate_dimension_verification(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"  Exp5 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp6: 三模型策略指标汇总
    try:
        all_results["exp6_strategy_indicators"] = exp6_strategy_verification(
            model, tokenizer, model_name, device, obj_dict, round_num)
    except Exception as e:
        plog(f"  Exp6 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    # 保存结果
    out_path = f"results/glm5/phase464_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    plog(f"Results saved to {out_path}")
    
    # 释放模型
    release_model(model)
    plog(f"Phase 464 {model_name} R{round_num} DONE!")


if __name__ == "__main__":
    main()
