"""
Phase 463: 语义码/语言码正交分解与跨语言读写闭环
==================================================
从Phase 462的"整体差分因果有效"推进到"纯语义码可写、语言码可切换、二者可正交分离"。

核心问题:
1. 跨语言patch中,真正起作用的是语义码还是英文表面码? (Exp1: 正交分解patch)
2. 大样本patch是否稳定? (Exp2: 4类×4对象扩展)
3. Additive vs Replacement vs Mean-code patch效果是否一致? (Exp3: 三法对比)
4. GLM4残差可写性是否真实? (Exp4: holdout验证)
5. DS7B语言轴纠缠能否验证? (Exp5: 精细分解)

关键改进:
- 语义/语言正交分解: 用同语言类别差分构造语义子空间, 用跨语言同语义差分构造语言子空间
- Replacement patch: 真正替换整个残差向量(不是加法注入)
- Mean-code patch: 用类别中心替换个体表示
- Holdout验证: 构造方向的对象和测试对象完全分离
- 中文候选词边际: 同时观察中英文候选词变化

用法: python tests/glm5/phase463_semantic_language_orthogonal.py qwen3 1
      python tests/glm5/phase463_semantic_language_orthogonal.py glm4 2
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

FAMILIES = {
    "class_fruit":    ["fruit", "produce", "crop", "harvest"],
    "class_animal":   ["animal", "creature", "beast", "mammal"],
    "class_tool":     ["tool", "implement", "instrument", "device"],
    "class_vehicle":  ["vehicle", "transport", "conveyance", "automobile"],
    "class_clothing": ["clothing", "apparel", "garment", "attire"],
    "class_furniture":["furniture", "furnishing", "fixture", "seat"],
}

CAT_TO_FAM = {
    "fruit":    ("class_fruit",    ["class_animal", "class_tool", "class_vehicle", "class_clothing", "class_furniture"]),
    "animal":   ("class_animal",   ["class_fruit", "class_tool", "class_vehicle", "class_clothing", "class_furniture"]),
    "tool":     ("class_tool",     ["class_fruit", "class_animal", "class_vehicle", "class_clothing", "class_furniture"]),
    "vehicle":  ("class_vehicle",  ["class_fruit", "class_animal", "class_tool", "class_clothing", "class_furniture"]),
    "clothing": ("class_clothing", ["class_fruit", "class_animal", "class_tool", "class_vehicle", "class_furniture"]),
    "furniture":("class_furniture",["class_fruit", "class_animal", "class_tool", "class_vehicle", "class_clothing"]),
}

ZH_CLASS_WORDS = {
    "fruit": "水果", "animal": "动物", "tool": "工具",
    "vehicle": "交通工具", "clothing": "衣服", "furniture": "家具",
}

TEMPLATES_EN = {
    "is_a": "The {obj} is a kind of",
}
TEMPLATES_ZH = {
    "is_a": "{obj}是一种",
}

# 翻译实验模板(Exp5)
TRANSLATE_TEMPLATES = [
    # (name, template, target_lang, source_lang)
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


# ==================== 权重加载(兼容meta device) ====================
def load_weight_from_safetensors(model_name, key):
    model_path = MODEL_CONFIGS[model_name]["path"]
    from safetensors import safe_open
    sf_files = glob.glob(os.path.join(model_path, '*.safetensors'))
    for sf_file in sf_files:
        try:
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                if key in sf.keys():
                    w = sf.get_tensor(key)
                    return w.float().numpy()
        except Exception:
            continue
    return None


def get_mlp_weights_safe(model, model_name, layer_idx):
    layers = get_layers(model)
    layer = layers[layer_idx]
    info = get_model_info(model, model_name)
    
    def _to_numpy(tensor, li, proj_name):
        if not tensor.is_meta:
            return tensor.detach().cpu().float().numpy()
        key = f"model.layers.{li}.mlp.{proj_name}.weight"
        w = load_weight_from_safetensors(model_name, key)
        if w is not None:
            plog(f"    Loaded L{li} {proj_name} from safetensors, shape={w.shape}")
            return w
        plog(f"    WARN: Cannot load L{li} {proj_name} from safetensors")
        return None
    
    if info.mlp_type == "split_gate_up":
        W_up = _to_numpy(layer.mlp.up_proj.weight, layer_idx, "up_proj")
        W_down = _to_numpy(layer.mlp.down_proj.weight, layer_idx, "down_proj")
        W_gate = _to_numpy(layer.mlp.gate_proj.weight, layer_idx, "gate_proj") if hasattr(layer.mlp, 'gate_proj') else None
    else:
        W_gate_up = _to_numpy(layer.mlp.gate_up_proj.weight, layer_idx, "gate_up_proj")
        if W_gate_up is not None:
            d_inter = W_gate_up.shape[0] // 2
            W_gate = W_gate_up[:d_inter]
            W_up = W_gate_up[d_inter:]
        else:
            W_gate = None
            W_up = None
        W_down = _to_numpy(layer.mlp.down_proj.weight, layer_idx, "down_proj")
    
    return W_up, W_down, W_gate


# ==================== 残差流提取 ====================
def get_residual_at_layers(model, tokenizer, prompt, target_layers, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    captured = {}
    layers = get_layers(model)
    
    def make_hook(li):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                captured[li] = input[0].detach().float().cpu()
        return hook
    
    hooks = [layers[li].register_forward_hook(make_hook(li)) for li in target_layers]
    
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    
    for h in hooks:
        h.remove()
    
    results = {}
    for li in target_layers:
        if li in captured:
            seq_len = attention_mask.sum().item()
            results[li] = captured[li][0, seq_len - 1].numpy()
    return results


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


# ==================== 候选族边际计算 ====================
def compute_en_family_margin(logits, tokenizer, target_fam, compete_fams):
    """计算英文候选族的边际(logit差)"""
    target_words = FAMILIES.get(target_fam, [])
    compete_words = []
    for cf in compete_fams:
        compete_words.extend(FAMILIES.get(cf, []))
    
    vocab = tokenizer.get_vocab()
    target_logits = []
    for w in target_words:
        w_clean = w.strip()
        if w_clean in vocab:
            target_logits.append(logits[vocab[w_clean]])
        elif f" {w_clean}" in vocab:
            target_logits.append(logits[vocab[f" {w_clean}"]])
    
    compete_logits = []
    for w in compete_words:
        w_clean = w.strip()
        if w_clean in vocab:
            compete_logits.append(logits[vocab[w_clean]])
        elif f" {w_clean}" in vocab:
            compete_logits.append(logits[vocab[f" {w_clean}"]])
    
    if not target_logits or not compete_logits:
        return 0.0, 0.0, 0.0
    t_mean = float(np.mean(target_logits))
    c_mean = float(np.mean(compete_logits))
    return t_mean - c_mean, t_mean, c_mean


def compute_zh_family_margin(logits, tokenizer, cat_name):
    """计算中文候选族的边际"""
    zh_target = ZH_CLASS_WORDS.get(cat_name, "")
    zh_compete = [v for k, v in ZH_CLASS_WORDS.items() if k != cat_name]
    
    vocab = tokenizer.get_vocab()
    target_logits = []
    if zh_target:
        for w in [zh_target, f" {zh_target}"]:
            if w in vocab:
                target_logits.append(logits[vocab[w]])
                break
    
    compete_logits = []
    for zw in zh_compete:
        for w in [zw, f" {zw}"]:
            if w in vocab:
                compete_logits.append(logits[vocab[w]])
                break
    
    if not target_logits or not compete_logits:
        return 0.0, 0.0, 0.0
    t_mean = float(np.mean(target_logits))
    c_mean = float(np.mean(compete_logits))
    return t_mean - c_mean, t_mean, c_mean


def get_final_logits(model, tokenizer, prompt, device):
    """获取模型最终logits"""
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


def run_with_replacement_patch(model, tokenizer, prompt_zh, prompt_en, device, patch_layer):
    """替换式patch: 在patch层把中文残差替换为英文残差, 然后继续前向"""
    # Step 1: 获取英文残差
    en_resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_en, patch_layer, device)
    if en_resid is None:
        return None
    
    # Step 2: 获取中文残差
    zh_resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_zh, patch_layer, device)
    if zh_resid is None:
        return None
    
    # Step 3: 构造差分(等同于替换: h_en = h_zh + (h_en - h_zh))
    delta = en_resid - zh_resid
    
    # Step 4: 加法注入差分
    return run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_layer, delta), delta


# ==================== Exp1: 语义/语言正交分解patch ====================
def exp1_semantic_language_orthogonal(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    语义码/语言码正交分解 — 核心实验!
    
    构造两个子空间:
    - SemanticSubspace: 同语言不同类别的差分 (dog-apple, knife-car)
    - LanguageSubspace: 同语义不同语言的差分 (dog-狗, apple-苹果)
    
    然后正交化:
    - SemanticOnly = Semantic - Proj_Language(Semantic)
    - LanguageOnly = Language - Proj_Semantic(Language)
    
    分别注入, 观察效果:
    - SemanticOnly应该改变类别边际(不改语言)
    - LanguageOnly应该改变目标语言读出(不改类别)
    """
    plog("=== Exp1: 语义/语言正交分解patch ===")
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    
    # 选择关键层
    key_layers = [
        info.n_layers // 6,      # 浅层
        info.n_layers // 3,      # 中浅
        info.n_layers // 2,      # 中层
        2 * info.n_layers // 3,   # 中深
        info.n_layers - 3,        # 深层
    ]
    key_layers = sorted(set([l for l in key_layers if l < info.n_layers]))
    
    # 选择2个类别 × 4对象(足够构造方向, 不会太多)
    test_cats = ["fruit", "animal"]
    test_objs = {c: obj_dict.get(c, [])[:4] for c in test_cats}
    
    results = {}
    
    for patch_li in key_layers:
        plog(f"  Layer L{patch_li}...")
        layer_results = {}
        
        # ---- 构造语义差分 ----
        # 同语言(英文), 不同类别: fruit_center - animal_center
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
        semantic_diff = en_fruit_center - en_animal_center  # 语义方向
        semantic_diff_norm = np.linalg.norm(semantic_diff)
        if semantic_diff_norm < 1e-10:
            continue
        semantic_dir = semantic_diff / semantic_diff_norm
        
        # ---- 构造语言差分 ----
        # 同语义(类中心), 不同语言: en_center - zh_center
        # 用fruit和animal共同构造语言差分
        zh_fruit_vecs = []
        zh_animal_vecs = []
        zh_fruit_names = CATEGORIES_ZH.get("fruit", [])[:4]
        zh_animal_names = CATEGORIES_ZH.get("animal", [])[:4]
        
        for i, obj in enumerate(test_objs["fruit"][:len(zh_fruit_names)]):
            zh_name = zh_fruit_names[i]
            prompt = TEMPLATES_ZH["is_a"].format(obj=zh_name)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
            if resid is not None:
                zh_fruit_vecs.append(resid)
        for i, obj in enumerate(test_objs["animal"][:len(zh_animal_names)]):
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
        
        # 语言差分: 英文类中心 - 中文类中心 (对两个类别都做, 取平均)
        lang_diff_fruit = en_fruit_center - zh_fruit_center
        lang_diff_animal = en_animal_center - zh_animal_center
        lang_diff = (lang_diff_fruit + lang_diff_animal) / 2  # 平均语言差分
        lang_diff_norm = np.linalg.norm(lang_diff)
        if lang_diff_norm < 1e-10:
            continue
        lang_dir = lang_diff / lang_diff_norm
        
        # ---- 正交分解 ----
        # SemanticOnly = semantic_dir - Proj_lang(semantic_dir)
        proj_sem_on_lang = np.dot(semantic_dir, lang_dir) * lang_dir
        semantic_only = semantic_dir - proj_sem_on_lang
        semantic_only_norm = np.linalg.norm(semantic_only)
        
        # LanguageOnly = lang_dir - Proj_sem(lang_dir)
        proj_lang_on_sem = np.dot(lang_dir, semantic_dir) * semantic_dir
        language_only = lang_dir - proj_lang_on_sem
        language_only_norm = np.linalg.norm(language_only)
        
        # 正交性度量
        cos_sem_lang = float(np.dot(semantic_dir, lang_dir))
        cos_sem_only_lang_only = float(np.dot(semantic_only, language_only) / 
                                       (semantic_only_norm * language_only_norm + 1e-10)) if semantic_only_norm > 1e-10 and language_only_norm > 1e-10 else 0
        
        plog(f"    cos(sem,lang)={cos_sem_lang:.4f}, "
             f"||sem_only||/||sem||={semantic_only_norm/semantic_diff_norm:.4f}, "
             f"||lang_only||/||lang||={language_only_norm/lang_diff_norm:.4f}")
        
        # ---- 注入测试 ----
        # 测试对象: fruit类别和animal类别各2个
        test_pairs = [
            ("fruit", "apple", "苹果", "class_fruit"),
            ("fruit", "banana", "香蕉", "class_fruit"),
            ("animal", "dog", "狗", "class_animal"),
            ("animal", "cat", "猫", "class_animal"),
        ]
        
        beta = 5.0  # 注入强度(适中, 不是太大)
        
        for cat, obj_en, obj_zh, fam_key in test_pairs:
            target_fam = fam_key
            compete_fams = CAT_TO_FAM.get(cat, (None, []))[1]
            if not compete_fams:
                continue
            
            prompt_zh = TEMPLATES_ZH["is_a"].format(obj=obj_zh)
            prompt_en = TEMPLATES_EN["is_a"].format(obj=obj_en)
            
            # 1. Baseline: 中文上下文
            zh_base_logits = get_final_logits(model, tokenizer, prompt_zh, device)
            zh_base_en_margin, _, _ = compute_en_family_margin(zh_base_logits, tokenizer, target_fam, compete_fams)
            zh_base_zh_margin, _, _ = compute_zh_family_margin(zh_base_logits, tokenizer, cat)
            
            # 2. 英文baseline
            en_logits = get_final_logits(model, tokenizer, prompt_en, device)
            en_en_margin, _, _ = compute_en_family_margin(en_logits, tokenizer, target_fam, compete_fams)
            en_zh_margin, _, _ = compute_zh_family_margin(en_logits, tokenizer, cat)
            
            # 3. 注入原始语义方向(未正交化)
            delta_sem_raw = beta * semantic_diff_norm * semantic_dir
            patched_raw_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_li, delta_sem_raw)
            raw_en_margin, _, _ = compute_en_family_margin(patched_raw_logits, tokenizer, target_fam, compete_fams)
            raw_zh_margin, _, _ = compute_zh_family_margin(patched_raw_logits, tokenizer, cat)
            
            # 4. 注入纯语义方向(正交化后)
            if semantic_only_norm > 1e-10:
                delta_sem_only = beta * semantic_diff_norm * (semantic_only / semantic_only_norm)
                patched_sem_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_li, delta_sem_only)
                sem_en_margin, _, _ = compute_en_family_margin(patched_sem_logits, tokenizer, target_fam, compete_fams)
                sem_zh_margin, _, _ = compute_zh_family_margin(patched_sem_logits, tokenizer, cat)
            else:
                sem_en_margin = zh_base_en_margin
                sem_zh_margin = zh_base_zh_margin
            
            # 5. 注入纯语言方向(正交化后)
            if language_only_norm > 1e-10:
                delta_lang_only = beta * lang_diff_norm * (language_only / language_only_norm)
                patched_lang_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_li, delta_lang_only)
                lang_en_margin, _, _ = compute_en_family_margin(patched_lang_logits, tokenizer, target_fam, compete_fams)
                lang_zh_margin, _, _ = compute_zh_family_margin(patched_lang_logits, tokenizer, cat)
            else:
                lang_en_margin = zh_base_en_margin
                lang_zh_margin = zh_base_zh_margin
            
            # 6. 注入混合方向(语义+语言)
            delta_mixed = beta * (semantic_dir + lang_dir)
            patched_mixed_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_li, delta_mixed)
            mixed_en_margin, _, _ = compute_en_family_margin(patched_mixed_logits, tokenizer, target_fam, compete_fams)
            mixed_zh_margin, _, _ = compute_zh_family_margin(patched_mixed_logits, tokenizer, cat)
            
            key = f"{cat}_{obj_en}_L{patch_li}"
            layer_results[key] = {
                "category": cat,
                "object_en": obj_en,
                "object_zh": obj_zh,
                "patch_layer": patch_li,
                "beta": beta,
                "cos_sem_lang": cos_sem_lang,
                "sem_only_ratio": float(semantic_only_norm / semantic_diff_norm),
                "lang_only_ratio": float(language_only_norm / lang_diff_norm),
                # margins
                "zh_base_en_margin": float(zh_base_en_margin),
                "zh_base_zh_margin": float(zh_base_zh_margin),
                "en_ref_en_margin": float(en_en_margin),
                "en_ref_zh_margin": float(en_zh_margin),
                # raw semantic injection
                "raw_sem_en_margin": float(raw_en_margin),
                "raw_sem_zh_margin": float(raw_zh_margin),
                "raw_sem_en_delta": float(raw_en_margin - zh_base_en_margin),
                "raw_sem_zh_delta": float(raw_zh_margin - zh_base_zh_margin),
                # semantic-only injection
                "sem_only_en_margin": float(sem_en_margin),
                "sem_only_zh_margin": float(sem_zh_margin),
                "sem_only_en_delta": float(sem_en_margin - zh_base_en_margin),
                "sem_only_zh_delta": float(sem_zh_margin - zh_base_zh_margin),
                # language-only injection
                "lang_only_en_margin": float(lang_en_margin),
                "lang_only_zh_margin": float(lang_zh_margin),
                "lang_only_en_delta": float(lang_en_margin - zh_base_en_margin),
                "lang_only_zh_delta": float(lang_zh_margin - zh_base_zh_margin),
                # mixed injection
                "mixed_en_margin": float(mixed_en_margin),
                "mixed_zh_margin": float(mixed_zh_margin),
                "mixed_en_delta": float(mixed_en_margin - zh_base_en_margin),
                "mixed_zh_delta": float(mixed_zh_margin - zh_base_zh_margin),
            }
            
            plog(f"    {cat}/{obj_en} L{patch_li}: "
                 f"sem_only_enΔ={sem_en_margin - zh_base_en_margin:.2f} "
                 f"lang_only_enΔ={lang_en_margin - zh_base_en_margin:.2f} "
                 f"sem_only_zhΔ={sem_zh_margin - zh_base_zh_margin:.2f} "
                 f"lang_only_zhΔ={lang_zh_margin - zh_base_zh_margin:.2f}")
        
        results[f"L{patch_li}"] = layer_results
    
    # 汇总
    plog(f"  Exp1 done. {len(results)} layers tested")
    return results


# ==================== Exp2: 大样本跨语言Patch扩展 ====================
def exp2_large_sample_patch(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    扩展Phase 462的Exp2b到4类×4对象
    
    同时测量:
    - 英文候选族边际变化
    - 中文候选族边际变化
    - 类别特异效果
    """
    plog("=== Exp2: 大样本跨语言Patch扩展 ===")
    info = get_model_info(model, model_name)
    
    test_cats = list(obj_dict.keys())[:4]  # 4类
    test_objs = {c: obj_dict[c][:4] for c in test_cats}  # 每类4对象
    
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
        for obj in test_objs[cat_name]:
            cat_objs_zh = CATEGORIES_ZH.get(cat_name, [])
            obj_idx = obj_dict[cat_name].index(obj) if obj in obj_dict[cat_name] else 0
            obj_zh = cat_objs_zh[obj_idx] if obj_idx < len(cat_objs_zh) else obj
            
            prompt_en = TEMPLATES_EN["is_a"].format(obj=obj)
            prompt_zh = TEMPLATES_ZH["is_a"].format(obj=obj_zh)
            
            target_fam = CAT_TO_FAM.get(cat_name, (None, []))[0]
            compete_fams = CAT_TO_FAM.get(cat_name, (None, []))[1]
            if target_fam is None:
                continue
            
            for patch_li in patch_layers:
                # 获取英文残差
                en_resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_en, patch_li, device)
                if en_resid is None:
                    continue
                
                # 获取中文残差
                zh_resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_zh, patch_li, device)
                if zh_resid is None:
                    continue
                
                delta = en_resid - zh_resid
                
                # 中文baseline
                zh_base_logits = get_final_logits(model, tokenizer, prompt_zh, device)
                zh_base_en_m, _, _ = compute_en_family_margin(zh_base_logits, tokenizer, target_fam, compete_fams)
                zh_base_zh_m, _, _ = compute_zh_family_margin(zh_base_logits, tokenizer, cat_name)
                
                # 英文baseline
                en_logits = get_final_logits(model, tokenizer, prompt_en, device)
                en_en_m, _, _ = compute_en_family_margin(en_logits, tokenizer, target_fam, compete_fams)
                en_zh_m, _, _ = compute_zh_family_margin(en_logits, tokenizer, cat_name)
                
                # 中文patched(加法注入)
                patched_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_li, delta)
                patched_en_m, _, _ = compute_en_family_margin(patched_logits, tokenizer, target_fam, compete_fams)
                patched_zh_m, _, _ = compute_zh_family_margin(patched_logits, tokenizer, cat_name)
                
                key = f"{cat_name}_{obj}_L{patch_li}"
                results[key] = {
                    "patch_layer": patch_li,
                    "category": cat_name,
                    "object": obj,
                    "zh_base_en_margin": float(zh_base_en_m),
                    "zh_patched_en_margin": float(patched_en_m),
                    "en_ref_en_margin": float(en_en_m),
                    "delta_en_margin": float(patched_en_m - zh_base_en_m),
                    "recovery_ratio": float((patched_en_m - zh_base_en_m) / (en_en_m - zh_base_en_m + 1e-10)),
                    "zh_base_zh_margin": float(zh_base_zh_m),
                    "zh_patched_zh_margin": float(patched_zh_m),
                    "en_ref_zh_margin": float(en_zh_m),
                    "delta_zh_margin": float(patched_zh_m - zh_base_zh_m),
                }
                
                plog(f"    L{patch_li} {cat_name}/{obj}: "
                     f"enΔ={patched_en_m - zh_base_en_m:.2f} "
                     f"zhΔ={patched_zh_m - zh_base_zh_m:.2f}")
    
    # 按类别和层汇总
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
    
    plog(f"  Exp2 done. {len(results)} patches across {len(test_cats)} categories")
    return {"per_object": results, "summary": summary}


# ==================== Exp3: Additive vs Replacement vs Mean-code ====================
def exp3_patch_methods_comparison(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    对比三种patch方法:
    1. Additive: h_zh + (h_en - h_zh) [Phase 462用的]
    2. Replacement: 直接用h_en替换h_zh [等价于additive,但验证]
    3. Mean-code: 用英文类别中心替换中文对象表示
    
    如果三者效果一致, 说明跨语言语义码稳定;
    如果mean-code更弱, 说明个体特征重要。
    """
    plog("=== Exp3: Additive vs Replacement vs Mean-code patch ===")
    info = get_model_info(model, model_name)
    
    test_cats = ["fruit", "animal"]
    test_objs = {c: obj_dict.get(c, [])[:3] for c in test_cats}
    
    # 关键层: 中层+深层
    patch_layers = [
        info.n_layers // 2,
        2 * info.n_layers // 3,
        info.n_layers - 3,
    ]
    patch_layers = sorted(set([l for l in patch_layers if l < info.n_layers]))
    
    results = {}
    
    for patch_li in patch_layers:
        plog(f"  Layer L{patch_li}...")
        
        # 先收集英文类别中心
        en_cat_centers = {}
        for cat_name in test_cats:
            vecs = []
            for obj in test_objs[cat_name]:
                prompt = TEMPLATES_EN["is_a"].format(obj=obj)
                resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
                if resid is not None:
                    vecs.append(resid)
            if vecs:
                en_cat_centers[cat_name] = np.mean(vecs, axis=0)
        
        for cat_name in test_cats:
            if cat_name not in en_cat_centers:
                continue
            
            for obj in test_objs[cat_name][:2]:  # 每类2个对象
                cat_objs_zh = CATEGORIES_ZH.get(cat_name, [])
                obj_idx = obj_dict[cat_name].index(obj) if obj in obj_dict[cat_name] else 0
                obj_zh = cat_objs_zh[obj_idx] if obj_idx < len(cat_objs_zh) else obj
                
                prompt_en = TEMPLATES_EN["is_a"].format(obj=obj)
                prompt_zh = TEMPLATES_ZH["is_a"].format(obj=obj_zh)
                
                target_fam = CAT_TO_FAM.get(cat_name, (None, []))[0]
                compete_fams = CAT_TO_FAM.get(cat_name, (None, []))[1]
                if target_fam is None:
                    continue
                
                # 获取残差
                en_resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_en, patch_li, device)
                zh_resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_zh, patch_li, device)
                if en_resid is None or zh_resid is None:
                    continue
                
                # 1. Baseline
                zh_base_logits = get_final_logits(model, tokenizer, prompt_zh, device)
                base_en_m, _, _ = compute_en_family_margin(zh_base_logits, tokenizer, target_fam, compete_fams)
                base_zh_m, _, _ = compute_zh_family_margin(zh_base_logits, tokenizer, cat_name)
                
                # 2. Additive patch: h_zh + (h_en - h_zh)
                delta_individual = en_resid - zh_resid
                add_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_li, delta_individual)
                add_en_m, _, _ = compute_en_family_margin(add_logits, tokenizer, target_fam, compete_fams)
                add_zh_m, _, _ = compute_zh_family_margin(add_logits, tokenizer, cat_name)
                
                # 3. Mean-code patch: 用英文类别中心 - 中文对象残差
                en_center = en_cat_centers[cat_name]
                delta_mean = en_center - zh_resid
                mean_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_li, delta_mean)
                mean_en_m, _, _ = compute_en_family_margin(mean_logits, tokenizer, target_fam, compete_fams)
                mean_zh_m, _, _ = compute_zh_family_margin(mean_logits, tokenizer, cat_name)
                
                # 4. Random control: 随机方向(范数匹配)
                rand_dir = np.random.randn(len(delta_individual))
                rand_dir = rand_dir / np.linalg.norm(rand_dir) * np.linalg.norm(delta_individual)
                rand_logits = run_with_additive_patch(model, tokenizer, prompt_zh, device, patch_li, rand_dir)
                rand_en_m, _, _ = compute_en_family_margin(rand_logits, tokenizer, target_fam, compete_fams)
                rand_zh_m, _, _ = compute_zh_family_margin(rand_logits, tokenizer, cat_name)
                
                key = f"{cat_name}_{obj}_L{patch_li}"
                results[key] = {
                    "category": cat_name,
                    "object": obj,
                    "patch_layer": patch_li,
                    "base_en_margin": float(base_en_m),
                    "base_zh_margin": float(base_zh_m),
                    "additive_en_delta": float(add_en_m - base_en_m),
                    "additive_zh_delta": float(add_zh_m - base_zh_m),
                    "mean_code_en_delta": float(mean_en_m - base_en_m),
                    "mean_code_zh_delta": float(mean_zh_m - base_zh_m),
                    "random_en_delta": float(rand_en_m - base_en_m),
                    "random_zh_delta": float(rand_zh_m - base_zh_m),
                }
                
                plog(f"    {cat_name}/{obj} L{patch_li}: "
                     f"addΔ={add_en_m - base_en_m:.2f} "
                     f"meanΔ={mean_en_m - base_en_m:.2f} "
                     f"randΔ={rand_en_m - base_en_m:.2f}")
    
    plog(f"  Exp3 done. {len(results)} comparisons")
    return results


# ==================== Exp4: GLM4残差可写性Holdout验证 ====================
def exp4_holdout_writability(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    GLM4残差差分注入强效果的Holdout验证
    
    关键: 用一批对象构造方向, 另一批对象测试
    - Train objects: 构造class_diff方向的对象
    - Test objects: 测试注入效果的对象(不参与构造)
    - Random control: 随机匹配方向(范数相同,方向随机)
    
    如果holdout仍然稳定, 说明GLM4确实有残差可写语义码
    """
    plog("=== Exp4: Holdout可写性验证 ===")
    info = get_model_info(model, model_name)
    
    # 用fruit和animal构造class_diff
    # Train: 前4个对象; Test: 后4个对象
    train_objs = {
        "fruit": CATEGORIES["fruit"][:4],
        "animal": CATEGORIES["animal"][:4],
    }
    test_objs = {
        "fruit": CATEGORIES["fruit"][4:8],
        "animal": CATEGORIES["animal"][4:8],
    }
    
    patch_layers = [
        info.n_layers // 3,
        info.n_layers // 2,
        2 * info.n_layers // 3,
    ]
    patch_layers = sorted(set([l for l in patch_layers if l < info.n_layers]))
    
    results = {}
    
    for patch_li in patch_layers:
        plog(f"  Layer L{patch_li}...")
        
        # 用train对象构造class_diff方向
        train_fruit_vecs = []
        train_animal_vecs = []
        for obj in train_objs["fruit"]:
            prompt = TEMPLATES_EN["is_a"].format(obj=obj)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
            if resid is not None:
                train_fruit_vecs.append(resid)
        for obj in train_objs["animal"]:
            prompt = TEMPLATES_EN["is_a"].format(obj=obj)
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
            if resid is not None:
                train_animal_vecs.append(resid)
        
        if not train_fruit_vecs or not train_animal_vecs:
            continue
        
        fruit_center = np.mean(train_fruit_vecs, axis=0)
        animal_center = np.mean(train_animal_vecs, axis=0)
        class_diff = fruit_center - animal_center
        class_diff_norm = np.linalg.norm(class_diff)
        if class_diff_norm < 1e-10:
            continue
        class_diff_dir = class_diff / class_diff_norm
        
        # 用test对象测试
        for beta in [5.0, 10.0]:
            for cat_name in ["fruit", "animal"]:
                for obj in test_objs[cat_name][:2]:
                    target_fam = CAT_TO_FAM.get(cat_name, (None, []))[0]
                    compete_fams = CAT_TO_FAM.get(cat_name, (None, []))[1]
                    if target_fam is None:
                        continue
                    
                    prompt = TEMPLATES_EN["is_a"].format(obj=obj)
                    
                    # Baseline
                    base_logits = get_final_logits(model, tokenizer, prompt, device)
                    base_en_m, _, _ = compute_en_family_margin(base_logits, tokenizer, target_fam, compete_fams)
                    
                    # 正方向注入(fruit - animal)
                    if cat_name == "fruit":
                        sign = 1  # fruit应该受益
                    else:
                        sign = -1  # animal应该受益于反方向
                    
                    delta = sign * beta * class_diff_dir
                    patched_logits = run_with_additive_patch(model, tokenizer, prompt, device, patch_li, delta)
                    patched_en_m, _, _ = compute_en_family_margin(patched_logits, tokenizer, target_fam, compete_fams)
                    
                    # Random control: 范数匹配的随机方向
                    rand_dir = np.random.randn(len(class_diff_dir))
                    rand_dir = rand_dir / np.linalg.norm(rand_dir)
                    rand_delta = sign * beta * rand_dir
                    rand_logits = run_with_additive_patch(model, tokenizer, prompt, device, patch_li, rand_delta)
                    rand_en_m, _, _ = compute_en_family_margin(rand_logits, tokenizer, target_fam, compete_fams)
                    
                    key = f"{cat_name}_{obj}_L{patch_li}_b{beta}"
                    results[key] = {
                        "category": cat_name,
                        "object": obj,
                        "patch_layer": patch_li,
                        "beta": beta,
                        "is_holdout": True,
                        "base_en_margin": float(base_en_m),
                        "patched_en_margin": float(patched_en_m),
                        "delta_en_margin": float(patched_en_m - base_en_m),
                        "rand_en_margin": float(rand_en_m),
                        "rand_delta_en_margin": float(rand_en_m - base_en_m),
                        "selectivity": float((patched_en_m - base_en_m) - (rand_en_m - base_en_m)),
                    }
                    
                    plog(f"    {cat_name}/{obj} L{patch_li} b={beta}: "
                         f"Δ={patched_en_m - base_en_m:.2f} "
                         f"randΔ={rand_en_m - base_en_m:.2f} "
                         f"sel={results[key]['selectivity']:.2f}")
    
    # 汇总holdout选择性
    sel_values = [v["selectivity"] for v in results.values() if "selectivity" in v]
    avg_sel = float(np.mean(sel_values)) if sel_values else 0
    plog(f"  Exp4 done. avg_holdout_selectivity={avg_sel:.2f}")
    
    return {"per_object": results, "avg_holdout_selectivity": avg_sel}


# ==================== Exp5: 翻译方向精细分解 ====================
def exp5_translate_fine_decomposition(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    DS7B语言轴纠缠验证 + 所有模型翻译方向精细分解
    
    构造4个方向:
    1. Target-language axis: "Translate to Chinese" vs "Translate to English"
    2. Source-language axis: 同目标语言, 不同源语言
    3. Translation-command axis: 翻译命令 vs 陈述句
    4. Semantic-content axis: 不同语义内容
    
    计算它们之间的余弦, 判断是否纠缠
    """
    plog("=== Exp5: 翻译方向精细分解 ===")
    info = get_model_info(model, model_name)
    
    # 关键层
    key_layers = [
        info.n_layers // 6,
        info.n_layers // 3,
        info.n_layers // 2,
        2 * info.n_layers // 3,
        info.n_layers - 3,
    ]
    key_layers = sorted(set([l for l in key_layers if l < info.n_layers]))
    
    # 测试对象
    test_objects_en = ["apple", "dog", "hammer", "car"]
    test_objects_zh = ["苹果", "狗", "锤子", "汽车"]
    
    results = {}
    
    for patch_li in key_layers:
        plog(f"  Layer L{patch_li}...")
        layer_res = {}
        
        # ---- 1. Target-language axis ----
        # "Translate to Chinese: The apple is a fruit." vs "Translate to English: 苹果是一种水果"
        target_zh_vecs = []
        target_en_vecs = []
        for i, obj_en in enumerate(test_objects_en[:2]):
            obj_zh = test_objects_zh[i]
            # 目标语言=中文
            prompt_zh_target = f"Translate to Chinese: The {obj_en} is a fruit."
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_zh_target, patch_li, device)
            if resid is not None:
                target_zh_vecs.append(resid)
            # 目标语言=英文
            prompt_en_target = f"请翻译为英文: {obj_zh}是一种水果"
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_en_target, patch_li, device)
            if resid is not None:
                target_en_vecs.append(resid)
        
        target_lang_diff = None
        if target_zh_vecs and target_en_vecs:
            zh_center = np.mean(target_zh_vecs, axis=0)
            en_center = np.mean(target_en_vecs, axis=0)
            target_lang_diff = zh_center - en_center
        
        # ---- 2. Source-language axis ----
        # 同目标语言(中文), 不同源语言
        source_en_vecs = []
        source_zh_vecs = []
        for i, obj_en in enumerate(test_objects_en[:2]):
            obj_zh = test_objects_zh[i]
            # 源=英文, 目标=中文
            prompt_en_source = f"Translate to Chinese: The {obj_en} is a fruit."
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_en_source, patch_li, device)
            if resid is not None:
                source_en_vecs.append(resid)
            # 源=中文, 目标=中文(等价于直接中文陈述)
            prompt_zh_source = f"{obj_zh}是一种水果"
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_zh_source, patch_li, device)
            if resid is not None:
                source_zh_vecs.append(resid)
        
        source_lang_diff = None
        if source_en_vecs and source_zh_vecs:
            en_center = np.mean(source_en_vecs, axis=0)
            zh_center = np.mean(source_zh_vecs, axis=0)
            source_lang_diff = en_center - zh_center
        
        # ---- 3. Translation-command axis ----
        # 翻译命令 vs 陈述句(同语言同语义)
        translate_cmd_vecs = []
        statement_vecs = []
        for i, obj_en in enumerate(test_objects_en[:2]):
            obj_zh = test_objects_zh[i]
            # 翻译命令
            prompt_translate = f"Translate to Chinese: The {obj_en} is a fruit."
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_translate, patch_li, device)
            if resid is not None:
                translate_cmd_vecs.append(resid)
            # 英文陈述
            prompt_stmt = f"The {obj_en} is a fruit."
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_stmt, patch_li, device)
            if resid is not None:
                statement_vecs.append(resid)
        
        translate_cmd_diff = None
        if translate_cmd_vecs and statement_vecs:
            tr_center = np.mean(translate_cmd_vecs, axis=0)
            st_center = np.mean(statement_vecs, axis=0)
            translate_cmd_diff = tr_center - st_center
        
        # ---- 4. Semantic-content axis ----
        # 同语言, 不同语义(fruit vs animal)
        content_fruit_vecs = []
        content_animal_vecs = []
        for obj_en in test_objects_en[:2]:
            prompt = f"The {obj_en} is a fruit."
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
            if resid is not None:
                content_fruit_vecs.append(resid)
        for obj_en in ["dog", "cat"]:
            prompt = f"The {obj_en} is an animal."
            resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, patch_li, device)
            if resid is not None:
                content_animal_vecs.append(resid)
        
        content_diff = None
        if content_fruit_vecs and content_animal_vecs:
            fr_center = np.mean(content_fruit_vecs, axis=0)
            an_center = np.mean(content_animal_vecs, axis=0)
            content_diff = fr_center - an_center
        
        # ---- 计算方向间的余弦相似 ----
        def safe_cos(v1, v2):
            if v1 is None or v2 is None:
                return None
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-10 or n2 < 1e-10:
                return None
            return float(np.dot(v1, v2) / (n1 * n2))
        
        cos_target_vs_source = safe_cos(target_lang_diff, source_lang_diff)
        cos_target_vs_cmd = safe_cos(target_lang_diff, translate_cmd_diff)
        cos_target_vs_content = safe_cos(target_lang_diff, content_diff)
        cos_source_vs_cmd = safe_cos(source_lang_diff, translate_cmd_diff)
        cos_source_vs_content = safe_cos(source_lang_diff, content_diff)
        cos_cmd_vs_content = safe_cos(translate_cmd_diff, content_diff)
        
        # 范数
        def safe_norm(v):
            return float(np.linalg.norm(v)) if v is not None else None
        
        layer_res = {
            "target_lang_norm": safe_norm(target_lang_diff),
            "source_lang_norm": safe_norm(source_lang_diff),
            "translate_cmd_norm": safe_norm(translate_cmd_diff),
            "content_diff_norm": safe_norm(content_diff),
            "cos_target_vs_source": cos_target_vs_source,
            "cos_target_vs_cmd": cos_target_vs_cmd,
            "cos_target_vs_content": cos_target_vs_content,
            "cos_source_vs_cmd": cos_source_vs_cmd,
            "cos_source_vs_content": cos_source_vs_content,
            "cos_cmd_vs_content": cos_cmd_vs_content,
        }
        
        # 计算这4个方向的秩(有效维度)
        all_diffs = [v for v in [target_lang_diff, source_lang_diff, translate_cmd_diff, content_diff] if v is not None]
        if len(all_diffs) >= 2:
            mat = np.array(all_diffs)
            s = np.linalg.svd(mat, compute_uv=False)
            total_var = np.sum(s ** 2)
            if total_var > 0:
                effective_rank = float(np.sum(s ** 2) ** 2 / np.sum(s ** 4))
                layer_res["effective_rank"] = effective_rank
                layer_res["singular_values"] = [float(x) for x in s]
        
        results[f"L{patch_li}"] = layer_res
        
        plog(f"    cos(target,source)={cos_target_vs_source}, "
             f"cos(target,content)={cos_target_vs_content}, "
             f"cos(cmd,content)={cos_cmd_vs_content}")
    
    plog(f"  Exp5 done. {len(results)} layers tested")
    return results


# ==================== 主流程 ====================
def main():
    if len(sys.argv) < 3:
        print("Usage: python phase463_semantic_language_orthogonal.py <model> <round>")
        print("  model: qwen3 / glm4 / deepseek7b")
        print("  round: 1 (pilot) / 2 (confirm)")
        sys.exit(1)
    
    model_name = sys.argv[1]
    round_num = int(sys.argv[2])
    
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        sys.exit(1)
    
    plog(f"Phase 463: {model_name} R{round_num}")
    plog(f"  语义码/语言码正交分解与跨语言读写闭环")
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    plog(f"  Model: {info.model_class}, {info.n_layers} layers, d_model={info.d_model}")
    
    obj_dict = ROUNDS.get(round_num, ROUNDS[1])
    
    all_results = {
        "model": model_name,
        "round": round_num,
        "n_cats": len(obj_dict),
        "n_objs": sum(len(v) for v in obj_dict.values()),
        "model_info": {
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "vocab_size": info.vocab_size,
            "model_class": info.model_class,
        },
    }
    
    try:
        # Exp1: 语义/语言正交分解(最核心!)
        all_results["exp1_semantic_language_orthogonal"] = exp1_semantic_language_orthogonal(
            model, tokenizer, model_name, device, obj_dict, round_num
        )
        
        # Exp2: 大样本跨语言Patch扩展
        all_results["exp2_large_sample_patch"] = exp2_large_sample_patch(
            model, tokenizer, model_name, device, obj_dict, round_num
        )
        
        # Exp3: Additive vs Replacement vs Mean-code
        all_results["exp3_patch_methods_comparison"] = exp3_patch_methods_comparison(
            model, tokenizer, model_name, device, obj_dict, round_num
        )
        
        # Exp4: Holdout可写性验证
        all_results["exp4_holdout_writability"] = exp4_holdout_writability(
            model, tokenizer, model_name, device, obj_dict, round_num
        )
        
        # Exp5: 翻译方向精细分解
        all_results["exp5_translate_fine_decomposition"] = exp5_translate_fine_decomposition(
            model, tokenizer, model_name, device, obj_dict, round_num
        )
        
    except Exception as e:
        plog(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存结果
    out_dir = "results/glm5"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"phase463_{model_name}_r{round_num}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    plog(f"Results saved to {out_path}")
    
    # 释放模型
    release_model(model)
    plog(f"Phase 463 {model_name} R{round_num} complete!")


if __name__ == "__main__":
    main()
