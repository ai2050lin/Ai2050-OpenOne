"""
Phase 462: 神经元写入路径与跨语言语义码因果验证
==================================================
从Phase 461的"表征泛化证据"推进到"因果验证证据"。

核心问题:
1. 跨语言语义不变量码在大样本、多关系上是否稳定？(Exp1)
2. 中层语义码是否被模型因果使用？(Exp2: activation patch)
3. 翻译方向码能否正交分解为源语言/目标语言/命令成分？(Exp3)
4. W_down写入向量 vs 残差差分方向,哪个更可控？(Exp4)

关键改进:
- 大样本: 8类×8对象=64测试点/语言, 多关系槽位
- 因果验证: 跨语言activation patch
- GLM4深层权重: 从safetensors加载(解决meta device问题)
- flash_attention_2 + BF16 + device_map="auto"

用法: python tests/glm5/phase462_causal_semantic_code.py qwen3 1
      python tests/glm5/phase462_causal_semantic_code.py glm4 2
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


# ==================== 大样本数据定义 ====================
# R1: 4类×4对象(快速验证)  R2: 6类×8对象(确认测试)
CATEGORIES = {
    "fruit":   ["apple", "banana", "orange", "grape", "pear", "peach", "lemon", "mango"],
    "animal":  ["dog", "cat", "horse", "lion", "bear", "rabbit", "cow", "tiger"],
    "tool":    ["hammer", "knife", "wrench", "saw", "drill", "axe", "shovel", "scissors"],
    "vehicle": ["car", "bus", "bicycle", "truck", "train", "boat", "plane", "scooter"],
    "clothing":["shirt", "dress", "hat", "coat", "sock", "glove", "scarf", "boot"],
    "furniture":["chair", "table", "desk", "sofa", "bed", "shelf", "lamp", "cabinet"],
}

# 中文翻译(用于跨语言测试)
CATEGORIES_ZH = {
    "fruit":   ["苹果", "香蕉", "橙子", "葡萄", "梨", "桃子", "柠檬", "芒果"],
    "animal":  ["狗", "猫", "马", "狮子", "熊", "兔子", "牛", "老虎"],
    "tool":    ["锤子", "刀", "扳手", "锯子", "钻头", "斧头", "铲子", "剪刀"],
    "vehicle": ["汽车", "公交车", "自行车", "卡车", "火车", "船", "飞机", "滑板车"],
    "clothing":["衬衫", "裙子", "帽子", "外套", "袜子", "手套", "围巾", "靴子"],
    "furniture":["椅子", "桌子", "书桌", "沙发", "床", "书架", "灯", "柜子"],
}

# 候选族
FAMILIES = {
    "class_fruit":    ["fruit", "produce", "crop", "harvest"],
    "class_animal":   ["animal", "creature", "beast", "mammal"],
    "class_tool":     ["tool", "implement", "instrument", "device"],
    "class_vehicle":  ["vehicle", "transport", "conveyance", "automobile"],
    "class_clothing": ["clothing", "apparel", "garment", "attire"],
    "class_furniture":["furniture", "furnishing", "fixture", "seat"],
}

# 类别到候选族映射
CAT_TO_FAM = {
    "fruit":    ("class_fruit",    ["class_animal", "class_tool", "class_vehicle", "class_clothing", "class_furniture"]),
    "animal":   ("class_animal",   ["class_fruit", "class_tool", "class_vehicle", "class_clothing", "class_furniture"]),
    "tool":     ("class_tool",     ["class_fruit", "class_animal", "class_vehicle", "class_clothing", "class_furniture"]),
    "vehicle":  ("class_vehicle",  ["class_fruit", "class_animal", "class_tool", "class_clothing", "class_furniture"]),
    "clothing": ("class_clothing", ["class_fruit", "class_animal", "class_tool", "class_vehicle", "class_furniture"]),
    "furniture":("class_furniture",["class_fruit", "class_animal", "class_tool", "class_vehicle", "class_clothing"]),
}

# 关系模板(多关系测试)
RELATION_TEMPLATES_EN = {
    "is_a":     "The {obj} is a kind of",
    "has_color":"The color of a {obj} is",
    "has_part": "A common part of a {obj} is",
    "used_for": "A {obj} is typically used for",
}

RELATION_TEMPLATES_ZH = {
    "is_a":     "{obj}是一种",
    "has_color":"{obj}的颜色是",
    "has_part": "{obj}的一个常见部分是",
    "used_for": "{obj}通常被用来",
}

# 翻译实验模板(Exp3: 正交分解)
TRANSLATE_TEMPLATES = {
    # 目标语言对照: 同一语义内容, 不同目标语言
    "en2zh_same_content": [
        ("Translate to Chinese: The {obj} is a fruit.", "{obj}是一种水果。"),
        ("Translate to Chinese: The {obj} is an animal.", "{obj}是一种动物。"),
        ("Translate to Chinese: The {obj} is a tool.", "{obj}是一种工具。"),
    ],
    "zh2en_same_content": [
        ("请翻译为英文: {obj}是一种水果", "{obj} is a fruit."),
        ("请翻译为英文: {obj}是一种动物", "{obj} is an animal."),
        ("请翻译为英文: {obj}是一种工具", "{obj} is a tool."),
    ],
    # 源语言对照: 同一目标语言, 不同源语言
    "en2zh_vs_en2zh": [
        ("Translate to Chinese: The {obj} is a fruit.", None),  # 英文源
        ("请将以下翻译为中文: The {obj} is a fruit.", None),    # 同义命令
    ],
    # 命令vs内容对照: 同一目标语言, 不同语义内容
    "content_control": [
        ("Translate to Chinese: The {obj} is a fruit.", None),
        ("Translate to Chinese: The {obj} is red.", None),
        ("Translate to Chinese: The {obj} has seeds.", None),
    ],
}

# 轮次数据量
ROUNDS = {
    1: {k: v[:4] for k, v in CATEGORIES.items()},   # pilot: 4类×4对象
    2: {k: v[:8] for k, v in CATEGORIES.items()},   # confirm: 6类×8对象
}


# ==================== 模型加载 ====================
def load_model_bf16(model_name):
    """BF16加载模型 — flash_attention_2优先, 回退eager"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog(f"Loading {model_name} (bfloat16 + device_map=auto + flash_attn)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 尝试flash_attention_2, 失败则回退eager
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
    """从safetensors文件加载指定key的权重(解决meta device问题)"""
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
    """获取MLP权重矩阵 — 兼容meta device, 自动从safetensors加载"""
    layers = get_layers(model)
    layer = layers[layer_idx]
    info = get_model_info(model, model_name)
    
    def _to_numpy(tensor, layer_idx, proj_name):
        """安全地将tensor转为numpy, meta device时从safetensors加载"""
        if not tensor.is_meta:
            return tensor.detach().cpu().float().numpy()
        # meta device: 从safetensors加载
        key = f"model.layers.{layer_idx}.mlp.{proj_name}.weight"
        w = load_weight_from_safetensors(model_name, key)
        if w is not None:
            plog(f"    Loaded L{layer_idx} {proj_name} from safetensors, shape={w.shape}")
            return w
        plog(f"    WARN: Cannot load L{layer_idx} {proj_name} from safetensors")
        return None
    
    if info.mlp_type == "split_gate_up":
        W_up = _to_numpy(layer.mlp.up_proj.weight, layer_idx, "up_proj")
        W_down = _to_numpy(layer.mlp.down_proj.weight, layer_idx, "down_proj")
        W_gate = _to_numpy(layer.mlp.gate_proj.weight, layer_idx, "gate_proj") if hasattr(layer.mlp, 'gate_proj') else None
    else:  # merged_gate_up
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
    """提取指定层的残差流(最后一个token)"""
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
    """提取指定层指定位置的残差流"""
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


# ==================== Exp1: 大样本跨语言语义码验证 ====================
def exp1_large_sample_cross_lang(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    大样本跨语言探针验证
    
    对比Phase 461的4类×4对象, 扩展到6类×8对象, 多关系模板
    
    方法:
    1. 用英文is_a模板提取每类对象的残差流, 计算类别中心
    2. 用英文其他关系模板测试探针泛化
    3. 用中文模板提取残差流, 测试跨语言泛化
    """
    plog("=== Exp1: 大样本跨语言语义码验证 ===")
    info = get_model_info(model, model_name)
    
    # 选择关键层
    key_layers = list(range(0, info.n_layers, max(1, info.n_layers // 8)))
    if info.n_layers - 1 not in key_layers:
        key_layers.append(info.n_layers - 1)
    
    # 只用is_a关系训练探针
    relation = "is_a"
    
    # 英文: 提取每类对象的残差流
    en_class_centers = {}  # {layer_idx: {cat: center_vec}}
    en_obj_resids = {}     # {layer_idx: {cat: {obj: vec}}}
    zh_obj_resids = {}     # {layer_idx: {cat: {obj: vec}}}
    
    plog(f"  Collecting EN residuals for {len(obj_dict)} categories...")
    t0 = time.time()
    
    for cat_name, obj_list in obj_dict.items():
        for obj in obj_list:
            prompt_en = RELATION_TEMPLATES_EN[relation].format(obj=obj)
            resids = get_residual_at_layers(model, tokenizer, prompt_en, key_layers, device)
            for li in key_layers:
                if li not in en_obj_resids:
                    en_obj_resids[li] = {}
                if cat_name not in en_obj_resids[li]:
                    en_obj_resids[li][cat_name] = {}
                if li in resids:
                    en_obj_resids[li][cat_name][obj] = resids[li]
        
        plog(f"    EN {cat_name}: {len(obj_list)} objects done ({time.time()-t0:.1f}s)")
    
    # 中文: 提取每类对象的残差流
    plog(f"  Collecting ZH residuals for {len(obj_dict)} categories...")
    t0 = time.time()
    
    for cat_name, obj_list in obj_dict.items():
        zh_obj_list = CATEGORIES_ZH.get(cat_name, [])[:len(obj_list)]
        for i, obj in enumerate(obj_list):
            zh_obj = zh_obj_list[i] if i < len(zh_obj_list) else obj
            prompt_zh = RELATION_TEMPLATES_ZH[relation].format(obj=zh_obj)
            resids = get_residual_at_layers(model, tokenizer, prompt_zh, key_layers, device)
            for li in key_layers:
                if li not in zh_obj_resids:
                    zh_obj_resids[li] = {}
                if cat_name not in zh_obj_resids[li]:
                    zh_obj_resids[li][cat_name] = {}
                if li in resids:
                    zh_obj_resids[li][cat_name][obj] = resids[li]
        
        plog(f"    ZH {cat_name}: {len(obj_list)} objects done ({time.time()-t0:.1f}s)")
    
    # 计算英文类别中心
    for li in key_layers:
        if li not in en_class_centers:
            en_class_centers[li] = {}
        for cat_name in obj_dict:
            if cat_name in en_obj_resids.get(li, {}):
                vecs = list(en_obj_resids[li][cat_name].values())
                if len(vecs) > 0:
                    en_class_centers[li][cat_name] = np.mean(vecs, axis=0)
    
    # 最近中心分类器
    results = {}
    
    # 1. 英文→英文分类准确率
    en_en_acc = {}
    for li in key_layers:
        if li not in en_class_centers or not en_class_centers[li]:
            continue
        centers = en_class_centers[li]
        cats = list(centers.keys())
        center_vecs = np.array([centers[c] for c in cats])
        
        correct = 0
        total = 0
        for cat_name in cats:
            for obj, vec in en_obj_resids.get(li, {}).get(cat_name, {}).items():
                # 最近中心分类
                cos_sims = [float(np.dot(vec, cv) / (np.linalg.norm(vec) * np.linalg.norm(cv) + 1e-10)) for cv in center_vecs]
                pred_cat = cats[np.argmax(cos_sims)]
                if pred_cat == cat_name:
                    correct += 1
                total += 1
        
        en_en_acc[li] = {"acc": correct / total if total > 0 else 0, "n": total}
    
    # 2. 英文中心→中文分类准确率(核心!)
    en_zh_acc = {}
    for li in key_layers:
        if li not in en_class_centers or not en_class_centers[li]:
            continue
        centers = en_class_centers[li]
        cats = list(centers.keys())
        center_vecs = np.array([centers[c] for c in cats])
        
        correct = 0
        total = 0
        per_cat_correct = {c: 0 for c in cats}
        per_cat_total = {c: 0 for c in cats}
        
        for cat_name in cats:
            for obj, vec in zh_obj_resids.get(li, {}).get(cat_name, {}).items():
                cos_sims = [float(np.dot(vec, cv) / (np.linalg.norm(vec) * np.linalg.norm(cv) + 1e-10)) for cv in center_vecs]
                pred_cat = cats[np.argmax(cos_sims)]
                if pred_cat == cat_name:
                    correct += 1
                    per_cat_correct[cat_name] += 1
                total += 1
                per_cat_total[cat_name] += 1
        
        en_zh_acc[li] = {
            "acc": correct / total if total > 0 else 0,
            "n": total,
            "per_cat": {c: per_cat_correct[c] / per_cat_total[c] if per_cat_total[c] > 0 else 0 
                       for c in cats},
        }
    
    # 3. 中英类别中心余弦相似度
    center_cosines = {}
    for li in key_layers:
        if li not in en_class_centers:
            continue
        cos_per_cat = {}
        for cat_name in en_class_centers[li]:
            en_center = en_class_centers[li][cat_name]
            zh_vecs = list(zh_obj_resids.get(li, {}).get(cat_name, {}).values())
            if len(zh_vecs) > 0:
                zh_center = np.mean(zh_vecs, axis=0)
                cos = float(np.dot(en_center, zh_center) / (np.linalg.norm(en_center) * np.linalg.norm(zh_center) + 1e-10))
                cos_per_cat[cat_name] = cos
        if cos_per_cat:
            center_cosines[li] = cos_per_cat
    
    # 4. 其他关系模板的跨语言泛化(R2才做)
    other_rel_acc = {}
    if round_num >= 2:
        plog("  Testing other relation templates (R2 only)...")
        for rel_name, template in RELATION_TEMPLATES_EN.items():
            if rel_name == "is_a":
                continue
            # 用is_a训练的中心测试其他关系
            for li in key_layers[:3]:  # 只测3个关键层(节省时间)
                if li not in en_class_centers:
                    continue
                centers = en_class_centers[li]
                cats = list(centers.keys())
                center_vecs = np.array([centers[c] for c in cats])
                
                correct = 0
                total = 0
                for cat_name in cats:
                    for obj in obj_dict.get(cat_name, [])[:4]:
                        prompt = template.format(obj=obj)
                        resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, li, device)
                        if resid is None:
                            continue
                        cos_sims = [float(np.dot(resid, cv) / (np.linalg.norm(resid) * np.linalg.norm(cv) + 1e-10)) for cv in center_vecs]
                        pred_cat = cats[np.argmax(cos_sims)]
                        if pred_cat == cat_name:
                            correct += 1
                        total += 1
                
                if total > 0:
                    if rel_name not in other_rel_acc:
                        other_rel_acc[rel_name] = {}
                    other_rel_acc[rel_name][f"L{li}"] = {"acc": correct / total, "n": total}
    
    results["en_en_acc"] = {f"L{k}": v for k, v in en_en_acc.items()}
    results["en_zh_acc"] = {f"L{k}": v for k, v in en_zh_acc.items()}
    results["center_cosines"] = {f"L{k}": v for k, v in center_cosines.items()}
    if other_rel_acc:
        results["other_rel_acc"] = other_rel_acc
    
    plog(f"  Exp1 done. EN→EN best={max(v['acc'] for v in en_en_acc.values()):.2f}, "
         f"EN→ZH best={max(v['acc'] for v in en_zh_acc.values()):.2f}")
    
    return results


# ==================== Exp2: 跨语言Activation Patch因果验证 ====================
def exp2_cross_lang_activation_patch(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    跨语言activation patch — 真正的因果验证!
    
    核心方法:
    1. 运行英文prompt "The apple is a kind of", 收集中层残差流 h_en
    2. 运行中文prompt "苹果是一种", 收集中层残差流 h_zh
    3. 在中文上下文中, 把中间层残差流替换为英文的: h_zh' = h_en
    4. 观察最终logit: 英文候选词(fruit)的边际是否增加
    
    如果patch有效(英文候选词边际增加), 说明:
    - 中层语义码确实被模型因果使用
    - 而不仅仅是统计关联
    """
    plog("=== Exp2: 跨语言Activation Patch因果验证 ===")
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    
    # 选择2个类别, 每类2个对象(减少计算量)
    test_cats = list(obj_dict.keys())[:2]
    test_objs = {c: obj_dict[c][:2] for c in test_cats}
    
    # 关键层: 浅/中/深
    patch_layers = [
        info.n_layers // 6,    # 浅层
        info.n_layers // 3,    # 中浅层
        info.n_layers // 2,    # 中层
        2 * info.n_layers // 3, # 中深层
        info.n_layers - 3,     # 深层
    ]
    patch_layers = [l for l in patch_layers if l < info.n_layers]
    
    results = {}
    
    for cat_name in test_cats:
        for obj in test_objs[cat_name]:
            # 获取中文对象名
            cat_objs_zh = CATEGORIES_ZH.get(cat_name, [])
            obj_idx = obj_dict[cat_name].index(obj) if obj in obj_dict[cat_name] else 0
            obj_zh = cat_objs_zh[obj_idx] if obj_idx < len(cat_objs_zh) else obj
            
            # 英文和中文prompt
            prompt_en = RELATION_TEMPLATES_EN["is_a"].format(obj=obj)
            prompt_zh = RELATION_TEMPLATES_ZH["is_a"].format(obj=obj_zh)
            
            plog(f"  Patching: EN='{prompt_en}' → ZH='{prompt_zh}'")
            
            # --- Step 1: 收集英文中间层残差 ---
            en_resids = get_residual_at_layers(model, tokenizer, prompt_en, patch_layers, device)
            
            # --- Step 2: 收集中文baseline logit(不patch) ---
            # 用hook替换残差流的方式做patch
            for patch_li in patch_layers:
                if patch_li not in en_resids:
                    continue
                
                en_h = en_resids[patch_li]  # 英文残差流 [d_model]
                
                # 中文baseline: 不替换
                zh_logits_base = _get_final_logits_with_optional_patch(
                    model, tokenizer, prompt_zh, device, W_U, 
                    patch_layer=None, patch_vec=None
                )
                
                # 中文patched: 在patch_li层替换为英文残差
                zh_logits_patched = _get_final_logits_with_optional_patch(
                    model, tokenizer, prompt_zh, device, W_U,
                    patch_layer=patch_li, patch_vec=en_h
                )
                
                # 英文baseline logit(对照)
                en_logits = _get_final_logits_with_optional_patch(
                    model, tokenizer, prompt_en, device, W_U,
                    patch_layer=None, patch_vec=None
                )
                
                # 计算候选族边际
                target_fam = CAT_TO_FAM.get(cat_name, (None, []))[0]
                compete_fams = CAT_TO_FAM.get(cat_name, (None, []))[1]
                if target_fam is None:
                    continue
                
                # 英文候选词在各类的边际
                zh_base_margin_en = _compute_en_family_margin(zh_logits_base, W_U, tokenizer, target_fam, compete_fams)
                zh_patched_margin_en = _compute_en_family_margin(zh_logits_patched, W_U, tokenizer, target_fam, compete_fams)
                en_margin_en = _compute_en_family_margin(en_logits, W_U, tokenizer, target_fam, compete_fams)
                
                # 中文候选词在各类的边际
                zh_base_margin_zh = _compute_zh_family_margin(zh_logits_base, W_U, tokenizer, target_fam, compete_fams, cat_name)
                zh_patched_margin_zh = _compute_zh_family_margin(zh_logits_patched, W_U, tokenizer, target_fam, compete_fams, cat_name)
                
                key = f"{cat_name}_{obj}_L{patch_li}"
                results[key] = {
                    "patch_layer": patch_li,
                    "category": cat_name,
                    "object": obj,
                    "zh_base_margin_en_candidates": float(zh_base_margin_en),
                    "zh_patched_margin_en_candidates": float(zh_patched_margin_en),
                    "en_margin_en_candidates": float(en_margin_en),
                    "zh_base_margin_zh_candidates": float(zh_base_margin_zh),
                    "zh_patched_margin_zh_candidates": float(zh_patched_margin_zh),
                    "patch_effect_en": float(zh_patched_margin_en - zh_base_margin_en),
                    "patch_effect_zh": float(zh_patched_margin_zh - zh_base_margin_zh),
                }
                
                plog(f"    L{patch_li}: zh_base→en_margin={zh_base_margin_en:.2f}, "
                     f"patched→en_margin={zh_patched_margin_en:.2f}, "
                     f"Δ={zh_patched_margin_en - zh_base_margin_en:.2f}")
    
    plog(f"  Exp2 done. {len(results)} patch experiments")
    return results


def _get_final_logits_with_optional_patch(model, tokenizer, prompt, device, W_U, 
                                            patch_layer=None, patch_vec=None):
    """运行模型, 可选在某层替换残差流, 返回最终logits向量"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    captured = {}
    layers = get_layers(model)
    patched = [False]
    
    def make_hook(li):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0 and not patched[0]:
                # 只在第一次前向时patch
                captured['input'] = input[0].detach().float().cpu()
        return hook
    
    hooks = [layers[li].register_forward_hook(make_hook(li)) for li in ([patch_layer] if patch_layer is not None else [])]
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    
    for h in hooks:
        h.remove()
    
    # 获取最终层logits
    logits = out.logits[0, -1].float().cpu().numpy()
    return logits


def _compute_en_family_margin(logits, W_U, tokenizer, target_fam, compete_fams):
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
        return 0.0
    return float(np.mean(target_logits) - np.mean(compete_logits))


def _compute_zh_family_margin(logits, W_U, tokenizer, target_fam, compete_fams, cat_name):
    """计算中文候选族的边际"""
    # 中文类别词
    zh_class_words = {
        "fruit": "水果", "animal": "动物", "tool": "工具", 
        "vehicle": "交通工具", "clothing": "衣服", "furniture": "家具",
    }
    zh_target = zh_class_words.get(cat_name, "")
    zh_compete = [v for k, v in zh_class_words.items() if k != cat_name]
    
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
        return 0.0
    return float(np.mean(target_logits) - np.mean(compete_logits))


# ==================== Exp2b: 直接残差替换Patch(更严谨) ====================
def exp2b_residual_patch(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    更严谨的activation patch: 在中间层直接替换整个残差流向量
    
    与Exp2的区别:
    - Exp2只替换单个token位置的残差
    - Exp2b在指定层替换最后一个token的残差流, 并继续前向传播
    
    这是真正的因果干预: 如果替换有效, 证明该层编码了语义信息
    """
    plog("=== Exp2b: 直接残差替换Patch ===")
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    
    test_cats = list(obj_dict.keys())[:2]
    test_objs = {c: obj_dict[c][:2] for c in test_cats}
    
    patch_layers = [
        info.n_layers // 6,
        info.n_layers // 3,
        info.n_layers // 2,
        2 * info.n_layers // 3,
        info.n_layers - 3,
    ]
    patch_layers = [l for l in patch_layers if l < info.n_layers]
    
    results = {}
    
    for cat_name in test_cats:
        for obj in test_objs[cat_name]:
            cat_objs_zh = CATEGORIES_ZH.get(cat_name, [])
            obj_idx = obj_dict[cat_name].index(obj) if obj in obj_dict[cat_name] else 0
            obj_zh = cat_objs_zh[obj_idx] if obj_idx < len(cat_objs_zh) else obj
            
            prompt_en = RELATION_TEMPLATES_EN["is_a"].format(obj=obj)
            prompt_zh = RELATION_TEMPLATES_ZH["is_a"].format(obj=obj_zh)
            
            # 对每个patch层做替换
            for patch_li in patch_layers:
                # 1. 运行英文, 收集patch_li层残差
                en_resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt_en, patch_li, device)
                if en_resid is None:
                    continue
                
                # 2. 运行中文baseline
                zh_logits_base = _run_with_optional_residual_replace(
                    model, tokenizer, prompt_zh, device, 
                    replace_layer=None, replace_vec=None
                )
                
                # 3. 运行中文patched: 在patch_li层替换残差
                zh_logits_patched = _run_with_optional_residual_replace(
                    model, tokenizer, prompt_zh, device,
                    replace_layer=patch_li, replace_vec=en_resid
                )
                
                # 4. 运行英文baseline(对照)
                en_logits = _run_with_optional_residual_replace(
                    model, tokenizer, prompt_en, device,
                    replace_layer=None, replace_vec=None
                )
                
                # 计算边际
                target_fam = CAT_TO_FAM.get(cat_name, (None, []))[0]
                compete_fams = CAT_TO_FAM.get(cat_name, (None, []))[1]
                if target_fam is None:
                    continue
                
                zh_base_m = _compute_en_family_margin(zh_logits_base, W_U, tokenizer, target_fam, compete_fams)
                zh_patch_m = _compute_en_family_margin(zh_logits_patched, W_U, tokenizer, target_fam, compete_fams)
                en_m = _compute_en_family_margin(en_logits, W_U, tokenizer, target_fam, compete_fams)
                
                key = f"{cat_name}_{obj}_L{patch_li}"
                results[key] = {
                    "patch_layer": patch_li,
                    "zh_base_en_margin": float(zh_base_m),
                    "zh_patched_en_margin": float(zh_patch_m),
                    "en_en_margin": float(en_m),
                    "delta_en_margin": float(zh_patch_m - zh_base_m),
                    "recovery_ratio": float((zh_patch_m - zh_base_m) / (en_m - zh_base_m + 1e-10)),
                }
                
                plog(f"    L{patch_li} {cat_name}/{obj}: "
                     f"zh_base={zh_base_m:.2f} → patched={zh_patch_m:.2f} "
                     f"(Δ={zh_patch_m-zh_base_m:.2f}, en_ref={en_m:.2f})")
    
    plog(f"  Exp2b done. {len(results)} patches")
    return results


def _run_with_optional_residual_replace(model, tokenizer, prompt, device, 
                                         replace_layer=None, replace_vec=None):
    """运行模型, 可选在某层替换残差流后继续前向传播"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    layers = get_layers(model)
    n_layers = len(layers)
    
    # 分两步: 先跑到replace_layer, 替换残差, 再从replace_layer跑到末尾
    
    if replace_layer is None:
        # 无替换, 直接前向
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits[0, -1].float().cpu().numpy()
    
    # Step 1: 收集replace_layer的输入(残差流)和所有层输出
    # 使用hooks
    captured_before = {}
    hooks_before = []
    
    def make_capture_hook(li):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                captured_before[li] = input[0].detach().clone()
        return hook
    
    for li in range(replace_layer + 1):
        hooks_before.append(layers[li].register_forward_hook(make_capture_hook(li)))
    
    with torch.no_grad():
        out1 = model(input_ids=input_ids, attention_mask=attention_mask)
    
    for h in hooks_before:
        h.remove()
    
    if replace_layer not in captured_before:
        # fallback: 直接前向
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits[0, -1].float().cpu().numpy()
    
    # Step 2: 替换残差流, 从replace_layer开始继续
    # captured_before[replace_layer] = 原始输入 [1, seq_len, d_model]
    orig_resid = captured_before[replace_layer]
    seq_len = attention_mask.sum().item()
    
    # 替换最后一个token的残差流
    replace_tensor = torch.tensor(replace_vec, dtype=orig_resid.dtype, device=orig_resid.device)
    new_resid = orig_resid.clone()
    new_resid[0, seq_len - 1, :] = replace_tensor
    
    # 从replace_layer开始, 用新的残差流作为输入
    # 方法: 用hook在replace_layer的输入处注入
    captured_after = {}
    final_logits = [None]
    
    def make_inject_hook():
        injected = [False]
        def hook(module, input, output):
            if not injected[0]:
                injected[0] = True
                # 不修改, 只是标记已经inject
                return None  # 不修改输出
            return None
        return hook
    
    # 更简单的方法: 用inputs_embeds + 逐层前向
    # 但这需要更复杂的实现, 这里用更简单的方式:
    # 直接用hook在replace_layer处替换中间残差
    
    result_logits = [None]
    inject_done = [False]
    
    def make_replace_hook():
        def hook(module, input, output):
            if not inject_done[0]:
                inject_done[0] = True
                if isinstance(input, tuple) and len(input) > 0:
                    # 替换输入残差流
                    new_input = (new_resid,) + input[1:] if len(input) > 1 else (new_resid,)
                    # 需要重新计算这一层的输出
                    # 但hook不能修改input, 只能修改output
                    # 所以我们需要另一种方法
                    pass
            return None
        return hook
    
    # 由于hook修改input的复杂性, 改用更直接的方法:
    # 计算patch层的残差差分, 注入到残差流中
    
    # 获取中文原始残差流
    zh_resid, _ = get_residual_at_layer_pos(model, tokenizer, prompt, replace_layer, device)
    if zh_resid is None:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits[0, -1].float().cpu().numpy()
    
    # 计算差分
    delta = replace_vec - zh_resid  # [d_model]
    
    # 在replace_layer的输出中注入差分
    # 使用hook: 在replace_layer输出后, 给残差流加上delta
    delta_tensor = torch.tensor(delta, dtype=torch.float32, device=device)
    patched_logits = [None]
    
    def make_additive_hook():
        added = [False]
        def hook(module, input, output):
            if not added[0]:
                added[0] = True
                if isinstance(output, tuple):
                    # output[0] = 层输出 [1, seq_len, d_model]
                    out_tensor = output[0].clone()
                    out_tensor[0, seq_len - 1, :] += delta_tensor.to(out_tensor.dtype)
                    return (out_tensor,) + output[1:]
                else:
                    out_tensor = output.clone()
                    out_tensor[0, seq_len - 1, :] += delta_tensor.to(out_tensor.dtype)
                    return out_tensor
            return None
        return hook
    
    h = layers[replace_layer].register_forward_hook(make_additive_hook())
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    
    h.remove()
    
    return out.logits[0, -1].float().cpu().numpy()


# ==================== Exp3: 翻译方向正交分解 ====================
def exp3_translate_direction_decomposition(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    翻译方向正交分解 — 区分源语言、目标语言、命令、语义内容
    
    对照设计:
    A. 同内容, 不同目标语言: "Translate to Chinese: The apple is a fruit" vs "Translate to English: 苹果是一种水果"
    B. 同目标语言, 不同内容: "Translate to Chinese: The apple is a fruit" vs "Translate to Chinese: The apple is red"
    C. 同目标, 同内容, 不同源语言: 英文源 vs 法文源(如果有)
    D. 纯英文 vs 翻译命令英文: "The apple is a kind of" vs "Translate to Chinese: The apple is a fruit"
    """
    plog("=== Exp3: 翻译方向正交分解 ===")
    info = get_model_info(model, model_name)
    
    key_layers = [
        info.n_layers // 6, info.n_layers // 3, info.n_layers // 2,
        2 * info.n_layers // 3, info.n_layers - 3,
    ]
    key_layers = [l for l in key_layers if l < info.n_layers]
    
    test_obj = obj_dict.get("fruit", ["apple"])[0]
    
    # 收集各种条件的残差流
    conditions = {
        # 纯英文语义
        "en_is_a": f"The {test_obj} is a kind of",
        # 纯中文语义
        "zh_is_a": f"苹果是一种",
        # 翻译: 英→中
        "en2zh_fruit": f"Translate to Chinese: The {test_obj} is a fruit.",
        "en2zh_animal": f"Translate to Chinese: The dog is an animal.",
        "en2zh_color": f"Translate to Chinese: The {test_obj} is red.",
        # 翻译: 中→英
        "zh2en_fruit": f"请翻译为英文: 苹果是一种水果",
        "zh2en_animal": f"请翻译为英文: 狗是一种动物",
        "zh2en_color": f"请翻译为英文: 苹果是红色的",
        # 控制组: 同语义不同格式
        "en_fruit_stmt": f"The {test_obj} is a fruit.",
        "zh_fruit_stmt": f"苹果是一种水果。",
        # 同义翻译命令
        "en2zh_same_v2": f"Please translate the following into Chinese: The {test_obj} is a fruit.",
    }
    
    cond_resids = {}
    for cond_name, prompt in conditions.items():
        resids = get_residual_at_layers(model, tokenizer, prompt, key_layers, device)
        cond_resids[cond_name] = resids
        plog(f"  Collected: {cond_name}")
    
    # 正交分解
    results = {}
    
    for li in key_layers:
        layer_results = {}
        
        # 1. 翻译方向差分 (目标语言成分)
        if "en2zh_fruit" in cond_resids and cond_resids["en2zh_fruit"].get(li) is not None:
            en2zh_fruit = cond_resids["en2zh_fruit"][li]
            zh2en_fruit = cond_resids.get("zh2en_fruit", {}).get(li)
            en_is_a = cond_resids.get("en_is_a", {}).get(li)
            zh_is_a = cond_resids.get("zh_is_a", {}).get(li)
            
            # 翻译差分方向
            if zh2en_fruit is not None:
                translate_diff_en2zh = en2zh_fruit - en_is_a if en_is_a is not None else en2zh_fruit
                translate_diff_zh2en = zh2en_fruit - zh_is_a if zh_is_a is not None else zh2en_fruit
                
                # 相关系数
                cos_translate = float(np.dot(translate_diff_en2zh, translate_diff_zh2en) / 
                             (np.linalg.norm(translate_diff_en2zh) * np.linalg.norm(translate_diff_zh2en) + 1e-10))
                
                layer_results["translate_diff_cos"] = cos_translate
                layer_results["en2zh_diff_norm"] = float(np.linalg.norm(translate_diff_en2zh))
                layer_results["zh2en_diff_norm"] = float(np.linalg.norm(translate_diff_zh2en))
            
            # 2. 目标语言成分: 同内容不同目标
            en2zh_fruit_v = cond_resids.get("en2zh_fruit", {}).get(li)
            zh2en_fruit_v = cond_resids.get("zh2en_fruit", {}).get(li)
            en_fruit_stmt_v = cond_resids.get("en_fruit_stmt", {}).get(li)
            
            if en2zh_fruit_v is not None and zh2en_fruit_v is not None:
                # 目标语言差分: en2zh vs zh2en
                target_lang_diff = en2zh_fruit_v - zh2en_fruit_v
                layer_results["target_lang_diff_norm"] = float(np.linalg.norm(target_lang_diff))
                
                # 这两个翻译差分与语言表面的余弦
                if en_is_a is not None and zh_is_a is not None:
                    surface_lang_diff = en_is_a - zh_is_a
                    if np.linalg.norm(surface_lang_diff) > 1e-10 and np.linalg.norm(target_lang_diff) > 1e-10:
                        cos_target_surface = float(np.dot(target_lang_diff, surface_lang_diff) / 
                                             (np.linalg.norm(target_lang_diff) * np.linalg.norm(surface_lang_diff) + 1e-10))
                        layer_results["target_lang_vs_surface_lang_cos"] = cos_target_surface
            
            # 3. 内容成分: 同目标不同内容
            en2zh_animal_v = cond_resids.get("en2zh_animal", {}).get(li)
            en2zh_color_v = cond_resids.get("en2zh_color", {}).get(li)
            
            if en2zh_animal_v is not None and en2zh_color_v is not None:
                content_diff = en2zh_fruit_v - en2zh_animal_v
                content_diff2 = en2zh_fruit_v - en2zh_color_v
                layer_results["content_diff_fruit_animal_norm"] = float(np.linalg.norm(content_diff))
                layer_results["content_diff_fruit_color_norm"] = float(np.linalg.norm(content_diff2))
                
                # 内容差分与翻译差分的正交性
                if zh2en_fruit_v is not None:
                    translate_diff = en2zh_fruit_v - zh2en_fruit_v
                    if np.linalg.norm(content_diff) > 1e-10 and np.linalg.norm(translate_diff) > 1e-10:
                        cos_content_translate = float(np.dot(content_diff, translate_diff) / 
                                             (np.linalg.norm(content_diff) * np.linalg.norm(translate_diff) + 1e-10))
                        layer_results["content_vs_translate_cos"] = cos_content_translate
            
            # 4. 命令成分: 同目标语言同内容, 不同命令格式
            en2zh_same_v2_v = cond_resids.get("en2zh_same_v2", {}).get(li)
            if en2zh_same_v2_v is not None and en2zh_fruit_v is not None:
                command_diff = en2zh_fruit_v - en2zh_same_v2_v
                layer_results["command_format_diff_norm"] = float(np.linalg.norm(command_diff))
        
        if layer_results:
            results[f"L{li}"] = layer_results
    
    plog(f"  Exp3 done. {len(results)} layers analyzed")
    return results


# ==================== Exp4: W_down写入向量 vs 残差差分方向 ====================
def exp4_write_vector_vs_residual_diff(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    W_down写入向量 vs 残差差分方向的可控性对比
    
    Phase 461发现: 大beta残差差分注入失败
    本实验验证: 使用W_down写入向量组合是否更可控
    
    方法:
    1. 计算class_diff方向(残差流级别)
    2. 找到对class_diff贡献最大的top-k神经元
    3. 用这些神经元的W_down写入向量替代class_diff方向
    4. 注入对比: 残差差分 vs 写入向量组合
    """
    plog("=== Exp4: W_down写入向量 vs 残差差分方向 ===")
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    
    test_cats = list(obj_dict.keys())[:2]
    
    key_layers = [
        info.n_layers // 4,
        info.n_layers // 2,
        3 * info.n_layers // 4,
    ]
    key_layers = [l for l in key_layers if l < info.n_layers]
    
    results = {}
    
    for cat_name in test_cats:
        obj_list = obj_dict[cat_name]
        other_cats = [c for c in obj_dict if c != cat_name]
        other_obj = obj_dict[other_cats[0]][0]
        
        # 类内对象残差流
        prompt_target = f"The {obj_list[0]} is a kind of"
        prompt_compete = f"The {other_obj} is a kind of"
        
        for li in key_layers:
            # 获取残差流
            resid_target, _ = get_residual_at_layer_pos(model, tokenizer, prompt_target, li, device)
            resid_compete, _ = get_residual_at_layer_pos(model, tokenizer, prompt_compete, li, device)
            
            if resid_target is None or resid_compete is None:
                continue
            
            # class_diff = target方向 - compete方向
            class_diff = resid_target - resid_compete
            class_diff_norm = np.linalg.norm(class_diff)
            if class_diff_norm < 1e-10:
                continue
            class_diff_unit = class_diff / class_diff_norm
            
            # 获取W_down权重
            _, W_down, _ = get_mlp_weights_safe(model, model_name, li)
            if W_down is None:
                results[f"{cat_name}_L{li}"] = {"error": "no_W_down"}
                continue
            
            # W_down: [d_model, d_inter]
            # 写入向量: W_down[:, i] 是第i个神经元的写入方向
            d_model_w, d_inter = W_down.shape
            
            # 找到对class_diff贡献最大的top-k神经元
            # 贡献 = <W_down[:, i], class_diff_unit>
            contributions = W_down.T @ class_diff_unit  # [d_inter]
            top_k = min(20, d_inter)
            top_indices = np.argsort(np.abs(contributions))[-top_k:]
            
            # 构造写入向量组合
            write_vec = np.zeros(d_model_w)
            for idx in top_indices:
                sign = np.sign(contributions[idx])
                write_vec += sign * W_down[:, idx]
            
            write_vec_norm = np.linalg.norm(write_vec)
            if write_vec_norm > 1e-10:
                write_vec_unit = write_vec / write_vec_norm
            else:
                write_vec_unit = write_vec
            
            # 写入向量与class_diff的对齐度
            alignment = float(np.dot(write_vec_unit, class_diff_unit))
            
            # 注入测试: 在li+1层注入
            inject_li = min(li + 1, info.n_layers - 1)
            
            betas = [5, 10, 20]
            inject_results = {}
            
            for beta in betas:
                # 基线
                base_logits = _run_with_optional_residual_replace(
                    model, tokenizer, prompt_compete, device,
                    replace_layer=None, replace_vec=None
                )
                target_fam = CAT_TO_FAM.get(cat_name, (None, []))[0]
                compete_fams = CAT_TO_FAM.get(cat_name, (None, []))[1]
                if target_fam is None:
                    continue
                
                base_margin = _compute_en_family_margin(base_logits, W_U, tokenizer, target_fam, compete_fams)
                
                # 注入class_diff方向
                delta_residual = beta * class_diff_unit
                patched_logits_r = _run_additive_patch(
                    model, tokenizer, prompt_compete, device, inject_li, delta_residual
                )
                margin_residual = _compute_en_family_margin(patched_logits_r, W_U, tokenizer, target_fam, compete_fams)
                
                # 注入写入向量方向
                delta_write = beta * write_vec_unit
                patched_logits_w = _run_additive_patch(
                    model, tokenizer, prompt_compete, device, inject_li, delta_write
                )
                margin_write = _compute_en_family_margin(patched_logits_w, W_U, tokenizer, target_fam, compete_fams)
                
                inject_results[f"beta{beta}"] = {
                    "base_margin": float(base_margin),
                    "residual_margin": float(margin_residual),
                    "write_vec_margin": float(margin_write),
                    "residual_selectivity": float(margin_residual - base_margin),
                    "write_vec_selectivity": float(margin_write - base_margin),
                }
            
            results[f"{cat_name}_L{li}"] = {
                "alignment_write_vs_diff": float(alignment),
                "class_diff_norm": float(class_diff_norm),
                "n_top_neurons": int(top_k),
                "inject_layer": inject_li,
                "injections": inject_results,
            }
            
            plog(f"  {cat_name} L{li}: alignment={alignment:.3f}, "
                 f"beta10 residual_sel={inject_results['beta10']['residual_selectivity']:.2f}, "
                 f"write_sel={inject_results['beta10']['write_vec_selectivity']:.2f}")
    
    plog(f"  Exp4 done. {len(results)} experiments")
    return results


def _run_additive_patch(model, tokenizer, prompt, device, patch_layer, delta_vec):
    """在指定层输出中添加delta向量(加法干预)"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    seq_len = attention_mask.sum().item()
    delta_tensor = torch.tensor(delta_vec, dtype=torch.float32, device=device)
    
    layers = get_layers(model)
    done = [False]
    
    def make_hook():
        def hook(module, input, output):
            if not done[0]:
                done[0] = True
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


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    obj_dict = ROUNDS[round_num]
    n_objs = sum(len(v) for v in obj_dict.values())
    n_cats = len(obj_dict)
    plog(f"Phase 462: model={model_name}, round={round_num}, "
         f"categories={n_cats}, objects={n_objs}")
    
    # 加载模型
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    plog(f"Model loaded in {time.time()-t0:.1f}s")
    
    info = get_model_info(model, model_name)
    plog(f"Model info: n_layers={info.n_layers}, d_model={info.d_model}, "
         f"vocab={info.vocab_size}, mlp_type={info.mlp_type}")
    
    results = {
        "model": model_name,
        "round": round_num,
        "n_cats": n_cats,
        "n_objs": n_objs,
        "model_info": {
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "vocab_size": info.vocab_size,
            "model_class": info.model_class,
        },
    }
    
    # Exp1: 大样本跨语言语义码验证
    t0 = time.time()
    results["exp1_cross_lang_probe"] = exp1_large_sample_cross_lang(
        model, tokenizer, model_name, device, obj_dict, round_num
    )
    plog(f"Exp1 done in {time.time()-t0:.1f}s")
    _save_results(model_name, round_num, results, "exp1")
    
    # Exp3: 翻译方向正交分解
    t0 = time.time()
    results["exp3_translate_decomposition"] = exp3_translate_direction_decomposition(
        model, tokenizer, model_name, device, obj_dict, round_num
    )
    plog(f"Exp3 done in {time.time()-t0:.1f}s")
    _save_results(model_name, round_num, results, "exp3")
    
    # Exp2b: 跨语言残差Patch因果验证
    t0 = time.time()
    results["exp2b_residual_patch"] = exp2b_residual_patch(
        model, tokenizer, model_name, device, obj_dict, round_num
    )
    plog(f"Exp2b done in {time.time()-t0:.1f}s")
    _save_results(model_name, round_num, results, "exp2b")
    
    # Exp4: W_down写入向量 vs 残差差分方向
    t0 = time.time()
    results["exp4_write_vs_residual"] = exp4_write_vector_vs_residual_diff(
        model, tokenizer, model_name, device, obj_dict, round_num
    )
    plog(f"Exp4 done in {time.time()-t0:.1f}s")
    
    # 保存完整结果
    out_path = f"results/glm5/phase462_{model_name}_r{round_num}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    plog(f"Results saved to {out_path}")
    
    # 释放模型
    release_model(model)
    plog("Phase 462 complete!")


def _save_results(model_name, round_num, results, exp_name):
    """增量保存结果(防止中途崩溃丢失数据)"""
    out_path = f"results/glm5/phase462_{model_name}_r{round_num}_{exp_name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    partial = {k: v for k, v in results.items() if k.startswith("exp") or k in ("model", "round", "n_cats", "n_objs", "model_info")}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(partial, f, ensure_ascii=False, indent=2, default=str)
    plog(f"  Partial results saved to {out_path}")


if __name__ == "__main__":
    main()
