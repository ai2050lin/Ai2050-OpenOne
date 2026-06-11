"""
Phase 460: 核心语义编码成分分解与中间路径因果闭环
===================================================
从"候选族边际读出"推进到"编码本体恢复"。

Exp1: 对象编码成分分解 — 残差流方向分离(ObjectAnchor/ClassShared/PrivateFeature/RelationAccess)
Exp2: Shared/Private重组因果实验 — 激活替换证明可组合性
Exp3: 多跳中间节点Patch因果闭环 — 替换中间概念表示
Exp4: 否定算子层间轨迹定位 — 逐层追踪否定信号转换
Exp5: 语法角色绑定大样本 — 30+动词×4候选族
Exp6: 跨语言翻译重构初测 — 中英文语义不变量分离
Exp7: 人工编码合成预实验 — 组合方向注入测试

用法: python tests/glm5/phase460_semantic_code_recovery.py qwen3 1
      python tests/glm5/phase460_semantic_code_recovery.py glm4 2
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

# ==================== 候选族定义 ====================
FAM = {
    "class_fruit":   ["fruit", "produce", "crop", "harvest"],
    "class_animal":  ["animal", "creature", "beast", "mammal"],
    "class_tool":    ["tool", "implement", "instrument", "device"],
    "class_vehicle": ["vehicle", "transport", "conveyance", "automobile"],
    "attr_color":    ["red", "green", "yellow", "blue", "brown", "black", "white", "orange", "gray", "pink"],
    "attr_part_bio": ["seed", "leaf", "stem", "root", "skin", "bone", "leg", "wing", "tail", "heart"],
    "attr_part_mech":["wheel", "blade", "handle", "engine", "gear", "axle", "lever", "spring", "bolt", "valve"],
    "attr_function": ["move", "cut", "carry", "hold", "drive", "build", "eat", "grow", "protect", "transport"],
}

CAT_OBJ = {
    "fruit":   ["apple", "banana", "orange", "grape", "pear", "peach", "lemon", "mango"],
    "animal":  ["dog", "cat", "horse", "lion", "bear", "rabbit", "cow", "tiger"],
    "tool":    ["hammer", "knife", "wrench", "saw", "drill", "axe", "shovel", "scissors"],
    "vehicle": ["car", "bus", "bicycle", "truck", "train", "boat", "plane", "scooter"],
}

CAT_FAM = {
    "fruit":  {"target": "class_fruit",  "compete": ["class_animal", "class_tool", "class_vehicle"]},
    "animal": {"target": "class_animal", "compete": ["class_fruit", "class_tool", "class_vehicle"]},
    "tool":   {"target": "class_tool",   "compete": ["class_fruit", "class_animal", "class_vehicle"]},
    "vehicle":{"target": "class_vehicle","compete": ["class_fruit", "class_animal", "class_tool"]},
}

# 关系槽位模板
RELATION_TEMPLATES = {
    "is_a":      "The {obj} is a kind of",
    "has_color": "The color of a {obj} is",
    "has_part":  "A common part of a {obj} is",
    "used_for":  "A {obj} is commonly used for",
    "made_of":   "A {obj} is typically made of",
    "located":   "A {obj} is typically found in",
}

# 多条is_a模板(用于验证子槽位)
IS_A_VARIANTS = {
    "kind_of":   "The {obj} is a kind of",
    "type_of":   "The {obj} is a type of",
    "simple_is": "The {obj} is a",
    "category":  "The {obj} belongs to the category",
    "example":   "The {obj} is an example of a",
}

ROUNDS = {
    1: {k: v[:4] for k, v in CAT_OBJ.items()},   # pilot: 4/类
    2: {k: v[:6] for k, v in CAT_OBJ.items()},   # main: 6/类
}

# ==================== 语法角色绑定扩展数据 ====================
AGENT_PATIENT_EXTENDED = [
    # (agent, verb, expected_patient_class)
    ("dog", "chased", "class_animal"),
    ("cat", "chased", "class_animal"),
    ("boy", "ate", "class_fruit"),
    ("girl", "cut", "class_tool"),
    ("monkey", "rode", "class_vehicle"),
    ("farmer", "drove", "class_vehicle"),
    ("chef", "cut", "class_tool"),
    ("child", "ate", "class_fruit"),
    ("hunter", "shot", "class_animal"),
    ("worker", "lifted", "class_tool"),
    ("driver", "parked", "class_vehicle"),
    ("bird", "ate", "class_fruit"),
    ("man", "rode", "class_vehicle"),
    ("woman", "carried", "class_tool"),
    ("dog", "bit", "class_animal"),
    ("cat", "caught", "class_animal"),
    ("boy", "threw", "class_tool"),
    ("girl", "picked", "class_fruit"),
    ("man", "fixed", "class_vehicle"),
    ("woman", "peeled", "class_fruit"),
]

# 主动/被动对照
VOICE_PAIRS = [
    {"active": "The dog chased the", "passive": "The cat was chased by the"},
    {"active": "The boy ate the", "passive": "The fruit was eaten by the"},
    {"active": "The farmer drove the", "passive": "The truck was driven by the"},
    {"active": "The girl cut the", "passive": "The bread was cut by the"},
    {"active": "The man rode the", "passive": "The horse was ridden by the"},
]

# ==================== 跨语言数据 ====================
CROSS_LANG_SENTENCES = [
    {"en": "The apple is a fruit", "zh": "苹果是一种水果", "key": "apple_fruit"},
    {"en": "The dog is an animal", "zh": "狗是一种动物", "key": "dog_animal"},
    {"en": "The knife is a tool", "zh": "刀是一种工具", "key": "knife_tool"},
    {"en": "The car is a vehicle", "zh": "汽车是一种交通工具", "key": "car_vehicle"},
    {"en": "The apple is red", "zh": "苹果是红色的", "key": "apple_red"},
    {"en": "The dog chased the cat", "zh": "狗追猫", "key": "dog_chase_cat"},
]

# ==================== 多跳路径 ====================
MULTIHOP_PATHS = [
    {"name": "robin_bird_animal", "premises": ["A robin is a bird.", "A bird is an animal."],
     "query": "Therefore, a robin is a kind of", "final": "class_animal",
     "intermediate_word": "bird", "intermediate_class": "class_animal"},
    {"name": "salmon_fish_animal", "premises": ["A salmon is a fish.", "A fish is an animal."],
     "query": "Therefore, a salmon is a kind of", "final": "class_animal",
     "intermediate_word": "fish", "intermediate_class": "class_animal"},
    {"name": "robin_0hop", "premises": [],
     "query": "A robin is a kind of", "final": "class_animal",
     "intermediate_word": None, "intermediate_class": None},
]


# ==================== 模型加载 ====================
def load_model_auto(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog(f"Loading {model_name} (bf16 + auto + flash)...")
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for ai in ["flash_attention_2", "sdpa", "eager"]:
        try:
            m = AutoModelForCausalLM.from_pretrained(cfg["path"], torch_dtype=torch.bfloat16,
                    device_map="auto", trust_remote_code=True, local_files_only=True, attn_implementation=ai)
            plog(f"  attn={ai}")
            break
        except:
            continue
    m.eval()
    if hasattr(m, 'hf_device_map'):
        dm = m.hf_device_map
        ld = {}
        for k, v in dm.items():
            if k.startswith('model.layers.'):
                lid = k.split('.')[2]
                if lid not in ld: ld[lid] = str(v)
        plog(f"  {sum(1 for v in ld.values() if 'cuda' in str(v))} GPU + {sum(1 for v in ld.values() if 'cpu' in str(v))} CPU layers")
    return m, tok

def get_dev(model):
    try:
        return model.model.embed_tokens.weight.device
    except:
        try: return model.get_input_embeddings().weight.device
        except: return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==================== 工具函数 ====================
def fam_logit(logits_np, tok, words):
    ids = [tok.encode(w, add_special_tokens=False)[0] for w in words if tok.encode(w, add_special_tokens=False)]
    return float(np.mean(logits_np[ids])) if ids else None

def fam_logits(logits_np, tok, fam_dict=None):
    fd = fam_dict or FAM
    return {k: round(v, 4) for k, v in ((k, fam_logit(logits_np, tok, w)) for k, w in fd.items()) if v is not None}

def get_logits(model, tok, text):
    dev = get_dev(model)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    iid = inputs["input_ids"].to(dev)
    amask = inputs["attention_mask"].to(dev)
    with torch.no_grad():
        cl = model(input_ids=iid, attention_mask=amask).logits[0, -1].float().cpu().numpy()
    return cl

def avg_std(vals):
    c = [v for v in vals if v is not None]
    return (round(float(np.mean(c)), 4), round(float(np.std(c)), 4)) if c else (None, None)

def eff_type(v, th=0.1):
    return "PROMOTES" if v and v > th else ("SUPPRESSES" if v and v < -th else "NEUTRAL")

def family_lse(logits_np, tok, fam_dict):
    r = {}
    for fname, words in fam_dict.items():
        ids = [tok.encode(w, add_special_tokens=False)[0] for w in words if tok.encode(w, add_special_tokens=False)]
        if ids:
            fl = logits_np[ids]
            mx = np.max(fl)
            r[fname] = round(float(mx + np.log(np.sum(np.exp(fl - mx)))), 4)
    return r

def local_softmax(fam_lse):
    if not fam_lse: return {}
    vals = np.array(list(fam_lse.values()))
    mx = np.max(vals)
    ev = np.exp(vals - mx)
    tot = np.sum(ev)
    return {k: round(float(p), 6) for k, p in zip(fam_lse.keys(), ev / tot)}

def compute_margin(logits_np, tok, target_fam, compete_fams, fam_dict=None):
    fd = fam_dict or FAM
    target_logit = fam_logit(logits_np, tok, fd.get(target_fam, []))
    margin = round(target_logit - max(fam_logit(logits_np, tok, fd.get(c, [])) or -999 for c in compete_fams), 4) if target_logit else None
    return {"margin": margin, "target_logit": round(target_logit, 4) if target_logit else None}

def zero_hook(pos):
    def h(m, inp, out):
        o = out[0].clone() if isinstance(out, tuple) else out.clone()
        o[0, pos] = 0.0
        return (o,) + out[1:] if isinstance(out, tuple) else o
    return h


def extract_residual_stream(model, tok, text, sample_layers=None):
    """提取指定层的残差流(最后一个token位置)"""
    dev = get_dev(model)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    iid = inputs["input_ids"].to(dev)
    amask = inputs["attention_mask"].to(dev)
    
    layers = get_layers(model)
    n_layers = len(layers)
    if sample_layers is None:
        sample_layers = sorted(set(list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]))
    
    captured = {}
    hooks = []
    
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0][0, -1].detach().float().cpu().numpy()  # [d_model]
            else:
                captured[key] = output[0, -1].detach().float().cpu().numpy()
        return hook
    
    for li in sample_layers:
        hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
    
    with torch.no_grad():
        model(input_ids=iid, attention_mask=amask)
    
    for h in hooks:
        h.remove()
    
    return captured  # {f"L{li}": np.ndarray[d_model]}


def extract_mlp_attn_output(model, tok, text, sample_layers=None):
    """提取指定层的MLP和attention输出(最后一个token位置)"""
    dev = get_dev(model)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    iid = inputs["input_ids"].to(dev)
    amask = inputs["attention_mask"].to(dev)
    
    layers = get_layers(model)
    n_layers = len(layers)
    if sample_layers is None:
        sample_layers = sorted(set(list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]))
    
    captured = {}
    hooks = []
    
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0][0, -1].detach().float().cpu().numpy()
            else:
                captured[key] = output[0, -1].detach().float().cpu().numpy()
        return hook
    
    for li in sample_layers:
        layer = layers[li]
        # MLP output
        if hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(make_hook(f"L{li}_mlp")))
        # Self-attn output
        if hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(make_hook(f"L{li}_attn")))
    
    with torch.no_grad():
        model(input_ids=iid, attention_mask=amask)
    
    for h in hooks:
        h.remove()
    
    return captured  # {f"L{li}_mlp": np.ndarray[d_model], f"L{li}_attn": np.ndarray[d_model]}


# ==================== Exp1: 对象编码成分分解 ====================
def exp1_object_code_decomposition(model, tok, info, rnd=1):
    """
    核心实验: 分离对象编码中的不同成分.
    
    方法:
    1. 同对象不同关系 → 提取RelationAccessCode
    2. 同类别不同对象 → 提取ClassSharedCode vs PrivateFeatureCode
    3. 不同类别同关系 → 提取ObjectAnchorCode vs ClassSharedCode
    4. 同对象不同is_a变体 → 验证SlotSubTypeCode
    
    分析: 余弦相似度 + 方向差分解 + PCA投影
    """
    plog(f"\n{'='*60}\nExp1: Object Code Decomposition\n{'='*60}")
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    n_layers = info.n_layers
    sample_layers = sorted(set(list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]))
    
    results = {}
    
    # ---- Part 1: 同对象不同关系 → RelationAccessCode ----
    plog("  Part1: Same object, different relations → RelationAccessCode")
    test_objs = []
    for cat, objs in obj_set.items():
        test_objs.extend([(o, cat) for o in objs[:2]])  # 每类2个
    
    relation_streams = {}
    for obj, cat in test_objs:
        obj_streams = {}
        for rel_name, tmpl in RELATION_TEMPLATES.items():
            text = tmpl.format(obj=obj)
            streams = extract_residual_stream(model, tok, text, sample_layers)
            obj_streams[rel_name] = streams
            plog(f"    {obj}/{rel_name}: L0 norm={np.linalg.norm(streams.get('L0', [0])):.2f}")
        relation_streams[f"{cat}_{obj}"] = obj_streams
    
    # 分析: 同对象不同关系间的余弦相似度(逐层)
    relation_analysis = {}
    for obj_key, obj_streams in relation_streams.items():
        layer_cosines = {}
        for layer_key in sample_layers:
            lk = f"L{layer_key}"
            vectors = {}
            for rel_name, streams in obj_streams.items():
                if lk in streams:
                    vectors[rel_name] = streams[lk]
            
            if len(vectors) >= 2:
                rel_names = list(vectors.keys())
                cos_matrix = {}
                for i, rn1 in enumerate(rel_names):
                    for j, rn2 in enumerate(rel_names):
                        if i < j:
                            v1, v2 = vectors[rn1], vectors[rn2]
                            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                            cos = float(np.dot(v1, v2) / max(n1 * n2, 1e-10))
                            cos_matrix[f"{rn1}_vs_{rn2}"] = round(cos, 4)
                layer_cosines[lk] = cos_matrix
        
        # 找关系差异最大的层
        avg_diffs = {}
        for lk, cm in layer_cosines.items():
            avg_diffs[lk] = round(1 - np.mean(list(cm.values())), 4)
        
        best_layer = max(avg_diffs, key=avg_diffs.get) if avg_diffs else "L0"
        relation_analysis[obj_key] = {
            "layer_cosines_sample": {lk: v for lk, v in list(layer_cosines.items())[::2]},
            "avg_relation_diff_per_layer": avg_diffs,
            "best_relation_split_layer": best_layer,
        }
    
    results["relation_access_code"] = relation_analysis
    
    # ---- Part 2: 同类别不同对象 → ClassShared vs Private ----
    plog("  Part2: Same class, different objects → ClassSharedCode vs PrivateFeatureCode")
    class_streams = {}
    for cat, objs in obj_set.items():
        cat_streams = {}
        for obj in objs[:3]:  # 每类3个
            # 用is_a模板
            text = f"The {obj} is a kind of"
            streams = extract_residual_stream(model, tok, text, sample_layers)
            cat_streams[obj] = streams
        class_streams[cat] = cat_streams
    
    # 分析: 同类内对象间相似度 vs 跨类对象间相似度
    class_analysis = {}
    for cat, cat_streams in class_streams.items():
        objs = list(cat_streams.keys())
        if len(objs) < 2:
            continue
        
        layer_within_class = {}
        layer_across_class = {}
        
        for layer_key in sample_layers:
            lk = f"L{layer_key}"
            
            # 同类内相似度
            within_cos = []
            for i in range(len(objs)):
                for j in range(i+1, len(objs)):
                    v1 = cat_streams[objs[i]].get(lk)
                    v2 = cat_streams[objs[j]].get(lk)
                    if v1 is not None and v2 is not None:
                        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if n1 > 0 and n2 > 0:
                            within_cos.append(float(np.dot(v1, v2) / (n1 * n2)))
            
            layer_within_class[lk] = round(float(np.mean(within_cos)), 4) if within_cos else None
            
            # 跨类相似度(和第一个其他类对象比)
            other_cats = [c for c in class_streams if c != cat]
            across_cos = []
            if other_cats:
                other_cat = other_cats[0]
                other_objs = list(class_streams[other_cat].keys())
                for obj1 in objs[:1]:  # 只用第1个
                    for obj2 in other_objs[:1]:
                        v1 = cat_streams[obj1].get(lk)
                        v2 = class_streams[other_cat][obj2].get(lk)
                        if v1 is not None and v2 is not None:
                            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                            if n1 > 0 and n2 > 0:
                                across_cos.append(float(np.dot(v1, v2) / (n1 * n2)))
            
            layer_across_class[lk] = round(float(np.mean(across_cos)), 4) if across_cos else None
        
        # 找"类别分离"最强的层: within最高 + across最低
        separation_scores = {}
        for lk in layer_within_class:
            w = layer_within_class.get(lk, 0) or 0
            a = layer_across_class.get(lk, 0) or 0
            separation_scores[lk] = round(w - a, 4)  # within高, across低 → 分离好
        
        best_sep_layer = max(separation_scores, key=separation_scores.get) if separation_scores else "L0"
        
        class_analysis[cat] = {
            "within_class_cosine": layer_within_class,
            "across_class_cosine": layer_across_class,
            "separation_scores": separation_scores,
            "best_separation_layer": best_sep_layer,
        }
        plog(f"    {cat}: best_sep={best_sep_layer}, score={separation_scores.get(best_sep_layer, 0)}")
    
    results["class_shared_private"] = class_analysis
    
    # ---- Part 3: 同对象不同is_a变体 → SlotSubTypeCode ----
    plog("  Part3: Same object, different is_a variants → SlotSubTypeCode")
    slot_streams = {}
    for cat, objs in obj_set.items():
        obj = objs[0]  # 每类1个
        obj_streams = {}
        for var_name, tmpl in IS_A_VARIANTS.items():
            text = tmpl.format(obj=obj)
            streams = extract_residual_stream(model, tok, text, sample_layers)
            obj_streams[var_name] = streams
        slot_streams[f"{cat}_{obj}"] = obj_streams
    
    slot_analysis = {}
    for obj_key, obj_streams in slot_streams.items():
        layer_cosines = {}
        for layer_key in sample_layers:
            lk = f"L{layer_key}"
            vectors = {}
            for var_name, streams in obj_streams.items():
                if lk in streams:
                    vectors[var_name] = streams[lk]
            
            if len(vectors) >= 2:
                var_names = list(vectors.keys())
                cos_pairs = {}
                for i in range(len(var_names)):
                    for j in range(i+1, len(var_names)):
                        v1, v2 = vectors[var_names[i]], vectors[var_names[j]]
                        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        cos = float(np.dot(v1, v2) / max(n1 * n2, 1e-10))
                        cos_pairs[f"{var_names[i]}_vs_{var_names[j]}"] = round(cos, 4)
                layer_cosines[lk] = cos_pairs
        
        # 找"category"变体差异最大的层
        cat_diffs = {}
        for lk, pairs in layer_cosines.items():
            # category vs kind_of 的差异
            cat_vs_kind = pairs.get("category_vs_kind_of", pairs.get("kind_of_vs_category", 1.0))
            cat_diffs[lk] = round(1 - cat_vs_kind, 4)
        
        best_slot_layer = max(cat_diffs, key=cat_diffs.get) if cat_diffs else "L0"
        slot_analysis[obj_key] = {
            "layer_cosines_sample": {lk: v for lk, v in list(layer_cosines.items())[::2]},
            "category_vs_kindof_diff": cat_diffs,
            "best_slot_split_layer": best_slot_layer,
        }
    
    results["slot_subtype_code"] = slot_analysis
    
    # ---- Part 4: 编码成分方向分离(PCA) ----
    plog("  Part4: PCA decomposition of encoding directions")
    # 收集所有is_a模板的对象流(最后几层平均)
    all_vectors = []
    all_labels = []
    
    for cat, objs in obj_set.items():
        for obj in objs[:3]:
            text = f"The {obj} is a kind of"
            streams = extract_residual_stream(model, tok, text, sample_layers[-3:])  # 最后3层
            for lk, vec in streams.items():
                all_vectors.append(vec)
                all_labels.append({"cat": cat, "obj": obj, "layer": lk})
    
    if all_vectors:
        X = np.array(all_vectors)  # [n, d_model]
        # PCA
        X_centered = X - X.mean(axis=0)
        cov = np.cov(X_centered.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # 取top-10
            top_idx = np.argsort(eigenvalues)[-10:][::-1]
            top_eigenvalues = eigenvalues[top_idx]
            top_eigenvectors = eigenvectors[:, top_idx]
            
            # 投影
            projections = X_centered @ top_eigenvectors  # [n, 10]
            
            # 分析: 哪个PC对应类别? 哪个PC对应对象?
            pc_class_correlation = {}
            for pc_idx in range(min(10, projections.shape[1])):
                proj_vals = projections[:, pc_idx]
                # 类别one-hot
                cats = list(set(l["cat"] for l in all_labels))
                cat_to_idx = {c: i for i, c in enumerate(cats)}
                cat_onehot = np.array([cat_to_idx[l["cat"]] for l in all_labels])
                # 简单ANOVA: 类间方差/总方差
                total_var = np.var(proj_vals)
                if total_var < 1e-10:
                    pc_class_correlation[f"PC{pc_idx}"] = 0
                    continue
                between_var = sum(len([l for l in all_labels if l["cat"] == c]) * 
                    (np.mean(proj_vals[[i for i, l in enumerate(all_labels) if l["cat"] == c]]) - np.mean(proj_vals))**2
                    for c in cats) / len(proj_vals)
                pc_class_correlation[f"PC{pc_idx}"] = round(float(between_var / total_var), 4)
            
            results["pca_decomposition"] = {
                "top_eigenvalues": [round(float(e), 4) for e in top_eigenvalues],
                "pc_class_correlation": pc_class_correlation,
                "variance_explained": [round(float(e / sum(top_eigenvalues)), 4) for e in top_eigenvalues],
                "n_samples": len(all_vectors),
            }
            plog(f"    PCA: top-3 eigenvalues={top_eigenvalues[:3].tolist()}, class_corr={list(pc_class_correlation.values())[:3]}")
        except Exception as e:
            plog(f"    PCA failed: {e}")
            results["pca_decomposition"] = {"error": str(e)}
    
    plog("Exp1 done")
    return results


# ==================== Exp2: Shared/Private重组因果实验 ====================
def exp2_shared_private_recombination(model, tok, info, rnd=1):
    """
    核心实验: 证明shared/private是可重组编码.
    
    方法:
    1. 提取fruit对象的残差流(apple, orange) → 提取fruit shared方向
    2. 提取tool对象的残差流(knife, hammer) → 提取tool shared方向
    3. 在apple的残差流中, 用方向替换把fruit shared → tool shared
    4. 看输出是否从fruit转向tool
    
    替换策略: 用余弦差方向注入
    - fruit_shared ≈ mean(apple_stream, orange_stream) - mean(knife_stream, hammer_stream) 的shared成分
    - 简化: 直接用类别均值差作为shared方向
    """
    plog(f"\n{'='*60}\nExp2: Shared/Private Recombination\n{'='*60}")
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    n_layers = info.n_layers
    # 只在关键层做替换: 中层(1/3和2/3处)
    key_layers = sorted(set([n_layers // 3, 2 * n_layers // 3, n_layers - 1]))
    
    results = {}
    
    # 收集类别均值方向
    plog("  Collecting class mean directions...")
    class_means = {}
    for cat in ["fruit", "animal", "tool", "vehicle"]:
        objs = obj_set[cat][:3]
        cat_streams = []
        for obj in objs:
            text = f"The {obj} is a kind of"
            streams = extract_residual_stream(model, tok, text, key_layers)
            cat_streams.append(streams)
        class_means[cat] = cat_streams
    
    # 计算类别shared方向: 类均值 - 全局均值
    plog("  Computing class shared directions...")
    all_mean_vecs = {}
    for lk_tag in [f"L{l}" for l in key_layers]:
        all_vecs = []
        for cat, cat_streams in class_means.items():
            for obj_streams in cat_streams:
                if lk_tag in obj_streams:
                    all_vecs.append(obj_streams[lk_tag])
        all_mean_vecs[lk_tag] = np.mean(all_vecs, axis=0) if all_vecs else None
    
    class_shared_dirs = {}
    for cat, cat_streams in class_means.items():
        cat_shared = {}
        for lk_tag in [f"L{l}" for l in key_layers]:
            cat_vecs = [s[lk_tag] for s in cat_streams if lk_tag in s]
            if cat_vecs and all_mean_vecs.get(lk_tag) is not None:
                cat_mean = np.mean(cat_vecs, axis=0)
                # shared方向 = 类均值 - 全局均值
                shared_dir = cat_mean - all_mean_vecs[lk_tag]
                norm = np.linalg.norm(shared_dir)
                if norm > 0:
                    shared_dir = shared_dir / norm
                cat_shared[lk_tag] = shared_dir
        class_shared_dirs[cat] = cat_shared
    
    # 替换实验: fruit→tool shared替换
    plog("  Running recombination experiments...")
    recomb_tests = [
        ("apple", "fruit", "tool"),   # 把apple的fruit shared换成tool shared
        ("dog", "animal", "vehicle"), # 把dog的animal shared换成vehicle shared
        ("knife", "tool", "fruit"),   # 把knife的tool shared换成fruit shared
    ]
    
    recomb_results = {}
    for obj_name, src_class, tgt_class in recomb_tests:
        plog(f"    Recombination: {obj_name} ({src_class}→{tgt_class})")
        
        # 基线: 原始对象在is_a下的logits
        base_text = f"The {obj_name} is a kind of"
        base_logits = get_logits(model, tok, base_text)
        base_fam = fam_logits(base_logits, tok, FAM)
        base_margin_fruit = compute_margin(base_logits, tok, CAT_FAM["fruit"]["target"], CAT_FAM["fruit"]["compete"])
        base_margin_tool = compute_margin(base_logits, tok, CAT_FAM["tool"]["target"], CAT_FAM["tool"]["compete"])
        
        # 在关键层注入shared方向差
        # 方法: 在残差流中添加 (tgt_shared - src_shared) * beta
        for layer_idx in key_layers:
            lk_tag = f"L{layer_idx}"
            
            src_dir = class_shared_dirs.get(src_class, {}).get(lk_tag)
            tgt_dir = class_shared_dirs.get(tgt_class, {}).get(lk_tag)
            
            if src_dir is None or tgt_dir is None:
                continue
            
            # 方向差: tool_shared - fruit_shared
            delta = tgt_dir - src_dir
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-10:
                continue
            delta = delta / delta_norm
            
            # 用不同beta注入
            for beta in [2.0, 5.0, 10.0]:
                dev = get_dev(model)
                inputs = tok(base_text, return_tensors="pt", truncation=True, max_length=256)
                iid = inputs["input_ids"].to(dev)
                amask = inputs["attention_mask"].to(dev)
                seq_len = iid.shape[1]
                last_pos = seq_len - 1
                
                # Hook: 在指定层添加方向
                captured = {}
                def make_inject_hook(delta_np, beta_val, pos):
                    delta_t = torch.tensor(delta_np, dtype=torch.bfloat16, device=dev)
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            o = output[0].clone()
                            o[0, pos, :] += (beta_val * delta_t).to(o.dtype)
                            return (o,) + output[1:]
                        else:
                            o = output.clone()
                            o[0, pos, :] += (beta_val * delta_t).to(o.dtype)
                            return o
                    return hook
                
                layers = get_layers(model)
                hook = layers[layer_idx].register_forward_hook(make_inject_hook(delta, beta, last_pos))
                
                with torch.no_grad():
                    inj_logits = model(input_ids=iid, attention_mask=amask).logits[0, -1].float().cpu().numpy()
                
                hook.remove()
                
                inj_fam = fam_logits(inj_logits, tok, FAM)
                inj_margin_fruit = compute_margin(inj_logits, tok, CAT_FAM["fruit"]["target"], CAT_FAM["fruit"]["compete"])
                inj_margin_tool = compute_margin(inj_logits, tok, CAT_FAM["tool"]["target"], CAT_FAM["tool"]["compete"])
                
                recomb_key = f"{obj_name}_{src_class}2{tgt_class}_L{layer_idx}_b{beta}"
                recomb_results[recomb_key] = {
                    "base_fam": base_fam,
                    "injected_fam": inj_fam,
                    "base_margin_fruit": base_margin_fruit.get("margin"),
                    "injected_margin_fruit": inj_margin_fruit.get("margin"),
                    "base_margin_tool": base_margin_tool.get("margin"),
                    "injected_margin_tool": inj_margin_tool.get("margin"),
                    "fruit_logit_change": round(inj_fam.get("class_fruit", 0) - base_fam.get("class_fruit", 0), 4),
                    "tool_logit_change": round(inj_fam.get("class_tool", 0) - base_fam.get("class_tool", 0), 4),
                }
                plog(f"      L{layer_idx} b={beta}: fruit_change={recomb_results[recomb_key]['fruit_logit_change']}, tool_change={recomb_results[recomb_key]['tool_logit_change']}")
    
    results["recombination"] = recomb_results
    plog("Exp2 done")
    return results


# ==================== Exp3: 多跳中间节点Patch因果闭环 ====================
def exp3_multihop_patch_causal(model, tok, info, rnd=1):
    """
    核心实验: 证明多跳推理真的依赖中间节点.
    
    方法:
    1. 提取2-hop路径中"bird"在中间层的表示
    2. Patch: 在2-hop推理时, 把bird位置替换成fish/zero
    3. 看最终animal margin是否变化
    
    这比Phase 459的"移除前提"更接近因果验证.
    """
    plog(f"\n{'='*60}\nExp3: Multi-Hop Patch Causal Closure\n{'='*60}")
    n_layers = info.n_layers
    
    results = {}
    
    for path in MULTIHOP_PATHS:
        pn = path["name"]
        plog(f"  Path: {pn}")
        pr = {}
        
        # 基线logits
        if path["premises"]:
            full_prompt = " ".join(path["premises"]) + " " + path["query"]
        else:
            full_prompt = path["query"]
        
        final_fam = path["final"]
        compete = [k for k in ["class_fruit", "class_animal", "class_tool", "class_vehicle"] if k != final_fam]
        
        base_logits = get_logits(model, tok, full_prompt)
        base_margin = compute_margin(base_logits, tok, final_fam, compete)
        pr["baseline"] = base_margin
        
        # 只对2-hop路径做patch
        if path["intermediate_word"] is None:
            results[pn] = pr
            continue
        
        # 中间词替换实验
        # 构建"替换中间概念"的prompt
        intermediate = path["intermediate_word"]
        premises = path["premises"]
        
        # 替换中间概念: "A robin is a bird" → "A robin is a fish"
        replace_words = ["fish", "tool", "fruit", "rock"]  # 不同替换词
        for rep_word in replace_words:
            # 替换第1个前提中的中间词
            modified_premise = premises[0].replace(intermediate, rep_word)
            if len(premises) > 1:
                mod_prompt = modified_premise + " " + premises[1] + " " + path["query"]
            else:
                mod_prompt = modified_premise + " " + path["query"]
            
            mod_logits = get_logits(model, tok, mod_prompt)
            mod_margin = compute_margin(mod_logits, tok, final_fam, compete)
            mod_all_fam = fam_logits(mod_logits, tok, FAM)
            
            pr[f"replace_{intermediate}_with_{rep_word}"] = {
                **mod_margin,
                "all_fam_logits": mod_all_fam,
                "margin_change": round((mod_margin.get("margin") or 0) - (base_margin.get("margin") or 0), 4),
            }
            plog(f"    replace {intermediate}→{rep_word}: margin_change={pr[f'replace_{intermediate}_with_{rep_word}']['margin_change']}")
        
        # 层间activation patch: 用残差流方向替换
        plog(f"    Running activation patch for {pn}...")
        sample_layers = sorted(set(list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]))
        
        # 提取2-hop和0-hop的残差流
        hop2_streams = extract_residual_stream(model, tok, full_prompt, sample_layers)
        hop0_prompt = path["query"]
        hop0_streams = extract_residual_stream(model, tok, hop0_prompt, sample_layers)
        
        # 在每层: 把2-hop的残差流方向patch到0-hop上
        # 方法: 在0-hop prompt中, 在指定层注入 (hop2_residual - hop0_residual) * alpha
        patch_results = {}
        for li in sample_layers:
            lk = f"L{li}"
            v2 = hop2_streams.get(lk)
            v0 = hop0_streams.get(lk)
            if v2 is None or v0 is None:
                continue
            
            delta = v2 - v0
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-10:
                continue
            delta_normed = delta / delta_norm
            
            for alpha in [0.5, 1.0, 2.0]:
                dev = get_dev(model)
                inputs = tok(hop0_prompt, return_tensors="pt", truncation=True, max_length=256)
                iid = inputs["input_ids"].to(dev)
                amask = inputs["attention_mask"].to(dev)
                last_pos = iid.shape[1] - 1
                
                layers = get_layers(model)
                
                def make_patch_hook(delta_np, alpha_val, pos):
                    delta_t = torch.tensor(delta_np * alpha_val, dtype=torch.bfloat16, device=dev)
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            o = output[0].clone()
                            o[0, pos, :] += delta_t.to(o.dtype)
                            return (o,) + output[1:]
                        else:
                            o = output.clone()
                            o[0, pos, :] += delta_t.to(o.dtype)
                            return o
                    return hook
                
                hook = layers[li].register_forward_hook(make_patch_hook(delta_normed, alpha, last_pos))
                
                with torch.no_grad():
                    patched_logits = model(input_ids=iid, attention_mask=amask).logits[0, -1].float().cpu().numpy()
                
                hook.remove()
                
                patched_margin = compute_margin(patched_logits, tok, final_fam, compete)
                patch_results[f"L{li}_alpha{alpha}"] = {
                    **patched_margin,
                    "margin_vs_0hop": round((patched_margin.get("margin") or 0) - (compute_margin(get_logits(model, tok, hop0_prompt), tok, final_fam, compete).get("margin") or 0), 4),
                }
        
        # 只保留关键结果
        pr["_activation_patch"] = {k: v for k, v in list(patch_results.items())[::3]}
        
        # 汇总
        m2 = pr.get("baseline", {}).get("margin", 0) or 0
        m0_result = compute_margin(get_logits(model, tok, path["query"]), tok, final_fam, compete)
        m0 = m0_result.get("margin", 0) or 0
        
        pr["_analysis"] = {
            "2hop_margin": m2,
            "0hop_margin": m0,
            "2hop_vs_0hop": round(m2 - m0, 4),
            "replace_effective": any(
                abs(pr.get(f"replace_{intermediate}_with_{rw}", {}).get("margin_change", 0)) > 0.5
                for rw in replace_words
            ),
        }
        
        results[pn] = pr
    
    plog("Exp3 done")
    return results


# ==================== Exp4: 否定算子层间轨迹定位 ====================
def exp4_negation_layer_tracing(model, tok, info, rnd=1):
    """
    逐层追踪否定信号: 在每层记录否定上下文的表示变化.
    找到DS7B把"not X"转成"释放非X"的层.
    """
    plog(f"\n{'='*60}\nExp4: Negation Layer Tracing\n{'='*60}")
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    n_layers = info.n_layers
    sample_layers = sorted(set(list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]))
    
    results = {}
    
    # 对每个类别: 比较 affirmative vs simple_neg 在每层的残差流
    negation_templates = {
        "affirmative": "The {obj} is a",
        "simple_neg": "The {obj} is not a",
        "contrast_neg": "The {obj} is not a {wrong}, but a",
        "double_neg": "It is not false that the {obj} is a",
    }
    
    for cat_name in ["fruit", "animal"]:
        objs = obj_set[cat_name][:3]
        ci = CAT_FAM[cat_name]
        tc = ci["target"]
        comp_fams = ci["compete"]
        wrong_class = comp_fams[0] if comp_fams else "class_tool"
        wrong_words = FAM.get(wrong_class, ["thing"])
        wrong_word = wrong_words[0] if wrong_words else "thing"
        
        cat_result = {}
        for obj in objs:
            obj_result = {}
            obj_streams = {}
            
            for neg_name, tmpl in negation_templates.items():
                if "{wrong}" in tmpl:
                    text = tmpl.format(obj=obj, wrong=wrong_word)
                else:
                    text = tmpl.format(obj=obj)
                
                streams = extract_residual_stream(model, tok, text, sample_layers)
                obj_streams[neg_name] = streams
                
                # 同时记录logits
                logits = get_logits(model, tok, text)
                obj_result[f"{neg_name}_logits"] = fam_logits(logits, tok, FAM)
            
            # 层间分析: 否定 vs 肯定的方向差
            layer_neg_deltas = {}
            for lk in [f"L{l}" for l in sample_layers]:
                aff_vec = obj_streams["affirmative"].get(lk)
                neg_vec = obj_streams["simple_neg"].get(lk)
                
                if aff_vec is not None and neg_vec is not None:
                    delta = neg_vec - aff_vec
                    delta_norm = np.linalg.norm(delta)
                    
                    # 方向差与W_U的投影: 看哪些token被提升/压制
                    W_U = get_W_U(model, model_name) if 'model_name' in dir() else None
                    
                    # 计算方向差在候选族方向的投影
                    fam_projections = {}
                    for fam_name, words in FAM.items():
                        fam_ids = [tok.encode(w, add_special_tokens=False)[0] for w in words if tok.encode(w, add_special_tokens=False)]
                        if fam_ids:
                            # delta在fam方向上的平均投影
                            fam_dirs = W_U[fam_ids] if W_U is not None else None
                            if fam_dirs is not None:
                                projections = fam_dirs @ delta / max(np.linalg.norm(delta), 1e-10)
                                fam_projections[fam_name] = round(float(np.mean(projections)), 4)
                    
                    layer_neg_deltas[lk] = {
                        "delta_norm": round(float(delta_norm), 4),
                        "fam_projections": fam_projections,
                        "cos_aff_neg": round(float(np.dot(aff_vec, neg_vec) / max(np.linalg.norm(aff_vec) * np.linalg.norm(neg_vec), 1e-10)), 4),
                    }
            
            obj_result["layer_neg_deltas"] = layer_neg_deltas
            cat_result[obj] = obj_result
        
        # 汇总: 找否定信号最强的层
        all_delta_norms = {}
        for obj, obj_result in cat_result.items():
            if not isinstance(obj_result, dict) or "layer_neg_deltas" not in obj_result:
                continue
            for lk, ld in obj_result["layer_neg_deltas"].items():
                if lk not in all_delta_norms:
                    all_delta_norms[lk] = []
                all_delta_norms[lk].append(ld.get("delta_norm", 0))
        
        avg_delta_norms = {lk: round(float(np.mean(v)), 4) for lk, v in all_delta_norms.items()}
        # 找否定信号峰值层
        peak_layer = max(avg_delta_norms, key=avg_delta_norms.get) if avg_delta_norms else "L0"
        
        cat_result["_summary"] = {
            "avg_neg_delta_norms": avg_delta_norms,
            "peak_negation_layer": peak_layer,
        }
        plog(f"  {cat_name}: peak_negation_layer={peak_layer}")
        
        results[cat_name] = cat_result
    
    plog("Exp4 done")
    return results


# ==================== Exp5: 语法角色绑定大样本 ====================
def exp5_syntax_role_extended(model, tok, info, rnd=1):
    """
    扩展语法角色绑定: 20+动词, 主动/被动对照.
    """
    plog(f"\n{'='*60}\nExp5: Syntax Role Binding Extended\n{'='*60}")
    results = {}
    
    # Part 1: 动词→patient候选族路由
    plog("  Part1: Verb→patient family routing")
    verb_patient = {}
    for agent, verb, expected_class in AGENT_PATIENT_EXTENDED:
        text = f"The {agent} {verb} the"
        logits = get_logits(model, tok, text)
        all_fam = fam_logits(logits, tok, FAM)
        
        # 判断: 最高logit的候选族是否匹配expected
        max_fam = max(all_fam, key=all_fam.get) if all_fam else "none"
        match = max_fam == expected_class
        
        verb_patient[f"{agent}_{verb}"] = {
            "all_fam_logits": all_fam,
            "expected_class": expected_class,
            "predicted_class": max_fam,
            "match": match,
        }
    
    # 汇总: 匹配率
    matches = [v["match"] for v in verb_patient.values()]
    match_rate = round(float(np.mean(matches)), 4) if matches else 0
    results["verb_patient_routing"] = {
        "pairs": verb_patient,
        "match_rate": match_rate,
        "n_pairs": len(verb_patient),
    }
    plog(f"  Verb→patient match rate: {match_rate} ({sum(matches)}/{len(matches)})")
    
    # Part 2: 主动/被动对照
    plog("  Part2: Active/Passive voice comparison")
    voice_results = {}
    for vp in VOICE_PAIRS:
        active_text = vp["active"]
        passive_text = vp["passive"]
        
        active_logits = get_logits(model, tok, active_text)
        passive_logits = get_logits(model, tok, passive_text)
        
        active_fam = fam_logits(active_logits, tok, FAM)
        passive_fam = fam_logits(passive_logits, tok, FAM)
        
        diff = {k: round(active_fam.get(k, 0) - passive_fam.get(k, 0), 4) for k in FAM.keys()}
        
        voice_results[f"{active_text[:20]}"] = {
            "active_fam": active_fam,
            "passive_fam": passive_fam,
            "difference": diff,
        }
        plog(f"    '{active_text}' vs '{passive_text}': diff={diff}")
    
    results["voice_comparison"] = voice_results
    
    # Part 3: 主宾交换 — 残差流层面
    plog("  Part3: Subject-Object swap residual stream")
    swap_pairs = [
        ("The dog chased the cat", "The cat chased the dog"),
        ("The boy ate the apple", "The apple was eaten by the boy"),
    ]
    
    swap_results = {}
    for active, passive in swap_pairs:
        n_layers = info.n_layers
        sample_layers = sorted(set(list(range(0, n_layers, max(1, n_layers // 4))) + [n_layers - 1]))
        
        active_streams = extract_residual_stream(model, tok, active, sample_layers)
        passive_streams = extract_residual_stream(model, tok, passive, sample_layers)
        
        layer_cosines = {}
        for lk in [f"L{l}" for l in sample_layers]:
            va = active_streams.get(lk)
            vp = passive_streams.get(lk)
            if va is not None and vp is not None:
                n1, n2 = np.linalg.norm(va), np.linalg.norm(vp)
                cos = float(np.dot(va, vp) / max(n1 * n2, 1e-10))
                layer_cosines[lk] = round(cos, 4)
        
        swap_results[f"{active[:20]}"] = {
            "layer_cosines": layer_cosines,
            "active_fam": fam_logits(get_logits(model, tok, active), tok, FAM),
            "passive_fam": fam_logits(get_logits(model, tok, passive), tok, FAM),
        }
    
    results["swap_residual"] = swap_results
    plog("Exp5 done")
    return results


# ==================== Exp6: 跨语言翻译重构初测 ====================
def exp6_cross_language_invariance(model, tok, info, rnd=1):
    """
    对比中英文同义句的内部表示: 寻找语义不变量.
    """
    plog(f"\n{'='*60}\nExp6: Cross-Language Semantic Invariance\n{'='*60}")
    n_layers = info.n_layers
    sample_layers = sorted(set(list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]))
    
    results = {}
    
    for sent in CROSS_LANG_SENTENCES:
        key = sent["key"]
        en_text = sent["en"]
        zh_text = sent["zh"]
        
        plog(f"  Testing: {key} (EN: {en_text[:30]}, ZH: {zh_text[:15]})")
        
        # 提取残差流
        en_streams = extract_residual_stream(model, tok, en_text, sample_layers)
        zh_streams = extract_residual_stream(model, tok, zh_text, sample_layers)
        
        # 层间余弦相似度
        layer_cosines = {}
        for lk in [f"L{l}" for l in sample_layers]:
            ve = en_streams.get(lk)
            vz = zh_streams.get(lk)
            if ve is not None and vz is not None:
                n1, n2 = np.linalg.norm(ve), np.linalg.norm(vz)
                cos = float(np.dot(ve, vz) / max(n1 * n2, 1e-10))
                layer_cosines[lk] = round(cos, 4)
        
        # Logits比较
        en_logits = get_logits(model, tok, en_text)
        zh_logits = get_logits(model, tok, zh_text)
        
        en_fam = fam_logits(en_logits, tok, FAM)
        zh_fam = fam_logits(zh_logits, tok, FAM)
        
        results[key] = {
            "en_text": en_text,
            "zh_text": zh_text,
            "layer_cosines": layer_cosines,
            "en_fam_logits": en_fam,
            "zh_fam_logits": zh_fam,
            "en_last_token": [tok.decode([i]).strip() for i in np.argsort(en_logits)[-5:][::-1]],
            "zh_last_token": [tok.decode([i]).strip() for i in np.argsort(zh_logits)[-5:][::-1]],
        }
        
        # 找语义最接近的层
        if layer_cosines:
            max_cos_layer = max(layer_cosines, key=layer_cosines.get)
            plog(f"    Best semantic invariance: {max_cos_layer} (cos={layer_cosines[max_cos_layer]})")
    
    # 汇总: 哪些层跨语言相似度最高
    all_layer_cos = {}
    for key, r in results.items():
        for lk, cos in r.get("layer_cosines", {}).items():
            all_layer_cos.setdefault(lk, []).append(cos)
    
    avg_lang_cos = {lk: round(float(np.mean(v)), 4) for lk, v in all_layer_cos.items()}
    best_invariance_layer = max(avg_lang_cos, key=avg_lang_cos.get) if avg_lang_cos else "L0"
    
    results["_summary"] = {
        "avg_cross_language_cosine": avg_lang_cos,
        "best_invariance_layer": best_invariance_layer,
    }
    plog(f"  Best invariance layer: {best_invariance_layer}")
    
    plog("Exp6 done")
    return results


# ==================== Exp7: 人工编码合成预实验 ====================
def exp7_artificial_code_synthesis(model, tok, info, rnd=1):
    """
    尝试人工组合内部编码: 注入类别方向+特征方向, 看是否产生预期输出.
    
    合成1: fruit_shared方向 + red_color方向 + is_a槽位 → apple/fruit
    合成2: tool_shared方向 + sharp_feature方向 + used_for槽位 → knife/cutting
    """
    plog(f"\n{'='*60}\nExp7: Artificial Code Synthesis\n{'='*60}")
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name) if 'model_name' in dir() else None
    
    results = {}
    
    # Step 1: 收集方向
    plog("  Step1: Collecting direction vectors...")
    
    # 类别方向: 从is_a模板的对象流中提取
    cat_dirs = {}
    for cat in ["fruit", "tool"]:
        objs = CAT_OBJ[cat][:3]
        cat_vecs = []
        for obj in objs:
            text = f"The {obj} is a kind of"
            streams = extract_residual_stream(model, tok, text, [n_layers - 1])
            lk = f"L{n_layers-1}"
            if lk in streams:
                cat_vecs.append(streams[lk])
        if cat_vecs:
            cat_mean = np.mean(cat_vecs, axis=0)
            cat_dirs[cat] = cat_mean
    
    # 颜色方向: W_U中red的行向量
    color_dirs = {}
    for color in ["red", "green", "blue"]:
        color_ids = tok.encode(color, add_special_tokens=False)
        if color_ids and W_U is not None:
            color_dirs[color] = W_U[color_ids[0]]
    
    # 特征方向
    feature_dirs = {}
    for feat in ["sharp", "soft", "round"]:
        feat_ids = tok.encode(feat, add_special_tokens=False)
        if feat_ids and W_U is not None:
            feature_dirs[feat] = W_U[feat_ids[0]]
    
    # Step 2: 合成实验
    plog("  Step2: Synthesis experiments...")
    
    # 中性基线prompt
    neutral_prompts = [
        "The object is a",
        "Something that is",
        "A thing that is a",
    ]
    
    synthesis_tests = [
        {
            "name": "fruit_red_is_a",
            "desc": "fruit_shared + red_color + is_a slot",
            "directions": [
                ("fruit_class", cat_dirs.get("fruit"), 5.0),
                ("red_color", color_dirs.get("red"), 3.0),
            ],
            "expected_top_families": ["class_fruit", "attr_color"],
        },
        {
            "name": "tool_sharp_used_for",
            "desc": "tool_shared + sharp_feature + used_for slot",
            "directions": [
                ("tool_class", cat_dirs.get("tool"), 5.0),
                ("sharp", feature_dirs.get("sharp"), 3.0),
            ],
            "expected_top_families": ["class_tool"],
        },
        {
            "name": "fruit_only",
            "desc": "fruit_shared only (no color/feature)",
            "directions": [
                ("fruit_class", cat_dirs.get("fruit"), 5.0),
            ],
            "expected_top_families": ["class_fruit"],
        },
        {
            "name": "tool_only",
            "desc": "tool_shared only",
            "directions": [
                ("tool_class", cat_dirs.get("tool"), 5.0),
            ],
            "expected_top_families": ["class_tool"],
        },
    ]
    
    for test in synthesis_tests:
        plog(f"    Testing: {test['name']} — {test['desc']}")
        
        for prompt in neutral_prompts[:1]:  # 只用1个基线prompt
            # 基线
            base_logits = get_logits(model, tok, prompt)
            base_fam = fam_logits(base_logits, tok, FAM)
            
            # 注入
            for layer_idx in [n_layers // 2, n_layers - 1]:
                dev = get_dev(model)
                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256)
                iid = inputs["input_ids"].to(dev)
                amask = inputs["attention_mask"].to(dev)
                last_pos = iid.shape[1] - 1
                
                # 合成方向
                combined_delta = np.zeros(info.d_model, dtype=np.float32)
                dir_info = []
                for dir_name, dir_vec, beta in test["directions"]:
                    if dir_vec is not None:
                        norm = np.linalg.norm(dir_vec)
                        if norm > 0:
                            combined_delta += beta * dir_vec / norm
                            dir_info.append(f"{dir_name}(b={beta})")
                
                if np.linalg.norm(combined_delta) < 1e-10:
                    continue
                
                # Hook
                layers = get_layers(model)
                
                def make_synthesis_hook(delta_np, pos):
                    delta_t = torch.tensor(delta_np, dtype=torch.bfloat16, device=dev)
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            o = output[0].clone()
                            o[0, pos, :] += delta_t.to(o.dtype)
                            return (o,) + output[1:]
                        else:
                            o = output.clone()
                            o[0, pos, :] += delta_t.to(o.dtype)
                            return o
                    return hook
                
                hook = layers[layer_idx].register_forward_hook(make_synthesis_hook(combined_delta, last_pos))
                
                with torch.no_grad():
                    synth_logits = model(input_ids=iid, attention_mask=amask).logits[0, -1].float().cpu().numpy()
                
                hook.remove()
                
                synth_fam = fam_logits(synth_logits, tok, FAM)
                top5_ids = np.argsort(synth_logits)[-5:][::-1]
                top5 = [(tok.decode([i]).strip(), round(float(synth_logits[i]), 4)) for i in top5_ids]
                
                # 检查期望的候选族是否提升
                expected_present = any(
                    synth_fam.get(ef, 0) > base_fam.get(ef, 0)
                    for ef in test["expected_top_families"]
                )
                
                synth_key = f"{test['name']}_L{layer_idx}"
                results[synth_key] = {
                    "desc": test["desc"],
                    "directions": dir_info,
                    "base_fam": base_fam,
                    "synth_fam": synth_fam,
                    "fam_changes": {k: round(synth_fam.get(k, 0) - base_fam.get(k, 0), 4) for k in FAM.keys()},
                    "top5": top5,
                    "expected_present": expected_present,
                }
                plog(f"      L{layer_idx}: fam_changes={results[synth_key]['fam_changes']}, expected_present={expected_present}")
    
    plog("Exp7 done")
    return results


# ==================== 主函数 ====================
def main():
    global model_name
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    rnd = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    plog(f"Phase 460: {model_name} R{rnd}")
    plog(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    model, tok = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, {info.n_layers}L, d={info.d_model}")
    
    all_results = {
        "model": model_name,
        "round": rnd,
        "model_info": {"class": info.model_class, "n_layers": info.n_layers, "d_model": info.d_model},
    }
    
    # Exp1: 对象编码成分分解
    try:
        r1 = exp1_object_code_decomposition(model, tok, info, rnd)
        all_results["exp1_code_decomposition"] = r1
    except Exception as e:
        plog(f"Exp1 error: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp2: Shared/Private重组
    try:
        r2 = exp2_shared_private_recombination(model, tok, info, rnd)
        all_results["exp2_recombination"] = r2
    except Exception as e:
        plog(f"Exp2 error: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp3: 多跳Patch因果
    try:
        r3 = exp3_multihop_patch_causal(model, tok, info, rnd)
        all_results["exp3_multihop_patch"] = r3
    except Exception as e:
        plog(f"Exp3 error: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp4: 否定层间追踪
    try:
        r4 = exp4_negation_layer_tracing(model, tok, info, rnd)
        all_results["exp4_negation_tracing"] = r4
    except Exception as e:
        plog(f"Exp4 error: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp5: 语法角色大样本
    try:
        r5 = exp5_syntax_role_extended(model, tok, info, rnd)
        all_results["exp5_syntax_extended"] = r5
    except Exception as e:
        plog(f"Exp5 error: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp6: 跨语言
    try:
        r6 = exp6_cross_language_invariance(model, tok, info, rnd)
        all_results["exp6_cross_language"] = r6
    except Exception as e:
        plog(f"Exp6 error: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp7: 人工编码合成
    try:
        r7 = exp7_artificial_code_synthesis(model, tok, info, rnd)
        all_results["exp7_code_synthesis"] = r7
    except Exception as e:
        plog(f"Exp7 error: {e}")
        import traceback; traceback.print_exc()
    
    # 保存
    os.makedirs("results/glm5", exist_ok=True)
    outf = f"results/glm5/phase460_{model_name}_r{rnd}.json"
    with open(outf, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    plog(f"Saved: {outf}")
    
    # 释放
    release_model(model)
    plog(f"Phase 460 {model_name} R{rnd} complete!")


if __name__ == "__main__":
    main()
