"""
Phase 458: 关系槽位纯化、否定作用域与多跳知识路径验证
======================================================
Exp1: 关系槽位纯化 — 多模板一致性测试
Exp2: 否定作用域分解 — 多模板否定对比
Exp3: 多跳知识路径验证 — 传递推理链
Exp4: DS7B路径分裂大样本层扫描 — 12对象/类
Exp5: has_part槽位修复 — 改进模板+部件候选族
Exp6: 候选族词表控制 — 单token/词频/bootstrap

用法: python tests/glm5/phase458_slot_purification_negation_multihop.py qwen3 1
      python tests/glm5/phase458_slot_purification_negation_multihop.py glm4 2
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
    "attr_part_mech": ["wheel", "blade", "handle", "engine", "gear", "axle", "lever", "spring", "bolt", "valve"],
    "attr_material": ["metal", "wood", "plastic", "steel", "iron", "rubber", "glass", "leather", "stone", "fabric"],
    "attr_function": ["move", "cut", "carry", "hold", "drive", "build", "eat", "grow", "protect", "transport"],
}

CAT_OBJ = {
    "fruit":   ["apple", "banana", "orange", "grape", "pear", "peach", "lemon", "mango",
                "cherry", "plum", "melon", "pineapple"],
    "animal":  ["dog", "cat", "horse", "lion", "bear", "rabbit", "cow", "tiger",
                "sheep", "wolf", "deer", "monkey"],
    "tool":    ["hammer", "knife", "wrench", "saw", "drill", "axe", "shovel", "scissors",
                "pliers", "needle", "brush", "chisel"],
    "vehicle": ["car", "bus", "bicycle", "truck", "train", "boat", "plane", "scooter",
                "motorcycle", "taxi", "van", "ship"],
}

CAT_FAM = {
    "fruit": {"target": "class_fruit", "compete": ["class_animal", "class_tool", "class_vehicle"]},
    "animal": {"target": "class_animal", "compete": ["class_fruit", "class_tool", "class_vehicle"]},
    "tool":   {"target": "class_tool", "compete": ["class_fruit", "class_animal", "class_vehicle"]},
    "vehicle":{"target": "class_vehicle", "compete": ["class_fruit", "class_animal", "class_tool"]},
}

# ==================== Exp1: 关系槽位多模板定义 ====================
RELATION_TEMPLATES = {
    "is_a": {
        "templates": [
            "The {obj} is a kind of",
            "The {obj} belongs to the category",
            "People classify the {obj} as",
            "The correct class for the {obj} is",
            "A {obj} is a type of",
            "The {obj} is a",
        ],
        "target_for": "dynamic",  # 根据对象类别动态确定
        "compete_for": "dynamic",
    },
    "has_color": {
        "templates": [
            "The color of the {obj} is",
            "A typical {obj} looks",
            "The {obj} is usually",
            "People describe the {obj}'s color as",
        ],
        "target_for": "attr_color",
        "compete_for": ["attr_part_bio", "attr_material", "attr_function"],
    },
    "has_part": {
        "templates": [
            "A common part of a {obj} is",
            "One component of a {obj} is",
            "The {obj} contains a",
            "A physical part of the {obj} is",
            "The {obj} has a",
        ],
        "target_for": "attr_part_bio",
        "compete_for": ["attr_color", "attr_material", "attr_function"],
    },
    "used_for": {
        "templates": [
            "A {obj} is used to",
            "The purpose of a {obj} is to",
            "People use a {obj} to",
            "The function of a {obj} is to",
        ],
        "target_for": "attr_function",
        "compete_for": ["attr_color", "attr_part_bio", "attr_material"],
    },
}

# ==================== Exp2: 否定作用域模板 ====================
NEGATION_TEMPLATES = {
    "simple_neg":    "The {obj} is not a",
    "explicit_alt":  "The {obj} is not an animal; it is a",
    "contrast_neg":  "The {obj} is not an animal but a",
    "scope_control":  "It is false that the {obj} is an animal. The {obj} is a",
    "double_neg":    "It is not false that the {obj} is a",
}

# ==================== Exp3: 多跳知识路径 ====================
MULTIHOP_PATHS = [
    {
        "name": "robin_bird_animal",
        "premises": ["A robin is a bird.", "A bird is an animal."],
        "query": "Therefore, a robin is a kind of",
        "intermediate": "class_animal",
        "final": "class_animal",
        "alt_intermediate": "class_fruit",
    },
    {
        "name": "hammer_tool_object",
        "premises": ["A hammer is a tool.", "A tool is an object."],
        "query": "Therefore, a hammer is a kind of",
        "intermediate": "class_tool",
        "final": "class_tool",  # "object"没有候选族, 用tool
        "alt_intermediate": "class_vehicle",
    },
    {
        "name": "salmon_fish_animal",
        "premises": ["A salmon is a fish.", "A fish is an animal."],
        "query": "Therefore, a salmon is a kind of",
        "intermediate": "class_animal",
        "final": "class_animal",
        "alt_intermediate": "class_fruit",
    },
    {
        "name": "car_vehicle_machine",
        "premises": ["A car is a vehicle.", "A vehicle is a machine."],
        "query": "Therefore, a car is a kind of",
        "intermediate": "class_vehicle",
        "final": "class_vehicle",
        "alt_intermediate": "class_tool",
    },
    # 单跳对照
    {
        "name": "robin_single",
        "premises": ["A robin is a bird."],
        "query": "Therefore, a robin is a kind of",
        "intermediate": "class_fruit",  # bird不是候选族, 看fruit/animal竞争
        "final": "class_animal",
        "alt_intermediate": "class_vehicle",
    },
    {
        "name": "apple_single",
        "premises": ["An apple is a fruit."],
        "query": "Therefore, an apple is a kind of",
        "intermediate": "class_fruit",
        "final": "class_fruit",
        "alt_intermediate": "class_animal",
    },
]

OBJ2CAT = {}
for cat, objs in CAT_OBJ.items():
    for o in objs:
        OBJ2CAT[o] = cat

ROUNDS = {
    1: {k: v[:4] for k, v in CAT_OBJ.items()},   # pilot: 4/类
    2: {k: v[:8] for k, v in CAT_OBJ.items()},   # main: 8/类
}

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

def zero_hook(pos):
    def h(m, inp, out):
        o = out[0].clone() if isinstance(out, tuple) else out.clone()
        o[0, pos] = 0.0
        return (o,) + out[1:] if isinstance(out, tuple) else o
    return h

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

def get_last_pos(tok, text):
    return len(tok.encode(text, add_special_tokens=False)) - 1

def get_logits(model, tok, text):
    dev = get_dev(model)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
    iid = inputs["input_ids"].to(dev)
    amask = inputs["attention_mask"].to(dev)
    with torch.no_grad():
        cl = model(input_ids=iid, attention_mask=amask).logits[0, -1].float().cpu().numpy()
    return cl

def run_ablation_full(model, tok, text, layer, fam_keys, fam_dict=None):
    """Run clean + zero_attn + zero_mlp, return family logits for each"""
    fd = fam_dict or FAM
    dev = get_dev(model)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
    iid = inputs["input_ids"].to(dev)
    amask = inputs["attention_mask"].to(dev)
    last_pos = iid.shape[1] - 1
    
    # Clean
    with torch.no_grad():
        cl = model(input_ids=iid, attention_mask=amask).logits[0, -1].float().cpu().numpy()
    cf = fam_logits(cl, tok, {k: fd[k] for k in fam_keys if k in fd})
    
    # Zero attn
    ha = layer.self_attn.register_forward_hook(zero_hook(last_pos)) if hasattr(layer, 'self_attn') else None
    with torch.no_grad():
        za = model(input_ids=iid, attention_mask=amask).logits[0, -1].float().cpu().numpy()
    if ha: ha.remove()
    zf = fam_logits(za, tok, {k: fd[k] for k in fam_keys if k in fd})
    
    # Zero MLP
    mm = getattr(layer, 'mlp', None) or getattr(layer, 'feed_forward', None)
    hm = mm.register_forward_hook(zero_hook(last_pos)) if mm else None
    with torch.no_grad():
        zm = model(input_ids=iid, attention_mask=amask).logits[0, -1].float().cpu().numpy()
    if hm: hm.remove()
    mf = fam_logits(zm, tok, {k: fd[k] for k in fam_keys if k in fd})
    
    return cl, cf, za, zf, zm, mf


# ==================== Exp1: 关系槽位纯化 ====================
def exp1_relation_slot_purification(model, tok, info, rnd=1):
    """
    对每个关系槽位使用多个模板, 测量模板间一致性.
    如果一致性高, 说明模型形成了模板无关的relation slot.
    """
    plog(f"\n{'='*60}\nExp1: Relation Slot Purification (Multi-Template Consistency)\n{'='*60}")
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    results = {}
    
    for rel_name, rel_info in RELATION_TEMPLATES.items():
        plog(f"  Relation: {rel_name} ({len(rel_info['templates'])} templates)")
        rel_r = {}
        
        for cat_name, obj_list in obj_set.items():
            cat_r = {}
            
            for obj in obj_list:
                obj_r = {}
                
                # 确定目标族和竞争族
                if rel_info["target_for"] == "dynamic":
                    ci = CAT_FAM[cat_name]
                    tc = ci["target"]
                    comp_fams = ci["compete"]
                else:
                    tc = rel_info["target_for"]
                    comp_fams = rel_info["compete_for"]
                
                for tmpl in rel_info["templates"]:
                    text = tmpl.format(obj=obj)
                    fam_keys = [tc] + comp_fams
                    
                    cl = get_logits(model, tok, text)
                    cf = fam_logits(cl, tok, {k: FAM[k] for k in fam_keys if k in FAM})
                    
                    target_logit = cf.get(tc, -999)
                    margin = round(target_logit - max(cf.get(c, -999) for c in comp_fams), 4)
                    
                    # family-local softmax margin
                    flse = family_lse(cl, tok, {k: FAM[k] for k in fam_keys if k in FAM})
                    fprobs = local_softmax(flse)
                    soft_margin = round(fprobs.get(tc, 0) - max(fprobs.get(c, 0) for c in comp_fams), 6)
                    
                    obj_r[tmpl] = {
                        "margin": margin,
                        "softmax_margin": soft_margin,
                        "target_logit": round(target_logit, 4),
                    }
                
                cat_r[obj] = obj_r
            
            # 计算模板间一致性
            # 对每个对象, 检查所有模板的margin方向是否一致
            tmpls = list(rel_info["templates"])
            consistencies = []
            for obj in obj_list:
                if obj not in cat_r: continue
                margins = [cat_r[obj][t]["margin"] for t in tmpls if t in cat_r[obj]]
                if len(margins) >= 2:
                    # 方向一致性: 是否所有margin同号
                    pos_count = sum(1 for m in margins if m > 0)
                    neg_count = sum(1 for m in margins if m < 0)
                    direction_consistency = max(pos_count, neg_count) / len(margins)
                    # 变异系数
                    mean_m = float(np.mean(margins))
                    std_m = float(np.std(margins))
                    cv = abs(std_m / mean_m) if abs(mean_m) > 0.01 else None
                    consistencies.append({
                        "direction_consistency": round(direction_consistency, 4),
                        "cv": round(cv, 4) if cv is not None else None,
                        "margins": margins,
                    })
            
            # 汇总跨对象一致性
            avg_dc = round(float(np.mean([c["direction_consistency"] for c in consistencies])), 4) if consistencies else None
            cat_r["_summary"] = {
                "avg_direction_consistency": avg_dc,
                "n_objects": len(obj_list),
                "n_templates": len(tmpls),
            }
            plog(f"    {cat_name}: direction_consist={avg_dc}")
            rel_r[cat_name] = cat_r
        
        results[rel_name] = rel_r
    
    return results


# ==================== Exp2: 否定作用域分解 ====================
def exp2_negation_scope_decomposition(model, tok, info, rnd=1):
    """
    多个否定模板对比, 判断DS7B否定反常是模板假象还是机制差异.
    """
    plog(f"\n{'='*60}\nExp2: Negation Scope Decomposition\n{'='*60}")
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    results = {}
    
    # 用animal类对象测试, 因为Phase 457发现DS7B对animal否定异常
    # 也测其他类做对照
    test_cats = ["animal", "fruit", "tool"]
    
    for cat_name in test_cats:
        if cat_name not in obj_set: continue
        ci = CAT_FAM[cat_name]
        tc = ci["target"]
        comp_fams = ci["compete"]
        cat_r = {}
        
        for obj in obj_set[cat_name]:
            obj_r = {}
            
            for neg_name, neg_tmpl in NEGATION_TEMPLATES.items():
                text = neg_tmpl.format(obj=obj)
                fam_keys = [tc] + comp_fams
                
                cl = get_logits(model, tok, text)
                cf = fam_logits(cl, tok, {k: FAM[k] for k in fam_keys if k in FAM})
                
                target_logit = cf.get(tc, -999)
                margin = round(target_logit - max(cf.get(c, -999) for c in comp_fams), 4)
                
                # 各竞争族logit
                all_fam = {k: v for k, v in cf.items()}
                
                obj_r[neg_name] = {
                    "margin": margin,
                    "target_logit": round(target_logit, 4),
                    "family_logits": all_fam,
                }
            
            # 对比: 肯定句基线
            text_aff = f"The {obj} is a"
            cl_aff = get_logits(model, tok, text_aff)
            cf_aff = fam_logits(cl_aff, tok, {k: FAM[k] for k in fam_keys if k in FAM})
            target_aff = cf_aff.get(tc, -999)
            margin_aff = round(target_aff - max(cf_aff.get(c, -999) for c in comp_fams), 4)
            
            obj_r["affirmative"] = {
                "margin": margin_aff,
                "target_logit": round(target_aff, 4),
                "family_logits": {k: v for k, v in cf_aff.items()},
            }
            
            cat_r[obj] = obj_r
        
        # 汇总: 每种否定模板的margin变化
        summary = {}
        for neg_name in list(NEGATION_TEMPLATES.keys()) + ["affirmative"]:
            margins = [cat_r[obj][neg_name]["margin"] for obj in obj_set[cat_name] if obj in cat_r]
            avg_m, std_m = avg_std(margins)
            summary[neg_name] = {"avg_margin": avg_m, "std": std_m}
        
        cat_r["_summary"] = summary
        plog(f"  {cat_name}: " + ", ".join(f"{k}={v['avg_margin']}" for k, v in summary.items()))
        results[cat_name] = cat_r
    
    return results


# ==================== Exp3: 多跳知识路径验证 ====================
def exp3_multihop_knowledge_path(model, tok, info, rnd=1):
    """
    验证模型是否编码多跳知识路径.
    如果破坏前提会破坏结论, 说明使用了多跳路径.
    """
    plog(f"\n{'='*60}\nExp3: Multi-Hop Knowledge Path Verification\n{'='*60}")
    layers = get_layers(model)
    n_layers = info.n_layers
    results = {}
    
    for path_def in MULTIHOP_PATHS:
        plog(f"  Path: {path_def['name']}")
        path_r = {}
        
        # 构造完整文本
        premises = path_def["premises"]
        query = path_def["query"]
        
        # 完整版本 (2跳)
        full_text = " ".join(premises) + " " + query
        # 只第一前提 (1跳)
        one_hop_text = premises[0] + " " + query if len(premises) > 1 else full_text
        # 无前提 (0跳)
        no_prem_text = query
        # 颠倒前提顺序
        reversed_premises = list(reversed(premises))
        reversed_text = " ".join(reversed_premises) + " " + query
        
        conditions = {
            "full_2hop": full_text,
            "1hop": one_hop_text,
            "0hop": no_prem_text,
            "reversed": reversed_text,
        }
        
        # 确定候选族
        final_fam = path_def["final"]
        # 用class族+attr族
        fam_keys = ["class_fruit", "class_animal", "class_tool", "class_vehicle"]
        
        for cond_name, cond_text in conditions.items():
            cl = get_logits(model, tok, cond_text)
            cf = fam_logits(cl, tok, {k: FAM[k] for k in fam_keys if k in FAM})
            
            # 目标族logit和margin
            target_logit = cf.get(final_fam, -999)
            all_class = ["class_fruit", "class_animal", "class_tool", "class_vehicle"]
            margin = round(target_logit - max(cf.get(c, -999) for c in all_class if c != final_fam), 4)
            
            # family-local softmax
            flse = family_lse(cl, tok, {k: FAM[k] for k in fam_keys if k in FAM})
            fprobs = local_softmax(flse)
            target_prob = fprobs.get(final_fam, 0)
            
            path_r[cond_name] = {
                "margin": margin,
                "target_logit": round(target_logit, 4),
                "target_softmax_prob": round(target_prob, 6),
                "family_logits": cf,
                "family_probs": fprobs,
            }
        
        # 关键指标: 2hop vs 1hop vs 0hop
        m_2hop = path_r["full_2hop"]["margin"]
        m_1hop = path_r["1hop"]["margin"] if "1hop" in path_r else None
        m_0hop = path_r["0hop"]["margin"]
        
        path_r["_analysis"] = {
            "2hop_margin": m_2hop,
            "1hop_margin": m_1hop,
            "0hop_margin": m_0hop,
            "2hop_vs_0hop": round(m_2hop - m_0hop, 4) if m_2hop is not None and m_0hop is not None else None,
            "2hop_vs_1hop": round(m_2hop - m_1hop, 4) if m_2hop is not None and m_1hop is not None else None,
        }
        
        plog(f"    2hop={m_2hop}, 1hop={m_1hop}, 0hop={m_0hop}, "
             f"2vs0={path_r['_analysis']['2hop_vs_0hop']}")
        results[path_def["name"]] = path_r
    
    # 层间消融: 对最重要的2-hop路径, 在关键层做attn/MLP消融
    plog(f"\n  Layer ablation for multi-hop paths...")
    layer_ablation = {}
    
    # 选择2个关键路径做层间消融
    key_paths = [p for p in MULTIHOP_PATHS if len(p["premises"]) >= 2][:2]
    sample_layers = sorted(set(
        list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]
    ))
    plog(f"  Ablation layers: {sample_layers}")
    
    for path_def in key_paths:
        full_text = " ".join(path_def["premises"]) + " " + path_def["query"]
        final_fam = path_def["final"]
        fam_keys = ["class_fruit", "class_animal", "class_tool", "class_vehicle"]
        path_abl = {}
        
        for li in sample_layers:
            layer = layers[li]
            try:
                cl, cf, za, zf, zm, mf = run_ablation_full(model, tok, full_text, layer, fam_keys)
                
                target_c = cf.get(final_fam, -999)
                margin_c = target_c - max(cf.get(c, -999) for c in fam_keys if c != final_fam)
                margin_za = zf.get(final_fam, -999) - max(zf.get(c, -999) for c in fam_keys if c != final_fam)
                margin_zm = mf.get(final_fam, -999) - max(mf.get(c, -999) for c in fam_keys if c != final_fam)
                
                path_abl[f"L{li}"] = {
                    "attn_effect": round(margin_c - margin_za, 4),
                    "mlp_effect": round(margin_c - margin_zm, 4),
                    "attn_type": eff_type(round(margin_c - margin_za, 4)),
                    "mlp_type": eff_type(round(margin_c - margin_zm, 4)),
                }
            except Exception as e:
                path_abl[f"L{li}"] = {"error": str(e)}
        
        layer_ablation[path_def["name"]] = path_abl
        plog(f"    {path_def['name']}: " + 
             ", ".join(f"L{k}={v.get('attn_type','?')}/{v.get('mlp_type','?')}" 
                      for k, v in path_abl.items() if "error" not in v))
    
    results["_layer_ablation"] = layer_ablation
    return results


# ==================== Exp4: DS7B路径分裂大样本层扫描 ====================
def exp4_path_split_large_sample(model, tok, info, rnd=1):
    """
    用12对象/类做全层扫描, 确认fruit/animal路径分裂是否稳定.
    """
    plog(f"\n{'='*60}\nExp4: Path Split Large-Sample Layer Scan\n{'='*60}")
    layers = get_layers(model)
    n_layers = info.n_layers
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    
    # 采样层: R1用8层, R2用12层
    if rnd == 1:
        step = max(1, n_layers // 8)
        test_layers = sorted(set(list(range(0, n_layers, step)) + [n_layers - 1]))
    else:
        step = max(1, n_layers // 12)
        test_layers = sorted(set(list(range(0, n_layers, step)) + list(range(max(0, n_layers - 4), n_layers))))
    
    plog(f"  Testing {len(test_layers)} layers with {sum(len(v) for v in obj_set.values())} objects")
    
    results = {}
    
    for li in test_layers:
        layer = layers[li]
        layer_r = {}
        
        for cat_name in ["fruit", "animal"]:
            ci = CAT_FAM[cat_name]
            tc = ci["target"]
            comp_fams = ci["compete"]
            cat_vals_a = []
            cat_vals_m = []
            
            for obj in obj_set[cat_name]:
                text = f"The {obj} is a"
                fam_keys = [tc] + comp_fams
                try:
                    cl, cf, za, zf, zm, mf = run_ablation_full(model, tok, text, layer, fam_keys)
                    
                    target_c = cf.get(tc, -999)
                    margin_c = target_c - max(cf.get(c, -999) for c in comp_fams)
                    margin_za = zf.get(tc, -999) - max(zf.get(c, -999) for c in comp_fams)
                    margin_zm = mf.get(tc, -999) - max(mf.get(c, -999) for c in comp_fams)
                    
                    cat_vals_a.append(round(margin_c - margin_za, 4))
                    cat_vals_m.append(round(margin_c - margin_zm, 4))
                except Exception as e:
                    plog(f"    L{li} {cat_name}/{obj}: error {e}")
            
            avg_a, std_a = avg_std(cat_vals_a)
            avg_m, std_m = avg_std(cat_vals_m)
            layer_r[cat_name] = {
                "attn_avg": avg_a, "attn_std": std_a,
                "mlp_avg": avg_m, "mlp_std": std_m,
                "attn_type": eff_type(avg_a), "mlp_type": eff_type(avg_m),
                "n_objects": len(cat_vals_a),
            }
        
        # fruit/animal路径是否相反
        fa = layer_r["fruit"]
        aa = layer_r["animal"]
        flipped_attn = (fa["attn_avg"] is not None and aa["attn_avg"] is not None and 
                       fa["attn_avg"] * aa["attn_avg"] < 0)
        flipped_mlp = (fa["mlp_avg"] is not None and aa["mlp_avg"] is not None and 
                      fa["mlp_avg"] * aa["mlp_avg"] < 0)
        flipped = flipped_attn or flipped_mlp
        
        layer_r["_flip"] = {
            "attn_flipped": flipped_attn,
            "mlp_flipped": flipped_mlp,
            "any_flipped": flipped,
        }
        
        plog(f"  L{li}: fruit=[a={fa['attn_type']},m={fa['mlp_type']}] "
             f"animal=[a={aa['attn_type']},m={aa['mlp_type']}] "
             f"{'FLIP!' if flipped else ''}")
        results[f"L{li}"] = layer_r
    
    # 汇总: flip比例
    flip_count_attn = sum(1 for v in results.values() if v.get("_flip", {}).get("attn_flipped"))
    flip_count_mlp = sum(1 for v in results.values() if v.get("_flip", {}).get("mlp_flipped"))
    total = len(results)
    results["_summary"] = {
        "attn_flip_ratio": f"{flip_count_attn}/{total}",
        "mlp_flip_ratio": f"{flip_count_mlp}/{total}",
        "total_layers": total,
    }
    plog(f"  Flip ratio: attn={flip_count_attn}/{total}, mlp={flip_count_mlp}/{total}")
    
    return results


# ==================== Exp5: has_part槽位修复 ====================
def exp5_has_part_repair(model, tok, info, rnd=1):
    """
    改进has_part模板和候选族, 测试部件知识是否可读出.
    """
    plog(f"\n{'='*60}\nExp5: has_part Slot Repair\n{'='*60}")
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    results = {}
    
    # 原始模板 vs 改进模板
    part_templates = {
        "original": "The {obj} has a",
        "component": "A common part of a {obj} is",
        "physical": "A physical part of the {obj} is",
        "contains": "The {obj} contains a",
        "body_part": "A body part of the {obj} is",
    }
    
    # 不同候选族: 生物部件 vs 机械部件
    part_families = {
        "bio_parts": ["seed", "leaf", "stem", "root", "skin", "bone", "leg", "wing", "tail", "heart"],
        "mech_parts": ["wheel", "blade", "handle", "engine", "gear", "axle", "lever", "spring", "bolt", "valve"],
        "generic_parts": ["piece", "section", "component", "element", "fragment", "portion", "segment", "unit", "part", "section"],
    }
    
    compete_fams = ["attr_color", "attr_material", "attr_function"]
    
    for tmpl_name, tmpl in part_templates.items():
        tmpl_r = {}
        
        for cat_name, obj_list in obj_set.items():
            cat_r = []
            
            # 根据类别选择部件候选族
            if cat_name in ["fruit", "animal"]:
                part_fam = "bio_parts"
            else:
                part_fam = "mech_parts"
            
            for obj in obj_list:
                text = tmpl.format(obj=obj)
                
                # 测试不同部件候选族
                obj_r = {}
                for pf_name, pf_words in part_families.items():
                    cl = get_logits(model, tok, text)
                    
                    # 部件族logit
                    pf_ids = [tok.encode(w, add_special_tokens=False)[0] for w in pf_words if tok.encode(w, add_special_tokens=False)]
                    pf_logit = float(np.mean(cl[pf_ids])) if pf_ids else None
                    
                    # 竞争族logit
                    comp_logits = {}
                    for cf_name in compete_fams:
                        cf_ids = [tok.encode(w, add_special_tokens=False)[0] for w in FAM[cf_name] if tok.encode(w, add_special_tokens=False)]
                        comp_logits[cf_name] = round(float(np.mean(cl[cf_ids])), 4) if cf_ids else None
                    
                    if pf_logit is not None:
                        max_comp = max(v for v in comp_logits.values() if v is not None) if any(v is not None for v in comp_logits.values()) else -999
                        margin = round(pf_logit - max_comp, 4)
                    else:
                        margin = None
                    
                    obj_r[pf_name] = {
                        "part_logit": round(pf_logit, 4) if pf_logit is not None else None,
                        "margin": margin,
                        "compete": comp_logits,
                    }
                
                cat_r.append({"obj": obj, **obj_r})
            
            # 汇总
            summary = {}
            for pf_name in part_families:
                margins = [v[pf_name]["margin"] for v in cat_r if v.get(pf_name, {}).get("margin") is not None]
                avg_m, std_m = avg_std(margins)
                summary[pf_name] = {"avg_margin": avg_m, "std": std_m}
            
            tmpl_r[cat_name] = {"objects": cat_r, "summary": summary}
            plog(f"  {tmpl_name}/{cat_name}: " + 
                 ", ".join(f"{k}={v['avg_margin']}" for k, v in summary.items()))
        
        results[tmpl_name] = tmpl_r
    
    return results


# ==================== Exp6: 候选族词表控制 ====================
def exp6_candidate_vocab_control(model, tok, info, rnd=1):
    """
    控制候选族词表: 单token, 词频匹配, bootstrap.
    排除词表统计伪影.
    """
    plog(f"\n{'='*60}\nExp6: Candidate Vocabulary Control\n{'='*60}")
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    results = {}
    
    # 只测is_a关系, 用类别族
    class_fams = {
        "class_fruit":   FAM["class_fruit"],
        "class_animal":  FAM["class_animal"],
        "class_tool":    FAM["class_tool"],
        "class_vehicle": FAM["class_vehicle"],
    }
    
    # 单token候选族: 只保留编码为单个token的词
    single_tok_fams = {}
    for fname, words in class_fams.items():
        single_words = [w for w in words if len(tok.encode(w, add_special_tokens=False)) == 1]
        if single_words:
            single_tok_fams[fname] = single_words
    
    plog(f"  Single-token families: { {k: v for k, v in single_tok_fams.items()} }")
    
    # Bootstrap: 随机子集重复测试
    n_bootstrap = 5
    rng = np.random.RandomState(42)
    
    for cat_name, obj_list in obj_set.items():
        ci = CAT_FAM[cat_name]
        tc = ci["target"]
        comp_fams = ci["compete"]
        cat_r = {}
        
        for obj in obj_list:
            text = f"The {obj} is a"
            
            # 1. 全候选族 (原始)
            cl = get_logits(model, tok, text)
            cf_full = fam_logits(cl, tok, class_fams)
            margin_full = round(cf_full.get(tc, -999) - max(cf_full.get(c, -999) for c in comp_fams), 4)
            
            # 2. 单token候选族
            cf_single = fam_logits(cl, tok, single_tok_fams)
            margin_single = round(cf_single.get(tc, -999) - max(cf_single.get(c, -999) for c in comp_fams if c in cf_single), 4)
            
            # 3. Bootstrap: 随机2个词/族
            bootstrap_margins = []
            for bi in range(n_bootstrap):
                boot_fams = {}
                for fname, words in class_fams.items():
                    if len(words) >= 2:
                        boot_fams[fname] = list(rng.choice(words, size=2, replace=False))
                    else:
                        boot_fams[fname] = words
                cf_boot = fam_logits(cl, tok, boot_fams)
                margin_boot = round(cf_boot.get(tc, -999) - max(cf_boot.get(c, -999) for c in comp_fams if c in cf_boot), 4)
                bootstrap_margins.append(margin_boot)
            
            avg_boot, std_boot = avg_std(bootstrap_margins)
            
            # 4. W_U norm 检查
            W_U = get_W_U(model, None)
            fam_norms = {}
            for fname, words in class_fams.items():
                norms = []
                for w in words:
                    ids = tok.encode(w, add_special_tokens=False)
                    if ids:
                        n = float(np.linalg.norm(W_U[ids[0]]))
                        norms.append(n)
                fam_norms[fname] = round(float(np.mean(norms)), 4) if norms else None
            
            cat_r[obj] = {
                "margin_full": margin_full,
                "margin_single": margin_single,
                "margin_bootstrap_avg": avg_boot,
                "margin_bootstrap_std": std_boot,
                "family_norms": fam_norms,
            }
        
        # 汇总
        margins_full = [v["margin_full"] for k, v in cat_r.items() if not k.startswith("_")]
        margins_single = [v["margin_single"] for k, v in cat_r.items() if not k.startswith("_")]
        margins_boot = [v["margin_bootstrap_avg"] for k, v in cat_r.items() if not k.startswith("_")]
        
        avg_f, _ = avg_std(margins_full)
        avg_s, _ = avg_std(margins_single)
        avg_b, _ = avg_std(margins_boot)
        
        # 一致性: 三种方法是否方向一致
        direction_consistent = (avg_f > 0 and avg_s > 0 and avg_b > 0) or \
                               (avg_f < 0 and avg_s < 0 and avg_b < 0)
        
        cat_r["_summary"] = {
            "avg_full": avg_f,
            "avg_single": avg_s,
            "avg_bootstrap": avg_b,
            "direction_consistent": direction_consistent,
        }
        plog(f"  {cat_name}: full={avg_f}, single={avg_s}, boot={avg_b}, consistent={direction_consistent}")
        results[cat_name] = cat_r
    
    return results


# ==================== 主函数 ====================
def main():
    if len(sys.argv) < 2:
        print("Usage: python phase458_slot_purification_negation_multihop.py <model> [round]")
        print("  model: qwen3 | glm4 | deepseek7b")
        print("  round: 1 (pilot) | 2 (main)")
        sys.exit(1)
    
    model_name = sys.argv[1]
    rnd = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        sys.exit(1)
    
    plog(f"Phase 458: {model_name}, round={rnd}")
    
    # 加载模型
    model, tok = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, {info.n_layers} layers, d={info.d_model}")
    
    results = {
        "model": model_name,
        "round": rnd,
        "model_info": {"class": info.model_class, "n_layers": info.n_layers, "d_model": info.d_model},
    }
    
    # 运行所有实验
    try:
        results["exp1_relation_slot_purification"] = exp1_relation_slot_purification(model, tok, info, rnd)
    except Exception as e:
        plog(f"Exp1 error: {e}")
        import traceback; traceback.print_exc()
        results["exp1_error"] = str(e)
    
    try:
        results["exp2_negation_scope_decomposition"] = exp2_negation_scope_decomposition(model, tok, info, rnd)
    except Exception as e:
        plog(f"Exp2 error: {e}")
        import traceback; traceback.print_exc()
        results["exp2_error"] = str(e)
    
    try:
        results["exp3_multihop_knowledge_path"] = exp3_multihop_knowledge_path(model, tok, info, rnd)
    except Exception as e:
        plog(f"Exp3 error: {e}")
        import traceback; traceback.print_exc()
        results["exp3_error"] = str(e)
    
    try:
        results["exp4_path_split_large_sample"] = exp4_path_split_large_sample(model, tok, info, rnd)
    except Exception as e:
        plog(f"Exp4 error: {e}")
        import traceback; traceback.print_exc()
        results["exp4_error"] = str(e)
    
    try:
        results["exp5_has_part_repair"] = exp5_has_part_repair(model, tok, info, rnd)
    except Exception as e:
        plog(f"Exp5 error: {e}")
        import traceback; traceback.print_exc()
        results["exp5_error"] = str(e)
    
    try:
        results["exp6_candidate_vocab_control"] = exp6_candidate_vocab_control(model, tok, info, rnd)
    except Exception as e:
        plog(f"Exp6 error: {e}")
        import traceback; traceback.print_exc()
        results["exp6_error"] = str(e)
    
    # 保存结果
    out_dir = "results/glm5"
    os.makedirs(out_dir, exist_ok=True)
    key = "deepseek7b" if model_name == "deepseek7b" else model_name
    out_file = os.path.join(out_dir, f"phase458_{key}_r{rnd}.json")
    
    # 清理不可序列化的内容
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results = make_serializable(results)
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    plog(f"Results saved to {out_file}")
    
    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    
    plog(f"Phase 458 {model_name} R{rnd} complete!")


if __name__ == "__main__":
    main()
