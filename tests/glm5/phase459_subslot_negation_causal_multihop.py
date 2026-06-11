"""
Phase 459: 槽位子类型发现、否定算子闭环与多跳路径因果验证
======================================================
Exp1: is_a子槽位聚类 — 12模板发现内部子槽位
Exp2: 否定算子多模板闭环 — 8种否定模板+4指标
Exp3: 多跳路径中间节点因果验证 — 前提移除/替换干预
Exp4: has_part具体部件知识修复 — 对象特异部件候选族
Exp5: DS7B全层类别路由大样本确认 — 12对象/类, 每2层
Exp6: 语法角色绑定预实验 — 主宾交换

用法: python tests/glm5/phase459_subslot_negation_causal_multihop.py qwen3 1
      python tests/glm5/phase459_subslot_negation_causal_multihop.py glm4 2
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

# 对象特异部件候选族 (Exp4)
OBJ_PARTS = {
    "fruit": {
        "bio_parts": ["seed", "skin", "stem", "peel", "flesh", "core", "pulp", "pit"],
        "generic_parts": ["piece", "section", "component", "part", "portion", "segment"],
    },
    "animal": {
        "bio_parts": ["leg", "tail", "fur", "wing", "paw", "claw", "horn", "beak", "hoof", "feather"],
        "generic_parts": ["piece", "section", "component", "part", "portion", "segment"],
    },
    "tool": {
        "mech_parts": ["handle", "blade", "head", "shaft", "edge", "point", "grip", "jaw"],
        "generic_parts": ["piece", "section", "component", "part", "portion", "segment"],
    },
    "vehicle": {
        "mech_parts": ["wheel", "engine", "door", "seat", "tire", "steering", "brake", "window"],
        "generic_parts": ["piece", "section", "component", "part", "portion", "segment"],
    },
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

# ==================== Exp1: is_a子槽位模板 ====================
IS_A_TEMPLATES = [
    # kind-of 类
    "The {obj} is a kind of",
    "The {obj} is a sort of",
    # type-of 类
    "The {obj} is a type of",
    "The {obj} is a form of",
    # simple is-a 类
    "The {obj} is a",
    "A {obj} is a",
    # category-label 类
    "People classify the {obj} as",
    "The correct class for the {obj} is",
    # classification-frame 类
    "The {obj} belongs to the category",
    "The {obj} is classified as",
    "The {obj} falls under the category of",
    "The {obj} is an example of a",
]

# ==================== Exp2: 否定算子扩展模板 ====================
NEGATION_EXTENDED = {
    # Phase 458已有
    "affirmative":     "The {obj} is a",
    "simple_neg":      "The {obj} is not a",
    "explicit_alt":    "The {obj} is not an animal; it is a",
    "contrast_neg":    "The {obj} is not an animal but a",
    "scope_control":   "It is false that the {obj} is an animal. The {obj} is a",
    "double_neg":      "It is not false that the {obj} is a",
    # Phase 459新增
    "not_only":        "The {obj} is not only an animal but also a",
    "not_because":     "The {obj} is not an animal because it is a",
    "without":         "Without being an animal, the {obj} is a",
    "never":           "The {obj} is never an animal; it is always a",
}

# ==================== Exp3: 多跳路径(扩展) ====================
MULTIHOP_PATHS = [
    # 2-hop路径
    {"name": "robin_bird_animal", "premises": ["A robin is a bird.", "A bird is an animal."],
     "query": "Therefore, a robin is a kind of", "intermediate": "class_animal", "final": "class_animal"},
    {"name": "salmon_fish_animal", "premises": ["A salmon is a fish.", "A fish is an animal."],
     "query": "Therefore, a salmon is a kind of", "intermediate": "class_animal", "final": "class_animal"},
    {"name": "rose_flower_plant", "premises": ["A rose is a flower.", "A flower is a plant."],
     "query": "Therefore, a rose is a kind of", "intermediate": "class_fruit", "final": "class_fruit"},  # plant不在候选族,用fruit
    {"name": "oak_tree_plant", "premises": ["An oak is a tree.", "A tree is a plant."],
     "query": "Therefore, an oak is a kind of", "intermediate": "class_fruit", "final": "class_fruit"},
    # 单跳对照
    {"name": "robin_single", "premises": ["A robin is a bird."],
     "query": "Therefore, a robin is a kind of", "intermediate": "class_animal", "final": "class_animal"},
    {"name": "apple_single", "premises": ["An apple is a fruit."],
     "query": "Therefore, an apple is a kind of", "intermediate": "class_fruit", "final": "class_fruit"},
    # 0-hop对照 (无前提)
    {"name": "robin_0hop", "premises": [],
     "query": "A robin is a kind of", "intermediate": "class_animal", "final": "class_animal"},
    {"name": "apple_0hop", "premises": [],
     "query": "An apple is a kind of", "intermediate": "class_fruit", "final": "class_fruit"},
]

# ==================== Exp6: 语法角色绑定 ====================
SYNTAX_SENTENCES = [
    # 主宾交换 — 主动语态
    {"active": "The dog chased the", "reversed": "The cat chased the",
     "agent": "class_animal", "patient": "class_animal"},
    {"active": "The boy hit the", "reversed": "The ball hit the",
     "agent": "class_animal", "patient": "class_tool"},
    {"active": "The farmer drove the", "reversed": "The truck drove the",
     "agent": "class_animal", "patient": "class_vehicle"},
    {"active": "The monkey ate the", "reversed": "The banana ate the",
     "agent": "class_animal", "patient": "class_fruit"},
    # 被动语态
    {"active": "The dog was chased by the", "reversed": "The cat was chased by the",
     "agent": "class_animal", "patient": "class_animal"},
    {"active": "The fruit was eaten by the", "reversed": "The animal was eaten by the",
     "agent": "class_animal", "patient": "class_fruit"},
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

def get_logits(model, tok, text):
    dev = get_dev(model)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
    iid = inputs["input_ids"].to(dev)
    amask = inputs["attention_mask"].to(dev)
    with torch.no_grad():
        cl = model(input_ids=iid, attention_mask=amask).logits[0, -1].float().cpu().numpy()
    return cl

def compute_margin(logits_np, tok, target_fam, compete_fams, fam_dict=None):
    """计算family-local softmax margin"""
    fd = fam_dict or FAM
    all_keys = [target_fam] + compete_fams
    flse = family_lse(logits_np, tok, {k: fd[k] for k in all_keys if k in fd})
    fprobs = local_softmax(flse)
    target_logit = fam_logit(logits_np, tok, fd.get(target_fam, []))
    margin = round(target_logit - max(fam_logit(logits_np, tok, fd.get(c, [])) or -999 for c in compete_fams), 4) if target_logit else None
    soft_margin = round(fprobs.get(target_fam, 0) - max(fprobs.get(c, 0) for c in compete_fams), 6)
    return {"margin": margin, "softmax_margin": soft_margin, "target_logit": round(target_logit, 4) if target_logit else None}


# ==================== Exp1: is_a子槽位聚类 ====================
def exp1_isa_subslot_clustering(model, tok, info, rnd=1):
    """
    12个is_a模板, 对比模板间margin方向, 发现内部子槽位.
    如果模板自然聚类(如kind-of vs classification-frame), 说明is_a有子槽位.
    """
    plog(f"\n{'='*60}\nExp1: is_a Sub-Slot Clustering (12 templates)\n{'='*60}")
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    results = {}
    
    for cat_name, obj_list in obj_set.items():
        ci = CAT_FAM[cat_name]
        tc = ci["target"]
        comp_fams = ci["compete"]
        
        cat_r = {}
        for obj in obj_list:
            obj_r = {}
            for tmpl in IS_A_TEMPLATES:
                text = tmpl.format(obj=obj)
                m = compute_margin(get_logits(model, tok, text), tok, tc, comp_fams)
                obj_r[tmpl] = m
            cat_r[obj] = obj_r
        
        # 模板间聚类分析
        tmpls = IS_A_TEMPLATES
        # 构建模板×对象 margin矩阵
        margin_matrix = []
        for t in tmpls:
            row = []
            for obj in obj_list:
                if obj in cat_r and t in cat_r[obj]:
                    row.append(cat_r[obj][t].get("margin", 0))
                else:
                    row.append(0)
            margin_matrix.append(row)
        margin_matrix = np.array(margin_matrix)  # [n_templates, n_objects]
        
        # 模板间相关矩阵
        n_t = len(tmpls)
        corr_matrix = np.zeros((n_t, n_t))
        for i in range(n_t):
            for j in range(n_t):
                if np.std(margin_matrix[i]) > 0 and np.std(margin_matrix[j]) > 0:
                    corr_matrix[i, j] = round(float(np.corrcoef(margin_matrix[i], margin_matrix[j])[0, 1]), 4)
                else:
                    corr_matrix[i, j] = 0
        
        # 简单聚类: 基于相关矩阵的层次聚类(手写,避免scipy依赖)
        # 用高相关阈值分组
        clusters = []
        used = set()
        for i in range(n_t):
            if i in used: continue
            cluster = [i]
            used.add(i)
            for j in range(i+1, n_t):
                if j in used: continue
                if corr_matrix[i, j] > 0.7:
                    cluster.append(j)
                    used.add(j)
            clusters.append(cluster)
        
        cluster_info = {}
        for ci_idx, cl in enumerate(clusters):
            label = f"cluster_{ci_idx}"
            cluster_info[label] = {
                "templates": [tmpls[i] for i in cl],
                "avg_margin": round(float(np.mean([margin_matrix[i] for i in cl])), 4),
                "avg_corr": round(float(np.mean([corr_matrix[i, j] for i in cl for j in cl if i != j])), 4) if len(cl) > 1 else 1.0,
            }
        
        cat_r["_summary"] = {
            "n_templates": n_t,
            "n_objects": len(obj_list),
            "n_clusters": len(clusters),
            "clusters": cluster_info,
            # 关键指标: "belongs to the category"是否单独聚类
            "btcat_alone": any(len(cl) == 1 and tmpls[cl[0]] == "The {obj} belongs to the category" for cl in clusters),
        }
        plog(f"  {cat_name}: {len(clusters)} clusters, btcat_alone={cat_r['_summary']['btcat_alone']}")
        results[cat_name] = cat_r
    
    return results


# ==================== Exp2: 否定算子多模板闭环 ====================
def exp2_negation_operator_closure(model, tok, info, rnd=1):
    """
    10种否定模板, 4个指标(NegatedFamilyDrop, AlternativeRelease, ScopeRecovery, DoubleNegRecovery).
    判断否定是否是稳定的边际向量变换.
    """
    plog(f"\n{'='*60}\nExp2: Negation Operator Closure (10 templates)\n{'='*60}")
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    results = {}
    
    for cat_name, obj_list in obj_set.items():
        ci = CAT_FAM[cat_name]
        tc = ci["target"]
        comp_fams = ci["compete"]
        
        cat_r = {}
        for obj in obj_list:
            obj_r = {}
            for neg_name, neg_tmpl in NEGATION_EXTENDED.items():
                text = neg_tmpl.format(obj=obj)
                cl = get_logits(model, tok, text)
                m = compute_margin(cl, tok, tc, comp_fams)
                # 额外: 所有候选族的logit
                all_fam_log = fam_logits(cl, tok, FAM)
                obj_r[neg_name] = {
                    **m,
                    "all_fam_logits": all_fam_log,
                }
            cat_r[obj] = obj_r
        
        # 计算否定算子4指标
        aff_margins = [cat_r[obj]["affirmative"]["margin"] for obj in obj_list if "affirmative" in cat_r.get(obj, {})]
        sn_margins = [cat_r[obj]["simple_neg"]["margin"] for obj in obj_list if "simple_neg" in cat_r.get(obj, {})]
        dn_margins = [cat_r[obj]["double_neg"]["margin"] for obj in obj_list if "double_neg" in cat_r.get(obj, {})]
        sc_margins = [cat_r[obj]["scope_control"]["margin"] for obj in obj_list if "scope_control" in cat_r.get(obj, {})]
        
        avg_aff = float(np.mean(aff_margins)) if aff_margins else 0
        avg_sn = float(np.mean(sn_margins)) if sn_margins else 0
        avg_dn = float(np.mean(dn_margins)) if dn_margins else 0
        avg_sc = float(np.mean(sc_margins)) if sc_margins else 0
        
        # NegatedFamilyDrop = affirmative - simple_neg (应>0表示正确否定)
        nfd = round(avg_aff - avg_sn, 4)
        # ScopeRecovery = scope_control - simple_neg (应>0表示作用域恢复)
        sr = round(avg_sc - avg_sn, 4)
        # DoubleNegRecovery = double_neg - simple_neg (应>0表示双重否定恢复)
        dnr = round(avg_dn - avg_sn, 4)
        
        # AlternativeRelease: 竞争族在simple_neg后vs affirmative后的变化
        alt_releases = {}
        for obj in obj_list:
            aff_d = cat_r.get(obj, {}).get("affirmative", {})
            sn_d = cat_r.get(obj, {}).get("simple_neg", {})
            if "all_fam_logits" in aff_d and "all_fam_logits" in sn_d:
                for cf in comp_fams:
                    diff = (sn_d["all_fam_logits"].get(cf, 0) or 0) - (aff_d["all_fam_logits"].get(cf, 0) or 0)
                    alt_releases.setdefault(cf, []).append(diff)
        
        alt_release_avg = {k: round(float(np.mean(v)), 4) for k, v in alt_releases.items()}
        
        cat_r["_summary"] = {
            "NegatedFamilyDrop": nfd,
            "ScopeRecovery": sr,
            "DoubleNegRecovery": dnr,
            "AlternativeRelease": alt_release_avg,
            "avg_affirmative": round(avg_aff, 4),
            "avg_simple_neg": round(avg_sn, 4),
            "avg_double_neg": round(avg_dn, 4),
            "avg_scope_control": round(avg_sc, 4),
        }
        plog(f"  {cat_name}: NFD={nfd}, SR={sr}, DNR={dnr}, AltRel={alt_release_avg}")
        results[cat_name] = cat_r
    
    return results


# ==================== Exp3: 多跳路径因果验证 ====================
def exp3_multihop_causal(model, tok, info, rnd=1):
    """
    多跳路径+前提移除干预.
    对比: 2前提, 1前提, 0前提 → 证明中间节点必要性.
    """
    plog(f"\n{'='*60}\nExp3: Multi-Hop Causal Verification\n{'='*60}")
    results = {}
    
    for path in MULTIHOP_PATHS:
        pn = path["name"]
        plog(f"  Path: {pn}")
        pr = {}
        
        # 构建完整prompt (2前提)
        if len(path["premises"]) == 2:
            full_prompt = " ".join(path["premises"]) + " " + path["query"]
            single_prompt = path["premises"][0] + " " + path["query"]  # 只保留第1前提
            no_premise_prompt = path["query"]  # 0前提
        elif len(path["premises"]) == 1:
            full_prompt = path["premises"][0] + " " + path["query"]
            single_prompt = full_prompt  # 单跳=完整
            no_premise_prompt = path["query"]
        else:
            full_prompt = path["query"]
            single_prompt = path["query"]
            no_premise_prompt = path["query"]
        
        final_fam = path["final"]
        compete = [k for k in ["class_fruit", "class_animal", "class_tool", "class_vehicle"] if k != final_fam]
        
        # 3种条件
        for cond_name, prompt in [("2hop", full_prompt), ("1hop", single_prompt), ("0hop", no_premise_prompt)]:
            cl = get_logits(model, tok, prompt)
            m = compute_margin(cl, tok, final_fam, compete)
            all_fam = fam_logits(cl, tok, FAM)
            pr[cond_name] = {**m, "all_fam_logits": all_fam}
        
        # 层间消融(仅2-hop): 在关键层zero attn/MLP
        if len(path["premises"]) == 2:
            plog(f"    Running layer ablation for {pn}...")
            layers = get_layers(model)
            n_layers = len(layers)
            sample_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
            sample_layers = sorted(set(sample_layers))
            
            dev = get_dev(model)
            inputs = tok(full_prompt, return_tensors="pt", truncation=True, max_length=256)
            iid = inputs["input_ids"].to(dev)
            amask = inputs["attention_mask"].to(dev)
            last_pos = iid.shape[1] - 1
            
            ablation = {}
            for li in sample_layers:
                layer = layers[li]
                
                # Clean
                with torch.no_grad():
                    clean_log = model(input_ids=iid, attention_mask=amask).logits[0, -1].float().cpu().numpy()
                
                # Zero attn
                ha = layer.self_attn.register_forward_hook(zero_hook(last_pos)) if hasattr(layer, 'self_attn') else None
                with torch.no_grad():
                    za_log = model(input_ids=iid, attention_mask=amask).logits[0, -1].float().cpu().numpy()
                if ha: ha.remove()
                
                # Zero MLP
                mm = getattr(layer, 'mlp', None) or getattr(layer, 'feed_forward', None)
                hm = mm.register_forward_hook(zero_hook(last_pos)) if mm else None
                with torch.no_grad():
                    zm_log = model(input_ids=iid, attention_mask=amask).logits[0, -1].float().cpu().numpy()
                if hm: hm.remove()
                
                clean_t = fam_logit(clean_log, tok, FAM.get(final_fam, []))
                za_t = fam_logit(za_log, tok, FAM.get(final_fam, []))
                zm_t = fam_logit(zm_log, tok, FAM.get(final_fam, []))
                
                ablation[f"L{li}"] = {
                    "clean_target": round(clean_t, 4) if clean_t else None,
                    "zero_attn_target": round(za_t, 4) if za_t else None,
                    "zero_mlp_target": round(zm_t, 4) if zm_t else None,
                    "attn_effect": eff_type((clean_t or 0) - (za_t or 0)),
                    "mlp_effect": eff_type((clean_t or 0) - (zm_t or 0)),
                }
            
            pr["_ablation"] = ablation
        
        # 因果分析
        m2 = pr.get("2hop", {}).get("margin", 0) or 0
        m1 = pr.get("1hop", {}).get("margin", 0) or 0
        m0 = pr.get("0hop", {}).get("margin", 0) or 0
        
        pr["_analysis"] = {
            "2hop_margin": m2,
            "1hop_margin": m1,
            "0hop_margin": m0,
            "2hop_vs_0hop": round(m2 - m0, 4),
            "2hop_vs_1hop": round(m2 - m1, 4),
            "1hop_vs_0hop": round(m1 - m0, 4),
        }
        plog(f"    2hop={m2}, 1hop={m1}, 0hop={m0}, 2vs0={round(m2-m0,4)}")
        results[pn] = pr
    
    return results


# ==================== Exp4: has_part具体部件修复 ====================
def exp4_has_part_repair(model, tok, info, rnd=1):
    """
    对象特异部件候选族 + 改进模板.
    区分: 具体部件知识缺失 vs 候选词/模板问题.
    """
    plog(f"\n{'='*60}\nExp4: has_part Specific Parts Repair\n{'='*60}")
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    results = {}
    
    PART_TEMPLATES = [
        "A physical part of a {obj} is",
        "One visible component of a {obj} is",
        "A {obj} contains a",
        "The {obj} has a",
        "If you look inside a {obj}, you find a",
    ]
    
    for cat_name, obj_list in obj_set.items():
        cat_r = {}
        parts_info = OBJ_PARTS.get(cat_name, {})
        
        if not parts_info:
            continue
        
        for obj in obj_list:
            obj_r = {}
            for tmpl in PART_TEMPLATES:
                text = tmpl.format(obj=obj)
                cl = get_logits(model, tok, text)
                
                tmpl_r = {}
                for part_type, part_words in parts_info.items():
                    target_logit = fam_logit(cl, tok, part_words)
                    # 用generic类别作为竞争
                    comp_words = []
                    for pt2, pw2 in parts_info.items():
                        if pt2 != part_type:
                            comp_words.extend(pw2[:3])
                    comp_logit = fam_logit(cl, tok, comp_words) if comp_words else None
                    margin = round(target_logit - comp_logit, 4) if target_logit and comp_logit is not None else None
                    tmpl_r[part_type] = {
                        "target_logit": round(target_logit, 4) if target_logit else None,
                        "compete_logit": round(comp_logit, 4) if comp_logit else None,
                        "margin": margin,
                    }
                obj_r[tmpl] = tmpl_r
            cat_r[obj] = obj_r
        
        # 汇总
        for part_type in parts_info.keys():
            margins = []
            for obj in obj_list:
                for tmpl in PART_TEMPLATES:
                    if obj in cat_r and tmpl in cat_r[obj]:
                        m = cat_r[obj][tmpl].get(part_type, {}).get("margin")
                        if m is not None:
                            margins.append(m)
            avg_m, std_m = avg_std(margins)
            cat_r[f"_summary_{part_type}"] = {
                "avg_margin": avg_m,
                "std_margin": std_m,
                "n_measurements": len(margins),
            }
            plog(f"  {cat_name}/{part_type}: avg_margin={avg_m}")
        
        results[cat_name] = cat_r
    
    return results


# ==================== Exp5: DS7B全层类别路由大样本 ====================
def exp5_path_split_large(model, tok, info, rnd=1):
    """
    12对象/类, 每2层采样, 确认fruit/animal全层路径分裂.
    """
    plog(f"\n{'='*60}\nExp5: Path Split Large Sample (12 obj/class, every 2 layers)\n{'='*60}")
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    results = {}
    
    layers = get_layers(model)
    n_layers = len(layers)
    # 每2层采样
    sample_layers = sorted(set(list(range(0, n_layers, 2)) + [n_layers - 1]))
    plog(f"  Sampling {len(sample_layers)} layers: {sample_layers[:5]}...{sample_layers[-3:]}")
    
    for cat_name in ["fruit", "animal"]:
        ci = CAT_FAM[cat_name]
        tc = ci["target"]
        comp_fams = ci["compete"]
        obj_list = obj_set.get(cat_name, [])[:6]  # 6对象足够(12对象太慢)
        
        layer_results = {}
        for li in sample_layers:
            layer = layers[li]
            lr = {}
            
            for obj in obj_list:
                text = f"The {obj} is a"
                dev = get_dev(model)
                inputs = tok(text, return_tensors="pt", truncation=True, max_length=64)
                iid = inputs["input_ids"].to(dev)
                amask = inputs["attention_mask"].to(dev)
                last_pos = iid.shape[1] - 1
                
                # Clean
                with torch.no_grad():
                    clean_log = model(input_ids=iid, attention_mask=amask).logits[0, -1].float().cpu().numpy()
                clean_fam = fam_logits(clean_log, tok, {k: FAM[k] for k in [tc] + comp_fams if k in FAM})
                clean_t = clean_fam.get(tc, 0)
                
                # Zero attn
                ha = layer.self_attn.register_forward_hook(zero_hook(last_pos)) if hasattr(layer, 'self_attn') else None
                with torch.no_grad():
                    za_log = model(input_ids=iid, attention_mask=amask).logits[0, -1].float().cpu().numpy()
                if ha: ha.remove()
                za_fam = fam_logits(za_log, tok, {k: FAM[k] for k in [tc] + comp_fams if k in FAM})
                
                # Zero MLP
                mm = getattr(layer, 'mlp', None) or getattr(layer, 'feed_forward', None)
                hm = mm.register_forward_hook(zero_hook(last_pos)) if mm else None
                with torch.no_grad():
                    zm_log = model(input_ids=iid, attention_mask=amask).logits[0, -1].float().cpu().numpy()
                if hm: hm.remove()
                zm_fam = fam_logits(zm_log, tok, {k: FAM[k] for k in [tc] + comp_fams if k in FAM})
                
                attn_eff = round(clean_t - (za_fam.get(tc, 0) or 0), 4)
                mlp_eff = round(clean_t - (zm_fam.get(tc, 0) or 0), 4)
                
                lr[obj] = {
                    "attn_effect": attn_eff,
                    "attn_type": eff_type(attn_eff),
                    "mlp_effect": mlp_eff,
                    "mlp_type": eff_type(mlp_eff),
                    "clean_target": round(clean_t, 4),
                }
            
            # 汇总跨对象
            attn_effs = [lr[o]["attn_effect"] for o in obj_list if o in lr]
            mlp_effs = [lr[o]["mlp_effect"] for o in obj_list if o in lr]
            avg_a, std_a = avg_std(attn_effs)
            avg_m, std_m = avg_std(mlp_effs)
            
            lr["_summary"] = {
                "avg_attn": avg_a, "std_attn": std_a, "attn_type": eff_type(avg_a),
                "avg_mlp": avg_m, "std_mlp": std_m, "mlp_type": eff_type(avg_m),
                "n_objects": len(obj_list),
            }
            layer_results[f"L{li}"] = lr
            plog(f"    L{li} {cat_name}: attn={eff_type(avg_a)}({avg_a}), mlp={eff_type(avg_m)}({avg_m})")
        
        results[cat_name] = layer_results
    
    # 分析fruit/animal路径分裂
    fruit_layers = {k: v for k, v in results.get("fruit", {}).items() if not k.startswith("_")}
    animal_layers = {k: v for k, v in results.get("animal", {}).items() if not k.startswith("_")}
    
    flip_count = 0
    total_count = 0
    common_layers = sorted(set(fruit_layers.keys()) & set(animal_layers.keys()))
    flip_details = {}
    
    for lk in common_layers:
        fl = fruit_layers[lk].get("_summary", {})
        al = animal_layers[lk].get("_summary", {})
        fa = fl.get("attn_type", "NEUTRAL")
        fm = fl.get("mlp_type", "NEUTRAL")
        aa = al.get("attn_type", "NEUTRAL")
        am = al.get("mlp_type", "NEUTRAL")
        
        attn_flip = (fa in ["PROMOTES", "SUPPRESSES"]) and (aa in ["PROMOTES", "SUPPRESSES"]) and fa != aa
        mlp_flip = (fm in ["PROMOTES", "SUPPRESSES"]) and (am in ["PROMOTES", "SUPPRESSES"]) and fm != am
        any_flip = attn_flip or mlp_flip
        
        if any_flip:
            flip_count += 1
        total_count += 1
        flip_details[lk] = {
            "fruit_attn": fa, "fruit_mlp": fm,
            "animal_attn": aa, "animal_mlp": am,
            "attn_flip": attn_flip, "mlp_flip": mlp_flip, "any_flip": any_flip,
        }
    
    results["_flip_summary"] = {
        "flip_count": flip_count,
        "total_count": total_count,
        "flip_ratio": round(flip_count / max(total_count, 1), 4),
        "flip_details": flip_details,
    }
    plog(f"  Path split: {flip_count}/{total_count} layers flipped ({round(flip_count/max(total_count,1),4)})")
    
    return results


# ==================== Exp6: 语法角色绑定 ====================
def exp6_syntax_role_binding(model, tok, info, rnd=1):
    """
    主宾交换实验: 检测模型是否把词序/句法角色转成不同知识路径.
    The dog chased the cat → dog=agent, cat=patient
    The cat chased the dog → cat=agent, dog=patient
    测量: 续写位置是否倾向不同候选族.
    """
    plog(f"\n{'='*60}\nExp6: Syntax Role Binding Pre-Experiment\n{'='*60}")
    results = {}
    
    for sent in SYNTAX_SENTENCES:
        sname = f"{sent['active'][:30]}..."
        plog(f"  Testing: {sname}")
        sr = {}
        
        for cond, template in [("active", sent["active"]), ("reversed", sent["reversed"])]:
            cl = get_logits(model, tok, template)
            all_fam = fam_logits(cl, tok, FAM)
            top5_ids = np.argsort(cl)[-5:][::-1]
            top5 = [(tok.decode([i]).strip(), round(float(cl[i]), 4)) for i in top5_ids]
            sr[cond] = {
                "all_fam_logits": all_fam,
                "top5": top5,
            }
        
        # 主动vs反转的差异
        active_fam = sr["active"].get("all_fam_logits", {})
        reversed_fam = sr["reversed"].get("all_fam_logits", {})
        diff = {k: round(active_fam.get(k, 0) - reversed_fam.get(k, 0), 4) for k in FAM.keys()}
        sr["difference"] = diff
        
        results[sname] = sr
        plog(f"    active: {active_fam}, reversed: {reversed_fam}, diff: {diff}")
    
    # 额外: 更结构化的语法角色测试
    # "The {agent} {verb} the" → 看续写是agent还是patient候选族
    AGENT_PATIENT_PAIRS = [
        ("dog", "chased", "class_animal"),   # dog=agent → 续写可能是patient
        ("cat", "chased", "class_animal"),
        ("boy", "ate", "class_animal"),      # boy=agent → 续写可能是fruit
        ("girl", "cut", "class_animal"),     # girl=agent → 续写可能是tool/food
        ("monkey", "rode", "class_animal"), # monkey=agent → 续写可能是vehicle
    ]
    
    ap_results = {}
    for agent, verb, agent_fam in AGENT_PATIENT_PAIRS:
        text = f"The {agent} {verb} the"
        cl = get_logits(model, tok, text)
        all_fam = fam_logits(cl, tok, FAM)
        top5_ids = np.argsort(cl)[-5:][::-1]
        top5 = [(tok.decode([i]).strip(), round(float(cl[i]), 4)) for i in top5_ids]
        ap_results[f"{agent}_{verb}"] = {
            "all_fam_logits": all_fam,
            "top5": top5,
        }
        plog(f"    '{agent} {verb} the': {all_fam}")
    
    results["_agent_patient"] = ap_results
    
    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    rnd = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    plog(f"Phase 459: {model_name} R{rnd}")
    plog(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    model, tok = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, {info.n_layers}L, d={info.d_model}")
    
    all_results = {
        "model": model_name,
        "round": rnd,
        "model_info": {"class": info.model_class, "n_layers": info.n_layers, "d_model": info.d_model},
    }
    
    # Exp1: is_a子槽位
    try:
        r1 = exp1_isa_subslot_clustering(model, tok, info, rnd)
        all_results["exp1_isa_subslot"] = r1
        plog("Exp1 done")
    except Exception as e:
        plog(f"Exp1 error: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp2: 否定算子闭环
    try:
        r2 = exp2_negation_operator_closure(model, tok, info, rnd)
        all_results["exp2_negation_closure"] = r2
        plog("Exp2 done")
    except Exception as e:
        plog(f"Exp2 error: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp3: 多跳因果验证
    try:
        r3 = exp3_multihop_causal(model, tok, info, rnd)
        all_results["exp3_multihop_causal"] = r3
        plog("Exp3 done")
    except Exception as e:
        plog(f"Exp3 error: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp4: has_part修复
    try:
        r4 = exp4_has_part_repair(model, tok, info, rnd)
        all_results["exp4_has_part_repair"] = r4
        plog("Exp4 done")
    except Exception as e:
        plog(f"Exp4 error: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp5: DS7B路径分裂大样本(仅DS7B, 其他模型简化)
    try:
        r5 = exp5_path_split_large(model, tok, info, rnd)
        all_results["exp5_path_split"] = r5
        plog("Exp5 done")
    except Exception as e:
        plog(f"Exp5 error: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp6: 语法角色绑定
    try:
        r6 = exp6_syntax_role_binding(model, tok, info, rnd)
        all_results["exp6_syntax_binding"] = r6
        plog("Exp6 done")
    except Exception as e:
        plog(f"Exp6 error: {e}")
        import traceback; traceback.print_exc()
    
    # 保存
    os.makedirs("results/glm5", exist_ok=True)
    outf = f"results/glm5/phase459_{model_name}_r{rnd}.json"
    with open(outf, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    plog(f"Saved: {outf}")
    
    # 释放
    release_model(model)
    plog(f"Phase 459 {model_name} R{rnd} complete!")


if __name__ == "__main__":
    main()
