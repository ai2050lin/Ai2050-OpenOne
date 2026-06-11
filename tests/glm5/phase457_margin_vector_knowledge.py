"""
Phase 457: 候选族竞争边际向量与知识图边验证
==============================================
Exp1: 竞争族特异边际向量 (对每个竞争族分别算边际效应)
Exp2: Family-local softmax + 多聚合器对照
Exp3: 知识图边验证 (4种关系槽位)
Exp4: DS7B路径翻转密集层扫描
Exp5: 否定效应

用法: python tests/glm5/phase457_margin_vector_knowledge.py qwen3 1
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
    "attr_part":     ["wheel", "blade", "handle", "engine", "seat", "wing", "leg", "head", "body", "tail"],
    "attr_material": ["metal", "wood", "plastic", "steel", "iron", "rubber", "glass", "leather", "stone", "fabric"],
    "attr_function": ["move", "cut", "carry", "hold", "drive", "build", "eat", "grow", "protect", "transport"],
    "attr_location": ["kitchen", "garage", "forest", "road", "garden", "water", "sky", "field", "house", "farm"],
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

# 知识图关系
RELATIONS = {
    "is_a":      {"templates": ["The {obj} is a"], "target": "dynamic", "compete": "dynamic"},
    "has_color": {"templates": ["The color of the {obj} is"], "target": "attr_color",
                 "compete": ["attr_part", "attr_material", "attr_function"]},
    "has_part":  {"templates": ["The {obj} has a"], "target": "attr_part",
                 "compete": ["attr_color", "attr_material", "attr_function"]},
    "used_for":  {"templates": ["A {obj} is used to"], "target": "attr_function",
                 "compete": ["attr_color", "attr_part", "attr_material"]},
}

OBJ2CAT = {}
for cat, objs in CAT_OBJ.items():
    for o in objs:
        OBJ2CAT[o] = cat

ROUNDS = {
    1: {k: v[:2] for k, v in CAT_OBJ.items()},   # pilot: 2/类
    2: {k: v[:8] for k, v in CAT_OBJ.items()},   # main: 8/类
    3: CAT_OBJ,                                    # confirm: 12/类
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
    """logsumexp for each family"""
    r = {}
    for fname, words in fam_dict.items():
        ids = [tok.encode(w, add_special_tokens=False)[0] for w in words if tok.encode(w, add_special_tokens=False)]
        if ids:
            fl = logits_np[ids]
            mx = np.max(fl)
            r[fname] = round(float(mx + np.log(np.sum(np.exp(fl - mx)))), 4)
    return r

def local_softmax(fam_lse):
    """softmax over family logsumexp values"""
    if not fam_lse: return {}
    vals = np.array(list(fam_lse.values()))
    mx = np.max(vals)
    ev = np.exp(vals - mx)
    tot = np.sum(ev)
    return {k: round(float(p), 6) for k, p in zip(fam_lse.keys(), ev / tot)}

# ==================== 通用消融测试 ====================
def run_ablation(model, tok, text, layer, fam_keys, fam_dict=None):
    """Run clean + zero_attn + zero_mlp, return family logits for each"""
    fd = fam_dict or FAM
    dev = get_dev(model)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=64)
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

# ==================== Exp1: 竞争族特异边际向量 ====================
def exp1_competitor_margin_vector(model, tok, info, rnd=1):
    plog(f"\n{'='*60}\nExp1: Competitor-Specific Margin Vectors\n{'='*60}")
    layers = get_layers(model)
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    li = info.n_layers - 1
    layer = layers[li]
    results = {}
    
    for cat_name, obj_list in obj_set.items():
        ci = CAT_FAM[cat_name]
        tc = ci["target"]
        comps = ci["compete"]
        cat_r = {}
        
        for obj in obj_list:
            text = f"The {obj} is a"
            fam_keys = [tc] + comps
            cl, cf, za, zf, zm, mf = run_ablation(model, tok, text, layer, fam_keys)
            
            # 边际向量: target vs each competitor
            def margin_vec(fam_log):
                ts = fam_log.get(tc, -999)
                return {c: round(ts - fam_log.get(c, -999), 4) for c in comps}
            
            cm = margin_vec(cf)
            am = margin_vec(zf)
            mm = margin_vec(mf)
            
            attn_mv = {c: round(cm[c] - am[c], 4) for c in comps}
            mlp_mv = {c: round(cm[c] - mm[c], 4) for c in comps}
            
            cat_r[obj] = {
                "clean_margins": cm,
                "attn_margin_vector": attn_mv,
                "mlp_margin_vector": mlp_mv,
            }
        
        # 汇总
        summary = {"attn": {}, "mlp": {}}
        for c in comps:
            ae, se = avg_std([v["attn_margin_vector"][c] for v in cat_r.values()])
            me, se2 = avg_std([v["mlp_margin_vector"][c] for v in cat_r.values()])
            summary["attn"][c] = {"avg": ae, "type": eff_type(ae)}
            summary["mlp"][c] = {"avg": me, "type": eff_type(me)}
        
        cat_r["summary"] = summary
        plog(f"  {cat_name}: attn_mv={ {k: v['avg'] for k, v in summary['attn'].items()} }, "
             f"mlp_mv={ {k: v['avg'] for k, v in summary['mlp'].items()} }")
        results[cat_name] = cat_r
    
    return results


# ==================== Exp2: Family-Local Softmax ====================
def exp2_family_local_softmax(model, tok, info, rnd=1):
    plog(f"\n{'='*60}\nExp2: Family-Local Softmax + Aggregator Comparison\n{'='*60}")
    layers = get_layers(model)
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    li = info.n_layers - 1
    layer = layers[li]
    
    softmax_fams = {k: FAM[k] for k in ["class_fruit", "class_animal", "class_tool", "class_vehicle"]}
    results = {}
    
    for cat_name, obj_list in obj_set.items():
        ci = CAT_FAM[cat_name]
        tc = ci["target"]
        comps = ci["compete"]
        cat_r = {}
        
        for obj in obj_list:
            text = f"The {obj} is a"
            dev = get_dev(model)
            iid = tok(text, return_tensors="pt", truncation=True, max_length=64)
            iid_d = iid["input_ids"].to(dev)
            amask = iid["attention_mask"].to(dev)
            
            # Clean
            with torch.no_grad():
                cl = model(input_ids=iid_d, attention_mask=amask).logits[0, -1].float().cpu().numpy()
            
            # Mean logit margin
            cf_mean = fam_logits(cl, tok, softmax_fams)
            target_mean = cf_mean.get(tc, -999)
            top1_margin = round(target_mean - max(cf_mean.get(c, -999) for c in comps), 4)
            mean_margin = round(target_mean - float(np.mean([cf_mean.get(c, -999) for c in comps])), 4)
            
            # LogSumExp margin
            cf_lse = family_lse(cl, tok, softmax_fams)
            target_lse = cf_lse.get(tc, -999)
            lse_margin = round(target_lse - max(cf_lse.get(c, -999) for c in comps), 4)
            
            # Family-local softmax
            cf_probs = local_softmax(cf_lse)
            target_prob = cf_probs.get(tc, 0)
            compete_probs = [cf_probs.get(c, 0) for c in comps]
            softmax_margin = round(target_prob - max(compete_probs), 6)
            
            cat_r[obj] = {
                "top1_margin": top1_margin,
                "mean_margin": mean_margin,
                "lse_margin": lse_margin,
                "softmax_margin": softmax_margin,
                "family_probs": cf_probs,
            }
        
        # 汇总
        summary = {}
        for mtype in ["top1_margin", "mean_margin", "lse_margin", "softmax_margin"]:
            vals = [v[mtype] for v in cat_r.values() if v.get(mtype) is not None]
            avg, std = avg_std(vals)
            summary[mtype] = {"avg": avg, "std": std}
        
        # 方向一致性: top1和mean方向是否一致
        consistent = 0
        total = 0
        for v in cat_r.values():
            t1 = v.get("top1_margin")
            mm = v.get("mean_margin")
            if t1 is not None and mm is not None:
                total += 1
                if (t1 > 0 and mm > 0) or (t1 < 0 and mm < 0) or (abs(t1) < 0.1 and abs(mm) < 0.1):
                    consistent += 1
        
        summary["top1_mean_consistency"] = f"{consistent}/{total}"
        cat_r["summary"] = summary
        plog(f"  {cat_name}: top1={summary['top1_margin']['avg']}, mean={summary['mean_margin']['avg']}, "
             f"lse={summary['lse_margin']['avg']}, softmax={summary['softmax_margin']['avg']}, "
             f"consist={summary['top1_mean_consistency']}")
        results[cat_name] = cat_r
    
    return results


# ==================== Exp3: 知识图边验证 ====================
def exp3_knowledge_graph_edges(model, tok, info, rnd=1):
    plog(f"\n{'='*60}\nExp3: Knowledge Graph Edge Verification\n{'='*60}")
    layers = get_layers(model)
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    li = info.n_layers - 1
    layer = layers[li]
    results = {}
    
    for rel_name, rel_info in RELATIONS.items():
        plog(f"  Relation: {rel_name}")
        rel_r = {}
        
        for cat_name, obj_list in obj_set.items():
            # 确定目标族和竞争族
            if rel_info["target"] == "dynamic":
                ci = CAT_FAM[cat_name]
                tc = ci["target"]
                comp_fams = ci["compete"]
            else:
                tc = rel_info["target"]
                comp_fams = rel_info["compete"]
            
            cat_r = []
            for obj in obj_list:
                text = rel_info["templates"][0].format(obj=obj)
                fam_keys = [tc] + comp_fams
                cl, cf, za, zf, zm, mf = run_ablation(model, tok, text, layer, fam_keys)
                
                # 目标族的logit
                clean_target = cf.get(tc, -999)
                za_target = zf.get(tc, -999)
                zm_target = mf.get(tc, -999)
                
                # 目标族 vs 最强竞争族的margin
                clean_margin = clean_target - max(cf.get(c, -999) for c in comp_fams)
                za_margin = za_target - max(zf.get(c, -999) for c in comp_fams)
                zm_margin = zm_target - max(mf.get(c, -999) for c in comp_fams)
                
                cat_r.append({
                    "obj": obj,
                    "clean_margin": round(clean_margin, 4),
                    "attn_effect": round(clean_margin - za_margin, 4),
                    "mlp_effect": round(clean_margin - zm_margin, 4),
                    "clean_target_logit": round(clean_target, 4),
                })
            
            # 汇总
            avg_ae, _ = avg_std([v["attn_effect"] for v in cat_r])
            avg_me, _ = avg_std([v["mlp_effect"] for v in cat_r])
            avg_cm, _ = avg_std([v["clean_margin"] for v in cat_r])
            
            rel_r[cat_name] = {
                "objects": cat_r,
                "avg_clean_margin": avg_cm,
                "avg_attn_effect": avg_ae,
                "avg_mlp_effect": avg_me,
                "attn_type": eff_type(avg_ae),
                "mlp_type": eff_type(avg_me),
            }
            plog(f"    {cat_name}: margin={avg_cm}, attn={avg_ae}({eff_type(avg_ae)}), mlp={avg_me}({eff_type(avg_me)})")
        
        results[rel_name] = rel_r
    
    return results


# ==================== Exp4: DS7B路径翻转密集层扫描 ====================
def exp4_path_flip_layer_scan(model, tok, info, rnd=1):
    plog(f"\n{'='*60}\nExp4: Fruit/Animal Path Flip Layer-by-Layer\n{'='*60}")
    layers = get_layers(model)
    n_layers = info.n_layers
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    
    # 对每个类别只用2个对象(速度), 但对更多层
    test_objs = {k: v[:2] for k, v in obj_set.items()}
    
    # 密集采样后层
    if n_layers <= 12:
        test_layers = list(range(n_layers))
    else:
        test_layers = sorted(set(list(range(0, n_layers, max(1, n_layers // 8))) + list(range(max(0, n_layers - 8), n_layers))))
    
    plog(f"  Testing layers: {test_layers}")
    results = {}
    
    for li in test_layers:
        layer = layers[li]
        layer_r = {}
        
        for cat_name in ["fruit", "animal"]:  # 只测关键的2个类别
            ci = CAT_FAM[cat_name]
            tc = ci["target"]
            comp_fams = ci["compete"]
            cat_vals_a = []
            cat_vals_m = []
            
            for obj in test_objs[cat_name]:
                text = f"The {obj} is a"
                fam_keys = [tc] + comp_fams
                cl, cf, za, zf, zm, mf = run_ablation(model, tok, text, layer, fam_keys)
                
                target_c = cf.get(tc, -999)
                margin_c = target_c - max(cf.get(c, -999) for c in comp_fams)
                margin_za = zf.get(tc, -999) - max(zf.get(c, -999) for c in comp_fams)
                margin_zm = mf.get(tc, -999) - max(mf.get(c, -999) for c in comp_fams)
                
                cat_vals_a.append(round(margin_c - margin_za, 4))
                cat_vals_m.append(round(margin_c - margin_zm, 4))
            
            avg_a, _ = avg_std(cat_vals_a)
            avg_m, _ = avg_std(cat_vals_m)
            layer_r[cat_name] = {"attn": avg_a, "mlp": avg_m, "attn_t": eff_type(avg_a), "mlp_t": eff_type(avg_m)}
        
        # fruit/animal路径是否相反
        fruit_path = f"a={layer_r['fruit']['attn_t']},m={layer_r['fruit']['mlp_t']}"
        animal_path = f"a={layer_r['animal']['attn_t']},m={layer_r['animal']['mlp_t']}"
        flipped = (layer_r['fruit']['attn'] and layer_r['animal']['attn'] and
                   layer_r['fruit']['attn'] * layer_r['animal']['attn'] < 0) or \
                  (layer_r['fruit']['mlp'] and layer_r['animal']['mlp'] and
                   layer_r['fruit']['mlp'] * layer_r['animal']['mlp'] < 0)
        
        layer_r["_fruit_path"] = fruit_path
        layer_r["_animal_path"] = animal_path
        layer_r["_flipped"] = flipped
        
        plog(f"  L{li}: fruit=[{fruit_path}] animal=[{animal_path}] {'FLIP!' if flipped else ''}")
        results[f"L{li}"] = layer_r
    
    return results


# ==================== Exp5: 否定效应 ====================
def exp5_negation_effect(model, tok, info, rnd=1):
    plog(f"\n{'='*60}\nExp5: Negation Effect on Family Margin\n{'='*60}")
    layers = get_layers(model)
    obj_set = ROUNDS.get(rnd, ROUNDS[1])
    li = info.n_layers - 1
    layer = layers[li]
    results = {}
    
    for cat_name, obj_list in obj_set.items():
        ci = CAT_FAM[cat_name]
        tc = ci["target"]
        comp_fams = ci["compete"]
        cat_r = []
        
        for obj in obj_list:
            # 肯定句
            text_aff = f"The {obj} is a"
            # 否定句
            text_neg = f"The {obj} is not a"
            
            for cond, text in [("affirmative", text_aff), ("negation", text_neg)]:
                fam_keys = [tc] + comp_fams
                dev = get_dev(model)
                iid = tok(text, return_tensors="pt", truncation=True, max_length=64)
                iid_d = iid["input_ids"].to(dev)
                amask = iid["attention_mask"].to(dev)
                
                with torch.no_grad():
                    cl = model(input_ids=iid_d, attention_mask=amask).logits[0, -1].float().cpu().numpy()
                cf = fam_logits(cl, tok, {k: FAM[k] for k in fam_keys if k in FAM})
                
                target_logit = cf.get(tc, -999)
                margin = target_logit - max(cf.get(c, -999) for c in comp_fams)
                
                # 各竞争族的logit
                all_logits = {k: v for k, v in cf.items()}
                
                cat_r.append({
                    "obj": obj, "condition": cond,
                    "target_logit": round(target_logit, 4),
                    "margin": round(margin, 4),
                    "family_logits": all_logits,
                })
        
        # 对比肯定vs否定
        aff_margins = [v["margin"] for v in cat_r if v["condition"] == "affirmative"]
        neg_margins = [v["margin"] for v in cat_r if v["condition"] == "negation"]
        aff_targets = [v["target_logit"] for v in cat_r if v["condition"] == "affirmative"]
        neg_targets = [v["target_logit"] for v in cat_r if v["condition"] == "negation"]
        
        avg_am, _ = avg_std(aff_margins)
        avg_nm, _ = avg_std(neg_margins)
        avg_at, _ = avg_std(aff_targets)
        avg_nt, _ = avg_std(neg_targets)
        
        margin_change = round(avg_nm - avg_am, 4) if avg_nm is not None and avg_am is not None else None
        
        results[cat_name] = {
            "objects": cat_r,
            "affirmative_margin": avg_am,
            "negation_margin": avg_nm,
            "margin_change": margin_change,
            "affirmative_target": avg_at,
            "negation_target": avg_nt,
        }
        plog(f"  {cat_name}: aff_margin={avg_am}, neg_margin={avg_nm}, change={margin_change}, "
             f"aff_target={avg_at}, neg_target={avg_nt}")
    
    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    plog(f"Phase 457: {model_name} Round {round_num}")
    plog(f"{'='*60}")
    
    # 加载模型
    t0 = time.time()
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    plog(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    plog(f"  Load time: {time.time()-t0:.1f}s")
    
    all_results = {
        "model": model_name,
        "round": round_num,
        "sign_convention": "ComponentMarginEffect = Margin_clean - Margin_zero_ablated (positive=promotes)",
        "model_info": {"class": info.model_class, "n_layers": info.n_layers, "d_model": info.d_model},
    }
    
    # Run all experiments
    try:
        all_results["exp1_competitor_margin_vector"] = exp1_competitor_margin_vector(model, tokenizer, info, round_num)
    except Exception as e:
        plog(f"Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_competitor_margin_vector"] = {"error": str(e)}
    
    gc.collect(); torch.cuda.empty_cache()
    
    try:
        all_results["exp2_family_local_softmax"] = exp2_family_local_softmax(model, tokenizer, info, round_num)
    except Exception as e:
        plog(f"Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_family_local_softmax"] = {"error": str(e)}
    
    gc.collect(); torch.cuda.empty_cache()
    
    try:
        all_results["exp3_knowledge_graph_edges"] = exp3_knowledge_graph_edges(model, tokenizer, info, round_num)
    except Exception as e:
        plog(f"Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_knowledge_graph_edges"] = {"error": str(e)}
    
    gc.collect(); torch.cuda.empty_cache()
    
    try:
        all_results["exp4_path_flip_scan"] = exp4_path_flip_layer_scan(model, tokenizer, info, round_num)
    except Exception as e:
        plog(f"Exp4 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp4_path_flip_scan"] = {"error": str(e)}
    
    gc.collect(); torch.cuda.empty_cache()
    
    try:
        all_results["exp5_negation_effect"] = exp5_negation_effect(model, tokenizer, info, round_num)
    except Exception as e:
        plog(f"Exp5 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp5_negation_effect"] = {"error": str(e)}
    
    # 保存
    out_dir = "results/glm5"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"phase457_{model_name}_r{round_num}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    plog(f"\nSaved to {out_path}")
    
    # 释放
    release_model(model)
    del model; gc.collect(); torch.cuda.empty_cache()
    plog(f"Phase 457 {model_name} R{round_num} done!")


if __name__ == "__main__":
    main()
