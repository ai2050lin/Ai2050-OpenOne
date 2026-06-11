"""
Phase 456: 候选族边际动力学的跨模板、跨对象、跨槽位闭环验证
================================================================
核心验证:
1. 三种Margin定义的鲁棒性: Top1Margin / MeanMargin / SoftmaxFamilyMargin
2. 多模板验证: 每个槽位4个模板
3. 对象扩展: 8个/类别
4. 全层attn/MLP margin效应扫描
5. DS7B fruit/animal路径翻转复验

符号约定 (强制):
  ComponentMarginEffect = Margin_clean - Margin_zero_ablated
  > 0: 组件促进目标边际
  < 0: 组件压制目标边际

用法:
  python tests/glm5/phase456_margin_dynamics.py qwen3 1
  python tests/glm5/phase456_margin_dynamics.py glm4 1
  python tests/glm5/phase456_margin_dynamics.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import os, gc, time, json, logging, math
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger("p456")

def plog(msg):
    log.info(msg)

def plog_always(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 符号约定 (强制) ====================
# ComponentMarginEffect = Margin_clean - Margin_zero_ablated
#   > 0: 组件促进目标边际 (移除组件后边际下降)
#   < 0: 组件压制目标边际 (移除组件后边际上升)


# ==================== 候选族标准化定义 (与455一致) ====================
FAMILY_STANDARD = {
    "class_fruit":   ["fruit", "produce", "crop", "harvest"],
    "class_animal":  ["animal", "creature", "beast", "mammal"],
    "class_tool":    ["tool", "implement", "instrument", "device"],
    "class_vehicle": ["vehicle", "transport", "conveyance", "automobile"],
    "member_fruit":   ["apple", "banana", "orange", "grape", "pear", "peach", "lemon", "mango", "cherry", "plum"],
    "member_animal":  ["dog", "cat", "horse", "lion", "bear", "rabbit", "cow", "tiger", "eagle", "deer"],
    "member_tool":    ["hammer", "wrench", "saw", "drill", "axe", "knife", "chisel", "pliers", "shovel", "screwdriver"],
    "member_vehicle": ["car", "truck", "bus", "train", "bicycle", "boat", "plane", "motorcycle", "van", "scooter"],
    "attr_color":    ["red", "green", "yellow", "blue", "brown", "black", "white", "orange", "gray", "pink"],
    "attr_part":     ["wheel", "blade", "handle", "engine", "seat", "wing", "leg", "head", "body", "tail"],
    "attr_material": ["metal", "wood", "plastic", "steel", "iron", "rubber", "glass", "leather", "stone", "fabric"],
    "attr_function": ["move", "cut", "carry", "hold", "drive", "build", "eat", "grow", "protect", "transport"],
    "generic":       ["thing", "item", "object", "entity", "piece", "one", "it", "that"],
}

# 对象集 (R1=8/类, R2=12/类)
CATEGORY_OBJECTS = {
    "fruit":   ["apple", "banana", "orange", "grape", "pear", "peach", "lemon", "mango",
                "cherry", "plum", "melon", "pineapple"],
    "animal":  ["dog", "cat", "horse", "lion", "bear", "rabbit", "cow", "tiger",
                "sheep", "wolf", "deer", "monkey"],
    "tool":    ["hammer", "knife", "wrench", "saw", "drill", "axe", "shovel", "scissors",
                "pliers", "needle", "brush", "chisel"],
    "vehicle": ["car", "bus", "bicycle", "truck", "train", "boat", "plane", "scooter",
                "motorcycle", "taxi", "van", "ship"],
}

# 类别→候选族映射
CATEGORY_FAMILIES = {
    "fruit": {"target_class": "class_fruit", "target_member": "member_fruit",
              "compete_class": ["class_animal", "class_tool", "class_vehicle"]},
    "animal": {"target_class": "class_animal", "target_member": "member_animal",
              "compete_class": ["class_fruit", "class_tool", "class_vehicle"]},
    "tool":   {"target_class": "class_tool", "target_member": "member_tool",
              "compete_class": ["class_fruit", "class_animal", "class_vehicle"]},
    "vehicle":{"target_class": "class_vehicle", "target_member": "member_vehicle",
              "compete_class": ["class_fruit", "class_animal", "class_tool"]},
}

# 多模板 (每个槽位4个模板)
SLOT_TEMPLATES = {
    "cat": [
        "The {obj} is a",
        "A {obj} is classified as a",
        "{obj} belongs to the category of",
        "The correct class for {obj} is",
    ],
    "color": [
        "The color of the {obj} is",
        "A typical {obj} looks",
        "The {obj} appears",
        "People describe the color of {obj} as",
    ],
    "function": [
        "You use a {obj} to",
        "A {obj} can",
        "The purpose of the {obj} is to",
        "People use {obj} for",
    ],
}

# 槽位→目标族/竞争族
SLOT_FAMILIES = {
    "cat": {"target_type": "class", "compete_type": "class"},  # 由对象类别动态决定
    "color": {"target": "attr_color", "compete": ["attr_part", "attr_material", "attr_function"]},
    "function": {"target": "attr_function", "compete": ["attr_color", "attr_part", "attr_material"]},
}

OBJ_TO_CATEGORY = {}
for cat, objs in CATEGORY_OBJECTS.items():
    for obj in objs:
        OBJ_TO_CATEGORY[obj] = cat

# Round定义
ROUNDS = {
    1: {k: v[:8] for k, v in CATEGORY_OBJECTS.items()},
    2: CATEGORY_OBJECTS,  # 全部12个
}

# 采样层
SAMPLE_LAYERS = {
    "qwen3": [0, 6, 12, 18, 24, 27, 30, 33, 35],
    "glm4":  [0, 6, 12, 18, 24, 30, 36, 38, 39],
    "deepseek7b": [0, 4, 8, 14, 20, 24, 26, 27],
}


# ==================== 模型加载 ====================
def load_model_auto(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog_always(f"Loading {model_name} (bf16 + auto + flash)...")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True,
                                               local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=attn_impl)
            plog_always(f"  attn={attn_impl}")
            break
        except Exception as e:
            plog(f"  attn={attn_impl} failed: {e}")
            continue
    
    model.eval()
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        layer_devs = {}
        for k, v in dmap.items():
            if k.startswith('model.layers.'):
                lid = k.split('.')[2]
                if lid not in layer_devs:
                    layer_devs[lid] = str(v)
        gpu_l = sum(1 for v in layer_devs.values() if 'cuda' in str(v))
        cpu_l = sum(1 for v in layer_devs.values() if 'cpu' in str(v))
        plog_always(f"  Layers: {gpu_l} GPU + {cpu_l} CPU")
    
    return model, tokenizer


def get_input_device(model):
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            return model.model.embed_tokens.weight.device
    except:
        pass
    try:
        return model.get_input_embeddings().weight.device
    except:
        pass
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==================== 工具函数 ====================
def get_logit_for_words(logits_np, tokenizer, word_list):
    ids = []
    for w in word_list:
        tok_ids = tokenizer.encode(w, add_special_tokens=False)
        if tok_ids:
            ids.append(tok_ids[0])
    if not ids:
        return None
    return float(np.mean(logits_np[ids]))


def compute_family_logits(logits_np, tokenizer, family_dict=None):
    if family_dict is None:
        family_dict = FAMILY_STANDARD
    result = {}
    for family_name, words in family_dict.items():
        val = get_logit_for_words(logits_np, tokenizer, words)
        if val is not None:
            result[family_name] = round(val, 4)
    return result


# ==================== 三种Margin定义 ====================
def compute_top1_margin(family_logits, target_key, compete_keys):
    """Top1Margin = target - max(compete)"""
    target = family_logits.get(target_key, -999)
    competes = [family_logits.get(k, -999) for k in compete_keys]
    return round(target - max(competes), 4) if competes else round(target, 4)


def compute_mean_margin(family_logits, target_key, compete_keys):
    """MeanMargin = target - mean(compete)"""
    target = family_logits.get(target_key, -999)
    competes = [family_logits.get(k, -999) for k in compete_keys]
    return round(target - float(np.mean(competes)), 4) if competes else round(target, 4)


def compute_softmax_margin(logits_np, tokenizer, target_words, compete_word_lists):
    """SoftmaxFamilyMargin = P(target_family) - P(compete_families)"""
    # 收集所有词的token ids
    all_ids = []
    target_ids = []
    compete_ids = []
    
    for w in target_words:
        tok_ids = tokenizer.encode(w, add_special_tokens=False)
        if tok_ids:
            target_ids.append(tok_ids[0])
    
    for word_list in compete_word_lists:
        cids = []
        for w in word_list:
            tok_ids = tokenizer.encode(w, add_special_tokens=False)
            if tok_ids:
                cids.append(tok_ids[0])
        compete_ids.append(cids)
    
    if not target_ids:
        return None
    
    # softmax
    all_target_ids = set(target_ids)
    all_compete_ids = set()
    for cids in compete_ids:
        all_compete_ids.update(cids)
    
    max_logit = logits_np.max()
    exp_logits = np.exp(logits_np - max_logit)
    total_prob = exp_logits.sum()
    
    target_prob = float(exp_logits[list(all_target_ids)].sum() / total_prob)
    compete_prob = float(exp_logits[list(all_compete_ids)].sum() / total_prob) if all_compete_ids else 0.0
    
    return round(target_prob - compete_prob, 6)


def compute_three_margins(family_logits, logits_np, tokenizer, target_key, compete_keys):
    """同时计算三种margin"""
    target_words = FAMILY_STANDARD.get(target_key, [target_key])
    compete_word_lists = [FAMILY_STANDARD.get(k, [k]) for k in compete_keys]
    
    top1 = compute_top1_margin(family_logits, target_key, compete_keys)
    mean_m = compute_mean_margin(family_logits, target_key, compete_keys)
    soft_m = compute_softmax_margin(logits_np, tokenizer, target_words, compete_word_lists)
    
    return {"top1": top1, "mean": mean_m, "softmax": soft_m}


def make_zero_hook(lpos):
    def hook(m, inp, out):
        if isinstance(out, tuple):
            new_out = out[0].clone()
            new_out[0, lpos] = 0.0
            return (new_out,) + out[1:]
        else:
            new_out = out.clone()
            new_out[0, lpos] = 0.0
            return new_out
    return hook


def avg_with_std(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    return round(float(np.mean(clean)), 4), round(float(np.std(clean)), 4)


def classify_effect(val, threshold=0.1):
    if val is None:
        return "N/A"
    if val > threshold:
        return "PROMOTES"
    elif val < -threshold:
        return "SUPPRESSES"
    return "NEUTRAL"


# ==================== Exp1: 三种Margin定义鲁棒性 ====================
def exp1_margin_robustness(model, tokenizer, info, round_num=1):
    """
    验证三种Margin定义是否给出一致的组件效应方向
    
    对最后2层, 4类别, cat槽位, 计算:
    - Top1Margin效应
    - MeanMargin效应  
    - SoftmaxFamilyMargin效应
    
    如果三种指标方向一致, 说明结论鲁棒
    """
    plog_always(f"\n{'='*60}")
    plog_always(f"Exp1: Margin Definition Robustness")
    plog_always(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)
    obj_set = ROUNDS.get(round_num, ROUNDS[1])
    test_layers = [n_layers - 2, n_layers - 1]
    
    results = {}
    consistency_count = 0  # 三种margin方向一致次数
    total_count = 0
    
    for li in test_layers:
        layer = layers[li]
        plog_always(f"  L{li}...")
        layer_result = {}
        
        for cat_name, obj_list in obj_set.items():
            cat_info = CATEGORY_FAMILIES[cat_name]
            cat_result = {}
            
            for obj_name in obj_list:
                text = f"The {obj_name} is a"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attention_mask = inputs["attention_mask"].to(input_device)
                last_pos = input_ids.shape[1] - 1
                
                # Clean
                with torch.no_grad():
                    out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
                clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
                clean_family = compute_family_logits(clean_logits, tokenizer)
                clean_margins = compute_three_margins(clean_family, clean_logits, tokenizer,
                                                       cat_info["target_class"], cat_info["compete_class"])
                
                # Zero attn
                h_attn = None
                if hasattr(layer, 'self_attn'):
                    h_attn = layer.self_attn.register_forward_hook(make_zero_hook(last_pos))
                with torch.no_grad():
                    out_za = model(input_ids=input_ids, attention_mask=attention_mask)
                if h_attn:
                    h_attn.remove()
                za_logits = out_za.logits[0, -1].float().cpu().numpy()
                za_family = compute_family_logits(za_logits, tokenizer)
                za_margins = compute_three_margins(za_family, za_logits, tokenizer,
                                                    cat_info["target_class"], cat_info["compete_class"])
                
                # Zero MLP
                mlp_mod = getattr(layer, 'mlp', None) or getattr(layer, 'feed_forward', None)
                h_mlp = None
                if mlp_mod is not None:
                    h_mlp = mlp_mod.register_forward_hook(make_zero_hook(last_pos))
                with torch.no_grad():
                    out_zm = model(input_ids=input_ids, attention_mask=attention_mask)
                if h_mlp:
                    h_mlp.remove()
                zm_logits = out_zm.logits[0, -1].float().cpu().numpy()
                zm_family = compute_family_logits(zm_logits, tokenizer)
                zm_margins = compute_three_margins(zm_family, zm_logits, tokenizer,
                                                    cat_info["target_class"], cat_info["compete_class"])
                
                # 计算三种margin效应
                attn_effects = {}
                mlp_effects = {}
                for mtype in ["top1", "mean", "softmax"]:
                    if clean_margins[mtype] is not None and za_margins[mtype] is not None:
                        attn_effects[mtype] = round(clean_margins[mtype] - za_margins[mtype], 4)
                    else:
                        attn_effects[mtype] = None
                    if clean_margins[mtype] is not None and zm_margins[mtype] is not None:
                        mlp_effects[mtype] = round(clean_margins[mtype] - zm_margins[mtype], 4)
                    else:
                        mlp_effects[mtype] = None
                
                # 方向一致性检查
                for comp_name, effects in [("attn", attn_effects), ("mlp", mlp_effects)]:
                    valid = [v for v in effects.values() if v is not None]
                    if len(valid) >= 2:
                        total_count += 1
                        signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in valid]
                        if len(set(signs)) == 1:
                            consistency_count += 1
                
                cat_result[obj_name] = {
                    "clean_margins": clean_margins,
                    "attn_effects": attn_effects,
                    "mlp_effects": mlp_effects,
                }
            
            # 类别平均
            summary = {}
            for mtype in ["top1", "mean", "softmax"]:
                ae_vals = [v["attn_effects"].get(mtype) for v in cat_result.values() 
                           if v["attn_effects"].get(mtype) is not None]
                me_vals = [v["mlp_effects"].get(mtype) for v in cat_result.values() 
                           if v["mlp_effects"].get(mtype) is not None]
                avg_ae, std_ae = avg_with_std(ae_vals)
                avg_me, std_me = avg_with_std(me_vals)
                summary[f"attn_{mtype}"] = {"avg": avg_ae, "std": std_ae, "type": classify_effect(avg_ae)}
                summary[f"mlp_{mtype}"] = {"avg": avg_me, "std": std_me, "type": classify_effect(avg_me)}
            
            cat_result["summary"] = summary
            plog_always(f"    {cat_name}: top1_attn={summary['attn_top1']['avg']}({summary['attn_top1']['type']}), "
                        f"mean_attn={summary['attn_mean']['avg']}({summary['attn_mean']['type']}), "
                        f"soft_attn={summary['attn_softmax']['avg']}({summary['attn_softmax']['type']})")
            
            layer_result[cat_name] = cat_result
        results[f"L{li}"] = layer_result
    
    consistency_rate = round(consistency_count / max(total_count, 1), 4)
    plog_always(f"\n  Margin consistency rate: {consistency_count}/{total_count} = {consistency_rate}")
    results["_margin_consistency"] = {"consistent": consistency_count, "total": total_count,
                                       "rate": consistency_rate}
    return results


# ==================== Exp2: 多模板验证 ====================
def exp2_multi_template(model, tokenizer, info, round_num=1):
    """
    验证组件边际效应是否跨模板稳定
    
    对cat/color/function三个槽位, 各4个模板
    只测最后层, 4类别, 每类4个对象(减少计算量)
    """
    plog_always(f"\n{'='*60}")
    plog_always(f"Exp2: Multi-Template Verification")
    plog_always(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)
    last_layer = n_layers - 1
    layer = layers[last_layer]
    
    # 每类4个对象(减少计算量)
    obj_set = {k: v[:4] for k, v in ROUNDS.get(round_num, ROUNDS[1]).items()}
    
    results = {}
    
    for slot_name, templates in SLOT_TEMPLATES.items():
        plog_always(f"  Slot: {slot_name} ({len(templates)} templates)")
        slot_result = {}
        
        for cat_name, obj_list in obj_set.items():
            cat_info = CATEGORY_FAMILIES[cat_name]
            cat_result = {}
            
            for obj_name in obj_list:
                obj_template_results = []
                
                for tidx, template in enumerate(templates):
                    text = template.format(obj=obj_name)
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                    input_ids = inputs["input_ids"].to(input_device)
                    attention_mask = inputs["attention_mask"].to(input_device)
                    last_pos = input_ids.shape[1] - 1
                    
                    # 确定目标族
                    if slot_name == "cat":
                        target_family = cat_info["target_class"]
                        compete_families = cat_info["compete_class"]
                    elif slot_name == "color":
                        target_family = "attr_color"
                        compete_families = ["attr_part", "attr_material", "attr_function"]
                    else:  # function
                        target_family = "attr_function"
                        compete_families = ["attr_color", "attr_part", "attr_material"]
                    
                    # Clean
                    with torch.no_grad():
                        out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
                    clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
                    clean_family = compute_family_logits(clean_logits, tokenizer)
                    clean_margin = compute_top1_margin(clean_family, target_family, compete_families)
                    
                    # Zero attn
                    h_attn = None
                    if hasattr(layer, 'self_attn'):
                        h_attn = layer.self_attn.register_forward_hook(make_zero_hook(last_pos))
                    with torch.no_grad():
                        out_za = model(input_ids=input_ids, attention_mask=attention_mask)
                    if h_attn:
                        h_attn.remove()
                    za_logits = out_za.logits[0, -1].float().cpu().numpy()
                    za_family = compute_family_logits(za_logits, tokenizer)
                    za_margin = compute_top1_margin(za_family, target_family, compete_families)
                    
                    # Zero MLP
                    mlp_mod = getattr(layer, 'mlp', None) or getattr(layer, 'feed_forward', None)
                    h_mlp = None
                    if mlp_mod is not None:
                        h_mlp = mlp_mod.register_forward_hook(make_zero_hook(last_pos))
                    with torch.no_grad():
                        out_zm = model(input_ids=input_ids, attention_mask=attention_mask)
                    if h_mlp:
                        h_mlp.remove()
                    zm_logits = out_zm.logits[0, -1].float().cpu().numpy()
                    zm_family = compute_family_logits(zm_logits, tokenizer)
                    zm_margin = compute_top1_margin(zm_family, target_family, compete_families)
                    
                    obj_template_results.append({
                        "template_idx": tidx,
                        "template": template,
                        "attn_effect": round(clean_margin - za_margin, 4),
                        "mlp_effect": round(clean_margin - zm_margin, 4),
                    })
                
                # 跨模板平均和稳定性
                attn_vals = [r["attn_effect"] for r in obj_template_results]
                mlp_vals = [r["mlp_effect"] for r in obj_template_results]
                avg_ae, std_ae = avg_with_std(attn_vals)
                avg_me, std_me = avg_with_std(mlp_vals)
                
                # 方向一致性: 所有模板是否同方向
                attn_signs = [1 if v > 0.1 else (-1 if v < -0.1 else 0) for v in attn_vals]
                mlp_signs = [1 if v > 0.1 else (-1 if v < -0.1 else 0) for v in mlp_vals]
                attn_consistent = len(set(s for s in attn_signs if s != 0)) <= 1
                mlp_consistent = len(set(s for s in mlp_signs if s != 0)) <= 1
                
                cat_result[obj_name] = {
                    "templates": obj_template_results,
                    "avg_attn": avg_ae, "std_attn": std_ae,
                    "avg_mlp": avg_me, "std_mlp": std_me,
                    "attn_consistent": attn_consistent,
                    "mlp_consistent": mlp_consistent,
                }
            
            # 类别汇总
            avg_attns = [v["avg_attn"] for v in cat_result.values() if v.get("avg_attn") is not None]
            avg_mlps = [v["avg_mlp"] for v in cat_result.values() if v.get("avg_mlp") is not None]
            attn_consist = sum(1 for v in cat_result.values() if v.get("attn_consistent"))
            mlp_consist = sum(1 for v in cat_result.values() if v.get("mlp_consistent"))
            n_objs = len(cat_result)
            
            cat_result["summary"] = {
                "avg_attn": avg_with_std(avg_attns)[0],
                "avg_mlp": avg_with_std(avg_mlps)[0],
                "attn_consistency_rate": round(attn_consist / n_objs, 4) if n_objs > 0 else None,
                "mlp_consistency_rate": round(mlp_consist / n_objs, 4) if n_objs > 0 else None,
            }
            plog_always(f"    {cat_name}: attn={cat_result['summary']['avg_attn']} "
                        f"(consist={attn_consist}/{n_objs}), "
                        f"mlp={cat_result['summary']['avg_mlp']} "
                        f"(consist={mlp_consist}/{n_objs})")
            
            slot_result[cat_name] = cat_result
        
        results[slot_name] = slot_result
    
    return results


# ==================== Exp3: 全层Margin效应扫描 ====================
def exp3_full_layer_margin_scan(model, tokenizer, info, round_num=1):
    """
    对所有采样层计算attn/MLP的margin效应
    
    验证:
    - Qwen3: MLP从前层AMPLIFIER转为后层SUPPRESSOR
    - GLM4: 最后层MLP是AMPLIFIER
    - DS7B: L26→L27翻转
    
    每类2个对象(减少计算量), cat槽位
    """
    plog_always(f"\n{'='*60}")
    plog_always(f"Exp3: Full-Layer Margin Effect Scan")
    plog_always(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)
    model_name = info.name
    
    sample_layers = SAMPLE_LAYERS.get(model_name, list(range(0, n_layers, 4)) + [n_layers - 1])
    obj_set = {k: v[:2] for k, v in ROUNDS.get(round_num, ROUNDS[1]).items()}  # 每类2个
    
    results = {}
    
    for li in sample_layers:
        if li >= n_layers:
            continue
        layer = layers[li]
        plog_always(f"  L{li}...")
        
        layer_attn_effects = []
        layer_mlp_effects = []
        
        for cat_name, obj_list in obj_set.items():
            cat_info = CATEGORY_FAMILIES[cat_name]
            
            for obj_name in obj_list:
                text = f"The {obj_name} is a"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attention_mask = inputs["attention_mask"].to(input_device)
                last_pos = input_ids.shape[1] - 1
                
                # Clean
                with torch.no_grad():
                    out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
                clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
                clean_family = compute_family_logits(clean_logits, tokenizer)
                clean_margin = compute_top1_margin(clean_family, cat_info["target_class"],
                                                    cat_info["compete_class"])
                
                # Zero attn
                h_attn = None
                if hasattr(layer, 'self_attn'):
                    h_attn = layer.self_attn.register_forward_hook(make_zero_hook(last_pos))
                with torch.no_grad():
                    out_za = model(input_ids=input_ids, attention_mask=attention_mask)
                if h_attn:
                    h_attn.remove()
                za_logits = out_za.logits[0, -1].float().cpu().numpy()
                za_family = compute_family_logits(za_logits, tokenizer)
                za_margin = compute_top1_margin(za_family, cat_info["target_class"],
                                                 cat_info["compete_class"])
                
                # Zero MLP
                mlp_mod = getattr(layer, 'mlp', None) or getattr(layer, 'feed_forward', None)
                h_mlp = None
                if mlp_mod is not None:
                    h_mlp = mlp_mod.register_forward_hook(make_zero_hook(last_pos))
                with torch.no_grad():
                    out_zm = model(input_ids=input_ids, attention_mask=attention_mask)
                if h_mlp:
                    h_mlp.remove()
                zm_logits = out_zm.logits[0, -1].float().cpu().numpy()
                zm_family = compute_family_logits(zm_logits, tokenizer)
                zm_margin = compute_top1_margin(zm_family, cat_info["target_class"],
                                                 cat_info["compete_class"])
                
                layer_attn_effects.append(round(clean_margin - za_margin, 4))
                layer_mlp_effects.append(round(clean_margin - zm_margin, 4))
        
        avg_ae, std_ae = avg_with_std(layer_attn_effects)
        avg_me, std_me = avg_with_std(layer_mlp_effects)
        
        results[f"L{li}"] = {
            "avg_attn_margin_effect": avg_ae, "std_attn": std_ae,
            "avg_mlp_margin_effect": avg_me, "std_mlp": std_me,
            "attn_type": classify_effect(avg_ae),
            "mlp_type": classify_effect(avg_me),
        }
        plog_always(f" attn={avg_ae}({classify_effect(avg_ae)}), mlp={avg_me}({classify_effect(avg_me)})")
    
    # 找转折点
    layer_keys = sorted([k for k in results.keys() if k.startswith("L")],
                        key=lambda x: int(x[1:]))
    mlp_vals = [results[k]["avg_mlp_margin_effect"] for k in layer_keys]
    attn_vals = [results[k]["avg_attn_margin_effect"] for k in layer_keys]
    
    transitions = []
    for i in range(1, len(mlp_vals)):
        if mlp_vals[i-1] is not None and mlp_vals[i] is not None:
            if mlp_vals[i-1] > 0.1 and mlp_vals[i] < -0.1:
                transitions.append(f"{layer_keys[i-1]}->{layer_keys[i]}: AMP->SUPP")
            elif mlp_vals[i-1] < -0.1 and mlp_vals[i] > 0.1:
                transitions.append(f"{layer_keys[i-1]}->{layer_keys[i]}: SUPP->AMP")
    
    if transitions:
        plog_always(f"\n  MLP Transitions: {transitions}")
    results["_transitions"] = transitions
    
    return results


# ==================== Exp4: DS7B fruit/animal路径翻转复验 ====================
def exp4_category_path_flipping(model, tokenizer, info, round_num=1):
    """
    对最后3层, 详细分析4类别的attn/MLP margin效应
    
    核心验证: DS7B L27 fruit vs animal的attn/MLP路径翻转
    
    同时记录每个族的logit变化, 解释margin效应的来源
    """
    plog_always(f"\n{'='*60}")
    plog_always(f"Exp4: Category Path Flipping Verification")
    plog_always(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)
    
    # 最后3层
    test_layers = [max(0, n_layers - 3), max(0, n_layers - 2), n_layers - 1]
    obj_set = ROUNDS.get(round_num, ROUNDS[1])
    
    results = {}
    
    for li in test_layers:
        layer = layers[li]
        plog_always(f"  L{li}...")
        layer_result = {}
        
        for cat_name, obj_list in obj_set.items():
            cat_info = CATEGORY_FAMILIES[cat_name]
            cat_result = {}
            
            # 每类4个对象
            for obj_name in obj_list[:4]:
                text = f"The {obj_name} is a"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attention_mask = inputs["attention_mask"].to(input_device)
                last_pos = input_ids.shape[1] - 1
                
                # Clean
                with torch.no_grad():
                    out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
                clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
                clean_family = compute_family_logits(clean_logits, tokenizer)
                clean_margin = compute_top1_margin(clean_family, cat_info["target_class"],
                                                    cat_info["compete_class"])
                
                # Zero attn
                h_attn = None
                if hasattr(layer, 'self_attn'):
                    h_attn = layer.self_attn.register_forward_hook(make_zero_hook(last_pos))
                with torch.no_grad():
                    out_za = model(input_ids=input_ids, attention_mask=attention_mask)
                if h_attn:
                    h_attn.remove()
                za_logits = out_za.logits[0, -1].float().cpu().numpy()
                za_family = compute_family_logits(za_logits, tokenizer)
                za_margin = compute_top1_margin(za_family, cat_info["target_class"],
                                                 cat_info["compete_class"])
                
                # Zero MLP
                mlp_mod = getattr(layer, 'mlp', None) or getattr(layer, 'feed_forward', None)
                h_mlp = None
                if mlp_mod is not None:
                    h_mlp = mlp_mod.register_forward_hook(make_zero_hook(last_pos))
                with torch.no_grad():
                    out_zm = model(input_ids=input_ids, attention_mask=attention_mask)
                if h_mlp:
                    h_mlp.remove()
                zm_logits = out_zm.logits[0, -1].float().cpu().numpy()
                zm_family = compute_family_logits(zm_logits, tokenizer)
                zm_margin = compute_top1_margin(zm_family, cat_info["target_class"],
                                                 cat_info["compete_class"])
                
                # Margin效应
                attn_margin_eff = round(clean_margin - za_margin, 4)
                mlp_margin_eff = round(clean_margin - zm_margin, 4)
                
                # 族级logit变化 (解释margin效应来源)
                family_logit_attn = {}
                family_logit_mlp = {}
                for fk in ["class_fruit", "class_animal", "class_tool", "class_vehicle"]:
                    family_logit_attn[fk] = round(clean_family.get(fk, 0) - za_family.get(fk, 0), 4)
                    family_logit_mlp[fk] = round(clean_family.get(fk, 0) - zm_family.get(fk, 0), 4)
                
                cat_result[obj_name] = {
                    "attn_margin_effect": attn_margin_eff,
                    "mlp_margin_effect": mlp_margin_eff,
                    "family_logit_attn": family_logit_attn,
                    "family_logit_mlp": family_logit_mlp,
                }
            
            # 类别平均
            attn_effs = [v["attn_margin_effect"] for v in cat_result.values()]
            mlp_effs = [v["mlp_margin_effect"] for v in cat_result.values()]
            avg_ae, std_ae = avg_with_std(attn_effs)
            avg_me, std_me = avg_with_std(mlp_effs)
            
            # 族级logit平均
            avg_fam_attn = {}
            avg_fam_mlp = {}
            for fk in ["class_fruit", "class_animal", "class_tool", "class_vehicle"]:
                vals_a = [v["family_logit_attn"].get(fk, 0) for v in cat_result.values()]
                vals_m = [v["family_logit_mlp"].get(fk, 0) for v in cat_result.values()]
                avg_fam_attn[fk] = round(float(np.mean(vals_a)), 4)
                avg_fam_mlp[fk] = round(float(np.mean(vals_m)), 4)
            
            cat_result["summary"] = {
                "avg_attn_margin": avg_ae, "std_attn": std_ae,
                "avg_mlp_margin": avg_me, "std_mlp": std_me,
                "attn_type": classify_effect(avg_ae),
                "mlp_type": classify_effect(avg_me),
                "avg_family_logit_attn": avg_fam_attn,
                "avg_family_logit_mlp": avg_fam_mlp,
            }
            
            plog_always(f"    {cat_name}: attn_margin={avg_ae}±{std_ae} ({classify_effect(avg_ae)}), "
                        f"mlp_margin={avg_me}±{std_me} ({classify_effect(avg_me)})")
            plog_always(f"      fam_attn: {avg_fam_attn}")
            plog_always(f"      fam_mlp: {avg_fam_mlp}")
            
            layer_result[cat_name] = cat_result
        results[f"L{li}"] = layer_result
    
    # 路径翻转检测
    last_key = f"L{n_layers - 1}"
    if last_key in results:
        last_data = results[last_key]
        flipping = {}
        for cat_name in obj_set.keys():
            if cat_name in last_data and "summary" in last_data[cat_name]:
                s = last_data[cat_name]["summary"]
                flipping[cat_name] = {
                    "attn": s["attn_type"],
                    "mlp": s["mlp_type"],
                    "path": f"attn={s['attn_type'][:3]}+mlp={s['mlp_type'][:3]}",
                }
        results["_path_summary"] = flipping
        plog_always(f"\n  Path summary: {flipping}")
    
    return results


# ==================== Exp5: 候选族logit级别分解 ====================
def exp5_family_logit_decomposition(model, tokenizer, info, round_num=1):
    """
    对最后层, 详细分析每个族的logit变化
    
    核心问题: MLP压制所有logit但促进目标边际的原因是什么?
    答案: 它对竞争族压制更多
    
    记录每个族的logit变化, 计算:
    - target族logit变化
    - 各竞争族logit变化
    - 变化差异 = target_delta - max(compete_delta)
    """
    plog_always(f"\n{'='*60}")
    plog_always(f"Exp5: Family Logit Decomposition")
    plog_always(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)
    last_layer = n_layers - 1
    layer = layers[last_layer]
    
    obj_set = ROUNDS.get(round_num, ROUNDS[1])
    results = {}
    
    for cat_name, obj_list in obj_set.items():
        cat_info = CATEGORY_FAMILIES[cat_name]
        cat_result = {}
        
        for obj_name in obj_list[:4]:  # 每类4个
            text = f"The {obj_name} is a"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            # Clean
            with torch.no_grad():
                out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
            clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
            clean_family = compute_family_logits(clean_logits, tokenizer)
            
            # Zero MLP
            mlp_mod = getattr(layer, 'mlp', None) or getattr(layer, 'feed_forward', None)
            h_mlp = None
            if mlp_mod is not None:
                h_mlp = mlp_mod.register_forward_hook(make_zero_hook(last_pos))
            with torch.no_grad():
                out_zm = model(input_ids=input_ids, attention_mask=attention_mask)
            if h_mlp:
                h_mlp.remove()
            zm_logits = out_zm.logits[0, -1].float().cpu().numpy()
            zm_family = compute_family_logits(zm_logits, tokenizer)
            
            # 每个族的logit变化 (ComponentEffect = clean - zero)
            delta_family = {}
            for fk in clean_family:
                delta_family[fk] = round(clean_family[fk] - zm_family.get(fk, 0), 4)
            
            # 目标族和竞争族的变化
            target_delta = delta_family.get(cat_info["target_class"], 0)
            compete_deltas = {k: delta_family.get(k, 0) for k in cat_info["compete_class"]}
            max_compete_delta = max(compete_deltas.values()) if compete_deltas else 0
            delta_diff = round(target_delta - max_compete_delta, 4)
            
            cat_result[obj_name] = {
                "delta_family": delta_family,
                "target_delta": target_delta,
                "compete_deltas": compete_deltas,
                "delta_diff": delta_diff,
                "interpretation": "MLP promotes target more" if delta_diff > 0 else "MLP promotes compete more",
            }
        
        # 类别平均
        avg_target = float(np.mean([v["target_delta"] for v in cat_result.values()]))
        avg_compete_deltas = {}
        for k in cat_info["compete_class"]:
            avg_compete_deltas[k] = round(float(np.mean([v["compete_deltas"].get(k, 0) for v in cat_result.values()])), 4)
        avg_delta_diff = float(np.mean([v["delta_diff"] for v in cat_result.values()]))
        
        cat_result["summary"] = {
            "avg_target_delta": round(avg_target, 4),
            "avg_compete_deltas": avg_compete_deltas,
            "avg_delta_diff": round(avg_delta_diff, 4),
            "interpretation": "MLP promotes target more" if avg_delta_diff > 0 else "MLP promotes compete more",
        }
        
        plog_always(f"  {cat_name}: target_Δ={avg_target:.3f}, compete_Δ={avg_compete_deltas}, "
                    f"diff={avg_delta_diff:.3f} ({cat_result['summary']['interpretation']})")
        
        results[cat_name] = cat_result
    
    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    plog_always(f"Phase 456: Margin Dynamics Verification")
    plog_always(f"Model: {model_name}, Round: {round_num}")
    
    # 加载模型
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    plog_always(f"  {info.model_class}, {info.n_layers} layers, d={info.d_model}")
    
    results = {
        "model": model_name,
        "round": round_num,
        "sign_convention": "ComponentMarginEffect = Margin_clean - Margin_zero_ablated (positive=promotes, negative=suppresses)",
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "mlp_type": info.mlp_type,
        },
    }
    
    # Exp1: Margin定义鲁棒性
    try:
        results["exp1_margin_robustness"] = exp1_margin_robustness(model, tokenizer, info, round_num)
    except Exception as e:
        plog_always(f"Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    plog_always(f"  GPU after Exp1: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # Exp2: 多模板验证
    try:
        results["exp2_multi_template"] = exp2_multi_template(model, tokenizer, info, round_num)
    except Exception as e:
        plog_always(f"Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    plog_always(f"  GPU after Exp2: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # Exp3: 全层margin扫描
    try:
        results["exp3_full_layer_scan"] = exp3_full_layer_margin_scan(model, tokenizer, info, round_num)
    except Exception as e:
        plog_always(f"Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    plog_always(f"  GPU after Exp3: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # Exp4: 类别路径翻转
    try:
        results["exp4_category_path"] = exp4_category_path_flipping(model, tokenizer, info, round_num)
    except Exception as e:
        plog_always(f"Exp4 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp5: 族级logit分解
    try:
        results["exp5_family_logit_decomposition"] = exp5_family_logit_decomposition(model, tokenizer, info, round_num)
    except Exception as e:
        plog_always(f"Exp5 FAILED: {e}")
        import traceback; traceback.print_exc()
    
    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase456_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    plog_always(f"Results saved to {out_path}")
    
    # 释放模型
    release_model(model)
    plog_always(f"Phase 456 {model_name} R{round_num} complete!")


if __name__ == "__main__":
    main()
