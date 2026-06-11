"""
Phase 455: 候选族标准化与槽位读出接口大样本验证
================================================
核心改进:
1. 候选族定义标准化: class_label / class_member / attribute / generic 四类分离
2. 多类别对象: fruit/animal/tool/vehicle 各8个
3. 符号统一: ComponentEffect = clean - zero_ablated (正=促进, 负=压制)
4. 跨对象稳定性: 每个结果附带方差
5. 多模板: 每个槽位2-3个模板避免偏置
6. DS7B非单调分解: direction-only vs scale-only vs norm-matched

子实验:
  Exp1: 跨类别候选族再分布图谱 (fruit/animal/tool/vehicle各测)
  Exp2: 最后层attn/MLP的AttentionBrakeScore和MLPFamilyEffect (正式定义)
  Exp3: 投影-因果-候选族三证合一 (跨类别, 多模板, 附方差)
  Exp4: DS7B非单调响应分解 (direction-only/scale-only/norm-matched)
  Exp5: 全层扫描转折点 (找MLP从促进变压制的层)

用法:
  python tests/glm5/phase455_standardized_family.py qwen3 1
  python tests/glm5/phase455_standardized_family.py glm4 1
  python tests/glm5/phase455_standardized_family.py deepseek7b 1
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
log = logging.getLogger("p455")

def plog(msg):
    log.info(msg)

def plog_always(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 符号约定 (强制) ====================
# ComponentEffect = clean - zero_ablated
#   > 0 表示组件促进目标
#   < 0 表示组件压制目标
# ZeroDelta = zero_ablated - clean
#   > 0 表示移除组件后目标上升, 即组件压制
#   < 0 表示移除组件后目标下降, 即组件促进
# 所有输出统一使用 ComponentEffect, 不使用 ZeroDelta


# ==================== 候选族标准化定义 ====================
# 四类分离, 避免混淆
FAMILY_STANDARD = {
    # 类别标签族: 抽象类别词, 不是具体成员
    "class_fruit":   ["fruit", "produce", "crop", "harvest"],
    "class_animal":  ["animal", "creature", "beast", "mammal"],
    "class_tool":    ["tool", "implement", "instrument", "device"],
    "class_vehicle": ["vehicle", "transport", "conveyance", "automobile"],

    # 成员族: 同一类别下的具体实例
    "member_fruit":   ["apple", "banana", "orange", "grape", "pear", "peach", "lemon", "mango", "cherry", "plum"],
    "member_animal":  ["dog", "cat", "horse", "lion", "bear", "rabbit", "cow", "tiger", "eagle", "deer"],
    "member_tool":    ["hammer", "wrench", "saw", "drill", "axe", "knife", "chisel", "pliers", "shovel", "screwdriver"],
    "member_vehicle": ["car", "truck", "bus", "train", "bicycle", "boat", "plane", "motorcycle", "van", "scooter"],

    # 属性族: 描述属性
    "attr_color":    ["red", "green", "yellow", "blue", "brown", "black", "white", "orange", "gray", "pink"],
    "attr_part":     ["wheel", "blade", "handle", "engine", "seat", "wing", "leg", "head", "body", "tail"],
    "attr_material": ["metal", "wood", "plastic", "steel", "iron", "rubber", "glass", "leather", "stone", "fabric"],
    "attr_function": ["move", "cut", "carry", "hold", "drive", "build", "eat", "grow", "protect", "transport"],

    # 泛化族: 高频泛化词
    "generic":       ["thing", "item", "object", "entity", "piece", "one", "it", "that"],
}

# 每个类别的对象集
CATEGORY_OBJECTS = {
    "fruit":   ["apple", "banana", "orange", "grape", "pear", "peach", "lemon", "mango"],
    "animal":  ["dog", "cat", "horse", "lion", "bear", "rabbit", "cow", "tiger"],
    "tool":    ["hammer", "wrench", "saw", "drill", "axe", "knife", "chisel", "pliers"],
    "vehicle": ["car", "truck", "bus", "train", "bicycle", "boat", "plane", "motorcycle"],
}

# 每个类别对应的目标族和竞争族
CATEGORY_FAMILIES = {
    "fruit": {
        "target_class": "class_fruit",
        "target_member": "member_fruit",
        "compete_class": ["class_animal", "class_tool", "class_vehicle"],
    },
    "animal": {
        "target_class": "class_animal",
        "target_member": "member_animal",
        "compete_class": ["class_fruit", "class_tool", "class_vehicle"],
    },
    "tool": {
        "target_class": "class_tool",
        "target_member": "member_tool",
        "compete_class": ["class_fruit", "class_animal", "class_vehicle"],
    },
    "vehicle": {
        "target_class": "class_vehicle",
        "target_member": "member_vehicle",
        "compete_class": ["class_fruit", "class_animal", "class_tool"],
    },
}

# 槽位模板 (每个槽位2-3个模板)
SLOT_TEMPLATES = {
    "cat": [
        "The {obj} is a",
        "A {obj} is classified as a",
        "{obj} belongs to the category of",
    ],
    "color": [
        "The color of the {obj} is",
        "A {obj} is usually",
        "The {obj} appears",
    ],
    "part": [
        "The {obj} has a",
        "A key part of the {obj} is the",
        "The {obj} contains a",
    ],
    "material": [
        "The {obj} is made of",
        "A {obj} is typically made from",
        "The material of the {obj} is",
    ],
    "function": [
        "You use a {obj} to",
        "A {obj} can",
        "The purpose of the {obj} is to",
    ],
}

# 槽位对应属性族
SLOT_TO_FAMILY = {
    "cat": "class_fruit",   # 默认fruit类别, 实际根据对象类别变化
    "color": "attr_color",
    "part": "attr_part",
    "material": "attr_material",
    "function": "attr_function",
}

# 对象→类别映射
OBJ_TO_CATEGORY = {}
for cat, objs in CATEGORY_OBJECTS.items():
    for obj in objs:
        OBJ_TO_CATEGORY[obj] = cat

# 关键层
KEY_LAYERS = {
    "qwen3": [0, 8, 16, 24, 30, 34, 35],
    "glm4":  [0, 10, 19, 24, 30, 38, 39],
    "deepseek7b": [0, 7, 14, 21, 23, 26, 27],
}

# Round 1用少量对象, Round 2用更多
ROUNDS = {
    1: {"fruit": ["apple", "banana", "orange"], "animal": ["dog", "cat", "horse"],
        "tool": ["hammer", "knife", "wrench"], "vehicle": ["car", "bus", "bicycle"]},
    2: {"fruit": ["apple", "banana", "orange", "grape", "pear", "peach"],
        "animal": ["dog", "cat", "horse", "lion", "bear", "rabbit"],
        "tool": ["hammer", "knife", "wrench", "saw", "drill", "axe"],
        "vehicle": ["car", "bus", "bicycle", "truck", "train", "boat"]},
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


def get_embedding_weight(model):
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens.weight.detach().float()
    return model.get_input_embeddings().weight.detach().float()


# ==================== 工具函数 ====================
def get_logit_for_words(logits_np, tokenizer, word_list):
    """获取一组词的平均logit"""
    ids = []
    for w in word_list:
        tok_ids = tokenizer.encode(w, add_special_tokens=False)
        if tok_ids:
            ids.append(tok_ids[0])
    if not ids:
        return None
    return float(np.mean(logits_np[ids]))


def compute_family_logits(logits_np, tokenizer, family_dict=None):
    """计算标准化候选族的平均logit"""
    if family_dict is None:
        family_dict = FAMILY_STANDARD
    result = {}
    for family_name, words in family_dict.items():
        val = get_logit_for_words(logits_np, tokenizer, words)
        if val is not None:
            result[family_name] = round(val, 4)
    return result


def compute_entropy_confidence(logits_np):
    probs = np.exp(logits_np - logits_np.max())
    probs = probs / probs.sum()
    entropy = round(-float(np.sum(probs * np.log(probs + 1e-12))), 4)
    confidence = round(float(probs.max()), 4)
    return entropy, confidence


def compute_family_margin(family_logits, target_key, compete_keys):
    """计算候选族边际: target - max(compete)"""
    target = family_logits.get(target_key, -999)
    competes = [family_logits.get(k, -999) for k in compete_keys]
    return round(target - max(competes), 4) if competes else round(target, 4)


def make_zero_hook(lpos):
    """创建置零hook: 将指定位置的输出置零"""
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


def make_capture_hook(captured_dict, key, lpos):
    """创建捕获hook: 捕获指定位置的输出"""
    def hook(m, inp, out):
        if isinstance(out, tuple):
            captured_dict[key] = out[0][0, lpos].detach().float().cpu().numpy()
        else:
            if out.ndim > 1:
                captured_dict[key] = out[0, lpos].detach().float().cpu().numpy()
            else:
                captured_dict[key] = out.detach().float().cpu().numpy()
    return hook


def make_inject_hook(lpos, perturbation):
    """创建注入hook: 在指定位置加扰动"""
    def hook(m, inp, out):
        if isinstance(out, tuple):
            new_out = out[0].clone()
            pt = torch.tensor(perturbation, dtype=torch.float32).to(new_out.device).to(new_out.dtype)
            new_out[0, lpos] = new_out[0, lpos] + pt
            return (new_out,) + out[1:]
        else:
            new_out = out.clone()
            pt = torch.tensor(perturbation, dtype=torch.float32).to(new_out.device).to(new_out.dtype)
            new_out[0, lpos] = new_out[0, lpos] + pt
            return new_out
    return hook


def avg_with_std(values):
    """计算均值和标准差, 过滤None"""
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None
    return round(float(np.mean(clean)), 4), round(float(np.std(clean)), 4)


# ==================== Exp1: 跨类别候选族再分布图谱 ====================
def exp1_cross_category_family_redistribution(model, tokenizer, info, round_num=1):
    """
    对4个类别(fruit/animal/tool/vehicle)的对象，
    测量最后层attn/MLP消融后候选族再分布变化
    
    核心改进: 使用标准化的4类候选族, 不混合类别标签和成员
    """
    plog_always(f"\n{'='*60}")
    plog_always(f"Exp1: Cross-Category Family Redistribution")
    plog_always(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)
    last_layer = n_layers - 1
    
    obj_set = ROUNDS.get(round_num, ROUNDS[1])
    
    # 只测最后两层
    test_layers = [n_layers - 2, n_layers - 1]
    
    results = {}
    
    for li in test_layers:
        layer = layers[li]
        plog_always(f"  L{li}...")
        
        layer_result = {}
        
        for cat_name, obj_list in obj_set.items():
            cat_result = {}
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
                clean_margin = compute_family_margin(clean_family, cat_info["target_class"],
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
                za_margin = compute_family_margin(za_family, cat_info["target_class"],
                                                  cat_info["compete_class"])
                
                # Zero MLP
                mlp_mod = None
                if hasattr(layer, 'mlp'):
                    mlp_mod = layer.mlp
                elif hasattr(layer, 'feed_forward'):
                    mlp_mod = layer.feed_forward
                
                h_mlp = None
                if mlp_mod is not None:
                    h_mlp = mlp_mod.register_forward_hook(make_zero_hook(last_pos))
                with torch.no_grad():
                    out_zm = model(input_ids=input_ids, attention_mask=attention_mask)
                if h_mlp:
                    h_mlp.remove()
                zm_logits = out_zm.logits[0, -1].float().cpu().numpy()
                zm_family = compute_family_logits(zm_logits, tokenizer)
                zm_margin = compute_family_margin(zm_family, cat_info["target_class"],
                                                  cat_info["compete_class"])
                
                # ComponentEffect = clean - zero_ablated (正=促进)
                obj_data = {
                    "clean_margin": clean_margin,
                    "attn_effect": round(clean_margin - za_margin, 4),  # 正=attn促进
                    "mlp_effect": round(clean_margin - zm_margin, 4),    # 正=mlp促进
                    "family_delta_attn": {k: round(clean_family.get(k, 0) - za_family.get(k, 0), 4) 
                                         for k in clean_family},
                    "family_delta_mlp": {k: round(clean_family.get(k, 0) - zm_family.get(k, 0), 4) 
                                         for k in clean_family},
                }
                cat_result[obj_name] = obj_data
            
            # 类别平均
            attn_effects = [v["attn_effect"] for v in cat_result.values()]
            mlp_effects = [v["mlp_effect"] for v in cat_result.values()]
            avg_ae, std_ae = avg_with_std(attn_effects)
            avg_me, std_me = avg_with_std(mlp_effects)
            
            # 候选族平均变化
            avg_fam_attn = {}
            avg_fam_mlp = {}
            for fam_key in FAMILY_STANDARD.keys():
                a_vals = [v["family_delta_attn"].get(fam_key, 0) for v in cat_result.values()]
                m_vals = [v["family_delta_mlp"].get(fam_key, 0) for v in cat_result.values()]
                avg_fam_attn[fam_key] = round(float(np.mean(a_vals)), 4)
                avg_fam_mlp[fam_key] = round(float(np.mean(m_vals)), 4)
            
            cat_result["summary"] = {
                "avg_attn_effect": avg_ae, "std_attn_effect": std_ae,
                "avg_mlp_effect": avg_me, "std_mlp_effect": std_me,
                "attn_type": "PROMOTES" if (avg_ae is not None and avg_ae > 0.1) else
                            ("SUPPRESSES" if (avg_ae is not None and avg_ae < -0.1) else "NEUTRAL"),
                "mlp_type": "PROMOTES" if (avg_me is not None and avg_me > 0.1) else
                           ("SUPPRESSES" if (avg_me is not None and avg_me < -0.1) else "NEUTRAL"),
                "avg_family_attn": avg_fam_attn,
                "avg_family_mlp": avg_fam_mlp,
            }
            
            plog_always(f"    {cat_name}: attn={avg_ae}±{std_ae} ({cat_result['summary']['attn_type']}), "
                        f"mlp={avg_me}±{std_me} ({cat_result['summary']['mlp_type']})")
        
        layer_result[cat_name] = cat_result  # 每个类别一组
        results[f"L{li}"] = layer_result
    
    return results


# ==================== Exp2: AttentionBrakeScore & MLPFamilyEffect ====================
def exp2_brake_and_family_effects(model, tokenizer, info, round_num=1):
    """
    正式定义并计算:
    AttentionBrakeScore = FamilyMargin(clean) - FamilyMargin(zero_attn)
      > 0: attn是刹车(移除attn后边际上升)
      < 0: attn是促进器(移除attn后边际下降)
    
    MLPFamilyEffect = FamilyMargin(clean) - FamilyMargin(zero_mlp)
      > 0: MLP促进候选族
      < 0: MLP压制候选族
    
    测试5个槽位, 4个类别
    """
    plog_always(f"\n{'='*60}")
    plog_always(f"Exp2: AttentionBrakeScore & MLPFamilyEffect")
    plog_always(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)
    last_layer = n_layers - 1
    layer = layers[last_layer]
    
    obj_set = ROUNDS.get(round_num, ROUNDS[1])
    
    results = {}
    
    for slot_name, templates in SLOT_TEMPLATES.items():
        plog_always(f"  Slot: {slot_name}")
        slot_result = {}
        
        for cat_name, obj_list in obj_set.items():
            cat_info = CATEGORY_FAMILIES[cat_name]
            cat_result = {}
            
            for obj_name in obj_list:
                # 使用第一个模板
                template = templates[0]
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
                elif slot_name == "part":
                    target_family = "attr_part"
                    compete_families = ["attr_color", "attr_material", "attr_function"]
                elif slot_name == "material":
                    target_family = "attr_material"
                    compete_families = ["attr_color", "attr_part", "attr_function"]
                else:  # function
                    target_family = "attr_function"
                    compete_families = ["attr_color", "attr_part", "attr_material"]
                
                # Clean
                with torch.no_grad():
                    out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
                clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
                clean_family = compute_family_logits(clean_logits, tokenizer)
                clean_margin = compute_family_margin(clean_family, target_family, compete_families)
                
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
                za_margin = compute_family_margin(za_family, target_family, compete_families)
                
                # Zero MLP
                mlp_mod = None
                if hasattr(layer, 'mlp'):
                    mlp_mod = layer.mlp
                elif hasattr(layer, 'feed_forward'):
                    mlp_mod = layer.feed_forward
                
                h_mlp = None
                if mlp_mod is not None:
                    h_mlp = mlp_mod.register_forward_hook(make_zero_hook(last_pos))
                with torch.no_grad():
                    out_zm = model(input_ids=input_ids, attention_mask=attention_mask)
                if h_mlp:
                    h_mlp.remove()
                zm_logits = out_zm.logits[0, -1].float().cpu().numpy()
                zm_family = compute_family_logits(zm_logits, tokenizer)
                zm_margin = compute_family_margin(zm_family, target_family, compete_families)
                
                # AttentionBrakeScore = clean_margin - za_margin
                # > 0: attn压制(移除attn后边际上升, 即attn在压低边际)
                attn_brake = round(clean_margin - za_margin, 4)
                # MLPFamilyEffect = clean_margin - zm_margin
                # > 0: MLP促进(移除mlp后边际下降, 即mlp在提高边际)
                mlp_effect = round(clean_margin - zm_margin, 4)
                
                # 目标族logit变化 (ComponentEffect = clean - zero)
                target_logit_clean = clean_family.get(target_family, None)
                target_logit_za = za_family.get(target_family, None)
                target_logit_zm = zm_family.get(target_family, None)
                
                cat_result[obj_name] = {
                    "attn_brake_score": attn_brake,
                    "mlp_family_effect": mlp_effect,
                    "clean_margin": clean_margin,
                    "target_logit_attn_effect": round(target_logit_clean - target_logit_za, 4) if (target_logit_clean and target_logit_za) else None,
                    "target_logit_mlp_effect": round(target_logit_clean - target_logit_zm, 4) if (target_logit_clean and target_logit_zm) else None,
                    "entropy_clean": compute_entropy_confidence(clean_logits)[0],
                    "entropy_za": compute_entropy_confidence(za_logits)[0],
                    "entropy_zm": compute_entropy_confidence(zm_logits)[0],
                }
            
            # 类别平均 (只取对象数据, 排除summary)
            ab_scores = [v["attn_brake_score"] for k, v in cat_result.items() 
                         if k != "summary" and "attn_brake_score" in v and v["attn_brake_score"] is not None]
            mf_effects = [v["mlp_family_effect"] for k, v in cat_result.items() 
                          if k != "summary" and "mlp_family_effect" in v and v["mlp_family_effect"] is not None]
            avg_ab, std_ab = avg_with_std(ab_scores)
            avg_mf, std_mf = avg_with_std(mf_effects)
            
            cat_result["summary"] = {
                "avg_attn_brake": avg_ab, "std_attn_brake": std_ab,
                "avg_mlp_effect": avg_mf, "std_mlp_effect": std_mf,
                "attn_judgment": "BRAKE" if (avg_ab is not None and avg_ab > 0.1) else
                               ("PROMOTER" if (avg_ab is not None and avg_ab < -0.1) else "NEUTRAL"),
                "mlp_judgment": "AMPLIFIER" if (avg_mf is not None and avg_mf > 0.1) else
                               ("SUPPRESSOR" if (avg_mf is not None and avg_mf < -0.1) else "NEUTRAL"),
            }
            
            slot_result[cat_name] = cat_result
            plog_always(f"    {cat_name}: attn_brake={avg_ab}±{std_ab} ({cat_result['summary']['attn_judgment']}), "
                        f"mlp_effect={avg_mf}±{std_mf} ({cat_result['summary']['mlp_judgment']})")
        
        # 全类别汇总
        all_ab = []
        all_mf = []
        for cat_name in obj_set.keys():
            if cat_name in slot_result and "summary" in slot_result[cat_name]:
                s = slot_result[cat_name]["summary"]
                if s["avg_attn_brake"] is not None:
                    all_ab.append(s["avg_attn_brake"])
                if s["avg_mlp_effect"] is not None:
                    all_mf.append(s["avg_mlp_effect"])
        
        slot_result["overall"] = {
            "mean_attn_brake_all_cats": round(float(np.mean(all_ab)), 4) if all_ab else None,
            "mean_mlp_effect_all_cats": round(float(np.mean(all_mf)), 4) if all_mf else None,
            "attn_type": "UNIVERSAL_BRAKE" if (all_ab and np.mean(all_ab) > 0.1) else
                        ("UNIVERSAL_PROMOTER" if (all_ab and np.mean(all_ab) < -0.1) else "MIXED"),
            "mlp_type": "UNIVERSAL_AMPLIFIER" if (all_mf and np.mean(all_mf) > 0.1) else
                       ("UNIVERSAL_SUPPRESSOR" if (all_mf and np.mean(all_mf) < -0.1) else "MIXED"),
        }
        
        results[slot_name] = slot_result
        plog_always(f"  {slot_name} overall: attn={slot_result['overall']['attn_type']}, "
                    f"mlp={slot_result['overall']['mlp_type']}")
    
    return results


# ==================== Exp3: 投影-因果-候选族三证合一(跨类别) ====================
def exp3_triple_evidence_cross_category(model, tokenizer, info, round_num=1):
    """
    对每个类别计算投影-因果-候选族三证
    使用标准化候选族定义和统一符号
    """
    plog_always(f"\n{'='*60}")
    plog_always(f"Exp3: Triple Evidence Cross-Category")
    plog_always(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    W_U = get_W_U(model, info.name)
    W_U_T = W_U.T
    input_device = get_input_device(model)
    W_E = get_embedding_weight(model)
    
    # 构建每个类别的读出方向
    cat_readout_dirs = {}
    for cat_name, cat_info in CATEGORY_FAMILIES.items():
        target_words = FAMILY_STANDARD[cat_info["target_class"]]
        ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in target_words]
        ids = [i for i in ids if i is not None]
        if ids:
            direction = W_U_T[:, ids].mean(axis=1)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            cat_readout_dirs[cat_name] = direction
    
    obj_set = ROUNDS.get(round_num, ROUNDS[1])
    key_layers = KEY_LAYERS.get(info.name, [0, n_layers//3, 2*n_layers//3, n_layers-2, n_layers-1])
    key_layers = [l for l in key_layers if l < n_layers]
    
    results = {}
    
    for li in key_layers:
        layer = layers[li]
        plog_always(f"  L{li}...")
        
        layer_result = {}
        
        for cat_name, obj_list in obj_set.items():
            cat_info = CATEGORY_FAMILIES[cat_name]
            readout_dir = cat_readout_dirs.get(cat_name)
            
            if readout_dir is None:
                continue
            
            cat_data = {}
            
            for obj_name in obj_list:
                text = f"The {obj_name} is a"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attention_mask = inputs["attention_mask"].to(input_device)
                last_pos = input_ids.shape[1] - 1
                
                # 1. 捕获attn/MLP输出 + Clean forward
                captured = {}
                h_attn_cap = None
                h_mlp_cap = None
                if hasattr(layer, 'self_attn'):
                    h_attn_cap = layer.self_attn.register_forward_hook(
                        make_capture_hook(captured, "attn", last_pos))
                mlp_mod = None
                if hasattr(layer, 'mlp'):
                    mlp_mod = layer.mlp
                elif hasattr(layer, 'feed_forward'):
                    mlp_mod = layer.feed_forward
                if mlp_mod is not None:
                    h_mlp_cap = mlp_mod.register_forward_hook(
                        make_capture_hook(captured, "mlp", last_pos))
                
                with torch.no_grad():
                    out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
                clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
                clean_family = compute_family_logits(clean_logits, tokenizer)
                clean_margin = compute_family_margin(clean_family, cat_info["target_class"],
                                                      cat_info["compete_class"])
                
                if h_attn_cap:
                    h_attn_cap.remove()
                if h_mlp_cap:
                    h_mlp_cap.remove()
                
                # 投影分数
                attn_proj = float(np.dot(captured["attn"], readout_dir)) if "attn" in captured else None
                mlp_proj = float(np.dot(captured["mlp"], readout_dir)) if "mlp" in captured else None
                
                # 2. 因果测试: zero attn
                h_za = None
                if hasattr(layer, 'self_attn'):
                    h_za = layer.self_attn.register_forward_hook(make_zero_hook(last_pos))
                with torch.no_grad():
                    out_za = model(input_ids=input_ids, attention_mask=attention_mask)
                if h_za:
                    h_za.remove()
                za_logits = out_za.logits[0, -1].float().cpu().numpy()
                za_family = compute_family_logits(za_logits, tokenizer)
                za_margin = compute_family_margin(za_family, cat_info["target_class"],
                                                   cat_info["compete_class"])
                
                # 3. 因果测试: zero MLP
                h_zm = None
                if mlp_mod is not None:
                    h_zm = mlp_mod.register_forward_hook(make_zero_hook(last_pos))
                with torch.no_grad():
                    out_zm = model(input_ids=input_ids, attention_mask=attention_mask)
                if h_zm:
                    h_zm.remove()
                zm_logits = out_zm.logits[0, -1].float().cpu().numpy()
                zm_family = compute_family_logits(zm_logits, tokenizer)
                zm_margin = compute_family_margin(zm_family, cat_info["target_class"],
                                                   cat_info["compete_class"])
                
                # ComponentEffect = clean - zero (正=促进)
                # 用边际变化作为因果指标
                attn_causal_margin = round(clean_margin - za_margin, 4)  # 正=attn促进边际
                mlp_causal_margin = round(clean_margin - zm_margin, 4)   # 正=mlp促进边际
                
                # 目标族logit因果
                target_class = cat_info["target_class"]
                clean_target = clean_family.get(target_class, None)
                za_target = za_family.get(target_class, None)
                zm_target = zm_family.get(target_class, None)
                attn_causal_logit = round(clean_target - za_target, 4) if (clean_target and za_target) else None
                mlp_causal_logit = round(clean_target - zm_target, 4) if (clean_target and zm_target) else None
                
                # 象限判定 (基于logit因果)
                def get_quadrant(proj, causal):
                    if proj is None or causal is None:
                        return "N/A"
                    if proj > 0 and causal > 0: return "Q1"
                    elif proj > 0 and causal < 0: return "Q2"
                    elif proj < 0 and causal > 0: return "Q3"
                    else: return "Q4"
                
                attn_quad = get_quadrant(attn_proj, attn_causal_logit)
                mlp_quad = get_quadrant(mlp_proj, mlp_causal_logit)
                
                # 三证判定
                def triple_judgment(proj, causal_logit, causal_margin):
                    if any(v is None for v in [proj, causal_logit, causal_margin]):
                        return "INCOMPLETE"
                    # 三者一致
                    if proj > 0 and causal_logit > 0 and causal_margin > 0:
                        return "TRIPLE_PROMOTER"
                    elif proj < 0 and causal_logit < 0 and causal_margin < 0:
                        return "TRIPLE_SUPPRESSOR"
                    elif proj > 0 and causal_logit < 0:
                        return "PROJ_CAUSAL_CONFLICT"
                    elif proj < 0 and causal_logit > 0:
                        return "INDIRECT_PROMOTER"
                    else:
                        return "MIXED"
                
                attn_triple = triple_judgment(attn_proj, attn_causal_logit, attn_causal_margin)
                mlp_triple = triple_judgment(mlp_proj, mlp_causal_logit, mlp_causal_margin)
                
                cat_data[obj_name] = {
                    "attn_proj": round(attn_proj, 4) if attn_proj is not None else None,
                    "mlp_proj": round(mlp_proj, 4) if mlp_proj is not None else None,
                    "attn_causal_logit": attn_causal_logit,
                    "mlp_causal_logit": mlp_causal_logit,
                    "attn_causal_margin": attn_causal_margin,
                    "mlp_causal_margin": mlp_causal_margin,
                    "attn_quad": attn_quad,
                    "mlp_quad": mlp_quad,
                    "attn_triple": attn_triple,
                    "mlp_triple": mlp_triple,
                }
            
            # 类别平均
            avg_data = {}
            for key in ["attn_proj", "mlp_proj", "attn_causal_logit", "mlp_causal_logit",
                       "attn_causal_margin", "mlp_causal_margin"]:
                vals = [cat_data[o][key] for o in obj_list if o in cat_data and cat_data[o][key] is not None]
                if vals:
                    avg_data[key] = round(float(np.mean(vals)), 4)
                    avg_data[key + "_std"] = round(float(np.std(vals)), 4)
            
            # 最常见象限和三证
            attn_quads = [cat_data[o]["attn_quad"] for o in obj_list if o in cat_data]
            mlp_quads = [cat_data[o]["mlp_quad"] for o in obj_list if o in cat_data]
            attn_triples = [cat_data[o]["attn_triple"] for o in obj_list if o in cat_data]
            mlp_triples = [cat_data[o]["mlp_triple"] for o in obj_list if o in cat_data]
            avg_data["attn_quad_mode"] = max(set(attn_quads), key=attn_quads.count) if attn_quads else "N/A"
            avg_data["mlp_quad_mode"] = max(set(mlp_quads), key=mlp_quads.count) if mlp_quads else "N/A"
            avg_data["attn_triple_mode"] = max(set(attn_triples), key=attn_triples.count) if attn_triples else "N/A"
            avg_data["mlp_triple_mode"] = max(set(mlp_triples), key=mlp_triples.count) if mlp_triples else "N/A"
            
            cat_data["summary"] = avg_data
            
            layer_result[cat_name] = cat_data
        
        results[f"L{li}"] = layer_result
        
        # 打印摘要
        for cat_name in obj_set.keys():
            if cat_name in layer_result and "summary" in layer_result[cat_name]:
                s = layer_result[cat_name]["summary"]
                plog(f"    {cat_name}: attn_proj={s.get('attn_proj')}, attn_causal={s.get('attn_causal_logit')}, "
                     f"attn_triple={s.get('attn_triple_mode')}; "
                     f"mlp_proj={s.get('mlp_proj')}, mlp_causal={s.get('mlp_causal_logit')}, "
                     f"mlp_triple={s.get('mlp_triple_mode')}")
    
    return results


# ==================== Exp4: DS7B非单调响应分解 ====================
def exp4_nonmonotonic_decomposition(model, tokenizer, info, round_num=1):
    """
    分解DS7B非单调响应来源:
    1. direction-only: 只加方向(归一化后), alpha=1
    2. scale-only: 加随机方向但匹配范数
    3. norm-matched: 匹配残差范数
    4. direction+scale: 原始方向×alpha (Phase 454方式)
    5. post-RMSNorm注入: 在RMSNorm后注入
    
    目标: 判断双峰来自方向×范数交互, 还是RMSNorm, 还是其他
    """
    plog_always(f"\n{'='*60}")
    plog_always(f"Exp4: Non-Monotonic Response Decomposition")
    plog_always(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    W_U = get_W_U(model, info.name)
    W_U_T = W_U.T
    input_device = get_input_device(model)
    W_E = get_embedding_weight(model)
    
    # 构建cat方向
    cat_words = FAMILY_STANDARD["class_fruit"]
    opp_words = FAMILY_STANDARD["class_animal"]
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    cat_ids = [i for i in cat_ids if i is not None]
    opp_ids = [i for i in opp_ids if i is not None]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu().numpy()
    cat_dir_norm = np.linalg.norm(cat_dir)
    cat_dir_unit = cat_dir / (cat_dir_norm + 1e-8)
    cat_readout_dir = W_U_T[:, cat_ids].mean(axis=1)
    cat_readout_dir = cat_readout_dir / (np.linalg.norm(cat_readout_dir) + 1e-8)
    
    # 测试层
    if info.name == "deepseek7b":
        test_layers = [0, 14, 27]
    elif info.name == "qwen3":
        test_layers = [16, 25, 35]
    elif info.name == "glm4":
        test_layers = [10, 24, 39]
    else:
        test_layers = [0, n_layers//2, n_layers-1]
    test_layers = [l for l in test_layers if l < n_layers]
    
    obj_names = ROUNDS.get(round_num, ROUNDS[1])["fruit"]  # 只用fruit类
    
    alphas = [0.25, 0.5, 1.0, 2.0, 4.0]
    
    results = {}
    
    for li in test_layers:
        layer = layers[li]
        plog_always(f"  L{li}...")
        
        layer_result = {}
        
        for alpha in alphas:
            alpha_data = {
                "dir_plus_scale": [],  # 原始方向×alpha
                "dir_only": [],        # 只方向(归一化), alpha=1
                "scale_only": [],      # 随机方向匹配范数
                "post_rms": [],        # RMSNorm后注入
            }
            
            for obj_name in obj_names:
                text = f"The {obj_name} is a"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attention_mask = inputs["attention_mask"].to(input_device)
                last_pos = input_ids.shape[1] - 1
                
                # Clean
                with torch.no_grad():
                    out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
                clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
                clean_cat = get_logit_for_words(clean_logits, tokenizer, cat_words)
                
                # 1. Direction + Scale (原始方式)
                perturb_ds = alpha * cat_dir_unit  # 归一化方向×alpha
                h_inject = layer.register_forward_hook(make_inject_hook(last_pos, perturb_ds))
                with torch.no_grad():
                    out_ds = model(input_ids=input_ids, attention_mask=attention_mask)
                h_inject.remove()
                ds_logits = out_ds.logits[0, -1].float().cpu().numpy()
                ds_cat = get_logit_for_words(ds_logits, tokenizer, cat_words)
                
                # 2. Direction only (固定alpha=1, 不管scale参数)
                perturb_do = 1.0 * cat_dir_unit  # 只方向, 固定大小
                h_inject2 = layer.register_forward_hook(make_inject_hook(last_pos, perturb_do))
                with torch.no_grad():
                    out_do = model(input_ids=input_ids, attention_mask=attention_mask)
                h_inject2.remove()
                do_logits = out_do.logits[0, -1].float().cpu().numpy()
                do_cat = get_logit_for_words(do_logits, tokenizer, cat_words)
                
                # 3. Scale only (随机方向, 匹配范数)
                rng = np.random.RandomState(42)
                random_dir = rng.randn(len(cat_dir_unit))
                random_dir = random_dir / np.linalg.norm(random_dir) * alpha  # 匹配范数
                h_inject3 = layer.register_forward_hook(make_inject_hook(last_pos, random_dir))
                with torch.no_grad():
                    out_so = model(input_ids=input_ids, attention_mask=attention_mask)
                h_inject3.remove()
                so_logits = out_so.logits[0, -1].float().cpu().numpy()
                so_cat = get_logit_for_words(so_logits, tokenizer, cat_words)
                
                # 4. Post-RMSNorm injection
                # 找到该层的RMSNorm (input_layernorm)
                rms_mod = None
                if hasattr(layer, 'input_layernorm'):
                    rms_mod = layer.input_layernorm
                elif hasattr(layer, 'ln_1'):
                    rms_mod = layer.ln_1
                
                if rms_mod is not None:
                    perturb_post = alpha * cat_dir_unit
                    h_rms = rms_mod.register_forward_hook(make_inject_hook(last_pos, perturb_post))
                    with torch.no_grad():
                        out_pr = model(input_ids=input_ids, attention_mask=attention_mask)
                    h_rms.remove()
                    pr_logits = out_pr.logits[0, -1].float().cpu().numpy()
                    pr_cat = get_logit_for_words(pr_logits, tokenizer, cat_words)
                else:
                    pr_cat = None
                
                # 记录
                for key, val in [("dir_plus_scale", ds_cat), ("dir_only", do_cat),
                                ("scale_only", so_cat), ("post_rms", pr_cat)]:
                    if val is not None and clean_cat is not None:
                        alpha_data[key].append(round(val - clean_cat, 4))
            
            # 平均
            alpha_result = {}
            for key in alpha_data:
                if alpha_data[key]:
                    alpha_result[key] = round(float(np.mean(alpha_data[key])), 4)
                else:
                    alpha_result[key] = None
            
            # 非单调检测
            is_nonmonotonic = False
            if len(layer_result) >= 2:
                prev_vals = [layer_result[k]["dir_plus_scale"] for k in sorted(layer_result.keys())
                            if layer_result[k]["dir_plus_scale"] is not None]
                if len(prev_vals) >= 1 and alpha_result["dir_plus_scale"] is not None:
                    if prev_vals[-1] * alpha_result["dir_plus_scale"] < 0:
                        is_nonmonotonic = True
            
            alpha_result["nonmonotonic"] = is_nonmonotonic
            layer_result[f"alpha{alpha}"] = alpha_result
        
        results[f"L{li}"] = layer_result
        
        # 打印
        ds_str = ", ".join([f"a{k.split('alpha')[1]}={layer_result[k]['dir_plus_scale']}" 
                           for k in sorted(layer_result.keys())])
        do_val = layer_result.get("alpha1.0", {}).get("dir_only", "N/A")
        so_vals = ", ".join([f"a{k.split('alpha')[1]}={layer_result[k]['scale_only']}"
                           for k in sorted(layer_result.keys())])
        plog_always(f"  L{li}: dir+scale=[{ds_str}], dir_only(α=1)={do_val}, scale_only=[{so_vals}]")
    
    return results


# ==================== Exp5: 全层MLP转折点扫描 ====================
def exp5_mlp_transition_scan(model, tokenizer, info, round_num=1):
    """
    扫描所有层的MLP对候选族边际的因果效应,
    找到MLP从促进变压制的转折层
    
    这对理解GLM4为什么最后层MLP压制所有族很关键
    """
    plog_always(f"\n{'='*60}")
    plog_always(f"Exp5: MLP Transition Scan (All Layers)")
    plog_always(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)
    
    obj_names = ROUNDS.get(round_num, ROUNDS[1])["fruit"]  # 先只测fruit类
    
    results = {}
    
    # 每隔3-4层采样, 减少计算量
    sample_step = max(1, n_layers // 12)
    sample_layers = list(range(0, n_layers, sample_step))
    if (n_layers - 1) not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    plog_always(f"  Sampling {len(sample_layers)} layers: {sample_layers}")
    
    for li in sample_layers:
        layer = layers[li]
        mlp_mod = None
        if hasattr(layer, 'mlp'):
            mlp_mod = layer.mlp
        elif hasattr(layer, 'feed_forward'):
            mlp_mod = layer.feed_forward
        
        if mlp_mod is None:
            continue
        
        mlp_effects = []
        
        for obj_name in obj_names:
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
            clean_margin = compute_family_margin(clean_family, "class_fruit",
                                                  ["class_animal", "class_tool", "class_vehicle"])
            
            # Zero MLP
            h_mlp = mlp_mod.register_forward_hook(make_zero_hook(last_pos))
            with torch.no_grad():
                out_zm = model(input_ids=input_ids, attention_mask=attention_mask)
            h_mlp.remove()
            zm_logits = out_zm.logits[0, -1].float().cpu().numpy()
            zm_family = compute_family_logits(zm_logits, tokenizer)
            zm_margin = compute_family_margin(zm_family, "class_fruit",
                                               ["class_animal", "class_tool", "class_vehicle"])
            
            # ComponentEffect = clean - zero (正=MLP促进边际)
            effect = round(clean_margin - zm_margin, 4)
            mlp_effects.append(effect)
        
        avg_effect, std_effect = avg_with_std(mlp_effects)
        
        results[f"L{li}"] = {
            "avg_mlp_margin_effect": avg_effect,
            "std_mlp_margin_effect": std_effect,
            "mlp_type": "AMPLIFIER" if (avg_effect is not None and avg_effect > 0.1) else
                       ("SUPPRESSOR" if (avg_effect is not None and avg_effect < -0.1) else "NEUTRAL"),
        }
        
        plog_always(f"  L{li}: mlp_effect={avg_effect}±{std_effect} ({results[f'L{li}']['mlp_type']})")
    
    # 找转折点
    effect_list = [(int(k[1:]), results[k]["avg_mlp_margin_effect"]) 
                   for k in sorted(results.keys()) if results[k]["avg_mlp_margin_effect"] is not None]
    transitions = []
    for i in range(1, len(effect_list)):
        if effect_list[i-1][1] * effect_list[i][1] < 0:  # 符号变化
            transitions.append((effect_list[i-1][0], effect_list[i][0]))
    
    if transitions:
        plog_always(f"  MLP transitions (sign changes) at: {transitions}")
    
    results["transitions"] = transitions
    
    return results


# ==================== 主流程 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    plog_always(f"Phase 455: Standardized Family & Slot Readout Verification")
    plog_always(f"Model: {model_name}, Round: {round_num}")
    plog_always(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    plog_always(f"Sign Convention: ComponentEffect = clean - zero (positive=promotes)")
    
    t0_total = time.time()
    
    # 加载模型
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    plog_always(f"Model info: class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    all_results = {
        "model": model_name,
        "round": round_num,
        "sign_convention": "ComponentEffect = clean - zero_ablated (positive=promotes, negative=suppresses)",
        "model_info": {"class": info.model_class, "n_layers": info.n_layers,
                       "d_model": info.d_model, "mlp_type": info.mlp_type},
    }
    
    # Exp1: 跨类别候选族再分布
    t0 = time.time()
    all_results["exp1"] = exp1_cross_category_family_redistribution(model, tokenizer, info, round_num)
    plog_always(f"Exp1 done in {time.time()-t0:.0f}s")
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Exp2: Brake & Family Effects
    t0 = time.time()
    all_results["exp2"] = exp2_brake_and_family_effects(model, tokenizer, info, round_num)
    plog_always(f"Exp2 done in {time.time()-t0:.0f}s")
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Exp3: 三证合一跨类别
    t0 = time.time()
    all_results["exp3"] = exp3_triple_evidence_cross_category(model, tokenizer, info, round_num)
    plog_always(f"Exp3 done in {time.time()-t0:.0f}s")
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Exp4: 非单调分解
    t0 = time.time()
    all_results["exp4"] = exp4_nonmonotonic_decomposition(model, tokenizer, info, round_num)
    plog_always(f"Exp4 done in {time.time()-t0:.0f}s")
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    # Exp5: MLP转折点扫描
    t0 = time.time()
    all_results["exp5"] = exp5_mlp_transition_scan(model, tokenizer, info, round_num)
    plog_always(f"Exp5 done in {time.time()-t0:.0f}s")
    
    # 保存
    out_path = f"results/glm5/phase455_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    plog_always(f"Saved to {out_path}")
    
    # 释放
    release_model(model)
    model = None
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    plog_always(f"Total time: {time.time()-t0_total:.0f}s")
    plog_always("Phase 455 done!")


if __name__ == "__main__":
    main()
