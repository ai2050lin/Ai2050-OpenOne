"""
Phase 454: 候选族再分布与投影-因果-行为三证合一
================================================
核心目标:
1. Exp1: 候选族级别读出图谱 — 组件消融后候选族分布变化
2. Exp2: 跨槽位最后层attn压制测试 — category/color/part/material/function
3. Exp3: 多槽位读出接口画像(修复Phase453 Exp4 bug)
4. Exp4: DS7B双峰层注入复验(修复Phase453 Exp5 bug)
5. Exp5: 投影-因果-候选族三证合一

用法:
  python tests/glm5/phase454_candidate_family.py qwen3 1
  python tests/glm5/phase454_candidate_family.py glm4 1
  python tests/glm5/phase454_candidate_family.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import os, gc, time, json, logging, math, glob
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger("p454")

def plog(msg):
    log.info(msg)

def plog_always(msg):
    """强制输出到stdout（避免被logging级别过滤）"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


# ==================== 候选族定义 ====================
# 每个槽位对应多组候选族，每组包含同义词
CANDIDATE_FAMILIES = {
    "fruit": ["fruit", "apple", "banana", "orange", "grape", "pear", "peach", "mango", "lemon", "cherry", "plum", "berry"],
    "tool":  ["tool", "hammer", "wrench", "screwdriver", "pliers", "saw", "drill", "chisel", "axe", "knife", "blade", "implement"],
    "animal": ["animal", "dog", "cat", "bird", "fish", "horse", "cow", "bear", "lion", "tiger", "elephant", "creature"],
    "vehicle": ["vehicle", "car", "truck", "bus", "train", "plane", "boat", "ship", "bicycle", "motorcycle", "van", "automobile"],
    "food":  ["food", "meal", "dish", "snack", "cuisine", "produce", "grocery", "edible", "nourishment", "sustenance", "fare", "provision"],
    "object": ["object", "thing", "item", "entity", "article", "piece", "substance", "material", "element", "unit", "artifact", "device"],
    "plant": ["plant", "tree", "flower", "shrub", "herb", "bush", "vegetation", "flora", "foliage", "vine", "weed", "sapling"],
}

# 槽位关键词（用于测量logit）
SLOT_FAMILIES = {
    "cat":     {"target": "fruit", "compete": ["animal", "tool", "vehicle", "object"]},
    "color":   {"target": "color_words", "compete": []},  # 单独处理
    "part":    {"target": "part_words", "compete": []},
    "material":{"target": "mat_words", "compete": []},
    "function":{"target": "func_words", "compete": []},
}

# 具体槽位词汇
SLOT_WORDS = {
    "cat": ["fruit", "food", "produce"],
    "opp_cat": ["animal", "dog", "cat"],
    "color": ["red", "green", "yellow", "orange", "brown"],
    "part": ["seed", "skin", "core", "stem", "peel", "flesh"],
    "material": ["organic", "natural", "fresh", "plant", "biological"],
    "function": ["eat", "cook", "grow", "harvest", "taste", "peel"],
    "habitat": ["tree", "garden", "farm", "orchard", "field"],
}

# 扩展候选族定义（用于Exp1候选族分布分析）
FAMILY_WORDS = {
    "fruit":   ["fruit", "apple", "banana", "orange", "grape", "pear", "peach"],
    "animal":  ["animal", "dog", "cat", "bird", "fish", "horse", "cow"],
    "tool":    ["tool", "hammer", "wrench", "screwdriver", "knife", "blade"],
    "vehicle": ["vehicle", "car", "truck", "bus", "train", "plane", "boat"],
    "food":    ["food", "meal", "dish", "snack", "cuisine", "produce"],
    "object":  ["object", "thing", "item", "entity", "article"],
    "color_r": ["red", "crimson", "scarlet", "ruby"],
    "color_g": ["green", "emerald", "olive", "lime"],
    "color_y": ["yellow", "golden", "amber", "lemon"],
    "part_f":  ["seed", "core", "pit", "kernel", "stone"],
    "part_s":  ["skin", "peel", "rind", "shell", "bark"],
    "mat_org": ["organic", "natural", "biological", "living", "fresh"],
    "mat_art": ["metal", "plastic", "synthetic", "artificial", "manufactured"],
    "func_eat":["eat", "consume", "taste", "bite", "chew", "swallow"],
    "func_use":["use", "apply", "utilize", "employ", "operate"],
    "high_freq":["the", "a", "an", "is", "was", "it", "that", "this", "of", "and"],
}

OBJ_NAMES = ["apple", "orange", "banana", "grape", "lemon", "peach", "pear", "mango"]
OBJ_NAMES_SHORT = ["apple", "orange", "banana", "grape", "lemon", "peach"]

KEY_LAYERS = {
    "qwen3": [16, 24, 25, 34, 35],
    "glm4": [10, 19, 24, 28, 38, 39],
    "deepseek7b": [0, 14, 23, 26, 27],
}


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


def compute_family_logits(logits_np, tokenizer):
    """计算每个候选族的平均logit"""
    result = {}
    for family_name, words in FAMILY_WORDS.items():
        val = get_logit_for_words(logits_np, tokenizer, words)
        if val is not None:
            result[family_name] = round(val, 4)
    return result


def compute_entropy_confidence(logits_np):
    """计算熵和置信度"""
    probs = np.exp(logits_np - logits_np.max())
    probs = probs / probs.sum()
    entropy = round(-float(np.sum(probs * np.log(probs + 1e-12))), 4)
    confidence = round(float(probs.max()), 4)
    return entropy, confidence


def compute_slot_metrics(logits_np, tokenizer):
    """计算所有槽位的logit指标"""
    result = {}
    for slot_name, words in SLOT_WORDS.items():
        val = get_logit_for_words(logits_np, tokenizer, words)
        if val is not None:
            result[f"{slot_name}_logit"] = round(val, 4)
    entropy, confidence = compute_entropy_confidence(logits_np)
    result["entropy"] = entropy
    result["confidence"] = confidence
    return result


def get_token_id(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    return ids[0] if ids else None


# ==================== Exp1: 候选族级别读出图谱 ====================
def exp1_family_readout(model, tokenizer, info, round_num=1):
    """
    对每个关键层的attn/MLP进行消融，
    测量候选族分布变化（不是单个logit）
    """
    plog_always(f"\n{'='*60}\nExp1: Candidate Family Readout Map\n{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)
    
    key_layers = KEY_LAYERS.get(info.name, [0, n_layers//3, 2*n_layers//3, n_layers-2, n_layers-1])
    key_layers = [l for l in key_layers if l < n_layers]
    
    obj_names = OBJ_NAMES_SHORT if round_num == 1 else OBJ_NAMES[:6]
    plog_always(f"  Objects: {obj_names}, Layers: {key_layers}")
    
    results = {}
    
    for li in key_layers:
        layer = layers[li]
        plog_always(f"  Processing L{li}...")
        
        layer_result = {"clean": {}, "zero_attn": {}, "zero_mlp": {}}
        
        for obj_name in obj_names:
            text = f"The {obj_name} is a"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            # === Clean forward ===
            with torch.no_grad():
                out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
            clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
            clean_family = compute_family_logits(clean_logits, tokenizer)
            clean_entropy, clean_conf = compute_entropy_confidence(clean_logits)
            
            # === Zero attn ===
            def make_zero_hook_attn(lpos):
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
            
            h_attn = None
            if hasattr(layer, 'self_attn'):
                h_attn = layer.self_attn.register_forward_hook(make_zero_hook_attn(last_pos))
            
            with torch.no_grad():
                out_za = model(input_ids=input_ids, attention_mask=attention_mask)
            if h_attn:
                h_attn.remove()
            
            za_logits = out_za.logits[0, -1].float().cpu().numpy()
            za_family = compute_family_logits(za_logits, tokenizer)
            za_entropy, za_conf = compute_entropy_confidence(za_logits)
            
            # === Zero MLP ===
            # 找到MLP模块
            mlp_mod = None
            if hasattr(layer, 'mlp'):
                mlp_mod = layer.mlp
            elif hasattr(layer, 'feed_forward'):
                mlp_mod = layer.feed_forward
            
            h_mlp = None
            if mlp_mod is not None:
                h_mlp = mlp_mod.register_forward_hook(make_zero_hook_attn(last_pos))
            
            with torch.no_grad():
                out_zm = model(input_ids=input_ids, attention_mask=attention_mask)
            if h_mlp:
                h_mlp.remove()
            
            zm_logits = out_zm.logits[0, -1].float().cpu().numpy()
            zm_family = compute_family_logits(zm_logits, tokenizer)
            zm_entropy, zm_conf = compute_entropy_confidence(zm_logits)
            
            # === 汇总 ===
            obj_result = {
                "clean_family": clean_family,
                "clean_entropy": clean_entropy,
                "clean_confidence": clean_conf,
                "zero_attn_family": za_family,
                "zero_attn_entropy": za_entropy,
                "zero_attn_confidence": za_conf,
                "zero_mlp_family": zm_family,
                "zero_mlp_entropy": zm_entropy,
                "zero_mlp_confidence": zm_conf,
                # 候选族变化
                "family_delta_attn": {k: round(za_family.get(k, 0) - clean_family.get(k, 0), 4) 
                                      for k in clean_family},
                "family_delta_mlp": {k: round(zm_family.get(k, 0) - clean_family.get(k, 0), 4) 
                                     for k in clean_family},
                # 边际变化 (target - top_compete)
                "fruit_margin_attn": round(za_family.get("fruit", 0) - max(za_family.get("animal", 0), za_family.get("tool", 0)), 4),
                "fruit_margin_mlp": round(zm_family.get("fruit", 0) - max(zm_family.get("animal", 0), zm_family.get("tool", 0)), 4),
                "fruit_margin_clean": round(clean_family.get("fruit", 0) - max(clean_family.get("animal", 0), clean_family.get("tool", 0)), 4),
            }
            layer_result[obj_name] = obj_result
        
        # 对象间平均
        avg_result = {"avg_family_delta_attn": {}, "avg_family_delta_mlp": {},
                     "avg_margin_clean": 0, "avg_margin_attn": 0, "avg_margin_mlp": 0}
        for obj_name in obj_names:
            if obj_name in layer_result:
                for k, v in layer_result[obj_name]["family_delta_attn"].items():
                    avg_result["avg_family_delta_attn"][k] = round(
                        avg_result["avg_family_delta_attn"].get(k, 0) + v / len(obj_names), 4)
                for k, v in layer_result[obj_name]["family_delta_mlp"].items():
                    avg_result["avg_family_delta_mlp"][k] = round(
                        avg_result["avg_family_delta_mlp"].get(k, 0) + v / len(obj_names), 4)
                avg_result["avg_margin_clean"] += layer_result[obj_name]["fruit_margin_clean"] / len(obj_names)
                avg_result["avg_margin_attn"] += layer_result[obj_name]["fruit_margin_attn"] / len(obj_names)
                avg_result["avg_margin_mlp"] += layer_result[obj_name]["fruit_margin_mlp"] / len(obj_names)
        
        avg_result["avg_margin_clean"] = round(avg_result["avg_margin_clean"], 4)
        avg_result["avg_margin_attn"] = round(avg_result["avg_margin_attn"], 4)
        avg_result["avg_margin_mlp"] = round(avg_result["avg_margin_mlp"], 4)
        
        layer_result["summary"] = avg_result
        results[f"L{li}"] = layer_result
        
        plog_always(f"  L{li} summary: margin_clean={avg_result['avg_margin_clean']}, "
                    f"margin_za={avg_result['avg_margin_attn']}, margin_zm={avg_result['avg_margin_mlp']}")
        # 打印关键族变化
        for fam in ["fruit", "animal", "tool", "object", "food"]:
            if fam in avg_result["avg_family_delta_attn"]:
                plog(f"    {fam}: attn_Δ={avg_result['avg_family_delta_attn'][fam]}, "
                     f"mlp_Δ={avg_result['avg_family_delta_mlp'][fam]}")
    
    return results


# ==================== Exp2: 跨槽位最后层attn压制测试 ====================
def exp2_cross_slot_attn_suppression(model, tokenizer, info, round_num=1):
    """
    测试最后一层attn是否只压制category，还是普遍压制所有槽位
    """
    plog_always(f"\n{'='*60}\nExp2: Cross-Slot Last Layer Attn Suppression\n{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)
    last_layer = n_layers - 1
    
    obj_names = OBJ_NAMES_SHORT if round_num == 1 else OBJ_NAMES[:6]
    
    # 每个槽位的模板
    SLOT_TEMPLATES = {
        "cat":      "The {obj} is a",
        "color":    "The color of the {obj} is",
        "part":     "The {obj} has a",
        "material": "The {obj} is made of",
        "function": "You can {obj} with the",
    }
    
    results = {}
    
    for slot_name, template in SLOT_TEMPLATES.items():
        plog_always(f"  Slot: {slot_name}, template: '{template}'")
        slot_result = {}
        
        target_words = SLOT_WORDS.get(slot_name, [])
        if not target_words:
            continue
        
        for obj_name in obj_names:
            text = template.format(obj=obj_name)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            # Clean
            with torch.no_grad():
                out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
            clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
            clean_target = get_logit_for_words(clean_logits, tokenizer, target_words)
            clean_entropy, clean_conf = compute_entropy_confidence(clean_logits)
            
            # Zero attn on last layer
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
            
            layer = layers[last_layer]
            h_attn = None
            if hasattr(layer, 'self_attn'):
                h_attn = layer.self_attn.register_forward_hook(make_zero_hook(last_pos))
            
            with torch.no_grad():
                out_za = model(input_ids=input_ids, attention_mask=attention_mask)
            if h_attn:
                h_attn.remove()
            
            za_logits = out_za.logits[0, -1].float().cpu().numpy()
            za_target = get_logit_for_words(za_logits, tokenizer, target_words)
            za_entropy, za_conf = compute_entropy_confidence(za_logits)
            
            target_delta = round(za_target - clean_target, 4) if (clean_target is not None and za_target is not None) else None
            entropy_delta = round(za_entropy - clean_entropy, 4)
            
            slot_result[obj_name] = {
                "clean_target_logit": round(clean_target, 4) if clean_target is not None else None,
                "zero_attn_target_logit": round(za_target, 4) if za_target is not None else None,
                "target_delta": target_delta,
                "clean_entropy": clean_entropy,
                "zero_attn_entropy": za_entropy,
                "entropy_delta": entropy_delta,
            }
        
        # 平均
        deltas = [v["target_delta"] for v in slot_result.values() if v["target_delta"] is not None]
        ent_deltas = [v["entropy_delta"] for v in slot_result.values()]
        
        avg_target_delta = round(float(np.mean(deltas)), 4) if deltas else None
        avg_entropy_delta = round(float(np.mean(ent_deltas)), 4) if ent_deltas else None
        
        # 判断：attn压制还是促进该槽位
        if avg_target_delta is not None:
            if avg_target_delta > 0.1:
                suppression = "SUPPRESSES (zeroing attn increases target)"
            elif avg_target_delta < -0.1:
                suppression = "PROMOTES (zeroing attn decreases target)"
            else:
                suppression = "NEUTRAL"
        else:
            suppression = "N/A"
        
        slot_result["summary"] = {
            "avg_target_delta": avg_target_delta,
            "avg_entropy_delta": avg_entropy_delta,
            "suppression_judgment": suppression,
        }
        
        results[slot_name] = slot_result
        plog_always(f"  {slot_name}: avg_target_Δ={avg_target_delta}, "
                    f"avg_entropy_Δ={avg_entropy_delta}, judgment={suppression}")
    
    return results


# ==================== Exp3: 多槽位读出接口画像(修复版) ====================
def exp3_multi_slot_readout(model, tokenizer, info, round_num=1):
    """
    修复Phase453 Exp4 bug — 正确计算每个槽位的logit变化
    
    对关键层测量:
    - shared injection (embedding注入)
    - zero_attn / zero_mlp
    - dir_only / scale_only
    """
    plog_always(f"\n{'='*60}\nExp3: Multi-Slot Readout Interface (Fixed)\n{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    W_U = get_W_U(model, info.name)
    W_U_T = W_U.T
    input_device = get_input_device(model)
    W_E = get_embedding_weight(model)
    
    # 构建cat方向
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["cat"]]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["opp_cat"]]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu().numpy()
    cat_dir = cat_dir / (np.linalg.norm(cat_dir) + 1e-8)
    cat_readout_dir = W_U_T[:, cat_ids].mean(axis=1)
    cat_readout_dir = cat_readout_dir / (np.linalg.norm(cat_readout_dir) + 1e-8)
    
    key_layers = KEY_LAYERS.get(info.name, [0, n_layers//3, 2*n_layers//3, n_layers-2, n_layers-1])
    key_layers = [l for l in key_layers if l < n_layers]
    
    obj_names = OBJ_NAMES_SHORT if round_num == 1 else OBJ_NAMES[:5]
    alpha = 1.0
    
    results = {}
    
    for li in key_layers:
        layer = layers[li]
        plog_always(f"  L{li}...")
        
        layer_data = {}
        
        for obj_name in obj_names:
            text = f"The {obj_name} is a"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            # === Clean ===
            with torch.no_grad():
                out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
            clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
            clean_metrics = compute_slot_metrics(clean_logits, tokenizer)
            
            # === Shared injection (embedding) ===
            perturb_t = torch.tensor(alpha * cat_dir, dtype=torch.float32).to(input_device).to(torch.bfloat16)
            embed_hook = None
            def on_embed_shared(m, inp, out, lp=last_pos, pt=perturb_t):
                if isinstance(out, torch.Tensor):
                    out = out.clone()
                    out[0, lp] = out[0, lp] + pt.to(out.dtype)
                return out
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                embed_hook = model.model.embed_tokens.register_forward_hook(on_embed_shared)
            
            try:
                with torch.no_grad():
                    out_pert = model(input_ids=input_ids, attention_mask=attention_mask)
            except:
                pass
            finally:
                if embed_hook:
                    embed_hook.remove()
            
            pert_logits = out_pert.logits[0, -1].float().cpu().numpy()
            pert_metrics = compute_slot_metrics(pert_logits, tokenizer)
            
            # === Zero attn ===
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
            
            h_attn = None
            if hasattr(layer, 'self_attn'):
                h_attn = layer.self_attn.register_forward_hook(make_zero_hook(last_pos))
            
            with torch.no_grad():
                out_za = model(input_ids=input_ids, attention_mask=attention_mask)
            if h_attn:
                h_attn.remove()
            
            za_logits = out_za.logits[0, -1].float().cpu().numpy()
            za_metrics = compute_slot_metrics(za_logits, tokenizer)
            
            # === Zero MLP ===
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
            zm_metrics = compute_slot_metrics(zm_logits, tokenizer)
            
            # === Compute deltas ===
            obj_data = {}
            for slot_name in SLOT_WORDS:
                clean_key = f"{slot_name}_logit"
                if clean_key in clean_metrics:
                    clean_val = clean_metrics[clean_key]
                    pert_val = pert_metrics.get(clean_key, clean_val)
                    za_val = za_metrics.get(clean_key, clean_val)
                    zm_val = zm_metrics.get(clean_key, clean_val)
                    
                    obj_data[clean_key + "_delta_shared"] = round(pert_val - clean_val, 4)
                    obj_data[clean_key + "_delta_zero_attn"] = round(za_val - clean_val, 4)
                    obj_data[clean_key + "_delta_zero_mlp"] = round(zm_val - clean_val, 4)
            
            obj_data["entropy_delta_shared"] = round(pert_metrics.get("entropy", 0) - clean_metrics.get("entropy", 0), 4)
            obj_data["entropy_delta_zero_attn"] = round(za_metrics.get("entropy", 0) - clean_metrics.get("entropy", 0), 4)
            obj_data["entropy_delta_zero_mlp"] = round(zm_metrics.get("entropy", 0) - clean_metrics.get("entropy", 0), 4)
            
            layer_data[obj_name] = obj_data
        
        # 平均
        avg_data = {}
        delta_keys = [k for k in layer_data[obj_names[0]].keys() if k.startswith("cat_") or k.startswith("color_") or k.startswith("part_") or k.startswith("entropy_")]
        for dk in delta_keys:
            vals = [layer_data[o][dk] for o in obj_names if o in layer_data and dk in layer_data[o]]
            if vals:
                avg_data[dk] = round(float(np.mean(vals)), 4)
        
        layer_data["summary"] = avg_data
        results[f"L{li}"] = layer_data
        
        # 打印
        cat_shared = avg_data.get("cat_logit_delta_shared", "N/A")
        cat_za = avg_data.get("cat_logit_delta_zero_attn", "N/A")
        cat_zm = avg_data.get("cat_logit_delta_zero_mlp", "N/A")
        color_shared = avg_data.get("color_logit_delta_shared", "N/A")
        part_shared = avg_data.get("part_logit_delta_shared", "N/A")
        plog_always(f"  L{li}: cat_shared={cat_shared}, cat_za={cat_za}, cat_zm={cat_zm}, "
                    f"color_shared={color_shared}, part_shared={part_shared}")
    
    return results


# ==================== Exp4: DS7B双峰层注入复验 ====================
def exp4_ds7b_bimodal_layer_injection(model, tokenizer, info, round_num=1):
    """
    使用层注入而非嵌入注入来复验DS7B双峰
    
    对所有模型的后几层做scale sweep, 看范数响应是否非单调
    """
    plog_always(f"\n{'='*60}\nExp4: Layer Injection Scale Sweep (DS7B Bimodal Verify)\n{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    W_U = get_W_U(model, info.name)
    W_U_T = W_U.T
    input_device = get_input_device(model)
    W_E = get_embedding_weight(model)
    
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["cat"]]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["opp_cat"]]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu().numpy()
    cat_dir = cat_dir / (np.linalg.norm(cat_dir) + 1e-8)
    cat_readout_dir = W_U_T[:, cat_ids].mean(axis=1)
    cat_readout_dir = cat_readout_dir / (np.linalg.norm(cat_readout_dir) + 1e-8)
    
    # 关键层 — 针对每个模型
    if info.name == "qwen3":
        test_layers = [16, 25, 34, 35]
    elif info.name == "glm4":
        test_layers = [10, 24, 38, 39]
    elif info.name == "deepseek7b":
        test_layers = [0, 14, 23, 26, 27]
    else:
        test_layers = [0, n_layers//2, n_layers-2, n_layers-1]
    test_layers = [l for l in test_layers if l < n_layers]
    
    obj_names = OBJ_NAMES_SHORT if round_num == 1 else OBJ_NAMES[:5]
    alphas = [0.25, 0.5, 1.0, 2.0, 4.0]
    
    results = {}
    
    for li in test_layers:
        layer = layers[li]
        plog_always(f"  L{li} scale sweep...")
        layer_result = {}
        
        for alpha in alphas:
            alpha_data = {"cat_delta": [], "color_delta": [], "part_delta": [],
                         "entropy_delta": [], "fruit_margin_delta": []}
            
            for obj_name in obj_names:
                text = f"The {obj_name} is a"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attention_mask = inputs["attention_mask"].to(input_device)
                last_pos = input_ids.shape[1] - 1
                
                # === Clean ===
                with torch.no_grad():
                    out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
                clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
                clean_cat = get_logit_for_words(clean_logits, tokenizer, SLOT_WORDS["cat"])
                clean_color = get_logit_for_words(clean_logits, tokenizer, SLOT_WORDS["color"])
                clean_part = get_logit_for_words(clean_logits, tokenizer, SLOT_WORDS["part"])
                clean_entropy, _ = compute_entropy_confidence(clean_logits)
                
                # === 层注入: 在layer的输入处注入cat_dir ===
                def make_inject_hook(lpos, alpha_val, direction):
                    def hook(m, inp, out):
                        # 注入到输出上(等同于在残差上加)
                        if isinstance(out, tuple):
                            new_out = out[0].clone()
                            perturb = torch.tensor(alpha_val * direction, dtype=torch.float32).to(new_out.device).to(new_out.dtype)
                            new_out[0, lpos] = new_out[0, lpos] + perturb
                            return (new_out,) + out[1:]
                        else:
                            new_out = out.clone()
                            perturb = torch.tensor(alpha_val * direction, dtype=torch.float32).to(new_out.device).to(new_out.dtype)
                            new_out[0, lpos] = new_out[0, lpos] + perturb
                            return new_out
                    return hook
                
                h_inject = layer.register_forward_hook(make_inject_hook(last_pos, alpha, cat_dir))
                
                try:
                    with torch.no_grad():
                        out_pert = model(input_ids=input_ids, attention_mask=attention_mask)
                except:
                    pass
                finally:
                    h_inject.remove()
                
                pert_logits = out_pert.logits[0, -1].float().cpu().numpy()
                pert_cat = get_logit_for_words(pert_logits, tokenizer, SLOT_WORDS["cat"])
                pert_color = get_logit_for_words(pert_logits, tokenizer, SLOT_WORDS["color"])
                pert_part = get_logit_for_words(pert_logits, tokenizer, SLOT_WORDS["part"])
                pert_entropy, _ = compute_entropy_confidence(pert_logits)
                
                cat_delta = round(pert_cat - clean_cat, 4) if (pert_cat is not None and clean_cat is not None) else None
                color_delta = round(pert_color - clean_color, 4) if (pert_color is not None and clean_color is not None) else None
                part_delta = round(pert_part - clean_part, 4) if (pert_part is not None and clean_part is not None) else None
                entropy_delta = round(pert_entropy - clean_entropy, 4)
                
                alpha_data["cat_delta"].append(cat_delta)
                alpha_data["color_delta"].append(color_delta)
                alpha_data["part_delta"].append(part_delta)
                alpha_data["entropy_delta"].append(entropy_delta)
            
            # 平均
            avg_cat = round(float(np.mean([v for v in alpha_data["cat_delta"] if v is not None])), 4) if alpha_data["cat_delta"] else None
            avg_color = round(float(np.mean([v for v in alpha_data["color_delta"] if v is not None])), 4) if alpha_data["color_delta"] else None
            avg_part = round(float(np.mean([v for v in alpha_data["part_delta"] if v is not None])), 4) if alpha_data["part_delta"] else None
            avg_entropy = round(float(np.mean(alpha_data["entropy_delta"])), 4)
            
            # 检测非单调性
            is_nonmonotonic = False
            if len(layer_result) >= 2:
                prev_cats = [layer_result[k]["avg_cat_delta"] for k in sorted(layer_result.keys()) if layer_result[k]["avg_cat_delta"] is not None]
                if len(prev_cats) >= 2 and avg_cat is not None:
                    # 如果方向变化了（从正变负或从负变正）
                    if prev_cats[-1] * avg_cat < 0:
                        is_nonmonotonic = True
            
            layer_result[f"alpha{alpha}"] = {
                "avg_cat_delta": avg_cat,
                "avg_color_delta": avg_color,
                "avg_part_delta": avg_part,
                "avg_entropy_delta": avg_entropy,
                "nonmonotonic": is_nonmonotonic,
            }
        
        results[f"L{li}"] = layer_result
        
        # 检查非单调性
        cat_deltas = [layer_result[k]["avg_cat_delta"] for k in sorted(layer_result.keys()) if layer_result[k]["avg_cat_delta"] is not None]
        is_bimodal = False
        if len(cat_deltas) >= 3:
            for i in range(1, len(cat_deltas)-1):
                if cat_deltas[i-1] * cat_deltas[i+1] > 0 and cat_deltas[i] * cat_deltas[i-1] < 0:
                    is_bimodal = True
                    break
        
        cat_str = ", ".join([f"a{k.split('alpha')[1]}={layer_result[k]['avg_cat_delta']}" for k in sorted(layer_result.keys())])
        plog_always(f"  L{li}: {cat_str}  bimodal={is_bimodal}")
    
    return results


# ==================== Exp5: 投影-因果-候选族三证合一 ====================
def exp5_triple_evidence(model, tokenizer, info, round_num=1):
    """
    对每个关键组件计算:
    1. ProjectionScore: 组件输出→cat读出方向投影
    2. CausalEffect: 消融组件→cat logit变化
    3. FamilyRedistribution: 消融组件→候选族分布变化
    
    三者结合判断机制
    """
    plog_always(f"\n{'='*60}\nExp5: Projection-Causality-Family Triple Evidence\n{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    W_U = get_W_U(model, info.name)
    W_U_T = W_U.T
    input_device = get_input_device(model)
    
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["cat"]]
    cat_readout_dir = W_U_T[:, cat_ids].mean(axis=1)
    cat_readout_dir = cat_readout_dir / (np.linalg.norm(cat_readout_dir) + 1e-8)
    
    key_layers = KEY_LAYERS.get(info.name, [0, n_layers//3, 2*n_layers//3, n_layers-2, n_layers-1])
    key_layers = [l for l in key_layers if l < n_layers]
    
    obj_names = OBJ_NAMES_SHORT if round_num == 1 else OBJ_NAMES[:5]
    
    results = {}
    
    for li in key_layers:
        layer = layers[li]
        plog_always(f"  L{li} triple evidence...")
        
        layer_result = {}
        
        for obj_name in obj_names:
            text = f"The {obj_name} is a"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            # === 1. 投影分数: 捕获attn/MLP输出 ===
            captured = {}
            def make_capture_hook(comp_name, lpos):
                def hook(m, inp, out):
                    if isinstance(out, tuple):
                        captured[comp_name] = out[0][0, lpos].detach().float().cpu().numpy()
                    else:
                        captured[comp_name] = out[0, lpos].detach().float().cpu().numpy() if out.ndim > 1 else out.detach().float().cpu().numpy()
                return hook
            
            h_attn_cap = None
            h_mlp_cap = None
            if hasattr(layer, 'self_attn'):
                h_attn_cap = layer.self_attn.register_forward_hook(make_capture_hook("attn", last_pos))
            mlp_mod = None
            if hasattr(layer, 'mlp'):
                mlp_mod = layer.mlp
            elif hasattr(layer, 'feed_forward'):
                mlp_mod = layer.feed_forward
            if mlp_mod is not None:
                h_mlp_cap = mlp_mod.register_forward_hook(make_capture_hook("mlp", last_pos))
            
            # Clean forward
            with torch.no_grad():
                out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
            clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
            clean_cat = get_logit_for_words(clean_logits, tokenizer, SLOT_WORDS["cat"])
            clean_family = compute_family_logits(clean_logits, tokenizer)
            clean_entropy, clean_conf = compute_entropy_confidence(clean_logits)
            
            if h_attn_cap:
                h_attn_cap.remove()
            if h_mlp_cap:
                h_mlp_cap.remove()
            
            # 投影分数
            attn_proj = float(np.dot(captured["attn"], cat_readout_dir)) if "attn" in captured else None
            mlp_proj = float(np.dot(captured["mlp"], cat_readout_dir)) if "mlp" in captured else None
            attn_norm = float(np.linalg.norm(captured["attn"])) if "attn" in captured else None
            mlp_norm = float(np.linalg.norm(captured["mlp"])) if "mlp" in captured else None
            
            # === 2. 因果分数: zero attn/MLP ===
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
            
            # Zero attn
            h_za = None
            if hasattr(layer, 'self_attn'):
                h_za = layer.self_attn.register_forward_hook(make_zero_hook(last_pos))
            with torch.no_grad():
                out_za = model(input_ids=input_ids, attention_mask=attention_mask)
            if h_za:
                h_za.remove()
            za_logits = out_za.logits[0, -1].float().cpu().numpy()
            za_cat = get_logit_for_words(za_logits, tokenizer, SLOT_WORDS["cat"])
            za_family = compute_family_logits(za_logits, tokenizer)
            
            # Zero MLP
            h_zm = None
            if mlp_mod is not None:
                h_zm = mlp_mod.register_forward_hook(make_zero_hook(last_pos))
            with torch.no_grad():
                out_zm = model(input_ids=input_ids, attention_mask=attention_mask)
            if h_zm:
                h_zm.remove()
            zm_logits = out_zm.logits[0, -1].float().cpu().numpy()
            zm_cat = get_logit_for_words(zm_logits, tokenizer, SLOT_WORDS["cat"])
            zm_family = compute_family_logits(zm_logits, tokenizer)
            
            # === 3. 候选族再分布 ===
            # 计算family margin: target(fruit) - max_compete(animal, tool, vehicle)
            def compute_family_margin(family_dict):
                target = family_dict.get("fruit", 0)
                competes = [family_dict.get(k, -999) for k in ["animal", "tool", "vehicle", "object", "food"]]
                return round(target - max(competes), 4)
            
            clean_margin = compute_family_margin(clean_family)
            za_margin = compute_family_margin(za_family)
            zm_margin = compute_family_margin(zm_family)
            
            # 候选族变化
            family_delta_attn = {k: round(za_family.get(k, 0) - clean_family.get(k, 0), 4) for k in clean_family}
            family_delta_mlp = {k: round(zm_family.get(k, 0) - clean_family.get(k, 0), 4) for k in clean_family}
            
            # === 三证合一 ===
            # CausalEffect = -(zero_delta)  → 正=促进, 负=压制
            attn_causal = round(-(za_cat - clean_cat), 4) if (za_cat is not None and clean_cat is not None) else None
            mlp_causal = round(-(zm_cat - clean_cat), 4) if (zm_cat is not None and clean_cat is not None) else None
            
            # 象限判定
            def get_quadrant(proj, causal):
                if proj is None or causal is None:
                    return "N/A"
                if proj > 0 and causal > 0:
                    return "Q1: true promoter"
                elif proj > 0 and causal < 0:
                    return "Q2: geometric promoter but causal suppressor"
                elif proj < 0 and causal > 0:
                    return "Q3: indirect promoter"
                else:
                    return "Q4: true suppressor"
            
            attn_quad = get_quadrant(attn_proj, attn_causal)
            mlp_quad = get_quadrant(mlp_proj, mlp_causal)
            
            # 三证判定: proj + causal + family_margin all agree?
            def triple_judgment(proj, causal, margin_clean, margin_ablated):
                """三证是否一致"""
                if any(v is None for v in [proj, causal, margin_clean, margin_ablated]):
                    return "INCOMPLETE"
                # proj>0 & causal>0 & margin increases → TRUE PROMOTER
                # proj>0 & causal<0 & margin decreases → GEOMETRIC TRAP
                # proj<0 & causal>0 & margin increases → INDIRECT PROMOTER
                margin_delta = margin_ablated - margin_clean
                if proj > 0 and causal > 0 and margin_delta < 0:
                    return "TRIPLE_PROMOTER"  # all 3 agree: it promotes
                elif proj > 0 and causal < 0 and margin_delta > 0:
                    return "TRIPLE_SUPPRESSOR"  # all 3 agree: it suppresses
                elif proj > 0 and causal < 0:
                    return "PROJ_CAUSAL_CONFLICT"  # proj and causal disagree
                elif proj < 0 and causal > 0:
                    return "INDIRECT_PROMOTER"  # indirect promotion
                else:
                    return "MIXED"
            
            attn_triple = triple_judgment(attn_proj, attn_causal, clean_margin, za_margin)
            mlp_triple = triple_judgment(mlp_proj, mlp_causal, clean_margin, zm_margin)
            
            obj_result = {
                "attn_proj": round(attn_proj, 4) if attn_proj is not None else None,
                "mlp_proj": round(mlp_proj, 4) if mlp_proj is not None else None,
                "attn_causal": attn_causal,
                "mlp_causal": mlp_causal,
                "attn_norm": round(attn_norm, 2) if attn_norm is not None else None,
                "mlp_norm": round(mlp_norm, 2) if mlp_norm is not None else None,
                "attn_quadrant": attn_quad,
                "mlp_quadrant": mlp_quad,
                "attn_triple": attn_triple,
                "mlp_triple": mlp_triple,
                "clean_margin": clean_margin,
                "za_margin": za_margin,
                "zm_margin": zm_margin,
                "margin_delta_attn": round(za_margin - clean_margin, 4),
                "margin_delta_mlp": round(zm_margin - clean_margin, 4),
                "family_delta_attn": family_delta_attn,
                "family_delta_mlp": family_delta_mlp,
            }
            layer_result[obj_name] = obj_result
        
        # 平均
        avg_result = {}
        for key in ["attn_proj", "mlp_proj", "attn_causal", "mlp_causal", "attn_norm", "mlp_norm",
                    "clean_margin", "za_margin", "zm_margin", "margin_delta_attn", "margin_delta_mlp"]:
            vals = [layer_result[o][key] for o in obj_names if o in layer_result and layer_result[o][key] is not None]
            if vals:
                avg_result[key] = round(float(np.mean(vals)), 4)
        
        # 最常见象限
        attn_quads = [layer_result[o]["attn_quadrant"] for o in obj_names if o in layer_result]
        mlp_quads = [layer_result[o]["mlp_quadrant"] for o in obj_names if o in layer_result]
        attn_triples = [layer_result[o]["attn_triple"] for o in obj_names if o in layer_result]
        mlp_triples = [layer_result[o]["mlp_triple"] for o in obj_names if o in layer_result]
        avg_result["attn_quadrant"] = max(set(attn_quads), key=attn_quads.count) if attn_quads else "N/A"
        avg_result["mlp_quadrant"] = max(set(mlp_quads), key=mlp_quads.count) if mlp_quads else "N/A"
        avg_result["attn_triple"] = max(set(attn_triples), key=attn_triples.count) if attn_triples else "N/A"
        avg_result["mlp_triple"] = max(set(mlp_triples), key=mlp_triples.count) if mlp_triples else "N/A"
        
        # 平均候选族变化
        all_families = set()
        for o in obj_names:
            if o in layer_result:
                all_families.update(layer_result[o]["family_delta_attn"].keys())
        
        avg_family_delta_attn = {}
        avg_family_delta_mlp = {}
        for fam in sorted(all_families):
            attn_vals = [layer_result[o]["family_delta_attn"].get(fam, 0) for o in obj_names if o in layer_result]
            mlp_vals = [layer_result[o]["family_delta_mlp"].get(fam, 0) for o in obj_names if o in layer_result]
            if attn_vals:
                avg_family_delta_attn[fam] = round(float(np.mean(attn_vals)), 4)
            if mlp_vals:
                avg_family_delta_mlp[fam] = round(float(np.mean(mlp_vals)), 4)
        
        avg_result["avg_family_delta_attn"] = avg_family_delta_attn
        avg_result["avg_family_delta_mlp"] = avg_family_delta_mlp
        
        layer_result["summary"] = avg_result
        results[f"L{li}"] = layer_result
        
        plog_always(f"  L{li}: attn_proj={avg_result.get('attn_proj')}, attn_causal={avg_result.get('attn_causal')}, "
                    f"attn_quad={avg_result.get('attn_quadrant')}, attn_triple={avg_result.get('attn_triple')}")
        plog_always(f"         mlp_proj={avg_result.get('mlp_proj')}, mlp_causal={avg_result.get('mlp_causal')}, "
                    f"mlp_quad={avg_result.get('mlp_quadrant')}, mlp_triple={avg_result.get('mlp_triple')}")
        plog_always(f"         margin: clean={avg_result.get('clean_margin')}, za={avg_result.get('za_margin')}, zm={avg_result.get('zm_margin')}")
        for fam in ["fruit", "animal", "tool", "object", "food"]:
            if fam in avg_family_delta_attn:
                plog(f"    {fam}: attn_Δ={avg_family_delta_attn[fam]}, mlp_Δ={avg_family_delta_mlp[fam]}")
    
    return results


# ==================== 主流程 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    plog_always(f"Phase 454: Candidate Family Redistribution")
    plog_always(f"Model: {model_name}, Round: {round_num}")
    plog_always(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    t0_total = time.time()
    
    # 加载模型
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    plog_always(f"Model info: class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    all_results = {
        "model": model_name,
        "round": round_num,
        "model_info": {"class": info.model_class, "n_layers": info.n_layers, 
                       "d_model": info.d_model, "mlp_type": info.mlp_type},
    }
    
    # Exp1: 候选族级别读出图谱
    t0 = time.time()
    all_results["exp1"] = exp1_family_readout(model, tokenizer, info, round_num)
    plog_always(f"Exp1 done in {time.time()-t0:.0f}s")
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Exp2: 跨槽位最后层attn压制测试
    t0 = time.time()
    all_results["exp2"] = exp2_cross_slot_attn_suppression(model, tokenizer, info, round_num)
    plog_always(f"Exp2 done in {time.time()-t0:.0f}s")
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Exp3: 多槽位读出接口画像
    t0 = time.time()
    all_results["exp3"] = exp3_multi_slot_readout(model, tokenizer, info, round_num)
    plog_always(f"Exp3 done in {time.time()-t0:.0f}s")
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Exp4: 层注入scale sweep
    t0 = time.time()
    all_results["exp4"] = exp4_ds7b_bimodal_layer_injection(model, tokenizer, info, round_num)
    plog_always(f"Exp4 done in {time.time()-t0:.0f}s")
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Exp5: 投影-因果-候选族三证合一
    t0 = time.time()
    all_results["exp5"] = exp5_triple_evidence(model, tokenizer, info, round_num)
    plog_always(f"Exp5 done in {time.time()-t0:.0f}s")
    
    # 保存结果
    out_path = f"results/glm5/phase454_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    plog_always(f"Saved to {out_path}")
    
    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    plog_always(f"Total time: {time.time()-t0_total:.0f}s")
    plog_always("Phase 454 done!")


if __name__ == "__main__":
    main()
