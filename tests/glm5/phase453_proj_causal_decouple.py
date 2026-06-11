"""
Phase 453: 投影-因果解耦验证与读出接口标准化
=============================================
核心目标:
1. Exp1: 投影-因果四象限图谱 — projection vs causal effect for attn/MLP
2. Exp2: RMSNorm行为因果测试 — bypass RMSNorm对比logit变化
3. Exp3: 标准化direction/scale分解 — 统一跨阶段定义
4. Exp4: 多槽位读出接口画像 — category/color/part/material
5. Exp5: DS7B双峰精细复验 — 细粒度scale sweep

用法:
  python tests/glm5/phase453_proj_causal_decouple.py qwen3 1
  python tests/glm5/phase453_proj_causal_decouple.py glm4 1
  python tests/glm5/phase453_proj_causal_decouple.py deepseek7b 1
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
log = logging.getLogger("p453")

def plog(msg):
    log.info(msg)

def load_model_auto(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog(f"Loading {model_name} (bf16 + auto + flash)...")
    
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
            plog(f"  attn={attn_impl}")
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
        plog(f"  Layers: {gpu_l} GPU + {cpu_l} CPU")
    
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


def get_logit_for_words(logits_np, tokenizer, word_list):
    ids = []
    for w in word_list:
        tok_ids = tokenizer.encode(w, add_special_tokens=False)
        if tok_ids:
            ids.append(tok_ids[0])
    if not ids:
        return None
    return float(np.mean(logits_np[ids]))


def compute_logit_metrics(logits_np, tokenizer, slot_words):
    result = {}
    for slot_name, words in slot_words.items():
        val = get_logit_for_words(logits_np, tokenizer, words)
        if val is not None:
            result[f"{slot_name}_logit"] = round(val, 4)
    
    probs = np.exp(logits_np - logits_np.max())
    probs = probs / probs.sum()
    result["entropy"] = round(-float(np.sum(probs * np.log(probs + 1e-12))), 4)
    result["confidence"] = round(float(probs.max()), 4)
    return result


# === 槽位定义 ===
SLOT_WORDS = {
    "cat": ["fruit", "food", "produce"],
    "opp_cat": ["animal", "dog", "cat"],
    "color": ["red", "green", "yellow"],
    "part": ["seed", "skin", "core", "stem"],
    "material": ["organic", "natural", "fresh"],
    "function": ["eat", "cook", "grow"],
    "habitat": ["tree", "garden", "farm"],
}

OBJ_NAMES = ["apple", "orange", "banana", "grape", "lemon", "peach", "pear", "mango", "plum", "cherry"]
OBJ_NAMES_SHORT = ["apple", "orange", "banana"]

KEY_LAYERS = {
    "qwen3": [16, 24, 25, 34, 35],
    "glm4": [10, 19, 24, 28, 38, 39],
    "deepseek7b": [0, 14, 23, 26, 27],
}

def get_key_layers(model_name, n_layers):
    if model_name in KEY_LAYERS:
        return [l for l in KEY_LAYERS[model_name] if l < n_layers]
    return [0, n_layers//3, 2*n_layers//3, n_layers-2, n_layers-1]


# === Hook helpers ===
def make_zero_hook(lpos):
    """Hook to zero out output at position lpos"""
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

def make_negate_hook(lpos):
    """Hook to negate output at position lpos"""
    def hook(m, inp, out):
        if isinstance(out, tuple):
            new_out = out[0].clone()
            new_out[0, lpos] = -new_out[0, lpos]
            return (new_out,) + out[1:]
        else:
            new_out = out.clone()
            new_out[0, lpos] = -new_out[0, lpos]
            return new_out
    return hook

def make_capture_hook(captured, key, lpos):
    """Hook to capture output at position lpos"""
    def hook(m, inp, out):
        if isinstance(out, tuple):
            captured[key] = out[0][0, lpos].detach().float().cpu().numpy()
        else:
            captured[key] = out[0, lpos].detach().float().cpu().numpy() if out.ndim > 1 else out.detach().float().cpu().numpy()
    return hook

def make_input_capture_hook(captured, key, lpos):
    """Hook to capture input at position lpos"""
    def hook(m, inp, out):
        if isinstance(inp, tuple) and len(inp) > 0:
            captured[key] = inp[0][0, lpos].detach().float().cpu().numpy()
    return hook


# ===== Exp1: 投影-因果四象限图谱 =====
def exp1_projection_causality_map(model, tokenizer, info, round_num=1):
    """
    对每个关键层计算attn/MLP的投影分数和因果分数, 建立四象限图谱
    """
    plog(f"\n{'='*60}\nExp1: Projection-Causality 4-Quadrant Map\n{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    W_U = get_W_U(model, info.name)
    W_U_T = W_U.T
    input_device = get_input_device(model)
    
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["cat"]]
    cat_readout_dir = W_U_T[:, cat_ids].mean(axis=1)
    cat_readout_dir = cat_readout_dir / (np.linalg.norm(cat_readout_dir) + 1e-8)
    
    target_layers = get_key_layers(info.name, n_layers)
    obj_names = OBJ_NAMES_SHORT if round_num == 1 else OBJ_NAMES[:6]
    plog(f"  Target layers: {target_layers}, objects: {len(obj_names)}")
    
    results = {}
    
    for li in target_layers:
        layer_data = {"attn_proj_cat": [], "mlp_proj_cat": [],
                      "zero_attn_cat_delta": [], "zero_mlp_cat_delta": [],
                      "negate_attn_cat_delta": [], "negate_mlp_cat_delta": [],
                      "attn_out_norm": [], "mlp_out_norm": []}
        
        for obj_name in obj_names:
            text = f"The {obj_name} is a"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            # Clean forward + capture attn_out, mlp_out
            captured = {}
            h_attn = layers[li].self_attn.register_forward_hook(make_capture_hook(captured, "attn_out", last_pos))
            h_mlp = layers[li].mlp.register_forward_hook(make_capture_hook(captured, "mlp_out", last_pos))
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            clean_logits = out.logits[0, -1].float().cpu().numpy()
            clean_cat = get_logit_for_words(clean_logits, tokenizer, SLOT_WORDS["cat"])
            
            h_attn.remove()
            h_mlp.remove()
            
            if "attn_out" not in captured or "mlp_out" not in captured:
                plog(f"  L{li} {obj_name}: capture failed")
                continue
            
            attn_out = captured["attn_out"]
            mlp_out = captured["mlp_out"]
            attn_norm = float(np.linalg.norm(attn_out))
            mlp_norm = float(np.linalg.norm(mlp_out))
            attn_proj = float(np.dot(attn_out, cat_readout_dir))
            mlp_proj = float(np.dot(mlp_out, cat_readout_dir))
            
            layer_data["attn_proj_cat"].append(attn_proj)
            layer_data["mlp_proj_cat"].append(mlp_proj)
            layer_data["attn_out_norm"].append(attn_norm)
            layer_data["mlp_out_norm"].append(mlp_norm)
            
            # Zero attn
            h_za = layers[li].self_attn.register_forward_hook(make_zero_hook(last_pos))
            with torch.no_grad():
                out_za = model(input_ids=input_ids, attention_mask=attention_mask)
            h_za.remove()
            za_cat = get_logit_for_words(out_za.logits[0, -1].float().cpu().numpy(), tokenizer, SLOT_WORDS["cat"])
            layer_data["zero_attn_cat_delta"].append(za_cat - clean_cat if clean_cat is not None else None)
            
            # Zero MLP
            h_zm = layers[li].mlp.register_forward_hook(make_zero_hook(last_pos))
            with torch.no_grad():
                out_zm = model(input_ids=input_ids, attention_mask=attention_mask)
            h_zm.remove()
            zm_cat = get_logit_for_words(out_zm.logits[0, -1].float().cpu().numpy(), tokenizer, SLOT_WORDS["cat"])
            layer_data["zero_mlp_cat_delta"].append(zm_cat - clean_cat if clean_cat is not None else None)
            
            # Negate attn
            h_na = layers[li].self_attn.register_forward_hook(make_negate_hook(last_pos))
            with torch.no_grad():
                out_na = model(input_ids=input_ids, attention_mask=attention_mask)
            h_na.remove()
            na_cat = get_logit_for_words(out_na.logits[0, -1].float().cpu().numpy(), tokenizer, SLOT_WORDS["cat"])
            layer_data["negate_attn_cat_delta"].append(na_cat - clean_cat if clean_cat is not None else None)
            
            # Negate MLP
            h_nm = layers[li].mlp.register_forward_hook(make_negate_hook(last_pos))
            with torch.no_grad():
                out_nm = model(input_ids=input_ids, attention_mask=attention_mask)
            h_nm.remove()
            nm_cat = get_logit_for_words(out_nm.logits[0, -1].float().cpu().numpy(), tokenizer, SLOT_WORDS["cat"])
            layer_data["negate_mlp_cat_delta"].append(nm_cat - clean_cat if clean_cat is not None else None)
            
            gc.collect()
        
        # Average over objects
        result = {}
        for key in layer_data:
            vals = [v for v in layer_data[key] if v is not None]
            result[key] = round(float(np.mean(vals)), 4) if vals else None
        
        # Quadrant classification
        attn_proj_sign = 1 if (result.get("attn_proj_cat") or 0) > 0 else -1
        attn_causal_sign = 1 if (result.get("zero_attn_cat_delta") or 0) < 0 else -1
        mlp_proj_sign = 1 if (result.get("mlp_proj_cat") or 0) > 0 else -1
        mlp_causal_sign = 1 if (result.get("zero_mlp_cat_delta") or 0) < 0 else -1
        
        def classify_quadrant(proj_s, causal_s):
            if proj_s > 0 and causal_s > 0:
                return "Q1: proj+ & causal+ (true promoter)"
            elif proj_s > 0 and causal_s < 0:
                return "Q2: proj+ & causal- (suppressor)"
            elif proj_s < 0 and causal_s > 0:
                return "Q3: proj- & causal+ (indirect promoter)"
            else:
                return "Q4: proj- & causal- (true suppressor)"
        
        result["attn_quadrant"] = classify_quadrant(attn_proj_sign, attn_causal_sign)
        result["mlp_quadrant"] = classify_quadrant(mlp_proj_sign, mlp_causal_sign)
        result["attn_proj_sign"] = attn_proj_sign
        result["attn_causal_sign"] = attn_causal_sign
        result["mlp_proj_sign"] = mlp_proj_sign
        result["mlp_causal_sign"] = mlp_causal_sign
        
        results[f"L{li}"] = result
        plog(f"  L{li}: attn_proj={result.get('attn_proj_cat'):.3f}, "
             f"zero_attn→Δ={result.get('zero_attn_cat_delta'):.3f} → {result['attn_quadrant']}")
        plog(f"         mlp_proj={result.get('mlp_proj_cat'):.3f}, "
             f"zero_mlp→Δ={result.get('zero_mlp_cat_delta'):.3f} → {result['mlp_quadrant']}")
    
    return results


# ===== Exp2: RMSNorm行为因果测试 =====
def exp2_rmsnorm_behavior_causal(model, tokenizer, info, round_num=1):
    """
    核心测试: bypass RMSNorm at specific layer, 看是否改变logit效应的方向
    
    1. Clean → clean_cat
    2. Perturbed (embed inject) → pert_cat, delta_with_rms = pert_cat - clean_cat
    3. Clean + RMSNorm-bypass → clean_bypass_cat
    4. Perturbed + RMSNorm-bypass → pert_bypass_cat, delta_without_rms = pert_bypass_cat - clean_bypass_cat
    5. If sign(delta_with_rms) ≠ sign(delta_without_rms) → RMSNorm causes behavior flip
    
    Also compute projection flip as secondary measure.
    """
    plog(f"\n{'='*60}\nExp2: RMSNorm Behavior Causal Test\n{'='*60}")
    
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
    
    if info.name == "qwen3":
        rms_layers = [25, 34, 35]
    elif info.name == "glm4":
        rms_layers = [24, 38, 39]
    elif info.name == "deepseek7b":
        rms_layers = [26, 27]
    else:
        rms_layers = [n_layers-2, n_layers-1]
    rms_layers = [l for l in rms_layers if l < n_layers]
    
    obj_names = OBJ_NAMES_SHORT if round_num == 1 else OBJ_NAMES[:5]
    alpha = 1.0
    plog(f"  RMS test layers: {rms_layers}, objects: {len(obj_names)}")
    
    results = {}
    
    for li in rms_layers:
        layer = layers[li]
        ln_module = None
        for ln_name in ['input_layernorm', 'ln1']:
            if hasattr(layer, ln_name):
                ln_module = getattr(layer, ln_name)
                break
        
        rms_weight = None
        if ln_module is not None:
            try:
                w = ln_module.weight
                if not w.is_meta:
                    rms_weight = w.detach().float().cpu().numpy()
                else:
                    from safetensors import safe_open
                    cfg = MODEL_CONFIGS[info.name]
                    target_key = f'model.layers.{li}.input_layernorm.weight'
                    for sf in glob.glob(os.path.join(cfg["path"], '*.safetensors')):
                        with safe_open(sf, framework='pt', device='cpu') as f:
                            if target_key in f.keys():
                                rms_weight = f.get_tensor(target_key).float().numpy()
                                break
                    if rms_weight is None:
                        plog(f"  Warning: RMSNorm weight not found in safetensors for L{li}")
            except Exception as e:
                plog(f"  Warning: could not load RMSNorm weight for L{li}: {e}")
                rms_weight = None
        
        layer_data = {"pre_rms_proj": [], "post_rms_proj": [],
                     "logit_delta_with_rms": [], "logit_delta_without_rms": []}
        
        for obj_name in obj_names:
            text = f"The {obj_name} is a"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            # 1. Clean forward + capture layer input
            captured_clean = {}
            h_cap = layer.register_forward_hook(make_input_capture_hook(captured_clean, "input", last_pos))
            with torch.no_grad():
                out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
            h_cap.remove()
            clean_cat = get_logit_for_words(out_clean.logits[0, -1].float().cpu().numpy(), tokenizer, SLOT_WORDS["cat"])
            
            # 2. Perturbed forward (embed inject) + capture layer input
            perturb_t = torch.tensor(alpha * cat_dir, dtype=torch.float32).to(input_device).to(torch.bfloat16)
            
            captured_pert = {}
            h_cap_p = layer.register_forward_hook(make_input_capture_hook(captured_pert, "input", last_pos))
            
            embed_hook = None
            def on_embed_pert(m, inp, out):
                if isinstance(out, torch.Tensor):
                    out = out.clone()
                    out[0, last_pos] = out[0, last_pos] + perturb_t.to(out.dtype)
                return out
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                embed_hook = model.model.embed_tokens.register_forward_hook(on_embed_pert)
            
            try:
                with torch.no_grad():
                    out_pert = model(input_ids=input_ids, attention_mask=attention_mask)
            except:
                pass
            finally:
                if embed_hook: embed_hook.remove()
                h_cap_p.remove()
            
            pert_cat = get_logit_for_words(out_pert.logits[0, -1].float().cpu().numpy(), tokenizer, SLOT_WORDS["cat"])
            
            # Projection analysis
            pre_rms_proj = None
            post_rms_proj = None
            if "input" in captured_clean and "input" in captured_pert:
                clean_res = captured_clean["input"]
                pert_res = captured_pert["input"]
                shared_delta = pert_res - clean_res
                pre_rms_proj = float(np.dot(shared_delta, cat_readout_dir))
                
                if rms_weight is not None:
                    combined = clean_res + shared_delta
                    rms_norm_c = np.sqrt(np.mean(clean_res ** 2) + 1e-6)
                    rms_norm_p = np.sqrt(np.mean(combined ** 2) + 1e-6)
                    post_rms_c = clean_res / rms_norm_c * rms_weight
                    post_rms_p = combined / rms_norm_p * rms_weight
                    post_rms_delta = post_rms_p - post_rms_c
                    post_rms_proj = float(np.dot(post_rms_delta, cat_readout_dir))
            
            layer_data["pre_rms_proj"].append(round(pre_rms_proj, 4) if pre_rms_proj is not None else None)
            layer_data["post_rms_proj"].append(round(post_rms_proj, 4) if post_rms_proj is not None else None)
            layer_data["logit_delta_with_rms"].append(round(pert_cat - clean_cat, 4) if (pert_cat is not None and clean_cat is not None) else None)
            
            # 3. RMSNorm-bypass test
            if ln_module is not None:
                def make_bypass_hook(lpos):
                    """Replace RMSNorm with identity * scale (preserve total norm)"""
                    def hook(m, inp, out):
                        if isinstance(inp, tuple):
                            x = inp[0]
                        else:
                            x = inp
                        if isinstance(out, tuple):
                            out_norm = float(out[0].float().norm())
                            x_norm = float(x.float().norm())
                            scale = out_norm / (x_norm + 1e-8)
                            return (x * scale,) + out[1:]
                        else:
                            out_norm = float(out.float().norm())
                            x_norm = float(x.float().norm())
                            scale = out_norm / (x_norm + 1e-8)
                            return x * scale
                    return hook
                
                # Clean + bypass
                h_bypass_c = ln_module.register_forward_hook(make_bypass_hook(last_pos))
                with torch.no_grad():
                    out_cb = model(input_ids=input_ids, attention_mask=attention_mask)
                h_bypass_c.remove()
                clean_bypass_cat = get_logit_for_words(out_cb.logits[0, -1].float().cpu().numpy(), tokenizer, SLOT_WORDS["cat"])
                
                # Perturbed + bypass
                embed_hook2 = None
                if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                    embed_hook2 = model.model.embed_tokens.register_forward_hook(on_embed_pert)
                
                h_bypass_p = ln_module.register_forward_hook(make_bypass_hook(last_pos))
                try:
                    with torch.no_grad():
                        out_pb = model(input_ids=input_ids, attention_mask=attention_mask)
                except:
                    pass
                finally:
                    if embed_hook2: embed_hook2.remove()
                    h_bypass_p.remove()
                
                pert_bypass_cat = get_logit_for_words(out_pb.logits[0, -1].float().cpu().numpy(), tokenizer, SLOT_WORDS["cat"])
                delta_without_rms = pert_bypass_cat - clean_bypass_cat if (pert_bypass_cat is not None and clean_bypass_cat is not None) else None
            else:
                delta_without_rms = None
            
            layer_data["logit_delta_without_rms"].append(round(delta_without_rms, 4) if delta_without_rms is not None else None)
            gc.collect()
        
        # Aggregate
        result = {}
        for key in layer_data:
            vals = [v for v in layer_data[key] if v is not None]
            result[key] = round(float(np.mean(vals)), 4) if vals else None
        
        # Key comparisons
        proj_flip = "YES" if (result.get("pre_rms_proj") and result.get("post_rms_proj") and
                              result["pre_rms_proj"] * result["post_rms_proj"] < 0) else "NO"
        beh_flip = "YES" if (result.get("logit_delta_with_rms") is not None and 
                            result.get("logit_delta_without_rms") is not None and
                            result["logit_delta_with_rms"] * result["logit_delta_without_rms"] < 0) else "NO"
        
        result["projection_flip"] = proj_flip
        result["behavior_flip"] = beh_flip
        result["rmsnorm_causal_effect"] = "RMSNorm FLIPS behavior" if beh_flip == "YES" else "RMSNorm does NOT flip behavior"
        
        results[f"L{li}"] = result
        plog(f"  L{li}: pre_rms={result.get('pre_rms_proj')}, post_rms={result.get('post_rms_proj')}, "
             f"proj_flip={proj_flip}")
        plog(f"         with_rms_Δ={result.get('logit_delta_with_rms')}, "
             f"without_rms_Δ={result.get('logit_delta_without_rms')}, "
             f"beh_flip={beh_flip}")
    
    return results


# ===== Exp3: 标准化Direction/Scale分解 =====
def exp3_standardized_dir_scale(model, tokenizer, info, round_num=1):
    """
    统一定义direction_only/scale_only/full_vector/random_matched
    """
    plog(f"\n{'='*60}\nExp3: Standardized Direction/Scale Decomposition\n{'='*60}")
    
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
    
    target_layers = get_key_layers(info.name, n_layers)
    obj_names = OBJ_NAMES_SHORT if round_num == 1 else OBJ_NAMES[:5]
    plog(f"  Target layers: {target_layers}")
    
    results = {}
    
    for li in target_layers:
        layer_result = {}
        
        for obj_name in obj_names:
            text = f"The {obj_name} is a"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            # Clean forward + capture layer input
            captured = {}
            h_cap = layers[li].register_forward_hook(make_input_capture_hook(captured, "input", last_pos))
            with torch.no_grad():
                out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
            h_cap.remove()
            clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
            clean_metrics = compute_logit_metrics(clean_logits, tokenizer, SLOT_WORDS)
            
            if "input" not in captured:
                continue
            clean_res = captured["input"]
            clean_norm = float(np.linalg.norm(clean_res))
            clean_dir = clean_res / (clean_norm + 1e-8)
            
            # Get shared delta by embedding perturbation
            perturb_t = torch.tensor(1.0 * cat_dir, dtype=torch.float32).to(input_device).to(torch.bfloat16)
            captured_p = {}
            h_cap_p = layers[li].register_forward_hook(make_input_capture_hook(captured_p, "input", last_pos))
            embed_hook = None
            def on_embed(m, inp, out):
                if isinstance(out, torch.Tensor):
                    out = out.clone()
                    out[0, last_pos] = out[0, last_pos] + perturb_t.to(out.dtype)
                return out
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                embed_hook = model.model.embed_tokens.register_forward_hook(on_embed)
            try:
                with torch.no_grad():
                    out_pert = model(input_ids=input_ids, attention_mask=attention_mask)
            except:
                pass
            finally:
                if embed_hook: embed_hook.remove()
                h_cap_p.remove()
            
            if "input" not in captured_p:
                continue
            
            pert_res = captured_p["input"]
            shared_delta = pert_res - clean_res
            shared_norm = float(np.linalg.norm(shared_delta))
            shared_dir = shared_delta / (shared_norm + 1e-8)
            
            # Standardized injection vectors
            # direction_only: shared_direction * clean_norm (same norm as clean, different direction)
            vec_dir_only = shared_dir * clean_norm
            # scale_only: clean_direction * shared_norm (same direction as clean, different norm)
            vec_scale_only = clean_dir * shared_norm
            # full_vector: shared_direction * shared_norm
            vec_full = shared_delta
            # random_matched: random_direction * shared_norm
            rng = np.random.RandomState(42)
            random_dir = rng.randn(info.d_model).astype(np.float32)
            random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-8)
            vec_random = random_dir * shared_norm
            
            inj_results = {}
            for vec_name, vec_np in [("dir_only", vec_dir_only), ("scale_only", vec_scale_only),
                                      ("full_vector", vec_full), ("random_matched", vec_random)]:
                vec_t = torch.tensor(vec_np, dtype=torch.float32).to(input_device).to(torch.bfloat16)
                
                def make_inject_hook(inj_vec, lpos):
                    def hook(m, inp, out):
                        if isinstance(out, tuple):
                            new_out = out[0].clone()
                            new_out[0, lpos] = new_out[0, lpos] + inj_vec.to(device=new_out.device, dtype=new_out.dtype)
                            return (new_out,) + out[1:]
                        else:
                            new_out = out.clone()
                            new_out[0, lpos] = new_out[0, lpos] + inj_vec.to(device=new_out.device, dtype=new_out.dtype)
                            return new_out
                    return hook
                
                h_inj = layers[li].register_forward_hook(make_inject_hook(vec_t, last_pos))
                with torch.no_grad():
                    out_inj = model(input_ids=input_ids, attention_mask=attention_mask)
                h_inj.remove()
                
                inj_logits = out_inj.logits[0, -1].float().cpu().numpy()
                inj_metrics = compute_logit_metrics(inj_logits, tokenizer, SLOT_WORDS)
                
                deltas = {}
                for k in ["cat_logit", "color_logit", "part_logit", "material_logit", "entropy", "confidence"]:
                    if k in clean_metrics and k in inj_metrics:
                        deltas[f"{k}_delta"] = round(inj_metrics[k] - clean_metrics[k], 4)
                inj_results[vec_name] = deltas
            
            layer_result[obj_name] = inj_results
        
        # Average over objects
        avg_result = {}
        for vec_name in ["dir_only", "scale_only", "full_vector", "random_matched"]:
            avg_deltas = {}
            for obj_data in layer_result.values():
                if vec_name in obj_data:
                    for k, v in obj_data[vec_name].items():
                        if k not in avg_deltas:
                            avg_deltas[k] = []
                        avg_deltas[k].append(v)
            avg_result[vec_name] = {k: round(float(np.mean(v)), 4) for k, v in avg_deltas.items()}
        
        results[f"L{li}"] = avg_result
        plog(f"  L{li}: dir_catΔ={avg_result.get('dir_only',{}).get('cat_logit_delta')}, "
             f"scale_catΔ={avg_result.get('scale_only',{}).get('cat_logit_delta')}, "
             f"full_catΔ={avg_result.get('full_vector',{}).get('cat_logit_delta')}, "
             f"rand_catΔ={avg_result.get('random_matched',{}).get('cat_logit_delta')}")
    
    return results


# ===== Exp4: 多槽位读出接口画像 =====
def exp4_multi_slot_readout(model, tokenizer, info, round_num=1):
    """
    对最后2-3层测试shared注入对各属性槽位的影响
    """
    plog(f"\n{'='*60}\nExp4: Multi-Slot Readout Interface\n{'='*60}")
    
    n_layers = info.n_layers
    input_device = get_input_device(model)
    W_E = get_embedding_weight(model)
    
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["cat"]]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["opp_cat"]]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu().numpy()
    cat_dir = cat_dir / (np.linalg.norm(cat_dir) + 1e-8)
    
    target_layers = [n_layers-3, n_layers-2, n_layers-1]
    target_layers = [l for l in target_layers if l >= 0]
    obj_names = OBJ_NAMES_SHORT if round_num == 1 else OBJ_NAMES[:5]
    alpha = 1.0
    
    results = {}
    
    for li in target_layers:
        layer_result = {}
        
        for obj_name in obj_names:
            text = f"The {obj_name} is a"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            with torch.no_grad():
                out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
            clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
            clean_metrics = compute_logit_metrics(clean_logits, tokenizer, SLOT_WORDS)
            
            perturb_t = torch.tensor(alpha * cat_dir, dtype=torch.float32).to(input_device).to(torch.bfloat16)
            embed_hook = None
            def on_embed(m, inp, out):
                if isinstance(out, torch.Tensor):
                    out = out.clone()
                    out[0, last_pos] = out[0, last_pos] + perturb_t.to(out.dtype)
                return out
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                embed_hook = model.model.embed_tokens.register_forward_hook(on_embed)
            try:
                with torch.no_grad():
                    out_pert = model(input_ids=input_ids, attention_mask=attention_mask)
            except:
                pass
            finally:
                if embed_hook: embed_hook.remove()
            
            pert_logits = out_pert.logits[0, -1].float().cpu().numpy()
            pert_metrics = compute_logit_metrics(pert_logits, tokenizer, SLOT_WORDS)
            
            deltas = {}
            for k in clean_metrics:
                if k in pert_metrics:
                    deltas[k] = round(pert_metrics[k] - clean_metrics[k], 4)
            layer_result[obj_name] = deltas
        
        avg_deltas = {}
        for obj_data in layer_result.values():
            for k, v in obj_data.items():
                if k not in avg_deltas:
                    avg_deltas[k] = []
                avg_deltas[k].append(v)
        results[f"L{li}"] = {k: round(float(np.mean(v)), 4) for k, v in avg_deltas.items()}
        
        plog(f"  L{li}: catΔ={results[f'L{li}'].get('cat_logit_delta')}, "
             f"colorΔ={results[f'L{li}'].get('color_logit_delta')}, "
             f"partΔ={results[f'L{li}'].get('part_logit_delta')}")
    
    return results


# ===== Exp5: DS7B双峰精细复验 =====
def exp5_ds7b_bimodal_fine(model, tokenizer, info, round_num=1):
    """
    对DS7B做细粒度scale sweep; 对其他模型做粗粒度对比
    """
    plog(f"\n{'='*60}\nExp5: Bimodal Fine-Grained Verification\n{'='*60}")
    
    n_layers = info.n_layers
    input_device = get_input_device(model)
    W_E = get_embedding_weight(model)
    
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["cat"]]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["opp_cat"]]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu().numpy()
    cat_dir = cat_dir / (np.linalg.norm(cat_dir) + 1e-8)
    
    if info.name == "deepseek7b":
        scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                  1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0]
        target_layers = [0, 14, 23, 26, 27]
    else:
        scales = [0.1, 0.5, 1.0, 2.0, 4.0]
        target_layers = get_key_layers(info.name, n_layers)
    
    target_layers = [l for l in target_layers if l < n_layers]
    obj_names = OBJ_NAMES_SHORT if round_num == 1 else OBJ_NAMES[:6]
    plog(f"  Model={info.name}, {len(scales)} scales, layers={target_layers}")
    
    results = {}
    
    for li in target_layers:
        layer_data = {}
        
        for scale in scales:
            cat_deltas = []
            color_deltas = []
            part_deltas = []
            
            for obj_name in obj_names:
                text = f"The {obj_name} is a"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attention_mask = inputs["attention_mask"].to(input_device)
                last_pos = input_ids.shape[1] - 1
                
                with torch.no_grad():
                    out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
                clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
                clean_m = compute_logit_metrics(clean_logits, tokenizer, SLOT_WORDS)
                
                perturb_t = torch.tensor(scale * cat_dir, dtype=torch.float32).to(input_device).to(torch.bfloat16)
                embed_hook = None
                def on_embed(m, inp, out):
                    if isinstance(out, torch.Tensor):
                        out = out.clone()
                        out[0, last_pos] = out[0, last_pos] + perturb_t.to(out.dtype)
                    return out
                if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                    embed_hook = model.model.embed_tokens.register_forward_hook(on_embed)
                try:
                    with torch.no_grad():
                        out_pert = model(input_ids=input_ids, attention_mask=attention_mask)
                except:
                    pass
                finally:
                    if embed_hook: embed_hook.remove()
                
                pert_logits = out_pert.logits[0, -1].float().cpu().numpy()
                pert_m = compute_logit_metrics(pert_logits, tokenizer, SLOT_WORDS)
                
                cat_deltas.append(pert_m.get("cat_logit", 0) - clean_m.get("cat_logit", 0))
                color_deltas.append(pert_m.get("color_logit", 0) - clean_m.get("color_logit", 0))
                part_deltas.append(pert_m.get("part_logit", 0) - clean_m.get("part_logit", 0))
            
            layer_data[str(scale)] = {
                "cat_delta": round(float(np.mean(cat_deltas)), 4),
                "color_delta": round(float(np.mean(color_deltas)), 4),
                "part_delta": round(float(np.mean(part_deltas)), 4),
                "n_objects": len(obj_names),
            }
        
        results[f"L{li}"] = layer_data
        key_scales = [s for s in scales if s in [0.1, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]]
        report = f"  L{li}: " + ", ".join(f"s={s}→{layer_data[str(s)]['cat_delta']}" for s in key_scales if str(s) in layer_data)
        plog(report)
    
    return results


# ===== 主函数 =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Use: qwen3, glm4, deepseek7b")
        return
    
    plog(f"Phase 453: {model_name} round={round_num}")
    plog(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model: class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    results = {
        "model": model_name, "round": round_num,
        "model_info": {"class": info.model_class, "n_layers": info.n_layers,
                       "d_model": info.d_model, "mlp_type": info.mlp_type},
    }
    
    try:
        plog("=== Exp1: Projection-Causality Map ===")
        t1 = time.time()
        results["exp1"] = exp1_projection_causality_map(model, tokenizer, info, round_num)
        plog(f"Exp1 done: {time.time()-t1:.1f}s")
        gc.collect(); torch.cuda.empty_cache()
        
        plog("=== Exp2: RMSNorm Behavior Causal ===")
        t2 = time.time()
        results["exp2"] = exp2_rmsnorm_behavior_causal(model, tokenizer, info, round_num)
        plog(f"Exp2 done: {time.time()-t2:.1f}s")
        gc.collect(); torch.cuda.empty_cache()
        
        plog("=== Exp3: Standardized Dir/Scale ===")
        t3 = time.time()
        results["exp3"] = exp3_standardized_dir_scale(model, tokenizer, info, round_num)
        plog(f"Exp3 done: {time.time()-t3:.1f}s")
        gc.collect(); torch.cuda.empty_cache()
        
        plog("=== Exp4: Multi-Slot Readout ===")
        t4 = time.time()
        results["exp4"] = exp4_multi_slot_readout(model, tokenizer, info, round_num)
        plog(f"Exp4 done: {time.time()-t4:.1f}s")
        gc.collect(); torch.cuda.empty_cache()
        
        plog("=== Exp5: Bimodal Verification ===")
        t5 = time.time()
        results["exp5"] = exp5_ds7b_bimodal_fine(model, tokenizer, info, round_num)
        plog(f"Exp5 done: {time.time()-t5:.1f}s")
        
    except Exception as e:
        plog(f"!!! Error: {e}")
        import traceback; traceback.print_exc()
    
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase453_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    plog(f"Saved to {out_path}")
    
    release_model(model)
    plog(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
