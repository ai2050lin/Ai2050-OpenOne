"""
Phase 452: 方向-范数-RMSNorm读出接口的因果闭环验证
=================================================
核心目标:
1. Exp1: GLM4 L24 范数阈值曲线 — scale sweep证明norm-triggered suppression
2. Exp2: RMSNorm单独因果测试 — 受控向量过RMSNorm证明翻转能力
3. Exp3: 读出接口分型验证 — 三模型最后层的全面画像
4. Exp4: DS7B attention主导验证 — direction-only vs scale-only
5. Exp5: 多槽位验证 — category/color/part/material

用法:
  python tests/glm5/phase452_norm_rmsnorm_causal.py qwen3 1
  python tests/glm5/phase452_norm_rmsnorm_causal.py glm4 1
  python tests/glm5/phase452_norm_rmsnorm_causal.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import os, gc, time, json, logging, copy
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger("p452")

_last_log = [time.time()]
def plog(msg, interval=30):
    now = time.time()
    if now - _last_log[0] >= interval or any(k in msg.lower() for k in ['complete','step','failed','summary','exp','===']):
        log.info(msg)
        _last_log[0] = now

def plog_always(msg):
    log.info(msg)


def load_model_auto(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log.info(f"Loading {model_name} (bf16 + auto + sdpa)...")
    
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
            log.info(f"  attn={attn_impl}")
            break
        except Exception as e:
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
        log.info(f"  Layers: {gpu_l} GPU + {cpu_l} CPU")
    
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


def get_layer_norm_modules(layer):
    norms = {}
    if hasattr(layer, 'input_layernorm'):
        norms['input_ln'] = layer.input_layernorm
    elif hasattr(layer, 'ln1'):
        norms['input_ln'] = layer.ln1
    
    if hasattr(layer, 'post_attention_layernorm'):
        norms['post_attn_ln'] = layer.post_attention_layernorm
    elif hasattr(layer, 'ln2'):
        norms['post_attn_ln'] = layer.ln2
    
    return norms


def get_logit_for_words(logits_np, tokenizer, word_list):
    ids = []
    for w in word_list:
        tok_ids = tokenizer.encode(w, add_special_tokens=False)
        if tok_ids:
            ids.append(tok_ids[0])
    if not ids:
        return None
    return float(np.mean(logits_np[ids]))


def make_safe_inject_hook(vec_np, last_pos, beta=1.0):
    def hook(m, inp, out):
        if isinstance(out, tuple):
            new_out = out[0].clone()
            target_device = new_out.device
            target_dtype = new_out.dtype
            inj_t = torch.tensor(vec_np * beta, dtype=torch.float32)
            inj_t = inj_t.to(device=target_device, dtype=target_dtype)
            new_out[0, last_pos] = new_out[0, last_pos] + inj_t
            return (new_out,) + out[1:]
        return out
    return hook


def compute_logit_metrics(logits_np, tokenizer, slot_words):
    """Compute logit metrics for multiple slots + entropy + confidence"""
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


# === 共享数据准备 ===
SLOT_WORDS = {
    "cat": ["fruit", "food", "produce"],
    "opp_cat": ["animal", "dog", "cat"],
    "color": ["red", "green", "yellow"],
    "part": ["seed", "skin", "core", "stem"],
    "material": ["organic", "natural", "fresh"],
}

OBJ_NAMES = ["apple", "orange", "banana", "grape", "lemon", "peach", "pear", "mango", "plum", "cherry"]
OBJ_NAMES_SHORT = ["apple", "orange", "banana", "grape", "lemon"]

SCALES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


# ===== Exp1: GLM4 L24 范数阈值曲线 =====
def exp1_norm_threshold_curve(model, tokenizer, info):
    """
    在关键层注入同一shared方向, 只改变scale
    scale = 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0
    测: cat logit, color logit, part logit, entropy, confidence, RMSNorm前后投影
    
    目标: 如果低scale为正, 高scale转负, 证明norm-triggered suppression
    """
    print(f"\n{'='*60}")
    print("Exp1: Norm Threshold Curve (Scale Sweep)")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    W_E = get_embedding_weight(model)
    W_U = get_W_U(model, info.name).T  # [d_model, vocab]
    input_device = get_input_device(model)
    
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["cat"]]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["opp_cat"]]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu().numpy()
    cat_dir = cat_dir / (np.linalg.norm(cat_dir) + 1e-8)
    
    # 类别读出方向
    cat_readout_dir = W_U[:, cat_ids].mean(axis=1)
    cat_readout_dir = cat_readout_dir / (np.linalg.norm(cat_readout_dir) + 1e-8)
    
    # Step 1: 收集各层的shared方向
    obj_names = OBJ_NAMES_SHORT
    alpha = 1.0
    
    # 关键层: 反转区附近 + 最后几层
    if n_layers >= 30:
        mid = n_layers * 3 // 5
        target_layers = sorted(set(
            [0, n_layers//4] +
            list(range(max(0, mid-4), min(n_layers, mid+4))) +
            [n_layers-3, n_layers-2, n_layers-1]
        ))
    else:
        target_layers = sorted(set(
            [0, n_layers//2] +
            list(range(max(0, n_layers-3), n_layers))
        ))
    target_layers = [l for l in target_layers if l < n_layers]
    print(f"  Target layers: {target_layers}")
    
    # Step 2: 获取各层shared向量
    print("  Step 1: Computing shared vectors per layer...")
    t1 = time.time()
    
    shared_vecs = {}  # {li: shared_vector}
    
    for li in target_layers:
        deltas = {}
        for obj_name in obj_names:
            text = f"The {obj_name} is a"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            base_h = {}
            def make_h_b(lidx):
                def hook(m, inp, out):
                    if isinstance(out, tuple):
                        base_h[lidx] = out[0][0, last_pos].detach().float().cpu().numpy()
                return hook
            hooks_b = [layers[li].register_forward_hook(make_h_b(li))]
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            for h in hooks_b:
                h.remove()
            
            perturb_vec = (alpha * cat_dir)
            perturb_t = torch.tensor(perturb_vec, dtype=torch.float32).to(input_device).to(torch.bfloat16)
            pert_h = {}
            def make_h_p(lidx):
                def hook(m, inp, out):
                    if isinstance(out, tuple):
                        pert_h[lidx] = out[0][0, last_pos].detach().float().cpu().numpy()
                return hook
            hooks_p = [layers[li].register_forward_hook(make_h_p(li))]
            
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
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except:
                pass
            finally:
                if embed_hook:
                    embed_hook.remove()
                for h in hooks_p:
                    h.remove()
            
            if li in base_h and li in pert_h:
                d = pert_h[li] - base_h[li]
                if np.linalg.norm(d) > 1e-8:
                    deltas[obj_name] = d
        
        if len(deltas) >= 3:
            delta_matrix = np.stack(list(deltas.values()))
            shared_vecs[li] = delta_matrix.mean(axis=0)
        
        plog(f"  Exp1: L{li} shared computed ({len(deltas)} objects)")
    
    print(f"  Shared vectors computed for {len(shared_vecs)} layers ({time.time()-t1:.1f}s)")
    
    # Step 3: 基准logits
    test_text = "The apple is a"
    test_inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=64)
    test_input_ids = test_inputs["input_ids"].to(input_device)
    test_attention_mask = test_inputs["attention_mask"].to(input_device)
    test_last_pos = test_input_ids.shape[1] - 1
    
    with torch.no_grad():
        out_base = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
        base_logits = out_base.logits[0, -1].float().cpu().numpy()
    
    base_metrics = compute_logit_metrics(base_logits, tokenizer, SLOT_WORDS)
    print(f"  Base: cat={base_metrics.get('cat_logit','N/A'):.2f}, "
          f"color={base_metrics.get('color_logit','N/A'):.2f}, "
          f"entropy={base_metrics.get('entropy','N/A'):.2f}")
    
    # Step 4: Scale sweep
    print("  Step 2: Running scale sweep...")
    t2 = time.time()
    
    results = {"base": base_metrics, "layers": {}}
    
    for li in target_layers:
        if li not in shared_vecs:
            continue
        
        shared_vec = shared_vecs[li]
        shared_norm = np.linalg.norm(shared_vec)
        shared_unit = shared_vec / (shared_norm + 1e-8)
        
        layer_result = {"shared_norm": round(float(shared_norm), 4)}
        scale_results = {}
        
        for scale in SCALES:
            # 注入 shared_unit * scale * shared_norm (等效于原始范数的scale倍)
            inj_vec = shared_unit * scale * shared_norm
            
            inj_hook = layers[li].register_forward_hook(
                make_safe_inject_hook(inj_vec, test_last_pos, beta=1.0))
            
            with torch.no_grad():
                try:
                    out_inj = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                    inj_logits = out_inj.logits[0, -1].float().cpu().numpy()
                except Exception as e:
                    inj_hook.remove()
                    continue
            
            inj_hook.remove()
            
            inj_metrics = compute_logit_metrics(inj_logits, tokenizer, SLOT_WORDS)
            
            # 计算delta
            deltas = {}
            for key in ["cat_logit", "color_logit", "part_logit", "material_logit"]:
                if key in base_metrics and key in inj_metrics:
                    deltas[key.replace("_logit", "_delta")] = round(inj_metrics[key] - base_metrics[key], 4)
            deltas["entropy_delta"] = round(inj_metrics.get("entropy", 0) - base_metrics.get("entropy", 0), 4)
            deltas["confidence_delta"] = round(inj_metrics.get("confidence", 0) - base_metrics.get("confidence", 0), 6)
            
            scale_results[str(scale)] = deltas
            
            cat_d = deltas.get("cat_delta", 0)
            plog(f"  L{li} scale={scale:.2f}: catΔ={cat_d:+.4f}")
        
        layer_result["scale_curve"] = scale_results
        
        # 投影到cat_readout_dir
        proj_readout = float(np.dot(shared_unit, cat_readout_dir))
        layer_result["shared_unit_proj_cat_readout"] = round(proj_readout, 4)
        layer_result["shared_unit_proj_cat_embed"] = round(float(np.dot(shared_unit, cat_dir)), 4)
        
        results["layers"][f"L{li}"] = layer_result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总
    print(f"\n  === Exp1 Summary: Norm Threshold Curve ===")
    print(f"  Layer | shared_norm | dir→cat_readout | scale=0.1→cat | scale=1.0→cat | scale=4.0→cat | sign_change?")
    for lk in sorted(results["layers"].keys()):
        lr = results["layers"][lk]
        sn = lr.get("shared_norm", 0)
        dr = lr.get("shared_unit_proj_cat_readout", 0)
        sc = lr.get("scale_curve", {})
        s01 = sc.get("0.1", {}).get("cat_delta", "N/A")
        s10 = sc.get("1.0", {}).get("cat_delta", "N/A")
        s40 = sc.get("4.0", {}).get("cat_delta", "N/A")
        # 检查符号变化
        sign_change = ""
        if isinstance(s01, (int, float)) and isinstance(s40, (int, float)):
            sign_change = "YES→NEG" if s01 > 0 and s40 < 0 else ("YES→POS" if s01 < 0 and s40 > 0 else "NO")
        print(f"  {lk:5s} | {sn:.3f}      | {dr:+.4f}           | {s01}          | {s10}          | {s40}          | {sign_change}")
    
    print(f"  Exp1 complete ({time.time()-t2:.1f}s)")
    return results


# ===== Exp2: RMSNorm单独因果测试 =====
def exp2_rmsnorm_standalone_causal(model, tokenizer, info):
    """
    构造受控残差向量 v = clean_residual + alpha * shared_direction
    分别测:
    1. pre_RMSNorm投影 vs post_RMSNorm投影
    2. 通过RMSNorm前后的方向是否翻转
    3. 不同alpha下的RMSNorm变换
    
    关键: 不运行模型, 只取RMSNorm模块单独前向, 证明RMSNorm本身能翻转方向
    """
    print(f"\n{'='*60}")
    print("Exp2: RMSNorm Standalone Causal Test")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    W_U = get_W_U(model, info.name).T  # [d_model, vocab]
    input_device = get_input_device(model)
    W_E = get_embedding_weight(model)
    
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["cat"]]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["opp_cat"]]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu().numpy()
    cat_dir = cat_dir / (np.linalg.norm(cat_dir) + 1e-8)
    
    # 类别读出方向
    cat_readout_dir = W_U[:, cat_ids].mean(axis=1)
    cat_readout_dir = cat_readout_dir / (np.linalg.norm(cat_readout_dir) + 1e-8)
    
    obj_name = "apple"
    alpha = 1.0
    
    # 关键层
    if n_layers >= 30:
        mid = n_layers * 3 // 5
        target_layers = sorted(set(
            list(range(max(0, mid-5), min(n_layers, mid+5))) +
            [n_layers-2, n_layers-1]
        ))
    else:
        target_layers = list(range(max(0, n_layers-5), n_layers))
    target_layers = [l for l in target_layers if l < n_layers]
    print(f"  Target layers: {target_layers}")
    
    # Step 1: 获取clean residual和shared方向
    text = f"The {obj_name} is a"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    last_pos = input_ids.shape[1] - 1
    
    # 收集clean状态各层residual
    clean_residuals = {}
    def make_resid_hook(lidx):
        def hook(m, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                clean_residuals[lidx] = inp[0][0, last_pos].detach().float().cpu()
        return hook
    
    hooks = []
    for li in target_layers:
        layer = layers[li]
        norm_modules = get_layer_norm_modules(layer)
        if 'input_ln' in norm_modules:
            hooks.append(norm_modules['input_ln'].register_forward_hook(
                make_resid_hook(li)))
    
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    for h in hooks:
        h.remove()
    
    # 收集perturbed状态
    perturb_vec = (alpha * cat_dir)
    perturb_t = torch.tensor(perturb_vec, dtype=torch.float32).to(input_device).to(torch.bfloat16)
    
    pert_residuals = {}
    def make_resid_hook_p(lidx):
        def hook(m, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                pert_residuals[lidx] = inp[0][0, last_pos].detach().float().cpu()
        return hook
    
    hooks = []
    embed_hook = None
    def on_embed(m, inp, out):
        if isinstance(out, torch.Tensor):
            out = out.clone()
            out[0, last_pos] = out[0, last_pos] + perturb_t.to(out.dtype)
        return out
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_hook = model.model.embed_tokens.register_forward_hook(on_embed)
    
    for li in target_layers:
        layer = layers[li]
        norm_modules = get_layer_norm_modules(layer)
        if 'input_ln' in norm_modules:
            hooks.append(norm_modules['input_ln'].register_forward_hook(
                make_resid_hook_p(li)))
    
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    for h in hooks:
        h.remove()
    if embed_hook:
        embed_hook.remove()
    
    # Step 2: 提取RMSNorm模块并做独立前向
    print("  Step 2: Running RMSNorm standalone forward pass...")
    t2 = time.time()
    
    results = {}
    
    for li in target_layers:
        if li not in clean_residuals or li not in pert_residuals:
            continue
        
        layer = layers[li]
        norm_modules = get_layer_norm_modules(layer)
        if 'input_ln' not in norm_modules:
            continue
        
        rmsnorm = norm_modules['input_ln']
        clean_res = clean_residuals[li]  # [d_model]
        pert_res = pert_residuals[li]    # [d_model]
        
        # Shared delta = pert_res - clean_res
        shared_delta = pert_res - clean_res
        shared_norm = float(shared_delta.norm())
        
        if shared_norm < 1e-8:
            continue
        
        layer_result = {
            "shared_delta_norm": round(shared_norm, 4),
            "clean_res_norm": round(float(clean_res.norm()), 4),
        }
        
        # 投影到cat_readout_dir
        cat_readout_t = torch.tensor(cat_readout_dir, dtype=torch.float32)
        proj_before = float(torch.dot(shared_delta, cat_readout_t))
        layer_result["pre_rms_proj_cat_readout"] = round(proj_before, 4)
        
        # Test different alpha scales
        alpha_sweep_results = {}
        for a in [0.5, 1.0, 2.0, 4.0, 8.0]:
            # Construct: clean_res + a * shared_delta
            test_input = clean_res + a * shared_delta
            test_input_2d = test_input.unsqueeze(0)  # [1, d_model]
            
            # Get the device of the RMSNorm module
            rms_device = next(rmsnorm.parameters()).device
            test_input_2d = test_input_2d.to(rms_device)
            
            with torch.no_grad():
                try:
                    test_output = rmsnorm(test_input_2d)
                    if isinstance(test_output, tuple):
                        test_output = test_output[0]
                    test_output = test_output[0].cpu().float()  # [d_model]
                except Exception as e:
                    layer_result[f"rmsnorm_error_alpha{a}"] = str(e)
                    continue
            
            # Project the output residual's shared component
            # output_shared = test_output - (clean_output) but we need clean output too
            # Instead: compute projection of the whole vector to cat_readout
            proj_after = float(torch.dot(test_output, cat_readout_t))
            layer_result[f"post_rms_proj_cat_readout_alpha{a}"] = round(proj_after, 4)
            
            # Also run clean through RMSNorm for comparison
            clean_input_2d = clean_res.unsqueeze(0).to(rms_device)
            with torch.no_grad():
                try:
                    clean_output = rmsnorm(clean_input_2d)
                    if isinstance(clean_output, tuple):
                        clean_output = clean_output[0]
                    clean_output = clean_output[0].cpu().float()
                except:
                    clean_output = None
            
            if clean_output is not None:
                # Shared delta in output space
                output_shared = test_output - clean_output
                proj_output_shared = float(torch.dot(output_shared, cat_readout_t))
                alpha_sweep_results[str(a)] = {
                    "input_shared_proj": round(proj_before * a / 1.0, 4),  # scaled
                    "output_shared_proj": round(proj_output_shared, 4),
                    "sign_flip": "YES" if proj_before * a * proj_output_shared < 0 else "NO",
                    "norm_ratio": round(float(output_shared.norm() / (shared_norm * a + 1e-8)), 4),
                }
        
        layer_result["alpha_sweep"] = alpha_sweep_results
        
        # Also do a direct test: put clean+shared through RMSNorm, check sign of shared component
        # This is the most direct test of "does RMSNorm flip shared direction?"
        direct_test = clean_res + shared_delta  # alpha=1.0
        direct_2d = direct_test.unsqueeze(0).to(next(rmsnorm.parameters()).device)
        with torch.no_grad():
            try:
                direct_out = rmsnorm(direct_2d)
                if isinstance(direct_out, tuple):
                    direct_out = direct_out[0]
                direct_out = direct_out[0].cpu().float()
            except:
                direct_out = None
        
        if direct_out is not None and clean_output is not None:
            output_shared_direct = direct_out - clean_output
            proj_direct = float(torch.dot(output_shared_direct, cat_readout_t))
            layer_result["rmsnorm_shared_proj_cat_readout"] = round(proj_direct, 4)
            layer_result["rmsnorm_flip_alpha1"] = "YES" if proj_before * proj_direct < 0 else "NO"
        
        results[f"L{li}"] = layer_result
        plog(f"  Exp2: L{li} pre_rms={proj_before:+.3f}, flip_alpha1={layer_result.get('rmsnorm_flip_alpha1', 'N/A')}")
    
    # 汇总
    print(f"\n  === Exp2 Summary: RMSNorm Standalone Causal ===")
    print(f"  Layer | pre_rms_proj | post_rms_proj(alpha=1) | flip? | alpha=2 flip? | alpha=4 flip?")
    for lk in sorted(results.keys()):
        lr = results[lk]
        pre = lr.get("pre_rms_proj_cat_readout", "N/A")
        post1 = lr.get("rmsnorm_shared_proj_cat_readout", "N/A")
        flip1 = lr.get("rmsnorm_flip_alpha1", "N/A")
        as2 = lr.get("alpha_sweep", {}).get("2.0", {}).get("sign_flip", "N/A")
        as4 = lr.get("alpha_sweep", {}).get("4.0", {}).get("sign_flip", "N/A")
        print(f"  {lk:5s} | {pre:+.4f}    | {post1:+.4f}               | {flip1}  | {as2}            | {as4}")
    
    print(f"  Exp2 complete ({time.time()-t2:.1f}s)")
    return results


# ===== Exp3: 读出接口分型验证 =====
def exp3_readout_interface_profile(model, tokenizer, info):
    """
    在最后3层做全面画像:
    - remove shared / negate shared / direction-only shared / scale-only shared
    - remove private / negate private / direction-only private
    - zero MLP output / zero attention output
    - Measure: cat, color, part, material, entropy, confidence, top-k candidate family
    
    建立ReadoutInterfaceProfile
    """
    print(f"\n{'='*60}")
    print("Exp3: Readout Interface Profile")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    W_E = get_embedding_weight(model)
    W_U = get_W_U(model, info.name).T  # [d_model, vocab]
    input_device = get_input_device(model)
    
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["cat"]]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["opp_cat"]]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu().numpy()
    cat_dir = cat_dir / (np.linalg.norm(cat_dir) + 1e-8)
    
    obj_names = OBJ_NAMES_SHORT
    alpha = 1.0
    
    # 关键层: 最后3层
    target_layers = sorted(set(
        list(range(max(0, n_layers-3), n_layers))
    ))
    target_layers = [l for l in target_layers if 0 <= l < n_layers]
    print(f"  Target layers: {target_layers}")
    
    # Step 1: 分解各层shared/private
    print("  Step 1: Computing shared/private decomposition...")
    t1 = time.time()
    
    decomposition = {}
    for li in target_layers:
        deltas = {}
        for obj_name in obj_names:
            text = f"The {obj_name} is a"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            base_h = {}
            def make_h_b(lidx):
                def hook(m, inp, out):
                    if isinstance(out, tuple):
                        base_h[lidx] = out[0][0, last_pos].detach().float().cpu().numpy()
                return hook
            hooks_b = [layers[li].register_forward_hook(make_h_b(li))]
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            for h in hooks_b:
                h.remove()
            
            perturb_vec = (alpha * cat_dir)
            perturb_t = torch.tensor(perturb_vec, dtype=torch.float32).to(input_device).to(torch.bfloat16)
            pert_h = {}
            def make_h_p(lidx):
                def hook(m, inp, out):
                    if isinstance(out, tuple):
                        pert_h[lidx] = out[0][0, last_pos].detach().float().cpu().numpy()
                return hook
            hooks_p = [layers[li].register_forward_hook(make_h_p(li))]
            
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
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except:
                pass
            finally:
                if embed_hook:
                    embed_hook.remove()
                for h in hooks_p:
                    h.remove()
            
            if li in base_h and li in pert_h:
                d = pert_h[li] - base_h[li]
                if np.linalg.norm(d) > 1e-8:
                    deltas[obj_name] = d
        
        if len(deltas) >= 3:
            delta_matrix = np.stack(list(deltas.values()))
            shared = delta_matrix.mean(axis=0)
            private = {o: d - shared for o, d in deltas.items()}
            decomposition[li] = {"shared": shared, "private": private}
        
        plog(f"  Exp3: L{li} decomposition done ({len(deltas)} objects)")
    
    print(f"  Decomposition done ({time.time()-t1:.1f}s)")
    
    # Step 2: 获取MLP output和attention output的shared分量
    print("  Step 2: Collecting MLP/attention output shared components...")
    
    test_text = "The apple is a"
    test_inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=64)
    test_input_ids = test_inputs["input_ids"].to(input_device)
    test_attention_mask = test_inputs["attention_mask"].to(input_device)
    test_last_pos = test_input_ids.shape[1] - 1
    
    # 收集各层的MLP和attn输出
    component_outputs = {}  # {li: {"mlp_out": vec, "attn_out": vec}}
    for li in target_layers:
        layer = layers[li]
        comp_data = {}
        
        # MLP输出
        mlp_h = {}
        def make_mlp_h(lidx):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    mlp_h[lidx] = out[0][0, test_last_pos].detach().float().cpu().numpy()
                else:
                    mlp_h[lidx] = out[0, test_last_pos].detach().float().cpu().numpy()
            return hook
        hooks_m = [layer.mlp.register_forward_hook(make_mlp_h(li))] if hasattr(layer, 'mlp') else []
        
        # Attn输出
        attn_h = {}
        def make_attn_h(lidx):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    attn_h[lidx] = out[0][0, test_last_pos].detach().float().cpu().numpy()
                else:
                    attn_h[lidx] = out[0, test_last_pos].detach().float().cpu().numpy()
            return hook
        hooks_a = [layer.self_attn.register_forward_hook(make_attn_h(li))] if hasattr(layer, 'self_attn') else []
        
        with torch.no_grad():
            _ = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
        
        for h in hooks_m + hooks_a:
            h.remove()
        
        if li in mlp_h:
            comp_data["mlp_out"] = mlp_h[li]
        if li in attn_h:
            comp_data["attn_out"] = attn_h[li]
        
        component_outputs[li] = comp_data
    
    # Step 3: 基准logits
    with torch.no_grad():
        out_base = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
        base_logits = out_base.logits[0, -1].float().cpu().numpy()
    
    base_metrics = compute_logit_metrics(base_logits, tokenizer, SLOT_WORDS)
    
    # Top-k candidates
    top10_ids = np.argsort(base_logits)[-10:][::-1]
    top10_words = [tokenizer.decode([i]).strip() for i in top10_ids]
    top10_scores = [float(base_logits[i]) for i in top10_ids]
    
    results = {
        "base": {**base_metrics, "top10_words": top10_words, "top10_scores": [round(s, 4) for s in top10_scores]},
        "layers": {}
    }
    
    # Step 4: 各层各操作
    print("  Step 3: Running readout interface operations...")
    t3 = time.time()
    
    for li in target_layers:
        if li not in decomposition:
            continue
        
        shared_vec = decomposition[li]["shared"]
        private_vec = decomposition[li]["private"].get("apple", np.zeros(d_model))
        shared_norm = np.linalg.norm(shared_vec)
        private_norm = np.linalg.norm(private_vec)
        shared_unit = shared_vec / (shared_norm + 1e-8)
        private_unit = private_vec / (private_norm + 1e-8)
        
        layer_result = {
            "shared_norm": round(float(shared_norm), 4),
            "private_norm": round(float(private_norm), 4),
        }
        
        # 操作定义: (name, inject_vector)
        operations = {
            # Shared操作
            "remove_shared": -shared_vec,           # 移除: 减去shared
            "negate_shared": -2.0 * shared_vec,     # 反转: 减去2倍shared
            "inject_shared": shared_vec,            # 注入: 加shared
            "dir_only_shared": shared_unit,          # 只方向: 单位向量
            "scale_shared_2x": shared_vec,           # (通过beta控制)
            # Private操作
            "remove_private": -private_vec,
            "negate_private": -2.0 * private_vec,
            "inject_private": private_vec,
            "dir_only_private": private_unit,
            # Scale-only: 随机方向 + shared范数
            "scale_only_shared_norm": np.random.randn(d_model) * shared_norm / (np.linalg.norm(np.random.randn(d_model)) + 1e-8),
        }
        
        # Beta控制: scale_shared_2x 用beta=2.0
        beta_map = {
            "scale_shared_2x": 2.0,
        }
        
        # MLP/Attn zeroing
        comp_data = component_outputs.get(li, {})
        if "mlp_out" in comp_data:
            operations["zero_mlp"] = -comp_data["mlp_out"]  # 减去MLP输出等效于置零
        if "attn_out" in comp_data:
            operations["zero_attn"] = -comp_data["attn_out"]  # 减去attn输出等效于置零
        
        for op_name, inj_vec in operations.items():
            beta = beta_map.get(op_name, 1.0)
            
            inj_hook = layers[li].register_forward_hook(
                make_safe_inject_hook(inj_vec, test_last_pos, beta=beta))
            
            with torch.no_grad():
                try:
                    out_inj = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                    inj_logits = out_inj.logits[0, -1].float().cpu().numpy()
                except Exception as e:
                    inj_hook.remove()
                    layer_result[op_name] = {"error": str(e)}
                    continue
            
            inj_hook.remove()
            
            inj_metrics = compute_logit_metrics(inj_logits, tokenizer, SLOT_WORDS)
            
            # Deltas
            deltas = {}
            for key in ["cat_logit", "color_logit", "part_logit", "material_logit"]:
                if key in base_metrics and key in inj_metrics:
                    deltas[key.replace("_logit", "_delta")] = round(inj_metrics[key] - base_metrics[key], 4)
            deltas["entropy_delta"] = round(inj_metrics.get("entropy", 0) - base_metrics.get("entropy", 0), 4)
            deltas["confidence_delta"] = round(inj_metrics.get("confidence", 0) - base_metrics.get("confidence", 0), 6)
            
            # Top-5 candidate shift
            top5_ids = np.argsort(inj_logits)[-5:][::-1]
            top5_words = [tokenizer.decode([i]).strip() for i in top5_ids]
            deltas["top5"] = top5_words
            
            layer_result[op_name] = deltas
            
            cat_d = deltas.get("cat_delta", 0)
            plog(f"  L{li} {op_name}: catΔ={cat_d:+.3f}")
        
        results["layers"][f"L{li}"] = layer_result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总
    print(f"\n  === Exp3 Summary: Readout Interface Profile ===")
    for lk in sorted(results["layers"].keys()):
        lr = results["layers"][lk]
        print(f"\n  {lk} (shared_norm={lr.get('shared_norm',0):.2f}, private_norm={lr.get('private_norm',0):.2f}):")
        for op in ["remove_shared", "negate_shared", "inject_shared", "dir_only_shared",
                    "remove_private", "inject_private", "dir_only_private", 
                    "scale_only_shared_norm", "zero_mlp", "zero_attn"]:
            if op in lr and isinstance(lr[op], dict):
                cat_d = lr[op].get("cat_delta", "N/A")
                col_d = lr[op].get("color_delta", "N/A")
                print(f"    {op:25s}: catΔ={cat_d}, colorΔ={col_d}")
    
    print(f"  Exp3 complete ({time.time()-t3:.1f}s)")
    return results


# ===== Exp4: DS7B attention主导验证 =====
def exp4_attention_dominance_verify(model, tokenizer, info):
    """
    验证DS7B最后一层attention主导:
    1. 捕获attention输出和MLP输出的范数
    2. direction-only attention injection
    3. scale-only attention injection  
    4. norm-matched MLP injection (MLP注入但范数匹配attention)
    5. attention ablation (减去attn输出)
    
    测: cat logit, entropy, top-k
    """
    print(f"\n{'='*60}")
    print("Exp4: Attention Dominance Verification")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    W_E = get_embedding_weight(model)
    W_U = get_W_U(model, info.name).T
    input_device = get_input_device(model)
    
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["cat"]]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in SLOT_WORDS["opp_cat"]]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu().numpy()
    cat_dir = cat_dir / (np.linalg.norm(cat_dir) + 1e-8)
    
    # 类别读出方向
    cat_readout_dir = W_U[:, cat_ids].mean(axis=1)
    cat_readout_dir = cat_readout_dir / (np.linalg.norm(cat_readout_dir) + 1e-8)
    
    # 关键层: 最后2层
    target_layers = [n_layers-2, n_layers-1]
    target_layers = [l for l in target_layers if 0 <= l < n_layers]
    print(f"  Target layers: {target_layers}")
    
    test_text = "The apple is a"
    test_inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=64)
    test_input_ids = test_inputs["input_ids"].to(input_device)
    test_attention_mask = test_inputs["attention_mask"].to(input_device)
    test_last_pos = test_input_ids.shape[1] - 1
    
    # Step 1: 收集attn和mlp输出
    component_data = {}
    for li in target_layers:
        layer = layers[li]
        comp = {}
        
        mlp_h = {}
        def make_mlp_h(lidx):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    mlp_h[lidx] = out[0][0, test_last_pos].detach().float().cpu().numpy()
                else:
                    mlp_h[lidx] = out[0, test_last_pos].detach().float().cpu().numpy()
            return hook
        hooks_m = [layer.mlp.register_forward_hook(make_mlp_h(li))] if hasattr(layer, 'mlp') else []
        
        attn_h = {}
        def make_attn_h(lidx):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    attn_h[lidx] = out[0][0, test_last_pos].detach().float().cpu().numpy()
                else:
                    attn_h[lidx] = out[0, test_last_pos].detach().float().cpu().numpy()
            return hook
        hooks_a = [layer.self_attn.register_forward_hook(make_attn_h(li))] if hasattr(layer, 'self_attn') else []
        
        with torch.no_grad():
            _ = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
        
        for h in hooks_m + hooks_a:
            h.remove()
        
        if li in mlp_h:
            comp["mlp_out"] = mlp_h[li]
        if li in attn_h:
            comp["attn_out"] = attn_h[li]
        
        component_data[li] = comp
        plog(f"  Exp4: L{li} components collected")
    
    # Step 2: 基准logits
    with torch.no_grad():
        out_base = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
        base_logits = out_base.logits[0, -1].float().cpu().numpy()
    base_metrics = compute_logit_metrics(base_logits, tokenizer, SLOT_WORDS)
    
    results = {"base": base_metrics, "layers": {}}
    
    # Step 3: 各种操作
    for li in target_layers:
        comp = component_data.get(li, {})
        if "attn_out" not in comp or "mlp_out" not in comp:
            continue
        
        attn_out = comp["attn_out"]
        mlp_out = comp["mlp_out"]
        attn_norm = np.linalg.norm(attn_out)
        mlp_norm = np.linalg.norm(mlp_out)
        
        attn_unit = attn_out / (attn_norm + 1e-8)
        mlp_unit = mlp_out / (mlp_norm + 1e-8)
        
        layer_result = {
            "attn_out_norm": round(float(attn_norm), 4),
            "mlp_out_norm": round(float(mlp_norm), 4),
            "attn_proj_cat_readout": round(float(np.dot(attn_unit, cat_readout_dir)), 4),
            "mlp_proj_cat_readout": round(float(np.dot(mlp_unit, cat_readout_dir)), 4),
            "norm_ratio_attn_vs_mlp": round(float(attn_norm / (mlp_norm + 1e-8)), 4),
        }
        
        # 操作: (name, inject_vector)
        operations = {
            # Attention操作
            "zero_attn": -attn_out,                        # 移除attention输出
            "dir_only_attn": attn_unit,                     # 只attention方向(范数=1)
            "scale_only_attn": np.random.randn(d_model) * attn_norm / (np.linalg.norm(np.random.randn(d_model)) + 1e-8),  # 随机方向+attn范数
            "negate_attn": -2.0 * attn_out,                 # 反转attention
            "double_attn": attn_out,                         # 双倍attention(beta=2)
            # MLP操作
            "zero_mlp": -mlp_out,                           # 移除MLP输出
            "dir_only_mlp": mlp_unit,                        # 只MLP方向
            "negate_mlp": -2.0 * mlp_out,                   # 反转MLP
            # MLP with attn-norm: MLP方向 * attn范数
            "mlp_dir_attn_scale": mlp_unit * attn_norm,     # MLP方向 * attn范数
            # Attn with mlp-norm: attn方向 * mlp范数
            "attn_dir_mlp_scale": attn_unit * mlp_norm,     # attn方向 * mlp范数
        }
        
        beta_map = {
            "double_attn": 2.0,
        }
        
        for op_name, inj_vec in operations.items():
            beta = beta_map.get(op_name, 1.0)
            
            inj_hook = layers[li].register_forward_hook(
                make_safe_inject_hook(inj_vec, test_last_pos, beta=beta))
            
            with torch.no_grad():
                try:
                    out_inj = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                    inj_logits = out_inj.logits[0, -1].float().cpu().numpy()
                except Exception as e:
                    inj_hook.remove()
                    layer_result[op_name] = {"error": str(e)}
                    continue
            
            inj_hook.remove()
            
            inj_metrics = compute_logit_metrics(inj_logits, tokenizer, SLOT_WORDS)
            
            deltas = {}
            for key in ["cat_logit", "color_logit", "part_logit", "material_logit"]:
                if key in base_metrics and key in inj_metrics:
                    deltas[key.replace("_logit", "_delta")] = round(inj_metrics[key] - base_metrics[key], 4)
            deltas["entropy_delta"] = round(inj_metrics.get("entropy", 0) - base_metrics.get("entropy", 0), 4)
            
            layer_result[op_name] = deltas
            
            cat_d = deltas.get("cat_delta", 0)
            plog(f"  L{li} {op_name}: catΔ={cat_d:+.3f}")
        
        results["layers"][f"L{li}"] = layer_result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总
    print(f"\n  === Exp4 Summary: Attention Dominance ===")
    for lk in sorted(results["layers"].keys()):
        lr = results["layers"][lk]
        print(f"\n  {lk} (attn_norm={lr.get('attn_out_norm',0):.1f}, mlp_norm={lr.get('mlp_out_norm',0):.1f}, "
              f"ratio={lr.get('norm_ratio_attn_vs_mlp',0):.2f}):")
        for op in ["zero_attn", "zero_mlp", "dir_only_attn", "dir_only_mlp", 
                    "negate_attn", "negate_mlp", "mlp_dir_attn_scale", "attn_dir_mlp_scale"]:
            if op in lr and isinstance(lr[op], dict):
                cat_d = lr[op].get("cat_delta", "N/A")
                ent_d = lr[op].get("entropy_delta", "N/A")
                print(f"    {op:25s}: catΔ={cat_d}, entΔ={ent_d}")
    
    return results


# ===== Exp5: 多槽位验证 =====
def exp5_multi_slot_verification(model, tokenizer, info):
    """
    在关键层做shared/private注入, 测多槽位效果:
    - category, color, part, material, habitat
    
    使用更多对象(10个)和更多类别(5类)
    """
    print(f"\n{'='*60}")
    print("Exp5: Multi-Slot Verification")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    W_E = get_embedding_weight(model)
    W_U = get_W_U(model, info.name).T
    input_device = get_input_device(model)
    
    # 扩展槽位词
    extended_slots = {
        "cat_fruit": ["fruit", "food", "produce"],
        "cat_animal": ["animal", "creature", "beast"],
        "cat_tool": ["tool", "instrument", "device"],
        "cat_vehicle": ["vehicle", "car", "transport"],
        "color": ["red", "green", "yellow", "blue"],
        "part": ["seed", "skin", "core", "stem", "wheel"],
        "material": ["organic", "natural", "fresh", "metal"],
        "habitat": ["tree", "farm", "garden", "field", "forest"],
        "function": ["eat", "grow", "use", "drive"],
    }
    
    # 多类别对象
    multi_objects = {
        "fruit": ["apple", "orange", "banana", "grape", "lemon"],
        "animal": ["dog", "cat", "horse", "cow", "sheep"],
        "tool": ["hammer", "screwdriver", "wrench", "drill", "saw"],
    }
    
    # 每个类别测3个对象 (减少测试时间)
    test_objects = []
    for cat, objs in multi_objects.items():
        test_objects.extend([(cat, o) for o in objs[:3]])
    
    # 类别方向
    cat_fruit_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in extended_slots["cat_fruit"]]
    cat_animal_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in extended_slots["cat_animal"]]
    cat_tool_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in extended_slots["cat_tool"]]
    
    fruit_dir = (W_E[cat_fruit_ids].mean(dim=0) - W_E[cat_animal_ids].mean(dim=0)).cpu().numpy()
    fruit_dir = fruit_dir / (np.linalg.norm(fruit_dir) + 1e-8)
    
    tool_dir = (W_E[cat_tool_ids].mean(dim=0) - W_E[cat_animal_ids].mean(dim=0)).cpu().numpy()
    tool_dir = tool_dir / (np.linalg.norm(tool_dir) + 1e-8)
    
    # 关键层: 中层和最后层
    target_layers = sorted(set(
        [n_layers//2, n_layers-2, n_layers-1]
    ))
    target_layers = [l for l in target_layers if 0 <= l < n_layers]
    print(f"  Target layers: {target_layers}")
    print(f"  Test objects: {len(test_objects)} from {len(multi_objects)} categories")
    
    results = {"layers": {}}
    
    # 对每类对象: 计算shared/private + 注入测试
    for cat_name, cat_dir_vec in [("fruit", fruit_dir), ("tool", tool_dir)]:
        cat_objects = [o for c, o in test_objects if c == cat_name]
        if not cat_objects:
            continue
        
        print(f"\n  Category: {cat_name}, objects: {cat_objects}")
        
        # 分解
        for li in target_layers:
            deltas = {}
            for obj_name in cat_objects:
                text = f"The {obj_name} is a"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                input_ids_obj = inputs["input_ids"].to(input_device)
                attention_mask_obj = inputs["attention_mask"].to(input_device)
                last_pos = input_ids_obj.shape[1] - 1
                
                base_h = {}
                def make_h_b(lidx):
                    def hook(m, inp, out):
                        if isinstance(out, tuple):
                            base_h[lidx] = out[0][0, last_pos].detach().float().cpu().numpy()
                    return hook
                hooks_b = [layers[li].register_forward_hook(make_h_b(li))]
                with torch.no_grad():
                    _ = model(input_ids=input_ids_obj, attention_mask=attention_mask_obj)
                for h in hooks_b:
                    h.remove()
                
                perturb_t = torch.tensor(cat_dir_vec, dtype=torch.float32).to(input_device).to(torch.bfloat16)
                pert_h = {}
                def make_h_p(lidx):
                    def hook(m, inp, out):
                        if isinstance(out, tuple):
                            pert_h[lidx] = out[0][0, last_pos].detach().float().cpu().numpy()
                    return hook
                hooks_p = [layers[li].register_forward_hook(make_h_p(li))]
                
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
                        _ = model(input_ids=input_ids_obj, attention_mask=attention_mask_obj)
                except:
                    pass
                finally:
                    if embed_hook:
                        embed_hook.remove()
                    for h in hooks_p:
                        h.remove()
                
                if li in base_h and li in pert_h:
                    d = pert_h[li] - base_h[li]
                    if np.linalg.norm(d) > 1e-8:
                        deltas[obj_name] = d
            
            if len(deltas) < 2:
                continue
            
            delta_matrix = np.stack(list(deltas.values()))
            shared = delta_matrix.mean(axis=0)
            shared_norm = np.linalg.norm(shared)
            
            # 注入测试: 用第一个对象做测试
            test_obj = cat_objects[0]
            test_text = f"The {test_obj} is a"
            test_inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=64)
            test_input_ids = test_inputs["input_ids"].to(input_device)
            test_attention_mask = test_inputs["attention_mask"].to(input_device)
            test_last_pos = test_input_ids.shape[1] - 1
            
            with torch.no_grad():
                out_base = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                base_logits = out_base.logits[0, -1].float().cpu().numpy()
            base_metrics = compute_logit_metrics(base_logits, tokenizer, extended_slots)
            
            # 注入shared
            inj_hook = layers[li].register_forward_hook(
                make_safe_inject_hook(shared, test_last_pos, beta=1.0))
            
            with torch.no_grad():
                try:
                    out_inj = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                    inj_logits = out_inj.logits[0, -1].float().cpu().numpy()
                except:
                    inj_hook.remove()
                    continue
            inj_hook.remove()
            
            inj_metrics = compute_logit_metrics(inj_logits, tokenizer, extended_slots)
            
            slot_deltas = {}
            for key in list(extended_slots.keys()) + ["entropy", "confidence"]:
                bk = base_metrics.get(f"{key}_logit", base_metrics.get(key))
                ik = inj_metrics.get(f"{key}_logit", inj_metrics.get(key))
                if isinstance(bk, (int, float)) and isinstance(ik, (int, float)):
                    if key in ["entropy", "confidence"]:
                        slot_deltas[f"{key}_delta"] = round(ik - bk, 4)
                    else:
                        slot_deltas[f"{key}_delta"] = round(ik - bk, 4)
            
            layer_key = f"L{li}_{cat_name}"
            results["layers"][layer_key] = {
                "shared_norm": round(float(shared_norm), 4),
                "n_objects": len(deltas),
                "test_object": test_obj,
                "slot_deltas": slot_deltas,
            }
            
            plog(f"  Exp5: L{li} {cat_name} (shared_norm={shared_norm:.2f}): " + 
                  ", ".join(f"{k}={v:+.3f}" for k, v in slot_deltas.items() if "delta" in k))
    
    # 汇总
    print(f"\n  === Exp5 Summary: Multi-Slot Verification ===")
    for lk in sorted(results["layers"].keys()):
        lr = results["layers"][lk]
        sd = lr.get("slot_deltas", {})
        cat_d = sd.get("cat_fruit_delta", sd.get("cat_animal_delta", sd.get("cat_tool_delta", "N/A")))
        col_d = sd.get("color_delta", "N/A")
        part_d = sd.get("part_delta", "N/A")
        mat_d = sd.get("material_delta", "N/A")
        hab_d = sd.get("habitat_delta", "N/A")
        print(f"  {lk:20s} | catΔ={cat_d} | colorΔ={col_d} | partΔ={part_d} | matΔ={mat_d} | habΔ={hab_d}")
    
    return results


# ===== 主函数 =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print(f"\n{'='*60}")
    print(f"Phase 452: Norm-RMSNorm Causal Closure Verification")
    print(f"Model: {model_name}, Round: {round_num}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    t0_load = time.time()
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    print(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}, mlp_type={info.mlp_type}")
    print(f"  Load: {time.time()-t0_load:.1f}s")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB")
    
    all_results = {"model": model_name, "round": round_num,
                   "model_info": {"class": info.model_class, "n_layers": info.n_layers,
                                  "d_model": info.d_model, "mlp_type": info.mlp_type}}
    
    # Exp1: Norm threshold curve (最关键 — 证明norm-triggered suppression)
    t0 = time.time()
    try:
        r1 = exp1_norm_threshold_curve(model, tokenizer, info)
        all_results["exp1"] = r1
        print(f"  Exp1 complete ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1"] = {"error": str(e)}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Exp2: RMSNorm standalone causal test (证明RMSNorm本身能翻转)
    t0 = time.time()
    try:
        r2 = exp2_rmsnorm_standalone_causal(model, tokenizer, info)
        all_results["exp2"] = r2
        print(f"  Exp2 complete ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2"] = {"error": str(e)}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Exp3: Readout interface profile (全面画像)
    t0 = time.time()
    try:
        r3 = exp3_readout_interface_profile(model, tokenizer, info)
        all_results["exp3"] = r3
        print(f"  Exp3 complete ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3"] = {"error": str(e)}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Exp4: Attention dominance verification
    t0 = time.time()
    try:
        r4 = exp4_attention_dominance_verify(model, tokenizer, info)
        all_results["exp4"] = r4
        print(f"  Exp4 complete ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  Exp4 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp4"] = {"error": str(e)}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Exp5: Multi-slot verification
    t0 = time.time()
    try:
        r5 = exp5_multi_slot_verification(model, tokenizer, info)
        all_results["exp5"] = r5
        print(f"  Exp5 complete ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  Exp5 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp5"] = {"error": str(e)}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase452_{model_name}_r{round_num}.json"
    
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj
    
    all_results = convert(all_results)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")
    
    # 释放模型
    release_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n  Phase 452 complete for {model_name}!")


if __name__ == "__main__":
    main()
