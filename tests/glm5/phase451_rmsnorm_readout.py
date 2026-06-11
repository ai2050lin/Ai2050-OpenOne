"""
Phase 451: RMSNorm-MLP读出接口与反转恢复机制验证
=================================================
核心目标:
1. Exp1: RMSNorm方向重排分析 — pre/post-norm的shared/private方向变化
2. Exp2: 最后两层读出接口验证 — remove/negate shared/private测多属性
3. Exp3: GLM4反转路径定位 — pre-RMSNorm vs post-RMSNorm residual对比

用法:
  python tests/glm5/phase451_rmsnorm_readout.py qwen3 1
  python tests/glm5/phase451_rmsnorm_readout.py glm4 1
  python tests/glm5/phase451_rmsnorm_readout.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import os, gc, time, json, logging
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger("p451")

_last_log = [time.time()]
def plog(msg, interval=30):
    now = time.time()
    if now - _last_log[0] >= interval or any(k in msg.lower() for k in ['complete','step','failed','summary','exp']):
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
    """注入hook — 不预设device, 在hook内部动态转移"""
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


def get_layer_norm_modules(layer):
    """获取层的RMSNorm/LayerNorm模块"""
    norms = {}
    # input_layernorm (pre-attention)
    if hasattr(layer, 'input_layernorm'):
        norms['input_ln'] = layer.input_layernorm
    elif hasattr(layer, 'ln1'):
        norms['input_ln'] = layer.ln1
    
    # post_attention_layernorm (pre-MLP)
    if hasattr(layer, 'post_attention_layernorm'):
        norms['post_attn_ln'] = layer.post_attention_layernorm
    elif hasattr(layer, 'ln2'):
        norms['post_attn_ln'] = layer.ln2
    
    return norms


# ===== 实验1: RMSNorm方向重排分析 =====
def exp1_rmsnorm_direction_rearrangement(model, tokenizer, info):
    """
    在每层捕获:
    1. pre-RMSNorm residual (输入layer前)
    2. post-input-RMSNorm residual (attention前)
    3. post-attn-RMSNorm residual (MLP前)
    4. MLP输出
    5. 层输出
    
    测量:
    - cos(pre_norm_shared, post_norm_shared)
    - cos(pre_norm_private, post_norm_private)
    - norm_ratio变化
    - shared方向是否在RMSNorm后反转
    """
    print(f"\n{'='*60}")
    print("Exp1: RMSNorm Direction Rearrangement Analysis")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    W_E = get_embedding_weight(model)
    input_device = get_input_device(model)
    
    cat_words = ["fruit", "food", "produce"]
    opp_words = ["animal", "dog", "cat"]
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu()
    cat_dir = cat_dir / (cat_dir.norm() + 1e-8)
    
    obj_names = ["apple", "orange", "banana", "grape", "lemon"]
    alpha = 1.0
    
    # 采样层 — 聚焦反转区和最后几层
    if n_layers >= 30:
        mid = n_layers * 3 // 5
        target_layers = sorted(set(
            [0, n_layers//4, n_layers//2] +
            list(range(max(0, mid-8), min(n_layers, mid+10))) +
            [n_layers-4, n_layers-3, n_layers-2, n_layers-1]
        ))
    else:
        target_layers = sorted(set(
            [0, n_layers//3, n_layers//2] +
            list(range(max(0, n_layers-5), n_layers))
        ))
    target_layers = [l for l in target_layers if l < n_layers]
    print(f"  Target layers: {target_layers}")
    
    # 对每个对象: 获取基准和扰动下各阶段的hidden states
    all_data = {}  # {obj_name: {li: {phase: vector}}}
    
    for obj_name in obj_names:
        text = f"The {obj_name} is a"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        last_pos = input_ids.shape[1] - 1
        
        obj_data = {}
        
        for condition in ["clean", "perturbed"]:
            captured = {}  # {li: {"pre_input_ln": vec, "post_input_ln": vec, "pre_post_attn_ln": vec, "mlp_out": vec, "layer_out": vec}}
            
            def make_capture_hook(phase_name, li, is_mlp=False, is_tuple_out=True):
                def hook(m, inp, out):
                    try:
                        if is_mlp:
                            # MLP输出是单tensor
                            if isinstance(out, tuple):
                                vec = out[0][0, last_pos].detach().float().cpu().numpy()
                            else:
                                vec = out[0, last_pos].detach().float().cpu().numpy()
                        elif is_tuple_out:
                            if isinstance(out, tuple):
                                vec = out[0][0, last_pos].detach().float().cpu().numpy()
                            else:
                                vec = out[0, last_pos].detach().float().cpu().numpy()
                        else:
                            # RMSNorm: 输入是tensor, 输出也是tensor
                            if isinstance(inp, tuple) and len(inp) > 0:
                                vec = inp[0][0, last_pos].detach().float().cpu().numpy()
                            elif isinstance(inp, torch.Tensor):
                                vec = inp[0, last_pos].detach().float().cpu().numpy()
                            else:
                                return
                        if li not in captured:
                            captured[li] = {}
                        captured[li][phase_name] = vec
                    except Exception as e:
                        pass
                return hook
            
            hooks = []
            for li in target_layers:
                layer = layers[li]
                
                # 1. 层输出 (包含RMSNorm后完整输出)
                hooks.append(layer.register_forward_hook(
                    make_capture_hook("layer_out", li, is_tuple_out=True)))
                
                # 2. MLP输出
                if hasattr(layer, 'mlp'):
                    hooks.append(layer.mlp.register_forward_hook(
                        make_capture_hook("mlp_out", li, is_mlp=True)))
                
                # 3. Attention输出
                if hasattr(layer, 'self_attn'):
                    hooks.append(layer.self_attn.register_forward_hook(
                        make_capture_hook("attn_out", li, is_tuple_out=True)))
                
                # 4. Post-attention RMSNorm输入 (即attention输出残差后)
                # 注意: post_attention_layernorm的输入就是残差流
                norm_modules = get_layer_norm_modules(layer)
                for norm_name, norm_mod in norm_modules.items():
                    hooks.append(norm_mod.register_forward_hook(
                        make_capture_hook(f"post_{norm_name}", li, is_tuple_out=False)))
            
            if condition == "perturbed":
                perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
                embed_hook = None
                def on_embed(m, inp, out):
                    if isinstance(out, torch.Tensor):
                        out = out.clone()
                        out[0, last_pos] = out[0, last_pos] + perturb_vec.to(out.dtype)
                    return out
                if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                    embed_hook = model.model.embed_tokens.register_forward_hook(on_embed)
            
            with torch.no_grad():
                try:
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
                except Exception as e:
                    print(f"  Warning: {condition} fwd failed for {obj_name}: {e}")
            
            for h in hooks:
                h.remove()
            if condition == "perturbed" and embed_hook:
                embed_hook.remove()
            
            obj_data[condition] = {li: captured.get(li, {}) for li in target_layers}
        
        all_data[obj_name] = obj_data
        plog(f"  Exp1: {obj_name} captured ({len(obj_data.get('clean', {}))} layers)")
    
    # 计算deltas和shared/private分解
    print("  Computing deltas and shared/private decomposition...")
    
    results = {}
    for li in target_layers:
        # 收集各对象的delta
        deltas = {}
        for o in obj_names:
            clean_vecs = all_data.get(o, {}).get("clean", {}).get(li, {})
            pert_vecs = all_data.get(o, {}).get("perturbed", {}).get(li, {})
            
            if "layer_out" in clean_vecs and "layer_out" in pert_vecs:
                delta = pert_vecs["layer_out"] - clean_vecs["layer_out"]
                if np.linalg.norm(delta) > 1e-8:
                    deltas[o] = delta
        
        if len(deltas) < 3:
            continue
        
        delta_matrix = np.stack(list(deltas.values()))
        shared = delta_matrix.mean(axis=0)
        private = {o: d - shared for o, d in deltas.items()}
        shared_norm = np.linalg.norm(shared)
        
        layer_result = {
            "shared_norm": round(float(shared_norm), 4),
            "n_objects": len(deltas),
        }
        
        # 对各阶段也做shared/private分解
        for phase in ["mlp_out", "attn_out"]:
            phase_deltas = {}
            for o in obj_names:
                clean_vecs = all_data.get(o, {}).get("clean", {}).get(li, {})
                pert_vecs = all_data.get(o, {}).get("perturbed", {}).get(li, {})
                if phase in clean_vecs and phase in pert_vecs:
                    delta = pert_vecs[phase] - clean_vecs[phase]
                    if np.linalg.norm(delta) > 1e-8:
                        phase_deltas[o] = delta
            
            if len(phase_deltas) >= 3:
                phase_matrix = np.stack(list(phase_deltas.values()))
                phase_shared = phase_matrix.mean(axis=0)
                phase_shared_norm = np.linalg.norm(phase_shared)
                # cos(layer_shared, phase_shared)
                cos_val = 0.0
                if phase_shared_norm > 1e-8 and shared_norm > 1e-8:
                    cos_val = float(np.dot(shared, phase_shared) / (shared_norm * phase_shared_norm))
                layer_result[f"{phase}_shared_norm"] = round(float(phase_shared_norm), 4)
                layer_result[f"cos_shared_{phase}_shared"] = round(cos_val, 4)
        
        # 检查RMSNorm前后的方向变化
        # post_input_ln: 输入RMSNorm的输出 (即normalized residual)
        for phase in ["post_input_ln", "post_post_attn_ln"]:
            phase_deltas = {}
            for o in obj_names:
                clean_vecs = all_data.get(o, {}).get("clean", {}).get(li, {})
                pert_vecs = all_data.get(o, {}).get("perturbed", {}).get(li, {})
                if phase in clean_vecs and phase in pert_vecs:
                    delta = pert_vecs[phase] - clean_vecs[phase]
                    if np.linalg.norm(delta) > 1e-8:
                        phase_deltas[o] = delta
            
            if len(phase_deltas) >= 3:
                phase_matrix = np.stack(list(phase_deltas.values()))
                phase_shared = phase_matrix.mean(axis=0)
                phase_shared_norm = np.linalg.norm(phase_shared)
                cos_val = 0.0
                if phase_shared_norm > 1e-8 and shared_norm > 1e-8:
                    cos_val = float(np.dot(shared, phase_shared) / (shared_norm * phase_shared_norm))
                layer_result[f"{phase}_shared_norm"] = round(float(phase_shared_norm), 4)
                layer_result[f"cos_shared_{phase}_shared"] = round(cos_val, 4)
        
        results[f"L{li}"] = layer_result
        print(f"  L{li}: shared_norm={shared_norm:.3f}, " +
              ", ".join(f"{k}={v}" for k, v in layer_result.items() 
                       if k.startswith("cos_")))
    
    print(f"\n  === Exp1 Summary: RMSNorm Direction Rearrangement ===")
    print(f"  Layer | shared_norm | cos(layer,mlp) | cos(layer,attn) | cos(layer,post_input_ln) | cos(layer,post_post_attn_ln)")
    for lk in sorted(results.keys()):
        lr = results[lk]
        sn = lr.get("shared_norm", 0)
        cm = lr.get("cos_shared_mlp_out_shared", "N/A")
        ca = lr.get("cos_shared_attn_out_shared", "N/A")
        ci = lr.get("cos_shared_post_input_ln_shared", "N/A")
        cp = lr.get("cos_shared_post_post_attn_ln_shared", "N/A")
        print(f"  {lk:5s} | {sn:.4f}    | {cm}        | {ca}          | {ci}                   | {cp}")
    
    return results


# ===== 实验2: 最后两层读出接口验证 =====
def exp2_readout_interface(model, tokenizer, info):
    """
    在最后几层做:
    - remove shared (移除共享分量)
    - remove private (移除私有分量)
    - negate shared (反转共享分量)
    - negate private (反转私有分量)
    - direction-only injection (只改方向不改范数)
    - scale-only injection (只改范数不改方向)
    
    测: category, color, part, material, entropy, confidence
    """
    print(f"\n{'='*60}")
    print("Exp2: Readout Interface Verification (Last 2-3 Layers)")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    W_E = get_embedding_weight(model)
    W_U = get_W_U(model, info.name).T  # [d_model, vocab]
    input_device = get_input_device(model)
    
    cat_words = ["fruit", "food", "produce"]
    opp_words = ["animal", "dog", "cat"]
    color_words = ["red", "green", "yellow"]
    part_words = ["seed", "skin", "core", "stem"]
    material_words = ["organic", "natural", "fresh"]
    
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu()
    cat_dir = cat_dir / (cat_dir.norm() + 1e-8)
    
    obj_names = ["apple", "orange", "banana", "grape", "lemon"]
    alpha = 1.0
    
    # 关键层: 最后3层 + 反转区(如果有的话)
    if n_layers >= 30:
        # 反转区层 (对GLM4是L24附近)
        mid = n_layers * 3 // 5
        target_layers = sorted(set(
            [n_layers-3, n_layers-2, n_layers-1] +
            list(range(max(0, mid-2), min(n_layers, mid+3)))
        ))
    else:
        target_layers = [n_layers-3, n_layers-2, n_layers-1]
    target_layers = [l for l in target_layers if 0 <= l < n_layers]
    print(f"  Target layers: {target_layers}")
    
    # Step 1: 收集各层delta (shared/private分解)
    print("  Step 1: Collecting deltas and decomposition...")
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
            
            perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
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
                    out[0, last_pos] = out[0, last_pos] + perturb_vec.to(out.dtype)
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
    
    print(f"  Decomposition done: {len(decomposition)} layers ({time.time()-t1:.1f}s)")
    
    # Step 2: 基准logits
    test_text = "The apple is a"
    test_inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=64)
    test_input_ids = test_inputs["input_ids"].to(input_device)
    test_attention_mask = test_inputs["attention_mask"].to(input_device)
    test_last_pos = test_input_ids.shape[1] - 1
    
    with torch.no_grad():
        out_base = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
        base_logits = out_base.logits[0, -1].float().cpu().numpy()
    
    base_cat = get_logit_for_words(base_logits, tokenizer, cat_words)
    base_color = get_logit_for_words(base_logits, tokenizer, color_words)
    base_part = get_logit_for_words(base_logits, tokenizer, part_words)
    base_material = get_logit_for_words(base_logits, tokenizer, material_words)
    
    # entropy & confidence
    probs = np.exp(base_logits - base_logits.max())
    probs = probs / probs.sum()
    base_entropy = -np.sum(probs * np.log(probs + 1e-12))
    base_confidence = float(probs.max())
    
    print(f"  Base: cat={base_cat:.2f}, color={base_color:.2f}, part={base_part:.2f}, "
          f"material={base_material:.2f}, entropy={base_entropy:.2f}, conf={base_confidence:.4f}")
    
    # Step 3: 各层各操作
    print("  Step 2: Running readout interface tests...")
    t2 = time.time()
    
    results = {"base": {"cat": base_cat, "color": base_color, "part": base_part,
                         "material": base_material, "entropy": base_entropy, "confidence": base_confidence},
               "layers": {}}
    
    for li in target_layers:
        if li not in decomposition:
            continue
        
        shared_vec = decomposition[li]["shared"]
        private_vec = decomposition[li]["private"].get("apple", np.zeros(d_model))
        shared_norm = np.linalg.norm(shared_vec)
        private_norm = np.linalg.norm(private_vec)
        shared_unit = shared_vec / (shared_norm + 1e-8)
        private_unit = private_vec / (private_norm + 1e-8)
        
        layer_result = {"shared_norm": round(float(shared_norm), 4),
                       "private_norm": round(float(private_norm), 4)}
        
        # 定义操作: (name, vector_to_inject)
        operations = {
            # 标准注入
            "inject_shared": shared_vec * 2.0,
            "inject_private": private_vec * 2.0,
            # Direction-only: 固定范数为1, 方向为shared/private
            "dir_only_shared": shared_unit * 1.0,  # 范数=1
            "dir_only_private": private_unit * 1.0,  # 范数=1
            # Scale-only: 固定方向为random, 范数=shared_norm
            "scale_matched": np.random.randn(d_model) * shared_norm / (np.linalg.norm(np.random.randn(d_model)) + 1e-8),
            # Negate
            "negate_shared": -shared_vec * 2.0,
            "negate_private": -private_vec * 2.0,
        }
        
        for op_name, inj_vec in operations.items():
            inj_hook = layers[li].register_forward_hook(
                make_safe_inject_hook(inj_vec, test_last_pos, beta=1.0))
            
            with torch.no_grad():
                try:
                    out_inj = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                    inj_logits = out_inj.logits[0, -1].float().cpu().numpy()
                except Exception as e:
                    print(f"  L{li} {op_name}: FAILED - {e}")
                    layer_result[op_name] = {"error": str(e)}
                    inj_hook.remove()
                    continue
            
            inj_hook.remove()
            
            inj_cat = get_logit_for_words(inj_logits, tokenizer, cat_words)
            inj_color = get_logit_for_words(inj_logits, tokenizer, color_words)
            inj_part = get_logit_for_words(inj_logits, tokenizer, part_words)
            inj_material = get_logit_for_words(inj_logits, tokenizer, material_words)
            
            probs_inj = np.exp(inj_logits - inj_logits.max())
            probs_inj = probs_inj / probs_inj.sum()
            inj_entropy = -np.sum(probs_inj * np.log(probs_inj + 1e-12))
            inj_confidence = float(probs_inj.max())
            
            op_result = {}
            for attr, (inj_val, base_val) in [
                ("cat", (inj_cat, base_cat)),
                ("color", (inj_color, base_color)),
                ("part", (inj_part, base_part)),
                ("material", (inj_material, base_material)),
            ]:
                if inj_val is not None and base_val is not None:
                    op_result[f"{attr}_delta"] = round(inj_val - base_val, 4)
            op_result["entropy_delta"] = round(inj_entropy - base_entropy, 4)
            op_result["confidence_delta"] = round(inj_confidence - base_confidence, 4)
            
            layer_result[op_name] = op_result
            
            cat_d = op_result.get("cat_delta", 0)
            plog(f"  L{li} {op_name}: catΔ={cat_d:+.3f}")
        
        results["layers"][f"L{li}"] = layer_result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总
    print(f"\n  === Exp2 Summary: Readout Interface ===")
    print(f"  Layer | inj_shared→cat | inj_priv→cat | dir_shared→cat | dir_priv→cat | neg_shared→cat | scale→cat | inj_shared→color")
    for lk in sorted(results["layers"].keys()):
        lr = results["layers"][lk]
        is_c = lr.get("inject_shared", {}).get("cat_delta", "N/A")
        ip_c = lr.get("inject_private", {}).get("cat_delta", "N/A")
        ds_c = lr.get("dir_only_shared", {}).get("cat_delta", "N/A")
        dp_c = lr.get("dir_only_private", {}).get("cat_delta", "N/A")
        ns_c = lr.get("negate_shared", {}).get("cat_delta", "N/A")
        sc_c = lr.get("scale_matched", {}).get("cat_delta", "N/A")
        is_col = lr.get("inject_shared", {}).get("color_delta", "N/A")
        print(f"  {lk:5s} | {is_c}          | {ip_c}         | {ds_c}           | {dp_c}          | {ns_c}           | {sc_c}      | {is_col}")
    
    return results


# ===== 实验3: GLM4反转路径定位 — pre/post-RMSNorm residual对比 =====
def exp3_rmsnorm_inversion_path(model, tokenizer, info):
    """
    在GLM4反转区(L20-L30)做精细RMSNorm分析:
    1. 捕获pre-input-LN和post-input-LN的residual
    2. 捕获pre-post-attn-LN和post-post-attn-LN的residual
    3. 测shared方向在各阶段的读出投影
    
    目的: 确定反转是发生在RMSNorm内部还是组件输出中
    """
    print(f"\n{'='*60}")
    print("Exp3: RMSNorm Inversion Path Localization")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    W_E = get_embedding_weight(model)
    W_U_np = get_W_U(model, info.name).T  # [d_model, vocab]
    input_device = get_input_device(model)
    
    cat_words = ["fruit", "food", "produce"]
    opp_words = ["animal", "dog", "cat"]
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu().numpy()
    cat_dir = cat_dir / (np.linalg.norm(cat_dir) + 1e-8)
    
    # 类别读出方向
    cat_readout_dir = W_U_np[:, cat_ids].mean(axis=1)
    cat_readout_dir = cat_readout_dir / (np.linalg.norm(cat_readout_dir) + 1e-8)
    
    obj_names = ["apple", "orange", "banana"]
    alpha = 1.0
    
    # 关键层: 反转区附近
    if n_layers >= 30:
        mid = n_layers * 3 // 5
        scan_layers = sorted(set(
            [0, n_layers//4] +
            list(range(max(0, mid-8), min(n_layers, mid+10))) +
            [n_layers-2, n_layers-1]
        ))
    else:
        scan_layers = list(range(n_layers))
    scan_layers = [l for l in scan_layers if l < n_layers]
    print(f"  Scan layers: {scan_layers}")
    
    # 对每个对象, 收集各阶段hidden states
    all_stage_data = {}
    
    for obj_name in obj_names:
        text = f"The {obj_name} is a"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        last_pos = input_ids.shape[1] - 1
        
        for condition in ["clean", "perturbed"]:
            stage_captures = {}  # {li: {"pre_attn_resid": vec, "attn_out": vec, "pre_mlp_resid": vec, "mlp_out": vec, "post_mlp_resid": vec}}
            
            hooks = []
            
            for li in scan_layers:
                layer = layers[li]
                
                # 捕获MLP输出
                if hasattr(layer, 'mlp'):
                    def make_mlp_hook(lidx):
                        def hook(m, inp, out):
                            if isinstance(out, tuple):
                                vec = out[0][0, last_pos].detach().float().cpu().numpy()
                            else:
                                vec = out[0, last_pos].detach().float().cpu().numpy()
                            if lidx not in stage_captures:
                                stage_captures[lidx] = {}
                            stage_captures[lidx]["mlp_out"] = vec
                        return hook
                    hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(li)))
                
                # 捕获Attention输出
                if hasattr(layer, 'self_attn'):
                    def make_attn_hook(lidx):
                        def hook(m, inp, out):
                            if isinstance(out, tuple):
                                vec = out[0][0, last_pos].detach().float().cpu().numpy()
                            else:
                                vec = out[0, last_pos].detach().float().cpu().numpy()
                            if lidx not in stage_captures:
                                stage_captures[lidx] = {}
                            stage_captures[lidx]["attn_out"] = vec
                        return hook
                    hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(li)))
                
                # 捕获post-attention-RMSNorm (MLP前的RMSNorm)的输入和输出
                norm_modules = get_layer_norm_modules(layer)
                for norm_name, norm_mod in norm_modules.items():
                    def make_norm_hook(lidx, nname, capture_type):
                        """capture_type: 'input' or 'output'"""
                        if capture_type == 'input':
                            def hook(m, inp, out):
                                try:
                                    if isinstance(inp, tuple) and len(inp) > 0:
                                        vec = inp[0][0, last_pos].detach().float().cpu().numpy()
                                    elif isinstance(inp, torch.Tensor):
                                        vec = inp[0, last_pos].detach().float().cpu().numpy()
                                    else:
                                        return
                                    if lidx not in stage_captures:
                                        stage_captures[lidx] = {}
                                    stage_captures[lidx][f"{nname}_input"] = vec
                                except:
                                    pass
                            return hook
                        else:  # output
                            def hook(m, inp, out):
                                try:
                                    if isinstance(out, torch.Tensor):
                                        vec = out[0, last_pos].detach().float().cpu().numpy()
                                    elif isinstance(out, tuple):
                                        vec = out[0][0, last_pos].detach().float().cpu().numpy()
                                    else:
                                        return
                                    if lidx not in stage_captures:
                                        stage_captures[lidx] = {}
                                    stage_captures[lidx][f"{nname}_output"] = vec
                                except:
                                    pass
                            return hook
                    hooks.append(norm_mod.register_forward_hook(make_norm_hook(li, norm_name, 'input')))
                    hooks.append(norm_mod.register_forward_hook(make_norm_hook(li, norm_name, 'output')))
                
                # 捕获层输出
                def make_layer_hook(lidx):
                    def hook(m, inp, out):
                        if isinstance(out, tuple):
                            vec = out[0][0, last_pos].detach().float().cpu().numpy()
                        else:
                            vec = out[0, last_pos].detach().float().cpu().numpy()
                        if lidx not in stage_captures:
                            stage_captures[lidx] = {}
                        stage_captures[lidx]["layer_out"] = vec
                    return hook
                hooks.append(layer.register_forward_hook(make_layer_hook(li)))
            
            # 扰动
            embed_hook = None
            if condition == "perturbed":
                perturb_vec = (alpha * cat_dir)
                perturb_t = torch.tensor(perturb_vec, dtype=torch.float32).to(input_device).to(torch.bfloat16)
                def on_embed(m, inp, out):
                    if isinstance(out, torch.Tensor):
                        out = out.clone()
                        out[0, last_pos] = out[0, last_pos] + perturb_t.to(out.dtype)
                    return out
                if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                    embed_hook = model.model.embed_tokens.register_forward_hook(on_embed)
            
            with torch.no_grad():
                try:
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
                except Exception as e:
                    print(f"  Warning: {condition} fwd failed for {obj_name}: {e}")
            
            for h in hooks:
                h.remove()
            if embed_hook:
                embed_hook.remove()
            
            key = f"{obj_name}_{condition}"
            all_stage_data[key] = {li: stage_captures.get(li, {}) for li in scan_layers}
        
        plog(f"  Exp3: {obj_name} done")
    
    # 计算各阶段shared delta的方向和读出投影
    print("  Computing stage-wise shared delta analysis...")
    
    results = {}
    for li in scan_layers:
        # 收集各对象的各阶段delta
        stage_deltas = {}  # {stage: {obj: delta}}
        
        for stage in ["layer_out", "mlp_out", "attn_out", "input_ln_input", "input_ln_output",
                      "post_attn_ln_input", "post_attn_ln_output"]:
            deltas = {}
            for obj_name in obj_names:
                clean = all_stage_data.get(f"{obj_name}_clean", {}).get(li, {})
                pert = all_stage_data.get(f"{obj_name}_perturbed", {}).get(li, {})
                if stage in clean and stage in pert:
                    d = pert[stage] - clean[stage]
                    if np.linalg.norm(d) > 1e-8:
                        deltas[obj_name] = d
            if len(deltas) >= 2:
                stage_deltas[stage] = deltas
        
        if not stage_deltas:
            continue
        
        layer_result = {}
        
        for stage, deltas in stage_deltas.items():
            delta_matrix = np.stack(list(deltas.values()))
            shared = delta_matrix.mean(axis=0)
            shared_norm = np.linalg.norm(shared)
            
            # 投影到cat_dir和cat_readout_dir
            proj_cat_embed = float(np.dot(shared, cat_dir)) if shared_norm > 1e-8 else 0.0
            proj_cat_readout = float(np.dot(shared, cat_readout_dir)) if shared_norm > 1e-8 else 0.0
            
            layer_result[stage] = {
                "shared_norm": round(float(shared_norm), 4),
                "proj_cat_embed_dir": round(proj_cat_embed, 4),
                "proj_cat_readout_dir": round(proj_cat_readout, 4),
            }
        
        # 检查RMSNorm是否导致方向反转
        # 比较input_ln_input vs input_ln_output的shared方向
        if "input_ln_input" in layer_result and "input_ln_output" in layer_result:
            in_sign = layer_result["input_ln_input"].get("proj_cat_readout_dir", 0)
            out_sign = layer_result["input_ln_output"].get("proj_cat_readout_dir", 0)
            layer_result["input_ln_sign_flip"] = "YES" if in_sign * out_sign < 0 else "NO"
        
        if "post_attn_ln_input" in layer_result and "post_attn_ln_output" in layer_result:
            in_sign = layer_result["post_attn_ln_input"].get("proj_cat_readout_dir", 0)
            out_sign = layer_result["post_attn_ln_output"].get("proj_cat_readout_dir", 0)
            layer_result["post_attn_ln_sign_flip"] = "YES" if in_sign * out_sign < 0 else "NO"
        
        # layer_out vs mlp_out vs attn_out的shared读出方向
        for stage in ["layer_out", "mlp_out", "attn_out"]:
            if stage in layer_result:
                r = layer_result[stage]
                sign = "+" if r.get("proj_cat_readout_dir", 0) > 0 else "-"
                layer_result[f"{stage}_readout_sign"] = sign
        
        results[f"L{li}"] = layer_result
        
        # 简洁输出
        lo_proj = layer_result.get("layer_out", {}).get("proj_cat_readout_dir", "N/A")
        mlp_proj = layer_result.get("mlp_out", {}).get("proj_cat_readout_dir", "N/A")
        attn_proj = layer_result.get("attn_out", {}).get("proj_cat_readout_dir", "N/A")
        ln_flip = layer_result.get("input_ln_sign_flip", "N/A")
        post_ln_flip = layer_result.get("post_attn_ln_sign_flip", "N/A")
        print(f"  L{li}: layer={lo_proj:+.3f}, mlp={mlp_proj:+.3f}, attn={attn_proj:+.3f}, "
              f"input_ln_flip={ln_flip}, post_attn_ln_flip={post_ln_flip}")
    
    print(f"\n  === Exp3 Summary: RMSNorm Inversion Path ===")
    print(f"  Layer | layer_out→cat | mlp_out→cat | attn_out→cat | input_ln_flip | post_attn_ln_flip")
    for lk in sorted(results.keys()):
        lr = results[lk]
        lo = lr.get("layer_out", {}).get("proj_cat_readout_dir", "N/A")
        ml = lr.get("mlp_out", {}).get("proj_cat_readout_dir", "N/A")
        at = lr.get("attn_out", {}).get("proj_cat_readout_dir", "N/A")
        ilf = lr.get("input_ln_sign_flip", "N/A")
        plf = lr.get("post_attn_ln_sign_flip", "N/A")
        print(f"  {lk:5s} | {lo:+.4f}      | {ml:+.4f}     | {at:+.4f}      | {ilf}             | {plf}")
    
    return results


# ===== 主函数 =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print(f"\n{'='*60}")
    print(f"Phase 451: RMSNorm-MLP Readout Interface & Inversion Recovery")
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
    
    # Exp1
    t0 = time.time()
    try:
        r1 = exp1_rmsnorm_direction_rearrangement(model, tokenizer, info)
        all_results["exp1"] = r1
        print(f"  Exp1 complete ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1"] = {"error": str(e)}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Exp2
    t0 = time.time()
    try:
        r2 = exp2_readout_interface(model, tokenizer, info)
        all_results["exp2"] = r2
        print(f"  Exp2 complete ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2"] = {"error": str(e)}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Exp3
    t0 = time.time()
    try:
        r3 = exp3_rmsnorm_inversion_path(model, tokenizer, info)
        all_results["exp3"] = r3
        print(f"  Exp3 complete ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3"] = {"error": str(e)}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase451_{model_name}_r{round_num}.json"
    
    # 转换numpy类型
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
    
    print(f"\nPhase 451 complete! Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
