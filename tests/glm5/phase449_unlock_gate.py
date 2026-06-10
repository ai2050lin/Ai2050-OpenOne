"""
Phase 449: Object Unlock Gate + MLP Internal Binding + Shared/Private Causal Verification
==========================================================================================
核心假说验证:
1. GLM4的负先验是否是真正的对象解锁机制(排除对象重复/位置假象)
2. MLP内部gate/up/down分别在绑定更新中起什么作用
3. shared/private分量注入是否产生可预测的因果效果

实验1: 对象解锁门控验证 — 6类模板精细控制
实验2: MLP内部组件消融 — gate vs up vs down分别消融
实验3: Shared/Private因果注入 — 分量注入测因果效果

用法:
  python tests/glm5/phase449_unlock_gate.py qwen3 1
  python tests/glm5/phase449_unlock_gate.py glm4 1
  python tests/glm5/phase449_unlock_gate.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import os, gc, time, json
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS)

# ===== 配置 =====
TEST_OBJECTS = {
    "apple":  {"category": "fruit",  "opp_category": "animal", "slot_words": {
        "category": ["fruit", "food", "produce"],
        "color": ["red", "green", "yellow"],
        "taste": ["sweet", "sour", "juicy"],
        "part": ["seed", "skin", "core"],
        "material": ["organic", "fresh", "natural"],
    }},
    "dog":    {"category": "animal", "opp_category": "fruit",  "slot_words": {
        "category": ["animal", "pet", "mammal"],
        "color": ["brown", "black", "white"],
        "part": ["leg", "tail", "fur"],
    }},
    "knife":  {"category": "tool",   "opp_category": "fruit",  "slot_words": {
        "category": ["tool", "weapon", "instrument"],
        "color": ["silver", "gray", "metallic"],
        "part": ["blade", "handle", "edge"],
        "material": ["metal", "steel", "iron"],
    }},
    "orange": {"category": "fruit",  "opp_category": "animal", "slot_words": {
        "category": ["fruit", "food", "citrus"],
        "color": ["orange", "yellow"],
        "taste": ["sweet", "sour", "juicy"],
    }},
    "hammer": {"category": "tool",   "opp_category": "animal", "slot_words": {
        "category": ["tool", "instrument", "equipment"],
        "color": ["brown", "gray"],
        "part": ["head", "handle", "nail"],
        "material": ["metal", "wood", "steel"],
    }},
    "cat":    {"category": "animal", "opp_category": "tool",   "slot_words": {
        "category": ["animal", "pet", "mammal"],
        "color": ["black", "white", "orange"],
        "part": ["leg", "tail", "fur"],
    }},
}


def load_model_auto(model_name):
    """BF16 + device_map='auto' + sdpa"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="sdpa")
    model.eval()
    return model, tokenizer


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


def forward_with_hooks(model, tokenizer, text, hooks_dict, max_len=64):
    """带hooks的前向传播"""
    input_device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    
    # 注册hooks
    registered = []
    for target, hook_fn in hooks_dict.items():
        h = target.register_forward_hook(hook_fn)
        registered.append(h)
    
    with torch.no_grad():
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[0, -1].float().cpu().numpy()
        except Exception as e:
            print(f"  [forward_with_hooks] Error: {e}")
            logits = None
    
    for h in registered:
        h.remove()
    
    return logits


# ===== 实验1: 对象解锁门控验证 =====
def exp1_unlock_gate(model, tokenizer, info):
    """
    6类模板精细控制,区分:
    - UnlockScore = T1 - T0 (对象解锁)
    - ConflictRecovery = T3 - T1 (冲突恢复)
    - ObjectRepeatCtrl = T2 - T1 (对象重复控制)
    - PositionCtrl = T4 - T3 (位置控制)
    - ObjectReplaceCtrl = T5 - T3 (对象替换控制)
    """
    print(f"\n{'='*60}")
    print("Exp1: Object Unlock Gate Verification")
    print(f"{'='*60}")
    
    results = {}
    
    for obj_name, obj_cfg in TEST_OBJECTS.items():
        cat = obj_cfg["category"]
        opp_cat = obj_cfg["opp_category"]
        
        # 6类模板 (使用is_a槽位,最稳定)
        templates = {
            "T0_no_obj": f"A thing is a kind of",
            "T1_with_obj": f"The {obj_name} is a kind of",
            "T2_obj_repeat": f"The {obj_name}, mentioned once, the {obj_name} is a kind of",
            "T3_conflict": f"Although the {obj_name} is described as a {opp_cat}, it is a kind of",
            "T4_conflict_obj_near": f"Although something is described as a {opp_cat}, the {obj_name} is a kind of",
            "T5_obj_replace": f"Although the {obj_name} is described as a {opp_cat}, something is a kind of",
        }
        
        obj_results = {}
        
        for t_name, t_text in templates.items():
            logits = forward_with_hooks(model, tokenizer, t_text, {})
            if logits is None:
                continue
            
            slot_scores = {}
            for slot_name, words in obj_cfg["slot_words"].items():
                val = get_logit_for_words(logits, tokenizer, words)
                if val is not None:
                    slot_scores[slot_name] = round(val, 4)
            
            obj_results[t_name] = slot_scores
        
        # 计算解锁/恢复/控制分数
        metrics = {}
        for slot_name in obj_cfg["slot_words"]:
            vals = {}
            for t_name in templates:
                if t_name in obj_results and slot_name in obj_results[t_name]:
                    vals[t_name] = obj_results[t_name][slot_name]
            
            if len(vals) >= 4:
                m = {
                    "unlock_score": round(vals.get("T1_with_obj", 0) - vals.get("T0_no_obj", 0), 4),
                    "conflict_recovery": round(vals.get("T3_conflict", 0) - vals.get("T1_with_obj", 0), 4),
                    "obj_repeat_ctrl": round(vals.get("T2_obj_repeat", 0) - vals.get("T1_with_obj", 0), 4),
                    "position_ctrl": round(vals.get("T4_conflict_obj_near", 0) - vals.get("T3_conflict", 0), 4),
                    "obj_replace_ctrl": round(vals.get("T5_obj_replace", 0) - vals.get("T3_conflict", 0), 4),
                }
                metrics[slot_name] = m
        
        results[obj_name] = {"templates": obj_results, "metrics": metrics}
        
        # 日志
        print(f"  {obj_name} ({cat}):")
        for slot_name, m in metrics.items():
            print(f"    {slot_name}: unlock={m['unlock_score']:+.2f}, "
                  f"conflict_recov={m['conflict_recovery']:+.2f}, "
                  f"repeat_ctrl={m['obj_repeat_ctrl']:+.2f}, "
                  f"pos_ctrl={m['position_ctrl']:+.2f}, "
                  f"replace_ctrl={m['obj_replace_ctrl']:+.2f}")
    
    # 汇总
    print(f"\n  === Exp1 Summary ===")
    for obj_name in TEST_OBJECTS:
        if obj_name not in results:
            continue
        metrics = results[obj_name]["metrics"]
        avg_unlock = np.mean([m["unlock_score"] for m in metrics.values()])
        avg_conflict = np.mean([m["conflict_recovery"] for m in metrics.values()])
        avg_repeat = np.mean([m["obj_repeat_ctrl"] for m in metrics.values()])
        avg_pos = np.mean([m["position_ctrl"] for m in metrics.values()])
        print(f"  {obj_name}: avg_unlock={avg_unlock:+.2f}, avg_conflict={avg_conflict:+.2f}, "
              f"avg_repeat={avg_repeat:+.2f}, avg_pos={avg_pos:+.2f}")
    
    return results


# ===== 实验2: MLP内部组件消融 =====
def exp2_mlp_internal_ablation(model, tokenizer, info):
    """
    在关键层分别消融gate/up/down,测对shared_ratio和属性读出的影响
    """
    print(f"\n{'='*60}")
    print("Exp2: MLP Internal Component Ablation")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    mlp_type = info.mlp_type
    
    # 测3个对象, fruit类别
    obj_names = ["apple", "orange", "banana"]
    cat_words = ["fruit", "food", "produce"]
    opp_words = ["animal", "dog", "cat"]
    
    # 类别方向
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        W_E = model.model.embed_tokens.weight.detach().float()
    else:
        W_E = model.get_input_embeddings().weight.detach().float()
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu()
    cat_dir = cat_dir / (cat_dir.norm() + 1e-8)
    
    alpha = 1.0
    
    # 采样层
    sample_layers = [0, 1, 2, n_layers//4, n_layers//3, n_layers//2, 2*n_layers//3, 3*n_layers//4, n_layers-2, n_layers-1]
    sample_layers = sorted(set([l for l in sample_layers if l < n_layers]))
    
    def compute_shared_ratio(obj_deltas):
        """从对象delta字典计算shared_ratio"""
        valid = {o: d for o, d in obj_deltas.items() if np.linalg.norm(d) > 1e-8}
        if len(valid) < 2:
            return None
        delta_matrix = np.stack(list(valid.values()))
        shared = delta_matrix.mean(axis=0)
        total_var = np.sum(delta_matrix ** 2)
        shared_var = np.sum(shared ** 2) * len(valid)
        return float(shared_var / total_var) if total_var > 1e-10 else 0
    
    def get_last_layer_deltas(model, tokenizer, obj_names, cat_dir, alpha, abl_layer=None, abl_component=None):
        """获取最后一层的对象delta(可选消融某层某组件)"""
        input_device = next(model.parameters()).device
        obj_deltas = {}
        
        for obj_name in obj_names:
            text = f"The {obj_name} is a"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            # 基准
            base_last = {}
            def make_h_base():
                def hook(m, inp, out):
                    if isinstance(out, tuple):
                        base_last['v'] = out[0][0, last_pos].detach().float().cpu()
                return hook
            
            hooks_base = [layers[li].register_forward_hook(make_h_base()) for li in [n_layers-1]]
            
            # 消融hook
            abl_hook = None
            if abl_layer is not None and abl_component is not None:
                layer = layers[abl_layer]
                mlp = layer.mlp if hasattr(layer, 'mlp') else None
                if mlp is not None:
                    target_module = None
                    if abl_component == "gate":
                        if hasattr(mlp, 'gate_proj'):
                            target_module = mlp.gate_proj
                        elif hasattr(mlp, 'gate_up_proj'):
                            # GLM4合并的gate_up, 消融gate部分
                            def gate_abl_hook(m, inp, out):
                                if isinstance(out, tuple):
                                    half = out[0].shape[-1] // 2
                                    new_out = out[0].clone()
                                    new_out[..., :half] = 0
                                    return (new_out,) + out[1:]
                                return out
                            abl_hook = target_module.register_forward_hook(gate_abl_hook) if target_module else None
                            if abl_hook is None:
                                # 直接在gate_up_proj上做
                                target_module = mlp.gate_up_proj
                                def gate_abl_hook2(m, inp, out):
                                    if isinstance(out, tuple):
                                        half = out[0].shape[-1] // 2
                                        new_out = out[0].clone()
                                        new_out[..., :half] = 0
                                        return (new_out,) + out[1:]
                                    return out
                                abl_hook = target_module.register_forward_hook(gate_abl_hook2)
                    elif abl_component == "up":
                        if hasattr(mlp, 'up_proj'):
                            target_module = mlp.up_proj
                        elif hasattr(mlp, 'gate_up_proj'):
                            target_module = mlp.gate_up_proj
                            def up_abl_hook(m, inp, out):
                                if isinstance(out, tuple):
                                    half = out[0].shape[-1] // 2
                                    new_out = out[0].clone()
                                    new_out[..., half:] = 0
                                    return (new_out,) + out[1:]
                                return out
                            abl_hook = target_module.register_forward_hook(up_abl_hook)
                    elif abl_component == "down":
                        if hasattr(mlp, 'down_proj'):
                            target_module = mlp.down_proj
                    
                    if abl_hook is None and target_module is not None:
                        def zero_hook(m, inp, out):
                            if isinstance(out, tuple):
                                return (torch.zeros_like(out[0]),) + out[1:]
                            return torch.zeros_like(out)
                        abl_hook = target_module.register_forward_hook(zero_hook)
            
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            base_val = base_last.get('v')
            if abl_hook:
                abl_hook.remove()
            for h in hooks_base:
                h.remove()
            
            # 扰动
            perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
            pert_last = {}
            def make_h_pert():
                def hook(m, inp, out):
                    if isinstance(out, tuple):
                        pert_last['v'] = out[0][0, last_pos].detach().float().cpu()
                return hook
            
            hooks_pert = [layers[li].register_forward_hook(make_h_pert()) for li in [n_layers-1]]
            
            # 再次注册消融
            abl_hook2 = None
            if abl_layer is not None and abl_component is not None:
                layer = layers[abl_layer]
                mlp = layer.mlp if hasattr(layer, 'mlp') else None
                if mlp is not None:
                    target_module = None
                    if abl_component == "gate":
                        if hasattr(mlp, 'gate_proj'):
                            target_module = mlp.gate_proj
                        elif hasattr(mlp, 'gate_up_proj'):
                            target_module = mlp.gate_up_proj
                            def gate_abl_h(m, inp, out):
                                if isinstance(out, tuple):
                                    half = out[0].shape[-1] // 2
                                    new_out = out[0].clone()
                                    new_out[..., :half] = 0
                                    return (new_out,) + out[1:]
                                return out
                            abl_hook2 = target_module.register_forward_hook(gate_abl_h)
                    elif abl_component == "up":
                        if hasattr(mlp, 'up_proj'):
                            target_module = mlp.up_proj
                        elif hasattr(mlp, 'gate_up_proj'):
                            target_module = mlp.gate_up_proj
                            def up_abl_h(m, inp, out):
                                if isinstance(out, tuple):
                                    half = out[0].shape[-1] // 2
                                    new_out = out[0].clone()
                                    new_out[..., half:] = 0
                                    return (new_out,) + out[1:]
                                return out
                            abl_hook2 = target_module.register_forward_hook(up_abl_h)
                    elif abl_component == "down":
                        if hasattr(mlp, 'down_proj'):
                            target_module = mlp.down_proj
                    
                    if abl_hook2 is None and target_module is not None:
                        def zero_h(m, inp, out):
                            if isinstance(out, tuple):
                                return (torch.zeros_like(out[0]),) + out[1:]
                            return torch.zeros_like(out)
                        abl_hook2 = target_module.register_forward_hook(zero_h)
            
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
                if abl_hook2:
                    abl_hook2.remove()
                for h in hooks_pert:
                    h.remove()
            
            pert_val = pert_last.get('v')
            if base_val is not None and pert_val is not None:
                obj_deltas[obj_name] = (pert_val - base_val).numpy()
        
        return obj_deltas
    
    # 基准: 完整模型的shared_ratio
    print("  Computing baseline shared_ratio...")
    base_deltas = get_last_layer_deltas(model, tokenizer, obj_names, cat_dir, alpha)
    base_sr = compute_shared_ratio(base_deltas)
    print(f"  Baseline shared_ratio (last layer) = {base_sr:.4f}")
    
    # 逐层消融gate/up/down
    results = {"baseline_shared_ratio": base_sr, "ablation_effects": {}}
    
    # 采样少量关键层避免耗时过长
    key_layers = [0, 1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
    key_layers = sorted(set([l for l in key_layers if l < n_layers]))
    
    for abl_layer in key_layers:
        layer_results = {}
        for component in ["gate", "up", "down"]:
            print(f"  Ablating L{abl_layer} {component}...")
            
            try:
                abl_deltas = get_last_layer_deltas(model, tokenizer, obj_names, cat_dir, alpha,
                                                    abl_layer=abl_layer, abl_component=component)
                abl_sr = compute_shared_ratio(abl_deltas)
                delta_sr = abl_sr - base_sr if abl_sr is not None else None
                
                # 同时测属性读出
                for obj_name in obj_names[:1]:  # 只测apple节省时间
                    text = f"The {obj_name} is a"
                    logits = forward_with_hooks(model, tokenizer, text, {})
                    if logits is not None:
                        cat_logit = get_logit_for_words(logits, tokenizer, cat_words)
                        color_logit = get_logit_for_words(logits, tokenizer, ["red", "green", "yellow"])
                    else:
                        cat_logit = None
                        color_logit = None
                
                layer_results[component] = {
                    "shared_ratio": round(abl_sr, 4) if abl_sr is not None else None,
                    "delta_shared_ratio": round(delta_sr, 4) if delta_sr is not None else None,
                    "cat_logit": round(cat_logit, 4) if cat_logit is not None else None,
                    "color_logit": round(color_logit, 4) if color_logit is not None else None,
                }
                
                print(f"    {component}: sr={abl_sr:.4f}, delta_sr={delta_sr:+.4f}, "
                      f"cat_logit={cat_logit:.2f}, color_logit={color_logit:.2f}")
            except Exception as e:
                print(f"    {component}: FAILED - {e}")
                layer_results[component] = {"error": str(e)}
        
        results["ablation_effects"][f"L{abl_layer}"] = layer_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总
    print(f"\n  === Exp2 Summary: MLP Component Effects on Shared Ratio ===")
    print(f"  Baseline SR = {base_sr:.4f}")
    print(f"  Layer | gate_delta | up_delta | down_delta | dominant")
    for layer_key in sorted(results["ablation_effects"].keys()):
        lr = results["ablation_effects"][layer_key]
        g = lr.get("gate", {}).get("delta_shared_ratio", 0) or 0
        u = lr.get("up", {}).get("delta_shared_ratio", 0) or 0
        d = lr.get("down", {}).get("delta_shared_ratio", 0) or 0
        abs_vals = {"gate": abs(g), "up": abs(u), "down": abs(d)}
        dominant = max(abs_vals, key=abs_vals.get) if max(abs_vals.values()) > 0.005 else "neutral"
        print(f"  {layer_key:5s} | {g:+.4f}    | {u:+.4f}   | {d:+.4f}    | {dominant}")
    
    return results


# ===== 实验3: Shared/Private因果注入 =====
def exp3_shared_private_causal(model, tokenizer, info):
    """
    分解delta为shared+private, 分别注入测因果效果
    """
    print(f"\n{'='*60}")
    print("Exp3: Shared/Private Causal Injection")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    
    # 类别方向
    cat_words = ["fruit", "food", "produce"]
    opp_words = ["animal", "dog", "cat"]
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        W_E = model.model.embed_tokens.weight.detach().float()
    else:
        W_E = model.get_input_embeddings().weight.detach().float()
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu()
    cat_dir = cat_dir / (cat_dir.norm() + 1e-8)
    
    # 测试对象(fruit类别)
    obj_names = ["apple", "orange", "banana", "grape", "lemon", "mango"]
    alpha = 1.0
    
    # W_U
    W_U = get_W_U(model, info.name)
    
    # 对象属性词
    OBJ_ATTRS = {
        "apple": {"cat": ["fruit", "food"], "color": ["red", "green"], "taste": ["sweet", "sour"]},
        "orange": {"cat": ["fruit", "citrus"], "color": ["orange"], "taste": ["sweet", "juicy"]},
        "banana": {"cat": ["fruit", "food"], "color": ["yellow"], "taste": ["sweet"]},
        "grape": {"cat": ["fruit", "food"], "color": ["purple", "green"], "taste": ["sweet", "sour"]},
        "lemon": {"cat": ["fruit", "citrus"], "color": ["yellow"], "taste": ["sour", "bitter"]},
        "mango": {"cat": ["fruit", "tropical"], "color": ["yellow", "orange"], "taste": ["sweet"]},
    }
    
    # 在L0注入扰动, 收集各层delta
    print("  Step 1: Collecting per-object deltas at each layer...")
    input_device = next(model.parameters()).device
    
    obj_layer_deltas = {}  # {obj_name: {layer: delta_np}}
    obj_layer_base = {}    # {obj_name: {layer: hidden_np}}
    
    for obj_name in obj_names:
        text = f"The {obj_name} is a"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        last_pos = input_ids.shape[1] - 1
        
        # 基准前向
        base_h = {}
        def make_h_b(li):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    base_h[li] = out[0][0, last_pos].detach().float().cpu()
            return hook
        
        hooks_b = [layers[li].register_forward_hook(make_h_b(li)) for li in range(n_layers)]
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        for h in hooks_b:
            h.remove()
        
        obj_layer_base[obj_name] = {li: base_h[li].numpy() for li in base_h}
        
        # 扰动前向
        perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
        pert_h = {}
        def make_h_p(li):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    pert_h[li] = out[0][0, last_pos].detach().float().cpu()
            return hook
        
        hooks_p = [layers[li].register_forward_hook(make_h_p(li)) for li in range(n_layers)]
        
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
        
        obj_layer_deltas[obj_name] = {li: (pert_h[li] - base_h[li]).numpy() 
                                       for li in range(n_layers) if li in pert_h and li in base_h}
    
    # 分解shared/private在关键层
    target_layers = [0, n_layers//3, n_layers//2, 2*n_layers//3, n_layers-1]
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    print(f"  Step 2: Decomposing shared/private at layers {target_layers}...")
    
    decomposition = {}  # {layer: {shared: np, private: {obj: np}}}
    for li in target_layers:
        deltas = {o: obj_layer_deltas[o].get(li) for o in obj_names if li in obj_layer_deltas.get(o, {})}
        valid = {o: d for o, d in deltas.items() if d is not None and np.linalg.norm(d) > 1e-8}
        if len(valid) < 3:
            continue
        delta_matrix = np.stack(list(valid.values()))
        shared = delta_matrix.mean(axis=0)
        private = {o: d - shared for o, d in valid.items()}
        decomposition[li] = {"shared": shared, "private": private}
    
    # 因果注入: 在不同层注入shared/private,测属性读出
    print(f"  Step 3: Causal injection at each layer...")
    
    injection_results = {}
    inject_beta = 2.0  # 注入强度
    
    for inject_layer in target_layers:
        if inject_layer not in decomposition:
            continue
        
        shared_vec = decomposition[inject_layer]["shared"]
        private_apple = decomposition[inject_layer]["private"].get("apple", np.zeros(d_model))
        
        # 3种注入: shared_only, private_only, shared+private
        injections = {
            "shared_only": shared_vec * inject_beta,
            "private_only": private_apple * inject_beta,
            "shared_plus_private": (shared_vec + private_apple) * inject_beta,
        }
        
        layer_results = {}
        
        for inj_name, inj_vec in injections.items():
            # 在inject_layer处注入
            text = "The apple is a"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            # 基准logits
            with torch.no_grad():
                out_base = model(input_ids=input_ids, attention_mask=attention_mask)
                base_logits = out_base.logits[0, -1].float().cpu().numpy()
            
            # 注入logits
            inj_tensor = torch.tensor(inj_vec, dtype=torch.bfloat16, device=input_device)
            
            def make_inject_hook(vec):
                def hook(m, inp, out):
                    if isinstance(out, tuple):
                        new_out = out[0].clone()
                        new_out[0, last_pos] = new_out[0, last_pos] + vec.to(new_out.dtype)
                        return (new_out,) + out[1:]
                    return out
                return hook
            
            inj_hook = layers[inject_layer].register_forward_hook(make_inject_hook(inj_tensor))
            
            with torch.no_grad():
                try:
                    out_inj = model(input_ids=input_ids, attention_mask=attention_mask)
                    inj_logits = out_inj.logits[0, -1].float().cpu().numpy()
                except:
                    inj_logits = None
            
            inj_hook.remove()
            
            if inj_logits is not None:
                # 测属性读出
                cat_logit_base = get_logit_for_words(base_logits, tokenizer, cat_words)
                cat_logit_inj = get_logit_for_words(inj_logits, tokenizer, cat_words)
                color_logit_base = get_logit_for_words(base_logits, tokenizer, ["red", "green", "yellow"])
                color_logit_inj = get_logit_for_words(inj_logits, tokenizer, ["red", "green", "yellow"])
                taste_logit_base = get_logit_for_words(base_logits, tokenizer, ["sweet", "sour"])
                taste_logit_inj = get_logit_for_words(inj_logits, tokenizer, ["sweet", "sour"])
                
                layer_results[inj_name] = {
                    "cat_delta": round(cat_logit_inj - cat_logit_base, 4) if cat_logit_base is not None else None,
                    "color_delta": round(color_logit_inj - color_logit_base, 4) if color_logit_base is not None else None,
                    "taste_delta": round(taste_logit_inj - taste_logit_base, 4) if taste_logit_base is not None else None,
                }
        
        injection_results[f"L{inject_layer}"] = layer_results
        print(f"  L{inject_layer}: " + 
              " | ".join(f"{k}: cat={v.get('cat_delta',0):+.2f}, color={v.get('color_delta',0):+.2f}, taste={v.get('taste_delta',0):+.2f}" 
                        for k, v in layer_results.items()))
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总
    print(f"\n  === Exp3 Summary: Causal Effects of Shared vs Private ===")
    print(f"  Layer | shared→cat | shared→color | shared→taste | private→cat | private→color | private→taste | S+P→cat | S+P→color | S+P→taste")
    for layer_key in sorted(injection_results.keys()):
        lr = injection_results[layer_key]
        s = lr.get("shared_only", {})
        p = lr.get("private_only", {})
        sp = lr.get("shared_plus_private", {})
        print(f"  {layer_key:5s} | {s.get('cat_delta',0):+.3f}     | {s.get('color_delta',0):+.3f}       | {s.get('taste_delta',0):+.3f}       | "
              f"{p.get('cat_delta',0):+.3f}       | {p.get('color_delta',0):+.3f}         | {p.get('taste_delta',0):+.3f}       | "
              f"{sp.get('cat_delta',0):+.3f}   | {sp.get('color_delta',0):+.3f}     | {sp.get('taste_delta',0):+.3f}")
    
    return {"decomposition_layers": list(decomposition.keys()),
            "injection_results": injection_results}


# ===== 主函数 =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print(f"Phase 449: Object Unlock Gate + MLP Internal Binding + Causal Verification")
    print(f"Model: {model_name}, Round: {round_num}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载模型
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    print(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    all_results = {}
    
    # 实验1
    try:
        r1 = exp1_unlock_gate(model, tokenizer, info)
        all_results["exp1_unlock_gate"] = r1
    except Exception as e:
        print(f"Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_unlock_gate"] = {"error": str(e)}
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 实验2
    try:
        r2 = exp2_mlp_internal_ablation(model, tokenizer, info)
        all_results["exp2_mlp_internal"] = r2
    except Exception as e:
        print(f"Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_mlp_internal"] = {"error": str(e)}
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 实验3
    try:
        r3 = exp3_shared_private_causal(model, tokenizer, info)
        all_results["exp3_causal_injection"] = r3
    except Exception as e:
        print(f"Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_causal_injection"] = {"error": str(e)}
    
    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase449_{model_name}_r{round_num}.json"
    
    # 序列化处理
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj
    
    all_results = convert(all_results)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")
    
    # 释放模型
    release_model(model)
    
    print(f"\nPhase 449 {model_name} R{round_num} complete!")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
