"""
Phase 450: Inversion Gate Localization + Shared/Private Dual Channel + MLP Internal Fix
=======================================================================================
核心目标:
1. 定位GLM4 L24之后shared→cat反转来自哪个组件(attn path vs MLP path)
2. 验证shared/private双通道功能分离(类别 vs 属性)
3. 修复MLP内部消融: 直接替换MLP输出而非hook子模块

实验1: 组件路径替换 — 在GLM4 L20-L30逐层替换attn/MLP/residual路径,测shared→cat
实验2: 双通道功能分离 — shared/private分别注入,测6个属性维度
实验3: MLP输出级消融 — 将MLP输出中与shared方向对齐/正交的分量分别置零

用法:
  python tests/glm5/phase450_inversion_gate.py qwen3 1
  python tests/glm5/phase450_inversion_gate.py glm4 1
  python tests/glm5/phase450_inversion_gate.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import os, gc, time, json
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS)


def load_model_auto(model_name):
    """BF16 + device_map='auto' + sdpa"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True,
                                               local_files_only=True, use_fast=False)
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


def get_embedding_weight(model):
    """获取embedding权重矩阵"""
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens.weight.detach().float()
    return model.get_input_embeddings().weight.detach().float()


# ===== 实验1: 组件路径替换定位反转门控 =====
def exp1_component_path_patching(model, tokenizer, info):
    """
    核心思路: 
    - 用clean输入(无对象)和patched输入(有对象)分别前向
    - 在目标层替换某条路径(attn output / MLP output / residual) 
    - 测量替换后shared→cat效应是否改变
    - 重点关注GLM4的L20-L30区间
    
    更简洁的实现: 直接在目标层分别零化attn输出和MLP输出,看shared→cat变化
    """
    print(f"\n{'='*60}")
    print("Exp1: Component Path Ablation for Inversion Gate Localization")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    input_device = next(model.parameters()).device
    
    # 类别方向
    cat_words = ["fruit", "food", "produce"]
    opp_words = ["animal", "dog", "cat"]
    W_E = get_embedding_weight(model)
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu()
    cat_dir = cat_dir / (cat_dir.norm() + 1e-8)
    
    # 测试对象(fruit类别, 用于shared/private分解)
    obj_names = ["apple", "orange", "banana", "grape", "lemon", "mango"]
    alpha = 1.0
    
    # 采样层: 对于GLM4聚焦L18-L30, 对其他模型均匀采样
    if n_layers >= 30:
        scan_layers = list(range(max(0, n_layers*3//5 - 5), min(n_layers, n_layers*3//5 + 8)))
        # 加上首尾
        scan_layers = sorted(set([0, n_layers//4, n_layers//2] + scan_layers + [n_layers-2, n_layers-1]))
    else:
        scan_layers = list(range(n_layers))
    scan_layers = [l for l in scan_layers if l < n_layers]
    
    print(f"  Scanning layers: {scan_layers}")
    
    # Step 1: 收集各层基准hidden states和扰动hidden states
    print("  Step 1: Collecting base and perturbed hidden states...")
    
    obj_base_h = {}   # {obj: {layer: hidden_np}}
    obj_pert_h = {}   # {obj: {layer: hidden_np}}
    
    for obj_name in obj_names:
        text = f"The {obj_name} is a"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        last_pos = input_ids.shape[1] - 1
        
        # 基准
        base_h = {}
        def make_h_b(li):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    base_h[li] = out[0][0, last_pos].detach().float().cpu()
            return hook
        hooks_b = [layers[li].register_forward_hook(make_h_b(li)) for li in scan_layers]
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        for h in hooks_b:
            h.remove()
        obj_base_h[obj_name] = {li: base_h[li].numpy() for li in base_h if li in scan_layers}
        
        # 扰动
        perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
        pert_h = {}
        def make_h_p(li):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    pert_h[li] = out[0][0, last_pos].detach().float().cpu()
            return hook
        hooks_p = [layers[li].register_forward_hook(make_h_p(li)) for li in scan_layers]
        
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
        
        obj_pert_h[obj_name] = {li: pert_h[li].numpy() for li in pert_h if li in scan_layers}
    
    # Step 2: 在各层分解shared/private
    print("  Step 2: Decomposing shared/private at scan layers...")
    
    decomposition = {}
    for li in scan_layers:
        deltas = {}
        for o in obj_names:
            if li in obj_base_h.get(o, {}) and li in obj_pert_h.get(o, {}):
                d = obj_pert_h[o][li] - obj_base_h[o][li]
                if np.linalg.norm(d) > 1e-8:
                    deltas[o] = d
        if len(deltas) < 3:
            continue
        delta_matrix = np.stack(list(deltas.values()))
        shared = delta_matrix.mean(axis=0)
        private = {o: d - shared for o, d in deltas.items()}
        decomposition[li] = {"shared": shared, "private": private}
    
    # Step 3: 在每个扫描层做组件路径消融
    # 对每个层li: 分别消融attn输出和MLP输出,测shared→cat效应
    print("  Step 3: Component path ablation at each layer...")
    
    results = {"decomposition_layers": sorted(decomposition.keys()), "ablation_results": {}}
    
    # 用apple作为测试对象
    test_text = "The apple is a"
    test_inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=64)
    test_input_ids = test_inputs["input_ids"].to(input_device)
    test_attention_mask = test_inputs["attention_mask"].to(input_device)
    test_last_pos = test_input_ids.shape[1] - 1
    
    # 基准logits
    with torch.no_grad():
        out_base = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
        base_logits = out_base.logits[0, -1].float().cpu().numpy()
    
    base_cat = get_logit_for_words(base_logits, tokenizer, cat_words)
    base_color = get_logit_for_words(base_logits, tokenizer, ["red", "green", "yellow"])
    base_taste = get_logit_for_words(base_logits, tokenizer, ["sweet", "sour"])
    
    # 只在关键扫描层做(避免太慢)
    # 选择有decomposition的层 + GLM4反转区间附近的层
    ablation_layers = sorted(decomposition.keys())
    # 限制到15个关键层
    if len(ablation_layers) > 15:
        # 保留首尾和反转区间附近
        critical = [l for l in ablation_layers if n_layers*3//5 - 3 <= l <= n_layers*3//5 + 5]
        rest = [l for l in ablation_layers if l not in critical]
        step = max(1, len(rest) // 8)
        selected = sorted(set([rest[0]] + rest[::step] + critical + [rest[-1]]))[:15]
        ablation_layers = sorted(selected)
    
    print(f"  Ablation layers: {ablation_layers}")
    
    for abl_li in ablation_layers:
        layer = layers[abl_li]
        layer_result = {}
        
        # 确定attn和MLP子模块
        attn_module = None
        mlp_module = None
        if hasattr(layer, 'self_attn'):
            attn_module = layer.self_attn
        if hasattr(layer, 'mlp'):
            mlp_module = layer.mlp
        
        # 3种消融: zero_attn, zero_mlp, zero_both
        for abl_type, target_module in [("zero_attn", attn_module), ("zero_mlp", mlp_module)]:
            if target_module is None:
                continue
            
            def make_zero_hook():
                def hook(m, inp, out):
                    if isinstance(out, tuple):
                        return (torch.zeros_like(out[0]),) + out[1:]
                    return torch.zeros_like(out)
                return hook
            
            abl_hook = target_module.register_forward_hook(make_zero_hook())
            
            with torch.no_grad():
                try:
                    out_abl = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                    abl_logits = out_abl.logits[0, -1].float().cpu().numpy()
                except:
                    abl_logits = None
            
            abl_hook.remove()
            
            if abl_logits is not None:
                abl_cat = get_logit_for_words(abl_logits, tokenizer, cat_words)
                abl_color = get_logit_for_words(abl_logits, tokenizer, ["red", "green", "yellow"])
                abl_taste = get_logit_for_words(abl_logits, tokenizer, ["sweet", "sour"])
                
                layer_result[abl_type] = {
                    "cat_logit": round(abl_cat, 4) if abl_cat is not None else None,
                    "color_logit": round(abl_color, 4) if abl_color is not None else None,
                    "taste_logit": round(abl_taste, 4) if abl_taste is not None else None,
                    "cat_delta": round(abl_cat - base_cat, 4) if abl_cat is not None and base_cat is not None else None,
                    "color_delta": round(abl_color - base_color, 4) if abl_color is not None and base_color is not None else None,
                    "taste_delta": round(abl_taste - base_taste, 4) if abl_taste is not None and base_taste is not None else None,
                }
                
                print(f"  L{abl_li} {abl_type}: cat={abl_cat:.2f}(Δ{(abl_cat-base_cat):+.2f}), "
                      f"color={abl_color:.2f}(Δ{(abl_color-base_color):+.2f}), "
                      f"taste={abl_taste:.2f}(Δ{(abl_taste-base_taste):+.2f})")
            else:
                layer_result[abl_type] = {"error": "forward failed"}
        
        # Step 3b: 在消融状态下做shared注入,测shared→cat效应
        # 关键: 如果消融attn后shared→cat不再为负,则attn是反转来源
        if abl_li in decomposition:
            shared_vec = decomposition[abl_li]["shared"]
            inject_beta = 2.0
            inj_vec = shared_vec * inject_beta
            
            for abl_type, target_module in [("zero_attn", attn_module), ("zero_mlp", mlp_module)]:
                if target_module is None:
                    continue
                
                def make_zero_hook2():
                    def hook(m, inp, out):
                        if isinstance(out, tuple):
                            return (torch.zeros_like(out[0]),) + out[1:]
                        return torch.zeros_like(out)
                    return hook
                
                abl_hook = target_module.register_forward_hook(make_zero_hook2())
                
                # 在该层注入shared
                inj_tensor = torch.tensor(inj_vec, dtype=torch.bfloat16, device=input_device)
                def make_inject_hook(vec):
                    def hook(m, inp, out):
                        if isinstance(out, tuple):
                            new_out = out[0].clone()
                            new_out[0, test_last_pos] = new_out[0, test_last_pos] + vec.to(new_out.dtype)
                            return (new_out,) + out[1:]
                        return out
                    return hook
                
                inj_hook = layers[abl_li].register_forward_hook(make_inject_hook(inj_tensor))
                
                with torch.no_grad():
                    try:
                        out_inj_abl = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                        inj_abl_logits = out_inj_abl.logits[0, -1].float().cpu().numpy()
                    except:
                        inj_abl_logits = None
                
                abl_hook.remove()
                inj_hook.remove()
                
                if inj_abl_logits is not None:
                    inj_abl_cat = get_logit_for_words(inj_abl_logits, tokenizer, cat_words)
                    inj_abl_color = get_logit_for_words(inj_abl_logits, tokenizer, ["red", "green", "yellow"])
                    
                    # shared→cat在消融状态下的效应
                    base_for_this = layer_result.get(abl_type, {}).get("cat_logit", base_cat)
                    shared_cat_effect = (inj_abl_cat - base_for_this) if inj_abl_cat is not None and base_for_this is not None else None
                    
                    layer_result[f"{abl_type}_shared_inj_cat"] = round(shared_cat_effect, 4) if shared_cat_effect is not None else None
                    layer_result[f"{abl_type}_shared_inj_color"] = round(inj_abl_color - (layer_result.get(abl_type, {}).get("color_logit", base_color or 0)), 4)
                    
                    print(f"  L{abl_li} {abl_type}+shared_inj: shared→cat={shared_cat_effect:+.3f}")
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        results["ablation_results"][f"L{abl_li}"] = layer_result
    
    # Step 4: 无消融状态下的shared注入(作为对照)
    print("  Step 4: Baseline shared injection (no ablation)...")
    baseline_shared_inj = {}
    
    for li in ablation_layers:
        if li not in decomposition:
            continue
        shared_vec = decomposition[li]["shared"]
        inj_vec = shared_vec * 2.0
        inj_tensor = torch.tensor(inj_vec, dtype=torch.bfloat16, device=input_device)
        
        def make_inject_hook(vec):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    new_out = out[0].clone()
                    new_out[0, test_last_pos] = new_out[0, test_last_pos] + vec.to(new_out.dtype)
                    return (new_out,) + out[1:]
                return out
            return hook
        
        inj_hook = layers[li].register_forward_hook(make_inject_hook(inj_tensor))
        with torch.no_grad():
            try:
                out_inj = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                inj_logits = out_inj.logits[0, -1].float().cpu().numpy()
            except:
                inj_logits = None
        inj_hook.remove()
        
        if inj_logits is not None:
            inj_cat = get_logit_for_words(inj_logits, tokenizer, cat_words)
            shared_cat_effect = inj_cat - base_cat if inj_cat is not None and base_cat is not None else None
            baseline_shared_inj[f"L{li}"] = round(shared_cat_effect, 4) if shared_cat_effect is not None else None
            print(f"  L{li} shared→cat (no ablation): {shared_cat_effect:+.3f}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    results["baseline_shared_injection"] = baseline_shared_inj
    results["base_logits"] = {"cat": round(base_cat, 4), "color": round(base_color, 4), "taste": round(base_taste, 4)}
    
    # 汇总
    print(f"\n  === Exp1 Summary: Component Path Effects ===")
    print(f"  Layer | attn→catΔ | mlp→catΔ | shared→cat(clean) | shared→cat(+zero_attn) | shared→cat(+zero_mlp)")
    for layer_key in sorted(results["ablation_results"].keys()):
        lr = results["ablation_results"][layer_key]
        a = lr.get("zero_attn", {}).get("cat_delta", "N/A")
        m = lr.get("zero_mlp", {}).get("cat_delta", "N/A")
        s_clean = baseline_shared_inj.get(layer_key, "N/A")
        s_zero_attn = lr.get("zero_attn_shared_inj_cat", "N/A")
        s_zero_mlp = lr.get("zero_mlp_shared_inj_cat", "N/A")
        print(f"  {layer_key:5s} | {a:+.3f}    | {m:+.3f}    | {s_clean:+.3f}             | {s_zero_attn}                  | {s_zero_mlp}")
    
    return results


# ===== 实验2: Shared/Private双通道功能分离 =====
def exp2_dual_channel_functional(model, tokenizer, info):
    """
    验证shared通道主要控制类别泛化, private通道主要控制对象特异属性
    
    方法: 在每个目标层分别注入shared/private,测6个属性维度:
    - category (类别)
    - color (颜色)  
    - taste (味道)
    - part (部件)
    - material (材料)
    - function (功能)
    
    同时测试3个类别: fruit, animal, tool
    """
    print(f"\n{'='*60}")
    print("Exp2: Shared/Private Dual Channel Functional Separation")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    input_device = next(model.parameters()).device
    W_E = get_embedding_weight(model)
    
    # 3个类别,每个类别3个对象
    CATEGORY_CONFIGS = {
        "fruit": {
            "objects": ["apple", "orange", "banana"],
            "opp_words": ["animal", "dog", "cat"],
            "category_words": ["fruit", "food", "produce"],
            "attrs": {
                "color": ["red", "green", "yellow"],
                "taste": ["sweet", "sour", "juicy"],
                "part": ["seed", "skin", "core"],
            }
        },
        "animal": {
            "objects": ["dog", "cat", "horse"],
            "opp_words": ["fruit", "apple", "tool"],
            "category_words": ["animal", "pet", "mammal"],
            "attrs": {
                "color": ["brown", "black", "white"],
                "part": ["leg", "tail", "fur"],
                "habitat": ["home", "field", "forest"],
            }
        },
        "tool": {
            "objects": ["knife", "hammer", "spoon"],
            "opp_words": ["fruit", "animal", "dog"],
            "category_words": ["tool", "weapon", "instrument"],
            "attrs": {
                "color": ["silver", "gray", "metallic"],
                "part": ["blade", "handle", "edge"],
                "material": ["metal", "steel", "iron"],
            }
        },
    }
    
    # 采样层
    target_layers = [0, n_layers//6, n_layers//3, n_layers//2, 2*n_layers//3, 5*n_layers//6, n_layers-2, n_layers-1]
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    all_results = {}
    
    for cat_name, cat_cfg in CATEGORY_CONFIGS.items():
        print(f"\n  --- Category: {cat_name} ---")
        
        obj_names = cat_cfg["objects"]
        opp_words = cat_cfg["opp_words"]
        cat_words = cat_cfg["category_words"]
        
        # 类别方向
        cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
        opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
        cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu()
        cat_dir = cat_dir / (cat_dir.norm() + 1e-8)
        
        alpha = 1.0
        
        # 收集各层delta
        obj_layer_deltas = {}
        for obj_name in obj_names:
            text = f"The {obj_name} is a"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            base_h = {}
            def make_h_b(li):
                def hook(m, inp, out):
                    if isinstance(out, tuple):
                        base_h[li] = out[0][0, last_pos].detach().float().cpu()
                return hook
            hooks_b = [layers[li].register_forward_hook(make_h_b(li)) for li in target_layers]
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            for h in hooks_b:
                h.remove()
            
            perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
            pert_h = {}
            def make_h_p(li):
                def hook(m, inp, out):
                    if isinstance(out, tuple):
                        pert_h[li] = out[0][0, last_pos].detach().float().cpu()
                return hook
            hooks_p = [layers[li].register_forward_hook(make_h_p(li)) for li in target_layers]
            
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
            
            obj_layer_deltas[obj_name] = {}
            for li in target_layers:
                if li in pert_h and li in base_h:
                    obj_layer_deltas[obj_name][li] = (pert_h[li] - base_h[li]).numpy()
        
        # 分解shared/private
        decomposition = {}
        for li in target_layers:
            deltas = {o: obj_layer_deltas[o].get(li) for o in obj_names if li in obj_layer_deltas.get(o, {})}
            valid = {o: d for o, d in deltas.items() if d is not None and np.linalg.norm(d) > 1e-8}
            if len(valid) < 2:
                continue
            delta_matrix = np.stack(list(valid.values()))
            shared = delta_matrix.mean(axis=0)
            private = {o: d - shared for o, d in valid.items()}
            decomposition[li] = {"shared": shared, "private": private}
        
        # 因果注入
        inject_beta = 2.0
        cat_results = {}
        
        # 用第一个对象作为测试对象
        test_obj = obj_names[0]
        test_text = f"The {test_obj} is a"
        test_inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=64)
        test_input_ids = test_inputs["input_ids"].to(input_device)
        test_attention_mask = test_inputs["attention_mask"].to(input_device)
        test_last_pos = test_input_ids.shape[1] - 1
        
        # 基准logits
        with torch.no_grad():
            out_base = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
            base_logits = out_base.logits[0, -1].float().cpu().numpy()
        
        base_attrs = {}
        base_attrs["category"] = get_logit_for_words(base_logits, tokenizer, cat_words)
        for attr_name, attr_words in cat_cfg["attrs"].items():
            base_attrs[attr_name] = get_logit_for_words(base_logits, tokenizer, attr_words)
        
        for li in target_layers:
            if li not in decomposition:
                continue
            
            shared_vec = decomposition[li]["shared"]
            private_vec = decomposition[li]["private"].get(test_obj, np.zeros(d_model))
            
            injections = {
                "shared": shared_vec * inject_beta,
                "private": private_vec * inject_beta,
                "shared+private": (shared_vec + private_vec) * inject_beta,
            }
            
            layer_result = {}
            
            for inj_name, inj_vec in injections.items():
                inj_tensor = torch.tensor(inj_vec, dtype=torch.bfloat16, device=input_device)
                
                def make_inject_hook(vec):
                    def hook(m, inp, out):
                        if isinstance(out, tuple):
                            new_out = out[0].clone()
                            new_out[0, test_last_pos] = new_out[0, test_last_pos] + vec.to(new_out.dtype)
                            return (new_out,) + out[1:]
                        return out
                    return hook
                
                inj_hook = layers[li].register_forward_hook(make_inject_hook(inj_tensor))
                with torch.no_grad():
                    try:
                        out_inj = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                        inj_logits = out_inj.logits[0, -1].float().cpu().numpy()
                    except:
                        inj_logits = None
                inj_hook.remove()
                
                if inj_logits is not None:
                    deltas = {}
                    deltas["category"] = get_logit_for_words(inj_logits, tokenizer, cat_words)
                    for attr_name, attr_words in cat_cfg["attrs"].items():
                        deltas[attr_name] = get_logit_for_words(inj_logits, tokenizer, attr_words)
                    
                    # 计算delta
                    delta_dict = {}
                    for k in deltas:
                        if deltas[k] is not None and base_attrs.get(k) is not None:
                            delta_dict[k] = round(deltas[k] - base_attrs[k], 4)
                    layer_result[inj_name] = delta_dict
            
            cat_results[f"L{li}"] = layer_result
            
            # 日志
            s = layer_result.get("shared", {})
            p = layer_result.get("private", {})
            print(f"  L{li}: shared→cat={s.get('category',0):+.2f}, shared→color={s.get('color',s.get('part',0)):+.2f} | "
                  f"private→cat={p.get('category',0):+.2f}, private→color={p.get('color',p.get('part',0)):+.2f}")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        all_results[cat_name] = cat_results
    
    # 汇总
    print(f"\n  === Exp2 Summary: Shared/Private Functional Separation ===")
    for cat_name in CATEGORY_CONFIGS:
        cat_results = all_results.get(cat_name, {})
        print(f"\n  --- {cat_name} ---")
        print(f"  Layer | shared→cat | private→cat | shared→attr1 | private→attr1 | private/cat_ratio")
        for layer_key in sorted(cat_results.keys()):
            lr = cat_results[layer_key]
            s = lr.get("shared", {})
            p = lr.get("private", {})
            sc = s.get("category", 0)
            pc = p.get("category", 0)
            # 第一个非category属性
            attr_keys = [k for k in s if k != "category"]
            sa1 = s.get(attr_keys[0], 0) if attr_keys else 0
            pa1 = p.get(attr_keys[0], 0) if attr_keys else 0
            ratio = abs(pc) / (abs(sc) + abs(pc) + 1e-6)
            print(f"  {layer_key:5s} | {sc:+.3f}     | {pc:+.3f}      | {sa1:+.3f}       | {pa1:+.3f}        | {ratio:.3f}")
    
    return all_results


# ===== 实验3: MLP输出级消融(修复版) =====
def exp3_mlp_output_ablation(model, tokenizer, info):
    """
    修复Phase 449 Exp2的bug: 不再hook子模块,而是:
    1. 捕获MLP输出向量
    2. 在MLP输出中分离与shared方向对齐/正交的分量
    3. 分别置零测效果
    
    这避免了gate/up/down hook返回相同值的问题
    """
    print(f"\n{'='*60}")
    print("Exp3: MLP Output-Level Ablation (Fixed)")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    input_device = next(model.parameters()).device
    W_E = get_embedding_weight(model)
    
    cat_words = ["fruit", "food", "produce"]
    opp_words = ["animal", "dog", "cat"]
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu()
    cat_dir = cat_dir / (cat_dir.norm() + 1e-8)
    
    # 测试对象
    obj_names = ["apple", "orange", "banana", "grape", "lemon", "mango"]
    alpha = 1.0
    
    # Step 1: 收集各层MLP输出和residual
    print("  Step 1: Capturing MLP outputs and residuals...")
    
    mlp_outputs = {}  # {obj: {layer: mlp_out_np}}
    attn_outputs = {}  # {obj: {layer: attn_out_np}}
    layer_outputs = {}  # {obj: {layer: layer_out_np}}
    
    for obj_name in obj_names:
        text = f"The {obj_name} is a"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        last_pos = input_ids.shape[1] - 1
        
        # 用hook捕获MLP输出和attn输出
        captured = {}
        
        def make_capture_hook(li, component):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    captured[(li, component)] = out[0][0, last_pos].detach().float().cpu()
            return hook
        
        hooks = []
        for li in range(n_layers):
            layer = layers[li]
            if hasattr(layer, 'self_attn'):
                hooks.append(layer.self_attn.register_forward_hook(make_capture_hook(li, 'attn')))
            if hasattr(layer, 'mlp'):
                hooks.append(layer.mlp.register_forward_hook(make_capture_hook(li, 'mlp')))
            # 层输出
            hooks.append(layer.register_forward_hook(make_capture_hook(li, 'layer')))
        
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        for h in hooks:
            h.remove()
        
        mlp_outputs[obj_name] = {li: captured[(li, 'mlp')].numpy() 
                                  for li in range(n_layers) if (li, 'mlp') in captured}
        attn_outputs[obj_name] = {li: captured[(li, 'attn')].numpy() 
                                   for li in range(n_layers) if (li, 'attn') in captured}
        layer_outputs[obj_name] = {li: captured[(li, 'layer')].numpy() 
                                    for li in range(n_layers) if (li, 'layer') in captured}
        
        print(f"  {obj_name}: captured {len(mlp_outputs[obj_name])} MLP outputs, "
              f"{len(attn_outputs[obj_name])} attn outputs")
    
    # Step 2: 分解MLP输出中与shared方向对齐/正交的分量
    print("  Step 2: Decomposing MLP output into shared-aligned and shared-orthogonal...")
    
    # 采样层
    target_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    # 先收集各层的shared方向(从delta分解)
    layer_shared_dirs = {}
    for li in target_layers:
        deltas = {}
        for o in obj_names:
            mlp_out = mlp_outputs.get(o, {}).get(li)
            if mlp_out is not None and np.linalg.norm(mlp_out) > 1e-8:
                deltas[o] = mlp_out
        if len(deltas) < 3:
            continue
        delta_matrix = np.stack(list(deltas.values()))
        shared = delta_matrix.mean(axis=0)
        shared_norm = np.linalg.norm(shared)
        if shared_norm > 1e-8:
            layer_shared_dirs[li] = shared / shared_norm
    
    # Step 3: MLP输出投影分解
    print("  Step 3: Projecting MLP output onto shared direction...")
    
    results = {"target_layers": target_layers, "projection_results": {}}
    
    test_text = "The apple is a"
    test_inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=64)
    test_input_ids = test_inputs["input_ids"].to(input_device)
    test_attention_mask = test_inputs["attention_mask"].to(input_device)
    test_last_pos = test_input_ids.shape[1] - 1
    
    # 基准
    with torch.no_grad():
        out_base = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
        base_logits = out_base.logits[0, -1].float().cpu().numpy()
    base_cat = get_logit_for_words(base_logits, tokenizer, cat_words)
    base_color = get_logit_for_words(base_logits, tokenizer, ["red", "green", "yellow"])
    
    for li in target_layers:
        if li not in layer_shared_dirs:
            continue
        
        shared_dir = layer_shared_dirs[li]
        
        # apple的MLP输出在该层的投影
        apple_mlp = mlp_outputs.get("apple", {}).get(li)
        if apple_mlp is None:
            continue
        
        # 分解: mlp_out = shared_component + orthogonal_component
        proj_coeff = np.dot(apple_mlp, shared_dir)
        shared_component = proj_coeff * shared_dir
        ortho_component = apple_mlp - shared_component
        
        # 3种MLP输出替换:
        # 1. remove_shared: 只保留正交分量(去掉shared对齐部分)
        # 2. remove_ortho: 只保留shared对齐部分(去掉正交分量)
        # 3. zero_mlp: 完全零化MLP
        # 4. negate_shared: 反转shared分量
        modifications = {
            "remove_shared": ortho_component,
            "remove_ortho": shared_component,
            "negate_shared": -shared_component + ortho_component,
        }
        
        layer_result = {"proj_coeff": round(float(proj_coeff), 4),
                       "shared_norm": round(float(np.linalg.norm(shared_component)), 4),
                       "ortho_norm": round(float(np.linalg.norm(ortho_component)), 4)}
        
        for mod_name, target_mlp_out in modifications.items():
            # 在目标层替换MLP输出
            def make_mlp_replace_hook(target_out):
                target_t = torch.tensor(target_out, dtype=torch.bfloat16, device=input_device)
                def hook(m, inp, out):
                    if isinstance(out, tuple):
                        # 替换MLP输出为目标值
                        new_out = out[0].clone()
                        new_out[0, test_last_pos] = target_t.to(new_out.dtype)
                        return (new_out,) + out[1:]
                    return out
                return hook
            
            mlp_module = layers[li].mlp if hasattr(layers[li], 'mlp') else None
            if mlp_module is None:
                continue
            
            replace_hook = mlp_module.register_forward_hook(make_mlp_replace_hook(target_mlp_out))
            
            with torch.no_grad():
                try:
                    out_mod = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                    mod_logits = out_mod.logits[0, -1].float().cpu().numpy()
                except:
                    mod_logits = None
            
            replace_hook.remove()
            
            if mod_logits is not None:
                mod_cat = get_logit_for_words(mod_logits, tokenizer, cat_words)
                mod_color = get_logit_for_words(mod_logits, tokenizer, ["red", "green", "yellow"])
                
                layer_result[mod_name] = {
                    "cat_logit": round(mod_cat, 4) if mod_cat is not None else None,
                    "cat_delta": round(mod_cat - base_cat, 4) if mod_cat is not None and base_cat is not None else None,
                    "color_logit": round(mod_color, 4) if mod_color is not None else None,
                    "color_delta": round(mod_color - base_color, 4) if mod_color is not None and base_color is not None else None,
                }
                
                print(f"  L{li} {mod_name}: cat={mod_cat:.2f}(Δ{(mod_cat-base_cat):+.2f}), "
                      f"color={mod_color:.2f}(Δ{(mod_color-base_color):+.2f})")
            else:
                layer_result[mod_name] = {"error": "forward failed"}
        
        results["projection_results"][f"L{li}"] = layer_result
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总
    print(f"\n  === Exp3 Summary: MLP Output-Level Ablation ===")
    print(f"  Layer | proj_coeff | shared_norm | ortho_norm | remove_shared→catΔ | remove_ortho→catΔ | negate_shared→catΔ")
    for layer_key in sorted(results["projection_results"].keys()):
        lr = results["projection_results"][layer_key]
        pc = lr.get("proj_coeff", 0)
        sn = lr.get("shared_norm", 0)
        on = lr.get("ortho_norm", 0)
        rs = lr.get("remove_shared", {}).get("cat_delta", "N/A")
        ro = lr.get("remove_ortho", {}).get("cat_delta", "N/A")
        ns = lr.get("negate_shared", {}).get("cat_delta", "N/A")
        print(f"  {layer_key:5s} | {pc:+.4f}   | {sn:.4f}     | {on:.4f}    | {rs}                | {ro}               | {ns}")
    
    return results


# ===== 主函数 =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print(f"Phase 450: Inversion Gate Localization + Dual Channel + MLP Fix")
    print(f"Model: {model_name}, Round: {round_num}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载模型
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    print(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}, mlp_type={info.mlp_type}")
    
    all_results = {}
    
    # 实验1: 组件路径替换
    t0 = time.time()
    try:
        r1 = exp1_component_path_patching(model, tokenizer, info)
        all_results["exp1_component_patching"] = r1
    except Exception as e:
        print(f"Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_component_patching"] = {"error": str(e)}
    print(f"  Exp1 elapsed: {time.time()-t0:.1f}s")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 实验2: 双通道功能分离
    t0 = time.time()
    try:
        r2 = exp2_dual_channel_functional(model, tokenizer, info)
        all_results["exp2_dual_channel"] = r2
    except Exception as e:
        print(f"Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_dual_channel"] = {"error": str(e)}
    print(f"  Exp2 elapsed: {time.time()-t0:.1f}s")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 实验3: MLP输出级消融
    t0 = time.time()
    try:
        r3 = exp3_mlp_output_ablation(model, tokenizer, info)
        all_results["exp3_mlp_output_ablation"] = r3
    except Exception as e:
        print(f"Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_mlp_output_ablation"] = {"error": str(e)}
    print(f"  Exp3 elapsed: {time.time()-t0:.1f}s")
    
    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase450_{model_name}_r{round_num}.json"
    
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
    
    print(f"\nPhase 450 {model_name} R{round_num} complete!")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
