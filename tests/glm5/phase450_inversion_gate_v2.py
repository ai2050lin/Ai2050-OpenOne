"""
Phase 450 v2: Inversion Gate Localization + Shared/Private Dual Channel + MLP Internal Fix
============================================================================================
核心目标:
1. 定位GLM4 L24之后shared→cat反转来自哪个组件(attn path vs MLP path)
2. 验证shared/private双通道功能分离(类别 vs 属性)
3. 修复MLP内部消融: 直接替换MLP输出而非hook子模块

改进(相对v1):
- 修复device_map="auto"下的设备兼容性(hook中动态获取tensor设备)
- 增加flash attention (attn_implementation="flash_attention_2" 或 sdpa)
- 增加定时日志输出
- 减少Exp3的内存占用(分批收集)
- 确保GLM4/DS7B深层不缺失

用法:
  python tests/glm5/phase450_inversion_gate_v2.py qwen3 1
  python tests/glm5/phase450_inversion_gate_v2.py glm4 1
  python tests/glm5/phase450_inversion_gate_v2.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import os, gc, time, json, logging
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS)

# 定时日志
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger("phase450")

# 进度计时器
_last_log_time = [time.time()]
def periodic_log(msg, interval=30):
    """每interval秒输出一次日志"""
    now = time.time()
    if now - _last_log_time[0] >= interval or 'complete' in msg.lower() or 'step' in msg.lower():
        log.info(msg)
        _last_log_time[0] = now


def load_model_auto(model_name):
    """BF16 + device_map='auto' + flash attention"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log.info(f"Loading {model_name} (bfloat16 + device_map=auto + flash)...")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True,
                                               local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 尝试flash_attention_2, 失败则回退sdpa
    for attn_impl in ["flash_attention_2", "sdpa"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=attn_impl)
            log.info(f"  Loaded with attn_implementation={attn_impl}")
            break
        except Exception as e:
            log.info(f"  {attn_impl} failed: {e}, trying next...")
            continue
    else:
        # 最后回退eager
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="eager")
        log.info(f"  Loaded with attn_implementation=eager (fallback)")
    
    model.eval()
    
    # 显示层分配
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        log.info(f"  {model_name}: GPU={gpu_count} components, CPU={cpu_count} components")
    
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


def get_layer_device(layers, layer_idx):
    """获取某层的设备"""
    try:
        # 尝试从self_attn.q_proj获取设备
        layer = layers[layer_idx]
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'q_proj'):
            return layer.self_attn.q_proj.weight.device
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            if hasattr(mlp, 'down_proj'):
                return mlp.down_proj.weight.device
    except:
        pass
    return next(layers[0].parameters()).device


# ===== 实验1: 组件路径消融定位反转门控 =====
def exp1_component_path_ablation(model, tokenizer, info):
    """
    核心思路:
    - 在GLM4关键层(L20-L30)分别消融attn和MLP路径
    - 消融后注入shared分量,测shared→cat效应
    - 如果消融某组件后shared→cat不再为负,则该组件是反转来源
    - 对Qwen3/DS7B同样测试,作为对照
    """
    print(f"\n{'='*60}")
    print("Exp1: Component Path Ablation for Inversion Gate Localization")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    
    # 类别方向
    cat_words = ["fruit", "food", "produce"]
    opp_words = ["animal", "dog", "cat"]
    W_E = get_embedding_weight(model)
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu()
    cat_dir = cat_dir / (cat_dir.norm() + 1e-8)
    
    # 测试对象
    obj_names = ["apple", "orange", "banana", "grape", "lemon"]
    alpha = 1.0
    
    # 采样层: 覆盖反转区间 + 首尾
    if n_layers >= 30:
        mid = n_layers * 3 // 5  # ~60%深度
        scan_layers = sorted(set(
            [0, n_layers//4, n_layers//2] +
            list(range(max(0, mid-5), min(n_layers, mid+8))) +
            [n_layers-2, n_layers-1]
        ))
    else:
        scan_layers = list(range(n_layers))
    scan_layers = [l for l in scan_layers if l < n_layers]
    print(f"  Scan layers: {scan_layers}")
    
    # Step 1: 收集各层shared/private分解
    print("  Step 1: Collecting deltas for shared/private decomposition...")
    t_step1 = time.time()
    
    obj_base_h = {}
    obj_pert_h = {}
    
    for obj_name in obj_names:
        text = f"The {obj_name} is a"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_device = get_layer_device(layers, 0)
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
        
        # 扰动(embedding层注入)
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
        except Exception as e:
            print(f"  Warning: perturbed forward failed for {obj_name}: {e}")
        finally:
            if embed_hook:
                embed_hook.remove()
            for h in hooks_p:
                h.remove()
        
        obj_pert_h[obj_name] = {li: pert_h[li].numpy() for li in pert_h if li in scan_layers}
        
        periodic_log(f"  Step1: {obj_name} done ({time.time()-t_step1:.1f}s)")
    
    # 分解shared/private
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
    
    print(f"  Step 1 complete: {len(decomposition)} layers decomposed ({time.time()-t_step1:.1f}s)")
    
    # Step 2: 基准logits
    print("  Step 2: Computing baseline logits...")
    test_text = "The apple is a"
    test_inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=64)
    input_device = get_layer_device(layers, 0)
    test_input_ids = test_inputs["input_ids"].to(input_device)
    test_attention_mask = test_inputs["attention_mask"].to(input_device)
    test_last_pos = test_input_ids.shape[1] - 1
    
    with torch.no_grad():
        out_base = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
        base_logits = out_base.logits[0, -1].float().cpu().numpy()
    
    base_cat = get_logit_for_words(base_logits, tokenizer, cat_words)
    base_color = get_logit_for_words(base_logits, tokenizer, ["red", "green", "yellow"])
    print(f"  Base: cat={base_cat:.2f}, color={base_color:.2f}")
    
    # Step 3: 逐层消融 + 注入
    # 只在关键层做(有decomposition的)
    ablation_layers = sorted(decomposition.keys())
    # 限制数量避免太慢(R1基础测试)
    if len(ablation_layers) > 12:
        critical = [l for l in ablation_layers if n_layers//2-5 <= l <= n_layers//2+8]
        rest = [l for l in ablation_layers if l not in critical]
        step = max(1, len(rest) // 6)
        selected = sorted(set([rest[0]] + rest[::step] + critical + [rest[-1]]))[:12]
        ablation_layers = selected
    
    print(f"  Ablation layers: {ablation_layers}")
    
    results = {"decomposition_layers": sorted(decomposition.keys()),
               "ablation_layers": ablation_layers, "ablation_results": {},
               "base_logits": {"cat": round(base_cat, 4), "color": round(base_color, 4)}}
    
    # Step 3a: 基准shared→cat(无消融)
    print("  Step 3a: Baseline shared→cat (no ablation)...")
    baseline_shared_inj = {}
    for li in ablation_layers:
        if li not in decomposition:
            continue
        shared_vec = decomposition[li]["shared"]
        inj_vec = shared_vec * 2.0
        
        layer_dev = get_layer_device(layers, li)
        inj_tensor = torch.tensor(inj_vec, dtype=torch.bfloat16, device=layer_dev)
        
        def make_inject_hook(vec, _li=li):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    new_out = out[0].clone()
                    new_out[0, test_last_pos] = new_out[0, test_last_pos] + vec.to(new_out.device).to(new_out.dtype)
                    return (new_out,) + out[1:]
                return out
            return hook
        
        inj_hook = layers[li].register_forward_hook(make_inject_hook(inj_tensor, li))
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
    
    # Step 3b: 消融attn/MLP后测shared→cat
    print("  Step 3b: Component ablation + shared injection...")
    t_step3 = time.time()
    
    for idx, abl_li in enumerate(ablation_layers):
        if abl_li not in decomposition:
            continue
        
        layer = layers[abl_li]
        layer_result = {}
        
        attn_module = getattr(layer, 'self_attn', None)
        mlp_module = getattr(layer, 'mlp', None)
        
        # 先做消融(无注入),测基础属性变化
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
                
                layer_result[abl_type] = {
                    "cat_logit": round(abl_cat, 4) if abl_cat is not None else None,
                    "cat_delta": round(abl_cat - base_cat, 4) if abl_cat is not None and base_cat is not None else None,
                    "color_delta": round(abl_color - base_color, 4) if abl_color is not None and base_color is not None else None,
                }
                print(f"  L{abl_li} {abl_type}: catΔ={((abl_cat or 0)-base_cat):+.3f}")
            else:
                layer_result[abl_type] = {"error": "forward failed"}
        
        # 消融 + shared注入
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
            
            # 在消融状态下注入shared
            layer_dev = get_layer_device(layers, abl_li)
            inj_tensor = torch.tensor(inj_vec, dtype=torch.bfloat16, device=layer_dev)
            
            def make_inject_hook2(vec, _li=abl_li):
                def hook(m, inp, out):
                    if isinstance(out, tuple):
                        new_out = out[0].clone()
                        new_out[0, test_last_pos] = new_out[0, test_last_pos] + vec.to(new_out.device).to(new_out.dtype)
                        return (new_out,) + out[1:]
                    return out
                return hook
            
            inj_hook = layers[abl_li].register_forward_hook(make_inject_hook2(inj_tensor, abl_li))
            
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
                # 消融后的base
                abl_base_cat = layer_result.get(abl_type, {}).get("cat_logit", base_cat)
                shared_cat_effect = (inj_abl_cat - abl_base_cat) if inj_abl_cat is not None and abl_base_cat is not None else None
                
                layer_result[f"{abl_type}_shared_inj_cat"] = round(shared_cat_effect, 4) if shared_cat_effect is not None else None
                print(f"  L{abl_li} {abl_type}+shared_inj: shared→cat={shared_cat_effect:+.3f}")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        results["ablation_results"][f"L{abl_li}"] = layer_result
        periodic_log(f"  Exp1 L{abl_li} done ({idx+1}/{len(ablation_layers)}, {time.time()-t_step3:.1f}s)")
    
    # 汇总
    print(f"\n  === Exp1 Summary ===")
    print(f"  Layer | shared→cat(clean) | attn→catΔ | mlp→catΔ | shared→cat(+zero_attn) | shared→cat(+zero_mlp)")
    for layer_key in sorted(results["ablation_results"].keys()):
        lr = results["ablation_results"][layer_key]
        s_clean = baseline_shared_inj.get(layer_key, "N/A")
        a_delta = lr.get("zero_attn", {}).get("cat_delta", "N/A")
        m_delta = lr.get("zero_mlp", {}).get("cat_delta", "N/A")
        s_zero_a = lr.get("zero_attn_shared_inj_cat", "N/A")
        s_zero_m = lr.get("zero_mlp_shared_inj_cat", "N/A")
        print(f"  {layer_key:5s} | {s_clean}              | {a_delta}    | {m_delta}    | {s_zero_a}                  | {s_zero_m}")
    
    results["baseline_shared_injection"] = baseline_shared_inj
    return results


# ===== 实验2: Shared/Private双通道功能分离 =====
def exp2_dual_channel_functional(model, tokenizer, info):
    """
    验证shared通道主要控制类别泛化, private通道主要控制对象特异属性
    3个类别(fruit/animal/tool), 每个类别3个对象, 8个采样层
    """
    print(f"\n{'='*60}")
    print("Exp2: Shared/Private Dual Channel Functional Separation")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    W_E = get_embedding_weight(model)
    
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
        t_cat = time.time()
        
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
            input_device = get_layer_device(layers, 0)
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
        
        test_obj = obj_names[0]
        test_text = f"The {test_obj} is a"
        test_inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=64)
        input_device = get_layer_device(layers, 0)
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
                layer_dev = get_layer_device(layers, li)
                inj_tensor = torch.tensor(inj_vec, dtype=torch.bfloat16, device=layer_dev)
                
                def make_inject_hook(vec, _li=li):
                    def hook(m, inp, out):
                        if isinstance(out, tuple):
                            new_out = out[0].clone()
                            new_out[0, test_last_pos] = new_out[0, test_last_pos] + vec.to(new_out.device).to(new_out.dtype)
                            return (new_out,) + out[1:]
                        return out
                    return hook
                
                inj_hook = layers[li].register_forward_hook(make_inject_hook(inj_tensor, li))
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
                    
                    delta_dict = {}
                    for k in deltas:
                        if deltas[k] is not None and base_attrs.get(k) is not None:
                            delta_dict[k] = round(deltas[k] - base_attrs[k], 4)
                    layer_result[inj_name] = delta_dict
            
            cat_results[f"L{li}"] = layer_result
            
            s = layer_result.get("shared", {})
            p = layer_result.get("private", {})
            sp = layer_result.get("shared+private", {})
            print(f"  L{li}: shared→cat={s.get('category',0):+.2f}, private→cat={p.get('category',0):+.2f}, "
                  f"S+P→cat={sp.get('category',0):+.2f}")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        all_results[cat_name] = cat_results
        print(f"  Category {cat_name} done ({time.time()-t_cat:.1f}s)")
    
    # 汇总
    print(f"\n  === Exp2 Summary ===")
    for cat_name in CATEGORY_CONFIGS:
        cat_results = all_results.get(cat_name, {})
        print(f"\n  --- {cat_name} ---")
        print(f"  Layer | shared→cat | private→cat | S+P→cat | shared→attr | private→attr | private/cat ratio")
        for layer_key in sorted(cat_results.keys()):
            lr = cat_results[layer_key]
            s = lr.get("shared", {})
            p = lr.get("private", {})
            sp = lr.get("shared+private", {})
            sc = s.get("category", 0)
            pc = p.get("category", 0)
            spc = sp.get("category", 0)
            attr_keys = [k for k in s if k != "category"]
            sa1 = s.get(attr_keys[0], 0) if attr_keys else 0
            pa1 = p.get(attr_keys[0], 0) if attr_keys else 0
            ratio = abs(pc) / (abs(sc) + abs(pc) + 1e-6)
            print(f"  {layer_key:5s} | {sc:+.3f}     | {pc:+.3f}      | {spc:+.3f}  | {sa1:+.3f}       | {pa1:+.3f}        | {ratio:.3f}")
    
    return all_results


# ===== 实验3: MLP输出级消融(修复版) =====
def exp3_mlp_output_ablation(model, tokenizer, info):
    """
    修复Phase 449 Exp2的bug:
    不再hook子模块gate/up/down,而是:
    1. 捕获MLP输出向量
    2. 在MLP输出中分离与shared方向对齐/正交的分量
    3. 分别替换测效果
    
    关键改进: 分批收集,避免内存爆炸
    """
    print(f"\n{'='*60}")
    print("Exp3: MLP Output-Level Ablation (Fixed)")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    W_E = get_embedding_weight(model)
    
    cat_words = ["fruit", "food", "produce"]
    opp_words = ["animal", "dog", "cat"]
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu()
    cat_dir = cat_dir / (cat_dir.norm() + 1e-8)
    
    # 测试对象
    obj_names = ["apple", "orange", "banana", "grape", "lemon"]
    alpha = 1.0
    
    # 采样层(减少, 避免太慢)
    target_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    # Step 1: 在目标层收集MLP输出(分批,每批1个对象)
    print("  Step 1: Capturing MLP outputs (batched)...")
    t_step1 = time.time()
    
    mlp_outputs = {}  # {obj: {layer: mlp_out_np}}
    
    for obj_name in obj_names:
        text = f"The {obj_name} is a"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_device = get_layer_device(layers, 0)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        last_pos = input_ids.shape[1] - 1
        
        captured = {}
        def make_capture_hook(li):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    captured[li] = out[0][0, last_pos].detach().float().cpu().numpy()
            return hook
        
        hooks = []
        for li in target_layers:
            layer = layers[li]
            if hasattr(layer, 'mlp'):
                hooks.append(layer.mlp.register_forward_hook(make_capture_hook(li)))
        
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        for h in hooks:
            h.remove()
        
        mlp_outputs[obj_name] = {li: captured[li] for li in target_layers if li in captured}
        periodic_log(f"  MLP capture: {obj_name} done ({time.time()-t_step1:.1f}s)")
    
    print(f"  Step 1 complete ({time.time()-t_step1:.1f}s)")
    
    # Step 2: 分解MLP输出的shared/private
    print("  Step 2: Decomposing MLP output into shared/private...")
    
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
    
    # Step 3: MLP输出投影分解与替换
    print("  Step 3: MLP output projection ablation...")
    t_step3 = time.time()
    
    results = {"target_layers": target_layers, "projection_results": {}}
    
    test_text = "The apple is a"
    test_inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=64)
    input_device = get_layer_device(layers, 0)
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
        
        apple_mlp = mlp_outputs.get("apple", {}).get(li)
        if apple_mlp is None:
            continue
        
        # 分解: mlp_out = shared_component + orthogonal_component
        proj_coeff = np.dot(apple_mlp, shared_dir)
        shared_component = proj_coeff * shared_dir
        ortho_component = apple_mlp - shared_component
        
        modifications = {
            "remove_shared": ortho_component,       # 去掉shared对齐部分
            "remove_ortho": shared_component,        # 去掉正交分量
            "negate_shared": -shared_component + ortho_component,  # 反转shared
        }
        
        layer_result = {
            "proj_coeff": round(float(proj_coeff), 4),
            "shared_norm": round(float(np.linalg.norm(shared_component)), 4),
            "ortho_norm": round(float(np.linalg.norm(ortho_component)), 4),
            "ortho_ratio": round(float(np.linalg.norm(ortho_component) / (np.linalg.norm(apple_mlp) + 1e-8)), 4),
        }
        
        for mod_name, target_mlp_out in modifications.items():
            layer_dev = get_layer_device(layers, li)
            target_t = torch.tensor(target_mlp_out, dtype=torch.bfloat16, device=layer_dev)
            
            def make_mlp_replace_hook(target_out, _li=li):
                def hook(m, inp, out):
                    if isinstance(out, tuple):
                        new_out = out[0].clone()
                        # 在last_pos位置替换MLP输出
                        new_out[0, test_last_pos] = target_out.to(new_out.device).to(new_out.dtype)
                        return (new_out,) + out[1:]
                    return out
                return hook
            
            mlp_module = layers[li].mlp if hasattr(layers[li], 'mlp') else None
            if mlp_module is None:
                continue
            
            replace_hook = mlp_module.register_forward_hook(make_mlp_replace_hook(target_t, li))
            
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
                    "cat_delta": round(mod_cat - base_cat, 4) if mod_cat is not None and base_cat is not None else None,
                    "color_delta": round(mod_color - base_color, 4) if mod_color is not None and base_color is not None else None,
                }
                print(f"  L{li} {mod_name}: catΔ={((mod_cat or 0)-base_cat):+.3f}, "
                      f"colorΔ={((mod_color or 0)-base_color):+.3f}")
            else:
                layer_result[mod_name] = {"error": "forward failed"}
        
        results["projection_results"][f"L{li}"] = layer_result
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        periodic_log(f"  Exp3 L{li} done ({time.time()-t_step3:.1f}s)")
    
    # 汇总
    print(f"\n  === Exp3 Summary ===")
    print(f"  Layer | proj_coeff | shared_norm | ortho_norm | ortho_ratio | remove_shared→catΔ | remove_ortho→catΔ | negate_shared→catΔ")
    for layer_key in sorted(results["projection_results"].keys()):
        lr = results["projection_results"][layer_key]
        pc = lr.get("proj_coeff", 0)
        sn = lr.get("shared_norm", 0)
        on = lr.get("ortho_norm", 0)
        orr = lr.get("ortho_ratio", 0)
        rs = lr.get("remove_shared", {}).get("cat_delta", "N/A")
        ro = lr.get("remove_ortho", {}).get("cat_delta", "N/A")
        ns = lr.get("negate_shared", {}).get("cat_delta", "N/A")
        print(f"  {layer_key:5s} | {pc:+.4f}   | {sn:.4f}     | {on:.4f}    | {orr:.4f}     | {rs}                | {ro}               | {ns}")
    
    return results


# ===== 主函数 =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print(f"\n{'='*60}")
    print(f"Phase 450 v2: Inversion Gate Localization + Dual Channel + MLP Fix")
    print(f"Model: {model_name}, Round: {round_num}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # 加载模型
    t0_load = time.time()
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    print(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}, mlp_type={info.mlp_type}")
    print(f"  Load time: {time.time()-t0_load:.1f}s")
    
    if torch.cuda.is_available():
        print(f"  GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
              f"reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    
    all_results = {"model": model_name, "round": round_num,
                   "model_info": {"class": info.model_class, "n_layers": info.n_layers,
                                  "d_model": info.d_model, "mlp_type": info.mlp_type}}
    
    # 实验1: 组件路径消融
    t0 = time.time()
    try:
        r1 = exp1_component_path_ablation(model, tokenizer, info)
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
    
    print(f"\nPhase 450 v2 {model_name} R{round_num} complete!")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
