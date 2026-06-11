"""
Phase 450 v3: Inversion Gate Localization + Shared/Private Dual Channel + MLP Internal Fix
============================================================================================
v3 fix: device_map="auto"下深层注入问题
- 注入tensor不指定device(默认CPU), 在hook内部通过vec.to(out.device)动态转移
- input_device从embed_tokens直接获取
- 确保GLM4/DS7B深层不缺失

用法:
  python tests/glm5/phase450_inversion_gate_v3.py qwen3 1
  python tests/glm5/phase450_inversion_gate_v3.py glm4 1
  python tests/glm5/phase450_inversion_gate_v3.py deepseek7b 1
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
log = logging.getLogger("p450")

_last_log = [time.time()]
def plog(msg, interval=30):
    now = time.time()
    if now - _last_log[0] >= interval or any(k in msg.lower() for k in ['complete','step','failed','summary']):
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
    
    # 显示层分配
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
    """从embed_tokens获取输入设备"""
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


def get_logit_for_words(logits_np, tokenizer, word_list):
    ids = []
    for w in word_list:
        tok_ids = tokenizer.encode(w, add_special_tokens=False)
        if tok_ids:
            ids.append(tok_ids[0])
    if not ids:
        return None
    return float(np.mean(logits_np[ids]))


def get_embedding_weight(model):
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens.weight.detach().float()
    return model.get_input_embeddings().weight.detach().float()


# ===== 关键fix: 注入tensor不指定device, 在hook内部转移 =====
def make_safe_inject_hook(vec_np, last_pos, beta=1.0):
    """创建安全的注入hook - vec_np是numpy数组, 在hook内部动态处理设备"""
    def hook(m, inp, out):
        if isinstance(out, tuple):
            new_out = out[0].clone()
            target_device = new_out.device
            target_dtype = new_out.dtype
            # 动态创建tensor并转移到正确设备
            inj_t = torch.tensor(vec_np * beta, dtype=torch.float32)
            inj_t = inj_t.to(device=target_device, dtype=target_dtype)
            new_out[0, last_pos] = new_out[0, last_pos] + inj_t
            return (new_out,) + out[1:]
        return out
    return hook


# ===== 实验1: 组件路径消融定位反转门控 =====
def exp1_component_path_ablation(model, tokenizer, info):
    """
    在GLM4关键层(L19-L30)分别消融attn和MLP路径
    消融后注入shared分量,测shared→cat效应
    """
    print(f"\n{'='*60}")
    print("Exp1: Component Path Ablation for Inversion Gate")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    input_device = get_input_device(model)
    
    cat_words = ["fruit", "food", "produce"]
    opp_words = ["animal", "dog", "cat"]
    W_E = get_embedding_weight(model)
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    cat_dir = (W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)).cpu()
    cat_dir = cat_dir / (cat_dir.norm() + 1e-8)
    
    obj_names = ["apple", "orange", "banana", "grape", "lemon"]
    alpha = 1.0
    
    # 采样层
    if n_layers >= 30:
        mid = n_layers * 3 // 5
        scan_layers = sorted(set(
            [0, n_layers//4, n_layers//2] +
            list(range(max(0, mid-5), min(n_layers, mid+8))) +
            [n_layers-2, n_layers-1]
        ))
    else:
        scan_layers = list(range(n_layers))
    scan_layers = [l for l in scan_layers if l < n_layers]
    print(f"  Scan layers: {scan_layers}")
    
    # Step 1: 收集各层deltas
    print("  Step 1: Collecting per-object deltas...")
    t1 = time.time()
    
    obj_base_h = {}
    obj_pert_h = {}
    
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
        
        # 扰动(embedding注入)
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
            print(f"  Warning: perturb failed for {obj_name}: {e}")
        finally:
            if embed_hook:
                embed_hook.remove()
            for h in hooks_p:
                h.remove()
        
        obj_pert_h[obj_name] = {li: pert_h[li].numpy() for li in pert_h if li in scan_layers}
        plog(f"  Step1: {obj_name} done ({time.time()-t1:.1f}s)")
    
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
    
    print(f"  Step1 complete: {len(decomposition)} layers ({time.time()-t1:.1f}s)")
    
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
    base_color = get_logit_for_words(base_logits, tokenizer, ["red", "green", "yellow"])
    print(f"  Base: cat={base_cat:.2f}, color={base_color:.2f}")
    
    # Step 3a: 基准shared→cat(无消融)
    print("  Step 3a: Baseline shared→cat...")
    ablation_layers = sorted(decomposition.keys())
    if len(ablation_layers) > 12:
        critical = [l for l in ablation_layers if n_layers*3//5-5 <= l <= n_layers*3//5+8]
        rest = [l for l in ablation_layers if l not in critical]
        step = max(1, len(rest) // 6)
        selected = sorted(set([rest[0]] + rest[::step] + critical + [rest[-1]]))[:12]
        ablation_layers = selected
    print(f"  Ablation layers: {ablation_layers}")
    
    baseline_shared_inj = {}
    for li in ablation_layers:
        if li not in decomposition:
            continue
        shared_vec = decomposition[li]["shared"]
        
        inj_hook = layers[li].register_forward_hook(
            make_safe_inject_hook(shared_vec, test_last_pos, beta=2.0))
        
        with torch.no_grad():
            try:
                out_inj = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                inj_logits = out_inj.logits[0, -1].float().cpu().numpy()
                inj_cat = get_logit_for_words(inj_logits, tokenizer, cat_words)
                shared_cat_effect = inj_cat - base_cat if inj_cat is not None and base_cat is not None else None
                baseline_shared_inj[f"L{li}"] = round(shared_cat_effect, 4) if shared_cat_effect is not None else None
                print(f"  L{li} shared→cat: {shared_cat_effect:+.3f}")
            except Exception as e:
                print(f"  L{li} shared inject FAILED: {e}")
                baseline_shared_inj[f"L{li}"] = None
        
        inj_hook.remove()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Step 3b: 消融attn/MLP后测shared→cat
    print("  Step 3b: Component ablation + shared injection...")
    t3 = time.time()
    results = {"ablation_results": {}, "base_logits": {"cat": round(base_cat, 4), "color": round(base_color, 4)},
               "baseline_shared_injection": baseline_shared_inj}
    
    for idx, abl_li in enumerate(ablation_layers):
        if abl_li not in decomposition:
            continue
        
        layer = layers[abl_li]
        layer_result = {}
        shared_vec = decomposition[abl_li]["shared"]
        
        attn_module = getattr(layer, 'self_attn', None)
        mlp_module = getattr(layer, 'mlp', None)
        
        # 消融(无注入)
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
                    abl_cat = get_logit_for_words(abl_logits, tokenizer, cat_words)
                    abl_color = get_logit_for_words(abl_logits, tokenizer, ["red", "green", "yellow"])
                    layer_result[abl_type] = {
                        "cat_delta": round(abl_cat - base_cat, 4) if abl_cat is not None and base_cat is not None else None,
                        "color_delta": round(abl_color - base_color, 4) if abl_color is not None and base_color is not None else None,
                    }
                    print(f"  L{abl_li} {abl_type}: catΔ={((abl_cat or 0)-base_cat):+.3f}")
                except Exception as e:
                    print(f"  L{abl_li} {abl_type}: FAILED - {e}")
                    layer_result[abl_type] = {"error": str(e)}
            abl_hook.remove()
        
        # 消融 + shared注入
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
            inj_hook = layers[abl_li].register_forward_hook(
                make_safe_inject_hook(shared_vec, test_last_pos, beta=2.0))
            
            with torch.no_grad():
                try:
                    out_inj_abl = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                    inj_abl_logits = out_inj_abl.logits[0, -1].float().cpu().numpy()
                    inj_abl_cat = get_logit_for_words(inj_abl_logits, tokenizer, cat_words)
                    abl_base_cat = layer_result.get(abl_type, {}).get("cat_delta")
                    if abl_base_cat is not None:
                        abl_base_cat = base_cat + abl_base_cat
                    else:
                        abl_base_cat = base_cat
                    shared_cat_effect = inj_abl_cat - abl_base_cat if inj_abl_cat is not None else None
                    layer_result[f"{abl_type}_shared_inj_cat"] = round(shared_cat_effect, 4) if shared_cat_effect is not None else None
                    print(f"  L{abl_li} {abl_type}+shared: shared→cat={shared_cat_effect:+.3f}")
                except Exception as e:
                    print(f"  L{abl_li} {abl_type}+shared: FAILED - {e}")
                    layer_result[f"{abl_type}_shared_inj_cat"] = None
            
            abl_hook.remove()
            inj_hook.remove()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        results["ablation_results"][f"L{abl_li}"] = layer_result
        plog(f"  Exp1 L{abl_li} ({idx+1}/{len(ablation_layers)}, {time.time()-t3:.1f}s)")
    
    # 汇总
    print(f"\n  === Exp1 Summary ===")
    print(f"  Layer | shared→cat(clean) | attn→catΔ | mlp→catΔ | +zero_attn→shared | +zero_mlp→shared")
    for lk in sorted(results["ablation_results"].keys()):
        lr = results["ablation_results"][lk]
        sc = baseline_shared_inj.get(lk, "N/A")
        ad = lr.get("zero_attn", {}).get("cat_delta", "N/A")
        md = lr.get("zero_mlp", {}).get("cat_delta", "N/A")
        sa = lr.get("zero_attn_shared_inj_cat", "N/A")
        sm = lr.get("zero_mlp_shared_inj_cat", "N/A")
        print(f"  {lk:5s} | {sc}              | {ad}    | {md}    | {sa}                  | {sm}")
    
    return results


# ===== 实验2: Shared/Private双通道功能分离 =====
def exp2_dual_channel_functional(model, tokenizer, info):
    """
    验证shared通道主要控制类别泛化, private通道主要控制对象特异属性
    """
    print(f"\n{'='*60}")
    print("Exp2: Shared/Private Dual Channel Functional Separation")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    W_E = get_embedding_weight(model)
    input_device = get_input_device(model)
    
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
    
    target_layers = [0, n_layers//6, n_layers//3, n_layers//2, 2*n_layers//3, 5*n_layers//6, n_layers-2, n_layers-1]
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    all_results = {}
    
    for cat_name, cat_cfg in CATEGORY_CONFIGS.items():
        print(f"\n  --- Category: {cat_name} ---")
        t_cat = time.time()
        
        obj_names = cat_cfg["objects"]
        opp_words = cat_cfg["opp_words"]
        cat_words = cat_cfg["category_words"]
        
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
        
        # 分解
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
        test_input_ids = test_inputs["input_ids"].to(input_device)
        test_attention_mask = test_inputs["attention_mask"].to(input_device)
        test_last_pos = test_input_ids.shape[1] - 1
        
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
                inj_hook = layers[li].register_forward_hook(
                    make_safe_inject_hook(inj_vec, test_last_pos, beta=1.0))
                
                with torch.no_grad():
                    try:
                        out_inj = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                        inj_logits = out_inj.logits[0, -1].float().cpu().numpy()
                    except:
                        inj_logits = None
                inj_hook.remove()
                
                if inj_logits is not None:
                    deltas_out = {}
                    deltas_out["category"] = get_logit_for_words(inj_logits, tokenizer, cat_words)
                    for attr_name, attr_words in cat_cfg["attrs"].items():
                        deltas_out[attr_name] = get_logit_for_words(inj_logits, tokenizer, attr_words)
                    
                    delta_dict = {}
                    for k in deltas_out:
                        if deltas_out[k] is not None and base_attrs.get(k) is not None:
                            delta_dict[k] = round(deltas_out[k] - base_attrs[k], 4)
                    layer_result[inj_name] = delta_dict
                else:
                    layer_result[inj_name] = {"error": "forward failed"}
            
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
        print(f"  {cat_name} done ({time.time()-t_cat:.1f}s)")
    
    # 汇总
    print(f"\n  === Exp2 Summary ===")
    for cat_name in CATEGORY_CONFIGS:
        cr = all_results.get(cat_name, {})
        print(f"\n  --- {cat_name} ---")
        print(f"  Layer | shared→cat | private→cat | S+P→cat | priv/cat ratio")
        for lk in sorted(cr.keys()):
            lr = cr[lk]
            s = lr.get("shared", {})
            p = lr.get("private", {})
            sp = lr.get("shared+private", {})
            sc = s.get("category", 0)
            pc = p.get("category", 0)
            spc = sp.get("category", 0)
            ratio = abs(pc) / (abs(sc) + abs(pc) + 1e-6)
            print(f"  {lk:5s} | {sc:+.3f}     | {pc:+.3f}      | {spc:+.3f}  | {ratio:.3f}")
    
    return all_results


# ===== 实验3: MLP输出级消融(修复版) =====
def exp3_mlp_output_ablation(model, tokenizer, info):
    """
    修复版: 在MLP输出级别分解shared/ortho, 替换MLP输出测效果
    """
    print(f"\n{'='*60}")
    print("Exp3: MLP Output-Level Ablation (Fixed)")
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
    
    target_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    # Step 1: 收集MLP输出
    print("  Step 1: Capturing MLP outputs...")
    t1 = time.time()
    
    mlp_outputs = {}
    for obj_name in obj_names:
        text = f"The {obj_name} is a"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        last_pos = input_ids.shape[1] - 1
        
        captured = {}
        def make_capture_hook(li):
            def hook(m, inp, out):
                # MLP输出是单tensor, Layer输出是tuple
                if isinstance(out, tuple):
                    captured[li] = out[0][0, last_pos].detach().float().cpu().numpy()
                else:
                    captured[li] = out[0, last_pos].detach().float().cpu().numpy()
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
        plog(f"  MLP: {obj_name} captured {len(mlp_outputs[obj_name])} layers")
    
    print(f"  Step1 done ({time.time()-t1:.1f}s)")
    
    # Step 2: 分解shared方向
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
    
    print(f"  Shared dirs computed for {len(layer_shared_dirs)} layers: {list(layer_shared_dirs.keys())}")
    
    # Step 3: MLP输出替换
    print("  Step 3: MLP output projection ablation...")
    t3 = time.time()
    
    results = {"target_layers": target_layers, "projection_results": {}}
    
    test_text = "The apple is a"
    test_inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=64)
    test_input_ids = test_inputs["input_ids"].to(input_device)
    test_attention_mask = test_inputs["attention_mask"].to(input_device)
    test_last_pos = test_input_ids.shape[1] - 1
    
    with torch.no_grad():
        out_base = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
        base_logits = out_base.logits[0, -1].float().cpu().numpy()
    base_cat = get_logit_for_words(base_logits, tokenizer, cat_words)
    base_color = get_logit_for_words(base_logits, tokenizer, ["red", "green", "yellow"])
    
    for li in target_layers:
        if li not in layer_shared_dirs:
            print(f"  L{li}: no shared dir, skipping")
            continue
        
        shared_dir = layer_shared_dirs[li]
        apple_mlp = mlp_outputs.get("apple", {}).get(li)
        if apple_mlp is None:
            print(f"  L{li}: no apple MLP output, skipping")
            continue
        
        proj_coeff = np.dot(apple_mlp, shared_dir)
        shared_component = proj_coeff * shared_dir
        ortho_component = apple_mlp - shared_component
        
        modifications = {
            "remove_shared": ortho_component,
            "remove_ortho": shared_component,
            "negate_shared": -shared_component + ortho_component,
        }
        
        layer_result = {
            "proj_coeff": round(float(proj_coeff), 4),
            "shared_norm": round(float(np.linalg.norm(shared_component)), 4),
            "ortho_norm": round(float(np.linalg.norm(ortho_component)), 4),
        }
        
        for mod_name, target_mlp_out in modifications.items():
            # 用hook替换MLP输出
            def make_mlp_replace_hook(target_out_np, _li=li, _lp=test_last_pos):
                def hook(m, inp, out):
                    t = torch.tensor(target_out_np, dtype=torch.float32)
                    if isinstance(out, tuple):
                        new_out = out[0].clone()
                        t = t.to(device=new_out.device, dtype=new_out.dtype)
                        new_out[0, _lp] = t
                        return (new_out,) + out[1:]
                    else:
                        # MLP输出是单tensor [batch, seq, d_model]
                        new_out = out.clone()
                        t = t.to(device=new_out.device, dtype=new_out.dtype)
                        new_out[0, _lp] = t
                        return new_out
                return hook
            
            mlp_module = layers[li].mlp if hasattr(layers[li], 'mlp') else None
            if mlp_module is None:
                continue
            
            replace_hook = mlp_module.register_forward_hook(make_mlp_replace_hook(target_mlp_out, li, test_last_pos))
            
            with torch.no_grad():
                try:
                    out_mod = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
                    mod_logits = out_mod.logits[0, -1].float().cpu().numpy()
                    mod_cat = get_logit_for_words(mod_logits, tokenizer, cat_words)
                    mod_color = get_logit_for_words(mod_logits, tokenizer, ["red", "green", "yellow"])
                    
                    layer_result[mod_name] = {
                        "cat_delta": round(mod_cat - base_cat, 4) if mod_cat is not None and base_cat is not None else None,
                        "color_delta": round(mod_color - base_color, 4) if mod_color is not None and base_color is not None else None,
                    }
                    print(f"  L{li} {mod_name}: catΔ={((mod_cat or 0)-base_cat):+.3f}")
                except Exception as e:
                    print(f"  L{li} {mod_name}: FAILED - {e}")
                    layer_result[mod_name] = {"error": str(e)}
            
            replace_hook.remove()
        
        results["projection_results"][f"L{li}"] = layer_result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        plog(f"  Exp3 L{li} done ({time.time()-t3:.1f}s)")
    
    # 汇总
    print(f"\n  === Exp3 Summary ===")
    print(f"  Layer | proj_coeff | shared_norm | ortho_norm | remove_shared→catΔ | remove_ortho→catΔ | negate_shared→catΔ")
    for lk in sorted(results["projection_results"].keys()):
        lr = results["projection_results"][lk]
        pc = lr.get("proj_coeff", 0)
        sn = lr.get("shared_norm", 0)
        on = lr.get("ortho_norm", 0)
        rs = lr.get("remove_shared", {}).get("cat_delta", "N/A")
        ro = lr.get("remove_ortho", {}).get("cat_delta", "N/A")
        ns = lr.get("negate_shared", {}).get("cat_delta", "N/A")
        print(f"  {lk:5s} | {pc:+.4f}   | {sn:.4f}     | {on:.4f}    | {rs}                | {ro}               | {ns}")
    
    return results


# ===== 主函数 =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print(f"\n{'='*60}")
    print(f"Phase 450 v3: Inversion Gate + Dual Channel + MLP Fix")
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
        r1 = exp1_component_path_ablation(model, tokenizer, info)
        all_results["exp1"] = r1
    except Exception as e:
        print(f"Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1"] = {"error": str(e)}
    print(f"  Exp1: {time.time()-t0:.1f}s")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Exp2
    t0 = time.time()
    try:
        r2 = exp2_dual_channel_functional(model, tokenizer, info)
        all_results["exp2"] = r2
    except Exception as e:
        print(f"Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2"] = {"error": str(e)}
    print(f"  Exp2: {time.time()-t0:.1f}s")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Exp3
    t0 = time.time()
    try:
        r3 = exp3_mlp_output_ablation(model, tokenizer, info)
        all_results["exp3"] = r3
    except Exception as e:
        print(f"Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3"] = {"error": str(e)}
    print(f"  Exp3: {time.time()-t0:.1f}s")
    
    # 保存
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase450_{model_name}_r{round_num}.json"
    
    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(x) for x in obj]
        return obj
    
    all_results = convert(all_results)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
    
    release_model(model)
    print(f"\nPhase 450 v3 {model_name} R{round_num} complete!")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
