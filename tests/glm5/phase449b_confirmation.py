"""
Phase 449 R2: 确认测试 — 因果注入稳定性 + GLM4 L26负效应精确定位
=====================================================================
重点验证:
1. Shared/Private因果注入在不同beta下是否稳定
2. GLM4 L26的shared→cat负效应在更精细层间是否复现
3. 对象替换控制(GLM4的-2.31)在更多槽位下是否稳定

用法:
  python tests/glm5/phase449b_confirmation.py qwen3 2
  python tests/glm5/phase449b_confirmation.py glm4 2
  python tests/glm5/phase449b_confirmation.py deepseek7b 2
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
    ids = []
    for w in word_list:
        tok_ids = tokenizer.encode(w, add_special_tokens=False)
        if tok_ids:
            ids.append(tok_ids[0])
    if not ids:
        return None
    return float(np.mean(logits_np[ids]))


# ===== 确认1: 多beta因果注入 =====
def confirm_multi_beta_injection(model, tokenizer, info):
    """在不同beta下验证shared/private因果效应"""
    print(f"\n{'='*60}")
    print("Confirmation 1: Multi-Beta Causal Injection")
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
    
    # 收集对象deltas
    obj_names = ["apple", "orange", "banana", "grape", "lemon", "mango"]
    alpha = 1.0
    input_device = next(model.parameters()).device
    
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
        hooks_b = [layers[li].register_forward_hook(make_h_b(li)) for li in range(n_layers)]
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
    
    # 分解shared/private
    # 只在L0和最后一层做(节省时间)
    target_layers = [0, n_layers - 1]
    decomposition = {}
    for li in target_layers:
        deltas = {o: obj_layer_deltas[o].get(li) for o in obj_names if li in obj_layer_deltas.get(o, {})}
        valid = {o: d for o, d in deltas.items() if d is not None and np.linalg.norm(d) > 1e-8}
        if len(valid) < 3:
            continue
        delta_matrix = np.stack(list(valid.values()))
        shared = delta_matrix.mean(axis=0)
        private = {o: d - shared for o, d in valid.items()}
        decomposition[li] = {"shared": shared, "private": private}
    
    # 多beta注入
    betas = [0.5, 1.0, 2.0, 4.0]
    results = {}
    
    for li in target_layers:
        if li not in decomposition:
            continue
        shared_vec = decomposition[li]["shared"]
        private_apple = decomposition[li]["private"].get("apple", np.zeros(d_model))
        
        layer_results = {}
        for beta in betas:
            beta_results = {}
            for inj_name, base_vec in [("shared", shared_vec), ("private", private_apple)]:
                inj_vec = base_vec * beta
                text = "The apple is a"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attention_mask = inputs["attention_mask"].to(input_device)
                last_pos = input_ids.shape[1] - 1
                
                # 基准
                with torch.no_grad():
                    out_base = model(input_ids=input_ids, attention_mask=attention_mask)
                    base_logits = out_base.logits[0, -1].float().cpu().numpy()
                
                # 注入
                inj_tensor = torch.tensor(inj_vec, dtype=torch.bfloat16, device=input_device)
                def make_inject_hook(vec):
                    def hook(m, inp, out):
                        if isinstance(out, tuple):
                            new_out = out[0].clone()
                            new_out[0, last_pos] = new_out[0, last_pos] + vec.to(new_out.dtype)
                            return (new_out,) + out[1:]
                        return out
                    return hook
                
                inj_hook = layers[li].register_forward_hook(make_inject_hook(inj_tensor))
                with torch.no_grad():
                    try:
                        out_inj = model(input_ids=input_ids, attention_mask=attention_mask)
                        inj_logits = out_inj.logits[0, -1].float().cpu().numpy()
                    except:
                        inj_logits = None
                inj_hook.remove()
                
                if inj_logits is not None:
                    cat_base = get_logit_for_words(base_logits, tokenizer, cat_words)
                    cat_inj = get_logit_for_words(inj_logits, tokenizer, cat_words)
                    beta_results[inj_name] = {
                        "cat_delta": round(cat_inj - cat_base, 4) if cat_base is not None else None,
                    }
            
            layer_results[f"beta_{beta}"] = beta_results
        
        results[f"L{li}"] = layer_results
        print(f"  L{li}: " + " | ".join(
            f"b={b}: shared→cat={results[f'L{li}'][f'beta_{b}'].get('shared',{}).get('cat_delta',0):+.3f}, "
            f"private→cat={results[f'L{li}'][f'beta_{b}'].get('private',{}).get('cat_delta',0):+.3f}"
            for b in betas))
    
    return results


# ===== 确认2: GLM4 L26区域精细扫描 =====
def confirm_glm4_negative_shared(model, tokenizer, info):
    """在GLM4 L22-L30区间精细扫描shared→cat效应"""
    print(f"\n{'='*60}")
    print("Confirmation 2: Fine-Grained Shared Effect Scan (around L26)")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    d_model = info.d_model
    
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
    
    obj_names = ["apple", "orange", "banana", "grape", "lemon", "mango"]
    alpha = 1.0
    input_device = next(model.parameters()).device
    inject_beta = 2.0
    
    # 收集对象deltas
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
        hooks_b = [layers[li].register_forward_hook(make_h_b(li)) for li in range(n_layers)]
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
    
    # 对所有模型: 扫描L0到L_last每5层
    # 对GLM4: 额外精细扫描L22-L30
    scan_layers = list(range(0, n_layers, 5))
    if n_layers > 25:
        scan_layers += list(range(max(0, n_layers//2-5), min(n_layers, n_layers//2+5)))
    scan_layers = sorted(set([l for l in scan_layers if l < n_layers]))
    
    results = {}
    
    for li in scan_layers:
        deltas = {o: obj_layer_deltas[o].get(li) for o in obj_names if li in obj_layer_deltas.get(o, {})}
        valid = {o: d for o, d in deltas.items() if d is not None and np.linalg.norm(d) > 1e-8}
        if len(valid) < 3:
            continue
        delta_matrix = np.stack(list(valid.values()))
        shared = delta_matrix.mean(axis=0)
        
        # 注入shared
        inj_vec = shared * inject_beta
        text = "The apple is a"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        last_pos = input_ids.shape[1] - 1
        
        with torch.no_grad():
            out_base = model(input_ids=input_ids, attention_mask=attention_mask)
            base_logits = out_base.logits[0, -1].float().cpu().numpy()
        
        inj_tensor = torch.tensor(inj_vec, dtype=torch.bfloat16, device=input_device)
        def make_inject_hook(vec):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    new_out = out[0].clone()
                    new_out[0, last_pos] = new_out[0, last_pos] + vec.to(new_out.dtype)
                    return (new_out,) + out[1:]
                return out
            return hook
        
        inj_hook = layers[li].register_forward_hook(make_inject_hook(inj_tensor))
        with torch.no_grad():
            try:
                out_inj = model(input_ids=input_ids, attention_mask=attention_mask)
                inj_logits = out_inj.logits[0, -1].float().cpu().numpy()
            except:
                inj_logits = None
        inj_hook.remove()
        
        if inj_logits is not None:
            cat_base = get_logit_for_words(base_logits, tokenizer, cat_words)
            cat_inj = get_logit_for_words(inj_logits, tokenizer, cat_words)
            color_base = get_logit_for_words(base_logits, tokenizer, ["red", "green", "yellow"])
            color_inj = get_logit_for_words(inj_logits, tokenizer, ["red", "green", "yellow"])
            
            results[f"L{li}"] = {
                "shared_cat_delta": round(cat_inj - cat_base, 4) if cat_base is not None else None,
                "shared_color_delta": round(color_inj - color_base, 4) if color_base is not None else None,
            }
            print(f"  L{li}: shared→cat={results[f'L{li}']['shared_cat_delta']:+.3f}, "
                  f"shared→color={results[f'L{li}']['shared_color_delta']:+.3f}")
    
    # 找反转点
    neg_layers = [k for k, v in results.items() if v.get('shared_cat_delta', 0) is not None and v['shared_cat_delta'] < 0]
    if neg_layers:
        print(f"\n  *** NEGATIVE shared→cat at layers: {neg_layers} ***")
    
    return results


# ===== 确认3: 多槽位对象替换控制 =====
def confirm_multi_slot_replace(model, tokenizer, info):
    """在has/feels/color等多槽位下验证对象替换控制"""
    print(f"\n{'='*60}")
    print("Confirmation 3: Multi-Slot Object Replace Control")
    print(f"{'='*60}")
    
    test_cases = [
        ("apple", "is_a", "A thing is a kind of", "The apple is a kind of",
         "Although the apple is described as an animal, it is a kind of",
         "Although the apple is described as an animal, something is a kind of"),
        ("apple", "has", "A thing has", "The apple has",
         "Although the apple is described as an animal, it has",
         "Although the apple is described as an animal, something has"),
        ("apple", "feels", "A thing feels", "The apple feels",
         "Although the apple is described as an animal, it feels",
         "Although the apple is described as an animal, something feels"),
        ("dog", "is_a", "A thing is a kind of", "The dog is a kind of",
         "Although the dog is described as a fruit, it is a kind of",
         "Although the dog is described as a fruit, something is a kind of"),
        ("dog", "has", "A thing has", "The dog has",
         "Although the dog is described as a fruit, it has",
         "Although the dog is described as a fruit, something has"),
        ("knife", "is_a", "A thing is a kind of", "The knife is a kind of",
         "Although the knife is described as a fruit, it is a kind of",
         "Although the knife is described as a fruit, something is a kind of"),
    ]
    
    attr_words = {
        "apple_is_a": ["fruit", "food", "produce"],
        "apple_has": ["seed", "skin", "core", "stem"],
        "apple_feels": ["smooth", "round", "firm"],
        "dog_is_a": ["animal", "pet", "mammal"],
        "dog_has": ["leg", "tail", "fur", "ear"],
        "knife_is_a": ["tool", "weapon", "instrument"],
    }
    
    results = {}
    
    for obj_name, slot, T0, T1, T3, T5 in test_cases:
        key = f"{obj_name}_{slot}"
        attr_key = key
        words = attr_words.get(attr_key, [])
        if not words:
            continue
        
        logits = {}
        for t_name, t_text in [("T0", T0), ("T1", T1), ("T3", T3), ("T5", T5)]:
            input_device = next(model.parameters()).device
            inputs = tokenizer(t_text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits[t_name] = out.logits[0, -1].float().cpu().numpy()
        
        vals = {}
        for t_name in ["T0", "T1", "T3", "T5"]:
            vals[t_name] = get_logit_for_words(logits[t_name], tokenizer, words)
        
        unlock = (vals["T1"] or 0) - (vals["T0"] or 0)
        conflict_recovery = (vals["T3"] or 0) - (vals["T1"] or 0)
        replace_ctrl = (vals["T5"] or 0) - (vals["T3"] or 0)
        
        results[key] = {
            "unlock": round(unlock, 4),
            "conflict_recovery": round(conflict_recovery, 4),
            "replace_ctrl": round(replace_ctrl, 4),
        }
        print(f"  {key}: unlock={unlock:+.2f}, conflict={conflict_recovery:+.2f}, replace={replace_ctrl:+.2f}")
    
    # 汇总
    avg_replace = np.mean([v["replace_ctrl"] for v in results.values()])
    avg_unlock = np.mean([v["unlock"] for v in results.values()])
    print(f"\n  Avg: unlock={avg_unlock:+.2f}, replace_ctrl={avg_replace:+.2f}")
    
    return results


# ===== 主函数 =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    print(f"Phase 449 R2: Confirmation — Multi-Beta + Fine-Grained + Multi-Slot")
    print(f"Model: {model_name}, Round: {round_num}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    print(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    all_results = {}
    
    # 确认1
    try:
        r1 = confirm_multi_beta_injection(model, tokenizer, info)
        all_results["confirm1_multi_beta"] = r1
    except Exception as e:
        print(f"Confirm1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["confirm1_multi_beta"] = {"error": str(e)}
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 确认2
    try:
        r2 = confirm_glm4_negative_shared(model, tokenizer, info)
        all_results["confirm2_fine_grained"] = r2
    except Exception as e:
        print(f"Confirm2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["confirm2_fine_grained"] = {"error": str(e)}
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 确认3
    try:
        r3 = confirm_multi_slot_replace(model, tokenizer, info)
        all_results["confirm3_multi_slot"] = r3
    except Exception as e:
        print(f"Confirm3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["confirm3_multi_slot"] = {"error": str(e)}
    
    # 保存
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase449_{model_name}_r{round_num}.json"
    
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
    
    release_model(model)
    print(f"\nPhase 449 R2 {model_name} complete!")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
