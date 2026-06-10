"""
Phase 448 R2: 确认测试 — MLP/Attn私有化角色 + GLM4负先验验证
================================================================
重点验证:
1. MLP/Attn在私有化中的角色是否在更多层上稳定
2. GLM4的负先验+高冲突恢复力是否稳定
3. 更多对象验证SlotMediation拆分

用法:
  python tests/glm5/phase448b_confirmation.py qwen3 2
  python tests/glm5/phase448b_confirmation.py glm4 2
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import os, gc, time, json
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS)

CATEGORY_OBJECTS = {
    "fruit": {
        "objects": ["apple", "orange", "banana", "grape", "lemon", "mango"],
        "cat_words": ["fruit", "apple", "orange", "banana"],
        "opp_words": ["animal", "dog", "cat", "horse"],
    },
    "animal": {
        "objects": ["dog", "cat", "horse", "lion", "tiger", "eagle"],
        "cat_words": ["animal", "dog", "cat", "horse"],
        "opp_words": ["fruit", "apple", "orange", "banana"],
    },
    "tool": {
        "objects": ["knife", "hammer", "scissors", "axe", "drill", "saw"],
        "cat_words": ["tool", "knife", "hammer", "scissors"],
        "opp_words": ["vehicle", "car", "bus", "train"],
    },
}

SLOT_ATTRS = {
    "apple": {"color": ["red", "green", "yellow"], "taste": ["sweet", "sour", "juicy"], "part": ["seed", "skin", "core", "stem"], "category": ["fruit", "food", "produce"], "non_category": ["animal", "tool", "vehicle"], "material": ["organic", "fresh", "natural"], "random": ["square", "loud", "electric", "digital"]},
    "dog": {"color": ["brown", "black", "white"], "part": ["leg", "tail", "fur", "ear"], "category": ["animal", "pet", "mammal"], "non_category": ["fruit", "tool", "vehicle"], "random": ["square", "sweet", "metallic", "digital"]},
    "knife": {"color": ["silver", "gray", "metallic"], "part": ["blade", "handle", "edge", "tip"], "category": ["tool", "weapon", "instrument"], "non_category": ["fruit", "animal", "vehicle"], "material": ["metal", "steel", "iron"], "random": ["sweet", "furry", "organic", "digital"]},
}


def load_model_auto(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["path"], torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, local_files_only=True, attn_implementation="sdpa")
    model.eval()
    return model, tokenizer


def get_cat_direction(model, tokenizer, cat_words, opp_words):
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        W_E = model.model.embed_tokens.weight.detach().float()
    elif hasattr(model, 'get_input_embeddings'):
        W_E = model.get_input_embeddings().weight.detach().float()
    else:
        return None
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    d = W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)
    d = d.cpu()
    d = d / (d.norm() + 1e-8)
    return d


def get_logit_values(logits, tokenizer, word_list):
    ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in word_list if tokenizer.encode(w, add_special_tokens=False)]
    if not ids:
        return None
    return float(logits[ids].mean())


def compute_binding_stats(layer_deltas_dict, cat_dir_np):
    layer_deltas = []
    valid_objs = []
    for obj_name, d in layer_deltas_dict.items():
        if np.linalg.norm(d) > 1e-8:
            layer_deltas.append(d)
            valid_objs.append(obj_name)
    if len(layer_deltas) < 3:
        return None
    delta_matrix = np.stack(layer_deltas)
    shared_direction = delta_matrix.mean(axis=0)
    residuals = delta_matrix - shared_direction
    total_var = np.sum(delta_matrix ** 2)
    shared_var = np.sum(shared_direction ** 2) * len(layer_deltas)
    shared_ratio = shared_var / total_var if total_var > 1e-10 else 0
    
    pair_cosines = []
    for i in range(len(layer_deltas)):
        for j in range(i+1, len(layer_deltas)):
            ni = np.linalg.norm(layer_deltas[i])
            nj = np.linalg.norm(layer_deltas[j])
            if ni > 1e-8 and nj > 1e-8:
                cos_ij = float(np.dot(layer_deltas[i], layer_deltas[j]) / (ni * nj))
                pair_cosines.append(cos_ij)
    avg_pair_cos = float(np.mean(pair_cosines)) if pair_cosines else 0
    return {"shared_ratio": round(float(shared_ratio), 4), "avg_pair_cosine": round(float(avg_pair_cos), 4), "n_objects": len(valid_objs)}


# ===== 确认1: 精确MLP/Attn消融 — 每层都做 =====
def confirm_privatization_dynamics(model, tokenizer, info, cat_directions):
    """逐层MLP/Attn消融,精确定位私有化转折点"""
    print(f"\n{'='*60}")
    print("Confirmation 1: Layer-by-Layer MLP/Attn Ablation for Privatization")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    layers = get_layers(model)
    alpha = 1.0
    
    results = {}
    
    # 只测fruit类别(最稳定的)
    cat_name = "fruit"
    if cat_name not in cat_directions:
        print("No fruit direction available")
        return results
    
    cat_dir = cat_directions[cat_name]
    obj_names = CATEGORY_OBJECTS[cat_name]["objects"]
    
    # 先收集完整模型的逐层delta
    print("  Collecting full model deltas...")
    obj_deltas_full = {}
    for obj_name in obj_names:
        text = f"The {obj_name} is a"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        last_pos = input_ids.shape[1] - 1
        
        # 基准
        base_hiddens = {}
        def make_hook_base(li):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    base_hiddens[li] = out[0][0, last_pos].detach().float().cpu()
                elif out.dim() == 3:
                    base_hiddens[li] = out[0, last_pos].detach().float().cpu()
            return hook
        hooks = [layers[li].register_forward_hook(make_hook_base(li)) for li in range(n_layers)]
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        for h in hooks:
            h.remove()
        
        # 扰动
        perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
        pert_hiddens = {}
        def make_hook_pert(li):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    pert_hiddens[li] = out[0][0, last_pos].detach().float().cpu()
                elif out.dim() == 3:
                    pert_hiddens[li] = out[0, last_pos].detach().float().cpu()
            return hook
        hooks2 = [layers[li].register_forward_hook(make_hook_pert(li)) for li in range(n_layers)]
        
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
            for h in hooks2:
                h.remove()
        
        deltas = {}
        for li in range(n_layers):
            if li in base_hiddens and li in pert_hiddens:
                deltas[li] = (pert_hiddens[li] - base_hiddens[li]).numpy()
        obj_deltas_full[obj_name] = deltas
    
    # 完整模型的shared_ratio
    full_shared = {}
    for li in range(n_layers):
        layer_deltas = {o: d[li] for o, d in obj_deltas_full.items() if li in d}
        stats = compute_binding_stats(layer_deltas, cat_dir.numpy() if isinstance(cat_dir, torch.Tensor) else cat_dir)
        if stats:
            full_shared[li] = stats["shared_ratio"]
    
    print(f"  Full model shared_ratio: L0={full_shared.get(0,0):.3f}, L_mid={full_shared.get(n_layers//2,0):.3f}, L_last={full_shared.get(n_layers-1,0):.3f}")
    
    # 逐层消融 — 只测关键层: L0, L1, L2, L_mid-2, L_mid-1, L_mid, L_mid+1, L_mid+2, L_last-2, L_last-1
    abl_layers = [0, 1, 2, n_layers//4, n_layers//3, n_layers//2, 2*n_layers//3, 3*n_layers//4, n_layers-2, n_layers-1]
    abl_layers = sorted(set([l for l in abl_layers if l < n_layers]))
    
    mlp_effects = {}
    attn_effects = {}
    
    for abl_layer in abl_layers:
        print(f"  Ablating Layer {abl_layer}...")
        
        for abl_type in ["mlp", "attn"]:
            # 收集消融后delta
            obj_deltas_abl = {}
            
            for obj_name in obj_names:
                text = f"The {obj_name} is a"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                input_device = next(model.parameters()).device
                input_ids = inputs["input_ids"].to(input_device)
                attention_mask = inputs["attention_mask"].to(input_device)
                last_pos = input_ids.shape[1] - 1
                
                # 基准 + 消融
                base_h = {}
                def make_h_b(li):
                    def hook(m, inp, out):
                        if isinstance(out, tuple):
                            base_h[li] = out[0][0, last_pos].detach().float().cpu()
                        elif out.dim() == 3:
                            base_h[li] = out[0, last_pos].detach().float().cpu()
                    return hook
                
                hooks_base = [layers[li].register_forward_hook(make_h_b(li)) for li in range(n_layers)]
                
                def make_abl_hook():
                    def hook(m, inp, out):
                        if isinstance(out, tuple):
                            return (torch.zeros_like(out[0]),) + out[1:]
                        return torch.zeros_like(out)
                    return hook
                
                if abl_type == "mlp" and hasattr(layers[abl_layer], 'mlp'):
                    abl_h = layers[abl_layer].mlp.register_forward_hook(make_abl_hook())
                elif abl_type == "attn" and hasattr(layers[abl_layer], 'self_attn'):
                    abl_h = layers[abl_layer].self_attn.register_forward_hook(make_abl_hook())
                else:
                    abl_h = None
                
                with torch.no_grad():
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
                if abl_h:
                    abl_h.remove()
                for h in hooks_base:
                    h.remove()
                
                # 扰动 + 消融
                perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
                pert_h = {}
                def make_h_p(li):
                    def hook(m, inp, out):
                        if isinstance(out, tuple):
                            pert_h[li] = out[0][0, last_pos].detach().float().cpu()
                        elif out.dim() == 3:
                            pert_h[li] = out[0, last_pos].detach().float().cpu()
                    return hook
                
                hooks_pert = [layers[li].register_forward_hook(make_h_p(li)) for li in range(n_layers)]
                
                if abl_type == "mlp" and hasattr(layers[abl_layer], 'mlp'):
                    abl_h2 = layers[abl_layer].mlp.register_forward_hook(make_abl_hook())
                elif abl_type == "attn" and hasattr(layers[abl_layer], 'self_attn'):
                    abl_h2 = layers[abl_layer].self_attn.register_forward_hook(make_abl_hook())
                else:
                    abl_h2 = None
                
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
                    if abl_h2:
                        abl_h2.remove()
                    for h in hooks_pert:
                        h.remove()
                
                deltas = {}
                for li in range(n_layers):
                    if li in base_h and li in pert_h:
                        deltas[li] = (pert_h[li] - base_h[li]).numpy()
                obj_deltas_abl[obj_name] = deltas
            
            # 消融后shared_ratio
            abl_shared = {}
            for li in range(n_layers):
                layer_deltas = {o: d[li] for o, d in obj_deltas_abl.items() if li in d}
                stats = compute_binding_stats(layer_deltas, cat_dir.numpy() if isinstance(cat_dir, torch.Tensor) else cat_dir)
                if stats:
                    abl_shared[li] = stats["shared_ratio"]
            
            # 计算消融层之后的shared_ratio变化
            delta_shared = {}
            for li in range(n_layers):
                if li > abl_layer and li in full_shared and li in abl_shared:
                    delta_shared[li] = abl_shared[li] - full_shared[li]
            
            # 只保留消融层后3层的平均变化
            post_layers = sorted([l for l in delta_shared.keys()])[:5]
            if post_layers:
                avg_delta = np.mean([delta_shared[l] for l in post_layers])
            else:
                avg_delta = 0
            
            if abl_type == "mlp":
                mlp_effects[abl_layer] = round(avg_delta, 4)
            else:
                attn_effects[abl_layer] = round(avg_delta, 4)
            
            print(f"    {abl_type} L{abl_layer}: avg_delta_shared_post={avg_delta:.4f}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 打印完整MLP/Attn效应曲线
    print(f"\n  === MLP/Attn Privatization Effect Curve ===")
    print(f"  Layer | MLP_effect | Attn_effect | Interpretation")
    for li in abl_layers:
        mlp_e = mlp_effects.get(li, 0)
        attn_e = attn_effects.get(li, 0)
        if mlp_e < -0.02:
            mlp_role = "maintains sharing"
        elif mlp_e > 0.02:
            mlp_role = "promotes privatization"
        else:
            mlp_role = "neutral"
        
        if attn_e < -0.02:
            attn_role = "maintains sharing"
        elif attn_e > 0.02:
            attn_role = "promotes privatization"
        else:
            attn_role = "neutral"
        
        print(f"  L{li:2d}   | {mlp_e:+.4f}    | {attn_e:+.4f}     | MLP={mlp_role}, Attn={attn_role}")
    
    results = {
        "full_shared_ratios": full_shared,
        "mlp_effects": mlp_effects,
        "attn_effects": attn_effects,
    }
    
    return results


# ===== 确认2: GLM4负先验+高冲突恢复力验证 =====
def confirm_glm4_negative_prior(model, tokenizer, info):
    """验证GLM4的负模板先验和冲突恢复力在不同对象和模板下是否稳定"""
    print(f"\n{'='*60}")
    print("Confirmation 2: GLM4 Negative Prior + High Conflict Resilience")
    print(f"{'='*60}")
    
    templates = {
        "is_a": {"no_obj": "A thing is a kind of", "with_obj": "The {obj} is a kind of", "conflict": "Although the {obj} is described as a {opp_cat}, it is a kind of"},
        "has_a": {"no_obj": "A thing has a", "with_obj": "The {obj} has a", "conflict": "Although the {obj} is described as a {opp_cat}, it has a"},
    }
    
    opp_cats = {"apple": "animal", "dog": "fruit", "knife": "fruit", "orange": "animal", "hammer": "fruit", "cat": "fruit"}
    test_objects = ["apple", "dog", "knife", "orange", "hammer", "cat"]
    
    results = {}
    
    for obj_name in test_objects:
        if obj_name not in SLOT_ATTRS and obj_name not in {"orange", "hammer", "cat"}:
            continue
        
        obj_attrs = SLOT_ATTRS.get(obj_name, SLOT_ATTRS.get("apple", {}))
        if obj_name == "orange":
            obj_attrs = {"color": ["orange"], "taste": ["sweet", "citrus"], "part": ["peel", "seed"], "category": ["fruit", "citrus"], "non_category": ["animal", "tool"]}
        elif obj_name == "hammer":
            obj_attrs = {"color": ["gray", "brown"], "part": ["head", "handle"], "category": ["tool", "weapon"], "non_category": ["fruit", "animal"]}
        elif obj_name == "cat":
            obj_attrs = {"color": ["black", "white", "orange"], "part": ["tail", "fur", "whisker"], "category": ["animal", "pet"], "non_category": ["fruit", "tool"]}
        
        opp_cat = opp_cats.get(obj_name, "vehicle")
        obj_results = {}
        
        for tmpl_name, templates_dict in templates.items():
            slot_result = {}
            for cond_name, tmpl in templates_dict.items():
                if cond_name == "no_obj":
                    text = tmpl
                elif cond_name == "with_obj":
                    text = tmpl.format(obj=obj_name)
                elif cond_name == "conflict":
                    text = tmpl.format(obj=obj_name, opp_cat=opp_cat)
                else:
                    continue
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                input_device = next(model.parameters()).device
                inp_ids = inputs["input_ids"].to(input_device)
                attn_mask = inputs["attention_mask"].to(input_device)
                
                with torch.no_grad():
                    out = model(input_ids=inp_ids, attention_mask=attn_mask)
                logits = out.logits[0, -1].float().cpu().numpy()
                
                group_logits = {}
                for group_name, words in obj_attrs.items():
                    if not words:
                        continue
                    val = get_logit_values(logits, tokenizer, words)
                    if val is not None:
                        group_logits[group_name] = round(val, 4)
                slot_result[cond_name] = group_logits
            
            obj_results[tmpl_name] = slot_result
        
        # 计算拆分指标
        all_priors = []
        all_obj_conds = []
        all_conflict_deltas = []
        
        for tmpl_name, slot_data in obj_results.items():
            no_obj = slot_data.get("no_obj", {})
            with_obj = slot_data.get("with_obj", {})
            conflict = slot_data.get("conflict", {})
            
            for group_name in obj_attrs:
                if not obj_attrs[group_name]:
                    continue
                no_v = no_obj.get(group_name)
                with_v = with_obj.get(group_name)
                conf_v = conflict.get(group_name)
                
                if no_v is not None:
                    all_priors.append(no_v)
                if no_v is not None and with_v is not None:
                    all_obj_conds.append(with_v - no_v)
                if no_v is not None and conf_v is not None:
                    all_conflict_deltas.append(conf_v - no_v)
        
        avg_prior = np.mean(all_priors) if all_priors else 0
        avg_obj_cond = np.mean(all_obj_conds) if all_obj_conds else 0
        avg_conflict = np.mean(all_conflict_deltas) if all_conflict_deltas else 0
        
        prior_ratio = abs(avg_prior) / (abs(avg_prior) + abs(avg_obj_cond) + 1e-8)
        obj_cond_ratio = abs(avg_obj_cond) / (abs(avg_prior) + abs(avg_obj_cond) + 1e-8)
        conflict_resilience = avg_conflict / (avg_obj_cond + 1e-8) if abs(avg_obj_cond) > 0.01 else 0
        
        results[obj_name] = {
            "avg_prior": round(avg_prior, 3),
            "avg_obj_cond": round(avg_obj_cond, 3),
            "avg_conflict": round(avg_conflict, 3),
            "PriorScore": round(prior_ratio, 3),
            "ObjCondScore": round(obj_cond_ratio, 3),
            "ConflictResilience": round(conflict_resilience, 3),
            "prior_is_negative": avg_prior < 0,
        }
        
        print(f"  {obj_name}: prior={avg_prior:.3f}({'NEG' if avg_prior < 0 else 'POS'}), "
              f"obj_cond={avg_obj_cond:.3f}, conflict={avg_conflict:.3f}, "
              f"Resilience={conflict_resilience:.3f}")
    
    return results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    print(f"\n{'='*70}")
    print(f"Phase 448 R2: Confirmation Tests")
    print(f"Model: {model_name}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    t0 = time.time()
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    print(f"Model: {info.model_class}, {info.n_layers} layers, d_model={info.d_model}")
    
    cat_directions = {}
    for cat_name, cat_info in CATEGORY_OBJECTS.items():
        cat_dir = get_cat_direction(model, tokenizer, cat_info["cat_words"], cat_info["opp_words"])
        if cat_dir is not None:
            cat_directions[cat_name] = cat_dir
    
    all_results = {}
    
    # 确认1: MLP/Attn逐层消融
    r1 = confirm_privatization_dynamics(model, tokenizer, info, cat_directions)
    all_results["confirm1_privatization_dynamics"] = r1
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 确认2: GLM4负先验验证 (只对GLM4和DS7B做, Qwen3作为对比)
    r2 = confirm_glm4_negative_prior(model, tokenizer, info)
    all_results["confirm2_negative_prior"] = r2
    
    # 保存结果
    output_dir = "results/glm5"
    os.makedirs(output_dir, exist_ok=True)
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    all_results = convert_numpy(all_results)
    
    output_file = os.path.join(output_dir, f"phase448b_{model_name}_r2.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to: {output_file}")
    release_model(model)
    print(f"Phase 448 R2 complete!")


if __name__ == "__main__":
    main()
