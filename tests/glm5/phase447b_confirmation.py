"""
Phase 447 R2: 确认测试 — 绑定态分解稳定性 + SlotMediation深入
================================================================
目标:
1. 验证共享→私有化趋势在不同alpha下是否稳定
2. 深入分析SlotMediation: 不同模板下的属性读出变化
3. 对齐SwitchMediation: 用Phase 437的方法重新测,确认差异来源

用法:
  python tests/glm5/phase447b_confirmation.py qwen3 2
  python tests/glm5/phase447b_confirmation.py glm4 2
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


def load_model_auto(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
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


def collect_layer_deltas(model, tokenizer, input_ids, attention_mask, last_pos,
                        cat_dir, alpha, n_layers):
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
    
    base_hiddens = {}
    def make_base_hook(li):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                base_hiddens[li] = out[0][0, last_pos].detach().float().cpu()
            else:
                base_hiddens[li] = out[0, last_pos].detach().float().cpu() if out.dim() == 3 else out.detach().float().cpu()
        return hook
    
    hooks_base = [layers[li].register_forward_hook(make_base_hook(li)) for li in range(n_layers)]
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    for h in hooks_base:
        h.remove()
    
    pert_hiddens = {}
    def make_pert_hook(li):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                pert_hiddens[li] = out[0][0, last_pos].detach().float().cpu()
            else:
                pert_hiddens[li] = out[0, last_pos].detach().float().cpu() if out.dim() == 3 else out.detach().float().cpu()
        return hook
    
    hooks_pert = [layers[li].register_forward_hook(make_pert_hook(li)) for li in range(n_layers)]
    
    embed_hook = None
    def on_embed(module, inp, out):
        if isinstance(out, torch.Tensor):
            out = out.clone()
            out[0, last_pos] = out[0, last_pos] + perturb_vec
        return out
    
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_hook = model.model.embed_tokens.register_forward_hook(on_embed)
    
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    except Exception as e:
        print(f"    Perturbed forward failed: {e}")
    finally:
        if embed_hook:
            embed_hook.remove()
        for h in hooks_pert:
            h.remove()
    
    deltas = {}
    for li in range(n_layers):
        if li in base_hiddens and li in pert_hiddens:
            deltas[li] = (pert_hiddens[li] - base_hiddens[li]).numpy()
    
    return deltas


def compute_binding_stats(layer_deltas_dict, cat_dir_np):
    """计算绑定态分解统计量"""
    # layer_deltas_dict: {obj_name: delta_np}
    layer_deltas = []
    valid_objs = []
    for obj_name, d in layer_deltas_dict.items():
        if np.linalg.norm(d) > 1e-8:
            layer_deltas.append(d)
            valid_objs.append(obj_name)
    
    if len(layer_deltas) < 3:
        return None
    
    delta_matrix = np.stack(layer_deltas)
    
    # 共享/私有分解
    shared_direction = delta_matrix.mean(axis=0)
    residuals = delta_matrix - shared_direction
    
    total_var = np.sum(delta_matrix ** 2)
    shared_var = np.sum(shared_direction ** 2) * len(layer_deltas)
    private_var = np.sum(residuals ** 2)
    
    shared_ratio = shared_var / total_var if total_var > 1e-10 else 0
    
    # PCA
    centered = delta_matrix - delta_matrix.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pca1_ratio = float(S[0]**2 / np.sum(S**2)) if np.sum(S**2) > 0 else 0
    except:
        pca1_ratio = 0
    
    # 两两余弦
    pair_cosines = []
    for i in range(len(layer_deltas)):
        for j in range(i+1, len(layer_deltas)):
            ni = np.linalg.norm(layer_deltas[i])
            nj = np.linalg.norm(layer_deltas[j])
            if ni > 1e-8 and nj > 1e-8:
                cos_ij = float(np.dot(layer_deltas[i], layer_deltas[j]) / (ni * nj))
                pair_cosines.append(cos_ij)
    
    avg_pair_cos = float(np.mean(pair_cosines)) if pair_cosines else 0
    
    # 类别投影
    shared_norm = np.linalg.norm(shared_direction)
    shared_cat_proj = float(np.dot(shared_direction, cat_dir_np) / shared_norm) if shared_norm > 1e-8 else 0
    
    return {
        "shared_ratio": round(float(shared_ratio), 4),
        "pca1_ratio": round(float(pca1_ratio), 4),
        "avg_pair_cosine": round(float(avg_pair_cos), 4),
        "shared_cat_proj": round(float(shared_cat_proj), 4),
        "n_objects": len(valid_objs),
    }


# ===== 确认实验1: 不同alpha下的绑定态分解 =====
def confirm_binding_stability(model, tokenizer, info, cat_directions):
    """验证共享→私有化趋势在不同alpha下是否稳定"""
    print(f"\n{'='*60}")
    print("Confirmation 1: Binding Decomposition Stability across Alpha")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    results = {}
    
    for cat_name, cat_info in CATEGORY_OBJECTS.items():
        if cat_name not in cat_directions:
            continue
        
        cat_dir = cat_directions[cat_name]
        cat_dir_np = cat_dir.numpy() if isinstance(cat_dir, torch.Tensor) else cat_dir
        obj_names = cat_info["objects"]
        
        print(f"\n  Category: {cat_name}")
        
        for alpha in [0.5, 1.0, 2.0]:
            print(f"    alpha={alpha}:")
            
            # 收集每个对象的delta
            obj_deltas = {}
            for obj_name in obj_names:
                text = f"The {obj_name} is a"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                input_device = next(model.parameters()).device
                input_ids = inputs["input_ids"].to(input_device)
                attention_mask = inputs["attention_mask"].to(input_device)
                last_pos = input_ids.shape[1] - 1
                
                deltas = collect_layer_deltas(model, tokenizer, input_ids, attention_mask,
                                             last_pos, cat_dir, alpha, n_layers)
                obj_deltas[obj_name] = deltas
            
            # 逐层分析
            layer_stats = {}
            sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
            
            for li in sample_layers:
                layer_deltas = {o: d[li] for o, d in obj_deltas.items() if li in d}
                stats = compute_binding_stats(layer_deltas, cat_dir_np)
                if stats:
                    layer_stats[f"L{li}"] = stats
            
            results[f"{cat_name}_a{alpha}"] = layer_stats
            
            # 打印摘要
            for lk, ls in layer_stats.items():
                print(f"      {lk}: shared={ls['shared_ratio']:.3f}, pair_cos={ls['avg_pair_cosine']:.3f}")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return results


# ===== 确认实验2: SlotMediation深入分析 =====
def confirm_slot_mediation(model, tokenizer, info):
    """深入分析SlotMediation: 为什么不同模板对属性影响如此大"""
    print(f"\n{'='*60}")
    print("Confirmation 2: SlotMediation Deep Analysis")
    print(f"{'='*60}")
    
    # 更多模板 + 更多属性
    templates = {
        "is_a": "The {obj} is a",
        "has_a": "The {obj} has a",
        "feels": "The {obj} feels",
        "is_made_of": "The {obj} is made of",
        "is_used_for": "The {obj} is used for",
        "tastes": "The {obj} tastes",
        "looks_like": "The {obj} looks like",
    }
    
    objects_attrs = {
        "apple": {
            "color": ["red", "green", "yellow"],
            "taste": ["sweet", "sour", "juicy"],
            "part": ["seed", "skin", "core", "stem"],
            "category": ["fruit", "food"],
            "non_category": ["animal", "tool"],
        },
        "dog": {
            "color": ["brown", "black", "white"],
            "taste": [],
            "part": ["leg", "tail", "fur", "ear"],
            "category": ["animal", "pet"],
            "non_category": ["fruit", "tool"],
        },
        "knife": {
            "color": ["silver", "gray", "metallic"],
            "taste": [],
            "part": ["blade", "handle", "edge", "tip"],
            "category": ["tool", "weapon"],
            "non_category": ["fruit", "animal"],
        },
    }
    
    results = {}
    
    for obj_name, attr_groups in objects_attrs.items():
        print(f"\n  {obj_name}:")
        obj_results = {}
        
        for tmpl_name, tmpl in templates.items():
            text = tmpl.format(obj=obj_name)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_device = next(model.parameters()).device
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[0, -1].float().cpu().numpy()
            
            # 各属性组的平均logit
            group_logits = {}
            for group_name, words in attr_groups.items():
                if not words:
                    continue
                ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in words if tokenizer.encode(w, add_special_tokens=False)]
                if ids:
                    group_logits[group_name] = round(float(np.mean(logits[ids])), 4)
            
            obj_results[tmpl_name] = group_logits
            
            group_str = ", ".join(f"{k}={v:.2f}" for k, v in group_logits.items())
            print(f"    {tmpl_name}: {group_str}")
        
        results[obj_name] = obj_results
        
        # 计算SlotMediation: 各属性组在不同模板下的logit范围
        for group_name in attr_groups:
            if not attr_groups[group_name]:
                continue
            values = [obj_results[t][group_name] for t in obj_results if group_name in obj_results[t]]
            if len(values) >= 2:
                slot_range = max(values) - min(values)
                print(f"    -> {group_name} SlotRange: {slot_range:.3f}")
    
    return results


# ===== 确认实验3: SwitchMediation方法对齐 =====
def confirm_switch_mediation_alignment(model, tokenizer, info, cat_directions):
    """用Phase 437的方法重新测SwitchMediation，确认差异来源"""
    print(f"\n{'='*60}")
    print("Confirmation 3: SwitchMediation Method Alignment")
    print(f"{'='*60}")
    
    results = {}
    
    test_cases = [
        {
            "obj": "apple", "cat": "fruit",
            "cat_words": ["fruit", "apple", "orange", "banana"],
            "opp_words": ["animal", "dog", "cat", "horse"],
        },
        {
            "obj": "dog", "cat": "animal",
            "cat_words": ["animal", "dog", "cat", "horse"],
            "opp_words": ["fruit", "apple", "orange", "banana"],
        },
    ]
    
    for tc in test_cases:
        obj_name = tc["obj"]
        cat_name = tc["cat"]
        cat_words = tc["cat_words"]
        opp_words = tc["opp_words"]
        
        if cat_name not in cat_directions:
            continue
        
        cat_dir = cat_directions[cat_name]
        input_device = next(model.parameters()).device
        
        text = f"The {obj_name} is a"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        last_pos = input_ids.shape[1] - 1
        
        # 基准
        with torch.no_grad():
            base_out = model(input_ids=input_ids, attention_mask=attention_mask)
        base_logits = base_out.logits[0, -1].float().cpu().numpy()
        
        cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
        opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
        
        base_cat_logit = float(np.mean(base_logits[cat_ids]))
        base_opp_logit = float(np.mean(base_logits[opp_ids]))
        base_cat_gap = base_cat_logit - base_opp_logit
        
        # 方法1: Phase 437的"push类别到对立方向"(cat_gap变化)
        # 方法2: Phase 447的"related vs unrelated属性差"
        
        # 相关属性和无关属性
        related_attrs = {"apple": ["sweet", "edible", "seed", "juicy"],
                        "dog": ["alive", "wild", "furry", "leg"]}
        unrelated_attrs = {"apple": ["fast", "loud", "metal", "engine"],
                          "dog": ["sweet", "round", "metal", "sharp"]}
        
        rel_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in related_attrs.get(obj_name, []) if tokenizer.encode(w, add_special_tokens=False)]
        unrel_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in unrelated_attrs.get(obj_name, []) if tokenizer.encode(w, add_special_tokens=False)]
        
        base_rel_logit = float(np.mean(base_logits[rel_ids])) if rel_ids else 0
        base_unrel_logit = float(np.mean(base_logits[unrel_ids])) if unrel_ids else 0
        
        mediation_by_alpha = {}
        
        for alpha in [0.5, 1.0, 1.5, 2.0]:
            # push到对立类别方向
            perturb_vec = (alpha * (-cat_dir)).to(input_device).to(torch.bfloat16)
            embed_hook = None
            
            def make_hook(pv, lp):
                def hook(module, inp, out):
                    if isinstance(out, torch.Tensor):
                        out = out.clone()
                        out[0, lp] = out[0, lp] + pv
                    return out
                return hook
            
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                embed_hook = model.model.embed_tokens.register_forward_hook(
                    make_hook(perturb_vec, last_pos))
            
            with torch.no_grad():
                pert_out = model(input_ids=input_ids, attention_mask=attention_mask)
            if embed_hook:
                embed_hook.remove()
            
            pert_logits = pert_out.logits[0, -1].float().cpu().numpy()
            
            pert_cat_logit = float(np.mean(pert_logits[cat_ids]))
            pert_opp_logit = float(np.mean(pert_logits[opp_ids]))
            pert_cat_gap = pert_cat_logit - pert_opp_logit
            
            pert_rel_logit = float(np.mean(pert_logits[rel_ids])) if rel_ids else 0
            pert_unrel_logit = float(np.mean(pert_logits[unrel_ids])) if unrel_ids else 0
            
            # 方法1: cat_gap变化 (Phase 437)
            cat_gap_shift = pert_cat_gap - base_cat_gap
            
            # 方法2: related vs unrelated差 (Phase 447)
            attr_mediation = (base_rel_logit - pert_rel_logit) - (base_unrel_logit - pert_unrel_logit)
            
            # 方法3: 类别候选切换 — top1是否从cat变opp
            top5_pert = set(np.argsort(pert_logits)[-5:])
            top5_base = set(np.argsort(base_logits)[-5:])
            top5_overlap = len(top5_pert & top5_base) / 5
            
            mediation_by_alpha[f"alpha_{alpha}"] = {
                "cat_gap_shift": round(float(cat_gap_shift), 4),
                "cat_shift_per_alpha": round(float(cat_gap_shift / alpha), 4),
                "attr_mediation": round(float(attr_mediation), 4),
                "attr_med_per_alpha": round(float(attr_mediation / alpha), 4),
                "top5_overlap": round(float(top5_overlap), 4),
                "pert_rel_logit": round(float(pert_rel_logit), 4),
                "pert_unrel_logit": round(float(pert_unrel_logit), 4),
            }
        
        results[obj_name] = {
            "base_cat_gap": round(float(base_cat_gap), 4),
            "base_rel_logit": round(float(base_rel_logit), 4),
            "base_unrel_logit": round(float(base_unrel_logit), 4),
            "mediation_by_alpha": mediation_by_alpha,
        }
        
        print(f"\n  {obj_name}:")
        print(f"    base_cat_gap={base_cat_gap:.3f}")
        for ak, av in mediation_by_alpha.items():
            print(f"    {ak}: cat_shift={av['cat_gap_shift']:.3f}, "
                  f"attr_med={av['attr_mediation']:.3f}, top5_overlap={av['top5_overlap']:.3f}")
    
    return results


def run_experiment(model_name, round_num):
    print(f"\n{'='*70}")
    print(f"Phase 447 R2: Confirmation Tests")
    print(f"Model: {model_name}, Round: {round_num}")
    print(f"{'='*70}")
    
    t0 = time.time()
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  Loaded: {info.model_class}, {n_layers} layers, d_model={info.d_model}")
    
    # 计算类别方向
    cat_directions = {}
    for cat_name, cat_info in CATEGORY_OBJECTS.items():
        d = get_cat_direction(model, tokenizer, cat_info["cat_words"], cat_info["opp_words"])
        if d is not None:
            cat_directions[cat_name] = d
    
    all_results = {}
    
    # 确认1: 绑定态稳定性
    r1 = confirm_binding_stability(model, tokenizer, info, cat_directions)
    all_results["binding_stability"] = r1
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 确认2: SlotMediation深入
    r2 = confirm_slot_mediation(model, tokenizer, info)
    all_results["slot_mediation_deep"] = r2
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 确认3: SwitchMediation对齐
    r3 = confirm_switch_mediation_alignment(model, tokenizer, info, cat_directions)
    all_results["switch_mediation_alignment"] = r3
    
    # 保存
    output = {
        "model": model_name,
        "round": round_num,
        "n_layers": n_layers,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": all_results,
    }
    
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj
    
    os.makedirs("results/phase447_binding_decomposition", exist_ok=True)
    out_path = f"results/phase447_binding_decomposition/{model_name}_phase447_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(convert(output), f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")
    
    print(f"\n  Total time: {time.time()-t0:.1f}s")
    
    release_model(model)
    model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    run_experiment(model_name, round_num)
