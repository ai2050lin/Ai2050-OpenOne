"""
Phase 448: Slot-Dominated Binding State Privatization
=====================================================
核心假说: 关系槽位是属性检索的第一控制变量;
类别绑定态从共享入口逐步私有化;
MLP/Attn分别负责什么需要逐层消融确定。

实验1: SlotMediation拆分 — 模板先验 vs 对象知识 vs 冲突对象
实验2: 共享→私有化层间动力学 — MLP vs Attn消融,逐层测shared_ratio
实验3: Alpha机制区间扫描 — 定义natural/transition/forced regime

用法:
  python tests/glm5/phase448_slot_privatization.py qwen3 1
  python tests/glm5/phase448_slot_privatization.py glm4 1
  python tests/glm5/phase448_slot_privatization.py deepseek7b 1
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

# 属性词组 (for SlotMediation)
SLOT_ATTRS = {
    "apple": {
        "color": ["red", "green", "yellow"],
        "taste": ["sweet", "sour", "juicy"],
        "part": ["seed", "skin", "core", "stem"],
        "category": ["fruit", "food", "produce"],
        "non_category": ["animal", "tool", "vehicle"],
        "material": ["organic", "fresh", "natural"],
        "random": ["square", "loud", "electric", "digital"],
    },
    "dog": {
        "color": ["brown", "black", "white"],
        "taste": [],
        "part": ["leg", "tail", "fur", "ear"],
        "category": ["animal", "pet", "mammal"],
        "non_category": ["fruit", "tool", "vehicle"],
        "material": [],
        "random": ["square", "sweet", "metallic", "digital"],
    },
    "knife": {
        "color": ["silver", "gray", "metallic"],
        "taste": [],
        "part": ["blade", "handle", "edge", "tip"],
        "category": ["tool", "weapon", "instrument"],
        "non_category": ["fruit", "animal", "vehicle"],
        "material": ["metal", "steel", "iron"],
        "random": ["sweet", "furry", "organic", "digital"],
    },
}


def load_model_auto(model_name):
    """BF16 + device_map='auto' + sdpa"""
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
    """从W_E计算类别方向"""
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
    """获取词列表的平均logit值"""
    ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in word_list 
           if tokenizer.encode(w, add_special_tokens=False)]
    if not ids:
        return None
    return float(logits[ids].mean())


def collect_layer_hiddens(model, tokenizer, input_ids, attention_mask, last_pos, n_layers):
    """收集每层hidden state (last_pos位置)"""
    layers = get_layers(model)
    hiddens = {}
    
    def make_hook(li):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                hiddens[li] = out[0][0, last_pos].detach().float().cpu()
            else:
                t = out
                if t.dim() == 3:
                    hiddens[li] = t[0, last_pos].detach().float().cpu()
                else:
                    hiddens[li] = t.detach().float().cpu()
        return hook
    
    hooks = [layers[li].register_forward_hook(make_hook(li)) for li in range(n_layers)]
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    for h in hooks:
        h.remove()
    
    return hiddens


def collect_deltas_with_perturbation(model, tokenizer, input_ids, attention_mask, last_pos,
                                      cat_dir, alpha, n_layers):
    """注入类别方向扰动,收集逐层delta"""
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
    
    # 基准
    base_hiddens = {}
    def make_base_hook(li):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                base_hiddens[li] = out[0][0, last_pos].detach().float().cpu()
            else:
                t = out
                if t.dim() == 3:
                    base_hiddens[li] = t[0, last_pos].detach().float().cpu()
                else:
                    base_hiddens[li] = t.detach().float().cpu()
        return hook
    
    hooks_base = [layers[li].register_forward_hook(make_base_hook(li)) for li in range(n_layers)]
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    for h in hooks_base:
        h.remove()
    
    # 扰动
    pert_hiddens = {}
    def make_pert_hook(li):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                pert_hiddens[li] = out[0][0, last_pos].detach().float().cpu()
            else:
                t = out
                if t.dim() == 3:
                    pert_hiddens[li] = t[0, last_pos].detach().float().cpu()
                else:
                    pert_hiddens[li] = t.detach().float().cpu()
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


def collect_mlp_attn_outputs(model, tokenizer, input_ids, attention_mask, last_pos, n_layers):
    """分别收集MLP输出和Attn输出 (用于消融实验)"""
    layers = get_layers(model)
    
    mlp_outs = {}
    attn_outs = {}
    residual_outs = {}
    
    def make_mlp_hook(li):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                mlp_outs[li] = out[0][0, last_pos].detach().float().cpu()
            else:
                mlp_outs[li] = out[0, last_pos].detach().float().cpu() if out.dim() == 3 else out.detach().float().cpu()
        return hook
    
    def make_attn_hook(li):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                attn_outs[li] = out[0][0, last_pos].detach().float().cpu()
            else:
                attn_outs[li] = out[0, last_pos].detach().float().cpu() if out.dim() == 3 else out.detach().float().cpu()
        return hook
    
    hooks = []
    for li in range(n_layers):
        layer = layers[li]
        if hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(li)))
        if hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(li)))
    
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    for h in hooks:
        h.remove()
    
    return mlp_outs, attn_outs


def compute_binding_stats(layer_deltas_dict, cat_dir_np):
    """计算绑定态分解统计量"""
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
    private_var = np.sum(residuals ** 2)
    
    shared_ratio = shared_var / total_var if total_var > 1e-10 else 0
    
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
    
    # PCA第一主成分
    centered = delta_matrix - delta_matrix.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pca1_ratio = float(S[0]**2 / np.sum(S**2)) if np.sum(S**2) > 0 else 0
    except:
        pca1_ratio = 0
    
    return {
        "shared_ratio": round(float(shared_ratio), 4),
        "pca1_ratio": round(float(pca1_ratio), 4),
        "avg_pair_cosine": round(float(avg_pair_cos), 4),
        "n_objects": len(valid_objs),
        "total_var": round(float(total_var), 4),
    }


# ===== 实验1: SlotMediation拆分 =====
def experiment1_slot_decomposition(model, tokenizer, info):
    """
    区分三种成分:
    - TemplatePrior: 无对象模板("A thing has ___") → 纯模板先验
    - ObjectCondition: 有对象模板("An apple has ___") vs 无对象 → 对象条件化增量
    - ConflictCondition: 冲突模板("Although an apple is an animal, it has ___") → 类别冲突下对象知识
    """
    print(f"\n{'='*60}")
    print("Experiment 1: SlotMediation Decomposition")
    print(f"{'='*60}")
    
    results = {}
    
    # 三类模板
    slot_templates = {
        "is_a": {
            "no_obj": "A thing is a kind of",
            "with_obj": "The {obj} is a kind of",
            "conflict": "Although the {obj} is described as a {opp_cat}, it is a kind of",
        },
        "has_a": {
            "no_obj": "A thing has a",
            "with_obj": "The {obj} has a",
            "conflict": "Although the {obj} is described as a {opp_cat}, it has a",
        },
        "feels": {
            "no_obj": "A thing feels",
            "with_obj": "The {obj} feels",
            "conflict": "Although the {obj} is described as a {opp_cat}, it feels",
        },
        "tastes": {
            "no_obj": "A thing tastes",
            "with_obj": "The {obj} tastes",
            "conflict": "Although the {obj} is described as a {opp_cat}, it tastes",
        },
        "is_made_of": {
            "no_obj": "A thing is made of",
            "with_obj": "The {obj} is made of",
            "conflict": "Although the {obj} is described as a {opp_cat}, it is made of",
        },
    }
    
    opp_cats = {"apple": "animal", "dog": "fruit", "knife": "fruit"}
    test_objects = ["apple", "dog", "knife"]
    
    for obj_name in test_objects:
        if obj_name not in SLOT_ATTRS:
            continue
        
        print(f"\n  Object: {obj_name}")
        obj_attrs = SLOT_ATTRS[obj_name]
        opp_cat = opp_cats.get(obj_name, "vehicle")
        obj_results = {}
        
        for slot_name, templates in slot_templates.items():
            slot_result = {}
            
            for cond_name, tmpl in templates.items():
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
                
                # 收集各属性组logit
                group_logits = {}
                for group_name, words in obj_attrs.items():
                    if not words:
                        continue
                    val = get_logit_values(logits, tokenizer, words)
                    if val is not None:
                        group_logits[group_name] = round(val, 4)
                
                slot_result[cond_name] = group_logits
            
            obj_results[slot_name] = slot_result
            
            # 计算拆分指标
            no_obj = slot_result.get("no_obj", {})
            with_obj = slot_result.get("with_obj", {})
            conflict = slot_result.get("conflict", {})
            
            # 拆分: TemplatePrior = no_obj的属性logit
            # ObjectCondition = with_obj - no_obj (对象知识增量)
            # ConflictResilience = conflict保持对象属性的程度
            
            for group_name in obj_attrs:
                if not obj_attrs[group_name]:
                    continue
                
                no_v = no_obj.get(group_name)
                with_v = with_obj.get(group_name)
                conf_v = conflict.get(group_name)
                
                if no_v is not None and with_v is not None:
                    obj_cond = with_v - no_v  # 对象知识增量
                    
                    conf_resilience = None
                    if conf_v is not None and with_v is not None:
                        # 冲突下对象属性保持度 = conflict / with_obj
                        conf_resilience = conf_v - no_v  # 冲突下的对象增量
                    
                    print(f"    {slot_name}/{group_name}: "
                          f"prior={no_v:.2f}, obj_cond={obj_cond:.2f}, "
                          f"conflict_delta={conf_resilience:.2f}" if conf_resilience is not None 
                          else f"    {slot_name}/{group_name}: prior={no_v:.2f}, obj_cond={obj_cond:.2f}")
        
        results[obj_name] = obj_results
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总: 对每个对象, 计算平均拆分
    print(f"\n  === SlotMediation Decomposition Summary ===")
    for obj_name in test_objects:
        if obj_name not in results:
            continue
        
        obj_attrs = SLOT_ATTRS.get(obj_name, {})
        
        # 收集所有slot的拆分
        all_priors = []
        all_obj_conds = []
        all_conflict_deltas = []
        
        for slot_name, slot_data in results[obj_name].items():
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
        
        print(f"  {obj_name}: avg_prior={avg_prior:.3f}, avg_obj_condition={avg_obj_cond:.3f}, "
              f"avg_conflict_delta={avg_conflict:.3f}")
        
        # 关键指标
        prior_ratio = abs(avg_prior) / (abs(avg_prior) + abs(avg_obj_cond) + 1e-8)
        obj_cond_ratio = abs(avg_obj_cond) / (abs(avg_prior) + abs(avg_obj_cond) + 1e-8)
        conflict_resilience = avg_conflict / (avg_obj_cond + 1e-8) if abs(avg_obj_cond) > 0.01 else 0
        
        print(f"    TemplatePriorScore={prior_ratio:.3f}, ObjectConditionScore={obj_cond_ratio:.3f}, "
              f"ConflictResilience={conflict_resilience:.3f}")
    
    return results


# ===== 实验2: 共享→私有化层间动力学 =====
def experiment2_privatization_dynamics(model, tokenizer, info, cat_directions):
    """
    逐层测量shared_ratio, 并消融MLP/Attn确定私有化驱动来源
    
    方法:
    1. 对多对象收集delta, 逐层计算shared_ratio (完整模型)
    2. 在目标层消融MLP输出(置零)后, 看下一层shared_ratio变化
    3. 在目标层消融Attn输出(置零)后, 看下一层shared_ratio变化
    """
    print(f"\n{'='*60}")
    print("Experiment 2: Shared→Private Dynamics (MLP vs Attn Ablation)")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    results = {}
    
    for cat_name, cat_info in CATEGORY_OBJECTS.items():
        if cat_name not in cat_directions:
            continue
        
        cat_dir = cat_directions[cat_name]
        obj_names = cat_info["objects"]
        
        print(f"\n  Category: {cat_name}")
        
        # ===== Step 1: 完整模型的逐层shared_ratio =====
        print(f"    Step 1: Full model shared_ratio...")
        alpha = 1.0
        obj_deltas_full = {}
        
        for oi, obj_name in enumerate(obj_names):
            text = f"The {obj_name} is a"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_device = next(model.parameters()).device
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            deltas = collect_deltas_with_perturbation(
                model, tokenizer, input_ids, attention_mask,
                last_pos, cat_dir, alpha, n_layers
            )
            obj_deltas_full[obj_name] = deltas
            
            if (oi+1) % 3 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 逐层shared_ratio
        full_shared_ratios = {}
        sample_layers = list(range(0, n_layers, max(1, n_layers // 12))) + [n_layers - 1]
        sample_layers = sorted(set(sample_layers))
        
        for li in sample_layers:
            layer_deltas = {o: d[li] for o, d in obj_deltas_full.items() if li in d}
            stats = compute_binding_stats(layer_deltas, cat_dir.numpy() if isinstance(cat_dir, torch.Tensor) else cat_dir)
            if stats:
                full_shared_ratios[li] = stats["shared_ratio"]
        
        # 打印趋势
        print(f"    Full model shared_ratio trend:")
        for li in sample_layers:
            sr = full_shared_ratios.get(li, None)
            if sr is not None:
                print(f"      L{li}: {sr:.4f}")
        
        # ===== Step 2: MLP消融 / Attn消融 =====
        # 选取关键层做消融: L0, L_mid, L_late
        ablation_layers = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
        ablation_layers = sorted(set([l for l in ablation_layers if l < n_layers - 1]))
        
        layers = get_layers(model)
        input_device = next(model.parameters()).device
        
        ablation_results = {}
        
        for abl_layer in ablation_layers:
            print(f"\n    Step 2: Ablating Layer {abl_layer}...")
            
            # 测两种消融: MLP=0, Attn=0
            for abl_type in ["mlp", "attn"]:
                key = f"L{abl_layer}_{abl_type}"
                print(f"      Ablation type: {abl_type}")
                
                # 重新收集消融后的delta
                obj_deltas_abl = {}
                
                for oi, obj_name in enumerate(obj_names):
                    text = f"The {obj_name} is a"
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                    input_ids_base = inputs["input_ids"].to(input_device)
                    attention_mask_base = inputs["attention_mask"].to(input_device)
                    last_pos = input_ids_base.shape[1] - 1
                    
                    # --- 基准 + 消融 ---
                    base_hiddens = {}
                    def make_base_hook_abl(li):
                        def hook(module, inp, out):
                            if isinstance(out, tuple):
                                base_hiddens[li] = out[0][0, last_pos].detach().float().cpu()
                            else:
                                t = out
                                if t.dim() == 3:
                                    base_hiddens[li] = t[0, last_pos].detach().float().cpu()
                                else:
                                    base_hiddens[li] = t.detach().float().cpu()
                        return hook
                    
                    # 消融hook: 在abl_layer将MLP或Attn输出置零
                    def make_ablation_hook():
                        def hook(module, inp, out):
                            if isinstance(out, tuple):
                                # 返回shape一样的零张量
                                return (torch.zeros_like(out[0]),) + out[1:]
                            else:
                                return torch.zeros_like(out)
                        return hook
                    
                    hooks_base = [layers[li].register_forward_hook(make_base_hook_abl(li)) 
                                 for li in range(n_layers)]
                    
                    # 消融hook
                    if abl_type == "mlp" and hasattr(layers[abl_layer], 'mlp'):
                        abl_hook = layers[abl_layer].mlp.register_forward_hook(make_ablation_hook())
                    elif abl_type == "attn" and hasattr(layers[abl_layer], 'self_attn'):
                        abl_hook = layers[abl_layer].self_attn.register_forward_hook(make_ablation_hook())
                    else:
                        abl_hook = None
                    
                    with torch.no_grad():
                        _ = model(input_ids=input_ids_base, attention_mask=attention_mask_base)
                    
                    if abl_hook is not None:
                        abl_hook.remove()
                    for h in hooks_base:
                        h.remove()
                    
                    # --- 扰动 + 消融 ---
                    perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
                    
                    pert_hiddens = {}
                    def make_pert_hook_abl(li):
                        def hook(module, inp, out):
                            if isinstance(out, tuple):
                                pert_hiddens[li] = out[0][0, last_pos].detach().float().cpu()
                            else:
                                t = out
                                if t.dim() == 3:
                                    pert_hiddens[li] = t[0, last_pos].detach().float().cpu()
                                else:
                                    pert_hiddens[li] = t.detach().float().cpu()
                        return hook
                    
                    hooks_pert = [layers[li].register_forward_hook(make_pert_hook_abl(li)) 
                                 for li in range(n_layers)]
                    
                    if abl_type == "mlp" and hasattr(layers[abl_layer], 'mlp'):
                        abl_hook2 = layers[abl_layer].mlp.register_forward_hook(make_ablation_hook())
                    elif abl_type == "attn" and hasattr(layers[abl_layer], 'self_attn'):
                        abl_hook2 = layers[abl_layer].self_attn.register_forward_hook(make_ablation_hook())
                    else:
                        abl_hook2 = None
                    
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
                            _ = model(input_ids=input_ids_base, attention_mask=attention_mask_base)
                    except Exception as e:
                        print(f"        Forward failed for {obj_name}: {e}")
                    finally:
                        if embed_hook:
                            embed_hook.remove()
                        if abl_hook2 is not None:
                            abl_hook2.remove()
                        for h in hooks_pert:
                            h.remove()
                    
                    # 计算delta
                    deltas = {}
                    for li in range(n_layers):
                        if li in base_hiddens and li in pert_hiddens:
                            deltas[li] = (pert_hiddens[li] - base_hiddens[li]).numpy()
                    obj_deltas_abl[obj_name] = deltas
                
                # 逐层shared_ratio (消融后)
                abl_shared_ratios = {}
                for li in sample_layers:
                    layer_deltas = {o: d[li] for o, d in obj_deltas_abl.items() if li in d}
                    stats = compute_binding_stats(layer_deltas, cat_dir.numpy() if isinstance(cat_dir, torch.Tensor) else cat_dir)
                    if stats:
                        abl_shared_ratios[li] = stats["shared_ratio"]
                
                # 关键: 对比消融前后, 消融层之后的shared_ratio变化
                # 如果消融MLP后shared_ratio下降 → MLP维持共享
                # 如果消融MLP后shared_ratio上升 → MLP促进私有化
                delta_shared = {}
                for li in sample_layers:
                    if li > abl_layer and li in full_shared_ratios and li in abl_shared_ratios:
                        delta_shared[li] = abl_shared_ratios[li] - full_shared_ratios[li]
                
                ablation_results[key] = {
                    "ablation_layer": abl_layer,
                    "ablation_type": abl_type,
                    "shared_ratios_after_ablation": abl_shared_ratios,
                    "full_model_shared_ratios": {k: v for k, v in full_shared_ratios.items()},
                    "delta_shared_post_ablation": delta_shared,
                }
                
                # 打印关键结果
                print(f"      Shared ratio changes after L{abl_layer} {abl_type} ablation:")
                for li in sorted(delta_shared.keys()):
                    sign = "+" if delta_shared[li] > 0 else ""
                    print(f"        L{li}: {full_shared_ratios.get(li, 0):.4f} → {abl_shared_ratios.get(li, 0):.4f} ({sign}{delta_shared[li]:.4f})")
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        results[cat_name] = {
            "full_shared_ratios": full_shared_ratios,
            "ablation_results": ablation_results,
        }
    
    return results


# ===== 实验3: Alpha机制区间扫描 =====
def experiment3_alpha_regime_scan(model, tokenizer, info, cat_directions):
    """
    扫描alpha, 测量:
    - shared_ratio (共享比例)
    - 流形有效性 (logit entropy)
    - 类别切换效果 (cat_logit_gap)
    - BoostMediation
    
    目标: 定义 natural / transition / forced regime
    """
    print(f"\n{'='*60}")
    print("Experiment 3: Alpha Regime Scan")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    results = {}
    
    test_cases = [
        {"obj": "apple", "cat": "fruit", "cat_words": ["fruit"], "opp_words": ["animal"]},
        {"obj": "dog", "cat": "animal", "cat_words": ["animal"], "opp_words": ["fruit"]},
        {"obj": "knife", "cat": "tool", "cat_words": ["tool"], "opp_words": ["fruit"]},
    ]
    
    alphas = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    
    for tc in test_cases:
        obj_name = tc["obj"]
        cat_name = tc["cat"]
        
        if cat_name not in cat_directions:
            continue
        
        cat_dir = cat_directions[cat_name]
        input_device = next(model.parameters()).device
        
        print(f"\n  Object: {obj_name} (category: {cat_name})")
        
        # 基准运行
        text = f"The {obj_name} is a"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        last_pos = input_ids.shape[1] - 1
        
        with torch.no_grad():
            base_out = model(input_ids=input_ids, attention_mask=attention_mask)
        base_logits = base_out.logits[0, -1].float().cpu().numpy()
        
        cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in tc["cat_words"]]
        opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in tc["opp_words"]]
        base_cat_gap = float(np.mean(base_logits[cat_ids]) - np.mean(base_logits[opp_ids]))
        
        # 计算基准entropy
        base_probs = np.exp(base_logits - base_logits.max())
        base_probs = base_probs / base_probs.sum()
        base_entropy = float(-np.sum(base_probs * np.log(base_probs + 1e-10)))
        
        alpha_results = {}
        
        for alpha in alphas:
            # 扰动运行
            perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
            
            # 注入扰动到embedding
            embed_layer = model.get_input_embeddings()
            input_ids_for_embed = input_ids.clone()
            inputs_embeds_base = embed_layer(input_ids_for_embed).detach().clone()
            inputs_embeds_pert = inputs_embeds_base.clone()
            inputs_embeds_pert[0, last_pos] = inputs_embeds_pert[0, last_pos] + perturb_vec.to(inputs_embeds_pert.dtype)
            
            with torch.no_grad():
                pert_out = model(inputs_embeds=inputs_embeds_pert)
            pert_logits = pert_out.logits[0, -1].float().cpu().numpy()
            
            pert_cat_gap = float(np.mean(pert_logits[cat_ids]) - np.mean(pert_logits[opp_ids]))
            
            pert_probs = np.exp(pert_logits - pert_logits.max())
            pert_probs = pert_probs / pert_probs.sum()
            pert_entropy = float(-np.sum(pert_probs * np.log(pert_probs + 1e-10)))
            
            # 流形有效性: top-1概率
            base_top1_prob = float(base_probs.max())
            pert_top1_prob = float(pert_probs.max())
            validity = pert_top1_prob / max(base_top1_prob, 1e-8)
            
            # 类别切换效果
            cat_shift = pert_cat_gap - base_cat_gap
            
            # 收集逐层delta (只需要sample_layers)
            deltas = collect_deltas_with_perturbation(
                model, tokenizer, input_ids, attention_mask,
                last_pos, cat_dir, alpha, n_layers
            )
            
            # 只在最后一层算简单统计
            last_delta = deltas.get(n_layers - 1, None)
            delta_norm = float(np.linalg.norm(last_delta)) if last_delta is not None else 0
            
            alpha_results[alpha] = {
                "cat_shift": round(cat_shift, 4),
                "pert_cat_gap": round(pert_cat_gap, 4),
                "base_cat_gap": round(base_cat_gap, 4),
                "entropy": round(pert_entropy, 4),
                "base_entropy": round(base_entropy, 4),
                "validity": round(validity, 4),
                "delta_norm_last": round(delta_norm, 4),
            }
            
            print(f"    alpha={alpha:.2f}: cat_shift={cat_shift:.3f}, "
                  f"entropy={pert_entropy:.3f}(base={base_entropy:.3f}), "
                  f"validity={validity:.3f}, delta_norm={delta_norm:.3f}")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 判断regime
        # natural: validity > 0.8, entropy变化 < 10%
        # forced: validity < 0.5 或 entropy变化 > 30%
        # transition: 中间
        regimes = {}
        for alpha, ar in alpha_results.items():
            ent_change = abs(ar["entropy"] - ar["base_entropy"]) / max(ar["base_entropy"], 1e-8)
            val = ar["validity"]
            
            if val > 0.8 and ent_change < 0.1:
                regime = "natural"
            elif val < 0.5 or ent_change > 0.3:
                regime = "forced"
            else:
                regime = "transition"
            
            regimes[alpha] = regime
        
        print(f"\n  Regime classification:")
        for alpha in alphas:
            print(f"    alpha={alpha:.2f}: {regimes[alpha]}")
        
        results[obj_name] = {
            "alpha_results": alpha_results,
            "regimes": regimes,
        }
    
    return results


# ===== 主函数 =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print(f"\n{'='*70}")
    print(f"Phase 448: Slot-Dominated Binding State Privatization")
    print(f"Model: {model_name}, Round: {round_num}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    # 加载模型
    t0 = time.time()
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    print(f"Model loaded: {info.model_class}, {info.n_layers} layers, d_model={info.d_model}")
    print(f"Load time: {time.time()-t0:.1f}s")
    
    # 计算类别方向
    cat_directions = {}
    for cat_name, cat_info in CATEGORY_OBJECTS.items():
        cat_dir = get_cat_direction(model, tokenizer, cat_info["cat_words"], cat_info["opp_words"])
        if cat_dir is not None:
            cat_directions[cat_name] = cat_dir
            print(f"Category direction: {cat_name}")
    
    # 运行实验
    all_results = {}
    
    if round_num >= 1:
        # 实验1: SlotMediation拆分
        r1 = experiment1_slot_decomposition(model, tokenizer, info)
        all_results["exp1_slot_decomposition"] = r1
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if round_num >= 1:
        # 实验2: 共享→私有化动力学
        r2 = experiment2_privatization_dynamics(model, tokenizer, info, cat_directions)
        all_results["exp2_privatization_dynamics"] = r2
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if round_num >= 1:
        # 实验3: Alpha区间扫描
        r3 = experiment3_alpha_regime_scan(model, tokenizer, info, cat_directions)
        all_results["exp3_alpha_regime"] = r3
    
    # 保存结果
    output_dir = "results/glm5"
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换numpy类型
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
    
    output_file = os.path.join(output_dir, f"phase448_{model_name}_r{round_num}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # 释放模型
    release_model(model)
    
    print(f"\nPhase 448 Round {round_num} complete!")


if __name__ == "__main__":
    main()
