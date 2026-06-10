"""
Phase 447: Category Binding State Decomposition
=================================================
核心假说: 类别泛化不是由可迁移纯类别方向实现,
而是由对象条件化的类别绑定态实现。

实验1: 类别绑定态分解
- 多对象同类, 收集每层自然运输delta
- PCA分解: 共享成分 vs 对象私有成分
- 关键指标: SharedRatio = Var(shared) / Var(total)

实验2: 功能等价验证
- 对同类对象训练线性映射 A_{o1→o2}
- 看映射后的绑定态是否能复现target对象的功能
- 如果线性映射有效 → 共享可变换结构

实验3: L0校准目标精确定位
- 分别测量消融后: 范数变化、方向变化、噪声方向抑制、流形有效性

实验4: 中介机制分型
- SwitchMediation / BoostMediation / IdentityMediation / SlotMediation

用法:
  python tests/glm5/phase447_binding_state_decomposition.py qwen3 1
  python tests/glm5/phase447_binding_state_decomposition.py glm4 1
  python tests/glm5/phase447_binding_state_decomposition.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import os, gc, time, json
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS)

# ===== 实验配置 =====
# 多对象同类 - 每类5-6个对象
CATEGORY_OBJECTS = {
    "fruit": {
        "objects": ["apple", "orange", "banana", "grape", "lemon", "mango"],
        "cat_words": ["fruit", "apple", "orange", "banana"],
        "opp_words": ["animal", "dog", "cat", "horse"],
        "attr_words": ["sweet", "edible", "seed", "juicy", "ripe"],
    },
    "animal": {
        "objects": ["dog", "cat", "horse", "lion", "tiger", "eagle"],
        "cat_words": ["animal", "dog", "cat", "horse"],
        "opp_words": ["fruit", "apple", "orange", "banana"],
        "attr_words": ["alive", "wild", "furry", "leg", "tail"],
    },
    "tool": {
        "objects": ["knife", "hammer", "scissors", "axe", "drill", "saw"],
        "cat_words": ["tool", "knife", "hammer", "scissors"],
        "opp_words": ["vehicle", "car", "bus", "train"],
        "attr_words": ["sharp", "metal", "handle", "cut", "heavy"],
    },
}

ALPHA = 1.0
SAMPLE_LAYERS = None  # None=所有层, int=均匀采样


def load_model_auto(model_name):
    """BF16 + device_map='auto' + flash_attention_2"""
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
    ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in word_list]
    return float(logits[ids].mean())


def collect_layer_deltas(model, tokenizer, input_ids, attention_mask, last_pos,
                        cat_dir, alpha, n_layers):
    """
    收集逐层delta (自然运输轨迹)
    
    Returns: dict {layer_idx: delta_vector_np}
    """
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
    
    # 基准运行 - 收集每层hidden state
    base_hiddens = {}
    
    def make_base_hook(li):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                base_hiddens[li] = out[0][0, last_pos].detach().float().cpu()
            else:
                base_hiddens[li] = out[0, last_pos].detach().float().cpu() if out.dim() == 3 else out.detach().float().cpu()
        return hook
    
    hooks_base = []
    for li in range(n_layers):
        hooks_base.append(layers[li].register_forward_hook(make_base_hook(li)))
    
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    for h in hooks_base:
        h.remove()
    
    # 扰动运行 - 收集每层hidden state
    pert_hiddens = {}
    embed_hook = None
    
    def make_pert_hook(li):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                pert_hiddens[li] = out[0][0, last_pos].detach().float().cpu()
            else:
                pert_hiddens[li] = out[0, last_pos].detach().float().cpu() if out.dim() == 3 else out.detach().float().cpu()
        return hook
    
    hooks_pert = []
    for li in range(n_layers):
        hooks_pert.append(layers[li].register_forward_hook(make_pert_hook(li)))
    
    # Embedding hook: 注入扰动
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
        if embed_hook is not None:
            embed_hook.remove()
        for h in hooks_pert:
            h.remove()
    
    # 计算delta
    deltas = {}
    for li in range(n_layers):
        if li in base_hiddens and li in pert_hiddens:
            delta = pert_hiddens[li] - base_hiddens[li]
            deltas[li] = delta.numpy()
    
    return deltas


# ===== 实验1: 类别绑定态分解 =====
def experiment1_binding_decomposition(model, tokenizer, info, cat_directions):
    """类别绑定态分解: 共享成分 vs 对象私有成分"""
    print(f"\n{'='*60}")
    print("Experiment 1: Category Binding State Decomposition")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    results = {}
    
    for cat_name, cat_info in CATEGORY_OBJECTS.items():
        print(f"\n  Category: {cat_name}")
        obj_names = cat_info["objects"]
        cat_words = cat_info["cat_words"]
        opp_words = cat_info["opp_words"]
        
        if cat_name not in cat_directions:
            print(f"    No direction for {cat_name}, skipping")
            continue
        
        cat_dir = cat_directions[cat_name]
        
        # 收集每个对象的逐层delta
        all_deltas = {}  # {obj_name: {layer_idx: delta_np}}
        
        for oi, obj_name in enumerate(obj_names):
            text = f"The {obj_name} is a"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_device = next(model.parameters()).device
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            print(f"    [{oi+1}/{len(obj_names)}] {obj_name} - collecting deltas...")
            t0 = time.time()
            
            deltas = collect_layer_deltas(model, tokenizer, input_ids, attention_mask,
                                         last_pos, cat_dir, ALPHA, n_layers)
            all_deltas[obj_name] = deltas
            
            print(f"      Collected {len(deltas)} layers in {time.time()-t0:.1f}s")
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 逐层分解: PCA + 共享/私有方差比
        layer_decomposition = {}
        
        # 采样层 (避免太多层导致内存问题)
        all_layer_indices = sorted(set().union(*[set(d.keys()) for d in all_deltas.values()]))
        if SAMPLE_LAYERS and len(all_layer_indices) > SAMPLE_LAYERS:
            step = len(all_layer_indices) // SAMPLE_LAYERS
            sample_layers = all_layer_indices[::step] + [all_layer_indices[-1]]
            sample_layers = sorted(set(sample_layers))
        else:
            sample_layers = all_layer_indices
        
        for li in sample_layers:
            # 收集该层所有对象的delta
            layer_deltas = []
            valid_objs = []
            for obj_name in obj_names:
                if obj_name in all_deltas and li in all_deltas[obj_name]:
                    d = all_deltas[obj_name][li]
                    if np.linalg.norm(d) > 1e-8:
                        layer_deltas.append(d)
                        valid_objs.append(obj_name)
            
            if len(layer_deltas) < 3:
                continue
            
            # 堆叠为矩阵 [n_objects, d_model]
            delta_matrix = np.stack(layer_deltas)
            
            # 方法1: 均值 = 共享方向, 残差 = 私有成分
            shared_direction = delta_matrix.mean(axis=0)  # [d_model]
            shared_norm = np.linalg.norm(shared_direction)
            
            residuals = delta_matrix - shared_direction  # [n_obj, d_model]
            private_norms = [np.linalg.norm(r) for r in residuals]
            
            total_var = np.sum(delta_matrix ** 2)
            shared_var = np.sum(shared_direction ** 2) * len(layer_deltas)  # 广播回总方差
            private_var = np.sum(residuals ** 2)
            
            shared_ratio = shared_var / total_var if total_var > 1e-10 else 0
            private_ratio = private_var / total_var if total_var > 1e-10 else 0
            
            # 方法2: PCA - 第一主成分解释的方差比
            try:
                # 中心化
                centered = delta_matrix - delta_matrix.mean(axis=0)
                if centered.shape[0] > 1 and centered.shape[1] > 1:
                    # 用SVD做PCA (避免大矩阵)
                    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                    pca1_ratio = float(S[0]**2 / np.sum(S**2)) if np.sum(S**2) > 0 else 0
                    pca2_ratio = float(S[1]**2 / np.sum(S**2)) if len(S) > 1 and np.sum(S**2) > 0 else 0
                    pca_top3_ratio = float(np.sum(S[:min(3,len(S))]**2) / np.sum(S**2)) if np.sum(S**2) > 0 else 0
                    n_significant = int(np.sum(S > S[0] * 0.1)) if len(S) > 0 else 0
                else:
                    pca1_ratio = 0
                    pca2_ratio = 0
                    pca_top3_ratio = 0
                    n_significant = 0
            except Exception as e:
                pca1_ratio = 0
                pca2_ratio = 0
                pca_top3_ratio = 0
                n_significant = 0
            
            # 方法3: 两两余弦相似度
            pair_cosines = []
            for i in range(len(layer_deltas)):
                for j in range(i+1, len(layer_deltas)):
                    ni = np.linalg.norm(layer_deltas[i])
                    nj = np.linalg.norm(layer_deltas[j])
                    if ni > 1e-8 and nj > 1e-8:
                        cos_ij = float(np.dot(layer_deltas[i], layer_deltas[j]) / (ni * nj))
                        pair_cosines.append(cos_ij)
            
            avg_pair_cos = float(np.mean(pair_cosines)) if pair_cosines else 0
            min_pair_cos = float(np.min(pair_cosines)) if pair_cosines else 0
            max_pair_cos = float(np.max(pair_cosines)) if pair_cosines else 0
            
            # 共享方向的类别投影
            if shared_norm > 1e-8:
                cat_dir_np = cat_dir.numpy() if isinstance(cat_dir, torch.Tensor) else cat_dir
                shared_cat_proj = float(np.dot(shared_direction, cat_dir_np) / shared_norm)
            else:
                shared_cat_proj = 0
            
            layer_decomposition[f"L{li}"] = {
                "n_objects": len(valid_objs),
                "shared_ratio": round(float(shared_ratio), 4),
                "private_ratio": round(float(private_ratio), 4),
                "pca1_ratio": round(float(pca1_ratio), 4),
                "pca2_ratio": round(float(pca2_ratio), 4),
                "pca_top3_ratio": round(float(pca_top3_ratio), 4),
                "n_significant_pcs": n_significant,
                "avg_pair_cosine": round(float(avg_pair_cos), 4),
                "min_pair_cosine": round(float(min_pair_cos), 4),
                "max_pair_cosine": round(float(max_pair_cos), 4),
                "shared_cat_proj": round(float(shared_cat_proj), 4),
                "avg_delta_norm": round(float(np.mean([np.linalg.norm(d) for d in layer_deltas])), 4),
                "shared_norm": round(float(shared_norm), 4),
                "avg_private_norm": round(float(np.mean(private_norms)), 4),
            }
        
        results[cat_name] = {
            "n_objects": len(obj_names),
            "objects": obj_names,
            "layer_decomposition": layer_decomposition,
        }
        
        print(f"\n  {cat_name} Summary:")
        key_layers = sorted(layer_decomposition.keys(), key=lambda x: int(x[1:]))
        for lk in key_layers[::max(1, len(key_layers)//6)]:
            ld = layer_decomposition[lk]
            print(f"    {lk}: shared={ld['shared_ratio']:.3f}, pca1={ld['pca1_ratio']:.3f}, "
                  f"pair_cos={ld['avg_pair_cosine']:.3f}, n_sig={ld['n_significant_pcs']}")
    
    return results


# ===== 实验2: 功能等价验证 (线性映射) =====
def experiment2_functional_equivalence(model, tokenizer, info, cat_directions):
    """功能等价验证: 同类对象的类别绑定态之间是否有线性变换"""
    print(f"\n{'='*60}")
    print("Experiment 2: Functional Equivalence Verification")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    W_U = get_W_U(model, info.name)
    results = {}
    
    for cat_name, cat_info in CATEGORY_OBJECTS.items():
        if cat_name not in cat_directions:
            continue
        
        obj_names = cat_info["objects"]
        cat_words = cat_info["cat_words"]
        opp_words = cat_info["opp_words"]
        attr_words = cat_info["attr_words"]
        cat_dir = cat_directions[cat_name]
        
        print(f"\n  Category: {cat_name}")
        
        # 收集每个对象在中间层的delta + 基准logits
        mid_layer = n_layers // 2
        obj_deltas = {}
        obj_base_logits = {}
        obj_base_cat_gap = {}
        
        for obj_name in obj_names:
            text = f"The {obj_name} is a"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_device = next(model.parameters()).device
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            last_pos = input_ids.shape[1] - 1
            
            # 基准logits
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[0, -1].float().cpu().numpy()
            cat_gap = get_logit_values(logits, tokenizer, cat_words) - get_logit_values(logits, tokenizer, opp_words)
            obj_base_logits[obj_name] = logits
            obj_base_cat_gap[obj_name] = cat_gap
            
            # 收集delta
            deltas = collect_layer_deltas(model, tokenizer, input_ids, attention_mask,
                                         last_pos, cat_dir, ALPHA, n_layers)
            if mid_layer in deltas:
                obj_deltas[obj_name] = deltas[mid_layer]
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if len(obj_deltas) < 3:
            print(f"    Not enough valid deltas ({len(obj_deltas)}), skipping")
            continue
        
        # 功能等价测试:
        # 对每对对象, 测量delta之间是否能通过线性变换恢复功能
        pair_results = {}
        obj_list = list(obj_deltas.keys())
        
        for i in range(len(obj_list)):
            for j in range(i+1, len(obj_list)):
                o1, o2 = obj_list[i], obj_list[j]
                d1, d2 = obj_deltas[o1], obj_deltas[o2]
                
                n1 = np.linalg.norm(d1)
                n2 = np.linalg.norm(d2)
                
                if n1 < 1e-8 or n2 < 1e-8:
                    continue
                
                # 余弦相似度
                cos_12 = float(np.dot(d1, d2) / (n1 * n2))
                
                # 范数比
                norm_ratio = n2 / n1 if n1 > 1e-8 else 0
                
                # 缩放后余弦 (按范数对齐后)
                d2_scaled = d1 * norm_ratio
                cos_scaled = float(np.dot(d2_scaled, d2) / (np.linalg.norm(d2_scaled) * n2)) if np.linalg.norm(d2_scaled) > 1e-8 else 0
                
                # 投影: d1在d2方向上的投影占d2的比例
                proj_ratio = float(np.dot(d1, d2) / (n2 ** 2))
                
                # 重建误差: 用d1重建d2
                d2_reconstructed = d1 * proj_ratio
                recon_error = float(np.linalg.norm(d2 - d2_reconstructed) / n2)
                
                # 功能测试: 在W_U空间中的效果
                # d1和d2分别投影到W_U的读出空间
                logit_shift_d1 = W_U @ d1  # [vocab]
                logit_shift_d2 = W_U @ d2  # [vocab]
                
                # 读出余弦: logit空间中的方向一致性
                ls1_norm = np.linalg.norm(logit_shift_d1)
                ls2_norm = np.linalg.norm(logit_shift_d2)
                if ls1_norm > 1e-8 and ls2_norm > 1e-8:
                    logit_cos = float(np.dot(logit_shift_d1, logit_shift_d2) / (ls1_norm * ls2_norm))
                else:
                    logit_cos = 0
                
                # 类别方向在logit空间中的投影
                cat_logit_shift_d1 = float(np.mean(logit_shift_d1[
                    [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
                ]))
                cat_logit_shift_d2 = float(np.mean(logit_shift_d2[
                    [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
                ]))
                
                # 属性方向在logit空间中的投影
                attr_logit_shift_d1 = float(np.mean(logit_shift_d1[
                    [tokenizer.encode(w, add_special_tokens=False)[0] for w in attr_words]
                ])) if all(tokenizer.encode(w, add_special_tokens=False) for w in attr_words) else 0
                attr_logit_shift_d2 = float(np.mean(logit_shift_d2[
                    [tokenizer.encode(w, add_special_tokens=False)[0] for w in attr_words]
                ])) if all(tokenizer.encode(w, add_special_tokens=False) for w in attr_words) else 0
                
                pair_key = f"{o1}_vs_{o2}"
                pair_results[pair_key] = {
                    "cosine": round(cos_12, 4),
                    "norm_ratio": round(float(norm_ratio), 4),
                    "recon_error": round(recon_error, 4),
                    "logit_cos": round(float(logit_cos), 4),
                    "cat_logit_shift_d1": round(float(cat_logit_shift_d1), 4),
                    "cat_logit_shift_d2": round(float(cat_logit_shift_d2), 4),
                    "attr_logit_shift_d1": round(float(attr_logit_shift_d1), 4),
                    "attr_logit_shift_d2": round(float(attr_logit_shift_d2), 4),
                }
        
        # 汇总
        all_cosines = [v["cosine"] for v in pair_results.values()]
        all_logit_cos = [v["logit_cos"] for v in pair_results.values()]
        all_recon_errors = [v["recon_error"] for v in pair_results.values()]
        
        results[cat_name] = {
            "mid_layer": mid_layer,
            "n_pairs": len(pair_results),
            "avg_cosine": round(float(np.mean(all_cosines)), 4) if all_cosines else 0,
            "avg_logit_cos": round(float(np.mean(all_logit_cos)), 4) if all_logit_cos else 0,
            "avg_recon_error": round(float(np.mean(all_recon_errors)), 4) if all_recon_errors else 0,
            "pair_details": pair_results,
        }
        
        print(f"    n_pairs={len(pair_results)}, avg_cos={np.mean(all_cosines):.3f}, "
              f"avg_logit_cos={np.mean(all_logit_cos):.3f}, avg_recon={np.mean(all_recon_errors):.3f}")
    
    return results


# ===== 实验3: L0校准目标精确定位 =====
def experiment3_l0_calibration_detail(model, tokenizer, info, cat_directions):
    """L0 attention校准目标: 到底校准了什么?"""
    print(f"\n{'='*60}")
    print("Experiment 3: L0 Attention Calibration Detail")
    print(f"{'='*60}")
    
    n_layers = info.n_layers
    W_U = get_W_U(model, info.name)
    results = {}
    
    test_objects = [
        ("apple", "fruit", ["fruit", "apple", "orange", "banana"], ["animal", "dog", "cat", "horse"]),
        ("dog", "animal", ["animal", "dog", "cat", "horse"], ["fruit", "apple", "orange", "banana"]),
        ("knife", "tool", ["tool", "knife", "hammer", "scissors"], ["vehicle", "car", "bus", "train"]),
    ]
    
    for obj_name, cat_name, cat_words, opp_words in test_objects:
        if cat_name not in cat_directions:
            continue
        
        cat_dir = cat_directions[cat_name]
        
        text = f"The {obj_name} is a"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        last_pos = input_ids.shape[1] - 1
        
        print(f"\n  {obj_name} ({cat_name}):")
        
        # --- 基准: 无扰动 ---
        with torch.no_grad():
            base_out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
        base_logits = base_out.logits[0, -1].float().cpu().numpy()
        base_last = base_out.hidden_states[-1][0, last_pos].detach().float().cpu().numpy()
        
        # --- 扰动: 无消融 ---
        perturb_vec = (ALPHA * cat_dir).to(input_device).to(torch.bfloat16)
        embed_hook = None
        
        def on_embed(module, inp, out):
            if isinstance(out, torch.Tensor):
                out = out.clone()
                out[0, last_pos] = out[0, last_pos] + perturb_vec
            return out
        
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embed_hook = model.model.embed_tokens.register_forward_hook(on_embed)
        
        with torch.no_grad():
            pert_out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
        if embed_hook:
            embed_hook.remove()
        
        pert_logits = pert_out.logits[0, -1].float().cpu().numpy()
        pert_last = pert_out.hidden_states[-1][0, last_pos].detach().float().cpu().numpy()
        
        orig_delta = pert_last - base_last
        orig_delta_norm = np.linalg.norm(orig_delta)
        
        # --- 扰动 + L0 attention消融 ---
        embed_hook2 = None
        layers = get_layers(model)
        
        def on_embed2(module, inp, out):
            if isinstance(out, torch.Tensor):
                out = out.clone()
                out[0, last_pos] = out[0, last_pos] + perturb_vec
            return out
        
        def ablation_hook(module, inp, out):
            if isinstance(out, tuple):
                zero_hidden = torch.zeros_like(out[0])
                return (zero_hidden,) + out[1:]
            return torch.zeros_like(out)
        
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embed_hook2 = model.model.embed_tokens.register_forward_hook(on_embed2)
        abl_hook = layers[0].self_attn.register_forward_hook(ablation_hook)
        
        with torch.no_grad():
            abl_out = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
        if embed_hook2:
            embed_hook2.remove()
        abl_hook.remove()
        
        abl_logits = abl_out.logits[0, -1].float().cpu().numpy()
        abl_last = abl_out.hidden_states[-1][0, last_pos].detach().float().cpu().numpy()
        
        abl_delta = abl_last - base_last
        abl_delta_norm = np.linalg.norm(abl_delta)
        
        # ===== 详细分析 =====
        cat_dir_np = cat_dir.numpy() if isinstance(cat_dir, torch.Tensor) else cat_dir
        
        # 1. 范数变化
        norm_ratio = abl_delta_norm / orig_delta_norm if orig_delta_norm > 1e-8 else 0
        
        # 2. 方向变化
        if orig_delta_norm > 1e-8 and abl_delta_norm > 1e-8:
            direction_cos = float(np.dot(orig_delta, abl_delta) / (orig_delta_norm * abl_delta_norm))
        else:
            direction_cos = 0
        
        # 3. 类别方向投影
        cat_proj_orig = float(np.dot(orig_delta, cat_dir_np) / orig_delta_norm) if orig_delta_norm > 1e-8 else 0
        cat_proj_abl = float(np.dot(abl_delta, cat_dir_np) / abl_delta_norm) if abl_delta_norm > 1e-8 else 0
        cat_proj_change = cat_proj_abl - cat_proj_orig
        
        # 4. 噪声方向投影 (随机方向的平均投影)
        rng = np.random.RandomState(42)
        n_random_dirs = 50
        random_projs_orig = []
        random_projs_abl = []
        for _ in range(n_random_dirs):
            rand_dir = rng.randn(len(cat_dir_np))
            rand_dir = rand_dir / np.linalg.norm(rand_dir)
            # 正交化到cat_dir
            rand_dir = rand_dir - np.dot(rand_dir, cat_dir_np) * cat_dir_np
            if np.linalg.norm(rand_dir) > 1e-8:
                rand_dir = rand_dir / np.linalg.norm(rand_dir)
                rp_orig = float(np.dot(orig_delta, rand_dir) / orig_delta_norm) if orig_delta_norm > 1e-8 else 0
                rp_abl = float(np.dot(abl_delta, rand_dir) / abl_delta_norm) if abl_delta_norm > 1e-8 else 0
                random_projs_orig.append(abs(rp_orig))
                random_projs_abl.append(abs(rp_abl))
        
        noise_proj_orig = float(np.mean(random_projs_orig))
        noise_proj_abl = float(np.mean(random_projs_abl))
        noise_suppression = noise_proj_orig / noise_proj_abl if noise_proj_abl > 1e-8 else 0
        
        # 5. Logit空间分析
        logit_shift_orig = W_U @ orig_delta
        logit_shift_abl = W_U @ abl_delta
        
        # 类别词logit变化
        cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
        opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
        
        cat_logit_shift_orig = float(np.mean(logit_shift_orig[cat_ids]))
        cat_logit_shift_abl = float(np.mean(logit_shift_abl[cat_ids]))
        opp_logit_shift_orig = float(np.mean(logit_shift_orig[opp_ids]))
        opp_logit_shift_abl = float(np.mean(logit_shift_abl[opp_ids]))
        
        # 非类别词logit变化 (随机采样50个词)
        vocab_size = W_U.shape[0]
        random_word_ids = rng.choice(vocab_size, size=50, replace=False)
        random_logit_shift_orig = float(np.mean(np.abs(logit_shift_orig[random_word_ids])))
        random_logit_shift_abl = float(np.mean(np.abs(logit_shift_abl[random_word_ids])))
        
        # 6. Entropy分析
        def softmax_entropy(logits_vec):
            exp_l = np.exp(logits_vec - np.max(logits_vec))
            probs = exp_l / exp_l.sum()
            return -float(np.sum(probs * np.log(probs + 1e-10)))
        
        entropy_base = softmax_entropy(base_logits)
        entropy_pert = softmax_entropy(pert_logits)
        entropy_abl = softmax_entropy(abl_logits)
        
        # 7. Top-1候选变化
        top1_base = int(np.argmax(base_logits))
        top1_pert = int(np.argmax(pert_logits))
        top1_abl = int(np.argmax(abl_logits))
        
        top5_base = set(np.argsort(base_logits)[-5:])
        top5_pert = set(np.argsort(pert_logits)[-5:])
        top5_abl = set(np.argsort(abl_logits)[-5:])
        top5_overlap_pert_base = len(top5_pert & top5_base) / 5
        top5_overlap_abl_base = len(top5_abl & top5_base) / 5
        
        results[obj_name] = {
            "category": cat_name,
            "norm": {
                "orig_delta_norm": round(float(orig_delta_norm), 4),
                "abl_delta_norm": round(float(abl_delta_norm), 4),
                "norm_ratio_abl_orig": round(float(norm_ratio), 4),
            },
            "direction": {
                "direction_cos": round(float(direction_cos), 4),
                "cat_proj_orig": round(float(cat_proj_orig), 4),
                "cat_proj_abl": round(float(cat_proj_abl), 4),
                "cat_proj_change": round(float(cat_proj_change), 4),
            },
            "noise_suppression": {
                "noise_proj_orig": round(noise_proj_orig, 6),
                "noise_proj_abl": round(noise_proj_abl, 6),
                "suppression_ratio": round(float(noise_suppression), 4),
            },
            "logit_analysis": {
                "cat_logit_shift_orig": round(float(cat_logit_shift_orig), 4),
                "cat_logit_shift_abl": round(float(cat_logit_shift_abl), 4),
                "opp_logit_shift_orig": round(float(opp_logit_shift_orig), 4),
                "opp_logit_shift_abl": round(float(opp_logit_shift_abl), 4),
                "random_logit_shift_orig": round(float(random_logit_shift_orig), 4),
                "random_logit_shift_abl": round(float(random_logit_shift_abl), 4),
            },
            "entropy": {
                "base": round(float(entropy_base), 4),
                "pert": round(float(entropy_pert), 4),
                "abl": round(float(entropy_abl), 4),
                "pert_delta": round(float(entropy_pert - entropy_base), 4),
                "abl_delta": round(float(entropy_abl - entropy_base), 4),
            },
            "top_candidate": {
                "top1_base": tokenizer.decode([top1_base]).strip(),
                "top1_pert": tokenizer.decode([top1_pert]).strip(),
                "top1_abl": tokenizer.decode([top1_abl]).strip(),
                "top5_overlap_pert_base": round(float(top5_overlap_pert_base), 4),
                "top5_overlap_abl_base": round(float(top5_overlap_abl_base), 4),
            },
        }
        
        print(f"    norm_ratio={norm_ratio:.2f}, dir_cos={direction_cos:.3f}, "
              f"cat_proj_change={cat_proj_change:.3f}, noise_supp={noise_suppression:.2f}")
        print(f"    entropy: base={entropy_base:.2f}, pert={entropy_pert:.2f}, abl={entropy_abl:.2f}")
        print(f"    cat_logit: orig={cat_logit_shift_orig:.3f}, abl={cat_logit_shift_abl:.3f}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


# ===== 实验4: 中介机制分型 =====
def experiment4_mediation_typing(model, tokenizer, info, cat_directions):
    """中介机制分型: Switch/Boost/Identity/Slot"""
    print(f"\n{'='*60}")
    print("Experiment 4: Mediation Mechanism Typing")
    print(f"{'='*60}")
    
    W_U = get_W_U(model, info.name)
    results = {}
    
    test_cases = [
        {
            "obj": "apple", "cat": "fruit",
            "cat_words": ["fruit", "apple", "orange", "banana"],
            "opp_words": ["animal", "dog", "cat", "horse"],
            "related_attrs": ["sweet", "edible", "seed", "juicy"],
            "unrelated_attrs": ["fast", "loud", "metal", "engine"],
        },
        {
            "obj": "dog", "cat": "animal",
            "cat_words": ["animal", "dog", "cat", "horse"],
            "opp_words": ["fruit", "apple", "orange", "banana"],
            "related_attrs": ["alive", "wild", "furry", "leg"],
            "unrelated_attrs": ["sweet", "round", "metal", "sharp"],
        },
        {
            "obj": "knife", "cat": "tool",
            "cat_words": ["tool", "knife", "hammer", "scissors"],
            "opp_words": ["vehicle", "car", "bus", "train"],
            "related_attrs": ["sharp", "metal", "cut", "blade"],
            "unrelated_attrs": ["sweet", "alive", "fast", "round"],
        },
    ]
    
    for tc in test_cases:
        obj_name = tc["obj"]
        cat_name = tc["cat"]
        cat_words = tc["cat_words"]
        opp_words = tc["opp_words"]
        related_attrs = tc["related_attrs"]
        unrelated_attrs = tc["unrelated_attrs"]
        
        if cat_name not in cat_directions:
            continue
        
        cat_dir = cat_directions[cat_name]
        
        text = f"The {obj_name} is a"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        
        print(f"\n  {obj_name} ({cat_name}):")
        
        # 1. SwitchMediation: push类别到对立方向, 测属性是否切换
        # 2. BoostMediation: push类别增强方向, 测属性是否增强
        # 3. IdentityMediation: 替换对象名, 测属性变化
        # 4. SlotMediation: 改变关系槽位(问题), 测属性变化
        
        mediation_scores = {}
        
        # --- 基准logits ---
        with torch.no_grad():
            base_out = model(input_ids=input_ids, attention_mask=attention_mask)
        base_logits = base_out.logits[0, -1].float().cpu().numpy()
        
        def get_attr_logits(logits_vec, attr_list):
            """获取属性词的平均logit"""
            ids = []
            for w in attr_list:
                tids = tokenizer.encode(w, add_special_tokens=False)
                if tids:
                    ids.append(tids[0])
            if not ids:
                return 0.0
            return float(np.mean(logits_vec[ids]))
        
        base_related_logit = get_attr_logits(base_logits, related_attrs)
        base_unrelated_logit = get_attr_logits(base_logits, unrelated_attrs)
        base_cat_logit = get_logit_values(base_logits, tokenizer, cat_words)
        base_opp_logit = get_logit_values(base_logits, tokenizer, opp_words)
        
        # --- SwitchMediation ---
        # push到对立类别方向
        opp_dir = -cat_dir  # 反方向 = 推向对立类别
        switch_scores = []
        for alpha in [0.5, 1.0, 1.5]:
            perturb_vec = (alpha * opp_dir).to(input_device).to(torch.bfloat16)
            embed_hook = None
            
            def make_switch_hook(pv, lp):
                def hook(module, inp, out):
                    if isinstance(out, torch.Tensor):
                        out = out.clone()
                        out[0, lp] = out[0, lp] + pv
                    return out
                return hook
            
            last_pos = input_ids.shape[1] - 1
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                embed_hook = model.model.embed_tokens.register_forward_hook(
                    make_switch_hook(perturb_vec, last_pos))
            
            with torch.no_grad():
                switch_out = model(input_ids=input_ids, attention_mask=attention_mask)
            if embed_hook:
                embed_hook.remove()
            
            switch_logits = switch_out.logits[0, -1].float().cpu().numpy()
            switch_related = get_attr_logits(switch_logits, related_attrs)
            switch_unrelated = get_attr_logits(switch_logits, unrelated_attrs)
            
            # SwitchMediation: 属性是否跟随类别切换
            # 如果push到对立类别, related属性应该下降, unrelated属性应该上升
            switch_med = (base_related_logit - switch_related) - (base_unrelated_logit - switch_unrelated)
            switch_scores.append(switch_med)
        
        mediation_scores["SwitchMediation"] = {
            "alpha_0.5": round(float(switch_scores[0]), 4),
            "alpha_1.0": round(float(switch_scores[1]), 4),
            "alpha_1.5": round(float(switch_scores[2]), 4),
            "avg": round(float(np.mean(switch_scores)), 4),
        }
        
        # --- BoostMediation ---
        # push增强当前类别方向
        boost_scores = []
        for alpha in [0.5, 1.0, 1.5]:
            perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
            embed_hook = None
            
            def make_boost_hook(pv, lp):
                def hook(module, inp, out):
                    if isinstance(out, torch.Tensor):
                        out = out.clone()
                        out[0, lp] = out[0, lp] + pv
                    return out
                return hook
            
            last_pos = input_ids.shape[1] - 1
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                embed_hook = model.model.embed_tokens.register_forward_hook(
                    make_boost_hook(perturb_vec, last_pos))
            
            with torch.no_grad():
                boost_out = model(input_ids=input_ids, attention_mask=attention_mask)
            if embed_hook:
                embed_hook.remove()
            
            boost_logits = boost_out.logits[0, -1].float().cpu().numpy()
            boost_related = get_attr_logits(boost_logits, related_attrs)
            boost_unrelated = get_attr_logits(boost_logits, unrelated_attrs)
            
            # BoostMediation: 增强类别时, related属性是否也增强
            boost_med = (boost_related - base_related_logit) - (boost_unrelated - base_unrelated_logit)
            boost_scores.append(boost_med)
        
        mediation_scores["BoostMediation"] = {
            "alpha_0.5": round(float(boost_scores[0]), 4),
            "alpha_1.0": round(float(boost_scores[1]), 4),
            "alpha_1.5": round(float(boost_scores[2]), 4),
            "avg": round(float(np.mean(boost_scores)), 4),
        }
        
        # --- IdentityMediation ---
        # 替换对象名: "The apple is a" → "The orange is a"
        identity_obj = {"apple": "orange", "dog": "cat", "knife": "hammer"}
        if obj_name in identity_obj:
            alt_obj = identity_obj[obj_name]
            alt_text = f"The {alt_obj} is a"
            alt_inputs = tokenizer(alt_text, return_tensors="pt", truncation=True, max_length=64)
            alt_input_ids = alt_inputs["input_ids"].to(input_device)
            alt_attention_mask = alt_inputs["attention_mask"].to(input_device)
            
            with torch.no_grad():
                alt_out = model(input_ids=alt_input_ids, attention_mask=alt_attention_mask)
            alt_logits = alt_out.logits[0, -1].float().cpu().numpy()
            
            alt_related = get_attr_logits(alt_logits, related_attrs)
            alt_unrelated = get_attr_logits(alt_logits, unrelated_attrs)
            
            # IdentityMediation: 对象替换后属性变化
            identity_med = abs(alt_related - base_related_logit) - abs(alt_unrelated - base_unrelated_logit)
            
            mediation_scores["IdentityMediation"] = {
                "identity_shift": round(float(identity_med), 4),
                "related_change": round(float(alt_related - base_related_logit), 4),
                "unrelated_change": round(float(alt_unrelated - base_unrelated_logit), 4),
            }
        
        # --- SlotMediation ---
        # 改变关系槽位: "The {obj} is a" → "The {obj} has a" → "The {obj} feels"
        slot_templates = {
            "is_a": f"The {obj_name} is a",
            "has_a": f"The {obj_name} has a",
            "feels": f"The {obj_name} feels",
            "is_made": f"The {obj_name} is made of",
        }
        
        slot_results = {}
        for slot_name, slot_text in slot_templates.items():
            slot_inputs = tokenizer(slot_text, return_tensors="pt", truncation=True, max_length=64)
            slot_input_ids = slot_inputs["input_ids"].to(input_device)
            slot_attention_mask = slot_inputs["attention_mask"].to(input_device)
            
            with torch.no_grad():
                slot_out = model(input_ids=slot_input_ids, attention_mask=slot_attention_mask)
            slot_logits = slot_out.logits[0, -1].float().cpu().numpy()
            
            slot_related = get_attr_logits(slot_logits, related_attrs)
            slot_unrelated = get_attr_logits(slot_logits, unrelated_attrs)
            
            slot_results[slot_name] = {
                "related": round(float(slot_related), 4),
                "unrelated": round(float(slot_unrelated), 4),
            }
        
        # SlotMediation: 不同槽位对属性读出的影响
        slot_med = max(slot_results.values(), key=lambda x: x["related"])["related"] - \
                   min(slot_results.values(), key=lambda x: x["related"])["related"]
        
        mediation_scores["SlotMediation"] = {
            "range": round(float(slot_med), 4),
            "slots": slot_results,
        }
        
        results[obj_name] = {
            "category": cat_name,
            "base_related_attr_logit": round(float(base_related_logit), 4),
            "base_unrelated_attr_logit": round(float(base_unrelated_logit), 4),
            "mediation_scores": mediation_scores,
        }
        
        print(f"    Switch={mediation_scores['SwitchMediation']['avg']:.4f}, "
              f"Boost={mediation_scores['BoostMediation']['avg']:.4f}, "
              f"Identity={mediation_scores.get('IdentityMediation', {}).get('identity_shift', 'N/A')}, "
              f"Slot={mediation_scores['SlotMediation']['range']:.4f}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


# ===== 主函数 =====
def run_experiment(model_name, round_num):
    print(f"\n{'='*70}")
    print(f"Phase 447: Category Binding State Decomposition")
    print(f"Model: {model_name}, Round: {round_num}")
    print(f"{'='*70}")
    
    # 加载模型
    print("\n[0] Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  Loaded: {info.model_class}, {n_layers} layers, d_model={info.d_model}")
    print(f"  Load time: {time.time()-t0:.1f}s")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # 计算类别方向
    print("\n[1] Computing category directions...")
    cat_directions = {}
    for cat_name, cat_info in CATEGORY_OBJECTS.items():
        d = get_cat_direction(model, tokenizer, cat_info["cat_words"], cat_info["opp_words"])
        if d is not None:
            cat_directions[cat_name] = d
            print(f"  {cat_name}: direction computed")
    
    # 依次运行4个实验
    all_results = {}
    
    # 实验1: 绑定态分解
    print("\n" + "="*70)
    print("Running Experiment 1: Binding State Decomposition")
    t1 = time.time()
    exp1_results = experiment1_binding_decomposition(model, tokenizer, info, cat_directions)
    all_results["exp1_binding_decomposition"] = exp1_results
    print(f"  Exp1 time: {time.time()-t1:.1f}s")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 实验2: 功能等价验证
    print("\n" + "="*70)
    print("Running Experiment 2: Functional Equivalence")
    t2 = time.time()
    exp2_results = experiment2_functional_equivalence(model, tokenizer, info, cat_directions)
    all_results["exp2_functional_equivalence"] = exp2_results
    print(f"  Exp2 time: {time.time()-t2:.1f}s")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 实验3: L0校准细节
    print("\n" + "="*70)
    print("Running Experiment 3: L0 Calibration Detail")
    t3 = time.time()
    exp3_results = experiment3_l0_calibration_detail(model, tokenizer, info, cat_directions)
    all_results["exp3_l0_calibration_detail"] = exp3_results
    print(f"  Exp3 time: {time.time()-t3:.1f}s")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 实验4: 中介分型
    print("\n" + "="*70)
    print("Running Experiment 4: Mediation Typing")
    t4 = time.time()
    exp4_results = experiment4_mediation_typing(model, tokenizer, info, cat_directions)
    all_results["exp4_mediation_typing"] = exp4_results
    print(f"  Exp4 time: {time.time()-t4:.1f}s")
    
    # 汇总
    print(f"\n{'='*70}")
    print("PHASE 447 SUMMARY")
    print(f"{'='*70}")
    
    # Exp1汇总
    print("\n  Exp1 - Binding Decomposition:")
    for cat_name, cat_data in exp1_results.items():
        layers_data = cat_data.get("layer_decomposition", {})
        if layers_data:
            # 取中间层和最后层
            key_layers = sorted(layers_data.keys(), key=lambda x: int(x[1:]))
            mid_idx = len(key_layers) // 2
            for lk in [key_layers[0], key_layers[mid_idx], key_layers[-1]]:
                ld = layers_data[lk]
                print(f"    {cat_name} {lk}: shared={ld['shared_ratio']:.3f}, "
                      f"pca1={ld['pca1_ratio']:.3f}, pair_cos={ld['avg_pair_cosine']:.3f}")
    
    # Exp2汇总
    print("\n  Exp2 - Functional Equivalence:")
    for cat_name, cat_data in exp2_results.items():
        print(f"    {cat_name}: avg_cos={cat_data['avg_cosine']:.3f}, "
              f"avg_logit_cos={cat_data['avg_logit_cos']:.3f}, "
              f"avg_recon={cat_data['avg_recon_error']:.3f}")
    
    # Exp3汇总
    print("\n  Exp3 - L0 Calibration:")
    for obj_name, obj_data in exp3_results.items():
        print(f"    {obj_name}: norm_ratio={obj_data['norm']['norm_ratio_abl_orig']:.2f}, "
              f"dir_cos={obj_data['direction']['direction_cos']:.3f}, "
              f"noise_supp={obj_data['noise_suppression']['suppression_ratio']:.2f}, "
              f"entropy_abl_delta={obj_data['entropy']['abl_delta']:.2f}")
    
    # Exp4汇总
    print("\n  Exp4 - Mediation Typing:")
    for obj_name, obj_data in exp4_results.items():
        ms = obj_data["mediation_scores"]
        print(f"    {obj_name}: Switch={ms['SwitchMediation']['avg']:.4f}, "
              f"Boost={ms['BoostMediation']['avg']:.4f}, "
              f"Slot={ms['SlotMediation']['range']:.4f}")
    
    # 保存
    output = {
        "model": model_name,
        "round": round_num,
        "n_layers": n_layers,
        "alpha": ALPHA,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiments": all_results,
    }
    
    # numpy类型转换
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
    
    # 释放模型
    print("\n  Releasing model...")
    release_model(model)
    model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run_experiment(model_name, round_num)
