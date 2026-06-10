"""
Phase 446: Natural Transport Transfer (跨对象运输迁移)
========================================================
目标: 用自然运输方式(输入层扰动→让模型自然传播)重新测试跨对象迁移

Phase 442失败的原因是最后层delta注入方法不够好。
本实验改用: 在source对象输入层注入类别扰动,记录自然运输到每层的delta,
然后把整个delta轨迹在target对象的对应层注入,测量迁移效果。

方法:
1. 对source对象做输入层类别扰动,记录每层last token的delta
2. 对target对象,在每层注入source的对应层delta
3. 测量迁移后target的类别logit gap变化
4. 比较: 同类迁移 vs 跨类迁移 vs 随机对照

用法:
  python tests/glm5/phase446_natural_transport_transfer.py qwen3 1
  python tests/glm5/phase446_natural_transport_transfer.py glm4 1
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
PAIRS = {
    "same_class": [
        ("apple", "orange", "fruit"),
        ("knife", "hammer", "tool"),
        ("dog", "cat", "animal"),
    ],
    "cross_class": [
        ("apple", "knife", "fruit", "tool"),
        ("dog", "apple", "animal", "fruit"),
        ("knife", "dog", "tool", "animal"),
    ],
}

OBJECTS = {
    "apple": {"template": "The {obj} is a", "cat_words": ["fruit", "apple", "orange", "banana"],
              "opp_words": ["animal", "dog", "cat", "horse"]},
    "orange": {"template": "The {obj} is a", "cat_words": ["fruit", "apple", "orange", "banana"],
               "opp_words": ["animal", "dog", "cat", "horse"]},
    "knife": {"template": "The {obj} is a", "cat_words": ["tool", "knife", "hammer", "scissors"],
              "opp_words": ["vehicle", "car", "bus", "train"]},
    "hammer": {"template": "The {obj} is a", "cat_words": ["tool", "knife", "hammer", "scissors"],
               "opp_words": ["vehicle", "car", "bus", "train"]},
    "dog": {"template": "The {obj} is a", "cat_words": ["animal", "dog", "cat", "horse"],
            "opp_words": ["fruit", "apple", "orange", "banana"]},
    "cat": {"template": "The {obj} is a", "cat_words": ["animal", "dog", "cat", "horse"],
            "opp_words": ["fruit", "apple", "orange", "banana"]},
}

ALPHA = 1.5


def load_model_auto(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager",
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


def get_cat_logit_gap(logits, tokenizer, cat_words, opp_words):
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    return float(logits[cat_ids].mean()) - float(logits[opp_ids].mean())


def collect_transport_trajectory(model, tokenizer, input_ids, attention_mask,
                                  last_pos, cat_dir, alpha):
    """收集类别自然运输的每层delta轨迹"""
    input_device = next(model.parameters()).device
    perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
    
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
            pert_out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
        pert_hidden = {li: h[0, last_pos].detach().float().cpu() 
                      for li, h in enumerate(pert_out.hidden_states)}
    except Exception as e:
        print(f"    Perturbed forward failed: {e}")
        pert_hidden = None
    finally:
        if embed_hook is not None:
            embed_hook.remove()
    
    # 基准
    with torch.no_grad():
        base_out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
    base_hidden = {li: h[0, last_pos].detach().float().cpu() 
                  for li, h in enumerate(base_out.hidden_states)}
    
    if pert_hidden is None:
        return None
    
    # 计算delta轨迹
    deltas = {}
    for li in base_hidden:
        deltas[li] = pert_hidden[li] - base_hidden[li]
    
    return deltas, base_out.logits[0, -1].float().cpu().numpy(), pert_out.logits[0, -1].float().cpu().numpy()


def inject_trajectory_and_measure(model, tokenizer, input_ids, attention_mask,
                                   last_pos, deltas, src_cat_gap, obj_info):
    """在target对象的中间层注入source的delta轨迹,测量迁移效果"""
    # 方法: 在指定层(layer output)之后注入delta
    # 使用register_forward_hook在layer的输出上添加delta
    
    n_layers = len(deltas)
    layers = get_layers(model)
    
    # 选择注入层: 尝试多个中间层
    inject_layers = [n_layers // 2, n_layers // 4, n_layers * 3 // 4]
    inject_layers = [l for l in inject_layers if l < n_layers]
    
    all_scores = {}
    
    for inject_li in inject_layers:
        delta_vec = deltas[inject_li].to(torch.bfloat16)
        hook_handle = None
        
        def make_inject_hook(dv, li):
            def hook(module, inp, out):
                # 在residual stream上注入delta
                if isinstance(out, tuple):
                    h = out[0].clone()
                    h[0, last_pos] = h[0, last_pos] + dv.to(h.device).to(h.dtype)
                    return (h,) + out[1:]
                return out
            return hook
        
        hook_handle = layers[inject_li].register_forward_hook(make_inject_hook(delta_vec, inject_li))
        
        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[0, -1].float().cpu().numpy()
        except Exception as e:
            print(f"    Inject at L{inject_li} failed: {e}")
            logits = None
        finally:
            if hook_handle is not None:
                hook_handle.remove()
        
        if logits is not None:
            tgt_cat_gap = get_cat_logit_gap(logits, tokenizer, obj_info["cat_words"], obj_info["opp_words"])
            transfer_score = tgt_cat_gap - src_cat_gap
            all_scores[f"L{inject_li}"] = round(float(transfer_score), 4)
    
    if not all_scores:
        return None
    
    # 返回中间层的结果
    best_layer = max(all_scores, key=lambda k: abs(all_scores[k]))
    return all_scores[best_layer]


def run_experiment(model_name, round_num):
    print(f"\n{'='*60}")
    print(f"Phase 446: Natural Transport Transfer")
    print(f"Model: {model_name}, Round: {round_num}")
    print(f"{'='*60}")
    
    print("\n[1] Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  Loaded: {info.model_class}, {n_layers} layers")
    print(f"  Load time: {time.time()-t0:.1f}s")
    
    results = {"same_class": {}, "cross_class": {}}
    
    # ===== 同类迁移 =====
    print(f"\n[2] Same-class transfer...")
    for src, tgt, cat in PAIRS["same_class"]:
        print(f"\n  {src} → {tgt} ({cat})")
        
        # Source: 收集运输轨迹
        src_info = OBJECTS[src]
        cat_dir = get_cat_direction(model, tokenizer, src_info["cat_words"], src_info["opp_words"])
        if cat_dir is None:
            continue
        
        src_text = src_info["template"].format(obj=src)
        src_inputs = tokenizer(src_text, return_tensors="pt", truncation=True, max_length=64)
        input_device = next(model.parameters()).device
        src_ids = src_inputs["input_ids"].to(input_device)
        src_mask = src_inputs["attention_mask"].to(input_device)
        src_last = src_ids.shape[1] - 1
        
        traj_result = collect_transport_trajectory(model, tokenizer, src_ids, src_mask,
                                                    src_last, cat_dir, ALPHA)
        if traj_result is None:
            continue
        deltas, base_logits, pert_logits = traj_result
        
        src_base_gap = get_cat_logit_gap(base_logits, tokenizer, src_info["cat_words"], src_info["opp_words"])
        src_pert_gap = get_cat_logit_gap(pert_logits, tokenizer, src_info["cat_words"], src_info["opp_words"])
        
        print(f"    src base_gap={src_base_gap:.3f}, pert_gap={src_pert_gap:.3f}")
        print(f"    delta trajectory: L0_norm={deltas[0].norm():.2f}, "
              f"Lmid_norm={deltas[n_layers//2].norm():.2f}, "
              f"Llast_norm={deltas[n_layers-1].norm():.2f}")
        
        # Target: 基准cat_gap
        tgt_info = OBJECTS[tgt]
        tgt_text = tgt_info["template"].format(obj=tgt)
        tgt_inputs = tokenizer(tgt_text, return_tensors="pt", truncation=True, max_length=64)
        tgt_ids = tgt_inputs["input_ids"].to(input_device)
        tgt_mask = tgt_inputs["attention_mask"].to(input_device)
        tgt_last = tgt_ids.shape[1] - 1
        
        with torch.no_grad():
            tgt_base_out = model(input_ids=tgt_ids, attention_mask=tgt_mask)
        tgt_base_logits = tgt_base_out.logits[0, -1].float().cpu().numpy()
        tgt_base_gap = get_cat_logit_gap(tgt_base_logits, tokenizer, tgt_info["cat_words"], tgt_info["opp_words"])
        
        # 注入source的运输轨迹
        transfer_score = inject_trajectory_and_measure(
            model, tokenizer, tgt_ids, tgt_mask, tgt_last, deltas, tgt_base_gap, tgt_info
        )
        
        if transfer_score is not None:
            results["same_class"][f"{src}→{tgt}"] = {
                "category": cat,
                "src_base_gap": round(src_base_gap, 4),
                "src_pert_gap": round(src_pert_gap, 4),
                "tgt_base_gap": round(tgt_base_gap, 4),
                "transfer_score": round(float(transfer_score), 4),
            }
            print(f"    transfer_score={transfer_score:.4f}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ===== 跨类迁移 =====
    print(f"\n[3] Cross-class transfer...")
    for src, tgt, src_cat, tgt_cat in PAIRS["cross_class"]:
        print(f"\n  {src}({src_cat}) → {tgt}({tgt_cat})")
        
        src_info = OBJECTS[src]
        cat_dir = get_cat_direction(model, tokenizer, src_info["cat_words"], src_info["opp_words"])
        if cat_dir is None:
            continue
        
        src_text = src_info["template"].format(obj=src)
        src_inputs = tokenizer(src_text, return_tensors="pt", truncation=True, max_length=64)
        input_device = next(model.parameters()).device
        src_ids = src_inputs["input_ids"].to(input_device)
        src_mask = src_inputs["attention_mask"].to(input_device)
        src_last = src_ids.shape[1] - 1
        
        traj_result = collect_transport_trajectory(model, tokenizer, src_ids, src_mask,
                                                    src_last, cat_dir, ALPHA)
        if traj_result is None:
            continue
        deltas, base_logits, pert_logits = traj_result
        
        tgt_info = OBJECTS[tgt]
        tgt_text = tgt_info["template"].format(obj=tgt)
        tgt_inputs = tokenizer(tgt_text, return_tensors="pt", truncation=True, max_length=64)
        tgt_ids = tgt_inputs["input_ids"].to(input_device)
        tgt_mask = tgt_inputs["attention_mask"].to(input_device)
        tgt_last = tgt_ids.shape[1] - 1
        
        with torch.no_grad():
            tgt_base_out = model(input_ids=tgt_ids, attention_mask=tgt_mask)
        tgt_base_logits = tgt_base_out.logits[0, -1].float().cpu().numpy()
        tgt_base_gap = get_cat_logit_gap(tgt_base_logits, tokenizer, tgt_info["cat_words"], tgt_info["opp_words"])
        
        transfer_score = inject_trajectory_and_measure(
            model, tokenizer, tgt_ids, tgt_mask, tgt_last, deltas, tgt_base_gap, tgt_info
        )
        
        if transfer_score is not None:
            results["cross_class"][f"{src}({src_cat})→{tgt}({tgt_cat})"] = {
                "src_category": src_cat,
                "tgt_category": tgt_cat,
                "tgt_base_gap": round(tgt_base_gap, 4),
                "transfer_score": round(float(transfer_score), 4),
            }
            print(f"    transfer_score={transfer_score:.4f}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ===== 汇总 =====
    print(f"\n{'='*60}")
    print("PHASE 446 SUMMARY")
    print(f"{'='*60}")
    
    same_scores = [v["transfer_score"] for v in results["same_class"].values()]
    cross_scores = [v["transfer_score"] for v in results["cross_class"].values()]
    
    print(f"  Same-class transfer: {same_scores}")
    print(f"  Avg same-class: {np.mean(same_scores):.4f}" if same_scores else "  No same-class data")
    print(f"  Cross-class transfer: {cross_scores}")
    print(f"  Avg cross-class: {np.mean(cross_scores):.4f}" if cross_scores else "  No cross-class data")
    
    if same_scores and cross_scores:
        ratio = np.mean(same_scores) / (np.mean(cross_scores) + 1e-8)
        print(f"  Same/Cross ratio: {ratio:.2f}")
        if ratio > 1.5:
            print(f"  Verdict: CATEGORY-SPECIFIC transfer exists")
        else:
            print(f"  Verdict: Transfer is NOT category-specific")
    
    # 保存
    output = {
        "model": model_name,
        "round": round_num,
        "n_layers": n_layers,
        "alpha": ALPHA,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
        "summary": {
            "avg_same_class": round(float(np.mean(same_scores)), 4) if same_scores else 0,
            "avg_cross_class": round(float(np.mean(cross_scores)), 4) if cross_scores else 0,
        }
    }
    
    os.makedirs("results/phase446_natural_transport_transfer", exist_ok=True)
    out_path = f"results/phase446_natural_transport_transfer/{model_name}_phase446_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n  Saved: {out_path}")
    
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
