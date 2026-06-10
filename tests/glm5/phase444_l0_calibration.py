"""
Phase 444: L0 Attention Calibration Mechanism
==============================================
目标: 解释为什么消融L0 attention后信号反而增强

核心问题:
Phase 439中发现消融L0 attention后norm_score=-28~-30,
说明L0 attention不是简单的"搬运器",而可能是"校准器/过滤器"。

方法:
1. 在输入层注入类别方向扰动(alpha=1.0)
2. 消融L0 attention
3. 比较:
   a. delta在类别方向上的投影 vs 在正交方向上的投影
   b. 非类别候选词的logit变化
   c. entropy变化
   d. 不同方向的投影分解

预期: 如果L0 attention是校准器,消融后:
   - 类别方向投影增大(信号放大)
   - 正交方向投影也增大(噪声放大)
   - 非类别候选logit也变化
   - entropy增大(更不确定)

用法:
  python tests/glm5/phase444_l0_calibration.py qwen3 1
  python tests/glm5/phase444_l0_calibration.py glm4 1
  python tests/glm5/phase444_l0_calibration.py deepseek7b 1
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
OBJECTS = {
    "apple": {
        "template": "The {obj} is a",
        "cat_words": ["fruit", "apple", "orange", "banana"],
        "opp_words": ["animal", "dog", "cat", "horse"],
        "other_cats": ["tool", "vehicle"],  # 非目标类别
    },
    "knife": {
        "template": "The {obj} is a",
        "cat_words": ["tool", "knife", "hammer", "scissors"],
        "opp_words": ["vehicle", "car", "bus", "train"],
        "other_cats": ["fruit", "animal"],
    },
    "dog": {
        "template": "The {obj} is a",
        "cat_words": ["animal", "dog", "cat", "horse"],
        "opp_words": ["fruit", "apple", "orange", "banana"],
        "other_cats": ["tool", "vehicle"],
    },
}

ALPHA = 1.0


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
        attn_implementation="eager",
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


def get_orthogonal_directions(cat_dir, n_directions=5):
    """生成与类别方向正交的随机方向"""
    d = cat_dir.shape[0]
    dirs = []
    for _ in range(n_directions):
        rand_d = torch.randn(d)
        # Gram-Schmidt正交化
        proj = (rand_d @ cat_dir) / (cat_dir @ cat_dir + 1e-8) * cat_dir
        rand_d = rand_d - proj
        if rand_d.norm() > 1e-6:
            rand_d = rand_d / rand_d.norm()
            dirs.append(rand_d)
    return dirs


def compute_entropy(logits):
    """计算softmax entropy"""
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    return -np.sum(probs * np.log(probs + 1e-10))


def run_with_perturbation(model, tokenizer, input_ids, attention_mask, last_pos,
                          cat_dir, alpha, ablate_l0_attn=False):
    """运行模型: 注入扰动 + 可选L0 attention消融"""
    input_device = next(model.parameters()).device
    perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
    
    embed_hook = None
    abl_hook = None
    
    def on_embed(module, inp, out):
        if isinstance(out, torch.Tensor):
            out = out.clone()
            out[0, last_pos] = out[0, last_pos] + perturb_vec
        return out
    
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_hook = model.model.embed_tokens.register_forward_hook(on_embed)
    
    if ablate_l0_attn:
        layers = get_layers(model)
        def zero_attn_hook(module, inp, out):
            if isinstance(out, tuple):
                return (torch.zeros_like(out[0]),) + out[1:]
            return torch.zeros_like(out)
        abl_hook = layers[0].self_attn.register_forward_hook(zero_attn_hook)
    
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        last_hidden = out.hidden_states[-1][0, last_pos].detach().float().cpu()
        logits = out.logits[0, -1].float().cpu().numpy()
        # 也获取中间层hidden states
        mid_hidden = {}
        for li in [0, 1, 2, 4, 8]:
            if li < len(out.hidden_states):
                mid_hidden[li] = out.hidden_states[li][0, last_pos].detach().float().cpu()
    except Exception as e:
        print(f"    Forward failed: {e}")
        last_hidden = logits = mid_hidden = None
    finally:
        if embed_hook is not None:
            embed_hook.remove()
        if abl_hook is not None:
            abl_hook.remove()
    
    return last_hidden, logits, mid_hidden


def run_experiment(model_name, round_num):
    print(f"\n{'='*60}")
    print(f"Phase 444: L0 Attention Calibration Mechanism")
    print(f"Model: {model_name}, Round: {round_num}")
    print(f"{'='*60}")
    
    # 加载模型
    print("\n[1] Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  Loaded: {info.model_class}, {n_layers} layers")
    print(f"  Load time: {time.time()-t0:.1f}s")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # 对每个对象做分析
    results = {}
    
    for obj_name, obj_info in OBJECTS.items():
        print(f"\n{'='*50}")
        print(f"  Processing {obj_name}")
        print(f"{'='*50}")
        
        cat_dir = get_cat_direction(model, tokenizer, obj_info["cat_words"], obj_info["opp_words"])
        if cat_dir is None:
            continue
        
        # 生成正交方向
        ortho_dirs = get_orthogonal_directions(cat_dir, n_directions=5)
        
        # 构建输入
        text = obj_info["template"].format(obj=obj_name)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        last_pos = input_ids.shape[1] - 1
        
        # ===== 基准运行 =====
        print(f"  Running baseline...")
        with torch.no_grad():
            base_out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
        base_last = base_out.hidden_states[-1][0, last_pos].detach().float().cpu()
        base_logits = base_out.logits[0, -1].float().cpu().numpy()
        base_entropy = compute_entropy(base_logits)
        
        # 类别和非类别logit
        cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in obj_info["cat_words"]]
        opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in obj_info["opp_words"]]
        other_ids = []
        for cat in obj_info["other_cats"]:
            other_ids.extend([tokenizer.encode(w, add_special_tokens=False)[0] for w in [cat]])
        
        base_cat_logit = float(base_logits[cat_ids].mean())
        base_opp_logit = float(base_logits[opp_ids].mean())
        base_other_logit = float(base_logits[other_ids].mean()) if other_ids else 0
        
        # ===== 扰动运行(无消融) =====
        print(f"  Running perturbed (no ablation)...")
        pert_last, pert_logits, pert_mid = run_with_perturbation(
            model, tokenizer, input_ids, attention_mask, last_pos, cat_dir, ALPHA
        )
        if pert_last is None:
            continue
        
        pert_entropy = compute_entropy(pert_logits)
        pert_cat_logit = float(pert_logits[cat_ids].mean())
        pert_opp_logit = float(pert_logits[opp_ids].mean())
        pert_other_logit = float(pert_logits[other_ids].mean()) if other_ids else 0
        
        # 基准delta
        orig_delta = pert_last - base_last
        orig_delta_norm = float(orig_delta.norm())
        
        # 类别方向投影
        cat_proj = float(orig_delta @ cat_dir)
        cat_proj_normalized = cat_proj / orig_delta_norm if orig_delta_norm > 1e-6 else 0
        
        # 正交方向投影
        ortho_projs = [float(orig_delta @ d) for d in ortho_dirs]
        ortho_proj_norm = np.sqrt(sum(p**2 for p in ortho_projs))
        
        print(f"  orig_delta_norm={orig_delta_norm:.4f}")
        print(f"  cat_proj={cat_proj:.4f} (norm={cat_proj_normalized:.4f})")
        print(f"  ortho_proj_norm={ortho_proj_norm:.4f}")
        print(f"  cat/total ratio={abs(cat_proj)/(orig_delta_norm+1e-8):.4f}")
        
        # ===== 扰动运行(消融L0 attention) =====
        print(f"  Running perturbed (L0 attn ablated)...")
        abl_last, abl_logits, abl_mid = run_with_perturbation(
            model, tokenizer, input_ids, attention_mask, last_pos, cat_dir, ALPHA,
            ablate_l0_attn=True
        )
        if abl_last is None:
            continue
        
        abl_entropy = compute_entropy(abl_logits)
        abl_cat_logit = float(abl_logits[cat_ids].mean())
        abl_opp_logit = float(abl_logits[opp_ids].mean())
        abl_other_logit = float(abl_logits[other_ids].mean()) if other_ids else 0
        
        # 消融后的delta
        abl_delta = abl_last - base_last
        abl_delta_norm = float(abl_delta.norm())
        
        # 类别方向投影(消融后)
        abl_cat_proj = float(abl_delta @ cat_dir)
        abl_cat_proj_normalized = abl_cat_proj / abl_delta_norm if abl_delta_norm > 1e-6 else 0
        
        # 正交方向投影(消融后)
        abl_ortho_projs = [float(abl_delta @ d) for d in ortho_dirs]
        abl_ortho_proj_norm = np.sqrt(sum(p**2 for p in abl_ortho_projs))
        
        # 关键比较
        print(f"\n  === L0 Attention Calibration Analysis ===")
        print(f"  delta_norm: orig={orig_delta_norm:.4f} -> ablated={abl_delta_norm:.4f} "
              f"(change={abl_delta_norm/orig_delta_norm - 1:.2%})")
        print(f"  cat_proj: orig={cat_proj:.4f} -> ablated={abl_cat_proj:.4f} "
              f"(change={abl_cat_proj/cat_proj - 1:.2%})" if abs(cat_proj) > 1e-6 else "")
        print(f"  ortho_proj_norm: orig={ortho_proj_norm:.4f} -> ablated={abl_ortho_proj_norm:.4f} "
              f"(change={abl_ortho_proj_norm/(ortho_proj_norm+1e-8) - 1:.2%})")
        print(f"  cat/total: orig={abs(cat_proj)/(orig_delta_norm+1e-8):.4f} "
              f"-> ablated={abs(abl_cat_proj)/(abl_delta_norm+1e-8):.4f}")
        
        # 类别/非类别logit变化
        print(f"\n  Logit changes:")
        print(f"  cat_logit: base={base_cat_logit:.3f} -> pert={pert_cat_logit:.3f} -> abl={abl_cat_logit:.3f}")
        print(f"  opp_logit: base={base_opp_logit:.3f} -> pert={pert_opp_logit:.3f} -> abl={abl_opp_logit:.3f}")
        print(f"  other_logit: base={base_other_logit:.3f} -> pert={pert_other_logit:.3f} -> abl={abl_other_logit:.3f}")
        print(f"  entropy: base={base_entropy:.4f} -> pert={pert_entropy:.4f} -> abl={abl_entropy:.4f}")
        
        # 保存结果
        results[obj_name] = {
            "category": obj_info["cat_words"][0],
            "orig_delta_norm": round(orig_delta_norm, 4),
            "abl_delta_norm": round(abl_delta_norm, 4),
            "delta_norm_ratio": round(abl_delta_norm / orig_delta_norm, 4) if orig_delta_norm > 1e-6 else 0,
            "cat_proj_orig": round(cat_proj, 4),
            "cat_proj_abl": round(abl_cat_proj, 4),
            "cat_proj_ratio": round(abl_cat_proj / cat_proj, 4) if abs(cat_proj) > 1e-6 else 0,
            "ortho_proj_norm_orig": round(ortho_proj_norm, 4),
            "ortho_proj_norm_abl": round(abl_ortho_proj_norm, 4),
            "ortho_proj_ratio": round(abl_ortho_proj_norm / (ortho_proj_norm + 1e-8), 4),
            "cat_frac_orig": round(abs(cat_proj) / (orig_delta_norm + 1e-8), 4),
            "cat_frac_abl": round(abs(abl_cat_proj) / (abl_delta_norm + 1e-8), 4),
            "entropy": {
                "base": round(base_entropy, 4),
                "perturbed": round(pert_entropy, 4),
                "ablated": round(abl_entropy, 4),
                "pert_delta": round(pert_entropy - base_entropy, 4),
                "abl_delta": round(abl_entropy - base_entropy, 4),
            },
            "logits": {
                "cat": {"base": round(base_cat_logit, 3), "pert": round(pert_cat_logit, 3), "abl": round(abl_cat_logit, 3)},
                "opp": {"base": round(base_opp_logit, 3), "pert": round(pert_opp_logit, 3), "abl": round(abl_opp_logit, 3)},
                "other": {"base": round(base_other_logit, 3), "pert": round(pert_other_logit, 3), "abl": round(abl_other_logit, 3)},
            },
        }
        
        # 中间层分析
        if pert_mid and abl_mid:
            mid_results = {}
            for li in sorted(pert_mid.keys()):
                if li in abl_mid:
                    pert_h = pert_mid[li]
                    abl_h = abl_mid[li]
                    base_h = base_out.hidden_states[li][0, last_pos].detach().float().cpu()
                    
                    mid_delta = pert_h - base_h
                    mid_abl_delta = abl_h - base_h
                    
                    mid_cat_proj = float(mid_delta @ cat_dir)
                    mid_abl_cat_proj = float(mid_abl_delta @ cat_dir)
                    
                    mid_results[f"L{li}"] = {
                        "delta_norm": round(float(mid_delta.norm()), 4),
                        "abl_delta_norm": round(float(mid_abl_delta.norm()), 4),
                        "cat_proj": round(mid_cat_proj, 4),
                        "abl_cat_proj": round(mid_abl_cat_proj, 4),
                    }
            
            results[obj_name]["mid_layer_analysis"] = mid_results
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ===== 汇总 =====
    print(f"\n{'='*60}")
    print("PHASE 444 SUMMARY")
    print(f"{'='*60}")
    
    for obj_name, r in results.items():
        print(f"\n  {obj_name}:")
        print(f"    delta_norm ratio (ablated/orig): {r['delta_norm_ratio']}")
        print(f"    cat_proj ratio (ablated/orig): {r['cat_proj_ratio']}")
        print(f"    ortho_proj ratio (ablated/orig): {r['ortho_proj_ratio']}")
        print(f"    cat_frac: orig={r['cat_frac_orig']} -> ablated={r['cat_frac_abl']}")
        print(f"    entropy change: pert={r['entropy']['pert_delta']}, abl={r['entropy']['abl_delta']}")
        
        if "mid_layer_analysis" in r:
            print(f"    Mid-layer cat_proj:")
            for li, mr in r["mid_layer_analysis"].items():
                print(f"      {li}: orig={mr['cat_proj']}, abl={mr['abl_cat_proj']}")
    
    # 判断校准 vs 放大
    print(f"\n  === Calibration vs Amplification ===")
    for obj_name, r in results.items():
        cat_ratio = r['cat_proj_ratio']
        ortho_ratio = r['ortho_proj_ratio']
        
        if cat_ratio > 1.0 and ortho_ratio > 1.0:
            if ortho_ratio > cat_ratio:
                verdict = "AMPLIFY+NOISE (L0 attn filters noise)"
            else:
                verdict = "AMPLIFY+BIAS (L0 attn biases toward category)"
        elif cat_ratio > 1.0 and ortho_ratio <= 1.0:
            verdict = "BIAS (L0 attn suppresses category signal)"
        else:
            verdict = "COMPLEX"
        
        print(f"  {obj_name}: cat_ratio={cat_ratio:.2f}, ortho_ratio={ortho_ratio:.2f} -> {verdict}")
    
    # 保存
    output = {
        "model": model_name,
        "round": round_num,
        "n_layers": n_layers,
        "alpha": ALPHA,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_object": results,
    }
    
    os.makedirs("results/phase444_l0_calibration", exist_ok=True)
    out_path = f"results/phase444_l0_calibration/{model_name}_phase444_r{round_num}.json"
    
    # Convert numpy types to Python native
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
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(convert(output), f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")
    
    # 释放
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
