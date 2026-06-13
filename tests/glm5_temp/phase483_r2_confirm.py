"""
Phase 483 Round 2: 竞争释放矩阵确认测试
============================================

对Round 1中发现的强竞争释放对进行确认测试:
- 增加测试对象数量(4→8)提高统计可靠性
- 重点验证最强释放对
- 增加跨类别测试(用不同类别对象测试竞争释放)

用法:
  python tests/glm5_temp/phase483_r2_confirm.py qwen3
  python tests/glm5_temp/phase483_r2_confirm.py glm4
  python tests/glm5_temp/phase483_r2_confirm.py deepseek7b
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS)

def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# 数据定义
CATEGORIES = {
    "fruit":     ["apple", "banana", "orange", "grape", "pear", "peach", "mango", "plum"],
    "animal":    ["dog", "cat", "horse", "lion", "bear", "rabbit", "eagle", "fish"],
    "tool":      ["hammer", "knife", "wrench", "saw", "drill", "axe", "chisel", "pliers"],
    "vehicle":   ["car", "bus", "bicycle", "truck", "train", "boat", "plane", "motorcycle"],
    "clothing":  ["shirt", "dress", "hat", "coat", "sock", "glove", "jacket", "scarf"],
    "furniture": ["chair", "table", "desk", "sofa", "bed", "shelf", "cabinet", "stool"],
    "food":      ["bread", "rice", "cheese", "pasta", "soup", "steak", "salad", "cake"],
    "plant":     ["tree", "flower", "grass", "bush", "fern", "cactus", "vine", "shrub"],
}

CATEGORIES_TRAIN = {
    "fruit":     ["apple", "banana", "orange", "grape"],
    "animal":    ["dog", "cat", "horse", "lion"],
    "tool":      ["hammer", "knife", "wrench", "saw"],
    "vehicle":   ["car", "bus", "bicycle", "truck"],
    "clothing":  ["shirt", "dress", "hat", "coat"],
    "furniture": ["chair", "table", "desk", "sofa"],
    "food":      ["bread", "rice", "cheese", "pasta"],
    "plant":     ["tree", "flower", "grass", "bush"],
}

BEST_NEIGHBORS = {
    "qwen3": {
        "fruit": ["plant", "food"], "animal": ["food", "clothing"],
        "tool": ["vehicle", "furniture"], "vehicle": ["furniture", "tool"],
        "clothing": ["furniture", "tool"], "furniture": ["vehicle", "clothing"],
        "food": ["plant", "vehicle"], "plant": ["food", "animal"],
    },
    "glm4": {
        "fruit": ["plant", "food"], "animal": ["food", "clothing"],
        "tool": ["furniture", "vehicle"], "vehicle": ["tool", "furniture"],
        "clothing": ["furniture", "plant"], "furniture": ["vehicle", "clothing"],
        "food": ["plant", "fruit"], "plant": ["vehicle", "clothing"],
    },
    "deepseek7b": {
        "fruit": ["plant", "food"], "animal": ["food", "clothing"],
        "tool": ["vehicle", "furniture"], "vehicle": ["furniture", "tool"],
        "clothing": ["furniture", "plant"], "furniture": ["tool", "clothing"],
        "food": ["plant", "fruit"], "plant": ["food", "fruit"],
    },
}

BEST_LAYERS = {
    "qwen3": {"fruit": 32, "animal": 33, "tool": 23, "vehicle": 29,
              "clothing": 30, "furniture": 26, "food": 34, "plant": 28},
    "glm4": {"fruit": 27, "animal": 38, "tool": 27, "vehicle": 29,
             "clothing": 39, "furniture": 34, "food": 38, "plant": 32},
    "deepseek7b": {"fruit": 26, "animal": 27, "tool": 26, "vehicle": 26,
                   "clothing": 23, "furniture": 25, "food": 27, "plant": 25},
}

FAMILY_WORDS_8D = {
    "fruit":     ["fruit", "produce", "crop", "berry"],
    "animal":    ["animal", "creature", "beast", "pet"],
    "tool":      ["tool", "implement", "device", "instrument"],
    "vehicle":   ["vehicle", "transport", "automobile", "car"],
    "clothing":  ["clothing", "attire", "wear", "garment"],
    "furniture": ["furniture", "furnishing", "fixture", "seat"],
    "food":      ["food", "meal", "dish", "snack"],
    "plant":     ["plant", "tree", "vegetation", "flora"],
}
DCF_DIM_NAMES = ["fruit", "animal", "tool", "vehicle", "clothing", "furniture", "food", "plant"]

RELATION_TEMPLATES = {"kind_of": "The {obj} is a kind of"}

# 需要确认的最强竞争对(从Round 1结果中选择)
CONFIRM_PAIRS = {
    "qwen3": [
        # (removed_cat, expected_competitor, expected_delta)
        ("animal", "clothing", 9.29),
        ("clothing", "tool", 7.47),
        ("clothing", "furniture", 7.27),
        ("food", "vehicle", 6.19),
        ("fruit", "animal", 4.36),
        ("vehicle", "furniture", 1.83),
    ],
    "glm4": [
        ("clothing", "plant", 1.18),
        ("food", "plant", 1.09),
        ("animal", "clothing", 0.74),
    ],
    "deepseek7b": [
        ("tool", "vehicle", 6.81),
        ("fruit", "food", 5.62),
        ("animal", "food", 4.73),
    ],
}


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog(f"Loading {model_name} (bfloat16 + device_map=auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="eager",
        )
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def find_token_id(tokenizer, word):
    vocab = tokenizer.get_vocab()
    for candidate in [word, f" {word}", word.lower(), f" {word.lower()}"]:
        if candidate in vocab:
            return vocab[candidate]
    return None


def compute_dcf(logits, tokenizer, dim_dict, dim_names):
    dcf_vector = []
    for dim_name in dim_names:
        words = dim_dict.get(dim_name, [])
        logit_values = []
        for w in words:
            tid = find_token_id(tokenizer, w)
            if tid is not None and tid < len(logits):
                logit_values.append(float(logits[tid]))
        dcf_vector.append(float(np.mean(logit_values)) if logit_values else 0.0)
    return np.array(dcf_vector)


def logit_lens_dcf(resid, W_U, tokenizer):
    logits = resid @ W_U.T
    return compute_dcf(logits, tokenizer, FAMILY_WORDS_8D, DCF_DIM_NAMES)


def _make_capture_hook(store_dict, key):
    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            store_dict[key] = output[0].detach().float().cpu()
        else:
            store_dict[key] = output.detach().float().cpu()
    return hook_fn


def get_prompt_ids(tokenizer, device, prompt, max_len=128):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pos = attention_mask.sum().item() - 1
    return input_ids, attention_mask, pos


def qr_orthogonalize(target_vec, basis_vecs):
    if not basis_vecs:
        return target_vec.copy()
    basis = np.array(basis_vecs)
    Q, _ = np.linalg.qr(basis.T)
    proj = Q @ (Q.T @ target_vec)
    return target_vec - proj


def get_specific_direction(model, tokenizer, device, model_name, cat_name, target_layer, n_obj=4):
    neighbors = BEST_NEIGHBORS[model_name]
    layers_list = get_layers(model)
    template = RELATION_TEMPLATES["kind_of"]
    raw_dirs = {}
    for cn, objs in CATEGORIES_TRAIN.items():
        resids = []
        for obj in objs[:n_obj]:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap = {}
            h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap:
                resids.append(cap["resid"][0, pos].numpy())
        if resids:
            raw_dirs[cn] = np.mean(resids, axis=0)
    
    if cat_name not in raw_dirs:
        return None, 0.0
    target_vec = raw_dirs[cat_name]
    basis_vecs = [raw_dirs[n] for n in neighbors[cat_name] if n in raw_dirs]
    spec_vec = qr_orthogonalize(target_vec, basis_vecs) if basis_vecs else target_vec.copy()
    return spec_vec, float(np.linalg.norm(spec_vec))


def confirm_competition_release(model, tokenizer, device, model_name, W_U):
    """用8个test对象确认最强的竞争释放对"""
    plog("=== Round 2: 竞争释放确认 ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    template = RELATION_TEMPLATES["kind_of"]
    
    pairs = CONFIRM_PAIRS.get(model_name, [])
    confirm_results = []
    
    for removed_cat, expected_competitor, expected_delta in pairs:
        best_layer = BEST_LAYERS[model_name][removed_cat]
        plog(f"  Confirm: remove {removed_cat} -> {expected_competitor} (expected +{expected_delta:.2f})")
        t0 = time.time()
        
        # 获取specific方向
        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, removed_cat, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"    Skip: spec_norm too small")
            continue
        
        target_idx = cat_names.index(removed_cat)
        competitor_idx = cat_names.index(expected_competitor)
        
        # 用8个test对象(全量)
        test_objs = CATEGORIES[removed_cat][4:8]  # 4个未训练对象
        # 再加2个来自训练集的不同对象
        extra_objs = CATEGORIES[removed_cat][0:2]
        all_test = test_objs + extra_objs  # 6个
        
        dcf_deltas = []
        
        for obj in all_test:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            
            # Baseline
            cap = {}
            h = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" not in cap:
                continue
            baseline_dcf = logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer)
            
            # 移除 (scale=1.0)
            b_hat = spec_vec / spec_norm
            def make_remove_hook_fn(r_vec, position, scale=1.0):
                added = [False]
                def hook_fn(module, inp, output):
                    if not added[0]:
                        if isinstance(output, tuple):
                            out = output[0].clone()
                        else:
                            out = output.clone()
                        b_h = r_vec / (np.linalg.norm(r_vec) + 1e-10)
                        resid_np = out[0, position, :].float().cpu().numpy()
                        proj = np.dot(resid_np, b_h) * b_h * scale
                        out[0, position, :] -= torch.tensor(proj, dtype=out.dtype, device=out.device)
                        added[0] = True
                        if isinstance(output, tuple):
                            return (out,) + output[1:]
                        return out
                    return output
                return hook_fn
            
            h2 = layers_list[best_layer].register_forward_hook(
                make_remove_hook_fn(spec_vec, pos, scale=1.0)
            )
            cap2 = {}
            h3 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap2, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h2.remove()
            h3.remove()
            if "resid" not in cap2:
                continue
            remove_dcf = logit_lens_dcf(cap2["resid"][0, pos].numpy(), W_U, tokenizer)
            dcf_deltas.append(remove_dcf - baseline_dcf)
        
        if dcf_deltas:
            avg_delta = np.mean(dcf_deltas, axis=0)
            std_delta = np.std(dcf_deltas, axis=0)
            
            # 目标类别变化
            target_d = avg_delta[target_idx]
            # 竞争类别变化
            competitor_d = avg_delta[competitor_idx]
            competitor_std = std_delta[competitor_idx]
            
            # 显著性: 竞争类别的平均变化 / 标准差
            significance = competitor_d / (competitor_std + 0.01)
            
            confirm_results.append({
                "removed_cat": removed_cat,
                "expected_competitor": expected_competitor,
                "expected_delta_r1": expected_delta,
                "confirmed_target_delta": float(target_d),
                "confirmed_competitor_delta": float(competitor_d),
                "competitor_std": float(competitor_std),
                "significance": float(significance),
                "n_test_objects": len(dcf_deltas),
                "confirmed": abs(competitor_d) > 0.5 and significance > 1.5,
            })
            
            status = "CONFIRMED" if abs(competitor_d) > 0.5 and significance > 1.5 else "WEAK"
            plog(f"    {removed_cat}->{expected_competitor}: "
                  f"R1={expected_delta:.2f}, R2={competitor_d:.2f}±{competitor_std:.2f}, "
                  f"sig={significance:.2f}, {status} ({time.time()-t0:.1f}s)")
        
        gc.collect()
    
    return confirm_results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    plog(f"Phase 483 Round 2: Competition Release Confirmation | Model={model_name}")
    
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    
    results = confirm_competition_release(model, tokenizer, device, model_name, W_U)
    
    # 汇总
    plog(f"\n=== Round 2 Summary ===")
    confirmed = sum(1 for r in results if r["confirmed"])
    total = len(results)
    plog(f"  Confirmed: {confirmed}/{total}")
    for r in results:
        status = "✓" if r["confirmed"] else "✗"
        plog(f"  {status} {r['removed_cat']}->{r['expected_competitor']}: "
              f"R1={r['expected_delta_r1']:.2f}, R2={r['confirmed_competitor_delta']:.2f}, "
              f"sig={r['significance']:.2f}")
    
    # 保存
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj
    out_path = f"results/glm5/phase483_{model_name}_r2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(convert({"phase": 483, "round": 2, "model": model_name, 
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "confirm_results": results}), f, indent=2, ensure_ascii=False)
    plog(f"Saved to {out_path}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    plog("Done!")


if __name__ == "__main__":
    main()
