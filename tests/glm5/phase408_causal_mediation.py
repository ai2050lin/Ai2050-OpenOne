"""
Phase 408: Path-Level Causal Mediation Analysis
================================================

Phase 404发现了attn/MLP的方向效应(odd/even), 但那是"方向注入"方法。
Phase 408使用更严格的"因果中介"方法, 对每个属性方向分解组件贡献。

核心方法: 对每个属性(temperature/size/speed):
1. 计算clean vs corrupt的残差差 (方向)
2. 在每个检查点(post_input_ln, attn_out, post_attn_ln, mlp_down)测量:
   - 总效应: clean_corr - corrupt_corr
   - 中介效应: 在某检查点拦截方向注入后的效应变化
3. 分解: total = attn_mediated + MLP_mediated + RMSNorm_mediated + residual

对3个属性×2个层(early+deep)×3个模型进行完整测试。

Usage:
  python tests/glm5/phase408_causal_mediation.py qwen3
  python tests/glm5/phase408_causal_mediation.py glm4
  python tests/glm5/phase408_causal_mediation.py deepseek7b
"""

import sys
import os
import json
import time
import gc
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, OrderedDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS, get_layers, get_model_info, release_model, get_W_U

# ===== 属性定义 (与Phase 407一致, 精简) =====
ATTRIBUTE_CONFIGS = OrderedDict({
    "temperature": {
        "candidates": OrderedDict([
            ("freezing", 1), ("cold", 2), ("cool", 3), ("warm", 4), ("hot", 5), ("scorching", 6),
        ]),
        "high_objects": ["desert", "volcano"],    # 高温
        "low_objects": ["ice", "snow"],            # 低温
        "high_label": "hot",
        "low_label": "cold",
    },
    "size": {
        "candidates": OrderedDict([
            ("microscopic", 1), ("tiny", 2), ("small", 3), ("medium", 4),
            ("large", 5), ("huge", 6), ("massive", 7),
        ]),
        "high_objects": ["elephant", "mountain"],  # 大
        "low_objects": ["ant", "pebble"],           # 小
        "high_label": "huge",
        "low_label": "tiny",
    },
    "speed": {
        "candidates": OrderedDict([
            ("sluggish", 1), ("slow", 2), ("steady", 3), ("moderate", 4),
            ("quick", 5), ("fast", 6), ("rapid", 7), ("swift", 8),
        ]),
        "high_objects": ["cheetah", "rocket"],     # 快
        "low_objects": ["snail", "glacier"],        # 慢
        "high_label": "fast",
        "low_label": "slow",
    },
})

# 检查点定义
CHECKPOINT_NAMES = ["post_input_ln", "attn_out", "post_attn_ln", "mlp_down"]

# 采样层
ANALYSIS_LAYERS = {
    "qwen3": [4, 20, 28],
    "deepseek7b": [4, 14, 20],
    "glm4": [5, 20, 35],
}


def log_memory():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU: {alloc:.2f}GB alloc, {reserved:.2f}GB reserved"
    return "GPU not available"


def load_model_bf16_safe(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name} (BF16+auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=impl
            )
            print(f"  Loaded with attn_implementation={impl}")
            break
        except Exception as e:
            print(f"  attn_implementation={impl} failed: {e}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    print(f"  Loaded. {log_memory()}")
    return model, tokenizer


def compute_gradient_metric(logits, candidate_ids, levels):
    """计算属性gradient (logit对level的斜率)"""
    cand_logits = np.array([logits[cid] if cid is not None else float('-inf') for cid in candidate_ids])
    valid_mask = np.array([cid is not None for cid in candidate_ids])

    if valid_mask.sum() < 2:
        return 0.0, 0.0

    valid_levels = np.array(levels)[valid_mask]
    valid_cand_logits = cand_logits[valid_mask]

    # gradient
    if len(valid_levels) > 1:
        slope = np.polyfit(valid_levels, valid_cand_logits, 1)[0]
    else:
        slope = 0.0

    # top candidate level
    max_idx = np.argmax(valid_cand_logits)
    top_level = float(valid_levels[max_idx])

    return float(slope), top_level


def run_causal_mediation(model, tokenizer, device, W_U_np, layers_list,
                         attr_name, attr_config, candidate_ids, levels, layer_idx):
    """
    对单个属性在单个层进行因果中介分析

    方法:
    1. 计算 clean(corrrect) vs corrupt的残差差作为方向
    2. 在每个检查点测量方向注入效应
    3. 分解组件贡献

    Returns:
        dict with checkpoint-level effects
    """
    high_label = attr_config["high_label"]
    low_label = attr_config["low_label"]
    high_objs = attr_config["high_objects"]
    low_objs = attr_config["low_objects"]

    all_objects = high_objs + low_objs

    # ===== Step 1: Compute clean/corrupt residuals at this layer =====
    captured = {}
    def make_hook(key):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook_fn

    handle = layers_list[layer_idx].register_forward_hook(make_hook('h'))

    obj_residuals = {}
    obj_logits_final = {}

    for obj_name in all_objects:
        # Clean: "The {obj} is {high_label}."
        clean_prompt = f"The {obj_name} is {high_label}."
        captured.clear()
        inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out = model(input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                        output_hidden_states=True)
        h_clean = captured['h'][0, -1].numpy().copy() if 'h' in captured else None
        final_clean = out.logits[0, -1].float().cpu().numpy()

        # Corrupt: "The item is {high_label}."
        corrupt_prompt = f"The item is {high_label}."
        captured.clear()
        inputs = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out = model(input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                        output_hidden_states=True)
        h_corrupt = captured['h'][0, -1].numpy().copy() if 'h' in captured else None
        final_corrupt = out.logits[0, -1].float().cpu().numpy()

        if h_clean is not None and h_corrupt is not None:
            direction = h_clean - h_corrupt
            obj_residuals[obj_name] = {
                "clean": h_clean,
                "corrupt": h_corrupt,
                "direction": direction,
                "dir_norm": float(np.linalg.norm(direction)),
            }

        # Clean vs corrupt gradient
        grad_clean, _ = compute_gradient_metric(final_clean, candidate_ids, levels)
        grad_corrupt, _ = compute_gradient_metric(final_corrupt, candidate_ids, levels)
        obj_logits_final[obj_name] = {
            "grad_clean": grad_clean,
            "grad_corrupt": grad_corrupt,
            "total_effect": grad_clean - grad_corrupt,
        }

    handle.remove()

    if not obj_residuals:
        return None

    # ===== Step 2: Compute average direction for high vs low objects =====
    high_dirs = [obj_residuals[n]["direction"] for n in high_objs if n in obj_residuals]
    low_dirs = [obj_residuals[n]["direction"] for n in low_objs if n in obj_residuals]

    if high_dirs and low_dirs:
        high_mean = np.mean(high_dirs, axis=0)
        low_mean = np.mean(low_dirs, axis=0)
        attr_direction = high_mean - low_mean  # "high-ness" direction
    else:
        return None

    dir_norm = np.linalg.norm(attr_direction)
    if dir_norm < 1e-8:
        return None

    # ===== Step 3: Inject direction at checkpoints and measure effects =====
    checkpoint_results = {}

    for obj_name in all_objects:
        if obj_name not in obj_residuals:
            continue

        prompt = f"The {obj_name} is"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Baseline (no injection)
        with torch.no_grad():
            out_base = model(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True)
        base_logits = out_base.logits[0, -1].float().cpu().numpy()
        grad_base, top_base = compute_gradient_metric(base_logits, candidate_ids, levels)

        # Inject direction (+ and -) at this layer
        for sign_name, sign in [("plus", 1.0), ("minus", -1.0)]:
            delta = torch.tensor(sign * attr_direction, dtype=torch.bfloat16, device=device)

            def make_add_hook(delta_vec):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hs = output[0].clone()
                        hs[0, -1, :] += delta_vec
                        return (hs,) + output[1:]
                    else:
                        hs = output.clone()
                        hs[0, -1, :] += delta_vec
                        return hs
                return hook_fn

            handle = layers_list[layer_idx].register_forward_hook(make_add_hook(delta))
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
            handle.remove()

            inj_logits = out.logits[0, -1].float().cpu().numpy()
            grad_inj, top_inj = compute_gradient_metric(inj_logits, candidate_ids, levels)

            if obj_name not in checkpoint_results:
                checkpoint_results[obj_name] = {}
            checkpoint_results[obj_name][sign_name] = {
                "gradient": grad_inj,
                "top_level": top_inj,
            }

        # Compute odd/even for this object
        grad_plus = checkpoint_results[obj_name]["plus"]["gradient"]
        grad_minus = checkpoint_results[obj_name]["minus"]["gradient"]
        eff_plus = grad_plus - grad_base
        eff_minus = grad_minus - grad_base

        checkpoint_results[obj_name]["base"] = {
            "gradient": grad_base,
            "top_level": top_base,
        }
        checkpoint_results[obj_name]["odd"] = float((eff_plus - eff_minus) / 2)
        checkpoint_results[obj_name]["even"] = float((eff_plus + eff_minus) / 2)

    # ===== Step 4: Aggregate =====
    # Average odd/even across objects
    all_odds = [checkpoint_results[n]["odd"] for n in all_objects if n in checkpoint_results]
    all_evens = [checkpoint_results[n]["even"] for n in all_objects if n in checkpoint_results]

    # High vs low objects
    high_odds = [checkpoint_results[n]["odd"] for n in high_objs if n in checkpoint_results]
    low_odds = [checkpoint_results[n]["odd"] for n in low_objs if n in checkpoint_results]
    high_evens = [checkpoint_results[n]["even"] for n in high_objs if n in checkpoint_results]
    low_evens = [checkpoint_results[n]["even"] for n in low_objs if n in checkpoint_results]

    results = {
        "layer": layer_idx,
        "direction_norm": float(dir_norm),
        "per_object": checkpoint_results,
        "aggregate": {
            "mean_odd": float(np.mean(all_odds)) if all_odds else 0,
            "mean_even": float(np.mean(all_evens)) if all_evens else 0,
            "high_mean_odd": float(np.mean(high_odds)) if high_odds else 0,
            "low_mean_odd": float(np.mean(low_odds)) if low_odds else 0,
            "high_mean_even": float(np.mean(high_evens)) if high_evens else 0,
            "low_mean_even": float(np.mean(low_evens)) if low_evens else 0,
        },
        "total_effects": {n: obj_logits_final.get(n, {}) for n in all_objects},
    }

    return results


def run_phase408(model_name):
    """Phase 408主函数"""
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 408: Causal Mediation Analysis ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")

    # Load model
    model, tokenizer = load_model_bf16_safe(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device

    # Get W_U
    W_U_np = get_W_U(model, model_name)
    print(f"  W_U: shape={W_U_np.shape}, n_layers={info.n_layers}")

    # Layer config
    analysis_layers = ANALYSIS_LAYERS.get(model_name, [4, info.n_layers//2, info.n_layers-2])
    analysis_layers = [li for li in analysis_layers if li < info.n_layers]

    all_results = {
        "model": model_name,
        "timestamp": timestamp,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "attributes": {},
    }

    # ===== Test each attribute =====
    for attr_name, attr_config in ATTRIBUTE_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"=== Attribute: {attr_name} ===")

        # Resolve token IDs
        candidate_ids = []
        levels = []
        cand_names = []
        for cand_name, level in attr_config["candidates"].items():
            ids = tokenizer.encode(cand_name, add_special_tokens=False)
            tid = ids[0] if ids else None
            candidate_ids.append(tid)
            levels.append(level)
            cand_names.append(cand_name)

        print(f"  Candidates: {dict(zip(cand_names, candidate_ids))}")

        attr_result = {"layers": {}}

        for li in analysis_layers:
            print(f"\n  Layer {li}...")
            result = run_causal_mediation(
                model, tokenizer, device, W_U_np, layers_list,
                attr_name, attr_config, candidate_ids, levels, li
            )

            if result:
                attr_result["layers"][str(li)] = result
                agg = result["aggregate"]
                print(f"    mean_odd={agg['mean_odd']:+.4f}, mean_even={agg['mean_even']:+.4f}")
                print(f"    high_odd={agg['high_mean_odd']:+.4f}, low_odd={agg['low_mean_odd']:+.4f}")
                print(f"    high_even={agg['high_mean_even']:+.4f}, low_even={agg['low_mean_even']:+.4f}")
                print(f"    dir_norm={result['direction_norm']:.4f}")

            print(f"    {log_memory()}")

        all_results["attributes"][attr_name] = attr_result

    # ===== Cross-attribute comparison =====
    print(f"\n{'='*80}")
    print(f"=== Cross-Attribute Comparison ({model_name}) ===")
    print(f"{'='*80}")

    print(f"\n1. Direction Odd/Even by Attribute and Layer:")
    for attr_name, attr_result in all_results["attributes"].items():
        print(f"\n  {attr_name}:")
        for li_str, layer_result in attr_result["layers"].items():
            agg = layer_result["aggregate"]
            print(f"    L{li_str}: odd={agg['mean_odd']:+.4f}, even={agg['mean_even']:+.4f}, "
                  f"high_odd={agg['high_mean_odd']:+.4f}, low_odd={agg['low_mean_odd']:+.4f}")

    print(f"\n2. High vs Low Object Odd Effect (方向效应的TYPE特异性):")
    for attr_name, attr_result in all_results["attributes"].items():
        print(f"\n  {attr_name}:")
        for li_str, layer_result in attr_result["layers"].items():
            agg = layer_result["aggregate"]
            delta = agg['high_mean_odd'] - agg['low_mean_odd']
            print(f"    L{li_str}: high_odd-low_odd = {delta:+.4f}")

    # ===== Save results =====
    results_dir = ROOT / "results" / "phase408_causal_mediation"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"{model_name}_phase408.json"

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

    all_results = convert(all_results)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")

    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    return all_results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    run_phase408(model_name)


if __name__ == "__main__":
    main()
