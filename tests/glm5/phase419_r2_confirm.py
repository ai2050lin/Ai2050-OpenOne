"""
Phase 419-R2: 大规模Token轨道图确认测试
========================================

R1发现: 
- Qwen系(Qwen3/DS7B): down-reversal更容易 (asymmetry < 0)
- GLM4: up-reversal更容易 (asymmetry > 0)
- Size属性模式最强

R2: 每属性100个token(50 LOW + 50 HIGH), 只测temperature和speed(最强信号)
总计: 200 tokens, 扩大样本量验证架构分叉

Usage:
  python tests/glm5/phase419_r2_confirm.py qwen3
  python tests/glm5/phase419_r2_confirm.py glm4
  python tests/glm5/phase419_r2_confirm.py deepseek7b
"""

import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import time
import gc
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS, get_model_info


ATTRIBUTES = {
    "temperature": OrderedDict([
        ("freezing", 1), ("cold", 2), ("cool", 3), ("warm", 4), ("hot", 5), ("scorching", 6),
    ]),
    "speed": OrderedDict([
        ("stationary", 1), ("slow", 2), ("moderate", 3), ("fast", 4), ("swift", 5), ("lightning", 6),
    ]),
}

ATTR_TEMPLATES = {
    "temperature": {
        "L0_low": "A {name} is a thing whose temperature is cold.",
        "L0_high": "A {name} is a thing whose temperature is hot.",
        "L4_low2high": "Q: Is {name} hot or cold in this world? A: {name} is scorching hot.\nQ: Is {ref_low} hot or cold in this world? A: {ref_low} is scorching hot.\nQ: Is {ref_high} hot or cold in this world? A: {ref_high} is freezing cold.\n",
        "L4_high2low": "Q: Is {name} hot or cold in this world? A: {name} is freezing cold.\nQ: Is {ref_high} hot or cold in this world? A: {ref_high} is freezing cold.\nQ: Is {ref_low} hot or cold in this world? A: {ref_low} is scorching hot.\n",
        "query": "The {name} is",
    },
    "speed": {
        "L0_low": "A {name} is a thing whose speed is slow.",
        "L0_high": "A {name} is a thing whose speed is fast.",
        "L4_low2high": "Q: Is {name} fast or slow in this world? A: {name} is lightning fast.\nQ: Is {ref_low} fast or slow in this world? A: {ref_low} is lightning fast.\nQ: Is {ref_high} fast or slow in this world? A: {ref_high} is completely stationary.\n",
        "L4_high2low": "Q: Is {name} fast or slow in this world? A: {name} is completely stationary.\nQ: Is {ref_high} fast or slow in this world? A: {ref_high} is completely stationary.\nQ: Is {ref_low} fast or slow in this world? A: {ref_low} is lightning fast.\n",
        "query": "The {name} is",
    },
}


def load_model_bf16(model_name):
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
            print(f"  {impl} failed: {e}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  device={device}, GPU={gpu_mem:.2f}GB")

    return model, tokenizer, device


def find_low_freq_tokens(tokenizer, vocab_size, n_needed=200, seed=123):
    """R2: 不同seed, 更多token"""
    rng = np.random.RandomState(seed)
    
    common_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "has", "his", "how", "its", "may",
        "new", "now", "old", "see", "way", "who", "did", "get", "got", "let",
        "say", "she", "too", "use", "yes", "car", "set", "try", "ask", "men",
        "ran", "own", "put", "end", "day", "far", "few", "run", "top", "red",
        "big", "hot", "cut", "sit", "lot", "low", "bad", "fun", "map", "bed",
        "bit", "hat", "cat", "dog", "sun", "cup", "box", "bag", "pen", "six",
        "cold", "warm", "cool", "slow", "fast", "tiny", "huge", "deep", "high",
        "long", "wide", "dark", "rich", "poor", "safe", "weak", "wild", "soft",
        "hard", "free", "full", "glad", "keen", "loud", "pure", "real", "thin",
    }
    
    bad_suffixes = ["ing", "tion", "ment", "ness", "able", "ful", "less", "ous", "ive", "ize", "ise", "ate", "ent", "ant", "ary", "ory", "ical", "ally", "ting", "ned", "red", "sed"]
    bad_prefixes = ["un", "re", "pre", "dis", "over", "non", "anti", "semi", "sub", "inter", "multi", "super"]
    
    candidates = []
    start_id = vocab_size // 4
    
    for tid in range(start_id, vocab_size):
        try:
            decoded = tokenizer.decode([tid])
            if decoded is None:
                continue
            decoded = decoded.strip()
        except Exception:
            continue
        
        if not decoded:
            continue
        if not all(c.isalpha() and ord(c) < 128 for c in decoded):
            continue
        if not decoded.islower():
            continue
        if len(decoded) < 3 or len(decoded) > 6:
            continue
        if decoded in common_words:
            continue
        if any(decoded.endswith(s) for s in bad_suffixes):
            continue
        if any(decoded.startswith(s) for s in bad_prefixes):
            continue
        if len(set(decoded)) < 2:
            continue
        
        candidates.append((tid, decoded))
    
    print(f"  Found {len(candidates)} candidate tokens")
    
    if len(candidates) <= n_needed:
        return candidates
    
    indices = rng.choice(len(candidates), size=n_needed, replace=False)
    selected = [candidates[i] for i in sorted(indices)]
    return selected


def get_cand_ids(tokenizer, candidates):
    cand_ids = {}
    for cand in candidates:
        for prefix in [" ", ""]:
            ids = tokenizer.encode(prefix + cand, add_special_tokens=False)
            if ids:
                cand_ids[cand] = ids[-1]
                break
    return cand_ids


def get_probs(logits_tensor, cand_ids):
    probs = torch.softmax(logits_tensor.float(), dim=-1)
    result = {}
    for cand, tid in cand_ids.items():
        if tid < probs.shape[-1]:
            result[cand] = float(probs[tid].item())
    total = sum(result.values())
    if total > 0:
        for k in result:
            result[k] /= total
    return result


def compute_level(probs, candidates):
    return sum(candidates[c] * probs.get(c, 0.0) for c in candidates)


def test_prompt(model, tokenizer, device, prompt, cand_ids, candidates):
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        next_logits = outputs.logits[0, -1, :]
    
    probs = get_probs(next_logits.cpu(), cand_ids)
    level = compute_level(probs, candidates)
    return level, probs


def run_phase419_r2(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 419-R2: 确认测试 ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")
    
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    
    # 200个token, 分2组(temperature/speed各100: 50LOW+50HIGH)
    print(f"\n[{time.strftime('%H:%M:%S')}] Finding 200 low-frequency tokens (seed=123)...")
    all_tokens = find_low_freq_tokens(tokenizer, info.vocab_size, n_needed=200, seed=123)
    
    n_per_attr = min(100, len(all_tokens) // 2)
    n_half = n_per_attr // 2
    
    results = {
        "model": model_name,
        "model_class": info.model_class,
        "timestamp": timestamp,
        "phase": "419-R2",
        "n_per_attr": n_per_attr,
        "attributes": {},
    }
    
    for attr_idx, attr_name in enumerate(["temperature", "speed"]):
        print(f"\n[{time.strftime('%H:%M:%S')}] === Testing {attr_name} ===")
        
        candidates = ATTRIBUTES[attr_name]
        cand_ids = get_cand_ids(tokenizer, candidates)
        templates = ATTR_TEMPLATES[attr_name]
        
        start = attr_idx * n_per_attr
        tokens = all_tokens[start:start + n_per_attr]
        low_tokens = tokens[:n_half]
        high_tokens = tokens[n_half:]
        
        print(f"  n_low={len(low_tokens)}, n_high={len(high_tokens)}")
        
        ref_low = low_tokens[0][1]
        ref_high = high_tokens[0][1]
        
        attr_data = {"per_token": {}, "summary": {}}
        
        low_deltas = []
        high_deltas = []
        low_L0_levels = []
        high_L0_levels = []
        
        all_tokens_this_attr = (
            [("low", t) for t in low_tokens] + [("high", t) for t in high_tokens]
        )
        
        for idx, (polarity, (tid, name)) in enumerate(all_tokens_this_attr):
            if polarity == "low":
                L0_prompt = templates["L0_low"].format(name=name)
                L4_rule = templates["L4_low2high"].format(name=name, ref_low=ref_low, ref_high=ref_high)
            else:
                L0_prompt = templates["L0_high"].format(name=name)
                L4_rule = templates["L4_high2low"].format(name=name, ref_low=ref_low, ref_high=ref_high)
            
            query = templates["query"].format(name=name)
            
            prompt_L0 = L0_prompt + "\n" + query
            level_L0, _ = test_prompt(model, tokenizer, device, prompt_L0, cand_ids, candidates)
            
            prompt_L4 = L0_prompt + "\n" + L4_rule + query
            level_L4, _ = test_prompt(model, tokenizer, device, prompt_L4, cand_ids, candidates)
            
            delta = level_L4 - level_L0
            
            if polarity == "low":
                low_L0_levels.append(level_L0)
                low_deltas.append(delta)
            else:
                high_L0_levels.append(level_L0)
                high_deltas.append(delta)
            
            if (idx + 1) % 20 == 0:
                gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(f"  [{idx+1}/{len(all_tokens_this_attr)}] {name}({polarity}): "
                      f"L0={level_L0:.3f} L4={level_L4:.3f} delta={delta:+.3f} | GPU={gpu:.2f}GB")
        
        # 统计
        up_mean = float(np.mean(low_deltas))
        down_abs_mean = float(np.mean([abs(d) for d in high_deltas]))
        asymmetry = up_mean - down_abs_mean
        
        # Bootstrap CI
        n_bootstrap = 2000
        rng = np.random.RandomState(42)
        asym_samples = []
        for _ in range(n_bootstrap):
            bs_low = rng.choice(low_deltas, size=len(low_deltas), replace=True)
            bs_high = rng.choice(high_deltas, size=len(high_deltas), replace=True)
            asym_samples.append(float(np.mean(bs_low)) - float(np.mean([abs(d) for d in bs_high])))
        
        ci_low = float(np.percentile(asym_samples, 2.5))
        ci_high = float(np.percentile(asym_samples, 97.5))
        
        # Per-token方差
        delta_std = float(np.std(low_deltas + [abs(d) for d in high_deltas]))
        
        # 反转成功率
        up_success = sum(1 for d in low_deltas if d > 1.0) / max(len(low_deltas), 1)
        down_success = sum(1 for d in high_deltas if d < -1.0) / max(len(high_deltas), 1)
        
        # 离群分析: 排除最大最小5%后重算
        low_sorted = sorted(low_deltas)
        high_sorted = sorted([abs(d) for d in high_deltas])
        trim = max(1, len(low_sorted) // 10)
        low_trimmed = low_sorted[trim:-trim] if len(low_sorted) > 2*trim+1 else low_sorted
        high_trimmed = high_sorted[trim:-trim] if len(high_sorted) > 2*trim+1 else high_sorted
        asym_trimmed = float(np.mean(low_trimmed)) - float(np.mean(high_trimmed))
        
        print(f"\n  === {attr_name} Summary (R2) ===")
        print(f"  n_low={len(low_deltas)}, n_high={len(high_deltas)}")
        print(f"  LOW L0 mean:  {float(np.mean(low_L0_levels)):.3f}")
        print(f"  HIGH L0 mean: {float(np.mean(high_L0_levels)):.3f}")
        print(f"  Up-reversal:   {up_mean:+.3f}")
        print(f"  Down-reversal: {down_abs_mean:+.3f}")
        print(f"  Asymmetry:     {asymmetry:+.3f}")
        print(f"  Bootstrap CI:  [{ci_low:+.3f}, {ci_high:+.3f}]")
        print(f"  Trimmed asym:  {asym_trimmed:+.3f}")
        print(f"  Up success:    {up_success*100:.1f}%")
        print(f"  Down success:  {down_success*100:.1f}%")
        print(f"  Delta std:     {delta_std:.3f}")
        
        attr_data["summary"] = {
            "n_low": len(low_deltas),
            "n_high": len(high_deltas),
            "low_L0_mean": round(float(np.mean(low_L0_levels)), 4),
            "high_L0_mean": round(float(np.mean(high_L0_levels)), 4),
            "up_mean": round(up_mean, 4),
            "down_abs_mean": round(down_abs_mean, 4),
            "asymmetry": round(asymmetry, 4),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "asym_trimmed": round(asym_trimmed, 4),
            "up_success_rate": round(up_success, 4),
            "down_success_rate": round(down_success, 4),
            "delta_std": round(delta_std, 4),
        }
        
        results["attributes"][attr_name] = attr_data
    
    # 保存
    results_dir = ROOT / "results" / "phase419_token_trajectory"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_name}_phase419_r2.json"
    
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
    
    results = convert(results)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Model released. GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    return results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    
    run_phase419_r2(model_name)


if __name__ == "__main__":
    main()
