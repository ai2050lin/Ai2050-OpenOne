"""
Phase 416-R2: Large-Scale Random Token Confirmation Test
=========================================================

Phase 416-R1发现: 随机token仍有非对称性, 但方向跨模型不一致。
问题: 选出的"随机token"不是真正随机的(如caric, retard, QPointF有语义)。

R2策略:
1. 扩大随机token数量: 30个(15 LOW + 15 HIGH), 取平均消除个别偏见
2. 精确筛选: 排除有明确语义的token, 只用无意义子词碎片
3. 只测试temperature属性(最强信号)
4. 3种条件: L0(定义), L4(定义+反转), 计算3组的asymmetry

如果30个随机token的平均asymmetry接近0 → 知识锚定+嵌入偏见是唯一来源
如果仍偏离0 → W_U方向结构或其他因素

Usage:
  python tests/glm5/phase416_r2_large_random.py qwen3
  python tests/glm5/phase416_r2_large_random.py glm4
  python tests/glm5/phase416_r2_large_random.py deepseek7b
"""

import sys
import os
import json
import time
import gc
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS, get_model_info


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[bf16] Loading {model_name}...")

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
    print(f"[bf16] {model_name}: class={type(model).__name__}, GPU={gpu_mem:.2f}GB")

    return model, tokenizer, device


TEMPERATURE_CANDIDATES = OrderedDict([
    ("freezing", 1), ("cold", 2), ("cool", 3), ("warm", 4), ("hot", 5), ("scorching", 6),
])


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


def find_neutral_tokens(tokenizer, vocab_size, n_needed=30):
    """找到中性token: 无明确语义的子词碎片
    
    筛选条件:
    1. 长度3-6个字符
    2. 全小写字母
    3. 不含常见英文单词
    4. 不是常见缩写
    5. 在vocab中间区域(tid > vocab_size/3)
    """
    common_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "has", "his", "how", "its", "may",
        "new", "now", "old", "see", "way", "who", "did", "get", "got", "let",
        "say", "she", "too", "use", "yes", "car", "set", "try", "ask", "men",
        "ran", "own", "put", "end", "day", "far", "few", "got", "run", "top",
        "red", "big", "hot", "cut", "sit", "lot", "low", "bad", "fun", "map",
        "bed", "bit", "hat", "cat", "dog", "sun", "cup", "box", "bag", "pen",
    }
    
    neutral_tokens = []
    start_id = vocab_size // 3  # 避开高频区域
    
    for tid in range(start_id, vocab_size):
        if len(neutral_tokens) >= n_needed:
            break
            
        decoded = tokenizer.decode([tid]).strip()
        
        # 筛选条件
        if not decoded:
            continue
        if not decoded.isalpha():
            continue
        if not decoded.islower():
            continue
        if len(decoded) < 3 or len(decoded) > 6:
            continue
        if decoded in common_words:
            continue
        # 排除含常见后缀的词
        if any(decoded.endswith(s) for s in ["ing", "tion", "ment", "ness", "able", "ful"]):
            continue
        # 排除含常见前缀的词
        if any(decoded.startswith(s) for s in ["un", "re", "pre", "dis", "over", "non"]):
            continue
        
        neutral_tokens.append((tid, decoded))
    
    print(f"  Found {len(neutral_tokens)} neutral tokens (from tid={start_id}+)")
    return neutral_tokens


def test_prompt(model, tokenizer, device, prompt, cand_ids, candidates):
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        next_logits = outputs.logits[0, -1, :]

    probs = get_probs(next_logits.cpu(), cand_ids)
    level = compute_level(probs, candidates)
    return level, probs


def run_phase416_r2(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 416-R2: Large-Scale Random Token ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    
    candidates = TEMPERATURE_CANDIDATES
    cand_ids = get_cand_ids(tokenizer, candidates)
    
    # 找30个中性token
    neutral_tokens = find_neutral_tokens(tokenizer, info.vocab_size, n_needed=30)
    
    if len(neutral_tokens) < 10:
        print(f"WARNING: Only found {len(neutral_tokens)} neutral tokens, results may be unreliable")
    
    n_tokens = len(neutral_tokens)
    n_low = n_tokens // 2
    n_high = n_tokens - n_low
    
    # 分配极性
    low_tokens = neutral_tokens[:n_low]    # 定义为cold
    high_tokens = neutral_tokens[n_low:]   # 定义为hot
    
    print(f"\n  LOW tokens ({n_low}): {[t[1] for t in low_tokens[:5]]}...")
    print(f"  HIGH tokens ({n_high}): {[t[1] for t in high_tokens[:5]]}...")
    
    results = {
        "model": model_name,
        "timestamp": timestamp,
        "phase": "416-R2",
        "n_tokens": n_tokens,
        "low_tokens": [(tid, name) for tid, name in low_tokens],
        "high_tokens": [(tid, name) for tid, name in high_tokens],
        "per_token": {},
    }
    
    # ===== 测试每个token =====
    low_L0_levels = []
    low_L4_levels = []
    high_L0_levels = []
    high_L4_levels = []
    
    for i, (tid, name) in enumerate(neutral_tokens):
        polarity = "low" if i < n_low else "high"
        defined_level = 2 if polarity == "low" else 5
        defined_word = "cold" if polarity == "low" else "hot"
        definition = f"A {name} is a thing whose temperature is {defined_word}."
        
        # L0: 定义
        prompt_L0 = definition + "\nThe " + name + " is"
        level_L0, probs_L0 = test_prompt(model, tokenizer, device, prompt_L0, cand_ids, candidates)
        
        # L4: 定义 + 反转
        if polarity == "low":
            # LOW对象: 反转为hot
            rule = (f"Q: Is {name} hot or cold in this world? A: {name} is scorching hot.\n"
                    f"Q: Is {low_tokens[0][1]} hot or cold in this world? A: {low_tokens[0][1]} is scorching hot.\n"
                    f"Q: Is {high_tokens[0][1]} hot or cold in this world? A: {high_tokens[0][1]} is freezing cold.\n")
        else:
            # HIGH对象: 反转为cold
            rule = (f"Q: Is {name} hot or cold in this world? A: {name} is freezing cold.\n"
                    f"Q: Is {high_tokens[0][1]} hot or cold in this world? A: {high_tokens[0][1]} is freezing cold.\n"
                    f"Q: Is {low_tokens[0][1]} hot or cold in this world? A: {low_tokens[0][1]} is scorching hot.\n")
        
        prompt_L4 = definition + "\n" + rule + "The " + name + " is"
        level_L4, probs_L4 = test_prompt(model, tokenizer, device, prompt_L4, cand_ids, candidates)
        
        delta = level_L4 - level_L0
        
        if polarity == "low":
            low_L0_levels.append(level_L0)
            low_L4_levels.append(level_L4)
        else:
            high_L0_levels.append(level_L0)
            high_L4_levels.append(level_L4)
        
        results["per_token"][name] = {
            "tid": tid,
            "polarity": polarity,
            "L0_level": level_L0,
            "L4_level": level_L4,
            "delta": delta,
        }
        
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{n_tokens}] {name}({polarity}): L0={level_L0:.3f} L4={level_L4:.3f} delta={delta:+.3f}")
            # 定期输出GPU状态
            if torch.cuda.is_available():
                gpu = torch.cuda.memory_allocated() / 1e9
                print(f"    GPU: {gpu:.2f}GB")
    
    # ===== 统计分析 =====
    low_deltas = [results["per_token"][t[1]]["delta"] for t in low_tokens]
    high_deltas = [results["per_token"][t[1]]["delta"] for t in high_tokens]
    
    # up-reversal: LOW对象的delta (应>0, 从cold推向hot)
    up_mean = float(np.mean(low_deltas))
    # down-reversal: HIGH对象的delta绝对值 (应<0, 从hot推向cold)
    down_mean = float(np.mean([abs(d) for d in high_deltas]))
    asymmetry = up_mean - down_mean
    
    print(f"\n{'='*60}")
    print(f"=== Results ({model_name}) ===")
    print(f"{'='*60}")
    print(f"  n_low={n_low}, n_high={n_high}")
    print(f"  LOW tokens  (up-reversal):   mean_delta = {up_mean:+.3f}")
    print(f"  HIGH tokens (down-reversal):  mean_|delta| = {down_mean:+.3f}")
    print(f"  Asymmetry (up - down):        {asymmetry:+.3f}")
    print(f"  LOW L0 mean:   {float(np.mean(low_L0_levels)):.3f}")
    print(f"  HIGH L0 mean:  {float(np.mean(high_L0_levels)):.3f}")
    
    # 检查L0基线是否居中
    l0_mean_all = float(np.mean(low_L0_levels + high_L0_levels))
    print(f"  Overall L0 mean: {l0_mean_all:.3f} (midpoint=3.5)")
    
    if abs(asymmetry) < 0.2:
        verdict = "NEAR ZERO → No structural asymmetry without knowledge"
    elif abs(asymmetry) < 0.5:
        verdict = "SMALL → Mostly knowledge/embedding driven"
    else:
        verdict = "SIGNIFICANT → Structural factors exist beyond knowledge"
    
    print(f"\n  VERDICT: {verdict}")
    
    results["summary"] = {
        "up_mean": up_mean,
        "down_mean": down_mean,
        "asymmetry": asymmetry,
        "low_L0_mean": float(np.mean(low_L0_levels)),
        "high_L0_mean": float(np.mean(high_L0_levels)),
        "overall_L0_mean": l0_mean_all,
        "verdict": verdict,
    }
    
    # 保存
    results_dir = ROOT / "results" / "phase416_neutral_control"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_name}_phase416_r2.json"
    
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
    print(f"  Saved to {out_path}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    run_phase416_r2(model_name)


if __name__ == "__main__":
    main()
