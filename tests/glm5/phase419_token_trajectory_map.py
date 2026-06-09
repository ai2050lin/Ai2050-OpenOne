"""
Phase 419: 大规模随机Token轨道图
=================================

Phase 416发现: 随机token仍有非对称性, 且跨模型不一致。
关键问题: token embedding如何被吸入已有语义轨道?

本实验目标:
1. 大规模采样: 200个低频token, 分3组(temperature/speed/size)
2. 每个token记录: 初始偏向(无定义), 定义后偏向, 反转delta
3. 构建 token → semantic basin 映射
4. 计算bootstrap置信区间
5. 分析轨道捕获模式: 哪些token被吸入cold/hot轨道? 模式是否可预测?

R1: 基础测试 - 每属性60个token(30 LOW定义 + 30 HIGH定义), 3属性 = 180 tokens
R2: 如R1发现重要模式, 扩大到500 tokens

Usage:
  python tests/glm5/phase419_token_trajectory_map.py qwen3
  python tests/glm5/phase419_token_trajectory_map.py glm4
  python tests/glm5/phase419_token_trajectory_map.py deepseek7b
"""

import sys
import os

# Windows UTF-8 输出
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


# ===== 属性候选词 =====
ATTRIBUTES = {
    "temperature": OrderedDict([
        ("freezing", 1), ("cold", 2), ("cool", 3), ("warm", 4), ("hot", 5), ("scorching", 6),
    ]),
    "speed": OrderedDict([
        ("stationary", 1), ("slow", 2), ("moderate", 3), ("fast", 4), ("swift", 5), ("lightning", 6),
    ]),
    "size": OrderedDict([
        ("microscopic", 1), ("tiny", 2), ("small", 3), ("large", 4), ("huge", 5), ("massive", 6),
    ]),
}

# 属性的LOW/HIGH定义词
ATTR_DEFINE = {
    "temperature": {"low": "cold", "high": "hot"},
    "speed": {"low": "slow", "high": "fast"},
    "size": {"low": "small", "high": "big"},
}

# 属性的提示模板
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
    "size": {
        "L0_low": "A {name} is a thing whose size is small.",
        "L0_high": "A {name} is a thing whose size is big.",
        "L4_low2high": "Q: Is {name} big or small in this world? A: {name} is massive.\nQ: Is {ref_low} big or small in this world? A: {ref_low} is massive.\nQ: Is {ref_high} big or small in this world? A: {ref_high} is microscopic.\n",
        "L4_high2low": "Q: Is {name} big or small in this world? A: {name} is microscopic.\nQ: Is {ref_high} big or small in this world? A: {ref_high} is microscopic.\nQ: Is {ref_low} big or small in this world? A: {ref_low} is massive.\n",
        "query": "The {name} is",
    },
}


def load_model_bf16(model_name):
    """BF16 + device_map=auto 加载模型"""
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


def find_low_freq_tokens(tokenizer, vocab_size, n_needed=200, seed=42):
    """筛选低频无语义token
    
    策略:
    1. 扫描vocab中间到后段(token id > vocab_size/4)
    2. 只保留3-6字符的全小写字母串
    3. 排除常见英文单词、常见前后缀
    4. 随机采样n_needed个
    """
    rng = np.random.RandomState(seed)
    
    common_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
        "her", "was", "one", "our", "out", "has", "his", "how", "its", "may",
        "new", "now", "old", "see", "way", "who", "did", "get", "got", "let",
        "say", "she", "too", "use", "yes", "car", "set", "try", "ask", "men",
        "ran", "own", "put", "end", "day", "far", "few", "run", "top", "red",
        "big", "hot", "cut", "sit", "lot", "low", "bad", "fun", "map", "bed",
        "bit", "hat", "cat", "dog", "sun", "cup", "box", "bag", "pen", "six",
        "ten", "air", "arm", "art", "bar", "bus", "buy", "die", "eat", "era",
        "eye", "fan", "fit", "fix", "fly", "gas", "gun", "gym", "hip", "hit",
        "ice", "job", "joy", "key", "kid", "kit", "lab", "lap", "law", "lay",
        "leg", "lip", "log", "mad", "mix", "mob", "net", "nut", "odd", "oil",
        "pan", "pay", "pet", "pie", "pin", "pit", "pop", "pot", "pub", "raw",
        "rid", "row", "sad", "sea", "sin", "sir", "sky", "son", "spy", "sum",
        "tab", "tap", "tax", "tea", "tie", "tip", "toe", "ton", "toy", "van",
        "via", "war", "wet", "win", "wit", "wow", "yard", "zip", "zoo",
        "cold", "warm", "cool", "slow", "fast", "tiny", "huge", "deep", "high",
        "long", "wide", "dark", "rich", "poor", "safe", "weak", "wild", "soft",
        "hard", "free", "full", "glad", "keen", "loud", "pure", "real", "thin",
    }
    
    # 常见后缀/前缀
    bad_suffixes = ["ing", "tion", "ment", "ness", "able", "ful", "less", "ous", "ive", "ize", "ise", "ate", "ent", "ant", "ary", "ory", "ical", "ally", "ting", "ned", "red", "sed"]
    bad_prefixes = ["un", "re", "pre", "dis", "over", "non", "anti", "semi", "sub", "inter", "multi", "super"]
    
    candidates = []
    start_id = vocab_size // 4  # 避开最高频区域
    
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
        # 只保留纯ASCII小写字母
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
        # 排除重复字符过多(如aaa, abab)
        if len(set(decoded)) < 2:
            continue
        
        candidates.append((tid, decoded))
    
    print(f"  Found {len(candidates)} candidate tokens (from tid={start_id}+)")
    
    if len(candidates) <= n_needed:
        return candidates
    
    # 随机采样
    indices = rng.choice(len(candidates), size=n_needed, replace=False)
    selected = [candidates[i] for i in sorted(indices)]
    return selected


def get_cand_ids(tokenizer, candidates):
    """获取候选词的token IDs"""
    cand_ids = {}
    for cand in candidates:
        for prefix in [" ", ""]:
            ids = tokenizer.encode(prefix + cand, add_special_tokens=False)
            if ids:
                cand_ids[cand] = ids[-1]
                break
    return cand_ids


def get_probs(logits_tensor, cand_ids):
    """从logits提取候选词概率"""
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
    """计算属性等级"""
    return sum(candidates[c] * probs.get(c, 0.0) for c in candidates)


def test_prompt(model, tokenizer, device, prompt, cand_ids, candidates):
    """测试单个prompt, 返回level和概率分布"""
    input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        next_logits = outputs.logits[0, -1, :]
    
    probs = get_probs(next_logits.cpu(), cand_ids)
    level = compute_level(probs, candidates)
    return level, probs


def run_phase419(model_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 419: 大规模Token轨道图 ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")
    
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    
    # ===== 1. 筛选200个低频token =====
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 1: Finding low-frequency tokens...")
    all_tokens = find_low_freq_tokens(tokenizer, info.vocab_size, n_needed=200, seed=42)
    n_total = len(all_tokens)
    
    # 按属性分组: 每属性约60个token(30 LOW定义 + 30 HIGH定义)
    n_per_attr = min(60, n_total // 3)
    n_half = n_per_attr // 2
    
    attr_tokens = {}
    for i, attr in enumerate(["temperature", "speed", "size"]):
        start = i * n_per_attr
        end = start + n_per_attr
        tokens = all_tokens[start:end]
        low_tokens = tokens[:n_half]     # 定义为LOW
        high_tokens = tokens[n_half:]    # 定义为HIGH
        attr_tokens[attr] = {"low": low_tokens, "high": high_tokens}
        print(f"  {attr}: {len(low_tokens)} LOW + {len(high_tokens)} HIGH tokens")
        print(f"    LOW sample: {[t[1] for t in low_tokens[:5]]}")
        print(f"    HIGH sample: {[t[1] for t in high_tokens[:5]]}")
    
    # ===== 2. 测试每个token =====
    results = {
        "model": model_name,
        "model_class": info.model_class,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "vocab_size": info.vocab_size,
        "timestamp": timestamp,
        "phase": "419",
        "n_total_tokens": n_total,
        "n_per_attr": n_per_attr,
        "attributes": {},
    }
    
    for attr_name in ["temperature", "speed", "size"]:
        print(f"\n[{time.strftime('%H:%M:%S')}] === Testing {attr_name} ===")
        
        candidates = ATTRIBUTES[attr_name]
        cand_ids = get_cand_ids(tokenizer, candidates)
        templates = ATTR_TEMPLATES[attr_name]
        
        attr_data = {
            "low_tokens": [],
            "high_tokens": [],
            "per_token": {},
            "summary": {},
        }
        
        # 参考token(用于L4规则的Q&A)
        ref_low = attr_tokens[attr_name]["low"][0][1]
        ref_high = attr_tokens[attr_name]["high"][0][1]
        
        # 测试所有token
        low_L0_levels = []
        low_L4_levels = []
        high_L0_levels = []
        high_L4_levels = []
        low_deltas = []
        high_deltas = []
        
        all_tokens_this_attr = (
            [("low", t) for t in attr_tokens[attr_name]["low"]] +
            [("high", t) for t in attr_tokens[attr_name]["high"]]
        )
        
        for idx, (polarity, (tid, name)) in enumerate(all_tokens_this_attr):
            # 构建prompt
            if polarity == "low":
                L0_prompt = templates["L0_low"].format(name=name)
                L4_rule = templates["L4_low2high"].format(name=name, ref_low=ref_low, ref_high=ref_high)
            else:
                L0_prompt = templates["L0_high"].format(name=name)
                L4_rule = templates["L4_high2low"].format(name=name, ref_low=ref_low, ref_high=ref_high)
            
            query = templates["query"].format(name=name)
            
            # L0: 定义
            prompt_L0 = L0_prompt + "\n" + query
            level_L0, probs_L0 = test_prompt(model, tokenizer, device, prompt_L0, cand_ids, candidates)
            
            # L4: 定义 + 反转
            prompt_L4 = L0_prompt + "\n" + L4_rule + query
            level_L4, probs_L4 = test_prompt(model, tokenizer, device, prompt_L4, cand_ids, candidates)
            
            delta = level_L4 - level_L0
            
            # 记录
            token_data = {
                "tid": tid,
                "name": name,
                "polarity": polarity,
                "L0_level": round(level_L0, 4),
                "L4_level": round(level_L4, 4),
                "delta": round(delta, 4),
                "L0_probs": {k: round(v, 4) for k, v in probs_L0.items()},
                "L4_probs": {k: round(v, 4) for k, v in probs_L4.items()},
            }
            attr_data["per_token"][name] = token_data
            
            if polarity == "low":
                low_L0_levels.append(level_L0)
                low_L4_levels.append(level_L4)
                low_deltas.append(delta)
            else:
                high_L0_levels.append(level_L0)
                high_L4_levels.append(level_L4)
                high_deltas.append(delta)
            
            # 日志
            if (idx + 1) % 10 == 0:
                gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(f"  [{idx+1}/{len(all_tokens_this_attr)}] {name}({polarity}): "
                      f"L0={level_L0:.3f} L4={level_L4:.3f} delta={delta:+.3f} | GPU={gpu:.2f}GB")
        
        # ===== 3. 统计分析 =====
        n_low = len(low_deltas)
        n_high = len(high_deltas)
        
        # 非对称性: up-reversal均值 - down-reversal均值
        up_mean = float(np.mean(low_deltas)) if low_deltas else 0.0
        down_abs_mean = float(np.mean([abs(d) for d in high_deltas])) if high_deltas else 0.0
        asymmetry = up_mean - down_abs_mean
        
        # Bootstrap置信区间 (1000次重采样)
        n_bootstrap = 1000
        rng = np.random.RandomState(42)
        asymmetry_samples = []
        for _ in range(n_bootstrap):
            if len(low_deltas) > 0 and len(high_deltas) > 0:
                bs_low = rng.choice(low_deltas, size=len(low_deltas), replace=True)
                bs_high = rng.choice(high_deltas, size=len(high_deltas), replace=True)
                bs_up = float(np.mean(bs_low))
                bs_down = float(np.mean([abs(d) for d in bs_high]))
                asymmetry_samples.append(bs_up - bs_down)
        
        if asymmetry_samples:
            ci_low = float(np.percentile(asymmetry_samples, 2.5))
            ci_high = float(np.percentile(asymmetry_samples, 97.5))
        else:
            ci_low, ci_high = 0.0, 0.0
        
        # L0基线分布
        low_L0_mean = float(np.mean(low_L0_levels)) if low_L0_levels else 0.0
        high_L0_mean = float(np.mean(high_L0_levels)) if high_L0_levels else 0.0
        
        # 轨道捕获分析: 每个token被吸入哪个方向?
        # 定义: L0_level > 3.5 = 被吸入HIGH轨道, L0_level < 3.5 = 被吸入LOW轨道
        capture_stats = {"low_captured_by_low": 0, "low_captured_by_high": 0,
                         "high_captured_by_low": 0, "high_captured_by_high": 0}
        for name, td in attr_data["per_token"].items():
            captured_high = td["L0_level"] > 3.5
            if td["polarity"] == "low":
                if captured_high:
                    capture_stats["low_captured_by_high"] += 1
                else:
                    capture_stats["low_captured_by_low"] += 1
            else:
                if captured_high:
                    capture_stats["high_captured_by_high"] += 1
                else:
                    capture_stats["high_captured_by_low"] += 1
        
        # 反转成功率: LOW→HIGH的delta>1, HIGH→LOW的delta<-1
        reversal_success = {
            "up_success_rate": sum(1 for d in low_deltas if d > 1.0) / max(len(low_deltas), 1),
            "down_success_rate": sum(1 for d in high_deltas if d < -1.0) / max(len(high_deltas), 1),
        }
        
        # Per-token方差
        low_delta_std = float(np.std(low_deltas)) if len(low_deltas) > 1 else 0.0
        high_delta_std = float(np.std(high_deltas)) if len(high_deltas) > 1 else 0.0
        
        print(f"\n  === {attr_name} Summary ===")
        print(f"  n_low={n_low}, n_high={n_high}")
        print(f"  LOW L0 mean:  {low_L0_mean:.3f} (defined=cold/slow/small → expect ~2)")
        print(f"  HIGH L0 mean: {high_L0_mean:.3f} (defined=hot/fast/big → expect ~5)")
        print(f"  Up-reversal mean:   {up_mean:+.3f} (LOW→HIGH delta)")
        print(f"  Down-reversal mean: {down_abs_mean:+.3f} (|HIGH→LOW delta|)")
        print(f"  Asymmetry (up-down): {asymmetry:+.3f}")
        print(f"  Bootstrap 95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]")
        print(f"  Delta std: LOW={low_delta_std:.3f}, HIGH={high_delta_std:.3f}")
        print(f"  Capture: {capture_stats}")
        print(f"  Reversal success: {reversal_success}")
        
        attr_data["summary"] = {
            "n_low": n_low,
            "n_high": n_high,
            "low_L0_mean": round(low_L0_mean, 4),
            "high_L0_mean": round(high_L0_mean, 4),
            "up_mean": round(up_mean, 4),
            "down_abs_mean": round(down_abs_mean, 4),
            "asymmetry": round(asymmetry, 4),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
            "low_delta_std": round(low_delta_std, 4),
            "high_delta_std": round(high_delta_std, 4),
            "capture_stats": capture_stats,
            "reversal_success": {k: round(v, 4) for k, v in reversal_success.items()},
        }
        
        results["attributes"][attr_name] = attr_data
    
    # ===== 4. 跨属性对比 =====
    print(f"\n{'='*80}")
    print(f"=== Cross-Attribute Summary ({model_name}) ===")
    print(f"{'='*80}")
    for attr_name in ["temperature", "speed", "size"]:
        s = results["attributes"][attr_name]["summary"]
        sig = "YES" if (s["ci_95_low"] > 0 or s["ci_95_high"] < 0) else "NO"
        print(f"  {attr_name}: asymmetry={s['asymmetry']:+.3f} "
              f"CI=[{s['ci_95_low']:+.3f}, {s['ci_95_high']:+.3f}] "
              f"significant={sig} "
              f"up={s['up_mean']:+.3f} down={s['down_abs_mean']:+.3f}")
    
    # ===== 5. 保存结果 =====
    results_dir = ROOT / "results" / "phase419_token_trajectory"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{model_name}_phase419.json"
    
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
    
    # ===== 6. 释放模型 =====
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
    
    run_phase419(model_name)


if __name__ == "__main__":
    main()
