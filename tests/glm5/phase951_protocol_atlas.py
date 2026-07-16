#!/usr/bin/env python3
"""
Phase 951: Protocol Field Physical Atlas and Semantic-to-Protocol Transition Audit
==================================================================================
协议场物理图谱与语义到协议转换审计

5个任务:
  Task1: 扩展Protocol token物理图谱 (160+ prompts, 8类)
  Task2: 最后层MLP通道级归因 (回归探针法, 无需权重访问)
  Task3: 语义方向 -> 协议通道桥接测试
  Task4: 语义层 -> 协议层信息流追踪
  Task5: Protocol-neutral prompt对照实验

设计原则:
  - 增量保存: 每个Task完成后立即写JSON
  - 回归探针: 用Ridge回归做通道归因, 避免meta device权重问题
  - 跨模型兼容: 支持 split_gate_up (qwen3/DS7B) 和 merged_gate_up (GLM4)
  - 大数据量: 160+ prompts, 40+ protocol tokens
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS

PHASE = 951
RESULT_DIR = Path("results/phase951_protocol_atlas")

# ============================================================
# PROMPT SET: 160+ prompts across 8 categories
# ============================================================

QA_PROMPTS = [
    "What is the capital of France? The answer is",
    "Who wrote Romeo and Juliet? The answer is",
    "What is the chemical symbol for water? The answer is",
    "Which planet is closest to the Sun? The answer is",
    "How many continents are there? The answer is",
    "What is the largest ocean? The answer is",
    "Who painted the Mona Lisa? The answer is",
    "What is the speed of light? The answer is",
    "Which element has atomic number 79? The answer is",
    "What is the longest river? The answer is",
    "In what year did World War II end? The answer is",
    "What is the square root of 144? The answer is",
    "Which country has the largest population? The answer is",
    "What is the boiling point of water in Celsius? The answer is",
    "Who discovered penicillin? The answer is",
    "What is the capital of Japan? The answer is",
    "How many sides does a hexagon have? The answer is",
    "What gas do plants absorb? The answer is",
    "Who developed the theory of relativity? The answer is",
    "What is the largest mammal? The answer is",
    "What is the freezing point of water in Fahrenheit? The answer is",
    "How many bones in the human body? The answer is",
    "What is the currency of England? The answer is",
    "Who composed Symphony No. 9? The answer is",
    "What is the smallest prime number? The answer is",
]

CLASSIFICATION_PROMPTS = [
    "Classify: 'apple' is a type of",
    "Category: 'dog' belongs to the category of",
    "Subclass: 'rose' is a kind of",
    "Category: 'hammer' is a",
    "Classify: 'Mars' is a",
    "Category: 'oxygen' is a",
    "Classify: 'Shakespeare' was a",
    "Subclass: 'diamond' is a type of",
    "Category: 'Python' (programming language) is a",
    "Classify: 'eagle' is a kind of",
    "Category: 'copper' is a",
    "Classify: 'novel' is a form of",
    "Subclass: 'bicycle' is a type of",
    "Category: 'triangle' is a",
    "Classify: 'Beethoven' was a",
    "Category: 'goldfish' is a",
    "Classify: 'oak' is a type of",
    "Subclass: 'marble' is a",
    "Category: 'helium' is a",
    "Classify: 'whale' is a",
]

EXPLANATION_PROMPTS = [
    "Explain why the sky is blue:",
    "What causes earthquakes?",
    "How does photosynthesis work?",
    "Why do we need sleep?",
    "What is artificial intelligence?",
    "How do vaccines work?",
    "Why is the ocean salty?",
    "What causes seasons?",
    "How does a computer work?",
    "Why do we have leap years?",
    "What is gravity?",
    "How do airplanes fly?",
    "Why does ice float?",
    "What is climate change?",
    "How does the immune system work?",
]

STRUCTURED_PROMPTS = [
    "List three primary colors:\n1.",
    "Name two types of energy:\n-",
    "List the planets in order:\n1.",
    "Name the components of blood:\n-",
    "List four cardinal directions:\n1.",
    "Name three states of matter:\n-",
    "List the noble gases:\n1.",
    "Name two types of cell division:\n-",
    "List the human senses:\n1.",
    "Name the parts of a plant:\n-",
    "List three types of rocks:\n1.",
    "Name the seasons:\n-",
    "List five countries in Europe:\n1.",
    "Name three programming languages:\n-",
    "List the Fibonacci sequence:\n1.",
]

PROTOCOL_CONTEXT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "The Earth revolves around the Sun once per year.",
    "Water freezes at zero degrees Celsius.",
    "Shakespeare wrote many famous plays.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The Industrial Revolution began in Britain.",
    "DNA contains the genetic instructions for life.",
    "Gravity is one of the four fundamental forces.",
    "The Renaissance was a period of great cultural change.",
    "Electricity powers most modern devices.",
    "The internet connects computers worldwide.",
    "Democracy originated in ancient Greece.",
    "The wheel was invented thousands of years ago.",
    "Languages evolve over centuries.",
]

MID_SENTENCE_PROMPTS = [
    "The quick brown fox",
    "Machine learning is",
    "The Earth revolves",
    "Water freezes",
    "Shakespeare wrote",
    "Photosynthesis converts",
    "The Industrial Revolution",
    "DNA contains",
    "Gravity is",
    "The Renaissance was",
    "Electricity powers",
    "The internet connects",
    "Democracy originated",
    "The wheel was",
    "Languages evolve",
]

CODE_PROMPTS = [
    "def fibonacci(n):",
    "import numpy as",
    "class Dog:",
    "for i in range",
    "if x > 0:",
    "return result",
    "print(",
    "x = [1, 2,",
    "while True:",
    "try:",
]

BILINGUAL_PROMPTS = [
    "什么是人工智能？",
    "水的化学式是",
    "法国的首都是",
    "地球绕着太阳",
    "光合作用是",
    "什么是重力？",
    "DNA包含",
    "工业革命始于",
    "文艺复兴是",
    "互联网连接",
]

ALL_PROMPT_GROUPS = {
    "qa": QA_PROMPTS,
    "classification": CLASSIFICATION_PROMPTS,
    "explanation": EXPLANATION_PROMPTS,
    "structured": STRUCTURED_PROMPTS,
    "protocol_context": PROTOCOL_CONTEXT_PROMPTS,
    "mid_sentence": MID_SENTENCE_PROMPTS,
    "code": CODE_PROMPTS,
    "bilingual": BILINGUAL_PROMPTS,
}

# ============================================================
# PROTOCOL TOKEN SET: expanded
# ============================================================

PROTOCOL_TOKEN_STRINGS = [
    # Punctuation
    ".", " .", ".\n", ",", " ,", ";", " ;", ":", " :", "!", " !", "?", " ?",
    "-", " -", "—", " –",
    # Brackets
    "(", " (", ")", " )", "[", " [", "]", " ]", "{", " {", "}", " }",
    # Whitespace
    " ", "\n", "\n\n", "\t",
    # Articles
    "the", " the", "a", " a", "an", " an", "The", " The", "A", " A",
    # Structural words
    "Answer", " Answer", "Solution", " Solution", "Step", " Step",
    "Category", " Category", "Item", " Item", "List", " List",
    "First", " First", "Second", " Second", "Third", " Third",
    # Connectors
    "and", " and", "or", " or", "but", " but", "because", " because",
    "so", " so", "therefore", " therefore",
    # Explanation starters
    "This", " This", "That", " That", "It", " It", "We", " We",
    "I", " I", "You", " You",
    # List markers
    "1", " 1", "\n1", "2", " 2", "\n2", "3", " 3",
    # Formatting
    "...", "...", "__", " __", "**", " **", "#", " #", "##", " ##",
    "|", " |", ">", " >",
]

# Semantic color words for Task 3
COLOR_WORDS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple"]

# Protocol-neutral prompt templates for Task 5
PROTOCOL_NEUTRAL_PROMPTS = [
    "Red",
    "Blue",
    "Apple",
    "Dog",
    "Cat",
    "Paris",
    "Tokyo",
    "Water",
    "Fire",
    "Earth",
    "One",
    "Two",
    "Yes",
    "No",
    "True",
    "False",
    "Hot",
    "Cold",
    "Big",
    "Small",
]

PROTOCOL_HEAVY_PROMPTS = [
    "The answer is red. The answer is",
    "The answer is blue. The answer is",
    "Category: apple. Category:",
    "Category: dog. Category:",
    "Answer: Paris. Answer:",
    "Answer: Tokyo. Answer:",
    "Solution: water. Solution:",
    "Solution: fire. Solution:",
    "Step 1: one. Step 2:",
    "Step 1: two. Step 2:",
    "Item 1: yes. Item 2:",
    "Item 1: no. Item 2:",
    "First: true. Second:",
    "First: false. Second:",
    "The color is hot. The color is",
    "The color is cold. The color is",
    "List: big. List:",
    "List: small. List:",
    "Result: red. Result:",
    "Result: blue. Result:",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_decode(tokenizer, tid: int) -> str:
    try:
        r = tokenizer.decode([tid], skip_special_tokens=False)
        return r if r else f"<tok_{tid}>"
    except:
        return f"<tok_{tid}>"


def get_protocol_token_ids(tokenizer) -> dict[str, list[int]]:
    """Get token IDs for protocol token strings."""
    result = {}
    for s in PROTOCOL_TOKEN_STRINGS:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                tid = ids[0]
                if str(tid) not in result:
                    result[str(tid)] = {"str": s, "id": tid}
        except:
            pass
    # Add EOS
    if tokenizer.eos_token_id is not None:
        result[str(tokenizer.eos_token_id)] = {"str": "<EOS>", "id": tokenizer.eos_token_id}
    return result


# ============================================================
# TASK 1: Protocol Token Physical Atlas
# ============================================================

def task1_protocol_atlas(model, tokenizer, device, info, model_name: str,
                          protocol_tokens: dict, save_dir: Path) -> dict:
    """Task 1: Expanded protocol token atlas with 160+ prompts."""
    log("  Task 1: Protocol token atlas...")

    all_prompts = []
    prompt_groups = {}
    for gname, prompts in ALL_PROMPT_GROUPS.items():
        prompt_groups[gname] = len(prompts)
        all_prompts.extend(prompts)

    # For each prompt, get top-100 logits and protocol token ranks
    results = []
    protocol_id_list = [v["id"] for v in protocol_tokens.values()]

    for pi, prompt in enumerate(all_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
            logits = out.logits[0, -1].detach().float().cpu()

        # Top-100
        topk_vals, topk_ids = torch.topk(logits, k=min(100, logits.numel()))
        topk_ids = topk_ids.numpy().tolist()
        topk_vals = topk_vals.numpy().tolist()

        # Protocol token ranks
        proto_ranks = {}
        for ptid_str, ptinfo in protocol_tokens.items():
            tid = ptinfo["id"]
            if 0 <= tid < logits.numel():
                rank = int((logits > logits[tid]).sum().item()) + 1
                proto_ranks[ptid_str] = {
                    "str": ptinfo["str"],
                    "rank": rank,
                    "logit": float(logits[tid].item()),
                }

        results.append({
            "prompt": prompt,
            "group": next((g for g, ps in ALL_PROMPT_GROUPS.items() if prompt in ps), "unknown"),
            "topk_ids": topk_ids[:50],
            "topk_logits": topk_vals[:50],
            "protocol_ranks": proto_ranks,
        })

        if (pi + 1) % 40 == 0:
            log(f"    {pi+1}/{len(all_prompts)} prompts")

    # Aggregate: per-protocol-token cross-prompt stats
    token_stats = {}
    for ptid_str, ptinfo in protocol_tokens.items():
        ranks = [r["protocol_ranks"].get(ptid_str, {}).get("rank", 999) for r in results]
        logits_vals = [r["protocol_ranks"].get(ptid_str, {}).get("logit", None) for r in results]
        valid_ranks = [r for r in ranks if r < 999]
        valid_logits = [l for l in logits_vals if l is not None]

        # Per-group stats
        group_stats = {}
        for gname in ALL_PROMPT_GROUPS:
            g_ranks = [r["protocol_ranks"].get(ptid_str, {}).get("rank", 999)
                       for r in results if r["group"] == gname]
            g_valid = [r for r in g_ranks if r < 999]
            if g_valid:
                group_stats[gname] = {
                    "mean_rank": float(np.mean(g_valid)),
                    "in_top10": sum(1 for r in g_valid if r <= 10),
                    "in_top50": sum(1 for r in g_valid if r <= 50),
                    "n_prompts": len(g_ranks),
                    "n_valid": len(g_valid),
                }

        token_stats[ptid_str] = {
            "str": ptinfo["str"],
            "cross_prompt_count": len(valid_ranks),
            "cross_prompt_ratio": len(valid_ranks) / len(results),
            "mean_rank": float(np.mean(valid_ranks)) if valid_ranks else 999,
            "median_rank": float(np.median(valid_ranks)) if valid_ranks else 999,
            "mean_logit": float(np.mean(valid_logits)) if valid_logits else 0,
            "std_rank": float(np.std(valid_ranks)) if valid_ranks else 0,
            "in_top10_count": sum(1 for r in valid_ranks if r <= 10),
            "in_top50_count": sum(1 for r in valid_ranks if r <= 50),
            "group_stats": group_stats,
        }

    # Sort by cross_prompt_count
    sorted_stats = sorted(token_stats.values(), key=lambda x: (-x["cross_prompt_count"], x["mean_rank"]))

    output = {
        "task": "task1_protocol_atlas",
        "model": model_name,
        "n_prompts": len(all_prompts),
        "prompt_groups": prompt_groups,
        "n_protocol_tokens": len(protocol_tokens),
        "token_stats": sorted_stats,
    }

    save_path = save_dir / "task1_atlas.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print top-15
    log(f"    Top 15 protocol tokens:")
    for t in sorted_stats[:15]:
        log(f"      {t['str']:15s}  cross={t['cross_prompt_count']:3d}/{len(all_prompts)}  "
            f"rank={t['mean_rank']:6.1f}  top10={t['in_top10_count']:3d}")

    return output


# ============================================================
# TASK 2: Last-layer MLP Channel-level Attribution (Regression Probe)
# ============================================================

def _get_mlp_activation_hook(model, info, layer_idx: int):
    """Register hook to capture MLP intermediate activation (after activation function).
    Returns (handle_list, captured_dict).
    Handles both split_gate_up and merged_gate_up architectures.
    """
    layers = get_layers(model)
    layer = layers[layer_idx]
    mlp = layer.mlp
    captured = {}
    handles = []

    mlp_type = info.mlp_type
    inter_size = info.intermediate_size

    if mlp_type == "split_gate_up":
        # qwen3/DS7B: gate_proj + up_proj + down_proj
        # Activation = silu(gate_proj(x)) * up_proj(x)
        # We hook down_proj input to get the activated intermediate
        def down_hook(module, inputs, output):
            # inputs[0] is the activated intermediate: [batch, seq, inter]
            captured["act"] = inputs[0].detach().float().cpu()
        h = mlp.down_proj.register_forward_hook(down_hook)
        handles.append(h)
    else:
        # GLM4: gate_up_proj (merged) + down_proj
        # Hook down_proj input to get the activated intermediate
        def down_hook_glm(module, inputs, output):
            captured["act"] = inputs[0].detach().float().cpu()
        h = mlp.down_proj.register_forward_hook(down_hook_glm)
        handles.append(h)

    return handles, captured


def task2_channel_attribution(model, tokenizer, device, info, model_name: str,
                               protocol_tokens: dict, save_dir: Path,
                               n_prompts: int = 60) -> dict:
    """Task 2: Channel-level attribution using ridge regression probes.
    No weight access needed - uses activation-logit correlation.
    """
    log("  Task 2: Channel-level attribution (regression probe)...")

    all_prompts = []
    for gname, prompts in ALL_PROMPT_GROUPS.items():
        all_prompts.extend(prompts)
    selected = all_prompts[:n_prompts]

    # Last 3 layers for attribution
    n_layers = info.n_layers
    target_layers = [n_layers - 1, n_layers - 2, n_layers - 3]
    log(f"    Target layers: {target_layers}")

    # Collect activations and logits
    layer_activations = {li: [] for li in target_layers}  # [n_prompts, inter]
    all_logits = []  # [n_prompts, vocab]
    protocol_id_list = sorted(set(v["id"] for v in protocol_tokens.values()))

    for pi, prompt in enumerate(selected):
        # Register hooks for target layers
        hooks_data = {}
        hooks_list = []
        for li in target_layers:
            hs, cap = _get_mlp_activation_hook(model, info, li)
            hooks_data[li] = cap
            hooks_list.extend(hs)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
            logits = out.logits[0, -1].detach().float().cpu()

        for h in hooks_list:
            h.remove()

        # Store last-token activation
        for li in target_layers:
            act = hooks_data[li].get("act")
            if act is not None:
                # act shape: [1, seq_len, inter] -> take last token
                last_act = act[0, -1, :].numpy()
                layer_activations[li].append(last_act)

        all_logits.append(logits.numpy())

        if (pi + 1) % 20 == 0:
            log(f"    {pi+1}/{len(selected)} prompts")

    all_logits = np.array(all_logits)  # [n_prompts, vocab]

    # For each target layer, fit ridge regression for each protocol token
    from sklearn.linear_model import Ridge

    layer_results = {}
    for li in target_layers:
        X = np.array(layer_activations[li])  # [n_prompts, inter]
        if X.shape[0] < 5:
            log(f"    Layer {li}: insufficient data ({X.shape[0]} prompts)")
            continue

        token_channels = {}
        for ptid_str, ptinfo in protocol_tokens.items():
            tid = ptinfo["id"]
            if tid >= all_logits.shape[1]:
                continue
            y = all_logits[:, tid]  # [n_prompts]

            # Ridge regression
            ridge = Ridge(alpha=1.0)
            ridge.fit(X, y)
            coefs = ridge.coef_  # [inter]
            score = ridge.score(X, y)

            # Top contributing channels
            top_indices = np.argsort(np.abs(coefs))[-20:][::-1]
            top_channels = [
                {"channel": int(idx), "coef": float(coefs[idx])}
                for idx in top_indices
            ]

            token_channels[ptid_str] = {
                "str": ptinfo["str"],
                "r2": float(score),
                "top_channels": top_channels,
                "mean_coef": float(np.mean(np.abs(coefs))),
            }

        # Find channel families: channels that contribute to multiple protocol tokens
        channel_token_map = defaultdict(list)
        for ptid_str, tc in token_channels.items():
            for ch in tc["top_channels"][:10]:
                channel_token_map[ch["channel"]].append({
                    "token_str": tc["str"],
                    "coef": ch["coef"],
                })

        # Channels supporting 3+ protocol tokens
        shared_channels = {
            str(ch): tokens
            for ch, tokens in channel_token_map.items()
            if len(tokens) >= 3
        }
        shared_channels = dict(sorted(shared_channels.items(),
                                       key=lambda x: -len(x[1])))

        layer_results[f"L{li}"] = {
            "n_prompts": X.shape[0],
            "n_intermediate": X.shape[1],
            "token_attribution": token_channels,
            "shared_channels": {k: v for k, v in list(shared_channels.items())[:30]},
            "n_shared_channels": len(shared_channels),
        }

        log(f"    L{li}: {len(shared_channels)} shared channels (3+ tokens), "
            f"mean R2={np.mean([tc['r2'] for tc in token_channels.values()]):.3f}")

    output = {
        "task": "task2_channel_attribution",
        "model": model_name,
        "method": "ridge_regression_probe",
        "n_prompts": len(selected),
        "target_layers": target_layers,
        "layer_results": layer_results,
    }

    save_path = save_dir / "task2_channels.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    return output


# ============================================================
# TASK 3: Semantic Direction -> Protocol Channel Bridging
# ============================================================

def task3_semantic_to_protocol(model, tokenizer, device, info, model_name: str,
                                save_dir: Path, n_prompts: int = 20) -> dict:
    """Task 3: Inject semantic (color) direction, measure protocol channel change."""
    log("  Task 3: Semantic -> Protocol channel bridging...")

    # Get color directions from W_U
    W_U = get_W_U(model, model_name)  # [vocab, d_model]

    color_dirs = {}
    for cw in COLOR_WORDS:
        ids = tokenizer.encode(cw, add_special_tokens=False)
        if ids:
            tid = ids[0]
            if tid < W_U.shape[0]:
                color_dirs[cw] = W_U[tid].copy()

    # Compute color difference directions
    diff_dirs = {}
    pairs = [("red", "blue"), ("red", "green"), ("yellow", "black"),
             ("white", "black"), ("orange", "purple")]
    for c1, c2 in pairs:
        if c1 in color_dirs and c2 in color_dirs:
            d = color_dirs[c1] - color_dirs[c2]
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                diff_dirs[f"{c1}-{c2}"] = d / norm

    if not diff_dirs:
        log("    No valid color directions found")
        return {"task": "task3", "error": "no color directions"}

    # Last layer for protocol channel measurement
    last_layer = info.n_layers - 1

    # Prompts for injection
    prompts = [
        "The apple is",
        "The sky is",
        "The grass is",
        "The sun is",
        "The night is",
        "The blood is",
        "The snow is",
        "The fire is",
        "The ocean is",
        "The lemon is",
        "The rose is",
        "The coal is",
        "The cloud is",
        "The orange is",
        "The wood is",
        "The metal is",
        "The stone is",
        "The leaf is",
        "The sand is",
        "The wine is",
    ][:n_prompts]

    results = []
    beta = 8.0  # injection strength

    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        embed = model.get_input_embeddings()

        # Baseline
        hs, cap = _get_mlp_activation_hook(model, info, last_layer)
        with torch.no_grad():
            base_out = model(input_ids=input_ids, use_cache=False)
            base_logits = base_out.logits[0, -1].detach().float().cpu()
        base_act = cap.get("act")
        for h in hs:
            h.remove()

        if base_act is None:
            continue
        base_last_act = base_act[0, -1, :].numpy()

        # Inject each color direction
        for dname, direction in diff_dirs.items():
            # Inject at embedding level, last token
            inputs_embeds = embed(input_ids).detach().clone()
            dir_tensor = torch.tensor(direction, dtype=inputs_embeds.dtype, device=device)
            inputs_embeds[0, -1, :] += (beta * dir_tensor).to(inputs_embeds.dtype)

            hs2, cap2 = _get_mlp_activation_hook(model, info, last_layer)
            with torch.no_grad():
                inj_out = model(inputs_embeds=inputs_embeds, use_cache=False)
                inj_logits = inj_out.logits[0, -1].detach().float().cpu()
            inj_act = cap2.get("act")
            for h in hs2:
                h.remove()

            if inj_act is None:
                continue
            inj_last_act = inj_act[0, -1, :].numpy()

            # Activation change
            act_delta = inj_last_act - base_last_act
            act_delta_norm = float(np.linalg.norm(act_delta))

            # Top changed channels
            top_changed = np.argsort(np.abs(act_delta))[-10:][::-1]
            top_channels = [
                {"channel": int(ch), "delta": float(act_delta[ch])}
                for ch in top_changed
            ]

            # Logit change for key tokens
            proto_tokens_to_check = [".", " ", "\n", "the", "The", "is", "a"]
            logit_changes = {}
            for pt in proto_tokens_to_check:
                ids = tokenizer.encode(pt, add_special_tokens=False)
                if ids and ids[0] < inj_logits.numel():
                    logit_changes[pt] = float(inj_logits[ids[0]].item() - base_logits[ids[0]].item())

            # Also check color token logit change
            color_logit_changes = {}
            for cw in COLOR_WORDS:
                ids = tokenizer.encode(cw, add_special_tokens=False)
                if ids and ids[0] < inj_logits.numel():
                    color_logit_changes[cw] = float(inj_logits[ids[0]].item() - base_logits[ids[0]].item())

            results.append({
                "prompt": prompt,
                "direction": dname,
                "beta": beta,
                "act_delta_norm": act_delta_norm,
                "top_changed_channels": top_channels,
                "protocol_logit_changes": logit_changes,
                "color_logit_changes": color_logit_changes,
            })

        if (pi + 1) % 5 == 0:
            log(f"    {pi+1}/{len(prompts)} prompts")

    # Aggregate
    dir_stats = defaultdict(lambda: {
        "act_delta_norms": [], "protocol_changes": defaultdict(list),
        "color_changes": defaultdict(list),
    })
    for r in results:
        d = r["direction"]
        dir_stats[d]["act_delta_norms"].append(r["act_delta_norm"])
        for pt, delta in r["protocol_logit_changes"].items():
            dir_stats[d]["protocol_changes"][pt].append(delta)
        for cw, delta in r["color_logit_changes"].items():
            dir_stats[d]["color_changes"][cw].append(delta)

    summary = {}
    for d, s in dir_stats.items():
        summary[d] = {
            "mean_act_delta_norm": float(np.mean(s["act_delta_norms"])),
            "protocol_logit_changes": {
                pt: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                for pt, vals in s["protocol_changes"].items()
            },
            "color_logit_changes": {
                cw: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                for cw, vals in s["color_changes"].items()
            },
        }

    output = {
        "task": "task3_semantic_to_protocol",
        "model": model_name,
        "method": "embedding_injection",
        "beta": beta,
        "n_prompts": len(prompts),
        "n_directions": len(diff_dirs),
        "last_layer": last_layer,
        "summary": summary,
        "raw_results": results[:20],  # Keep first 20 for inspection
    }

    save_path = save_dir / "task3_bridging.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print key finding
    for d, s in summary.items():
        proto_mean = np.mean([v["mean"] for v in s["protocol_logit_changes"].values()])
        color_mean = np.mean([v["mean"] for v in s["color_logit_changes"].values()])
        log(f"    {d}: act_delta={s['mean_act_delta_norm']:.4f}, "
            f"proto_logit_change={proto_mean:.4f}, color_logit_change={color_mean:.4f}")

    return output


# ============================================================
# TASK 4: Semantic -> Protocol Information Flow Tracking
# ============================================================

def task4_info_flow(model, tokenizer, device, info, model_name: str,
                     save_dir: Path, n_prompts: int = 15) -> dict:
    """Task 4: Track residual stream propagation from semantic to protocol layers."""
    log("  Task 4: Information flow tracking...")

    n_layers = info.n_layers
    # Sample layers across the model
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))

    log(f"    Tracking layers: {sample_layers}")

    # Use color-containing prompts
    prompts = [
        "The apple is red.",
        "The sky is blue.",
        "The grass is green.",
        "The sun is yellow.",
        "The night is black.",
        "The snow is white.",
        "The fire is orange.",
        "The flower is purple.",
        "The apple is",
        "The sky is",
        "The grass is",
        "The sun is",
        "The night is",
        "The snow is",
        "The fire is",
    ][:n_prompts]

    # For each prompt, capture hidden states at all sample layers
    results = []
    layers_list = get_layers(model)

    for pi, prompt in enumerate(prompts):
        captured = {}
        hooks = []

        def make_hook(li):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[li] = output[0][0, -1, :].detach().float().cpu().numpy()
                else:
                    captured[li] = output[0, -1, :].detach().float().cpu().numpy()
            return hook

        for li in sample_layers:
            h = layers_list[li].register_forward_hook(make_hook(li))
            hooks.append(h)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
            final_logits = out.logits[0, -1].detach().float().cpu().numpy()

        for h in hooks:
            h.remove()

        # Compute color direction projection at each layer
        W_U = get_W_U(model, model_name)

        # Get "red" direction
        red_ids = tokenizer.encode("red", add_special_tokens=False)
        blue_ids = tokenizer.encode("blue", add_special_tokens=False)
        if red_ids and blue_ids:
            red_dir = W_U[red_ids[0]].copy()
            blue_dir = W_U[blue_ids[0]].copy()
            color_diff = red_dir - blue_dir
            color_diff_norm = np.linalg.norm(color_diff)
            if color_diff_norm > 1e-10:
                color_diff = color_diff / color_diff_norm

            # Protocol direction (period)
            period_ids = tokenizer.encode(".", add_special_tokens=False)
            space_ids = tokenizer.encode(" ", add_special_tokens=False)

            layer_projections = {}
            for li in sample_layers:
                h_state = captured.get(li)
                if h_state is None:
                    continue

                # Project onto color direction
                color_proj = float(np.dot(h_state, color_diff) / max(np.linalg.norm(h_state), 1e-10))

                # Project onto period direction
                period_proj = 0.0
                if period_ids and period_ids[0] < W_U.shape[0]:
                    pd = W_U[period_ids[0]]
                    period_proj = float(np.dot(h_state, pd) / max(np.linalg.norm(h_state) * np.linalg.norm(pd), 1e-10))

                # Project onto space direction
                space_proj = 0.0
                if space_ids and space_ids[0] < W_U.shape[0]:
                    sd = W_U[space_ids[0]]
                    space_proj = float(np.dot(h_state, sd) / max(np.linalg.norm(h_state) * np.linalg.norm(sd), 1e-10))

                layer_projections[f"L{li}"] = {
                    "color_proj": color_proj,
                    "period_proj": period_proj,
                    "space_proj": space_proj,
                }

            results.append({
                "prompt": prompt,
                "has_color": "red" in prompt or "blue" in prompt or "green" in prompt
                            or "yellow" in prompt or "black" in prompt or "white" in prompt
                            or "orange" in prompt or "purple" in prompt,
                "layer_projections": layer_projections,
            })

        if (pi + 1) % 5 == 0:
            log(f"    {pi+1}/{len(prompts)} prompts")

    # Aggregate: mean projection per layer, split by has_color
    layer_flow = {}
    for li in sample_layers:
        key = f"L{li}"
        color_vals = []
        nocolor_vals = []
        period_vals = []
        space_vals = []

        for r in results:
            lp = r["layer_projections"].get(key)
            if lp:
                if r["has_color"]:
                    color_vals.append(lp["color_proj"])
                else:
                    nocolor_vals.append(lp["color_proj"])
                period_vals.append(lp["period_proj"])
                space_vals.append(lp["space_proj"])

        layer_flow[key] = {
            "color_proj_mean": float(np.mean(color_vals)) if color_vals else 0,
            "color_proj_nocolor": float(np.mean(nocolor_vals)) if nocolor_vals else 0,
            "color_diff": float(np.mean(color_vals) - np.mean(nocolor_vals)) if color_vals and nocolor_vals else 0,
            "period_proj_mean": float(np.mean(period_vals)) if period_vals else 0,
            "space_proj_mean": float(np.mean(space_vals)) if space_vals else 0,
        }

    output = {
        "task": "task4_info_flow",
        "model": model_name,
        "n_prompts": len(prompts),
        "sample_layers": sample_layers,
        "layer_flow": layer_flow,
        "raw_results": results[:10],
    }

    save_path = save_dir / "task4_infoflow.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print flow summary
    log(f"    Layer flow (color_diff = color - nocolor):")
    for key, val in sorted(layer_flow.items()):
        log(f"      {key}: color_diff={val['color_diff']:.4f}  "
            f"period={val['period_proj_mean']:.4f}  space={val['space_proj_mean']:.4f}")

    return output


# ============================================================
# TASK 5: Protocol-Neutral vs Protocol-Heavy Comparison
# ============================================================

def task5_protocol_neutral(model, tokenizer, device, info, model_name: str,
                            save_dir: Path) -> dict:
    """Task 5: Compare protocol token logits in neutral vs heavy prompts."""
    log("  Task 5: Protocol-neutral vs protocol-heavy comparison...")

    protocol_tokens = get_protocol_token_ids(tokenizer)
    key_tokens = [".", " ", "\n", "the", "The", "is", "a", "Answer", "Solution", "Step"]

    results_neutral = []
    results_heavy = []

    for pi, prompt in enumerate(PROTOCOL_NEUTRAL_PROMPTS):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
            logits = out.logits[0, -1].detach().float().cpu().numpy()

        token_logits = {}
        for pt in key_tokens:
            ids = tokenizer.encode(pt, add_special_tokens=False)
            if ids and ids[0] < len(logits):
                token_logits[pt] = float(logits[ids[0]])

        # Top-5 tokens
        top5 = np.argsort(logits)[-5:][::-1]
        top5_tokens = [(safe_decode(tokenizer, int(t)), float(logits[t])) for t in top5]

        # EOS
        eos_logit = float(logits[tokenizer.eos_token_id]) if tokenizer.eos_token_id and tokenizer.eos_token_id < len(logits) else None

        results_neutral.append({
            "prompt": prompt,
            "token_logits": token_logits,
            "top5": top5_tokens,
            "eos_logit": eos_logit,
        })

    for pi, prompt in enumerate(PROTOCOL_HEAVY_PROMPTS):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
            logits = out.logits[0, -1].detach().float().cpu().numpy()

        token_logits = {}
        for pt in key_tokens:
            ids = tokenizer.encode(pt, add_special_tokens=False)
            if ids and ids[0] < len(logits):
                token_logits[pt] = float(logits[ids[0]])

        top5 = np.argsort(logits)[-5:][::-1]
        top5_tokens = [(safe_decode(tokenizer, int(t)), float(logits[t])) for t in top5]

        eos_logit = float(logits[tokenizer.eos_token_id]) if tokenizer.eos_token_id and tokenizer.eos_token_id < len(logits) else None

        results_heavy.append({
            "prompt": prompt,
            "token_logits": token_logits,
            "top5": top5_tokens,
            "eos_logit": eos_logit,
        })

    # Aggregate
    def aggregate(results, label):
        agg = {"label": label, "n": len(results)}
        for pt in key_tokens:
            vals = [r["token_logits"].get(pt, 0) for r in results]
            agg[f"{pt}_mean"] = float(np.mean(vals))
            agg[f"{pt}_std"] = float(np.std(vals))

        eos_vals = [r["eos_logit"] for r in results if r["eos_logit"] is not None]
        agg["eos_mean"] = float(np.mean(eos_vals)) if eos_vals else 0
        agg["eos_std"] = float(np.std(eos_vals)) if eos_vals else 0

        # Top-5 token frequency
        top_token_freq = Counter()
        for r in results:
            for t, _ in r["top5"]:
                top_token_freq[t] += 1
        agg["top5_freq"] = dict(top_token_freq.most_common(10))

        return agg

    neutral_agg = aggregate(results_neutral, "neutral")
    heavy_agg = aggregate(results_heavy, "heavy")

    # Compute differences
    diff = {"label": "heavy_minus_neutral"}
    for pt in key_tokens:
        diff[f"{pt}_diff"] = heavy_agg[f"{pt}_mean"] - neutral_agg[f"{pt}_mean"]
    diff["eos_diff"] = heavy_agg["eos_mean"] - neutral_agg["eos_mean"]

    output = {
        "task": "task5_protocol_neutral",
        "model": model_name,
        "n_neutral": len(results_neutral),
        "n_heavy": len(results_heavy),
        "key_tokens": key_tokens,
        "neutral_summary": neutral_agg,
        "heavy_summary": heavy_agg,
        "difference": diff,
        "neutral_raw": results_neutral[:10],
        "heavy_raw": results_heavy[:10],
    }

    save_path = save_dir / "task5_neutral.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print comparison
    log(f"    Token logit comparison (heavy - neutral):")
    for pt in key_tokens:
        d = diff.get(f"{pt}_diff", 0)
        log(f"      {pt:12s}: {d:+.4f}")
    log(f"      {'EOS':12s}: {diff.get('eos_diff', 0):+.4f}")

    return output


# ============================================================
# MAIN
# ============================================================

def run_model(model_name: str, args: argparse.Namespace) -> None:
    """Run all 5 tasks for a single model."""
    log(f"\n{'='*60}")
    log(f"Phase 951: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    # Load model
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  Model: {info.model_class}, {info.n_layers}L, d={info.d_model}, "
        f"inter={info.intermediate_size}, mlp={info.mlp_type}")

    # Get protocol tokens
    protocol_tokens = get_protocol_token_ids(tokenizer)
    log(f"  Protocol tokens: {len(protocol_tokens)}")

    t_start = time.time()

    # Task 1: Protocol atlas
    try:
        task1_protocol_atlas(model, tokenizer, device, info, model_name, protocol_tokens, model_dir)
    except Exception as e:
        log(f"  Task 1 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Task 2: Channel attribution
    try:
        task2_channel_attribution(model, tokenizer, device, info, model_name,
                                   protocol_tokens, model_dir, n_prompts=args.task2_prompts)
    except Exception as e:
        log(f"  Task 2 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Task 3: Semantic -> Protocol bridging
    try:
        task3_semantic_to_protocol(model, tokenizer, device, info, model_name,
                                    model_dir, n_prompts=args.task3_prompts)
    except Exception as e:
        log(f"  Task 3 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Task 4: Information flow
    try:
        task4_info_flow(model, tokenizer, device, info, model_name, model_dir,
                        n_prompts=args.task4_prompts)
    except Exception as e:
        log(f"  Task 4 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Task 5: Protocol neutral
    try:
        task5_protocol_neutral(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 5 FAILED: {e}")
        import traceback; traceback.print_exc()

    elapsed = time.time() - t_start
    log(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Release
    release_model(model)
    log(f"  {model_name} complete")


def main():
    parser = argparse.ArgumentParser(description="Phase 951: Protocol Field Atlas")
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b", "all"])
    parser.add_argument("--task2_prompts", type=int, default=60)
    parser.add_argument("--task3_prompts", type=int, default=15)
    parser.add_argument("--task4_prompts", type=int, default=12)
    args = parser.parse_args()

    ensure_dir(RESULT_DIR)
    log(f"Phase {PHASE} started, model={args.model}")

    models = [args.model] if args.model != "all" else ["qwen3", "glm4", "deepseek7b"]
    for m in models:
        run_model(m, args)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
