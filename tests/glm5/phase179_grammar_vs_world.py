"""
Phase 179: ★★★ 从语法到世界 — 解缠语言约束与本体约束机制 ★★★
===================================================================

用户核心洞察 (完全正确):
  1. 最大风险: 把语言约束(linguistic constraint)误认为世界约束(ontological constraint)
  2. 主谓一致是语法约定(grammar convention)，不是世界结构
  3. 真正概念约束必须: cross-linguistic + perceptual-grounded + action-relevant
  4. 当前"约束"是decoder-relative，不是gauge-invariant

核心假设:
  H1 (统一机制): 语法约束和世界约束由同一机制处理
  H2 (专门化机制): 语法约束和世界约束由不同机制处理

关键实验:
  Phase A: 世界扎根约束信号 — 物理可供性、生命性、因果性
  Phase B: 语法约束信号 — 数一致 (Phase 178复现, 用于对比)
  Phase C: 语法 vs 世界层剖面对比 — 是否共享同一传播动力学?
  Phase D: 因果层重要性对比 — 哪些层对语法/世界约束分别关键?
  Phase E: 跨类别约束传播 — 约束类型间的传播R²对比

★★★ 内存优化 (继承Phase 178) ★★★
  只计算2个token的logit差值, 不计算全词表

Usage: python tests/glm5/phase179_grammar_vs_world.py <model_name>
  model_name: qwen3, glm4, deepseek7b
"""

import sys
import os
import time
import json
import gc
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'glm5'))

from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS


# =====================================================================
# MODEL LOADING (BF16 + device_map="auto")
# =====================================================================

def load_model_bf16(model_name):
    """BF16 + device_map=auto loading for all models"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[bf16] Loading {model_name} (bfloat16 + device_map=auto)...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
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

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[bf16] {model_name} loaded: GPU={gpu_mem:.2f}GB", flush=True)

    return model, tokenizer, device


# =====================================================================
# ★★★ 世界扎根约束句子 — 跨3个本体类别 ★★★
# =====================================================================

# 格式: (sentence, expected_token, unexpected_token, constraint_type, category)
# expected_token: 世界知识预期的token
# unexpected_token: 语法上可能但违反世界知识的token
# 在句子中, 我们测量最后一个token之前的position的margin

# ★ Category 1: Physical Affordance (物理可供性)
# 物理属性约束什么可以发生
WORLD_PHYSICAL = [
    # (sentence_template, expected, unexpected)
    # Glass breaks when dropped, doesn't float
    ("The glass fell and it", " broke", " floated"),
    ("The glass dropped and it", " shattered", " bounced"),
    ("The mirror fell and it", " broke", " floated"),
    # Feather floats, doesn't shatter
    ("The feather fell and it", " floated", " shattered"),
    ("The leaf drifted and it", " floated", " shattered"),
    # Rock sinks because heavy
    ("The rock sank because it was", " heavy", " light"),
    ("The stone sank in the water because it was", " heavy", " light"),
    # Wood floats because light
    ("The wood floated because it was", " light", " heavy"),
    ("The cork floated on the water because it was", " light", " heavy"),
    # Ice melts in heat
    ("The ice melted in the", " sun", " freezer"),
    ("The snow melted because of the", " heat", " cold"),
    # Metal conducts heat
    ("The metal pan was hot because metal", " conducts", " insulates"),
]

# ★ Category 2: Animacy / Biological (生命性/生物性)
# 有生命vs无生命实体可执行不同动作
WORLD_ANIMACY = [
    # Dogs run, don't grow across fields
    ("The dog", " ran", " grew"),
    ("The cat", " walked", " sprouted"),
    ("The horse", " galloped", " bloomed"),
    # Trees grow, don't run
    ("The tree", " grew", " ran"),
    ("The flower", " bloomed", " walked"),
    ("The grass", " grew", " jumped"),
    # Fish swim, don't fly
    ("The fish", " swam", " flew"),
    ("The whale", " swam", " flew"),
    # Birds fly, don't swim (usually)
    ("The bird", " flew", " swam"),
    ("The eagle", " soared", " dived"),
    # People think
    ("The woman", " thought", " rusted"),
    ("The man", " spoke", " melted"),
    # Rivers flow
    ("The river", " flowed", " walked"),
    ("The wind", " blew", " slept"),
]

# ★ Category 3: Causality / World Knowledge (因果性/世界知识)
# 物理原因产生特定结果
WORLD_CAUSALITY = [
    # Fire burns, doesn't freeze
    ("The fire", " burned", " froze"),
    ("The flame", " heated", " cooled"),
    # Rain wets, doesn't dry
    ("The rain", " wet", " dried"),
    ("The storm", " soaked", " dried"),
    # Sun melts ice
    ("The sun", " melted", " froze"),
    # Cold freezes water
    ("The cold", " froze", " melted"),
    ("The winter", " chilled", " warmed"),
    # Knives cut, don't heal
    ("The knife", " cut", " healed"),
    ("The scissors", " cut", " healed"),
    # Cars drive, don't swim
    ("The car", " drove", " swam"),
    ("The train", " rolled", " flew"),
    # Boats float, don't sink (normally)
    ("The boat", " floated", " sank"),
    ("The ship", " sailed", " crawled"),
]

# ★ Category 4: Object Function / Tool Use (对象功能/工具使用)
WORLD_FUNCTION = [
    # Keys open doors
    ("The key", " opened", " ate"),
    ("The lock", " secured", " drank"),
    # Pens write
    ("The pen", " wrote", " ate"),
    ("The pencil", " drew", " drank"),
    # Chairs are for sitting
    ("The chair", " supported", " ate"),
    # Beds are for sleeping
    ("The bed", " rested", " ran"),
    # Stoves cook
    ("The stove", " cooked", " froze"),
    ("The oven", " baked", " chilled"),
    # Phones communicate
    ("The phone", " rang", " grew"),
    # Bridges connect
    ("The bridge", " connected", " swam"),
]


# ★★★ 语法约束句子 (Phase 178复现) ★★★
VERB_PAIRS = [
    ("sleeps", "sleep"), ("runs", "run"), ("walks", "walk"),
    ("flies", "fly"), ("swims", "swim"), ("jumps", "jump"),
    ("sings", "sing"), ("reads", "read"), ("writes", "write"),
    ("eats", "eat"), ("drinks", "drink"), ("thinks", "think"),
    ("knows", "know"), ("grows", "grow"), ("moves", "move"),
    ("works", "work"), ("plays", "play"), ("talks", "talk"),
    ("sits", "sit"), ("stands", "stand"), ("falls", "fall"),
    ("rises", "rise"), ("shines", "shine"), ("drifts", "drift"),
    ("breaks", "break"), ("drives", "drive"), ("hides", "hide"),
    ("cries", "cry"), ("smiles", "smile"), ("waits", "wait"),
]

NOUN_PAIRS = [
    ("cat", "cats"), ("dog", "dogs"), ("bird", "birds"),
    ("child", "children"), ("man", "men"), ("woman", "women"),
    ("horse", "horses"), ("student", "students"), ("teacher", "teachers"),
    ("flower", "flowers"), ("river", "rivers"), ("cloud", "clouds"),
    ("lamp", "lamps"), ("clock", "clocks"), ("bell", "bells"),
    ("door", "doors"), ("tree", "trees"), ("book", "books"),
    ("car", "cars"), ("star", "stars"),
]


# =====================================================================
# ★★★ 核心工具函数 (继承Phase 178) ★★★
# =====================================================================

def get_final_norm_weight(model):
    """Get the final layer norm weight from the model"""
    norm = None
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        norm = model.model.norm
    elif hasattr(model, 'model') and hasattr(model.model, 'final_layernorm'):
        norm = model.model.final_layernorm
    
    if norm is None:
        print("  [WARN] Could not find final norm, using identity", flush=True)
        return None
    
    w = norm.weight
    if not w.is_meta:
        return w.detach().cpu().float().numpy()
    
    print("  [INFO] Final norm on meta device, loading from safetensors...", flush=True)
    try:
        import glob
        from safetensors import safe_open
        model_path = None
        if hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
            model_path = model.config._name_or_path
        if model_path is None:
            print("  [WARN] Cannot find model path, using identity norm", flush=True)
            return None
        
        sf_files = glob.glob(os.path.join(model_path, '*.safetensors'))
        for name in ['model.norm.weight', 'model.final_layernorm.weight']:
            for sf_file in sf_files:
                with safe_open(sf_file, framework='pt', device='cpu') as sf:
                    if name in sf.keys():
                        w = sf.get_tensor(name)
                        print(f"  [INFO] Loaded norm from safetensors, key={name}", flush=True)
                        return w.float().numpy()
        
        print("  [WARN] Could not find norm weight in safetensors, using identity", flush=True)
        return None
    except Exception as e:
        print(f"  [WARN] Failed to load norm from safetensors: {e}, using identity", flush=True)
        return None


def compute_margin_from_h(h, w_row_a, w_row_b, norm_weight=None, eps=1e-6):
    """
    直接计算2个token的logit差值, 无需全词表
    margin = (W_U[a] - W_U[b]) · h_normed
    """
    h_float = h.astype(np.float32)
    if norm_weight is not None:
        rms = np.sqrt(np.mean(h_float.astype(np.float64) ** 2) + eps)
        h_normed = (h_float / np.float32(rms)) * norm_weight.astype(np.float32)
    else:
        h_normed = h_float
    
    diff = w_row_a.astype(np.float32) - w_row_b.astype(np.float32)
    margin = float(np.dot(diff, h_normed))
    
    if not np.isfinite(margin):
        return 0.0
    return margin


def get_hidden_states(model, tokenizer, sentence, positions, n_layers):
    """
    前向传播, 返回指定位置的hidden states (float32)
    """
    input_device = next(model.parameters()).device
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attn_mask = inputs["attention_mask"].to(input_device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
    
    hs = out.hidden_states
    
    result = {}
    for pos_name, pos_idx in positions.items():
        if pos_idx >= hs[0].shape[1]:
            continue
        pos_hs = {}
        for li in range(min(n_layers + 1, len(hs))):
            h = hs[li][0, pos_idx].float().cpu().numpy()
            pos_hs[li] = h
        result[pos_name] = pos_hs
    
    return result


def find_last_token_position(tokenizer, sentence):
    """
    找到句子中最后一个token的位置 (用于测量约束信号)
    即: 句子最后一个token的position (不包含padding)
    """
    input_ids = tokenizer.encode(sentence)
    return len(input_ids) - 1


def find_subject_position(tokenizer, sentence):
    """找到subject token的position"""
    full_ids = tokenizer.encode(sentence)
    no_special_ids = tokenizer.encode(sentence, add_special_tokens=False)
    
    if len(full_ids) == len(no_special_ids):
        return 1  # Qwen3: no special tokens, "The" at 0, subject at 1
    else:
        n_special = len(full_ids) - len(no_special_ids)
        return n_special + 1


# =====================================================================
# Phase A: ★★★ 世界扎根约束信号 ★★★
# =====================================================================

def run_world_constraints(model, tokenizer, device, model_info, W_U, norm_weight):
    """
    ★★★ 核心实验: 世界约束是否在Transformer内部产生信号? ★★★
    
    对于每个世界约束句子:
    - 在最后一个token位置测量 margin = logit(expected) - logit(unexpected)
    - 如果 margin > 0 → 模型"知道"世界约束
    - 如果 margin < 0 → 模型违反世界约束
    
    同时在subject/object位置测量, 看约束信号如何传播
    """
    n_layers = model_info.n_layers
    
    print("\n" + "="*70, flush=True)
    print("Phase A: ★★★ 世界扎根约束信号 ★★★", flush=True)
    print("="*70, flush=True)
    
    # ★★★ Pre-extract W_U rows for world constraint tokens ★★★
    def get_wu_rows(tokenizer, W_U, token_a, token_b):
        """Extract W_U rows for a token pair. Returns (w_a, w_b) or None."""
        a_ids = tokenizer.encode(token_a, add_special_tokens=False)
        b_ids = tokenizer.encode(token_b, add_special_tokens=False)
        if len(a_ids) == 1 and len(b_ids) == 1:
            return W_U[a_ids[0]].astype(np.float32), W_U[b_ids[0]].astype(np.float32)
        return None
    
    # Collect all world constraint test items
    all_world_items = []
    
    for sent, exp_tok, unexp_tok in WORLD_PHYSICAL:
        rows = get_wu_rows(tokenizer, W_U, exp_tok, unexp_tok)
        if rows is not None:
            all_world_items.append({
                "sentence": sent, "expected": exp_tok, "unexpected": unexp_tok,
                "category": "physical", "w_exp": rows[0], "w_unexp": rows[1],
            })
    
    for sent, exp_tok, unexp_tok in WORLD_ANIMACY:
        rows = get_wu_rows(tokenizer, W_U, exp_tok, unexp_tok)
        if rows is not None:
            all_world_items.append({
                "sentence": sent, "expected": exp_tok, "unexpected": unexp_tok,
                "category": "animacy", "w_exp": rows[0], "w_unexp": rows[1],
            })
    
    for sent, exp_tok, unexp_tok in WORLD_CAUSALITY:
        rows = get_wu_rows(tokenizer, W_U, exp_tok, unexp_tok)
        if rows is not None:
            all_world_items.append({
                "sentence": sent, "expected": exp_tok, "unexpected": unexp_tok,
                "category": "causality", "w_exp": rows[0], "w_unexp": rows[1],
            })
    
    for sent, exp_tok, unexp_tok in WORLD_FUNCTION:
        rows = get_wu_rows(tokenizer, W_U, exp_tok, unexp_tok)
        if rows is not None:
            all_world_items.append({
                "sentence": sent, "expected": exp_tok, "unexpected": unexp_tok,
                "category": "function", "w_exp": rows[0], "w_unexp": rows[1],
            })
    
    print(f"  Valid world constraint items: {len(all_world_items)}", flush=True)
    
    # Count by category
    cat_counts = defaultdict(int)
    for item in all_world_items:
        cat_counts[item["category"]] += 1
    for cat, count in sorted(cat_counts.items()):
        print(f"    {cat}: {count} items", flush=True)
    
    # ★★★ Run each item and compute layer-wise margins ★★★
    world_signals = []
    
    for idx, item in enumerate(all_world_items):
        sent = item["sentence"]
        
        # Measure at the LAST TOKEN position (where the model predicts the next token)
        # For "The glass fell and it", the last token is " it", and the model predicts what comes next
        last_pos = find_last_token_position(tokenizer, sent)
        
        # Also measure at the subject/object position
        subj_pos = find_subject_position(tokenizer, sent)
        
        positions = {"last_token": last_pos, "subject": subj_pos}
        
        try:
            pos_hs = get_hidden_states(model, tokenizer, sent, positions, n_layers)
        except Exception as e:
            continue
        
        # Compute margin at each (position, layer)
        pos_margins = {}
        for pos_name in ["last_token", "subject"]:
            if pos_name not in pos_hs:
                continue
            layer_margins = {}
            for li, h in pos_hs[pos_name].items():
                margin = compute_margin_from_h(h, item["w_exp"], item["w_unexp"], norm_weight)
                layer_margins[li] = margin
            pos_margins[pos_name] = layer_margins
        
        world_signals.append({
            "category": item["category"],
            "sentence": sent,
            "expected": item["expected"],
            "unexpected": item["unexpected"],
            "margins": pos_margins,
        })
        
        if (idx + 1) % 10 == 0:
            print(f"    Processed {idx+1}/{len(all_world_items)} world items", flush=True)
    
    print(f"  Successfully processed: {len(world_signals)} world items", flush=True)
    
    # ====== Analysis ======
    
    # 1. Average margin by category at last_token position, key layers
    cat_margins = defaultdict(lambda: defaultdict(list))
    for sig in world_signals:
        cat = sig["category"]
        if "last_token" in sig["margins"]:
            for li, m in sig["margins"]["last_token"].items():
                cat_margins[cat][li].append(m)
    
    avg_cat_margins = {}
    for cat, layer_data in cat_margins.items():
        avg_cat_margins[cat] = {str(li): round(float(np.mean(vals)), 4) 
                                for li, vals in layer_data.items()}
    
    # 2. Overall world constraint signal (all categories combined)
    all_world_at_last = defaultdict(list)
    for sig in world_signals:
        if "last_token" in sig["margins"]:
            for li, m in sig["margins"]["last_token"].items():
                all_world_at_last[li].append(m)
    
    avg_world_margin = {li: round(float(np.mean(vals)), 4) 
                        for li, vals in all_world_at_last.items()}
    
    # 3. World constraint satisfaction rate: fraction of items where margin > 0 at last layer
    world_satisfaction = {}
    for li in range(n_layers + 1):
        n_satisfied = sum(1 for sig in world_signals 
                         if "last_token" in sig["margins"] 
                         and sig["margins"]["last_token"].get(li, 0) > 0)
        n_total = sum(1 for sig in world_signals 
                     if "last_token" in sig["margins"] and li in sig["margins"]["last_token"])
        if n_total > 0:
            world_satisfaction[li] = round(n_satisfied / n_total, 4)
    
    # 4. Propagation dynamics: R² for world constraints
    r_squared_values = []
    for sig in world_signals:
        if "last_token" not in sig["margins"]:
            continue
        margins = sig["margins"]["last_token"]
        layers_sorted = sorted(margins.keys())
        if len(layers_sorted) < 3:
            continue
        X = np.array([margins[layers_sorted[i]] for i in range(len(layers_sorted) - 1)], dtype=np.float64)
        Y = np.array([margins[layers_sorted[i+1]] for i in range(len(layers_sorted) - 1)], dtype=np.float64)
        if np.std(X) < 1e-10 or np.std(Y) < 1e-10:
            continue
        try:
            A, b = np.polyfit(X, Y, 1)
            Y_pred = A * X + b
            ss_res = np.sum((Y - Y_pred) ** 2)
            ss_tot = np.sum((Y - np.mean(Y)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            if np.isfinite(r2):
                r_squared_values.append(max(0, min(1, r2)))
        except:
            continue
    
    avg_r2 = float(np.mean(r_squared_values)) if r_squared_values else 0
    median_r2 = float(np.median(r_squared_values)) if r_squared_values else 0
    
    # Print results
    print(f"\n  ★★★ World Constraint Results ★★★", flush=True)
    print(f"    Overall margin at L0: {avg_world_margin.get(0, 0):.4f}", flush=True)
    print(f"    Overall margin at L_last: {avg_world_margin.get(n_layers, 0):.4f}", flush=True)
    print(f"    Satisfaction rate at L_last: {world_satisfaction.get(n_layers, 0):.2%}", flush=True)
    print(f"    Propagation R² (avg): {avg_r2:.4f}", flush=True)
    
    print(f"\n    By category (margin at L_last):", flush=True)
    for cat in ["physical", "animacy", "causality", "function"]:
        m = avg_cat_margins.get(cat, {}).get(str(n_layers), 0)
        print(f"      {cat}: {m:.4f}", flush=True)
    
    print(f"\n    Satisfaction by category at L_last:", flush=True)
    for cat in ["physical", "animacy", "causality", "function"]:
        cat_sigs = [s for s in world_signals if s["category"] == cat and "last_token" in s["margins"]]
        n_sat = sum(1 for s in cat_sigs if s["margins"]["last_token"].get(n_layers, 0) > 0)
        n_tot = len(cat_sigs)
        rate = n_sat / max(n_tot, 1)
        print(f"      {cat}: {rate:.2%} ({n_sat}/{n_tot})", flush=True)
    
    return {
        "n_items": len(world_signals),
        "avg_margin": {str(li): v for li, v in avg_world_margin.items()},
        "avg_margin_by_category": avg_cat_margins,
        "satisfaction_rate": {str(li): v for li, v in world_satisfaction.items()},
        "propagation_r2_avg": round(avg_r2, 4),
        "propagation_r2_median": round(median_r2, 4),
        "world_signals_sample": [{
            "category": s["category"],
            "sentence": s["sentence"],
            "margin_L0": s["margins"].get("last_token", {}).get(0, 0),
            "margin_Llast": s["margins"].get("last_token", {}).get(n_layers, 0),
        } for s in world_signals[:20]],
    }


# =====================================================================
# Phase B: ★★★ 语法约束信号 (数一致, 用于对比) ★★★
# =====================================================================

def run_grammar_constraints(model, tokenizer, device, model_info, W_U, norm_weight):
    """
    ★★★ 语法约束: 数一致 (Phase 178复现) ★★★
    
    在subject位置测量 margin = logit(sg_verb) - logit(pl_verb)
    sg_subj句子 → margin应该>0
    pl_subj句子 → margin应该<0
    """
    n_layers = model_info.n_layers
    
    print("\n" + "="*70, flush=True)
    print("Phase B: ★★★ 语法约束信号 (数一致) ★★★", flush=True)
    print("="*70, flush=True)
    
    # Pre-extract W_U rows for verb pairs
    verb_pair_rows = {}
    for verb_sg, verb_pl in VERB_PAIRS:
        sg_ids = tokenizer.encode(" " + verb_sg, add_special_tokens=False)
        pl_ids = tokenizer.encode(" " + verb_pl, add_special_tokens=False)
        if len(sg_ids) == 1 and len(pl_ids) == 1:
            verb_pair_rows[(verb_sg, verb_pl)] = {
                "w_sg": W_U[sg_ids[0]].astype(np.float32),
                "w_pl": W_U[pl_ids[0]].astype(np.float32),
            }
    
    print(f"  Valid verb pairs: {len(verb_pair_rows)}/{len(VERB_PAIRS)}", flush=True)
    
    # Generate sentence pairs
    sentences = []
    for i, (noun_sg, noun_pl) in enumerate(NOUN_PAIRS):
        verb_sg, verb_pl = VERB_PAIRS[i % len(VERB_PAIRS)]
        sentences.append((f"The {noun_sg} {verb_sg}", verb_sg, verb_pl, +1, "sg_subj"))
        sentences.append((f"The {noun_pl} {verb_pl}", verb_sg, verb_pl, -1, "pl_subj"))
    
    print(f"  Total grammar sentences: {len(sentences)}", flush=True)
    
    grammar_signals = []
    
    for idx, (sent, v_sg, v_pl, exp_sign, stype) in enumerate(sentences):
        vkey = (v_sg, v_pl)
        if vkey not in verb_pair_rows:
            continue
        rows = verb_pair_rows[vkey]
        
        # Measure at subject position AND last token position
        subj_pos = find_subject_position(tokenizer, sent)
        last_pos = find_last_token_position(tokenizer, sent)
        positions = {"subject": subj_pos, "last_token": last_pos}
        
        try:
            pos_hs = get_hidden_states(model, tokenizer, sent, positions, n_layers)
        except:
            continue
        
        pos_margins = {}
        for pos_name in ["subject", "last_token"]:
            if pos_name not in pos_hs:
                continue
            layer_margins = {}
            for li, h in pos_hs[pos_name].items():
                margin = compute_margin_from_h(h, rows["w_sg"], rows["w_pl"], norm_weight)
                layer_margins[li] = margin
            pos_margins[pos_name] = layer_margins
        
        grammar_signals.append({
            "type": stype,
            "expected_sign": exp_sign,
            "margins": pos_margins,
        })
        
        if (idx + 1) % 20 == 0:
            print(f"    Processed {idx+1}/{len(sentences)} grammar sentences", flush=True)
    
    print(f"  Successfully processed: {len(grammar_signals)} grammar signals", flush=True)
    
    # ====== Analysis ======
    # ★★★ 关键: 语法约束在subject位置测量 (pre-verb, 非verb本身!) ★★★
    # 原因: "The cat sleeps" 的last_token是"sleeps"本身, 测量那里是trivially true
    # 正确位置: subject位置 (pre-verb), 模型需要编码number信息来影响动词选择
    
    GRAMMAR_POS = "subject"  # Use subject position for grammar analysis
    
    # Average margin by type at the correct position
    type_margins = defaultdict(lambda: defaultdict(list))
    for sig in grammar_signals:
        if GRAMMAR_POS in sig["margins"]:
            for li, m in sig["margins"][GRAMMAR_POS].items():
                type_margins[sig["type"]][li].append(m)
    
    avg_type_margins = {}
    for stype, layer_data in type_margins.items():
        avg_type_margins[stype] = {str(li): round(float(np.mean(vals)), 4) 
                                    for li, vals in layer_data.items()}
    
    # Violation signal at the correct position
    violation_signal = {}
    for li in range(n_layers + 1):
        sg_m = avg_type_margins.get("sg_subj", {}).get(str(li), 0)
        pl_m = avg_type_margins.get("pl_subj", {}).get(str(li), 0)
        violation_signal[li] = sg_m - pl_m
    
    # Grammar satisfaction rate at the correct position
    grammar_satisfaction = {}
    for li in range(n_layers + 1):
        n_satisfied = 0
        n_total = 0
        for sig in grammar_signals:
            if GRAMMAR_POS not in sig["margins"]:
                continue
            if li not in sig["margins"][GRAMMAR_POS]:
                continue
            n_total += 1
            m = sig["margins"][GRAMMAR_POS][li]
            if sig["expected_sign"] * m > 0:
                n_satisfied += 1
        if n_total > 0:
            grammar_satisfaction[li] = round(n_satisfied / n_total, 4)
    
    # Propagation R² at the correct position
    r_squared_values = []
    for sig in grammar_signals:
        if GRAMMAR_POS not in sig["margins"]:
            continue
        margins = sig["margins"][GRAMMAR_POS]
        layers_sorted = sorted(margins.keys())
        if len(layers_sorted) < 3:
            continue
        X = np.array([margins[layers_sorted[i]] for i in range(len(layers_sorted) - 1)], dtype=np.float64)
        Y = np.array([margins[layers_sorted[i+1]] for i in range(len(layers_sorted) - 1)], dtype=np.float64)
        if np.std(X) < 1e-10 or np.std(Y) < 1e-10:
            continue
        try:
            A, b = np.polyfit(X, Y, 1)
            Y_pred = A * X + b
            ss_res = np.sum((Y - Y_pred) ** 2)
            ss_tot = np.sum((Y - np.mean(Y)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            if np.isfinite(r2):
                r_squared_values.append(max(0, min(1, r2)))
        except:
            continue
    
    avg_r2 = float(np.mean(r_squared_values)) if r_squared_values else 0
    median_r2 = float(np.median(r_squared_values)) if r_squared_values else 0
    
    print(f"\n  ★★★ Grammar Constraint Results (measured at {GRAMMAR_POS} position) ★★★", flush=True)
    print(f"    Violation at L0: {violation_signal.get(0, 0):.4f}", flush=True)
    print(f"    Violation at L_last: {violation_signal.get(n_layers, 0):.4f}", flush=True)
    print(f"    Satisfaction rate at L_last: {grammar_satisfaction.get(n_layers, 0):.2%}", flush=True)
    print(f"    Propagation R² (avg): {avg_r2:.4f}", flush=True)
    
    return {
        "n_signals": len(grammar_signals),
        "avg_margins_by_type": avg_type_margins,
        "violation_signal": {str(li): round(v, 4) for li, v in violation_signal.items()},
        "satisfaction_rate": {str(li): v for li, v in grammar_satisfaction.items()},
        "propagation_r2_avg": round(avg_r2, 4),
        "propagation_r2_median": round(median_r2, 4),
    }


# =====================================================================
# Phase C: ★★★ 语法 vs 世界层剖面对比 ★★★
# =====================================================================

def run_grammar_vs_world_profile(world_results, grammar_results, model_info):
    """
    ★★★ 核心对比: 语法约束和世界约束的层剖面是否相同? ★★★
    
    如果相同 → H1: 统一机制
    如果不同 → H2: 专门化机制
    """
    n_layers = model_info.n_layers
    
    print("\n" + "="*70, flush=True)
    print("Phase C: ★★★ 语法 vs 世界层剖面对比 ★★★", flush=True)
    print("="*70, flush=True)
    
    # Get layer-wise margins for both
    world_margins = world_results["avg_margin"]
    grammar_violation = grammar_results["violation_signal"]
    
    # Normalize both signals to [0, 1] for comparison
    def normalize_signal(signal_dict):
        vals = list(signal_dict.values())
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < 1e-10:
            return {k: 0.5 for k in signal_dict}
        return {k: (signal_dict[k] - vmin) / (vmax - vmin) for k in signal_dict}
    
    world_norm = normalize_signal(world_margins)
    grammar_norm = normalize_signal(grammar_violation)
    
    # Compute correlation between normalized profiles
    common_layers = sorted(set(world_norm.keys()) & set(grammar_norm.keys()))
    
    if len(common_layers) < 3:
        print("  [WARN] Not enough common layers for profile comparison", flush=True)
        return {"profile_correlation": 0, "verdict": "insufficient_data"}
    
    world_vec = np.array([world_norm[li] for li in common_layers], dtype=np.float64)
    grammar_vec = np.array([grammar_norm[li] for li in common_layers], dtype=np.float64)
    
    # Pearson correlation
    if np.std(world_vec) > 1e-10 and np.std(grammar_vec) > 1e-10:
        profile_corr = float(np.corrcoef(world_vec, grammar_vec)[0, 1])
    else:
        profile_corr = 0.0
    
    # Cosine similarity
    if np.linalg.norm(world_vec) > 1e-10 and np.linalg.norm(grammar_vec) > 1e-10:
        profile_cos = float(np.dot(world_vec, grammar_vec) / (np.linalg.norm(world_vec) * np.linalg.norm(grammar_vec)))
    else:
        profile_cos = 0.0
    
    # Peak layer comparison: which layer has the strongest signal?
    world_peak = max(world_margins.items(), key=lambda x: abs(x[1]))[0] if world_margins else None
    grammar_peak = max(grammar_violation.items(), key=lambda x: abs(x[1]))[0] if grammar_violation else None
    
    # Onset layer: first layer where signal exceeds threshold
    world_onset = None
    grammar_onset = None
    threshold = 0.5
    
    for li in sorted(world_margins.keys(), key=int):
        if abs(world_margins[li]) > threshold:
            world_onset = li
            break
    
    for li in sorted(grammar_violation.keys(), key=int):
        if abs(grammar_violation[li]) > threshold:
            grammar_onset = li
            break
    
    # ★★★ Key comparison: satisfaction rates ★★★
    world_sat = world_results["satisfaction_rate"]
    grammar_sat = grammar_results["satisfaction_rate"]
    
    world_sat_last = world_sat.get(str(n_layers), 0)
    grammar_sat_last = grammar_sat.get(str(n_layers), 0)
    
    # ★★★ Verdict ★★★
    if abs(profile_corr) > 0.8:
        verdict = "H1_LIKELY: Unified mechanism — grammar and world constraints share the same propagation dynamics"
    elif abs(profile_corr) < 0.3:
        verdict = "H2_LIKELY: Specialized mechanisms — grammar and world constraints use different processing"
    else:
        verdict = "INCONCLUSIVE: Partial overlap — some shared, some specialized mechanisms"
    
    print(f"\n  ★★★ Grammar vs World Profile Comparison ★★★", flush=True)
    print(f"    Profile correlation (Pearson): {profile_corr:.4f}", flush=True)
    print(f"    Profile cosine similarity: {profile_cos:.4f}", flush=True)
    print(f"    World constraint onset: L{world_onset}", flush=True)
    print(f"    Grammar constraint onset: L{grammar_onset}", flush=True)
    print(f"    World peak layer: L{world_peak}", flush=True)
    print(f"    Grammar peak layer: L{grammar_peak}", flush=True)
    print(f"    World satisfaction at L_last: {world_sat_last:.2%}", flush=True)
    print(f"    Grammar satisfaction at L_last: {grammar_sat_last:.2%}", flush=True)
    print(f"\n    ★ VERDICT: {verdict}", flush=True)
    
    return {
        "profile_correlation": round(profile_corr, 4),
        "profile_cosine": round(profile_cos, 4),
        "world_onset_layer": world_onset,
        "grammar_onset_layer": grammar_onset,
        "world_peak_layer": world_peak,
        "grammar_peak_layer": grammar_peak,
        "world_satisfaction_last": world_sat_last,
        "grammar_satisfaction_last": grammar_sat_last,
        "verdict": verdict,
    }


# =====================================================================
# Phase D: ★★★ 因果层重要性对比 ★★★
# =====================================================================

def run_causal_layer_importance(model, tokenizer, device, model_info, W_U, norm_weight):
    """
    ★★★ 因果干预: 哪些层对语法/世界约束分别关键? ★★★
    
    方法: Layer-wise noise corruption
    - 在每层注入高斯噪声到residual stream
    - 测量约束信号的变化
    - 变化大的层 = 因果关键的层
    
    分别测试语法和世界约束，对比重要性剖面
    """
    n_layers = model_info.n_layers
    layers = get_layers(model)
    d_model = model_info.d_model
    
    print("\n" + "="*70, flush=True)
    print("Phase D: ★★★ 因果层重要性对比 ★★★", flush=True)
    print("="*70, flush=True)
    
    # ★★★ Selected layers for intervention (every 4th layer) ★★★
    test_layers = list(range(0, n_layers, max(1, n_layers // 9))) + [n_layers - 1]
    test_layers = sorted(set(test_layers))
    print(f"  Intervention layers: {test_layers}", flush=True)
    
    # ---- Test sentences ----
    
    # Grammar test: "The cat sleeps" (sg_subj, sg_verb)
    grammar_sent = "The cat sleeps"
    grammar_verb_sg, grammar_verb_pl = "sleeps", "sleep"
    
    # World test: "The glass fell and it" (expected: broke, unexpected: floated)
    world_sent = "The glass fell and it"
    world_exp, world_unexp = " broke", " floated"
    
    # Get W_U rows
    def get_wu_rows_safe(tok, W_U, t1, t2):
        ids1 = tok.encode(t1, add_special_tokens=False)
        ids2 = tok.encode(t2, add_special_tokens=False)
        if len(ids1) == 1 and len(ids2) == 1:
            return W_U[ids1[0]].astype(np.float32), W_U[ids2[0]].astype(np.float32)
        return None
    
    grammar_rows = get_wu_rows_safe(tokenizer, W_U, " " + grammar_verb_sg, " " + grammar_verb_pl)
    world_rows = get_wu_rows_safe(tokenizer, W_U, world_exp, world_unexp)
    
    if grammar_rows is None:
        print("  [WARN] Cannot get grammar token rows, using alternatives", flush=True)
        # Try alternative verb pairs
        for vsg, vpl in VERB_PAIRS[:10]:
            grammar_rows = get_wu_rows_safe(tokenizer, W_U, " " + vsg, " " + vpl)
            if grammar_rows is not None:
                grammar_verb_sg, grammar_verb_pl = vsg, vpl
                grammar_sent = f"The cat {vsg}"
                break
    
    if world_rows is None:
        print("  [WARN] Cannot get world token rows, using alternatives", flush=True)
        for sent, exp, unexp in WORLD_PHYSICAL[:5]:
            world_rows = get_wu_rows_safe(tokenizer, W_U, exp, unexp)
            if world_rows is not None:
                world_sent = sent
                world_exp, world_unexp = exp, unexp
                break
    
    if grammar_rows is None or world_rows is None:
        print("  [ERROR] Cannot get token rows for causal test, skipping", flush=True)
        return {"error": "no_valid_token_rows"}
    
    print(f"  Grammar sentence: '{grammar_sent}' (margin: {' '+grammar_verb_sg} vs {' '+grammar_verb_pl})", flush=True)
    print(f"  World sentence: '{world_sent}' (margin: {world_exp} vs {world_unexp})", flush=True)
    
    # ---- Baseline: no corruption ----
    # ★★★ 语法约束在subject位置测量, 世界约束在last_token位置测量 ★★★
    def get_baseline_margins(sentence, w_exp, w_unexp, use_subject=False):
        """Get baseline margins at the appropriate position"""
        if use_subject:
            pos = find_subject_position(tokenizer, sentence)
            positions = {"target": pos}
        else:
            pos = find_last_token_position(tokenizer, sentence)
            positions = {"target": pos}
        pos_hs = get_hidden_states(model, tokenizer, sentence, positions, n_layers)
        if "target" not in pos_hs:
            return None
        margins = {}
        for li, h in pos_hs["target"].items():
            margins[li] = compute_margin_from_h(h, w_exp, w_unexp, norm_weight)
        return margins
    
    grammar_baseline = get_baseline_margins(grammar_sent, grammar_rows[0], grammar_rows[1], use_subject=True)
    world_baseline = get_baseline_margins(world_sent, world_rows[0], world_rows[1], use_subject=False)
    
    if grammar_baseline is None or world_baseline is None:
        print("  [ERROR] Cannot compute baseline margins, skipping", flush=True)
        return {"error": "no_baseline"}
    
    print(f"  Grammar baseline margin at L_last: {grammar_baseline.get(n_layers, 0):.4f}", flush=True)
    print(f"  World baseline margin at L_last: {world_baseline.get(n_layers, 0):.4f}", flush=True)
    
    # ---- Corruption: add noise at each test layer ----
    noise_std = 1.0  # Standard deviation of Gaussian noise
    n_trials = 3  # Average over multiple noise samples
    
    grammar_importance = {}
    world_importance = {}
    
    for patch_layer in test_layers:
        grammar_deltas = []
        world_deltas = []
        
        for trial in range(n_trials):
            # Register hook to add noise at the target layer
            noise = torch.randn(1, d_model, dtype=torch.bfloat16, device=device) * noise_std
            
            def make_noise_hook(noise_tensor, target_layer):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        # Add noise to the residual stream output
                        output_list = list(output)
                        output_list[0] = output_list[0] + noise_tensor.to(output_list[0].device)
                        return tuple(output_list)
                    else:
                        return output + noise_tensor.to(output.device)
                return hook
            
            hook = layers[patch_layer].register_forward_hook(make_noise_hook(noise, patch_layer))
            
            # Grammar (measure at subject position)
            try:
                grammar_corrupted = get_baseline_margins(grammar_sent, grammar_rows[0], grammar_rows[1], use_subject=True)
                if grammar_corrupted is not None:
                    delta = abs(grammar_baseline.get(n_layers, 0) - grammar_corrupted.get(n_layers, 0))
                    grammar_deltas.append(delta)
            except:
                pass
            
            # World (measure at last_token position)
            try:
                world_corrupted = get_baseline_margins(world_sent, world_rows[0], world_rows[1], use_subject=False)
                if world_corrupted is not None:
                    delta = abs(world_baseline.get(n_layers, 0) - world_corrupted.get(n_layers, 0))
                    world_deltas.append(delta)
            except:
                pass
            
            hook.remove()
        
        avg_grammar_delta = float(np.mean(grammar_deltas)) if grammar_deltas else 0
        avg_world_delta = float(np.mean(world_deltas)) if world_deltas else 0
        
        grammar_importance[patch_layer] = round(avg_grammar_delta, 4)
        world_importance[patch_layer] = round(avg_world_delta, 4)
        
        print(f"    L{patch_layer}: grammar_imp={avg_grammar_delta:.4f}, world_imp={avg_world_delta:.4f}", flush=True)
    
    # ---- Compare importance profiles ----
    grammar_imp_vec = np.array([grammar_importance.get(l, 0) for l in test_layers], dtype=np.float64)
    world_imp_vec = np.array([world_importance.get(l, 0) for l in test_layers], dtype=np.float64)
    
    if np.std(grammar_imp_vec) > 1e-10 and np.std(world_imp_vec) > 1e-10:
        importance_corr = float(np.corrcoef(grammar_imp_vec, world_imp_vec)[0, 1])
    else:
        importance_corr = 0.0
    
    # Find most important layers for each
    grammar_top_layer = test_layers[np.argmax(grammar_imp_vec)] if len(grammar_imp_vec) > 0 else None
    world_top_layer = test_layers[np.argmax(world_imp_vec)] if len(world_imp_vec) > 0 else None
    
    print(f"\n  ★★★ Causal Importance Results ★★★", flush=True)
    print(f"    Grammar top layer: L{grammar_top_layer}", flush=True)
    print(f"    World top layer: L{world_top_layer}", flush=True)
    print(f"    Importance profile correlation: {importance_corr:.4f}", flush=True)
    print(f"    → {'Same causal mechanism' if abs(importance_corr) > 0.7 else 'Different causal mechanisms' if abs(importance_corr) < 0.3 else 'Partially overlapping mechanisms'}", flush=True)
    
    return {
        "test_layers": test_layers,
        "grammar_importance": {str(l): v for l, v in grammar_importance.items()},
        "world_importance": {str(l): v for l, v in world_importance.items()},
        "importance_correlation": round(importance_corr, 4),
        "grammar_top_layer": grammar_top_layer,
        "world_top_layer": world_top_layer,
        "noise_std": noise_std,
        "n_trials": n_trials,
    }


# =====================================================================
# Phase E: ★★★ 约束传播动力学对比 ★★★
# =====================================================================

def run_propagation_comparison(world_results, grammar_results, model_info):
    """
    ★★★ 传播动力学对比: 语法和世界约束的R²和闭合率是否相同? ★★★
    """
    n_layers = model_info.n_layers
    
    print("\n" + "="*70, flush=True)
    print("Phase E: ★★★ 约束传播动力学对比 ★★★", flush=True)
    print("="*70, flush=True)
    
    world_r2 = world_results["propagation_r2_avg"]
    grammar_r2 = grammar_results["propagation_r2_median"]
    
    world_sat = world_results["satisfaction_rate"]
    grammar_sat = grammar_results["satisfaction_rate"]
    
    # Compare satisfaction rates across layers
    sat_comparison = {}
    for li in range(0, n_layers + 1, max(1, n_layers // 8)):
        li_str = str(li)
        ws = world_sat.get(li_str, 0)
        gs = grammar_sat.get(li_str, 0)
        sat_comparison[li] = {"world": ws, "grammar": gs, "gap": round(ws - gs, 4)}
    
    # Compute "constraint closure rate" for both
    world_closure_rate = 0
    grammar_closure_rate = 0
    
    world_sat_vals = [world_sat.get(str(li), 0) for li in range(n_layers + 1)]
    grammar_sat_vals = [grammar_sat.get(str(li), 0) for li in range(n_layers + 1)]
    
    if len(world_sat_vals) > 1:
        world_closure_rate = world_sat_vals[-1] - world_sat_vals[0]
    if len(grammar_sat_vals) > 1:
        grammar_closure_rate = grammar_sat_vals[-1] - grammar_sat_vals[0]
    
    print(f"\n  ★★★ Propagation Comparison ★★★", flush=True)
    print(f"    World R²: {world_r2:.4f}", flush=True)
    print(f"    Grammar R²: {grammar_r2:.4f}", flush=True)
    print(f"    World closure rate: {world_closure_rate:.4f}", flush=True)
    print(f"    Grammar closure rate: {grammar_closure_rate:.4f}", flush=True)
    
    print(f"\n    Satisfaction rate comparison:", flush=True)
    for li, data in sorted(sat_comparison.items()):
        print(f"      L{li}: world={data['world']:.2%}, grammar={data['grammar']:.2%}, gap={data['gap']:.4f}", flush=True)
    
    r2_diff = abs(world_r2 - grammar_r2)
    if r2_diff < 0.1:
        prop_verdict = "SIMILAR propagation dynamics — consistent with unified mechanism"
    elif r2_diff > 0.3:
        prop_verdict = "DIFFERENT propagation dynamics — evidence for specialized mechanisms"
    else:
        prop_verdict = "PARTIALLY different — needs further investigation"
    
    print(f"\n    ★ VERDICT: {prop_verdict}", flush=True)
    
    return {
        "world_r2": world_r2,
        "grammar_r2": grammar_r2,
        "r2_difference": round(r2_diff, 4),
        "world_closure_rate": round(world_closure_rate, 4),
        "grammar_closure_rate": round(grammar_closure_rate, 4),
        "satisfaction_comparison": {str(li): data for li, data in sat_comparison.items()},
        "verdict": prop_verdict,
    }


# =====================================================================
# MAIN
# =====================================================================

def run_phase179(model_name):
    print(f"\n{'='*70}", flush=True)
    print(f"Phase 179: ★★★ 从语法到世界 — 解缠语言约束与本体约束机制 ★★★", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"{'='*70}", flush=True)

    t_start = time.time()

    # Load model (BF16 + device_map="auto")
    model, tokenizer, device = load_model_bf16(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    print(f"  Model: {model_info.model_class}, L={n_layers}, d={d_model}", flush=True)

    # Load W_U and final norm weight
    print("  Loading W_U and final norm...", flush=True)
    W_U = get_W_U(model, model_name)
    norm_weight = get_final_norm_weight(model)
    print(f"  W_U shape: {W_U.shape}, Norm weight: {'found' if norm_weight is not None else 'not found'}", flush=True)

    # =====================================================================
    # Run all experiments
    # =====================================================================

    # Phase A: World-grounded constraint signals
    exp_a = run_world_constraints(model, tokenizer, device, model_info, W_U, norm_weight)

    # Phase B: Grammar constraint signals (for comparison)
    exp_b = run_grammar_constraints(model, tokenizer, device, model_info, W_U, norm_weight)

    # Phase C: Grammar vs World profile comparison
    exp_c = run_grammar_vs_world_profile(exp_a, exp_b, model_info)

    # Phase D: Causal layer importance comparison
    exp_d = run_causal_layer_importance(model, tokenizer, device, model_info, W_U, norm_weight)

    # Phase E: Propagation dynamics comparison
    exp_e = run_propagation_comparison(exp_a, exp_b, model_info)

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "timestamp": timestamp,
        "phase_A_world_constraints": exp_a,
        "phase_B_grammar_constraints": exp_b,
        "phase_C_profile_comparison": exp_c,
        "phase_D_causal_importance": exp_d,
        "phase_E_propagation_comparison": exp_e,
    }

    out_path = f"tests/glm5_temp/phase179_{model_name}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    # Release model
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\nPhase 179 ({model_name}) completed in {elapsed:.1f}s", flush=True)

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase179_grammar_vs_world.py <model_name>")
        print("  model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    run_phase179(model_name)
