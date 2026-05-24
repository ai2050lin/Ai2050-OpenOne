"""
Phase 257: Grammar Attention Head Decoding + W_U Geometric Analysis
===================================================================

Parts:
  Part 1: Grammar attention head identification (per-head logit attribution)
  Part 2: Grammar head OV circuit decoding
  Part 3: W_U geometric analysis by POS (no hypothesis, just observe)
  Part 4: Model-based co-occurrence measurement (refined)

Usage:
  python tests/glm5/phase257_grammar_wu_geometry.py --model qwen3 --part 1
  python tests/glm5/phase257_grammar_wu_geometry.py --model qwen3 --part all
  python tests/glm5/phase257_grammar_wu_geometry.py --model glm4 --part 1
  python tests/glm5/phase257_grammar_wu_geometry.py --model deepseek7b --part 1
"""

import sys, os, json, argparse, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = Path("results/phase257_grammar_geometry")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 工具函数
# ============================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

def load_model_safe(model_name):
    """加载模型, bfloat16 + device_map=auto + flash attention"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS

    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} from {cfg['path']}...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try flash_attention_2 first (memory efficient), then eager
    for attn_impl in ["flash_attention_2", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=attn_impl,
            )
            log_time(f"Loaded with attn_implementation={attn_impl}")
            break
        except Exception as e:
            log_time(f"  {attn_impl} failed: {e}, trying next...")
            continue

    model.eval()
    from model_utils import get_model_info
    info = get_model_info(model, model_name)
    log_time(f"{model_name}: class={info.model_class}, layers={info.n_layers}, "
             f"d_model={info.d_model}, vocab={info.vocab_size}")

    # Get attention head info from config
    config = model.config
    n_heads = getattr(config, 'num_attention_heads', 32)
    # head_dim may be specified explicitly in config (e.g., Qwen3 has head_dim=128)
    head_dim = getattr(config, 'head_dim', info.d_model // n_heads)
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)

    log_time(f"  n_heads={n_heads}, head_dim={head_dim}, n_kv_heads={n_kv_heads}")

    return model, tokenizer, info, n_heads, head_dim

def get_W_U_safe(model, model_name):
    """获取W_U, 处理meta tensor"""
    from model_utils import get_W_U
    return get_W_U(model, model_name)  # [vocab_size, d_model]

def release_model_safe(model):
    """释放模型"""
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log_time("Model released, GPU cleared")

def save_result(model_name, part, data):
    """保存结果"""
    fname = RESULT_DIR / f"{model_name}_part{part}.json"
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)
    log_time(f"Results saved to {fname}")

def safe_decode(tokenizer, token_id):
    """安全解码token"""
    try:
        r = tokenizer.decode([token_id])
        return r.strip() if r else f"<tok_{token_id}>"
    except:
        return f"<tok_{token_id}>"

def get_input_device(model):
    """获取模型输入设备"""
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_weight_to_numpy(weight_tensor, model_name=None, layer_name=None):
    """安全地将权重tensor转为numpy, 处理meta device"""
    import torch
    if weight_tensor.is_meta:
        # Load from safetensors
        from model_utils import MODEL_CONFIGS
        import glob, os
        from safetensors import safe_open
        model_path = MODEL_CONFIGS.get(model_name, {}).get("path", None)
        if model_path and layer_name:
            sf_files = glob.glob(os.path.join(model_path, '*.safetensors'))
            for sf_file in sf_files:
                with safe_open(sf_file, framework='pt', device='cpu') as sf:
                    if layer_name in sf.keys():
                        w = sf.get_tensor(layer_name)
                        return w.float().numpy()
        raise ValueError(f"Cannot load meta tensor {layer_name} for {model_name}")
    return weight_tensor.detach().cpu().float().numpy()


# ============================================================
# Part 1: Grammar Attention Head Identification
# ============================================================

def part1_grammar_head_identification(model_name):
    """
    用per-head logit attribution找到负责语法处理的attention heads

    方法:
    1. 定义语法任务(prompt, correct_token, incorrect_token)
    2. 对每个任务, hook每层self_attn的o_proj输入, 获取per-head输出
    3. 计算每个head对correct_token和incorrect_token的logit贡献
    4. grammar_signal = contribution(correct) - contribution(incorrect)
    5. 找到跨多个语法任务grammar_signal一致高的heads
    """
    import torch
    from model_utils import get_layers

    model, tokenizer, info, n_heads, head_dim = load_model_safe(model_name)
    W_U = get_W_U_safe(model, model_name)  # [vocab_size, d_model]
    layers = get_layers(model)
    input_device = get_input_device(model)

    results = {"model": model_name, "n_layers": info.n_layers,
               "n_heads": n_heads, "head_dim": head_dim}

    # ---- Step 1: 定义语法任务 ----
    grammar_tasks = [
        # SV agreement: singular → singular verb
        {"name": "sv_sing_cat", "prompt": "The cat", "target_word": " sits", "competitor_word": " sit", "type": "agreement"},
        {"name": "sv_sing_dog", "prompt": "The dog", "target_word": " runs", "competitor_word": " run", "type": "agreement"},
        {"name": "sv_sing_she", "prompt": "She", "target_word": " walks", "competitor_word": " walk", "type": "agreement"},
        # SV agreement: plural → plural verb
        {"name": "sv_plur_cats", "prompt": "The cats", "target_word": " sit", "competitor_word": " sits", "type": "agreement"},
        {"name": "sv_plur_dogs", "prompt": "The dogs", "target_word": " run", "competitor_word": " runs", "type": "agreement"},
        {"name": "sv_plur_they", "prompt": "They", "target_word": " walk", "competitor_word": " walks", "type": "agreement"},
        # Past tense
        {"name": "past_went", "prompt": "Yesterday, she", "target_word": " went", "competitor_word": " goes", "type": "tense"},
        {"name": "past_ate", "prompt": "Last night, he", "target_word": " ate", "competitor_word": " eats", "type": "tense"},
        # Comparative vs superlative
        {"name": "comp_than", "prompt": "She is taller", "target_word": " than", "competitor_word": " then", "type": "comparative"},
        # Pronoun case
        {"name": "pronoun_he", "prompt": "___ is happy", "target_word": "He", "competitor_word": "Him", "type": "case"},
        # Article
        {"name": "article_a", "prompt": "I saw", "target_word": " a", "competitor_word": " an", "type": "article"},
        # Chinese classifier
        {"name": "clf_zhi", "prompt": "一", "target_word": "只", "competitor_word": "条", "type": "classifier"},
    ]

    # ---- Step 2: 获取W_O权重 ----
    log_time("Extracting W_O weights for all layers...")
    W_O_all = {}  # {layer_idx: [d_model, n_heads*head_dim]}
    for li in range(info.n_layers):
        layer = layers[li]
        w = layer.self_attn.o_proj.weight
        if w.is_meta:
            layer_name = f"model.layers.{li}.self_attn.o_proj.weight"
            W_O_all[li] = safe_weight_to_numpy(w, model_name, layer_name)
        else:
            W_O_all[li] = w.detach().cpu().float().numpy()
    log_time(f"W_O extracted for {len(W_O_all)} layers")

    # ---- Step 3: 对每个任务做per-head attribution ----
    all_task_results = {}

    for task_idx, task in enumerate(grammar_tasks):
        task_name = task["name"]
        prompt = task["prompt"]

        log_time(f"Task {task_idx+1}/{len(grammar_tasks)}: {task_name} — '{prompt}'")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        # Get target and competitor token IDs
        target_ids = tokenizer.encode(task["target_word"], add_special_tokens=False)
        competitor_ids = tokenizer.encode(task["competitor_word"], add_special_tokens=False)

        if not target_ids or not competitor_ids:
            log_time(f"  Cannot tokenize target/competitor, skipping")
            continue

        target_id = target_ids[0]
        competitor_id = competitor_ids[0]

        if target_id >= W_U.shape[0] or competitor_id >= W_U.shape[0]:
            log_time(f"  Token IDs out of range, skipping")
            continue

        # Target directions in W_U
        target_dir = W_U[target_id]  # [d_model]
        competitor_dir = W_U[competitor_id]  # [d_model]

        # Hook o_proj input to get per-head outputs
        # o_proj input shape: [batch, seq, n_heads * head_dim]
        head_outputs_captured = {}

        def make_o_proj_hook(li):
            def hook(module, input, output):
                # input is a tuple; input[0] is the tensor before o_proj
                inp = input[0]  # [batch, seq, n_heads * head_dim]
                # Split into per-head outputs
                batch, seq, _ = inp.shape
                head_outs = inp.view(batch, seq, n_heads, head_dim)
                # Store last position's head outputs
                head_outputs_captured[li] = head_outs[0, -1, :, :].detach().float().cpu().numpy()
                # shape: [n_heads, head_dim]
            return hook

        hooks = []
        for li in range(info.n_layers):
            layer = layers[li]
            hooks.append(layer.self_attn.o_proj.register_forward_hook(make_o_proj_hook(li)))

        # Also hook MLP output for comparison
        mlp_outputs_captured = {}
        def make_mlp_hook(li):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    mlp_outputs_captured[li] = output[0][0, -1].detach().float().cpu().numpy()
                else:
                    mlp_outputs_captured[li] = output[0, -1].detach().float().cpu().numpy()
            return hook

        mlp_hooks = []
        for li in range(info.n_layers):
            layer = layers[li]
            if hasattr(layer, 'mlp'):
                mlp_hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(li)))

        # Forward pass
        with torch.no_grad():
            try:
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_hidden_states=True)
            except Exception as e:
                log_time(f"  Forward failed: {e}")
                for h in hooks + mlp_hooks:
                    h.remove()
                continue

        for h in hooks + mlp_hooks:
            h.remove()

        # ---- Compute per-head logit attribution ----
        # For head h at layer l:
        #   contribution = target_dir @ W_O[l][:, h*hd:(h+1)*hd] @ head_output[l][h]
        # This is: (target_dir @ W_O_h) dot head_output_h

        head_grammar_signals = {}
        head_target_contribs = {}
        head_competitor_contribs = {}
        layer_mlp_contribs = {}

        for li in range(info.n_layers):
            if li not in head_outputs_captured:
                continue

            head_outs = head_outputs_captured[li]  # [n_heads, head_dim]
            W_O_l = W_O_all[li]  # [d_model, n_heads * head_dim]

            for h in range(n_heads):
                W_O_h = W_O_l[:, h * head_dim:(h + 1) * head_dim]  # [d_model, head_dim]

                # Target contribution
                proj_target = target_dir @ W_O_h  # [head_dim]
                contrib_target = float(np.dot(proj_target, head_outs[h]))

                # Competitor contribution
                proj_competitor = competitor_dir @ W_O_h  # [head_dim]
                contrib_competitor = float(np.dot(proj_competitor, head_outs[h]))

                # Grammar signal
                grammar_signal = contrib_target - contrib_competitor

                head_grammar_signals[(li, h)] = grammar_signal
                head_target_contribs[(li, h)] = contrib_target
                head_competitor_contribs[(li, h)] = contrib_competitor

            # MLP contribution
            if li in mlp_outputs_captured:
                mlp_out = mlp_outputs_captured[li]
                contrib_target_mlp = float(np.dot(target_dir, mlp_out))
                contrib_competitor_mlp = float(np.dot(competitor_dir, mlp_out))
                layer_mlp_contribs[li] = {
                    "target": contrib_target_mlp,
                    "competitor": contrib_competitor_mlp,
                    "grammar_signal": contrib_target_mlp - contrib_competitor_mlp,
                }

        # Actual logits for verification
        final_logits = out.logits[0, -1].float().cpu().numpy()
        actual_target_logit = float(final_logits[target_id])
        actual_competitor_logit = float(final_logits[competitor_id])

        task_result = {
            "prompt": prompt,
            "target_word": task["target_word"],
            "competitor_word": task["competitor_word"],
            "target_id": target_id,
            "competitor_id": competitor_id,
            "actual_target_logit": round(actual_target_logit, 3),
            "actual_competitor_logit": round(actual_competitor_logit, 3),
            "logit_diff": round(actual_target_logit - actual_competitor_logit, 3),
            "task_type": task["type"],
        }

        # Top-10 heads by grammar signal (positive = promotes correct grammar)
        sorted_heads = sorted(head_grammar_signals.items(), key=lambda x: x[1], reverse=True)
        top10_positive = [(f"L{l}_H{h}", round(gs, 3)) for (l, h), gs in sorted_heads[:10]]
        top10_negative = [(f"L{l}_H{h}", round(gs, 3)) for (l, h), gs in sorted_heads[-10:]]

        task_result["top10_positive_grammar_heads"] = top10_positive
        task_result["top10_negative_grammar_heads"] = top10_negative

        # Layer-level summary
        layer_grammar = defaultdict(float)
        for (l, h), gs in head_grammar_signals.items():
            layer_grammar[l] += gs

        sorted_layers = sorted(layer_grammar.items(), key=lambda x: x[1], reverse=True)
        task_result["top5_grammar_layers"] = [(f"L{l}", round(gs, 3)) for l, gs in sorted_layers[:5]]

        # MLP grammar signal
        mlp_grammar = [(f"L{l}", round(v["grammar_signal"], 3))
                       for l, v in sorted(layer_mlp_contribs.items(), key=lambda x: x[1]["grammar_signal"], reverse=True)[:5]]
        task_result["top5_grammar_mlp_layers"] = mlp_grammar

        all_task_results[task_name] = task_result

        log_time(f"  Logit diff (target-competitor): {actual_target_logit - actual_competitor_logit:.3f}")
        log_time(f"  Top heads: {top10_positive[:3]}")
        log_time(f"  Top layers: {task_result['top5_grammar_layers'][:3]}")

        # Clean up
        del head_outputs_captured, mlp_outputs_captured, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Step 4: 跨任务一致性分析 ----
    log_time("\n=== Cross-task consistency analysis ===")

    # For each head, collect grammar signals across all tasks of each type
    head_signals_by_type = defaultdict(lambda: defaultdict(list))
    for task_name, tr in all_task_results.items():
        task_type = tr["task_type"]
        # Recompute per-head signals (we stored them in head_grammar_signals but that's per-task)
        # Actually we need to rerun... but that's too expensive
        # Instead, use the top heads from each task
        pass

    # Alternative: count how many times each head appears in top-10 positive heads
    head_appearance_count = Counter()
    for task_name, tr in all_task_results.items():
        for head_label, _ in tr["top10_positive_grammar_heads"]:
            head_appearance_count[head_label] += 1

    # Heads that appear in top-10 for multiple tasks
    consistent_heads = [(h, c) for h, c in head_appearance_count.most_common(20) if c >= 2]

    results["task_results"] = all_task_results
    results["consistent_grammar_heads"] = consistent_heads

    log_time(f"\nConsistent grammar heads (appear in top-10 for >=2 tasks):")
    for h, c in consistent_heads:
        log_time(f"  {h}: {c} tasks")

    save_result(model_name, 1, results)
    release_model_safe(model)
    return results


# ============================================================
# Part 2: Grammar Head OV Circuit Decoding
# ============================================================

def part2_grammar_ov_decoding(model_name):
    """
    对Part 1找到的grammar heads, 解码它们的OV电路

    OV电路: W_O[h] @ W_V[h]
    当head h attend到位置p时, 它写入: OV_h @ residual[p]

    对语法head, 我们想知道:
    1. 当attend到单数名词时, 是否促进单数动词的logit?
    2. 当attend到复数名词时, 是否促进复数动词的logit?

    方法: 计算 logit_effect[Y, X] = W_U[Y] @ W_O_h @ W_V_h @ W_E[X]
    """
    import torch
    from model_utils import get_layers

    model, tokenizer, info, n_heads, head_dim = load_model_safe(model_name)
    W_U = get_W_U_safe(model, model_name)  # [vocab_size, d_model]
    layers = get_layers(model)

    results = {"model": model_name}

    # Load Part 1 results to get grammar heads
    part1_path = RESULT_DIR / f"{model_name}_part1.json"
    if not part1_path.exists():
        log_time("Part 1 results not found, analyzing all heads in grammar-relevant layers")
        # Default: analyze last 1/3 of layers
        grammar_layers = list(range(info.n_layers * 2 // 3, info.n_layers))
    else:
        with open(part1_path, 'r', encoding='utf-8') as f:
            part1_data = json.load(f)
        consistent_heads = part1_data.get("consistent_grammar_heads", [])
        # Parse layer and head indices from labels like "L5_H12"
        target_heads = []
        for label, count in consistent_heads:
            parts = label.replace("L", "").split("_H")
            if len(parts) == 2:
                target_heads.append((int(parts[0]), int(parts[1])))
        log_time(f"Loaded {len(target_heads)} grammar heads from Part 1")

        if not target_heads:
            grammar_layers = list(range(info.n_layers * 2 // 3, info.n_layers))
            target_heads = None
        else:
            grammar_layers = sorted(set(l for l, h in target_heads))

    # ---- Step 1: 定义语法相关token ----
    singular_nouns = ["cat", "dog", "girl", "boy", "man", "woman", "child", "bird",
                      "fish", "tree", "house", "car", "book", "table", "chair", "door"]
    plural_nouns = ["cats", "dogs", "girls", "boys", "men", "women", "children", "birds",
                    "fish", "trees", "houses", "cars", "books", "tables", "chairs", "doors"]
    singular_verbs = ["sits", "runs", "walks", "eats", "sleeps", "jumps", "swims", "reads",
                      "writes", "speaks", "thinks", "works", "plays", "sings", "drives", "moves"]
    plural_verbs = ["sit", "run", "walk", "eat", "sleep", "jump", "swim", "read",
                    "write", "speak", "think", "work", "play", "sing", "drive", "move"]

    # Tokenize all words
    def get_token_ids(words):
        ids = []
        valid_words = []
        for w in words:
            tok_ids = tokenizer.encode(w, add_special_tokens=False)
            if tok_ids and len(tok_ids) == 1:
                ids.append(tok_ids[0])
                valid_words.append(w)
        return ids, valid_words

    sing_noun_ids, sing_noun_words = get_token_ids(singular_nouns)
    plur_noun_ids, plur_noun_words = get_token_ids(plural_nouns)
    sing_verb_ids, sing_verb_words = get_token_ids(singular_verbs)
    plur_verb_ids, plur_verb_words = get_token_ids(plural_verbs)

    log_time(f"Valid tokens: {len(sing_noun_ids)} sing nouns, {len(plur_noun_ids)} plur nouns, "
             f"{len(sing_verb_ids)} sing verbs, {len(plur_verb_ids)} plur verbs")

    # Get word embeddings
    embed_layer = model.get_input_embeddings()
    W_E = embed_layer.weight.detach().cpu().float().numpy()  # [vocab_size, d_model]

    results["valid_token_counts"] = {
        "singular_nouns": len(sing_noun_ids),
        "plural_nouns": len(plur_noun_ids),
        "singular_verbs": len(sing_verb_ids),
        "plural_verbs": len(plur_verb_ids),
    }

    # ---- Step 2: 计算OV电路 ----
    # For each head, compute: W_U @ W_O_h @ W_V_h @ W_E
    # This is [vocab_size, vocab_size] but we only compute for selected token pairs

    # If no specific grammar heads from Part 1, sample from grammar_layers
    if target_heads is None:
        # Sample 5 heads from each grammar layer
        np.random.seed(42)
        target_heads = []
        for li in grammar_layers[:5]:  # Only top 5 grammar layers
            for h in np.random.choice(n_heads, min(5, n_heads), replace=False):
                target_heads.append((li, int(h)))

    log_time(f"Analyzing OV circuits for {len(target_heads)} heads...")

    ov_results = {}

    for li, h in target_heads:
        layer = layers[li]

        # W_V: shape depends on grouped query attention
        w_v = layer.self_attn.v_proj.weight
        if w_v.is_meta:
            W_V_full = safe_weight_to_numpy(w_v, model_name, f"model.layers.{li}.self_attn.v_proj.weight")
        else:
            W_V_full = w_v.detach().cpu().float().numpy()

        # Handle GQA: if n_kv_heads < n_heads, heads share KV
        n_kv_heads = getattr(model.config, 'num_key_value_heads', n_heads)
        kv_head_dim = W_V_full.shape[0] // n_kv_heads

        # Map attention head to KV head
        kv_head_idx = h // (n_heads // n_kv_heads) if n_kv_heads < n_heads else h
        W_V_h = W_V_full[kv_head_idx * kv_head_dim:(kv_head_idx + 1) * kv_head_dim, :]  # [head_dim, d_model]

        # W_O: [d_model, n_heads * head_dim] → W_O_h: [d_model, head_dim]
        w_o = layer.self_attn.o_proj.weight
        if w_o.is_meta:
            W_O_full = safe_weight_to_numpy(w_o, model_name, f"model.layers.{li}.self_attn.o_proj.weight")
        else:
            W_O_full = w_o.detach().cpu().float().numpy()
        W_O_h = W_O_full[:, h * head_dim:(h + 1) * head_dim]  # [d_model, head_dim]

        # OV circuit: W_O_h @ W_V_h → [d_model, d_model]
        OV_h = W_O_h @ W_V_h  # [d_model, d_model]

        # Logit effect: W_U @ OV_h → [vocab_size, d_model]
        # logit_effect[Y] @ embedding[X] = logit change for Y when attending to X
        # But full computation is [vocab, d_model] which is huge
        # Instead: compute only for our selected token pairs

        # For singular nouns → what verbs are promoted?
        sing_noun_effects = {}  # verb_id → mean logit effect
        for verb_id in sing_verb_ids + plur_verb_ids:
            verb_dir = W_U[verb_id]  # [d_model]
            # verb_dir @ OV_h → [d_model]
            verb_ov = verb_dir @ OV_h  # [d_model]
            # Effect when attending to each singular noun
            effects = [float(verb_ov @ W_E[nid]) for nid in sing_noun_ids]
            sing_noun_effects[verb_id] = float(np.mean(effects))

        # For plural nouns → what verbs are promoted?
        plur_noun_effects = {}
        for verb_id in sing_verb_ids + plur_verb_ids:
            verb_dir = W_U[verb_id]
            verb_ov = verb_dir @ OV_h
            effects = [float(verb_ov @ W_E[nid]) for nid in plur_noun_ids]
            plur_noun_effects[verb_id] = float(np.mean(effects))

        # Grammar agreement score:
        # A grammar head should: promote singular verbs when attending to singular nouns,
        #                       promote plural verbs when attending to plural nouns
        sing_verb_promoted_by_sing = np.mean([sing_noun_effects[vid] for vid in sing_verb_ids])
        plur_verb_promoted_by_sing = np.mean([sing_noun_effects[vid] for vid in plur_verb_ids])
        sing_verb_promoted_by_plur = np.mean([plur_noun_effects[vid] for vid in sing_verb_ids])
        plur_verb_promoted_by_plur = np.mean([plur_noun_effects[vid] for vid in plur_verb_ids])

        agreement_score = (sing_verb_promoted_by_sing - plur_verb_promoted_by_sing) + \
                         (plur_verb_promoted_by_plur - sing_verb_promoted_by_plur)

        # Top promoted verbs by singular/plural nouns
        all_verb_ids = sing_verb_ids + plur_verb_ids
        top_sing_promoted = sorted(all_verb_ids, key=lambda vid: sing_noun_effects[vid], reverse=True)[:5]
        top_plur_promoted = sorted(all_verb_ids, key=lambda vid: plur_noun_effects[vid], reverse=True)[:5]

        ov_results[f"L{li}_H{h}"] = {
            "agreement_score": round(float(agreement_score), 4),
            "sing_verb_promoted_by_sing_nouns": round(float(sing_verb_promoted_by_sing), 4),
            "plur_verb_promoted_by_sing_nouns": round(float(plur_verb_promoted_by_sing), 4),
            "sing_verb_promoted_by_plur_nouns": round(float(sing_verb_promoted_by_plur), 4),
            "plur_verb_promoted_by_plur_nouns": round(float(plur_verb_promoted_by_plur), 4),
            "top_sing_noun_promoted_verbs": [(safe_decode(tokenizer, vid), round(sing_noun_effects[vid], 3))
                                              for vid in top_sing_promoted],
            "top_plur_noun_promoted_verbs": [(safe_decode(tokenizer, vid), round(plur_noun_effects[vid], 3))
                                              for vid in top_plur_promoted],
        }

        log_time(f"  L{li}_H{h}: agreement={agreement_score:.4f}, "
                 f"sing→sing_v={sing_verb_promoted_by_sing:.4f}, "
                 f"plur→plur_v={plur_verb_promoted_by_plur:.4f}")

    # Sort by agreement score
    sorted_heads = sorted(ov_results.items(), key=lambda x: x[1]["agreement_score"], reverse=True)

    results["ov_circuit_analysis"] = ov_results
    results["top_grammar_ov_heads"] = [(h, d["agreement_score"]) for h, d in sorted_heads[:10]]

    log_time(f"\nTop grammar OV heads (by agreement score):")
    for h, d in sorted_heads[:10]:
        log_time(f"  {h}: agreement={d['agreement_score']:.4f}")

    # ---- Step 3: 每层的OV电路分析汇总 ----
    # Sample a few heads per layer across all layers to find grammar-dedicated layers
    log_time("\nScanning all layers for grammar OV signal...")
    layer_agreement_scores = {}

    sample_heads_per_layer = min(8, n_heads)  # Sample 8 heads per layer
    np.random.seed(42)

    for li in range(info.n_layers):
        layer = layers[li]
        w_v = layer.self_attn.v_proj.weight
        w_o = layer.self_attn.o_proj.weight
        if w_v.is_meta:
            W_V_full = safe_weight_to_numpy(w_v, model_name, f"model.layers.{li}.self_attn.v_proj.weight")
        else:
            W_V_full = w_v.detach().cpu().float().numpy()
        if w_o.is_meta:
            W_O_full = safe_weight_to_numpy(w_o, model_name, f"model.layers.{li}.self_attn.o_proj.weight")
        else:
            W_O_full = w_o.detach().cpu().float().numpy()

        n_kv_heads = getattr(model.config, 'num_key_value_heads', n_heads)
        kv_head_dim = W_V_full.shape[0] // n_kv_heads

        head_agreements = []
        sample_h = np.random.choice(n_heads, sample_heads_per_layer, replace=False)

        for h in sample_h:
            kv_head_idx = h // (n_heads // n_kv_heads) if n_kv_heads < n_heads else h
            W_V_h = W_V_full[kv_head_idx * kv_head_dim:(kv_head_idx + 1) * kv_head_dim, :]
            W_O_h = W_O_full[:, h * head_dim:(h + 1) * head_dim]
            OV_h = W_O_h @ W_V_h

            # Quick agreement test with fewer tokens
            verb_ov_sing = np.mean([W_U[vid] @ OV_h for vid in sing_verb_ids[:5]], axis=0)
            verb_ov_plur = np.mean([W_U[vid] @ OV_h for vid in plur_verb_ids[:5]], axis=0)

            sing_noun_effect_sing_v = float(np.mean([verb_ov_sing @ W_E[nid] for nid in sing_noun_ids[:5]]))
            sing_noun_effect_plur_v = float(np.mean([verb_ov_plur @ W_E[nid] for nid in sing_noun_ids[:5]]))
            plur_noun_effect_sing_v = float(np.mean([verb_ov_sing @ W_E[nid] for nid in plur_noun_ids[:5]]))
            plur_noun_effect_plur_v = float(np.mean([verb_ov_plur @ W_E[nid] for nid in plur_noun_ids[:5]]))

            agreement = (sing_noun_effect_sing_v - sing_noun_effect_plur_v) + \
                       (plur_noun_effect_plur_v - plur_noun_effect_sing_v)
            head_agreements.append(agreement)

        layer_agreement_scores[li] = {
            "mean": round(float(np.mean(head_agreements)), 4),
            "max": round(float(np.max(head_agreements)), 4),
            "min": round(float(np.min(head_agreements)), 4),
        }

        if (li + 1) % 10 == 0:
            log_time(f"  Processed {li+1}/{info.n_layers} layers")

    results["layer_agreement_scores"] = layer_agreement_scores

    # Top grammar layers
    sorted_layers = sorted(layer_agreement_scores.items(), key=lambda x: x[1]["max"], reverse=True)
    log_time(f"\nTop grammar layers (by max agreement score):")
    for li, scores in sorted_layers[:5]:
        log_time(f"  L{li}: max={scores['max']:.4f}, mean={scores['mean']:.4f}")

    save_result(model_name, 2, results)
    release_model_safe(model)
    return results


# ============================================================
# Part 3: W_U Geometric Analysis by POS
# ============================================================

def part3_wu_pos_geometry(model_name):
    """
    不提假说, 直接观测W_U的几何结构

    分析:
    1. 按词性(POS)分组, 计算组内/组间cosine
    2. 功能词 vs 内容词的聚类紧密度
    3. 各词类在W_U空间中的分布维度
    4. 层次聚类
    """
    import torch
    from model_utils import get_layers

    model, tokenizer, info, n_heads, head_dim = load_model_safe(model_name)
    W_U = get_W_U_safe(model, model_name)  # [vocab_size, d_model]

    results = {"model": model_name, "d_model": info.d_model, "vocab_size": info.vocab_size}

    # ---- Step 1: 按词性定义词组 ----
    pos_groups = {
        "nouns": [
            "cat", "dog", "house", "car", "tree", "book", "water", "food",
            "city", "hand", "eye", "door", "road", "fire", "air", "land",
            "man", "woman", "child", "day", "night", "year", "time",
            "world", "life", "work", "heart", "mind", "power",
            # Chinese
            "猫", "狗", "房子", "车", "树", "书", "水", "食物",
        ],
        "verbs": [
            "run", "walk", "eat", "sleep", "read", "write", "speak", "think",
            "work", "play", "make", "take", "give", "come", "go", "see",
            "know", "feel", "want", "need", "love", "help", "start", "stop",
            "open", "close", "move", "turn", "fall", "rise",
            # Chinese
            "跑", "走", "吃", "睡", "读", "写", "说", "想",
        ],
        "adjectives": [
            "big", "small", "hot", "cold", "fast", "slow", "good", "bad",
            "new", "old", "high", "low", "long", "short", "hard", "soft",
            "dark", "bright", "rich", "poor", "strong", "weak", "happy", "sad",
            "beautiful", "ugly", "clean", "dirty", "safe", "dangerous",
            # Chinese
            "大", "小", "热", "冷", "快", "慢", "好", "坏",
        ],
        "adverbs": [
            "quickly", "slowly", "carefully", "easily", "often", "never",
            "always", "sometimes", "already", "still", "just", "very",
            "really", "probably", "certainly", "suddenly",
        ],
        "prepositions": [
            "in", "on", "at", "to", "from", "with", "by", "for",
            "of", "about", "into", "through", "between", "under", "over", "after",
        ],
        "conjunctions": [
            "and", "but", "or", "nor", "so", "yet", "because", "although",
            "while", "if", "when", "since", "unless", "until", "whether", "though",
        ],
        "determiners": [
            "the", "a", "an", "this", "that", "these", "those",
            "my", "your", "his", "her", "its", "our", "their",
            "some", "any", "all", "each", "every", "no",
        ],
        "pronouns": [
            "I", "you", "he", "she", "it", "we", "they",
            "me", "him", "us", "them", "who", "what", "which",
        ],
        "auxiliaries": [
            "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did",
            "will", "would", "can", "could", "may", "might", "shall", "should",
        ],
    }

    # ---- Step 2: 获取有效token的W_U向量 ----
    pos_vectors = {}  # {pos_name: {word: vector}}
    pos_valid_words = {}

    for pos_name, words in pos_groups.items():
        vectors = {}
        valid = []
        for w in words:
            tok_ids = tokenizer.encode(w, add_special_tokens=False)
            if not tok_ids:
                continue
            tid = tok_ids[0]
            if tid >= W_U.shape[0]:
                continue
            # Only use single-token words
            if len(tok_ids) == 1:
                vectors[w] = W_U[tid]
                valid.append(w)
        pos_vectors[pos_name] = vectors
        pos_valid_words[pos_name] = valid

    results["valid_word_counts"] = {pos: len(words) for pos, words in pos_valid_words.items()}

    for pos, words in pos_valid_words.items():
        log_time(f"  {pos}: {len(words)} valid words")

    # ---- Step 3: 组内cosine分析 ----
    log_time("\n--- Intra-POS cosine analysis ---")

    intra_cosines = {}
    for pos_name, vectors in pos_vectors.items():
        if len(vectors) < 2:
            continue
        vec_list = list(vectors.values())
        n = len(vec_list)
        cosines = []
        for i in range(n):
            for j in range(i + 1, n):
                cosines.append(cosine_sim(vec_list[i], vec_list[j]))
        mean_cos = float(np.mean(cosines))
        std_cos = float(np.std(cosines))
        intra_cosines[pos_name] = {
            "mean": round(mean_cos, 4),
            "std": round(std_cos, 4),
            "n_words": n,
            "n_pairs": len(cosines),
        }
        log_time(f"  {pos_name}: mean_cos={mean_cos:.4f}, std={std_cos:.4f}, n={n}")

    results["intra_pos_cosine"] = intra_cosines

    # ---- Step 4: 组间cosine分析 ----
    log_time("\n--- Inter-POS cosine analysis ---")

    pos_names = list(pos_vectors.keys())
    inter_cosines = {}

    for i, pos_a in enumerate(pos_names):
        for j, pos_b in enumerate(pos_names):
            if i >= j:
                continue
            vecs_a = list(pos_vectors[pos_a].values())
            vecs_b = list(pos_vectors[pos_b].values())
            if not vecs_a or not vecs_b:
                continue

            cosines = []
            for va in vecs_a:
                for vb in vecs_b:
                    cosines.append(cosine_sim(va, vb))

            key = f"{pos_a}_vs_{pos_b}"
            inter_cosines[key] = {
                "mean": round(float(np.mean(cosines)), 4),
                "std": round(float(np.std(cosines)), 4),
            }

    results["inter_pos_cosine"] = inter_cosines

    # ---- Step 5: 功能词 vs 内容词 ----
    log_time("\n--- Function words vs Content words ---")

    function_pos = ["prepositions", "conjunctions", "determiners", "pronouns", "auxiliaries"]
    content_pos = ["nouns", "verbs", "adjectives", "adverbs"]

    func_intra = [intra_cosines[p]["mean"] for p in function_pos if p in intra_cosines]
    content_intra = [intra_cosines[p]["mean"] for p in content_pos if p in intra_cosines]

    results["function_vs_content"] = {
        "function_words_mean_intra_cosine": round(float(np.mean(func_intra)), 4) if func_intra else None,
        "content_words_mean_intra_cosine": round(float(np.mean(content_intra)), 4) if content_intra else None,
        "function_pos_details": {p: intra_cosines.get(p, {}) for p in function_pos},
        "content_pos_details": {p: intra_cosines.get(p, {}) for p in content_pos},
    }

    log_time(f"  Function words mean intra-cosine: {np.mean(func_intra):.4f}" if func_intra else "  No function word data")
    log_time(f"  Content words mean intra-cosine: {np.mean(content_intra):.4f}" if content_intra else "  No content word data")

    # ---- Step 6: 各词类在W_U SVD空间中的分布 ----
    log_time("\n--- POS distribution in W_U SVD space ---")

    from scipy.sparse.linalg import svds
    n_components = min(100, min(W_U.shape) - 1)
    U_svd, S_svd, Vt_svd = svds(W_U.astype(np.float32), k=n_components)
    order = np.argsort(S_svd)[::-1]
    U_svd, S_svd, Vt_svd = U_svd[:, order], S_svd[order], Vt_svd[order, :]

    # Vt_svd[i, :] is the i-th right singular vector in vocab space
    # For each POS group, compute the projection onto top SVD dimensions
    pos_svd_projections = {}
    for pos_name, vectors in pos_vectors.items():
        if not vectors:
            continue
        vec_matrix = np.stack(list(vectors.values()))  # [n_words, d_model]
        # Project onto SVD right vectors
        projections = vec_matrix @ Vt_svd[:50, :].T  # [n_words, 50]
        # Mean projection on each axis
        mean_proj = np.mean(projections, axis=0)
        # Energy distribution
        energy_per_axis = np.mean(projections ** 2, axis=0)
        total_energy = np.sum(energy_per_axis)

        pos_svd_projections[pos_name] = {
            "top_axis": int(np.argmax(np.abs(mean_proj))),
            "top_axis_mean_proj": round(float(np.max(np.abs(mean_proj))), 4),
            "energy_concentration_top10": round(float(np.sum(energy_per_axis[:10]) / total_energy), 4) if total_energy > 0 else 0,
            "energy_concentration_top50": round(float(np.sum(energy_per_axis[:50]) / total_energy), 4) if total_energy > 0 else 0,
            "mean_proj_top10": mean_proj[:10].tolist(),
        }

    results["pos_svd_projections"] = pos_svd_projections

    for pos_name, proj in pos_svd_projections.items():
        log_time(f"  {pos_name}: top_axis={proj['top_axis']}, "
                 f"energy_top10={proj['energy_concentration_top10']:.4f}")

    # ---- Step 7: 子空间正交性 ----
    log_time("\n--- Subspace orthogonality between POS groups ---")

    # For each POS group, compute the principal subspace (top-5 PCA)
    from numpy.linalg import svd

    pos_subspaces = {}
    for pos_name, vectors in pos_vectors.items():
        if len(vectors) < 5:
            continue
        vec_matrix = np.stack(list(vectors.values()))  # [n_words, d_model]
        # Subtract mean
        vec_centered = vec_matrix - np.mean(vec_matrix, axis=0, keepdims=True)
        # SVD to get principal directions
        U_pos, S_pos, Vt_pos = svd(vec_centered, full_matrices=False)
        # Top-5 principal directions
        pos_subspaces[pos_name] = U_pos[:, :5]  # [n_words, 5] — actually we want Vt_pos[:5, :] = [5, d_model]
        pos_subspaces[pos_name] = Vt_pos[:5, :]  # [5, d_model]

    # Compute principal angles between POS subspaces
    pos_pair_angles = {}
    pos_names_with_subspace = list(pos_subspaces.keys())

    for i, pos_a in enumerate(pos_names_with_subspace):
        for j, pos_b in enumerate(pos_names_with_subspace):
            if i >= j:
                continue
            U_a = pos_subspaces[pos_a]  # [5, d_model]
            U_b = pos_subspaces[pos_b]  # [5, d_model]

            # Principal angles via SVD of U_a @ U_b.T
            M = U_a @ U_b.T  # [5, 5]
            _, s, _ = svd(M)
            # s = cos(principal angles)
            min_angle = float(np.arccos(np.clip(s.min(), -1, 1))) * 180 / np.pi
            mean_angle = float(np.arccos(np.clip(np.mean(s), -1, 1))) * 180 / np.pi

            key = f"{pos_a}_vs_{pos_b}"
            pos_pair_angles[key] = {
                "min_principal_angle_deg": round(min_angle, 1),
                "mean_principal_angle_deg": round(mean_angle, 1),
                "mean_cosine_of_angles": round(float(np.mean(s)), 4),
            }

    results["subspace_angles"] = pos_pair_angles

    # ---- Step 8: 高频token的W_U分布 ----
    log_time("\n--- Top-1000 frequent token W_U distribution ---")

    # Use token frequency as proxy (approximate by token ID range)
    # Skip special tokens (first ~10)
    valid_start = 10
    valid_end = min(50000, W_U.shape[0])  # Top 50K tokens

    # Compute W_U norms for all tokens
    wu_norms = np.linalg.norm(W_U[valid_start:valid_end], axis=1)

    # Top-1000 tokens by W_U norm
    top_norm_idx = np.argsort(wu_norms)[-1000:][::-1] + valid_start

    # What POS categories do these tokens belong to?
    top_token_words = [safe_decode(tokenizer, int(idx)) for idx in top_norm_idx]

    # Classify by simple heuristics
    heuristic_counts = {"function": 0, "content": 0, "punctuation": 0, "other": 0}
    function_words_set = set()
    for pos in function_pos:
        function_words_set.update(pos_valid_words.get(pos, []))
    content_words_set = set()
    for pos in content_pos:
        content_words_set.update(pos_valid_words.get(pos, []))

    for w in top_token_words:
        w_lower = w.lower().strip()
        if w_lower in function_words_set:
            heuristic_counts["function"] += 1
        elif w_lower in content_words_set:
            heuristic_counts["content"] += 1
        elif len(w) <= 2 or not w.isalpha():
            heuristic_counts["punctuation"] += 1
        else:
            heuristic_counts["other"] += 1

    results["top1000_norm_tokens"] = {
        "heuristic_classification": heuristic_counts,
        "top20_by_norm": [(safe_decode(tokenizer, int(idx)), round(float(wu_norms[idx - valid_start]), 3))
                          for idx in top_norm_idx[:20]],
    }

    log_time(f"  Top-1000 by W_U norm: {heuristic_counts}")

    save_result(model_name, 3, results)
    release_model_safe(model)
    return results


# ============================================================
# Part 4: Model-Based Co-occurrence Measurement
# ============================================================

def part4_model_cooccurrence(model_name):
    """
    用模型自身的条件概率作为共现代理, 重新验证反义词cosine

    方法:
    1. 对每个词对(A, B), 测量模型在包含A的上下文中生成B的概率
    2. 回归分析: cosine ~ cooccurrence + semantic_type
    3. 看semantic_type在控制cooccurrence后是否仍有显著效应
    """
    import torch
    from model_utils import get_layers

    model, tokenizer, info, n_heads, head_dim = load_model_safe(model_name)
    W_U = get_W_U_safe(model, model_name)  # [vocab_size, d_model]
    input_device = get_input_device(model)

    results = {"model": model_name}

    # ---- Step 1: 定义词对和模板 ----
    # 所有词对 (word_A, word_B, semantic_type)
    word_pairs = []

    # 反义词
    antonym_pairs = [
        ("hot", "cold"), ("big", "small"), ("fast", "slow"), ("light", "dark"),
        ("good", "bad"), ("love", "hate"), ("rich", "poor"), ("strong", "weak"),
        ("happy", "sad"), ("beautiful", "ugly"), ("young", "old"), ("full", "empty"),
        ("loud", "quiet"), ("hard", "soft"), ("sharp", "dull"), ("wet", "dry"),
        ("clean", "dirty"), ("safe", "dangerous"), ("easy", "difficult"),
        ("open", "closed"), ("high", "low"), ("deep", "shallow"), ("wide", "narrow"),
        ("bright", "dim"), ("sweet", "bitter"),
        # More pairs for statistical power
        ("alive", "dead"), ("ancient", "modern"), ("attack", "defend"),
        ("begin", "end"), ("borrow", "lend"), ("buy", "sell"),
        ("create", "destroy"), ("enter", "exit"), ("freeze", "melt"),
        ("include", "exclude"), ("remember", "forget"), ("win", "lose"),
    ]
    for wA, wB in antonym_pairs:
        word_pairs.append((wA, wB, "antonym"))

    # 近义词
    synonym_pairs = [
        ("big", "large"), ("small", "tiny"), ("fast", "quick"), ("smart", "clever"),
        ("happy", "joyful"), ("sad", "unhappy"), ("beautiful", "pretty"),
        ("strong", "powerful"), ("begin", "start"), ("end", "finish"),
        ("help", "assist"), ("walk", "stroll"), ("talk", "speak"),
        ("look", "see"), ("make", "create"), ("think", "consider"),
        ("move", "travel"), ("change", "transform"), ("build", "construct"),
        ("find", "discover"), ("show", "display"), ("choose", "select"),
        ("stop", "halt"), ("grow", "expand"), ("fix", "repair"),
    ]
    for wA, wB in synonym_pairs:
        word_pairs.append((wA, wB, "synonym"))

    # 无关词
    unrelated_pairs = [
        ("apple", "car"), ("table", "sky"), ("river", "book"),
        ("dance", "metal"), ("cloud", "shoe"), ("sleep", "hammer"),
        ("ocean", "pencil"), ("garden", "rocket"), ("music", "brick"),
        ("color", "engine"), ("bread", "diamond"), ("chair", "storm"),
        ("flower", "keyboard"), ("mirror", "forest"), ("coffee", "bridge"),
        ("paint", "gravity"), ("shirt", "volcano"), ("candle", "bicycle"),
        ("paper", "thunder"), ("window", "cake"), ("silver", "mango"),
        ("stone", "violin"), ("blanket", "sword"), ("ticket", "pear"),
    ]
    for wA, wB in unrelated_pairs:
        word_pairs.append((wA, wB, "unrelated"))

    log_time(f"Total word pairs: {len(word_pairs)} "
             f"(antonym={len(antonym_pairs)}, synonym={len(synonym_pairs)}, unrelated={len(unrelated_pairs)})")

    # ---- Step 2: 计算W_U cosine ----
    log_time("Computing W_U cosines...")

    pair_data = []
    for wA, wB, sem_type in word_pairs:
        ids_A = tokenizer.encode(wA, add_special_tokens=False)
        ids_B = tokenizer.encode(wB, add_special_tokens=False)
        if not ids_A or not ids_B:
            continue
        id_A, id_B = ids_A[0], ids_B[0]
        if id_A >= W_U.shape[0] or id_B >= W_U.shape[0]:
            continue

        cos = cosine_sim(W_U[id_A], W_U[id_B])
        pair_data.append({
            "word_A": wA, "word_B": wB, "semantic_type": sem_type,
            "token_A": id_A, "token_B": id_B,
            "wu_cosine": round(cos, 4),
        })

    log_time(f"Valid pairs with W_U cosine: {len(pair_data)}")

    # ---- Step 3: 用模型测量共现概率 ----
    log_time("Measuring model-based co-occurrence probability...")

    templates = [
        "The words {A} and {B}",
        "Between {A} and {B}",
        "From {A} to {B}",
    ]

    for pi, pd in enumerate(pair_data):
        wA, wB = pd["word_A"], pd["word_B"]
        id_B = pd["token_B"]

        cooc_probs = []
        for template in templates:
            prompt = template.replace("{A}", wA).replace("{B}", wB)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            with torch.no_grad():
                try:
                    out = model(input_ids=input_ids, attention_mask=attn_mask)
                    logits = out.logits[0, -1].float().cpu().numpy()
                    # P(B | context) via softmax
                    probs = np.exp(logits - np.max(logits))
                    probs = probs / probs.sum()
                    cooc_probs.append(float(probs[id_B]))
                except:
                    cooc_probs.append(0.0)

        pd["model_cooc_prob_mean"] = round(float(np.mean(cooc_probs)), 6)
        pd["model_cooc_prob_max"] = round(float(np.max(cooc_probs)), 6)

        # Also measure reverse: P(A | context_with_B)
        cooc_probs_rev = []
        for template in templates:
            prompt = template.replace("{A}", wB).replace("{B}", wA)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            id_A = pd["token_A"]
            with torch.no_grad():
                try:
                    out = model(input_ids=input_ids, attention_mask=attn_mask)
                    logits = out.logits[0, -1].float().cpu().numpy()
                    probs = np.exp(logits - np.max(logits))
                    probs = probs / probs.sum()
                    cooc_probs_rev.append(float(probs[id_A]))
                except:
                    cooc_probs_rev.append(0.0)

        pd["model_cooc_prob_rev_mean"] = round(float(np.mean(cooc_probs_rev)), 6)

        # Symmetric co-occurrence
        pd["model_cooc_symmetric"] = round((pd["model_cooc_prob_mean"] + pd["model_cooc_prob_rev_mean"]) / 2, 6)

        if (pi + 1) % 20 == 0:
            log_time(f"  Processed {pi+1}/{len(pair_data)} pairs")

    # ---- Step 4: 回归分析 ----
    log_time("\n--- Regression analysis ---")

    # Prepare data for regression
    from numpy.linalg import lstsq

    Y = np.array([pd["wu_cosine"] for pd in pair_data])
    cooc = np.array([pd["model_cooc_symmetric"] for pd in pair_data])

    # Binary indicators for semantic type
    is_antonym = np.array([1.0 if pd["semantic_type"] == "antonym" else 0.0 for pd in pair_data])
    is_synonym = np.array([1.0 if pd["semantic_type"] == "synonym" else 0.0 for pd in pair_data])
    is_unrelated = np.array([1.0 if pd["semantic_type"] == "unrelated" else 0.0 for pd in pair_data])

    # Model 1: cosine ~ cooccurrence only
    X1 = np.column_stack([np.ones(len(Y)), cooc])
    beta1, _, _, _ = lstsq(X1, Y, rcond=None)
    Y_pred1 = X1 @ beta1
    ss_res1 = np.sum((Y - Y_pred1) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r2_model1 = 1 - ss_res1 / ss_tot if ss_tot > 0 else 0

    # Model 2: cosine ~ cooccurrence + antonym + synonym
    X2 = np.column_stack([np.ones(len(Y)), cooc, is_antonym, is_synonym])
    beta2, _, _, _ = lstsq(X2, Y, rcond=None)
    Y_pred2 = X2 @ beta2
    ss_res2 = np.sum((Y - Y_pred2) ** 2)
    r2_model2 = 1 - ss_res2 / ss_tot if ss_tot > 0 else 0

    # Partial F-test: does adding semantic type improve the model?
    n = len(Y)
    p1 = X1.shape[1]
    p2 = X2.shape[1]
    F_stat = ((ss_res1 - ss_res2) / (p2 - p1)) / (ss_res2 / (n - p2)) if (ss_res2 > 0 and n > p2) else 0

    regression_results = {
        "model1_cooc_only": {
            "r_squared": round(float(r2_model1), 4),
            "intercept": round(float(beta1[0]), 4),
            "cooc_coefficient": round(float(beta1[1]), 4),
        },
        "model2_cooc_plus_semantic": {
            "r_squared": round(float(r2_model2), 4),
            "intercept": round(float(beta2[0]), 4),
            "cooc_coefficient": round(float(beta2[1]), 4),
            "antonym_coefficient": round(float(beta2[2]), 4),
            "synonym_coefficient": round(float(beta2[3]), 4),
            "partial_F_stat": round(float(F_stat), 4),
        },
        "r_squared_improvement": round(float(r2_model2 - r2_model1), 4),
    }

    results["regression"] = regression_results
    results["pair_data"] = pair_data

    log_time(f"  Model 1 (cooc only): R²={r2_model1:.4f}, cooc_coef={beta1[1]:.4f}")
    log_time(f"  Model 2 (cooc + semantic): R²={r2_model2:.4f}, cooc={beta2[1]:.4f}, "
             f"antonym={beta2[2]:.4f}, synonym={beta2[3]:.4f}")
    log_time(f"  R² improvement: {r2_model2 - r2_model1:.4f}")
    log_time(f"  Partial F-statistic: {F_stat:.4f}")

    # ---- Step 5: 按语义类型的分组统计 ----
    log_time("\n--- Group statistics ---")

    for sem_type in ["antonym", "synonym", "unrelated"]:
        group = [pd for pd in pair_data if pd["semantic_type"] == sem_type]
        if not group:
            continue
        mean_cos = np.mean([pd["wu_cosine"] for pd in group])
        mean_cooc = np.mean([pd["model_cooc_symmetric"] for pd in group])
        log_time(f"  {sem_type}: n={len(group)}, mean_cosine={mean_cos:.4f}, mean_cooc={mean_cooc:.6f}")

    save_result(model_name, 4, results)
    release_model_safe(model)
    return results


# ============================================================
# Main
# ============================================================

PART_FUNCTIONS = {
    1: part1_grammar_head_identification,
    2: part2_grammar_ov_decoding,
    3: part3_wu_pos_geometry,
    4: part4_model_cooccurrence,
}

def main():
    parser = argparse.ArgumentParser(description="Phase 257: Grammar Head + W_U Geometry")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"],
                       help="Model to test")
    parser.add_argument("--part", type=str, required=True,
                       help="Part number (1-4) or 'all'")
    args = parser.parse_args()

    model_name = args.model

    if args.part == "all":
        parts = [1, 2, 3, 4]
    else:
        parts = [int(args.part)]

    log_time(f"Phase 257: Grammar Head + W_U Geometry")
    log_time(f"Model: {model_name}, Parts: {parts}")
    log_time(f"=" * 60)

    for part_num in parts:
        if part_num not in PART_FUNCTIONS:
            log_time(f"Unknown part: {part_num}, skipping")
            continue

        log_time(f"\n{'#' * 60}")
        log_time(f"# Starting Part {part_num}")
        log_time(f"{'#' * 60}")

        try:
            result = PART_FUNCTIONS[part_num](model_name)
            log_time(f"Part {part_num} completed successfully!")
        except Exception as e:
            log_time(f"Part {part_num} FAILED: {e}")
            import traceback
            traceback.print_exc()

        gc.collect()
        import torch
        torch.cuda.empty_cache()
        time.sleep(2)

    log_time(f"\nPhase 257 completed for {model_name}!")

if __name__ == "__main__":
    main()
