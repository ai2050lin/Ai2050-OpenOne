"""
Phase 261: Attention Disruption Mechanism & Multi-Feature Probing
================================================================

Based on Phase 260 deep review, 4 experiment parts:

  Part 1: L0 Attention disruption analysis
    - Dimension-level analysis of number information disruption by attention
    - Direction cosine between embed/attn/mlp number directions
    - Per-head attention pattern analysis (number sensitivity)
    - Separation analysis on number direction

  Part 2: Number direction cosine analysis (all 3 models)
    - Compare probe direction at: embedding, post-L0-attn, post-L0-MLP
    - Distinguish denoising vs recoding

  Part 3: First-token multi-feature probing (all 3 models)
    - Probe first token position for: number, tense, sentence_length, subject_type

  Part 4: L35_H0 compensation mechanism (Qwen3 primarily)
    - Compute per-head logit contribution at verb position
    - Find heads that compensate L35_H0's -3.0 suppression

Usage:
  python tests/glm5/phase261_attention_disruption_mechanism.py --model glm4 --part 1
  python tests/glm5/phase261_attention_disruption_mechanism.py --model glm4 --part 2
  python tests/glm5/phase261_attention_disruption_mechanism.py --model qwen3 --part 3
  python tests/glm5/phase261_attention_disruption_mechanism.py --model qwen3 --part 4
"""

import sys, os, json, argparse, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = Path("results/phase261_disruption_mechanism")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def get_input_device(model):
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_special_token_offset(tokenizer, prompt_text):
    no_special = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    with_special = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
    return with_special - no_special

def safe_decode(tokenizer, token_id):
    try:
        return tokenizer.decode([token_id]).strip()
    except Exception:
        return f"<id:{token_id}>"

def safe_get_token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None

SINGULAR_SUBJECTS = [
    "cat", "dog", "bird", "fish", "child", "woman", "man", "person",
    "teacher", "doctor", "student", "writer", "artist", "driver", "worker",
    "tree", "flower", "river", "mountain", "book", "car", "house", "door",
    "girl", "boy", "king", "queen", "hero", "friend", "mother", "father",
    "sister", "brother", "teacher", "scientist", "engineer", "lawyer",
    "apple", "orange", "banana", "grape", "peach", "cherry", "lemon",
    "horse", "sheep", "goose", "mouse", "tooth", "foot", "ox",
    "knife", "life", "leaf", "wolf", "calf", "half", "loaf",
    "city", "country", "village", "story", "party", "army", "baby",
    "duty", "journey", "valley", "lady", "body", "day", "way"
]

PLURAL_SUBJECTS = [
    "cats", "dogs", "birds", "fish", "children", "women", "men", "people",
    "teachers", "doctors", "students", "writers", "artists", "drivers", "workers",
    "trees", "flowers", "rivers", "mountains", "books", "cars", "houses", "doors",
    "girls", "boys", "kings", "queens", "heroes", "friends", "mothers", "fathers",
    "sisters", "brothers", "teachers", "scientists", "engineers", "lawyers",
    "apples", "oranges", "bananas", "grapes", "peaches", "cherries", "lemons",
    "horses", "sheep", "geese", "mice", "teeth", "feet", "oxen",
    "knives", "lives", "leaves", "wolves", "calves", "halves", "loaves",
    "cities", "countries", "villages", "stories", "parties", "armies", "babies",
    "duties", "journeys", "valleys", "ladies", "bodies", "days", "ways"
]

def load_model_safe(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS, get_model_info, get_layers

    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} from {cfg['path']}...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
            log_time(f"  {attn_impl} failed: {str(e)[:80]}, trying next...")
            continue

    model.eval()
    info = get_model_info(model, model_name)
    log_time(f"{model_name}: class={info.model_class}, layers={info.n_layers}, "
             f"d_model={info.d_model}, vocab={info.vocab_size}")

    config = model.config
    n_heads = getattr(config, 'num_attention_heads', 32)
    head_dim = getattr(config, 'head_dim', info.d_model // n_heads)
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
    kv_group_size = n_heads // n_kv_heads

    log_time(f"  n_heads={n_heads}, head_dim={head_dim}, n_kv_heads={n_kv_heads}, kv_group_size={kv_group_size}")
    return model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size

def get_W_U_safe(model, model_name):
    from model_utils import get_W_U
    return get_W_U(model, model_name)

def train_probe(data_dict):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    sing_data = data_dict["sing"]
    plur_data = data_dict["plur"]
    if len(sing_data) < 5 or len(plur_data) < 5:
        return None, None
    X = np.array(sing_data + plur_data)
    y = np.array([1] * len(sing_data) + [0] * len(plur_data))
    probe = LogisticRegression(max_iter=2000, C=1.0)
    cv = min(5, min(len(sing_data), len(plur_data)))
    scores = cross_val_score(probe, X, y, cv=cv)
    probe.fit(X, y)
    direction = probe.coef_[0]
    direction = direction / (np.linalg.norm(direction) + 1e-10)
    return round(float(np.mean(scores)), 4), direction


# ============================================================
# Part 1: L0 Attention Disruption Analysis
# ============================================================

def run_part1(model_name):
    """
    Analyze L0 attention's effect on number information at subject position.
    Dimension-level analysis + per-head attention pattern analysis.
    """
    import torch
    from model_utils import get_layers

    log_time(f"=== Part 1: L0 Attention Disruption Analysis for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_safe(model_name)
    input_device = get_input_device(model)
    layers = get_layers(model)

    n_prompts = 80
    sing_prompts = [f"The {SINGULAR_SUBJECTS[i]} sits" for i in range(min(n_prompts, len(SINGULAR_SUBJECTS)))]
    plur_prompts = [f"The {PLURAL_SUBJECTS[i]} sit" for i in range(min(n_prompts, len(PLURAL_SUBJECTS)))]
    all_prompts = sing_prompts + plur_prompts
    labels_list = ["sing"] * len(sing_prompts) + ["plur"] * len(plur_prompts)

    log_time(f"Total prompts: {len(all_prompts)}")

    embed_data = {"sing": [], "plur": []}
    attn_data = {"sing": [], "plur": []}
    mlp_data = {"sing": [], "plur": []}
    head_attn_stats = defaultdict(lambda: {"sing": [], "plur": []})

    for pi, (prompt, label) in enumerate(zip(all_prompts, labels_list)):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        offset = get_special_token_offset(tokenizer, prompt)
        subj_pos = 1 + offset

        captured = {}
        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    if len(output) >= 2 and output[1] is not None:
                        captured[key + "_attn"] = output[1].detach().float().cpu()
                    captured[key + "_out"] = output[0].detach().float().cpu()
                else:
                    captured[key + "_out"] = output.detach().float().cpu()
            return hook

        hooks = []
        hooks.append(layers[0].self_attn.register_forward_hook(make_hook("L0_attn")))
        hooks.append(layers[0].register_forward_hook(make_hook("L0_full")))

        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_hidden_states=True)
        except Exception:
            for h in hooks: h.remove()
            continue
        for h in hooks: h.remove()

        if out.hidden_states and len(out.hidden_states) > 0:
            embed_vec = out.hidden_states[0][0, subj_pos, :].float().cpu().numpy()
            embed_data[label].append(embed_vec)
        if "L0_attn_out" in captured:
            attn_vec = captured["L0_attn_out"][0, subj_pos, :].float().cpu().numpy()
            attn_data[label].append(attn_vec)
        if "L0_full_out" in captured:
            mlp_vec = captured["L0_full_out"][0, subj_pos, :].float().cpu().numpy()
            mlp_data[label].append(mlp_vec)

        if "L0_attn_attn" in captured:
            attn_w = captured["L0_attn_attn"]
            for h in range(min(n_heads, attn_w.shape[1])):
                subj_attn = attn_w[0, h, subj_pos, :].numpy()
                entropy = float(-np.sum(subj_attn * np.log(subj_attn + 1e-10)))
                n_pos = len(subj_attn)
                head_attn_stats[h][label].append({
                    "entropy": entropy,
                    "attn_to_first": float(subj_attn[0]) if n_pos > 0 else 0,
                    "attn_to_self": float(subj_attn[offset + 1]) if n_pos > offset + 1 else 0,
                    "attn_to_verb": float(subj_attn[-1]) if n_pos > 0 else 0,
                    "max_attn_pos": int(np.argmax(subj_attn)),
                    "max_attn_weight": float(np.max(subj_attn)),
                })

        if pi % 20 == 0:
            log_time(f"  prompt {pi+1}/{len(all_prompts)}")

    # Baseline probes
    embed_acc, embed_dir = train_probe(embed_data)
    attn_acc, attn_dir = train_probe(attn_data)
    mlp_acc, mlp_dir = train_probe(mlp_data)

    log_time(f"Probe accuracies: embed={embed_acc}, attn={attn_acc}, mlp={mlp_acc}")

    # Direction cosine
    cos_ea = float(np.dot(embed_dir, attn_dir)) if embed_dir is not None and attn_dir is not None else None
    cos_em = float(np.dot(embed_dir, mlp_dir)) if embed_dir is not None and mlp_dir is not None else None
    cos_am = float(np.dot(attn_dir, mlp_dir)) if attn_dir is not None and mlp_dir is not None else None

    log_time(f"Direction cosine: embed-vs-attn={cos_ea}, embed-vs-mlp={cos_em}, attn-vs-mlp={cos_am}")

    # Separation on number direction
    embed_diff = np.mean(embed_data["sing"], axis=0) - np.mean(embed_data["plur"], axis=0)
    
    def compute_separation(data_sing, data_plur, direction):
        proj_s = np.dot(data_sing, direction) / (np.linalg.norm(direction) + 1e-10)
        proj_p = np.dot(data_plur, direction) / (np.linalg.norm(direction) + 1e-10)
        mean_diff = np.mean(proj_s) - np.mean(proj_p)
        pooled_std = np.sqrt((np.var(proj_s) + np.var(proj_p)) / 2 + 1e-10)
        return float(mean_diff / pooled_std)

    sep_embed = compute_separation(np.array(embed_data["sing"]), np.array(embed_data["plur"]), embed_diff)
    sep_attn = compute_separation(np.array(attn_data["sing"]), np.array(attn_data["plur"]), embed_diff)
    sep_mlp = compute_separation(np.array(mlp_data["sing"]), np.array(mlp_data["plur"]), embed_diff)

    log_time(f"Separation on embed direction: embed={sep_embed:.3f}, attn={sep_attn:.3f}, mlp={sep_mlp:.3f}")
    log_time(f"  attn change: {sep_attn - sep_embed:+.3f}, mlp change: {sep_mlp - sep_attn:+.3f}")

    # Per-head attention summary
    head_summary = {}
    for h in range(n_heads):
        sing_s = head_attn_stats[h]["sing"]
        plur_s = head_attn_stats[h]["plur"]
        if not sing_s or not plur_s:
            continue
        s = {
            "entropy_sing": round(float(np.mean([x["entropy"] for x in sing_s])), 4),
            "entropy_plur": round(float(np.mean([x["entropy"] for x in plur_s])), 4),
            "attn_first_sing": round(float(np.mean([x["attn_to_first"] for x in sing_s])), 4),
            "attn_first_plur": round(float(np.mean([x["attn_to_first"] for x in plur_s])), 4),
            "attn_self_sing": round(float(np.mean([x["attn_to_self"] for x in sing_s])), 4),
            "attn_self_plur": round(float(np.mean([x["attn_to_self"] for x in plur_s])), 4),
            "attn_verb_sing": round(float(np.mean([x["attn_to_verb"] for x in sing_s])), 4),
            "attn_verb_plur": round(float(np.mean([x["attn_to_verb"] for x in plur_s])), 4),
        }
        s["entropy_diff"] = round(abs(s["entropy_sing"] - s["entropy_plur"]), 4)
        s["number_sensitivity"] = round(s["entropy_diff"] + 
            abs(s["attn_first_sing"] - s["attn_first_plur"]) +
            abs(s["attn_verb_sing"] - s["attn_verb_plur"]), 4)
        head_summary[h] = s

    sorted_by_sens = sorted(head_summary.items(), key=lambda x: -x[1]["number_sensitivity"])
    sorted_by_self = sorted(head_summary.items(), key=lambda x: -(x[1]["attn_self_sing"] + x[1]["attn_self_plur"]) / 2)

    log_time("Top 10 heads by number sensitivity:")
    for h, s in sorted_by_sens[:10]:
        log_time(f"  H{h}: sens={s['number_sensitivity']}, entropy_diff={s['entropy_diff']}, "
                 f"self_sing={s['attn_self_sing']}, self_plur={s['attn_self_plur']}")

    log_time("Top 10 heads by self-attention (subject-locating):")
    for h, s in sorted_by_self[:10]:
        log_time(f"  H{h}: self_sing={s['attn_self_sing']}, self_plur={s['attn_self_plur']}, "
                 f"first_sing={s['attn_first_sing']}, verb_sing={s['attn_verb_sing']}")

    # Per-dimension disruption
    dim_disruption = []
    for d in range(len(embed_diff)):
        embed_s = np.mean([v[d] for v in embed_data["sing"]]) - np.mean([v[d] for v in embed_data["plur"]])
        attn_s = np.mean([v[d] for v in attn_data["sing"]]) - np.mean([v[d] for v in attn_data["plur"]])
        mlp_s = np.mean([v[d] for v in mlp_data["sing"]]) - np.mean([v[d] for v in mlp_data["plur"]])
        disruption = abs(embed_s) - abs(attn_s)
        recovery = abs(mlp_s) - abs(attn_s)
        dim_disruption.append({
            "dim": d, "embed_sep": round(float(embed_s), 4),
            "attn_sep": round(float(attn_s), 4), "mlp_sep": round(float(mlp_s), 4),
            "disruption": round(float(disruption), 4), "recovery": round(float(recovery), 4),
        })

    dim_disruption.sort(key=lambda x: -x["disruption"])
    top_disrupted = dim_disruption[:20]
    dim_disruption.sort(key=lambda x: -x["recovery"])
    top_recovered = dim_disruption[:20]

    log_time("Top 5 disrupted dimensions:")
    for d in top_disrupted[:5]:
        log_time(f"  Dim {d['dim']}: embed={d['embed_sep']}, attn={d['attn_sep']}, mlp={d['mlp_sep']}, disruption={d['disruption']}")

    results = {
        "model": model_name,
        "n_prompts": n_prompts,
        "probe_accuracy": {"embedding": embed_acc, "L0_attn_out": attn_acc, "L0_mlp_out": mlp_acc},
        "direction_cosine": {"embed_vs_attn": round(cos_ea, 4) if cos_ea else None,
                             "embed_vs_mlp": round(cos_em, 4) if cos_em else None,
                             "attn_vs_mlp": round(cos_am, 4) if cos_am else None},
        "number_separation": {"embedding": round(sep_embed, 4), "L0_attn_out": round(sep_attn, 4),
                              "L0_mlp_out": round(sep_mlp, 4), "attn_change": round(sep_attn - sep_embed, 4),
                              "mlp_change": round(sep_mlp - sep_attn, 4)},
        "top_disrupted_dims": top_disrupted[:20],
        "top_recovered_dims": top_recovered[:20],
        "head_attention_summary": {str(k): v for k, v in head_summary.items()},
        "top_number_sensitive_heads": [(h, s) for h, s in sorted_by_sens[:10]],
        "top_subject_locating_heads": [(h, s) for h, s in sorted_by_self[:10]],
    }

    out_path = RESULT_DIR / f"{model_name}_part1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 1 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Part 2: Number Direction Cosine Analysis (all 3 models)
# ============================================================

def run_part2(model_name):
    """
    Compare probe direction at embedding, post-L0-attn, post-L0-MLP.
    Distinguish denoising vs recoding.
    """
    import torch
    from model_utils import get_layers

    log_time(f"=== Part 2: Number Direction Cosine for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_safe(model_name)
    input_device = get_input_device(model)
    layers = get_layers(model)

    n_prompts = 80
    sing_prompts = [f"The {SINGULAR_SUBJECTS[i]} sits" for i in range(min(n_prompts, len(SINGULAR_SUBJECTS)))]
    plur_prompts = [f"The {PLURAL_SUBJECTS[i]} sit" for i in range(min(n_prompts, len(PLURAL_SUBJECTS)))]
    all_prompts = sing_prompts + plur_prompts
    labels_list = ["sing"] * len(sing_prompts) + ["plur"] * len(plur_prompts)

    # Collect states at multiple layers
    mid_layer = info.n_layers // 2
    late_layer = info.n_layers - 5
    state_names = ["embed", "L0_attn_out", "L0_mlp_out", "L1_attn_out", "L1_mlp_out",
                   f"L{mid_layer}", f"L{late_layer}", f"L{info.n_layers}"]
    states = {name: {"sing": [], "plur": []} for name in state_names}

    for pi, (prompt, label) in enumerate(zip(all_prompts, labels_list)):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        offset = get_special_token_offset(tokenizer, prompt)
        subj_pos = 1 + offset

        captured = {}
        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    if len(output) >= 2 and output[1] is not None:
                        captured[key + "_attn"] = output[1].detach().float().cpu()
                    captured[key + "_out"] = output[0].detach().float().cpu()
                else:
                    captured[key + "_out"] = output.detach().float().cpu()
            return hook

        hooks = []
        hooks.append(layers[0].self_attn.register_forward_hook(make_hook("L0_attn")))
        hooks.append(layers[0].register_forward_hook(make_hook("L0_full")))
        hooks.append(layers[1].self_attn.register_forward_hook(make_hook("L1_attn")))
        hooks.append(layers[1].register_forward_hook(make_hook("L1_full")))

        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        except Exception:
            for h in hooks: h.remove()
            continue
        for h in hooks: h.remove()

        if out.hidden_states:
            if len(out.hidden_states) > 0:
                states["embed"][label].append(out.hidden_states[0][0, subj_pos, :].float().cpu().numpy())
            if f"L{mid_layer}" in state_names and mid_layer < len(out.hidden_states):
                states[f"L{mid_layer}"][label].append(out.hidden_states[mid_layer][0, subj_pos, :].float().cpu().numpy())
            if f"L{late_layer}" in state_names and late_layer < len(out.hidden_states):
                states[f"L{late_layer}"][label].append(out.hidden_states[late_layer][0, subj_pos, :].float().cpu().numpy())
            if info.n_layers < len(out.hidden_states):
                states[f"L{info.n_layers}"][label].append(out.hidden_states[info.n_layers][0, subj_pos, :].float().cpu().numpy())

        for cap_name, state_name in [("L0_attn_out", "L0_attn_out"), ("L0_full_out", "L0_mlp_out"),
                                      ("L1_attn_out", "L1_attn_out"), ("L1_full_out", "L1_mlp_out")]:
            if cap_name in captured:
                states[state_name][label].append(captured[cap_name][0, subj_pos, :].float().cpu().numpy())

        if pi % 20 == 0:
            log_time(f"  prompt {pi+1}/{len(all_prompts)}")

    # Train probes and extract directions
    directions = {}
    probe_accs = {}
    for name in state_names:
        acc, direction = train_probe(states[name])
        if acc is not None:
            directions[name] = direction
            probe_accs[name] = acc

    log_time("Probe accuracies:")
    for name in state_names:
        if name in probe_accs:
            log_time(f"  {name}: {probe_accs[name]}")

    # Pairwise direction cosine
    cos_matrix = {}
    state_list = list(directions.keys())
    for i, s1 in enumerate(state_list):
        for j, s2 in enumerate(state_list):
            if j <= i:
                continue
            cos = float(np.dot(directions[s1], directions[s2]))
            cos_matrix[f"{s1}_vs_{s2}"] = round(cos, 4)

    # Key comparisons
    key_pairs = ["embed_vs_L0_attn_out", "embed_vs_L0_mlp_out", "L0_attn_out_vs_L0_mlp_out",
                 "embed_vs_L1_attn_out", "embed_vs_L1_mlp_out",
                 "L0_mlp_out_vs_L1_mlp_out", "embed_vs_L" + str(info.n_layers)]
    
    log_time("Key direction cosines:")
    for pair in key_pairs:
        if pair in cos_matrix:
            log_time(f"  {pair}: {cos_matrix[pair]}")

    results = {
        "model": model_name,
        "probe_accuracies": probe_accs,
        "direction_cosine_matrix": cos_matrix,
    }

    out_path = RESULT_DIR / f"{model_name}_part2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 2 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Part 3: First-Token Multi-Feature Probing
# ============================================================

def run_part3(model_name):
    """
    Probe first token position for multiple features.
    """
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    log_time(f"=== Part 3: First-Token Multi-Feature Probing for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_safe(model_name)
    input_device = get_input_device(model)

    # Feature 1: Number
    number_prompts = []
    for s, p in list(zip(SINGULAR_SUBJECTS[:40], PLURAL_SUBJECTS[:40])):
        number_prompts.append((f"The {s} runs", "sing"))
        number_prompts.append((f"The {p} run", "plur"))

    # Feature 2: Tense
    tense_prompts = []
    for s in SINGULAR_SUBJECTS[:40]:
        tense_prompts.append((f"The {s} runs", "present"))
        tense_prompts.append((f"The {s} ran", "past"))

    # Feature 3: Sentence length
    length_prompts = []
    for s in SINGULAR_SUBJECTS[:40]:
        length_prompts.append((f"The {s} runs", "short"))
        length_prompts.append((f"The {s} that the teacher saw runs", "long"))

    # Feature 4: Subject type (animate/inanimate)
    animate = ["cat", "dog", "child", "woman", "man", "teacher", "doctor",
               "student", "girl", "boy", "king", "queen", "hero", "friend",
               "mother", "father", "sister", "brother", "horse", "sheep"]
    inanimate = ["tree", "flower", "river", "mountain", "book", "car",
                 "house", "door", "apple", "orange", "city", "country",
                 "knife", "leaf", "loaf", "baby", "party", "army", "day", "way"]

    subtype_prompts = []
    for s in animate:
        subtype_prompts.append((f"The {s} runs", "animate"))
    for s in inanimate:
        subtype_prompts.append((f"The {s} runs", "inanimate"))

    all_features = {"number": number_prompts, "tense": tense_prompts,
                    "sentence_length": length_prompts, "subject_type": subtype_prompts}

    key_layers = [0, 1, 5, 10, info.n_layers // 2, info.n_layers - 5, info.n_layers]

    results = {"model": model_name, "key_layers": key_layers, "feature_probing": {}}

    for feature_name, prompts in all_features.items():
        log_time(f"Probing {feature_name} ({len(prompts)} prompts)...")

        unique_labels = list(set(l for _, l in prompts))
        if len(unique_labels) != 2:
            continue

        layer_data = defaultdict(lambda: defaultdict(list))

        for pi, (prompt, label) in enumerate(prompts):
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attn_mask = inputs["attention_mask"].to(input_device)

                offset = get_special_token_offset(tokenizer, prompt)
                first_pos = offset

                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)

                if out.hidden_states:
                    for layer_idx in key_layers:
                        if layer_idx < len(out.hidden_states):
                            vec = out.hidden_states[layer_idx][0, first_pos, :].float().cpu().numpy()
                            layer_data[layer_idx][label].append(vec)
            except Exception:
                continue
            if pi % 40 == 0:
                log_time(f"  {feature_name}: prompt {pi+1}/{len(prompts)}")

        feature_acc = {}
        for layer_idx in sorted(layer_data.keys()):
            d0 = layer_data[layer_idx].get(unique_labels[0], [])
            d1 = layer_data[layer_idx].get(unique_labels[1], [])
            if len(d0) < 5 or len(d1) < 5:
                continue
            X = np.array(d0 + d1)
            y = np.array([0] * len(d0) + [1] * len(d1))
            try:
                probe = LogisticRegression(max_iter=2000, C=1.0)
                cv = min(5, min(len(d0), len(d1)))
                scores = cross_val_score(probe, X, y, cv=cv)
                feature_acc[str(layer_idx)] = round(float(np.mean(scores)), 4)
            except Exception:
                pass

        results["feature_probing"][feature_name] = feature_acc
        log_time(f"  {feature_name} first-token probe:")
        for li in sorted(feature_acc.keys(), key=lambda x: int(x)):
            log_time(f"    L{li}: {feature_acc[li]}")

    out_path = RESULT_DIR / f"{model_name}_part3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 3 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Part 4: Verb-Position Per-Head Logit Contribution
# ============================================================

def run_part4(model_name):
    """
    Compute per-head logit contribution at verb position.
    Find heads that compensate L35_H0's -3.0 suppression.
    """
    import torch
    from model_utils import get_layers

    log_time(f"=== Part 4: Per-Head Logit Contribution for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_safe(model_name)
    input_device = get_input_device(model)
    layers = get_layers(model)

    W_U = get_W_U_safe(model, model_name)

    n_prompts = 30
    sing_prompts = [f"The {SINGULAR_SUBJECTS[i]} runs" for i in range(min(n_prompts, len(SINGULAR_SUBJECTS)))]
    plur_prompts = [f"The {PLURAL_SUBJECTS[i]} run" for i in range(min(n_prompts, len(PLURAL_SUBJECTS)))]
    all_prompts = sing_prompts + plur_prompts
    labels_list = ["sing"] * len(sing_prompts) + ["plur"] * len(plur_prompts)

    sing_verbs = ["runs", "walks", "sits", "speaks", "flies", "writes", "falls", "rules",
                  "is", "has", "does", "goes", "comes", "makes", "takes", "gives"]
    plur_verbs = ["run", "walk", "sit", "speak", "fly", "write", "fall", "rule",
                  "are", "have", "do", "go", "come", "make", "take", "give"]

    # Scan layers - try top 2 first, then fall back to lower layers if weights are on meta device
    top_layers = [info.n_layers - 1, info.n_layers - 2, info.n_layers // 2, 0]
    head_contribs = defaultdict(lambda: {"sing": [], "plur": []})

    for li in top_layers:
        log_time(f"  Scanning layer {li}...")

        # Check if weights are accessible
        try:
            w_o = layers[li].self_attn.o_proj.weight
            if hasattr(w_o, 'is_meta') and w_o.is_meta:
                log_time(f"  L{li} weights on meta device, skipping")
                continue
            W_O = w_o.detach().cpu().float().numpy()

            w_v = layers[li].self_attn.v_proj.weight
            if hasattr(w_v, 'is_meta') and w_v.is_meta:
                log_time(f"  L{li} V weights on meta device, skipping")
                continue
            W_V = w_v.detach().cpu().float().numpy()
        except Exception as e:
            log_time(f"  L{li} weight access failed: {str(e)[:60]}")
            continue

        for pi, (prompt, label) in enumerate(zip(all_prompts, labels_list)):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            offset = get_special_token_offset(tokenizer, prompt)
            no_special = tokenizer.encode(prompt, add_special_tokens=False)
            verb_pos = len(no_special) - 1 + offset

            captured = {}
            def make_attn_hook():
                def hook(module, input, output):
                    if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                        captured["attn"] = output[1].detach().float().cpu()
                return hook

            hook_handle = layers[li].self_attn.register_forward_hook(make_attn_hook())

            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
            except Exception:
                hook_handle.remove()
                continue
            hook_handle.remove()

            if out.hidden_states is None or li >= len(out.hidden_states) or "attn" not in captured:
                continue

            resid_pre = out.hidden_states[li][0].float().cpu().numpy()
            attn_w = captured["attn"]

            for hi in range(min(n_heads, attn_w.shape[1])):
                W_O_h = W_O[:, hi * head_dim:(hi + 1) * head_dim]
                kv_h = hi // kv_group_size
                W_V_h = W_V[kv_h * head_dim:(kv_h + 1) * head_dim, :]

                attn_from_verb = attn_w[0, hi, verb_pos, :].numpy()
                V_h_input = W_V_h @ resid_pre.T  # [head_dim, seq]
                z_h = V_h_input @ attn_from_verb  # [head_dim]
                head_contrib = W_O_h @ z_h  # [d_model]
                logit_effect = W_U @ head_contrib  # [vocab]

                sv_eff = [logit_effect[safe_get_token_id(tokenizer, v)]
                          for v in sing_verbs if safe_get_token_id(tokenizer, v) is not None]
                pv_eff = [logit_effect[safe_get_token_id(tokenizer, v)]
                          for v in plur_verbs if safe_get_token_id(tokenizer, v) is not None]

                head_contribs[(li, hi)][label].append({
                    "sing_verb_mean": round(float(np.mean(sv_eff)), 4) if sv_eff else 0,
                    "plur_verb_mean": round(float(np.mean(pv_eff)), 4) if pv_eff else 0,
                    "verb_diff": round(float(np.mean(sv_eff) - np.mean(pv_eff)), 4) if sv_eff and pv_eff else 0,
                })

            if pi % 10 == 0:
                log_time(f"    L{li} prompt {pi+1}/{len(all_prompts)}")

    # Aggregate results
    head_summary = {}
    for (li, hi), data in head_contribs.items():
        sing_diffs = [d["verb_diff"] for d in data.get("sing", [])]
        plur_diffs = [d["verb_diff"] for d in data.get("plur", [])]
        sing_sverb = [d["sing_verb_mean"] for d in data.get("sing", [])]
        plur_sverb = [d["sing_verb_mean"] for d in data.get("plur", [])]
        sing_pverb = [d["plur_verb_mean"] for d in data.get("sing", [])]
        plur_pverb = [d["plur_verb_mean"] for d in data.get("plur", [])]

        head_summary[f"L{li}_H{hi}"] = {
            "verb_diff_sing": round(float(np.mean(sing_diffs)), 4) if sing_diffs else None,
            "verb_diff_plur": round(float(np.mean(plur_diffs)), 4) if plur_diffs else None,
            "sing_verb_effect_sing": round(float(np.mean(sing_sverb)), 4) if sing_sverb else None,
            "sing_verb_effect_plur": round(float(np.mean(plur_sverb)), 4) if plur_sverb else None,
            "plur_verb_effect_sing": round(float(np.mean(sing_pverb)), 4) if sing_pverb else None,
            "plur_verb_effect_plur": round(float(np.mean(plur_pverb)), 4) if plur_pverb else None,
            "mean_verb_diff": round(float(np.mean(sing_diffs + plur_diffs)), 4) if sing_diffs or plur_diffs else None,
        }

    # Find compensating heads (positive verb_diff to compensate L35_H0's negative)
    compensating = {k: v for k, v in head_summary.items()
                    if v.get("mean_verb_diff") is not None and v["mean_verb_diff"] > 0.5}

    # Find grammar heads (heads that differentiate sing/plur in verb_diff)
    grammar_heads = {k: v for k, v in head_summary.items()
                     if v.get("verb_diff_sing") is not None and v.get("verb_diff_plur") is not None
                     and abs(v["verb_diff_sing"] - v["verb_diff_plur"]) > 0.3}

    log_time(f"Total heads analyzed: {len(head_summary)}")
    log_time(f"Compensating heads (mean_verb_diff > 0.5): {len(compensating)}")
    for name in sorted(compensating.keys(), key=lambda k: -compensating[k]["mean_verb_diff"]):
        d = compensating[name]
        log_time(f"  {name}: mean_diff={d['mean_verb_diff']}, sing_diff={d['verb_diff_sing']}, plur_diff={d['verb_diff_plur']}")

    log_time(f"Grammar heads (sing/plur verb_diff difference > 0.3): {len(grammar_heads)}")
    for name in sorted(grammar_heads.keys(), key=lambda k: -abs(grammar_heads[k]["verb_diff_sing"] - grammar_heads[k]["verb_diff_plur"])):
        d = grammar_heads[name]
        log_time(f"  {name}: sing_diff={d['verb_diff_sing']}, plur_diff={d['verb_diff_plur']}, "
                 f"diff={abs(d['verb_diff_sing'] - d['verb_diff_plur']):.4f}")

    results = {
        "model": model_name,
        "head_logit_summary": head_summary,
        "compensating_heads": compensating,
        "grammar_heads": grammar_heads,
    }

    out_path = RESULT_DIR / f"{model_name}_part4.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 4 saved to {out_path}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--part", type=int, required=True, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    if args.part == 1:
        run_part1(args.model)
    elif args.part == 2:
        run_part2(args.model)
    elif args.part == 3:
        run_part3(args.model)
    elif args.part == 4:
        run_part4(args.model)

if __name__ == "__main__":
    main()
