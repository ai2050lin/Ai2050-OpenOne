"""
Phase 324b: Confirmation — Color Type>Value Pattern & Binding Paradox
=====================================================================

Critical confirmations needed:
1. GLM4 color type direction (1.80) >> value direction (0.16) — is this stable?
2. GLM4 binding all negative — is this an alpha artifact?
3. Which attribute types have type>value vs value>type pattern?

Design:
- Test ALL 30 color pairs (not just 8) at GLM4 L3
- Test with multiple alpha values (1.0, 2.0, 3.0)
- Test binding with lower alpha (0.5, 1.0) to check if negative binding is alpha artifact
- Test each attribute type with ALL 30 pairs for slot/type/value

Usage:
  python tests/glm5/phase324b_color_confirm.py qwen3
  python tests/glm5/phase324b_color_confirm.py glm4
  python tests/glm5/phase324b_color_confirm.py deepseek7b
"""
import sys, os, gc, time, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model, get_W_U

RESULT_DIR = Path("results/phase324b_color")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp"); TMP_DIR.mkdir(parents=True, exist_ok=True)
_log_file = None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except:
            pass


# 30 color pairs (same as Phase 324)
COLOR_PAIRS = [
    ("apple", "red"), ("sky", "blue"), ("grass", "green"), ("sun", "yellow"),
    ("snow", "white"), ("night", "black"), ("orange", "orange"), ("grape", "purple"),
    ("rose", "red"), ("ocean", "blue"), ("leaf", "green"), ("gold", "yellow"),
    ("cloud", "white"), ("coal", "black"), ("carrot", "orange"), ("plum", "purple"),
    ("cherry", "red"), ("sapphire", "blue"), ("emerald", "green"), ("lemon", "yellow"),
    ("ivory", "white"), ("raven", "black"), ("marigold", "orange"), ("lavender", "purple"),
    ("strawberry", "red"), ("turquoise", "blue"), ("mint", "green"), ("banana", "yellow"),
    ("pearl", "white"), ("obsidian", "black"),
]

# Additional attribute types with full 30 pairs
TASTE_PAIRS = [
    ("lemon", "sour"), ("honey", "sweet"), ("coffee", "bitter"), ("salt", "salty"),
    ("chili", "spicy"), ("vinegar", "sour"), ("candy", "sweet"), ("dark chocolate", "bitter"),
    ("soy sauce", "salty"), ("pepper", "spicy"), ("grapefruit", "sour"), ("sugar", "sweet"),
    ("espresso", "bitter"), ("seawater", "salty"), ("ginger", "spicy"), ("lime", "sour"),
    ("maple syrup", "sweet"), ("kale", "bitter"), ("pretzel", "salty"), ("wasabi", "spicy"),
    ("tamarind", "sour"), ("caramel", "sweet"), ("coffee bean", "bitter"), ("bacon", "salty"),
    ("jalapeno", "spicy"), ("yogurt", "sour"), ("vanilla", "sweet"), ("olive", "bitter"),
    ("cheese", "salty"), ("cinnamon", "spicy"),
]

TEXTURE_PAIRS = [
    ("silk", "smooth"), ("sandpaper", "rough"), ("pillow", "soft"), ("diamond", "hard"),
    ("glass", "smooth"), ("bark", "rough"), ("cotton", "soft"), ("rock", "hard"),
    ("velvet", "smooth"), ("concrete", "rough"), ("feather", "soft"), ("steel", "hard"),
    ("marble", "smooth"), ("gravel", "rough"), ("wool", "soft"), ("iron", "hard"),
    ("ice", "smooth"), ("brick", "rough"), ("sponge", "soft"), ("bone", "hard"),
    ("porcelain", "smooth"), ("asphalt", "rough"), ("fur", "soft"), ("shell", "hard"),
    ("polish", "smooth"), ("rust", "rough"), ("cashmere", "soft"), ("granite", "hard"),
    ("ceramic", "smooth"), ("sand", "rough"),
]

TEMPLATES = {
    "color": {
        "slot": "The {obj} has some feature",
        "type": "The {obj} has a color",
        "value": "The {obj} is {val}",
        "baseline": "The {obj} is something",
    },
    "taste": {
        "slot": "The {obj} has some feature",
        "type": "The {obj} has a taste",
        "value": "The {obj} tastes {val}",
        "baseline": "The {obj} is something",
    },
    "texture": {
        "slot": "The {obj} has some feature",
        "type": "The {obj} has a texture",
        "value": "The {obj} feels {val}",
        "baseline": "The {obj} is something",
    },
}

WORD_CLUSTERS = {
    "color": ["red", "blue", "green", "yellow", "white", "black", "orange", "purple", "pink", "brown"],
    "taste": ["sweet", "sour", "bitter", "salty", "spicy", "savory", "tangy", "umami"],
    "texture": ["smooth", "rough", "soft", "hard", "sharp", "fluffy", "slick", "bumpy"],
    "object": ["apple", "table", "car", "house", "book", "water", "idea", "music"],
}

# Binding pairs with compatible/incompatible
BINDING_PAIRS = [
    ("apple", "red", "color", ["red", "green", "yellow"], ["blue", "purple", "black"]),
    ("sky", "blue", "color", ["blue", "white"], ["red", "green", "orange"]),
    ("snow", "white", "color", ["white", "blue"], ["red", "green", "orange"]),
    ("lemon", "sour", "taste", ["sour", "bitter"], ["sweet", "spicy"]),
    ("honey", "sweet", "taste", ["sweet", "savory"], ["sour", "bitter"]),
    ("silk", "smooth", "texture", ["smooth", "soft"], ["rough", "hard"]),
    ("sandpaper", "rough", "texture", ["rough", "hard"], ["smooth", "soft"]),
]


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=impl,
            )
            log(f"  Loaded {model_name} with attn_impl={impl}")
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Model: {type(model).__name__}, device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def extract_rep_at_layer(model, tokenizer, device, sentence, target_layer):
    layers_list = get_layers(model)
    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['rep'] = output[0].detach().float().cpu()
        else:
            captured['rep'] = output.detach().float().cpu()
    hook = layers_list[target_layer].register_forward_hook(hook_fn)
    inp = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
    try:
        with torch.no_grad():
            model(**inp)
        return captured['rep'][0, -1].numpy()
    finally:
        hook.remove()


def inject_direction_at_layer(model, tokenizer, device, prompt, direction, layer_idx, alpha):
    layers_list = get_layers(model)
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        d_tensor = torch.tensor(direction, dtype=hidden.dtype, device=hidden.device)
        hidden_modified = hidden.clone()
        hidden_modified[0, -1, :] += (alpha * d_tensor).to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden_modified,) + output[1:]
        return hidden_modified
    hook = layers_list[layer_idx].register_forward_hook(hook_fn)
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    try:
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[0, -1].float().cpu().numpy()
    finally:
        hook.remove()
    return logits


def get_baseline_logits(model, tokenizer, device, prompt):
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp)
    return out.logits[0, -1].float().cpu().numpy()


def get_cluster_token_ids(tokenizer, cluster_words):
    ids = []
    for w in cluster_words:
        tok_ids = tokenizer.encode(w, add_special_tokens=False)
        if tok_ids:
            ids.append((w, tok_ids[0]))
    return ids


def compute_cluster_mean(logits, cluster_ids):
    if not cluster_ids:
        return 0.0
    return float(np.mean([float(logits[tid]) for _, tid in cluster_ids]))


def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase324b_{model_name}.log")

    log(f"=== Phase 324b: Color Type>Value Confirmation for {model_name} ===")

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}")

    if model_name == "glm4":
        opt_layer = 3
    elif model_name == "qwen3":
        opt_layer = 0
    else:
        opt_layer = 6

    results = {}

    # ===================================================================
    # Test 1: Full 30-pair color test at multiple alphas
    # ===================================================================
    log("\n" + "="*60)
    log("Test 1: Full 30-pair color Type vs Value — Multi-Alpha")
    log("="*60)

    color_detail = []
    alphas = [1.0, 2.0, 3.0]
    tmpl = TEMPLATES["color"]

    for alpha in alphas:
        slot_tgts, type_tgts, value_tgts = [], [], []
        slot_clusters, type_clusters, value_clusters = [], [], []

        for pair_idx, (noun, val) in enumerate(COLOR_PAIRS):
            val_ids = tokenizer.encode(val, add_special_tokens=False)
            if not val_ids:
                continue
            tgt_id = val_ids[0]
            target_cluster_ids = get_cluster_token_ids(tokenizer, WORD_CLUSTERS["color"])

            target_prompt = f"The {noun} is"
            baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)
            baseline_logit = float(baseline_logits[tgt_id])
            baseline_cluster = compute_cluster_mean(baseline_logits, target_cluster_ids)

            base_sent = tmpl["baseline"].format(obj=noun, val=val)
            h_baseline = extract_rep_at_layer(model, tokenizer, device, base_sent, opt_layer)

            for level, tgt_list, cluster_list in [
                ("slot", slot_tgts, slot_clusters),
                ("type", type_tgts, type_clusters),
                ("value", value_tgts, value_clusters),
            ]:
                sent = tmpl[level].format(obj=noun, val=val)
                h_level = extract_rep_at_layer(model, tokenizer, device, sent, opt_layer)
                d = h_level - h_baseline
                norm = np.linalg.norm(d)
                if norm < 1e-10:
                    continue
                d_unit = d / norm

                inj_logits = inject_direction_at_layer(
                    model, tokenizer, device, target_prompt, d_unit, opt_layer, alpha)
                tgt_list.append(float(inj_logits[tgt_id] - baseline_logit))
                cluster_list.append(compute_cluster_mean(inj_logits, target_cluster_ids) - baseline_cluster)

            if pair_idx % 10 == 9:
                log(f"    α={alpha} pair {pair_idx+1}/30 done")
            torch.cuda.empty_cache()

        entry = {
            "alpha": alpha,
            "n_pairs": len(slot_tgts),
            "slot": {"tgt_mean": round(float(np.mean(slot_tgts)), 4), "cluster_mean": round(float(np.mean(slot_clusters)), 4),
                     "tgt_negative_rate": round(sum(1 for x in slot_tgts if x < 0)/max(len(slot_tgts),1), 4)},
            "type": {"tgt_mean": round(float(np.mean(type_tgts)), 4), "cluster_mean": round(float(np.mean(type_clusters)), 4),
                     "tgt_negative_rate": round(sum(1 for x in type_tgts if x < 0)/max(len(type_tgts),1), 4)},
            "value": {"tgt_mean": round(float(np.mean(value_tgts)), 4), "cluster_mean": round(float(np.mean(value_clusters)), 4),
                      "tgt_negative_rate": round(sum(1 for x in value_tgts if x < 0)/max(len(value_tgts),1), 4)},
            "type_vs_value_ratio": round(float(np.mean(type_tgts)) / max(float(np.mean(value_tgts)), 0.001), 2),
        }
        color_detail.append(entry)
        log(f"  α={alpha}: slot_tgt={entry['slot']['tgt_mean']:.4f}, "
            f"type_tgt={entry['type']['tgt_mean']:.4f}, value_tgt={entry['value']['tgt_mean']:.4f}, "
            f"type/value={entry['type_vs_value_ratio']}")

    results["color_full"] = color_detail

    # ===================================================================
    # Test 2: Taste & Texture full 30-pair test
    # ===================================================================
    log("\n" + "="*60)
    log("Test 2: Taste & Texture Full 30-pair")
    log("="*60)

    for attr_type, pairs in [("taste", TASTE_PAIRS), ("texture", TEXTURE_PAIRS)]:
        tmpl = TEMPLATES[attr_type]
        alpha = 2.0
        slot_tgts, type_tgts, value_tgts = [], [], []
        slot_clusters, type_clusters, value_clusters = [], [], []

        for pair_idx, (noun, val) in enumerate(pairs):
            val_ids = tokenizer.encode(val, add_special_tokens=False)
            if not val_ids:
                continue
            tgt_id = val_ids[0]
            target_cluster_ids = get_cluster_token_ids(tokenizer, WORD_CLUSTERS[attr_type])

            target_prompt = f"The {noun} is"
            baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)
            baseline_logit = float(baseline_logits[tgt_id])
            baseline_cluster = compute_cluster_mean(baseline_logits, target_cluster_ids)

            base_sent = tmpl["baseline"].format(obj=noun, val=val)
            h_baseline = extract_rep_at_layer(model, tokenizer, device, base_sent, opt_layer)

            for level, tgt_list, cluster_list in [
                ("slot", slot_tgts, slot_clusters),
                ("type", type_tgts, type_clusters),
                ("value", value_tgts, value_clusters),
            ]:
                sent = tmpl[level].format(obj=noun, val=val)
                h_level = extract_rep_at_layer(model, tokenizer, device, sent, opt_layer)
                d = h_level - h_baseline
                norm = np.linalg.norm(d)
                if norm < 1e-10:
                    continue
                d_unit = d / norm

                inj_logits = inject_direction_at_layer(
                    model, tokenizer, device, target_prompt, d_unit, opt_layer, alpha)
                tgt_list.append(float(inj_logits[tgt_id] - baseline_logit))
                cluster_list.append(compute_cluster_mean(inj_logits, target_cluster_ids) - baseline_cluster)

            if pair_idx % 10 == 9:
                log(f"    {attr_type} pair {pair_idx+1}/30 done")
            torch.cuda.empty_cache()

        entry = {
            "attr_type": attr_type,
            "alpha": alpha,
            "n_pairs": len(slot_tgts),
            "slot": {"tgt_mean": round(float(np.mean(slot_tgts)), 4), "cluster_mean": round(float(np.mean(slot_clusters)), 4)},
            "type": {"tgt_mean": round(float(np.mean(type_tgts)), 4), "cluster_mean": round(float(np.mean(type_clusters)), 4)},
            "value": {"tgt_mean": round(float(np.mean(value_tgts)), 4), "cluster_mean": round(float(np.mean(value_clusters)), 4)},
        }
        results[f"{attr_type}_full"] = entry
        log(f"  {attr_type}: slot_tgt={entry['slot']['tgt_mean']:.4f}, "
            f"type_tgt={entry['type']['tgt_mean']:.4f}, value_tgt={entry['value']['tgt_mean']:.4f}")

    # ===================================================================
    # Test 3: Binding at low alpha — is negative binding an alpha artifact?
    # ===================================================================
    log("\n" + "="*60)
    log("Test 3: Binding at Low Alpha (0.5, 1.0)")
    log("="*60)

    binding_results = []

    for alpha in [0.5, 1.0, 2.0]:
        for noun, val, attr_type, compat_words, incompat_words in BINDING_PAIRS:
            val_ids = tokenizer.encode(val, add_special_tokens=False)
            if not val_ids:
                continue
            tgt_id = val_ids[0]

            compat_ids = get_cluster_token_ids(tokenizer, compat_words)
            incompat_ids = get_cluster_token_ids(tokenizer, incompat_words)
            tmpl = TEMPLATES[attr_type]

            target_prompt = f"The {noun} is"
            baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)
            baseline_compat = compute_cluster_mean(baseline_logits, compat_ids)
            baseline_incompat = compute_cluster_mean(baseline_logits, incompat_ids)

            base_sent = tmpl["baseline"].format(obj=noun, val=val)
            h_baseline = extract_rep_at_layer(model, tokenizer, device, base_sent, opt_layer)

            for level in ["slot", "type", "value"]:
                sent = tmpl[level].format(obj=noun, val=val)
                h_level = extract_rep_at_layer(model, tokenizer, device, sent, opt_layer)
                d = h_level - h_baseline
                norm = np.linalg.norm(d)
                if norm < 1e-10:
                    continue
                d_unit = d / norm

                inj_logits = inject_direction_at_layer(
                    model, tokenizer, device, target_prompt, d_unit, opt_layer, alpha)

                compat_delta = compute_cluster_mean(inj_logits, compat_ids) - baseline_compat
                incompat_delta = compute_cluster_mean(inj_logits, incompat_ids) - baseline_incompat
                binding_score = compat_delta - incompat_delta

                binding_results.append({
                    "alpha": alpha,
                    "pair": f"{noun}→{val}",
                    "attr_type": attr_type,
                    "level": level,
                    "compat_delta": round(compat_delta, 4),
                    "incompat_delta": round(incompat_delta, 4),
                    "binding_score": round(binding_score, 4),
                })

        torch.cuda.empty_cache()

    # Aggregate binding by alpha and level
    log("\n  Binding by alpha and level:")
    alpha_level_binding = defaultdict(list)
    for r in binding_results:
        alpha_level_binding[(r["alpha"], r["level"])].append(r["binding_score"])

    for alpha in [0.5, 1.0, 2.0]:
        for level in ["slot", "type", "value"]:
            scores = alpha_level_binding.get((alpha, level), [])
            if scores:
                log(f"    α={alpha}/{level}: binding={np.mean(scores):.4f}")

    results["binding"] = binding_results

    # ===================================================================
    # Save
    # ===================================================================
    output = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "opt_layer": opt_layer,
        "results": results,
    }

    out_path = RESULT_DIR / f"{model_name}_phase324b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")

    # Summary
    log("\n" + "="*60)
    log(f"PHASE 324b SUMMARY - {model_name}")
    log("="*60)

    log("\n  Color Full 30-pair:")
    for entry in color_detail:
        log(f"    α={entry['alpha']}: type_tgt={entry['type']['tgt_mean']:.4f}, "
            f"value_tgt={entry['value']['tgt_mean']:.4f}, "
            f"type/value={entry['type_vs_value_ratio']}")

    for attr_type in ["taste", "texture"]:
        key = f"{attr_type}_full"
        if key in results:
            r = results[key]
            log(f"\n  {attr_type.capitalize()} Full 30-pair:")
            log(f"    slot_tgt={r['slot']['tgt_mean']:.4f}, "
                f"type_tgt={r['type']['tgt_mean']:.4f}, value_tgt={r['value']['tgt_mean']:.4f}")

    log("\n  Binding at low alpha:")
    for alpha in [0.5, 1.0, 2.0]:
        for level in ["slot", "type", "value"]:
            scores = alpha_level_binding.get((alpha, level), [])
            if scores:
                log(f"    α={alpha}/{level}: binding={np.mean(scores):.4f}")

    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Model {model_name} released. Total time: {time.time()-t0:.1f}s")

    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    if model_name == "all":
        for mn in ["qwen3", "glm4", "deepseek7b"]:
            try:
                run_model(mn)
            except Exception as e:
                log(f"ERROR running {mn}: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(10)
    else:
        run_model(model_name)
