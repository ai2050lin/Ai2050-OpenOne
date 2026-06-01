"""
Phase 325: Attribute Type-Specialized Template Confirmation
============================================================

Core question: Are taste/temperature/shape/size weak because templates don't fit?

Design:
1. Each attribute type gets its OWN natural template set (slot/type/value/baseline)
2. Compare: generic template vs specialized template effect on tgt_delta
3. 6 attribute types × 15 pairs × 2 template sets (generic vs specialized)
4. Cross-template consistency within specialized templates

Key hypothesis: If specialized templates recover taste/temperature/shape/size,
then the weakness was template mismatch, not missing encoding.

Usage:
  python tests/glm5/phase325_specialized_templates.py qwen3
  python tests/glm5/phase325_specialized_templates.py glm4
  python tests/glm5/phase325_specialized_templates.py deepseek7b
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

RESULT_DIR = Path("results/phase325_specialized")
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


# ======================================================================
# Attribute-specific template sets
# ======================================================================
# Each type: 2 slot templates, 2 type templates, 2 value templates, 1 baseline
# "generic" = same template for all types (from Phase 324)
# "specialized" = type-specific natural language templates

TEMPLATE_SETS = {
    "color": {
        "generic": {
            "slot": "The {obj} has some feature",
            "type": "The {obj} has a color",
            "value": "The {obj} is {val}",
            "baseline": "The {obj} is something",
        },
        "specialized": {
            "slot": ["The {obj} has some visual feature", "The {obj} has some appearance"],
            "type": ["The {obj} has a color", "The {obj} has a certain color"],
            "value": ["The {obj} is {val}", "The {obj} looks {val}"],
            "baseline": "The {obj} is something",
        },
    },
    "taste": {
        "generic": {
            "slot": "The {obj} has some feature",
            "type": "The {obj} has a taste",
            "value": "The {obj} is {val}",
            "baseline": "The {obj} is something",
        },
        "specialized": {
            "slot": ["The {obj} has some flavor", "The {obj} has some taste quality"],
            "type": ["The {obj} has a taste", "The {obj} has a flavor"],
            "value": ["The {obj} tastes {val}", "The {obj} has a {val} taste"],
            "baseline": "The {obj} is something",
        },
    },
    "temperature": {
        "generic": {
            "slot": "The {obj} has some feature",
            "type": "The {obj} has a temperature",
            "value": "The {obj} is {val}",
            "baseline": "The {obj} is something",
        },
        "specialized": {
            "slot": ["The {obj} has some thermal quality", "The {obj} has some temperature quality"],
            "type": ["The {obj} has a temperature", "The {obj} feels a certain temperature"],
            "value": ["The {obj} feels {val}", "The {obj} is {val} to the touch"],
            "baseline": "The {obj} is something",
        },
    },
    "texture": {
        "generic": {
            "slot": "The {obj} has some feature",
            "type": "The {obj} has a texture",
            "value": "The {obj} is {val}",
            "baseline": "The {obj} is something",
        },
        "specialized": {
            "slot": ["The {obj} has some surface quality", "The {obj} has some texture quality"],
            "type": ["The {obj} has a texture", "The {obj} has a surface feel"],
            "value": ["The {obj} feels {val}", "The {obj} has a {val} surface"],
            "baseline": "The {obj} is something",
        },
    },
    "shape": {
        "generic": {
            "slot": "The {obj} has some feature",
            "type": "The {obj} has a shape",
            "value": "The {obj} is {val}",
            "baseline": "The {obj} is something",
        },
        "specialized": {
            "slot": ["The {obj} has some geometric form", "The {obj} has some shape quality"],
            "type": ["The {obj} has a shape", "The {obj} has a geometric form"],
            "value": ["The {obj} is {val} in shape", "The {obj} has a {val} shape"],
            "baseline": "The {obj} is something",
        },
    },
    "size": {
        "generic": {
            "slot": "The {obj} has some feature",
            "type": "The {obj} has a size",
            "value": "The {obj} is {val}",
            "baseline": "The {obj} is something",
        },
        "specialized": {
            "slot": ["The {obj} has some size quality", "The {obj} has some dimension"],
            "type": ["The {obj} has a size", "The {obj} has a certain dimension"],
            "value": ["The {obj} is {val} in size", "The {obj} is {val} compared to others"],
            "baseline": "The {obj} is something",
        },
    },
}

# 15 pairs per attribute type (balanced, diverse)
ATTR_PAIRS = {
    "color": [
        ("apple", "red"), ("sky", "blue"), ("grass", "green"), ("snow", "white"),
        ("night", "black"), ("orange", "orange"), ("grape", "purple"), ("rose", "red"),
        ("ocean", "blue"), ("lemon", "yellow"), ("coal", "black"), ("carrot", "orange"),
        ("cherry", "red"), ("emerald", "green"), ("pearl", "white"),
    ],
    "taste": [
        ("lemon", "sour"), ("honey", "sweet"), ("coffee", "bitter"), ("salt", "salty"),
        ("chili", "spicy"), ("vinegar", "sour"), ("candy", "sweet"), ("dark chocolate", "bitter"),
        ("soy sauce", "salty"), ("pepper", "spicy"), ("grapefruit", "sour"), ("sugar", "sweet"),
        ("espresso", "bitter"), ("wasabi", "spicy"), ("caramel", "sweet"),
    ],
    "temperature": [
        ("ice", "cold"), ("fire", "hot"), ("snow", "cold"), ("stove", "hot"),
        ("freezer", "cold"), ("oven", "hot"), ("glacier", "cold"), ("lava", "hot"),
        ("refrigerator", "cold"), ("furnace", "hot"), ("arctic", "cold"), ("desert", "hot"),
        ("frost", "cold"), ("ember", "hot"), ("winter wind", "cold"),
    ],
    "texture": [
        ("silk", "smooth"), ("sandpaper", "rough"), ("pillow", "soft"), ("diamond", "hard"),
        ("glass", "smooth"), ("bark", "rough"), ("cotton", "soft"), ("rock", "hard"),
        ("velvet", "smooth"), ("concrete", "rough"), ("feather", "soft"), ("steel", "hard"),
        ("marble", "smooth"), ("gravel", "rough"), ("wool", "soft"),
    ],
    "shape": [
        ("ball", "round"), ("box", "square"), ("needle", "thin"), ("plate", "flat"),
        ("sphere", "round"), ("cube", "square"), ("wire", "thin"), ("table", "flat"),
        ("moon", "curved"), ("brick", "rectangular"), ("sword", "long"), ("coin", "flat"),
        ("wheel", "round"), ("building", "rectangular"), ("ribbon", "thin"),
    ],
    "size": [
        ("elephant", "large"), ("ant", "small"), ("mountain", "huge"), ("grain", "tiny"),
        ("whale", "large"), ("flea", "small"), ("planet", "huge"), ("atom", "tiny"),
        ("ocean", "vast"), ("droplet", "tiny"), ("tower", "tall"), ("seed", "small"),
        ("continent", "vast"), ("pebble", "small"), ("giraffe", "tall"),
    ],
}

WORD_CLUSTERS = {
    "color": ["red", "blue", "green", "yellow", "white", "black", "orange", "purple", "pink", "brown"],
    "taste": ["sweet", "sour", "bitter", "salty", "spicy", "savory", "tangy", "umami"],
    "temperature": ["hot", "cold", "warm", "cool", "freezing", "boiling", "scalding", "frigid"],
    "texture": ["smooth", "rough", "soft", "hard", "sharp", "fluffy", "slick", "bumpy"],
    "shape": ["round", "square", "flat", "thin", "long", "curved", "rectangular", "circular"],
    "size": ["large", "small", "huge", "tiny", "vast", "tall", "big", "little"],
}


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


def format_templates(template_set, noun, val):
    """Format template set into dict of sentences. Handle list-of-templates."""
    result = {}
    for level in ["slot", "type", "value", "baseline"]:
        tmpl = template_set[level]
        if isinstance(tmpl, list):
            result[level] = [t.format(obj=noun, val=val) for t in tmpl]
        else:
            result[level] = tmpl.format(obj=noun, val=val)
    return result


def compute_direction_from_templates(model, tokenizer, device, sentences, baseline_sent, target_layer):
    """
    Extract direction from template sentences vs baseline.
    If sentences is a list, average the representations.
    """
    h_baseline = extract_rep_at_layer(model, tokenizer, device, baseline_sent, target_layer)
    
    if isinstance(sentences, list):
        h_levels = []
        for s in sentences:
            h_levels.append(extract_rep_at_layer(model, tokenizer, device, s, target_layer))
        h_avg = np.mean(h_levels, axis=0)
    else:
        h_avg = extract_rep_at_layer(model, tokenizer, device, sentences, target_layer)
    
    d = h_avg - h_baseline
    norm = np.linalg.norm(d)
    if norm < 1e-10:
        return None
    return d / norm


def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase325_{model_name}.log")

    log(f"=== Phase 325: Specialized Template Confirmation for {model_name} ===")

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

    alpha = 2.0  # Use alpha=2.0 for stronger signal (confirmed in 324b)
    results = {}

    # ===================================================================
    # Test 1: Generic vs Specialized templates — 6 types × 15 pairs
    # ===================================================================
    log("\n" + "="*60)
    log("Test 1: Generic vs Specialized Template Comparison")
    log("="*60)

    comparison_results = {}

    for attr_type in ["color", "taste", "temperature", "texture", "shape", "size"]:
        log(f"\n  --- {attr_type} ---")
        pairs = ATTR_PAIRS[attr_type]
        generic_tmpl = TEMPLATE_SETS[attr_type]["generic"]
        specialized_tmpl = TEMPLATE_SETS[attr_type]["specialized"]
        cluster_ids = get_cluster_token_ids(tokenizer, WORD_CLUSTERS[attr_type])

        generic_stats = {"slot_tgt": [], "type_tgt": [], "value_tgt": [],
                         "slot_clst": [], "type_clst": [], "value_clst": []}
        spec_stats = {"slot_tgt": [], "type_tgt": [], "value_tgt": [],
                      "slot_clst": [], "type_clst": [], "value_clst": []}

        for pair_idx, (noun, val) in enumerate(pairs):
            val_ids = tokenizer.encode(val, add_special_tokens=False)
            if not val_ids:
                continue
            tgt_id = val_ids[0]

            target_prompt = f"The {noun} is"
            baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)
            baseline_logit = float(baseline_logits[tgt_id])
            baseline_cluster = compute_cluster_mean(baseline_logits, cluster_ids)

            # --- Generic templates ---
            gen_sents = format_templates(generic_tmpl, noun, val)
            gen_baseline = gen_sents["baseline"]
            
            for level in ["slot", "type", "value"]:
                d = compute_direction_from_templates(
                    model, tokenizer, device, gen_sents[level], gen_baseline, opt_layer)
                if d is None:
                    continue
                inj_logits = inject_direction_at_layer(
                    model, tokenizer, device, target_prompt, d, opt_layer, alpha)
                tgt_delta = float(inj_logits[tgt_id] - baseline_logit)
                cluster_delta = compute_cluster_mean(inj_logits, cluster_ids) - baseline_cluster
                generic_stats[f"{level}_tgt"].append(tgt_delta)
                generic_stats[f"{level}_clst"].append(cluster_delta)

            # --- Specialized templates ---
            spec_sents = format_templates(specialized_tmpl, noun, val)
            spec_baseline = spec_sents["baseline"]
            
            for level in ["slot", "type", "value"]:
                d = compute_direction_from_templates(
                    model, tokenizer, device, spec_sents[level], spec_baseline, opt_layer)
                if d is None:
                    continue
                inj_logits = inject_direction_at_layer(
                    model, tokenizer, device, target_prompt, d, opt_layer, alpha)
                tgt_delta = float(inj_logits[tgt_id] - baseline_logit)
                cluster_delta = compute_cluster_mean(inj_logits, cluster_ids) - baseline_cluster
                spec_stats[f"{level}_tgt"].append(tgt_delta)
                spec_stats[f"{level}_clst"].append(cluster_delta)

            if pair_idx % 5 == 4:
                log(f"    {attr_type} pair {pair_idx+1}/{len(pairs)} done")
            torch.cuda.empty_cache()

        # Aggregate
        gen_agg = {}
        spec_agg = {}
        for level in ["slot", "type", "value"]:
            gt = generic_stats[f"{level}_tgt"]
            gc_ = generic_stats[f"{level}_clst"]
            st = spec_stats[f"{level}_tgt"]
            sc_ = spec_stats[f"{level}_clst"]
            gen_agg[level] = {
                "tgt_mean": round(float(np.mean(gt)), 4) if gt else 0,
                "cluster_mean": round(float(np.mean(gc_)), 4) if gc_ else 0,
                "tgt_negative_rate": round(sum(1 for x in gt if x < 0)/max(len(gt),1), 4),
            }
            spec_agg[level] = {
                "tgt_mean": round(float(np.mean(st)), 4) if st else 0,
                "cluster_mean": round(float(np.mean(sc_)), 4) if sc_ else 0,
                "tgt_negative_rate": round(sum(1 for x in st if x < 0)/max(len(st),1), 4),
            }

        # Improvement ratio: specialized / generic
        improvement = {}
        for level in ["slot", "type", "value"]:
            g = gen_agg[level]["tgt_mean"]
            s = spec_agg[level]["tgt_mean"]
            if abs(g) < 0.001:
                improvement[level] = "N/A (generic~0)"
            else:
                improvement[level] = round(s / g, 2) if g > 0 else round(s / g, 2)
        
        comparison_results[attr_type] = {
            "generic": gen_agg,
            "specialized": spec_agg,
            "improvement_ratio": improvement,
        }

        log(f"  {attr_type} Generic:    slot={gen_agg['slot']['tgt_mean']:.4f}, "
            f"type={gen_agg['type']['tgt_mean']:.4f}, value={gen_agg['value']['tgt_mean']:.4f}")
        log(f"  {attr_type} Specialized: slot={spec_agg['slot']['tgt_mean']:.4f}, "
            f"type={spec_agg['type']['tgt_mean']:.4f}, value={spec_agg['value']['tgt_mean']:.4f}")
        log(f"  Improvement: {improvement}")

    results["comparison"] = comparison_results

    # ===================================================================
    # Test 2: Cross-template consistency for specialized templates
    # ===================================================================
    log("\n" + "="*60)
    log("Test 2: Cross-Template Consistency (Specialized)")
    log("="*60)

    consistency_results = {}

    for attr_type in ["color", "taste", "temperature", "texture", "shape", "size"]:
        log(f"\n  --- {attr_type} consistency ---")
        pairs = ATTR_PAIRS[attr_type][:8]  # Use 8 pairs for consistency
        spec_tmpl = TEMPLATE_SETS[attr_type]["specialized"]

        # For each level, compute direction for each template separately
        # Then measure cosine between them
        level_coss = defaultdict(list)

        for noun, val in pairs:
            spec_sents = format_templates(spec_tmpl, noun, val)
            spec_baseline = spec_sents["baseline"]

            for level in ["slot", "type", "value"]:
                sents = spec_sents[level]
                if not isinstance(sents, list) or len(sents) < 2:
                    continue

                # Get direction for each template separately
                dirs = []
                for s in sents:
                    d = compute_direction_from_templates(
                        model, tokenizer, device, s, spec_baseline, opt_layer)
                    if d is not None:
                        dirs.append(d)

                # Pairwise cosine
                if len(dirs) >= 2:
                    cos_val = float(np.dot(dirs[0], dirs[1]) / 
                                   (np.linalg.norm(dirs[0]) * np.linalg.norm(dirs[1]) + 1e-10))
                    level_coss[level].append(cos_val)

            torch.cuda.empty_cache()

        agg_consistency = {}
        for level in ["slot", "type", "value"]:
            coss = level_coss[level]
            if coss:
                agg_consistency[level] = {
                    "mean_cos": round(float(np.mean(coss)), 4),
                    "n_pairs": len(coss),
                }
            else:
                agg_consistency[level] = {"mean_cos": "N/A", "n_pairs": 0}

        consistency_results[attr_type] = agg_consistency
        log(f"  {attr_type}: " + 
            ", ".join(f"{lv}={agg_consistency[lv]['mean_cos']}" for lv in ["slot","type","value"]))

    results["consistency"] = consistency_results

    # ===================================================================
    # Test 3: Best-level identification per type
    # ===================================================================
    log("\n" + "="*60)
    log("Test 3: Best Level per Attribute Type")
    log("="*60)

    best_level = {}
    for attr_type in ["color", "taste", "temperature", "texture", "shape", "size"]:
        spec = comparison_results[attr_type]["specialized"]
        vals = {lv: spec[lv]["tgt_mean"] for lv in ["slot", "type", "value"]}
        best = max(vals, key=lambda k: vals[k])
        best_val = vals[best]
        
        # Also check if best is positive
        if best_val <= 0:
            best = "none (all negative)"
        
        best_level[attr_type] = {
            "best_level": best,
            "best_value": round(best_val, 4),
            "slot": round(vals["slot"], 4),
            "type": round(vals["type"], 4),
            "value": round(vals["value"], 4),
        }
        log(f"  {attr_type}: best={best} ({best_val:.4f}), "
            f"slot={vals['slot']:.4f}, type={vals['type']:.4f}, value={vals['value']:.4f}")

    results["best_level"] = best_level

    # ===================================================================
    # Save
    # ===================================================================
    output = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "opt_layer": opt_layer,
        "alpha": alpha,
        "results": results,
    }

    out_path = RESULT_DIR / f"{model_name}_phase325.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")

    # ===================================================================
    # Summary
    # ===================================================================
    log("\n" + "="*60)
    log(f"PHASE 325 SUMMARY - {model_name}")
    log("="*60)

    log("\n  Generic vs Specialized tgt_delta comparison:")
    for attr_type in ["color", "taste", "temperature", "texture", "shape", "size"]:
        g = comparison_results[attr_type]["generic"]
        s = comparison_results[attr_type]["specialized"]
        imp = comparison_results[attr_type]["improvement_ratio"]
        log(f"    {attr_type}:")
        log(f"      Generic:    slot={g['slot']['tgt_mean']:.4f}, type={g['type']['tgt_mean']:.4f}, value={g['value']['tgt_mean']:.4f}")
        log(f"      Specialized: slot={s['slot']['tgt_mean']:.4f}, type={s['type']['tgt_mean']:.4f}, value={s['value']['tgt_mean']:.4f}")
        log(f"      Improvement: {imp}")

    log("\n  Best level per type (specialized):")
    for attr_type, bl in best_level.items():
        log(f"    {attr_type}: {bl['best_level']} ({bl['best_value']:.4f})")

    log("\n  Cross-template consistency (specialized):")
    for attr_type, con in consistency_results.items():
        log(f"    {attr_type}: " + 
            ", ".join(f"{lv}={con[lv]['mean_cos']}" for lv in ["slot","type","value"]))

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
