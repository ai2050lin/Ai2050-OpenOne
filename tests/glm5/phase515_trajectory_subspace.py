"""
Phase 515: U_trajectory Discovery + Multi-Step Hub Dynamics + Action Natural Templates
========================================================================================

Phase 514关键发现：
1. kind/type的V_c_semantic最高但cat_logit_delta很低(甚至略负)
   → hub不是一步提升类别logit，而是重新配置状态空间使未来更容易到达类别词
2. qwen3 fruit: match_rate=0.0 → greedy top-1从未是路径价值最高的token
3. DS7B的hub效应与qwen3/GLM4完全相反(负效应)

Phase 515核心实验：
- Exp1: U_trajectory发现 — 用h_0区分成功/失败轨迹的子空间
- Exp2: 多步hub状态动态 — 追踪hub→category路径的类别logit演变
- Exp3: Action自然模板 — 用自然模板测试action失败是否是模板问题
- Exp4: U_trajectory干预 — 添加/移除轨迹方向对类别logit的效果

用法:
  python tests/glm5/phase515_trajectory_subspace.py qwen3
  python tests/glm5/phase515_trajectory_subspace.py glm4 --test-objects 5
  python tests/glm5/phase515_trajectory_subspace.py deepseek7b --test-objects 5
"""
import sys, os, gc, time, argparse, json, re
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import numpy as np
import torch
from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS

# ============== Configuration ==============
FRUIT_OBJECTS = ["apple", "banana", "orange", "grape", "strawberry",
                 "mango", "pear", "cherry", "watermelon", "pineapple"]
EMOTION_OBJECTS = ["happiness", "sadness", "anger", "fear", "surprise"]
COLOR_OBJECTS = ["red", "blue", "green", "yellow", "purple"]
ACTION_OBJECTS = ["running", "jumping", "eating", "writing", "speaking"]

FRUIT_TEMPLATES = [
    "belongs to the category of",
    "is classified as a type of",
    "is a kind of",
]
ACTION_NATURAL_TEMPLATES = [
    "The person is",
    "This activity is called",
    "The action is",
]
ACTION_ORIGINAL_TEMPLATES = [
    "belongs to the category of",
    "is classified as a type of",
    "is a kind of",
]

CATEGORIES = {
    "fruit": {"objects": FRUIT_OBJECTS, "templates": FRUIT_TEMPLATES,
              "cat_words": ["fruit", "fruits", "Fruit"]},
    "emotion": {"objects": EMOTION_OBJECTS, "templates": FRUIT_TEMPLATES[:2],
                "cat_words": ["emotion", "Emotion", "emotions", "feeling", "feelings"]},
    "color": {"objects": COLOR_OBJECTS, "templates": FRUIT_TEMPLATES[:2],
              "cat_words": ["color", "Color", "colors", "colour", "colours"]},
    "action_natural": {"objects": ACTION_OBJECTS, "templates": ACTION_NATURAL_TEMPLATES,
                       "cat_words": ["action", "running", "jumping", "eating", "writing",
                                     "speaking", "run", "jump", "eat", "write", "speak"]},
    "action_original": {"objects": ACTION_OBJECTS, "templates": ACTION_ORIGINAL_TEMPLATES,
                        "cat_words": ["action", "running", "jumping", "eating", "writing",
                                      "speaking", "run", "jump", "eat", "write", "speak"]},
}

HUB_TOKENS = [" a", " the", " kind", " type"]


def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True,
        attn_implementation="eager")
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"Loaded: device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def make_prompts(objects, templates):
    """Create prompt texts"""
    prompts = []
    for obj in objects:
        for tmpl in templates:
            if tmpl.startswith("The") or tmpl.startswith("This"):
                text = f"{tmpl} {obj}"
            else:
                text = f"{obj} {tmpl}"
            prompts.append(text)
    return prompts


def tok_ids(tokenizer, tokens):
    ids = []
    for t in tokens:
        ids.extend(tokenizer.encode(t, add_special_tokens=False))
    return ids


def classify_hit(generated_text, cat_words):
    """
    Classify trajectory quality:
    miss / lexical / natural_phrase / semantic_answer
    """
    text_lower = generated_text.lower()
    found_cat = None
    for cw in cat_words:
        if cw.lower() in text_lower:
            found_cat = cw.lower()
            break

    if found_cat is None:
        return "miss", {}

    # Check natural phrase patterns
    natural_patterns = [
        r"a\s+" + found_cat,
        r"an\s+" + found_cat,
        r"the\s+" + found_cat,
        r"type\s+of\s+" + found_cat,
        r"kind\s+of\s+" + found_cat,
        r"category\s+of\s+" + found_cat,
        r"classified\s+as\s+a\s+" + found_cat,
    ]
    for pat in natural_patterns:
        if re.search(pat, text_lower):
            return "semantic_answer", {"found_cat": found_cat}

    return "lexical", {"found_cat": found_cat}


def generate_greedy_trajectory(model, tokenizer, device, prompt, steps=8):
    """Generate greedy trajectory, collecting hidden states at key positions"""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    gen_ids = ids.clone()
    gen_mask = mask.clone()
    gen_tokens = []

    with torch.no_grad():
        for step in range(steps):
            out = model(input_ids=gen_ids, attention_mask=gen_mask)
            next_logits = out.logits[:, -1, :].float().cpu().numpy()
            next_id = int(np.argmax(next_logits))
            next_tok = tokenizer.decode([next_id])
            gen_tokens.append(next_tok)
            gen_ids = torch.cat([gen_ids, torch.tensor([[next_id]], device=device)], dim=-1)
            gen_mask = torch.cat([gen_mask, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)

    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return gen_text, gen_tokens, gen_ids, gen_mask


def get_h0_at_layers(model, tokenizer, device, prompt, layer_indices):
    """Get hidden state at specified layers for the last position of the prompt"""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    # Use output_hidden_states=True instead of hooks (more reliable)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        logits = out.logits[:, -1, :].float().cpu().numpy().flatten()
        hs = out.hidden_states  # tuple of (n_layers+1) tensors, each [1, seq_len, d_model]

    captured = {}
    # hidden_states[0] = embedding output, hidden_states[1] = after layer 0, etc.
    # So layer index li corresponds to hidden_states[li+1]
    for li in layer_indices:
        hs_idx = li + 1  # shift by 1 because hs[0] is embedding
        if hs_idx < len(hs):
            h_vec = hs[hs_idx][:, -1, :].detach().float().cpu().numpy().flatten()
            captured[li] = h_vec

    return captured, logits


def run_exp1_utrajectory(model, tokenizer, device, cat_name, cat_config, n_test):
    """
    Exp1: U_trajectory Discovery
    - For each prompt: get h_0 at key layers, generate greedy trajectory
    - Separate into success/failure groups
    - Compute d_traj = h_success_mean - h_fail_mean
    """
    log(f"  Exp1: U_trajectory discovery for {cat_name}...")

    objects = cat_config["objects"][:n_test]
    templates = cat_config["templates"]
    cat_words = cat_config["cat_words"]
    cat_ids = tok_ids(tokenizer, cat_words)
    prompts = make_prompts(objects, templates)

    info = get_model_info(model, model_name_global)
    n_layers = info.n_layers
    # Track 3 key layers
    layer_indices = [n_layers // 4, n_layers // 2, n_layers - 1]

    success_hiddens = {li: [] for li in layer_indices}
    fail_hiddens = {li: [] for li in layer_indices}
    success_cat_logits = []
    fail_cat_logits = []
    trajectory_results = []

    for i, prompt in enumerate(prompts):
        # Get h_0 at key layers
        h0_data, h0_logits = get_h0_at_layers(model, tokenizer, device, prompt, layer_indices)
        cat_logit_h0 = float(h0_logits[cat_ids].max())

        # Generate greedy trajectory
        gen_text, gen_tokens, _, _ = generate_greedy_trajectory(model, tokenizer, device, prompt, steps=8)
        quality, info_dict = classify_hit(gen_text, cat_words)

        trajectory_results.append({
            "prompt": prompt,
            "gen_text": gen_text[:200],
            "quality": quality,
            "gen_tokens": gen_tokens[:8],
            "cat_logit_h0": cat_logit_h0,
        })

        # Group by quality:
        # success = semantic_answer (natural phrase with category word)
        # contrast = lexical (category word appears unnaturally) OR miss (no category word)
        if quality in ["semantic_answer", "natural_phrase"]:
            for li in layer_indices:
                if li in h0_data:
                    success_hiddens[li].append(h0_data[li])
            success_cat_logits.append(cat_logit_h0)
        else:  # lexical or miss — both are "non-natural" trajectory outcomes
            for li in layer_indices:
                if li in h0_data:
                    fail_hiddens[li].append(h0_data[li])
            fail_cat_logits.append(cat_logit_h0)

        if (i + 1) % 5 == 0:
            log(f"    Processed {i+1}/{len(prompts)} prompts")
        gc.collect()
        torch.cuda.empty_cache()

    # Compute trajectory direction
    traj_directions = {}
    for li in layer_indices:
        s_n = len(success_hiddens[li])
        f_n = len(fail_hiddens[li])
        if s_n >= 2 and f_n >= 2:
            suc_mean = np.mean(success_hiddens[li], axis=0)
            fail_mean = np.mean(fail_hiddens[li], axis=0)
            d_traj = suc_mean - fail_mean
            # Normalize
            d_norm = np.linalg.norm(d_traj)
            traj_directions[li] = {
                "direction_norm": float(d_norm),
                "suc_n": s_n,
                "fail_n": f_n,
                "suc_cat_logit_mean": float(np.mean(success_cat_logits)) if success_cat_logits else None,
                "fail_cat_logit_mean": float(np.mean(fail_cat_logits)) if fail_cat_logits else None,
                # Store the raw direction for intervention (will save separately)
            }
            # Store raw direction as numpy file for intervention use
            np.save(f"tests/glm5_temp/phase515_d_traj_{cat_name}_L{li}.npy", d_traj)
        else:
            traj_directions[li] = {
                "direction_norm": 0.0,
                "suc_n": s_n,
                "fail_n": f_n,
            }

    quality_dist = {}
    for tr in trajectory_results:
        q = tr["quality"]
        quality_dist[q] = quality_dist.get(q, 0) + 1

    return {
        "trajectory_results": trajectory_results,
        "quality_dist": quality_dist,
        "traj_directions": traj_directions,
        "success_cat_logit_mean": float(np.mean(success_cat_logits)) if success_cat_logits else None,
        "fail_cat_logit_mean": float(np.mean(fail_cat_logits)) if fail_cat_logits else None,
        "layer_indices": layer_indices,
    }


def run_exp2_multistep_hub(model, tokenizer, device, cat_name, cat_config, n_test):
    """
    Exp2: Multi-Step Hub Dynamics
    - Force step1 = hub token, then greedy for 6 steps
    - Track category logit at each step
    - See how hub tokens reconfigure the state
    """
    log(f"  Exp2: Multi-step hub dynamics for {cat_name}...")

    objects = cat_config["objects"][:n_test]
    templates = cat_config["templates"]
    cat_words = cat_config["cat_words"]
    cat_ids = tok_ids(tokenizer, cat_words)
    prompts = make_prompts(objects, templates)

    info = get_model_info(model, model_name_global)
    n_layers = info.n_layers
    mid_layer = n_layers // 2

    hub_tokens_to_test = HUB_TOKENS  # a, the, kind, type
    steps = 6

    all_hub_results = {}

    for hub_tok in hub_tokens_to_test:
        log(f"    Testing hub '{hub_tok}'...")
        hub_id_list = tok_ids(tokenizer, [hub_tok])

        per_prompt_cat_logits = []  # cat_logit at each step for each prompt
        per_prompt_qualities = []

        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            # Force step1 = hub token
            gen_ids = torch.cat([ids, torch.tensor(hub_id_list, device=device).unsqueeze(0)], dim=-1)
            gen_mask = torch.cat([mask, torch.ones(1, len(hub_id_list), device=device, dtype=torch.long)], dim=-1)

            step_cat_logits = []
            gen_tokens_list = []

            with torch.no_grad():
                for s in range(steps):
                    out = model(input_ids=gen_ids, attention_mask=gen_mask)
                    logits = out.logits[:, -1, :].float().cpu().numpy().flatten()
                    cat_logit = float(logits[cat_ids].max())
                    step_cat_logits.append(cat_logit)

                    next_id = int(np.argmax(logits))
                    next_tok = tokenizer.decode([next_id])
                    gen_tokens_list.append(next_tok)

                    gen_ids = torch.cat([gen_ids, torch.tensor([[next_id]], device=device)], dim=-1)
                    gen_mask = torch.cat([gen_mask, torch.ones(1, 1, device=device, dtype=torch.long)], dim=-1)

            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            quality, _ = classify_hit(gen_text, cat_words)

            per_prompt_cat_logits.append(step_cat_logits)
            per_prompt_qualities.append(quality)

            gc.collect()
            torch.cuda.empty_cache()

        # Average cat_logit trajectory across prompts
        avg_cat_traj = []
        for s in range(steps):
            vals = [pcl[s] for pcl in per_prompt_cat_logits if len(pcl) > s]
            avg_cat_traj.append(float(np.mean(vals)) if vals else None)

        quality_dist = {}
        for q in per_prompt_qualities:
            quality_dist[q] = quality_dist.get(q, 0) + 1

        all_hub_results[hub_tok] = {
            "avg_cat_logit_trajectory": avg_cat_traj,
            "quality_dist": quality_dist,
            "n_prompts": len(prompts),
        }

        log(f"    Hub '{hub_tok}': cat_logit traj = {avg_cat_traj}, quality = {quality_dist}")

    return all_hub_results


def run_exp3_action_natural(model, tokenizer, device, n_test):
    """
    Exp3: Action Natural Templates
    - Compare natural vs original templates for action category
    """
    log(f"  Exp3: Action natural templates...")

    action_objs = ACTION_OBJECTS[:n_test]

    # Natural templates: "The person is running" etc.
    natural_templates = [
        "The person is",
        "This activity is called",
        "The action is",
    ]
    # Original templates: "running belongs to the category of" etc.
    original_templates = [
        "belongs to the category of",
        "is classified as a type of",
        "is a kind of",
    ]

    cat_words_action = ["action", "running", "jumping", "eating", "writing", "speaking",
                        "run", "jump", "eat", "write", "speak", "move"]

    results = {}

    for tmpl_type, templates in [("natural", natural_templates), ("original", original_templates)]:
        log(f"    {tmpl_type} templates...")
        prompts = make_prompts(action_objs, templates)
        trajectory_results = []

        for prompt in prompts:
            gen_text, gen_tokens, _, _ = generate_greedy_trajectory(model, tokenizer, device, prompt, steps=8)
            quality, _ = classify_hit(gen_text, cat_words_action)
            trajectory_results.append({
                "prompt": prompt,
                "gen_text": gen_text[:200],
                "quality": quality,
                "gen_tokens": gen_tokens[:8],
            })
            gc.collect()
            torch.cuda.empty_cache()

        # Aggregate
        action_hit = sum(1 for tr in trajectory_results
                        if any(aw in tr["gen_text"].lower()
                               for aw in ["run", "jump", "eat", "write", "speak", "moving"])) / len(trajectory_results)
        cat_hit = sum(1 for tr in trajectory_results if "action" in tr["gen_text"].lower()) / len(trajectory_results)
        any_hit = sum(1 for tr in trajectory_results if tr["quality"] != "miss") / len(trajectory_results)

        quality_dist = {}
        for tr in trajectory_results:
            q = tr["quality"]
            quality_dist[q] = quality_dist.get(q, 0) + 1

        results[tmpl_type] = {
            "n_prompts": len(trajectory_results),
            "action_word_hit_rate": round(action_hit, 4),
            "category_word_hit_rate": round(cat_hit, 4),
            "any_hit_rate": round(any_hit, 4),
            "quality_dist": quality_dist,
            "examples": trajectory_results[:5],
        }
        log(f"    {tmpl_type}: action_hit={action_hit:.3f}, cat_hit={cat_hit:.3f}, quality={quality_dist}")

    return results


def run_exp4_utrajectory_intervention(model, tokenizer, device, cat_name, cat_config,
                                       n_test, layer_indices):
    """
    Exp4: U_trajectory Intervention
    - Load d_traj from numpy files
    - Add/remove d_traj at a single key layer
    - Measure effect on category logit
    """
    log(f"  Exp4: U_trajectory intervention for {cat_name}...")

    objects = cat_config["objects"][:n_test]
    templates = cat_config["templates"]
    cat_words = cat_config["cat_words"]
    cat_ids = tok_ids(tokenizer, cat_words)
    prompts = make_prompts(objects, templates)

    info = get_model_info(model, model_name_global)
    all_layers = get_layers(model)

    results = {}
    scale_factors = [1.0, 2.0]  # just 2 scales to save time

    for li in layer_indices:
        np_file = f"tests/glm5_temp/phase515_d_traj_{cat_name}_L{li}.npy"
        if not os.path.exists(np_file):
            log(f"    L{li}: no d_traj file, skipping")
            continue

        d_traj = np.load(np_file)
        d_norm = float(np.linalg.norm(d_traj))
        log(f"    L{li}: d_traj norm={d_norm:.2f}")

        d_tensor = torch.tensor(d_traj, dtype=torch.float32)

        for scale in scale_factors:
            scaled_vec = d_tensor * scale

            cat_logits_add = []
            cat_logits_remove = []
            cat_logits_clean = []

            # Process each prompt individually (more reliable)
            for prompt in prompts:
                enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)

                # Clean baseline
                with torch.no_grad():
                    out_clean = model(input_ids=ids, attention_mask=mask)
                logits_clean = out_clean.logits[:, -1, :].float().cpu().numpy().flatten()
                cat_logit_clean = float(logits_clean[cat_ids].max())
                cat_logits_clean.append(cat_logit_clean)

                # Add d_traj at layer li
                def add_hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        h[:, -1, :] = h[:, -1, :] + scaled_vec.to(h.device)
                    return output

                hook_add = all_layers[li].register_forward_hook(add_hook)
                with torch.no_grad():
                    out_add = model(input_ids=ids, attention_mask=mask)
                hook_add.remove()
                logits_add = out_add.logits[:, -1, :].float().cpu().numpy().flatten()
                cat_logit_add = float(logits_add[cat_ids].max())
                cat_logits_add.append(cat_logit_add)

                # Remove d_traj at layer li
                def remove_hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        h[:, -1, :] = h[:, -1, :] - scaled_vec.to(h.device)
                    return output

                hook_rem = all_layers[li].register_forward_hook(remove_hook)
                with torch.no_grad():
                    out_rem = model(input_ids=ids, attention_mask=mask)
                hook_rem.remove()
                logits_rem = out_rem.logits[:, -1, :].float().cpu().numpy().flatten()
                cat_logit_rem = float(logits_rem[cat_ids].max())
                cat_logits_remove.append(cat_logit_rem)

                gc.collect()
                torch.cuda.empty_cache()

            avg_clean = float(np.mean(cat_logits_clean))
            avg_add = float(np.mean(cat_logits_add))
            avg_rem = float(np.mean(cat_logits_remove))

            key = f"L{li}_scale{scale}"
            results[key] = {
                "avg_cat_logit_clean": round(avg_clean, 4),
                "avg_cat_logit_add": round(avg_add, 4),
                "avg_cat_logit_remove": round(avg_rem, 4),
                "delta_add": round(avg_add - avg_clean, 4),
                "delta_remove": round(avg_rem - avg_clean, 4),
            }
            log(f"    L{li} scale={scale}: add_delta={avg_add-avg_clean:+.3f}, remove_delta={avg_rem-avg_clean:+.3f}")

    return results


def run_category(model, tokenizer, device, cat_name, cat_config, n_test):
    """Run all experiments for one category"""
    log(f"\n=== {cat_name} (n={n_test} objects) ===")

    results = {}

    # Exp1: U_trajectory discovery
    exp1 = run_exp1_utrajectory(model, tokenizer, device, cat_name, cat_config, n_test)
    results["utraj_discovery"] = exp1

    # Exp2: Multi-step hub dynamics
    exp2 = run_exp2_multistep_hub(model, tokenizer, device, cat_name, cat_config, n_test)
    results["multistep_hub"] = exp2

    # Exp4: U_trajectory intervention (only if we found directions)
    traj_dirs = exp1.get("traj_directions", {})
    has_dirs = any(v.get("direction_norm", 0) > 0 and v.get("suc_n", 0) >= 2
                   for v in traj_dirs.values())
    if has_dirs:
        exp4 = run_exp4_utrajectory_intervention(model, tokenizer, device, cat_name,
                                                  cat_config, n_test,
                                                  exp1.get("layer_indices", []))
        results["utraj_intervention"] = exp4
    else:
        log(f"  No valid trajectory directions, skipping intervention")
        results["utraj_intervention"] = {"note": "no_directions_found"}

    return results


def main():
    global model_name_global
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--test-objects", type=int, default=10)
    args = parser.parse_args()

    model_name = args.model
    model_name_global = model_name
    n_test = args.test_objects

    log(f"Phase 515: U_trajectory + Multi-Step Hub + Action Templates")
    log(f"Model: {model_name}, n_test: {n_test}")

    # Load model
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)

    # Categories to test
    cats_to_test = ["fruit", "emotion", "action_natural", "action_original"]

    all_results = {
        "phase": 515,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "n_test_objects": n_test,
        "categories": cats_to_test,
        "model_info": {"n_layers": info.n_layers, "d_model": info.d_model},
        "category_results": {},
    }

    for cat_name in cats_to_test:
        cat_config = CATEGORIES[cat_name]
        n_cat = min(n_test, len(cat_config["objects"]))
        cat_results = run_category(model, tokenizer, device, cat_name, cat_config, n_cat)
        all_results["category_results"][cat_name] = cat_results

    # Exp3: Action template comparison (separate)
    exp3 = run_exp3_action_natural(model, tokenizer, device, n_test)
    all_results["action_template_comparison"] = exp3

    # Save results
    out_dir = "results/glm5_phase515_trajectory_subspace"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"phase515_{model_name}_trajectory_subspace.json")

    def clean_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        return obj

    cleaned = clean_for_json(all_results)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_file}")

    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Phase 515 complete for {model_name}")


if __name__ == "__main__":
    main()