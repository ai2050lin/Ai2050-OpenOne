"""
Phase 266 Fast: GLM4 Random Direction Control Test — Using output_hidden_states
=============================================================================

Key fix: Use output_hidden_states=True instead of hooks for device_map="auto" models.
Hooks fail on GLM4 because layers show as "meta" device with accelerate's dispatch.
output_hidden_states is the standard way and works with device_map="auto".

Optimizations:
- Only test deepest available layer (the decisive test)
- n_random=15 (instead of 30)
- 6 alpha points (instead of 11)
- Aggressive gc after each direction
- Save intermediate results every 3 directions
"""
import sys, os, json, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = Path("results/phase266_semantic_axis_vs_lowdim")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Word lists
SING_WORDS = [
    "cat", "dog", "bird", "fish", "child", "woman", "man", "person",
    "teacher", "doctor", "student", "writer", "artist", "driver", "worker",
    "tree", "flower", "river", "mountain", "book", "car", "house", "door",
    "girl", "boy", "king", "queen", "hero", "friend", "mother", "father",
]
PLUR_WORDS = [
    "cats", "dogs", "birds", "fish", "children", "women", "men", "people",
    "teachers", "doctors", "students", "writers", "artists", "drivers", "workers",
    "trees", "flowers", "rivers", "mountains", "books", "cars", "houses", "doors",
    "girls", "boys", "kings", "queens", "heroes", "friends", "mothers", "fathers",
]
SING_VERBS = ["runs", "walks", "sits", "is", "has", "does", "goes", "was", "eats", "makes"]
PLUR_VERBS = ["run", "walk", "sit", "are", "have", "do", "go", "were", "eat", "make"]
TEST_SING = [
    "bear", "eagle", "rabbit", "tiger", "whale", "fox", "deer", "wolf",
    "snake", "crow", "ant", "owl", "penguin", "dolphin", "spider",
]
NUMBER_TEST = [f"The {w}" for w in TEST_SING]


def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def safe_get_token_id(tokenizer, word):
    try:
        ids = tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None
    except:
        return None


def get_special_token_offset(tokenizer, text):
    try:
        full_ids = tokenizer.encode(text, add_special_tokens=True)
        no_special = tokenizer.encode(text, add_special_tokens=False)
        return len(full_ids) - len(no_special)
    except:
        return 0


def get_input_device(model):
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_bf16(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS, get_model_info

    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (BF16 + device_map=auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
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
            log_time(f"  Loaded with attn_implementation={attn_impl}")
            break
        except Exception as e:
            log_time(f"  {attn_impl} failed: {str(e)[:80]}, trying next...")
            continue

    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")

    model.eval()
    info = get_model_info(model, model_name)

    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"  class={info.model_class}, layers={info.n_layers}, d_model={info.d_model}, "
             f"GPU={gpu_mem:.2f}GB")
    
    # Show device map summary
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        devices = {}
        for k, v in dmap.items():
            dv = str(v)
            devices[dv] = devices.get(dv, 0) + 1
        log_time(f"  Device map: {devices}")

    return model, tokenizer, info


def get_hidden_state_at_layer(model, tokenizer, input_device, prompt, target_layer):
    """Extract hidden state at a specific layer using output_hidden_states=True"""
    import torch
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attn_mask = inputs["attention_mask"].to(input_device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                   output_hidden_states=True)
    
    # hidden_states[0] = embedding output, hidden_states[i+1] = after layer i
    if target_layer + 1 < len(out.hidden_states):
        hs = out.hidden_states[target_layer + 1][0, :, :].float().cpu().numpy()
    else:
        hs = out.hidden_states[-1][0, :, :].float().cpu().numpy()
    
    logits = out.logits[0, -1, :].float().cpu().numpy()
    
    torch.cuda.empty_cache()
    return hs, logits


def extract_probe_direction_hs(model, tokenizer, info, input_device,
                               words_a, words_b, template_a, template_b,
                               target_layer, pos_func, n_train=25):
    """Extract probe direction using output_hidden_states (hook-free, device_map compatible)"""
    from sklearn.linear_model import LogisticRegression

    activations = {"a": [], "b": []}
    errors = 0

    for words, key in [(words_a, "a"), (words_b, "b")]:
        template = template_a if key == "a" else template_b
        for word in words[:n_train]:
            prompt = template.format(word)
            try:
                hs, _ = get_hidden_state_at_layer(
                    model, tokenizer, input_device, prompt, target_layer
                )
                
                offset = get_special_token_offset(tokenizer, prompt)
                subj_pos = pos_func(prompt, offset)
                n_tokens = hs.shape[0]
                if subj_pos < 0:
                    subj_pos = n_tokens + subj_pos
                if 0 <= subj_pos < n_tokens:
                    activations[key].append(hs[subj_pos, :])
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    log_time(f"    Error extracting: {str(e)[:80]}")

    log_time(f"    Collected: a={len(activations['a'])}, b={len(activations['b'])}, errors={errors}")

    if len(activations["a"]) < 5 or len(activations["b"]) < 5:
        return None, None

    X = np.vstack([activations["a"], activations["b"]])
    y = np.array([0] * len(activations["a"]) + [1] * len(activations["b"]))

    try:
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y)
        acc = float(clf.score(X, y))
        direction = clf.coef_[0]
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        return acc, direction
    except Exception as e:
        log_time(f"    Probe training failed: {str(e)[:80]}")
        return None, None


def measure_intervention_r2_hs(model, tokenizer, info, input_device,
                               direction, target_layer, test_prompts,
                               verb_ids_a, verb_ids_b, alpha_values,
                               pos_func):
    """Measure R² using output_hidden_states with manual direction injection
    
    Instead of hooks, we:
    1. Get hidden state at target layer (baseline)
    2. Add direction * alpha to the hidden state
    3. Continue forward pass from modified hidden state to get logits
    
    BUT this requires model to accept pre-computed hidden states, which isn't standard.
    
    Alternative: Use the hook approach but with proper device handling for device_map="auto"
    """
    import torch
    
    # For device_map="auto" models, we need to find where to inject
    # Use the model's forward with hooks, but handle device carefully
    
    # Get the device where the target layer's output lives
    # by running a test forward pass
    test_prompt = test_prompts[0]
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attn_mask = inputs["attention_mask"].to(input_device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask,
                   output_hidden_states=True)
    
    # Check the device of hidden states at target layer
    if target_layer + 1 < len(out.hidden_states):
        hs_device = out.hidden_states[target_layer + 1].device
    else:
        hs_device = out.hidden_states[-1].device
    
    log_time(f"    Hidden state device at L{target_layer}: {hs_device}")
    
    # Use hook-based intervention, but with proper device handling
    from model_utils import get_layers
    layers = get_layers(model)
    target_module = layers[target_layer]
    
    # Determine the device for the direction tensor
    # Use the device of the hidden state (not the module parameters, which may be meta)
    dir_device = hs_device
    
    mean_scores = []

    for alpha in alpha_values:
        score_changes = []

        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            offset = get_special_token_offset(tokenizer, prompt)
            subj_pos = pos_func(prompt, offset)
            n_tokens = input_ids.shape[1]
            if subj_pos < 0:
                subj_pos = n_tokens + subj_pos
            if subj_pos < 0 or subj_pos >= n_tokens:
                continue

            # Baseline
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask)
            baseline_logits = out.logits[0, -1, :].float().cpu().numpy()
            bl_a = float(np.mean([baseline_logits[tid] for tid in verb_ids_a if tid < len(baseline_logits)]))
            bl_b = float(np.mean([baseline_logits[tid] for tid in verb_ids_b if tid < len(baseline_logits)]))
            bl_score = bl_a - bl_b

            # Intervention with hook
            dir_tensor = torch.tensor(direction, dtype=torch.float32, device=dir_device)
            if model.dtype != torch.float32:
                dir_tensor = dir_tensor.to(model.dtype)

            def make_hook(d, a, p, dev):
                def hook(module, inp, output):
                    if isinstance(output, tuple):
                        modified = output[0].clone()
                        # Move direction to same device as output
                        d_dev = d.to(modified.device)
                        modified[:, p, :] += a * d_dev
                        return (modified,) + output[1:]
                    else:
                        modified = output.clone()
                        d_dev = d.to(modified.device)
                        modified[:, p, :] += a * d_dev
                        return modified
                return hook

            hook = target_module.register_forward_hook(
                make_hook(dir_tensor, alpha, subj_pos, dir_device)
            )

            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask)
            except Exception as e:
                hook.remove()
                continue

            hook.remove()

            logits = out.logits[0, -1, :].float().cpu().numpy()
            a_v = float(np.mean([logits[tid] for tid in verb_ids_a if tid < len(logits)]))
            b_v = float(np.mean([logits[tid] for tid in verb_ids_b if tid < len(logits)]))
            score = a_v - b_v
            score_changes.append(score - bl_score)

            torch.cuda.empty_cache()

        mean_score = float(np.mean(score_changes)) if score_changes else 0.0
        mean_scores.append(mean_score)

    # Analyze
    alpha_arr = np.array(alpha_values, dtype=float)
    score_arr = np.array(mean_scores)

    diffs = np.diff(score_arr)
    n_increasing = sum(1 for d in diffs if d > 0)
    monotonicity = n_increasing / len(diffs) if len(diffs) > 0 else 0

    if np.std(score_arr) > 1e-10:
        correlation = float(np.corrcoef(alpha_arr, score_arr)[0, 1])
        r_squared = correlation ** 2
    else:
        correlation = 0.0
        r_squared = 0.0

    pos_score = float(np.mean([s for a, s in zip(alpha_values, mean_scores) if a > 0])) if any(a > 0 for a in alpha_values) else 0.0
    neg_score = float(np.mean([s for a, s in zip(alpha_values, mean_scores) if a < 0])) if any(a < 0 for a in alpha_values) else 0.0
    bidirectional = bool(pos_score * neg_score < 0)

    return {
        "r_squared": round(float(r_squared), 4),
        "monotonicity": round(float(monotonicity), 4),
        "correlation": round(float(correlation), 4),
        "bidirectional": bidirectional,
        "mean_scores": [round(s, 4) for s in mean_scores],
    }


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "glm4"
    log_time(f"Phase 266 Fast: {model_name} — Decisive Random Direction Test")
    log_time(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    import torch
    from model_utils import release_model

    model, tokenizer, info = load_model_bf16(model_name)
    input_device = get_input_device(model)
    n_layers = info.n_layers
    d_model = info.d_model

    # Try multiple layers
    if model_name == "deepseek7b":
        layer_candidates = [27, 26, 25, 20, 15]
    else:
        layer_candidates = [n_layers - 2, n_layers - 3, n_layers - 5, n_layers - 8, n_layers // 2]
    n_random = 15
    alpha_values = list(np.arange(-10, 10.5, 4.0))  # 6 points

    verb_ids_sing = [safe_get_token_id(tokenizer, v) for v in SING_VERBS]
    verb_ids_plur = [safe_get_token_id(tokenizer, v) for v in PLUR_VERBS]
    verb_ids_sing = [v for v in verb_ids_sing if v is not None]
    verb_ids_plur = [v for v in verb_ids_plur if v is not None]

    log_time(f"  Layer candidates: {layer_candidates}, n_random: {n_random}")

    # Step 1: Extract number semantic direction (try multiple layers)
    sem_direction = None
    target_layer = None
    probe_acc = None
    
    for try_layer in layer_candidates:
        if try_layer < 0 or try_layer >= n_layers:
            continue
        log_time(f"  Trying to extract number direction at L{try_layer}...")
        probe_acc, sem_direction = extract_probe_direction_hs(
            model, tokenizer, info, input_device,
            SING_WORDS, PLUR_WORDS, "The {} sits", "The {} sit",
            try_layer, lambda p, off: 1 + off, n_train=25
        )
        if sem_direction is not None:
            target_layer = try_layer
            log_time(f"  SUCCESS: Number direction at L{target_layer}, probe_acc={probe_acc}")
            break
        log_time(f"  Failed at L{try_layer}, trying next...")
        gc.collect()
        torch.cuda.empty_cache()

    if sem_direction is None:
        log_time(f"  FAILED: Could not extract number direction at any layer")
        release_model(model)
        return

    # Step 2: Measure R² for SEMANTIC direction
    log_time(f"  Measuring R² for SEMANTIC (number) direction at L{target_layer}...")
    sem_result = measure_intervention_r2_hs(
        model, tokenizer, info, input_device,
        sem_direction, target_layer, NUMBER_TEST,
        verb_ids_sing, verb_ids_plur, alpha_values,
        lambda p, off: 1 + off
    )
    log_time(f"  SEMANTIC R²: {sem_result['r_squared']:.4f}, mono: {sem_result['monotonicity']:.4f}")

    # Step 3: Test RANDOM directions
    log_time(f"  Testing {n_random} random directions at L{target_layer}...")
    random_r2_values = []

    for i in range(n_random):
        random_dir = np.random.randn(d_model).astype(np.float32)
        random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-10)

        rand_result = measure_intervention_r2_hs(
            model, tokenizer, info, input_device,
            random_dir, target_layer, NUMBER_TEST,
            verb_ids_sing, verb_ids_plur, alpha_values,
            lambda p, off: 1 + off
        )
        random_r2_values.append(rand_result['r_squared'])

        log_time(f"    Direction {i+1}/{n_random}: R²={rand_result['r_squared']:.4f}, "
                 f"running_mean={np.mean(random_r2_values):.4f}")

        torch.cuda.empty_cache()
        gc.collect()

        # Save intermediate every 3 directions
        if (i + 1) % 3 == 0:
            interim = {
                "model": model_name,
                "layer": int(target_layer),
                "semantic_r2": round(float(sem_result['r_squared']), 4),
                "random_r2_values": [round(float(x), 4) for x in random_r2_values],
                "random_r2_mean": round(float(np.mean(random_r2_values)), 4),
                "random_r2_std": round(float(np.std(random_r2_values)), 4),
                "n_completed": i + 1,
                "n_total": n_random,
            }
            interim_file = RESULT_DIR / f"part1_{model_name}_interim.json"
            with open(interim_file, "w", encoding="utf-8") as f:
                json.dump(interim, f, indent=2, ensure_ascii=False)
            log_time(f"    Intermediate saved")

    # Step 4: Final verdict
    random_r2_mean = float(np.mean(random_r2_values))
    random_r2_std = float(np.std(random_r2_values))
    random_r2_max = float(np.max(random_r2_values))
    sem_r2 = sem_result['r_squared']

    if random_r2_std > 1e-10:
        effect_size = (sem_r2 - random_r2_mean) / random_r2_std
    else:
        effect_size = float('inf') if sem_r2 > random_r2_mean else 0

    if random_r2_mean > 0.8:
        verdict = "LOW_DIM_TRIVIAL_EFFECT"
    elif random_r2_mean < 0.1 and sem_r2 > 0.9:
        verdict = "TRUE_SEMANTIC_SELECTION"
    elif random_r2_mean < 0.3 and sem_r2 - random_r2_mean > 0.5:
        verdict = "PARTIALLY_SEMANTIC"
    else:
        verdict = "MIXED_EFFECT"

    result = {
        "model": model_name,
        "layer": int(target_layer),
        "n_layers": int(n_layers),
        "d_model": int(d_model),
        "n_random": int(n_random),
        "alpha_points": len(alpha_values),
        "semantic_r2": round(float(sem_r2), 4),
        "semantic_monotonicity": round(float(sem_result['monotonicity']), 4),
        "semantic_bidirectional": bool(sem_result['bidirectional']),
        "random_r2_mean": round(random_r2_mean, 4),
        "random_r2_std": round(random_r2_std, 4),
        "random_r2_max": round(random_r2_max, 4),
        "random_r2_values": [round(float(x), 4) for x in random_r2_values],
        "effect_size_sigma": round(effect_size, 2),
        "verdict": verdict,
    }

    result_file = RESULT_DIR / f"part1_{model_name}_random_direction.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log_time(f"\n{'='*60}")
    log_time(f"FINAL RESULT: {model_name} at L{target_layer}")
    log_time(f"  Semantic R²: {sem_r2:.4f}")
    log_time(f"  Random R²: mean={random_r2_mean:.4f}, std={random_r2_std:.4f}, max={random_r2_max:.4f}")
    log_time(f"  Effect size: {effect_size:.2f} sigma")
    log_time(f"  VERDICT: {verdict}")
    log_time(f"{'='*60}")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    log_time(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
