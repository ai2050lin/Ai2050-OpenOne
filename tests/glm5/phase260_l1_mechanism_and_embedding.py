"""
Phase 260: GLM4 L1 Mutation Mechanism & Cross-Model Embedding Analysis
======================================================================

Based on Phase 259 deep review, 4 experiment parts:

  Part 1: Cross-model embedding cosine similarity for singular/plural pairs
    - Why GLM4 embedding only has 51.4% probe accuracy while Qwen3 has 95.9%?
    - Compare singular/plural word embedding cosine across 3 models
    - Test if GLM4 tokenizer merges singular/plural forms

  Part 2: GLM4 L0->L1 mutation decomposition (HIGHEST PRIORITY)
    - Decompose L1 output = embedding + attn_output + mlp_output
    - Probe each intermediate state for number information
    - Analyze L1 attention patterns: which heads, what patterns
    - Identify which component creates the number signal

  Part 3: L34_H15 information sink analysis (Qwen3)
    - What does L34_H15 write to verb position when attending first token?
    - Does it carry number signal? (logit effect analysis)
    - Compare with L35_H8 (query-driven) and L31_H17 (subject-locator)

  Part 4: DS7B number information decay analysis
    - Linear probe vs MLP probe on late layers
    - Is the decay real information loss or nonlinear encoding?
    - Compare L1 vs L25 hidden state number separation

Usage:
  python tests/glm5/phase260_l1_mechanism_and_embedding.py --model qwen3 --part 1
  python tests/glm5/phase260_l1_mechanism_and_embedding.py --model glm4 --part 2
  python tests/glm5/phase260_l1_mechanism_and_embedding.py --model qwen3 --part 3
  python tests/glm5/phase260_l1_mechanism_and_embedding.py --model deepseek7b --part 4
"""

import sys, os, json, argparse, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = Path("results/phase260_l1_mechanism")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Utility Functions
# ============================================================

def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def get_W_U_safe(model, model_name):
    """Get W_U weight matrix safely."""
    from model_utils import get_W_U
    W_U = get_W_U(model, model_name)
    return W_U

def safe_weight_to_numpy(weight, model_name, name=""):
    """Convert weight to numpy, handling GLM4's QuantLinear and meta tensors."""
    import torch
    # Try direct conversion first
    try:
        if hasattr(weight, 'is_meta') and weight.is_meta:
            raise ValueError("Meta tensor, need to fetch from model")
        if hasattr(weight, 'shape') and len(weight.shape) == 2:
            return weight.detach().cpu().float().numpy()
        return weight.detach().cpu().float().numpy()
    except (NotImplementedError, ValueError):
        pass
    # For meta tensors or QuantLinear, try getting the actual weight
    try:
        # For layers offloaded to disk/CPU with accelerate
        if hasattr(weight, 'quant_state'):
            # Quantized weight - dequantize
            dequant = weight.dequantize()
            return dequant.detach().cpu().float().numpy()
    except Exception:
        pass
    # Try accessing .weight attribute
    try:
        w = weight.weight
        if hasattr(w, 'is_meta') and w.is_meta:
            raise ValueError("Still meta")
        return w.detach().cpu().float().numpy()
    except Exception:
        pass
    raise ValueError(f"Cannot convert {name} to numpy for {model_name}")

def get_layer_weight_numpy(layer, attr_name, model_name):
    """Safely get a weight matrix from a layer, handling offloaded weights."""
    import torch
    try:
        weight = getattr(layer, attr_name).weight
    except AttributeError:
        weight = getattr(layer, attr_name)
    
    # If meta tensor, we need to fetch the actual parameter from the model
    if hasattr(weight, 'is_meta') and weight.is_meta:
        # The parameter is offloaded; we need to access it differently
        # Use model.hf_device_map to find where it is
        raise ValueError(f"Weight {attr_name} is on meta device - layer is offloaded")
    
    return safe_weight_to_numpy(weight, model_name, attr_name)

def safe_decode(tokenizer, token_id):
    """Safely decode a token ID."""
    try:
        return tokenizer.decode([token_id]).strip()
    except Exception:
        return f"<id:{token_id}>"

def get_special_token_offset(tokenizer, prompt_text):
    """Calculate offset caused by special tokens."""
    no_special = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    with_special = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
    return with_special - no_special

def get_input_device(model):
    """Get the device for input tensors."""
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Grammar prompts for probing
# ============================================================

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

# Singular/plural word pairs for embedding cosine analysis
SINGULAR_PLURAL_PAIRS = [
    ("cat", "cats"), ("dog", "dogs"), ("bird", "birds"), ("book", "books"),
    ("car", "cars"), ("tree", "trees"), ("house", "houses"), ("door", "doors"),
    ("girl", "girls"), ("boy", "boys"), ("king", "kings"), ("queen", "queens"),
    ("teacher", "teachers"), ("doctor", "doctors"), ("student", "students"),
    ("apple", "apples"), ("orange", "oranges"), ("horse", "horses"),
    ("city", "cities"), ("country", "countries"), ("baby", "babies"),
    ("party", "parties"), ("lady", "ladies"), ("body", "bodies"),
    ("knife", "knives"), ("life", "lives"), ("leaf", "leaves"),
    ("wolf", "wolves"), ("child", "children"), ("woman", "women"),
    ("man", "men"), ("person", "people"), ("mouse", "mice"),
    ("goose", "geese"), ("tooth", "teeth"), ("foot", "feet"),
    ("ox", "oxen"), ("sheep", "sheep"), ("fish", "fish"),
    ("day", "days"), ("way", "ways"), ("story", "stories"),
    ("army", "armies"), ("valley", "valleys"), ("journey", "journeys"),
    ("hero", "heroes"), ("half", "halves"), ("loaf", "loaves"),
    ("cherry", "cherries"), ("calf", "calves"),
]

def load_model_safe(model_name):
    """Load model with bfloat16 + device_map=auto + flash attention"""
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
    from model_utils import get_model_info, get_layers
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

def get_W_V_h(W_V, query_head, head_dim, kv_group_size):
    """Extract W_V for a specific query head, handling GQA."""
    kv_h = query_head // kv_group_size
    return W_V[kv_h * head_dim:(kv_h + 1) * head_dim, :]

def get_W_K_h(W_K, query_head, head_dim, kv_group_size):
    """Extract W_K for a specific query head, handling GQA."""
    kv_h = query_head // kv_group_size
    return W_K[kv_h * head_dim:(kv_h + 1) * head_dim, :]

# ============================================================
# Part 1: Cross-model embedding cosine similarity
# ============================================================

def run_part1(model_name):
    """Analyze singular/plural word embedding similarity across models."""
    import torch
    from model_utils import get_layers
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    log_time(f"=== Part 1: Embedding Cosine Analysis for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_safe(model_name)
    input_device = get_input_device(model)

    # Get embedding matrix
    embed_layer = model.get_input_embeddings()
    embed_weight = embed_layer.weight.detach().cpu().float().numpy()  # [vocab, d_model]

    results = {
        "model": model_name,
        "d_model": info.d_model,
        "vocab_size": info.vocab_size,
        "pairs": [],
    }

    # 1a. Cosine similarity between singular/plural embeddings
    log_time("Computing singular/plural embedding cosine similarity...")
    cos_sims = []
    token_match_rates = []

    for sing, plur in SINGULAR_PLURAL_PAIRS:
        sing_ids = tokenizer.encode(sing, add_special_tokens=False)
        plur_ids = tokenizer.encode(plur, add_special_tokens=False)

        # Check if both are single tokens
        sing_single = len(sing_ids) == 1
        plur_single = len(plur_ids) == 1

        if sing_single and plur_single:
            sing_emb = embed_weight[sing_ids[0]]
            plur_emb = embed_weight[plur_ids[0]]
            cos_sim = float(np.dot(sing_emb, plur_emb) / (np.linalg.norm(sing_emb) * np.linalg.norm(plur_emb) + 1e-10))
            cos_sims.append(cos_sim)
            token_match = "single-single"
        elif sing_single and not plur_single:
            # Plural is multi-token
            plur_avg = np.mean(embed_weight[plur_ids], axis=0)
            sing_emb = embed_weight[sing_ids[0]]
            cos_sim = float(np.dot(sing_emb, plur_avg) / (np.linalg.norm(sing_emb) * np.linalg.norm(plur_avg) + 1e-10))
            cos_sims.append(cos_sim)
            token_match = "single-multi"
        elif not sing_single and plur_single:
            sing_avg = np.mean(embed_weight[sing_ids], axis=0)
            plur_emb = embed_weight[plur_ids[0]]
            cos_sim = float(np.dot(sing_avg, plur_emb) / (np.linalg.norm(sing_avg) * np.linalg.norm(plur_emb) + 1e-10))
            cos_sims.append(cos_sim)
            token_match = "multi-single"
        else:
            sing_avg = np.mean(embed_weight[sing_ids], axis=0)
            plur_avg = np.mean(embed_weight[plur_ids], axis=0)
            cos_sim = float(np.dot(sing_avg, plur_avg) / (np.linalg.norm(sing_avg) * np.linalg.norm(plur_avg) + 1e-10))
            cos_sims.append(cos_sim)
            token_match = "multi-multi"

        results["pairs"].append({
            "singular": sing,
            "plural": plur,
            "sing_tokens": len(sing_ids),
            "plur_tokens": len(plur_ids),
            "token_match": token_match,
            "cosine_sim": round(cos_sim, 4),
        })

    single_single_pairs = [p for p in results["pairs"] if p["token_match"] == "single-single"]
    single_single_cossim = [p["cosine_sim"] for p in single_single_pairs]

    log_time(f"Total pairs: {len(SINGULAR_PLURAL_PAIRS)}")
    log_time(f"Single-single token pairs: {len(single_single_pairs)}/{len(SINGULAR_PLURAL_PAIRS)}")
    log_time(f"Mean cosine sim (single-single): {np.mean(single_single_cossim):.4f} +/- {np.std(single_single_cossim):.4f}")
    log_time(f"Min cosine sim: {np.min(single_single_cossim):.4f}, Max: {np.max(single_single_cossim):.4f}")

    # 1b. Tokenizer analysis: how many singular/plural words are single tokens?
    sing_single_count = sum(1 for sing, _ in SINGULAR_PLURAL_PAIRS if len(tokenizer.encode(sing, add_special_tokens=False)) == 1)
    plur_single_count = sum(1 for _, plur in SINGULAR_PLURAL_PAIRS if len(tokenizer.encode(plur, add_special_tokens=False)) == 1)
    both_single_count = len(single_single_pairs)
    log_time(f"Singular words as single token: {sing_single_count}/{len(SINGULAR_PLURAL_PAIRS)}")
    log_time(f"Plural words as single token: {plur_single_count}/{len(SINGULAR_PLURAL_PAIRS)}")
    log_time(f"Both as single token: {both_single_count}/{len(SINGULAR_PLURAL_PAIRS)}")

    results["summary"] = {
        "total_pairs": len(SINGULAR_PLURAL_PAIRS),
        "single_single_count": len(single_single_pairs),
        "sing_single_count": sing_single_count,
        "plur_single_count": plur_single_count,
        "mean_cosine_single_single": round(float(np.mean(single_single_cossim)), 4),
        "std_cosine_single_single": round(float(np.std(single_single_cossim)), 4),
        "min_cosine_single_single": round(float(np.min(single_single_cossim)), 4),
        "max_cosine_single_single": round(float(np.max(single_single_cossim)), 4),
    }

    # 1c. Cross-position probe: probe at VERB position instead of subject position
    log_time("Running verb-position probe (50 singular + 50 plural)...")

    verb_prompts_sing = [f"The {s} runs" for s in SINGULAR_SUBJECTS[:50]]
    verb_prompts_plur = [f"The {s} run" for s in PLURAL_SUBJECTS[:50]]

    all_hidden_at_verb = defaultdict(lambda: {"sing": [], "plur": []})

    for prompts, label in [(verb_prompts_sing, "sing"), (verb_prompts_plur, "plur")]:
        for pi, prompt in enumerate(prompts):
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attn_mask = inputs["attention_mask"].to(input_device)

                offset = get_special_token_offset(tokenizer, prompt)
                no_special_toks = tokenizer.encode(prompt, add_special_tokens=False)
                verb_pos = len(no_special_toks) - 1 + offset  # Last token position

                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                               output_hidden_states=True)

                if out.hidden_states:
                    for layer_idx, hs in enumerate(out.hidden_states):
                        hidden_vec = hs[0, verb_pos, :].float().cpu().numpy()
                        all_hidden_at_verb[layer_idx][label].append(hidden_vec)
            except Exception as e:
                continue

            if pi % 25 == 0:
                log_time(f"  Verb-position probe: {label} prompt {pi+1}/{len(prompts)}")

    # Train probe at verb position for each layer
    verb_layer_acc = {}
    for layer_idx in sorted(all_hidden_at_verb.keys()):
        sing_data = all_hidden_at_verb[layer_idx]["sing"]
        plur_data = all_hidden_at_verb[layer_idx]["plur"]
        if len(sing_data) < 5 or len(plur_data) < 5:
            continue

        X = np.array(sing_data + plur_data)
        y = np.array([1] * len(sing_data) + [0] * len(plur_data))

        try:
            probe = LogisticRegression(max_iter=1000, C=1.0)
            scores = cross_val_score(probe, X, y, cv=min(5, min(len(sing_data), len(plur_data))))
            verb_layer_acc[layer_idx] = round(float(np.mean(scores)), 4)
        except Exception:
            pass

    results["verb_position_probing"] = verb_layer_acc
    log_time(f"Verb position probe accuracy (L0, L1, L5, L10, L_last): " +
             f"L0={verb_layer_acc.get(0, 'N/A')}, L1={verb_layer_acc.get(1, 'N/A')}, " +
             f"L5={verb_layer_acc.get(5, 'N/A')}, L10={verb_layer_acc.get(10, 'N/A')}, " +
             f"L_last={verb_layer_acc.get(info.n_layers, 'N/A')}")

    # Save results
    out_path = RESULT_DIR / f"{model_name}_part1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 1 results saved to {out_path}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ============================================================
# Part 2: GLM4 L0->L1 Mutation Decomposition (HIGHEST PRIORITY)
# ============================================================

def run_part2(model_name):
    """Decompose GLM4 L1 to find which component creates number information."""
    import torch
    from model_utils import get_layers
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    log_time(f"=== Part 2: L0->L1 Mutation Decomposition for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_safe(model_name)
    input_device = get_input_device(model)
    layers = get_layers(model)

    results = {
        "model": model_name,
        "n_layers": info.n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
    }

    # Prepare prompts
    n_prompts = 60
    sing_prompts = [(f"The {SINGULAR_SUBJECTS[i]} sits", "sing") for i in range(min(n_prompts, len(SINGULAR_SUBJECTS)))]
    plur_prompts = [(f"The {PLURAL_SUBJECTS[i]} sit", "plur") for i in range(min(n_prompts, len(PLURAL_SUBJECTS)))]
    all_prompts = sing_prompts + plur_prompts
    log_time(f"Total prompts: {len(all_prompts)} ({len(sing_prompts)} sing + {len(plur_prompts)} plur)")

    # 2a. Decompose L0/L1: collect intermediate states
    # For each prompt, collect:
    #   - embedding (L0 input)
    #   - L0 attention output (post-attn, pre-MLP)
    #   - L0 full output (post-MLP = L1 input)
    #   - L1 attention output (post-attn, pre-MLP)
    #   - L1 full output (post-MLP = L2 input)

    # We need hooks to capture intermediate states
    intermediate_data = {
        "embed": {"sing": [], "plur": []},         # L0 input (embedding)
        "L0_attn_out": {"sing": [], "plur": []},   # L0 attn output + residual
        "L0_mlp_out": {"sing": [], "plur": []},    # L0 full output = L1 input
        "L1_attn_out": {"sing": [], "plur": []},   # L1 attn output + residual
        "L1_mlp_out": {"sing": [], "plur": []},    # L1 full output = L2 input
    }

    for pi, (prompt, label) in enumerate(all_prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        offset = get_special_token_offset(tokenizer, prompt)
        no_special_toks = tokenizer.encode(prompt, add_special_tokens=False)

        # Subject is the 2nd token (index 1, before "The")
        # After special token offset: position 0+offset and 1+offset
        # "The cat sits" -> tokens: [The, cat, sits] -> subject at position 1+offset
        subj_pos = 1 + offset  # Subject token position

        captured = {}

        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook

        # Register hooks on L0 and L1
        hooks = []
        # L0 self-attention output (before MLP)
        hooks.append(layers[0].self_attn.register_forward_hook(make_hook("L0_attn_raw")))
        # L0 MLP output (after MLP) - the full layer output
        hooks.append(layers[0].register_forward_hook(make_hook("L0_full")))
        # L1 self-attention output
        hooks.append(layers[1].self_attn.register_forward_hook(make_hook("L1_attn_raw")))
        # L1 full output
        hooks.append(layers[1].register_forward_hook(make_hook("L1_full")))

        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_hidden_states=True)
        except Exception as e:
            log_time(f"  Prompt {pi} failed: {str(e)[:60]}")
            for h in hooks:
                h.remove()
            continue

        for h in hooks:
            h.remove()

        # Extract subject hidden states at each intermediate point
        # hidden_states[0] = embedding output, hidden_states[1] = L0 output, etc.
        hs = out.hidden_states

        # Embedding (L0 input)
        if hs and len(hs) > 0:
            embed_h = hs[0][0, subj_pos, :].float().cpu().numpy()
            intermediate_data["embed"][label].append(embed_h)

        # L0 attn output: residual + attn_output = pre-MLP hidden state
        # We need to reconstruct: L0_attn_out = L0_input + L0_attn_output
        # captured["L0_attn_raw"] is the attn output (before adding residual + layernorm)
        # Actually, the hook captures the output of self_attn module, which already includes
        # the residual connection in most implementations
        # Let's use hidden_states to be safe:
        # hs[0] = embedding output
        # hs[1] = L0 output (after attn + MLP)
        # We need the state after L0 attn but before L0 MLP

        # For accurate decomposition, use captured hooks:
        # captured["L0_attn_raw"] = output of self_attn module
        # captured["L0_full"] = output of entire L0 block

        if "L0_attn_raw" in captured:
            l0_attn_h = captured["L0_attn_raw"][0, subj_pos, :].numpy()
            intermediate_data["L0_attn_out"][label].append(l0_attn_h)

        if "L0_full" in captured:
            l0_full_h = captured["L0_full"][0, subj_pos, :].numpy()
            intermediate_data["L0_mlp_out"][label].append(l0_full_h)

        if "L1_attn_raw" in captured:
            l1_attn_h = captured["L1_attn_raw"][0, subj_pos, :].numpy()
            intermediate_data["L1_attn_out"][label].append(l1_attn_h)

        if "L1_full" in captured:
            l1_full_h = captured["L1_full"][0, subj_pos, :].numpy()
            intermediate_data["L1_mlp_out"][label].append(l1_full_h)

        if pi % 20 == 0:
            log_time(f"  Processed prompt {pi+1}/{len(all_prompts)}, captured keys: {list(captured.keys())}")

    # 2b. Probe each intermediate state
    log_time("Training probes on each intermediate state...")
    probe_results = {}

    for state_name in ["embed", "L0_attn_out", "L0_mlp_out", "L1_attn_out", "L1_mlp_out"]:
        sing_data = intermediate_data[state_name]["sing"]
        plur_data = intermediate_data[state_name]["plur"]

        if len(sing_data) < 5 or len(plur_data) < 5:
            log_time(f"  {state_name}: insufficient data (sing={len(sing_data)}, plur={len(plur_data)})")
            probe_results[state_name] = {"accuracy": None, "n_samples": len(sing_data) + len(plur_data)}
            continue

        X = np.array(sing_data + plur_data)
        y = np.array([1] * len(sing_data) + [0] * len(plur_data))

        try:
            probe = LogisticRegression(max_iter=1000, C=1.0)
            n_cv = min(5, min(len(sing_data), len(plur_data)))
            scores = cross_val_score(probe, X, y, cv=n_cv)
            acc = float(np.mean(scores))
            std = float(np.std(scores))
            probe_results[state_name] = {
                "accuracy": round(acc, 4),
                "std": round(std, 4),
                "n_sing": len(sing_data),
                "n_plur": len(plur_data),
            }
            log_time(f"  {state_name}: accuracy={acc:.4f} +/- {std:.4f} (n={len(sing_data)+len(plur_data)})")
        except Exception as e:
            log_time(f"  {state_name}: probe failed: {str(e)[:60]}")
            probe_results[state_name] = {"accuracy": None, "error": str(e)[:60]}

    results["intermediate_probing"] = probe_results

    # Calculate incremental changes
    if all(probe_results.get(k, {}).get("accuracy") for k in ["embed", "L0_attn_out", "L0_mlp_out", "L1_attn_out", "L1_mlp_out"]):
        embed_acc = probe_results["embed"]["accuracy"]
        l0_attn_acc = probe_results["L0_attn_out"]["accuracy"]
        l0_mlp_acc = probe_results["L0_mlp_out"]["accuracy"]
        l1_attn_acc = probe_results["L1_attn_out"]["accuracy"]
        l1_mlp_acc = probe_results["L1_mlp_out"]["accuracy"]

        results["incremental_changes"] = {
            "embed_to_L0_attn": round(l0_attn_acc - embed_acc, 4),
            "L0_attn_to_L0_mlp": round(l0_mlp_acc - l0_attn_acc, 4),
            "L0_mlp_to_L1_attn": round(l1_attn_acc - l0_mlp_acc, 4),
            "L1_attn_to_L1_mlp": round(l1_mlp_acc - l1_attn_acc, 4),
            "total_L0_change": round(l0_mlp_acc - embed_acc, 4),
            "total_L1_change": round(l1_mlp_acc - l0_mlp_acc, 4),
        }
        log_time(f"Incremental: embed->L0_attn={results['incremental_changes']['embed_to_L0_attn']:+.4f}, "
                 f"L0_attn->L0_mlp={results['incremental_changes']['L0_attn_to_L0_mlp']:+.4f}, "
                 f"L0_mlp->L1_attn={results['incremental_changes']['L0_mlp_to_L1_attn']:+.4f}, "
                 f"L1_attn->L1_mlp={results['incremental_changes']['L1_attn_to_L1_mlp']:+.4f}")

    # 2c. L0 and L1 attention pattern analysis
    log_time("Analyzing L0 and L1 attention patterns...")
    attn_analysis = {"L0": {}, "L1": {}}

    test_prompts_for_attn = [
        "The cat sits",
        "The cats sit",
        "The dog runs",
        "The dogs run",
        "The child walks",
        "The children walk",
    ]

    for layer_idx in [0, 1]:
        layer_attn_data = defaultdict(list)

        for prompt in test_prompts_for_attn:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)

            captured_attn = {}
            def make_attn_hook(key):
                def hook(module, input, output):
                    # output = (attn_output, attn_weights, past_key_value)
                    if isinstance(output, tuple) and len(output) >= 2:
                        if output[1] is not None:
                            captured_attn[key] = output[1].detach().float().cpu()
                return hook

            hook = layers[layer_idx].self_attn.register_forward_hook(make_attn_hook("attn_w"))

            try:
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attn_mask)
            except Exception:
                pass

            hook.remove()

            if "attn_w" not in captured_attn:
                continue

            attn_w = captured_attn["attn_w"]  # [1, n_heads, seq, seq]
            if attn_w.shape[0] == 0:
                continue

            offset = get_special_token_offset(tokenizer, prompt)
            no_special_toks = tokenizer.encode(prompt, add_special_tokens=False)
            all_toks_with_special = input_ids[0].cpu().tolist()
            verb_pos = len(no_special_toks) - 1 + offset

            for h in range(min(n_heads, attn_w.shape[1])):
                head_attn = attn_w[0, h, verb_pos, :].numpy()

                # Find top-3 attended positions from verb
                top3_pos = np.argsort(head_attn)[-3:][::-1]
                top3_info = []
                for pos in top3_pos:
                    tok_str = safe_decode(tokenizer, int(all_toks_with_special[pos])) if pos < len(all_toks_with_special) else "<pad>"
                    top3_info.append({
                        "pos": int(pos),
                        "token": tok_str,
                        "weight": round(float(head_attn[pos]), 4),
                    })

                layer_attn_data[h].append({
                    "prompt": prompt,
                    "top3_from_verb": top3_info,
                    "attn_to_subj": round(float(head_attn[1 + offset]), 4) if 1 + offset < len(head_attn) else 0,
                    "attn_to_verb_itself": round(float(head_attn[verb_pos]), 4),
                })

        # Summarize attention patterns for this layer
        for h, data_list in layer_attn_data.items():
            if not data_list:
                continue
            avg_subj_attn = np.mean([d["attn_to_subj"] for d in data_list])
            avg_top1_is_subj = np.mean([1 if d["top3_from_verb"][0]["pos"] == 1 + offset else 0
                                        for d in data_list])
            attn_analysis[f"L{layer_idx}"][f"H{h}"] = {
                "avg_subj_attn_from_verb": round(float(avg_subj_attn), 4),
                "avg_top1_is_subj": round(float(avg_top1_is_subj), 4),
                "n_prompts": len(data_list),
                "sample_prompt": data_list[0] if data_list else None,
            }

    # Report most interesting heads
    for layer_key in ["L0", "L1"]:
        log_time(f"\n{layer_key} attention heads (sorted by subj_attn):")
        heads_sorted = sorted(attn_analysis[layer_key].items(),
                             key=lambda x: x[1].get("avg_subj_attn_from_verb", 0), reverse=True)
        for h_name, h_data in heads_sorted[:5]:
            log_time(f"  {h_name}: subj_attn={h_data['avg_subj_attn_from_verb']:.4f}, "
                     f"top1_is_subj={h_data['avg_top1_is_subj']:.2f}")

    results["attn_analysis"] = attn_analysis

    # 2d. Full layer-by-layer probe (for reference, including verb position)
    log_time("Running full layer-by-layer probe at subject position...")
    layer_probe_acc = {}

    for pi, (prompt, label) in enumerate(all_prompts[:40]):  # Use 40 prompts for speed
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)

        offset = get_special_token_offset(tokenizer, prompt)
        subj_pos = 1 + offset

        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_hidden_states=True)
        except Exception:
            continue

        if out.hidden_states:
            for layer_idx, hs in enumerate(out.hidden_states):
                hidden_vec = hs[0, subj_pos, :].float().cpu().numpy()
                if layer_idx not in layer_probe_acc:
                    layer_probe_acc[layer_idx] = {"sing": [], "plur": []}
                layer_probe_acc[layer_idx][label].append(hidden_vec)

    layer_acc_summary = {}
    for layer_idx in sorted(layer_probe_acc.keys()):
        sing_data = layer_probe_acc[layer_idx]["sing"]
        plur_data = layer_probe_acc[layer_idx]["plur"]
        if len(sing_data) < 5 or len(plur_data) < 5:
            continue
        X = np.array(sing_data + plur_data)
        y = np.array([1] * len(sing_data) + [0] * len(plur_data))
        try:
            probe = LogisticRegression(max_iter=1000, C=1.0)
            scores = cross_val_score(probe, X, y, cv=min(5, min(len(sing_data), len(plur_data))))
            layer_acc_summary[layer_idx] = round(float(np.mean(scores)), 4)
        except Exception:
            pass

    results["layer_probe_at_subject"] = layer_acc_summary

    # Print key layers
    for l in [0, 1, 2, 5, 10, info.n_layers // 2, info.n_layers - 2, info.n_layers]:
        if l in layer_acc_summary:
            log_time(f"  Subject position probe: L{l} = {layer_acc_summary[l]:.4f}")

    # Save results
    out_path = RESULT_DIR / f"{model_name}_part2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 2 results saved to {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ============================================================
# Part 3: L34_H15 Information Sink Analysis (Qwen3)
# ============================================================

def run_part3(model_name):
    """Analyze L34_H15's function: what does it write when attending first token?"""
    import torch
    from model_utils import get_layers

    log_time(f"=== Part 3: Information Sink Analysis for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_safe(model_name)
    W_U = get_W_U_safe(model, model_name)
    layers = get_layers(model)
    input_device = get_input_device(model)

    results = {"model": model_name, "n_layers": info.n_layers}

    # Test prompts: singular vs plural pairs
    test_prompts = [
        ("The cat runs", "The cats run", "cat", "cats"),
        ("The dog walks", "The dogs walk", "dog", "dogs"),
        ("The child sits", "The children sit", "child", "children"),
        ("The woman speaks", "The women speak", "woman", "women"),
        ("The bird flies", "The birds fly", "bird", "birds"),
        ("The teacher writes", "The teachers write", "teacher", "teachers"),
        ("The apple falls", "The apples fall", "apple", "apples"),
        ("The king rules", "The kings rule", "king", "kings"),
    ]

    # Grammar heads to analyze
    if model_name == "qwen3":
        analysis_heads = [
            (34, 15, "L34_H15_infra"),
            (35, 8, "L35_H8_driver"),
            (31, 17, "L31_H17_subj_loc"),
            (35, 0, "L35_H0_follower"),
        ]
    elif model_name == "glm4":
        analysis_heads = [
            (38, 28, "L38_H28_infra"),
            (39, 25, "L39_H25_driver"),
        ]
    elif model_name == "deepseek7b":
        analysis_heads = [
            (27, 7, "L27_H7_infra"),
            (27, 9, "L27_H9_driver"),
        ]
    else:
        analysis_heads = []

    head_analysis = {}

    for li, h, h_label in analysis_heads:
        log_time(f"Analyzing {h_label}...")

        # Try to get W_O and W_V weights; if meta tensor, use hook-only approach
        use_weight_method = True
        W_O_h = None
        W_V_h = None
        try:
            w_o = layers[li].self_attn.o_proj.weight
            if hasattr(w_o, 'is_meta') and w_o.is_meta:
                use_weight_method = False
                log_time(f"  {h_label}: o_proj weight on meta device, using hook-only method")
            else:
                W_O = safe_weight_to_numpy(w_o, model_name, f"model.layers.{li}.self_attn.o_proj.weight")
                w_v = layers[li].self_attn.v_proj.weight
                if hasattr(w_v, 'is_meta') and w_v.is_meta:
                    use_weight_method = False
                    log_time(f"  {h_label}: v_proj weight on meta device, using hook-only method")
                else:
                    W_V = safe_weight_to_numpy(w_v, model_name, f"model.layers.{li}.self_attn.v_proj.weight")
                    W_O_h = W_O[:, h * head_dim:(h + 1) * head_dim]
                    W_V_h = get_W_V_h(W_V, h, head_dim, kv_group_size)
        except Exception as e:
            use_weight_method = False
            log_time(f"  {h_label}: weight access failed ({str(e)[:50]}), using hook-only method")

        head_data = {"sing": [], "plur": []}

        for sing_prompt, plur_prompt, sing_subj, plur_subj in test_prompts:
            for prompt, label in [(sing_prompt, "sing"), (plur_prompt, "plur")]:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attn_mask = inputs["attention_mask"].to(input_device)

                offset = get_special_token_offset(tokenizer, prompt)
                no_special_toks = tokenizer.encode(prompt, add_special_tokens=False)
                verb_pos = len(no_special_toks) - 1 + offset
                subj_pos = 1 + offset
                first_pos = offset  # First real token position

                # Capture attention weights and residual stream
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

                hooks_list = []
                hooks_list.append(layers[li].self_attn.register_forward_hook(make_hook(f"L{li}_attn")))
                hooks_list.append(layers[li].register_forward_hook(make_hook(f"L{li}_full")))

                try:
                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attn_mask,
                                   output_hidden_states=True)
                except Exception:
                    for hh in hooks_list:
                        hh.remove()
                    continue

                for hh in hooks_list:
                    hh.remove()

                all_toks_with_special = input_ids[0].cpu().tolist()

                # Get attention pattern for this head
                attn_from_verb = None
                if f"L{li}_attn_attn" in captured:
                    attn_w = captured[f"L{li}_attn_attn"]
                    if h < attn_w.shape[1]:
                        attn_from_verb = attn_w[0, h, verb_pos, :].numpy()

                # Get residual stream before this layer
                resid_pre = None
                if out.hidden_states and li < len(out.hidden_states):
                    resid_pre = out.hidden_states[li][0].float().cpu().numpy()

                # Compute head's output at each position
                head_logit_effects = {}

                if use_weight_method and resid_pre is not None:
                    # Method 1: Use weights directly (only when available)
                    for pos_name, pos_idx in [("verb", verb_pos), ("subj", subj_pos), ("first", first_pos)]:
                        attended_hidden = resid_pre[pos_idx, :]  # [d_model]
                        head_out = W_O_h @ (W_V_h @ attended_hidden)  # [d_model]
                        logit_effect = W_U @ head_out  # [vocab]

                        top10_idx = np.argsort(logit_effect)[-10:][::-1]
                        top10 = [(safe_decode(tokenizer, int(i)), round(float(logit_effect[i]), 4))
                                for i in top10_idx]

                        sing_verbs = ["runs", "walks", "sits", "speaks", "flies", "writes", "falls", "rules"]
                        plur_verbs = ["run", "walk", "sit", "speak", "fly", "write", "fall", "rule"]

                        sing_verb_effect = float(np.mean([logit_effect[safe_get_token_id(tokenizer, v)] for v in sing_verbs if safe_get_token_id(tokenizer, v) is not None]))
                        plur_verb_effect = float(np.mean([logit_effect[safe_get_token_id(tokenizer, v)] for v in plur_verbs if safe_get_token_id(tokenizer, v) is not None]))

                        head_logit_effects[pos_name] = {
                            "top10": top10[:5],
                            "sing_verb_effect": round(sing_verb_effect, 4),
                            "plur_verb_effect": round(plur_verb_effect, 4),
                            "verb_diff": round(sing_verb_effect - plur_verb_effect, 4),
                        }
                elif resid_pre is not None:
                    # Method 2: Hook-based - use difference between layer output and input
                    # attn_output = L{li}_out - resid_pre (at each position)
                    if f"L{li}_attn_out" in captured:
                        attn_out = captured[f"L{li}_attn_out"][0].numpy()  # [seq, d_model]
                        # The attention output already includes residual connection
                        # So attn_out - resid_pre = pure attention contribution
                        # But we need only head h's contribution
                        # Since we can't isolate a single head without W_O, we use the full attn output
                        # and project through W_U for verb position
                        for pos_name, pos_idx in [("verb", verb_pos)]:
                            full_attn_contribution = attn_out[pos_idx] - resid_pre[pos_idx]
                            logit_effect = W_U @ full_attn_contribution

                            top10_idx = np.argsort(logit_effect)[-10:][::-1]
                            top10 = [(safe_decode(tokenizer, int(i)), round(float(logit_effect[i]), 4))
                                    for i in top10_idx]

                            sing_verbs = ["runs", "walks", "sits", "speaks", "flies", "writes", "falls", "rules"]
                            plur_verbs = ["run", "walk", "sit", "speak", "fly", "write", "fall", "rule"]

                            sing_verb_effect = float(np.mean([logit_effect[safe_get_token_id(tokenizer, v)] for v in sing_verbs if safe_get_token_id(tokenizer, v) is not None]))
                            plur_verb_effect = float(np.mean([logit_effect[safe_get_token_id(tokenizer, v)] for v in plur_verbs if safe_get_token_id(tokenizer, v) is not None]))

                            head_logit_effects[pos_name] = {
                                "top10": top10[:5],
                                "sing_verb_effect": round(sing_verb_effect, 4),
                                "plur_verb_effect": round(plur_verb_effect, 4),
                                "verb_diff": round(sing_verb_effect - plur_verb_effect, 4),
                                "note": "full_attn_contribution (all heads, not just h)",
                            }

                head_data[label].append({
                    "prompt": prompt,
                    "attn_from_verb": {
                        "to_first": round(float(attn_from_verb[first_pos]), 4) if attn_from_verb is not None and first_pos < len(attn_from_verb) else None,
                        "to_subj": round(float(attn_from_verb[subj_pos]), 4) if attn_from_verb is not None and subj_pos < len(attn_from_verb) else None,
                        "to_verb": round(float(attn_from_verb[verb_pos]), 4) if attn_from_verb is not None and verb_pos < len(attn_from_verb) else None,
                    },
                    "logit_effects": head_logit_effects,
                })

        # Summarize
        sing_verb_diffs = [d["logit_effects"]["verb"]["verb_diff"] for d in head_data["sing"] if "verb" in d.get("logit_effects", {})]
        plur_verb_diffs = [d["logit_effects"]["verb"]["verb_diff"] for d in head_data["plur"] if "verb" in d.get("logit_effects", {})]

        summary = {
            "head": h_label,
            "n_prompts": len(head_data["sing"]) + len(head_data["plur"]),
        }
        if sing_verb_diffs and plur_verb_diffs:
            summary["avg_verb_diff_sing"] = round(float(np.mean(sing_verb_diffs)), 4)
            summary["avg_verb_diff_plur"] = round(float(np.mean(plur_verb_diffs)), 4)
            log_time(f"  {h_label}: verb_diff(sing)={summary['avg_verb_diff_sing']:.4f}, "
                     f"verb_diff(plur)={summary['avg_verb_diff_plur']:.4f}")

        # Average attention from verb to each position
        for pos in ["to_first", "to_subj", "to_verb"]:
            sing_attns = [d["attn_from_verb"][pos] for d in head_data["sing"] if d["attn_from_verb"].get(pos) is not None]
            plur_attns = [d["attn_from_verb"][pos] for d in head_data["plur"] if d["attn_from_verb"].get(pos) is not None]
            if sing_attns:
                summary[f"avg_attn_{pos}_sing"] = round(float(np.mean(sing_attns)), 4)
            if plur_attns:
                summary[f"avg_attn_{pos}_plur"] = round(float(np.mean(plur_attns)), 4)

        head_analysis[h_label] = summary
        log_time(f"  {h_label}: attn_to_first={summary.get('avg_attn_to_first_sing', 'N/A')}, "
                 f"attn_to_subj={summary.get('avg_attn_to_subj_sing', 'N/A')}")

    results["head_analysis"] = head_analysis

    # 3b. First token hidden state probing: does first token carry number info?
    log_time("Probing first token position for number information...")
    first_token_probe_data = {"sing": [], "plur": []}

    for si in range(min(50, len(SINGULAR_SUBJECTS))):
        for subj_list, label in [(SINGULAR_SUBJECTS, "sing"), (PLURAL_SUBJECTS, "plur")]:
            prompt = f"The {subj_list[si]} sits" if label == "sing" else f"The {subj_list[si]} sit"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attn_mask = inputs["attention_mask"].to(input_device)
            offset = get_special_token_offset(tokenizer, prompt)
            first_pos = offset  # First real token ("The")

            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                               output_hidden_states=True)
            except Exception:
                continue

            if out.hidden_states:
                for layer_idx in [0, 5, 10, 15, 20, 25, 30, 34, 35]:
                    if layer_idx < len(out.hidden_states):
                        hidden_vec = out.hidden_states[layer_idx][0, first_pos, :].float().cpu().numpy()
                        if layer_idx not in first_token_probe_data:
                            first_token_probe_data[layer_idx] = {"sing": [], "plur": []}
                        first_token_probe_data[layer_idx][label].append(hidden_vec)

    first_token_probe_acc = {}
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    for layer_idx in sorted([k for k in first_token_probe_data.keys() if isinstance(k, int)]):
        sing_data = first_token_probe_data[layer_idx]["sing"]
        plur_data = first_token_probe_data[layer_idx]["plur"]
        if len(sing_data) < 5 or len(plur_data) < 5:
            continue
        X = np.array(sing_data + plur_data)
        y = np.array([1] * len(sing_data) + [0] * len(plur_data))
        try:
            probe = LogisticRegression(max_iter=1000, C=1.0)
            scores = cross_val_score(probe, X, y, cv=min(5, min(len(sing_data), len(plur_data))))
            first_token_probe_acc[layer_idx] = round(float(np.mean(scores)), 4)
        except Exception:
            pass

    results["first_token_probe"] = first_token_probe_acc
    log_time("First token position probe accuracy:")
    for l, acc in sorted(first_token_probe_acc.items()):
        log_time(f"  L{l}: {acc:.4f}")

    # Save results
    out_path = RESULT_DIR / f"{model_name}_part3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 3 results saved to {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def safe_get_token_id(tokenizer, text):
    """Safely get token ID for a text string."""
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    except Exception:
        pass
    return None


# ============================================================
# Part 4: DS7B Number Information Decay Analysis
# ============================================================

def run_part4(model_name):
    """Test if DS7B's late-layer number info decay is real or just nonlinear encoding."""
    import torch
    from model_utils import get_layers
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPClassifier

    log_time(f"=== Part 4: Number Info Decay Analysis for {model_name} ===")

    model, tokenizer, info, n_heads, head_dim, n_kv_heads, kv_group_size = load_model_safe(model_name)
    input_device = get_input_device(model)

    results = {"model": model_name, "n_layers": info.n_layers}

    # Collect subject position hidden states
    n_prompts = 50
    sing_prompts = [f"The {SINGULAR_SUBJECTS[i]} sits" for i in range(min(n_prompts, len(SINGULAR_SUBJECTS)))]
    plur_prompts = [f"The {PLURAL_SUBJECTS[i]} sit" for i in range(min(n_prompts, len(PLURAL_SUBJECTS)))]

    layer_data = defaultdict(lambda: {"sing": [], "plur": []})

    all_prompts = [(p, "sing") for p in sing_prompts] + [(p, "plur") for p in plur_prompts]

    for pi, (prompt, label) in enumerate(all_prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attn_mask = inputs["attention_mask"].to(input_device)
        offset = get_special_token_offset(tokenizer, prompt)
        subj_pos = 1 + offset

        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_hidden_states=True)
        except Exception:
            continue

        if out.hidden_states:
            for layer_idx, hs in enumerate(out.hidden_states):
                hidden_vec = hs[0, subj_pos, :].float().cpu().numpy()
                layer_data[layer_idx][label].append(hidden_vec)

        if pi % 25 == 0:
            log_time(f"  Collected hidden states: prompt {pi+1}/{len(all_prompts)}")

    # 4a. Linear probe vs MLP probe for all layers
    log_time("Training linear and MLP probes for each layer...")
    probe_comparison = {}

    for layer_idx in sorted(layer_data.keys()):
        sing_data = layer_data[layer_idx]["sing"]
        plur_data = layer_data[layer_idx]["plur"]
        if len(sing_data) < 5 or len(plur_data) < 5:
            continue

        X = np.array(sing_data + plur_data)
        y = np.array([1] * len(sing_data) + [0] * len(plur_data))
        n_cv = min(5, min(len(sing_data), len(plur_data)))

        # Linear probe
        try:
            lin_probe = LogisticRegression(max_iter=1000, C=1.0)
            lin_scores = cross_val_score(lin_probe, X, y, cv=n_cv)
            lin_acc = float(np.mean(lin_scores))
        except Exception:
            lin_acc = None

        # MLP probe (2-layer, small)
        try:
            mlp_probe = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
            mlp_scores = cross_val_score(mlp_probe, X, y, cv=n_cv)
            mlp_acc = float(np.mean(mlp_scores))
        except Exception:
            mlp_acc = None

        probe_comparison[layer_idx] = {
            "linear_acc": round(lin_acc, 4) if lin_acc is not None else None,
            "mlp_acc": round(mlp_acc, 4) if mlp_acc is not None else None,
            "gap": round(mlp_acc - lin_acc, 4) if mlp_acc is not None and lin_acc is not None else None,
            "n_samples": len(sing_data) + len(plur_data),
        }

        if layer_idx in [0, 1, 5, 10, 15, 20, 22, 25, 27] or layer_idx == info.n_layers:
            lin_str = f"{lin_acc:.4f}" if lin_acc is not None else "N/A"
            mlp_str = f"{mlp_acc:.4f}" if mlp_acc is not None else "N/A"
            gap_str = f"{mlp_acc-lin_acc:+.4f}" if mlp_acc is not None and lin_acc is not None else "N/A"
            log_time(f"  L{layer_idx}: linear={lin_str}, mlp={mlp_str}, gap={gap_str}")

    results["probe_comparison"] = probe_comparison

    # 4b. Number direction stability across layers
    log_time("Analyzing number direction stability across layers...")
    direction_data = {}

    # Load number directions from Phase 259 if available
    p259_dir = Path("results/phase259_two_stage_probing")
    direction_file = p259_dir / f"{model_name}_number_directions.npz"

    if direction_file.exists():
        npz = np.load(str(direction_file))
        log_time(f"Loaded Phase 259 number directions from {direction_file}")

        # Check alignment between early and late layer directions
        early_layers = [0, 1, 5, 10]
        late_layers = [max(0, info.n_layers - 5 + i) for i in range(5)]

        for el in early_layers:
            for ll in late_layers:
                key_e = f"L{el}"
                key_l = f"L{ll}"
                if key_e in npz and key_l in npz:
                    dir_e = npz[key_e]
                    dir_l = npz[key_l]
                    cos_sim = float(np.dot(dir_e, dir_l) / (np.linalg.norm(dir_e) * np.linalg.norm(dir_l) + 1e-10))
                    log_time(f"  Direction alignment L{el} vs L{ll}: {cos_sim:.4f}")
                    direction_data[f"L{el}_vs_L{ll}"] = round(cos_sim, 4)

    results["direction_stability"] = direction_data

    # 4c. Hidden state distance analysis: sing vs plur separation
    log_time("Computing sing/plur hidden state separation per layer...")
    separation_data = {}

    for layer_idx in sorted(layer_data.keys()):
        sing_data = np.array(layer_data[layer_idx]["sing"])
        plur_data = np.array(layer_data[layer_idx]["plur"])
        if len(sing_data) < 3 or len(plur_data) < 3:
            continue

        # Center of mass for each class
        sing_mean = np.mean(sing_data, axis=0)
        plur_mean = np.mean(plur_data, axis=0)

        # Inter-class distance
        inter_dist = float(np.linalg.norm(sing_mean - plur_mean))

        # Intra-class variance
        sing_var = float(np.mean(np.linalg.norm(sing_data - sing_mean, axis=1)))
        plur_var = float(np.mean(np.linalg.norm(plur_data - plur_mean, axis=1)))

        # Signal-to-noise ratio
        snr = inter_dist / (sing_var + plur_var + 1e-10)

        separation_data[layer_idx] = {
            "inter_dist": round(inter_dist, 4),
            "sing_var": round(sing_var, 4),
            "plur_var": round(plur_var, 4),
            "snr": round(snr, 4),
        }

    results["separation"] = separation_data

    # Report key layers
    for l in [0, 1, 5, 10, 15, 20, 22, 25, 27]:
        if l in separation_data:
            log_time(f"  L{l}: inter_dist={separation_data[l]['inter_dist']:.4f}, "
                     f"SNR={separation_data[l]['snr']:.4f}")

    # Save results
    out_path = RESULT_DIR / f"{model_name}_part4.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Part 4 results saved to {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 260: L1 Mechanism & Embedding Analysis")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--part", type=str, required=True, choices=["1", "2", "3", "4", "all"])
    args = parser.parse_args()

    model_name = args.model

    if args.part == "1":
        run_part1(model_name)
    elif args.part == "2":
        run_part2(model_name)
    elif args.part == "3":
        run_part3(model_name)
    elif args.part == "4":
        run_part4(model_name)
    elif args.part == "all":
        log_time(f"Running all parts for {model_name}...")
        run_part1(model_name)
        gc.collect()
        run_part2(model_name)
        gc.collect()
        run_part3(model_name)
        gc.collect()
        run_part4(model_name)

    log_time("Phase 260 complete!")


if __name__ == "__main__":
    main()
