"""
Phase 323b: Confirmation Test — Attention Ablation Paradox & Attribute Decomposition
==================================================================================

Two critical confirmations:
1. "Attention ablation ENHANCES causal efficacy" — is this stable?
2. Attribute direction decomposition: slot vs type vs value

Test 1: Verify attention ablation paradox
- Run ablation at multiple layers (not just optimal)
- Use multiple alpha values
- Compare with random ablation control

Test 2: Attribute decomposition
- "The apple is something" → "The apple has a property" (slot)
- "The apple has a property" → "The apple has a color" (type)
- "The apple has a color" → "The apple is red" (value)
- Measure each level's contribution to logit change

Usage:
  python tests/glm5/phase323b_attn_paradox.py qwen3
  python tests/glm5/phase323b_attn_paradox.py glm4
  python tests/glm5/phase323b_attn_paradox.py deepseek7b
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

RESULT_DIR = Path("results/phase323b_attn_paradox")
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


# ===== Attribute Decomposition Templates =====
# Level 0 (baseline): neutral object reference
# Level 1 (slot): has a property/attribute
# Level 2 (type): has a color/taste/temperature/texture
# Level 3 (value): is red/sweet/hot/smooth

DECOMPOSITION_TEMPLATES = {
    "color": {
        "base": "The apple is something",
        "slot": "The apple has a property",
        "type": "The apple has a color",
        "value_pos": "The apple is red",
        "value_neg": "The apple is just an object",
    },
    "taste": {
        "base": "The lemon is something",
        "slot": "The lemon has a property",
        "type": "The lemon has a taste",
        "value_pos": "The lemon is sour",
        "value_neg": "The lemon is just an object",
    },
    "temperature": {
        "base": "The fire is something",
        "slot": "The fire has a property",
        "type": "The fire has a temperature",
        "value_pos": "The fire is hot",
        "value_neg": "The fire is just an object",
    },
    "texture": {
        "base": "The silk is something",
        "slot": "The silk has a property",
        "type": "The silk has a texture",
        "value_pos": "The silk is smooth",
        "value_neg": "The silk is just an object",
    },
}

# For transfer test
TRANSFER_TEMPLATES = {
    "color": [
        ("apple", "red", "strawberry"),
        ("sky", "blue", "ocean"),
        ("fire", "orange", "sunset"),
    ],
    "taste": [
        ("lemon", "sour", "grapefruit"),
        ("honey", "sweet", "candy"),
        ("salt", "salty", "soy sauce"),
    ],
    "temperature": [
        ("fire", "hot", "stove"),
        ("ice", "cold", "frost"),
        ("sun", "warm", "blanket"),
    ],
}

ATTRIBUTE_PAIRS = [
    ("apple", "red", "color"),
    ("sky", "blue", "color"),
    ("fire", "hot", "temperature"),
    ("ice", "cold", "temperature"),
    ("lemon", "sour", "taste"),
    ("honey", "sweet", "taste"),
    ("silk", "smooth", "texture"),
    ("sandpaper", "rough", "texture"),
]

WORD_CLUSTERS = {
    "color": ["red", "blue", "green", "yellow", "white", "black", "orange", "purple"],
    "temperature": ["hot", "cold", "warm", "cool", "freezing", "boiling"],
    "taste": ["sweet", "sour", "bitter", "salty", "spicy"],
    "texture": ["smooth", "rough", "soft", "hard", "sharp"],
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


def inject_direction_at_layer(model, tokenizer, device, prompt, direction, layer_idx, alpha):
    """Inject direction at a specific layer and return logits."""
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


def ablate_component_at_layer(model, tokenizer, device, prompt, layer_idx, component,
                               direction=None, alpha=0):
    """Ablate a component and optionally inject direction."""
    layers_list = get_layers(model)
    layer = layers_list[layer_idx]
    hooks = []

    def make_ablation_hook():
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            zeroed = torch.zeros_like(hidden)
            if isinstance(output, tuple):
                return (zeroed,) + output[1:]
            return zeroed
        return hook_fn

    if component in ('attn', 'both') and hasattr(layer, 'self_attn'):
        hooks.append(layer.self_attn.register_forward_hook(make_ablation_hook()))
    if component in ('mlp', 'both') and hasattr(layer, 'mlp'):
        hooks.append(layer.mlp.register_forward_hook(make_ablation_hook()))

    if direction is not None:
        def inject_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            d_tensor = torch.tensor(direction, dtype=hidden.dtype, device=hidden.device)
            hidden_modified = hidden.clone()
            hidden_modified[0, -1, :] += (alpha * d_tensor).to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden_modified,) + output[1:]
            return hidden_modified
        hooks.append(layers_list[layer_idx].register_forward_hook(inject_fn))

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    try:
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[0, -1].float().cpu().numpy()
    finally:
        for h in hooks:
            h.remove()
    return logits


def extract_rep_at_layer(model, tokenizer, device, sentence, target_layer):
    """Extract representation at a single layer."""
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
    _log_file = str(TMP_DIR / f"phase323b_{model_name}.log")

    log(f"=== Phase 323b: Attention Paradox & Attribute Decomposition for {model_name} ===")

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}")

    W_U = get_W_U(model, model_name)

    if model_name == "glm4":
        opt_attr_layer = 3
        test_layers = [2, 3, 4]
    elif model_name == "qwen3":
        opt_attr_layer = 0
        test_layers = [0, 1, 2]
    else:
        opt_attr_layer = 6
        test_layers = [5, 6, 7]

    results = {}
    alpha = 2.0

    # ===================================================================
    # Test 1: Attention Ablation Paradox — Multi-layer + Multi-alpha
    # ===================================================================
    log("\n" + "="*60)
    log("Test 1: Attention Ablation Paradox — Stability Check")
    log("="*60)

    paradox_results = []

    for pair_idx, (noun, attr_val, cluster_type) in enumerate(ATTRIBUTE_PAIRS[:6]):
        sent_pos = f"the {noun} is {attr_val}"
        sent_neg = f"the {noun} is just an object"
        target_prompt = f"The {noun} is"

        attr_ids = tokenizer.encode(attr_val, add_special_tokens=False)
        if not attr_ids:
            continue
        tgt_id = attr_ids[0]

        for li in test_layers:
            # Extract direction
            h_pos = extract_rep_at_layer(model, tokenizer, device, sent_pos, li)
            h_neg = extract_rep_at_layer(model, tokenizer, device, sent_neg, li)
            d_attr = h_pos - h_neg
            d_norm = np.linalg.norm(d_attr)
            if d_norm < 1e-10:
                continue
            d_attr_unit = d_attr / d_norm

            # Test with multiple alpha values
            for alpha_test in [1.0, 2.0, 3.0]:
                baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)
                baseline_logit = float(baseline_logits[tgt_id])

                # Full injection
                inj_logits = inject_direction_at_layer(model, tokenizer, device, target_prompt, d_attr_unit, li, alpha_test)
                full_delta = float(inj_logits[tgt_id] - baseline_logit)

                # Ablate attn + inject
                attn_abl_logits = ablate_component_at_layer(model, tokenizer, device, target_prompt, li, 'attn', d_attr_unit, alpha_test)
                attn_abl_delta = float(attn_abl_logits[tgt_id] - baseline_logit)

                # Ablate MLP + inject
                mlp_abl_logits = ablate_component_at_layer(model, tokenizer, device, target_prompt, li, 'mlp', d_attr_unit, alpha_test)
                mlp_abl_delta = float(mlp_abl_logits[tgt_id] - baseline_logit)

                # Random direction control (3 seeds)
                random_deltas = []
                for seed in [42, 43, 44]:
                    np.random.seed(seed + pair_idx)
                    rand_dir = np.random.randn(d_attr.shape[0])
                    rand_dir /= np.linalg.norm(rand_dir)
                    rand_logits = inject_direction_at_layer(model, tokenizer, device, target_prompt, rand_dir, li, alpha_test)
                    random_deltas.append(float(rand_logits[tgt_id] - baseline_logit))

                entry = {
                    "pair": f"{noun}→{attr_val}",
                    "layer": li,
                    "alpha": alpha_test,
                    "full_delta": round(full_delta, 4),
                    "attn_abl_delta": round(attn_abl_delta, 4),
                    "mlp_abl_delta": round(mlp_abl_delta, 4),
                    "attn_importance": round(full_delta - attn_abl_delta, 4),
                    "mlp_importance": round(full_delta - mlp_abl_delta, 4),
                    "random_mean": round(float(np.mean(random_deltas)), 4),
                    "attn_paradox": full_delta < attn_abl_delta,  # True if ablation helps
                }
                paradox_results.append(entry)

                if pair_idx == 0 and alpha_test == 2.0:
                    log(f"    L{li} α={alpha_test}: full={full_delta:.4f}, "
                        f"attn_abl={attn_abl_delta:.4f}, mlp_abl={mlp_abl_delta:.4f}, "
                        f"random={np.mean(random_deltas):.4f}, paradox={full_delta < attn_abl_delta}")

        torch.cuda.empty_cache()

    results["paradox"] = paradox_results

    # Summary
    log("\n  Attention Ablation Paradox Summary:")
    paradox_count = sum(1 for r in paradox_results if r["attn_paradox"])
    total_count = len(paradox_results)
    log(f"    Paradox rate: {paradox_count}/{total_count} ({paradox_count/max(total_count,1)*100:.1f}%)")

    attn_imp_all = [r["attn_importance"] for r in paradox_results]
    mlp_imp_all = [r["mlp_importance"] for r in paradox_results]
    log(f"    attn_importance: mean={np.mean(attn_imp_all):.4f}, positive%={sum(1 for x in attn_imp_all if x > 0)/max(len(attn_imp_all),1)*100:.1f}%")
    log(f"    mlp_importance: mean={np.mean(mlp_imp_all):.4f}, positive%={sum(1 for x in mlp_imp_all if x > 0)/max(len(mlp_imp_all),1)*100:.1f}%")

    # By layer
    layer_data = defaultdict(lambda: {"attn_imp": [], "mlp_imp": [], "paradox": []})
    for r in paradox_results:
        li = r["layer"]
        layer_data[li]["attn_imp"].append(r["attn_importance"])
        layer_data[li]["mlp_imp"].append(r["mlp_importance"])
        layer_data[li]["paradox"].append(r["attn_paradox"])

    for li in sorted(layer_data.keys()):
        d = layer_data[li]
        log(f"    L{li}: attn_imp={np.mean(d['attn_imp']):.4f}, mlp_imp={np.mean(d['mlp_imp']):.4f}, "
            f"paradox_rate={sum(d['paradox'])/max(len(d['paradox']),1)*100:.1f}%")

    # ===================================================================
    # Test 2: Attribute Decomposition — Slot / Type / Value
    # ===================================================================
    log("\n" + "="*60)
    log("Test 2: Attribute Decomposition (Slot → Type → Value)")
    log("="*60)

    decomposition_results = []

    for cluster_type, templates in DECOMPOSITION_TEMPLATES.items():
        # Get the value word from templates
        value_word = templates["value_pos"].split()[-1]  # e.g., "red"
        value_ids = tokenizer.encode(value_word, add_special_tokens=False)
        if not value_ids:
            continue
        tgt_id = value_ids[0]

        # Get cluster IDs
        target_cluster_ids = get_cluster_token_ids(tokenizer, WORD_CLUSTERS.get(cluster_type, []))

        for li in [opt_attr_layer]:
            log(f"  Decomposition: {cluster_type} at L{li}...")

            # Extract directions at each level
            directions = {}
            for level_name, sentence_key in [("slot", "slot"), ("type", "type"), ("value", "value_pos")]:
                if sentence_key not in templates:
                    continue
                h_high = extract_rep_at_layer(model, tokenizer, device, templates[sentence_key], li)
                h_base = extract_rep_at_layer(model, tokenizer, device, templates["base"], li)
                d = h_high - h_base
                norm = np.linalg.norm(d)
                if norm < 1e-10:
                    continue
                directions[level_name] = d / norm

            # Also get the full attribute direction
            h_pos = extract_rep_at_layer(model, tokenizer, device, templates["value_pos"], li)
            h_neg = extract_rep_at_layer(model, tokenizer, device, templates["value_neg"], li)
            d_full = h_pos - h_neg
            d_full_norm = np.linalg.norm(d_full)
            if d_full_norm > 1e-10:
                directions["full_attr"] = d_full / d_full_norm

            # Baseline
            target_prompt = templates["base"].replace(" is something", " is")
            baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)
            baseline_logit = float(baseline_logits[tgt_id])
            baseline_cluster = compute_cluster_mean(baseline_logits, target_cluster_ids)

            # Test each direction level
            level_deltas = {}
            for level_name, direction in directions.items():
                inj_logits = inject_direction_at_layer(
                    model, tokenizer, device, target_prompt, direction, li, alpha)
                level_deltas[level_name] = {
                    "tgt_logit_delta": round(float(inj_logits[tgt_id] - baseline_logit), 4),
                    "cluster_mean_delta": round(compute_cluster_mean(inj_logits, target_cluster_ids) - baseline_cluster, 4),
                }

            # Compare directions: how orthogonal are slot/type/value?
            cos_matrix = {}
            dir_names = list(directions.keys())
            for i, n1 in enumerate(dir_names):
                for j, n2 in enumerate(dir_names):
                    if i < j:
                        cos_matrix[f"cos({n1},{n2})"] = round(float(np.dot(directions[n1], directions[n2])), 4)

            entry = {
                "cluster_type": cluster_type,
                "layer": li,
                "directions_available": list(directions.keys()),
                "level_deltas": level_deltas,
                "cos_matrix": cos_matrix,
            }
            decomposition_results.append(entry)

            log(f"    Levels: {list(directions.keys())}")
            for level_name, deltas in level_deltas.items():
                log(f"      {level_name}: tgt_delta={deltas['tgt_logit_delta']:.4f}, cluster_delta={deltas['cluster_mean_delta']:.4f}")
            for k, v in cos_matrix.items():
                log(f"      {k}={v:.4f}")

        torch.cuda.empty_cache()

    results["decomposition"] = decomposition_results

    # ===================================================================
    # Test 3: Attribute Slot vs Value Transfer
    # Is the "slot" direction transferable across different attribute types?
    # ===================================================================
    log("\n" + "="*60)
    log("Test 3: Attribute Slot Transfer")
    log("="*60)

    transfer_results = []

    for cluster_type, pairs in TRANSFER_TEMPLATES.items():
        for src_noun, src_val, tgt_noun in pairs[:2]:
            # Slot direction from source
            slot_sent_src = f"The {src_noun} has a property"
            base_sent_src = f"The {src_noun} is something"

            tgt_ids = tokenizer.encode(src_val, add_special_tokens=False)
            if not tgt_ids:
                continue

            for li in [opt_attr_layer]:
                h_slot = extract_rep_at_layer(model, tokenizer, device, slot_sent_src, li)
                h_base = extract_rep_at_layer(model, tokenizer, device, base_sent_src, li)
                d_slot = h_slot - h_base
                d_norm = np.linalg.norm(d_slot)
                if d_norm < 1e-10:
                    continue
                d_slot_unit = d_slot / d_norm

                # Inject slot direction into target prompt
                target_prompt = f"The {tgt_noun} is"
                baseline_logits = get_baseline_logits(model, tokenizer, device, target_prompt)

                inj_logits = inject_direction_at_layer(
                    model, tokenizer, device, target_prompt, d_slot_unit, li, alpha)

                # Check cluster effects
                for c_type, c_words in WORD_CLUSTERS.items():
                    c_ids = get_cluster_token_ids(tokenizer, c_words)
                    base_mean = compute_cluster_mean(baseline_logits, c_ids)
                    inj_mean = compute_cluster_mean(inj_logits, c_ids)
                    if c_type == cluster_type:
                        transfer_results.append({
                            "source": f"{src_noun}({src_val})",
                            "target": tgt_noun,
                            "source_cluster": cluster_type,
                            "target_cluster": c_type,
                            "same_cluster": True,
                            "cluster_delta": round(inj_mean - base_mean, 4),
                        })
                    elif c_type in ("color", "temperature", "taste", "texture"):
                        transfer_results.append({
                            "source": f"{src_noun}({src_val})",
                            "target": tgt_noun,
                            "source_cluster": cluster_type,
                            "target_cluster": c_type,
                            "same_cluster": False,
                            "cluster_delta": round(inj_mean - base_mean, 4),
                        })

    results["transfer"] = transfer_results

    # Summary
    if transfer_results:
        same = [r["cluster_delta"] for r in transfer_results if r["same_cluster"]]
        diff = [r["cluster_delta"] for r in transfer_results if not r["same_cluster"]]
        log(f"\n  Slot Transfer: same_cluster_mean={np.mean(same) if same else 0:.4f}, "
            f"diff_cluster_mean={np.mean(diff) if diff else 0:.4f}")

    # ===================================================================
    # Save
    # ===================================================================
    output = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "opt_attr_layer": opt_attr_layer,
        "alpha": alpha,
        "results": results,
    }

    out_path = RESULT_DIR / f"{model_name}_phase323b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")

    # Summary
    log("\n" + "="*60)
    log(f"PHASE 323b SUMMARY - {model_name}")
    log("="*60)
    log(f"  Attention ablation paradox rate: {paradox_count}/{total_count} ({paradox_count/max(total_count,1)*100:.1f}%)")
    log(f"  attn_importance mean: {np.mean(attn_imp_all):.4f}")
    log(f"  mlp_importance mean: {np.mean(mlp_imp_all):.4f}")
    if decomposition_results:
        for dr in decomposition_results:
            log(f"  Decomposition {dr['cluster_type']}:")
            for level, deltas in dr["level_deltas"].items():
                log(f"    {level}: tgt_delta={deltas['tgt_logit_delta']:.4f}, cluster_delta={deltas['cluster_mean_delta']:.4f}")

    del W_U
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
