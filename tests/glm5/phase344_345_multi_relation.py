"""
Phase 344+345: Direction-Matched Random Controls + Multi-Relation Verification
==============================================================================

Critical tests to address Phase 343/343b's hard issues:

Phase 344 — Direction-Matched Random Controls:
  1. norm-matched random: random direction with same L2 norm as binding direction
  2. W_U-subspace random: random direction within W_U column space
  3. binding-orthogonal random: random direction orthogonal to binding direction
  Goal: Confirm binding's net/gross advantage isn't a direction distribution artifact

Phase 345 — Multi-Relation Direction Verification:
  1. binding (object-attribute): apple-red, banana-yellow, etc.
  2. negation: "not red" vs "red"
  3. antonym: hot→cold, big→small
  4. same_class: apple→banana (both fruits)
  5. role: subject vs object position
  6. tense: past vs present
  Goal: Test if "balanced amplification + elevated net/gross" generalizes to other language relations

Phase 346 — High-Order Interaction Decomposition:
  clean gate + clean up, clean gate + corrupt up, corrupt gate + clean up, corrupt gate + corrupt up
  Goal: Quantify gate main effect, up main effect, and gate×up interaction

Usage:
  python tests/glm5/phase344_345_multi_relation.py qwen3
  python tests/glm5/phase344_345_multi_relation.py deepseek7b
  python tests/glm5/phase344_345_multi_relation.py glm4
"""
import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", end=end, flush=True)


MODEL_CONFIGS = {
    "qwen3": {
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "n_layers": 36, "d_model": 2560,
        "binding_layers": [21, 23, 25, 27, 29],
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
        "binding_layers": [30, 33, 36, 38],
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
        "binding_layers": [19, 21, 23, 24],
    },
}

# ===== Phase 345: Multi-Relation Test Pairs =====
# Each tuple: (relation_type, clean_prompt, corrupt_prompt, target_word, competitor_word)
# Direction = W_U[target] - W_U[competitor]

BINDING_PAIRS = [
    ("binding", "The apple", "The item", "red", "blue"),
    ("binding", "The banana", "The item", "yellow", "purple"),
    ("binding", "The snow", "The item", "white", "black"),
    ("binding", "The sky", "The item", "blue", "green"),
    ("binding", "The fire", "The item", "hot", "cold"),
    ("binding", "The grass", "The item", "green", "red"),
    ("binding", "The sun", "The item", "yellow", "purple"),
    ("binding", "The ocean", "The item", "blue", "yellow"),
    ("binding", "The blood", "The item", "red", "green"),
    ("binding", "The ice", "The item", "cold", "hot"),
]

NEGATION_PAIRS = [
    # negation: "not X" vs X — direction from "not" word to positive word
    # We test: clean="The apple is red" vs corrupt="The apple is not red"
    # direction = W_U["red"] - W_U["not"]  (positive vs negative)
    ("negation", "The apple is red", "The apple is not", "red", "not"),
    ("negation", "The sky is blue", "The sky is not", "blue", "not"),
    ("negation", "The fire is hot", "The fire is not", "hot", "not"),
    ("negation", "The snow is white", "The snow is not", "white", "not"),
    ("negation", "The grass is green", "The grass is not", "green", "not"),
]

ANTONYM_PAIRS = [
    # antonym: opposite meaning words
    # direction = W_U[word1] - W_U[word2]
    ("antonym", "The temperature is hot", "The temperature is", "hot", "cold"),
    ("antonym", "The size is big", "The size is", "big", "small"),
    ("antonym", "The room is dark", "The room is", "dark", "bright"),
    ("antonym", "The surface is rough", "The surface is", "rough", "smooth"),
    ("antonym", "The animal is alive", "The animal is", "alive", "dead"),
]

ROLE_PAIRS = [
    # role: subject vs object — "The cat chased the dog" vs "The dog chased the cat"
    # direction = W_U[subject_word] - W_U[object_word]
    ("role", "The cat chased the dog", "The dog chased the cat", "cat", "dog"),
    ("role", "The boy helped the girl", "The girl helped the boy", "boy", "girl"),
    ("role", "The teacher questioned the student", "The student questioned the teacher", "teacher", "student"),
    ("role", "The king ruled the kingdom", "The kingdom ruled the king", "king", "kingdom"),
    ("role", "The dog followed the owner", "The owner followed the dog", "dog", "owner"),
]

TENSE_PAIRS = [
    # tense: past vs present
    ("tense", "Yesterday he ran", "Today he runs", "ran", "runs"),
    ("tense", "She walked slowly", "She walks slowly", "walked", "walks"),
    ("tense", "They ate dinner", "They eat dinner", "ate", "eat"),
    ("tense", "He sang a song", "He sings a song", "sang", "sings"),
    ("tense", "The bird flew away", "The bird flies away", "flew", "flies"),
]

SAME_CLASS_PAIRS = [
    # same_class: within-category comparison
    ("same_class", "The apple", "The banana", "apple", "banana"),
    ("same_class", "The cat", "The dog", "cat", "dog"),
    ("same_class", "The rose", "The lily", "rose", "lily"),
    ("same_class", "The car", "The bus", "car", "bus"),
    ("same_class", "The oak", "The pine", "oak", "pine"),
]

N_MATCHED_RANDOM = 20  # per control type


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=impl)
            log(f"  Loaded {model_name} with attn_impl={impl}")
            break
        except Exception as e:
            log(f"  Failed with {impl}: {e}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def get_W_U(model, model_name):
    if hasattr(model, "lm_head"):
        w = model.lm_head.weight
        if not w.is_meta:
            return w.detach().cpu().float().numpy()
    import glob
    from safetensors import safe_open
    for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
        with safe_open(sf_file, framework='pt', device='cpu') as sf:
            if 'lm_head.weight' in sf.keys():
                return sf.get_tensor('lm_head.weight').float().numpy()
    raise ValueError(f"Cannot load lm_head for {model_name}")


def get_token_id(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    return ids[0] if ids else None


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find layers")


def safe_weight_to_numpy(w):
    if w.is_meta:
        return None
    try:
        return w.detach().cpu().float().numpy()
    except:
        return None


def get_mlp_weights_from_disk(model_name, layer_idx):
    import glob
    from safetensors import safe_open
    W_gate = W_up = W_down = None; d_ff = 0
    for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
        try:
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                keys = sf.keys()
                p = f"model.layers.{layer_idx}.mlp"
                guk = f"{p}.gate_up_proj.weight"
                if guk in keys:
                    w = sf.get_tensor(guk).float().numpy()
                    d_ff = w.shape[0] // 2; W_gate, W_up = w[:d_ff], w[d_ff:]
                gk = f"{p}.gate_proj.weight"
                if gk in keys and W_gate is None:
                    W_gate = sf.get_tensor(gk).float().numpy(); d_ff = W_gate.shape[0]
                uk = f"{p}.up_proj.weight"
                if uk in keys and W_up is None:
                    W_up = sf.get_tensor(uk).float().numpy()
                    if d_ff == 0:
                        d_ff = W_up.shape[0]
                dk = f"{p}.down_proj.weight"
                if dk in keys and W_down is None:
                    W_down = sf.get_tensor(dk).float().numpy()
                if W_down is not None:
                    break
        except:
            continue
    return W_gate, W_up, W_down, d_ff


def get_mlp_weights(layer, model_name=None, model=None):
    mlp = layer.mlp
    W_gate = W_up = W_down = None; d_ff = 0
    if hasattr(mlp, 'gate_up_proj'):
        w = safe_weight_to_numpy(mlp.gate_up_proj.weight)
        if w is not None:
            d_ff = w.shape[0] // 2; W_gate, W_up = w[:d_ff], w[d_ff:]
    elif hasattr(mlp, 'gate_proj'):
        W_gate = safe_weight_to_numpy(mlp.gate_proj.weight)
        W_up = safe_weight_to_numpy(mlp.up_proj.weight)
        if W_gate is not None:
            d_ff = W_gate.shape[0]
        elif W_up is not None:
            d_ff = W_up.shape[0]
    elif hasattr(mlp, 'up_proj'):
        W_up = safe_weight_to_numpy(mlp.up_proj.weight)
        if W_up is not None:
            d_ff = W_up.shape[0]
    if hasattr(mlp, 'down_proj'):
        W_down = safe_weight_to_numpy(mlp.down_proj.weight)
    if W_down is None and model_name is not None:
        layers = get_layers(model)
        for i, l in enumerate(layers):
            if l is layer:
                W_gate, W_up, W_down, d_ff = get_mlp_weights_from_disk(model_name, i)
                break
    return W_gate, W_up, W_down, d_ff


def capture_mlp_internals(model, tokenizer, device, prompt, target_layers, n_layers):
    layers = get_layers(model)
    captured = {}
    def make_hook(key):
        def hook(module, input, output):
            val = output[0] if isinstance(output, tuple) else output
            captured[key] = val[0, -1, :].detach().cpu().float().numpy()
        return hook
    hooks = []
    for li in target_layers:
        layer = layers[li]
        if hasattr(layer.mlp, 'gate_proj'):
            hooks.append(layer.mlp.gate_proj.register_forward_hook(make_hook(f"gate_{li}")))
        elif hasattr(layer.mlp, 'gate_up_proj'):
            def make_glm4_hook(idx):
                def hook(module, input, output):
                    val = output[0] if isinstance(output, tuple) else output
                    v = val[0, -1, :].detach().cpu().float().numpy()
                    d = v.shape[0] // 2
                    captured[f"gate_{idx}"] = v[:d]; captured[f"up_{idx}"] = v[d:]
                return hook
            hooks.append(layer.mlp.gate_up_proj.register_forward_hook(make_glm4_hook(li)))
        if hasattr(layer.mlp, 'up_proj'):
            hooks.append(layer.mlp.up_proj.register_forward_hook(make_hook(f"up_{li}")))
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        model(**inp, output_hidden_states=True)
    for h in hooks:
        h.remove()
    return captured


def channel_decomposition(W_down, direction, gate_clean, up_clean, gate_corrupt, up_corrupt):
    """Full channel decomposition returning balance ratio, net/gross ratio, and per-channel info."""
    d_ff = W_down.shape[1]
    min_d = min(gate_clean.shape[0], d_ff, W_down.shape[1])
    def silu(x):
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))
    gsc = silu(gate_clean[:min_d]); gsr = silu(gate_corrupt[:min_d])
    uc = up_clean[:min_d]; ur = up_corrupt[:min_d]
    Wd = W_down[:, :min_d]
    dp = direction @ Wd  # direction projection per channel
    cc = dp * gsc * uc; cr = dp * gsr * ur
    delta = cc - cr
    pos_mask = dp > 0; neg_mask = dp < 0
    pos_gross = float(np.sum(np.abs(delta[pos_mask])))
    neg_gross = float(np.sum(np.abs(delta[neg_mask])))
    total_gross = pos_gross + neg_gross
    net = float(np.sum(delta))
    balance = neg_gross / max(pos_gross, 1e-10)
    net_gross = abs(net) / max(total_gross, 1e-10)
    return {
        "balance": balance, "net_gross_ratio": net_gross,
        "total_gross": total_gross, "net": net,
        "pos_gross": pos_gross, "neg_gross": neg_gross,
        "n_pos_channels": int(np.sum(pos_mask)),
        "n_neg_channels": int(np.sum(neg_mask)),
    }


def interaction_decomposition(W_down, direction, gate_clean, up_clean, gate_corrupt, up_corrupt):
    """
    Phase 346: Decompose the net bias into gate main effect, up main effect, and interaction.
    
    MLP_output_diff = W_down @ (SiLU(gate_c)*up_c - SiLU(gate_r)*up_r)
    
    Let Δgate = SiLU(gate_c) - SiLU(gate_r), Δup = up_c - up_r
    Let gate_avg = (SiLU(gate_c) + SiLU(gate_r))/2, up_avg = (up_c + up_r)/2
    
    Then: Δoutput ≈ W_down @ (Δgate * up_avg + gate_avg * Δup + Δgate * Δup)
                                                     ^gate_main    ^up_main    ^interaction
    """
    d_ff = W_down.shape[1]
    min_d = min(gate_clean.shape[0], d_ff, W_down.shape[1])
    def silu(x):
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))
    
    gsc = silu(gate_clean[:min_d]); gsr = silu(gate_corrupt[:min_d])
    uc = up_clean[:min_d]; ur = up_corrupt[:min_d]
    Wd = W_down[:, :min_d]
    
    delta_gate = gsc - gsr
    delta_up = uc - ur
    gate_avg = (gsc + gsr) / 2
    up_avg = (uc + ur) / 2
    
    # Three decomposition terms
    gate_main = Wd @ (delta_gate * up_avg)  # gate main effect
    up_main = Wd @ (gate_avg * delta_up)     # up main effect
    interaction = Wd @ (delta_gate * delta_up)  # gate×up interaction
    
    # Project onto direction
    gate_main_proj = float(direction @ gate_main)
    up_main_proj = float(direction @ up_main)
    interaction_proj = float(direction @ interaction)
    
    # Total
    total_proj = gate_main_proj + up_main_proj + interaction_proj
    
    # Also compute the exact difference for comparison
    exact_diff = Wd @ (gsc * uc - gsr * ur)
    exact_proj = float(direction @ exact_diff)
    
    return {
        "gate_main": gate_main_proj,
        "up_main": up_main_proj,
        "interaction": interaction_proj,
        "total_decomposed": total_proj,
        "exact_diff": exact_proj,
        "decomposition_error": abs(total_proj - exact_proj),
        "gate_main_pct": abs(gate_main_proj) / max(abs(total_proj), 1e-10),
        "up_main_pct": abs(up_main_proj) / max(abs(total_proj), 1e-10),
        "interaction_pct": abs(interaction_proj) / max(abs(total_proj), 1e-10),
        "interaction_sign": 1 if interaction_proj > 0 else -1,
    }


def run_experiment(model_name):
    log(f"Phase 344+345+346: Multi-Relation + Matched Controls + Interaction — {model_name}")
    log("=" * 70)
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    binding_layers = cfg["binding_layers"]
    W_U = get_W_U(model, model_name)
    d_model = W_U.shape[1]
    vocab_size = W_U.shape[0]
    log(f"  W_U shape: {W_U.shape}")

    # Pre-extract MLP weights
    layers = get_layers(model)
    mlp_weights = {}
    for li in binding_layers:
        _, _, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
        mlp_weights[li] = {"W_down": W_down, "d_ff": d_ff}
        log(f"  Layer {li}: W_down shape={W_down.shape if W_down is not None else 'None'}")

    results = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "binding_layers": binding_layers,
    }

    # ======================================================================
    # PART 1: Phase 345 — Multi-Relation Direction Verification
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART 1: Multi-Relation Direction Verification")
    log(f"{'='*70}")

    all_relation_types = [
        ("binding", BINDING_PAIRS),
        ("negation", NEGATION_PAIRS),
        ("antonym", ANTONYM_PAIRS),
        ("role", ROLE_PAIRS),
        ("tense", TENSE_PAIRS),
        ("same_class", SAME_CLASS_PAIRS),
    ]

    relation_results = {}

    for rel_name, pairs in all_relation_types:
        log(f"\n  Testing relation type: {rel_name} ({len(pairs)} pairs)")
        rel_data = {li: {"balance": [], "net_gross": [], "total_gross": [], "net": [],
                         "gate_main": [], "up_main": [], "interaction": []}
                    for li in binding_layers}
        valid_pairs = 0

        for pidx, (rtype, clean_prompt, corrupt_prompt, target_word, competitor_word) in enumerate(pairs):
            tid_t = get_token_id(tokenizer, target_word)
            tid_c = get_token_id(tokenizer, competitor_word)
            if tid_t is None or tid_c is None:
                log(f"    [{pidx}] SKIP: cannot tokenize '{target_word}' or '{competitor_word}'")
                continue

            direction = W_U[tid_t] - W_U[tid_c]
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-10:
                continue
            direction_normed = direction / dir_norm

            # Quick binding range check for binding type
            if rtype == "binding":
                inp_c = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
                inp_cl = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    out_c = model(**inp_c, output_hidden_states=True)
                    out_cl = model(**inp_cl, output_hidden_states=True)
                final_c = out_c.hidden_states[-1][0, -1].detach().cpu().float().numpy()
                final_cl = out_cl.hidden_states[-1][0, -1].detach().cpu().float().numpy()
                br = float(direction @ final_cl) - float(direction @ final_c)
                del out_c, out_cl; gc.collect(); torch.cuda.empty_cache()
                if br < 0.3:
                    log(f"    [{pidx}] SKIP: binding range {br:.2f} < 0.3")
                    continue

            # Capture MLP internals
            clean_caps = capture_mlp_internals(model, tokenizer, device, clean_prompt, binding_layers, cfg["n_layers"])
            corrupt_caps = capture_mlp_internals(model, tokenizer, device, corrupt_prompt, binding_layers, cfg["n_layers"])

            for li in binding_layers:
                mw = mlp_weights[li]
                W_down = mw["W_down"]; d_ff = mw["d_ff"]
                if W_down is None:
                    continue
                gk = f"gate_{li}"; uk = f"up_{li}"
                if gk not in clean_caps or gk not in corrupt_caps:
                    continue
                cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
                cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
                cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]

                # Channel decomposition
                res = channel_decomposition(W_down, direction_normed, cg, cu, crg, cru)
                rel_data[li]["balance"].append(res["balance"])
                rel_data[li]["net_gross"].append(res["net_gross_ratio"])
                rel_data[li]["total_gross"].append(res["total_gross"])
                rel_data[li]["net"].append(res["net"])

                # Interaction decomposition (Phase 346)
                ires = interaction_decomposition(W_down, direction_normed, cg, cu, crg, cru)
                rel_data[li]["gate_main"].append(ires["gate_main"])
                rel_data[li]["up_main"].append(ires["up_main"])
                rel_data[li]["interaction"].append(ires["interaction"])

            del clean_caps, corrupt_caps; gc.collect(); torch.cuda.empty_cache()
            valid_pairs += 1

            if (pidx + 1) % 5 == 0 or pidx == len(pairs) - 1:
                log(f"    [{pidx+1}/{len(pairs)}] valid={valid_pairs}, elapsed={time.time()-t0:.0f}s")

        # Aggregate for this relation type
        agg = {"per_layer": {}}
        all_bal = []; all_ng = []; all_gm = []; all_um = []; all_ia = []
        for li in binding_layers:
            bb = rel_data[li]["balance"]; bng = rel_data[li]["net_gross"]
            gm = rel_data[li]["gate_main"]; um = rel_data[li]["up_main"]; ia = rel_data[li]["interaction"]
            if bb:
                agg["per_layer"][str(li)] = {
                    "balance_mean": float(np.mean(bb)),
                    "net_gross_mean": float(np.mean(bng)),
                    "gate_main_mean": float(np.mean(gm)),
                    "up_main_mean": float(np.mean(um)),
                    "interaction_mean": float(np.mean(ia)),
                    "n": len(bb),
                }
                all_bal.extend(bb); all_ng.extend(bng)
                all_gm.extend(gm); all_um.extend(um); all_ia.extend(ia)

        if all_bal:
            agg["balance_mean"] = float(np.mean(all_bal))
            agg["balance_std"] = float(np.std(all_bal))
            agg["net_gross_mean"] = float(np.mean(all_ng))
            agg["net_gross_std"] = float(np.std(all_ng))
            # Interaction decomposition percentages
            total_abs = np.mean(np.abs(all_gm)) + np.mean(np.abs(all_um)) + np.mean(np.abs(all_ia))
            if total_abs > 1e-10:
                agg["gate_main_pct"] = float(np.mean(np.abs(all_gm)) / total_abs)
                agg["up_main_pct"] = float(np.mean(np.abs(all_um)) / total_abs)
                agg["interaction_pct"] = float(np.mean(np.abs(all_ia)) / total_abs)
            agg["n_observations"] = len(all_bal)
            log(f"  {rel_name}: balance={agg['balance_mean']:.4f}±{agg['balance_std']:.4f}, "
                f"net/gross={agg['net_gross_mean']:.4f}±{agg['net_gross_std']:.4f}, n={len(all_bal)}")

        relation_results[rel_name] = agg

    # ======================================================================
    # PART 2: Phase 344 — Direction-Matched Random Controls
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART 2: Direction-Matched Random Controls")
    log(f"{'='*70}")

    # Use one binding direction as reference for matching
    apple_red_dir = W_U[get_token_id(tokenizer, "red")] - W_U[get_token_id(tokenizer, "blue")]
    if apple_red_dir is None:
        apple_red_dir = np.random.randn(d_model)
    binding_dir_norm = np.linalg.norm(apple_red_dir)
    apple_red_dir_normed = apple_red_dir / max(binding_dir_norm, 1e-10)

    # W_U subspace: use SVD to get the principal subspace of W_U
    log(f"  Computing W_U subspace (SVD)...")
    # Use a subset of W_U for SVD (too large otherwise)
    W_U_subset = W_U[:min(5000, vocab_size)]
    U_wu, S_wu, Vt_wu = np.linalg.svd(W_U_subset, full_matrices=False)
    W_U_subspace = Vt_wu[:d_model]  # top-d_model principal directions
    log(f"  W_U subspace: {W_U_subspace.shape}, top-10 singular values: {S_wu[:10].tolist()}")

    # Capture activations once for random direction tests
    clean_caps = capture_mlp_internals(model, tokenizer, device, "The apple", binding_layers, cfg["n_layers"])
    corrupt_caps = capture_mlp_internals(model, tokenizer, device, "The item", binding_layers, cfg["n_layers"])

    np.random.seed(42)

    # Control type 1: Norm-matched random
    log(f"\n  Control 1: Norm-matched random ({N_MATCHED_RANDOM} samples)")
    norm_matched = {li: {"balance": [], "net_gross": []} for li in binding_layers}
    for ri in range(N_MATCHED_RANDOM):
        d = np.random.randn(d_model) * binding_dir_norm  # same L2 norm
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-10:
            continue
        d = d / d_norm  # normalize to unit

        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]; d_ff = mw["d_ff"]
            if W_down is None:
                continue
            gk = f"gate_{li}"; uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps:
                continue
            cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]
            res = channel_decomposition(W_down, d, cg, cu, crg, cru)
            norm_matched[li]["balance"].append(res["balance"])
            norm_matched[li]["net_gross"].append(res["net_gross_ratio"])

    # Control type 2: W_U-subspace random
    log(f"\n  Control 2: W_U-subspace random ({N_MATCHED_RANDOM} samples)")
    wu_subspace = {li: {"balance": [], "net_gross": []} for li in binding_layers}
    for ri in range(N_MATCHED_RANDOM):
        # Random combination of W_U principal directions
        coeffs = np.random.randn(d_model)
        d = W_U_subspace.T @ coeffs
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-10:
            continue
        d = d / d_norm

        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]; d_ff = mw["d_ff"]
            if W_down is None:
                continue
            gk = f"gate_{li}"; uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps:
                continue
            cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]
            res = channel_decomposition(W_down, d, cg, cu, crg, cru)
            wu_subspace[li]["balance"].append(res["balance"])
            wu_subspace[li]["net_gross"].append(res["net_gross_ratio"])

    # Control type 3: Binding-orthogonal random
    log(f"\n  Control 3: Binding-orthogonal random ({N_MATCHED_RANDOM} samples)")
    binding_orth = {li: {"balance": [], "net_gross": []} for li in binding_layers}
    for ri in range(N_MATCHED_RANDOM):
        d = np.random.randn(d_model)
        # Remove component along binding direction
        d = d - (d @ apple_red_dir_normed) * apple_red_dir_normed
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-10:
            continue
        d = d / d_norm

        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]; d_ff = mw["d_ff"]
            if W_down is None:
                continue
            gk = f"gate_{li}"; uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps:
                continue
            cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]
            res = channel_decomposition(W_down, d, cg, cu, crg, cru)
            binding_orth[li]["balance"].append(res["balance"])
            binding_orth[li]["net_gross"].append(res["net_gross_ratio"])

    # Pure random (standard Gaussian)
    log(f"\n  Control 4: Pure random ({N_MATCHED_RANDOM} samples)")
    pure_random = {li: {"balance": [], "net_gross": []} for li in binding_layers}
    for ri in range(N_MATCHED_RANDOM):
        d = np.random.randn(d_model)
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-10:
            continue
        d = d / d_norm

        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]; d_ff = mw["d_ff"]
            if W_down is None:
                continue
            gk = f"gate_{li}"; uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps:
                continue
            cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]
            res = channel_decomposition(W_down, d, cg, cu, crg, cru)
            pure_random[li]["balance"].append(res["balance"])
            pure_random[li]["net_gross"].append(res["net_gross_ratio"])

    del clean_caps, corrupt_caps; gc.collect(); torch.cuda.empty_cache()

    # ======================================================================
    # PART 3: Phase 346 — Interaction Decomposition for Binding
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART 3: Interaction Decomposition (Phase 346)")
    log(f"{'='*70}")

    interaction_results = {}
    for pidx, (rtype, clean_prompt, corrupt_prompt, target_word, competitor_word) in enumerate(BINDING_PAIRS[:6]):
        tid_t = get_token_id(tokenizer, target_word)
        tid_c = get_token_id(tokenizer, competitor_word)
        if tid_t is None or tid_c is None:
            continue
        direction = W_U[tid_t] - W_U[tid_c]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10:
            continue
        direction_normed = direction / dir_norm

        clean_caps = capture_mlp_internals(model, tokenizer, device, clean_prompt, binding_layers, cfg["n_layers"])
        corrupt_caps = capture_mlp_internals(model, tokenizer, device, corrupt_prompt, binding_layers, cfg["n_layers"])

        pair_key = f"{target_word}-{competitor_word}"
        interaction_results[pair_key] = {}

        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]; d_ff = mw["d_ff"]
            if W_down is None:
                continue
            gk = f"gate_{li}"; uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps:
                continue
            cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]

            ires = interaction_decomposition(W_down, direction_normed, cg, cu, crg, cru)
            interaction_results[pair_key][str(li)] = ires

        del clean_caps, corrupt_caps; gc.collect(); torch.cuda.empty_cache()
        log(f"  Pair {pidx+1}: {pair_key}")

    # ======================================================================
    # Aggregate and Display Results
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"RESULTS SUMMARY")
    log(f"{'='*70}")

    # --- Part 1: Multi-Relation ---
    log(f"\n  PART 1: Multi-Relation Comparison")
    log(f"  {'Relation':>12} {'Balance':>10} {'±std':>8} {'Net/Gross':>10} {'±std':>8} {'N':>5}")
    log("  " + "-" * 60)

    relation_summary = {}
    for rel_name in ["binding", "negation", "antonym", "role", "tense", "same_class"]:
        if rel_name in relation_results and "balance_mean" in relation_results[rel_name]:
            r = relation_results[rel_name]
            log(f"  {rel_name:>12} {r['balance_mean']:>10.4f} {r['balance_std']:>8.4f} "
                f"{r['net_gross_mean']:>10.4f} {r['net_gross_std']:>8.4f} {r['n_observations']:>5}")
            relation_summary[rel_name] = {
                "balance": r["balance_mean"],
                "net_gross": r["net_gross_mean"],
            }

    # --- Part 2: Matched Controls ---
    log(f"\n  PART 2: Direction-Matched Controls")
    log(f"  {'Control Type':>25} {'Balance':>10} {'Net/Gross':>10} {'N':>5}")
    log("  " + "-" * 55)

    control_summary = {}
    for cname, cdata in [("binding (reference)", None),  # will fill from relation_results
                          ("norm-matched random", norm_matched),
                          ("W_U-subspace random", wu_subspace),
                          ("binding-orthogonal random", binding_orth),
                          ("pure random", pure_random)]:
        if cdata is None:
            # Use binding relation results
            if "binding" in relation_results and "balance_mean" in relation_results["binding"]:
                r = relation_results["binding"]
                b_mean = r["balance_mean"]; ng_mean = r["net_gross_mean"]; n = r["n_observations"]
            else:
                b_mean = ng_mean = n = 0
        else:
            all_b = []; all_ng = []
            for li in binding_layers:
                all_b.extend(cdata[li]["balance"])
                all_ng.extend(cdata[li]["net_gross"])
            b_mean = np.mean(all_b) if all_b else 0
            ng_mean = np.mean(all_ng) if all_ng else 0
            n = len(all_b)

        log(f"  {cname:>25} {b_mean:>10.4f} {ng_mean:>10.4f} {n:>5}")
        control_summary[cname] = {"balance": float(b_mean), "net_gross": float(ng_mean), "n": int(n)}

    # Statistical test: binding vs each control
    from scipy import stats as scipy_stats

    log(f"\n  Statistical Test: Binding vs Controls (Net/Gross)")
    binding_ng_all = []
    if "binding" in relation_results:
        for li in binding_layers:
            if str(li) in relation_results["binding"].get("per_layer", {}):
                binding_ng_all.extend(
                    [relation_results["binding"]["per_layer"][str(li)].get("net_gross_mean", 0)])

    # Use per-layer binding net/gross
    binding_per_layer_ng = []
    if "binding" in relation_results:
        for li in binding_layers:
            pl = relation_results["binding"].get("per_layer", {}).get(str(li), {})
            if "net_gross_mean" in pl:
                binding_per_layer_ng.append(pl["net_gross_mean"])

    for cname, cdata in [("norm-matched random", norm_matched),
                          ("W_U-subspace random", wu_subspace),
                          ("binding-orthogonal random", binding_orth),
                          ("pure random", pure_random)]:
        ctrl_ng = []
        for li in binding_layers:
            ctrl_ng.extend(cdata[li]["net_gross"])

        if binding_per_layer_ng and ctrl_ng and len(ctrl_ng) > 5:
            t_val, p_val = scipy_stats.ttest_ind(binding_per_layer_ng, ctrl_ng[:len(binding_per_layer_ng)*3])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            log(f"    vs {cname}: t={t_val:.3f}, p={p_val:.4f} {sig}")

    # --- Part 3: Interaction Decomposition ---
    log(f"\n  PART 3: Interaction Decomposition")
    all_gm_pct = []; all_um_pct = []; all_ia_pct = []
    for pair_key, layer_data in interaction_results.items():
        for li_str, idata in layer_data.items():
            all_gm_pct.append(idata["gate_main_pct"])
            all_um_pct.append(idata["up_main_pct"])
            all_ia_pct.append(idata["interaction_pct"])

    if all_gm_pct:
        log(f"  Gate main effect:    {np.mean(all_gm_pct):.4f} ± {np.std(all_gm_pct):.4f} ({np.mean(all_gm_pct)*100:.1f}%)")
        log(f"  Up main effect:      {np.mean(all_um_pct):.4f} ± {np.std(all_um_pct):.4f} ({np.mean(all_um_pct)*100:.1f}%)")
        log(f"  Gate×Up interaction: {np.mean(all_ia_pct):.4f} ± {np.std(all_ia_pct):.4f} ({np.mean(all_ia_pct)*100:.1f}%)")

        # Interaction sign analysis
        pos_ia = sum(1 for x in all_ia_pct if x > 0.3)
        neg_ia = sum(1 for x in all_ia_pct if x < -0.3)
        log(f"  Interaction >30%: {pos_ia}/{len(all_ia_pct)} positive, {neg_ia}/{len(all_ia_pct)} negative")

    # Decomposition quality check
    decomp_errors = []
    for pair_key, layer_data in interaction_results.items():
        for li_str, idata in layer_data.items():
            decomp_errors.append(idata["decomposition_error"])
    if decomp_errors:
        log(f"  Decomposition error: mean={np.mean(decomp_errors):.6f}, max={np.max(decomp_errors):.6f}")

    # ======================================================================
    # Save Results
    # ======================================================================
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    save_data = convert({
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "relation_results": relation_results,
        "control_summary": control_summary,
        "norm_matched": {str(li): {"balance": norm_matched[li]["balance"],
                                    "net_gross": norm_matched[li]["net_gross"]}
                         for li in binding_layers},
        "wu_subspace": {str(li): {"balance": wu_subspace[li]["balance"],
                                   "net_gross": wu_subspace[li]["net_gross"]}
                        for li in binding_layers},
        "binding_orth": {str(li): {"balance": binding_orth[li]["balance"],
                                    "net_gross": binding_orth[li]["net_gross"]}
                         for li in binding_layers},
        "pure_random": {str(li): {"balance": pure_random[li]["balance"],
                                   "net_gross": pure_random[li]["net_gross"]}
                        for li in binding_layers},
        "interaction_results": interaction_results,
        "interaction_summary": {
            "gate_main_pct_mean": float(np.mean(all_gm_pct)) if all_gm_pct else 0,
            "up_main_pct_mean": float(np.mean(all_um_pct)) if all_um_pct else 0,
            "interaction_pct_mean": float(np.mean(all_ia_pct)) if all_ia_pct else 0,
            "gate_main_pct_std": float(np.std(all_gm_pct)) if all_gm_pct else 0,
            "up_main_pct_std": float(np.std(all_um_pct)) if all_um_pct else 0,
            "interaction_pct_std": float(np.std(all_ia_pct)) if all_ia_pct else 0,
        },
    })

    os.makedirs("results/phase344_345_multi_relation", exist_ok=True)
    out_path = f"results/phase344_345_multi_relation/{model_name}_phase344_345.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log(f"\nResults saved to {out_path}")

    del model, W_U, mlp_weights; gc.collect(); torch.cuda.empty_cache()
    total_time = time.time() - t0
    log(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}"); sys.exit(1)
    run_experiment(model_name)
    log("Phase 344+345+346 complete!")
