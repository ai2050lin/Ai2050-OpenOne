"""
Phase 518: Corrected Trajectory Value Causality & Geometric Filtering
=====================================================================
Phase 517 修正：hidden_states[-1] 是 post-norm，读出纯线性。
Phase 518 新增（基于两份分析的综合修正）：
1. hidden_states 索引审计（-1/-2 到底是什么）
2. Prompt 鲁棒语义基线（强/弱/无提示）
3. 切向 vs 径向干预（验证 RMSNorm 几何过滤假设）
4. 修正后的 d_traj 因果验证（post-norm 读出 + 分层 classify）

用法:
  python tests/glm5/phase518_corrected_causality.py qwen3
  python tests/glm5/phase518_corrected_causality.py glm4
  python tests/glm5/phase518_corrected_causality.py deepseek7b
"""
import sys, os, gc, time, json, re
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import numpy as np
import torch
from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS

# ============== Configuration ==============
FRUIT_OBJECTS = ["apple", "banana", "orange", "grape", "strawberry",
                 "mango", "pear", "cherry", "watermelon", "pineapple",
                 "peach", "lemon", "lime", "coconut", "kiwi"]

# Phase 518: 三种提示强度
PROMPT_CUES = {
    "strong": [  # 强提示：含 "category of" 等
        "belongs to the category of",
        "is classified as a type of",
        "is a kind of",
    ],
    "weak": [  # 弱提示：只有基本句法
        "is a",
        "is an",
    ],
    "none": [  # 无提示：只有冒号
        "is:",
        ":",
    ],
}

_WEIGHT_CACHE = {}


def log(msg):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bfloat16 + device_map=auto, sdpa)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True,
        attn_implementation="sdpa")
    model.eval()
    input_device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"{model_name} loaded: class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, input_device


def safe_encode(tokenizer, text, device, max_length=64):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    return {"input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device)}


def get_norm_weight(model, model_name):
    cache_key = f"{model_name}_norm"
    if cache_key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[cache_key]
    w = model.model.norm.weight
    if not w.is_meta:
        g = w.detach().float().cpu().numpy()
    else:
        import glob
        from safetensors import safe_open
        cfg = MODEL_CONFIGS[model_name]
        sf_files = glob.glob(os.path.join(cfg["path"], '*.safetensors'))
        g = None
        for sf_file in sf_files:
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                for pk in ['model.norm.weight', 'model.model.norm.weight']:
                    if pk in sf.keys():
                        g = sf.get_tensor(pk).float().numpy()
                        break
                if g is None:
                    for k in sf.keys():
                        if k.endswith('norm.weight') and 'layers' not in k:
                            g = sf.get_tensor(k).float().numpy()
                            break
            if g is not None:
                break
    _WEIGHT_CACHE[cache_key] = g
    return g


def get_W_U_cached(model, model_name):
    cache_key = f"{model_name}_WU"
    if cache_key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[cache_key]
    W_U = get_W_U(model, model_name)
    _WEIGHT_CACHE[cache_key] = W_U
    return W_U


def correct_readout(h_post, W_U):
    """正确的 post-norm 线性读出：logits = h_post @ W_U.T"""
    return h_post @ W_U.T


# ============== 分层 Semantic Hit ==============

def classify_layered(full_text, prompt, cat_words):
    """
    分层语义命中评价:
    S0=miss, S1=lexical, S2=scaffolded_semantic, S3=continuation_phrase, S4=free_semantic
    """
    text_lower = full_text.lower()
    cont = full_text[len(prompt):].strip().lower()
    found_cat = None
    for cw in cat_words:
        if cw.lower() in text_lower:
            found_cat = cw.lower()
            break
    if found_cat is None:
        return "S0_miss"

    # S1: lexical — category word present
    if found_cat not in cont:
        # category word only in prompt (scaffold)
        return "S1_lexical"

    # category word in continuation
    # S3: continuation forms natural phrase (e.g., "a fruit", "type of fruit")
    cont_phrases = [f"a {found_cat}", f"an {found_cat}", f"the {found_cat}",
                    f"type of {found_cat}", f"kind of {found_cat}",
                    f"category of {found_cat}", f"is a {found_cat}", f"is an {found_cat}"]
    has_cont_phrase = any(p in cont for p in cont_phrases)
    if has_cont_phrase:
        return "S3_cont_phrase"

    # S2: scaffolded semantic — prompt + continuation forms semantic answer
    # e.g., prompt="belongs to the category of" + cont="fruit" → "category of fruit"
    scaffold_phrases = [f"category of {found_cat}", f"type of {found_cat}",
                        f"kind of {found_cat}", f"a {found_cat}", f"an {found_cat}"]
    has_scaffold = any(p in text_lower for p in scaffold_phrases)
    if has_scaffold:
        return "S2_scaffold"

    # S4: free semantic — category word in continuation without scaffold
    return "S4_free"


# ============== Exp1: Hidden State Indexing Audit ==============

def exp1_indexing_audit(model, tokenizer, input_device, model_name):
    """
    Verify what hidden_states[-1] and hidden_states[-2] actually are.
    Test: does RMSNorm(Layer_{L-1}(hidden_states[-2])) ≈ hidden_states[-1]?
    If yes: hidden_states[-2] is input to last layer.
    Or: does RMSNorm(hidden_states[-2]) ≈ hidden_states[-1]?
    If yes: hidden_states[-2] is final pre-norm (output of last layer before norm).
    """
    log("="*60)
    log("Exp1: Hidden State Indexing Audit")
    log("="*60)

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    g = get_norm_weight(model, model_name)
    eps = getattr(model.config, 'rms_norm_eps', 1e-6)

    prompt = "An apple belongs to the category of"
    enc = safe_encode(tokenizer, prompt, input_device)

    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True)

    hs = out.hidden_states  # tuple of (n_layers+1) tensors
    log(f"  n_hidden_states={len(hs)}, n_layers={n_layers}")
    log(f"  (Expected: {n_layers+1} = embedding + {n_layers} layer outputs)")

    h_last_post = hs[-1][0, -1, :].detach().float().cpu().numpy()  # hidden_states[-1]
    h_minus2 = hs[-2][0, -1, :].detach().float().cpu().numpy()  # hidden_states[-2]
    h_minus3 = hs[-3][0, -1, :].detach().float().cpu().numpy()  # hidden_states[-3]

    # Test A: Is hidden_states[-2] the final pre-norm? (output of last layer before norm)
    # If so: RMSNorm(hidden_states[-2]) should ≈ hidden_states[-1]
    rms_m2 = np.sqrt(np.mean(h_minus2**2) + eps)
    h_m2_normed = g * h_minus2 / rms_m2
    diff_A = np.abs(h_m2_normed - h_last_post)

    log(f"\n  Test A: RMSNorm(hidden_states[-2]) ≈ hidden_states[-1]?")
    log(f"    ||hidden_states[-1]||     = {np.linalg.norm(h_last_post):.4f}")
    log(f"    ||hidden_states[-2]||     = {np.linalg.norm(h_minus2):.4f}")
    log(f"    ||RMSNorm(h[-2])||        = {np.linalg.norm(h_m2_normed):.4f}")
    log(f"    |RMSNorm(h[-2]) - h[-1]|  max={diff_A.max():.6f}, mean={diff_A.mean():.6f}")

    if diff_A.max() < 0.1:
        verdict_A = "YES — hidden_states[-2] IS the final pre-norm state"
    else:
        verdict_A = "NO — hidden_states[-2] is NOT the final pre-norm state"
    log(f"    Verdict: {verdict_A}")

    # Test B: Run last layer manually (needs position_embeddings for Qwen3)
    # Skip if it fails — Test A is sufficient to determine if h[-2] is pre-norm
    try:
        layers = get_layers(model)
        last_layer = layers[n_layers - 1]
        layer_device = next(last_layer.parameters()).device
        full_h = hs[-2].detach().to(layer_device)  # [1, seq_len, d_model]

        with torch.no_grad():
            layer_out = last_layer(full_h)
            if isinstance(layer_out, tuple):
                h_pre_final = layer_out[0][:, -1, :].detach().float().cpu().numpy()
            else:
                h_pre_final = layer_out[:, -1, :].detach().float().cpu().numpy()

        rms_pf = np.sqrt(np.mean(h_pre_final**2) + eps)
        h_pf_normed = g * h_pre_final / rms_pf
        diff_B = np.abs(h_pf_normed - h_last_post)

        log(f"\n  Test B: RMSNorm(Layer_{{L-1}}(hidden_states[-2])) ≈ hidden_states[-1]?")
        log(f"    ||Layer_{{L-1}}(h[-2])[-1]|| = {np.linalg.norm(h_pre_final):.4f}")
        log(f"    ||RMSNorm(Layer(h[-2]))||    = {np.linalg.norm(h_pf_normed):.4f}")
        log(f"    |RMSNorm(Layer(h[-2])) - h[-1]| max={diff_B.max():.6f}, mean={diff_B.mean():.6f}")

        if diff_B.max() < 0.1:
            verdict_B = "YES — hidden_states[-2] is INPUT to last layer (output of layer L-2)"
        else:
            verdict_B = "NO — manual layer forward doesn't match either"
        log(f"    Verdict: {verdict_B}")
        test_b_diff = float(diff_B.max())
    except Exception as e:
        log(f"\n  Test B skipped (layer forward needs position_embeddings): {e}")
        verdict_B = f"SKIPPED — {e}"
        test_b_diff = None

    # Also check: hidden_states[-1] vs model logits (sanity from Phase 517)
    W_U = get_W_U_cached(model, model_name)
    model_logits = out.logits[0, -1, :].float().cpu().numpy()
    manual_logits = h_last_post @ W_U.T
    diff_readout = np.abs(manual_logits - model_logits)
    log(f"\n  Sanity: |hidden_states[-1] @ W_U.T - model_logits| max={diff_readout.max():.6f}")

    return {
        "n_hidden_states": len(hs),
        "n_layers": n_layers,
        "test_A_diff_max": float(diff_A.max()),
        "test_A_verdict": verdict_A,
        "test_B_diff_max": test_b_diff,
        "test_B_verdict": verdict_B,
        "readout_diff_max": float(diff_readout.max()),
        "h_last_norm": float(np.linalg.norm(h_last_post)),
        "h_minus2_norm": float(np.linalg.norm(h_minus2)),
    }


# ============== Exp2: Prompt-Robust Semantic Baseline ==============

def exp2_prompt_robust_baseline(model, tokenizer, input_device, model_name, n_objects=10):
    """
    Test semantic hit under strong/weak/none cue conditions.
    Strong: "belongs to the category of" (contains category scaffold)
    Weak: "is a" / "is an" (basic syntax only)
    None: "is:" / ":" (minimal scaffold)
    """
    log("="*60)
    log("Exp2: Prompt-Robust Semantic Baseline")
    log("="*60)

    cat_words = ["fruit", "fruits", "Fruit"]
    objects = FRUIT_OBJECTS[:n_objects]
    results = {}

    for cue_type, templates in PROMPT_CUES.items():
        layer_counts = {"S0_miss": 0, "S1_lexical": 0, "S2_scaffold": 0,
                        "S3_cont_phrase": 0, "S4_free": 0}
        examples = []
        total = 0

        for obj in objects:
            for tmpl in templates:
                prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" else f"A {obj} {tmpl}"
                enc = safe_encode(tokenizer, prompt, input_device)

                # Pure greedy, no repetition penalty
                gen_kwargs = dict(max_new_tokens=8, do_sample=False)
                with torch.no_grad():
                    gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                                             **gen_kwargs)
                gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                layer = classify_layered(gen_text, prompt, cat_words)
                layer_counts[layer] += 1
                total += 1

                if len(examples) < 4:
                    cont = gen_text[len(prompt):].strip()
                    examples.append({"obj": obj, "tmpl": tmpl, "cont": cont[:40], "layer": layer})

        results[cue_type] = {"total": total, "layers": layer_counts, "examples": examples}
        log(f"\n  [{cue_type}] (n={total}):")
        for k in ["S0_miss", "S1_lexical", "S2_scaffold", "S3_cont_phrase", "S4_free"]:
            c = layer_counts[k]
            pct = 100*c/total if total > 0 else 0
            log(f"    {k}: {c}/{total} ({pct:.0f}%)")
        for ex in examples[:3]:
            log(f"    e.g. [{ex['obj']}] cont='{ex['cont']}' → {ex['layer']}")

    return results


# ============== Exp3: Tangential vs Radial Intervention ==============

def exp3_tangential_radial(model, tokenizer, input_device, model_name, n_objects=8):
    """
    Test RMSNorm geometric filtering hypothesis:
    - Tangential component (⊥ to h) should change logits
    - Radial component (∥ to h) should be absorbed by RMSNorm

    Since hidden_states[-1] is post-norm, we test on hidden_states[-2] (pre-last-layer).
    We apply Δh to h[-2], run last layer + RMSNorm + W_U, and compare.

    Actually, simpler test: use hidden_states[-1] (post-norm) directly.
    Post-norm readout is linear: z = h_post @ W_U.T
    So ANY direction changes logits (no RMSNorm filtering at this point).

    The RMSNorm filtering matters for PRE-NORM interventions.
    We test: perturb h[-2] (pre-last-layer), run through last layer + norm, measure effect.
    """
    log("="*60)
    log("Exp3: Tangential vs Radial Intervention")
    log("="*60)

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    g = get_norm_weight(model, model_name)
    W_U = get_W_U_cached(model, model_name)
    eps = getattr(model.config, 'rms_norm_eps', 1e-6)

    cat_words = ["fruit", "fruits"]
    objects = FRUIT_OBJECTS[:n_objects]
    templates = PROMPT_CUES["strong"][:1]  # use one template

    # Readout direction: q_c = W_U(fruit) - W_U(competitor)
    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    target_id = fruit_ids[0]

    results = {"tangential": [], "radial": [], "full": [], "details": []}

    for obj in objects:
        for tmpl in templates:
            prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" else f"A {obj} {tmpl}"
            enc = safe_encode(tokenizer, prompt, input_device)

            with torch.no_grad():
                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                            output_hidden_states=True)

            h_post = out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()
            model_logits = out.logits[0, -1, :].float().cpu().numpy()

            # Get competitor (top-1 if not target)
            top1_id = int(np.argmax(model_logits))
            if top1_id == target_id:
                top1_id = int(np.argsort(model_logits)[-2])

            w_target = W_U[target_id]
            w_comp = W_U[top1_id]
            # Direction in post-norm space that increases D_c
            d_full = w_target - w_comp  # [d_model]
            d_norm = np.linalg.norm(d_full)
            if d_norm < 1e-8:
                continue
            d_unit = d_full / d_norm

            # Decompose d_unit into tangential and radial w.r.t. h_post
            # Radial: component parallel to h_post
            h_unit = h_post / (np.linalg.norm(h_post) + 1e-8)
            radial_component = np.dot(d_unit, h_unit) * h_unit
            tangential_component = d_unit - radial_component

            # Scale to same magnitude for fair comparison
            alpha = 5.0  # perturbation scale

            # Test 1: Full direction (baseline)
            h_full = h_post + alpha * d_unit
            z_full = correct_readout(h_full, W_U)
            D_c_full = z_full[target_id] - z_full[top1_id]

            # Test 2: Tangential only
            h_tan = h_post + alpha * tangential_component
            z_tan = correct_readout(h_tan, W_U)
            D_c_tan = z_tan[target_id] - z_tan[top1_id]

            # Test 3: Radial only
            h_rad = h_post + alpha * radial_component
            z_rad = correct_readout(h_rad, W_U)
            D_c_rad = z_rad[target_id] - z_rad[top1_id]

            # Baseline D_c
            D_c_base = model_logits[target_id] - model_logits[top1_id]

            delta_full = D_c_full - D_c_base
            delta_tan = D_c_tan - D_c_base
            delta_rad = D_c_rad - D_c_base

            results["full"].append(float(delta_full))
            results["tangential"].append(float(delta_tan))
            results["radial"].append(float(delta_rad))
            results["details"].append({
                "obj": obj,
                "D_c_base": float(D_c_base),
                "delta_full": float(delta_full),
                "delta_tan": float(delta_tan),
                "delta_rad": float(delta_rad),
                "tan_norm": float(np.linalg.norm(tangential_component)),
                "rad_norm": float(np.linalg.norm(radial_component)),
            })

    if results["full"]:
        log(f"  n_samples={len(results['full'])}")
        log(f"  delta_full:      mean={np.mean(results['full']):.4f}, std={np.std(results['full']):.4f}")
        log(f"  delta_tan:       mean={np.mean(results['tangential']):.4f}, std={np.std(results['tangential']):.4f}")
        log(f"  delta_rad:       mean={np.mean(results['radial']):.4f}, std={np.std(results['radial']):.4f}")
        log(f"  tan/rad ratio:   {np.mean(results['tangential'])/(abs(np.mean(results['radial']))+1e-8):.2f}")

        # Note: in POST-NORM space, readout is linear, so BOTH tangential and radial change logits.
        # The RMSNorm filtering only applies to PRE-NORM interventions.
        # This test shows that in post-norm space, the direction component parallel to h
        # still changes logits (because readout is linear, not normed again).
        log(f"\n  NOTE: In post-norm space, readout is LINEAR (z = h@W_U.T).")
        log(f"  Both tangential AND radial components change D_c.")
        log(f"  RMSNorm filtering only applies to PRE-NORM interventions.")

    return {
        "n_samples": len(results["full"]),
        "delta_full_mean": float(np.mean(results["full"])) if results["full"] else None,
        "delta_tan_mean": float(np.mean(results["tangential"])) if results["tangential"] else None,
        "delta_rad_mean": float(np.mean(results["radial"])) if results["radial"] else None,
        "details": results["details"],
    }


# ============== Exp4: Corrected d_traj Causal Validation ==============

def exp4_corrected_dtraj(model, tokenizer, input_device, model_name, n_objects=10):
    """
    Use corrected readout (nonorm) and layered classify to rebuild d_traj.
    Test if d_traj (post-norm) can causally change D_c and semantic hit.
    Use strong+weak+none cues to get both success and failure samples.
    """
    log("="*60)
    log("Exp4: Corrected d_traj Causal Validation")
    log("="*60)

    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U_cached(model, model_name)

    cat_words = ["fruit", "fruits", "Fruit"]
    objects = FRUIT_OBJECTS[:n_objects]

    fruit_ids = tokenizer.encode("fruit", add_special_tokens=False)
    target_id = fruit_ids[0]

    # Collect success/failure using corrected classify
    # Use ALL cue types to get both success and failure
    success_h = []
    fail_h = []
    success_Dc = []
    fail_Dc = []
    all_samples = []

    for cue_type, templates in PROMPT_CUES.items():
        for obj in objects:
            for tmpl in templates:
                prompt = f"An {obj} {tmpl}" if obj[0] in "aeiou" else f"A {obj} {tmpl}"
                enc = safe_encode(tokenizer, prompt, input_device)

                # Generate with pure greedy
                gen_kwargs = dict(max_new_tokens=8, do_sample=False)
                with torch.no_grad():
                    gen_ids = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                                             **gen_kwargs)
                gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                layer = classify_layered(gen_text, prompt, cat_words)

                # Get hidden state (post-norm)
                with torch.no_grad():
                    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                output_hidden_states=True)
                h_post = out.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()

                # Correct D_c using post-norm linear readout
                logits = correct_readout(h_post, W_U)
                top1_id = int(np.argmax(logits))
                if top1_id == target_id:
                    top1_id = int(np.argsort(logits)[-2])
                D_c = logits[target_id] - logits[top1_id]

                # Success = S3 or S4 (continuation phrase or free — model itself forms the phrase)
                # S2 (scaffold) is weaker because prompt provides structure
                is_success = layer in ["S3_cont_phrase", "S4_free"]
                is_fail = layer in ["S0_miss", "S1_lexical"]

                sample = {"obj": obj, "cue": cue_type, "layer": layer, "D_c": float(D_c)}
                all_samples.append(sample)

                if is_success:
                    success_h.append(h_post)
                    success_Dc.append(D_c)
                elif is_fail:
                    fail_h.append(h_post)
                    fail_Dc.append(D_c)

    n_suc = len(success_h)
    n_fail = len(fail_h)
    log(f"  Success (S3+S4): {n_suc}, Fail (S0+S1): {n_fail}")
    log(f"  (S2_scaffold excluded from both to avoid prompt-leakage bias)")

    # Count layers
    layer_counts = {}
    for s in all_samples:
        layer_counts[s["layer"]] = layer_counts.get(s["layer"], 0) + 1
    log(f"  Layer distribution: {layer_counts}")

    if n_suc < 2 or n_fail < 2:
        log(f"  Insufficient data for d_traj")
        return {"n_success": n_suc, "n_failure": n_fail, "error": "insufficient data"}

    # Compute d_traj in post-norm space
    suc_mean = np.mean(success_h, axis=0)
    fail_mean = np.mean(fail_h, axis=0)
    d_traj = suc_mean - fail_mean
    d_norm = np.linalg.norm(d_traj)
    log(f"  d_traj norm (post-norm): {d_norm:.4f}")
    log(f"  Suc D_c mean: {np.mean(success_Dc):.4f}, Fail D_c mean: {np.mean(fail_Dc):.4f}")

    # Test intervention: add/remove d_traj to h_post, measure D_c change
    d_scaled = d_traj / (d_norm + 1e-8) * 5.0
    alphas = [0.5, 1.0, 2.0, 5.0]
    intervention_results = []

    for alpha in alphas:
        for action in ["add", "remove"]:
            deltas = []
            for h in fail_h[:5]:  # test on failure samples
                logits_base = correct_readout(h, W_U)
                top1 = int(np.argmax(logits_base))
                if top1 == target_id:
                    top1 = int(np.argsort(logits_base)[-2])
                D_c_base = logits_base[target_id] - logits_base[top1]

                d = d_scaled if action == "add" else -d_scaled
                h_mod = h + alpha * d
                logits_mod = correct_readout(h_mod, W_U)
                D_c_mod = logits_mod[target_id] - logits_mod[top1]
                deltas.append(float(D_c_mod - D_c_base))

            mean_delta = np.mean(deltas) if deltas else 0
            intervention_results.append({
                "alpha": alpha, "action": action,
                "mean_delta_Dc": float(mean_delta),
                "n": len(deltas),
            })
            log(f"  {action} α={alpha}: ΔD_c={mean_delta:+.4f} (n={len(deltas)})")

    # Also test: does d_traj align with q_c (readout direction)?
    w_target = W_U[target_id]
    # Use mean competitor
    cos_d_qc = np.dot(d_traj, w_target) / (np.linalg.norm(d_traj) * np.linalg.norm(w_target) + 1e-8)
    log(f"  cos(d_traj, W_U(fruit)) = {cos_d_qc:.4f}")

    return {
        "n_success": n_suc,
        "n_failure": n_fail,
        "layer_distribution": layer_counts,
        "d_traj_norm": float(d_norm),
        "suc_Dc_mean": float(np.mean(success_Dc)),
        "fail_Dc_mean": float(np.mean(fail_Dc)),
        "cos_d_traj_w_fruit": float(cos_d_qc),
        "intervention": intervention_results,
    }


# ============== Main ==============

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-objects", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode (fewer objects)")
    args = parser.parse_args()

    if args.smoke:
        args.n_objects = 4
        log("SMOKE TEST MODE: n_objects=4")

    t_start = time.time()
    model, tokenizer, input_device = load_model_bf16(args.model)
    info = get_model_info(model, args.model)
    log(f"  n_layers={info.n_layers}, d_model={info.d_model}")

    results = {"model": args.model, "model_info": {"n_layers": info.n_layers, "d_model": info.d_model}}

    # Exp1: Indexing Audit (always run, fast)
    try:
        results["exp1_indexing_audit"] = exp1_indexing_audit(model, tokenizer, input_device, args.model)
    except Exception as e:
        import traceback
        log(f"Exp1 failed: {e}")
        traceback.print_exc()
        results["exp1_indexing_audit"] = {"error": str(e)}

    # Exp2: Prompt-Robust Baseline
    try:
        results["exp2_prompt_robust"] = exp2_prompt_robust_baseline(
            model, tokenizer, input_device, args.model, args.n_objects)
    except Exception as e:
        import traceback
        log(f"Exp2 failed: {e}")
        traceback.print_exc()
        results["exp2_prompt_robust"] = {"error": str(e)}

    # Exp3: Tangential vs Radial
    try:
        n3 = min(args.n_objects, 8)
        results["exp3_tangential_radial"] = exp3_tangential_radial(
            model, tokenizer, input_device, args.model, n3)
    except Exception as e:
        import traceback
        log(f"Exp3 failed: {e}")
        traceback.print_exc()
        results["exp3_tangential_radial"] = {"error": str(e)}

    # Exp4: Corrected d_traj
    try:
        results["exp4_corrected_dtraj"] = exp4_corrected_dtraj(
            model, tokenizer, input_device, args.model, args.n_objects)
    except Exception as e:
        import traceback
        log(f"Exp4 failed: {e}")
        traceback.print_exc()
        results["exp4_corrected_dtraj"] = {"error": str(e)}

    # Save
    os.makedirs("results/glm5_phase518_corrected_causality", exist_ok=True)
    out_path = f"results/glm5_phase518_corrected_causality/phase518_{args.model}_corrected_causality.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    log(f"\nSaved to {out_path}")

    release_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(f"\nTotal: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
