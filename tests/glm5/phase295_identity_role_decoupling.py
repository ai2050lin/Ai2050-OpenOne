"""
Phase 295: Identity-Role Decoupling
====================================
Goal: Verify whether residual stream transitions from token-identity dominated
      to functional-role dominated across layers.

Core Design: 4-way controlled comparison
  A. Same token, same role    (e.g., "happy" as operand in two affirmatives)
  B. Same token, different role (e.g., "open" as adj vs "open" as verb)
  C. Different token, same role  (e.g., "happy" vs "sad" both as operand)
  D. Different token, different role (e.g., "not" as operator, "happy" as operand)

Measurements per layer:
  1. Cosine similarity of same-token pairs (identity signal)
  2. Cosine similarity of same-role pairs (role signal)
  3. Norm-matched patching: aligned vs misaligned with equal perturbation norm
  4. Token identity probe: linear probe predicting token identity
  5. Role probe: linear probe predicting functional role

Usage:
  python tests/glm5/phase295_identity_role_decoupling.py qwen3
  python tests/glm5/phase295_identity_role_decoupling.py glm4
  python tests/glm5/phase295_identity_role_decoupling.py deepseek7b
"""
import sys, os, gc, time, json, math
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase295_identity_role")
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


# =====================================================================
# STIMULUS DESIGN
# =====================================================================
def build_stimulus_set():
    """
    Build 4-way controlled stimulus pairs.

    Category A: Same token, same role
      Two affirmative sentences with same word in same grammatical role.
      e.g., "she is happy" vs "they are happy" -> "happy" is operand in both

    Category B: Same token, different role
      Same word in different grammatical roles.
      e.g., "the door is open" (adj/operand) vs "they open the door" (verb/operator)

    Category C: Different token, same role
      Different words in same grammatical role.
      e.g., "she is happy" vs "she is sad" -> both operand

    Category D: Different token, different role
      Different words in different roles (negation baseline).
      e.g., "she is happy" (operand) vs "she is not happy" (operator "not")
    """
    stimuli = []

    # ---- Category A: Same token, same role ----
    # Multiple sentence frames with same target word in same role
    # Expand: same adjective in many sentence contexts
    adj_words = ["happy", "ready", "correct", "important", "safe", "simple", "fair",
                 "clear", "possible", "strong", "beautiful", "bright", "calm", "deep",
                 "easy", "fast", "gentle", "hard", "kind", "loud"]
    frames = [
        ("she is {}", "they are {}"),
        ("she is {}", "the result is {}"),
        ("it is {}", "this is {}"),
        ("he seems {}", "the idea seems {}"),
        ("the place is {}", "the plan is {}"),
    ]
    for adj in adj_words:
        for f1, f2 in frames:
            stimuli.append({
                "category": "same_token_same_role",
                "token": adj, "role": "operand",
                "sent1": f1.format(adj), "sent2": f2.format(adj),
            })

    # ---- Category B: Same token, different role ----
    cat_b_frames = [
        # "open" as adjective (operand) vs verb
        ("open", "the door is open", "they open the door", "operand", "verb"),
        ("open", "the window is open", "she opens the window", "operand", "verb"),
        ("open", "the shop is open", "they open the shop", "operand", "verb"),
        # "clear" as adjective vs verb
        ("clear", "the answer is clear", "they clear the table", "operand", "verb"),
        ("clear", "the sky is clear", "she clears the desk", "operand", "verb"),
        # "fair" as adjective vs noun
        ("fair", "the decision is fair", "they go to the fair", "operand", "noun"),
        ("fair", "the game is fair", "we visit the fair", "operand", "noun"),
        # "right" as adjective vs noun
        ("right", "the answer is right", "they know the right", "operand", "noun"),
        ("right", "the choice is right", "she chose the right", "operand", "noun"),
        # "free" as adjective vs verb
        ("free", "the bird is free", "they free the bird", "operand", "verb"),
        ("free", "the man is free", "we free the prisoner", "operand", "verb"),
        # "warm" as adjective vs verb
        ("warm", "the room is warm", "she warms the room", "operand", "verb"),
        ("warm", "the food is warm", "he warms the food", "operand", "verb"),
        # "clean" as adjective vs verb
        ("clean", "the house is clean", "they clean the house", "operand", "verb"),
        ("clean", "the room is clean", "she cleans the room", "operand", "verb"),
        # "light" as adjective vs noun
        ("light", "the bag is light", "turn on the light", "operand", "noun"),
        ("light", "the box is light", "we need more light", "operand", "noun"),
        # "sharp" as adjective vs adverb
        ("sharp", "the knife is sharp", "he turned sharp left", "operand", "modifier"),
        # "cold" as adjective vs noun
        ("cold", "the water is cold", "she caught a cold", "operand", "noun"),
        # "dry" as adjective vs verb
        ("dry", "the cloth is dry", "they dry the cloth", "operand", "verb"),
        # "quiet" as adjective vs noun
        ("quiet", "the room is quiet", "enjoy the quiet", "operand", "noun"),
    ]
    for tok, s1, s2, r1, r2 in cat_b_frames:
        stimuli.append({
            "category": "same_token_diff_role",
            "token": tok,
            "role1": r1, "role2": r2,
            "sent1": s1, "sent2": s2,
        })

    # ---- Category C: Different token, same role ----
    # Antonym pairs in same grammatical position
    antonym_pairs = [
        ("happy", "sad"), ("happy", "tired"), ("good", "bad"), ("good", "great"),
        ("ready", "prepared"), ("simple", "hard"), ("safe", "dangerous"),
        ("tall", "short"), ("hot", "cold"), ("easy", "difficult"),
        ("fast", "slow"), ("strong", "weak"), ("bright", "dark"),
        ("clean", "dirty"), ("old", "young"), ("rich", "poor"),
        ("loud", "quiet"), ("full", "empty"), ("soft", "hard"),
        ("wide", "narrow"), ("deep", "shallow"), ("heavy", "light"),
        ("open", "closed"), ("rough", "smooth"), ("thick", "thin"),
    ]
    role_frames = [
        ("she is {}", "she is {}", "operand"),
        ("it is {}", "it is {}", "operand"),
        ("they seem {}", "they seem {}", "operand"),
        ("the result is {}", "the result is {}", "operand"),
    ]
    for t1, t2 in antonym_pairs:
        for f1, f2, role in role_frames:
            stimuli.append({
                "category": "diff_token_same_role",
                "token1": t1, "token2": t2, "role": role,
                "sent1": f1.format(t1), "sent2": f2.format(t2),
            })
    # Also verb pairs
    verb_pairs = [
        ("agrees", "disagrees"), ("likes", "hates"), ("runs", "walks"),
        ("laughs", "cries"), ("wins", "loses"), ("sings", "dances"),
        ("helps", "hurts"), ("builds", "destroys"), ("gives", "takes"),
    ]
    for v1, v2 in verb_pairs:
        stimuli.append({
            "category": "diff_token_same_role",
            "token1": v1, "token2": v2, "role": "verb",
            "sent1": f"she {v1}", "sent2": f"she {v2}",
        })

    # ---- Category D: Different token, different role (negation baseline) ----
    cat_d_frames = [
        ("she is happy", "she is not happy", "happy", "not", "operand", "operator"),
        ("they are ready", "they are not ready", "ready", "not", "operand", "operator"),
        ("the result is good", "the result is not good", "good", "not", "operand", "operator"),
        ("he agrees", "he does not agree", "agrees", "not", "verb", "operator"),
        ("they came", "they never came", "came", "never", "verb", "operator"),
        ("the task is simple", "the task is not simple", "simple", "not", "operand", "operator"),
        ("she is safe", "she is not safe", "safe", "not", "operand", "operator"),
        ("the water is clear", "the water is not clear", "clear", "not", "operand", "operator"),
        ("he tells the truth", "he never tells the truth", "tells", "never", "verb", "operator"),
        ("there is evidence", "there is no evidence", "evidence", "no", "noun", "operator"),
        ("the area is safe", "the area is not safe", "safe", "not", "operand", "operator"),
        ("the plan is fair", "the plan is not fair", "fair", "not", "operand", "operator"),
        ("she is strong", "she is not strong", "strong", "not", "operand", "operator"),
        ("the answer is correct", "the answer is not correct", "correct", "not", "operand", "operator"),
        ("he is tall", "he is not tall", "tall", "not", "operand", "operator"),
        ("the problem is easy", "the problem is not easy", "easy", "not", "operand", "operator"),
        ("they are fast", "they are not fast", "fast", "not", "operand", "operator"),
        ("the room is bright", "the room is not bright", "bright", "not", "operand", "operator"),
        ("she remembers", "she does not remember", "remembers", "not", "verb", "operator"),
        ("he understands", "he does not understand", "understands", "not", "verb", "operator"),
    ]
    for s1, s2, t1, t2, r1, r2 in cat_d_frames:
        stimuli.append({
            "category": "diff_token_diff_role",
            "token1": t1, "token2": t2,
            "role1": r1, "role2": r2,
            "sent1": s1, "sent2": s2,
        })

    return stimuli


# =====================================================================
# MODEL LOADING (BF16 + device_map=auto + flash)
# =====================================================================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16, device_map=auto)...")

    tok = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = None
    used_attn = "eager"
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=attn_impl,
            )
            used_attn = attn_impl
            break
        except Exception as e:
            log(f"  attn={attn_impl} failed: {str(e)[:80]}")

    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()

    gpu = torch.cuda.memory_allocated() / 1e9
    log(f"  Loaded with attn={used_attn}, GPU={gpu:.1f}GB")

    # Show layer distribution
    layers = get_layers(model)
    nl = len(layers)
    gpu_l, cpu_l = [], []
    for li in range(nl):
        wdev = layers[li].self_attn.o_proj.weight.device
        (gpu_l if wdev.type == 'cuda' else cpu_l).append(li)
    log(f"  GPU layers: {len(gpu_l)}{' (' + str(gpu_l[0]) + '-' + str(gpu_l[-1]) + ')' if gpu_l else ''}, "
        f"CPU: {len(cpu_l)}{' (' + str(cpu_l[0]) + '-' + str(cpu_l[-1]) + ')' if cpu_l else ''}")
    return model, tok


# =====================================================================
# CAPTURE HIDDEN STATES
# =====================================================================
def capture_hidden_states(model, tokenizer, sentences, n_layers, max_len=64):
    """Capture hidden states for a list of sentences. Returns dict of {sent_idx: {layer: tensor}}."""
    input_device = next(model.parameters()).device
    results = {}

    for idx, sent in enumerate(sentences):
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        hs = {}
        for li, h in enumerate(out.hidden_states):
            hs[li] = h.detach().cpu().float()  # [1, seq_len, d_model]

        results[idx] = {
            "hidden": hs,
            "input_ids": inputs["input_ids"].cpu(),
            "tokens": [tokenizer.decode([t]).strip() for t in inputs["input_ids"][0].tolist()],
        }

    return results


# =====================================================================
# MEASUREMENT 1: LAYER-WISE COSINE SIMILARITY ANALYSIS
# =====================================================================
def measure_cosine_similarity(captures, stimuli, n_layers):
    """
    For each stimulus pair, compute cosine similarity of target position
    representations across layers.

    Returns per-category, per-layer similarity distributions.
    """
    results = {
        "same_token_same_role": defaultdict(list),
        "same_token_diff_role": defaultdict(list),
        "diff_token_same_role": defaultdict(list),
        "diff_token_diff_role": defaultdict(list),
    }

    for stim in stimuli:
        cat = stim["category"]
        idx1, idx2 = stim.get("_idx1"), stim.get("_idx2")
        if idx1 is None or idx2 is None:
            continue

        c1, c2 = captures[idx1], captures[idx2]

        # Determine target positions based on category
        if cat == "same_token_same_role":
            p1, p2 = stim["target_pos_1"], stim["target_pos_2"]
        elif cat == "same_token_diff_role":
            p1, p2 = stim["target_pos_1"], stim["target_pos_2"]
        elif cat == "diff_token_same_role":
            p1, p2 = stim["target_pos_1"], stim["target_pos_2"]
        elif cat == "diff_token_diff_role":
            # Use operand position in s1, operator position in s2
            # These need to be resolved dynamically
            p1 = None
            p2 = None
        else:
            continue

        if p1 is None or p2 is None:
            # For category D, find positions dynamically
            tokens1 = c1["tokens"]
            tokens2 = c2["tokens"]
            t1 = stim.get("token1", "").lower()
            t2 = stim.get("token2", "").lower()

            p1 = None
            for i, t in enumerate(tokens1):
                if t1 in t.lower() or t.lower() in t1:
                    p1 = i
                    break
            p2 = None
            for i, t in enumerate(tokens2):
                if t2 in t.lower() or t.lower() in t2:
                    p2 = i
                    break

            if p1 is None or p2 is None:
                continue

        for li in range(n_layers + 1):
            h1 = c1["hidden"].get(li)
            h2 = c2["hidden"].get(li)
            if h1 is None or h2 is None:
                continue

            # Check bounds
            if p1 >= h1.shape[1] or p2 >= h2.shape[1]:
                continue

            v1 = h1[0, p1, :]  # [d_model]
            v2 = h2[0, p2, :]  # [d_model]

            cos_sim = float(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)))
            results[cat][li].append(cos_sim)

    return results


# =====================================================================
# MEASUREMENT 2: NORM-MATCHED PATCHING
# =====================================================================
def norm_matched_patch(model, tokenizer, captures, stimuli, n_layers, max_len=64):
    """
    For negation pairs, compare aligned vs misaligned patching with
    NORM-MATCHED perturbations.

    Key: scale the aligned perturbation to match the norm of misaligned.
    If early layers still show misaligned > aligned after norm matching,
    then token-identity dominance is confirmed.
    """
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    results = []

    # Only use negation-relevant stimuli (Category D + paired with Category A/C)
    neg_pairs = []
    for stim in stimuli:
        if stim["category"] == "diff_token_diff_role":
            idx1 = stim.get("_idx1")
            idx2 = stim.get("_idx2")
            if idx1 is None:
                continue
            neg_pairs.append(stim)

    total = len(neg_pairs) * n_layers * 2  # 2: aligned vs norm-matched-aligned
    done = 0
    t0 = time.time()

    for stim in neg_pairs:
        idx1 = stim.get("_idx1")
        idx2 = stim.get("_idx2")
        c1, c2 = captures[idx1], captures[idx2]

        sent_a = stim["sent1"]  # affirmative
        sent_b = stim["sent2"]  # negated

        # Find operand position in A and B, operator position in B
        tokens_a = c1["tokens"]
        tokens_b = c2["tokens"]

        operand_a = None
        t1 = stim.get("token1", "").lower()
        for i, t in enumerate(tokens_a):
            if t1 in t.lower() or t.lower() in t1:
                operand_a = i
                break

        operand_b = None
        for i, t in enumerate(tokens_b):
            if t1 in t.lower() or t.lower() in t1:
                operand_b = i
                break

        operator_b = None
        t2 = stim.get("token2", "").lower()
        for i, t in enumerate(tokens_b):
            if t2 in t.lower() or t.lower() in t2:
                operator_b = i
                break

        if operand_a is None or operand_b is None:
            continue

        # Get base logits for A and B
        inputs_a = tokenizer(sent_a, return_tensors="pt", truncation=True, max_length=max_len)
        inputs_a = {k: v.to(input_device) for k, v in inputs_a.items()}
        inputs_b = tokenizer(sent_b, return_tensors="pt", truncation=True, max_length=max_len)
        inputs_b = {k: v.to(input_device) for k, v in inputs_b.items()}

        with torch.no_grad():
            logits_a = model(**inputs_a).logits[0, -1, :].detach().cpu().float()
            logits_b = model(**inputs_b).logits[0, -1, :].detach().cpu().float()

        kl_ab = float(F.kl_div(F.log_softmax(logits_b, -1), F.softmax(logits_a, -1), reduction='sum'))
        kl_ab = max(kl_ab, 1e-6)

        for li in range(n_layers):
            # Get B's hidden state at layer li+1
            pv_b = c2["hidden"].get(li + 1)
            pv_a = c1["hidden"].get(li + 1)
            if pv_b is None or pv_a is None:
                continue

            mod = layers[li]

            # ---- 1. ALIGNED operand patching (B[operand] -> A[operand]) ----
            if operand_b < pv_b.shape[1] and operand_a < pv_a.shape[1]:
                delta_aligned = pv_b[0, operand_b, :] - pv_a[0, operand_a, :]
            else:
                delta_aligned = None

            # ---- 2. MISALIGNED operand patching (B[operand] -> A[operand_pos_B]) ----
            if operand_b < pv_b.shape[1] and operand_b < pv_a.shape[1]:
                delta_misaligned = pv_b[0, operand_b, :] - pv_a[0, operand_b, :]
            else:
                delta_misaligned = None

            if delta_aligned is None or delta_misaligned is None:
                continue

            norm_aligned = float(torch.norm(delta_aligned))
            norm_misaligned = float(torch.norm(delta_misaligned))

            # ---- Compute metrics for all 3 conditions ----
            # Condition 1: Raw aligned patching (B[operand_b] -> A[operand_a])
            logits_p1 = _run_patch_pos(model, tokenizer, sent_a, mod, pv_b, operand_b, operand_a, n_layers, max_len)
            np_aligned = _compute_np(logits_p1, logits_a, logits_b, kl_ab) if logits_p1 is not None else None

            # Condition 2: Raw misaligned patching (B[operand_b] -> A[operand_b])
            if operand_b < pv_a.shape[1]:
                logits_p2 = _run_patch_pos(model, tokenizer, sent_a, mod, pv_b, operand_b, operand_b, n_layers, max_len)
                np_misaligned = _compute_np(logits_p2, logits_a, logits_b, kl_ab) if logits_p2 is not None else None
            else:
                np_misaligned = None

            # Condition 3: Norm-matched aligned patching
            # Scale aligned delta to match misaligned norm
            np_norm_matched = None
            if norm_aligned > 1e-8 and norm_misaligned > 1e-8:
                scale = norm_misaligned / norm_aligned
                scale = min(scale, 3.0)  # Don't overscale
                delta_scaled = delta_aligned * scale
                logits_p3 = _run_patch_delta(model, tokenizer, sent_a, mod, pv_a, operand_a, delta_scaled, n_layers, max_len)
                np_norm_matched = _compute_np(logits_p3, logits_a, logits_b, kl_ab) if logits_p3 is not None else None

            results.append({
                "layer": li,
                "stimulus": stim.get("token1", "") + "_" + stim.get("token2", ""),
                "np_aligned": np_aligned,
                "np_misaligned": np_misaligned,
                "np_norm_matched": np_norm_matched,
                "norm_aligned": round(norm_aligned, 4),
                "norm_misaligned": round(norm_misaligned, 4),
                "norm_ratio": round(norm_misaligned / max(norm_aligned, 1e-8), 4),
            })
            done += 1

        if done % 10 == 0:
            el = time.time() - t0
            rate = done / max(el, 1)
            eta = (total - done) / rate if rate > 0 else 0
            log(f"  NormMatch {done}/{total} ({rate:.1f}/s) ETA={eta:.0f}s")

    log(f"  Norm-matched patching: {len(results)} results, {time.time()-t0:.0f}s")
    return results


def _run_patch_pos(model, tokenizer, sent_run, mod, pv_src, src_pos, dst_pos, n_layers, max_len):
    """Run a patched forward pass: replace dst_pos in running sentence with src_pos from pv_src."""
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent_run, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    def patch_hook(m, inp, out):
        ref = out[0] if isinstance(out, tuple) else out
        pv = pv_src.to(ref.device).to(ref.dtype)
        if isinstance(out, tuple):
            no = (out[0].clone(),) + out[1:]
            if src_pos < pv.shape[1] and dst_pos < no[0].shape[1]:
                no[0][:, dst_pos, :] = pv[:, src_pos, :]
            return no
        no = out.clone()
        if src_pos < pv.shape[1] and dst_pos < no.shape[1]:
            no[:, dst_pos, :] = pv[:, src_pos, :]
        return no

    hook = mod.register_forward_hook(patch_hook)
    try:
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :].detach().cpu().float().clone()
    except Exception as e:
        log(f"    Patch ERR: {e}")
        logits = None
    hook.remove()
    return logits


def _run_patch_delta(model, tokenizer, sent_run, mod, base_hidden, dst_pos, delta, n_layers, max_len):
    """Run a patched forward pass: add delta to dst_pos in running sentence."""
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent_run, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    def patch_hook(m, inp, out):
        ref = out[0] if isinstance(out, tuple) else out
        bh = base_hidden.to(ref.device).to(ref.dtype)
        dt = delta.to(ref.device).to(ref.dtype)
        if isinstance(out, tuple):
            no = (out[0].clone(),) + out[1:]
            if dst_pos < no[0].shape[1]:
                no[0][:, dst_pos, :] = bh[:, dst_pos, :] + dt.unsqueeze(0)
            return no
        no = out.clone()
        if dst_pos < no.shape[1]:
            no[:, dst_pos, :] = bh[:, dst_pos, :] + dt.unsqueeze(0)
        return no

    hook = mod.register_forward_hook(patch_hook)
    try:
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :].detach().cpu().float().clone()
    except Exception as e:
        log(f"    Patch ERR: {e}")
        logits = None
    hook.remove()
    return logits


def _compute_np(logits_p, logits_a, logits_b, kl_ab):
    """Compute natural progress metric."""
    kp = float(F.kl_div(F.log_softmax(logits_p, -1), F.softmax(logits_b, -1), reduction='sum'))
    kr = min(kp / max(kl_ab, 1e-6), 100.0)
    db = logits_b - logits_a
    dp = logits_p - logits_a
    nb, np_ = float(torch.norm(db)), float(torch.norm(dp))
    cd = float(torch.dot(dp, db) / (nb * np_)) if nb > 1e-8 and np_ > 1e-8 else 0
    prog = cd * min(np_ / nb, 2.0) if nb > 1e-8 else 0
    return round(prog / (1.0 + kr), 6)


# =====================================================================
# MEASUREMENT 3: LINEAR PROBE (token identity vs role)
# =====================================================================
def train_linear_probes(captures, stimuli, n_layers, d_model):
    """
    Train simple linear probes to predict:
    1. Token identity (which word is at this position)
    2. Functional role (what role does this position play)

    Use logistic regression on hidden states.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    results = {"token_probe": {}, "role_probe": {}}

    # Collect data for each layer
    for li in range(n_layers + 1):
        X_token, y_token = [], []
        X_role, y_role = [], []

        for stim in stimuli:
            cat = stim["category"]
            idx1 = stim.get("_idx1")
            idx2 = stim.get("_idx2")
            if idx1 is None:
                continue

            # --- Token identity labels ---
            # Collect (hidden_state, token_label) pairs
            for idx, label_key in [(idx1, "token1" if "token1" in stim else "token"),
                                   (idx2, "token2" if "token2" in stim else "token")]:
                c = captures[idx]
                h = c["hidden"].get(li)
                if h is None:
                    continue

                if cat == "same_token_same_role":
                    tok_label = stim["token"]
                    pos = stim.get("target_pos_1") if idx == idx1 else stim.get("target_pos_2")
                elif cat == "same_token_diff_role":
                    tok_label = stim["token"]
                    pos = stim.get("target_pos_1") if idx == idx1 else stim.get("target_pos_2")
                elif cat == "diff_token_same_role":
                    tok_label = stim["token1"] if idx == idx1 else stim["token2"]
                    pos = stim.get("target_pos_1") if idx == idx1 else stim.get("target_pos_2")
                elif cat == "diff_token_diff_role":
                    tok_label = stim["token1"] if idx == idx1 else stim["token2"]
                    pos = None  # Need to find dynamically
                else:
                    continue

                if pos is None:
                    # Find position from tokens
                    target = tok_label.lower()
                    for i, t in enumerate(c["tokens"]):
                        if target in t.lower() or t.lower() in target:
                            pos = i
                            break

                if pos is not None and pos < h.shape[1]:
                    X_token.append(h[0, pos, :].numpy())
                    y_token.append(tok_label)

                # --- Role labels ---
                if cat == "same_token_same_role":
                    role_label = stim["role"]
                elif cat == "same_token_diff_role":
                    role_label = stim["role1"] if idx == idx1 else stim["role2"]
                elif cat == "diff_token_same_role":
                    role_label = stim["role"]
                elif cat == "diff_token_diff_role":
                    role_label = stim["role1"] if idx == idx1 else stim["role2"]
                else:
                    continue

                if pos is not None and pos < h.shape[1]:
                    X_role.append(h[0, pos, :].numpy())
                    y_role.append(role_label)

        # Train probes if enough data
        if len(set(y_token)) >= 2 and len(y_token) >= 4:
            try:
                n_cv = min(3, min(len(y_token) // 2, len(set(y_token))))
                if n_cv < 2:
                    n_cv = 2
                clf = LogisticRegression(max_iter=200, solver='lbfgs')
                scores = cross_val_score(clf, np.array(X_token), y_token, cv=n_cv)
                results["token_probe"][li] = {
                    "accuracy": round(float(scores.mean()), 4),
                    "n_samples": len(y_token),
                    "n_classes": len(set(y_token)),
                }
            except Exception as e:
                results["token_probe"][li] = {"error": str(e)[:80]}

        if len(set(y_role)) >= 2 and len(y_role) >= 4:
            try:
                n_cv = min(3, min(len(y_role) // 2, len(set(y_role))))
                if n_cv < 2:
                    n_cv = 2
                clf = LogisticRegression(max_iter=200, solver='lbfgs')
                scores = cross_val_score(clf, np.array(X_role), y_role, cv=n_cv)
                results["role_probe"][li] = {
                    "accuracy": round(float(scores.mean()), 4),
                    "n_samples": len(y_role),
                    "n_classes": len(set(y_role)),
                }
            except Exception as e:
                results["role_probe"][li] = {"error": str(e)[:80]}

    return results


# =====================================================================
# MEASUREMENT 4: INTRA- VS INTER- TOKEN/ROLE DISTANCE
# =====================================================================
def measure_distance_structure(captures, stimuli, n_layers):
    """
    For each layer, compute:
    - Avg cosine sim of same-token pairs (intra-token)
    - Avg cosine sim of different-token pairs (inter-token)
    - Avg cosine sim of same-role pairs (intra-role)
    - Avg cosine sim of different-role pairs (inter-role)

    If identity dominates: intra-token >> inter-token, intra-role ≈ inter-role
    If role dominates: intra-role >> inter-role, intra-token ≈ inter-token
    """
    results = {
        "intra_token": defaultdict(list),
        "inter_token": defaultdict(list),
        "intra_role": defaultdict(list),
        "inter_role": defaultdict(list),
    }

    # Collect all (layer, position, token, role) tuples
    entries = []
    for stim in stimuli:
        cat = stim["category"]
        for idx_key, tok_key, role_key, pos_key in [
            ("_idx1", "token1" if "token1" in stim else "token",
             "role1" if "role1" in stim else "role", "target_pos_1"),
            ("_idx2", "token2" if "token2" in stim else "token",
             "role2" if "role2" in stim else "role", "target_pos_2"),
        ]:
            idx = stim.get(idx_key)
            if idx is None:
                continue
            c = captures[idx]
            tok_label = stim.get(tok_key)
            role_label = stim.get(role_key)
            pos = stim.get(pos_key)

            if pos is None:
                target = tok_label.lower() if tok_label else ""
                for i, t in enumerate(c["tokens"]):
                    if target in t.lower() or t.lower() in target:
                        pos = i
                        break

            if pos is not None and tok_label and role_label:
                entries.append({
                    "capture_idx": idx, "pos": pos,
                    "token": tok_label, "role": role_label,
                })

    # Compute pairwise similarities
    for li in range(n_layers + 1):
        vecs = []
        for e in entries:
            c = captures[e["capture_idx"]]
            h = c["hidden"].get(li)
            if h is None or e["pos"] >= h.shape[1]:
                vecs.append(None)
                continue
            vecs.append(h[0, e["pos"], :])

        for i in range(len(entries)):
            if vecs[i] is None:
                continue
            for j in range(i + 1, len(entries)):
                if vecs[j] is None:
                    continue

                cos = float(F.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)))

                same_tok = entries[i]["token"].lower() == entries[j]["token"].lower()
                same_role = entries[i]["role"].lower() == entries[j]["role"].lower()

                if same_tok:
                    results["intra_token"][li].append(cos)
                else:
                    results["inter_token"][li].append(cos)

                if same_role:
                    results["intra_role"][li].append(cos)
                else:
                    results["inter_role"][li].append(cos)

    return results


# =====================================================================
# RESOLVE TOKEN POSITIONS
# =====================================================================
def resolve_positions(stimuli, tokenizer):
    """Resolve token positions for all stimuli using tokenizer."""
    resolved = []
    for stim in stimuli:
        s1, s2 = stim["sent1"], stim["sent2"]
        toks1 = tokenizer.encode(s1, add_special_tokens=True)
        toks2 = tokenizer.encode(s2, add_special_tokens=True)
        dec1 = [tokenizer.decode([t]).strip().lower() for t in toks1]
        dec2 = [tokenizer.decode([t]).strip().lower() for t in toks2]

        new_stim = dict(stim)
        new_stim["toks1"] = toks1
        new_stim["toks2"] = toks2
        new_stim["dec1"] = dec1
        new_stim["dec2"] = dec2
        new_stim["len1"] = len(toks1)
        new_stim["len2"] = len(toks2)

        # Resolve target positions by matching tokens
        cat = stim["category"]
        if cat == "same_token_same_role":
            target = stim["token"].lower()
            p1 = _find_token_pos(dec1, target)
            p2 = _find_token_pos(dec2, target)
            new_stim["target_pos_1"] = p1
            new_stim["target_pos_2"] = p2
        elif cat == "same_token_diff_role":
            target = stim["token"].lower()
            p1 = _find_token_pos(dec1, target)
            p2 = _find_token_pos(dec2, target)
            new_stim["target_pos_1"] = p1
            new_stim["target_pos_2"] = p2
        elif cat == "diff_token_same_role":
            t1 = stim["token1"].lower()
            t2 = stim["token2"].lower()
            new_stim["target_pos_1"] = _find_token_pos(dec1, t1)
            new_stim["target_pos_2"] = _find_token_pos(dec2, t2)
        elif cat == "diff_token_diff_role":
            t1 = stim["token1"].lower()
            t2 = stim["token2"].lower()
            new_stim["target_pos_1"] = _find_token_pos(dec1, t1)
            new_stim["target_pos_2"] = _find_token_pos(dec2, t2)

        resolved.append(new_stim)
    return resolved


def _find_token_pos(decoded_tokens, target):
    """Find position of target token in decoded token list."""
    target_lower = target.lower()
    # Exact match
    for i, t in enumerate(decoded_tokens):
        if t == target_lower:
            return i
    # Substring match
    for i, t in enumerate(decoded_tokens):
        if target_lower in t or t in target_lower:
            return i
    # Prefix match
    if len(target_lower) >= 2:
        for i, t in enumerate(decoded_tokens):
            if target_lower[:3] in t or t[:3] in target_lower:
                return i
    return None


# =====================================================================
# MAIN
# =====================================================================
def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase295_{model_name}.txt"
    _log_file = str(log_file)

    log(f"Phase 295: Identity-Role Decoupling — {model_name}")
    log(f"=" * 60)

    # ---- 1. Load model ----
    model, tok = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    nl = info.n_layers
    d_model = info.d_model
    log(f"  n_layers={nl}, d_model={d_model}, class={info.model_class}")

    # ---- 2. Build and resolve stimuli ----
    stimuli = build_stimulus_set()
    stimuli = resolve_positions(stimuli, tok)

    # Filter out stimuli with unresolved positions
    valid_stimuli = []
    for s in stimuli:
        cat = s["category"]
        if cat in ("same_token_same_role", "same_token_diff_role", "diff_token_same_role"):
            if s.get("target_pos_1") is not None and s.get("target_pos_2") is not None:
                valid_stimuli.append(s)
            else:
                log(f"  SKIP {cat}: {s.get('token', s.get('token1', ''))} — position unresolved")
        elif cat == "diff_token_diff_role":
            if s.get("target_pos_1") is not None and s.get("target_pos_2") is not None:
                valid_stimuli.append(s)
            else:
                log(f"  SKIP {cat}: {s.get('token1', '')}/{s.get('token2', '')} — position unresolved")

    # Collect unique sentences
    all_sentences = []
    sent_to_idx = {}
    for s in valid_stimuli:
        for key in ["sent1", "sent2"]:
            sent = s[key]
            if sent not in sent_to_idx:
                sent_to_idx[sent] = len(all_sentences)
                all_sentences.append(sent)
        s["_idx1"] = sent_to_idx[s["sent1"]]
        s["_idx2"] = sent_to_idx[s["sent2"]]

    log(f"  Valid stimuli: {len(valid_stimuli)} pairs, {len(all_sentences)} unique sentences")
    for cat in ["same_token_same_role", "same_token_diff_role", "diff_token_same_role", "diff_token_diff_role"]:
        n = sum(1 for s in valid_stimuli if s["category"] == cat)
        log(f"    {cat}: {n}")

    # ---- 3. Capture hidden states ----
    log(f"\n--- Capturing hidden states for {len(all_sentences)} sentences ---")
    t0 = time.time()
    captures = {}
    for i, sent in enumerate(all_sentences):
        captures[i] = _capture_single(model, tok, sent, nl)
        if (i + 1) % 10 == 0:
            log(f"  Captured {i+1}/{len(all_sentences)} sentences ({time.time()-t0:.0f}s)")
            gc.collect()
            torch.cuda.empty_cache()
    log(f"  Captured all sentences in {time.time()-t0:.0f}s")

    # ---- 4. Measurement 1: Cosine Similarity ----
    log(f"\n--- Measurement 1: Layer-wise Cosine Similarity ---")
    cos_results = measure_cosine_similarity(captures, valid_stimuli, nl)
    for cat in ["same_token_same_role", "same_token_diff_role", "diff_token_same_role", "diff_token_diff_role"]:
        n_pairs = len(cos_results[cat].get(0, []))
        log(f"  {cat}: {n_pairs} pairs")
        if n_pairs > 0:
            for li in [0, nl//4, nl//2, 3*nl//4, nl]:
                if li in cos_results[cat] and cos_results[cat][li]:
                    avg = np.mean(cos_results[cat][li])
                    log(f"    L{li}: avg_cos={avg:.4f}")

    # ---- 5. Measurement 2: Norm-Matched Patching ----
    log(f"\n--- Measurement 2: Norm-Matched Patching ---")
    nm_results = norm_matched_patch(model, tok, captures, valid_stimuli, nl)

    # ---- 6. Measurement 3: Linear Probes ----
    log(f"\n--- Measurement 3: Linear Probes (token identity vs role) ---")
    probe_results = train_linear_probes(captures, valid_stimuli, nl, d_model)

    # Print probe results summary
    log("  Token identity probe accuracy:")
    for li in sorted(probe_results["token_probe"].keys()):
        r = probe_results["token_probe"][li]
        if "accuracy" in r:
            log(f"    L{li}: {r['accuracy']:.4f} (n={r['n_samples']}, classes={r['n_classes']})")
    log("  Role probe accuracy:")
    for li in sorted(probe_results["role_probe"].keys()):
        r = probe_results["role_probe"][li]
        if "accuracy" in r:
            log(f"    L{li}: {r['accuracy']:.4f} (n={r['n_samples']}, classes={r['n_classes']})")

    # ---- 7. Measurement 4: Distance Structure ----
    log(f"\n--- Measurement 4: Distance Structure ---")
    dist_results = measure_distance_structure(captures, valid_stimuli, nl)
    log("  Intra-token vs Inter-token cosine sim:")
    for li in [0, nl//4, nl//2, 3*nl//4, nl]:
        intra = np.mean(dist_results["intra_token"][li]) if dist_results["intra_token"][li] else 0
        inter = np.mean(dist_results["inter_token"][li]) if dist_results["inter_token"][li] else 0
        gap_t = intra - inter
        log(f"    L{li}: intra={intra:.4f} inter={inter:.4f} gap={gap_t:+.4f}")
    log("  Intra-role vs Inter-role cosine sim:")
    for li in [0, nl//4, nl//2, 3*nl//4, nl]:
        intra = np.mean(dist_results["intra_role"][li]) if dist_results["intra_role"][li] else 0
        inter = np.mean(dist_results["inter_role"][li]) if dist_results["inter_role"][li] else 0
        gap_r = intra - inter
        log(f"    L{li}: intra={intra:.4f} inter={inter:.4f} gap={gap_r:+.4f}")

    # ---- 8. Save results ----
    log(f"\n--- Saving results ---")

    # Convert defaultdict to regular dict for JSON serialization
    cos_out = {}
    for cat in cos_results:
        cos_out[cat] = {str(li): [round(v, 6) for v in vals]
                        for li, vals in cos_results[cat].items()}

    dist_out = {}
    for key in dist_results:
        dist_out[key] = {str(li): [round(v, 6) for v in vals]
                         for li, vals in dist_results[key].items()}

    output = {
        "model": model_name,
        "n_layers": nl,
        "d_model": d_model,
        "n_stimuli": len(valid_stimuli),
        "n_sentences": len(all_sentences),
        "cosine_similarity": cos_out,
        "norm_matched_patching": nm_results,
        "linear_probes": probe_results,
        "distance_structure": dist_out,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = RESULT_DIR / f"{model_name}_identity_role.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"  Saved to {out_path}")

    # ---- 9. Print summary ----
    log(f"\n{'='*60}")
    log(f"PHASE 295 SUMMARY — {model_name}")
    log(f"{'='*60}")

    # Identity vs Role signal per layer
    log("\n  Identity signal (intra_token - inter_token gap):")
    for li in range(nl + 1):
        intra_t = np.mean(dist_results["intra_token"][li]) if dist_results["intra_token"][li] else 0
        inter_t = np.mean(dist_results["inter_token"][li]) if dist_results["inter_token"][li] else 0
        intra_r = np.mean(dist_results["intra_role"][li]) if dist_results["intra_role"][li] else 0
        inter_r = np.mean(dist_results["inter_role"][li]) if dist_results["inter_role"][li] else 0
        gap_t = intra_t - inter_t
        gap_r = intra_r - inter_r
        dom = "TOKEN" if gap_t > gap_r else "ROLE"
        if li % 4 == 0 or li >= nl - 2:
            log(f"    L{li:2d}: tok_gap={gap_t:+.4f} role_gap={gap_r:+.4f} dominant={dom}")

    # Norm-matched results summary
    if nm_results:
        log("\n  Norm-matched patching (early vs late):")
        early = [r for r in nm_results if r["layer"] <= nl // 3]
        late = [r for r in nm_results if r["layer"] > 2 * nl // 3]
        for label, subset in [("Early", early), ("Late", late)]:
            if not subset:
                continue
            avg_a = np.mean([r["np_aligned"] for r in subset if r["np_aligned"] is not None] or [0])
            avg_m = np.mean([r["np_misaligned"] for r in subset if r["np_misaligned"] is not None] or [0])
            avg_nm = np.mean([r["np_norm_matched"] for r in subset if r["np_norm_matched"] is not None] or [0])
            avg_nr = np.mean([r["norm_ratio"] for r in subset if r.get("norm_ratio")] or [0])
            log(f"    {label}: aligned={avg_a:.4f} misaligned={avg_m:.4f} "
                f"norm_matched={avg_nm:.4f} norm_ratio={avg_nr:.2f}")

    # ---- 10. Release model ----
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  Model released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    log(f"Phase 295 complete for {model_name}!")


def _capture_single(model, tokenizer, sent, n_layers, max_len=64):
    """Capture hidden states for a single sentence."""
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    hs = {}
    for li, h in enumerate(out.hidden_states):
        hs[li] = h.detach().cpu().float()

    return {
        "hidden": hs,
        "input_ids": inputs["input_ids"].cpu(),
        "tokens": [tokenizer.decode([t]).strip() for t in inputs["input_ids"][0].tolist()],
    }


if __name__ == "__main__":
    main()
