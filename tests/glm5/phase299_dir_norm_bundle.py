"""
Phase 299: Direction-Norm Dual-Channel Decomposition & Role-Frame Bundle Causal Test
=====================================================================================
Goal: Two critical experiments in one script:

Part A - Direction-Norm Dual-Channel:
  For each role increment Δh = ||Δh|| · u_direction
  Test 4 conditions:
    1. correct_dir + correct_norm (full replacement)
    2. correct_dir + wrong_norm (direction only)
    3. wrong_dir + correct_norm (norm only)
    4. random_dir + correct_norm (norm gate test)
  Question: Is DS7B direction-norm coupled?

Part B - Role-Frame Bundle Causal:
  Test 4 replacement conditions:
    1. role only (replace R, keep F and I×F)
    2. frame only (replace F, keep R and I×F)
    3. role + frame (replace both)
    4. interaction only (replace I×F residual)
  Question: Is the real causal unit role-frame bundle?

Usage:
  python tests/glm5/phase299_dir_norm_bundle.py qwen3
  python tests/glm5/phase299_dir_norm_bundle.py glm4
  python tests/glm5/phase299_dir_norm_bundle.py deepseek7b
"""
import sys, os, gc, time, json, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase299_dir_norm_bundle")
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
# STIMULUS — Extended from Phase 297/298
# =====================================================================
def build_stimuli():
    """Same stimuli as Phase 297/298 for consistency"""
    stimuli = []
    adj_verb_tokens = {
        "open": {"adj": {"P1": ["the door is open", "the gate is open"], "P2": ["the door remains open", "the gate remains open"], "P3": ["the open door", "the open gate"], "P4": ["the shop seemed open", "the road seemed open"]},
                 "verb": {"P1": ["they open the door", "they open the gate"], "P2": ["we open the door", "we open the gate"], "P3": ["the door will open", "the gate will open"], "P4": ["they began to open the shop", "they began to open the road"]}},
        "clear": {"adj": {"P1": ["the path is clear", "the road is clear"], "P2": ["the path remains clear", "the road remains clear"], "P3": ["the clear path", "the clear road"], "P4": ["the desk seemed clear", "the table seemed clear"]},
                  "verb": {"P1": ["they clear the path", "they clear the road"], "P2": ["we clear the path", "we clear the road"], "P3": ["the path will clear", "the road will clear"], "P4": ["they began to clear the desk", "they began to clear the table"]}},
        "warm": {"adj": {"P1": ["the room is warm", "the house is warm"], "P2": ["the room remains warm", "the house remains warm"], "P3": ["the warm room", "the warm house"], "P4": ["the water seemed warm", "the food seemed warm"]},
                 "verb": {"P1": ["they warm the room", "they warm the house"], "P2": ["we warm the room", "we warm the house"], "P3": ["the room will warm", "the house will warm"], "P4": ["they began to warm the water", "they began to warm the food"]}},
        "clean": {"adj": {"P1": ["the floor is clean", "the table is clean"], "P2": ["the floor remains clean", "the table remains clean"], "P3": ["the clean floor", "the clean table"], "P4": ["the room seemed clean", "the house seemed clean"]},
                  "verb": {"P1": ["they clean the floor", "they clean the table"], "P2": ["we clean the floor", "we clean the table"], "P3": ["the floor will clean", "the table will clean"], "P4": ["they began to clean the room", "they began to clean the house"]}},
    }
    adj_noun_tokens = {
        "light": {"adj": {"P1": ["the bag is light", "the box is light"], "P2": ["the bag remains light", "the box remains light"], "P3": ["the light bag", "the light box"], "P4": ["the load seemed light", "the dress seemed light"]},
                  "noun": {"P1": ["the light is bright", "the light is warm"], "P2": ["that light is bright", "that light is warm"], "P3": ["near the light", "by the light"], "P4": ["they saw the light", "they found the light"]}},
        "cold": {"adj": {"P1": ["the water is cold", "the wind is cold"], "P2": ["the water remains cold", "the wind remains cold"], "P3": ["the cold water", "the cold wind"], "P4": ["the room seemed cold", "the air seemed cold"]},
                 "noun": {"P1": ["the cold is severe", "the cold is bitter"], "P2": ["that cold is severe", "that cold is bitter"], "P3": ["in the cold", "despite the cold"], "P4": ["they felt the cold", "they noticed the cold"]}},
    }
    noun_verb_tokens = {
        "fire": {"noun": {"P1": ["the fire is hot", "the fire is big"], "P2": ["that fire is hot", "that fire is big"], "P3": ["near the fire", "by the fire"], "P4": ["they saw the fire", "they started the fire"]},
                 "verb": {"P1": ["they fire the gun", "they fire the worker"], "P2": ["they will fire the gun", "they will fire the worker"], "P3": ["the gun will fire", "the engine will fire"], "P4": ["they began to fire the gun", "they began to fire the worker"]}},
        "record": {"noun": {"P1": ["the record is old", "the record is broken"], "P2": ["that record is old", "that record is broken"], "P3": ["on the record", "for the record"], "P4": ["they broke the record", "they set the record"]},
                   "verb": {"P1": ["they record music", "they record data"], "P2": ["they will record music", "they will record data"], "P3": ["the device will record", "the system will record"], "P4": ["they began to record music", "they began to record data"]}},
    }
    all_tokens = {}; all_tokens.update(adj_verb_tokens); all_tokens.update(adj_noun_tokens); all_tokens.update(noun_verb_tokens)
    for token, roles in all_tokens.items():
        rp = "adj_verb" if token in adj_verb_tokens else ("adj_noun" if token in adj_noun_tokens else "noun_verb")
        for role, pairs in roles.items():
            for pair_label, sentences in pairs.items():
                for sent in sentences:
                    stimuli.append({"sentence": sent, "target_word": token, "token_label": token,
                                    "role_label": role, "pair_label": pair_label, "role_pair": rp})
    return stimuli

def build_causal_stimuli():
    """Extended causal test pairs with more variants"""
    test_pairs = [
        # adj_verb
        ("the window is open", "open", "adj", "adj_verb"), ("they open the window", "open", "verb", "adj_verb"),
        ("the market is open", "open", "adj", "adj_verb"), ("they open the market", "open", "verb", "adj_verb"),
        ("the field is clear", "clear", "adj", "adj_verb"), ("they clear the field", "clear", "verb", "adj_verb"),
        ("the meal is warm", "warm", "adj", "adj_verb"), ("they warm the meal", "warm", "verb", "adj_verb"),
        ("the shirt is clean", "clean", "adj", "adj_verb"), ("they clean the shirt", "clean", "verb", "adj_verb"),
        # adj_noun
        ("the feather is light", "light", "adj", "adj_noun"), ("the light is on", "light", "noun", "adj_noun"),
        ("the drink is cold", "cold", "adj", "adj_noun"), ("the cold is harsh", "cold", "noun", "adj_noun"),
        # noun_verb
        ("the fire is bright", "fire", "noun", "noun_verb"), ("they fire the employee", "fire", "verb", "noun_verb"),
        ("the record is famous", "record", "noun", "noun_verb"), ("they record the song", "record", "verb", "noun_verb"),
    ]
    stimuli = []
    for sent, target, role, rp in test_pairs:
        stimuli.append({"sentence": sent, "target_word": target, "token_label": target,
                        "role_label": role, "pair_label": "test", "role_pair": rp, "group": "causal_test"})
    return stimuli

# =====================================================================
# MODEL LOADING
# =====================================================================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = None
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=attn_impl)
            break
        except Exception as e:
            log(f"  attn_impl={attn_impl} failed: {e}")
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    log(f"  Loaded. GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return model, tok

def _capture_single(model, tokenizer, sent, max_len=64):
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = {li: h.detach().cpu().float() for li, h in enumerate(out.hidden_states)}
    logits = out.logits.detach().cpu().float()
    return {"hidden": hs, "logits": logits}

def _find_token_pos(decoded_tokens, target):
    target_lower = target.lower()
    for i, t in enumerate(decoded_tokens):
        if t == target_lower: return i
    for i, t in enumerate(decoded_tokens):
        if target_lower in t or t in target_lower: return i
    if len(target_lower) >= 2:
        for i, t in enumerate(decoded_tokens):
            if target_lower[:3] in t or t[:3] in target_lower: return i
    return None

def resolve_positions(stimuli, tokenizer):
    resolved = []
    for stim in stimuli:
        toks = tokenizer.encode(stim["sentence"], add_special_tokens=True)
        dec = [tokenizer.decode([t]).strip().lower() for t in toks]
        pos = _find_token_pos(dec, stim["target_word"])
        if pos is not None:
            new_stim = dict(stim); new_stim["target_pos"] = pos; resolved.append(new_stim)
    return resolved

# =====================================================================
# ACTIVATION PATCHING UTILITIES
# =====================================================================
def run_with_patched_hidden(model, tokenizer, sent, layer_idx, pos, patch_vec, max_len=64):
    """Run model with a patch added to hidden state at (layer_idx, pos).
    patch_vec: numpy array [d_model] to ADD to the hidden state.
    Returns logits tensor."""
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(input_device)

    captured = {}
    def hook_fn(module, input, output):
        captured["hidden"] = output[0].detach()

    layers = get_layers(model)
    handle = layers[layer_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        try:
            out = model(input_ids=input_ids, output_hidden_states=False)
        except Exception as e:
            handle.remove()
            raise e

    handle.remove()

    if "hidden" not in captured:
        return None

    # Get the hidden state and patch it
    h = captured["hidden"].clone()  # [1, seq_len, d_model]
    patch_tensor = torch.tensor(patch_vec, dtype=h.dtype, device=h.device)
    h[0, pos, :] += patch_tensor

    # Re-run from patched layer onward
    # We need to run the remaining layers manually
    # Strategy: capture at layer_idx, patch, then run remaining layers
    position_ids = torch.arange(input_ids.shape[1], device=input_device).unsqueeze(0)

    # Get embedding
    embed_layer = model.get_input_embeddings()
    inputs_embeds = embed_layer(input_ids)

    # Run through layers 0..layer_idx-1 normally
    # Then at layer_idx, inject the patched hidden
    # Then run layers layer_idx+1..end

    # Actually, a simpler approach: use hooks to inject at the right point
    patched_logits = None
    injection_done = [False]

    def inject_hook(module, input, output):
        if not injection_done[0]:
            out_tuple = list(output)
            out_tuple[0] = out_tuple[0].clone()
            out_tuple[0][0, pos, :] += patch_tensor.to(out_tuple[0].dtype)
            injection_done[0] = True
            return tuple(out_tuple)
        return output

    handle2 = layers[layer_idx].register_forward_hook(inject_hook)

    with torch.no_grad():
        try:
            out2 = model(input_ids=input_ids, output_hidden_states=False)
            patched_logits = out2.logits.detach().cpu().float()
        except Exception as e:
            log(f"  Patched forward failed: {e}")

    handle2.remove()
    return patched_logits

def cosine_sim(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

# =====================================================================
# MAIN
# =====================================================================
def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase299_{model_name}.txt"
    _log_file = str(log_file)
    log(f"Phase 299: Direction-Norm Dual-Channel & Role-Frame Bundle -- {model_name}")

    model, tok = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    nl = info.n_layers; d_model = info.d_model
    log(f"  n_layers={nl}, d_model={d_model}")

    sub_stimuli = resolve_positions(build_stimuli(), tok)
    causal_stimuli = resolve_positions(build_causal_stimuli(), tok)

    # Deduplicate sentences
    all_sentences = []; sent_to_idx = {}
    for s in sub_stimuli + causal_stimuli:
        sent = s["sentence"]
        if sent not in sent_to_idx:
            sent_to_idx[sent] = len(all_sentences); all_sentences.append(sent)
        s["_idx"] = sent_to_idx[sent]

    log(f"Capturing {len(all_sentences)} sentences...")
    t0 = time.time()
    captures = {}
    for i, sent in enumerate(all_sentences):
        captures[i] = _capture_single(model, tok, sent)
        if (i + 1) % 20 == 0:
            el = time.time() - t0; rate = (i + 1) / max(el, 1)
            log(f"  {i+1}/{len(all_sentences)} ({rate:.1f}/s) ETA={(len(all_sentences)-i-1)/rate:.0f}s")
            gc.collect(); torch.cuda.empty_cache()
    log(f"Done capturing in {time.time()-t0:.0f}s")

    # =====================================================================
    # Organize data
    # =====================================================================
    obs = defaultdict(list)
    for stim in sub_stimuli:
        token = stim["token_label"]; role = stim["role_label"]; pair = stim["pair_label"]
        idx = stim.get("_idx"); pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role, pair)].append((idx, pos))

    token_roles = defaultdict(set); token_pairs = defaultdict(set); token_rp = {}
    for stim in sub_stimuli:
        token_roles[stim["token_label"]].add(stim["role_label"])
        token_pairs[stim["token_label"]].add(stim["pair_label"])
        token_rp[stim["token_label"]] = stim.get("role_pair", "")
    dual_tokens = sorted([t for t, roles in token_roles.items() if len(roles) >= 2])

    # Organize causal test pairs
    test_pairs = defaultdict(dict)
    for stim in causal_stimuli:
        token = stim["token_label"]; role = stim["role_label"]
        if token not in test_pairs or role not in test_pairs[token]:
            test_pairs[token][role] = stim
    dual_test = [(t, sorted(rs.keys())) for t, rs in test_pairs.items() if len(rs) >= 2]

    # Sample layers (focus on middle layers ±3)
    mid = nl // 2
    sample_layers = sorted(set(
        [max(1, mid - 5), max(1, mid - 3), mid - 1, mid, mid + 1, mid + 3, min(nl - 2, mid + 5)]
    ) & set(range(1, nl)))
    log(f"Sample layers: {sample_layers}")

    # =====================================================================
    # PART A: Direction-Norm Dual-Channel Decomposition
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART A: Direction-Norm Dual-Channel Test")
    log(f"{'='*60}")

    results_a = {}

    for li in sample_layers:
        log(f"\n--- Layer {li} ---")

        # Compute cell means
        cell_means = {}
        for (token, role, pair), entries in obs.items():
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is not None and pos < h.shape[1]:
                    vecs.append(h[0, pos, :].numpy().copy())
            if vecs:
                cell_means[(token, role, pair)] = np.mean(vecs, axis=0)

        # Grand mean per (token, role) across pairs
        token_role_means = {}
        for token in dual_tokens:
            for role in sorted(token_roles[token]):
                means = [cell_means.get((token, role, p)) for p in token_pairs[token]]
                means = [m for m in means if m is not None]
                if means:
                    token_role_means[(token, role)] = np.mean(means, axis=0)

        # Compute per-token role deltas with direction and norm
        token_delta_info = {}
        for token in dual_tokens:
            roles_list = sorted(token_roles[token])
            if len(roles_list) != 2: continue
            r1, r2 = roles_list  # r1=adj/noun, r2=verb/noun
            m1 = token_role_means.get((token, r1))
            m2 = token_role_means.get((token, r2))
            if m1 is None or m2 is None: continue

            delta = m2 - m1  # role1 -> role2
            norm_delta = np.linalg.norm(delta)
            direction = delta / max(norm_delta, 1e-10)

            token_delta_info[token] = {
                "delta": delta,
                "norm": float(norm_delta),
                "direction": direction,
                "r1": r1, "r2": r2,
            }

        # Compute shared direction (normalized mean of unit vectors)
        all_directions = [info["direction"] for info in token_delta_info.values()]
        if not all_directions:
            continue

        # Shared direction = normalized mean of unit direction vectors
        mean_dir = np.mean(all_directions, axis=0)
        mean_dir_norm = np.linalg.norm(mean_dir)
        shared_direction = mean_dir / max(mean_dir_norm, 1e-10)

        # Per role-pair shared directions
        rp_directions = {}
        for rp in ["adj_verb", "adj_noun", "noun_verb"]:
            rp_dirs = [token_delta_info[t]["direction"] for t in token_delta_info
                      if token_rp.get(t) == rp]
            if rp_dirs:
                rp_mean = np.mean(rp_dirs, axis=0)
                rp_norm = np.linalg.norm(rp_mean)
                rp_directions[rp] = rp_mean / max(rp_norm, 1e-10)

        # Average norm per model (for scaling)
        avg_norm = np.mean([info["norm"] for info in token_delta_info.values()])

        # ---- Causal test: 4 conditions ----
        layer_results = {}

        for token, roles_list in dual_test:
            if len(roles_list) != 2: continue
            r1, r2 = roles_list
            s1 = test_pairs[token][r1]; s2 = test_pairs[token][r2]

            idx1 = s1.get("_idx"); pos1 = s1.get("target_pos")
            idx2 = s2.get("_idx"); pos2 = s2.get("target_pos")
            if idx1 is None or idx2 is None: continue

            h1 = captures[idx1]["hidden"].get(li)
            h2 = captures[idx2]["hidden"].get(li)
            if h1 is None or h2 is None: continue
            if pos1 >= h1.shape[1] or pos2 >= h2.shape[1]: continue

            logits1 = captures[idx1]["logits"][0, -1, :].numpy().copy()
            logits2 = captures[idx2]["logits"][0, -1, :].numpy().copy()
            target_shift = logits2 - logits1  # direction we want to push toward

            delta_info = token_delta_info.get(token)
            if delta_info is None: continue

            # The actual delta for this token
            actual_delta = delta_info["delta"]
            actual_norm = delta_info["norm"]
            actual_dir = delta_info["direction"]

            # Shared direction (from all tokens except this one - LOO style)
            other_dirs = [token_delta_info[t]["direction"] for t in token_delta_info if t != token]
            if not other_dirs:
                continue
            loo_mean_dir = np.mean(other_dirs, axis=0)
            loo_dir = loo_mean_dir / max(np.linalg.norm(loo_mean_dir), 1e-10)

            # Per-role-pair LOO direction
            rp = token_rp.get(token, "")
            rp_loo_dir = loo_dir  # fallback
            if rp in rp_directions:
                rp_other_dirs = [token_delta_info[t]["direction"] for t in token_delta_info
                                if t != token and token_rp.get(t) == rp]
                if rp_other_dirs:
                    rp_loo_mean = np.mean(rp_other_dirs, axis=0)
                    rp_loo_dir = rp_loo_mean / max(np.linalg.norm(rp_loo_mean), 1e-10)

            # Random direction (for specificity)
            rng = np.random.RandomState(42 + hash(token) % 1000)
            rand_dir = rng.randn(d_model)
            rand_dir = rand_dir / np.linalg.norm(rand_dir)

            # === Condition 1: correct_dir + correct_norm (full) ===
            patch_full = loo_dir * actual_norm

            # === Condition 2: correct_dir + avg_norm (direction only, norm=average) ===
            patch_dir_only = loo_dir * avg_norm

            # === Condition 3: random_dir + correct_norm (norm gate test) ===
            patch_norm_only = rand_dir * actual_norm

            # === Condition 4: correct_dir + unit_norm (direction, minimal norm) ===
            patch_dir_unit = loo_dir * 1.0

            # === Condition 5: rp_specific_dir + correct_norm ===
            patch_rp_full = rp_loo_dir * actual_norm

            # === Condition 6: rp_specific_dir + avg_norm ===
            patch_rp_dir = rp_loo_dir * avg_norm

            # Run all 6 conditions
            conditions = {
                "full_correct": patch_full,
                "dir_only_avg_norm": patch_dir_only,
                "norm_gate_rand_dir": patch_norm_only,
                "dir_unit_norm": patch_dir_unit,
                "rp_full": patch_rp_full,
                "rp_dir_avg_norm": patch_rp_dir,
            }

            key = f"{token}_{r1}->{r2}"
            layer_results[key] = {"token": token, "r1": r1, "r2": r2, "role_pair": rp,
                                  "actual_norm": float(actual_norm)}

            for cond_name, patch_vec in conditions.items():
                patched_logits = run_with_patched_hidden(model, tok, s1["sentence"],
                                                          li, pos1, patch_vec)
                if patched_logits is not None:
                    p_logits = patched_logits[0, -1, :].numpy().copy()
                    cos_shift = cosine_sim(p_logits - logits1, target_shift)
                    layer_results[key][f"{cond_name}_cos_shift"] = float(cos_shift)
                else:
                    layer_results[key][f"{cond_name}_cos_shift"] = None

            # Random control (5 random directions)
            rand_shifts = []
            for ri in range(5):
                rng2 = np.random.RandomState(ri * 100 + hash(token) % 100)
                rdir = rng2.randn(d_model); rdir = rdir / np.linalg.norm(rdir)
                rpatch = rdir * actual_norm
                plogits = run_with_patched_hidden(model, tok, s1["sentence"], li, pos1, rpatch)
                if plogits is not None:
                    pl = plogits[0, -1, :].numpy().copy()
                    rand_shifts.append(cosine_sim(pl - logits1, target_shift))
            if rand_shifts:
                layer_results[key]["avg_random_shift"] = float(np.mean(rand_shifts))
            else:
                layer_results[key]["avg_random_shift"] = 0.0

            if len(layer_results) % 4 == 0:
                log(f"  {len(layer_results)} conditions done, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
                gc.collect(); torch.cuda.empty_cache()

        results_a[str(li)] = layer_results
        log(f"  Layer {li}: {len(layer_results)} test pairs")

    # =====================================================================
    # PART B: Role-Frame Bundle Causal Test
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART B: Role-Frame Bundle Causal Test")
    log(f"{'='*60}")

    results_b = {}

    for li in sample_layers:
        log(f"\n--- Layer {li} ---")

        # Compute cell means
        cell_means = {}
        for (token, role, pair), entries in obs.items():
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is not None and pos < h.shape[1]:
                    vecs.append(h[0, pos, :].numpy().copy())
            if vecs:
                cell_means[(token, role, pair)] = np.mean(vecs, axis=0)

        # Compute decomposition: h ≈ μ + I + R + F + ε
        # μ = grand mean
        # I(token) = token mean - μ
        # R(role) = role mean - I(token) - μ  (pair-averaged role effect)
        # F(pair) = pair mean - I(token) - μ  (role-averaged pair/frame effect)
        # ε = residual

        # Grand mean
        all_vecs = list(cell_means.values())
        grand_mean = np.mean(all_vecs, axis=0)

        # Per-token means
        token_means = {}
        for token in dual_tokens:
            vecs = [cell_means[(token, role, pair)]
                   for role in token_roles[token] for pair in token_pairs[token]
                   if (token, role, pair) in cell_means]
            if vecs:
                token_means[token] = np.mean(vecs, axis=0)

        # Per-role means (across tokens and pairs)
        all_roles = set()
        for stim in sub_stimuli:
            all_roles.add(stim["role_label"])
        role_means = {}
        for role in all_roles:
            vecs = [cell_means[(token, role, pair)]
                   for token in dual_tokens for pair in token_pairs[token]
                   if (token, role, pair) in cell_means]
            if vecs:
                role_means[role] = np.mean(vecs, axis=0)

        # Per-pair means (across tokens and roles)
        all_pairs = set()
        for stim in sub_stimuli:
            all_pairs.add(stim["pair_label"])
        pair_means = {}
        for pair in all_pairs:
            vecs = [cell_means[(token, role, pair)]
                   for token in dual_tokens for role in token_roles[token]
                   if (token, role, pair) in cell_means]
            if vecs:
                pair_means[pair] = np.mean(vecs, axis=0)

        # Extract per-token role increment R and frame increment F
        # R(token) = avg(h[token, role2, :]) - avg(h[token, role1, :]) pair-averaged
        # F(token, pair) = avg(h[token, :, pair]) - avg(h[token, :, :]) role-averaged

        token_R = {}  # per-token role increment
        token_F = {}  # per-token, per-pair frame deviation

        for token in dual_tokens:
            roles_list = sorted(token_roles[token])
            if len(roles_list) != 2: continue
            r1, r2 = roles_list

            # Role increment: pair-averaged
            r1_pairs = [cell_means.get((token, r1, p)) for p in token_pairs[token]]
            r2_pairs = [cell_means.get((token, r2, p)) for p in token_pairs[token]]
            r1_pairs = [m for m in r1_pairs if m is not None]
            r2_pairs = [m for m in r2_pairs if m is not None]
            if r1_pairs and r2_pairs:
                token_R[token] = np.mean(r2_pairs, axis=0) - np.mean(r1_pairs, axis=0)

            # Frame deviations: role-averaged per pair
            for pair in token_pairs[token]:
                r_vecs = [cell_means.get((token, r, pair)) for r in roles_list]
                r_vecs = [m for m in r_vecs if m is not None]
                if r_vecs:
                    pair_mean = np.mean(r_vecs, axis=0)
                    token_R_mean = token_means.get(token)
                    if token_R_mean is not None:
                        token_F[(token, pair)] = pair_mean - token_R_mean

        # Causal test: replace R only, F only, R+F, interaction
        layer_results_b = {}

        for token, roles_list in dual_test:
            if len(roles_list) != 2: continue
            r1, r2 = roles_list
            s1 = test_pairs[token][r1]; s2 = test_pairs[token][r2]

            idx1 = s1.get("_idx"); pos1 = s1.get("target_pos")
            idx2 = s2.get("_idx"); pos2 = s2.get("target_pos")
            if idx1 is None or idx2 is None: continue

            h1 = captures[idx1]["hidden"].get(li)
            h2 = captures[idx2]["hidden"].get(li)
            if h1 is None or h2 is None: continue
            if pos1 >= h1.shape[1] or pos2 >= h2.shape[1]: continue

            logits1 = captures[idx1]["logits"][0, -1, :].numpy().copy()
            logits2 = captures[idx2]["logits"][0, -1, :].numpy().copy()
            target_shift = logits2 - logits1

            R_this = token_R.get(token)
            if R_this is None: continue

            # Full delta: h2_avg - h1_avg at this position
            v1 = captures[idx1]["hidden"][li][0, pos1, :].numpy().copy()
            v2 = captures[idx2]["hidden"][li][0, pos2, :].numpy().copy()
            full_delta = v2 - v1

            # Frame delta: use source sentence's frame deviation
            # We don't have pair info for causal sentences, so estimate F as full_delta - R
            F_estimated = full_delta - R_this

            # === Condition 1: role only (patch R) ===
            patch_role = R_this

            # === Condition 2: frame only (patch F_estimated) ===
            patch_frame = F_estimated

            # === Condition 3: role + frame (patch R + F = full_delta) ===
            patch_role_frame = full_delta

            # === Condition 4: interaction only (full_delta - R - F = 0 by construction, skip) ===
            # Since F = full_delta - R, interaction = 0. Not meaningful.
            # Instead, use LOO role direction

            # === Condition 5: LOO role direction only ===
            other_R = {t: v for t, v in token_R.items() if t != token}
            if other_R:
                loo_R = np.mean(list(other_R.values()), axis=0)
            else:
                loo_R = R_this

            # === Condition 6: LOO role + estimated frame ===
            patch_loo_role_frame = loo_R + F_estimated

            conditions_b = {
                "role_only": patch_role,
                "frame_only": patch_frame,
                "role_plus_frame": patch_role_frame,
                "loo_role_only": loo_R,
                "loo_role_plus_frame": patch_loo_role_frame,
            }

            key = f"{token}_{r1}->{r2}"
            layer_results_b[key] = {"token": token, "r1": r1, "r2": r2,
                                     "R_norm": float(np.linalg.norm(R_this)),
                                     "F_norm": float(np.linalg.norm(F_estimated)),
                                     "full_norm": float(np.linalg.norm(full_delta))}

            for cond_name, patch_vec in conditions_b.items():
                patched_logits = run_with_patched_hidden(model, tok, s1["sentence"],
                                                          li, pos1, patch_vec)
                if patched_logits is not None:
                    p_logits = patched_logits[0, -1, :].numpy().copy()
                    cos_shift = cosine_sim(p_logits - logits1, target_shift)
                    layer_results_b[key][f"{cond_name}_cos_shift"] = float(cos_shift)
                else:
                    layer_results_b[key][f"{cond_name}_cos_shift"] = None

            # Random control
            rand_shifts = []
            for ri in range(5):
                rng2 = np.random.RandomState(ri * 100 + hash(token) % 100)
                rdir = rng2.randn(d_model); rdir = rdir / np.linalg.norm(rdir)
                rpatch = rdir * np.linalg.norm(full_delta)
                plogits = run_with_patched_hidden(model, tok, s1["sentence"], li, pos1, rpatch)
                if plogits is not None:
                    pl = plogits[0, -1, :].numpy().copy()
                    rand_shifts.append(cosine_sim(pl - logits1, target_shift))
            if rand_shifts:
                layer_results_b[key]["avg_random_shift"] = float(np.mean(rand_shifts))
            else:
                layer_results_b[key]["avg_random_shift"] = 0.0

            if len(layer_results_b) % 4 == 0:
                log(f"  {len(layer_results_b)} conditions done, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
                gc.collect(); torch.cuda.empty_cache()

        results_b[str(li)] = layer_results_b
        log(f"  Layer {li}: {len(layer_results_b)} test pairs")

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    output = {
        "model": model_name,
        "n_layers": nl,
        "d_model": d_model,
        "sample_layers": sample_layers,
        "part_a_dir_norm": results_a,
        "part_b_role_frame_bundle": results_b,
    }

    out_path = RESULT_DIR / f"{model_name}_dir_norm_bundle.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"\nSaved to {out_path}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"SUMMARY for {model_name}")
    log(f"{'='*60}")

    # Part A summary
    log(f"\n--- Part A: Direction-Norm Dual-Channel ---")
    for li_str, layer_res in results_a.items():
        if not layer_res: continue
        for cond in ["full_correct", "dir_only_avg_norm", "norm_gate_rand_dir",
                      "dir_unit_norm", "rp_full", "rp_dir_avg_norm"]:
            shifts = [v.get(f"{cond}_cos_shift") for v in layer_res.values()
                     if v.get(f"{cond}_cos_shift") is not None]
            rand_shifts = [v.get("avg_random_shift", 0) for v in layer_res.values()]
            if shifts:
                avg_shift = np.mean(shifts)
                pos_rate = sum(1 for s in shifts if s > 0) / len(shifts)
                avg_rand = np.mean(rand_shifts) if rand_shifts else 0
                specificity = avg_shift / max(abs(avg_rand), 1e-6)
                log(f"  L{li_str} {cond}: avg_shift={avg_shift:+.4f} pos_rate={pos_rate:.0%} spec={specificity:.1f}x")

    # Part B summary
    log(f"\n--- Part B: Role-Frame Bundle ---")
    for li_str, layer_res in results_b.items():
        if not layer_res: continue
        for cond in ["role_only", "frame_only", "role_plus_frame", "loo_role_only", "loo_role_plus_frame"]:
            shifts = [v.get(f"{cond}_cos_shift") for v in layer_res.values()
                     if v.get(f"{cond}_cos_shift") is not None]
            rand_shifts = [v.get("avg_random_shift", 0) for v in layer_res.values()]
            if shifts:
                avg_shift = np.mean(shifts)
                pos_rate = sum(1 for s in shifts if s > 0) / len(shifts)
                avg_rand = np.mean(rand_shifts) if rand_shifts else 0
                specificity = avg_shift / max(abs(avg_rand), 1e-6)
                log(f"  L{li_str} {cond}: avg_shift={avg_shift:+.4f} pos_rate={pos_rate:.0%} spec={specificity:.1f}x")

        # Key comparison: role_only vs role_plus_frame
        role_shifts = [v.get("role_only_cos_shift") for v in layer_res.values()
                      if v.get("role_only_cos_shift") is not None]
        bundle_shifts = [v.get("role_plus_frame_cos_shift") for v in layer_res.values()
                        if v.get("role_plus_frame_cos_shift") is not None]
        if role_shifts and bundle_shifts:
            avg_role = np.mean(role_shifts); avg_bundle = np.mean(bundle_shifts)
            log(f"  L{li_str} BUNDLE EFFECT: role_only={avg_role:+.4f} role+frame={avg_bundle:+.4f} ratio={avg_bundle/max(abs(avg_role),1e-6):.2f}x")

    release_model(model)
    log(f"Phase 299 complete for {model_name}")

if __name__ == "__main__":
    main()
