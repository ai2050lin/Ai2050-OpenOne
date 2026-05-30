"""
Phase 298b: Normalized Role Increment PCA & Per-Role-Pair Analysis
===================================================================
Goal: Disentangle DS7B's "fake 1D" structure by normalizing role increments,
and analyze whether different role pairs (adj-verb, adj-noun, noun-verb) 
have different subspace structures.

Key Questions:
1. Does normalizing role increments change DS7B's apparent 1D structure?
2. Do different role pairs (adj-verb vs adj-noun vs noun-verb) have 
   different subspace dimensionalities?
3. What is the per-role-pair cross-token consistency?
4. How does per-role-pair LOO compare with overall LOO?

Usage:
  python tests/glm5/phase298b_normalized_pca.py qwen3
  python tests/glm5/phase298b_normalized_pca.py glm4
  python tests/glm5/phase298b_normalized_pca.py deepseek7b
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

RESULT_DIR = Path("results/phase298_role_subspace")
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
# STIMULUS DESIGN (same as Phase 297)
# =====================================================================
def build_phase297_stimuli():
    stimuli = []
    adj_verb_tokens = {
        "open": {
            "adj": {"P1": ["the door is open", "the gate is open"], "P2": ["the door remains open", "the gate remains open"],
                    "P3": ["the open door", "the open gate"], "P4": ["the shop seemed open", "the road seemed open"]},
            "verb": {"P1": ["they open the door", "they open the gate"], "P2": ["we open the door", "we open the gate"],
                     "P3": ["the door will open", "the gate will open"], "P4": ["they began to open the shop", "they began to open the road"]},
        },
        "clear": {
            "adj": {"P1": ["the path is clear", "the road is clear"], "P2": ["the path remains clear", "the road remains clear"],
                    "P3": ["the clear path", "the clear road"], "P4": ["the desk seemed clear", "the table seemed clear"]},
            "verb": {"P1": ["they clear the path", "they clear the road"], "P2": ["we clear the path", "we clear the road"],
                     "P3": ["the path will clear", "the road will clear"], "P4": ["they began to clear the desk", "they began to clear the table"]},
        },
        "warm": {
            "adj": {"P1": ["the room is warm", "the house is warm"], "P2": ["the room remains warm", "the house remains warm"],
                    "P3": ["the warm room", "the warm house"], "P4": ["the water seemed warm", "the food seemed warm"]},
            "verb": {"P1": ["they warm the room", "they warm the house"], "P2": ["we warm the room", "we warm the house"],
                     "P3": ["the room will warm", "the house will warm"], "P4": ["they began to warm the water", "they began to warm the food"]},
        },
        "clean": {
            "adj": {"P1": ["the floor is clean", "the table is clean"], "P2": ["the floor remains clean", "the table remains clean"],
                    "P3": ["the clean floor", "the clean table"], "P4": ["the room seemed clean", "the house seemed clean"]},
            "verb": {"P1": ["they clean the floor", "they clean the table"], "P2": ["we clean the floor", "we clean the table"],
                     "P3": ["the floor will clean", "the table will clean"], "P4": ["they began to clean the room", "they began to clean the house"]},
        },
    }
    adj_noun_tokens = {
        "light": {
            "adj": {"P1": ["the bag is light", "the box is light"], "P2": ["the bag remains light", "the box remains light"],
                    "P3": ["the light bag", "the light box"], "P4": ["the load seemed light", "the dress seemed light"]},
            "noun": {"P1": ["the light is bright", "the light is warm"], "P2": ["that light is bright", "that light is warm"],
                     "P3": ["near the light", "by the light"], "P4": ["they saw the light", "they found the light"]},
        },
        "cold": {
            "adj": {"P1": ["the water is cold", "the wind is cold"], "P2": ["the water remains cold", "the wind remains cold"],
                    "P3": ["the cold water", "the cold wind"], "P4": ["the room seemed cold", "the air seemed cold"]},
            "noun": {"P1": ["the cold is severe", "the cold is bitter"], "P2": ["that cold is severe", "that cold is bitter"],
                     "P3": ["in the cold", "despite the cold"], "P4": ["they felt the cold", "they noticed the cold"]},
        },
    }
    noun_verb_tokens = {
        "fire": {
            "noun": {"P1": ["the fire is hot", "the fire is big"], "P2": ["that fire is hot", "that fire is big"],
                     "P3": ["near the fire", "by the fire"], "P4": ["they saw the fire", "they started the fire"]},
            "verb": {"P1": ["they fire the gun", "they fire the worker"], "P2": ["they will fire the gun", "they will fire the worker"],
                     "P3": ["the gun will fire", "the engine will fire"], "P4": ["they began to fire the gun", "they began to fire the worker"]},
        },
        "record": {
            "noun": {"P1": ["the record is old", "the record is broken"], "P2": ["that record is old", "that record is broken"],
                     "P3": ["on the record", "for the record"], "P4": ["they broke the record", "they set the record"]},
            "verb": {"P1": ["they record music", "they record data"], "P2": ["they will record music", "they will record data"],
                     "P3": ["the device will record", "the system will record"], "P4": ["they began to record music", "they began to record data"]},
        },
    }
    all_tokens = {}
    all_tokens.update(adj_verb_tokens)
    all_tokens.update(adj_noun_tokens)
    all_tokens.update(noun_verb_tokens)

    for token, roles in all_tokens.items():
        role_pair = "adj_verb" if token in adj_verb_tokens else (
            "adj_noun" if token in adj_noun_tokens else "noun_verb")
        for role, pairs in roles.items():
            for pair_label, sentences in pairs.items():
                for sent in sentences:
                    stimuli.append({
                        "sentence": sent, "target_word": token, "token_label": token,
                        "role_label": role, "pair_label": pair_label, "role_pair": role_pair,
                    })
    return stimuli


# =====================================================================
# MODEL LOADING
# =====================================================================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16, device_map=auto, flash)...")
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = None
    used_attn = "eager"
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=attn_impl)
            used_attn = attn_impl; break
        except Exception as e:
            log(f"  attn={attn_impl} failed: {str(e)[:80]}")
    if model is None: raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    gpu = torch.cuda.memory_allocated() / 1e9
    log(f"  Loaded with attn={used_attn}, GPU={gpu:.1f}GB")
    layers = get_layers(model)
    nl = len(layers)
    gpu_l, cpu_l = [], []
    for li in range(nl):
        wdev = layers[li].self_attn.o_proj.weight.device
        (gpu_l if wdev.type == 'cuda' else cpu_l).append(li)
    log(f"  GPU layers: {len(gpu_l)}{' (' + str(gpu_l[0]) + '-' + str(gpu_l[-1]) + ')' if gpu_l else ''}, "
        f"CPU: {len(cpu_l)}{' (' + str(cpu_l[0]) + '-' + str(cpu_l[-1]) + ')' if cpu_l else ''}")
    return model, tok


def _capture_single(model, tokenizer, sent, n_layers, max_len=64):
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = {}
    for li, h in enumerate(out.hidden_states):
        hs[li] = h.detach().cpu().float()
    return {"hidden": hs}


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
            new_stim = dict(stim)
            new_stim["target_pos"] = pos
            resolved.append(new_stim)
    return resolved


# =====================================================================
# ANALYSIS 1: Normalized Role Increment PCA
# =====================================================================
def normalized_role_pca(captures, stimuli, n_layers):
    """Same as role_increment_pca but normalize each increment to unit length."""
    obs = defaultdict(list)
    for stim in stimuli:
        token = stim["token_label"]; role = stim["role_label"]; pair = stim["pair_label"]
        idx = stim.get("_idx"); pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role, pair)].append((idx, pos))

    token_roles = defaultdict(set)
    for (token, role, pair) in obs:
        token_roles[token].add(role)
    dual_tokens = {t for t, roles in token_roles.items() if len(roles) >= 2}

    results = {}
    for li in range(n_layers + 1):
        cell_means = {}
        for (token, role, pair), entries in obs.items():
            if token not in dual_tokens: continue
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is None or pos >= h.shape[1]: continue
                vecs.append(h[0, pos, :].numpy().copy())
            if len(vecs) >= 1:
                cell_means[(token, role, pair)] = np.mean(vecs, axis=0)

        increments = []
        inc_labels = []
        for token in dual_tokens:
            roles_list = sorted(token_roles[token])
            if len(roles_list) != 2: continue
            r1, r2 = roles_list
            for p in ["P1", "P2", "P3", "P4"]:
                m1 = cell_means.get((token, r1, p))
                m2 = cell_means.get((token, r2, p))
                if m1 is not None and m2 is not None:
                    delta = m2 - m1
                    # Normalize to unit length
                    nrm = np.linalg.norm(delta)
                    if nrm > 1e-8:
                        delta_norm = delta / nrm
                        increments.append(delta_norm)
                        inc_labels.append((token, p, f"{r1}-{r2}"))

        if len(increments) < 3: continue

        inc_matrix = np.array(increments)
        n_inc, d = inc_matrix.shape
        inc_centered = inc_matrix - np.mean(inc_matrix, axis=0, keepdims=True)

        _, S, Vt = np.linalg.svd(inc_centered, full_matrices=False)
        total_var = np.sum(S ** 2)
        explained_var = (S ** 2) / max(total_var, 1e-10)
        cumulative_var = np.cumsum(explained_var)

        dim_50 = int(np.searchsorted(cumulative_var, 0.50)) + 1
        dim_80 = int(np.searchsorted(cumulative_var, 0.80)) + 1
        dim_95 = int(np.searchsorted(cumulative_var, 0.95)) + 1

        results[li] = {
            "n_increments": n_inc,
            "top1_var": float(explained_var[0]) if len(explained_var) > 0 else 0.0,
            "top3_var": float(cumulative_var[2]) if len(cumulative_var) > 2 else 1.0,
            "top5_var": float(cumulative_var[4]) if len(cumulative_var) > 4 else 1.0,
            "dim_50": dim_50, "dim_80": dim_80, "dim_95": dim_95,
            "explained_var_top10": [float(v) for v in explained_var[:10]],
        }

    return results


# =====================================================================
# ANALYSIS 2: Per-Role-Pair PCA
# =====================================================================
def per_role_pair_pca(captures, stimuli, n_layers, normalize=False):
    """Do PCA separately for each role pair type (adj_verb, adj_noun, noun_verb)."""
    obs = defaultdict(list)
    for stim in stimuli:
        token = stim["token_label"]; role = stim["role_label"]; pair = stim["pair_label"]
        rp = stim.get("role_pair", "")
        idx = stim.get("_idx"); pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role, pair)].append((idx, pos))

    # Map tokens to role pairs
    token_rp = {}
    for stim in stimuli:
        token_rp[stim["token_label"]] = stim.get("role_pair", "")

    token_roles = defaultdict(set)
    for (token, role, pair) in obs:
        token_roles[token].add(role)
    dual_tokens = {t for t, roles in token_roles.items() if len(roles) >= 2}

    results = {}
    for li in range(n_layers + 1):
        cell_means = {}
        for (token, role, pair), entries in obs.items():
            if token not in dual_tokens: continue
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is None or pos >= h.shape[1]: continue
                vecs.append(h[0, pos, :].numpy().copy())
            if len(vecs) >= 1:
                cell_means[(token, role, pair)] = np.mean(vecs, axis=0)

        layer_result = {}
        for rp_type in ["adj_verb", "adj_noun", "noun_verb"]:
            rp_tokens = [t for t in dual_tokens if token_rp.get(t) == rp_type]
            if not rp_tokens: continue

            increments = []
            for token in rp_tokens:
                roles_list = sorted(token_roles[token])
                if len(roles_list) != 2: continue
                r1, r2 = roles_list
                for p in ["P1", "P2", "P3", "P4"]:
                    m1 = cell_means.get((token, r1, p))
                    m2 = cell_means.get((token, r2, p))
                    if m1 is not None and m2 is not None:
                        delta = m2 - m1
                        if normalize:
                            nrm = np.linalg.norm(delta)
                            if nrm > 1e-8:
                                delta = delta / nrm
                            else:
                                continue
                        increments.append(delta)

            if len(increments) < 2: continue

            inc_matrix = np.array(increments)
            n_inc = inc_matrix.shape[0]
            inc_centered = inc_matrix - np.mean(inc_matrix, axis=0, keepdims=True)

            _, S, Vt = np.linalg.svd(inc_centered, full_matrices=False)
            total_var = np.sum(S ** 2)
            explained_var = (S ** 2) / max(total_var, 1e-10)
            cumulative_var = np.cumsum(explained_var)

            dim_50 = int(np.searchsorted(cumulative_var, 0.50)) + 1
            dim_80 = int(np.searchsorted(cumulative_var, 0.80)) + 1

            # Cross-token cosine within this role pair
            if len(increments) >= 2:
                cos_vals = []
                for i in range(len(increments)):
                    for j in range(i + 1, len(increments)):
                        d1, d2 = increments[i], increments[j]
                        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                        if n1 > 1e-8 and n2 > 1e-8:
                            cos_vals.append(float(np.dot(d1, d2) / (n1 * n2)))
                avg_cos = float(np.mean(cos_vals)) if cos_vals else 0.0
            else:
                avg_cos = 0.0

            layer_result[rp_type] = {
                "n_increments": n_inc,
                "n_tokens": len(rp_tokens),
                "top1_var": float(explained_var[0]) if len(explained_var) > 0 else 0.0,
                "top3_var": float(cumulative_var[2]) if len(cumulative_var) > 2 else 1.0,
                "dim_50": dim_50, "dim_80": dim_80,
                "cross_token_cos": avg_cos,
            }

        results[li] = layer_result

    return results


# =====================================================================
# ANALYSIS 3: Per-Role-Pair LOO
# =====================================================================
def per_role_pair_loo(captures, stimuli, n_layers):
    """Leave-one-token-out within each role pair type."""
    obs = defaultdict(list)
    for stim in stimuli:
        token = stim["token_label"]; role = stim["role_label"]; pair = stim["pair_label"]
        rp = stim.get("role_pair", "")
        idx = stim.get("_idx"); pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role, pair)].append((idx, pos))

    token_rp = {}
    for stim in stimuli:
        token_rp[stim["token_label"]] = stim.get("role_pair", "")

    token_roles = defaultdict(set)
    token_pairs = defaultdict(set)
    for (token, role, pair) in obs:
        token_roles[token].add(role)
        token_pairs[token].add(pair)
    dual_tokens = {t for t, roles in token_roles.items() if len(roles) >= 2}

    results = {}
    for li in range(n_layers + 1):
        cell_means = {}
        for (token, role, pair), entries in obs.items():
            if token not in dual_tokens: continue
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is None or pos >= h.shape[1]: continue
                vecs.append(h[0, pos, :].numpy().copy())
            if len(vecs) >= 1:
                cell_means[(token, role, pair)] = np.mean(vecs, axis=0)

        layer_result = {}
        for rp_type in ["adj_verb", "adj_noun", "noun_verb"]:
            rp_tokens = sorted([t for t in dual_tokens if token_rp.get(t) == rp_type])

            # Compute per-token pair-averaged deltas
            token_deltas = {}
            for token in rp_tokens:
                roles_list = sorted(token_roles[token])
                if len(roles_list) != 2: continue
                r1, r2 = roles_list
                r1_means = [cell_means.get((token, r1, p)) for p in token_pairs[token]]
                r2_means = [cell_means.get((token, r2, p)) for p in token_pairs[token]]
                r1_means = [m for m in r1_means if m is not None]
                r2_means = [m for m in r2_means if m is not None]
                if r1_means and r2_means:
                    token_deltas[token] = np.mean(r2_means, axis=0) - np.mean(r1_means, axis=0)

            if len(token_deltas) < 2: continue

            # LOO
            loo_cosines = []
            all_tokens_list = sorted(token_deltas.keys())
            for held_out in all_tokens_list:
                remaining = [t for t in all_tokens_list if t != held_out]
                avg_dir = np.mean([token_deltas[t] for t in remaining], axis=0)
                actual = token_deltas[held_out]
                n1, n2 = np.linalg.norm(avg_dir), np.linalg.norm(actual)
                if n1 > 1e-8 and n2 > 1e-8:
                    loo_cosines.append(float(np.dot(avg_dir, actual) / (n1 * n2)))

            layer_result[rp_type] = {
                "n_tokens": len(token_deltas),
                "avg_loo_cos": float(np.mean(loo_cosines)) if loo_cosines else 0.0,
                "loo_cosines": {t: c for t, c in zip(all_tokens_list, loo_cosines)},
            }

        results[li] = layer_result

    return results


# =====================================================================
# ANALYSIS 4: Increment Norm Distribution
# =====================================================================
def increment_norm_distribution(captures, stimuli, n_layers):
    """Show per-token role increment norms to understand norm dominance."""
    obs = defaultdict(list)
    for stim in stimuli:
        token = stim["token_label"]; role = stim["role_label"]; pair = stim["pair_label"]
        idx = stim.get("_idx"); pos = stim.get("target_pos")
        if idx is not None and pos is not None:
            obs[(token, role, pair)].append((idx, pos))

    token_roles = defaultdict(set)
    token_pairs = defaultdict(set)
    for (token, role, pair) in obs:
        token_roles[token].add(role)
        token_pairs[token].add(pair)
    dual_tokens = {t for t, roles in token_roles.items() if len(roles) >= 2}

    results = {}
    for li in range(n_layers + 1):
        cell_means = {}
        for (token, role, pair), entries in obs.items():
            if token not in dual_tokens: continue
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is None or pos >= h.shape[1]: continue
                vecs.append(h[0, pos, :].numpy().copy())
            if len(vecs) >= 1:
                cell_means[(token, role, pair)] = np.mean(vecs, axis=0)

        layer_result = {}
        for token in sorted(dual_tokens):
            roles_list = sorted(token_roles[token])
            if len(roles_list) != 2: continue
            r1, r2 = roles_list
            r1_means = [cell_means.get((token, r1, p)) for p in token_pairs[token]]
            r2_means = [cell_means.get((token, r2, p)) for p in token_pairs[token]]
            r1_means = [m for m in r1_means if m is not None]
            r2_means = [m for m in r2_means if m is not None]
            if r1_means and r2_means:
                delta = np.mean(r2_means, axis=0) - np.mean(r1_means, axis=0)
                layer_result[token] = {
                    "norm": float(np.linalg.norm(delta)),
                    "role_pair": f"{r1}-{r2}",
                }

        results[li] = layer_result

    return results


# =====================================================================
# MAIN
# =====================================================================
def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase298b_{model_name}.txt"
    _log_file = str(log_file)

    log(f"Phase 298b: Normalized & Per-Role-Pair PCA -- {model_name}")
    log(f"=" * 60)

    # ---- 1. Load model ----
    model, tok = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    nl = info.n_layers
    log(f"  n_layers={nl}, d_model={info.d_model}")

    # ---- 2. Build stimuli ----
    stimuli = build_phase297_stimuli()
    stimuli = resolve_positions(stimuli, tok)
    log(f"  Valid stimuli: {len(stimuli)}")

    all_sentences = []
    sent_to_idx = {}
    for s in stimuli:
        sent = s["sentence"]
        if sent not in sent_to_idx:
            sent_to_idx[sent] = len(all_sentences)
            all_sentences.append(sent)
        s["_idx"] = sent_to_idx[sent]

    # ---- 3. Capture hidden states ----
    log(f"\n--- Capturing {len(all_sentences)} sentences ---")
    t0 = time.time()
    captures = {}
    for i, sent in enumerate(all_sentences):
        captures[i] = _capture_single(model, tok, sent, nl)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1)
            eta = (len(all_sentences) - i - 1) / rate
            log(f"  {i+1}/{len(all_sentences)} ({rate:.1f}/s) ETA={eta:.0f}s")
            gc.collect(); torch.cuda.empty_cache()
    log(f"  Done in {time.time()-t0:.0f}s")

    # ---- 4. Normalized Role PCA ----
    log(f"\n--- Analysis 1: Normalized Role Increment PCA ---")
    norm_pca = normalized_role_pca(captures, stimuli, nl)

    sample_layers = sorted(set([0, nl//4, nl//2, 3*nl//4, nl-1, nl]) & set(norm_pca.keys()))
    log("  Layer | top1% | top3% | top5% | dim50 | dim80 | dim95")
    for li in sorted(norm_pca.keys()):
        if li in sample_layers or li >= nl - 2:
            r = norm_pca[li]
            log(f"  L{li:3d} | {r['top1_var']*100:5.1f} | {r['top3_var']*100:5.1f} | "
                f"{r['top5_var']*100:5.1f} | {r['dim_50']:5d} | {r['dim_80']:5d} | {r['dim_95']:5d}")

    # ---- 5. Per-Role-Pair PCA (raw) ----
    log(f"\n--- Analysis 2: Per-Role-Pair PCA (raw) ---")
    prp_raw = per_role_pair_pca(captures, stimuli, nl, normalize=False)

    for li in sorted(prp_raw.keys()):
        if li in sample_layers or li >= nl - 2:
            log(f"  L{li}:")
            for rp, r in prp_raw[li].items():
                log(f"    {rp}: n_tok={r['n_tokens']} n_inc={r['n_increments']} "
                    f"top1={r['top1_var']*100:.1f}% top3={r['top3_var']*100:.1f}% "
                    f"dim50={r['dim_50']} cross_tok_cos={r['cross_token_cos']:+.4f}")

    # ---- 6. Per-Role-Pair PCA (normalized) ----
    log(f"\n--- Analysis 3: Per-Role-Pair PCA (normalized) ---")
    prp_norm = per_role_pair_pca(captures, stimuli, nl, normalize=True)

    for li in sorted(prp_norm.keys()):
        if li in sample_layers or li >= nl - 2:
            log(f"  L{li}:")
            for rp, r in prp_norm[li].items():
                log(f"    {rp}: n_tok={r['n_tokens']} n_inc={r['n_increments']} "
                    f"top1={r['top1_var']*100:.1f}% top3={r['top3_var']*100:.1f}% "
                    f"dim50={r['dim_50']} cross_tok_cos={r['cross_token_cos']:+.4f}")

    # ---- 7. Per-Role-Pair LOO ----
    log(f"\n--- Analysis 4: Per-Role-Pair LOO ---")
    prp_loo = per_role_pair_loo(captures, stimuli, nl)

    for li in sorted(prp_loo.keys()):
        if li in sample_layers or li >= nl - 2:
            log(f"  L{li}:")
            for rp, r in prp_loo[li].items():
                per_tok = " ".join(f"{t}={c:+.3f}" for t, c in r["loo_cosines"].items())
                log(f"    {rp}: avg_LOO={r['avg_loo_cos']:+.4f} | {per_tok}")

    # ---- 8. Increment Norm Distribution ----
    log(f"\n--- Analysis 5: Increment Norm Distribution ---")
    norm_dist = increment_norm_distribution(captures, stimuli, nl)

    for li in sorted(norm_dist.keys()):
        if li in sample_layers or li >= nl - 2:
            norms = [(t, norm_dist[li][t]["norm"], norm_dist[li][t]["role_pair"])
                     for t in sorted(norm_dist[li].keys())]
            if norms:
                max_norm = max(n for _, n, _ in norms)
                min_norm = min(n for _, n, _ in norms)
                log(f"  L{li}: min={min_norm:.1f} max={max_norm:.1f} ratio={max_norm/max(min_norm,1e-8):.1f}x")
                for t, n, rp in norms:
                    bar = "#" * int(n / max(max_norm, 1) * 30)
                    log(f"    {t:8s} ({rp:10s}): {n:8.1f} {bar}")

    # ---- 9. Save ----
    log(f"\n--- Saving results ---")

    def convert_keys(d):
        if isinstance(d, defaultdict): d = dict(d)
        if isinstance(d, dict): return {str(k): convert_keys(v) for k, v in d.items()}
        if isinstance(d, np.ndarray): return d.tolist()
        return d

    output = {
        "model": model_name,
        "n_layers": nl,
        "normalized_role_pca": convert_keys(norm_pca),
        "per_role_pair_pca_raw": convert_keys(prp_raw),
        "per_role_pair_pca_normalized": convert_keys(prp_norm),
        "per_role_pair_loo": convert_keys(prp_loo),
        "increment_norm_distribution": convert_keys(norm_dist),
        "timestamp": datetime.now().isoformat(),
    }

    out_path = RESULT_DIR / f"{model_name}_normalized_pca.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"  Saved to {out_path}")

    # ---- 10. Summary ----
    log(f"\n{'='*60}")
    log(f"PHASE 298b SUMMARY -- {model_name}")
    log(f"{'='*60}")

    mid = nl // 2
    mid_str = str(mid)

    log(f"\n  [A] Raw vs Normalized Role PCA at mid-layer (L{mid}):")
    if mid_str in norm_pca:
        r = norm_pca[mid_str]
        log(f"    Normalized: top1={r['top1_var']*100:.1f}% dim50={r['dim_50']} dim80={r['dim_80']}")
    # Compare with raw (from phase298 data)
    raw_path = RESULT_DIR / f"{model_name}_role_subspace.json"
    if raw_path.exists():
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        raw_rpca = raw_data.get("role_increment_pca", {}).get(mid_str, {})
        if raw_rpca:
            log(f"    Raw:        top1={raw_rpca['top1_var']*100:.1f}% dim50={raw_rpca['dim_50']} dim80={raw_rpca['dim_80']}")

    log(f"\n  [B] Per-Role-Pair at mid-layer (L{mid}):")
    mid_str = str(mid)
    if mid_str in prp_raw:
        for rp, r in prp_raw[mid_str].items():
            r_norm = prp_norm.get(mid_str, {}).get(rp, {})
            norm_top1 = r_norm.get("top1_var", 0) * 100 if r_norm else 0
            norm_dim50 = r_norm.get("dim_50", 0) if r_norm else 0
            log(f"    {rp}: raw top1={r['top1_var']*100:.1f}% dim50={r['dim_50']} | "
                f"norm top1={norm_top1:.1f}% dim50={norm_dim50} | "
                f"cross_tok_cos={r['cross_token_cos']:+.4f}")

    log(f"\n  [C] Per-Role-Pair LOO at mid-layer (L{mid}):")
    if mid_str in prp_loo:
        for rp, r in prp_loo[mid_str].items():
            per_tok = " ".join(f"{t}={c:+.3f}" for t, c in r["loo_cosines"].items())
            log(f"    {rp}: avg={r['avg_loo_cos']:+.4f} | {per_tok}")

    log(f"\n  [D] Norm Ratio at mid-layer (L{mid}):")
    if mid_str in norm_dist:
        norms = [norm_dist[mid_str][t]["norm"] for t in norm_dist[mid_str]]
        if norms:
            log(f"    min={min(norms):.1f} max={max(norms):.1f} "
                f"max/min={max(norms)/max(min(norms),1e-8):.1f}x "
                f"std/mean={np.std(norms)/max(np.mean(norms),1e-8):.2f}")

    # Release
    release_model(model)
    model = None
    gc.collect(); torch.cuda.empty_cache()
    log(f"  Model released. GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    log(f"Phase 298b complete for {model_name}!")


if __name__ == "__main__":
    main()
