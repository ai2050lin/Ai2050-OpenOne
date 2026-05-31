"""
Phase 308: Scope Causal Testing + O-C Orthogonality + Cross-Form Negation
=========================================================================
Three critical gaps to fill:

A. Scope causal test: Does scope change produce independent causal effects?
   - Minimal pairs where "not" applies to different syntactic scopes
   - O_shared, S_scope, O×S decomposition
   - Causal patching of each component

B. O-C orthogonality: Is operator direction orthogonal to construction direction?
   - cos(O_not, C_construction) direct measurement
   - O_clean = O - Proj_C(O), test if still causal

C. Cross-form negation: Do not/no/never/n't share the same subspace?
   - cos(O_not, O_no), cos(O_not, O_never), cos(O_not, O_n't)
   - Cross-form causal injection test

Stimulus design:
- Scope pairs: 5 types × 6 examples each = 30 pairs
- Construction sentences for C: 16 dual-role tokens × 3 frames = 96
- Cross-form negation: 15 adj × 4 neg-forms = 60 sentences

Usage:
  python tests/glm5/phase308_scope_causal.py qwen3
  python tests/glm5/phase308_scope_causal.py glm4
  python tests/glm5/phase308_scope_causal.py deepseek7b
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

RESULT_DIR = Path("results/phase308_scope_causal")
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
# SCOPE MINIMAL PAIRS
# =====================================================================
def build_scope_pairs():
    """
    Build scope minimal pairs where negation scope changes meaning.
    
    Each pair has:
    - baseline: affirmative sentence
    - narrow_scope: negation applies to part of the meaning
    - wide_scope: negation applies to the whole meaning
    
    Types:
    1. Quantifier scope: "not all X Y" vs "all X did not Y"
    2. Adverb scope: "not very ADJ" vs "very not ADJ"  
    3. Embedding scope: "I did not say he X" vs "I said he did not X"
    4. Infinitive scope: "not possible to X" vs "possible not to X"
    5. Modifier scope: "not only ADJ but also ADJ2" vs "only not ADJ but ADJ2"
    """
    pairs = []
    
    # Type 1: Quantifier scope
    quant_pairs = [
        ("all students passed", "not all students passed", "all students did not pass", "students"),
        ("all workers agreed", "not all workers agreed", "all workers did not agree", "workers"),
        ("all members arrived", "not all members arrived", "all members did not arrive", "members"),
        ("every student passed", "not every student passed", "every student did not pass", "student"),
        ("every worker agreed", "not every worker agreed", "every worker did not agree", "worker"),
        ("every member arrived", "not every member arrived", "every member did not arrive", "member"),
    ]
    for base, narrow, wide, target in quant_pairs:
        pairs.append({
            "type": "quantifier_scope",
            "baseline": base, "narrow": narrow, "wide": wide,
            "target_word": target,
            "narrow_scope": "partial_negation",
            "wide_scope": "total_negation",
        })
    
    # Type 2: Adverb scope  
    adv_pairs = [
        ("the result was very happy", "the result was not very happy", "the result was very not happy", "happy"),
        ("the weather was very warm", "the weather was not very warm", "the weather was very not warm", "warm"),
        ("the task was very easy", "the task was not very easy", "the task was very not easy", "easy"),
        ("the light was very bright", "the light was not very bright", "the light was very not bright", "bright"),
        ("the food was very good", "the food was not very good", "the food was very not good", "good"),
        ("the plan was very smart", "the plan was not very smart", "the plan was very not smart", "smart"),
    ]
    for base, narrow, wide, target in adv_pairs:
        pairs.append({
            "type": "adverb_scope",
            "baseline": base, "narrow": narrow, "wide": wide,
            "target_word": target,
            "narrow_scope": "negate_modifier",
            "wide_scope": "negate_adjective",
        })
    
    # Type 3: Embedding scope
    embed_pairs = [
        ("I said he lied", "I did not say he lied", "I said he did not lie", "lied"),
        ("I think he knows", "I do not think he knows", "I think he does not know", "knows"),
        ("I believe he tried", "I do not believe he tried", "I believe he did not try", "tried"),
        ("I claim he helped", "I do not claim he helped", "I claim he did not help", "helped"),
        ("I feel he wanted", "I do not feel he wanted", "I feel he did not want", "wanted"),
        ("I guess he liked", "I do not guess he liked", "I guess he did not like", "liked"),
    ]
    for base, narrow, wide, target in embed_pairs:
        pairs.append({
            "type": "embedding_scope",
            "baseline": base, "narrow": narrow, "wide": wide,
            "target_word": target,
            "narrow_scope": "negate_matrix",
            "wide_scope": "negate_embedded",
        })
    
    # Type 4: Infinitive scope
    inf_pairs = [
        ("it was possible to leave", "it was not possible to leave", "it was possible not to leave", "leave"),
        ("it was necessary to wait", "it was not necessary to wait", "it was necessary not to wait", "wait"),
        ("it was easy to find", "it was not easy to find", "it was easy not to find", "find"),
        ("it was safe to open", "it was not safe to open", "it was safe not to open", "open"),
        ("it was hard to solve", "it was not hard to solve", "it was hard not to solve", "solve"),
        ("it was allowed to enter", "it was not allowed to enter", "it was allowed not to enter", "enter"),
    ]
    for base, narrow, wide, target in inf_pairs:
        pairs.append({
            "type": "infinitive_scope",
            "baseline": base, "narrow": narrow, "wide": wide,
            "target_word": target,
            "narrow_scope": "negate_possibility",
            "wide_scope": "negate_action",
        })
    
    # Type 5: Double negation / affirmation via negation
    dbl_pairs = [
        ("the result was good", "the result was not bad", "the result was not good", "good"),
        ("the person was happy", "the person was not sad", "the person was not happy", "happy"),
        ("the plan was smart", "the plan was not stupid", "the plan was not smart", "smart"),
        ("the task was easy", "the task was not hard", "the task was not easy", "easy"),
        ("the weather was warm", "the weather was not cold", "the weather was not warm", "warm"),
        ("the water was clean", "the water was not dirty", "the water was not clean", "clean"),
    ]
    for base, narrow, wide, target in dbl_pairs:
        pairs.append({
            "type": "double_negation",
            "baseline": base, "narrow": narrow, "wide": wide,
            "target_word": target,
            "narrow_scope": "affirm_via_antonym_neg",
            "wide_scope": "direct_negation",
        })
    
    return pairs


# =====================================================================
# CONSTRUCTION SENTENCES (for C direction)
# =====================================================================
def build_construction_sentences():
    """Different frames for same token to extract C(construction) direction."""
    DUAL_TOKENS = {
        "open":   ("the door is open", "they open the door"),
        "clear":  ("the path is clear", "they clear the path"),
        "warm":   ("the room is warm", "they warm the room"),
        "clean":  ("the floor is clean", "they clean the floor"),
        "dry":    ("the cloth is dry", "they dry the cloth"),
        "close":  ("the store is close", "they close the store"),
        "free":   ("the bird is free", "they free the bird"),
        "quiet":  ("the room is quiet", "they quiet the room"),
        "cool":   ("the water is cool", "they cool the water"),
        "smooth": ("the surface is smooth", "they smooth the surface"),
        "cold":   ("the water is cold", "a cold hit them"),
        "light":  ("the bag is light", "a light shone through"),
        "fire":   ("the hot fire was", "they will fire the worker"),
        "record": ("the old record was", "they will record the data"),
        "run":    ("the long run was", "they will run the program"),
        "book":   ("the new book was", "they will book the room"),
    }
    
    # Add a third frame variant for each token
    THIRD_FRAMES = {
        "open":   "the open door was",
        "clear":  "the clear path was",
        "warm":   "the warm room was",
        "clean":  "the clean floor was",
        "dry":    "the dry cloth was",
        "close":  "the close store was",
        "free":   "the free bird was",
        "quiet":  "the quiet room was",
        "cool":   "the cool water was",
        "smooth": "the smooth surface was",
        "cold":   "a cold wind blew",
        "light":  "the light switch was",
        "fire":   "they will fire again",
        "record": "the new record was",
        "run":    "a fast run was",
        "book":   "the good book was",
    }
    
    stimuli = []
    for token, (adj_sent, verb_sent) in DUAL_TOKENS.items():
        stimuli.append({"sentence": adj_sent, "token": token, "role": "adj",
                        "frame": "predicative", "condition": "construction"})
        stimuli.append({"sentence": verb_sent, "token": token, "role": "verb",
                        "frame": "transitive", "condition": "construction"})
        if token in THIRD_FRAMES:
            stimuli.append({"sentence": THIRD_FRAMES[token], "token": token, "role": "attributive",
                            "frame": "attributive", "condition": "construction"})
    return stimuli


# =====================================================================
# CROSS-FORM NEGATION SENTENCES
# =====================================================================
def build_crossform_sentences():
    """Test if not/no/never/n't share the same negation subspace."""
    ADJS = ["happy", "bright", "warm", "strong", "safe", "clean", "rich", "fast",
            "smart", "kind", "calm", "free", "clear", "soft", "fresh"]
    
    stimuli = []
    for adj in ADJS:
        # Affirmative baseline
        stimuli.append({"sentence": f"the result was {adj}", "operand": adj, 
                        "neg_form": "affirm", "role": "adj"})
        
        # Different negation forms
        stimuli.append({"sentence": f"the result was not {adj}", "operand": adj,
                        "neg_form": "not", "role": "adj"})
        stimuli.append({"sentence": f"the result was no {adj}", "operand": adj,
                        "neg_form": "no", "role": "adj"})
        stimuli.append({"sentence": f"the result was never {adj}", "operand": adj,
                        "neg_form": "never", "role": "adj"})
    
    # Verb negation forms
    VERBS = ["like", "want", "know", "think", "feel", "need", "try", "help", "move", "work"]
    for verb in VERBS:
        # Affirmative baseline
        stimuli.append({"sentence": f"they {verb} the plan", "operand": verb,
                        "neg_form": "affirm", "role": "verb"})
        
        # Different negation forms
        stimuli.append({"sentence": f"they do not {verb} the plan", "operand": verb,
                        "neg_form": "not", "role": "verb"})
        stimuli.append({"sentence": f"they never {verb} the plan", "operand": verb,
                        "neg_form": "never", "role": "verb"})
        # n't contraction
        CONTRACTIONS = {
            "like": "don't like", "want": "don't want", "know": "don't know",
            "think": "don't think", "feel": "don't feel", "need": "don't need",
            "try": "don't try", "help": "don't help", "move": "don't move",
            "work": "don't work"
        }
        if verb in CONTRACTIONS:
            stimuli.append({"sentence": f"they {CONTRACTIONS[verb]} the plan", "operand": verb,
                            "neg_form": "n't", "role": "verb"})
    
    return stimuli


# =====================================================================
# MODEL LOADING
# =====================================================================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16 + device_map=auto)...")
    
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True,
                                        local_files_only=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = None
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=attn_impl)
            log(f"  attn_implementation={attn_impl} succeeded")
            break
        except Exception as e:
            log(f"  attn_implementation={attn_impl} failed: {str(e)[:120]}")
    
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    
    model.eval()
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Loaded. GPU={gpu_mem:.1f}GB")
    
    layers = get_layers(model)
    log(f"  n_layers={len(layers)}, class={type(model).__name__}")
    
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        layer_devices = {}
        for k, v in dmap.items():
            if k.startswith('model.layers.'):
                lid = k.split('.')[2]
                if lid not in layer_devices:
                    layer_devices[lid] = str(v)
        gpu_layers = sum(1 for v in layer_devices.values() if 'cuda' in str(v))
        cpu_layers = sum(1 for v in layer_devices.values() if 'cpu' in str(v))
        log(f"  Layer distribution: {gpu_layers} GPU + {cpu_layers} CPU")
        gpu_ids = sorted([int(lid) for lid, dev in layer_devices.items() if 'cuda' in str(dev)])
        if gpu_ids and gpu_layers < len(layers):
            log(f"  Last GPU layer: {max(gpu_ids)}, first CPU layer: {max(gpu_ids)+1}")
    
    return model, tok


# =====================================================================
# CAPTURE UTILITIES
# =====================================================================
def _capture_single(model, tokenizer, sent, max_len=64):
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = {li: h.detach().cpu().float() for li, h in enumerate(out.hidden_states)}
    return hs


def _find_token_pos(decoded_tokens, target):
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


def find_operand_position(tokenizer, sentence, target_word):
    """Find the position of a target word in tokenized sentence."""
    toks = tokenizer.encode(sentence, add_special_tokens=True)
    dec = [tokenizer.decode([t]).strip().lower() for t in toks]
    return _find_token_pos(dec, target_word)


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def project_onto(v, direction):
    dn = np.linalg.norm(direction)
    if dn < 1e-10:
        return np.zeros_like(v)
    d_hat = direction / dn
    return np.dot(v, d_hat) * d_hat


def remove_projection(v, direction):
    """Remove the projection of v onto direction."""
    return v - project_onto(v, direction)


# =====================================================================
# ACTIVATION PATCHING
# =====================================================================
def run_with_patched_hidden(model, tokenizer, sent, layer_idx, pos, patch_vec, max_len=64):
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(input_device)
    
    layers = get_layers(model)
    injection_done = [False]
    patch_tensor = torch.tensor(patch_vec, dtype=torch.bfloat16, device=input_device)
    
    def inject_hook(module, input, output):
        if not injection_done[0]:
            out_tuple = list(output)
            out_tuple[0] = out_tuple[0].clone()
            if pos < out_tuple[0].shape[1]:
                out_tuple[0][0, pos, :] += patch_tensor.to(out_tuple[0].dtype)
            injection_done[0] = True
            return tuple(out_tuple)
        return output
    
    handle = layers[layer_idx].register_forward_hook(inject_hook)
    with torch.no_grad():
        try:
            out = model(input_ids=input_ids, output_hidden_states=False)
            patched_logits = out.logits.detach().cpu().float()
        except Exception as e:
            log(f"  Patching error: {str(e)[:80]}")
            patched_logits = None
    handle.remove()
    return patched_logits


def compute_causal_effect(model, tokenizer, baseline_sent, target_word,
                          layer_idx, patch_vec, readout_tokens,
                          max_len=64):
    """
    Compute causal effect of injecting patch_vec at layer_idx on readout tokens.
    
    Returns: dict with logit changes for each readout token
    """
    # Baseline logits
    input_device = next(model.parameters()).device
    inputs = tokenizer(baseline_sent, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(input_device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=False)
        baseline_logits = out.logits[0, -1, :].float().cpu().numpy()
    
    # Patched logits
    patched_logits_t = run_with_patched_hidden(model, tokenizer, baseline_sent,
                                                layer_idx,
                                                find_operand_position(tokenizer, baseline_sent, target_word) or 0,
                                                patch_vec, max_len)
    if patched_logits_t is None:
        return None
    
    patched_logits = patched_logits_t[0, -1, :].float().numpy()
    
    # Compute effect on readout tokens
    effects = {}
    for tok_name, tok_id in readout_tokens.items():
        effects[tok_name] = float(patched_logits[tok_id] - baseline_logits[tok_id])
    
    return effects


# =====================================================================
# MAKE SERIALIZABLE
# =====================================================================
def make_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# =====================================================================
# MAIN
# =====================================================================
def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase308_{model_name}.txt"
    _log_file = str(log_file)
    log(f"Phase 308: Scope Causal + O-C Orthogonality + Cross-Form Negation -- {model_name}")
    
    # ---- Load model ----
    model, tok = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    nl = info.n_layers
    d_model = info.d_model
    log(f"  n_layers={nl}, d_model={d_model}")
    
    # Layer selection
    sample_layers = sorted(set([
        max(1, nl // 4), nl // 3, nl // 2, 2 * nl // 3, 3 * nl // 4, nl - 2
    ]) & set(range(1, nl)))
    log(f"Sample layers: {sample_layers}")
    
    # ---- Build stimuli ----
    scope_pairs = build_scope_pairs()
    constr_stimuli = build_construction_sentences()
    crossform_stimuli = build_crossform_sentences()
    
    log(f"  Scope pairs: {len(scope_pairs)}")
    log(f"  Construction stimuli: {len(constr_stimuli)}")
    log(f"  Cross-form stimuli: {len(crossform_stimuli)}")
    
    # Collect all unique sentences
    all_sentences = []
    sent_to_idx = {}
    
    def add_sentence(sent):
        if sent not in sent_to_idx:
            sent_to_idx[sent] = len(all_sentences)
            all_sentences.append(sent)
        return sent_to_idx[sent]
    
    # Add scope pair sentences
    for pair in scope_pairs:
        pair["_base_idx"] = add_sentence(pair["baseline"])
        pair["_narrow_idx"] = add_sentence(pair["narrow"])
        pair["_wide_idx"] = add_sentence(pair["wide"])
        # Find positions
        pair["_base_pos"] = find_operand_position(tok, pair["baseline"], pair["target_word"])
        pair["_narrow_pos"] = find_operand_position(tok, pair["narrow"], pair["target_word"])
        pair["_wide_pos"] = find_operand_position(tok, pair["wide"], pair["target_word"])
    
    # Add construction sentences
    for stim in constr_stimuli:
        stim["_idx"] = add_sentence(stim["sentence"])
        stim["target_pos"] = find_operand_position(tok, stim["sentence"], stim["token"])
    
    # Add cross-form sentences
    for stim in crossform_stimuli:
        stim["_idx"] = add_sentence(stim["sentence"])
        stim["target_pos"] = find_operand_position(tok, stim["sentence"], stim["operand"])
    
    log(f"  Unique sentences: {len(all_sentences)}")
    
    # ---- Capture all sentences ----
    log(f"Capturing {len(all_sentences)} sentences...")
    t0 = time.time()
    captures = {}
    for i, sent in enumerate(all_sentences):
        captures[i] = _capture_single(model, tok, sent)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1)
            eta = (len(all_sentences) - i - 1) / rate
            log(f"  {i+1}/{len(all_sentences)} ({rate:.1f}/s) ETA={eta:.0f}s GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
            gc.collect()
            torch.cuda.empty_cache()
    log(f"Done capturing in {time.time()-t0:.0f}s")
    
    # =====================================================================
    # PART A: SCOPE CAUSAL TESTING
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART A: SCOPE CAUSAL TESTING")
    log(f"{'='*60}")
    
    scope_results = {}
    
    for li in sample_layers:
        log(f"\n--- Layer {li}: Scope analysis ---")
        layer_results = {}
        
        # Collect scope deltas
        for pair in scope_pairs:
            ptype = pair["type"]
            base_idx, narrow_idx, wide_idx = pair["_base_idx"], pair["_narrow_idx"], pair["_wide_idx"]
            base_pos, narrow_pos, wide_pos = pair["_base_pos"], pair["_narrow_pos"], pair["_wide_pos"]
            
            if any(p is None for p in [base_pos, narrow_pos, wide_pos]):
                continue
            
            h_base = captures[base_idx].get(li)
            h_narrow = captures[narrow_idx].get(li)
            h_wide = captures[wide_idx].get(li)
            
            if h_base is None or h_narrow is None or h_wide is None:
                continue
            if base_pos >= h_base.shape[1] or narrow_pos >= h_narrow.shape[1] or wide_pos >= h_wide.shape[1]:
                continue
            
            v_base = h_base[0, base_pos, :].numpy().copy()
            v_narrow = h_narrow[0, narrow_pos, :].numpy().copy()
            v_wide = h_wide[0, wide_pos, :].numpy().copy()
            
            # Key decomposition:
            # O_narrow = v_narrow - v_base (narrow scope negation direction)
            # O_wide = v_wide - v_base (wide scope negation direction)
            # O_shared = common component (both have negation)
            # S_scope = O_wide - O_narrow (scope-specific difference)
            
            O_narrow = v_narrow - v_base
            O_wide = v_wide - v_base
            
            # O_shared: projection of O_narrow onto O_wide direction
            O_wide_norm = np.linalg.norm(O_wide)
            O_narrow_norm = np.linalg.norm(O_narrow)
            
            if O_wide_norm < 1e-10 or O_narrow_norm < 1e-10:
                continue
            
            cos_narrow_wide = cosine_sim(O_narrow, O_wide)
            
            # Shared component: average of normalized directions
            O_narrow_hat = O_narrow / O_narrow_norm
            O_wide_hat = O_wide / O_wide_norm
            O_shared_hat = (O_narrow_hat + O_wide_hat) / 2
            O_shared = O_shared_hat / max(np.linalg.norm(O_shared_hat), 1e-10) * (O_narrow_norm + O_wide_norm) / 2
            
            # Scope-specific: residual after removing shared
            S_narrow = O_narrow - project_onto(O_narrow, O_shared)
            S_wide = O_wide - project_onto(O_wide, O_shared)
            S_scope = S_wide - S_narrow  # Wide-scope minus narrow-scope residual
            
            key = f"{ptype}_{pair['target_word']}"
            layer_results[key] = {
                "type": ptype,
                "target": pair["target_word"],
                "O_narrow_norm": float(O_narrow_norm),
                "O_wide_norm": float(O_wide_norm),
                "cos_O_narrow_O_wide": float(cos_narrow_wide),
                "S_narrow_norm": float(np.linalg.norm(S_narrow)),
                "S_wide_norm": float(np.linalg.norm(S_wide)),
                "S_scope_norm": float(np.linalg.norm(S_scope)),
                "O_shared_norm": float(np.linalg.norm(O_shared)),
                "O_narrow": O_narrow,
                "O_wide": O_wide,
                "O_shared": O_shared,
                "S_scope": S_scope,
                "v_base": v_base,
                "base_sent": pair["baseline"],
                "narrow_sent": pair["narrow"],
                "wide_sent": pair["wide"],
                "base_pos": base_pos,
            }
            
            log(f"  {key}: cos(O_narrow,O_wide)={cos_narrow_wide:+.3f} "
                f"|O_narrow|={O_narrow_norm:.2f} |O_wide|={O_wide_norm:.2f} "
                f"|S_scope|={np.linalg.norm(S_scope):.2f}")
        
        scope_results[li] = layer_results
    
    # ---- Scope causal testing ----
    log(f"\n--- Scope Causal Patching ---")
    
    # Readout tokens for scope
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    some_id = tok.encode("some", add_special_tokens=False)[0]
    all_id = tok.encode("all", add_special_tokens=False)[0]
    none_id = tok.encode("none", add_special_tokens=False)[0]
    not_id = tok.encode("not", add_special_tokens=False)[0]
    
    scope_readout = {"yes": yes_id, "no": no_id, "some": some_id, "all": all_id, "none": none_id, "not": not_id}
    
    scope_causal_results = {}
    
    for li in sample_layers:
        if li not in scope_results:
            continue
        log(f"\n  Layer {li}: Scope causal patching")
        layer_causal = {}
        
        # Average directions across all scope pairs
        all_O_shared = []
        all_S_scope = []
        all_O_narrow = []
        all_O_wide = []
        
        for key, data in scope_results[li].items():
            all_O_shared.append(data["O_shared"])
            all_S_scope.append(data["S_scope"])
            all_O_narrow.append(data["O_narrow"])
            all_O_wide.append(data["O_wide"])
        
        if not all_O_shared:
            continue
        
        avg_O_shared = np.mean(all_O_shared, axis=0)
        avg_S_scope = np.mean(all_S_scope, axis=0)
        avg_O_narrow = np.mean(all_O_narrow, axis=0)
        avg_O_wide = np.mean(all_O_wide, axis=0)
        
        # Scale for patching (0.1 * norm of average O_narrow)
        scale = np.linalg.norm(avg_O_narrow) * 0.1
        
        # Random direction baseline
        rng = np.random.RandomState(42)
        random_dir = rng.randn(d_model)
        random_dir = random_dir / np.linalg.norm(random_dir) * scale
        
        # Test on a few representative scope pairs
        test_pairs = [(k, v) for k, v in scope_results[li].items() 
                      if v["type"] in ["quantifier_scope", "embedding_scope", "infinitive_scope"]][:6]
        
        for key, data in test_pairs:
            base_sent = data["base_sent"]
            base_pos = data["base_pos"]
            
            if base_pos is None:
                continue
            
            # Get baseline logits
            input_device = next(model.parameters()).device
            inputs = tok(base_sent, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, output_hidden_states=False)
                baseline_logits = out.logits[0, -1, :].float().cpu().numpy()
            
            # Patch with each direction
            patch_configs = {
                "O_shared": avg_O_shared / max(np.linalg.norm(avg_O_shared), 1e-10) * scale,
                "S_scope": avg_S_scope / max(np.linalg.norm(avg_S_scope), 1e-10) * scale,
                "O_narrow": avg_O_narrow / max(np.linalg.norm(avg_O_narrow), 1e-10) * scale,
                "O_wide": avg_O_wide / max(np.linalg.norm(avg_O_wide), 1e-10) * scale,
                "random": random_dir,
            }
            
            pair_effects = {}
            for pname, pvec in patch_configs.items():
                patched = run_with_patched_hidden(model, tok, base_sent, li, base_pos, pvec)
                if patched is not None:
                    patched_logits = patched[0, -1, :].float().numpy()
                    effects = {}
                    for tok_name, tok_id in scope_readout.items():
                        effects[tok_name] = float(patched_logits[tok_id] - baseline_logits[tok_id])
                    pair_effects[pname] = effects
                else:
                    pair_effects[pname] = None
            
            # Also test per-pair specific directions
            for dname, dvec in [("O_narrow_local", data["O_narrow"]),
                                ("O_wide_local", data["O_wide"]),
                                ("S_scope_local", data["S_scope"])]:
                dn = np.linalg.norm(dvec)
                if dn > 1e-10:
                    pvec = dvec / dn * scale
                    patched = run_with_patched_hidden(model, tok, base_sent, li, base_pos, pvec)
                    if patched is not None:
                        patched_logits = patched[0, -1, :].float().numpy()
                        effects = {}
                        for tok_name, tok_id in scope_readout.items():
                            effects[tok_name] = float(patched_logits[tok_id] - baseline_logits[tok_id])
                        pair_effects[dname] = effects
            
            layer_causal[key] = pair_effects
            
            # Log summary
            if pair_effects.get("O_shared") and pair_effects.get("S_scope"):
                o_not_eff = pair_effects["O_shared"].get("not", 0)
                s_scope_eff = pair_effects["S_scope"].get("not", 0)
                rand_eff = pair_effects.get("random", {}).get("not", 0)
                log(f"    {key}: O_shared→not={o_not_eff:+.4f} S_scope→not={s_scope_eff:+.4f} random→not={rand_eff:+.4f}")
        
        scope_causal_results[li] = layer_causal
    
    # =====================================================================
    # PART B: O-C ORTHOGONALITY
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART B: O(not) vs C(construction) ORTHOGONALITY")
    log(f"{'='*60}")
    
    oc_ortho_results = {}
    
    for li in sample_layers:
        log(f"\n--- Layer {li}: O-C orthogonality ---")
        
        # Extract O(not) direction from scope pairs
        O_not_dirs = []
        for pair in scope_pairs:
            base_idx, narrow_idx = pair["_base_idx"], pair["_narrow_idx"]
            base_pos, narrow_pos = pair["_base_pos"], pair["_narrow_pos"]
            
            if base_pos is None or narrow_pos is None:
                continue
            
            h_base = captures[base_idx].get(li)
            h_narrow = captures[narrow_idx].get(li)
            if h_base is None or h_narrow is None:
                continue
            if base_pos >= h_base.shape[1] or narrow_pos >= h_narrow.shape[1]:
                continue
            
            v_base = h_base[0, base_pos, :].numpy().copy()
            v_narrow = h_narrow[0, narrow_pos, :].numpy().copy()
            O_not_dirs.append(v_narrow - v_base)
        
        # Extract C(construction) direction from construction sentences
        # For each token, compute across-frame variation
        token_frames = defaultdict(list)
        for stim in constr_stimuli:
            idx = stim.get("_idx")
            pos = stim.get("target_pos")
            if idx is None or pos is None:
                continue
            hs = captures[idx].get(li)
            if hs is None or pos >= hs.shape[1]:
                continue
            h = hs[0, pos, :].numpy().copy()
            token_frames[stim["token"]].append({"vec": h, "frame": stim["frame"], "role": stim["role"]})
        
        # Compute C direction: within-token across-frame variation
        C_dirs = []
        for token, frames in token_frames.items():
            if len(frames) < 2:
                continue
            vecs = [f["vec"] for f in frames]
            mean_vec = np.mean(vecs, axis=0)
            for i in range(len(vecs)):
                for j in range(i+1, len(vecs)):
                    C_dirs.append(vecs[j] - vecs[i])
        
        if not O_not_dirs or not C_dirs:
            log(f"  Insufficient data: O_not={len(O_not_dirs)}, C={len(C_dirs)}")
            continue
        
        # Average O(not) direction
        avg_O_not = np.mean(O_not_dirs, axis=0)
        avg_O_not_norm = np.linalg.norm(avg_O_not)
        
        # Average C direction
        avg_C = np.mean(C_dirs, axis=0)
        avg_C_norm = np.linalg.norm(avg_C)
        
        cos_O_C = cosine_sim(avg_O_not, avg_C)
        
        # Also compute per-pair cosines
        cos_values = []
        for o_dir in O_not_dirs:
            for c_dir in C_dirs:
                cos_values.append(cosine_sim(o_dir, c_dir))
        
        # PCA on C directions to get principal construction direction
        C_matrix = np.array(C_dirs)
        C_mean = np.mean(C_matrix, axis=0)
        C_centered = C_matrix - C_mean
        
        if len(C_centered) > 1:
            from numpy.linalg import svd
            U, S, Vt = svd(C_centered, full_matrices=False)
            C_pc1 = Vt[0]
            C_pc1_var = S[0]**2 / np.sum(S**2)
        else:
            C_pc1 = avg_C / max(avg_C_norm, 1e-10)
            C_pc1_var = 1.0
        
        cos_O_Cpc1 = cosine_sim(avg_O_not, C_pc1)
        
        # O_clean: remove C projection from O
        O_clean = remove_projection(avg_O_not, avg_C)
        O_clean_norm = np.linalg.norm(O_clean)
        O_clean_ratio = O_clean_norm / max(avg_O_not_norm, 1e-10)
        
        # Also remove C_pc1 projection
        O_clean_pc1 = remove_projection(avg_O_not, C_pc1)
        O_clean_pc1_norm = np.linalg.norm(O_clean_pc1)
        O_clean_pc1_ratio = O_clean_pc1_norm / max(avg_O_not_norm, 1e-10)
        
        # Causal test: does O_clean still have effect?
        scale_oc = avg_O_not_norm * 0.1
        
        # Test on baseline sentence
        test_base = scope_pairs[0]["baseline"] if scope_pairs else "all students passed"
        test_pos = find_operand_position(tok, test_base, scope_pairs[0]["target_word"]) if scope_pairs else 0
        
        if test_pos is not None:
            input_device = next(model.parameters()).device
            inputs = tok(test_base, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, output_hidden_states=False)
                base_logits = out.logits[0, -1, :].float().cpu().numpy()
            
            # Patch configs
            oc_patches = {
                "O_not_raw": avg_O_not / max(avg_O_not_norm, 1e-10) * scale_oc,
                "O_clean_C": O_clean / max(O_clean_norm, 1e-10) * scale_oc if O_clean_norm > 1e-10 else np.zeros(d_model),
                "O_clean_Cpc1": O_clean_pc1 / max(O_clean_pc1_norm, 1e-10) * scale_oc if O_clean_pc1_norm > 1e-10 else np.zeros(d_model),
                "C_only": avg_C / max(avg_C_norm, 1e-10) * scale_oc,
                "random": np.random.RandomState(42).randn(d_model)
            }
            oc_patches["random"] = oc_patches["random"] / np.linalg.norm(oc_patches["random"]) * scale_oc
            
            oc_effects = {}
            for pname, pvec in oc_patches.items():
                patched = run_with_patched_hidden(model, tok, test_base, li, test_pos, pvec)
                if patched is not None:
                    patched_logits = patched[0, -1, :].float().numpy()
                    effects = {}
                    for tok_name, tok_id in scope_readout.items():
                        effects[tok_name] = float(patched_logits[tok_id] - base_logits[tok_id])
                    oc_effects[pname] = effects
                else:
                    oc_effects[pname] = None
        else:
            oc_effects = {}
        
        layer_ortho = {
            "cos_O_C": float(cos_O_C),
            "cos_O_Cpc1": float(cos_O_Cpc1),
            "O_not_norm": float(avg_O_not_norm),
            "C_norm": float(avg_C_norm),
            "C_pc1_var": float(C_pc1_var),
            "O_clean_C_ratio": float(O_clean_ratio),
            "O_clean_Cpc1_ratio": float(O_clean_pc1_ratio),
            "n_O_not": len(O_not_dirs),
            "n_C": len(C_dirs),
            "cos_O_C_mean": float(np.mean(cos_values)),
            "cos_O_C_std": float(np.std(cos_values)),
            "causal_effects": oc_effects,
        }
        
        oc_ortho_results[li] = layer_ortho
        
        log(f"  cos(O_not, C)={cos_O_C:+.3f}")
        log(f"  cos(O_not, C_pc1)={cos_O_Cpc1:+.3f} (C_pc1_var={C_pc1_var:.1%})")
        log(f"  O_clean/C ratio={O_clean_ratio:.3f} (after removing C)")
        log(f"  O_clean/Cpc1 ratio={O_clean_pc1_ratio:.3f} (after removing C_pc1)")
        
        if oc_effects:
            for pname, eff in oc_effects.items():
                if eff:
                    not_eff = eff.get("not", 0)
                    log(f"    {pname} → not: {not_eff:+.4f}")
    
    # =====================================================================
    # PART C: CROSS-FORM NEGATION
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART C: CROSS-FORM NEGATION SUBSPACE")
    log(f"{'='*60}")
    
    crossform_results = {}
    
    for li in sample_layers:
        log(f"\n--- Layer {li}: Cross-form negation ---")
        
        # Group cross-form stimuli by operand and role
        operand_groups = defaultdict(lambda: defaultdict(dict))
        for stim in crossform_stimuli:
            key = (stim["operand"], stim["role"])
            operand_groups[key][stim["neg_form"]] = stim
        
        # Extract per-form negation directions
        form_dirs = defaultdict(lambda: defaultdict(list))  # {form: {role: [dirs]}}
        
        for (operand, role), forms in operand_groups.items():
            if "affirm" not in forms:
                continue
            
            affirm_stim = forms["affirm"]
            affirm_idx = affirm_stim.get("_idx")
            affirm_pos = affirm_stim.get("target_pos")
            if affirm_idx is None or affirm_pos is None:
                continue
            
            h_affirm = captures[affirm_idx].get(li)
            if h_affirm is None or affirm_pos >= h_affirm.shape[1]:
                continue
            
            v_affirm = h_affirm[0, affirm_pos, :].numpy().copy()
            
            for form_name, form_stim in forms.items():
                if form_name == "affirm":
                    continue
                
                form_idx = form_stim.get("_idx")
                form_pos = form_stim.get("target_pos")
                if form_idx is None or form_pos is None:
                    continue
                
                h_form = captures[form_idx].get(li)
                if h_form is None or form_pos >= h_form.shape[1]:
                    continue
                
                v_form = h_form[0, form_pos, :].numpy().copy()
                form_dir = v_form - v_affirm
                form_dirs[form_name][role].append(form_dir)
        
        # Average directions
        avg_form_dirs = {}
        for form, role_data in form_dirs.items():
            avg_form_dirs[form] = {}
            for role, dirs in role_data.items():
                avg_dir = np.mean(dirs, axis=0)
                avg_form_dirs[form][role] = avg_dir
                log(f"  {form}/{role}: {len(dirs)} operands, norm={np.linalg.norm(avg_dir):.2f}")
        
        # Cross-form cosine similarities
        forms = sorted(avg_form_dirs.keys())
        cross_cos = {}
        
        for i, f1 in enumerate(forms):
            for j, f2 in enumerate(forms):
                if i >= j:
                    continue
                
                cos_per_role = {}
                for role in ["adj", "verb"]:
                    d1 = avg_form_dirs[f1].get(role)
                    d2 = avg_form_dirs[f2].get(role)
                    if d1 is not None and d2 is not None:
                        cos_per_role[role] = cosine_sim(d1, d2)
                
                cross_cos[f"{f1}_vs_{f2}"] = cos_per_role
                
                cos_vals = list(cos_per_role.values())
                if cos_vals:
                    adj_str = f"{cos_per_role['adj']:+.3f}" if 'adj' in cos_per_role else 'N/A'
                    verb_str = f"{cos_per_role['verb']:+.3f}" if 'verb' in cos_per_role else 'N/A'
                    log(f"  cos({f1}, {f2}): adj={adj_str} verb={verb_str}")
        
        # Cross-form causal injection test
        # Test: inject O_not into "no" sentence, and vice versa
        cross_causal = {}
        
        # Find test pairs for cross-form injection
        test_operands_adj = ["happy", "bright", "warm"][:3]
        test_operands_verb = ["like", "want", "know"][:3]
        
        for operand in test_operands_adj:
            for role in ["adj"]:
                forms_data = operand_groups.get((operand, role), {})
                if "affirm" not in forms_data or "not" not in forms_data:
                    continue
                
                affirm_stim = forms_data["affirm"]
                base_sent = affirm_stim["sentence"]
                affirm_pos = affirm_stim.get("target_pos")
                if affirm_pos is None:
                    continue
                
                # Get baseline logits
                input_device = next(model.parameters()).device
                inputs = tok(base_sent, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                
                with torch.no_grad():
                    out = model(input_ids=input_ids, output_hidden_states=False)
                    base_logits = out.logits[0, -1, :].float().cpu().numpy()
                
                # Inject each form's direction
                scale_cf = 0.1
                for form_name in ["not", "no", "never"]:
                    if form_name not in avg_form_dirs or role not in avg_form_dirs[form_name]:
                        continue
                    
                    form_dir = avg_form_dirs[form_name][role]
                    form_norm = np.linalg.norm(form_dir)
                    if form_norm < 1e-10:
                        continue
                    
                    pvec = form_dir / form_norm * scale_cf
                    patched = run_with_patched_hidden(model, tok, base_sent, li, affirm_pos, pvec)
                    if patched is not None:
                        patched_logits = patched[0, -1, :].float().cpu().numpy()
                        effects = {}
                        for tok_name, tok_id in scope_readout.items():
                            effects[tok_name] = float(patched_logits[tok_id] - base_logits[tok_id])
                        cross_causal[f"{operand}_{form_name}"] = effects
        
        crossform_results[li] = {
            "cross_cos": cross_cos,
            "cross_causal": cross_causal,
            "form_dir_norms": {f: {r: float(np.linalg.norm(d)) for r, d in rd.items()} 
                              for f, rd in avg_form_dirs.items()},
        }
    
    # =====================================================================
    # SCOPE SUMMARY BY TYPE
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"SCOPE SUMMARY BY TYPE")
    log(f"{'='*60}")
    
    scope_type_summary = {}
    for li in sample_layers:
        if li not in scope_results:
            continue
        
        type_data = defaultdict(list)
        for key, data in scope_results[li].items():
            type_data[data["type"]].append(data)
        
        type_summary = {}
        for stype, items in type_data.items():
            avg_cos = np.mean([d["cos_O_narrow_O_wide"] for d in items])
            avg_O_narrow = np.mean([d["O_narrow_norm"] for d in items])
            avg_O_wide = np.mean([d["O_wide_norm"] for d in items])
            avg_S_scope = np.mean([d["S_scope_norm"] for d in items])
            
            type_summary[stype] = {
                "n_pairs": len(items),
                "avg_cos_O_narrow_O_wide": float(avg_cos),
                "avg_O_narrow_norm": float(avg_O_narrow),
                "avg_O_wide_norm": float(avg_O_wide),
                "avg_S_scope_norm": float(avg_S_scope),
                "scope_ratio": float(avg_S_scope / max(avg_O_narrow, 1e-10)),
            }
            
            log(f"  L{li} {stype}: cos(O_narrow,O_wide)={avg_cos:+.3f} "
                f"|O_narrow|={avg_O_narrow:.2f} |O_wide|={avg_O_wide:.2f} "
                f"|S_scope|={avg_S_scope:.2f} ratio={avg_S_scope/max(avg_O_narrow,1e-10):.3f}")
        
        scope_type_summary[li] = type_summary
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    log(f"\nSaving results...")
    
    # Remove non-serializable data
    scope_results_clean = {}
    for li, layer_data in scope_results.items():
        scope_results_clean[li] = {}
        for key, data in layer_data.items():
            clean = {k: v for k, v in data.items() 
                     if k not in ["O_narrow", "O_wide", "O_shared", "S_scope", "v_base"]}
            scope_results_clean[li][key] = clean
    
    results = {
        "model": model_name,
        "n_layers": nl,
        "d_model": d_model,
        "sample_layers": sample_layers,
        "scope_results": make_serializable(scope_results_clean),
        "scope_type_summary": make_serializable(scope_type_summary),
        "scope_causal_results": make_serializable(scope_causal_results),
        "oc_ortho_results": make_serializable(oc_ortho_results),
        "crossform_results": make_serializable(crossform_results),
    }
    
    out_path = RESULT_DIR / f"{model_name}_scope_causal.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"Saved to {out_path}")
    
    # ---- Release model ----
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Phase 308 complete for {model_name}")


if __name__ == "__main__":
    main()
