"""
Phase 310: Unified Normalized Causal Test + Direction Gain Analysis
====================================================================
Resolve the Phase 308/309 O-C conflict by:
1. Unit direction injection (消除范数主导)
2. Multiple injection scales (alpha sweep)
3. Direction gain: measure how each direction is amplified through later layers
4. Multiple baseline sentences (消除单句偏差)
5. Unified O/R/C/S definitions

Key questions:
- Is DS7B's O-R-C collapse real after normalization?
- What is the direction gain ratio for O_clean vs O_raw?
- Does Phase 308's O⊥C hold under unit injection?

Stimulus: reuse Phase 309 definitions
Causal test: 5 baseline sentences × 8 directions × 4 scales × 6 layers

Usage:
  python tests/glm5/phase310_unified_norm_causal.py qwen3
  python tests/glm5/phase310_unified_norm_causal.py glm4
  python tests/glm5/phase310_unified_norm_causal.py deepseek7b
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

RESULT_DIR = Path("results/phase310_unified_norm_causal")
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
# STIMULUS DEFINITIONS (reuse from Phase 309)
# =====================================================================
DUAL_TOKENS = {
    "open":   ("the door is open", "they open the door", "the open door was"),
    "clear":  ("the path is clear", "they clear the path", "the clear path was"),
    "warm":   ("the room is warm", "they warm the room", "the warm room was"),
    "clean":  ("the floor is clean", "they clean the floor", "the clean floor was"),
    "dry":    ("the cloth is dry", "they dry the cloth", "the dry cloth was"),
    "close":  ("the store is close", "they close the store", "the close store was"),
    "free":   ("the bird is free", "they free the bird", "the free bird was"),
    "quiet":  ("the room is quiet", "they quiet the room", "the quiet room was"),
    "cool":   ("the water is cool", "they cool the water", "the cool water was"),
    "smooth": ("the surface is smooth", "they smooth the surface", "the smooth surface was"),
    "cold":   ("the water is cold", "a cold hit them", "the cold wind blew"),
    "light":  ("the bag is light", "a light shone through", "the light switch was"),
    "fire":   ("the hot fire was", "they will fire the worker", "they will fire again"),
    "record": ("the old record was", "they will record the data", "the new record was"),
    "run":    ("the long run was", "they will run the program", "a fast run was"),
    "book":   ("the new book was", "they will book the room", "the good book was"),
}

ADJ_OPERANDS = [
    "happy", "bright", "warm", "strong", "safe", "clean", "rich", "fast",
    "smart", "kind", "calm", "free", "clear", "soft", "fresh", "deep",
    "high", "wide", "loud", "sharp"
]

VERB_OPERANDS = [
    "like", "want", "know", "think", "feel", "need", "try", "help",
    "move", "work"
]

NOUN_OPERANDS = [
    "person", "student", "teacher", "doctor", "system",
    "problem", "method", "result", "reason", "chance"
]

# Baseline sentences for causal test (5 diverse affirmatives)
BASELINES = [
    "the result was happy",
    "the outcome was bright",
    "the feeling was warm",
    "the plan was safe",
    "the method was clear",
]


# =====================================================================
# MODEL LOADING
# =====================================================================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16 + device_map=auto + flash)...")
    
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
    
    return model, tok


# =====================================================================
# CAPTURE + POSITION
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
    for i, t in enumerate(decoded_tokens):
        if t == target_lower:
            return i
    for i, t in enumerate(decoded_tokens):
        if target_lower in t or t in target_lower:
            return i
    if len(target_lower) >= 2:
        for i, t in enumerate(decoded_tokens):
            if target_lower[:3] in t or t[:3] in target_lower:
                return i
    return None


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
    return v - project_onto(v, direction)


# =====================================================================
# ACTIVATION PATCHING (with unit + scaled injection)
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
            patched_logits = None
    handle.remove()
    return patched_logits


def get_not_prob(logits, tokenizer):
    """Get P('not') from logits at the position where 'not' would be predicted."""
    # For "the result was happy", the next token after "was" should be "happy"
    # We want to measure P("not") at the position before the operand
    # Actually we measure at last position (after the full sentence)
    # This gives the model's continuation probability
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    
    # Find "not" token
    not_id = tokenizer.encode("not", add_special_tokens=False)
    if not_id:
        return float(probs[not_id[0]])
    return 0.0


def get_target_probs(logits, tokenizer, target_words):
    """Get P(target_words) from logits at last position."""
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    result = {}
    for word in target_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            result[word] = float(probs[ids[0]])
    return result


# =====================================================================
# DIRECTION GAIN ANALYSIS
# =====================================================================
def compute_direction_gain(model, tokenizer, direction, layer_idx, pos, sent, max_len=64):
    """
    Compute how much a direction at layer l is amplified at the output.
    
    Method: inject a small unit direction at layer l, measure output logit change.
    Gain = ||Δlogits|| / ||injected_direction||
    
    For unit direction, this simplifies to ||Δlogits||.
    """
    # Clean baseline
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(input_device)
    
    with torch.no_grad():
        baseline_out = model(input_ids=input_ids, output_hidden_states=False)
        baseline_logits = baseline_out.logits.detach().cpu().float()
    
    # Patch with unit direction
    d_norm = np.linalg.norm(direction)
    if d_norm < 1e-10:
        return 0.0, baseline_logits, baseline_logits
    
    unit_dir = direction / d_norm
    # Inject at 1.0 scale (unit vector)
    patched = run_with_patched_hidden(model, tokenizer, sent, layer_idx, pos, unit_dir)
    
    if patched is None:
        return 0.0, baseline_logits, baseline_logits
    
    delta_logits = (patched - baseline_logits).numpy()
    gain = float(np.linalg.norm(delta_logits))
    
    return gain, baseline_logits, patched


# =====================================================================
# MAIN
# =====================================================================
def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase310_{model_name}.txt"
    _log_file = str(log_file)
    log(f"Phase 310: Unified Normalized Causal Test -- {model_name}")
    
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
    
    # ---- Build all stimuli ----
    all_sentences_for_dirs = []
    
    # Role stimuli
    role_sents = {}
    for token, (adj_sent, verb_sent, attr_sent) in DUAL_TOKENS.items():
        role_sents[f"{token}_adj"] = adj_sent
        role_sents[f"{token}_verb"] = verb_sent
        role_sents[f"{token}_attr"] = attr_sent
        all_sentences_for_dirs.extend([adj_sent, verb_sent, attr_sent])
    
    # Operator stimuli (affirm + not + maybe + must + can + never)
    op_sents = {}
    for adj in ADJ_OPERANDS:
        for op in ["affirm", "not", "maybe", "must", "can", "never"]:
            if op == "affirm":
                s = f"the result was {adj}"
            elif op == "not":
                s = f"the result was not {adj}"
            else:
                s = f"the result was {op} {adj}"
            op_sents[f"{adj}_adj_{op}"] = s
            all_sentences_for_dirs.append(s)
    
    for verb in VERB_OPERANDS:
        for op in ["affirm", "not", "maybe", "must", "can", "never"]:
            if op == "affirm":
                s = f"they {verb} the plan"
            elif op == "not":
                s = f"they do not {verb} the plan"
            else:
                s = f"they {op} {verb} the plan"
            op_sents[f"{verb}_verb_{op}"] = s
            all_sentences_for_dirs.append(s)
    
    for noun in NOUN_OPERANDS:
        s_aff = f"that {noun} was available"
        s_not = f"that {noun} was not available"
        op_sents[f"{noun}_noun_affirm"] = s_aff
        op_sents[f"{noun}_noun_not"] = s_not
        all_sentences_for_dirs.extend([s_aff, s_not])
    
    # Antonym stimuli
    antonym_pairs = [
        ("happy", "sad"), ("bright", "dark"), ("warm", "cold"), ("strong", "weak"),
        ("safe", "dangerous"), ("clean", "dirty"), ("rich", "poor"), ("fast", "slow"),
        ("smart", "stupid"), ("kind", "cruel"),
    ]
    for pos, neg in antonym_pairs:
        s_pos = f"the result was {pos}"
        s_neg = f"the result was {neg}"
        op_sents[f"{pos}_antonym_pos"] = s_pos
        op_sents[f"{neg}_antonym_neg"] = s_neg
        all_sentences_for_dirs.extend([s_pos, s_neg])
    
    # Deduplicate
    all_sentences_for_dirs = list(dict.fromkeys(all_sentences_for_dirs))
    log(f"Total sentences for direction extraction: {len(all_sentences_for_dirs)}")
    
    # ---- Capture all sentences ----
    log(f"Capturing {len(all_sentences_for_dirs)} sentences...")
    t0 = time.time()
    captures = {}
    for i, sent in enumerate(all_sentences_for_dirs):
        captures[sent] = _capture_single(model, tok, sent)
        if (i + 1) % 30 == 0:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1)
            eta = (len(all_sentences_for_dirs) - i - 1) / rate
            log(f"  {i+1}/{len(all_sentences_for_dirs)} ({rate:.1f}/s) ETA={eta:.0f}s GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
            gc.collect()
            torch.cuda.empty_cache()
    log(f"Done capturing in {time.time()-t0:.0f}s")
    
    # Resolve positions for all sentences
    def get_pos(sent, target):
        toks = tok.encode(sent, add_special_tokens=True)
        dec = [tok.decode([t]).strip().lower() for t in toks]
        return _find_token_pos(dec, target)
    
    # =====================================================================
    # EXTRACT FUNCTIONAL DIRECTIONS (UNIFIED)
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"EXTRACTING UNIFIED FUNCTIONAL DIRECTIONS")
    log(f"{'='*60}")
    
    all_results = {}
    
    for li in sample_layers:
        log(f"\n--- Layer {li} ---")
        dirs = {}
        
        # ---- R(role): mean(verb - adj) for dual tokens ----
        adj_vecs, verb_vecs = [], []
        for token, (adj_sent, verb_sent, _) in DUAL_TOKENS.items():
            h_adj = captures[adj_sent].get(li)
            h_verb = captures[verb_sent].get(li)
            p_adj = get_pos(adj_sent, token)
            p_verb = get_pos(verb_sent, token)
            if h_adj is not None and h_verb is not None and p_adj is not None and p_verb is not None:
                if p_adj < h_adj.shape[1] and p_verb < h_verb.shape[1]:
                    adj_vecs.append(h_adj[0, p_adj, :].numpy().copy())
                    verb_vecs.append(h_verb[0, p_verb, :].numpy().copy())
        
        if adj_vecs and verb_vecs:
            R_raw = np.mean(verb_vecs, axis=0) - np.mean(adj_vecs, axis=0)
            R_norm = np.linalg.norm(R_raw)
            R_unit = R_raw / R_norm if R_norm > 1e-10 else np.zeros_like(R_raw)
            dirs["R_raw"] = R_raw
            dirs["R_unit"] = R_unit
            log(f"  R: {len(adj_vecs)} pairs, norm={R_norm:.2f}")
        
        # ---- C(construction): centered frame variation ----
        C_vecs = []
        for token, (adj_sent, verb_sent, attr_sent) in DUAL_TOKENS.items():
            vecs = []
            for sent in [adj_sent, verb_sent, attr_sent]:
                hs = captures[sent].get(li)
                p = get_pos(sent, token)
                if hs is not None and p is not None and p < hs.shape[1]:
                    vecs.append(hs[0, p, :].numpy().copy())
            if len(vecs) >= 2:
                mean_v = np.mean(vecs, axis=0)
                for v in vecs:
                    C_vecs.append(v - mean_v)
        
        if C_vecs:
            C_raw = np.mean(C_vecs, axis=0)
            C_norm = np.linalg.norm(C_raw)
            
            # C_pc1 via SVD
            C_matrix = np.array(C_vecs)
            U, S, Vt = np.linalg.svd(C_matrix, full_matrices=False)
            C_pc1 = Vt[0]
            C_pc1_var = S[0]**2 / np.sum(S**2)
            
            dirs["C_raw"] = C_raw
            dirs["C_unit"] = C_raw / C_norm if C_norm > 1e-10 else np.zeros_like(C_raw)
            dirs["C_pc1"] = C_pc1
            dirs["C_pc1_unit"] = C_pc1  # already unit from SVD
            dirs["C_pc1_var"] = C_pc1_var
            log(f"  C: {len(C_vecs)} vecs, norm={C_norm:.4f}, pc1_var={C_pc1_var:.1%}")
        else:
            dirs["C_pc1_var"] = 0.0
        
        # ---- O(operator): not vs affirm ----
        for op_name in ["not", "maybe", "must", "can", "never"]:
            op_deltas = []
            for role, operands in [("adj", ADJ_OPERANDS), ("verb", VERB_OPERANDS)]:
                for operand in operands:
                    if role == "adj":
                        aff_sent = f"the result was {operand}"
                        if op_name == "not":
                            op_sent = f"the result was not {operand}"
                        else:
                            op_sent = f"the result was {op_name} {operand}"
                        target = operand
                    else:
                        aff_sent = f"they {operand} the plan"
                        if op_name == "not":
                            op_sent = f"they do not {operand} the plan"
                        else:
                            op_sent = f"they {op_name} {operand} the plan"
                        target = operand
                    
                    h_aff = captures.get(aff_sent, {}).get(li)
                    h_op = captures.get(op_sent, {}).get(li)
                    
                    if h_aff is not None and h_op is not None:
                        p_aff = get_pos(aff_sent, target)
                        p_op = get_pos(op_sent, target)
                        
                        if p_aff is not None and p_op is not None:
                            if p_aff < h_aff.shape[1] and p_op < h_op.shape[1]:
                                delta = h_op[0, p_op, :].numpy().copy() - h_aff[0, p_aff, :].numpy().copy()
                                op_deltas.append(delta)
            
            if op_deltas:
                O_raw = np.mean(op_deltas, axis=0)
                O_norm = np.linalg.norm(O_raw)
                dirs[f"O_{op_name}_raw"] = O_raw
                dirs[f"O_{op_name}_unit"] = O_raw / O_norm if O_norm > 1e-10 else np.zeros_like(O_raw)
                log(f"  O({op_name}): {len(op_deltas)} deltas, norm={O_norm:.2f}")
        
        # ---- A(antonym): negative - positive ----
        ant_deltas = []
        for pos, neg in antonym_pairs:
            s_pos = f"the result was {pos}"
            s_neg = f"the result was {neg}"
            h_pos = captures.get(s_pos, {}).get(li)
            h_neg = captures.get(s_neg, {}).get(li)
            p_pos = get_pos(s_pos, pos)
            p_neg = get_pos(s_neg, neg)
            if h_pos is not None and h_neg is not None and p_pos is not None and p_neg is not None:
                if p_pos < h_pos.shape[1] and p_neg < h_neg.shape[1]:
                    ant_deltas.append(h_neg[0, p_neg, :].numpy().copy() - h_pos[0, p_pos, :].numpy().copy())
        
        if ant_deltas:
            A_raw = np.mean(ant_deltas, axis=0)
            A_norm = np.linalg.norm(A_raw)
            dirs["A_raw"] = A_raw
            dirs["A_unit"] = A_raw / A_norm if A_norm > 1e-10 else np.zeros_like(A_raw)
            log(f"  A(antonym): {len(ant_deltas)} pairs, norm={A_norm:.2f}")
        
        # ---- O_clean directions: remove R, C, R+C projections ----
        O_not_raw = dirs.get("O_not_raw")
        R_raw = dirs.get("R_raw")
        C_pc1 = dirs.get("C_pc1")
        A_raw_d = dirs.get("A_raw")
        
        if O_not_raw is not None:
            # O_clean_R
            if R_raw is not None:
                O_clean_R = remove_projection(O_not_raw, R_raw)
                dirs["O_clean_R_raw"] = O_clean_R
                dirs["O_clean_R_unit"] = O_clean_R / np.linalg.norm(O_clean_R) if np.linalg.norm(O_clean_R) > 1e-10 else np.zeros_like(O_clean_R)
                log(f"  O_clean_R: norm={np.linalg.norm(O_clean_R):.4f}, ratio={np.linalg.norm(O_clean_R)/max(np.linalg.norm(O_not_raw),1e-10):.4f}")
            
            # O_clean_C (using C_pc1, not C_raw which may have tiny norm)
            if C_pc1 is not None:
                O_clean_C = remove_projection(O_not_raw, C_pc1)
                dirs["O_clean_C_raw"] = O_clean_C
                dirs["O_clean_C_unit"] = O_clean_C / np.linalg.norm(O_clean_C) if np.linalg.norm(O_clean_C) > 1e-10 else np.zeros_like(O_clean_C)
                log(f"  O_clean_C: norm={np.linalg.norm(O_clean_C):.4f}, ratio={np.linalg.norm(O_clean_C)/max(np.linalg.norm(O_not_raw),1e-10):.4f}")
            
            # O_clean_RC
            if R_raw is not None and C_pc1 is not None:
                O_clean_RC = remove_projection(remove_projection(O_not_raw, R_raw), C_pc1)
                dirs["O_clean_RC_raw"] = O_clean_RC
                dirs["O_clean_RC_unit"] = O_clean_RC / np.linalg.norm(O_clean_RC) if np.linalg.norm(O_clean_RC) > 1e-10 else np.zeros_like(O_clean_RC)
                log(f"  O_clean_RC: norm={np.linalg.norm(O_clean_RC):.4f}, ratio={np.linalg.norm(O_clean_RC)/max(np.linalg.norm(O_not_raw),1e-10):.4f}")
            
            # O_clean_RCA
            if R_raw is not None and C_pc1 is not None and A_raw_d is not None:
                O_clean_RCA = remove_projection(remove_projection(remove_projection(O_not_raw, R_raw), C_pc1), A_raw_d)
                dirs["O_clean_RCA_raw"] = O_clean_RCA
                dirs["O_clean_RCA_unit"] = O_clean_RCA / np.linalg.norm(O_clean_RCA) if np.linalg.norm(O_clean_RCA) > 1e-10 else np.zeros_like(O_clean_RCA)
                log(f"  O_clean_RCA: norm={np.linalg.norm(O_clean_RCA):.4f}, ratio={np.linalg.norm(O_clean_RCA)/max(np.linalg.norm(O_not_raw),1e-10):.4f}")
        
        # ---- Random direction baseline ----
        rng = np.random.default_rng(42)
        random_dir = rng.standard_normal(d_model)
        random_dir = random_dir / np.linalg.norm(random_dir)
        dirs["random_unit"] = random_dir
        
        # =====================================================================
        # PART 1: UNIFIED COSINE MATRIX
        # =====================================================================
        log(f"\n--- Unified Cosine Matrix (Layer {li}) ---")
        
        cos_names = ["R", "C_pc1", "O_not", "O_maybe", "O_must", "O_can", "O_never", "A"]
        cos_keys_raw = ["R_raw", "C_pc1", "O_not_raw", "O_maybe_raw", "O_must_raw", "O_can_raw", "O_never_raw", "A_raw"]
        cos_keys_unit = ["R_unit", "C_pc1_unit", "O_not_unit", "O_maybe_unit", "O_must_unit", "O_can_unit", "O_never_unit", "A_unit"]
        
        cos_matrix_raw = np.zeros((len(cos_names), len(cos_names)))
        cos_matrix_unit = np.zeros((len(cos_names), len(cos_names)))
        
        for i in range(len(cos_names)):
            for j in range(len(cos_names)):
                v_i = dirs.get(cos_keys_raw[i])
                v_j = dirs.get(cos_keys_raw[j])
                if v_i is not None and v_j is not None:
                    cos_matrix_raw[i, j] = cosine_sim(v_i, v_j)
                
                v_i_u = dirs.get(cos_keys_unit[i])
                v_j_u = dirs.get(cos_keys_unit[j])
                if v_i_u is not None and v_j_u is not None:
                    cos_matrix_unit[i, j] = cosine_sim(v_i_u, v_j_u)
        
        log(f"  Raw cosine matrix:")
        for i, name in enumerate(cos_names):
            row = " ".join(f"{cos_matrix_raw[i,j]:+.3f}" for j in range(len(cos_names)))
            log(f"    {name:>8}: {row}")
        
        # Key pairs
        log(f"  Key cosine pairs:")
        key_pairs = [("O_not", "R"), ("O_not", "C_pc1"), ("R", "C_pc1"), ("O_not", "A")]
        for n1, n2 in key_pairs:
            i1, i2 = cos_names.index(n1), cos_names.index(n2)
            log(f"    cos({n1}, {n2}) raw={cos_matrix_raw[i1,i2]:+.4f} unit={cos_matrix_unit[i1,i2]:+.4f}")
        
        # =====================================================================
        # PART 2: DIRECTION GAIN ANALYSIS
        # =====================================================================
        log(f"\n--- Direction Gain Analysis (Layer {li}) ---")
        
        # Use first baseline sentence
        base_sent = BASELINES[0]
        toks_base = tok.encode(base_sent, add_special_tokens=True)
        n_toks = len(toks_base)
        # Inject at last token position
        inject_pos = n_toks - 1
        
        gain_dirs = {
            "R_unit": dirs.get("R_unit"),
            "C_pc1_unit": dirs.get("C_pc1_unit"),
            "O_not_unit": dirs.get("O_not_unit"),
            "O_maybe_unit": dirs.get("O_maybe_unit"),
            "O_must_unit": dirs.get("O_must_unit"),
            "O_clean_R_unit": dirs.get("O_clean_R_unit"),
            "O_clean_C_unit": dirs.get("O_clean_C_unit"),
            "O_clean_RC_unit": dirs.get("O_clean_RC_unit"),
            "O_clean_RCA_unit": dirs.get("O_clean_RCA_unit"),
            "A_unit": dirs.get("A_unit"),
            "random_unit": dirs.get("random_unit"),
        }
        
        gains = {}
        for dname, dvec in gain_dirs.items():
            if dvec is not None and np.linalg.norm(dvec) > 1e-10:
                gain, _, _ = compute_direction_gain(model, tok, dvec, li, inject_pos, base_sent)
                gains[dname] = gain
                log(f"  Gain({dname}) = {gain:.4f}")
            else:
                gains[dname] = 0.0
                log(f"  Gain({dname}) = N/A (zero direction)")
        
        # Gain ratios
        random_gain = gains.get("random_unit", 1e-10)
        if random_gain > 1e-10:
            log(f"\n  Gain ratios (vs random):")
            for dname, gain in sorted(gains.items(), key=lambda x: -x[1]):
                ratio = gain / random_gain
                log(f"    {dname}: {ratio:.2f}x")
        
        # =====================================================================
        # PART 3: NORMALIZED CAUSAL TEST (UNIT + ALPHA SWEEP)
        # =====================================================================
        log(f"\n--- Normalized Causal Test (Layer {li}) ---")
        
        # Directions to test (all unit)
        test_dirs = {
            "R_unit": dirs.get("R_unit"),
            "C_pc1_unit": dirs.get("C_pc1_unit"),
            "O_not_unit": dirs.get("O_not_unit"),
            "O_clean_R_unit": dirs.get("O_clean_R_unit"),
            "O_clean_C_unit": dirs.get("O_clean_C_unit"),
            "O_clean_RC_unit": dirs.get("O_clean_RC_unit"),
            "O_clean_RCA_unit": dirs.get("O_clean_RCA_unit"),
            "A_unit": dirs.get("A_unit"),
            "random_unit": dirs.get("random_unit"),
        }
        
        # Alpha values to test
        alphas = [0.5, 1.0, 2.0, 5.0]
        
        # Baseline sentences (5)
        causal_results = {}
        
        for dname, dvec in test_dirs.items():
            if dvec is None or np.linalg.norm(dvec) < 1e-10:
                continue
            
            causal_results[dname] = {}
            
            for alpha in alphas:
                alpha_key = f"alpha_{alpha}"
                causal_results[dname][alpha_key] = {}
                
                patch_vec = dvec * alpha  # unit direction * alpha
                
                for bi, base_sent in enumerate(BASELINES):
                    toks_base = tok.encode(base_sent, add_special_tokens=True)
                    inject_pos = len(toks_base) - 1
                    
                    # Get baseline logits
                    input_device = next(model.parameters()).device
                    inputs = tok(base_sent, return_tensors="pt", truncation=True, max_length=64)
                    input_ids = inputs["input_ids"].to(input_device)
                    
                    with torch.no_grad():
                        base_out = model(input_ids=input_ids, output_hidden_states=False)
                        base_logits = base_out.logits.detach().cpu().float()
                    
                    # Patched logits
                    patched_logits = run_with_patched_hidden(model, tok, base_sent, li, inject_pos, patch_vec)
                    
                    if patched_logits is None:
                        causal_results[dname][alpha_key][f"baseline_{bi}"] = None
                        continue
                    
                    # Measure effect on key tokens
                    delta_logits = patched_logits[0, -1, :] - base_logits[0, -1, :]
                    
                    # P("not")
                    not_id = tok.encode("not", add_special_tokens=False)
                    # P("was") 
                    was_id = tok.encode(" was", add_special_tokens=False)
                    # P("very")
                    very_id = tok.encode(" very", add_special_tokens=False)
                    
                    effect = {}
                    if not_id:
                        delta_not = float(delta_logits[not_id[0]])
                        effect["delta_not"] = delta_not
                    if was_id:
                        delta_was = float(delta_logits[was_id[0]])
                        effect["delta_was"] = delta_was
                    
                    causal_results[dname][alpha_key][f"baseline_{bi}"] = effect
            
            # Compute mean effect across baselines
            for alpha_key in causal_results[dname]:
                effects_list = [v for v in causal_results[dname][alpha_key].values() if v is not None]
                if effects_list:
                    mean_not = np.mean([e.get("delta_not", 0) for e in effects_list])
                    std_not = np.std([e.get("delta_not", 0) for e in effects_list])
                    mean_was = np.mean([e.get("delta_was", 0) for e in effects_list])
                    causal_results[dname][alpha_key]["mean_delta_not"] = float(mean_not)
                    causal_results[dname][alpha_key]["std_delta_not"] = float(std_not)
                    causal_results[dname][alpha_key]["mean_delta_was"] = float(mean_was)
        
        # Print summary
        log(f"\n  Causal effect summary (delta_not):")
        log(f"  {'Direction':<20} " + " ".join(f"α={a:<5}" for a in alphas))
        for dname in test_dirs:
            if dname not in causal_results:
                continue
            row = f"  {dname:<20}"
            for alpha in alphas:
                alpha_key = f"alpha_{alpha}"
                mean_val = causal_results[dname].get(alpha_key, {}).get("mean_delta_not", 0)
                std_val = causal_results[dname].get(alpha_key, {}).get("std_delta_not", 0)
                row += f" {mean_val:+.3f}±{std_val:.2f}"
            log(row)
        
        # =====================================================================
        # PART 4: O-C CONFLICT RESOLUTION
        # =====================================================================
        log(f"\n--- O-C Conflict Resolution (Layer {li}) ---")
        
        # Phase 308 used C_raw (centered mean, small norm)
        # Phase 309 used C_pc1 (SVD first component)
        # Compare both
        
        C_raw_vec = dirs.get("C_raw")
        C_pc1_vec = dirs.get("C_pc1")
        O_not_vec = dirs.get("O_not_raw")
        
        if C_raw_vec is not None and C_pc1_vec is not None and O_not_vec is not None:
            cos_O_Craw = cosine_sim(O_not_vec, C_raw_vec)
            cos_O_Cpc1 = cosine_sim(O_not_vec, C_pc1_vec)
            cos_Craw_Cpc1 = cosine_sim(C_raw_vec, C_pc1_vec)
            
            C_raw_norm = np.linalg.norm(C_raw_vec)
            C_pc1_norm = np.linalg.norm(C_pc1_vec)
            
            log(f"  C_raw  norm={C_raw_norm:.6f}, cos(O,C_raw)={cos_O_Craw:+.4f}")
            log(f"  C_pc1  norm={C_pc1_norm:.6f}, cos(O,C_pc1)={cos_O_Cpc1:+.4f}")
            log(f"  cos(C_raw, C_pc1)={cos_Craw_Cpc1:+.4f}")
            
            # O projection onto C
            if C_raw_norm > 1e-10:
                proj_O_on_Craw = np.linalg.norm(project_onto(O_not_vec, C_raw_vec)) / np.linalg.norm(O_not_vec)
                log(f"  ||Proj_O(C_raw)|| / ||O|| = {proj_O_on_Craw:.4f}")
            
            proj_O_on_Cpc1 = np.linalg.norm(project_onto(O_not_vec, C_pc1_vec)) / np.linalg.norm(O_not_vec)
            log(f"  ||Proj_O(C_pc1)|| / ||O|| = {proj_O_on_Cpc1:.4f}")
            
            # Causal test: C_raw vs C_pc1
            log(f"\n  C_raw vs C_pc1 causal test (unit, alpha=2.0):")
            for c_name, c_vec in [("C_raw_unit", dirs.get("C_unit")), ("C_pc1_unit", dirs.get("C_pc1_unit"))]:
                if c_vec is not None and np.linalg.norm(c_vec) > 1e-10:
                    patch = c_vec * 2.0
                    for bi, base in enumerate(BASELINES[:3]):
                        toks_b = tok.encode(base, add_special_tokens=True)
                        pos_b = len(toks_b) - 1
                        patched = run_with_patched_hidden(model, tok, base, li, pos_b, patch)
                        if patched is not None:
                            inputs_b = tok(base, return_tensors="pt", truncation=True, max_length=64)
                            input_ids_b = inputs_b["input_ids"].to(next(model.parameters()).device)
                            with torch.no_grad():
                                base_logits = model(input_ids=input_ids_b, output_hidden_states=False).logits.detach().cpu().float()
                            delta = patched[0, -1, :] - base_logits[0, -1, :]
                            not_id = tok.encode("not", add_special_tokens=False)
                            if not_id:
                                log(f"    {c_name} baseline_{bi}: delta_not={float(delta[not_id[0]]):+.4f}")
        
        # =====================================================================
        # SAVE LAYER RESULTS
        # =====================================================================
        layer_result = {
            "layer": li,
            "cos_names": cos_names,
            "cos_matrix_raw": cos_matrix_raw.tolist(),
            "cos_matrix_unit": cos_matrix_unit.tolist(),
            "direction_norms": {},
            "gains": gains,
            "causal_results": {k: v for k, v in causal_results.items()},
        }
        
        # Save norms
        for key in ["R_raw", "C_raw", "C_pc1", "O_not_raw", "O_maybe_raw", "O_must_raw", 
                     "O_can_raw", "O_never_raw", "A_raw", "O_clean_R_raw", "O_clean_C_raw",
                     "O_clean_RC_raw", "O_clean_RCA_raw"]:
            v = dirs.get(key)
            if v is not None:
                layer_result["direction_norms"][key] = float(np.linalg.norm(v))
        
        all_results[li] = layer_result
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
    
    # =====================================================================
    # SAVE ALL RESULTS
    # =====================================================================
    final_result = {
        "model": model_name,
        "n_layers": nl,
        "d_model": d_model,
        "sample_layers": sample_layers,
        "baselines": BASELINES,
        "alphas": alphas,
        "layers": {str(k): v for k, v in all_results.items()},
    }
    
    out_path = RESULT_DIR / f"{model_name}_unified_norm_causal.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(final_result), f, indent=2, ensure_ascii=False)
    log(f"\nSaved to {out_path}")
    
    # Print cross-layer summary
    log(f"\n{'='*60}")
    log(f"CROSS-LAYER SUMMARY")
    log(f"{'='*60}")
    
    log(f"\n1. cos(O_not, C_pc1) across layers:")
    for li in sample_layers:
        lr = all_results.get(li, {})
        cos_mat = lr.get("cos_matrix_raw", [])
        names = lr.get("cos_names", [])
        if cos_mat and "O_not" in names and "C_pc1" in names:
            i_o = names.index("O_not")
            i_c = names.index("C_pc1")
            log(f"  L{li}: cos(O,C_pc1)={cos_mat[i_o][i_c]:+.4f}")
    
    log(f"\n2. Gain ratios (O_not_unit / random_unit) across layers:")
    for li in sample_layers:
        lr = all_results.get(li, {})
        gains = lr.get("gains", {})
        r_gain = gains.get("random_unit", 1e-10)
        o_gain = gains.get("O_not_unit", 0)
        ratio = o_gain / r_gain if r_gain > 1e-10 else 0
        oc_gain = gains.get("O_clean_RC_unit", 0)
        oc_ratio = oc_gain / r_gain if r_gain > 1e-10 else 0
        log(f"  L{li}: O_not={ratio:.2f}x, O_clean_RC={oc_ratio:.2f}x")
    
    log(f"\n3. Causal effect (alpha=2.0, unit direction, mean across baselines):")
    for li in sample_layers:
        lr = all_results.get(li, {})
        causal = lr.get("causal_results", {})
        log(f"  L{li}:")
        for dname in ["O_not_unit", "O_clean_RC_unit", "R_unit", "C_pc1_unit", "A_unit", "random_unit"]:
            if dname in causal:
                a2 = causal[dname].get("alpha_2.0", {})
                mean_not = a2.get("mean_delta_not", "N/A")
                if isinstance(mean_not, float):
                    log(f"    {dname}: delta_not={mean_not:+.4f}")
                else:
                    log(f"    {dname}: delta_not={mean_not}")
    
    # Release model
    release_model(model)
    log(f"Phase 310 complete for {model_name}")


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


if __name__ == "__main__":
    main()
