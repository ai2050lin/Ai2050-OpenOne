"""
Phase 309: O/R/C/S Complete Subspace Mapping
==============================================
Build a complete cross-function subspace map by computing:
1. All pairwise cosine similarities between O, R, C, S directions
2. Causal independence test: O_clean, R_clean, C_clean after removing projections
3. Subspace overlap: how much of each direction is unique vs shared

Key questions:
- Is O independent from R AND C? (Phase 307-308 suggest yes)
- Is R independent from C? (Phase 304 suggests partial overlap)
- What is the full subspace geometry?

Stimulus design:
- Role pairs: 16 dual-role tokens × 2 roles = 32 sentences
- Construction: 16 tokens × 3 frames = 48 sentences
- Operator: 20 adj + 10 verb × (affirm/not/maybe/must/can/should/never) = 245 sentences
- Scope: 6 infinitive pairs × 3 sentences = 18 sentences
- Total: ~343 unique sentences

Usage:
  python tests/glm5/phase309_subspace_map.py qwen3
  python tests/glm5/phase309_subspace_map.py glm4
  python tests/glm5/phase309_subspace_map.py deepseek7b
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

RESULT_DIR = Path("results/phase309_subspace_map")
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
# STIMULUS DEFINITIONS
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


def build_all_stimuli():
    """Build all stimuli for subspace mapping."""
    stimuli = []
    
    # ---- ROLE stimuli (adj vs verb for same token) ----
    for token, (adj_sent, verb_sent, attr_sent) in DUAL_TOKENS.items():
        stimuli.append({"sentence": adj_sent, "token": token, "role": "adj",
                        "frame": "predicative", "func": "role"})
        stimuli.append({"sentence": verb_sent, "token": token, "role": "verb",
                        "frame": "transitive", "func": "role"})
        stimuli.append({"sentence": attr_sent, "token": token, "role": "attributive",
                        "frame": "attributive", "func": "construction"})
    
    # ---- OPERATOR stimuli ----
    for adj in ADJ_OPERANDS:
        stimuli.append({"sentence": f"the result was {adj}", "operand": adj,
                        "operator": "affirm", "role": "adj", "func": "operator"})
        stimuli.append({"sentence": f"the result was not {adj}", "operand": adj,
                        "operator": "not", "role": "adj", "func": "operator"})
        stimuli.append({"sentence": f"the result was maybe {adj}", "operand": adj,
                        "operator": "maybe", "role": "adj", "func": "operator"})
        stimuli.append({"sentence": f"the result must be {adj}", "operand": adj,
                        "operator": "must", "role": "adj", "func": "operator"})
        stimuli.append({"sentence": f"the result can be {adj}", "operand": adj,
                        "operator": "can", "role": "adj", "func": "operator"})
        stimuli.append({"sentence": f"the result was never {adj}", "operand": adj,
                        "operator": "never", "role": "adj", "func": "operator"})
    
    for verb in VERB_OPERANDS:
        stimuli.append({"sentence": f"they {verb} the plan", "operand": verb,
                        "operator": "affirm", "role": "verb", "func": "operator"})
        stimuli.append({"sentence": f"they do not {verb} the plan", "operand": verb,
                        "operator": "not", "role": "verb", "func": "operator"})
        stimuli.append({"sentence": f"they maybe {verb} the plan", "operand": verb,
                        "operator": "maybe", "role": "verb", "func": "operator"})
        stimuli.append({"sentence": f"they must {verb} the plan", "operand": verb,
                        "operator": "must", "role": "verb", "func": "operator"})
        stimuli.append({"sentence": f"they can {verb} the plan", "operand": verb,
                        "operator": "can", "role": "verb", "func": "operator"})
        stimuli.append({"sentence": f"they never {verb} the plan", "operand": verb,
                        "operator": "never", "role": "verb", "func": "operator"})
    
    for noun in NOUN_OPERANDS:
        stimuli.append({"sentence": f"that {noun} was available", "operand": noun,
                        "operator": "affirm", "role": "noun", "func": "operator"})
        stimuli.append({"sentence": f"that {noun} was not available", "operand": noun,
                        "operator": "not", "role": "noun", "func": "operator"})
    
    # ---- SCOPE stimuli ----
    scope_pairs = [
        ("it was possible to leave", "it was not possible to leave", "it was possible not to leave"),
        ("it was necessary to wait", "it was not necessary to wait", "it was necessary not to wait"),
        ("it was easy to find", "it was not easy to find", "it was easy not to find"),
        ("it was safe to open", "it was not safe to open", "it was safe not to open"),
        ("it was hard to solve", "it was not hard to solve", "it was hard not to solve"),
        ("it was allowed to enter", "it was not allowed to enter", "it was allowed not to enter"),
    ]
    for base, narrow, wide in scope_pairs:
        stimuli.append({"sentence": base, "func": "scope", "scope": "baseline"})
        stimuli.append({"sentence": narrow, "func": "scope", "scope": "narrow"})
        stimuli.append({"sentence": wide, "func": "scope", "scope": "wide"})
    
    # ---- ANTonym stimuli ----
    antonym_pairs = [
        ("happy", "sad"), ("bright", "dark"), ("warm", "cold"), ("strong", "weak"),
        ("safe", "dangerous"), ("clean", "dirty"), ("rich", "poor"), ("fast", "slow"),
        ("smart", "stupid"), ("kind", "cruel"),
    ]
    for pos, neg in antonym_pairs:
        stimuli.append({"sentence": f"the result was {pos}", "operand": pos,
                        "operator": "affirm", "role": "adj", "func": "antonym_baseline"})
        stimuli.append({"sentence": f"the result was {neg}", "operand": neg,
                        "operator": "antonym", "role": "adj", "func": "antonym"})
    
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
            patched_logits = None
    handle.remove()
    return patched_logits


# =====================================================================
# MAIN
# =====================================================================
def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase309_{model_name}.txt"
    _log_file = str(log_file)
    log(f"Phase 309: O/R/C/S Complete Subspace Mapping -- {model_name}")
    
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
    stimuli = build_all_stimuli()
    log(f"Total stimuli: {len(stimuli)}")
    
    # Resolve positions
    for stim in stimuli:
        toks = tok.encode(stim["sentence"], add_special_tokens=True)
        dec = [tok.decode([t]).strip().lower() for t in toks]
        
        # Find target position
        target = stim.get("token") or stim.get("operand")
        if target:
            pos = _find_token_pos(dec, target)
            stim["target_pos"] = pos
        stim["n_tokens"] = len(toks)
    
    # Deduplicate sentences
    all_sentences = []
    sent_to_idx = {}
    for stim in stimuli:
        sent = stim["sentence"]
        if sent not in sent_to_idx:
            sent_to_idx[sent] = len(all_sentences)
            all_sentences.append(sent)
        stim["_idx"] = sent_to_idx[sent]
    
    log(f"Unique sentences: {len(all_sentences)}")
    
    # ---- Capture all sentences ----
    log(f"Capturing {len(all_sentences)} sentences...")
    t0 = time.time()
    captures = {}
    for i, sent in enumerate(all_sentences):
        captures[i] = _capture_single(model, tok, sent)
        if (i + 1) % 30 == 0:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1)
            eta = (len(all_sentences) - i - 1) / rate
            log(f"  {i+1}/{len(all_sentences)} ({rate:.1f}/s) ETA={eta:.0f}s GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
            gc.collect()
            torch.cuda.empty_cache()
    log(f"Done capturing in {time.time()-t0:.0f}s")
    
    # =====================================================================
    # EXTRACT FUNCTIONAL DIRECTIONS
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"EXTRACTING FUNCTIONAL DIRECTIONS")
    log(f"{'='*60}")
    
    functional_dirs = {}  # {layer: {func_name: direction}}
    
    for li in sample_layers:
        log(f"\n--- Layer {li} ---")
        dirs = {}
        
        # ---- R(role): adj → verb direction ----
        adj_vecs = []
        verb_vecs = []
        for token, (adj_sent, verb_sent, _) in DUAL_TOKENS.items():
            # Find adj stimulus
            adj_stim = next((s for s in stimuli if s.get("token") == token and s.get("role") == "adj" and s.get("func") == "role"), None)
            verb_stim = next((s for s in stimuli if s.get("token") == token and s.get("role") == "verb" and s.get("func") == "role"), None)
            
            if adj_stim and verb_stim:
                adj_idx, adj_pos = adj_stim["_idx"], adj_stim.get("target_pos")
                verb_idx, verb_pos = verb_stim["_idx"], verb_stim.get("target_pos")
                
                if adj_pos is not None and verb_pos is not None:
                    h_adj = captures[adj_idx].get(li)
                    h_verb = captures[verb_idx].get(li)
                    
                    if h_adj is not None and h_verb is not None:
                        if adj_pos < h_adj.shape[1] and verb_pos < h_verb.shape[1]:
                            adj_vecs.append(h_adj[0, adj_pos, :].numpy().copy())
                            verb_vecs.append(h_verb[0, verb_pos, :].numpy().copy())
        
        if adj_vecs and verb_vecs:
            R_dir = np.mean(verb_vecs, axis=0) - np.mean(adj_vecs, axis=0)
            dirs["R"] = R_dir
            log(f"  R(role): {len(adj_vecs)} pairs, norm={np.linalg.norm(R_dir):.2f}")
        
        # ---- C(construction): across-frame variation ----
        C_vecs = []
        for token, (adj_sent, verb_sent, attr_sent) in DUAL_TOKENS.items():
            token_stims = [s for s in stimuli if s.get("token") == token and s.get("func") in ["role", "construction"]]
            vecs = []
            for stim in token_stims:
                idx, pos = stim["_idx"], stim.get("target_pos")
                if pos is not None:
                    hs = captures[idx].get(li)
                    if hs is not None and pos < hs.shape[1]:
                        vecs.append(hs[0, pos, :].numpy().copy())
            
            if len(vecs) >= 2:
                mean_v = np.mean(vecs, axis=0)
                for v in vecs:
                    C_vecs.append(v - mean_v)
        
        if C_vecs:
            C_dir = np.mean(C_vecs, axis=0)
            dirs["C"] = C_dir
            # Also compute C_pc1
            C_matrix = np.array(C_vecs)
            from numpy.linalg import svd
            U, S, Vt = svd(C_matrix, full_matrices=False)
            C_pc1 = Vt[0]
            C_pc1_var = S[0]**2 / np.sum(S**2)
            dirs["C_pc1"] = C_pc1
            dirs["C_pc1_var"] = C_pc1_var
            log(f"  C(construction): {len(C_vecs)} vectors, norm={np.linalg.norm(C_dir):.2f}, pc1_var={C_pc1_var:.1%}")
        
        # ---- O(operator): not vs affirm direction ----
        op_dirs = {}
        for op_name in ["not", "maybe", "must", "can", "never"]:
            op_deltas = []
            for role in ["adj", "verb"]:
                # Find affirm and op stimuli for each operand
                operands = ADJ_OPERANDS if role == "adj" else VERB_OPERANDS
                
                for operand in operands:
                    affirm_stim = next((s for s in stimuli if s.get("operand") == operand 
                                       and s.get("operator") == "affirm" and s.get("role") == role), None)
                    op_stim = next((s for s in stimuli if s.get("operand") == operand 
                                    and s.get("operator") == op_name and s.get("role") == role), None)
                    
                    if affirm_stim and op_stim:
                        aff_idx, aff_pos = affirm_stim["_idx"], affirm_stim.get("target_pos")
                        op_idx, op_pos = op_stim["_idx"], op_stim.get("target_pos")
                        
                        if aff_pos is not None and op_pos is not None:
                            h_aff = captures[aff_idx].get(li)
                            h_op = captures[op_idx].get(li)
                            
                            if h_aff is not None and h_op is not None:
                                if aff_pos < h_aff.shape[1] and op_pos < h_op.shape[1]:
                                    delta = h_op[0, op_pos, :].numpy().copy() - h_aff[0, aff_pos, :].numpy().copy()
                                    op_deltas.append(delta)
            
            if op_deltas:
                avg_dir = np.mean(op_deltas, axis=0)
                op_dirs[op_name] = avg_dir
                dirs[f"O_{op_name}"] = avg_dir
                log(f"  O({op_name}): {len(op_deltas)} deltas, norm={np.linalg.norm(avg_dir):.2f}")
        
        # ---- S(scope): narrow vs wide scope ----
        scope_deltas_narrow = []
        scope_deltas_wide = []
        scope_pairs_data = [
            ("it was possible to leave", "it was not possible to leave", "it was possible not to leave"),
            ("it was necessary to wait", "it was not necessary to wait", "it was necessary not to wait"),
            ("it was easy to find", "it was not easy to find", "it was easy not to find"),
            ("it was safe to open", "it was not safe to open", "it was safe not to open"),
            ("it was hard to solve", "it was not hard to solve", "it was hard not to solve"),
            ("it was allowed to enter", "it was not allowed to enter", "it was allowed not to enter"),
        ]
        
        for base, narrow, wide in scope_pairs_data:
            base_stim = next((s for s in stimuli if s["sentence"] == base), None)
            narrow_stim = next((s for s in stimuli if s["sentence"] == narrow), None)
            wide_stim = next((s for s in stimuli if s["sentence"] == wide), None)
            
            if base_stim and narrow_stim and wide_stim:
                b_idx = base_stim["_idx"]
                n_idx = narrow_stim["_idx"]
                w_idx = wide_stim["_idx"]
                
                # Use last token position for scope (avoids position confound)
                h_b = captures[b_idx].get(li)
                h_n = captures[n_idx].get(li)
                h_w = captures[w_idx].get(li)
                
                if h_b is not None and h_n is not None and h_w is not None:
                    # Use last token
                    v_b = h_b[0, -1, :].numpy().copy()
                    v_n = h_n[0, -1, :].numpy().copy()
                    v_w = h_w[0, -1, :].numpy().copy()
                    
                    scope_deltas_narrow.append(v_n - v_b)
                    scope_deltas_wide.append(v_w - v_b)
        
        if scope_deltas_narrow:
            S_narrow = np.mean(scope_deltas_narrow, axis=0)
            S_wide = np.mean(scope_deltas_wide, axis=0)
            S_scope = S_wide - S_narrow  # scope-specific
            dirs["S_narrow"] = S_narrow
            dirs["S_wide"] = S_wide
            dirs["S_scope"] = S_scope
            log(f"  S(scope_narrow): {len(scope_deltas_narrow)} pairs, norm={np.linalg.norm(S_narrow):.2f}")
            log(f"  S(scope_wide):   {len(scope_deltas_wide)} pairs, norm={np.linalg.norm(S_wide):.2f}")
            log(f"  S(scope_diff):   norm={np.linalg.norm(S_scope):.2f}")
        
        # ---- A(antonym): positive vs negative ----
        ant_deltas = []
        for pos, neg in [("happy","sad"),("bright","dark"),("warm","cold"),("strong","weak"),
                         ("safe","dangerous"),("clean","dirty"),("rich","poor"),("fast","slow"),
                         ("smart","stupid"),("kind","cruel")]:
            pos_stim = next((s for s in stimuli if s.get("operand") == pos and s.get("func") == "antonym_baseline"), None)
            neg_stim = next((s for s in stimuli if s.get("operand") == neg and s.get("func") == "antonym"), None)
            
            if pos_stim and neg_stim:
                p_idx, p_pos = pos_stim["_idx"], pos_stim.get("target_pos")
                n_idx, n_pos = neg_stim["_idx"], neg_stim.get("target_pos")
                
                if p_pos is not None and n_pos is not None:
                    h_p = captures[p_idx].get(li)
                    h_n = captures[n_idx].get(li)
                    
                    if h_p is not None and h_n is not None:
                        if p_pos < h_p.shape[1] and n_pos < h_n.shape[1]:
                            delta = h_n[0, n_pos, :].numpy().copy() - h_p[0, p_pos, :].numpy().copy()
                            ant_deltas.append(delta)
        
        if ant_deltas:
            A_dir = np.mean(ant_deltas, axis=0)
            dirs["A_antonym"] = A_dir
            log(f"  A(antonym): {len(ant_deltas)} pairs, norm={np.linalg.norm(A_dir):.2f}")
        
        functional_dirs[li] = dirs
    
    # =====================================================================
    # CROSS-FUNCTION COSINE MATRIX
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"CROSS-FUNCTION COSINE MATRIX")
    log(f"{'='*60}")
    
    cos_matrices = {}
    
    for li in sample_layers:
        dirs = functional_dirs[li]
        func_names = [k for k in dirs.keys() if k != "C_pc1_var"]
        
        n = len(func_names)
        cos_matrix = np.zeros((n, n))
        
        for i, fn1 in enumerate(func_names):
            for j, fn2 in enumerate(func_names):
                d1 = dirs[fn1]
                d2 = dirs[fn2]
                if isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray):
                    cos_matrix[i, j] = cosine_sim(d1, d2)
        
        cos_matrices[li] = {"names": func_names, "matrix": cos_matrix}
        
        log(f"\nLayer {li}:")
        # Print formatted matrix
        header = "".join([f"{fn:>12s}" for fn in func_names])
        log(f"{'':>12s}{header}")
        for i, fn in enumerate(func_names):
            row = "".join([f"{cos_matrix[i,j]:>+12.3f}" for j in range(n)])
            log(f"{fn:>12s}{row}")
    
    # =====================================================================
    # INDEPENDENCE TEST: O_clean, R_clean, C_clean
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"INDEPENDENCE TEST: Remove projections and test causal effect")
    log(f"{'='*60}")
    
    # Readout tokens
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id = tok.encode("no", add_special_tokens=False)[0]
    not_id = tok.encode("not", add_special_tokens=False)[0]
    happy_id = tok.encode("happy", add_special_tokens=False)[0]
    sad_id = tok.encode("sad", add_special_tokens=False)[0]
    
    readout = {"yes": yes_id, "no": no_id, "not": not_id, "happy": happy_id, "sad": sad_id}
    
    independence_results = {}
    
    for li in sample_layers:
        dirs = functional_dirs[li]
        
        R = dirs.get("R")
        C = dirs.get("C")
        O_not = dirs.get("O_not")
        A = dirs.get("A_antonym")
        S_scope = dirs.get("S_scope")
        
        if R is None or C is None or O_not is None:
            continue
        
        log(f"\n--- Layer {li}: Independence test ---")
        
        # Compute clean directions
        # R_clean = R after removing O and C projections
        R_clean_OC = remove_projection(remove_projection(R, O_not), C)
        # O_clean = O after removing R and C projections
        O_clean_RC = remove_projection(remove_projection(O_not, R), C)
        # C_clean = C after removing R and O projections
        C_clean_RO = remove_projection(remove_projection(C, R), O_not)
        
        clean_dirs = {
            "R_raw": R,
            "O_raw": O_not,
            "C_raw": C,
            "R_clean_OC": R_clean_OC,
            "O_clean_RC": O_clean_RC,
            "C_clean_RO": C_clean_RO,
        }
        
        if A is not None:
            clean_dirs["A_raw"] = A
            O_clean_RCA = remove_projection(remove_projection(remove_projection(O_not, R), C), A)
            clean_dirs["O_clean_RCA"] = O_clean_RCA
        
        if S_scope is not None:
            clean_dirs["S_raw"] = S_scope
        
        # Norm ratios
        for name, d in clean_dirs.items():
            raw_name = name.replace("_clean_", "_raw_").split("_")[0] + "_raw"
            if "clean" in name:
                # Find corresponding raw
                parts = name.split("_clean_")
                raw_key = parts[0] + "_raw"
                if raw_key in clean_dirs:
                    ratio = np.linalg.norm(d) / max(np.linalg.norm(clean_dirs[raw_key]), 1e-10)
                    log(f"  {name}: norm={np.linalg.norm(d):.2f}, ratio={ratio:.3f}")
        
        # Causal test: inject clean directions into baseline sentence
        test_sent = "the result was happy"
        test_pos = _find_token_pos(
            [tok.decode([t]).strip().lower() for t in tok.encode(test_sent, add_special_tokens=True)],
            "happy"
        )
        
        if test_pos is None:
            continue
        
        # Baseline logits
        input_device = next(model.parameters()).device
        inputs = tok(test_sent, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=False)
            baseline_logits = out.logits[0, -1, :].float().cpu().numpy()
        
        scale = np.linalg.norm(O_not) * 0.1
        
        layer_causal = {}
        for name, d in clean_dirs.items():
            if not isinstance(d, np.ndarray):
                continue
            dn = np.linalg.norm(d)
            if dn < 1e-10:
                continue
            
            pvec = d / dn * scale
            patched = run_with_patched_hidden(model, tok, test_sent, li, test_pos, pvec)
            if patched is not None:
                patched_logits = patched[0, -1, :].float().numpy()
                effects = {}
                for tok_name, tok_id in readout.items():
                    effects[tok_name] = float(patched_logits[tok_id] - baseline_logits[tok_id])
                layer_causal[name] = effects
                log(f"  {name:20s}: not={effects.get('not',0):+.4f} happy={effects.get('happy',0):+.4f} sad={effects.get('sad',0):+.4f}")
            else:
                layer_causal[name] = None
        
        # Random baseline
        rng = np.random.RandomState(42)
        random_dir = rng.randn(d_model)
        random_dir = random_dir / np.linalg.norm(random_dir) * scale
        patched = run_with_patched_hidden(model, tok, test_sent, li, test_pos, random_dir)
        if patched is not None:
            patched_logits = patched[0, -1, :].float().numpy()
            effects = {}
            for tok_name, tok_id in readout.items():
                effects[tok_name] = float(patched_logits[tok_id] - baseline_logits[tok_id])
            layer_causal["random"] = effects
            log(f"  {'random':20s}: not={effects.get('not',0):+.4f} happy={effects.get('happy',0):+.4f} sad={effects.get('sad',0):+.4f}")
        
        independence_results[li] = layer_causal
    
    # =====================================================================
    # SUBSPACE OVERLAP QUANTIFICATION
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"SUBSPACE OVERLAP QUANTIFICATION")
    log(f"{'='*60}")
    
    overlap_results = {}
    
    for li in sample_layers:
        dirs = functional_dirs[li]
        
        R = dirs.get("R")
        C = dirs.get("C")
        O_not = dirs.get("O_not")
        A = dirs.get("A_antonym")
        S_scope = dirs.get("S_scope")
        
        if R is None or C is None or O_not is None:
            continue
        
        log(f"\n--- Layer {li}: Overlap ---")
        
        # How much of O is explained by each other direction?
        O_norm = np.linalg.norm(O_not)
        overlaps = {}
        
        for name, d in [("R", R), ("C", C), ("C_pc1", dirs.get("C_pc1")),
                         ("A", A) if A is not None else None,
                         ("S", S_scope) if S_scope is not None else None]:
            if d is None:
                continue
            proj = project_onto(O_not, d)
            overlap_ratio = np.linalg.norm(proj) / max(O_norm, 1e-10)
            overlaps[f"O_proj_{name}"] = float(overlap_ratio)
            log(f"  O projected onto {name}: {overlap_ratio:.3f} of O_norm")
        
        # How much of R is explained by each other direction?
        R_norm = np.linalg.norm(R)
        for name, d in [("C", C), ("O_not", O_not),
                         ("A", A) if A is not None else None]:
            if d is None:
                continue
            proj = project_onto(R, d)
            overlap_ratio = np.linalg.norm(proj) / max(R_norm, 1e-10)
            overlaps[f"R_proj_{name}"] = float(overlap_ratio)
            log(f"  R projected onto {name}: {overlap_ratio:.3f} of R_norm")
        
        overlap_results[li] = overlaps
    
    # =====================================================================
    # SAVE
    # =====================================================================
    log(f"\nSaving results...")
    
    # Prepare cos_matrices for serialization
    cos_matrices_clean = {}
    for li, data in cos_matrices.items():
        cos_matrices_clean[li] = {
            "names": data["names"],
            "matrix": data["matrix"].tolist(),
        }
    
    results = {
        "model": model_name,
        "n_layers": nl,
        "d_model": d_model,
        "sample_layers": sample_layers,
        "cos_matrices": make_serializable(cos_matrices_clean),
        "independence_results": make_serializable(independence_results),
        "overlap_results": make_serializable(overlap_results),
    }
    
    out_path = RESULT_DIR / f"{model_name}_subspace_map.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"Saved to {out_path}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Phase 309 complete for {model_name}")


if __name__ == "__main__":
    main()
