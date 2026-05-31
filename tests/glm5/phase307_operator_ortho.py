"""
Phase 307: Operator Orthogonality + Multi-Operator Causal Testing
=================================================================
Two key objectives:
1. O(not) vs R/C orthogonality: direct cos(O_not, R), cos(O_not, C) measurement
2. Multi-operator test: maybe/must/can/if/never — do different operators share O subspace?

Stimulus design:
- Negation: not + {20 adj, 10 verb, 8 noun} = 38 test pairs
- Modals: maybe/must/can/should + same operands = 152 test pairs
- Never: never + same operands = 38 test pairs
- Total: ~228 operator test pairs + 60 role pairs for R extraction

Usage:
  python tests/glm5/phase307_operator_ortho.py qwen3
  python tests/glm5/phase307_operator_ortho.py glm4
  python tests/glm5/phase307_operator_ortho.py deepseek7b
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

RESULT_DIR = Path("results/phase307_operator_ortho")
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
# OPERAND DEFINITIONS
# =====================================================================
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
    "person", "student", "teacher", "doctor", "system"
]

# =====================================================================
# OPERATOR SENTENCE TEMPLATES
# =====================================================================
def build_operator_sentences():
    """Build sentences with different operators."""
    stimuli = []
    
    # Adjectives with operators
    for adj in ADJ_OPERANDS:
        # Affirmative baseline
        stimuli.append({"sentence": f"the result was {adj}", "operand": adj, 
                        "operator": "affirm", "role": "adj", "condition": "baseline"})
        
        # Negation
        stimuli.append({"sentence": f"the result was not {adj}", "operand": adj,
                        "operator": "not", "role": "adj", "condition": "negated"})
        
        # Modals
        for modal in ["maybe", "must", "can", "should"]:
            stimuli.append({"sentence": f"the result {modal} be {adj}", "operand": adj,
                            "operator": modal, "role": "adj", "condition": "modal"})
        
        # Never
        stimuli.append({"sentence": f"the result was never {adj}", "operand": adj,
                        "operator": "never", "role": "adj", "condition": "negated"})
    
    # Verbs with operators
    for verb in VERB_OPERANDS:
        # Affirmative baseline
        stimuli.append({"sentence": f"they {verb} the plan", "operand": verb,
                        "operator": "affirm", "role": "verb", "condition": "baseline"})
        
        # Negation
        stimuli.append({"sentence": f"they do not {verb} the plan", "operand": verb,
                        "operator": "not", "role": "verb", "condition": "negated"})
        
        # Modals
        for modal in ["maybe", "must", "can", "should"]:
            stimuli.append({"sentence": f"they {modal} {verb} the plan", "operand": verb,
                            "operator": modal, "role": "verb", "condition": "modal"})
        
        # Never
        stimuli.append({"sentence": f"they never {verb} the plan", "operand": verb,
                        "operator": "never", "role": "verb", "condition": "negated"})
    
    # Nouns with operators (less natural but still informative)
    for noun in NOUN_OPERANDS:
        # Affirmative baseline
        stimuli.append({"sentence": f"that {noun} was available", "operand": noun,
                        "operator": "affirm", "role": "noun", "condition": "baseline"})
        
        # Negation
        stimuli.append({"sentence": f"that {noun} was not available", "operand": noun,
                        "operator": "not", "role": "noun", "condition": "negated"})
        
        # Modals
        for modal in ["maybe", "must", "can", "should"]:
            stimuli.append({"sentence": f"that {noun} {modal} be available", "operand": noun,
                            "operator": modal, "role": "noun", "condition": "modal"})
        
        # Never
        stimuli.append({"sentence": f"that {noun} was never available", "operand": noun,
                        "operator": "never", "role": "noun", "condition": "negated"})
    
    return stimuli


# =====================================================================
# ROLE EXTRACTION SENTENCES (for R direction)
# =====================================================================
def build_role_sentences():
    """Same-word-different-role sentences for R direction extraction."""
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
    
    stimuli = []
    for token, (adj_sent, verb_sent) in DUAL_TOKENS.items():
        stimuli.append({"sentence": adj_sent, "token": token, "role": "adj",
                        "operator": "role_baseline", "condition": "adj_role"})
        stimuli.append({"sentence": verb_sent, "token": token, "role": "verb",
                        "operator": "role_baseline", "condition": "verb_role"})
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
            log(f"  GPU layer IDs: {gpu_ids}")
    
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
    # Return last position as fallback for operator position
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
    log_file = TMP_DIR / f"phase307_{model_name}.txt"
    _log_file = str(log_file)
    log(f"Phase 307: Operator Orthogonality + Multi-Operator -- {model_name}")
    
    # ---- Load model ----
    model, tok = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    nl = info.n_layers
    d_model = info.d_model
    log(f"  n_layers={nl}, d_model={d_model}")
    
    layers = get_layers(model)
    for tl in [0, nl // 2, nl - 2]:
        try:
            _ = layers[tl]
            log(f"  Layer {tl}: accessible")
        except Exception as e:
            log(f"  Layer {tl}: FAILED - {e}")
    
    # ---- Build stimuli ----
    op_stimuli = build_operator_sentences()
    role_stimuli = build_role_sentences()
    log(f"  Operator stimuli: {len(op_stimuli)}, Role stimuli: {len(role_stimuli)}")
    
    # Resolve positions
    resolved_op = []
    for stim in op_stimuli:
        toks = tok.encode(stim["sentence"], add_special_tokens=True)
        dec = [tok.decode([t]).strip().lower() for t in toks]
        # Find operand position
        operand = stim["operand"].lower()
        pos = _find_token_pos(dec, operand)
        if pos is None:
            # For operator sentences, find the operator position
            op = stim["operator"]
            if op != "affirm":
                pos = _find_token_pos(dec, op)
        if pos is not None:
            stim["target_pos"] = pos
            stim["n_tokens"] = len(toks)
            resolved_op.append(stim)
    
    resolved_role = []
    for stim in role_stimuli:
        toks = tok.encode(stim["sentence"], add_special_tokens=True)
        dec = [tok.decode([t]).strip().lower() for t in toks]
        token = stim["token"].lower()
        pos = _find_token_pos(dec, token)
        if pos is not None:
            stim["target_pos"] = pos
            stim["n_tokens"] = len(toks)
            resolved_role.append(stim)
    
    log(f"  Resolved operator: {len(resolved_op)}, role: {len(resolved_role)}")
    
    # Deduplicate
    all_sentences = []
    sent_to_idx = {}
    for s in resolved_op + resolved_role:
        sent = s["sentence"]
        if sent not in sent_to_idx:
            sent_to_idx[sent] = len(all_sentences)
            all_sentences.append(sent)
        s["_idx"] = sent_to_idx[sent]
    log(f"  Unique sentences: {len(all_sentences)}")
    
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
    
    # Layer selection
    sample_layers = sorted(set([
        max(1, nl // 4), nl // 2, 3 * nl // 4, nl - 2
    ]) & set(range(1, nl)))
    log(f"Sample layers: {sample_layers}")
    
    # =====================================================================
    # PART A: OPERATOR DIRECTION EXTRACTION
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART A: OPERATOR DIRECTION EXTRACTION")
    log(f"{'='*60}")
    
    # Group operator stimuli by (operand, role)
    operand_groups = defaultdict(list)
    for stim in resolved_op:
        key = (stim["operand"], stim["role"])
        operand_groups[key].append(stim)
    
    operator_dirs = {}  # {layer: {operator: {role: direction}}}
    
    for li in sample_layers:
        log(f"\n--- Layer {li}: Operator directions ---")
        layer_dirs = defaultdict(lambda: defaultdict(list))
        
        for (operand, role), stim_list in operand_groups.items():
            # Find affirmative and operator versions
            affirm_stim = None
            op_stims = defaultdict(list)
            
            for stim in stim_list:
                if stim["operator"] == "affirm":
                    affirm_stim = stim
                else:
                    op_stims[stim["operator"]].append(stim)
            
            if affirm_stim is None:
                continue
            
            affirm_idx = affirm_stim.get("_idx")
            affirm_pos = affirm_stim.get("target_pos")
            if affirm_idx is None:
                continue
            
            affirm_hs = captures[affirm_idx].get(li)
            if affirm_hs is None or affirm_pos >= affirm_hs.shape[1]:
                continue
            
            h_affirm = affirm_hs[0, affirm_pos, :].numpy().copy()
            
            for op, op_stim_list in op_stims.items():
                for op_stim in op_stim_list:
                    op_idx = op_stim.get("_idx")
                    op_pos = op_stim.get("target_pos")
                    if op_idx is None:
                        continue
                    
                    op_hs = captures[op_idx].get(li)
                    if op_hs is None or op_pos >= op_hs.shape[1]:
                        continue
                    
                    h_op = op_hs[0, op_pos, :].numpy().copy()
                    op_dir = h_op - h_affirm  # operator direction
                    
                    layer_dirs[op][role].append(op_dir)
        
        # Average operator directions per role
        operator_dirs[li] = {}
        for op, role_dirs in layer_dirs.items():
            operator_dirs[li][op] = {}
            for role, dirs in role_dirs.items():
                avg_dir = np.mean(dirs, axis=0)
                operator_dirs[li][op][role] = avg_dir
                log(f"  {op}/{role}: {len(dirs)} operands, norm={np.linalg.norm(avg_dir):.2f}")
    
    # =====================================================================
    # PART B: R DIRECTION EXTRACTION
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART B: R DIRECTION EXTRACTION")
    log(f"{'='*60}")
    
    role_dirs = {}  # {layer: R_direction (avg adj→verb)}
    
    for li in sample_layers:
        adj_vecs = []
        verb_vecs = []
        
        for stim in resolved_role:
            idx = stim.get("_idx")
            pos = stim.get("target_pos")
            if idx is None:
                continue
            hs = captures[idx].get(li)
            if hs is None or pos >= hs.shape[1]:
                continue
            h = hs[0, pos, :].numpy().copy()
            
            if stim["role"] == "adj":
                adj_vecs.append(h)
            elif stim["role"] == "verb":
                verb_vecs.append(h)
        
        if adj_vecs and verb_vecs:
            adj_mean = np.mean(adj_vecs, axis=0)
            verb_mean = np.mean(verb_vecs, axis=0)
            R_dir = verb_mean - adj_mean
            role_dirs[li] = R_dir
            log(f"  L{li}: R_direction from {len(adj_vecs)} adj + {len(verb_vecs)} verb, norm={np.linalg.norm(R_dir):.2f}")
    
    # =====================================================================
    # PART C: O vs R ORTHOGONALITY
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART C: OPERATOR vs ROLE ORTHOGONALITY")
    log(f"{'='*60}")
    
    ortho_results = {}
    
    for li in sample_layers:
        if li not in operator_dirs or li not in role_dirs:
            continue
        
        R = role_dirs[li]
        layer_ortho = {}
        
        for op, role_data in operator_dirs[li].items():
            for role, O_dir in role_data.items():
                cos_O_R = cosine_sim(O_dir, R)
                layer_ortho[f"{op}_{role}"] = {
                    "operator": op, "role": role,
                    "cos_O_R": float(cos_O_R),
                    "O_norm": float(np.linalg.norm(O_dir)),
                }
                log(f"  L{li} {op}/{role}: cos(O,R)={cos_O_R:+.3f} O_norm={np.linalg.norm(O_dir):.2f}")
        
        ortho_results[li] = layer_ortho
    
    # =====================================================================
    # PART D: CROSS-OPERATOR SUBSPACE COMPARISON
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART D: CROSS-OPERATOR SUBSPACE COMPARISON")
    log(f"{'='*60}")
    
    cross_operator_results = {}
    
    for li in sample_layers:
        if li not in operator_dirs:
            continue
        
        op_dirs_layer = operator_dirs[li]
        # Compare each pair of operators
        operators = sorted(op_dirs_layer.keys())
        
        layer_cross = {}
        for i, op1 in enumerate(operators):
            for j, op2 in enumerate(operators):
                if i >= j:
                    continue
                
                # Average across roles
                cos_values = []
                for role in ["adj", "verb", "noun"]:
                    d1 = op_dirs_layer[op1].get(role)
                    d2 = op_dirs_layer[op2].get(role)
                    if d1 is not None and d2 is not None:
                        cos_values.append(cosine_sim(d1, d2))
                
                if cos_values:
                    avg_cos = np.mean(cos_values)
                    key = f"{op1}_vs_{op2}"
                    layer_cross[key] = {
                        "op1": op1, "op2": op2,
                        "avg_cos": float(avg_cos),
                        "per_role_cos": {r: float(cosine_sim(
                            op_dirs_layer[op1].get(r, np.zeros(d_model)),
                            op_dirs_layer[op2].get(r, np.zeros(d_model))
                        )) for r in ["adj", "verb", "noun"] 
                         if op_dirs_layer[op1].get(r) is not None and op_dirs_layer[op2].get(r) is not None},
                    }
                    log(f"  L{li} {op1} vs {op2}: avg_cos={avg_cos:+.3f}")
        
        cross_operator_results[li] = layer_cross
    
    # =====================================================================
    # PART E: OPERATOR LOO CONSISTENCY (Cross-operand sharing)
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART E: OPERATOR LOO CONSISTENCY")
    log(f"{'='*60}")
    
    loo_results = {}
    
    for li in sample_layers:
        if li not in operator_dirs:
            continue
        
        op_dirs_layer = operator_dirs[li]
        layer_loo = {}
        
        for op in op_dirs_layer:
            for role in ["adj", "verb", "noun"]:
                # Get per-operand operator directions
                operand_dirs_list = []
                for (operand, r), stim_list in operand_groups.items():
                    if r != role:
                        continue
                    
                    affirm_stim = None
                    op_stim = None
                    for stim in stim_list:
                        if stim["operator"] == "affirm":
                            affirm_stim = stim
                        elif stim["operator"] == op:
                            op_stim = stim
                    
                    if affirm_stim is None or op_stim is None:
                        continue
                    
                    a_idx = affirm_stim.get("_idx")
                    a_pos = affirm_stim.get("target_pos")
                    o_idx = op_stim.get("_idx")
                    o_pos = op_stim.get("target_pos")
                    
                    if any(x is None for x in [a_idx, a_pos, o_idx, o_pos]):
                        continue
                    
                    a_hs = captures[a_idx].get(li)
                    o_hs = captures[o_idx].get(li)
                    if a_hs is None or o_hs is None:
                        continue
                    if a_pos >= a_hs.shape[1] or o_pos >= o_hs.shape[1]:
                        continue
                    
                    a_h = a_hs[0, a_pos, :].numpy().copy()
                    o_h = o_hs[0, o_pos, :].numpy().copy()
                    operand_dirs_list.append(o_h - a_h)
                
                if len(operand_dirs_list) < 3:
                    continue
                
                # LOO: for each operand, compare its direction to the mean of all others
                loo_cos = []
                for i in range(len(operand_dirs_list)):
                    others = [d for j, d in enumerate(operand_dirs_list) if j != i]
                    loo_mean = np.mean(others, axis=0)
                    loo_cos.append(cosine_sim(operand_dirs_list[i], loo_mean))
                
                key = f"{op}_{role}"
                layer_loo[key] = {
                    "operator": op, "role": role,
                    "loo_consistency": float(np.mean(loo_cos)),
                    "loo_std": float(np.std(loo_cos)),
                    "n_operands": len(operand_dirs_list),
                }
                log(f"  L{li} {op}/{role}: LOO={np.mean(loo_cos):.3f}±{np.std(loo_cos):.3f} (n={len(operand_dirs_list)})")
        
        loo_results[li] = layer_loo
    
    # =====================================================================
    # PART F: OPERATOR CAUSAL TEST
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART F: OPERATOR CAUSAL TEST")
    log(f"{'='*60}")
    
    causal_results = {}
    
    # Select test operands: 5 adj + 5 verb + 3 noun
    test_operands = {
        "adj": ADJ_OPERANDS[:5],
        "verb": VERB_OPERANDS[:5],
        "noun": NOUN_OPERANDS[:3],
    }
    
    for li in sample_layers:
        if li not in operator_dirs:
            continue
        
        op_dirs_layer = operator_dirs[li]
        R = role_dirs.get(li)
        
        layer_causal = {}
        
        for role, operands in test_operands.items():
            for operand in operands:
                # Find affirmative sentence
                affirm_stim = None
                for stim in resolved_op:
                    if stim["operand"] == operand and stim["role"] == role and stim["operator"] == "affirm":
                        affirm_stim = stim
                        break
                
                if affirm_stim is None:
                    continue
                
                sent = affirm_stim["sentence"]
                pos = affirm_stim.get("target_pos")
                if pos is None:
                    continue
                
                # Get operand token IDs for readout
                target_ids = tok.encode(" " + operand, add_special_tokens=False)
                if not target_ids:
                    target_ids = tok.encode(operand, add_special_tokens=False)
                if not target_ids:
                    continue
                
                # Baseline logit
                base_logits = captures[affirm_stim["_idx"]].get(-1)  # Last layer logits
                # Need to recompute logits
                base_out = _capture_single(model, tok, sent)
                # Actually let's get from the last hidden state via lm_head
                # Simpler: use the logit difference
                base_logits = None
                input_device = next(model.parameters()).device
                inputs = tok(sent, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                with torch.no_grad():
                    out = model(input_ids=input_ids)
                    base_logits = out.logits[0, pos, :].float().cpu().numpy()
                
                base_logit_target = float(np.mean(base_logits[target_ids]))
                
                for op in ["not", "maybe", "must", "can", "never"]:
                    if op not in op_dirs_layer or role not in op_dirs_layer[op]:
                        continue
                    
                    O_dir = op_dirs_layer[op][role]
                    if np.linalg.norm(O_dir) < 1e-10:
                        continue
                    
                    # Scale to 10% of norm
                    inject_vec = O_dir / np.linalg.norm(O_dir) * np.linalg.norm(O_dir) * 0.1
                    
                    patched_logits = run_with_patched_hidden(model, tok, sent, li, pos, inject_vec)
                    if patched_logits is not None:
                        patched_logit_target = float(np.mean(patched_logits[0, pos, target_ids].numpy()))
                        causal_shift = patched_logit_target - base_logit_target
                    else:
                        causal_shift = 0.0
                    
                    key = f"{operand}_{role}_{op}"
                    layer_causal[key] = {
                        "operand": operand, "role": role, "operator": op,
                        "causal_shift": float(causal_shift),
                        "O_norm": float(np.linalg.norm(O_dir)),
                    }
        
        causal_results[li] = layer_causal
        
        # Summary
        if layer_causal:
            op_shifts = defaultdict(list)
            for val in layer_causal.values():
                op_shifts[val["operator"]].append(val["causal_shift"])
            for op, shifts in sorted(op_shifts.items()):
                log(f"  L{li} {op}: avg_causal={np.mean(shifts):+.4f} (n={len(shifts)})")
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    log(f"\nSaving results...")
    
    output = {
        "model": model_name,
        "n_layers": nl,
        "d_model": d_model,
        "sample_layers": sample_layers,
        "operators_tested": ["not", "maybe", "must", "can", "should", "never"],
        
        # Part A: Operator directions
        "operator_directions": make_serializable({
            li: {op: {role: dir.tolist() for role, dir in role_data.items()} 
                 for op, role_data in op_data.items()}
            for li, op_data in operator_dirs.items()
        }),
        
        # Part C: O vs R orthogonality
        "orthogonality_results": make_serializable(ortho_results),
        
        # Part D: Cross-operator comparison
        "cross_operator_results": make_serializable(cross_operator_results),
        
        # Part E: LOO consistency
        "loo_results": make_serializable(loo_results),
        
        # Part F: Causal tests
        "causal_results": make_serializable(causal_results),
    }
    
    out_path = RESULT_DIR / f"{model_name}_operator_ortho.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log(f"  Saved to {out_path}")
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"SUMMARY: Phase 307 -- {model_name}")
    log(f"{'='*60}")
    
    log(f"\n--- O vs R Orthogonality ---")
    for li, ortho in ortho_results.items():
        log(f"  Layer {li}:")
        for key, val in ortho.items():
            log(f"    {key}: cos(O,R)={val['cos_O_R']:+.3f}")
    
    log(f"\n--- Cross-Operator Similarity ---")
    for li, cross in cross_operator_results.items():
        log(f"  Layer {li}:")
        for key, val in cross.items():
            log(f"    {val['op1']} vs {val['op2']}: avg_cos={val['avg_cos']:+.3f}")
    
    log(f"\n--- LOO Consistency ---")
    for li, loo in loo_results.items():
        log(f"  Layer {li}:")
        for key, val in loo.items():
            log(f"    {val['operator']}/{val['role']}: LOO={val['loo_consistency']:.3f}")
    
    log(f"\n--- Causal Test ---")
    for li, causal in causal_results.items():
        op_shifts = defaultdict(list)
        for val in causal.values():
            op_shifts[val["operator"]].append(val["causal_shift"])
        log(f"  Layer {li}:")
        for op, shifts in sorted(op_shifts.items()):
            log(f"    {op}: {np.mean(shifts):+.4f}")
    
    # ---- Release model ----
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Phase 307 complete for {model_name}")


if __name__ == "__main__":
    main()
