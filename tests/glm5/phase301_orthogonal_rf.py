"""
Phase 301: Orthogonal R/F Decomposition + Causal Test
======================================================
Goal: Solve the R/F contamination problem identified in Phase 299.

Key hypothesis:
  DS7B's role+frame negative effect is due to R and F direction overlap.
  After Gram-Schmidt orthogonalization, R_clean + F_clean should be positive.

Decomposition:
  R_raw = original role increment (may contain F leakage)
  F_raw = original frame increment (may contain R leakage)
  R_clean = R_raw - Proj_F(R_raw)  (pure role, orthogonal to F)
  F_clean = F_raw - Proj_R(F_raw)  (pure frame, orthogonal to R)
  Interaction = R_raw - R_clean     (the shared/overlapping component)

Causal conditions:
  1. R_raw        — original role direction (with F contamination)
  2. F_raw        — original frame direction (with R contamination)
  3. R_clean      — pure role direction (orthogonal to F)
  4. F_clean      — pure frame direction (orthogonal to R)
  5. R_raw + F_raw — original bundle (double-counts shared component)
  6. R_clean + F_clean — orthogonal bundle (no overlap)
  7. Interaction  — shared component only
  8. R_clean + F_clean + Interaction — should ≈ R_raw + F_raw
  9. orthogonal_dir + correct_norm — pure norm test with neutral direction
 10. random controls (5 random directions × correct norm)

Key predictions:
  - If DS7B R_clean + F_clean > 0 while R_raw + F_raw < 0:
    Previous failure was from R/F contamination (SUPPORTED)
  - If DS7B R_clean + F_clean still < 0:
    DS7B truly uses token-specific role coding (REJECTED)

Layer coverage: 6 layers spanning early→deep (fixes deep layer gap)
  [nl//6, nl//3, nl//2, 2*nl//3, 5*nl//6, nl-2]

Usage:
  python tests/glm5/phase301_orthogonal_rf.py qwen3
  python tests/glm5/phase301_orthogonal_rf.py glm4
  python tests/glm5/phase301_orthogonal_rf.py deepseek7b
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

RESULT_DIR = Path("results/phase301_orthogonal_rf")
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
# STIMULUS — Same as Phase 299 for comparability
# =====================================================================
def build_stimuli():
    """Same stimuli as Phase 297/298/299 for consistency"""
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
    """Causal test pairs — same as Phase 299 for direct comparison"""
    test_pairs = [
        ("the window is open", "open", "adj", "adj_verb"), ("they open the window", "open", "verb", "adj_verb"),
        ("the market is open", "open", "adj", "adj_verb"), ("they open the market", "open", "verb", "adj_verb"),
        ("the field is clear", "clear", "adj", "adj_verb"), ("they clear the field", "clear", "verb", "adj_verb"),
        ("the meal is warm", "warm", "adj", "adj_verb"), ("they warm the meal", "warm", "verb", "adj_verb"),
        ("the shirt is clean", "clean", "adj", "adj_verb"), ("they clean the shirt", "clean", "verb", "adj_verb"),
        ("the feather is light", "light", "adj", "adj_noun"), ("the light is on", "light", "noun", "adj_noun"),
        ("the drink is cold", "cold", "adj", "adj_noun"), ("the cold is harsh", "cold", "noun", "adj_noun"),
        ("the fire is bright", "fire", "noun", "noun_verb"), ("they fire the employee", "fire", "verb", "noun_verb"),
        ("the record is famous", "record", "noun", "noun_verb"), ("they record the song", "record", "verb", "noun_verb"),
    ]
    stimuli = []
    for sent, target, role, rp in test_pairs:
        stimuli.append({"sentence": sent, "target_word": target, "token_label": target,
                        "role_label": role, "pair_label": "test", "role_pair": rp, "group": "causal_test"})
    return stimuli

# =====================================================================
# MODEL LOADING — BF16 + device_map="auto" + flash_attn priority
# =====================================================================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16 + device_map=auto + flash_attn)...")
    
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    model = None
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=attn_impl)
            log(f"  attn_implementation={attn_impl} succeeded")
            break
        except Exception as e:
            log(f"  attn_implementation={attn_impl} failed: {str(e)[:100]}")
    
    if model is None:
        raise RuntimeError(f"Failed to load {model_name} with any attention implementation")
    
    model.eval()
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Loaded. GPU={gpu_mem:.1f}GB")
    
    # Show layer distribution for device_map models
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        layer_devices = {}
        for k, v in dmap.items():
            if k.startswith('model.layers.'):
                lid = k.split('.')[2]
                if lid not in layer_devices:
                    layer_devices[lid] = str(v)
        gpu_layers = sum(1 for v in layer_devices.values() if 'cuda' in v)
        cpu_layers = sum(1 for v in layer_devices.values() if 'cpu' in v)
        log(f"  Layer distribution: {gpu_layers} GPU + {cpu_layers} CPU")
        # Show first and last 3 layers
        sorted_lids = sorted(layer_devices.keys(), key=int)
        for lid in sorted_lids[:3]:
            log(f"    L{lid}: {layer_devices[lid]}")
        if len(sorted_lids) > 6:
            log(f"    ...")
        for lid in sorted_lids[-3:]:
            log(f"    L{lid}: {layer_devices[lid]}")
    
    return model, tok

# =====================================================================
# CAPTURE & POSITION UTILITIES
# =====================================================================
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

def cosine_sim(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return float(np.dot(a, b) / (na * nb))

# =====================================================================
# ACTIVATION PATCHING
# =====================================================================
def run_with_patched_hidden(model, tokenizer, sent, layer_idx, pos, patch_vec, max_len=64):
    """Run model with a patch added to hidden state at (layer_idx, pos)."""
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(input_device)
    
    layers = get_layers(model)
    patched_logits = None
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
            log(f"  Patched forward failed at L{layer_idx}: {str(e)[:80]}")
    
    handle.remove()
    return patched_logits

# =====================================================================
# GRAM-SCHMIDT ORTHOGONALIZATION
# =====================================================================
def gram_schmidt_pair(R, F):
    """
    Orthogonalize R and F using modified Gram-Schmidt.
    
    Returns:
        R_clean: component of R orthogonal to F
        F_clean: component of F orthogonal to R  
        R_shared: component of R parallel to F (= R - R_clean)
        F_shared: component of F parallel to R (= F - F_clean)
        cos_RF: cosine similarity between R and F
    """
    R_norm = np.linalg.norm(R)
    F_norm = np.linalg.norm(F)
    
    if R_norm < 1e-10 or F_norm < 1e-10:
        return R.copy(), F.copy(), np.zeros_like(R), np.zeros_like(F), 0.0
    
    # Cosine similarity
    cos_RF = float(np.dot(R, F) / (R_norm * F_norm))
    
    # Project R onto F direction
    F_hat = F / F_norm
    R_proj_F = np.dot(R, F_hat) * F_hat  # projection of R onto F
    
    # R_clean = R - projection onto F
    R_clean = R - R_proj_F
    R_shared = R_proj_F  # shared component from R's perspective
    
    # Project F onto R direction
    R_hat = R / R_norm
    F_proj_R = np.dot(F, R_hat) * R_hat  # projection of F onto R
    
    # F_clean = F - projection onto R
    F_clean = F - F_proj_R
    F_shared = F_proj_R  # shared component from F's perspective
    
    return R_clean, F_clean, R_shared, F_shared, cos_RF

def generate_orthogonal_direction(R, F, d_model, rng_seed=42):
    """
    Generate a direction orthogonal to both R and F, for pure norm test.
    Uses QR decomposition approach.
    """
    # Stack R and F as basis vectors
    basis = np.stack([R / max(np.linalg.norm(R), 1e-10), 
                      F / max(np.linalg.norm(F), 1e-10)])
    
    # Generate random vector
    rng = np.random.RandomState(rng_seed)
    v = rng.randn(d_model)
    
    # Orthogonalize against basis using modified Gram-Schmidt
    for b in basis:
        v = v - np.dot(v, b) * b
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-10:
        # Very unlikely but handle it
        return rng.randn(d_model)
    return v / v_norm

# =====================================================================
# MAIN
# =====================================================================
def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase301_{model_name}.txt"
    _log_file = str(log_file)
    log(f"Phase 301: Orthogonal R/F Decomposition + Causal Test -- {model_name}")

    # ---- Load model ----
    model, tok = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    nl = info.n_layers; d_model = info.d_model
    log(f"  n_layers={nl}, d_model={d_model}, class={info.model_class}")
    
    # ---- Resolve positions ----
    sub_stimuli = resolve_positions(build_stimuli(), tok)
    causal_stimuli = resolve_positions(build_causal_stimuli(), tok)
    
    # Deduplicate sentences
    all_sentences = []; sent_to_idx = {}
    for s in sub_stimuli + causal_stimuli:
        sent = s["sentence"]
        if sent not in sent_to_idx:
            sent_to_idx[sent] = len(all_sentences); all_sentences.append(sent)
        s["_idx"] = sent_to_idx[sent]
    
    # ---- Capture all sentences ----
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
    
    # ---- Organize data ----
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
    
    # ---- Layer selection: deep coverage ----
    # 6 layers spanning early→deep to fix the deep-layer gap
    sample_layers = sorted(set([
        max(1, nl // 6), max(1, nl // 3), nl // 2,
        2 * nl // 3, 5 * nl // 6, nl - 2
    ]) & set(range(1, nl)))
    log(f"Sample layers (deep coverage): {sample_layers}")
    
    # =====================================================================
    # ORTHOGONAL R/F DECOMPOSITION + CAUSAL TEST
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"ORTHOGONAL R/F DECOMPOSITION + CAUSAL TEST")
    log(f"{'='*60}")
    
    results = {}
    
    for li in sample_layers:
        log(f"\n--- Layer {li} ---")
        
        # ---- Compute cell means ----
        cell_means = {}
        for (token, role, pair), entries in obs.items():
            vecs = []
            for (idx, pos) in entries:
                h = captures[idx]["hidden"].get(li)
                if h is not None and pos < h.shape[1]:
                    vecs.append(h[0, pos, :].numpy().copy())
            if vecs:
                cell_means[(token, role, pair)] = np.mean(vecs, axis=0)
        
        # ---- Compute R and F per token ----
        # R(token) = pair-averaged role increment: avg(h[role2]) - avg(h[role1])
        # F(token) = estimated frame increment for each causal pair
        token_R = {}
        token_role_means = {}
        
        for token in dual_tokens:
            roles_list = sorted(token_roles[token])
            if len(roles_list) != 2: continue
            r1, r2 = roles_list
            
            # Pair-averaged role means
            r1_means = [cell_means.get((token, r1, p)) for p in token_pairs[token]]
            r2_means = [cell_means.get((token, r2, p)) for p in token_pairs[token]]
            r1_means = [m for m in r1_means if m is not None]
            r2_means = [m for m in r2_means if m is not None]
            
            if r1_means and r2_means:
                m1 = np.mean(r1_means, axis=0)
                m2 = np.mean(r2_means, axis=0)
                token_R[token] = m2 - m1  # role increment
                token_role_means[(token, r1)] = m1
                token_role_means[(token, r2)] = m2
        
        # ---- LOO R direction ----
        token_R_loo = {}
        for token in dual_tokens:
            other_R = {t: v for t, v in token_R.items() if t != token}
            if other_R:
                token_R_loo[token] = np.mean(list(other_R.values()), axis=0)
            else:
                token_R_loo[token] = token_R.get(token, np.zeros(d_model))
        
        # ---- Causal test with orthogonal decomposition ----
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
            
            R_this = token_R.get(token)
            if R_this is None: continue
            
            # Full delta at this specific position
            v1 = captures[idx1]["hidden"][li][0, pos1, :].numpy().copy()
            v2 = captures[idx2]["hidden"][li][0, pos2, :].numpy().copy()
            full_delta = v2 - v1
            
            # F_estimated = full_delta - R_this
            F_this = full_delta - R_this
            
            # ---- Orthogonal decomposition ----
            R_clean, F_clean, R_shared, F_shared, cos_RF = gram_schmidt_pair(R_this, F_this)
            
            # Orthogonal direction for pure norm test
            ortho_dir = generate_orthogonal_direction(R_this, F_this, d_model, rng_seed=hash(token) % 10000)
            full_delta_norm = np.linalg.norm(full_delta)
            
            # LOO versions
            R_loo = token_R_loo[token]
            F_loo_estimated = full_delta - R_loo  # F estimated with LOO R
            R_loo_clean, F_loo_clean, _, _, cos_loo_RF = gram_schmidt_pair(R_loo, F_loo_estimated)
            
            # ---- Define all patch conditions ----
            conditions = {
                # Raw directions
                "R_raw": R_this,
                "F_raw": F_this,
                
                # Orthogonalized directions  
                "R_clean": R_clean,
                "F_clean": F_clean,
                
                # Bundles
                "R_raw+F_raw": R_this + F_this,  # = full_delta exactly
                "R_clean+F_clean": R_clean + F_clean,
                "R_clean+F_clean+interaction": R_clean + F_clean + R_shared,  # ≈ R_raw + F_clean
                
                # Interaction / shared component
                "interaction": R_shared,
                
                # LOO variants
                "R_loo_raw": R_loo,
                "R_loo_clean": R_loo_clean,
                "R_loo_clean+F_loo_clean": R_loo_clean + F_loo_clean,
                
                # Pure norm test: orthogonal direction + correct norm
                "ortho_dir+norm": ortho_dir * full_delta_norm,
            }
            
            key = f"{token}_{r1}->{r2}"
            layer_results[key] = {
                "token": token, "r1": r1, "r2": r2, "role_pair": token_rp.get(token, ""),
                "R_norm": float(np.linalg.norm(R_this)),
                "F_norm": float(np.linalg.norm(F_this)),
                "full_norm": float(full_delta_norm),
                "cos_RF": float(cos_RF),
                "R_clean_norm": float(np.linalg.norm(R_clean)),
                "F_clean_norm": float(np.linalg.norm(F_clean)),
                "interaction_norm": float(np.linalg.norm(R_shared)),
                "R_loo_norm": float(np.linalg.norm(R_loo)),
                "R_loo_clean_norm": float(np.linalg.norm(R_loo_clean)),
                "cos_loo_RF": float(cos_loo_RF),
            }
            
            # ---- Run causal tests ----
            for cond_name, patch_vec in conditions.items():
                pnorm = np.linalg.norm(patch_vec)
                if pnorm < 1e-10:
                    layer_results[key][f"{cond_name}_cos_shift"] = 0.0
                    layer_results[key][f"{cond_name}_norm"] = 0.0
                    continue
                
                patched_logits = run_with_patched_hidden(model, tok, s1["sentence"],
                                                          li, pos1, patch_vec)
                if patched_logits is not None:
                    p_logits = patched_logits[0, -1, :].numpy().copy()
                    cos_shift = cosine_sim(p_logits - logits1, target_shift)
                    layer_results[key][f"{cond_name}_cos_shift"] = float(cos_shift)
                    layer_results[key][f"{cond_name}_norm"] = float(pnorm)
                else:
                    layer_results[key][f"{cond_name}_cos_shift"] = None
                    layer_results[key][f"{cond_name}_norm"] = float(pnorm)
            
            # ---- Random controls (5 directions) ----
            rand_shifts = []
            for ri in range(5):
                rng2 = np.random.RandomState(ri * 100 + hash(token) % 100)
                rdir = rng2.randn(d_model); rdir = rdir / np.linalg.norm(rdir)
                rpatch = rdir * full_delta_norm
                plogits = run_with_patched_hidden(model, tok, s1["sentence"], li, pos1, rpatch)
                if plogits is not None:
                    pl = plogits[0, -1, :].numpy().copy()
                    rand_shifts.append(cosine_sim(pl - logits1, target_shift))
            layer_results[key]["avg_random_shift"] = float(np.mean(rand_shifts)) if rand_shifts else 0.0
            
            n_done = len(layer_results)
            if n_done % 2 == 0 or n_done == len(dual_test):
                log(f"  {n_done}/{len(dual_test)} test pairs done, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
                gc.collect(); torch.cuda.empty_cache()
        
        results[str(li)] = layer_results
        log(f"  Layer {li}: {len(layer_results)} test pairs completed")
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    output = {
        "model": model_name,
        "n_layers": nl,
        "d_model": d_model,
        "sample_layers": sample_layers,
        "dual_tokens": dual_tokens,
        "orthogonal_rf_causal": results,
    }
    
    out_path = RESULT_DIR / f"{model_name}_orthogonal_rf.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"\nSaved to {out_path}")
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"SUMMARY for {model_name}")
    log(f"{'='*60}")
    
    # Key comparison: R_raw vs R_clean, R_raw+F_raw vs R_clean+F_clean
    log(f"\n--- Core Comparison: Raw vs Orthogonal ---")
    for li_str, layer_res in results.items():
        if not layer_res: continue
        
        # R comparison
        R_raw_shifts = [v.get("R_raw_cos_shift") for v in layer_res.values()
                       if v.get("R_raw_cos_shift") is not None]
        R_clean_shifts = [v.get("R_clean_cos_shift") for v in layer_res.values()
                         if v.get("R_clean_cos_shift") is not None]
        
        # Bundle comparison (THE KEY TEST)
        Rraw_Fraw_shifts = [v.get("R_raw+F_raw_cos_shift") for v in layer_res.values()
                           if v.get("R_raw+F_raw_cos_shift") is not None]
        Rclean_Fclean_shifts = [v.get("R_clean+F_clean_cos_shift") for v in layer_res.values()
                               if v.get("R_clean+F_clean_cos_shift") is not None]
        
        # LOO clean bundle
        Rloo_Floo_clean = [v.get("R_loo_clean+F_loo_clean_cos_shift") for v in layer_res.values()
                          if v.get("R_loo_clean+F_loo_clean_cos_shift") is not None]
        
        # Interaction
        interaction_shifts = [v.get("interaction_cos_shift") for v in layer_res.values()
                            if v.get("interaction_cos_shift") is not None]
        
        # Orthogonal norm test
        ortho_shifts = [v.get("ortho_dir+norm_cos_shift") for v in layer_res.values()
                       if v.get("ortho_dir+norm_cos_shift") is not None]
        
        # Random
        rand_shifts = [v.get("avg_random_shift", 0) for v in layer_res.values()]
        
        avg_rand = np.mean(rand_shifts) if rand_shifts else 0
        
        log(f"\n  Layer {li_str}:")
        if R_raw_shifts:
            log(f"    R_raw:   avg={np.mean(R_raw_shifts):+.4f} pos={sum(1 for s in R_raw_shifts if s>0)}/{len(R_raw_shifts)}")
        if R_clean_shifts:
            log(f"    R_clean: avg={np.mean(R_clean_shifts):+.4f} pos={sum(1 for s in R_clean_shifts if s>0)}/{len(R_clean_shifts)}")
        if Rraw_Fraw_shifts:
            log(f"    R_raw+F_raw:     avg={np.mean(Rraw_Fraw_shifts):+.4f} pos={sum(1 for s in Rraw_Fraw_shifts if s>0)}/{len(Rraw_Fraw_shifts)}")
        if Rclean_Fclean_shifts:
            log(f"    R_clean+F_clean: avg={np.mean(Rclean_Fclean_shifts):+.4f} pos={sum(1 for s in Rclean_Fclean_shifts if s>0)}/{len(Rclean_Fclean_shifts)}")
        if Rloo_Floo_clean:
            log(f"    R_loo_clean+F_loo_clean: avg={np.mean(Rloo_Floo_clean):+.4f}")
        if interaction_shifts:
            log(f"    interaction:     avg={np.mean(interaction_shifts):+.4f}")
        if ortho_shifts:
            log(f"    ortho_dir+norm:  avg={np.mean(ortho_shifts):+.4f} (pure norm test)")
        log(f"    random baseline:  avg={avg_rand:+.4f}")
        
        # KEY DIAGNOSTIC: Does orthogonalization fix DS7B's negative bundle?
        if Rraw_Fraw_shifts and Rclean_Fclean_shifts:
            raw_avg = np.mean(Rraw_Fraw_shifts)
            clean_avg = np.mean(Rclean_Fclean_shifts)
            delta = clean_avg - raw_avg
            if raw_avg < 0 and clean_avg > 0:
                log(f"    *** ORTHOGONALIZATION FIXES NEGATIVE BUNDLE! {raw_avg:+.4f} → {clean_avg:+.4f} ***")
            elif raw_avg < 0 and clean_avg < 0:
                log(f"    *** ORTHOGONALIZATION DOES NOT FIX: {raw_avg:+.4f} → {clean_avg:+.4f} (token-specific coding) ***")
            elif raw_avg > 0 and clean_avg > raw_avg:
                log(f"    *** ORTHOGONALIZATION IMPROVES POSITIVE BUNDLE: {raw_avg:+.4f} → {clean_avg:+.4f} ***")
    
    # Per-token detail at mid layer
    mid_li = str(nl // 2)
    log(f"\n--- Per-Token Detail at Layer {mid_li} ---")
    if mid_li in results:
        for key, v in results[mid_li].items():
            token = v["token"]
            cos_RF = v.get("cos_RF", 0)
            R_raw_cs = v.get("R_raw_cos_shift", 0)
            R_clean_cs = v.get("R_clean_cos_shift", 0)
            raw_bundle = v.get("R_raw+F_raw_cos_shift", 0)
            clean_bundle = v.get("R_clean+F_clean_cos_shift", 0)
            ortho_norm = v.get("ortho_dir+norm_cos_shift", 0)
            log(f"    {key}: cos(R,F)={cos_RF:+.3f} | R_raw={R_raw_cs:+.3f} R_clean={R_clean_cs:+.3f} | "
                f"bundle_raw={raw_bundle:+.3f} bundle_clean={clean_bundle:+.3f} | ortho_norm={ortho_norm:+.3f}")
    
    # Cos(R,F) distribution
    log(f"\n--- Cos(R,F) Distribution Across All Tokens & Layers ---")
    all_cos = []
    for li_str, layer_res in results.items():
        for key, v in layer_res.items():
            cos_val = v.get("cos_RF")
            if cos_val is not None:
                all_cos.append((li_str, v["token"], v.get("role_pair", ""), cos_val))
    if all_cos:
        cos_vals = [c[3] for c in all_cos]
        log(f"  Mean cos(R,F) = {np.mean(cos_vals):+.4f}, std = {np.std(cos_vals):.4f}")
        log(f"  Range: [{min(cos_vals):+.4f}, {max(cos_vals):+.4f}]")
        # Show extreme cases
        by_cos = sorted(all_cos, key=lambda x: abs(x[3]), reverse=True)
        for li_s, tok, rp, cv in by_cos[:5]:
            log(f"    L{li_s} {tok} ({rp}): cos(R,F)={cv:+.4f}")
    
    release_model(model)
    log(f"Phase 301 complete for {model_name}")

if __name__ == "__main__":
    main()
