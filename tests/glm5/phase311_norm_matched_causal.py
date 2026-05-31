"""
Phase 311: Norm-Matched Causal Test + Multi-Target Token Analysis
=================================================================
Key improvements over Phase 310:
1. Norm-matched injection: use O_norm*0.1, R_norm*0.1 etc (proportional to natural scale)
2. Multi-target tokens: measure delta_happy, delta_sad, delta_very, delta_not etc
3. More baseline sentences (10)
4. Compare O_raw vs O_unit vs O_norm_matched

This tests whether DS7B's "O_clean_RC stronger than O_raw" holds under matched norm.

Usage:
  python tests/glm5/phase311_norm_matched_causal.py qwen3
  python tests/glm5/phase311_norm_matched_causal.py glm4
  python tests/glm5/phase311_norm_matched_causal.py deepseek7b
"""
import sys, os, gc, time, json, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase311_norm_matched_causal")
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
# STIMULUS
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

# Baseline sentences (10 diverse affirmatives)
BASELINES = [
    "the result was happy",
    "the outcome was bright",
    "the feeling was warm",
    "the plan was safe",
    "the method was clear",
    "the idea was smart",
    "the water was clean",
    "the sound was loud",
    "the path was smooth",
    "the room was quiet",
]

ANTONYM_PAIRS = [
    ("happy", "sad"), ("bright", "dark"), ("warm", "cold"), ("strong", "weak"),
    ("safe", "dangerous"), ("clean", "dirty"), ("rich", "poor"), ("fast", "slow"),
    ("smart", "stupid"), ("kind", "cruel"),
]

# Target tokens to measure
TARGET_TOKENS = ["not", " sad", " very", " happy", " but", " however", " never", " always"]


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
    return model, tok


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


def get_pos(tokenizer, sent, target):
    toks = tokenizer.encode(sent, add_special_tokens=True)
    dec = [tokenizer.decode([t]).strip().lower() for t in toks]
    return _find_token_pos(dec, target)


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


def measure_multi_target(patched_logits, base_logits, tokenizer, target_tokens):
    """Measure delta logit for multiple target tokens."""
    delta = patched_logits[0, -1, :] - base_logits[0, -1, :]
    results = {}
    for tok_str in target_tokens:
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        if ids:
            results[tok_str] = float(delta[ids[0]])
    return results


# =====================================================================
# MAIN
# =====================================================================
def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase311_{model_name}.txt"
    _log_file = str(log_file)
    log(f"Phase 311: Norm-Matched Causal Test -- {model_name}")
    
    model, tok = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    nl = info.n_layers
    d_model = info.d_model
    log(f"  n_layers={nl}, d_model={d_model}")
    
    # Only test 2-3 key layers to save time
    if nl >= 36:
        test_layers = [nl//2, 2*nl//3, nl-2]  # mid, late-mid, last
    else:
        test_layers = [nl//2, 2*nl//3]
    log(f"Test layers: {test_layers}")
    
    # ---- Build sentences and capture ----
    all_sents = []
    
    # Role sentences
    for token, (adj_s, verb_s, attr_s) in DUAL_TOKENS.items():
        all_sents.extend([adj_s, verb_s, attr_s])
    
    # Operator sentences
    for adj in ADJ_OPERANDS:
        for op, templ in [("affirm", f"the result was {adj}"),
                          ("not", f"the result was not {adj}"),
                          ("maybe", f"the result was maybe {adj}"),
                          ("must", f"the result must be {adj}"),
                          ("can", f"the result can be {adj}"),
                          ("never", f"the result was never {adj}")]:
            all_sents.append(templ)
    
    for verb in VERB_OPERANDS:
        for op, templ in [("affirm", f"they {verb} the plan"),
                          ("not", f"they do not {verb} the plan"),
                          ("maybe", f"they maybe {verb} the plan"),
                          ("must", f"they must {verb} the plan"),
                          ("can", f"they can {verb} the plan"),
                          ("never", f"they never {verb} the plan")]:
            all_sents.append(templ)
    
    for noun in NOUN_OPERANDS:
        all_sents.extend([f"that {noun} was available", f"that {noun} was not available"])
    
    # Antonym sentences
    for pos, neg in ANTONYM_PAIRS:
        all_sents.extend([f"the result was {pos}", f"the result was {neg}"])
    
    all_sents = list(dict.fromkeys(all_sents))
    log(f"Total sentences: {len(all_sents)}")
    
    # Capture
    log(f"Capturing {len(all_sents)} sentences...")
    t0 = time.time()
    captures = {}
    for i, sent in enumerate(all_sents):
        captures[sent] = _capture_single(model, tok, sent)
        if (i + 1) % 30 == 0:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1)
            eta = (len(all_sents) - i - 1) / rate
            log(f"  {i+1}/{len(all_sents)} ({rate:.1f}/s) ETA={eta:.0f}s GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
            gc.collect()
            torch.cuda.empty_cache()
    log(f"Done capturing in {time.time()-t0:.0f}s")
    
    # ---- Get baseline logits ----
    log("Computing baseline logits...")
    baseline_logits = {}
    for base in BASELINES:
        input_device = next(model.parameters()).device
        inputs = tok(base, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=False)
        baseline_logits[base] = out.logits.detach().cpu().float()
    
    # =====================================================================
    # EXTRACT DIRECTIONS AND RUN NORM-MATCHED CAUSAL TEST
    # =====================================================================
    all_results = {}
    
    for li in test_layers:
        log(f"\n{'='*60}")
        log(f"Layer {li}")
        log(f"{'='*60}")
        
        # ---- Extract directions ----
        dirs = {}
        
        # R(role)
        adj_vecs, verb_vecs = [], []
        for token, (adj_s, verb_s, _) in DUAL_TOKENS.items():
            h_a = captures.get(adj_s, {}).get(li)
            h_v = captures.get(verb_s, {}).get(li)
            p_a = get_pos(tok, adj_s, token)
            p_v = get_pos(tok, verb_s, token)
            if h_a is not None and h_v is not None and p_a is not None and p_v is not None:
                if p_a < h_a.shape[1] and p_v < h_v.shape[1]:
                    adj_vecs.append(h_a[0, p_a, :].numpy().copy())
                    verb_vecs.append(h_v[0, p_v, :].numpy().copy())
        
        if adj_vecs and verb_vecs:
            R_raw = np.mean(verb_vecs, axis=0) - np.mean(adj_vecs, axis=0)
            dirs["R"] = R_raw
            log(f"  R: {len(adj_vecs)} pairs, norm={np.linalg.norm(R_raw):.2f}")
        
        # C_pc1
        C_vecs = []
        for token, (adj_s, verb_s, attr_s) in DUAL_TOKENS.items():
            vecs = []
            for sent in [adj_s, verb_s, attr_s]:
                hs = captures.get(sent, {}).get(li)
                p = get_pos(tok, sent, token)
                if hs is not None and p is not None and p < hs.shape[1]:
                    vecs.append(hs[0, p, :].numpy().copy())
            if len(vecs) >= 2:
                mean_v = np.mean(vecs, axis=0)
                for v in vecs:
                    C_vecs.append(v - mean_v)
        
        if C_vecs:
            C_matrix = np.array(C_vecs)
            U, S, Vt = np.linalg.svd(C_matrix, full_matrices=False)
            dirs["C_pc1"] = Vt[0]
            log(f"  C_pc1: {len(C_vecs)} vecs, pc1_var={S[0]**2/np.sum(S**2):.1%}")
        
        # O(not)
        op_deltas = []
        for role, operands in [("adj", ADJ_OPERANDS), ("verb", VERB_OPERANDS)]:
            for operand in operands:
                if role == "adj":
                    aff_s = f"the result was {operand}"
                    not_s = f"the result was not {operand}"
                else:
                    aff_s = f"they {operand} the plan"
                    not_s = f"they do not {operand} the plan"
                
                h_a = captures.get(aff_s, {}).get(li)
                h_n = captures.get(not_s, {}).get(li)
                p_a = get_pos(tok, aff_s, operand)
                p_n = get_pos(tok, not_s, operand)
                
                if h_a is not None and h_n is not None and p_a is not None and p_n is not None:
                    if p_a < h_a.shape[1] and p_n < h_n.shape[1]:
                        op_deltas.append(h_n[0, p_n, :].numpy().copy() - h_a[0, p_a, :].numpy().copy())
        
        if op_deltas:
            O_raw = np.mean(op_deltas, axis=0)
            dirs["O_not"] = O_raw
            log(f"  O_not: {len(op_deltas)} deltas, norm={np.linalg.norm(O_raw):.2f}")
        
        # A(antonym)
        ant_deltas = []
        for pos, neg in ANTONYM_PAIRS:
            s_p = f"the result was {pos}"
            s_n = f"the result was {neg}"
            h_p = captures.get(s_p, {}).get(li)
            h_n = captures.get(s_n, {}).get(li)
            p_p = get_pos(tok, s_p, pos)
            p_n = get_pos(tok, s_n, neg)
            if h_p is not None and h_n is not None and p_p is not None and p_n is not None:
                if p_p < h_p.shape[1] and p_n < h_n.shape[1]:
                    ant_deltas.append(h_n[0, p_n, :].numpy().copy() - h_p[0, p_p, :].numpy().copy())
        
        if ant_deltas:
            A_raw = np.mean(ant_deltas, axis=0)
            dirs["A"] = A_raw
            log(f"  A: {len(ant_deltas)} pairs, norm={np.linalg.norm(A_raw):.2f}")
        
        # O_clean directions
        if "O_not" in dirs:
            if "R" in dirs:
                dirs["O_clean_R"] = remove_projection(dirs["O_not"], dirs["R"])
            if "C_pc1" in dirs:
                dirs["O_clean_C"] = remove_projection(dirs["O_not"], dirs["C_pc1"])
            if "R" in dirs and "C_pc1" in dirs:
                dirs["O_clean_RC"] = remove_projection(remove_projection(dirs["O_not"], dirs["R"]), dirs["C_pc1"])
            if "R" in dirs and "C_pc1" in dirs and "A" in dirs:
                dirs["O_clean_RCA"] = remove_projection(remove_projection(remove_projection(dirs["O_not"], dirs["R"]), dirs["C_pc1"]), dirs["A"])
        
        # Random direction
        rng = np.random.default_rng(42)
        random_dir = rng.standard_normal(d_model)
        random_dir = random_dir / np.linalg.norm(random_dir)
        dirs["random"] = random_dir
        
        # ---- Print direction stats ----
        log(f"\n  Direction norms and ratios:")
        O_norm = np.linalg.norm(dirs.get("O_not", np.zeros(1)))
        for dname in ["R", "C_pc1", "O_not", "O_clean_R", "O_clean_C", "O_clean_RC", "O_clean_RCA", "A"]:
            if dname in dirs:
                n = np.linalg.norm(dirs[dname])
                ratio = n / max(O_norm, 1e-10) if dname != "O_not" else 1.0
                log(f"    {dname}: norm={n:.2f}, ratio_to_O={ratio:.3f}")
        
        # Key cosine pairs
        log(f"\n  Key cosine pairs:")
        pairs = [("O_not", "R"), ("O_not", "C_pc1"), ("R", "C_pc1"), ("O_not", "A")]
        for n1, n2 in pairs:
            if n1 in dirs and n2 in dirs:
                log(f"    cos({n1}, {n2}) = {cosine_sim(dirs[n1], dirs[n2]):+.4f}")
        
        # ---- Norm-matched causal test ----
        log(f"\n  Norm-matched causal test:")
        
        # Injection scale: use a fraction of each direction's natural norm
        # Scale factors: 0.05, 0.1, 0.2 (fraction of natural norm)
        scale_factors = [0.05, 0.1, 0.2]
        
        test_dir_names = ["R", "C_pc1", "O_not", "O_clean_R", "O_clean_RC", "O_clean_RCA", "A", "random"]
        
        causal_data = {}
        
        for sf in scale_factors:
            sf_key = f"scale_{sf}"
            causal_data[sf_key] = {}
            log(f"\n  Scale factor = {sf}:")
            
            for dname in test_dir_names:
                if dname not in dirs:
                    continue
                
                d_vec = dirs[dname]
                d_norm = np.linalg.norm(d_vec)
                
                if d_norm < 1e-10:
                    continue
                
                # Norm-matched injection: direction * scale_factor * natural_norm
                # For random: use O_norm * scale_factor (same as O_not)
                if dname == "random":
                    patch_vec = d_vec * O_norm * sf
                else:
                    patch_vec = d_vec * sf  # d_vec already has natural norm
                
                patch_norm = np.linalg.norm(patch_vec)
                
                # Test on each baseline
                all_effects = {tok_str: [] for tok_str in TARGET_TOKENS}
                
                for base in BASELINES:
                    toks_b = tok.encode(base, add_special_tokens=True)
                    inject_pos = len(toks_b) - 1
                    
                    patched = run_with_patched_hidden(model, tok, base, li, inject_pos, patch_vec)
                    if patched is None:
                        continue
                    
                    effects = measure_multi_target(patched, baseline_logits[base], tok, TARGET_TOKENS)
                    for tok_str, delta in effects.items():
                        all_effects[tok_str].append(delta)
                
                # Compute means
                mean_effects = {}
                for tok_str, deltas in all_effects.items():
                    if deltas:
                        mean_effects[tok_str] = float(np.mean(deltas))
                
                causal_data[sf_key][dname] = {
                    "patch_norm": float(patch_norm),
                    "effects": mean_effects,
                }
                
                # Print key results
                dn = mean_effects.get("not", 0)
                ds = mean_effects.get(" sad", 0)
                dh = mean_effects.get(" happy", 0)
                log(f"    {dname:>15} (norm={patch_norm:.1f}): Δnot={dn:+.3f} Δsad={ds:+.3f} Δhappy={dh:+.3f}")
        
        # ---- Also test unit injection at same norm as O_not*0.1 ----
        log(f"\n  Unit direction injection (same norm as O_not*0.1):")
        unit_target_norm = O_norm * 0.1
        
        for dname in test_dir_names:
            if dname not in dirs:
                continue
            d_vec = dirs[dname]
            d_norm = np.linalg.norm(d_vec)
            if d_norm < 1e-10:
                continue
            
            # Unit direction scaled to same norm as O_not*0.1
            unit_dir = d_vec / d_norm * unit_target_norm
            patch_norm = np.linalg.norm(unit_dir)
            
            all_effects = {tok_str: [] for tok_str in TARGET_TOKENS}
            for base in BASELINES:
                toks_b = tok.encode(base, add_special_tokens=True)
                inject_pos = len(toks_b) - 1
                patched = run_with_patched_hidden(model, tok, base, li, inject_pos, unit_dir)
                if patched is None:
                    continue
                effects = measure_multi_target(patched, baseline_logits[base], tok, TARGET_TOKENS)
                for tok_str, delta in effects.items():
                    all_effects[tok_str].append(delta)
            
            mean_effects = {}
            for tok_str, deltas in all_effects.items():
                if deltas:
                    mean_effects[tok_str] = float(np.mean(deltas))
            
            dn = mean_effects.get("not", 0)
            ds = mean_effects.get(" sad", 0)
            log(f"    {dname:>15} unit(same_norm): Δnot={dn:+.3f} Δsad={ds:+.3f}")
        
        # Save layer results
        all_results[li] = {
            "direction_norms": {k: float(np.linalg.norm(v)) for k, v in dirs.items()},
            "cosines": {f"cos({n1},{n2})": float(cosine_sim(dirs[n1], dirs[n2])) 
                       for n1 in dirs for n2 in dirs if n1 < n2},
            "causal_data": causal_data,
        }
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # =====================================================================
    # SAVE
    # =====================================================================
    final = {
        "model": model_name,
        "n_layers": nl,
        "d_model": d_model,
        "test_layers": test_layers,
        "baselines": BASELINES,
        "target_tokens": TARGET_TOKENS,
        "scale_factors": scale_factors,
        "layers": {str(k): v for k, v in all_results.items()},
    }
    
    out_path = RESULT_DIR / f"{model_name}_norm_matched_causal.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(final), f, indent=2, ensure_ascii=False)
    log(f"\nSaved to {out_path}")
    
    # Print summary
    log(f"\n{'='*60}")
    log(f"SUMMARY: Norm-matched (scale=0.1) multi-target causal effects")
    log(f"{'='*60}")
    for li in test_layers:
        lr = all_results[li]
        cd = lr["causal_data"].get("scale_0.1", {})
        log(f"\nL{li}:")
        for dname in ["O_not", "O_clean_R", "O_clean_RC", "R", "C_pc1", "A", "random"]:
            if dname in cd:
                eff = cd[dname]["effects"]
                dn = eff.get("not", "N/A")
                ds = eff.get(" sad", "N/A")
                dh = eff.get(" happy", "N/A")
                pn = cd[dname]["patch_norm"]
                log(f"  {dname:>15} (norm={pn:.1f}): Δnot={dn:+.3f} Δsad={ds:+.3f} Δhappy={dh:+.3f}")
    
    release_model(model)
    log(f"Phase 311 complete for {model_name}")


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
