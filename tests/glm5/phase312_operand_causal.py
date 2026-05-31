"""
Phase 312: Operand-Position Injection + Probability-Based Causal Test
=====================================================================
Fix Phase 311's two main issues:
1. Inject at operand position (where "happy" is) not last token
2. Measure P(not) probability change, not just logit delta
3. Test both "the result was happy" and "the result was not happy"

This should give much cleaner causal results.

Usage:
  python tests/glm5/phase312_operand_causal.py qwen3
  python tests/glm5/phase312_operand_causal.py glm4
  python tests/glm5/phase312_operand_causal.py deepseek7b
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

RESULT_DIR = Path("results/phase312_operand_causal")
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

ANTONYM_PAIRS = [
    ("happy", "sad"), ("bright", "dark"), ("warm", "cold"), ("strong", "weak"),
    ("safe", "dangerous"), ("clean", "dirty"), ("rich", "poor"), ("fast", "slow"),
    ("smart", "stupid"), ("kind", "cruel"),
]

# Key baseline: we inject into "the result was happy" at the "happy" position
# and measure how the output distribution changes
# Also test into "the result was not happy" at the "happy" position


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name}...")
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
            break
        except Exception as e:
            log(f"  {attn_impl} failed: {str(e)[:80]}")
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    log(f"  Loaded. GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return model, tok


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def remove_projection(v, direction):
    dn = np.linalg.norm(direction)
    if dn < 1e-10:
        return np.zeros_like(v)
    d_hat = direction / dn
    return v - np.dot(v, d_hat) * d_hat


def _capture_single(model, tokenizer, sent, max_len=64):
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = {li: h.detach().cpu().float() for li, h in enumerate(out.hidden_states)}
    return hs


def get_pos(tokenizer, sent, target):
    toks = tokenizer.encode(sent, add_special_tokens=True)
    dec = [tokenizer.decode([t]).strip().lower() for t in toks]
    target_lower = target.lower()
    for i, t in enumerate(dec):
        if t == target_lower:
            return i
    for i, t in enumerate(dec):
        if target_lower in t or t in target_lower:
            return i
    return None


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


def get_token_probs(logits, tokenizer, token_strs, position=-1):
    """Get probabilities for target tokens at given position."""
    last_logits = logits[0, position, :]
    probs = torch.softmax(last_logits, dim=-1)
    result = {}
    for tok_str in token_strs:
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        if ids:
            result[tok_str] = float(probs[ids[0]])
    return result


def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase312_{model_name}.txt"
    _log_file = str(log_file)
    log(f"Phase 312: Operand-Position Causal Test -- {model_name}")
    
    model, tok = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    nl, d_model = info.n_layers, info.d_model
    log(f"  n_layers={nl}, d_model={d_model}")
    
    # Test layers
    test_layers = [nl//2, 2*nl//3] if nl < 32 else [nl//2, 2*nl//3, nl-2]
    log(f"Test layers: {test_layers}")
    
    # ---- Capture all direction sentences ----
    all_sents = []
    for token, (adj_s, verb_s, attr_s) in DUAL_TOKENS.items():
        all_sents.extend([adj_s, verb_s, attr_s])
    for adj in ADJ_OPERANDS:
        all_sents.extend([f"the result was {adj}", f"the result was not {adj}"])
    for verb in VERB_OPERANDS:
        all_sents.extend([f"they {verb} the plan", f"they do not {verb} the plan"])
    for noun in NOUN_OPERANDS:
        all_sents.extend([f"that {noun} was available", f"that {noun} was not available"])
    for pos, neg in ANTONYM_PAIRS:
        all_sents.extend([f"the result was {pos}", f"the result was {neg}"])
    
    all_sents = list(dict.fromkeys(all_sents))
    log(f"Total sentences: {len(all_sents)}")
    
    log("Capturing...")
    t0 = time.time()
    captures = {}
    for i, sent in enumerate(all_sents):
        captures[sent] = _capture_single(model, tok, sent)
        if (i+1) % 50 == 0:
            log(f"  {i+1}/{len(all_sents)} ({(i+1)/(time.time()-t0):.1f}/s) GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
            gc.collect(); torch.cuda.empty_cache()
    log(f"Done in {time.time()-t0:.0f}s")
    
    # ---- Extract directions at each layer ----
    # Use a set of target words for probability measurement
    PROB_TOKENS = ["not", " sad", " very", " happy", " extremely", " never", " also"]
    
    # Test sentences for causal test (inject at operand position)
    TEST_PAIRS = [
        # (affirmative, operand, negated)
        ("the result was happy", "happy", "the result was not happy"),
        ("the result was bright", "bright", "the result was not bright"),
        ("the result was warm", "warm", "the result was not warm"),
        ("the result was strong", "strong", "the result was not strong"),
        ("the result was safe", "safe", "the result was not safe"),
        ("the result was clean", "clean", "the result was not clean"),
        ("the result was smart", "smart", "the result was not smart"),
        ("the result was kind", "kind", "the result was not kind"),
        ("the result was fast", "fast", "the result was not fast"),
        ("the result was calm", "calm", "the result was not calm"),
    ]
    
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
                    adj_vecs.append(h_a[0, p_a].numpy().copy())
                    verb_vecs.append(h_v[0, p_v].numpy().copy())
        if adj_vecs:
            dirs["R"] = np.mean(verb_vecs, axis=0) - np.mean(adj_vecs, axis=0)
        
        # C_pc1
        C_vecs = []
        for token, (adj_s, verb_s, attr_s) in DUAL_TOKENS.items():
            vecs = []
            for sent in [adj_s, verb_s, attr_s]:
                hs = captures.get(sent, {}).get(li)
                p = get_pos(tok, sent, token)
                if hs is not None and p is not None and p < hs.shape[1]:
                    vecs.append(hs[0, p].numpy().copy())
            if len(vecs) >= 2:
                m = np.mean(vecs, axis=0)
                for v in vecs:
                    C_vecs.append(v - m)
        if C_vecs:
            U, S, Vt = np.linalg.svd(np.array(C_vecs), full_matrices=False)
            dirs["C_pc1"] = Vt[0]
        
        # O(not): mean(not_sent - affirm_sent) at operand position
        op_deltas = []
        for adj in ADJ_OPERANDS:
            aff_s = f"the result was {adj}"
            not_s = f"the result was not {adj}"
            h_a = captures.get(aff_s, {}).get(li)
            h_n = captures.get(not_s, {}).get(li)
            # Use operand position in both
            p_a = get_pos(tok, aff_s, adj)
            p_n = get_pos(tok, not_s, adj)
            if h_a is not None and h_n is not None and p_a is not None and p_n is not None:
                if p_a < h_a.shape[1] and p_n < h_n.shape[1]:
                    op_deltas.append(h_n[0, p_n].numpy().copy() - h_a[0, p_a].numpy().copy())
        
        for verb in VERB_OPERANDS:
            aff_s = f"they {verb} the plan"
            not_s = f"they do not {verb} the plan"
            h_a = captures.get(aff_s, {}).get(li)
            h_n = captures.get(not_s, {}).get(li)
            p_a = get_pos(tok, aff_s, verb)
            p_n = get_pos(tok, not_s, verb)
            if h_a is not None and h_n is not None and p_a is not None and p_n is not None:
                if p_a < h_a.shape[1] and p_n < h_n.shape[1]:
                    op_deltas.append(h_n[0, p_n].numpy().copy() - h_a[0, p_a].numpy().copy())
        
        if op_deltas:
            dirs["O_not"] = np.mean(op_deltas, axis=0)
            log(f"  O_not: {len(op_deltas)} deltas, norm={np.linalg.norm(dirs['O_not']):.2f}")
        
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
                    ant_deltas.append(h_n[0, p_n].numpy().copy() - h_p[0, p_p].numpy().copy())
        if ant_deltas:
            dirs["A"] = np.mean(ant_deltas, axis=0)
        
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
        
        rng = np.random.default_rng(42)
        dirs["random"] = rng.standard_normal(d_model)
        dirs["random"] = dirs["random"] / np.linalg.norm(dirs["random"])
        
        # Print direction stats
        log(f"  Direction stats:")
        for dname in ["R", "C_pc1", "O_not", "O_clean_R", "O_clean_RC", "O_clean_RCA", "A"]:
            if dname in dirs:
                n = np.linalg.norm(dirs[dname])
                ratio = n / max(np.linalg.norm(dirs.get("O_not", np.ones(1))), 1e-10)
                log(f"    {dname}: norm={n:.2f}, ratio_to_O={ratio:.3f}")
        
        # ---- Causal test: inject at OPERAND position ----
        log(f"\n  Causal test: inject at OPERAND position")
        
        test_dir_names = ["O_not", "O_clean_R", "O_clean_RC", "O_clean_RCA", "R", "A", "random"]
        O_norm = np.linalg.norm(dirs.get("O_not", np.ones(1)))
        scale = 0.1  # scale factor for norm-matched injection
        
        causal_results = {}
        
        for dname in test_dir_names:
            if dname not in dirs:
                continue
            
            d_vec = dirs[dname]
            d_norm = np.linalg.norm(d_vec)
            if d_norm < 1e-10:
                continue
            
            # Patch vector: direction * scale (norm-matched)
            patch_vec = d_vec * scale
            patch_norm = np.linalg.norm(patch_vec)
            
            # For random: use same norm as O_not * scale
            if dname == "random":
                patch_vec = d_vec * O_norm * scale
                patch_norm = np.linalg.norm(patch_vec)
            
            # Test on each affirmative sentence
            all_probs_affirm = {tok: [] for tok in PROB_TOKENS}
            all_probs_negated = {tok: [] for tok in PROB_TOKENS}
            
            for aff_sent, operand, neg_sent in TEST_PAIRS:
                # Find operand position in affirmative
                op_pos = get_pos(tok, aff_sent, operand)
                if op_pos is None:
                    continue
                
                # Baseline probabilities for affirmative
                input_device = next(model.parameters()).device
                inputs = tok(aff_sent, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                with torch.no_grad():
                    base_out = model(input_ids=input_ids, output_hidden_states=False)
                base_logits = base_out.logits.detach().cpu().float()
                
                # Measure at the position BEFORE operand (where "not" would be predicted)
                # This is position op_pos - 1 (the position predicting the next token)
                pred_pos = op_pos - 1  # position whose output predicts what comes next
                
                base_probs = get_token_probs(base_logits, tok, PROB_TOKENS, position=pred_pos)
                
                # Patched: inject at operand position
                patched_logits = run_with_patched_hidden(model, tok, aff_sent, li, op_pos, patch_vec)
                if patched_logits is None:
                    continue
                
                patched_probs = get_token_probs(patched_logits, tok, PROB_TOKENS, position=pred_pos)
                
                # Record probability changes
                for tok_str in PROB_TOKENS:
                    bp = base_probs.get(tok_str, 0)
                    pp = patched_probs.get(tok_str, 0)
                    all_probs_affirm[tok_str].append(pp - bp)
            
            # Compute mean probability changes
            mean_deltas = {}
            for tok_str in PROB_TOKENS:
                if all_probs_affirm[tok_str]:
                    mean_deltas[tok_str] = float(np.mean(all_probs_affirm[tok_str]))
            
            causal_results[dname] = {
                "patch_norm": float(patch_norm),
                "mean_delta_probs": mean_deltas,
            }
            
            dn = mean_deltas.get("not", 0)
            dh = mean_deltas.get(" happy", 0)
            ds = mean_deltas.get(" sad", 0)
            log(f"    {dname:>15} (norm={patch_norm:.1f}): ΔP(not)={dn:+.5f} ΔP(happy)={dh:+.5f} ΔP(sad)={ds:+.5f}")
        
        # ---- Also measure P(not) at the position between "was" and operand ----
        log(f"\n  P(not) summary (probability change at position before operand):")
        for dname in test_dir_names:
            if dname in causal_results:
                p_not = causal_results[dname]["mean_delta_probs"].get("not", 0)
                p_happy = causal_results[dname]["mean_delta_probs"].get(" happy", 0)
                pn = causal_results[dname]["patch_norm"]
                eff = p_not / max(pn, 1e-10)
                log(f"    {dname:>15}: ΔP(not)={p_not:+.5f} ΔP(happy)={p_happy:+.5f} eff={eff:+.7f}")
        
        all_results[li] = {
            "direction_norms": {k: float(np.linalg.norm(v)) for k, v in dirs.items()},
            "causal_results": causal_results,
        }
        
        gc.collect(); torch.cuda.empty_cache()
    
    # Save
    final = {
        "model": model_name,
        "n_layers": nl,
        "d_model": d_model,
        "test_layers": test_layers,
        "prob_tokens": PROB_TOKENS,
        "test_pairs_count": len(TEST_PAIRS),
        "layers": {str(k): v for k, v in all_results.items()},
    }
    
    out_path = RESULT_DIR / f"{model_name}_operand_causal.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(make_serializable(final), f, indent=2, ensure_ascii=False)
    log(f"\nSaved to {out_path}")
    
    # Print cross-layer summary
    log(f"\n{'='*60}")
    log(f"CROSS-LAYER SUMMARY: ΔP(not) at position before operand")
    log(f"{'='*60}")
    for li in test_layers:
        lr = all_results[li]
        log(f"\nL{li}:")
        for dname in ["O_not", "O_clean_R", "O_clean_RC", "O_clean_RCA", "R", "A", "random"]:
            if dname in lr["causal_results"]:
                cr = lr["causal_results"][dname]
                p_not = cr["mean_delta_probs"].get("not", 0)
                pn = cr["patch_norm"]
                eff = p_not / max(pn, 1e-10)
                log(f"  {dname:>15}: ΔP(not)={p_not:+.5f}  eff={eff:+.7f}")
    
    release_model(model)
    log(f"Phase 312 complete for {model_name}")


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
