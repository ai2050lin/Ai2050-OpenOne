"""
Phase 313: W_U Output Layer Readout Weight Analysis
====================================================
Core question: Where does the "differential amplification" happen?
- Is it W_U (output layer) that directly reads delta more strongly?
- Or is it intermediate layers that amplify delta before W_U?

This test:
1. Extracts W_U (lm_head weight matrix) [vocab_size, d_model]
2. Computes W_U @ v for each direction v
3. Measures: ||W_U @ delta|| / ||delta|| vs ||W_U @ shared|| / ||shared||
4. Also computes Jacobian-based gain via forward pass with/without injection
5. For DS7B: if W_U reads O_clean_R more strongly than O_not, the "amplification" is in W_U

Usage:
  python tests/glm5/phase313_WU_readout_analysis.py qwen3
  python tests/glm5/phase313_WU_readout_analysis.py glm4
  python tests/glm5/phase313_WU_readout_analysis.py deepseek7b
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
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model, get_W_U

RESULT_DIR = Path("results/phase313_WU_readout")
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
# STIMULUS SETS
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
    "cool":   ("the water is cool", "they cool the water", "the cool room was"),
    "smooth": ("the surface is smooth", "they smooth the surface", "the smooth surface was"),
    "cold":   ("the water is cold", "a cold hit them", "the cold wind blew"),
    "light":  ("the bag is light", "a light shone through", "the light switch was"),
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

FRAMES = {
    "adj_not": ["they are not {adj}", "they are very {adj}"],
    "verb_not": ["they do not {verb} it", "they {verb} it"],
    "noun_the": ["the {noun} was big", "they saw the {noun}"],
}

NOUN_OPERANDS = [
    "person", "student", "teacher", "doctor", "system",
    "car", "tree", "house", "city", "river"
]

ANTONYM_PAIRS = [
    ("happy", "sad"), ("bright", "dark"), ("warm", "cold"),
    ("strong", "weak"), ("safe", "dangerous"), ("clean", "dirty"),
    ("rich", "poor"), ("fast", "slow"), ("smart", "stupid"),
    ("kind", "cruel"), ("calm", "anxious"), ("free", "trapped"),
    ("clear", "confusing"), ("soft", "hard"), ("fresh", "stale"),
]

# Target tokens for W_U analysis
TARGET_TOKENS = ["not", " sad", " very", " happy", " but", " never", " always", " good"]
# Key vocab indices we'll focus on
FOCUS_VOCAB = ["not", "sad", "happy", "very", "never", "always", "good", "bad"]


def get_token_pos(tokenizer, text, target_word):
    """Get position of target_word in tokenized text"""
    tokens = tokenizer.encode(text, add_special_tokens=True)
    target_ids = tokenizer.encode(" " + target_word, add_special_tokens=False)
    if not target_ids:
        target_ids = tokenizer.encode(target_word, add_special_tokens=False)
    for i in range(len(tokens) - len(target_ids) + 1):
        if tokens[i:i+len(target_ids)] == target_ids:
            return i
    # Fallback: find by string match
    words = text.split()
    for i, w in enumerate(words):
        if target_word in w:
            # Approximate: count tokens up to this word
            prefix = " ".join(words[:i])
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
            return len(prefix_ids)
    return None


def extract_directions(model, tokenizer, device, layers_to_test, model_info):
    """Extract O/R/C/A directions at each test layer"""
    log("Extracting functional directions...")
    
    d_model = model_info.d_model
    all_directions = {}
    
    for li in layers_to_test:
        log(f"  Layer {li}: extracting directions...")
        layer = get_layers(model)[li]
        
        # Collect activations via hooks
        captures = {}
        
        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captures[name] = output[0].detach()
                else:
                    captures[name] = output.detach()
            return hook_fn
        
        handle = layer.register_forward_hook(make_hook(f"layer_{li}"))
        
        # === Direction extraction ===
        # 1. O(not): difference between not-ADJ and ADJ at last token
        O_deltas = []
        for adj in ADJ_OPERANDS:
            aff = f"they are very {adj}"
            neg = f"they are not {adj}"
            
            inp_a = tokenizer(aff, return_tensors="pt").to(device)
            inp_n = tokenizer(neg, return_tensors="pt").to(device)
            
            with torch.no_grad():
                captures.clear()
                model(**inp_a)
                h_a = captures.get(f"layer_{li}")
                
                captures.clear()
                model(**inp_n)
                h_n = captures.get(f"layer_{li}")
            
            if h_a is not None and h_n is not None:
                # Use last token position
                O_deltas.append((h_n[0, -1] - h_a[0, -1]).cpu().float().numpy())
        
        # 2. R(role): difference between noun-verb role and adj role
        R_deltas = []
        for word, (adj_s, verb_s, _) in DUAL_TOKENS.items():
            inp_a = tokenizer(adj_s, return_tensors="pt").to(device)
            inp_v = tokenizer(verb_s, return_tensors="pt").to(device)
            
            with torch.no_grad():
                captures.clear()
                model(**inp_a)
                h_a = captures.get(f"layer_{li}")
                
                captures.clear()
                model(**inp_v)
                h_v = captures.get(f"layer_{li}")
            
            if h_a is not None and h_v is not None:
                # Use the word position
                p_a = get_token_pos(tokenizer, adj_s, word)
                p_v = get_token_pos(tokenizer, verb_s, word)
                if p_a is not None and p_v is not None:
                    if p_a < h_a.shape[1] and p_v < h_v.shape[1]:
                        R_deltas.append((h_v[0, p_v] - h_a[0, p_a]).cpu().float().numpy())
        
        # 3. C(construction): frame differences
        C_deltas = []
        for noun in NOUN_OPERANDS[:8]:
            s1 = f"the {noun} was big"
            s2 = f"they saw the {noun}"
            
            inp1 = tokenizer(s1, return_tensors="pt").to(device)
            inp2 = tokenizer(s2, return_tensors="pt").to(device)
            
            with torch.no_grad():
                captures.clear()
                model(**inp1)
                h1 = captures.get(f"layer_{li}")
                
                captures.clear()
                model(**inp2)
                h2 = captures.get(f"layer_{li}")
            
            if h1 is not None and h2 is not None:
                # At noun position
                p1 = get_token_pos(tokenizer, s1, noun)
                p2 = get_token_pos(tokenizer, s2, noun)
                if p1 is not None and p2 is not None:
                    if p1 < h1.shape[1] and p2 < h2.shape[1]:
                        C_deltas.append((h1[0, p1] - h2[0, p2]).cpu().float().numpy())
        
        # 4. A(antonym): happy-sad differences
        A_deltas = []
        for pos_w, neg_w in ANTONYM_PAIRS[:10]:
            s1 = f"they are very {pos_w}"
            s2 = f"they are very {neg_w}"
            
            inp1 = tokenizer(s1, return_tensors="pt").to(device)
            inp2 = tokenizer(s2, return_tensors="pt").to(device)
            
            with torch.no_grad():
                captures.clear()
                model(**inp1)
                h1 = captures.get(f"layer_{li}")
                
                captures.clear()
                model(**inp2)
                h2 = captures.get(f"layer_{li}")
            
            if h1 is not None and h2 is not None:
                A_deltas.append((h1[0, -1] - h2[0, -1]).cpu().float().numpy())
        
        handle.remove()
        
        # === Compute direction vectors ===
        directions = {}
        
        if O_deltas:
            O_not = np.mean(O_deltas, axis=0)
            directions["O_not"] = O_not
        
        if R_deltas:
            R = np.mean(R_deltas, axis=0)
            directions["R"] = R
        
        if C_deltas:
            C_raw = np.mean(C_deltas, axis=0)
            directions["C_raw"] = C_raw
            
            # C_pc1: PCA on C_deltas
            C_stack = np.array(C_deltas)
            if len(C_deltas) >= 2:
                U, S, Vt = np.linalg.svd(C_stack - C_stack.mean(axis=0), full_matrices=False)
                C_pc1 = Vt[0]
                C_pc1_var = (S[0]**2) / (S**2).sum()
                directions["C_pc1"] = C_pc1
                directions["C_pc1_var"] = C_pc1_var
        
        if A_deltas:
            A = np.mean(A_deltas, axis=0)
            directions["A"] = A
        
        # Compute clean directions
        if "O_not" in directions and "R" in directions:
            O_not = directions["O_not"]
            R = directions["R"]
            R_norm_sq = np.dot(R, R)
            if R_norm_sq > 1e-12:
                O_clean_R = O_not - (np.dot(O_not, R) / R_norm_sq) * R
                directions["O_clean_R"] = O_clean_R
        
        if "O_not" in directions and "R" in directions and "C_pc1" in directions:
            O_not = directions["O_not"]
            R = directions["R"]
            C_pc1 = directions["C_pc1"]
            proj = np.zeros_like(O_not)
            for v in [R, C_pc1]:
                v_norm_sq = np.dot(v, v)
                if v_norm_sq > 1e-12:
                    proj += (np.dot(O_not, v) / v_norm_sq) * v
            O_clean_RC = O_not - proj
            directions["O_clean_RC"] = O_clean_RC
        
        # Compute shared component (PCA on all directions)
        all_vecs = []
        all_names = []
        for name in ["O_not", "R", "C_pc1", "A"]:
            if name in directions:
                all_vecs.append(directions[name])
                all_names.append(name)
        
        if len(all_vecs) >= 2:
            stack = np.array(all_vecs)
            U, S, Vt = np.linalg.svd(stack, full_matrices=False)
            shared_pc1 = Vt[0]
            shared_pc1_var = (S[0]**2) / (S**2).sum()
            directions["shared_pc1"] = shared_pc1
            directions["shared_pc1_var"] = shared_pc1_var
            
            # Residual of each direction from shared
            for i, name in enumerate(all_names):
                proj_on_shared = np.dot(all_vecs[i], shared_pc1) * shared_pc1
                delta = all_vecs[i] - proj_on_shared
                directions[f"{name}_delta_from_shared"] = delta
        
        all_directions[li] = directions
        log(f"  Layer {li}: {len(directions)} directions extracted")
    
    return all_directions


def compute_WU_readout(W_U, directions, target_token_ids, tokenizer):
    """
    Compute W_U readout analysis for each direction.
    
    For direction v:
    - W_U @ v gives logit changes for all vocab tokens
    - ||W_U @ v|| / ||v|| = output gain (how much output changes per unit input)
    - W_U @ v[target_token] = specific logit change for target tokens
    
    Key comparison:
    - output_gain(O_not) vs output_gain(O_clean_R) vs output_gain(R) vs output_gain(random)
    - If output_gain(O_clean_R) >> output_gain(O_not), W_U directly amplifies delta
    """
    results = {}
    
    for li, dirs in directions.items():
        li_results = {"direction_norms": {}, "WU_gains": {}, "target_logits": {}}
        
        # Random baseline: average gain over random unit vectors
        n_random = 100
        d_model = W_U.shape[1]
        random_gains = []
        for _ in range(n_random):
            rv = np.random.randn(d_model)
            rv = rv / np.linalg.norm(rv)
            WU_v = W_U @ rv
            random_gains.append(np.linalg.norm(WU_v))
        avg_random_gain = np.mean(random_gains)
        li_results["random_gain_avg"] = float(avg_random_gain)
        li_results["random_gain_std"] = float(np.std(random_gains))
        
        for name, vec in dirs.items():
            if not isinstance(vec, np.ndarray) or vec.ndim != 1:
                continue
            if name.endswith("_var"):
                continue
            
            norm = np.linalg.norm(vec)
            li_results["direction_norms"][name] = float(norm)
            
            if norm < 1e-10:
                continue
            
            # W_U @ v
            WU_v = W_U @ vec
            
            # Overall gain: ||W_U @ v|| / ||v||
            gain = np.linalg.norm(WU_v) / norm
            li_results["WU_gains"][name] = float(gain)
            
            # Gain relative to random
            li_results[f"WU_gain_ratio_{name}"] = float(gain / avg_random_gain) if avg_random_gain > 0 else 0
            
            # Target token logit changes: (W_U @ v)[target_id] / ||v||
            target_logits = {}
            for tname, tid in target_token_ids.items():
                if tid < WU_v.shape[0]:
                    target_logits[tname] = float(WU_v[tid] / norm)
            li_results["target_logits"][name] = target_logits
        
        # Key comparisons
        comparisons = {}
        for metric_name, metric_key in [
            ("gain", "WU_gains"),
            ("target_not", None),  # special handling
        ]:
            pass
        
        # Direct comparisons
        for pair_name, (n1, n2) in [
            ("O_not_vs_O_clean_R", ("O_not", "O_clean_R")),
            ("O_not_vs_R", ("O_not", "R")),
            ("O_not_vs_C_pc1", ("O_not", "C_pc1")),
            ("O_clean_R_vs_R", ("O_clean_R", "R")),
            ("delta_O_not_vs_O_not", ("O_not_delta_from_shared", "O_not")),
        ]:
            if n1 in li_results.get("WU_gains", {}) and n2 in li_results.get("WU_gains", {}):
                g1 = li_results["WU_gains"][n1]
                g2 = li_results["WU_gains"][n2]
                comparisons[f"{pair_name}_gain_ratio"] = float(g1/g2) if g2 > 0 else 0
            
            if n1 in li_results.get("target_logits", {}) and n2 in li_results.get("target_logits", {}):
                for tname in target_token_ids:
                    v1 = li_results["target_logits"][n1].get(tname, 0)
                    v2 = li_results["target_logits"][n2].get(tname, 0)
                    comparisons[f"{pair_name}_{tname}_ratio"] = float(v1/v2) if abs(v2) > 1e-10 else 0
                    comparisons[f"{pair_name}_{tname}_abs_diff"] = float(v1 - v2)
        
        li_results["comparisons"] = comparisons
        results[li] = li_results
    
    return results


def compute_jacobian_gain(model, tokenizer, device, directions, layers_to_test, model_info):
    """
    Compute empirical Jacobian-based gain via forward pass.
    
    For direction v at layer l:
    - Run baseline forward pass -> get output logits
    - Run with injection: h_l += epsilon * v -> get output logits
    - Jacobian_gain = ||delta_logits|| / (epsilon * ||v||)
    
    This captures ALL intermediate layer processing, not just W_U.
    If Jacobian_gain(O_clean_R) >> W_U_gain(O_clean_R),
    then intermediate layers amplify the delta.
    """
    results = {}
    epsilon = 0.1  # Small injection scale
    
    for li in layers_to_test:
        li_results = {}
        dirs = directions.get(li, {})
        layer = get_layers(model)[li]
        
        # Baseline sentence
        baseline = "they are very happy"
        inp = tokenizer(baseline, return_tensors="pt").to(device)
        
        # Capture hook for injection
        injected_logits = {}
        
        def make_inject_hook(dir_vec, eps, name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    h[0, -1] += eps * torch.tensor(dir_vec, dtype=h.dtype, device=h.device)
                    return (h,) + output[1:]
                else:
                    h = output.clone()
                    h[0, -1] += eps * torch.tensor(dir_vec, dtype=h.dtype, device=h.device)
                    return h
            return hook_fn
        
        # Baseline logits
        with torch.no_grad():
            base_out = model(**inp)
            base_logits = base_out.logits[0, -1].detach().cpu().float().numpy()
        
        # For each direction, inject and measure output change
        for name, vec in dirs.items():
            if not isinstance(vec, np.ndarray) or vec.ndim != 1:
                continue
            if name.endswith("_var"):
                continue
            norm = np.linalg.norm(vec)
            if norm < 1e-10:
                continue
            
            # Unit direction injection
            unit_vec = vec / norm
            
            handle = layer.register_forward_hook(make_inject_hook(unit_vec, epsilon, name))
            
            with torch.no_grad():
                inj_out = model(**inp)
                inj_logits = inj_out.logits[0, -1].detach().cpu().float().numpy()
            
            handle.remove()
            
            delta_logits = inj_logits - base_logits
            jacobian_gain = np.linalg.norm(delta_logits) / epsilon
            
            li_results[name] = {
                "jacobian_gain": float(jacobian_gain),
                "delta_not": float(delta_logits[tokenizer.encode(" not", add_special_tokens=False)[0]] 
                                   if tokenizer.encode(" not", add_special_tokens=False) else 0),
            }
        
        # Random baseline
        random_gains = []
        d_model = model_info.d_model
        for _ in range(50):
            rv = np.random.randn(d_model)
            rv = rv / np.linalg.norm(rv)
            handle = layer.register_forward_hook(make_inject_hook(rv, epsilon, "random"))
            with torch.no_grad():
                inj_out = model(**inp)
                inj_logits = inj_out.logits[0, -1].detach().cpu().float().numpy()
            handle.remove()
            delta_logits = inj_logits - base_logits
            random_gains.append(np.linalg.norm(delta_logits) / epsilon)
        
        li_results["random"] = {
            "jacobian_gain_avg": float(np.mean(random_gains)),
            "jacobian_gain_std": float(np.std(random_gains)),
        }
        
        results[li] = li_results
        log(f"  Layer {li}: Jacobian gains computed for {len(li_results)} directions")
    
    return results


def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase313_{model_name}.log")
    
    log(f"=== Phase 313: W_U Readout Analysis for {model_name} ===")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    
    log(f"Loading {model_name} (bf16 + device_map=auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()
    device = next(model.parameters()).device
    
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"Model loaded: {type(model).__name__}, device={device}, GPU={gpu_mem:.2f}GB")
    
    info = get_model_info(model, model_name)
    log(f"Model info: n_layers={info.n_layers}, d_model={info.d_model}, vocab={info.vocab_size}")
    
    # Select layers to test
    n_layers = info.n_layers
    # Early, mid, late
    layers_to_test = []
    if n_layers >= 36:
        layers_to_test = [6, 12, 18, 24, 30, n_layers-2]
    elif n_layers >= 24:
        layers_to_test = [4, 8, 12, 16, 20, n_layers-2]
    else:
        layers_to_test = [4, 8, 12, 16, n_layers-2]
    log(f"Test layers: {layers_to_test}")
    
    # Extract W_U
    log("Extracting W_U (lm_head weight matrix)...")
    W_U = get_W_U(model, model_name)
    log(f"W_U shape: {W_U.shape}")
    
    # Get target token IDs
    target_token_ids = {}
    for t in FOCUS_VOCAB:
        ids = tokenizer.encode(" " + t, add_special_tokens=False)
        if ids:
            target_token_ids[t] = ids[0]
            log(f"  Token '{t}' -> id={ids[0]}")
        else:
            ids = tokenizer.encode(t, add_special_tokens=False)
            if ids:
                target_token_ids[t] = ids[0]
                log(f"  Token '{t}' -> id={ids[0]} (no space)")
    
    # Step 1: Extract directions
    directions = extract_directions(model, tokenizer, device, layers_to_test, info)
    
    # Step 2: W_U readout analysis
    log("Computing W_U readout analysis...")
    WU_results = compute_WU_readout(W_U, directions, target_token_ids, tokenizer)
    
    # Step 3: Jacobian gain analysis (empirical)
    log("Computing Jacobian gain analysis...")
    jacobian_results = compute_jacobian_gain(model, tokenizer, device, directions, layers_to_test, info)
    
    # Combine results
    final_results = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "vocab_size": info.vocab_size,
        "test_layers": layers_to_test,
        "target_token_ids": {k: int(v) for k, v in target_token_ids.items()},
        "WU_shape": list(W_U.shape),
        "layers": {},
    }
    
    for li in layers_to_test:
        li_data = {
            "direction_norms": WU_results.get(li, {}).get("direction_norms", {}),
            "WU_gains": WU_results.get(li, {}).get("WU_gains", {}),
            "WU_gain_ratios": {},
            "target_logits": WU_results.get(li, {}).get("target_logits", {}),
            "comparisons": WU_results.get(li, {}).get("comparisons", {}),
            "jacobian_gains": {},
            "random_WU_gain": WU_results.get(li, {}).get("random_gain_avg", 0),
            "random_jacobian_gain": jacobian_results.get(li, {}).get("random", {}).get("jacobian_gain_avg", 0),
        }
        
        # W_U gain ratios
        random_wu = li_data["random_WU_gain"]
        for name, gain in li_data["WU_gains"].items():
            li_data["WU_gain_ratios"][name] = float(gain / random_wu) if random_wu > 0 else 0
        
        # Jacobian gains
        random_jac = li_data["random_jacobian_gain"]
        for name, data in jacobian_results.get(li, {}).items():
            if isinstance(data, dict) and "jacobian_gain" in data:
                jg = data["jacobian_gain"]
                li_data["jacobian_gains"][name] = {
                    "gain": float(jg),
                    "ratio_to_random": float(jg / random_jac) if random_jac > 0 else 0,
                    "delta_not": data.get("delta_not", 0),
                }
        
        # Key analysis: amplification source
        # If Jacobian_gain(O_clean_R) / WU_gain(O_clean_R) > Jacobian_gain(O_not) / WU_gain(O_not),
        # then intermediate layers amplify delta more than shared
        O_not_wu = li_data["WU_gains"].get("O_not", 0)
        O_clean_R_wu = li_data["WU_gains"].get("O_clean_R", 0)
        O_not_jac = li_data["jacobian_gains"].get("O_not", {}).get("gain", 0)
        O_clean_R_jac = li_data["jacobian_gains"].get("O_clean_R", {}).get("gain", 0)
        
        if O_not_wu > 0 and O_clean_R_wu > 0:
            O_not_intermediate_ratio = O_not_jac / O_not_wu if O_not_wu > 0 else 0
            O_clean_R_intermediate_ratio = O_clean_R_jac / O_clean_R_wu if O_clean_R_wu > 0 else 0
            li_data["amplification_source"] = {
                "O_not_jac_over_wu": float(O_not_intermediate_ratio),
                "O_clean_R_jac_over_wu": float(O_clean_R_intermediate_ratio),
                "delta_amplified_more": "O_clean_R" if O_clean_R_intermediate_ratio > O_not_intermediate_ratio else "O_not",
                "amplification_ratio": float(O_clean_R_intermediate_ratio / O_not_intermediate_ratio) if O_not_intermediate_ratio > 0 else 0,
            }
        
        final_results["layers"][str(li)] = li_data
    
    # Save results
    out_path = RESULT_DIR / f"{model_name}_WU_readout.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")
    
    # Print summary
    log("\n" + "="*70)
    log(f"W_U READOUT ANALYSIS SUMMARY - {model_name}")
    log("="*70)
    
    for li in layers_to_test:
        lr = final_results["layers"][str(li)]
        log(f"\n--- Layer {li} ---")
        log(f"  Direction norms:")
        for name, norm in lr["direction_norms"].items():
            log(f"    {name}: {norm:.4f}")
        
        log(f"  W_U gain (||WU@v||/||v||) and ratio to random:")
        random_wu = lr["random_WU_gain"]
        for name, gain in lr["WU_gains"].items():
            ratio = gain / random_wu if random_wu > 0 else 0
            log(f"    {name}: gain={gain:.2f}, ratio={ratio:.2f}x random")
        
        log(f"  Target token logit changes (W_U@v[target]/||v||):")
        for name, tdict in lr.get("target_logits", {}).items():
            not_val = tdict.get("not", 0)
            happy_val = tdict.get("happy", 0)
            sad_val = tdict.get("sad", 0)
            log(f"    {name}: not={not_val:.4f}, happy={happy_val:.4f}, sad={sad_val:.4f}")
        
        log(f"  Jacobian gain and ratio to random:")
        random_jac = lr["random_jacobian_gain"]
        for name, data in lr.get("jacobian_gains", {}).items():
            if isinstance(data, dict):
                jg = data.get("gain", 0)
                ratio = data.get("ratio_to_random", 0)
                dn = data.get("delta_not", 0)
                log(f"    {name}: jac_gain={jg:.4f}, ratio={ratio:.2f}x random, delta_not={dn:.4f}")
        
        if "amplification_source" in lr:
            amp = lr["amplification_source"]
            log(f"  AMPLIFICATION SOURCE:")
            log(f"    O_not: jac/wu = {amp['O_not_jac_over_wu']:.4f}")
            log(f"    O_clean_R: jac/wu = {amp['O_clean_R_jac_over_wu']:.4f}")
            log(f"    Delta amplified more: {amp['delta_amplified_more']}")
            log(f"    Amplification ratio: {amp['amplification_ratio']:.4f}")
    
    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Model {model_name} released.")
    
    return final_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    if model_name == "all":
        for mn in ["qwen3", "glm4", "deepseek7b"]:
            log(f"\n{'#'*70}")
            log(f"# Starting {mn}")
            log(f"{'#'*70}")
            try:
                run_model(mn)
            except Exception as e:
                log(f"ERROR running {mn}: {e}")
                import traceback
                traceback.print_exc()
            # Wait between models
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(10)
    else:
        run_model(model_name)
