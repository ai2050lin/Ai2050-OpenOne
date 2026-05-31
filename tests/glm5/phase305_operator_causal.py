"""
Phase 305: Operator-Scope (O/S) Causal Testing
================================================
Move beyond R/F to test language logic operators.

Core questions:
1. Does "not" have a unified operator direction O(not) across operands?
2. Is O(not) = "antonym" direction? Or is negation orthogonal to antonym?
3. Is O(not) shared across roles (adj, verb, noun)?
4. Does O(not) interact with S(scope)?
5. Cross-model comparison of operator encoding

Stimulus design:
- 20 adjectives × 4 frames × 2 conditions (affirm/negate) = 160 negation sentences
- 12 antonym pairs × 4 frames × 3 conditions (A/B/not-A) = 144 antonym sentences
- 10 verbs × 4 frames × 2 conditions (affirm/negate) = 80 negation sentences
- Total: ~384 sentences

Key innovations:
- Extract O(operator) direction: h(negated) - h(affirmed) at target token position
- Cross-operand LOO for O(not): leave one operand out, average remaining
- Compare O(not) with antonym direction: h(antonym) - h(word)
- Causal test: inject O(not) on affirmed → does output move toward negated?
- Per-role O(not) analysis: is O(not) for adjectives same as for verbs?

Theory: h = I + R + C + O + S + U

Usage:
  python tests/glm5/phase305_operator_causal.py qwen3
  python tests/glm5/phase305_operator_causal.py glm4
  python tests/glm5/phase305_operator_causal.py deepseek7b
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

RESULT_DIR = Path("results/phase305_operator_causal")
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
# Adjectives for negation test (20 words)
NEG_ADJECTIVES = [
    "happy", "sad", "big", "small", "good", "bad",
    "warm", "cold", "fast", "slow", "bright", "dark",
    "safe", "dangerous", "clean", "dirty", "rich", "poor",
    "strong", "weak",
]

# Antonym pairs (12 pairs)
ANTONYM_PAIRS = [
    ("happy", "sad"), ("big", "small"), ("good", "bad"),
    ("warm", "cold"), ("fast", "slow"), ("bright", "dark"),
    ("safe", "dangerous"), ("clean", "dirty"), ("rich", "poor"),
    ("strong", "weak"), ("young", "old"), ("open", "closed"),
]

# Verbs for negation test (10 words)
NEG_VERBS = [
    "run", "eat", "sleep", "work", "play",
    "read", "write", "sing", "dance", "fight",
]

# Nouns for negation test (8 words with determiner negation)
NEG_NOUNS = [
    "person", "child", "animal", "bird",
    "house", "car", "book", "food",
]

# Frames for negation
ADJ_NEG_FRAMES = {
    "F1_copula": ("the {subj} is {adj}", "the {subj} is not {adj}"),
    "F2_seem":   ("the {subj} seems {adj}", "the {subj} does not seem {adj}"),
    "F3_remain": ("the {subj} remains {adj}", "the {subj} does not remain {adj}"),
    "F4_look":   ("the {subj} looks {adj}", "the {subj} does not look {adj}"),
}

VERB_NEG_FRAMES = {
    "F1_simple": ("the {subj} {verb}s", "the {subj} does not {verb}"),
    "F2_will":   ("the {subj} will {verb}", "the {subj} will not {verb}"),
    "F3_can":    ("the {subj} can {verb}", "the {subj} cannot {verb}"),
    "F4_should": ("the {subj} should {verb}", "the {subj} should not {verb}"),
}

NOUN_NEG_FRAMES = {
    "F1_exist":  ("there is a {noun}", "there is no {noun}"),
    "F2_have":   ("they have a {noun}", "they have no {noun}"),
    "F3_see":    ("they saw the {noun}", "they saw no {noun}"),
    "F4_find":   ("they found a {noun}", "they found no {noun}"),
}

# Antonym frames
ANT_FRAMES = {
    "F1": "the person is {adj}",
    "F2": "the dog is {adj}",
    "F3": "the house is {adj}",
    "F4": "that thing is {adj}",
}

# Subjects for rotation
SUBJECTS = ["person", "dog", "cat", "car", "house", "river",
            "child", "bird", "tree", "road", "song", "food",
            "man", "woman", "city", "book", "student", "worker",
            "river", "mountain"]


def build_stimuli():
    """Build all stimuli for operator testing."""
    stimuli = []
    
    # ---- Part 1: Adjective negation (20 adj × 4 frames × 2 cond = 160) ----
    for i, adj in enumerate(NEG_ADJECTIVES):
        subj = SUBJECTS[i % len(SUBJECTS)]
        for flabel, (aff_t, neg_t) in ADJ_NEG_FRAMES.items():
            stimuli.append({
                "sentence": aff_t.format(subj=subj, adj=adj),
                "target_word": adj, "operand": adj, "operator": "none",
                "frame": flabel, "condition": "affirm", "group": "adj_negation",
                "role": "adj",
            })
            stimuli.append({
                "sentence": neg_t.format(subj=subj, adj=adj),
                "target_word": adj, "operand": adj, "operator": "not",
                "frame": flabel, "condition": "negate", "group": "adj_negation",
                "role": "adj",
            })
    
    # ---- Part 2: Verb negation (10 verb × 4 frames × 2 cond = 80) ----
    for i, verb in enumerate(NEG_VERBS):
        subj = SUBJECTS[i % len(SUBJECTS)]
        for flabel, (aff_t, neg_t) in VERB_NEG_FRAMES.items():
            stimuli.append({
                "sentence": aff_t.format(subj=subj, verb=verb),
                "target_word": verb, "operand": verb, "operator": "none",
                "frame": flabel, "condition": "affirm", "group": "verb_negation",
                "role": "verb",
            })
            stimuli.append({
                "sentence": neg_t.format(subj=subj, verb=verb),
                "target_word": verb, "operand": verb, "operator": "not",
                "frame": flabel, "condition": "negate", "group": "verb_negation",
                "role": "verb",
            })
    
    # ---- Part 3: Noun negation (8 noun × 4 frames × 2 cond = 64) ----
    for i, noun in enumerate(NEG_NOUNS):
        for flabel, (aff_t, neg_t) in NOUN_NEG_FRAMES.items():
            stimuli.append({
                "sentence": aff_t.format(noun=noun),
                "target_word": noun, "operand": noun, "operator": "none",
                "frame": flabel, "condition": "affirm", "group": "noun_negation",
                "role": "noun",
            })
            stimuli.append({
                "sentence": neg_t.format(noun=noun),
                "target_word": noun, "operand": noun, "operator": "not",
                "frame": flabel, "condition": "negate", "group": "noun_negation",
                "role": "noun",
            })
    
    # ---- Part 4: Antonym comparison (12 pairs × 4 frames × 3 cond = 144) ----
    for w1, w2 in ANTONYM_PAIRS:
        for flabel, template in ANT_FRAMES.items():
            stimuli.append({
                "sentence": template.format(adj=w1),
                "target_word": w1, "operand": w1, "operator": "none",
                "frame": flabel, "condition": "word_A", "group": "antonym",
                "role": "adj", "antonym_pair": (w1, w2),
            })
            stimuli.append({
                "sentence": template.format(adj=w2),
                "target_word": w2, "operand": w2, "operator": "none",
                "frame": flabel, "condition": "word_B", "group": "antonym",
                "role": "adj", "antonym_pair": (w1, w2),
            })
            stimuli.append({
                "sentence": template.format(adj=w1),
                "target_word": w1, "operand": w1, "operator": "not",
                "frame": flabel, "condition": "not_A", "group": "antonym_not",
                "role": "adj", "antonym_pair": (w1, w2),
            })
    
    return stimuli


# =====================================================================
# MODEL LOADING — BF16 + device_map="auto" + flash_attn
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
    log(f"  Loaded. GPU={gpu_mem:.1f}GB, class={type(model).__name__}")
    
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
        if t == target_lower:
            return i
    for i, t in enumerate(decoded_tokens):
        if target_lower in t or t in target_lower:
            return i
    if len(target_lower) >= 2:
        for i, t in enumerate(decoded_tokens):
            if target_lower[:3] in t:
                return i
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


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


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
    elif isinstance(obj, tuple):
        return list(obj)
    return obj


# =====================================================================
# MAIN
# =====================================================================
def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase305_{model_name}.txt"
    _log_file = str(log_file)
    log(f"Phase 305: Operator-Scope Causal Testing -- {model_name}")
    
    # ---- Load model ----
    model, tok = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    nl = info.n_layers
    d_model = info.d_model
    log(f"  n_layers={nl}, d_model={d_model}")
    
    # ---- Build and resolve stimuli ----
    all_stimuli = resolve_positions(build_stimuli(), tok)
    log(f"  Total stimuli resolved: {len(all_stimuli)}")
    
    # Count by group
    group_counts = defaultdict(int)
    role_counts = defaultdict(int)
    for s in all_stimuli:
        group_counts[s.get("group", "")] += 1
        role_counts[s.get("role", "")] += 1
    for g, c in sorted(group_counts.items()):
        log(f"    {g}: {c}")
    log(f"  Roles: {dict(role_counts)}")
    
    # Deduplicate sentences
    all_sentences = []
    sent_to_idx = {}
    for s in all_stimuli:
        sent = s["sentence"]
        if sent not in sent_to_idx:
            sent_to_idx[sent] = len(all_sentences)
            all_sentences.append(sent)
        s["_idx"] = sent_to_idx[sent]
    log(f"  Unique sentences: {len(all_sentences)}")
    
    # ---- Capture all sentences ----
    log(f"Capturing {len(all_sentences)} unique sentences...")
    t0 = time.time()
    captures = {}
    for i, sent in enumerate(all_sentences):
        captures[i] = _capture_single(model, tok, sent)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1)
            eta = (len(all_sentences) - i - 1) / rate
            log(f"  {i+1}/{len(all_sentences)} ({rate:.1f}/s) ETA={eta:.0f}s GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
            gc.collect()
            torch.cuda.empty_cache()
    log(f"Done capturing in {time.time()-t0:.0f}s")
    
    # Organize stimuli by group
    adj_neg = [s for s in all_stimuli if s.get("group") == "adj_negation"]
    verb_neg = [s for s in all_stimuli if s.get("group") == "verb_negation"]
    noun_neg = [s for s in all_stimuli if s.get("group") == "noun_negation"]
    antonym = [s for s in all_stimuli if s.get("group") == "antonym"]
    antonym_not = [s for s in all_stimuli if s.get("group") == "antonym_not"]
    
    log(f"  adj_negation: {len(adj_neg)}, verb_negation: {len(verb_neg)}, "
        f"noun_negation: {len(noun_neg)}, antonym: {len(antonym)}, antonym_not: {len(antonym_not)}")
    
    # ---- Layer selection ----
    sample_layers = sorted(set([
        max(1, nl // 8), max(1, nl // 4), max(1, 3 * nl // 8),
        nl // 2, 5 * nl // 8, 3 * nl // 4, 7 * nl // 8, nl - 2
    ]) & set(range(1, nl)))
    log(f"Sample layers: {sample_layers}")
    
    # =====================================================================
    # PART A: OPERATOR DIRECTION EXTRACTION
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"PART A: OPERATOR DIRECTION EXTRACTION")
    log(f"{'='*60}")
    
    results = {}
    
    for li in sample_layers:
        log(f"\n--- Layer {li} ---")
        layer_data = {}
        
        # ---- A1: Per-operand O(not) direction ----
        # For each operand, compute O(not) = mean(h_negated - h_affirmed) across frames
        
        # Adjective operands
        adj_O_not = {}
        for adj in NEG_ADJECTIVES:
            aff_vecs = []
            neg_vecs = []
            for s in adj_neg:
                if s["operand"] == adj and s["condition"] == "affirm":
                    idx = s.get("_idx")
                    pos = s.get("target_pos")
                    if idx is not None and pos is not None:
                        h = captures[idx]["hidden"].get(li)
                        if h is not None and pos < h.shape[1]:
                            aff_vecs.append(h[0, pos, :].numpy().copy())
                elif s["operand"] == adj and s["condition"] == "negate":
                    idx = s.get("_idx")
                    pos = s.get("target_pos")
                    if idx is not None and pos is not None:
                        h = captures[idx]["hidden"].get(li)
                        if h is not None and pos < h.shape[1]:
                            neg_vecs.append(h[0, pos, :].numpy().copy())
            
            if aff_vecs and neg_vecs:
                adj_O_not[adj] = np.mean(neg_vecs, axis=0) - np.mean(aff_vecs, axis=0)
        
        # Verb operands
        verb_O_not = {}
        for verb in NEG_VERBS:
            aff_vecs = []
            neg_vecs = []
            for s in verb_neg:
                if s["operand"] == verb and s["condition"] == "affirm":
                    idx = s.get("_idx")
                    pos = s.get("target_pos")
                    if idx is not None and pos is not None:
                        h = captures[idx]["hidden"].get(li)
                        if h is not None and pos < h.shape[1]:
                            aff_vecs.append(h[0, pos, :].numpy().copy())
                elif s["operand"] == verb and s["condition"] == "negate":
                    idx = s.get("_idx")
                    pos = s.get("target_pos")
                    if idx is not None and pos is not None:
                        h = captures[idx]["hidden"].get(li)
                        if h is not None and pos < h.shape[1]:
                            neg_vecs.append(h[0, pos, :].numpy().copy())
            
            if aff_vecs and neg_vecs:
                verb_O_not[verb] = np.mean(neg_vecs, axis=0) - np.mean(aff_vecs, axis=0)
        
        # Noun operands
        noun_O_not = {}
        for noun in NEG_NOUNS:
            aff_vecs = []
            neg_vecs = []
            for s in noun_neg:
                if s["operand"] == noun and s["condition"] == "affirm":
                    idx = s.get("_idx")
                    pos = s.get("target_pos")
                    if idx is not None and pos is not None:
                        h = captures[idx]["hidden"].get(li)
                        if h is not None and pos < h.shape[1]:
                            aff_vecs.append(h[0, pos, :].numpy().copy())
                elif s["operand"] == noun and s["condition"] == "negate":
                    idx = s.get("_idx")
                    pos = s.get("target_pos")
                    if idx is not None and pos is not None:
                        h = captures[idx]["hidden"].get(li)
                        if h is not None and pos < h.shape[1]:
                            neg_vecs.append(h[0, pos, :].numpy().copy())
            
            if aff_vecs and neg_vecs:
                noun_O_not[noun] = np.mean(neg_vecs, axis=0) - np.mean(aff_vecs, axis=0)
        
        log(f"  O(not) extracted: {len(adj_O_not)} adj, {len(verb_O_not)} verb, {len(noun_O_not)} noun")
        
        # ---- A2: LOO O(not) direction (cross-operand sharing) ----
        all_O_not = {**adj_O_not, **verb_O_not, **noun_O_not}
        
        # Per-role LOO
        adj_O_loo = {}
        for adj in adj_O_not:
            others = [v for k, v in adj_O_not.items() if k != adj]
            if others:
                adj_O_loo[adj] = np.mean(others, axis=0)
        
        verb_O_loo = {}
        for verb in verb_O_not:
            others = [v for k, v in verb_O_not.items() if k != verb]
            if others:
                verb_O_loo[verb] = np.mean(others, axis=0)
        
        noun_O_loo = {}
        for noun in noun_O_not:
            others = [v for k, v in noun_O_not.items() if k != noun]
            if others:
                noun_O_loo[noun] = np.mean(others, axis=0)
        
        # Cross-role O(not) average
        adj_O_avg = np.mean(list(adj_O_not.values()), axis=0) if adj_O_not else np.zeros(d_model)
        verb_O_avg = np.mean(list(verb_O_not.values()), axis=0) if verb_O_not else np.zeros(d_model)
        noun_O_avg = np.mean(list(noun_O_not.values()), axis=0) if noun_O_not else np.zeros(d_model)
        all_O_avg = np.mean(list(all_O_not.values()), axis=0) if all_O_not else np.zeros(d_model)
        
        # ---- A3: Antonym direction ----
        antonym_dirs = {}
        for w1, w2 in ANTONYM_PAIRS:
            w1_vecs = []
            w2_vecs = []
            for s in antonym:
                pair = s.get("antonym_pair")
                if pair is None:
                    continue
                pair = tuple(pair)
                idx = s.get("_idx")
                pos = s.get("target_pos")
                if idx is None or pos is None:
                    continue
                h = captures[idx]["hidden"].get(li)
                if h is None or pos >= h.shape[1]:
                    continue
                
                if pair == (w1, w2) and s["condition"] == "word_A":
                    w1_vecs.append(h[0, pos, :].numpy().copy())
                elif pair == (w1, w2) and s["condition"] == "word_B":
                    w2_vecs.append(h[0, pos, :].numpy().copy())
            
            if w1_vecs and w2_vecs:
                antonym_dirs[(w1, w2)] = np.mean(w2_vecs, axis=0) - np.mean(w1_vecs, axis=0)
        
        log(f"  Antonym directions: {len(antonym_dirs)} pairs")
        
        # ---- A4: O(not) vs Antonym comparison ----
        log(f"\n  O(not) vs Antonym comparison:")
        for w1, w2 in ANTONYM_PAIRS:
            if w1 in adj_O_not and (w1, w2) in antonym_dirs:
                cos_OA = cosine_sim(adj_O_not[w1], antonym_dirs[(w1, w2)])
                cos_OA_rev = cosine_sim(adj_O_not[w1], -antonym_dirs[(w1, w2)])
                log(f"    {w1}->{w2}: cos(O(not), antonym)={cos_OA:+.4f}, "
                    f"cos(O(not), -antonym)={cos_OA_rev:+.4f}")
        
        # ---- A5: Cross-role O(not) sharing ----
        log(f"\n  Cross-role O(not) sharing:")
        if np.linalg.norm(adj_O_avg) > 1e-10 and np.linalg.norm(verb_O_avg) > 1e-10:
            cos_av = cosine_sim(adj_O_avg, verb_O_avg)
            log(f"    cos(O_adj, O_verb) = {cos_av:+.4f}")
        if np.linalg.norm(adj_O_avg) > 1e-10 and np.linalg.norm(noun_O_avg) > 1e-10:
            cos_an = cosine_sim(adj_O_avg, noun_O_avg)
            log(f"    cos(O_adj, O_noun) = {cos_an:+.4f}")
        if np.linalg.norm(verb_O_avg) > 1e-10 and np.linalg.norm(noun_O_avg) > 1e-10:
            cos_vn = cosine_sim(verb_O_avg, noun_O_avg)
            log(f"    cos(O_verb, O_noun) = {cos_vn:+.4f}")
        
        # ---- A6: Per-operand O(not) consistency (within-role LOO) ----
        adj_loo_cos = []
        for adj in adj_O_not:
            if adj in adj_O_loo:
                c = cosine_sim(adj_O_not[adj], adj_O_loo[adj])
                adj_loo_cos.append(c)
        
        verb_loo_cos = []
        for verb in verb_O_not:
            if verb in verb_O_loo:
                c = cosine_sim(verb_O_not[verb], verb_O_loo[verb])
                verb_loo_cos.append(c)
        
        noun_loo_cos = []
        for noun in noun_O_not:
            if noun in noun_O_loo:
                c = cosine_sim(noun_O_not[noun], noun_O_loo[noun])
                noun_loo_cos.append(c)
        
        if adj_loo_cos:
            log(f"    adj LOO cos: {np.mean(adj_loo_cos):+.4f} ± {np.std(adj_loo_cos):.4f}")
        if verb_loo_cos:
            log(f"    verb LOO cos: {np.mean(verb_loo_cos):+.4f} ± {np.std(verb_loo_cos):.4f}")
        if noun_loo_cos:
            log(f"    noun LOO cos: {np.mean(noun_loo_cos):+.4f} ± {np.std(noun_loo_cos):.4f}")
        
        # ---- A7: O(not) norm statistics ----
        adj_O_norms = [np.linalg.norm(v) for v in adj_O_not.values()]
        verb_O_norms = [np.linalg.norm(v) for v in verb_O_not.values()]
        noun_O_norms = [np.linalg.norm(v) for v in noun_O_not.values()]
        
        log(f"\n  O(not) norm statistics:")
        if adj_O_norms:
            log(f"    adj: mean={np.mean(adj_O_norms):.2f} std={np.std(adj_O_norms):.2f}")
        if verb_O_norms:
            log(f"    verb: mean={np.mean(verb_O_norms):.2f} std={np.std(verb_O_norms):.2f}")
        if noun_O_norms:
            log(f"    noun: mean={np.mean(noun_O_norms):.2f} std={np.std(noun_O_norms):.2f}")
        
        # =====================================================================
        # PART B: CAUSAL TEST OF O(not)
        # =====================================================================
        log(f"\n  --- Causal test of O(not) ---")
        
        # Select test pairs: first frame, affirmed sentence
        test_cases = []
        
        # Adjective test cases
        for adj in NEG_ADJECTIVES:
            # Find affirmed sentence for this adj
            aff_stims = [s for s in adj_neg if s["operand"] == adj and s["condition"] == "affirm"]
            neg_stims = [s for s in adj_neg if s["operand"] == adj and s["condition"] == "negate"]
            if aff_stims and neg_stims:
                aff_s = aff_stims[0]
                neg_s = neg_stims[0]
                test_cases.append({
                    "aff_stim": aff_s, "neg_stim": neg_s,
                    "operand": adj, "role": "adj",
                    "O_not": adj_O_not.get(adj, np.zeros(d_model)),
                    "O_loo": adj_O_loo.get(adj, np.zeros(d_model)),
                    "O_avg": adj_O_avg,
                })
        
        # Verb test cases
        for verb in NEG_VERBS:
            aff_stims = [s for s in verb_neg if s["operand"] == verb and s["condition"] == "affirm"]
            neg_stims = [s for s in verb_neg if s["operand"] == verb and s["condition"] == "negate"]
            if aff_stims and neg_stims:
                aff_s = aff_stims[0]
                neg_s = neg_stims[0]
                test_cases.append({
                    "aff_stim": aff_s, "neg_stim": neg_s,
                    "operand": verb, "role": "verb",
                    "O_not": verb_O_not.get(verb, np.zeros(d_model)),
                    "O_loo": verb_O_loo.get(verb, np.zeros(d_model)),
                    "O_avg": verb_O_avg,
                })
        
        # Noun test cases
        for noun in NEG_NOUNS:
            aff_stims = [s for s in noun_neg if s["operand"] == noun and s["condition"] == "affirm"]
            neg_stims = [s for s in noun_neg if s["operand"] == noun and s["condition"] == "negate"]
            if aff_stims and neg_stims:
                aff_s = aff_stims[0]
                neg_s = neg_stims[0]
                test_cases.append({
                    "aff_stim": aff_s, "neg_stim": neg_s,
                    "operand": noun, "role": "noun",
                    "O_not": noun_O_not.get(noun, np.zeros(d_model)),
                    "O_loo": noun_O_loo.get(noun, np.zeros(d_model)),
                    "O_avg": noun_O_avg,
                })
        
        log(f"  Test cases: {len(test_cases)}")
        
        # Run causal tests
        causal_results = []
        
        for ti, tc in enumerate(test_cases):
            aff_s = tc["aff_stim"]
            neg_s = tc["neg_stim"]
            
            aff_idx = aff_s.get("_idx")
            aff_pos = aff_s.get("target_pos")
            neg_idx = neg_s.get("_idx")
            neg_pos = neg_s.get("target_pos")
            
            if aff_idx is None or neg_idx is None:
                continue
            
            # Get target shift in logit space
            aff_logits = captures[aff_idx]["logits"][0, -1, :].numpy().copy()
            neg_logits = captures[neg_idx]["logits"][0, -1, :].numpy().copy()
            target_shift = neg_logits - aff_logits
            
            # Get full_delta in activation space
            h_aff = captures[aff_idx]["hidden"].get(li)
            h_neg = captures[neg_idx]["hidden"].get(li)
            if h_aff is None or h_neg is None:
                continue
            if aff_pos >= h_aff.shape[1] or neg_pos >= h_neg.shape[1]:
                continue
            
            v_aff = h_aff[0, aff_pos, :].numpy().copy()
            v_neg = h_neg[0, neg_pos, :].numpy().copy()
            full_delta = v_neg - v_aff
            
            # Define patch conditions
            conditions = {
                "full_delta": full_delta,
                "O_not": tc["O_not"],
                "O_loo": tc["O_loo"],
                "O_avg": tc["O_avg"],
            }
            
            # Antonym direction (if available)
            operand = tc["operand"]
            for w1, w2 in ANTONYM_PAIRS:
                if w1 == operand and (w1, w2) in antonym_dirs:
                    conditions["antonym_dir"] = antonym_dirs[(w1, w2)]
                elif w2 == operand and (w1, w2) in antonym_dirs:
                    conditions["antonym_dir"] = -antonym_dirs[(w1, w2)]
            
            # Run causal tests
            tc_result = {
                "operand": tc["operand"], "role": tc["role"],
                "full_delta_norm": float(np.linalg.norm(full_delta)),
                "O_not_norm": float(np.linalg.norm(tc["O_not"])),
                "cos_O_full": float(cosine_sim(tc["O_not"], full_delta)),
            }
            
            # O(not) vs O_avg sharing
            if np.linalg.norm(tc["O_not"]) > 1e-10 and np.linalg.norm(tc["O_avg"]) > 1e-10:
                tc_result["cos_O_Oavg"] = float(cosine_sim(tc["O_not"], tc["O_avg"]))
            
            for cond_name, patch_vec in conditions.items():
                pnorm = np.linalg.norm(patch_vec)
                if pnorm < 1e-10:
                    tc_result[f"{cond_name}_cos_shift"] = 0.0
                    tc_result[f"{cond_name}_norm"] = 0.0
                    continue
                
                patched_logits = run_with_patched_hidden(
                    model, tok, aff_s["sentence"], li, aff_pos, patch_vec)
                if patched_logits is not None:
                    p_logits = patched_logits[0, -1, :].numpy().copy()
                    cos_shift = cosine_sim(p_logits - aff_logits, target_shift)
                    tc_result[f"{cond_name}_cos_shift"] = float(cos_shift)
                    tc_result[f"{cond_name}_norm"] = float(pnorm)
                else:
                    tc_result[f"{cond_name}_cos_shift"] = None
                    tc_result[f"{cond_name}_norm"] = float(pnorm)
            
            # Random controls (3 directions)
            rand_shifts = []
            for ri in range(3):
                rng2 = np.random.RandomState(ri * 100 + hash(operand) % 100)
                rdir = rng2.randn(d_model)
                rdir = rdir / np.linalg.norm(rdir)
                rpatch = rdir * np.linalg.norm(full_delta)
                plogits = run_with_patched_hidden(model, tok, aff_s["sentence"], li, aff_pos, rpatch)
                if plogits is not None:
                    pl = plogits[0, -1, :].numpy().copy()
                    rand_shifts.append(cosine_sim(pl - aff_logits, target_shift))
            tc_result["avg_random_shift"] = float(np.mean(rand_shifts)) if rand_shifts else 0.0
            
            causal_results.append(tc_result)
            
            if (ti + 1) % 10 == 0:
                log(f"  {ti+1}/{len(test_cases)} test cases done, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
                gc.collect()
                torch.cuda.empty_cache()
        
        # ---- Layer summary ----
        if causal_results:
            for metric in ["full_delta_cos_shift", "O_not_cos_shift", "O_loo_cos_shift",
                          "O_avg_cos_shift", "antonym_dir_cos_shift", "avg_random_shift"]:
                vals = [v.get(metric) for v in causal_results if v.get(metric) is not None]
                if vals:
                    log(f"    {metric}: {np.mean(vals):+.4f} pos={sum(1 for c in vals if c>0)}/{len(vals)}")
            
            # Per-role breakdown
            for role in ["adj", "verb", "noun"]:
                role_items = [v for v in causal_results if v.get("role") == role]
                if role_items:
                    O_not_vals = [v.get("O_not_cos_shift") for v in role_items if v.get("O_not_cos_shift") is not None]
                    FD_vals = [v.get("full_delta_cos_shift") for v in role_items if v.get("full_delta_cos_shift") is not None]
                    O_avg_vals = [v.get("O_avg_cos_shift") for v in role_items if v.get("O_avg_cos_shift") is not None]
                    if O_not_vals:
                        log(f"    [{role}] O_not: {np.mean(O_not_vals):+.4f} n={len(O_not_vals)}")
                    if FD_vals:
                        log(f"    [{role}] full_delta: {np.mean(FD_vals):+.4f}")
                    if O_avg_vals:
                        log(f"    [{role}] O_avg: {np.mean(O_avg_vals):+.4f}")
        
        layer_data["O_directions"] = {
            "adj": {k: v.tolist() for k, v in adj_O_not.items()},
            "verb": {k: v.tolist() for k, v in verb_O_not.items()},
            "noun": {k: v.tolist() for k, v in noun_O_not.items()},
        }
        layer_data["antonym_dirs"] = {f"{w1}_{w2}": v.tolist() for (w1, w2), v in antonym_dirs.items()}
        layer_data["cross_role_cos"] = {
            "adj_verb": float(cosine_sim(adj_O_avg, verb_O_avg)) if np.linalg.norm(adj_O_avg) > 1e-10 and np.linalg.norm(verb_O_avg) > 1e-10 else 0,
            "adj_noun": float(cosine_sim(adj_O_avg, noun_O_avg)) if np.linalg.norm(adj_O_avg) > 1e-10 and np.linalg.norm(noun_O_avg) > 1e-10 else 0,
            "verb_noun": float(cosine_sim(verb_O_avg, noun_O_avg)) if np.linalg.norm(verb_O_avg) > 1e-10 and np.linalg.norm(noun_O_avg) > 1e-10 else 0,
        }
        layer_data["loo_cos"] = {
            "adj": adj_loo_cos,
            "verb": verb_loo_cos,
            "noun": noun_loo_cos,
        }
        layer_data["causal"] = make_serializable(causal_results)
        
        results[str(li)] = layer_data
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    log(f"\nSaving results...")
    
    output = {
        "model": model_name,
        "n_layers": nl,
        "d_model": d_model,
        "sample_layers": sample_layers,
        "n_adj_operands": len(NEG_ADJECTIVES),
        "n_verb_operands": len(NEG_VERBS),
        "n_noun_operands": len(NEG_NOUNS),
        "n_antonym_pairs": len(ANTONYM_PAIRS),
        "results": make_serializable(results),
    }
    
    out_path = RESULT_DIR / f"{model_name}_operator_causal.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log(f"  Saved to {out_path}")
    
    # =====================================================================
    # KEY SUMMARY
    # =====================================================================
    log(f"\n{'='*60}")
    log(f"KEY SUMMARY")
    log(f"{'='*60}")
    
    mid_li = str(nl // 2)
    if mid_li in results:
        ld = results[mid_li]
        
        # O(not) vs antonym
        log(f"\n  O(not) vs Antonym at L{mid_li}:")
        causal = ld.get("causal", [])
        for cr in causal:
            if "antonym_dir_cos_shift" in cr:
                O_shift = cr.get("O_not_cos_shift", 0)
                ant_shift = cr.get("antonym_dir_cos_shift", 0)
                cos_OA = cr.get("cos_O_Oavg", 0)
                log(f"    {cr['operand']:12s} [{cr['role']:4s}]: "
                    f"O_not={O_shift:+.3f} antonym={ant_shift:+.3f} "
                    f"cos(O,O_avg)={cos_OA:+.3f}")
        
        # Cross-role O(not) sharing
        log(f"\n  Cross-role O(not) sharing at L{mid_li}:")
        crc = ld.get("cross_role_cos", {})
        for pair, cos_val in sorted(crc.items()):
            log(f"    {pair}: {cos_val:+.4f}")
        
        # LOO consistency
        log(f"\n  O(not) LOO consistency at L{mid_li}:")
        loo = ld.get("loo_cos", {})
        for role, cos_vals in loo.items():
            if cos_vals:
                log(f"    {role}: {np.mean(cos_vals):+.4f} ± {np.std(cos_vals):.4f}")
    
    release_model(model)
    log(f"Phase 305 complete for {model_name}!")


if __name__ == "__main__":
    main()
