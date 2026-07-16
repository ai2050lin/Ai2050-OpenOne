"""
Phase 946: Operator Algebra — Negation, Quantification, Modal Operators
========================================================================
路线B起点：从否定算子扩展到逻辑算子族，测试算子代数的封闭性和跨模型一致性。

Core questions:
1. REPRO: 复现Phase 300否定算子稳定性（跨operand LOO cosine）
2. EXTEND: 扩展到量化算子(all/some/no/every)和模态算子(must/can/should/may)
3. ALGEBRA: 测试双否定 closure (¬¬P ≈ P?), 算子组合 (¬all ≈ some-not?)
4. STRUCTURE: 算子间距离矩阵, 是否存在"算子空间"?
5. CROSS-MODEL: 三模型算子方向的跨模型一致性

Stimulus design:
- Negation (基线): 20 adj × 4 frames × 2 cond = 160 sentences
- Quantification: 5 quantifiers × 8 nouns × 4 frames = 160 sentences  
- Modal: 5 modals × 8 verbs × 4 frames = 160 sentences
- Double negation: 20 adj × 2 frames = 40 sentences
- Operator combination: not+all/every/must × 3 frames = ~30 sentences
- Total: ~550 unique sentences

Key metrics:
- LOO_cos: leave-one-operand-out cosine (operand稳定性)
- dim50/dim80: PCA维度集中度
- Cross-operator cosine matrix
- Double-neg closure: cos(h(not-not-X), h(X))
- Combo closure: cos(O(not-all), O(not)+O(all→some))

Usage:
  python tests/glm5/phase946_operator_algebra.py qwen3
  python tests/glm5/phase946_operator_algebra.py glm4
  python tests/glm5/phase946_operator_algebra.py deepseek7b
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

RESULT_DIR = Path("results/phase946_operator_algebra")
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
        except: pass

# =====================================================================
# STIMULUS DEFINITIONS
# =====================================================================

# Negation operands (基线复现, 与Phase300一致)
NEG_ADJECTIVES = [
    "happy", "sad", "big", "small", "good", "bad",
    "warm", "cold", "fast", "slow", "bright", "dark",
    "safe", "dangerous", "clean", "dirty", "rich", "poor",
    "strong", "weak",
]

NEG_SUBJECTS = ["person", "dog", "cat", "car", "house", "river",
                "child", "bird", "tree", "road", "song", "food",
                "man", "woman", "city", "book", "student", "worker",
                "river", "mountain"]

NEG_FRAMES = {
    "F1": ("the {subj} is {adj}", "the {subj} is not {adj}"),
    "F2": ("the {subj} seems {adj}", "the {subj} does not seem {adj}"),
    "F3": ("the {subj} remains {adj}", "the {subj} does not remain {adj}"),
    "F4": ("that {subj} is {adj}", "that {subj} is not {adj}"),
}

# Quantification operators
QUANTIFIERS = ["all", "some", "no", "every", "few"]
QUANT_PAIRS = [
    ("some", "all"), ("some", "no"), ("all", "no"),
    ("some", "every"), ("some", "few"),
]
QUANT_NOUNS = ["cats", "dogs", "birds", "students", "books", "cars", "trees", "houses"]
QUANT_PREDICATES = {
    "cats": ["are mammals", "sleep a lot", "like warmth"],
    "dogs": ["are loyal", "bark often", "enjoy walks"],
    "birds": ["can fly", "sing beautifully", "migrate south"],
    "students": ["work hard", "learn quickly", "need sleep"],
    "books": ["contain wisdom", "sit on shelves", "tell stories"],
    "cars": ["need fuel", "make noise", "pollute air"],
    "trees": ["provide shade", "grow tall", "produce oxygen"],
    "houses": ["have roofs", "need maintenance", "shelter people"],
}

# Modal operators
MODALS = ["must", "can", "should", "may", "might"]
MODAL_PAIRS = [
    ("can", "must"), ("can", "should"), ("can", "may"),
    ("may", "must"), ("might", "must"), ("should", "must"),
]
MODAL_VERBS = ["go", "come", "leave", "stay", "speak", "listen", "write", "read"]
MODAL_SUBJECTS = ["John", "Mary", "the doctor", "the teacher", "the student",
                  "the manager", "the worker", "the artist"]

# Double negation
DOUBLE_NEG_ADJ = NEG_ADJECTIVES[:16]  # use 16 for double negation

# Operator combinations
COMBO_PATTERNS = [
    ("not all", "some", "not"),   # not-all = some-not (De Morgan intuition)
    ("not every", "some", "not"),
    ("not must", "may", "not"),   # not-must = may-not (modal De Morgan)
    ("not should", "may", "not"),
]

def build_stimuli():
    """Build all stimuli for operator testing."""
    stimuli = []
    
    # === Part 1: Negation baseline (20 adj × 4 frames × 2 cond = 160) ===
    for i, adj in enumerate(NEG_ADJECTIVES):
        subj = NEG_SUBJECTS[i % len(NEG_SUBJECTS)]
        for flabel, (aff_t, neg_t) in NEG_FRAMES.items():
            stimuli.append({
                "sentence": aff_t.format(subj=subj, adj=adj),
                "target_word": adj, "operand": adj,
                "operator": "affirm", "op_type": "negation",
                "frame": flabel, "condition": "affirm",
            })
            stimuli.append({
                "sentence": neg_t.format(subj=subj, adj=adj),
                "target_word": adj, "operand": adj,
                "operator": "not", "op_type": "negation",
                "frame": flabel, "condition": "negate",
            })

    # === Part 2: Quantification operators ===
    for qnoun in QUANT_NOUNS:
        preds = QUANT_PREDICATES[qnoun]
        for pi, pred in enumerate(preds):
            for q in QUANTIFIERS:
                sent = f"{q} {qnoun} {pred}"
                stimuli.append({
                    "sentence": sent,
                    "target_word": qnoun,
                    "operand": f"{qnoun}_{pred[:20]}",  # unique operand id
                    "operator": q, "op_type": "quantification",
                    "frame": f"pred{pi}", "condition": f"quant_{q}",
                    "quant_noun": qnoun, "quant_pred": pred,
                })

    # === Part 3: Modal operators ===
    for vi, verb in enumerate(MODAL_VERBS):
        subj = MODAL_SUBJECTS[vi]
        for modal in MODALS:
            sent = f"{subj} {modal} {verb}"
            stimuli.append({
                "sentence": sent,
                "target_word": verb,
                "operand": f"{subj}_{verb}",
                "operator": modal, "op_type": "modal",
                "frame": "direct", "condition": f"modal_{modal}",
                "modal_verb": verb, "modal_subj": subj,
            })

    # === Part 4: Double negation ("not not happy") ===
    for adj in DOUBLE_NEG_ADJ:
        # Single neg vs double neg on same subject
        stimuli.append({
            "sentence": f"the person is {adj}",
            "target_word": adj, "operand": adj,
            "operator": "affirm", "op_type": "double_neg",
            "frame": "dn", "condition": "affirm",
        })
        stimuli.append({
            "sentence": f"the person is not {adj}",
            "target_word": adj, "operand": adj,
            "operator": "not", "op_type": "double_neg",
            "frame": "dn", "condition": "single_neg",
        })
        stimuli.append({
            "sentence": f"the person is not not {adj}",
            "target_word": adj, "operand": adj,
            "operator": "not_not", "op_type": "double_neg",
            "frame": "dn", "condition": "double_neg",
        })
        # Also test "not unhappy" type
        antonyms = {"happy": "sad", "sad": "happy", "big": "small", "small": "big",
                    "good": "bad", "bad": "good", "warm": "cold", "cold": "warm",
                    "fast": "slow", "slow": "fast", "bright": "dark", "dark": "bright",
                    "safe": "dangerous", "dangerous": "safe", "clean": "dirty", "dirty": "clean"}
        if adj in antonyms:
            ant = antonyms[adj]
            stimuli.append({
                "sentence": f"the person is not {ant}",
                "target_word": ant, "operand": f"ant_{ant}",
                "operator": "not_antonym", "op_type": "double_neg",
                "frame": "dn", "condition": "not_antonym",
                "related_adj": adj,
            })

    # === Part 5: Operator combinations ===
    for qnoun in QUANT_NOUNS[:4]:
        pred = QUANT_PREDICATES[qnoun][0]
        # Base sentences
        stimuli.append({
            "sentence": f"all {qnoun} {pred}",
            "target_word": qnoun, "operand": f"combo_{qnoun}",
            "operator": "all", "op_type": "combo",
            "frame": "combo", "condition": "combo_all",
        })
        stimuli.append({
            "sentence": f"not all {qnoun} {pred}",
            "target_word": qnoun, "operand": f"combo_{qnoun}",
            "operator": "not_all", "op_type": "combo",
            "frame": "combo", "condition": "combo_not_all",
        })
        stimuli.append({
            "sentence": f"some {qnoun} {pred}",
            "target_word": qnoun, "operand": f"combo_{qnoun}",
            "operator": "some", "op_type": "combo",
            "frame": "combo", "condition": "combo_some",
        })
        stimuli.append({
            "sentence": f"some {qnoun} not {pred}",
            "target_word": qnoun, "operand": f"combo_{qnoun}",
            "operator": "some_not", "op_type": "combo",
            "frame": "combo", "condition": "combo_some_not",
        })

    # Modal combos
    for vi, verb in enumerate(MODAL_VERBS[:4]):
        subj = MODAL_SUBJECTS[vi]
        stimuli.append({
            "sentence": f"{subj} must {verb}",
            "target_word": verb, "operand": f"combo_m_{subj}_{verb}",
            "operator": "must", "op_type": "combo",
            "frame": "combo_m", "condition": "combo_must",
        })
        stimuli.append({
            "sentence": f"{subj} not must {verb}",
            "target_word": verb, "operand": f"combo_m_{subj}_{verb}",
            "operator": "not_must", "op_type": "combo",
            "frame": "combo_m", "condition": "combo_not_must",
        })
        stimuli.append({
            "sentence": f"{subj} may {verb}",
            "target_word": verb, "operand": f"combo_m_{subj}_{verb}",
            "operator": "may", "op_type": "combo",
            "frame": "combo_m", "condition": "combo_may",
        })
        stimuli.append({
            "sentence": f"{subj} may not {verb}",
            "target_word": verb, "operand": f"combo_m_{subj}_{verb}",
            "operator": "may_not", "op_type": "combo",
            "frame": "combo_m", "condition": "combo_may_not",
        })

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
            log(f"  attn={attn_impl} OK")
            break
        except Exception as e:
            log(f"  attn={attn_impl} failed: {str(e)[:100]}")
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Loaded. GPU={gpu_mem:.1f}GB, class={type(model).__name__}")
    return model, tok


# =====================================================================
# CAPTURE & UTILITIES
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


def find_target_pos(tokenizer, sent, target_word):
    """Find position of target_word in tokenized sentence."""
    input_ids = tokenizer.encode(sent)
    toks_no_special = tokenizer.encode(sent, add_special_tokens=False)
    has_bos = len(input_ids) > len(toks_no_special) and input_ids[0] != toks_no_special[0] if toks_no_special else False
    bos_offset = 1 if has_bos else 0
    
    for prefix in ['', ' ']:
        target_ids = tokenizer.encode(prefix + target_word, add_special_tokens=False)
        if not target_ids:
            continue
        for i in range(len(toks_no_special) - len(target_ids) + 1):
            if toks_no_special[i:i+len(target_ids)] == target_ids:
                return i + bos_offset
    # Fuzzy match
    for i in range(len(toks_no_special) - 1, -1, -1):
        decoded = tokenizer.decode([toks_no_special[i]]).strip().lower()
        if target_word.lower() in decoded:
            return i + bos_offset
    return -1


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


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
    if len(sys.argv) < 2:
        print("Usage: python phase946_operator_algebra.py <qwen3|glm4|deepseek7b>")
        sys.exit(1)
    
    model_key = sys.argv[1]
    if model_key not in MODEL_CONFIGS:
        print(f"Unknown model: {model_key}")
        sys.exit(1)
    
    global _log_file
    _log_file = str(TMP_DIR / f"phase946_{model_key}.log")
    log(f"Phase 946: Operator Algebra -- {model_key}")
    
    stimuli = build_stimuli()
    log(f"Total stimuli: {len(stimuli)}")
    
    # Count by op_type
    type_counts = defaultdict(int)
    for s in stimuli:
        type_counts[s["op_type"]] += 1
    log(f"  by op_type: {dict(type_counts)}")
    
    # Load model
    model, tokenizer = load_model_bf16(model_key)
    n_layers = len(get_layers(model))
    d_model = model.config.hidden_size
    info = get_model_info(model, model_key)
    mid = n_layers // 2
    log(f"  n_layers={n_layers}, d_model={d_model}, mid={mid}")
    
    sample_layers = sorted(set(list(range(max(1, mid-5), min(n_layers-1, mid+6), 2)) + [mid]))
    log(f"  sample_layers={sample_layers}")
    
    # ---- Capture all unique sentences ----
    unique_sents = sorted(set(s["sentence"] for s in stimuli))
    log(f"Capturing {len(unique_sents)} unique sentences...")
    
    all_caps = {}
    t0 = time.time()
    for i, sent in enumerate(unique_sents):
        all_caps[sent] = _capture_single(model, tokenizer, sent)
        if (i+1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (len(unique_sents) - i - 1) / rate
            log(f"  {i+1}/{len(unique_sents)} ({rate:.1f}/s) ETA={eta:.0f}s GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
            gc.collect()
            torch.cuda.empty_cache()
    log(f"Done capturing in {time.time()-t0:.0f}s")
    
    # ---- Extract hidden states at target positions ----
    log("Extracting hidden states at target positions...")
    h_store = defaultdict(lambda: defaultdict(list))  # {key: {layer: [np.array]}}
    n_miss = 0
    n_ok = 0
    
    for stim in stimuli:
        sent = stim["sentence"]
        if sent not in all_caps:
            n_miss += 1; continue
        pos = find_target_pos(tokenizer, sent, stim["target_word"])
        if pos < 0:
            n_miss += 1; continue
        cap = all_caps[sent]
        for layer in sample_layers:
            if pos < cap["hidden"][layer].shape[1]:
                h = cap["hidden"][layer][0, pos, :].numpy()
                # Build key: op_type + operator + operand + condition
                key = (stim["op_type"], stim["operator"], stim["operand"], stim.get("frame", ""))
                h_store[key][layer].append(h)
        n_ok += 1
    
    log(f"  Extracted: ok={n_ok}, miss={n_miss}")
    
    # Average over repetitions for each key
    avg_h = defaultdict(dict)
    for key, layers in h_store.items():
        for layer, h_list in layers.items():
            avg_h[key][layer] = np.mean(h_list, axis=0)
    log(f"  Averaged {len(avg_h)} (type, op, operand, frame) combinations")
    
    # ================================================================
    # ANALYSIS 1: 否定算子基线复现 (Negation Operator Reproducibility)
    # ================================================================
    log(f"\n{'='*60}")
    log("ANALYSIS 1: Negation Operator Baseline")
    log(f"{'='*60}")
    
    neg_deltas = defaultdict(lambda: defaultdict(list))  # {operand: {layer: [delta]}}
    for stim in stimuli:
        if stim["op_type"] != "negation":
            continue
        op, frame, cond = stim["operand"], stim["frame"], stim["condition"]
        key_aff = ("negation", "affirm", op, frame)
        key_neg = ("negation", "not", op, frame)
        for layer in sample_layers:
            if layer in avg_h.get(key_aff, {}) and layer in avg_h.get(key_neg, {}):
                delta = avg_h[key_neg][layer] - avg_h[key_aff][layer]
                neg_deltas[op][layer].append(delta)
    
    # Per-operand average delta
    avg_neg_delta = defaultdict(dict)
    for op, layers in neg_deltas.items():
        for layer, deltas in layers.items():
            avg_neg_delta[op][layer] = np.mean(deltas, axis=0)
    
    # LOO cosine
    neg_loo = {}
    for layer in sample_layers:
        deltas = []
        op_labels = []
        for op in sorted(avg_neg_delta.keys()):
            if layer in avg_neg_delta[op]:
                deltas.append(avg_neg_delta[op][layer])
                op_labels.append(op)
        
        if len(deltas) < 3:
            continue
        
        cos_list = []
        for i in range(len(deltas)):
            rest = np.mean([d for j, d in enumerate(deltas) if j != i], axis=0)
            cos_list.append(cosine_sim(deltas[i], rest))
        
        neg_loo[layer] = {
            "avg_loo_cos": float(np.mean(cos_list)),
            "std_loo_cos": float(np.std(cos_list)),
            "min_loo_cos": float(np.min(cos_list)),
            "n_operands": len(cos_list),
        }
        log(f"  L{layer}: NEG LOO_cos = {neg_loo[layer]['avg_loo_cos']:.4f} ± {neg_loo[layer]['std_loo_cos']:.4f}")
    
    # Negation PCA
    neg_pca = {}
    for layer in sample_layers:
        deltas = [avg_neg_delta[op][layer] for op in sorted(avg_neg_delta.keys()) if layer in avg_neg_delta[op]]
        if len(deltas) < 3:
            continue
        mat = np.array(deltas) - np.mean(deltas, axis=0)
        try:
            U, S, Vt = np.linalg.svd(mat, full_matrices=False)
            total_var = np.sum(S**2)
            cumvar = np.cumsum(S**2) / total_var
            dim50 = int(np.searchsorted(cumvar, 0.5)) + 1
            dim80 = int(np.searchsorted(cumvar, 0.8)) + 1
            neg_pca[layer] = {
                "top1_var": float(S[0]**2 / total_var),
                "dim50": dim50, "dim80": dim80,
                "n_operands": len(deltas),
            }
            log(f"  L{layer}: NEG PCA top1={neg_pca[layer]['top1_var']:.2%}, dim50={dim50}, dim80={dim80}")
        except Exception as e:
            log(f"  L{layer} PCA failed: {e}")
    
    # ================================================================
    # ANALYSIS 2: 量化算子方向 (Quantification Operators)
    # ================================================================
    log(f"\n{'='*60}")
    log("ANALYSIS 2: Quantification Operators")
    log(f"{'='*60}")
    
    # Extract operator deltas between quantifier pairs
    quant_deltas = {}  # {(q_from, q_to): {layer: [delta]}}
    for (q_from, q_to) in QUANT_PAIRS:
        quant_deltas[(q_from, q_to)] = defaultdict(list)
    
    for stim in stimuli:
        if stim["op_type"] != "quantification":
            continue
        q = stim["operator"]
        operand = stim["operand"]
        frame = stim["frame"]
        key_q = ("quantification", q, operand, frame)
        
        for (q_from, q_to) in QUANT_PAIRS:
            key_from = ("quantification", q_from, operand, frame)
            key_to = ("quantification", q_to, operand, frame)
            if q == q_to and q_from in [s["operator"] for s in stimuli if s["op_type"]=="quantification" 
                                         and s["operand"]==operand and s["frame"]==frame]:
                for layer in sample_layers:
                    if layer in avg_h.get(key_from, {}) and layer in avg_h.get(key_to, {}):
                        delta = avg_h[key_to][layer] - avg_h[key_from][layer]
                        quant_deltas[(q_from, q_to)][layer].append(delta)
    
    # Average quant deltas per layer
    avg_quant_delta = {}  # {(q_from, q_to): {layer: avg_delta}}
    for pair, layers in quant_deltas.items():
        avg_quant_delta[pair] = {}
        for layer, deltas in layers.items():
            if deltas:
                avg_quant_delta[pair][layer] = np.mean(deltas, axis=0)
    
    # Quantifier LOO cosine (per quantifier direction, across operands)
    quant_loo = {}
    for layer in sample_layers:
        results = {}
        for pair in QUANT_PAIRS:
            if pair in avg_quant_delta and layer in avg_quant_delta[pair]:
                results[pair] = {"norm": float(np.linalg.norm(avg_quant_delta[pair][layer]))}
        quant_loo[layer] = results
    
    # Cross-quantifier cosine matrix
    quant_cross_cos = {}
    for layer in sample_layers:
        valid_pairs = [p for p in QUANT_PAIRS if p in avg_quant_delta and layer in avg_quant_delta[p]]
        if len(valid_pairs) >= 2:
            matrix = {}
            for pi in valid_pairs:
                for pj in valid_pairs:
                    c = cosine_sim(avg_quant_delta[pi][layer], avg_quant_delta[pj][layer])
                    if pi not in matrix:
                        matrix[str(pi)] = {}
                    matrix[str(pi)][str(pj)] = float(c)
            quant_cross_cos[layer] = matrix
            log(f"  L{layer}: Quant cross-cos computed for {len(valid_pairs)} pairs")
            
            # Show some key cross-pairs
            for pi in valid_pairs:
                for pj in valid_pairs:
                    if pi != pj:
                        c = cosine_sim(avg_quant_delta[pi][layer], avg_quant_delta[pj][layer])
                        log(f"    cos({pi[0]}→{pi[1]}, {pj[0]}→{pj[1]}) = {c:+.4f}")
    
    # ================================================================
    # ANALYSIS 3: 模态算子方向 (Modal Operators)
    # ================================================================
    log(f"\n{'='*60}")
    log("ANALYSIS 3: Modal Operators")
    log(f"{'='*60}")
    
    modal_deltas = {}
    for (m_from, m_to) in MODAL_PAIRS:
        modal_deltas[(m_from, m_to)] = defaultdict(list)
    
    for stim in stimuli:
        if stim["op_type"] != "modal":
            continue
        m = stim["operator"]
        operand = stim["operand"]
        frame = stim["frame"]
        key_m = ("modal", m, operand, frame)
        
        for (m_from, m_to) in MODAL_PAIRS:
            key_from = ("modal", m_from, operand, frame)
            key_to = ("modal", m_to, operand, frame)
            if m == m_to:
                for layer in sample_layers:
                    if layer in avg_h.get(key_from, {}) and layer in avg_h.get(key_to, {}):
                        delta = avg_h[key_to][layer] - avg_h[key_from][layer]
                        modal_deltas[(m_from, m_to)][layer].append(delta)
    
    avg_modal_delta = {}
    for pair, layers in modal_deltas.items():
        avg_modal_delta[pair] = {}
        for layer, deltas in layers.items():
            if deltas:
                avg_modal_delta[pair][layer] = np.mean(deltas, axis=0)
    
    # Modal cross-cosine
    modal_cross_cos = {}
    for layer in sample_layers:
        valid_pairs = [p for p in MODAL_PAIRS if p in avg_modal_delta and layer in avg_modal_delta[p]]
        if len(valid_pairs) >= 2:
            matrix = {}
            for pi in valid_pairs:
                for pj in valid_pairs:
                    c = cosine_sim(avg_modal_delta[pi][layer], avg_modal_delta[pj][layer])
                    if pi not in matrix:
                        matrix[str(pi)] = {}
                    matrix[str(pi)][str(pj)] = float(c)
            modal_cross_cos[layer] = matrix
            
            for pi in valid_pairs:
                for pj in valid_pairs:
                    if pi != pj:
                        c = cosine_sim(avg_modal_delta[pi][layer], avg_modal_delta[pj][layer])
                        log(f"    cos({pi[0]}→{pi[1]}, {pj[0]}→{pj[1]}) = {c:+.4f}")
    
    # ================================================================
    # ANALYSIS 4: 双否定闭合 (Double Negation Closure)
    # ================================================================
    log(f"\n{'='*60}")
    log("ANALYSIS 4: Double Negation Closure (¬¬P ≈ P?)")
    log(f"{'='*60}")
    
    dn_closure = {}
    for layer in sample_layers:
        closure_cos = []
        for adj in DOUBLE_NEG_ADJ:
            key_aff = ("double_neg", "affirm", adj, "dn")
            key_dn = ("double_neg", "not_not", adj, "dn")
            if layer in avg_h.get(key_aff, {}) and layer in avg_h.get(key_dn, {}):
                c = cosine_sim(avg_h[key_aff][layer], avg_h[key_dn][layer])
                closure_cos.append(c)
        
        if closure_cos:
            dn_closure[layer] = {
                "avg_cos": float(np.mean(closure_cos)),
                "std_cos": float(np.std(closure_cos)),
                "min_cos": float(np.min(closure_cos)),
                "pos_rate": float(sum(1 for c in closure_cos if c > 0) / len(closure_cos)),
            }
            log(f"  L{layer}: ¬¬P cos = {dn_closure[layer]['avg_cos']:.4f} ± {dn_closure[layer]['std_cos']:.4f}, "
                f"pos={dn_closure[layer]['pos_rate']:.0%}")
    
    # Not-antonym closure (not unhappy ≈ happy?)
    ant_closure = {}
    for layer in sample_layers:
        closure_cos = []
        for adj in DOUBLE_NEG_ADJ:
            antonyms_map = {"happy": "sad", "sad": "happy", "big": "small", "small": "big",
                          "good": "bad", "bad": "good", "warm": "cold", "cold": "warm",
                          "fast": "slow", "slow": "fast", "bright": "dark", "dark": "bright",
                          "safe": "dangerous", "dangerous": "safe", "clean": "dirty", "dirty": "clean"}
            if adj not in antonyms_map:
                continue
            ant = antonyms_map[adj]
            key_aff = ("double_neg", "affirm", adj, "dn")
            key_not_ant = ("double_neg", "not_antonym", f"ant_{ant}", "dn")
            if layer in avg_h.get(key_aff, {}) and layer in avg_h.get(key_not_ant, {}):
                c = cosine_sim(avg_h[key_aff][layer], avg_h[key_not_ant][layer])
                closure_cos.append(c)
        
        if closure_cos:
            ant_closure[layer] = {
                "avg_cos": float(np.mean(closure_cos)),
                "pos_rate": float(sum(1 for c in closure_cos if c > 0) / len(closure_cos)),
            }
            log(f"  L{layer}: not-antonym affinity cos = {ant_closure[layer]['avg_cos']:.4f}")
    
    # ================================================================
    # ANALYSIS 5: 算子组合代数 (Operator Composition)
    # ================================================================
    log(f"\n{'='*60}")
    log("ANALYSIS 5: Operator Composition Algebra")
    log(f"{'='*60}")
    
    # Test: O(not_all) ≈ O(not) + O(all→some)? De Morgan intuition
    combo_results = {}
    for layer in sample_layers:
        # Build operator "basis" deltas for this layer
        # O(not) direction: average negation delta
        neg_deltas_layer = [avg_neg_delta[op][layer] for op in avg_neg_delta if layer in avg_neg_delta[op]]
        O_not = np.mean(neg_deltas_layer, axis=0) if neg_deltas_layer else np.zeros(d_model)
        
        # O(all→some): from quant deltas
        O_all_some = np.zeros(d_model)
        if ("all", "some") in avg_quant_delta and layer in avg_quant_delta[("all", "some")]:
            O_all_some = avg_quant_delta[("all", "some")][layer]
        
        # O(must→may): from modal deltas
        O_must_may = np.zeros(d_model)
        if ("must", "may") in avg_modal_delta and layer in avg_modal_delta[("must", "may")]:
            O_must_may = avg_modal_delta[("must", "may")][layer]
        
        # Extract compound operator directions
        compound_ops = {}
        for key, layers_dict in avg_h.items():
            op_type, operator, operand, frame = key
            if op_type == "combo" and operator in ["not_all", "not_must"]:
                # Find base operator for comparison
                if operator == "not_all":
                    base_key = ("combo", "all", operand, frame)
                else:
                    base_key = ("combo", "must", operand, frame)
                
                if layer in layers_dict and layer in avg_h.get(base_key, {}):
                    compound_delta = avg_h[key][layer] - avg_h[base_key][layer]
                    compound_ops[operator] = compound_delta
        
        # Compute composition predictions
        layer_combo = {}
        for comp_name, comp_delta in compound_ops.items():
            if comp_name == "not_all":
                # Hypothesis: O(not_all) ≈ O(not) (De Morgan: not-all ≈ some-not)
                pred = O_not  # simplest model: just not
                c_pred = cosine_sim(comp_delta, pred)
                # Alternate: not-all ≈ all→some + not
                alt_pred = O_all_some + O_not
                c_alt = cosine_sim(comp_delta, alt_pred)
                layer_combo["not_all"] = {
                    "cos_with_O_not": float(c_pred),
                    "cos_with_O_not_plus_all_some": float(c_alt),
                }
                log(f"    not_all: cos(O_not)={c_pred:+.4f}, cos(O_not+O_all→some)={c_alt:+.4f}")
            
            elif comp_name == "not_must":
                pred = O_not
                c_pred = cosine_sim(comp_delta, pred)
                alt_pred = O_must_may + O_not
                c_alt = cosine_sim(comp_delta, alt_pred)
                layer_combo["not_must"] = {
                    "cos_with_O_not": float(c_pred),
                    "cos_with_O_not_plus_must_may": float(c_alt),
                }
                log(f"    not_must: cos(O_not)={c_pred:+.4f}, cos(O_not+O_must→may)={c_alt:+.4f}")
        
        combo_results[layer] = layer_combo
    
    # ================================================================
    # ANALYSIS 6: 全局算子间距离矩阵 (Cross-Operator Distance Matrix)
    # ================================================================
    log(f"\n{'='*60}")
    log("ANALYSIS 6: Cross-Operator Affinity Matrix")
    log(f"{'='*60}")
    
    # Collect all operator-average directions
    op_matrix = {}
    for layer in sample_layers:
        # Negation
        neg_all = np.mean([avg_neg_delta[op][layer] for op in avg_neg_delta if layer in avg_neg_delta[op]], axis=0) \
            if [op for op in avg_neg_delta if layer in avg_neg_delta[op]] else None
        
        # Quantification pairs
        quant_ops = {}
        for pair in QUANT_PAIRS:
            if pair in avg_quant_delta and layer in avg_quant_delta[pair]:
                quant_ops[f"Q({pair[0]}→{pair[1]})"] = avg_quant_delta[pair][layer]
        
        # Modal pairs
        modal_ops = {}
        for pair in MODAL_PAIRS:
            if pair in avg_modal_delta and layer in avg_modal_delta[pair]:
                modal_ops[f"M({pair[0]}→{pair[1]})"] = avg_modal_delta[pair][layer]
        
        # Combine all
        all_ops = {}
        if neg_all is not None:
            all_ops["NEG(not)"] = neg_all
        all_ops.update(quant_ops)
        all_ops.update(modal_ops)
        
        if len(all_ops) >= 2:
            # Build cosine matrix
            op_names = sorted(all_ops.keys())
            cos_mat = np.zeros((len(op_names), len(op_names)))
            for i, ni in enumerate(op_names):
                for j, nj in enumerate(op_names):
                    cos_mat[i, j] = cosine_sim(all_ops[ni], all_ops[nj])
            
            op_matrix[layer] = {
                "op_names": op_names,
                "cosine_matrix": cos_mat.tolist(),
            }
            
            # Log key affinities
            log(f"  L{layer} operator affinity (N={len(op_names)}):")
            for i, ni in enumerate(op_names):
                others = [(cos_mat[i,j], op_names[j]) for j in range(len(op_names)) if j != i]
                others.sort(reverse=True)
                top3 = others[:3]
                log(f"    {ni:20s}: top3={[(f'{c:+.3f}', n) for c,n in top3]}")
    
    # ================================================================
    # ANALYSIS 7: 跨角色算子共享 (Cross-Role Operator Sharing)
    # ================================================================
    log(f"\n{'='*60}")
    log("ANALYSIS 7: Cross-Role Negation (adj vs verb vs noun)")
    log(f"{'='*60}")
    
    cross_role_neg = {}
    for layer in sample_layers:
        if layer not in neg_loo:
            continue
        # All negation deltas
        all_neg = [avg_neg_delta[op][layer] for op in avg_neg_delta if layer in avg_neg_delta[op]]
        if not all_neg:
            continue
        
        neg_mean = np.mean(all_neg, axis=0)
        # Per-operand cosine with global mean
        per_op_cos = [cosine_sim(d, neg_mean) for d in all_neg]
        cross_role_neg[layer] = {
            "avg_cos_with_global": float(np.mean(per_op_cos)),
            "std_cos_with_global": float(np.std(per_op_cos)),
            "min_cos": float(np.min(per_op_cos)),
        }
        log(f"  L{layer}: per-op cos with global O(not) = {cross_role_neg[layer]['avg_cos_with_global']:.4f} "
            f"± {cross_role_neg[layer]['std_cos_with_global']:.4f}")
    
    # ================================================================
    # SAVE & SUMMARY
    # ================================================================
    results = {
        "model": model_key,
        "n_layers": n_layers,
        "d_model": d_model,
        "sample_layers": sample_layers,
        "mid_layer": mid,
        "n_stimuli": len(stimuli),
        "n_unique_sents": len(unique_sents),
        # Negation baseline
        "neg_baseline": {
            "loo_cos": {str(k): v for k, v in neg_loo.items()},
            "pca": {str(k): v for k, v in neg_pca.items()},
        },
        # Quantification
        "quantification": {
            "cross_cos": {str(k): v for k, v in quant_cross_cos.items()},
            "deltas": {str(k): v for k, v in avg_quant_delta.items()},
        },
        # Modal
        "modal": {
            "cross_cos": {str(k): v for k, v in modal_cross_cos.items()},
            "deltas": {str(k): v for k, v in avg_modal_delta.items()},
        },
        # Double negation
        "double_neg_closure": {str(k): v for k, v in dn_closure.items()},
        "antonym_closure": {str(k): v for k, v in ant_closure.items()},
        # Composition
        "composition": {str(k): v for k, v in combo_results.items()},
        # Cross-operator matrix
        "op_affinity_matrix": {str(k): v for k, v in op_matrix.items()},
        # Cross-role
        "cross_role_neg": {str(k): v for k, v in cross_role_neg.items()},
    }
    
    results = make_serializable(results)
    out_path = RESULT_DIR / f"{model_key}_operator_algebra.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"\nSaved to {out_path}")
    
    # ================================================================
    # KEY SUMMARY
    # ================================================================
    log(f"\n{'='*60}")
    log(f"PHASE 946 KEY SUMMARY: {model_key}")
    log(f"{'='*60}")
    
    log(f"\n  1. Negation Baseline (L{mid}):")
    if mid in neg_loo:
        log(f"     LOO_cos = {neg_loo[mid]['avg_loo_cos']:.4f}")
    if mid in neg_pca:
        log(f"     PCA top1={neg_pca[mid]['top1_var']:.2%}, dim50={neg_pca[mid]['dim50']}")
    
    log(f"\n  2. Quantification Operators:")
    if str(mid) in quant_cross_cos:
        log(f"     cross-cosine matrix computed")
    
    log(f"\n  3. Modal Operators:")
    if str(mid) in modal_cross_cos:
        log(f"     cross-cosine matrix computed")
    
    log(f"\n  4. Double Negation Closure (L{mid}):")
    if mid in dn_closure:
        log(f"     cos(¬¬P, P) = {dn_closure[mid]['avg_cos']:.4f}, pos_rate={dn_closure[mid]['pos_rate']:.0%}")
    
    log(f"\n  5. Operator Composition:")
    if mid in combo_results:
        for comp_name, vals in combo_results[mid].items():
            log(f"     {comp_name}: {json.dumps(vals)}")
    
    log(f"\n  6. Cross-Operator Affinity Matrix:")
    if str(mid) in op_matrix:
        ops = op_matrix[str(mid)]["op_names"]
        mat = np.array(op_matrix[str(mid)]["cosine_matrix"])
        log(f"     {len(ops)} operators analyzed")
        # Find strongest inter-family links
        for i, ni in enumerate(ops):
            for j, nj in enumerate(ops):
                if i < j and abs(mat[i,j]) > 0.3:
                    log(f"     strong: cos({ni}, {nj}) = {mat[i,j]:+.4f}")
    
    log(f"\n  7. Cross-Role Negation (L{mid}):")
    if mid in cross_role_neg:
        log(f"     avg_cos_with_global = {cross_role_neg[mid]['avg_cos_with_global']:.4f}")
    
    release_model(model)
    log(f"\nPhase 946 complete for {model_key}!")


if __name__ == "__main__":
    main()
