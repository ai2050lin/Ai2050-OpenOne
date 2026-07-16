"""
Phase 946 Fast: Operator Algebra for GLM4/DS7B (Reduced Stimuli)
==================================================================
精简版：减少帧数从4到2，减少量化谓词从3到1，确保在5分钟内完成。
"""
import sys, os, gc, time, json, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
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

# Reduced stimulus sets (2 frames instead of 4)
NEG_ADJECTIVES = [
    "happy", "sad", "big", "small", "good", "bad",
    "warm", "cold", "fast", "slow", "bright", "dark",
    "safe", "dangerous", "clean", "dirty", "rich", "poor", "strong", "weak",
]
NEG_SUBJECTS = ["person", "dog", "cat", "car", "house", "river",
                "child", "bird", "tree", "road", "song", "food",
                "man", "woman", "city", "book", "student", "worker", "river", "mountain"]
NEG_FRAMES_FAST = {
    "F1": ("the {subj} is {adj}", "the {subj} is not {adj}"),
    "F2": ("the {subj} seems {adj}", "the {subj} does not seem {adj}"),
}

QUANTIFIERS = ["all", "some", "no", "every", "few"]
QUANT_PAIRS = [("some", "all"), ("some", "no"), ("all", "no"), ("some", "every"), ("some", "few")]
QUANT_NOUNS = ["cats", "dogs", "birds", "students", "books", "cars", "trees", "houses"]
# Only 1 predicate per noun (not 3)
QUANT_PREDICATES_FAST = {
    "cats": ["are mammals"],
    "dogs": ["are loyal"],
    "birds": ["can fly"],
    "students": ["work hard"],
    "books": ["contain wisdom"],
    "cars": ["need fuel"],
    "trees": ["provide shade"],
    "houses": ["have roofs"],
}

MODALS = ["must", "can", "should", "may", "might"]
MODAL_PAIRS = [("can", "must"), ("can", "should"), ("can", "may"), ("may", "must"), ("might", "must"), ("should", "must")]
MODAL_VERBS = ["go", "come", "leave", "speak", "listen", "write"]
MODAL_SUBJECTS = ["John", "Mary", "the doctor", "the teacher", "the student", "the manager"]

DOUBLE_NEG_ADJ = NEG_ADJECTIVES[:16]

def build_stimuli_fast():
    stimuli = []
    # Negation (20 adj × 2 frames × 2 cond = 80)
    for i, adj in enumerate(NEG_ADJECTIVES):
        subj = NEG_SUBJECTS[i % len(NEG_SUBJECTS)]
        for flabel, (aff_t, neg_t) in NEG_FRAMES_FAST.items():
            stimuli.append({"sentence": aff_t.format(subj=subj, adj=adj), "target_word": adj, "operand": adj,
                           "operator": "affirm", "op_type": "negation", "frame": flabel, "condition": "affirm"})
            stimuli.append({"sentence": neg_t.format(subj=subj, adj=adj), "target_word": adj, "operand": adj,
                           "operator": "not", "op_type": "negation", "frame": flabel, "condition": "negate"})
    # Quantification (8 nouns × 5 quantifiers × 1 pred = 40)
    for qnoun in QUANT_NOUNS:
        pred = QUANT_PREDICATES_FAST[qnoun][0]
        for q in QUANTIFIERS:
            sent = f"{q} {qnoun} {pred}"
            stimuli.append({"sentence": sent, "target_word": qnoun, "operand": f"{qnoun}_{pred[:15]}",
                           "operator": q, "op_type": "quantification", "frame": "p0", "condition": f"quant_{q}"})
    # Modal (6 verbs × 5 modals = 30)
    for vi, verb in enumerate(MODAL_VERBS):
        subj = MODAL_SUBJECTS[vi]
        for modal in MODALS:
            sent = f"{subj} {modal} {verb}"
            stimuli.append({"sentence": sent, "target_word": verb, "operand": f"{subj}_{verb}",
                           "operator": modal, "op_type": "modal", "frame": "direct", "condition": f"modal_{modal}"})
    # Double negation (16 adj × 3 conditions = 48)
    for adj in DOUBLE_NEG_ADJ:
        stimuli.append({"sentence": f"the person is {adj}", "target_word": adj, "operand": adj,
                       "operator": "affirm", "op_type": "double_neg", "frame": "dn", "condition": "affirm"})
        stimuli.append({"sentence": f"the person is not {adj}", "target_word": adj, "operand": adj,
                       "operator": "not", "op_type": "double_neg", "frame": "dn", "condition": "single_neg"})
        stimuli.append({"sentence": f"the person is not not {adj}", "target_word": adj, "operand": adj,
                       "operator": "not_not", "op_type": "double_neg", "frame": "dn", "condition": "double_neg"})
    # Operator combos (~24)
    for qnoun in QUANT_NOUNS[:3]:
        pred = QUANT_PREDICATES_FAST[qnoun][0]
        stims_combo = [
            (f"all {qnoun} {pred}", "all"),
            (f"not all {qnoun} {pred}", "not_all"),
            (f"some {qnoun} {pred}", "some"),
            (f"some {qnoun} not {pred}", "some_not"),
        ]
        for sent, op in stims_combo:
            stimuli.append({"sentence": sent, "target_word": qnoun, "operand": f"combo_{qnoun}",
                           "operator": op, "op_type": "combo", "frame": "combo", "condition": f"combo_{op}"})
    for vi, verb in enumerate(MODAL_VERBS[:3]):
        subj = MODAL_SUBJECTS[vi]
        stims_mod = [
            (f"{subj} must {verb}", "must"), (f"{subj} not must {verb}", "not_must"),
            (f"{subj} may {verb}", "may"), (f"{subj} may not {verb}", "may_not"),
        ]
        for sent, op in stims_mod:
            stimuli.append({"sentence": sent, "target_word": verb, "operand": f"combo_m_{subj}_{verb}",
                           "operator": op, "op_type": "combo", "frame": "combo_m", "condition": f"combo_{op}"})
    return stimuli

def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16 fast)...")
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = None
    for attn in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(cfg["path"], torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True, local_files_only=True, attn_implementation=attn)
            log(f"  attn={attn} OK"); break
        except: pass
    if model is None: raise RuntimeError(f"Load failed {model_name}")
    model.eval()
    return model, tok

def _capture_single(model, tokenizer, sent, max_len=64):
    dev = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad(): out = model(**inputs, output_hidden_states=True)
    hs = {li: h.detach().cpu().float() for li, h in enumerate(out.hidden_states)}
    return {"hidden": hs}

def find_target_pos(tokenizer, sent, target_word):
    ids = tokenizer.encode(sent)
    no_special = tokenizer.encode(sent, add_special_tokens=False)
    bos_off = 1 if len(ids) > len(no_special) and ids[0] != no_special[0] else 0
    for pref in ['', ' ']:
        tids = tokenizer.encode(pref + target_word, add_special_tokens=False)
        if not tids: continue
        for i in range(len(no_special) - len(tids) + 1):
            if no_special[i:i+len(tids)] == tids: return i + bos_off
    for i in range(len(no_special)-1, -1, -1):
        if target_word.lower() in tokenizer.decode([no_special[i]]).strip().lower(): return i + bos_off
    return -1

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-10 or nb < 1e-10 else float(np.dot(a, b) / (na * nb))

def make_serializable(obj):
    if isinstance(obj, dict): return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [make_serializable(x) for x in obj]
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)): return float(obj)
    elif isinstance(obj, (np.int32, np.int64)): return int(obj)
    return obj

def main():
    model_key = sys.argv[1]
    global _log_file
    _log_file = str(TMP_DIR / f"phase946_fast_{model_key}.log")
    log(f"Phase 946 Fast: {model_key}")
    
    stimuli = build_stimuli_fast()
    type_counts = defaultdict(int)
    for s in stimuli: type_counts[s["op_type"]] += 1
    log(f"Total stimuli: {len(stimuli)}, by type: {dict(type_counts)}")
    
    model, tokenizer = load_model_bf16(model_key)
    n_layers = len(get_layers(model))
    d_model = model.config.hidden_size
    mid = n_layers // 2
    sample_layers = sorted(set(list(range(max(1, mid-5), min(n_layers-1, mid+6), 2)) + [mid]))
    log(f"n_layers={n_layers}, d_model={d_model}, mid={mid}, sample={sample_layers}")
    
    unique_sents = sorted(set(s["sentence"] for s in stimuli))
    log(f"Capturing {len(unique_sents)} unique sentences...")
    
    all_caps = {}
    t0 = time.time()
    for i, sent in enumerate(unique_sents):
        all_caps[sent] = _capture_single(model, tokenizer, sent)
        if (i+1) % 40 == 0:
            el = time.time() - t0
            rate = (i+1) / max(el, 1)
            eta = (len(unique_sents) - i - 1) / rate
            log(f"  {i+1}/{len(unique_sents)} ({rate:.1f}/s) ETA={eta:.0f}s")
            gc.collect(); torch.cuda.empty_cache()
    log(f"Capture done in {time.time()-t0:.0f}s")
    
    # Extract hidden states
    log("Extracting...")
    h_store = defaultdict(lambda: defaultdict(list))
    n_ok, n_miss = 0, 0
    for stim in stimuli:
        sent = stim["sentence"]
        if sent not in all_caps: n_miss += 1; continue
        pos = find_target_pos(tokenizer, sent, stim["target_word"])
        if pos < 0: n_miss += 1; continue
        cap = all_caps[sent]
        for layer in sample_layers:
            if pos < cap["hidden"][layer].shape[1]:
                h_store[(stim["op_type"], stim["operator"], stim["operand"], stim.get("frame",""))][layer].append(
                    cap["hidden"][layer][0, pos, :].numpy())
        n_ok += 1
    log(f"Extracted: ok={n_ok}, miss={n_miss}")
    
    avg_h = defaultdict(dict)
    for key, layers in h_store.items():
        for layer, h_list in layers.items():
            avg_h[key][layer] = np.mean(h_list, axis=0)
    
    # === ANALYSIS 1: Negation LOO ===
    log("\n=== ANALYSIS 1: Negation LOO ===")
    neg_deltas = defaultdict(lambda: defaultdict(list))
    for stim in stimuli:
        if stim["op_type"] != "negation": continue
        op, frame, cond = stim["operand"], stim["frame"], stim["condition"]
        key_aff = ("negation", "affirm", op, frame)
        key_neg = ("negation", "not", op, frame)
        for layer in sample_layers:
            if layer in avg_h.get(key_aff,{}) and layer in avg_h.get(key_neg,{}):
                neg_deltas[op][layer].append(avg_h[key_neg][layer] - avg_h[key_aff][layer])
    
    avg_neg_delta = defaultdict(dict)
    for op, layers in neg_deltas.items():
        for layer, deltas in layers.items():
            avg_neg_delta[op][layer] = np.mean(deltas, axis=0)
    
    neg_loo = {}
    for layer in sample_layers:
        deltas = [avg_neg_delta[op][layer] for op in sorted(avg_neg_delta.keys()) if layer in avg_neg_delta[op]]
        if len(deltas) < 3: continue
        cos_list = [cosine_sim(deltas[i], np.mean([d for j,d in enumerate(deltas) if j!=i], axis=0)) for i in range(len(deltas))]
        neg_loo[layer] = {"avg_loo_cos": float(np.mean(cos_list)), "std_loo_cos": float(np.std(cos_list)), "n_operands": len(cos_list)}
        log(f"  L{layer}: NEG LOO_cos = {neg_loo[layer]['avg_loo_cos']:.4f} ± {neg_loo[layer]['std_loo_cos']:.4f}")
    
    # Neg PCA
    neg_pca = {}
    for layer in sample_layers:
        deltas = [avg_neg_delta[op][layer] for op in sorted(avg_neg_delta.keys()) if layer in avg_neg_delta[op]]
        if len(deltas) < 3: continue
        mat = np.array(deltas) - np.mean(deltas, axis=0)
        try:
            U, S, Vt = np.linalg.svd(mat, full_matrices=False)
            total_var = np.sum(S**2)
            cumvar = np.cumsum(S**2) / total_var
            neg_pca[layer] = {"top1_var": float(S[0]**2/total_var), "dim50": int(np.searchsorted(cumvar,0.5))+1,
                              "dim80": int(np.searchsorted(cumvar,0.8))+1, "n_operands": len(deltas)}
            log(f"  L{layer}: NEG PCA top1={neg_pca[layer]['top1_var']:.2%}, dim50={neg_pca[layer]['dim50']}")
        except: pass
    
    # === ANALYSIS 2: Double Negation Closure ===
    log("\n=== ANALYSIS 2: Double Negation Closure ===")
    dn_closure = {}
    for layer in sample_layers:
        vals = []
        for adj in DOUBLE_NEG_ADJ:
            key_aff = ("double_neg", "affirm", adj, "dn")
            key_dn = ("double_neg", "not_not", adj, "dn")
            if layer in avg_h.get(key_aff,{}) and layer in avg_h.get(key_dn,{}):
                vals.append(cosine_sim(avg_h[key_aff][layer], avg_h[key_dn][layer]))
        if vals:
            dn_closure[layer] = {"avg_cos": float(np.mean(vals)), "std_cos": float(np.std(vals)), "pos_rate": float(sum(1 for c in vals if c>0)/len(vals))}
            log(f"  L{layer}: ¬¬P cos={dn_closure[layer]['avg_cos']:.4f}, pos_rate={dn_closure[layer]['pos_rate']:.0%}")
    
    # === ANALYSIS 3: Cross-Operator Affinity Matrix ===
    log("\n=== ANALYSIS 3: Cross-Operator Affinity ===")
    
    # Quant deltas
    quant_deltas = {}
    for (q_from, q_to) in QUANT_PAIRS:
        quant_deltas[(q_from, q_to)] = defaultdict(list)
    for stim in stimuli:
        if stim["op_type"] != "quantification": continue
        q, operand, frame = stim["operator"], stim["operand"], stim["frame"]
        key_q = ("quantification", q, operand, frame)
        for (q_from, q_to) in QUANT_PAIRS:
            key_from = ("quantification", q_from, operand, frame)
            key_to = ("quantification", q_to, operand, frame)
            if q == q_to:
                for layer in sample_layers:
                    if layer in avg_h.get(key_from,{}) and layer in avg_h.get(key_to,{}):
                        quant_deltas[(q_from,q_to)][layer].append(avg_h[key_to][layer] - avg_h[key_from][layer])
    avg_quant_delta = {}
    for pair, layers in quant_deltas.items():
        avg_quant_delta[pair] = {}
        for layer, deltas in layers.items():
            if deltas: avg_quant_delta[pair][layer] = np.mean(deltas, axis=0)
    
    # Modal deltas
    modal_deltas = {}
    for (m_from, m_to) in MODAL_PAIRS:
        modal_deltas[(m_from, m_to)] = defaultdict(list)
    for stim in stimuli:
        if stim["op_type"] != "modal": continue
        m, operand, frame = stim["operator"], stim["operand"], stim["frame"]
        key_m = ("modal", m, operand, frame)
        for (m_from, m_to) in MODAL_PAIRS:
            key_from = ("modal", m_from, operand, frame)
            key_to = ("modal", m_to, operand, frame)
            if m == m_to:
                for layer in sample_layers:
                    if layer in avg_h.get(key_from,{}) and layer in avg_h.get(key_to,{}):
                        modal_deltas[(m_from,m_to)][layer].append(avg_h[key_to][layer] - avg_h[key_from][layer])
    avg_modal_delta = {}
    for pair, layers in modal_deltas.items():
        avg_modal_delta[pair] = {}
        for layer, deltas in layers.items():
            if deltas: avg_modal_delta[pair][layer] = np.mean(deltas, axis=0)
    
    # Build operator affinity matrix
    op_matrix = {}
    for layer in sample_layers:
        neg_all = np.mean([avg_neg_delta[op][layer] for op in avg_neg_delta if layer in avg_neg_delta[op]], axis=0) if [op for op in avg_neg_delta if layer in avg_neg_delta[op]] else None
        
        quant_ops = {}
        for pair in QUANT_PAIRS:
            if pair in avg_quant_delta and layer in avg_quant_delta[pair]:
                quant_ops[f"Q({pair[0]}→{pair[1]})"] = avg_quant_delta[pair][layer]
        
        modal_ops = {}
        for pair in MODAL_PAIRS:
            if pair in avg_modal_delta and layer in avg_modal_delta[pair]:
                modal_ops[f"M({pair[0]}→{pair[1]})"] = avg_modal_delta[pair][layer]
        
        all_ops = {}
        if neg_all is not None: all_ops["NEG(not)"] = neg_all
        all_ops.update(quant_ops); all_ops.update(modal_ops)
        
        if len(all_ops) >= 2:
            op_names = sorted(all_ops.keys())
            cos_mat = np.zeros((len(op_names), len(op_names)))
            for i, ni in enumerate(op_names):
                for j, nj in enumerate(op_names):
                    cos_mat[i,j] = cosine_sim(all_ops[ni], all_ops[nj])
            op_matrix[layer] = {"op_names": op_names, "cosine_matrix": cos_mat.tolist()}
            
            log(f"  L{layer} ops ({len(op_names)}):")
            for i, ni in enumerate(op_names):
                others = [(cos_mat[i,j], op_names[j]) for j in range(len(op_names)) if j!=i]
                others.sort(reverse=True)
                log(f"    {ni:22s}: top3={[(f'{c:+.3f}',n) for c,n in others[:3]]}")
    
    # === ANALYSIS 4: Composition ===
    log("\n=== ANALYSIS 4: Operator Composition ===")
    combo_results = {}
    for layer in sample_layers:
        neg_deltas_layer = [avg_neg_delta[op][layer] for op in avg_neg_delta if layer in avg_neg_delta[op]]
        O_not = np.mean(neg_deltas_layer, axis=0) if neg_deltas_layer else np.zeros(d_model)
        
        compound_ops = {}
        for key, layers_dict in avg_h.items():
            op_type, operator, operand, frame = key
            if op_type != "combo" or operator not in ["not_all", "not_must"]: continue
            base_key = ("combo", "all" if operator=="not_all" else "must", operand, frame)
            if layer in layers_dict and layer in avg_h.get(base_key, {}):
                compound_ops[operator] = avg_h[key][layer] - avg_h[base_key][layer]
        
        layer_combo = {}
        for comp_name, comp_delta in compound_ops.items():
            c_pred = cosine_sim(comp_delta, O_not)
            layer_combo[comp_name] = {"cos_with_O_not": float(c_pred)}
            log(f"    {comp_name}: cos(O_not)={c_pred:+.4f}")
        combo_results[layer] = layer_combo
    
    # === SAVE ===
    results = make_serializable({
        "model": model_key, "n_layers": n_layers, "d_model": d_model, "mid_layer": mid,
        "sample_layers": sample_layers, "n_stimuli": len(stimuli), "n_unique_sents": len(unique_sents),
        "neg_baseline": {"loo_cos": {str(k): v for k, v in neg_loo.items()}, "pca": {str(k): v for k, v in neg_pca.items()}},
        "double_neg_closure": {str(k): v for k, v in dn_closure.items()},
        "op_affinity_matrix": {str(k): v for k, v in op_matrix.items()},
        "composition": {str(k): v for k, v in combo_results.items()},
    })
    out_path = RESULT_DIR / f"{model_key}_operator_algebra.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"\nSaved to {out_path}")
    
    # Summary
    log(f"\n{'='*50}")
    log(f"SUMMARY L{mid}:")
    if mid in neg_loo: log(f"  NEG LOO: {neg_loo[mid]['avg_loo_cos']:.4f}")
    if mid in neg_pca: log(f"  NEG PCA: top1={neg_pca[mid]['top1_var']:.2%}, dim50={neg_pca[mid]['dim50']}")
    if mid in dn_closure: log(f"  ¬¬P cos: {dn_closure[mid]['avg_cos']:.4f}")
    
    release_model(model)
    log("Done!")

if __name__ == "__main__":
    main()
