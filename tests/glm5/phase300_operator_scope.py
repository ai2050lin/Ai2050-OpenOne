"""
Phase 300: Operator-Scope Decomposition (Simplified & Robust)
==============================================================
Core questions:
1. Does "not" have a unified operator direction across operands?
2. Is "not happy" = "sad" direction? Or is negation orthogonal to antonym?
3. Cross-operand LOO for operator direction
4. Norm statistics of operator increments

Stimulus: 
- 16 adjectives × 4 frames × 2 conditions (affirm/negate) = 128 negation sentences
- 12 antonym pairs × 4 frames × 3 conditions (A/B/not-A) = 144 antonym sentences
- Total: ~272 sentences

Usage:
  python tests/glm5/phase300_operator_scope.py qwen3
  python tests/glm5/phase300_operator_scope.py glm4
  python tests/glm5/phase300_operator_scope.py deepseek7b
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

RESULT_DIR = Path("results/phase300_operator_scope")
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
ANTONYM_PAIRS = [
    ("happy", "sad"), ("big", "small"), ("good", "bad"),
    ("warm", "cold"), ("fast", "slow"), ("bright", "dark"),
    ("open", "closed"), ("safe", "dangerous"), ("clean", "dirty"),
    ("rich", "poor"), ("strong", "weak"), ("young", "old"),
]

EXTRA_NEG_ADJS = ["beautiful", "smart", "quiet", "soft"]

SUBJECTS = ["person", "dog", "cat", "car", "house", "river",
            "child", "bird", "tree", "road", "song", "food",
            "man", "woman", "city", "book"]

FRAMES_NEG = {
    "F1": ("the {subj} is {adj}", "the {subj} is not {adj}"),
    "F2": ("the {subj} seems {adj}", "the {subj} does not seem {adj}"),
    "F3": ("the {subj} remains {adj}", "the {subj} does not remain {adj}"),
    "F4": ("that {subj} is {adj}", "that {subj} is not {adj}"),
}

FRAMES_ANT = {
    "F1": "the person is {adj}",
    "F2": "the dog is {adj}",
    "F3": "the house is {adj}",
    "F4": "that thing is {adj}",
}

def build_stimuli():
    stimuli = []
    all_adjs = [p[0] for p in ANTONYM_PAIRS] + EXTRA_NEG_ADJS
    
    # Part 1: Negation stimuli (16 adj × 4 frames × 2 cond = 128)
    for i, adj in enumerate(all_adjs):
        subj = SUBJECTS[i % len(SUBJECTS)]
        for flabel, (aff_t, neg_t) in FRAMES_NEG.items():
            stimuli.append({
                "sentence": aff_t.format(subj=subj, adj=adj),
                "target_word": adj, "operand": adj,
                "operator": "none", "frame": flabel,
                "condition": "affirm", "group": "negation"
            })
            stimuli.append({
                "sentence": neg_t.format(subj=subj, adj=adj),
                "target_word": adj, "operand": adj,
                "operator": "not", "frame": flabel,
                "condition": "negate", "group": "negation"
            })
    
    # Part 2: Antonym stimuli (12 pairs × 4 frames × 3 cond = 144)
    for adj1, adj2 in ANTONYM_PAIRS:
        for flabel, tmpl in FRAMES_ANT.items():
            # A
            stimuli.append({
                "sentence": tmpl.format(adj=adj1),
                "target_word": adj1, "operand": adj1,
                "operator": "none", "frame": flabel,
                "condition": "antonym_A", "group": "antonym",
                "antonym_pair": (adj1, adj2)
            })
            # B (antonym)
            stimuli.append({
                "sentence": tmpl.format(adj=adj2),
                "target_word": adj2, "operand": adj2,
                "operator": "none", "frame": flabel,
                "condition": "antonym_B", "group": "antonym",
                "antonym_pair": (adj1, adj2)
            })
            # not A
            stimuli.append({
                "sentence": tmpl.format(adj="not " + adj1),
                "target_word": adj1, "operand": adj1,
                "operator": "not", "frame": flabel,
                "condition": "negate_A", "group": "antonym",
                "antonym_pair": (adj1, adj2)
            })
    
    log(f"Total stimuli: {len(stimuli)}")
    return stimuli

# =====================================================================
# MODEL
# =====================================================================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = None
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=attn_impl)
            break
        except Exception as e:
            log(f"  attn_impl={attn_impl} failed: {e}")
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    log(f"  Loaded. GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return model, tok

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
    """Find position of target_word in tokenized sentence. Returns 0-indexed position in hidden states."""
    # Use the same tokenization as model input (with special tokens)
    input_ids = tokenizer.encode(sent)
    toks_no_special = tokenizer.encode(sent, add_special_tokens=False)
    
    # Check if BOS token was added
    has_bos = len(input_ids) > len(toks_no_special) and input_ids[0] != toks_no_special[0] if toks_no_special else False
    bos_offset = 1 if has_bos else 0
    
    # Try exact match with different prefixes
    for prefix in ['', ' ']:
        target_ids = tokenizer.encode(prefix + target_word, add_special_tokens=False)
        if not target_ids:
            continue
        for i in range(len(toks_no_special) - len(target_ids) + 1):
            if toks_no_special[i:i+len(target_ids)] == target_ids:
                return i + bos_offset
    # Fuzzy: find token containing target_word
    for i in range(len(toks_no_special) - 1, -1, -1):
        decoded = tokenizer.decode([toks_no_special[i]]).strip().lower()
        if target_word.lower() in decoded:
            return i + bos_offset
    return -1

# =====================================================================
# MAIN
# =====================================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python phase300_operator_scope.py <qwen3|glm4|deepseek7b>")
        sys.exit(1)
    
    model_key = sys.argv[1]
    if model_key not in MODEL_CONFIGS:
        print(f"Unknown model: {model_key}")
        sys.exit(1)
    
    global _log_file
    _log_file = str(TMP_DIR / f"phase300_{model_key}.log")
    log(f"Phase 300: Operator-Scope Decomposition -- {model_key}")
    
    stimuli = build_stimuli()
    model, tokenizer = load_model_bf16(model_key)
    n_layers = len(get_layers(model))
    d_model = model.config.hidden_size
    info = get_model_info(model, model_key)
    mid = n_layers // 2  # mid layer
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
        if (i+1) % 30 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (len(unique_sents) - i - 1) / rate
            log(f"  {i+1}/{len(unique_sents)} ({rate:.1f}/s) ETA={eta:.0f}s GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    log(f"Done capturing in {time.time()-t0:.0f}s")
    
    # ---- Extract hidden states at target positions ----
    log("Extracting hidden states at target positions...")
    
    # Store: {(operand, frame, condition, group): {layer: [h_list]}}
    h_store = defaultdict(lambda: defaultdict(list))
    
    n_miss_sent = 0
    n_miss_pos = 0
    n_miss_len = 0
    n_ok = 0
    
    for stim in stimuli:
        sent = stim["sentence"]
        if sent not in all_caps:
            n_miss_sent += 1
            continue
        pos = find_target_pos(tokenizer, sent, stim["target_word"])
        if pos < 0:
            n_miss_pos += 1
            continue
        cap = all_caps[sent]
        if pos >= cap["hidden"][0].shape[1]:
            n_miss_len += 1
            continue
        
        key = (stim["operand"], stim["frame"], stim["condition"], stim["group"])
        for layer in sample_layers:
            h = cap["hidden"][layer][0, pos, :].numpy()
            h_store[key][layer].append(h)
        n_ok += 1
    
    log(f"  Extraction stats: ok={n_ok}, miss_sent={n_miss_sent}, miss_pos={n_miss_pos}, miss_len={n_miss_len}")
    
    # Average over repetitions
    avg_h = defaultdict(dict)
    for key, layers in h_store.items():
        for layer, h_list in layers.items():
            avg_h[key][layer] = np.mean(h_list, axis=0)
    
    log(f"  Averaged {len(avg_h)} (operand, frame, condition, group) combinations")
    
    # ---- Analysis 1: Negation operator increment ----
    log("Analysis 1: Negation operator increment...")
    neg_increments = defaultdict(dict)  # {operand: {layer: [delta_per_frame]}}
    
    for stim in stimuli:
        if stim["group"] != "negation":
            continue
        op = stim["operand"]
        frame = stim["frame"]
        
        key_aff = (op, frame, "affirm", "negation")
        key_neg = (op, frame, "negate", "negation")
        
        for layer in sample_layers:
            if layer in avg_h[key_aff] and layer in avg_h[key_neg]:
                delta = avg_h[key_neg][layer] - avg_h[key_aff][layer]
                if layer not in neg_increments[op]:
                    neg_increments[op][layer] = []
                neg_increments[op][layer].append(delta)
    
    # Average per operand
    avg_neg_inc = defaultdict(dict)
    for op, layers in neg_increments.items():
        for layer, deltas in layers.items():
            avg_neg_inc[op][layer] = np.mean(deltas, axis=0)
    
    # ---- Analysis 2: Operator PCA ----
    log("Analysis 2: Operator PCA across operands...")
    op_pca = {}
    for layer in sample_layers:
        all_deltas = []
        op_labels = []
        for op in sorted(avg_neg_inc.keys()):
            if layer in avg_neg_inc[op]:
                all_deltas.append(avg_neg_inc[op][layer])
                op_labels.append(op)
        
        if len(all_deltas) < 3:
            continue
        
        mat = np.array(all_deltas)
        mat_centered = mat - mat.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(mat_centered, full_matrices=False)
            total_var = np.sum(S**2)
            cumvar = np.cumsum(S**2) / total_var
            dim50 = int(np.searchsorted(cumvar, 0.5)) + 1
            dim80 = int(np.searchsorted(cumvar, 0.8)) + 1
            norms = [np.linalg.norm(d) for d in all_deltas]
            
            op_pca[layer] = {
                "top1_var": float(S[0]**2 / total_var),
                "top3_var": float(np.sum(S[:min(3,len(S))]**2) / total_var),
                "dim50": dim50, "dim80": dim80,
                "n_operands": len(all_deltas),
                "norm_mean": float(np.mean(norms)),
                "norm_std": float(np.std(norms)),
                "norm_min": float(np.min(norms)),
                "norm_max": float(np.max(norms)),
                "norm_ratio": float(np.max(norms) / np.min(norms)) if np.min(norms) > 0 else 0,
            }
            log(f"  L{layer}: top1={op_pca[layer]['top1_var']:.1%}, dim50={dim50}, dim80={dim80}, "
                f"norm_range=[{np.min(norms):.1f}, {np.max(norms):.1f}]")
        except Exception as e:
            log(f"  L{layer} PCA failed: {e}")
    
    # ---- Analysis 3: LOO cosine for operator direction ----
    log("Analysis 3: Cross-operand LOO cosine...")
    loo_cos = {}
    for layer in sample_layers:
        all_deltas = []
        op_labels = []
        for op in sorted(avg_neg_inc.keys()):
            if layer in avg_neg_inc[op]:
                all_deltas.append(avg_neg_inc[op][layer])
                op_labels.append(op)
        
        if len(all_deltas) < 3:
            continue
        
        cos_list = []
        for i in range(len(all_deltas)):
            rest = [d for j, d in enumerate(all_deltas) if j != i]
            loo_avg = np.mean(rest, axis=0)
            d_i = all_deltas[i]
            n1, n2 = np.linalg.norm(loo_avg), np.linalg.norm(d_i)
            if n1 > 0.01 and n2 > 0.01:
                cos_list.append(float(np.dot(loo_avg, d_i) / (n1 * n2)))
        
        if cos_list:
            loo_cos[layer] = {
                "avg_loo_cos": float(np.mean(cos_list)),
                "std_loo_cos": float(np.std(cos_list)),
                "min_loo_cos": float(np.min(cos_list)),
                "n_operands": len(cos_list),
            }
            log(f"  L{layer}: avg_LOO_cos={loo_cos[layer]['avg_loo_cos']:.3f} +/- {loo_cos[layer]['std_loo_cos']:.3f}")
    
    # ---- Analysis 4: "not happy" vs "sad" direction comparison ----
    log("Analysis 4: Negation direction vs antonym direction...")
    neg_ant = {}
    for layer in sample_layers:
        comparisons = []
        for adj1, adj2 in ANTONYM_PAIRS:
            # Use F1 frame for antonym group
            key_a = (adj1, "F1", "antonym_A", "antonym")
            key_not_a = (adj1, "F1", "negate_A", "antonym")
            key_b = (adj2, "F1", "antonym_B", "antonym")
            
            if (layer in avg_h[key_a] and layer in avg_h[key_not_a] and layer in avg_h[key_b]):
                h_a = avg_h[key_a][layer]
                h_not = avg_h[key_not_a][layer]
                h_b = avg_h[key_b][layer]
                
                d_neg = h_not - h_a  # not-X direction
                d_ant = h_b - h_a    # antonym direction
                
                n_neg, n_ant = np.linalg.norm(d_neg), np.linalg.norm(d_ant)
                if n_neg > 0.01 and n_ant > 0.01:
                    cos_na = float(np.dot(d_neg, d_ant) / (n_neg * n_ant))
                    # Projection and residual
                    d_ant_unit = d_ant / n_ant
                    proj_len = float(np.dot(d_neg, d_ant_unit))
                    residual = d_neg - proj_len * d_ant_unit
                    residual_norm = float(np.linalg.norm(residual))
                    
                    comparisons.append({
                        "adj1": adj1, "adj2": adj2,
                        "cosine": cos_na,
                        "neg_norm": float(n_neg),
                        "ant_norm": float(n_ant),
                        "projection": proj_len,
                        "residual_norm": residual_norm,
                        "residual_ratio": residual_norm / n_neg,
                    })
        
        if comparisons:
            avg_cos = np.mean([c["cosine"] for c in comparisons])
            avg_res_ratio = np.mean([c["residual_ratio"] for c in comparisons])
            neg_ant[layer] = {
                "avg_cosine": float(avg_cos),
                "avg_residual_ratio": float(avg_res_ratio),
                "n_pairs": len(comparisons),
                "comparisons": comparisons,
            }
            log(f"  L{layer}: avg_cosine={avg_cos:.3f}, avg_residual_ratio={avg_res_ratio:.3f}")
    
    # ---- Analysis 5: Normalized operator PCA (remove norm bias) ----
    log("Analysis 5: Normalized operator PCA...")
    norm_op_pca = {}
    for layer in sample_layers:
        all_deltas = []
        for op in sorted(avg_neg_inc.keys()):
            if layer in avg_neg_inc[op]:
                d = avg_neg_inc[op][layer]
                n = np.linalg.norm(d)
                if n > 0.01:
                    all_deltas.append(d / n)  # unit vectors
        
        if len(all_deltas) < 3:
            continue
        
        mat = np.array(all_deltas)
        mat_centered = mat - mat.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(mat_centered, full_matrices=False)
            total_var = np.sum(S**2)
            cumvar = np.cumsum(S**2) / total_var
            dim50 = int(np.searchsorted(cumvar, 0.5)) + 1
            dim80 = int(np.searchsorted(cumvar, 0.8)) + 1
            
            norm_op_pca[layer] = {
                "top1_var": float(S[0]**2 / total_var),
                "top3_var": float(np.sum(S[:min(3,len(S))]**2) / total_var),
                "dim50": dim50, "dim80": dim80,
            }
            log(f"  L{layer}: norm_top1={norm_op_pca[layer]['top1_var']:.1%}, dim50={dim50}, dim80={dim80}")
        except:
            pass
    
    # ---- Save results ----
    def to_serializable(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        if isinstance(obj, dict): return {str(k): to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_serializable(v) for v in obj]
        if isinstance(obj, tuple): return list(obj)
        return obj
    
    results = to_serializable({
        "model": model_key, "n_layers": n_layers, "d_model": d_model,
        "sample_layers": sample_layers, "mid_layer": mid,
        "n_stimuli": len(stimuli), "n_unique_sents": len(unique_sents),
        "n_operands": len(avg_neg_inc),
        "operator_pca": op_pca,
        "normalized_operator_pca": norm_op_pca,
        "loo_cosine": loo_cos,
        "neg_vs_antonym": neg_ant,
    })
    
    out_path = RESULT_DIR / f"{model_key}_operator_scope.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"Saved to {out_path}")
    
    # ---- Print summary ----
    log("\n" + "=" * 60)
    log(f"SUMMARY: {model_key} (L{mid})")
    log("=" * 60)
    
    if mid in op_pca:
        d = op_pca[mid]
        log(f"  Operator PCA: top1={d['top1_var']:.1%}, dim50={d['dim50']}, dim80={d['dim80']}, norm_ratio={d['norm_ratio']:.1f}x")
    
    if mid in norm_op_pca:
        d = norm_op_pca[mid]
        log(f"  Norm PCA:     top1={d['top1_var']:.1%}, dim50={d['dim50']}, dim80={d['dim80']}")
    
    if mid in loo_cos:
        d = loo_cos[mid]
        log(f"  LOO cosine:   avg={d['avg_loo_cos']:.3f}, min={d['min_loo_cos']:.3f}")
    
    if mid in neg_ant:
        d = neg_ant[mid]
        log(f"  Neg vs Ant:   avg_cosine={d['avg_cosine']:.3f}, avg_residual_ratio={d['avg_residual_ratio']:.3f}")
        for c in d["comparisons"]:
            log(f"    not-{c['adj1']} vs {c['adj2']}: cos={c['cosine']:.3f} residual={c['residual_ratio']:.3f}")
    
    # Compare with role subspace from Phase 298
    log("\n  Comparison with Phase 298 role subspace:")
    if mid in op_pca and mid in loo_cos:
        log(f"    Role PCA top1: 22-35% (Qwen3/GLM4), dim50=3-5")
        log(f"    Operator PCA top1: {op_pca[mid]['top1_var']:.1%}, dim50={op_pca[mid]['dim50']}")
        log(f"    Role LOO: +0.44 (Qwen3/GLM4), -0.49 (DS7B)")
        log(f"    Operator LOO: {loo_cos[mid]['avg_loo_cos']:.3f}")
    
    release_model(model_key)
    log(f"Phase 300 complete for {model_key}")

if __name__ == "__main__":
    main()
