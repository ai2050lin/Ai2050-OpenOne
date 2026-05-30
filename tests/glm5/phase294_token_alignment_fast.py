"""
Phase 294 Optimized: Token Alignment Fix (fast version for GLM4/DS7B)
======================================================================
Optimizations vs original:
1. Exp A: resid_post only (skip attn/mlp — covered in Phase 293)
2. Key layers sampling for Exp B/C (every 4 for mid, every 2 for edges)
3. More frequent logging and intermediate saves
4. Flash attention + gradient checkpointing

Usage:
  python tests/glm5/phase294_token_alignment_fast.py glm4
  python tests/glm5/phase294_token_alignment_fast.py deepseek7b
"""
import sys, os, gc, time, json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase294_token_alignment")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp"); TMP_DIR.mkdir(parents=True, exist_ok=True)
_log_file = None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file, "a", encoding="utf-8") as f: f.write(line + "\n")
        except: pass


# =====================================================================
# NEGATION PAIRS (same as Phase 293/294)
# =====================================================================
def build_negation_pairs():
    pairs = []
    # lexical_not_adj
    for n,p,ne,adj in [("happy","she is happy","she is not happy","happy"),
        ("open","the door is open","the door is not open","open"),
        ("possible","victory is possible","victory is not possible","possible"),
        ("ready","they are ready","they are not ready","ready"),
        ("important","this is important","this is not important","important"),
        ("clear","the answer is clear","the answer is not clear","clear"),
        ("safe","the area is safe","the area is not safe","safe"),
        ("fair","the decision is fair","the decision is not fair","fair"),
        ("simple","the problem is simple","the problem is not simple","simple"),
        ("correct","your answer is correct","your answer is not correct","correct")]:
        pairs.append({"name":f"adj_{n}","A":p,"B":ne,"subtype":"lexical_not_adj",
                      "op":"not","operand":adj})
    # syntactic_do_not
    for n,p,ne,v in [("agree","they agree with the proposal","they do not agree with the proposal","agree"),
        ("remember","i remember the meeting","i do not remember the meeting","remember"),
        ("understand","we understand the problem","we do not understand the problem","understand"),
        ("know","she knows the answer","she does not know the answer","knows"),
        ("believe","he believes the story","he does not believe the story","believes"),
        ("support","they support the plan","they do not support the plan","support"),
        ("accept","she accepts the offer","she does not accept the offer","accepts"),
        ("expect","we expect rain","we do not expect rain","expect"),
        ("trust","he trusts the source","he does not trust the source","trusts"),
        ("follow","they follow the rules","they do not follow the rules","follow")]:
        pairs.append({"name":f"verb_{n}","A":p,"B":ne,"subtype":"syntactic_do_not",
                      "op":"not","operand":v})
    # existential_no
    for n,p,ne in [("nothing","he found something interesting","he found nothing interesting"),
        ("no_one","someone came to the party","no one came to the party"),
        ("no_food","there was some food left","there was no food left"),
        ("no_idea","she had some idea what to do","she had no idea what to do"),
        ("no_reason","there is a reason to worry","there is no reason to worry"),
        ("no_choice","they had a choice in the matter","they had no choice in the matter"),
        ("no_doubt","there is some doubt about it","there is no doubt about it"),
        ("no_evidence","there is evidence of fraud","there is no evidence of fraud"),
        ("no_sign","there is a sign of life","there is no sign of life"),
        ("no_animal","an animal crossed the road","no animal crossed the road")]:
        pairs.append({"name":f"no_{n}","A":p,"B":ne,"subtype":"existential_no",
                      "op":"no","operand":n})
    # never
    for n,p,ne,v in [("seen","i have seen it before","i have never seen it before","seen"),
        ("been","she has been to Paris","she has never been to Paris","been"),
        ("told","he told someone the secret","he never told anyone the secret","told"),
        ("gives_up","she sometimes gives up","she never gives up","gives"),
        ("forgets","he sometimes forgets names","he never forgets a face","forgets"),
        ("complains","she often complains","she never complains","complains"),
        ("tells_truth","he sometimes tells the truth","he never tells the truth","tells"),
        ("late","she is sometimes late","she is never late","late"),
        ("apologizes","he sometimes apologizes","he never apologizes","apologizes"),
        ("admits","she sometimes admits mistakes","she never admits mistakes","admits")]:
        pairs.append({"name":f"never_{n}","A":p,"B":ne,"subtype":"never",
                      "op":"never","operand":v})
    # morphological_neg
    for n,p,ne,root in [("impossible","the task is possible","the task is impossible","possible"),
        ("unacceptable","the proposal is acceptable","the proposal is unacceptable","acceptable"),
        ("incomplete","the report is complete","the report is incomplete","complete"),
        ("irrelevant","the comment is relevant","the comment is irrelevant","relevant"),
        ("dishonest","the person is honest","the person is dishonest","honest"),
        ("unfair","the treatment was fair","the treatment was unfair","fair"),
        ("unlikely","the outcome is likely","the outcome is unlikely","likely"),
        ("incorrect","the assumption is correct","the assumption is incorrect","correct"),
        ("uncertain","the result is certain","the result is uncertain","certain"),
        ("disobey","the soldiers obey orders","the soldiers disobey orders","obey")]:
        pairs.append({"name":f"prefix_{n}","A":p,"B":ne,"subtype":"morphological_neg",
                      "op":n,"operand":root})
    # scope_quantifier
    for n,p,ne,q in [("not_all","all birds can fly","not all birds can fly","all"),
        ("not_everyone","everyone agreed","not everyone agreed","everyone"),
        ("not_always","she always tells the truth","she does not always tell the truth","always"),
        ("not_entirely","the plan is entirely successful","the plan is not entirely successful","entirely"),
        ("not_necessarily","wealth means happiness","wealth does not necessarily mean happiness","necessarily"),
        ("not_exactly","that is exactly what i meant","that is not exactly what i meant","exactly"),
        ("not_quite","the work is finished","the work is not quite finished","quite"),
        ("not_completely","the glass is full","the glass is not completely full","completely"),
        ("not_if","she will come if invited","she will not come if invited","if"),
        ("not_any","there are some problems","there are not any problems","any")]:
        pairs.append({"name":f"scope_{n}","A":p,"B":ne,"subtype":"scope_quantifier",
                      "op":"not","operand":q})
    return pairs


# =====================================================================
# TOKEN ALIGNMENT
# =====================================================================
def build_alignment(tokenizer, sent_a, sent_b, op_text, operand_text):
    toks_a = tokenizer.encode(sent_a, add_special_tokens=True)
    toks_b = tokenizer.encode(sent_b, add_special_tokens=True)
    dec_a = [tokenizer.decode([t]).strip().lower() for t in toks_a]
    dec_b = [tokenizer.decode([t]).strip().lower() for t in toks_b]
    len_a, len_b = len(toks_a), len(toks_b)

    prefix_len = 0
    for i in range(min(len_a, len_b)):
        if toks_a[i] == toks_b[i]: prefix_len = i + 1
        else: break

    suffix_len = 0
    max_suffix = min(len_a, len_b) - prefix_len
    for i in range(1, max_suffix + 1):
        if toks_a[len_a - i] == toks_b[len_b - i]: suffix_len = i
        else: break

    operator_pos_b = None
    op_lower = op_text.lower()
    for i, t in enumerate(dec_b):
        if op_lower in t or t in op_lower: operator_pos_b = i; break
    if operator_pos_b is None and len(op_lower) >= 2:
        for i, t in enumerate(dec_b):
            if op_lower[:3] in t or t[:3] in op_lower: operator_pos_b = i; break

    operand_pos_a = None; operand_pos_b = None
    opnd_lower = operand_text.lower()
    for i, t in enumerate(dec_a):
        if opnd_lower in t or t in opnd_lower: operand_pos_a = i; break
    if operand_pos_a is None and len(opnd_lower) >= 2:
        for i, t in enumerate(dec_a):
            if opnd_lower[:3] in t: operand_pos_a = i; break
    for i, t in enumerate(dec_b):
        if opnd_lower in t or t in opnd_lower: operand_pos_b = i; break
    if operand_pos_b is None and len(opnd_lower) >= 2:
        for i, t in enumerate(dec_b):
            if opnd_lower[:3] in t: operand_pos_b = i; break
    if operand_pos_b is None and operator_pos_b is not None:
        operand_pos_b = operator_pos_b

    last_pos_a = len_a - 1; last_pos_b = len_b - 1
    operator_corresponding_a = prefix_len if prefix_len < len_a else None
    len_diff = len_b - len_a

    return {
        "toks_a": toks_a, "toks_b": toks_b,
        "dec_a": dec_a, "dec_b": dec_b,
        "len_a": len_a, "len_b": len_b,
        "prefix_len": prefix_len, "suffix_len": suffix_len,
        "len_diff": len_diff,
        "operator_pos_b": operator_pos_b,
        "operand_pos_a": operand_pos_a, "operand_pos_b": operand_pos_b,
        "last_pos_a": last_pos_a, "last_pos_b": last_pos_b,
        "operator_corresponding_a": operator_corresponding_a,
    }


# =====================================================================
# MODEL LOADING
# =====================================================================
def load_model_fast(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16, device_map=auto, flash)...")
    tok = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    model = None; used_attn = "eager"
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=attn_impl,
            )
            used_attn = attn_impl; break
        except Exception as e:
            log(f"  attn={attn_impl} failed: {str(e)[:80]}")
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    gpu = torch.cuda.memory_allocated() / 1e9
    log(f"  Loaded with attn={used_attn}, GPU={gpu:.1f}GB")

    layers = get_layers(model); nl = len(layers)
    gpu_l, cpu_l = [], []
    for li in range(nl):
        wdev = layers[li].self_attn.o_proj.weight.device
        (gpu_l if wdev.type == 'cuda' else cpu_l).append(li)
    log(f"  GPU layers: {len(gpu_l)}{' ('+str(gpu_l[0])+'-'+str(gpu_l[-1])+')' if gpu_l else ''}, "
        f"CPU: {len(cpu_l)}{' ('+str(cpu_l[0])+'-'+str(cpu_l[-1])+')' if cpu_l else ''}")
    return model, tok


# =====================================================================
# CAPTURE (with hooks)
# =====================================================================
def capture_all(model, tokenizer, sent, n_layers, max_len=64):
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    cap = {"attn": {}, "mlp": {}, "hidden": {}}
    hooks = []
    for li in range(n_layers):
        def mah(li):
            def hook(m, i, o):
                cap["attn"][li] = (o[0] if isinstance(o, tuple) else o).detach().cpu().float().clone()
            return hook
        def mmh(li):
            def hook(m, i, o):
                cap["mlp"][li] = (o[0] if isinstance(o, tuple) else o).detach().cpu().float().clone()
            return hook
        hooks.append(layers[li].self_attn.register_forward_hook(mah(li)))
        hooks.append(layers[li].mlp.register_forward_hook(mmh(li)))
    with torch.no_grad():
        try:
            out = model(**inputs, output_hidden_states=True)
        except Exception as e:
            log(f"  capture ERR: {e}")
            cap["hidden"] = {}; cap["logits"] = None
        else:
            for li, hs in enumerate(out.hidden_states):
                cap["hidden"][li] = hs.detach().cpu().float().clone()
            cap["logits"] = out.logits[0, -1, :].detach().cpu().float().clone()
    for h in hooks: h.remove()
    return cap


# =====================================================================
# METRICS
# =====================================================================
def compute_metrics(logits, la, lb, kab):
    if logits is None: return None
    kp = float(F.kl_div(F.log_softmax(logits, -1), F.softmax(lb, -1), reduction='sum'))
    kr = min(kp / max(kab, 1e-6), 100.0)
    db = lb - la; dp = logits - la
    nb, np_ = float(torch.norm(db)), float(torch.norm(dp))
    cd = float(torch.dot(dp, db) / (nb * np_)) if nb > 1e-8 and np_ > 1e-8 else 0
    prog = cd * min(np_ / nb, 2.0) if nb > 1e-8 else 0
    np_metric = prog / (1.0 + kr)
    return {"kl_ratio": round(kr, 4), "progress": round(prog, 5),
            "cos_dir": round(cd, 5), "natural_prog": round(np_metric, 6)}


# =====================================================================
# ALIGNED PATCH HOOKS
# =====================================================================
def _aligned_pos_hook(pv_cpu, src_pos, dst_pos):
    def hook(m, i, o):
        ref = o[0] if isinstance(o, tuple) else o
        pv = pv_cpu.to(ref.device).to(ref.dtype)
        if isinstance(o, tuple):
            no = (o[0].clone(),) + o[1:]
            if src_pos < pv.shape[1] and dst_pos < no[0].shape[1]:
                no[0][:, dst_pos, :] = pv[:, src_pos, :]
            return no
        no = o.clone()
        if src_pos < pv.shape[1] and dst_pos < no.shape[1]:
            no[:, dst_pos, :] = pv[:, src_pos, :]
        return no
    return hook

def _aligned_multi_pos_hook(pv_cpu, src_dst_pairs):
    def hook(m, i, o):
        ref = o[0] if isinstance(o, tuple) else o
        pv = pv_cpu.to(ref.device).to(ref.dtype)
        if isinstance(o, tuple):
            no = (o[0].clone(),) + o[1:]
            for sp, dp in src_dst_pairs:
                if sp < pv.shape[1] and dp < no[0].shape[1]:
                    no[0][:, dp, :] = pv[:, sp, :]
            return no
        no = o.clone()
        for sp, dp in src_dst_pairs:
            if sp < pv.shape[1] and dp < no.shape[1]:
                no[:, dp, :] = pv[:, sp, :]
        return no
    return hook


def forward_aligned_patch(model, tokenizer, sent_run, n_layers, max_len, patches):
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent_run, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    hooks = []
    for patch in patches:
        mod, pv = patch[0], patch[1]
        if len(patch) == 4:
            _, _, sp, dp = patch
            hooks.append(mod.register_forward_hook(_aligned_pos_hook(pv, sp, dp)))
        elif len(patch) == 3:
            _, _, pairs = patch
            hooks.append(mod.register_forward_hook(_aligned_multi_pos_hook(pv, pairs)))
    try:
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :].detach().cpu().float().clone()
    except Exception as e:
        log(f"    PATCH ERR: {e}")
        logits = None
    for h in hooks: h.remove()
    return logits


# =====================================================================
# KEY LAYERS SELECTION
# =====================================================================
def get_key_layers(nl):
    """Select key layers for sampling. Full coverage at edges, sparse in middle."""
    if nl <= 20:
        return list(range(nl))
    # Full coverage for first 5 and last 6
    early = list(range(min(6, nl)))
    late = list(range(max(nl-6, 6), nl))
    # Sparse for middle (every 4th)
    mid = list(range(6, max(nl-6, 6), 4))
    return sorted(set(early + mid + late))


# =====================================================================
# EXP A: B→A resid_post only (aligned + misaligned)
# =====================================================================
def run_exp_A(model, tok, pair_data, pair_metrics, pair_align, nl, max_len):
    log("\n=== EXP A: B->A resid_post (aligned + misaligned) ===")
    layers = get_layers(model)
    results = []
    total = nl * 4 * len(pair_data)  # 4 roles per pair
    done = 0; t0 = time.time()

    for li in range(nl):
        for pn, pd in pair_data.items():
            pm = pair_metrics.get(pn); al = pair_align.get(pn)
            if not pm or not al: continue
            la, lb, kab = pm["la"], pm["lb"], pm["kab"]
            pv = pd["B"]["hidden"].get(li + 1)
            mod = layers[li]
            if pv is None: continue

            # ALIGNED operand
            if al["operand_pos_a"] is not None and al["operand_pos_b"] is not None:
                logits = forward_aligned_patch(model, tok, pm["sa"], nl, max_len,
                    [(mod, pv, al["operand_pos_b"], al["operand_pos_a"])])
                m = compute_metrics(logits, la, lb, kab)
                if m: results.append({"layer":li,"component":"resid_post","role":"operand_aligned",
                                      "dir":"B2A","name":pn,"subtype":pm["subtype"],**m})

            # ALIGNED last
            logits = forward_aligned_patch(model, tok, pm["sa"], nl, max_len,
                [(mod, pv, al["last_pos_b"], al["last_pos_a"])])
            m = compute_metrics(logits, la, lb, kab)
            if m: results.append({"layer":li,"component":"resid_post","role":"last_aligned",
                                  "dir":"B2A","name":pn,"subtype":pm["subtype"],**m})

            # MISALIGNED operand
            if al["operand_pos_b"] is not None and al["operand_pos_b"] < al["len_a"]:
                logits = forward_aligned_patch(model, tok, pm["sa"], nl, max_len,
                    [(mod, pv, al["operand_pos_b"], al["operand_pos_b"])])
                m = compute_metrics(logits, la, lb, kab)
                if m: results.append({"layer":li,"component":"resid_post","role":"operand_misaligned",
                                      "dir":"B2A","name":pn,"subtype":pm["subtype"],**m})

            # MISALIGNED last
            if al["last_pos_b"] < al["len_a"]:
                logits = forward_aligned_patch(model, tok, pm["sa"], nl, max_len,
                    [(mod, pv, al["last_pos_b"], al["last_pos_b"])])
                m = compute_metrics(logits, la, lb, kab)
                if m: results.append({"layer":li,"component":"resid_post","role":"last_misaligned",
                                      "dir":"B2A","name":pn,"subtype":pm["subtype"],**m})
            done += 1

        if (li + 1) % 2 == 0 or li == nl - 1:
            el = time.time() - t0; rate = done / el if el > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            log(f"  ExpA L{li+1}/{nl} {done}/{total} ({rate:.1f}/s) ETA={eta:.0f}s")

        # Intermediate save every 8 layers
        if (li + 1) % 8 == 0:
            _save_intermediate("A", results, pair_data)

    log(f"  Exp A: {len(results)} results, {time.time()-t0:.0f}s")
    return results


def _save_intermediate(exp, results, pair_data):
    """Save intermediate results to avoid data loss on crash."""
    if not results: return
    model_name = list(pair_data.keys())[0].split("_")[0] if pair_data else "unknown"
    # Build summary
    summary = defaultdict(lambda: {"np_vals": [], "n": 0})
    for r in results:
        k = f"{r['layer']},{r['component']},{r['role']}"
        summary[k]["np_vals"].append(r["natural_prog"])
        summary[k]["n"] += 1
    summary_out = {}
    for k, v in summary.items():
        summary_out[k] = {"mean_np": round(np.mean(v["np_vals"]), 6),
                          "std_np": round(np.std(v["np_vals"]), 6), "n": v["n"]}
    fpath = RESULT_DIR / f"intermediate_exp{exp}.json"
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump({"summary": summary_out, "count": len(results)}, f, indent=2)
    except: pass


# =====================================================================
# EXP B: A→B key layers only (resid_post)
# =====================================================================
def run_exp_B(model, tok, pair_data, pair_metrics, pair_align, nl, max_len):
    log("\n=== EXP B: A->B key layers (resid_post) ===")
    layers = get_layers(model)
    key_layers = get_key_layers(nl)
    log(f"  Key layers: {key_layers}")
    results = []
    total = len(key_layers) * 3 * len(pair_data)
    done = 0; t0 = time.time()

    for li in key_layers:
        for pn, pd in pair_data.items():
            pm = pair_metrics.get(pn); al = pair_align.get(pn)
            if not pm or not al: continue
            la_b2a, lb_b2a, kab = pm["lb"], pm["la"], pm["kab"]
            pv = pd["A"]["hidden"].get(li + 1)
            mod = layers[li]
            if pv is None: continue

            # OPERATOR
            if al["operator_pos_b"] is not None and al["operator_corresponding_a"] is not None:
                logits = forward_aligned_patch(model, tok, pm["sb"], nl, max_len,
                    [(mod, pv, al["operator_corresponding_a"], al["operator_pos_b"])])
                m = compute_metrics(logits, la_b2a, lb_b2a, kab)
                if m: results.append({"layer":li,"component":"resid_post","role":"operator",
                                      "dir":"A2B","name":pn,"subtype":pm["subtype"],**m})

            # OPERAND
            if al["operand_pos_a"] is not None and al["operand_pos_b"] is not None:
                logits = forward_aligned_patch(model, tok, pm["sb"], nl, max_len,
                    [(mod, pv, al["operand_pos_a"], al["operand_pos_b"])])
                m = compute_metrics(logits, la_b2a, lb_b2a, kab)
                if m: results.append({"layer":li,"component":"resid_post","role":"operand",
                                      "dir":"A2B","name":pn,"subtype":pm["subtype"],**m})

            # LAST
            logits = forward_aligned_patch(model, tok, pm["sb"], nl, max_len,
                [(mod, pv, al["last_pos_a"], al["last_pos_b"])])
            m = compute_metrics(logits, la_b2a, lb_b2a, kab)
            if m: results.append({"layer":li,"component":"resid_post","role":"last",
                                  "dir":"A2B","name":pn,"subtype":pm["subtype"],**m})
            done += 1

        el = time.time() - t0; rate = done / el if el > 0 else 0
        log(f"  ExpB L{li}/{nl} {done}/{total} ({rate:.1f}/s)")

    log(f"  Exp B: {len(results)} results, {time.time()-t0:.0f}s")
    return results


# =====================================================================
# EXP C: Component synergy at key layers
# =====================================================================
def run_exp_C(model, tok, pair_data, pair_metrics, pair_align, nl, max_len):
    log("\n=== EXP C: Component Synergy (key layers) ===")
    layers = get_layers(model)
    key_layers = get_key_layers(nl)
    results = []
    total = len(key_layers) * 3 * 2 * len(pair_data)  # 3 comps x 2 positions
    done = 0; t0 = time.time()

    for li in key_layers:
        for pn, pd in pair_data.items():
            pm = pair_metrics.get(pn); al = pair_align.get(pn)
            if not pm or not al: continue
            la, lb, kab = pm["la"], pm["lb"], pm["kab"]

            for pos_name, src_pos_b, dst_pos_a in [
                ("operand", al["operand_pos_b"], al["operand_pos_a"]),
                ("last", al["last_pos_b"], al["last_pos_a"])
            ]:
                if src_pos_b is None or dst_pos_a is None: continue

                # attn_only
                pv_attn = pd["B"]["attn"].get(li)
                if pv_attn is not None:
                    logits = forward_aligned_patch(model, tok, pm["sa"], nl, max_len,
                        [(layers[li].self_attn, pv_attn, src_pos_b, dst_pos_a)])
                    m = compute_metrics(logits, la, lb, kab)
                    if m: results.append({"layer":li,"component":"attn_only","role":pos_name,
                                          "dir":"B2A","name":pn,"subtype":pm["subtype"],**m})

                # mlp_only
                pv_mlp = pd["B"]["mlp"].get(li)
                if pv_mlp is not None:
                    logits = forward_aligned_patch(model, tok, pm["sa"], nl, max_len,
                        [(layers[li].mlp, pv_mlp, src_pos_b, dst_pos_a)])
                    m = compute_metrics(logits, la, lb, kab)
                    if m: results.append({"layer":li,"component":"mlp_only","role":pos_name,
                                          "dir":"B2A","name":pn,"subtype":pm["subtype"],**m})

                # both (attn + mlp at same position)
                if pv_attn is not None and pv_mlp is not None:
                    logits = forward_aligned_patch(model, tok, pm["sa"], nl, max_len,
                        [(layers[li].self_attn, pv_attn, src_pos_b, dst_pos_a),
                         (layers[li].mlp, pv_mlp, src_pos_b, dst_pos_a)])
                    m = compute_metrics(logits, la, lb, kab)
                    if m: results.append({"layer":li,"component":"attn_mlp","role":pos_name,
                                          "dir":"B2A","name":pn,"subtype":pm["subtype"],**m})
                done += 1

        el = time.time() - t0; rate = done / el if el > 0 else 0
        log(f"  ExpC L{li}/{nl} {done}/{total} ({rate:.1f}/s)")

    log(f"  Exp C: {len(results)} results, {time.time()-t0:.0f}s")
    return results


# =====================================================================
# MAIN
# =====================================================================
def main():
    global _log_file
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    log_file = TMP_DIR / f"phase294_{model_name}.txt"
    _log_file = str(log_file)

    log(f"Phase 294 FAST: Token Alignment Fix — {model_name}")
    model, tok = load_model_fast(model_name)
    nl = len(get_layers(model))
    max_len = 64

    # Build pairs
    pairs = build_negation_pairs()
    log(f"Pairs: {len(pairs)}")

    # Capture + align
    pair_data = {}; pair_metrics = {}; pair_align = {}
    t0 = time.time()
    for idx, p in enumerate(pairs):
        sa, sb = p["A"], p["B"]
        cap_a = capture_all(model, tok, sa, nl, max_len)
        cap_b = capture_all(model, tok, sb, nl, max_len)
        al = build_alignment(tok, sa, sb, p["op"], p["operand"])

        la = cap_a["logits"]; lb = cap_b["logits"]
        kab = max(float(F.kl_div(F.log_softmax(la, -1), F.softmax(lb, -1), reduction='sum')),
                  float(F.kl_div(F.log_softmax(lb, -1), F.softmax(la, -1), reduction='sum')))

        pair_data[p["name"]] = {"A": cap_a, "B": cap_b}
        pair_metrics[p["name"]] = {"la": la, "lb": lb, "kab": kab,
                                    "sa": sa, "sb": sb, "subtype": p["subtype"]}
        pair_align[p["name"]] = al

        # Free GPU memory from capture
        for cap in [cap_a, cap_b]:
            for k in ["attn", "mlp", "hidden"]:
                for li in list(cap[k].keys()):
                    cap[k][li] = cap[k][li].cpu()
            if cap.get("logits") is not None:
                cap["logits"] = cap["logits"].cpu()

        if (idx + 1) % 10 == 0:
            gpu = torch.cuda.memory_allocated() / 1e9
            log(f"  [{idx+1}/{len(pairs)}] {time.time()-t0:.0f}s, GPU={gpu:.1f}GB")

    log(f"Capture+Align: {len(pairs)} pairs, {time.time()-t0:.0f}s")

    # Alignment summary
    for st in ["lexical_not_adj","syntactic_do_not","existential_no","never","morphological_neg","scope_quantifier"]:
        st_pairs = [(pn, al) for pn, al in pair_align.items()
                    if pair_metrics[pn]["subtype"] == st]
        if st_pairs:
            same_op = sum(1 for _, al in st_pairs if al["operand_pos_a"] == al["operand_pos_b"])
            mean_diff = np.mean([al["len_diff"] for _, al in st_pairs])
            log(f"  {st}: n={len(st_pairs)}, mean_len_diff={mean_diff:.1f}, same_operand_pos={same_op}/{len(st_pairs)}")

    # Run experiments
    exp_a = run_exp_A(model, tok, pair_data, pair_metrics, pair_align, nl, max_len)
    exp_b = run_exp_B(model, tok, pair_data, pair_metrics, pair_align, nl, max_len)
    exp_c = run_exp_C(model, tok, pair_data, pair_metrics, pair_align, nl, max_len)

    # Save results
    def build_summary(raw):
        s = defaultdict(lambda: {"np_vals": [], "n": 0})
        for r in raw:
            k = f"{r['layer']},{r['component']},{r['role']}"
            s[k]["np_vals"].append(r["natural_prog"])
            s[k]["n"] += 1
        return {k: {"mean_np": round(np.mean(v["np_vals"]), 6),
                     "std_np": round(np.std(v["np_vals"]), 6), "n": v["n"]}
                for k, v in s.items()}

    out = {
        "model": model_name, "n_layers": nl, "n_pairs": len(pairs),
        "exp_A_count": len(exp_a), "exp_B_count": len(exp_b), "exp_C_count": len(exp_c),
        "exp_A_summary": build_summary(exp_a),
        "exp_B_summary": build_summary(exp_b),
        "exp_C_summary": build_summary(exp_c),
        "exp_A_raw": exp_a[:500],
        "exp_B_raw": exp_b[:500],
        "exp_C_raw": exp_c[:500],
    }

    fpath = RESULT_DIR / f"{model_name}_alignment.json"
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log(f"Saved to {fpath}")

    # Print key results
    log("\n=== KEY RESULTS ===")
    # Exp A: best aligned vs misaligned per role
    for role_pair in [("operand_aligned","operand_misaligned"), ("last_aligned","last_misaligned")]:
        for layer in [0, nl//4, nl//2, 3*nl//4, nl-2]:
            k_al = f"{layer},resid_post,{role_pair[0]}"
            k_mi = f"{layer},resid_post,{role_pair[1]}"
            al_v = out["exp_A_summary"].get(k_al, {}).get("mean_np", 0)
            mi_v = out["exp_A_summary"].get(k_mi, {}).get("mean_np", 0)
            ratio = al_v / mi_v if mi_v > 0.001 else 0
            log(f"  L{layer} {role_pair[0][:3]}_al={al_v:.4f} {role_pair[1][:3]}_mi={mi_v:.4f} ratio={ratio:.3f}")

    # Exp B: best operator effect
    op_results = [(k, v) for k, v in out["exp_B_summary"].items() if "operator" in k]
    if op_results:
        best = max(op_results, key=lambda x: x[1]["mean_np"])
        log(f"  Best A->B operator: {best[0]} NP={best[1]['mean_np']:.4f}")

    # Exp C: synergy
    for role in ["operand", "last"]:
        syn_vals = []
        for k, v in out["exp_C_summary"].items():
            if f",{role}" in k and "attn_mlp" in k:
                li = int(k.split(",")[0])
                a_k = f"{li},attn_only,{role}"
                m_k = f"{li},mlp_only,{role}"
                a_v = out["exp_C_summary"].get(a_k, {}).get("mean_np", 0)
                m_v = out["exp_C_summary"].get(m_k, {}).get("mean_np", 0)
                b_v = v["mean_np"]
                syn = b_v - a_v - m_v
                syn_vals.append((li, a_v, m_v, b_v, syn))
        if syn_vals:
            best_syn = max(syn_vals, key=lambda x: abs(x[4]))
            log(f"  Best synergy {role}: L{best_syn[0]} attn={best_syn[1]:.4f} mlp={best_syn[2]:.4f} both={best_syn[3]:.4f} syn={best_syn[4]:+.4f}")

    # Cleanup
    release_model(model)
    del pair_data; gc.collect(); torch.cuda.empty_cache()
    log("Done!")


if __name__ == "__main__":
    main()
