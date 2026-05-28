"""
Phase 293: Component Contract Decomposition
============================================
Core fixes vs Phase 292:
1. Natural Progress (NP) = PROG / (1 + KR) — penalizes unnatural patches
2. Proper resid_post: use hidden_states[li+1] from B (not fake both)
3. Full layer x component scan: attn_only, mlp_only, resid_post for ALL layers
4. Position x component: operator/operand/last/all for top layers
5. Alpha x component: interpolation curves for top layers

Usage:
  python tests/glm5/phase293_component_contract.py qwen3
  python tests/glm5/phase293_component_contract.py glm4
  python tests/glm5/phase293_component_contract.py deepseek7b
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

RESULT_DIR = Path("results/phase293_component_contract")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp"); TMP_DIR.mkdir(parents=True, exist_ok=True)
_log_file = None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file,"a",encoding="utf-8") as f: f.write(line+"\n")
        except: pass

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
        pairs.append({"name":f"adj_{n}","A":p,"B":ne,"subtype":"lexical_not_adj","op":"not","operand":adj})
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
        pairs.append({"name":f"verb_{n}","A":p,"B":ne,"subtype":"syntactic_do_not","op":"not","operand":v})
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
        pairs.append({"name":f"no_{n}","A":p,"B":ne,"subtype":"existential_no","op":"no","operand":n})
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
        pairs.append({"name":f"never_{n}","A":p,"B":ne,"subtype":"never","op":"never","operand":v})
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
        pairs.append({"name":f"prefix_{n}","A":p,"B":ne,"subtype":"morphological_neg","op":n,"operand":root})
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
        pairs.append({"name":f"scope_{n}","A":p,"B":ne,"subtype":"scope_quantifier","op":"not","operand":q})
    return pairs

def find_pos(tokenizer, sentence, target):
    """Find first token position of target word."""
    toks = tokenizer.encode(sentence, add_special_tokens=True)
    for i, tid in enumerate(toks):
        if target.lower() in tokenizer.decode([tid]).strip().lower():
            return i
    for i, tid in enumerate(toks):
        if any(c in tokenizer.decode([tid]).strip().lower() for c in target.lower()[:3]):
            return i
    return None

def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16, device_map=auto, eager)...")
    tok = AutoTokenizer.from_pretrained(cfg["path"],trust_remote_code=True,local_files_only=True,use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["path"],torch_dtype=torch.bfloat16,
        device_map="auto",trust_remote_code=True,local_files_only=True,attn_implementation="eager")
    model.eval()
    gpu = torch.cuda.memory_allocated()/1e9
    log(f"  Loaded, GPU={gpu:.1f}GB")
    layers = get_layers(model); nl = len(layers)
    gpu_l, cpu_l = [], []
    for li in range(nl):
        wdev = layers[li].self_attn.o_proj.weight.device
        (gpu_l if wdev.type=='cuda' else cpu_l).append(li)
    log(f"  GPU: {len(gpu_l)}{' ('+str(gpu_l[0])+'-'+str(gpu_l[-1])+')' if gpu_l else ''}, "
        f"CPU: {len(cpu_l)}{' ('+str(cpu_l[0])+'-'+str(cpu_l[-1])+')' if cpu_l else ''}")
    return model, tok

def capture_all(model, tokenizer, sent, n_layers, max_len=64):
    """Capture attn_out, mlp_out, hidden_states."""
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    cap = {"attn": {}, "mlp": {}, "hidden": {}}
    hooks = []
    for li in range(n_layers):
        def mah(li):
            def hook(m, i, o): cap["attn"][li] = (o[0] if isinstance(o,tuple) else o).detach().cpu().float().clone()
            return hook
        def mmh(li):
            def hook(m, i, o): cap["mlp"][li] = (o[0] if isinstance(o,tuple) else o).detach().cpu().float().clone()
            return hook
        hooks.append(layers[li].self_attn.register_forward_hook(mah(li)))
        hooks.append(layers[li].mlp.register_forward_hook(mmh(li)))
    with torch.no_grad():
        try: out = model(**inputs, output_hidden_states=True)
        except Exception as e: log(f"  capture ERR: {e}"); cap["hidden"]={}; cap["logits"]=None
        else:
            for li, hs in enumerate(out.hidden_states):
                cap["hidden"][li] = hs.detach().cpu().float().clone()
            cap["logits"] = out.logits[0,-1,:].detach().cpu().float().clone()
    for h in hooks: h.remove()
    return cap

def compute_metrics(logits, la, lb, kab):
    if logits is None: return None
    kp = float(F.kl_div(F.log_softmax(logits,-1), F.softmax(lb,-1), reduction='sum'))
    kr = min(kp / max(kab, 1e-6), 100.0)
    db, dp = lb - la, logits - la
    nb, np_ = float(torch.norm(db)), float(torch.norm(dp))
    cd = float(torch.dot(dp, db) / (nb * np_)) if nb > 1e-8 and np_ > 1e-8 else 0
    prog = cd * min(np_ / nb, 2.0) if nb > 1e-8 else 0
    np_metric = prog / (1.0 + kr)
    return {"kl_ratio": kr, "progress": prog, "cos_dir": cd, "natural_prog": np_metric}

def _generic_patch_hook(pv_cpu, positions=None):
    """Universal hook: replace output at all or specified positions."""
    def hook(m, i, o):
        ref = o[0] if isinstance(o, tuple) else o
        pv = pv_cpu.to(ref.device).to(ref.dtype)
        mms = min(pv.shape[1], ref.shape[1])
        if positions is None:
            # Full replacement
            if isinstance(o, tuple):
                no = (o[0].clone(),) + o[1:]
                no[0][:, :mms, :] = pv[:, :mms, :]
                return no
            no = o.clone()
            no[:, :mms, :] = pv[:, :mms, :]
            return no
        else:
            # Position-specific
            if isinstance(o, tuple):
                no = (o[0].clone(),) + o[1:]
                for p in positions:
                    if p < mms: no[0][:, p, :] = pv[:, p, :]
                return no
            no = o.clone()
            for p in positions:
                if p < mms: no[:, p, :] = pv[:, p, :]
            return no
    return hook

def forward_patched(model, tokenizer, sent_a, n_layers, max_len, patches):
    """
    patches: list of (module, pv_cpu, positions_or_None)
    module: layers[li].self_attn / layers[li].mlp / layers[li]
    """
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    inputs = tokenizer(sent_a, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    hooks = [mod.register_forward_hook(_generic_patch_hook(pv, pos)) for mod, pv, pos in patches]
    try:
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :].detach().cpu().float().clone()
    except Exception as e:
        log(f"    PATCH ERR: {e}")
        logits = None
    for h in hooks: h.remove()
    return logits


# ====== EXPERIMENT A: Component x Layer Full Scan ======
def run_exp_A(model, tok, pair_data, pair_metrics, nl, max_len):
    log("\n=== EXP A: Component x Layer (attn/mlp/resid_post, alpha=1.0) ===")
    layers = get_layers(model)
    results = []
    total = nl * 3 * len(pair_data)
    done = 0; t0 = time.time()

    for li in range(nl):
        for comp in ['attn', 'mlp', 'resid_post']:
            for pn, pd in pair_data.items():
                pm = pair_metrics.get(pn)
                if not pm: continue
                la, lb, kab = pm["la"], pm["lb"], pm["kab"]

                if comp == 'attn':
                    pv = pd["B"]["attn"].get(li)
                    if pv is None: continue
                    ms = min(pv.shape[1], max_len)
                    mod = layers[li].self_attn
                elif comp == 'mlp':
                    pv = pd["B"]["mlp"].get(li)
                    if pv is None: continue
                    ms = min(pv.shape[1], max_len)
                    mod = layers[li].mlp
                elif comp == 'resid_post':
                    pv = pd["B"]["hidden"].get(li + 1)
                    if pv is None: continue
                    ms = min(pv.shape[1], max_len)
                    mod = layers[li]

                logits = forward_patched(model, tok, pm["sa"], nl, max_len,
                                          [(mod, pv[:, :ms, :], None)])
                m = compute_metrics(logits, la, lb, kab)
                if m:
                    results.append({"layer": li, "component": comp, "name": pn,
                                    "subtype": pm["subtype"], **m})
                done += 1
        if (li+1) % 4 == 0 or li == nl-1:
            el = time.time()-t0; rate = done/el if el>0 else 0
            log(f"  L{li+1}/{nl} {done}/{total} ({rate:.1f}/s) ETA={((total-done)/rate):.0f}s")

    log(f"  Exp A: {len(results)} results, {time.time()-t0:.0f}s")
    return results


# ====== EXPERIMENT B: Position x Component (top layers) ======
def run_exp_B(model, tok, pair_data, pair_metrics, nl, max_len, top_layers):
    log(f"\n=== EXP B: Position x Component (top {len(top_layers)} layers) ===")
    layers = get_layers(model)
    results = []

    for li in top_layers:
        for comp in ['attn', 'resid_post']:
            for pn, pd in pair_data.items():
                pm = pair_metrics.get(pn)
                if not pm: continue
                la, lb, kab = pm["la"], pm["lb"], pm["kab"]
                sl = pm["sl"]
                op_pos = pm.get("op_pos")
                operand_pos = pm.get("operand_pos")
                last_pos = sl - 1

                if comp == 'attn':
                    pv = pd["B"]["attn"].get(li)
                    mod = layers[li].self_attn
                else:
                    pv = pd["B"]["hidden"].get(li + 1)
                    mod = layers[li]
                if pv is None: continue
                ms = min(pv.shape[1], max_len)

                for ptype, positions in [("operator", [op_pos] if op_pos is not None else []),
                                          ("operand", [operand_pos] if operand_pos is not None else []),
                                          ("last", [last_pos]),
                                          ("all", list(range(sl)))]:
                    if not positions: continue
                    logits = forward_patched(model, tok, pm["sa"], nl, max_len,
                                              [(mod, pv[:, :ms, :], positions)])
                    m = compute_metrics(logits, la, lb, kab)
                    if m:
                        results.append({"layer": li, "component": comp, "pos_type": ptype,
                                        "name": pn, "subtype": pm["subtype"], **m})

    log(f"  Exp B: {len(results)} results")
    return results


# ====== EXPERIMENT C: Alpha x Component (top layers) ======
def run_exp_C(model, tok, pair_data, pair_metrics, nl, max_len, top_layers):
    log(f"\n=== EXP C: Alpha x Component (top {len(top_layers)} layers) ===")
    layers = get_layers(model)
    results = []
    alphas = [0, 0.25, 0.5, 0.75, 1.0]

    for li in top_layers:
        for comp in ['attn', 'mlp', 'resid_post']:
            for alpha in alphas:
                for pn, pd in pair_data.items():
                    pm = pair_metrics.get(pn)
                    if not pm: continue
                    la, lb, kab = pm["la"], pm["lb"], pm["kab"]
                    ms = max_len

                    if comp == 'attn':
                        pa = pd["A"]["attn"].get(li)
                        pb = pd["B"]["attn"].get(li)
                        if pa is None or pb is None: continue
                        mms = min(pa.shape[1], pb.shape[1], ms)
                        pv = (1-alpha)*pa[:,:mms,:] + alpha*pb[:,:mms,:]
                        mod = layers[li].self_attn
                    elif comp == 'mlp':
                        pa = pd["A"]["mlp"].get(li)
                        pb = pd["B"]["mlp"].get(li)
                        if pa is None or pb is None: continue
                        mms = min(pa.shape[1], pb.shape[1], ms)
                        pv = (1-alpha)*pa[:,:mms,:] + alpha*pb[:,:mms,:]
                        mod = layers[li].mlp
                    elif comp == 'resid_post':
                        ha = pd["A"]["hidden"].get(li + 1)
                        hb = pd["B"]["hidden"].get(li + 1)
                        if ha is None or hb is None: continue
                        mms = min(ha.shape[1], hb.shape[1], ms)
                        pv = (1-alpha)*ha[:,:mms,:] + alpha*hb[:,:mms,:]
                        mod = layers[li]

                    logits = forward_patched(model, tok, pm["sa"], nl, max_len,
                                              [(mod, pv, None)])
                    m = compute_metrics(logits, la, lb, kab)
                    if m:
                        results.append({"layer": li, "component": comp, "alpha": alpha,
                                        "name": pn, "subtype": pm["subtype"], **m})

    log(f"  Exp C: {len(results)} results")
    return results


# ====== MAIN ======
def run_phase293(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase293_{model_name}.txt")
    log(f"{'='*60}")
    log(f"Phase 293: Component Contract — {model_name}")
    log(f"{'='*60}")

    model, tok = load_model(model_name)
    info = get_model_info(model, model_name); nl = info.n_layers
    log(f"Model: {info.model_class}, L={nl}, d={info.d_model}")
    # Warmup
    input_device = next(model.parameters()).device
    with torch.no_grad():
        try: model(**tok("warmup",return_tensors="pt").to(input_device).data)
        except: pass

    pairs = build_negation_pairs()
    log(f"Pairs: {len(pairs)} negation (6 subtypes)")

    MAX_LEN = 64
    layers = get_layers(model)

    # ====== CAPTURE ======
    log("\n=== CAPTURE ===")
    t0 = time.time()
    pair_data = {}  # {name: {A: {attn:{}, mlp:{}, hidden:{}}, B: {}}}
    pair_metrics = {}  # {name: {la, lb, kab, sa, sl, subtype, op_pos, operand_pos}}

    for pi, pr in enumerate(pairs):
        pn, sa, sb, st = pr["name"], pr["A"], pr["B"], pr["subtype"]
        cap_a = capture_all(model, tok, sa, nl, MAX_LEN)
        cap_b = capture_all(model, tok, sb, nl, MAX_LEN)
        pair_data[pn] = {"A": cap_a, "B": cap_b, "subtype": st}

        la = cap_a.get("logits")
        lb = cap_b.get("logits")
        if la is None or lb is None:
            log(f"  SKIP {pn}: no logits")
            continue
        kab = float(F.kl_div(F.log_softmax(la,-1), F.softmax(lb,-1), reduction='sum'))

        # Find positions
        sl = len(tok.encode(sb, add_special_tokens=True))
        op_pos = find_pos(tok, sb, pr["op"])
        operand_pos = find_pos(tok, sb, pr["operand"])

        pair_metrics[pn] = {"la": la, "lb": lb, "kab": kab, "sa": sa, "sl": sl,
                            "subtype": st, "op_pos": op_pos, "operand_pos": operand_pos}
        if (pi+1) % 20 == 0: log(f"  [{pi+1}/{len(pairs)}] {time.time()-t0:.0f}s")

    log(f"  Capture done: {len(pair_metrics)} pairs, {time.time()-t0:.0f}s")

    # ====== RUN EXPERIMENTS ======
    # Exp A: Full layer x component scan
    tA = time.time()
    resA = run_exp_A(model, tok, pair_data, pair_metrics, nl, MAX_LEN)
    log(f"  Exp A: {len(resA)} results, {time.time()-tA:.0f}s")

    # Determine top layers from Exp A (by natural_prog for each component)
    layer_np = defaultdict(list)
    for r in resA:
        layer_np[(r["layer"], r["component"])].append(r["natural_prog"])
    # Find top 5 layers per component
    top_by_comp = {}
    for comp in ['attn', 'mlp', 'resid_post']:
        scores = {}
        for (li, c), nps in layer_np.items():
            if c == comp:
                scores[li] = float(np.mean(nps))
        top5 = sorted(scores, key=scores.get, reverse=True)[:5]
        top_by_comp[comp] = top5
        log(f"  Top-5 {comp}: {top5} (NP: {[f'{scores[l]:.4f}' for l in top5]})")

    # Union of top layers for Exp B and C
    all_top = sorted(set(top_by_comp['attn'] + top_by_comp['mlp'] + top_by_comp['resid_post']))
    log(f"  Top layers for B/C: {all_top}")

    # Exp B: Position x Component
    tB = time.time()
    resB = run_exp_B(model, tok, pair_data, pair_metrics, nl, MAX_LEN, all_top)
    log(f"  Exp B: {len(resB)} results, {time.time()-tB:.0f}s")

    # Exp C: Alpha x Component (use top 3 per component to save time)
    top3 = sorted(set(top_by_comp['attn'][:3] + top_by_comp['mlp'][:3] + top_by_comp['resid_post'][:3]))
    tC = time.time()
    resC = run_exp_C(model, tok, pair_data, pair_metrics, nl, MAX_LEN, top3)
    log(f"  Exp C: {len(resC)} results, {time.time()-tC:.0f}s")

    # ====== ANALYSIS ======
    log("\n=== ANALYSIS ===")

    # A: Layer x Component summary
    a_agg = defaultdict(list)
    for r in resA:
        a_agg[(r["layer"], r["component"])].append(r)
    a_summary = {}
    for (li, comp), rows in sorted(a_agg.items()):
        progs = [r["progress"] for r in rows]
        krs = [r["kl_ratio"] for r in rows]
        nps = [r["natural_prog"] for r in rows]
        a_summary[f"L{li}_{comp}"] = {
            "layer": li, "component": comp,
            "mean_prog": round(float(np.mean(progs)), 4),
            "mean_kr": round(float(np.mean(krs)), 2),
            "mean_np": round(float(np.mean(nps)), 5),
            "n": len(progs)
        }

    # Find best layer per component
    for comp in ['attn', 'mlp', 'resid_post']:
        comp_rows = [(li, a_summary[f"L{li}_{comp}"]) for li in range(nl) if f"L{li}_{comp}" in a_summary]
        if comp_rows:
            best = max(comp_rows, key=lambda x: x[1]["mean_np"])
            log(f"  Best {comp}: L{best[0]} NP={best[1]['mean_np']:.5f} PROG={best[1]['mean_prog']:.4f} KR={best[1]['mean_kr']:.2f}")

    # Component comparison at best overall layer
    best_overall_layer = max(a_summary.items(), key=lambda x: x[1]["mean_np"])[1]["layer"]
    log(f"\n  Best overall layer: L{best_overall_layer}")
    for comp in ['attn', 'mlp', 'resid_post']:
        key = f"L{best_overall_layer}_{comp}"
        if key in a_summary:
            v = a_summary[key]
            log(f"    {comp}: NP={v['mean_np']:.5f} PROG={v['mean_prog']:.4f} KR={v['mean_kr']:.2f}")

    # B: Position effect summary
    b_agg = defaultdict(list)
    for r in resB:
        b_agg[(r["layer"], r["component"], r["pos_type"])].append(r)
    b_summary = {}
    for (li, comp, pt), rows in sorted(b_agg.items()):
        nps = [r["natural_prog"] for r in rows]
        progs = [r["progress"] for r in rows]
        krs = [r["kl_ratio"] for r in rows]
        b_summary[f"L{li}_{comp}_{pt}"] = {
            "layer": li, "component": comp, "pos_type": pt,
            "mean_prog": round(float(np.mean(progs)), 4),
            "mean_kr": round(float(np.mean(krs)), 2),
            "mean_np": round(float(np.mean(nps)), 5),
            "n": len(progs)
        }

    log(f"\n  POSITION EFFECT (resid_post component):")
    for key, v in sorted(b_summary.items()):
        if v["component"] == "resid_post":
            log(f"    {key}: NP={v['mean_np']:.5f} PROG={v['mean_prog']:.4f} KR={v['mean_kr']:.2f}")

    # C: Alpha curves
    c_agg = defaultdict(list)
    for r in resC:
        c_agg[(r["layer"], r["component"], r["alpha"])].append(r)
    c_summary = {}
    for (li, comp, alpha), rows in sorted(c_agg.items()):
        nps = [r["natural_prog"] for r in rows]
        progs = [r["progress"] for r in rows]
        krs = [r["kl_ratio"] for r in rows]
        c_summary[f"L{li}_{comp}_a{alpha:.2f}"] = {
            "layer": li, "component": comp, "alpha": alpha,
            "mean_prog": round(float(np.mean(progs)), 4),
            "mean_kr": round(float(np.mean(krs)), 2),
            "mean_np": round(float(np.mean(nps)), 5),
            "n": len(progs)
        }

    # ====== SAVE ======
    out_path = RESULT_DIR / f"{model_name}_component.json"
    save_data = {
        "model": model_name, "n_layers": nl, "n_pairs": len(pair_metrics),
        "top_layers": all_top, "top3_layers": top3,
        "exp_A_layer_component": a_summary,
        "exp_B_position": b_summary,
        "exp_C_alpha": c_summary,
        "exp_A_count": len(resA), "exp_B_count": len(resB), "exp_C_count": len(resC),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, default=str)
    log(f"\n  SAVED: {out_path}")

    release_model(model); del model; gc.collect(); torch.cuda.empty_cache()
    total = time.time()-t0
    log(f"Phase 293 {model_name} DONE. Total: {total:.0f}s")
    return save_data


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            run_phase293(name)
            gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
    else:
        run_phase293(model_name)

if __name__ == "__main__":
    main()
