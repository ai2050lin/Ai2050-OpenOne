"""
Phase 290: Recomputed Attention→MLP Contract Test
===================================================
CORE UPGRADE over Phase 289:
  - RECOMPUTED contract: inject attn, let MLP re-compute (NOT splice cached MLP)
  - Fine-grained α: [0, 0.25, 0.5, 0.75, 1.0, 1.25]
  - Naturalness check: MLP output norm after re-computation
  - Depth correction: track remaining layers per patch

EXPERIMENT: Replace attention output at layer L with α*A_attn+(1-α)*B_attn.
            Then let MLP at L re-compute naturally.
            → Does MLP compensate? Is the contract compatible?

Usage:
  python tests/glm5/phase290_recomputed_contract.py qwen3
  python tests/glm5/phase290_recomputed_contract.py glm4
  python tests/glm5/phase290_recomputed_contract.py deepseek7b
"""
import sys, os, json, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch, torch.nn.functional as F
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase290_recomputed")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp"); TMP_DIR.mkdir(parents=True, exist_ok=True)
_log_file = None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file, "a", encoding="utf-8") as f: f.write(line+"\n")
        except: pass

# ====== DATASET ======
def build_negation_pairs():
    pairs = []
    # 6 subtypes, ~7 each = 42 total
    for name, pos, neg, st in [
        ("happy","she is happy","she is not happy","lexical_not_adj"),
        ("open","the door is open","the door is not open","lexical_not_adj"),
        ("possible","victory is possible","victory is not possible","lexical_not_adj"),
        ("important","this is important","this is not important","lexical_not_adj"),
        ("clear","the answer is clear","the answer is not clear","lexical_not_adj"),
        ("safe","the area is safe","the area is not safe","lexical_not_adj"),
        ("fair","the decision is fair","the decision is not fair","lexical_not_adj"),
        ("agree","they agree with the proposal","they do not agree with the proposal","syntactic_do_not"),
        ("remember","i remember the meeting","i do not remember the meeting","syntactic_do_not"),
        ("know","she knows the answer","she does not know the answer","syntactic_do_not"),
        ("believe","he believes the story","he does not believe the story","syntactic_do_not"),
        ("support","they support the plan","they do not support the plan","syntactic_do_not"),
        ("accept","she accepts the offer","she does not accept the offer","syntactic_do_not"),
        ("trust","he trusts the source","he does not trust the source","syntactic_do_not"),
        ("nothing","he found something interesting","he found nothing interesting","existential_no"),
        ("no_one","someone came to the party","no one came to the party","existential_no"),
        ("no_food","there was some food left","there was no food left","existential_no"),
        ("no_idea","she had some idea what to do","she had no idea what to do","existential_no"),
        ("no_reason","there is a reason to worry","there is no reason to worry","existential_no"),
        ("no_doubt","there is some doubt about it","there is no doubt about it","existential_no"),
        ("no_evidence","there is evidence of fraud","there is no evidence of fraud","existential_no"),
        ("never_seen","i have seen it before","i have never seen it before","never"),
        ("never_been","she has been to Paris","she has never been to Paris","never"),
        ("never_gives","she sometimes gives up","she never gives up","never"),
        ("never_told","he told someone the secret","he never told anyone the secret","never"),
        ("impossible","the task is possible","the task is impossible","morphological_neg"),
        ("unacceptable","the proposal is acceptable","the proposal is unacceptable","morphological_neg"),
        ("incomplete","the report is complete","the report is incomplete","morphological_neg"),
        ("irrelevant","the comment is relevant","the comment is irrelevant","morphological_neg"),
        ("dishonest","the person is honest","the person is dishonest","morphological_neg"),
        ("uncertain","the result is certain","the result is uncertain","morphological_neg"),
        ("not_all","all birds can fly","not all birds can fly","scope_quantifier"),
        ("not_everyone","everyone agreed","not everyone agreed","scope_quantifier"),
        ("not_always","she always tells the truth","she does not always tell the truth","scope_quantifier"),
        ("not_necessarily","wealth means happiness","wealth does not necessarily mean happiness","scope_quantifier"),
        ("not_because","he left because he was angry","he did not leave because he was angry","scope_quantifier"),
        ("not_a_single","a single person helped","not a single person helped","scope_quantifier"),
    ]:
        pairs.append({"name":f"neg_{name}","A":pos,"B":neg,"category":"negation","subtype":st})
    return pairs

# ====== MODEL LOADING ======
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16, eager, device_map=auto)...")
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["path"], torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True, local_files_only=True, attn_implementation="eager")
    model.eval()
    dev = next(model.parameters()).device
    gpu = torch.cuda.memory_allocated()/1e9
    log(f"  Loaded: {dev}, GPU={gpu:.1f}GB, {type(model).__name__}")
    return model, tok, dev

# ====== CAPTURE ======
def capture_all_layers(model, tokenizer, sent, device, n_layers, max_len=48):
    layers = get_layers(model); al = min(n_layers, len(layers))
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    sl = inputs["input_ids"].shape[1]
    if sl < max_len:
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0,max_len-sl), value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0,max_len-sl), value=0)
    cap = {}; hooks = []
    for li in range(al):
        def mah(li):
            def hook(m,i,o): cap.setdefault(f"L{li}",{})["attn"] = (o[0] if isinstance(o,tuple) else o).detach().cpu().clone()
            return hook
        def mmh(li):
            def hook(m,i,o): cap.setdefault(f"L{li}",{})["mlp"] = (o[0] if isinstance(o,tuple) else o).detach().cpu().clone()
            return hook
        hooks.append(layers[li].self_attn.register_forward_hook(mah(li)))
        hooks.append(layers[li].mlp.register_forward_hook(mmh(li)))
    with torch.no_grad():
        try: model(**inputs)
        except Exception as e: log(f"  capture FAILED: {e}"); cap={}
    for h in hooks: h.remove()
    return cap

# ====== RECOMPUTED CONTRACT FORWARD ======
def forward_recomputed(model, tokenizer, sent_a, device, n_layers,
                        a_attn, b_attn, target_layer, alpha, max_len,
                        capture_mlp_norm=True):
    """
    RECOMPUTED contract test:
    1. Replace attn output at target_layer: (1-α)*A_attn + α*B_attn
    2. Let MLP at target_layer re-compute NATURALLY
    3. Capture MLP output norm + downstream L+1 norms for naturalness check
    
    Returns: logits, mlp_norm, downstream_norms
    """
    layers = get_layers(model)
    if target_layer >= len(layers): return None, None, None
    
    inputs = tokenizer(sent_a, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    sl = inputs["input_ids"].shape[1]
    if sl < max_len:
        inputs["input_ids"] = F.pad(inputs["input_ids"],(0,max_len-sl),value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"],(0,max_len-sl),value=0)

    o_proj = layers[target_layer].self_attn.o_proj
    dev_a = o_proj.weight.device; dtype_a = o_proj.weight.dtype

    # Interpolate attention output
    aa_t = a_attn.to(dev_a).to(dtype_a); ab_t = b_attn.to(dev_a).to(dtype_a)
    ms = min(aa_t.shape[1], ab_t.shape[1], max_len)
    pv = (1-alpha)*aa_t[:,:ms,:] + alpha*ab_t[:,:ms,:]

    hooks = []
    mlp_norm_cap = {}
    down_cap = {}

    # Hook 1: Replace attn output
    def make_attn_h(pv):
        def hook(m,i,o):
            if isinstance(o, tuple):
                no = (o[0].clone(),)+o[1:]; mms=min(pv.shape[1],no[0].shape[1]); no[0][:,:mms,:]=pv[:,:mms,:]
                return no
            else:
                no=o.clone(); mms=min(pv.shape[1],no.shape[1]); no[:,:mms,:]=pv[:,:mms,:]
                return no
        return hook
    hooks.append(layers[target_layer].self_attn.register_forward_hook(make_attn_h(pv)))

    # Hook 2: Capture MLP output at target_layer (NATURALLY re-computed!)
    if capture_mlp_norm:
        def make_mlp_cap(li):
            def hook(m,i,o):
                out = o[0] if isinstance(o,tuple) else o
                mlp_norm_cap["mlp_norm"] = float(out.float().norm())
            return hook
        hooks.append(layers[target_layer].mlp.register_forward_hook(make_mlp_cap(target_layer)))

    # Hook 3: Capture L+1 norms (downstream check)
    if target_layer + 1 < len(layers):
        def make_down_h(li):
            def hook(m,i,o):
                if isinstance(i,tuple) and len(i)>0: down_cap["resid_in"] = float(i[0].float().norm())
                out = o[0] if isinstance(o,tuple) else o
                down_cap["layer_out"] = float(out.float().norm())
            return hook
        hooks.append(layers[target_layer+1].register_forward_hook(make_down_h(target_layer+1)))

    try:
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0,-1,:].detach().cpu().float().clone()
    except Exception as e:
        log(f"    RECOMP ERR L{target_layer} α={alpha:.2f}: {e}")
        logits = None; mlp_norm_cap = {}; down_cap = {}

    for h in hooks: h.remove()
    return logits, mlp_norm_cap, down_cap


def forward_spliced_both(model, tokenizer, sent_a, device, n_layers,
                          a_attn, b_attn, a_mlp, b_mlp, target_layer, alpha, max_len):
    """For comparison: splice BOTH attn and MLP (no re-computation)."""
    layers = get_layers(model)
    inputs = tokenizer(sent_a, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    sl = inputs["input_ids"].shape[1]
    if sl < max_len:
        inputs["input_ids"] = F.pad(inputs["input_ids"],(0,max_len-sl),value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"],(0,max_len-sl),value=0)

    da = layers[target_layer].self_attn.o_proj.weight.device
    dta = layers[target_layer].self_attn.o_proj.weight.dtype
    dm = next(layers[target_layer].mlp.parameters()).device
    dtm = next(layers[target_layer].mlp.parameters()).dtype

    def interp(a,b,alpha,dev,dtype):
        at,bt=a.to(dev).to(dtype),b.to(dev).to(dtype)
        ms=min(at.shape[1],bt.shape[1],max_len)
        return (1-alpha)*at[:,:ms,:]+alpha*bt[:,:ms,:]

    pv_a = interp(a_attn,b_attn,alpha,da,dta)
    pv_m = interp(a_mlp,b_mlp,alpha,dm,dtm)

    hooks=[]
    def mah(pv):
        def hook(m,i,o):
            no=(o[0].clone(),)+o[1:] if isinstance(o,tuple) else o.clone()
            ref=no[0] if isinstance(o,tuple) else no
            ms=min(pv.shape[1],ref.shape[1]); ref[:,:ms,:]=pv[:,:ms,:]
            return no if isinstance(o,tuple) else no
        return hook
    hooks.append(layers[target_layer].self_attn.register_forward_hook(mah(pv_a)))
    hooks.append(layers[target_layer].mlp.register_forward_hook(mah(pv_m)))

    try:
        with torch.no_grad():
            out=model(**inputs); res=out.logits[0,-1,:].detach().cpu().float().clone()
    except: res=None
    for h in hooks: h.remove()
    return res

# ====== METRICS ======
def compute_metrics(logits, la, lb, kab, mlp_n, mlp_na, mlp_nb, down_n, down_na, down_nb):
    if logits is None: return None
    kp = float(F.kl_div(F.log_softmax(logits,-1),F.softmax(lb,-1),reduction='sum'))
    kr = min(kp/max(kab,1e-6), 100.0)
    db,dp = lb-la, logits-la
    nb,np = float(torch.norm(db)),float(torch.norm(dp))
    cd = float(torch.dot(dp,db)/(nb*np)) if nb>1e-8 and np>1e-8 else 0
    prog = cd*min(np/nb,2.0) if nb>1e-8 else 0
    
    # Naturalness: MLP output norm ratio (vs A's natural mlp norm)
    mlp_nr = mlp_n/max(mlp_na,1e-8) if mlp_n>0 and mlp_na>0 else 0
    # MLP compensation: is re-computed MLP closer to A or B?
    mlp_to_b = abs(mlp_n - mlp_nb)/max(abs(mlp_na-mlp_nb),1e-8) if mlp_n>0 and mlp_na>0 and mlp_nb>0 else 0
    
    # Downstream amplification
    down_amp = down_n/max(down_na,1e-8) if down_n>0 and down_na>0 else 0
    
    return {"kl_ratio":kr,"progress":prog,"cos_dir":cd,
            "mlp_norm_ratio":mlp_nr,"mlp_to_b_ratio":mlp_to_b,"down_amp":down_amp}

# ====== MAIN ======
def run_phase290(model_name):
    global _log_file
    _log_file = str(TMP_DIR/f"phase290_{model_name}.txt")
    log(f"{'='*60}")
    log(f"Phase 290: Recomputed Contract Test — {model_name}")
    log(f"{'='*60}")

    model, tok, dev = load_model_bf16(model_name)
    info = get_model_info(model, model_name); nl = info.n_layers
    log(f"Model: {info.model_class}, L={nl}, d={info.d_model}")
    with torch.no_grad():
        try: model(**tok("warmup",return_tensors="pt").to(dev))
        except: pass

    pairs = build_negation_pairs()
    log(f"Pairs: {len(pairs)} negation, 6 subtypes")
    for st in sorted(set(p["subtype"] for p in pairs)):
        log(f"  {st}: {sum(1 for p in pairs if p['subtype']==st)}")

    ALPHAS = [0, 0.25, 0.5, 0.75, 1.0, 1.25]
    MAX_LEN = 48

    # ====== CAPTURE ======
    log("\nPhase 1: Capture...")
    t0 = time.time()
    pd, pm = {}, {}
    layers = get_layers(model)

    for pi, pr in enumerate(pairs):
        pn, sa, sb, st = pr["name"], pr["A"], pr["B"], pr["subtype"]
        toks_a = len(tok.encode(sa, add_special_tokens=True))
        toks_b = len(tok.encode(sb, add_special_tokens=True))
        cl = min(max(toks_a, toks_b), MAX_LEN)
        oa = capture_all_layers(model, tok, sa, dev, nl, cl)
        ob = capture_all_layers(model, tok, sb, dev, nl, cl)
        if oa and ob: pd[pn] = {"A":oa,"B":ob,"seq_len":cl,"subtype":st}
        ia = tok(sa, return_tensors="pt", truncation=True, max_length=cl).to(dev)
        ib = tok(sb, return_tensors="pt", truncation=True, max_length=cl).to(dev)
        with torch.no_grad():
            la = model(**ia).logits[0,-1,:].detach().cpu().float()
            lb = model(**ib).logits[0,-1,:].detach().cpu().float()
        kab = float(F.kl_div(F.log_softmax(la,-1),F.softmax(lb,-1),reduction='sum'))
        pm[pn] = {"logits_a":la,"logits_b":lb,"kl_ab":kab,"sent_a":sa,"seq_len":cl,"subtype":st}
        if (pi+1)%15==0: log(f"  [{pi+1}/{len(pairs)}] {time.time()-t0:.0f}s")
    log(f"  Capture done: {len(pd)} pairs, {time.time()-t0:.0f}s")

    # Collect natural MLP norms for A and B at each layer
    log("  Computing natural MLP norms...")
    nat_mlp = {"A":{},"B":{}}  # {L{li}: {"A": [norms...], "B": [...]}}
    for li in range(0, nl, 2):
        lk = f"L{li}"
        nat_mlp["A"].setdefault(lk,[]); nat_mlp["B"].setdefault(lk,[])
        for pn in pd:
            for sk, lab in [("A","A"),("B","B")]:
                if lk in pd[pn][lab]:
                    v = pd[pn][lab][lk].get("mlp")
                    if v is not None:
                        nat_mlp[lab].setdefault(lk,[]).append(float(v.float().norm()))

    # Downstream norms: capture natural L+1 norms for baseline
    nat_down = {}
    for li in range(0, nl-1, 2):
        lk = f"L{li+1}"
        nat_down[lk] = {}
        for pn in list(pd.keys())[:5]:  # sample 5 pairs for efficiency
            sa = pm[pn]["sent_a"]; cl = pm[pn]["seq_len"]
            inputs = tok(sa, return_tensors="pt", truncation=True, max_length=cl).to(dev)
            with torch.no_grad():
                cap = {}
                def dh(l):
                    def hook(m,i,o): cap["n"]=float((o[0] if isinstance(o,tuple) else o).float().norm())
                    return hook
                h=layers[li+1].register_forward_hook(dh(li+1))
                model(**inputs); h.remove()
                nat_down[lk][pn] = cap.get("n",0)

    # ====== RECOMPUTED CONTRACT TEST ======
    log(f"\nPhase 2: Recomputed Contract ({nl//2} layers × {len(ALPHAS)}α)")
    
    test_layers = list(range(0, nl, 2))
    all_results = []
    fail_count = 0
    t0 = time.time()

    for pi, pn in enumerate(pd.keys()):
        m = pm[pn]; d = pd[pn]
        la, lb, kab, sa, cl, st = m["logits_a"], m["logits_b"], m["kl_ab"], m["sent_a"], m["seq_len"], m["subtype"]
        if kab < 1e-6: continue

        for tl in test_layers:
            lk = f"L{tl}"; lk_n = f"L{tl+1}"
            if lk not in d["A"] or lk not in d["B"]: continue
            a_attn = d["A"][lk].get("attn"); b_attn = d["B"][lk].get("attn")
            a_mlp = d["A"][lk].get("mlp"); b_mlp = d["B"][lk].get("mlp")
            if any(x is None for x in [a_attn,b_attn,a_mlp,b_mlp]): continue

            # Natural norms for this layer
            mlp_na = np.mean(nat_mlp["A"].get(lk,[1.0])) if nat_mlp["A"].get(lk) else 1.0
            mlp_nb = np.mean(nat_mlp["B"].get(lk,[1.0])) if nat_mlp["B"].get(lk) else 1.0
            down_na = np.mean(list(nat_down.get(lk_n,{}).values())) if nat_down.get(lk_n) else 1.0
            down_nb = down_na  # approximation

            for alpha in ALPHAS:
                # === RECOMPUTED: replace attn, let MLP re-compute ===
                logits_r, mlp_norms_r, down_norms_r = forward_recomputed(
                    model, tok, sa, dev, nl, a_attn, b_attn, tl, alpha, cl)
                mlp_n = mlp_norms_r.get("mlp_norm",0) if mlp_norms_r else 0
                down_n = down_norms_r.get("layer_out",0) if down_norms_r else 0
                m_r = compute_metrics(logits_r, la, lb, kab, mlp_n, mlp_na, mlp_nb, down_n, down_na, down_nb)
                if m_r: all_results.append({"pname":pn,"subtype":st,"layer":tl,"alpha":alpha,
                    "mode":"recomp","kl_ab":kab,**m_r,"remaining_layers":nl-tl-1})
                else: fail_count += 1

                # === SPLICED BOTH (for comparison) ===
                if alpha in [0, 0.5, 1.0]:
                    logits_s = forward_spliced_both(model, tok, sa, dev, nl,
                        a_attn, b_attn, a_mlp, b_mlp, tl, alpha, cl)
                    m_s = compute_metrics(logits_s, la, lb, kab, 0, 0, 0, 0, 0, 0)
                    if m_s: all_results.append({"pname":pn,"subtype":st,"layer":tl,"alpha":alpha,
                        "mode":"spliced","kl_ab":kab,**m_s,"remaining_layers":nl-tl-1})
                    else: fail_count += 1

        if (pi+1)%10==0:
            e=time.time()-t0; rem = len(pd)-pi-1
            eta = e/max(pi+1,1)*rem if pi>0 else 0
            log(f"  [{pi+1}/{len(pd)}] {e:.0f}s, {len(all_results)} results, {fail_count} fails, "
                f"ETA={eta/60:.0f}min, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

    t_patch = time.time()-t0
    log(f"  Done: {len(all_results)} results, {fail_count} fails, {t_patch:.0f}s ({t_patch/60:.1f}min)")

    # ====== ANALYSIS ======
    log(f"\n{'='*60}"); log("ANALYSIS"); log(f"{'='*60}")

    # === Per-layer recomputed curves (α=1.0) ===
    log(f"\n  === LAYER CURVE: Recomputed (α=1.0) ===")
    log(f"  {'Layer':>6} {'RemainL':>7} {'Attn_Prog':>9} {'Attn_KR':>9} {'MLP_NR':>9} {'MLP_toB':>9} {'Down_Amp':>9} {'Spliced_Prog':>9}")
    
    lay_recomp = defaultdict(list)
    lay_spliced = defaultdict(list)
    for r in all_results:
        if r["alpha"]==1.0:
            if r["mode"]=="recomp": lay_recomp[r["layer"]].append(r)
            else: lay_spliced[r["layer"]].append(r)
    
    lc = {}
    for li in sorted(lay_recomp.keys()):
        dr = lay_recomp[li]
        ds = lay_spliced.get(li, [])
        v = {}
        v["prog"] = float(np.mean([x["progress"] for x in dr]))
        v["kr"] = float(np.mean([x["kl_ratio"] for x in dr]))
        v["mlp_nr"] = float(np.mean([x["mlp_norm_ratio"] for x in dr if x["mlp_norm_ratio"]>0]))
        v["mlp_tob"] = float(np.mean([x["mlp_to_b_ratio"] for x in dr if x["mlp_to_b_ratio"]>0]))
        v["down_amp"] = float(np.mean([x["down_amp"] for x in dr if x["down_amp"]>0]))
        v["remain"] = int(np.mean([x.get("remaining_layers",0) for x in dr]))
        v["spliced_prog"] = float(np.mean([x["progress"] for x in ds])) if ds else 0
        lc[str(li)] = v
        log(f"  {li:>6} {v['remain']:>7} {v['prog']:9.4f} {v['kr']:9.3f} {v['mlp_nr']:9.3f} {v['mlp_tob']:9.3f} {v['down_amp']:9.3f} {v['spliced_prog']:9.4f}")

    # === α interpolation at strongest layer ===
    bl = max(lc.keys(), key=lambda l: lc[l].get("prog",0))
    log(f"\n  === α INTERPOLATION at L{bl} (recomputed) ===")
    log(f"  {'α':>6} {'Prog':>9} {'KR':>9} {'MLP_NR':>9} {'MLP_toB':>9} {'Down_Amp':>9} {'SplicedProg':>10}")
    
    ala = defaultdict(list); als = defaultdict(list)
    for r in all_results:
        if r["layer"]==int(bl):
            if r["mode"]=="recomp": ala[r["alpha"]].append(r)
            else: als[r["alpha"]].append(r)
    
    alh = {}
    for alpha in sorted(ala.keys()):
        dr = ala[alpha]; ds = als.get(alpha,[])
        v={}
        v["prog"]=float(np.mean([x["progress"] for x in dr]))
        v["kr"]=float(np.mean([x["kl_ratio"] for x in dr]))
        v["mlp_nr"]=float(np.mean([x["mlp_norm_ratio"] for x in dr if x["mlp_norm_ratio"]>0]))
        v["mlp_tob"]=float(np.mean([x["mlp_to_b_ratio"] for x in dr if x["mlp_to_b_ratio"]>0]))
        v["down_amp"]=float(np.mean([x["down_amp"] for x in dr if x["down_amp"]>0]))
        v["spliced_p"]=float(np.mean([x["progress"] for x in ds])) if ds else 0
        alh[str(alpha)] = v
        log(f"  {alpha:>6.2f} {v['prog']:9.4f} {v['kr']:9.3f} {v['mlp_nr']:9.3f} {v['mlp_tob']:9.3f} {v['down_amp']:9.3f} {v['spliced_p']:10.4f}")

    # === RECOMP vs SPLICED comparison ===
    log(f"\n  === RECOMP vs SPLICED (α=1.0, all layers) ===")
    rec_progs = [r["progress"] for r in all_results if r["alpha"]==1.0 and r["mode"]=="recomp"]
    spl_progs = [r["progress"] for r in all_results if r["alpha"]==1.0 and r["mode"]=="spliced"]
    if rec_progs and spl_progs:
        log(f"    recomp mean_prog={np.mean(rec_progs):.4f}, spliced mean_prog={np.mean(spl_progs):.4f}")
        log(f"    recomp > spliced in {sum(1 for a,b in zip(rec_progs,spl_progs) if a>b)}/{len(rec_progs)} layers")
    
    # === MLP COMPENSATION: does mlp_to_b correlate with progress? ===
    log(f"\n  === MLP COMPENSATION (α=1.0) ===")
    for li in sorted(lc.keys()):
        mtb = lc[li].get("mlp_tob",0)
        prog = lc[li].get("prog",0)
        mlp_nr = lc[li].get("mlp_nr",0)
        flag = ""
        if mtb < 0.3: flag = "MLP→A"  # MLP re-computed output closer to A (rejected B's attn)
        elif mtb > 0.7: flag = "MLP→B"  # MLP closer to B (accepted B's attn)
        elif mtb > 0.3: flag = "partial"
        log(f"    L{li}: mlp_to_B={mtb:.3f}, prog={prog:.4f}, mlp_nr={mlp_nr:.3f} [{flag}]")

    # === Subtype breakdown ===
    log(f"\n  === SUBTYPE BREAKDOWN (recomp α=1.0, top-5 layers) ===")
    top5 = sorted(lc.keys(), key=lambda l: lc[l].get("prog",0), reverse=True)[:5]
    st_agg = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if r["alpha"]==1.0 and r["mode"]=="recomp" and str(r["layer"]) in top5:
            st_agg[r["subtype"]][r["layer"]].append(r)
    for st in sorted(st_agg.keys()):
        parts = [f"  {st:>25}"]
        for tl in top5:
            d = st_agg[st].get(int(tl),[])
            p = float(np.mean([x["progress"] for x in d])) if d else 0
            mtb = float(np.mean([x["mlp_to_b_ratio"] for x in d if x["mlp_to_b_ratio"]>0])) if d else 0
            parts.append(f" L{tl}:p={p:.2f}/mtb={mtb:.2f}")
        log(" ".join(parts))

    # === Depth-corrected effect ===
    log(f"\n  === DEPTH-CORRECTED (prog / remaining_layers) ===")
    for li in sorted(lc.keys()):
        raw_p = lc[li].get("prog",0); rem = max(lc[li].get("remain",1), 1)
        dc = raw_p / rem
        log(f"    L{li}: raw_prog={raw_p:.4f}, remain={rem}, depth_corrected={dc:.6f}")

    # ====== SAVE ======
    save = {"model":model_name,"info":{"class":info.model_class,"L":nl,"d":info.d_model},
            "n_pairs":len(pairs),"n_results":len(all_results),"n_fails":fail_count,
            "alphas":ALPHAS,"layer_curve":lc,"alpha_curve":alh,"best_layer":bl}
    sp = RESULT_DIR/f"{model_name}_recomp.json"
    with open(sp,"w") as f: json.dump(save,f,indent=2,default=str)
    log(f"\nSaved to {sp}")
    release_model(model); gc.collect(); torch.cuda.empty_cache()
    return save

if __name__=="__main__":
    mn = sys.argv[1] if len(sys.argv)>1 else "qwen3"
    if mn=="all":
        for n in ["qwen3","glm4","deepseek7b"]:
            try:
                r=run_phase290(n); log(f"\n{n} DONE: {r['n_results']} results")
            except Exception as e:
                log(f"!!! {n} FAILED: {e}"); import traceback; traceback.print_exc()
            gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
    else: run_phase290(mn)
