"""
Phase 291: Multi-Layer Block Recomputational Contract Test
===========================================================
KEY UPGRADE over Phase 290:
  - Block-level recomputation: replace attention for ALL layers in a 4-layer block,
    let ALL MLPs in the block re-compute naturally
  - This tests distributed functional contracts, not just single-layer
  - Compare block-level vs single-layer effects

BLOCK DESIGN:  4 consecutive layers per block
  Qwen3 (36L): [0-3],[4-7],[8-11],[12-15],[16-19],[20-23],[24-27],[28-31],[32-35]
  GLM4  (40L): [0-3],[4-7],...,[36-39]
  DS7B  (28L): [0-3],[4-7],...,[24-27]

DEEP LAYER FIX: Use output_t.device in hooks (not o_proj.weight.device).
                Works for both GPU and CPU-offloaded layers.

Usage:
  python tests/glm5/phase291_multilayer_block.py qwen3
  python tests/glm5/phase291_multilayer_block.py glm4
  python tests/glm5/phase291_multilayer_block.py deepseek7b
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

RESULT_DIR = Path("results/phase291_multilayer_block")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp"); TMP_DIR.mkdir(parents=True, exist_ok=True)
_log_file = None
DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file,"a",encoding="utf-8") as f: f.write(line+"\n")
        except: pass

# ====== DATASET: Expanded Negation (80 pairs, 6 subtypes) ======
def build_negation_pairs():
    pairs = []
    # lexical_not_adj
    for name, pos, neg in [
        ("happy","she is happy","she is not happy"),("open","the door is open","the door is not open"),
        ("possible","victory is possible","victory is not possible"),("ready","they are ready","they are not ready"),
        ("important","this is important","this is not important"),("clear","the answer is clear","the answer is not clear"),
        ("safe","the area is safe","the area is not safe"),("fair","the decision is fair","the decision is not fair"),
        ("simple","the problem is simple","the problem is not simple"),("correct","your answer is correct","your answer is not correct"),
        ("reasonable","the price is reasonable","the price is not reasonable"),
        ("visible","the star is visible","the star is not visible"),("stable","the system is stable","the system is not stable"),
    ]:
        pairs.append({"name":f"neg_adj_{name}","A":pos,"B":neg,"category":"negation","subtype":"lexical_not_adj"})
    # syntactic_do_not
    for name, pos, neg in [
        ("agree","they agree with the proposal","they do not agree with the proposal"),
        ("remember","i remember the meeting","i do not remember the meeting"),
        ("understand","we understand the problem","we do not understand the problem"),
        ("know","she knows the answer","she does not know the answer"),
        ("believe","he believes the story","he does not believe the story"),
        ("support","they support the plan","they do not support the plan"),
        ("accept","she accepts the offer","she does not accept the offer"),
        ("expect","we expect rain","we do not expect rain"),
        ("trust","he trusts the source","he does not trust the source"),
        ("follow","they follow the rules","they do not follow the rules"),
        ("recognize","she recognizes the face","she does not recognize the face"),
        ("recommend","i recommend this book","i do not recommend this book"),
        ("require","the task requires effort","the task does not require effort"),
        ("confirm","the test confirms the theory","the test does not confirm the theory"),
    ]:
        pairs.append({"name":f"neg_verb_{name}","A":pos,"B":neg,"category":"negation","subtype":"syntactic_do_not"})
    # existential_no
    for name, pos, neg in [
        ("nothing","he found something interesting","he found nothing interesting"),
        ("no_one","someone came to the party","no one came to the party"),
        ("no_food","there was some food left","there was no food left"),
        ("no_idea","she had some idea what to do","she had no idea what to do"),
        ("no_reason","there is a reason to worry","there is no reason to worry"),
        ("no_choice","they had a choice in the matter","they had no choice in the matter"),
        ("no_doubt","there is some doubt about it","there is no doubt about it"),
        ("no_evidence","there is evidence of fraud","there is no evidence of fraud"),
        ("no_hope","there is some hope left","there is no hope left"),
        ("no_sign","there is a sign of life","there is no sign of life"),
        ("no_animal","an animal crossed the road","no animal crossed the road"),
        ("no_visitor","a visitor arrived today","no visitor arrived today"),
    ]:
        pairs.append({"name":f"neg_no_{name}","A":pos,"B":neg,"category":"negation","subtype":"existential_no"})
    # never
    for name, pos, neg in [
        ("seen","i have seen it before","i have never seen it before"),
        ("been","she has been to Paris","she has never been to Paris"),
        ("told","he told someone the secret","he never told anyone the secret"),
        ("gives_up","she sometimes gives up","she never gives up"),
        ("forgets","he sometimes forgets names","he never forgets a face"),
        ("complains","she often complains","she never complains"),
        ("tells_truth","he sometimes tells the truth","he never tells the truth"),
        ("late","she is sometimes late","she is never late"),
        ("apologizes","he sometimes apologizes","he never apologizes"),
        ("admits","she sometimes admits mistakes","she never admits mistakes"),
    ]:
        pairs.append({"name":f"neg_never_{name}","A":pos,"B":neg,"category":"negation","subtype":"never"})
    # morphological_neg
    for name, pos, neg in [
        ("impossible","the task is possible","the task is impossible"),
        ("unacceptable","the proposal is acceptable","the proposal is unacceptable"),
        ("incomplete","the report is complete","the report is incomplete"),
        ("irrelevant","the comment is relevant","the comment is irrelevant"),
        ("dishonest","the person is honest","the person is dishonest"),
        ("unfair","the treatment was fair","the treatment was unfair"),
        ("unlikely","the outcome is likely","the outcome is unlikely"),
        ("incorrect","the assumption is correct","the assumption is incorrect"),
        ("irregular","the pattern is regular","the pattern is irregular"),
        ("disagree","they agree on the terms","they disagree on the terms"),
        ("uncertain","the result is certain","the result is uncertain"),
        ("disobey","the soldiers obey orders","the soldiers disobey orders"),
    ]:
        pairs.append({"name":f"neg_prefix_{name}","A":pos,"B":neg,"category":"negation","subtype":"morphological_neg"})
    # scope_quantifier
    for name, pos, neg in [
        ("not_all","all birds can fly","not all birds can fly"),
        ("not_everyone","everyone agreed","not everyone agreed"),
        ("not_always","she always tells the truth","she does not always tell the truth"),
        ("not_entirely","the plan is entirely successful","the plan is not entirely successful"),
        ("not_necessarily","wealth means happiness","wealth does not necessarily mean happiness"),
        ("not_only","he is rich","he is not only rich but also kind"),
        ("not_exactly","that is exactly what i meant","that is not exactly what i meant"),
        ("not_quite","the work is finished","the work is not quite finished"),
        ("not_particularly","the movie was interesting","the movie was not particularly interesting"),
        ("not_completely","the glass is full","the glass is not completely full"),
        ("not_because","he left because he was angry","he did not leave because he was angry"),
        ("not_if","she will come if invited","she will not come if invited"),
        ("not_a_single","a single person helped","not a single person helped"),
        ("not_even_one","he ate one cookie","he did not eat even one cookie"),
        ("not_any","there are some problems","there are not any problems"),
        ("not_once","she called once","she did not call once"),
    ]:
        pairs.append({"name":f"neg_scope_{name}","A":pos,"B":neg,"category":"negation","subtype":"scope_quantifier"})
    return pairs

# ====== MODEL LOADING ======
def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name} (bf16, device_map=auto)...")
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
    log(f"  GPU: {len(gpu_l)}{' ('+str(gpu_l[0])+'-'+str(gpu_l[-1])+')' if gpu_l else ''}, CPU: {len(cpu_l)}{' ('+str(cpu_l[0])+'-'+str(cpu_l[-1])+')' if cpu_l else ''}")
    return model, tok

# ====== CAPTURE ======
def capture_all(model, tokenizer, sent, n_layers, max_len=64):
    layers = get_layers(model); al = min(n_layers, len(layers))
    inputs = tokenizer(sent,return_tensors="pt",truncation=True,max_length=max_len).to(DEV)
    sl = inputs["input_ids"].shape[1]
    if sl < max_len:
        inputs["input_ids"]=F.pad(inputs["input_ids"],(0,max_len-sl),value=tokenizer.pad_token_id)
        inputs["attention_mask"]=F.pad(inputs["attention_mask"],(0,max_len-sl),value=0)
    cap={}; hooks=[]
    for li in range(al):
        def mah(li):
            def hook(m,i,o): cap.setdefault(f"L{li}",{})["attn"]=(o[0] if isinstance(o,tuple) else o).detach().cpu().clone()
            return hook
        def mmh(li):
            def hook(m,i,o): cap.setdefault(f"L{li}",{})["mlp"]=(o[0] if isinstance(o,tuple) else o).detach().cpu().clone()
            return hook
        hooks.append(layers[li].self_attn.register_forward_hook(mah(li)))
        hooks.append(layers[li].mlp.register_forward_hook(mmh(li)))
    with torch.no_grad():
        try: model(**inputs)
        except Exception as e: log(f"  capture ERR: {e}"); cap={}
    for h in hooks: h.remove()
    return cap

# ====== MULTI-LAYER BLOCK RECOMPUTED FORWARD ======
def forward_block_recomp(model, tokenizer, sent_a, n_layers, all_attn_a, all_attn_b,
                          block_start, block_size, alpha, max_len):
    """
    Replace attention for ALL layers in [block_start, block_start+block_size-1].
    Let all MLPs in the block re-compute naturally.
    Capture MLP norms for last layer in block.
    
    DEEP LAYER FIX: use output_t.device in hooks.
    """
    layers = get_layers(model)
    block_end = min(block_start + block_size, n_layers)
    
    inputs = tokenizer(sent_a,return_tensors="pt",truncation=True,max_length=max_len).to(DEV)
    sl = inputs["input_ids"].shape[1]
    if sl < max_len:
        inputs["input_ids"]=F.pad(inputs["input_ids"],(0,max_len-sl),value=tokenizer.pad_token_id)
        inputs["attention_mask"]=F.pad(inputs["attention_mask"],(0,max_len-sl),value=0)

    hooks = []; mlp_caps = {}; down_cap = {}

    for li in range(block_start, block_end):
        lk = f"L{li}"
        aa = all_attn_a.get(lk,{}).get("attn")
        ab = all_attn_b.get(lk,{}).get("attn")
        if aa is None or ab is None: continue
        
        ms = min(aa.shape[1], ab.shape[1], max_len)
        pv_cpu = (1-alpha)*aa[:,:ms,:].float() + alpha*ab[:,:ms,:].float()
        
        # Hook: replace attn output (deep fix: dynamic device)
        def mah(pv_cpu):
            def hook(m,i,o):
                ref = o[0] if isinstance(o,tuple) else o
                pv = pv_cpu.to(ref.device).to(ref.dtype)
                mms = min(pv.shape[1], ref.shape[1])
                if isinstance(o,tuple): no=(o[0].clone(),)+o[1:]; no[0][:,:mms,:]=pv[:,:mms,:]; return no
                no=o.clone(); no[:,:mms,:]=pv[:,:mms,:]; return no
            return hook
        hooks.append(layers[li].self_attn.register_forward_hook(mah(pv_cpu)))
        
        # Hook: capture MLP norm after natural recomputation
        def mcap(li):
            def hook(m,i,o):
                out = o[0] if isinstance(o,tuple) else o
                mlp_caps[f"L{li}"] = float(out.float().norm())
            return hook
        hooks.append(layers[li].mlp.register_forward_hook(mcap(li)))

    # Capture block-end+1 norms
    if block_end < n_layers:
        def dcap(li):
            def hook(m,i,o): down_cap["layer_out"]=float((o[0] if isinstance(o,tuple) else o).float().norm())
            return hook
        hooks.append(layers[block_end].register_forward_hook(dcap(block_end)))

    try:
        with torch.no_grad(): out=model(**inputs); logits=out.logits[0,-1,:].detach().cpu().float().clone()
    except Exception as e:
        log(f"    BLOCK ERR [{block_start}-{block_end-1}] α={alpha}: {e}")
        logits=None; mlp_caps={}; down_cap={}
    for h in hooks: h.remove()
    return logits, mlp_caps, down_cap


def forward_single_recomp(model, tokenizer, sent_a, n_layers, all_attn_a, all_attn_b,
                           target_layer, alpha, max_len):
    """Single-layer recomputation (for comparison with block)."""
    layers = get_layers(model)
    if target_layer >= n_layers: return None, None, None
    
    inputs = tokenizer(sent_a,return_tensors="pt",truncation=True,max_length=max_len).to(DEV)
    sl = inputs["input_ids"].shape[1]
    if sl < max_len:
        inputs["input_ids"]=F.pad(inputs["input_ids"],(0,max_len-sl),value=tokenizer.pad_token_id)
        inputs["attention_mask"]=F.pad(inputs["attention_mask"],(0,max_len-sl),value=0)

    lk = f"L{target_layer}"
    aa = all_attn_a.get(lk,{}).get("attn"); ab = all_attn_b.get(lk,{}).get("attn")
    if aa is None or ab is None: return None, None, None

    ms = min(aa.shape[1], ab.shape[1], max_len)
    pv_cpu = (1-alpha)*aa[:,:ms,:].float() + alpha*ab[:,:ms,:].float()

    hooks = []; mlp_cap = {}; down_cap = {}

    def mah(pv_cpu):
        def hook(m,i,o):
            ref = o[0] if isinstance(o,tuple) else o; pv = pv_cpu.to(ref.device).to(ref.dtype)
            mms = min(pv.shape[1], ref.shape[1])
            if isinstance(o,tuple): no=(o[0].clone(),)+o[1:]; no[0][:,:mms,:]=pv[:,:mms,:]; return no
            no=o.clone(); no[:,:mms,:]=pv[:,:mms,:]; return no
        return hook
    hooks.append(layers[target_layer].self_attn.register_forward_hook(mah(pv_cpu)))

    def mcap(li):
        def hook(m,i,o): mlp_cap["mlp_norm"]=float((o[0] if isinstance(o,tuple) else o).float().norm())
        return hook
    hooks.append(layers[target_layer].mlp.register_forward_hook(mcap(target_layer)))

    if target_layer + 1 < n_layers:
        def dcap(li):
            def hook(m,i,o): down_cap["layer_out"]=float((o[0] if isinstance(o,tuple) else o).float().norm())
            return hook
        hooks.append(layers[target_layer+1].register_forward_hook(dcap(target_layer+1)))

    try:
        with torch.no_grad(): out=model(**inputs); logits=out.logits[0,-1,:].detach().cpu().float().clone()
    except Exception as e: log(f"    SINGLE ERR L{target_layer}: {e}"); logits=None
    for h in hooks: h.remove()
    return logits, mlp_cap, down_cap

# ====== METRICS ======
def compute_metrics(logits, la, lb, kab):
    if logits is None: return None
    kp=float(F.kl_div(F.log_softmax(logits,-1),F.softmax(lb,-1),reduction='sum'))
    kr=min(kp/max(kab,1e-6),100.0)
    db,dp=lb-la,logits-la
    nb,np=float(torch.norm(db)),float(torch.norm(dp))
    cd=float(torch.dot(dp,db)/(nb*np)) if nb>1e-8 and np>1e-8 else 0
    prog=cd*min(np/nb,2.0) if nb>1e-8 else 0
    return {"kl_ratio":kr,"progress":prog,"cos_dir":cd}

# ====== MAIN ======
def run_phase291(model_name):
    global _log_file
    _log_file = str(TMP_DIR/f"phase291_{model_name}.txt")
    log(f"{'='*60}")
    log(f"Phase 291: Multi-Layer Block Recomp Contract — {model_name}")
    log(f"{'='*60}")

    model, tok = load_model(model_name)
    info = get_model_info(model, model_name); nl = info.n_layers
    log(f"Model: {info.model_class}, L={nl}, d={info.d_model}")
    with torch.no_grad():
        try: model(**tok("warmup",return_tensors="pt").to(DEV))
        except: pass

    pairs = build_negation_pairs()
    log(f"Pairs: {len(pairs)} negation (6 subtypes)")
    for st in sorted(set(p["subtype"] for p in pairs)):
        n = sum(1 for p in pairs if p["subtype"]==st)
        log(f"  {st}: {n}")

    BLOCK_SIZE = 4
    ALPHAS = [0, 0.5, 1.0]
    MAX_LEN = 64
    layers = get_layers(model)

    # ====== CAPTURE ======
    log("\nPhase 1: Capture all layers...")
    t0 = time.time(); pd, pm = {}, {}
    for pi,pr in enumerate(pairs):
        pn,sa,sb,st=pr["name"],pr["A"],pr["B"],pr["subtype"]
        toks_a = len(tok.encode(sa, add_special_tokens=True))
        toks_b = len(tok.encode(sb, add_special_tokens=True))
        cl = min(max(toks_a, toks_b), MAX_LEN)
        oa=capture_all(model,tok,sa,nl,cl); ob=capture_all(model,tok,sb,nl,cl)
        if oa and ob: pd[pn]={"A":oa,"B":ob,"seq_len":cl,"subtype":st}
        ia=tok(sa,return_tensors="pt",truncation=True,max_length=cl).to(DEV)
        ib=tok(sb,return_tensors="pt",truncation=True,max_length=cl).to(DEV)
        with torch.no_grad():
            la=model(**ia).logits[0,-1,:].detach().cpu().float()
            lb=model(**ib).logits[0,-1,:].detach().cpu().float()
        kab=float(F.kl_div(F.log_softmax(la,-1),F.softmax(lb,-1),reduction='sum'))
        pm[pn]={"logits_a":la,"logits_b":lb,"kl_ab":kab,"sent_a":sa,"seq_len":cl,"subtype":st}
        if (pi+1)%20==0: log(f"  [{pi+1}/{len(pairs)}] {time.time()-t0:.0f}s")
    log(f"  Capture done: {len(pd)} pairs, {time.time()-t0:.0f}s")

    # ====== Block configs ======
    blocks = []
    bs = 0
    while bs < nl:
        be = min(bs + BLOCK_SIZE, nl)
        if be - bs >= 2:  # at least 2 layers in block
            blocks.append((bs, be-bs))
        bs += BLOCK_SIZE
    # Also add single layers at block midpoints for comparison
    single_layers = []
    for blk_start, blk_size in blocks:
        mid = blk_start + blk_size // 2
        single_layers.append(mid)

    log(f"\nBlocks: {len(blocks)} ({[f'{s}-{s+l-1}' for s,l in blocks]})")
    log(f"Single comparison layers: {single_layers}")

    # ====== PATCHING ======
    log(f"\nPhase 2: Block patching ({len(blocks)} blocks + {len(single_layers)} singles)")
    all_results = []; fail_count = 0; t0 = time.time()

    for pi,pn in enumerate(pd.keys()):
        m=pm[pn]; d=pd[pn]
        la,lb,kab,sa,cl,st=m["logits_a"],m["logits_b"],m["kl_ab"],m["sent_a"],m["seq_len"],m["subtype"]
        if kab<1e-6: continue

        # Block recomputation
        for blk_start, blk_size in blocks:
            for alpha in ALPHAS:
                logits_b, mlp_caps, down_cap = forward_block_recomp(
                    model,tok,sa,nl,d["A"],d["B"],blk_start,blk_size,alpha,cl)
                mr = compute_metrics(logits_b, la, lb, kab)
                if mr:
                    all_results.append({"pname":pn,"subtype":st,"block_start":blk_start,
                        "block_size":blk_size,"mode":"block","alpha":alpha,"kl_ab":kab,**mr})
                else: fail_count += 1

        # Single-layer comparison
        for tl in single_layers:
            logits_s, mlp_cap, down_cap = forward_single_recomp(
                model,tok,sa,nl,d["A"],d["B"],tl,1.0,cl)
            mr = compute_metrics(logits_s, la, lb, kab)
            if mr:
                all_results.append({"pname":pn,"subtype":st,"block_start":tl,
                    "block_size":1,"mode":"single","alpha":1.0,"kl_ab":kab,**mr})
            else: fail_count += 1

        if (pi+1)%10==0:
            e=time.time()-t0; rem=len(pd)-pi-1; eta=e/max(pi+1,1)*rem if pi>0 else 0
            log(f"  [{pi+1}/{len(pd)}] {e:.0f}s, {len(all_results)}r, {fail_count}f, "
                f"ETA={eta/60:.0f}m, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

    t_patch=time.time()-t0
    log(f"  Done: {len(all_results)} results, {fail_count} fails, {t_patch:.0f}s ({t_patch/60:.1f}min)")

    # ====== ANALYSIS ======
    log(f"\n{'='*60}"); log("ANALYSIS"); log(f"{'='*60}")

    # Block vs Single comparison (α=1.0)
    log(f"\n  === BLOCK vs SINGLE (α=1.0) ===")
    block_by_start = defaultdict(list)
    single_by_layer = defaultdict(list)
    for r in all_results:
        if r["alpha"]==1.0:
            if r["mode"]=="block" and r["block_size"]>=2:
                block_by_start[r["block_start"]].append(r)
            elif r["mode"]=="single":
                single_by_layer[r["block_start"]].append(r)

    log(f"  {'Block':>10} {'Blk_Prog':>9} {'Blk_KR':>9} {'Sgl_Prog':>9} {'Sgl_KR':>9} {'Ratio':>7}")
    blk_results = {}
    for blk_start in sorted(block_by_start.keys()):
        br = block_by_start[blk_start]
        blk_p = float(np.mean([x["progress"] for x in br]))
        blk_k = float(np.mean([x["kl_ratio"] for x in br]))
        
        mid = blk_start + br[0]["block_size"]//2 if br else blk_start+2
        sr = single_by_layer.get(mid, [])
        sgl_p = float(np.mean([x["progress"] for x in sr])) if sr else 0
        sgl_k = float(np.mean([x["kl_ratio"] for x in sr])) if sr else 0
        
        ratio = f"{blk_p/sgl_p:.1f}x" if sgl_p>0.01 else "N/A"
        blk_results[str(blk_start)] = {"block_prog":blk_p,"block_kr":blk_k,
                                        "single_prog":sgl_p,"single_kr":sgl_k,"ratio":ratio}
        log(f"  [{blk_start}-{blk_start+br[0]['block_size']-1}] {blk_p:9.4f} {blk_k:9.3f} {sgl_p:9.4f} {sgl_k:9.3f} {ratio:>7}")

    # Amplification ratio: block / single
    amp_ratios = []
    for blk_start, v in blk_results.items():
        if v["single_prog"] > 0.01:
            amp_ratios.append(v["block_prog"] / v["single_prog"])
    if amp_ratios:
        log(f"\n  Block/Single amplification: mean={np.mean(amp_ratios):.2f}x, max={np.max(amp_ratios):.2f}x")
        strong_blocks = sum(1 for a in amp_ratios if a > 1.5)
        log(f"  Blocks with >1.5x amplification: {strong_blocks}/{len(amp_ratios)}")

    # α interpolation for best block
    best_blk = max(blk_results.keys(), key=lambda b: blk_results[b]["block_prog"])
    log(f"\n  === α INTERPOLATION at block [{best_blk}-{int(best_blk)+3}] ===")
    alpha_agg = defaultdict(list)
    for r in all_results:
        if r["mode"]=="block" and str(r["block_start"])==str(best_blk):
            alpha_agg[r["alpha"]].append(r)
    log(f"  {'α':>6} {'Prog':>9} {'KR':>9}")
    for alpha in sorted(alpha_agg.keys()):
        dr = alpha_agg[alpha]; p=np.mean([x["progress"] for x in dr]); k=np.mean([x["kl_ratio"] for x in dr])
        log(f"  {alpha:>6.2f} {p:9.4f} {k:9.3f}")

    # Subtype breakdown at best block
    log(f"\n  === SUBTYPE at best block [{best_blk}-{int(best_blk)+3}] α=1.0 ===")
    st_agg = defaultdict(list)
    for r in all_results:
        if r["mode"]=="block" and str(r["block_start"])==str(best_blk) and r["alpha"]==1.0:
            st_agg[r["subtype"]].append(r)
    for st in sorted(st_agg.keys()):
        dr=st_agg[st]; p=np.mean([x["progress"] for x in dr]); k=np.mean([x["kl_ratio"] for x in dr])
        log(f"    {st:>25}: prog={p:.4f}, KR={k:.3f}, n={len(dr)}")

    # Layer-to-block progress correlation
    log(f"\n  === LAYER vs BLOCK correlation ===")
    for blk_start in sorted(block_by_start.keys()):
        br = block_by_start[blk_start]
        mid = blk_start + br[0]["block_size"]//2
        blk_p = blk_results[str(blk_start)]["block_prog"]
        sgl_p = blk_results[str(blk_start)]["single_prog"]
        amp = blk_p/sgl_p if sgl_p>0.01 else 0
        if amp > 1.3:
            log(f"    Block [{blk_start}-{blk_start+3}] amplification={amp:.2f}x: distributed contract")

    # ====== SAVE ======
    save = {"model":model_name,"info":{"class":info.model_class,"L":nl,"d":info.d_model},
            "n_pairs":len(pairs),"n_results":len(all_results),"n_fails":fail_count,
            "block_size":BLOCK_SIZE,"n_blocks":len(blocks),"alphas":ALPHAS,
            "block_results":blk_results,"best_block":best_blk}
    sp = RESULT_DIR/f"{model_name}_block.json"
    with open(sp,"w") as f: json.dump(save,f,indent=2,default=str)
    log(f"\nSaved to {sp}")
    release_model(model); gc.collect(); torch.cuda.empty_cache()
    return save

if __name__=="__main__":
    mn = sys.argv[1] if len(sys.argv)>1 else "qwen3"
    if mn=="all":
        for n in ["qwen3","glm4","deepseek7b"]:
            try:
                r=run_phase291(n); log(f"\n{n} DONE: {r['n_results']}")
            except Exception as e:
                log(f"!!! {n} FAILED: {e}"); import traceback; traceback.print_exc()
            gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
    else: run_phase291(mn)
