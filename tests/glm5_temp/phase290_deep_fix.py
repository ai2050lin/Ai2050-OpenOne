"""
Phase 290 Deep Fix: Handle CPU-offloaded layers
Key fix: CPU-side patch construction + output_t.device dynamic detection
"""
import sys, os, gc, time, json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../glm5'))
import torch, torch.nn.functional as F
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase290_recomputed")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
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

def build_pairs():
    pairs = []
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
    layers = get_layers(model)
    nl = len(layers)
    gpu_l, cpu_l = [], []
    for li in range(nl):
        wdev = layers[li].self_attn.o_proj.weight.device
        (gpu_l if wdev.type=='cuda' else cpu_l).append(li)
    log(f"  GPU: {len(gpu_l)} ({gpu_l[0]}-{gpu_l[-1] if gpu_l else 'none'}), CPU: {len(cpu_l)} ({cpu_l[0]}-{cpu_l[-1] if cpu_l else 'none'})")
    return model, tok

# Capture
def capture_all(model, tokenizer, sent, n_layers, max_len=48):
    layers = get_layers(model); al = min(n_layers,len(layers))
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

# FIXED recomputed forward
def forward_recomp_fixed(model, tokenizer, sent_a, n_layers, a_attn, b_attn, tl, alpha, max_len):
    layers = get_layers(model)
    if tl >= len(layers): return None, None, None
    
    inputs = tokenizer(sent_a,return_tensors="pt",truncation=True,max_length=max_len).to(DEV)
    sl = inputs["input_ids"].shape[1]
    if sl < max_len:
        inputs["input_ids"]=F.pad(inputs["input_ids"],(0,max_len-sl),value=tokenizer.pad_token_id)
        inputs["attention_mask"]=F.pad(inputs["attention_mask"],(0,max_len-sl),value=0)

    ms = min(a_attn.shape[1], b_attn.shape[1], max_len)
    pv_cpu = (1-alpha)*a_attn[:,:ms,:].float() + alpha*b_attn[:,:ms,:].float()  # CPU construction

    hooks = []; mlp_cap = {}; down_cap = {}

    def mah(pv_cpu):
        def hook(m,i,o):
            ref = o[0] if isinstance(o,tuple) else o
            pv = pv_cpu.to(ref.device).to(ref.dtype)  # KEY FIX: dynamic device
            mms = min(pv.shape[1], ref.shape[1])
            if isinstance(o,tuple): no=(o[0].clone(),)+o[1:]; no[0][:,:mms,:]=pv[:,:mms,:]; return no
            no=o.clone(); no[:,:mms,:]=pv[:,:mms,:]; return no
        return hook
    hooks.append(layers[tl].self_attn.register_forward_hook(mah(pv_cpu)))

    def mcap(li):
        def hook(m,i,o): mlp_cap["mlp_norm"]=float((o[0] if isinstance(o,tuple) else o).float().norm())
        return hook
    hooks.append(layers[tl].mlp.register_forward_hook(mcap(tl)))

    if tl+1 < len(layers):
        def dcap(li):
            def hook(m,i,o): down_cap["layer_out"]=float((o[0] if isinstance(o,tuple) else o).float().norm())
            return hook
        hooks.append(layers[tl+1].register_forward_hook(dcap(tl+1)))

    try:
        with torch.no_grad(): out=model(**inputs); logits=out.logits[0,-1,:].detach().cpu().float().clone()
    except Exception as e:
        log(f"    ERR L{tl} α={alpha}: {e}"); logits=None; mlp_cap={}; down_cap={}
    for h in hooks: h.remove()
    return logits, mlp_cap, down_cap

# Spliced both
def forward_spliced_fixed(model, tokenizer, sent_a, n_layers, a_attn, b_attn, a_mlp, b_mlp, tl, alpha, max_len):
    layers = get_layers(model)
    inputs = tokenizer(sent_a,return_tensors="pt",truncation=True,max_length=max_len).to(DEV)
    sl = inputs["input_ids"].shape[1]
    if sl < max_len:
        inputs["input_ids"]=F.pad(inputs["input_ids"],(0,max_len-sl),value=tokenizer.pad_token_id)
        inputs["attention_mask"]=F.pad(inputs["attention_mask"],(0,max_len-sl),value=0)
    ms = min(a_attn.shape[1], b_attn.shape[1], max_len)
    pv_a_cpu = (1-alpha)*a_attn[:,:ms,:].float() + alpha*b_attn[:,:ms,:].float()
    ms2 = min(a_mlp.shape[1], b_mlp.shape[1], max_len)
    pv_m_cpu = (1-alpha)*a_mlp[:,:ms2,:].float() + alpha*b_mlp[:,:ms2,:].float()
    hooks=[]
    def mah(pv_cpu):
        def hook(m,i,o):
            ref=o[0] if isinstance(o,tuple) else o; pv=pv_cpu.to(ref.device).to(ref.dtype)
            m_=min(pv.shape[1],ref.shape[1])
            no=(o[0].clone(),)+o[1:] if isinstance(o,tuple) else o.clone()
            (no[0] if isinstance(o,tuple) else no)[:,:m_,:]=pv[:,:m_,:]; return no
        return hook
    hooks.append(layers[tl].self_attn.register_forward_hook(mah(pv_a_cpu)))
    hooks.append(layers[tl].mlp.register_forward_hook(mah(pv_m_cpu)))
    try:
        with torch.no_grad(): out=model(**inputs); res=out.logits[0,-1,:].detach().cpu().float().clone()
    except: res=None
    for h in hooks: h.remove()
    return res

def compute_metrics(logits,la,lb,kab,mlp_n,mlp_na,mlp_nb,down_n,down_na):
    if logits is None: return None
    kp=float(F.kl_div(F.log_softmax(logits,-1),F.softmax(lb,-1),reduction='sum'))
    kr=min(kp/max(kab,1e-6),100.0)
    db,dp=lb-la,logits-la
    nb,np=float(torch.norm(db)),float(torch.norm(dp))
    cd=float(torch.dot(dp,db)/(nb*np)) if nb>1e-8 and np>1e-8 else 0
    prog=cd*min(np/nb,2.0) if nb>1e-8 else 0
    mlp_nr=mlp_n/max(mlp_na,1e-8) if mlp_n>0 and mlp_na>0 else 0
    mlp_tob=abs(mlp_n-mlp_nb)/max(abs(mlp_na-mlp_nb),1e-8) if mlp_n>0 and mlp_na>0 and mlp_nb>0 else 0
    down_amp=down_n/max(down_na,1e-8) if down_n>0 and down_na>0 else 0
    return {"kl_ratio":kr,"progress":prog,"cos_dir":cd,"mlp_norm_ratio":mlp_nr,"mlp_to_b_ratio":mlp_tob,"down_amp":down_amp}

def main():
    global _log_file
    mn = sys.argv[1] if len(sys.argv)>1 else "glm4"
    _log_file = str(Path("tmp")/f"phase290_deep_{mn}.txt")
    log(f"Phase 290 Deep Fix: {mn}")

    model, tok = load_model(mn)
    info = get_model_info(model, mn); nl = info.n_layers
    with torch.no_grad():
        try: model(**tok("warmup",return_tensors="pt").to(DEV))
        except: pass

    pairs = build_pairs(); log(f"Pairs: {len(pairs)}")
    ALPHAS = [0, 0.25, 0.5, 0.75, 1.0, 1.25]
    ML = 48; tls = list(range(0, nl, 2))
    layers = get_layers(model)

    # Determine deep start
    deep_start = nl
    for li in range(nl):
        if layers[li].self_attn.o_proj.weight.device.type != 'meta':
            deep_start = li + 1
    log(f"Deep layer start: L{deep_start} (GPU layers: 0-{deep_start-1}, CPU layers: {deep_start}-{nl-1})")

    # ===== CAPTURE =====
    log("Phase 1: Capture..."); t0=time.time(); pd,pm={},{}
    for pi,pr in enumerate(pairs):
        pn,sa,sb,st=pr["name"],pr["A"],pr["B"],pr["subtype"]
        cl=min(max(len(tok.encode(sa)),len(tok.encode(sb))),ML)
        oa=capture_all(model,tok,sa,nl,cl); ob=capture_all(model,tok,sb,nl,cl)
        if oa and ob: pd[pn]={"A":oa,"B":ob,"seq_len":cl,"subtype":st}
        ia=tok(sa,return_tensors="pt",truncation=True,max_length=cl).to(DEV)
        ib=tok(sb,return_tensors="pt",truncation=True,max_length=cl).to(DEV)
        with torch.no_grad():
            la=model(**ia).logits[0,-1,:].detach().cpu().float()
            lb=model(**ib).logits[0,-1,:].detach().cpu().float()
        kab=float(F.kl_div(F.log_softmax(la,-1),F.softmax(lb,-1),reduction='sum'))
        pm[pn]={"logits_a":la,"logits_b":lb,"kl_ab":kab,"sent_a":sa,"seq_len":cl,"subtype":st}
        if (pi+1)%15==0: log(f"  [{pi+1}] {time.time()-t0:.0f}s")
    log(f"  Done: {len(pd)} pairs, {time.time()-t0:.0f}s")

    # Natural MLP norms
    nat_mlp = {"A":defaultdict(list),"B":defaultdict(list)}
    for li in range(0,nl,2):
        lk = f"L{li}"
        for pn in pd:
            for sk,lab in [("A","A"),("B","B")]:
                if lk in pd[pn][lab] and pd[pn][lab][lk].get("mlp") is not None:
                    nat_mlp[lab][lk].append(float(pd[pn][lab][lk]["mlp"].float().norm()))

    # Downstream norms
    nat_down = {}
    for li in range(0,nl-1,2):
        down_norms_l = []
        for pi,pn in enumerate(list(pd.keys())[:5]):
            sa=pm[pn]["sent_a"]; cl=pm[pn]["seq_len"]
            inputs = tok(sa,return_tensors="pt",truncation=True,max_length=cl).to(DEV)
            cap={}
            def dh(l):
                def hook(m,i,o): cap["n"]=float((o[0] if isinstance(o,tuple) else o).float().norm())
                return hook
            h=layers[li+1].register_forward_hook(dh(li+1))
            with torch.no_grad(): model(**inputs)
            h.remove(); down_norms_l.append(cap.get("n",0))
        nat_down[f"L{li+1}"] = np.mean(down_norms_l) if down_norms_l else 1.0

    # ===== PATCHING =====
    log(f"\nPhase 2: Patching {len(tls)} layers × {len(ALPHAS)}α (DEEP FIX)")
    all_results = []; fail_count = 0; t0 = time.time()

    for pi,pn in enumerate(pd.keys()):
        m=pm[pn]; d=pd[pn]
        la,lb,kab,sa,cl,st=m["logits_a"],m["logits_b"],m["kl_ab"],m["sent_a"],m["seq_len"],m["subtype"]
        if kab<1e-6: continue
        for tl in tls:
            lk=f"L{tl}"; lkn=f"L{tl+1}"
            if lk not in d["A"] or lk not in d["B"]: continue
            aa,ab = d["A"][lk].get("attn"),d["B"][lk].get("attn")
            am,bm = d["A"][lk].get("mlp"),d["B"][lk].get("mlp")
            if any(x is None for x in [aa,ab,am,bm]): continue
            mlp_na = np.mean(nat_mlp["A"].get(lk,[1.0]))
            mlp_nb = np.mean(nat_mlp["B"].get(lk,[1.0]))
            down_na = nat_down.get(lkn,1.0)
            for alpha in ALPHAS:
                lr, mc, dc = forward_recomp_fixed(model,tok,sa,nl,aa,ab,tl,alpha,cl)
                mn2 = mc.get("mlp_norm",0) if mc else 0
                dn2 = dc.get("layer_out",0) if dc else 0
                mr = compute_metrics(lr,la,lb,kab,mn2,mlp_na,mlp_nb,dn2,down_na)
                if mr: all_results.append({"pname":pn,"subtype":st,"layer":tl,"alpha":alpha,
                    "mode":"recomp","kl_ab":kab,**mr,"remaining_layers":nl-tl-1})
                else: fail_count += 1
                if alpha in [0,0.5,1.0]:
                    ls = forward_spliced_fixed(model,tok,sa,nl,aa,ab,am,bm,tl,alpha,cl)
                    ms2 = compute_metrics(ls,la,lb,kab,0,0,0,0,0)
                    if ms2: all_results.append({"pname":pn,"subtype":st,"layer":tl,"alpha":alpha,
                        "mode":"spliced","kl_ab":kab,**ms2,"remaining_layers":nl-tl-1})
                    else: fail_count += 1
        if (pi+1)%10==0:
            e=time.time()-t0; rem=len(pd)-pi-1; eta=e/max(pi+1,1)*rem if pi>0 else 0
            log(f"  [{pi+1}/{len(pd)}] {e:.0f}s, {len(all_results)}r, {fail_count}f, ETA={eta/60:.0f}m")

    t_patch=time.time()-t0
    log(f"  Done: {len(all_results)} results, {fail_count} fails, {t_patch:.0f}s")

    # ===== ANALYSIS =====
    log(f"\n=== FULL LAYER CURVE (recomp α=1.0) ===")
    lay = defaultdict(list)
    for r in all_results:
        if r["alpha"]==1.0 and r["mode"]=="recomp": lay[r["layer"]].append(r)
    log(f"  {'Layer':>6} {'Dev':>5} {'Prog':>9} {'KR':>9} {'MLP_toB':>9} {'MLP_NR':>9} {'Down_Amp':>9}")
    for li in sorted(lay.keys()):
        dr=lay[li]; dv="GPU" if li<deep_start else "CPU"
        p=float(np.mean([x["progress"] for x in dr]))
        k=float(np.mean([x["kl_ratio"] for x in dr]))
        mtb=float(np.mean([x["mlp_to_b_ratio"] for x in dr if x["mlp_to_b_ratio"]>0]))
        mnr=float(np.mean([x["mlp_norm_ratio"] for x in dr if x["mlp_norm_ratio"]>0]))
        da=float(np.mean([x["down_amp"] for x in dr if x["down_amp"]>0]))
        log(f"  {li:>6} {dv:>5} {p:9.4f} {k:9.3f} {mtb:9.3f} {mnr:9.3f} {da:9.3f}")

    # GPU vs CPU layer stats
    gpu_res = [r for r in all_results if r["mode"]=="recomp" and r["alpha"]==1.0 and r["layer"]<deep_start]
    cpu_res = [r for r in all_results if r["mode"]=="recomp" and r["alpha"]==1.0 and r["layer"]>=deep_start]
    if gpu_res and cpu_res:
        log(f"\n  GPU layers: prog={np.mean([x['progress'] for x in gpu_res]):.4f}, KR={np.mean([x['kl_ratio'] for x in gpu_res]):.2f}")
        log(f"  CPU layers: prog={np.mean([x['progress'] for x in cpu_res]):.4f}, KR={np.mean([x['kl_ratio'] for x in cpu_res]):.2f}")

    # Save
    sv={"model":mn,"n_results":len(all_results),"n_fails":fail_count,"deep_start":deep_start,
        "n_gpu_layers":deep_start,"n_cpu_layers":nl-deep_start}
    sp=RESULT_DIR/f"{mn}_recomp_deep.json"
    with open(sp,"w") as f: json.dump(sv,f,indent=2,default=str)
    log(f"\nSaved to {sp}")
    release_model(model); gc.collect(); torch.cuda.empty_cache()
    log("DONE")

if __name__=="__main__": main()
