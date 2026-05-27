"""
Phase 289 DS7B: step=2 layers, α=[0,0.5,1.0], negation 40 pairs
Same architecture as GLM4 fix script.
"""
import sys, os, gc, time, json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../glm5'))

import torch, torch.nn.functional as F
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase289_layer_scan")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
_log_file = str(Path("tmp") / "phase289_ds7b_fix.txt")

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(_log_file, "a", encoding="utf-8") as f: f.write(line+"\n")

def build_pairs():
    pairs = []
    for name, pos, neg in [
        ("happy","she is happy","she is not happy"),
        ("open","the door is open","the door is not open"),
        ("possible","victory is possible","victory is not possible"),
        ("important","this is important","this is not important"),
        ("clear","the answer is clear","the answer is not clear"),
        ("safe","the area is safe","the area is not safe"),
        ("fair","the decision is fair","the decision is not fair"),
        ("correct","your answer is correct","your answer is not correct"),
        ("stable","the system is stable","the system is not stable"),
        ("agree","they agree with the proposal","they do not agree with the proposal"),
        ("remember","i remember the meeting","i do not remember the meeting"),
        ("know","she knows the answer","she does not know the answer"),
        ("believe","he believes the story","he does not believe the story"),
        ("support","they support the plan","they do not support the plan"),
        ("accept","she accepts the offer","she does not accept the offer"),
        ("trust","he trusts the source","he does not trust the source"),
        ("follow","they follow the rules","they do not follow the rules"),
        ("require","the task requires effort","the task does not require effort"),
        ("nothing","he found something interesting","he found nothing interesting"),
        ("no_one","someone came to the party","no one came to the party"),
        ("no_food","there was some food left","there was no food left"),
        ("no_idea","she had some idea what to do","she had no idea what to do"),
        ("no_reason","there is a reason to worry","there is no reason to worry"),
        ("no_doubt","there is some doubt about it","there is no doubt about it"),
        ("never_seen","i have seen it before","i have never seen it before"),
        ("never_been","she has been to Paris","she has never been to Paris"),
        ("never_gives","she sometimes gives up","she never gives up"),
        ("impossible","the task is possible","the task is impossible"),
        ("unacceptable","the proposal is acceptable","the proposal is unacceptable"),
        ("incomplete","the report is complete","the report is incomplete"),
        ("irrelevant","the comment is relevant","the comment is irrelevant"),
        ("dishonest","the person is honest","the person is dishonest"),
        ("not_all","all birds can fly","not all birds can fly"),
        ("not_everyone","everyone agreed","not everyone agreed"),
        ("not_always","she always tells the truth","she does not always tell the truth"),
        ("not_necessarily","wealth means happiness","wealth does not necessarily mean happiness"),
        ("not_because","he left because he was angry","he did not leave because he was angry"),
        ("not_if","she will come if invited","she will not come if invited"),
        ("not_a_single","a single person helped","not a single person helped"),
        ("not_even_one","he ate one cookie","he did not eat even one cookie"),
    ]:
        pairs.append({"name":f"neg_{name}","A":pos,"B":neg,"category":"negation"})
    return pairs

def capture_all(model, tokenizer, sent, device, n_layers, max_len=48):
    layers = get_layers(model)
    al = min(n_layers, len(layers))
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    sl = inputs["input_ids"].shape[1]
    if sl < max_len:
        inputs["input_ids"] = F.pad(inputs["input_ids"],(0,max_len-sl),value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"],(0,max_len-sl),value=0)
    cap = {}; hooks = []
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
        except Exception as e: log(f"  capture FAILED: {e}"); cap={}
    for h in hooks: h.remove()
    return cap

def fwd_patch(model, tokenizer, sa, device, ml, n_layers, a_d, b_d, tl, alpha, pt):
    layers = get_layers(model)
    if tl >= len(layers): return None
    inputs = tokenizer(sa, return_tensors="pt", truncation=True, max_length=ml).to(device)
    sl = inputs["input_ids"].shape[1]
    if sl < ml:
        inputs["input_ids"]=F.pad(inputs["input_ids"],(0,ml-sl),value=tokenizer.pad_token_id)
        inputs["attention_mask"]=F.pad(inputs["attention_mask"],(0,ml-sl),value=0)
    aa,ab,ma,mb = a_d.get("attn"),b_d.get("attn"),a_d.get("mlp"),b_d.get("mlp")
    if any(x is None for x in [aa,ab,ma,mb]): return None
    
    da = layers[tl].self_attn.o_proj.weight.device
    dta = layers[tl].self_attn.o_proj.weight.dtype
    dm = next(layers[tl].mlp.parameters()).device
    dtm = next(layers[tl].mlp.parameters()).dtype
    
    def interp(a,b,alpha,dev,dtype):
        at,bt = a.to(dev).to(dtype), b.to(dev).to(dtype)
        ms = min(at.shape[1],bt.shape[1],ml)
        return (1-alpha)*at[:,:ms,:]+alpha*bt[:,:ms,:]
    
    hooks=[]
    if pt in ("attn","both"):
        pv=interp(aa,ab,alpha,da,dta)
        def mah(pv):
            def hook(m,i,o):
                no=(o[0].clone(),)+o[1:] if isinstance(o,tuple) else o.clone()
                ref=no[0] if isinstance(o,tuple) else no
                ms=min(pv.shape[1],ref.shape[1]); ref[:,:ms,:]=pv[:,:ms,:]
                return no if isinstance(o,tuple) else no
            return hook
        hooks.append(layers[tl].self_attn.register_forward_hook(mah(pv)))
    if pt in ("mlp","both"):
        pv=interp(ma,mb,alpha,dm,dtm)
        def mmh(pv):
            def hook(m,i,o):
                no=(o[0].clone(),)+o[1:] if isinstance(o,tuple) else o.clone()
                ref=no[0] if isinstance(o,tuple) else no
                ms=min(pv.shape[1],ref.shape[1]); ref[:,:ms,:]=pv[:,:ms,:]
                return no if isinstance(o,tuple) else no
            return hook
        hooks.append(layers[tl].mlp.register_forward_hook(mmh(pv)))
    
    try:
        with torch.no_grad(): out=model(**inputs); res=out.logits[0,-1,:].detach().cpu().float().clone()
    except Exception as e:
        log(f"    FWD ERR L{tl} α={alpha} {pt}: {e}"); res=None
    for h in hooks: h.remove()
    return res

def met(patched,la,lb,kab):
    if patched is None: return None
    kp=float(F.kl_div(F.log_softmax(patched,-1),F.softmax(lb,-1),reduction='sum'))
    kr=min(kp/max(kab,1e-6),50.0)
    db,dp=lb-la,patched-la
    nb,np=float(torch.norm(db)),float(torch.norm(dp))
    cd=float(torch.dot(dp,db)/(nb*np)) if nb>1e-8 and np>1e-8 else 0
    pr=cd*min(np/nb,2.0) if nb>1e-8 else 0
    nr=np/nb if nb>1e-8 else 0
    return {"kl_ratio":kr,"progress":pr,"cos_dir":cd,"norm_ratio":nr}

def main():
    mn="deepseek7b"
    cfg=MODEL_CONFIGS[mn]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"Phase 289 DS7B — Negation 40 pairs, step=2, α=[0,0.5,1.0]")
    tok=AutoTokenizer.from_pretrained(cfg["path"],trust_remote_code=True,local_files_only=True,use_fast=False)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(cfg["path"],torch_dtype=torch.bfloat16,device_map="auto",
        trust_remote_code=True,local_files_only=True,attn_implementation="eager")
    model.eval()
    dev=next(model.parameters()).device
    info=get_model_info(model,mn); nl=info.n_layers
    log(f"Loaded: L={nl}, d={info.d_model}, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    with torch.no_grad(): model(**tok("warmup",return_tensors="pt").to(dev))
    pairs=build_pairs(); log(f"Pairs: {len(pairs)}")
    ml=48; ALPHAS=[0.0,0.5,1.0]; PTS=["attn","mlp","both"]; tls=list(range(0,nl,2))
    
    # Capture
    log("Phase 1: Capture..."); t0=time.time(); pd={}; pm={}
    for pi,pr in enumerate(pairs):
        pn=pr["name"]; sa,sb=pr["A"],pr["B"]
        cl=min(max(len(tok.encode(sa)),len(tok.encode(sb))),ml)
        oa=capture_all(model,tok,sa,dev,nl,cl); ob=capture_all(model,tok,sb,dev,nl,cl)
        if oa and ob: pd[pn]={"A":oa,"B":ob,"seq_len":cl}
        ia=tok(sa,return_tensors="pt",truncation=True,max_length=cl).to(dev)
        ib=tok(sb,return_tensors="pt",truncation=True,max_length=cl).to(dev)
        with torch.no_grad():
            la=model(**ia).logits[0,-1,:].detach().cpu().float()
            lb=model(**ib).logits[0,-1,:].detach().cpu().float()
        kab=float(F.kl_div(F.log_softmax(la,-1),F.softmax(lb,-1),reduction='sum'))
        pm[pn]={"logits_a":la,"logits_b":lb,"kl_ab":kab,"sent_a":sa,"seq_len":cl}
        if (pi+1)%20==0: log(f"  [{pi+1}/{len(pairs)}] {time.time()-t0:.0f}s")
    log(f"  Capture done: {len(pd)} pairs, {time.time()-t0:.0f}s")
    
    # Patch
    log(f"Phase 2: Patching {len(tls)} layers × {len(ALPHAS)}α × {len(PTS)} types")
    res=[]; fc=0; t0=time.time()
    for pi,pn in enumerate(pd.keys()):
        m=pm[pn]; d=pd[pn]; la,lb,kab,sa,cl=m["logits_a"],m["logits_b"],m["kl_ab"],m["sent_a"],m["seq_len"]
        if kab<1e-6: continue
        for tl in tls:
            lk=f"L{tl}"
            if lk not in d["A"] or lk not in d["B"]: continue
            for alpha in ALPHAS:
                for pt in PTS:
                    pp=fwd_patch(model,tok,sa,dev,cl,nl,d["A"][lk],d["B"][lk],tl,alpha,pt)
                    if pp is not None:
                        m2=met(pp,la,lb,kab)
                        if m2: res.append({"pname":pn,"layer":tl,"alpha":alpha,"patch_type":pt,**m2})
                    else: fc+=1
        if (pi+1)%10==0:
            e=time.time()-t0
            log(f"  [{pi+1}/{len(pd)}] {e:.0f}s, {len(res)} results, {fc} fails, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    log(f"  Done: {len(res)} results, {fc} fails, {time.time()-t0:.0f}s")
    
    # Analysis
    log("\nAnalysis:")
    lay=defaultdict(lambda:defaultdict(list))
    for r in res:
        if r["alpha"]==1.0: lay[r["layer"]][r["patch_type"]].append(r)
    log(f"  {'Layer':>6} {'Attn_KR':>9} {'Attn_Prog':>9} {'MLP_KR':>9} {'MLP_Prog':>9} {'Both_KR':>9} {'Both_Prog':>9}")
    lc={}
    for li in sorted(lay.keys()):
        d=lay[li]; vs={}
        for pt in ["attn","mlp","both"]:
            vs[f"{pt}_kr"]=float(np.mean([x["kl_ratio"] for x in d[pt]])) if d[pt] else 0
            vs[f"{pt}_prog"]=float(np.mean([x["progress"] for x in d[pt]])) if d[pt] else 0
        lc[str(li)]=vs
        log(f"  {li:>6} {vs['attn_kr']:9.3f} {vs['attn_prog']:9.4f} {vs['mlp_kr']:9.3f} {vs['mlp_prog']:9.4f} {vs['both_kr']:9.3f} {vs['both_prog']:9.4f}")
    bl=max(lc.keys(),key=lambda l:lc[l].get("both_prog",0))
    log(f"\n  α INTERP at L{bl}:")
    ala=defaultdict(lambda:defaultdict(list))
    for r in res:
        if r["layer"]==int(bl): ala[r["alpha"]][r["patch_type"]].append(r)
    log(f"  {'α':>6} {'Attn_KR':>9} {'Attn_Prog':>9} {'MLP_KR':>9} {'MLP_Prog':>9} {'Both_KR':>9} {'Both_Prog':>9}")
    for alpha in sorted(ala.keys()):
        d=ala[alpha]; vs={}
        for pt in ["attn","mlp","both"]:
            vs[f"{pt}_kr"]=float(np.mean([x["kl_ratio"] for x in d[pt]])) if d[pt] else 0
            vs[f"{pt}_prog"]=float(np.mean([x["progress"] for x in d[pt]])) if d[pt] else 0
        log(f"  {alpha:>6.2f} {vs['attn_kr']:9.3f} {vs['attn_prog']:9.4f} {vs['mlp_kr']:9.3f} {vs['mlp_prog']:9.4f} {vs['both_kr']:9.3f} {vs['both_prog']:9.4f}")
    sv={"model":mn,"n_pairs":len(pairs),"n_results":len(res),"layer_curve":lc,"best_layer":bl}
    sp=RESULT_DIR/f"deepseek7b_layer_scan.json"
    with open(sp,"w") as f: json.dump(sv,f,indent=2,default=str)
    log(f"\nSaved to {sp}")
    release_model(model); gc.collect(); torch.cuda.empty_cache()
    log("DONE")
if __name__=="__main__": main()
