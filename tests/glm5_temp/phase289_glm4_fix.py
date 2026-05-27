"""
Phase 289 GLM4 Fixed: Reduced scope + error diagnostics
========================================================
Only negation (80 pairs), α=[0, 0.5, 1.0], 3 patch types (attn, mlp, both)
Samples every 2nd layer to reduce total forwards.
"""
import sys, os, gc, time, json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../glm5'))

import torch, torch.nn.functional as F
from torch import nn
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase289_layer_scan")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

_log_file = str(Path("tmp") / "phase289_glm4_fix.txt")

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(_log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# Dataset
def build_negation_pairs():
    pairs = []
    for name, pos, neg in [
        ("happy", "she is happy", "she is not happy"),
        ("open", "the door is open", "the door is not open"),
        ("possible", "victory is possible", "victory is not possible"),
        ("ready", "they are ready", "they are not ready"),
        ("important", "this is important", "this is not important"),
        ("clear", "the answer is clear", "the answer is not clear"),
        ("safe", "the area is safe", "the area is not safe"),
        ("fair", "the decision is fair", "the decision is not fair"),
        ("correct", "your answer is correct", "your answer is not correct"),
        ("stable", "the system is stable", "the system is not stable"),
        ("agree", "they agree with the proposal", "they do not agree with the proposal"),
        ("remember", "i remember the meeting", "i do not remember the meeting"),
        ("know", "she knows the answer", "she does not know the answer"),
        ("believe", "he believes the story", "he does not believe the story"),
        ("support", "they support the plan", "they do not support the plan"),
        ("accept", "she accepts the offer", "she does not accept the offer"),
        ("trust", "he trusts the source", "he does not trust the source"),
        ("follow", "they follow the rules", "they do not follow the rules"),
        ("require", "the task requires effort", "the task does not require effort"),
        ("confirm", "the test confirms the theory", "the test does not confirm the theory"),
        ("nothing", "he found something interesting", "he found nothing interesting"),
        ("no_one", "someone came to the party", "no one came to the party"),
        ("no_food", "there was some food left", "there was no food left"),
        ("no_idea", "she had some idea what to do", "she had no idea what to do"),
        ("no_reason", "there is a reason to worry", "there is no reason to worry"),
        ("no_doubt", "there is some doubt about it", "there is no doubt about it"),
        ("no_evidence", "there is evidence of fraud", "there is no evidence of fraud"),
        ("never_seen", "i have seen it before", "i have never seen it before"),
        ("never_been", "she has been to Paris", "she has never been to Paris"),
        ("never_gives", "she sometimes gives up", "she never gives up"),
        ("impossible", "the task is possible", "the task is impossible"),
        ("unacceptable", "the proposal is acceptable", "the proposal is unacceptable"),
        ("incomplete", "the report is complete", "the report is incomplete"),
        ("irrelevant", "the comment is relevant", "the comment is irrelevant"),
        ("dishonest", "the person is honest", "the person is dishonest"),
        ("not_all", "all birds can fly", "not all birds can fly"),
        ("not_everyone", "everyone agreed", "not everyone agreed"),
        ("not_always", "she always tells the truth", "she does not always tell the truth"),
        ("not_necessarily", "wealth means happiness", "wealth does not necessarily mean happiness"),
        ("not_because", "he left because he was angry", "he did not leave because he was angry"),
    ]:
        pairs.append({"name": f"neg_{name}", "A": pos, "B": neg, "category": "negation"})
    return pairs

def capture_all_layers(model, tokenizer, sentence, device, n_layers, max_len=48):
    layers = get_layers(model)
    actual_layers = min(n_layers, len(layers))
    
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    seq_len = inputs["input_ids"].shape[1]
    if seq_len < max_len:
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, max_len-seq_len), value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, max_len-seq_len), value=0)

    captured = {}
    hooks = []
    for li in range(actual_layers):
        def make_h(li, key, mod_attr):
            def hook(module, input_t, output_t):
                val = output_t[0] if isinstance(output_t, tuple) else output_t
                captured.setdefault(f"L{li}", {})[key] = val.detach().cpu().clone()
            return hook
        hooks.append(layers[li].self_attn.register_forward_hook(make_h(li, "attn", "self_attn")))
        hooks.append(layers[li].mlp.register_forward_hook(make_h(li, "mlp", "mlp")))
    
    with torch.no_grad():
        try:
            model(**inputs)
        except Exception as e:
            log(f"  capture FAILED: {e}")
            captured = {}
    for h in hooks:
        h.remove()
    return captured


def forward_with_patch(model, tokenizer, sent_a, device, max_len, n_layers,
                       a_data, b_data, target_layer, alpha, patch_type):
    """Forward A with α-interpolated patch at single layer."""
    layers = get_layers(model)
    if target_layer >= len(layers):
        return None
    
    inputs = tokenizer(sent_a, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    seq_len = inputs["input_ids"].shape[1]
    if seq_len < max_len:
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, max_len-seq_len), value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, max_len-seq_len), value=0)

    attn_a = a_data.get("attn")
    attn_b = b_data.get("attn")
    mlp_a = a_data.get("mlp")
    mlp_b = b_data.get("mlp")
    
    if attn_a is None or attn_b is None or mlp_a is None or mlp_b is None:
        return None

    o_proj = layers[target_layer].self_attn.o_proj
    dev_attn = o_proj.weight.device
    dtype_attn = o_proj.weight.dtype
    
    mlp_mod = layers[target_layer].mlp
    dev_mlp = next(mlp_mod.parameters()).device
    dtype_mlp = next(mlp_mod.parameters()).dtype

    def interp(a, b, alpha, dev, dtype):
        a_t = a.to(dev).to(dtype)
        b_t = b.to(dev).to(dtype)
        ms = min(a_t.shape[1], b_t.shape[1], max_len)
        return (1-alpha)*a_t[:,:ms,:] + alpha*b_t[:,:ms,:]

    hooks = []
    if patch_type in ("attn", "both"):
        pv = interp(attn_a, attn_b, alpha, dev_attn, dtype_attn)
        def make_ah(pv):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    no = (out[0].clone(),) + out[1:]
                    ms = min(pv.shape[1], no[0].shape[1])
                    no[0][:,:ms,:] = pv[:,:ms,:]
                    return no
                else:
                    no = out.clone()
                    ms = min(pv.shape[1], no.shape[1])
                    no[:,:ms,:] = pv[:,:ms,:]
                    return no
            return hook
        hooks.append(layers[target_layer].self_attn.register_forward_hook(make_ah(pv)))
    
    if patch_type in ("mlp", "both"):
        pv = interp(mlp_a, mlp_b, alpha, dev_mlp, dtype_mlp)
        def make_mh(pv):
            def hook(m, inp, out):
                if isinstance(out, tuple):
                    no = (out[0].clone(),) + out[1:]
                    ms = min(pv.shape[1], no[0].shape[1])
                    no[0][:,:ms,:] = pv[:,:ms,:]
                    return no
                else:
                    no = out.clone()
                    ms = min(pv.shape[1], no.shape[1])
                    no[:,:ms,:] = pv[:,:ms,:]
                    return no
            return hook
        hooks.append(layers[target_layer].mlp.register_forward_hook(make_mh(pv)))

    try:
        with torch.no_grad():
            out = model(**inputs)
            result = out.logits[0, -1, :].detach().cpu().float().clone()
    except Exception as e:
        log(f"    FORWARD ERROR at L{target_layer} α={alpha} {patch_type}: {e}")
        result = None

    for h in hooks:
        h.remove()
    return result


def compute_kr(patched, logits_a, logits_b, kl_ab):
    if patched is None: return None
    kp = float(F.kl_div(F.log_softmax(patched, dim=-1), F.softmax(logits_b, dim=-1), reduction='sum'))
    kr = min(kp/max(kl_ab, 1e-6), 50.0)
    db = logits_b - logits_a
    dp = patched - logits_a
    nb = float(torch.norm(db)); np_ = float(torch.norm(dp))
    cos_d = float(torch.dot(dp, db)/(nb*np_)) if nb>1e-8 and np_>1e-8 else 0
    prog = cos_d * min(np_/nb, 2.0) if nb>1e-8 else 0
    nr = np_/nb if nb>1e-8 else 0
    return {"kl_ratio": kr, "progress": prog, "cos_dir": cos_d, "norm_ratio": nr}


def main():
    model_name = "glm4"
    cfg = MODEL_CONFIGS[model_name]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"Phase 289 GLM4 Fixed — Negation Only, step=2 layers, α=[0,0.5,1.0]")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager",
    )
    model.eval()
    device = next(model.parameters()).device
    n = model_info = get_model_info(model, model_name)
    n_layers = n.n_layers
    log(f"Loaded: L={n_layers}, d={n.d_model}, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Warmup
    with torch.no_grad():
        model(**tokenizer("warmup", return_tensors="pt").to(device))

    pairs = build_negation_pairs()
    log(f"Negation pairs: {len(pairs)}")

    max_len = 48
    ALPHAS = [0.0, 0.5, 1.0]
    PATCH_TYPES = ["attn", "mlp", "both"]
    test_layers = list(range(0, n_layers, 2))  # step=2
    
    # === CAPTURE ===
    log("Phase 1: Capture...")
    t0 = time.time()
    pair_data = {}
    pair_meta = {}
    
    for pidx, pair in enumerate(pairs):
        pname = pair["name"]
        sent_a, sent_b = pair["A"], pair["B"]
        cl = min(max(len(tokenizer.encode(sent_a)), len(tokenizer.encode(sent_b))), max_len)
        
        out_a = capture_all_layers(model, tokenizer, sent_a, device, n_layers, cl)
        out_b = capture_all_layers(model, tokenizer, sent_b, device, n_layers, cl)
        if out_a and out_b:
            pair_data[pname] = {"A": out_a, "B": out_b, "seq_len": cl}
        
        # Baseline
        ia = tokenizer(sent_a, return_tensors="pt", truncation=True, max_length=cl).to(device)
        ib = tokenizer(sent_b, return_tensors="pt", truncation=True, max_length=cl).to(device)
        with torch.no_grad():
            la = model(**ia).logits[0,-1,:].detach().cpu().float()
            lb = model(**ib).logits[0,-1,:].detach().cpu().float()
        kab = float(F.kl_div(F.log_softmax(la,-1), F.softmax(lb,-1), reduction='sum'))
        pair_meta[pname] = {"logits_a": la, "logits_b": lb, "kl_ab": kab, "sent_a": sent_a, "seq_len": cl}
        
        if (pidx+1)%20==0:
            log(f"  [{pidx+1}/{len(pairs)}] {time.time()-t0:.0f}s")

    log(f"  Capture done: {len(pair_data)} pairs, {time.time()-t0:.0f}s")

    # === PATCHING ===
    log(f"Phase 2: Patching {len(test_layers)} layers × {len(ALPHAS)}α × {len(PATCH_TYPES)} types")
    all_results = []
    fail_count = 0
    t0 = time.time()

    for pidx, pname in enumerate(pair_data.keys()):
        pm = pair_meta[pname]
        pd = pair_data[pname]
        la, lb, kab, sa, cl = pm["logits_a"], pm["logits_b"], pm["kl_ab"], pm["sent_a"], pm["seq_len"]
        
        if kab < 1e-6: continue

        for tl in test_layers:
            lk = f"L{tl}"
            if lk not in pd["A"] or lk not in pd["B"]: continue
            for alpha in ALPHAS:
                for pt in PATCH_TYPES:
                    patched = forward_with_patch(model, tokenizer, sa, device, cl, n_layers,
                                                  pd["A"][lk], pd["B"][lk], tl, alpha, pt)
                    if patched is not None:
                        m = compute_kr(patched, la, lb, kab)
                        if m:
                            all_results.append({"pname": pname, "layer": tl, "alpha": alpha,
                                               "patch_type": pt, **m})
                    else:
                        fail_count += 1

        if (pidx+1)%10==0:
            e = time.time()-t0
            log(f"  [{pidx+1}/{len(pair_data)}] {e:.0f}s, {len(all_results)} results, {fail_count} fails, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

    log(f"  Patching done: {len(all_results)} results, {fail_count} fails, {time.time()-t0:.0f}s")

    # === ANALYSIS ===
    log(f"\nAnalysis:")
    layer_agg = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if r["alpha"] == 1.0:
            layer_agg[r["layer"]][r["patch_type"]].append(r)

    log(f"  {'Layer':>6} {'Attn_KR':>9} {'Attn_Prog':>9} {'MLP_KR':>9} {'MLP_Prog':>9} {'Both_KR':>9} {'Both_Prog':>9}")
    layer_curve = {}
    for li in sorted(layer_agg.keys()):
        d = layer_agg[li]
        vals = {}
        for pt in ["attn", "mlp", "both"]:
            krs = np.mean([x["kl_ratio"] for x in d[pt]]) if d[pt] else 0
            prs = np.mean([x["progress"] for x in d[pt]]) if d[pt] else 0
            vals[f"{pt}_kr"] = float(krs); vals[f"{pt}_prog"] = float(prs)
        layer_curve[str(li)] = vals
        log(f"  {li:>6} {vals['attn_kr']:9.3f} {vals['attn_prog']:9.4f} {vals['mlp_kr']:9.3f} {vals['mlp_prog']:9.4f} {vals['both_kr']:9.3f} {vals['both_prog']:9.4f}")

    # α curve at strongest layer
    best_li = max(layer_curve.keys(), key=lambda l: layer_curve[l].get("both_prog", 0))
    log(f"\n  α INTERPOLATION at L{best_li}:")
    alpha_agg = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if r["layer"] == int(best_li):
            alpha_agg[r["alpha"]][r["patch_type"]].append(r)
    
    log(f"  {'α':>6} {'Attn_KR':>9} {'Attn_Prog':>9} {'MLP_KR':>9} {'MLP_Prog':>9} {'Both_KR':>9} {'Both_Prog':>9}")
    for alpha in sorted(alpha_agg.keys()):
        d = alpha_agg[alpha]
        vs = {}
        for pt in ["attn", "mlp", "both"]:
            vs[f"{pt}_kr"] = float(np.mean([x["kl_ratio"] for x in d[pt]])) if d[pt] else 0
            vs[f"{pt}_prog"] = float(np.mean([x["progress"] for x in d[pt]])) if d[pt] else 0
        log(f"  {alpha:>6.2f} {vs['attn_kr']:9.3f} {vs['attn_prog']:9.4f} {vs['mlp_kr']:9.3f} {vs['mlp_prog']:9.4f} {vs['both_kr']:9.3f} {vs['both_prog']:9.4f}")

    # Save
    save = {"model": model_name, "n_pairs": len(pairs), "n_results": len(all_results),
            "layer_curve": layer_curve, "best_layer": best_li}
    sp = RESULT_DIR / "glm4_layer_scan.json"
    with open(sp, "w") as f: json.dump(save, f, indent=2, default=str)
    log(f"\nSaved to {sp}")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    log("DONE")

if __name__ == "__main__":
    main()
