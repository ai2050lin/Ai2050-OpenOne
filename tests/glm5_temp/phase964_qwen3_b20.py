#!/usr/bin/env python3
"""qwen3 b=20 quick test."""
import sys; sys.path.insert(0, "tests/glm5")
from model_utils import load_model, get_model_info, release_model
from phase964_forward_diff import make_eos_inject_hook, evaluate_strict_clean, EN_PROMPTS_50, log
import torch, numpy as np
log("qwen3 b=20 quick test")
model, tok, dev = load_model("qwen3")
eos_id = tok.eos_token_id
prompts = EN_PROMPTS_50[:5]
for pi, prompt in enumerate(prompts):
    ids = tok.encode(prompt, return_tensors="pt").to(dev)
    with torch.no_grad(): bl = model(ids, use_cache=False).logits[0,-1].detach().float().cpu().numpy()
    top1 = float(np.sort(bl)[-1]); eos = float(bl[eos_id])
    log(f"  p{pi}: EOS={eos:.3f} top1={top1:.3f} gap={top1-eos:.3f}")

for cn, bias, delay in [("normal",0,0),("b=20",20,0),("delayed2_b=20",20,2)]:
    h = None
    if bias > 0: h = model.lm_head.register_forward_hook(make_eos_inject_hook(eos_id, bias, delay))
    clean_count = 0
    for pi, prompt in enumerate(prompts):
        ids = tok.encode(prompt, return_tensors="pt").to(dev)
        with torch.no_grad(): oid = model.generate(ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
        gt = oid[0][ids.shape[1]:]; gen = tok.decode(gt, skip_special_tokens=False)
        he = gt[-1].item() == eos_id; ng = len(gt)
        ce = evaluate_strict_clean(prompt, gen, he, ng)
        clean_count += int(ce["strict_clean"])
        if pi == 0:
            sc = ce["strict_clean"]
            log(f"  {cn} p0: eos={he} clean={sc} toks={ng} text={gen[:60]}")
    log(f"  {cn}: clean={clean_count}/{len(prompts)}")
    if h: h.remove()
release_model(model)
