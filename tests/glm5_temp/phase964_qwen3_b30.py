#!/usr/bin/env python3
"""qwen3 b=30 delayed test."""
import sys; sys.path.insert(0, "tests/glm5")
from model_utils import load_model, release_model
from phase964_forward_diff import make_eos_inject_hook, evaluate_strict_clean, EN_PROMPTS_50, log
import torch
log("qwen3 b=30 delayed2 test")
model, tok, dev = load_model("qwen3")
e = tok.eos_token_id
ps = EN_PROMPTS_50[:3]
for b in [30, 40]:
    h = model.lm_head.register_forward_hook(make_eos_inject_hook(e, b, 2))
    cc = 0
    for pi, p in enumerate(ps):
        ids = tok.encode(p, return_tensors="pt").to(dev)
        with torch.no_grad(): oid = model.generate(ids, max_new_tokens=30, do_sample=False, pad_token_id=e)
        gt = oid[0][ids.shape[1]:]
        gen = tok.decode(gt, skip_special_tokens=False)
        he = gt[-1].item() == e
        ng = len(gt)
        ce = evaluate_strict_clean(p, gen, he, ng)
        cc += int(ce["strict_clean"])
        log(f"  b={b} p{pi}: eos={he} clean={ce['strict_clean']} toks={ng} text={gen[:50]}")
    log(f"  b={b}: clean={cc}/{len(ps)}")
    h.remove()
release_model(model)
