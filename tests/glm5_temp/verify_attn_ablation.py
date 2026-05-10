"""验证: L6 Attn的83.6% drop是否是翻译特异的?"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))

import torch
import torch.nn.functional as F
import numpy as np
from model_utils import load_model, get_layers, get_model_info, release_model

model, tokenizer, device = load_model("qwen3")
info = get_model_info(model, "qwen3")
layers = get_layers(model)

test_pairs = [
    ("猫", "cat"), ("狗", "dog"), ("书", "book"),
    ("水", "water"), ("火", "fire"), ("花", "flower"),
    ("鱼", "fish"), ("树", "tree"),
]

# Baseline
baseline_trans = []
for zh, en in test_pairs:
    prompt = f"{zh}的英文是"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    best = 0.0
    for cand in [en, f" {en}", en.lower(), en.capitalize()]:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if ids:
            best = max(best, probs[ids[0]].item())
    baseline_trans.append(best)
    del outputs
baseline_trans_mean = np.mean(baseline_trans)

comp_prompts = [f"{zh}是什么动物" for zh, _ in test_pairs]
baseline_comp = []
for prompt in comp_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    baseline_comp.append(probs.max().item())
    del outputs
baseline_comp_mean = np.mean(baseline_comp)

print(f"Baseline: 翻译={baseline_trans_mean:.4f}, 补全={baseline_comp_mean:.4f}")

# 逐层Attn ablate
print(f"\n{'Layer':>6} {'翻译prob':>10} {'翻译drop':>10} {'补全prob':>10} {'补全drop':>10} {'特异性':>10}")
print("-"*60)

test_layers = [0, 3, 6, 9, 12, 18, 24, 26, 28, 30, 31, 35]

for li in test_layers:
    def make_hook():
        def hook(module, input, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
        return hook

    handle = layers[li].self_attn.register_forward_hook(make_hook())

    trans_probs = []
    for zh, en in test_pairs:
        prompt = f"{zh}的英文是"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        best = 0.0
        for cand in [en, f" {en}", en.lower(), en.capitalize()]:
            ids = tokenizer.encode(cand, add_special_tokens=False)
            if ids:
                best = max(best, probs[ids[0]].item())
        trans_probs.append(best)
        del outputs

    comp_probs = []
    for prompt in comp_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        comp_probs.append(probs.max().item())
        del outputs

    handle.remove()

    trans_mean = np.mean(trans_probs)
    comp_mean = np.mean(comp_probs)
    trans_drop = 1 - trans_mean / baseline_trans_mean if baseline_trans_mean > 0 else 0
    comp_drop = 1 - comp_mean / baseline_comp_mean if baseline_comp_mean > 0 else 0
    specificity = trans_drop - comp_drop

    marker = " ← 翻译关键!" if trans_drop > 0.5 else (" ← 特异!" if specificity > 0.3 else "")
    print(f"L{li:>4} {trans_mean:>10.4f} {trans_drop:>10.1%} {comp_mean:>10.4f} {comp_drop:>10.1%} {specificity:>10.1%}{marker}")

release_model(model)
