"""验证: L0/L6 MLP的100%崩塌是否是全局性的?
如果L3 MLP ablate也导致100%崩塌 → 说明这是MLP组件的通用问题
如果只有L0/L6导致崩塌 → 说明它们确实有特殊因果角色
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))

import torch
import torch.nn.functional as F
import numpy as np
from model_utils import load_model, get_layers, get_model_info, release_model

model, tokenizer, device = load_model("qwen3")
info = get_model_info(model, "qwen3")
n_layers = info.n_layers
layers = get_layers(model)

# 测试任务
test_pairs = [
    ("猫", "cat"), ("狗", "dog"), ("书", "book"),
    ("水", "water"), ("火", "fire"), ("花", "flower"),
    ("鱼", "fish"), ("树", "tree"),
]

# Task 1: 翻译
print("="*60)
print("任务1: 翻译 (猫的英文是)")
print("="*60)

baseline_trans = []
for zh, en in test_pairs:
    prompt = f"{zh}的英文是"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    # 找en token的概率
    best = 0.0
    for cand in [en, f" {en}", en.lower(), en.capitalize()]:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if ids:
            best = max(best, probs[ids[0]].item())
    baseline_trans.append(best)
    del outputs

baseline_trans_mean = np.mean(baseline_trans)
print(f"Baseline: {baseline_trans_mean:.4f}")

# Task 2: 补全 (猫是什么)
print("\n任务2: 补全 (猫是什么动物)")
print("="*60)

comp_prompts = [f"{zh}是什么动物" for zh, _ in test_pairs]
baseline_comp = []
for prompt in comp_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    top1 = probs.max().item()
    baseline_comp.append(top1)
    del outputs

baseline_comp_mean = np.mean(baseline_comp)
print(f"Baseline补全: {baseline_comp_mean:.4f}")

# 逐层MLP ablate
print("\n逐层MLP zero-ablate:")
print(f"{'Layer':>6} {'翻译prob':>10} {'翻译drop':>10} {'补全prob':>10} {'补全drop':>10}")
print("-"*50)

test_layers = [0, 3, 6, 9, 12, 18, 24, 30, 35]

for li in test_layers:
    def make_hook():
        def hook(module, input, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
        return hook

    handle = layers[li].mlp.register_forward_hook(make_hook())

    # 翻译
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

    # 补全
    comp_probs = []
    for prompt in comp_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        top1 = probs.max().item()
        comp_probs.append(top1)
        del outputs

    handle.remove()

    trans_mean = np.mean(trans_probs)
    comp_mean = np.mean(comp_probs)
    trans_drop = 1 - trans_mean / baseline_trans_mean if baseline_trans_mean > 0 else 0
    comp_drop = 1 - comp_mean / baseline_comp_mean if baseline_comp_mean > 0 else 0

    print(f"L{li:>4} {trans_mean:>10.4f} {trans_drop:>10.1%} {comp_mean:>10.4f} {comp_drop:>10.1%}")

release_model(model)
