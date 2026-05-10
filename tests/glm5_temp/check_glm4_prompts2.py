import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))

import torch
import torch.nn.functional as F
from model_utils import load_model, release_model

model, tokenizer, device = load_model("glm4")

test_entities = [
    ("猫", "cat"), ("狗", "dog"), ("书", "book"),
    ("水", "water"), ("火", "fire"), ("花", "flower"),
    ("鱼", "fish"), ("树", "tree"), ("鸟", "bird"),
    ("马", "horse"), ("铁", "iron"), ("金", "gold"),
    ("茶", "tea"), ("米", "rice"), ("血", "blood"),
]

for zh, en in test_entities:
    prompt = f"{zh}的英文是"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    top5_vals, top5_ids = torch.topk(probs, 5)
    top5 = [(tokenizer.decode([tid]), f"{probs[tid].item():.4f}") for tid in top5_ids.tolist()]
    
    # 检查en的各种形式
    for cand in [en, f" {en}", en.capitalize()]:
        cid = tokenizer.encode(cand, add_special_tokens=False)
        if cid:
            p = probs[cid[0]].item()
            if p > 0.01:
                print(f"{prompt} → Top5={top5}, {cand}={p:.4f}")
                break
    else:
        print(f"{prompt} → Top5={top5}, {en} not found!")
    del outputs

release_model(model)
