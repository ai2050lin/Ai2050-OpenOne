import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))

import torch
import torch.nn.functional as F
from model_utils import load_model, get_layers, release_model, get_model_info

model, tokenizer, device = load_model("glm4")
info = get_model_info(model, "glm4")

test_cases = [
    ("猫的英文是", "cat"),
    ("Translate 猫 to English:", "cat"),
    ("猫 in English is", "cat"),
    ("The English word for 猫 is", "cat"),
    ("请将猫翻译成英文:", "cat"),
    ("English translation of 猫:", "cat"),
]

for prompt, target in test_cases:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    top5_vals, top5_ids = torch.topk(probs, 5)
    top5 = [(tokenizer.decode([tid]), f"{probs[tid].item():.4f}") for tid in top5_ids.tolist()]
    print(f"Prompt: '{prompt}'")
    print(f"  Top-5: {top5}")
    del outputs

release_model(model)
