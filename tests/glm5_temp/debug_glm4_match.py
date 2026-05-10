import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))

import torch
import torch.nn.functional as F
from model_utils import load_model, release_model

model, tokenizer, device = load_model("glm4")

# Debug: 看"猫的英文是"的top-10 token及其decode结果
prompt = "猫的英文是"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits[0, -1, :]
probs = F.softmax(logits, dim=-1)
top10_vals, top10_ids = torch.topk(probs, 10)

print("Top-10 tokens for '猫的英文是':")
for i, (val, tid) in enumerate(zip(top10_vals.tolist(), top10_ids.tolist())):
    decoded = tokenizer.decode([tid])
    decoded_stripped = decoded.strip().lower()
    contains_cat = "cat" in decoded_stripped
    print(f"  #{i+1}: id={tid}, decoded='{decoded}', stripped='{decoded_stripped}', "
          f"prob={val:.4f}, contains_cat={contains_cat}")

# 现在debug exp4的匹配逻辑
en = "cat"
en_lower = en.lower()
top10_tokens = [tokenizer.decode([tid]).strip().lower() for tid in top10_ids.tolist()]
print(f"\ntop10_tokens = {top10_tokens}")
print(f"en_lower = {en_lower}")
print(f"en_lower in top10_tokens[0]? {en_lower in top10_tokens[0]}")
print(f"any check: {any(en_lower in t for t in top10_tokens[:5])}")

release_model(model)
