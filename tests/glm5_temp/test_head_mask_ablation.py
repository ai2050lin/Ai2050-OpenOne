import sys
sys.path.insert(0, 'tests/glm5')
from model_utils import load_model, release_model
import torch
import numpy as np

m, t, d = load_model('qwen3')
inputs = t('The apple is a type of fruit', return_tensors='pt')
input_ids = inputs['input_ids'].to(d)
attn_mask = inputs['attention_mask'].to(d)

# Baseline
with torch.no_grad():
    out_base = m(input_ids=input_ids, attention_mask=attn_mask)
base_logits = out_base.logits[0, -1].float().cpu().numpy()
print(f"Baseline top5: {np.argsort(base_logits)[-5:][::-1]}")

# Head mask: zero ALL heads in middle layer
n_layers = 36
n_heads = 32
head_mask = torch.ones(n_layers, n_heads, device=d)
head_mask[18, :] = 0.0  # zero ALL heads in layer 18

with torch.no_grad():
    out_abl = m(input_ids=input_ids, attention_mask=attn_mask, head_mask=head_mask)
abl_logits = out_abl.logits[0, -1].float().cpu().numpy()
print(f"Ablated L18 all heads top5: {np.argsort(abl_logits)[-5:][::-1]}")

# KL
p_base = np.exp(base_logits - np.max(base_logits)); p_base /= p_base.sum()
p_abl = np.exp(abl_logits - np.max(abl_logits)); p_abl /= p_abl.sum()
kl = float(np.sum(p_base * (np.log(p_base + 1e-10) - np.log(p_abl + 1e-10))))
print(f"KL = {kl:.4f}")

# Also test: zero heads in layers 10-20
head_mask2 = torch.ones(n_layers, n_heads, device=d)
for li in range(10, 20):
    head_mask2[li, :] = 0.0

with torch.no_grad():
    out_abl2 = m(input_ids=input_ids, attention_mask=attn_mask, head_mask=head_mask2)
abl2_logits = out_abl2.logits[0, -1].float().cpu().numpy()
p_abl2 = np.exp(abl2_logits - np.max(abl2_logits)); p_abl2 /= p_abl2.sum()
kl2 = float(np.sum(p_base * (np.log(p_base + 1e-10) - np.log(p_abl2 + 1e-10))))
print(f"KL (L10-19 all heads zeroed) = {kl2:.4f}")

# And ALL heads zeroed
head_mask3 = torch.zeros(n_layers, n_heads, device=d)
with torch.no_grad():
    out_abl3 = m(input_ids=input_ids, attention_mask=attn_mask, head_mask=head_mask3)
abl3_logits = out_abl3.logits[0, -1].float().cpu().numpy()
p_abl3 = np.exp(abl3_logits - np.max(abl3_logits)); p_abl3 /= p_abl3.sum()
kl3 = float(np.sum(p_base * (np.log(p_base + 1e-10) - np.log(p_abl3 + 1e-10))))
print(f"KL (ALL heads zeroed) = {kl3:.4f}")

release_model(m)
