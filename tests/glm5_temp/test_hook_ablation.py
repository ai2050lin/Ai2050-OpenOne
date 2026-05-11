import sys
sys.path.insert(0, 'tests/glm5')
from model_utils import load_model, get_layers, release_model
import torch
import numpy as np

m, t, d = load_model('qwen3')
layers = get_layers(m)
inputs = t('The apple is a type of fruit', return_tensors='pt')
input_ids = inputs['input_ids'].to(d)
attn_mask = inputs['attention_mask'].to(d)

# Baseline
with torch.no_grad():
    out_base = m(input_ids=input_ids, attention_mask=attn_mask)
base_logits = out_base.logits[0, -1].float().cpu().numpy()

# Hook-based ablation: zero out specific heads
# Strategy: hook into self_attn, capture attn_output, zero out specific heads
def make_attn_zero_hook(head_indices_to_zero, n_heads, head_dim):
    """创建一个hook, 将指定head的输出置零"""
    def hook(module, input, output):
        # output[0] = attn_output [batch, seq, d_model]
        # 需要在reshape后zero out特定head
        if isinstance(output, tuple):
            attn_output = output[0]  # [batch, seq, d_model]
        else:
            attn_output = output
        
        batch, seq, d = attn_output.shape
        # Reshape to [batch, seq, n_heads, head_dim]
        attn_output_reshaped = attn_output.view(batch, seq, n_heads, head_dim)
        # Zero out specified heads
        for hi in head_indices_to_zero:
            attn_output_reshaped[:, :, hi, :] = 0.0
        # Reshape back
        attn_output_modified = attn_output_reshaped.view(batch, seq, d)
        
        if isinstance(output, tuple):
            return (attn_output_modified,) + output[1:]
        return attn_output_modified
    return hook

n_heads = 32
head_dim = 80

# Test: zero ALL heads in layer 18
hooks = [layers[18].self_attn.register_forward_hook(
    make_attn_zero_hook(list(range(32)), 32, 80)
)]

with torch.no_grad():
    out_abl = m(input_ids=input_ids, attention_mask=attn_mask)
abl_logits = out_abl.logits[0, -1].float().cpu().numpy()

for h in hooks: h.remove()

p_base = np.exp(base_logits - np.max(base_logits)); p_base /= p_base.sum()
p_abl = np.exp(abl_logits - np.max(abl_logits)); p_abl /= p_abl.sum()
kl = float(np.sum(p_base * (np.log(p_base + 1e-10) - np.log(p_abl + 1e-10))))
print(f"KL (L18 ALL heads zeroed via hook) = {kl:.4f}")
print(f"Base top5: {np.argsort(base_logits)[-5:][::-1]}")
print(f"Ablated top5: {np.argsort(abl_logits)[-5:][::-1]}")

# Test: zero ALL heads in layers 10-19
hooks2 = []
for li in range(10, 20):
    hooks2.append(layers[li].self_attn.register_forward_hook(
        make_attn_zero_hook(list(range(32)), 32, 80)
    ))

with torch.no_grad():
    out_abl2 = m(input_ids=input_ids, attention_mask=attn_mask)
abl2_logits = out_abl2.logits[0, -1].float().cpu().numpy()

for h in hooks2: h.remove()

p_abl2 = np.exp(abl2_logits - np.max(abl2_logits)); p_abl2 /= p_abl2.sum()
kl2 = float(np.sum(p_base * (np.log(p_base + 1e-10) - np.log(p_abl2 + 1e-10))))
print(f"KL (L10-19 ALL heads zeroed via hook) = {kl2:.4f}")

# Test: zero 1 head per layer for 10 layers
hooks3 = []
for li in range(10, 20):
    hooks3.append(layers[li].self_attn.register_forward_hook(
        make_attn_zero_hook([0], 32, 80)  # just head 0
    ))

with torch.no_grad():
    out_abl3 = m(input_ids=input_ids, attention_mask=attn_mask)
abl3_logits = out_abl3.logits[0, -1].float().cpu().numpy()

for h in hooks3: h.remove()

p_abl3 = np.exp(abl3_logits - np.max(abl3_logits)); p_abl3 /= p_abl3.sum()
kl3 = float(np.sum(p_base * (np.log(p_base + 1e-10) - np.log(p_abl3 + 1e-10))))
print(f"KL (L10-19 head0 zeroed via hook) = {kl3:.4f}")

release_model(m)
