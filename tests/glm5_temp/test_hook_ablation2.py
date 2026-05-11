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

n_heads = 32
head_dim = 80

# Baseline
with torch.no_grad():
    out_base = m(input_ids=input_ids, attention_mask=attn_mask)
base_logits = out_base.logits[0, -1].float().cpu().numpy()
p_base = np.exp(base_logits - np.max(base_logits)); p_base /= p_base.sum()

def compute_kl(p1, logits2):
    p2 = np.exp(logits2 - np.max(logits2)); p2 /= p2.sum()
    return float(np.sum(p1 * (np.log(p1 + 1e-10) - np.log(p2 + 1e-10))))

# Approach: Use register_forward_pre_hook on o_proj to modify input
# o_proj input is [batch, seq, d_model] = [batch, seq, n_heads * head_dim]
def make_oproj_pre_hook(head_indices_to_zero, n_heads, head_dim):
    """Hook到o_proj的输入, 将指定head的输出置零"""
    def hook(module, input):
        # input is tuple of tensors
        attn_output = input[0]  # [batch, seq, d_model]
        batch, seq, d = attn_output.shape
        # Reshape to [batch, seq, n_heads, head_dim]
        attn_reshaped = attn_output.view(batch, seq, n_heads, head_dim)
        for hi in head_indices_to_zero:
            attn_reshaped[:, :, hi, :] = 0.0
        # No need to reshape back since view doesn't copy
        return input
    return hook

# Test: zero ALL heads in layer 18
hooks = [layers[18].self_attn.o_proj.register_forward_pre_hook(
    make_oproj_pre_hook(list(range(32)), 32, 80)
)]

with torch.no_grad():
    out_abl = m(input_ids=input_ids, attention_mask=attn_mask)
abl_logits = out_abl.logits[0, -1].float().cpu().numpy()

for h in hooks: h.remove()

kl = compute_kl(p_base, abl_logits)
print(f"KL (L18 ALL heads zeroed via o_proj pre_hook) = {kl:.6f}")

# Test: zero half heads (0-15) in layer 18
hooks2 = [layers[18].self_attn.o_proj.register_forward_pre_hook(
    make_oproj_pre_hook(list(range(16)), 32, 80)
)]
with torch.no_grad():
    out_abl2 = m(input_ids=input_ids, attention_mask=attn_mask)
abl2_logits = out_abl2.logits[0, -1].float().cpu().numpy()
for h in hooks2: h.remove()
kl2 = compute_kl(p_base, abl2_logits)
print(f"KL (L18 heads 0-15 zeroed via o_proj pre_hook) = {kl2:.6f}")

# Test: zero ALL heads in layers 10-19
hooks3 = []
for li in range(10, 20):
    hooks3.append(layers[li].self_attn.o_proj.register_forward_pre_hook(
        make_oproj_pre_hook(list(range(32)), 32, 80)
    ))
with torch.no_grad():
    out_abl3 = m(input_ids=input_ids, attention_mask=attn_mask)
abl3_logits = out_abl3.logits[0, -1].float().cpu().numpy()
for h in hooks3: h.remove()
kl3 = compute_kl(p_base, abl3_logits)
print(f"KL (L10-19 ALL heads zeroed via o_proj pre_hook) = {kl3:.6f}")

# Test: zero ALL heads in ALL layers
hooks4 = []
for li in range(36):
    hooks4.append(layers[li].self_attn.o_proj.register_forward_pre_hook(
        make_oproj_pre_hook(list(range(32)), 32, 80)
    ))
with torch.no_grad():
    out_abl4 = m(input_ids=input_ids, attention_mask=attn_mask)
abl4_logits = out_abl4.logits[0, -1].float().cpu().numpy()
for h in hooks4: h.remove()
kl4 = compute_kl(p_base, abl4_logits)
print(f"KL (ALL layers ALL heads zeroed via o_proj pre_hook) = {kl4:.6f}")

# Compare: what's the KL when we just remove residual connection from attention?
# This would be equivalent to zeroing all attention output
# We can do this by hooking into the layer and zeroing the attention contribution

release_model(m)
