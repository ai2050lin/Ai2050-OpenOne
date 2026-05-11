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

# 正确的维度
n_heads = 32
head_dim = 128  # 不是 d_model // n_heads = 80!
attn_output_dim = n_heads * head_dim  # 4096

# Baseline
with torch.no_grad():
    out_base = m(input_ids=input_ids, attention_mask=attn_mask)
base_logits = out_base.logits[0, -1].float().cpu().numpy()
p_base = np.exp(base_logits - np.max(base_logits)); p_base /= p_base.sum()

def compute_kl(p1, logits2):
    p2 = np.exp(logits2 - np.max(logits2)); p2 /= p2.sum()
    return float(np.sum(p1 * (np.log(p1 + 1e-10) - np.log(p2 + 1e-10))))

def make_oproj_pre_hook(head_indices_to_zero, n_heads, head_dim):
    """Hook到o_proj的输入, 将指定head的输出置零"""
    def hook(module, input):
        attn_output = input[0]  # [batch, seq, attn_output_dim]
        batch, seq, d = attn_output.shape
        assert d == n_heads * head_dim, f"Expected {n_heads*head_dim}, got {d}"
        # Reshape to [batch, seq, n_heads, head_dim]
        attn_reshaped = attn_output.view(batch, seq, n_heads, head_dim)
        for hi in head_indices_to_zero:
            attn_reshaped[:, :, hi, :] = 0.0
        return input
    return hook

# Test: zero ALL heads in layer 18
hooks = [layers[18].self_attn.o_proj.register_forward_pre_hook(
    make_oproj_pre_hook(list(range(32)), 32, 128)
)]
with torch.no_grad():
    out_abl = m(input_ids=input_ids, attention_mask=attn_mask)
abl_logits = out_abl.logits[0, -1].float().cpu().numpy()
for h in hooks: h.remove()
kl = compute_kl(p_base, abl_logits)
print(f"KL (L18 ALL heads zeroed) = {kl:.6f}")

# Test: zero half heads in layer 18
hooks2 = [layers[18].self_attn.o_proj.register_forward_pre_hook(
    make_oproj_pre_hook(list(range(16)), 32, 128)
)]
with torch.no_grad():
    out_abl2 = m(input_ids=input_ids, attention_mask=attn_mask)
abl2_logits = out_abl2.logits[0, -1].float().cpu().numpy()
for h in hooks2: h.remove()
kl2 = compute_kl(p_base, abl2_logits)
print(f"KL (L18 heads 0-15 zeroed) = {kl2:.6f}")

# Test: zero ALL heads in layers 10-19
hooks3 = []
for li in range(10, 20):
    hooks3.append(layers[li].self_attn.o_proj.register_forward_pre_hook(
        make_oproj_pre_hook(list(range(32)), 32, 128)
    ))
with torch.no_grad():
    out_abl3 = m(input_ids=input_ids, attention_mask=attn_mask)
abl3_logits = out_abl3.logits[0, -1].float().cpu().numpy()
for h in hooks3: h.remove()
kl3 = compute_kl(p_base, abl3_logits)
print(f"KL (L10-19 ALL heads zeroed) = {kl3:.6f}")

# Test: zero ALL heads in ALL layers (attention completely disabled)
hooks4 = []
for li in range(36):
    hooks4.append(layers[li].self_attn.o_proj.register_forward_pre_hook(
        make_oproj_pre_hook(list(range(32)), 32, 128)
    ))
with torch.no_grad():
    out_abl4 = m(input_ids=input_ids, attention_mask=attn_mask)
abl4_logits = out_abl4.logits[0, -1].float().cpu().numpy()
for h in hooks4: h.remove()
kl4 = compute_kl(p_base, abl4_logits)
print(f"KL (ALL layers ALL heads zeroed) = {kl4:.6f}")

# For comparison: MLP ablation in layers 10-19
def make_mlp_zero_hook():
    def hook(module, input, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)
    return hook

hooks5 = []
for li in range(10, 20):
    hooks5.append(layers[li].mlp.register_forward_hook(make_mlp_zero_hook()))
with torch.no_grad():
    out_abl5 = m(input_ids=input_ids, attention_mask=attn_mask)
abl5_logits = out_abl5.logits[0, -1].float().cpu().numpy()
for h in hooks5: h.remove()
kl5 = compute_kl(p_base, abl5_logits)
print(f"KL (L10-19 ALL MLP zeroed) = {kl5:.6f}")

release_model(m)
