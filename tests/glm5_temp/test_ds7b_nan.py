import sys
sys.path.insert(0, 'tests/glm5')
from model_utils import load_model, get_layers, release_model
import torch
import numpy as np

m, t, d = load_model('deepseek7b')
layers = get_layers(m)

inputs = t('The apple is a type of fruit', return_tensors='pt')
input_ids = inputs['input_ids'].to(d)
attn_mask = inputs['attention_mask'].to(d)

# Baseline
with torch.no_grad():
    out_base = m(input_ids=input_ids, attention_mask=attn_mask)
base_logits = out_base.logits[0, -1].float().cpu().numpy()
print(f"Base logits nan: {np.isnan(base_logits).any()}, inf: {np.isinf(base_logits).any()}")
print(f"Base top5: {np.argsort(base_logits)[-5:][::-1]}")

# Check hidden states for nan
with torch.no_grad():
    out = m(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
for i, hs in enumerate(out.hidden_states):
    h = hs[0, -1].float().cpu().numpy()
    if np.isnan(h).any() or np.isinf(h).any():
        print(f"  Hidden state L{i}: nan={np.isnan(h).any()}, inf={np.isinf(h).any()}")

# Test o_proj hook for DS7B
n_heads = 28
head_dim = 128

def make_oproj_pre_hook(head_indices_to_zero, n_heads, head_dim):
    def hook(module, input):
        attn_output = input[0]
        batch, seq, d = attn_output.shape
        print(f"  o_proj input shape: {attn_output.shape}, dtype={attn_output.dtype}")
        try:
            attn_reshaped = attn_output.view(batch, seq, n_heads, head_dim)
            for hi in head_indices_to_zero:
                attn_reshaped[:, :, hi, :] = 0.0
        except Exception as e:
            print(f"  Hook error: {e}")
        return input
    return hook

hooks = [layers[14].self_attn.o_proj.register_forward_pre_hook(
    make_oproj_pre_hook([0, 1, 2], 28, 128)
)]

with torch.no_grad():
    out_abl = m(input_ids=input_ids, attention_mask=attn_mask)
abl_logits = out_abl.logits[0, -1].float().cpu().numpy()
print(f"Ablated logits nan: {np.isnan(abl_logits).any()}, inf: {np.isinf(abl_logits).any()}")

for h in hooks: h.remove()

# Test MLP hook
def make_mlp_zero_hook():
    def hook(module, input, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)
    return hook

hooks2 = [layers[14].mlp.register_forward_hook(make_mlp_zero_hook())]
with torch.no_grad():
    out_abl2 = m(input_ids=input_ids, attention_mask=attn_mask)
abl2_logits = out_abl2.logits[0, -1].float().cpu().numpy()
print(f"MLP ablated logits nan: {np.isnan(abl2_logits).any()}, inf: {np.isinf(abl2_logits).any()}")

for h in hooks2: h.remove()
release_model(m)
