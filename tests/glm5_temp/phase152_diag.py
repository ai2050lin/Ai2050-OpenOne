"""诊断 Phase 152 Method B 的 corr(δ_ℓ, δ_0) = 0 问题"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import numpy as np
import torch
from model_utils import load_model, get_layers, get_model_info, release_model

model, tokenizer, device = load_model('qwen3')
info = get_model_info(model, 'qwen3')

prompt = "The scientist discovered that the"
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
last_pos = input_ids.shape[1] - 1

# Clean forward
with torch.no_grad():
    out_clean = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
clean_hs = out_clean.hidden_states

print(f"n_layers={info.n_layers}, d_model={info.d_model}")
print(f"Number of hidden states: {len(clean_hs)}")
for i in range(min(5, len(clean_hs))):
    print(f"  hs[{i}] shape={clean_hs[i].shape}, norm={clean_hs[i][0, last_pos, :].float().norm():.4f}")

# Test perturbation at L0
np.random.seed(42)
delta = np.random.randn(info.d_model)
delta = delta / np.linalg.norm(delta) * 1.0
delta_tensor = torch.tensor(delta, dtype=torch.float32)

layers = get_layers(model)

def make_hook(pos, delta_t):
    def hook(module, input, output):
        if isinstance(output, tuple):
            out = output[0].clone()
            out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
            return (out,) + output[1:]
        else:
            out = output.clone()
            out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
            return out
    return hook

hooks = [layers[0].register_forward_hook(make_hook(last_pos, delta_tensor))]

with torch.no_grad():
    out_p = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

for h in hooks:
    h.remove()

# 检查各层的delta_prop
print("\nDelta propagation from L0 hook injection:")
for li in [0, 1, 2, 4, 8, 12, 18, 24, 30, 36]:
    if li >= len(out_p.hidden_states):
        continue
    p_vec = out_p.hidden_states[li][0, last_pos, :].float().cpu().numpy()
    c_vec = clean_hs[li][0, last_pos, :].float().cpu().numpy()
    delta_prop = p_vec - c_vec
    norm_prop = np.linalg.norm(delta_prop)
    cos_with_input = float(np.dot(delta_prop, delta) / (norm_prop * np.linalg.norm(delta))) if norm_prop > 1e-10 else 0
    print(f"  hs[{li:>2d}]: ||delta_prop||={norm_prop:.6f}, cos(delta_prop, delta_input)={cos_with_input:.6f}")

# 关键: hidden_states[0] 是 embedding 层输出, 不受 L0 hook 影响!
# hidden_states[1] 是第1层 transformer 输出, 受 L0 hook 影响!
# 所以 delta_at_layer[0] = hs[0]_perturbed - hs[0]_clean 应该为0!

print("\n=== Key insight ===")
print("hidden_states[0] = embedding output (BEFORE any transformer layer)")
print("hidden_states[1] = output of layer 0 (AFTER L0 hook)")
print("So delta_at_layer for hs[0] should be ~0!")
print("The hook is on layers[0], which transforms hs[0] -> hs[1]")
print("Therefore delta propagation should be measured starting from hs[1], not hs[0]!")

# 验证
delta_hs0 = out_p.hidden_states[0][0, last_pos, :].float().cpu().numpy() - clean_hs[0][0, last_pos, :].float().cpu().numpy()
delta_hs1 = out_p.hidden_states[1][0, last_pos, :].float().cpu().numpy() - clean_hs[1][0, last_pos, :].float().cpu().numpy()
print(f"\n||delta at hs[0]|| = {np.linalg.norm(delta_hs0):.8f} (should be ~0)")
print(f"||delta at hs[1]|| = {np.linalg.norm(delta_hs1):.6f} (should be >0)")
print(f"cos(delta_hs1, delta_input) = {float(np.dot(delta_hs1, delta)/(np.linalg.norm(delta_hs1)*np.linalg.norm(delta))):.6f}")

release_model(model)
