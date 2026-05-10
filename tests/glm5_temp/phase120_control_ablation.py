"""Phase 120 Control: Random vs Spike vs Complement ablation comparison"""
import sys
sys.path.insert(0, '.')
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.special import softmax
import json
from pathlib import Path

print('=== Random Direction Ablation Control ===')
print('Loading Qwen3-4B...')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B', torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

test_words = ['cat', 'red', 'happy', 'bread', 'head', 'rain', 'hammer', 'shirt', 'car', 'house']
template = 'Translate the word "{}" into Chinese.'

# Collect L12 residuals
all_residuals = []
for w in test_words:
    prompt = template.format(w)
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hs = outputs.hidden_states[12]
    input_ids = inputs['input_ids'][0]
    non_pad = (input_ids != tokenizer.pad_token_id).nonzero()
    last_pos = non_pad[-1].item() if len(non_pad) > 0 else -1
    all_residuals.append(hs[0, last_pos, :].cpu().float().numpy())

H = np.array(all_residuals)
H_centered = H - H.mean(axis=0, keepdims=True)

# Spike PCA
U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
V_spike = Vt[:25]

# Activation spike fraction
spike_component = H_centered @ V_spike.T @ V_spike
spike_energy = np.mean(np.sum(spike_component**2, axis=1))
total_energy = np.mean(np.sum(H_centered**2, axis=1))
print(f'L12 spike_frac (activation) = {spike_energy/total_energy:.4f}')

V_spike_t = torch.tensor(V_spike, dtype=torch.float32, device=model.device)

def run_ablation(V_t, label):
    kl_list = []
    for w in test_words:
        prompt = template.format(w)
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            baseline_out = model(**inputs)
        input_ids = inputs['input_ids'][0]
        non_pad = (input_ids != tokenizer.pad_token_id).nonzero()
        last_pos = non_pad[-1].item() if len(non_pad) > 0 else -1
        baseline_logit = baseline_out.logits[0, last_pos, :].cpu().float().numpy()
        
        def ablation_hook(module, input, output, Vr=V_t):
            hs = output[0]
            hs_float = hs.float()
            proj = hs_float @ Vr.T @ Vr
            return ((hs_float - proj).to(hs.dtype),) + output[1:]
        
        hook = model.model.layers[12].register_forward_hook(ablation_hook)
        with torch.no_grad():
            ablated_out = model(**inputs)
        hook.remove()
        
        ablated_logit = ablated_out.logits[0, last_pos, :].cpu().float().numpy()
        p = softmax(baseline_logit)
        q = softmax(ablated_logit)
        kl = np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
        kl_list.append(float(kl))
    
    mean_kl = np.mean(kl_list)
    print(f'{label}: mean_KL = {mean_kl:.6f} +/- {np.std(kl_list):.6f}')
    return kl_list

# 1. Spike ablation
spike_kls = run_ablation(V_spike_t, 'Spike (25 PCA dims)')

# 2. Random ablation (5 trials)
np.random.seed(42)
random_all = []
for trial in range(5):
    V_random = np.random.randn(25, 2560)
    V_random = V_random / np.linalg.norm(V_random, axis=1, keepdims=True)
    V_random = np.linalg.qr(V_random.T)[0].T[:25]  # Orthogonalize
    V_random_t = torch.tensor(V_random, dtype=torch.float32, device=model.device)
    kls = run_ablation(V_random_t, f'Random trial {trial+1}')
    random_all.append(np.mean(kls))

print(f'Random ablation: mean_KL = {np.mean(random_all):.6f} +/- {np.std(random_all):.6f}')

# 3. Complement PCA ablation
comp_H = H_centered - H_centered @ V_spike.T @ V_spike
U_c, S_c, Vt_c = np.linalg.svd(comp_H, full_matrices=False)
V_comp = Vt_c[:25]
V_comp_t = torch.tensor(V_comp, dtype=torch.float32, device=model.device)
comp_kls = run_ablation(V_comp_t, 'Complement PCA (25 dims)')

# 4. Top spike only (5 dims)
V_spike5_t = torch.tensor(V_spike[:5], dtype=torch.float32, device=model.device)
spike5_kls = run_ablation(V_spike5_t, 'Spike top 5 PCA dims')

# Summary
results = {
    'spike_frac_activation': float(spike_energy / total_energy),
    'spike_25_kl_mean': float(np.mean(spike_kls)),
    'spike_5_kl_mean': float(np.mean(spike5_kls)),
    'random_25_kl_mean': float(np.mean(random_all)),
    'random_25_kl_std': float(np.std(random_all)),
    'comp_25_kl_mean': float(np.mean(comp_kls)),
    'spike_vs_random_ratio': float(np.mean(spike_kls) / np.mean(random_all)),
    'spike_vs_comp_ratio': float(np.mean(spike_kls) / np.mean(comp_kls)),
}

print()
print('=== SUMMARY ===')
print(f'Spike (25 dims, {spike_energy/total_energy*100:.1f}% act): KL = {np.mean(spike_kls):.4f}')
print(f'Spike top 5 (5 dims): KL = {np.mean(spike5_kls):.4f}')
print(f'Random (25 dims): KL = {np.mean(random_all):.4f}')
print(f'Complement PCA (25 dims): KL = {np.mean(comp_kls):.4f}')
print(f'Spike/Random ratio: {np.mean(spike_kls)/np.mean(random_all):.2f}x')
print(f'Spike/Complement ratio: {np.mean(spike_kls)/np.mean(comp_kls):.2f}x')

save_path = Path('tests/glm5_temp/phase120_control_ablation_qwen3.json')
with open(save_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to {save_path}')

del model
torch.cuda.empty_cache()
print('Done!')
