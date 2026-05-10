"""Verify Phase 81 vs Phase 82 discrepancy in Jacobian similarity."""
import torch, numpy as np
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained('gpt2-small', center_unembed=False, center_writing_weights=False, fold_ln=False, device='cpu')
model.eval()
layer = 6

def compute_jacobian(model, cache, layer):
    pre_gelu = cache[f'blocks.{layer}.mlp.hook_pre'][-1].detach()
    x = pre_gelu.clone().requires_grad_(True)
    y = torch.nn.functional.gelu(x)
    gd = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0].detach()
    W_in = model.blocks[layer].mlp.W_in.detach()
    W_out = model.blocks[layer].mlp.W_out.detach()
    J = W_out.T @ (gd.unsqueeze(1) * W_in.T)
    return J

tasks = {
    'addition': ['5 + 3 =', '12 + 7 =', '20 + 15 =', '8 + 2 =', '30 + 5 =', '15 + 9 =', '25 + 4 =', '7 + 11 =', '18 + 6 =', '3 + 14 ='],
    'antonym': ['The opposite of hot is', 'The opposite of big is', 'The opposite of fast is', 'The opposite of happy is', 'The opposite of strong is', 'The opposite of light is', 'The opposite of good is', 'The opposite of old is', 'The opposite of rich is', 'The opposite of tall is'],
    'capital': ['The capital of France is', 'The capital of Germany is', 'The capital of Japan is', 'The capital of Brazil is', 'The capital of Italy is', 'The capital of Spain is', 'The capital of China is', 'The capital of India is', 'The capital of Egypt is', 'The capital of Australia is'],
    'translate': ['The French word for cat is', 'The French word for dog is', 'The French word for house is', 'The French word for water is', 'The French word for book is', 'The French word for tree is', 'The French word for sun is', 'The French word for moon is', 'The French word for fire is', 'The French word for earth is'],
}

task_Js = {}
for task, prompts in tasks.items():
    Js = []
    for p in prompts:
        _, c = model.run_with_cache(model.to_tokens(p), remove_batch_dim=True)
        Js.append(compute_jacobian(model, c, layer))
    task_Js[task] = torch.stack(Js)

task_avg = {t: Js.mean(0) for t, Js in task_Js.items()}
task_names = list(tasks.keys())

# Flattened cosine matrix
print('=== Flattened Matrix Cosine (AVERAGE Jacobians) ===')
for t1 in task_names:
    row = f'{t1:12s}'
    for t2 in task_names:
        cos = torch.nn.functional.cosine_similarity(task_avg[t1].flatten().unsqueeze(0), task_avg[t2].flatten().unsqueeze(0)).item()
        row += f'{cos:>9.4f}'
    print(row)

# Spectral (SV) cosine
print()
print('=== Spectral (Singular Value) Cosine ===')
task_svs = {}
for t in task_names:
    U, S, Vt = torch.linalg.svd(task_avg[t], full_matrices=False)
    task_svs[t] = S

for t1 in task_names:
    row = f'{t1:12s}'
    for t2 in task_names:
        cos = torch.nn.functional.cosine_similarity(task_svs[t1].unsqueeze(0), task_svs[t2].unsqueeze(0)).item()
        row += f'{cos:>9.4f}'
    print(row)

# Within-task cosine
print()
print('=== Within-task Flattened Cosine ===')
for t in task_names:
    Js = task_Js[t]
    within = []
    for i in range(min(5, len(Js))):
        for j in range(i+1, min(5, len(Js))):
            cos = torch.nn.functional.cosine_similarity(Js[i].flatten().unsqueeze(0), Js[j].flatten().unsqueeze(0)).item()
            within.append(cos)
    print(f'  {t:12s}: {np.mean(within):.4f}')
