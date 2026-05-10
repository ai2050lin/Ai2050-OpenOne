"""Phase 81 Exp C+D (minimal): key analysis with 10 samples"""
import torch, numpy as np
from transformer_lens import HookedTransformer
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist

def get_model():
    model = HookedTransformer.from_pretrained("gpt2-small", center_unembed=False,
        center_writing_weights=False, fold_ln=False, device="cpu")
    model.eval()
    return model

def compute_gelu_derivative(pre_act):
    x = pre_act.clone().detach().requires_grad_(True)
    y = torch.nn.functional.gelu(x)
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0].detach()

def compute_mlp_jacobian(model, cache, layer, position=-1):
    pre = cache[f'blocks.{layer}.mlp.hook_pre'][position]
    gd = compute_gelu_derivative(pre)
    W_in = model.blocks[layer].mlp.W_in.detach()
    W_out = model.blocks[layer].mlp.W_out.detach()
    return W_out.T @ (gd.unsqueeze(1) * W_in.T)

def gen(task, n, seed=42):
    rng = np.random.RandomState(seed)
    s = []
    if task == "addition":
        for _ in range(n): a,b=rng.randint(1,50),rng.randint(1,50); s.append(f"{a} + {b} =")
    elif task == "translate_fr":
        ws = ["big","small","old","new","red","blue","fast","slow","hot","cold"]
        for i in range(n): s.append(f"Translate to French: The {ws[i%10]} cat runs")
    elif task == "antonym":
        ws = ["hot","big","fast","happy","light","strong","loud","rough","wide","tall"]
        for i in range(n): s.append(f"The opposite of {ws[i%10]} is")
    elif task == "capital":
        cs = ["France","Germany","Japan","Italy","Spain","China","Brazil","India","Russia","Egypt"]
        for i in range(n): s.append(f"The capital of {cs[i%10]} is")
    return s

model = get_model()
tasks = ["addition","translate_fr","antonym","capital"]
N = 10

# === Exp C: Topology ===
print("="*60)
print("Exp C: Jacobian Field Topology")
print("="*60)

for layer in [3, 6, 9]:
    print(f"\n  Layer {layer}:")
    all_feats, all_labels = [], []
    task_feats = {}
    
    for task in tasks:
        samples = gen(task, N)
        feats = []
        for text in samples:
            tokens = model.to_tokens(text)
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            J = compute_mlp_jacobian(model, cache, layer)
            _, S, _ = torch.linalg.svd(J, full_matrices=False)
            feats.append(S.numpy()[:30])
        task_feats[task] = np.array(feats)
        all_feats.append(feats)
        all_labels.extend([task]*N)
    
    all_feats = np.vstack(all_feats)
    pca = PCA(n_components=5)
    proj = pca.fit_transform(all_feats)
    print(f"    PCA var: {pca.explained_variance_ratio_[:3]}")
    
    for task in tasks:
        mask = np.array(all_labels) == task
        c = proj[mask, 0].mean()
        print(f"    {task}: PC1={c:.4f}")
    
    # Within-task cosine distances
    for task in tasks:
        d = pdist(task_feats[task], 'cosine')
        print(f"    {task} within-cos-dist: mean={d.mean():.6f}, max={d.max():.6f}")

# === Exp D: Spectrum Dynamics ===
print(f"\n{'='*60}")
print("Exp D: Spectrum Dynamics & Recursive Rollout")
print("="*60)

for task in tasks[:2]:
    print(f"\n  {task}:")
    samples = gen(task, 3)
    for si, text in enumerate(samples):
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        
        jacobians = []
        for layer in range(12):
            J = compute_mlp_jacobian(model, cache, layer)
            jacobians.append(J.numpy())
        
        # Single-layer spectrum
        print(f"    Sample {si} single-layer:")
        for l in [0,3,6,9,11]:
            _, S, _ = torch.linalg.svd(torch.tensor(jacobians[l]), full_matrices=False)
            print(f"      L{l}: OpNorm={torch.norm(torch.tensor(jacobians[l])):.2f}, "
                  f"Top5={S[:5].tolist()[:3]}")
        
        # Composed spectrum
        J_comp = np.eye(768)
        print(f"    Sample {si} composed:")
        for k in range(12):
            J_comp = jacobians[k] @ J_comp
            S_c = np.linalg.svd(J_comp, compute_uv=False)
            print(f"      After L{k}: OpNorm={np.sqrt(np.sum(S_c**2)):.2e}, "
                  f"Top1={S_c[0]:.2e}, Cond={S_c[0]/(S_c[S_c>1e-10][-1]+1e-12):.1e}")
        break

# Key: Conserved modes
print(f"\n  Conserved vs Unstable Modes (3 samples each):")
for task in tasks:
    samples = gen(task, 3)
    all_SV = []
    for text in samples:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        jacs = [compute_mlp_jacobian(model, cache, l).numpy() for l in range(12)]
        J_c = np.eye(768)
        for k in range(12): J_c = jacs[k] @ J_c
        all_SV.append(np.linalg.svd(J_c, compute_uv=False))
    
    m = np.mean(all_SV, axis=0)
    amp = np.sum(m > 1.1); con = np.sum((m>0.9)&(m<=1.1)); att = np.sum((m>0.01)&(m<=0.9)); nz = np.sum(m<=0.01)
    print(f"  {task}: amp={amp}, cons={con}, att={att}, near0={nz}, range=[{m[0]:.1e}, {m[m>0.01][-1]:.1e}]")

print("\nDone!")
