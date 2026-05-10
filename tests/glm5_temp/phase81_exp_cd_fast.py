"""Phase 81 Exp C+D (fast): Jacobian Topology & Spectrum Dynamics"""
import torch
import numpy as np
from transformer_lens import HookedTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from itertools import combinations
from scipy.spatial.distance import pdist
from collections import Counter

def get_model():
    model = HookedTransformer.from_pretrained("gpt2-small", center_unembed=False,
        center_writing_weights=False, fold_ln=False, device="cpu")
    model.eval()
    return model

def compute_gelu_derivative(pre_act):
    x = pre_act.clone().detach().requires_grad_(True)
    y = torch.nn.functional.gelu(x)
    grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0]
    return grad.detach()

def compute_mlp_jacobian(model, cache, layer, position=-1):
    pre = cache[f'blocks.{layer}.mlp.hook_pre'][position]
    gelu_deriv = compute_gelu_derivative(pre)
    W_in = model.blocks[layer].mlp.W_in.detach()
    W_out = model.blocks[layer].mlp.W_out.detach()
    scaled_W_in_T = gelu_deriv.unsqueeze(1) * W_in.T
    J = W_out.T @ scaled_W_in_T
    return J

def gen_samples(task, n, seed=42):
    rng = np.random.RandomState(seed)
    samples = []
    if task == "addition":
        for _ in range(n):
            a, b = rng.randint(1,50), rng.randint(1,50)
            samples.append(f"{a} + {b} =")
    elif task == "translate_fr":
        adjs = ["big","small","old","new","red","blue","fast","slow","hot","cold","dark","bright"]
        nouns = ["cat","dog","bird","fish","tree","house","car","book","child","river"]
        verbs = ["runs","walks","sits","stands","jumps","flies","swims","sleeps","eats","drinks"]
        for i in range(n):
            adj, noun, verb = adjs[i%len(adjs)], nouns[(i//len(adjs))%len(nouns)], verbs[(i//30)%len(verbs)]
            samples.append(f"Translate to French: The {adj} {noun} {verb}")
    elif task == "antonym":
        words = ["hot","big","fast","happy","light","strong","loud","rough","wide","tall",
                "cold","small","slow","sad","dark","weak","quiet","smooth","narrow","short",
                "bright","heavy","soft","hard","old","young","rich","poor","thick","thin",
                "open","closed","full","empty","dry","wet","clean","dirty","safe","dangerous"]
        for i in range(n):
            samples.append(f"The opposite of {words[i%len(words)]} is")
    elif task == "capital":
        countries = ["France","Germany","Japan","Italy","Spain","China","Brazil","India",
                    "Russia","Egypt","UK","Canada","Mexico","Korea","Turkey","Norway",
                    "Sweden","Poland","Greece","Portugal","Australia","Argentina","Chile",
                    "Peru","Colombia","Thailand","Vietnam","Finland","Denmark","Austria"]
        for i in range(n):
            samples.append(f"The capital of {countries[i%len(countries)]} is")
    elif task == "continue":
        starts = ["The cat sat on","The dog ran to","The bird flew up","The fish swam down",
                  "Once upon a time","In the beginning","Long ago there","It was a dark",
                  "She walked into","He looked at the","They went to the","We found a"]
        for i in range(n):
            samples.append(starts[i%len(starts)])
    return samples

model = get_model()
task_names_5 = ["addition", "translate_fr", "antonym", "capital", "continue"]
n_samples = 100
layers = [3, 6, 9]

# ============================================================
# Experiment C: Jacobian Field Topology
# ============================================================
print("="*70)
print("Experiment C: Jacobian Field Topology")
print("="*70)

for layer in layers:
    print(f"\n{'='*60}")
    print(f"  Layer {layer}")
    print(f"{'='*60}")
    
    all_features = []
    all_labels = []
    task_features = {}
    
    for task_name in task_names_5:
        samples = gen_samples(task_name, n_samples)
        features = []
        for text in samples:
            tokens = model.to_tokens(text)
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            J = compute_mlp_jacobian(model, cache, layer)
            _, S, _ = torch.linalg.svd(J, full_matrices=False)
            features.append(S.numpy()[:50])
        
        features = np.array(features)
        task_features[task_name] = features
        all_features.append(features)
        all_labels.extend([task_name] * len(features))
    
    all_features = np.vstack(all_features)
    
    # PCA of Jacobian spectra
    print(f"\n  PCA of Jacobian Spectra:")
    pca = PCA(n_components=10)
    projected = pca.fit_transform(all_features)
    print(f"    Var explained: {pca.explained_variance_ratio_[:5]}")
    print(f"    Cumulative: {np.cumsum(pca.explained_variance_ratio_[:5])}")
    
    print(f"\n    Task centroids on PC1-PC2:")
    for task_name in task_names_5:
        mask = np.array(all_labels) == task_name
        centroid = projected[mask, :2].mean(axis=0)
        spread = projected[mask, :2].std(axis=0)
        print(f"      {task_name}: PC1={centroid[0]:.4f}+/-{spread[0]:.4f}, PC2={centroid[1]:.4f}+/-{spread[1]:.4f}")
    
    # Task separability
    lda = LinearDiscriminantAnalysis()
    scores = cross_val_score(lda, all_features, all_labels, cv=5)
    print(f"\n    LDA 5-fold accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
    
    # Jacobian field continuity
    print(f"\n    Jacobian field continuity (cosine distances within task):")
    for task_name in task_names_5:
        features = task_features[task_name]
        dists = pdist(features[:50], metric='cosine')
        q25, q50, q75 = np.percentile(dists, [25, 50, 75])
        print(f"      {task_name}: [{q25:.6f}, {q50:.6f}, {q75:.6f}], max={np.max(dists):.6f}")
    
    # Cross-task overlap
    print(f"\n    Cross-task subspace alignment:")
    task_list = list(task_features.keys())
    for i, t1 in enumerate(task_list):
        for j, t2 in enumerate(task_list):
            if i >= j: continue
            pca1 = PCA(n_components=10); pca1.fit(task_features[t1])
            pca2 = PCA(n_components=10); pca2.fit(task_features[t2])
            Q1, Q2 = pca1.components_.T, pca2.components_.T
            alignment = np.linalg.norm(Q1.T @ Q2, 'fro') / np.sqrt(10)
            print(f"      {t1} vs {t2}: {alignment:.4f}")
    
    # Operator type clustering
    print(f"\n    K-means clustering of Jacobians:")
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        pred = km.fit_predict(all_features)
        ari = adjusted_rand_score(all_labels, pred)
        nmi = normalized_mutual_info_score(all_labels, pred)
        print(f"      K={k}: ARI={ari:.4f}, NMI={nmi:.4f}")

# ============================================================
# Experiment D: Spectrum Dynamics & Recursive Rollout
# ============================================================
print(f"\n{'='*70}")
print(f"Experiment D: Spectrum Dynamics & Recursive Rollout")
print(f"{'='*70}")

task_names_4 = ["addition", "translate_fr", "antonym", "capital"]
n_samples_d = 20
all_layers = list(range(12))

# Part 1: Layer-by-layer spectrum evolution
print(f"\n  Part 1: Jacobian Spectrum Evolution Across Layers")
print(f"  (Mean of {n_samples_d} samples per task)")

for task_name in task_names_4:
    print(f"\n  {task_name}:")
    samples = gen_samples(task_name, n_samples_d)
    
    layer_spectra = {l: [] for l in all_layers}
    layer_gelu = {l: [] for l in all_layers}
    
    for text in samples:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        for layer in all_layers:
            J = compute_mlp_jacobian(model, cache, layer)
            _, S, _ = torch.linalg.svd(J, full_matrices=False)
            layer_spectra[layer].append(S.numpy())
            
            pre = cache[f'blocks.{layer}.mlp.hook_pre'][-1].detach()
            gelu_deriv = compute_gelu_derivative(pre)
            layer_gelu[layer].append((gelu_deriv > 0.1).float().mean().item())
    
    print(f"    {'L':>2} {'OpNorm':>8} {'Top1':>8} {'Top3%':>8} {'Rank95':>6} {'GELU%':>6}")
    for layer in all_layers:
        spectra = np.array(layer_spectra[layer])
        mean_sv = spectra.mean(axis=0)
        total_e = np.sum(mean_sv**2)
        top3_e = np.sum(mean_sv[:3]**2)/total_e
        cum_e = np.cumsum(mean_sv**2)/total_e
        eff_r = np.searchsorted(cum_e, 0.95)+1
        gelu_pct = np.mean(layer_gelu[layer])
        print(f"    {layer:>2} {np.sqrt(total_e):>8.2f} {mean_sv[0]:>8.2f} "
              f"{top3_e:>8.4f} {eff_r:>6d} {gelu_pct:>6.3f}")
    
    # Spectral transitions
    print(f"    Spectral transitions (adjacent layer SV cosine):")
    mean_spectra = {l: np.mean(layer_spectra[l], axis=0) for l in all_layers}
    for l in range(11):
        sv1, sv2 = mean_spectra[l], mean_spectra[l+1]
        cos = np.dot(sv1, sv2)/(np.linalg.norm(sv1)*np.linalg.norm(sv2)+1e-12)
        print(f"      L{l}->L{l+1}: {cos:.4f}")

# Part 2: Recursive Rollout (1 sample per task for speed)
print(f"\n  Part 2: Recursive Rollout — Operator Composition")
print(f"  J_total(L0->Lk) = J_Lk @ ... @ J_L0")

for task_name in task_names_4[:2]:  # addition and translate_fr only
    print(f"\n  {task_name}:")
    samples = gen_samples(task_name, 2)
    
    for sample_idx, text in enumerate(samples):
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        
        jacobians = []
        for layer in all_layers:
            J = compute_mlp_jacobian(model, cache, layer)
            jacobians.append(J.numpy())
        
        J_composed = np.eye(768)
        print(f"    Sample {sample_idx}: Composed operator spectrum")
        print(f"    {'After':>6} {'OpNorm':>10} {'Top1':>10} {'s_min':>12} {'Cond':>12} {'Rank95':>8}")
        
        for k in range(12):
            J_composed = jacobians[k] @ J_composed
            S_c = np.linalg.svd(J_composed, compute_uv=False)
            total_e = np.sum(S_c**2)
            top1 = S_c[0]
            s_nonzero = S_c[S_c > 1e-10]
            s_min = s_nonzero[-1] if len(s_nonzero) > 0 else 0
            cond = top1/(s_min+1e-12)
            cum_e = np.cumsum(S_c**2)/(total_e+1e-12)
            eff_r = np.searchsorted(cum_e, 0.95)+1 if total_e > 0 else 0
            print(f"    L{k:>4} {np.sqrt(total_e):>10.2f} {top1:>10.2f} {s_min:>12.6f} {cond:>12.1f} {eff_r:>8d}")
        
        break  # Just 1 sample per task

# Part 3: Conserved vs Unstable Modes
print(f"\n  Part 3: Conserved vs Unstable Modes")
print(f"  (Composed over all 12 layers, {n_samples_d} samples)")

for task_name in task_names_4:
    samples = gen_samples(task_name, n_samples_d)
    all_SV = []
    
    for text in samples:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        
        jacobians = []
        for layer in all_layers:
            J = compute_mlp_jacobian(model, cache, layer)
            jacobians.append(J.numpy())
        
        J_composed = np.eye(768)
        for k in range(12):
            J_composed = jacobians[k] @ J_composed
        
        S = np.linalg.svd(J_composed, compute_uv=False)
        all_SV.append(S)
    
    all_SV = np.array(all_SV)
    mean_SV = all_SV.mean(axis=0)
    
    amplified = np.sum(mean_SV > 1.1)
    conserved = np.sum((mean_SV > 0.9) & (mean_SV <= 1.1))
    attenuated = np.sum((mean_SV > 0.01) & (mean_SV <= 0.9))
    near_zero = np.sum(mean_SV <= 0.01)
    
    print(f"\n  {task_name}:")
    print(f"    Top-5 SVs: {mean_SV[:5].tolist()}")
    print(f"    Amplified (>1.1): {amplified}")
    print(f"    Conserved (0.9-1.1): {conserved}")
    print(f"    Attenuated (<0.9, >0.01): {attenuated}")
    print(f"    Near zero (<0.01): {near_zero}")
    print(f"    Dynamic range: {mean_SV[0]:.2e} to {mean_SV[mean_SV>0.01][-1]:.2e}")

print("\nPhase 81 Exp C+D (fast) completed!")
