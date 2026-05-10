"""Phase 81 Exp B (fast): Local Jacobian Field — reduced sample size"""
import torch
import numpy as np
from transformer_lens import HookedTransformer
from sklearn.decomposition import PCA

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
    return samples

model = get_model()
task_names = ["addition", "translate_fr", "antonym", "capital"]
n_samples = 50
layers = [3, 6, 9]

all_jacobian_data = {}

for layer in layers:
    print(f"\n{'='*60}")
    print(f"  Layer {layer}")
    print(f"{'='*60}")
    layer_data = {}
    
    for task_name in task_names:
        samples = gen_samples(task_name, n_samples)
        singular_values_list = []
        gelu_active_ratios = []
        top3_energies = []
        effective_ranks = []
        operator_norms = []
        
        print(f"  {task_name}: computing {n_samples} Jacobians...")
        for text in samples:
            tokens = model.to_tokens(text)
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            J = compute_mlp_jacobian(model, cache, layer)
            U, S, Vt = torch.linalg.svd(J, full_matrices=False)
            S_np = S.numpy()
            
            total_energy = np.sum(S_np**2)
            top3_energy = np.sum(S_np[:3]**2)/total_energy
            cum_energy = np.cumsum(S_np**2)/total_energy
            eff_rank = np.searchsorted(cum_energy, 0.95)+1
            op_norm = np.sqrt(total_energy)
            
            pre = cache[f'blocks.{layer}.mlp.hook_pre'][-1].detach()
            gelu_deriv = compute_gelu_derivative(pre)
            active_ratio = (gelu_deriv > 0.1).float().mean().item()
            
            singular_values_list.append(S_np)
            top3_energies.append(top3_energy)
            effective_ranks.append(eff_rank)
            operator_norms.append(op_norm)
            gelu_active_ratios.append(active_ratio)
        
        sv_matrix = np.array(singular_values_list)
        mean_sv = sv_matrix.mean(axis=0)
        std_sv = sv_matrix.std(axis=0)
        cv_sv = std_sv / (mean_sv + 1e-12)
        
        layer_data[task_name] = {
            'sv_matrix': sv_matrix, 'mean_sv': mean_sv, 'std_sv': std_sv,
            'cv_sv': cv_sv, 'top3_energies': top3_energies,
            'effective_ranks': effective_ranks, 'operator_norms': operator_norms,
            'gelu_active_ratios': gelu_active_ratios,
        }
        
        print(f"    OpNorm: {np.mean(operator_norms):.4f} +/- {np.std(operator_norms):.4f}")
        print(f"    Top3 energy: {np.mean(top3_energies):.4f} +/- {np.std(top3_energies):.4f}")
        print(f"    Eff rank(95%): {np.mean(effective_ranks):.1f} +/- {np.std(effective_ranks):.1f}")
        print(f"    GELU active: {np.mean(gelu_active_ratios):.4f}")
        print(f"    CV(top-10 SV avg): {cv_sv[:10].mean():.4f}")
        print(f"    Top-5 SVs: {mean_sv[:5].tolist()}")
        print(f"    Top-5 SV std: {std_sv[:5].tolist()}")
    
    all_jacobian_data[layer] = layer_data
    
    # Cross-task comparison
    print(f"\n  Cross-task SV cosine similarity:")
    task_list = list(layer_data.keys())
    for i, t1 in enumerate(task_list):
        for j, t2 in enumerate(task_list):
            if i >= j: continue
            sv1 = layer_data[t1]['mean_sv']
            sv2 = layer_data[t2]['mean_sv']
            cos = np.dot(sv1, sv2)/(np.linalg.norm(sv1)*np.linalg.norm(sv2)+1e-12)
            print(f"    {t1} vs {t2}: {cos:.4f}")

# Jacobian Variation Summary
print(f"\n{'='*60}")
print(f"  JACOBIAN VARIATION ANALYSIS (KEY RESULT)")
print(f"{'='*60}")

for layer in layers:
    print(f"\n  Layer {layer}:")
    for task_name in task_names:
        data = all_jacobian_data[layer][task_name]
        cv = data['cv_sv']
        cv_top5 = cv[:5].mean()
        cv_top20 = cv[:20].mean()
        cv_mid = cv[50:100].mean()
        
        # Within-task pairwise SV similarity
        sv_matrix = data['sv_matrix']
        n = min(30, sv_matrix.shape[0])
        from itertools import combinations
        pair_sims = []
        for i, j in combinations(range(n), 2):
            sim = np.dot(sv_matrix[i], sv_matrix[j]) / (
                np.linalg.norm(sv_matrix[i])*np.linalg.norm(sv_matrix[j])+1e-12)
            pair_sims.append(sim)
        mean_sim = np.mean(pair_sims)
        
        print(f"    {task_name}: CV(top5)={cv_top5:.4f}, CV(top20)={cv_top20:.4f}, "
              f"CV(mid50-100)={cv_mid:.4f}, within_sim={mean_sim:.4f}")

# KEY: Compare Jacobian vs fitted A for Phase 80 reproduction
print(f"\n{'='*60}")
print(f"  JACOBIAN vs FITTED A: Which is the REAL operator?")
print(f"{'='*60}")

from sklearn.linear_model import LinearRegression

for layer in [6]:
    print(f"\n  Layer {layer}:")
    for task_name in task_names:
        samples = gen_samples(task_name, 30)  # Same n=30 as Phase 80
        
        h_ins, delta_mlps = [], []
        jac_svs = []
        
        for text in samples:
            tokens = model.to_tokens(text)
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            h_in = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach().cpu().numpy()
            h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach().cpu().numpy()
            h_out = cache[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu().numpy()
            delta_mlp = h_out - h_mid
            
            h_ins.append(h_in)
            delta_mlps.append(delta_mlp)
            
            J = compute_mlp_jacobian(model, cache, layer)
            _, S, _ = torch.linalg.svd(J, full_matrices=False)
            jac_svs.append(S.numpy())
        
        # Fitted A (Phase 80 method)
        reg = LinearRegression()
        reg.fit(np.array(h_ins), np.array(delta_mlps))
        A = reg.coef_
        U_A, S_A, Vt_A = np.linalg.svd(A, full_matrices=False)
        
        # Mean Jacobian SVs
        jac_svs = np.array(jac_svs)
        mean_jac_sv = jac_svs.mean(axis=0)
        
        # Compare spectra
        print(f"    {task_name}:")
        print(f"      Fitted A top-5 SVs: {S_A[:5].tolist()}")
        print(f"      Mean J top-5 SVs:    {mean_jac_sv[:5].tolist()}")
        print(f"      Ratio (A/J) top-5:    {(S_A[:5]/(mean_jac_sv[:5]+1e-8)).tolist()}")
        
        # Cosine similarity of SV spectra
        cos = np.dot(S_A[:100], mean_jac_sv[:100])/(
            np.linalg.norm(S_A[:100])*np.linalg.norm(mean_jac_sv[:100])+1e-12)
        print(f"      SV spectrum cosine: {cos:.4f}")

print("\nPhase 81 Exp B (fast) completed!")
