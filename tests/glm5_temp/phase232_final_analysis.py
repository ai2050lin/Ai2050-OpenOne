"""Phase 232 Complete Cross-Model Analysis"""
import json
import numpy as np

def load_results(model):
    path = f'tests/glm5_temp/phase232_{model}_results.json'
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    models_info = {
        'qwen3': {'n_layers': 36, 'd_model': 2560, 'name': 'Qwen3-8B'},
        'glm4': {'n_layers': 40, 'd_model': 4096, 'name': 'GLM4-9B'},
        'deepseek7b': {'n_layers': 28, 'd_model': 3584, 'name': 'DS7B'},
    }
    
    all_data = {}
    for m in models_info:
        try:
            all_data[m] = load_results(m)
        except:
            print(f"WARNING: {m} data not found")
    
    print("=" * 70)
    print("PHASE 232: NEGATION CIRCUIT LOCALIZATION — CROSS-MODEL SUMMARY")
    print("=" * 70)
    
    # ===== ExpA: Layer Localization =====
    print("\n" + "=" * 70)
    print("ExpA: Negation Layer Localization")
    print("=" * 70)
    print("NOTE: ExpA is broken for all models (missing LayerNorm in intermediate")
    print("layer projection). Using ExpC data instead for layer localization.")
    
    # ===== ExpB: Logit-Space Gate Analysis =====
    print("\n" + "=" * 70)
    print("ExpB: Logit-Space Gate Analysis (CORE FINDING)")
    print("=" * 70)
    
    print(f"\n{'Model':<12} {'CrossCos':<10} {'Sparsity':<10} {'SuppressR':<10} {'KL':<8} {'PCA1':<8} {'Verdict'}")
    print("-" * 75)
    
    for m, info in models_info.items():
        if m not in all_data:
            continue
        b = all_data[m].get('expB', {})
        if 'error' in b:
            print(f"{info['name']:<12} ERROR: {b['error'][:50]}")
            continue
        
        cos = b.get('mean_cross_cosine', 0)
        sp = b.get('sparsity', 0)
        sr = b.get('suppress_ratio', 0)
        kl = b.get('mean_kl', 0)
        pca1 = b.get('pca_var_explained', [0])[0]
        verdict = b.get('verdict', 'N/A').split(' - ')[0]
        
        print(f"{info['name']:<12} {cos:<10.4f} {sp:<10.4f} {sr:<10.3f} {kl:<8.2f} {pca1:<8.3f} {verdict}")
    
    print("\n>>> KEY FINDING: 'not' is CONDITIONAL in all models (cross_cos < 0.20)")
    print(">>> This means 'not' is NOT a fixed gate — its effect depends on context")
    
    # ===== ExpC: Activation Patching =====
    print("\n" + "=" * 70)
    print("ExpC: Activation Patching (CAUSAL EVIDENCE)")
    print("=" * 70)
    
    for m, info in models_info.items():
        if m not in all_data:
            continue
        c = all_data[m].get('expC', {})
        best_layer = c.get('best_patch_layer')
        best_kl = c.get('best_patch_kl', 0)
        
        # Layer results
        lr = c.get('layer_results', {})
        print(f"\n{info['name']} (best=L{best_layer}, KL={best_kl:.4f}):")
        
        if lr:
            # Convert and sort
            layer_data = []
            for k, v in lr.items():
                if isinstance(v, dict):
                    layer_data.append((int(k), v.get('mean_kl', 0)))
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                    layer_data.append((int(k), np.mean(v)))
            
            layer_data.sort()
            # Show top and trend
            top_layers = sorted(layer_data, key=lambda x: -x[1])[:5]
            print(f"  Top-5 patching layers: {[(f'L{l}', f'{k:.3f}') for l, k in top_layers]}")
    
    print("\n>>> KEY FINDING: Negation effect accumulates across layers")
    print(">>> Qwen3: Best at L35 (near final), GLM4: Best at L39 (final), DS7B: Best at L3")
    
    # ===== ExpD: Component Ablation =====
    print("\n" + "=" * 70)
    print("ExpD: Component Ablation (Self-Attn vs MLP)")
    print("=" * 70)
    
    for m, info in models_info.items():
        if m not in all_data:
            continue
        dd = all_data[m].get('expD', {})
        if 'error' in dd:
            print(f"\n{info['name']}: ERROR")
            continue
        
        ci = dd.get('component_importance', dd.get('head_importance', {}))
        baseline = dd.get('baseline_kl', 0)
        
        attn_reds = {k: v['mean_reduction'] for k, v in ci.items() if 'self_attn' in k}
        mlp_reds = {k: v['mean_reduction'] for k, v in ci.items() if 'mlp' in k}
        
        print(f"\n{info['name']} (baseline KL = {baseline:.4f}):")
        
        # Sort all by reduction
        all_comps = {**attn_reds, **mlp_reds}
        sorted_comps = sorted(all_comps.items(), key=lambda x: -x[1])
        
        for k, v in sorted_comps[:6]:
            comp_type = "Attn" if "self_attn" in k else "MLP"
            layer = k.split('_')[0]
            print(f"  {layer} {comp_type:<5}: reduction = {v:+.4f}")
        
        if attn_reds and mlp_reds:
            avg_attn = np.mean(list(attn_reds.values()))
            avg_mlp = np.mean(list(mlp_reds.values()))
            print(f"  AVERAGE: Attn={avg_attn:+.4f}, MLP={avg_mlp:+.4f}")
    
    print("\n>>> KEY FINDING:")
    print(">>> Qwen3: Self-attn slightly > MLP, both positive")
    print(">>> GLM4: Most ablations NEGATIVE (zeroing INCREASES negation effect!)")
    print(">>> DS7B: MLP > Self-attn, L16 is critical for both")
    
    # ===== ExpE: Cross-Negation Generalization =====
    print("\n" + "=" * 70)
    print("ExpE: Cross-Negation-Word Generalization")
    print("=" * 70)
    
    print(f"\n{'Model':<12} {'CrossCos':<10} {'Verdict'}")
    print("-" * 50)
    
    for m, info in models_info.items():
        if m not in all_data:
            continue
        e = all_data[m].get('expE', {})
        if 'error' in e:
            print(f"{info['name']:<12} ERROR")
            continue
        
        cos = e.get('mean_cross_cosine', 0)
        verdict = e.get('verdict', 'N/A').split(' - ')[0]
        print(f"{info['name']:<12} {cos:<10.4f} {verdict}")
    
    # Pairwise cosines
    print("\nPairwise cosine similarities:")
    for m, info in models_info.items():
        if m not in all_data:
            continue
        e = all_data[m].get('expE', {})
        pairs = e.get('pairwise_cosines', {})
        if pairs:
            print(f"\n{info['name']}:")
            for k, v in sorted(pairs.items()):
                print(f"  {k}: {v:.4f}")
    
    print("\n>>> KEY FINDING:")
    print(">>> Qwen3 & DS7B: INDEPENDENT paths (different negation words use different circuits)")
    print(">>> GLM4: PARTIALLY SHARED (some common component)")
    print(">>> 'never' and 'cannot' most similar; 'not' is the outlier")
    
    # ===== INTEGRATED ANALYSIS =====
    print("\n" + "=" * 70)
    print("INTEGRATED ANALYSIS")
    print("=" * 70)
    
    print("""
1. NEGATION IS CONDITIONAL, NOT FIXED
   - Cross-context cosine < 0.20 in all models
   - 'not' does NOT apply a fixed transformation
   - Its effect depends heavily on what it's negating
   - This refutes the "fixed gate" hypothesis

2. NEGATION EFFECT ACCUMULATES ACROSS LAYERS
   - Activation patching works best at deep layers
   - Qwen3: L35/36 (near-final), GLM4: L39/40 (final), DS7B: L3 (interesting!)
   - The negation circuit is distributed, not localized to one layer

3. COMPONENT CONTRIBUTIONS ARE MODEL-SPECIFIC
   - Qwen3: Self-attn ≈ MLP, both contribute positively
   - GLM4: NEGATIVE ablation effects — zeroing components makes negation STRONGER
     This is paradoxical and suggests GLM4 uses an inhibitory mechanism
   - DS7B: MLP dominant (especially L12-L16), L16 is critical hub

4. DIFFERENT NEGATION WORDS USE DIFFERENT CIRCUITS
   - Cross-negation cosine is low (0.25-0.62)
   - 'not', 'never', 'cannot' are NOT interchangeable at the circuit level
   - This means "negation" is not a single mechanism — it's a family of mechanisms

5. THE MOST IMPORTANT UNANSWERED QUESTION
   - Why does DS7B show L3 as best patching layer (very early)?
   - This may be related to DS7B's sliding window attention
   - Or it could indicate that DS7B implements negation very differently
""")

if __name__ == '__main__':
    main()
