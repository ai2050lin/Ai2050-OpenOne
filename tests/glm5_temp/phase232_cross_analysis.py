"""Phase 232 Cross-Model Analysis"""
import json
import numpy as np

def load_results(model):
    path = f'tests/glm5_temp/phase232_{model}_results.json'
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze():
    models = ['qwen3', 'glm4']
    all_data = {}
    for m in models:
        try:
            all_data[m] = load_results(m)
            print(f"=== {m.upper()} ===")
        except:
            print(f"=== {m.upper()}: NO DATA ===")
            continue
        
        d = all_data[m]
        
        # ExpA: Layer localization
        print("\n--- ExpA: Negation Layer Localization ---")
        expA = d.get('expA', {})
        lr = expA.get('layer_results', {})
        if lr:
            # Find peak KL layer
            peak_layer = max(lr.items(), key=lambda x: x[1].get('mean_kl', 0))
            onset = expA.get('onset_layer')
            print(f"  Onset layer: {onset}")
            print(f"  Peak KL layer: L{peak_layer[0]} (KL={peak_layer[1]['mean_kl']:.4f})")
            # Show trend
            for k in sorted(lr.keys(), key=int):
                v = lr[k]
                if v['mean_kl'] > 0.01:
                    print(f"    L{k}: KL={v['mean_kl']:.4f}, flip={v['mean_flip_ratio']:.4f}")
        else:
            print("  No ExpA data")
        
        # ExpB: Logit-space gate analysis
        print("\n--- ExpB: Logit-Space Gate Analysis ---")
        expB = d.get('expB', {})
        if 'error' not in expB:
            print(f"  Cross-context cosine: {expB.get('mean_cross_cosine', 'N/A')}")
            print(f"  Verdict: {expB.get('verdict', 'N/A')}")
            print(f"  Sparsity: {expB.get('sparsity', 'N/A')}")
            print(f"  Suppress ratio: {expB.get('suppress_ratio', 'N/A')}")
            print(f"  PCA variance: {expB.get('pca_var_explained', 'N/A')}")
            print(f"  Mean KL: {expB.get('mean_kl', 'N/A')}")
        else:
            print(f"  ERROR: {expB['error'][:200]}")
        
        # ExpC: Activation patching
        print("\n--- ExpC: Activation Patching ---")
        expC = d.get('expC', {})
        if 'error' not in expC:
            layer_effects = expC.get('layer_effects', {})
            if layer_effects:
                # Find best patching layer
                best_layer = max(layer_effects.items(), key=lambda x: x[1].get('mean_kl', 0))
                print(f"  Best patching layer: L{best_layer[0]} (KL={best_layer[1]['mean_kl']:.4f})")
                # Show trend
                for k in sorted(layer_effects.keys(), key=int):
                    v = layer_effects[k]
                    if v.get('mean_kl', 0) > 0.01:
                        print(f"    L{k}: KL={v['mean_kl']:.4f}, flip_change={v.get('mean_flip_change', 'N/A')}")
            print(f"  Affirm→Neg direction: {expC.get('affirm_to_neg_verdict', 'N/A')}")
        else:
            print(f"  ERROR: {expC['error'][:200]}")
        
        # ExpD: Component ablation
        print("\n--- ExpD: Component Ablation ---")
        expD = d.get('expD', {})
        if 'error' not in expD:
            ci = expD.get('component_importance', expD.get('head_importance', {}))
            baseline = expD.get('baseline_kl', 'N/A')
            print(f"  Baseline KL: {baseline}")
            if ci:
                sorted_ci = sorted(ci.items(), key=lambda x: -x[1].get('mean_reduction', 0))
                for k, v in sorted_ci[:8]:
                    print(f"    {k}: reduction={v['mean_reduction']:.4f}")
        else:
            print(f"  ERROR: {expD['error'][:200]}")
        
        # ExpE: Cross-negation-word comparison
        print("\n--- ExpE: Cross-Negation-Word Comparison ---")
        expE = d.get('expE', {})
        if 'error' not in expE:
            print(f"  Mean cross cosine: {expE.get('mean_cross_cosine', 'N/A')}")
            print(f"  Verdict: {expE.get('verdict', 'N/A')}")
            print(f"  Negation words tested: {expE.get('n_negation_words', 'N/A')}")
        else:
            print(f"  ERROR: {expE['error'][:200]}")
        
        print("\n" + "="*50)

    # Cross-model summary
    print("\n\n========== CROSS-MODEL SUMMARY ==========")
    print("\nExpB: Gate Analysis")
    for m in models:
        if m in all_data:
            expB = all_data[m].get('expB', {})
            if 'error' not in expB:
                print(f"  {m}: cross_cos={expB.get('mean_cross_cosine', 'N/A'):.4f}, "
                      f"sparsity={expB.get('sparsity', 'N/A'):.4f}, "
                      f"suppress={expB.get('suppress_ratio', 'N/A'):.3f}, "
                      f"verdict={expB.get('verdict', 'N/A')}")
    
    print("\nExpC: Activation Patching")
    for m in models:
        if m in all_data:
            expC = all_data[m].get('expC', {})
            if 'error' not in expC:
                le = expC.get('layer_effects', {})
                if le:
                    best = max(le.items(), key=lambda x: x[1].get('mean_kl', 0))
                    print(f"  {m}: best_layer=L{best[0]}, best_KL={best[1]['mean_kl']:.4f}")
    
    print("\nExpD: Component Ablation (Self-Attn vs MLP)")
    for m in models:
        if m in all_data:
            expD = all_data[m].get('expD', {})
            if 'error' not in expD:
                ci = expD.get('component_importance', expD.get('head_importance', {}))
                attn_reds = [v['mean_reduction'] for k, v in ci.items() if 'self_attn' in k]
                mlp_reds = [v['mean_reduction'] for k, v in ci.items() if 'mlp' in k]
                if attn_reds and mlp_reds:
                    print(f"  {m}: self_attn={np.mean(attn_reds):.4f}, mlp={np.mean(mlp_reds):.4f}, "
                          f"attn/mlp={np.mean(attn_reds)/(np.mean(mlp_reds)+1e-10):.2f}")
    
    print("\nExpE: Cross-Negation-Word Comparison")
    for m in models:
        if m in all_data:
            expE = all_data[m].get('expE', {})
            if 'error' not in expE:
                print(f"  {m}: cross_cos={expE.get('mean_cross_cosine', 'N/A'):.4f}, "
                      f"verdict={expE.get('verdict', 'N/A')}")

if __name__ == '__main__':
    analyze()
